use super::*;
use crate::dev::Devices;
use crate::disk::Disk;
use crate::platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE};
use crate::testkit::{CODE_BASE, FakeKernel, Harness, fake_bzimage, run_machine};
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

/// Generous: only a hung guest should ever hit it.
const GENEROUS_LIMIT: Duration = Duration::from_secs(20);
/// Short watchdog for guests that never exit on their own (the watchdog stops them).
const WATCHDOG_LIMIT: Duration = Duration::from_millis(100);
const MIB: usize = 1 << 20;

const SERIAL_PORT_SETUP: [u8; 4] = [0x66, 0xba, 0xf8, 0x03]; // mov dx, 0x3f8
/// mov al, 0xfe; out 0x64, al; hlt: the keyboard controller reset command.
const RESET_AND_HALT: [u8; 5] = [0xb0, 0xfe, 0xe6, 0x64, 0xf4];
/// The serial port must show this before a guest that never exits is stopped.
const ALIVE_MARKER: u8 = b'k';

// mov dx, 0x3f8; mov al, 'h'; out dx, al; mov al, 'i'; out dx, al; mov al, 0xfe; out 0x64, al; hlt
const HELLO_RESET: [u8; 15] = [
    0x66, 0xba, 0xf8, 0x03, 0xb0, 0x68, 0xee, 0xb0, 0x69, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
const PRINT_AND_SPIN: [u8; 13] = [
    0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe, 0x90, 0x90, 0x90, 0x90,
];
/// Offset of the `out 0x64, al` that resets, within `HELLO_RESET`.
const HELLO_RESET_OUT_OFFSET: u64 = 12;
// jmp $
const SPIN: [u8; 2] = [0xeb, 0xfe];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; cli; hlt
const PRINT_THEN_HALT_WITH_INTERRUPTS_OFF: [u8; 9] =
    [0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xfa, 0xf4];
// mov dx, 0x3f8; mov al, 'k'; out dx, al; sti; hlt: with no timer or input nothing ever wakes it,
// but it is waiting, not dead.
const PRINT_THEN_STI_HALT: [u8; 9] = [0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xfb, 0xf4];
// mov eax, 0x8000; lidt [rax]; int3: an empty IDT, so delivering #BP escalates to a triple fault.
const NULL_IDT_THEN_INT3: [u8; 9] = [0xb8, 0x00, 0x80, 0x00, 0x00, 0x0f, 0x01, 0x18, 0xcc];

/// Whether the host has generated-code support, as the machine itself asks.
fn host_has_jit() -> bool {
    crate::jit::probe()
}

fn auto_and_off() -> [JitMode; 2] {
    [JitMode::Off, JitMode::Auto]
}

fn run_guest(code: &[u8], jit: JitMode) -> crate::testkit::MachineRun {
    run_machine(fake_bzimage(code), jit, GENEROUS_LIMIT, |config| config)
}

fn run_spinning(code: &[u8], jit: JitMode) -> crate::testkit::MachineRun {
    run_machine(fake_bzimage(code), jit, WATCHDOG_LIMIT, |config| config)
}

fn new_machine(code: &[u8]) -> EmulatedMachine {
    EmulatedMachine::new(
        MachineConfig::new()
            .reboot(false)
            .kernel(fake_bzimage(code))
            .memory_mib(MIN_MEMORY_MIB),
    )
    .expect("valid configuration")
}

fn expect_config_error(result: Result<EmulatedMachine, Error>) -> String {
    match result {
        Err(Error::BadConfig(message)) => message,
        Err(other) => panic!("expected BadConfig, got {other:?}"),
        Ok(_) => panic!("expected BadConfig, got a machine"),
    }
}

/// Collects the strings the `log` hook receives.
fn log_collector() -> (
    Arc<Mutex<Vec<String>>>,
    impl Fn(&str) + Send + Sync + 'static,
) {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sink = lines.clone();
    (lines, move |line: &str| {
        sink.lock().unwrap().push(line.to_string())
    })
}

mod rtc {
    use super::*;

    /// 2023-11-14 22:13:20 UTC.
    const FIXED_UNIX_TIME: u64 = 1_700_000_000;
    const CMOS_SECONDS: u8 = 0x00;
    const CMOS_MINUTES: u8 = 0x02;
    const CMOS_HOURS: u8 = 0x04;
    const CMOS_DAY_OF_MONTH: u8 = 0x07;
    const CMOS_MONTH: u8 = 0x08;
    const CMOS_YEAR: u8 = 0x09;
    const CMOS_CENTURY: u8 = 0x32;
    const ALL_RTC_REGISTERS: [u8; 6] = [
        CMOS_SECONDS,
        CMOS_MINUTES,
        CMOS_HOURS,
        CMOS_DAY_OF_MONTH,
        CMOS_MONTH,
        CMOS_YEAR,
    ];
    /// BCD: 22:13:20 on 14 November 2023.
    const FIXED_TIME_AS_BCD: [u8; 6] = [0x20, 0x13, 0x22, 0x14, 0x11, 0x23];

    /// Guest that writes the given CMOS registers to the serial port, then resets.
    fn cmos_reader_kernel(registers: &[u8]) -> Vec<u8> {
        let mut code = SERIAL_PORT_SETUP.to_vec();
        for &register in registers {
            // mov al, register; out 0x70, al; in al, 0x71; out dx, al
            code.extend_from_slice(&[0xb0, register, 0xe6, 0x70, 0xe4, 0x71, 0xee]);
        }
        code.extend_from_slice(&RESET_AND_HALT);
        fake_bzimage(&code)
    }

    fn verify_run_reads_fixed_rtc(jit: JitMode) {
        let run = run_machine(
            cmos_reader_kernel(&ALL_RTC_REGISTERS),
            jit,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true).unix_time(FIXED_UNIX_TIME),
        );
        assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
        assert_eq!(run.console_output, FIXED_TIME_AS_BCD, "{jit:?}");
    }

    #[test]
    fn verify_with_fixed_time_and_auto_jit_sees_the_same_rtc_on_both_cpus() {
        verify_run_reads_fixed_rtc(JitMode::Auto);
    }

    #[test]
    fn verify_with_jit_off_runs_the_interpreter_alone() {
        verify_run_reads_fixed_rtc(JitMode::Off);
    }

    #[test]
    fn verify_without_a_fixed_time_also_agrees_across_the_two_cpus() {
        let run = run_machine(
            cmos_reader_kernel(&ALL_RTC_REGISTERS),
            JitMode::Auto,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert_eq!(run.console_output.len(), ALL_RTC_REGISTERS.len());
    }

    #[test]
    fn fixed_time_reaches_the_rtc_without_verify() {
        for jit in auto_and_off() {
            let run = run_machine(
                cmos_reader_kernel(&ALL_RTC_REGISTERS),
                jit,
                GENEROUS_LIMIT,
                |config| config.unix_time(FIXED_UNIX_TIME),
            );
            assert_eq!(run.console_output, FIXED_TIME_AS_BCD, "{jit:?}");
        }
    }

    #[test]
    fn unix_time_zero_is_the_epoch_not_the_wall_clock() {
        let run = run_machine(
            cmos_reader_kernel(&[CMOS_CENTURY, CMOS_YEAR]),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.unix_time(0),
        );
        assert_eq!(run.console_output, [0x19, 0x70]);
    }

    #[test]
    fn without_a_fixed_time_the_rtc_follows_the_host_clock() {
        let run = run_machine(
            cmos_reader_kernel(&[CMOS_CENTURY]),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config,
        );
        // Any real clock is past 2000 and before 2100.
        assert!(matches!(run.console_output[..], [0x20] | [0x21]));
    }

    fn twin_for(boot_unix: Option<u64>) -> Cpu {
        let mut ram = vec![0u8; 64 * MIB];
        let boot = boot::load(&mut ram, &fake_bzimage(&[0xf4]), None, "").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        match boot_unix {
            Some(boot_unix) => interpreter_twin(
                ram,
                boot,
                stop,
                boot_unix,
                DevConfig::default(),
                ConsoleHandle::default(),
            ),
            // What `Vcpu::run` built before the fix.
            None => Cpu::new(
                ram,
                boot,
                ConsoleHandle::default(),
                stop,
                None,
                DevConfig::default(),
            ),
        }
    }

    fn read_cmos(cpu: &mut Cpu, register: u8) -> u8 {
        cpu.dev.io_out(0x70, 1, register as u32, 0);
        cpu.dev.io_in(0x71, 1, 0) as u8
    }

    #[test]
    fn interpreter_twin_takes_the_given_boot_time() {
        let mut twin = twin_for(Some(FIXED_UNIX_TIME));
        assert_eq!(read_cmos(&mut twin, CMOS_SECONDS), 0x20);
        assert_eq!(read_cmos(&mut twin, CMOS_YEAR), 0x23);
        let mut epoch = twin_for(Some(0));
        assert_eq!(read_cmos(&mut epoch, CMOS_CENTURY), 0x19);
    }

    #[test]
    fn a_twin_without_the_boot_time_reads_the_wall_clock_and_would_diverge() {
        let mut fixed = twin_for(Some(0));
        let mut unfixed = twin_for(None);
        assert_eq!(read_cmos(&mut fixed, CMOS_CENTURY), 0x19);
        assert_ne!(
            read_cmos(&mut unfixed, CMOS_CENTURY),
            read_cmos(&mut fixed, CMOS_CENTURY),
            "a twin built like the main CPU but never given its boot time reads another RTC"
        );
    }

    #[test]
    fn interpreter_twin_is_interpreter_only_virtual_time_sharing_the_stop_flag() {
        let mut ram = vec![0u8; 64 * MIB];
        let boot = boot::load(&mut ram, &fake_bzimage(&[0xf4]), None, "").unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let twin = interpreter_twin(
            ram,
            boot,
            stop.clone(),
            1,
            DevConfig::default(),
            ConsoleHandle::default(),
        );
        assert!(twin.jit.is_none());
        assert!(twin.wall_clock.is_none());
        assert!(twin.poll.is_none());
        assert!(twin.log.is_none());
        stop.store(true, Ordering::SeqCst);
        assert!(twin.stop.load(Ordering::SeqCst));
    }
}

mod twin_devices {
    use super::*;

    const MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
    const DISK_BYTES: usize = 2 * MIB;
    const PCI_CONFIG_ADDRESS: u16 = 0xcf8;
    const PCI_CONFIG_DATA: u16 = 0xcfc;
    const PCI_ENABLE: u32 = 1 << 31;
    const SLOT_SHIFT: u32 = 11;
    const VIRTIO_VENDOR: u32 = 0x1af4;
    const VIRTIO_DEVICE_ID_BASE: u32 = 0x1040;
    const DEVICE_TYPE_NET: u32 = 1;
    const DEVICE_TYPE_BLOCK: u32 = 2;
    const DEVICE_TYPE_CONSOLE: u32 = 3;
    const DEVICE_TYPE_GPU: u32 = 16;
    const DEVICE_TYPE_INPUT: u32 = 18;
    /// Revision dword: class, subclass, programming interface, revision from the top down.
    const CLASS_REGISTER: u32 = 0x08;
    const CLASS_SHIFT: u32 = 8;
    const VGA_CLASS: u32 = 0x03_00_00;
    const INPUT_OTHER_CLASS: u32 = 0x09_80_00;

    fn display_config(
        boot_framebuffer: Option<crate::dev::bootfb::FramebufferRegion>,
    ) -> DevConfig {
        let display = crate::display::DisplayHandle::with_size(640, 480);
        DevConfig {
            input: Some(crate::display::InputHandle::for_display(display.clone())),
            display: Some(crate::dev::DisplayDevices {
                display,
                boot_framebuffer,
                gpu: None,
            }),
            ..DevConfig::default()
        }
    }

    fn config_dword(devices: &mut crate::dev::Devices, slot: u32, register: u32) -> u32 {
        devices.io_out(
            PCI_CONFIG_ADDRESS,
            4,
            PCI_ENABLE | (slot << SLOT_SHIFT) | register,
            0,
        );
        devices.io_in(PCI_CONFIG_DATA, 4, 0)
    }

    fn id_of(devices: &mut crate::dev::Devices, slot: u32) -> u32 {
        config_dword(devices, slot, 0)
    }

    fn mirrored(main: &mut DevConfig) -> DevConfig {
        mirrored_devices(main, &ConsoleHandle::default()).0
    }

    fn full_config(read_only: bool) -> DevConfig {
        DevConfig {
            disks: vec![Disk::from_backend(ReadOnlyFlagDisk(DISK_BYTES, read_only))],
            nets: vec![(MAC, NetHandle::default())],
            virtio_consoles: vec![ConsoleHandle::default()],
            ..Default::default()
        }
    }

    struct ReadOnlyFlagDisk(usize, bool);

    impl crate::disk::BlockBackend for ReadOnlyFlagDisk {
        fn len(&self) -> u64 {
            self.0 as u64
        }
        fn read_at(&mut self, _: u64, buf: &mut [u8]) -> std::io::Result<()> {
            buf.fill(0x5a);
            Ok(())
        }
        fn write_at(&mut self, _: u64, _: &[u8]) -> std::io::Result<()> {
            Ok(())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
        fn read_only(&self) -> bool {
            self.1
        }
    }

    fn device_id(device_type: u32) -> u32 {
        VIRTIO_VENDOR | ((VIRTIO_DEVICE_ID_BASE + device_type) << 16)
    }

    #[test]
    fn mirror_has_the_same_virtio_functions_in_the_same_slots() {
        let mut main = full_config(false);
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&mut main));
        let mut original = Devices::new(ConsoleHandle::default(), main);
        for slot in 0..5 {
            assert_eq!(
                id_of(&mut mirror, slot),
                id_of(&mut original, slot),
                "slot {slot}"
            );
        }
        // Console, disk, net: the order `Devices::new` assigns slots.
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_CONSOLE));
        assert_eq!(id_of(&mut mirror, 2), device_id(DEVICE_TYPE_BLOCK));
        assert_eq!(id_of(&mut mirror, 3), device_id(DEVICE_TYPE_NET));
    }

    #[test]
    fn the_display_is_a_vga_gpu_then_a_keyboard_and_a_tablet() {
        let mut devices = Devices::new(ConsoleHandle::default(), display_config(None));
        assert_eq!(id_of(&mut devices, 1), device_id(DEVICE_TYPE_GPU));
        assert_eq!(id_of(&mut devices, 2), device_id(DEVICE_TYPE_INPUT));
        assert_eq!(id_of(&mut devices, 3), device_id(DEVICE_TYPE_INPUT));
        let class = |devices: &mut Devices, slot| {
            config_dword(devices, slot, CLASS_REGISTER) >> CLASS_SHIFT
        };
        assert_eq!(class(&mut devices, 1), VGA_CLASS);
        assert_eq!(class(&mut devices, 2), INPUT_OTHER_CLASS);
    }

    #[test]
    fn the_display_comes_after_the_nics() {
        let mut config = display_config(None);
        config.nets = vec![(MAC, NetHandle::default())];
        config.virtio_consoles = vec![ConsoleHandle::default()];
        let mut devices = Devices::new(ConsoleHandle::default(), config);
        assert_eq!(id_of(&mut devices, 2), device_id(DEVICE_TYPE_NET));
        assert_eq!(id_of(&mut devices, 3), device_id(DEVICE_TYPE_GPU));
    }

    #[test]
    fn the_mirror_has_the_display_functions_and_the_same_boot_framebuffer() {
        let region = crate::dev::bootfb::FramebufferRegion {
            base: 0x10_0000,
            width: 640,
            height: 480,
        };
        let mut main = display_config(Some(region));
        let twin = mirrored(&mut main);
        let twin_display = twin.display.as_ref().unwrap();
        assert_eq!(twin_display.boot_framebuffer, Some(region));
        assert_eq!(twin_display.display.size(), (640, 480));
        assert!(!Arc::ptr_eq(
            &twin_display.display.shared,
            &main.display.as_ref().unwrap().display.shared
        ));
        let mut mirror = Devices::new(ConsoleHandle::default(), twin);
        let mut original = Devices::new(ConsoleHandle::default(), main);
        for slot in 1..=4 {
            assert_eq!(
                id_of(&mut mirror, slot),
                id_of(&mut original, slot),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn mirror_of_an_empty_config_is_empty() {
        let mut mirror = Devices::new(
            ConsoleHandle::default(),
            mirrored(&mut DevConfig::default()),
        );
        assert_eq!(id_of(&mut mirror, 1), u32::MAX);
    }

    #[test]
    fn mirror_gives_each_optional_device_independently() {
        let mut net_only = DevConfig {
            nets: vec![(MAC, NetHandle::default())],
            ..DevConfig::default()
        };
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&mut net_only));
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_NET));
        assert_eq!(id_of(&mut mirror, 2), u32::MAX);
        let mut console_only = DevConfig {
            virtio_consoles: vec![ConsoleHandle::default()],
            ..DevConfig::default()
        };
        let mut mirror = Devices::new(ConsoleHandle::default(), mirrored(&mut console_only));
        assert_eq!(id_of(&mut mirror, 1), device_id(DEVICE_TYPE_CONSOLE));
    }

    #[test]
    fn mirrored_net_device_has_its_own_frame_queue() {
        let host_handle = NetHandle::default();
        let mut main = DevConfig {
            nets: vec![(MAC, host_handle.clone())],
            ..DevConfig::default()
        };
        let mirror = mirrored(&mut main);
        let (mac, mirror_handle) = mirror.nets.into_iter().next().unwrap();
        assert_eq!(mac, MAC);
        assert!(host_handle.send_frame(b"frame"));
        assert!(!mirror_handle.has_pending());
    }

    #[test]
    fn mirrored_disk_has_the_capacity_and_read_only_flag_of_the_original() {
        for read_only in [false, true] {
            let mut mirror = mirrored(&mut full_config(read_only));
            let disk = mirror.disks.remove(0);
            assert_eq!(disk.backend.len(), DISK_BYTES as u64);
            assert_eq!(disk.backend.read_only(), read_only);
        }
    }

    #[test]
    fn mirrored_disk_replays_what_the_main_disk_returned_and_discards_writes() {
        let mut main = full_config(false);
        let mut twin = mirrored(&mut main).disks.remove(0);
        let mut real = main.disks.remove(0);
        let mut buf = [0u8; 16];
        real.backend.read_at(512, &mut buf).unwrap();
        assert_eq!(buf, [0x5a; 16]);
        let mut replayed = [0xffu8; 16];
        twin.backend.read_at(512, &mut replayed).unwrap();
        assert_eq!(replayed, [0x5a; 16]);
        twin.backend.write_at(0, &[1, 2, 3]).unwrap();
        twin.backend.flush().unwrap();
        // A request the main disk never served reads zeros: the check reports a divergence.
        twin.backend.read_at(0, &mut replayed).unwrap();
        assert_eq!(replayed, [0u8; 16]);
    }

    #[test]
    fn mirroring_a_huge_disk_does_not_allocate_it() {
        let mut huge = DevConfig {
            disks: vec![Disk::from_backend(ReadOnlyFlagDisk(usize::MAX / 2, false))],
            ..DevConfig::default()
        };
        let disk = mirrored(&mut huge).disks.remove(0);
        assert_eq!(disk.backend.len(), (usize::MAX / 2) as u64);
    }

    // Guest code: read the PCI config dword of slots 1..=3 and print each as 4 bytes.
    fn pci_probe_block(slot: u32) -> Vec<u8> {
        let address = PCI_ENABLE | (slot << SLOT_SHIFT);
        // mov dx, 0xcf8; mov eax, address; out dx, eax; mov dx, 0xcfc; in eax, dx;
        // mov dx, 0x3f8; out dx, al; shr eax, 8; out dx, al; shr eax, 8; out dx, al;
        // shr eax, 8; out dx, al
        let mut block = vec![0x66, 0xba, 0xf8, 0x0c, 0xb8];
        block.extend_from_slice(&address.to_le_bytes());
        block.extend_from_slice(&[
            0xef, 0x66, 0xba, 0xfc, 0x0c, 0xed, 0x66, 0xba, 0xf8, 0x03, 0xee, 0xc1, 0xe8, 0x08,
            0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        ]);
        block
    }

    fn pci_probe_kernel() -> Vec<u8> {
        let mut code = Vec::new();
        for slot in 1..=3 {
            code.extend(pci_probe_block(slot));
        }
        code.extend_from_slice(&RESET_AND_HALT);
        fake_bzimage(&code)
    }

    fn id_bytes(device_type: u32) -> [u8; 4] {
        device_id(device_type).to_le_bytes()
    }

    fn probe(configure: impl FnOnce(MachineConfig) -> MachineConfig, jit: JitMode) -> Vec<u8> {
        let run = run_machine(pci_probe_kernel(), jit, GENEROUS_LIMIT, configure);
        assert_eq!(run.exit, ExitReason::Reset);
        run.console_output
    }

    const NOTHING: [u8; 4] = [0xff; 4];

    fn expected_slots(slots: [[u8; 4]; 3]) -> Vec<u8> {
        slots.concat()
    }

    #[test]
    fn a_machine_with_a_serial_console_and_no_other_devices_has_an_empty_pci_bus() {
        for jit in auto_and_off() {
            assert_eq!(
                probe(|config| config.console(ConsoleKind::Serial), jit),
                expected_slots([NOTHING; 3]),
                "{jit:?}"
            );
        }
    }

    #[test]
    fn the_default_console_is_a_virtio_console_function() {
        for jit in auto_and_off() {
            assert_eq!(
                probe(|config| config, jit),
                expected_slots([id_bytes(DEVICE_TYPE_CONSOLE), NOTHING, NOTHING]),
                "{jit:?}"
            );
        }
    }

    #[test]
    fn virtio_console_kind_adds_a_console_function() {
        assert_eq!(
            probe(|config| config.console(ConsoleKind::Virtio), JitMode::Off),
            expected_slots([id_bytes(DEVICE_TYPE_CONSOLE), NOTHING, NOTHING])
        );
    }

    #[test]
    fn serial_console_kind_adds_no_function() {
        assert_eq!(
            probe(|config| config.console(ConsoleKind::Serial), JitMode::Off),
            expected_slots([NOTHING; 3])
        );
    }

    #[test]
    fn a_disk_adds_a_block_function() {
        let disk = Disk::memory(vec![0; MIB]);
        assert_eq!(
            probe(
                |config| config.console(ConsoleKind::Serial).disk(disk),
                JitMode::Off
            ),
            expected_slots([id_bytes(DEVICE_TYPE_BLOCK), NOTHING, NOTHING])
        );
    }

    #[test]
    fn a_net_mac_adds_a_net_function() {
        assert_eq!(
            probe(
                |config| config.console(ConsoleKind::Serial).net(MAC),
                JitMode::Off
            ),
            expected_slots([id_bytes(DEVICE_TYPE_NET), NOTHING, NOTHING])
        );
    }

    #[test]
    fn a_display_adds_a_gpu_a_keyboard_and_a_tablet() {
        for jit in auto_and_off() {
            assert_eq!(
                probe(
                    |config| config
                        .console(ConsoleKind::Serial)
                        .rng(false)
                        .display(DisplayMode::Framebuffer),
                    jit
                ),
                expected_slots([
                    id_bytes(DEVICE_TYPE_GPU),
                    id_bytes(DEVICE_TYPE_INPUT),
                    id_bytes(DEVICE_TYPE_INPUT)
                ]),
                "{jit:?}"
            );
        }
    }

    #[test]
    fn a_gpu_renderer_keeps_the_slots_and_is_told_to_start_over() {
        let link = crate::gpu::GpuLink::new(Vec::new(), crate::gpu::GpuLimits::default());
        let renderer = link.clone();
        assert_eq!(
            probe(
                |config| config
                    .console(ConsoleKind::Serial)
                    .rng(false)
                    .display(DisplayMode::Framebuffer)
                    .gpu(renderer),
                JitMode::Off
            ),
            expected_slots([
                id_bytes(DEVICE_TYPE_GPU),
                id_bytes(DEVICE_TYPE_INPUT),
                id_bytes(DEVICE_TYPE_INPUT)
            ])
        );
        assert_eq!(link.take_commands(), [crate::gpu::GpuCommand::Reset]);
    }

    #[test]
    fn display_and_input_are_chosen_apart_and_keep_their_slot_order() {
        let gpu = id_bytes(DEVICE_TYPE_GPU);
        let input = id_bytes(DEVICE_TYPE_INPUT);
        type Choose = fn(MachineConfig) -> MachineConfig;
        let cases: [(&str, Choose, [[u8; 4]; 3]); 4] = [
            ("neither", |config| config, [NOTHING, NOTHING, NOTHING]),
            (
                "screen without input",
                |config| config.display(DisplayMode::Framebuffer).input(false),
                [gpu, NOTHING, NOTHING],
            ),
            (
                "input without a screen",
                |config| config.input(true),
                [input, input, NOTHING],
            ),
            (
                "both, the input following the screen",
                |config| config.display(DisplayMode::Framebuffer),
                [gpu, input, input],
            ),
        ];
        for (name, choose, slots) in cases {
            assert_eq!(
                probe(
                    |config| choose(config.console(ConsoleKind::Serial).rng(false)),
                    JitMode::Off
                ),
                expected_slots(slots),
                "{name}"
            );
        }
    }

    #[test]
    fn all_devices_take_slots_in_console_disk_net_order() {
        let disk = Disk::memory(vec![0; MIB]);
        let configure =
            |config: MachineConfig| config.net(MAC).disk(disk).console(ConsoleKind::Virtio);
        assert_eq!(
            probe(configure, JitMode::Off),
            expected_slots([
                id_bytes(DEVICE_TYPE_CONSOLE),
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_NET)
            ])
        );
    }

    #[test]
    fn several_disks_and_nets_take_consecutive_slots_in_config_order() {
        let configure = |config: MachineConfig| {
            config
                .console(ConsoleKind::Serial)
                .net(MAC)
                .disk(Disk::memory(vec![0; MIB]))
                .net([2, 0, 0, 0, 0, 9])
                .disk(Disk::memory(vec![0; MIB]))
        };
        // Slots 1..=3 are the first three functions: disks come before nets whatever the call order.
        assert_eq!(
            probe(configure, JitMode::Off),
            expected_slots([
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_BLOCK),
                id_bytes(DEVICE_TYPE_NET)
            ])
        );
    }

    #[test]
    fn a_second_virtio_console_takes_the_next_slot() {
        assert_eq!(
            probe(
                |config| config
                    .console(ConsoleKind::Serial)
                    .console(ConsoleKind::Virtio)
                    .console(ConsoleKind::Virtio),
                JitMode::Off
            ),
            expected_slots([
                id_bytes(DEVICE_TYPE_CONSOLE),
                id_bytes(DEVICE_TYPE_CONSOLE),
                NOTHING
            ])
        );
    }

    #[test]
    fn the_verify_twin_sees_the_same_devices_as_the_main_cpu() {
        for jit in auto_and_off() {
            let disk = Disk::memory(vec![0; MIB]);
            let run = run_machine(pci_probe_kernel(), jit, GENEROUS_LIMIT, |config| {
                config
                    .net(MAC)
                    .disk(disk)
                    .console(ConsoleKind::Virtio)
                    .verify_jit(true)
                    .unix_time(1)
            });
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(
                run.console_output,
                expected_slots([
                    id_bytes(DEVICE_TYPE_CONSOLE),
                    id_bytes(DEVICE_TYPE_BLOCK),
                    id_bytes(DEVICE_TYPE_NET)
                ]),
                "{jit:?}"
            );
        }
    }
}

mod multiple_devices {
    use super::*;

    const MAC: [u8; 6] = [2, 0, 0, 0, 0, 1];
    const SECOND_MAC: [u8; 6] = [2, 0, 0, 0, 0, 2];

    fn machine_with(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> EmulatedMachine {
        EmulatedMachine::new(configure(
            MachineConfig::new()
                .reboot(false)
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB),
        ))
        .expect("valid configuration")
    }

    /// Runs `HELLO_RESET` (writes "hi" to the UART) and returns what each console received.
    fn run_hello(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> Vec<Vec<u8>> {
        let mut machine = machine_with(configure);
        let outputs: Vec<Arc<Mutex<Vec<u8>>>> = machine
            .consoles()
            .iter()
            .map(|console| {
                let sink = Arc::new(Mutex::new(Vec::new()));
                let writer = sink.clone();
                console.on_output(move |bytes| writer.lock().unwrap().extend_from_slice(bytes));
                sink
            })
            .collect();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        outputs
            .iter()
            .map(|sink| sink.lock().unwrap().clone())
            .collect()
    }

    #[test]
    fn with_no_console_there_is_one_and_the_uart_shares_it() {
        assert_eq!(run_hello(|config| config), [b"hi".to_vec()]);
    }

    #[test]
    fn a_serial_console_next_to_a_virtio_one_owns_the_uart() {
        assert_eq!(
            run_hello(|config| config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)),
            [b"hi".to_vec(), Vec::new()]
        );
        assert_eq!(
            run_hello(|config| config
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Serial)),
            [Vec::new(), b"hi".to_vec()]
        );
    }

    #[test]
    fn virtio_consoles_have_independent_handles() {
        let machine = machine_with(|config| {
            config
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Virtio)
        });
        let consoles = machine.consoles();
        assert_eq!(consoles.len(), 2);
        consoles[1].resize(100, 40);
        assert_eq!(consoles[0].size(), (80, 25));
        assert_eq!(consoles[1].size(), (100, 40));
        assert_eq!(machine.console().size(), (80, 25));
    }

    #[test]
    fn console_handles_follow_config_order_and_the_first_is_console() {
        let machine = machine_with(|config| {
            config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)
        });
        let consoles = machine.consoles();
        consoles[0].resize(10, 10);
        assert_eq!(machine.console().size(), (10, 10));
        assert_eq!(consoles[1].size(), (80, 25));
    }

    #[test]
    fn a_second_serial_console_is_a_config_error() {
        let message = expect_config_error(EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Serial),
        ));
        assert!(message.contains("only once"), "{message}");
    }

    #[test]
    fn nets_are_distinct_handles_wired_to_their_own_device_in_config_order() {
        let machine = machine_with(|config| config.net(MAC).net(SECOND_MAC));
        let handles = machine.nets();
        assert_eq!(handles.len(), 2);
        let devices = &machine.vcpu.as_ref().unwrap().devcfg.nets;
        assert_eq!([devices[0].0, devices[1].0], [MAC, SECOND_MAC]);
        assert!(handles[1].send_frame(b"to the second"));
        assert!(!devices[0].1.has_pending());
        assert_eq!(
            devices[1].1.guest_recv_bytes().as_deref(),
            Some(&b"to the second"[..])
        );
        devices[0].1.guest_send(b"from the first".to_vec());
        assert_eq!(
            handles[0].try_recv_frame().as_deref(),
            Some(&b"from the first"[..])
        );
        assert!(handles[1].try_recv_frame().is_none());
        assert!(machine.net().is_some());
    }

    #[test]
    fn disks_keep_their_own_contents_in_config_order() {
        let mut machine = machine_with(|config| {
            config
                .disk(Disk::memory(vec![0xaa; 4096]))
                .disk(Disk::memory(vec![0xbb; 4096]))
        });
        let disks = &mut machine.vcpu.as_mut().unwrap().devcfg.disks;
        let first_bytes: Vec<u8> = disks
            .iter_mut()
            .map(|disk| {
                let mut byte = [0u8; 1];
                disk.backend.read_at(0, &mut byte).unwrap();
                byte[0]
            })
            .collect();
        assert_eq!(first_bytes, [0xaa, 0xbb]);
    }

    #[test]
    fn a_reboot_hands_every_disk_nic_and_console_to_the_new_devices() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .disk(Disk::memory(vec![0; 4096]))
                .disk(Disk::memory(vec![0; 4096]))
                .net(MAC)
                .net(SECOND_MAC)
                .console(ConsoleKind::Virtio)
                .console(ConsoleKind::Virtio),
        )
        .unwrap();
        let image = machine.vcpu.as_ref().unwrap().reboot.as_ref().unwrap();
        assert_eq!(image.nets.len(), 2);
        assert_eq!(image.virtio_consoles.len(), 2);
        let held = machine.nets();
        assert!(held[1].send_frame(b"frame"));
        assert!(image.nets[1].1.has_pending());
        assert!(!image.nets[0].1.has_pending());
    }

    #[test]
    fn a_machine_that_reboots_keeps_both_disks_in_order() {
        const BOOT_SECTOR_OFFSET: u64 = 512;
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .jit(JitMode::Off)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .disk(Disk::memory(vec![0x11; 4096]))
                .disk(Disk::memory(vec![0x22; 4096])),
        )
        .unwrap();
        let mut vcpu = machine.vcpu.take().unwrap();
        // One power cycle, then the disks are recovered from the finished run.
        let (reason, finished) = vcpu.run_once();
        assert_eq!(reason, ExitReason::Reset);
        let mut disks = finished.disks;
        assert_eq!(disks.len(), 2);
        let mut byte = [0u8; 1];
        disks[0]
            .backend
            .read_at(BOOT_SECTOR_OFFSET, &mut byte)
            .unwrap();
        assert_eq!(byte[0], 0x11);
        disks[1]
            .backend
            .read_at(BOOT_SECTOR_OFFSET, &mut byte)
            .unwrap();
        assert_eq!(byte[0], 0x22);
    }

    #[test]
    fn too_many_virtio_devices_for_the_pci_bus_is_a_config_error() {
        let mut config = MachineConfig::new()
            .kernel(fake_bzimage(&[0xf4]))
            .console(ConsoleKind::Virtio);
        for _ in 0..MAX_VIRTIO_FUNCTIONS {
            config = config.net(MAC);
        }
        let message = expect_config_error(EmulatedMachine::new(config));
        assert!(message.contains("31"), "{message}");
        let mut config = MachineConfig::new()
            .kernel(fake_bzimage(&[0xf4]))
            .console(ConsoleKind::Serial)
            .rng(false);
        for _ in 0..MAX_VIRTIO_FUNCTIONS {
            config = config.net(MAC);
        }
        assert!(EmulatedMachine::new(config).is_ok());
    }

    #[test]
    fn the_entropy_device_takes_a_pci_slot_like_any_other() {
        let full = |rng: bool| {
            let mut config = MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .console(ConsoleKind::Serial)
                .rng(rng)
                .entropy(|bytes| bytes.fill(1));
            for _ in 0..MAX_VIRTIO_FUNCTIONS {
                config = config.net(MAC);
            }
            EmulatedMachine::new(config)
        };
        assert!(full(false).is_ok());
        assert!(expect_config_error(full(true)).contains("entropy"));
    }

    #[test]
    fn the_entropy_device_is_on_by_default() {
        assert!(MachineConfig::new().rng);
    }

    #[test]
    fn the_display_counts_against_the_pci_bus() {
        let config = |nets: usize| {
            let mut config = MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .console(ConsoleKind::Serial)
                .rng(false)
                .display(DisplayMode::Framebuffer);
            for _ in 0..nets {
                config = config.net(MAC);
            }
            EmulatedMachine::new(config)
        };
        // The input devices come with the screen by default.
        let functions = DISPLAY_FUNCTIONS + INPUT_FUNCTIONS;
        assert!(config(MAX_VIRTIO_FUNCTIONS - functions).is_ok());
        let message = expect_config_error(config(MAX_VIRTIO_FUNCTIONS - functions + 1));
        assert!(message.contains("display"), "{message}");
    }

    #[test]
    fn input_counts_against_the_pci_bus_on_its_own() {
        let config = |nets: usize| {
            let mut config = MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .console(ConsoleKind::Serial)
                .rng(false)
                .input(true);
            for _ in 0..nets {
                config = config.net(MAC);
            }
            EmulatedMachine::new(config)
        };
        assert!(config(MAX_VIRTIO_FUNCTIONS - INPUT_FUNCTIONS).is_ok());
        let message = expect_config_error(config(MAX_VIRTIO_FUNCTIONS - INPUT_FUNCTIONS + 1));
        assert!(message.contains("input"), "{message}");
    }

    #[test]
    fn input_without_a_display_has_a_handle_whose_pointer_area_is_the_display_size() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .input(true)
                .display_size(640, 480),
        )
        .unwrap();
        assert!(machine.display().is_none());
        let input = machine.input().unwrap();
        assert_eq!(input.pointer_area(), (640, 480));
        const KEY_A: u16 = 30;
        assert!(input.key(KEY_A, true));
    }

    #[test]
    fn a_display_without_input_has_no_input_handle() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .display(DisplayMode::Framebuffer)
                .input(false),
        )
        .unwrap();
        assert!(machine.display().is_some() && machine.input().is_none());
    }

    #[test]
    fn a_machine_has_display_and_input_handles_only_with_a_display() {
        let without = new_machine(&[0xf4]);
        assert!(without.display().is_none() && without.input().is_none());
        let with = EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .display(DisplayMode::Framebuffer)
                .display_size(800, 600),
        )
        .unwrap();
        assert_eq!(with.display().unwrap().size(), (800, 600));
        assert!(with.input().unwrap().pointer_move(799, 599));
    }

    #[test]
    fn display_sizes_outside_the_supported_range_are_refused() {
        for (width, height) in [(319, 600), (800, 199), (2561, 600), (800, 1601)] {
            let message = expect_config_error(EmulatedMachine::new(
                MachineConfig::new()
                    .kernel(fake_bzimage(&[0xf4]))
                    .display(DisplayMode::Framebuffer)
                    .display_size(width, height),
            ));
            assert!(message.contains("display_size"), "{message}");
            // It is the input's pointer area without a display.
            let message = expect_config_error(EmulatedMachine::new(
                MachineConfig::new()
                    .kernel(fake_bzimage(&[0xf4]))
                    .input(true)
                    .display_size(width, height),
            ));
            assert!(message.contains("display_size"), "{message}");
        }
        // Without a display or input the size is not looked at.
        assert!(
            EmulatedMachine::new(
                MachineConfig::new()
                    .kernel(fake_bzimage(&[0xf4]))
                    .display_size(1, 1)
            )
            .is_ok()
        );
    }
}

mod config {
    use super::*;

    const MAC: [u8; 6] = [2, 0, 0, 0, 0, 1];

    #[test]
    fn defaults_match_the_documented_ones() {
        let config = MachineConfig::new().reboot(false);
        assert!(config.kernel.is_empty());
        assert!(config.initrd.is_none());
        assert!(config.cmdline.is_none());
        assert!(config.cmdline_extra.is_empty());
        assert_eq!(config.memory_mib, 512);
        assert!(config.disks.is_empty());
        assert!(config.net_macs.is_empty());
        assert!(config.consoles.is_empty());
        assert_eq!(config.jit, JitMode::Auto);
        assert_eq!(config.clock, ClockMode::Realtime);
        assert!(config.log.is_none());
        assert!(config.poll.is_none());
        assert!(!config.debug);
        assert!(!config.verify);
        assert!(config.unix_time.is_none());
    }

    #[test]
    fn default_and_new_agree() {
        let (a, b) = (MachineConfig::new().reboot(false), MachineConfig::default());
        assert_eq!(
            (a.memory_mib, a.jit, a.clock),
            (b.memory_mib, b.jit, b.clock)
        );
    }

    #[test]
    fn every_builder_stores_its_value() {
        let config = MachineConfig::new()
            .reboot(false)
            .kernel(&[1u8, 2, 3][..])
            .initrd(vec![4u8, 5])
            .cmdline("quiet")
            .cmdline_append("a=1")
            .cmdline_append(String::from("b=2"))
            .disk(Disk::memory(vec![0; 512]))
            .net(MAC)
            .console(ConsoleKind::Virtio)
            .memory_mib(128)
            .jit(JitMode::On)
            .clock(ClockMode::Realtime)
            .debug(true)
            .verify_jit(true)
            .unix_time(42)
            .log(|_| {})
            .poll(|_, _| {});
        assert_eq!(config.kernel, [1, 2, 3]);
        assert_eq!(config.initrd.as_deref(), Some(&[4u8, 5][..]));
        assert_eq!(config.cmdline.as_deref(), Some("quiet"));
        assert_eq!(config.cmdline_extra, ["a=1", "b=2"]);
        assert_eq!(config.disks[0].backend.len(), 512);
        assert_eq!(config.net_macs, [MAC]);
        assert_eq!(config.consoles, [ConsoleKind::Virtio]);
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.jit, JitMode::On);
        assert_eq!(config.clock, ClockMode::Realtime);
        assert!(config.debug);
        assert!(config.verify);
        assert_eq!(config.unix_time, Some(42));
        assert!(config.log.is_some());
        assert!(config.poll.is_some());
    }

    #[test]
    fn a_later_builder_call_overrides_an_earlier_one() {
        let config = MachineConfig::new()
            .reboot(false)
            .memory_mib(100)
            .memory_mib(200)
            .cmdline("a")
            .cmdline("b")
            .debug(true)
            .debug(false)
            .jit(JitMode::Off)
            .jit(JitMode::Auto);
        assert_eq!(config.memory_mib, 200);
        assert_eq!(config.cmdline.as_deref(), Some("b"));
        assert!(!config.debug);
        assert_eq!(config.jit, JitMode::Auto);
    }

    #[test]
    fn enum_defaults() {
        assert_eq!(JitMode::default(), JitMode::Auto);
        assert_eq!(ClockMode::default(), ClockMode::Realtime);
        assert_eq!(ConsoleKind::default(), ConsoleKind::Virtio);
    }

    #[test]
    fn memory_limits_are_the_documented_ones() {
        assert_eq!(MIN_MEMORY_MIB, 64);
        // The last byte of the largest guest is the last one the 40-bit physical address space
        // has, once the PCI hole is skipped.
        assert_eq!(
            crate::ram::flat_bytes((MAX_MEMORY_MIB as u64) << BYTES_PER_MIB_SHIFT),
            1 << 40
        );
    }

    fn with_memory(mib: usize) -> Result<EmulatedMachine, Error> {
        EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(mib),
        )
    }

    #[test]
    fn memory_from_the_minimum_to_past_the_pci_hole_is_accepted() {
        // The upper limit is not probed: acceptance at it would reserve a terabyte of address
        // space, which a host that does not overcommit refuses.
        for mib in [MIN_MEMORY_MIB, MIN_MEMORY_MIB + 1, 3072, 3073, 4096] {
            assert!(with_memory(mib).is_ok(), "{mib}");
        }
    }

    #[test]
    fn memory_just_outside_the_limits_is_rejected_with_the_range_in_the_message() {
        for mib in [0, 1, MIN_MEMORY_MIB - 1, MAX_MEMORY_MIB + 1, usize::MAX] {
            let message = expect_config_error(with_memory(mib));
            assert_eq!(
                message,
                format!("memory_mib must be between {MIN_MEMORY_MIB} and {MAX_MEMORY_MIB}"),
                "{mib}"
            );
        }
    }

    #[test]
    fn memory_is_checked_before_the_kernel_is_looked_at() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(vec![0u8; 16])
                .memory_mib(1),
        );
        assert!(matches!(result, Err(Error::BadConfig(_))));
    }

    #[test]
    fn a_default_config_without_a_kernel_is_a_bad_kernel() {
        match EmulatedMachine::new(MachineConfig::new().reboot(false)) {
            Err(Error::BadKernel(message)) => assert!(message.contains("HdrS"), "{message}"),
            other => panic!("expected BadKernel, got {:?}", other.err()),
        }
    }

    #[test]
    fn a_kernel_the_loader_rejects_surfaces_its_error() {
        let old = FakeKernel {
            version: 0x0200,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(old)
                .memory_mib(64),
        );
        assert!(matches!(result, Err(Error::BadKernel(_))));
    }

    #[test]
    fn an_initrd_too_large_for_ram_is_a_config_error() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .initrd(vec![0u8; 65 * MIB])
                .memory_mib(MIN_MEMORY_MIB),
        );
        assert!(expect_config_error(result).contains("initrd"));
    }

    #[test]
    fn a_too_long_command_line_is_a_config_error() {
        let kernel = FakeKernel {
            cmdline_max: 32,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel)
                .cmdline("x".repeat(40))
                .memory_mib(64),
        );
        assert!(expect_config_error(result).contains("command line"));
    }

    #[test]
    fn appended_text_counts_towards_the_command_line_limit() {
        let kernel = FakeKernel {
            cmdline_max: 12,
            ..FakeKernel::new(&[0xf4])
        }
        .build();
        let fits = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel.clone())
                .cmdline("abcd")
                .cmdline_append("efgh")
                .memory_mib(64),
        );
        assert!(fits.is_ok());
        let too_long = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(kernel)
                .cmdline("abcd")
                .cmdline_append("efghijk")
                .memory_mib(64),
        );
        expect_config_error(too_long);
    }

    #[test]
    fn jit_off_and_auto_never_fail_for_lack_of_a_jit() {
        for mode in auto_and_off() {
            let config = MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .jit(mode);
            assert!(EmulatedMachine::new(config).is_ok(), "{mode:?}");
        }
    }

    #[test]
    fn jit_on_succeeds_exactly_when_the_host_can_run_generated_code() {
        let config = MachineConfig::new()
            .reboot(false)
            .kernel(fake_bzimage(&[0xf4]))
            .memory_mib(64)
            .jit(JitMode::On);
        match EmulatedMachine::new(config) {
            Ok(_) => assert!(host_has_jit()),
            Err(Error::JitUnavailable) => assert!(!host_has_jit()),
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }

    #[cfg(not(all(
        feature = "jit",
        any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            target_arch = "wasm32"
        )
    )))]
    #[test]
    fn without_a_jit_backend_jit_on_is_jit_unavailable_even_before_the_kernel_is_parsed() {
        let result = EmulatedMachine::new(MachineConfig::new().reboot(false).jit(JitMode::On));
        assert!(matches!(result, Err(Error::JitUnavailable)));
    }

    #[cfg(all(
        feature = "jit",
        any(
            target_arch = "aarch64",
            target_arch = "x86_64",
            target_arch = "wasm32"
        )
    ))]
    #[test]
    fn with_a_jit_backend_jit_on_is_accepted() {
        let result = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .jit(JitMode::On),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn errors_have_readable_messages() {
        assert_eq!(
            Error::BadKernel("x".into()).to_string(),
            "bad kernel image: x"
        );
        assert_eq!(
            Error::BadConfig("y".into()).to_string(),
            "bad configuration: y"
        );
        assert_eq!(
            Error::JitUnavailable.to_string(),
            "JIT requested but unavailable on this host"
        );
        assert_eq!(Error::AlreadyStarted.to_string(), "machine already started");
    }

    #[test]
    fn error_is_a_std_error_without_a_source() {
        let boxed: Box<dyn core::error::Error> = Box::new(Error::AlreadyStarted);
        assert!(boxed.source().is_none());
        assert_eq!(boxed.to_string(), "machine already started");
    }
}

mod guest_view {
    use super::*;

    // mov ebx, [rsi+0x228]; mov dx, 0x3f8; 1: mov al, [rbx]; test al, al; jz 2f; out dx, al;
    // inc rbx; jmp 1b; 2: mov al, 0xfe; out 0x64, al; hlt
    const PRINT_CMDLINE: [u8; 27] = [
        0x8b, 0x9e, 0x28, 0x02, 0x00, 0x00, 0x66, 0xba, 0xf8, 0x03, 0x8a, 0x03, 0x84, 0xc0, 0x74,
        0x06, 0xee, 0x48, 0xff, 0xc3, 0xeb, 0xf4, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    // mov al, [rsi+0x1e8]; mov dx, 0x3f8; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_E820_COUNT: [u8; 16] = [
        0x8a, 0x86, 0xe8, 0x01, 0x00, 0x00, 0x66, 0xba, 0xf8, 0x03, 0xee, 0xb0, 0xfe, 0xe6, 0x64,
        0xf4,
    ];
    // mov dx, 0x3f8; mov eax, [rsi+0x300]; out dx, al; shr eax, 8; out dx, al; shr eax, 8;
    // out dx, al; shr eax, 8; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_E820_LAST_SIZE: [u8; 28] = [
        0x66, 0xba, 0xf8, 0x03, 0x8b, 0x86, 0x00, 0x03, 0x00, 0x00, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    // Prints ramdisk_image and ramdisk_size (4 bytes each, little endian), then the byte sum of
    // the initrd, then resets.
    // mov dx, 0x3f8; mov eax, [rsi+0x218]; out dx, al; shr eax, 8; out dx, al; shr eax, 8;
    // out dx, al; shr eax, 8; out dx, al; mov eax, [rsi+0x21c]; mov ecx, eax; out dx, al;
    // shr eax, 8; out dx, al; shr eax, 8; out dx, al; shr eax, 8; out dx, al;
    // mov ebx, [rsi+0x218]; xor eax, eax; test ecx, ecx; jz 2f; 1: add al, [rbx]; inc rbx;
    // dec ecx; jnz 1b; 2: out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const PRINT_INITRD_AND_CHECKSUM: &[u8] = &[
        0x66, 0xba, 0xf8, 0x03, 0x8b, 0x86, 0x18, 0x02, 0x00, 0x00, 0xee, 0xc1, 0xe8, 0x08, 0xee,
        0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0x8b, 0x86, 0x1c, 0x02, 0x00, 0x00, 0x89,
        0xc1, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0xc1, 0xe8, 0x08, 0xee, 0x8b,
        0x9e, 0x18, 0x02, 0x00, 0x00, 0x31, 0xc0, 0x85, 0xc9, 0x74, 0x09, 0x02, 0x03, 0x48, 0xff,
        0xc3, 0xff, 0xc9, 0x75, 0xf7, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    fn printed_cmdline(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> String {
        let run = run_machine(
            fake_bzimage(&PRINT_CMDLINE),
            JitMode::Off,
            GENEROUS_LIMIT,
            configure,
        );
        assert_eq!(run.exit, ExitReason::Reset);
        run.text()
    }

    #[test]
    fn default_command_line_is_the_virtio_one_from_platform() {
        let printed = printed_cmdline(|config| config.acpi(false));
        assert_eq!(printed, DEFAULT_CMDLINE);
        assert!(printed.contains("console=hvc0"));
        assert!(printed.contains("earlyprintk=serial,ttyS0"));
    }

    #[test]
    fn default_command_line_for_a_serial_console_names_ttys0() {
        assert_eq!(
            printed_cmdline(|config| config.console(ConsoleKind::Serial).acpi(false)),
            DEFAULT_SERIAL_CMDLINE
        );
        assert!(DEFAULT_SERIAL_CMDLINE.contains("console=ttyS0"));
    }

    #[test]
    fn the_first_console_picks_the_default_command_line() {
        assert_eq!(
            printed_cmdline(|config| config
                .console(ConsoleKind::Serial)
                .console(ConsoleKind::Virtio)
                .acpi(false)),
            DEFAULT_SERIAL_CMDLINE
        );
    }

    #[test]
    fn explicit_command_line_replaces_the_default() {
        assert_eq!(
            printed_cmdline(|config| config.cmdline("root=/dev/vda ro")),
            "root=/dev/vda ro"
        );
    }

    #[test]
    fn explicit_command_line_ignores_the_console_kind() {
        for kind in [ConsoleKind::Serial, ConsoleKind::Virtio] {
            assert_eq!(
                printed_cmdline(|config| config.cmdline("single").console(kind)),
                "single"
            );
        }
    }

    #[test]
    fn appended_text_follows_the_default_after_a_space() {
        assert_eq!(
            printed_cmdline(|config| config.cmdline_append("loglevel=7").acpi(false)),
            format!("{DEFAULT_CMDLINE} loglevel=7")
        );
    }

    #[test]
    fn appended_text_follows_an_explicit_command_line_in_call_order() {
        assert_eq!(
            printed_cmdline(|config| config
                .cmdline_append("first")
                .cmdline("base")
                .cmdline_append("second")),
            "base first second"
        );
    }

    #[test]
    fn empty_explicit_command_line_is_an_empty_string_not_the_default() {
        assert_eq!(printed_cmdline(|config| config.cmdline("")), "");
        assert_eq!(
            printed_cmdline(|config| config.cmdline("").cmdline_append("x")),
            " x"
        );
    }

    #[test]
    fn a_display_puts_the_console_on_tty0_after_the_default_and_before_appended_text() {
        assert_eq!(
            printed_cmdline(|config| config
                .acpi(false)
                .display(DisplayMode::Framebuffer)
                .cmdline_append("console=ttyS0")),
            format!("{DEFAULT_CMDLINE} console=tty0 console=ttyS0")
        );
        assert_eq!(
            printed_cmdline(|config| config
                .cmdline("root=/dev/vda")
                .display(DisplayMode::Framebuffer)),
            "root=/dev/vda",
            "an explicit command line is the embedder's"
        );
    }

    #[test]
    fn a_display_that_is_not_the_console_leaves_the_command_line_alone() {
        assert_eq!(
            printed_cmdline(|config| config
                .acpi(false)
                .display(DisplayMode::Framebuffer)
                .console_on_screen(false)),
            DEFAULT_CMDLINE
        );
        assert_eq!(
            printed_cmdline(|config| config.acpi(false).input(true)),
            DEFAULT_CMDLINE,
            "input alone has no screen to put a console on"
        );
    }

    #[test]
    fn with_acpi_the_default_command_line_no_longer_says_acpi_off() {
        let printed = printed_cmdline(|config| config);
        assert_eq!(printed, DEFAULT_CMDLINE.replace(" acpi=off", ""));
        assert!(!printed.contains("acpi"));
        let serial = printed_cmdline(|config| config.console(ConsoleKind::Serial));
        assert_eq!(serial, DEFAULT_SERIAL_CMDLINE.replace(" acpi=off", ""));
    }

    #[test]
    fn e820_describes_three_ranges_without_acpi() {
        let run = run_machine(
            fake_bzimage(&PRINT_E820_COUNT),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.acpi(false),
        );
        assert_eq!(run.console_output, [3]);
    }

    #[test]
    fn with_acpi_e820_also_lists_the_tables_and_the_ecam_window() {
        let run = run_guest(&PRINT_E820_COUNT, JitMode::Off);
        assert_eq!(run.console_output, [6]);
    }

    // Writes at the first and the last dword of 4 GiB..5 GiB (the RAM that a 4096 MiB guest has
    // above the PCI hole), then prints the first byte of the first, the last of the second, the
    // bytes of TOP_MEM2 (bits 31..24, then 39..32) and the top byte of TOP_MEM.
    // Assembled with `hack/asm.sh`; high.s is
    // mov dx, 0x3f8; mov rbx, 0x100000000; mov dword ptr [rbx], 0x12345678;
    // mov rcx, 0x13ffffffc; mov dword ptr [rcx], 0x9abcdef0; mov al, [rbx]; out dx, al;
    // mov al, [rcx+3]; out dx, al; mov ecx, 0xc001001d; rdmsr; mov r8d, edx; mov dx, 0x3f8;
    // shr eax, 24; out dx, al; mov eax, r8d; out dx, al; mov ecx, 0xc001001a; rdmsr;
    // mov dx, 0x3f8; shr eax, 24; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    // (rdmsr overwrites edx, hence the second `mov dx`)
    const TOUCH_HIGH_RAM: [u8; 85] = [
        0x66, 0xba, 0xf8, 0x03, 0x48, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc7,
        0x03, 0x78, 0x56, 0x34, 0x12, 0x48, 0xb9, 0xfc, 0xff, 0xff, 0x3f, 0x01, 0x00, 0x00, 0x00,
        0xc7, 0x01, 0xf0, 0xde, 0xbc, 0x9a, 0x8a, 0x03, 0xee, 0x8a, 0x41, 0x03, 0xee, 0xb9, 0x1d,
        0x00, 0x01, 0xc0, 0x0f, 0x32, 0x41, 0x89, 0xd0, 0x66, 0xba, 0xf8, 0x03, 0xc1, 0xe8, 0x18,
        0xee, 0x44, 0x89, 0xc0, 0xee, 0xb9, 0x1a, 0x00, 0x01, 0xc0, 0x0f, 0x32, 0x66, 0xba, 0xf8,
        0x03, 0xc1, 0xe8, 0x18, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn ram_above_4_gib_is_there_and_the_amd_memory_top_msrs_say_so() {
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&TOUCH_HIGH_RAM),
                jit,
                GENEROUS_LIMIT,
                |config| config.memory_mib(4096),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            // TOP_MEM2 is 5 GiB = 0x1_4000_0000, TOP_MEM the start of the PCI hole.
            assert_eq!(
                run.console_output,
                [0x78, 0x9a, 0x40, 0x01, 0xc0],
                "{jit:?}"
            );
        }
    }

    // Reads the 16-bit `num_queues` field of the first virtio function's common config, which
    // sits at the start of the PCI hole (0xc000_0000 + 0x12), and prints its low byte.
    // mov dx, 0x3f8; mov ebx, 0xc0000012; mov ax, [rbx]; out dx, al; mov al, 0xfe; out 0x64, al;
    // hlt
    const READ_PCI_HOLE: [u8; 18] = [
        0x66, 0xba, 0xf8, 0x03, 0xbb, 0x12, 0x00, 0x00, 0xc0, 0x66, 0x8b, 0x03, 0xee, 0xb0, 0xfe,
        0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn the_pci_hole_is_the_devices_even_when_the_ram_around_it_is_there() {
        // With 4096 MiB the flat RAM buffer covers 0xc000_0000 as well; the access must still go
        // to the virtio BAR, which RAM would answer with zero.
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&READ_PCI_HOLE),
                jit,
                GENEROUS_LIMIT,
                |config| config.memory_mib(4096),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_ne!(run.console_output, [0], "{jit:?}");
            assert_eq!(run.console_output.len(), 1, "{jit:?}");
        }
    }

    // Prints the high dword of the fourth e820 entry's base, the top byte of its base's low
    // dword, the top byte of its size's low dword, and the entry count.
    // mov dx, 0x3f8; mov al, [rsi+0x310]; out dx, al; mov al, [rsi+0x30f]; out dx, al;
    // mov al, [rsi+0x317]; out dx, al; mov al, [rsi+0x1e8]; out dx, al; mov al, 0xfe;
    // out 0x64, al; hlt
    const PRINT_E820_HIGH_ENTRY: [u8; 37] = [
        0x66, 0xba, 0xf8, 0x03, 0x8a, 0x86, 0x10, 0x03, 0x00, 0x00, 0xee, 0x8a, 0x86, 0x0f, 0x03,
        0x00, 0x00, 0xee, 0x8a, 0x86, 0x17, 0x03, 0x00, 0x00, 0xee, 0x8a, 0x86, 0xe8, 0x01, 0x00,
        0x00, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn e820_lists_the_ram_above_4_gib_as_the_last_range() {
        let run = run_machine(
            fake_bzimage(&PRINT_E820_HIGH_ENTRY),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.memory_mib(4096).acpi(false),
        );
        // The fourth of four entries starts at 4 GiB and is the 1 GiB that does not fit below
        // the PCI hole.
        assert_eq!(run.console_output, [0x01, 0x00, 0x40, 0x04]);
    }

    #[test]
    fn the_third_e820_range_ends_at_the_pci_hole_with_ram_above_4_gib() {
        let run = run_machine(
            fake_bzimage(&PRINT_E820_LAST_SIZE),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.memory_mib(4096).acpi(false),
        );
        let size = u32::from_le_bytes(run.console_output[..4].try_into().unwrap());
        // 3 GiB less the first MiB.
        assert_eq!(size as usize, 3072 * MIB - MIB);
    }

    #[test]
    fn e820_extended_range_tracks_memory_mib() {
        for mib in [MIN_MEMORY_MIB, 100, 256] {
            let run = run_machine(
                fake_bzimage(&PRINT_E820_LAST_SIZE),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.memory_mib(mib).acpi(false),
            );
            let size = u32::from_le_bytes(run.console_output[..4].try_into().unwrap());
            assert_eq!(size as usize, (mib - 1) * MIB, "{mib} MiB");
        }
    }

    #[test]
    fn initrd_is_visible_to_the_guest_with_its_contents() {
        let initrd: Vec<u8> = (0..1000u32).map(|index| (index * 31 + 7) as u8).collect();
        let checksum = initrd.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte));
        for jit in auto_and_off() {
            let initrd = initrd.clone();
            let run = run_machine(
                fake_bzimage(PRINT_INITRD_AND_CHECKSUM),
                jit,
                GENEROUS_LIMIT,
                move |config| config.initrd(initrd),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            let expected_image = ((MIN_MEMORY_MIB * MIB) - 1000) & !0xfff;
            assert_eq!(
                run.console_output[..4],
                (expected_image as u32).to_le_bytes(),
                "{jit:?}"
            );
            assert_eq!(run.console_output[4..8], 1000u32.to_le_bytes(), "{jit:?}");
            assert_eq!(run.console_output[8], checksum, "{jit:?}");
        }
    }

    #[test]
    fn without_an_initrd_the_guest_sees_an_empty_ramdisk() {
        let run = run_guest(PRINT_INITRD_AND_CHECKSUM, JitMode::Off);
        assert_eq!(run.console_output, [0u8; 9]);
    }
}

mod lifecycle {
    use super::*;

    #[test]
    fn wait_without_start_is_stopped() {
        let mut machine = new_machine(&HELLO_RESET);
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn starting_twice_is_an_error() {
        let mut machine = new_machine(&SPIN);
        machine.start().unwrap();
        assert!(matches!(machine.start(), Err(Error::AlreadyStarted)));
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn starting_after_the_machine_ran_is_an_error() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert!(matches!(machine.start(), Err(Error::AlreadyStarted)));
    }

    #[test]
    fn waiting_twice_reports_stopped_the_second_time() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stop_before_start_ends_the_guest_before_it_runs() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .kernel(fake_bzimage(&HELLO_RESET))
                    .memory_mib(64)
                    .jit(jit),
            )
            .unwrap();
            let output = Arc::new(Mutex::new(Vec::new()));
            let sink = output.clone();
            machine
                .console()
                .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
            machine.stop();
            machine.start().unwrap();
            assert_eq!(machine.wait(), ExitReason::Stopped, "{jit:?}");
            assert!(output.lock().unwrap().is_empty(), "{jit:?}");
        }
    }

    #[test]
    fn a_stop_handle_from_before_start_also_prevents_the_run() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.stop_handle().stop();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stopping_twice_is_harmless() {
        let mut machine = new_machine(&SPIN);
        machine.start().unwrap();
        machine.stop();
        machine.stop();
        machine.stop_handle().stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn stop_handles_are_clones_of_one_flag() {
        let mut machine = new_machine(&SPIN);
        let first = machine.stop_handle();
        let second = first.clone();
        machine.start().unwrap();
        second.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
        first.stop();
    }

    #[test]
    fn a_stop_handle_stops_a_running_guest_from_another_thread() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .kernel(fake_bzimage(&SPIN))
                    .memory_mib(64)
                    .jit(jit),
            )
            .unwrap();
            let handle = machine.stop_handle();
            machine.start().unwrap();
            let stopper = std::thread::spawn(move || handle.stop());
            assert_eq!(machine.wait(), ExitReason::Stopped, "{jit:?}");
            stopper.join().unwrap();
        }
    }

    #[test]
    fn stop_after_the_guest_exited_does_not_change_the_exit_reason() {
        let mut machine = new_machine(&HELLO_RESET);
        machine.start().unwrap();
        // The reset happens within a few instructions; let it finish, then stop.
        let deadline = Instant::now() + GENEROUS_LIMIT;
        while machine
            .thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Reset);
    }

    #[test]
    fn a_new_machine_has_no_thread_until_started() {
        let machine = new_machine(&HELLO_RESET);
        assert!(machine.thread.is_none());
        drop(machine);
    }

    #[test]
    fn console_and_net_handles_are_shared_with_the_devices() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .net([2, 0, 0, 0, 0, 2]),
        )
        .unwrap();
        machine.console().write(b"in");
        let vcpu = machine.vcpu.as_ref().unwrap();
        assert_eq!(vcpu.uart_console.guest_rx(), Some(b'i'));
        assert_eq!(vcpu.uart_console.guest_rx(), Some(b'n'));
        let net = machine.net().unwrap();
        assert!(net.send_frame(b"frame"));
        let device_side = &vcpu.devcfg.nets[0].1;
        assert_eq!(
            device_side.guest_recv_bytes().as_deref(),
            Some(&b"frame"[..])
        );
        // And the other direction.
        device_side.guest_send(b"reply".to_vec());
        assert_eq!(net.try_recv_frame().as_deref(), Some(&b"reply"[..]));
        // The machine never started: nothing runs, nothing to stop.
        assert_eq!(machine.wait(), ExitReason::Stopped);
    }

    #[test]
    fn net_handle_exists_only_when_configured() {
        assert!(new_machine(&[0xf4]).net().is_none());
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(64)
                .net([0; 6]),
        )
        .unwrap();
        assert!(machine.net().is_some());
    }

    #[test]
    fn every_console_handle_from_one_machine_is_the_same_console() {
        let machine = new_machine(&[0xf4]);
        let (a, b) = (machine.console(), machine.console());
        a.resize(100, 40);
        assert_eq!(b.size(), (100, 40));
    }

    #[test]
    fn two_machines_do_not_share_a_console() {
        let (first, second) = (new_machine(&[0xf4]), new_machine(&[0xf4]));
        first.console().resize(10, 10);
        assert_eq!(second.console().size(), (80, 25));
    }
}

mod exits {
    use super::*;

    // mov al, 1; out 0x92, al; hlt
    const FAST_A20_RESET: [u8; 5] = [0xb0, 0x01, 0xe6, 0x92, 0xf4];
    // mov dx, 0xcf9; mov al, 4; out dx, al; hlt
    const RESET_CONTROL_RESET: [u8; 8] = [0x66, 0xba, 0xf9, 0x0c, 0xb0, 0x04, 0xee, 0xf4];
    // mov al, 0xd1; out 0x64, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const OTHER_KBC_COMMAND: [u8; 13] = [
        0xb0, 0xd1, 0xe6, 0x64, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe,
    ];
    // mov al, 2; out 0x92, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const A20_GATE_ONLY: [u8; 13] = [
        0xb0, 0x02, 0xe6, 0x92, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb, 0xfe,
    ];
    // mov dx, 0xcf9; mov al, 0xb; out dx, al; mov dx, 0x3f8; mov al, 'k'; out dx, al; jmp $
    const RESET_CONTROL_WITHOUT_CPU_RESET: [u8; 16] = [
        0x66, 0xba, 0xf9, 0x0c, 0xb0, 0x0b, 0xee, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xeb,
        0xfe,
    ];

    // mov dx, 0x604; mov ax, 0x3401 (SLP_EN, SLP_TYP 5, SCI_EN); out dx, ax; jmp $
    const ACPI_POWER_OFF: [u8; 10] = [0x66, 0xba, 0x04, 0x06, 0x66, 0xb8, 0x01, 0x34, 0x66, 0xef];
    // The same with sleep type 3 (an S3 the DSDT does not offer), then print 'k'; jmp $
    const ACPI_SLEEP_TYPE_3: [u8; 19] = [
        0x66, 0xba, 0x04, 0x06, 0x66, 0xb8, 0x01, 0x2c, 0x66, 0xef, 0x66, 0xba, 0xf8, 0x03, 0xb0,
        0x6b, 0xee, 0xeb, 0xfe,
    ];
    // mov dx, 0xcf9; mov al, 6 (what the FADT's reset register holds); out dx, al; jmp $
    const ACPI_RESET_REGISTER: [u8; 9] = [0x66, 0xba, 0xf9, 0x0c, 0xb0, 0x06, 0xee, 0xeb, 0xfe];

    #[test]
    fn writing_s5_to_the_acpi_sleep_register_powers_the_machine_off() {
        for jit in auto_and_off() {
            let mut code = ACPI_POWER_OFF.to_vec();
            code.extend_from_slice(&[0xeb, 0xfe]);
            assert_eq!(run_guest(&code, jit).exit, ExitReason::Shutdown, "{jit:?}");
        }
    }

    #[test]
    fn other_acpi_sleep_types_do_not_power_off() {
        for jit in auto_and_off() {
            let run = run_spinning(&ACPI_SLEEP_TYPE_3, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn the_fadts_reset_register_value_resets_the_machine() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&ACPI_RESET_REGISTER, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn keyboard_controller_pulse_reset_is_reset() {
        for jit in auto_and_off() {
            let run = run_guest(&HELLO_RESET, jit);
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.text(), "hi", "{jit:?}");
        }
    }

    #[test]
    fn fast_a20_bit_zero_is_reset() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&FAST_A20_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn reset_control_bit_two_is_reset() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&RESET_CONTROL_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn other_keyboard_controller_commands_do_not_reset() {
        for jit in auto_and_off() {
            let run = run_spinning(&OTHER_KBC_COMMAND, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn enabling_a20_alone_does_not_reset() {
        let run = run_spinning(&A20_GATE_ONLY, JitMode::Off);
        assert_eq!(run.exit, ExitReason::Stopped);
        assert_eq!(run.console_output, [ALIVE_MARKER]);
    }

    #[test]
    fn reset_control_without_the_cpu_reset_bit_does_not_reset() {
        let run = run_spinning(&RESET_CONTROL_WITHOUT_CPU_RESET, JitMode::Off);
        assert_eq!(run.exit, ExitReason::Stopped);
        assert_eq!(run.console_output, [ALIVE_MARKER]);
    }

    #[test]
    fn a_looping_guest_runs_until_stopped() {
        for jit in auto_and_off() {
            let run = run_spinning(&PRINT_AND_SPIN, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn a_halted_guest_with_interrupts_off_is_powered_off() {
        for jit in auto_and_off() {
            let run = run_guest(&PRINT_THEN_HALT_WITH_INTERRUPTS_OFF, jit);
            assert_eq!(run.exit, ExitReason::Shutdown, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn sti_then_hlt_waits_for_an_interrupt_instead_of_powering_off() {
        // `sti` leaves a one-instruction interrupt shadow, but IF is set when `hlt` runs.
        for jit in auto_and_off() {
            let run = run_spinning(&PRINT_THEN_STI_HALT, jit);
            assert_eq!(run.exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{jit:?}");
        }
    }

    #[test]
    fn a_reset_written_before_a_halt_with_interrupts_off_is_a_reset_not_a_power_off() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&HELLO_RESET, jit).exit,
                ExitReason::Reset,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn an_exception_with_no_idt_is_a_triple_fault() {
        for jit in auto_and_off() {
            assert_eq!(
                run_guest(&NULL_IDT_THEN_INT3, jit).exit,
                ExitReason::TripleFault,
                "{jit:?}"
            );
        }
    }

    #[test]
    fn a_triple_fault_logs_a_state_dump_even_without_debug() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&NULL_IDT_THEN_INT3),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log),
        );
        assert_eq!(run.exit, ExitReason::TripleFault);
        let lines = lines.lock().unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("icount="), "{}", lines[0]);
        assert!(lines[0].contains("rip="), "{}", lines[0]);
        // INT3 is a trap: the state is left after it, at the end of the 9-byte program.
        assert!(
            lines[0].contains(&format!(
                "rip={:016x}",
                CODE_BASE + NULL_IDT_THEN_INT3.len() as u64
            )),
            "{}",
            lines[0]
        );
        assert!(!lines[0].contains("last rips:"), "no trace without debug");
    }

    #[test]
    fn a_plain_reset_logs_nothing() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert!(lines.lock().unwrap().is_empty());
    }

    #[test]
    fn debug_logs_a_dump_with_a_trace_on_every_exit() {
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log).debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        let lines = lines.lock().unwrap();
        assert_eq!(lines.len(), 1);
        let dump = &lines[0];
        for expected in [
            "flushes=",
            "icount=",
            "cr3=9000",
            "efer=500",
            "halted=",
            "dev:",
            "last rips:",
        ] {
            assert!(dump.contains(expected), "missing {expected:?} in {dump}");
        }
        // The reset instruction is the last one executed; the trace lists newest first.
        let last_rip = format!("last rips: {:x}", CODE_BASE + HELLO_RESET_OUT_OFFSET);
        assert!(dump.contains(&last_rip), "{dump}");
    }

    #[test]
    fn debug_without_a_log_hook_is_silent_and_harmless() {
        let run = run_machine(
            fake_bzimage(&HELLO_RESET),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
    }

    #[test]
    fn the_dump_finds_a_kernel_banner_in_guest_ram() {
        let mut code = RESET_AND_HALT.to_vec();
        code.extend_from_slice(b"Linux version 9.9.9-fake (unit test) #1\0");
        let (lines, log) = log_collector();
        let run = run_machine(
            fake_bzimage(&code),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.log(log).debug(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        let lines = lines.lock().unwrap();
        assert!(lines[0].contains("--- ram match at"), "{}", lines[0]);
        assert!(
            lines[0].contains("Linux version 9.9.9-fake (unit test) #1"),
            "{}",
            lines[0]
        );
    }

    #[test]
    fn a_panic_in_a_host_hook_is_reported_as_an_internal_exit_not_propagated() {
        let run = run_machine(
            fake_bzimage(&SPIN),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| config.poll(|_, _| panic!("host hook failed")),
        );
        assert_eq!(
            run.exit,
            ExitReason::Internal("vcpu thread panicked".into())
        );
    }

    #[test]
    fn a_panic_in_the_console_callback_is_also_an_internal_exit() {
        // Interpreted: the x86_64 JIT calls port I/O helpers through `extern "C"`, so a panic
        // out of the callback there aborts the process instead of unwinding to the join.
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .jit(JitMode::Off),
        )
        .unwrap();
        machine.console().on_output(|_| panic!("embedder bug"));
        machine.start().unwrap();
        assert_eq!(
            machine.wait(),
            ExitReason::Internal("vcpu thread panicked".into())
        );
    }

    /// Exhaustive on purpose: adding or removing a variant must be a conscious API change.
    fn variant_name(reason: &ExitReason) -> &'static str {
        match reason {
            ExitReason::Shutdown => "shutdown",
            ExitReason::Reset => "reset",
            ExitReason::TripleFault => "triple fault",
            ExitReason::Stopped => "stopped",
            ExitReason::Internal(_) => "internal",
        }
    }

    #[test]
    fn exit_reasons_the_machine_produces_are_reset_shutdown_stopped_and_triple_fault() {
        let produced = [
            run_guest(&HELLO_RESET, JitMode::Off).exit,
            run_guest(&FAST_A20_RESET, JitMode::Off).exit,
            run_guest(&RESET_CONTROL_RESET, JitMode::Off).exit,
            run_guest(&NULL_IDT_THEN_INT3, JitMode::Off).exit,
            run_spinning(&SPIN, JitMode::Off).exit,
            run_guest(&PRINT_THEN_HALT_WITH_INTERRUPTS_OFF, JitMode::Off).exit,
            new_machine(&HELLO_RESET).wait(),
        ];
        let names: Vec<_> = produced.iter().map(variant_name).collect();
        assert_eq!(
            names,
            [
                "reset",
                "reset",
                "reset",
                "triple fault",
                "stopped",
                "shutdown",
                "stopped"
            ]
        );
    }

    #[test]
    fn exit_reasons_compare_and_clone() {
        let reason = ExitReason::Internal("x".into());
        assert_eq!(reason.clone(), reason);
        assert_ne!(reason, ExitReason::Internal("y".into()));
        assert_ne!(ExitReason::Reset, ExitReason::Stopped);
    }
}

mod jit_modes {
    use super::*;

    #[test]
    fn every_usable_mode_runs_the_same_guest_to_the_same_result() {
        let mut modes = vec![JitMode::Off, JitMode::Auto];
        if host_has_jit() {
            modes.push(JitMode::On);
        }
        for jit in modes {
            let run = run_guest(&HELLO_RESET, jit);
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.text(), "hi", "{jit:?}");
        }
    }

    #[test]
    fn jit_on_is_refused_before_anything_is_loaded_when_the_host_cannot_jit() {
        let config = MachineConfig::new().reboot(false).jit(JitMode::On);
        match EmulatedMachine::new(config) {
            // Valid host: the empty kernel is what fails.
            Err(Error::BadKernel(_)) => assert!(host_has_jit()),
            Err(Error::JitUnavailable) => assert!(!host_has_jit()),
            other => panic!("unexpected {:?}", other.err()),
        }
    }
}

mod clock {
    use super::*;

    // Sets rsp (interrupt delivery pushes on it; the boot state leaves it 0), installs a gate for vector 0x20 (IRQ 0 after the PIC remap below) in an IDT at 0x8000, then
    // programs PIT channel 0 for a 20 ms one-shot (mode 0, count 0x5d38 at 1.193182 MHz), unmasks
    // IRQ 0 and does `sti; hlt`. The handler prints 't' and resets.
    const WAIT_FOR_PIT_IRQ: [u8; 125] = [
        0xbc, 0x00, 0x00, 0x08, 0x00, 0x48, 0x8d, 0x05, 0x65, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3,
        0x00, 0x82, 0x00, 0x00, 0x66, 0x89, 0x03, 0x66, 0xc7, 0x43, 0x02, 0x10, 0x00, 0x66, 0xc7,
        0x43, 0x04, 0x00, 0x8e, 0x48, 0xc1, 0xe8, 0x10, 0x66, 0x89, 0x43, 0x06, 0x48, 0xc1, 0xe8,
        0x10, 0x89, 0x43, 0x08, 0xc7, 0x43, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc3, 0x00,
        0x8f, 0x00, 0x00, 0x66, 0xc7, 0x03, 0xff, 0x0f, 0x48, 0xc7, 0x43, 0x02, 0x00, 0x80, 0x00,
        0x00, 0x0f, 0x01, 0x1b, 0xb0, 0x11, 0xe6, 0x20, 0xb0, 0x20, 0xe6, 0x21, 0xb0, 0x04, 0xe6,
        0x21, 0xb0, 0x01, 0xe6, 0x21, 0xb0, 0xfe, 0xe6, 0x21, 0xb0, 0x30, 0xe6, 0x43, 0xb0, 0x38,
        0xe6, 0x40, 0xb0, 0x5d, 0xe6, 0x40, 0xfb, 0xf4, 0x66, 0xba, 0xf8, 0x03, 0xb0, 0x74, 0xee,
        0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    const HANDLER_PRINTS: u8 = b't';
    /// PIT input clock and the count programmed above.
    const PIT_HZ: u128 = 1_193_182;
    const PIT_COUNT: u128 = 0x5d38;
    const GUEST_DELAY_NS: u64 = (PIT_COUNT * 1_000_000_000).div_ceil(PIT_HZ) as u64;
    /// Upper bound on the instructions the guest retires before it halts (it is 33); virtual
    /// time counts one nanosecond each.
    const SETUP_INSTRUCTIONS: u64 = 40;

    fn cpu_waiting_for_the_pit(realtime: bool) -> Harness {
        let mut harness = Harness::new(&WAIT_FOR_PIT_IRQ);
        if realtime {
            harness.cpu.wall_clock = Some(crate::cpu::WallClock::new(crate::cpu::std_host_clock()));
        }
        harness
    }

    #[test]
    fn virtual_time_skips_the_whole_idle_wait_at_once() {
        let mut harness = cpu_waiting_for_the_pit(false);
        let started = Instant::now();
        assert_eq!(harness.cpu.run(), ExitReason::Reset);
        let skipped = harness.cpu.idle_skipped_ns;
        // Guest time advanced by the PIT delay less the few instructions retired meanwhile.
        assert!(skipped <= GUEST_DELAY_NS, "{skipped}");
        assert!(skipped + SETUP_INSTRUCTIONS >= GUEST_DELAY_NS, "{skipped}");
        assert!(harness.cpu.now_ns() >= GUEST_DELAY_NS);
        assert_eq!(harness.console.read(&mut [0u8; 4], None), 1);
        // Nothing slept on the host for 20 ms of guest time (a loose bound: it is not timing).
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn realtime_guest_time_covers_the_wait_and_the_host_really_sleeps_for_it() {
        let mut virtual_cpu = cpu_waiting_for_the_pit(false);
        virtual_cpu.cpu.run();
        let mut realtime_cpu = cpu_waiting_for_the_pit(true);
        let started = Instant::now();
        assert_eq!(realtime_cpu.cpu.run(), ExitReason::Reset);
        let slept = started.elapsed();
        // Guest time is the host's: it advanced at least by the wait the guest asked for, and
        // the wait was really slept.
        assert_eq!(realtime_cpu.cpu.idle_skipped_ns, 0);
        assert!(realtime_cpu.cpu.now_ns() >= virtual_cpu.cpu.idle_skipped_ns);
        assert!(
            slept >= Duration::from_nanos(virtual_cpu.cpu.idle_skipped_ns),
            "{slept:?}"
        );
    }

    #[test]
    fn realtime_machine_really_waits_for_the_timer_a_virtual_one_does_not() {
        let guest_delay = Duration::from_nanos(GUEST_DELAY_NS - SETUP_INSTRUCTIONS);
        let mut elapsed = Vec::new();
        for clock in [ClockMode::Virtual, ClockMode::Realtime] {
            let started = Instant::now();
            let run = run_machine(
                fake_bzimage(&WAIT_FOR_PIT_IRQ),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.clock(clock),
            );
            elapsed.push(started.elapsed());
            assert_eq!(run.exit, ExitReason::Reset, "{clock:?}");
            assert_eq!(run.console_output, [HANDLER_PRINTS], "{clock:?}");
        }
        assert!(
            elapsed[1] >= guest_delay,
            "realtime run took {:?}",
            elapsed[1]
        );
    }

    #[test]
    fn verify_runs_on_virtual_time_even_when_realtime_was_asked_for() {
        // Both CPUs must idle identically: the lockstep twin skips to the timer at once, so a
        // realtime main CPU (2 ms sleeps) would report a bogus pre-step divergence.
        for jit in [JitMode::Off, JitMode::Auto] {
            let run = run_machine(
                fake_bzimage(&WAIT_FOR_PIT_IRQ),
                jit,
                GENEROUS_LIMIT,
                |config| config.verify_jit(true).clock(ClockMode::Realtime),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, [HANDLER_PRINTS], "{jit:?}");
        }
    }

    #[test]
    fn verify_with_the_default_clock_also_idles_in_step() {
        let run = run_machine(
            fake_bzimage(&WAIT_FOR_PIT_IRQ),
            JitMode::Auto,
            GENEROUS_LIMIT,
            |config| config.verify_jit(true),
        );
        assert_eq!(run.exit, ExitReason::Reset);
        assert_eq!(run.console_output, [HANDLER_PRINTS]);
    }

    #[test]
    fn a_halt_with_interrupts_on_and_no_timer_idles_in_both_modes_until_stopped() {
        for clock in [ClockMode::Virtual, ClockMode::Realtime] {
            let run = run_machine(
                fake_bzimage(&PRINT_THEN_STI_HALT),
                JitMode::Off,
                WATCHDOG_LIMIT,
                |config| config.clock(clock),
            );
            assert_eq!(run.exit, ExitReason::Stopped, "{clock:?}");
            assert_eq!(run.console_output, [ALIVE_MARKER], "{clock:?}");
        }
    }

    #[test]
    fn virtual_clock_is_the_default_and_guest_time_is_the_instruction_count() {
        let mut harness = Harness::new(&[0x90, 0x90, 0x90, 0xf4]);
        harness.run_to_halt();
        assert!(harness.cpu.wall_clock.is_none());
        assert_eq!(harness.cpu.idle_skipped_ns, 0);
        assert_eq!(harness.cpu.now_ns(), 4 * crate::cpu::NS_PER_INSN);
    }
}

mod poll_hook {
    use super::*;

    // mov dx, 0x3fd; 1: in al, dx; test al, 1; jz 1b; mov dx, 0x3f8; in al, dx; out dx, al;
    // mov al, 0xfe; out 0x64, al; hlt
    const ECHO_ONE_INPUT_BYTE: [u8; 20] = [
        0x66, 0xba, 0xfd, 0x03, 0xec, 0xa8, 0x01, 0x74, 0xfb, 0x66, 0xba, 0xf8, 0x03, 0xec, 0xee,
        0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn the_hook_can_feed_the_guest_console_and_sees_every_tick() {
        for jit in auto_and_off() {
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = calls.clone();
            let run = run_machine(
                fake_bzimage(&ECHO_ONE_INPUT_BYTE),
                jit,
                GENEROUS_LIMIT,
                move |config| {
                    config.poll(move |console, _idle| {
                        if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                            console.write(b"Z");
                        }
                    })
                },
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"Z", "{jit:?}");
            assert!(calls.load(Ordering::SeqCst) >= 1, "{jit:?}");
        }
    }

    #[test]
    fn the_hook_receives_the_machines_own_console() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .reboot(false)
                .kernel(fake_bzimage(&SPIN))
                .memory_mib(64)
                .jit(JitMode::Off)
                .poll(|console, _| console.resize(132, 50)),
        )
        .unwrap();
        let console = machine.console();
        machine.start().unwrap();
        let deadline = Instant::now() + GENEROUS_LIMIT;
        while console.size() != (132, 50) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        machine.stop();
        assert_eq!(machine.wait(), ExitReason::Stopped);
        assert_eq!(console.size(), (132, 50));
    }

    #[test]
    fn host_input_written_before_start_reaches_the_guest() {
        let mut machine = new_machine(&ECHO_ONE_INPUT_BYTE);
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        machine
            .console()
            .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        machine.console().write(b"Q");
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        assert_eq!(output.lock().unwrap().as_slice(), b"Q");
    }

    #[test]
    fn guest_output_without_a_sink_is_readable_with_a_timeout() {
        let mut machine = new_machine(&HELLO_RESET);
        let console = machine.console();
        machine.start().unwrap();
        assert_eq!(machine.wait(), ExitReason::Reset);
        let mut buf = [0u8; 8];
        let count = console.read(&mut buf, Some(Duration::from_millis(1)));
        assert_eq!(&buf[..count], b"hi");
    }
}

mod reboot {
    use super::*;

    /// A guest that counts its boots in RAM it cannot keep: the counter lives at 0x400000, which
    /// only a cold boot (zeroed RAM) resets. It prints the counter as a digit, so a machine that
    /// reboots without clearing RAM prints 1, 2, 3 instead of 1, 1, 1.
    // mov ebx, 0x400000; inc byte [rbx]; mov al, [rbx]; add al, '0'; mov dx, 0x3f8; out dx, al;
    // mov al, 0xfe; out 0x64, al; hlt
    const COUNT_BOOTS_IN_RAM: [u8; 21] = [
        0xbb, 0x00, 0x00, 0x40, 0x00, 0xfe, 0x03, 0x8a, 0x03, 0x04, 0x30, 0x66, 0xba, 0xf8, 0x03,
        0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];
    /// Prints CMOS byte 0x40 as a digit, then stores 7 in it and resets through port 0xcf9. Only a
    /// device reset makes every boot print 0.
    // mov dx, 0x3f8; mov al, 0x40; out 0x70, al; in al, 0x71; add al, '0'; out dx, al;
    // mov al, 0x40; out 0x70, al; mov al, 7; out 0x71, al; mov al, 0x0e; mov dx, 0xcf9; out dx, al;
    // hlt
    const COUNT_BOOTS_IN_CMOS: [u8; 29] = [
        0x66, 0xba, 0xf8, 0x03, 0xb0, 0x40, 0xe6, 0x70, 0xe4, 0x71, 0x04, 0x30, 0xee, 0xb0, 0x40,
        0xe6, 0x70, 0xb0, 0x07, 0xe6, 0x71, 0xb0, 0x0e, 0x66, 0xba, 0xf9, 0x0c, 0xee, 0xf4,
    ];
    const BOOTS_TO_OBSERVE: usize = 4;

    /// Runs `kernel` with rebooting on until it has printed `bytes` bytes, then stops it.
    fn run_until_output(kernel: Vec<u8>, jit: JitMode, bytes: usize) -> (Vec<u8>, ExitReason) {
        let config = MachineConfig::new()
            .kernel(kernel)
            .memory_mib(MIN_MEMORY_MIB)
            .jit(jit)
            .reboot(true);
        let mut machine = EmulatedMachine::new(config).unwrap();
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = output.clone();
        let stop = machine.stop_handle();
        machine.console().on_output(move |piece| {
            let mut output = sink.lock().unwrap();
            output.extend_from_slice(piece);
            if output.len() >= bytes {
                stop.stop();
            }
        });
        // Backstop for a regression that stops printing.
        let backstop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(GENEROUS_LIMIT);
            backstop.stop();
        });
        machine.start().unwrap();
        let exit = machine.wait();
        let output = output.lock().unwrap().clone();
        (output, exit)
    }

    #[test]
    fn a_reset_reboots_into_zeroed_ram_like_a_cold_boot() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let (output, exit) =
                run_until_output(fake_bzimage(&COUNT_BOOTS_IN_RAM), jit, BOOTS_TO_OBSERVE);
            assert_eq!(exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(output, b"1".repeat(BOOTS_TO_OBSERVE), "{jit:?}");
        }
    }

    #[test]
    fn device_state_does_not_survive_a_reboot() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let (output, exit) =
                run_until_output(fake_bzimage(&COUNT_BOOTS_IN_CMOS), jit, BOOTS_TO_OBSERVE);
            assert_eq!(exit, ExitReason::Stopped, "{jit:?}");
            assert_eq!(output, b"0".repeat(BOOTS_TO_OBSERVE), "{jit:?}");
        }
    }

    #[test]
    fn the_console_sink_and_stop_flag_outlive_the_reboots() {
        // The sink is installed once and the stop handle taken before start; seeing every boot's
        // output and a clean `Stopped` shows both stayed attached.
        let (output, exit) = run_until_output(fake_bzimage(&HELLO_RESET), JitMode::Off, 2 * 2);
        assert_eq!(exit, ExitReason::Stopped);
        assert_eq!(output, b"hihi");
    }

    #[test]
    fn reboot_off_ends_the_run_with_reset_after_one_boot() {
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&COUNT_BOOTS_IN_RAM),
                jit,
                GENEROUS_LIMIT,
                |config| config.reboot(false),
            );
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"1", "{jit:?}");
        }
    }

    #[test]
    fn a_triple_fault_is_never_restarted() {
        for reboot in [true, false] {
            let run = run_machine(
                fake_bzimage(&NULL_IDT_THEN_INT3),
                JitMode::Off,
                GENEROUS_LIMIT,
                |config| config.reboot(reboot),
            );
            assert_eq!(run.exit, ExitReason::TripleFault, "reboot {reboot}");
        }
    }

    #[test]
    fn verification_never_restarts_the_machine() {
        for jit in auto_and_off() {
            let _turn = crate::jit::serialize_jit_tests();
            let run = run_machine(fake_bzimage(&HELLO_RESET), jit, GENEROUS_LIMIT, |config| {
                config.verify_jit(true).reboot(true)
            });
            assert_eq!(run.exit, ExitReason::Reset, "{jit:?}");
            assert_eq!(run.console_output, b"hi", "{jit:?}");
        }
    }

    #[test]
    fn the_reboot_image_shares_the_embedders_net_queues() {
        let machine = EmulatedMachine::new(
            MachineConfig::new()
                .kernel(fake_bzimage(&HELLO_RESET))
                .memory_mib(MIN_MEMORY_MIB)
                .net([2, 0, 0, 0, 0, 1]),
        )
        .unwrap();
        let held_by_embedder = machine.net().unwrap();
        let image = machine.vcpu.as_ref().unwrap().reboot.as_ref().unwrap();
        let (mac, rebuilt) = &image.nets[0];
        assert_eq!(*mac, [2, 0, 0, 0, 0, 1]);
        assert!(held_by_embedder.send_frame(b"frame"));
        assert!(
            rebuilt.has_pending(),
            "the rebuilt device sees the same queue"
        );
    }

    #[test]
    fn reboot_is_the_default_and_off_or_verify_keeps_no_image() {
        let image_kept = |config: MachineConfig| {
            let machine = EmulatedMachine::new(config.kernel(fake_bzimage(&SPIN))).unwrap();
            machine.vcpu.as_ref().unwrap().reboot.is_some()
        };
        let base = || MachineConfig::new().memory_mib(MIN_MEMORY_MIB);
        assert!(image_kept(base()));
        assert!(!image_kept(base().reboot(false)));
        assert!(!image_kept(base().verify_jit(true)));
    }
}

mod firmware {
    use super::*;

    /// Bytes from the top of a firmware image to its reset vector (F000:FFF0).
    const RESET_VECTOR_FROM_END: usize = 16;
    // mov dx, 0x3f8; mov al, 'h'; out dx, al; mov al, 'i'; out dx, al; mov al, 0xfe; out 0x64, al;
    // hlt, as 16-bit code: it fits the 16 bytes below the end of the image.
    const REAL_MODE_HELLO_RESET: [u8; 14] = [
        0xba, 0xf8, 0x03, 0xb0, 0x68, 0xee, 0xb0, 0x69, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    fn image_with_reset_code(code: &[u8], length: usize) -> Vec<u8> {
        let mut image = vec![0xf4; length];
        let at = length - RESET_VECTOR_FROM_END;
        image[at..at + code.len()].copy_from_slice(code);
        image
    }

    #[test]
    fn a_firmware_image_starts_in_real_mode_at_the_reset_vector() {
        for jit in auto_and_off() {
            let mut machine = EmulatedMachine::new(
                MachineConfig::new()
                    .reboot(false)
                    .jit(jit)
                    .firmware(image_with_reset_code(&REAL_MODE_HELLO_RESET, 0x1_0000))
                    .memory_mib(MIN_MEMORY_MIB),
            )
            .expect("valid configuration");
            let console = machine.console();
            machine.start().unwrap();
            assert_eq!(machine.wait(), ExitReason::Reset);
            let mut output = [0u8; 8];
            let length = console.read(&mut output, Some(Duration::from_millis(100)));
            assert_eq!(&output[..length], b"hi");
        }
    }

    #[test]
    fn an_empty_or_oversized_firmware_image_is_refused() {
        for length in [0, 0x6_0001] {
            let result = EmulatedMachine::new(
                MachineConfig::new()
                    .firmware(vec![0u8; length])
                    .memory_mib(MIN_MEMORY_MIB),
            );
            expect_config_error(result);
        }
    }

    #[test]
    fn a_guest_reset_with_a_firmware_image_ends_the_run() {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .firmware(image_with_reset_code(&REAL_MODE_HELLO_RESET, 0x2_0000))
                .memory_mib(MIN_MEMORY_MIB),
        )
        .expect("valid configuration");
        machine.start().unwrap();
        assert_eq!(
            machine.wait(),
            ExitReason::Reset,
            "reboot is on, but a firmware image is not rebooted"
        );
    }
}

mod boot_disk {
    use super::*;

    const DISK_BYTES: usize = MIB;
    /// mov ax,0x0e41; int 0x10; cli; hlt
    const PRINT_A_AND_POWER_OFF: [u8; 7] = [0xb8, 0x41, 0x0e, 0xcd, 0x10, 0xfa, 0xf4];
    /// sti; mov ax,[0x46c]; l: cmp ax,[0x46c]; je l; mov dx,0x3f8; mov al,'T'; out dx,al; cli; hlt:
    /// waits for the BIOS tick counter to move, which takes the PIT's IRQ0 through the PIC to the
    /// BIOS's INT 08h.
    const WAIT_FOR_A_TICK: [u8; 18] = [
        0xfb, 0xa1, 0x6c, 0x04, 0x3b, 0x06, 0x6c, 0x04, 0x74, 0xfa, 0xba, 0xf8, 0x03, 0xb0, 0x54,
        0xee, 0xfa, 0xf4,
    ];

    fn disk_with_boot_sector(code: &[u8]) -> Disk {
        let mut image = vec![0u8; DISK_BYTES];
        image[..code.len()].copy_from_slice(code);
        image[510] = 0x55;
        image[511] = 0xaa;
        Disk::memory(image)
    }

    fn boot(disk: Disk) -> (ExitReason, Vec<u8>) {
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .boot_disk(true)
                .disk(disk)
                .memory_mib(MIN_MEMORY_MIB),
        )
        .expect("valid configuration");
        let console = machine.console();
        machine.start().unwrap();
        let exit = machine.wait();
        let mut output = [0u8; 64];
        let length = console.read(&mut output, Some(Duration::from_millis(100)));
        (exit, output[..length].to_vec())
    }

    #[test]
    fn the_first_sector_runs_with_bios_services_and_power_off_ends_the_run() {
        let (exit, output) = boot(disk_with_boot_sector(&PRINT_A_AND_POWER_OFF));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"A");
    }

    #[test]
    fn the_pit_ticks_the_bios_counter_through_the_pic() {
        let (exit, output) = boot(disk_with_boot_sector(&WAIT_FOR_A_TICK));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"T");
    }

    #[test]
    fn a_blank_disk_has_nothing_to_boot() {
        let (exit, output) = boot(Disk::memory(vec![0u8; DISK_BYTES]));
        assert_eq!(exit, ExitReason::Shutdown);
        assert_eq!(output, b"No bootable device.\r\n");
    }

    #[test]
    fn boot_disk_needs_a_disk_and_excludes_a_kernel_a_firmware_and_verify() {
        let base = || {
            MachineConfig::new()
                .boot_disk(true)
                .memory_mib(MIN_MEMORY_MIB)
        };
        expect_config_error(EmulatedMachine::new(base()));
        expect_config_error(EmulatedMachine::new(
            base()
                .disk(disk_with_boot_sector(&[0xf4]))
                .kernel(fake_bzimage(&[0xf4])),
        ));
        expect_config_error(EmulatedMachine::new(
            base()
                .disk(disk_with_boot_sector(&[0xf4]))
                .firmware(vec![0xf4; 16]),
        ));
        expect_config_error(EmulatedMachine::new(
            base().disk(disk_with_boot_sector(&[0xf4])).verify_jit(true),
        ));
    }

    #[test]
    fn a_guest_reset_ends_a_bios_boot_instead_of_restarting_it() {
        // mov al,0xfe; out 0x64,al; hlt
        let (exit, _) = boot(disk_with_boot_sector(&[0xb0, 0xfe, 0xe6, 0x64, 0xf4]));
        assert_eq!(exit, ExitReason::Reset);
    }
}

mod uefi_firmware {
    use super::*;
    use crate::boot::uefi::{FIRMWARE_BASE, HANDOFF_ADDR};
    use crate::kv::{KeyValueError, KeyValueStore, MemoryKeyValueStore};

    /// Prints the first handoff byte ('U'), stores "k" = "v" through the key-value device, reads
    /// it back, prints the value and asks for a reset.
    const STORE_AND_PRINT: [u8; 130] = [
        0x44, 0x0f, 0xb7, 0x47, 0x4a, 0x66, 0xba, 0xf8, 0x03, 0x0f, 0xb6, 0x07, 0xee, 0x44, 0x89,
        0xc2, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08, 0xb8, 0x01, 0x00, 0x00,
        0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x6b, 0x00, 0x00, 0x00, 0xef, 0x44, 0x89, 0xc2,
        0xb8, 0x02, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08, 0xb8, 0x01, 0x00, 0x00, 0x00,
        0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x76, 0x00, 0x00, 0x00, 0xef, 0x44, 0x89, 0xc2, 0xb8,
        0x04, 0x00, 0x00, 0x00, 0xef, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x08,
        0xb8, 0x01, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xb8, 0x6b, 0x00, 0x00, 0x00,
        0xef, 0x44, 0x89, 0xc2, 0xb8, 0x03, 0x00, 0x00, 0x00, 0xef, 0x41, 0x8d, 0x50, 0x04, 0xed,
        0x66, 0xba, 0xf8, 0x03, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    /// A store the test can look into after the machine owns it.
    #[derive(Clone, Default)]
    struct SharedStore(Arc<Mutex<MemoryKeyValueStore>>);

    impl KeyValueStore for SharedStore {
        fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
            self.0.lock().unwrap().get(key)
        }
        fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), KeyValueError> {
            self.0.lock().unwrap().set(key, value)
        }
        fn delete(&mut self, key: &[u8]) -> Result<(), KeyValueError> {
            self.0.lock().unwrap().delete(key)
        }
        fn list(&mut self, prefix: &[u8]) -> Vec<Vec<u8>> {
            self.0.lock().unwrap().list(prefix)
        }
    }

    /// The image header the library expects, then `code` at `ENTRY_OFFSET`.
    const ENTRY_OFFSET: usize = 0x20;

    fn image(code: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; ENTRY_OFFSET];
        bytes[..8].copy_from_slice(b"UEMUFW64");
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(ENTRY_OFFSET as u32).to_le_bytes());
        bytes.extend_from_slice(code);
        let footprint = bytes.len() as u64 + 0x1000;
        bytes[16..24].copy_from_slice(&footprint.to_le_bytes());
        bytes
    }

    fn run(
        config: impl FnOnce(MachineConfig) -> MachineConfig,
        limit: Duration,
    ) -> crate::testkit::MachineRun {
        run_machine(Vec::new(), JitMode::Off, limit, config)
    }

    #[test]
    fn the_firmware_runs_with_the_handoff_and_reaches_the_store() {
        let store = SharedStore::default();
        let shared = store.clone();
        let result = run(
            |config| {
                config
                    .uefi_firmware(image(&STORE_AND_PRINT))
                    .key_value_store(shared)
            },
            GENEROUS_LIMIT,
        );
        assert_eq!(result.text(), "Uv");
        assert!(
            matches!(result.exit, ExitReason::Reset),
            "{:?}",
            result.exit
        );
        let mut store = store;
        assert_eq!(store.get(b"k"), Some(b"v".to_vec()));
    }

    #[test]
    fn a_reset_runs_the_firmware_again_and_the_store_survives() {
        let store = SharedStore::default();
        let result = run(
            |config| {
                config
                    .reboot(true)
                    .uefi_firmware(image(&STORE_AND_PRINT))
                    .key_value_store(store)
            },
            WATCHDOG_LIMIT,
        );
        assert!(result.text().starts_with("UvUv"), "{:?}", result.text());
    }

    #[test]
    fn the_handoff_and_image_are_in_guest_memory_at_the_documented_places() {
        assert_eq!(FIRMWARE_BASE, 0x10_0000);
        assert_eq!(HANDOFF_ADDR, 0x7000);
    }

    #[test]
    fn uefi_firmware_excludes_the_other_boot_paths() {
        let firmware = image(&[0xf4]);
        // A kernel is the firmware's to boot now, but only one with an EFI stub.
        let with_plain_kernel = MachineConfig::new()
            .kernel(fake_bzimage(&SPIN))
            .uefi_firmware(firmware.clone());
        assert!(
            expect_config_error(EmulatedMachine::new(with_plain_kernel)).contains("firmware none")
        );
        let with_bios = MachineConfig::new()
            .firmware(vec![0xf4])
            .uefi_firmware(firmware.clone());
        assert!(expect_config_error(EmulatedMachine::new(with_bios)).contains("uefi_firmware"));
        let bad = MachineConfig::new().uefi_firmware(vec![0; 64]);
        assert!(expect_config_error(EmulatedMachine::new(bad)).contains("UEFI firmware image"));
    }
}

mod cpus {
    use super::*;

    const AP_CODE_ADDRESS: u64 = 0x8000;
    // mov ecx, 0x1b; mov eax, 0xfee00d00; xor edx, edx; wrmsr; mov ecx, 0x830: x2APIC on, ecx is
    // the ICR for the writes that follow
    const ENABLE_X2APIC: [u8; 19] = [
        0xb9, 0x1b, 0x00, 0x00, 0x00, 0xb8, 0x00, 0x0d, 0xe0, 0xfe, 0x31, 0xd2, 0x0f, 0x30, 0xb9,
        0x30, 0x08, 0x00, 0x00,
    ];
    // mov byte [0x1000], 0x42; hlt, in real mode; padded with hlt to whole dwords
    const AP_CODE: [u8; 8] = [0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4, 0xf4, 0xf4];
    // wait: pause; cmp byte [0x1000], 0x42; jne wait
    const WAIT_FOR_AP: [u8; 12] = [
        0xf3, 0x90, 0x80, 0x3c, 0x25, 0x00, 0x10, 0x00, 0x00, 0x42, 0x75, 0xf4,
    ];
    // mov dx, 0x3f8; mov al, 'o'; out dx, al; mov al, 'k'; out dx, al; mov al, 0xfe; out 0x64, al; hlt
    const SAY_OK_AND_RESET: [u8; 17] = [
        0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6f, 0xee, 0xb0, 0x6b, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
        0xf4, 0xf4,
    ];
    const ICR_INIT_LOW: u32 = 0x4500;
    const ICR_STARTUP_LOW: u32 = 0x4608;

    fn store_dword(address: u32, value: u32) -> Vec<u8> {
        let mut code = vec![0xc7, 0x04, 0x25];
        code.extend_from_slice(&address.to_le_bytes());
        code.extend_from_slice(&value.to_le_bytes());
        code
    }

    fn send_icr(destination: u32, low: u32) -> Vec<u8> {
        let mut code = vec![0xba];
        code.extend_from_slice(&destination.to_le_bytes());
        code.push(0xb8);
        code.extend_from_slice(&low.to_le_bytes());
        code.extend_from_slice(&[0x0f, 0x30]);
        code
    }

    /// A kernel whose first processor starts the second at `AP_CODE`, waits for its note and
    /// says "ok".
    fn start_the_second_cpu() -> Vec<u8> {
        let mut code = ENABLE_X2APIC.to_vec();
        for (index, chunk) in AP_CODE.chunks(4).enumerate() {
            code.extend(store_dword(
                AP_CODE_ADDRESS as u32 + 4 * index as u32,
                u32::from_le_bytes(chunk.try_into().unwrap()),
            ));
        }
        code.extend(send_icr(1, ICR_INIT_LOW));
        code.extend(send_icr(1, ICR_STARTUP_LOW));
        code.extend(WAIT_FOR_AP);
        code.extend(SAY_OK_AND_RESET);
        code
    }

    #[test]
    fn the_count_must_be_between_1_and_64() {
        for cpus in [0, 65] {
            let message = expect_config_error(EmulatedMachine::new(
                MachineConfig::new()
                    .kernel(fake_bzimage(&[0xf4]))
                    .memory_mib(MIN_MEMORY_MIB)
                    .cpus(cpus),
            ));
            assert!(message.contains("cpus"), "{message}");
        }
        for cpus in [1, 64] {
            assert!(
                EmulatedMachine::new(
                    MachineConfig::new()
                        .kernel(fake_bzimage(&[0xf4]))
                        .memory_mib(MIN_MEMORY_MIB)
                        .cpus(cpus),
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn several_cpus_need_acpi_and_exclude_verification_and_firmware() {
        let base = || {
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(MIN_MEMORY_MIB)
                .cpus(2)
        };
        expect_config_error(EmulatedMachine::new(base().acpi(false)));
        expect_config_error(EmulatedMachine::new(base().verify_jit(true)));
        expect_config_error(EmulatedMachine::new(
            MachineConfig::new()
                .firmware(vec![0xf4; 16])
                .memory_mib(MIN_MEMORY_MIB)
                .cpus(2),
        ));
    }

    #[test]
    fn a_second_cpu_started_by_the_first_runs_on_a_full_machine() {
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&start_the_second_cpu()),
                jit,
                GENEROUS_LIMIT,
                |config| config.cpus(2),
            );
            assert_eq!(run.text(), "ok", "{:?}", run.exit);
            assert_eq!(run.exit, ExitReason::Reset);
        }
    }

    fn on_threads(config: MachineConfig) -> MachineConfig {
        config.clock(ClockMode::Realtime).smp(SmpMode::Threads)
    }

    #[test]
    fn a_second_cpu_started_by_the_first_runs_on_its_own_thread() {
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&start_the_second_cpu()),
                jit,
                GENEROUS_LIMIT,
                |config| on_threads(config.cpus(2)),
            );
            assert_eq!(run.text(), "ok", "{:?}", run.exit);
            assert_eq!(run.exit, ExitReason::Reset);
        }
    }

    #[test]
    fn the_same_kernel_waits_forever_for_a_cpu_that_does_not_exist() {
        // With one CPU the INIT/SIPI go nowhere, so the note never appears.
        let run = run_machine(
            fake_bzimage(&start_the_second_cpu()),
            JitMode::Off,
            WATCHDOG_LIMIT,
            |config| config,
        );
        assert_eq!(run.text(), "");
        assert_eq!(run.exit, ExitReason::Stopped);
    }

    #[test]
    fn a_reboot_joins_every_thread_and_brings_every_cpu_back_to_power_on() {
        // The same, with a thread per vCPU: the guest resets, every thread must end, and the
        // next boot builds new ones.
        let reboots = Arc::new(AtomicUsize::new(0));
        let counter = reboots.clone();
        let run = run_machine(
            fake_bzimage(&start_the_second_cpu()),
            JitMode::Off,
            Duration::from_secs(1),
            |config| {
                on_threads(config.cpus(2).reboot(true)).log(move |line| {
                    if line.contains("rebooting") {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            },
        );
        assert!(
            run.text().starts_with("okok"),
            "{:?} {:?}",
            run.text(),
            run.exit
        );
        assert!(reboots.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn the_embedders_thread_spawner_starts_one_thread_per_vcpu() {
        // The provider a freestanding host supplies is what starts the vCPU threads; hosted
        // builds accept one too.
        let spawned = Arc::new(AtomicUsize::new(0));
        let counter = spawned.clone();
        let run = run_machine(
            fake_bzimage(&start_the_second_cpu()),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| {
                on_threads(config.cpus(2)).thread_spawner(move |body| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    std::thread::spawn(body);
                })
            },
        );
        assert_eq!(run.text(), "ok", "{:?}", run.exit);
        assert_eq!(spawned.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_threaded_machine_stops_when_asked_even_with_every_vcpu_spinning() {
        // jmp $ on the bootstrap processor; the second CPU stays parked.
        for jit in auto_and_off() {
            let run = run_machine(
                fake_bzimage(&[0xeb, 0xfe]),
                jit,
                Duration::from_millis(300),
                |config| on_threads(config.cpus(3)),
            );
            assert_eq!(run.exit, ExitReason::Stopped);
        }
    }

    #[test]
    fn a_threaded_machine_ends_when_every_cpu_halts_with_interrupts_off() {
        let run = run_machine(
            fake_bzimage(&[0xfa, 0xf4]),
            JitMode::Off,
            GENEROUS_LIMIT,
            |config| on_threads(config.cpus(4)),
        );
        assert_eq!(run.exit, ExitReason::Shutdown);
    }

    #[test]
    fn threads_need_the_realtime_clock_and_the_other_modes_do_not() {
        let virtual_clock = |config: MachineConfig| config.cpus(2).clock(ClockMode::Virtual);
        let base = || {
            MachineConfig::new()
                .kernel(fake_bzimage(&[0xf4]))
                .memory_mib(MIN_MEMORY_MIB)
        };
        expect_config_error(EmulatedMachine::new(
            virtual_clock(base()).smp(SmpMode::Threads),
        ));
        for mode in [SmpMode::Auto, SmpMode::Timeshare] {
            assert!(EmulatedMachine::new(virtual_clock(base()).smp(mode)).is_ok());
        }
    }

    #[test]
    fn a_reboot_brings_every_cpu_back_to_power_on() {
        // The guest resets after the second CPU ran; a restart must park it again so the same
        // kernel can start it the same way (and say "ok" twice).
        let reboots = Arc::new(AtomicUsize::new(0));
        let counter = reboots.clone();
        let run = run_machine(
            fake_bzimage(&start_the_second_cpu()),
            JitMode::Off,
            Duration::from_secs(1),
            |config| {
                config.cpus(2).reboot(true).log(move |line| {
                    if line.contains("rebooting") {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            },
        );
        assert!(
            run.text().starts_with("okok"),
            "{:?} {:?}",
            run.text(),
            run.exit
        );
        assert!(reboots.load(Ordering::SeqCst) >= 1);
    }
}

mod uefi_kernel {
    use super::*;

    const KERNEL_BYTES: usize = 0x80;
    const PE_HEADER_OFFSET: usize = 0x40;
    const FIRMWARE_ENTRY_OFFSET: usize = 0x40;
    // mov al, [rdi + 0x310] (the low byte of the kernel's size in the handoff page);
    // mov dx, 0x3f8; out dx, al; mov dx, 0xcf9; mov al, 6; out dx, al; hlt
    const REPORT_KERNEL_SIZE_THEN_RESET: [u8; 19] = [
        0x8a, 0x87, 0x10, 0x03, 0x00, 0x00, 0x66, 0xba, 0xf8, 0x03, 0xee, 0x66, 0xba, 0xf9, 0x0c,
        0xb0, 0x06, 0xee, 0xf4,
    ];

    /// The smallest flat image the library accepts, running `code` at its entry.
    fn firmware_image(code: &[u8]) -> Vec<u8> {
        let mut image = vec![0u8; FIRMWARE_ENTRY_OFFSET + code.len()];
        image[..8].copy_from_slice(b"UEMUFW64");
        image[8..12].copy_from_slice(&1u32.to_le_bytes());
        image[12..16].copy_from_slice(&(FIRMWARE_ENTRY_OFFSET as u32).to_le_bytes());
        image[16..24].copy_from_slice(&0x2000u64.to_le_bytes());
        image[FIRMWARE_ENTRY_OFFSET..].copy_from_slice(code);
        image
    }

    fn efi_kernel() -> Vec<u8> {
        let mut kernel = vec![0u8; KERNEL_BYTES];
        kernel[..2].copy_from_slice(b"MZ");
        kernel[0x3c..0x40].copy_from_slice(&(PE_HEADER_OFFSET as u32).to_le_bytes());
        kernel[PE_HEADER_OFFSET..PE_HEADER_OFFSET + 4].copy_from_slice(b"PE\0\0");
        kernel
    }

    #[test]
    fn a_kernel_with_an_efi_header_is_accepted_together_with_the_firmware() {
        let config = MachineConfig::new()
            .uefi_firmware(firmware_image(&REPORT_KERNEL_SIZE_THEN_RESET))
            .kernel(efi_kernel())
            .initrd(vec![1, 2, 3])
            .memory_mib(64);
        assert!(EmulatedMachine::new(config).is_ok());
    }

    #[test]
    fn a_kernel_without_an_efi_stub_is_refused_with_the_firmware_none_hint() {
        let config = MachineConfig::new()
            .uefi_firmware(firmware_image(&REPORT_KERNEL_SIZE_THEN_RESET))
            .kernel(FakeKernel::new(&HELLO_RESET).build())
            .memory_mib(64);
        let error = EmulatedMachine::new(config).err().expect("must be refused");
        assert!(error.to_string().contains("firmware none"), "{error}");
    }

    #[test]
    fn a_firmware_image_still_cannot_be_combined_with_boot_disk() {
        let config = MachineConfig::new()
            .uefi_firmware(firmware_image(&REPORT_KERNEL_SIZE_THEN_RESET))
            .boot_disk(true)
            .memory_mib(64);
        assert!(EmulatedMachine::new(config).is_err());
    }

    // Writes the kernel command line from the handoff page to the 16550, then the low byte of
    // the hypervisor's size, then resets: mov rsi, [rdi + 0x328]; mov rcx, [rdi + 0x330];
    // mov dx, 0x3f8; rep outsb; mov al, [rdi + 0x360]; out dx, al; mov dx, 0xcf9; mov al, 6;
    // out dx, al; hlt
    const REPORT_CMDLINE_AND_HYPERVISOR_SIZE_THEN_RESET: [u8; 35] = [
        0x48, 0x8b, 0xb7, 0x28, 0x03, 0x00, 0x00, 0x48, 0x8b, 0x8f, 0x30, 0x03, 0x00, 0x00, 0x66,
        0xba, 0xf8, 0x03, 0xf3, 0x6e, 0x8a, 0x87, 0x60, 0x03, 0x00, 0x00, 0xee, 0x66, 0xba, 0xf9,
        0x0c, 0xb0, 0x06, 0xee, 0xf4,
    ];

    /// `data` in a gzip file of stored blocks, which is all a test needs to exercise the
    /// decompression path.
    fn gzip_stored(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        let chunks: Vec<&[u8]> = data.chunks(u16::MAX as usize).collect();
        for (index, chunk) in chunks.iter().enumerate() {
            out.push((index + 1 == chunks.len()) as u8);
            let length = chunk.len() as u16;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&(!length).to_le_bytes());
            out.extend_from_slice(chunk);
        }
        out.extend_from_slice(&crate::kv::crc32(data).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out
    }

    #[test]
    fn a_hypervisor_takes_a_dom0_kernel_without_an_efi_stub_and_needs_the_firmware() {
        let hypervisor = crate::boot::hypervisor::tests::multiboot2_image(0x98);
        let base = || {
            MachineConfig::new()
                .kernel(FakeKernel::new(&HELLO_RESET).build())
                .hypervisor(hypervisor.clone())
                .memory_mib(64)
        };
        let firmware = firmware_image(&REPORT_KERNEL_SIZE_THEN_RESET);
        assert!(EmulatedMachine::new(base().uefi_firmware(firmware.clone())).is_ok());
        let error = EmulatedMachine::new(base()).err().expect("no firmware");
        assert!(error.to_string().contains("uefi_firmware"), "{error}");
        let no_kernel = MachineConfig::new()
            .uefi_firmware(firmware.clone())
            .hypervisor(hypervisor.clone())
            .memory_mib(64);
        let error = EmulatedMachine::new(no_kernel).err().expect("no kernel");
        assert!(error.to_string().contains("dom0"), "{error}");
        let not_xen = base().uefi_firmware(firmware).hypervisor(vec![0u8; 64]);
        assert!(EmulatedMachine::new(not_xen).is_err());
    }

    #[test]
    fn a_gzipped_hypervisor_is_placed_decompressed_with_xens_dom0_defaults_and_again_after_a_reset()
    {
        let hypervisor = crate::boot::hypervisor::tests::multiboot2_image(0x98);
        let decompressed_low_byte = hypervisor.len() as u8;
        let gzipped = gzip_stored(&hypervisor);
        assert_ne!(gzipped.len() as u8, decompressed_low_byte);
        let run = run_machine(
            FakeKernel::new(&HELLO_RESET).build(),
            JitMode::Off,
            Duration::from_millis(500),
            |config| {
                config
                    .uefi_firmware(firmware_image(
                        &REPORT_CMDLINE_AND_HYPERVISOR_SIZE_THEN_RESET,
                    ))
                    .hypervisor(gzipped)
                    .reboot(true)
            },
        );
        // dom0's console is Xen's hvc0, and the 16550 is Xen's, so early output goes through Xen.
        let report = [
            b"console=hvc0 earlyprintk=xen panic=-1".as_slice(),
            &[decompressed_low_byte],
        ]
        .concat();
        let reports = run
            .console_output
            .windows(report.len())
            .filter(|window| *window == report)
            .count();
        assert!(
            reports >= 2,
            "reported {reports} times: {:?}",
            String::from_utf8_lossy(&run.console_output)
        );
    }

    #[test]
    fn a_guest_reset_places_the_payload_again() {
        let run = run_machine(
            efi_kernel(),
            JitMode::Off,
            Duration::from_millis(500),
            |config| {
                config
                    .uefi_firmware(firmware_image(&REPORT_KERNEL_SIZE_THEN_RESET))
                    .reboot(true)
            },
        );
        let reported = run
            .console_output
            .iter()
            .filter(|byte| **byte == KERNEL_BYTES as u8)
            .count();
        assert!(
            reported >= 2,
            "the kernel size was reported {reported} times"
        );
    }
}

mod tpm {
    use super::*;

    /// mov rax, 0xfed40030 (the CRB's TPM_CRB_INTF_ID); mov eax, [rax]; out 0x3f8, al.
    const PRINT_INTERFACE_ID: [u8; 17] = [
        0x48, 0xb8, 0x30, 0x00, 0xd4, 0xfe, 0x00, 0x00, 0x00, 0x00, 0x8b, 0x00, 0x66, 0xba, 0xf8,
        0x03, 0xee,
    ];
    /// CRB interface type 1, version 1.
    const CRB_INTERFACE_LOW_BYTE: u8 = 0x11;

    fn probe(configure: impl FnOnce(MachineConfig) -> MachineConfig) -> crate::testkit::MachineRun {
        let code = [&PRINT_INTERFACE_ID[..], &RESET_AND_HALT].concat();
        run_machine(
            fake_bzimage(&code),
            JitMode::Auto,
            GENEROUS_LIMIT,
            configure,
        )
    }

    #[test]
    fn the_crb_page_answers_only_on_a_machine_with_a_tpm() {
        let with = probe(|config| config.tpm(true));
        assert_eq!(with.exit, ExitReason::Reset);
        assert_eq!(with.console_output, [CRB_INTERFACE_LOW_BYTE]);
        let without = probe(|config| config);
        assert_eq!(without.exit, ExitReason::Reset);
        assert_ne!(without.console_output, [CRB_INTERFACE_LOW_BYTE]);
    }

    #[test]
    fn verify_builds_the_twin_a_tpm_of_its_own() {
        let run = probe(|config| config.tpm(true).verify_jit(true));
        assert_eq!(run.exit, ExitReason::Reset);
        assert_eq!(run.console_output, [CRB_INTERFACE_LOW_BYTE]);
    }

    #[test]
    fn secure_boot_needs_the_uefi_firmware() {
        let config = MachineConfig::new()
            .kernel(fake_bzimage(&RESET_AND_HALT))
            .secure_boot(true);
        let error = EmulatedMachine::new(config).err().expect("refused");
        assert!(error.to_string().contains("uefi_firmware"), "{error}");
    }
}
