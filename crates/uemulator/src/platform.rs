#[cfg(freestanding)]
use crate::prelude::*;

use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;
use crate::dev::guestmem::GuestMem;
use crate::dev::{DevConfig, Devices};
use crate::machine::Error;

/// The default `EmulatedMachine` builds for its default virtio console (`hvc0`); early boot text
/// still comes from the 16550 through `earlyprintk`. A `Platform` has no virtio devices, so a host
/// driving one needs `DEFAULT_SERIAL_CMDLINE`.
pub const DEFAULT_CMDLINE: &str =
    "console=hvc0 earlyprintk=serial,ttyS0 mitigations=off acpi=off panic=-1";
/// `DEFAULT_CMDLINE` with the console on the UART: what `EmulatedMachine` builds for
/// `ConsoleKind::Serial` and what a `Platform` host must use. Differs from `DEFAULT_CMDLINE` only
/// in the `console=` word (pinned by a test).
pub const DEFAULT_SERIAL_CMDLINE: &str =
    "console=ttyS0 earlyprintk=serial,ttyS0 mitigations=off acpi=off panic=-1";

/// Lays out a bzImage, an optional initramfs, the command line, boot parameters, a GDT and
/// identity-mapped page tables in `ram` (guest physical address 0 is `ram[0]`), and returns the
/// 64-bit entry state. For hosts that start the guest themselves; `EmulatedMachine` calls the
/// same loader. The guest is told about at most the first 3 GiB of `ram`: a host's flat buffer
/// has no PCI hole in it, and `EmulatedMachine` is the one that lays RAM out around the hole.
pub fn load_linux(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
) -> Result<BootState, Error> {
    let ram_bytes = ram.len().min(crate::ram::PCI_HOLE_START as usize);
    boot::load(&mut ram[..ram_bytes], kernel, initrd, cmdline)
}

/// The legacy PC devices (UART, PIC, PIT, CMOS, reset ports) without a CPU, for hosts that run the
/// guest on hardware virtualization and forward port I/O and interrupts to it. Times are
/// nanoseconds on any monotonic clock, as long as it is the same clock for every call.
pub struct Platform {
    dev: Devices,
    /// Empty `page_has_code` map for `GuestMem`: there is no translated code here to invalidate
    /// when a device writes guest RAM by DMA.
    no_code_pages: Vec<u8>,
}

impl Platform {
    pub fn new(console: ConsoleHandle) -> Self {
        Self {
            dev: Devices::new(console, DevConfig::default()),
            no_code_pages: Vec::new(),
        }
    }

    pub fn set_boot_time(&mut self, unix: u64) {
        self.dev.set_boot_time(unix);
    }

    pub fn io_in(&mut self, port: u16, size: u8, now_ns: u64) -> u32 {
        self.dev.io_in(port, size, now_ns)
    }

    pub fn io_out(&mut self, port: u16, size: u8, value: u32, now_ns: u64) {
        self.dev.io_out(port, size, value, now_ns);
    }

    /// Runs due device work. `Devices::tick` takes guest memory for DMA, though nothing in a
    /// `Platform` (no disk, net or virtio console) writes it.
    pub fn tick(&mut self, now_ns: u64, ram: &mut [u8]) {
        let mut mem = GuestMem::new(ram, &self.no_code_pages);
        self.dev.tick(now_ns, &mut mem);
    }

    pub fn irq_pending(&self) -> bool {
        self.dev.pic.has_pending()
    }

    /// Takes the highest-priority pending interrupt and returns its vector.
    pub fn irq_ack(&mut self) -> Option<u8> {
        self.dev.pic.ack()
    }

    /// Earliest time a device needs `tick`, e.g. the next PIT edge; `None` if nothing is scheduled.
    pub fn next_deadline(&self, now_ns: u64) -> Option<u64> {
        self.dev.next_deadline(now_ns)
    }

    /// Set once the guest wrote a reset command (ports 0x64, 0x92 or 0xcf9); stays set.
    pub fn reset_requested(&self) -> bool {
        self.dev.reset_requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{FakeKernel, fake_bzimage};
    use core::time::Duration;

    const UART_DATA: u16 = 0x3f8;
    const UART_IER: u16 = 0x3f9;
    const UART_LSR: u16 = 0x3fd;
    const LSR_DATA_READY: u32 = 0x01;
    const LSR_TRANSMITTER_IDLE: u32 = 0x60;
    const IER_RX_AVAILABLE: u32 = 0x01;

    const PIC_MASTER_COMMAND: u16 = 0x20;
    const PIC_MASTER_DATA: u16 = 0x21;
    const ICW1_INIT_WITH_ICW4: u32 = 0x11;
    const ICW3_CASCADE_ON_IRQ2: u32 = 0x04;
    const ICW4_8086: u32 = 0x01;
    /// Where the tests remap IRQ 0.
    const IRQ_BASE_VECTOR: u32 = 0x20;
    const IRQ_PIT: u8 = 0;
    const IRQ_UART: u8 = 4;
    const UNMASK_ALL: u32 = 0;

    const PIT_CHANNEL0: u16 = 0x40;
    const PIT_COMMAND: u16 = 0x43;
    /// Channel 0, low then high byte, mode 0 (interrupt on terminal count), binary.
    const PIT_ONE_SHOT: u32 = 0x30;
    const PIT_HZ: u128 = 1_193_182;

    const CMOS_INDEX: u16 = 0x70;
    const CMOS_DATA: u16 = 0x71;
    const CMOS_CENTURY: u32 = 0x32;
    const CMOS_YEAR: u32 = 0x09;
    const CMOS_SECONDS: u32 = 0x00;

    const NOW_NS: u64 = 1_000;
    const KERNEL_RAM_BYTES: usize = 64 << 20;
    /// Where the loader puts `boot_params` and the command line (low-memory layout in `boot.rs`).
    const BOOT_PARAMS_ADDR: usize = 0x7000;
    const BOOT_PARAMS_BYTES: usize = 0x1000;
    const CMDLINE_ADDR: usize = 0x2_0000;
    const RAMDISK_IMAGE_FIELD: usize = 0x218;
    const RAMDISK_SIZE_FIELD: usize = 0x21c;
    const NO_DEVICE_PORT: u16 = 0x1234;
    const KBC_COMMAND_PORT: u16 = 0x64;
    const KBC_PULSE_RESET: u32 = 0xfe;
    const KBC_OTHER_COMMAND: u32 = 0xd1;
    const FAST_A20_PORT: u16 = 0x92;
    const FAST_A20_RESET_BIT: u32 = 0x01;
    const FAST_A20_GATE_BIT: u32 = 0x02;
    const RESET_CONTROL_PORT: u16 = 0xcf9;
    const RESET_CONTROL_CPU_BIT: u32 = 0x04;
    const RESET_CONTROL_OTHER_BITS: u32 = 0x0a;
    const RAM_BYTES: usize = 64 * 1024;

    fn platform() -> (Platform, ConsoleHandle) {
        let console = ConsoleHandle::default();
        (Platform::new(console.clone()), console)
    }

    fn remap_pic(platform: &mut Platform) {
        platform.io_out(PIC_MASTER_COMMAND, 1, ICW1_INIT_WITH_ICW4, NOW_NS);
        platform.io_out(PIC_MASTER_DATA, 1, IRQ_BASE_VECTOR, NOW_NS);
        platform.io_out(PIC_MASTER_DATA, 1, ICW3_CASCADE_ON_IRQ2, NOW_NS);
        platform.io_out(PIC_MASTER_DATA, 1, ICW4_8086, NOW_NS);
        platform.io_out(PIC_MASTER_DATA, 1, UNMASK_ALL, NOW_NS);
    }

    fn program_pit_one_shot(platform: &mut Platform, count: u16, now_ns: u64) {
        platform.io_out(PIT_COMMAND, 1, PIT_ONE_SHOT, now_ns);
        platform.io_out(PIT_CHANNEL0, 1, count as u32 & 0xff, now_ns);
        platform.io_out(PIT_CHANNEL0, 1, count as u32 >> 8, now_ns);
    }

    fn pit_delay_ns(count: u16) -> u64 {
        (count as u128 * 1_000_000_000).div_ceil(PIT_HZ) as u64
    }

    fn tick(platform: &mut Platform, now_ns: u64) {
        let mut ram = vec![0u8; RAM_BYTES];
        platform.tick(now_ns, &mut ram);
    }

    #[test]
    fn default_command_lines_differ_only_in_the_console_word() {
        assert_eq!(
            DEFAULT_CMDLINE,
            "console=hvc0 earlyprintk=serial,ttyS0 mitigations=off acpi=off panic=-1"
        );
        assert_eq!(
            DEFAULT_SERIAL_CMDLINE,
            DEFAULT_CMDLINE.replace("console=hvc0", "console=ttyS0")
        );
    }

    #[test]
    fn default_command_line_fits_every_kernels_minimum_limit() {
        // 2.06+ kernels advertise at least 255; the loader also needs room for the NUL.
        for cmdline in [DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE] {
            assert!(cmdline.len() < 255);
            assert!(cmdline.is_ascii());
        }
    }

    #[test]
    fn load_linux_returns_the_64_bit_entry_state_and_fills_ram() {
        let image = fake_bzimage(&[0xf4]);
        let mut ram = vec![0u8; KERNEL_RAM_BYTES];
        let state = load_linux(&mut ram, &image, None, DEFAULT_CMDLINE).unwrap();
        assert_eq!(state.rsi, BOOT_PARAMS_ADDR as u64);
        assert_eq!(ram[state.rip as usize], 0xf4);
        let at = CMDLINE_ADDR;
        assert_eq!(
            &ram[at..at + DEFAULT_CMDLINE.len()],
            DEFAULT_CMDLINE.as_bytes()
        );
    }

    #[test]
    fn load_linux_reports_loader_errors() {
        let mut ram = vec![0u8; KERNEL_RAM_BYTES];
        assert!(matches!(
            load_linux(&mut ram, &[0; 64], None, ""),
            Err(Error::BadKernel(_))
        ));
        let image = FakeKernel::new(&[0xf4]).build();
        let mut tiny = vec![0u8; 1 << 20];
        assert!(matches!(
            load_linux(&mut tiny, &image, None, ""),
            Err(Error::BadConfig(_))
        ));
    }

    #[test]
    fn load_linux_places_an_initrd_like_the_machine_does() {
        let image = fake_bzimage(&[0xf4]);
        let mut ram = vec![0u8; KERNEL_RAM_BYTES];
        let initrd = [1u8, 2, 3, 4];
        load_linux(&mut ram, &image, Some(&initrd), "").unwrap();
        let params = &ram[BOOT_PARAMS_ADDR..BOOT_PARAMS_ADDR + BOOT_PARAMS_BYTES];
        let at = u32::from_le_bytes(
            params[RAMDISK_IMAGE_FIELD..RAMDISK_IMAGE_FIELD + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let size = u32::from_le_bytes(
            params[RAMDISK_SIZE_FIELD..RAMDISK_SIZE_FIELD + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(size, 4);
        assert_eq!(&ram[at..at + 4], initrd);
    }

    #[test]
    fn io_to_an_unclaimed_port_reads_all_ones_of_the_requested_width() {
        let (mut platform, _) = platform();
        assert_eq!(platform.io_in(NO_DEVICE_PORT, 1, NOW_NS), 0xff);
        assert_eq!(platform.io_in(NO_DEVICE_PORT, 2, NOW_NS), 0xffff);
        assert_eq!(platform.io_in(NO_DEVICE_PORT, 4, NOW_NS), 0xffff_ffff);
        platform.io_out(NO_DEVICE_PORT, 4, 0xdead_beef, NOW_NS);
        assert_eq!(platform.io_in(NO_DEVICE_PORT, 1, NOW_NS), 0xff);
    }

    #[test]
    fn nothing_is_pending_or_scheduled_on_a_new_platform() {
        let (platform, _) = platform();
        assert!(!platform.irq_pending());
        assert!(!platform.reset_requested());
        assert_eq!(platform.next_deadline(0), None);
    }

    #[test]
    fn the_uart_transmits_to_the_console() {
        let (mut platform, console) = platform();
        for &byte in b"ok" {
            platform.io_out(UART_DATA, 1, byte as u32, NOW_NS);
        }
        let mut buf = [0u8; 8];
        let count = console.read(&mut buf, Some(Duration::from_millis(1)));
        assert_eq!(&buf[..count], b"ok");
    }

    #[test]
    fn the_uart_reports_an_idle_transmitter_and_no_input() {
        let (mut platform, _) = platform();
        assert_eq!(platform.io_in(UART_LSR, 1, NOW_NS), LSR_TRANSMITTER_IDLE);
    }

    #[test]
    fn host_input_arrives_in_the_uart_after_a_tick() {
        let (mut platform, console) = platform();
        console.write(b"q");
        assert_eq!(platform.io_in(UART_LSR, 1, NOW_NS) & LSR_DATA_READY, 0);
        tick(&mut platform, NOW_NS);
        assert_eq!(
            platform.io_in(UART_LSR, 1, NOW_NS) & LSR_DATA_READY,
            LSR_DATA_READY
        );
        assert_eq!(platform.io_in(UART_DATA, 1, NOW_NS), b'q' as u32);
        assert_eq!(platform.io_in(UART_LSR, 1, NOW_NS) & LSR_DATA_READY, 0);
    }

    #[test]
    fn a_uart_receive_interrupt_reaches_the_pic_with_the_remapped_vector() {
        let (mut platform, console) = platform();
        remap_pic(&mut platform);
        platform.io_out(UART_IER, 1, IER_RX_AVAILABLE, NOW_NS);
        assert!(!platform.irq_pending());
        console.write(b"x");
        tick(&mut platform, NOW_NS);
        assert!(platform.irq_pending());
        assert_eq!(platform.irq_ack(), Some(IRQ_BASE_VECTOR as u8 + IRQ_UART));
        assert!(!platform.irq_pending());
        assert_eq!(platform.irq_ack(), None);
    }

    #[test]
    fn a_masked_interrupt_stays_pending_out_of_sight() {
        let (mut platform, console) = platform();
        remap_pic(&mut platform);
        platform.io_out(PIC_MASTER_DATA, 1, 0xff, NOW_NS);
        platform.io_out(UART_IER, 1, IER_RX_AVAILABLE, NOW_NS);
        console.write(b"x");
        tick(&mut platform, NOW_NS);
        assert!(!platform.irq_pending());
        assert_eq!(platform.irq_ack(), None);
        // Unmasking delivers it.
        platform.io_out(PIC_MASTER_DATA, 1, UNMASK_ALL, NOW_NS);
        assert!(platform.irq_pending());
    }

    #[test]
    fn the_pit_schedules_a_deadline_and_raises_irq0_when_it_passes() {
        let (mut platform, _) = platform();
        remap_pic(&mut platform);
        const COUNT: u16 = 1000;
        program_pit_one_shot(&mut platform, COUNT, NOW_NS);
        let deadline = NOW_NS + pit_delay_ns(COUNT);
        assert_eq!(platform.next_deadline(NOW_NS), Some(deadline));
        tick(&mut platform, deadline - 1);
        assert!(!platform.irq_pending());
        tick(&mut platform, deadline);
        assert!(platform.irq_pending());
        assert_eq!(platform.irq_ack(), Some(IRQ_BASE_VECTOR as u8 + IRQ_PIT));
    }

    #[test]
    fn a_one_shot_pit_leaves_no_deadline_after_it_fires() {
        let (mut platform, _) = platform();
        remap_pic(&mut platform);
        program_pit_one_shot(&mut platform, 500, NOW_NS);
        let deadline = platform.next_deadline(NOW_NS).unwrap();
        tick(&mut platform, deadline);
        assert_eq!(platform.next_deadline(deadline), None);
    }

    #[test]
    fn times_are_whatever_monotonic_clock_the_host_uses() {
        // Starting the same program at a very different origin shifts the deadline by the same.
        let (mut platform, _) = platform();
        let origin = 5_000_000_000_000;
        program_pit_one_shot(&mut platform, 2000, origin);
        assert_eq!(
            platform.next_deadline(origin),
            Some(origin + pit_delay_ns(2000))
        );
    }

    #[test]
    fn ticking_before_anything_is_due_changes_nothing() {
        let (mut platform, _) = platform();
        tick(&mut platform, 0);
        tick(&mut platform, u64::MAX / 2);
        assert!(!platform.irq_pending());
        assert!(!platform.reset_requested());
    }

    #[test]
    fn tick_may_be_given_any_ram_slice_including_an_empty_one() {
        let (mut platform, _) = platform();
        platform.tick(NOW_NS, &mut []);
    }

    #[test]
    fn every_reset_port_sets_the_sticky_reset_flag() {
        for (port, value) in [
            (KBC_COMMAND_PORT, KBC_PULSE_RESET),
            (FAST_A20_PORT, FAST_A20_RESET_BIT),
            (RESET_CONTROL_PORT, RESET_CONTROL_CPU_BIT),
        ] {
            let (mut platform, _) = platform();
            assert!(!platform.reset_requested(), "{port:#x}");
            platform.io_out(port, 1, value, NOW_NS);
            assert!(platform.reset_requested(), "{port:#x}");
            platform.io_out(KBC_COMMAND_PORT, 1, 0, NOW_NS);
            assert!(platform.reset_requested(), "{port:#x} stays set");
        }
    }

    #[test]
    fn writes_to_reset_ports_without_the_reset_bit_do_nothing() {
        let (mut platform, _) = platform();
        platform.io_out(KBC_COMMAND_PORT, 1, KBC_OTHER_COMMAND, NOW_NS);
        platform.io_out(FAST_A20_PORT, 1, FAST_A20_GATE_BIT, NOW_NS);
        platform.io_out(RESET_CONTROL_PORT, 1, RESET_CONTROL_OTHER_BITS, NOW_NS);
        assert!(!platform.reset_requested());
    }

    #[test]
    fn the_rtc_follows_the_boot_time_set_by_the_host() {
        let (mut platform, _) = platform();
        platform.set_boot_time(1_700_000_000);
        let mut read = |register: u32| {
            platform.io_out(CMOS_INDEX, 1, register, NOW_NS);
            platform.io_in(CMOS_DATA, 1, NOW_NS)
        };
        assert_eq!(read(CMOS_CENTURY), 0x20);
        assert_eq!(read(CMOS_YEAR), 0x23);
        assert_eq!(read(CMOS_SECONDS), 0x20);
    }

    #[test]
    fn the_rtc_advances_with_the_host_supplied_time_not_a_wall_clock() {
        let (mut platform, _) = platform();
        platform.set_boot_time(0);
        platform.io_out(CMOS_INDEX, 1, CMOS_SECONDS, 0);
        assert_eq!(platform.io_in(CMOS_DATA, 1, 0), 0x00);
        assert_eq!(platform.io_in(CMOS_DATA, 1, 59_000_000_000), 0x59);
        assert_eq!(platform.io_in(CMOS_DATA, 1, 61_000_000_000), 0x01);
    }

    #[test]
    fn two_platforms_do_not_share_devices() {
        let (mut first, first_console) = platform();
        let (mut second, second_console) = platform();
        first.io_out(UART_DATA, 1, b'a' as u32, NOW_NS);
        second.io_out(KBC_COMMAND_PORT, 1, KBC_PULSE_RESET, NOW_NS);
        assert_eq!(first_console.read(&mut [0u8; 4], None), 1);
        assert_eq!(second_console.read(&mut [0u8; 4], None), 0);
        assert!(!first.reset_requested());
    }
}
