//! The UEFI firmware image `build.rs` compiled from `crates/uemulator-firmware`, and the
//! variables the CLI seeds into a machine's nvram so it boots something specific.

use uemulator::{KeyValueError, KeyValueStore};

/// The flat image `MachineConfig::uefi_firmware` takes.
pub const IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/uemulator-firmware.bin"));

/// `8be4df61-93ca-11d2-aa0d-00e098032b8c`, the vendor of the variables the boot manager reads.
const GLOBAL_VARIABLE_GUID: &str = "8be4df61-93ca-11d2-aa0d-00e098032b8c";
const ATTRIBUTES_NV_BS_RT: u32 = 0x07;
const LOAD_OPTION_ACTIVE: u32 = 1;
const DESCRIPTION: &str = "uemulator";
const MEDIA_DEVICE_PATH: u8 = 4;
const FILE_PATH_SUBTYPE: u8 = 4;
const END_DEVICE_PATH: [u8; 4] = [0x7f, 0xff, 4, 0];
const NODE_HEADER_BYTES: usize = 4;
/// `EFI_OS_INDICATIONS_BOOT_TO_FW_UI`: the firmware opens its boot menu on the next start and
/// clears the bit.
const OS_INDICATIONS_BOOT_TO_FW_UI: u64 = 1 << 0;

/// The firmware's key for a variable (`var/<guid>/<name>`, `docs/plans/19-uefi-firmware.md`).
fn variable_key(name: &str) -> Vec<u8> {
    format!("var/{GLOBAL_VARIABLE_GUID}/{name}").into_bytes()
}

fn variable_value(data: &[u8]) -> Vec<u8> {
    let mut value = ATTRIBUTES_NV_BS_RT.to_le_bytes().to_vec();
    value.extend_from_slice(data);
    value
}

fn utf16_nul(text: &str) -> Vec<u8> {
    text.encode_utf16()
        .chain(std::iter::once(0))
        .flat_map(|unit| unit.to_le_bytes())
        .collect()
}

/// An `EFI_LOAD_OPTION` that runs `loader` (a path on the first file system that has it, such as
/// `\EFI\BOOT\BOOTX64.EFI`) with `options` as its load options.
pub fn load_option(loader: &str, options: &str) -> Vec<u8> {
    let name = utf16_nul(loader);
    let mut path = vec![MEDIA_DEVICE_PATH, FILE_PATH_SUBTYPE];
    path.extend_from_slice(&((NODE_HEADER_BYTES + name.len()) as u16).to_le_bytes());
    path.extend_from_slice(&name);
    path.extend_from_slice(&END_DEVICE_PATH);
    let mut option = LOAD_OPTION_ACTIVE.to_le_bytes().to_vec();
    option.extend_from_slice(&(path.len() as u16).to_le_bytes());
    option.extend_from_slice(&utf16_nul(DESCRIPTION));
    option.extend_from_slice(&path);
    if !options.is_empty() {
        option.extend_from_slice(&utf16_nul(options));
    }
    option
}

/// Makes `Boot0000` run `loader` with `options` and puts it first in `BootOrder`, keeping the
/// other entries.
pub fn set_boot_entry(
    store: &mut dyn KeyValueStore,
    loader: &str,
    options: &str,
) -> Result<(), KeyValueError> {
    store.set(
        &variable_key("Boot0000"),
        &variable_value(&load_option(loader, options)),
    )?;
    let order_key = variable_key("BootOrder");
    let mut order = vec![0u8, 0];
    if let Some(existing) = store.get(&order_key) {
        for pair in existing.get(4..).unwrap_or(&[]).as_chunks::<2>().0 {
            if *pair != [0, 0] {
                order.extend_from_slice(pair);
            }
        }
    }
    store.set(&order_key, &variable_value(&order))
}

/// Data of a variable as stored: the value after its attributes.
fn variable_data(store: &mut dyn KeyValueStore, name: &str) -> Option<Vec<u8>> {
    store
        .get(&variable_key(name))
        .and_then(|value| value.get(4..).map(<[u8]>::to_vec))
}

/// What to write into a machine's nvram before it boots through the UEFI firmware.
#[derive(Default)]
pub struct NvramSetup<'a> {
    /// A loader path and its load options for `Boot0000`, first in `BootOrder`.
    pub boot_entry: Option<(&'a str, &'a str)>,
    /// Open the boot menu once, through `OsIndications`, as `systemctl reboot --firmware-setup`
    /// does from inside the guest.
    pub boot_menu: bool,
    /// The `Timeout` variable: seconds the firmware counts down before the default boot.
    pub boot_timeout: Option<u16>,
}

impl NvramSetup<'_> {
    pub fn apply(&self, store: &mut dyn KeyValueStore) -> Result<(), KeyValueError> {
        if let Some((loader, options)) = self.boot_entry {
            set_boot_entry(store, loader, options)?;
        }
        if self.boot_menu {
            let indications = variable_data(store, "OsIndications")
                .and_then(|data| Some(u64::from_le_bytes(data.get(..8)?.try_into().ok()?)))
                .unwrap_or(0);
            store.set(
                &variable_key("OsIndications"),
                &variable_value(&(indications | OS_INDICATIONS_BOOT_TO_FW_UI).to_le_bytes()),
            )?;
        }
        if let Some(seconds) = self.boot_timeout {
            store.set(
                &variable_key("Timeout"),
                &variable_value(&seconds.to_le_bytes()),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uemulator::MemoryKeyValueStore;

    #[test]
    fn the_image_has_the_header_the_library_checks() {
        assert_eq!(&IMAGE[..8], b"UEMUFW64");
        assert!(IMAGE.len() > 4096);
    }

    #[test]
    fn a_load_option_lays_out_as_the_specification_says() {
        let option = load_option("\\EFI\\BOOT\\BOOTX64.EFI", "a b");
        assert_eq!(u32::from_le_bytes(option[..4].try_into().unwrap()), 1);
        let path_length = u16::from_le_bytes(option[4..6].try_into().unwrap()) as usize;
        let description = utf16_nul("uemulator");
        assert_eq!(&option[6..6 + description.len()], description.as_slice());
        let path = &option[6 + description.len()..][..path_length];
        assert_eq!(&path[..2], &[4, 4]);
        assert_eq!(&path[path.len() - 4..], &END_DEVICE_PATH);
        assert_eq!(
            &option[6 + description.len() + path_length..],
            utf16_nul("a b").as_slice()
        );
    }

    #[test]
    fn the_boot_entry_goes_first_and_the_others_stay() {
        let mut store = MemoryKeyValueStore::new();
        store
            .set(&variable_key("BootOrder"), &variable_value(&[1, 0, 2, 0]))
            .unwrap();
        set_boot_entry(&mut store, "\\x.efi", "").unwrap();
        let order = store.get(&variable_key("BootOrder")).unwrap();
        assert_eq!(&order[4..], &[0, 0, 1, 0, 2, 0]);
        // Setting it again does not duplicate it.
        set_boot_entry(&mut store, "\\y.efi", "").unwrap();
        assert_eq!(
            &store.get(&variable_key("BootOrder")).unwrap()[4..],
            &[0, 0, 1, 0, 2, 0]
        );
        assert!(store.get(&variable_key("Boot0000")).is_some());
    }

    use uemulator::{EmulatedMachine, ExitReason, MachineConfig};

    const DOS_HEADER_BYTES: usize = 0x40;
    const OPTIONAL_HEADER_BYTES: usize = 0xf0;
    const HEADERS_BYTES: usize = 0x200;
    const CODE_RVA: usize = 0x1000;
    const IMAGE_BYTES: usize = 0x2000;
    const SUBSYSTEM_EFI_APPLICATION: u16 = 10;
    /// Written with `hack/asm.sh`. An EFI application that echoes its load options to the 16550,
    /// then asks the `LINUX_EFI_INITRD_MEDIA_GUID` `LoadFile2` for the initramfs size and echoes
    /// the low two bytes of it, then halts with interrupts off.
    const ECHO_LOAD_OPTIONS_AND_INITRD_SIZE: [u8; 160] = [
        0x48, 0x83, 0xec, 0x48, 0x48, 0x8b, 0x5a, 0x60, 0x48, 0x8d, 0x15, 0x71, 0x00, 0x00, 0x00,
        0x4c, 0x8d, 0x44, 0x24, 0x20, 0xff, 0x93, 0x98, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x44, 0x24,
        0x20, 0x8b, 0x48, 0x30, 0x48, 0x8b, 0x70, 0x38, 0x66, 0xba, 0xf8, 0x03, 0xfc, 0x85, 0xc9,
        0x74, 0x06, 0xac, 0xee, 0xff, 0xc9, 0xeb, 0xf6, 0x48, 0x8d, 0x0d, 0x54, 0x00, 0x00, 0x00,
        0x31, 0xd2, 0x4c, 0x8d, 0x44, 0x24, 0x20, 0xff, 0x93, 0x40, 0x01, 0x00, 0x00, 0x48, 0x8b,
        0x4c, 0x24, 0x20, 0x48, 0x8b, 0x01, 0x31, 0xd2, 0x45, 0x31, 0xc0, 0x4c, 0x8d, 0x4c, 0x24,
        0x28, 0x48, 0xc7, 0x44, 0x24, 0x28, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0x44, 0x24, 0x20,
        0x00, 0x00, 0x00, 0x00, 0xff, 0xd0, 0x48, 0x8b, 0x44, 0x24, 0x28, 0x66, 0xba, 0xf8, 0x03,
        0xee, 0x48, 0xc1, 0xe8, 0x08, 0xee, 0xfa, 0xf4, 0xa1, 0x31, 0x1b, 0x5b, 0x62, 0x95, 0xd2,
        0x11, 0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b, 0xc1, 0xc0, 0x06, 0x40, 0xb3, 0xfc,
        0x3e, 0x40, 0x99, 0x6d, 0x4a, 0x6c, 0x87, 0x24, 0xe0, 0x6d,
    ];

    fn put(bytes: &mut [u8], at: usize, value: &[u8]) {
        bytes[at..at + value.len()].copy_from_slice(value);
    }

    /// A one-section PE32+ x86-64 EFI application whose entry point is `code`, which is all the
    /// firmware's loader needs of a Linux EFI-stub kernel.
    fn efi_application(code: &[u8]) -> Vec<u8> {
        let mut image = vec![0u8; HEADERS_BYTES + 0x200];
        let coff = DOS_HEADER_BYTES + 4;
        let optional = coff + 20;
        let section = optional + OPTIONAL_HEADER_BYTES;
        put(&mut image, 0, b"MZ");
        put(&mut image, 0x3c, &(DOS_HEADER_BYTES as u32).to_le_bytes());
        put(&mut image, DOS_HEADER_BYTES, b"PE\0\0");
        put(&mut image, coff, &0x8664u16.to_le_bytes());
        put(&mut image, coff + 2, &1u16.to_le_bytes());
        put(
            &mut image,
            coff + 16,
            &(OPTIONAL_HEADER_BYTES as u16).to_le_bytes(),
        );
        put(&mut image, optional, &0x20bu16.to_le_bytes());
        put(&mut image, optional + 16, &(CODE_RVA as u32).to_le_bytes());
        put(&mut image, optional + 32, &0x1000u32.to_le_bytes());
        put(&mut image, optional + 36, &0x200u32.to_le_bytes());
        put(
            &mut image,
            optional + 56,
            &(IMAGE_BYTES as u32).to_le_bytes(),
        );
        put(
            &mut image,
            optional + 60,
            &(HEADERS_BYTES as u32).to_le_bytes(),
        );
        put(
            &mut image,
            optional + 68,
            &SUBSYSTEM_EFI_APPLICATION.to_le_bytes(),
        );
        put(&mut image, optional + 108, &16u32.to_le_bytes());
        put(&mut image, section + 8, &(code.len() as u32).to_le_bytes());
        put(&mut image, section + 12, &(CODE_RVA as u32).to_le_bytes());
        put(&mut image, section + 16, &0x200u32.to_le_bytes());
        put(
            &mut image,
            section + 20,
            &(HEADERS_BYTES as u32).to_le_bytes(),
        );
        image[HEADERS_BYTES..HEADERS_BYTES + code.len()].copy_from_slice(code);
        image
    }

    fn utf16_bytes(text: &str) -> Vec<u8> {
        utf16_nul(text)
    }

    #[test]
    fn the_firmware_boots_the_kernel_it_is_given_with_its_command_line_and_initramfs() {
        const INITRD_BYTES: usize = 0x1234;
        let cmdline = "console=hvc0 uefi-test";
        let mut machine = EmulatedMachine::new(
            MachineConfig::new()
                .uefi_firmware(IMAGE)
                .kernel(efi_application(&ECHO_LOAD_OPTIONS_AND_INITRD_SIZE))
                .initrd(vec![0x49; INITRD_BYTES])
                .cmdline(cmdline)
                .memory_mib(64)
                .reboot(false),
        )
        .expect("config is valid");
        let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = output.clone();
        machine
            .console()
            .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        let stop = machine.stop_handle();
        let watchdog = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(60));
            stop.stop();
        });
        machine.start().expect("machine starts");
        let exit = machine.wait();
        drop(watchdog);
        let output = output.lock().unwrap().clone();
        let mut expected = utf16_bytes(cmdline);
        expected.extend_from_slice(&(INITRD_BYTES as u16).to_le_bytes());
        assert!(
            output
                .windows(expected.len())
                .any(|window| window == expected),
            "load options and initramfs size not echoed: {:?}",
            String::from_utf8_lossy(&output)
        );
        assert_eq!(exit, ExitReason::Shutdown);
    }

    /// Runs `cfg` to its end (at most a minute) and returns why it ended and its console output.
    fn run_capturing(cfg: MachineConfig) -> (ExitReason, Vec<u8>) {
        let mut machine = EmulatedMachine::new(cfg).expect("config is valid");
        let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = output.clone();
        machine
            .console()
            .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        let stop = machine.stop_handle();
        let watchdog = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(60));
            stop.stop();
        });
        machine.start().expect("machine starts");
        let exit = machine.wait();
        drop(watchdog);
        let output = output.lock().unwrap().clone();
        (exit, output)
    }

    /// Written with `hack/asm.sh`. A Multiboot2 EFI amd64 entry point: checks the magic in eax,
    /// then walks the boot information at rbx and writes the command line tag's string and each
    /// module's string to the 16550, each followed by `|`, then halts with interrupts off.
    const ECHO_MULTIBOOT2_STRINGS: [u8; 72] = [
        0x3d, 0x89, 0x62, 0xd7, 0x36, 0x75, 0x3f, 0x48, 0x8d, 0x73, 0x08, 0x8b, 0x0e, 0x85, 0xc9,
        0x74, 0x35, 0x48, 0x8d, 0x7e, 0x08, 0x83, 0xf9, 0x01, 0x74, 0x09, 0x48, 0x8d, 0x7e, 0x10,
        0x83, 0xf9, 0x03, 0x75, 0x13, 0x66, 0xba, 0xf8, 0x03, 0x8a, 0x07, 0x84, 0xc0, 0x74, 0x06,
        0xee, 0x48, 0xff, 0xc7, 0xeb, 0xf4, 0xb0, 0x7c, 0xee, 0x8b, 0x4e, 0x04, 0x48, 0x01, 0xce,
        0x48, 0x83, 0xc6, 0x07, 0x48, 0x83, 0xe6, 0xf8, 0xeb, 0xc5, 0xfa, 0xf4,
    ];
    /// Where the test image is linked, as Xen is: 2 MiB, so the firmware has to relocate it.
    const MULTIBOOT2_LINK_ADDRESS: u32 = 0x20_0000;
    const MULTIBOOT2_HEADER_OFFSET: usize = 0x60;
    const MULTIBOOT2_CODE_OFFSET: usize = 0x100;
    const MULTIBOOT2_HEADER_MAGIC: u32 = 0xe852_50d6;

    /// An ELF32 shaped like Xen's `xen`: one PT_LOAD at `MULTIBOOT2_LINK_ADDRESS` with a
    /// Multiboot2 header asking for basic memory and the map, a relocatable placement (2 MiB
    /// aligned, highest), boot services kept and the EFI amd64 entry at `code`.
    fn multiboot2_hypervisor(code: &[u8]) -> Vec<u8> {
        let entry = MULTIBOOT2_LINK_ADDRESS + MULTIBOOT2_CODE_OFFSET as u32;
        let tags: [(u16, u16, Vec<u32>); 5] = [
            (1, 0, vec![4, 6]),
            (
                10,
                1,
                vec![MULTIBOOT2_LINK_ADDRESS, 0xffff_ffff, 0x20_0000, 2],
            ),
            (7, 1, vec![]),
            (9, 1, vec![entry]),
            (0, 0, vec![]),
        ];
        let mut header = vec![0u8; 16];
        for (kind, flags, words) in tags {
            header.extend_from_slice(&kind.to_le_bytes());
            header.extend_from_slice(&flags.to_le_bytes());
            header.extend_from_slice(&(8 + 4 * words.len() as u32).to_le_bytes());
            for word in words {
                header.extend_from_slice(&word.to_le_bytes());
            }
            header.resize(header.len().next_multiple_of(8), 0);
        }
        let length = header.len() as u32;
        put(&mut header, 0, &MULTIBOOT2_HEADER_MAGIC.to_le_bytes());
        put(&mut header, 8, &length.to_le_bytes());
        let checksum = 0u32
            .wrapping_sub(MULTIBOOT2_HEADER_MAGIC)
            .wrapping_sub(length);
        put(&mut header, 12, &checksum.to_le_bytes());

        let file_bytes = MULTIBOOT2_CODE_OFFSET + code.len();
        let mut image = vec![0u8; file_bytes];
        put(&mut image, 0, b"\x7fELF\x01\x01\x01");
        put(&mut image, 0x10, &2u16.to_le_bytes());
        put(&mut image, 0x12, &3u16.to_le_bytes());
        put(&mut image, 0x18, &entry.to_le_bytes());
        put(&mut image, 0x1c, &0x34u32.to_le_bytes());
        put(&mut image, 0x2a, &32u16.to_le_bytes());
        put(&mut image, 0x2c, &1u16.to_le_bytes());
        let segment = [
            1,
            0,
            MULTIBOOT2_LINK_ADDRESS,
            MULTIBOOT2_LINK_ADDRESS,
            file_bytes as u32,
            0x1_0000,
            7,
            0x20_0000,
        ];
        for (index, word) in segment.iter().enumerate() {
            put(&mut image, 0x34 + index * 4, &word.to_le_bytes());
        }
        put(&mut image, MULTIBOOT2_HEADER_OFFSET, &header);
        put(&mut image, MULTIBOOT2_CODE_OFFSET, code);
        image
    }

    #[test]
    fn the_firmware_enters_a_multiboot2_hypervisor_with_the_kernel_as_a_module() {
        let (exit, output) = run_capturing(
            MachineConfig::new()
                .uefi_firmware(IMAGE)
                .hypervisor(multiboot2_hypervisor(&ECHO_MULTIBOOT2_STRINGS))
                .hypervisor_cmdline("console=com1 hv-test")
                // dom0 needs no EFI stub: the hypervisor loads it.
                .kernel(vec![0x4b; 0x3000])
                .initrd(vec![0x49; 0x1000])
                .cmdline("console=hvc0 dom0-test")
                .memory_mib(64)
                .reboot(false),
        );
        let output = String::from_utf8_lossy(&output);
        // Xen drops the first word of each unless the loader is GRUB 2, so the firmware puts
        // the file names there, as GRUB 1 did.
        assert!(
            output
                .contains("xen.gz console=com1 hv-test|vmlinuz console=hvc0 dom0-test|initramfs|"),
            "boot information strings not echoed: {output:?}"
        );
        assert_eq!(exit, ExitReason::Shutdown);
    }

    /// Written with `hack/asm.sh`. An EFI application that finds the `LoadFile2` under
    /// `XEN_EFI_RAMDISK_MEDIA_GUID`'s device path with `LocateDevicePath` and echoes the low two
    /// bytes of the size it reports, then halts with interrupts off.
    const ECHO_XEN_RAMDISK_SIZE: [u8; 169] = [
        0x48, 0x83, 0xec, 0x48, 0x48, 0x8b, 0x5a, 0x60, 0x48, 0x8d, 0x05, 0x82, 0x00, 0x00, 0x00,
        0x48, 0x89, 0x44, 0x24, 0x30, 0x48, 0x8d, 0x0d, 0x66, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x54,
        0x24, 0x30, 0x4c, 0x8d, 0x44, 0x24, 0x38, 0xff, 0x93, 0xb8, 0x00, 0x00, 0x00, 0x48, 0x85,
        0xc0, 0x75, 0x4f, 0x48, 0x8b, 0x4c, 0x24, 0x38, 0x48, 0x8d, 0x15, 0x45, 0x00, 0x00, 0x00,
        0x4c, 0x8d, 0x44, 0x24, 0x20, 0xff, 0x93, 0x98, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x4c, 0x24,
        0x20, 0x48, 0x8b, 0x54, 0x24, 0x30, 0x45, 0x31, 0xc0, 0x4c, 0x8d, 0x4c, 0x24, 0x28, 0x48,
        0xc7, 0x44, 0x24, 0x28, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0x44, 0x24, 0x20, 0x00, 0x00,
        0x00, 0x00, 0x48, 0x8b, 0x01, 0xff, 0xd0, 0x48, 0x8b, 0x44, 0x24, 0x28, 0x66, 0xba, 0xf8,
        0x03, 0xee, 0x48, 0xc1, 0xe8, 0x08, 0xee, 0xfa, 0xf4, 0xc1, 0xc0, 0x06, 0x40, 0xb3, 0xfc,
        0x3e, 0x40, 0x99, 0x6d, 0x4a, 0x6c, 0x87, 0x24, 0xe0, 0x6d, 0x04, 0x03, 0x14, 0x00, 0x01,
        0xfd, 0xb1, 0x5d, 0xcb, 0xc3, 0x12, 0x48, 0xb2, 0xba, 0x87, 0x91, 0xe5, 0x2d, 0x4a, 0x89,
        0x7f, 0xff, 0x04, 0x00,
    ];

    #[test]
    fn xen_efi_finds_each_payload_under_its_own_media_loader() {
        // Three loaders are published (configuration, kernel, ramdisk) with different sizes; the
        // one under the ramdisk's device path must answer with the ramdisk's.
        const RAMDISK_BYTES: usize = 0x2345;
        let (exit, output) = run_capturing(
            MachineConfig::new()
                .uefi_firmware(IMAGE)
                .hypervisor(efi_application(&ECHO_XEN_RAMDISK_SIZE))
                .kernel(vec![0x4b; 0x3000])
                .initrd(vec![0x49; RAMDISK_BYTES])
                .memory_mib(64)
                .reboot(false),
        );
        let expected = (RAMDISK_BYTES as u16).to_le_bytes();
        assert!(
            output.windows(2).any(|window| window == expected),
            "ramdisk size not echoed: {:?}",
            String::from_utf8_lossy(&output)
        );
        assert_eq!(exit, ExitReason::Shutdown);
    }

    #[test]
    fn the_boot_menu_request_keeps_the_other_indications() {
        let mut store = MemoryKeyValueStore::new();
        let file_capsule_delivery = 1u64 << 2;
        store
            .set(
                &variable_key("OsIndications"),
                &variable_value(&file_capsule_delivery.to_le_bytes()),
            )
            .unwrap();
        let setup = NvramSetup {
            boot_menu: true,
            boot_timeout: Some(5),
            ..NvramSetup::default()
        };
        setup.apply(&mut store).unwrap();
        assert_eq!(
            variable_data(&mut store, "OsIndications").unwrap(),
            (file_capsule_delivery | OS_INDICATIONS_BOOT_TO_FW_UI).to_le_bytes()
        );
        assert_eq!(
            variable_data(&mut store, "Timeout").unwrap(),
            5u16.to_le_bytes()
        );
        assert_eq!(
            store.get(&variable_key("Timeout")).unwrap()[..4],
            ATTRIBUTES_NV_BS_RT.to_le_bytes()
        );
    }

    #[test]
    fn an_empty_setup_writes_nothing() {
        let mut store = MemoryKeyValueStore::new();
        NvramSetup::default().apply(&mut store).unwrap();
        assert!(store.list(b"").is_empty());
    }
}
