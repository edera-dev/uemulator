use super::*;
use crate::testkit::{CODE_BASE, FakeKernel, Harness, RBX, RDX, Stop};

const MIB: usize = 1 << 20;
const GIB: u64 = 1 << 30;
/// Enough for the fake kernel at `KERNEL_LOAD` plus the initrds the tests use.
const TEST_RAM: usize = 64 * MIB;
const ENTRY_CODE: [u8; 1] = [0xf4];
/// The conventional 639 KiB end of low RAM, spelled out so a changed constant in `boot.rs` is caught.
const EXPECTED_LOW_RAM_END: u64 = 0x9_fc00;
/// A load address above the default one, for non-relocatable kernels and ignored `pref_address`es.
const ALTERNATE_KERNEL_ADDRESS: u64 = 32 << 20;
/// An `initrd_addr_max` (last reachable byte) 48 MiB up, below the end of the 64 MiB test RAM.
const LOW_INITRD_ADDR_MAX: u32 = (48 << 20) - 1;

const ENTRY_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const HUGE_PAGE_OFFSET_MASK: u64 = (1 << HUGE_PAGE_SHIFT) - 1;
const PAGE_TABLE_INDEX_MASK: u64 = ENTRIES_PER_TABLE - 1;
const GIB_OFFSET_MASK: u64 = GIB - 1;
const PML4_SHIFT: u32 = 39;
const PDPT_SHIFT: u32 = 30;
const FAKE_PAYLOAD_LEN: usize = ENTRY_64_OFFSET as usize + ENTRY_CODE.len();
/// Control byte patterns written into header fields the fake kernel leaves zero, to see which of
/// them reach `boot_params`.
const MARKER_INSIDE_HEADER: u8 = 0xaa;
const MARKER_PAST_HEADER: u8 = 0xbb;
const PAGE_FAULT_VECTOR: u8 = 14;

fn kernel() -> FakeKernel {
    FakeKernel::new(&ENTRY_CODE)
}

fn load_with(
    fake: &FakeKernel,
    initrd: Option<&[u8]>,
    cmdline: &str,
    ram_len: usize,
) -> (Result<BootState, Error>, Vec<u8>) {
    let mut ram = vec![0u8; ram_len];
    let result = load(&mut ram, &fake.build(), initrd, cmdline);
    (result, ram)
}

/// Loads into `TEST_RAM` and expects success.
fn load_ok(fake: &FakeKernel, initrd: Option<&[u8]>, cmdline: &str) -> (BootState, Vec<u8>) {
    let (result, ram) = load_with(fake, initrd, cmdline, TEST_RAM);
    (result.expect("load must succeed"), ram)
}

fn bad_kernel(result: Result<BootState, Error>) -> String {
    match result {
        Err(Error::BadKernel(message)) => message,
        other => panic!("expected BadKernel, got {other:?}"),
    }
}

fn parse_error(image: &[u8]) -> String {
    match SetupHeader::parse(image) {
        Err(Error::BadKernel(message)) => message,
        other => panic!("expected BadKernel, got {other:?}"),
    }
}

fn bad_config(result: Result<BootState, Error>) -> String {
    match result {
        Err(Error::BadConfig(message)) => message,
        other => panic!("expected BadConfig, got {other:?}"),
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn boot_params(ram: &[u8]) -> &[u8] {
    &ram[BOOT_PARAMS as usize..][..BOOT_PARAMS_SIZE]
}

fn initrd_fields(ram: &[u8]) -> (u32, u32) {
    let params = boot_params(ram);
    (
        u32_at(params, RAMDISK_IMAGE_OFFSET),
        u32_at(params, RAMDISK_SIZE_OFFSET),
    )
}

/// Four-level walk over the tables `load` built, understanding only present entries and the 1 GiB
/// and 2 MiB pages `load` creates.
fn walk(ram: &[u8], cr3: u64, linear: u64) -> Option<u64> {
    let entry =
        |table: u64, index: u64| u64_at(ram, (table + index * TABLE_ENTRY_SIZE as u64) as usize);
    let pml4e = entry(cr3, (linear >> PML4_SHIFT) & PAGE_TABLE_INDEX_MASK);
    if pml4e & PTE_PRESENT == 0 {
        return None;
    }
    let pdpte = entry(
        pml4e & ENTRY_ADDRESS_MASK,
        (linear >> PDPT_SHIFT) & PAGE_TABLE_INDEX_MASK,
    );
    if pdpte & PTE_PRESENT == 0 {
        return None;
    }
    if pdpte & PTE_PAGE_SIZE != 0 {
        return Some((pdpte & ENTRY_ADDRESS_MASK & !GIB_OFFSET_MASK) | (linear & GIB_OFFSET_MASK));
    }
    let pde = entry(
        pdpte & ENTRY_ADDRESS_MASK,
        (linear >> HUGE_PAGE_SHIFT) & PAGE_TABLE_INDEX_MASK,
    );
    if pde & PTE_PRESENT == 0 {
        return None;
    }
    assert_ne!(pde & PTE_PAGE_SIZE, 0, "leaf entries are 2 MiB pages");
    Some((pde & ENTRY_ADDRESS_MASK & !HUGE_PAGE_OFFSET_MASK) | (linear & HUGE_PAGE_OFFSET_MASK))
}

mod header {
    use super::*;

    #[test]
    fn rejects_garbage() {
        assert!(SetupHeader::parse(&[0u8; BOOT_PARAMS_SIZE]).is_err());
    }

    #[test]
    fn rejects_an_empty_image() {
        assert!(parse_error(&[]).contains("HdrS"));
    }

    #[test]
    fn missing_magic_is_rejected() {
        let mut image = kernel().build();
        image[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4].copy_from_slice(b"HdrX");
        assert!(parse_error(&image).contains("HdrS"));
    }

    #[test]
    fn image_shorter_than_the_setup_header_is_rejected_not_indexed() {
        let image = kernel().build();
        for len in [0, 1, HEADER_MAGIC_OFFSET + 4, SETUP_HEADER_MIN_LEN - 1] {
            assert!(SetupHeader::parse(&image[..len]).is_err(), "length {len}");
        }
    }

    #[test]
    fn old_protocol_versions_are_rejected() {
        for version in [0, 0x0200, 0x0204, MIN_PROTOCOL_VERSION - 1] {
            let message = bad_kernel(
                load_with(
                    &FakeKernel {
                        version,
                        ..kernel()
                    },
                    None,
                    "",
                    TEST_RAM,
                )
                .0,
            );
            assert!(message.contains("2.15"), "{version:#x}: {message}");
        }
    }

    #[test]
    fn protocol_2_15_and_newer_are_accepted() {
        for version in [MIN_PROTOCOL_VERSION, 0x0210, 0x0212, 0xffff] {
            let (result, _) = load_with(
                &FakeKernel {
                    version,
                    ..kernel()
                },
                None,
                "",
                TEST_RAM,
            );
            assert!(result.is_ok(), "{version:#x}");
        }
    }

    #[test]
    fn loaded_high_unset_is_rejected() {
        let fake = FakeKernel {
            loadflags: 0,
            ..kernel()
        };
        assert!(bad_kernel(load_with(&fake, None, "", TEST_RAM).0).contains("LOADED_HIGH"));
    }

    #[test]
    fn loaded_high_is_judged_on_its_own_bit() {
        // Other loadflags bits set but bit 0 clear is still a zImage.
        let fake = FakeKernel {
            loadflags: !LOADFLAGS_LOADED_HIGH,
            ..kernel()
        };
        bad_kernel(load_with(&fake, None, "", TEST_RAM).0);
    }

    #[test]
    fn a_kernel_without_xlf_kernel_64_starts_in_32_bit_protected_mode_at_its_first_byte() {
        for xloadflags in [0, 2, 0xfffe] {
            let fake = FakeKernel {
                xloadflags,
                ..kernel()
            };
            let (state, ram) = load_ok(&fake, None, "");
            assert_eq!(state.mode, BootMode::Protected32, "{xloadflags:#x}");
            assert_eq!(
                state.rip, KERNEL_LOAD,
                "startup_32 is the start of the image"
            );
            assert_eq!(state.rsi, BOOT_PARAMS);
            let code = u64::from_le_bytes(ram[GDT_ADDR as usize + 16..][..8].try_into().unwrap());
            assert_eq!(
                code, BOOT_CODE32_DESCRIPTOR,
                "the boot code segment is 32-bit"
            );
        }
    }

    #[test]
    fn truncated_image_without_protected_mode_part_is_rejected() {
        // Exactly the boot sector and setup sectors, nothing after them.
        let fake = FakeKernel {
            payload_len: Some(0),
            ..kernel()
        };
        assert!(bad_kernel(load_with(&fake, None, "", TEST_RAM).0).contains("truncated"));
    }

    #[test]
    fn one_payload_byte_is_enough_to_parse() {
        let fake = FakeKernel {
            payload_len: Some(1),
            ..kernel()
        };
        assert!(SetupHeader::parse(&fake.build()).is_ok());
    }

    #[test]
    fn setup_sects_zero_means_four() {
        let mut image = FakeKernel {
            setup_sects: DEFAULT_SETUP_SECTS,
            ..kernel()
        }
        .build();
        image[SETUP_SECTS_OFFSET] = 0;
        let header = SetupHeader::parse(&image).unwrap();
        assert_eq!(header.setup_sects, DEFAULT_SETUP_SECTS);
        assert_eq!(
            header.protected_offset(),
            (DEFAULT_SETUP_SECTS as usize + 1) * SECTOR_SIZE
        );
    }

    #[test]
    fn setup_sects_zero_image_loads_the_payload_after_four_setup_sectors() {
        let mut image = FakeKernel {
            setup_sects: DEFAULT_SETUP_SECTS,
            entry_code: vec![0x90, 0x90, 0xf4],
            ..kernel()
        }
        .build();
        image[SETUP_SECTS_OFFSET] = 0;
        let mut ram = vec![0u8; TEST_RAM];
        let state = load(&mut ram, &image, None, "").unwrap();
        assert_eq!(&ram[state.rip as usize..][..3], [0x90, 0x90, 0xf4]);
    }

    #[test]
    fn setup_sects_zero_with_a_one_sector_image_is_truncated() {
        // With setup_sects == 1 the payload starts at 0x400, but 0 means 4, which is past the end.
        let mut image = kernel().build();
        image[SETUP_SECTS_OFFSET] = 0;
        assert!(image.len() <= (DEFAULT_SETUP_SECTS as usize + 1) * SECTOR_SIZE);
        assert!(parse_error(&image).contains("truncated"));
    }

    #[test]
    fn maximum_setup_sects_does_not_overflow_the_offset() {
        let header = SetupHeader {
            setup_sects: u8::MAX,
            version: 0,
            loadflags: 0,
            xloadflags: 0,
            relocatable: false,
            pref_address: 0,
            init_size: 0,
            cmdline_max: 0,
            initrd_addr_max: 0,
        };
        assert_eq!(header.protected_offset(), 256 * SECTOR_SIZE);
    }

    #[test]
    fn parse_reads_every_field_at_its_documented_offset() {
        let fake = FakeKernel {
            setup_sects: 3,
            version: 0x0215,
            loadflags: LOADFLAGS_LOADED_HIGH | 0x20,
            xloadflags: XLF_KERNEL_64 | 0x8,
            relocatable: true,
            pref_address: 0x0123_4567_89ab_c000,
            init_size: 0x00dd_ee00,
            cmdline_max: 4096,
            initrd_addr_max: 0x3fff_ffff,
            ..kernel()
        };
        let header = SetupHeader::parse(&fake.build()).unwrap();
        assert_eq!(header.setup_sects, 3);
        assert_eq!(header.version, 0x0215);
        assert_eq!(header.loadflags, LOADFLAGS_LOADED_HIGH | 0x20);
        assert_eq!(header.xloadflags, XLF_KERNEL_64 | 0x8);
        assert!(header.relocatable);
        assert_eq!(header.pref_address, 0x0123_4567_89ab_c000);
        assert_eq!(header.init_size, 0x00dd_ee00);
        assert_eq!(header.cmdline_max, 4096);
        assert_eq!(header.initrd_addr_max, 0x3fff_ffff);
    }

    #[test]
    fn any_nonzero_relocatable_byte_counts() {
        let mut image = kernel().build();
        for (value, expected) in [(0u8, false), (1, true), (0x80, true), (0xff, true)] {
            image[RELOCATABLE_KERNEL_OFFSET] = value;
            assert_eq!(SetupHeader::parse(&image).unwrap().relocatable, expected);
        }
    }
}

mod placement {
    use super::*;

    #[test]
    fn relocatable_kernel_is_loaded_at_the_default_address_and_entered_at_startup_64() {
        let fake = FakeKernel {
            pref_address: ALTERNATE_KERNEL_ADDRESS,
            ..kernel()
        };
        let (state, ram) = load_ok(&fake, None, "");
        assert_eq!(state.rip, KERNEL_LOAD + ENTRY_64_OFFSET);
        assert_eq!(ram[state.rip as usize], ENTRY_CODE[0]);
        // The payload's first bytes (the 32-bit entry) are zero in the fake, and nothing was
        // written at the ignored preferred address.
        assert!(
            ram[ALTERNATE_KERNEL_ADDRESS as usize..][..FAKE_PAYLOAD_LEN]
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    #[test]
    fn non_relocatable_kernel_is_loaded_at_its_preferred_address() {
        let pref_address = ALTERNATE_KERNEL_ADDRESS;
        let fake = FakeKernel {
            relocatable: false,
            pref_address,
            ..kernel()
        };
        let (state, ram) = load_ok(&fake, None, "");
        assert_eq!(state.rip, pref_address + ENTRY_64_OFFSET);
        assert_eq!(ram[state.rip as usize], ENTRY_CODE[0]);
        assert_eq!(ram[(KERNEL_LOAD + ENTRY_64_OFFSET) as usize], 0);
    }

    #[test]
    fn non_relocatable_kernel_at_one_mib_is_accepted() {
        let fake = FakeKernel {
            relocatable: false,
            pref_address: ONE_MIB,
            ..kernel()
        };
        let (state, _) = load_ok(&fake, None, "");
        assert_eq!(state.rip, ONE_MIB + ENTRY_64_OFFSET);
    }

    #[test]
    fn non_relocatable_kernel_below_one_mib_is_rejected_not_written_over_boot_data() {
        for pref_address in [0, 0x1000, BOOT_PARAMS, CMDLINE_ADDR, ONE_MIB - 1] {
            let fake = FakeKernel {
                relocatable: false,
                pref_address,
                ..kernel()
            };
            bad_kernel(load_with(&fake, None, "", TEST_RAM).0);
        }
    }

    #[test]
    fn non_relocatable_kernel_beyond_ram_is_rejected() {
        let fake = FakeKernel {
            relocatable: false,
            pref_address: TEST_RAM as u64,
            ..kernel()
        };
        assert!(bad_config(load_with(&fake, None, "", TEST_RAM).0).contains("too small"));
    }

    #[test]
    fn entry_code_is_copied_verbatim() {
        let code = [0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0xf4];
        let (state, ram) = load_ok(&FakeKernel::new(&code), None, "");
        assert_eq!(&ram[state.rip as usize..][..code.len()], code);
    }

    #[test]
    fn init_size_larger_than_ram_is_rejected() {
        for init_size in [TEST_RAM as u32, u32::MAX] {
            let fake = FakeKernel {
                init_size,
                ..kernel()
            };
            assert!(bad_config(load_with(&fake, None, "", TEST_RAM).0).contains("kernel"));
        }
    }

    #[test]
    fn init_size_larger_than_ram_is_rejected_with_an_initrd_too() {
        let fake = FakeKernel {
            init_size: u32::MAX,
            ..kernel()
        };
        assert!(bad_config(load_with(&fake, Some(&[1]), "", TEST_RAM).0).contains("kernel"));
    }

    #[test]
    fn kernel_footprint_may_end_exactly_at_the_end_of_ram() {
        let fake = FakeKernel {
            init_size: (TEST_RAM as u64 - KERNEL_LOAD) as u32,
            ..kernel()
        };
        assert!(load_with(&fake, None, "", TEST_RAM).0.is_ok());
        let too_big = FakeKernel {
            init_size: (TEST_RAM as u64 - KERNEL_LOAD) as u32 + 1,
            ..kernel()
        };
        bad_config(load_with(&too_big, None, "", TEST_RAM).0);
    }

    #[test]
    fn kernel_footprint_is_the_larger_of_init_size_and_payload() {
        // A payload longer than init_size still has to fit.
        let fake = FakeKernel {
            init_size: 1,
            payload_len: Some(2 * MIB),
            ..kernel()
        };
        let ram_len = KERNEL_LOAD as usize + 2 * MIB;
        assert!(load_with(&fake, None, "", ram_len).0.is_ok());
        bad_config(load_with(&fake, None, "", ram_len - 1).0);
    }

    #[test]
    fn preferred_address_plus_footprint_overflowing_u64_is_an_error() {
        let fake = FakeKernel {
            relocatable: false,
            pref_address: u64::MAX - 2,
            init_size: u32::MAX,
            ..kernel()
        };
        bad_config(load_with(&fake, None, "", TEST_RAM).0);
        let at_the_top = FakeKernel {
            pref_address: u64::MAX,
            ..fake
        };
        bad_config(load_with(&at_the_top, None, "", TEST_RAM).0);
    }

    #[test]
    fn address_arithmetic_in_put_cannot_wrap() {
        let mut bytes = vec![0u8; 16];
        let mut ram = RamSpan::from(&mut bytes);
        assert!(put(&mut ram, u64::MAX, &[1, 2]).is_err());
        assert!(put(&mut ram, u64::MAX - 1, &[1, 2, 3]).is_err());
        assert!(put(&mut ram, 15, &[1, 2]).is_err());
        assert!(put(&mut ram, 17, &[]).is_err());
        assert!(put(&mut ram, 16, &[]).is_ok());
        assert!(put(&mut ram, 14, &[7, 8]).is_ok());
        assert_eq!(bytes[14..], [7, 8]);
    }

    #[test]
    fn ram_that_cannot_hold_the_kernel_is_an_error_for_every_length() {
        for ram_len in [
            0,
            1,
            GDT_ADDR as usize - 1,
            BOOT_PARAMS as usize,
            (PML4 + PAGE_TABLE_PAGES as u64 * PAGE_SIZE) as usize - 1,
            ONE_MIB as usize - 1,
            ONE_MIB as usize,
        ] {
            let (result, _) = load_with(&kernel(), None, "", ram_len);
            bad_config(result);
        }
    }

    #[test]
    fn ram_smaller_than_one_mib_with_an_initrd_is_an_error_not_an_underflow() {
        for ram_len in [0, 1, ONE_MIB as usize - 1] {
            let (result, _) = load_with(&kernel(), Some(&[1, 2, 3]), "", ram_len);
            bad_config(result);
        }
    }

    #[test]
    fn smallest_ram_that_fits_the_kernel_loads() {
        let ram_len = KERNEL_LOAD as usize + FAKE_PAYLOAD_LEN;
        let (result, _) = load_with(&kernel(), None, "", ram_len);
        assert!(result.is_ok());
        let (result, _) = load_with(&kernel(), None, "", ram_len - 1);
        bad_config(result);
    }
}

mod cmdline {
    use super::*;

    #[test]
    fn text_is_nul_terminated_at_the_command_line_address() {
        let (state, ram) = load_ok(&kernel(), None, "console=ttyS0 quiet");
        assert_eq!(state.rsi, BOOT_PARAMS);
        let text = &ram[CMDLINE_ADDR as usize..][..20];
        assert_eq!(&text[..19], b"console=ttyS0 quiet");
        assert_eq!(text[19], 0);
    }

    #[test]
    fn boot_params_points_at_the_command_line() {
        let (_, ram) = load_ok(&kernel(), None, "x");
        assert_eq!(
            u32_at(boot_params(&ram), CMD_LINE_PTR_OFFSET),
            CMDLINE_ADDR as u32
        );
    }

    #[test]
    fn empty_command_line_is_just_a_nul() {
        let (_, ram) = load_ok(&kernel(), None, "");
        assert_eq!(ram[CMDLINE_ADDR as usize], 0);
    }

    #[test]
    fn longest_allowed_command_line_leaves_room_for_the_nul() {
        let fake = FakeKernel {
            cmdline_max: 16,
            ..kernel()
        };
        let (result, ram) = load_with(&fake, None, &"a".repeat(15), TEST_RAM);
        result.unwrap();
        assert_eq!(&ram[CMDLINE_ADDR as usize..][..16], b"aaaaaaaaaaaaaaa\0");
    }

    #[test]
    fn command_line_of_cmdline_max_bytes_is_too_long() {
        let fake = FakeKernel {
            cmdline_max: 16,
            ..kernel()
        };
        let message = bad_config(load_with(&fake, None, &"a".repeat(16), TEST_RAM).0);
        assert!(message.contains("command line too long"), "{message}");
    }

    #[test]
    fn a_kernel_with_cmdline_max_zero_accepts_no_command_line_at_all() {
        let fake = FakeKernel {
            cmdline_max: 0,
            ..kernel()
        };
        bad_config(load_with(&fake, None, "", TEST_RAM).0);
    }

    #[test]
    fn command_line_longer_than_the_low_memory_region_is_rejected_even_if_the_kernel_allows_it() {
        // 0x20000 + this would run into the reserved area (and, far enough, the kernel itself).
        let fake = FakeKernel {
            cmdline_max: u32::MAX,
            ..kernel()
        };
        let longest_fitting = (LOW_RAM_END - CMDLINE_ADDR) as usize - 1;
        assert!(
            load_with(&fake, None, &"a".repeat(longest_fitting), TEST_RAM)
                .0
                .is_ok()
        );
        let message =
            bad_config(load_with(&fake, None, &"a".repeat(longest_fitting + 1), TEST_RAM).0);
        assert!(message.contains("command line too long"), "{message}");
    }

    #[test]
    fn non_ascii_command_line_bytes_are_copied_as_utf8() {
        let (_, ram) = load_ok(&kernel(), None, "caf\u{e9}");
        assert_eq!(&ram[CMDLINE_ADDR as usize..][..6], "caf\u{e9}\0".as_bytes());
    }
}

mod initrd {
    use super::*;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index * 7 + 3) as u8).collect()
    }

    #[test]
    fn is_placed_at_the_top_of_ram_page_aligned() {
        let data = pattern(MIB + 1);
        let (_, ram) = load_ok(&kernel(), Some(&data), "");
        let (image, size) = initrd_fields(&ram);
        let expected = (TEST_RAM as u64 - data.len() as u64) & !PAGE_MASK;
        assert_eq!(image as u64, expected);
        assert_eq!(image as u64 % PAGE_SIZE, 0);
        assert_eq!(size as usize, data.len());
        assert_eq!(&ram[image as usize..][..data.len()], data.as_slice());
    }

    #[test]
    fn a_page_multiple_ends_exactly_at_the_end_of_ram() {
        let data = pattern(8 * PAGE_SIZE as usize);
        let (_, ram) = load_ok(&kernel(), Some(&data), "");
        let (image, _) = initrd_fields(&ram);
        assert_eq!(image as usize + data.len(), TEST_RAM);
    }

    #[test]
    fn stays_below_initrd_addr_max() {
        let fake = FakeKernel {
            initrd_addr_max: LOW_INITRD_ADDR_MAX,
            ..kernel()
        };
        let data = pattern(100_000);
        let (_, ram) = load_ok(&fake, Some(&data), "");
        let (image, size) = initrd_fields(&ram);
        assert!(
            image as u64 + size as u64 <= LOW_INITRD_ADDR_MAX as u64 + 1,
            "{image:#x}+{size:#x}"
        );
        assert_eq!(
            image as u64,
            (LOW_INITRD_ADDR_MAX as u64 + 1 - data.len() as u64) & !PAGE_MASK
        );
        assert_eq!(&ram[image as usize..][..data.len()], data.as_slice());
    }

    #[test]
    fn initrd_addr_max_beyond_ram_is_capped_by_ram() {
        let fake = FakeKernel {
            initrd_addr_max: u32::MAX,
            ..kernel()
        };
        let data = pattern(4096);
        let (_, ram) = load_ok(&fake, Some(&data), "");
        let (image, _) = initrd_fields(&ram);
        assert_eq!(image as usize, TEST_RAM - 4096);
    }

    #[test]
    fn initrd_addr_max_of_the_default_2gib_window_does_not_touch_ram_above_it() {
        let ram_len = 3 * GIB as usize;
        let fake = FakeKernel {
            initrd_addr_max: 0x7fff_ffff,
            ..kernel()
        };
        let data = pattern(4096);
        let (result, ram) = load_with(&fake, Some(&data), "", ram_len);
        result.unwrap();
        let (image, _) = initrd_fields(&ram);
        assert_eq!(image as u64, 0x8000_0000 - 4096);
    }

    #[test]
    fn must_not_overlap_the_kernel_footprint() {
        // init_size reaches within 4 KiB of the top, leaving no page-aligned room for 8 KiB.
        let fake = FakeKernel {
            init_size: (TEST_RAM as u64 - KERNEL_LOAD - PAGE_SIZE) as u32,
            ..kernel()
        };
        let message = bad_config(load_with(&fake, Some(&pattern(8192)), "", TEST_RAM).0);
        assert!(message.contains("initrd"), "{message}");
        // One page fits in exactly that gap.
        let (result, ram) = load_with(&fake, Some(&pattern(4096)), "", TEST_RAM);
        result.unwrap();
        assert_eq!(initrd_fields(&ram).0 as usize, TEST_RAM - 4096);
    }

    #[test]
    fn alignment_may_not_push_it_into_the_kernel() {
        // The aligned-down start lands below the footprint although the unaligned one is above.
        let init_size = (TEST_RAM as u64 - KERNEL_LOAD - 4096) as u32;
        let fake = FakeKernel {
            init_size,
            ..kernel()
        };
        bad_config(load_with(&fake, Some(&pattern(4097)), "", TEST_RAM).0);
    }

    #[test]
    fn larger_than_ram_is_an_error_not_an_underflow() {
        for len in [TEST_RAM + 1, TEST_RAM + 4096, 2 * TEST_RAM] {
            let data = vec![0u8; len];
            let message = bad_config(load_with(&kernel(), Some(&data), "", TEST_RAM).0);
            assert!(message.contains("initrd"), "{len}: {message}");
        }
    }

    #[test]
    fn larger_than_the_addressable_window_is_an_error_not_an_underflow() {
        let fake = FakeKernel {
            initrd_addr_max: 0xfff,
            ..kernel()
        };
        bad_config(load_with(&fake, Some(&pattern(4097)), "", TEST_RAM).0);
        bad_config(
            load_with(
                &FakeKernel {
                    initrd_addr_max: 0,
                    ..kernel()
                },
                Some(&pattern(2)),
                "",
                TEST_RAM,
            )
            .0,
        );
    }

    #[test]
    fn exactly_ram_sized_is_rejected_because_the_kernel_is_in_the_way() {
        let data = vec![0u8; TEST_RAM];
        bad_config(load_with(&kernel(), Some(&data), "", TEST_RAM).0);
    }

    #[test]
    fn empty_initrd_is_placed_without_failing() {
        let (_, ram) = load_ok(&kernel(), Some(&[]), "");
        let (image, size) = initrd_fields(&ram);
        assert_eq!(size, 0);
        assert_eq!(image as usize, TEST_RAM);
    }

    #[test]
    fn fields_are_zero_without_an_initrd() {
        let (_, ram) = load_ok(&kernel(), None, "");
        assert_eq!(initrd_fields(&ram), (0, 0));
    }

    #[test]
    fn initrd_does_not_disturb_the_kernel_image() {
        let data = pattern(MIB);
        let (state, ram) = load_ok(&kernel(), Some(&data), "");
        assert_eq!(ram[state.rip as usize], ENTRY_CODE[0]);
    }
}

mod boot_params_contents {
    use super::*;

    #[test]
    fn e820_map_reports_low_ram_the_reserved_hole_and_extended_ram() {
        let (_, ram) = load_ok(&kernel(), None, "");
        let params = boot_params(&ram);
        assert_eq!(params[E820_ENTRIES_OFFSET], 3);
        let entry = |index: usize| {
            let at = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
            (
                u64_at(params, at),
                u64_at(params, at + 8),
                u32_at(params, at + 16),
            )
        };
        assert_eq!(entry(0), (0, EXPECTED_LOW_RAM_END, E820_RAM));
        assert_eq!(
            entry(1),
            (
                EXPECTED_LOW_RAM_END,
                ONE_MIB - EXPECTED_LOW_RAM_END,
                E820_RESERVED
            )
        );
        assert_eq!(entry(2), (ONE_MIB, TEST_RAM as u64 - ONE_MIB, E820_RAM));
        assert_eq!(entry(3), (0, 0, 0), "no fourth entry");
    }

    fn load_with_tables() -> (BootState, Vec<u8>) {
        let mut ram = vec![0u8; TEST_RAM];
        let acpi = AcpiConfig::default();
        let state =
            load_with_acpi(&mut ram, &kernel().build(), None, "", Some(&acpi)).expect("loads");
        (state, ram)
    }

    #[test]
    fn without_acpi_there_is_no_rsdp_pointer_and_no_tables() {
        let (_, ram) = load_ok(&kernel(), None, "");
        assert_eq!(u64_at(boot_params(&ram), ACPI_RSDP_ADDR_OFFSET), 0);
        assert!(
            ram[acpi::REGION_BASE as usize..][..64]
                .iter()
                .all(|b| *b == 0)
        );
    }

    #[test]
    fn acpi_puts_the_rsdp_in_boot_params_and_the_tables_in_ram() {
        let (_, ram) = load_with_tables();
        let rsdp = u64_at(boot_params(&ram), ACPI_RSDP_ADDR_OFFSET);
        assert_eq!(rsdp, acpi::REGION_BASE);
        assert_eq!(&ram[rsdp as usize..][..8], b"RSD PTR ");
    }

    #[test]
    fn acpi_e820_marks_the_tables_as_acpi_data_and_reserves_the_ecam_window() {
        let (_, ram) = load_with_tables();
        let params = boot_params(&ram);
        let entries: Vec<(u64, u64, u32)> = (0..params[E820_ENTRIES_OFFSET] as usize)
            .map(|index| {
                let at = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
                (
                    u64_at(params, at),
                    u64_at(params, at + 8),
                    u32_at(params, at + 16),
                )
            })
            .collect();
        assert!(entries.contains(&(acpi::REGION_BASE, acpi::REGION_BYTES, E820_ACPI)));
        assert!(entries.contains(&(acpi::ECAM_RANGE.0, acpi::ECAM_RANGE.1, E820_RESERVED)));
        // Sorted and non-overlapping, as Linux expects of the BIOS map.
        for pair in entries.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{pair:?}");
        }
        assert_eq!(entries[0], (0, EXPECTED_LOW_RAM_END, E820_RAM));
        assert!(entries.contains(&(ONE_MIB, TEST_RAM as u64 - ONE_MIB, E820_RAM)));
    }

    #[test]
    fn acpi_does_not_move_the_kernel_or_the_entry_state() {
        let (with, _) = load_with_tables();
        let (without, _) = load_ok(&kernel(), None, "");
        assert_eq!(
            (with.rip, with.rsi, with.cr3),
            (without.rip, without.rsi, without.cr3)
        );
    }

    #[test]
    fn e820_extended_range_follows_the_ram_size() {
        for ram_mib in [64usize, 100, 512] {
            let (result, ram) = load_with(&kernel(), None, "", ram_mib * MIB);
            result.unwrap();
            let at = E820_TABLE_OFFSET + 2 * E820_ENTRY_SIZE;
            assert_eq!(u64_at(boot_params(&ram), at), ONE_MIB);
            assert_eq!(
                u64_at(boot_params(&ram), at + 8),
                (ram_mib * MIB) as u64 - ONE_MIB
            );
        }
    }

    #[test]
    fn e820_ranges_are_contiguous_and_cover_ram_exactly() {
        let (_, ram) = load_ok(&kernel(), None, "");
        let params = boot_params(&ram);
        let mut next = 0;
        for index in 0..params[E820_ENTRIES_OFFSET] as usize {
            let at = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
            assert_eq!(u64_at(params, at), next);
            next += u64_at(params, at + 8);
        }
        assert_eq!(next, TEST_RAM as u64);
    }

    #[test]
    fn every_boot_structure_sits_in_usable_low_memory() {
        // The layout comment promises this; the kernel would otherwise reclaim the memory.
        for (name, start, len) in [
            ("gdt", GDT_ADDR, 32),
            ("boot_params", BOOT_PARAMS, BOOT_PARAMS_SIZE as u64),
            ("page tables", PML4, PAGE_TABLE_PAGES as u64 * PAGE_SIZE),
            ("cmdline", CMDLINE_ADDR, 1),
        ] {
            assert!(start + len <= LOW_RAM_END, "{name}");
        }
        let regions = [
            (GDT_ADDR, 32),
            (BOOT_PARAMS, BOOT_PARAMS_SIZE as u64),
            (PML4, PAGE_TABLE_PAGES as u64 * PAGE_SIZE),
            (CMDLINE_ADDR, 1),
        ];
        for (index, (start, len)) in regions.iter().enumerate() {
            for (other_start, other_len) in &regions[index + 1..] {
                assert!(start + len <= *other_start || other_start + other_len <= *start);
            }
        }
    }

    #[test]
    fn loader_type_is_undefined() {
        let (_, ram) = load_ok(&kernel(), None, "");
        assert_eq!(
            boot_params(&ram)[TYPE_OF_LOADER_OFFSET],
            LOADER_TYPE_UNDEFINED
        );
    }

    #[test]
    fn can_use_heap_is_added_to_the_kernels_loadflags() {
        let fake = FakeKernel {
            loadflags: LOADFLAGS_LOADED_HIGH | 0x20,
            ..kernel()
        };
        let (_, ram) = load_ok(&fake, None, "");
        assert_eq!(
            boot_params(&ram)[LOADFLAGS_OFFSET],
            LOADFLAGS_LOADED_HIGH | 0x20 | LOADFLAGS_CAN_USE_HEAP
        );
    }

    #[test]
    fn setup_header_fields_are_copied_through() {
        let fake = FakeKernel {
            version: 0x0210,
            init_size: 0x1234,
            cmdline_max: 0xabc,
            initrd_addr_max: 0x7654_3210,
            ..kernel()
        };
        let (_, ram) = load_ok(&fake, None, "");
        let params = boot_params(&ram);
        assert_eq!(
            &params[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4],
            HEADER_MAGIC
        );
        assert_eq!(u16_at(params, VERSION_OFFSET), 0x0210);
        assert_eq!(u32_at(params, INIT_SIZE_OFFSET), 0x1234);
        assert_eq!(u32_at(params, CMDLINE_SIZE_OFFSET), 0xabc);
        assert_eq!(u32_at(params, INITRD_ADDR_MAX_OFFSET), 0x7654_3210);
        assert_eq!(u16_at(params, XLOADFLAGS_OFFSET), XLF_KERNEL_64);
        assert_eq!(params[SETUP_SECTS_OFFSET], 1);
    }

    #[test]
    fn nothing_before_the_setup_header_is_copied() {
        let (_, ram) = load_ok(&kernel(), None, "");
        let params = boot_params(&ram);
        assert!(params[..E820_ENTRIES_OFFSET].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn copy_stops_where_the_header_says_it_ends() {
        let mut image = kernel().build();
        // jump length 0x10: the header ends at 0x212, covering type_of_loader and loadflags.
        image[JUMP_LENGTH_OFFSET] = 0x10;
        image[VERSION_OFFSET] = MARKER_INSIDE_HEADER;
        image[RAMDISK_IMAGE_OFFSET] = MARKER_PAST_HEADER;
        let mut ram = vec![0u8; TEST_RAM];
        load(&mut ram, &image, None, "").unwrap();
        let params = boot_params(&ram);
        assert_eq!(params[VERSION_OFFSET], MARKER_INSIDE_HEADER);
        assert_eq!(params[RAMDISK_IMAGE_OFFSET], 0);
        // Not copied, but set by the loader regardless.
        assert_eq!(
            params[LOADFLAGS_OFFSET] & LOADFLAGS_CAN_USE_HEAP,
            LOADFLAGS_CAN_USE_HEAP
        );
    }

    #[test]
    fn a_zero_header_length_copies_only_up_to_the_magic() {
        let mut image = kernel().build();
        image[JUMP_LENGTH_OFFSET] = 0;
        image[VERSION_OFFSET] = MARKER_INSIDE_HEADER;
        let mut ram = vec![0u8; TEST_RAM];
        load(&mut ram, &image, None, "").unwrap();
        let params = boot_params(&ram);
        assert_eq!(params[VERSION_OFFSET], 0);
        assert_eq!(params[SETUP_SECTS_OFFSET], 1);
        assert_eq!(
            &params[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4],
            [0; 4]
        );
    }

    #[test]
    fn a_header_length_of_ff_is_clamped_to_the_end_of_the_header_area() {
        let mut image = kernel().build();
        image[JUMP_LENGTH_OFFSET] = 0xff;
        image[SETUP_HEADER_END_LIMIT - 1] = MARKER_INSIDE_HEADER;
        image[SETUP_HEADER_END_LIMIT] = MARKER_PAST_HEADER;
        let mut ram = vec![0u8; TEST_RAM];
        load(&mut ram, &image, None, "").unwrap();
        let params = boot_params(&ram);
        assert_eq!(params[SETUP_HEADER_END_LIMIT - 1], MARKER_INSIDE_HEADER);
        assert_eq!(params[SETUP_HEADER_END_LIMIT], 0);
    }

    #[test]
    fn the_rest_of_boot_params_stays_zero() {
        let (_, ram) = load_ok(&kernel(), Some(&[1, 2, 3]), "x");
        let params = boot_params(&ram);
        let used = [
            SETUP_SECTS_OFFSET..SETUP_HEADER_END_LIMIT,
            E820_ENTRIES_OFFSET..E820_ENTRIES_OFFSET + 1,
            E820_TABLE_OFFSET..E820_TABLE_OFFSET + 3 * E820_ENTRY_SIZE,
        ];
        for (offset, byte) in params.iter().enumerate() {
            if !used.iter().any(|range| range.contains(&offset)) {
                assert_eq!(*byte, 0, "boot_params[{offset:#x}]");
            }
        }
    }

    #[test]
    fn state_hands_the_zero_page_to_the_kernel_in_rsi() {
        let (state, _) = load_ok(&kernel(), None, "");
        assert_eq!(state.rsi, BOOT_PARAMS);
        assert_eq!(state.cr3, PML4);
        assert_eq!(state.gdt_base, GDT_ADDR);
    }
}

mod gdt_and_page_tables {
    use super::*;

    #[test]
    fn gdt_holds_two_null_entries_then_flat_code_and_data() {
        let (state, ram) = load_ok(&kernel(), None, "");
        let gdt = &ram[state.gdt_base as usize..];
        assert_eq!(u64_at(gdt, 0), 0);
        assert_eq!(u64_at(gdt, 8), 0);
        assert_eq!(u64_at(gdt, 16), BOOT_CODE_DESCRIPTOR);
        assert_eq!(u64_at(gdt, 24), BOOT_DATA_DESCRIPTOR);
        assert_eq!(state.gdt_limit, 4 * 8 - 1);
    }

    #[test]
    fn selectors_index_the_code_and_data_descriptors() {
        let (state, ram) = load_ok(&kernel(), None, "");
        let descriptor = |selector: u16| u64_at(&ram, (state.gdt_base + selector as u64) as usize);
        assert_eq!(state.cs, BOOT_CS);
        assert_eq!(state.ds, BOOT_DS);
        assert_eq!(descriptor(state.cs), BOOT_CODE_DESCRIPTOR);
        assert_eq!(descriptor(state.ds), BOOT_DATA_DESCRIPTOR);
        // The kernel's __BOOT_CS / __BOOT_DS.
        assert_eq!((state.cs, state.ds), (0x10, 0x18));
    }

    #[test]
    fn code_descriptor_is_present_dpl0_64_bit_code() {
        const PRESENT: u64 = 1 << 47;
        const DPL_MASK: u64 = 3 << 45;
        const CODE_AND_NOT_SYSTEM: u64 = 0b11000 << 40;
        const LONG_MODE: u64 = 1 << 53;
        const DEFAULT_32: u64 = 1 << 54;
        assert_ne!(BOOT_CODE_DESCRIPTOR & PRESENT, 0);
        assert_eq!(BOOT_CODE_DESCRIPTOR & DPL_MASK, 0);
        assert_eq!(
            BOOT_CODE_DESCRIPTOR & CODE_AND_NOT_SYSTEM,
            CODE_AND_NOT_SYSTEM
        );
        assert_ne!(BOOT_CODE_DESCRIPTOR & LONG_MODE, 0);
        assert_eq!(
            BOOT_CODE_DESCRIPTOR & DEFAULT_32,
            0,
            "L and D/B must not both be set"
        );
    }

    #[test]
    fn data_descriptor_is_present_dpl0_writable_data() {
        const PRESENT: u64 = 1 << 47;
        const DPL_MASK: u64 = 3 << 45;
        const NOT_SYSTEM: u64 = 1 << 44;
        const CODE: u64 = 1 << 43;
        const WRITABLE: u64 = 1 << 41;
        assert_ne!(BOOT_DATA_DESCRIPTOR & PRESENT, 0);
        assert_eq!(BOOT_DATA_DESCRIPTOR & DPL_MASK, 0);
        assert_ne!(BOOT_DATA_DESCRIPTOR & NOT_SYSTEM, 0);
        assert_eq!(BOOT_DATA_DESCRIPTOR & CODE, 0);
        assert_ne!(BOOT_DATA_DESCRIPTOR & WRITABLE, 0);
    }

    #[test]
    fn page_tables_identity_map_the_first_four_gib() {
        let (state, ram) = load_ok(&kernel(), None, "");
        for linear in [
            0,
            1,
            0xfff,
            0x1000,
            BOOT_PARAMS,
            ONE_MIB,
            KERNEL_LOAD + ENTRY_64_OFFSET,
            (2 << 20) - 1,
            2 << 20,
            GIB - 1,
            GIB,
            2 * GIB + 0x1234_5678,
            3 * GIB - 1,
            0xc000_0000,
            4 * GIB - 1,
        ] {
            assert_eq!(walk(&ram, state.cr3, linear), Some(linear), "{linear:#x}");
        }
    }

    #[test]
    fn page_tables_map_every_2mib_page_to_itself() {
        let (state, ram) = load_ok(&kernel(), None, "");
        for page in 0..IDENTITY_MAP_2M_GIB * ENTRIES_PER_TABLE {
            let linear = page << HUGE_PAGE_SHIFT;
            assert_eq!(walk(&ram, state.cr3, linear), Some(linear), "{linear:#x}");
        }
    }

    #[test]
    fn addresses_at_and_above_4_gib_are_unmapped() {
        let (state, ram) = load_ok(&kernel(), None, "");
        for linear in [
            4 * GIB,
            4 * GIB + 0x1000,
            1 << 39,
            1 << 47,
            0x0000_7fff_ffff_f000,
        ] {
            assert_eq!(walk(&ram, state.cr3, linear), None, "{linear:#x}");
        }
    }

    #[test]
    fn only_the_first_pml4_and_four_pdpt_entries_are_used() {
        let (state, ram) = load_ok(&kernel(), None, "");
        let table = &ram[state.cr3 as usize..][..PAGE_SIZE as usize];
        assert_ne!(u64_at(table, 0), 0);
        assert!(table[TABLE_ENTRY_SIZE..].iter().all(|&byte| byte == 0));
        let pdpt = &ram[(state.cr3 + PAGE_SIZE) as usize..][..PAGE_SIZE as usize];
        for gib in 0..IDENTITY_MAP_2M_GIB as usize {
            assert_ne!(u64_at(pdpt, gib * TABLE_ENTRY_SIZE) & PTE_PRESENT, 0);
        }
        assert!(
            pdpt[IDENTITY_MAP_2M_GIB as usize * TABLE_ENTRY_SIZE..]
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    #[test]
    fn table_pointers_are_present_writable_and_leaves_are_huge_pages() {
        let (state, ram) = load_ok(&kernel(), None, "");
        let pml4e = u64_at(&ram, state.cr3 as usize);
        assert_eq!(pml4e, (PML4 + PAGE_SIZE) | PTE_PRESENT | PTE_WRITABLE);
        let first_pde = u64_at(&ram, (PML4 + 2 * PAGE_SIZE) as usize);
        assert_eq!(first_pde, PTE_PRESENT | PTE_WRITABLE | PTE_PAGE_SIZE);
        let last_pde = u64_at(
            &ram,
            (PML4 + (2 + IDENTITY_MAP_2M_GIB) * PAGE_SIZE - TABLE_ENTRY_SIZE as u64) as usize,
        );
        assert_eq!(
            last_pde,
            (4 * GIB - (1 << HUGE_PAGE_SHIFT)) | PTE_PRESENT | PTE_WRITABLE | PTE_PAGE_SIZE
        );
    }

    #[test]
    fn a_cpu_started_from_the_boot_state_runs_with_that_gdt_and_paging() {
        let harness = Harness::new(&ENTRY_CODE);
        let cpu = &harness.cpu;
        assert_eq!(cpu.rip, CODE_BASE);
        assert_eq!(cpu.cr3, PML4);
        assert_eq!(cpu.gdtr.base, GDT_ADDR);
        assert_eq!(cpu.gdtr.limit, 31);
        assert_eq!(cpu.seg[crate::cpu::SEG_CS].sel, BOOT_CS);
        assert_eq!(cpu.seg[crate::cpu::SEG_DS].sel, BOOT_DS);
        assert_eq!(cpu.gpr[crate::cpu::GPR_RSI], BOOT_PARAMS);
    }

    #[test]
    fn the_identity_map_lets_a_guest_read_ram_and_faults_above_4_gib() {
        // mov rax, [rbx]; mov rcx, [rdx]; hlt
        let code = [0x48, 0x8b, 0x03, 0x48, 0x8b, 0x0a, 0xf4];
        let mut harness = Harness::new(&code);
        harness.cpu.gpr[RBX] = BOOT_PARAMS + SETUP_SECTS_OFFSET as u64;
        harness.cpu.gpr[RDX] = 4 * GIB;
        let stop = harness.run(10);
        assert_eq!(
            stop,
            Stop::Fault {
                vector: PAGE_FAULT_VECTOR,
                // Not present, supervisor read.
                error_code: Some(0),
                software: false
            }
        );
        assert_eq!(harness.cpu.cr2, 4 * GIB);
        // The first read, through the identity map, returned the boot_params bytes.
        assert_eq!(
            harness.cpu.gpr[0],
            u64_at(
                &harness.cpu.ram[0..],
                BOOT_PARAMS as usize + SETUP_SECTS_OFFSET
            )
        );
    }
}

/// RAM past the PCI hole: the flat buffer spans the hole, and the memory map and page tables
/// say so. The buffers are lazily committed zeroes, so only the pages the loader writes cost
/// anything.
mod high_memory {
    use super::*;
    use crate::ram::{PCI_HOLE_END, PCI_HOLE_START, flat_bytes};

    /// Guest RAM of `memory_gib` GiB as the flat buffer `EmulatedMachine` allocates.
    fn flat(memory_gib: u64) -> usize {
        flat_bytes(memory_gib * GIB) as usize
    }

    fn e820_entries(ram: &[u8]) -> Vec<(u64, u64, u32)> {
        let params = boot_params(ram);
        (0..params[E820_ENTRIES_OFFSET] as usize)
            .map(|index| {
                let at = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
                (
                    u64_at(params, at),
                    u64_at(params, at + 8),
                    u32::from_le_bytes(params[at + 16..at + 20].try_into().unwrap()),
                )
            })
            .collect()
    }

    #[test]
    fn up_to_the_hole_there_is_no_high_entry() {
        let top = flat_bytes(3 * GIB);
        assert_eq!(
            e820_map(top).last().copied(),
            Some((ONE_MIB, 3 * GIB - ONE_MIB, E820_RAM))
        );
    }

    #[test]
    fn ram_beyond_the_hole_is_reported_from_4_gib() {
        let (result, ram) = load_with(&kernel(), None, "", flat(4));
        result.unwrap();
        let entries = e820_entries(&ram);
        assert_eq!(
            entries[2..],
            [
                (ONE_MIB, PCI_HOLE_START - ONE_MIB, E820_RAM),
                (PCI_HOLE_END, GIB, E820_RAM),
            ]
        );
        // Nothing the guest is told is RAM lies in the hole.
        for (base, size, kind) in entries {
            if kind == E820_RAM {
                assert!(crate::ram::is_ram_range(flat(4) as u64, base, size));
            }
        }
    }

    #[test]
    fn the_acpi_map_stays_sorted_with_high_ram() {
        let map = e820_map_with_acpi(flat_bytes(8 * GIB));
        assert!(
            map.windows(2)
                .all(|pair| pair[0].0 + pair[0].1 <= pair[1].0)
        );
        assert_eq!(map.last().copied(), Some((PCI_HOLE_END, 5 * GIB, E820_RAM)));
    }

    #[test]
    fn the_kernel_and_initrd_stay_below_the_hole() {
        // An initrd ceiling of 4 GiB must not tempt the loader into the hole.
        let mut fake = kernel();
        fake.initrd_addr_max = u32::MAX;
        let initrd = vec![0x5au8; MIB];
        let (result, ram) = load_with(&fake, Some(&initrd), "", flat(4));
        result.unwrap();
        let (image, size) = initrd_fields(&ram);
        assert_eq!(size as usize, initrd.len());
        assert!(image as u64 + size as u64 <= PCI_HOLE_START);
        assert_eq!(&ram[image as usize..][..initrd.len()], &initrd[..]);
    }

    #[test]
    fn page_tables_map_high_ram_with_1_gib_pages() {
        let (result, ram) = load_with(&kernel(), None, "", flat(4));
        let state = result.unwrap();
        for linear in [
            0,
            0xc000_0000,
            4 * GIB - 1,
            4 * GIB,
            4 * GIB + 0x1234_5678,
            5 * GIB - 0x1000,
            flat(4) as u64 - 1,
        ] {
            assert_eq!(walk(&ram, state.cr3, linear), Some(linear), "{linear:#x}");
        }
        // The map ends at the last GiB of RAM.
        assert_eq!(walk(&ram, state.cr3, flat(4) as u64), None);
    }

    #[test]
    fn page_tables_reach_the_last_gib_of_the_physical_address_space() {
        // A terabyte guest needs the second PDPT. The tables are written into a small buffer
        // while `ram_top` says how much RAM there is.
        let ram_top = flat(1023) as u64;
        let mut ram = vec![0u8; ONE_MIB as usize];
        write_gdt_and_page_tables(&mut ram, ram_top, BOOT_CODE_DESCRIPTOR).unwrap();
        for linear in [4 * GIB, 511 * GIB + 0x77, 512 * GIB, ram_top - 1] {
            assert_eq!(walk(&ram, PML4, linear), Some(linear), "{linear:#x}");
        }
        assert_eq!(walk(&ram, PML4, ram_top), None);
    }

    #[test]
    fn more_ram_than_the_address_space_is_an_error() {
        let mut ram = vec![0u8; ONE_MIB as usize];
        let ram_top = flat(1024) as u64;
        assert!(write_gdt_and_page_tables(&mut ram, ram_top, BOOT_CODE_DESCRIPTOR).is_err());
    }
}

mod totality {
    use super::*;

    /// Header values at and around the edges of every field the loader does arithmetic on.
    #[test]
    fn load_never_panics_and_only_succeeds_when_everything_fits() {
        let initrd_lengths = [
            None,
            Some(0usize),
            Some(1),
            Some(PAGE_SIZE as usize),
            Some(64 * 1024),
        ];
        let ram_lengths = [
            0usize,
            (PML4 + PAGE_TABLE_PAGES as u64 * PAGE_SIZE) as usize,
            ONE_MIB as usize,
            17 * MIB,
            TEST_RAM,
        ];
        let mut successes = 0;
        for setup_sects in [1u8, 255] {
            for jump_length in [0u8, 0x6a, 0xff] {
                for (relocatable, pref_address) in [
                    (true, 0),
                    (false, 0),
                    (false, ONE_MIB),
                    (false, ALTERNATE_KERNEL_ADDRESS),
                    (false, u64::MAX),
                    (false, u64::MAX - 0xfff),
                ] {
                    for init_size in [0u32, u32::MAX] {
                        for initrd_addr_max in [0u32, 0x7fff_ffff, u32::MAX] {
                            for cmdline_max in [2048u32] {
                                let fake = FakeKernel {
                                    setup_sects,
                                    jump_length,
                                    relocatable,
                                    pref_address,
                                    init_size,
                                    initrd_addr_max,
                                    cmdline_max,
                                    payload_len: Some(FAKE_PAYLOAD_LEN),
                                    ..kernel()
                                };
                                let image = fake.build();
                                for initrd_len in initrd_lengths {
                                    let initrd = initrd_len.map(|len| vec![0xa5u8; len]);
                                    for ram_len in ram_lengths {
                                        let mut ram = vec![0u8; ram_len];
                                        let result = load(&mut ram, &image, initrd.as_deref(), "x");
                                        if let Ok(state) = result {
                                            successes += 1;
                                            assert!((state.rip as usize) < ram_len);
                                            if let Some(initrd) = &initrd {
                                                let (image, size) = initrd_fields(&ram);
                                                assert_eq!(size as usize, initrd.len());
                                                assert!(image as usize + initrd.len() <= ram_len);
                                                assert!(image as u64 <= initrd_addr_max as u64 + 1);
                                                assert_eq!(image as u64 % PAGE_SIZE, 0);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            successes > 0,
            "the sweep must include loadable combinations"
        );
    }

    #[test]
    fn load_linux_is_the_same_loader() {
        let image = kernel().build();
        let mut via_platform = vec![0u8; TEST_RAM];
        let mut direct = vec![0u8; TEST_RAM];
        let a =
            crate::platform::load_linux(&mut via_platform, &image, None, "console=ttyS0").unwrap();
        let b = load(&mut direct, &image, None, "console=ttyS0").unwrap();
        assert_eq!((a.rip, a.rsi, a.cr3), (b.rip, b.rsi, b.cr3));
        assert!(via_platform == direct);
    }
}
