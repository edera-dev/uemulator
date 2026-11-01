//! The page the library leaves at the firmware's `rdi` (layout in `docs/plans/19-uefi-firmware.md`,
//! written by `crates/uemulator/src/boot/uefi.rs`). Parsed once into plain values.

use alloc::vec::Vec;

pub const SIZE: usize = 0x358;
const MAGIC: &[u8; 8] = b"UEMUHAND";
/// Version 2 added the framebuffer fields.
const VERSION: u32 = 2;

const VERSION_OFFSET: usize = 0x08;
const FIRMWARE_BASE: usize = 0x10;
const FIRMWARE_SIZE: usize = 0x18;
const RAM_TOP: usize = 0x20;
const RSDP: usize = 0x28;
const ACPI_BASE: usize = 0x30;
const ACPI_SIZE: usize = 0x38;
const UNIX_TIME: usize = 0x40;
const SERIAL_PORT: usize = 0x48;
const KV_PORT: usize = 0x4a;
const PCI_CONFIG_PORT: usize = 0x4e;
const PCI_ECAM_BASE: usize = 0x50;
const RAM_REGION_COUNT: usize = 0x58;
const MMIO_REGION_COUNT: usize = 0x5c;
const RANDOM_SEED: usize = 0x60;
const RANDOM_SEED_BYTES: usize = 32;
const RAM_REGIONS: usize = 0x80;
const RAM_REGION_BYTES: usize = 24;
const MAX_RAM_REGIONS: usize = 16;
const MMIO_REGIONS: usize = 0x200;
const MMIO_REGION_BYTES: usize = 16;
const MAX_MMIO_REGIONS: usize = 16;
const TSC_HZ: usize = 0x300;
const KERNEL_BASE: usize = 0x308;
const KERNEL_SIZE: usize = 0x310;
const INITRD_BASE: usize = 0x318;
const INITRD_SIZE: usize = 0x320;
const CMDLINE_BASE: usize = 0x328;
const CMDLINE_SIZE: usize = 0x330;
const FRAMEBUFFER_BASE: usize = 0x338;
const FRAMEBUFFER_SIZE: usize = 0x340;
const FRAMEBUFFER_WIDTH: usize = 0x348;
const FRAMEBUFFER_HEIGHT: usize = 0x34c;
const FRAMEBUFFER_PIXELS_PER_LINE: usize = 0x350;
/// Every pixel is four bytes, blue, green, red, unused.
pub const FRAMEBUFFER_BYTES_PER_PIXEL: u64 = 4;

/// e820 type 1; everything else in the list is reserved.
pub const RAM_USABLE: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RamRegion {
    pub base: u64,
    pub size: u64,
    pub kind: u32,
}

/// A kernel the library put in RAM for the firmware to boot, as (base, size) byte ranges. The
/// command line is UTF-8 without a NUL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payload {
    pub kernel: (u64, u64),
    pub initrd: Option<(u64, u64)>,
    pub cmdline: (u64, u64),
}

/// The machine's screen as a linear framebuffer in RAM (reserved in the memory map), which the
/// firmware draws its console on and publishes as the Graphics Output Protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Framebuffer {
    pub base: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    pub pixels_per_line: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handoff {
    pub firmware_base: u64,
    pub firmware_size: u64,
    pub ram_top: u64,
    /// Zero when the machine has no ACPI tables.
    pub rsdp: u64,
    pub acpi_base: u64,
    pub acpi_size: u64,
    pub unix_time: u64,
    pub serial_port: u16,
    /// Zero when the machine has no key-value store.
    pub kv_port: u16,
    pub pci_config_port: u16,
    pub pci_ecam_base: u64,
    pub random_seed: [u8; RANDOM_SEED_BYTES],
    pub ram_regions: Vec<RamRegion>,
    /// (base, size) of device MMIO.
    pub mmio_regions: Vec<(u64, u64)>,
    pub tsc_hz: u64,
    /// Present when the machine was given a kernel to boot, which comes before any disk.
    pub payload: Option<Payload>,
    /// Present when the machine has a display.
    pub framebuffer: Option<Framebuffer>,
}

impl Handoff {
    /// Reads the page the library left at `address`. The library and the firmware ship
    /// together, so a bad magic means the machine is not the one this image was built for: stop
    /// with a message rather than guess.
    #[cfg(target_os = "none")]
    pub fn read(address: *const u8) -> Self {
        let page = unsafe { core::slice::from_raw_parts(address, SIZE) };
        match Self::parse(page) {
            Some(handoff) => handoff,
            None => {
                crate::serial::println!("firmware: bad handoff page at {address:p}");
                crate::cpu::halt_forever()
            }
        }
    }

    pub fn parse(page: &[u8]) -> Option<Self> {
        if page.len() < SIZE || &page[..8] != MAGIC {
            return None;
        }
        let u16_at = |at: usize| u16::from_le_bytes(page[at..at + 2].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(page[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(page[at..at + 8].try_into().unwrap());
        if u32_at(VERSION_OFFSET) != VERSION {
            return None;
        }
        let ram_count = (u32_at(RAM_REGION_COUNT) as usize).min(MAX_RAM_REGIONS);
        let mmio_count = (u32_at(MMIO_REGION_COUNT) as usize).min(MAX_MMIO_REGIONS);
        let range = |base: usize, size: usize| (u64_at(base), u64_at(size));
        let payload = (u64_at(KERNEL_SIZE) != 0).then(|| Payload {
            kernel: range(KERNEL_BASE, KERNEL_SIZE),
            initrd: (u64_at(INITRD_SIZE) != 0).then(|| range(INITRD_BASE, INITRD_SIZE)),
            cmdline: range(CMDLINE_BASE, CMDLINE_SIZE),
        });
        let framebuffer = Framebuffer {
            base: u64_at(FRAMEBUFFER_BASE),
            size: u64_at(FRAMEBUFFER_SIZE),
            width: u32_at(FRAMEBUFFER_WIDTH),
            height: u32_at(FRAMEBUFFER_HEIGHT),
            pixels_per_line: u32_at(FRAMEBUFFER_PIXELS_PER_LINE),
        };
        // A framebuffer whose rows do not fit its size would have the firmware draw past it.
        let consistent = framebuffer.width > 0
            && framebuffer.height > 0
            && framebuffer.pixels_per_line >= framebuffer.width
            && (framebuffer.pixels_per_line as u64)
                .checked_mul(framebuffer.height as u64 * FRAMEBUFFER_BYTES_PER_PIXEL)
                .is_some_and(|bytes| bytes <= framebuffer.size);
        Some(Self {
            firmware_base: u64_at(FIRMWARE_BASE),
            firmware_size: u64_at(FIRMWARE_SIZE),
            ram_top: u64_at(RAM_TOP),
            rsdp: u64_at(RSDP),
            acpi_base: u64_at(ACPI_BASE),
            acpi_size: u64_at(ACPI_SIZE),
            unix_time: u64_at(UNIX_TIME),
            serial_port: u16_at(SERIAL_PORT),
            kv_port: u16_at(KV_PORT),
            pci_config_port: u16_at(PCI_CONFIG_PORT),
            pci_ecam_base: u64_at(PCI_ECAM_BASE),
            random_seed: page[RANDOM_SEED..RANDOM_SEED + RANDOM_SEED_BYTES]
                .try_into()
                .unwrap(),
            ram_regions: (0..ram_count)
                .map(|index| {
                    let at = RAM_REGIONS + index * RAM_REGION_BYTES;
                    RamRegion {
                        base: u64_at(at),
                        size: u64_at(at + 8),
                        kind: u32_at(at + 16),
                    }
                })
                .collect(),
            mmio_regions: (0..mmio_count)
                .map(|index| {
                    let at = MMIO_REGIONS + index * MMIO_REGION_BYTES;
                    (u64_at(at), u64_at(at + 8))
                })
                .collect(),
            tsc_hz: u64_at(TSC_HZ),
            payload,
            framebuffer: consistent.then_some(framebuffer),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page() -> Vec<u8> {
        let mut page = vec![0u8; SIZE];
        page[..8].copy_from_slice(MAGIC);
        page[VERSION_OFFSET..VERSION_OFFSET + 4].copy_from_slice(&VERSION.to_le_bytes());
        page[RAM_TOP..RAM_TOP + 8].copy_from_slice(&(64u64 << 20).to_le_bytes());
        page[SERIAL_PORT..SERIAL_PORT + 2].copy_from_slice(&0x3f8u16.to_le_bytes());
        page[RAM_REGION_COUNT..RAM_REGION_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        page[RAM_REGIONS + 8..RAM_REGIONS + 16].copy_from_slice(&0x9_fc00u64.to_le_bytes());
        page[RAM_REGIONS + 16..RAM_REGIONS + 20].copy_from_slice(&RAM_USABLE.to_le_bytes());
        page[MMIO_REGION_COUNT..MMIO_REGION_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        page[MMIO_REGIONS..MMIO_REGIONS + 8].copy_from_slice(&0xc000_0000u64.to_le_bytes());
        page[TSC_HZ..TSC_HZ + 8].copy_from_slice(&1_000_000_000u64.to_le_bytes());
        page
    }

    #[test]
    fn parses_the_fields_the_library_writes() {
        let handoff = Handoff::parse(&page()).unwrap();
        assert_eq!(handoff.ram_top, 64 << 20);
        assert_eq!(handoff.serial_port, 0x3f8);
        assert_eq!(handoff.kv_port, 0);
        assert_eq!(handoff.rsdp, 0);
        assert_eq!(
            handoff.ram_regions,
            [RamRegion {
                base: 0,
                size: 0x9_fc00,
                kind: RAM_USABLE
            }]
        );
        assert_eq!(handoff.mmio_regions, [(0xc000_0000, 0)]);
        assert_eq!(handoff.tsc_hz, 1_000_000_000);
    }

    fn put(page: &mut [u8], at: usize, value: u64) {
        page[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn parses_a_payload() {
        let mut with_kernel = page();
        put(&mut with_kernel, KERNEL_BASE, 0x400_0000);
        put(&mut with_kernel, KERNEL_SIZE, 0x1000);
        put(&mut with_kernel, INITRD_BASE, 0x500_0000);
        put(&mut with_kernel, INITRD_SIZE, 0x2000);
        put(&mut with_kernel, CMDLINE_BASE, 0x3ff_0000);
        put(&mut with_kernel, CMDLINE_SIZE, 12);
        assert_eq!(
            Handoff::parse(&with_kernel).unwrap().payload,
            Some(Payload {
                kernel: (0x400_0000, 0x1000),
                initrd: Some((0x500_0000, 0x2000)),
                cmdline: (0x3ff_0000, 12),
            })
        );
        put(&mut with_kernel, INITRD_SIZE, 0);
        assert_eq!(
            Handoff::parse(&with_kernel)
                .unwrap()
                .payload
                .unwrap()
                .initrd,
            None
        );
    }

    fn put32(page: &mut [u8], at: usize, value: u32) {
        page[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn with_framebuffer(width: u32, height: u32, pixels_per_line: u32, size: u64) -> Vec<u8> {
        let mut page = page();
        put(&mut page, FRAMEBUFFER_BASE, 0xbe00_0000);
        put(&mut page, FRAMEBUFFER_SIZE, size);
        put32(&mut page, FRAMEBUFFER_WIDTH, width);
        put32(&mut page, FRAMEBUFFER_HEIGHT, height);
        put32(&mut page, FRAMEBUFFER_PIXELS_PER_LINE, pixels_per_line);
        page
    }

    #[test]
    fn parses_the_framebuffer() {
        let page = with_framebuffer(1280, 800, 1280, 1280 * 800 * 4);
        assert_eq!(
            Handoff::parse(&page).unwrap().framebuffer,
            Some(Framebuffer {
                base: 0xbe00_0000,
                size: 1280 * 800 * 4,
                width: 1280,
                height: 800,
                pixels_per_line: 1280,
            })
        );
        assert_eq!(Handoff::parse(&self::page()).unwrap().framebuffer, None);
    }

    #[test]
    fn a_framebuffer_whose_rows_do_not_fit_is_ignored() {
        for page in [
            with_framebuffer(1280, 800, 1280, 1280 * 800 * 4 - 1),
            with_framebuffer(1280, 800, 1000, u64::MAX),
            with_framebuffer(0, 800, 1280, u64::MAX),
            with_framebuffer(1280, 800, u32::MAX, 16),
        ] {
            assert_eq!(Handoff::parse(&page).unwrap().framebuffer, None);
        }
    }

    #[test]
    fn no_kernel_means_no_payload() {
        assert_eq!(Handoff::parse(&page()).unwrap().payload, None);
    }

    #[test]
    fn refuses_a_page_that_is_not_ours() {
        let mut wrong_magic = page();
        wrong_magic[0] = 0;
        assert_eq!(Handoff::parse(&wrong_magic), None);
        let mut wrong_version = page();
        wrong_version[VERSION_OFFSET] = 9;
        assert_eq!(Handoff::parse(&wrong_version), None);
        assert_eq!(Handoff::parse(&page()[..SIZE - 1]), None);
    }

    #[test]
    fn region_counts_are_clamped_to_the_page() {
        let mut huge = page();
        huge[RAM_REGION_COUNT..RAM_REGION_COUNT + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Handoff::parse(&huge).unwrap().ram_regions.len(),
            MAX_RAM_REGIONS
        );
    }
}
