//! Loading `crates/uemulator-firmware` and the page it is handed. The byte layouts here are
//! the contract with the firmware (`docs/plans/19-uefi-firmware.md`); the firmware crate has the same
//! offsets under tests of its own.

use super::{
    BOOT_CODE_DESCRIPTOR, BOOT_CS, BOOT_DS, BootMode, BootState, Error, GDT_ADDR, PML4, e820_map,
    put, write_gdt_and_page_tables,
};
use crate::dev::kv::{KV_PORT_BASE, KV_PORT_COUNT};
#[cfg(freestanding)]
use crate::prelude::*;
use crate::ram;

/// Where the image is placed: the first MiB belongs to legacy structures.
pub(crate) const FIRMWARE_BASE: u64 = 0x10_0000;
/// The page the firmware's `rdi` points at. `boot_params` has no use on this path, so its slot is
/// reused.
pub(crate) const HANDOFF_ADDR: u64 = super::BOOT_PARAMS;

const IMAGE_MAGIC: &[u8; 8] = b"UEMUFW64";
const IMAGE_VERSION: u32 = 1;
/// magic, version u32, entry offset u32, footprint u64: the image's first 24 bytes. The
/// footprint is the image plus its zero-initialised data, which the library reserves.
const IMAGE_HEADER_BYTES: usize = 24;
const IMAGE_ENTRY_OFFSET: usize = 12;
const IMAGE_FOOTPRINT_OFFSET: usize = 16;
const MAX_FOOTPRINT: u64 = 64 << 20;

const HANDOFF_MAGIC: &[u8; 8] = b"UEMUHAND";
const HANDOFF_VERSION: u32 = 1;
pub(crate) const HANDOFF_SIZE: usize = 0x338;

const HANDOFF_VERSION_OFFSET: usize = 0x08;
const HANDOFF_SIZE_OFFSET: usize = 0x0c;
const HANDOFF_FIRMWARE_BASE: usize = 0x10;
const HANDOFF_FIRMWARE_SIZE: usize = 0x18;
const HANDOFF_RAM_TOP: usize = 0x20;
const HANDOFF_RSDP: usize = 0x28;
const HANDOFF_ACPI_BASE: usize = 0x30;
const HANDOFF_ACPI_SIZE: usize = 0x38;
const HANDOFF_UNIX_TIME: usize = 0x40;
const HANDOFF_SERIAL_PORT: usize = 0x48;
const HANDOFF_KV_PORT: usize = 0x4a;
const HANDOFF_KV_PORT_COUNT: usize = 0x4c;
const HANDOFF_PCI_CONFIG_PORT: usize = 0x4e;
const HANDOFF_PCI_ECAM_BASE: usize = 0x50;
const HANDOFF_RAM_REGION_COUNT: usize = 0x58;
const HANDOFF_MMIO_REGION_COUNT: usize = 0x5c;
const HANDOFF_RANDOM_SEED: usize = 0x60;
const RANDOM_SEED_BYTES: usize = 32;
const HANDOFF_RAM_REGIONS: usize = 0x80;
/// base u64, size u64, e820 type u32, pad u32.
const RAM_REGION_BYTES: usize = 24;
const MAX_RAM_REGIONS: usize = 16;
const HANDOFF_MMIO_REGIONS: usize = 0x200;
/// base u64, size u64.
const MMIO_REGION_BYTES: usize = 16;
const MAX_MMIO_REGIONS: usize = 16;
const HANDOFF_TSC_HZ: usize = 0x300;
const HANDOFF_KERNEL_BASE: usize = 0x308;
const HANDOFF_KERNEL_SIZE: usize = 0x310;
const HANDOFF_INITRD_BASE: usize = 0x318;
const HANDOFF_INITRD_SIZE: usize = 0x320;
const HANDOFF_CMDLINE_BASE: usize = 0x328;
const HANDOFF_CMDLINE_SIZE: usize = 0x330;
/// Each payload piece starts on its own page: the firmware reserves them page by page.
const PAYLOAD_ALIGNMENT: u64 = 0x1000;
/// Free RAM the firmware and the kernel's EFI stub still need after the payload is placed, on top
/// of a second copy of the initramfs (the stub reads it through `LoadFile2` into its own buffer):
/// the firmware's pools, the kernel image the firmware loads from its buffer, the decompressed
/// kernel.
const PAYLOAD_HEADROOM_BYTES: u64 = 16 << 20;
/// `rdtsc` ticks per second: the guest clock is in nanoseconds (`Cpu::tsc`).
const TSC_HZ: u64 = 1_000_000_000;

const SERIAL_PORT: u16 = 0x3f8;
const PCI_CONFIG_ADDRESS_PORT: u16 = 0xcf8;

/// The MMIO the machine decodes, as the firmware reports it in the memory map: the virtio BARs
/// (up to `MAX_VIRTIO_FUNCTIONS` of them), the IOAPIC and the local APIC.
const MMIO_REGIONS: [(u64, u64); 3] = [
    (crate::dev::PCI_MMIO_BASE, crate::dev::PCI_MMIO_BYTES),
    (0xfec0_0000, 0x1000),
    (0xfee0_0000, 0x1000),
];

/// What the machine tells the firmware besides its RAM size.
#[derive(Clone)]
pub(crate) struct FirmwareParams {
    pub key_value_store: bool,
    pub unix_time: u64,
    pub random_seed: u64,
    /// Physical address of the ACPI RSDP, 0 when the machine has no ACPI tables.
    pub rsdp: u64,
    /// The guest RAM the ACPI tables occupy (reported as ACPI reclaim memory), (0, 0) for none.
    pub acpi_region: (u64, u64),
}

/// What the firmware boots instead of looking at the disks: placed at the top of the RAM below
/// the PCI hole, with the initramfs highest, then the kernel, then the command line.
pub(crate) struct UefiPayload<'a> {
    pub kernel: &'a [u8],
    pub initrd: Option<&'a [u8]>,
    pub cmdline: &'a str,
}

struct Handoff(Vec<u8>);

impl Handoff {
    fn put_u16(&mut self, offset: usize, value: u16) {
        self.0[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    fn put_u32(&mut self, offset: usize, value: u32) {
        self.0[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn put_u64(&mut self, offset: usize, value: u64) {
        self.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const DOS_MAGIC: &[u8; 2] = b"MZ";
const PE_MAGIC: &[u8; 4] = b"PE\0\0";
/// Offset of the u32 that points at the PE header in the DOS header.
const PE_POINTER_OFFSET: usize = 0x3c;

/// Whether `kernel` starts with the DOS and PE headers of an EFI-stub kernel, which the firmware
/// loads as an ordinary EFI application.
pub(crate) fn has_efi_header(kernel: &[u8]) -> bool {
    let pe_offset = kernel
        .get(PE_POINTER_OFFSET..PE_POINTER_OFFSET + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize);
    kernel.starts_with(DOS_MAGIC)
        && pe_offset
            .and_then(|offset| kernel.get(offset..offset.checked_add(PE_MAGIC.len())?))
            .is_some_and(|bytes| bytes == PE_MAGIC)
}

struct PlacedPayload {
    kernel: (u64, u64),
    /// Size 0 when there is no initramfs.
    initrd: (u64, u64),
    cmdline: (u64, u64),
}

/// Copies the payload into the top of the low RAM, above `floor` (the end of the firmware's footprint).
/// The firmware reserves exactly these ranges (`memory::init`), so they must not overlap each
/// other or anything the library wrote.
fn place_payload(
    ram: &mut [u8],
    payload: &UefiPayload,
    floor: u64,
) -> Result<PlacedPayload, Error> {
    let too_big = || Error::BadConfig("kernel and initramfs do not fit in guest memory".into());
    // The top of the RAM below the PCI hole: there may be more RAM above 4 GiB, but the payload
    // is one contiguous run and that RAM is not contiguous with this one.
    let mut top = ram::low_ram_end(ram.len() as u64) & !(PAYLOAD_ALIGNMENT - 1);
    let mut place = |data: &[u8]| -> Result<(u64, u64), Error> {
        let size = data.len() as u64;
        let base = top
            .checked_sub(size.next_multiple_of(PAYLOAD_ALIGNMENT))
            .filter(|base| *base >= floor)
            .ok_or_else(too_big)?;
        put(ram, base, data)?;
        top = base;
        Ok((base, size))
    };
    let initrd = match payload.initrd {
        Some(data) if !data.is_empty() => place(data)?,
        _ => (0, 0),
    };
    let kernel = place(payload.kernel)?;
    let cmdline = place(payload.cmdline.as_bytes())?;
    if top - floor < initrd.1 + PAYLOAD_HEADROOM_BYTES {
        return Err(too_big());
    }
    Ok(PlacedPayload {
        kernel,
        initrd,
        cmdline,
    })
}

/// Places the firmware image at `FIRMWARE_BASE`, the handoff page and the boot GDT and
/// identity-mapped page tables, and returns the 64-bit entry state (`rdi` = handoff).
pub(crate) fn load_uefi_firmware(
    ram: &mut [u8],
    image: &[u8],
    params: &FirmwareParams,
    acpi: Option<&crate::acpi::AcpiConfig>,
    payload: Option<&UefiPayload>,
) -> Result<BootState, Error> {
    let mut params = params.clone();
    if let Some(config) = acpi {
        params.rsdp = crate::acpi::install(ram, config)?;
        params.acpi_region = (crate::acpi::REGION_BASE, crate::acpi::REGION_BYTES);
    }
    let params = &params;
    let bad = |message: &str| Error::BadConfig(format!("UEFI firmware image: {message}"));
    if image.len() < IMAGE_HEADER_BYTES || &image[..8] != IMAGE_MAGIC {
        return Err(bad("missing UEMUFW64 header"));
    }
    let word = |offset: usize| u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
    if word(8) != IMAGE_VERSION {
        return Err(bad("unsupported header version"));
    }
    let entry = word(IMAGE_ENTRY_OFFSET) as u64;
    let footprint = u64::from_le_bytes(
        image[IMAGE_FOOTPRINT_OFFSET..IMAGE_FOOTPRINT_OFFSET + 8]
            .try_into()
            .unwrap(),
    );
    if entry >= image.len() as u64 || footprint < image.len() as u64 || footprint > MAX_FOOTPRINT {
        return Err(bad("entry or size out of range"));
    }
    let ram_top = ram.len() as u64;
    if FIRMWARE_BASE + footprint > ram::low_ram_end(ram_top) {
        return Err(Error::BadConfig(
            "guest memory too small for firmware".into(),
        ));
    }
    put(ram, FIRMWARE_BASE, image)?;
    let placed = match payload {
        Some(payload) => Some(place_payload(ram, payload, FIRMWARE_BASE + footprint)?),
        None => None,
    };

    let mut handoff = vec![0u8; HANDOFF_SIZE];
    handoff[..8].copy_from_slice(HANDOFF_MAGIC);
    let mut handoff = Handoff(handoff);
    handoff.put_u16(HANDOFF_SERIAL_PORT, SERIAL_PORT);
    handoff.put_u16(
        HANDOFF_KV_PORT,
        if params.key_value_store {
            KV_PORT_BASE
        } else {
            0
        },
    );
    handoff.put_u16(HANDOFF_KV_PORT_COUNT, KV_PORT_COUNT);
    handoff.put_u16(HANDOFF_PCI_CONFIG_PORT, PCI_CONFIG_ADDRESS_PORT);
    handoff.put_u32(HANDOFF_VERSION_OFFSET, HANDOFF_VERSION);
    handoff.put_u32(HANDOFF_SIZE_OFFSET, HANDOFF_SIZE as u32);
    let regions = e820_map(ram_top);
    debug_assert!(regions.len() <= MAX_RAM_REGIONS && MMIO_REGIONS.len() <= MAX_MMIO_REGIONS);
    handoff.put_u32(HANDOFF_RAM_REGION_COUNT, regions.len() as u32);
    handoff.put_u32(HANDOFF_MMIO_REGION_COUNT, MMIO_REGIONS.len() as u32);
    handoff.put_u64(HANDOFF_FIRMWARE_BASE, FIRMWARE_BASE);
    handoff.put_u64(HANDOFF_FIRMWARE_SIZE, footprint);
    handoff.put_u64(HANDOFF_RAM_TOP, ram_top);
    handoff.put_u64(HANDOFF_RSDP, params.rsdp);
    handoff.put_u64(HANDOFF_ACPI_BASE, params.acpi_region.0);
    handoff.put_u64(HANDOFF_ACPI_SIZE, params.acpi_region.1);
    handoff.put_u64(HANDOFF_UNIX_TIME, params.unix_time);
    handoff.put_u64(
        HANDOFF_PCI_ECAM_BASE,
        if acpi.is_some() {
            crate::dev::pci::ECAM_BASE
        } else {
            0
        },
    );
    handoff.put_u64(HANDOFF_TSC_HZ, TSC_HZ);
    if let Some(placed) = placed {
        for (base_at, size_at, (base, size)) in [
            (HANDOFF_KERNEL_BASE, HANDOFF_KERNEL_SIZE, placed.kernel),
            (HANDOFF_INITRD_BASE, HANDOFF_INITRD_SIZE, placed.initrd),
            (HANDOFF_CMDLINE_BASE, HANDOFF_CMDLINE_SIZE, placed.cmdline),
        ] {
            handoff.put_u64(base_at, base);
            handoff.put_u64(size_at, size);
        }
    }
    let mut seed_state = params.random_seed;
    for chunk in 0..RANDOM_SEED_BYTES / 8 {
        handoff.put_u64(HANDOFF_RANDOM_SEED + chunk * 8, splitmix64(&mut seed_state));
    }
    for (index, (base, size, kind)) in regions.iter().enumerate() {
        let at = HANDOFF_RAM_REGIONS + index * RAM_REGION_BYTES;
        handoff.put_u64(at, *base);
        handoff.put_u64(at + 8, *size);
        handoff.put_u32(at + 16, *kind);
    }
    for (index, (base, size)) in MMIO_REGIONS.iter().enumerate() {
        let at = HANDOFF_MMIO_REGIONS + index * MMIO_REGION_BYTES;
        handoff.put_u64(at, *base);
        handoff.put_u64(at + 8, *size);
    }
    let handoff = handoff.0;
    put(ram, HANDOFF_ADDR, &handoff)?;

    let gdt_limit = write_gdt_and_page_tables(ram, ram_top, BOOT_CODE_DESCRIPTOR)?;
    Ok(BootState {
        rip: FIRMWARE_BASE + entry,
        rdi: HANDOFF_ADDR,
        cr3: PML4,
        gdt_base: GDT_ADDR,
        gdt_limit,
        cs: BOOT_CS,
        ds: BOOT_DS,
        mode: BootMode::Long64,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAM_BYTES: usize = 64 << 20;
    const ENTRY: u32 = 0x40;

    fn image(entry: u32, footprint: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 0x100];
        bytes[..8].copy_from_slice(IMAGE_MAGIC);
        bytes[8..12].copy_from_slice(&IMAGE_VERSION.to_le_bytes());
        bytes[IMAGE_ENTRY_OFFSET..IMAGE_ENTRY_OFFSET + 4].copy_from_slice(&entry.to_le_bytes());
        bytes[IMAGE_FOOTPRINT_OFFSET..IMAGE_FOOTPRINT_OFFSET + 8]
            .copy_from_slice(&footprint.to_le_bytes());
        bytes[ENTRY as usize] = 0xf4;
        bytes
    }

    fn params() -> FirmwareParams {
        FirmwareParams {
            key_value_store: true,
            unix_time: 1_700_000_000,
            random_seed: 7,
            rsdp: 0,
            acpi_region: (0, 0),
        }
    }

    fn u64_at(ram: &[u8], at: u64) -> u64 {
        u64::from_le_bytes(ram[at as usize..at as usize + 8].try_into().unwrap())
    }

    fn u16_at(ram: &[u8], at: u64) -> u16 {
        u16::from_le_bytes(ram[at as usize..at as usize + 2].try_into().unwrap())
    }

    fn load(image: &[u8], params: &FirmwareParams) -> Result<(BootState, Vec<u8>), Error> {
        let mut ram = vec![0u8; RAM_BYTES];
        load_uefi_firmware(&mut ram, image, params, None, None).map(|state| (state, ram))
    }

    fn load_payload(
        kernel: &[u8],
        initrd: Option<&[u8]>,
        cmdline: &str,
        ram_bytes: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut ram = vec![0u8; ram_bytes];
        let payload = UefiPayload {
            kernel,
            initrd,
            cmdline,
        };
        load_uefi_firmware(
            &mut ram,
            &image(ENTRY, 0x2000),
            &params(),
            None,
            Some(&payload),
        )
        .map(|_| ram)
    }

    fn range_at(ram: &[u8], base_offset: usize) -> (u64, u64) {
        let at = HANDOFF_ADDR + base_offset as u64;
        (u64_at(ram, at), u64_at(ram, at + 8))
    }

    #[test]
    fn the_payload_is_placed_at_the_top_of_ram_and_described_in_the_handoff() {
        let kernel = vec![0x4b; 5000];
        let initrd = vec![0x49; 9000];
        let ram = load_payload(&kernel, Some(&initrd), "console=hvc0 quiet", RAM_BYTES).unwrap();
        let kernel_range = range_at(&ram, HANDOFF_KERNEL_BASE);
        let initrd_range = range_at(&ram, HANDOFF_INITRD_BASE);
        let cmdline_range = range_at(&ram, HANDOFF_CMDLINE_BASE);
        let bytes = |(base, size): (u64, u64)| ram[base as usize..(base + size) as usize].to_vec();
        assert_eq!(bytes(kernel_range), kernel);
        assert_eq!(bytes(initrd_range), initrd);
        assert_eq!(bytes(cmdline_range), b"console=hvc0 quiet");
        let mut ranges = [kernel_range, initrd_range, cmdline_range];
        ranges.sort();
        for pair in ranges.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{ranges:x?} overlap");
        }
        for (base, size) in ranges {
            assert_eq!(base % PAYLOAD_ALIGNMENT, 0);
            assert!(base + size <= RAM_BYTES as u64);
            assert!(
                base >= FIRMWARE_BASE + 0x2000,
                "below the firmware's footprint"
            );
        }
        assert!(
            initrd_range.0 > kernel_range.0,
            "the initramfs sits at the very top"
        );
    }

    #[test]
    fn no_payload_leaves_the_handoff_fields_zero() {
        let (_, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        assert_eq!(range_at(&ram, HANDOFF_KERNEL_BASE), (0, 0));
        assert_eq!(range_at(&ram, HANDOFF_INITRD_BASE), (0, 0));
        assert_eq!(range_at(&ram, HANDOFF_CMDLINE_BASE), (0, 0));
    }

    #[test]
    fn a_kernel_without_an_initramfs_leaves_that_range_empty() {
        let ram = load_payload(&[1; 100], None, "x", RAM_BYTES).unwrap();
        assert_eq!(range_at(&ram, HANDOFF_INITRD_BASE).1, 0);
        assert_ne!(range_at(&ram, HANDOFF_KERNEL_BASE).1, 0);
    }

    #[test]
    fn a_payload_that_leaves_the_firmware_no_room_to_work_is_refused() {
        // The firmware loads the kernel image and the Linux stub copies the initramfs again, so
        // fitting once is not enough.
        let initrd = vec![0u8; 40 << 20];
        let error = load_payload(&[1; 100], Some(&initrd), "", RAM_BYTES).unwrap_err();
        assert!(error.to_string().contains("do not fit"), "{error}");
        assert!(load_payload(&[1; 100], Some(&vec![0u8; 8 << 20]), "", RAM_BYTES).is_ok());
    }

    #[test]
    fn a_payload_that_does_not_fit_is_refused() {
        let huge = vec![0u8; RAM_BYTES];
        assert!(load_payload(&huge, None, "", RAM_BYTES).is_err());
        assert!(load_payload(&[1; 100], Some(&huge), "", RAM_BYTES).is_err());
    }

    #[test]
    fn starts_at_the_entry_in_long_mode_with_rdi_at_the_handoff() {
        let (state, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        assert_eq!(state.rip, FIRMWARE_BASE + ENTRY as u64);
        assert_eq!(state.rdi, HANDOFF_ADDR);
        assert_eq!(state.mode, BootMode::Long64);
        assert_eq!(ram[(FIRMWARE_BASE + ENTRY as u64) as usize], 0xf4);
        assert_eq!(state.cr3, PML4);
    }

    #[test]
    fn the_handoff_describes_the_machine() {
        let (_, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        let at = |offset: usize| HANDOFF_ADDR + offset as u64;
        assert_eq!(&ram[HANDOFF_ADDR as usize..][..8], HANDOFF_MAGIC);
        assert_eq!(u64_at(&ram, at(HANDOFF_FIRMWARE_BASE)), FIRMWARE_BASE);
        assert_eq!(u64_at(&ram, at(HANDOFF_FIRMWARE_SIZE)), 0x2000);
        assert_eq!(u64_at(&ram, at(HANDOFF_RAM_TOP)), RAM_BYTES as u64);
        assert_eq!(u64_at(&ram, at(HANDOFF_UNIX_TIME)), 1_700_000_000);
        assert_eq!(u64_at(&ram, at(HANDOFF_RSDP)), 0);
        assert_eq!(u16_at(&ram, at(HANDOFF_SERIAL_PORT)), 0x3f8);
        assert_eq!(u16_at(&ram, at(HANDOFF_KV_PORT)), KV_PORT_BASE);
        assert_eq!(u16_at(&ram, at(HANDOFF_PCI_CONFIG_PORT)), 0xcf8);
        assert_eq!(u64_at(&ram, at(HANDOFF_RAM_REGIONS)), 0);
        let high = at(HANDOFF_RAM_REGIONS + 2 * RAM_REGION_BYTES);
        assert_eq!(u64_at(&ram, high), 0x10_0000);
        assert_eq!(u64_at(&ram, high + 8), RAM_BYTES as u64 - 0x10_0000);
        assert_eq!(
            u64_at(&ram, at(HANDOFF_MMIO_REGIONS)),
            crate::dev::PCI_MMIO_BASE
        );
    }

    #[test]
    fn no_store_means_no_kv_port() {
        let mut without = params();
        without.key_value_store = false;
        let (_, ram) = load(&image(ENTRY, 0x2000), &without).unwrap();
        assert_eq!(u16_at(&ram, HANDOFF_ADDR + HANDOFF_KV_PORT as u64), 0);
    }

    #[test]
    fn the_random_seed_depends_on_the_configured_seed() {
        let seed_of = |seed| {
            let mut with = params();
            with.random_seed = seed;
            let (_, ram) = load(&image(ENTRY, 0x2000), &with).unwrap();
            ram[(HANDOFF_ADDR as usize + HANDOFF_RANDOM_SEED)..][..RANDOM_SEED_BYTES].to_vec()
        };
        assert_ne!(seed_of(1), seed_of(2));
        assert_eq!(seed_of(1), seed_of(1));
    }

    #[test]
    fn acpi_details_are_passed_through() {
        let mut with = params();
        with.rsdp = 0xe0000;
        with.acpi_region = (0xe0000, 0x4000);
        let (_, ram) = load(&image(ENTRY, 0x2000), &with).unwrap();
        assert_eq!(u64_at(&ram, HANDOFF_ADDR + HANDOFF_RSDP as u64), 0xe0000);
        assert_eq!(
            u64_at(&ram, HANDOFF_ADDR + HANDOFF_ACPI_SIZE as u64),
            0x4000
        );
    }

    #[test]
    fn malformed_images_are_refused() {
        let mut wrong_magic = image(ENTRY, 0x2000);
        wrong_magic[0] = b'X';
        assert!(load(&wrong_magic, &params()).is_err());
        assert!(
            load(&image(0x100, 0x2000), &params()).is_err(),
            "entry past the image"
        );
        assert!(
            load(&image(ENTRY, 0x10), &params()).is_err(),
            "footprint below the image"
        );
        assert!(load(&image(ENTRY, u64::MAX), &params()).is_err());
        assert!(
            load(&image(ENTRY, RAM_BYTES as u64), &params()).is_err(),
            "does not fit"
        );
        assert!(load(&[0; 4], &params()).is_err());
    }
}
