//! Loading `crates/uemulator-firmware` and the page it is handed. The byte layouts here are
//! the contract with the firmware (`docs/plans/19-uefi-firmware.md`); the firmware crate has the same
//! offsets under tests of its own.

use super::{
    BOOT_CODE_DESCRIPTOR, BOOT_CS, BOOT_DS, BootMode, BootState, E820_RAM, E820_RESERVED, Error,
    GDT_ADDR, PML4, e820_map, put, write_gdt_and_page_tables,
};
use crate::dev::bootfb::FramebufferRegion;
use crate::dev::kv::{KV_PORT_BASE, KV_PORT_COUNT};
#[cfg(freestanding)]
use crate::prelude::*;
use crate::ram::{self, RamSpan};

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
/// Version 2 added the framebuffer fields at 0x338, version 3 the hypervisor at 0x358, version 4
/// the flags at 0x378 and the TPM at 0x380.
const HANDOFF_VERSION: u32 = 4;
pub(crate) const HANDOFF_SIZE: usize = 0x388;

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
/// The boot framebuffer (`dev/bootfb.rs`): base u64, size u64, width u32, height u32, pixels
/// per scan line u32, pad. Size 0: the machine has no display.
const HANDOFF_FRAMEBUFFER_BASE: usize = 0x338;
const HANDOFF_FRAMEBUFFER_SIZE: usize = 0x340;
const HANDOFF_FRAMEBUFFER_WIDTH: usize = 0x348;
const HANDOFF_FRAMEBUFFER_HEIGHT: usize = 0x34c;
const HANDOFF_FRAMEBUFFER_PIXELS_PER_LINE: usize = 0x350;
/// A hypervisor the firmware boots with the kernel as its dom0, and the hypervisor's own command
/// line. Size 0: no hypervisor.
const HANDOFF_HYPERVISOR_BASE: usize = 0x358;
const HANDOFF_HYPERVISOR_SIZE: usize = 0x360;
const HANDOFF_HYPERVISOR_CMDLINE_BASE: usize = 0x368;
const HANDOFF_HYPERVISOR_CMDLINE_SIZE: usize = 0x370;
const HANDOFF_FLAGS: usize = 0x378;
/// The firmware enforces Secure Boot: it enrolls its default keys if the store has no PK and
/// refuses images that db does not allow.
const HANDOFF_FLAG_SECURE_BOOT: u32 = 1 << 0;
/// Physical address of the TPM's CRB registers, 0 when the machine has no TPM.
const HANDOFF_TPM_BASE: usize = 0x380;
/// The framebuffer starts on a 2 MiB boundary so that the guest can map it with large pages
/// (Linux's `ioremap` of a firmware framebuffer, an OS's direct map) without splitting one
/// around RAM it does own.
const FRAMEBUFFER_ALIGNMENT: u64 = 2 << 20;
const PAGE_BYTES: u64 = 0x1000;
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
    /// Width and height of the boot framebuffer the firmware publishes as its GOP, when the
    /// machine has a display (`framebuffer_region` says where it goes).
    pub framebuffer: Option<(u32, u32)>,
    pub secure_boot: bool,
    /// The machine has a TPM, which the firmware starts and measures the boot into.
    pub tpm: bool,
}

/// Where a machine with `ram_top` bytes of flat RAM keeps a `width` x `height` boot framebuffer:
/// the top of the RAM below the PCI hole, page-rounded and aligned down to
/// `FRAMEBUFFER_ALIGNMENT`. The firmware's payload is placed below it and the memory map reports
/// it as reserved, so the firmware never allocates it and the OS never takes it for RAM. The
/// machine computes the same region for the device that scans it, so this must stay a pure
/// function of its arguments.
pub(crate) fn framebuffer_region(ram_top: u64, width: u32, height: u32) -> FramebufferRegion {
    let region = FramebufferRegion {
        base: 0,
        width,
        height,
    };
    let bytes = region.pixel_bytes().next_multiple_of(PAGE_BYTES);
    let base = ram::low_ram_end(ram_top).saturating_sub(bytes) & !(FRAMEBUFFER_ALIGNMENT - 1);
    FramebufferRegion { base, ..region }
}

/// Bytes of RAM the region takes out of the map: whole pages.
fn reserved_bytes(region: &FramebufferRegion) -> u64 {
    region.pixel_bytes().next_multiple_of(PAGE_BYTES)
}

/// `regions` (an e820 map) with `base..base + size` cut out of the RAM entry holding it and
/// listed as reserved instead.
fn reserve(regions: Vec<(u64, u64, u32)>, base: u64, size: u64) -> Vec<(u64, u64, u32)> {
    let end = base + size;
    let mut out = Vec::with_capacity(regions.len() + 2);
    for (start, length, kind) in regions {
        let stop = start + length;
        if kind != E820_RAM || end <= start || base >= stop {
            out.push((start, length, kind));
            continue;
        }
        if base > start {
            out.push((start, base - start, E820_RAM));
        }
        out.push((base, size, E820_RESERVED));
        if stop > end {
            out.push((end, stop - end, E820_RAM));
        }
    }
    out
}

/// What the firmware boots instead of looking at the disks: placed at the top of the RAM below
/// the PCI hole, with the initramfs highest, then the kernel, then the command line, then the
/// hypervisor and its command line.
pub(crate) struct UefiPayload<'a> {
    pub kernel: &'a [u8],
    pub initrd: Option<&'a [u8]>,
    pub cmdline: &'a str,
    /// The hypervisor image (decompressed: the firmware cannot inflate) and its command line.
    /// The kernel is then the hypervisor's dom0 and needs no EFI stub.
    pub hypervisor: Option<(&'a [u8], &'a str)>,
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
    /// Size 0 when there is no hypervisor.
    hypervisor: (u64, u64),
    hypervisor_cmdline: (u64, u64),
}

/// Copies the payload into the top of the low RAM, above `floor` (the end of the firmware's footprint).
/// The firmware reserves exactly these ranges (`memory::init`), so they must not overlap each
/// other or anything the library wrote.
fn place_payload(
    ram: &mut RamSpan<'_>,
    payload: &UefiPayload,
    floor: u64,
    ceiling: u64,
) -> Result<PlacedPayload, Error> {
    let too_big = || Error::BadConfig("kernel and initramfs do not fit in guest memory".into());
    // Below `ceiling`, the top of the RAM below the PCI hole or the framebuffer: there may be
    // more RAM above 4 GiB, but the payload is one contiguous run and that RAM is not contiguous
    // with this one.
    let mut top = ceiling & !(PAYLOAD_ALIGNMENT - 1);
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
    let (hypervisor, hypervisor_cmdline) = match payload.hypervisor {
        Some((image, text)) => (place(image)?, place(text.as_bytes())?),
        None => ((0, 0), (0, 0)),
    };
    // A hypervisor adds its own loaded copy, and xen.efi reads the kernel and the initramfs
    // into buffers of its own through its media loaders (the initramfs copy is counted already).
    let copies = initrd.1 + hypervisor.1 + if hypervisor.1 != 0 { kernel.1 } else { 0 };
    if top.saturating_sub(floor) < copies + PAYLOAD_HEADROOM_BYTES {
        return Err(too_big());
    }
    Ok(PlacedPayload {
        kernel,
        initrd,
        cmdline,
        hypervisor,
        hypervisor_cmdline,
    })
}

/// Places the firmware image at `FIRMWARE_BASE`, the handoff page and the boot GDT and
/// identity-mapped page tables, and returns the 64-bit entry state (`rdi` = handoff).
pub(crate) fn load_uefi_firmware<'r>(
    ram: impl Into<RamSpan<'r>>,
    image: &[u8],
    params: &FirmwareParams,
    acpi: Option<&crate::acpi::AcpiConfig>,
    payload: Option<&UefiPayload>,
) -> Result<BootState, Error> {
    let mut ram = ram.into();
    let mut params = params.clone();
    if let Some(config) = acpi {
        params.rsdp = crate::acpi::install(ram.reborrow(), config)?;
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
    put(&mut ram, FIRMWARE_BASE, image)?;
    let framebuffer = params
        .framebuffer
        .map(|(width, height)| framebuffer_region(ram_top, width, height));
    let ceiling = framebuffer.map_or(ram::low_ram_end(ram_top), |region| region.base);
    if FIRMWARE_BASE + footprint > ceiling {
        return Err(Error::BadConfig(
            "guest memory too small for firmware and framebuffer".into(),
        ));
    }
    let placed = match payload {
        Some(payload) => Some(place_payload(
            &mut ram,
            payload,
            FIRMWARE_BASE + footprint,
            ceiling,
        )?),
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
    let regions = match &framebuffer {
        Some(region) => reserve(e820_map(ram_top), region.base, reserved_bytes(region)),
        None => e820_map(ram_top),
    };
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
    handoff.put_u32(
        HANDOFF_FLAGS,
        if params.secure_boot {
            HANDOFF_FLAG_SECURE_BOOT
        } else {
            0
        },
    );
    if params.tpm {
        handoff.put_u64(HANDOFF_TPM_BASE, crate::dev::tpm::TPM_CRB_BASE);
    }
    if let Some(region) = &framebuffer {
        handoff.put_u64(HANDOFF_FRAMEBUFFER_BASE, region.base);
        handoff.put_u64(HANDOFF_FRAMEBUFFER_SIZE, region.pixel_bytes());
        handoff.put_u32(HANDOFF_FRAMEBUFFER_WIDTH, region.width);
        handoff.put_u32(HANDOFF_FRAMEBUFFER_HEIGHT, region.height);
        handoff.put_u32(HANDOFF_FRAMEBUFFER_PIXELS_PER_LINE, region.width);
    }
    if let Some(placed) = placed {
        for (base_at, size_at, (base, size)) in [
            (HANDOFF_KERNEL_BASE, HANDOFF_KERNEL_SIZE, placed.kernel),
            (HANDOFF_INITRD_BASE, HANDOFF_INITRD_SIZE, placed.initrd),
            (HANDOFF_CMDLINE_BASE, HANDOFF_CMDLINE_SIZE, placed.cmdline),
            (
                HANDOFF_HYPERVISOR_BASE,
                HANDOFF_HYPERVISOR_SIZE,
                placed.hypervisor,
            ),
            (
                HANDOFF_HYPERVISOR_CMDLINE_BASE,
                HANDOFF_HYPERVISOR_CMDLINE_SIZE,
                placed.hypervisor_cmdline,
            ),
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
    put(&mut ram, HANDOFF_ADDR, &handoff)?;

    let gdt_limit = write_gdt_and_page_tables(ram.reborrow(), ram_top, BOOT_CODE_DESCRIPTOR)?;
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
            framebuffer: None,
            secure_boot: false,
            tpm: false,
        }
    }

    #[test]
    fn secure_boot_and_the_tpm_are_off_unless_asked_for() {
        let field = |offset: usize| HANDOFF_ADDR + offset as u64;
        let (_, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        assert_eq!(u32_at(&ram, field(HANDOFF_VERSION_OFFSET)), 4);
        assert_eq!(u32_at(&ram, field(HANDOFF_FLAGS)), 0);
        assert_eq!(u64_at(&ram, field(HANDOFF_TPM_BASE)), 0);
        let both = FirmwareParams {
            secure_boot: true,
            tpm: true,
            ..params()
        };
        let (_, ram) = load(&image(ENTRY, 0x2000), &both).unwrap();
        assert_eq!(u32_at(&ram, field(HANDOFF_FLAGS)), HANDOFF_FLAG_SECURE_BOOT);
        assert_eq!(
            u64_at(&ram, field(HANDOFF_TPM_BASE)),
            crate::dev::tpm::TPM_CRB_BASE
        );
    }

    fn u32_at(ram: &[u8], at: u64) -> u32 {
        u32::from_le_bytes(ram[at as usize..at as usize + 4].try_into().unwrap())
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
            hypervisor: None,
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
    fn a_hypervisor_and_its_command_line_are_placed_below_the_kernel() {
        let kernel = vec![0x4b; 5000];
        let hypervisor = vec![0x58; 7000];
        let mut ram = vec![0u8; RAM_BYTES];
        let payload = UefiPayload {
            kernel: &kernel,
            initrd: None,
            cmdline: "console=hvc0",
            hypervisor: Some((&hypervisor, "console=com1")),
        };
        load_uefi_firmware(
            &mut ram,
            &image(ENTRY, 0x2000),
            &params(),
            None,
            Some(&payload),
        )
        .unwrap();
        let bytes = |(base, size): (u64, u64)| ram[base as usize..(base + size) as usize].to_vec();
        let hypervisor_range = range_at(&ram, HANDOFF_HYPERVISOR_BASE);
        assert_eq!(bytes(hypervisor_range), hypervisor);
        assert_eq!(
            bytes(range_at(&ram, HANDOFF_HYPERVISOR_CMDLINE_BASE)),
            b"console=com1"
        );
        assert!(hypervisor_range.0 < range_at(&ram, HANDOFF_KERNEL_BASE).0);
        assert_eq!(hypervisor_range.0 % PAYLOAD_ALIGNMENT, 0);
        // Without one the fields stay zero, which is how the firmware tells.
        let ram = load_payload(&kernel, None, "x", RAM_BYTES).unwrap();
        assert_eq!(range_at(&ram, HANDOFF_HYPERVISOR_BASE), (0, 0));
    }

    /// (base, size, e820 type) of every RAM region the handoff lists.
    fn ram_regions(ram: &[u8]) -> Vec<(u64, u64, u32)> {
        let count = u32_at(ram, HANDOFF_ADDR + HANDOFF_RAM_REGION_COUNT as u64) as usize;
        (0..count)
            .map(|index| {
                let at = HANDOFF_ADDR + (HANDOFF_RAM_REGIONS + index * RAM_REGION_BYTES) as u64;
                (u64_at(ram, at), u64_at(ram, at + 8), u32_at(ram, at + 16))
            })
            .collect()
    }

    const FRAMEBUFFER_WIDTH: u32 = 1280;
    const FRAMEBUFFER_HEIGHT: u32 = 800;

    fn with_framebuffer() -> FirmwareParams {
        FirmwareParams {
            framebuffer: Some((FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT)),
            ..params()
        }
    }

    #[test]
    fn the_framebuffer_is_described_in_the_handoff_and_reserved_in_the_map() {
        let (_, ram) = load(&image(ENTRY, 0x2000), &with_framebuffer()).unwrap();
        let region = framebuffer_region(RAM_BYTES as u64, FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
        let field = |offset: usize| HANDOFF_ADDR + offset as u64;
        assert_eq!(u64_at(&ram, field(HANDOFF_FRAMEBUFFER_BASE)), region.base);
        assert_eq!(
            u64_at(&ram, field(HANDOFF_FRAMEBUFFER_SIZE)),
            (FRAMEBUFFER_WIDTH * FRAMEBUFFER_HEIGHT * 4) as u64
        );
        assert_eq!(
            u32_at(&ram, field(HANDOFF_FRAMEBUFFER_WIDTH)),
            FRAMEBUFFER_WIDTH
        );
        assert_eq!(
            u32_at(&ram, field(HANDOFF_FRAMEBUFFER_HEIGHT)),
            FRAMEBUFFER_HEIGHT
        );
        assert_eq!(
            u32_at(&ram, field(HANDOFF_FRAMEBUFFER_PIXELS_PER_LINE)),
            FRAMEBUFFER_WIDTH
        );
        assert_eq!(region.base % FRAMEBUFFER_ALIGNMENT, 0);
        let regions = ram_regions(&ram);
        let reserved = regions
            .iter()
            .find(|(base, _, _)| *base == region.base)
            .expect("a region at the framebuffer");
        assert_eq!(reserved.2, E820_RESERVED);
        assert!(reserved.1 >= region.pixel_bytes());
        for (base, size, kind) in &regions {
            let overlaps = *base < region.base + reserved.1 && region.base < base + size;
            assert!(!overlaps || *kind == E820_RESERVED, "{regions:x?}");
        }
        let ram_total: u64 = regions
            .iter()
            .filter(|(_, _, kind)| *kind == E820_RAM)
            .map(|(_, size, _)| size)
            .sum();
        assert!(ram_total < RAM_BYTES as u64 - region.pixel_bytes());
    }

    #[test]
    fn without_a_display_the_framebuffer_fields_are_zero_and_the_map_is_unchanged() {
        let (_, ram) = load(&image(ENTRY, 0x2000), &params()).unwrap();
        assert_eq!(range_at(&ram, HANDOFF_FRAMEBUFFER_BASE), (0, 0));
        assert_eq!(ram_regions(&ram), e820_map(RAM_BYTES as u64));
    }

    #[test]
    fn the_payload_goes_below_the_framebuffer() {
        let mut ram = vec![0u8; RAM_BYTES];
        let initrd = vec![0x49; 9000];
        let payload = UefiPayload {
            kernel: &[0x4b; 5000],
            initrd: Some(&initrd),
            cmdline: "x",
            hypervisor: None,
        };
        let parameters = with_framebuffer();
        load_uefi_firmware(
            &mut ram,
            &image(ENTRY, 0x2000),
            &parameters,
            None,
            Some(&payload),
        )
        .unwrap();
        let region = framebuffer_region(RAM_BYTES as u64, FRAMEBUFFER_WIDTH, FRAMEBUFFER_HEIGHT);
        let (initrd_base, initrd_size) = range_at(&ram, HANDOFF_INITRD_BASE);
        assert!(initrd_base + initrd_size <= region.base);
    }

    #[test]
    fn reserving_splits_the_ram_entry_around_the_hole() {
        let map = vec![(0, 0x1000, E820_RAM), (0x10_0000, 0x90_0000, E820_RAM)];
        assert_eq!(
            reserve(map, 0x40_0000, 0x10_0000),
            vec![
                (0, 0x1000, E820_RAM),
                (0x10_0000, 0x30_0000, E820_RAM),
                (0x40_0000, 0x10_0000, E820_RESERVED),
                (0x50_0000, 0x50_0000, E820_RAM),
            ]
        );
        let at_the_top = vec![(0x10_0000, 0x90_0000, E820_RAM)];
        assert_eq!(
            reserve(at_the_top, 0x90_0000, 0x10_0000),
            vec![
                (0x10_0000, 0x80_0000, E820_RAM),
                (0x90_0000, 0x10_0000, E820_RESERVED)
            ]
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
