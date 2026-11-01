use crate::acpi::{self, AcpiConfig};
use crate::machine::Error;
#[cfg(freestanding)]
use crate::prelude::*;
use crate::ram;

/// Where a relocatable kernel's protected-mode part is loaded. 16 MiB is the Debian kernel's
/// `pref_address` and a multiple of its 2 MiB `kernel_alignment`; a non-relocatable kernel is
/// loaded at its own `pref_address` instead.
pub(crate) const KERNEL_LOAD: u64 = 0x100_0000;

// The low-memory layout. Everything here sits inside the first e820 usable range (below
// `LOW_RAM_END`) and the ranges must not overlap: GDT 0x6000..0x6020, boot_params
// 0x7000..0x8000, page tables 0x9000..0x10000, command line from 0x20000. The UEFI firmware runs
// its application-processor start-up stub from 0x8000..0x9000 (`processors.rs` there).
const GDT_ADDR: u64 = 0x6000;
const BOOT_PARAMS: u64 = 0x7000;
const PML4: u64 = 0x9000;
const CMDLINE_ADDR: u64 = 0x2_0000;

/// The conventional 639 KiB end of low RAM in a PC memory map; the e820 map reports the rest of
/// the first MiB as reserved.
const LOW_RAM_END: u64 = 0x9_fc00;
const ONE_MIB: u64 = 0x10_0000;

/// GDT selectors the 64-bit boot protocol requires (`__BOOT_CS` and `__BOOT_DS` in the kernel),
/// i.e. descriptor indices 2 and 3 of `GDT_ADDR`'s table.
const BOOT_CS: u16 = 0x10;
const BOOT_DS: u16 = 0x18;

/// Flat 64-bit code segment: limit 0xfffff with 4 KiB granularity, access 0x9a (present, DPL 0,
/// code, execute/read), flags G|L. `Cpu::new` loads the cached segment state for `cs` with the
/// same access byte; keep the two in sync.
const BOOT_CODE_DESCRIPTOR: u64 = 0x00af_9a00_0000_ffff;
/// The same selector for a 32-bit kernel: flags G|D instead of G|L.
const BOOT_CODE32_DESCRIPTOR: u64 = 0x00cf_9a00_0000_ffff;
/// Flat data segment: access 0x92 (present, DPL 0, data, read/write), flags G|D/B. Matches the
/// cached `ds`/`es`/`ss` state in `Cpu::new`.
const BOOT_DATA_DESCRIPTOR: u64 = 0x00cf_9200_0000_ffff;

const PAGE_SIZE: u64 = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;
const SECTOR_SIZE: usize = 512;

// Page-table entry bits (SDM Vol. 3A, IA-32e paging). Table pointers are present|writable; leaf entries
// are 2 MiB pages (PS set in a PDE).
const PTE_PRESENT: u64 = 1;
const PTE_WRITABLE: u64 = 2;
const PTE_PAGE_SIZE: u64 = 0x80;
const TABLE_ENTRY_SIZE: usize = 8;
const ENTRIES_PER_TABLE: u64 = 512;
const HUGE_PAGE_SHIFT: u32 = 21;
const GIB_SHIFT: u32 = 30;
/// The first 4 GiB are mapped with 2 MiB pages (four page directories of 512 entries each): the
/// RAM below the PCI hole and the MMIO window in it.
const IDENTITY_MAP_2M_GIB: u64 = 4;
/// Everything above 4 GiB is mapped with 1 GiB pages straight from the PDPT entries, so that any
/// amount of RAM needs no more page-table pages than the 512 GiB of one PDPT, plus a second PDPT
/// for the last half of the 40-bit physical address space. (The CPU advertises pdpe1gb.)
const PDPT_GIB: u64 = ENTRIES_PER_TABLE;
/// Physical address space the CPU decodes (`PHYSICAL_ADDRESS_BITS` = 40), in GiB.
const PHYSICAL_SPACE_GIB: u64 = 1 << 10;
/// PML4, the first PDPT, four page directories, then the second PDPT, back to back from `PML4`.
const PAGE_TABLE_PAGES: usize = 2 + IDENTITY_MAP_2M_GIB as usize + 1;

// Offsets into the bzImage's setup header, which is also where `struct boot_params` keeps it
// (Documentation/arch/x86/boot.rst).
const SETUP_SECTS_OFFSET: usize = 0x1f1;
/// Second byte of the header's `jump` field: the length of the header after the jump.
const JUMP_LENGTH_OFFSET: usize = 0x201;
const HEADER_MAGIC_OFFSET: usize = 0x202;
const VERSION_OFFSET: usize = 0x206;
const TYPE_OF_LOADER_OFFSET: usize = 0x210;
const LOADFLAGS_OFFSET: usize = 0x211;
const RAMDISK_IMAGE_OFFSET: usize = 0x218;
const RAMDISK_SIZE_OFFSET: usize = 0x21c;
const CMD_LINE_PTR_OFFSET: usize = 0x228;
const INITRD_ADDR_MAX_OFFSET: usize = 0x22c;
const RELOCATABLE_KERNEL_OFFSET: usize = 0x234;
const XLOADFLAGS_OFFSET: usize = 0x236;
const CMDLINE_SIZE_OFFSET: usize = 0x238;
const PREF_ADDRESS_OFFSET: usize = 0x258;
const INIT_SIZE_OFFSET: usize = 0x260;
/// Past `handover_offset` (0x264, 4 bytes): the shortest image that has every field we read.
const SETUP_HEADER_MIN_LEN: usize = 0x268;
/// `edd_mbr_sig_buffer`: the first `boot_params` field after the setup header.
const SETUP_HEADER_END_LIMIT: usize = 0x290;

const HEADER_MAGIC: &[u8; 4] = b"HdrS";
/// `setup_sects` of 0 means 4 (boot.rst).
const DEFAULT_SETUP_SECTS: u8 = 4;
/// Boot protocol 2.15, what the target kernel speaks (`docs/plans/02-boot.md`); older ones are
/// rejected rather than checked field by field.
const MIN_PROTOCOL_VERSION: u16 = 0x020f;
const LOADFLAGS_LOADED_HIGH: u8 = 1 << 0;
const LOADFLAGS_CAN_USE_HEAP: u8 = 1 << 7;
const XLF_KERNEL_64: u16 = 1 << 0;
/// "Undefined bootloader" in `type_of_loader`.
const LOADER_TYPE_UNDEFINED: u8 = 0xff;
/// A 64-bit kernel enters at this offset into its protected-mode image (`startup_64`), not at
/// the start of it (which is the 32-bit entry).
const ENTRY_64_OFFSET: u64 = 0x200;

// `struct boot_params` (the "zero page") fields.
const BOOT_PARAMS_SIZE: usize = 4096;
const E820_ENTRIES_OFFSET: usize = 0x1e8;
const E820_TABLE_OFFSET: usize = 0x2d0;
/// `struct boot_e820_entry` is packed: addr u64, size u64, type u32.
const E820_ENTRY_SIZE: usize = 20;
pub(crate) const E820_RAM: u32 = 1;
pub(crate) const E820_RESERVED: u32 = 2;
pub(crate) const E820_ACPI: u32 = 3;
/// `boot_params.acpi_rsdp_addr` (boot protocol 2.14): where the kernel finds the RSDP without
/// scanning for it.
const ACPI_RSDP_ADDR_OFFSET: usize = 0x70;

#[derive(Debug, Clone)]
pub(crate) struct SetupHeader {
    pub setup_sects: u8,
    pub version: u16,
    pub loadflags: u8,
    pub xloadflags: u16,
    pub relocatable: bool,
    pub pref_address: u64,
    pub init_size: u32,
    pub cmdline_max: u32,
    pub initrd_addr_max: u32,
}

impl SetupHeader {
    pub fn parse(image: &[u8]) -> Result<Self, Error> {
        let bad = |message: &str| Error::BadKernel(message.to_string());
        if image.len() < SETUP_HEADER_MIN_LEN
            || &image[HEADER_MAGIC_OFFSET..HEADER_MAGIC_OFFSET + 4] != HEADER_MAGIC
        {
            return Err(bad("missing HdrS magic"));
        }
        let u16_at = |offset| u16::from_le_bytes(image[offset..offset + 2].try_into().unwrap());
        let u32_at = |offset| u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(image[offset..offset + 8].try_into().unwrap());
        let header = SetupHeader {
            setup_sects: if image[SETUP_SECTS_OFFSET] == 0 {
                DEFAULT_SETUP_SECTS
            } else {
                image[SETUP_SECTS_OFFSET]
            },
            version: u16_at(VERSION_OFFSET),
            loadflags: image[LOADFLAGS_OFFSET],
            xloadflags: u16_at(XLOADFLAGS_OFFSET),
            relocatable: image[RELOCATABLE_KERNEL_OFFSET] != 0,
            pref_address: u64_at(PREF_ADDRESS_OFFSET),
            init_size: u32_at(INIT_SIZE_OFFSET),
            cmdline_max: u32_at(CMDLINE_SIZE_OFFSET),
            initrd_addr_max: u32_at(INITRD_ADDR_MAX_OFFSET),
        };
        if header.version < MIN_PROTOCOL_VERSION {
            return Err(bad("boot protocol older than 2.15"));
        }
        if header.loadflags & LOADFLAGS_LOADED_HIGH == 0 {
            return Err(bad("not a bzImage (LOADED_HIGH unset)"));
        }
        if header.protected_offset() >= image.len() {
            return Err(bad("truncated image"));
        }
        Ok(header)
    }

    /// File offset of the protected-mode kernel: the boot sector plus `setup_sects` sectors of
    /// real-mode setup code come first.
    pub fn protected_offset(&self) -> usize {
        (self.setup_sects as usize + 1) * SECTOR_SIZE
    }
}

/// How the CPU starts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BootMode {
    /// 64-bit mode with paging on (the 64-bit Linux boot protocol).
    #[default]
    Long64,
    /// 32-bit protected mode, flat segments, paging off (the 32-bit Linux boot protocol, for a
    /// kernel without a 64-bit entry point).
    Protected32,
    /// Real mode at the reset vector (a firmware image).
    Real16,
    /// What a BIOS leaves behind: real mode at 0000:7C00 with the disk's first sector loaded
    /// there (`cpu::bios`).
    Bios,
}

/// The memory map every boot path reports, as (base, size, type): the low range, which the
/// kernel needs usable for its real-mode trampoline, then the reserved video and ROM hole, then
/// the RAM above 1 MiB up to the PCI hole, then whatever RAM continues at 4 GiB. `ram_top` is
/// the end of the flat RAM buffer (`ram::flat_bytes`).
pub(crate) fn e820_map(ram_top: u64) -> Vec<(u64, u64, u32)> {
    let mut map = vec![
        (0, LOW_RAM_END, E820_RAM),
        (LOW_RAM_END, ONE_MIB - LOW_RAM_END, E820_RESERVED),
        (ONE_MIB, ram::low_ram_end(ram_top) - ONE_MIB, E820_RAM),
    ];
    map.extend(high_ram_entry(ram_top));
    map
}

/// The e820 entry for the RAM above 4 GiB, if there is any.
fn high_ram_entry(ram_top: u64) -> Option<(u64, u64, u32)> {
    let high_end = ram::high_ram_end(ram_top);
    (high_end != 0).then(|| (ram::PCI_HOLE_END, high_end - ram::PCI_HOLE_END, E820_RAM))
}

/// `e820_map` with the ACPI tables marked as ACPI data and the ECAM window reserved: the
/// regions Linux checks before it trusts MCFG and the tables.
pub(crate) fn e820_map_with_acpi(ram_top: u64) -> Vec<(u64, u64, u32)> {
    let (ecam_base, ecam_bytes) = acpi::ECAM_RANGE;
    let mut map = vec![
        (0, LOW_RAM_END, E820_RAM),
        (LOW_RAM_END, acpi::REGION_BASE - LOW_RAM_END, E820_RESERVED),
        (acpi::REGION_BASE, acpi::REGION_BYTES, E820_ACPI),
        (
            acpi::REGION_BASE + acpi::REGION_BYTES,
            ONE_MIB - (acpi::REGION_BASE + acpi::REGION_BYTES),
            E820_RESERVED,
        ),
        (ONE_MIB, ram::low_ram_end(ram_top) - ONE_MIB, E820_RAM),
        (ecam_base, ecam_bytes, E820_RESERVED),
    ];
    // After the ECAM, which is below 4 GiB: the table stays sorted.
    map.extend(high_ram_entry(ram_top));
    map
}

/// Register state of a Linux entry: in 64-bit mode paging on with `cr3`; in 32-bit mode paging
/// off. The GDT is at `gdt_base`, `cs` and `ds` select its flat code and data descriptors.
#[derive(Debug, Clone, Copy, Default)]
pub struct BootState {
    pub rip: u64,
    /// Physical address of `boot_params`; the 64-bit boot protocol passes it in `rsi`.
    pub rsi: u64,
    /// First argument of the UEFI firmware entry (`FirmwareHandoff`); zero otherwise.
    pub rdi: u64,
    pub cr3: u64,
    pub gdt_base: u64,
    pub gdt_limit: u16,
    pub cs: u16,
    pub ds: u16,
    pub mode: BootMode,
}

fn put(ram: &mut [u8], addr: u64, data: &[u8]) -> Result<(), Error> {
    let too_small = || Error::BadConfig("guest memory too small".into());
    let start = usize::try_from(addr).map_err(|_| too_small())?;
    let end = start.checked_add(data.len()).ok_or_else(too_small)?;
    ram.get_mut(start..end)
        .ok_or_else(too_small)?
        .copy_from_slice(data);
    Ok(())
}

/// Writes the boot GDT (null, null, `code_descriptor`, flat data) and the identity map where
/// `BootState` says they are: the first 4 GiB, and every GiB above it up to the end of RAM.
/// `ram_top` is the end of the flat RAM buffer (`ram::flat_bytes`), which `ram` itself may be
/// shorter than only in tests. Returns the GDT limit.
fn write_gdt_and_page_tables(
    ram: &mut [u8],
    ram_top: u64,
    code_descriptor: u64,
) -> Result<u16, Error> {
    // Entries 0 and 1 stay null so the code and data descriptors land on the selectors
    // `BOOT_CS` and `BOOT_DS`.
    let gdt: [u64; 4] = [0, 0, code_descriptor, BOOT_DATA_DESCRIPTOR];
    let gdt_bytes: Vec<u8> = gdt.iter().flat_map(|entry| entry.to_le_bytes()).collect();
    put(ram, GDT_ADDR, &gdt_bytes)?;

    let first_pdpt = PML4 + PAGE_SIZE;
    let first_page_directory = first_pdpt + PAGE_SIZE;
    let second_pdpt = first_page_directory + IDENTITY_MAP_2M_GIB * PAGE_SIZE;
    let mapped_gib = ram::high_ram_end(ram_top)
        .div_ceil(1 << GIB_SHIFT)
        .max(IDENTITY_MAP_2M_GIB);
    if mapped_gib > PHYSICAL_SPACE_GIB {
        return Err(Error::BadConfig("guest memory too large".into()));
    }
    let mut tables = vec![0u8; PAGE_TABLE_PAGES * PAGE_SIZE as usize];
    let mut set_entry = |table: u64, index: u64, value: u64| {
        let at = (table - PML4) as usize + index as usize * TABLE_ENTRY_SIZE;
        tables[at..at + TABLE_ENTRY_SIZE].copy_from_slice(&value.to_le_bytes());
    };
    set_entry(PML4, 0, first_pdpt | PTE_PRESENT | PTE_WRITABLE);
    for gib in 0..IDENTITY_MAP_2M_GIB {
        let page_directory = first_page_directory + gib * PAGE_SIZE;
        set_entry(first_pdpt, gib, page_directory | PTE_PRESENT | PTE_WRITABLE);
        for slot in 0..ENTRIES_PER_TABLE {
            let physical = (gib * ENTRIES_PER_TABLE + slot) << HUGE_PAGE_SHIFT;
            set_entry(
                page_directory,
                slot,
                physical | PTE_PRESENT | PTE_WRITABLE | PTE_PAGE_SIZE,
            );
        }
    }
    if mapped_gib > PDPT_GIB {
        set_entry(PML4, 1, second_pdpt | PTE_PRESENT | PTE_WRITABLE);
    }
    for gib in IDENTITY_MAP_2M_GIB..mapped_gib {
        let pdpt = if gib < PDPT_GIB {
            first_pdpt
        } else {
            second_pdpt
        };
        set_entry(
            pdpt,
            gib % PDPT_GIB,
            (gib << GIB_SHIFT) | PTE_PRESENT | PTE_WRITABLE | PTE_PAGE_SIZE,
        );
    }
    put(ram, PML4, &tables)?;
    Ok((gdt_bytes.len() - 1) as u16)
}

/// Places the kernel, initrd, command line, `boot_params`, GDT and page tables in `ram` and
/// returns the register state for the 64-bit entry (Documentation/arch/x86/boot.rst, "64-bit
/// Boot Protocol"; `boot_params` layout follows QEMU's `x86_load_linux`). No firmware runs, so
/// everything the kernel expects a bootloader to have set up is done here.
///
/// Every size and address comes from the image or the caller, so all arithmetic on them is
/// checked and a misfit is an `Error`, never a panic.
pub(crate) fn load(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
) -> Result<BootState, Error> {
    load_with_acpi(ram, kernel, initrd, cmdline, None)
}

/// `load`, and with `acpi` the tables are written into RAM, the memory map marks them, and
/// `boot_params.acpi_rsdp_addr` points the kernel at them.
pub(crate) fn load_with_acpi(
    ram: &mut [u8],
    kernel: &[u8],
    initrd: Option<&[u8]>,
    cmdline: &str,
    acpi: Option<&AcpiConfig>,
) -> Result<BootState, Error> {
    let header = SetupHeader::parse(kernel)?;
    // +1 for the NUL; RAM is zeroed so the terminator is already there. The command line must
    // also end inside the usable low-memory range, or it would run into the reserved area (and,
    // with a huge `cmdline_max`, into the kernel).
    let cmdline_end = CMDLINE_ADDR + cmdline.len() as u64 + 1;
    if cmdline.len() + 1 > header.cmdline_max as usize || cmdline_end > LOW_RAM_END {
        return Err(Error::BadConfig("kernel command line too long".into()));
    }
    let ram_top = ram.len() as u64;
    // The kernel and initrd go below the PCI hole, where every boot path can reach them.
    let low_ram_top = ram::low_ram_end(ram_top);
    let payload = &kernel[header.protected_offset()..];
    let load_addr = if header.relocatable {
        KERNEL_LOAD
    } else {
        header.pref_address
    };
    // The protected-mode kernel lives above the first MiB (boot.rst); below it are the structures
    // placed here and the legacy memory map.
    if load_addr < ONE_MIB {
        return Err(Error::BadKernel(
            "preferred load address of a non-relocatable kernel is below 1 MiB".into(),
        ));
    }
    // `init_size` is the footprint the kernel needs while it decompresses itself, which can
    // exceed the compressed payload; nothing else may live in it.
    let kernel_end = load_addr
        .checked_add((header.init_size as u64).max(payload.len() as u64))
        .filter(|&end| end <= low_ram_top)
        .ok_or_else(|| Error::BadConfig("guest memory too small for kernel".into()))?;
    put(ram, load_addr, payload)?;
    put(ram, CMDLINE_ADDR, cmdline.as_bytes())?;

    let mut boot_params = vec![0u8; BOOT_PARAMS_SIZE];
    // The header's own length byte says where it ends; boot_params has other fields past
    // `SETUP_HEADER_END_LIMIT` that must stay zero. `parse` guarantees the image is longer than
    // that limit (the payload starts at 0x400 or later).
    let copy_end =
        (HEADER_MAGIC_OFFSET + kernel[JUMP_LENGTH_OFFSET] as usize).min(SETUP_HEADER_END_LIMIT);
    boot_params[SETUP_SECTS_OFFSET..copy_end]
        .copy_from_slice(&kernel[SETUP_SECTS_OFFSET..copy_end]);
    boot_params[TYPE_OF_LOADER_OFFSET] = LOADER_TYPE_UNDEFINED;
    boot_params[LOADFLAGS_OFFSET] = header.loadflags | LOADFLAGS_CAN_USE_HEAP;
    boot_params[CMD_LINE_PTR_OFFSET..CMD_LINE_PTR_OFFSET + 4]
        .copy_from_slice(&(CMDLINE_ADDR as u32).to_le_bytes());

    if let Some(initrd) = initrd {
        // Highest page-aligned spot the kernel can still reach: it cannot read an initrd above
        // `initrd_addr_max`, and RAM may end below that.
        let limit = low_ram_top.min(header.initrd_addr_max as u64 + 1);
        let addr = limit
            .checked_sub(initrd.len() as u64)
            .map(|unaligned| unaligned & !PAGE_MASK)
            .filter(|&addr| addr >= kernel_end)
            .ok_or_else(|| Error::BadConfig("guest memory too small for initrd".into()))?;
        put(ram, addr, initrd)?;
        // `addr` is at most `initrd_addr_max`, so it and the length (below `limit`) fit in 32 bits.
        boot_params[RAMDISK_IMAGE_OFFSET..RAMDISK_IMAGE_OFFSET + 4]
            .copy_from_slice(&(addr as u32).to_le_bytes());
        boot_params[RAMDISK_SIZE_OFFSET..RAMDISK_SIZE_OFFSET + 4]
            .copy_from_slice(&(initrd.len() as u32).to_le_bytes());
    }

    let e820 = match acpi {
        Some(config) => {
            let rsdp = acpi::install(ram, config)?;
            boot_params[ACPI_RSDP_ADDR_OFFSET..ACPI_RSDP_ADDR_OFFSET + 8]
                .copy_from_slice(&rsdp.to_le_bytes());
            e820_map_with_acpi(ram_top)
        }
        None => e820_map(ram_top),
    };
    boot_params[E820_ENTRIES_OFFSET] = e820.len() as u8;
    for (index, (base, size, kind)) in e820.iter().enumerate() {
        let entry = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
        boot_params[entry..entry + 8].copy_from_slice(&base.to_le_bytes());
        boot_params[entry + 8..entry + 16].copy_from_slice(&size.to_le_bytes());
        boot_params[entry + 16..entry + 20].copy_from_slice(&kind.to_le_bytes());
    }
    put(ram, BOOT_PARAMS, &boot_params)?;

    let kernel_is_64_bit = header.xloadflags & XLF_KERNEL_64 != 0;
    let code_descriptor = if kernel_is_64_bit {
        BOOT_CODE_DESCRIPTOR
    } else {
        BOOT_CODE32_DESCRIPTOR
    };
    let gdt_limit = write_gdt_and_page_tables(ram, ram_top, code_descriptor)?;

    Ok(BootState {
        // A 32-bit kernel starts at the first byte of its protected-mode image (`startup_32`).
        rip: load_addr + if kernel_is_64_bit { ENTRY_64_OFFSET } else { 0 },
        rsi: BOOT_PARAMS,
        rdi: 0,
        cr3: PML4,
        gdt_base: GDT_ADDR,
        gdt_limit,
        cs: BOOT_CS,
        ds: BOOT_DS,
        mode: if header.xloadflags & XLF_KERNEL_64 != 0 {
            BootMode::Long64
        } else {
            BootMode::Protected32
        },
    })
}

/// The state for booting a disk's boot sector. The BIOS itself is host code, set up by
/// `Cpu::bios_install` once the devices exist.
pub(crate) fn load_bios() -> BootState {
    BootState {
        mode: BootMode::Bios,
        ..Default::default()
    }
}

/// Largest firmware image: it is mapped so that it ends at 1 MiB, below the video memory hole.
const FIRMWARE_MAX_BYTES: usize = 0x6_0000;
/// Top of the first MiB, where a BIOS image ends. The processor's reset vector (F000:FFF0, which is
/// 0xFFFF0) is 16 bytes below it.
const FIRMWARE_END: u64 = 0x10_0000;

/// Places a firmware image at the top of the first mebibyte and returns the real-mode reset
/// state. There is no ROM or chipset: the image sits in ordinary RAM, which is enough for a BIOS
/// that does not write to its own range.
pub(crate) fn load_firmware(ram: &mut [u8], image: &[u8]) -> Result<BootState, Error> {
    if image.is_empty() || image.len() > FIRMWARE_MAX_BYTES {
        return Err(Error::BadConfig(format!(
            "firmware must be between 1 byte and {FIRMWARE_MAX_BYTES} bytes"
        )));
    }
    put(ram, FIRMWARE_END - image.len() as u64, image)?;
    Ok(BootState {
        mode: BootMode::Real16,
        ..Default::default()
    })
}

pub(crate) mod hypervisor;
mod inflate;
pub(crate) mod uefi;
pub(crate) use uefi::{
    FirmwareParams, UefiPayload, framebuffer_region, has_efi_header, load_uefi_firmware,
};

#[cfg(test)]
mod tests;
