//! The ACPI IVRS table (I/O Virtualization Reporting Structure, AMD IOMMU spec 5.2): how an OS
//! learns that this IOMMU exists, where its registers are, and which devices, IOAPICs and HPETs
//! it translates for.
//!
//! One IVHD block is emitted per supported header type: 10h (the original format) and 11h and
//! 40h (both add a copy of the extended feature register). An OS picks the newest type it
//! knows, so all three describe the same unit with the same device list.
#[cfg(freestanding)]
use crate::prelude::*;

use super::{
    CAPABILITY_AT, EXTENDED_FEATURES, Iommu, MMIO_BASE, PHYSICAL_ADDRESS_BITS, REQUESTER_ID,
    VIRTUAL_ADDRESS_BITS,
};

const ACPI_HEADER_BYTES: usize = 36;
const ACPI_LENGTH_AT: usize = 4;
const ACPI_CHECKSUM_AT: usize = 9;
const IVRS_SIGNATURE: &[u8; 4] = b"IVRS";
/// Revision 1 has only IVHD 10h; 2 adds 11h and 40h.
const IVRS_REVISION: u8 = 2;
const IVRS_OEM_ID: &[u8; 6] = b"UEMUL ";
const IVRS_OEM_TABLE_ID: &[u8; 8] = b"UEMUIVRS";
const IVRS_OEM_REVISION: u32 = 1;
const IVRS_CREATOR_ID: &[u8; 4] = b"UEMU";
const IVRS_CREATOR_REVISION: u32 = 1;
/// IVinfo (4 bytes) and 8 reserved bytes precede the first block.
const IVRS_INFO_AND_RESERVED_BYTES: usize = 12;
/// IVinfo: EFRSup (bit 0), GVASize (7:5), PASize (14:8), VASize (21:15).
const IVINFO_EFR_SUPPORTED: u32 = 1 << 0;
const IVINFO_GVA_SIZE_48: u32 = 2 << 5;
const IVINFO_PA_SIZE_SHIFT: u32 = 8;
const IVINFO_VA_SIZE_SHIFT: u32 = 15;

const IVHD_TYPE_10: u8 = 0x10;
const IVHD_TYPE_11: u8 = 0x11;
const IVHD_TYPE_40: u8 = 0x40;
/// IVHD flags: Coherent (5) and PrefSup (6).
const IVHD_FLAG_COHERENT: u8 = 1 << 5;
const IVHD_FLAG_PREFETCH_SUPPORTED: u8 = 1 << 6;
/// Common header of a block: type, flags, length, device id, capability offset, MMIO base (8),
/// PCI segment, IOMMU info, feature info (4) = 24 bytes; 11h and 40h add the EFR image (8) and
/// a second reserved qword.
const IVHD_BASE_HEADER_BYTES: usize = 24;
const IVHD_EXTENDED_HEADER_BYTES: usize = 40;
/// IOMMU info: MSINum (4:0), UnitID (12:8); both 0 here.
const IVHD_IOMMU_INFO: u16 = 0;
/// IVHD 10h has no EFR image; its feature reporting field says the same in its own layout
/// (its "IOMMU feature reporting"): XTSup (0), NXSup (1), GTSup (2), IASup (5), GASup (6), HESup (7), HATS
/// (31:30). Linux reads the GASup and XTSup bits from it when 10h is the newest block it knows,
/// and turns interrupt remapping's x2APIC and 128-bit modes off when they are clear. 11h and
/// 40h carry IOMMU attributes there instead (counters, MSI numbers), all zero here.
const IVHD10_XT: u32 = 1 << 0;
const IVHD10_NX: u32 = 1 << 1;
const IVHD10_GT: u32 = 1 << 2;
const IVHD10_IA: u32 = 1 << 5;
const IVHD10_GA: u32 = 1 << 6;
const IVHD10_HE: u32 = 1 << 7;
const IVHD10_HATS_SHIFT: u32 = 30;
/// The EFR bits the 10h field repeats, in the same order as the constants above.
const EFR_FOR_IVHD10: [(u64, u32); 6] = [
    (1 << 2, IVHD10_XT),
    (1 << 3, IVHD10_NX),
    (1 << 4, IVHD10_GT),
    (1 << 6, IVHD10_IA),
    (1 << 7, IVHD10_GA),
    (1 << 8, IVHD10_HE),
];
const EFR_HATS_SHIFT: u32 = 10;
const HATS_MASK: u64 = 0b11;

/// The IVHD 10h feature reporting field `EXTENDED_FEATURES` implies.
fn ivhd10_features() -> u32 {
    let hats = ((EXTENDED_FEATURES >> EFR_HATS_SHIFT) & HATS_MASK) as u32;
    let mut features = hats << IVHD10_HATS_SHIFT;
    for (efr, bit) in EFR_FOR_IVHD10 {
        if EXTENDED_FEATURES & efr != 0 {
            features |= bit;
        }
    }
    features
}

/// Device entries (4 bytes: type, device id, data; the special entry is 8 bytes).
const ENTRY_SELECT: u8 = 0x02;
const ENTRY_SPECIAL: u8 = 0x48;
/// Data byte of a device entry; no pass-through, no system management bits.
const ENTRY_DATA_NONE: u8 = 0;
/// Special entry variety: which kind of device the handle names.
const SPECIAL_VARIETY_IOAPIC: u8 = 1;
const SPECIAL_VARIETY_HPET: u8 = 2;

/// An IOAPIC or HPET and the requester id its interrupt messages carry. `handle` is the IOAPIC
/// id or HPET block number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SpecialDevice {
    pub handle: u8,
    pub requester: u16,
}

/// The devices behind this IOMMU: every PCI function (as a 16-bit bus/device/function), the
/// IOAPICs and the HPETs.
#[derive(Clone, Default, Debug)]
pub(crate) struct IvrsTopology {
    pub pci_devices: Vec<u16>,
    pub ioapics: Vec<SpecialDevice>,
    pub hpets: Vec<SpecialDevice>,
}

fn device_entries(topology: &IvrsTopology) -> Vec<u8> {
    let mut entries = Vec::new();
    for &device in &topology.pci_devices {
        entries.push(ENTRY_SELECT);
        entries.extend_from_slice(&device.to_le_bytes());
        entries.push(ENTRY_DATA_NONE);
    }
    let specials = [
        (&topology.ioapics, SPECIAL_VARIETY_IOAPIC),
        (&topology.hpets, SPECIAL_VARIETY_HPET),
    ];
    for (devices, variety) in specials {
        for special in devices {
            entries.push(ENTRY_SPECIAL);
            entries.extend_from_slice(&0u16.to_le_bytes());
            entries.push(ENTRY_DATA_NONE);
            entries.push(special.handle);
            entries.extend_from_slice(&special.requester.to_le_bytes());
            entries.push(variety);
        }
    }
    entries
}

fn ivhd_block(block_type: u8, topology: &IvrsTopology) -> Vec<u8> {
    let extended = block_type != IVHD_TYPE_10;
    let header_bytes = if extended {
        IVHD_EXTENDED_HEADER_BYTES
    } else {
        IVHD_BASE_HEADER_BYTES
    };
    let entries = device_entries(topology);
    let mut block = Vec::with_capacity(header_bytes + entries.len());
    block.push(block_type);
    block.push(IVHD_FLAG_COHERENT | IVHD_FLAG_PREFETCH_SUPPORTED);
    block.extend_from_slice(&((header_bytes + entries.len()) as u16).to_le_bytes());
    block.extend_from_slice(&REQUESTER_ID.to_le_bytes());
    block.extend_from_slice(&(CAPABILITY_AT as u16).to_le_bytes());
    block.extend_from_slice(&MMIO_BASE.to_le_bytes());
    block.extend_from_slice(&0u16.to_le_bytes());
    block.extend_from_slice(&IVHD_IOMMU_INFO.to_le_bytes());
    let features = if extended { 0 } else { ivhd10_features() };
    block.extend_from_slice(&features.to_le_bytes());
    if extended {
        block.extend_from_slice(&EXTENDED_FEATURES.to_le_bytes());
        block.extend_from_slice(&0u64.to_le_bytes());
    }
    debug_assert_eq!(block.len(), header_bytes);
    block.extend_from_slice(&entries);
    block
}

impl Iommu {
    /// The complete IVRS table, header and checksum included, ready to be placed in the XSDT.
    pub fn ivrs_table(&self, topology: &IvrsTopology) -> Vec<u8> {
        let mut table = Vec::new();
        table.extend_from_slice(IVRS_SIGNATURE);
        table.extend_from_slice(&0u32.to_le_bytes());
        table.push(IVRS_REVISION);
        table.push(0);
        table.extend_from_slice(IVRS_OEM_ID);
        table.extend_from_slice(IVRS_OEM_TABLE_ID);
        table.extend_from_slice(&IVRS_OEM_REVISION.to_le_bytes());
        table.extend_from_slice(IVRS_CREATOR_ID);
        table.extend_from_slice(&IVRS_CREATOR_REVISION.to_le_bytes());
        debug_assert_eq!(table.len(), ACPI_HEADER_BYTES);
        let info = IVINFO_EFR_SUPPORTED
            | IVINFO_GVA_SIZE_48
            | PHYSICAL_ADDRESS_BITS << IVINFO_PA_SIZE_SHIFT
            | VIRTUAL_ADDRESS_BITS << IVINFO_VA_SIZE_SHIFT;
        table.extend_from_slice(&info.to_le_bytes());
        table.extend_from_slice(&[0; IVRS_INFO_AND_RESERVED_BYTES - 4]);
        for block_type in [IVHD_TYPE_10, IVHD_TYPE_11, IVHD_TYPE_40] {
            table.extend_from_slice(&ivhd_block(block_type, topology));
        }
        let length = table.len() as u32;
        table[ACPI_LENGTH_AT..ACPI_LENGTH_AT + 4].copy_from_slice(&length.to_le_bytes());
        let sum = table.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte));
        table[ACPI_CHECKSUM_AT] = 0u8.wrapping_sub(sum);
        table
    }
}
