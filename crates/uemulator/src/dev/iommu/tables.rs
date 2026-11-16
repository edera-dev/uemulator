//! Device table entries (spec 3.2.2.1, 256 bits as four little-endian qwords) and the host page
//! table walk (spec 2.2.3 and 3.3.x "I/O page tables for host translations").
use super::{PhysRead, read_u64};
use crate::snapshot::Snapshot;

const DTE_VALID: u64 = 1 << 0;
const DTE_TRANSLATION_VALID: u64 = 1 << 1;
const DTE_MODE_SHIFT: u32 = 9;
const DTE_MODE_MASK: u64 = 0x7;
const DTE_PAGE_TABLE_ROOT_MASK: u64 = 0x000f_ffff_ffff_f000;
const DTE_READ_ALLOWED: u64 = 1 << 61;
const DTE_WRITE_ALLOWED: u64 = 1 << 62;
/// Qword 0 bits 6:2 are reserved. Bit 7 (HAD) and 8 are fields on parts with dirty tracking, so
/// they are not checked; QEMU also treats a reserved bit as an illegal entry.
const DTE_RESERVED_QUAD0: u64 = 0x7c;
/// Mode 7 is reserved.
const DTE_MODE_RESERVED: u64 = 7;

const DTE_DOMAIN_ID_MASK: u64 = 0xffff;
/// Qword 1 bit 39: EX, the device is allowed into the exclusion range.
const DTE_EXCLUSION_ALLOWED: u64 = 1 << 39;

/// Qword 2: interrupt remapping. IV enables it for the device, IntTabLen is log2 of the table's
/// entry count, the pointer is 128-byte aligned, IntCtl (61:60) says what to do with interrupts.
const DTE_INTERRUPT_MAP_VALID: u64 = 1 << 0;
const DTE_INTERRUPT_TABLE_LENGTH_SHIFT: u32 = 1;
const DTE_INTERRUPT_TABLE_LENGTH_MASK: u64 = 0xf;
const DTE_INTERRUPT_TABLE_ROOT_MASK: u64 = 0x000f_ffff_ffff_ffc0;
const DTE_INTERRUPT_CONTROL_SHIFT: u32 = 60;
const DTE_INTERRUPT_CONTROL_MASK: u64 = 0x3;
/// DTE bits 184..=191 are the pass-through flags (qword 2 bits 56 and up): INIT, ExtInt, NMI,
/// then LINT0 and LINT1.
const DTE_INIT_PASS: u64 = 1 << 56;
const DTE_EXTINT_PASS: u64 = 1 << 57;
const DTE_NMI_PASS: u64 = 1 << 58;
const DTE_LINT0_PASS: u64 = 1 << 62;
const DTE_LINT1_PASS: u64 = 1 << 63;

#[derive(Clone, Copy, Snapshot)]
#[snapshot(new)]
pub(super) struct DeviceTableEntry {
    quads: [u64; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DeviceEntryError {
    /// The requester id is beyond the table's size.
    OutsideTable,
    /// Reading the entry failed; carries the address.
    Unreadable(u64),
}

/// How to treat interrupts from a device (DTE.IntCtl).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InterruptControl {
    Abort,
    Forward,
    Remap,
    Reserved,
}

impl DeviceTableEntry {
    pub fn new(quads: [u64; 4]) -> Self {
        Self { quads }
    }

    pub fn valid(&self) -> bool {
        self.quads[0] & DTE_VALID != 0
    }

    pub fn translation_valid(&self) -> bool {
        self.quads[0] & DTE_TRANSLATION_VALID != 0
    }

    /// Paging levels of the host page table; 0 means no translation.
    pub fn mode(&self) -> u64 {
        (self.quads[0] >> DTE_MODE_SHIFT) & DTE_MODE_MASK
    }

    pub fn page_table_root(&self) -> u64 {
        self.quads[0] & DTE_PAGE_TABLE_ROOT_MASK
    }

    pub fn read_allowed(&self) -> bool {
        self.quads[0] & DTE_READ_ALLOWED != 0
    }

    pub fn write_allowed(&self) -> bool {
        self.quads[0] & DTE_WRITE_ALLOWED != 0
    }

    pub fn has_reserved_bits(&self) -> bool {
        self.quads[0] & DTE_RESERVED_QUAD0 != 0
            || (self.translation_valid() && self.mode() == DTE_MODE_RESERVED)
    }

    pub fn domain_id(&self) -> u16 {
        (self.quads[1] & DTE_DOMAIN_ID_MASK) as u16
    }

    pub fn exclusion(&self) -> bool {
        self.quads[1] & DTE_EXCLUSION_ALLOWED != 0
    }

    pub fn interrupt_map_valid(&self) -> bool {
        self.quads[2] & DTE_INTERRUPT_MAP_VALID != 0
    }

    /// log2 of the interrupt remapping table's entry count.
    pub fn interrupt_table_length(&self) -> u64 {
        (self.quads[2] >> DTE_INTERRUPT_TABLE_LENGTH_SHIFT) & DTE_INTERRUPT_TABLE_LENGTH_MASK
    }

    pub fn interrupt_table_root(&self) -> u64 {
        self.quads[2] & DTE_INTERRUPT_TABLE_ROOT_MASK
    }

    pub fn interrupt_control(&self) -> InterruptControl {
        match (self.quads[2] >> DTE_INTERRUPT_CONTROL_SHIFT) & DTE_INTERRUPT_CONTROL_MASK {
            0 => InterruptControl::Abort,
            1 => InterruptControl::Forward,
            2 => InterruptControl::Remap,
            _ => InterruptControl::Reserved,
        }
    }

    pub fn init_pass(&self) -> bool {
        self.quads[2] & DTE_INIT_PASS != 0
    }

    pub fn extint_pass(&self) -> bool {
        self.quads[2] & DTE_EXTINT_PASS != 0
    }

    pub fn nmi_pass(&self) -> bool {
        self.quads[2] & DTE_NMI_PASS != 0
    }

    pub fn lint0_pass(&self) -> bool {
        self.quads[2] & DTE_LINT0_PASS != 0
    }

    pub fn lint1_pass(&self) -> bool {
        self.quads[2] & DTE_LINT1_PASS != 0
    }
}

const PAGE_SHIFT: u32 = 12;
/// Each level indexes 9 bits of the address.
const LEVEL_BITS: u32 = 9;
const LEVEL_INDEX_MASK: u64 = (1 << LEVEL_BITS) - 1;
const PTE_BYTES: u64 = 8;
const MAX_LEVELS: u64 = 6;
const PTE_PRESENT: u64 = 1 << 0;
const PTE_NEXT_LEVEL_SHIFT: u32 = 9;
const PTE_NEXT_LEVEL_MASK: u64 = 0x7;
const PTE_OUTPUT_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const PTE_READ_ALLOWED: u64 = 1 << 61;
const PTE_WRITE_ALLOWED: u64 = 1 << 62;
/// NextLevel 0 makes the entry a leaf of its level's size; 7 makes it a leaf whose size is
/// encoded in the output address (below).
const NEXT_LEVEL_LEAF: u64 = 0;
const NEXT_LEVEL_SIZED_LEAF: u64 = 7;
/// Output address bits that can encode a size: 12 up to the top of the field (51).
const SIZED_LEAF_LOW_BITS: u64 = 0xfff;
const SIZED_LEAF_MAX_SHIFT: u32 = 51;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Leaf {
    /// 4 KiB-aligned physical page that `iova`'s page maps to.
    pub phys_page: u64,
    pub read: bool,
    pub write: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum WalkError {
    NotPresent,
    /// A level that goes the wrong way, or a size encoding with no room for the page.
    Malformed,
    /// Reading a page table entry failed; carries its address.
    PageTableRead(u64),
}

/// Walks `entry`'s page table for `iova`. Permissions are the AND of the DTE's IR/IW and every
/// entry on the way, as the spec has it.
///
/// An entry's NextLevel is the level of the table it points to, so a table may skip levels. 0
/// ends the walk with a page the size of the current level (4 KiB, 2 MiB, 1 GiB, ...) and 7 with
/// an arbitrary power-of-two size: the lowest clear bit at or above bit 12 of the output
/// address, position p, gives a page of 2^(p+1) bytes, and the bits below p in the address field
/// are ones that stand for the in-page offset (Linux `PAGE_SIZE_PTE`).
pub(super) fn walk(
    mem: &dyn PhysRead,
    entry: &DeviceTableEntry,
    iova: u64,
) -> Result<Leaf, WalkError> {
    let mut level = entry.mode();
    if level == 0 || level > MAX_LEVELS {
        return Err(WalkError::Malformed);
    }
    let covered_bits = PAGE_SHIFT + LEVEL_BITS * level as u32;
    if covered_bits < u64::BITS && iova >> covered_bits != 0 {
        return Err(WalkError::NotPresent);
    }
    let mut table = entry.page_table_root();
    let mut read = entry.read_allowed();
    let mut write = entry.write_allowed();
    loop {
        let shift = PAGE_SHIFT + LEVEL_BITS * (level as u32 - 1);
        let index = (iova >> shift) & LEVEL_INDEX_MASK;
        let address = table + index * PTE_BYTES;
        let pte = read_u64(mem, address).ok_or(WalkError::PageTableRead(address))?;
        if pte & PTE_PRESENT == 0 {
            return Err(WalkError::NotPresent);
        }
        read &= pte & PTE_READ_ALLOWED != 0;
        write &= pte & PTE_WRITE_ALLOWED != 0;
        let next_level = (pte >> PTE_NEXT_LEVEL_SHIFT) & PTE_NEXT_LEVEL_MASK;
        let output = pte & PTE_OUTPUT_ADDRESS_MASK;
        let page_shift = match next_level {
            NEXT_LEVEL_LEAF => shift,
            NEXT_LEVEL_SIZED_LEAF => {
                let first_clear = (!(pte | SIZED_LEAF_LOW_BITS)).trailing_zeros();
                if first_clear > SIZED_LEAF_MAX_SHIFT {
                    return Err(WalkError::Malformed);
                }
                first_clear + 1
            }
            next if next < level => {
                table = output;
                level = next;
                continue;
            }
            _ => return Err(WalkError::Malformed),
        };
        let page_mask = (1u64 << page_shift) - 1;
        let phys = (output & !page_mask) | (iova & page_mask);
        return Ok(Leaf {
            phys_page: phys & !((1 << PAGE_SHIFT) - 1),
            read,
            write,
        });
    }
}
