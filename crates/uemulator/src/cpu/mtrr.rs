//! Memory type range registers and the PAT.
//!
//! The emulated memory system has no caches, so memory types change nothing a guest can observe;
//! these registers are validated storage. Validation matters: an OS probes the MTRR layout and
//! the PAT with `wrmsr` and expects #GP for a reserved memory type (SDM vol. 3 sec. 11.11.2,
//! 11.12.2; AMD APM vol. 2 sec. 7.7 and 7.8).

use super::sys::PHYSICAL_ADDRESS_BITS;
use crate::snapshot::Snapshot;

pub(crate) const MSR_MTRR_CAP: u32 = 0xfe;
pub(crate) const MSR_MTRR_DEF_TYPE: u32 = 0x2ff;
const MSR_MTRR_PHYS_BASE_FIRST: u32 = 0x200;
const MSR_MTRR_PHYS_MASK_LAST: u32 = MSR_MTRR_PHYS_BASE_FIRST + 2 * VARIABLE_RANGES as u32 - 1;
const MSR_MTRR_FIX_64K_00000: u32 = 0x250;
const MSR_MTRR_FIX_16K_80000: u32 = 0x258;
const MSR_MTRR_FIX_16K_A0000: u32 = 0x259;
const MSR_MTRR_FIX_4K_FIRST: u32 = 0x268;
const MSR_MTRR_FIX_4K_LAST: u32 = 0x26f;

pub(crate) const VARIABLE_RANGES: usize = 8;
/// Fixed-range registers: one 64K, two 16K and eight 4K.
const FIXED_REGISTERS: usize = 11;

const CAP_VARIABLE_COUNT_MASK: u64 = 0xff;
const CAP_FIXED_RANGES: u64 = 1 << 8;
/// Write-combining is supported.
const CAP_WRITE_COMBINING: u64 = 1 << 10;
/// What the capability register reads as.
const CAP_VALUE: u64 =
    (VARIABLE_RANGES as u64 & CAP_VARIABLE_COUNT_MASK) | CAP_FIXED_RANGES | CAP_WRITE_COMBINING;

const DEF_TYPE_FIXED_ENABLE: u64 = 1 << 10;
const DEF_TYPE_ENABLE: u64 = 1 << 11;
const DEF_TYPE_WRITABLE: u64 = 0xff | DEF_TYPE_FIXED_ENABLE | DEF_TYPE_ENABLE;

const MASK_VALID: u64 = 1 << 11;
/// Base and mask registers hold a physical address in bits 12..MAXPHYADDR.
const PHYS_ADDRESS_BITS_MASK: u64 = ((1u64 << PHYSICAL_ADDRESS_BITS) - 1) & !0xfff;
const BASE_WRITABLE: u64 = PHYS_ADDRESS_BITS_MASK | 0xff;
const MASK_WRITABLE: u64 = PHYS_ADDRESS_BITS_MASK | MASK_VALID;

const TYPE_UNCACHEABLE: u8 = 0;
const TYPE_WRITE_COMBINING: u8 = 1;
const TYPE_WRITE_THROUGH: u8 = 4;
const TYPE_WRITE_PROTECTED: u8 = 5;
const TYPE_WRITE_BACK: u8 = 6;
const TYPE_UNCACHED_MINUS: u8 = 7;

/// Memory types an MTRR may name. 2 and 3 are reserved, and 7 exists only in the PAT.
fn valid_mtrr_type(memory_type: u8) -> bool {
    matches!(
        memory_type,
        TYPE_UNCACHEABLE
            | TYPE_WRITE_COMBINING
            | TYPE_WRITE_THROUGH
            | TYPE_WRITE_PROTECTED
            | TYPE_WRITE_BACK
    )
}

fn valid_pat_type(memory_type: u8) -> bool {
    valid_mtrr_type(memory_type) || memory_type == TYPE_UNCACHED_MINUS
}

/// Whether every byte of `value` is accepted by `valid`.
fn all_bytes(value: u64, valid: fn(u8) -> bool) -> bool {
    value.to_le_bytes().into_iter().all(valid)
}

pub(crate) fn pat_is_valid(value: u64) -> bool {
    all_bytes(value, valid_pat_type)
}

#[derive(Clone, Snapshot)]
pub(crate) struct Mtrr {
    def_type: u64,
    fixed: [u64; FIXED_REGISTERS],
    variable_base: [u64; VARIABLE_RANGES],
    variable_mask: [u64; VARIABLE_RANGES],
}

impl Mtrr {
    pub fn new() -> Self {
        Mtrr {
            def_type: 0,
            fixed: [0; FIXED_REGISTERS],
            variable_base: [0; VARIABLE_RANGES],
            variable_mask: [0; VARIABLE_RANGES],
        }
    }

    pub fn owns(msr: u32) -> bool {
        matches!(
            msr,
            MSR_MTRR_CAP
                | MSR_MTRR_DEF_TYPE
                | MSR_MTRR_PHYS_BASE_FIRST..=MSR_MTRR_PHYS_MASK_LAST
                | MSR_MTRR_FIX_64K_00000
                | MSR_MTRR_FIX_16K_80000
                | MSR_MTRR_FIX_16K_A0000
                | MSR_MTRR_FIX_4K_FIRST..=MSR_MTRR_FIX_4K_LAST
        )
    }

    fn fixed_index(msr: u32) -> Option<usize> {
        match msr {
            MSR_MTRR_FIX_64K_00000 => Some(0),
            MSR_MTRR_FIX_16K_80000 => Some(1),
            MSR_MTRR_FIX_16K_A0000 => Some(2),
            MSR_MTRR_FIX_4K_FIRST..=MSR_MTRR_FIX_4K_LAST => {
                Some(3 + (msr - MSR_MTRR_FIX_4K_FIRST) as usize)
            }
            _ => None,
        }
    }

    pub fn read(&self, msr: u32) -> Option<u64> {
        match msr {
            MSR_MTRR_CAP => Some(CAP_VALUE),
            MSR_MTRR_DEF_TYPE => Some(self.def_type),
            MSR_MTRR_PHYS_BASE_FIRST..=MSR_MTRR_PHYS_MASK_LAST => {
                let index = ((msr - MSR_MTRR_PHYS_BASE_FIRST) / 2) as usize;
                Some(if msr & 1 == 0 {
                    self.variable_base[index]
                } else {
                    self.variable_mask[index]
                })
            }
            _ => Self::fixed_index(msr).map(|index| self.fixed[index]),
        }
    }

    /// `false` is #GP: a read-only register, a reserved memory type or a reserved bit.
    pub fn write(&mut self, msr: u32, value: u64) -> bool {
        match msr {
            MSR_MTRR_CAP => false,
            MSR_MTRR_DEF_TYPE => {
                if value & !DEF_TYPE_WRITABLE != 0 || !valid_mtrr_type(value as u8) {
                    return false;
                }
                self.def_type = value;
                true
            }
            MSR_MTRR_PHYS_BASE_FIRST..=MSR_MTRR_PHYS_MASK_LAST => {
                let index = ((msr - MSR_MTRR_PHYS_BASE_FIRST) / 2) as usize;
                if msr & 1 == 0 {
                    if value & !BASE_WRITABLE != 0 || !valid_mtrr_type(value as u8) {
                        return false;
                    }
                    self.variable_base[index] = value;
                } else {
                    if value & !MASK_WRITABLE != 0 {
                        return false;
                    }
                    self.variable_mask[index] = value;
                }
                true
            }
            _ => match Self::fixed_index(msr) {
                Some(index) if all_bytes(value, valid_mtrr_type) => {
                    self.fixed[index] = value;
                    true
                }
                _ => false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_reports_eight_variable_ranges_fixed_ranges_and_write_combining() {
        assert_eq!(Mtrr::new().read(MSR_MTRR_CAP), Some(0x508));
    }

    #[test]
    fn capability_is_read_only() {
        assert!(!Mtrr::new().write(MSR_MTRR_CAP, 0));
    }

    #[test]
    fn default_type_accepts_enable_bits_and_valid_types_only() {
        let mut mtrr = Mtrr::new();
        assert!(mtrr.write(
            MSR_MTRR_DEF_TYPE,
            DEF_TYPE_ENABLE | DEF_TYPE_FIXED_ENABLE | 6
        ));
        assert_eq!(mtrr.read(MSR_MTRR_DEF_TYPE), Some(0xc06));
        assert!(!mtrr.write(MSR_MTRR_DEF_TYPE, 2));
        assert!(!mtrr.write(MSR_MTRR_DEF_TYPE, 7));
        assert!(!mtrr.write(MSR_MTRR_DEF_TYPE, 1 << 9));
        assert!(!mtrr.write(MSR_MTRR_DEF_TYPE, 1 << 12));
    }

    #[test]
    fn variable_base_checks_type_and_reserved_bits() {
        let mut mtrr = Mtrr::new();
        assert!(mtrr.write(0x200, 0x8000_0000 | 1));
        assert_eq!(mtrr.read(0x200), Some(0x8000_0001));
        assert!(!mtrr.write(0x200, 3));
        assert!(!mtrr.write(0x200, 0x100));
        assert!(!mtrr.write(0x200, 1 << PHYSICAL_ADDRESS_BITS));
    }

    #[test]
    fn variable_mask_checks_reserved_bits() {
        let mut mtrr = Mtrr::new();
        assert!(mtrr.write(0x201, MASK_VALID | 0xff_c000_0000));
        assert!(!mtrr.write(0x201, 1 << 10));
        assert!(!mtrr.write(0x201, 1 << PHYSICAL_ADDRESS_BITS));
    }

    #[test]
    fn all_sixteen_variable_registers_exist_and_the_seventeenth_does_not() {
        let mtrr = Mtrr::new();
        for msr in 0x200..=0x20f {
            assert!(mtrr.read(msr).is_some(), "{msr:#x}");
        }
        assert!(!Mtrr::owns(0x210));
        assert!(Mtrr::owns(0x20f));
    }

    #[test]
    fn fixed_registers_validate_every_byte() {
        let mut mtrr = Mtrr::new();
        for msr in [0x250, 0x258, 0x259, 0x268, 0x26f] {
            assert!(mtrr.write(msr, 0x0605_0401_0600_0406), "{msr:#x}");
            assert!(!mtrr.write(msr, 0x0605_0401_0600_0306), "{msr:#x}");
            assert!(!mtrr.write(msr, 7), "{msr:#x}");
        }
        assert!(!Mtrr::owns(0x25a));
        assert!(!Mtrr::owns(0x267));
        assert!(!Mtrr::owns(0x270));
    }

    #[test]
    fn pat_accepts_types_0_1_4_5_6_7() {
        assert!(pat_is_valid(0x0007_0406_0007_0406));
        assert!(pat_is_valid(0x0706_0504_0100_0000));
        assert!(!pat_is_valid(0x0706_0504_0302_0100));
        assert!(!pat_is_valid(2));
        assert!(!pat_is_valid(3 << 56));
        assert!(!pat_is_valid(8));
    }
}
