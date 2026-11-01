//! The memory type range registers as a PC firmware leaves them: enabled, write-back by default,
//! the legacy VGA window uncacheable through the fixed ranges, and variable ranges making the MMIO
//! hole below 4 GiB uncacheable. Linux keeps whatever it finds (it only reads them, then builds
//! its PAT view on top) and says "MTRRs disabled by BIOS" when the enable bit is clear. The
//! emulator has no caches, so the types change nothing but what the OS believes.
//!
//! MTRRs are per logical processor and INIT keeps them (SDM vol. 3 table 10-1), so every vCPU is
//! programmed: the BSP here, the others through `processors.rs`, before the OS starts them.
//! Linux compares the APs with the BSP and complains ("your CPUs had inconsistent ... MTRR
//! settings", "probably your BIOS does not setup all CPUs") otherwise.

use alloc::vec;
use alloc::vec::Vec;

pub const MSR_MTRR_CAP: u32 = 0xfe;
pub const MSR_MTRR_DEF_TYPE: u32 = 0x2ff;
const MSR_MTRR_PHYS_BASE_FIRST: u32 = 0x200;
const MSR_MTRR_FIX_64K_00000: u32 = 0x250;
const MSR_MTRR_FIX_16K_80000: u32 = 0x258;
const MSR_MTRR_FIX_16K_A0000: u32 = 0x259;
const MSR_MTRR_FIX_4K_C0000: u32 = 0x268;
const FIXED_4K_REGISTERS: u32 = 8;

/// MTRRcap bits 7:0 (VCNT) and bit 8 (FIX).
const CAP_VARIABLE_COUNT_MASK: u64 = 0xff;
const CAP_FIXED_RANGES: u64 = 1 << 8;

const DEF_TYPE_FIXED_ENABLE: u64 = 1 << 10;
const DEF_TYPE_ENABLE: u64 = 1 << 11;
const MASK_VALID: u64 = 1 << 11;

const TYPE_UNCACHEABLE: u8 = 0;
const TYPE_WRITE_BACK: u8 = 6;
/// A fixed-range register holds eight one-byte types.
const ALL_WRITE_BACK: u64 = u64::from_ne_bytes([TYPE_WRITE_BACK; 8]);
const ALL_UNCACHEABLE: u64 = u64::from_ne_bytes([TYPE_UNCACHEABLE; 8]);

const PAGE_BYTES: u64 = 0x1000;
const FOUR_GIB: u64 = 1 << 32;

/// The registers to write, in the order the SDM (11.11.7.2) wants them: everything with the
/// MTRRs disabled, then `MTRRdefType` with the enable bits last.
#[derive(Debug, PartialEq, Eq)]
pub struct Settings {
    pub writes: Vec<(u32, u64)>,
}

/// Naturally aligned power-of-two blocks covering `[start, FOUR_GIB)` for some `start` in
/// `[earliest, latest]`, chosen to need the fewest blocks: a variable range can only describe such
/// a block, and the space between the top of RAM and the first MMIO holds nothing, so its type
/// does not matter.
fn hole_blocks(earliest: u64, latest: u64) -> Vec<(u64, u64)> {
    // Lengths of `[start, FOUR_GIB)` to choose from, and the one with the fewest bits set: take
    // the highest bit of the longest; if that alone is long enough it is the answer, else keep it
    // and solve the rest the same way.
    let (mut shortest, mut longest) = (FOUR_GIB - latest, FOUR_GIB - earliest);
    let mut length = 0;
    while shortest > 0 {
        let top_bit = 1u64 << (63 - longest.leading_zeros());
        length += top_bit;
        if top_bit >= shortest {
            break;
        }
        shortest -= top_bit;
        longest -= top_bit;
    }
    // Largest blocks at the top keep every block aligned to its own size, since 4 GiB is.
    let mut blocks = Vec::new();
    let mut end = FOUR_GIB;
    for bit in (0..u64::BITS).rev() {
        let size = 1u64 << bit;
        if length & size != 0 {
            end -= size;
            blocks.push((end, size));
        }
    }
    blocks.reverse();
    blocks
}

/// What to program for a machine whose RAM below 4 GiB ends at `ram_top_below_4g` and whose
/// lowest MMIO below 4 GiB starts at `mmio_start`, given MTRRcap and the physical address width.
/// `None` when the hole cannot be described (no fixed ranges, or too few variable ranges): the
/// MTRRs then stay disabled, as before.
pub fn settings(
    ram_top_below_4g: u64,
    mmio_start: u64,
    capability: u64,
    physical_address_bits: u32,
) -> Option<Settings> {
    let variable_count = (capability & CAP_VARIABLE_COUNT_MASK) as usize;
    if capability & CAP_FIXED_RANGES == 0 {
        return None;
    }
    let earliest = ram_top_below_4g.next_multiple_of(PAGE_BYTES);
    let latest = mmio_start.max(earliest);
    let blocks = if earliest >= FOUR_GIB {
        Vec::new()
    } else {
        hole_blocks(earliest, latest)
    };
    if blocks.len() > variable_count {
        return None;
    }
    let address_mask = ((1u64 << physical_address_bits) - 1) & !(PAGE_BYTES - 1);
    let mut writes = vec![(MSR_MTRR_DEF_TYPE, 0)];
    // 0..0x9ffff write-back (one 64K register for 0..0x7ffff, one 16K for 0x80000..0x9ffff), the
    // VGA window 0xa0000..0xbffff uncacheable, and the ROM/BIOS area write-back: here it is RAM
    // (the ACPI tables live at 0xe0000), not ROM.
    writes.push((MSR_MTRR_FIX_64K_00000, ALL_WRITE_BACK));
    writes.push((MSR_MTRR_FIX_16K_80000, ALL_WRITE_BACK));
    writes.push((MSR_MTRR_FIX_16K_A0000, ALL_UNCACHEABLE));
    for index in 0..FIXED_4K_REGISTERS {
        writes.push((MSR_MTRR_FIX_4K_C0000 + index, ALL_WRITE_BACK));
    }
    for index in 0..variable_count {
        let base_msr = MSR_MTRR_PHYS_BASE_FIRST + 2 * index as u32;
        let (base, mask) = match blocks.get(index) {
            Some(&(base, size)) => (
                base | TYPE_UNCACHEABLE as u64,
                (!(size - 1) & address_mask) | MASK_VALID,
            ),
            // Clears whatever an earlier boot left in the unused ranges.
            None => (0, 0),
        };
        writes.push((base_msr, base));
        writes.push((base_msr + 1, mask));
    }
    writes.push((
        MSR_MTRR_DEF_TYPE,
        DEF_TYPE_ENABLE | DEF_TYPE_FIXED_ENABLE | TYPE_WRITE_BACK as u64,
    ));
    Some(Settings { writes })
}

/// The end of the RAM below 4 GiB and the lowest MMIO below 4 GiB (4 GiB when there is none),
/// from the handoff's regions.
pub fn hole_bounds(ram: &[(u64, u64)], mmio: &[(u64, u64)], ecam_base: u64) -> (u64, u64) {
    let ram_top = ram
        .iter()
        .map(|&(base, size)| base + size)
        .filter(|&end| end <= FOUR_GIB)
        .max()
        .unwrap_or(0);
    let mmio_start = mmio
        .iter()
        .map(|&(base, _)| base)
        .chain((ecam_base != 0).then_some(ecam_base))
        .filter(|&base| base < FOUR_GIB)
        .min()
        .unwrap_or(FOUR_GIB);
    (ram_top, mmio_start)
}

#[cfg(target_os = "none")]
pub fn init(handoff: &crate::handoff::Handoff) {
    use crate::cpu;
    use crate::handoff::RAM_USABLE;

    /// CPUID Fn8000_0008 EAX bits 7:0: physical address width.
    const CPUID_ADDRESS_SIZES: u32 = 0x8000_0008;
    const PHYSICAL_ADDRESS_BITS_MASK: u32 = 0xff;

    let ram: Vec<(u64, u64)> = handoff
        .ram_regions
        .iter()
        .filter(|region| region.kind == RAM_USABLE)
        .map(|region| (region.base, region.size))
        .collect();
    let (ram_top, mmio_start) = hole_bounds(&ram, &handoff.mmio_regions, handoff.pci_ecam_base);
    let physical_address_bits = cpu::cpuid(CPUID_ADDRESS_SIZES, 0)[0] & PHYSICAL_ADDRESS_BITS_MASK;
    let Some(settings) = settings(
        ram_top,
        mmio_start,
        cpu::rdmsr(MSR_MTRR_CAP),
        physical_address_bits,
    ) else {
        crate::serial::trace!("mtrr: hole does not fit the variable ranges, left disabled");
        return;
    };
    for &(msr, value) in &settings.writes {
        cpu::wrmsr(msr, value);
    }
    crate::processors::write_msrs_on_application_processors(&settings.writes, handoff.tsc_hz);
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    /// MTRRcap as the emulator reports it: 8 variable ranges, fixed ranges, write-combining.
    const CAPABILITY: u64 = 0x508;
    const PHYSICAL_BITS: u32 = 40;
    const PCI_HOLE_START: u64 = 0xc000_0000;

    fn variable_ranges(settings: &Settings) -> Vec<(u64, u64)> {
        settings
            .writes
            .iter()
            .filter(|(msr, _)| (0x200..0x210).contains(msr))
            .collect::<Vec<_>>()
            .chunks(2)
            .map(|pair| (pair[0].1, pair[1].1))
            .filter(|&(_, mask)| mask & MASK_VALID != 0)
            .collect()
    }

    fn written(settings: &Settings, msr: u32) -> Vec<u64> {
        settings
            .writes
            .iter()
            .filter(|&&(m, _)| m == msr)
            .map(|&(_, value)| value)
            .collect()
    }

    #[test]
    fn default_type_is_disabled_first_and_enabled_write_back_last() {
        let settings = settings(2 * GIB, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS).unwrap();
        assert_eq!(settings.writes.first(), Some(&(0x2ff, 0)));
        assert_eq!(settings.writes.last(), Some(&(0x2ff, 0xc06)));
    }

    #[test]
    fn fixed_ranges_make_only_the_vga_window_uncacheable() {
        let settings = settings(2 * GIB, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS).unwrap();
        assert_eq!(written(&settings, 0x250), [0x0606_0606_0606_0606]);
        assert_eq!(written(&settings, 0x258), [0x0606_0606_0606_0606]);
        assert_eq!(written(&settings, 0x259), [0]);
        for msr in 0x268..=0x26f {
            assert_eq!(written(&settings, msr), [0x0606_0606_0606_0606], "{msr:#x}");
        }
    }

    #[test]
    fn the_pci_hole_of_a_large_machine_is_one_uncacheable_gigabyte() {
        let settings = settings(PCI_HOLE_START, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS).unwrap();
        assert_eq!(
            variable_ranges(&settings),
            [(0xc000_0000, 0xff_c000_0800)],
            "base 3 GiB type UC; mask 1 GiB within 40 bits, valid"
        );
    }

    #[test]
    fn a_small_machine_covers_the_empty_space_below_the_hole_with_the_same_range() {
        let settings = settings(2 * GIB, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS).unwrap();
        assert_eq!(variable_ranges(&settings), [(0x8000_0000, 0xff_8000_0800)]);
        let between = super::settings(
            2 * GIB + 64 * MIB,
            PCI_HOLE_START,
            CAPABILITY,
            PHYSICAL_BITS,
        );
        assert_eq!(
            variable_ranges(&between.unwrap()),
            [(0xc000_0000, 0xff_c000_0800)]
        );
        let tiny = super::settings(64 * MIB, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS);
        assert_eq!(
            variable_ranges(&tiny.unwrap()),
            [(0x8000_0000, 0xff_8000_0800)]
        );
    }

    #[test]
    fn unused_variable_ranges_are_cleared() {
        let settings = settings(2 * GIB, PCI_HOLE_START, CAPABILITY, PHYSICAL_BITS).unwrap();
        for msr in 0x202..0x210 {
            assert_eq!(written(&settings, msr), [0], "{msr:#x}");
        }
    }

    #[test]
    fn a_hole_that_starts_right_at_the_end_of_ram_takes_one_block_per_bit() {
        // RAM to 3 GiB + 5 MiB with MMIO right behind it: 1 MiB + 2 MiB at the bottom, then 8,
        // 16, 32, ..., 512 MiB up to 4 GiB.
        let top = 3 * GIB + 5 * MIB;
        let blocks = hole_blocks(top, top);
        assert_eq!(blocks.first(), Some(&(top, MIB)));
        assert_eq!(
            blocks.iter().map(|(_, size)| size).sum::<u64>(),
            4 * GIB - top
        );
        for &(base, size) in &blocks {
            assert!(
                size.is_power_of_two() && base % size == 0,
                "{base:#x} {size:#x}"
            );
        }
        assert_eq!(blocks.len(), (4 * GIB - top).count_ones() as usize);
    }

    #[test]
    fn more_blocks_than_variable_ranges_leave_the_mtrrs_alone() {
        let top = 3 * GIB + 5 * MIB;
        assert_eq!(settings(top, top, CAPABILITY, PHYSICAL_BITS), None);
        assert!(settings(top, top, 0x100 | 16, PHYSICAL_BITS).is_some());
    }

    #[test]
    fn no_fixed_ranges_means_no_mtrr_setup() {
        assert_eq!(settings(2 * GIB, PCI_HOLE_START, 8, PHYSICAL_BITS), None);
    }

    #[test]
    fn hole_bounds_come_from_the_ram_below_4_gib_and_the_lowest_mmio() {
        let ram = [(0, 0x9_fc00), (MIB, 3 * GIB - MIB), (4 * GIB, 5 * GIB)];
        let mmio = [(0xfec0_0000, 0x1000), (PCI_HOLE_START, 0x10_0000)];
        assert_eq!(
            hole_bounds(&ram, &mmio, 0xe000_0000),
            (3 * GIB, PCI_HOLE_START)
        );
        assert_eq!(hole_bounds(&ram[..2], &[], 0), (3 * GIB, 4 * GIB));
    }
}
