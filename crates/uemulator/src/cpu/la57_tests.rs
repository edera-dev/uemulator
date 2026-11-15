//! 5-level paging (CR4.LA57): the PML5 walk, 57-bit canonical addresses, and the rule that LA57
//! only changes outside IA-32e mode. Expected values restate the APM (Vol. 2 5.3.6) and the SDM
//! (Vol. 3A 4.5) instead of reusing the constants in `mem.rs`.

use super::*;
use crate::cpu::decode;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT, VECTOR_STACK_FAULT};
use crate::cpu::test_support::*;
use crate::testkit::{HLT, Harness, SCRATCH};

const PA: u64 = SCRATCH;
/// Canonical with 57 bits (bit 56 clear and everything above it too) but not with 48: it needs
/// PML5 index 1, so a 4-level walk could not even name it.
const HIGH_LOW_HALF: u64 = 0x00ff_1234_5678_9000;
/// The first address above the 57-bit low half, and the last one below the high half.
const FIRST_NON_CANONICAL_57: u64 = 0x0100_0000_0000_0000;
const LAST_NON_CANONICAL_57: u64 = 0xfeff_ffff_ffff_ffff;
const FIRST_HIGH_CANONICAL_57: u64 = 0xff00_0000_0000_0000;
/// Non-canonical with 48 bits, canonical with 57.
const NON_CANONICAL_48: u64 = 0x0000_8000_0000_0000;
const CR4: u8 = 4;
const CR0: u8 = 0;
const CR0_PG_BIT: u64 = 1 << 31;
const EFER_LME_BIT: u64 = 1 << 8;
const EFER_LMA_BIT: u64 = 1 << 10;

fn five_level() -> (Harness, PageTables) {
    let mut h = Harness::new(&[HLT]);
    let tables = PageTables::new_five_level(&mut h.cpu);
    (h, tables)
}

/// `mov crN, rax` through the interpreter's own handler (CPL 0, no fetch involved).
fn write_cr(cpu: &mut Cpu, register: u8, value: u64) -> ExecResult<()> {
    let insn = decode::decode(&[0x0f, 0x22, 0xc0 | (register << 3)]).unwrap();
    cpu.gpr[0] = value;
    cpu.mov_cr_dr(&insn)
}

#[test]
fn a_four_kib_page_translates_through_five_levels() {
    let (mut h, mut tables) = five_level();
    tables.map_4k(&mut h.cpu, HIGH_LOW_HALF, PA, PTE_PRESENT_WRITABLE_USER);
    assert_eq!(
        h.cpu
            .translate(HIGH_LOW_HALF + 0x123, AccessKind::Read, false)
            .unwrap(),
        PA + 0x123
    );
    h.cpu.write(HIGH_LOW_HALF + 8, 8, 0x1122_3344).unwrap();
    assert_eq!(h.read_u64(PA + 8), 0x1122_3344);
}

#[test]
fn the_pml5_index_selects_the_pml4() {
    // Same PML4..PT indices, different PML5 index: two different pages.
    let (mut h, mut tables) = five_level();
    let low = HIGH_LOW_HALF & ((1 << 48) - 1);
    tables.map_4k(&mut h.cpu, low, PA, PTE_PRESENT_WRITABLE_USER);
    tables.map_4k(
        &mut h.cpu,
        HIGH_LOW_HALF,
        PA + PAGE,
        PTE_PRESENT_WRITABLE_USER,
    );
    assert_eq!(h.cpu.translate(low, AccessKind::Read, false).unwrap(), PA);
    assert_eq!(
        h.cpu
            .translate(HIGH_LOW_HALF, AccessKind::Read, false)
            .unwrap(),
        PA + PAGE
    );
}

#[test]
fn the_walk_sets_accessed_in_the_pml5_entry() {
    let (mut h, mut tables) = five_level();
    tables.map_4k(&mut h.cpu, HIGH_LOW_HALF, PA, PTE_PRESENT_WRITABLE_USER);
    h.cpu
        .translate(HIGH_LOW_HALF, AccessKind::Read, false)
        .unwrap();
    for level in 0..=4 {
        let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, HIGH_LOW_HALF, level));
        assert_ne!(entry & PTE_A, 0, "level {level}");
    }
}

#[test]
fn a_missing_pml5_entry_is_a_not_present_fault() {
    let (mut h, _tables) = five_level();
    let result = h.cpu.translate(HIGH_LOW_HALF, AccessKind::Write, false);
    assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_W)));
    assert_eq!(h.cpu.cr2, HIGH_LOW_HALF);
}

#[test]
fn page_size_in_a_pml5_or_pml4_entry_is_reserved() {
    for level in [4, 3] {
        let (mut h, mut tables) = five_level();
        tables.map_4k(&mut h.cpu, HIGH_LOW_HALF, PA, PTE_PRESENT_WRITABLE_USER);
        let address = tables.entry_address(&h.cpu, HIGH_LOW_HALF, level);
        let entry = read_pte(&h.cpu, address);
        write_pte(&mut h.cpu, address, entry | PTE_PS);
        let result = h.cpu.translate(HIGH_LOW_HALF, AccessKind::Read, false);
        assert_eq!(
            fault_of(result),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_RSVD)),
            "level {level}"
        );
    }
}

#[test]
fn user_and_write_permissions_are_anded_over_the_pml5_entry_too() {
    let (mut h, mut tables) = five_level();
    tables.map(
        &mut h.cpu,
        HIGH_LOW_HALF,
        PA,
        &[
            PTE_P | PTE_W,
            PTE_PRESENT_WRITABLE_USER,
            PTE_PRESENT_WRITABLE_USER,
            PTE_PRESENT_WRITABLE_USER,
            PTE_PRESENT_WRITABLE_USER,
        ],
    );
    h.cpu.set_cpl(3);
    let result = h.cpu.translate(HIGH_LOW_HALF, AccessKind::Read, false);
    assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
}

#[test]
fn large_leaves_still_work_under_five_levels() {
    let (mut h, mut tables) = five_level();
    let flags = [PTE_PRESENT_WRITABLE_USER; 4];
    tables.map(
        &mut h.cpu,
        HIGH_LOW_HALF & !(TWO_MIB - 1),
        0x40_0000,
        &flags,
    );
    assert_eq!(
        h.cpu
            .translate(HIGH_LOW_HALF, AccessKind::Read, false)
            .unwrap(),
        0x40_0000 + (HIGH_LOW_HALF & (TWO_MIB - 1))
    );
    let flags = [PTE_PRESENT_WRITABLE_USER; 3];
    tables.map(&mut h.cpu, 0x00f0_0000_0000_0000, 0, &flags);
    assert_eq!(
        h.cpu
            .translate(0x00f0_0000_1234_5000, AccessKind::Read, false)
            .unwrap(),
        0x1234_5000
    );
}

mod canonical {
    use super::*;

    #[test]
    fn addresses_beyond_48_bits_are_canonical_with_la57() {
        let (mut h, _tables) = five_level();
        // Canonical now, so they reach the (empty) walk and fault as not present.
        for address in [
            NON_CANONICAL_48,
            FIRST_HIGH_CANONICAL_57,
            0x00ff_ffff_ffff_f000,
        ] {
            let result = h.cpu.translate(address, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(0)),
                "{address:#x}"
            );
        }
    }

    #[test]
    fn addresses_beyond_57_bits_raise_gp0_and_ss0_through_the_stack() {
        let (mut h, _tables) = five_level();
        for address in [FIRST_NON_CANONICAL_57, LAST_NON_CANONICAL_57, 1 << 63] {
            assert_eq!(
                fault_of(h.cpu.read(address, 8)),
                (VECTOR_GENERAL_PROTECTION, Some(0)),
                "{address:#x}"
            );
            assert_eq!(
                fault_of(h.cpu.read_stack(address, 8)),
                (VECTOR_STACK_FAULT, Some(0)),
                "{address:#x}"
            );
        }
    }

    #[test]
    fn without_la57_the_48_bit_check_still_applies() {
        let mut h = Harness::new(&[HLT]);
        let result = h.cpu.translate(NON_CANONICAL_48, AccessKind::Read, false);
        assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
    }
}

mod control_registers {
    use super::*;

    #[test]
    fn la57_cannot_change_in_long_mode() {
        let mut h = Harness::new(&[HLT]);
        let cr4 = h.cpu.cr4;
        assert_eq!(
            fault_of(write_cr(&mut h.cpu, CR4, cr4 | CR4_LA57_BIT)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(h.cpu.cr4, cr4);
        h.cpu.cr4 |= CR4_LA57_BIT;
        assert_eq!(
            fault_of(write_cr(&mut h.cpu, CR4, cr4)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        // Rewriting it with LA57 unchanged is fine.
        write_cr(&mut h.cpu, CR4, cr4 | CR4_LA57_BIT).unwrap();
    }

    /// What Linux's trampoline does: leave long mode, set LA57 with paging off, point CR3 at a
    /// PML5 whose first entry is the old PML4, and turn paging back on.
    #[test]
    fn la57_set_with_paging_off_gives_five_level_paging_on_reentry() {
        let mut h = Harness::new(&[HLT]);
        let mut tables = PageTables::new_five_level(&mut h.cpu);
        tables.map_4k(&mut h.cpu, HIGH_LOW_HALF, PA, PTE_PRESENT_WRITABLE_USER);
        let pml5 = tables.pml4;
        let cr4 = h.cpu.cr4 & !CR4_LA57_BIT;
        h.cpu.cr4 = cr4;
        // Out of long mode as the trampoline does it (from compatibility mode).
        h.cpu.cr0 &= !CR0_PG_BIT;
        h.cpu.efer &= !EFER_LMA_BIT;
        h.cpu.flush_tlb();
        write_cr(&mut h.cpu, CR4, cr4 | CR4_LA57_BIT).unwrap();
        h.cpu.cr3 = pml5;
        assert_eq!(h.cpu.efer & EFER_LME_BIT, EFER_LME_BIT);
        // CS is a compatibility-mode segment in the real sequence.
        h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
        let cr0 = h.cpu.cr0;
        write_cr(&mut h.cpu, CR0, cr0 | CR0_PG_BIT).unwrap();
        assert_ne!(h.cpu.efer & EFER_LMA_BIT, 0);
        assert_eq!(
            h.cpu
                .translate(HIGH_LOW_HALF, AccessKind::Read, false)
                .unwrap(),
            PA
        );
    }
}
