//! Protection keys: the PKRU checks in the page walk, the soft TLB after a PKRU change, the
//! instructions, and PKRU as XSAVE component 9. Encodings are restated from the SDM rather than
//! taken from the code under test.

use super::*;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT};
use crate::cpu::mem::AccessKind;
use crate::cpu::test_support::*;
use crate::cpu::{TLB_ENTRIES_PER_MODE, TLB_TAG_INVALID};
use crate::testkit::{HLT, Harness, SCRATCH, Stop};

const VA: u64 = 0x0000_7f12_3456_0000;
const PA: u64 = SCRATCH;
/// The key the test pages carry, in leaf bits 62:59.
const KEY: u64 = 5;
const PTE_KEY_SHIFT: u64 = 59;
/// #PF error code bit 5: the fault came from a protection key.
const PF_PK: u32 = 1 << 5;
const CR4_PKE_BIT: u64 = 1 << 22;
const CR4_OSXSAVE_BIT: u64 = 1 << 18;
const CR4_OSFXSR_BIT: u64 = 1 << 9;
const CR0_WP_BIT: u64 = 1 << 16;
const ACCESS_DISABLE: u32 = 1;
const WRITE_DISABLE: u32 = 2;

fn pkru_for(key: u64, bits: u32) -> u32 {
    bits << (2 * key)
}

/// 4-level tables with VA mapped user-writable under `KEY`, and CR4.PKE on.
fn keyed() -> (Harness, PageTables) {
    let mut h = Harness::new(&[HLT]);
    let mut tables = PageTables::new(&mut h.cpu);
    let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
    write_pte(
        &mut h.cpu,
        leaf,
        PA | PTE_PRESENT_WRITABLE_USER | KEY << PTE_KEY_SHIFT,
    );
    h.cpu.cr4 |= CR4_PKE_BIT;
    (h, tables)
}

fn access(cpu: &mut Cpu, linear: u64, kind: AccessKind, user: bool) -> ExecResult<u64> {
    cpu.set_cpl(if user { 3 } else { 0 });
    let result = cpu.translate(linear, kind, false);
    cpu.set_cpl(0);
    result
}

mod page_walk {
    use super::*;

    #[test]
    fn access_disable_faults_data_accesses_at_any_cpl_with_the_pk_bit() {
        let (mut h, _tables) = keyed();
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE);
        let cases = [
            (AccessKind::Read, true, PF_P | PF_U | PF_PK),
            (AccessKind::Write, true, PF_P | PF_W | PF_U | PF_PK),
            (AccessKind::Read, false, PF_P | PF_PK),
            (AccessKind::Write, false, PF_P | PF_W | PF_PK),
        ];
        for (kind, user, code) in cases {
            let result = access(&mut h.cpu, VA, kind, user);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(code)),
                "{} {}",
                if user { "user" } else { "supervisor" },
                kind as u8
            );
            assert_eq!(h.cpu.cr2, VA);
        }
    }

    #[test]
    fn instruction_fetches_ignore_the_key() {
        let (mut h, _tables) = keyed();
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE | WRITE_DISABLE);
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Exec, true).unwrap(), PA);
    }

    #[test]
    fn write_disable_faults_user_writes_and_supervisor_writes_only_with_cr0_wp() {
        let (mut h, _tables) = keyed();
        h.cpu.pkru = pkru_for(KEY, WRITE_DISABLE);
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Read, true).unwrap(), PA);
        assert_eq!(
            fault_of(access(&mut h.cpu, VA, AccessKind::Write, true)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U | PF_PK))
        );
        h.cpu.cr0 &= !CR0_WP_BIT;
        h.cpu.flush_tlb();
        assert_eq!(
            access(&mut h.cpu, VA, AccessKind::Write, false).unwrap(),
            PA
        );
        h.cpu.cr0 |= CR0_WP_BIT;
        h.cpu.flush_tlb();
        assert_eq!(
            fault_of(access(&mut h.cpu, VA, AccessKind::Write, false)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_PK))
        );
    }

    #[test]
    fn the_pk_bit_joins_an_ordinary_permission_fault() {
        // Read-only page and a write-disabled key: both refuse the user store.
        let (mut h, tables) = keyed();
        let leaf = tables.entry_address(&h.cpu, VA, 0);
        write_pte(&mut h.cpu, leaf, PA | PTE_P | PTE_U | KEY << PTE_KEY_SHIFT);
        h.cpu.pkru = pkru_for(KEY, WRITE_DISABLE);
        assert_eq!(
            fault_of(access(&mut h.cpu, VA, AccessKind::Write, true)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U | PF_PK))
        );
    }

    #[test]
    fn supervisor_pages_ignore_the_key() {
        let (mut h, tables) = keyed();
        let leaf = tables.entry_address(&h.cpu, VA, 0);
        write_pte(&mut h.cpu, leaf, PA | PTE_P | PTE_W | KEY << PTE_KEY_SHIFT);
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE | WRITE_DISABLE);
        assert_eq!(
            access(&mut h.cpu, VA, AccessKind::Write, false).unwrap(),
            PA
        );
    }

    #[test]
    fn a_page_is_user_only_if_every_level_says_so() {
        // U clear in the PML4 entry: a supervisor page, whatever the leaf says.
        let (mut h, tables) = keyed();
        let pml4e = tables.entry_address(&h.cpu, VA, 3);
        let entry = read_pte(&h.cpu, pml4e);
        write_pte(&mut h.cpu, pml4e, entry & !PTE_U);
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE);
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Read, false).unwrap(), PA);
    }

    #[test]
    fn only_the_pages_own_key_counts() {
        let (mut h, _tables) = keyed();
        h.cpu.pkru = !pkru_for(KEY, ACCESS_DISABLE | WRITE_DISABLE);
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Write, true).unwrap(), PA);
    }

    #[test]
    fn key_zero_is_bits_one_and_zero() {
        let (mut h, tables) = keyed();
        let leaf = tables.entry_address(&h.cpu, VA, 0);
        write_pte(&mut h.cpu, leaf, PA | PTE_PRESENT_WRITABLE_USER);
        h.cpu.pkru = ACCESS_DISABLE;
        assert_eq!(
            fault_of(access(&mut h.cpu, VA, AccessKind::Read, true)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U | PF_PK))
        );
    }

    #[test]
    fn pkru_is_ignored_without_cr4_pke() {
        let (mut h, _tables) = keyed();
        h.cpu.cr4 &= !CR4_PKE_BIT;
        h.cpu.pkru = u32::MAX;
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Write, true).unwrap(), PA);
    }

    #[test]
    fn pkru_is_ignored_under_pae_paging() {
        let mut h = Harness::new(&[HLT]);
        let mut tables = PaeTables::new(&mut h.cpu);
        let leaf = tables.map(
            &mut h.cpu,
            0x1234_5000,
            PA,
            &[PTE_P, PTE_PRESENT_WRITABLE_USER, PTE_PRESENT_WRITABLE_USER],
        );
        write_pte(
            &mut h.cpu,
            leaf,
            PA | PTE_PRESENT_WRITABLE_USER | KEY << PTE_KEY_SHIFT,
        );
        h.cpu.cr4 |= CR4_PKE_BIT;
        h.cpu.pkru = u32::MAX;
        assert_eq!(
            access(&mut h.cpu, 0x1234_5000, AccessKind::Write, true).unwrap(),
            PA
        );
    }

    #[test]
    fn a_large_leaf_carries_its_key_and_its_cached_copy_too() {
        let mut h = Harness::new(&[HLT]);
        let mut tables = PageTables::new(&mut h.cpu);
        let flags = PTE_PRESENT_WRITABLE_USER;
        let pde = tables.map(&mut h.cpu, 0x4000_0000, 0x40_0000, &[flags; 3]);
        let entry = read_pte(&h.cpu, pde);
        write_pte(&mut h.cpu, pde, entry | KEY << PTE_KEY_SHIFT);
        h.cpu.cr4 |= CR4_PKE_BIT;
        access(&mut h.cpu, 0x4000_0000, AccessKind::Read, true).unwrap();
        h.cpu.set_pkru(pkru_for(KEY, ACCESS_DISABLE));
        // Another 4 KiB page of the same leaf: served from the large-leaf cache, if anything.
        assert_eq!(
            fault_of(access(&mut h.cpu, 0x4000_5000, AccessKind::Read, true)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U | PF_PK))
        );
        // The cache keeps the key: a restriction set without a flush still applies on a hit.
        h.cpu.pkru = 0;
        access(&mut h.cpu, 0x4000_6000, AccessKind::Read, true).unwrap();
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE);
        assert_eq!(
            fault_of(access(&mut h.cpu, 0x4000_7000, AccessKind::Read, true)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U | PF_PK))
        );
    }
}

mod soft_tlb {
    use super::*;

    #[test]
    fn a_cached_page_faults_once_wrpkru_takes_access_away() {
        let (mut h, _tables) = keyed();
        h.cpu.set_cpl(3);
        h.cpu.write(VA, 8, 0x1234).unwrap();
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x1234);
        h.cpu.set_pkru(pkru_for(KEY, ACCESS_DISABLE));
        assert_eq!(
            fault_of(h.cpu.read(VA, 8)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U | PF_PK))
        );
        h.cpu.set_pkru(pkru_for(KEY, WRITE_DISABLE));
        assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x1234);
        assert_eq!(
            fault_of(h.cpu.write(VA, 8, 0)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U | PF_PK))
        );
    }

    #[test]
    fn restricted_tags_are_withheld_so_the_fast_path_never_skips_the_check() {
        // Dirty already, so only the key can keep the write tag out.
        let (mut h, tables) = keyed();
        let leaf = tables.entry_address(&h.cpu, VA, 0);
        let entry = read_pte(&h.cpu, leaf);
        write_pte(&mut h.cpu, leaf, entry | PTE_A | PTE_D);
        let slot = TLB_ENTRIES_PER_MODE + (VA >> 12) as usize % TLB_ENTRIES_PER_MODE;
        h.cpu.pkru = pkru_for(KEY, WRITE_DISABLE);
        access(&mut h.cpu, VA, AccessKind::Read, true).unwrap();
        let entry = h.cpu.tlb[slot];
        assert_eq!((entry.tag_r, entry.tag_x), (VA, VA));
        assert_eq!(entry.tag_w, TLB_TAG_INVALID);
        h.cpu.flush_tlb();
        h.cpu.pkru = pkru_for(KEY, ACCESS_DISABLE);
        access(&mut h.cpu, VA, AccessKind::Exec, true).unwrap();
        let entry = h.cpu.tlb[slot];
        assert_eq!(entry.tag_x, VA, "fetches are not restricted");
        assert_eq!(
            (entry.tag_r, entry.tag_w),
            (TLB_TAG_INVALID, TLB_TAG_INVALID)
        );
        // Without a restriction the same page gets all three.
        h.cpu.flush_tlb();
        h.cpu.pkru = 0;
        access(&mut h.cpu, VA, AccessKind::Read, true).unwrap();
        assert_eq!(h.cpu.tlb[slot].tag_w, VA);
    }

    #[test]
    fn rewriting_the_same_value_keeps_the_tlb() {
        let (mut h, _tables) = keyed();
        access(&mut h.cpu, VA, AccessKind::Read, true).unwrap();
        h.cpu.set_pkru(0);
        let entry = h.cpu.tlb[TLB_ENTRIES_PER_MODE + (VA >> 12) as usize % TLB_ENTRIES_PER_MODE];
        assert_eq!(entry.tag_r, VA);
    }
}

mod instructions {
    use super::*;

    const RDPKRU: &[u8] = &[0x0f, 0x01, 0xee, HLT];
    const WRPKRU: &[u8] = &[0x0f, 0x01, 0xef, HLT];
    const RAX_INDEX: usize = 0;
    const RCX_INDEX: usize = 1;
    const RDX_INDEX: usize = 2;

    fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
        let mut h = Harness::new(code);
        h.cpu.cr4 |= CR4_PKE_BIT;
        setup(&mut h);
        let stop = h.run(10);
        (h, stop)
    }

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    #[test]
    fn both_are_undefined_without_cr4_pke() {
        for code in [RDPKRU, WRPKRU] {
            let (_, stop) = run(code, |h| h.cpu.cr4 &= !CR4_PKE_BIT);
            assert_eq!(stop, fault(6, None));
        }
    }

    #[test]
    fn wrpkru_loads_eax_and_rdpkru_returns_it_with_edx_cleared() {
        let (h, stop) = run(WRPKRU, |h| {
            h.cpu.gpr[RAX_INDEX] = 0xdead_beef_5555_5554;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.pkru, 0x5555_5554);
        let (h, stop) = run(RDPKRU, |h| {
            h.cpu.pkru = 0x5555_5554;
            h.cpu.gpr[RAX_INDEX] = u64::MAX;
            h.cpu.gpr[RDX_INDEX] = u64::MAX;
            // Only ECX is checked; the upper half of RCX is ignored.
            h.cpu.gpr[RCX_INDEX] = 1 << 32;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(
            (h.cpu.gpr[RAX_INDEX], h.cpu.gpr[RDX_INDEX]),
            (0x5555_5554, 0)
        );
    }

    #[test]
    fn nonzero_ecx_or_edx_is_gp0_and_leaves_pkru() {
        let (h, stop) = run(RDPKRU, |h| h.cpu.gpr[RCX_INDEX] = 1);
        assert_eq!(stop, fault(VECTOR_GENERAL_PROTECTION, Some(0)));
        assert_eq!(h.cpu.gpr[RDX_INDEX], 0);
        for (register, value) in [(RCX_INDEX, 1), (RDX_INDEX, 1)] {
            let (h, stop) = run(WRPKRU, |h| {
                h.cpu.gpr[RAX_INDEX] = 0xc;
                h.cpu.gpr[register] = value;
            });
            assert_eq!(stop, fault(VECTOR_GENERAL_PROTECTION, Some(0)));
            assert_eq!(h.cpu.pkru, 0);
        }
    }

    #[test]
    fn cr4_pke_is_writable_in_long_mode() {
        // mov cr4, rax
        let (h, stop) = run(&[0x0f, 0x22, 0xe0, HLT], |h| {
            h.cpu.cr4 &= !CR4_PKE_BIT;
            h.cpu.gpr[RAX_INDEX] = h.cpu.cr4 | CR4_PKE_BIT;
        });
        assert_eq!(stop, Stop::Halted);
        assert_ne!(h.cpu.cr4 & CR4_PKE_BIT, 0);
    }
}

mod xsave_component {
    use super::*;

    const XCR0_X87_SSE: u64 = 0b11;
    const XCR0_PKRU_BIT: u64 = 1 << 9;
    /// PKRU's place in the standard format (CPUID 0xD.9 EBX), and XSTATE_BV/XCOMP_BV.
    const PKRU_OFFSET: u64 = 2688;
    const XSTATE_BV: u64 = 512;
    const XCOMP_BV: u64 = 520;
    const LEGACY_AND_HEADER: u64 = 576;
    const AREA: u64 = SCRATCH + 0x1000;
    const XSAVE_RDI: &[u8] = &[0x0f, 0xae, 0x27, HLT];
    const XRSTOR_RDI: &[u8] = &[0x0f, 0xae, 0x2f, HLT];
    const XSAVEC_RDI: &[u8] = &[0x0f, 0xc7, 0x27, HLT];
    const XSETBV: &[u8] = &[0x0f, 0x01, 0xd1, HLT];
    const XGETBV: &[u8] = &[0x0f, 0x01, 0xd0, HLT];
    const RAX_INDEX: usize = 0;
    const RCX_INDEX: usize = 1;
    const RDX_INDEX: usize = 2;
    const RDI_INDEX: usize = 7;

    fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
        let mut h = Harness::new(code);
        h.cpu.cr4 |= CR4_OSFXSR_BIT | CR4_OSXSAVE_BIT | CR4_PKE_BIT;
        h.cpu.xcr0 = XCR0_X87_SSE | XCR0_PKRU_BIT;
        h.cpu.gpr[RDI_INDEX] = AREA;
        h.cpu.gpr[RAX_INDEX] = u32::MAX as u64;
        h.cpu.gpr[RDX_INDEX] = u32::MAX as u64;
        setup(&mut h);
        assert_eq!(h.run(10), Stop::Halted);
        h
    }

    #[test]
    fn xsetbv_accepts_pkru_with_or_without_the_vector_state() {
        for xcr0 in [
            1 | XCR0_PKRU_BIT,
            XCR0_X87_SSE | XCR0_PKRU_BIT,
            0xe7 | XCR0_PKRU_BIT,
        ] {
            let h = run(XSETBV, |h| {
                h.cpu.xcr0 = 1;
                h.cpu.gpr[RCX_INDEX] = 0;
                h.cpu.gpr[RAX_INDEX] = xcr0;
                h.cpu.gpr[RDX_INDEX] = 0;
            });
            assert_eq!(h.cpu.xcr0, xcr0);
        }
    }

    #[test]
    fn xsave_stores_pkru_and_marks_it_in_use() {
        let h = run(XSAVE_RDI, |h| h.cpu.pkru = 0x5555_5554);
        assert_eq!(h.read_u64(AREA + PKRU_OFFSET), 0x5555_5554);
        assert_ne!(h.read_u64(AREA + XSTATE_BV) & XCR0_PKRU_BIT, 0);
    }

    #[test]
    fn xrstor_loads_pkru_or_resets_it_by_xstate_bv() {
        let h = run(XRSTOR_RDI, |rig| {
            rig.write_u64(AREA + PKRU_OFFSET, 0xc);
            rig.write_u64(AREA + XSTATE_BV, XCR0_PKRU_BIT);
            rig.cpu.pkru = 0x5555_5554;
        });
        assert_eq!(h.cpu.pkru, 0xc);
        // In the bitmap but not in XSTATE_BV: back to the initial value, zero.
        let h = run(XRSTOR_RDI, |rig| {
            rig.write_u64(AREA + PKRU_OFFSET, 0xc);
            rig.cpu.pkru = 0x5555_5554;
        });
        assert_eq!(h.cpu.pkru, 0);
        // Not requested in EDX:EAX: untouched.
        let h = run(XRSTOR_RDI, |rig| {
            rig.write_u64(AREA + PKRU_OFFSET, 0xc);
            rig.write_u64(AREA + XSTATE_BV, XCR0_PKRU_BIT);
            rig.cpu.pkru = 0x5555_5554;
            rig.cpu.gpr[RAX_INDEX] = XCR0_X87_SSE;
        });
        assert_eq!(h.cpu.pkru, 0x5555_5554);
    }

    #[test]
    fn a_restoring_xrstor_takes_effect_at_the_next_access() {
        let (mut h, _tables) = keyed();
        h.cpu.set_cpl(3);
        h.cpu.read(VA, 8).unwrap();
        h.cpu.cr4 |= CR4_OSFXSR_BIT | CR4_OSXSAVE_BIT;
        h.cpu.xcr0 = XCR0_X87_SSE | XCR0_PKRU_BIT;
        h.cpu.ram[PA as usize..][..4096].fill(0);
        let area = PA;
        h.cpu.ram[(area + PKRU_OFFSET) as usize..][..4]
            .copy_from_slice(&pkru_for(KEY, ACCESS_DISABLE).to_le_bytes());
        h.cpu.ram[(area + XSTATE_BV) as usize..][..8].copy_from_slice(&XCR0_PKRU_BIT.to_le_bytes());
        h.cpu.gpr[RAX_INDEX] = XCR0_PKRU_BIT;
        h.cpu.gpr[RDX_INDEX] = 0;
        h.cpu.set_cpl(0);
        h.cpu.xrstor(VA).unwrap();
        h.cpu.set_cpl(3);
        assert_eq!(
            fault_of(h.cpu.read(VA, 8)),
            (VECTOR_PAGE_FAULT, Some(PF_P | PF_U | PF_PK))
        );
    }

    #[test]
    fn xsavec_packs_pkru_right_after_the_header_when_it_is_the_only_extended_component() {
        let h = run(XSAVEC_RDI, |h| h.cpu.pkru = 0x30);
        assert_eq!(h.read_u64(AREA + LEGACY_AND_HEADER), 0x30);
        assert_eq!(
            h.read_u64(AREA + XCOMP_BV),
            1 << 63 | XCR0_X87_SSE | XCR0_PKRU_BIT
        );
    }

    #[test]
    fn xinuse_reports_pkru_once_it_is_not_zero() {
        let h = run(XGETBV, |h| h.cpu.gpr[RCX_INDEX] = 1);
        assert_eq!(h.cpu.gpr[RAX_INDEX] & XCR0_PKRU_BIT, 0);
        let h = run(XGETBV, |h| {
            h.cpu.gpr[RCX_INDEX] = 1;
            h.cpu.pkru = 4;
        });
        assert_ne!(h.cpu.gpr[RAX_INDEX] & XCR0_PKRU_BIT, 0);
    }
}
