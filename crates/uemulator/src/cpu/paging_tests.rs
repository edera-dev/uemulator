//! Paging formats other than IA-32e: paging off, 32-bit (with PSE and PSE-36) and PAE, and the
//! control-register writes that move between them. Expected values restate the SDM encodings
//! (Vol. 3A 4.3 to 4.5) instead of reusing the constants in `mem.rs`.

use super::*;
use crate::cpu::decode;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT};
use crate::cpu::test_support::*;
use crate::testkit::{HLT, Harness, SCRATCH};

const VA: u64 = 0x1234_5000;
const VA_4M_BASE: u64 = 0x4080_0000;
const PA: u64 = SCRATCH;
/// A 4 MiB-aligned physical base inside harness RAM (32 MiB).
const PA_4M: u64 = 0x80_0000;
/// A 2 MiB-aligned physical base inside harness RAM.
const PA_2M: u64 = 0xa0_0000;
const FIRST_PHYSICAL_ABOVE_4GIB: u64 = 1 << 32;
const SUPERVISOR_RW: u64 = PTE_P | PTE_W;
const USER_READ_ONLY: u64 = PTE_P | PTE_U;
const USER_RW: u64 = PTE_P | PTE_W | PTE_U;

const CR0: u8 = 0;
const CR3: u8 = 3;
const CR4: u8 = 4;
const MSR_EFER: u32 = 0xc000_0080;
const EFER_LME_BIT: u64 = 1 << 8;
const EFER_LMA_BIT: u64 = 1 << 10;
const CR0_PE_BIT: u64 = 1;
const CR0_WP_BIT: u64 = 1 << 16;
const CR4_PCIDE_BIT: u64 = 1 << 17;
const CR4_SMEP_BIT: u64 = 1 << 20;
const BIT_63: u64 = 1 << 63;

fn harness() -> Harness {
    Harness::new(&[HLT])
}

fn legacy() -> (Harness, LegacyTables) {
    let mut h = harness();
    let tables = LegacyTables::new(&mut h.cpu);
    (h, tables)
}

fn pae() -> (Harness, PaeTables) {
    let mut h = harness();
    let tables = PaeTables::new(&mut h.cpu);
    (h, tables)
}

fn read_access(cpu: &mut Cpu, linear: u64) -> ExecResult<u64> {
    cpu.translate(linear, AccessKind::Read, false)
}

fn access(cpu: &mut Cpu, linear: u64, kind: AccessKind, user: bool) -> ExecResult<u64> {
    cpu.set_cpl(if user { 3 } else { 0 });
    let result = cpu.translate(linear, kind, false);
    cpu.set_cpl(0);
    result
}

/// `mov crN, rax` executed through the interpreter's own handler (CPL 0, no fetch involved).
fn write_cr(cpu: &mut Cpu, register: u8, value: u64) -> ExecResult<()> {
    let insn = decode::decode(&[0x0f, 0x22, 0xc0 | (register << 3)]).unwrap();
    cpu.gpr[0] = value;
    cpu.mov_cr_dr(&insn)
}

mod paging_off {
    use super::*;

    fn unpaged() -> Harness {
        let mut h = harness();
        h.cpu.efer = 0;
        h.cpu.cr4 = 0;
        h.cpu.cr0 &= !CR0_PG_BIT;
        h.cpu.flush_tlb();
        h
    }

    #[test]
    fn linear_equals_physical_and_ignores_the_tables() {
        let mut h = unpaged();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Off);
        for linear in [0, 0x1234, 0x20_0123, 0x7ff_f000] {
            assert_eq!(read_access(&mut h.cpu, linear).unwrap(), linear);
        }
    }

    #[test]
    fn linear_addresses_wrap_at_32_bits() {
        let mut h = unpaged();
        assert_eq!(
            read_access(&mut h.cpu, 0x1_0000_1000 + 0x234).unwrap(),
            0x1234
        );
        // no canonical check outside long mode: a "non-canonical" 64-bit value is just truncated
        assert_eq!(
            read_access(&mut h.cpu, 0x8000_0000_0000_2000).unwrap(),
            0x2000
        );
    }

    #[test]
    fn user_writes_and_fetches_are_allowed_everywhere() {
        let mut h = unpaged();
        for kind in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
            assert_eq!(
                access(&mut h.cpu, 0x30_0000, kind, true).unwrap(),
                0x30_0000
            );
        }
    }

    #[test]
    fn smep_and_smap_do_not_make_unpaged_memory_uncacheable() {
        let mut h = unpaged();
        h.cpu.cr4 = CR4_SMEP_BIT | (1 << 21);
        read_access(&mut h.cpu, 0x30_0000).unwrap();
        let slot = h.cpu.tlb[tlb_slot(0x30_0000, false)];
        assert_eq!(slot.tag_r, 0x30_0000, "the translation was cached");
    }

    #[test]
    fn memory_is_readable_and_writable_through_the_identity_map() {
        let mut h = unpaged();
        h.cpu.write(0x30_0010, 4, 0xdead_beef).unwrap();
        assert_eq!(h.cpu.read(0x30_0010, 4).unwrap(), 0xdead_beef);
        assert_eq!(
            h.cpu.ram[0x30_0010..0x30_0014],
            0xdead_beefu32.to_le_bytes()
        );
    }
}

mod legacy_paging {
    use super::*;

    #[test]
    fn the_mode_is_legacy_with_pg_and_without_pae() {
        let (h, _) = legacy();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Legacy);
    }

    #[test]
    fn four_kib_mapping_translates_with_the_page_offset() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        for offset in [0, 1, 0x123, 0xabc, 0xfff] {
            assert_eq!(read_access(&mut h.cpu, VA + offset).unwrap(), PA + offset);
        }
    }

    #[test]
    fn upper_address_bits_are_dropped() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        assert_eq!(
            read_access(&mut h.cpu, (7 << 32) | VA | 0x10).unwrap(),
            PA + 0x10
        );
    }

    #[test]
    fn neighbouring_pages_map_independently() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        tables.map_4k(&mut h.cpu, VA + PAGE, PA + 5 * PAGE, USER_RW, USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA + 8).unwrap(), PA + 8);
        assert_eq!(
            read_access(&mut h.cpu, VA + PAGE + 8).unwrap(),
            PA + 5 * PAGE + 8
        );
    }

    #[test]
    fn the_page_table_is_used_for_the_right_directory_slot() {
        let (mut h, mut tables) = legacy();
        // two linear addresses 4 MiB apart share a PT index but not a PDE
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        tables.map_4k(&mut h.cpu, VA + FOUR_MIB, PA + PAGE, USER_RW, USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
        assert_eq!(read_access(&mut h.cpu, VA + FOUR_MIB).unwrap(), PA + PAGE);
    }

    #[test]
    fn memory_round_trips_through_a_mapped_page() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        h.cpu.write(VA + 0x40, 8, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(h.cpu.read(VA + 0x40, 8).unwrap(), 0x0123_4567_89ab_cdef);
        assert_eq!(
            h.cpu.ram[(PA + 0x40) as usize..][..8],
            0x0123_4567_89ab_cdefu64.to_le_bytes()
        );
    }

    #[test]
    fn page_table_entries_are_four_bytes_wide() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        // a stale upper half next to the entry must not matter
        h.cpu.ram[(pte + 4) as usize..][..4].copy_from_slice(&[0xff; 4]);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn four_mib_page_translates_every_offset() {
        let (mut h, mut tables) = legacy();
        tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW);
        for offset in [0, 0x1fff, 0x12_3456, 0x3f_ffff] {
            assert_eq!(
                read_access(&mut h.cpu, VA_4M_BASE + offset).unwrap(),
                PA_4M + offset
            );
        }
    }

    #[test]
    fn four_mib_page_reads_the_right_ram_bytes() {
        let (mut h, mut tables) = legacy();
        tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW);
        h.cpu.ram[(PA_4M + 0x2_0abc) as usize] = 0x5a;
        assert_eq!(h.cpu.read(VA_4M_BASE + 0x2_0abc, 1).unwrap(), 0x5a);
    }

    #[test]
    fn four_mib_page_without_pse_is_a_pointer_to_a_page_table() {
        let (mut h, mut tables) = legacy();
        // PS is ignored with CR4.PSE clear: bits 31:12 name a page table, here the zeroed one at
        // the start of the pool, so the page is not present
        let pde = tables.map_4m(&mut h.cpu, VA_4M_BASE, TABLE_POOL + 0x10_0000, USER_RW);
        h.cpu.cr4 &= !CR4_PSE_BIT;
        h.cpu.flush_tlb();
        let (vector, code) = fault_of(read_access(&mut h.cpu, VA_4M_BASE));
        assert_eq!((vector, code), (VECTOR_PAGE_FAULT, Some(0)));
        // and a real page table behind the same PDE works
        let table = TABLE_POOL + 0x10_0000;
        write_pte32(&mut h.cpu, table, PA | USER_RW);
        write_pte32(&mut h.cpu, pde, table | USER_RW);
        h.cpu.flush_tlb();
        assert_eq!(read_access(&mut h.cpu, VA_4M_BASE).unwrap(), PA);
    }

    #[test]
    fn pse_36_takes_physical_bits_39_to_32_from_pde_bits_20_to_13() {
        let (mut h, mut tables) = legacy();
        let physical = (0xa5 << 32) | PA_4M;
        let pde = tables.map_4m(&mut h.cpu, VA_4M_BASE, physical, USER_RW);
        assert_eq!(
            read_pte32(&h.cpu, pde) >> 13 & 0xff,
            0xa5,
            "the helper follows the SDM layout"
        );
        assert_eq!(
            read_access(&mut h.cpu, VA_4M_BASE + 0x1234).unwrap(),
            physical + 0x1234
        );
    }

    #[test]
    fn pse_36_bit_21_is_reserved() {
        let (mut h, mut tables) = legacy();
        let pde = tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW);
        let value = read_pte32(&h.cpu, pde) | (1 << 21);
        write_pte32(&mut h.cpu, pde, value);
        let (vector, code) = fault_of(read_access(&mut h.cpu, VA_4M_BASE));
        assert_eq!((vector, code), (VECTOR_PAGE_FAULT, Some(PF_P | PF_RSVD)));
    }

    #[test]
    fn the_reserved_check_does_not_apply_to_a_pde_that_points_at_a_table() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        let pde_address = tables.directory_entry_address(VA);
        let value = read_pte32(&h.cpu, pde_address) | (1 << 21);
        write_pte32(&mut h.cpu, pde_address, value);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn pat_bit_of_a_four_mib_page_is_not_reserved() {
        let (mut h, mut tables) = legacy();
        tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW | (1 << 12));
        assert_eq!(read_access(&mut h.cpu, VA_4M_BASE).unwrap(), PA_4M);
    }

    #[test]
    fn missing_directory_or_table_entry_is_a_not_present_fault() {
        let (mut h, mut tables) = legacy();
        let (vector, code) = fault_of(read_access(&mut h.cpu, VA));
        assert_eq!((vector, code), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, VA);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        let (vector, code) = fault_of(read_access(&mut h.cpu, VA + PAGE));
        assert_eq!((vector, code), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, VA + PAGE);
    }

    #[test]
    fn not_present_error_codes_carry_write_user_and_nothing_else() {
        let (mut h, _) = legacy();
        for (kind, user, expected) in [
            (AccessKind::Read, false, 0),
            (AccessKind::Write, false, PF_W),
            (AccessKind::Read, true, PF_U),
            (AccessKind::Write, true, PF_W | PF_U),
        ] {
            let (_, code) = fault_of(access(&mut h.cpu, VA, kind, user));
            assert_eq!(code, Some(expected), "{} {}", user, expected);
        }
    }

    #[test]
    fn instruction_fetch_bit_needs_smep_and_never_nxe_in_32_bit_paging() {
        let (mut h, _) = legacy();
        h.cpu.efer = EFER_NXE_BIT;
        let (_, code) = fault_of(read_access(&mut h.cpu, VA));
        assert_eq!(code, Some(0));
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Exec, false));
        assert_eq!(code, Some(0), "EFER.NXE has no effect on 32-bit paging");
        h.cpu.cr4 |= CR4_SMEP_BIT;
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Exec, false));
        assert_eq!(code, Some(PF_ID), "SMEP does report it");
    }

    #[test]
    fn pde_without_a_page_table_behind_ram_counts_as_not_present() {
        let (mut h, tables) = legacy();
        let pde = tables.directory_entry_address(VA);
        write_pte32(&mut h.cpu, pde, 0xfff0_0000 | USER_RW);
        let (_, code) = fault_of(read_access(&mut h.cpu, VA));
        assert_eq!(code, Some(0));
    }

    #[test]
    fn cr3_low_bits_are_ignored_by_the_walk() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        h.cpu.cr3 |= 0x18 | 0xfe0;
        h.cpu.flush_tlb();
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn read_sets_accessed_in_both_levels_and_never_dirty() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        let pde = tables.directory_entry_address(VA);
        read_access(&mut h.cpu, VA).unwrap();
        assert_eq!(read_pte32(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A);
        assert_eq!(read_pte32(&h.cpu, pte) & (PTE_A | PTE_D), PTE_A);
    }

    #[test]
    fn write_sets_dirty_on_the_leaf_only() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        let pde = tables.directory_entry_address(VA);
        access(&mut h.cpu, VA, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte32(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A);
        assert_eq!(read_pte32(&h.cpu, pte) & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn four_mib_leaf_gets_accessed_and_dirty_on_the_pde() {
        let (mut h, mut tables) = legacy();
        let pde = tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW);
        read_access(&mut h.cpu, VA_4M_BASE).unwrap();
        assert_eq!(read_pte32(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A);
        access(&mut h.cpu, VA_4M_BASE, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte32(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn accessed_and_dirty_updates_preserve_the_other_bits() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW | (1 << 9) | (1 << 3));
        access(&mut h.cpu, VA, AccessKind::Write, false).unwrap();
        assert_eq!(
            read_pte32(&h.cpu, pte),
            PA | USER_RW | (1 << 9) | (1 << 3) | PTE_A | PTE_D
        );
    }

    #[test]
    fn a_denied_access_leaves_accessed_and_dirty_untouched() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_READ_ONLY);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Write, true));
        assert_eq!(code, Some(PF_P | PF_W | PF_U));
        assert_eq!(read_pte32(&h.cpu, pte) & (PTE_A | PTE_D), 0);
    }

    #[test]
    fn a_reserved_violation_does_not_set_accessed() {
        let (mut h, mut tables) = legacy();
        let pde = tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW | (1 << 21));
        let _ = fault_of(read_access(&mut h.cpu, VA_4M_BASE));
        assert_eq!(read_pte32(&h.cpu, pde) & PTE_A, 0);
    }

    #[test]
    fn user_bit_is_the_and_of_both_levels() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_RW, USER_RW);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Read, true));
        assert_eq!(code, Some(PF_P | PF_U));
        tables.map_4k(&mut h.cpu, VA + FOUR_MIB, PA, USER_RW, SUPERVISOR_RW);
        let (_, code) = fault_of(access(&mut h.cpu, VA + FOUR_MIB, AccessKind::Read, true));
        assert_eq!(code, Some(PF_P | PF_U));
    }

    #[test]
    fn writable_bit_is_the_and_of_both_levels() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_READ_ONLY, USER_RW);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Write, true));
        assert_eq!(code, Some(PF_P | PF_W | PF_U));
    }

    #[test]
    fn leaf_permission_matrix() {
        for user_access in [false, true] {
            for kind in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                for write_protect in [false, true] {
                    for large in [false, true] {
                        for entry_user in [false, true] {
                            for entry_writable in [false, true] {
                                let (mut h, mut tables) = legacy();
                                if write_protect {
                                    h.cpu.cr0 |= CR0_WP_BIT;
                                } else {
                                    h.cpu.cr0 &= !CR0_WP_BIT;
                                }
                                let flags = PTE_P
                                    | if entry_user { PTE_U } else { 0 }
                                    | if entry_writable { PTE_W } else { 0 };
                                if large {
                                    tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, flags);
                                } else {
                                    tables.map_4k(&mut h.cpu, VA, PA, USER_RW, flags);
                                }
                                let linear = if large { VA_4M_BASE } else { VA };
                                let result = access(&mut h.cpu, linear, kind, user_access);
                                let denied = (user_access && !entry_user)
                                    || (kind == AccessKind::Write
                                        && !entry_writable
                                        && (user_access || write_protect));
                                let context = format!(
                                    "user={user_access} {} wp={write_protect} large={large} \
                                     u={entry_user} w={entry_writable}",
                                    match kind {
                                        AccessKind::Read => "read",
                                        AccessKind::Write => "write",
                                        AccessKind::Exec => "exec",
                                    }
                                );
                                if denied {
                                    let mut expected = PF_P;
                                    if kind == AccessKind::Write {
                                        expected |= PF_W;
                                    }
                                    if user_access {
                                        expected |= PF_U;
                                    }
                                    assert_eq!(fault_of(result), (14, Some(expected)), "{context}");
                                } else {
                                    result.unwrap_or_else(|e| panic!("{context}: {e:?}"));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn smep_blocks_supervisor_fetches_from_user_pages() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        h.cpu.cr4 |= CR4_SMEP_BIT;
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Exec, false));
        assert_eq!(code, Some(PF_P | PF_ID));
        access(&mut h.cpu, VA, AccessKind::Read, false).unwrap();
    }

    #[test]
    fn four_mib_pages_cache_in_the_tlb_per_four_kib_page() {
        let (mut h, mut tables) = legacy();
        tables.map_4m(&mut h.cpu, VA_4M_BASE, PA_4M, USER_RW);
        h.cpu.write(VA_4M_BASE + 0x3000, 4, 7).unwrap();
        h.cpu.write(VA_4M_BASE + 0x5000, 4, 9).unwrap();
        assert_eq!(h.cpu.read(VA_4M_BASE + 0x3000, 4).unwrap(), 7);
        assert_eq!(h.cpu.read(VA_4M_BASE + 0x5000, 4).unwrap(), 9);
        assert_eq!(h.cpu.ram[(PA_4M + 0x5000) as usize], 9);
    }

    #[test]
    fn invlpg_forgets_one_cached_translation() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
        write_pte32(&mut h.cpu, pte, (PA + PAGE) | USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA, "still cached");
        h.cpu.invlpg(VA);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA + PAGE);
    }

    #[test]
    fn a_cr3_write_flushes_cached_translations() {
        let (mut h, mut tables) = legacy();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        read_access(&mut h.cpu, VA).unwrap();
        write_pte32(&mut h.cpu, pte, (PA + PAGE) | USER_RW);
        write_cr(&mut h.cpu, CR3, tables.directory).unwrap();
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA + PAGE);
    }

    #[test]
    fn physical_addresses_above_4_gib_via_pse_36_are_reported_unmapped_not_cached() {
        let (mut h, mut tables) = legacy();
        tables.map_4m(&mut h.cpu, VA_4M_BASE, FIRST_PHYSICAL_ABOVE_4GIB, USER_RW);
        assert_eq!(
            read_access(&mut h.cpu, VA_4M_BASE + 0x10).unwrap(),
            FIRST_PHYSICAL_ABOVE_4GIB + 0x10
        );
        assert_eq!(
            h.cpu.read(VA_4M_BASE, 4).unwrap(),
            0xffff_ffff,
            "no RAM there"
        );
    }
}

mod pae_paging {
    use super::*;

    #[test]
    fn the_mode_is_pae_with_pg_pae_and_no_lma() {
        let (h, _) = pae();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Pae);
    }

    #[test]
    fn four_kib_mapping_translates_with_the_page_offset() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        for offset in [0, 1, 0x123, 0xabc, 0xfff] {
            assert_eq!(read_access(&mut h.cpu, VA + offset).unwrap(), PA + offset);
        }
    }

    #[test]
    fn each_of_the_four_gigabytes_has_its_own_pdpte() {
        let (mut h, mut tables) = pae();
        for quarter in 0..4u64 {
            let linear = quarter * ONE_GIB + 0x2000;
            tables.map_4k(&mut h.cpu, linear, PA + quarter * PAGE, USER_RW);
        }
        for quarter in 0..4u64 {
            let linear = quarter * ONE_GIB + 0x2000;
            assert_eq!(
                read_access(&mut h.cpu, linear + 4).unwrap(),
                PA + quarter * PAGE + 4
            );
        }
    }

    #[test]
    fn two_mib_leaf_translates_every_offset() {
        let (mut h, mut tables) = pae();
        tables.map(&mut h.cpu, 0x4060_0000, PA_2M, &[PTE_P, USER_RW]);
        for offset in [0, 0x1fff, 0x12_3456, 0x1f_ffff] {
            assert_eq!(
                read_access(&mut h.cpu, 0x4060_0000 + offset).unwrap(),
                PA_2M + offset
            );
        }
    }

    #[test]
    fn two_mib_leaf_gets_accessed_and_dirty_on_the_pde_only() {
        let (mut h, mut tables) = pae();
        let pde = tables.map(&mut h.cpu, 0x4060_0000, PA_2M, &[PTE_P, USER_RW]);
        let pdpte = tables.pdpte_address(0x4060_0000);
        access(&mut h.cpu, 0x4060_0000, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A | PTE_D);
        assert_eq!(
            read_pte(&h.cpu, pdpte) & (PTE_A | PTE_D),
            0,
            "PDPTEs have no accessed or dirty bits"
        );
    }

    #[test]
    fn two_mib_leaf_with_low_address_bits_set_is_reserved() {
        let (mut h, mut tables) = pae();
        let pde = tables.map(&mut h.cpu, 0x4060_0000, PA_2M, &[PTE_P, USER_RW]);
        let value = read_pte(&h.cpu, pde) | (1 << 13);
        write_pte(&mut h.cpu, pde, value);
        let (_, code) = fault_of(read_access(&mut h.cpu, 0x4060_0000));
        assert_eq!(code, Some(PF_P | PF_RSVD));
    }

    #[test]
    fn two_mib_leaf_with_pat_is_not_reserved() {
        let (mut h, mut tables) = pae();
        tables.map(
            &mut h.cpu,
            0x4060_0000,
            PA_2M,
            &[PTE_P, USER_RW | (1 << 12)],
        );
        assert_eq!(read_access(&mut h.cpu, 0x4060_0000).unwrap(), PA_2M);
    }

    #[test]
    fn four_kib_accessed_and_dirty_follow_the_long_mode_rules() {
        let (mut h, mut tables) = pae();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        read_access(&mut h.cpu, VA).unwrap();
        assert_eq!(read_pte(&h.cpu, pte) & (PTE_A | PTE_D), PTE_A);
        access(&mut h.cpu, VA, AccessKind::Write, false).unwrap();
        assert_eq!(read_pte(&h.cpu, pte) & (PTE_A | PTE_D), PTE_A | PTE_D);
    }

    #[test]
    fn missing_pdpte_is_a_not_present_fault() {
        let (mut h, _) = pae();
        let (vector, code) = fault_of(read_access(&mut h.cpu, 0xc000_0000));
        assert_eq!((vector, code), (VECTOR_PAGE_FAULT, Some(0)));
        assert_eq!(h.cpu.cr2, 0xc000_0000);
    }

    #[test]
    fn missing_pde_or_pte_is_not_present() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        let (_, code) = fault_of(read_access(&mut h.cpu, VA + PAGE));
        assert_eq!(code, Some(0));
        let (_, code) = fault_of(read_access(&mut h.cpu, VA + TWO_MIB));
        assert_eq!(code, Some(0));
    }

    #[test]
    fn pdpte_is_latched_not_reread() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        // remove the PDPTE in memory without reloading CR3: the CPU keeps using its copy
        write_pte(&mut h.cpu, tables.pdpte_address(VA), 0);
        h.cpu.flush_tlb();
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
        write_cr(&mut h.cpu, CR3, tables.pdpt).unwrap();
        let (_, code) = fault_of(read_access(&mut h.cpu, VA));
        assert_eq!(code, Some(0), "the CR3 write reloaded the PDPT");
    }

    #[test]
    fn pdpte_permission_bits_do_not_restrict_the_mapping() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        // present with only PWT and PCD (bits 3 and 4), which are the other defined bits
        let pdpte = read_pte(&h.cpu, tables.pdpte_address(VA));
        write_pte(&mut h.cpu, tables.pdpte_address(VA), pdpte | 0x18);
        write_cr(&mut h.cpu, CR3, tables.pdpt).unwrap();
        assert_eq!(access(&mut h.cpu, VA, AccessKind::Write, true).unwrap(), PA);
    }

    #[test]
    fn nx_is_a_reserved_bit_without_nxe_and_blocks_fetches_with_it() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW | PTE_NX);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Read, false));
        assert_eq!(code, Some(PF_P | PF_RSVD));
        h.cpu.efer = EFER_NXE_BIT;
        h.cpu.flush_tlb();
        access(&mut h.cpu, VA, AccessKind::Read, false).unwrap();
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Exec, false));
        assert_eq!(code, Some(PF_P | PF_ID));
    }

    #[test]
    fn every_address_bit_up_to_51_is_an_address_bit() {
        // CPUID 0x80000008 reports 52 physical address bits, so none of 51:12 is reserved.
        let (mut h, mut tables) = pae();
        let pte = tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        for bit in [40, 45, 51] {
            write_pte(&mut h.cpu, pte, PA | USER_RW | 1 << bit);
            h.cpu.flush_tlb();
            assert_eq!(
                read_access(&mut h.cpu, VA).unwrap(),
                PA | 1 << bit,
                "bit {bit}"
            );
        }
        // bits 52 to 62 are available to software
        write_pte(&mut h.cpu, pte, PA | USER_RW | 1 << 55);
        h.cpu.flush_tlb();
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn user_and_writable_bits_are_the_and_over_pde_and_pte() {
        let (mut h, mut tables) = pae();
        tables.map(&mut h.cpu, VA, PA, &[PTE_P, USER_READ_ONLY, USER_RW]);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Write, true));
        assert_eq!(code, Some(PF_P | PF_W | PF_U));
        tables.map(
            &mut h.cpu,
            VA + TWO_MIB,
            PA,
            &[PTE_P, SUPERVISOR_RW, USER_RW],
        );
        let (_, code) = fault_of(access(&mut h.cpu, VA + TWO_MIB, AccessKind::Read, true));
        assert_eq!(code, Some(PF_P | PF_U));
    }

    #[test]
    fn write_protect_decides_supervisor_writes_to_read_only_pages() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, PTE_P | PTE_U);
        h.cpu.cr0 &= !CR0_WP_BIT;
        access(&mut h.cpu, VA, AccessKind::Write, false).unwrap();
        h.cpu.cr0 |= CR0_WP_BIT;
        h.cpu.flush_tlb();
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Write, false));
        assert_eq!(code, Some(PF_P | PF_W));
    }

    #[test]
    fn upper_address_bits_are_dropped() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        assert_eq!(read_access(&mut h.cpu, (3 << 40) | VA | 8).unwrap(), PA + 8);
    }

    #[test]
    fn identity_map_runs_code_in_pae_mode() {
        let (mut h, mut tables) = pae();
        tables.identity_map(&mut h.cpu, 32);
        assert_eq!(h.run(1), crate::testkit::Stop::Halted);
    }

    #[test]
    fn identity_map_runs_code_in_legacy_mode() {
        let (mut h, mut tables) = legacy();
        tables.identity_map(&mut h.cpu, 32);
        assert_eq!(h.run(1), crate::testkit::Stop::Halted);
    }
}

mod long_mode_is_unchanged {
    use super::*;

    #[test]
    fn the_boot_state_is_long_mode_and_still_walks_four_levels() {
        let mut h = harness();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Long);
        let mut tables = PageTables::new(&mut h.cpu);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn non_canonical_addresses_still_fault_in_long_mode() {
        let mut h = harness();
        let (vector, _) = fault_of(read_access(&mut h.cpu, 0x0000_8000_0000_0000));
        assert_eq!(vector, VECTOR_GENERAL_PROTECTION);
    }

    #[test]
    fn nx_still_reports_the_fetch_bit() {
        let mut h = harness();
        let mut tables = PageTables::new(&mut h.cpu);
        h.cpu.efer |= EFER_NXE_BIT;
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW | PTE_NX);
        let (_, code) = fault_of(access(&mut h.cpu, VA, AccessKind::Exec, false));
        assert_eq!(code, Some(PF_P | PF_ID));
    }
}

mod control_register_transitions {
    use super::*;

    fn gp(result: ExecResult<()>) {
        assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
    }

    /// Protected mode, paging off, EFER 0: where software sits before it picks a paging format.
    fn unpaged() -> Harness {
        let mut h = harness();
        h.cpu.efer = 0;
        h.cpu.cr4 = 0;
        h.cpu.cr0 = CR0_PE_BIT | (1 << 4) | (1 << 5);
        h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
        h.cpu.flush_tlb();
        h
    }

    #[test]
    fn enabling_pg_selects_legacy_paging() {
        let mut h = unpaged();
        let tables = LegacyTables::new(&mut h.cpu);
        h.cpu.cr0 &= !CR0_PG_BIT;
        h.cpu.cr4 = 0;
        let cr0 = h.cpu.cr0 | CR0_PG_BIT;
        write_cr(&mut h.cpu, CR3, tables.directory).unwrap();
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Legacy);
        assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0);
    }

    #[test]
    fn enabling_pg_with_pae_selects_pae_paging_and_loads_the_pdptes() {
        let mut h = unpaged();
        let mut tables = PaeTables::new(&mut h.cpu);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        h.cpu.cr0 &= !CR0_PG_BIT;
        h.cpu.pdpte = [0; 4];
        let cr0 = h.cpu.cr0 | CR0_PG_BIT;
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Pae);
        assert_ne!(h.cpu.pdpte[(VA >> 30) as usize & 3] & PTE_P, 0);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn a_reserved_bit_in_a_present_pdpte_is_gp_and_changes_nothing() {
        let mut h = unpaged();
        let mut tables = PaeTables::new(&mut h.cpu);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        h.cpu.cr0 &= !CR0_PG_BIT;
        let pdpte_address = tables.pdpte_address(VA);
        for reserved in [1 << 1, 1 << 2, 1 << 5, 1 << 8, BIT_63] {
            let good = read_pte(&h.cpu, pdpte_address);
            write_pte(&mut h.cpu, pdpte_address, good | reserved);
            let before = h.cpu.cr0;
            gp(write_cr(&mut h.cpu, CR0, before | CR0_PG_BIT));
            assert_eq!(h.cpu.cr0, before, "CR0 unchanged for {reserved:#x}");
            write_pte(&mut h.cpu, pdpte_address, good);
        }
    }

    #[test]
    fn a_not_present_pdpte_may_hold_anything() {
        let mut h = unpaged();
        let tables = PaeTables::new(&mut h.cpu);
        h.cpu.cr0 &= !CR0_PG_BIT;
        write_pte(&mut h.cpu, tables.pdpt + 8, BIT_63 | 0x1e6);
        let cr0 = h.cpu.cr0 | CR0_PG_BIT;
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
    }

    #[test]
    fn a_cr3_write_in_pae_mode_latches_the_pdptes_and_masks_the_value() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        h.cpu.pdpte = [0; 4];
        write_cr(&mut h.cpu, CR3, tables.pdpt | 0x18 | 0x7).unwrap();
        assert_eq!(
            h.cpu.cr3,
            tables.pdpt | 0x18,
            "PWT and PCD kept, bits 2:0 dropped"
        );
        assert_ne!(h.cpu.pdpte[(VA >> 30) as usize & 3] & PTE_P, 0);
    }

    #[test]
    fn a_bad_pdpte_makes_the_cr3_write_gp_and_leaves_cr3_alone() {
        let (mut h, mut tables) = pae();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        let old_cr3 = h.cpu.cr3;
        let old_pdpte = h.cpu.pdpte;
        let other = tables.alloc(&mut h.cpu);
        write_pte(&mut h.cpu, other, PTE_P | (1 << 1));
        gp(write_cr(&mut h.cpu, CR3, other));
        assert_eq!(h.cpu.cr3, old_cr3);
        assert_eq!(h.cpu.pdpte, old_pdpte);
    }

    #[test]
    fn a_pdpt_only_needs_32_byte_alignment() {
        let (mut h, mut tables) = pae();
        let page = tables.alloc(&mut h.cpu);
        let pdpt = page + 0x60;
        write_pte(&mut h.cpu, pdpt, PTE_P | PA);
        write_cr(&mut h.cpu, CR3, pdpt).unwrap();
        assert_eq!(h.cpu.pdpte[0], PTE_P | PA);
    }

    #[test]
    fn a_cr4_write_that_selects_pae_paging_loads_the_pdptes() {
        let mut h = unpaged();
        let mut tables = PaeTables::new(&mut h.cpu);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        h.cpu.cr4 = 0;
        h.cpu.pdpte = [0; 4];
        write_cr(&mut h.cpu, CR4, CR4_PAE_BIT).unwrap();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Pae);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn a_bad_pdpte_makes_the_cr4_write_gp() {
        let mut h = unpaged();
        let tables = PaeTables::new(&mut h.cpu);
        h.cpu.cr4 = 0;
        write_pte(&mut h.cpu, tables.pdpt, PTE_P | (1 << 2));
        gp(write_cr(&mut h.cpu, CR4, CR4_PAE_BIT));
        assert_eq!(h.cpu.cr4, 0);
    }

    #[test]
    fn a_cr3_write_in_legacy_mode_keeps_bits_31_12_and_pwt_pcd() {
        let (mut h, tables) = legacy();
        write_cr(&mut h.cpu, CR3, tables.directory | 0xfff).unwrap();
        assert_eq!(h.cpu.cr3, tables.directory | 0x18);
    }

    #[test]
    fn cr3_above_32_bits_is_gp_outside_long_mode() {
        // `mov cr3` is 32 bits wide outside IA-32e mode, so this is the CR3 check itself.
        let (mut h, tables) = legacy();
        gp(h.cpu.write_cr3(tables.directory | (1 << 32)));
        let (mut h, tables) = pae();
        gp(h.cpu.write_cr3(tables.pdpt | (1 << 32)));
    }

    #[test]
    fn cr3_may_use_the_full_physical_width_in_long_mode() {
        let mut h = harness();
        let tables = PageTables::new(&mut h.cpu);
        write_cr(&mut h.cpu, CR3, tables.pml4 | (1 << 32)).unwrap();
        write_cr(&mut h.cpu, CR3, tables.pml4 | (1 << 51)).unwrap();
        gp(write_cr(&mut h.cpu, CR3, tables.pml4 | (1 << 52)));
    }

    #[test]
    fn a_cr3_write_in_legacy_mode_does_not_touch_pdpte() {
        let (mut h, tables) = legacy();
        h.cpu.pdpte = [PTE_P; 4];
        write_cr(&mut h.cpu, CR3, tables.directory).unwrap();
        assert_eq!(h.cpu.pdpte, [PTE_P; 4]);
    }

    #[test]
    fn pcid_needs_ia32e_mode() {
        let mut h = unpaged();
        gp(write_cr(&mut h.cpu, CR4, CR4_PAE_BIT | CR4_PCIDE_BIT));
        let (mut h, _) = legacy();
        gp(write_cr(&mut h.cpu, CR4, CR4_PCIDE_BIT));
    }

    #[test]
    fn pcid_is_accepted_in_long_mode_and_blocks_leaving_paging() {
        let mut h = harness();
        h.cpu.cr3 &= !0xfff;
        write_cr(&mut h.cpu, CR4, CR4_PAE_BIT | CR4_PCIDE_BIT).unwrap();
        h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
        gp(write_cr(&mut h.cpu, CR0, CR0_PE_BIT));
    }

    #[test]
    fn lme_cannot_change_while_paging_is_on() {
        let (mut h, _) = legacy();
        assert_eq!(
            fault_of(h.cpu.wrmsr(MSR_EFER, EFER_LME_BIT)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(h.cpu.efer & EFER_LME_BIT, 0);
        h.cpu.wrmsr(MSR_EFER, EFER_NXE_BIT).unwrap();
    }

    #[test]
    fn lme_can_be_set_and_cleared_with_paging_off() {
        let mut h = unpaged();
        h.cpu.wrmsr(MSR_EFER, EFER_LME_BIT).unwrap();
        assert_eq!(h.cpu.efer & EFER_LME_BIT, EFER_LME_BIT);
        h.cpu.wrmsr(MSR_EFER, 0).unwrap();
        assert_eq!(h.cpu.efer & EFER_LME_BIT, 0);
    }

    #[test]
    fn the_long_mode_entry_sequence_from_pae_paging_ends_in_ia32e_paging() {
        // PAE paging cannot be turned on with LME and then PG... the architectural sequence is:
        // PAE on, CR3 -> PML4, LME on, then PG on (all from paging off).
        let mut h = unpaged();
        let mut tables = PageTables::new(&mut h.cpu);
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW);
        h.cpu.cr0 = CR0_PE_BIT | (1 << 4) | (1 << 5);
        h.cpu.efer = 0;
        h.cpu.cr4 = 0;
        write_cr(&mut h.cpu, CR4, CR4_PAE_BIT).unwrap();
        write_cr(&mut h.cpu, CR3, tables.pml4).unwrap();
        h.cpu.wrmsr(MSR_EFER, EFER_LME_BIT).unwrap();
        write_cr(
            &mut h.cpu,
            CR0,
            CR0_PE_BIT | (1 << 4) | (1 << 5) | CR0_PG_BIT,
        )
        .unwrap();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Long);
        assert_ne!(h.cpu.efer & EFER_LMA_BIT, 0);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
    }

    #[test]
    fn activating_long_mode_from_a_64_bit_code_segment_is_gp() {
        let mut h = unpaged();
        h.cpu.seg[SEG_CS].flags |= SEG_LONG;
        write_cr(&mut h.cpu, CR4, CR4_PAE_BIT).unwrap();
        h.cpu.wrmsr(MSR_EFER, EFER_LME_BIT).unwrap();
        let cr0 = h.cpu.cr0 | CR0_PG_BIT;
        gp(write_cr(&mut h.cpu, CR0, cr0));
    }

    #[test]
    fn leaving_long_mode_returns_to_the_selected_legacy_format() {
        let mut h = harness();
        h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
        let cr0 = h.cpu.cr0 & !CR0_PG_BIT;
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
        assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0);
        assert_eq!(h.cpu.paging_mode(), PagingMode::Off);
        h.cpu.wrmsr(MSR_EFER, 0).unwrap();
        write_cr(&mut h.cpu, CR4, 0).unwrap();
        assert_eq!(h.cpu.paging_mode(), PagingMode::Off);
    }

    #[test]
    fn a_paging_mode_change_flushes_the_tlb_and_the_decode_cache() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        read_access(&mut h.cpu, VA).unwrap();
        let generation = h.cpu.decode_cache_generation;
        // PSE does not change the format: translations go, decoded code stays
        write_cr(&mut h.cpu, CR4, 0).unwrap();
        assert_eq!(h.cpu.decode_cache_generation, generation);
        assert_eq!(h.cpu.tlb[tlb_slot(VA, false)].tag_r, TLB_TAG_INVALID);
        // PG off does change it
        let cr0 = h.cpu.cr0 & !CR0_PG_BIT;
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
        assert_eq!(h.cpu.decode_cache_generation, generation + 1);
        assert_eq!(h.cpu.tlb[tlb_slot(VA, false)].tag_r, TLB_TAG_INVALID);
    }

    #[test]
    fn a_cached_translation_from_one_format_is_not_used_in_the_next() {
        let (mut h, mut tables) = legacy();
        tables.map_4k(&mut h.cpu, VA, PA, USER_RW, USER_RW);
        assert_eq!(read_access(&mut h.cpu, VA).unwrap(), PA);
        let cr0 = h.cpu.cr0 & !CR0_PG_BIT;
        write_cr(&mut h.cpu, CR0, cr0).unwrap();
        assert_eq!(
            read_access(&mut h.cpu, VA).unwrap(),
            VA,
            "paging off: identity"
        );
    }
}
