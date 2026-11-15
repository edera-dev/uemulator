//! CET shadow stacks (`cet.rs`): CPUID, CR4/MSR rules, shadow-stack pages, near and far
//! transfers, event delivery, the CET instructions, XSAVES state and the JIT fallback. Snippets
//! are assembled with `hack/asm.sh`; the assembly is in the comment above each.

use super::cet::*;
use super::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE, VECTOR_PAGE_FAULT};
use super::flags::CF;
use super::test_support::*;
use super::xsave::{XCR0_SSE, XCR0_X87, XSS_CET_S, XSS_CET_U};
use super::*;
use crate::machine::ExitReason;
use crate::testkit::{CODE_BASE, HLT, Harness, RAX, RBX, RCX, RDI, SCRATCH, Stop};

/// A 2 MiB region of the boot identity map that the tests turn into shadow-stack pages.
const SHADOW_REGION: u64 = 0xa0_0000;
/// Top of the shadow stack most tests start on.
const SHADOW_TOP: u64 = SHADOW_REGION + 0x1000;
/// A second shadow stack in the same region, for switches.
const OTHER_SHADOW_TOP: u64 = SHADOW_REGION + 0x3000;
/// Where tests that need a second piece of code (an iretq) put it.
const SIDE_CODE: u64 = SCRATCH + 0x8000;
const CR4_OSXSAVE_BIT: u64 = 1 << 18;
/// #PF error code bits as the architecture defines them.
const PF_SHADOW_STACK_BIT: u32 = 1 << 6;
const EFER_SCE_BIT: u64 = 1;
const MSR_STAR_NUMBER: u32 = 0xc000_0081;

// call f; hlt; f: ret
const CALL_RET: &[u8] = &[0xe8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xc3];
// call f; hlt; f: hlt
const CALL_THEN_STOP: &[u8] = &[0xe8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xf4];
// call f; hlt; f: add qword ptr [rsp], 1; ret
const CALL_TAMPERED_RET: &[u8] = &[
    0xe8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0x48, 0x83, 0x04, 0x24, 0x01, 0xc3,
];
// call 1f; 1: pop rax; hlt
const CALL_NEXT: &[u8] = &[0xe8, 0x00, 0x00, 0x00, 0x00, 0x58, 0xf4];
// mov ecx, 1000; 1: call f; dec ecx; jnz 1b; hlt; f: ret
const CALL_LOOP: &[u8] = &[
    0xb9, 0xe8, 0x03, 0x00, 0x00, 0xe8, 0x05, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf7, 0xf4, 0xc3,
];
// call qword ptr [rax]; hlt
const CALL_INDIRECT: &[u8] = &[0xff, 0x10, 0xf4];
// mov qword ptr [rax], rbx; hlt
const STORE: &[u8] = &[0x48, 0x89, 0x18, 0xf4];
// mov rbx, qword ptr [rax]; hlt
const LOAD: &[u8] = &[0x48, 0x8b, 0x18, 0xf4];
// mov cr4, rax; hlt
const MOV_CR4: &[u8] = &[0x0f, 0x22, 0xe0, 0xf4];
// mov cr0, rax; hlt
const MOV_CR0: &[u8] = &[0x0f, 0x22, 0xc0, 0xf4];
// rdsspq rax; hlt
const RDSSPQ: &[u8] = &[0xf3, 0x48, 0x0f, 0x1e, 0xc8, 0xf4];
// rdsspd eax; hlt
const RDSSPD: &[u8] = &[0xf3, 0x0f, 0x1e, 0xc8, 0xf4];
// incsspq rcx; hlt
const INCSSPQ: &[u8] = &[0xf3, 0x48, 0x0f, 0xae, 0xe9, 0xf4];
// wrssq [rax], rbx; hlt
const WRSSQ: &[u8] = &[0x48, 0x0f, 0x38, 0xf6, 0x18, 0xf4];
// wrussq [rax], rbx; hlt
const WRUSSQ: &[u8] = &[0x66, 0x48, 0x0f, 0x38, 0xf5, 0x18, 0xf4];
// setssbsy; hlt
const SETSSBSY: &[u8] = &[0xf3, 0x0f, 0x01, 0xe8, 0xf4];
// clrssbsy [rax]; hlt
const CLRSSBSY: &[u8] = &[0xf3, 0x0f, 0xae, 0x30, 0xf4];
// rstorssp [rax]; saveprevssp; hlt
const RSTORSSP_SAVEPREVSSP: &[u8] = &[0xf3, 0x0f, 0x01, 0x28, 0xf3, 0x0f, 0x01, 0xea, 0xf4];
// rstorssp [rax]; hlt
const RSTORSSP: &[u8] = &[0xf3, 0x0f, 0x01, 0x28, 0xf4];
// iretq
const IRETQ: &[u8] = &[0x48, 0xcf];
// xsaves64 [rdi]; hlt
const XSAVES64: &[u8] = &[0x48, 0x0f, 0xc7, 0x2f, 0xf4];
// xrstors64 [rdi]; hlt
const XRSTORS64: &[u8] = &[0x48, 0x0f, 0xc7, 0x1f, 0xf4];

fn fault(vector: u8, error_code: Option<u32>) -> Stop {
    Stop::Fault {
        vector,
        error_code,
        software: false,
    }
}

fn control_protection_stop(code: u32) -> Stop {
    fault(VECTOR_CONTROL_PROTECTION, Some(code))
}

/// The boot identity map's 2 MiB entry for `linear` (and its PML4 and PDPT entries).
fn identity_map_entries(h: &Harness, linear: u64) -> [u64; 3] {
    let pml4e = h.cpu.cr3 & PTE_ADDRESS_BITS;
    let pdpte = (read_pte(&h.cpu, pml4e) & PTE_ADDRESS_BITS) + ((linear >> 30) & 0x1ff) * 8;
    let pde = (read_pte(&h.cpu, pdpte) & PTE_ADDRESS_BITS) + ((linear >> 21) & 0x1ff) * 8;
    [pml4e, pdpte, pde]
}

/// Makes the 2 MiB region at `linear` a shadow-stack page: read-only and dirty. `user` also makes
/// it and the levels above it user-accessible (the other 2 MiB leaves stay supervisor pages).
fn make_shadow_stack_region(h: &mut Harness, linear: u64, user: bool) {
    let [pml4e, pdpte, pde] = identity_map_entries(h, linear);
    let leaf = (read_pte(&h.cpu, pde) & !PTE_W) | PTE_D | PTE_A;
    write_pte(&mut h.cpu, pde, if user { leaf | PTE_U } else { leaf });
    if user {
        for upper in [pml4e, pdpte] {
            let entry = read_pte(&h.cpu, upper);
            write_pte(&mut h.cpu, upper, entry | PTE_U);
        }
    }
    h.cpu.flush_tlb();
}

fn enable_cet(h: &mut Harness) {
    h.cpu.cr0 |= CR0_WP;
    h.cpu.cr4 |= CR4_CET;
}

/// Kernel shadow stacks on, SSP at `SHADOW_TOP` on a supervisor shadow-stack page.
fn kernel_shadow_stack(code: &[u8]) -> Harness {
    let mut h = Harness::new(code);
    enable_cet(&mut h);
    make_shadow_stack_region(&mut h, SHADOW_REGION, false);
    h.cpu.cet.s_cet = CET_SH_STK_EN;
    h.cpu.cet.ssp = SHADOW_TOP;
    h
}

/// A supervisor shadow-stack token at `address`, busy or not.
fn put_token(h: &mut Harness, address: u64, busy: bool) {
    h.write_u64(address, address | busy as u64);
}

mod identity {
    use super::*;

    fn cpuid(h: &mut Harness, leaf: u32, subleaf: u32) -> [u32; 4] {
        h.cpu.gpr[RAX] = leaf as u64;
        h.cpu.gpr[RCX] = subleaf as u64;
        h.cpu.cpuid();
        [RAX, RBX, RCX, 2].map(|register| h.cpu.gpr[register] as u32)
    }

    #[test]
    fn leaf_7_has_shadow_stacks_and_no_indirect_branch_tracking() {
        let mut h = Harness::new(&[HLT]);
        let [_, _, ecx, edx] = cpuid(&mut h, 7, 0);
        assert_ne!(ecx & (1 << 7), 0, "CET_SS");
        assert_eq!(edx & (1 << 20), 0, "no CET_IBT on Zen");
    }

    #[test]
    fn leaf_0xd_describes_the_cet_supervisor_components() {
        let mut h = Harness::new(&[HLT]);
        assert_eq!(cpuid(&mut h, 0xd, 11), [16, 0, 1, 0]);
        assert_eq!(cpuid(&mut h, 0xd, 12), [24, 0, 1, 0]);
        assert_eq!(cpuid(&mut h, 0xd, 1)[2], (XSS_CET_U | XSS_CET_S) as u32);
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        let without = cpuid(&mut h, 0xd, 1)[1];
        h.cpu.xss = XSS_CET_U | XSS_CET_S;
        assert_eq!(cpuid(&mut h, 0xd, 1)[1], without + 16 + 24);
    }
}

mod control_registers_and_msrs {
    use super::*;

    #[test]
    fn cr4_cet_needs_cr0_wp() {
        let mut h = Harness::new(MOV_CR4);
        h.cpu.gpr[RAX] = h.cpu.cr4 | CR4_CET;
        assert_eq!(h.run(10), fault(VECTOR_GENERAL_PROTECTION, Some(0)));
        let mut h = Harness::new(MOV_CR4);
        h.cpu.cr0 |= CR0_WP;
        h.cpu.gpr[RAX] = h.cpu.cr4 | CR4_CET;
        assert_eq!(h.run(10), Stop::Halted);
        assert_ne!(h.cpu.cr4 & CR4_CET, 0);
    }

    #[test]
    fn cr0_wp_cannot_be_cleared_under_cr4_cet() {
        let mut h = Harness::new(MOV_CR0);
        enable_cet(&mut h);
        h.cpu.gpr[RAX] = h.cpu.cr0 & !CR0_WP;
        assert_eq!(h.run(10), fault(VECTOR_GENERAL_PROTECTION, Some(0)));
        assert_ne!(h.cpu.cr0 & CR0_WP, 0);
    }

    #[test]
    fn the_msrs_reject_ibt_bits_and_bad_addresses_and_keep_the_rest() {
        let mut h = Harness::new(&[HLT]);
        for (msr, value) in [
            (MSR_U_CET, 1 << 2),
            (MSR_S_CET, 1 << 10),
            (MSR_PL0_SSP, 0x8000_0000_0000),
            (MSR_PL3_SSP, 0x1002),
            (MSR_ISST_ADDR, 0x0001_0000_0000_0000),
        ] {
            assert_eq!(
                fault_of(h.cpu.wrmsr(msr, value)),
                (VECTOR_GENERAL_PROTECTION, Some(0)),
                "{msr:#x} <- {value:#x}"
            );
        }
        for (msr, value) in [
            (MSR_U_CET, CET_SH_STK_EN | CET_WR_SHSTK_EN),
            (MSR_S_CET, CET_SH_STK_EN),
            (MSR_PL0_SSP, 0xffff_8000_0000_1000),
            (MSR_PL0_SSP + 1, 0x2000),
            (MSR_PL0_SSP + 2, 0x3004),
            (MSR_PL3_SSP, 0x7fff_ffff_f000),
            (MSR_ISST_ADDR, 0xffff_8000_0000_0123),
        ] {
            h.cpu.wrmsr(msr, value).unwrap();
            assert_eq!(h.cpu.rdmsr(msr).unwrap(), value, "{msr:#x}");
        }
        assert_eq!(
            h.cpu.cet.pl_ssp,
            [0xffff_8000_0000_1000, 0x2000, 0x3004, 0x7fff_ffff_f000]
        );
    }

    #[test]
    fn xss_accepts_exactly_the_cet_components() {
        let mut h = Harness::new(&[HLT]);
        h.cpu.wrmsr(0xda0, XSS_CET_U | XSS_CET_S).unwrap();
        assert_eq!(h.cpu.rdmsr(0xda0).unwrap(), XSS_CET_U | XSS_CET_S);
        assert!(h.cpu.wrmsr(0xda0, 1 << 8).is_err());
    }
}

mod pages {
    use super::*;

    #[test]
    fn a_shadow_stack_on_an_ordinary_page_faults_with_the_shadow_stack_bit() {
        let mut h = kernel_shadow_stack(CALL_RET);
        h.cpu.cet.ssp = SCRATCH + 0x1000;
        let rsp = h.cpu.gpr[GPR_RSP];
        assert_eq!(
            h.run(10),
            fault(VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_SHADOW_STACK_BIT))
        );
        assert_eq!(h.cpu.cr2, SCRATCH + 0x1000 - 8);
        assert_eq!(h.cpu.gpr[GPR_RSP], rsp, "the call is undone");
        assert_eq!(h.cpu.rip, CODE_BASE);
    }

    #[test]
    fn ordinary_stores_fault_on_a_shadow_stack_page_and_loads_do_not() {
        let mut h = kernel_shadow_stack(STORE);
        h.cpu.gpr[RAX] = SHADOW_REGION;
        assert_eq!(h.run(10), fault(VECTOR_PAGE_FAULT, Some(PF_P | PF_W)));
        let mut h = kernel_shadow_stack(LOAD);
        h.write_u64(SHADOW_REGION, 0x1234);
        h.cpu.gpr[RAX] = SHADOW_REGION;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[RBX], 0x1234);
    }

    #[test]
    fn a_kernel_shadow_stack_access_to_a_user_page_faults() {
        let mut h = kernel_shadow_stack(CALL_RET);
        make_shadow_stack_region(&mut h, SHADOW_REGION, true);
        assert_eq!(
            h.run(10),
            fault(VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_SHADOW_STACK_BIT))
        );
    }

    #[test]
    fn a_writable_level_above_the_leaf_is_needed() {
        // incssp only reads the shadow stack, so the read-only PDPT entry (which covers the
        // ordinary stack too) cannot fault any other access first.
        let mut h = kernel_shadow_stack(INCSSPQ);
        h.cpu.gpr[RCX] = 1;
        let [_, pdpte, _] = identity_map_entries(&h, SHADOW_REGION);
        let entry = read_pte(&h.cpu, pdpte);
        write_pte(&mut h.cpu, pdpte, entry & !PTE_W);
        h.cpu.flush_tlb();
        assert_eq!(
            h.run(10),
            fault(VECTOR_PAGE_FAULT, Some(PF_P | PF_SHADOW_STACK_BIT))
        );
    }
}

mod near_transfers {
    use super::*;

    #[test]
    fn call_pushes_the_return_address_on_the_shadow_stack() {
        let mut h = kernel_shadow_stack(CALL_THEN_STOP);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP - 8);
        assert_eq!(h.read_u64(SHADOW_TOP - 8), CODE_BASE + 5);
    }

    #[test]
    fn ret_pops_the_matching_shadow_entry() {
        let mut h = kernel_shadow_stack(CALL_RET);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.rip, CODE_BASE + 6, "halted after returning");
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP);
    }

    #[test]
    fn ret_to_a_changed_address_is_control_protection() {
        let mut h = kernel_shadow_stack(CALL_TAMPERED_RET);
        assert_eq!(h.run(10), control_protection_stop(CP_NEAR_RET));
        assert_eq!(h.cpu.rip, CODE_BASE + 11, "the ret faults");
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP - 8, "nothing popped");
    }

    #[test]
    fn call_to_the_next_instruction_leaves_the_shadow_stack_alone() {
        let mut h = kernel_shadow_stack(CALL_NEXT);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP);
    }

    #[test]
    fn indirect_call_pushes_too() {
        let mut h = kernel_shadow_stack(CALL_INDIRECT);
        h.write(SIDE_CODE, &[HLT]);
        h.write_u64(SCRATCH, SIDE_CODE);
        h.cpu.gpr[RAX] = SCRATCH;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.read_u64(SHADOW_TOP - 8), CODE_BASE + 2);
    }

    #[test]
    fn without_cet_nothing_touches_ssp() {
        let mut h = Harness::new(CALL_THEN_STOP);
        h.cpu.cet.s_cet = CET_SH_STK_EN;
        h.cpu.cet.ssp = SCRATCH;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, SCRATCH);
    }

    #[test]
    fn protected_mode_calls_push_four_byte_entries() {
        // call f; hlt; f: ret (32-bit code, paging off: any memory can hold the shadow stack)
        let mut h = crate::cpu::legacy_support::protected32(CALL_THEN_STOP);
        enable_cet(&mut h);
        h.cpu.cet.s_cet = CET_SH_STK_EN;
        h.cpu.cet.ssp = SCRATCH + 0x100;
        let start = h.cpu.rip;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, SCRATCH + 0x100 - 4);
        assert_eq!(
            h.read(SCRATCH + 0x100 - 4, 4),
            ((start + 5) as u32).to_le_bytes()
        );
    }
}

mod instructions {
    use super::*;

    #[test]
    fn rdssp_is_a_nop_without_a_shadow_stack() {
        let mut h = Harness::new(RDSSPQ);
        h.cpu.cet.ssp = 0x5000;
        h.cpu.gpr[RAX] = 0x1234;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[RAX], 0x1234);
    }

    #[test]
    fn rdssp_reads_ssp() {
        let mut h = kernel_shadow_stack(RDSSPQ);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[RAX], SHADOW_TOP);
        let mut h = kernel_shadow_stack(RDSSPD);
        h.cpu.gpr[RAX] = u64::MAX;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[RAX], SHADOW_TOP, "the 32-bit form zero-extends");
    }

    #[test]
    fn incssp_skips_entries_and_needs_a_shadow_stack() {
        let mut h = Harness::new(INCSSPQ);
        assert_eq!(h.run(10), fault(VECTOR_INVALID_OPCODE, None));
        let mut h = kernel_shadow_stack(INCSSPQ);
        h.cpu.cet.ssp = SHADOW_TOP - 32;
        h.cpu.gpr[RCX] = 0x103; // only the low byte counts
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP - 8);
    }

    #[test]
    fn incssp_reads_the_last_entry_it_skips() {
        let mut h = kernel_shadow_stack(INCSSPQ);
        h.cpu.cet.ssp = SHADOW_REGION + TWO_MIB - 8;
        h.cpu.gpr[RCX] = 2;
        assert_eq!(
            h.run(10),
            fault(VECTOR_PAGE_FAULT, Some(PF_P | PF_SHADOW_STACK_BIT))
        );
        assert_eq!(h.cpu.cr2, SHADOW_REGION + TWO_MIB);
        assert_eq!(h.cpu.cet.ssp, SHADOW_REGION + TWO_MIB - 8);
    }

    #[test]
    fn wrss_needs_wr_shstk_en_and_writes_the_shadow_stack() {
        let mut h = kernel_shadow_stack(WRSSQ);
        h.cpu.gpr[RAX] = SHADOW_TOP - 8;
        h.cpu.gpr[RBX] = 0xabcd;
        assert_eq!(h.run(10), fault(VECTOR_INVALID_OPCODE, None));
        let mut h = kernel_shadow_stack(WRSSQ);
        h.cpu.cet.s_cet |= CET_WR_SHSTK_EN;
        h.cpu.gpr[RAX] = SHADOW_TOP - 8;
        h.cpu.gpr[RBX] = 0xabcd;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.read_u64(SHADOW_TOP - 8), 0xabcd);
        let mut h = kernel_shadow_stack(WRSSQ);
        h.cpu.cet.s_cet |= CET_WR_SHSTK_EN;
        h.cpu.gpr[RAX] = SHADOW_TOP - 4;
        assert_eq!(h.run(10), fault(VECTOR_GENERAL_PROTECTION, Some(0)));
    }

    #[test]
    fn wruss_writes_a_user_shadow_stack_from_the_kernel_only() {
        let mut h = Harness::new(WRUSSQ);
        h.cpu.gpr[RAX] = SHADOW_TOP - 8;
        assert_eq!(
            h.run(10),
            fault(VECTOR_INVALID_OPCODE, None),
            "needs CR4.CET"
        );
        // Linux: CR4.CET with no kernel shadow stack, writing a user one.
        let mut h = Harness::new(WRUSSQ);
        enable_cet(&mut h);
        make_shadow_stack_region(&mut h, SHADOW_REGION, true);
        h.cpu.gpr[RAX] = SHADOW_TOP - 8;
        h.cpu.gpr[RBX] = 0x7777;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.read_u64(SHADOW_TOP - 8), 0x7777);
        // a supervisor shadow-stack page is not a user one
        let mut h = kernel_shadow_stack(WRUSSQ);
        h.cpu.gpr[RAX] = SHADOW_TOP - 8;
        assert_eq!(
            h.run(10),
            fault(
                VECTOR_PAGE_FAULT,
                Some(PF_P | PF_W | PF_U | PF_SHADOW_STACK_BIT)
            )
        );
    }

    #[test]
    fn wruss_from_user_mode_is_general_protection() {
        let mut h = Harness::new(&[HLT]);
        enable_cet(&mut h);
        enter_user_mode(&mut h);
        assert_eq!(
            fault_of(h.cpu.wruss(SHADOW_TOP - 8, 8, 0)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn setssbsy_claims_the_pl0_token_and_clrssbsy_releases_it() {
        let mut h = kernel_shadow_stack(SETSSBSY);
        h.cpu.cet.ssp = 0;
        h.cpu.cet.pl_ssp[0] = OTHER_SHADOW_TOP - 8;
        put_token(&mut h, OTHER_SHADOW_TOP - 8, false);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.ssp, OTHER_SHADOW_TOP - 8);
        assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8), (OTHER_SHADOW_TOP - 8) | 1);

        let mut again = kernel_shadow_stack(SETSSBSY);
        again.cpu.cet.pl_ssp[0] = OTHER_SHADOW_TOP - 8;
        put_token(&mut again, OTHER_SHADOW_TOP - 8, true);
        assert_eq!(again.run(10), control_protection_stop(CP_SETSSBSY));

        let mut h = kernel_shadow_stack(CLRSSBSY);
        put_token(&mut h, OTHER_SHADOW_TOP - 8, true);
        h.cpu.gpr[RAX] = OTHER_SHADOW_TOP - 8;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8), OTHER_SHADOW_TOP - 8);
        assert_eq!(h.cpu.eflags_full() & CF, 0);
        assert_eq!(h.cpu.cet.ssp, 0);

        let mut h = kernel_shadow_stack(CLRSSBSY);
        put_token(&mut h, OTHER_SHADOW_TOP - 8, false);
        h.cpu.gpr[RAX] = OTHER_SHADOW_TOP - 8;
        assert_eq!(h.run(10), Stop::Halted);
        assert_ne!(h.cpu.eflags_full() & CF, 0, "the token was not busy");
    }

    #[test]
    fn token_instructions_need_ring_0_and_a_kernel_shadow_stack() {
        let mut h = Harness::new(SETSSBSY);
        enable_cet(&mut h);
        assert_eq!(h.run(10), fault(VECTOR_INVALID_OPCODE, None));
        let mut h = kernel_shadow_stack(&[HLT]);
        enter_user_mode(&mut h);
        assert_eq!(
            fault_of(h.cpu.setssbsy()),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn rstorssp_and_saveprevssp_switch_stacks_and_leave_a_way_back() {
        let mut h = kernel_shadow_stack(RSTORSSP_SAVEPREVSSP);
        // A restore token for 64-bit code at the top of the other stack.
        h.write_u64(OTHER_SHADOW_TOP - 8, OTHER_SHADOW_TOP | 1);
        h.cpu.gpr[RAX] = OTHER_SHADOW_TOP - 8;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(
            h.cpu.cet.ssp, OTHER_SHADOW_TOP,
            "the previous-SSP token was popped"
        );
        assert_eq!(
            h.read_u64(SHADOW_TOP - 8),
            SHADOW_TOP | 1,
            "a restore token for the old stack"
        );

        let mut back = kernel_shadow_stack(RSTORSSP);
        back.cpu.cet.ssp = OTHER_SHADOW_TOP;
        back.write_u64(SHADOW_TOP - 8, SHADOW_TOP | 1);
        back.cpu.gpr[RAX] = SHADOW_TOP - 8;
        assert_eq!(back.run(10), Stop::Halted);
        assert_eq!(back.cpu.cet.ssp, SHADOW_TOP - 8);
        assert_eq!(back.read_u64(SHADOW_TOP - 8), OTHER_SHADOW_TOP | 1 | 2);
        assert_eq!(back.cpu.eflags_full() & CF, 0);
    }

    #[test]
    fn rstorssp_refuses_a_token_for_another_address_or_mode() {
        for token in [
            (OTHER_SHADOW_TOP + 8) | 1,
            OTHER_SHADOW_TOP,
            OTHER_SHADOW_TOP | 3,
        ] {
            let mut h = kernel_shadow_stack(RSTORSSP);
            h.write_u64(OTHER_SHADOW_TOP - 8, token);
            h.cpu.gpr[RAX] = OTHER_SHADOW_TOP - 8;
            assert_eq!(
                h.run(10),
                control_protection_stop(CP_RSTORSSP),
                "{token:#x}"
            );
            assert_eq!(h.cpu.cet.ssp, SHADOW_TOP);
            assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8), token);
        }
    }
}

mod events {
    use super::*;

    const VECTOR: u8 = 0x40;
    const IRETQ_AT: u64 = SIDE_CODE;

    /// Delivers `VECTOR` and then runs an iretq on the frame it built.
    fn deliver_then_iretq(h: &mut Harness) -> Stop {
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        h.write(IRETQ_AT, IRETQ);
        h.cpu.rip = IRETQ_AT;
        h.run(1)
    }

    fn kernel_with_idt(code: &[u8]) -> Harness {
        let mut h = kernel_shadow_stack(code);
        install_idt(&mut h);
        install_tss(&mut h, [crate::testkit::STACK_TOP, 0, 0], [0; 7]);
        h
    }

    #[test]
    fn an_event_at_the_same_level_pushes_cs_lip_and_ssp() {
        let mut h = kernel_with_idt(&[HLT]);
        let rip = h.cpu.rip;
        let cs = h.cpu.seg[SEG_CS].sel as u64;
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP - 24);
        assert_eq!(h.read_u64(SHADOW_TOP - 24), SHADOW_TOP, "the old SSP");
        assert_eq!(
            h.read_u64(SHADOW_TOP - 16),
            rip,
            "the linear return address"
        );
        assert_eq!(h.read_u64(SHADOW_TOP - 8), cs);
    }

    #[test]
    fn iretq_checks_and_pops_the_frame() {
        let mut h = kernel_with_idt(&[HLT]);
        let rip = h.cpu.rip;
        assert_eq!(deliver_then_iretq(&mut h), Stop::Limit);
        assert_eq!(h.cpu.rip, rip);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP);
    }

    #[test]
    fn iretq_to_a_different_address_is_control_protection() {
        let mut h = kernel_with_idt(&[HLT]);
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        let rsp = h.cpu.gpr[GPR_RSP];
        let rip = h.read_u64(rsp);
        h.write_u64(rsp, rip + 1);
        h.write(IRETQ_AT, IRETQ);
        h.cpu.rip = IRETQ_AT;
        assert_eq!(h.run(1), control_protection_stop(CP_FAR_RET_IRET));
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP - 24, "nothing popped");
    }

    #[test]
    fn from_user_mode_the_kernel_stack_is_claimed_and_released_on_return() {
        let mut h = kernel_with_idt(&[HLT]);
        const USER_SSP: u64 = 0x7000_0000;
        h.cpu.cet.u_cet = CET_SH_STK_EN;
        h.cpu.cet.pl_ssp[0] = OTHER_SHADOW_TOP - 8;
        put_token(&mut h, OTHER_SHADOW_TOP - 8, false);
        enter_user_mode(&mut h);
        h.cpu.cet.ssp = USER_SSP;
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        assert_eq!(h.cpu.cpl, 0);
        assert_eq!(h.cpu.cet.pl_ssp[3], USER_SSP);
        assert_eq!(
            h.cpu.cet.ssp,
            OTHER_SHADOW_TOP - 8,
            "nothing pushed from user mode"
        );
        assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8), (OTHER_SHADOW_TOP - 8) | 1);
        h.cpu.cet.pl_ssp[3] = USER_SSP + 0x40;
        h.write(IRETQ_AT, IRETQ);
        h.cpu.rip = IRETQ_AT;
        assert_eq!(h.run(1), Stop::Limit);
        assert_eq!(h.cpu.cpl, 3);
        assert_eq!(h.cpu.cet.ssp, USER_SSP + 0x40, "from PL3_SSP");
        assert_eq!(
            h.read_u64(OTHER_SHADOW_TOP - 8),
            OTHER_SHADOW_TOP - 8,
            "released"
        );
    }

    #[test]
    fn a_busy_kernel_token_makes_delivery_fail() {
        let mut h = kernel_with_idt(&[HLT]);
        h.cpu.cet.pl_ssp[0] = OTHER_SHADOW_TOP - 8;
        put_token(&mut h, OTHER_SHADOW_TOP - 8, true);
        enter_user_mode(&mut h);
        // #GP while delivering, #DF while delivering that, then a triple fault: every delivery
        // from ring 3 needs the token.
        assert_eq!(
            h.cpu.deliver_event(VECTOR, None, false),
            Some(ExitReason::TripleFault)
        );
    }

    #[test]
    fn an_ist_event_switches_through_the_interrupt_ssp_table_and_iretq_releases_it() {
        let mut h = kernel_with_idt(&[HLT]);
        let table = SCRATCH + 0x2000;
        install_tss(
            &mut h,
            [crate::testkit::STACK_TOP, 0, 0],
            [crate::testkit::STACK_TOP - 0x1000, 0, 0, 0, 0, 0, 0],
        );
        set_gate(
            &mut h,
            VECTOR,
            Gate {
                ist: 1,
                ..Gate::interrupt(handler_for(VECTOR))
            },
        );
        h.cpu.cet.isst_addr = table;
        h.write_u64(table + 8, OTHER_SHADOW_TOP - 8);
        put_token(&mut h, OTHER_SHADOW_TOP - 8, false);
        let rip = h.cpu.rip;
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        assert_eq!(h.cpu.cet.ssp, OTHER_SHADOW_TOP - 8 - 24);
        assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8 - 24), SHADOW_TOP);
        assert_eq!(h.read_u64(OTHER_SHADOW_TOP - 8), (OTHER_SHADOW_TOP - 8) | 1);
        h.write(IRETQ_AT, IRETQ);
        h.cpu.rip = IRETQ_AT;
        assert_eq!(h.run(1), Stop::Limit);
        assert_eq!(h.cpu.rip, rip);
        assert_eq!(h.cpu.cet.ssp, SHADOW_TOP);
        assert_eq!(
            h.read_u64(OTHER_SHADOW_TOP - 8),
            OTHER_SHADOW_TOP - 8,
            "released"
        );
    }

    #[test]
    fn protected_mode_events_pad_the_shadow_stack_and_iretd_checks_the_frame() {
        use crate::cpu::legacy_support::{LEGACY_CS, protected32, set_idt_gate};
        const INTERRUPT_GATE32_DPL0: u8 = 0x8e;
        // A 4-byte-aligned SSP, which only 32-bit code can have: the frame gets a zero dword.
        const SSP: u64 = SCRATCH + 0x104;
        let mut h = protected32(&[HLT]);
        enable_cet(&mut h);
        h.cpu.cet.s_cet = CET_SH_STK_EN;
        h.cpu.cet.ssp = SSP;
        set_idt_gate(
            &mut h,
            VECTOR,
            LEGACY_CS,
            IRETQ_AT as u32,
            INTERRUPT_GATE32_DPL0,
        );
        // iretd
        h.write(IRETQ_AT, &[0xcf]);
        let rip = h.cpu.rip;
        assert_eq!(h.cpu.deliver_event(VECTOR, None, false), None);
        let frame = SSP - 4 - 24;
        assert_eq!(h.cpu.cet.ssp, frame);
        assert_eq!(h.read(SSP - 4, 4), [0; 4], "alignment padding");
        assert_eq!(h.read_u64(frame), SSP);
        assert_eq!(h.read_u64(frame + 8), rip);
        assert_eq!(h.read_u64(frame + 16), LEGACY_CS as u64);
        assert_eq!(h.run(1), Stop::Limit);
        assert_eq!(h.cpu.rip, rip);
        assert_eq!(h.cpu.cet.ssp, SSP);
    }

    #[test]
    fn syscall_parks_the_user_ssp_and_sysret_restores_it() {
        let mut h = kernel_shadow_stack(&[HLT]);
        h.cpu.cet.u_cet = CET_SH_STK_EN;
        h.cpu.efer |= EFER_SCE_BIT;
        h.cpu
            .wrmsr(MSR_STAR_NUMBER, (0x10u64 << 32) | (0x1bu64 << 48))
            .unwrap();
        enter_user_mode(&mut h);
        h.cpu.cet.ssp = 0x7000_0000;
        h.cpu.do_syscall(CODE_BASE).unwrap();
        assert_eq!(h.cpu.cet.pl_ssp[3], 0x7000_0000);
        assert_eq!(h.cpu.cet.ssp, 0, "the kernel claims its own (setssbsy)");
        h.cpu.cet.pl_ssp[3] = 0x7000_1000;
        let sysretq = decode::decode(&[0x48, 0x0f, 0x07]).unwrap();
        h.cpu.do_sysret(&sysretq).unwrap();
        assert_eq!(h.cpu.cpl, 3);
        assert_eq!(h.cpu.cet.ssp, 0x7000_1000);
    }

    #[test]
    fn sysret_to_compatibility_mode_needs_a_32_bit_user_ssp() {
        let mut h = kernel_shadow_stack(&[HLT]);
        h.cpu.cet.u_cet = CET_SH_STK_EN;
        h.cpu.efer |= EFER_SCE_BIT;
        h.cpu.cet.pl_ssp[3] = 0x1_0000_0000;
        let sysretl = decode::decode(&[0x0f, 0x07]).unwrap();
        assert_eq!(
            fault_of(h.cpu.do_sysret(&sysretl)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
        assert_eq!(h.cpu.cpl, 0);
    }
}

mod xsaves_state {
    use super::*;

    const AREA: u64 = SCRATCH + 0x4000;
    /// The legacy region and header, then CET_U (16 bytes), then CET_S.
    const CET_U_OFFSET: u64 = 576;
    const CET_S_OFFSET: u64 = 592;

    fn with_xsaves(code: &[u8]) -> Harness {
        let mut h = Harness::new(code);
        h.cpu.cr4 |= CR4_OSXSAVE_BIT;
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        h.cpu.xss = XSS_CET_U | XSS_CET_S;
        h.cpu.gpr[RAX] = u32::MAX as u64;
        h.cpu.gpr[2] = u32::MAX as u64;
        h.cpu.gpr[RDI] = AREA;
        h
    }

    #[test]
    fn xsaves_stores_the_cet_components_after_the_user_ones() {
        let mut h = with_xsaves(XSAVES64);
        h.cpu.cet.u_cet = CET_SH_STK_EN;
        h.cpu.cet.pl_ssp = [0x1000, 0x2000, 0x3000, 0x4000];
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(
            h.read_u64(AREA + 520),
            (1 << 63) | XCR0_X87 | XCR0_SSE | XSS_CET_U | XSS_CET_S,
            "XCOMP_BV"
        );
        assert_eq!(h.read_u64(AREA + CET_U_OFFSET), CET_SH_STK_EN);
        assert_eq!(h.read_u64(AREA + CET_U_OFFSET + 8), 0x4000);
        assert_eq!(h.read_u64(AREA + CET_S_OFFSET), 0x1000);
        assert_eq!(h.read_u64(AREA + CET_S_OFFSET + 16), 0x3000);
    }

    #[test]
    fn xrstors_loads_them_back_and_checks_them() {
        let mut saved = with_xsaves(XSAVES64);
        saved.cpu.cet.u_cet = CET_SH_STK_EN;
        saved.cpu.cet.pl_ssp = [0x1000, 0x2000, 0x3000, 0x4000];
        assert_eq!(saved.run(10), Stop::Halted);
        let area = saved.read(AREA, 616);

        let mut h = with_xsaves(XRSTORS64);
        h.write(AREA, &area);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.cet.u_cet, CET_SH_STK_EN);
        assert_eq!(h.cpu.cet.pl_ssp, [0x1000, 0x2000, 0x3000, 0x4000]);

        let mut bad = with_xsaves(XRSTORS64);
        bad.write(AREA, &area);
        bad.write_u64(AREA + CET_U_OFFSET, 1 << 2);
        assert_eq!(bad.run(10), fault(VECTOR_GENERAL_PROTECTION, Some(0)));
        assert_eq!(bad.cpu.cet.u_cet, 0);
    }

    #[test]
    fn xsaves_without_xss_leaves_the_cet_state_out() {
        let mut h = with_xsaves(XSAVES64);
        h.cpu.xss = 0;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.read_u64(AREA + 520), (1 << 63) | XCR0_X87 | XCR0_SSE);
    }
}

mod jit {
    use super::*;
    use crate::testkit::assert_jit_matches_interpreter;

    #[test]
    fn the_run_loop_keeps_shadow_stack_code_on_the_interpreter() {
        let h = kernel_shadow_stack(&[HLT]);
        assert!(!h.cpu.jit_may_run());
        let mut h = Harness::new(&[HLT]);
        enable_cet(&mut h);
        h.cpu.cet.u_cet = CET_SH_STK_EN;
        assert!(
            h.cpu.jit_may_run(),
            "only the current level's shadow stack counts"
        );
    }

    #[test]
    fn calls_and_returns_match_with_and_without_a_shadow_stack() {
        assert_jit_matches_interpreter(CALL_LOOP, |h| {
            enable_cet(h);
            make_shadow_stack_region(h, SHADOW_REGION, false);
            h.cpu.cet.s_cet = CET_SH_STK_EN;
            h.cpu.cet.ssp = SHADOW_TOP;
        });
        assert_jit_matches_interpreter(CALL_LOOP, |h| {
            enable_cet(h);
            h.cpu.cet.ssp = SHADOW_TOP;
        });
        assert_jit_matches_interpreter(CALL_TAMPERED_RET, |h| {
            enable_cet(h);
            make_shadow_stack_region(h, SHADOW_REGION, false);
            h.cpu.cet.s_cet = CET_SH_STK_EN;
            h.cpu.cet.ssp = SHADOW_TOP;
        });
    }

    #[test]
    fn a_jitted_harness_raises_control_protection_too() {
        let mut h = kernel_shadow_stack(CALL_TAMPERED_RET);
        h.enable_jit();
        assert_eq!(h.run(100), control_protection_stop(CP_NEAR_RET));
    }
}
