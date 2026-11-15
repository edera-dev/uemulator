//! The Zen 5 model MSRs of `model_msr.rs`, the instructions that read them (rdpru), and the EFER
//! and HWCR bits that come with the same CPUID leaves. Expected values are spelled out from the
//! AMD APM and the family 1Ah PPR, not taken from the constants under test.

use super::*;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE, VECTOR_PAGE_FAULT};
use crate::testkit::{CODE_BASE, HLT, Harness, RAX, RBX, RCX, RDX, RSI, SCRATCH, Stop};

const MSR_TSC: u32 = 0x10;
const MSR_MPERF: u32 = 0xe7;
const MSR_APERF: u32 = 0xe8;
const MSR_MPERF_READ_ONLY: u32 = 0xc000_00e7;
const MSR_APERF_READ_ONLY: u32 = 0xc000_00e8;
const MSR_EFER: u32 = 0xc000_0080;
const MSR_HWCR: u32 = 0xc001_0015;
const MSR_ZEN4_BP_CFG: u32 = 0xc001_102e;
const BP_CFG_BP_SPEC_REDUCE: u64 = 1 << 4;
const HWCR_CPUID_USER_DISABLE: u64 = 1 << 35;
const EFER_FFXSR: u64 = 1 << 14;
const EFER_TCE: u64 = 1 << 15;
const EFER_MCOMMIT: u64 = 1 << 17;
const EFER_RESERVED_16: u64 = 1 << 16;
const CR4_TSD: u64 = 1 << 2;
const CR4_OSFXSR: u64 = 1 << 9;
const CARRY: u64 = 1;
/// CF, PF, AF, ZF, SF and OF.
const ARITHMETIC_FLAGS: u64 = 0x8d5;
const IF_BIT: u64 = 1 << 9;
const DATA: u64 = SCRATCH + 0x1000;
const PTE_USER: u64 = 4;
const PTE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;

fn blank() -> Harness {
    Harness::new(&[HLT])
}

fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
    let mut h = Harness::new(code);
    setup(&mut h);
    let stop = h.run(1_000);
    (h, stop)
}

fn fault(vector: u8, error_code: Option<u32>) -> Stop {
    Stop::Fault {
        vector,
        error_code,
        software: false,
    }
}

fn gp0() -> Stop {
    fault(VECTOR_GENERAL_PROTECTION, Some(0))
}

fn ud() -> Stop {
    fault(VECTOR_INVALID_OPCODE, None)
}

/// CPL 3 with every page of the boot identity map user-accessible, so the snippet keeps running.
fn user_mode(h: &mut Harness) {
    let pml4 = h.cpu.cr3 & PTE_ADDRESS_MASK;
    let pml4_entry = h.read_u64(pml4);
    h.write_u64(pml4, pml4_entry | PTE_USER);
    let pdpt = pml4_entry & PTE_ADDRESS_MASK;
    for gib in 0..4 {
        let pdpt_entry = h.read_u64(pdpt + gib * 8);
        h.write_u64(pdpt + gib * 8, pdpt_entry | PTE_USER);
        let pd = pdpt_entry & PTE_ADDRESS_MASK;
        for slot in 0..512 {
            let entry = h.read_u64(pd + slot * 8);
            h.write_u64(pd + slot * 8, entry | PTE_USER);
        }
    }
    h.cpu.flush_tlb();
    h.cpu.set_cpl(3);
}

fn is_gp(result: ExecResult<impl core::fmt::Debug>) -> bool {
    matches!(
        result,
        Err(Exception::Fault {
            vector: VECTOR_GENERAL_PROTECTION,
            error_code: Some(0),
            ..
        })
    )
}

fn cpuid(h: &mut Harness, leaf: u32, subleaf: u32) -> [u32; 4] {
    h.cpu.gpr[RAX] = leaf as u64;
    h.cpu.gpr[RCX] = subleaf as u64;
    h.cpu.cpuid();
    [RAX, RBX, RCX, RDX].map(|register| h.cpu.gpr[register] as u32)
}

mod counters {
    use super::*;

    #[test]
    fn mperf_and_aperf_count_guest_nanoseconds_and_can_be_written() {
        let mut h = blank();
        h.cpu.icount = 1_000;
        let mperf = h.cpu.rdmsr(MSR_MPERF).unwrap();
        assert_eq!(mperf, h.cpu.now_ns(), "P0 is the TSC rate");
        assert_eq!(
            h.cpu.rdmsr(MSR_APERF).unwrap(),
            mperf,
            "no boost, no throttling"
        );
        h.cpu.wrmsr(MSR_MPERF, 5).unwrap();
        h.cpu.wrmsr(MSR_APERF, 7).unwrap();
        h.cpu.icount += 100;
        let elapsed = h.cpu.now_ns() - mperf;
        assert_eq!(h.cpu.rdmsr(MSR_MPERF).unwrap(), 5 + elapsed);
        assert_eq!(h.cpu.rdmsr(MSR_APERF).unwrap(), 7 + elapsed);
    }

    #[test]
    fn the_read_only_copies_read_the_same_and_refuse_writes() {
        let mut h = blank();
        h.cpu.icount = 500;
        h.cpu.wrmsr(MSR_APERF, 1 << 40).unwrap();
        assert_eq!(
            h.cpu.rdmsr(MSR_MPERF_READ_ONLY).unwrap(),
            h.cpu.rdmsr(MSR_MPERF).unwrap()
        );
        assert_eq!(h.cpu.rdmsr(MSR_APERF_READ_ONLY).unwrap(), 1 << 40);
        assert!(is_gp(h.cpu.wrmsr(MSR_MPERF_READ_ONLY, 0)));
        assert!(is_gp(h.cpu.wrmsr(MSR_APERF_READ_ONLY, 0)));
    }

    #[test]
    fn a_tsc_write_moves_the_tsc_and_later_reads_continue_from_there() {
        let mut h = blank();
        h.cpu.icount = 1_000;
        h.cpu.wrmsr(MSR_TSC, 1 << 50).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_TSC).unwrap(), 1 << 50);
        let before = h.cpu.now_ns();
        h.cpu.icount += 10;
        let elapsed = h.cpu.now_ns() - before;
        assert_eq!(h.cpu.tsc(), (1 << 50) + elapsed);
        // MPERF is not the TSC: it keeps its own count
        assert_eq!(h.cpu.rdmsr(MSR_MPERF).unwrap(), h.cpu.now_ns());
        h.cpu.wrmsr(MSR_TSC, 0).unwrap();
        assert_eq!(h.cpu.tsc(), 0);
    }

    #[test]
    fn rdtsc_sees_a_written_tsc() {
        // rdtsc; hlt
        let (h, stop) = run(&[0x0f, 0x31, HLT], |h| {
            h.cpu.wrmsr(MSR_TSC, 0x1234_5678_0000_0000).unwrap()
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.gpr[RDX], 0x1234_5678);
        assert!(h.cpu.gpr[RAX] < 0x1000, "only a few nanoseconds later");
    }
}

mod power {
    use super::*;

    const MSR_PSTATE_CURRENT_LIMIT: u32 = 0xc001_0061;
    const MSR_PSTATE_CONTROL: u32 = 0xc001_0062;
    const MSR_PSTATE_STATUS: u32 = 0xc001_0063;
    const MSR_PSTATE_DEFINITION_0: u32 = 0xc001_0064;
    const MSR_RAPL_POWER_UNIT: u32 = 0xc001_0299;
    const MSR_CORE_ENERGY: u32 = 0xc001_029a;
    const MSR_PACKAGE_ENERGY: u32 = 0xc001_029b;

    #[test]
    fn three_pstates_are_defined_with_p0_at_the_tsc_rate() {
        let mut h = blank();
        // CpuFid in 5 MHz steps (family 1Ah), enable in bit 63
        let definitions: Vec<u64> = (0..8)
            .map(|k| h.cpu.rdmsr(MSR_PSTATE_DEFINITION_0 + k).unwrap())
            .collect();
        assert_eq!(
            definitions,
            [1 << 63 | 200, 1 << 63 | 160, 1 << 63 | 120, 0, 0, 0, 0, 0]
        );
        // PstateMaxVal (6:4) is the last one, the current limit (2:0) P0
        assert_eq!(h.cpu.rdmsr(MSR_PSTATE_CURRENT_LIMIT).unwrap(), 2 << 4);
    }

    #[test]
    fn the_pstate_command_is_reported_back_within_the_limit() {
        let mut h = blank();
        assert_eq!(h.cpu.rdmsr(MSR_PSTATE_STATUS).unwrap(), 0);
        h.cpu.wrmsr(MSR_PSTATE_CONTROL, 1).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_PSTATE_CONTROL).unwrap(), 1);
        assert_eq!(h.cpu.rdmsr(MSR_PSTATE_STATUS).unwrap(), 1);
        h.cpu.wrmsr(MSR_PSTATE_CONTROL, 7).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_PSTATE_STATUS).unwrap(), 2, "clamped to P2");
        assert!(is_gp(h.cpu.wrmsr(MSR_PSTATE_CONTROL, 8)), "reserved bits");
        assert!(is_gp(h.cpu.wrmsr(MSR_PSTATE_STATUS, 0)), "read-only");
        assert!(is_gp(h.cpu.wrmsr(MSR_PSTATE_CURRENT_LIMIT, 0)), "read-only");
        // the definitions accept only themselves
        let p0 = h.cpu.rdmsr(MSR_PSTATE_DEFINITION_0).unwrap();
        h.cpu.wrmsr(MSR_PSTATE_DEFINITION_0, p0).unwrap();
        assert!(is_gp(h.cpu.wrmsr(MSR_PSTATE_DEFINITION_0, p0 + 1)));
    }

    #[test]
    fn rapl_reports_zen_units_and_energy_that_grows_with_time() {
        let mut h = blank();
        h.cpu.topology.logical_processors = 4;
        assert_eq!(h.cpu.rdmsr(MSR_RAPL_POWER_UNIT).unwrap(), 0x000a_1003);
        h.cpu.idle_skipped_ns = 1_000_000_000;
        let core = h.cpu.rdmsr(MSR_CORE_ENERGY).unwrap();
        let package = h.cpu.rdmsr(MSR_PACKAGE_ENERGY).unwrap();
        // one second: some watts, in 1/65536 J units, more for the package than one core
        assert!(core >= 65_536, "{core}");
        assert!(package > 4 * core, "{package} {core}");
        h.cpu.idle_skipped_ns += 1_000_000;
        assert!(h.cpu.rdmsr(MSR_CORE_ENERGY).unwrap() > core);
        assert!(h.cpu.rdmsr(MSR_PACKAGE_ENERGY).unwrap() <= 0xffff_ffff);
        for msr in [MSR_RAPL_POWER_UNIT, MSR_CORE_ENERGY, MSR_PACKAGE_ENERGY] {
            assert!(is_gp(h.cpu.wrmsr(msr, 0)), "{msr:#x} is read-only");
        }
    }
}

mod identity {
    use super::*;

    const MSR_PPIN_CONTROL: u32 = 0xc001_02f0;
    const MSR_PPIN: u32 = 0xc001_02f1;
    const MSR_MMIO_CONF_BASE: u32 = 0xc001_0058;
    const MSR_SEV_STATUS: u32 = 0xc001_0131;
    const MSR_SMM_CTL: u32 = 0xc001_0116;

    #[test]
    fn ppin_reads_only_once_enabled_and_the_control_locks() {
        let mut h = blank();
        assert_eq!(h.cpu.rdmsr(MSR_PPIN_CONTROL).unwrap(), 0);
        assert!(is_gp(h.cpu.rdmsr(MSR_PPIN)), "disabled");
        h.cpu.wrmsr(MSR_PPIN_CONTROL, 2).unwrap();
        let ppin = h.cpu.rdmsr(MSR_PPIN).unwrap();
        assert_ne!(ppin, 0);
        assert!(is_gp(h.cpu.wrmsr(MSR_PPIN, 0)), "read-only");
        assert!(is_gp(h.cpu.wrmsr(MSR_PPIN_CONTROL, 4)), "reserved bit");
        // lock it enabled: no further change
        h.cpu.wrmsr(MSR_PPIN_CONTROL, 3).unwrap();
        assert!(is_gp(h.cpu.wrmsr(MSR_PPIN_CONTROL, 1)));
        assert_eq!(h.cpu.rdmsr(MSR_PPIN).unwrap(), ppin);
    }

    #[test]
    fn the_mmio_configuration_base_is_the_mcfg_window() {
        let mut h = blank();
        // base 0xE000_0000, one bus (range field 0), enabled
        assert_eq!(h.cpu.rdmsr(MSR_MMIO_CONF_BASE).unwrap(), 0xe000_0001);
        h.cpu.wrmsr(MSR_MMIO_CONF_BASE, 0xe000_0001).unwrap();
        assert!(is_gp(h.cpu.wrmsr(MSR_MMIO_CONF_BASE, 0xf000_0001)));
    }

    #[test]
    fn sev_status_reads_zero_and_smm_ctl_does_not_exist() {
        let mut h = blank();
        assert_eq!(h.cpu.rdmsr(MSR_SEV_STATUS).unwrap(), 0);
        assert!(is_gp(h.cpu.wrmsr(MSR_SEV_STATUS, 1)));
        // leaf 0x8000_0021 EAX bit 9 (NoSmmCtlMSR) promises this
        assert!(is_gp(h.cpu.rdmsr(MSR_SMM_CTL)));
        assert!(is_gp(h.cpu.wrmsr(MSR_SMM_CTL, 0)));
    }

    #[test]
    fn configuration_msrs_start_as_on_hardware_and_hold_what_is_written() {
        let mut h = blank();
        for (msr, reset) in [
            (0xc001_0140, 0),        // OSVW_ID_Length: no errata described
            (0xc001_0141, 0),        // OSVW_Status
            (0xc001_0111, 0x3_0000), // SMM_BASE
            (0xc001_0112, 0),        // SMM_ADDR
            (0xc001_0113, 0),        // SMM_MASK
            (0xc001_1023, 0),        // TW_CFG
            (0xc001_1028, 0),        // FP_CFG
            (0xc000_0108, 0),        // PREFETCH_CONTROL
        ] {
            assert_eq!(h.cpu.rdmsr(msr).unwrap(), reset, "{msr:#x}");
            h.cpu.wrmsr(msr, 0x2f).unwrap();
            assert_eq!(h.cpu.rdmsr(msr).unwrap(), 0x2f, "{msr:#x}");
        }
    }

    /// KVM sets BP_CFG bit 4 on every CPU while it has guests when SRSO_BP_SPEC_REDUCE is
    /// advertised, with a read-modify-write that must stick.
    #[test]
    fn bp_cfg_holds_the_speculation_reduce_bit() {
        let mut h = blank();
        let value = h.cpu.rdmsr(MSR_ZEN4_BP_CFG).unwrap() | BP_CFG_BP_SPEC_REDUCE;
        h.cpu.wrmsr(MSR_ZEN4_BP_CFG, value).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_ZEN4_BP_CFG).unwrap(), value);
    }
}

mod cpuid_override {
    use super::*;

    const MSR_CPUID_FN7: u32 = 0xc001_1002;
    const MSR_CPUID_FN1: u32 = 0xc001_1004;
    const MSR_CPUID_EXT1: u32 = 0xc001_1005;
    /// Leaf 7 EBX bit 18, the bit Linux clears through 0xC001_1002 on Zen 5 with old microcode.
    const RDSEED_BIT: u32 = 18;
    /// Leaf 1 ECX bit 30, cleared by Linux through bit 62 of 0xC001_1004.
    const RDRAND_BIT: u32 = 30;
    /// Leaf 0x8000_0001 ECX bit 22, which Linux sets through bit 54 of 0xC001_1005.
    const TOPOEXT_BIT: u32 = 22;

    #[test]
    fn reads_are_the_cpuid_registers_high_then_low() {
        let mut h = blank();
        let [_, _, ecx, edx] = cpuid(&mut h, 1, 0);
        assert_eq!(
            h.cpu.rdmsr(MSR_CPUID_FN1).unwrap(),
            (ecx as u64) << 32 | edx as u64
        );
        let [eax, ebx, _, _] = cpuid(&mut h, 7, 0);
        assert_eq!(
            h.cpu.rdmsr(MSR_CPUID_FN7).unwrap(),
            (eax as u64) << 32 | ebx as u64
        );
        let [_, _, ecx, edx] = cpuid(&mut h, 0x8000_0001, 0);
        assert_eq!(
            h.cpu.rdmsr(MSR_CPUID_EXT1).unwrap(),
            (ecx as u64) << 32 | edx as u64
        );
    }

    #[test]
    fn linux_hides_rdseed_and_rdrand_by_clearing_their_bits() {
        let mut h = blank();
        assert_ne!(cpuid(&mut h, 7, 0)[1] & 1 << RDSEED_BIT, 0);
        let fn7 = h.cpu.rdmsr(MSR_CPUID_FN7).unwrap();
        h.cpu
            .wrmsr(MSR_CPUID_FN7, fn7 & !(1 << RDSEED_BIT))
            .unwrap();
        let [eax, ebx, ecx, edx] = cpuid(&mut h, 7, 0);
        assert_eq!(ebx & 1 << RDSEED_BIT, 0);
        assert_eq!(
            (eax as u64) << 32 | ebx as u64,
            fn7 & !(1 << RDSEED_BIT),
            "only that bit went"
        );
        assert_ne!(ecx | edx, 0, "registers outside the MSR are untouched");
        assert_eq!(cpuid(&mut h, 7, 1)[0], bits_of_subleaf_1(&mut blank()));

        let fn1 = h.cpu.rdmsr(MSR_CPUID_FN1).unwrap();
        h.cpu
            .wrmsr(MSR_CPUID_FN1, fn1 & !(1 << (32 + RDRAND_BIT)))
            .unwrap();
        assert_eq!(cpuid(&mut h, 1, 0)[2] & 1 << RDRAND_BIT, 0);
        // and back
        h.cpu.wrmsr(MSR_CPUID_FN1, fn1).unwrap();
        assert_ne!(cpuid(&mut h, 1, 0)[2] & 1 << RDRAND_BIT, 0);
    }

    fn bits_of_subleaf_1(h: &mut Harness) -> u32 {
        cpuid(h, 7, 1)[0]
    }

    #[test]
    fn a_mask_cannot_add_a_feature_the_cpu_lacks() {
        let mut h = blank();
        h.cpu.wrmsr(MSR_CPUID_FN1, u64::MAX).unwrap();
        h.cpu.wrmsr(MSR_CPUID_EXT1, u64::MAX).unwrap();
        let mut reference = blank();
        assert_eq!(cpuid(&mut h, 1, 0), cpuid(&mut reference, 1, 0));
        assert_eq!(
            cpuid(&mut h, 0x8000_0001, 0),
            cpuid(&mut reference, 0x8000_0001, 0)
        );
        // the read reports what CPUID reports, not the mask
        let [_, _, ecx, edx] = cpuid(&mut reference, 1, 0);
        assert_eq!(
            h.cpu.rdmsr(MSR_CPUID_FN1).unwrap(),
            (ecx as u64) << 32 | edx as u64
        );
    }

    #[test]
    fn topoext_set_through_the_extended_mask_is_already_there() {
        let mut h = blank();
        let ext1 = h.cpu.rdmsr(MSR_CPUID_EXT1).unwrap();
        assert_ne!(ext1 & 1 << (32 + TOPOEXT_BIT), 0);
        h.cpu
            .wrmsr(MSR_CPUID_EXT1, ext1 | 1 << (32 + TOPOEXT_BIT))
            .unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_CPUID_EXT1).unwrap(), ext1);
    }

    #[test]
    fn osxsave_still_follows_cr4_under_an_untouched_mask() {
        const OSXSAVE_BIT: u32 = 27;
        const CR4_OSXSAVE: u64 = 1 << 18;
        let mut h = blank();
        h.cpu.cr4 |= CR4_OSXSAVE;
        assert_ne!(cpuid(&mut h, 1, 0)[2] & 1 << OSXSAVE_BIT, 0);
        assert_ne!(
            h.cpu.rdmsr(MSR_CPUID_FN1).unwrap() & 1 << (32 + OSXSAVE_BIT),
            0
        );
    }
}

mod rdpru {
    use super::*;

    /// rdpru; hlt
    const RDPRU: &[u8] = &[0x0f, 0x01, 0xfd, HLT];

    #[test]
    fn ecx_0_and_1_read_mperf_and_aperf_and_set_cf() {
        for (selector, msr) in [(0, MSR_MPERF), (1, MSR_APERF)] {
            let (h, stop) = run(RDPRU, |h| {
                h.cpu.wrmsr(msr, 0x0000_0123_0000_0000).unwrap();
                h.cpu.gpr[RCX] = 0xffff_ffff_0000_0000 | selector;
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
                h.cpu.set_eflags_arith(ARITHMETIC_FLAGS & !CARRY);
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.gpr[RDX], 0x123, "high half, zero-extended");
            assert!(h.cpu.gpr[RAX] < 0x1000, "a few nanoseconds");
            assert_eq!(h.cpu.eflags_full() & ARITHMETIC_FLAGS, CARRY);
        }
    }

    #[test]
    fn an_unknown_register_reads_zero_with_cf_clear() {
        let (h, stop) = run(RDPRU, |h| {
            h.cpu.gpr[RCX] = 2;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
            h.cpu.set_eflags_arith(ARITHMETIC_FLAGS);
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RDX]), (0, 0));
        assert_eq!(h.cpu.eflags_full() & ARITHMETIC_FLAGS, 0);
    }

    #[test]
    fn it_runs_at_cpl3_unless_cr4_tsd_makes_it_ud() {
        // the hlt behind it faults at CPL 3, after rdpru ran
        let (h, stop) = run(RDPRU, |h| {
            user_mode(h);
            h.cpu.gpr[RCX] = 0;
        });
        assert_eq!(stop, gp0());
        assert_eq!(h.cpu.rip, CODE_BASE + 3);
        let (h, stop) = run(RDPRU, |h| {
            user_mode(h);
            h.cpu.cr4 |= CR4_TSD;
        });
        assert_eq!(stop, ud());
        assert_eq!(h.cpu.rip, CODE_BASE);
        let (_, stop) = run(RDPRU, |h| h.cpu.cr4 |= CR4_TSD);
        assert_eq!(stop, Stop::Halted, "CPL 0 ignores TSD");
    }

    #[test]
    fn the_jit_matches_the_interpreter() {
        // rdpru; mov rbx, rax; mov ecx, 1; rdpru; hlt
        let code = [
            0x0f, 0x01, 0xfd, 0x48, 0x89, 0xc3, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x01, 0xfd, HLT,
        ];
        crate::testkit::assert_jit_matches_interpreter(&code, |h| h.cpu.gpr[RCX] = 0);
    }
}

mod efer_and_hwcr {
    use super::*;

    /// mcommit; hlt
    const MCOMMIT: &[u8] = &[0xf3, 0x0f, 0x01, 0xfa, HLT];
    /// cpuid; hlt
    const CPUID: &[u8] = &[0x0f, 0xa2, HLT];

    #[test]
    fn ffxsr_tce_and_mcommit_are_efer_bits_that_stick() {
        let mut h = blank();
        let efer = h.cpu.rdmsr(MSR_EFER).unwrap();
        for bit in [EFER_FFXSR, EFER_TCE, EFER_MCOMMIT] {
            h.cpu.wrmsr(MSR_EFER, efer | bit).unwrap();
            assert_eq!(h.cpu.rdmsr(MSR_EFER).unwrap(), efer | bit, "{bit:#x}");
        }
        assert!(is_gp(h.cpu.wrmsr(MSR_EFER, efer | EFER_RESERVED_16)));
    }

    #[test]
    fn mcommit_is_ud_until_efer_enables_it_and_then_reports_success() {
        let (_, stop) = run(MCOMMIT, |_| {});
        assert_eq!(stop, ud());
        let (h, stop) = run(MCOMMIT, |h| {
            let efer = h.cpu.efer;
            h.cpu.wrmsr(MSR_EFER, efer | EFER_MCOMMIT).unwrap();
            h.cpu.set_eflags_arith(ARITHMETIC_FLAGS);
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(
            h.cpu.eflags_full() & ARITHMETIC_FLAGS,
            0,
            "CF clear: committed"
        );
    }

    #[test]
    fn hwcr_cpuid_user_disable_faults_cpuid_outside_cpl0() {
        let disable = |h: &mut Harness| {
            let hwcr = h.cpu.rdmsr(MSR_HWCR).unwrap();
            h.cpu
                .wrmsr(MSR_HWCR, hwcr | HWCR_CPUID_USER_DISABLE)
                .unwrap();
        };
        let (h, stop) = run(CPUID, |h| {
            disable(h);
            user_mode(h);
            h.cpu.gpr[RAX] = 0;
        });
        assert_eq!(stop, gp0());
        assert_eq!(
            (h.cpu.rip, h.cpu.gpr[RAX]),
            (CODE_BASE, 0),
            "nothing written"
        );
        // CPL 0 is unaffected, and so is CPL 3 without the bit (its hlt faults instead)
        let (_, stop) = run(CPUID, disable);
        assert_eq!(stop, Stop::Halted);
        let (h, stop) = run(CPUID, user_mode);
        assert_eq!(stop, gp0());
        assert_eq!(h.cpu.rip, CODE_BASE + 2);
    }
}

mod monitorx {
    use super::*;

    /// monitorx; mwaitx; mov eax, 5; hlt
    const MONITORX_MWAITX_MOV: &[u8] = &[
        0x0f, 0x01, 0xfa, 0x0f, 0x01, 0xfb, 0xb8, 0x05, 0x00, 0x00, 0x00, HLT,
    ];
    const MONITORX: &[u8] = &[0x0f, 0x01, 0xfa, HLT];
    const MWAITX: &[u8] = &[0x0f, 0x01, 0xfb, HLT];
    const MWAITX_TIMER: u64 = 1 << 1;

    fn armed_wait(setup: impl FnOnce(&mut Harness)) -> Harness {
        let mut h = Harness::new(MONITORX_MWAITX_MOV);
        h.cpu.gpr[RAX] = DATA;
        h.cpu.gpr[RCX] = 0;
        h.cpu.set_eflags_arith(0);
        h.cpu.eflags |= IF_BIT;
        setup(&mut h);
        h.run(3);
        h
    }

    #[test]
    fn mwaitx_waits_like_mwait_at_any_cpl() {
        let waiting = armed_wait(|_| {});
        assert!(waiting.cpu.halted);
        assert_eq!(waiting.cpu.rip, CODE_BASE + 6, "stopped after mwaitx");
        let user = armed_wait(user_mode);
        assert!(user.cpu.halted, "unprivileged, unlike mwait");
        let masked = armed_wait(|h| h.cpu.eflags &= !IF_BIT);
        assert!(!masked.cpu.halted);
        assert_eq!(masked.cpu.gpr[RAX], 5, "returns at once with IF=0");
    }

    #[test]
    fn the_timed_form_returns_at_once_and_disarms_the_monitor() {
        let mut h = Harness::new(MONITORX_MWAITX_MOV);
        h.cpu.gpr[RAX] = DATA;
        h.cpu.eflags |= IF_BIT;
        h.cpu.gpr[RBX] = 1_000_000;
        assert_eq!(h.run(1), Stop::Limit);
        h.cpu.gpr[RCX] = MWAITX_TIMER;
        assert_eq!(h.run(1), Stop::Limit);
        assert!(!h.cpu.halted);
        assert!(!h.cpu.monitor_armed);
        h.cpu.gpr[RAX] = 0;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[RAX], 5);
    }

    #[test]
    fn reserved_ecx_bits_are_gp() {
        let (_, stop) = run(MONITORX, |h| h.cpu.gpr[RCX] = 1);
        assert_eq!(stop, gp0());
        // EDX hints are ignored
        let (_, stop) = run(MONITORX, |h| {
            h.cpu.gpr[RAX] = DATA;
            h.cpu.gpr[RDX] = 1;
        });
        assert_eq!(stop, Stop::Halted);
        let (_, stop) = run(MWAITX, |h| h.cpu.gpr[RCX] = 4);
        assert_eq!(stop, gp0());
        let (_, stop) = run(MWAITX, |h| h.cpu.gpr[RCX] = 3);
        assert_eq!(stop, Stop::Halted, "both defined bits, nothing armed");
    }
}

mod cr8_legacy {
    use super::*;

    #[test]
    fn lock_mov_cr0_is_mov_cr8() {
        // lock mov cr0, rax; lock mov rbx, cr0; hlt
        let code = [0xf0, 0x0f, 0x22, 0xc0, 0xf0, 0x0f, 0x20, 0xc3, HLT];
        let (h, stop) = run(&code, |h| h.cpu.gpr[RAX] = 0x9);
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.gpr[RBX], 0x9);
        assert_eq!(h.cpu.dev.lock().lapic().tpr(), 0x90, "CR8 is TPR[7:4]");
        assert_ne!(h.cpu.cr0, 0x9, "CR0 untouched");
        // the CR8 rules apply: bits above 3 are reserved
        let (_, stop) = run(&code, |h| h.cpu.gpr[RAX] = 0x10);
        assert_eq!(stop, gp0());
    }

    #[test]
    fn it_works_in_32_bit_code_where_rex_cannot_reach_cr8() {
        use crate::cpu::decode::{CodeMode, decode_in_mode};
        let insn = decode_in_mode(&[0xf0, 0x0f, 0x22, 0xc0], CodeMode::Bits32).unwrap();
        assert_eq!((insn.op, insn.reg, insn.lock), (0x122, 8, false));
        let plain = decode_in_mode(&[0x0f, 0x22, 0xc0], CodeMode::Bits32).unwrap();
        assert_eq!(plain.reg, 0);
    }
}

mod ffxsr {
    use super::*;

    /// fxsave [rsi]; fxrstor [rdi]; hlt
    const FXSAVE_FXRSTOR: &[u8] = &[0x0f, 0xae, 0x06, 0x0f, 0xae, 0x0f, HLT];
    const RESTORE_AREA: u64 = DATA + 0x400;
    const MXCSR_OFFSET: u64 = 24;
    const XMM_OFFSET: u64 = 160;
    const MARKER: u8 = 0x5a;

    fn setup(h: &mut Harness, fast: bool) {
        use crate::testkit::RDI;
        h.cpu.cr4 |= CR4_OSFXSR;
        if fast {
            let efer = h.cpu.efer;
            h.cpu.wrmsr(MSR_EFER, efer | EFER_FFXSR).unwrap();
        }
        h.cpu.xmm[3] = 0x1111_2222_3333_4444_5555_6666_7777_8888;
        h.cpu.mxcsr = 0x1fa0;
        h.write(DATA, &[MARKER; 512]);
        let mut image = [0u8; 512];
        image[0..2].copy_from_slice(&0x037fu16.to_le_bytes());
        image[MXCSR_OFFSET as usize..][..4].copy_from_slice(&0x1f80u32.to_le_bytes());
        image[XMM_OFFSET as usize + 3 * 16..][..16].copy_from_slice(&[0xee; 16]);
        h.write(RESTORE_AREA, &image);
        h.cpu.gpr[RSI] = DATA;
        h.cpu.gpr[RDI] = RESTORE_AREA;
    }

    #[test]
    fn efer_ffxsr_leaves_the_xmm_registers_out_in_64_bit_mode_at_cpl0() {
        let (h, stop) = run(FXSAVE_FXRSTOR, |h| setup(h, true));
        assert_eq!(stop, Stop::Halted);
        let image = h.read(DATA, 512);
        assert_eq!(
            image[MXCSR_OFFSET as usize..][..4],
            0x1fa0u32.to_le_bytes(),
            "MXCSR is still saved"
        );
        assert!(
            image[XMM_OFFSET as usize..].iter().all(|&b| b == MARKER),
            "XMM area untouched"
        );
        assert_eq!(h.cpu.mxcsr, 0x1f80, "and restored");
        assert_eq!(h.cpu.xmm[3], 0x1111_2222_3333_4444_5555_6666_7777_8888);
    }

    #[test]
    fn without_the_bit_both_are_complete() {
        let (h, stop) = run(FXSAVE_FXRSTOR, |h| setup(h, false));
        assert_eq!(stop, Stop::Halted);
        let image = h.read(DATA, 512);
        assert_eq!(
            image[XMM_OFFSET as usize + 3 * 16..][..16],
            0x1111_2222_3333_4444_5555_6666_7777_8888u128.to_le_bytes()
        );
        assert_eq!(h.cpu.xmm[3], u128::from_le_bytes([0xee; 16]));
    }

    #[test]
    fn outside_cpl0_the_bit_changes_nothing() {
        // fxsave [rsi]; hlt: the hlt faults at CPL 3 after the save
        let (h, stop) = run(&[0x0f, 0xae, 0x06, HLT], |h| {
            setup(h, true);
            user_mode(h);
        });
        assert_eq!(stop, gp0());
        assert_eq!(
            h.read(DATA + XMM_OFFSET + 3 * 16, 16),
            0x1111_2222_3333_4444_5555_6666_7777_8888u128.to_le_bytes()
        );
    }

    #[test]
    fn a_fast_save_still_faults_before_writing() {
        let (h, stop) = run(FXSAVE_FXRSTOR, |h| {
            setup(h, true);
            h.cpu.gpr[RSI] = 0x4000_0000_0000;
        });
        assert!(
            matches!(stop, Stop::Fault { vector, .. } if vector == VECTOR_GENERAL_PROTECTION || vector == VECTOR_PAGE_FAULT),
            "{stop:?}"
        );
        assert_eq!(h.cpu.rip, CODE_BASE);
    }
}
