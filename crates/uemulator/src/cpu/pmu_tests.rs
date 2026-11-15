use super::*;
use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE};
use crate::cpu::test_support::*;
use crate::machine::ExitReason;
use crate::testkit::{
    CODE_BASE, HLT, Harness, R8, RAX, RCX, SCRATCH, STACK_TOP, assert_jit_matches_interpreter,
    run_both,
};

const MSR_PERF_CTL0: u32 = 0xc001_0200;
const MSR_PERF_CTR0: u32 = 0xc001_0201;
const COUNTS_EVERYWHERE: u64 = SELECT_OS | SELECT_USR | SELECT_EN;
const INSTRUCTIONS: u64 = EVENT_RETIRED_INSTRUCTIONS | COUNTS_EVERYWHERE;
const CYCLES: u64 = EVENT_CYCLES_NOT_HALTED | COUNTS_EVERYWHERE;
const BRANCHES: u64 = EVENT_RETIRED_BRANCHES | COUNTS_EVERYWHERE;
const TAKEN_BRANCHES: u64 = EVENT_RETIRED_TAKEN_BRANCHES | COUNTS_EVERYWHERE;
/// The local APIC's LVT performance counter entry and the NMI delivery mode in it.
const APIC_LVT_PERFORMANCE: u32 = 0x340;
const LVT_DELIVERY_NMI: u32 = 4 << 8;
const NMI_VECTOR: u8 = 2;

// xor ecx, ecx; rdpmc; mov r8, rax; mov ecx, 1; rdpmc; hlt
// r8 = counter 0 and rax = counter 1, each as it was before the rdpmc reading it.
const READ_COUNTERS_0_AND_1: [u8; 15] = [
    0x31, 0xc9, 0x0f, 0x33, 0x49, 0x89, 0xc0, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x33, HLT,
];

fn program(h: &mut Harness, counter: u32, select: u64, value: u64) {
    h.cpu
        .wrmsr(MSR_PERF_CTR0 + 2 * counter, value)
        .expect("counter");
    h.cpu
        .wrmsr(MSR_PERF_CTL0 + 2 * counter, select)
        .expect("event select");
}

fn counter(h: &mut Harness, index: usize) -> u64 {
    h.cpu.pmu_sync();
    h.cpu.pmu.counter[index]
}

fn with_code_after(prefix: &[u8], suffix: &[u8]) -> Vec<u8> {
    [prefix, suffix].concat()
}

fn general_protection<T: core::fmt::Debug>(result: ExecResult<T>) -> bool {
    fault_of(result) == (VECTOR_GENERAL_PROTECTION, Some(0))
}

/// Runs `code` through `Cpu::run` (event servicing, interrupt delivery) until it stops, which for
/// these snippets is a `cli; hlt` or a `hlt` with IF clear: the machine then shuts down.
fn run_machine_loop(h: &mut Harness) -> ExitReason {
    h.cpu.run()
}

#[test]
fn cpuid_advertises_perfctr_core_nb_llc_extapic_and_perfmon_v2() {
    let mut h = Harness::new(&[HLT]);
    h.cpu.gpr[RAX] = 0x8000_0001;
    h.cpu.cpuid();
    let ecx = h.cpu.gpr[RCX] as u32;
    for (bit, name) in [
        (3, "EXTAPIC"),
        (23, "PERFCTR_CORE"),
        (24, "PERFCTR_NB"),
        (28, "PERFCTR_LLC"),
    ] {
        assert_ne!(ecx & (1 << bit), 0, "{name}");
    }
    assert_ne!(ecx & (1 << 10), 0, "IBS (ibs_tests.rs)");
    h.cpu.gpr[RAX] = 0x8000_0022;
    h.cpu.gpr[RCX] = 0;
    h.cpu.cpuid();
    assert_eq!(
        h.cpu.gpr[RAX], 0b011,
        "PerfMonV2 and the LBR stack, no freeze"
    );
    let ebx = h.cpu.gpr[crate::testkit::RBX];
    assert_eq!(ebx & 0xf, 6, "core counters");
    assert_eq!((ebx >> 4) & 0x3f, 16, "LBR entries");
    assert_eq!((ebx >> 10) & 0x3f, 4, "data-fabric counters");
    assert_eq!((ebx >> 16) & 0x3f, 0, "no UMC counters");
}

#[test]
fn legacy_msrs_alias_core_counters_zero_to_three() {
    let mut h = Harness::new(&[HLT]);
    for index in 0..4u32 {
        let select = EVENT_RETIRED_OPS | SELECT_OS | (index as u64) << 8;
        h.cpu.wrmsr(0xc001_0000 + index, select).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_PERF_CTL0 + 2 * index).unwrap(), select);
        h.cpu
            .wrmsr(MSR_PERF_CTR0 + 2 * index, 0x1234 + index as u64)
            .unwrap();
        assert_eq!(
            h.cpu.rdmsr(0xc001_0004 + index).unwrap(),
            0x1234 + index as u64
        );
    }
    h.cpu.wrmsr(MSR_PERF_CTR0 + 2 * 5, u64::MAX).unwrap();
    assert_eq!(
        h.cpu.rdmsr(MSR_PERF_CTR0 + 2 * 5).unwrap(),
        (1 << 48) - 1,
        "48-bit counters"
    );
}

#[test]
fn reserved_bits_fault_and_status_is_read_only() {
    let mut h = Harness::new(&[HLT]);
    for reserved in [19, 21, 36, 39, 42, 63] {
        assert!(
            general_protection(h.cpu.wrmsr(MSR_PERF_CTL0, 1 << reserved)),
            "bit {reserved}"
        );
    }
    let defined = SELECT_HOST_ONLY | SELECT_GUEST_ONLY | EVENT_SELECT_HIGH_MASK | 0xff_ffff;
    h.cpu
        .wrmsr(MSR_PERF_CTL0, defined & !((1 << 19) | (1 << 21)))
        .expect("every defined bit is writable");
    assert_eq!(
        h.cpu.rdmsr(0xc000_0301).unwrap(),
        0x3f,
        "reset: all enabled"
    );
    assert!(general_protection(h.cpu.wrmsr(0xc000_0301, 1 << 6)));
    assert!(general_protection(h.cpu.wrmsr(0xc000_0300, 0)));
    h.cpu.wrmsr(0xc000_0303, 0b101).unwrap();
    assert_eq!(h.cpu.rdmsr(0xc000_0300).unwrap(), 0b101);
    // Linux's v2 handler acknowledges with all 64 bits.
    h.cpu.wrmsr(0xc000_0302, u64::MAX).unwrap();
    assert_eq!(h.cpu.rdmsr(0xc000_0300).unwrap(), 0);
    assert_eq!(h.cpu.rdmsr(0xc000_0302).unwrap(), 0, "write-only");
}

#[test]
fn retired_instructions_count_exactly_on_interpreter_and_jit() {
    // xor ecx, ecx; nop x5; rdpmc; hlt
    let code = [0x31, 0xc9, 0x90, 0x90, 0x90, 0x90, 0x90, 0x0f, 0x33, HLT];
    let setup = |h: &mut Harness| program(h, 0, INSTRUCTIONS, 0);
    let result = run_both(&code, setup);
    assert_eq!(
        result.interpreter.cpu.gpr[RAX], 6,
        "rdpmc reads what retired before it"
    );
    if let Some((mut jit, _)) = result.jit {
        assert_eq!(jit.cpu.gpr[RAX], 6, "jit");
        assert_eq!(counter(&mut jit, 0), 8, "rdpmc and hlt retired too");
    }
    let mut interpreter = result.interpreter;
    assert_eq!(counter(&mut interpreter, 0), 8);
    assert_jit_matches_interpreter(&code, setup);
}

#[test]
fn cycles_count_rep_iterations_and_instructions_do_not() {
    // mov edi, SCRATCH; mov ecx, 8; rep stosb; then read counters 0 and 1
    let prefix = [
        0xbf, 0x00, 0x00, 0x40, 0x00, 0xb9, 0x08, 0x00, 0x00, 0x00, 0xf3, 0xaa,
    ];
    assert_eq!(SCRATCH, 0x40_0000);
    let code = with_code_after(&prefix, &READ_COUNTERS_0_AND_1);
    let setup = |h: &mut Harness| {
        program(h, 0, INSTRUCTIONS, 0);
        program(h, 1, CYCLES, 0);
    };
    let result = run_both(&code, setup);
    let mut engines = vec![result.interpreter];
    engines.extend(result.jit.map(|(h, _)| h));
    for h in &engines {
        assert_eq!(h.cpu.gpr[R8], 4, "mov, mov, rep stosb, xor");
        // three more instructions (rdpmc, mov, mov), and the stores took a cycle each
        assert_eq!(h.cpu.gpr[RAX], 7 + 8);
    }
    assert_jit_matches_interpreter(&code, setup);
}

#[test]
fn global_ctl_and_the_enable_bit_both_gate_counting() {
    let code = [0x90, 0x90, 0x90, HLT];
    let mut h = Harness::new(&code);
    program(&mut h, 0, INSTRUCTIONS, 0);
    program(&mut h, 1, INSTRUCTIONS & !SELECT_EN, 0);
    program(&mut h, 2, INSTRUCTIONS, 0);
    h.cpu.wrmsr(0xc000_0301, 0b011).unwrap();
    h.run_to_halt();
    assert_eq!(counter(&mut h, 0), 4);
    assert_eq!(counter(&mut h, 1), 0, "PerfEvtSel.En clear");
    assert_eq!(counter(&mut h, 2), 0, "PerfCntrGlobalCtl bit clear");
}

#[test]
fn unmodelled_events_are_programmable_and_count_nothing() {
    let mut h = Harness::new(&[0x90, 0x90, HLT]);
    // L1 data cache misses, mispredicted branches
    program(&mut h, 0, 0x041 | COUNTS_EVERYWHERE | 0xff << 8, 5);
    program(&mut h, 1, 0x0c3 | COUNTS_EVERYWHERE, 0);
    h.run_to_halt();
    assert_eq!(counter(&mut h, 0), 5);
    assert_eq!(counter(&mut h, 1), 0);
    assert_eq!(h.cpu.pmu.deadline, u64::MAX, "nothing can overflow");
}

#[test]
fn count_mask_and_invert_follow_one_event_per_cycle() {
    let code = [0x90, 0x90, 0x90, 0x90, HLT];
    let mut h = Harness::new(&code);
    const MASK_1: u64 = 1 << SELECT_CNT_MASK_SHIFT;
    const MASK_2: u64 = 2 << SELECT_CNT_MASK_SHIFT;
    program(&mut h, 0, INSTRUCTIONS | MASK_1, 0);
    program(&mut h, 1, INSTRUCTIONS | MASK_2, 0);
    program(&mut h, 2, INSTRUCTIONS | MASK_1 | SELECT_INV, 0);
    program(&mut h, 3, INSTRUCTIONS | MASK_2 | SELECT_INV, 0);
    program(&mut h, 4, INSTRUCTIONS | SELECT_INV, 0);
    program(&mut h, 5, INSTRUCTIONS | SELECT_EDGE, 0);
    h.run_to_halt();
    assert_eq!(counter(&mut h, 0), 5, "cycles with an instruction");
    assert_eq!(counter(&mut h, 1), 0, "never two in a cycle");
    assert_eq!(counter(&mut h, 2), 0, "no cycle without one");
    assert_eq!(counter(&mut h, 3), 5, "every cycle has fewer than two");
    assert_eq!(counter(&mut h, 4), 5, "Inv without CntMask is ignored");
    assert_eq!(counter(&mut h, 5), 0, "Edge is not modelled");
}

/// User code at `CODE_BASE` runs `user`, then a syscall enters `kernel` at CPL 0.
fn user_to_kernel(user: &[u8], kernel: &[u8], setup: impl Fn(&mut Harness)) -> Vec<Harness> {
    const KERNEL_ENTRY: u64 = SCRATCH + 0x1000;
    const EFER_SCE: u64 = 1;
    const STAR_KERNEL_CS_SHIFT: u32 = 32;
    let prepare = |h: &mut Harness| {
        h.write(KERNEL_ENTRY, kernel);
        let pml4 = h.cpu.cr3 & PTE_ADDRESS_BITS;
        let pml4_entry = h.read_u64(pml4);
        h.write_u64(pml4, pml4_entry | PTE_U);
        let pdpt = pml4_entry & PTE_ADDRESS_BITS;
        for gib in 0..4 {
            let pdpt_entry = h.read_u64(pdpt + gib * 8);
            h.write_u64(pdpt + gib * 8, pdpt_entry | PTE_U);
            let pd = pdpt_entry & PTE_ADDRESS_BITS;
            for slot in 0..512 {
                let entry = h.read_u64(pd + slot * 8);
                h.write_u64(pd + slot * 8, entry | PTE_U);
            }
        }
        h.cpu.flush_tlb();
        enter_user_mode(h);
        h.cpu.efer |= EFER_SCE;
        h.cpu
            .wrmsr(0xc000_0081, (KERNEL_CS as u64) << STAR_KERNEL_CS_SHIFT)
            .unwrap();
        h.cpu.wrmsr(0xc000_0082, KERNEL_ENTRY).unwrap();
        setup(h);
    };
    let result = run_both(user, prepare);
    assert_eq!(result.interpreter_stop, crate::testkit::Stop::Halted);
    let mut engines = vec![result.interpreter];
    engines.extend(result.jit.map(|(h, _)| h));
    assert_jit_matches_interpreter(user, prepare);
    engines
}

#[test]
fn os_and_usr_split_at_the_syscall_which_counts_in_user_mode() {
    // nop x3; syscall
    let user = [0x90, 0x90, 0x90, 0x0f, 0x05];
    let engines = user_to_kernel(&user, &READ_COUNTERS_0_AND_1, |h| {
        program(h, 0, EVENT_RETIRED_INSTRUCTIONS | SELECT_OS | SELECT_EN, 0);
        program(h, 1, EVENT_RETIRED_INSTRUCTIONS | SELECT_USR | SELECT_EN, 0);
        program(h, 2, EVENT_RETIRED_INSTRUCTIONS | SELECT_EN, 0);
    });
    for mut h in engines {
        assert_eq!(h.cpu.cpl, 0);
        assert_eq!(h.cpu.gpr[R8], 1, "kernel: the xor before the rdpmc");
        assert_eq!(h.cpu.gpr[RAX], 4, "user: three nops and the syscall");
        assert_eq!(counter(&mut h, 2), 0, "neither OS nor USR counts nothing");
    }
}

#[test]
fn host_only_and_guest_only_count_nothing_while_svm_is_off() {
    let mut h = Harness::new(&[0x90, 0x90, HLT]);
    program(&mut h, 0, INSTRUCTIONS | SELECT_HOST_ONLY, 0);
    program(&mut h, 1, INSTRUCTIONS | SELECT_GUEST_ONLY, 0);
    program(
        &mut h,
        2,
        INSTRUCTIONS | SELECT_HOST_ONLY | SELECT_GUEST_ONLY,
        0,
    );
    h.run_to_halt();
    assert_eq!(counter(&mut h, 0), 0);
    assert_eq!(counter(&mut h, 1), 0);
    assert_eq!(counter(&mut h, 2), 3, "both bits: counts everywhere");

    let mut h = Harness::new(&[0x90, 0x90, HLT]);
    h.cpu
        .wrmsr(0xc000_0080, h.cpu.efer | svm::EFER_SVME)
        .unwrap();
    program(&mut h, 0, INSTRUCTIONS | SELECT_HOST_ONLY, 0);
    program(&mut h, 1, INSTRUCTIONS | SELECT_GUEST_ONLY, 0);
    h.run_to_halt();
    assert_eq!(counter(&mut h, 0), 3, "the host is running");
    assert_eq!(counter(&mut h, 1), 0);
}

#[test]
fn a_faulting_instruction_does_not_retire() {
    const HANDLER: u64 = SCRATCH + 0x2000;
    // ud2, handled by: xor ecx, ecx; rdpmc; hlt (IF is clear, so the machine stops)
    for jit in [false, true] {
        let mut h = Harness::new(&[0x0f, 0x0b]);
        if jit && !h.enable_jit() {
            continue;
        }
        install_idt(&mut h);
        h.cpu.seg[SEG_CS] = h.cpu.load_desc_seg(KERNEL_CS, true).unwrap();
        set_gate(&mut h, VECTOR_INVALID_OPCODE, Gate::interrupt(HANDLER));
        h.write(HANDLER, &[0x31, 0xc9, 0x0f, 0x33, HLT]);
        program(&mut h, 0, INSTRUCTIONS, 0);
        assert_eq!(run_machine_loop(&mut h), ExitReason::Shutdown);
        assert_eq!(h.cpu.gpr[RAX], 1, "jit {jit}: only the xor retired");
    }
}

#[test]
fn overflow_sets_the_status_bit_and_raises_an_nmi_through_the_lvt() {
    const HANDLER: u64 = SCRATCH + 0x2000;
    const BEFORE_OVERFLOW: u64 = 3;
    // nop x10; mov eax, 1; cli; hlt. The NMI handler: mov eax, 0x1234; cli; hlt.
    let mut code = vec![0x90; 10];
    code.extend([0xb8, 0x01, 0x00, 0x00, 0x00, 0xfa, HLT]);
    for jit in [false, true] {
        let mut h = Harness::new(&code);
        if jit && !h.enable_jit() {
            continue;
        }
        install_idt(&mut h);
        h.cpu.seg[SEG_CS] = h.cpu.load_desc_seg(KERNEL_CS, true).unwrap();
        set_gate(&mut h, NMI_VECTOR, Gate::interrupt(HANDLER));
        h.write(HANDLER, &[0xb8, 0x34, 0x12, 0x00, 0x00, 0xfa, HLT]);
        h.cpu
            .dev
            .lock()
            .lapic_mut()
            .write_register(APIC_LVT_PERFORMANCE, LVT_DELIVERY_NMI, 0)
            .unwrap();
        program(
            &mut h,
            0,
            INSTRUCTIONS | SELECT_INT,
            (1 << 48) - BEFORE_OVERFLOW,
        );
        program(&mut h, 1, INSTRUCTIONS, (1 << 48) - BEFORE_OVERFLOW);
        assert_eq!(run_machine_loop(&mut h), ExitReason::Shutdown);
        assert_eq!(h.cpu.gpr[RAX], 0x1234, "jit {jit}: the NMI handler ran");
        assert_eq!(
            h.cpu.rdmsr(0xc000_0300).unwrap(),
            0b11,
            "both overflowed; only counter 0 interrupts"
        );
        let frame = read_frame(&h, h.cpu.gpr[crate::testkit::RSP], false);
        if !jit {
            // The interpreter takes it at the next instruction boundary; the JIT at the end of
            // the block.
            assert_eq!(frame.rip, CODE_BASE + BEFORE_OVERFLOW);
        }
        assert!(h.cpu.pmu.counter[0] < 16, "wrapped to a small count");
        // the NMI is not re-raised by the counter staying overflowed
        assert!(!h.cpu.dev.lock().nmi_pending());
    }
}

#[test]
fn a_masked_lvt_entry_drops_the_interrupt_but_keeps_the_status() {
    let mut h = Harness::new(&[0x90, 0x90, 0x90, HLT]);
    program(&mut h, 0, INSTRUCTIONS | SELECT_INT, (1 << 48) - 1);
    h.run_to_halt();
    h.cpu.pmu_sync();
    assert!(!h.cpu.dev.lock().nmi_pending());
    assert_eq!(h.cpu.pmu.global_status, 1);
    assert_eq!(h.cpu.pmu.counter[0], 3);
}

#[test]
fn the_overflow_deadline_pulls_next_event_in() {
    let mut h = Harness::new(&[HLT]);
    h.cpu.next_event = 1_000_000;
    program(&mut h, 3, CYCLES, (1 << 48) - 100);
    assert_eq!(h.cpu.pmu.deadline, h.cpu.icount + 100);
    assert_eq!(h.cpu.next_event, h.cpu.icount + 100);
    assert!(!h.cpu.pmu.precise, "cycles need no interpreter");
}

// mov ecx, 3; l: dec ecx; jnz l; jmp a; nop; a: call f; <read counters 0 and 1>; f: ret
const BRANCHY: [u8; 33] = [
    0xb9, 0x03, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xfc, 0xeb, 0x01, 0x90, 0xe8, 0x0f, 0x00, 0x00,
    0x00, 0x31, 0xc9, 0x0f, 0x33, 0x49, 0x89, 0xc0, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x33, HLT,
    0xc3,
];
const BRANCHY_JNZ: u64 = 7;
const BRANCHY_LOOP: u64 = 5;
const BRANCHY_JMP: u64 = 9;
const BRANCHY_CALL: u64 = 12;
const BRANCHY_AFTER_CALL: u64 = 17;
const BRANCHY_RET: u64 = 32;

#[test]
fn branch_events_count_every_branch_and_the_taken_ones() {
    let setup = |h: &mut Harness| {
        program(h, 0, BRANCHES, 0);
        program(h, 1, TAKEN_BRANCHES, 0);
    };
    let result = run_both(&BRANCHY, setup);
    let mut engines = vec![result.interpreter];
    engines.extend(result.jit.map(|(h, _)| h));
    for h in &engines {
        assert!(h.cpu.pmu.precise);
        assert_eq!(h.cpu.gpr[R8], 6, "three jnz, jmp, call, ret");
        assert_eq!(h.cpu.gpr[RAX], 5, "the last jnz falls through");
    }
    assert_jit_matches_interpreter(&BRANCHY, setup);
}

#[test]
fn the_jit_takes_over_again_once_branch_counting_stops() {
    let mut h = Harness::new(&[HLT]);
    let jit = h.enable_jit();
    program(&mut h, 0, BRANCHES, 0);
    assert!(!h.cpu.uses_jit());
    h.cpu.wrmsr(MSR_PERF_CTL0, 0).unwrap();
    assert_eq!(h.cpu.uses_jit(), jit);
}

/// With a hardware breakpoint armed every instruction goes through `step_debug`, which must still
/// let the branch counters see them (`Cpu::exec_interpreted`).
#[test]
fn branch_events_still_count_while_a_breakpoint_is_armed() {
    const DR7_L0: u64 = 1;
    let setup = |h: &mut Harness| {
        program(h, 0, BRANCHES, 0);
        program(h, 1, TAKEN_BRANCHES, 0);
        // an instruction breakpoint nothing reaches
        h.cpu.dr[0] = SCRATCH;
        h.cpu.dr[7] |= DR7_L0;
    };
    let result = run_both(&BRANCHY, setup);
    let mut engines = vec![result.interpreter];
    engines.extend(result.jit.map(|(h, _)| h));
    for h in &engines {
        assert!(!h.cpu.uses_jit());
        assert_eq!(h.cpu.gpr[R8], 6, "three jnz, jmp, call, ret");
        assert_eq!(h.cpu.gpr[RAX], 5, "the last jnz falls through");
    }
}

fn lbr_entry(h: &mut Harness, index: u32) -> (u64, u64) {
    (
        h.cpu.rdmsr(0xc001_0300 + 2 * index).unwrap(),
        h.cpu.rdmsr(0xc001_0301 + 2 * index).unwrap(),
    )
}

fn recorded(from: u64, to: u64) -> (u64, u64) {
    (CODE_BASE + from, (CODE_BASE + to) | LBR_TO_VALID)
}

#[test]
fn the_lbr_stack_records_taken_branches_newest_first() {
    let setup = |h: &mut Harness| h.cpu.wrmsr(0xc000_010f, DBG_EXTN_CFG_LBRV2EN).unwrap();
    let result = run_both(&BRANCHY, setup);
    let mut engines = vec![result.interpreter];
    engines.extend(result.jit.map(|(h, _)| h));
    for h in &mut engines {
        assert_eq!(lbr_entry(h, 0), recorded(BRANCHY_RET, BRANCHY_AFTER_CALL));
        assert_eq!(lbr_entry(h, 1), recorded(BRANCHY_CALL, BRANCHY_RET));
        assert_eq!(lbr_entry(h, 2), recorded(BRANCHY_JMP, BRANCHY_CALL));
        assert_eq!(lbr_entry(h, 3), recorded(BRANCHY_JNZ, BRANCHY_LOOP));
        assert_eq!(lbr_entry(h, 4), recorded(BRANCHY_JNZ, BRANCHY_LOOP));
        assert_eq!(lbr_entry(h, 5), (0, 0), "the fall-through is not recorded");
    }
}

#[test]
fn lbr_select_suppresses_branch_types_and_privilege_levels() {
    let mut h = Harness::new(&BRANCHY);
    h.cpu
        .wrmsr(0xc000_010e, LBR_SELECT_CONDITIONAL | LBR_SELECT_RETURN)
        .unwrap();
    h.cpu.wrmsr(0xc000_010f, DBG_EXTN_CFG_LBRV2EN).unwrap();
    h.run_to_halt();
    assert_eq!(lbr_entry(&mut h, 0), recorded(BRANCHY_CALL, BRANCHY_RET));
    assert_eq!(lbr_entry(&mut h, 1), recorded(BRANCHY_JMP, BRANCHY_CALL));
    assert_eq!(lbr_entry(&mut h, 2), (0, 0));

    let mut h = Harness::new(&BRANCHY);
    h.cpu.wrmsr(0xc000_010e, LBR_SELECT_KERNEL).unwrap();
    h.cpu.wrmsr(0xc000_010f, DBG_EXTN_CFG_LBRV2EN).unwrap();
    h.run_to_halt();
    assert_eq!(lbr_entry(&mut h, 0), (0, 0), "all of it ran at CPL 0");
    assert!(general_protection(h.cpu.wrmsr(0xc000_010e, 1 << 9)));
}

#[test]
fn delivered_exceptions_count_as_taken_branches_and_far_lbr_entries() {
    const HANDLER: u64 = SCRATCH + 0x2000;
    let mut h = Harness::new(&[0x0f, 0x0b]);
    install_idt(&mut h);
    h.cpu.seg[SEG_CS] = h.cpu.load_desc_seg(KERNEL_CS, true).unwrap();
    set_gate(&mut h, VECTOR_INVALID_OPCODE, Gate::interrupt(HANDLER));
    h.write(HANDLER, &[HLT]);
    h.cpu.gpr[crate::testkit::RSP] = STACK_TOP;
    program(&mut h, 0, BRANCHES, 0);
    h.cpu.wrmsr(0xc000_010e, LBR_SELECT_FAR).unwrap();
    h.cpu.wrmsr(0xc000_010f, DBG_EXTN_CFG_LBRV2EN).unwrap();
    assert_eq!(run_machine_loop(&mut h), ExitReason::Shutdown);
    assert_eq!(counter(&mut h, 0), 1, "the #UD delivery");
    assert_eq!(lbr_entry(&mut h, 0), (0, 0), "far branches suppressed");

    let mut h = Harness::new(&[0x0f, 0x0b]);
    install_idt(&mut h);
    h.cpu.seg[SEG_CS] = h.cpu.load_desc_seg(KERNEL_CS, true).unwrap();
    set_gate(&mut h, VECTOR_INVALID_OPCODE, Gate::interrupt(HANDLER));
    h.write(HANDLER, &[HLT]);
    h.cpu.wrmsr(0xc000_010f, DBG_EXTN_CFG_LBRV2EN).unwrap();
    assert_eq!(run_machine_loop(&mut h), ExitReason::Shutdown);
    assert_eq!(lbr_entry(&mut h, 0), (CODE_BASE, HANDLER | LBR_TO_VALID));
}

#[test]
fn uncore_counters_are_programmable_and_readable_by_rdpmc() {
    let mut h = Harness::new(&[HLT]);
    for index in 0..4u32 {
        let select = 0xc001_0240 + 2 * index;
        h.cpu.wrmsr(select, 0x1_0000_00ff | index as u64).unwrap();
        assert_eq!(h.cpu.rdmsr(select).unwrap(), 0x1_0000_00ff | index as u64);
        h.cpu.wrmsr(select + 1, 0x100 + index as u64).unwrap();
        assert_eq!(h.cpu.pmu_rdpmc(6 + index), Some(0x100 + index as u64));
    }
    for index in 0..6u32 {
        let select = 0xc001_0230 + 2 * index;
        h.cpu.wrmsr(select, 0xff0f_0000_0040_ff04).unwrap();
        assert_eq!(h.cpu.rdmsr(select).unwrap(), 0xff0f_0000_0040_ff04);
        h.cpu.wrmsr(select + 1, u64::MAX).unwrap();
        assert_eq!(h.cpu.pmu_rdpmc(10 + index), Some((1 << 48) - 1));
    }
    assert_eq!(h.cpu.pmu_rdpmc(16), None);
    assert!(general_protection(h.cpu.rdmsr(0xc001_0248)));
    assert!(general_protection(h.cpu.rdmsr(0xc001_023c)));
}

#[test]
fn the_reading_instruction_is_not_in_what_rdmsr_returns() {
    // mov ecx, PERF_CTR0; rdmsr; hlt
    let code = [0xb9, 0x01, 0x02, 0x01, 0xc0, 0x0f, 0x32, HLT];
    let setup = |h: &mut Harness| program(h, 0, INSTRUCTIONS, 0);
    let result = run_both(&code, setup);
    assert_eq!(result.interpreter.cpu.gpr[RAX], 1);
    if let Some((jit, _)) = result.jit {
        assert_eq!(jit.cpu.gpr[RAX], 1);
    }
    assert_jit_matches_interpreter(&code, setup);
}

const MSR_IRPERF: u32 = 0xc000_00e9;
const MSR_HWCR: u32 = 0xc001_0015;
const HWCR_IRPERF_ENABLE: u64 = 1 << 30;

#[test]
fn irperf_counts_retired_instructions_only_while_hwcr_enables_it() {
    let mut h = Harness::new(&[HLT]);
    h.cpu.gpr[RAX] = 0x8000_0008;
    h.cpu.cpuid();
    assert_ne!(h.cpu.gpr[crate::cpu::sys::RBX] & 1 << 1, 0, "IRPERF");
    h.cpu.icount += 100;
    assert_eq!(h.cpu.rdmsr(MSR_IRPERF).unwrap(), 0, "off at reset");
    let hwcr = h.cpu.rdmsr(MSR_HWCR).unwrap();
    h.cpu.wrmsr(MSR_HWCR, hwcr | HWCR_IRPERF_ENABLE).unwrap();
    h.cpu.icount += 40;
    // a faulting instruction is charged to icount but does not retire
    h.cpu.pmu.unretired += 3;
    // the reading rdmsr is in icount already and does not count itself
    h.cpu.icount += 1;
    assert_eq!(h.cpu.rdmsr(MSR_IRPERF).unwrap(), 37);
    h.cpu.wrmsr(MSR_HWCR, hwcr).unwrap();
    h.cpu.icount += 50;
    assert_eq!(
        h.cpu.rdmsr(MSR_IRPERF).unwrap(),
        38,
        "the disabling wrmsr counted"
    );
    assert!(general_protection(h.cpu.wrmsr(MSR_IRPERF, 0)), "read-only");
}

/// Time-shared vCPUs hand one `icount` around: what the other vCPU retires while this one is
/// parked is not this one's, for the core counters, IRPERF and the overflow deadline alike.
#[test]
fn a_parked_vcpu_does_not_count_what_the_others_retire() {
    let mut this = Harness::new(&[HLT]);
    let mut other = Harness::new(&[HLT]);
    program(&mut this, 0, INSTRUCTIONS, COUNTER_MASK - 99);
    program(&mut this, 1, CYCLES, 0);
    let hwcr = this.cpu.rdmsr(MSR_HWCR).unwrap();
    this.cpu.wrmsr(MSR_HWCR, hwcr | HWCR_IRPERF_ENABLE).unwrap();
    this.cpu.icount += 30;
    let deadline = this.cpu.pmu.deadline;

    // `Smp::switch_to` there and back, the other vCPU retiring 1000 in between
    this.cpu.pmu_park();
    this.cpu.swap_shared(&mut other.cpu);
    other.cpu.pmu_unpark();
    other.cpu.icount += 1000;
    other.cpu.pmu_park();
    other.cpu.swap_shared(&mut this.cpu);
    this.cpu.pmu_unpark();

    assert_eq!(
        this.cpu.pmu.deadline,
        deadline + 1000,
        "the overflow moved with the gap"
    );
    assert!(this.cpu.next_event <= this.cpu.pmu.deadline);
    this.cpu.icount += 20;
    assert_eq!(counter(&mut this, 0), COUNTER_MASK - 99 + 50);
    assert_eq!(counter(&mut this, 1), 50 * CYCLES_PER_ICOUNT);
    assert_eq!(this.cpu.pmu.irperf, 50);
}
