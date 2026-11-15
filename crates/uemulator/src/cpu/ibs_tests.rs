//! Instruction-based sampling (`ibs.rs`), on the interpreter and with the JIT, which must hand
//! the instructions around a tag to the interpreter.

use super::*;
use crate::cpu::test_support::*;
use crate::machine::ExitReason;
use crate::testkit::{CODE_BASE, HLT, Harness, RAX, RBX, RCX, SCRATCH, Stop};

/// The local APIC's first extended LVT entry, and the NMI delivery mode Linux programs in it.
const APIC_EXTENDED_LVT0: u32 = 0x500;
const LVT_DELIVERY_NMI: u32 = 4 << 8;
const NMI_VECTOR: u8 = 2;
const NOP: u8 = 0x90;
/// A sampling period of 16 events (MaxCnt 1, in units of 16): the 17th instruction is tagged.
const PERIOD_16: u64 = 1;
const TAGGED_AFTER_16: u64 = CODE_BASE + 16;

// mov ecx, 3000; 1: add rax, 1; dec ecx; jnz 1b; hlt
const LOOP: [u8; 14] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x01, 0xff, 0xc9, 0x75, 0xf8, HLT,
];
const LOOP_ADD: u64 = CODE_BASE + 5;
const LOOP_JNZ: u64 = CODE_BASE + 11;

fn nops_then(count: usize, tail: &[u8]) -> Vec<u8> {
    let mut code = vec![NOP; count];
    code.extend_from_slice(tail);
    code
}

/// Runs `check` on a fresh harness for the interpreter, then for the JIT where the host has one.
fn on_both_engines(code: &[u8], setup: impl Fn(&mut Harness), check: impl Fn(&mut Harness, &str)) {
    let mut interpreted = Harness::new(code);
    setup(&mut interpreted);
    check(&mut interpreted, "interpreter");
    let mut jitted = Harness::new(code);
    if jitted.enable_jit() {
        setup(&mut jitted);
        check(&mut jitted, "jit");
    }
}

/// Op sampling every `period_field * 16` ops (MaxCnt bits 19:4; bits 26:20 go in place).
fn sample_ops(h: &mut Harness, maximum: u64) {
    let control = ((maximum >> 4) & OP_MAX_COUNT_MASK)
        | (maximum & OP_MAX_COUNT_EXTENSION_MASK)
        | OP_ENABLE
        | OP_COUNT_CONTROL;
    h.cpu.wrmsr(MSR_IBS_OP_CONTROL, control).unwrap();
}

fn rdmsr(h: &mut Harness, msr: u32) -> u64 {
    h.cpu.rdmsr(msr).unwrap()
}

fn nmi_through_the_extended_lvt(h: &mut Harness) {
    h.cpu
        .dev
        .lock()
        .lapic_mut()
        .write_register(APIC_EXTENDED_LVT0, LVT_DELIVERY_NMI, 0)
        .unwrap();
}

#[test]
fn cpuid_advertises_ibs_and_its_capabilities() {
    let mut h = Harness::new(&[HLT]);
    h.cpu.gpr[RAX] = 0x8000_0001;
    h.cpu.cpuid();
    assert_ne!(h.cpu.gpr[RCX] & 1 << 10, 0, "IBS");
    h.cpu.gpr[RAX] = 0x8000_001b;
    h.cpu.cpuid();
    // IBSFFV, FetchSam, OpSam, RdWrOpCnt, OpCnt, BrnTrgt, OpCntExt, RipInvalidChk, OpBrnFuse
    // and the Zen 4 bit; not the fetch control extension, IbsOpData4 or load-latency filtering
    assert_eq!(h.cpu.gpr[RAX], 0b1001_1111_1111);
}

#[test]
fn ibs_control_names_extended_lvt_entry_zero_and_is_read_only() {
    let mut h = Harness::new(&[HLT]);
    assert_eq!(
        rdmsr(&mut h, MSR_IBS_CONTROL),
        0x100,
        "LvtOffsetVal, offset 0"
    );
    assert!(h.cpu.wrmsr(MSR_IBS_CONTROL, 0x101).is_err());
}

#[test]
fn op_sampling_tags_the_instruction_after_the_period() {
    on_both_engines(
        &nops_then(40, &[HLT]),
        |h| sample_ops(h, PERIOD_16 << 4),
        |h, engine| {
            assert_eq!(h.run(100), Stop::Halted, "{engine}");
            let control = rdmsr(h, MSR_IBS_OP_CONTROL);
            assert_ne!(control & OP_VALID, 0, "{engine}");
            assert_eq!(rdmsr(h, MSR_IBS_OP_RIP), TAGGED_AFTER_16, "{engine}");
            assert_eq!(rdmsr(h, MSR_IBS_OP_DATA), 0, "{engine}: no branch");
            assert_eq!(rdmsr(h, MSR_IBS_OP_DATA3), 0, "{engine}: no memory access");
            assert!(!h.cpu.pmu.precise, "{engine}: back to the JIT once sampled");
        },
    );
}

#[test]
fn a_sample_raises_the_extended_lvt_interrupt() {
    on_both_engines(
        &nops_then(40, &[HLT]),
        |h| {
            nmi_through_the_extended_lvt(h);
            sample_ops(h, PERIOD_16 << 4);
        },
        |h, engine| {
            assert!(!h.cpu.dev.lock().nmi_pending(), "{engine}");
            h.run(100);
            assert!(h.cpu.dev.lock().nmi_pending(), "{engine}");
        },
    );
    // masked at reset: nothing
    let mut h = Harness::new(&nops_then(40, &[HLT]));
    sample_ops(&mut h, PERIOD_16 << 4);
    h.run(100);
    assert_ne!(rdmsr(&mut h, MSR_IBS_OP_CONTROL) & OP_VALID, 0);
    assert!(!h.cpu.dev.lock().nmi_pending());
}

/// The JIT runs the loop until shortly before the tag, which must still land on the right
/// instruction; a tagged branch reports where it went. Through `Cpu::run`, whose `before_step`
/// is what notices the deadline.
#[test]
fn a_far_tag_lands_on_the_right_instruction_under_the_jit() {
    // (period, tagged instruction, branch data)
    for (period, rip, data, target) in [
        (0x1000, LOOP_ADD, 0, 0),
        (
            0x12c0,
            LOOP_JNZ,
            OP_DATA_BRANCH_RETIRED | OP_DATA_BRANCH_TAKEN,
            LOOP_ADD,
        ),
        (
            0x30,
            LOOP_JNZ,
            OP_DATA_BRANCH_RETIRED | OP_DATA_BRANCH_TAKEN,
            LOOP_ADD,
        ),
    ] {
        on_both_engines(
            &LOOP,
            |h| sample_ops(h, period),
            |h, engine| {
                assert_eq!(h.cpu.run(), ExitReason::Shutdown, "{engine}");
                assert_eq!(h.cpu.gpr[RAX], 3000, "{engine}");
                assert_ne!(rdmsr(h, MSR_IBS_OP_CONTROL) & OP_VALID, 0, "{engine}");
                assert_eq!(rdmsr(h, MSR_IBS_OP_RIP), rip, "{engine} {period:#x}");
                assert_eq!(rdmsr(h, MSR_IBS_OP_DATA), data, "{engine} {period:#x}");
                assert_eq!(
                    rdmsr(h, MSR_IBS_BRANCH_TARGET),
                    target,
                    "{engine} {period:#x}"
                );
            },
        );
    }
}

#[test]
fn a_tagged_store_and_load_report_their_address_width_and_direction() {
    // mov dword [rbx], eax / mov eax, dword [rbx], at the 17th instruction
    for (instruction, direction) in [
        ([0x89, 0x03], OP_DATA3_STORE),
        ([0x8b, 0x03], OP_DATA3_LOAD),
    ] {
        on_both_engines(
            &nops_then(16, &[instruction[0], instruction[1], HLT]),
            |h| {
                h.cpu.gpr[RBX] = SCRATCH + 0x24;
                sample_ops(h, PERIOD_16 << 4);
            },
            |h, engine| {
                assert_eq!(h.run(100), Stop::Halted, "{engine}");
                assert_eq!(rdmsr(h, MSR_IBS_OP_RIP), TAGGED_AFTER_16, "{engine}");
                assert_eq!(
                    rdmsr(h, MSR_IBS_OP_DATA3),
                    direction
                        | OP_DATA3_LINEAR_VALID
                        | OP_DATA3_PHYSICAL_VALID
                        | 3 << OP_DATA3_WIDTH_SHIFT,
                    "{engine}"
                );
                assert_eq!(rdmsr(h, MSR_IBS_DC_LINEAR), SCRATCH + 0x24, "{engine}");
                assert_eq!(rdmsr(h, MSR_IBS_DC_PHYSICAL), SCRATCH + 0x24, "{engine}");
            },
        );
    }
}

#[test]
fn fetch_sampling_reports_the_fetch_address() {
    on_both_engines(
        &nops_then(40, &[HLT]),
        |h| {
            h.cpu
                .wrmsr(MSR_IBS_FETCH_CONTROL, PERIOD_16 | FETCH_ENABLE)
                .unwrap()
        },
        |h, engine| {
            assert_eq!(h.run(100), Stop::Halted, "{engine}");
            let control = rdmsr(h, MSR_IBS_FETCH_CONTROL);
            assert_eq!(
                control & (FETCH_VALID | FETCH_COMPLETE | FETCH_PHYSICAL_VALID | FETCH_ENABLE),
                FETCH_VALID | FETCH_COMPLETE | FETCH_PHYSICAL_VALID | FETCH_ENABLE,
                "{engine}: {control:#x}"
            );
            assert_eq!(rdmsr(h, MSR_IBS_FETCH_LINEAR), TAGGED_AFTER_16, "{engine}");
            assert_eq!(
                rdmsr(h, MSR_IBS_FETCH_PHYSICAL),
                TAGGED_AFTER_16,
                "{engine}"
            );
        },
    );
}

#[test]
fn the_current_op_count_reads_back_and_a_rewrite_samples_again() {
    // nop x5; mov ecx, IbsOpCtl; rdmsr; hlt
    let code = nops_then(5, &[0xb9, 0x33, 0x10, 0x01, 0xc0, 0x0f, 0x32, HLT]);
    let mut h = Harness::new(&code);
    sample_ops(&mut h, 0x100);
    h.run_to_halt();
    assert_eq!(
        h.cpu.gpr[crate::testkit::RDX] & 0x7ff_ffff,
        6,
        "five nops and the mov, not the rdmsr"
    );
    // Linux's handler rewrites the control register with Val clear to take the next sample.
    let mut h = Harness::new(&nops_then(60, &[HLT]));
    sample_ops(&mut h, PERIOD_16 << 4);
    h.run(17);
    assert_eq!(rdmsr(&mut h, MSR_IBS_OP_RIP), TAGGED_AFTER_16);
    sample_ops(&mut h, PERIOD_16 << 4);
    h.run_to_halt();
    assert_eq!(rdmsr(&mut h, MSR_IBS_OP_RIP), TAGGED_AFTER_16 + 17);
}

#[test]
fn l3_miss_only_never_samples() {
    let mut h = Harness::new(&nops_then(40, &[HLT]));
    h.cpu
        .wrmsr(
            MSR_IBS_OP_CONTROL,
            PERIOD_16 | OP_ENABLE | OP_COUNT_CONTROL | OP_L3_MISS_ONLY,
        )
        .unwrap();
    h.run_to_halt();
    assert_eq!(rdmsr(&mut h, MSR_IBS_OP_CONTROL) & OP_VALID, 0);
    assert!(!h.cpu.pmu.precise, "nothing to wait for");
}

/// Through `Cpu::run`: the NMI reaches its handler, at the instruction after the tagged one on
/// the interpreter.
#[test]
fn the_nmi_handler_runs_after_a_sample() {
    const HANDLER: u64 = SCRATCH + 0x2000;
    // nop x40; cli; hlt. The NMI handler: mov eax, 0x1234; cli; hlt.
    let code = nops_then(40, &[0xfa, HLT]);
    for jit in [false, true] {
        let mut h = Harness::new(&code);
        if jit && !h.enable_jit() {
            continue;
        }
        install_idt(&mut h);
        h.cpu.seg[crate::cpu::SEG_CS] = h.cpu.load_desc_seg(KERNEL_CS, true).unwrap();
        set_gate(&mut h, NMI_VECTOR, Gate::interrupt(HANDLER));
        h.write(HANDLER, &[0xb8, 0x34, 0x12, 0x00, 0x00, 0xfa, HLT]);
        nmi_through_the_extended_lvt(&mut h);
        sample_ops(&mut h, PERIOD_16 << 4);
        assert_eq!(h.cpu.run(), ExitReason::Shutdown);
        assert_eq!(h.cpu.gpr[RAX], 0x1234, "jit {jit}: the NMI handler ran");
        let frame = read_frame(&h, h.cpu.gpr[crate::testkit::RSP], false);
        assert_eq!(frame.rip, TAGGED_AFTER_16 + 1, "jit {jit}");
    }
}
