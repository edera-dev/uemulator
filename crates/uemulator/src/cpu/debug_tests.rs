//! Hardware breakpoints, single-step, general detect and bus-lock detect (`debug.rs`), on the
//! interpreter and with the JIT enabled (which must hand every armed step to the interpreter).

use super::debug::*;
use super::flags::{RF, TF};
use crate::testkit::*;

const DEBUG_TRAP: Stop = Stop::Fault {
    vector: VECTOR_DEBUG,
    error_code: None,
    software: false,
};
const WATCHED: u64 = SCRATCH + 0x100;
/// DR7 bit 10, which always reads as one.
const DR7_RESERVED_ONE: u64 = 0x400;
const DR6_RESET: u64 = 0xffff_0ff0;
const RW_EXECUTE: u64 = 0;
const RW_WRITE: u64 = 1;
const RW_IO: u64 = 2;
const RW_READ_WRITE: u64 = 3;
const LEN_1: u64 = 0;
const LEN_2: u64 = 1;
const LEN_8: u64 = 2;
const LEN_4: u64 = 3;
const CR4_DE: u64 = 1 << 3;
const RBX: usize = 3;
const RCX: usize = 1;

// mov dword [rbx], eax; nop; hlt
const STORE_DWORD: [u8; 4] = [0x89, 0x03, 0x90, 0xf4];
// mov eax, dword [rbx]; nop; hlt
const LOAD_DWORD: [u8; 4] = [0x8b, 0x03, 0x90, 0xf4];
// mov byte [rbx], al; hlt
const STORE_BYTE: [u8; 3] = [0x88, 0x03, 0xf4];
// nop; nop; nop; hlt
const NOPS: [u8; 4] = [0x90, 0x90, 0x90, 0xf4];
// out 0x80, al; nop; hlt
const OUT_PORT_80: [u8; 4] = [0xe6, 0x80, 0x90, 0xf4];
// mov rax, dr0; hlt
const READ_DR0: [u8; 4] = [0x0f, 0x21, 0xc0, 0xf4];
// pushfq; or qword [rsp], 0x100; popfq; nop; nop; hlt
const SET_TF_WITH_POPF: [u8; 13] = [
    0x9c, 0x48, 0x81, 0x0c, 0x24, 0x00, 0x01, 0x00, 0x00, 0x9d, 0x90, 0x90, 0xf4,
];
// mov ax, ss; mov ss, ax; nop; hlt
const MOV_SS_THEN_NOP: [u8; 7] = [0x8c, 0xd0, 0x66, 0x8e, 0xd0, 0x90, 0xf4];
// nop; jmp 1f; nop; 1: nop; hlt
const NOP_JMP_NOP: [u8; 6] = [0x90, 0xeb, 0x01, 0x90, 0x90, 0xf4];
// lock add dword [rbx], eax; hlt
const LOCKED_ADD: [u8; 4] = [0xf0, 0x01, 0x03, 0xf4];
// add dword [rbx], eax; hlt
const PLAIN_ADD: [u8; 3] = [0x01, 0x03, 0xf4];
// rep stosb; hlt
const REP_STOSB: [u8; 3] = [0xf3, 0xaa, 0xf4];
// mov ecx, 100; 1: mov dword [rbx], ecx; add rbx, 4; dec ecx; jnz 1b; hlt
const STORE_LOOP: [u8; 16] = [
    0xb9, 0x64, 0x00, 0x00, 0x00, 0x89, 0x0b, 0x48, 0x83, 0xc3, 0x04, 0xff, 0xc9, 0x75, 0xf6, 0xf4,
];
// mov dr7, rax; mov dword [rbx], eax; nop; hlt
const ARM_THEN_STORE: [u8; 7] = [0x0f, 0x23, 0xf8, 0x89, 0x03, 0x90, 0xf4];

/// DR7 with breakpoint `index` locally enabled with R/W `rw` and LEN code `len`.
fn dr7_for(index: usize, rw: u64, len: u64) -> u64 {
    DR7_RESERVED_ONE | 1 << (2 * index) | (rw | len << 2) << (16 + 4 * index)
}

fn arm(h: &mut Harness, index: usize, address: u64, rw: u64, len: u64) {
    h.cpu.dr[index] = address;
    h.cpu.dr[7] |= dr7_for(index, rw, len);
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

#[test]
fn an_instruction_breakpoint_faults_before_the_instruction_runs() {
    on_both_engines(
        &NOPS,
        |h| arm(h, 2, CODE_BASE + 1, RW_EXECUTE, LEN_1),
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 1, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET | 1 << 2, "{engine}");
            assert_eq!(h.cpu.eflags & RF, 0, "the handler sets RF, not the CPU");
        },
    );
}

#[test]
fn rf_steps_over_an_instruction_breakpoint_once_and_is_cleared() {
    on_both_engines(
        &NOPS,
        |h| {
            arm(h, 0, CODE_BASE, RW_EXECUTE, LEN_1);
            h.cpu.eflags |= RF;
        },
        |h, engine| {
            assert_eq!(h.run(100), Stop::Halted, "{engine}");
            assert_eq!(h.cpu.eflags & RF, 0, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET, "{engine}");
        },
    );
}

#[test]
fn a_disabled_breakpoint_never_fires() {
    on_both_engines(
        &STORE_DWORD,
        |h| {
            arm(h, 0, WATCHED, RW_WRITE, LEN_4);
            h.cpu.dr[7] &= !3;
            h.cpu.gpr[RBX] = WATCHED;
        },
        |h, engine| assert_eq!(h.run(100), Stop::Halted, "{engine}"),
    );
}

#[test]
fn a_write_watchpoint_traps_after_the_store_completes() {
    on_both_engines(
        &STORE_DWORD,
        |h| {
            arm(h, 1, WATCHED, RW_WRITE, LEN_4);
            h.cpu.gpr[RBX] = WATCHED;
            h.cpu.gpr[0] = 0x1234_5678;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 2, "{engine}: trap, after the store");
            assert_eq!(h.read(WATCHED, 4), 0x1234_5678u32.to_le_bytes(), "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET | 1 << 1, "{engine}");
        },
    );
}

#[test]
fn a_write_watchpoint_ignores_loads_and_a_read_write_one_catches_them() {
    on_both_engines(
        &LOAD_DWORD,
        |h| {
            arm(h, 0, WATCHED, RW_WRITE, LEN_4);
            h.cpu.gpr[RBX] = WATCHED;
        },
        |h, engine| assert_eq!(h.run(100), Stop::Halted, "{engine}"),
    );
    on_both_engines(
        &LOAD_DWORD,
        |h| {
            arm(h, 3, WATCHED, RW_READ_WRITE, LEN_4);
            h.cpu.gpr[RBX] = WATCHED;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET | 1 << 3, "{engine}");
        },
    );
}

#[test]
fn the_length_aligns_the_watched_range_down() {
    // (breakpoint address, LEN code, byte stored to, whether it hits)
    let cases = [
        (WATCHED + 3, LEN_4, WATCHED, true),
        (WATCHED + 3, LEN_4, WATCHED + 3, true),
        (WATCHED + 3, LEN_4, WATCHED + 4, false),
        (WATCHED + 1, LEN_2, WATCHED, true),
        (WATCHED + 1, LEN_2, WATCHED + 2, false),
        (WATCHED + 8, LEN_8, WATCHED + 15, true),
        (WATCHED + 8, LEN_8, WATCHED + 7, false),
        (WATCHED + 8, LEN_8, WATCHED + 16, false),
        (WATCHED, LEN_1, WATCHED + 1, false),
    ];
    for (address, len, target, hits) in cases {
        on_both_engines(
            &STORE_BYTE,
            |h| {
                arm(h, 0, address, RW_WRITE, len);
                h.cpu.gpr[RBX] = target;
            },
            |h, engine| {
                let expected = if hits { DEBUG_TRAP } else { Stop::Halted };
                assert_eq!(
                    h.run(100),
                    expected,
                    "{engine}: dr0 {address:#x} len {len} store {target:#x}"
                );
            },
        );
    }
}

#[test]
fn an_access_that_overlaps_the_watched_range_hits_it() {
    // A dword store at WATCHED - 2 covers WATCHED and WATCHED + 1.
    on_both_engines(
        &STORE_DWORD,
        |h| {
            arm(h, 0, WATCHED + 1, RW_WRITE, LEN_1);
            h.cpu.gpr[RBX] = WATCHED - 2;
        },
        |h, engine| assert_eq!(h.run(100), DEBUG_TRAP, "{engine}"),
    );
}

#[test]
fn every_breakpoint_one_access_hits_is_reported() {
    on_both_engines(
        &STORE_DWORD,
        |h| {
            arm(h, 0, WATCHED, RW_WRITE, LEN_1);
            arm(h, 2, WATCHED + 3, RW_READ_WRITE, LEN_1);
            arm(h, 3, WATCHED + 4, RW_WRITE, LEN_1);
            h.cpu.gpr[RBX] = WATCHED;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET | 0b0101, "{engine}");
        },
    );
}

#[test]
fn an_address_mask_widens_a_data_breakpoint() {
    // wrmsr DR1_ADDR_MASK = 0xff: DR1 then watches the whole 256-byte block.
    on_both_engines(
        &STORE_BYTE,
        |h| {
            h.cpu.wrmsr(MSR_DR1_ADDRESS_MASK, 0xff).unwrap();
            arm(h, 1, WATCHED, RW_WRITE, LEN_1);
            h.cpu.gpr[RBX] = WATCHED + 0x80;
        },
        |h, engine| assert_eq!(h.run(100), DEBUG_TRAP, "{engine}"),
    );
    on_both_engines(
        &STORE_BYTE,
        |h| {
            h.cpu.wrmsr(MSR_DR1_ADDRESS_MASK, 0xff).unwrap();
            arm(h, 1, WATCHED, RW_WRITE, LEN_1);
            h.cpu.gpr[RBX] = WATCHED + 0x100;
        },
        |h, engine| assert_eq!(h.run(100), Stop::Halted, "{engine}"),
    );
}

#[test]
fn the_address_mask_msrs_hold_32_bits() {
    let mut h = Harness::new(&[0xf4]);
    for (msr, slot) in [
        (MSR_DR0_ADDRESS_MASK, 0),
        (MSR_DR1_ADDRESS_MASK, 1),
        (MSR_DR1_ADDRESS_MASK + 1, 2),
        (MSR_DR3_ADDRESS_MASK, 3),
    ] {
        h.cpu.wrmsr(msr, 0xffff_ffff).unwrap();
        assert_eq!(h.cpu.rdmsr(msr).unwrap(), 0xffff_ffff);
        assert_eq!(h.cpu.debug.address_masks[slot], 0xffff_ffff);
        assert!(h.cpu.wrmsr(msr, 1 << 32).is_err(), "{msr:#x}");
    }
}

#[test]
fn an_io_breakpoint_needs_cr4_de_and_traps_after_the_access() {
    on_both_engines(
        &OUT_PORT_80,
        |h| {
            arm(h, 0, 0x80, RW_IO, LEN_1);
            h.cpu.cr4 |= CR4_DE;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 2, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET | 1, "{engine}");
        },
    );
    on_both_engines(
        &OUT_PORT_80,
        |h| arm(h, 0, 0x80, RW_IO, LEN_1),
        |h, engine| assert_eq!(h.run(100), Stop::Halted, "{engine}"),
    );
}

#[test]
fn general_detect_faults_on_a_debug_register_access_and_clears_gd() {
    on_both_engines(
        &READ_DR0,
        |h| {
            h.cpu.dr[0] = 0x1234;
            h.cpu.dr[7] |= DR7_GENERAL_DETECT;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE, "{engine}: a fault");
            assert_eq!(h.cpu.gpr[0], 0, "{engine}");
            assert_eq!(
                h.cpu.dr[6],
                DR6_RESET | DR6_DEBUG_REGISTER_ACCESS,
                "{engine}"
            );
            assert_eq!(h.cpu.dr[7] & DR7_GENERAL_DETECT, 0, "{engine}");
            assert_eq!(
                h.run(100),
                Stop::Halted,
                "{engine}: the handler may use them"
            );
            assert_eq!(h.cpu.gpr[0], 0x1234, "{engine}");
        },
    );
}

#[test]
fn tf_traps_after_each_instruction() {
    on_both_engines(
        &NOPS,
        |h| h.cpu.eflags |= TF,
        |h, engine| {
            for step in 1..=3 {
                assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
                assert_eq!(h.cpu.rip, CODE_BASE + step, "{engine}");
                assert_eq!(h.cpu.dr[6], DR6_RESET | DR6_SINGLE_STEP, "{engine}");
            }
        },
    );
}

#[test]
fn popf_that_sets_tf_traps_only_after_the_next_instruction() {
    on_both_engines(
        &SET_TF_WITH_POPF,
        |_| {},
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 11, "{engine}: after the first nop");
            assert_eq!(h.cpu.icount, 4, "{engine}");
        },
    );
}

#[test]
fn mov_ss_holds_its_single_step_trap_over_the_next_instruction() {
    on_both_engines(
        &MOV_SS_THEN_NOP,
        |h| h.cpu.eflags |= TF,
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 2, "{engine}: after mov ax, ss");
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(
                h.cpu.rip,
                CODE_BASE + 6,
                "{engine}: after the nop, not the mov ss"
            );
        },
    );
}

#[test]
fn btf_single_steps_branches_only() {
    on_both_engines(
        &NOP_JMP_NOP,
        |h| {
            h.cpu.wrmsr(MSR_DEBUGCTL, DEBUGCTL_BTF).unwrap();
            h.cpu.eflags |= TF;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 4, "{engine}: after the jmp");
            assert_eq!(h.cpu.icount, 2, "{engine}");
        },
    );
}

#[test]
fn a_repeated_string_instruction_traps_between_iterations() {
    on_both_engines(
        &REP_STOSB,
        |h| {
            arm(h, 0, WATCHED + 4, RW_WRITE, LEN_1);
            h.cpu.gpr[RCX] = 16;
            h.cpu.gpr[7] = WATCHED;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE, "{engine}: still on the instruction");
            assert_eq!(h.cpu.gpr[RCX], 11, "{engine}");
            assert_eq!(h.cpu.gpr[7], WATCHED + 5, "{engine}");
            assert_ne!(
                h.cpu.eflags & RF,
                0,
                "{engine}: resumes past its own breakpoint"
            );
            assert_eq!(h.run(100), Stop::Halted, "{engine}");
            assert_eq!(h.cpu.gpr[RCX], 0, "{engine}");
        },
    );
}

#[test]
fn bus_lock_detect_traps_after_a_split_lock_and_clears_dr6_bit_11() {
    on_both_engines(
        &LOCKED_ADD,
        |h| {
            h.cpu.wrmsr(MSR_DEBUGCTL, DEBUGCTL_BUS_LOCK_DETECT).unwrap();
            h.cpu.gpr[RBX] = WATCHED + 62;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 3, "{engine}");
            assert_eq!(h.cpu.dr[6], DR6_RESET & !DR6_BUS_LOCK, "{engine}");
        },
    );
}

#[test]
fn bus_lock_detect_ignores_aligned_locks_and_unlocked_splits() {
    for (code, address) in [
        (&LOCKED_ADD[..], WATCHED + 60),
        (&PLAIN_ADD[..], WATCHED + 62),
    ] {
        on_both_engines(
            code,
            |h| {
                h.cpu.wrmsr(MSR_DEBUGCTL, DEBUGCTL_BUS_LOCK_DETECT).unwrap();
                h.cpu.gpr[RBX] = address;
            },
            |h, engine| assert_eq!(h.run(100), Stop::Halted, "{engine}"),
        );
    }
}

#[test]
fn dr6_bit_11_is_writable_now_that_bus_lock_detect_exists() {
    // mov dr6, rax; hlt
    let mut h = Harness::new(&[0x0f, 0x23, 0xf0, 0xf4]);
    h.cpu.gpr[0] = 0;
    h.run_to_halt();
    assert_eq!(h.cpu.dr[6], DR6_RESET & !DR6_BUS_LOCK);
}

#[test]
fn debugctl_rejects_bits_that_do_not_exist() {
    let mut h = Harness::new(&[0xf4]);
    assert!(h.cpu.wrmsr(MSR_DEBUGCTL, 1 << 6).is_err());
    h.cpu.wrmsr(MSR_DEBUGCTL, 0x3f).unwrap();
    assert_eq!(h.cpu.rdmsr(MSR_DEBUGCTL).unwrap(), 0x3f);
}

#[test]
fn a_watchpoint_inside_a_loop_stops_both_engines_at_the_same_iteration() {
    assert_jit_matches_interpreter(&STORE_LOOP, |h| {
        arm(h, 0, WATCHED + 40, RW_WRITE, LEN_4);
        h.cpu.gpr[RBX] = WATCHED;
    });
    let mut h = Harness::new(&STORE_LOOP);
    arm(&mut h, 0, WATCHED + 40, RW_WRITE, LEN_4);
    h.cpu.gpr[RBX] = WATCHED;
    assert_eq!(h.run(10_000), DEBUG_TRAP);
    assert_eq!(h.cpu.gpr[RCX], 90);
}

#[test]
fn arming_dr7_mid_block_stops_the_block_before_the_watched_store() {
    assert_jit_matches_interpreter(&ARM_THEN_STORE, |h| {
        h.cpu.dr[0] = WATCHED;
        h.cpu.gpr[0] = dr7_for(0, RW_WRITE, LEN_4);
        h.cpu.gpr[RBX] = WATCHED;
    });
    on_both_engines(
        &ARM_THEN_STORE,
        |h| {
            h.cpu.dr[0] = WATCHED;
            h.cpu.gpr[0] = dr7_for(0, RW_WRITE, LEN_4);
            h.cpu.gpr[RBX] = WATCHED;
        },
        |h, engine| {
            assert_eq!(h.run(100), DEBUG_TRAP, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 5, "{engine}");
        },
    );
}

#[test]
fn single_stepping_matches_between_the_engines() {
    assert_jit_matches_interpreter(&SET_TF_WITH_POPF, |_| {});
}

/// A #DB handler the way Linux's `hw_breakpoint_handler` returns from an instruction breakpoint:
/// `or dword [rsp+16], 0x10000` (RF in the frame's RFLAGS); `iretq`.
const SET_RF_AND_RETURN: [u8; 10] = [0x81, 0x4c, 0x24, 0x10, 0x00, 0x00, 0x01, 0x00, 0x48, 0xcf];
const DEBUG_HANDLER: u64 = SCRATCH + 0x800;

#[test]
fn a_handler_that_returns_with_rf_resumes_past_the_breakpoint_and_delivery_hits_no_watchpoint() {
    use super::test_support::{Gate, install_idt, set_gate};
    on_both_engines(
        &NOPS,
        |h| {
            install_idt(h);
            h.cpu.seg[super::SEG_CS].sel = super::test_support::KERNEL_CS;
            h.cpu.seg[super::SEG_SS].sel = super::test_support::KERNEL_DS;
            set_gate(h, VECTOR_DEBUG, Gate::interrupt(DEBUG_HANDLER));
            h.write(DEBUG_HANDLER, &SET_RF_AND_RETURN);
            arm(h, 0, CODE_BASE + 1, RW_EXECUTE, LEN_1);
            // The SS slot of the frame the #DB delivery pushes: written by delivery only.
            arm(h, 1, STACK_TOP - 8, RW_WRITE, LEN_8);
        },
        |h, engine| {
            assert_eq!(h.run_delivering(1000), Stop::Halted, "{engine}");
            assert_eq!(h.cpu.rip, CODE_BASE + 4, "{engine}");
            assert_eq!(
                h.cpu.dr[6],
                DR6_RESET | 1,
                "{engine}: one instruction breakpoint"
            );
            assert_eq!(h.cpu.eflags & RF, 0, "{engine}");
        },
    );
}

#[test]
fn bus_lock_detect_alone_keeps_the_jit_running() {
    // Linux enables it at boot wherever it exists; it must not send everything to the
    // interpreter. Split locks are caught in `exec`, which runs every locked instruction.
    let mut h = Harness::new(&[0xf4]);
    h.cpu.wrmsr(MSR_DEBUGCTL, DEBUGCTL_BUS_LOCK_DETECT).unwrap();
    assert!(!h.cpu.debug_slow_path());
    assert_jit_matches_interpreter(&LOCKED_ADD, |h| {
        h.cpu.wrmsr(MSR_DEBUGCTL, DEBUGCTL_BUS_LOCK_DETECT).unwrap();
        h.cpu.gpr[RBX] = WATCHED + 62;
    });
}

#[test]
fn a_32_bit_iret_in_real_mode_loads_rf() {
    // push dword 0x10002 (RF); push dword 0 (CS); push dword 0x7c14; iretd; hlt
    let code = [
        0x66, 0x68, 0x02, 0x00, 0x01, 0x00, 0x66, 0x68, 0x00, 0x00, 0x00, 0x00, 0x66, 0x68, 0x14,
        0x7c, 0x00, 0x00, 0x66, 0xcf, 0xf4,
    ];
    let mut h = Harness::real(&code);
    assert_eq!(h.run(4), Stop::Limit);
    assert_eq!(h.cpu.rip, REAL_CODE + 20);
    assert_ne!(h.cpu.eflags & RF, 0);
    assert_eq!(h.run(10), Stop::Halted);
    assert_eq!(
        h.cpu.eflags & RF,
        0,
        "cleared once the next instruction completes"
    );
}
