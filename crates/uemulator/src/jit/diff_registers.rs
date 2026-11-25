//! Where a backend that keeps guest registers in host registers (the aarch64 one keeps rax..rdi in
//! x8..x15 across blocks, `docs/plans/06-jit.md`, "Pinned guest registers") or in locals has to
//! write them back to `Cpu` or reload them: a fault after earlier instructions of the block wrote
//! registers (also delivered through the IDT, to a handler that reads them), an interpreted
//! instruction between inline ones (it reads the written registers from `Cpu` and writes its own
//! there), the flag helpers and slow-path memory helpers in the middle of a block, byte and word
//! writes to registers the block has not touched yet, `push rsp`/`pop rsp`, a rotate by CL, a
//! store that ends the block because it hit translated code, and successors that read what their
//! predecessor wrote (linked exits and the chain stub).
//!
//! Such a backend must also write registers back before the slow-path load and store helpers,
//! not only before `helper_exec`, and reload them after even when the helper failed: a nested
//! page fault in an SVM guest is a #VMEXIT taken inside the helper (`Cpu::vmexit`), which saves
//! the guest's rsp and rax to the VMCB and loads the host's into `Cpu::gpr`. That case is
//! `cpu/svm.rs`'s `a_nested_fault_in_the_middle_of_a_block_saves_the_guest_and_restores_the_host`,
//! which runs on the JIT too.

use super::diff_tests::*;
use crate::testkit::*;

/// The non-canonical addresses below fault with #GP(0), or #SS(0) for a stack access.
const GENERAL_PROTECTION: u8 = 13;
const STACK_FAULT: u8 = 12;

#[test]
fn a_faulting_load_sees_the_registers_written_before_it() {
    // mov rax, 0x1111; mov ah, 0x55; inc r9; mov rdx, 0x8000000000000000; mov rcx, [rdx]; hlt
    check_stop(
        &[
            0x48, 0xc7, 0xc0, 0x11, 0x11, 0x00, 0x00, 0xb4, 0x55, 0x49, 0xff, 0xc1, 0x48, 0xba,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48, 0x8b, 0x0a, 0xf4,
        ],
        fault(GENERAL_PROTECTION, Some(0)),
    );
}

#[test]
fn a_faulting_store_sees_the_registers_written_before_it() {
    // lea rbx, [rbx + 8]; mov bl, 0x7f; mov rdx, 0x8000000000000000; mov [rdx], rbx; hlt
    check_stop(
        &[
            0x48, 0x8d, 0x5b, 0x08, 0xb3, 0x7f, 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x80, 0x48, 0x89, 0x1a, 0xf4,
        ],
        fault(GENERAL_PROTECTION, Some(0)),
    );
}

#[test]
fn a_faulting_pop_leaves_rsp_and_the_earlier_writes() {
    // mov rax, 5; add rbp, rax; mov rsp, 0x8000000000000000; pop rbx; hlt
    check_stop(
        &[
            0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x01, 0xc5, 0x48, 0xbc, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x5b, 0xf4,
        ],
        fault(STACK_FAULT, Some(0)),
    );
}

#[test]
fn an_interpreted_instruction_reads_and_writes_registers_between_inline_ones() {
    // mov rax, 3; mov rcx, 5; mul rcx; add rax, rdx; lea rbx, [rax + rcx]; mov r8, rdx; hlt
    check(&[
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00, 0x48,
        0xf7, 0xe1, 0x48, 0x01, 0xd0, 0x48, 0x8d, 0x1c, 0x08, 0x49, 0x89, 0xd0, 0xf4,
    ]);
}

#[test]
fn a_fault_after_an_interpreted_instruction_keeps_what_it_wrote() {
    // mov rax, 3; mov rcx, 5; mul rcx; mov rdx, 0x8000000000000000; add rax, [rdx]; hlt
    check_stop(
        &[
            0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc1, 0x05, 0x00, 0x00, 0x00,
            0x48, 0xf7, 0xe1, 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48,
            0x03, 0x02, 0xf4,
        ],
        fault(GENERAL_PROTECTION, Some(0)),
    );
}

#[test]
fn byte_and_word_writes_to_untouched_registers_keep_the_other_bits() {
    // mov al, 0x12; mov ah, 0x34; mov bx, 0x5678; mov ch, dl; add dh, al; mov dl, ah; inc si; hlt
    check(&[
        0xb0, 0x12, 0xb4, 0x34, 0x66, 0xbb, 0x78, 0x56, 0x88, 0xd5, 0x00, 0xc6, 0x88, 0xe2, 0x66,
        0xff, 0xc6, 0xf4,
    ]);
}

#[test]
fn byte_writes_merge_into_registers_written_in_full() {
    // mov rax, -1; mov ah, 0x12; mov ecx, eax; mov cl, 0x80; mov rdx, rcx; sub dh, ah; hlt
    check(&[
        0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0xb4, 0x12, 0x89, 0xc1, 0xb1, 0x80, 0x48, 0x89,
        0xca, 0x28, 0xe6, 0xf4,
    ]);
}

#[test]
fn push_rsp_pushes_the_old_value_and_pop_rsp_keeps_the_popped_one() {
    // mov rax, rsp; push rsp; pop rbx; push rax; pop rsp; push rsp; pop rcx; hlt
    check(&[0x48, 0x89, 0xe0, 0x54, 0x5b, 0x50, 0x5c, 0x54, 0x59, 0xf4]);
    // lea rax, [rsp - 64]; push rax; pop rsp; mov rbx, rsp; hlt
    check(&[
        0x48, 0x8d, 0x44, 0x24, 0xc0, 0x50, 0x5c, 0x48, 0x89, 0xe3, 0xf4,
    ]);
}

#[test]
fn a_rotate_by_cl_reads_the_count_just_written() {
    // mov ecx, 5; rol eax, cl; ror rdx, cl; add rax, rdx; mov ecx, 33; rol esi, cl;
    // rol rdi, cl; sub rsi, rdi; hlt
    check(&[
        0xb9, 0x05, 0x00, 0x00, 0x00, 0xd3, 0xc0, 0x48, 0xd3, 0xca, 0x48, 0x01, 0xd0, 0xb9, 0x21,
        0x00, 0x00, 0x00, 0xd3, 0xc6, 0x48, 0xd3, 0xc7, 0x48, 0x29, 0xfe, 0xf4,
    ]);
}

#[test]
fn setcc_and_cmov_write_registers_the_block_holds() {
    // mov eax, 1; cmp rax, rcx; setb ah; cmovae rdx, rax; seta bh; cmovb ebx, ecx; hlt
    check(&[
        0xb8, 0x01, 0x00, 0x00, 0x00, 0x48, 0x39, 0xc8, 0x0f, 0x92, 0xc4, 0x48, 0x0f, 0x43, 0xd0,
        0x0f, 0x97, 0xc7, 0x0f, 0x42, 0xd9, 0xf4,
    ]);
}

#[test]
fn a_store_into_the_running_code_writes_the_registers_back_before_leaving() {
    // inc rcx; lea rdx, [rip + target]; mov byte ptr [rdx], 0x90; target: inc rcx;
    // add rcx, rcx; hlt
    check(&[
        0x48, 0xff, 0xc1, 0x48, 0x8d, 0x15, 0x03, 0x00, 0x00, 0x00, 0xc6, 0x02, 0x90, 0x48, 0xff,
        0xc1, 0x48, 0x01, 0xc9, 0xf4,
    ]);
}

#[test]
fn a_loop_through_its_own_link_carries_the_registers() {
    // mov ecx, 10; xor eax, eax; again: add rax, rcx; lea rbx, [rbx + rax]; dec ecx;
    // jnz again; hlt
    check(&[
        0xb9, 0x0a, 0x00, 0x00, 0x00, 0x31, 0xc0, 0x48, 0x01, 0xc8, 0x48, 0x8d, 0x1c, 0x03, 0xff,
        0xc9, 0x75, 0xf5, 0xf4,
    ]);
}

#[test]
fn chained_blocks_read_what_their_predecessor_wrote() {
    // mov rbx, 1; jmp next; next: add rbx, rbx; call func; add rbx, rax; hlt;
    // func: lea rax, [rbx + 7]; ret
    check(&[
        0x48, 0xc7, 0xc3, 0x01, 0x00, 0x00, 0x00, 0xeb, 0x00, 0x48, 0x01, 0xdb, 0xe8, 0x04, 0x00,
        0x00, 0x00, 0x48, 0x01, 0xc3, 0xf4, 0x48, 0x8d, 0x43, 0x07, 0xc3,
    ]);
}

/// The handler of a fault taken in the middle of a block runs on the pushed frame and reads the
/// registers the block had written by then: the exit stub stored them to `Cpu` before the run loop
/// delivered the exception through the IDT.
#[test]
fn a_fault_delivered_through_the_idt_shows_the_handler_the_registers_written_before_it() {
    use crate::cpu::test_support::{Gate, install_idt, read_frame, set_gate};
    const HANDLER_OFFSET: usize = 0x40;
    const SAVED: u64 = SCRATCH + 0x500;
    // mov rax, 0x1234; mov rbx, 0x5678; sub rsp, 0x40; mov rdx, 1 << 63; mov [rdx], rax; hlt
    let mut code = vec![
        0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, 0x48, 0xc7, 0xc3, 0x78, 0x56, 0x00, 0x00, 0x48,
        0x83, 0xec, 0x40, 0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48, 0x89,
        0x02, 0xf4,
    ];
    code.resize(HANDLER_OFFSET, 0x90);
    // handler: mov [SAVED], rax; mov [SAVED + 8], rbx; mov [SAVED + 16], rsp; hlt
    code.extend_from_slice(&[
        0x48, 0x89, 0x04, 0x25, 0x00, 0x05, 0x40, 0x00, 0x48, 0x89, 0x1c, 0x25, 0x08, 0x05, 0x40,
        0x00, 0x48, 0x89, 0x24, 0x25, 0x10, 0x05, 0x40, 0x00, 0xf4,
    ]);
    let run = |jit: bool| {
        let mut harness = Harness::new(&code);
        install_idt(&mut harness);
        set_gate(
            &mut harness,
            GENERAL_PROTECTION,
            Gate::interrupt(CODE_BASE + HANDLER_OFFSET as u64),
        );
        if jit && !harness.enable_jit() {
            return None;
        }
        let stop = harness.run_delivering(1000);
        assert_eq!(stop, Stop::Halted, "{}", harness.cpu.dump_regs());
        Some(harness)
    };
    let oracle = run(false).unwrap();
    let Some(jitted) = run(true) else {
        return;
    };
    assert!(jitted.cpu.jit.as_ref().unwrap().stats().translated > 0);
    for harness in [&oracle, &jitted] {
        assert_eq!(harness.read_u64(SAVED), 0x1234, "rax in the handler");
        assert_eq!(harness.read_u64(SAVED + 8), 0x5678, "rbx in the handler");
        let handler_rsp = harness.read_u64(SAVED + 16);
        let frame = read_frame(harness, handler_rsp, true);
        assert_eq!(frame.rsp, STACK_TOP - 0x40, "the rsp the block had lowered");
        assert_eq!(frame.error_code, Some(0));
    }
    assert_eq!(jitted.snapshot(), oracle.snapshot());
    assert!(jitted.cpu.ram[0..] == oracle.cpu.ram[0..], "RAM differs");
}

/// `cpuid` always runs in the interpreter (`helper_exec`): it reads eax and ecx the block just
/// wrote and writes all four, which the block's later inline instructions then read.
#[test]
fn an_interpreted_cpuid_reads_and_writes_the_pinned_registers() {
    // mov eax, 7; xor ecx, ecx; mov rbx, -1; mov rdx, -1; cpuid; add rax, rbx; xor rsi, rdx;
    // lea rdi, [rcx + rax]; hlt
    check(&[
        0xb8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xc9, 0x48, 0xc7, 0xc3, 0xff, 0xff, 0xff, 0xff, 0x48,
        0xc7, 0xc2, 0xff, 0xff, 0xff, 0xff, 0x0f, 0xa2, 0x48, 0x01, 0xd8, 0x48, 0x31, 0xd6, 0x48,
        0x8d, 0x3c, 0x01, 0xf4,
    ]);
}

/// Conditions the host flags cannot express call `helper_cond` (parity here), and rotates by an
/// immediate merge CF/OF through `helper_set_rotate_flags`: both are Rust calls in the middle of a
/// block, around which the pinned registers must survive.
#[test]
fn registers_survive_the_flag_helpers() {
    // mov rax, 3; add rax, rbx; setp cl; mov rsi, rax; rol rdx, 5; jp 1f; inc rdi; 1: adc rdi,
    // rsi; hlt
    check(&[
        0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x48, 0x01, 0xd8, 0x0f, 0x9a, 0xc1, 0x48, 0x89,
        0xc6, 0x48, 0xc1, 0xc2, 0x05, 0x7a, 0x03, 0x48, 0xff, 0xc7, 0x48, 0x11, 0xf7, 0xf4,
    ]);
}

/// The first access to a page goes through the slow path (the load or store thunk and its Rust
/// helper) between inline register writes.
#[test]
fn registers_survive_a_slow_path_load_and_store() {
    // mov rax, 9; mov rcx, [rbx]; add rcx, rax; mov [rsi], rcx; lea rdx, [rcx + rax]; hlt
    check(&[
        0x48, 0xc7, 0xc0, 0x09, 0x00, 0x00, 0x00, 0x48, 0x8b, 0x0b, 0x48, 0x01, 0xc1, 0x48, 0x89,
        0x0e, 0x48, 0x8d, 0x14, 0x01, 0xf4,
    ]);
}
