//! Real-mode and flat 32-bit tests of the instructions in `legacy.rs` and `legacy_flow.rs`. The
//! snippets are assembled by `hack/asm.sh` (`BITS=16` / `BITS=32`); real-mode ones run at
//! 0000:7c00.

use super::decode::CodeMode;
use super::legacy_support::*;
use super::*;
use crate::testkit::*;

fn real(code: &[u8]) -> Harness {
    let mut h = Harness::real(code);
    h.run_to_halt();
    h
}

fn real_fault(code: &[u8]) -> (u8, Option<u32>) {
    let mut h = Harness::real(code);
    match h.run(10_000) {
        Stop::Fault {
            vector, error_code, ..
        } => (vector, error_code),
        other => panic!("expected a fault, got {other:?}"),
    }
}

fn reg16(h: &Harness, index: usize) -> u64 {
    h.cpu.gpr[index] & 0xffff
}

const HANDLER: u64 = 0x60_0000;
const RING3_CODE: u64 = 0x61_0000;
const RING3_STACK: u64 = 0x62_0000;
const DATA: u64 = 0x63_0000;

fn prot(code: &[u8]) -> Harness {
    protected32(code)
}

fn halted(h: &mut Harness) {
    assert_eq!(
        h.run_delivering(100_000),
        Stop::Halted,
        "{}",
        h.cpu.dump_regs()
    );
}

fn prot_fault(h: &mut Harness) -> (u8, Option<u32>) {
    match h.run(100_000) {
        Stop::Fault {
            vector, error_code, ..
        } => (vector, error_code),
        other => panic!("expected a fault, got {other:?}"),
    }
}

fn dword(h: &Harness, address: u64) -> u32 {
    u32::from_le_bytes(h.read(address, 4).try_into().unwrap())
}

#[test]
fn sixteen_bit_addressing_forms_use_the_data_segment() {
    let code = [
        0xb8, 0x34, 0x12, 0x8e, 0xd8, 0xbb, 0x02, 0x00, 0xbe, 0x03, 0x00, 0xc6, 0x40, 0x05, 0x77,
        0x8a, 0x40, 0x05, 0x8a, 0x0e, 0x0a, 0x00, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.read(0x12340 + 10, 1), [0x77]);
    assert_eq!(h.cpu.gpr[RAX] & 0xff, 0x77);
    assert_eq!(h.cpu.gpr[RCX] & 0xff, 0x77);
}

#[test]
fn sixteen_bit_effective_addresses_wrap_at_64_kib() {
    let code = [
        0xbb, 0xff, 0xff, 0xc6, 0x06, 0x01, 0x00, 0x5a, 0x8a, 0x47, 0x02, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RAX] & 0xff, 0x5a);
}

#[test]
fn bp_based_addresses_default_to_the_stack_segment() {
    let code = [
        0xb8, 0x00, 0x01, 0x8e, 0xd0, 0xbd, 0x10, 0x00, 0xc7, 0x46, 0x00, 0xef, 0xbe, 0xbb, 0x10,
        0x00, 0x3e, 0x8b, 0x0f, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.read(0x1010, 2), [0xef, 0xbe]);
    assert_eq!(reg16(&h, RCX), 0, "ds:[0x10] is not ss:[0x10]");
}

#[test]
fn push_pop_call_and_ret_use_sixteen_bit_slots() {
    let code = [
        0xb8, 0x34, 0x12, 0x50, 0x5b, 0x66, 0xe8, 0x04, 0x00, 0x00, 0x00, 0xba, 0x07, 0x00, 0xf4,
        0xb9, 0x05, 0x00, 0x66, 0xc3,
    ];
    let h = real(&code);
    assert_eq!(reg16(&h, RBX), 0x1234);
    assert_eq!(reg16(&h, RCX), 5);
    assert_eq!(reg16(&h, RDX), 7);
    assert_eq!(reg16(&h, RSP), REAL_STACK);
}

#[test]
fn sixteen_bit_stack_pointer_wraps_without_touching_the_upper_half() {
    let code = [0x66, 0xbc, 0x00, 0x00, 0x34, 0x12, 0x31, 0xe4, 0x50, 0xf4];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RSP], 0x1234fffe);
}

#[test]
fn operand_size_prefix_gives_thirty_two_bit_registers_in_real_mode() {
    let code = [
        0x66, 0xb8, 0x78, 0x56, 0x34, 0x12, 0x66, 0x31, 0xdb, 0x66, 0x01, 0xc3, 0x66, 0xc1, 0xeb,
        0x10, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RBX], 0x1234);
    assert_eq!(h.cpu.gpr[RAX], 0x1234_5678);
}

#[test]
fn address_size_prefix_allows_offsets_up_to_the_segment_limit_only() {
    let code = [
        0x66, 0xbb, 0x00, 0x20, 0x00, 0x00, 0x67, 0x66, 0xc7, 0x03, 0xef, 0xbe, 0xad, 0xde, 0x67,
        0x66, 0x8b, 0x0b, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RCX], 0xdead_beef);
}

#[test]
fn offset_above_the_real_mode_limit_is_a_general_protection_fault() {
    let code = [0x66, 0xbb, 0x00, 0x00, 0x01, 0x00, 0x67, 0x8a, 0x03, 0xf4];
    assert_eq!(real_fault(&code), (13, Some(0)));
}

#[test]
fn int_through_the_vector_table_and_iret() {
    let code = [
        0xc7, 0x06, 0x84, 0x00, 0x00, 0x7d, 0xc7, 0x06, 0x86, 0x00, 0x00, 0x00, 0xfb, 0xcd, 0x21,
        0xba, 0x09, 0x00, 0xf4,
    ];
    let handler = [0xb8, 0x55, 0x00, 0xcf];
    let mut h = Harness::real(&code);
    h.write(0x7d00, &handler);
    assert_eq!(h.run_delivering(10_000), Stop::Halted);
    assert_eq!(reg16(&h, RAX), 0x55);
    assert_eq!(reg16(&h, RDX), 9);
    assert_eq!(reg16(&h, RSP), REAL_STACK);
    assert_ne!(h.cpu.eflags_full() & flags::IF, 0, "iret restores IF");
}

#[test]
fn int_pushes_flags_cs_ip_and_clears_if_and_tf() {
    let code = [
        0xc7, 0x06, 0x84, 0x00, 0x00, 0x7d, 0xc7, 0x06, 0x86, 0x00, 0x00, 0x00, 0xfb, 0x9c, 0x58,
        0x0d, 0x00, 0x01, 0x50, 0x9d, 0xcd, 0x21, 0xf4,
    ];
    let handler = [0xf4];
    let mut h = Harness::real(&code);
    h.write(0x7d00, &handler);
    assert_eq!(h.run_delivering(10_000), Stop::Halted);
    assert_eq!(h.cpu.rip, 0x7d01, "stopped on the handler's hlt");
    assert_eq!(reg16(&h, RSP), REAL_STACK - 6);
    let frame = h.read(REAL_STACK - 6, 6);
    let return_ip = u16::from_le_bytes([frame[0], frame[1]]);
    assert_eq!(return_ip as u64, REAL_CODE + code.len() as u64 - 1);
    assert_eq!(u16::from_le_bytes([frame[2], frame[3]]), 0);
    let pushed_flags = u16::from_le_bytes([frame[4], frame[5]]) as u64;
    assert_ne!(pushed_flags & flags::IF, 0);
    assert_ne!(pushed_flags & flags::TF, 0);
    assert_eq!(h.cpu.eflags_full() & (flags::IF | flags::TF), 0);
}

#[test]
fn far_jump_call_and_return_in_real_mode() {
    let code = [0xea, 0x20, 0x00, 0x00, 0x10];
    let far_code = [0xb8, 0x44, 0x00, 0x9a, 0x40, 0x00, 0x00, 0x20, 0xf4];
    let callee = [0xcb];
    let mut h = Harness::real(&code);
    h.write(0x10020, &far_code);
    h.write(0x20040, &callee);
    h.run_to_halt();
    assert_eq!(h.cpu.seg[SEG_CS].sel, 0x1000);
    assert_eq!(h.cpu.seg[SEG_CS].base, 0x10000);
    assert_eq!(reg16(&h, RAX), 0x44);
    assert_eq!(reg16(&h, RSP), REAL_STACK);
}

#[test]
fn rep_movsb_applies_the_segment_bases_of_ds_and_es() {
    let code = [
        0xb8, 0x00, 0x01, 0x8e, 0xd8, 0xb8, 0x00, 0x02, 0x8e, 0xc0, 0x31, 0xf6, 0xbf, 0x10, 0x00,
        0xb9, 0x04, 0x00, 0xf3, 0xa4, 0xf4,
    ];
    let mut h = Harness::real(&code);
    h.write(0x1000, &[1, 2, 3, 4, 5]);
    h.run_to_halt();
    assert_eq!(h.read(0x2010, 5), [1, 2, 3, 4, 0]);
    assert_eq!(reg16(&h, RCX), 0);
    assert_eq!(reg16(&h, RSI), 4);
    assert_eq!(reg16(&h, RDI), 0x14);
}

#[test]
fn string_instructions_with_a_sixteen_bit_counter_leave_the_upper_halves_alone() {
    let code = [
        0x66, 0xb9, 0x03, 0x00, 0xcd, 0xab, 0x66, 0xbf, 0x00, 0x00, 0x23, 0x01, 0xbf, 0x00, 0x01,
        0xb0, 0x7e, 0xf3, 0xaa, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RCX], 0xabcd_0000);
    assert_eq!(h.cpu.gpr[RDI], 0x0123_0103);
    assert_eq!(h.read(0x100, 3), [0x7e; 3]);
}

#[test]
fn pusha_and_popa_round_trip_the_registers() {
    let code = [
        0xb8, 0x01, 0x00, 0xb9, 0x02, 0x00, 0xba, 0x03, 0x00, 0xbb, 0x04, 0x00, 0xbd, 0x06, 0x00,
        0xbe, 0x07, 0x00, 0xbf, 0x08, 0x00, 0x60, 0x31, 0xc0, 0x31, 0xc9, 0x31, 0xdb, 0xbf, 0x00,
        0x00, 0x61, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(
        [RAX, RCX, RDX, RBX, RBP, RSI, RDI].map(|r| reg16(&h, r)),
        [1, 2, 3, 4, 6, 7, 8]
    );
    assert_eq!(reg16(&h, RSP), REAL_STACK);
}

#[test]
fn enter_and_leave_build_and_tear_down_a_frame() {
    let code = [
        0xbd, 0x11, 0x11, 0xc8, 0x08, 0x00, 0x00, 0x89, 0xe8, 0xc9, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(reg16(&h, RAX), REAL_STACK - 2);
    assert_eq!(reg16(&h, RBP), 0x1111);
    assert_eq!(reg16(&h, RSP), REAL_STACK);
}

#[test]
fn inc_and_dec_registers_are_opcodes_outside_64_bit_mode() {
    let code = [
        0xb8, 0xff, 0xff, 0x40, 0x9c, 0x59, 0xbb, 0x01, 0x00, 0x4b, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(reg16(&h, RAX), 0);
    assert_ne!(reg16(&h, RCX) & flags::ZF, 0);
    assert_eq!(reg16(&h, RBX), 0);
    assert_eq!(reg16(&h, RCX) & flags::CF, 0, "inc leaves CF alone");
}

#[test]
fn lea_truncates_to_the_address_size() {
    let code = [0xbb, 0xff, 0xff, 0x8d, 0x47, 0x02, 0xf4];
    let h = real(&code);
    assert_eq!(reg16(&h, RAX), 1);
}

#[test]
fn daa_and_aam_adjust_decimal_values() {
    let code = [
        0xb0, 0x15, 0x04, 0x27, 0x27, 0x88, 0xc3, 0xb0, 0x19, 0xd4, 0x0a, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(h.cpu.gpr[RBX] & 0xff, 0x42);
    assert_eq!(reg16(&h, RAX), 0x0205);
}

#[test]
fn loop_and_jcxz_count_in_cx() {
    let code = [
        0x66, 0xb9, 0x03, 0x00, 0x01, 0x00, 0x31, 0xc0, 0x40, 0xe2, 0xfd, 0xe3, 0x03, 0xb8, 0xad,
        0xde, 0xf4,
    ];
    let h = real(&code);
    assert_eq!(reg16(&h, RAX), 3);
    assert_eq!(h.cpu.gpr[RCX], 0x10000);
}

#[test]
fn xlat_indexes_through_bx_in_the_data_segment() {
    let code = [0xbb, 0x00, 0x03, 0xb0, 0x02, 0xd7, 0xf4];
    let mut h = Harness::real(&code);
    h.write(0x300, &[0x10, 0x20, 0x30]);
    h.run_to_halt();
    assert_eq!(h.cpu.gpr[RAX] & 0xff, 0x30);
}

#[test]
fn les_and_lds_load_a_far_pointer() {
    let code = [0xbb, 0x00, 0x04, 0xc4, 0x3f, 0xc5, 0x77, 0x04, 0xf4];
    let mut h = Harness::real(&code);
    h.write(0x400, &[0x34, 0x12, 0x00, 0x20, 0x78, 0x56, 0x00, 0x30]);
    h.run_to_halt();
    assert_eq!(reg16(&h, RDI), 0x1234);
    assert_eq!(h.cpu.seg[SEG_ES].sel, 0x2000);
    assert_eq!(h.cpu.seg[SEG_ES].base, 0x20000);
    assert_eq!(reg16(&h, RSI), 0x5678);
    assert_eq!(h.cpu.seg[SEG_DS].sel, 0x3000);
}

#[test]
fn flat_thirty_two_bit_code_uses_dword_stack_slots() {
    let code = [
        0x40, 0x4b, 0x50, 0xe8, 0x07, 0x00, 0x00, 0x00, 0x59, 0xba, 0x09, 0x00, 0x00, 0x00, 0xf4,
        0xc3,
    ];
    let mut h = prot(&code);
    let esp = h.cpu.gpr[RSP];
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RAX], 1);
    assert_eq!(h.cpu.gpr[RBX] & 0xffff_ffff, 0xffff_ffff);
    assert_eq!(h.cpu.gpr[RCX], 1, "the popped eax");
    assert_eq!(h.cpu.gpr[RDX], 9);
    assert_eq!(
        h.cpu.gpr[RSP], esp,
        "balanced pushes and pops of 4 bytes each"
    );
}

#[test]
fn data_segment_base_offsets_every_access() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0xc7, 0x05, 0x10, 0x00, 0x00, 0x00, 0x78, 0x56,
        0x34, 0x12, 0x8b, 0x1d, 0x10, 0x00, 0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32, 0xfffff, 0x92, 0xc),
    );
    halted(&mut h);
    assert_eq!(dword(&h, DATA + 0x10), 0x1234_5678);
    assert_eq!(h.cpu.gpr[RBX], 0x1234_5678);
    assert_eq!(h.cpu.seg[SEG_DS].base, DATA);
}

#[test]
fn access_past_the_segment_limit_is_a_general_protection_fault() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0xa1, 0x00, 0x10, 0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32, 0xfff, 0x92, 0x4),
    );
    assert_eq!(prot_fault(&mut h), (13, Some(0)));
}

#[test]
fn stack_segment_limit_violation_is_a_stack_fault() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd0, 0xbc, 0x00, 0x20, 0x00, 0x00, 0x50, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32, 0xfff, 0x92, 0x4),
    );
    assert_eq!(prot_fault(&mut h), (12, Some(0)));
}

#[test]
fn loading_a_not_present_segment_faults_with_the_selector() {
    let code = [0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0xf4];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(0, 0xfffff, 0x12, 0xc),
    );
    assert_eq!(prot_fault(&mut h), (11, Some(0x38)));
}

#[test]
fn loading_ds_with_a_selector_beyond_the_gdt_is_a_general_protection_fault() {
    let code = [0x66, 0xb8, 0x38, 0x12, 0x66, 0x8e, 0xd8, 0xf4];
    let mut h = prot(&code);
    assert_eq!(prot_fault(&mut h), (13, Some(0x1238)));
}

#[test]
fn interrupt_gate_pushes_eflags_cs_eip_and_clears_if() {
    let code = [0xfb, 0xcd, 0x30, 0xba, 0x05, 0x00, 0x00, 0x00, 0xf4];
    let handler = [0xb8, 0x77, 0x00, 0x00, 0x00, 0x9c, 0x5b, 0xcf];
    let mut h = prot(&code);
    set_idt_gate(&mut h, 0x30, LEGACY_CS, HANDLER as u32, 0x8e);
    h.write(HANDLER, &handler);
    let esp = h.cpu.gpr[RSP];
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RAX], 0x77);
    assert_eq!(h.cpu.gpr[RDX], 5);
    assert_eq!(h.cpu.gpr[RSP], esp, "iretd restored the stack");
    assert_eq!(
        h.cpu.gpr[RBX] & flags::IF,
        0,
        "IF was clear inside an interrupt gate"
    );
    assert_ne!(h.cpu.eflags_full() & flags::IF, 0, "iretd restored IF");
}

#[test]
fn trap_gate_leaves_if_set() {
    let code = [0xfb, 0xcd, 0x31, 0xf4];
    let handler = [0x9c, 0x5b, 0xcf];
    let mut h = prot(&code);
    set_idt_gate(&mut h, 0x31, LEGACY_CS, HANDLER as u32, 0x8f);
    h.write(HANDLER, &handler);
    halted(&mut h);
    assert_ne!(h.cpu.gpr[RBX] & flags::IF, 0);
}

#[test]
fn exception_with_an_error_code_pushes_it_below_eip() {
    let code = [0x66, 0xb8, 0x38, 0x12, 0x66, 0x8e, 0xd8, 0xf4];
    let handler = [0x5b, 0x59, 0xf4];
    let mut h = prot(&code);
    set_idt_gate(&mut h, 13, LEGACY_CS, HANDLER as u32, 0x8e);
    h.write(HANDLER, &handler);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x1238, "the error code");
    assert_eq!(h.cpu.gpr[RCX], CODE_BASE + 4, "the faulting eip");
}

#[test]
fn gate_through_a_missing_idt_entry_is_a_general_protection_fault_with_the_idt_flag() {
    let code = [0xcd, 0x90, 0xf4];
    let handler = [0x5b, 0xf4];
    let mut h = prot(&code);
    h.cpu.idtr.limit = 0x7f;
    set_idt_gate(&mut h, 13, LEGACY_CS, HANDLER as u32, 0x8e);
    h.write(HANDLER, &handler);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x90 * 8 + 2);
}

#[test]
fn software_interrupt_through_a_gate_with_lower_dpl_is_refused_in_ring_three() {
    let code = [0xcd, 0x30, 0xf4];
    let handler = [0x5b, 0xf4];
    let mut h = prot(&code);
    set_idt_gate(&mut h, 0x30, LEGACY_CS, HANDLER as u32, 0x8e);
    set_idt_gate(&mut h, 13, LEGACY_CS, HANDLER as u32, 0x8e);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    h.write(HANDLER, &handler);
    h.write(RING3_CODE, &code);
    enter_ring3(&mut h, RING3_STACK);
    h.cpu.rip = RING3_CODE;
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x30 * 8 + 2);
}

#[test]
fn interrupt_from_ring_three_switches_to_the_tss_stack_and_iretd_returns() {
    let code = [
        0xb8, 0x01, 0x00, 0x00, 0x00, 0xcd, 0x30, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xcd, 0x32,
    ];
    let handler = [0x8b, 0x5c, 0x24, 0x0c, 0x89, 0xe1, 0xcf];
    let mut h = prot(&[0xf4]);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    set_idt_gate(&mut h, 0x30, LEGACY_CS, HANDLER as u32, 0xee);
    set_idt_gate(&mut h, 0x32, LEGACY_CS, HANDLER as u32 + 0x100, 0xee);
    h.write(HANDLER, &handler);
    h.write(HANDLER + 0x100, &[0xf4]);
    h.write(RING3_CODE, &code);
    enter_ring3(&mut h, RING3_STACK);
    h.cpu.rip = RING3_CODE;
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RAX], 2);
    assert_eq!(
        h.cpu.gpr[RBX], RING3_STACK as u32 as u64,
        "the handler saw the user ESP in the frame"
    );
    assert_eq!(
        h.cpu.gpr[RCX],
        KERNEL_STACK - 20,
        "frame of 5 dwords on the kernel stack"
    );
    assert_eq!(h.cpu.cpl, 0, "stopped in the ring 0 handler of int 0x32");
}

#[test]
fn v86_task_enters_through_iretd_and_interrupts_back_to_ring_zero() {
    let code = [0xf4];
    let handler = [0x8b, 0x5c, 0x24, 0x10, 0x8b, 0x54, 0x24, 0x20, 0xf4];
    let mut h = prot(&code);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    set_idt_gate(&mut h, 0x21, LEGACY_CS, HANDLER as u32, 0xee);
    h.write(HANDLER, &handler);
    // v86 code at 0x2000:0x0000 = 0x20000: int 0x21; hlt
    h.write(0x20000, &[0xcd, 0x21, 0xf4]);
    // iretd frame: eip, cs, eflags, esp, ss, es, ds, fs, gs
    let frame: [u32; 9] = [
        0,
        0x2000,
        (flags::VM as u32) | 0x2 | 0x3000,
        0x1000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
        0x7000,
    ];
    let mut sp = 0x50_0000u64;
    h.cpu.gpr[RSP] = sp;
    for word in frame {
        h.write(sp, &word.to_le_bytes());
        sp += 4;
    }
    h.cpu.eflags |= 0x3000;
    h.write(CODE_BASE, &[0xcf]);
    h.cpu.rip = CODE_BASE;
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x3000, "SS of the interrupted v86 frame");
    assert_eq!(h.cpu.gpr[RDX], 0x7000, "GS was pushed first");
}

#[test]
fn sysenter_and_sysexit_switch_between_ring_zero_and_three() {
    let code = [0xf4];
    let handler = [0xf4];
    let user = [0x0f, 0x34];
    let mut h = prot(&code);
    h.cpu.sysenter = [LEGACY_CS as u64, KERNEL_STACK, HANDLER];
    h.write(HANDLER, &handler);
    h.write(RING3_CODE, &user);
    // sysexit selectors: CS+16 and CS+24 are the ring 3 code/data descriptors
    set_descriptor(
        &mut h,
        0x18,
        super::test_support::descriptor(0, 0xfffff, 0xfa, 0xc),
    );
    set_descriptor(
        &mut h,
        0x20,
        super::test_support::descriptor(0, 0xfffff, 0xf2, 0xc),
    );
    h.write(CODE_BASE, &[0xf4]);
    enter_ring3(&mut h, RING3_STACK);
    h.cpu.rip = RING3_CODE;
    let stop = h.run_delivering(100);
    assert_eq!(stop, Stop::Halted, "{}", h.cpu.dump_regs());
    assert_eq!(h.cpu.cpl, 0);
    assert_eq!(h.cpu.gpr[RSP], KERNEL_STACK);
}

#[test]
fn sgdt_with_a_sixteen_bit_operand_stores_a_24_bit_base() {
    let code = [
        0xbb, 0x00, 0x00, 0x63, 0x00, 0x66, 0x0f, 0x01, 0x03, 0x0f, 0x01, 0x43, 0x10, 0xf4,
    ];
    let mut h = prot(&code);
    halted(&mut h);
    let limit = u16::from_le_bytes([h.read(DATA, 1)[0], h.read(DATA + 1, 1)[0]]);
    assert_eq!(limit, h.cpu.gdtr.limit);
    assert_eq!(dword(&h, DATA + 2), (h.cpu.gdtr.base & 0x00ff_ffff) as u32);
    assert_eq!(dword(&h, DATA + 18), h.cpu.gdtr.base as u32);
}

#[test]
fn real_mode_boot_code_reaches_flat_protected_mode() {
    let code = [
        0xfa, 0x0f, 0x01, 0x16, 0x00, 0x7e, 0x0f, 0x20, 0xc0, 0x66, 0x83, 0xc8, 0x01, 0x0f, 0x22,
        0xc0, 0x66, 0xea, 0x00, 0x7d, 0x00, 0x00, 0x08, 0x00,
    ];
    let entry32 = [
        0x66, 0xb8, 0x10, 0x00, 0x66, 0x8e, 0xd8, 0x66, 0x8e, 0xd0, 0xbc, 0x00, 0x90, 0x00, 0x00,
        0xbb, 0x34, 0x12, 0x00, 0x00, 0xf4,
    ];
    let mut h = Harness::real(&code);
    install_legacy_gdt(&mut h);
    h.cpu.gdtr = DescriptorTable::default();
    let mut pointer = (h.cpu.gdtr.limit).to_le_bytes().to_vec();
    pointer.clear();
    pointer.extend_from_slice(&0x6fu16.to_le_bytes());
    pointer.extend_from_slice(&0x6000u32.to_le_bytes());
    h.write(0x7e00, &pointer);
    h.write(0x7d00, &entry32);
    halted(&mut h);
    assert_ne!(h.cpu.cr0 & CR0_PE, 0);
    assert_eq!(h.cpu.seg[SEG_CS].sel, LEGACY_CS);
    assert_eq!(h.cpu.code_mode(), CodeMode::Bits32);
    assert_eq!(h.cpu.gpr[RBX], 0x1234);
    assert_eq!(h.cpu.seg[SEG_DS].sel, LEGACY_DS);
}

#[test]
fn protected_mode_boot_code_enters_long_mode_through_a_far_jump() {
    let code = [
        0x0f, 0x20, 0xe0, 0x83, 0xc8, 0x20, 0x0f, 0x22, 0xe0, 0xb8, 0x00, 0x40, 0x00, 0x00, 0x0f,
        0x22, 0xd8, 0xb9, 0x80, 0x00, 0x00, 0xc0, 0x0f, 0x32, 0x0d, 0x00, 0x01, 0x00, 0x00, 0x0f,
        0x30, 0x0f, 0x20, 0xc0, 0x0d, 0x00, 0x00, 0x00, 0x80, 0x0f, 0x22, 0xc0, 0xea, 0x00, 0x7d,
        0x00, 0x00, 0x60, 0x00,
    ];
    let entry64 = [
        0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xf4,
    ];
    let mut h = prot(&code);
    // the harness' boot identity map is the page table to load
    let boot_cr3 = {
        let reference = Harness::new(&[HLT]);
        reference.cpu.cr3
    };
    let code = {
        let mut patched = code.to_vec();
        let position = patched
            .windows(4)
            .position(|window| window == [0x00, 0x40, 0x00, 0x00])
            .unwrap();
        patched[position..position + 4].copy_from_slice(&(boot_cr3 as u32).to_le_bytes());
        patched
    };
    h.write(CODE_BASE, &code);
    h.write(0x7d00, &entry64);
    halted(&mut h);
    assert_ne!(h.cpu.efer & EFER_LMA, 0);
    assert_eq!(h.cpu.code_mode(), CodeMode::Long64);
    assert_eq!(h.cpu.gpr[RAX], 0x1122_3344_5566_7788);
}

#[test]
fn compatibility_mode_runs_thirty_two_bit_code_under_long_mode() {
    let code = [0x40, 0x50, 0x5b, 0xea, 0x00, 0x7d, 0x00, 0x00, 0x60, 0x00];
    let entry64 = [0x48, 0xc7, 0xc1, 0x66, 0x55, 0x00, 0x00, 0xf4];
    let mut h = Harness::new(&code);
    h.cpu.gpr[RAX] = 0xffff_ffff;
    // the boot GDT has no compatibility segment, so install one
    set_descriptor(
        &mut h,
        LEGACY_COMPAT_CS,
        super::test_support::descriptor(0, 0xfffff, 0x9a, 0xc),
    );
    set_descriptor(
        &mut h,
        LEGACY_LONG_CS,
        super::test_support::descriptor(0, 0xfffff, 0x9a, 0xa),
    );
    h.cpu.gdtr = DescriptorTable {
        base: super::test_support::GDT_BASE,
        limit: 0x6f,
    };
    h.cpu.seg[SEG_CS] = seg_from_descriptor(
        LEGACY_COMPAT_CS,
        super::test_support::descriptor(0, 0xfffff, 0x9a, 0xc),
    );
    h.write(0x7d00, &entry64);
    assert_eq!(h.cpu.code_mode(), CodeMode::Bits32);
    halted(&mut h);
    assert_eq!(
        h.cpu.gpr[RAX], 0,
        "32-bit inc wrapped and cleared the upper half"
    );
    assert_eq!(h.cpu.gpr[RBX], 0, "pushed and popped a dword");
    assert_eq!(
        h.cpu.code_mode(),
        CodeMode::Long64,
        "the far jump to a long-mode segment re-entered 64-bit mode"
    );
    assert_eq!(h.cpu.gpr[RCX], 0x5566);
}

#[test]
fn call_gate_from_ring_three_switches_stacks_and_copies_parameters() {
    let code = [
        0x6a, 0x55, 0x9a, 0x00, 0x00, 0x00, 0x00, 0x43, 0x00, 0xcd, 0x32,
    ];
    let handler = [
        0x8b, 0x5c, 0x24, 0x08, 0x8b, 0x4c, 0x24, 0x0c, 0xca, 0x04, 0x00,
    ];
    let mut h = prot(&[0xf4]);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    set_descriptor(
        &mut h,
        0x40,
        gate_descriptor(LEGACY_CS, HANDLER as u32, 0xec, 1),
    );
    set_idt_gate(&mut h, 0x32, LEGACY_CS, HANDLER as u32 + 0x100, 0xee);
    h.write(HANDLER, &handler);
    h.write(HANDLER + 0x100, &[0xf4]);
    h.write(RING3_CODE, &code);
    enter_ring3(&mut h, RING3_STACK);
    h.cpu.rip = RING3_CODE;
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x55, "the copied parameter");
    assert_eq!(
        h.cpu.gpr[RCX],
        RING3_STACK - 4,
        "the old ESP saved by the gate"
    );
}

#[test]
fn jump_to_a_tss_switches_tasks() {
    let code = [
        0xb8, 0x07, 0x00, 0x00, 0x00, 0xea, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00,
    ];
    let mut h = prot(&code);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    let mut tss = [0u8; 0x68];
    tss[32..36].copy_from_slice(&(HANDLER as u32).to_le_bytes());
    tss[36..40].copy_from_slice(&2u32.to_le_bytes());
    tss[56..60].copy_from_slice(&(KERNEL_STACK as u32).to_le_bytes());
    tss[72..76].copy_from_slice(&(LEGACY_DS as u32).to_le_bytes());
    tss[76..80].copy_from_slice(&(LEGACY_CS as u32).to_le_bytes());
    tss[80..84].copy_from_slice(&(LEGACY_DS as u32).to_le_bytes());
    tss[84..88].copy_from_slice(&(LEGACY_DS as u32).to_le_bytes());
    h.write(LEGACY_SECOND_TSS_BASE, &tss);
    h.write(HANDLER, &[0xb8, 0x99, 0, 0, 0, 0xf4]);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RAX], 0x99);
    assert_eq!(h.cpu.tr.sel, LEGACY_SECOND_TSS);
    assert_eq!(
        dword(&h, LEGACY_TSS_BASE + 40),
        7,
        "the old task's EAX was saved"
    );
    assert_eq!(
        dword(&h, LEGACY_TSS_BASE + 32),
        CODE_BASE as u32 + code.len() as u32,
        "and the EIP after the jump"
    );
    assert_ne!(
        h.cpu.cr0 & super::sys::CR0_TS,
        0,
        "TS is set by a task switch"
    );
}

#[test]
fn call_to_a_tss_links_back_and_iret_with_nt_returns() {
    let code = [
        0xb8, 0x07, 0x00, 0x00, 0x00, 0x9a, 0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x8b, 0x1d, 0x00,
        0x00, 0x63, 0x00, 0xf4,
    ];
    let handler = [
        0xc7, 0x05, 0x00, 0x00, 0x63, 0x00, 0x99, 0x00, 0x00, 0x00, 0xcf,
    ];
    let mut h = prot(&code);
    install_legacy_tss(&mut h, KERNEL_STACK as u32, LEGACY_DS);
    let mut tss = [0u8; 0x68];
    tss[32..36].copy_from_slice(&(HANDLER as u32).to_le_bytes());
    tss[36..40].copy_from_slice(&2u32.to_le_bytes());
    tss[56..60].copy_from_slice(&(KERNEL_STACK as u32).to_le_bytes());
    for offset in [72, 80, 84] {
        tss[offset..offset + 4].copy_from_slice(&(LEGACY_DS as u32).to_le_bytes());
    }
    tss[76..80].copy_from_slice(&(LEGACY_CS as u32).to_le_bytes());
    h.write(LEGACY_SECOND_TSS_BASE, &tss);
    h.write(HANDLER, &handler);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x99);
    assert_eq!(h.cpu.gpr[RAX], 7, "the first task got its registers back");
    assert_eq!(h.cpu.tr.sel, LEGACY_TSS);
    assert_eq!(h.cpu.eflags_full() & flags::NT, 0);
    assert_eq!(
        dword(&h, LEGACY_SECOND_TSS_BASE + 32),
        (HANDLER + handler.len() as u64) as u32,
        "the task iret left resumes after the iret, not on it"
    );
}

#[test]
fn ldt_descriptors_can_be_loaded_into_segment_registers() {
    let code = [
        0x66, 0xb8, 0x58, 0x00, 0x0f, 0x00, 0xd0, 0x66, 0xb8, 0x04, 0x00, 0x66, 0x8e, 0xd8, 0x8b,
        0x1d, 0x10, 0x00, 0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    h.write(
        LEGACY_LDT_BASE,
        &super::test_support::descriptor(DATA as u32, 0xfffff, 0x92, 0xc).to_le_bytes(),
    );
    h.write(DATA + 0x10, &0xcafe_f00du32.to_le_bytes());
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0xcafe_f00d);
    assert_eq!(h.cpu.ldtr.base, LEGACY_LDT_BASE);
}

#[test]
fn sixteen_bit_code_segment_runs_under_protected_mode() {
    let code = [0xea, 0x00, 0x70, 0x00, 0x00, 0x38, 0x00];
    let code16 = [0xb8, 0x34, 0x12, 0x50, 0x5b, 0xf4];
    let mut h = prot(&code);
    h.write(0x7000, &code16);
    halted(&mut h);
    assert_eq!(h.cpu.code_mode(), CodeMode::Bits16);
    assert_eq!(h.cpu.gpr[RAX] & 0xffff, 0x1234);
    assert_eq!(
        h.cpu.gpr[RSP] & 0xffff_0000,
        crate::testkit::STACK_TOP & 0xffff_0000,
        "push ax only moved SP"
    );
}

#[test]
fn thirty_two_bit_string_instructions_in_protected_mode() {
    let code = [
        0xbe, 0x00, 0x00, 0x63, 0x00, 0xbf, 0x00, 0x01, 0x63, 0x00, 0xb9, 0x03, 0x00, 0x00, 0x00,
        0xf3, 0xa5, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xab, 0xf4,
    ];
    let mut h = prot(&code);
    h.write(DATA, &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    halted(&mut h);
    assert_eq!(
        h.read(DATA + 0x100, 12),
        [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
    );
    assert_eq!(dword(&h, DATA + 0x10c), 5);
    assert_eq!(h.cpu.gpr[RCX], 0);
}

#[test]
fn bound_faults_when_the_index_is_out_of_range() {
    let code = [
        0xb8, 0x0a, 0x00, 0x00, 0x00, 0xbb, 0x00, 0x00, 0x63, 0x00, 0x62, 0x03, 0xf4,
    ];
    let mut h = prot(&code);
    h.write(DATA, &[0, 0, 0, 0, 5, 0, 0, 0]);
    assert_eq!(prot_fault(&mut h), (5, None));
}

#[test]
fn arpl_raises_the_rpl_of_a_selector() {
    let code = [
        0x66, 0xb8, 0x10, 0x00, 0x66, 0xbb, 0x03, 0x00, 0x63, 0xd8, 0x9c, 0x59, 0xf4,
    ];
    let mut h = prot(&code);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RAX] & 0xffff, 0x13);
    assert_ne!(h.cpu.gpr[RCX] & flags::ZF, 0);
}

#[test]
fn fnstenv_with_a_sixteen_bit_operand_size_writes_a_14_byte_image() {
    let code = [0xbb, 0x00, 0x03, 0xd9, 0xee, 0xd9, 0x37, 0xf4];
    let h = real(&code);
    assert_eq!(h.read(0x300, 2), 0x37fu16.to_le_bytes());
    assert_eq!(
        h.read(0x304, 2),
        0x7fffu16.to_le_bytes(),
        "tag word: register 7 holds zero"
    );
    assert_eq!(h.read(0x30e, 2), [0, 0], "nothing past the 14-byte image");
}

#[test]
fn mov_to_and_from_control_registers_are_thirty_two_bits_wide_outside_long_mode() {
    let code = [0x0f, 0x20, 0xc0, 0x89, 0xc3, 0x0f, 0x20, 0xe0, 0xf4];
    let mut h = prot(&code);
    h.cpu.gpr[RAX] = u64::MAX;
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], CR0_PE | CR0_ET);
    assert_eq!(h.cpu.gpr[RAX], 0);
}

#[test]
fn lar_and_lsl_read_access_rights_and_limits() {
    let code = [
        0x66, 0xb8, 0x10, 0x00, 0x0f, 0x03, 0xd8, 0x0f, 0x02, 0xc8, 0x9c, 0x5a, 0xf4,
    ];
    let mut h = prot(&code);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0xffff_ffff);
    assert_eq!(h.cpu.gpr[RCX], 0x00cf_9200);
    assert_ne!(h.cpu.gpr[RDX] & flags::ZF, 0);
}

#[test]
fn lar_with_an_unusable_selector_clears_zf_and_keeps_the_destination() {
    let code = [
        0xbb, 0x34, 0x12, 0x00, 0x00, 0x66, 0xb8, 0x38, 0x12, 0x0f, 0x02, 0xd8, 0x9c, 0x5a, 0xf4,
    ];
    let mut h = prot(&code);
    halted(&mut h);
    assert_eq!(h.cpu.gpr[RBX], 0x1234);
    assert_eq!(h.cpu.gpr[RDX] & flags::ZF, 0);
}

#[test]
fn lar_is_undefined_in_real_mode() {
    let code = [0x0f, 0x02, 0xc3, 0xf4];
    assert_eq!(real_fault(&code), (6, None));
}

#[test]
fn compatibility_mode_system_calls_enter_at_cstar_and_sysretl_returns() {
    let code = [
        0xbb, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x05, 0x43, 0x0f, 0x05,
        0xf4,
    ];
    let handler = [0x83, 0xfb, 0x07, 0x74, 0x02, 0x0f, 0x07, 0xf4];
    let mut h = Harness::new(&code);
    h.cpu.efer |= 1;
    h.cpu.star = (0x1bu64 << 48) | (0x08u64 << 32);
    h.cpu.cstar = HANDLER;
    h.cpu.seg[SEG_CS] = flat_segment(0x1b, dpl(0x9a, 3), 0xc);
    h.cpu.set_cpl(3);
    make_identity_map_user_accessible(&mut h);
    h.write(HANDLER, &handler);
    assert_eq!(h.cpu.code_mode(), CodeMode::Bits32);
    halted(&mut h);
    assert_eq!(
        h.cpu.gpr[RBX], 7,
        "inc ebx ran as 32-bit code after sysretl"
    );
    assert_eq!(
        h.cpu.cpl, 0,
        "the second syscall stopped in the 64-bit handler"
    );
    assert_eq!(h.cpu.code_mode(), CodeMode::Long64);
}

#[test]
fn an_access_that_straddles_the_segment_limit_faults_but_one_that_fits_does_not() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0x66, 0xa1, 0xfe, 0x0f, 0x00, 0x00, 0x8b, 0x1d,
        0xfe, 0x0f, 0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32, 0xfff, 0x92, 0x4),
    );
    h.write(DATA + 0xffe, &[0x34, 0x12]);
    assert_eq!(prot_fault(&mut h), (13, Some(0)));
    assert_eq!(
        h.cpu.gpr[RAX] & 0xffff,
        0x1234,
        "the 16-bit load at the limit worked"
    );
    assert_eq!(h.cpu.gpr[RBX], 0);
}

#[test]
fn storing_into_a_read_only_data_segment_is_a_general_protection_fault() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0xa1, 0x10, 0x00, 0x00, 0x00, 0xa3, 0x20, 0x00,
        0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32, 0xfffff, 0x90, 0xc),
    );
    h.write(DATA + 0x10, &0xabcdu32.to_le_bytes());
    assert_eq!(prot_fault(&mut h), (13, Some(0)));
    assert_eq!(h.cpu.gpr[RAX], 0xabcd, "reading it is fine");
    assert_eq!(dword(&h, DATA + 0x20), 0);
}

#[test]
fn writing_through_a_cs_override_is_refused_in_protected_mode() {
    let code = [0x2e, 0xa3, 0x10, 0x00, 0x00, 0x00, 0xf4];
    let mut h = prot(&code);
    assert_eq!(prot_fault(&mut h), (13, Some(0)));
}

#[test]
fn an_expand_down_segment_accepts_only_offsets_above_the_limit() {
    let code = [
        0x66, 0xb8, 0x38, 0x00, 0x66, 0x8e, 0xd8, 0xa1, 0x00, 0x20, 0x00, 0x00, 0x8b, 0x1d, 0x00,
        0x10, 0x00, 0x00, 0xf4,
    ];
    let mut h = prot(&code);
    set_descriptor(
        &mut h,
        0x38,
        super::test_support::descriptor(DATA as u32 - 0x2000, 0x1fff, 0x96, 0x4),
    );
    h.write(DATA, &0xfeedu32.to_le_bytes());
    assert_eq!(prot_fault(&mut h), (13, Some(0)));
    assert_eq!(
        h.cpu.gpr[RAX], 0xfeed,
        "[0x2000] is above the limit, so it is valid"
    );
    assert_eq!(h.cpu.gpr[RBX], 0, "[0x1000] is not");
}
