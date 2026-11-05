use super::blocks::JUMP_CACHE_BITS;
pub(crate) use super::emit_x64::Addrs;
use super::emit_x64::{CPU, FRAME, Gen, JUMP_CACHE_SLOT, PAGE_BASE_SLOT, TLB_BASE, VALID_SLOT};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use super::native::{
    CompiledBlock, JUMP_ENTRY_BLOCK_OFFSET, JUMP_ENTRY_CODE_OFFSET, JUMP_ENTRY_PHYS_OFFSET,
    JUMP_ENTRY_SHIFT, jump_cache_mask,
};
use super::x64::*;
use crate::cpu::decode::Insn;
use crate::cpu::{Cpu, TLB_ENTRIES_PER_MODE, TlbEntry};
use std::mem::{offset_of, size_of};

/// Entry/exit stubs placed at the start of the code buffer, shared by all blocks.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    /// Absolute address of the success exit (rax already holds the exit code).
    pub exit_ok: u64,
    /// Absolute address of the failure exit (rax = helper status, rcx = failing instruction index).
    pub exit_fail: u64,
    /// Block-to-block transfer stub; see `chain_stub`.
    pub chain: u64,
}

/// Pushed in this order and popped in reverse. This is the Win64 callee-saved set (RSI/RDI are
/// callee-saved there and blocks use them as scratch); on System V the extra saves are harmless.
const SAVED_REGS: [u8; 8] = [RBX, RBP, RSI, RDI, R12, R13, R14, R15];

/// Builds `enter(cpu: *mut Cpu, code: u64, jump_cache: u64, valid: u64) -> exit_code`, which `Cpu::step_block` calls through a
/// function pointer: it sets up the frame, pins the `Cpu` pointer and TLB base for the block (see
/// `emit_x64`), then jumps to `code`. Blocks never return themselves; they jump to `exit_ok` or
/// `exit_fail` below.
pub(crate) fn prologue(base: u64) -> Prologue {
    let mut a = Asm::new(base);
    for r in SAVED_REGS {
        a.push(r);
    }
    a.alu_ri(SUB, RSP, FRAME, true);
    a.mov_rr(CPU, ARGS[0], true);
    a.st(mem(RSP, JUMP_CACHE_SLOT), ARGS[2], 8);
    a.st(mem(RSP, VALID_SLOT), ARGS[3], 8);
    a.ld(TLB_BASE, mem(CPU, offset_of!(Cpu, tlb_active_base) as i32));
    // The run loop enters at `Cpu::rip`, so that is the page the block runs from. rax is neither
    // an argument register nor needed by the block.
    a.ld(RAX, mem(CPU, offset_of!(Cpu, rip) as i32));
    a.alu_ri(AND, RAX, PAGE_BASE_MASK, true);
    a.st(mem(RSP, PAGE_BASE_SLOT), RAX, 8);
    a.jmp_reg(ARGS[1]);
    // rax = helper status, rcx = index of the failing instruction. Fold into the exit code
    // `(retired << EXIT_COUNT_SHIFT) | status`: an exception means rcx instructions retired, but
    // STATUS_EXIT_REQUESTED means the instruction itself retired too. The xor runs before the cmp
    // because it clobbers flags and sete only writes the low byte.
    let exit_fail = a.here();
    a.alu_rr(XOR, RDX, RDX, false);
    a.alu_ri(CMP, RAX, STATUS_EXIT_REQUESTED as i32, true);
    a.sete(RDX);
    a.alu_rr(ADD, RCX, RDX, true);
    a.shift_ri(SHL, RCX, EXIT_COUNT_SHIFT as u8, true);
    a.alu_rr(OR, RAX, RCX, true);
    let exit_ok = a.here();
    a.alu_ri(ADD, RSP, FRAME, true);
    for r in SAVED_REGS.iter().rev() {
        a.pop(*r);
    }
    a.ret();
    let chain = chain_stub(&mut a, exit_ok);
    a.finish();
    Prologue {
        code: a.code,
        exit_ok,
        exit_fail,
        chain,
    }
}

const PAGE_SHIFT: u8 = 12;
/// Sign-extends to `!0xfff` in a 64-bit `and`.
const PAGE_BASE_MASK: i32 = !((1 << PAGE_SHIFT) - 1);

/// Registers of the chain stub's calling convention (see `Gen::chain_exit`). The scratch
/// registers are caller-saved on both ABIs and hold nothing a block needs across the jump;
/// rbx and rbp (`CPU`, `TLB_BASE`) are read, never written.
const CHAIN_ICOUNT: u8 = RAX;
const CHAIN_RIP: u8 = RCX;

/// Block-to-block transfer, the equivalent of QEMU's `lookup_and_goto_ptr`. Entered by a `jmp`
/// with `rax` = `Cpu::icount` as just stored by the block (already charged for the first
/// instruction of the successor, as `Cpu::run` does before `step_block`) and `rcx` = the guest
/// rip to continue at. The stack is as the prologue left it, so a successor block starts exactly
/// like one entered from `enter`.
///
/// It jumps straight into the successor block when all of these hold, and otherwise leaves for
/// the run loop with `Cpu::rip` set, which then takes the normal path (event servicing,
/// interrupts, page walk, translation):
/// - `icount` has not reached `Cpu::next_event`, so the device tick and interrupt check the loop
///   does at block boundaries are not starved;
/// - `block_exit_requested` is clear: a store of the block just left (for example the push of a
///   `call`) hit a page holding translated code, and the successor may be stale;
/// - the exec TLB (the half `TLB_BASE` selects, which a block never leaves) has the page, so the
///   physical address is known without a walk and permissions were already checked;
/// - the jump cache has a block for exactly this rip and physical address, the same key as
///   `Jit::lookup`, and the block was not invalidated since (`Jit::valid`).
///
/// The miss path gives back the successor's pre-charge: the loop adds it again. It reports one
/// retired instruction in the exit code, which `step_block` counts as nothing further (it
/// subtracts the first instruction, which `Cpu::run` already counted).
fn chain_stub(a: &mut Asm, exit_ok: u64) -> u64 {
    const TLB_SLOT_MASK: i32 = (TLB_ENTRIES_PER_MODE - 1) as i32;
    const TLB_ENTRY_SHIFT: u8 = size_of::<TlbEntry>().trailing_zeros() as u8;
    // `jump_cache_hash` in `native.rs`: rip ^ (rip >> bits), masked.
    const JUMP_HASH_FOLD_SHIFT: u8 = JUMP_CACHE_BITS as u8;
    let (entry, page, phys, slot, key) = (RDX, RSI, RDX, RSI, RDI);
    let miss = a.label();
    let start = a.here();
    a.ld(entry, mem(CPU, offset_of!(Cpu, next_event) as i32));
    a.alu_rr(CMP, CHAIN_ICOUNT, entry, true);
    a.jcc(AE, miss);
    a.cmp_m8_imm(mem(CPU, offset_of!(Cpu, block_exit_requested) as i32), 0);
    a.jcc(NE, miss);
    a.mov_rr(entry, CHAIN_RIP, true);
    a.shift_ri(SHR, entry, PAGE_SHIFT, true);
    a.alu_ri(AND, entry, TLB_SLOT_MASK, true);
    a.shift_ri(SHL, entry, TLB_ENTRY_SHIFT, true);
    a.alu_rr(ADD, entry, TLB_BASE, true);
    a.mov_rr(page, CHAIN_RIP, true);
    a.alu_ri(AND, page, PAGE_BASE_MASK, true);
    // The successor is entered with its own page as the page base. Stored before the checks (a
    // miss leaves the block anyway, and the next entry sets it again).
    a.st(mem(RSP, PAGE_BASE_SLOT), page, 8);
    a.cmp_mr(mem(entry, offset_of!(TlbEntry, tag_x) as i32), page);
    a.jcc(NE, miss);
    // phys = rip + addend - ram_base
    a.ld(phys, mem(entry, offset_of!(TlbEntry, addend) as i32));
    a.alu_rr(ADD, phys, CHAIN_RIP, true);
    a.ld(key, mem(CPU, offset_of!(Cpu, ram_base) as i32));
    a.alu_rr(SUB, phys, key, true);
    a.mov_rr(slot, CHAIN_RIP, true);
    a.shift_ri(SHR, slot, JUMP_HASH_FOLD_SHIFT, true);
    a.alu_rr(XOR, slot, CHAIN_RIP, true);
    a.alu_ri(AND, slot, jump_cache_mask() as i32, false);
    a.shift_ri(SHL, slot, JUMP_ENTRY_SHIFT as u8, true);
    a.ld(key, mem(RSP, JUMP_CACHE_SLOT));
    a.alu_rr(ADD, slot, key, true);
    a.cmp_mr(mem(slot, 0), CHAIN_RIP);
    a.jcc(NE, miss);
    a.cmp_mr(mem(slot, JUMP_ENTRY_PHYS_OFFSET as i32), phys);
    a.jcc(NE, miss);
    a.ld(key, mem(slot, JUMP_ENTRY_CODE_OFFSET as i32));
    a.test_rr(key, key, true);
    a.jcc(E, miss);
    // The slot may name a block whose page was written since (`Jit::valid`). r8 and r9 are
    // caller-saved scratch the blocks keep nothing in.
    a.ld(R8, mem(slot, JUMP_ENTRY_BLOCK_OFFSET as i32));
    a.ld(R9, mem(RSP, VALID_SLOT));
    a.cmp_m8_imm(mem_idx(R9, R8, 0, 0), 0);
    a.jcc(E, miss);
    a.jmp_reg(key);
    a.bind(miss);
    a.alu_ri(SUB, CHAIN_ICOUNT, 1, true);
    a.st(mem(CPU, offset_of!(Cpu, icount) as i32), CHAIN_ICOUNT, 8);
    a.st(mem(CPU, offset_of!(Cpu, rip) as i32), CHAIN_RIP, 8);
    a.mov_imm(RAX, 1 << EXIT_COUNT_SHIFT);
    a.jmp_abs(exit_ok);
    start
}

pub(crate) fn addrs(prologue: &Prologue, chaining: bool) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u64,
        ld: helpers::helper_load_via_out as *const () as usize as u64,
        st: helpers::helper_store_x64 as *const () as usize as u64,
        ld_stack: helpers::helper_load_stack_via_out as *const () as usize as u64,
        st_stack: helpers::helper_store_stack_x64 as *const () as usize as u64,
        cond: helpers::helper_cond as *const () as usize as u64,
        whole_vector: helpers::helper_whole_vector as *const () as usize as u64,
        exit_ok: prologue.exit_ok,
        exit_fail: prologue.exit_fail,
        chain: if chaining { prologue.chain } else { 0 },
    }
}

/// Generates host code for one block placed at `addr`, which must be its final address
/// (`jmp_abs` encodes displacements relative to it). Returns the code and the number of
/// instructions lowered inline. `insns` must outlive the code, which points into it.
pub(crate) fn compile_block(
    addr: u64,
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
    ordered: bool,
) -> CompiledBlock {
    let mut generator = Gen::new(addr, addrs, insns);
    let _ = ordered;
    generator.compile(rips, end_rip);
    CompiledBlock {
        code: generator.a.code,
        inlined: generator.inlined,
        body_offset: generator.body_offset,
        links: generator.links,
        flag_records_skipped: generator.flag_records_skipped,
    }
}

const JMP_REL32_BYTES: i64 = 5;
/// A link slot is `jmp rel32` (`emit_x64::LINK_JMP`); only the displacement after the opcode is
/// rewritten, as one aligned 32-bit store.
pub(crate) const LINK_WORD_OFFSET: usize = 1;

/// The displacement that makes the slot at `site` (its opcode byte) jump to `destination`, or
/// `None` when it does not fit (the buffer is far smaller than 2 GiB, so only a corrupt address).
pub(crate) fn link_word(site: u64, destination: u64) -> Option<u32> {
    let rel = i32::try_from(destination as i64 - (site as i64 + JMP_REL32_BYTES)).ok()?;
    Some(rel as u32)
}

/// The displacement of an unlinked slot: zero, a jump to the next instruction, which is the chain
/// stub sequence.
pub(crate) fn unlink_word() -> u32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: u64 = 0x7000_0000_0000;

    #[test]
    fn a_link_to_the_next_instruction_after_the_jump_is_displacement_zero() {
        assert_eq!(link_word(SITE, SITE + 5), Some(0));
        assert_eq!(
            unlink_word(),
            0,
            "unlinked is the same jump to the same place"
        );
    }

    #[test]
    fn displacements_count_from_the_end_of_the_five_byte_jump() {
        assert_eq!(link_word(SITE, SITE + 5 + 0x100), Some(0x100));
        assert_eq!(link_word(SITE, SITE), Some((-5i32) as u32));
        assert_eq!(link_word(SITE, SITE - 0x100), Some((-0x105i32) as u32));
    }

    #[test]
    fn links_reach_two_gibibytes_either_way_and_no_further() {
        assert!(link_word(SITE, SITE + 5 + i32::MAX as u64).is_some());
        assert_eq!(link_word(SITE, SITE + 6 + i32::MAX as u64), None);
        assert!(link_word(SITE, SITE + 5 - (1 << 31)).is_some());
        assert_eq!(link_word(SITE, SITE + 4 - (1 << 31)), None);
    }

    #[test]
    fn the_patched_word_is_the_displacement_after_the_opcode_byte() {
        assert_eq!(LINK_WORD_OFFSET, 1);
        assert_eq!(
            super::super::emit_x64::LINK_JMP[0],
            JMP_REL32_OPCODE_FOR_TEST
        );
    }

    const JMP_REL32_OPCODE_FOR_TEST: u8 = 0xe9;
}
