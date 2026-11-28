//! aarch64 glue for `native.rs`: the entry/exit trampoline, helper addresses and the block
//! compiler entry point. The x86_64 host has the same three functions in `arch_x64.rs`.

use std::mem::offset_of;

use super::a64::{self, Asm, NOP_WORD};
use super::blocks::JUMP_CACHE_BITS;
use super::emit::{
    CALL_TARGET, CPU, Gen, PAGE_BASE, PINNED_GUEST_COUNT, PINNED_HOST_FIRST, SAVED_ADDRESS,
    SAVED_INFO, SAVED_LR, SAVED_NZCV, SAVED_OPERAND, SAVED_RESULT, TLB_BASE, VALUE,
};
use super::helpers::{self, EXIT_COUNT_SHIFT, STATUS_EXIT_REQUESTED};
use super::native::{
    CompiledBlock, JUMP_ENTRY_BLOCK_OFFSET, JUMP_ENTRY_CODE_OFFSET, JUMP_ENTRY_PHYS_OFFSET,
    JUMP_ENTRY_SHIFT, ResolveTarget, jump_cache_mask,
};
use crate::cpu::decode::Insn;
use crate::cpu::{Cpu, TLB_ENTRIES_PER_MODE, TlbEntry};

pub(crate) use super::emit::Addrs;

/// The trampoline placed at the start of the code buffer, followed by the stubs and thunks
/// every block reaches with a direct branch. See the module docs of `emit.rs`.
pub(crate) struct Prologue {
    pub code: Vec<u8>,
    addrs: Addrs,
}

/// A prologue stub or memory thunk a block may call (`Gen::call`). Only this file makes them, and
/// every one stores the pinned guest registers to `Cpu::gpr` before its helper runs and loads
/// them back after (`spill_pinned`, `reload_pinned`): a helper reads and writes guest registers
/// through `Cpu`, and the Rust call clobbers x8..x15. A helper reached any other way would see
/// stale registers and lose the guest's, which is why `Gen::call` takes this and not an address.
#[derive(Clone, Copy)]
pub(crate) struct Stub(u64);

impl Stub {
    /// For generator tests that never call a stub.
    #[cfg(test)]
    pub(super) const DUMMY: Stub = Stub(0);

    pub(super) fn address(self) -> u64 {
        self.0
    }
}

/// `Cpu::gpr` is the first field, so `stp`/`ldp` (offsets up to 504) reach the pinned slots.
const _: () = assert!(offset_of!(Cpu, gpr) + 8 * PINNED_GUEST_COUNT as usize <= 504);
const GPR_PAIR_BYTES: i32 = 16;

/// Stores the pinned guest registers (`emit::PINNED_HOST_FIRST`) to `Cpu::gpr`.
fn spill_pinned(asm: &mut Asm) {
    for pair in 0..PINNED_GUEST_COUNT as u32 / 2 {
        let host = PINNED_HOST_FIRST + 2 * pair;
        let offset = offset_of!(Cpu, gpr) as i32 + GPR_PAIR_BYTES * pair as i32;
        asm.stp_off(host, host + 1, CPU, offset);
    }
}

/// Loads the pinned guest registers from `Cpu::gpr`.
fn reload_pinned(asm: &mut Asm) {
    for pair in 0..PINNED_GUEST_COUNT as u32 / 2 {
        let host = PINNED_HOST_FIRST + 2 * pair;
        let offset = offset_of!(Cpu, gpr) as i32 + GPR_PAIR_BYTES * pair as i32;
        asm.ldp_off(host, host + 1, CPU, offset);
    }
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const JUMP_CACHE_ARGUMENT: u32 = 2;
const VALID_ARGUMENT: u32 = 3;
const PAGE_SHIFT: u32 = 12;
const PAGE_OFFSET_MASK: u64 = (1 << PAGE_SHIFT) - 1;
const FRAME_POINTER: u32 = 29;
const LINK_REGISTER: u32 = 30;
/// 16-byte aligned frame: fp/lr, then the callee-saved registers the blocks use. Those are x19 and
/// x20 (`CPU`, `TLB_BASE`), x21 (`PAGE_BASE`), x22 and x23 (slow-path state), x24..x26 (values
/// that must survive a slow-path helper call) and x27/x28 (the memory thunks' info pointer and
/// return address, see `emit.rs`).
const FRAME_SIZE: i32 = 112;
const SAVE_X21: u32 = 88;
/// The jump cache of the vCPU running the block, which the run loop passes to the trampoline: the
/// code (and so the chain stub) is shared by every vCPU, the cache is not.
const JUMP_CACHE_SLOT: u32 = 96;
/// The validity table (`Jit::valid`), passed the same way.
const VALID_SLOT: u32 = 104;
const SAVE_X19_X20: i32 = 16;
const SAVE_X22_X23: i32 = 32;
const SAVE_X24_X25: i32 = 48;
const SAVE_X26_X27: i32 = 64;
const SAVE_X28: i32 = 80;

/// Builds `extern "C" fn(cpu: *mut Cpu, block: u64, jump_cache: u64, valid: u64) -> u64` (the
/// call in `Cpu::step_block`).
/// It saves the callee-saved registers the blocks use, latches `CPU` and `TLB_BASE`, and jumps
/// into the block. Blocks call helpers with `blr`, which overwrites x30, so the return address
/// is saved in the frame and the stubs return through the restored x30.
pub(crate) fn prologue(base: u64) -> Prologue {
    let mut asm = Asm::new(base);
    asm.stp_pre(FRAME_POINTER, LINK_REGISTER, a64::SP, -FRAME_SIZE);
    asm.stp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.stp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.stp_off(24, 25, a64::SP, SAVE_X24_X25);
    asm.stp_off(26, 27, a64::SP, SAVE_X26_X27);
    asm.str(28, a64::SP, SAVE_X28 as u32);
    asm.str(PAGE_BASE, a64::SP, SAVE_X21);
    asm.str(JUMP_CACHE_ARGUMENT, a64::SP, JUMP_CACHE_SLOT);
    asm.str(VALID_ARGUMENT, a64::SP, VALID_SLOT);
    asm.mov(CPU, 0);
    asm.ldr(
        TLB_BASE,
        CPU,
        std::mem::offset_of!(Cpu, tlb_active_base) as u32,
    );
    // The run loop enters at `Cpu::rip`, so that is the page the block runs from.
    asm.ldr(PAGE_BASE, CPU, offset_of!(Cpu, rip) as u32);
    assert!(asm.and_imm(PAGE_BASE, PAGE_BASE, !PAGE_OFFSET_MASK, true));
    reload_pinned(&mut asm);
    asm.br(1);
    // Block exit with a helper status in x0 and the stopping instruction's index in x1: the
    // exit code is `status | retired << EXIT_COUNT_SHIFT`, where an instruction that
    // completed (`STATUS_EXIT_REQUESTED`) counts as retired and one that faulted does not.
    // Both store the pinned guest registers back first: the run loop reads `Cpu`.
    let exit_fail = asm.here();
    spill_pinned(&mut asm);
    asm.cmp_imm(0, STATUS_EXIT_REQUESTED as u32, true);
    asm.cinc_eq(1, 1);
    asm.lsl_imm(1, 1, EXIT_COUNT_SHIFT);
    asm.orr(0, 0, 1, true);
    let restore = asm.label();
    asm.b(restore);
    // Normal block exit: x0 already holds the finished exit code.
    let exit_ok = asm.here();
    spill_pinned(&mut asm);
    asm.bind(restore);
    asm.ldr(PAGE_BASE, a64::SP, SAVE_X21);
    asm.ldr(28, a64::SP, SAVE_X28 as u32);
    asm.ldp_off(26, 27, a64::SP, SAVE_X26_X27);
    asm.ldp_off(24, 25, a64::SP, SAVE_X24_X25);
    asm.ldp_off(22, 23, a64::SP, SAVE_X22_X23);
    asm.ldp_off(19, 20, a64::SP, SAVE_X19_X20);
    asm.ldp_post(FRAME_POINTER, LINK_REGISTER, a64::SP, FRAME_SIZE);
    asm.ret();
    let chain = chain_stub(&mut asm, exit_ok);
    let cross_page_check = cross_page_check(&mut asm);
    let addrs = Addrs {
        exec: exec_stub(&mut asm),
        cond: helper_stub(&mut asm, helpers::helper_cond as *const () as usize as u64),
        rotate_flags: helper_stub(
            &mut asm,
            helpers::helper_set_rotate_flags as *const () as usize as u64,
        ),
        whole_vector: helper_stub(
            &mut asm,
            helpers::helper_whole_vector as *const () as usize as u64,
        ),
        materialize_flags: helper_stub(
            &mut asm,
            helpers::helper_materialize_flags as *const () as usize as u64,
        ),
        rdtsc: helper_stub(&mut asm, helpers::helper_rdtsc as *const () as usize as u64),
        pause: helper_stub(&mut asm, helpers::helper_pause as *const () as usize as u64),
        exit_ok,
        exit_fail,
        chain,
        cross_page_check,
        load: SIZES.map(|size| {
            thunk(
                &mut asm,
                Access::Load {
                    size,
                    signed: false,
                },
                exit_fail,
            )
        }),
        load_signed: [1u8, 2, 4]
            .map(|size| thunk(&mut asm, Access::Load { size, signed: true }, exit_fail)),
        load_stack: thunk(&mut asm, Access::LoadStack, exit_fail),
        store: SIZES.map(|size| thunk(&mut asm, Access::Store { size, keep: false }, exit_fail)),
        store_keep: SIZES
            .map(|size| thunk(&mut asm, Access::Store { size, keep: true }, exit_fail)),
        store_stack: thunk(&mut asm, Access::StoreStack, exit_fail),
    };
    asm.finish();
    Prologue {
        code: bytes(&asm.code),
        addrs,
    }
}

/// Operand sizes of the memory thunks, indexed by `Gen::size_index`.
const SIZES: [u8; 4] = [1, 2, 4, 8];

/// A stub that calls a Rust helper with the block's x0..x7 as arguments, so a block reaches it
/// with a one-instruction `bl` instead of materializing a 64-bit address, and spills and reloads
/// the pinned guest registers around it (see `Stub`). The return address waits in `SAVED_LR`,
/// which only the memory thunks use otherwise, and stubs and thunks never nest.
fn helper_stub(asm: &mut Asm, helper: u64) -> Stub {
    let start = asm.here();
    asm.mov(SAVED_LR, LINK_REGISTER);
    spill_pinned(asm);
    asm.mov_imm(CALL_TARGET, helper);
    asm.blr(CALL_TARGET);
    reload_pinned(asm);
    asm.ret_reg(SAVED_LR);
    Stub(start)
}

/// `helper_exec_pair(cpu, pair, page_base)`: the pair holds the instruction's page offset, and the
/// helper needs the page base to turn it into its `rip` (see `PAGE_BASE`).
fn exec_stub(asm: &mut Asm) -> Stub {
    const PAGE_BASE_ARGUMENT: u32 = 2;
    let start = asm.here();
    asm.mov(PAGE_BASE_ARGUMENT, PAGE_BASE);
    let call = helper_stub(asm, helpers::helper_exec_pair as *const () as usize as u64);
    debug_assert_eq!(call.address(), start + 4);
    Stub(start)
}

#[derive(Clone, Copy)]
enum Access {
    Load { size: u8, signed: bool },
    LoadStack,
    Store { size: u8, keep: bool },
    StoreStack,
}

/// The out-of-line half of an inline memory access, shared by every block: what a block's slow
/// path does is `adr x7, info; bl thunk; mov dst, x0; b resume`, where `info` is the
/// instruction's `[page offset, block index]` pair next to the block (the thunk adds `PAGE_BASE` to get the rip). The thunk calls the Rust helper
/// with the address in x2 (and the value in `VALUE` for stores), preserving the host flags,
/// the address and (for read-modify-write stores, `keep`) x0 and x1 across the call, and
/// returns to the block with the loaded value in x0. If the helper reports an exception the
/// thunk leaves the block through `exit_fail` instead, with the status and the instruction's
/// index. All of that used to be emitted per access, which made blocks several times bigger.
fn thunk(asm: &mut Asm, access: Access, exit_fail: u64) -> Stub {
    const ADDRESS: u32 = 2;
    const INFO: u32 = 7;
    const LOAD_STATUS: u32 = 1;
    let start = asm.here();
    asm.mrs_nzcv(SAVED_NZCV);
    asm.mov(SAVED_ADDRESS, ADDRESS);
    asm.mov(SAVED_INFO, INFO);
    asm.mov(SAVED_LR, LINK_REGISTER);
    // Tell the clock where in the block this access is (`Cpu::slow_access_insn`, index + 1);
    // the helper clears it. CALL_TARGET is a scratch until the call below.
    asm.ldr(CALL_TARGET, INFO, 8);
    asm.add_imm(CALL_TARGET, CALL_TARGET, 1, true);
    asm.str(CALL_TARGET, CPU, offset_of!(Cpu, slow_access_insn) as u32);
    let fail = asm.label();
    let (helper, size, is_load, keep) = match access {
        Access::Load { size, .. } => (
            helpers::helper_load as *const () as usize as u64,
            size,
            true,
            false,
        ),
        Access::LoadStack => (
            helpers::helper_load_stack as *const () as usize as u64,
            8,
            true,
            false,
        ),
        Access::Store { size, keep } => (
            helpers::helper_store as *const () as usize as u64,
            size,
            false,
            keep,
        ),
        Access::StoreStack => (
            helpers::helper_store_stack as *const () as usize as u64,
            8,
            false,
            false,
        ),
    };
    if keep {
        asm.mov(SAVED_RESULT, 0);
        asm.mov(SAVED_OPERAND, 1);
    }
    if is_load {
        // helper_load(cpu, address, size, rip) -> (value in x0, status in x1)
        asm.mov(1, ADDRESS);
        asm.mov(0, CPU);
        asm.mov_imm(2, size as u64);
        asm.ldr(3, INFO, 0);
        asm.add(3, 3, PAGE_BASE, true);
        spill_pinned(asm);
        asm.mov_imm(CALL_TARGET, helper);
        asm.blr(CALL_TARGET);
        // Before the status check: the fail path's exit stub stores them again, and a nested
        // page fault taken as a #VMEXIT inside the helper changed rsp and rax in `Cpu`.
        reload_pinned(asm);
        asm.cbnz(LOAD_STATUS, fail, true);
        if let Access::Load { size, signed: true } = access {
            match size {
                1 => asm.sxtb(0, 0, true),
                2 => asm.sxth(0, 0, true),
                _ => asm.sxtw(0, 0),
            }
        }
    } else {
        // helper_store(cpu, address, size, value, rip) -> status
        asm.mov(3, VALUE);
        asm.mov(1, ADDRESS);
        asm.mov_imm(2, size as u64);
        asm.ldr(4, INFO, 0);
        asm.add(4, 4, PAGE_BASE, true);
        asm.mov(0, CPU);
        spill_pinned(asm);
        asm.mov_imm(CALL_TARGET, helper);
        asm.blr(CALL_TARGET);
        reload_pinned(asm);
        asm.cbnz(0, fail, true);
    }
    asm.msr_nzcv(SAVED_NZCV);
    asm.mov(ADDRESS, SAVED_ADDRESS);
    if keep {
        asm.mov(0, SAVED_RESULT);
        asm.mov(1, SAVED_OPERAND);
    }
    asm.ret_reg(SAVED_LR);
    asm.bind(fail);
    if is_load {
        // `exit_fail` wants the status in x0.
        asm.mov(0, LOAD_STATUS);
    }
    asm.ldr(1, SAVED_INFO, 8);
    asm.b_abs(exit_fail);
    Stub(start)
}

/// Registers of the chain stub's calling convention (see `Gen::chain_exit`): the block
/// has stored `Cpu::icount`, and x1 holds the guest rip to continue at.
const CHAIN_ICOUNT: u32 = 0;
const CHAIN_RIP: u32 = 1;

/// Block-to-block transfer, the equivalent of QEMU's `lookup_and_goto_ptr`. Entered with
/// `Cpu::icount` as just stored by the block (already charged for the first instruction of the
/// successor, as `Cpu::run` does before `step_block`) and `x1` = the guest rip to continue at.
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
/// The miss path gives back the successor's pre-charge: the loop adds it again.
fn chain_stub(asm: &mut Asm, exit_ok: u64) -> u64 {
    const TLB_SLOT_MASK: u64 = (TLB_ENTRIES_PER_MODE - 1) as u64;
    const TLB_ENTRY_SHIFT: u32 = std::mem::size_of::<TlbEntry>().trailing_zeros();
    // `jump_cache_hash` in `native.rs`: rip ^ (rip >> bits), masked.
    const JUMP_HASH_FOLD_SHIFT: u32 = JUMP_CACHE_BITS as u32;
    let (entry, page, tag, phys, slot, key, miss) = (2, 3, 4, 5, 2, 4, asm.label());
    let start = asm.here();
    asm.ldr(CHAIN_ICOUNT, CPU, offset_of!(Cpu, icount) as u32);
    asm.ldr(entry, CPU, offset_of!(Cpu, next_event) as u32);
    asm.cmp(CHAIN_ICOUNT, entry, true);
    asm.b_cond(a64::CS, miss);
    asm.ldrb(entry, CPU, offset_of!(Cpu, block_exit_requested) as u32);
    asm.cbnz(entry, miss, false);
    asm.lsr_imm(entry, CHAIN_RIP, PAGE_SHIFT);
    assert!(asm.and_imm(entry, entry, TLB_SLOT_MASK, true));
    asm.add_lsl(entry, TLB_BASE, entry, TLB_ENTRY_SHIFT);
    assert!(asm.and_imm(page, CHAIN_RIP, !PAGE_OFFSET_MASK, true));
    asm.ldr(tag, entry, offset_of!(TlbEntry, tag_x) as u32);
    asm.cmp(page, tag, true);
    asm.b_cond(a64::NE, miss);
    // phys = rip + addend - ram_base
    asm.ldr(phys, entry, offset_of!(TlbEntry, addend) as u32);
    asm.add(phys, phys, CHAIN_RIP, true);
    asm.ldr(tag, CPU, offset_of!(Cpu, ram_base) as u32);
    asm.sub(phys, phys, tag, true);
    asm.lsr_imm(slot, CHAIN_RIP, JUMP_HASH_FOLD_SHIFT);
    asm.eor(slot, slot, CHAIN_RIP, true);
    assert!(asm.and_imm(slot, slot, jump_cache_mask(), true));
    asm.ldr(key, a64::SP, JUMP_CACHE_SLOT);
    asm.add_lsl(slot, key, slot, JUMP_ENTRY_SHIFT);
    asm.ldr(key, slot, 0);
    asm.cmp(key, CHAIN_RIP, true);
    asm.b_cond(a64::NE, miss);
    asm.ldr(key, slot, JUMP_ENTRY_PHYS_OFFSET);
    asm.cmp(key, phys, true);
    asm.b_cond(a64::NE, miss);
    asm.ldr(key, slot, JUMP_ENTRY_CODE_OFFSET);
    asm.cbz(key, miss, true);
    // The slot may name a block whose page was written since (`Jit::valid`).
    asm.ldr(phys, slot, JUMP_ENTRY_BLOCK_OFFSET);
    asm.ldr(slot, a64::SP, VALID_SLOT);
    asm.ldr_r(slot, slot, phys, 1, false);
    asm.cbz(slot, miss, false);
    // The successor is entered with its own page as `PAGE_BASE`; `page` still holds it.
    asm.mov(PAGE_BASE, page);
    asm.br(key);
    asm.bind(miss);
    asm.sub_imm(CHAIN_ICOUNT, CHAIN_ICOUNT, 1, true);
    asm.str(CHAIN_ICOUNT, CPU, offset_of!(Cpu, icount) as u32);
    asm.str(CHAIN_RIP, CPU, offset_of!(Cpu, rip) as u32);
    // One instruction retired in the exit code: `step_block` then adds nothing more.
    asm.mov_imm(0, 1 << EXIT_COUNT_SHIFT);
    asm.b_abs(exit_ok);
    start
}

/// The mapping check of a cross-page link (`Gen::fixed_exit`), called with `bl` and x1 = the
/// target rip, x2 = the host address its page had when the block was translated. Returns with the
/// host flags EQ when the exec TLB half the block runs in (the one the chain stub would probe)
/// maps the target's page to that host page, having set `PAGE_BASE` to the target's page for the
/// successor, and NE otherwise, when the block goes through the chain stub instead. A plain
/// subroutine that returns either way, so the return predictor stays balanced; it calls no Rust
/// and touches only x2..x5, so the pinned guest registers stay live across it.
///
/// Comparing host addresses is the same test as the chain stub's physical one (`ram_base` is
/// fixed), one load shorter. The tag check covers everything a TLB fill checked (present,
/// executable at this privilege); a remap or flush of the target page changes or empties the
/// entry, so a link made for another mapping is never taken.
fn cross_page_check(asm: &mut Asm) -> u64 {
    const TLB_SLOT_MASK: u64 = (TLB_ENTRIES_PER_MODE - 1) as u64;
    const TLB_ENTRY_SHIFT: u32 = std::mem::size_of::<TlbEntry>().trailing_zeros();
    const TARGET: u32 = 1;
    const EXPECTED_HOST_PAGE: u32 = 2;
    let (entry, page, value) = (3, 4, 5);
    let start = asm.here();
    let done = asm.label();
    asm.lsr_imm(entry, TARGET, PAGE_SHIFT);
    assert!(asm.and_imm(entry, entry, TLB_SLOT_MASK, true));
    asm.add_lsl(entry, TLB_BASE, entry, TLB_ENTRY_SHIFT);
    assert!(asm.and_imm(page, TARGET, !PAGE_OFFSET_MASK, true));
    asm.ldr(value, entry, offset_of!(TlbEntry, tag_x) as u32);
    asm.cmp(page, value, true);
    asm.b_cond(a64::NE, done);
    asm.ldr(value, entry, offset_of!(TlbEntry, addend) as u32);
    asm.add(value, value, page, true);
    asm.cmp(value, EXPECTED_HOST_PAGE, true);
    asm.b_cond(a64::NE, done);
    // `mov` leaves NZCV alone, so the caller still sees EQ.
    asm.mov(PAGE_BASE, page);
    asm.bind(done);
    asm.ret();
    start
}

/// The addresses blocks are compiled against. Without `chaining` the chain address is 0 and
/// every control transfer returns to the run loop.
pub(crate) fn addrs(prologue: &Prologue, chaining: bool) -> Addrs {
    let mut addrs = prologue.addrs.clone();
    if !chaining {
        addrs.chain = 0;
    }
    addrs
}

/// Compiles one block to machine code meant to run at `addr` (the encoder bakes in pc-relative
/// branches to the exit stubs).
pub(crate) fn compile_block(
    addr: u64,
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
    ordered: bool,
    resolve: ResolveTarget,
) -> CompiledBlock {
    let mut generator = Gen::new(addr, addrs, insns);
    generator.ordered_memory = ordered;
    generator.resolve = resolve;
    generator.compile(rips, end_rip);
    CompiledBlock {
        code: bytes(&generator.asm.code),
        inlined: generator.inlined,
        body_offset: generator.body_offset,
        links: generator.links,
        flag_records_skipped: generator.flag_records_skipped,
    }
}

/// A link slot is one instruction, patched whole.
pub(crate) const LINK_WORD_OFFSET: usize = 0;

const BRANCH_OPCODE: u32 = 0x1400_0000;
const BRANCH_IMM26_MASK: u32 = 0x03ff_ffff;
const INSN_BYTES: i64 = 4;
/// `b` reaches +-2^25 instructions.
const BRANCH_REACH_INSNS: i64 = 1 << 25;

/// The instruction that makes the slot at `site` jump to `destination`, or `None` when the two
/// are further apart than a `b` reaches (blocks in different code segments). Written with
/// `ExecMem::write_word`: another vCPU may be executing the slot, and aarch64 allows a `nop` to
/// be replaced by a branch (and back) under a running thread.
pub(crate) fn link_word(site: u64, destination: u64) -> Option<u32> {
    let offset = (destination as i64 - site as i64) / INSN_BYTES;
    if offset.abs() >= BRANCH_REACH_INSNS {
        return None;
    }
    Some(BRANCH_OPCODE | (offset as u32 & BRANCH_IMM26_MASK))
}

/// What an unlinked slot holds: a nop, so execution falls into the chain stub sequence after it.
pub(crate) fn unlink_word() -> u32 {
    NOP_WORD
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: u64 = 0x1_0000_0000;

    #[test]
    fn a_link_to_the_next_instruction_is_branch_plus_one() {
        assert_eq!(link_word(SITE, SITE + 4), Some(0x1400_0001));
    }

    #[test]
    fn a_link_to_itself_is_branch_zero_and_one_backwards_is_all_ones() {
        assert_eq!(link_word(SITE, SITE), Some(0x1400_0000));
        assert_eq!(link_word(SITE, SITE - 4), Some(0x17ff_ffff));
    }

    #[test]
    fn links_reach_just_under_128_mebibytes_either_way() {
        let reach = (BRANCH_REACH_INSNS - 1) * INSN_BYTES;
        assert!(link_word(SITE, SITE + reach as u64).is_some());
        assert!(link_word(SITE, SITE - reach as u64).is_some());
        assert_eq!(link_word(SITE, SITE + reach as u64 + 4), None);
        assert_eq!(link_word(SITE, SITE - reach as u64 - 4), None);
    }

    #[test]
    fn a_link_word_branches_where_it_says_when_decoded() {
        for delta in [-1_000_000i64, -4, 0, 4, 8, 1_000_000, 100_000_000] {
            let destination = (SITE as i64 + delta) as u64;
            let word = link_word(SITE, destination).unwrap();
            assert_eq!(word & !BRANCH_IMM26_MASK, BRANCH_OPCODE);
            let offset = ((word & BRANCH_IMM26_MASK) << 6) as i32 >> 6;
            assert_eq!(SITE as i64 + offset as i64 * INSN_BYTES, destination as i64);
        }
    }

    #[test]
    fn an_unlinked_slot_is_a_nop_and_the_word_replaces_the_whole_instruction() {
        assert_eq!(unlink_word(), NOP_WORD);
        assert_eq!(LINK_WORD_OFFSET, 0);
    }
}
