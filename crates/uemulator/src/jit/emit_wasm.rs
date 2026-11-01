//! WebAssembly block generator. Mirrors `emit_x64.rs` (read its module docs for the exit protocol
//! and the lazy flags), with the differences a wasm host makes:
//!
//! - A block is one function `(cpu: i32) -> i64` in a module of its own (`wasm_encode`). It reads
//!   and writes the `Cpu` and guest RAM directly, because both live in the emulator's linear
//!   memory, and reaches the helpers with `call_indirect` through the emulator's function table:
//!   a Rust function pointer on wasm32 is such a table index (`Addrs`).
//! - There is no host flags register, so nothing mirrors guest flags except the lazy record in
//!   `Cpu` (`store_lazy`). To still make `cmp; jcc` cheap, the operands and result of the last
//!   flag-writing instruction stay in the locals `LHS`, `RHS` and `RES`, and a condition is
//!   computed from them (`Fused`) instead of calling `helper_cond`. The record is stored anyway:
//!   the successor block, an exception frame or `pushf` may read it.
//! - A block leaves for its successor by a tail call to the chain stub (`compile_chain_stub`), the
//!   `lookup_and_goto_ptr` of the native backends: it checks the same things (the event budget,
//!   the exec TLB, the jump cache, the block's validity) and tail-calls the successor, or returns
//!   to the run loop. An exit to a fixed address in the block's own page goes through a link
//!   cell instead (`Gen::linked_exit`): once the stub has found the successor it is tail-called
//!   directly from there. Every other way out stores `Cpu::rip` and returns
//!   `retired << 8 | status`, and so does every exit when chaining is off (`Addrs::chain == 0`,
//!   the lockstep verifier).
//! - The slow path of a memory access is inline (`if`/`else` around the TLB hit), not out of line,
//!   as wasm has no jump to a label in another place.
//!
//! Every guest instruction without an inline lowering runs through `helper_exec`, so coverage can
//! only matter for speed, never for correctness. A wrong helper type or an out-of-bounds access
//! traps and takes the whole emulator down, which is why the TLB probe mirrors `Cpu::mem_read`
//! exactly and the function types are pinned in `wasm_encode`.

#[cfg(freestanding)]
use crate::prelude::*;
use core::mem::{offset_of, size_of};

use super::blocks::{SLOT_BLOCK_BITS, SLOT_BLOCK_MASK, STRADDLER_TAG};
use super::helpers::*;
use super::wasm_encode::{
    BLOCK_I64, BLOCK_TYPE, BLOCK_VOID, COND_TYPE, Code, EXEC_TYPE, LINK_TYPE, LOAD_TYPE,
    STORE_TYPE, block_module, module_of_type, op,
};
use crate::cpu::decode::{Insn, NO_REG};
use crate::cpu::flags::LazyFlagsOp;
use crate::cpu::{Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE, TlbEntry};

// Offsets into `Cpu` baked into generated code; see the same list in `emit_x64.rs`. They come from
// `offset_of!` on the wasm32 layout, so they match what the module is instantiated against.
const GPR: u32 = offset_of!(Cpu, gpr) as u32;
const GPR_SIZE: u32 = size_of::<u64>() as u32;
const RIP: u32 = offset_of!(Cpu, rip) as u32;
const ICOUNT: u32 = offset_of!(Cpu, icount) as u32;
const PREPAID_ICOUNT: u32 = offset_of!(Cpu, prepaid_icount) as u32;
const NEXT_EVENT: u32 = offset_of!(Cpu, next_event) as u32;
const RAM_BASE: u32 = offset_of!(Cpu, ram_base) as u32;
const BLOCK_EXIT_REQUESTED: u32 = offset_of!(Cpu, block_exit_requested) as u32;
const SLOW_ACCESS_INSN: u32 = offset_of!(Cpu, slow_access_insn) as u32;
/// `tlb_active_base` is a `u64` holding a 32-bit address here; the low word is the address.
const TLB_ACTIVE_BASE: u32 = offset_of!(Cpu, tlb_active_base) as u32;
const LAZY_OP: u32 = offset_of!(Cpu, lazy_op) as u32;
const LAZY_SIZE_SHIFT: u32 = 8;
const LAZY_DST: u32 = offset_of!(Cpu, lazy_dst) as u32;
const LAZY_SRC: u32 = offset_of!(Cpu, lazy_src) as u32;
/// The carry-in of an adc or sbb record.
const LAZY_SRC2: u32 = offset_of!(Cpu, lazy_src2) as u32;
const SEG_BASE: u32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as u32;
const SEG_SIZE: u32 = size_of::<Seg>() as u32;

const _: () = assert!(size_of::<LazyFlagsOp>() == 1);
const _: () = assert!(offset_of!(Cpu, lazy_size) == offset_of!(Cpu, lazy_op) + 1);
const _: () = assert!(offset_of!(Cpu, gpr) == 0 || offset_of!(Cpu, gpr) < 1 << 16);

const TLB_TAG_READ: u32 = offset_of!(TlbEntry, tag_r) as u32;
const TLB_TAG_EXEC: u32 = offset_of!(TlbEntry, tag_x) as u32;
const TLB_TAG_WRITE: u32 = offset_of!(TlbEntry, tag_w) as u32;
const TLB_ADDEND: u32 = offset_of!(TlbEntry, addend) as u32;
const TLB_ENTRY_SHIFT: i32 = size_of::<TlbEntry>().trailing_zeros() as i32;
const _: () = assert!(size_of::<TlbEntry>().is_power_of_two());

const PAGE_SHIFT: i64 = 12;
/// `!0xfff` as an i64 constant (sign-extended).
const PAGE_MASK_NEG: i64 = !((1 << PAGE_SHIFT) - 1);

/// Locals. Parameter 0 is the `Cpu` pointer; the i32 locals follow, then the i64 ones.
const CPU: u32 = 0;
/// `Cpu::tlb_active_base`, loaded once at block entry (CPL only changes inside block-ending
/// instructions).
const TLB_BASE: u32 = 1;
/// The TLB entry a probe selected.
const SLOT: u32 = 2;
const I32_LOCALS: u32 = 2;
/// `rip & !0xfff` of the page the block runs in; every rip in the block is an offset from it
/// (`Gen::rip_to`), which is what makes blocks position independent.
const PAGE_BASE: u32 = 3;
/// Guest linear address of the memory operand being accessed.
const ADDR: u32 = 4;
/// The value a mov-style instruction moves, or an operand being built.
const VALUE: u32 = 5;
/// Operands and result of the last flag-writing instruction (`Fused`), zero-extended to the
/// operand size. Only flag writers touch them, so a fused condition stays valid across anything
/// else that does not change the flags.
const LHS: u32 = 6;
const RHS: u32 = 7;
const RES: u32 = 8;
/// Status returned by a helper.
const STATUS: u32 = 9;
const SCRATCH: u32 = 10;
const SCRATCH_B: u32 = 11;
/// The carry-in of an inc/dec, which keep CF and record it in `lazy_src`.
const CARRY: u32 = 12;
const I64_LOCALS: u32 = 10;

const GUEST_RAX: u8 = 0;
const GUEST_RCX: u8 = 1;
const GUEST_RDX: u8 = 2;
const GUEST_RSP: u8 = 4;
const GUEST_REX_B: u8 = 1;
const REX_B_REG_SHIFT: u8 = 3;
const OPCODE_REG_MASK: u8 = 7;
const MODRM_MOD_REGISTER: u8 = 3;
const HIGH_BYTE_REGS: core::ops::Range<u8> = 4..8;
const HIGH_BYTE_REG_BASE: u8 = 4;
/// Byte offset of bits 15:8 inside a guest register slot (little-endian).
const HIGH_BYTE_OFFSET: u32 = 1;
const PUSH_POP_SIZE: i64 = 8;

const CONDITION_BELOW: u8 = 2;
const PARITY_BYTE_MASK: i64 = 0xff;
/// The low 32 bits: the result of a 32-bit operation, which also clears the upper half of a
/// register.
const LOW_DWORD: i64 = 0xffff_ffff;
/// An arithmetic shift by this fills every bit with the sign (`cdq`, `cqo`).
const SIGN_FILL_SHIFT: i64 = 63;
/// The `SzpCfOf` word for CF = OF = 1.
const CF_AND_OF: i64 = 3;

/// Group-1 ALU operation numbers (the x86 opcode extension), plus a pseudo operation.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_CMP: u8 = 7;
/// `test`: and without writeback. Not a real group-1 number; must stay outside 0..=7.
const ALU_TEST: u8 = 8;

const SHIFT_ROL: u8 = 0;
const SHIFT_ROR: u8 = 1;
const SHIFT_SHL: u8 = 4;
const SHIFT_SHR: u8 = 5;
const SHIFT_SAL_ALIAS: u8 = 6;
const SHIFT_SAR: u8 = 7;
/// `SzpCfOf` packs CF in bit 0 and OF in bit 1 of `lazy_src`.
const SHIFT_OF_BIT: i64 = 1;

/// Helper and scratch addresses the generated code bakes in: table indices of the Rust helpers
/// (a function pointer on wasm32 is its index in the function table) and the address of the
/// 8-byte slot slow-path loads write their value to. Built once per `Jit` by `wasm::addrs`.
pub(crate) struct Addrs {
    pub exec: u32,
    pub ld: u32,
    pub st: u32,
    /// Variants of `ld`/`st` for push/pop/call/ret, which fault with #SS instead of #GP.
    pub ld_stack: u32,
    pub st_stack: u32,
    pub cond: u32,
    pub out: u32,
    /// Table slot of the chain stub, or 0 when blocks return to the run loop between each other.
    pub chain: u32,
    /// Table slot of the chain stub that also fills a link cell (`compile_chain_stub`), or 0
    /// when exits are not linked.
    pub chain_link: u32,
    /// Address of `SharedJit::valid`, the per-block validity bytes a linked exit checks.
    pub valid: u32,
}

/// A link cell: where a block's exit to a fixed address in its own page remembers the block it
/// leads to, so the next time it can tail-call it directly instead of going through the chain
/// stub. The block index sits at offset 0 and the table slot at offset 4 (one little-endian
/// `u64` on the Rust side, `wasm::Block::_links`). Only the linking stub writes a cell, and only
/// after its full hit path; a cell is trusted as long as the block it names is valid, which is
/// enough because cells belong to their block and are freed with it at a flush, the only time
/// block indices are reused.
pub(crate) const LINK_CELL_BLOCK: u32 = 0;
pub(crate) const LINK_CELL_SLOT: u32 = 4;
pub(crate) const LINK_CELL_SIZE: u32 = 8;
/// A block has at most two exits to fixed addresses: a jcc's two outcomes, or a jmp, a call or
/// the fall-through after its last instruction.
pub(crate) const LINKS_PER_BLOCK: u32 = 2;
/// Page offsets below this are in the block's own page (`Gen::rip_to`).
const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;

/// What `compile_block` produces.
pub(crate) struct CompiledBlock {
    /// The module the host instantiates; its `block` export is the block's function.
    pub module: Vec<u8>,
    /// Guest instructions lowered inline.
    pub inlined: u32,
    /// Flag records the block leaves out because a later instruction overwrites them.
    pub flag_records_skipped: u32,
}

/// A register operand is its number plus whether it is the high byte of a legacy register.
#[derive(Clone, Copy)]
enum Dst {
    Reg(u8, bool),
    Mem,
}

#[derive(Clone, Copy)]
enum Src {
    Reg(u8, bool),
    Mem,
    Imm(u64),
}

#[derive(Clone, Copy, PartialEq)]
enum ShiftKind {
    Shl,
    Shr,
    Sar,
}

#[derive(Clone, Copy, PartialEq)]
enum FusedOp {
    Add,
    Sub,
    Logic,
}

/// What the locals `LHS`, `RHS` and `RES` hold: the operands and (masked) result of the last
/// flag-writing instruction, so conditions need no call.
#[derive(Clone, Copy)]
struct Fused {
    op: FusedOp,
    size: u8,
    /// False after inc/dec: CF is the carry-in, which only the lazy record knows.
    carry_known: bool,
}

fn opcode_register(insn: &Insn) -> u8 {
    (insn.op as u8 & OPCODE_REG_MASK) | ((insn.rex & GUEST_REX_B) << REX_B_REG_SHIFT)
}

fn has_fs_gs_override(insn: &Insn) -> bool {
    matches!(insn.seg as usize, SEG_FS | SEG_GS)
}

fn register_offset(reg: u8, high: bool) -> u32 {
    GPR + GPR_SIZE * reg as u32 + if high { HIGH_BYTE_OFFSET } else { 0 }
}

fn width_mask(size: u8) -> i64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => LOW_DWORD,
        _ => -1,
    }
}

fn sign_bit(size: u8) -> i64 {
    (1u64 << (8 * size as u32 - 1)) as i64
}

fn load_opcode(size: u8, signed: bool) -> u8 {
    match (size, signed) {
        (1, false) => op::I64_LOAD8_U,
        (1, true) => op::I64_LOAD8_S,
        (2, false) => op::I64_LOAD16_U,
        (2, true) => op::I64_LOAD16_S,
        (4, false) => op::I64_LOAD32_U,
        (4, true) => op::I64_LOAD32_S,
        _ => op::I64_LOAD,
    }
}

/// A register write: byte and word operands are merged by a narrow store, wider ones replace all
/// 64 bits, so a 4-byte value must already be zero-extended.
fn register_store_opcode(size: u8) -> u8 {
    match size {
        1 => op::I64_STORE8,
        2 => op::I64_STORE16,
        _ => op::I64_STORE,
    }
}

fn memory_store_opcode(size: u8) -> u8 {
    match size {
        1 => op::I64_STORE8,
        2 => op::I64_STORE16,
        4 => op::I64_STORE32,
        _ => op::I64_STORE,
    }
}

fn sign_extend_opcode(size: u8) -> Option<u8> {
    match size {
        1 => Some(op::I64_EXTEND8_S),
        2 => Some(op::I64_EXTEND16_S),
        4 => Some(op::I64_EXTEND32_S),
        _ => None,
    }
}

pub(crate) struct Gen<'a> {
    c: Code,
    addrs: &'a Addrs,
    /// Generated code embeds addresses into this slice (for `helper_exec`), so the caller must
    /// keep it alive as long as the block.
    insns: &'a [Insn],
    flags: Option<Fused>,
    /// Block index of the last instruction whose clock read was paid for in advance; see
    /// `Gen::helper`.
    prepaid_insns: u32,
    pub inlined: u32,
    /// Offset of the block's first instruction in its page.
    entry_rip: u64,
    flag_record_needed: Vec<bool>,
    skip_flag_record: bool,
    pub flag_records_skipped: u32,
    /// Address of the block's `LINKS_PER_BLOCK` link cells, or 0 when its exits are not linked.
    link_cells: u32,
    links_used: u32,
}

impl<'a> Gen<'a> {
    pub fn new(addrs: &'a Addrs, insns: &'a [Insn], link_cells: u32) -> Self {
        Gen {
            c: Code::default(),
            addrs,
            insns,
            flags: None,
            prepaid_insns: 0,
            inlined: 0,
            entry_rip: 0,
            flag_record_needed: flag_record_needed(insns),
            skip_flag_record: false,
            flag_records_skipped: 0,
            link_cells,
            links_used: 0,
        }
    }

    fn end(&mut self) {
        self.c.op(op::END);
    }

    /// `PAGE_BASE + rip`, where `rip` is the offset in the block's page that blocks carry around
    /// instead of guest addresses (wrapping: a jump to a lower page is negative).
    fn rip_to(&mut self, rip: u64) {
        self.c.local_get(PAGE_BASE);
        if rip != 0 {
            self.c.i64_const(rip as i64).op(op::I64_ADD);
        }
    }

    fn store_rip(&mut self, rip: u64) {
        self.c.local_get(CPU);
        self.rip_to(rip);
        self.c.mem(op::I64_STORE, RIP);
    }

    fn store_cpu_u64(&mut self, offset: u32, value: u64) {
        self.c
            .local_get(CPU)
            .i64_const(value as i64)
            .mem(op::I64_STORE, offset);
    }

    /// Leave the block normally having retired `retired` instructions.
    fn exit_ok(&mut self, retired: u32) {
        self.c
            .i64_const((retired as i64) << EXIT_COUNT_SHIFT)
            .op(op::RETURN);
    }

    /// Ends the block at a fixed guest address.
    fn chain_to(&mut self, rip: u64, retired: u32) {
        self.store_rip(rip);
        match self.next_link_cell(rip) {
            Some(cell) => self.linked_exit(cell, retired),
            None => self.chain_exit(retired),
        }
    }

    /// A link cell for an exit to page offset `rip`, if it may be linked: a fixed address in
    /// the block's own page. Same page is what lets a linked exit skip the stub's exec TLB probe
    /// and jump-cache key check: the block was entered through a validated mapping of this
    /// physical page, so the target's physical address is this page plus the offset, and the
    /// block the cell names was found for exactly that address.
    fn next_link_cell(&mut self, rip: u64) -> Option<u32> {
        if self.addrs.chain_link == 0
            || self.link_cells == 0
            || rip >= PAGE_SIZE
            || self.links_used == LINKS_PER_BLOCK
        {
            return None;
        }
        let cell = self.link_cells + self.links_used * LINK_CELL_SIZE;
        self.links_used += 1;
        Some(cell)
    }

    /// `chain_exit` through link cell `cell`, with `Cpu::rip` stored: the event budget and
    /// `block_exit_requested` checks the stub would make (in that order: a store of this very
    /// block may have invalidated the target), then the cell's block, if still valid, is
    /// tail-called directly. Otherwise the linking stub runs the full lookup and, on a hit,
    /// fills the cell for next time.
    ///
    /// The valid byte is checked before the slot is used: an empty cell names the sentinel
    /// block `wasm::NO_BLOCK`, never valid, and holds slot 0, the null function, which traps.
    fn linked_exit(&mut self, cell: u32, retired: u32) {
        let owed = retired - self.prepaid_insns;
        self.c
            .local_get(CPU)
            .local_get(CPU)
            .mem(op::I64_LOAD, ICOUNT)
            .i64_const(owed as i64)
            .op(op::I64_ADD)
            .local_tee(SCRATCH)
            .mem(op::I64_STORE, ICOUNT);
        if self.prepaid_insns > 0 {
            self.store_cpu_u64(PREPAID_ICOUNT, 0);
        }
        self.c.begin_block();
        self.c
            .local_get(SCRATCH)
            .local_get(CPU)
            .mem(op::I64_LOAD, NEXT_EVENT)
            .op(op::I64_GE_U)
            .br_if(0);
        self.c
            .local_get(CPU)
            .mem(op::I32_LOAD8_U, BLOCK_EXIT_REQUESTED)
            .br_if(0);
        self.c
            .i32_const(cell as i32)
            .mem(op::I32_LOAD, LINK_CELL_BLOCK)
            .mem(op::I32_LOAD8_U, self.addrs.valid)
            .op(op::I32_EQZ)
            .br_if(0);
        self.c
            .local_get(CPU)
            .i32_const(cell as i32)
            .mem(op::I32_LOAD, LINK_CELL_SLOT)
            .return_call_indirect(BLOCK_TYPE);
        self.end();
        self.c
            .local_get(CPU)
            .i32_const(cell as i32)
            .i32_const(self.addrs.chain_link as i32)
            .return_call_indirect(LINK_TYPE);
    }

    /// Ends the block at the guest address in local `target`.
    fn chain_to_local(&mut self, target: u32, retired: u32) {
        self.c
            .local_get(CPU)
            .local_get(target)
            .mem(op::I64_STORE, RIP);
        self.chain_exit(retired);
    }

    /// Leaves for the block at `Cpu::rip`, with `retired` instructions done. With chaining that
    /// is a tail call to the stub, which wants `icount` as just stored and already including the
    /// successor's first instruction (see `compile_chain_stub`): the block pays for its own
    /// instructions except those paid ahead of a helper, plus that one (the run loop's charge for
    /// this block's first instruction stands in for the successor's). Without chaining it returns
    /// to the run loop.
    fn chain_exit(&mut self, retired: u32) {
        if self.addrs.chain == 0 {
            self.exit_ok(retired);
            return;
        }
        let owed = retired - self.prepaid_insns;
        self.c
            .local_get(CPU)
            .local_get(CPU)
            .mem(op::I64_LOAD, ICOUNT)
            .i64_const(owed as i64)
            .op(op::I64_ADD)
            .mem(op::I64_STORE, ICOUNT);
        if self.prepaid_insns > 0 {
            // The next block's own accounting must not subtract this block's prepayment.
            self.store_cpu_u64(PREPAID_ICOUNT, 0);
        }
        self.c
            .local_get(CPU)
            .i32_const(self.addrs.chain as i32)
            .return_call_indirect(BLOCK_TYPE);
    }

    /// With the helper status in `STATUS` non-zero, leaves the block reporting `insn_index`
    /// instructions retired (one more for `STATUS_EXIT_REQUESTED`, which retires the instruction).
    fn fail_exit(&mut self, insn_index: u32, status_can_request_exit: bool) {
        if status_can_request_exit {
            self.c
                .local_get(STATUS)
                .i64_const(STATUS_EXIT_REQUESTED as i64)
                .op(op::I64_EQ)
                .op(op::I64_EXTEND_I32_U)
                .i64_const(insn_index as i64)
                .op(op::I64_ADD)
                .i64_const(EXIT_COUNT_SHIFT as i64)
                .op(op::I64_SHL);
        } else {
            self.c.i64_const((insn_index as i64) << EXIT_COUNT_SHIFT);
        }
        self.c.local_get(STATUS).op(op::I64_OR).op(op::RETURN);
    }

    /// With a helper's status on the stack, leaves the block when it is not zero.
    fn check_status(&mut self, insn_index: u32, status_can_request_exit: bool) {
        self.c
            .local_tee(STATUS)
            .op(op::I64_EQZ)
            .op(op::I32_EQZ)
            .begin_if(BLOCK_VOID);
        self.fail_exit(insn_index, status_can_request_exit);
        self.end();
    }

    /// Run instruction `insn_index` through the interpreter. `terminal` ends the block after it.
    fn helper(&mut self, insn_index: usize, rip: u64, terminal: bool) {
        let insn = &self.insns[insn_index] as *const Insn as usize as i32;
        // The run loop already counted one instruction for the block. The interpreter may read
        // guest time (rdtsc, port I/O, a device load), which derives from `icount`, so add the
        // ones before this now and tell `step_block` through `prepaid_icount` not to count them
        // again.
        if insn_index as u32 > self.prepaid_insns {
            let owed = insn_index as u32 - self.prepaid_insns;
            self.c
                .local_get(CPU)
                .local_get(CPU)
                .mem(op::I64_LOAD, ICOUNT)
                .i64_const(owed as i64)
                .op(op::I64_ADD)
                .mem(op::I64_STORE, ICOUNT);
            self.store_cpu_u64(PREPAID_ICOUNT, insn_index as u64);
            self.prepaid_insns = insn_index as u32;
        }
        self.c.local_get(CPU).i32_const(insn);
        self.rip_to(rip);
        self.c
            .i32_const(self.addrs.exec as i32)
            .call_indirect(EXEC_TYPE);
        self.check_status(insn_index as u32, true);
        if terminal {
            self.exit_ok(insn_index as u32 + 1);
        }
        self.flags = None;
    }

    /// Guest linear address of `insn`'s memory operand into `ADDR`.
    fn ea(&mut self, insn: &Insn, next: u64) {
        let disp = insn.disp as i32 as i64;
        if insn.rip_rel {
            self.rip_to(next.wrapping_add(insn.disp as u64));
        } else if insn.index != NO_REG {
            if insn.base != NO_REG {
                self.read_reg(insn.base, false, 8);
            } else {
                self.c.i64_const(0);
            }
            self.read_reg(insn.index, false, 8);
            self.c
                .i64_const(insn.scale as i64)
                .op(op::I64_SHL)
                .op(op::I64_ADD);
            if disp != 0 {
                self.c.i64_const(disp).op(op::I64_ADD);
            }
        } else if insn.base != NO_REG {
            self.read_reg(insn.base, false, 8);
            if disp != 0 {
                self.c.i64_const(disp).op(op::I64_ADD);
            }
        } else {
            self.c.i64_const(insn.disp);
        }
        // In 64-bit mode only the FS and GS bases apply; other overrides are ignored.
        if has_fs_gs_override(insn) {
            self.c
                .local_get(CPU)
                .mem(op::I64_LOAD, SEG_BASE + SEG_SIZE * insn.seg as u32)
                .op(op::I64_ADD);
        }
        self.c.local_set(ADDR);
    }

    /// Leaves 1 on the stack if the TLB entry for `ADDR` holds the page of the access's last byte
    /// (and selects that entry in `SLOT`). Mirrors the fast path of `Cpu::mem_read`/`mem_write`:
    /// the entry is picked by the first byte's page, so an access that crosses a page never
    /// matches (the next page maps to a different slot) and takes the slow path, which splits it.
    /// `TLB_TAG_INVALID` has low bits set, so it never equals a page-aligned value.
    fn tlb_probe(&mut self, size: u8, tag_offset: u32) {
        self.c
            .local_get(ADDR)
            .i64_const(PAGE_SHIFT)
            .op(op::I64_SHR_U)
            .i64_const(TLB_ENTRIES_PER_MODE as i64 - 1)
            .op(op::I64_AND)
            .op(op::I32_WRAP_I64)
            .i32_const(TLB_ENTRY_SHIFT)
            .op(op::I32_SHL)
            .local_get(TLB_BASE)
            .op(op::I32_ADD)
            .local_set(SLOT);
        self.c
            .local_get(SLOT)
            .mem(op::I64_LOAD, tag_offset)
            .local_get(ADDR);
        if size > 1 {
            self.c.i64_const(size as i64 - 1).op(op::I64_ADD);
        }
        self.c
            .i64_const(PAGE_MASK_NEG)
            .op(op::I64_AND)
            .op(op::I64_EQ);
    }

    /// The host address of `ADDR` through the entry `tlb_probe` selected: `la + addend`, wrapped to
    /// the 32 bits of a wasm address.
    fn host_address(&mut self) {
        self.c
            .local_get(ADDR)
            .local_get(SLOT)
            .mem(op::I64_LOAD, TLB_ADDEND)
            .op(op::I64_ADD)
            .op(op::I32_WRAP_I64);
    }

    /// What the slow-path helpers need that they do not set themselves: the faulting instruction's
    /// `rip`, and where in the block the access is, for the device clock
    /// (`Cpu::slow_access_insn`, index + 1; the helper clears it).
    fn prepare_slow_access(&mut self, insn_index: u32, rip: u64) {
        self.store_rip(rip);
        self.store_cpu_u64(SLOW_ACCESS_INSN, insn_index as u64 + 1);
    }

    /// Leaves the `size`-byte value at guest address `ADDR` on the stack as an i64 (zero- or
    /// sign-extended). A TLB miss runs the helper inline; a fault leaves the block.
    fn mem_load(&mut self, size: u8, signed: bool, insn_index: u32, rip: u64, stack: bool) {
        self.tlb_probe(size, TLB_TAG_READ);
        self.c.begin_if(BLOCK_I64);
        self.host_address();
        self.c.mem(load_opcode(size, signed), 0);
        self.c.op(op::ELSE);
        self.prepare_slow_access(insn_index, rip);
        self.c
            .local_get(CPU)
            .local_get(ADDR)
            .i64_const(size as i64)
            .i32_const(self.addrs.out as i32)
            .i32_const(
                (if stack {
                    self.addrs.ld_stack
                } else {
                    self.addrs.ld
                }) as i32,
            )
            .call_indirect(LOAD_TYPE);
        self.check_status(insn_index, false);
        self.c.i32_const(self.addrs.out as i32).mem(op::I64_LOAD, 0);
        if signed && let Some(extend) = sign_extend_opcode(size) {
            self.c.op(extend);
        }
        self.end();
    }

    /// Stores the low `size` bytes of local `value` at guest address `ADDR`. The write tag is
    /// invalidated for pages holding translated code, so stores to code always miss and reach
    /// `Cpu::code_written`.
    fn mem_store(&mut self, size: u8, value: u32, insn_index: u32, rip: u64, stack: bool) {
        self.tlb_probe(size, TLB_TAG_WRITE);
        self.c.begin_if(BLOCK_VOID);
        self.host_address();
        self.c.local_get(value).mem(memory_store_opcode(size), 0);
        self.c.op(op::ELSE);
        self.prepare_slow_access(insn_index, rip);
        self.c
            .local_get(CPU)
            .local_get(ADDR)
            .i64_const(size as i64)
            .local_get(value)
            .i32_const(
                (if stack {
                    self.addrs.st_stack
                } else {
                    self.addrs.st
                }) as i32,
            )
            .call_indirect(STORE_TYPE);
        self.check_status(insn_index, false);
        self.end();
    }

    /// After a store: if it invalidated translated code (`Cpu::code_written` sets
    /// `block_exit_requested`), leave the block so the rest of it does not run from stale code.
    fn smc_check(&mut self, insn_index: u32, next: u64) {
        self.c
            .local_get(CPU)
            .mem(op::I32_LOAD8_U, BLOCK_EXIT_REQUESTED)
            .begin_if(BLOCK_VOID);
        self.store_rip(next);
        self.exit_ok(insn_index + 1);
        self.end();
    }

    /// Pushes the low `size` bytes of guest register `reg` (or its bits 15:8), zero-extended.
    fn read_reg(&mut self, reg: u8, high: bool, size: u8) {
        self.c
            .local_get(CPU)
            .mem(load_opcode(size, false), register_offset(reg, high));
    }

    /// Writes the low `size` bytes of local `value` to guest register `reg`.
    fn write_reg(&mut self, reg: u8, high: bool, size: u8, value: u32) {
        self.c
            .local_get(CPU)
            .local_get(value)
            .mem(register_store_opcode(size), register_offset(reg, high));
    }

    fn named_register(insn: &Insn, reg: u8, size: u8) -> (u8, bool) {
        if size == 1 && !insn.rex_present && HIGH_BYTE_REGS.contains(&reg) {
            (reg - HIGH_BYTE_REG_BASE, true)
        } else {
            (reg, false)
        }
    }

    fn rm_dst(insn: &Insn, size: u8) -> Dst {
        if insn.md == MODRM_MOD_REGISTER {
            let (reg, high) = Self::named_register(insn, insn.rm, size);
            Dst::Reg(reg, high)
        } else {
            Dst::Mem
        }
    }

    fn rm_src(insn: &Insn, size: u8) -> Src {
        match Self::rm_dst(insn, size) {
            Dst::Reg(reg, high) => Src::Reg(reg, high),
            Dst::Mem => Src::Mem,
        }
    }

    fn reg_dst(insn: &Insn, size: u8) -> Dst {
        let (reg, high) = Self::named_register(insn, insn.reg, size);
        Dst::Reg(reg, high)
    }

    fn reg_src(insn: &Insn, size: u8) -> Src {
        let (reg, high) = Self::named_register(insn, insn.reg, size);
        Src::Reg(reg, high)
    }

    /// An immediate as the operand of an operation of `size` bytes: truncated to the width.
    fn imm_for(size: u8, imm: i64) -> u64 {
        match size {
            1 => imm as u8 as u64,
            2 => imm as u16 as u64,
            4 => imm as u32 as u64,
            _ => imm as u64,
        }
    }

    /// Memory operands the inline path handles: 64-bit addressing, no lock prefix.
    fn mem_ok(insn: &Insn) -> bool {
        !insn.addr32 && !insn.lock
    }

    /// Truncates the i64 on the stack to `size` bytes.
    fn mask_to(&mut self, size: u8) {
        if size < 8 {
            self.c.i64_const(width_mask(size)).op(op::I64_AND);
        }
    }

    /// Loads the `size`-byte operand `dst` names into local `into`, zero-extended.
    #[allow(clippy::too_many_arguments)]
    fn load_dst(
        &mut self,
        insn: &Insn,
        dst: Dst,
        into: u32,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
    ) {
        match dst {
            Dst::Reg(reg, high) => self.read_reg(reg, high, size),
            Dst::Mem => {
                self.ea(insn, next);
                self.mem_load(size, false, index, rip, false);
            }
        }
        self.c.local_set(into);
    }

    /// Writes local `value` to the operand `dst` names. A memory write returns true: the caller
    /// must follow with `smc_check`. The address is the one `load_dst` computed (`ADDR` survives
    /// everything in between), and the store may fault, which is why flags are recorded only after
    /// it. Instructions that leave the flags alone must not use `LHS`, `RHS` or `RES` for their
    /// value (`Fused`).
    fn store_dst(&mut self, dst: Dst, size: u8, value: u32, index: u32, rip: u64) -> bool {
        match dst {
            Dst::Reg(reg, high) => {
                self.write_reg(reg, high, size, value);
                false
            }
            Dst::Mem => {
                self.mem_store(size, value, index, rip, false);
                true
            }
        }
    }

    /// Records the flags of the instruction just emitted: result in `RES`, second operand in local
    /// `src`. `lazy_op` and `lazy_size` are adjacent bytes, written with one 16-bit store. For
    /// logic ops the interpreter stores 0 in `lazy_src` while this stores the operand; that is
    /// equivalent because `flags::compute` derives a Logic result from the result alone.
    fn store_lazy(&mut self, lazy: LazyFlagsOp, size: u8, src: u32) {
        self.c
            .local_get(CPU)
            .i32_const((lazy as u32 | (size as u32) << LAZY_SIZE_SHIFT) as i32)
            .mem(op::I32_STORE16, LAZY_OP);
        self.c
            .local_get(CPU)
            .local_get(RES)
            .mem(op::I64_STORE, LAZY_DST);
        self.c
            .local_get(CPU)
            .local_get(src)
            .mem(op::I64_STORE, LAZY_SRC);
    }

    /// `store_lazy` for an instruction whose flags `Fused` can also serve.
    fn record_flags(
        &mut self,
        lazy: LazyFlagsOp,
        fused: FusedOp,
        size: u8,
        carry_known: bool,
        src: u32,
    ) {
        if self.skip_flag_record {
            // A later instruction of the block overwrites these flags before anything can read
            // them, and the locals still serve a jcc that follows.
            self.flag_records_skipped += 1;
        } else {
            self.store_lazy(lazy, size, src);
        }
        self.flags = Some(Fused {
            op: fused,
            size,
            carry_known,
        });
    }

    /// The pieces of a condition from the fused locals, each leaving an i32 0/1.
    fn zero_flag(&mut self) {
        self.c.local_get(RES).op(op::I64_EQZ);
    }

    fn sign_flag(&mut self, size: u8) {
        self.c
            .local_get(RES)
            .i64_const(sign_bit(size))
            .op(op::I64_AND)
            .op(op::I64_EQZ)
            .op(op::I32_EQZ);
    }

    fn carry_flag(&mut self, fused: Fused) {
        match fused.op {
            FusedOp::Sub => {
                self.c.local_get(LHS).local_get(RHS).op(op::I64_LT_U);
            }
            FusedOp::Add => {
                self.c.local_get(RES).local_get(LHS).op(op::I64_LT_U);
            }
            FusedOp::Logic => {
                self.c.i32_const(0);
            }
        }
    }

    fn overflow_flag(&mut self, fused: Fused) {
        match fused.op {
            FusedOp::Sub => {
                // ((a ^ b) & (a ^ res)) sign
                self.c
                    .local_get(LHS)
                    .local_get(RHS)
                    .op(op::I64_XOR)
                    .local_get(LHS)
                    .local_get(RES)
                    .op(op::I64_XOR)
                    .op(op::I64_AND);
            }
            FusedOp::Add => {
                // ((a ^ res) & (b ^ res)) sign
                self.c
                    .local_get(LHS)
                    .local_get(RES)
                    .op(op::I64_XOR)
                    .local_get(RHS)
                    .local_get(RES)
                    .op(op::I64_XOR)
                    .op(op::I64_AND);
            }
            FusedOp::Logic => {
                self.c.i32_const(0);
                return;
            }
        }
        self.c
            .i64_const(sign_bit(fused.size))
            .op(op::I64_AND)
            .op(op::I64_EQZ)
            .op(op::I32_EQZ);
    }

    fn parity_flag(&mut self) {
        self.c
            .local_get(RES)
            .i64_const(PARITY_BYTE_MASK)
            .op(op::I64_AND)
            .op(op::I64_POPCNT)
            .i64_const(1)
            .op(op::I64_AND)
            .op(op::I64_EQZ);
    }

    /// `sf != of`, the condition of `l`.
    fn sign_differs_from_overflow(&mut self, fused: Fused) {
        match fused.op {
            // after a subtraction, a signed compare of the operands is exactly that (the operands
            // of a narrow one are zero-extended, so they are first moved to the top of the word)
            FusedOp::Sub => {
                if fused.size == 8 {
                    self.c.local_get(LHS).local_get(RHS).op(op::I64_LT_S);
                } else {
                    let align = 64 - 8 * fused.size as i64;
                    self.c
                        .local_get(LHS)
                        .i64_const(align)
                        .op(op::I64_SHL)
                        .local_get(RHS)
                        .i64_const(align)
                        .op(op::I64_SHL)
                        .op(op::I64_LT_S);
                }
            }
            FusedOp::Add | FusedOp::Logic => {
                self.sign_flag(fused.size);
                self.overflow_flag(fused);
                self.c.op(op::I32_XOR);
            }
        }
    }

    /// Whether `condition` can compute guest condition code `cc` without a call.
    fn fused_serves(fused: Fused, cc: u8) -> bool {
        // B, AE, BE and A read CF, which inc and dec do not recompute.
        fused.carry_known || !matches!(cc & 15, 2 | 3 | 6 | 7)
    }

    /// Leaves 1 or 0 on the stack (i32) according to guest condition code `cc`: from the locals
    /// when the flags are still the ones the last flag writer left in them, else from
    /// `helper_cond` over the lazy record.
    fn condition(&mut self, cc: u8) {
        let cc = cc & 15;
        let Some(fused) = self.flags.filter(|fused| Self::fused_serves(*fused, cc)) else {
            self.c
                .local_get(CPU)
                .i64_const(cc as i64)
                .i32_const(self.addrs.cond as i32)
                .call_indirect(COND_TYPE)
                .op(op::I32_WRAP_I64);
            return;
        };
        let negate = cc & 1 != 0;
        match cc >> 1 {
            0 => self.overflow_flag(fused),
            1 => self.carry_flag(fused),
            2 => self.zero_flag(),
            3 => {
                // be: CF or ZF. After a subtraction that is a <= b unsigned.
                if fused.op == FusedOp::Sub {
                    self.c.local_get(LHS).local_get(RHS).op(op::I64_LE_U);
                } else {
                    self.carry_flag(fused);
                    self.zero_flag();
                    self.c.op(op::I32_OR);
                }
            }
            4 => self.sign_flag(fused.size),
            5 => self.parity_flag(),
            6 => self.sign_differs_from_overflow(fused),
            _ => {
                // le: ZF or SF != OF
                self.zero_flag();
                self.sign_differs_from_overflow(fused);
                self.c.op(op::I32_OR);
            }
        }
        if negate {
            self.c.op(op::I32_EQZ);
        }
    }

    /// Emits `dst = dst <alu_op> src` at `size` (1, 2, 4 or 8) and records its flags. `alu_op` is
    /// a group-1 number or `ALU_TEST`; `ALU_CMP` and `ALU_TEST` discard the result. Returns true
    /// if it stored to memory (the caller follows with `smc_check`).
    ///
    /// adc and sbb take CF first, before the operand loads overwrite the fused locals it may be
    /// computed from, and record `Adc`/`Sbb` with the carry-in in `lazy_src2`. The fused locals
    /// cannot express a borrow with a carry-in, so conditions after them use the record.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        insn: &Insn,
        alu_op: u8,
        dst: Dst,
        src: Src,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
    ) -> bool {
        let with_carry = matches!(alu_op, ALU_ADC | ALU_SBB);
        if with_carry {
            self.condition(CONDITION_BELOW);
            self.c.op(op::I64_EXTEND_I32_U).local_set(CARRY);
        }
        // The memory operand is loaded first: it may fault, and nothing has been written yet.
        match (dst, src) {
            (Dst::Mem, Src::Reg(reg, high)) => {
                self.load_dst(insn, dst, LHS, size, next, index, rip);
                self.read_reg(reg, high, size);
                self.c.local_set(RHS);
            }
            (Dst::Mem, Src::Imm(value)) => {
                self.load_dst(insn, dst, LHS, size, next, index, rip);
                self.c.i64_const(value as i64).local_set(RHS);
            }
            (Dst::Reg(reg, high), Src::Mem) => {
                self.ea(insn, next);
                self.mem_load(size, false, index, rip, false);
                self.c.local_set(RHS);
                self.read_reg(reg, high, size);
                self.c.local_set(LHS);
            }
            (Dst::Reg(dst_reg, dst_high), Src::Reg(src_reg, src_high)) => {
                self.read_reg(dst_reg, dst_high, size);
                self.c.local_set(LHS);
                self.read_reg(src_reg, src_high, size);
                self.c.local_set(RHS);
            }
            (Dst::Reg(reg, high), Src::Imm(value)) => {
                self.read_reg(reg, high, size);
                self.c.local_set(LHS);
                self.c.i64_const(value as i64).local_set(RHS);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        self.c.local_get(LHS).local_get(RHS);
        let (lazy, fused) = match alu_op {
            ALU_ADD => {
                self.c.op(op::I64_ADD);
                self.mask_to(size);
                (LazyFlagsOp::Add, FusedOp::Add)
            }
            ALU_ADC | ALU_SBB => {
                let (arithmetic, lazy) = if alu_op == ALU_ADC {
                    (op::I64_ADD, LazyFlagsOp::Adc)
                } else {
                    (op::I64_SUB, LazyFlagsOp::Sbb)
                };
                self.c.op(arithmetic).local_get(CARRY).op(arithmetic);
                self.mask_to(size);
                self.c.local_set(RES);
                let stored = self.store_dst(dst, size, RES, index, rip);
                self.store_lazy(lazy, size, RHS);
                self.c
                    .local_get(CPU)
                    .local_get(CARRY)
                    .mem(op::I64_STORE, LAZY_SRC2);
                self.flags = None;
                return stored;
            }
            ALU_SUB | ALU_CMP => {
                self.c.op(op::I64_SUB);
                self.mask_to(size);
                (LazyFlagsOp::Sub, FusedOp::Sub)
            }
            ALU_OR => {
                self.c.op(op::I64_OR);
                (LazyFlagsOp::Logic, FusedOp::Logic)
            }
            ALU_AND | ALU_TEST => {
                self.c.op(op::I64_AND);
                (LazyFlagsOp::Logic, FusedOp::Logic)
            }
            _ => {
                self.c.op(op::I64_XOR);
                (LazyFlagsOp::Logic, FusedOp::Logic)
            }
        };
        self.c.local_set(RES);
        let stored =
            alu_op != ALU_CMP && alu_op != ALU_TEST && self.store_dst(dst, size, RES, index, rip);
        self.record_flags(lazy, fused, size, true, RHS);
        stored
    }

    /// `inc`/`dec` of a register or memory operand. x86 leaves CF alone, so the lazy record
    /// carries the current CF in as its second operand; it is read before the operand load.
    #[allow(clippy::too_many_arguments)]
    fn inc_dec(
        &mut self,
        insn: &Insn,
        decrement: bool,
        dst: Dst,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
    ) -> bool {
        self.condition(CONDITION_BELOW);
        self.c.op(op::I64_EXTEND_I32_U).local_set(CARRY);
        self.load_dst(insn, dst, LHS, size, next, index, rip);
        self.c.i64_const(1).local_set(RHS);
        self.c.local_get(LHS).local_get(RHS);
        self.c.op(if decrement { op::I64_SUB } else { op::I64_ADD });
        self.mask_to(size);
        self.c.local_set(RES);
        let stored = self.store_dst(dst, size, RES, index, rip);
        self.record_flags(
            if decrement {
                LazyFlagsOp::Dec
            } else {
                LazyFlagsOp::Inc
            },
            if decrement {
                FusedOp::Sub
            } else {
                FusedOp::Add
            },
            size,
            false,
            CARRY,
        );
        stored
    }

    /// `neg`: flags are those of `0 - operand`, recorded as a subtraction whose second operand is
    /// the original value.
    fn neg(&mut self, insn: &Insn, dst: Dst, size: u8, next: u64, index: u32, rip: u64) -> bool {
        self.load_dst(insn, dst, RHS, size, next, index, rip);
        self.c.i64_const(0).local_set(LHS);
        self.c.local_get(LHS).local_get(RHS).op(op::I64_SUB);
        self.mask_to(size);
        self.c.local_set(RES);
        let stored = self.store_dst(dst, size, RES, index, rip);
        self.record_flags(LazyFlagsOp::Sub, FusedOp::Sub, size, true, RHS);
        stored
    }

    /// `not`: no flags change.
    fn not(&mut self, insn: &Insn, dst: Dst, size: u8, next: u64, index: u32, rip: u64) -> bool {
        self.load_dst(insn, dst, SCRATCH, size, next, index, rip);
        self.c.local_get(SCRATCH).i64_const(-1).op(op::I64_XOR);
        self.mask_to(size);
        self.c.local_set(SCRATCH_B);
        self.store_dst(dst, size, SCRATCH_B, index, rip)
    }

    /// `shl`/`shr`/`sar` of a 32- or 64-bit operand by an immediate (`count` is Some: already
    /// masked and non-zero) or by CL. The flags are recorded as `LazyFlagsOp::SzpCfOf` with CF the
    /// last bit shifted out and OF as `Cpu::shift_rotate` computes it for every count: shl = MSB
    /// of the result xor CF, shr = MSB of the original, sar = 0. A CL count of zero changes
    /// neither flags nor value; the destination is still written, as the interpreter does (a
    /// 32-bit destination zero-extends, which the zero-extending load did).
    #[allow(clippy::too_many_arguments)]
    fn shift(
        &mut self,
        insn: &Insn,
        kind: ShiftKind,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
    ) -> bool {
        let bits = 8 * size as i64;
        self.load_dst(insn, dst, LHS, size, next, index, rip);
        // RES = the value as the destination will hold it if the count is zero.
        self.c.local_get(LHS).local_set(RES);
        match count {
            Some(count) => {
                self.c.i64_const(count as i64).local_set(SCRATCH);
                self.shift_value(kind, size, bits);
            }
            None => {
                self.read_reg(GUEST_RCX, false, 8);
                self.c
                    .i64_const(bits - 1)
                    .op(op::I64_AND)
                    .local_set(SCRATCH);
                self.c
                    .local_get(SCRATCH)
                    .op(op::I64_EQZ)
                    .op(op::I32_EQZ)
                    .begin_if(BLOCK_VOID);
                self.shift_value(kind, size, bits);
                self.end();
            }
        }
        let stored = self.store_dst(dst, size, RES, index, rip);
        if count.is_none() {
            self.c
                .local_get(SCRATCH)
                .op(op::I64_EQZ)
                .op(op::I32_EQZ)
                .begin_if(BLOCK_VOID);
            self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
            self.end();
        } else {
            self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        }
        self.flags = None;
        stored
    }

    /// With the count in `SCRATCH` (1..bits) and the operand in `LHS`: sets `RES` to the shifted
    /// value and `SCRATCH_B` to the `SzpCfOf` word (CF in bit 0, OF in bit 1).
    fn shift_value(&mut self, kind: ShiftKind, size: u8, bits: i64) {
        // The value to shift: sar works on the sign-extended operand.
        self.c.local_get(LHS);
        if kind == ShiftKind::Sar
            && let Some(extend) = sign_extend_opcode(size)
        {
            self.c.op(extend);
        }
        self.c.local_set(VALUE);
        // CF: shl takes bit `bits - count` of the operand, shr and sar bit `count - 1`.
        self.c.local_get(VALUE);
        match kind {
            ShiftKind::Shl => {
                self.c.i64_const(bits).local_get(SCRATCH).op(op::I64_SUB);
            }
            _ => {
                self.c.local_get(SCRATCH).i64_const(1).op(op::I64_SUB);
            }
        }
        self.c
            .op(op::I64_SHR_U)
            .i64_const(1)
            .op(op::I64_AND)
            .local_set(SCRATCH_B);
        self.c.local_get(VALUE).local_get(SCRATCH);
        self.c.op(match kind {
            ShiftKind::Shl => op::I64_SHL,
            ShiftKind::Shr => op::I64_SHR_U,
            ShiftKind::Sar => op::I64_SHR_S,
        });
        self.mask_to(size);
        self.c.local_set(RES);
        match kind {
            ShiftKind::Shl => {
                // OF = MSB(result) ^ CF
                self.c
                    .local_get(RES)
                    .i64_const(bits - 1)
                    .op(op::I64_SHR_U)
                    .local_get(SCRATCH_B)
                    .op(op::I64_XOR)
                    .i64_const(SHIFT_OF_BIT)
                    .op(op::I64_SHL)
                    .local_get(SCRATCH_B)
                    .op(op::I64_OR)
                    .local_set(SCRATCH_B);
            }
            ShiftKind::Shr => {
                // OF = MSB(original)
                self.c
                    .local_get(LHS)
                    .i64_const(bits - 1)
                    .op(op::I64_SHR_U)
                    .i64_const(SHIFT_OF_BIT)
                    .op(op::I64_SHL)
                    .local_get(SCRATCH_B)
                    .op(op::I64_OR)
                    .local_set(SCRATCH_B);
            }
            ShiftKind::Sar => {}
        }
    }

    /// `imul r, r/m` and the three-operand `imul r, r/m, imm` at 32 bits: the product of the
    /// sign-extended operands fits in 64 bits, so CF = OF = (it differs from its own low 32 bits
    /// sign-extended), which is what the interpreter computes. The 64-bit forms need the high half
    /// of a 128-bit product, which wasm lacks, and go to the interpreter.
    fn imul32(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        let (first, second) = if insn.op == 0x1af {
            (RHS, LHS)
        } else {
            (LHS, RHS)
        };
        match Self::rm_src(insn, 4) {
            Src::Reg(reg, _) => self.read_reg(reg, false, 4),
            _ => {
                self.ea(insn, next);
                self.mem_load(4, false, index, rip, false);
            }
        }
        self.c.op(op::I64_EXTEND32_S).local_set(first);
        if insn.op == 0x1af {
            self.read_reg(insn.reg, false, 4);
            self.c.op(op::I64_EXTEND32_S).local_set(second);
        } else {
            self.c.i64_const(insn.imm as i32 as i64).local_set(second);
        }
        self.c
            .local_get(LHS)
            .local_get(RHS)
            .op(op::I64_MUL)
            .local_set(VALUE);
        // truncated result
        self.c
            .local_get(VALUE)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .local_set(RES);
        self.write_reg(insn.reg, false, 4, RES);
        // CF = OF = 3 when the product does not fit
        self.c
            .local_get(VALUE)
            .local_get(VALUE)
            .op(op::I64_EXTEND32_S)
            .op(op::I64_NE)
            .op(op::I64_EXTEND_I32_U)
            .i64_const(CF_AND_OF)
            .op(op::I64_MUL)
            .local_set(SCRATCH_B);
        self.store_lazy(LazyFlagsOp::SzpCfOf, 4, SCRATCH_B);
        self.flags = None;
    }

    /// The 64-bit `imul r, r/m` and `imul r, r/m, imm` when their flags are dead
    /// (`flags_dead_after`): the low half of the product is one `i64.mul`, and CF/OF, which need
    /// the high half wasm has no instruction for, are never computed. The flags the record holds
    /// are then stale, which is fine because the instruction that overwrites them comes before
    /// anything that could look.
    fn imul64_without_flags(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        match Self::rm_src(insn, 8) {
            Src::Reg(reg, _) => self.read_reg(reg, false, 8),
            _ => {
                self.ea(insn, next);
                self.mem_load(8, false, index, rip, false);
            }
        }
        if insn.op == 0x1af {
            self.read_reg(insn.reg, false, 8);
        } else {
            self.c.i64_const(insn.imm as i32 as i64);
        }
        self.c.op(op::I64_MUL).local_set(VALUE);
        self.write_reg(insn.reg, false, 8, VALUE);
        self.flags = None;
    }

    /// `rol`/`ror` of a 32- or 64-bit operand by an immediate (`count`: masked, non-zero) or CL,
    /// when nothing reads the flags they would change (`flags_dead_after`). The wasm rotates take
    /// their count modulo the width, which is the x86 count mask at these sizes; a CL count of
    /// zero still writes the destination, zero-extending a 32-bit one, as the interpreter does.
    #[allow(clippy::too_many_arguments)]
    fn rotate_without_flags(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
    ) -> bool {
        self.load_dst(insn, dst, VALUE, size, next, index, rip);
        self.c.local_get(VALUE);
        if size == 8 {
            match count {
                Some(count) => {
                    self.c.i64_const(count as i64);
                }
                None => self.read_reg(GUEST_RCX, false, 1),
            }
            self.c.op(if left { op::I64_ROTL } else { op::I64_ROTR });
        } else {
            self.c.op(op::I32_WRAP_I64);
            match count {
                Some(count) => {
                    self.c.i32_const(count as i32);
                }
                None => {
                    self.c
                        .local_get(CPU)
                        .mem(op::I32_LOAD8_U, register_offset(GUEST_RCX, false));
                }
            }
            self.c
                .op(if left { op::I32_ROTL } else { op::I32_ROTR })
                .op(op::I64_EXTEND_I32_U);
        }
        self.c.local_set(SCRATCH_B);
        self.flags = None;
        self.store_dst(dst, size, SCRATCH_B, index, rip)
    }

    fn jcc(&mut self, cc: u8, target: u64, next: u64, retired: u32) {
        self.condition(cc);
        self.c.begin_if(BLOCK_VOID);
        self.chain_to(target, retired);
        self.end();
        self.chain_to(next, retired);
    }

    /// `push` of the value in local `VALUE` onto the guest stack. `rsp` is updated only after the
    /// store succeeded.
    fn push_value(&mut self, index: u32, rip: u64) {
        self.read_reg(GUEST_RSP, false, 8);
        self.c
            .i64_const(-PUSH_POP_SIZE)
            .op(op::I64_ADD)
            .local_set(ADDR);
        self.mem_store(8, VALUE, index, rip, true);
        self.c
            .local_get(CPU)
            .local_get(ADDR)
            .mem(op::I64_STORE, register_offset(GUEST_RSP, false));
    }

    /// `pop` into local `VALUE`; `rsp` moves after the load succeeded.
    fn pop_value(&mut self, index: u32, rip: u64) {
        self.read_reg(GUEST_RSP, false, 8);
        self.c.local_set(ADDR);
        self.mem_load(8, false, index, rip, true);
        self.c.local_set(VALUE);
        self.c
            .local_get(CPU)
            .local_get(ADDR)
            .i64_const(PUSH_POP_SIZE)
            .op(op::I64_ADD)
            .mem(op::I64_STORE, register_offset(GUEST_RSP, false));
    }

    /// Lower instruction `index` inline if it is a supported form. Returns false with nothing
    /// emitted otherwise (callers fall back to `helper_exec`), so every `return false` below must
    /// come before the first emit for that arm. Which forms are lowered mirrors `emit_x64.rs`
    /// except for narrow shifts and BMI, and rotates and 64-bit imul only when their flags are dead
    /// (`flags_dead_after`); adc/sbb are lowered at 32 and 64
    /// bits, as on the native backends.
    fn try_inline(&mut self, index: usize, rip: u64, next: u64) -> bool {
        let insn: Insn = self.insns[index];
        self.skip_flag_record = !self.flag_record_needed[index];
        let k = index as u32;
        let retired = k + 1;
        let osz = insn.opsize;
        let wide = !insn.addr32 && !insn.lock && !insn.p66 && (osz == 4 || osz == 8);
        let is_mem = insn.md != MODRM_MOD_REGISTER;
        let plain = Self::mem_ok(&insn);
        let rex = insn.rex_present;
        let mut store = false;
        match insn.op {
            0x00..=0x3f if insn.op & 7 < 6 && plain => {
                let alu_op = ((insn.op >> 3) & 7) as u8;
                let form = insn.op & 7;
                let size = if form & 1 == 0 { 1 } else { osz };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && size < 4 {
                    return false;
                }
                let (dst, src) = match form {
                    0 | 1 => (Self::rm_dst(&insn, size), Self::reg_src(&insn, size)),
                    2 | 3 => (Self::reg_dst(&insn, size), Self::rm_src(&insn, size)),
                    _ => (
                        Dst::Reg(GUEST_RAX, false),
                        Src::Imm(Self::imm_for(size, insn.imm)),
                    ),
                };
                store = self.alu(&insn, alu_op, dst, src, size, next, k, rip);
            }
            0x80 | 0x81 | 0x83 if plain => {
                let alu_op = insn.reg & 7;
                let size = if insn.op == 0x80 { 1 } else { osz };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && size < 4 {
                    return false;
                }
                store = self.alu(
                    &insn,
                    alu_op,
                    Self::rm_dst(&insn, size),
                    Src::Imm(Self::imm_for(size, insn.imm)),
                    size,
                    next,
                    k,
                    rip,
                );
            }
            0x84 | 0x85 | 0xa8 | 0xa9 if plain => {
                let size = if insn.op & 1 == 0 { 1 } else { osz };
                let (dst, src) = if insn.op >= 0xa8 {
                    (
                        Dst::Reg(GUEST_RAX, false),
                        Src::Imm(Self::imm_for(size, insn.imm)),
                    )
                } else {
                    (Self::rm_dst(&insn, size), Self::reg_src(&insn, size))
                };
                store = self.alu(&insn, ALU_TEST, dst, src, size, next, k, rip);
            }
            0xf6 | 0xf7 if plain && matches!(insn.reg & 7, 0 | 2 | 3) => {
                let size = if insn.op == 0xf6 { 1 } else { osz };
                let dst = Self::rm_dst(&insn, size);
                store = match insn.reg & 7 {
                    0 => self.alu(
                        &insn,
                        ALU_TEST,
                        dst,
                        Src::Imm(Self::imm_for(size, insn.imm)),
                        size,
                        next,
                        k,
                        rip,
                    ),
                    2 => self.not(&insn, dst, size, next, k, rip),
                    _ => self.neg(&insn, dst, size, next, k, rip),
                };
            }
            0xfe | 0xff if plain && insn.reg & 7 <= 1 => {
                let size = if insn.op == 0xfe { 1 } else { osz };
                store = self.inc_dec(
                    &insn,
                    insn.reg & 7 == 1,
                    Self::rm_dst(&insn, size),
                    size,
                    next,
                    k,
                    rip,
                );
            }
            // rol/ror at 32 and 64 bits whose CF and OF nothing reads; the flags they keep
            // (SF, ZF, PF, AF) would need the interpreter to merge.
            0xc1 | 0xd1 | 0xd3
                if wide
                    && matches!(insn.reg & 7, SHIFT_ROL | SHIFT_ROR)
                    && flags_dead_after(self.insns, index) =>
            {
                let count = match insn.op {
                    0xc1 => Some(insn.imm as u32 & (8 * osz as u32 - 1)),
                    0xd1 => Some(1),
                    _ => None,
                };
                if count == Some(0) {
                    return false;
                }
                store = self.rotate_without_flags(
                    &insn,
                    insn.reg & 7 == SHIFT_ROL,
                    Self::rm_dst(&insn, osz),
                    count,
                    osz,
                    next,
                    k,
                    rip,
                );
            }
            0xc1 | 0xd1 | 0xd3 if wide && matches!(insn.reg & 7, SHIFT_SHL..=SHIFT_SAR) => {
                let kind = match insn.reg & 7 {
                    SHIFT_SHL | SHIFT_SAL_ALIAS => ShiftKind::Shl,
                    SHIFT_SHR => ShiftKind::Shr,
                    _ => ShiftKind::Sar,
                };
                let count = match insn.op {
                    0xc1 => Some(insn.imm as u32 & (8 * osz as u32 - 1)),
                    0xd1 => Some(1),
                    _ => None,
                };
                if count == Some(0) {
                    return false;
                }
                store = self.shift(
                    &insn,
                    kind,
                    Self::rm_dst(&insn, osz),
                    count,
                    osz,
                    next,
                    k,
                    rip,
                );
            }
            0x88..=0x8b if plain => {
                let size = if insn.op & 1 == 0 { 1 } else { osz };
                if insn.op < 0x8a {
                    let (src_reg, src_high) = Self::named_register(&insn, insn.reg, size);
                    self.read_reg(src_reg, src_high, size);
                    self.c.local_set(VALUE);
                    match Self::rm_dst(&insn, size) {
                        Dst::Mem => {
                            self.ea(&insn, next);
                            self.mem_store(size, VALUE, k, rip, false);
                            store = true;
                        }
                        Dst::Reg(reg, high) => self.write_reg(reg, high, size, VALUE),
                    }
                } else {
                    let (dst_reg, dst_high) = Self::named_register(&insn, insn.reg, size);
                    match Self::rm_src(&insn, size) {
                        Src::Reg(reg, high) => self.read_reg(reg, high, size),
                        _ => {
                            self.ea(&insn, next);
                            self.mem_load(size, false, k, rip, false);
                        }
                    }
                    self.c.local_set(VALUE);
                    self.write_reg(dst_reg, dst_high, size, VALUE);
                }
            }
            0xc6 | 0xc7 if plain && insn.reg & 7 == 0 => {
                let size = if insn.op == 0xc6 { 1 } else { osz };
                self.c
                    .i64_const(Self::imm_for(size, insn.imm) as i64)
                    .local_set(VALUE);
                match Self::rm_dst(&insn, size) {
                    Dst::Mem => {
                        self.ea(&insn, next);
                        self.mem_store(size, VALUE, k, rip, false);
                        store = true;
                    }
                    Dst::Reg(reg, high) => self.write_reg(reg, high, size, VALUE),
                }
            }
            0xb0..=0xbf if plain => {
                let size = if insn.op < 0xb8 { 1 } else { osz };
                let (reg, high) = Self::named_register(&insn, opcode_register(&insn), size);
                self.c
                    .i64_const(Self::imm_for(size, insn.imm) as i64)
                    .local_set(VALUE);
                self.write_reg(reg, high, size, VALUE);
            }
            // setcc r/m8
            0x190..=0x19f if plain => {
                self.condition((insn.op & 15) as u8);
                self.c.op(op::I64_EXTEND_I32_U).local_set(VALUE);
                match Self::rm_dst(&insn, 1) {
                    Dst::Reg(reg, high) => self.write_reg(reg, high, 1, VALUE),
                    Dst::Mem => {
                        self.ea(&insn, next);
                        self.mem_store(1, VALUE, k, rip, false);
                        store = true;
                    }
                }
            }
            // cmovcc r, r/m at 32 and 64 bits. The source is always read (it can fault) and a
            // 32-bit destination is zero-extended whether or not it moves.
            0x140..=0x14f if wide => {
                match Self::rm_src(&insn, osz) {
                    Src::Reg(reg, _) => self.read_reg(reg, false, osz),
                    _ => {
                        self.ea(&insn, next);
                        self.mem_load(osz, false, k, rip, false);
                    }
                }
                self.read_reg(insn.reg, false, osz);
                self.condition((insn.op & 15) as u8);
                self.c.op(op::SELECT).local_set(VALUE);
                self.write_reg(insn.reg, false, 8, VALUE);
            }
            0x1af | 0x69 | 0x6b if wide && osz == 4 => self.imul32(&insn, next, k, rip),
            0x1af | 0x69 | 0x6b if wide && osz == 8 && flags_dead_after(self.insns, index) => {
                self.imul64_without_flags(&insn, next, k, rip);
            }
            // group 5: /2 call near, /4 jmp near (r/m64), /6 push r/m64
            0xff if plain && !insn.p66 && matches!(insn.reg & 7, 2 | 4 | 6) => {
                match Self::rm_src(&insn, 8) {
                    Src::Reg(reg, _) => self.read_reg(reg, false, 8),
                    _ => {
                        self.ea(&insn, next);
                        self.mem_load(8, false, k, rip, false);
                    }
                }
                match insn.reg & 7 {
                    2 => {
                        // The target is read before the push, so `call [rsp + n]` sees the old rsp.
                        self.c.local_set(SCRATCH);
                        self.rip_to(next);
                        self.c.local_set(VALUE);
                        self.push_value(k, rip);
                        self.chain_to_local(SCRATCH, retired);
                        self.inlined += 1;
                        return true;
                    }
                    4 => {
                        self.c.local_set(SCRATCH);
                        self.chain_to_local(SCRATCH, retired);
                        self.inlined += 1;
                        return true;
                    }
                    _ => {
                        self.c.local_set(VALUE);
                        self.push_value(k, rip);
                        store = true;
                    }
                }
            }
            // lfence, mfence, sfence: nothing to order
            0x1ae if !is_mem && insn.rep == 0 && matches!(insn.reg & 7, 5..=7) => {}
            // cdqe / cwde
            0x98 if wide => {
                if osz == 8 {
                    self.read_reg(GUEST_RAX, false, 4);
                    self.c.op(op::I64_EXTEND32_S);
                } else {
                    self.read_reg(GUEST_RAX, false, 2);
                    self.c
                        .op(op::I64_EXTEND16_S)
                        .i64_const(LOW_DWORD)
                        .op(op::I64_AND);
                }
                self.c.local_set(VALUE);
                self.write_reg(GUEST_RAX, false, 8, VALUE);
            }
            // cqo / cdq: rdx (edx, zero-extended) gets the sign of rax (eax)
            0x99 if wide => {
                self.read_reg(GUEST_RAX, false, osz);
                if osz == 8 {
                    self.c.i64_const(SIGN_FILL_SHIFT).op(op::I64_SHR_S);
                } else {
                    self.c
                        .op(op::I64_EXTEND32_S)
                        .i64_const(SIGN_FILL_SHIFT)
                        .op(op::I64_SHR_S)
                        .i64_const(LOW_DWORD)
                        .op(op::I64_AND);
                }
                self.c.local_set(VALUE);
                self.write_reg(GUEST_RDX, false, 8, VALUE);
            }
            // lea; `ea` would add a segment base, which lea must not
            0x8d if wide && is_mem => {
                if has_fs_gs_override(&insn) {
                    return false;
                }
                self.ea(&insn, next);
                self.c.local_get(ADDR);
                if osz == 4 {
                    self.c.i64_const(LOW_DWORD).op(op::I64_AND);
                }
                self.c.local_set(VALUE);
                self.write_reg(insn.reg, false, 8, VALUE);
            }
            // movsxd r64, r/m32
            0x63 if osz == 8 && !insn.addr32 && !insn.lock && !insn.p66 && (!is_mem || plain) => {
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(4, true, k, rip, false);
                } else {
                    self.read_reg(insn.rm, false, 4);
                    self.c.op(op::I64_EXTEND32_S);
                }
                self.c.local_set(VALUE);
                self.write_reg(insn.reg, false, 8, VALUE);
            }
            // movzx (0f b6, b7) / movsx (0f be, bf) from a byte (even opcode) or word source
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if wide && (!is_mem || plain) => {
                let source_size: u8 = if insn.op & 1 == 0 { 1 } else { 2 };
                let signed = insn.op >= 0x1b8;
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(source_size, signed, k, rip, false);
                } else {
                    if source_size == 1 && !rex && HIGH_BYTE_REGS.contains(&insn.rm) {
                        return false;
                    }
                    self.read_reg(insn.rm, false, source_size);
                    if signed && let Some(extend) = sign_extend_opcode(source_size) {
                        self.c.op(extend);
                    }
                }
                if osz == 4 {
                    self.c.i64_const(LOW_DWORD).op(op::I64_AND);
                }
                self.c.local_set(VALUE);
                self.write_reg(insn.reg, false, 8, VALUE);
            }
            // push r64. The register is read before rsp is stored, so `push rsp` pushes the old
            // value.
            0x50..=0x57 if !insn.p66 => {
                self.read_reg(opcode_register(&insn), false, 8);
                self.c.local_set(VALUE);
                self.push_value(k, rip);
                store = true;
            }
            // push imm32 / imm8, sign-extended to 64 bits by the decoder.
            0x68 | 0x6a if !insn.p66 && !insn.lock => {
                self.c.i64_const(insn.imm).local_set(VALUE);
                self.push_value(k, rip);
                store = true;
            }
            // pop r64. The register is written last so `pop rsp` ends up with the popped value.
            0x58..=0x5f if !insn.p66 => {
                self.pop_value(k, rip);
                self.write_reg(opcode_register(&insn), false, 8, VALUE);
            }
            // nop, prefetchw (0f 0d), and the 0f 18..1f hint/nop group. 0x90 with REX.B is xchg
            // r8, rax, and `f3 90` is pause, which an SVM guest can intercept.
            0x90 | 0x10d | 0x118..=0x11f => {
                if insn.op == 0x90 && (insn.rex & GUEST_REX_B != 0 || insn.rep != 0) {
                    return false;
                }
            }
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next.wrapping_add(insn.imm as u64);
                self.jcc((insn.op & 15) as u8, target, next, retired);
                self.inlined += 1;
                return true;
            }
            0xe9 | 0xeb => {
                self.chain_to(next.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            // call rel32. The block ends here, so no `smc_check` is needed after the push.
            0xe8 => {
                self.rip_to(next);
                self.c.local_set(VALUE);
                self.push_value(k, rip);
                self.chain_to(next.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            0xc3 => {
                self.pop_value(k, rip);
                self.chain_to_local(VALUE, retired);
                self.inlined += 1;
                return true;
            }
            _ => return false,
        }
        if store {
            self.smc_check(k, next);
        }
        self.inlined += 1;
        true
    }

    /// `rips[k]` is the guest rip of instruction `k`; `end_rip` is the rip after the last one.
    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let count = self.insns.len();
        self.entry_rip = rips[0];
        self.c
            .local_get(CPU)
            .mem(op::I32_LOAD, TLB_ACTIVE_BASE)
            .local_set(TLB_BASE);
        self.c
            .local_get(CPU)
            .mem(op::I64_LOAD, RIP)
            .i64_const(PAGE_MASK_NEG)
            .op(op::I64_AND)
            .local_set(PAGE_BASE);
        for index in 0..count {
            let rip = rips[index];
            let next = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next) {
                if is_inline_terminator(&self.insns[index]) {
                    self.end();
                    return;
                }
                continue;
            }
            self.helper(index, rip, terminal);
            if terminal {
                self.end();
                return;
            }
        }
        self.chain_to(end_rip, count as u32);
        self.end();
    }
}

/// Generates the module for one block. `insns` must outlive the code, which points into it, and
/// so must the `LINKS_PER_BLOCK` cells at `link_cells` (0: the exits are not linked).
pub(crate) fn compile_block(
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
    link_cells: u32,
) -> CompiledBlock {
    let mut generator = Gen::new(addrs, insns, link_cells);
    generator.compile(rips, end_rip);
    CompiledBlock {
        module: block_module(&generator.c.bytes, I32_LOCALS, I64_LOCALS),
        inlined: generator.inlined,
        flag_records_skipped: generator.flag_records_skipped,
    }
}

/// Inside the chain stub's `if` for a jump cache entry whose physical address is not that of
/// `stub_rip`: falls out of the `if` (and on to the slot and validity checks) only when the entry
/// is the straddling block of this address and the page after `stub_rip`'s is still mapped
/// executable to the physical page that block was decoded from, as `Jit::lookup_straddler` checks;
/// otherwise branches to the stub's miss path (`br 1`). Without this, every run of a straddler
/// (a hot loop that crosses a page) returned to the run loop. Reuses `tlb_entry` and `stub_phys`,
/// which nothing after it reads.
fn straddler_check(
    c: &mut Code,
    tables: &ChainTables,
    cache_entry: u32,
    tlb_entry: u32,
    stub_rip: u32,
    stub_phys: u32,
) {
    const MISS: u32 = 1;
    let next_page = stub_phys;
    c.local_get(cache_entry)
        .mem(op::I64_LOAD, tables.entry_phys)
        .local_get(stub_phys)
        .i64_const(STRADDLER_TAG as i64)
        .op(op::I64_OR)
        .op(op::I64_NE)
        .br_if(MISS);
    c.local_get(stub_rip)
        .i64_const(PAGE_MASK_NEG)
        .op(op::I64_AND)
        .i64_const(PAGE_SIZE as i64)
        .op(op::I64_ADD)
        .local_set(next_page);
    c.local_get(next_page)
        .i64_const(PAGE_SHIFT)
        .op(op::I64_SHR_U)
        .i64_const(TLB_ENTRIES_PER_MODE as i64 - 1)
        .op(op::I64_AND)
        .op(op::I32_WRAP_I64)
        .i32_const(TLB_ENTRY_SHIFT)
        .op(op::I32_SHL)
        .local_get(CPU)
        .mem(op::I32_LOAD, TLB_ACTIVE_BASE)
        .op(op::I32_ADD)
        .local_set(tlb_entry);
    c.local_get(tlb_entry)
        .mem(op::I64_LOAD, TLB_TAG_EXEC)
        .local_get(next_page)
        .op(op::I64_NE)
        .br_if(MISS);
    // the second page's physical page number against the one kept above the block index
    c.local_get(next_page)
        .local_get(tlb_entry)
        .mem(op::I64_LOAD, TLB_ADDEND)
        .op(op::I64_ADD)
        .local_get(CPU)
        .mem(op::I64_LOAD, RAM_BASE)
        .op(op::I64_SUB)
        .i64_const(PAGE_SHIFT)
        .op(op::I64_SHR_U)
        .local_get(cache_entry)
        .mem(op::I64_LOAD, tables.entry_block)
        .i64_const(SLOT_BLOCK_BITS as i64)
        .op(op::I64_SHR_U)
        .op(op::I64_NE)
        .br_if(MISS);
}

/// Where the chain stub finds what it looks blocks up in: the jump cache and the validity table
/// (`wasm.rs`), as addresses and the layout of one cache entry.
pub(crate) struct ChainTables {
    pub jump_cache: u32,
    pub valid: u32,
    /// log2 of the size of one jump cache entry, and the offsets of its fields.
    pub entry_shift: u32,
    pub entry_phys: u32,
    pub entry_block: u32,
    pub entry_slot: u32,
    /// log2 of the number of entries.
    pub index_bits: u32,
}

/// The module of the chain stub, a `block`-typed function every block tail-calls to continue at
/// `Cpu::rip`, the equivalent of `arch_x64::chain_stub`. It is entered with `icount` already
/// charged for the successor's first instruction (which `Cpu::run` would have charged) and
/// `Cpu::rip` set, and tail-calls the successor when all of these hold, else leaves for the run
/// loop, which then takes the normal path (event servicing, interrupts, page walk, translation):
///
/// - `icount` has not reached `Cpu::next_event`, so the device tick and interrupt check the loop
///   does at block boundaries are not starved;
/// - `block_exit_requested` is clear: a store of the block just left hit a page holding
///   translated code, and the successor may be stale;
/// - the exec TLB half the blocks run in has the page, so the physical address is known without
///   a walk and permissions were already checked;
/// - the jump cache has a block for exactly this rip and physical address (the key `Jit::lookup`
///   uses), and it was not invalidated since.
///
/// The miss path gives back the successor's pre-charge, as `native.rs` documents for
/// `Cpu::step_block`, and reports one retired instruction.
///
/// With `link` the stub is the `LINK_TYPE` variant a linkable exit calls (`Gen::linked_exit`):
/// its second parameter is the exit's link cell, which a hit fills with the block it enters.
pub(crate) fn compile_chain_stub(tables: &ChainTables, link: bool) -> Vec<u8> {
    // parameters: 0 cpu, then the link cell for the linking variant; locals after them: i32 tlb
    // entry, jump cache entry, block slot; i64 rip, phys
    const LINK_CELL: u32 = 1;
    let first_local = if link { 2 } else { 1 };
    let tlb_entry = first_local;
    let cache_entry = first_local + 1;
    let block_slot = first_local + 2;
    let stub_rip = first_local + 3;
    let stub_phys = first_local + 4;
    const JUMP_HASH_FOLD_SHIFT: i64 = 16;
    let mut c = Code::default();
    c.begin_block();
    // the event budget
    c.local_get(CPU)
        .mem(op::I64_LOAD, ICOUNT)
        .local_get(CPU)
        .mem(op::I64_LOAD, NEXT_EVENT)
        .op(op::I64_GE_U)
        .br_if(0);
    c.local_get(CPU)
        .mem(op::I32_LOAD8_U, BLOCK_EXIT_REQUESTED)
        .br_if(0);
    c.local_get(CPU).mem(op::I64_LOAD, RIP).local_set(stub_rip);
    // the exec TLB entry of rip's page
    c.local_get(stub_rip)
        .i64_const(PAGE_SHIFT)
        .op(op::I64_SHR_U)
        .i64_const(TLB_ENTRIES_PER_MODE as i64 - 1)
        .op(op::I64_AND)
        .op(op::I32_WRAP_I64)
        .i32_const(TLB_ENTRY_SHIFT)
        .op(op::I32_SHL)
        .local_get(CPU)
        .mem(op::I32_LOAD, TLB_ACTIVE_BASE)
        .op(op::I32_ADD)
        .local_set(tlb_entry);
    c.local_get(tlb_entry)
        .mem(op::I64_LOAD, TLB_TAG_EXEC)
        .local_get(stub_rip)
        .i64_const(PAGE_MASK_NEG)
        .op(op::I64_AND)
        .op(op::I64_NE)
        .br_if(0);
    // phys = rip + addend - ram_base
    c.local_get(stub_rip)
        .local_get(tlb_entry)
        .mem(op::I64_LOAD, TLB_ADDEND)
        .op(op::I64_ADD)
        .local_get(CPU)
        .mem(op::I64_LOAD, RAM_BASE)
        .op(op::I64_SUB)
        .local_set(stub_phys);
    // the jump cache entry: index = (rip ^ (rip >> 16)) & mask, as `jump_cache_hash`
    c.local_get(stub_rip)
        .local_get(stub_rip)
        .i64_const(JUMP_HASH_FOLD_SHIFT)
        .op(op::I64_SHR_U)
        .op(op::I64_XOR)
        .i64_const((1 << tables.index_bits) - 1)
        .op(op::I64_AND)
        .op(op::I32_WRAP_I64)
        .i32_const(tables.entry_shift as i32)
        .op(op::I32_SHL)
        .i32_const(tables.jump_cache as i32)
        .op(op::I32_ADD)
        .local_set(cache_entry);
    c.local_get(cache_entry)
        .mem(op::I64_LOAD, 0)
        .local_get(stub_rip)
        .op(op::I64_NE)
        .br_if(0);
    c.local_get(cache_entry)
        .mem(op::I64_LOAD, tables.entry_phys)
        .local_get(stub_phys)
        .op(op::I64_NE)
        .begin_if(BLOCK_VOID);
    straddler_check(&mut c, tables, cache_entry, tlb_entry, stub_rip, stub_phys);
    c.op(op::END);
    c.local_get(cache_entry)
        .mem(op::I32_LOAD, tables.entry_slot)
        .local_tee(block_slot)
        .op(op::I32_EQZ)
        .br_if(0);
    // the block may have been invalidated since the entry was made (a straddler's index has its
    // second page above it)
    c.local_get(cache_entry)
        .mem(op::I32_LOAD, tables.entry_block)
        .i32_const(SLOT_BLOCK_MASK as i32)
        .op(op::I32_AND)
        .i32_const(tables.valid as i32)
        .op(op::I32_ADD)
        .mem(op::I32_LOAD8_U, 0)
        .op(op::I32_EQZ)
        .br_if(0);
    // A straddler is never linked: a linked exit skips the second page's check.
    if link {
        c.local_get(cache_entry)
            .mem(op::I64_LOAD, tables.entry_phys)
            .i64_const(0)
            .op(op::I64_GE_S)
            .begin_if(BLOCK_VOID);
        c.local_get(LINK_CELL)
            .local_get(cache_entry)
            .mem(op::I32_LOAD, tables.entry_block)
            .mem(op::I32_STORE, LINK_CELL_BLOCK);
        c.local_get(LINK_CELL)
            .local_get(block_slot)
            .mem(op::I32_STORE, LINK_CELL_SLOT);
        c.op(op::END);
    }
    c.local_get(CPU)
        .local_get(block_slot)
        .return_call_indirect(BLOCK_TYPE);
    c.op(op::END);
    // miss: give back the pre-charge and return to the run loop (rip is already stored)
    c.local_get(CPU)
        .local_get(CPU)
        .mem(op::I64_LOAD, ICOUNT)
        .i64_const(1)
        .op(op::I64_SUB)
        .mem(op::I64_STORE, ICOUNT);
    c.i64_const(1 << EXIT_COUNT_SHIFT)
        .op(op::RETURN)
        .op(op::END);
    module_of_type(if link { LINK_TYPE } else { BLOCK_TYPE }, &c.bytes, 3, 2)
}
