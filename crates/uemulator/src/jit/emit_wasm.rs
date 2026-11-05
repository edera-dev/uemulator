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
//!   computed from them (`Fused`) instead of calling `helper_cond`. The record of an add, sub,
//!   logic op, inc, dec or neg is stored only where something can read it (`Gen::pending_record`):
//!   before a helper call, on a slow-path access and on every way out of the block.
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
    BLOCK_I64, BLOCK_TYPE, BLOCK_VOID, COND_TYPE, Code, EXEC_TYPE, Function, LINK_TYPE, LOAD_TYPE,
    STORE_TYPE, module_of_type, op, simd,
};
use crate::cpu::decode::{Insn, NO_REG, VEX_OP_FLAG};
use crate::cpu::flags::LazyFlagsOp;
use crate::cpu::xsave::{XCR0_AVX, XCR0_AVX512, XCR0_SSE};
use crate::cpu::{
    CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXSAVE, Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE,
    TlbEntry,
};

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
const CR0: u32 = offset_of!(Cpu, cr0) as u32;
const CR4: u32 = offset_of!(Cpu, cr4) as u32;
const XMM: u32 = offset_of!(Cpu, xmm) as u32;
const XMM_SIZE: u32 = size_of::<u128>() as u32;
const XMM_HALF: u32 = XMM_SIZE / 2;
/// Legacy-SSE 128-bit memory operands must be aligned to this.
const VECTOR_ALIGNMENT: i64 = XMM_SIZE as i64;
const XCR0: u32 = offset_of!(Cpu, xcr0) as u32;
/// What `Cpu::evex_gate` requires of XCR0: the SSE, AVX and AVX-512 state components.
const XCR0_EVEX_STATE: u64 = XCR0_SSE | XCR0_AVX | XCR0_AVX512;
/// A vector register is four 128-bit lanes (`helpers::vector_lane_offset`), each two i64 halves.
const VECTOR_LANES: u32 = 4;
const LANE_BYTES: u32 = 16;
const HALF_LANE_BYTES: u32 = 8;
const OPMASK: u32 = offset_of!(Cpu, opmask) as u32;
const OPMASK_SIZE: u32 = size_of::<u64>() as u32;
/// k0-k7: the register fields of a kmov name one with their low three bits.
const OPMASK_REGISTER_MASK: u8 = 7;
/// `Insn::op` of VEX `0f 92` (kmov k, r) and `0f 93` (kmov r, k); with F2, kmovd/kmovq.
const VEX_KMOV_FROM_GENERAL: u16 = VEX_OP_FLAG | 0x192;
const VEX_KMOV_TO_GENERAL: u16 = VEX_OP_FLAG | 0x193;
/// `Insn::rep` of an F2 prefix.
const PREFIX_F2: u8 = 2;
/// The vpcmp immediate's predicates (SDM table of VPCMP), in order: eq, lt, le, false, ne, nlt,
/// nle, true.
const VPCMP_FALSE: u8 = 3;
const VPCMP_TRUE: u8 = 7;
/// The bits of an i8x16.bitmask: one per byte of a 128-bit lane.
const LANE_MASK_BITS: i64 = 16;
/// `66 0f 6e` (movd/movq xmm, r/m) and `66 0f 7e` (movd/movq r/m, xmm).
const MOVD_TO_VECTOR: u16 = 0x16e;
const MOVD_FROM_VECTOR: u16 = 0x17e;
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
/// Shift counts of 8-, 16- and 32-bit operands are masked to five bits, not to the width (SDM,
/// SHL), so a narrow shift can move every bit out.
const NARROW_SHIFT_COUNT_MASK: i64 = 0x1f;
const WIDE_SHIFT_COUNT_MASK: i64 = 0x3f;

/// `Insn::op` of the VEX map-2 (0f38) BMI opcodes: bextr/shlx/sarx/shrx, bzhi/pdep/pext, andn,
/// and the blsr/blsmsk/blsi group (`cpu/bmi.rs`).
const VEX_BMI_SHIFT_OR_BEXTR: u16 = VEX_OP_FLAG | 0x2f7;
const VEX_BMI_BZHI_PEXT_PDEP: u16 = VEX_OP_FLAG | 0x2f5;
const VEX_BMI_ANDN: u16 = VEX_OP_FLAG | 0x2f2;
const VEX_BMI_GROUP17: u16 = VEX_OP_FLAG | 0x2f3;
/// mulx (with F2; the other prefixes are not instructions there).
const VEX_BMI_MULX: u16 = VEX_OP_FLAG | 0x2f6;
const BMI_GROUP_BLSR: u8 = 1;
const BMI_GROUP_BLSMSK: u8 = 2;
const BMI_GROUP_BLSI: u8 = 3;
/// bzhi reads its index from bits 7:0 of vvvv.
const BZHI_INDEX_MASK: i64 = 0xff;

/// A 64-bit operand as two 32-bit limbs, for the high half of a product (`Gen::mul_high`).
const LIMB_BITS: i64 = 32;

/// Byte-swap masks: every other byte, every other 16-bit word.
const ALTERNATE_BYTES_64: i64 = 0x00ff_00ff_00ff_00ff;
const ALTERNATE_WORDS_64: i64 = 0x0000_ffff_0000_ffff;
const ALTERNATE_BYTES_32: i64 = 0x00ff_00ff;

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
    /// `helper_set_rotate_flags`, called through `COND_TYPE` (its result is dropped).
    pub rotate_flags: u32,
    /// `helper_whole_vector`, called through `STORE_TYPE` (its result is dropped).
    pub whole_vector: u32,
    pub out: u32,
    /// Table slot of the chain stub, or 0 when blocks return to the run loop between each other.
    pub chain: u32,
    /// Table slot of the chain stub that also fills a link cell (`compile_chain_stub`), or 0
    /// when exits are not linked.
    pub chain_link: u32,
    /// Address of `SharedJit::valid`, the per-block validity bytes a linked exit checks.
    pub valid: u32,
    /// The host compiles fixed-width SIMD, which the vector lowerings other than moves need.
    pub simd: bool,
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
/// Exits to fixed addresses a block can link: one per conditional branch before its last
/// instruction (blocks run on past them, `blocks::decode_block`), plus two at its end (a jcc's
/// two outcomes, or a jmp, a call or the fall-through). Past `MAX_LINKS_PER_BLOCK` the rest go
/// through the chain stub unlinked.
pub(crate) fn link_cells_for(insns: &[Insn]) -> u32 {
    const END_EXITS: u32 = 2;
    let side_exits = insns[..insns.len() - 1]
        .iter()
        .filter(|insn| is_conditional_branch(insn))
        .count() as u32;
    (side_exits + END_EXITS).min(MAX_LINKS_PER_BLOCK)
}
const MAX_LINKS_PER_BLOCK: u32 = 8;
/// Page offsets below this are in the block's own page (`Gen::rip_to`).
const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;

/// What `compile_block` produces.
pub(crate) struct CompiledBlock {
    /// The block's function body, a `BLOCK_TYPE` function with `I32_LOCALS` and `I64_LOCALS`
    /// (`function`); `wasm_encode::module_of_functions` wraps one or several in a module.
    pub code: Vec<u8>,
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
enum BitScan {
    Bsf,
    Bsr,
    Tzcnt,
    Lzcnt,
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
    /// The lazy record of the last flag writer, not stored yet: its operation and size, with
    /// `RES` and this local (`RHS`, or `CARRY` for inc/dec) holding its values. Stored where
    /// something can read the record (`Gen::flush_flags`, `Gen::materialize_flags`), and dropped
    /// when the next flag writer replaces it, which is most of the time.
    pending_record: Option<(LazyFlagsOp, u8, u32)>,
    pub flag_records_skipped: u32,
    /// Address of the block's `link_count` link cells (`link_cells_for`), or 0 when its exits
    /// are not linked.
    link_cells: u32,
    link_count: u32,
    links_used: u32,
    /// Whether the instruction being lowered is the block's last: a conditional branch anywhere
    /// else is a side exit, and the block goes on after it.
    last: bool,
    /// The block jumps back to its own first instruction, and its body is a wasm `loop` opened
    /// at this `Code::depth` (`Gen::back_edge`).
    self_loop: Option<u32>,
    /// An inline SSE (or AVX-512) instruction already checked `Cpu::sse`'s (`Cpu::evex_gate`'s)
    /// conditions on the way here: the block's later ones skip the check. What changes CR0, CR4
    /// or XCR0 ends a block, and an earlier check that failed left it through the interpreter.
    sse_gated: bool,
    evex_gated: bool,
}

impl<'a> Gen<'a> {
    pub fn new(addrs: &'a Addrs, insns: &'a [Insn], link_cells: u32, link_count: u32) -> Self {
        Gen {
            c: Code::default(),
            addrs,
            insns,
            flags: None,
            prepaid_insns: 0,
            inlined: 0,
            entry_rip: 0,
            pending_record: None,
            flag_records_skipped: 0,
            link_cells,
            link_count,
            links_used: 0,
            last: false,
            self_loop: None,
            sse_gated: false,
            evex_gated: false,
        }
    }

    fn end(&mut self) {
        self.c.op(op::END);
    }

    /// Closes the function body, and the self-loop around it if there is one.
    fn finish(&mut self) {
        if self.self_loop.is_some() {
            self.end();
        }
        self.end();
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
        self.materialize_flags();
        if let Some(loop_depth) = self.self_loop
            && rip == self.entry_rip
        {
            self.back_edge(loop_depth, retired);
            return;
        }
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
            || self.links_used == self.link_count
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

    /// Goes round a block that jumps back to its own start again without leaving it: the exit
    /// a linked exit to itself would be, minus the tail call. It pays `icount` the same way and
    /// makes the same budget and `block_exit_requested` checks, leaving through the chain stub
    /// (which then returns to the run loop) when one fails, with `Cpu::rip` stored only then.
    /// The validity check is not needed: only a write to the block's page invalidates it, and
    /// that sets `block_exit_requested`, which `smc_check` after a store, a helper's status and
    /// this check all see. The loop's head starts from the state a block starts from: no fused
    /// flags (the lazy record is always stored before an exit) and nothing prepaid (paid here).
    fn back_edge(&mut self, loop_depth: u32, retired: u32) {
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
        let label = self.c.depth() - loop_depth;
        self.c.br(label);
        self.end();
        self.store_rip(self.entry_rip);
        self.c
            .local_get(CPU)
            .i32_const(self.addrs.chain as i32)
            .return_call_indirect(BLOCK_TYPE);
    }

    /// Ends the block at the guest address in local `target`.
    fn chain_to_local(&mut self, target: u32, retired: u32) {
        self.materialize_flags();
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
        self.flush_flags();
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

    /// Runs instruction `insn_index` through the interpreter from inside a branch, as the slow
    /// path of an inline lowering, leaving the generator's state as the fast path leaves it: the
    /// flags (the instruction must leave them alone), the fused locals, and the prepaid clock,
    /// which `helper` would advance for good. Here the clock is prepaid only for the call and
    /// handed back after it, so the code after the merge sees the same `icount` on both paths.
    fn helper_in_branch(&mut self, insn_index: usize, rip: u64) {
        self.materialize_flags();
        let insn = &self.insns[insn_index] as *const Insn as usize as i32;
        let owed = insn_index as u32 - self.prepaid_insns;
        if owed > 0 {
            self.add_to_icount(owed as i64);
            self.store_cpu_u64(PREPAID_ICOUNT, insn_index as u64);
        }
        self.c.local_get(CPU).i32_const(insn);
        self.rip_to(rip);
        self.c
            .i32_const(self.addrs.exec as i32)
            .call_indirect(EXEC_TYPE);
        self.check_status(insn_index as u32, true);
        if owed > 0 {
            self.add_to_icount(-(owed as i64));
            self.store_cpu_u64(PREPAID_ICOUNT, self.prepaid_insns as u64);
        }
    }

    fn add_to_icount(&mut self, count: i64) {
        self.c
            .local_get(CPU)
            .local_get(CPU)
            .mem(op::I64_LOAD, ICOUNT)
            .i64_const(count)
            .op(op::I64_ADD)
            .mem(op::I64_STORE, ICOUNT);
    }

    /// Pushes an i32 that is 1 when `Cpu::sse` (with `evex`, `Cpu::evex_gate`) would let a vector
    /// instruction run, and returns true; returns false, pushing nothing, when an earlier
    /// instruction of the block checked it already.
    fn vector_gate(&mut self, evex: bool) -> bool {
        if evex {
            if self.evex_gated {
                return false;
            }
            self.evex_gated = true;
            // CR4.OSXSAVE, the AVX-512 state in XCR0, and no CR0.TS
            self.c
                .local_get(CPU)
                .mem(op::I64_LOAD, CR4)
                .i64_const(CR4_OSXSAVE as i64)
                .op(op::I64_AND)
                .op(op::I64_EQZ)
                .op(op::I32_EQZ)
                .local_get(CPU)
                .mem(op::I64_LOAD, XCR0)
                .i64_const(XCR0_EVEX_STATE as i64)
                .op(op::I64_AND)
                .i64_const(XCR0_EVEX_STATE as i64)
                .op(op::I64_EQ)
                .op(op::I32_AND)
                .local_get(CPU)
                .mem(op::I64_LOAD, CR0)
                .i64_const(CR0_TS as i64)
                .op(op::I64_AND)
                .op(op::I64_EQZ)
                .op(op::I32_AND);
        } else {
            if self.sse_gated {
                return false;
            }
            self.sse_gated = true;
            // neither CR0.EM nor CR0.TS, and CR4.OSFXSR
            self.c
                .local_get(CPU)
                .mem(op::I64_LOAD, CR0)
                .i64_const((CR0_EM | CR0_TS) as i64)
                .op(op::I64_AND)
                .op(op::I64_EQZ)
                .local_get(CPU)
                .mem(op::I64_LOAD, CR4)
                .i64_const(CR4_OSFXSR as i64)
                .op(op::I64_AND)
                .op(op::I64_EQZ)
                .op(op::I32_EQZ)
                .op(op::I32_AND);
        }
        true
    }

    /// A whole-vector move of `bytes` (16, 32 or 64) between vector registers or a vector
    /// register and memory: the legacy movups/movaps/movdqu/movdqa (`evex` false: only the low
    /// lane is written) and the unmasked EVEX vmovdqu*/vmovdqa*/vmovup*/vmovap* (which zero the
    /// register's lanes above `bytes`). The inline path needs the gate to hold, an aligned form's
    /// operand to be aligned to `bytes`, and the operand to be inside one page with a TLB hit;
    /// otherwise the interpreter runs the instruction (`helper_in_branch`), which also raises
    /// every fault. Returns true for a store to memory.
    #[allow(clippy::too_many_arguments)]
    fn vector_move(
        &mut self,
        insn: &Insn,
        load: bool,
        aligned: bool,
        bytes: u32,
        evex: bool,
        next: u64,
        index: usize,
        rip: u64,
    ) -> bool {
        let lanes = bytes / LANE_BYTES;
        let lane = |register: u8, lane: u32| vector_lane_offset(register, lane) as u32;
        let zero_upper = |generator: &mut Self, register: u8| {
            if !evex {
                return;
            }
            for upper in lanes..VECTOR_LANES {
                for half in [0, HALF_LANE_BYTES] {
                    generator
                        .c
                        .local_get(CPU)
                        .i64_const(0)
                        .mem(op::I64_STORE, lane(register, upper) + half);
                }
            }
        };
        if insn.md == MODRM_MOD_REGISTER {
            let (from, to) = if load {
                (insn.rm, insn.reg)
            } else {
                (insn.reg, insn.rm)
            };
            let gated = self.vector_gate(evex);
            if gated {
                self.c.begin_if(BLOCK_VOID);
            }
            for source_lane in 0..lanes {
                for half in [0, HALF_LANE_BYTES] {
                    self.c
                        .local_get(CPU)
                        .local_get(CPU)
                        .mem(op::I64_LOAD, lane(from, source_lane) + half)
                        .mem(op::I64_STORE, lane(to, source_lane) + half);
                }
            }
            zero_upper(self, to);
            if gated {
                self.c.op(op::ELSE);
                self.helper_in_branch(index, rip);
                self.end();
            }
            return false;
        }
        // EVEX compresses an 8-bit displacement: it counts in units of the operand size.
        let mut scaled = *insn;
        if insn.evex_disp8 {
            scaled.disp *= bytes as i64;
        }
        self.ea(&scaled, next);
        let gated = self.vector_gate(evex);
        if aligned {
            self.c
                .local_get(ADDR)
                .i64_const(bytes as i64 - 1)
                .op(op::I64_AND)
                .op(op::I64_EQZ);
            if gated {
                self.c.op(op::I32_AND);
            }
        }
        self.tlb_probe(bytes as u8, if load { TLB_TAG_READ } else { TLB_TAG_WRITE });
        if gated || aligned {
            self.c.op(op::I32_AND);
        }
        self.c.begin_if(BLOCK_VOID);
        // SLOT is done with once the addend is read: it holds the host address from here.
        self.host_address();
        self.c.local_set(SLOT);
        for memory_lane in 0..lanes {
            for half in [0, HALF_LANE_BYTES] {
                let at = memory_lane * LANE_BYTES + half;
                let register = lane(insn.reg, memory_lane) + half;
                if load {
                    self.c
                        .local_get(CPU)
                        .local_get(SLOT)
                        .mem(op::I64_LOAD, at)
                        .mem(op::I64_STORE, register);
                } else {
                    self.c
                        .local_get(SLOT)
                        .local_get(CPU)
                        .mem(op::I64_LOAD, register)
                        .mem(op::I64_STORE, at);
                }
            }
        }
        if load {
            zero_upper(self, insn.reg);
        }
        self.c.op(op::ELSE);
        self.helper_in_branch(index, rip);
        self.end();
        !load
    }

    /// kmovd/kmovq between an opmask register and a general register (`to_general`: `reg` gets
    /// opmask `rm`, else opmask `reg` gets general `rm`), 32 bits zero-extended or all 64, as
    /// `Cpu::exec_kmov` does. Behind the AVX-512 gate like the other inline EVEX instructions.
    fn kmov(&mut self, insn: &Insn, to_general: bool, index: usize, rip: u64) {
        let size = if insn.opsize == 8 { 8 } else { 4 };
        let gated = self.vector_gate(true);
        if gated {
            self.c.begin_if(BLOCK_VOID);
        }
        if to_general {
            self.c.local_get(CPU).mem(
                load_opcode(size, false),
                OPMASK + OPMASK_SIZE * (insn.rm & OPMASK_REGISTER_MASK) as u32,
            );
            self.c.local_set(VALUE);
            self.write_reg(insn.reg, false, 8, VALUE);
        } else {
            self.c.local_get(CPU);
            self.read_reg(insn.rm, false, size);
            self.c.mem(
                op::I64_STORE,
                OPMASK + OPMASK_SIZE * (insn.reg & OPMASK_REGISTER_MASK) as u32,
            );
        }
        if gated {
            self.c.op(op::ELSE);
            self.helper_in_branch(index, rip);
            self.end();
        }
    }

    /// Opens the fast path of an inline vector instruction: pushes the gate's condition when
    /// the block has not checked it yet (`vector_gate`), and for a memory operand of `bytes`
    /// computes its address (an EVEX disp8 scaled by `bytes`), requires it aligned to `bytes`
    /// when `aligned`, and requires a TLB hit for all of it, leaving the host address in `SLOT`
    /// inside the branch. Returns whether a branch was opened; `end_vector_access` closes it with
    /// the interpreter as the other side.
    #[allow(clippy::too_many_arguments)]
    fn begin_vector_access(
        &mut self,
        insn: &Insn,
        bytes: u32,
        aligned: bool,
        write: bool,
        evex: bool,
        next: u64,
    ) -> bool {
        if insn.md == MODRM_MOD_REGISTER {
            let gated = self.vector_gate(evex);
            if gated {
                self.c.begin_if(BLOCK_VOID);
            }
            return gated;
        }
        let mut scaled = *insn;
        if insn.evex && insn.evex_disp8 {
            scaled.disp *= bytes as i64;
        }
        self.ea(&scaled, next);
        let gated = self.vector_gate(evex);
        if aligned {
            self.c
                .local_get(ADDR)
                .i64_const(bytes as i64 - 1)
                .op(op::I64_AND)
                .op(op::I64_EQZ);
            if gated {
                self.c.op(op::I32_AND);
            }
        }
        self.tlb_probe(
            bytes as u8,
            if write { TLB_TAG_WRITE } else { TLB_TAG_READ },
        );
        if gated || aligned {
            self.c.op(op::I32_AND);
        }
        self.c.begin_if(BLOCK_VOID);
        self.host_address();
        self.c.local_set(SLOT);
        true
    }

    fn end_vector_access(&mut self, opened: bool, index: usize, rip: u64) {
        if opened {
            self.c.op(op::ELSE);
            self.helper_in_branch(index, rip);
            self.end();
        }
    }

    /// Pushes 128-bit lane `lane` of the r/m operand: vector register `rm`, or the memory
    /// `begin_vector_access` found (`SLOT`).
    fn push_rm_lane(&mut self, insn: &Insn, lane: u32) {
        if insn.md == MODRM_MOD_REGISTER {
            self.c
                .local_get(CPU)
                .simd_mem(simd::V128_LOAD, vector_lane_offset(insn.rm, lane) as u32);
        } else {
            self.c
                .local_get(SLOT)
                .simd_mem(simd::V128_LOAD, lane * LANE_BYTES);
        }
    }

    fn push_register_lane(&mut self, register: u8, lane: u32) {
        self.c
            .local_get(CPU)
            .simd_mem(simd::V128_LOAD, vector_lane_offset(register, lane) as u32);
    }

    /// Applies a lane-wise `VectorOp` to the two v128 on the stack (first, second).
    fn vector_op(&mut self, operation: VectorOp, lane_log2: u32) {
        let opcode = match operation {
            VectorOp::And => simd::V128_AND,
            // v128.andnot is first & !second and the x86 forms want !first & second: the callers
            // push the operands the other way round and emit it themselves
            VectorOp::AndNot => unreachable!("emitted by the caller"),
            VectorOp::Or => simd::V128_OR,
            VectorOp::Xor => simd::V128_XOR,
            VectorOp::CompareEqual => {
                [simd::I8X16_EQ, simd::I16X8_EQ, simd::I32X4_EQ][lane_log2 as usize]
            }
            VectorOp::MinUnsigned8 => simd::I8X16_MIN_U,
            VectorOp::MaxUnsigned8 => simd::I8X16_MAX_U,
            VectorOp::Add => [
                simd::I8X16_ADD,
                simd::I16X8_ADD,
                simd::I32X4_ADD,
                simd::I64X2_ADD,
            ][lane_log2 as usize],
            VectorOp::Subtract => [
                simd::I8X16_SUB,
                simd::I16X8_SUB,
                simd::I32X4_SUB,
                simd::I64X2_SUB,
            ][lane_log2 as usize],
            VectorOp::MoveMask => unreachable!("pmovmskb is not lane-wise"),
        };
        self.c.simd(opcode);
    }

    /// The AVX-512 integer instructions of `evex_form` that are not moves: lane-wise logic and
    /// arithmetic without a write mask (`reg = vvvv op rm`, the lanes above the vector zeroed),
    /// and the byte compares and tests into an opmask register (ANDed with the write mask), from
    /// a register or a memory operand inside one page. Each 128-bit lane's comparison becomes
    /// 16 mask bits through i8x16.bitmask.
    fn evex_vector(&mut self, insn: &Insn, form: EvexForm, next: u64, index: usize, rip: u64) {
        let lanes = form.bytes / LANE_BYTES;
        let opened = self.begin_vector_access(insn, form.bytes, false, false, true, next);
        if let EvexKind::Alu {
            op: operation,
            lane_log2,
        } = form.kind
        {
            for lane in 0..lanes {
                self.c.local_get(CPU);
                if operation == VectorOp::AndNot {
                    // !vvvv & rm = v128.andnot(rm, vvvv)
                    self.push_rm_lane(insn, lane);
                    self.push_register_lane(insn.vex_v, lane);
                    self.c.simd(simd::V128_ANDNOT);
                } else {
                    self.push_register_lane(insn.vex_v, lane);
                    self.push_rm_lane(insn, lane);
                    self.vector_op(operation, lane_log2);
                }
                self.c
                    .simd_mem(simd::V128_STORE, vector_lane_offset(insn.reg, lane) as u32);
            }
            for upper in lanes..VECTOR_LANES {
                for half in [0, HALF_LANE_BYTES] {
                    self.c.local_get(CPU).i64_const(0).mem(
                        op::I64_STORE,
                        vector_lane_offset(insn.reg, upper) as u32 + half,
                    );
                }
            }
        } else {
            self.c.local_get(CPU);
            match form.kind {
                EvexKind::CompareBytes {
                    predicate: VPCMP_FALSE,
                    ..
                } => {
                    self.c.i64_const(0);
                }
                EvexKind::CompareBytes {
                    predicate: VPCMP_TRUE,
                    ..
                } => {
                    self.c.i64_const(width_mask(form.bytes as u8 / 8));
                }
                _ => {
                    for lane in 0..lanes {
                        self.push_register_lane(insn.vex_v, lane);
                        self.push_rm_lane(insn, lane);
                        match form.kind {
                            EvexKind::CompareEqualBytes => {
                                self.c.simd(simd::I8X16_EQ);
                            }
                            EvexKind::CompareBytes { signed, predicate } => {
                                self.c.simd(match (predicate, signed) {
                                    (0, _) => simd::I8X16_EQ,
                                    (1, true) => simd::I8X16_LT_S,
                                    (1, false) => simd::I8X16_LT_U,
                                    (2, true) => simd::I8X16_LE_S,
                                    (2, false) => simd::I8X16_LE_U,
                                    (4, _) => simd::I8X16_NE,
                                    (5, true) => simd::I8X16_GE_S,
                                    (5, false) => simd::I8X16_GE_U,
                                    (_, true) => simd::I8X16_GT_S,
                                    (_, false) => simd::I8X16_GT_U,
                                });
                            }
                            // (vvvv & rm) != 0, or == 0 for vptestnmb
                            _ => {
                                self.c
                                    .simd(simd::V128_AND)
                                    .i32_const(0)
                                    .simd(simd::I8X16_SPLAT)
                                    .simd(if form.kind == EvexKind::TestBytes {
                                        simd::I8X16_NE
                                    } else {
                                        simd::I8X16_EQ
                                    });
                            }
                        }
                        self.c.simd(simd::I8X16_BITMASK).op(op::I64_EXTEND_I32_U);
                        if lane > 0 {
                            self.c
                                .i64_const(LANE_MASK_BITS * lane as i64)
                                .op(op::I64_SHL)
                                .op(op::I64_OR);
                        }
                    }
                }
            }
            if insn.evex_mask != 0 {
                self.c
                    .local_get(CPU)
                    .mem(op::I64_LOAD, OPMASK + OPMASK_SIZE * insn.evex_mask as u32)
                    .op(op::I64_AND);
            }
            self.c.mem(
                op::I64_STORE,
                OPMASK + OPMASK_SIZE * (insn.reg & OPMASK_REGISTER_MASK) as u32,
            );
        }
        self.end_vector_access(opened, index, rip);
    }

    /// A legacy `66 0f xx` integer instruction on a whole xmm register (`VectorAlu`):
    /// `xmm[reg] = xmm[reg] op src` from a register or a 16-byte aligned memory operand, or
    /// pmovmskb's byte signs into a general register. Only the low lane is written: legacy SSE
    /// leaves the upper parts of the ymm/zmm register alone.
    fn legacy_vector_alu(
        &mut self,
        insn: &Insn,
        form: VectorAlu,
        next: u64,
        index: usize,
        rip: u64,
    ) {
        let opened = self.begin_vector_access(insn, LANE_BYTES, true, false, false, next);
        if form.op == VectorOp::MoveMask {
            self.push_rm_lane(insn, 0);
            self.c
                .simd(simd::I8X16_BITMASK)
                .op(op::I64_EXTEND_I32_U)
                .local_set(VALUE);
            self.write_reg(insn.reg, false, 8, VALUE);
        } else {
            self.c.local_get(CPU);
            if form.op == VectorOp::AndNot {
                // !dst & src = v128.andnot(src, dst)
                self.push_rm_lane(insn, 0);
                self.push_register_lane(insn.reg, 0);
                self.c.simd(simd::V128_ANDNOT);
            } else {
                self.push_register_lane(insn.reg, 0);
                self.push_rm_lane(insn, 0);
                self.vector_op(form.op, form.lane_log2);
            }
            self.c
                .simd_mem(simd::V128_STORE, vector_lane_offset(insn.reg, 0) as u32);
        }
        self.end_vector_access(opened, index, rip);
    }

    /// `66 [REX.W] 0f 6e` movd/movq xmm, r/m32/64 (the value zero-extended into all 128 bits of
    /// the xmm part; the ymm/zmm upper part stays, as for every legacy SSE write) and `0f 7e`
    /// movd/movq r/m32/64, xmm (the low 32 or 64 bits). Returns true for a store to memory.
    fn movd(&mut self, insn: &Insn, to_vector: bool, next: u64, index: usize, rip: u64) -> bool {
        let size = if insn.opsize == 8 { 8 } else { 4 };
        let xmm = vector_lane_offset(insn.reg, 0) as u32;
        let memory = insn.md != MODRM_MOD_REGISTER;
        let opened = self.begin_vector_access(insn, size as u32, false, !to_vector, false, next);
        if to_vector {
            self.c.local_get(CPU);
            if memory {
                self.c.local_get(SLOT).mem(load_opcode(size, false), 0);
            } else {
                self.read_reg(insn.rm, false, size);
            }
            self.c.mem(op::I64_STORE, xmm);
            self.c
                .local_get(CPU)
                .i64_const(0)
                .mem(op::I64_STORE, xmm + HALF_LANE_BYTES);
        } else {
            self.c.local_get(CPU).mem(load_opcode(size, false), xmm);
            self.c.local_set(VALUE);
            if memory {
                self.c
                    .local_get(SLOT)
                    .local_get(VALUE)
                    .mem(memory_store_opcode(size), 0);
            } else {
                self.write_reg(insn.rm, false, 8, VALUE);
            }
        }
        self.end_vector_access(opened, index, rip);
        memory && !to_vector
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
        // The helper may fault, and the exception frame takes the flags from the record.
        self.materialize_flags();
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
        self.materialize_flags();
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
        self.materialize_flags();
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

    /// Defers the record of an instruction whose flags `Fused` can also serve (`pending_record`).
    /// Its record replaces the pending one whole (an inc or dec carries CF in `CARRY`), so that
    /// one is dropped unstored.
    fn record_flags(
        &mut self,
        lazy: LazyFlagsOp,
        fused: FusedOp,
        size: u8,
        carry_known: bool,
        src: u32,
    ) {
        if self.pending_record.replace((lazy, size, src)).is_some() {
            self.flag_records_skipped += 1;
        }
        self.flags = Some(Fused {
            op: fused,
            size,
            carry_known,
        });
    }

    /// Stores the pending record, here where every path from now on has it in `Cpu`: before a
    /// helper call or an instruction lowered without deferring its flags.
    fn flush_flags(&mut self) {
        if let Some((lazy, size, src)) = self.pending_record.take() {
            self.store_lazy(lazy, size, src);
        }
    }

    /// Stores the pending record on a path that leaves the block or calls out (inside a branch,
    /// like a slow-path access or a side exit), leaving it pending for the path that does not.
    fn materialize_flags(&mut self) {
        if let Some((lazy, size, src)) = self.pending_record {
            self.store_lazy(lazy, size, src);
        }
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
            // `helper_cond` reads the record.
            self.flush_flags();
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

    /// `shl`/`shr`/`sar` of an operand of any size by an immediate (`count` is Some: already
    /// masked, non-zero and below the width) or by CL. The flags are recorded as
    /// `LazyFlagsOp::SzpCfOf` with CF the last bit shifted out and OF as `Cpu::shift_rotate`
    /// computes it for every count: shl = MSB of the result xor CF, shr = MSB of the original,
    /// sar = 0. A CL count of zero changes neither flags nor value; the destination is still
    /// written, as the interpreter does (a 32-bit destination zero-extends, which the
    /// zero-extending load did).
    ///
    /// A CL count of an 8- or 16-bit operand is masked to five bits, so it can reach past the
    /// width. The i64 arithmetic of `shift_value` gives the interpreter's answer there without a
    /// special case: the operand is zero-extended (sign-extended for sar), so the result is 0 or
    /// the sign, and shl's CF position `bits - count` goes negative, which wasm takes modulo 64,
    /// far above the operand's bits.
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
                    .i64_const(if size == 8 {
                        WIDE_SHIFT_COUNT_MASK
                    } else {
                        NARROW_SHIFT_COUNT_MASK
                    })
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

    /// Pushes the high 64 bits of the 128-bit product of locals `first` and `second`, signed or
    /// unsigned.
    /// wasm has no such instruction, so it is schoolbook multiplication on 32-bit limbs (each
    /// partial product fits in 64 bits, and the middle column, three 32-bit values, in 34), then
    /// the signed correction: as a two's complement number a negative operand stands for itself
    /// plus 2^64, which adds the other operand to the high half. Clobbers `SCRATCH` and
    /// `SCRATCH_B`.
    fn mul_high(&mut self, signed: bool, first: u32, second: u32) {
        // SCRATCH = lo(first) * hi(second), SCRATCH_B = hi(first) * lo(second)
        self.c
            .local_get(first)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .local_get(second)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .op(op::I64_MUL)
            .local_set(SCRATCH);
        self.c
            .local_get(first)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .local_get(second)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .op(op::I64_MUL)
            .local_set(SCRATCH_B);
        // hi(first) * hi(second) + hi(SCRATCH) + hi(SCRATCH_B)
        self.c
            .local_get(first)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .local_get(second)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .op(op::I64_MUL)
            .local_get(SCRATCH)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .op(op::I64_ADD)
            .local_get(SCRATCH_B)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .op(op::I64_ADD);
        // + the carry out of the middle column: hi(lo * lo) + lo(SCRATCH) + lo(SCRATCH_B)
        self.c
            .local_get(first)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .local_get(second)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .op(op::I64_MUL)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .local_get(SCRATCH)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .op(op::I64_ADD)
            .local_get(SCRATCH_B)
            .i64_const(LOW_DWORD)
            .op(op::I64_AND)
            .op(op::I64_ADD)
            .i64_const(LIMB_BITS)
            .op(op::I64_SHR_U)
            .op(op::I64_ADD);
        if signed {
            // - (first < 0 ? second : 0) - (second < 0 ? first : 0)
            self.c
                .local_get(first)
                .i64_const(SIGN_FILL_SHIFT)
                .op(op::I64_SHR_S)
                .local_get(second)
                .op(op::I64_AND)
                .op(op::I64_SUB)
                .local_get(second)
                .i64_const(SIGN_FILL_SHIFT)
                .op(op::I64_SHR_S)
                .local_get(first)
                .op(op::I64_AND)
                .op(op::I64_SUB);
        }
    }

    /// The 64-bit `imul r, r/m` and `imul r, r/m, imm` with their flags: SF/ZF/PF from the
    /// truncated product and CF = OF set when the full product does not fit, that is when its
    /// high half is not the sign of the low one (`LazyFlagsOp::SzpCfOf`, as the interpreter).
    fn imul64(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        match Self::rm_src(insn, 8) {
            Src::Reg(reg, _) => self.read_reg(reg, false, 8),
            _ => {
                self.ea(insn, next);
                self.mem_load(8, false, index, rip, false);
            }
        }
        self.c.local_set(LHS);
        if insn.op == 0x1af {
            self.read_reg(insn.reg, false, 8);
        } else {
            self.c.i64_const(insn.imm as i32 as i64);
        }
        self.c.local_set(RHS);
        self.c
            .local_get(LHS)
            .local_get(RHS)
            .op(op::I64_MUL)
            .local_set(RES);
        self.mul_high(true, LHS, RHS);
        self.c
            .local_get(RES)
            .i64_const(SIGN_FILL_SHIFT)
            .op(op::I64_SHR_S)
            .op(op::I64_NE)
            .op(op::I64_EXTEND_I32_U)
            .i64_const(CF_AND_OF)
            .op(op::I64_MUL)
            .local_set(SCRATCH_B);
        self.write_reg(insn.reg, false, 8, RES);
        self.store_lazy(LazyFlagsOp::SzpCfOf, 8, SCRATCH_B);
        self.flags = None;
    }

    /// One-operand `mul` and `imul` (group 3 /4 and /5) at 32 and 64 bits: rDX:rAX = rAX * r/m.
    /// CF = OF = (the high half is not zero for `mul`, not the sign of the low half for `imul`),
    /// SF/ZF/PF from the low half: `LazyFlagsOp::SzpCfOf`, as `Cpu::mul_div` leaves them. The
    /// memory operand is read first (it may fault; nothing is written yet).
    fn mul_wide(&mut self, insn: &Insn, signed: bool, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.c.local_set(RHS);
        self.read_reg(GUEST_RAX, false, size);
        self.c.local_set(LHS);
        if size == 8 {
            self.c
                .local_get(LHS)
                .local_get(RHS)
                .op(op::I64_MUL)
                .local_set(RES);
            self.mul_high(signed, LHS, RHS);
            self.c.local_tee(VALUE);
            if signed {
                self.c
                    .local_get(RES)
                    .i64_const(SIGN_FILL_SHIFT)
                    .op(op::I64_SHR_S)
                    .op(op::I64_NE);
            } else {
                self.c.op(op::I64_EQZ).op(op::I32_EQZ);
            }
        } else {
            // Two 32-bit operands extended to 64 bits multiply exactly in 64 bits.
            if signed {
                for operand in [LHS, RHS] {
                    self.c
                        .local_get(operand)
                        .op(op::I64_EXTEND32_S)
                        .local_set(operand);
                }
            }
            self.c
                .local_get(LHS)
                .local_get(RHS)
                .op(op::I64_MUL)
                .local_tee(SCRATCH)
                .i64_const(LIMB_BITS)
                .op(op::I64_SHR_U)
                .i64_const(LOW_DWORD)
                .op(op::I64_AND)
                .local_set(VALUE);
            self.c
                .local_get(SCRATCH)
                .i64_const(LOW_DWORD)
                .op(op::I64_AND)
                .local_set(RES);
            if signed {
                self.c
                    .local_get(SCRATCH)
                    .local_get(SCRATCH)
                    .op(op::I64_EXTEND32_S)
                    .op(op::I64_NE);
            } else {
                self.c.local_get(VALUE).op(op::I64_EQZ).op(op::I32_EQZ);
            }
        }
        self.c
            .op(op::I64_EXTEND_I32_U)
            .i64_const(CF_AND_OF)
            .op(op::I64_MUL)
            .local_set(SCRATCH_B);
        self.write_reg(GUEST_RAX, false, 8, RES);
        self.write_reg(GUEST_RDX, false, 8, VALUE);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
    }

    /// mulx: rdx times r/m, unsigned, the low half to vvvv and then the high half to ModRM.reg
    /// (which wins when they are one register), at 32 or 64 bits. No flags change, so the
    /// operands go in `VALUE` and `ADDR` (free once the memory operand is loaded), not in the
    /// fused locals.
    fn mulx(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.c.local_set(ADDR);
        self.read_reg(GUEST_RDX, false, size);
        self.c.local_set(VALUE);
        if size == 8 {
            self.c
                .local_get(CPU)
                .local_get(VALUE)
                .local_get(ADDR)
                .op(op::I64_MUL)
                .mem(op::I64_STORE, register_offset(insn.vex_v, false));
            self.c.local_get(CPU);
            self.mul_high(false, VALUE, ADDR);
            self.c.mem(op::I64_STORE, register_offset(insn.reg, false));
        } else {
            // Two zero-extended 32-bit operands multiply exactly in 64 bits.
            self.c
                .local_get(VALUE)
                .local_get(ADDR)
                .op(op::I64_MUL)
                .local_set(SCRATCH);
            self.c
                .local_get(CPU)
                .local_get(SCRATCH)
                .i64_const(LOW_DWORD)
                .op(op::I64_AND)
                .mem(op::I64_STORE, register_offset(insn.vex_v, false));
            self.c
                .local_get(CPU)
                .local_get(SCRATCH)
                .i64_const(LIMB_BITS)
                .op(op::I64_SHR_U)
                .mem(op::I64_STORE, register_offset(insn.reg, false));
        }
    }

    /// Pushes the r/m operand of a BMI-style instruction at its operand size, zero-extended.
    fn bmi_source(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        if insn.md == MODRM_MOD_REGISTER {
            self.read_reg(insn.rm, false, insn.opsize);
        } else {
            self.ea(insn, next);
            self.mem_load(insn.opsize, false, index, rip, false);
        }
    }

    /// shlx/shrx/sarx: ModRM.reg = r/m shifted by the count in vvvv, modulo the width. No flags,
    /// so the fused locals stay as they are and are not touched.
    fn bmi_shift(&mut self, insn: &Insn, kind: ShiftKind, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        if kind == ShiftKind::Sar && size == 4 {
            self.c.op(op::I64_EXTEND32_S);
        }
        self.read_reg(insn.vex_v, false, 8);
        self.c.i64_const(8 * size as i64 - 1).op(op::I64_AND);
        self.c.op(match kind {
            ShiftKind::Shl => op::I64_SHL,
            ShiftKind::Shr => op::I64_SHR_U,
            ShiftKind::Sar => op::I64_SHR_S,
        });
        self.mask_to(size);
        self.c.local_set(VALUE);
        self.write_reg(insn.reg, false, 8, VALUE);
    }

    /// bzhi: ModRM.reg = r/m with the bits from index (vvvv bits 7:0) up cleared; an index at or
    /// above the width leaves it whole and sets CF. SF/ZF/PF follow the result, OF and AF clear:
    /// `LazyFlagsOp::SzpCfOf` with CF in the packed word.
    fn bzhi(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.c.local_set(VALUE);
        self.read_reg(insn.vex_v, false, 8);
        self.c
            .i64_const(BZHI_INDEX_MASK)
            .op(op::I64_AND)
            .local_set(SCRATCH);
        // CF = index >= width; the mask is then all ones (wasm would shift by index modulo 64).
        self.c
            .local_get(SCRATCH)
            .i64_const(8 * size as i64)
            .op(op::I64_GE_U)
            .op(op::I64_EXTEND_I32_U)
            .local_set(SCRATCH_B);
        self.c
            .local_get(VALUE)
            .local_get(VALUE)
            .i64_const(1)
            .local_get(SCRATCH)
            .op(op::I64_SHL)
            .i64_const(1)
            .op(op::I64_SUB)
            .op(op::I64_AND)
            .local_get(SCRATCH_B)
            .op(op::I32_WRAP_I64)
            .op(op::SELECT)
            .local_set(RES);
        self.write_reg(insn.reg, false, 8, RES);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
    }

    /// andn: ModRM.reg = r/m & !vvvv, flags from the result with CF and OF clear.
    fn andn(&mut self, insn: &Insn, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.read_reg(insn.vex_v, false, size);
        self.c
            .i64_const(-1)
            .op(op::I64_XOR)
            .op(op::I64_AND)
            .local_set(RES);
        self.c.i64_const(0).local_set(SCRATCH_B);
        self.write_reg(insn.reg, false, 8, RES);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
    }

    /// blsr, blsmsk and blsi (`group` 1, 2 and 3): vvvv = r/m with its lowest set bit cleared,
    /// the mask up to and including it, or just it. Flags from the result; CF is set when the
    /// source is zero (blsr, blsmsk) or not zero (blsi).
    fn lowest_set_bit(&mut self, insn: &Insn, group: u8, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.c.local_tee(VALUE).op(op::I64_EQZ);
        if group == BMI_GROUP_BLSI {
            self.c.op(op::I32_EQZ);
        }
        self.c.op(op::I64_EXTEND_I32_U).local_set(SCRATCH_B);
        self.c.local_get(VALUE);
        if group == BMI_GROUP_BLSI {
            self.c.i64_const(0).local_get(VALUE).op(op::I64_SUB);
        } else {
            self.c.local_get(VALUE).i64_const(1).op(op::I64_SUB);
        }
        self.c.op(if group == BMI_GROUP_BLSMSK {
            op::I64_XOR
        } else {
            op::I64_AND
        });
        self.mask_to(size);
        self.c.local_set(RES);
        self.write_reg(insn.vex_v, false, 8, RES);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
    }

    /// bsf, bsr, tzcnt and lzcnt of a 32- or 64-bit operand. tzcnt/lzcnt write the count (the
    /// width for a zero source) with CF = (source is zero) and SF/ZF/PF from the count; bsf/bsr
    /// leave the whole destination alone for a zero source, else write the bit index, and take
    /// SF/ZF/PF from the source with CF clear. Both are `LazyFlagsOp::SzpCfOf`, which is what the
    /// interpreter leaves (every flag but ZF is undefined for bsf/bsr, and QEMU does the same).
    fn bit_scan(&mut self, insn: &Insn, scan: BitScan, next: u64, index: u32, rip: u64) {
        let size = insn.opsize;
        self.bmi_source(insn, next, index, rip);
        self.c.local_set(VALUE);
        // The count instructions of the operand's own width: i32.clz/ctz of zero is 32.
        let count = |generator: &mut Self, leading: bool| {
            generator.c.local_get(VALUE);
            if size == 8 {
                generator
                    .c
                    .op(if leading { op::I64_CLZ } else { op::I64_CTZ });
            } else {
                generator
                    .c
                    .op(op::I32_WRAP_I64)
                    .op(if leading { op::I32_CLZ } else { op::I32_CTZ })
                    .op(op::I64_EXTEND_I32_U);
            }
        };
        match scan {
            BitScan::Tzcnt | BitScan::Lzcnt => {
                count(self, scan == BitScan::Lzcnt);
                self.c.local_set(RES);
                self.c
                    .local_get(VALUE)
                    .op(op::I64_EQZ)
                    .op(op::I64_EXTEND_I32_U)
                    .local_set(SCRATCH_B);
                self.write_reg(insn.reg, false, 8, RES);
            }
            BitScan::Bsf | BitScan::Bsr => {
                if scan == BitScan::Bsf {
                    count(self, false);
                } else {
                    self.c.i64_const(8 * size as i64 - 1);
                    count(self, true);
                    self.c.op(op::I64_SUB);
                }
                self.read_reg(insn.reg, false, 8);
                self.c
                    .local_get(VALUE)
                    .op(op::I64_EQZ)
                    .op(op::I32_EQZ)
                    .op(op::SELECT)
                    .local_set(SCRATCH);
                self.write_reg(insn.reg, false, 8, SCRATCH);
                // SzpCfOf of the source with CF = OF = 0
                self.c.local_get(VALUE).local_set(RES);
                self.c.i64_const(0).local_set(SCRATCH_B);
            }
        }
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
    }

    /// `shld`/`shrd r/m, r, imm8` at 32 and 64 bits with a count of 1 to the width minus 1 (a
    /// masked count of zero changes nothing and stays in the interpreter). CF is the last bit
    /// shifted out of r/m, OF is set when the sign changed.
    #[allow(clippy::too_many_arguments)]
    fn double_shift(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: u32,
        next: u64,
        index: u32,
        rip: u64,
    ) -> bool {
        let size = insn.opsize;
        let bits = 8 * size as i64;
        let count = count as i64;
        self.load_dst(insn, dst, LHS, size, next, index, rip);
        self.read_reg(insn.reg, false, size);
        self.c.local_set(RHS);
        let (dst_shift, dst_op, src_shift, src_op, carry_bit) = if left {
            (
                count,
                op::I64_SHL,
                bits - count,
                op::I64_SHR_U,
                bits - count,
            )
        } else {
            (count, op::I64_SHR_U, bits - count, op::I64_SHL, count - 1)
        };
        self.c
            .local_get(LHS)
            .i64_const(dst_shift)
            .op(dst_op)
            .local_get(RHS)
            .i64_const(src_shift)
            .op(src_op)
            .op(op::I64_OR);
        self.mask_to(size);
        self.c.local_set(RES);
        // SCRATCH_B = CF | OF << 1, OF = MSB(result ^ original)
        self.c
            .local_get(LHS)
            .i64_const(carry_bit)
            .op(op::I64_SHR_U)
            .i64_const(1)
            .op(op::I64_AND)
            .local_get(RES)
            .local_get(LHS)
            .op(op::I64_XOR)
            .i64_const(bits - 1)
            .op(op::I64_SHR_U)
            .i64_const(1)
            .op(op::I64_AND)
            .i64_const(SHIFT_OF_BIT)
            .op(op::I64_SHL)
            .op(op::I64_OR)
            .local_set(SCRATCH_B);
        let stored = self.store_dst(dst, size, RES, index, rip);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size, SCRATCH_B);
        self.flags = None;
        stored
    }

    /// `bswap r32`/`bswap r64`: swap neighbouring bytes, then neighbouring 16-bit words, then
    /// rotate the halves (wasm has no byte swap). No flags.
    fn bswap(&mut self, reg: u8, size: u8) {
        if size == 8 {
            self.read_reg(reg, false, 8);
            self.c
                .local_tee(VALUE)
                .i64_const(8)
                .op(op::I64_SHR_U)
                .i64_const(ALTERNATE_BYTES_64)
                .op(op::I64_AND)
                .local_get(VALUE)
                .i64_const(ALTERNATE_BYTES_64)
                .op(op::I64_AND)
                .i64_const(8)
                .op(op::I64_SHL)
                .op(op::I64_OR)
                .local_tee(VALUE)
                .i64_const(16)
                .op(op::I64_SHR_U)
                .i64_const(ALTERNATE_WORDS_64)
                .op(op::I64_AND)
                .local_get(VALUE)
                .i64_const(ALTERNATE_WORDS_64)
                .op(op::I64_AND)
                .i64_const(16)
                .op(op::I64_SHL)
                .op(op::I64_OR)
                .i64_const(32)
                .op(op::I64_ROTL)
                .local_set(VALUE);
        } else {
            // On the zero-extended value: swap the bytes of each 16-bit half, then the halves.
            self.read_reg(reg, false, 4);
            self.c
                .local_tee(VALUE)
                .i64_const(8)
                .op(op::I64_SHR_U)
                .i64_const(ALTERNATE_BYTES_32)
                .op(op::I64_AND)
                .local_get(VALUE)
                .i64_const(ALTERNATE_BYTES_32)
                .op(op::I64_AND)
                .i64_const(8)
                .op(op::I64_SHL)
                .op(op::I64_OR)
                .local_tee(VALUE)
                .i64_const(16)
                .op(op::I64_SHR_U)
                .local_get(VALUE)
                .i64_const(16)
                .op(op::I64_SHL)
                .op(op::I64_OR)
                .i64_const(LOW_DWORD)
                .op(op::I64_AND)
                .local_set(VALUE);
        }
        self.write_reg(reg, false, 8, VALUE);
    }

    /// `rol`/`ror` of a 32- or 64-bit operand by an immediate (`count`: masked, non-zero) or CL.
    /// The wasm rotates take their count modulo the width, which is the x86 count mask at these
    /// sizes; a CL count of zero still writes the destination, zero-extending a 32-bit one, as the
    /// interpreter does, and changes no flag.
    ///
    /// A rotate replaces CF and OF and keeps SF/ZF/PF/AF, which only the lazy record of whatever
    /// wrote them can tell, so with `merge_flags` CF and OF are packed (CF in bit 0, OF in bit 1,
    /// OF for every count as `Cpu::shift_rotate` computes it) for `helper_set_rotate_flags`, which
    /// merges them after the store succeeded. Without it (`flags_dead_after`: a later instruction
    /// of the block overwrites every flag) the flags are not computed at all. A rotate whose block
    /// ends right after it (`rol; jmp`, Linux's PV start-up runs 134M of them) needs the merge:
    /// the successor, an interrupt frame or `--verify` may look.
    #[allow(clippy::too_many_arguments)]
    fn rotate(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next: u64,
        index: u32,
        rip: u64,
        merge_flags: bool,
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
        if merge_flags {
            // SCRATCH = CF | OF << 1. A left rotation's CF is the new low bit and OF the new
            // msb ^ CF; a right rotation's CF is the new msb and OF the msb ^ the bit below it.
            let msb = 8 * size as i64 - 1;
            let (carry_bit, other_bit) = if left { (0, msb) } else { (msb, msb - 1) };
            self.result_bit(carry_bit);
            self.c.local_tee(SCRATCH);
            self.result_bit(other_bit);
            self.c
                .op(op::I64_XOR)
                .i64_const(1)
                .op(op::I64_SHL)
                .local_get(SCRATCH)
                .op(op::I64_OR)
                .local_set(SCRATCH);
            if count.is_none() {
                // The masked count, read before the store: `rol rcx, cl` overwrites it.
                self.read_reg(GUEST_RCX, false, 1);
                self.c
                    .i64_const(8 * size as i64 - 1)
                    .op(op::I64_AND)
                    .local_set(CARRY);
            }
        }
        let stored = self.store_dst(dst, size, SCRATCH_B, index, rip);
        if merge_flags {
            if count.is_none() {
                self.c
                    .local_get(CARRY)
                    .op(op::I64_EQZ)
                    .op(op::I32_EQZ)
                    .begin_if(BLOCK_VOID);
            }
            self.c
                .local_get(CPU)
                .local_get(SCRATCH)
                .i32_const(self.addrs.rotate_flags as i32)
                .call_indirect(COND_TYPE)
                .op(op::DROP);
            if count.is_none() {
                self.end();
            }
        }
        stored
    }

    /// Pushes bit `bit` of the rotated value in `SCRATCH_B` as an i64 0 or 1.
    fn result_bit(&mut self, bit: i64) {
        self.c
            .local_get(SCRATCH_B)
            .i64_const(bit)
            .op(op::I64_SHR_U)
            .i64_const(1)
            .op(op::I64_AND);
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
    /// come before the first emit for that arm. Which forms are lowered follows the aarch64
    /// backend (`emit.rs`), plus bswap; a 64-bit imul whose flags are dead (`flags_dead_after`)
    /// skips the high half of the product.
    fn try_inline(&mut self, index: usize, rip: u64, next: u64) -> bool {
        let insn: Insn = self.insns[index];
        if !keeps_pending_record(&insn) {
            self.flush_flags();
        }
        let k = index as u32;
        let retired = k + 1;
        let osz = insn.opsize;
        let wide = !insn.addr32 && !insn.lock && !insn.p66 && (osz == 4 || osz == 8);
        let is_mem = insn.md != MODRM_MOD_REGISTER;
        let plain = Self::mem_ok(&insn);
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
            // rol/ror at 32 and 64 bits; CF and OF are merged into the flags by a helper unless
            // a later instruction of the block overwrites them all.
            0xc1 | 0xd1 | 0xd3 if wide && matches!(insn.reg & 7, SHIFT_ROL | SHIFT_ROR) => {
                let count = match insn.op {
                    0xc1 => Some(insn.imm as u32 & (8 * osz as u32 - 1)),
                    0xd1 => Some(1),
                    _ => None,
                };
                if count == Some(0) {
                    return false;
                }
                store = self.rotate(
                    &insn,
                    insn.reg & 7 == SHIFT_ROL,
                    Self::rm_dst(&insn, osz),
                    count,
                    osz,
                    next,
                    k,
                    rip,
                    !flags_dead_after(self.insns, index),
                );
            }
            // shl/shr/sar r/m8 and r/m16 by imm8 | 1 | cl. An immediate count of zero or at least
            // the width goes to the interpreter: it is rare, and CF there is not a fixed bit.
            0xc0 | 0xc1 | 0xd0 | 0xd1 | 0xd2 | 0xd3
                if plain
                    && matches!(insn.reg & 7, SHIFT_SHL..=SHIFT_SAR)
                    && (insn.op & 1 == 0 || (insn.p66 && osz == 2)) =>
            {
                let size = if insn.op & 1 == 0 { 1 } else { 2 };
                let count = match insn.op {
                    0xc0 | 0xc1 => Some(insn.imm as u32 & NARROW_SHIFT_COUNT_MASK as u32),
                    0xd0 | 0xd1 => Some(1),
                    _ => None,
                };
                if count.is_some_and(|count| count == 0 || count >= 8 * size as u32) {
                    return false;
                }
                let kind = match insn.reg & 7 {
                    SHIFT_SHL | SHIFT_SAL_ALIAS => ShiftKind::Shl,
                    SHIFT_SHR => ShiftKind::Shr,
                    _ => ShiftKind::Sar,
                };
                store = self.shift(
                    &insn,
                    kind,
                    Self::rm_dst(&insn, size),
                    count,
                    size,
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
            0x1af | 0x69 | 0x6b if wide && osz == 8 => self.imul64(&insn, next, k, rip),
            // group 3 /4 mul and /5 imul r/m, one operand, at 32 and 64 bits
            0xf7 if wide && matches!(insn.reg & 7, 4 | 5) => {
                self.mul_wide(&insn, insn.reg & 7 == 5, next, k, rip);
            }
            // shld/shrd r/m, r, imm8 at 32 and 64 bits
            0x1a4 | 0x1ac if wide => {
                let count = insn.imm as u32 & (8 * osz as u32 - 1);
                if count == 0 {
                    return false;
                }
                store = self.double_shift(
                    &insn,
                    insn.op == 0x1a4,
                    Self::rm_dst(&insn, osz),
                    count,
                    next,
                    k,
                    rip,
                );
            }
            // BMI1/BMI2 (VEX.LZ, map 0f38): shlx/shrx/sarx f7 with 66/F2/F3, bzhi f5, andn f2 and
            // the blsr/blsmsk/blsi group f3. bextr, pdep and pext go to the interpreter.
            VEX_BMI_SHIFT_OR_BEXTR
            | VEX_BMI_BZHI_PEXT_PDEP
            | VEX_BMI_ANDN
            | VEX_BMI_GROUP17
            | VEX_BMI_MULX
                if insn.vex && !insn.vex_l && plain && (osz == 4 || osz == 8) =>
            {
                match (insn.op, insn.rep, insn.p66) {
                    (VEX_BMI_SHIFT_OR_BEXTR, 0, true) => {
                        self.bmi_shift(&insn, ShiftKind::Shl, next, k, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, 2, false) => {
                        self.bmi_shift(&insn, ShiftKind::Shr, next, k, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, 1, false) => {
                        self.bmi_shift(&insn, ShiftKind::Sar, next, k, rip)
                    }
                    (VEX_BMI_BZHI_PEXT_PDEP, 0, false) => self.bzhi(&insn, next, k, rip),
                    (VEX_BMI_ANDN, 0, false) => self.andn(&insn, next, k, rip),
                    (VEX_BMI_MULX, PREFIX_F2, false) => self.mulx(&insn, next, k, rip),
                    (VEX_BMI_GROUP17, 0, false)
                        if matches!(
                            insn.reg & 7,
                            BMI_GROUP_BLSR | BMI_GROUP_BLSMSK | BMI_GROUP_BLSI
                        ) =>
                    {
                        self.lowest_set_bit(&insn, insn.reg & 7, next, k, rip)
                    }
                    _ => return false,
                }
            }
            // bsf/bsr (no prefix) and tzcnt/lzcnt (F3) at 32 and 64 bits
            0x1bc | 0x1bd if wide && matches!(insn.rep, 0 | 1) => {
                let scan = match (insn.op, insn.rep) {
                    (0x1bc, 0) => BitScan::Bsf,
                    (0x1bd, 0) => BitScan::Bsr,
                    (0x1bc, _) => BitScan::Tzcnt,
                    _ => BitScan::Lzcnt,
                };
                self.bit_scan(&insn, scan, next, k, rip);
            }
            // movups/movupd, movaps/movapd, movdqu/movdqa
            0x110 | 0x111 | 0x128 | 0x129 | 0x16f | 0x17f if plain => {
                let Some(form) = vector_move(&insn) else {
                    return false;
                };
                store =
                    self.vector_move(&insn, form.load, form.aligned, 16, false, next, index, rip);
            }
            // kmovd/kmovq k, r and r, k (VEX.L0, no vvvv, register forms)
            VEX_KMOV_FROM_GENERAL | VEX_KMOV_TO_GENERAL
                if insn.vex
                    && insn.rep == PREFIX_F2
                    && !is_mem
                    && !insn.vex_l
                    && insn.vex_v == 0 =>
            {
                self.kmov(&insn, insn.op == VEX_KMOV_TO_GENERAL, index, rip);
            }
            // pxor/por/pand/pandn, pcmpeq*, pminub/pmaxub, padd*, psub*, pmovmskb (66 prefix)
            0x1d4
            | 0x1d7
            | 0x1da
            | 0x1db
            | 0x1de
            | 0x1df
            | 0x1eb
            | 0x1ef
            | 0x174..=0x176
            | 0x1f8..=0x1fe
                if plain && self.addrs.simd =>
            {
                let Some(form) = vector_alu(&insn) else {
                    return false;
                };
                // pmovmskb has no memory form (the decoder-level #UD stays with the interpreter)
                if form.op == VectorOp::MoveMask && is_mem {
                    return false;
                }
                self.legacy_vector_alu(&insn, form, next, index, rip);
            }
            // movd/movq between xmm and a general register or memory
            MOVD_TO_VECTOR | MOVD_FROM_VECTOR if plain && insn.p66 && insn.rep == 0 => {
                store = self.movd(&insn, insn.op == MOVD_TO_VECTOR, next, index, rip);
            }
            // AVX-512 moves without a write mask, integer logic and arithmetic without one, and
            // the byte compares and tests into an opmask register
            _ if insn.evex && (!is_mem || plain) => {
                let Some(form) = evex_form(&insn) else {
                    return false;
                };
                match form.kind {
                    EvexKind::Move { load, aligned } if insn.evex_mask == 0 => {
                        store = self
                            .vector_move(&insn, load, aligned, form.bytes, true, next, index, rip);
                    }
                    EvexKind::Alu { .. } if insn.evex_mask == 0 && self.addrs.simd => {
                        self.evex_vector(&insn, form, next, index, rip);
                    }
                    EvexKind::CompareEqualBytes
                    | EvexKind::TestBytes
                    | EvexKind::TestNotBytes
                    | EvexKind::CompareBytes { .. }
                        if self.addrs.simd =>
                    {
                        self.evex_vector(&insn, form, next, index, rip);
                    }
                    _ => return false,
                }
            }
            // bswap r32 / r64 (the 16-bit form is undefined and stays in the interpreter)
            0x1c8..=0x1cf if wide => self.bswap(opcode_register(&insn), osz),
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
                    let (reg, high) = if source_size == 1 {
                        Self::named_register(&insn, insn.rm, 1)
                    } else {
                        (insn.rm, false)
                    };
                    self.read_reg(reg, high, source_size);
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
                if self.last {
                    self.jcc((insn.op & 15) as u8, target, next, retired);
                } else {
                    // A side exit; the not-taken path goes on in this block, with the flags (and
                    // the fused locals) as they were.
                    self.condition((insn.op & 15) as u8);
                    self.c.begin_if(BLOCK_VOID);
                    self.chain_to(target, retired);
                    self.end();
                }
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
            // The SHA extensions (no prefix), pshufd, palignr and pshufb (66 prefix): see
            // `whole_vector`
            0x170 | 0x200 | 0x30f | 0x2c8..=0x2cd | 0x3cc if plain => {
                let Some(form) = whole_vector(&insn) else {
                    return false;
                };
                self.whole_vector(&insn, form, next, k, rip);
            }
            _ => return false,
        }
        if store {
            self.smc_check(k, next);
        }
        self.inlined += 1;
        true
    }

    /// A `WholeVector` instruction (the SHA extensions, pshufd, palignr, pshufb) through
    /// `helper_whole_vector`, with the source loaded from its register or, 16-byte aligned, from
    /// memory. The SSE gate failing (checked at the block's first SSE instruction), a misaligned
    /// operand or a TLB miss runs the instruction in the interpreter instead, which raises the
    /// fault or does the slow access. Neither path touches the flags.
    fn whole_vector(&mut self, insn: &Insn, form: WholeVector, next: u64, k: u32, rip: u64) {
        let is_register = insn.md == MODRM_MOD_REGISTER;
        // Pushes the fast path's i32 condition, each part `and`ed onto the ones before.
        let mut conditions = 0;
        if !self.sse_gated {
            self.sse_gated = true;
            self.c
                .local_get(CPU)
                .mem(op::I64_LOAD, CR0)
                .i64_const((CR0_EM | CR0_TS) as i64)
                .op(op::I64_AND)
                .op(op::I64_EQZ);
            self.c
                .local_get(CPU)
                .mem(op::I64_LOAD, CR4)
                .i64_const(CR4_OSFXSR as i64)
                .op(op::I64_AND)
                .i64_const(0)
                .op(op::I64_NE)
                .op(op::I32_AND);
            conditions += 1;
        }
        if !is_register {
            self.ea(insn, next);
            self.c
                .local_get(ADDR)
                .i64_const(VECTOR_ALIGNMENT - 1)
                .op(op::I64_AND)
                .op(op::I64_EQZ);
            // An aligned 16-byte access never crosses a page, so the probe's one entry covers it.
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_READ);
            self.c.op(op::I32_AND);
            if conditions > 0 {
                self.c.op(op::I32_AND);
            }
            conditions += 1;
        }
        if conditions > 0 {
            self.c.begin_if(BLOCK_VOID);
        }
        self.c
            .local_get(CPU)
            .i64_const(form.packed(insn.reg) as i64);
        for half in [0, XMM_HALF] {
            if is_register {
                self.c
                    .local_get(CPU)
                    .mem(op::I64_LOAD, XMM + XMM_SIZE * insn.rm as u32 + half);
            } else {
                self.host_address();
                self.c.mem(op::I64_LOAD, half);
            }
        }
        self.c
            .i32_const(self.addrs.whole_vector as i32)
            .call_indirect(STORE_TYPE)
            .op(op::DROP);
        if conditions > 0 {
            // Not `helper`: that pays for the instructions before this one at translation time,
            // which a path that is only sometimes taken cannot do (the native generators' slow
            // paths are the same).
            self.c.op(op::ELSE);
            self.c
                .local_get(CPU)
                .i32_const(&self.insns[k as usize] as *const Insn as usize as i32);
            self.rip_to(rip);
            self.c
                .i32_const(self.addrs.exec as i32)
                .call_indirect(EXEC_TYPE);
            self.check_status(k, true);
            self.end();
        }
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
        // Without chaining (the lockstep verifier) every block runs once per call; a straddler
        // has no loop to speak of.
        if self.addrs.chain != 0
            && self.link_cells != 0
            && jumps_to(self.insns, rips, end_rip, self.entry_rip)
        {
            // Of type i64 like the function: every way out of the loop leaves the block, so
            // its end is never reached, and the function's result type checks after it.
            self.c.begin_loop(BLOCK_I64);
            self.self_loop = Some(self.c.depth());
        }
        for index in 0..count {
            let rip = rips[index];
            let next = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            self.last = index + 1 == count;
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next) {
                let side_exit = !self.last && is_conditional_branch(&self.insns[index]);
                if is_inline_terminator(&self.insns[index]) && !side_exit {
                    self.finish();
                    return;
                }
                continue;
            }
            self.helper(index, rip, terminal);
            if terminal {
                self.finish();
                return;
            }
        }
        self.chain_to(end_rip, count as u32);
        self.finish();
    }
}

/// Whether `insn`, lowered inline, can leave a pending flag record (`Gen::pending_record`) where
/// it is: it either replaces the record by deferring its own (add, sub, logic ops, cmp, test,
/// inc, dec, neg into a register), or leaves the flags alone and neither reads the record from
/// `Cpu` nor writes `RES`, `RHS` or `CARRY`, which hold its values. Anything else stores the
/// record first; so does anything not lowered inline (`Gen::helper`).
///
/// A flag writer with a memory destination must store the old record first: it computes its
/// result into `RES` (and `RHS`) before the store, and if the store faults the exception frame
/// must get the flags from before the instruction.
fn keeps_pending_record(insn: &Insn) -> bool {
    let register_destination = insn.md == MODRM_MOD_REGISTER;
    match insn.op {
        0x00..=0x3f if insn.op & 7 < 6 => {
            let alu_op = ((insn.op >> 3) & 7) as u8;
            let writes_memory = insn.op & 7 < 2 && !register_destination && alu_op != ALU_CMP;
            !matches!(alu_op, ALU_ADC | ALU_SBB) && !writes_memory
        }
        0x80 | 0x81 | 0x83 => {
            let alu_op = insn.reg & 7;
            !matches!(alu_op, ALU_ADC | ALU_SBB) && (register_destination || alu_op == ALU_CMP)
        }
        0x84 | 0x85 | 0xa8 | 0xa9 => true,
        // test, not (no flags), neg
        0xf6 | 0xf7 => match insn.reg & 7 {
            0 | 2 => true,
            3 => register_destination,
            _ => false,
        },
        // inc, dec; call, jmp and push r/m
        0xfe | 0xff => match insn.reg & 7 {
            0 | 1 => register_destination,
            2 | 4 | 6 => insn.op == 0xff,
            _ => false,
        },
        // mov, lea, movsxd, cbw/cdq family, push, pop, nops, fences, setcc, cmovcc, movzx/movsx,
        // the jumps, call and ret, bswap, shlx/shrx/sarx
        0x88..=0x8b
        | 0xc6
        | 0xc7
        | 0xb0..=0xbf
        | 0x8d
        | 0x63
        | 0x98
        | 0x99
        | 0x50..=0x5f
        | 0x68
        | 0x6a
        | 0x90
        | 0x10d
        | 0x118..=0x11f
        | 0x1ae
        | 0x190..=0x19f
        | 0x140..=0x14f
        | 0x1b6
        | 0x1b7
        | 0x1be
        | 0x1bf
        | 0x70..=0x7f
        | 0x180..=0x18f
        | 0xe8
        | 0xe9
        | 0xeb
        | 0xc3
        | 0x1c8..=0x1cf
        | VEX_BMI_SHIFT_OR_BEXTR
        | VEX_KMOV_FROM_GENERAL
        | VEX_KMOV_TO_GENERAL => true,
        VEX_BMI_MULX => insn.rep == PREFIX_F2,
        // the vector moves and integer instructions, legacy and EVEX, and movd/movq
        0x110 | 0x111 | 0x128 | 0x129 | 0x16f | 0x17f | MOVD_TO_VECTOR | MOVD_FROM_VECTOR => true,
        0x1d4 | 0x1d7 | 0x1da | 0x1db | 0x1de | 0x1df | 0x1eb | 0x1ef | 0x174..=0x176 => true,
        0x1f8..=0x1fe => true,
        _ => insn.evex,
    }
}

/// Whether a jcc or a direct jmp among `insns` (at page offsets `rips`, the last ending at
/// `end_rip`) targets page offset `target`.
fn jumps_to(insns: &[Insn], rips: &[u64], end_rip: u64, target: u64) -> bool {
    insns.iter().enumerate().any(|(index, insn)| {
        let next = rips.get(index + 1).copied().unwrap_or(end_rip);
        (is_conditional_branch(insn) || matches!(insn.op, 0xe9 | 0xeb))
            && next.wrapping_add(insn.imm as u64) == target
    })
}

impl CompiledBlock {
    pub fn function(&self) -> Function<'_> {
        Function {
            type_index: BLOCK_TYPE,
            code: &self.code,
            i32_locals: I32_LOCALS,
            i64_locals: I64_LOCALS,
        }
    }
}

/// Generates the function for one block. `insns` must outlive the code, which points into it, and
/// so must the `link_cells_for(insns)` cells at `link_cells` (0: the exits are not linked).
pub(crate) fn compile_block(
    addrs: &Addrs,
    insns: &[Insn],
    rips: &[u64],
    end_rip: u64,
    link_cells: u32,
) -> CompiledBlock {
    let mut generator = Gen::new(addrs, insns, link_cells, link_cells_for(insns));
    generator.compile(rips, end_rip);
    CompiledBlock {
        code: generator.c.bytes,
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
