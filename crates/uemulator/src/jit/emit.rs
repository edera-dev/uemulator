//! aarch64 block compiler. Lowers the decoded instructions of one translated block (see
//! `Jit::translate` in `native.rs`) to machine code. Common integer instructions (ALU at every
//! operand size including read-modify-write memory forms, mov, lea, movzx/movsx, inc/dec/neg/not,
//! shl/shr/sar, imul, setcc/cmovcc, push/pop, near jmp/jcc/call/ret and indirect call/jmp) are
//! emitted inline; every other instruction becomes a call to `helper_exec`, which runs the
//! interpreter on it, so coverage can grow one opcode at a time. See `plans/06-jit.md`.
//!
//! Register roles. The prologue in `arch_a64.rs` sets up the callee-saved ones and enters the
//! block with `br`, so a block is straight-line code, not a function:
//! - `CPU` (x19): `*mut Cpu`.
//! - `TLB_BASE` (x20): `Cpu::tlb_active_base`, latched once on entry. A block therefore never
//!   runs across a CPL change; every instruction that changes CPL ends the block.
//! - `PAGE_BASE` (x21): guest address of the block's page. Blocks are position independent: every
//!   guest rip they use is `PAGE_BASE` plus an offset known when the block is translated (see
//!   `Gen::rip_to`), and the `rip`s in `compile` and below are such offsets, not addresses.
//! - `SAVED_ADDRESS` (x22) and `SAVED_NZCV` (x23): hold the access address and the host flags
//!   across a slow-path helper call. Together with x19/x20 these are the only callee-saved
//!   registers the code touches, so they are the only ones the prologue saves.
//! - x0..x6 and x16 are caller-saved scratch: x0/x1 are the ALU operands and the helper
//!   argument/return registers, `ADDRESS` (x2) the guest linear address being accessed,
//!   `SCRATCH_A..C` (x3..x5) temporaries of address arithmetic and the TLB probe, `VALUE` (x6)
//!   the value moved by mov-style instructions, x16 the helper call target.
//!   Any helper call clobbers all of them, which is why operands are loaded in an order that
//!   keeps nothing live across a memory access (see `Gen::alu`).
//!
//! Exit protocol. A block never returns by itself; it ends with a branch to one of the stubs in
//! the prologue (addresses in `Addrs`): `exit_ok` with x0 = `retired << EXIT_COUNT_SHIFT`,
//! `exit_fail` with x0 = helper status and x1 = index of the instruction that stopped the block,
//! or, for a control transfer to a known guest address, `chain` (see `Gen::chain_exit`), which
//! continues straight into the successor block when it exists and otherwise falls back to
//! `exit_ok`. The exit stubs pack `status | retired << EXIT_COUNT_SHIFT` into x0 and return to
//! `Cpu::step_block`. Instructions that change what the run loop must see (interrupt delivery,
//! halts, privilege, port I/O) always leave through `exit_ok`; the loop is where interrupts are
//! taken, and a chain hands back to it whenever `Cpu::next_event` is reached.
//!
//! `Cpu::rip` is not kept current inside a block. Whatever can fault or leave the block stores
//! it first: helpers get the instruction's rip as an argument, exits store the next rip.
//!
//! Flags are lazy (`LazyFlagsOp`). After every inlined ALU instruction the result and second
//! operand are stored into `Cpu::lazy_*`, so the interpreter and `helper_cond` can compute any flag
//! on demand. Additionally the host NZCV flags still hold that result (`Gen::live`) until
//! something clobbers them, which lets a following `jcc` test them directly.

use std::mem::{offset_of, size_of};

use super::a64::*;
use super::helpers::*;
use super::native::LinkSite;
use crate::cpu::decode::{Insn, NO_REG, VEX_OP_FLAG};
use crate::cpu::flags::{AC, CF, IF, LazyFlagsOp, SZP_CF_OF_SRC_CF_BIT};
use crate::cpu::xsave::{XCR0_AVX, XCR0_AVX512, XCR0_SSE};
use crate::cpu::{CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXSAVE};
use crate::cpu::{Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE, TlbEntry};

// Byte offsets into the `#[repr(C)]` `Cpu` and `TlbEntry`, burned into the generated code.
// `offset_of!` makes them follow field moves; the generated code only breaks if a field's
// meaning or an adjacency noted below changes. Each must stay below 32 KiB (the scaled
// `ldr`/`str` immediate), or the assembler asserts when the first block is compiled.
const GPR_OFFSET: u32 = offset_of!(Cpu, gpr) as u32;
const GPR_SIZE: u32 = size_of::<u64>() as u32;
const RIP_OFFSET: u32 = offset_of!(Cpu, rip) as u32;
const ICOUNT_OFFSET: u32 = offset_of!(Cpu, icount) as u32;
const BLOCK_EXIT_REQUESTED_OFFSET: u32 = offset_of!(Cpu, block_exit_requested) as u32;
const NEXT_EVENT_OFFSET: u32 = offset_of!(Cpu, next_event) as u32;
const PREPAID_ICOUNT_OFFSET: u32 = offset_of!(Cpu, prepaid_icount) as u32;
/// `lazy_size` must directly follow `lazy_op`: `store_lazy_op` writes both with one halfword
/// store (op in the low byte, size in the high byte).
const LAZY_OP_OFFSET: u32 = offset_of!(Cpu, lazy_op) as u32;
const LAZY_SIZE_SHIFT: u32 = 8;
const LAZY_DST_OFFSET: u32 = offset_of!(Cpu, lazy_dst) as u32;
const LAZY_SRC_OFFSET: u32 = offset_of!(Cpu, lazy_src) as u32;
const EFLAGS_OFFSET: u32 = offset_of!(Cpu, eflags) as u32;
const LAZY_SRC2_OFFSET: u32 = offset_of!(Cpu, lazy_src2) as u32;
const CR0_OFFSET: u32 = offset_of!(Cpu, cr0) as u32;
const CR4_OFFSET: u32 = offset_of!(Cpu, cr4) as u32;
const XMM_OFFSET: u32 = offset_of!(Cpu, xmm) as u32;
const OPMASK_OFFSET: u32 = offset_of!(Cpu, opmask) as u32;
const OPMASK_SIZE: u32 = size_of::<u64>() as u32;
/// An opmask register number is the low three bits of its ModRM field.
const OPMASK_REGISTER_MASK: u8 = 7;
/// 128-bit lanes of a 512-bit vector register.
const VECTOR_LANES: u32 = 4;
/// `OPMASK_EXPANSION` rows are 256 `u64`s apart: the shift that turns a table row index into
/// the byte offset.
const OPMASK_EXPANSION_SHIFT: u32 = 3;
/// The vpcmp immediate's predicates (SDM table of VPCMP).
const VPCMP_EQUAL: u8 = 0;
const VPCMP_LESS: u8 = 1;
const VPCMP_LESS_EQUAL: u8 = 2;
const VPCMP_FALSE: u8 = 3;
const VPCMP_NOT_EQUAL: u8 = 4;
const VPCMP_NOT_LESS: u8 = 5;
const VPCMP_NOT_LESS_EQUAL: u8 = 6;
const XCR0_OFFSET: u32 = offset_of!(Cpu, xcr0) as u32;
/// What `Cpu::evex_gate` requires of XCR0: the SSE, AVX and AVX-512 state components.
const XCR0_EVEX_STATE: u64 = XCR0_SSE | XCR0_AVX | XCR0_AVX512;
/// Every 128-bit lane `ldr_q`/`str_q` of a vector register reaches must be within the 12-bit
/// scaled offset of the instruction (the highest lane of zmm31).
const _: () =
    assert!(offset_of!(Cpu, zmm_extra) + size_of::<[[u128; 4]; 16]>() <= 4096 * size_of::<u128>());
const _: () = assert!(offset_of!(Cpu, opmask) + size_of::<[u64; 8]>() < 4096 * size_of::<u64>());
const XMM_SIZE: u32 = size_of::<u128>() as u32;
const XMM_HALF: u32 = XMM_SIZE / 2;
/// Legacy-SSE 128-bit memory operands of movaps/movdqa and friends must be aligned to this.
const VECTOR_ALIGNMENT: u64 = XMM_SIZE as u64;
/// Host vector registers of an inline vector instruction: the guest destination (also the result)
/// and the source operand.
const VECTOR_DST: u32 = 0;
const VECTOR_SRC: u32 = 1;
const RCX_INDEX: u32 = 1;
const SEG_BASE_OFFSET: u32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as u32;
const SEG_STRIDE: u32 = size_of::<Seg>() as u32;
const GS_BASE_OFFSET: u32 = SEG_BASE_OFFSET + SEG_STRIDE * SEG_GS as u32;
const KGSBASE_OFFSET: u32 = offset_of!(Cpu, kgsbase) as u32;
const CPL_OFFSET: u32 = offset_of!(Cpu, cpl) as u32;

const TLB_TAG_READ_OFFSET: u32 = offset_of!(TlbEntry, tag_r) as u32;
const TLB_TAG_WRITE_OFFSET: u32 = offset_of!(TlbEntry, tag_w) as u32;
const TLB_ADDEND_OFFSET: u32 = offset_of!(TlbEntry, addend) as u32;
/// log2 of the entry size, so an entry address is `base + (index << TLB_ENTRY_SHIFT)`.
const TLB_ENTRY_SHIFT: u32 = size_of::<TlbEntry>().trailing_zeros();
const _: () = assert!(size_of::<TlbEntry>().is_power_of_two());

const PAGE_SHIFT: u32 = 12;
const PAGE_MASK: u64 = (1 << PAGE_SHIFT) - 1;

pub(super) const CPU: u32 = 19;
pub(super) const TLB_BASE: u32 = 20;
/// Guest virtual address of the page the block runs from (`rip & !0xfff`), latched on entry like
/// `TLB_BASE`: the trampoline sets it from `Cpu::rip`, the chain stub from its target, and a direct
/// link keeps it because links stay within a page. Blocks are keyed by physical address alone
/// (`Jit::translate`), so the same translation runs at whatever virtual address the page is
/// mapped at (ASLR gives every process its own address for the same libc page), and every guest
/// address a block needs is this plus a translation-time offset. Callee-saved, so helper calls
/// keep it.
pub(super) const PAGE_BASE: u32 = 21;
pub(super) const SAVED_ADDRESS: u32 = 22;
pub(super) const SAVED_NZCV: u32 = 23;
/// Callee-saved homes for values that must outlive a slow-path helper call: the result and
/// second operand of a read-modify-write instruction whose store takes the slow path (its flags
/// are recorded only after the store succeeded), a computed branch target, and a value the
/// instruction still needs after a call (a carry-in or a shift count).
pub(super) const SAVED_RESULT: u32 = 24;
pub(super) const SAVED_OPERAND: u32 = 25;
const SAVED_EXTRA: u32 = 26;
/// Used only by the memory thunks in the prologue (`arch_a64::thunk`): the address of the
/// `[rip, block index]` pair of the access that called them, and their return address.
pub(super) const SAVED_INFO: u32 = 27;
pub(super) const SAVED_LR: u32 = 28;
/// The slow path hands its info pair to the thunk in this register.
const SLOW_INFO: u32 = 7;
const ADDRESS: u32 = 2;
const SCRATCH_A: u32 = 3;
const SCRATCH_B: u32 = 4;
const SCRATCH_C: u32 = 5;
pub(super) const VALUE: u32 = 6;
/// Intra-procedure-call scratch register of AAPCS64; holds the target of `blr`.
pub(super) const CALL_TARGET: u32 = 16;

/// Indices of rax, rdx and rsp in `Cpu::gpr`.
const RAX: u8 = 0;
const RDX: u8 = 2;
const RSP: u8 = 4;
/// Without a REX prefix, byte-register numbers 4..=7 mean ah/ch/dh/bh (bits 15:8 of registers
/// 0..=3); with one they mean spl/bpl/sil/dil.
const FIRST_HIGH_BYTE_REG: u8 = 4;
const LAST_HIGH_BYTE_REG: u8 = 7;
const HIGH_BYTE_SHIFT: u32 = 8;
/// REX.B extends the register number embedded in the opcode (push/pop/mov r, imm).
const REX_B: u8 = 1;
const MODRM_MOD_REGISTER: u8 = 3;
const OPCODE_REG_MASK: u8 = 7;
const REX_B_REG_SHIFT: u8 = 3;

/// The x86 "group 1" ALU operation numbers (opcode bits 5:3, or ModRM.reg of 0x81/0x83).
/// `ALU_TEST` is not an x86 value: it stands for the non-storing `and` used by `test`.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_XOR: u8 = 6;
const ALU_CMP: u8 = 7;
const ALU_TEST: u8 = 8;

/// ModRM.reg of the shift group (c0/c1/d0..d3): rol, ror, rcl, rcr, shl, shr, sal (an alias of
/// shl), sar.
/// Shift counts of 8-, 16- and 32-bit operands are masked to five bits (SDM, SHL).
const NARROW_SHIFT_COUNT_MASK: u32 = 0x1f;
/// `Insn::op` of the VEX map-2 (0f38) BMI opcodes: bextr/shlx/sarx/shrx, bzhi/pdep/pext, andn,
/// and the blsr/blsmsk/blsi group (`cpu/bmi.rs`).
const VEX_BMI_SHIFT_OR_BEXTR: u16 = VEX_OP_FLAG | 0x2f7;
const VEX_BMI_BZHI_PEXT_PDEP: u16 = VEX_OP_FLAG | 0x2f5;
const VEX_BMI_ANDN: u16 = VEX_OP_FLAG | 0x2f2;
const VEX_BMI_GROUP17: u16 = VEX_OP_FLAG | 0x2f3;
const BMI_GROUP_BLSR: u8 = 1;
const BMI_GROUP_BLSMSK: u8 = 2;
const BMI_GROUP_BLSI: u8 = 3;
const SHIFT_SHL: u8 = 4;
const SHIFT_SHR: u8 = 5;
const SHIFT_SAL_ALIAS: u8 = 6;
const SHIFT_SAR: u8 = 7;

const PUSH_POP_SIZE: u32 = 8;
/// Condition code 2 (B): CF.
const CONDITION_BELOW: u64 = 2;
/// Width of a W register, which byte and word operations are computed in.
const WORD_BITS: u32 = 32;
/// Bit of the packed CF/OF word of `LazyFlagsOp::SzpCfOf` that OF goes to (CF is bit 0).
const CF_OF_OF_BIT: u32 = crate::cpu::flags::SZP_CF_OF_SRC_OF_BIT;
/// The largest displacement `add_imm`/`sub_imm` encode directly.
const IMM12_LIMIT_I64: i64 = IMM12_LIMIT as i64;

/// Which x86 flag-producing instruction family the host NZCV flags currently mirror. These are
/// the only `LazyFlagsOp`s the JIT produces; `cond_map` depends on the flag semantics of each.
#[derive(Clone, Copy, PartialEq, Debug)]
enum FlagKind {
    Add,
    Sub,
    /// `adc`/`sbb`: the host `adcs`/`sbcs` leave NZCV exactly as `adds`/`subs` would for the same
    /// result, so conditions map as for `Add`/`Sub`; only the lazy record differs (it carries the
    /// carry-in, see `Gen::alu`).
    Adc,
    Sbb,
    Logic,
    /// `inc`/`dec`: N, Z and V of the host flags are right but C is not (x86 leaves CF alone),
    /// so conditions that read CF cannot use them.
    Inc,
    Dec,
}

/// Host NZCV holds the flags of the last inlined ALU instruction. Not set at block start and
/// cleared by `helper`, whose call clobbers NZCV. Inline memory accesses preserve it.
#[derive(Clone, Copy)]
struct LiveFlags {
    kind: FlagKind,
}

enum SlowAccess {
    /// A whole-instruction fallback (`vector_move`): the inline probe, an alignment or SSE gate
    /// check failed, so run the instruction in the interpreter, which raises the right fault.
    Exec,
    Load {
        dst: u32,
        size: u8,
        signed: bool,
        stack: bool,
    },
    Store {
        size: u8,
        stack: bool,
        /// The caller still needs x0 and x1 (result and operand of a read-modify-write) after
        /// the store; the slow path parks them in `SAVED_RESULT`/`SAVED_OPERAND` across the call.
        preserve: bool,
    },
}

/// An out-of-line memory access, emitted after the block body by `emit_slows`. Entered when the
/// inline TLB probe misses, and jumps back to `resume` when the access succeeded.
struct SlowPath {
    entry: Label,
    resume: Label,
    access: SlowAccess,
    /// Index in the block of the instruction making the access.
    insn_index: u32,
    rip: u64,
}

/// A register operand is its number plus whether it is the high byte of a legacy register
/// (ah/ch/dh/bh, bits 15:8 of registers 0..=3).
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

/// What a condition code reduces to right after a flag-setting instruction of a given kind.
enum CondMap {
    /// Test this aarch64 condition with `b.cond`.
    Cond(u32),
    Always,
    Never,
}

/// Maps an x86 condition code (low nibble of the Jcc opcode: O, NO, B, AE, E, NE, BE, A, S, NS,
/// P, NP, L, GE, LE, G) to a test of the host NZCV flags, valid when they were set by the
/// aarch64 instruction that corresponds to `kind`. `None` means the host flags cannot express it
/// and the caller must evaluate it from the lazy flags with `helper_cond`.
///
/// - Sub: aarch64 `subs` carry is the inverse of x86 borrow, so B/AE use CC/CS and BE/A use
///   LS/HI. N, Z and V mean the same as SF, ZF, OF.
/// - Add: carry has the same sense as x86 CF, so B/AE use CS/CC. BE/A (CF | ZF) have no single
///   aarch64 condition.
/// - Logic: x86 clears CF and OF and aarch64 `ands` clears C and V, so O/B are constantly false
///   and NO/AE true, BE reduces to ZF (EQ) and A to NE; L/GE/LE/G compare N against V = 0 as is.
/// - P/NP (parity) are never available: aarch64 has no parity flag.
fn cond_map(kind: FlagKind, condition: u8) -> Option<CondMap> {
    use CondMap::*;
    let c = |cond| Some(Cond(cond));
    let index = condition as usize & 15;
    match kind {
        FlagKind::Sub | FlagKind::Sbb => [
            c(VS),
            c(VC),
            c(CC),
            c(CS),
            c(EQ),
            c(NE),
            c(LS),
            c(HI),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
        FlagKind::Add | FlagKind::Adc => [
            c(VS),
            c(VC),
            c(CS),
            c(CC),
            c(EQ),
            c(NE),
            None,
            None,
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
        // inc/dec: carry is preserved, so conditions on CF and the parity flag are out.
        FlagKind::Inc | FlagKind::Dec => [
            c(VS),
            c(VC),
            None,
            None,
            c(EQ),
            c(NE),
            None,
            None,
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
        FlagKind::Logic => [
            Some(Never),
            Some(Always),
            Some(Never),
            Some(Always),
            c(EQ),
            c(NE),
            c(EQ),
            c(NE),
            c(MI),
            c(PL),
            None,
            None,
            c(LT),
            c(GE),
            c(LE),
            c(GT),
        ][index]
            .take(),
    }
}

/// The register embedded in the low three opcode bits (push/pop/mov r, imm), extended by REX.B.
fn opcode_register(insn: &Insn) -> u8 {
    (insn.op as u8 & OPCODE_REG_MASK) | ((insn.rex & REX_B) << REX_B_REG_SHIFT)
}

/// ES, CS, SS and DS bases are zero in 64-bit mode (the interpreter's `Cpu::linear_address` ignores them
/// too), so only FS and GS add a base.
fn has_segment_base(insn: &Insn) -> bool {
    insn.seg == SEG_FS as u8 || insn.seg == SEG_GS as u8
}

/// Absolute addresses the generated code reaches with a direct `bl`/`b`: stubs and thunks that
/// `arch_a64::prologue` places at the start of the code buffer, within the +-128 MiB of any
/// block (`CODE_SIZE`).
#[derive(Clone)]
pub(crate) struct Addrs {
    /// Tail-call stubs for `helper_exec_pair` and `helper_cond`.
    pub exec: u64,
    pub cond: u64,
    /// Tail-call stub for `helper_set_rotate_flags(cpu, packed CF/OF)`.
    pub rotate_flags: u64,
    pub exit_ok: u64,
    pub exit_fail: u64,
    /// The block-to-block transfer stub (see `arch_a64::chain_stub`), or 0 when chaining is off
    /// and every block returns to the run loop.
    pub chain: u64,
    /// Memory-access thunks, indexed by `size_index` (1, 2, 4, 8 bytes), for the slow path of
    /// the inline TLB probe: zero-extending loads, stores, and stores that keep x0 and x1.
    /// Sign-extending loads exist for 1, 2 and 4 bytes. The stack variants (push/pop/call/ret)
    /// differ only in the fault they raise for a non-canonical address.
    pub load: [u64; 4],
    pub load_signed: [u64; 3],
    pub load_stack: u64,
    pub store: [u64; 4],
    pub store_keep: [u64; 4],
    pub store_stack: u64,
}

/// Index of an operand size of 1, 2, 4 or 8 bytes in the `Addrs` thunk tables.
fn size_index(size: u8) -> usize {
    size.trailing_zeros() as usize
}

pub(crate) struct Gen<'a> {
    pub asm: Asm,
    addrs: &'a Addrs,
    insns: &'a [Insn],
    slow_paths: Vec<SlowPath>,
    /// Out-of-line exits taken after a store set `block_exit_requested`: (label, index of the
    /// storing instruction, rip after it).
    smc_exits: Vec<(Label, u32, u64)>,
    live: Option<LiveFlags>,
    /// An inline SSE instruction of this block already checks CR0.EM/TS and CR4.OSFXSR. Anything
    /// that changes them (mov to cr, clts, lmsw, vmrun...) ends a block, so one check at the first
    /// such instruction covers the rest.
    sse_gated: bool,
    /// The block already checked the AVX-512 state gate (CR4.OSXSAVE, XCR0, CR0.TS).
    evex_gated: bool,
    /// Instructions of this block already added to `Cpu::icount` ahead of an interpreter helper
    /// (the block index of the last one). Exact at translation time because a block is
    /// straight-line code: every instruction before a reachable one has executed.
    prepaid_insns: u32,
    /// Number of instructions lowered inline (not via `helper_exec`), for `Jit::inlined` statistics.
    pub inlined: u32,
    /// Offset of the block's first instruction in its page. Every `rip` in this file is such an
    /// offset from the page start (`PAGE_BASE` holds the start), compared and combined with
    /// wrapping arithmetic.
    entry_rip: u64,
    /// Byte offset of the body entry; the bytes before it are the link entry's checks (none
    /// without chaining).
    pub body_offset: u32,
    /// Label of the link entry's out-of-line exit, bound by `emit_slows`.
    link_entry_miss: Option<Label>,
    /// Patchable slots of exits to fixed addresses in the block's own page.
    pub links: Vec<LinkSite>,
    /// Per instruction: must the lazy flags it sets be recorded (`helpers::flag_record_needed`).
    flag_record_needed: Vec<bool>,
    /// The instruction being lowered has a dead flag record, which `store_lazy_op` leaves out.
    skip_flag_record: bool,
    /// Flag records left out, for `JitStats`.
    pub flag_records_skipped: u32,
    /// Guest loads are acquires and stores are releases, and only naturally aligned accesses stay
    /// inline: x86 is TSO, and plain aarch64 loads and stores are not, which breaks a guest spin
    /// lock (a plain store releases it) once another vCPU runs on another host core.
    pub ordered_memory: bool,
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, addrs: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            asm: Asm::new(base),
            addrs,
            insns,
            slow_paths: Vec::new(),
            smc_exits: Vec::new(),
            live: None,
            sse_gated: false,
            evex_gated: false,
            prepaid_insns: 0,
            inlined: 0,
            entry_rip: 0,
            body_offset: 0,
            link_entry_miss: None,
            links: Vec::new(),
            flag_record_needed: flag_record_needed(insns),
            skip_flag_record: false,
            flag_records_skipped: 0,
            ordered_memory: false,
        }
    }

    fn load_gpr(&mut self, rd: u32, reg: u8) {
        self.asm.ldr(rd, CPU, GPR_OFFSET + GPR_SIZE * reg as u32);
    }

    fn store_gpr(&mut self, reg: u8, rs: u32) {
        self.asm.str(rs, CPU, GPR_OFFSET + GPR_SIZE * reg as u32);
    }

    /// Calls a stub or thunk in the prologue.
    fn call(&mut self, stub: u64) {
        self.asm.bl_abs(stub);
    }

    /// `rd = PAGE_BASE + rip`, where `rip` is the page-relative offset a block carries around
    /// (wrapping: a jump to a lower page is a negative offset). Plain `add`, so host flags stay
    /// live across it.
    fn rip_to(&mut self, rd: u32, rip: u64) {
        let offset = rip as i64;
        if (0..IMM12_LIMIT_I64).contains(&offset) {
            self.asm.add_imm(rd, PAGE_BASE, offset as u32, true);
        } else if offset < 0 && offset > -IMM12_LIMIT_I64 {
            self.asm.sub_imm(rd, PAGE_BASE, (-offset) as u32, true);
        } else {
            self.asm.mov_imm(rd, rip);
            self.asm.add(rd, rd, PAGE_BASE, true);
        }
    }

    fn set_rip(&mut self, rip: u64) {
        self.rip_to(1, rip);
        self.asm.str(1, CPU, RIP_OFFSET);
    }

    /// Leaves the block normally with `retired` instructions completed.
    fn exit_ok(&mut self, retired: u32) {
        self.asm.mov_imm(0, (retired as u64) << EXIT_COUNT_SHIFT);
        self.asm.b_abs(self.addrs.exit_ok);
    }

    /// Ends the block by continuing at the guest address in `x1`, with `retired` instructions
    /// done. With chaining that is a jump through the chain stub, which needs `icount` to
    /// already include the successor's first instruction (see `chain_stub`): the block pays
    /// for its own instructions except those paid ahead of a helper, plus that one. Without
    /// chaining it stores rip and returns to the run loop.
    fn chain_exit(&mut self, retired: u32) {
        if self.addrs.chain == 0 {
            self.asm.str(1, CPU, RIP_OFFSET);
            self.exit_ok(retired);
            return;
        }
        self.pay_icount(retired);
        self.asm.b_abs(self.addrs.chain);
    }

    /// Charges `Cpu::icount` for the `retired` instructions of this block that are not paid yet,
    /// plus the successor's first (see `chain_stub`). Touches no host flags and only
    /// `SCRATCH_A`, so a `jcc` can pay once before its conditional branch.
    fn pay_icount(&mut self, retired: u32) {
        let owed = retired - self.prepaid_insns;
        self.asm.ldr(SCRATCH_A, CPU, ICOUNT_OFFSET);
        self.asm.add_imm(SCRATCH_A, SCRATCH_A, owed, true);
        self.asm.str(SCRATCH_A, CPU, ICOUNT_OFFSET);
        if self.prepaid_insns > 0 {
            // The next block's own accounting must not subtract this block's prepayment.
            self.asm.str(XZR, CPU, PREPAID_ICOUNT_OFFSET);
        }
    }

    /// `chain_exit` to a fixed guest address.
    fn chain_to(&mut self, rip: u64, retired: u32) {
        self.rip_to(1, rip);
        if self.addrs.chain == 0 {
            self.chain_exit(retired);
            return;
        }
        self.pay_icount(retired);
        self.link_slot(rip);
        self.asm.b_abs(self.addrs.chain);
    }

    /// Emits the nop that `Jit::translate` turns into a direct jump to the block at `target_rip`
    /// once it exists, when `target_rip` is in this block's page (see `LinkSite`). It sits right
    /// before the `mov x1, rip; b chain` exit, which therefore still runs while the slot is a nop.
    /// The jump skips the stub and its probes, so it lands on the successor's link entry, which
    /// repeats the stub's checks; icount must already be paid, as for the stub.
    fn link_slot(&mut self, target_rip: u64) {
        if self.addrs.chain == 0 || target_rip >> PAGE_SHIFT != self.entry_rip >> PAGE_SHIFT {
            return;
        }
        self.links.push(LinkSite {
            code_offset: (self.asm.here() - self.asm.base()) as u32,
            target_rip,
        });
        self.asm.emit(NOP_WORD);
    }

    /// The checks `chain_stub` does before it jumps into a block, for a block reached by a
    /// direct link instead: out of time (`icount` has reached `next_event`) or told to leave
    /// (`block_exit_requested`). On a miss it gives back the pre-charge and returns to the run
    /// loop at this block's first instruction, as the stub's miss path does.
    fn link_entry(&mut self) {
        let miss = self.asm.label();
        self.asm.ldr(SCRATCH_A, CPU, ICOUNT_OFFSET);
        self.asm.ldr(SCRATCH_B, CPU, NEXT_EVENT_OFFSET);
        self.asm.cmp(SCRATCH_A, SCRATCH_B, true);
        self.asm.b_cond(CS, miss);
        self.asm.ldrb(SCRATCH_A, CPU, BLOCK_EXIT_REQUESTED_OFFSET);
        self.asm.cbnz(SCRATCH_A, miss, false);
        self.link_entry_miss = Some(miss);
    }

    /// Leaves the block because of a non-zero helper status already in x0; `insn_index` is the
    /// instruction that stopped the block (the stub decides whether it counts as retired).
    fn fail_exit(&mut self, insn_index: u32) {
        self.asm.mov_imm(1, insn_index as u64);
        self.asm.b_abs(self.addrs.exit_fail);
    }

    /// Runs instruction `index` through the interpreter. If `terminal`, the block ends after
    /// it: `exec` already left `Cpu::rip` at the next instruction or the branch target.
    fn helper(&mut self, index: usize, rip: u64, terminal: bool) {
        // Address of the decoded instruction, baked into the code: `Jit::translate` keeps the
        // owning slice alive in the `Block` for as long as this code can run.
        let insn = &self.insns[index] as *const Insn as u64;
        if index as u32 > self.prepaid_insns {
            // `Cpu::icount` is only advanced when the block exits, but the interpreter may read
            // the clock now (rdtsc, port I/O, a load from a device such as the HPET counter, which
            // is how Xen reads it with a moffs `mov`). Pay for the instructions executed since
            // the last payment (the first one of the block was counted by `run`) and record the
            // running total in `prepaid_icount`, which `step_block` subtracts from the block's
            // count. Paying the full `index` again for a second helper would double-count the
            // first payment.
            let owed = index as u32 - self.prepaid_insns;
            self.asm.ldr(SCRATCH_B, CPU, ICOUNT_OFFSET);
            self.asm.add_imm(SCRATCH_B, SCRATCH_B, owed, true);
            self.asm.str(SCRATCH_B, CPU, ICOUNT_OFFSET);
            self.asm.mov_imm(SCRATCH_B, index as u64);
            self.asm.str(SCRATCH_B, CPU, PREPAID_ICOUNT_OFFSET);
            self.prepaid_insns = index as u32;
        }
        // helper_exec_pair(cpu, [insn, rip]) -> status
        let pair = self.asm.literal_pair(insn, rip);
        self.asm.mov(0, CPU);
        self.asm.adr(1, pair);
        self.call(self.addrs.exec);
        let ok = self.asm.label();
        self.asm.cbz(0, ok, true);
        self.fail_exit(index as u32);
        self.asm.bind(ok);
        if terminal {
            self.exit_ok(index as u32 + 1);
        }
        // The interpreter wrote its own flag state and the call clobbered host NZCV.
        self.live = None;
    }

    /// Computes the guest linear address of the memory operand into `dest` (the effective
    /// address plus the FS/GS base). Clobbers `SCRATCH_A`. Only valid for 64-bit address size
    /// (see `memory_inlinable`).
    fn effective_address(&mut self, insn: &Insn, next_rip: u64, dest: u32) {
        if insn.rip_rel {
            self.rip_to(dest, next_rip.wrapping_add(insn.disp as u64));
        } else {
            let mut have = false;
            if insn.base != NO_REG {
                self.load_gpr(dest, insn.base);
                have = true;
            }
            if insn.index != NO_REG {
                if have {
                    self.load_gpr(SCRATCH_A, insn.index);
                    self.asm.add_lsl(dest, dest, SCRATCH_A, insn.scale as u32);
                } else {
                    self.load_gpr(dest, insn.index);
                    if insn.scale != 0 {
                        self.asm.lsl_imm(dest, dest, insn.scale as u32);
                    }
                    have = true;
                }
            }
            if !have {
                self.asm.mov_imm(dest, insn.disp as u64);
            } else if insn.disp > 0 && insn.disp < IMM12_LIMIT_I64 {
                self.asm.add_imm(dest, dest, insn.disp as u32, true);
            } else if insn.disp < 0 && insn.disp > -IMM12_LIMIT_I64 {
                self.asm.sub_imm(dest, dest, (-insn.disp) as u32, true);
            } else if insn.disp != 0 {
                self.asm.mov_imm(SCRATCH_A, insn.disp as u64);
                self.asm.add(dest, dest, SCRATCH_A, true);
            }
        }
        if has_segment_base(insn) {
            self.asm.ldr(
                SCRATCH_A,
                CPU,
                SEG_BASE_OFFSET + SEG_STRIDE * insn.seg as u32,
            );
            self.asm.add(dest, dest, SCRATCH_A, true);
        }
    }

    /// `rd = rn & constant` (64-bit). A constant that is not an encodable bitmask immediate is
    /// materialised in `CALL_TARGET`, which nothing else uses while a probe runs; `and_imm`
    /// emits nothing in that case, so ignoring its result would silently drop the mask.
    fn and_constant(&mut self, rd: u32, rn: u32, constant: u64) {
        if !self.asm.and_imm(rd, rn, constant, true) {
            self.asm.mov_imm(CALL_TARGET, constant);
            self.asm.and(rd, rn, CALL_TARGET, true);
        }
    }

    /// Inline soft-TLB lookup for an access of `size` bytes at `ADDRESS`; branches to `slow` on a
    /// miss. On a hit `SCRATCH_A` points at the TLB entry. `tag_offset` selects the read or
    /// write tag.
    ///
    /// This is the same lookup as the first lines of `Cpu::translate`/`mem_read`/`mem_write`:
    /// the slot is `(la >> 12) & (TLB_ENTRIES_PER_MODE - 1)` within the half that `TLB_BASE`
    /// already points at, and the access hits iff the tag equals the page address. The tag is
    /// compared against the page of the access's *last* byte while the slot comes from its
    /// first, so an access that crosses a page never matches and takes the slow path, which
    /// splits it. Tags are only ever set for validated pages, so non-canonical addresses, MMIO,
    /// not-present pages, permission failures and (for stores) pages holding translated code
    /// all miss. `TLB_TAG_INVALID` (all ones) never equals a page-aligned address.
    ///
    /// Nothing here sets the host flags, so a live NZCV survives a fast-path access.
    fn tlb_probe(&mut self, size: u8, tag_offset: u32, slow: Label) {
        let entry = SCRATCH_A;
        let tag = SCRATCH_B;
        let page = SCRATCH_C;
        self.asm.lsr_imm(entry, ADDRESS, PAGE_SHIFT);
        self.and_constant(entry, entry, (TLB_ENTRIES_PER_MODE - 1) as u64);
        self.asm.add_lsl(entry, TLB_BASE, entry, TLB_ENTRY_SHIFT);
        self.asm.ldr(tag, entry, tag_offset);
        if size == 1 {
            self.and_constant(page, ADDRESS, !PAGE_MASK);
        } else {
            self.asm.add_imm(page, ADDRESS, size as u32 - 1, true);
            self.and_constant(page, page, !PAGE_MASK);
        }
        self.asm.eor(page, tag, page, true);
        self.asm.cbnz(page, slow, true);
        if self.ordered_memory && size > 1 {
            // `ldar` and `stlr` fault on a misaligned address, so such an access takes the slow
            // path, where the interpreter's accessor copes with it.
            self.and_constant(page, ADDRESS, size as u64 - 1);
            self.asm.cbnz(page, slow, true);
        }
    }

    /// Loads `size` bytes at `ADDRESS` into `dst` (zero- or sign-extended to 64 bits). The slow
    /// path preserves `ADDRESS` and the host flags but, being a call, nothing else.
    fn mem_load(&mut self, dst: u32, size: u8, signed: bool, insn_index: u32, rip: u64) {
        self.load_access(dst, size, signed, insn_index, rip, false);
    }

    /// `mem_load` for an implicit stack access: only the slow path differs (#SS, not #GP).
    fn stack_load(&mut self, dst: u32, insn_index: u32, rip: u64) {
        self.load_access(dst, 8, false, insn_index, rip, true);
    }

    fn load_access(
        &mut self,
        dst: u32,
        size: u8,
        signed: bool,
        insn_index: u32,
        rip: u64,
        stack: bool,
    ) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.tlb_probe(size, TLB_TAG_READ_OFFSET, entry);
        // addend = host page - guest page, so the host address is ADDRESS + addend.
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        if self.ordered_memory {
            self.asm.add(SCRATCH_B, ADDRESS, SCRATCH_B, true);
            self.asm.ldar(dst, SCRATCH_B, size);
            if signed {
                match size {
                    1 => self.asm.sxtb(dst, dst, true),
                    2 => self.asm.sxth(dst, dst, true),
                    4 => self.asm.sxtw(dst, dst),
                    _ => {}
                }
            }
        } else {
            self.asm.ldr_r(dst, ADDRESS, SCRATCH_B, size, signed);
        }
        self.asm.bind(resume);
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Load {
                dst,
                size,
                signed,
                stack,
            },
            insn_index,
            rip,
        });
    }

    /// Stores the low `size` bytes of `VALUE` to `ADDRESS`. The slow path reads the value from
    /// `VALUE`. A store to a page holding translated code always misses the write tag; the slow
    /// path then invalidates those blocks (`Cpu::code_written`) and sets `block_exit_requested`,
    /// which `smc_check` must poll after the store.
    fn mem_store(&mut self, size: u8, insn_index: u32, rip: u64) {
        self.store_access(size, insn_index, rip, false);
    }

    /// `mem_store` for an implicit stack access; see `stack_load`.
    fn stack_store(&mut self, insn_index: u32, rip: u64) {
        self.store_access(8, insn_index, rip, true);
    }

    /// `mem_store` for a read-modify-write instruction: x0 (result) and x1 (second operand) are
    /// still wanted afterwards to record the flags, so a slow-path store keeps them.
    fn mem_store_keeping_operands(&mut self, size: u8, insn_index: u32, rip: u64) {
        self.store_access_with(size, insn_index, rip, false, true);
    }

    fn store_access(&mut self, size: u8, insn_index: u32, rip: u64, stack: bool) {
        self.store_access_with(size, insn_index, rip, stack, false);
    }

    fn store_access_with(
        &mut self,
        size: u8,
        insn_index: u32,
        rip: u64,
        stack: bool,
        preserve: bool,
    ) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.tlb_probe(size, TLB_TAG_WRITE_OFFSET, entry);
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        if self.ordered_memory {
            self.asm.add(SCRATCH_B, ADDRESS, SCRATCH_B, true);
            self.asm.stlr(VALUE, SCRATCH_B, size);
        } else {
            self.asm.str_r(VALUE, ADDRESS, SCRATCH_B, size);
        }
        self.asm.bind(resume);
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Store {
                size,
                stack,
                preserve,
            },
            insn_index,
            rip,
        });
    }

    /// Self-modifying code: after a store that may have invalidated this very block, stop before
    /// the next instruction. The store itself retired, so the exit counts `insn_index + 1`.
    fn smc_check(&mut self, insn_index: u32, next_rip: u64) {
        let exit = self.asm.label();
        self.asm.ldrb(SCRATCH_B, CPU, BLOCK_EXIT_REQUESTED_OFFSET);
        self.asm.cbnz(SCRATCH_B, exit, false);
        self.smc_exits.push((exit, insn_index, next_rip));
    }

    /// Emits everything out of line: the self-modifying-code exits and the slow memory paths.
    /// Placed after the block body (and after its final exit branch) so the hot path falls
    /// through. Also resolves all labels.
    fn emit_slows(&mut self) {
        let smc_exits = std::mem::take(&mut self.smc_exits);
        for (exit, insn_index, next_rip) in smc_exits {
            self.asm.bind(exit);
            self.set_rip(next_rip);
            self.exit_ok(insn_index + 1);
        }
        if let Some(miss) = self.link_entry_miss.take() {
            self.asm.bind(miss);
            self.asm.ldr(SCRATCH_A, CPU, ICOUNT_OFFSET);
            self.asm.sub_imm(SCRATCH_A, SCRATCH_A, 1, true);
            self.asm.str(SCRATCH_A, CPU, ICOUNT_OFFSET);
            self.set_rip(self.entry_rip);
            // One instruction retired in the exit code, as in the stub's miss path.
            self.exit_ok(1);
        }
        let slow_paths = std::mem::take(&mut self.slow_paths);
        for slow in slow_paths {
            self.asm.bind(slow.entry);
            if matches!(slow.access, SlowAccess::Exec) {
                // Not `helper`: that pays for the instructions before this one at translation time,
                // which a path that is only sometimes taken cannot do. The exit stub works out the
                // retired count from the index, as for a faulting memory access.
                let insn = &self.insns[slow.insn_index as usize] as *const Insn as u64;
                let pair = self.asm.literal_pair(insn, slow.rip);
                self.asm.mov(0, CPU);
                self.asm.adr(1, pair);
                self.call(self.addrs.exec);
                self.asm.cbz(0, slow.resume, true);
                self.fail_exit(slow.insn_index);
                continue;
            }
            // The thunk keeps the host flags and `ADDRESS` alive across its helper call; see
            // `arch_a64::thunk`. On an exception it leaves the block itself.
            let info = self.asm.literal_pair(slow.rip, slow.insn_index as u64);
            self.asm.adr(SLOW_INFO, info);
            match slow.access {
                SlowAccess::Exec => unreachable!("handled above"),
                SlowAccess::Load {
                    dst,
                    size,
                    signed,
                    stack,
                } => {
                    let thunk = if stack {
                        self.addrs.load_stack
                    } else if signed && size < 8 {
                        self.addrs.load_signed[size_index(size)]
                    } else {
                        self.addrs.load[size_index(size)]
                    };
                    self.call(thunk);
                    if dst != 0 {
                        self.asm.mov(dst, 0);
                    }
                }
                SlowAccess::Store {
                    size,
                    stack,
                    preserve,
                } => {
                    let thunk = if stack {
                        self.addrs.store_stack
                    } else if preserve {
                        self.addrs.store_keep[size_index(size)]
                    } else {
                        self.addrs.store[size_index(size)]
                    };
                    self.call(thunk);
                }
            }
            self.asm.b(slow.resume);
        }
        self.asm.finish();
    }

    /// Records the flags of the instruction just emitted: the result is in x0, the second
    /// operand in x1 (the carry-in for `inc`/`dec`, CF and OF packed for shifts). For logic ops
    /// the interpreter stores 0 in `lazy_src` while this stores the operand; that is equivalent
    /// because `flags::compute` derives a Logic result from the result alone (CF and OF are
    /// cleared), and `inc`/`dec` take their carry-in from the materialised flags, not from the
    /// previous `lazy_src` (`diff_flags` checks all consumers).
    fn store_lazy(&mut self, lazy_op: LazyFlagsOp, operand_size: u8) {
        self.asm.mov_imm(
            SCRATCH_B,
            (lazy_op as u32 | (operand_size as u32) << LAZY_SIZE_SHIFT) as u64,
        );
        self.asm.strh(SCRATCH_B, CPU, LAZY_OP_OFFSET);
        self.asm.str(0, CPU, LAZY_DST_OFFSET);
        self.asm.str(1, CPU, LAZY_SRC_OFFSET);
    }

    /// `store_lazy` for an instruction whose result the host NZCV flags also mirror.
    fn store_lazy_op(&mut self, kind: FlagKind, operand_size: u8) {
        let lazy_op = match kind {
            FlagKind::Add => LazyFlagsOp::Add,
            FlagKind::Sub => LazyFlagsOp::Sub,
            FlagKind::Adc => LazyFlagsOp::Adc,
            FlagKind::Sbb => LazyFlagsOp::Sbb,
            FlagKind::Logic => LazyFlagsOp::Logic,
            FlagKind::Inc => LazyFlagsOp::Inc,
            FlagKind::Dec => LazyFlagsOp::Dec,
        };
        if self.skip_flag_record {
            // A later instruction of the block overwrites these flags before anything can read
            // them; the host flags (`live`) still serve a jcc that follows.
            self.flag_records_skipped += 1;
        } else {
            self.store_lazy(lazy_op, operand_size);
            if matches!(kind, FlagKind::Adc | FlagKind::Sbb) {
                // The carry-in, which `Gen::alu` parked in `SAVED_EXTRA`.
                self.asm.str(SAVED_EXTRA, CPU, LAZY_SRC2_OFFSET);
            }
        }
        self.live = Some(LiveFlags { kind });
    }

    /// Whether a memory operand can use the inline access path: 64-bit address size (a 67 prefix
    /// would need 32-bit wrapping) and no LOCK prefix. Every segment override is fine:
    /// `effective_address` adds the FS/GS bases and 64-bit mode ignores the rest.
    fn memory_inlinable(insn: &Insn) -> bool {
        !insn.addr32 && !insn.lock
    }

    /// An immediate as the operand of an operation of `size` bytes: truncated to the width,
    /// which only differs from the sign-extended decode for 8-byte operations.
    fn immediate_for(size: u8, imm: i64) -> u64 {
        match size {
            1 => imm as u8 as u64,
            2 => imm as u16 as u64,
            4 => imm as u32 as u64,
            _ => imm as u64,
        }
    }

    /// The register an instruction names with `reg` at operand `size`: without a REX prefix
    /// byte-register numbers 4..=7 are ah, ch, dh and bh.
    fn named_register(insn: &Insn, reg: u8, size: u8) -> (u8, bool) {
        if size == 1
            && !insn.rex_present
            && (FIRST_HIGH_BYTE_REG..=LAST_HIGH_BYTE_REG).contains(&reg)
        {
            (reg - FIRST_HIGH_BYTE_REG, true)
        } else {
            (reg, false)
        }
    }

    /// The ModRM.rm operand as a destination or source.
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

    /// The ModRM.reg operand.
    fn reg_dst(insn: &Insn, size: u8) -> Dst {
        let (reg, high) = Self::named_register(insn, insn.reg, size);
        Dst::Reg(reg, high)
    }

    fn reg_src(insn: &Insn, size: u8) -> Src {
        let (reg, high) = Self::named_register(insn, insn.reg, size);
        Src::Reg(reg, high)
    }

    /// Loads register `reg` into `rd`. A high-byte register comes down to bits 7:0 with the rest
    /// clear; otherwise the bits above the operand width are whatever the register holds, so
    /// callers must mask or shift them away.
    fn read_reg(&mut self, rd: u32, reg: u8, high: bool) {
        self.load_gpr(rd, reg);
        if high {
            self.asm.ubfx(rd, rd, HIGH_BYTE_SHIFT, 8, true);
        }
    }

    /// Writes the low `size` bytes of `rs` to register `reg`: 1 and 2 byte operands merge into
    /// the old value, 4 byte ones must already be zero-extended in `rs` (a 32-bit write clears
    /// the upper half). Clobbers `SCRATCH_C`.
    fn write_reg(&mut self, reg: u8, high: bool, size: u8, rs: u32) {
        if size >= 4 {
            self.store_gpr(reg, rs);
            return;
        }
        let (lsb, width) = match (size, high) {
            (2, _) => (0, 16),
            (_, true) => (HIGH_BYTE_SHIFT, 8),
            (_, false) => (0, 8),
        };
        self.load_gpr(SCRATCH_C, reg);
        self.asm.bfi(SCRATCH_C, rs, lsb, width, true);
        self.store_gpr(reg, SCRATCH_C);
    }

    /// Loads the `size`-byte operand `dst` names into `rd` (zero-extended for memory, raw for
    /// registers; see `read_reg`).
    #[allow(clippy::too_many_arguments)]
    fn load_dst(
        &mut self,
        insn: &Insn,
        dst: Dst,
        rd: u32,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) {
        match dst {
            Dst::Reg(reg, high) => self.read_reg(rd, reg, high),
            Dst::Mem => {
                self.effective_address(insn, next_rip, ADDRESS);
                self.mem_load(rd, size, false, insn_index, rip);
            }
        }
    }

    /// Writes x0 to the operand `dst` names. A memory write returns true: the caller must follow
    /// with `smc_check`. `keep_operands` is for a read-modify-write whose flags are recorded
    /// from x0 and x1 afterwards. The address is the one `load_dst` computed; the store may
    /// fault, which is why flags are recorded only after it.
    fn store_dst(
        &mut self,
        dst: Dst,
        size: u8,
        keep_operands: bool,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        match dst {
            Dst::Reg(reg, high) => {
                self.write_reg(reg, high, size, 0);
                false
            }
            Dst::Mem => {
                self.asm.mov(VALUE, 0);
                if keep_operands {
                    self.mem_store_keeping_operands(size, insn_index, rip);
                } else {
                    self.mem_store(size, insn_index, rip);
                }
                true
            }
        }
    }

    /// Emits `dst = dst <alu_op> src` at `size` (1, 2, 4 or 8) and records its flags. `alu_op`
    /// is one of the `ALU_*` values; `ALU_CMP` and `ALU_TEST` discard the result. Returns true
    /// if it stored to memory (the caller follows with `smc_check`).
    ///
    /// Operand order matters: the memory operand is loaded first because its slow path calls a
    /// helper that clobbers x0..x15, and only then is the register/immediate operand loaded.
    ///
    /// Byte and word operations run in W registers on operands shifted up to the top of the
    /// word, so the host NZCV flags describe the narrow operation exactly and `jcc` can use
    /// them; the result is shifted back down for the write.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        insn: &Insn,
        alu_op: u8,
        dst: Dst,
        src: Src,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        // adc and sbb (32- and 64-bit only: a narrow add runs on operands shifted to the top of
        // the word, where the carry-in would have to enter below them) take the guest CF as
        // their carry-in. With no host flags to read it from, the interpreter's
        // `condition_holds` computes it, and that call comes before the operands are loaded
        // because it clobbers every scratch register.
        let carry_in = matches!(alu_op, ALU_ADC | ALU_SBB);
        debug_assert!(!carry_in || size >= 4);
        if carry_in && self.live.is_none() {
            self.carry_from_lazy_record();
        }
        match (dst, src) {
            (Dst::Mem, Src::Reg(reg, high)) => {
                self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
                self.read_reg(1, reg, high);
            }
            (Dst::Mem, Src::Imm(immediate)) => {
                self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
                self.asm.mov_imm(1, immediate);
            }
            (Dst::Reg(reg, high), Src::Mem) => {
                self.effective_address(insn, next_rip, ADDRESS);
                self.mem_load(1, size, false, insn_index, rip);
                self.read_reg(0, reg, high);
            }
            (Dst::Reg(dst_reg, dst_high), Src::Reg(src_reg, src_high)) => {
                self.read_reg(0, dst_reg, dst_high);
                self.read_reg(1, src_reg, src_high);
            }
            (Dst::Reg(reg, high), Src::Imm(immediate)) => {
                self.read_reg(0, reg, high);
                self.asm.mov_imm(1, immediate);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        let is64 = size == 8;
        let (a, b, shift) = if size < 4 {
            let shift = WORD_BITS - 8 * size as u32;
            self.asm.lsl_imm_sized(SCRATCH_A, 0, shift, false);
            self.asm.lsl_imm_sized(SCRATCH_B, 1, shift, false);
            (SCRATCH_A, SCRATCH_B, shift)
        } else {
            (0, 1, 0)
        };
        if carry_in {
            // Inline memory accesses preserve the host flags, so a live carry is still there.
            if let Some(live) = self.live {
                self.carry_from_live_flags(SAVED_EXTRA, live);
            }
            // Host C = the carry-in for `adcs`, its complement for `sbcs`.
            if alu_op == ALU_ADC {
                self.asm.cmp_imm(SAVED_EXTRA, 1, true);
            } else {
                self.asm.subs(XZR, XZR, SAVED_EXTRA, true);
            }
        }
        // `orr` and `eor` do not set flags, so a flag-setting `ands` of the result follows.
        let kind = match alu_op {
            ALU_ADD => {
                self.asm.adds(a, a, b, is64);
                FlagKind::Add
            }
            ALU_OR => {
                self.asm.orr(a, a, b, is64);
                self.asm.ands(XZR, a, a, is64);
                FlagKind::Logic
            }
            ALU_AND | ALU_TEST => {
                self.asm.ands(a, a, b, is64);
                FlagKind::Logic
            }
            ALU_ADC => {
                self.asm.adcs(a, a, b, is64);
                FlagKind::Adc
            }
            ALU_SBB => {
                self.asm.sbcs(a, a, b, is64);
                FlagKind::Sbb
            }
            ALU_SUB | ALU_CMP => {
                self.asm.subs(a, a, b, is64);
                FlagKind::Sub
            }
            _ => {
                debug_assert_eq!(alu_op, ALU_XOR);
                self.asm.eor(a, a, b, is64);
                self.asm.ands(XZR, a, a, is64);
                FlagKind::Logic
            }
        };
        if size < 4 {
            self.asm.lsr_imm_sized(0, a, shift, false);
        }
        // A 32-bit result written to a register zero-extends, which the W-form op already did.
        let stored = if alu_op != ALU_CMP && alu_op != ALU_TEST {
            self.store_dst(dst, size, true, insn_index, rip)
        } else {
            false
        };
        self.store_lazy_op(kind, size);
        stored
    }

    /// Leaves the guest CF (0 or 1) in `SAVED_EXTRA` when no host flags are live (the carry-in of
    /// `adc`/`sbb`, and what `inc`/`dec` must preserve), from the record the last flag writer
    /// left in `Cpu`. Add, sub, logic and the kinds that carry CF in `lazy_src` are worked out
    /// inline, the way `Cpu::condition_holds` would after `flags::compute`; the rest call
    /// `helper_cond`. Clobbers x0..x3 and, on that path, every caller-saved register.
    fn carry_from_lazy_record(&mut self) {
        let (generic, done) = (self.asm.label(), self.asm.label());
        let (subtract, add, source_bit, logic, eflags) = (
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
        );
        self.asm.ldrh(0, CPU, LAZY_OP_OFFSET);
        assert!(self.asm.and_imm(1, 0, 0xff, false));
        self.asm.lsr_imm(0, 0, LAZY_SIZE_SHIFT);
        // The shift that aligns the operand width to the top of 64 bits: 64 - 8 * size
        self.asm.mov_imm(2, 8);
        self.asm.sub(2, 2, 0, false);
        self.asm.lsl_imm(2, 2, 3);
        self.asm.ldr(0, CPU, LAZY_DST_OFFSET);
        self.asm.ldr(3, CPU, LAZY_SRC_OFFSET);
        for (op, label) in [
            (LazyFlagsOp::Sub, subtract),
            (LazyFlagsOp::Add, add),
            (LazyFlagsOp::Logic, logic),
            (LazyFlagsOp::Inc, source_bit),
            (LazyFlagsOp::Dec, source_bit),
            (LazyFlagsOp::SzpCfOf, source_bit),
            (LazyFlagsOp::Eflags, eflags),
        ] {
            self.asm.cmp_imm(1, op as u32, false);
            self.asm.b_cond(EQ, label);
        }
        self.asm.bind(generic);
        self.asm.mov(0, CPU);
        self.asm.mov_imm(1, CONDITION_BELOW);
        self.call(self.addrs.cond);
        self.asm.mov(SAVED_EXTRA, 0);
        self.asm.b(done);
        // CF = (result + src) < src for a sub (a < b), result < src for an add, at the width
        self.asm.bind(subtract);
        self.asm.add(0, 0, 3, true);
        self.asm.bind(add);
        self.asm.lslv(0, 0, 2, true);
        self.asm.lslv(3, 3, 2, true);
        self.asm.cmp(0, 3, true);
        self.asm.cset(SAVED_EXTRA, CC);
        self.asm.b(done);
        self.asm.bind(logic);
        self.asm.mov_imm(SAVED_EXTRA, 0);
        self.asm.b(done);
        self.asm.bind(source_bit);
        assert!(
            self.asm
                .and_imm(SAVED_EXTRA, 3, 1 << SZP_CF_OF_SRC_CF_BIT, true)
        );
        self.asm.b(done);
        self.asm.bind(eflags);
        self.asm.ldr(0, CPU, EFLAGS_OFFSET);
        assert!(self.asm.and_imm(SAVED_EXTRA, 0, CF, true));
        self.asm.bind(done);
    }

    /// Puts the current CF (0 or 1) into `rd`. Needs the host flags (`live`); `inc`/`dec`
    /// leave CF alone, so after one the carry is the one they recorded in `lazy_src`.
    fn carry_from_live_flags(&mut self, rd: u32, live: LiveFlags) {
        match live.kind {
            FlagKind::Add | FlagKind::Adc => self.asm.cset(rd, CS),
            FlagKind::Sub | FlagKind::Sbb => self.asm.cset(rd, CC),
            FlagKind::Logic => self.asm.mov_imm(rd, 0),
            FlagKind::Inc | FlagKind::Dec => self.asm.ldr(rd, CPU, LAZY_SRC_OFFSET),
        }
    }

    /// `inc`/`dec` of a register or memory operand. x86 leaves CF alone, so the lazy flags carry
    /// the current CF in as their second operand; with no host flags to read it from, the
    /// interpreter's `condition_holds` computes it (and its call comes before the operand is
    /// loaded because it clobbers every scratch register).
    #[allow(clippy::too_many_arguments)]
    fn inc_dec(
        &mut self,
        insn: &Insn,
        decrement: bool,
        dst: Dst,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let live = self.live;
        if live.is_none() {
            self.carry_from_lazy_record();
        }
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        match live {
            Some(live) => self.carry_from_live_flags(1, live),
            None => self.asm.mov(1, SAVED_EXTRA),
        }
        let is64 = size == 8;
        if size >= 4 {
            if decrement {
                self.asm.subs_imm(0, 0, 1, is64);
            } else {
                self.asm.adds_imm(0, 0, 1, is64);
            }
        } else {
            let shift = WORD_BITS - 8 * size as u32;
            self.asm.lsl_imm_sized(SCRATCH_A, 0, shift, false);
            self.asm.mov_imm(SCRATCH_B, 1 << shift);
            if decrement {
                self.asm.subs(SCRATCH_A, SCRATCH_A, SCRATCH_B, false);
            } else {
                self.asm.adds(SCRATCH_A, SCRATCH_A, SCRATCH_B, false);
            }
            self.asm.lsr_imm_sized(0, SCRATCH_A, shift, false);
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy_op(
            if decrement {
                FlagKind::Dec
            } else {
                FlagKind::Inc
            },
            size,
        );
        stored
    }

    /// `neg`: flags are those of `0 - operand`.
    #[allow(clippy::too_many_arguments)]
    fn neg(
        &mut self,
        insn: &Insn,
        dst: Dst,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        self.asm.mov(1, 0);
        if size >= 4 {
            self.asm.subs(0, XZR, 0, size == 8);
        } else {
            let shift = WORD_BITS - 8 * size as u32;
            self.asm.lsl_imm_sized(SCRATCH_A, 0, shift, false);
            self.asm.subs(SCRATCH_A, XZR, SCRATCH_A, false);
            self.asm.lsr_imm_sized(0, SCRATCH_A, shift, false);
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy_op(FlagKind::Sub, size);
        stored
    }

    /// `not`: no flags change.
    #[allow(clippy::too_many_arguments)]
    fn not(
        &mut self,
        insn: &Insn,
        dst: Dst,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        self.asm.mov_imm(1, crate::cpu::flags::mask(size));
        self.asm.eor(0, 0, 1, size == 8);
        self.store_dst(dst, size, false, insn_index, rip)
    }

    /// `shl`/`shr`/`sar` of an 8-, 16-, 32- or 64-bit operand by an immediate (`count_register`
    /// false; the count is `immediate`, already masked and non-zero) or by CL. Narrow operands
    /// run in W registers on the zero-extended value (sign-extended for `sar`): CL counts up to
    /// 31 are not masked by the width, and the W-register shifts give the interpreter's answer
    /// for counts beyond it (result 0 or the sign, CF 0 past the width) without special cases.
    /// The caller keeps immediate counts below the width, where the CF position is a fixed bit. The flags are recorded
    /// as `LazyFlagsOp::SzpCfOf` with CF and OF worked out here, the same formulas as
    /// `Cpu::shift_rotate` (OF for every count, not only 1). A CL count of zero changes
    /// neither flags nor value, which is a branch around the flag work; the destination is
    /// still written, as the interpreter does (a 32-bit destination zero-extends).
    #[allow(clippy::too_many_arguments)]
    fn shift(
        &mut self,
        insn: &Insn,
        kind: ShiftKind,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let is64 = size == 8;
        let bits = 8 * size as u32;
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        match (size, kind) {
            // A register read is raw; a count that is zero after masking still zero-extends.
            (4, _) => self.asm.uxtw(0, 0),
            (1, ShiftKind::Sar) => self.asm.sxtb(0, 0, false),
            (2, ShiftKind::Sar) => self.asm.sxth(0, 0, false),
            (1, _) => self.asm.uxtb(0, 0),
            (2, _) => self.asm.uxth(0, 0),
            _ => {}
        }
        let skip = self.asm.label();
        let (result, flags, scratch) = (SCRATCH_A, 1, SCRATCH_C);
        match count {
            Some(count) => {
                match kind {
                    ShiftKind::Shl => self.asm.lsl_imm_sized(result, 0, count, is64),
                    ShiftKind::Shr => self.asm.lsr_imm_sized(result, 0, count, is64),
                    ShiftKind::Sar => self.asm.asr_imm_sized(result, 0, count, is64),
                }
                // CF: the last bit shifted out.
                let carry_bit = if kind == ShiftKind::Shl {
                    bits - count
                } else {
                    count - 1
                };
                self.asm.ubfx(flags, 0, carry_bit, 1, is64);
                match kind {
                    ShiftKind::Shl => {
                        self.asm.ubfx(scratch, result, bits - 1, 1, is64);
                        self.asm.eor(scratch, scratch, flags, false);
                        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
                    }
                    ShiftKind::Shr => {
                        self.asm.ubfx(scratch, 0, bits - 1, 1, is64);
                        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
                    }
                    ShiftKind::Sar => {}
                }
                self.asm.mov(0, result);
            }
            None => {
                self.asm.ldr(flags, CPU, GPR_OFFSET + GPR_SIZE * RCX_INDEX);
                let count_mask = if is64 { 63 } else { 31 };
                assert!(self.asm.and_imm(flags, flags, count_mask, true));
                self.asm.mov(SAVED_EXTRA, flags);
                self.asm.cbz(SAVED_EXTRA, skip, true);
                // x1 = count from here on; the result lands in `result`, CF/OF in `flags`.
                let count_register = SCRATCH_B;
                self.asm.mov(count_register, flags);
                match kind {
                    ShiftKind::Shl => {
                        self.asm.lslv(result, 0, count_register, is64);
                        // CF = (value >> (bits - count)) & 1
                        self.asm.mov_imm(scratch, bits as u64);
                        self.asm.sub(scratch, scratch, count_register, false);
                        self.asm.lsrv(scratch, 0, scratch, is64);
                    }
                    ShiftKind::Shr | ShiftKind::Sar => {
                        if kind == ShiftKind::Shr {
                            self.asm.lsrv(result, 0, count_register, is64);
                        } else {
                            self.asm.asrv(result, 0, count_register, is64);
                        }
                        // CF = (value >> (count - 1)) & 1
                        self.asm.sub_imm(scratch, count_register, 1, false);
                        if kind == ShiftKind::Shr {
                            self.asm.lsrv(scratch, 0, scratch, is64);
                        } else {
                            self.asm.asrv(scratch, 0, scratch, is64);
                        }
                    }
                }
                self.asm.ubfx(flags, scratch, 0, 1, false);
                match kind {
                    ShiftKind::Shl => {
                        self.asm.ubfx(scratch, result, bits - 1, 1, is64);
                        self.asm.eor(scratch, scratch, flags, false);
                        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
                    }
                    ShiftKind::Shr => {
                        self.asm.ubfx(scratch, 0, bits - 1, 1, is64);
                        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
                    }
                    ShiftKind::Sar => {}
                }
                self.asm.mov(0, result);
                self.asm.bind(skip);
            }
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        if count.is_none() {
            let no_flags = self.asm.label();
            self.asm.cbz(SAVED_EXTRA, no_flags, true);
            self.store_lazy(LazyFlagsOp::SzpCfOf, size);
            self.asm.bind(no_flags);
        } else {
            self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        }
        // Not mirrored by the host flags (a zero count leaves the old ones in place, so this is
        // also the right state for the CL form).
        self.live = None;
        stored
    }

    /// `imul r, r/m` and the three-operand `imul r, r/m, imm`, at 32 and 64 bits. SF/ZF/PF
    /// come from the truncated product and CF = OF is set when the full product does not fit:
    /// `LazyFlagsOp::SzpCfOf` with both bits equal.
    fn imul(&mut self, insn: &Insn, size: u8, next_rip: u64, insn_index: u32, rip: u64) {
        if insn.op == 0x1af {
            match Self::rm_src(insn, size) {
                Src::Reg(reg, _) => self.load_gpr(1, reg),
                _ => {
                    self.effective_address(insn, next_rip, ADDRESS);
                    self.mem_load(1, size, false, insn_index, rip);
                }
            }
            self.load_gpr(0, insn.reg);
        } else {
            match Self::rm_src(insn, size) {
                Src::Reg(reg, _) => self.load_gpr(0, reg),
                _ => {
                    self.effective_address(insn, next_rip, ADDRESS);
                    self.mem_load(0, size, false, insn_index, rip);
                }
            }
            self.asm.mov_imm(1, Self::immediate_for(size, insn.imm));
        }
        let (product, check) = (SCRATCH_A, SCRATCH_B);
        if size == 8 {
            self.asm.mul(product, 0, 1, true);
            self.asm.smulh(check, 0, 1);
            self.asm.asr_imm(SCRATCH_C, product, 63);
            self.asm.cmp(check, SCRATCH_C, true);
        } else {
            self.asm.sxtw(0, 0);
            self.asm.sxtw(1, 1);
            self.asm.mul(product, 0, 1, true);
            self.asm.sxtw(check, product);
            self.asm.cmp(product, check, true);
        }
        // CF = OF = (the product does not fit): x1 = 3 or 0 for `store_lazy`.
        self.asm.mov_imm(SCRATCH_C, 3);
        self.asm.csel(1, SCRATCH_C, XZR, NE);
        if size == 8 {
            self.asm.mov(0, product);
        } else {
            self.asm.uxtw(0, product);
        }
        self.store_gpr(insn.reg, 0);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        self.live = None;
    }

    /// `shld`/`shrd r/m, r, imm8` at 32 and 64 bits with a count of 1 to the width minus 1 (a
    /// masked count of zero changes nothing and stays in the interpreter): one `extr` of the two
    /// operands. CF is the last bit shifted out of r/m, OF is set when the sign changed.
    #[allow(clippy::too_many_arguments)]
    fn double_shift(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: u32,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let size = insn.opsize;
        let is64 = size == 8;
        let bits = 8 * size as u32;
        let (result, source, scratch) = (SCRATCH_A, SCRATCH_B, SCRATCH_C);
        let flags = 1;
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        if !is64 {
            self.asm.uxtw(0, 0);
        }
        self.load_gpr(source, insn.reg);
        if left {
            // (r/m:reg) >> (bits - count)
            self.asm.extr(result, 0, source, bits - count, is64);
            self.asm.ubfx(flags, 0, bits - count, 1, is64);
        } else {
            // (reg:r/m) >> count
            self.asm.extr(result, source, 0, count, is64);
            self.asm.ubfx(flags, 0, count - 1, 1, is64);
        }
        self.asm.eor(scratch, result, 0, is64);
        self.asm.ubfx(scratch, scratch, bits - 1, 1, is64);
        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
        self.asm.mov(0, result);
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        self.live = None;
        stored
    }

    /// One-operand `mul` and `imul` (`group3` /4 and /5) at 32 and 64 bits: rDX:rAX = rAX * r/m.
    /// CF = OF = (the high half is not zero for `mul`, not the sign of the low half for `imul`),
    /// SF/ZF/PF from the low half: `LazyFlagsOp::SzpCfOf`, as `Cpu::mul_div` leaves them.
    fn mul_wide(&mut self, insn: &Insn, signed: bool, next_rip: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let (low, high, check) = (SCRATCH_A, SCRATCH_B, SCRATCH_C);
        self.bmi_source(insn, next_rip, insn_index, rip);
        self.load_gpr(1, RAX);
        if is64 {
            self.asm.mul(low, 0, 1, true);
            if signed {
                self.asm.smulh(high, 0, 1);
                self.asm.asr_imm(check, low, 63);
                self.asm.cmp(high, check, true);
            } else {
                self.asm.umulh(high, 0, 1);
                self.asm.cmp_imm(high, 0, true);
            }
        } else {
            // Two 32-bit operands extended to 64 bits multiply exactly in 64 bits.
            if signed {
                self.asm.sxtw(0, 0);
                self.asm.sxtw(1, 1);
            } else {
                self.asm.uxtw(0, 0);
                self.asm.uxtw(1, 1);
            }
            self.asm.mul(low, 0, 1, true);
            self.asm.ubfx(high, low, 32, 32, true);
            if signed {
                self.asm.sxtw(check, low);
                self.asm.cmp(low, check, true);
            } else {
                self.asm.cmp_imm(high, 0, true);
            }
            self.asm.uxtw(low, low);
        }
        // x1 = 3 (CF and OF) or 0 for `store_lazy`
        self.asm.mov_imm(check, 3);
        self.asm.csel(1, check, XZR, NE);
        self.store_gpr(RAX, low);
        self.store_gpr(RDX, high);
        self.asm.mov(0, low);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = None;
    }

    /// Loads the r/m operand of a VEX BMI instruction into x0 (a 32-bit operation reads only the
    /// low half). The memory operand comes first: its slow path calls a helper that clobbers
    /// every scratch register.
    fn bmi_source(&mut self, insn: &Insn, next_rip: u64, insn_index: u32, rip: u64) {
        if insn.md == MODRM_MOD_REGISTER {
            self.load_gpr(0, insn.rm);
        } else {
            self.effective_address(insn, next_rip, ADDRESS);
            self.mem_load(0, insn.opsize, false, insn_index, rip);
        }
    }

    /// shlx/shrx/sarx: ModRM.reg = r/m shifted by the count in vvvv, modulo the width. No flags,
    /// so the host flags stay as they are.
    fn bmi_shift(
        &mut self,
        insn: &Insn,
        kind: ShiftKind,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) {
        let is64 = insn.opsize == 8;
        self.bmi_source(insn, next_rip, insn_index, rip);
        self.load_gpr(1, insn.vex_v);
        assert!(self.asm.and_imm(1, 1, 8 * insn.opsize as u64 - 1, true));
        match kind {
            ShiftKind::Shl => self.asm.lslv(0, 0, 1, is64),
            ShiftKind::Shr => self.asm.lsrv(0, 0, 1, is64),
            ShiftKind::Sar => self.asm.asrv(0, 0, 1, is64),
        }
        self.store_gpr(insn.reg, 0);
    }

    /// bzhi: ModRM.reg = r/m with the bits from index (vvvv bits 7:0) up cleared; an index at or
    /// above the width leaves it whole and sets CF. SF/ZF/PF follow the result, OF and AF clear:
    /// `LazyFlagsOp::SzpCfOf` with CF in the packed word.
    fn bzhi(&mut self, insn: &Insn, next_rip: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let (mask, all_ones) = (2, 3);
        self.bmi_source(insn, next_rip, insn_index, rip);
        self.load_gpr(1, insn.vex_v);
        self.asm.uxtb(1, 1);
        // mask = (1 << index) - 1, or all ones when the index is out of range: the host shifts
        // only by index modulo 64 there, which the compare discards.
        self.asm.mov_imm(mask, 1);
        self.asm.lslv(mask, mask, 1, true);
        self.asm.sub_imm(mask, mask, 1, true);
        self.asm.cmp_imm(1, 8 * insn.opsize as u32, false);
        self.asm.mov_imm(all_ones, u64::MAX);
        self.asm.csel(mask, all_ones, mask, CS);
        self.asm.and(0, 0, mask, is64);
        self.asm.cset(1, CS);
        self.store_gpr(insn.reg, 0);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = None;
    }

    /// andn: ModRM.reg = r/m & !vvvv, flags from the result with CF and OF clear.
    fn andn(&mut self, insn: &Insn, next_rip: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        self.bmi_source(insn, next_rip, insn_index, rip);
        self.load_gpr(1, insn.vex_v);
        self.asm.mov_imm(2, u64::MAX);
        self.asm.eor(1, 1, 2, true);
        self.asm.and(0, 0, 1, is64);
        self.store_gpr(insn.reg, 0);
        self.asm.mov_imm(1, 0);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = None;
    }

    /// blsr, blsmsk and blsi (`group` 1, 2 and 3): vvvv = r/m with its lowest set bit cleared,
    /// the mask up to and including it, or just it. Flags from the result; CF is set when the
    /// source is zero (blsr, blsmsk) or not zero (blsi).
    fn lowest_set_bit(&mut self, insn: &Insn, group: u8, next_rip: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        self.bmi_source(insn, next_rip, insn_index, rip);
        self.asm.cmp_imm(0, 0, is64);
        self.asm
            .cset(1, if group == BMI_GROUP_BLSI { NE } else { EQ });
        if group == BMI_GROUP_BLSI {
            self.asm.sub(2, XZR, 0, is64);
        } else {
            self.asm.sub_imm(2, 0, 1, is64);
        }
        if group == BMI_GROUP_BLSMSK {
            self.asm.eor(0, 0, 2, is64);
        } else {
            self.asm.and(0, 0, 2, is64);
        }
        self.store_gpr(insn.vex_v, 0);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = None;
    }

    /// bsf, bsr, tzcnt and lzcnt of a 32- or 64-bit operand. tzcnt/lzcnt write the count (the
    /// width for a zero source) with CF = (source is zero) and SF/ZF/PF from the count; bsf/bsr
    /// leave the destination alone for a zero source, else write the bit index, and take
    /// SF/ZF/PF from the source with CF clear. Both are `LazyFlagsOp::SzpCfOf`, which is what the
    /// interpreter leaves (every flag but ZF is undefined for bsf/bsr, and QEMU does the same).
    fn bit_scan(&mut self, insn: &Insn, scan: BitScan, next_rip: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let (index, old_destination) = (2, 3);
        self.bmi_source(insn, next_rip, insn_index, rip);
        if !is64 {
            self.asm.uxtw(0, 0);
        }
        self.asm.cmp_imm(0, 0, true);
        match scan {
            BitScan::Tzcnt | BitScan::Lzcnt => {
                self.asm.cset(1, EQ);
                if scan == BitScan::Tzcnt {
                    self.asm.rbit(0, 0, is64);
                }
                self.asm.clz(0, 0, is64);
                self.store_gpr(insn.reg, 0);
            }
            BitScan::Bsf | BitScan::Bsr => {
                if scan == BitScan::Bsf {
                    self.asm.rbit(index, 0, is64);
                    self.asm.clz(index, index, is64);
                } else {
                    self.asm.clz(index, 0, is64);
                    self.asm
                        .mov_imm(old_destination, 8 * insn.opsize as u64 - 1);
                    self.asm.sub(index, old_destination, index, false);
                }
                self.load_gpr(old_destination, insn.reg);
                self.asm.csel(index, old_destination, index, EQ);
                self.store_gpr(insn.reg, index);
                // SzpCfOf of the source with CF = OF = 0
                self.asm.mov_imm(1, 0);
            }
        }
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = None;
    }

    /// `rol`/`ror` of a 32- or 64-bit operand by an immediate or CL. The rotation is inline; CF
    /// and OF are packed (CF in bit 0, OF in bit 1) for `helper_set_rotate_flags`, which merges
    /// them into the guest flags because rotates keep SF/ZF/PF/AF, so the host flags stop
    /// mirroring the guest's afterwards. As for shifts, a CL count that is zero after masking
    /// changes no flag and no value, but a 32-bit destination is still zero-extended. OF is
    /// computed for every count, as in `Cpu::shift_rotate`.
    #[allow(clippy::too_many_arguments)]
    fn rotate(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let is64 = size == 8;
        let bits = 8 * size as u32;
        self.load_dst(insn, dst, 0, size, next_rip, insn_index, rip);
        if size == 4 {
            self.asm.uxtw(0, 0);
        }
        let skip = self.asm.label();
        let (result, flags, scratch) = (SCRATCH_A, 1, SCRATCH_C);
        match count {
            // A left rotation is a right rotation by the complement.
            Some(count) => {
                let amount = if left { bits - count } else { count };
                self.asm.ror_imm(result, 0, amount, is64);
            }
            None => {
                self.asm.ldr(flags, CPU, GPR_OFFSET + GPR_SIZE * RCX_INDEX);
                assert!(self.asm.and_imm(flags, flags, bits as u64 - 1, true));
                self.asm.mov(SAVED_EXTRA, flags);
                self.asm.cbz(SAVED_EXTRA, skip, true);
                if left {
                    self.asm.mov_imm(scratch, bits as u64);
                    self.asm.sub(scratch, scratch, flags, false);
                    self.asm.rorv(result, 0, scratch, is64);
                } else {
                    self.asm.rorv(result, 0, flags, is64);
                }
            }
        }
        if left {
            // CF = the new low bit, OF = new msb ^ CF
            self.asm.ubfx(flags, result, 0, 1, is64);
            self.asm.ubfx(scratch, result, bits - 1, 1, is64);
        } else {
            // CF = the new msb, OF = new msb ^ the bit below it
            self.asm.ubfx(flags, result, bits - 1, 1, is64);
            self.asm.ubfx(scratch, result, bits - 2, 1, is64);
        }
        self.asm.eor(scratch, scratch, flags, false);
        self.asm.bfi(flags, scratch, CF_OF_OF_BIT, 1, false);
        self.asm.mov(0, result);
        if count.is_none() {
            self.asm.bind(skip);
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        let no_flags = self.asm.label();
        if count.is_none() {
            self.asm.cbz(SAVED_EXTRA, no_flags, true);
        }
        // x1 still holds the packed CF/OF.
        self.asm.mov(0, CPU);
        self.call(self.addrs.rotate_flags);
        self.asm.bind(no_flags);
        self.live = None;
        stored
    }

    /// With no live host flags, sets them from the record the previous flag writer left in `Cpu`
    /// when it was a `sub` or `cmp`: `lazy_dst` is the result and `lazy_src` the subtrahend, so
    /// `(dst + src) << s` and `src << s`, with `s` aligning the operand width to 64 bits, compare
    /// exactly as the guest did. Any other record jumps to `generic`. A conditional branch that
    /// starts a block (the second of two on one compare) would otherwise call `helper_cond`.
    /// Clobbers x0..x3.
    fn rebuild_flags_from_sub(&mut self, generic: Label) {
        self.asm.ldrh(0, CPU, LAZY_OP_OFFSET);
        assert!(self.asm.and_imm(1, 0, 0xff, false));
        self.asm.cmp_imm(1, LazyFlagsOp::Sub as u32, false);
        self.asm.b_cond(NE, generic);
        self.asm.lsr_imm(0, 0, LAZY_SIZE_SHIFT);
        self.asm.mov_imm(1, 8);
        self.asm.sub(0, 1, 0, false);
        self.asm.lsl_imm(0, 0, 3);
        self.asm.ldr(2, CPU, LAZY_DST_OFFSET);
        self.asm.ldr(3, CPU, LAZY_SRC_OFFSET);
        self.asm.add(2, 2, 3, true);
        self.asm.lslv(2, 2, 0, true);
        self.asm.lslv(3, 3, 0, true);
        self.asm.cmp(2, 3, true);
    }

    /// Branch condition for x86 condition code `cc` from the host flags: the live ones when
    /// they can express it, otherwise `helper_cond` over the lazy flags and a compare of its
    /// result. Either way the host flags afterwards are not the guest's any more only in the
    /// second case, which is why that one clears `live`. Clobbers x0..x1 and the other
    /// caller-saved registers in the helper case.
    fn host_condition(&mut self, cc: u8) -> CondMap {
        if let Some(map) = self.live.and_then(|live| cond_map(live.kind, cc)) {
            return map;
        }
        let merged = self.asm.label();
        if let Some(CondMap::Cond(condition)) = cond_map(FlagKind::Sub, cc) {
            let generic = self.asm.label();
            self.rebuild_flags_from_sub(generic);
            self.asm.cset(0, condition);
            self.asm.b(merged);
            self.asm.bind(generic);
        }
        self.asm.mov(0, CPU);
        self.asm.mov_imm(1, cc as u64);
        self.call(self.addrs.cond);
        self.asm.bind(merged);
        self.asm.cmp_imm(0, 0, true);
        self.live = None;
        CondMap::Cond(NE)
    }

    /// Conditional branch ending the block; both outcomes exit with `retired` instructions done.
    fn jcc(&mut self, condition: u8, target: u64, next_rip: u64, retired: u32) {
        let chaining = self.addrs.chain != 0;
        if chaining {
            // Both outcomes retire the same instructions, so pay once, ahead of the branch.
            self.pay_icount(retired);
        }
        let taken = self.asm.label();
        match self.live.and_then(|live| cond_map(live.kind, condition)) {
            Some(CondMap::Cond(cond)) => self.asm.b_cond(cond, taken),
            Some(CondMap::Always) => self.asm.b(taken),
            Some(CondMap::Never) => {}
            None => {
                let resume = self.asm.label();
                if let Some(CondMap::Cond(host_condition)) = cond_map(FlagKind::Sub, condition) {
                    let generic = self.asm.label();
                    self.rebuild_flags_from_sub(generic);
                    self.asm.b_cond(host_condition, taken);
                    self.asm.b(resume);
                    self.asm.bind(generic);
                }
                // helper_cond(cpu, cc) -> 0 or 1, from the lazy flags in `Cpu`
                self.asm.mov(0, CPU);
                self.asm.mov_imm(1, condition as u64);
                self.call(self.addrs.cond);
                self.asm.cbnz(0, taken, true);
                self.asm.bind(resume);
            }
        }
        if chaining {
            self.link_slot(next_rip);
            self.rip_to(1, next_rip);
            self.asm.b_abs(self.addrs.chain);
            self.asm.bind(taken);
            self.link_slot(target);
            self.rip_to(1, target);
            self.asm.b_abs(self.addrs.chain);
        } else {
            self.chain_to(next_rip, retired);
            self.asm.bind(taken);
            self.chain_to(target, retired);
        }
    }

    /// Emits instruction `index` inline and returns true, or emits nothing and returns false if
    /// it has no inline lowering (or an unsupported form), in which case the caller uses
    /// `helper`. Only 32- and 64-bit operand sizes are lowered; 8- and 16-bit forms need
    /// partial-register merging and go to the interpreter.
    fn try_inline(&mut self, index: usize, rip: u64, next_rip: u64) -> bool {
        let insn: Insn = self.insns[index];
        self.skip_flag_record = !self.flag_record_needed[index];
        let insn_index = index as u32;
        let retired = insn_index + 1;
        let operand_size = insn.opsize;
        // No 67/66/LOCK prefix and a 32- or 64-bit operand size.
        let wide_operand =
            !insn.addr32 && !insn.lock && !insn.p66 && (operand_size == 4 || operand_size == 8);
        let is_memory = insn.md != MODRM_MOD_REGISTER;
        let memory_ok = Self::memory_inlinable(&insn);
        // No 67 or LOCK prefix, so memory operands can use the inline access path.
        let plain = memory_ok;
        let has_rex = insn.rex_present;
        let mut is_store = false;
        match insn.op {
            // add/or/adc/sbb/and/sub/xor/cmp: r/m8,r8 (0), r/m,r (1), r8,r/m8 (2), r,r/m (3),
            // al,imm8 (4) and eax,imm32 (5). adc and sbb below 32 bits go to the interpreter.
            0x00..=0x3f if insn.op & 7 < 6 && plain => {
                let alu_op = ((insn.op >> 3) & 7) as u8;
                let form = insn.op & 7;
                let size = if form & 1 == 0 { 1 } else { operand_size };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && size < 4 {
                    return false;
                }
                let (dst, src) = match form {
                    0 | 1 => (Self::rm_dst(&insn, size), Self::reg_src(&insn, size)),
                    2 | 3 => (Self::reg_dst(&insn, size), Self::rm_src(&insn, size)),
                    _ => (
                        Dst::Reg(0, false),
                        Src::Imm(Self::immediate_for(size, insn.imm)),
                    ),
                };
                is_store = self.alu(&insn, alu_op, dst, src, size, next_rip, insn_index, rip);
            }
            // group 1: add/or/.../cmp r/m, imm
            0x80 | 0x81 | 0x83 if plain => {
                let alu_op = insn.reg & 7;
                let size = if insn.op == 0x80 { 1 } else { operand_size };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && size < 4 {
                    return false;
                }
                let immediate = Self::immediate_for(size, insn.imm);
                is_store = self.alu(
                    &insn,
                    alu_op,
                    Self::rm_dst(&insn, size),
                    Src::Imm(immediate),
                    size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // test r/m, r and test al/eax, imm
            0x84 | 0x85 | 0xa8 | 0xa9 if plain => {
                let size = if insn.op & 1 == 0 { 1 } else { operand_size };
                let (dst, src) = if insn.op >= 0xa8 {
                    (
                        Dst::Reg(0, false),
                        Src::Imm(Self::immediate_for(size, insn.imm)),
                    )
                } else {
                    (Self::rm_dst(&insn, size), Self::reg_src(&insn, size))
                };
                is_store = self.alu(&insn, ALU_TEST, dst, src, size, next_rip, insn_index, rip);
            }
            // group 3 /0: test r/m, imm; /2 not; /3 neg
            // shld/shrd r/m, r, imm8 at 32 and 64 bits
            0x1a4 | 0x1ac if wide_operand && plain => {
                let count = insn.imm as u32 & (8 * operand_size as u32 - 1);
                if count == 0 {
                    return false;
                }
                is_store = self.double_shift(
                    &insn,
                    insn.op == 0x1a4,
                    Self::rm_dst(&insn, operand_size),
                    count,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // group 3 /4 mul and /5 imul r/m, one operand, at 32 and 64 bits
            0xf7 if wide_operand && plain && matches!(insn.reg & 7, 4 | 5) => {
                self.mul_wide(&insn, insn.reg & 7 == 5, next_rip, insn_index, rip);
            }
            0xf6 | 0xf7 if plain && matches!(insn.reg & 7, 0 | 2 | 3) => {
                let size = if insn.op == 0xf6 { 1 } else { operand_size };
                let dst = Self::rm_dst(&insn, size);
                is_store = match insn.reg & 7 {
                    0 => {
                        let immediate = Self::immediate_for(size, insn.imm);
                        self.alu(
                            &insn,
                            ALU_TEST,
                            dst,
                            Src::Imm(immediate),
                            size,
                            next_rip,
                            insn_index,
                            rip,
                        )
                    }
                    2 => self.not(&insn, dst, size, next_rip, insn_index, rip),
                    _ => self.neg(&insn, dst, size, next_rip, insn_index, rip),
                };
            }
            // group 4/5 /0 inc, /1 dec
            0xfe | 0xff if plain && insn.reg & 7 <= 1 => {
                let size = if insn.op == 0xfe { 1 } else { operand_size };
                is_store = self.inc_dec(
                    &insn,
                    insn.reg & 7 == 1,
                    Self::rm_dst(&insn, size),
                    size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // rol/ror r/m, imm8 | 1 | cl at 32 and 64 bits (a zero immediate count goes to the
            // interpreter)
            0xc1 | 0xd1 | 0xd3 if wide_operand && insn.reg & 7 <= 1 => {
                let count = match insn.op {
                    0xc1 => Some(insn.imm as u32 & (8 * operand_size as u32 - 1)),
                    0xd1 => Some(1),
                    _ => None,
                };
                if count == Some(0) {
                    return false;
                }
                is_store = self.rotate(
                    &insn,
                    insn.reg & 7 == 0,
                    Self::rm_dst(&insn, operand_size),
                    count,
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // shl/shr/sar r/m8 and r/m16 by imm8 | 1 | cl. An immediate count of zero or at least
            // the width goes to the interpreter: it is rare, and CF there is not a fixed bit.
            0xc0 | 0xc1 | 0xd0 | 0xd1 | 0xd2 | 0xd3
                if plain
                    && matches!(insn.reg & 7, SHIFT_SHL..=SHIFT_SAR)
                    && (insn.op & 1 == 0 || (insn.p66 && operand_size == 2)) =>
            {
                let size = if insn.op & 1 == 0 { 1 } else { 2 };
                let count = match insn.op {
                    0xc0 | 0xc1 => Some(insn.imm as u32 & NARROW_SHIFT_COUNT_MASK),
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
                is_store = self.shift(
                    &insn,
                    kind,
                    Self::rm_dst(&insn, size),
                    count,
                    size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // shl/shr/sar r/m, imm8 | 1 | cl at 32 and 64 bits (/6 is the alias of /4). Rotates
            // and a zero immediate count go to the interpreter.
            0xc1 | 0xd1 | 0xd3 if wide_operand && matches!(insn.reg & 7, SHIFT_SHL..=SHIFT_SAR) => {
                let kind = match insn.reg & 7 {
                    SHIFT_SHL | SHIFT_SAL_ALIAS => ShiftKind::Shl,
                    SHIFT_SHR => ShiftKind::Shr,
                    _ => ShiftKind::Sar,
                };
                let count = match insn.op {
                    0xc1 => Some(insn.imm as u32 & (8 * operand_size as u32 - 1)),
                    0xd1 => Some(1),
                    _ => None,
                };
                if count == Some(0) {
                    return false;
                }
                is_store = self.shift(
                    &insn,
                    kind,
                    Self::rm_dst(&insn, operand_size),
                    count,
                    operand_size,
                    next_rip,
                    insn_index,
                    rip,
                );
            }
            // mov r/m8,r8 (88), r/m,r (89), r8,r/m8 (8a), r,r/m (8b)
            0x88..=0x8b if plain => {
                let size = if insn.op & 1 == 0 { 1 } else { operand_size };
                if insn.op < 0x8a {
                    // store, or register to register
                    let (src_reg, src_high) = Self::named_register(&insn, insn.reg, size);
                    self.read_reg(VALUE, src_reg, src_high);
                    match Self::rm_dst(&insn, size) {
                        Dst::Mem => {
                            self.effective_address(&insn, next_rip, ADDRESS);
                            self.mem_store(size, insn_index, rip);
                            is_store = true;
                        }
                        Dst::Reg(reg, high) => {
                            if size == 4 {
                                self.asm.uxtw(VALUE, VALUE);
                            }
                            self.write_reg(reg, high, size, VALUE);
                        }
                    }
                } else {
                    let (dst_reg, dst_high) = Self::named_register(&insn, insn.reg, size);
                    match Self::rm_src(&insn, size) {
                        Src::Reg(reg, high) => {
                            self.read_reg(VALUE, reg, high);
                            if size == 4 {
                                self.asm.uxtw(VALUE, VALUE);
                            }
                        }
                        _ => {
                            self.effective_address(&insn, next_rip, ADDRESS);
                            self.mem_load(VALUE, size, false, insn_index, rip);
                        }
                    }
                    self.write_reg(dst_reg, dst_high, size, VALUE);
                }
            }
            // group 11 /0: mov r/m8, imm8 (c6) and mov r/m, imm (c7)
            0xc6 | 0xc7 if plain && insn.reg & 7 == 0 => {
                let size = if insn.op == 0xc6 { 1 } else { operand_size };
                let immediate = Self::immediate_for(size, insn.imm);
                match Self::rm_dst(&insn, size) {
                    Dst::Mem => {
                        self.effective_address(&insn, next_rip, ADDRESS);
                        self.asm.mov_imm(VALUE, immediate);
                        self.mem_store(size, insn_index, rip);
                        is_store = true;
                    }
                    Dst::Reg(reg, high) => {
                        self.asm.mov_imm(VALUE, immediate);
                        self.write_reg(reg, high, size, VALUE);
                    }
                }
            }
            // mov r8, imm8 and mov r, imm32/imm64
            0xb0..=0xbf if plain => {
                let size = if insn.op < 0xb8 { 1 } else { operand_size };
                let (reg, high) = Self::named_register(&insn, opcode_register(&insn), size);
                self.asm.mov_imm(VALUE, Self::immediate_for(size, insn.imm));
                self.write_reg(reg, high, size, VALUE);
            }
            // setcc r/m8
            0x190..=0x19f if plain => {
                match self.host_condition((insn.op & 15) as u8) {
                    CondMap::Cond(condition) => self.asm.cset(VALUE, condition),
                    CondMap::Always => self.asm.mov_imm(VALUE, 1),
                    CondMap::Never => self.asm.mov_imm(VALUE, 0),
                }
                match Self::rm_dst(&insn, 1) {
                    Dst::Reg(reg, high) => self.write_reg(reg, high, 1, VALUE),
                    Dst::Mem => {
                        self.effective_address(&insn, next_rip, ADDRESS);
                        self.mem_store(1, insn_index, rip);
                        is_store = true;
                    }
                }
            }
            // cmovcc r, r/m at 32 and 64 bits. The source is always read (it can fault) and a
            // 32-bit destination is zero-extended whether or not it moves.
            0x140..=0x14f if wide_operand => {
                let condition = self.host_condition((insn.op & 15) as u8);
                match Self::rm_src(&insn, operand_size) {
                    Src::Reg(reg, _) => self.load_gpr(VALUE, reg),
                    _ => {
                        self.effective_address(&insn, next_rip, ADDRESS);
                        self.mem_load(VALUE, operand_size, false, insn_index, rip);
                    }
                }
                self.load_gpr(0, insn.reg);
                if operand_size == 4 {
                    self.asm.uxtw(VALUE, VALUE);
                    self.asm.uxtw(0, 0);
                }
                match condition {
                    CondMap::Cond(condition) => self.asm.csel(0, VALUE, 0, condition),
                    CondMap::Always => self.asm.mov(0, VALUE),
                    CondMap::Never => {}
                }
                self.store_gpr(insn.reg, 0);
            }
            // imul r, r/m and imul r, r/m, imm at 32 and 64 bits
            0x1af | 0x69 | 0x6b if wide_operand => {
                self.imul(&insn, operand_size, next_rip, insn_index, rip);
            }
            // group 5 /2 call near, /4 jmp near (r/m64), /6 push r/m64
            0xff if plain && !insn.p66 && matches!(insn.reg & 7, 2 | 4 | 6) => {
                match Self::rm_src(&insn, 8) {
                    Src::Reg(reg, _) => self.load_gpr(VALUE, reg),
                    _ => {
                        self.effective_address(&insn, next_rip, ADDRESS);
                        self.mem_load(VALUE, 8, false, insn_index, rip);
                    }
                }
                match insn.reg & 7 {
                    2 => {
                        // The target is read before the push, so `call [rsp + n]` sees the old
                        // rsp. It waits in a callee-saved register across the push's slow path.
                        self.asm.mov(SAVED_OPERAND, VALUE);
                        self.load_gpr(ADDRESS, RSP);
                        self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                        self.rip_to(VALUE, next_rip);
                        self.stack_store(insn_index, rip);
                        self.store_gpr(RSP, ADDRESS);
                        self.asm.mov(1, SAVED_OPERAND);
                        self.chain_exit(retired);
                        self.inlined += 1;
                        return true;
                    }
                    4 => {
                        self.asm.mov(1, VALUE);
                        self.chain_exit(retired);
                        self.inlined += 1;
                        return true;
                    }
                    _ => {
                        self.load_gpr(ADDRESS, RSP);
                        self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                        self.stack_store(insn_index, rip);
                        self.store_gpr(RSP, ADDRESS);
                        is_store = true;
                    }
                }
            }
            // lfence, mfence, sfence: one CPU, so ordering is free
            0x1ae
                if insn.md == MODRM_MOD_REGISTER
                    && insn.rep == 0
                    && matches!(insn.reg & 7, 5..=7) => {}
            // cdqe / cwde
            0x98 if wide_operand => {
                self.load_gpr(VALUE, 0);
                if operand_size == 8 {
                    self.asm.sxtw(VALUE, VALUE);
                } else {
                    self.asm.sxth(VALUE, VALUE, false);
                }
                self.store_gpr(0, VALUE);
            }
            // cqo / cdq: rdx (edx, zero-extended) gets the sign of rax (eax)
            0x99 if wide_operand => {
                self.load_gpr(VALUE, 0);
                self.asm.asr_imm_sized(
                    VALUE,
                    VALUE,
                    8 * operand_size as u32 - 1,
                    operand_size == 8,
                );
                self.store_gpr(2, VALUE);
            }
            // lea: the segment base is not part of the result, so FS/GS forms fall back
            0x8d if wide_operand && is_memory => {
                if has_segment_base(&insn) {
                    return false;
                }
                self.effective_address(&insn, next_rip, ADDRESS);
                if operand_size == 4 {
                    self.asm.uxtw(ADDRESS, ADDRESS);
                }
                self.store_gpr(insn.reg, ADDRESS);
            }
            // movsxd r64, r/m32 (REX.W only)
            0x63 if operand_size == 8
                && !insn.addr32
                && !insn.lock
                && !insn.p66
                && (!is_memory || memory_ok) =>
            {
                if is_memory {
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.mem_load(VALUE, 4, true, insn_index, rip);
                } else {
                    self.load_gpr(VALUE, insn.rm);
                    self.asm.sxtw(VALUE, VALUE);
                }
                self.store_gpr(insn.reg, VALUE);
            }
            // movzx/movsx r, r/m8 and r/m16 (0f b6, 0f b7, 0f be, 0f bf)
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if wide_operand && (!is_memory || memory_ok) => {
                let source_size: u8 = if insn.op & 1 == 0 { 1 } else { 2 };
                let signed = insn.op >= 0x1b8;
                if is_memory {
                    self.effective_address(&insn, next_rip, ADDRESS);
                    self.mem_load(VALUE, source_size, signed, insn_index, rip);
                } else {
                    if source_size == 1
                        && !has_rex
                        && (FIRST_HIGH_BYTE_REG..=LAST_HIGH_BYTE_REG).contains(&insn.rm)
                    {
                        return false;
                    }
                    self.load_gpr(VALUE, insn.rm);
                    match (source_size, signed) {
                        (1, false) => self.asm.uxtb(VALUE, VALUE),
                        (2, false) => self.asm.uxth(VALUE, VALUE),
                        (1, true) => self.asm.sxtb(VALUE, VALUE, true),
                        _ => self.asm.sxth(VALUE, VALUE, true),
                    }
                }
                if operand_size == 4 {
                    self.asm.uxtw(VALUE, VALUE);
                }
                self.store_gpr(insn.reg, VALUE);
            }
            // BMI1/BMI2 (VEX.LZ, map 0f38): shlx/shrx/sarx f7 with 66/F2/F3, bzhi f5, andn f2 and
            // the blsr/blsmsk/blsi group f3. bextr and the rest go to the interpreter.
            VEX_BMI_SHIFT_OR_BEXTR | VEX_BMI_BZHI_PEXT_PDEP | VEX_BMI_ANDN | VEX_BMI_GROUP17
                if insn.vex
                    && !insn.vex_l
                    && !insn.lock
                    && (operand_size == 4 || operand_size == 8)
                    && (!is_memory || memory_ok) =>
            {
                match (insn.op, insn.rep, insn.p66) {
                    (VEX_BMI_SHIFT_OR_BEXTR, 0, true) => {
                        self.bmi_shift(&insn, ShiftKind::Shl, next_rip, insn_index, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, 2, false) => {
                        self.bmi_shift(&insn, ShiftKind::Shr, next_rip, insn_index, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, 1, false) => {
                        self.bmi_shift(&insn, ShiftKind::Sar, next_rip, insn_index, rip)
                    }
                    (VEX_BMI_BZHI_PEXT_PDEP, 0, false) => {
                        self.bzhi(&insn, next_rip, insn_index, rip)
                    }
                    (VEX_BMI_ANDN, 0, false) => self.andn(&insn, next_rip, insn_index, rip),
                    (VEX_BMI_GROUP17, 0, false)
                        if matches!(
                            insn.reg & 7,
                            BMI_GROUP_BLSR | BMI_GROUP_BLSMSK | BMI_GROUP_BLSI
                        ) =>
                    {
                        self.lowest_set_bit(&insn, insn.reg & 7, next_rip, insn_index, rip)
                    }
                    _ => return false,
                }
            }
            // bsf/bsr (no prefix) and tzcnt/lzcnt (F3) at 32 and 64 bits
            0x1bc | 0x1bd if wide_operand && plain && matches!(insn.rep, 0 | 1) => {
                let scan = match (insn.op, insn.rep) {
                    (0x1bc, 0) => BitScan::Bsf,
                    (0x1bd, 0) => BitScan::Bsr,
                    (0x1bc, _) => BitScan::Tzcnt,
                    _ => BitScan::Lzcnt,
                };
                self.bit_scan(&insn, scan, next_rip, insn_index, rip);
            }
            // push r64. rsp is written back only after the store, so a faulting push leaves it
            // unchanged; the value is read from the register file, so `push rsp` pushes the old rsp.
            0x50..=0x57 if !insn.p66 => {
                let reg = opcode_register(&insn);
                self.load_gpr(ADDRESS, RSP);
                self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.load_gpr(VALUE, reg);
                self.stack_store(insn_index, rip);
                self.store_gpr(RSP, ADDRESS);
                is_store = true;
            }
            // swapgs, clac, stac and cli: the kernel's syscall entry and exit. Only CPL 0 is
            // lowered; anywhere else the interpreter raises the fault.
            0x101 if !insn.lock && matches!(insn.modrm, MODRM_SWAPGS | MODRM_CLAC | MODRM_STAC) => {
                self.ring0_system(&insn, insn_index, rip)
            }
            0xfa if !insn.lock => self.ring0_system(&insn, insn_index, rip),
            // push imm32 / imm8, sign-extended to 64 bits by the decoder. The kernel's syscall
            // and interrupt entries build their frames with these.
            0x68 | 0x6a if !insn.p66 && !insn.lock => {
                self.load_gpr(ADDRESS, RSP);
                self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.asm.mov_imm(VALUE, insn.imm as u64);
                self.stack_store(insn_index, rip);
                self.store_gpr(RSP, ADDRESS);
                is_store = true;
            }
            // pop r64. rsp is stored before the destination so `pop rsp` ends up with the
            // popped value.
            0x58..=0x5f if !insn.p66 => {
                let reg = opcode_register(&insn);
                self.load_gpr(ADDRESS, RSP);
                self.stack_load(VALUE, insn_index, rip);
                self.asm.add_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.store_gpr(RSP, ADDRESS);
                self.store_gpr(reg, VALUE);
            }
            // nop, prefetchw, prefetch/hint-nop group (`xchg r8, rax` with REX.B is not a nop)
            0x90 | 0x10d | 0x118..=0x11f => {
                // `f3 90` is pause: an SVM guest can intercept it, so it goes through `exec`
                if insn.op == 0x90 && (insn.rex & REX_B != 0 || insn.rep != 0) {
                    return false;
                }
            }
            // jcc rel8 / rel32
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next_rip.wrapping_add(insn.imm as u64);
                self.jcc((insn.op & 15) as u8, target, next_rip, retired);
                self.inlined += 1;
                return true;
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.chain_to(next_rip.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            // call rel32. The block ends here, so no `smc_check` is needed after the push: the chain
            // stub looks at `block_exit_requested` itself.
            0xe8 => {
                self.load_gpr(ADDRESS, RSP);
                self.asm.sub_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.rip_to(VALUE, next_rip);
                self.stack_store(insn_index, rip);
                self.store_gpr(RSP, ADDRESS);
                self.chain_to(next_rip.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            // ret
            0xc3 => {
                self.load_gpr(ADDRESS, RSP);
                self.stack_load(VALUE, insn_index, rip);
                self.asm.add_imm(ADDRESS, ADDRESS, PUSH_POP_SIZE, true);
                self.store_gpr(RSP, ADDRESS);
                self.asm.mov(1, VALUE);
                self.chain_exit(retired);
                self.inlined += 1;
                return true;
            }
            // movups/movupd, movaps/movapd, movdqa/movdqu
            0x110 | 0x111 | 0x128 | 0x129 | 0x16f | 0x17f if plain => {
                let Some(form) = vector_move(&insn) else {
                    return false;
                };
                is_store = self.vector_move(&insn, form, next_rip, insn_index, rip);
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
                if plain =>
            {
                let Some(form) = vector_alu(&insn) else {
                    return false;
                };
                // pmovmskb has no memory form (the decoder-level #UD stays with the interpreter)
                if form.op == VectorOp::MoveMask && is_memory {
                    return false;
                }
                is_store = self.vector_alu(&insn, form, next_rip, insn_index, rip);
            }
            // AVX-512 data movement, integer logic and arithmetic, byte compares into an opmask
            // register, and kmov of an opmask register (see `evex_form`)
            _ if insn.evex || insn.vex => {
                let Some(form) = evex_form(&insn) else {
                    return false;
                };
                // A memory operand needs the inline access path (no 67 prefix, no LOCK)
                if is_memory && !plain {
                    return false;
                }
                is_store = self.evex(&insn, form, next_rip, insn_index, rip);
            }
            _ => return false,
        }
        if is_store {
            self.smc_check(insn_index, next_rip);
        }
        self.inlined += 1;
        true
    }

    /// swapgs, clac, stac or cli, inline at CPL 0. At any other CPL (or, for cli, CPL > IOPL,
    /// which CPL 0 never is) the out-of-line interpreter fallback raises #GP or #UD. None of the
    /// four can be intercepted by an SVM guest's host, so a nested guest runs them here too.
    fn ring0_system(&mut self, insn: &Insn, insn_index: u32, rip: u64) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Exec,
            insn_index,
            rip,
        });
        self.asm.ldrb(SCRATCH_A, CPU, CPL_OFFSET);
        self.asm.cbnz(SCRATCH_A, entry, false);
        if insn.op == 0x101 && insn.modrm == MODRM_SWAPGS {
            self.asm.ldr(SCRATCH_A, CPU, GS_BASE_OFFSET);
            self.asm.ldr(SCRATCH_B, CPU, KGSBASE_OFFSET);
            self.asm.str(SCRATCH_B, CPU, GS_BASE_OFFSET);
            self.asm.str(SCRATCH_A, CPU, KGSBASE_OFFSET);
        } else {
            self.asm.ldr(SCRATCH_A, CPU, EFLAGS_OFFSET);
            match (insn.op, insn.modrm) {
                (0xfa, _) => self.and_constant(SCRATCH_A, SCRATCH_A, !IF),
                (_, MODRM_CLAC) => self.and_constant(SCRATCH_A, SCRATCH_A, !AC),
                _ => {
                    self.asm.mov_imm(SCRATCH_B, AC);
                    self.asm.orr(SCRATCH_A, SCRATCH_A, SCRATCH_B, true);
                }
            }
            self.asm.str(SCRATCH_A, CPU, EFLAGS_OFFSET);
        }
        self.asm.bind(resume);
        // The fallback's call clobbered host NZCV.
        self.live = None;
    }

    /// Starts an inline SSE instruction: registers its out-of-line interpreter fallback and, for
    /// the first one of the block, checks the SSE gate (#UD for CR0.EM or no CR4.OSFXSR, #NM for
    /// CR0.TS), jumping to the fallback so the interpreter raises the exception. Returns the
    /// fallback's entry and the label to bind where the inline code ends.
    fn sse_entry(&mut self, insn_index: u32, rip: u64) -> (Label, Label) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Exec,
            insn_index,
            rip,
        });
        if !self.sse_gated {
            self.sse_gated = true;
            self.asm.ldr(SCRATCH_A, CPU, CR0_OFFSET);
            self.and_constant(SCRATCH_A, SCRATCH_A, CR0_EM | CR0_TS);
            self.asm.cbnz(SCRATCH_A, entry, true);
            self.asm.ldr(SCRATCH_A, CPU, CR4_OFFSET);
            self.and_constant(SCRATCH_A, SCRATCH_A, CR4_OSFXSR);
            self.asm.cbz(SCRATCH_A, entry, true);
        }
        (entry, resume)
    }

    /// Starts an inline AVX-512 instruction: registers its out-of-line interpreter fallback and,
    /// for the first one of the block, checks `Cpu::evex_gate` (#UD without CR4.OSXSAVE or the
    /// SSE, AVX and AVX-512 state in XCR0, #NM under CR0.TS), jumping to the fallback so the
    /// interpreter raises the exception. Like the SSE gate it holds for the whole block: whatever
    /// changes CR0, CR4 or XCR0 ends it.
    fn evex_entry(&mut self, insn_index: u32, rip: u64) -> (Label, Label) {
        let entry = self.asm.label();
        let resume = self.asm.label();
        self.slow_paths.push(SlowPath {
            entry,
            resume,
            access: SlowAccess::Exec,
            insn_index,
            rip,
        });
        if !self.evex_gated {
            self.evex_gated = true;
            self.asm.ldr(SCRATCH_A, CPU, CR4_OFFSET);
            self.and_constant(SCRATCH_A, SCRATCH_A, CR4_OSXSAVE);
            self.asm.cbz(SCRATCH_A, entry, true);
            self.asm.ldr(SCRATCH_A, CPU, XCR0_OFFSET);
            self.and_constant(SCRATCH_A, SCRATCH_A, XCR0_EVEX_STATE);
            self.asm
                .sub_imm(SCRATCH_A, SCRATCH_A, XCR0_EVEX_STATE as u32, true);
            self.asm.cbnz(SCRATCH_A, entry, true);
            self.asm.ldr(SCRATCH_A, CPU, CR0_OFFSET);
            self.and_constant(SCRATCH_A, SCRATCH_A, CR0_TS);
            self.asm.cbnz(SCRATCH_A, entry, true);
        }
        (entry, resume)
    }

    /// Zeroes the 128-bit lanes `from..4` of vector register `register`: every VEX or EVEX write
    /// of a vector shorter than 512 bits clears the rest of the register.
    fn zero_vector_lanes(&mut self, register: u8, from: u32) {
        for lane in from..VECTOR_LANES {
            let offset = vector_lane_offset(register, lane) as u32;
            self.asm.str(XZR, CPU, offset);
            self.asm.str(XZR, CPU, offset + XMM_HALF);
        }
    }

    /// Points `SCRATCH_B` at the host address of the memory operand, `bytes` long and inside one
    /// page, for a read or a write; anything else (an `aligned` operand off a `bytes` boundary,
    /// a TLB miss, a page crossing) jumps to `slow`, the interpreter. The operand's compressed
    /// disp8 is scaled by the vector length, as in `Cpu::evex_address`.
    fn evex_memory(
        &mut self,
        insn: &Insn,
        bytes: u32,
        aligned: bool,
        write: bool,
        next_rip: u64,
        slow: Label,
    ) {
        let mut insn = *insn;
        if insn.evex_disp8 {
            insn.disp *= bytes as i64;
        }
        self.effective_address(&insn, next_rip, ADDRESS);
        if aligned {
            self.and_constant(SCRATCH_A, ADDRESS, bytes as u64 - 1);
            self.asm.cbnz(SCRATCH_A, slow, true);
        }
        let tag = if write {
            TLB_TAG_WRITE_OFFSET
        } else {
            TLB_TAG_READ_OFFSET
        };
        self.tlb_probe(bytes as u8, tag, slow);
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        self.asm.add(SCRATCH_B, ADDRESS, SCRATCH_B, true);
    }

    /// Per byte, `predicate` (the vpcmp immediate: eq, lt, le, false, ne, nlt, nle, true) of v0
    /// against v1 into v0, all ones where it holds.
    fn compare_bytes(&mut self, signed: bool, predicate: u8) {
        let (greater, greater_equal) = if signed {
            (VEC_CMGT, VEC_CMGE)
        } else {
            (VEC_CMHI, VEC_CMHS)
        };
        let (a, b) = (VECTOR_DST, VECTOR_SRC);
        match predicate {
            VPCMP_EQUAL => self.asm.vec_op(VEC_CMEQ, 0, a, a, b),
            // a < b is b > a
            VPCMP_LESS => self.asm.vec_op(greater, 0, a, b, a),
            VPCMP_LESS_EQUAL => self.asm.vec_op(greater_equal, 0, a, b, a),
            VPCMP_FALSE => self.asm.vec_op(VEC_EOR, 0, a, a, a),
            VPCMP_NOT_EQUAL => {
                self.asm.vec_op(VEC_CMEQ, 0, a, a, b);
                self.asm.vec_op(VEC_NOT, 0, a, a, 0);
            }
            VPCMP_NOT_LESS => self.asm.vec_op(greater_equal, 0, a, a, b),
            VPCMP_NOT_LESS_EQUAL => self.asm.vec_op(greater, 0, a, a, b),
            _ => self.asm.vec_op(VEC_CMEQ, 0, a, a, a),
        }
    }

    /// Loads the write mask register of `insn` into x0 and the address of the expansion table for
    /// `element_log2` into x1, for `commit_vector_lane`.
    fn begin_masked_write(&mut self, insn: &Insn, element_log2: u32) {
        self.asm
            .ldr(0, CPU, OPMASK_OFFSET + OPMASK_SIZE * insn.evex_mask as u32);
        let table = OPMASK_EXPANSION[element_log2 as usize].as_ptr() as u64;
        self.asm.mov_imm(1, table);
    }

    /// Stores the lane in v0 to lane `lane` of vector register `register`, under the write mask
    /// when the instruction has one (`begin_masked_write` came first): elements whose mask bit is
    /// clear keep the old value, or become zero with EVEX.z. The mask bits of the lane are turned
    /// into an all-ones-or-zero vector half by half through `OPMASK_EXPANSION`.
    fn commit_vector_lane(&mut self, insn: &Insn, element_log2: u32, register: u8, lane: u32) {
        let offset = vector_lane_offset(register, lane) as u32;
        if insn.evex_mask == 0 {
            self.asm.str_q(VECTOR_DST, CPU, offset);
            return;
        }
        let (select, old) = (2, 3);
        let bits_per_half = 8 >> element_log2;
        let elements_per_lane = 16 >> element_log2;
        let halves = [VALUE, SCRATCH_C];
        for (half, register) in halves.into_iter().enumerate() {
            let shift = lane * elements_per_lane + half as u32 * bits_per_half;
            self.asm.ubfx(SCRATCH_A, 0, shift, bits_per_half, true);
            self.asm
                .add_lsl(SCRATCH_A, 1, SCRATCH_A, OPMASK_EXPANSION_SHIFT);
            self.asm.ldr(register, SCRATCH_A, 0);
        }
        self.asm.fmov_d_x(select, VALUE);
        self.asm.ins_d1(select, SCRATCH_C);
        if insn.evex_zeroing {
            self.asm.vec_op(VEC_AND, 0, VECTOR_DST, VECTOR_DST, select);
            self.asm.str_q(VECTOR_DST, CPU, offset);
        } else {
            self.asm.ldr_q(old, CPU, offset);
            self.asm.vec_op(VEC_BSL, 0, select, VECTOR_DST, old);
            self.asm.str_q(select, CPU, offset);
        }
    }

    /// An inline AVX-512 instruction (`EvexForm`): whole 128, 256 or 512-bit vectors processed
    /// one 128-bit lane at a time through v0 and v1, loaded from and stored to `Cpu` per lane, so
    /// nothing lives in host vector registers across instructions. A memory operand that is
    /// misaligned (for the aligned moves), misses the TLB or crosses a page, and a failing gate,
    /// run the instruction in the interpreter, which also owns every fault and the partial-write
    /// rules. A masked load reads the whole vector inside the page the TLB vouches for, which
    /// cannot fault, so the elements the mask drops are read and discarded. Returns whether the
    /// instruction stores to memory.
    fn evex(
        &mut self,
        insn: &Insn,
        form: EvexForm,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.evex_entry(insn_index, rip);
        let lanes = form.bytes / XMM_SIZE;
        let lane_offset = |register: u8, lane: u32| vector_lane_offset(register, lane) as u32;
        let mut is_store = false;
        let writes_vector = matches!(form.kind, EvexKind::Alu { .. } | EvexKind::Move { .. });
        let memory_store = matches!(form.kind, EvexKind::Move { load: false, .. })
            && insn.md != MODRM_MOD_REGISTER;
        if writes_vector && !memory_store && insn.evex_mask != 0 {
            self.begin_masked_write(insn, form.element_log2);
        }
        match form.kind {
            EvexKind::MaskToGeneral { wide } => {
                self.asm.ldr(
                    VALUE,
                    CPU,
                    OPMASK_OFFSET + OPMASK_SIZE * (insn.rm & OPMASK_REGISTER_MASK) as u32,
                );
                if !wide {
                    self.asm.uxtw(VALUE, VALUE);
                }
                self.write_reg(insn.reg, false, if wide { 8 } else { 4 }, VALUE);
            }
            EvexKind::Move { load, aligned } => {
                if insn.md == MODRM_MOD_REGISTER {
                    let (from, to) = if load {
                        (insn.rm, insn.reg)
                    } else {
                        (insn.reg, insn.rm)
                    };
                    for lane in 0..lanes {
                        self.asm.ldr_q(VECTOR_DST, CPU, lane_offset(from, lane));
                        self.commit_vector_lane(insn, form.element_log2, to, lane);
                    }
                    self.zero_vector_lanes(to, lanes);
                } else if load {
                    self.evex_memory(insn, form.bytes, aligned, false, next_rip, slow);
                    for lane in 0..lanes {
                        self.asm.ldr_q(VECTOR_DST, SCRATCH_B, lane * XMM_SIZE);
                        self.commit_vector_lane(insn, form.element_log2, insn.reg, lane);
                    }
                    if self.ordered_memory {
                        // Loads are acquires (see `ordered_memory`); `ldar` has no 128-bit form.
                        self.asm.dmb_ishld();
                    }
                    self.zero_vector_lanes(insn.reg, lanes);
                } else {
                    self.evex_memory(insn, form.bytes, aligned, true, next_rip, slow);
                    for lane in 0..lanes {
                        let register = lane_offset(insn.reg, lane);
                        self.asm.ldr(VALUE, CPU, register);
                        self.asm.ldr(SCRATCH_A, CPU, register + XMM_HALF);
                        if self.ordered_memory {
                            self.asm.stlr(VALUE, SCRATCH_B, 8);
                            self.asm.add_imm(SCRATCH_B, SCRATCH_B, XMM_HALF, true);
                            self.asm.stlr(SCRATCH_A, SCRATCH_B, 8);
                            self.asm.add_imm(SCRATCH_B, SCRATCH_B, XMM_HALF, true);
                        } else {
                            self.asm
                                .stp_off(VALUE, SCRATCH_A, SCRATCH_B, (lane * XMM_SIZE) as i32);
                        }
                    }
                    is_store = true;
                }
            }
            EvexKind::Alu { .. }
            | EvexKind::CompareEqualBytes
            | EvexKind::TestBytes
            | EvexKind::TestNotBytes
            | EvexKind::CompareBytes { .. } => {
                let memory = insn.md != MODRM_MOD_REGISTER;
                if memory {
                    self.evex_memory(insn, form.bytes, false, false, next_rip, slow);
                }
                for lane in 0..lanes {
                    // v0 = vvvv, v1 = r/m, as the guest names them
                    self.asm
                        .ldr_q(VECTOR_DST, CPU, lane_offset(insn.vex_v, lane));
                    if memory {
                        self.asm.ldr_q(VECTOR_SRC, SCRATCH_B, lane * XMM_SIZE);
                    } else {
                        self.asm.ldr_q(VECTOR_SRC, CPU, lane_offset(insn.rm, lane));
                    }
                    match form.kind {
                        EvexKind::Alu { op, lane_log2 } => {
                            let (base, size, first, second) = match op {
                                VectorOp::And => (VEC_AND, 0, VECTOR_DST, VECTOR_SRC),
                                // bic computes its first operand & !second: rm & !vvvv
                                VectorOp::AndNot => (VEC_BIC, 0, VECTOR_SRC, VECTOR_DST),
                                VectorOp::Or => (VEC_ORR, 0, VECTOR_DST, VECTOR_SRC),
                                VectorOp::Xor => (VEC_EOR, 0, VECTOR_DST, VECTOR_SRC),
                                VectorOp::MinUnsigned8 => (VEC_UMIN, 0, VECTOR_DST, VECTOR_SRC),
                                VectorOp::MaxUnsigned8 => (VEC_UMAX, 0, VECTOR_DST, VECTOR_SRC),
                                VectorOp::Add => (VEC_ADD, lane_log2, VECTOR_DST, VECTOR_SRC),
                                VectorOp::Subtract => (VEC_SUB, lane_log2, VECTOR_DST, VECTOR_SRC),
                                VectorOp::CompareEqual | VectorOp::MoveMask => {
                                    unreachable!("not an EvexKind::Alu operation")
                                }
                            };
                            self.asm.vec_op(base, size, VECTOR_DST, first, second);
                            self.commit_vector_lane(insn, form.element_log2, insn.reg, lane);
                        }
                        _ => {
                            match form.kind {
                                EvexKind::CompareEqualBytes => self
                                    .asm
                                    .vec_op(VEC_CMEQ, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
                                EvexKind::CompareBytes { signed, predicate } => {
                                    self.compare_bytes(signed, predicate)
                                }
                                EvexKind::TestBytes => self
                                    .asm
                                    .vec_op(VEC_CMTST, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
                                _ => {
                                    self.asm
                                        .vec_op(VEC_CMTST, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC);
                                    self.asm.vec_op(VEC_NOT, 0, VECTOR_DST, VECTOR_DST, 0);
                                }
                            }
                            // The sign bit of every byte, gathered as for pmovmskb (see
                            // `vector_alu`), into bits 16 * lane.. of x0.
                            self.asm.vec_ushr(VECTOR_DST, VECTOR_DST, 8, 7);
                            self.asm.vec_usra(VECTOR_DST, VECTOR_DST, 16, 7);
                            self.asm.vec_usra(VECTOR_DST, VECTOR_DST, 32, 14);
                            self.asm.vec_usra(VECTOR_DST, VECTOR_DST, 64, 28);
                            self.asm.umov_byte(VALUE, VECTOR_DST, 0);
                            self.asm.umov_byte(SCRATCH_A, VECTOR_DST, XMM_HALF);
                            self.asm.add_lsl(VALUE, VALUE, SCRATCH_A, u8::BITS);
                            if lane == 0 {
                                self.asm.mov(0, VALUE);
                            } else {
                                self.asm.add_lsl(0, 0, VALUE, 16 * lane);
                            }
                        }
                    }
                }
                if memory && self.ordered_memory {
                    self.asm.dmb_ishld();
                }
                if matches!(form.kind, EvexKind::Alu { .. }) {
                    self.zero_vector_lanes(insn.reg, lanes);
                } else {
                    if insn.evex_mask != 0 {
                        self.asm.ldr(
                            SCRATCH_A,
                            CPU,
                            OPMASK_OFFSET + OPMASK_SIZE * insn.evex_mask as u32,
                        );
                        self.asm.and(0, 0, SCRATCH_A, true);
                    }
                    self.asm.str(
                        0,
                        CPU,
                        OPMASK_OFFSET + OPMASK_SIZE * (insn.reg & OPMASK_REGISTER_MASK) as u32,
                    );
                }
            }
        }
        self.asm.bind(resume);
        // The slow path's call clobbered host NZCV.
        self.live = None;
        is_store
    }

    /// A 128-bit whole-register move (`VectorMove`). The fast path needs the memory operand
    /// inside one page and, for the aligned forms, 16-byte aligned; anything else, and the SSE
    /// gate (#UD/#NM) failing, runs the instruction in the interpreter so faults and the
    /// partial-store rules (`Cpu::write128` probes both pages first) stay in one place. Returns
    /// whether the instruction stores to memory (the caller then polls for self-modifying code).
    fn vector_move(
        &mut self,
        insn: &Insn,
        form: VectorMove,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM_OFFSET + XMM_SIZE * reg as u32;
        let memory_is_store = !form.load;
        if insn.md == MODRM_MOD_REGISTER {
            // The register form: movaps xmm1, xmm2 and friends copy the whole register.
            let (from, to) = if form.load {
                (insn.rm, insn.reg)
            } else {
                (insn.reg, insn.rm)
            };
            self.asm.ldr(VALUE, CPU, xmm(from));
            self.asm.ldr(SCRATCH_A, CPU, xmm(from) + XMM_HALF);
            self.asm.str(VALUE, CPU, xmm(to));
            self.asm.str(SCRATCH_A, CPU, xmm(to) + XMM_HALF);
            self.asm.bind(resume);
            self.live = None;
            return false;
        }
        self.effective_address(insn, next_rip, ADDRESS);
        if form.aligned {
            self.and_constant(SCRATCH_A, ADDRESS, VECTOR_ALIGNMENT - 1);
            self.asm.cbnz(SCRATCH_A, slow, true);
        }
        let tag = if memory_is_store {
            TLB_TAG_WRITE_OFFSET
        } else {
            TLB_TAG_READ_OFFSET
        };
        self.tlb_probe(XMM_SIZE as u8, tag, slow);
        self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
        self.asm.add(SCRATCH_B, ADDRESS, SCRATCH_B, true);
        let register = xmm(insn.reg);
        if memory_is_store {
            self.asm.ldr(VALUE, CPU, register);
            self.asm.ldr(SCRATCH_A, CPU, register + XMM_HALF);
            if self.ordered_memory {
                self.asm.stlr(VALUE, SCRATCH_B, 8);
                self.asm.add_imm(SCRATCH_B, SCRATCH_B, XMM_HALF, true);
                self.asm.stlr(SCRATCH_A, SCRATCH_B, 8);
            } else {
                self.asm.stp_off(VALUE, SCRATCH_A, SCRATCH_B, 0);
            }
        } else {
            if self.ordered_memory {
                self.asm.ldar(VALUE, SCRATCH_B, 8);
                self.asm.add_imm(SCRATCH_B, SCRATCH_B, XMM_HALF, true);
                self.asm.ldar(SCRATCH_A, SCRATCH_B, 8);
            } else {
                self.asm.ldp_off(VALUE, SCRATCH_A, SCRATCH_B, 0);
            }
            self.asm.str(VALUE, CPU, register);
            self.asm.str(SCRATCH_A, CPU, register + XMM_HALF);
        }
        self.asm.bind(resume);
        // The slow path's call clobbered host NZCV.
        self.live = None;
        memory_is_store
    }

    /// A lane-wise integer vector instruction (`VectorAlu`) on xmm registers, the source being a
    /// register or a 16-byte aligned memory operand (misaligned, page-crossing and gate failures
    /// run in the interpreter, as for `vector_move`). The registers are loaded into v0 and v1
    /// from `Cpu::xmm`, so nothing is live in host vector registers across instructions.
    fn vector_alu(
        &mut self,
        insn: &Insn,
        form: VectorAlu,
        next_rip: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM_OFFSET + XMM_SIZE * reg as u32;
        if form.op == VectorOp::MoveMask {
            // The sign bit of every byte, gathered by the `sse2neon` sequence: shift each bit 7
            // down to bit 0 of its byte, then fold neighbouring bytes, words and dwords together
            // so each byte of the low and of the high half ends up holding its half's eight bits.
            self.asm.ldr_q(VECTOR_SRC, CPU, xmm(insn.rm));
            self.asm.vec_ushr(VECTOR_SRC, VECTOR_SRC, 8, 7);
            self.asm.vec_usra(VECTOR_SRC, VECTOR_SRC, 16, 7);
            self.asm.vec_usra(VECTOR_SRC, VECTOR_SRC, 32, 14);
            self.asm.vec_usra(VECTOR_SRC, VECTOR_SRC, 64, 28);
            self.asm.umov_byte(VALUE, VECTOR_SRC, 0);
            self.asm.umov_byte(SCRATCH_A, VECTOR_SRC, XMM_HALF);
            self.asm.add_lsl(VALUE, VALUE, SCRATCH_A, u8::BITS);
            self.write_reg(insn.reg, false, 4, VALUE);
            self.asm.bind(resume);
            self.live = None;
            return false;
        }
        self.asm.ldr_q(VECTOR_DST, CPU, xmm(insn.reg));
        if insn.md == MODRM_MOD_REGISTER {
            self.asm.ldr_q(VECTOR_SRC, CPU, xmm(insn.rm));
        } else {
            self.effective_address(insn, next_rip, ADDRESS);
            // Every legacy-SSE form of these instructions needs a 16-byte aligned memory operand.
            self.and_constant(SCRATCH_A, ADDRESS, VECTOR_ALIGNMENT - 1);
            self.asm.cbnz(SCRATCH_A, slow, true);
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_READ_OFFSET, slow);
            self.asm.ldr(SCRATCH_B, SCRATCH_A, TLB_ADDEND_OFFSET);
            self.asm.add(SCRATCH_B, ADDRESS, SCRATCH_B, true);
            self.asm.ldr_q(VECTOR_SRC, SCRATCH_B, 0);
            if self.ordered_memory {
                // Loads are acquires (see `ordered_memory`); `ldar` has no 128-bit form.
                self.asm.dmb_ishld();
            }
        }
        let lane = form.lane_log2;
        let (base, size, destination, first, second) = match form.op {
            VectorOp::And => (VEC_AND, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            // bic computes its first operand & !second: src & !dst
            VectorOp::AndNot => (VEC_BIC, 0, VECTOR_DST, VECTOR_SRC, VECTOR_DST),
            VectorOp::Or => (VEC_ORR, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::Xor => (VEC_EOR, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::CompareEqual => (VEC_CMEQ, lane, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::MinUnsigned8 => (VEC_UMIN, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::MaxUnsigned8 => (VEC_UMAX, 0, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::Add => (VEC_ADD, lane, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::Subtract => (VEC_SUB, lane, VECTOR_DST, VECTOR_DST, VECTOR_SRC),
            VectorOp::MoveMask => unreachable!("handled above"),
        };
        self.asm.vec_op(base, size, destination, first, second);
        self.asm.str_q(VECTOR_DST, CPU, xmm(insn.reg));
        self.asm.bind(resume);
        self.live = None;
        false
    }

    /// Compiles the whole block. `rips[k]` is the offset in the block's page of instruction `k`
    /// and `end_rip` the offset after the last one (past 4096 for an instruction that straddles
    /// the page end). A block ends after a terminator (see
    /// `ends_block`), or by falling off the end (instruction limit or page boundary), in which case
    /// it exits to `end_rip`.
    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let count = self.insns.len();
        self.entry_rip = rips[0];
        if self.addrs.chain != 0 {
            self.link_entry();
        }
        self.body_offset = (self.asm.here() - self.asm.base()) as u32;
        for index in 0..count {
            let rip = rips[index];
            let next_rip = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next_rip) {
                if is_inline_terminator(&self.insns[index]) {
                    self.emit_slows();
                    return;
                }
                continue;
            }
            self.helper(index, rip, terminal);
            if terminal {
                self.emit_slows();
                return;
            }
        }
        self.chain_to(end_rip, count as u32);
        self.emit_slows();
    }
}

#[cfg(test)]
mod tests {
    //! The pure helpers of the aarch64 block compiler: the x86-condition to NZCV mapping (checked
    //! against a model of both flag definitions), the constant-mask fallback, and small decoders.

    use super::*;
    use crate::cpu::decode::NO_SEG;
    use crate::cpu::flags::{self, CF, OF, PF, SF, ZF};
    use crate::sys::ExecMem;

    fn size_in_bits(size: u8) -> u32 {
        size as u32 * 8
    }

    /// aarch64 NZCV after the instruction `kind` corresponds to (`adds`, `subs`, `ands`) at `size`.
    struct Nzcv {
        n: bool,
        z: bool,
        c: bool,
        v: bool,
    }

    fn nzcv_after(kind: FlagKind, size: u8, a: u64, b: u64) -> (u64, Nzcv) {
        let bits = size_in_bits(size);
        let mask = flags::mask(size);
        let sign = 1u64 << (bits - 1);
        let (a, b) = (a & mask, b & mask);
        let (result, c, v) = match kind {
            FlagKind::Add => {
                let sum = a as u128 + b as u128;
                let result = (sum as u64) & mask;
                (
                    result,
                    sum >> bits != 0,
                    (!(a ^ b) & (a ^ result)) & sign != 0,
                )
            }
            FlagKind::Sub => {
                let result = a.wrapping_sub(b) & mask;
                // aarch64 carry is "no borrow"
                (result, a >= b, ((a ^ b) & (a ^ result)) & sign != 0)
            }
            FlagKind::Logic => ((a & b) & mask, false, false),
            FlagKind::Adc | FlagKind::Sbb => unreachable!("see nzcv_after_with_carry_in"),
            // The host op is `adds`/`subs` with the immediate 1; the caller passes b = 1.
            FlagKind::Inc => {
                let result = a.wrapping_add(1) & mask;
                (result, false, (!(a ^ 1) & (a ^ result)) & sign != 0)
            }
            FlagKind::Dec => {
                let result = a.wrapping_sub(1) & mask;
                (result, false, ((a ^ 1) & (a ^ result)) & sign != 0)
            }
        };
        (
            result,
            Nzcv {
                n: result & sign != 0,
                z: result == 0,
                c,
                v,
            },
        )
    }

    /// `nzcv_after` for `adcs` and `sbcs` with the carry-in `carry` (0 or 1).
    fn nzcv_after_with_carry_in(
        kind: FlagKind,
        size: u8,
        a: u64,
        b: u64,
        carry: u64,
    ) -> (u64, Nzcv) {
        let bits = size_in_bits(size);
        let mask = flags::mask(size);
        let sign = 1u64 << (bits - 1);
        let (a, b) = (a & mask, b & mask);
        let (result, c, v) = match kind {
            FlagKind::Adc => {
                let sum = a as u128 + b as u128 + carry as u128;
                let result = (sum as u64) & mask;
                (
                    result,
                    sum >> bits != 0,
                    (!(a ^ b) & (a ^ result)) & sign != 0,
                )
            }
            FlagKind::Sbb => {
                let result = a.wrapping_sub(b).wrapping_sub(carry) & mask;
                // aarch64 carry is "no borrow"
                (
                    result,
                    a as u128 >= b as u128 + carry as u128,
                    ((a ^ b) & (a ^ result)) & sign != 0,
                )
            }
            _ => unreachable!(),
        };
        (
            result,
            Nzcv {
                n: result & sign != 0,
                z: result == 0,
                c,
                v,
            },
        )
    }

    fn aarch64_condition_holds(condition: u32, f: &Nzcv) -> bool {
        match condition {
            EQ => f.z,
            NE => !f.z,
            CS => f.c,
            CC => !f.c,
            MI => f.n,
            PL => !f.n,
            VS => f.v,
            VC => !f.v,
            HI => f.c && !f.z,
            LS => !(f.c && !f.z),
            GE => f.n == f.v,
            LT => f.n != f.v,
            GT => !f.z && f.n == f.v,
            LE => f.z || f.n != f.v,
            other => panic!("condition {other}"),
        }
    }

    /// The x86 condition codes (low nibble of Jcc) evaluated from EFLAGS.
    fn x86_condition_holds(condition: u8, eflags: u64) -> bool {
        let (of, cf, zf, sf, pf) = (
            eflags & OF != 0,
            eflags & CF != 0,
            eflags & ZF != 0,
            eflags & SF != 0,
            eflags & PF != 0,
        );
        match condition {
            0 => of,
            1 => !of,
            2 => cf,
            3 => !cf,
            4 => zf,
            5 => !zf,
            6 => cf || zf,
            7 => !(cf || zf),
            8 => sf,
            9 => !sf,
            10 => pf,
            11 => !pf,
            12 => sf != of,
            13 => sf == of,
            14 => zf || sf != of,
            _ => !zf && sf == of,
        }
    }

    const OPERAND_VALUES: [u64; 12] = [
        0,
        1,
        2,
        0x7f,
        0x80,
        0xff,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        u64::MAX,
    ];

    fn lazy_op(kind: FlagKind) -> LazyFlagsOp {
        match kind {
            FlagKind::Add => LazyFlagsOp::Add,
            FlagKind::Sub => LazyFlagsOp::Sub,
            FlagKind::Adc => LazyFlagsOp::Adc,
            FlagKind::Sbb => LazyFlagsOp::Sbb,
            FlagKind::Logic => LazyFlagsOp::Logic,
            FlagKind::Inc => LazyFlagsOp::Inc,
            FlagKind::Dec => LazyFlagsOp::Dec,
        }
    }

    /// The conditions offered after `adcs`/`sbcs` are the `Add`/`Sub` ones; they must hold for
    /// either carry-in, as the x86 flags of the same `adc`/`sbb` say.
    #[test]
    fn adc_and_sbb_conditions_match_the_x86_flags_for_either_carry_in() {
        for kind in [FlagKind::Adc, FlagKind::Sbb] {
            for size in [4u8, 8] {
                for a in OPERAND_VALUES {
                    for b in OPERAND_VALUES {
                        for carry in [0, 1] {
                            let (result, nzcv) = nzcv_after_with_carry_in(kind, size, a, b, carry);
                            let eflags = flags::compute(lazy_op(kind), size, result, b, carry, 0);
                            for condition in 0..16u8 {
                                let expected = x86_condition_holds(condition, eflags);
                                if let Some(CondMap::Cond(host)) = cond_map(kind, condition) {
                                    assert_eq!(
                                        aarch64_condition_holds(host, &nzcv),
                                        expected,
                                        "{kind:?} size {size} {a:#x}, {b:#x}, carry {carry}: condition {condition}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_mapped_condition_matches_the_x86_flags_the_same_operation_produces() {
        for kind in [FlagKind::Add, FlagKind::Sub, FlagKind::Logic] {
            for size in [4u8, 8] {
                for a in OPERAND_VALUES {
                    for b in OPERAND_VALUES {
                        let (result, nzcv) = nzcv_after(kind, size, a, b);
                        let eflags = flags::compute(lazy_op(kind), size, result, b, 0, 0);
                        for condition in 0..16u8 {
                            let expected = x86_condition_holds(condition, eflags);
                            match cond_map(kind, condition) {
                                Some(CondMap::Cond(host)) => assert_eq!(
                                    aarch64_condition_holds(host, &nzcv),
                                    expected,
                                    "{kind:?} size {size} {a:#x}, {b:#x}: condition {condition}"
                                ),
                                Some(CondMap::Always) => {
                                    assert!(expected, "{kind:?} cond {condition}")
                                }
                                Some(CondMap::Never) => {
                                    assert!(!expected, "{kind:?} cond {condition}")
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        }
    }

    /// `inc`/`dec` keep CF whatever it was, so every condition the map offers must hold for both
    /// incoming carries; the ones that read CF must not be offered at all.
    #[test]
    fn inc_and_dec_conditions_are_right_for_either_incoming_carry() {
        for kind in [FlagKind::Inc, FlagKind::Dec] {
            for size in [4u8, 8] {
                for a in OPERAND_VALUES {
                    let (result, nzcv) = nzcv_after(kind, size, a, 1);
                    for carry in [0, 1] {
                        let eflags = flags::compute(lazy_op(kind), size, result, carry, 0, 0);
                        for condition in 0..16u8 {
                            let expected = x86_condition_holds(condition, eflags);
                            match cond_map(kind, condition) {
                                Some(CondMap::Cond(host)) => assert_eq!(
                                    aarch64_condition_holds(host, &nzcv),
                                    expected,
                                    "{kind:?} size {size} {a:#x} carry {carry}: condition {condition}"
                                ),
                                Some(_) => panic!("{kind:?} condition {condition} is not constant"),
                                None => {}
                            }
                        }
                    }
                }
            }
            for condition in [2, 3, 6, 7, 10, 11] {
                assert!(cond_map(kind, condition).is_none(), "{kind:?} {condition}");
            }
        }
    }

    #[test]
    fn parity_and_add_below_or_equal_have_no_host_condition() {
        for kind in [FlagKind::Add, FlagKind::Sub, FlagKind::Logic] {
            assert!(cond_map(kind, 10).is_none(), "{kind:?} p");
            assert!(cond_map(kind, 11).is_none(), "{kind:?} np");
        }
        assert!(
            cond_map(FlagKind::Add, 6).is_none(),
            "add: be needs CF | ZF"
        );
        assert!(
            cond_map(FlagKind::Add, 7).is_none(),
            "add: a needs !CF & !ZF"
        );
        assert!(cond_map(FlagKind::Sub, 6).is_some());
        assert!(cond_map(FlagKind::Logic, 6).is_some());
    }

    #[test]
    fn logic_never_overflows_or_carries() {
        assert!(
            matches!(cond_map(FlagKind::Logic, 0), Some(CondMap::Never)),
            "o"
        );
        assert!(
            matches!(cond_map(FlagKind::Logic, 1), Some(CondMap::Always)),
            "no"
        );
        assert!(
            matches!(cond_map(FlagKind::Logic, 2), Some(CondMap::Never)),
            "b"
        );
        assert!(
            matches!(cond_map(FlagKind::Logic, 3), Some(CondMap::Always)),
            "ae"
        );
    }

    #[test]
    fn cond_map_uses_only_the_low_four_bits_of_the_condition() {
        for condition in 0..16u8 {
            let high = condition | 0x70;
            assert_eq!(
                format!(
                    "{:?}",
                    cond_map(FlagKind::Sub, condition).map(|m| matches!(m, CondMap::Cond(_)))
                ),
                format!(
                    "{:?}",
                    cond_map(FlagKind::Sub, high).map(|m| matches!(m, CondMap::Cond(_)))
                ),
            );
        }
    }

    #[test]
    fn the_flag_kinds_map_to_the_lazy_ops_the_interpreter_understands() {
        // `store_lazy_op` writes these discriminants into `Cpu::lazy_op`; they must stay the
        // `LazyFlagsOp` values the interpreter decodes (Add = 1, Sub = 3, Logic = 5).
        assert_eq!(LazyFlagsOp::Add as u8, 1);
        assert_eq!(LazyFlagsOp::Sub as u8, 3);
        assert_eq!(LazyFlagsOp::Logic as u8, 5);
    }

    fn insn_with(op: u16, rex: u8, seg: u8) -> Insn {
        Insn {
            op,
            rex,
            seg,
            ..Default::default()
        }
    }

    #[test]
    fn opcode_register_combines_the_opcode_bits_with_rex_b() {
        for low in 0..8u16 {
            assert_eq!(opcode_register(&insn_with(0x50 + low, 0, 0xff)), low as u8);
            assert_eq!(
                opcode_register(&insn_with(0x58 + low, REX_B, 0xff)),
                8 + low as u8
            );
        }
        assert_eq!(
            opcode_register(&insn_with(0xbf, 0x4f & 0xf, 0xff)),
            15,
            "REX.W/R/X do not matter, only B"
        );
    }

    #[test]
    fn only_fs_and_gs_overrides_add_a_segment_base() {
        for seg in [0u8, 1, 2, 3, NO_SEG] {
            assert!(!has_segment_base(&insn_with(0, 0, seg)), "seg {seg}");
        }
        assert!(has_segment_base(&insn_with(0, 0, SEG_FS as u8)));
        assert!(has_segment_base(&insn_with(0, 0, SEG_GS as u8)));
    }

    #[test]
    fn immediates_are_sign_extended_for_64_bit_operations_and_truncated_for_32_bit() {
        assert_eq!(Gen::immediate_for(8, -1), u64::MAX);
        assert_eq!(Gen::immediate_for(4, -1), 0xffff_ffff);
        assert_eq!(Gen::immediate_for(8, 0x7fff_ffff), 0x7fff_ffff);
        assert_eq!(Gen::immediate_for(4, i32::MIN as i64), 0x8000_0000);
    }

    #[test]
    fn memory_operands_with_address_size_override_or_lock_stay_out_of_line() {
        let plain = Insn::default();
        assert!(Gen::memory_inlinable(&plain));
        assert!(!Gen::memory_inlinable(&Insn {
            addr32: true,
            ..Default::default()
        }));
        assert!(!Gen::memory_inlinable(&Insn {
            lock: true,
            ..Default::default()
        }));
        for seg in [0u8, 3, 4, 5, NO_SEG] {
            assert!(
                Gen::memory_inlinable(&Insn {
                    seg,
                    ..Default::default()
                }),
                "seg {seg}"
            );
        }
    }

    const DUMMY_ADDRESSES: Addrs = Addrs {
        exec: 0,
        cond: 0,
        rotate_flags: 0,
        exit_ok: 0,
        exit_fail: 0,
        chain: 0,
        load: [0; 4],
        load_signed: [0; 3],
        load_stack: 0,
        store: [0; 4],
        store_keep: [0; 4],
        store_stack: 0,
    };

    fn and_constant_result(constant: u64, input: u64) -> (u64, usize) {
        let _guard = crate::jit::serialize_jit_tests();
        let memory = ExecMem::new(16384).unwrap();
        let insns: [Insn; 0] = [];
        let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
        generator.and_constant(0, 1, constant);
        let words_for_mask = generator.asm.code.len();
        generator.asm.ret();
        generator.asm.finish();
        memory.write(0, &generator.asm.code);
        let function: extern "C" fn(u64, u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        (function(0, input), words_for_mask)
    }

    #[test]
    fn and_constant_uses_one_instruction_for_an_encodable_mask() {
        let (result, words) = and_constant_result(0x3ff, 0xffff_ffff_ffff_ffff);
        assert_eq!(result, 0x3ff);
        assert_eq!(words, 1);
        let (result, words) = and_constant_result(!0xfff, 0x1234_5678_9abc_def0);
        assert_eq!(result, 0x1234_5678_9abc_d000);
        assert_eq!(words, 1);
    }

    #[test]
    fn and_constant_falls_back_to_a_register_for_a_mask_that_is_not_a_bitmask_immediate() {
        // 0x1234 and 0xdead_beef are not rotated runs of ones, so `and_imm` would emit nothing.
        for mask in [0x1234u64, 0xdead_beef, 0x0123_4567_89ab_cdef, 0] {
            let (result, words) = and_constant_result(mask, 0xffff_ffff_0f0f_0f0f);
            assert_eq!(result, mask & 0xffff_ffff_0f0f_0f0f, "mask {mask:#x}");
            assert!(words > 1, "mask {mask:#x} needs the constant materialised");
        }
    }

    #[test]
    fn and_constant_does_not_disturb_the_value_register_when_destination_differs() {
        let _guard = crate::jit::serialize_jit_tests();
        let memory = ExecMem::new(16384).unwrap();
        let insns: [Insn; 0] = [];
        let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
        generator.and_constant(2, 1, 0x1234);
        generator.asm.mov(0, 1);
        generator.asm.add(0, 0, 2, true);
        generator.asm.ret();
        generator.asm.finish();
        memory.write(0, &generator.asm.code);
        let function: extern "C" fn(u64, u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        assert_eq!(function(0, 0xffff), 0xffff + 0x1234);
    }

    /// Runs the code `rip_to` emits for `offset` with `PAGE_BASE` set to `page_base`. x21 is
    /// callee-saved, so the function saves it (and the link register) around the test.
    fn rip_to_result(page_base: u64, offset: u64) -> (u64, usize) {
        let _guard = crate::jit::serialize_jit_tests();
        let memory = ExecMem::new(16384).unwrap();
        let insns: [Insn; 0] = [];
        let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
        generator.asm.stp_pre(PAGE_BASE, 30, SP, -16);
        generator.asm.mov(PAGE_BASE, 0);
        let before = generator.asm.code.len();
        generator.rip_to(0, offset);
        let words = generator.asm.code.len() - before;
        generator.asm.ldp_post(PAGE_BASE, 30, SP, 16);
        generator.asm.ret();
        generator.asm.finish();
        memory.write(0, &generator.asm.code);
        let function: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        (function(page_base), words)
    }

    const PAGE_BASES: [u64; 4] = [
        0,
        0x0000_7f12_3456_7000,
        0xffff_ffff_8100_0000,
        0xffff_ffff_ffff_f000,
    ];

    #[test]
    fn rip_to_adds_every_kind_of_offset_to_the_page_base_with_wraparound() {
        let offsets: [i64; 20] = [
            0,
            1,
            4095,
            4096,
            4097,
            0x12_3456,
            0xff_ffff,
            0x100_0000,
            0x4000_0000,
            0x7fff_f000,
            i64::MAX,
            -1,
            -4095,
            -4096,
            -4097,
            -0x12_3456,
            -0x4000_0000,
            -0x7fff_f000,
            i64::MIN,
            i64::MIN + 1,
        ];
        for page_base in PAGE_BASES {
            for offset in offsets {
                let (result, _) = rip_to_result(page_base, offset as u64);
                assert_eq!(
                    result,
                    page_base.wrapping_add(offset as u64),
                    "base {page_base:#x} offset {offset:#x}"
                );
            }
        }
    }

    #[test]
    fn rip_to_is_one_instruction_for_an_offset_within_a_page_either_way() {
        for offset in [0i64, 1, 100, 4095, -1, -100, -4095] {
            assert_eq!(rip_to_result(0x1000, offset as u64).1, 1, "offset {offset}");
        }
    }

    #[test]
    fn rip_to_needs_more_than_one_instruction_beyond_a_page() {
        for offset in [4096i64, -4096, 0x12_3456, i64::MAX, i64::MIN] {
            assert!(rip_to_result(0, offset as u64).1 > 1, "offset {offset}");
        }
    }

    #[test]
    fn rip_to_leaves_the_host_flags_alone() {
        let _guard = crate::jit::serialize_jit_tests();
        for offset in [0u64, 8, 4096, 0x12_3456, u64::MAX] {
            let memory = ExecMem::new(16384).unwrap();
            let insns: [Insn; 0] = [];
            let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
            generator.asm.stp_pre(PAGE_BASE, 30, SP, -16);
            generator.asm.mov(PAGE_BASE, 1);
            // x0 - x0 sets Z and C; the materialisation must not clear them.
            generator.asm.cmp(0, 0, true);
            generator.rip_to(2, offset);
            generator.asm.cset(0, EQ);
            generator.asm.cset(3, CS);
            generator.asm.add(0, 0, 3, true);
            generator.asm.ldp_post(PAGE_BASE, 30, SP, 16);
            generator.asm.ret();
            generator.asm.finish();
            memory.write(0, &generator.asm.code);
            let function: extern "C" fn(u64) -> u64 =
                unsafe { std::mem::transmute(memory.base() as usize) };
            assert_eq!(function(5), 2, "Z and C survived rip_to({offset:#x})");
        }
    }

    #[test]
    fn rip_to_writes_only_its_destination() {
        let _guard = crate::jit::serialize_jit_tests();
        let memory = ExecMem::new(16384).unwrap();
        let insns: [Insn; 0] = [];
        let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
        generator.asm.stp_pre(PAGE_BASE, 30, SP, -16);
        generator.asm.mov(PAGE_BASE, 0);
        // Sentinels in the scratch registers the address arithmetic uses.
        for register in [SCRATCH_A, SCRATCH_B, SCRATCH_C, VALUE, 1] {
            generator
                .asm
                .mov_imm(register, 0x1111_0000 + register as u64);
        }
        generator.rip_to(ADDRESS, 0x12_3456);
        generator.asm.mov(0, SCRATCH_A);
        generator.asm.add(0, 0, SCRATCH_B, true);
        generator.asm.add(0, 0, SCRATCH_C, true);
        generator.asm.add(0, 0, VALUE, true);
        generator.asm.add(0, 0, 1, true);
        generator.asm.ldp_post(PAGE_BASE, 30, SP, 16);
        generator.asm.ret();
        generator.asm.finish();
        memory.write(0, &generator.asm.code);
        let function: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        let expected: u64 = [SCRATCH_A, SCRATCH_B, SCRATCH_C, VALUE, 1]
            .iter()
            .map(|&register| 0x1111_0000 + register as u64)
            .sum();
        assert_eq!(function(0x7000), expected);
    }
}
