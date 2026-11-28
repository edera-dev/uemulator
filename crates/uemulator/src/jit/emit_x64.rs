//! x86-64 block generator. Mirrors `emit.rs` (read its module docs for the exit protocol and the
//! lazy flags); the guest flags map straight onto host flags, so a conditional jump, setcc or
//! cmovcc right after an inlined add/sub/logic/inc/dec/neg tests them directly (`Gen::live`).
//! Anything that clobbers host flags (TLB probes, helper calls, the store of a read-modify-write
//! instruction) ends that window; then the condition comes from the lazy flags via `helper_cond`.
//!
//! Instructions without an inline lowering call `helper_exec` (the interpreter). Every way out of a
//! block jumps to a shared stub in `arch_x64::prologue`: `exit_ok` with rax = `retired <<
//! EXIT_COUNT_SHIFT`, `exit_fail` with rax = the helper status and rcx = index of the instruction
//! that stopped the block, or, for a control transfer to a known guest address, `chain` (see
//! `Gen::chain_exit`), which continues straight into the successor block when it exists.
//!
//! Register roles. `CPU` and `TLB_BASE` live for the whole block. `ACC` and `OPERAND` hold the two
//! operands of an ALU-style instruction (rcx doubles as the shift count), `ADDR` the guest address
//! of the memory operand, `VALUE` what a mov-style instruction moves. Any helper call clobbers all
//! of those and the scratch registers, so the slow paths park what must survive in the
//! callee-saved `SAVED_*` registers (the prologue saves r12-r15 on both ABIs).
//!
//! Guest registers are only ever read and written through `Cpu::gpr` in memory. Byte and word
//! writes (al, ax, ah, ...) are plain narrow stores into the little-endian register slot, which
//! merges them without touching host flags; that is what lets flags stay live across them.

use std::mem::{offset_of, size_of};

use super::helpers::*;
use super::native::{LinkSite, ResolveTarget};
use super::x64::*;
use crate::cpu::cet::Cet;
use crate::cpu::decode::{Insn, NO_REG, VEX_OP_FLAG};
use crate::cpu::flags::{
    AC, ARITH, CF, DF, IF, LazyFlagsOp, OF, RF, SZP_CF_OF_SRC_CF_BIT, SZP_CF_OF_SRC_OF_BIT, VM, ZF,
};
use crate::cpu::xsave::{XCR0_AVX, XCR0_AVX512, XCR0_SSE};
use crate::cpu::{
    CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXSAVE, Cpu, SEG_FS, SEG_GS, Seg, TLB_ENTRIES_PER_MODE,
    TLB_TAG_SHADOW_STACK, TlbEntry,
};

// Offsets into `Cpu` baked into generated code. They come from `offset_of!`, so they always match
// the compiled layout; what has to stay true is the shape of the `#[repr(C)]` `Cpu` and `TlbEntry`:
// `gpr` is indexed by guest register number, `rip`/`icount`/`prepaid_icount`/`lazy_dst`/`lazy_src`
// are u64, `block_exit_requested` is a bool (tested as a byte), and `lazy_size` immediately follows
// the one-byte `lazy_op` (written together by `store_lazy`).
const GPR: i32 = offset_of!(Cpu, gpr) as i32;
const GPR_SIZE: i32 = size_of::<u64>() as i32;
const RIP: i32 = offset_of!(Cpu, rip) as i32;
const ICOUNT: i32 = offset_of!(Cpu, icount) as i32;
const BLOCK_EXIT_REQUESTED: i32 = offset_of!(Cpu, block_exit_requested) as i32;
const NEXT_EVENT: i32 = offset_of!(Cpu, next_event) as i32;
/// A link slot: `jmp rel32` to the very next instruction, until `Jit` points it at a successor
/// block. The jump is always there and only its 32-bit displacement is rewritten, in one aligned
/// atomic store (`ExecMem::write_word`), because another vCPU may be executing the slot while it
/// is patched and a five-byte rewrite could be seen half done.
pub(crate) const LINK_JMP: [u8; 5] = [0xe9, 0x00, 0x00, 0x00, 0x00];
const LINK_JMP_OPCODE_BYTES: u64 = 1;
const LINK_WORD_ALIGN: u64 = 4;
const PREPAID_ICOUNT: i32 = offset_of!(Cpu, prepaid_icount) as i32;
const SLOW_ACCESS_INSN: i32 = offset_of!(Cpu, slow_access_insn) as i32;
const LAZY_OP: i32 = offset_of!(Cpu, lazy_op) as i32;
const LAZY_SIZE_SHIFT: u32 = 8;
const LAZY_DST: i32 = offset_of!(Cpu, lazy_dst) as i32;
const LAZY_SRC: i32 = offset_of!(Cpu, lazy_src) as i32;
const LAZY_SRC2: i32 = offset_of!(Cpu, lazy_src2) as i32;
const CR0: i32 = offset_of!(Cpu, cr0) as i32;
const CR4: i32 = offset_of!(Cpu, cr4) as i32;
const XMM: i32 = offset_of!(Cpu, xmm) as i32;
const OPMASK: i32 = offset_of!(Cpu, opmask) as i32;
const OPMASK_SIZE: i32 = size_of::<u64>() as i32;
const XCR0: i32 = offset_of!(Cpu, xcr0) as i32;
/// What `Cpu::evex_gate` requires of XCR0: the SSE, AVX and AVX-512 state components.
const XCR0_EVEX_STATE: u64 = XCR0_SSE | XCR0_AVX | XCR0_AVX512;
const XCR0_VEX_STATE: u64 = XCR0_SSE | XCR0_AVX;
/// Multiplying an element zero-extended to 64 bits by `BROADCAST_MULTIPLIER[log2]` copies it
/// into every element of a quadword (byte, word, dword; a quadword needs nothing).
const BROADCAST_MULTIPLIER: [u64; 3] = [
    0x0101_0101_0101_0101,
    0x0001_0001_0001_0001,
    0x0000_0001_0000_0001,
];
/// An opmask register number is the low three bits of its ModRM field.
const OPMASK_REGISTER_MASK: u8 = 7;
/// 128-bit lanes of a 512-bit vector register.
const VECTOR_LANES: u32 = 4;
/// Condition codes 0xa (p) and 0xb (np) are the parity pair: `cc >> 1`.
const CONDITION_PARITY: u8 = 5;
/// The vpcmp immediate's predicates (SDM table of VPCMP).
const VPCMP_EQUAL: u8 = 0;
const VPCMP_LESS: u8 = 1;
const VPCMP_LESS_EQUAL: u8 = 2;
const VPCMP_FALSE: u8 = 3;
const VPCMP_NOT_EQUAL: u8 = 4;
const VPCMP_NOT_LESS: u8 = 5;
const VPCMP_NOT_LESS_EQUAL: u8 = 6;
/// Host xmm registers for the write mask of a masked write and the old destination it merges
/// with; xmm2 is also `HOST_XMM_ZERO`, which a masked write never needs.
const HOST_XMM_SELECT: u8 = 2;
const HOST_XMM_OLD: u8 = 3;
const XMM_SIZE: i32 = size_of::<u128>() as i32;
const XMM_HALF: i32 = XMM_SIZE / 2;
/// Legacy-SSE 128-bit memory operands of movaps/movdqa and friends must be aligned to this.
const VECTOR_ALIGNMENT: i32 = XMM_SIZE;
/// Host xmm registers of an inline vector instruction: the guest destination (also the result)
/// and the source operand. Both are caller-saved on every host ABI.
const HOST_XMM_DST: u8 = 0;
const HOST_XMM_SRC: u8 = 1;
/// Holds zero while a vector is compared against it (`vptestmb`) or stored over upper lanes.
const HOST_XMM_ZERO: u8 = 2;
// `Seg` is not `repr(C)`; `offset_of!` and `size_of` still report the real layout.
const SEG_BASE: i32 = (offset_of!(Cpu, seg) + offset_of!(Seg, base)) as i32;
const SEG_SIZE: i32 = size_of::<Seg>() as i32;
const GS_BASE: i32 = SEG_BASE + SEG_SIZE * SEG_GS as i32;
const KGSBASE: i32 = offset_of!(Cpu, kgsbase) as i32;
const CPL: i32 = offset_of!(Cpu, cpl) as i32;
const SHADOW_STACK_ON: i32 = offset_of!(Cpu, shadow_stack_on) as i32;
const SSP: i32 = (offset_of!(Cpu, cet) + offset_of!(Cet, ssp)) as i32;
/// Bytes a near call pushes on the shadow stack in 64-bit mode, the only mode compiled.
const SHADOW_STACK_ENTRY: i32 = 8;
const EFLAGS: i32 = offset_of!(Cpu, eflags) as i32;
const SEG_SELECTOR: i32 = (offset_of!(Cpu, seg) + offset_of!(Seg, sel)) as i32;
const INTERRUPT_SHADOW: i32 = offset_of!(Cpu, interrupt_shadow) as i32;
const MULTI: i32 = offset_of!(Cpu, multi) as i32;
const PKRU: i32 = offset_of!(Cpu, pkru) as i32;
const TSC_AUX: i32 = offset_of!(Cpu, tsc_aux) as i32;
/// `Svm::guest`, an `Option<Box<_>>`: zero exactly when no SVM guest runs, whose host could
/// intercept pause, rdtsc and pushf.
const SVM_GUEST: i32 = offset_of!(Cpu, svm.guest) as i32;
const _: () = assert!(size_of::<bool>() == 1);

const _: () = assert!(size_of::<LazyFlagsOp>() == 1);
const _: () = assert!(offset_of!(Cpu, lazy_size) == offset_of!(Cpu, lazy_op) + 1);

// `Cpu::translate` indexes `tlb` as `[user << TLB_INDEX_BITS | page_number & (ENTRIES - 1)]`; the
// inline probe gets the user/supervisor half from `Cpu::tlb_active_base` and only adds the
// in-half index, scaled by the size of a `TlbEntry` (32 bytes; `Cpu::set_cpl` uses the same
// scale for its half offset).
const TLB_TAG_READ: i32 = offset_of!(TlbEntry, tag_r) as i32;
const TLB_TAG_WRITE: i32 = offset_of!(TlbEntry, tag_w) as i32;
const TLB_ADDEND: i32 = offset_of!(TlbEntry, addend) as i32;
const TLB_TAG_EXEC: i32 = offset_of!(TlbEntry, tag_x) as i32;
const TLB_ENTRY_SHIFT: u8 = size_of::<TlbEntry>().trailing_zeros() as u8;
const _: () = assert!(size_of::<TlbEntry>().is_power_of_two());

const PAGE_SHIFT: u8 = 12;
/// Sign-extends to `!0xfff` in a 64-bit `and`.
const PAGE_MASK: i32 = !((1 << PAGE_SHIFT) - 1);

// Host register roles. RBX, RBP and R12..R15 are callee-saved on both ABIs and the prologue saves
// them (and the Win64 callee-saved RSI/RDI too); everything else is clobbered by any helper call.
pub(super) const CPU: u8 = RBX;
/// `Cpu::tlb_active_base`, loaded once at block entry. CPL only changes inside block-terminating
/// instructions, so it cannot go stale within a block.
pub(super) const TLB_BASE: u8 = RBP;
/// First ALU operand and result; also the destination of a loaded first operand.
const ACC: u8 = RAX;
/// Second ALU operand, the shift count, and the register that carries `lazy_src` to `store_lazy`.
const OPERAND: u8 = RCX;
/// Guest linear address of the memory operand being accessed.
const ADDR: u8 = RDX;
/// Scratch: TLB entry pointer during a probe, index/segment base in `ea`.
const SCRATCH_A: u8 = RSI;
/// Scratch: the TLB addend while completing an access.
const SCRATCH_B: u8 = RDI;
/// Scratch: the page tag compared against the TLB entry, and the carry bit of shifts and imul.
const SCRATCH_C: u8 = R8;
/// Scratch for 64-bit constants stored to `Cpu` (rip, icount), and the overflow bit of shifts.
const SCRATCH_D: u8 = R10;
/// Value being stored to memory or just loaded; moves between a guest register and memory.
const VALUE: u8 = R9;
/// Holds the helper address for `call`; helpers are too far away for a rel32 call.
const CALL_TARGET: u8 = R11;
const SAVED_ADDR: u8 = R12;
const SAVED_VALUE: u8 = R13;
/// The result and second operand of a read-modify-write instruction whose store takes the slow
/// path: its flags are recorded only after the store succeeded, so both must outlive the helper
/// call. Also the home of a value that a memory access in the middle of an instruction would
/// otherwise lose (an inc/dec carry-in, the outcome of a condition code for a cmov with a memory
/// source); those are dead before any such store runs, hence the sharing.
const SAVED_RESULT: u8 = R14;
const SAVED_OPERAND: u8 = R15;
const SAVED_CONDITION: u8 = R14;

const GUEST_RAX: u8 = 0;
const GUEST_RCX: u8 = 1;
const GUEST_RDX: u8 = 2;
const GUEST_RBX: u8 = 3;
const GUEST_RSP: u8 = 4;
const GUEST_RBP: u8 = 5;
/// `Insn::rex` bit B, which extends the register number encoded in the opcode.
const GUEST_REX_B: u8 = 1;
const REX_B_REG_SHIFT: u8 = 3;
const OPCODE_REG_MASK: u8 = 7;
const MODRM_MOD_REGISTER: u8 = 3;
/// Without a REX prefix, byte-register numbers 4..8 mean ah/ch/dh/bh (bits 8..16 of guest
/// registers 0..4), not the low byte of rsp/rbp/rsi/rdi.
const HIGH_BYTE_REGS: std::ops::Range<u8> = 4..8;
const HIGH_BYTE_REG_BASE: u8 = 4;
/// Byte offset of bits 15:8 inside a guest register slot (little-endian host).
const HIGH_BYTE_OFFSET: i32 = 1;
const PUSH_POP_SIZE: i32 = 8;

/// Condition code 2 (B): CF.
const CONDITION_BELOW: u64 = 2;

const NOP: u8 = 0x90;
/// `mov rdi, imm64` (REX.W B8+7), the expected host page of a cross-page link (`fixed_exit`).
const MOVABS_RDI: [u8; 2] = [0x48, 0xbf];
const MOVABS_IMMEDIATE_OFFSET: u64 = MOVABS_RDI.len() as u64;
const CROSS_PAGE_WORD_ALIGN: u64 = 8;

/// Stack frame the prologue reserves below its saved registers. The eight pushes leave rsp at
/// 8 mod 16, so the frame must be 8 mod 16 for calls from blocks to see rsp 16-byte aligned (both
/// ABIs require it). It also has to hold the 32 bytes of Win64 shadow space at the bottom, with
/// the slow-path load result slot (`OUT`) above it. Shared with `arch_x64`.
pub(crate) const FRAME: i32 = 72;
/// Offset from rsp of the 8-byte slot holding the guest address of the page the running block
/// belongs to (`rip & !0xfff`), where aarch64 keeps it in a register (`PAGE_BASE` in `emit.rs`;
/// every host register here has a role). The trampoline sets it from `Cpu::rip`, the chain stub
/// from its target, a cross-page link's check (`Gen::fixed_exit`) from the page it checked, and a
/// same-page link keeps it. Above `OUT`, in the
/// frame's spare bytes, which no callee touches.
pub(crate) const PAGE_BASE_SLOT: i32 = 40;
/// Where the trampoline keeps the jump cache of the vCPU running the block (the code and the chain
/// stub are shared by every vCPU, the cache is not).
pub(crate) const JUMP_CACHE_SLOT: i32 = 48;
/// The validity table (`Jit::valid`), passed the same way.
pub(crate) const VALID_SLOT: i32 = 56;
const _: () = assert!(PAGE_BASE_SLOT >= OUT + 8 && VALID_SLOT + 8 <= FRAME);
/// Offset from rsp of the 8-byte slot `helper_load_via_out` writes its result to. Above the shadow space so the
/// callee cannot spill its register arguments over it.
const OUT: i32 = 32;

/// Group-1 ALU operation numbers (the x86 opcode extension), plus a pseudo operation.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_XOR: u8 = 6;
const ALU_CMP: u8 = 7;
/// `test`: and without writeback. Not a real group-1 number; must stay outside 0..=7.
const ALU_TEST: u8 = 8;

/// ModRM.reg of the shift group (c0/c1/d0..d3): rol, ror, rcl, rcr, shl, shr, sal (an alias of
/// shl), sar.
/// Shift counts of 8-, 16- and 32-bit operands are masked to five bits (SDM, SHL).
const NARROW_SHIFT_COUNT_MASK: u32 = 0x1f;
/// Width of the host shift a narrow operand is widened to.
const HOST_SHIFT_BITS: u8 = 32;
/// `Insn::op` of the VEX map-2 (0f38) BMI opcodes: bextr/shlx/sarx/shrx, bzhi/pdep/pext, andn,
/// and the blsr/blsmsk/blsi group (`cpu/bmi.rs`).
const VEX_BMI_SHIFT_OR_BEXTR: u16 = VEX_OP_FLAG | 0x2f7;
const VEX_BMI_BZHI_PEXT_PDEP: u16 = VEX_OP_FLAG | 0x2f5;
const VEX_BMI_ANDN: u16 = VEX_OP_FLAG | 0x2f2;
const VEX_BMI_GROUP17: u16 = VEX_OP_FLAG | 0x2f3;
/// mulx (VEX F2 0f38 f6); without VEX the same opcode is adcx (66) and adox (F3).
const VEX_BMI_MULX: u16 = VEX_OP_FLAG | 0x2f6;
const ADCX_ADOX: u16 = 0x2f6;
/// `Insn::rep` for an F3 and an F2 prefix (also the mandatory prefixes of SSE and VEX forms).
const PREFIX_F3: u8 = 1;
const PREFIX_F2: u8 = 2;
/// movd/movq xmm, r/m (66 0f 6e) and r/m, xmm (66 0f 7e).
const MOVD_TO_VECTOR: u16 = 0x16e;
const MOVD_FROM_VECTOR: u16 = 0x17e;
/// bt/bts/btr/btc r/m, r (0f a3, ab, b3, bb) and group 8 (0f ba /4../7) with an imm8 offset.
const BIT_TEST_REGISTER_OFFSET: [(u16, u8); 4] = [
    (0x1a3, BIT_TEST),
    (0x1ab, BIT_SET),
    (0x1b3, BIT_RESET),
    (0x1bb, BIT_COMPLEMENT),
];
const BIT_TEST_IMMEDIATE: u16 = 0x1ba;
/// Group 8's /4 is bt; /5, /6 and /7 are bts, btr and btc (/0../3 are #UD).
const GROUP8_FIRST_BIT_OP: u8 = 4;
/// Group 3 /6 and /7: div and idiv.
const GROUP3_DIV: u8 = 6;
const GROUP3_IDIV: u8 = 7;
/// VEX 0f 98 kortest and 0f 99 ktest (the opmask register tests).
const KORTEST: u16 = VEX_OP_FLAG | 0x198;
const KTEST: u16 = VEX_OP_FLAG | 0x199;
/// VEX 0f 77: vzeroupper (VEX.L 0) and vzeroall (VEX.L 1).
const VZERO: u16 = VEX_OP_FLAG | 0x177;
/// Vector registers vzeroupper and vzeroall clear in 64-bit mode.
const VZERO_REGISTERS: u8 = 16;
/// CR4.TSD: rdtsc above CPL 0 raises #GP.
const CR4_TSD: u64 = 1 << 2;
/// CR4.PKE: rdpkru and wrpkru exist.
const CR4_PKE: u64 = 1 << 22;
/// EFLAGS.IOPL (bits 13:12) and the flags pushf leaves out of its image.
const IOPL_SHIFT: u8 = 12;
const IOPL_MASK: i32 = 3;
const PUSHF_CLEARED: u64 = RF | VM;
/// Operand of cmpxchg16b, which must be aligned to it.
const CMPXCHG16B_BYTES: u8 = 16;
/// Condition code 7 (A, unsigned above).
const CONDITION_ABOVE: u8 = 7;
const BMI_GROUP_BLSR: u8 = 1;
const BMI_GROUP_BLSMSK: u8 = 2;
const BMI_GROUP_BLSI: u8 = 3;
const SHIFT_SHL: u8 = 4;
const SHIFT_SHR: u8 = 5;
const SHIFT_SAL_ALIAS: u8 = 6;
const SHIFT_SAR: u8 = 7;

/// `SzpCfOf` packs CF in bit 0 and OF in bit 1 of `lazy_src`; the shift code builds that word
/// with a scaled `lea`, which needs OF's position to be the scale.
const _: () = assert!(SZP_CF_OF_SRC_CF_BIT == 0 && SZP_CF_OF_SRC_OF_BIT == 1);
/// Extra bit above CF/OF in the word a shift by CL leaves in `OPERAND`: set when the masked count
/// was not zero, i.e. when the flags change. It has to survive the destination store (a slow
/// path keeps `OPERAND`), where the count itself would not, and is masked off before recording.
const SHIFT_APPLIED: i32 = 1 << 2;
const CF_OF_MASK: i32 = (1 << 2) - 1;
/// Value of CF = OF in the `SzpCfOf` word when `imul`'s product does not fit.
const IMUL_OVERFLOW_FLAGS: u8 = 3;

/// A TLB-miss path, emitted out of line after the block body.
enum SlowKind {
    /// A whole-instruction fallback (`vector_move`): the probe or a gate check failed, so the
    /// interpreter runs the instruction and raises the right fault.
    Exec,
    /// `Exec` for a near call or ret whose shadow-stack access missed the TLB or found another
    /// return address (`shadow_stack_push`/`shadow_stack_pop`): the interpreter runs the whole
    /// instruction, leaving `Cpu::rip` at its target, and the block ends. `resume` is unused.
    ExecAndLeave,
    Ld {
        dst: u8,
        size: u8,
        signed: bool,
        stack: bool,
    },
    St {
        size: u8,
        stack: bool,
        /// The caller still needs `ACC` and `OPERAND` after the store (result and operand of a
        /// read-modify-write); the slow path parks them in `SAVED_RESULT`/`SAVED_OPERAND`.
        preserve: bool,
    },
}

struct Slow {
    /// Taken from the inline TLB probe on a miss.
    slow: Label,
    /// Where the inline path continues after the access completes.
    resume: Label,
    kind: SlowKind,
    /// Block index of the accessing instruction, reported to `exit_fail` if the access faults.
    insn_index: u32,
    /// Guest rip of that instruction, stored to `Cpu::rip` before the helper runs because
    /// `helper_load_via_out`/`helper_store_x64` do not set it.
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
enum BitScan {
    Bsf,
    Bsr,
    Tzcnt,
    Lzcnt,
}

/// The read-modify-write of a memory operand that `Gen::memory_rmw` makes atomic.
#[derive(Clone, Copy)]
enum RmwOp {
    /// add, or, and, sub or xor (`ALU_*`) with a register or immediate operand
    Alu(u8, Src),
    Inc,
    Dec,
    Xadd,
    Xchg,
    Cmpxchg,
    /// bts, btr or btc (`BIT_SET`..`BIT_COMPLEMENT`) with an immediate bit offset (masked to the
    /// width)
    BitTest(u8, u8),
}

#[derive(Clone, Copy, PartialEq)]
enum ShiftKind {
    Shl,
    Shr,
    Sar,
}

/// Absolute addresses of the helpers and of the shared exit stubs, loaded into registers at the
/// call sites. Built once per `Jit` by `arch_x64::addrs`.
pub(crate) struct Addrs {
    pub exec: u64,
    pub ld: u64,
    pub st: u64,
    /// Variants of `ld`/`st` for push/pop/call/ret, which fault with #SS instead of #GP.
    pub ld_stack: u64,
    pub st_stack: u64,
    pub cond: u64,
    /// `helper_whole_vector(cpu, packed, source low, source high)`.
    pub whole_vector: u64,
    /// `helper_materialize_flags(cpu, unused)`.
    pub materialize_flags: u64,
    /// `helper_rdtsc(cpu, unused)` and `helper_pause(cpu, unused)`.
    pub rdtsc: u64,
    pub pause: u64,
    pub exit_ok: u64,
    pub exit_fail: u64,
    /// The block-to-block transfer stub (see `arch_x64::chain_stub`), or 0 when chaining is off
    /// and every block returns to the run loop.
    pub chain: u64,
}

pub(crate) struct Gen<'a> {
    pub a: Asm,
    addrs: &'a Addrs,
    /// Generated code embeds raw pointers into this slice (for `helper_exec`), so the caller must keep
    /// it alive as long as the block.
    insns: &'a [Insn],
    slows: Vec<Slow>,
    /// Out-of-line `(label, insn_index, next rip)` exits taken when a store requested a block exit.
    smc: Vec<(Label, u32, u64)>,
    /// Taken sides of conditional branches before the last instruction, emitted out of line
    /// by `emit_slows`: (label, target, retired, `prepaid_insns` at the branch).
    side_exits: Vec<(Label, u64, u32, u32)>,
    /// Host flags currently equal the guest flags (set by the last inlined flag-setting
    /// instruction). Every instruction that emits code which may clobber them must clear this;
    /// the memory access paths do it in `tlb_probe`.
    live: bool,
    /// An inline SSE instruction of this block already checks CR0.EM/TS and CR4.OSFXSR. Whatever
    /// changes them (mov to cr, clts, lmsw, vmrun...) ends a block, so the first check covers the
    /// rest.
    sse_gated: bool,
    /// The block already checked the AVX-512 state gate (CR4.OSXSAVE, XCR0, CR0.TS).
    evex_gated: bool,
    /// The block already checked the AVX state gate (`VectorGate::Avx`), or the AVX-512 one,
    /// which implies it. Not implied by `sse_gated`: the SSE gate checks other bits.
    vex_gated: bool,
    /// Block index of the last instruction whose clock read was paid for in advance; see
    /// `Gen::helper` and the field of the same name in `emit.rs`.
    prepaid_insns: u32,
    /// Number of guest instructions lowered inline (not through `helper_exec`), for statistics.
    pub inlined: u32,
    /// Offset of the block's first instruction in its page; every `rip` here is such an offset
    /// (see `rip_to`).
    entry_rip: u64,
    /// Byte offset of the body entry; the bytes before it are the link entry's checks (none
    /// without chaining). See `emit.rs`.
    pub body_offset: u32,
    /// Label of the link entry's out-of-line exit, bound by `emit_slows`.
    link_entry_miss: Option<Label>,
    /// Patchable slots of exits to fixed addresses in the block's own page.
    pub links: Vec<LinkSite>,
    /// Where the targets of cross-page exits are mapped as the block is translated (see
    /// `emit.rs`).
    pub resolve: ResolveTarget<'a>,
    /// Per instruction: must the lazy flags it sets be recorded (`helpers::flag_record_needed`).
    flag_record_needed: Vec<bool>,
    /// The instruction being lowered has a dead flag record, which `store_lazy_op` leaves out.
    skip_flag_record: bool,
    /// Flag records left out, for `JitStats`.
    pub flag_records_skipped: u32,
}

/// The register embedded in the low three opcode bits (push/pop/mov r, imm), extended by REX.B.
fn opcode_register(insn: &Insn) -> u8 {
    (insn.op as u8 & OPCODE_REG_MASK) | ((insn.rex & GUEST_REX_B) << REX_B_REG_SHIFT)
}

fn has_fs_gs_override(insn: &Insn) -> bool {
    matches!(insn.seg as usize, SEG_FS | SEG_GS)
}

/// Memory slot of a guest register, or of its bits 15:8 for ah/ch/dh/bh.
fn register_slot(reg: u8, high: bool) -> Mem {
    mem(
        CPU,
        GPR + GPR_SIZE * reg as i32 + if high { HIGH_BYTE_OFFSET } else { 0 },
    )
}

impl<'a> Gen<'a> {
    pub fn new(base: u64, addrs: &'a Addrs, insns: &'a [Insn]) -> Self {
        Gen {
            a: Asm::new(base),
            addrs,
            insns,
            slows: Vec::new(),
            smc: Vec::new(),
            side_exits: Vec::new(),
            live: false,
            sse_gated: false,
            evex_gated: false,
            vex_gated: false,
            prepaid_insns: 0,
            inlined: 0,
            entry_rip: 0,
            body_offset: 0,
            link_entry_miss: None,
            links: Vec::new(),
            resolve: &|_| None,
            flag_record_needed: flag_record_needed(insns),
            skip_flag_record: false,
            flag_records_skipped: 0,
        }
    }

    fn ld_gpr(&mut self, rd: u8, r: u8) {
        self.a.ld(rd, mem(CPU, GPR + GPR_SIZE * r as i32));
    }

    fn st_gpr(&mut self, r: u8, rs: u8) {
        self.a.st(mem(CPU, GPR + GPR_SIZE * r as i32), rs, 8);
    }

    fn call(&mut self, addr: u64) {
        self.a.mov_imm(CALL_TARGET, addr);
        self.a.call_reg(CALL_TARGET);
    }

    /// `dest = page base + rip`, where `rip` is the offset in the block's page that blocks carry
    /// around instead of guest addresses (wrapping: a jump to a lower page is negative). Blocks
    /// are position independent, see `PAGE_BASE_SLOT`. Leaves host flags alone.
    fn rip_to(&mut self, dest: u8, rip: u64) {
        self.a.ld(dest, mem(RSP, PAGE_BASE_SLOT));
        match i32::try_from(rip as i64) {
            Ok(0) => {}
            Ok(offset) => self.a.lea(dest, mem(dest, offset)),
            Err(_) => {
                self.a.mov_imm(CALL_TARGET, rip);
                self.a.lea(dest, mem_idx(dest, CALL_TARGET, 0, 0));
            }
        }
    }

    fn set_rip(&mut self, rip: u64) {
        self.rip_to(SCRATCH_D, rip);
        self.a.st(mem(CPU, RIP), SCRATCH_D, 8);
    }

    /// Leave the block normally having retired `retired` instructions.
    fn exit_ok(&mut self, retired: u32) {
        self.a.mov_imm(ACC, (retired as u64) << EXIT_COUNT_SHIFT);
        self.a.jmp_abs(self.addrs.exit_ok);
    }

    /// Ends the block by continuing at the guest address in `OPERAND` (rcx), with `retired`
    /// instructions done. With chaining that is a jump through the chain stub, which wants rax =
    /// `icount` as just stored and already including the successor's first instruction (see
    /// `arch_x64::chain_stub`): the block pays for its own instructions except those paid ahead
    /// of a helper, plus that one. Without chaining it stores rip and returns to the run loop.
    /// Clobbers the host flags, so conditional exits call it only after the branch.
    fn chain_exit(&mut self, retired: u32) {
        self.chain_exit_to(retired, None);
    }

    /// `chain_exit`; with a fixed `target` the exit also gets a link slot (`link_slot`).
    fn chain_exit_to(&mut self, retired: u32, target: Option<u64>) {
        if self.addrs.chain == 0 {
            self.a.st(mem(CPU, RIP), OPERAND, 8);
            self.exit_ok(retired);
            return;
        }
        let owed = retired - self.prepaid_insns;
        self.a.ld(ACC, mem(CPU, ICOUNT));
        self.a.alu_ri(ADD, ACC, owed as i32, true);
        self.a.st(mem(CPU, ICOUNT), ACC, 8);
        if self.prepaid_insns > 0 {
            // The next block's own accounting must not subtract this block's prepayment.
            self.a.mov_imm(SCRATCH_D, 0);
            self.a.st(mem(CPU, PREPAID_ICOUNT), SCRATCH_D, 8);
        }
        if let Some(target) = target {
            self.fixed_exit(target);
        }
        self.a.jmp_abs(self.addrs.chain);
    }

    /// The slot in front of the `jmp chain` of an exit to the fixed guest address `target` (in
    /// rcx, `icount` paid). Within the block's page it is the slot alone. For another page it sits
    /// behind an inline check that the target's page still maps to the host page `resolve` found
    /// at translation, which also sets the page base for the successor; see `emit.rs`
    /// (`Gen::fixed_exit`) and `arch_a64::cross_page_check` for why. Inline here: x86 code is
    /// compact, and a call that does not return would unbalance the return predictor. rdx, rsi
    /// and rdi are free at an exit, as for the chain stub.
    fn fixed_exit(&mut self, target: u64) {
        if target >> PAGE_SHIFT == self.entry_rip >> PAGE_SHIFT {
            self.link_slot(target, None);
            return;
        }
        if self.addrs.chain == 0 {
            return;
        }
        let resolved = (self.resolve)(target);
        let (entry, page, expected) = (RDX, RSI, RDI);
        let elsewhere = self.a.label();
        // `mov rdi, imm64` with the immediate 8-byte aligned, so `Jit::link_unresolved` can
        // replace it with one atomic store while another vCPU runs past it.
        while !(self.a.here() + MOVABS_IMMEDIATE_OFFSET).is_multiple_of(CROSS_PAGE_WORD_ALIGN) {
            self.a.code.push(NOP);
        }
        let word_offset = (self.a.here() - self.a.base()) as u32 + MOVABS_IMMEDIATE_OFFSET as u32;
        self.a.code.extend_from_slice(&MOVABS_RDI);
        self.a.code.extend_from_slice(
            &resolved
                .map_or(0, |resolved| resolved.host_page)
                .to_le_bytes(),
        );
        self.a.mov_rr(entry, OPERAND, true);
        self.a.shift_ri(SHR, entry, PAGE_SHIFT, true);
        self.a
            .alu_ri(AND, entry, (TLB_ENTRIES_PER_MODE - 1) as i32, true);
        self.a.shift_ri(SHL, entry, TLB_ENTRY_SHIFT, true);
        self.a.alu_rr(ADD, entry, TLB_BASE, true);
        self.a.mov_rr(page, OPERAND, true);
        self.a.alu_ri(AND, page, PAGE_MASK, true);
        self.a.cmp_mr(mem(entry, TLB_TAG_EXEC), page);
        self.a.jcc(NE, elsewhere);
        self.a.ld(entry, mem(entry, TLB_ADDEND));
        self.a.alu_rr(ADD, entry, page, true);
        self.a.alu_rr(CMP, entry, expected, true);
        self.a.jcc(NE, elsewhere);
        self.a.st(mem(RSP, PAGE_BASE_SLOT), page, 8);
        self.link_slot(target, resolved.map(|resolved| resolved.phys_page));
        if let Some(link) = self.links.last_mut() {
            link.word_offset = Some(word_offset);
        }
        self.a.bind(elsewhere);
    }

    /// Emits the `jmp rel32` that `Jit::translate` points at the block for `target_rip` once it
    /// exists, when `target_rip` is in this block's page; see `emit.rs`. It sits right before the
    /// `jmp chain`, so while it is unpatched (a jump to the next instruction) the stub path runs.
    /// The displacement is padded to a 4-byte boundary so one store replaces it.
    fn link_slot(&mut self, target_rip: u64, phys_page: Option<u64>) {
        if self.addrs.chain == 0 {
            return;
        }
        while !(self.a.here() + LINK_JMP_OPCODE_BYTES).is_multiple_of(LINK_WORD_ALIGN) {
            self.a.code.push(NOP);
        }
        self.links.push(LinkSite {
            code_offset: (self.a.here() - self.a.base()) as u32,
            target_rip,
            word_offset: None,
            phys_page,
        });
        self.a.code.extend_from_slice(&LINK_JMP);
    }

    /// The checks `chain_stub` does before it jumps into a block, for a block reached by a direct
    /// link: see `emit.rs`.
    fn link_entry(&mut self) {
        let miss = self.a.label();
        self.a.ld(ACC, mem(CPU, ICOUNT));
        self.a.ld(SCRATCH_A, mem(CPU, NEXT_EVENT));
        self.a.alu_rr(CMP, ACC, SCRATCH_A, true);
        self.a.jcc(AE, miss);
        self.a.cmp_m8_imm(mem(CPU, BLOCK_EXIT_REQUESTED), 0);
        self.a.jcc(NE, miss);
        self.link_entry_miss = Some(miss);
    }

    /// `chain_exit` to a fixed guest address.
    fn chain_to(&mut self, rip: u64, retired: u32) {
        self.rip_to(OPERAND, rip);
        self.chain_exit_to(retired, Some(rip));
    }

    /// Leave the block because the status in rax is non-zero; the epilogue wants the failing
    /// instruction's block index in rcx.
    fn fail_exit(&mut self, insn_index: u32) {
        self.a.mov_imm(OPERAND, insn_index as u64);
        self.a.jmp_abs(self.addrs.exit_fail);
    }

    /// Run instruction `insn_index` through the interpreter. `terminal` ends the block after it.
    fn helper(&mut self, insn_index: usize, rip: u64, terminal: bool) {
        let insn = &self.insns[insn_index] as *const Insn as u64;
        self.prepay(insn_index);
        self.a.mov_rr(ARGS[0], CPU, true);
        self.a.mov_imm(ARGS[1], insn);
        self.rip_to(ARGS[2], rip);
        self.call(self.addrs.exec);
        let ok = self.a.label();
        self.a.test_rr(ACC, ACC, true);
        self.a.jcc(E, ok);
        self.fail_exit(insn_index as u32);
        self.a.bind(ok);
        if terminal {
            self.exit_ok(insn_index as u32 + 1);
        }
        self.live = false;
    }

    /// Adds the instructions before `insn_index` to `Cpu::icount` ahead of something that reads
    /// the clock in the middle of the block (an interpreter helper, rdtsc).
    fn prepay(&mut self, insn_index: usize) {
        // The run loop already counted one instruction for the block. The interpreter may read
        // guest time (rdtsc, port I/O, a device load), which derives from `icount`, so add the
        // ones before this now and tell `step_block` through `prepaid_icount` not to count them
        // again.
        if insn_index as u32 > self.prepaid_insns {
            // Pay only for the instructions executed since the last payment and record the
            // running total in `prepaid_icount`; see the same code in `emit.rs`.
            let owed = insn_index as u32 - self.prepaid_insns;
            self.a.ld(SCRATCH_D, mem(CPU, ICOUNT));
            self.a.alu_ri(ADD, SCRATCH_D, owed as i32, true);
            self.a.st(mem(CPU, ICOUNT), SCRATCH_D, 8);
            self.a.mov_imm(SCRATCH_D, insn_index as u64);
            self.a.st(mem(CPU, PREPAID_ICOUNT), SCRATCH_D, 8);
            self.prepaid_insns = insn_index as u32;
        }
    }

    /// Guest linear address of `insn`'s memory operand into `ADDR`. Only lea and moves here: the
    /// effective address must not disturb live host flags.
    fn ea(&mut self, insn: &Insn, next: u64) {
        let disp = insn.disp as i32;
        if insn.rip_rel {
            self.rip_to(ADDR, next.wrapping_add(insn.disp as u64));
        } else if insn.index != NO_REG {
            if insn.base != NO_REG {
                self.ld_gpr(ADDR, insn.base);
            } else {
                self.a.mov_imm(ADDR, 0);
            }
            self.ld_gpr(SCRATCH_A, insn.index);
            self.a.lea(ADDR, mem_idx(ADDR, SCRATCH_A, insn.scale, disp));
        } else if insn.base != NO_REG {
            self.ld_gpr(ADDR, insn.base);
            if disp != 0 {
                self.a.lea(ADDR, mem(ADDR, disp));
            }
        } else {
            self.a.mov_imm(ADDR, insn.disp as u64);
        }
        // In 64-bit mode only the FS and GS bases apply; other overrides are ignored.
        if has_fs_gs_override(insn) {
            self.a
                .ld(SCRATCH_A, mem(CPU, SEG_BASE + SEG_SIZE * insn.seg as i32));
            self.a.lea(ADDR, mem_idx(ADDR, SCRATCH_A, 0, 0));
        }
    }

    /// Inline mirror of the fast path in `Cpu::mem_read`/`mem_write`. Jumps to `slow` on a miss;
    /// on a hit falls through with `SCRATCH_A` pointing at the TLB entry. Clobbers host flags.
    fn tlb_probe(&mut self, size: u8, tag_offset: i32, slow: Label) {
        self.tlb_probe_tagged(size, tag_offset, slow, 0);
    }

    /// `tlb_probe` for a tag that is the page plus `tag_bits` (`TLB_TAG_SHADOW_STACK`).
    fn tlb_probe_tagged(&mut self, size: u8, tag_offset: i32, slow: Label, tag_bits: u64) {
        self.live = false;
        self.a.mov_rr(SCRATCH_A, ADDR, true);
        self.a.shift_ri(SHR, SCRATCH_A, PAGE_SHIFT, true);
        self.a
            .alu_ri(AND, SCRATCH_A, (TLB_ENTRIES_PER_MODE - 1) as i32, true);
        self.a.shift_ri(SHL, SCRATCH_A, TLB_ENTRY_SHIFT, true);
        self.a.lea(SCRATCH_A, mem_idx(TLB_BASE, SCRATCH_A, 0, 0));
        // Compare the page of the access's last byte against the tag. The entry was picked by the
        // first byte's page, so an access that crosses a page can never match (the next page
        // maps to a different slot) and takes the slow path, which splits it. This replaces the
        // interpreter's separate `(la & 0xfff) + size <= 4096` check. TLB_TAG_INVALID has low
        // bits set, so it never equals a page-aligned value.
        if size == 1 {
            self.a.mov_rr(SCRATCH_C, ADDR, true);
        } else {
            self.a.lea(SCRATCH_C, mem(ADDR, size as i32 - 1));
        }
        self.a.alu_ri(AND, SCRATCH_C, PAGE_MASK, true);
        if tag_bits != 0 {
            self.a.alu_ri(OR, SCRATCH_C, tag_bits as i32, true);
        }
        self.a.cmp_mr(mem(SCRATCH_A, tag_offset), SCRATCH_C);
        self.a.jcc(NE, slow);
    }

    /// Load `size` bytes at `ADDR` into `dst`. A TLB miss runs the helper out of line and resumes
    /// after the load with `ADDR` intact.
    fn mem_load(&mut self, dst: u8, size: u8, signed: bool, insn_index: u32, rip: u64) {
        self.load_access(dst, size, signed, insn_index, rip, false);
    }

    /// `mem_load` for an implicit stack access: only the slow path differs (#SS, not #GP).
    fn stack_load(&mut self, dst: u8, insn_index: u32, rip: u64) {
        self.load_access(dst, 8, false, insn_index, rip, true);
    }

    fn load_access(
        &mut self,
        dst: u8,
        size: u8,
        signed: bool,
        insn_index: u32,
        rip: u64,
        stack: bool,
    ) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, TLB_TAG_READ, slow);
        // The addend turns a guest linear address into a host pointer: host = la + addend.
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a
            .ld_ext(dst, mem_idx(ADDR, SCRATCH_B, 0, 0), size, signed);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Ld {
                dst,
                size,
                signed,
                stack,
            },
            insn_index,
            rip,
        });
    }

    /// Store the low `size` bytes of `VALUE` at `ADDR`. The write tag is invalidated for pages
    /// holding translated code, so stores to code always miss and reach `Cpu::code_written`.
    fn mem_store(&mut self, size: u8, insn_index: u32, rip: u64) {
        self.store_access(size, insn_index, rip, false, false);
    }

    /// `mem_store` for an implicit stack access; see `stack_load`.
    fn stack_store(&mut self, insn_index: u32, rip: u64) {
        self.store_access(8, insn_index, rip, true, false);
    }

    /// `mem_store` for a read-modify-write instruction: `ACC` (result) and `OPERAND` are still
    /// wanted afterwards to record the flags, so a slow-path store keeps them.
    fn mem_store_keeping_operands(&mut self, size: u8, insn_index: u32, rip: u64) {
        self.store_access(size, insn_index, rip, false, true);
    }

    fn store_access(&mut self, size: u8, insn_index: u32, rip: u64, stack: bool, preserve: bool) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.tlb_probe(size, TLB_TAG_WRITE, slow);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a.st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, size);
        self.a.bind(resume);
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::St {
                size,
                stack,
                preserve,
            },
            insn_index,
            rip,
        });
    }

    /// CET: with a shadow stack in use, pushes the return address `return_rip` on it too. The
    /// run-time test of `Cpu::shadow_stack_on`, the shadow-stack TLB tag and the whole-instruction
    /// fallback are explained at `shadow_stack_push` in `emit.rs`. Emit after the ordinary push's
    /// store and before rsp is written. Keeps `ADDR` (through `SCRATCH_D`).
    fn shadow_stack_push(&mut self, return_rip: u64, insn_index: u32, rip: u64) {
        let skip = self.a.label();
        let fallback = self.shadow_stack_fallback(insn_index, rip);
        self.a.cmp_m8_imm(mem(CPU, SHADOW_STACK_ON), 0);
        self.a.jcc(E, skip);
        // The ordinary push's slow path clobbers `VALUE`.
        self.rip_to(VALUE, return_rip);
        self.a.mov_rr(SCRATCH_D, ADDR, true);
        self.a.ld(ADDR, mem(CPU, SSP));
        self.a.lea(ADDR, mem(ADDR, -SHADOW_STACK_ENTRY));
        self.tlb_probe_tagged(8, TLB_TAG_WRITE, fallback, TLB_TAG_SHADOW_STACK);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a.st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, 8);
        self.a.st(mem(CPU, SSP), ADDR, 8);
        self.a.mov_rr(ADDR, SCRATCH_D, true);
        self.a.bind(skip);
        self.live = false;
    }

    /// The shadow-stack half of a near `ret` whose target is in `VALUE`: pops the shadow stack's
    /// copy and compares, leaving a mismatch (#CP) to the interpreter like a TLB miss. Keeps
    /// `ADDR` and `VALUE`.
    fn shadow_stack_pop(&mut self, insn_index: u32, rip: u64) {
        let skip = self.a.label();
        let fallback = self.shadow_stack_fallback(insn_index, rip);
        self.a.cmp_m8_imm(mem(CPU, SHADOW_STACK_ON), 0);
        self.a.jcc(E, skip);
        self.a.mov_rr(SCRATCH_D, ADDR, true);
        self.a.ld(ADDR, mem(CPU, SSP));
        self.tlb_probe_tagged(8, TLB_TAG_WRITE, fallback, TLB_TAG_SHADOW_STACK);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a.ld(SCRATCH_C, mem_idx(ADDR, SCRATCH_B, 0, 0));
        self.a.alu_rr(CMP, SCRATCH_C, VALUE, true);
        self.a.jcc(NE, fallback);
        self.a.lea(ADDR, mem(ADDR, SHADOW_STACK_ENTRY));
        self.a.st(mem(CPU, SSP), ADDR, 8);
        self.a.mov_rr(ADDR, SCRATCH_D, true);
        self.a.bind(skip);
        self.live = false;
    }

    fn shadow_stack_fallback(&mut self, insn_index: u32, rip: u64) -> Label {
        let slow = self.a.label();
        let resume = self.a.label();
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::ExecAndLeave,
            insn_index,
            rip,
        });
        slow
    }

    /// After a store: if it invalidated translated code (`Cpu::code_written` sets
    /// `block_exit_requested`), leave the block so the rest of it does not run from stale code.
    fn smc_check(&mut self, insn_index: u32, next: u64) {
        let exit = self.a.label();
        self.a.cmp_m8_imm(mem(CPU, BLOCK_EXIT_REQUESTED), 0);
        self.a.jcc(NE, exit);
        self.smc.push((exit, insn_index, next));
        self.live = false;
    }

    /// Emit everything deferred out of line, then resolve labels. Must be the last thing emitted
    /// for a block.
    ///
    /// Slow paths call `helper_load_via_out`/`helper_store_x64` with the guest address saved in `SAVED_ADDR` (and a
    /// store's value in `SAVED_VALUE`), because the call clobbers every other working register.
    /// They restore `ADDR` before jumping back, so the inline code may only rely on `ADDR` (and
    /// on `dst`, for loads, and `ACC`/`OPERAND` for a preserving store) surviving a memory
    /// access.
    fn emit_slows(&mut self) {
        let side_exits = std::mem::take(&mut self.side_exits);
        let prepaid_at_end = self.prepaid_insns;
        for (label, target, retired, prepaid) in side_exits {
            self.a.bind(label);
            // Pay as the branch would have: helpers after it may have prepaid more.
            self.prepaid_insns = prepaid;
            self.chain_to(target, retired);
        }
        self.prepaid_insns = prepaid_at_end;
        let smc = std::mem::take(&mut self.smc);
        for (label, insn_index, next) in smc {
            self.a.bind(label);
            self.set_rip(next);
            self.exit_ok(insn_index + 1);
        }
        if let Some(miss) = self.link_entry_miss.take() {
            // Give back the pre-charge and return to the run loop at this block's first
            // instruction, as the stub's miss path does.
            self.a.bind(miss);
            self.a.ld(ACC, mem(CPU, ICOUNT));
            self.a.alu_ri(SUB, ACC, 1, true);
            self.a.st(mem(CPU, ICOUNT), ACC, 8);
            self.set_rip(self.entry_rip);
            self.exit_ok(1);
        }
        let slows = std::mem::take(&mut self.slows);
        for slow in slows {
            self.a.bind(slow.slow);
            if matches!(slow.kind, SlowKind::ExecAndLeave) {
                let insn = &self.insns[slow.insn_index as usize] as *const Insn as u64;
                self.a.mov_rr(ARGS[0], CPU, true);
                self.a.mov_imm(ARGS[1], insn);
                self.rip_to(ARGS[2], slow.rip);
                self.call(self.addrs.exec);
                let done = self.a.label();
                self.a.test_rr(ACC, ACC, true);
                self.a.jcc(E, done);
                self.fail_exit(slow.insn_index);
                self.a.bind(done);
                self.exit_ok(slow.insn_index + 1);
                continue;
            }
            if matches!(slow.kind, SlowKind::Exec) {
                // Not `helper`: that pays for the instructions before this one at translation time,
                // which a path that is only sometimes taken cannot do. `exit_fail` works out the
                // retired count from the index, as for a faulting memory access.
                let insn = &self.insns[slow.insn_index as usize] as *const Insn as u64;
                self.a.mov_rr(ARGS[0], CPU, true);
                self.a.mov_imm(ARGS[1], insn);
                self.rip_to(ARGS[2], slow.rip);
                self.call(self.addrs.exec);
                self.a.test_rr(ACC, ACC, true);
                self.a.jcc(E, slow.resume);
                self.fail_exit(slow.insn_index);
                continue;
            }
            // `OPERAND` is an argument register on both ABIs, so a preserving store parks it
            // before any argument is set up.
            if let SlowKind::St { preserve: true, .. } = slow.kind {
                self.a.mov_rr(SAVED_RESULT, ACC, true);
                self.a.mov_rr(SAVED_OPERAND, OPERAND, true);
            }
            self.a.mov_rr(SAVED_ADDR, ADDR, true);
            self.set_rip(slow.rip);
            // Where in the block the access is, for the device clock (`Cpu::slow_access_insn`,
            // index + 1); the helper clears it.
            self.a.mov_imm(SCRATCH_D, slow.insn_index as u64 + 1);
            self.a.st(mem(CPU, SLOW_ACCESS_INSN), SCRATCH_D, 8);
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_rr(ARGS[1], SAVED_ADDR, true);
            let fail = self.a.label();
            match slow.kind {
                SlowKind::Exec | SlowKind::ExecAndLeave => unreachable!("handled above"),
                SlowKind::Ld {
                    dst,
                    size,
                    signed,
                    stack,
                } => {
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.lea(ARGS[3], mem(RSP, OUT));
                    self.call(if stack {
                        self.addrs.ld_stack
                    } else {
                        self.addrs.ld
                    });
                    self.a.test_rr(ACC, ACC, true);
                    self.a.jcc(NE, fail);
                    self.a.ld_ext(dst, mem(RSP, OUT), size, signed);
                }
                SlowKind::St {
                    size,
                    stack,
                    preserve,
                } => {
                    self.a.mov_rr(SAVED_VALUE, VALUE, true);
                    self.a.mov_imm(ARGS[2], size as u64);
                    self.a.mov_rr(ARGS[3], SAVED_VALUE, true);
                    self.call(if stack {
                        self.addrs.st_stack
                    } else {
                        self.addrs.st
                    });
                    self.a.test_rr(ACC, ACC, true);
                    self.a.jcc(NE, fail);
                    if preserve {
                        self.a.mov_rr(ACC, SAVED_RESULT, true);
                        self.a.mov_rr(OPERAND, SAVED_OPERAND, true);
                    }
                }
            }
            self.a.mov_rr(ADDR, SAVED_ADDR, true);
            self.a.jmp(slow.resume);
            self.a.bind(fail);
            self.fail_exit(slow.insn_index);
        }
        self.a.finish();
    }

    /// Record the flags of the instruction just emitted: result in `ACC`, second operand in
    /// `OPERAND` (the carry-in for `inc`/`dec`, CF and OF packed for shifts and `imul`).
    /// `lazy_op` and `lazy_size` are adjacent bytes, written with one 16-bit store. For logic ops
    /// the interpreter stores 0 in `lazy_src` while this stores the operand; that is equivalent
    /// because `flags::compute` derives a Logic result from the result alone, and `inc`/`dec`
    /// take their carry-in from `lazy_src`, which they set themselves (`diff_flags` checks every
    /// consumer). Does not touch host flags, and does not claim they are live: see
    /// `store_lazy_op`.
    fn store_lazy(&mut self, op: LazyFlagsOp, size: u8) {
        self.a.st_imm16(
            mem(CPU, LAZY_OP),
            (op as u32 | (size as u32) << LAZY_SIZE_SHIFT) as u16,
        );
        self.a.st(mem(CPU, LAZY_DST), ACC, 8);
        self.a.st(mem(CPU, LAZY_SRC), OPERAND, 8);
    }

    /// `store_lazy` for an instruction whose result the host flags also mirror. `stored` says the
    /// instruction wrote memory after computing its flags: the probe of that store clobbered
    /// them, so they are no longer live.
    fn store_lazy_op(&mut self, op: LazyFlagsOp, size: u8, stored: bool) {
        if self.skip_flag_record {
            // A later instruction of the block overwrites these flags before anything can read
            // them; the host flags (`live`) still serve a jcc that follows.
            self.flag_records_skipped += 1;
        } else {
            self.store_lazy(op, size);
            if matches!(op, LazyFlagsOp::Adc | LazyFlagsOp::Sbb) {
                self.a.st(mem(CPU, LAZY_SRC2), SAVED_CONDITION, 8);
            }
        }
        self.live = !stored;
    }

    /// An immediate as the operand of an operation of `size` bytes: truncated to the width,
    /// which only differs from the sign-extended decode for 8-byte operations.
    fn imm_for(size: u8, imm: i64) -> u64 {
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
        if size == 1 && !insn.rex_present && HIGH_BYTE_REGS.contains(&reg) {
            (reg - HIGH_BYTE_REG_BASE, true)
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

    /// Loads the low `size` bytes of guest register `reg` (or its bits 15:8) into `rd`,
    /// zero-extended. No host flag is touched.
    fn read_reg(&mut self, rd: u8, reg: u8, high: bool, size: u8) {
        self.a.ld_ext(rd, register_slot(reg, high), size, false);
    }

    /// Writes the low `size` bytes of `rs` to guest register `reg`: byte and word operands are
    /// merged into the register by a narrow store, wider ones replace all of it, so a 4-byte
    /// `rs` must already be zero-extended (a 32-bit write clears the upper half). No host flag
    /// is touched.
    fn write_reg(&mut self, reg: u8, high: bool, size: u8, rs: u8) {
        self.a.st(
            register_slot(reg, high),
            rs,
            if size >= 4 { 8 } else { size },
        );
    }

    /// Loads the `size`-byte operand `dst` names into `rd`, zero-extended.
    #[allow(clippy::too_many_arguments)]
    fn load_dst(
        &mut self,
        insn: &Insn,
        dst: Dst,
        rd: u8,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) {
        match dst {
            Dst::Reg(reg, high) => self.read_reg(rd, reg, high, size),
            Dst::Mem => {
                self.ea(insn, next);
                self.mem_load(rd, size, false, insn_index, rip);
            }
        }
    }

    /// Writes `ACC` to the operand `dst` names. A memory write returns true: the caller must
    /// follow with `smc_check`. `keep_operands` is for a read-modify-write whose flags are
    /// recorded from `ACC` and `OPERAND` afterwards. The address is the one `load_dst` computed;
    /// the store may fault, which is why flags are recorded only after it.
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
                self.write_reg(reg, high, size, ACC);
                false
            }
            Dst::Mem => {
                self.a.mov_rr(VALUE, ACC, true);
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
    /// is a group-1 number or `ALU_TEST`; `ALU_CMP` and `ALU_TEST` discard the result. Returns
    /// true if it stored to memory (the caller follows with `smc_check`).
    ///
    /// Operand order matters: the memory operand is loaded first because its slow path calls a
    /// helper that clobbers the working registers, and only then is the register or immediate
    /// operand loaded. The host op runs at the operand's own width, so the host flags describe
    /// exactly the narrow operation.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &mut self,
        insn: &Insn,
        alu_op: u8,
        dst: Dst,
        src: Src,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        // adc and sbb take the guest CF as their carry-in: taken from the host flags when they
        // are live, else from `helper_cond`, before the operands are loaded (which clobbers the
        // host flags). It waits in `SAVED_CONDITION` until the record of the flags is written.
        // That register is also where a read-modify-write store parks its result, so adc and
        // sbb only get here with a register destination (see `try_inline`).
        let carry_in = matches!(alu_op, ALU_ADC | ALU_SBB);
        debug_assert!(!carry_in || (size >= 4 && matches!(dst, Dst::Reg(..))));
        if carry_in {
            self.condition_to_register(CONDITION_BELOW as u8, SAVED_CONDITION);
        }
        match (dst, src) {
            (Dst::Mem, Src::Reg(reg, high)) => {
                self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
                self.read_reg(OPERAND, reg, high, size);
            }
            (Dst::Mem, Src::Imm(v)) => {
                self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
                self.a.mov_imm(OPERAND, v);
            }
            (Dst::Reg(reg, high), Src::Mem) => {
                self.ea(insn, next);
                self.mem_load(OPERAND, size, false, insn_index, rip);
                self.read_reg(ACC, reg, high, size);
            }
            (Dst::Reg(dst_reg, dst_high), Src::Reg(src_reg, src_high)) => {
                self.read_reg(ACC, dst_reg, dst_high, size);
                self.read_reg(OPERAND, src_reg, src_high, size);
            }
            (Dst::Reg(reg, high), Src::Imm(v)) => {
                self.read_reg(ACC, reg, high, size);
                self.a.mov_imm(OPERAND, v);
            }
            (Dst::Mem, Src::Mem) => unreachable!(),
        }
        // cmp and test run as sub and and on the scratch copy: the result feeds the lazy flags
        let (host_op, lazy_op) = match alu_op {
            ALU_ADD => (ADD, LazyFlagsOp::Add),
            ALU_ADC => (ADC, LazyFlagsOp::Adc),
            ALU_SBB => (SBB, LazyFlagsOp::Sbb),
            ALU_OR => (OR, LazyFlagsOp::Logic),
            ALU_AND | ALU_TEST => (AND, LazyFlagsOp::Logic),
            ALU_SUB | ALU_CMP => (SUB, LazyFlagsOp::Sub),
            _ => (XOR, LazyFlagsOp::Logic),
        };
        if carry_in {
            // Host CF = the carry-in; `bt` leaves the other flags for the instruction to set.
            self.a.bt_ri(SAVED_CONDITION, 0, false);
        }
        self.a.alu_rr_sized(host_op, ACC, OPERAND, size);
        let stored = alu_op != ALU_CMP
            && alu_op != ALU_TEST
            && self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy_op(lazy_op, size, stored);
        stored
    }

    /// Puts 1 or 0 into `rd` according to guest condition code `cc`: from the host flags when
    /// they are live, otherwise from `helper_cond` over the lazy flags (which leaves the host
    /// flags dead). `rd` must not be a register the helper call uses for arguments.
    fn condition_to_register(&mut self, cc: u8, rd: u8) {
        if self.live {
            // A `mov`, never an `xor`, so the flags survive until the setcc reads them.
            self.a.mov_imm(rd, 0);
            self.a.setcc(cc & 15, rd);
        } else {
            let done = self.a.label();
            let generic = self.a.label();
            if cc >> 1 != CONDITION_PARITY {
                self.rebuild_flags_from_sub(generic);
                self.a.mov_imm(rd, 0);
                self.a.setcc(cc & 15, rd);
                self.a.jmp(done);
            }
            self.a.bind(generic);
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_imm(ARGS[1], cc as u64);
            self.call(self.addrs.cond);
            self.a.mov_rr(rd, ACC, true);
            self.a.bind(done);
        }
    }

    /// `inc`/`dec` of a register or memory operand. x86 leaves CF alone, so the lazy flags carry
    /// the current CF in as their second operand and the host CF is re-established from it
    /// (`bt`) right before the instruction: after the operand's memory access, which clobbers
    /// the host flags, CF would otherwise be lost. The carry is taken from the host flags when
    /// they are live, else from `helper_cond`; either way before the operand load.
    #[allow(clippy::too_many_arguments)]
    fn inc_dec(
        &mut self,
        insn: &Insn,
        decrement: bool,
        dst: Dst,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        self.condition_to_register(CONDITION_BELOW as u8, SAVED_CONDITION);
        self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
        self.a.mov_rr(OPERAND, SAVED_CONDITION, true);
        self.a.bt_ri(OPERAND, 0, false);
        if decrement {
            self.a.dec_r(ACC, size);
        } else {
            self.a.inc_r(ACC, size);
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy_op(
            if decrement {
                LazyFlagsOp::Dec
            } else {
                LazyFlagsOp::Inc
            },
            size,
            stored,
        );
        stored
    }

    /// `neg`: flags are those of `0 - operand`, recorded as a subtraction whose second operand
    /// is the original value.
    fn neg(
        &mut self,
        insn: &Insn,
        dst: Dst,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
        self.a.mov_rr(OPERAND, ACC, true);
        self.a.neg_r(ACC, size);
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy_op(LazyFlagsOp::Sub, size, stored);
        stored
    }

    /// `not`: no flags change, and neither do the host flags unless the operand is in memory.
    fn not(
        &mut self,
        insn: &Insn,
        dst: Dst,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
        self.a.not_r(ACC, size);
        self.store_dst(dst, size, false, insn_index, rip)
    }

    /// Builds the `SzpCfOf` word for a shift that has just run on `ACC`, in `OPERAND`, plus
    /// `extra` (a bit above CF/OF, or 0). The host CF after a shift by a count that is not zero
    /// modulo the width is the last bit shifted out, which is what `Cpu::shift_rotate` computes
    /// too. The host OF is not used: it is only defined for a count of 1, while the interpreter
    /// computes it for every count: shl = MSB of the result xor CF, shr = MSB of the original,
    /// sar = 0. For shr the caller has put a copy of the operand from before the shift in
    /// `SCRATCH_D` (the overflow register).
    fn shift_flags_word(&mut self, kind: ShiftKind, is64: bool, top_bit: u8, extra: i32) {
        let (carry, overflow) = (SCRATCH_C, SCRATCH_D);
        // setcc leaves the rest of the register alone, so clear it with a zero-extending move.
        self.a.setcc(B, carry);
        self.a.ext_rr(carry, carry, 1, false);
        match kind {
            ShiftKind::Shl => {
                self.a.mov_rr(overflow, ACC, true);
                self.a.shift_ri(SHR, overflow, top_bit, is64);
                self.a.alu_rr(XOR, overflow, carry, false);
            }
            ShiftKind::Shr => self.a.shift_ri(SHR, overflow, top_bit, is64),
            ShiftKind::Sar => {
                self.a.lea(OPERAND, mem(carry, extra));
                return;
            }
        }
        self.a.lea(
            OPERAND,
            mem_idx(carry, overflow, SZP_CF_OF_SRC_OF_BIT as u8, extra),
        );
    }

    /// `shl`/`shr`/`sar` of an 8-, 16-, 32- or 64-bit operand by an immediate (`count` is Some:
    /// already masked and non-zero) or by CL. A narrow operand runs as a 32-bit shift so the host
    /// CF is still the last bit out: `shl` first moves the operand to the top of the register
    /// (and back afterwards), `sar` sign-extends it, `shr` needs nothing as loads zero-extend.
    /// Counts of 17..31 then give the interpreter's answer (0 or the sign, CF 0) by themselves. The flags are recorded as `LazyFlagsOp::SzpCfOf` with CF
    /// and OF worked out by `shift_flags_word`. A CL count of zero changes neither flags nor
    /// value, which is a branch around the flag work; the destination is still written, as the
    /// interpreter does (a 32-bit destination zero-extends, which the zero-extending load did).
    #[allow(clippy::too_many_arguments)]
    fn shift(
        &mut self,
        insn: &Insn,
        kind: ShiftKind,
        dst: Dst,
        count: Option<u32>,
        size: u8,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let is64 = size == 8;
        let host_op = match kind {
            ShiftKind::Shl => SHL,
            ShiftKind::Shr => SHR,
            ShiftKind::Sar => SAR,
        };
        let (carry, overflow) = (SCRATCH_C, SCRATCH_D);
        let narrow = size < 4;
        let operand_bits = 8 * size;
        let top_align = HOST_SHIFT_BITS - operand_bits.min(HOST_SHIFT_BITS);
        // The bit the overflow flag reads: the top of the (aligned) result for shl, the top of
        // the original operand for shr.
        let overflow_bit = match (narrow, kind) {
            (true, ShiftKind::Shl) => HOST_SHIFT_BITS - 1,
            _ => operand_bits - 1,
        };
        self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
        if narrow && kind == ShiftKind::Sar {
            self.a.ext_rr(ACC, ACC, size, true);
        }
        if narrow && kind == ShiftKind::Shl {
            self.a.shift_ri(SHL, ACC, top_align, false);
        }
        match count {
            Some(count) => {
                if kind == ShiftKind::Shr {
                    self.a.mov_rr(overflow, ACC, true);
                }
                self.a.shift_ri(host_op, ACC, count as u8, is64);
                self.shift_flags_word(kind, is64, overflow_bit, 0);
            }
            None => {
                self.a
                    .ld(OPERAND, mem(CPU, GPR + GPR_SIZE * GUEST_RCX as i32));
                self.a.mov_rr(carry, OPERAND, false);
                self.a.alu_ri(AND, carry, if is64 { 63 } else { 31 }, false);
                let zero_count = self.a.label();
                let merge = self.a.label();
                self.a.jcc(E, zero_count);
                if kind == ShiftKind::Shr {
                    self.a.mov_rr(overflow, ACC, true);
                }
                self.a.shift_cl(host_op, ACC, is64);
                self.shift_flags_word(kind, is64, overflow_bit, SHIFT_APPLIED);
                self.a.jmp(merge);
                self.a.bind(zero_count);
                self.a.alu_rr(XOR, OPERAND, OPERAND, false);
                self.a.bind(merge);
            }
        }
        if narrow && kind == ShiftKind::Shl {
            self.a.shift_ri(SHR, ACC, top_align, false);
        }
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        if count.is_none() {
            let unchanged = self.a.label();
            self.a.test_ri(OPERAND, SHIFT_APPLIED, false);
            self.a.jcc(E, unchanged);
            self.a.alu_ri(AND, OPERAND, CF_OF_MASK, false);
            self.store_lazy(LazyFlagsOp::SzpCfOf, size);
            self.a.bind(unchanged);
        } else {
            self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        }
        // Not mirrored by the host flags (a zero count leaves the old ones in place, which is
        // also the right state for the CL form, but the code above clobbered them).
        self.live = false;
        stored
    }

    /// Loads the r/m operand of a VEX BMI instruction into `ACC`, zero-extended. The memory
    /// operand comes first: its slow path calls a helper that clobbers every scratch register.
    fn bmi_source(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) {
        if insn.md == MODRM_MOD_REGISTER {
            self.read_reg(ACC, insn.rm, false, insn.opsize);
        } else {
            self.ea(insn, next);
            self.mem_load(ACC, insn.opsize, false, insn_index, rip);
        }
    }

    /// shlx/shrx/sarx: ModRM.reg = r/m shifted by the count in vvvv, modulo the width, which is
    /// what the host shift by CL does. No flags.
    fn bmi_shift(&mut self, insn: &Insn, host_op: u8, next: u64, insn_index: u32, rip: u64) {
        self.bmi_source(insn, next, insn_index, rip);
        self.ld_gpr(OPERAND, insn.vex_v);
        self.a.shift_cl(host_op, ACC, insn.opsize == 8);
        self.st_gpr(insn.reg, ACC);
        // The shift clobbered the host flags.
        self.live = false;
    }

    /// What `memory_rmw` does for `insn`, if it is one of the read-modify-writes it handles (at
    /// any prefix: the caller checks LOCK and the operand size).
    fn memory_rmw_op(insn: &Insn) -> Option<RmwOp> {
        let alu = |alu_op: u8| matches!(alu_op, ALU_ADD | ALU_OR | ALU_AND | ALU_SUB | ALU_XOR);
        match insn.op {
            // add/or/and/sub/xor r/m, r
            0x01 | 0x09 | 0x21 | 0x29 | 0x31 => Some(RmwOp::Alu(
                (insn.op >> 3) as u8,
                Self::reg_src(insn, insn.opsize),
            )),
            0x81 | 0x83 if alu(insn.reg & 7) => Some(RmwOp::Alu(
                insn.reg & 7,
                Src::Imm(Self::imm_for(insn.opsize, insn.imm)),
            )),
            0xff if insn.reg & 7 == 0 => Some(RmwOp::Inc),
            0xff if insn.reg & 7 == 1 => Some(RmwOp::Dec),
            0x1c1 => Some(RmwOp::Xadd),
            0x87 => Some(RmwOp::Xchg),
            0x1b1 => Some(RmwOp::Cmpxchg),
            BIT_TEST_IMMEDIATE if insn.reg & 7 > GROUP8_FIRST_BIT_OP => Some(RmwOp::BitTest(
                (insn.reg & 7) - GROUP8_FIRST_BIT_OP,
                insn.imm as u8 & (8 * insn.opsize - 1),
            )),
            _ => None,
        }
    }

    /// A read-modify-write of the 4- or 8-byte memory operand of `insn`, atomic: the host's own
    /// locked instruction on the host address the TLB gives (`lock xadd`, `lock cmpxchg`,
    /// `xchg`), or for the rest a `lock cmpxchg` loop that recomputes the new value until no
    /// other vCPU wrote the location in between. The loop is what leaves the new value in a
    /// register for the lazy flag record, which the host's `lock add` and friends would not. A
    /// misaligned operand (a split lock, which raises #DB under DEBUGCTL.BLD, as Linux sets it)
    /// and a TLB miss go to the interpreter, which does the locked access its own way. Returns
    /// true: it always stores.
    fn memory_rmw(&mut self, insn: &Insn, op: RmwOp, next: u64, insn_index: u32, rip: u64) -> bool {
        let size = insn.opsize;
        let is64 = size == 8;
        let host = mem(SCRATCH_B, 0);
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        if matches!(op, RmwOp::BitTest(..)) {
            // Only CF changes, in `Cpu::eflags` (see `bit_test`).
            self.materialize_flags();
        }
        if matches!(op, RmwOp::Inc | RmwOp::Dec) {
            // The CF they keep; it may call a helper, so it comes first.
            self.condition_to_register(CONDITION_BELOW as u8, SAVED_CONDITION);
        }
        self.ea(insn, next);
        self.a.test_ri(ADDR, size as i32 - 1, true);
        self.a.jcc(NE, slow);
        self.tlb_probe(size, TLB_TAG_WRITE, slow);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        self.a.lea(SCRATCH_B, mem_idx(ADDR, SCRATCH_B, 0, 0));
        match op {
            RmwOp::Alu(_, Src::Imm(immediate)) => self.a.mov_imm(OPERAND, immediate),
            RmwOp::Alu(_, Src::Reg(reg, high)) => self.read_reg(OPERAND, reg, high, size),
            RmwOp::Xadd | RmwOp::Xchg | RmwOp::Cmpxchg => {
                self.read_reg(OPERAND, insn.reg, false, size)
            }
            _ => {}
        }
        match op {
            RmwOp::Xchg => {
                self.a.xchg_mr(host, OPERAND, is64);
                self.write_reg(insn.reg, false, size, OPERAND);
            }
            RmwOp::Xadd => {
                // OPERAND becomes the old value; the sum is recomputed for the flag record,
                // whose second operand is the register's value.
                self.a.mov_rr(VALUE, OPERAND, true);
                self.a.lock_xadd_mr(host, OPERAND, is64);
                self.write_reg(insn.reg, false, size, OPERAND);
                self.a.mov_rr(ACC, OPERAND, true);
                self.a.alu_rr_sized(ADD, ACC, VALUE, size);
                self.a.mov_rr(OPERAND, VALUE, true);
                self.store_lazy_op(LazyFlagsOp::Add, size, true);
            }
            RmwOp::Cmpxchg => {
                // rAX ends up holding the old value whether or not it matched; the flags are
                // those of `accumulator - old`.
                let accumulator = VALUE;
                self.read_reg(ACC, GUEST_RAX, false, size);
                self.a.mov_rr(accumulator, ACC, true);
                self.a.lock_cmpxchg_mr(host, OPERAND, is64);
                let equal = self.a.label();
                self.a.jcc(E, equal);
                self.write_reg(GUEST_RAX, false, size, ACC);
                self.a.bind(equal);
                self.a.mov_rr(OPERAND, ACC, true);
                self.a.mov_rr(ACC, accumulator, true);
                self.a.alu_rr_sized(SUB, ACC, OPERAND, size);
                self.store_lazy_op(LazyFlagsOp::Sub, size, true);
            }
            RmwOp::Alu(..) | RmwOp::Inc | RmwOp::Dec | RmwOp::BitTest(..) => {
                // The old value into rAX, the new one into VALUE, then swapped in only if the
                // location still holds the old value.
                let retry = self.a.label();
                self.a.bind(retry);
                self.a.ld_ext(ACC, host, size, false);
                self.a.mov_rr(VALUE, ACC, true);
                match op {
                    RmwOp::Alu(alu_op, _) => self.a.alu_rr_sized(alu_op, VALUE, OPERAND, size),
                    RmwOp::Inc => self.a.alu_ri(ADD, VALUE, 1, is64),
                    RmwOp::Dec => self.a.alu_ri(SUB, VALUE, 1, is64),
                    RmwOp::BitTest(operation, bit) => self.a.bit_op_ri(operation, VALUE, bit, is64),
                    _ => unreachable!(),
                }
                self.a.lock_cmpxchg_mr(host, VALUE, is64);
                self.a.jcc(NE, retry);
                match op {
                    RmwOp::Alu(alu_op, _) => {
                        self.a.mov_rr(ACC, VALUE, true);
                        let lazy = match alu_op {
                            ALU_ADD => LazyFlagsOp::Add,
                            ALU_SUB => LazyFlagsOp::Sub,
                            _ => LazyFlagsOp::Logic,
                        };
                        self.store_lazy_op(lazy, size, true);
                    }
                    RmwOp::Inc | RmwOp::Dec => {
                        self.a.mov_rr(ACC, VALUE, true);
                        self.a.mov_rr(OPERAND, SAVED_CONDITION, true);
                        let lazy = if matches!(op, RmwOp::Inc) {
                            LazyFlagsOp::Inc
                        } else {
                            LazyFlagsOp::Dec
                        };
                        self.store_lazy_op(lazy, size, true);
                    }
                    RmwOp::BitTest(_, bit) => {
                        let (carry, flags) = (SCRATCH_C, SCRATCH_D);
                        self.a.bt_ri(ACC, bit, is64);
                        self.a.setcc(B, carry);
                        self.a.ext_rr(carry, carry, 1, false);
                        self.replace_carry(carry, flags);
                    }
                    _ => unreachable!(),
                }
            }
        }
        self.a.bind(resume);
        // The interpreter fallback joins here with the host flags clobbered.
        self.live = false;
        true
    }

    /// Replaces CF in `Cpu::eflags` (already materialized) with `carry`, 0 or 1. Uses `flags`.
    fn replace_carry(&mut self, carry: u8, flags: u8) {
        // CF is bit 0, so the 0/1 carry ORs straight in.
        const { assert!(CF == 1) };
        self.a.ld(flags, mem(CPU, EFLAGS));
        self.a.alu_ri(AND, flags, !(CF as i32), true);
        self.a.alu_rr(OR, flags, carry, true);
        self.a.st(mem(CPU, EFLAGS), flags, 8);
    }

    /// Branches to `slow` while an SVM guest runs (its host may intercept the instruction).
    fn exit_to_interpreter_in_svm_guest(&mut self, slow: Label) {
        self.a.ld(SCRATCH_A, mem(CPU, SVM_GUEST));
        self.a.test_rr(SCRATCH_A, SCRATCH_A, true);
        self.a.jcc(NE, slow);
    }

    /// pause: nothing with one vCPU; with more, `helper_pause` (which may ask the block to end,
    /// so it is followed by the self-modifying-code exit check). In an SVM guest the interpreter
    /// runs it: the intercept and its pause filter.
    fn pause(&mut self, insn_index: u32, next: u64, rip: u64) {
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        self.exit_to_interpreter_in_svm_guest(slow);
        self.a.cmp_m8_imm(mem(CPU, MULTI), 0);
        self.a.jcc(E, resume);
        self.a.mov_rr(ARGS[0], CPU, true);
        self.call(self.addrs.pause);
        self.a.bind(resume);
        self.smc_check(insn_index, next);
        self.live = false;
    }

    /// rdtsc: edx:eax = the TSC, from `helper_rdtsc` after paying the block's instructions so
    /// far (the virtual clock is the instruction count); rdtscp (`with_aux`) also loads ecx
    /// from IA32_TSC_AUX. CR4.TSD above CPL 0 (#GP) and an SVM guest go to the interpreter.
    fn rdtsc(&mut self, index: usize, with_aux: bool, insn_index: u32, rip: u64) {
        self.prepay(index);
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        let any_cpl = self.a.label();
        self.exit_to_interpreter_in_svm_guest(slow);
        self.a.ld(SCRATCH_A, mem(CPU, CR4));
        self.a.test_ri(SCRATCH_A, CR4_TSD as i32, true);
        self.a.jcc(E, any_cpl);
        self.a.cmp_m8_imm(mem(CPU, CPL), 0);
        self.a.jcc(NE, slow);
        self.a.bind(any_cpl);
        self.a.mov_rr(ARGS[0], CPU, true);
        self.call(self.addrs.rdtsc);
        self.a.mov_rr(ADDR, ACC, true);
        self.a.shift_ri(SHR, ADDR, 32, true);
        self.a.mov_rr(ACC, ACC, false);
        self.st_gpr(GUEST_RAX, ACC);
        self.st_gpr(GUEST_RDX, ADDR);
        if with_aux {
            self.a.ld_ext(VALUE, mem(CPU, TSC_AUX), 4, false);
            self.st_gpr(GUEST_RCX, VALUE);
        }
        self.a.bind(resume);
        self.live = false;
    }

    /// rdpkru: eax = PKRU, edx = 0. Without CR4.PKE (#UD) or with ecx not zero (#GP) the
    /// interpreter runs it.
    fn rdpkru(&mut self, insn_index: u32, rip: u64) {
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        self.a.ld(SCRATCH_A, mem(CPU, CR4));
        self.a.test_ri(SCRATCH_A, CR4_PKE as i32, true);
        self.a.jcc(E, slow);
        self.read_reg(SCRATCH_A, GUEST_RCX, false, 4);
        self.a.test_rr(SCRATCH_A, SCRATCH_A, false);
        self.a.jcc(NE, slow);
        self.a.ld_ext(VALUE, mem(CPU, PKRU), 4, false);
        self.st_gpr(GUEST_RAX, VALUE);
        self.a.mov_imm(VALUE, 0);
        self.st_gpr(GUEST_RDX, VALUE);
        self.a.bind(resume);
        self.live = false;
    }

    /// pushf: the flags with RF and VM clear, pushed as 8 bytes. The arithmetic flags are
    /// materialized first. An SVM guest goes to the interpreter (its host may intercept pushf).
    fn pushf(&mut self, insn_index: u32, rip: u64) -> bool {
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        self.exit_to_interpreter_in_svm_guest(slow);
        self.materialize_flags();
        self.ld_gpr(ADDR, GUEST_RSP);
        self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
        self.a.ld(VALUE, mem(CPU, EFLAGS));
        self.a.alu_ri(AND, VALUE, !(PUSHF_CLEARED as i32), true);
        self.stack_store(insn_index, rip);
        self.st_gpr(GUEST_RSP, ADDR);
        self.a.bind(resume);
        self.live = false;
        true
    }

    /// sti at CPL <= IOPL: IF set, and the one-instruction interrupt shadow opened when IF was
    /// clear. Above IOPL (#GP, or VME/PVI's virtual flag) the interpreter runs it.
    fn sti(&mut self, insn_index: u32, rip: u64) {
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        let (flags, iopl, cpl) = (SCRATCH_A, SCRATCH_B, SCRATCH_C);
        self.a.ld(flags, mem(CPU, EFLAGS));
        self.a.mov_rr(iopl, flags, true);
        self.a.shift_ri(SHR, iopl, IOPL_SHIFT, true);
        self.a.alu_ri(AND, iopl, IOPL_MASK, true);
        self.a.ld_ext(cpl, mem(CPU, CPL), 1, false);
        self.a.alu_rr(CMP, cpl, iopl, true);
        self.a.jcc(CONDITION_ABOVE, slow);
        self.a.test_ri(flags, IF as i32, true);
        self.a.jcc(NE, resume);
        self.a.mov_imm(VALUE, 1);
        self.a.st(mem(CPU, INTERRUPT_SHADOW), VALUE, 1);
        self.a.alu_ri(OR, flags, IF as i32, true);
        self.a.st(mem(CPU, EFLAGS), flags, 8);
        self.a.bind(resume);
        self.live = false;
    }

    /// kortest (`or`) and ktest k1, k2 at 8, 16, 32 or 64 bits (66 and VEX.W pick the width):
    /// ZF and CF from the opmask registers, the other arithmetic flags clear, written straight
    /// into `Cpu::eflags` with `lazy_op` set to `Eflags`. Behind the AVX-512 state gate.
    fn opmask_test(&mut self, insn: &Insn, or: bool, insn_index: u32, rip: u64) {
        let bits = match (insn.p66, insn.opsize == 8) {
            (true, false) => 8,
            (false, false) => 16,
            (true, true) => 32,
            (false, true) => 64,
        };
        let full = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        // The gate's failure path is the only use of the fallback entry.
        let (_, resume) = self.evex_entry(VectorGate::Avx512, insn_index, rip);
        let (first, second, zero, carry, flags) = (ACC, OPERAND, SCRATCH_C, SCRATCH_D, SCRATCH_A);
        let opmask = |reg: u8| {
            mem(
                CPU,
                OPMASK + OPMASK_SIZE * (reg & OPMASK_REGISTER_MASK) as i32,
            )
        };
        self.a.ld(first, opmask(insn.reg));
        self.a.ld(second, opmask(insn.rm));
        self.a.mov_imm(VALUE, full);
        if bits < 64 {
            self.a.alu_rr(AND, first, VALUE, true);
            self.a.alu_rr(AND, second, VALUE, true);
        }
        if or {
            // ZF: k1 | k2 is zero; CF: it is all ones
            self.a.alu_rr(OR, first, second, true);
            self.a.setcc(E, zero);
            self.a.alu_rr(CMP, first, VALUE, true);
            self.a.setcc(E, carry);
        } else {
            // ZF: k1 & k2 is zero; CF: !k1 & k2 is
            self.a.mov_rr(SCRATCH_B, first, true);
            self.a.alu_rr(AND, SCRATCH_B, second, true);
            self.a.setcc(E, zero);
            self.a.not_r(first, 8);
            self.a.alu_rr(AND, first, second, true);
            self.a.setcc(E, carry);
        }
        self.a.ext_rr(zero, zero, 1, false);
        self.a.ext_rr(carry, carry, 1, false);
        self.a.shift_ri(SHL, zero, ZF.trailing_zeros() as u8, false);
        self.a.ld(flags, mem(CPU, EFLAGS));
        self.a.alu_ri(AND, flags, !(ARITH as i32), true);
        self.a.alu_rr(OR, flags, zero, true);
        // CF is bit 0.
        const { assert!(CF == 1) };
        self.a.alu_rr(OR, flags, carry, true);
        self.a.st(mem(CPU, EFLAGS), flags, 8);
        self.a.mov_imm(VALUE, LazyFlagsOp::Eflags as u64);
        self.a.st(mem(CPU, LAZY_OP), VALUE, 1);
        self.a.bind(resume);
        self.live = false;
    }

    /// cmpxchg16b: rdx:rax against the 16 aligned bytes; equal, they become rcx:rbx and ZF is
    /// set, else rdx:rax takes them and ZF is clear (the only flag that changes, so the flags
    /// are materialized first). The host's own `lock cmpxchg16b` on the host address; a
    /// misaligned operand (#GP) and a TLB miss go to the interpreter. The host instruction
    /// takes the new low half in rbx, which is `CPU`: the pointer waits in `SCRATCH_D` meanwhile,
    /// and nothing between the two moves may use `CPU`.
    fn cmpxchg16b(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) -> bool {
        const _: () = assert!(CPU == RBX);
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        let (host, zero, flags) = (SCRATCH_B, SCRATCH_C, VALUE);
        self.materialize_flags();
        self.ea(insn, next);
        self.a.test_ri(ADDR, CMPXCHG16B_BYTES as i32 - 1, true);
        self.a.jcc(NE, slow);
        self.tlb_probe(CMPXCHG16B_BYTES, TLB_TAG_WRITE, slow);
        self.a.ld(host, mem(SCRATCH_A, TLB_ADDEND));
        self.a.lea(host, mem_idx(ADDR, host, 0, 0));
        self.ld_gpr(RAX, GUEST_RAX);
        self.ld_gpr(RDX, GUEST_RDX);
        self.ld_gpr(RCX, GUEST_RCX);
        self.a.mov_rr(SCRATCH_D, CPU, true);
        self.a
            .ld(RBX, mem(SCRATCH_D, GPR + GPR_SIZE * GUEST_RBX as i32));
        self.a.lock_cmpxchg16b(mem(host, 0));
        self.a.mov_rr(CPU, SCRATCH_D, true);
        self.a.setcc(E, zero);
        let equal = self.a.label();
        self.a.jcc(E, equal);
        self.st_gpr(GUEST_RAX, RAX);
        self.st_gpr(GUEST_RDX, RDX);
        self.a.bind(equal);
        self.a.ext_rr(zero, zero, 1, false);
        self.a.shift_ri(SHL, zero, ZF.trailing_zeros() as u8, false);
        self.a.ld(flags, mem(CPU, EFLAGS));
        self.a.alu_ri(AND, flags, !(ZF as i32), true);
        self.a.alu_rr(OR, flags, zero, true);
        self.a.st(mem(CPU, EFLAGS), flags, 8);
        self.a.bind(resume);
        self.live = false;
        true
    }

    /// Registers an out-of-line fallback that runs the instruction in the interpreter and comes
    /// back to `resume` (bind it where the inline code ends). Nothing of the instruction may be
    /// committed before a branch to the entry: the interpreter starts it over.
    fn exec_fallback(&mut self, insn_index: u32, rip: u64) -> (Label, Label) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Exec,
            insn_index,
            rip,
        });
        (slow, resume)
    }

    /// div and idiv r/m at 32 and 64 bits: rdx:rax (edx:eax) by r/m, quotient to rax and
    /// remainder to rdx; the guest flags stay as they are (the interpreter leaves them alone,
    /// which the architecture allows: they are undefined). The host's own division does the
    /// work, but it must never raise #DE in the host: a zero divisor, a quotient that does not
    /// fit, and a 64-bit dividend that needs 128 bits (rdx not zero for div, not the sign of rax
    /// for idiv; compiled code puts `xor edx, edx` or `cqo` first) all go to the interpreter,
    /// which raises the guest's #DE or divides the rare wide dividend. A 32-bit division runs as
    /// a 64-bit one of the exact edx:eax, whose quotient is then checked against 32 bits.
    fn divide(&mut self, insn: &Insn, signed: bool, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let (divisor, scratch) = (OPERAND, SCRATCH_C);
        // The memory operand first: its slow path clobbers every scratch register.
        self.bmi_source(insn, next, insn_index, rip);
        self.a.mov_rr(divisor, ACC, true);
        let (slow, resume) = self.exec_fallback(insn_index, rip);
        self.ld_gpr(ACC, GUEST_RAX);
        self.ld_gpr(ADDR, GUEST_RDX);
        if !is64 {
            // rax = the 64-bit dividend edx:eax.
            self.a.mov_rr(scratch, ADDR, true);
            self.a.shift_ri(SHL, scratch, 32, true);
            self.a.mov_rr(ACC, ACC, false);
            self.a.alu_rr(OR, ACC, scratch, true);
            if signed {
                self.a.ext_rr(divisor, divisor, 4, true);
            }
        }
        self.a.test_rr(divisor, divisor, true);
        self.a.jcc(E, slow);
        if signed {
            if is64 {
                self.a.mov_rr(scratch, ACC, true);
                self.a.shift_ri(SAR, scratch, 63, true);
                self.a.alu_rr(CMP, scratch, ADDR, true);
                self.a.jcc(NE, slow);
            }
            // The most negative dividend by -1 is the one 64-bit quotient that overflows (a
            // 32-bit edx:eax can be that dividend too).
            let fits = self.a.label();
            self.a.alu_ri(CMP, divisor, -1, true);
            self.a.jcc(NE, fits);
            self.a.mov_imm(scratch, i64::MIN as u64);
            self.a.alu_rr(CMP, ACC, scratch, true);
            self.a.jcc(E, slow);
            self.a.bind(fits);
            self.a.cdq(true);
            self.a.idiv_r(divisor, true);
            if !is64 {
                self.a.ext_rr(scratch, ACC, 4, true);
                self.a.alu_rr(CMP, scratch, ACC, true);
                self.a.jcc(NE, slow);
            }
        } else {
            if is64 {
                self.a.test_rr(ADDR, ADDR, true);
                self.a.jcc(NE, slow);
            } else {
                self.a.mov_imm(ADDR, 0);
            }
            self.a.div_r(divisor, true);
            if !is64 {
                self.a.mov_rr(scratch, ACC, true);
                self.a.shift_ri(SHR, scratch, 32, true);
                self.a.test_rr(scratch, scratch, true);
                self.a.jcc(NE, slow);
            }
        }
        if !is64 {
            // Both fit 32 bits (the remainder is smaller than the divisor), so this only clears
            // the upper halves, as a 32-bit register write does.
            self.a.mov_rr(ACC, ACC, false);
            self.a.mov_rr(ADDR, ADDR, false);
        }
        self.st_gpr(GUEST_RAX, ACC);
        self.st_gpr(GUEST_RDX, ADDR);
        self.a.bind(resume);
        self.live = false;
    }

    /// bt, bts, btr and btc (`operation`, `BIT_TEST`..`BIT_COMPLEMENT`) at 32 and 64 bits: CF
    /// takes the selected bit, which the last three then set, clear or flip; every other flag
    /// stays (the interpreter keeps them; the architecture leaves OF/SF/AF/PF undefined and ZF
    /// alone), so the flags are materialized first and CF is replaced in `Cpu::eflags` once the
    /// store has succeeded. With a register offset into memory the operand is the operand-size
    /// word at the address plus the sign-extended offset divided by the width, rounded down, as
    /// for a bit string. The host instruction then works on a register copy, which takes the
    /// offset modulo the width just as the guest's register form does.
    fn bit_test(
        &mut self,
        insn: &Insn,
        operation: u8,
        immediate_offset: Option<u8>,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let size = insn.opsize;
        let is64 = size == 8;
        let bits = 8 * size as u32;
        let (carry, flags) = (SCRATCH_C, SCRATCH_D);
        self.materialize_flags();
        let memory = insn.md != MODRM_MOD_REGISTER;
        if memory {
            self.ea(insn, next);
            if immediate_offset.is_none() {
                // The offset waits in a callee-saved register: the load's slow path clobbers
                // the rest.
                self.read_reg(SAVED_OPERAND, insn.reg, false, size);
                if !is64 {
                    self.a.ext_rr(SAVED_OPERAND, SAVED_OPERAND, 4, true);
                }
                self.a.mov_rr(SCRATCH_A, SAVED_OPERAND, true);
                self.a
                    .shift_ri(SAR, SCRATCH_A, bits.trailing_zeros() as u8, true);
                self.a.lea(
                    ADDR,
                    mem_idx(ADDR, SCRATCH_A, size.trailing_zeros() as u8, 0),
                );
            }
            self.mem_load(ACC, size, false, insn_index, rip);
            match immediate_offset {
                Some(offset) => self.a.mov_imm(OPERAND, offset as u64),
                None => self.a.mov_rr(OPERAND, SAVED_OPERAND, true),
            }
        } else {
            self.read_reg(ACC, insn.rm, false, size);
            match immediate_offset {
                Some(offset) => self.a.mov_imm(OPERAND, offset as u64),
                None => self.read_reg(OPERAND, insn.reg, false, size),
            }
        }
        self.a.bit_op_rr(operation, ACC, OPERAND, is64);
        self.a.setcc(B, carry);
        // The carry rides in `OPERAND`, which a preserving slow-path store gives back.
        self.a.ext_rr(OPERAND, carry, 1, false);
        let mut stored = false;
        if operation != BIT_TEST {
            if memory {
                stored = self.store_dst(Dst::Mem, size, true, insn_index, rip);
            } else {
                self.write_reg(insn.rm, false, size, ACC);
            }
        }
        self.replace_carry(OPERAND, flags);
        self.live = false;
        stored
    }

    /// mulx: vvvv and ModRM.reg get the low and high halves of rdx times r/m, unsigned, guest
    /// flags untouched (the lazy record stays valid; only the host flags are lost). The host
    /// `mul` leaves the halves in rax and rdx (`ADDR`, free once the operand is loaded). The high
    /// half is stored last, so it wins when both name the same register, as on hardware.
    fn mulx(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        self.bmi_source(insn, next, insn_index, rip);
        self.a.mov_rr(OPERAND, ACC, true);
        self.ld_gpr(ACC, GUEST_RDX);
        self.a.mul_r(OPERAND, is64);
        self.st_gpr(insn.vex_v, ACC);
        self.st_gpr(insn.reg, ADDR);
        self.live = false;
    }

    /// Makes `Cpu::eflags` hold the arithmetic flags (`helper_materialize_flags`), unless they
    /// already do (`lazy_op` is `Eflags`, 0, as after the previous adcx). Clobbers every
    /// caller-saved register and the host flags.
    fn materialize_flags(&mut self) {
        let done = self.a.label();
        self.a
            .cmp_m8_imm(mem(CPU, LAZY_OP), LazyFlagsOp::Eflags as u8);
        self.a.jcc(E, done);
        self.a.mov_rr(ARGS[0], CPU, true);
        self.call(self.addrs.materialize_flags);
        self.a.bind(done);
        self.live = false;
    }

    /// adcx (`overflow` false) and adox: ModRM.reg += r/m + CF (OF), and only that flag takes
    /// the carry out, which the host `adc` leaves in its CF either way. With the flags in
    /// `Cpu::eflags` the carry is a bit there; a chain of these (bignum code interleaves the
    /// two) pays for materializing the flags once.
    fn add_carry_chain(
        &mut self,
        insn: &Insn,
        overflow: bool,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) {
        let size = if insn.opsize == 8 { 8 } else { 4 };
        let flag = if overflow { OF } else { CF };
        let bit = flag.trailing_zeros() as u8;
        let (flags, carry_out) = (SCRATCH_C, SCRATCH_D);
        self.materialize_flags();
        if insn.md == MODRM_MOD_REGISTER {
            self.read_reg(OPERAND, insn.rm, false, size);
        } else {
            self.ea(insn, next);
            self.mem_load(OPERAND, size, false, insn_index, rip);
        }
        self.read_reg(ACC, insn.reg, false, size);
        self.a.ld(flags, mem(CPU, EFLAGS));
        self.a.bt_ri(flags, bit, true);
        self.a.alu_rr_sized(ADC, ACC, OPERAND, size);
        self.a.setcc(B, carry_out);
        self.a.ext_rr(carry_out, carry_out, 1, false);
        self.a.shift_ri(SHL, carry_out, bit, false);
        self.a.alu_ri(AND, flags, !(flag as i32), true);
        self.a.alu_rr(OR, flags, carry_out, true);
        self.a.st(mem(CPU, EFLAGS), flags, 8);
        // A 32-bit `adc` already zero-extended the sum.
        self.write_reg(insn.reg, false, size, ACC);
        self.live = false;
    }

    /// movd/movq xmm, r/m (`to_vector`: the low 32 or 64 bits, the rest of the 128 cleared and
    /// anything above left alone, as legacy SSE does) or r/m, xmm. A memory operand is one
    /// naturally sized access through the TLB probe; a miss (and the SSE gate failing) runs the
    /// instruction in the interpreter. Returns whether it stores to memory.
    fn movd(&mut self, insn: &Insn, to_vector: bool, next: u64, insn_index: u32, rip: u64) -> bool {
        let size: u8 = if insn.opsize == 8 { 8 } else { 4 };
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = XMM + XMM_SIZE * insn.reg as i32;
        let memory = insn.md != MODRM_MOD_REGISTER;
        if memory {
            self.ea(insn, next);
            let tag = if to_vector {
                TLB_TAG_READ
            } else {
                TLB_TAG_WRITE
            };
            self.tlb_probe(size, tag, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        }
        if to_vector {
            if memory {
                self.a
                    .ld_ext(VALUE, mem_idx(ADDR, SCRATCH_B, 0, 0), size, false);
            } else {
                self.read_reg(VALUE, insn.rm, false, size);
            }
            self.a.st(mem(CPU, xmm), VALUE, 8);
            self.a.mov_imm(VALUE, 0);
            self.a.st(mem(CPU, xmm + XMM_HALF), VALUE, 8);
        } else {
            self.a.ld_ext(VALUE, mem(CPU, xmm), size, false);
            if memory {
                self.a.st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, size);
            } else {
                self.write_reg(insn.rm, false, size, VALUE);
            }
        }
        self.a.bind(resume);
        self.live = false;
        memory && !to_vector
    }

    /// bzhi: ModRM.reg = r/m with the bits from index (vvvv bits 7:0) up cleared; an index at or
    /// above the width leaves it whole and sets CF. SF/ZF/PF follow the result, OF and AF clear:
    /// `LazyFlagsOp::SzpCfOf` with CF in the packed word.
    fn bzhi(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let (mask, all_ones) = (SCRATCH_C, SCRATCH_D);
        self.bmi_source(insn, next, insn_index, rip);
        self.ld_gpr(OPERAND, insn.vex_v);
        self.a.ext_rr(OPERAND, OPERAND, 1, false);
        // mask = (1 << index) - 1, or all ones when the index is out of range: the host shifts
        // only by index modulo 64 there, which the compare discards.
        self.a.mov_imm(mask, 1);
        self.a.shift_cl(SHL, mask, true);
        self.a.alu_ri(SUB, mask, 1, true);
        self.a.mov_imm(all_ones, u64::MAX);
        self.a.alu_ri(CMP, OPERAND, 8 * insn.opsize as i32, false);
        self.a.cmov_rr(AE, mask, all_ones, true);
        // CF, taken from the compare before `and` replaces the host flags.
        self.a.setcc(AE, OPERAND);
        self.a.ext_rr(OPERAND, OPERAND, 1, false);
        self.a.alu_rr(AND, ACC, mask, is64);
        self.st_gpr(insn.reg, ACC);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = false;
    }

    /// andn: ModRM.reg = r/m & !vvvv, flags from the result with CF and OF clear.
    fn andn(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) {
        self.bmi_source(insn, next, insn_index, rip);
        self.ld_gpr(OPERAND, insn.vex_v);
        self.a.not_r(OPERAND, 8);
        self.a.alu_rr(AND, ACC, OPERAND, insn.opsize == 8);
        self.st_gpr(insn.reg, ACC);
        self.a.mov_imm(OPERAND, 0);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = false;
    }

    /// blsr, blsmsk and blsi (`group` 1, 2 and 3): vvvv = r/m with its lowest set bit cleared,
    /// the mask up to and including it, or just it. Flags from the result; CF is set when the
    /// source is zero (blsr, blsmsk) or not zero (blsi).
    fn lowest_set_bit(&mut self, insn: &Insn, group: u8, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let scratch = SCRATCH_C;
        self.bmi_source(insn, next, insn_index, rip);
        self.a.test_rr(ACC, ACC, is64);
        self.a
            .setcc(if group == BMI_GROUP_BLSI { NE } else { E }, OPERAND);
        self.a.ext_rr(OPERAND, OPERAND, 1, false);
        if group == BMI_GROUP_BLSI {
            self.a.mov_rr(scratch, ACC, true);
            self.a.neg_r(scratch, insn.opsize);
        } else {
            self.a.lea(scratch, mem(ACC, -1));
        }
        self.a.alu_rr(
            if group == BMI_GROUP_BLSMSK { XOR } else { AND },
            ACC,
            scratch,
            is64,
        );
        self.st_gpr(insn.vex_v, ACC);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = false;
    }

    /// bsf, bsr, tzcnt and lzcnt of a 32- or 64-bit operand, the same results and
    /// `LazyFlagsOp::SzpCfOf` records as the aarch64 `bit_scan`. The host `bsf`/`bsr` stand in
    /// for tzcnt/lzcnt too, so no host BMI is needed: a zero source (ZF) selects the width.
    fn bit_scan(&mut self, insn: &Insn, scan: BitScan, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        let bits = 8 * insn.opsize as u64;
        let (index, other) = (SCRATCH_C, SCRATCH_D);
        self.bmi_source(insn, next, insn_index, rip);
        match scan {
            BitScan::Tzcnt => {
                self.a.bsf_rr(index, ACC, is64);
                // CF = source is zero, read before the host flags change; `mov_imm` of a
                // non-zero constant leaves them alone, and so does `cmov`.
                self.a.mov_imm(other, bits);
                self.a.cmov_rr(E, index, other, true);
                self.a.setcc(E, OPERAND);
                self.a.ext_rr(OPERAND, OPERAND, 1, false);
                self.a.mov_rr(ACC, index, true);
            }
            BitScan::Lzcnt => {
                self.a.bsr_rr(index, ACC, is64);
                // For a zero source use 2 * bits - 1, which xor (bits - 1) turns into `bits`; for
                // any other the xor turns the bit index into the leading-zero count.
                self.a.mov_imm(other, 2 * bits - 1);
                self.a.cmov_rr(E, index, other, true);
                self.a.setcc(E, OPERAND);
                self.a.ext_rr(OPERAND, OPERAND, 1, false);
                self.a.alu_ri(XOR, index, bits as i32 - 1, true);
                self.a.mov_rr(ACC, index, true);
            }
            BitScan::Bsf | BitScan::Bsr => {
                if scan == BitScan::Bsf {
                    self.a.bsf_rr(index, ACC, is64);
                } else {
                    self.a.bsr_rr(index, ACC, is64);
                }
                // A zero source leaves the whole destination register as it was.
                self.ld_gpr(other, insn.reg);
                self.a.cmov_rr(E, index, other, true);
                self.st_gpr(insn.reg, index);
                // SzpCfOf of the source (still in ACC) with CF = OF = 0
                self.a.mov_imm(OPERAND, 0);
                self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
                self.live = false;
                return;
            }
        }
        self.st_gpr(insn.reg, ACC);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = false;
    }

    /// `shld`/`shrd r/m, r, imm8` at 32 and 64 bits with a count of 1 to the width minus 1, as on
    /// aarch64. The host instruction gives the value and CF; OF (defined by the host for a count
    /// of 1 only) is the sign change, worked out from the original operand.
    #[allow(clippy::too_many_arguments)]
    fn double_shift(
        &mut self,
        insn: &Insn,
        left: bool,
        dst: Dst,
        count: u32,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let size = insn.opsize;
        let is64 = size == 8;
        let (source, original) = (SCRATCH_C, SCRATCH_D);
        self.load_dst(insn, dst, ACC, size, next, insn_index, rip);
        self.read_reg(source, insn.reg, false, size);
        self.a.mov_rr(original, ACC, true);
        self.a.double_shift_ri(left, ACC, source, count as u8, is64);
        self.a.setcc(B, OPERAND);
        self.a.ext_rr(OPERAND, OPERAND, 1, false);
        self.a.alu_rr(XOR, original, ACC, true);
        self.a.shift_ri(SHR, original, 8 * size - 1, true);
        self.a.alu_ri(AND, original, 1, false);
        self.a
            .shift_ri(SHL, original, SZP_CF_OF_SRC_OF_BIT as u8, false);
        self.a.alu_rr(OR, OPERAND, original, false);
        let stored = self.store_dst(dst, size, true, insn_index, rip);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        self.live = false;
        stored
    }

    /// One-operand `mul` and `imul` (`group3` /4 and /5) at 32 and 64 bits, as on aarch64: the
    /// host instruction produces rDX:rAX and the right CF = OF, so only the flag record and the
    /// two register stores are ours. The memory operand goes first: its slow path clobbers
    /// everything, and afterwards `ADDR` (rdx) is free for the high half.
    fn mul_wide(&mut self, insn: &Insn, signed: bool, next: u64, insn_index: u32, rip: u64) {
        let is64 = insn.opsize == 8;
        self.bmi_source(insn, next, insn_index, rip);
        self.a.mov_rr(OPERAND, ACC, true);
        self.ld_gpr(ACC, GUEST_RAX);
        if signed {
            self.a.imul_r(OPERAND, is64);
        } else {
            self.a.mul_r(OPERAND, is64);
        }
        let overflow = SCRATCH_C;
        self.a.setcc(O, overflow);
        self.a.ext_rr(overflow, overflow, 1, false);
        self.a.mov_imm(OPERAND, IMUL_OVERFLOW_FLAGS as u64);
        self.a.imul_rr(OPERAND, overflow, false);
        self.st_gpr(GUEST_RAX, ACC);
        self.st_gpr(GUEST_RDX, ADDR);
        self.store_lazy(LazyFlagsOp::SzpCfOf, insn.opsize);
        self.live = false;
    }

    /// `imul r, r/m` and the three-operand `imul r, r/m, imm`, at 32 and 64 bits. The host
    /// `imul` sets CF = OF when the full product does not fit, which is what the interpreter
    /// computes; SF/ZF/PF are undefined on the host, so they come from the truncated result
    /// (`LazyFlagsOp::SzpCfOf`).
    fn imul(&mut self, insn: &Insn, size: u8, next: u64, insn_index: u32, rip: u64) {
        // `0f af` multiplies the destination register (ACC) by r/m (OPERAND); the three-operand
        // forms multiply r/m (ACC) by the immediate (OPERAND). The memory source goes first so
        // its slow path cannot clobber the other operand.
        let source = if insn.op == 0x1af { OPERAND } else { ACC };
        match Self::rm_src(insn, size) {
            Src::Reg(reg, _) => self.read_reg(source, reg, false, size),
            _ => {
                self.ea(insn, next);
                self.mem_load(source, size, false, insn_index, rip);
            }
        }
        if insn.op == 0x1af {
            self.read_reg(ACC, insn.reg, false, size);
        } else {
            self.a.mov_imm(OPERAND, Self::imm_for(size, insn.imm));
        }
        self.a.imul_rr(ACC, OPERAND, size == 8);
        // CF = OF = (the product does not fit): OPERAND = 3 or 0 for `store_lazy`.
        let overflow = SCRATCH_C;
        self.a.setcc(O, overflow);
        self.a.ext_rr(overflow, overflow, 1, false);
        self.a.mov_imm(OPERAND, IMUL_OVERFLOW_FLAGS as u64);
        self.a.imul_rr(OPERAND, overflow, false);
        self.st_gpr(insn.reg, ACC);
        self.store_lazy(LazyFlagsOp::SzpCfOf, size);
        self.live = false;
    }

    /// With no live host flags, sets them from the record the previous flag writer left in `Cpu`
    /// when it was a `sub` or `cmp`: `lazy_dst` is the result and `lazy_src` the subtrahend, so
    /// `(dst + src) << s` and `src << s`, with `s` aligning the operand width to 64 bits, compare
    /// exactly as the guest did. Any other record jumps to `generic`. A conditional branch that
    /// starts a block (the second of two on one compare) would otherwise call `helper_cond`.
    /// Only SF, ZF, CF and OF come out right: the operands are shifted up, so the low byte the
    /// parity flag reads is always zero, which is why the parity conditions use the helper
    /// (`CONDITION_PARITY`). Clobbers `ACC`, `OPERAND`, `SCRATCH_C` and `SCRATCH_D`.
    fn rebuild_flags_from_sub(&mut self, generic: Label) {
        let (difference, subtrahend) = (SCRATCH_C, SCRATCH_D);
        self.a.ld_ext(ACC, mem(CPU, LAZY_OP), 2, false);
        self.a.mov_rr(OPERAND, ACC, false);
        self.a.alu_ri(AND, OPERAND, 0xff, false);
        self.a.alu_ri(CMP, OPERAND, LazyFlagsOp::Sub as i32, false);
        self.a.jcc(NE, generic);
        self.a.shift_ri(SHR, ACC, LAZY_SIZE_SHIFT as u8, false);
        // The count the shifts below take from cl: 64 - 8 * size
        self.a.mov_imm(OPERAND, 8);
        self.a.alu_rr(SUB, OPERAND, ACC, false);
        self.a.shift_ri(SHL, OPERAND, 3, false);
        self.a.ld(difference, mem(CPU, LAZY_DST));
        self.a.ld(subtrahend, mem(CPU, LAZY_SRC));
        self.a
            .lea(difference, mem_idx(difference, subtrahend, 0, 0));
        self.a.shift_cl(SHL, difference, true);
        self.a.shift_cl(SHL, subtrahend, true);
        self.a.alu_rr(CMP, difference, subtrahend, true);
    }

    /// Conditional branch ending the block. With `live` host flags the guest condition code is
    /// also the host one; otherwise `helper_cond` evaluates it from the lazy flags. Both exits
    /// run `chain_exit` after the branch, which clobbers the flags the branch has already used.
    /// Conditional branch. As the block's last instruction both outcomes exit; before it (blocks
    /// run on past conditional branches, `blocks::decode_block`) the taken side is a side exit
    /// emitted out of line and the not-taken side falls through into the rest of the block,
    /// with `live` as the branch left it.
    fn jcc(&mut self, cc: u8, target: u64, next: u64, retired: u32, last: bool) {
        let taken = self.a.label();
        if self.live {
            self.a.jcc(cc & 15, taken);
        } else {
            let resume = self.a.label();
            let generic = self.a.label();
            let rebuild = cc >> 1 != CONDITION_PARITY;
            if rebuild {
                self.rebuild_flags_from_sub(generic);
                self.a.jcc(cc & 15, taken);
                self.a.jmp(resume);
            }
            self.a.bind(generic);
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_imm(ARGS[1], cc as u64);
            self.call(self.addrs.cond);
            self.a.test_rr(ACC, ACC, true);
            self.a.jcc(NE, taken);
            self.a.bind(resume);
        }
        if !last {
            self.side_exits
                .push((taken, target, retired, self.prepaid_insns));
            return;
        }
        self.chain_to(next, retired);
        self.a.bind(taken);
        self.chain_to(target, retired);
    }

    /// Lower instruction `index` inline if it is a supported form. Returns false with nothing
    /// emitted otherwise (callers fall back to `helper_exec`), so every `return false` below must come
    /// before the first emit for that arm. Which forms are lowered mirrors `emit.rs`, which the
    /// translation statistics tests rely on.
    fn try_inline(&mut self, index: usize, rip: u64, next: u64) -> bool {
        let insn: Insn = self.insns[index];
        self.skip_flag_record = !self.flag_record_needed[index];
        let insn_index = index as u32;
        let retired = insn_index + 1;
        let osz = insn.opsize;
        let is_mem = insn.md != MODRM_MOD_REGISTER;
        // A 67 prefix only changes how a memory operand's address is formed, so a register form
        // with one (compilers pad with it) is the same instruction.
        let register_form = insn.has_modrm && !is_mem;
        let addressing_ok = !insn.addr32 || register_form;
        // 32/64-bit operation without LOCK, 66 or an effective 67 prefix (`opsize` alone cannot
        // tell: REX.W overrides 66).
        let wide = addressing_ok && !insn.lock && !insn.p66 && (osz == 4 || osz == 8);
        // No LOCK prefix, nor a 67 one on a memory operand, so memory operands can use the
        // inline access path.
        let plain = addressing_ok && !insn.lock;
        let mut store = false;
        match insn.op {
            // Memory read-modify-writes with LOCK (add/or/and/sub/xor, inc/dec, xadd, cmpxchg,
            // bts/btr/btc imm8), xchg with memory (always locked), and cmpxchg/xadd with memory
            // without one, at 32 and 64 bits
            _ if insn.has_modrm
                && is_mem
                && !insn.addr32
                && !insn.p66
                && matches!(osz, 4 | 8)
                && Self::memory_rmw_op(&insn).is_some_and(|op| {
                    insn.lock || matches!(op, RmwOp::Xadd | RmwOp::Xchg | RmwOp::Cmpxchg)
                }) =>
            {
                let op = Self::memory_rmw_op(&insn).unwrap();
                store = self.memory_rmw(&insn, op, next, insn_index, rip);
            }
            // mov r/m, sreg: the selector, into a register at the operand size (zero-extended at
            // 32 and 64 bits, merged at 16) or 2 bytes of memory
            0x8c if insn.reg & 7 <= SEG_GS as u8 && plain => {
                let selector = SEG_SELECTOR + SEG_SIZE * (insn.reg & 7) as i32;
                if is_mem {
                    self.ea(&insn, next);
                }
                self.a.ld_ext(VALUE, mem(CPU, selector), 2, false);
                if is_mem {
                    self.mem_store(2, insn_index, rip);
                    store = true;
                } else {
                    self.write_reg(insn.rm, false, osz.max(2), VALUE);
                }
            }
            // cld / std: only DF changes; the string instructions read it from `Cpu::eflags`
            0xfc | 0xfd if !insn.lock => {
                self.a.ld(SCRATCH_A, mem(CPU, EFLAGS));
                if insn.op == 0xfc {
                    self.a.alu_ri(AND, SCRATCH_A, !(DF as i32), true);
                } else {
                    self.a.alu_ri(OR, SCRATCH_A, DF as i32, true);
                }
                self.a.st(mem(CPU, EFLAGS), SCRATCH_A, 8);
                self.live = false;
            }
            // pause (f3 90)
            0x90 if insn.rep == PREFIX_F3 && insn.rex & GUEST_REX_B == 0 && !insn.p66 => {
                self.pause(insn_index, next, rip);
            }
            // rdtsc, and rdtscp, which also reads IA32_TSC_AUX into ecx
            0x131 => self.rdtsc(index, false, insn_index, rip),
            0x101 if insn.modrm == MODRM_RDTSCP && !insn.lock => {
                self.rdtsc(index, true, insn_index, rip)
            }
            // rdpkru: eax = PKRU, edx = 0. Without CR4.PKE (#UD) or with ecx not zero (#GP) the
            // interpreter runs it.
            0x101 if insn.modrm == MODRM_RDPKRU && !insn.lock => self.rdpkru(insn_index, rip),
            // pushf (64-bit operand)
            0x9c if !insn.p66 => {
                store = self.pushf(insn_index, rip);
            }
            // sti; it ends the block (`ends_block`), which `compile` does with an exit to the run
            // loop, where a pending interrupt is taken once the shadow has passed
            0xfb if !insn.lock => self.sti(insn_index, rip),
            // vzeroupper / vzeroall: bits 511:128 (vzeroall: all) of zmm0-15 cleared
            VZERO if insn.vex && insn.rep == 0 && !insn.p66 && insn.vex_v == 0 => {
                let (_, resume) = self.evex_entry(VectorGate::Avx, insn_index, rip);
                let from = if insn.vex_l { 0 } else { 1 };
                for register in 0..VZERO_REGISTERS {
                    self.zero_vector_lanes(register, from);
                }
                self.a.bind(resume);
                self.live = false;
            }
            // kortest and ktest k, k
            KORTEST | KTEST if insn.vex && insn.rep == 0 && !insn.vex_l && !is_mem => {
                self.opmask_test(&insn, insn.op == KORTEST, insn_index, rip);
            }
            // cmpxchg16b m128 (REX.W 0f c7 /1), with or without LOCK
            0x1c7 if is_mem && insn.reg & 7 == 1 && osz == 8 && !insn.addr32 => {
                store = self.cmpxchg16b(&insn, next, insn_index, rip);
            }
            // add/or/adc/sbb/and/sub/xor/cmp: r/m8,r8 (0), r/m,r (1), r8,r/m8 (2), r,r/m (3),
            // al,imm8 (4) and eax,imm32 (5). adc and sbb below 32 bits or into memory go to the
            // interpreter. 6 and 7 are push/pop of segment registers and prefixes.
            0x00..=0x3f if insn.op & 7 < 6 && plain => {
                let alu_op = ((insn.op >> 3) & 7) as u8;
                let form = insn.op & 7;
                let size = if form & 1 == 0 { 1 } else { osz };
                let (dst, src) = match form {
                    0 | 1 => (Self::rm_dst(&insn, size), Self::reg_src(&insn, size)),
                    2 | 3 => (Self::reg_dst(&insn, size), Self::rm_src(&insn, size)),
                    _ => (
                        Dst::Reg(GUEST_RAX, false),
                        Src::Imm(Self::imm_for(size, insn.imm)),
                    ),
                };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && (size < 4 || matches!(dst, Dst::Mem)) {
                    return false;
                }
                store = self.alu(&insn, alu_op, dst, src, size, next, insn_index, rip);
            }
            // group 1: add/or/.../cmp r/m, imm8 (80) or imm32/sign-extended imm8 (81, 83)
            0x80 | 0x81 | 0x83 if plain => {
                let alu_op = insn.reg & 7;
                let size = if insn.op == 0x80 { 1 } else { osz };
                if matches!(alu_op, ALU_ADC | ALU_SBB) && (size < 4 || is_mem) {
                    return false;
                }
                store = self.alu(
                    &insn,
                    alu_op,
                    Self::rm_dst(&insn, size),
                    Src::Imm(Self::imm_for(size, insn.imm)),
                    size,
                    next,
                    insn_index,
                    rip,
                );
            }
            // test r/m, r (84, 85) and test al/eax, imm (a8, a9)
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
                store = self.alu(&insn, ALU_TEST, dst, src, size, next, insn_index, rip);
            }
            // group 3: /0 test r/m, imm; /2 not; /3 neg
            // shld/shrd r/m, r, imm8 at 32 and 64 bits
            0x1a4 | 0x1ac if wide && plain => {
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
                    insn_index,
                    rip,
                );
            }
            // group 3 /6 div and /7 idiv r/m at 32 and 64 bits
            0xf7 if wide && plain && matches!(insn.reg & 7, GROUP3_DIV | GROUP3_IDIV) => {
                self.divide(&insn, insn.reg & 7 == GROUP3_IDIV, next, insn_index, rip);
            }
            // bt/bts/btr/btc r/m, r and r/m, imm8 at 32 and 64 bits
            0x1a3 | 0x1ab | 0x1b3 | 0x1bb if wide && plain => {
                let (_, operation) = *BIT_TEST_REGISTER_OFFSET
                    .iter()
                    .find(|&&(op, _)| op == insn.op)
                    .unwrap();
                store = self.bit_test(&insn, operation, None, next, insn_index, rip);
            }
            BIT_TEST_IMMEDIATE if wide && plain && insn.reg & 7 >= GROUP8_FIRST_BIT_OP => {
                let operation = (insn.reg & 7) - GROUP8_FIRST_BIT_OP;
                let offset = insn.imm as u8 & (8 * osz - 1);
                store = self.bit_test(&insn, operation, Some(offset), next, insn_index, rip);
            }
            // leave: rsp = rbp, then pop rbp. The load comes first, so a faulting one changes
            // neither register.
            0xc9 if !insn.p66 && !insn.lock => {
                self.ld_gpr(ADDR, GUEST_RBP);
                self.stack_load(VALUE, insn_index, rip);
                self.a.lea(ADDR, mem(ADDR, PUSH_POP_SIZE));
                self.st_gpr(GUEST_RSP, ADDR);
                self.st_gpr(GUEST_RBP, VALUE);
            }
            // bswap r32 / r64 (the 16-bit form is undefined and stays in the interpreter)
            0x1c8..=0x1cf if wide => {
                let reg = opcode_register(&insn);
                self.ld_gpr(VALUE, reg);
                self.a.bswap_r(VALUE, osz == 8);
                self.st_gpr(reg, VALUE);
            }
            // xchg r, r (87 register form) and xchg rax, r (91..97, or 90 with REX.B; `f3 90` is
            // pause, left to the nop arm below) at 32 and 64 bits. The memory form is a locked
            // access and stays in the interpreter.
            0x87 | 0x90..=0x97
                if wide
                    && (insn.op != 0x87 || !is_mem)
                    && (insn.op != 0x90 || (insn.rex & GUEST_REX_B != 0 && insn.rep == 0)) =>
            {
                let (first, second) = if insn.op == 0x87 {
                    (insn.reg, insn.rm)
                } else {
                    (GUEST_RAX, opcode_register(&insn))
                };
                // A 4-byte read zero-extends and a 4-byte write fills the whole slot.
                self.read_reg(ACC, first, false, osz);
                self.read_reg(OPERAND, second, false, osz);
                self.write_reg(first, false, osz, OPERAND);
                self.write_reg(second, false, osz, ACC);
            }
            // group 3 /4 mul and /5 imul r/m, one operand, at 32 and 64 bits
            0xf7 if wide && plain && matches!(insn.reg & 7, 4 | 5) => {
                self.mul_wide(&insn, insn.reg & 7 == 5, next, insn_index, rip);
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
                        insn_index,
                        rip,
                    ),
                    2 => self.not(&insn, dst, size, next, insn_index, rip),
                    _ => self.neg(&insn, dst, size, next, insn_index, rip),
                };
            }
            // group 4/5: /0 inc, /1 dec
            0xfe | 0xff if plain && insn.reg & 7 <= 1 => {
                let size = if insn.op == 0xfe { 1 } else { osz };
                store = self.inc_dec(
                    &insn,
                    insn.reg & 7 == 1,
                    Self::rm_dst(&insn, size),
                    size,
                    next,
                    insn_index,
                    rip,
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
                store = self.shift(
                    &insn,
                    kind,
                    Self::rm_dst(&insn, size),
                    count,
                    size,
                    next,
                    insn_index,
                    rip,
                );
            }
            // shl/shr/sar r/m, imm8 | 1 | cl at 32 and 64 bits (/6 is the alias of /4). Rotates
            // and a zero immediate count go to the interpreter.
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
                    insn_index,
                    rip,
                );
            }
            // mov r/m8,r8 (88), r/m,r (89), r8,r/m8 (8a), r,r/m (8b)
            0x88..=0x8b if plain => {
                let size = if insn.op & 1 == 0 { 1 } else { osz };
                if insn.op < 0x8a {
                    let (src_reg, src_high) = Self::named_register(&insn, insn.reg, size);
                    self.read_reg(VALUE, src_reg, src_high, size);
                    match Self::rm_dst(&insn, size) {
                        Dst::Mem => {
                            self.ea(&insn, next);
                            self.mem_store(size, insn_index, rip);
                            store = true;
                        }
                        Dst::Reg(reg, high) => self.write_reg(reg, high, size, VALUE),
                    }
                } else {
                    let (dst_reg, dst_high) = Self::named_register(&insn, insn.reg, size);
                    match Self::rm_src(&insn, size) {
                        Src::Reg(reg, high) => self.read_reg(VALUE, reg, high, size),
                        _ => {
                            self.ea(&insn, next);
                            self.mem_load(VALUE, size, false, insn_index, rip);
                        }
                    }
                    self.write_reg(dst_reg, dst_high, size, VALUE);
                }
            }
            // group 11 /0: mov r/m8, imm8 (c6) and mov r/m, imm (c7)
            0xc6 | 0xc7 if plain && insn.reg & 7 == 0 => {
                let size = if insn.op == 0xc6 { 1 } else { osz };
                let v = Self::imm_for(size, insn.imm);
                match Self::rm_dst(&insn, size) {
                    Dst::Mem => {
                        self.ea(&insn, next);
                        self.a.mov_imm(VALUE, v);
                        self.mem_store(size, insn_index, rip);
                        store = true;
                    }
                    Dst::Reg(reg, high) => {
                        self.a.mov_imm(VALUE, v);
                        self.write_reg(reg, high, size, VALUE);
                    }
                }
            }
            // mov r8, imm8 and mov r, imm32/imm64 (register in the opcode, extended by REX.B)
            0xb0..=0xbf if plain => {
                let size = if insn.op < 0xb8 { 1 } else { osz };
                let (reg, high) = Self::named_register(&insn, opcode_register(&insn), size);
                self.a.mov_imm(VALUE, Self::imm_for(size, insn.imm));
                self.write_reg(reg, high, size, VALUE);
            }
            // setcc r/m8. The condition goes into VALUE before the address is formed: for a
            // memory destination the store's probe would clobber the host flags.
            0x190..=0x19f if plain => {
                self.condition_to_register((insn.op & 15) as u8, VALUE);
                match Self::rm_dst(&insn, 1) {
                    Dst::Reg(reg, high) => self.write_reg(reg, high, 1, VALUE),
                    Dst::Mem => {
                        self.ea(&insn, next);
                        self.mem_store(1, insn_index, rip);
                        store = true;
                    }
                }
            }
            // cmovcc r, r/m at 32 and 64 bits. The source is always read (it can fault) and a
            // 32-bit destination is zero-extended whether or not it moves, which the host cmov
            // does too. With live flags and a register source the host cmov reads them directly;
            // otherwise the condition is evaluated first, into a register that survives the
            // source's memory access, and tested afterwards.
            0x140..=0x14f if wide => {
                let cc = (insn.op & 15) as u8;
                let source = Self::rm_src(&insn, osz);
                match source {
                    Src::Reg(reg, _) if self.live => {
                        self.read_reg(VALUE, reg, false, osz);
                        self.read_reg(ACC, insn.reg, false, osz);
                        self.a.cmov_rr(cc, ACC, VALUE, osz == 8);
                    }
                    _ => {
                        self.condition_to_register(cc, SAVED_CONDITION);
                        match source {
                            Src::Reg(reg, _) => self.read_reg(VALUE, reg, false, osz),
                            _ => {
                                self.ea(&insn, next);
                                self.mem_load(VALUE, osz, false, insn_index, rip);
                            }
                        }
                        self.read_reg(ACC, insn.reg, false, osz);
                        self.a.test_rr(SAVED_CONDITION, SAVED_CONDITION, false);
                        self.a.cmov_rr(NE, ACC, VALUE, osz == 8);
                        self.live = false;
                    }
                }
                self.st_gpr(insn.reg, ACC);
            }
            // imul r, r/m (0f af) and imul r, r/m, imm32 / imm8 (69, 6b) at 32 and 64 bits
            0x1af | 0x69 | 0x6b if wide => self.imul(&insn, osz, next, insn_index, rip),
            // group 5: /2 call near, /4 jmp near (r/m64), /6 push r/m64
            0xff if plain && !insn.p66 && matches!(insn.reg & 7, 2 | 4 | 6) => {
                match Self::rm_src(&insn, 8) {
                    Src::Reg(reg, _) => self.read_reg(VALUE, reg, false, 8),
                    _ => {
                        self.ea(&insn, next);
                        self.mem_load(VALUE, 8, false, insn_index, rip);
                    }
                }
                match insn.reg & 7 {
                    2 => {
                        // The target is read before the push, so `call [rsp + n]` sees the old
                        // rsp. It waits in a callee-saved register across the push's slow path.
                        self.a.mov_rr(SAVED_OPERAND, VALUE, true);
                        self.ld_gpr(ADDR, GUEST_RSP);
                        self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
                        self.rip_to(VALUE, next);
                        self.stack_store(insn_index, rip);
                        self.shadow_stack_push(next, insn_index, rip);
                        self.st_gpr(GUEST_RSP, ADDR);
                        self.a.mov_rr(OPERAND, SAVED_OPERAND, true);
                        self.chain_exit(retired);
                        self.inlined += 1;
                        return true;
                    }
                    4 => {
                        self.a.mov_rr(OPERAND, VALUE, true);
                        self.chain_exit(retired);
                        self.inlined += 1;
                        return true;
                    }
                    _ => {
                        self.ld_gpr(ADDR, GUEST_RSP);
                        self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
                        self.stack_store(insn_index, rip);
                        self.st_gpr(GUEST_RSP, ADDR);
                        store = true;
                    }
                }
            }
            // lfence, mfence, sfence: one CPU, so ordering is free
            0x1ae if !is_mem && insn.rep == 0 && matches!(insn.reg & 7, 5..=7) => {}
            // cdqe / cwde: sign-extend eax (ax) into rax (eax). Loads and `mov`s only, so the
            // host flags survive.
            0x98 if wide => {
                if osz == 8 {
                    self.a
                        .ld_ext(VALUE, register_slot(GUEST_RAX, false), 4, true);
                } else {
                    self.a
                        .ld_ext(VALUE, register_slot(GUEST_RAX, false), 2, true);
                    self.a.mov_rr(VALUE, VALUE, false);
                }
                self.st_gpr(GUEST_RAX, VALUE);
            }
            // cqo / cdq: rdx (edx, zero-extended) gets the sign of rax (eax). The host `cdq`
            // leaves the flags alone.
            0x99 if wide => {
                self.read_reg(ACC, GUEST_RAX, false, osz);
                self.a.cdq(osz == 8);
                if osz == 4 {
                    self.a.mov_rr(ADDR, ADDR, false);
                }
                self.st_gpr(GUEST_RDX, ADDR);
            }
            // lea; `ea` would add a segment base, which lea must not
            0x8d if wide && is_mem => {
                if has_fs_gs_override(&insn) {
                    return false;
                }
                self.ea(&insn, next);
                if osz == 4 {
                    self.a.mov_rr(ADDR, ADDR, false);
                }
                self.st_gpr(insn.reg, ADDR);
            }
            // movsxd r64, r/m32
            0x63 if osz == 8 && !insn.addr32 && !insn.lock && !insn.p66 && (!is_mem || plain) => {
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(VALUE, 4, true, insn_index, rip);
                } else {
                    self.ld_gpr(VALUE, insn.rm);
                    self.a.ext_rr(VALUE, VALUE, 4, true);
                }
                self.st_gpr(insn.reg, VALUE);
            }
            // movzx (0f b6, b7) / movsx (0f be, bf) from a byte (even opcode) or word source
            0x1b6 | 0x1b7 | 0x1be | 0x1bf if wide && (!is_mem || plain) => {
                let source_size: u8 = if insn.op & 1 == 0 { 1 } else { 2 };
                let signed = insn.op >= 0x1b8;
                if is_mem {
                    self.ea(&insn, next);
                    self.mem_load(VALUE, source_size, signed, insn_index, rip);
                } else {
                    // Straight from the register's slot, extended by the load: an ah/ch/dh/bh
                    // source is the byte at offset 1.
                    let (source, high) = Self::named_register(&insn, insn.rm, source_size);
                    self.a
                        .ld_ext(VALUE, register_slot(source, high), source_size, signed);
                }
                if osz == 4 {
                    self.a.mov_rr(VALUE, VALUE, false);
                }
                self.st_gpr(insn.reg, VALUE);
            }
            // bsf/bsr (no prefix) and tzcnt/lzcnt (F3) at 32 and 64 bits
            0x1bc | 0x1bd if wide && plain && matches!(insn.rep, 0 | 1) => {
                let scan = match (insn.op, insn.rep) {
                    (0x1bc, 0) => BitScan::Bsf,
                    (0x1bd, 0) => BitScan::Bsr,
                    (0x1bc, _) => BitScan::Tzcnt,
                    _ => BitScan::Lzcnt,
                };
                self.bit_scan(&insn, scan, next, insn_index, rip);
            }
            // BMI1/BMI2 (VEX.LZ, map 0f38): shlx/shrx/sarx f7 with 66/F2/F3, bzhi f5, andn f2,
            // the blsr/blsmsk/blsi group f3 and mulx f6. bextr, pdep and pext go to the
            // interpreter.
            VEX_BMI_SHIFT_OR_BEXTR
            | VEX_BMI_BZHI_PEXT_PDEP
            | VEX_BMI_ANDN
            | VEX_BMI_GROUP17
            | VEX_BMI_MULX
                if insn.vex
                    && !insn.vex_l
                    && !insn.lock
                    && (osz == 4 || osz == 8)
                    && (!is_mem || plain) =>
            {
                match (insn.op, insn.rep, insn.p66) {
                    (VEX_BMI_SHIFT_OR_BEXTR, 0, true) => {
                        self.bmi_shift(&insn, SHL, next, insn_index, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, PREFIX_F2, false) => {
                        self.bmi_shift(&insn, SHR, next, insn_index, rip)
                    }
                    (VEX_BMI_SHIFT_OR_BEXTR, PREFIX_F3, false) => {
                        self.bmi_shift(&insn, SAR, next, insn_index, rip)
                    }
                    (VEX_BMI_BZHI_PEXT_PDEP, 0, false) => self.bzhi(&insn, next, insn_index, rip),
                    (VEX_BMI_ANDN, 0, false) => self.andn(&insn, next, insn_index, rip),
                    (VEX_BMI_MULX, PREFIX_F2, false) => self.mulx(&insn, next, insn_index, rip),
                    (VEX_BMI_GROUP17, 0, false)
                        if matches!(
                            insn.reg & 7,
                            BMI_GROUP_BLSR | BMI_GROUP_BLSMSK | BMI_GROUP_BLSI
                        ) =>
                    {
                        self.lowest_set_bit(&insn, insn.reg & 7, next, insn_index, rip)
                    }
                    _ => return false,
                }
            }
            // adcx (66) and adox (F3) r, r/m at 32 and 64 bits. The 66 is part of the opcode, so
            // the operand size is 64 with REX.W and 32 otherwise.
            ADCX_ADOX
                if !insn.vex
                    && !insn.evex
                    && plain
                    && matches!((insn.p66, insn.rep), (true, 0) | (false, PREFIX_F3)) =>
            {
                self.add_carry_chain(&insn, insn.rep == PREFIX_F3, next, insn_index, rip);
            }
            // movd/movq between an xmm register and a general register or memory
            MOVD_TO_VECTOR | MOVD_FROM_VECTOR
                if plain && !insn.vex && !insn.evex && insn.p66 && insn.rep == 0 =>
            {
                store = self.movd(&insn, insn.op == MOVD_TO_VECTOR, next, insn_index, rip);
            }
            // push r64. rsp is updated only after the store succeeded, and the register is read
            // after the new rsp is computed but before it is stored, so `push rsp` pushes the
            // old value.
            0x50..=0x57 if !insn.p66 => {
                let r = opcode_register(&insn);
                self.ld_gpr(ADDR, GUEST_RSP);
                self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
                self.ld_gpr(VALUE, r);
                self.stack_store(insn_index, rip);
                self.st_gpr(GUEST_RSP, ADDR);
                store = true;
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
                self.ld_gpr(ADDR, GUEST_RSP);
                self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
                self.a.mov_imm(VALUE, insn.imm as u64);
                self.stack_store(insn_index, rip);
                self.st_gpr(GUEST_RSP, ADDR);
                store = true;
            }
            // pop r64. The register is written last so `pop rsp` ends up with the popped value.
            0x58..=0x5f if !insn.p66 => {
                let r = opcode_register(&insn);
                self.ld_gpr(ADDR, GUEST_RSP);
                self.stack_load(VALUE, insn_index, rip);
                self.a.lea(ADDR, mem(ADDR, PUSH_POP_SIZE));
                self.st_gpr(GUEST_RSP, ADDR);
                self.st_gpr(r, VALUE);
            }
            // nop, prefetchw (0f 0d), and the 0f 18..1f hint/nop group: nothing to emit. 0x90
            // with REX.B is xchg r8, rax.
            0x90 | 0x10d | 0x118..=0x11f => {
                // `f3 90` is pause: an SVM guest can intercept it, so it goes through `exec`
                if insn.op == 0x90 && (insn.rex & GUEST_REX_B != 0 || insn.rep != 0)
                    || is_rdssp(&insn)
                {
                    return false;
                }
            }
            // jcc rel8 / rel32
            0x70..=0x7f | 0x180..=0x18f => {
                let target = next.wrapping_add(insn.imm as u64);
                let last = index + 1 == self.insns.len();
                self.jcc((insn.op & 15) as u8, target, next, retired, last);
                self.inlined += 1;
                return true;
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.chain_to(next.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            // call rel32. The block ends here, so no `smc_check` is needed after the push: the
            // chain stub looks at `block_exit_requested` itself.
            0xe8 => {
                self.ld_gpr(ADDR, GUEST_RSP);
                self.a.lea(ADDR, mem(ADDR, -PUSH_POP_SIZE));
                self.rip_to(VALUE, next);
                self.stack_store(insn_index, rip);
                // `call` to the next instruction (getting rip) leaves the shadow stack alone.
                if insn.imm != 0 {
                    self.shadow_stack_push(next, insn_index, rip);
                }
                self.st_gpr(GUEST_RSP, ADDR);
                self.chain_to(next.wrapping_add(insn.imm as u64), retired);
                self.inlined += 1;
                return true;
            }
            // ret
            0xc3 => {
                self.ld_gpr(ADDR, GUEST_RSP);
                self.stack_load(VALUE, insn_index, rip);
                self.shadow_stack_pop(insn_index, rip);
                self.a.lea(ADDR, mem(ADDR, PUSH_POP_SIZE));
                self.st_gpr(GUEST_RSP, ADDR);
                self.a.mov_rr(OPERAND, VALUE, true);
                self.chain_exit(retired);
                self.inlined += 1;
                return true;
            }
            // movups/movupd, movaps/movapd, movdqa/movdqu
            0x110 | 0x111 | 0x128 | 0x129 | 0x16f | 0x17f
                if plain && vector_move(&insn).is_some() =>
            {
                let form = vector_move(&insn).unwrap();
                store = self.vector_move(&insn, form, next, insn_index, rip);
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
                // pmovmskb has no memory form (the #UD stays with the interpreter)
                if plain
                    && vector_alu(&insn)
                        .is_some_and(|form| form.op != VectorOp::MoveMask || !is_mem) =>
            {
                let form = vector_alu(&insn).unwrap();
                store = self.vector_alu(&insn, form, next, insn_index, rip);
            }
            // psrlq/psllq/psrldq/pslldq xmm, imm8 (66 0f 73 /2 /6 /3 /7)
            PSHIFT_QUADWORD_IMMEDIATE if plain => {
                let Some((_, count)) = vector_shift(&insn) else {
                    return false;
                };
                self.vector_shift(&insn, count, insn_index, rip);
            }
            // movq xmm/m64, xmm (66 0f d6)
            MOVQ_STORE if plain && is_movq_store(&insn) => {
                store = self.movq_store(&insn, next, insn_index, rip);
            }
            // movss/movsd, movlps/movlpd, movhps/movhpd and movq xmm, xmm/m64 (`partial_move`)
            _ if plain && partial_move(&insn).is_some() => {
                let form = partial_move(&insn).unwrap();
                store = self.partial_move(&insn, form, next, insn_index, rip);
            }
            // The SHA extensions, pshufd, palignr, pshufb, the SSE2 integer instructions that
            // `vector_alu` leaves, pshuflw/pshufhw, pblendw, movhlps/movlhps: see `whole_vector`
            _ if plain && whole_vector(&insn).is_some() => {
                let form = whole_vector(&insn).unwrap();
                store = self.whole_vector(&insn, form, next, insn_index, rip);
            }
            // AVX and AVX-512 data movement, integer logic, arithmetic and compares, byte compares
            // into an opmask register, vpmovmskb, broadcasts from a general register and kmov
            // (see `evex_form`)
            _ if insn.evex || insn.vex => {
                let Some(form) = evex_form(&insn) else {
                    return false;
                };
                if is_mem && !plain {
                    return false;
                }
                store = self.evex(&insn, form, next, insn_index, rip);
            }
            _ => return false,
        }
        if store {
            self.smc_check(insn_index, next);
        }
        self.inlined += 1;
        true
    }

    /// swapgs, clac, stac or cli, inline at CPL 0. At any other CPL (or, for cli, CPL > IOPL,
    /// which CPL 0 never is) the out-of-line interpreter fallback raises #GP or #UD. None of the
    /// four can be intercepted by an SVM guest's host, so a nested guest runs them here too.
    fn ring0_system(&mut self, insn: &Insn, insn_index: u32, rip: u64) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Exec,
            insn_index,
            rip,
        });
        self.a.cmp_m8_imm(mem(CPU, CPL), 0);
        self.a.jcc(NE, slow);
        if insn.op == 0x101 && insn.modrm == MODRM_SWAPGS {
            self.a.ld(SCRATCH_A, mem(CPU, GS_BASE));
            self.a.ld(SCRATCH_B, mem(CPU, KGSBASE));
            self.a.st(mem(CPU, GS_BASE), SCRATCH_B, 8);
            self.a.st(mem(CPU, KGSBASE), SCRATCH_A, 8);
        } else {
            // `!IF` and `!AC` fit a sign-extended imm32, so the and keeps the upper half.
            let (op, bits) = match (insn.op, insn.modrm) {
                (0xfa, _) => (AND, !IF as i32),
                (_, MODRM_CLAC) => (AND, !AC as i32),
                _ => (OR, AC as i32),
            };
            self.a.ld(SCRATCH_A, mem(CPU, EFLAGS));
            self.a.alu_ri(op, SCRATCH_A, bits, true);
            self.a.st(mem(CPU, EFLAGS), SCRATCH_A, 8);
        }
        self.a.bind(resume);
        // The compare (and the fallback's call) clobbered the host flags.
        self.live = false;
    }

    /// Starts an inline SSE instruction: registers its out-of-line interpreter fallback and, for
    /// the first one of the block, checks the SSE gate (#UD for CR0.EM or no CR4.OSFXSR, #NM for
    /// CR0.TS), jumping to the fallback so the interpreter raises the exception. Returns the
    /// fallback's entry and the label to bind where the inline code ends.
    fn sse_entry(&mut self, insn_index: u32, rip: u64) -> (Label, Label) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Exec,
            insn_index,
            rip,
        });
        if !self.sse_gated {
            self.sse_gated = true;
            self.a.ld(SCRATCH_A, mem(CPU, CR0));
            self.a
                .alu_ri(AND, SCRATCH_A, (CR0_EM | CR0_TS) as i32, true);
            self.a.jcc(NE, slow);
            self.a.ld(SCRATCH_A, mem(CPU, CR4));
            self.a.alu_ri(AND, SCRATCH_A, CR4_OSFXSR as i32, true);
            self.a.jcc(E, slow);
        }
        (slow, resume)
    }

    /// Starts an inline AVX or AVX-512 instruction: registers its out-of-line interpreter fallback
    /// and, for the first one of the block that needs `gate`, checks `Cpu::vex_gate` (`Avx`) or
    /// `Cpu::evex_gate` (`Avx512`): #UD without CR4.OSXSAVE or the state in XCR0, #NM under
    /// CR0.TS, jumping to the fallback so the interpreter raises the exception. A passed AVX-512
    /// check covers later AVX ones, not the reverse. Like the SSE gate it holds for the whole
    /// block.
    fn evex_entry(&mut self, gate: VectorGate, insn_index: u32, rip: u64) -> (Label, Label) {
        let slow = self.a.label();
        let resume = self.a.label();
        self.slows.push(Slow {
            slow,
            resume,
            kind: SlowKind::Exec,
            insn_index,
            rip,
        });
        let (gated, state) = match gate {
            VectorGate::Avx512 => (self.evex_gated, XCR0_EVEX_STATE),
            VectorGate::Avx => (self.vex_gated || self.evex_gated, XCR0_VEX_STATE),
        };
        if !gated {
            self.vex_gated = true;
            self.evex_gated |= gate == VectorGate::Avx512;
            self.a.ld(SCRATCH_A, mem(CPU, CR4));
            self.a.alu_ri(AND, SCRATCH_A, CR4_OSXSAVE as i32, true);
            self.a.jcc(E, slow);
            self.a.ld(SCRATCH_A, mem(CPU, XCR0));
            self.a.alu_ri(AND, SCRATCH_A, state as i32, true);
            self.a.alu_ri(CMP, SCRATCH_A, state as i32, true);
            self.a.jcc(NE, slow);
            self.a.ld(SCRATCH_A, mem(CPU, CR0));
            self.a.alu_ri(AND, SCRATCH_A, CR0_TS as i32, true);
            self.a.jcc(NE, slow);
        }
        (slow, resume)
    }

    /// Zeroes the 128-bit lanes `from..4` of vector register `register`: every VEX or EVEX write
    /// of a vector shorter than 512 bits clears the rest of the register. Needs `HOST_XMM_ZERO`.
    fn zero_vector_lanes(&mut self, register: u8, from: u32) {
        if from < VECTOR_LANES {
            self.a.xmm_op(OPCODE_PXOR, HOST_XMM_ZERO, HOST_XMM_ZERO);
        }
        for lane in from..VECTOR_LANES {
            self.a.xmm_st(
                mem(CPU, vector_lane_offset(register, lane) as i32),
                HOST_XMM_ZERO,
            );
        }
    }

    /// Leaves `ADDR` (the guest address) and `SCRATCH_B` (its TLB addend) set up for the memory
    /// operand of an inline AVX-512 instruction, `bytes` long and inside one page; an `aligned`
    /// operand off a `bytes` boundary, a TLB miss or a page crossing jumps to `slow`, the
    /// interpreter. The compressed disp8 is scaled by the vector length, as in
    /// `Cpu::evex_address`.
    fn evex_memory(
        &mut self,
        insn: &Insn,
        bytes: u32,
        aligned: bool,
        write: bool,
        next: u64,
        slow: Label,
    ) {
        let mut insn = *insn;
        if insn.evex_disp8 {
            insn.disp *= bytes as i64;
        }
        self.ea(&insn, next);
        if aligned {
            self.a.mov_rr(SCRATCH_A, ADDR, true);
            self.a.alu_ri(AND, SCRATCH_A, bytes as i32 - 1, true);
            self.a.jcc(NE, slow);
        }
        let tag = if write { TLB_TAG_WRITE } else { TLB_TAG_READ };
        self.tlb_probe(bytes as u8, tag, slow);
        self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
    }

    /// Per byte, `predicate` (the vpcmp immediate) of xmm0 against xmm1 into xmm0, all ones where
    /// it holds. SSE2 has only signed greater-than and equality, so the unsigned orders go
    /// through min/max and the rest through inversion; xmm3 is the scratch.
    fn compare_bytes(&mut self, signed: bool, predicate: u8) {
        let (a, b, temporary) = (HOST_XMM_DST, HOST_XMM_SRC, HOST_XMM_OLD);
        // a >= b / a <= b unsigned: min or max of the pair equals a
        let bounded = |this: &mut Self, opcode: u8| {
            this.a.xmm_op(OPCODE_MOVDQA, temporary, a);
            this.a.xmm_op(opcode, temporary, b);
            this.a.xmm_op(OPCODE_PCMPEQB, temporary, a);
            this.a.xmm_op(OPCODE_MOVDQA, a, temporary);
        };
        let greater_signed = |this: &mut Self| this.a.xmm_op(OPCODE_PCMPGTB, a, b);
        let less_signed = |this: &mut Self| {
            this.a.xmm_op(OPCODE_MOVDQA, temporary, b);
            this.a.xmm_op(OPCODE_PCMPGTB, temporary, a);
            this.a.xmm_op(OPCODE_MOVDQA, a, temporary);
        };
        let invert = |this: &mut Self| {
            this.a.xmm_op(OPCODE_PCMPEQB, temporary, temporary);
            this.a.xmm_op(OPCODE_PXOR, a, temporary);
        };
        match (predicate, signed) {
            (VPCMP_EQUAL, _) => self.a.xmm_op(OPCODE_PCMPEQB, a, b),
            (VPCMP_NOT_EQUAL, _) => {
                self.a.xmm_op(OPCODE_PCMPEQB, a, b);
                invert(self);
            }
            (VPCMP_FALSE, _) => self.a.xmm_op(OPCODE_PXOR, a, a),
            (VPCMP_LESS, false) => {
                bounded(self, OPCODE_PMAXUB);
                invert(self);
            }
            (VPCMP_LESS_EQUAL, false) => bounded(self, OPCODE_PMINUB),
            (VPCMP_NOT_LESS, false) => bounded(self, OPCODE_PMAXUB),
            (VPCMP_NOT_LESS_EQUAL, false) => {
                bounded(self, OPCODE_PMINUB);
                invert(self);
            }
            (VPCMP_LESS, true) => less_signed(self),
            (VPCMP_LESS_EQUAL, true) => {
                greater_signed(self);
                invert(self);
            }
            (VPCMP_NOT_LESS, true) => {
                less_signed(self);
                invert(self);
            }
            (VPCMP_NOT_LESS_EQUAL, true) => greater_signed(self),
            _ => self.a.xmm_op(OPCODE_PCMPEQB, a, a),
        }
    }

    /// Loads the write mask register of `insn` into `ACC` and the address of the expansion table
    /// for `element_log2` into `OPERAND`, for `commit_vector_lane`.
    fn begin_masked_write(&mut self, insn: &Insn, element_log2: u32) {
        self.a
            .ld(ACC, mem(CPU, OPMASK + OPMASK_SIZE * insn.evex_mask as i32));
        let table = OPMASK_EXPANSION[element_log2 as usize].as_ptr() as u64;
        self.a.mov_imm(OPERAND, table);
    }

    /// Stores the lane in xmm0 to lane `lane` of vector register `register`, under the write mask
    /// when the instruction has one (`begin_masked_write` came first): elements whose mask bit is
    /// clear keep the old value, or become zero with EVEX.z. The mask bits of the lane become an
    /// all-ones-or-zero vector half by half through `OPMASK_EXPANSION`.
    fn commit_vector_lane(&mut self, insn: &Insn, element_log2: u32, register: u8, lane: u32) {
        let offset = vector_lane_offset(register, lane) as i32;
        if insn.evex_mask == 0 {
            self.a.xmm_st(mem(CPU, offset), HOST_XMM_DST);
            return;
        }
        let bits_per_half = 8 >> element_log2;
        let elements_per_lane = 16 >> element_log2;
        for half in 0..2u32 {
            let shift = lane * elements_per_lane + half * bits_per_half;
            self.a.mov_rr(SCRATCH_A, ACC, true);
            if shift != 0 {
                self.a.shift_ri(SHR, SCRATCH_A, shift as u8, true);
            }
            self.a
                .alu_ri(AND, SCRATCH_A, (1 << bits_per_half) - 1, false);
            self.a.ld(VALUE, mem_idx(OPERAND, SCRATCH_A, 3, 0));
            let target = if half == 0 {
                HOST_XMM_SELECT
            } else {
                HOST_XMM_OLD
            };
            self.a.movq_xmm_gpr(target, VALUE);
        }
        self.a
            .xmm_op(OPCODE_PUNPCKLQDQ, HOST_XMM_SELECT, HOST_XMM_OLD);
        // xmm0 = new & select, then merged with the old & !select unless zeroing
        self.a.xmm_op(OPCODE_PAND, HOST_XMM_DST, HOST_XMM_SELECT);
        if !insn.evex_zeroing {
            self.a.xmm_ld(HOST_XMM_OLD, mem(CPU, offset));
            self.a.xmm_op(OPCODE_PANDN, HOST_XMM_SELECT, HOST_XMM_OLD);
            self.a.xmm_op(OPCODE_POR, HOST_XMM_DST, HOST_XMM_SELECT);
        }
        self.a.xmm_st(mem(CPU, offset), HOST_XMM_DST);
    }

    /// An inline AVX or AVX-512 instruction (`EvexForm`), as on aarch64: one 128-bit lane at a time
    /// through host xmm0 and xmm1, loaded from and stored to `Cpu` per lane. Anything but the
    /// plain forms and a clean fast path runs in the interpreter. A masked load reads the whole
    /// vector inside the page the TLB vouches for, which cannot fault, so the elements the mask
    /// drops are read and discarded. Returns whether the instruction stores to memory.
    fn evex(&mut self, insn: &Insn, form: EvexForm, next: u64, insn_index: u32, rip: u64) -> bool {
        let (slow, resume) = self.evex_entry(form.gate, insn_index, rip);
        let lanes = form.bytes / XMM_SIZE as u32;
        let lane_offset = |register: u8, lane: u32| vector_lane_offset(register, lane) as i32;
        let lane_in_memory =
            |lane: u32| mem_idx(ADDR, SCRATCH_B, 0, (lane * XMM_SIZE as u32) as i32);
        let mut store = false;
        let writes_vector = matches!(
            form.kind,
            EvexKind::Alu { .. } | EvexKind::Move { .. } | EvexKind::BroadcastGeneral
        );
        let memory_store = matches!(form.kind, EvexKind::Move { load: false, .. })
            && insn.md != MODRM_MOD_REGISTER;
        if writes_vector && !memory_store && insn.evex_mask != 0 {
            self.begin_masked_write(insn, form.element_log2);
        }
        match form.kind {
            EvexKind::MaskToGeneral { bits } => {
                // A zero-extending load of the width's bytes is the masking.
                self.a.ld_ext(
                    VALUE,
                    mem(
                        CPU,
                        OPMASK + OPMASK_SIZE * (insn.rm & OPMASK_REGISTER_MASK) as i32,
                    ),
                    (bits / u8::BITS) as u8,
                    false,
                );
                self.write_reg(
                    insn.reg,
                    false,
                    if bits == u64::BITS { 8 } else { 4 },
                    VALUE,
                );
            }
            EvexKind::GeneralToMask { bits } => {
                self.read_reg(
                    VALUE,
                    insn.rm & GENERAL_REGISTER_MASK,
                    false,
                    (bits / u8::BITS) as u8,
                );
                self.a.st(
                    mem(
                        CPU,
                        OPMASK + OPMASK_SIZE * (insn.reg & OPMASK_REGISTER_MASK) as i32,
                    ),
                    VALUE,
                    8,
                );
            }
            EvexKind::BroadcastGeneral => {
                // SSE2 only: the element repeated across a quadword by a multiply, then both
                // halves of xmm1, copied to xmm0 for each commit (which rewrites xmm0).
                let element_bytes = 1u8 << form.element_log2;
                self.read_reg(VALUE, insn.rm & GENERAL_REGISTER_MASK, false, element_bytes);
                if let Some(&multiplier) = BROADCAST_MULTIPLIER.get(form.element_log2 as usize) {
                    self.a.mov_imm(SCRATCH_A, multiplier);
                    self.a.imul_rr(VALUE, SCRATCH_A, true);
                }
                self.a.movq_xmm_gpr(HOST_XMM_SRC, VALUE);
                self.a.xmm_op(OPCODE_PUNPCKLQDQ, HOST_XMM_SRC, HOST_XMM_SRC);
                for lane in 0..lanes {
                    self.a.xmm_op(OPCODE_MOVDQA, HOST_XMM_DST, HOST_XMM_SRC);
                    self.commit_vector_lane(insn, form.element_log2, insn.reg, lane);
                }
                self.zero_vector_lanes(insn.reg, lanes);
            }
            EvexKind::MoveMaskBytes => {
                for lane in 0..lanes {
                    self.a
                        .xmm_ld(HOST_XMM_SRC, mem(CPU, lane_offset(insn.rm, lane)));
                    self.a.pmovmskb(VALUE, HOST_XMM_SRC);
                    if lane == 0 {
                        self.a.mov_rr(ACC, VALUE, false);
                    } else {
                        self.a
                            .shift_ri(SHL, VALUE, (LANE_BYTE_MASK_BITS * lane) as u8, false);
                        self.a.alu_rr(OR, ACC, VALUE, false);
                    }
                }
                self.write_reg(insn.reg, false, 4, ACC);
            }
            EvexKind::Move { load, aligned } => {
                if insn.md == MODRM_MOD_REGISTER {
                    let (from, to) = if load {
                        (insn.rm, insn.reg)
                    } else {
                        (insn.reg, insn.rm)
                    };
                    for lane in 0..lanes {
                        self.a
                            .xmm_ld(HOST_XMM_DST, mem(CPU, lane_offset(from, lane)));
                        self.commit_vector_lane(insn, form.element_log2, to, lane);
                    }
                    self.zero_vector_lanes(to, lanes);
                } else if load {
                    self.evex_memory(insn, form.bytes, aligned, false, next, slow);
                    for lane in 0..lanes {
                        self.a.xmm_ld(HOST_XMM_DST, lane_in_memory(lane));
                        self.commit_vector_lane(insn, form.element_log2, insn.reg, lane);
                    }
                    self.zero_vector_lanes(insn.reg, lanes);
                } else {
                    self.evex_memory(insn, form.bytes, aligned, true, next, slow);
                    for lane in 0..lanes {
                        self.a
                            .xmm_ld(HOST_XMM_DST, mem(CPU, lane_offset(insn.reg, lane)));
                        self.a.xmm_st(lane_in_memory(lane), HOST_XMM_DST);
                    }
                    store = true;
                }
            }
            EvexKind::Alu { .. }
            | EvexKind::CompareEqualBytes
            | EvexKind::TestBytes
            | EvexKind::TestNotBytes
            | EvexKind::CompareBytes { .. } => {
                let memory = insn.md != MODRM_MOD_REGISTER;
                if memory {
                    self.evex_memory(insn, form.bytes, false, false, next, slow);
                }
                if matches!(form.kind, EvexKind::TestBytes | EvexKind::TestNotBytes) {
                    self.a.xmm_op(OPCODE_PXOR, HOST_XMM_ZERO, HOST_XMM_ZERO);
                }
                for lane in 0..lanes {
                    // xmm0 = vvvv, xmm1 = r/m, as the guest names them
                    self.a
                        .xmm_ld(HOST_XMM_DST, mem(CPU, lane_offset(insn.vex_v, lane)));
                    if memory {
                        self.a.xmm_ld(HOST_XMM_SRC, lane_in_memory(lane));
                    } else {
                        self.a
                            .xmm_ld(HOST_XMM_SRC, mem(CPU, lane_offset(insn.rm, lane)));
                    }
                    if let EvexKind::Alu { .. } = form.kind {
                        // The guest and host are the same `66 0f xx` instruction, so the opcode
                        // byte carries over (pandn included: both compute `!dst & src`).
                        self.a
                            .xmm_op((insn.op & 0xff) as u8, HOST_XMM_DST, HOST_XMM_SRC);
                        self.commit_vector_lane(insn, form.element_log2, insn.reg, lane);
                        continue;
                    }
                    match form.kind {
                        EvexKind::CompareEqualBytes => {
                            self.a.xmm_op(OPCODE_PCMPEQB, HOST_XMM_DST, HOST_XMM_SRC);
                        }
                        EvexKind::CompareBytes { signed, predicate } => {
                            self.compare_bytes(signed, predicate);
                        }
                        _ => {
                            // (a & b) == 0 per byte
                            self.a.xmm_op(OPCODE_PAND, HOST_XMM_DST, HOST_XMM_SRC);
                            self.a.xmm_op(OPCODE_PCMPEQB, HOST_XMM_DST, HOST_XMM_ZERO);
                        }
                    }
                    self.a.pmovmskb(VALUE, HOST_XMM_DST);
                    if form.kind == EvexKind::TestBytes {
                        // vptestmb wants the non-zero bytes
                        self.a.alu_ri(XOR, VALUE, 0xffff, false);
                    }
                    if lane == 0 {
                        self.a.mov_rr(ACC, VALUE, false);
                    } else {
                        self.a
                            .shift_ri(SHL, VALUE, (LANE_BYTE_MASK_BITS * lane) as u8, true);
                        self.a.alu_rr(OR, ACC, VALUE, true);
                    }
                }
                if matches!(form.kind, EvexKind::Alu { .. }) {
                    self.zero_vector_lanes(insn.reg, lanes);
                } else {
                    if insn.evex_mask != 0 {
                        self.a.ld(
                            SCRATCH_A,
                            mem(CPU, OPMASK + OPMASK_SIZE * insn.evex_mask as i32),
                        );
                        self.a.alu_rr(AND, ACC, SCRATCH_A, true);
                    }
                    self.a.st(
                        mem(
                            CPU,
                            OPMASK + OPMASK_SIZE * (insn.reg & OPMASK_REGISTER_MASK) as i32,
                        ),
                        ACC,
                        8,
                    );
                }
            }
        }
        self.a.bind(resume);
        self.live = false;
        store
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
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM + XMM_SIZE * reg as i32;
        // The two halves travel through VALUE and SCRATCH_C, both free once the probe is done.
        if insn.md == MODRM_MOD_REGISTER {
            let (from, to) = if form.load {
                (insn.rm, insn.reg)
            } else {
                (insn.reg, insn.rm)
            };
            self.a.ld(VALUE, mem(CPU, xmm(from)));
            self.a.ld(SCRATCH_C, mem(CPU, xmm(from) + XMM_HALF));
            self.a.st(mem(CPU, xmm(to)), VALUE, 8);
            self.a.st(mem(CPU, xmm(to) + XMM_HALF), SCRATCH_C, 8);
            self.a.bind(resume);
            self.live = false;
            return false;
        }
        self.ea(insn, next);
        if form.aligned {
            self.a.mov_rr(SCRATCH_A, ADDR, true);
            self.a.alu_ri(AND, SCRATCH_A, VECTOR_ALIGNMENT - 1, true);
            self.a.jcc(NE, slow);
        }
        let register = xmm(insn.reg);
        if form.load {
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_READ, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
            self.a.ld(VALUE, mem_idx(ADDR, SCRATCH_B, 0, 0));
            self.a.ld(SCRATCH_C, mem_idx(ADDR, SCRATCH_B, 0, XMM_HALF));
            self.a.st(mem(CPU, register), VALUE, 8);
            self.a.st(mem(CPU, register + XMM_HALF), SCRATCH_C, 8);
        } else {
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_WRITE, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
            self.a.ld(VALUE, mem(CPU, register));
            self.a.ld(SCRATCH_C, mem(CPU, register + XMM_HALF));
            self.a.st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, 8);
            self.a
                .st(mem_idx(ADDR, SCRATCH_B, 0, XMM_HALF), SCRATCH_C, 8);
        }
        self.a.bind(resume);
        self.live = false;
        !form.load
    }

    /// A lane-wise integer vector instruction (`VectorAlu`) on xmm registers, the source being a
    /// register or a 16-byte aligned memory operand (misaligned, page-crossing and gate failures
    /// run in the interpreter, as for `vector_move`). Both operands are loaded from `Cpu::xmm`
    /// into host xmm0 and xmm1 with `movdqu`, so nothing depends on how `Cpu` is aligned and
    /// nothing is live in host vector registers across instructions (every helper call clobbers
    /// them).
    fn vector_alu(
        &mut self,
        insn: &Insn,
        form: VectorAlu,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM + XMM_SIZE * reg as i32;
        if form.op == VectorOp::MoveMask {
            self.a.xmm_ld(HOST_XMM_SRC, mem(CPU, xmm(insn.rm)));
            self.a.pmovmskb(VALUE, HOST_XMM_SRC);
            self.write_reg(insn.reg, false, 4, VALUE);
            self.a.bind(resume);
            self.live = false;
            return false;
        }
        self.a.xmm_ld(HOST_XMM_DST, mem(CPU, xmm(insn.reg)));
        if insn.md == MODRM_MOD_REGISTER {
            self.a.xmm_ld(HOST_XMM_SRC, mem(CPU, xmm(insn.rm)));
        } else {
            self.ea(insn, next);
            // Every legacy-SSE form of these instructions needs a 16-byte aligned memory operand.
            self.a.mov_rr(SCRATCH_A, ADDR, true);
            self.a.alu_ri(AND, SCRATCH_A, VECTOR_ALIGNMENT - 1, true);
            self.a.jcc(NE, slow);
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_READ, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
            self.a.xmm_ld(HOST_XMM_SRC, mem_idx(ADDR, SCRATCH_B, 0, 0));
        }
        // The guest and host are the same `66 0f xx` instruction, so the opcode byte carries over
        // (pandn included: both compute `!dst & src`).
        self.a
            .xmm_op((insn.op & 0xff) as u8, HOST_XMM_DST, HOST_XMM_SRC);
        self.a.xmm_st(mem(CPU, xmm(insn.reg)), HOST_XMM_DST);
        self.a.bind(resume);
        self.live = false;
        false
    }

    /// psrlq/psllq/psrldq/pslldq of xmm `rm` by an immediate (`vector_shift`): the host's own SSE2
    /// instruction, whose counts past the width clear the register as the guest's do.
    fn vector_shift(&mut self, insn: &Insn, count: u8, insn_index: u32, rip: u64) {
        let (_, resume) = self.sse_entry(insn_index, rip);
        let xmm = XMM + XMM_SIZE * insn.rm as i32;
        self.a.xmm_ld(HOST_XMM_DST, mem(CPU, xmm));
        self.a
            .xmm_shift_immediate(insn.reg & OPCODE_REG_MASK, HOST_XMM_DST, count);
        self.a.xmm_st(mem(CPU, xmm), HOST_XMM_DST);
        self.a.bind(resume);
        self.live = false;
    }

    /// movq xmm/m64, xmm (`is_movq_store`): the low quadword of xmm `reg` to memory inside one
    /// page (no alignment rule; a miss or a page crossing runs in the interpreter), or to xmm `rm`
    /// with its high quadword cleared. Returns whether it stores to memory.
    fn movq_store(&mut self, insn: &Insn, next: u64, insn_index: u32, rip: u64) -> bool {
        const QUADWORD_BYTES: u8 = 8;
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM + XMM_SIZE * reg as i32;
        let memory = insn.md != MODRM_MOD_REGISTER;
        if memory {
            self.ea(insn, next);
            self.tlb_probe(QUADWORD_BYTES, TLB_TAG_WRITE, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
            self.a.ld(VALUE, mem(CPU, xmm(insn.reg)));
            self.a
                .st(mem_idx(ADDR, SCRATCH_B, 0, 0), VALUE, QUADWORD_BYTES);
        } else {
            self.a.ld(VALUE, mem(CPU, xmm(insn.reg)));
            self.a.st(mem(CPU, xmm(insn.rm)), VALUE, QUADWORD_BYTES);
            self.a.mov_imm(VALUE, 0);
            self.a
                .st(mem(CPU, xmm(insn.rm) + XMM_HALF), VALUE, QUADWORD_BYTES);
        }
        self.a.bind(resume);
        self.live = false;
        memory
    }

    /// A `PartialMove`: 4 or 8 bytes between part of an xmm register and memory (one access
    /// through the TLB probe, at any alignment that stays in the page) or another xmm register,
    /// behind the SSE gate. Returns whether it stores to memory.
    fn partial_move(
        &mut self,
        insn: &Insn,
        form: PartialMove,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM + XMM_SIZE * reg as i32;
        let memory = insn.md != MODRM_MOD_REGISTER;
        let bytes = form.bytes;
        let part = mem(CPU, xmm(insn.reg) + form.offset as i32);
        if memory {
            let tag = if form.from_register {
                TLB_TAG_WRITE
            } else {
                TLB_TAG_READ
            };
            self.ea(insn, next);
            self.tlb_probe(bytes, tag, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
        }
        let other = if memory {
            mem_idx(ADDR, SCRATCH_B, 0, 0)
        } else {
            mem(CPU, xmm(insn.rm))
        };
        if form.from_register {
            self.a.ld_ext(VALUE, part, bytes, false);
            self.a.st(other, VALUE, bytes);
        } else {
            self.a.ld_ext(VALUE, other, bytes, false);
            if form.zero_rest {
                // The value is zero-extended to 8 bytes, so one quadword store clears the rest
                // of the low half too.
                self.a.st(part, VALUE, 8);
                self.a.mov_imm(VALUE, 0);
                self.a.st(mem(CPU, xmm(insn.reg) + XMM_HALF), VALUE, 8);
            } else {
                self.a.st(part, VALUE, bytes);
            }
        }
        self.a.bind(resume);
        self.live = false;
        memory && form.from_register
    }

    /// A `WholeVector` instruction (the SHA extensions, pshufd, palignr, pshufb), the source being
    /// a register or a 16-byte aligned memory operand with the fallbacks of `vector_alu`. pshufd
    /// is the host's own (SSE2, which every x86-64 has); palignr and pshufb need SSSE3 and the
    /// rest SHA-NI, which the host may lack, so those load the source into the last two argument
    /// registers and call `helper_whole_vector`, which skips the interpreter's decode and dispatch.
    fn whole_vector(
        &mut self,
        insn: &Insn,
        form: WholeVector,
        next: u64,
        insn_index: u32,
        rip: u64,
    ) -> bool {
        let (slow, resume) = self.sse_entry(insn_index, rip);
        let xmm = |reg: u8| XMM + XMM_SIZE * reg as i32;
        let half = XMM_SIZE / 2;
        // Where the source is: in `Cpu::xmm`, or at the operand's host address.
        let source = if insn.md == MODRM_MOD_REGISTER {
            mem(CPU, xmm(insn.rm))
        } else {
            self.ea(insn, next);
            // Every one of these needs a 16-byte aligned memory operand (legacy SSE).
            self.a.mov_rr(SCRATCH_A, ADDR, true);
            self.a.alu_ri(AND, SCRATCH_A, VECTOR_ALIGNMENT - 1, true);
            self.a.jcc(NE, slow);
            self.tlb_probe(XMM_SIZE as u8, TLB_TAG_READ, slow);
            self.a.ld(SCRATCH_B, mem(SCRATCH_A, TLB_ADDEND));
            mem_idx(ADDR, SCRATCH_B, 0, 0)
        };
        let source_high = source.plus(half);
        if form.op == WholeVectorOp::ShuffleDwords {
            self.a.xmm_ld(HOST_XMM_SRC, source);
            self.a.pshufd(HOST_XMM_DST, HOST_XMM_SRC, form.imm);
            self.a.xmm_st(mem(CPU, xmm(insn.reg)), HOST_XMM_DST);
        } else {
            // Loaded into registers that are not the first two arguments on either ABI, then
            // moved into the last two (on SysV those are rdx, the guest address, and rcx).
            self.a.ld(SCRATCH_C, source);
            self.a.ld(VALUE, source_high);
            if ARGS[2] != SCRATCH_C {
                self.a.mov_rr(ARGS[2], SCRATCH_C, true);
            }
            if ARGS[3] != VALUE {
                self.a.mov_rr(ARGS[3], VALUE, true);
            }
            self.a.mov_rr(ARGS[0], CPU, true);
            self.a.mov_imm(ARGS[1], form.packed(insn.reg));
            self.call(self.addrs.whole_vector);
        }
        self.a.bind(resume);
        // The probe, and the call when there is one, clobbered the host flags.
        self.live = false;
        false
    }

    /// `rips[k]` is the guest rip of instruction `k`; `end_rip` is the rip after the last one.
    pub fn compile(&mut self, rips: &[u64], end_rip: u64) {
        let count = self.insns.len();
        self.entry_rip = rips[0];
        if self.addrs.chain != 0 {
            self.link_entry();
        }
        self.body_offset = (self.a.here() - self.a.base()) as u32;
        for index in 0..count {
            let rip = rips[index];
            let next = if index + 1 < count {
                rips[index + 1]
            } else {
                end_rip
            };
            let terminal = ends_block(&self.insns[index]);
            if self.try_inline(index, rip, next) {
                if is_inline_terminator(&self.insns[index])
                    && !(is_conditional_branch(&self.insns[index]) && index + 1 < count)
                {
                    self.emit_slows();
                    return;
                }
                if terminal && !is_inline_terminator(&self.insns[index]) {
                    // An inline instruction the run loop must see the effect of (sti): back to
                    // it, not on to the next block.
                    self.set_rip(next);
                    self.exit_ok(index as u32 + 1);
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
    //! `Gen::rip_to` run as real code: the page base is in the frame slot the trampoline fills.

    use super::*;
    use crate::sys::ExecMem;

    const DUMMY_ADDRESSES: Addrs = Addrs {
        exec: 0,
        ld: 0,
        st: 0,
        ld_stack: 0,
        st_stack: 0,
        cond: 0,
        whole_vector: 0,
        materialize_flags: 0,
        rdtsc: 0,
        pause: 0,
        exit_ok: 0,
        exit_fail: 0,
        chain: 0,
    };

    /// `extern "C" fn(page_base) -> rip_to(offset)`, with the frame slot set up as the
    /// trampoline does.
    fn rip_to_function(offset: u64, live_flags: bool) -> (ExecMem, usize) {
        let memory = ExecMem::new(16384).unwrap();
        let insns: [Insn; 0] = [];
        let mut generator = Gen::new(memory.base(), &DUMMY_ADDRESSES, &insns);
        generator.a.alu_ri(SUB, RSP, FRAME, true);
        generator.a.st(mem(RSP, PAGE_BASE_SLOT), ARGS[0], 8);
        if live_flags {
            // 0 - 0 sets ZF, and the result must survive.
            generator.a.alu_rr(XOR, RAX, RAX, true);
        }
        let before = generator.a.code.len();
        generator.rip_to(RDX, offset);
        let bytes = generator.a.code.len() - before;
        if live_flags {
            generator.a.mov_imm(RAX, 0);
            generator.a.mov_imm(RCX, 1);
            // cmove rax, rcx
            generator
                .a
                .code
                .extend_from_slice(&[0x48, 0x0f, 0x44, 0xc1]);
            generator.a.alu_rr(ADD, RAX, RDX, true);
        } else {
            generator.a.mov_rr(RAX, RDX, true);
        }
        generator.a.alu_ri(ADD, RSP, FRAME, true);
        generator.a.ret();
        generator.a.finish();
        memory.write_bytes(0, &generator.a.code);
        (memory, bytes)
    }

    fn rip_to_result(page_base: u64, offset: u64) -> (u64, usize) {
        let _guard = crate::jit::serialize_jit_tests();
        let (memory, bytes) = rip_to_function(offset, false);
        let function: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        (function(page_base), bytes)
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
            0x12_3456,
            0x4000_0000,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0fff,
            i64::MAX,
            -1,
            -4096,
            -0x12_3456,
            -0x4000_0000,
            -0x8000_0000,
            -0x8000_0001,
            -0x8000_1000,
            i64::MIN,
            i64::MIN + 1,
            0x1_0000_0000,
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
    fn rip_to_leaves_the_host_flags_alone_for_every_offset_size() {
        let _guard = crate::jit::serialize_jit_tests();
        for offset in [0u64, 8, 0x7fff_ffff, 0x8000_0000, u64::MAX, 1 << 40] {
            let (memory, _) = rip_to_function(offset, true);
            let function: extern "C" fn(u64) -> u64 =
                unsafe { std::mem::transmute(memory.base() as usize) };
            // rax = 1 (ZF was still set) + the address.
            assert_eq!(
                function(0x1000),
                1 + 0x1000u64.wrapping_add(offset),
                "offset {offset:#x}"
            );
        }
    }

    #[test]
    fn a_zero_offset_is_one_load_and_a_small_one_adds_a_single_lea() {
        let zero = rip_to_result(0, 0).1;
        let small = rip_to_result(0, 8).1;
        assert!(small > zero);
        assert!(
            rip_to_result(0, 1 << 40).1 > small,
            "an offset beyond i32 needs a scratch constant"
        );
    }

    #[test]
    fn the_page_base_slot_lies_above_the_helper_result_slot_and_inside_the_frame() {
        const {
            assert!(PAGE_BASE_SLOT >= OUT + 8);
            assert!(JUMP_CACHE_SLOT >= PAGE_BASE_SLOT + 8);
            assert!(VALID_SLOT >= JUMP_CACHE_SLOT + 8);
            assert!(VALID_SLOT + 8 <= FRAME);
            // The eight pushes leave rsp 8 mod 16, so calls from blocks see it 16-byte aligned.
            assert!((FRAME + 8 * 8) % 16 == 8);
        }
    }

    #[test]
    fn link_slots_are_padded_so_the_displacement_is_four_byte_aligned() {
        for padding in 0..8u64 {
            let memory = ExecMem::new(16384).unwrap();
            let insns: [Insn; 0] = [];
            let mut generator =
                Gen::new(memory.base() + padding, &DUMMY_ADDRESSES_WITH_CHAIN, &insns);
            generator.entry_rip = 0;
            for _ in 0..padding {
                generator.a.code.push(0x90);
            }
            generator.link_slot(0x10, None);
            let site = generator.links[0].code_offset as u64;
            assert_eq!(
                (memory.base() + padding + site + 1) % 4,
                0,
                "padding {padding}"
            );
            assert_eq!(&generator.a.code[site as usize..][..5], LINK_JMP);
        }
    }

    const DUMMY_ADDRESSES_WITH_CHAIN: Addrs = Addrs {
        exec: 0,
        ld: 0,
        st: 0,
        ld_stack: 0,
        st_stack: 0,
        cond: 0,
        whole_vector: 0,
        materialize_flags: 0,
        rdtsc: 0,
        pause: 0,
        exit_ok: 0,
        exit_fail: 0,
        chain: 1,
    };
}
