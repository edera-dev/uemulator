//! Minimal x86-64 assembler for the JIT backend.
//!
//! Only the instruction forms `emit_x64.rs` needs are encoded. Register operands are the
//! architectural 0..=15 numbers; the low three bits go into ModRM/SIB and bit 3 into a REX prefix.

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

/// Integer argument registers in order. Windows x64 and System V differ only here (and in the
/// shadow space the caller must reserve on Windows, see `emit_x64::FRAME`); the generated code
/// never passes more than four arguments, so the stack-passed tail never comes up.
#[cfg(windows)]
pub const ARGS: [u8; 4] = [RCX, RDX, R8, R9];
#[cfg(not(windows))]
pub const ARGS: [u8; 4] = [RDI, RSI, RDX, RCX];

/// Group-1 ALU operation numbers. They are both the ModRM.reg opcode extension of the immediate
/// forms (`0x81`/`0x83`) and bits 5:3 of the `op r/m, r` opcodes (`0x01`, `0x09`, ...).
pub const ADD: u8 = 0;
pub const OR: u8 = 1;
pub const ADC: u8 = 2;
pub const SBB: u8 = 3;
pub const AND: u8 = 4;
pub const SUB: u8 = 5;
pub const XOR: u8 = 6;
pub const CMP: u8 = 7;

/// Shift operation numbers: ModRM.reg opcode extension of `0xc1`/`0xd3`.
pub const SHL: u8 = 4;
pub const SHR: u8 = 5;
pub const SAR: u8 = 7;

/// Condition codes as encoded in `0f 8x` (Jcc rel32), `0f 9x` (setcc) and `0f 4x` (cmovcc). The
/// same numbers are the low nibble of the guest `70..7f` / `0f 80..8f` opcodes, so the JIT passes
/// guest condition codes straight through. Only the ones the generated code names itself are
/// listed.
pub const B: u8 = 2;
pub const AE: u8 = 3;
pub const E: u8 = 4;
pub const NE: u8 = 5;
pub const O: u8 = 0;

const REGISTER_COUNT: u8 = 16;

const REX_BASE: u8 = 0x40;
const REX_W: u8 = 1 << 3;
const REX_R: u8 = 1 << 2;
const REX_X: u8 = 1 << 1;
const REX_B: u8 = 1;

const MANDATORY_PREFIX_F3: u8 = 0xf3;
const TWO_BYTE_OPCODE: u8 = 0x0f;
const OPCODE_MOVDQU_LOAD: u8 = 0x6f;
const OPCODE_MOVDQU_STORE: u8 = 0x7f;
const OPCODE_PMOVMSKB: u8 = 0xd7;
const OPCODE_PSHUFD: u8 = 0x70;
const OPCODE_MOVQ_XMM_GPR: u8 = 0x6e;
/// `66 0f xx` integer SSE2 opcodes for `xmm_op`.
pub const OPCODE_PCMPEQB: u8 = 0x74;
pub const OPCODE_PAND: u8 = 0xdb;
pub const OPCODE_PXOR: u8 = 0xef;
pub const OPCODE_PANDN: u8 = 0xdf;
pub const OPCODE_POR: u8 = 0xeb;
pub const OPCODE_PCMPGTB: u8 = 0x64;
pub const OPCODE_PMINUB: u8 = 0xda;
pub const OPCODE_PMAXUB: u8 = 0xde;
/// `66 0f 6f` between two registers is `movdqa`.
pub const OPCODE_MOVDQA: u8 = 0x6f;
pub const OPCODE_PUNPCKLQDQ: u8 = 0x6c;
const OPERAND_SIZE_PREFIX: u8 = 0x66;

/// ModRM.mod values.
const MOD_NO_DISPLACEMENT: u8 = 0;
const MOD_DISP8: u8 = 1;
const MOD_DISP32: u8 = 2;
const MOD_REGISTER: u8 = 3;

/// ModRM.rm (or SIB.base, low three bits) value 4 means "a SIB byte follows", so rsp and r12
/// as base always need one.
const RM_NEEDS_SIB: u8 = 4;
/// With mod=00, base/rm value 5 means disp32 with no base (RIP-relative as rm), so rbp and r13
/// as base cannot use mod=00 and get a zero disp8 instead.
const BASE_DISP32_ONLY: u8 = 5;
/// SIB.index value 4 (without REX.X) means "no index".
const SIB_NO_INDEX: u8 = 4;

/// Low bit of an opcode pair: clear for the byte form, set for the 16/32/64-bit form.
const WIDE_FORM: u8 = 1;
/// Low bits of the `op r/m, r` ALU opcodes: the 64/32-bit "r/m is the destination" form.
const ALU_RM_REG: u8 = WIDE_FORM;

/// ModRM.reg extensions of the unary group (`f6`/`f7`) and the inc/dec group (`fe`/`ff`).
const UNARY_TEST: u8 = 0;
const UNARY_NOT: u8 = 2;
const UNARY_NEG: u8 = 3;
const UNARY_MUL: u8 = 4;
const UNARY_IMUL: u8 = 5;
const UNARY_INC: u8 = 0;
const UNARY_DEC: u8 = 1;
/// ModRM.reg extension of `bt r/m, imm8` (`0f ba`).
const BT_IMMEDIATE: u8 = 4;

/// A memory operand `[base + index << scale + disp]`.
#[derive(Clone, Copy)]
pub struct Mem {
    base: u8,
    /// `(register, log2 scale)`.
    index: Option<(u8, u8)>,
    disp: i32,
}

impl Mem {
    /// The same operand `bytes` further on.
    pub fn plus(self, bytes: i32) -> Mem {
        Mem {
            disp: self.disp + bytes,
            ..self
        }
    }
}

pub fn mem(base: u8, disp: i32) -> Mem {
    Mem {
        base,
        index: None,
        disp,
    }
}

/// `[base + index << scale + disp]`; `scale` is log2 (0..=3) and `index` must not be RSP, whose
/// SIB encoding means "no index" (R12 is fine, REX.X distinguishes it).
pub fn mem_idx(base: u8, index: u8, scale: u8, disp: i32) -> Mem {
    debug_assert!(index != RSP && scale <= 3);
    Mem {
        base,
        index: Some((index, scale)),
        disp,
    }
}

#[derive(Clone, Copy)]
pub struct Label(usize);

pub struct Asm {
    /// Address the finished code will execute at: `jmp_abs` and `here` are relative to it, so it
    /// must be the final location, not a staging buffer.
    base: u64,
    pub code: Vec<u8>,
    /// Bound offset into `code` per label.
    labels: Vec<Option<usize>>,
    /// `(offset of a rel32 field, label)`; resolved by `finish`.
    fixups: Vec<(usize, usize)>,
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            base,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
        }
    }

    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64
    }

    /// Address the first byte is assembled for.
    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.code.len());
    }

    /// Patches every label reference. All labels that were jumped to must be bound by now.
    pub fn finish(&mut self) {
        for &(pos, label) in &self.fixups {
            let target = self.labels[label].expect("unbound label");
            // rel32 is relative to the end of the instruction, and the field is always last.
            let rel = target as i64 - (pos as i64 + 4);
            self.code[pos..pos + 4].copy_from_slice(&(rel as i32).to_le_bytes());
        }
        self.fixups.clear();
    }

    /// Emits a REX prefix only when it carries a bit. `force_rex` is for byte operations on a
    /// register numbered 4..=7: without any REX they address ah/ch/dh/bh instead of spl/bpl/sil/dil.
    fn rex(&mut self, w: bool, reg: u8, index: u8, base: u8, force_rex: bool) {
        // Every encoder funnels through here (`mov_imm`, `alu_ri`, ...); a number of 16 or more
        // would silently alias register `n & 15` because only bit 3 is kept.
        debug_assert!(reg < REGISTER_COUNT && index < REGISTER_COUNT && base < REGISTER_COUNT);
        let mut rex = REX_BASE;
        if w {
            rex |= REX_W;
        }
        if reg & 8 != 0 {
            rex |= REX_R;
        }
        if index & 8 != 0 {
            rex |= REX_X;
        }
        if base & 8 != 0 {
            rex |= REX_B;
        }
        if rex != REX_BASE || force_rex {
            self.code.push(rex);
        }
    }

    fn modrm_mem(&mut self, reg: u8, m: Mem) {
        let base = m.base & 7;
        let sib = m.index.is_some() || base == RM_NEEDS_SIB;
        let mode = if m.disp == 0 && base != BASE_DISP32_ONLY {
            MOD_NO_DISPLACEMENT
        } else if i8::try_from(m.disp).is_ok() {
            MOD_DISP8
        } else {
            MOD_DISP32
        };
        self.code
            .push(mode << 6 | (reg & 7) << 3 | if sib { RM_NEEDS_SIB } else { base });
        if sib {
            let (index, scale) = m.index.map_or((SIB_NO_INDEX, 0), |(i, s)| (i & 7, s));
            self.code.push(scale << 6 | index << 3 | base);
        }
        match mode {
            MOD_DISP8 => self.code.push(m.disp as i8 as u8),
            MOD_DISP32 => self.code.extend_from_slice(&m.disp.to_le_bytes()),
            _ => {}
        }
    }

    /// `w` is REX.W (64-bit operand size). `reg` is the ModRM.reg field: a register or an
    /// opcode extension digit.
    fn op_rm(
        &mut self,
        operand_size_prefix: bool,
        w: bool,
        force_rex: bool,
        opcode: &[u8],
        reg: u8,
        m: Mem,
    ) {
        if operand_size_prefix {
            self.code.push(OPERAND_SIZE_PREFIX);
        }
        self.rex(w, reg, m.index.map_or(0, |i| i.0), m.base, force_rex);
        self.code.extend_from_slice(opcode);
        self.modrm_mem(reg, m);
    }

    fn op_rr(
        &mut self,
        operand_size_prefix: bool,
        w: bool,
        force_rex: bool,
        opcode: &[u8],
        reg: u8,
        rm: u8,
    ) {
        if operand_size_prefix {
            self.code.push(OPERAND_SIZE_PREFIX);
        }
        self.rex(w, reg, 0, rm, force_rex);
        self.code.extend_from_slice(opcode);
        self.code
            .push(MOD_REGISTER << 6 | (reg & 7) << 3 | (rm & 7));
    }

    /// With `w == false` this is a 32-bit move that zero-extends into the upper half, which is
    /// how the JIT truncates a value to 32 bits (also with `d == s`).
    pub fn mov_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x89], s, d);
    }

    /// Picks the shortest encoding. Deliberately a `mov` even for zero (never `xor`): emitted
    /// blocks load constants while host flags are still live for a following `jcc`.
    pub fn mov_imm(&mut self, r: u8, v: u64) {
        if v <= u32::MAX as u64 {
            // mov r32, imm32 zero-extends.
            self.rex(false, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else if i32::try_from(v as i64).is_ok() {
            // mov r/m64, imm32 sign-extends: covers the negative values 32-bit mov cannot.
            self.op_rr(false, true, false, &[0xc7], 0, r);
            self.code.extend_from_slice(&(v as u32).to_le_bytes());
        } else {
            // movabs r64, imm64
            self.rex(true, 0, 0, r, false);
            self.code.push(0xb8 | (r & 7));
            self.code.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// 64-bit load.
    pub fn ld(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8b], d, m);
    }

    /// Load `size` bytes, zero- or sign-extended to 64 bits. The 32-bit zero-extending form
    /// relies on the CPU clearing the upper half; `size` outside 1/2/4 means 8.
    pub fn ld_ext(&mut self, d: u8, m: Mem, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rm(false, false, false, &[0x0f, 0xb6], d, m),
            (1, true) => self.op_rm(false, true, false, &[0x0f, 0xbe], d, m),
            (2, false) => self.op_rm(false, false, false, &[0x0f, 0xb7], d, m),
            (2, true) => self.op_rm(false, true, false, &[0x0f, 0xbf], d, m),
            (4, false) => self.op_rm(false, false, false, &[0x8b], d, m),
            // movsxd
            (4, true) => self.op_rm(false, true, false, &[0x63], d, m),
            _ => self.op_rm(false, true, false, &[0x8b], d, m),
        }
    }

    /// Register-to-register counterpart of `ld_ext`. For `size == 1` the source is a byte
    /// register, so REX is forced to reach sil/dil.
    pub fn ext_rr(&mut self, d: u8, s: u8, size: u8, signed: bool) {
        match (size, signed) {
            (1, false) => self.op_rr(false, false, true, &[0x0f, 0xb6], d, s),
            (1, true) => self.op_rr(false, true, true, &[0x0f, 0xbe], d, s),
            (2, false) => self.op_rr(false, false, false, &[0x0f, 0xb7], d, s),
            (2, true) => self.op_rr(false, true, false, &[0x0f, 0xbf], d, s),
            (4, false) => self.mov_rr(d, s, false),
            (4, true) => self.op_rr(false, true, false, &[0x63], d, s),
            _ => self.mov_rr(d, s, true),
        }
    }

    /// Store the low `size` bytes of `s`; `size` outside 1/2/4 means 8. The byte form forces REX
    /// so that `s` = rsi/rdi means sil/dil rather than dh/bh.
    pub fn st(&mut self, m: Mem, s: u8, size: u8) {
        match size {
            1 => self.op_rm(false, false, true, &[0x88], s, m),
            2 => self.op_rm(true, false, false, &[0x89], s, m),
            4 => self.op_rm(false, false, false, &[0x89], s, m),
            _ => self.op_rm(false, true, false, &[0x89], s, m),
        }
    }

    pub fn st_imm16(&mut self, m: Mem, v: u16) {
        self.op_rm(true, false, false, &[0xc7], 0, m);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// `movdqu xmm, m` and `movdqu m, xmm`: 128-bit loads and stores with no alignment rule.
    pub fn xmm_ld(&mut self, x: u8, m: Mem) {
        self.sse_rm(MANDATORY_PREFIX_F3, OPCODE_MOVDQU_LOAD, x, m);
    }
    pub fn xmm_st(&mut self, m: Mem, x: u8) {
        self.sse_rm(MANDATORY_PREFIX_F3, OPCODE_MOVDQU_STORE, x, m);
    }

    /// A `66 0f xx` integer SSE2 instruction `d = d op s` on two xmm registers: pxor, pcmpeqb,
    /// paddq and the like, named by the `0f` opcode byte, which the guest instruction shares.
    pub fn xmm_op(&mut self, opcode: u8, d: u8, s: u8) {
        self.sse_rr(OPERAND_SIZE_PREFIX, opcode, d, s);
    }

    /// `movq xmm, r64`: the low 64 bits of `x` from a general register, the rest zeroed.
    pub fn movq_xmm_gpr(&mut self, x: u8, r: u8) {
        self.code.push(OPERAND_SIZE_PREFIX);
        self.rex(true, x, 0, r, false);
        self.code
            .extend_from_slice(&[TWO_BYTE_OPCODE, OPCODE_MOVQ_XMM_GPR]);
        self.code.push(MOD_REGISTER << 6 | (x & 7) << 3 | (r & 7));
    }

    /// `pmovmskb r32, xmm`.
    pub fn pmovmskb(&mut self, d: u8, x: u8) {
        self.sse_rr(OPERAND_SIZE_PREFIX, OPCODE_PMOVMSKB, d, x);
    }

    /// `pshufd d, s, order`: result dword `k` is dword `order[2k+1:2k]` of `s`.
    pub fn pshufd(&mut self, d: u8, s: u8, order: u8) {
        self.sse_rr(OPERAND_SIZE_PREFIX, OPCODE_PSHUFD, d, s);
        self.code.push(order);
    }

    /// The mandatory prefix comes before REX.
    fn sse_rm(&mut self, prefix: u8, opcode: u8, reg: u8, m: Mem) {
        self.code.push(prefix);
        self.rex(false, reg, m.index.map_or(0, |i| i.0), m.base, false);
        self.code.extend_from_slice(&[TWO_BYTE_OPCODE, opcode]);
        self.modrm_mem(reg, m);
    }

    fn sse_rr(&mut self, prefix: u8, opcode: u8, reg: u8, rm: u8) {
        self.code.push(prefix);
        self.rex(false, reg, 0, rm, false);
        self.code.extend_from_slice(&[TWO_BYTE_OPCODE, opcode]);
        self.code
            .push(MOD_REGISTER << 6 | (reg & 7) << 3 | (rm & 7));
    }

    /// Does not touch host flags, which is why the block generator computes effective addresses
    /// with it.
    pub fn lea(&mut self, d: u8, m: Mem) {
        self.op_rm(false, true, false, &[0x8d], d, m);
    }

    /// `op d, s` for ADD/OR/AND/SUB/XOR/CMP (the `op r/m, r` forms).
    pub fn alu_rr(&mut self, op: u8, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[op << 3 | ALU_RM_REG], s, d);
    }

    pub fn test_rr(&mut self, a: u8, b: u8, w: bool) {
        self.op_rr(false, w, false, &[0x85], b, a);
    }

    /// A register-to-register form that exists as an opcode pair (byte form `opcode`, wider form
    /// `opcode | 1`) at any operand `size`. The byte form forces REX so that rsi/rdi mean
    /// sil/dil; the 16-bit form carries the operand-size prefix.
    fn op_rr_sized(&mut self, size: u8, opcode: u8, reg: u8, rm: u8) {
        match size {
            1 => self.op_rr(false, false, true, &[opcode], reg, rm),
            2 => self.op_rr(true, false, false, &[opcode | WIDE_FORM], reg, rm),
            4 => self.op_rr(false, false, false, &[opcode | WIDE_FORM], reg, rm),
            _ => self.op_rr(false, true, false, &[opcode | WIDE_FORM], reg, rm),
        }
    }

    /// `op d, s` at operand `size` (1, 2, 4 or 8) for ADD/OR/AND/SUB/XOR/CMP. Only the low
    /// `size` bytes of the registers take part; a 4-byte result zero-extends, narrower ones
    /// leave the upper bits of `d` alone. The flags are those of an operation of that width.
    pub fn alu_rr_sized(&mut self, op: u8, d: u8, s: u8, size: u8) {
        self.op_rr_sized(size, op << 3, s, d);
    }

    /// `test r, imm32` (sign-extended with `w`) at 32 or 64 bits.
    pub fn test_ri(&mut self, r: u8, imm: i32, w: bool) {
        self.op_rr(false, w, false, &[0xf7], UNARY_TEST, r);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    pub fn not_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xf6, UNARY_NOT, r);
    }

    pub fn neg_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xf6, UNARY_NEG, r);
    }

    /// Leaves CF alone, like the guest instruction.
    pub fn inc_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xfe, UNARY_INC, r);
    }

    pub fn dec_r(&mut self, r: u8, size: u8) {
        self.op_rr_sized(size, 0xfe, UNARY_DEC, r);
    }

    /// `bt r, imm8`: CF = bit `bit` of `r`. The other flags are undefined.
    pub fn bt_ri(&mut self, r: u8, bit: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xba], BT_IMMEDIATE, r);
        self.code.push(bit);
    }

    /// `d = d * s` truncated, signed. CF and OF are set when the full product does not fit; the
    /// other flags are undefined.
    pub fn imul_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xaf], d, s);
    }

    /// `shld d, s, count` (`left`) or `shrd d, s, count`: shifts `d` by `count` (1 to the width
    /// minus 1) and fills the vacated bits from `s`. CF is the last bit shifted out; OF is defined
    /// only for a count of 1.
    pub fn double_shift_ri(&mut self, left: bool, d: u8, s: u8, count: u8, w: bool) {
        let opcode = if left { 0xa4 } else { 0xac };
        self.op_rr(false, w, false, &[0x0f, opcode], s, d);
        self.code.push(count);
    }

    /// One-operand `mul r`: rDX:rAX = rAX * r (unsigned; EDX:EAX without `w`, both zero-extended).
    /// CF = OF = the high half is not zero; the other flags are undefined.
    pub fn mul_r(&mut self, r: u8, w: bool) {
        self.op_rr(false, w, false, &[0xf7], UNARY_MUL, r);
    }

    /// One-operand `imul r`: the signed `mul_r`. CF = OF = the product does not fit the low half.
    pub fn imul_r(&mut self, r: u8, w: bool) {
        self.op_rr(false, w, false, &[0xf7], UNARY_IMUL, r);
    }

    /// `d = index of the lowest set bit of s`; ZF is set (and `d` undefined) when `s` is zero.
    pub fn bsf_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xbc], d, s);
    }

    /// `d = index of the highest set bit of s`; ZF is set (and `d` undefined) when `s` is zero.
    pub fn bsr_rr(&mut self, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0xbd], d, s);
    }

    /// `if cc { d = s }`. With a 32-bit operand `d` is zero-extended whether or not it moved.
    pub fn cmov_rr(&mut self, cc: u8, d: u8, s: u8, w: bool) {
        self.op_rr(false, w, false, &[0x0f, 0x40 | cc], d, s);
    }

    /// Shift `r` by `cl`; the CPU masks the count to 5 bits, or 6 with `w`.
    pub fn shift_cl(&mut self, op: u8, r: u8, w: bool) {
        self.op_rr(false, w, false, &[0xd3], op, r);
    }

    /// `edx:eax` sign extension (`cdq`) or `rdx:rax` (`cqo`). Does not touch flags.
    pub fn cdq(&mut self, w: bool) {
        if w {
            self.code.push(REX_BASE | REX_W);
        }
        self.code.push(0x99);
    }

    /// Uses the sign-extended imm8 form when the value fits. With `w` the imm32 is
    /// sign-extended to 64 bits.
    pub fn alu_ri(&mut self, op: u8, r: u8, imm: i32, w: bool) {
        if let Ok(imm8) = i8::try_from(imm) {
            self.op_rr(false, w, false, &[0x83], op, r);
            self.code.push(imm8 as u8);
        } else {
            self.op_rr(false, w, false, &[0x81], op, r);
            self.code.extend_from_slice(&imm.to_le_bytes());
        }
    }

    pub fn shift_ri(&mut self, op: u8, r: u8, count: u8, w: bool) {
        self.op_rr(false, w, false, &[0xc1], op, r);
        self.code.push(count);
    }

    /// 64-bit `cmp [m], r`.
    pub fn cmp_mr(&mut self, m: Mem, r: u8) {
        self.op_rm(false, true, false, &[0x39], r, m);
    }

    pub fn cmp_m8_imm(&mut self, m: Mem, v: u8) {
        self.op_rm(false, false, false, &[0x80], CMP, m);
        self.code.push(v);
    }

    /// Byte register `r` is set to 1 if condition `cc` holds, else 0; the rest of the register is
    /// left alone, and so are the flags. REX is forced for sil/dil.
    pub fn setcc(&mut self, cc: u8, r: u8) {
        self.op_rr(false, false, true, &[0x0f, 0x90 | cc], 0, r);
    }

    pub fn sete(&mut self, r: u8) {
        self.setcc(E, r);
    }

    /// Always the rel32 form so label distances never need relaxation.
    pub fn jcc(&mut self, cc: u8, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80 | cc]);
        self.fixups.push((self.code.len(), label.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    pub fn jmp(&mut self, label: Label) {
        self.code.push(0xe9);
        self.fixups.push((self.code.len(), label.0));
        self.code.extend_from_slice(&[0; 4]);
    }

    /// Jump to an absolute address within rel32 range (the code buffer is far smaller than 2 GiB).
    pub fn jmp_abs(&mut self, target: u64) {
        let rel = target as i64 - (self.here() as i64 + 5);
        self.code.push(0xe9);
        self.code
            .extend_from_slice(&i32::try_from(rel).expect("jump out of range").to_le_bytes());
    }

    /// REX.B (if `r >= 8`) is the only prefix needed: indirect call/jump default to 64-bit.
    pub fn call_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 2, r);
    }

    pub fn jmp_reg(&mut self, r: u8) {
        self.op_rr(false, false, false, &[0xff], 4, r);
    }

    pub fn push(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x50 | (r & 7));
    }

    pub fn pop(&mut self, r: u8) {
        self.rex(false, 0, 0, r, false);
        self.code.push(0x58 | (r & 7));
    }

    pub fn ret(&mut self) {
        self.code.push(0xc3);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        a.mov_rr(RAX, ARGS[0], true);
        a.mov_rr(RCX, ARGS[1], true);
        f(&mut a);
        a.ret();
        a.finish();
        m.write_bytes(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.alu_rr(ADD, RAX, RCX, true), 40, 2), 42);
        assert_eq!(run(|a| a.alu_rr(SUB, RAX, RCX, true), 40, 2), 38);
        assert_eq!(run(|a| a.alu_rr(AND, RAX, RCX, true), 0xff0, 0x0ff), 0x0f0);
        assert_eq!(run(|a| a.alu_rr(OR, RAX, RCX, true), 0xf0, 0x0f), 0xff);
        assert_eq!(run(|a| a.alu_rr(XOR, RAX, RCX, true), 0xff, 0x0f), 0xf0);
        assert_eq!(
            run(|a| a.alu_rr(ADD, RAX, RCX, false), 0xffff_ffff, 2),
            1,
            "32-bit ops zero-extend"
        );
        assert_eq!(run(|a| a.alu_ri(ADD, RAX, 5, true), 1, 0), 6);
        assert_eq!(run(|a| a.alu_ri(SUB, RAX, 0x1000, true), 0x1001, 0), 1);
        assert_eq!(run(|a| a.shift_ri(SHL, RAX, 4, true), 3, 0), 48);
        assert_eq!(run(|a| a.shift_ri(SHR, RAX, 4, true), 0xf0, 0), 0xf);
        assert_eq!(run(|a| a.lea(RAX, mem_idx(RAX, RCX, 3, -2)), 100, 5), 138);
    }

    #[test]
    fn immediates_and_high_registers() {
        for &v in &[
            0u64,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0xffff_ffff_ffff_fff0,
            0x1234_5678_9abc_def0,
        ] {
            assert_eq!(run(|a| a.mov_imm(RAX, v), 0, 0), v, "{v:#x}");
            assert_eq!(
                run(
                    |a| {
                        a.mov_imm(R9, v);
                        a.mov_rr(RAX, R9, true);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r9"
            );
            assert_eq!(
                run(
                    |a| {
                        a.push(R13);
                        a.mov_imm(R13, v);
                        a.mov_rr(RAX, R13, true);
                        a.pop(R13);
                    },
                    0,
                    0
                ),
                v,
                "{v:#x} via r13"
            );
        }
    }

    #[test]
    fn loads_and_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(RDX, 0xdead_beef_1234_5678);
                a.st(mem(RAX, 8), RDX, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        let ld = |size, signed, off: i32| {
            run(
                move |a| {
                    a.mov_imm(RDX, off as u64);
                    a.ld_ext(RAX, mem_idx(RAX, RDX, 0, 0), size, signed);
                },
                p,
                0,
            )
        };
        assert_eq!(ld(8, false, 8), 0xdead_beef_1234_5678);
        assert_eq!(ld(4, false, 8), 0x1234_5678);
        assert_eq!(ld(4, true, 12), 0xffff_ffff_dead_beef);
        assert_eq!(ld(2, false, 8), 0x5678);
        assert_eq!(ld(2, true, 12), 0xffff_ffff_ffff_beef);
        assert_eq!(ld(1, false, 8), 0x78);
        assert_eq!(ld(1, true, 15), 0xffff_ffff_ffff_ffde);
        for (size, mask) in [(1u8, 0xffu64), (2, 0xffff), (4, 0xffff_ffff)] {
            buf[2] = !0;
            run(
                |a| {
                    a.mov_imm(R9, 0x1111_2222_3333_4444);
                    a.mov_imm(RDI, 16);
                    a.st(mem_idx(RAX, RDI, 0, 0), R9, size);
                },
                p,
                0,
            );
            assert_eq!(
                buf[2],
                !mask | (0x1111_2222_3333_4444 & mask),
                "size {size}"
            );
        }
    }

    #[test]
    fn base_register_encodings() {
        let mut buf = [0u64; 4];
        buf[1] = 0x42;
        let p = buf.as_mut_ptr() as u64;
        for base in [RBX, RBP, RSI, R8, R12, R13, R15] {
            let got = run(
                |a| {
                    a.push(base);
                    a.mov_rr(base, RAX, true);
                    a.ld(RAX, mem(base, 8));
                    a.pop(base);
                },
                p,
                0,
            );
            assert_eq!(got, 0x42, "base {base}");
        }
        assert_eq!(
            run(
                |a| {
                    a.alu_ri(SUB, RSP, 16, true);
                    a.st(mem(RSP, 8), RAX, 8);
                    a.ld(RAX, mem(RSP, 8));
                    a.alu_ri(ADD, RSP, 16, true);
                },
                7,
                0
            ),
            7
        );
    }

    #[test]
    fn extensions() {
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, true), 0, 0x80), !0x7f);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 1, false), !0, 0x1ff), 0xff);
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 2, true), 0, 0x8000), !0x7fff);
        assert_eq!(
            run(|a| a.ext_rr(RAX, RCX, 4, true), 0, 0x8000_0000),
            !0x7fff_ffff
        );
        assert_eq!(run(|a| a.ext_rr(RAX, RCX, 4, false), 0, !0), 0xffff_ffff);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(RSI, 0x1ff);
                    a.ext_rr(RAX, RSI, 1, false);
                },
                0,
                0
            ),
            0xff
        );
    }

    #[test]
    fn branches_and_flags() {
        const COND_L: u8 = 0xc;
        let r = run(
            |a| {
                let l = a.label();
                a.alu_rr(CMP, RAX, RCX, true);
                a.mov_imm(RAX, 1);
                a.jcc(COND_L, l);
                a.mov_imm(RAX, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        let r = run(
            |a| {
                a.alu_rr(ADD, RAX, RCX, true);
                a.sete(RDX);
                a.mov_imm(RAX, 0);
                a.ext_rr(RAX, RDX, 1, false);
            },
            u64::MAX,
            1,
        );
        assert_eq!(r, 1);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod encoding_tests {
    //! Encoder tests for the x86-64 assembler. The expected bytes come from `clang -target x86_64`
    //! (the instruction is named in each case), covering the registers and addressing forms whose
    //! encodings are special: rsp/r12 as a base need a SIB byte, rbp/r13 need a displacement, byte
    //! stores from sil/dil/spl/bpl need a REX prefix, and r8..r15 set the REX extension bits.

    use super::*;
    use crate::sys::ExecMem;

    const NOP_BYTE: u8 = 0x90;
    const BASE_ADDRESS: u64 = 0x1000_0000;

    /// An assembly line, the encoder call it must match, and the bytes the assembler produced.
    type EncodeCase = (&'static str, fn(&mut Asm), &'static [u8]);

    fn nop(a: &mut Asm) {
        a.code.push(NOP_BYTE);
    }

    fn assemble(build: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut a = Asm::new(BASE_ADDRESS);
        build(&mut a);
        a.finish();
        a.code
    }

    /// The encoder always puts a REX prefix on byte-sized operations so sil/dil/spl/bpl work, even
    /// when the registers are al..bl and an assembler would leave it out. An empty REX (0x40) is
    /// harmless, so cases compare with it removed from both sides; tests of the byte registers that
    /// need it check the prefix explicitly.
    fn without_empty_rex(bytes: &[u8]) -> Vec<u8> {
        let rex_position = usize::from(bytes.first() == Some(&0x66));
        let mut stripped = bytes.to_vec();
        if stripped.get(rex_position) == Some(&0x40) {
            stripped.remove(rex_position);
        }
        stripped
    }

    #[track_caller]
    fn check_encodings(cases: &[EncodeCase]) {
        for (assembly, encode, expected) in cases {
            let mut a = Asm::new(BASE_ADDRESS);
            encode(&mut a);
            assert_eq!(
                without_empty_rex(&a.code),
                without_empty_rex(expected),
                "`{assembly}`"
            );
        }
    }

    /// 64-bit loads with the base registers whose encodings are special (rsp/r12 need a SIB byte, rbp/r13 a displacement), every displacement width and scaled indexes.
#[rustfmt::skip]
#[test]
fn loads_of_every_base_and_index_shape_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("mov rax, qword ptr [rax]", |a| { a.ld(RAX, mem(RAX, 0)); }, &[0x48, 0x8b, 0x00]),
        ("mov rax, qword ptr [rax + 1]", |a| { a.ld(RAX, mem(RAX, 1)); }, &[0x48, 0x8b, 0x40, 0x01]),
        ("mov rax, qword ptr [rax - 1]", |a| { a.ld(RAX, mem(RAX, -1)); }, &[0x48, 0x8b, 0x40, 0xff]),
        ("mov rax, qword ptr [rax + 127]", |a| { a.ld(RAX, mem(RAX, 127)); }, &[0x48, 0x8b, 0x40, 0x7f]),
        ("mov rax, qword ptr [rax + 128]", |a| { a.ld(RAX, mem(RAX, 128)); }, &[0x48, 0x8b, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [rax - 128]", |a| { a.ld(RAX, mem(RAX, -128)); }, &[0x48, 0x8b, 0x40, 0x80]),
        ("mov rax, qword ptr [rax - 129]", |a| { a.ld(RAX, mem(RAX, -129)); }, &[0x48, 0x8b, 0x80, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [rax + 305419896]", |a| { a.ld(RAX, mem(RAX, 305419896)); }, &[0x48, 0x8b, 0x80, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [rax - 305419896]", |a| { a.ld(RAX, mem(RAX, -305419896)); }, &[0x48, 0x8b, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [rsp]", |a| { a.ld(RAX, mem(RSP, 0)); }, &[0x48, 0x8b, 0x04, 0x24]),
        ("mov rax, qword ptr [rsp + 1]", |a| { a.ld(RAX, mem(RSP, 1)); }, &[0x48, 0x8b, 0x44, 0x24, 0x01]),
        ("mov rax, qword ptr [rsp - 1]", |a| { a.ld(RAX, mem(RSP, -1)); }, &[0x48, 0x8b, 0x44, 0x24, 0xff]),
        ("mov rax, qword ptr [rsp + 127]", |a| { a.ld(RAX, mem(RSP, 127)); }, &[0x48, 0x8b, 0x44, 0x24, 0x7f]),
        ("mov rax, qword ptr [rsp + 128]", |a| { a.ld(RAX, mem(RSP, 128)); }, &[0x48, 0x8b, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp - 128]", |a| { a.ld(RAX, mem(RSP, -128)); }, &[0x48, 0x8b, 0x44, 0x24, 0x80]),
        ("mov rax, qword ptr [rsp - 129]", |a| { a.ld(RAX, mem(RSP, -129)); }, &[0x48, 0x8b, 0x84, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [rsp + 305419896]", |a| { a.ld(RAX, mem(RSP, 305419896)); }, &[0x48, 0x8b, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [rsp - 305419896]", |a| { a.ld(RAX, mem(RSP, -305419896)); }, &[0x48, 0x8b, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [rbp]", |a| { a.ld(RAX, mem(RBP, 0)); }, &[0x48, 0x8b, 0x45, 0x00]),
        ("mov rax, qword ptr [rbp + 1]", |a| { a.ld(RAX, mem(RBP, 1)); }, &[0x48, 0x8b, 0x45, 0x01]),
        ("mov rax, qword ptr [rbp - 1]", |a| { a.ld(RAX, mem(RBP, -1)); }, &[0x48, 0x8b, 0x45, 0xff]),
        ("mov rax, qword ptr [rbp + 127]", |a| { a.ld(RAX, mem(RBP, 127)); }, &[0x48, 0x8b, 0x45, 0x7f]),
        ("mov rax, qword ptr [rbp + 128]", |a| { a.ld(RAX, mem(RBP, 128)); }, &[0x48, 0x8b, 0x85, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp - 128]", |a| { a.ld(RAX, mem(RBP, -128)); }, &[0x48, 0x8b, 0x45, 0x80]),
        ("mov rax, qword ptr [rbp - 129]", |a| { a.ld(RAX, mem(RBP, -129)); }, &[0x48, 0x8b, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [rbp + 305419896]", |a| { a.ld(RAX, mem(RBP, 305419896)); }, &[0x48, 0x8b, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [rbp - 305419896]", |a| { a.ld(RAX, mem(RBP, -305419896)); }, &[0x48, 0x8b, 0x85, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [r8]", |a| { a.ld(RAX, mem(R8, 0)); }, &[0x49, 0x8b, 0x00]),
        ("mov rax, qword ptr [r8 + 1]", |a| { a.ld(RAX, mem(R8, 1)); }, &[0x49, 0x8b, 0x40, 0x01]),
        ("mov rax, qword ptr [r8 - 1]", |a| { a.ld(RAX, mem(R8, -1)); }, &[0x49, 0x8b, 0x40, 0xff]),
        ("mov rax, qword ptr [r8 + 127]", |a| { a.ld(RAX, mem(R8, 127)); }, &[0x49, 0x8b, 0x40, 0x7f]),
        ("mov rax, qword ptr [r8 + 128]", |a| { a.ld(RAX, mem(R8, 128)); }, &[0x49, 0x8b, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [r8 - 128]", |a| { a.ld(RAX, mem(R8, -128)); }, &[0x49, 0x8b, 0x40, 0x80]),
        ("mov rax, qword ptr [r8 - 129]", |a| { a.ld(RAX, mem(R8, -129)); }, &[0x49, 0x8b, 0x80, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [r8 + 305419896]", |a| { a.ld(RAX, mem(R8, 305419896)); }, &[0x49, 0x8b, 0x80, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [r8 - 305419896]", |a| { a.ld(RAX, mem(R8, -305419896)); }, &[0x49, 0x8b, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [r12]", |a| { a.ld(RAX, mem(R12, 0)); }, &[0x49, 0x8b, 0x04, 0x24]),
        ("mov rax, qword ptr [r12 + 1]", |a| { a.ld(RAX, mem(R12, 1)); }, &[0x49, 0x8b, 0x44, 0x24, 0x01]),
        ("mov rax, qword ptr [r12 - 1]", |a| { a.ld(RAX, mem(R12, -1)); }, &[0x49, 0x8b, 0x44, 0x24, 0xff]),
        ("mov rax, qword ptr [r12 + 127]", |a| { a.ld(RAX, mem(R12, 127)); }, &[0x49, 0x8b, 0x44, 0x24, 0x7f]),
        ("mov rax, qword ptr [r12 + 128]", |a| { a.ld(RAX, mem(R12, 128)); }, &[0x49, 0x8b, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 - 128]", |a| { a.ld(RAX, mem(R12, -128)); }, &[0x49, 0x8b, 0x44, 0x24, 0x80]),
        ("mov rax, qword ptr [r12 - 129]", |a| { a.ld(RAX, mem(R12, -129)); }, &[0x49, 0x8b, 0x84, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [r12 + 305419896]", |a| { a.ld(RAX, mem(R12, 305419896)); }, &[0x49, 0x8b, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [r12 - 305419896]", |a| { a.ld(RAX, mem(R12, -305419896)); }, &[0x49, 0x8b, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [r13]", |a| { a.ld(RAX, mem(R13, 0)); }, &[0x49, 0x8b, 0x45, 0x00]),
        ("mov rax, qword ptr [r13 + 1]", |a| { a.ld(RAX, mem(R13, 1)); }, &[0x49, 0x8b, 0x45, 0x01]),
        ("mov rax, qword ptr [r13 - 1]", |a| { a.ld(RAX, mem(R13, -1)); }, &[0x49, 0x8b, 0x45, 0xff]),
        ("mov rax, qword ptr [r13 + 127]", |a| { a.ld(RAX, mem(R13, 127)); }, &[0x49, 0x8b, 0x45, 0x7f]),
        ("mov rax, qword ptr [r13 + 128]", |a| { a.ld(RAX, mem(R13, 128)); }, &[0x49, 0x8b, 0x85, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 - 128]", |a| { a.ld(RAX, mem(R13, -128)); }, &[0x49, 0x8b, 0x45, 0x80]),
        ("mov rax, qword ptr [r13 - 129]", |a| { a.ld(RAX, mem(R13, -129)); }, &[0x49, 0x8b, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [r13 + 305419896]", |a| { a.ld(RAX, mem(R13, 305419896)); }, &[0x49, 0x8b, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [r13 - 305419896]", |a| { a.ld(RAX, mem(R13, -305419896)); }, &[0x49, 0x8b, 0x85, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [r15]", |a| { a.ld(RAX, mem(R15, 0)); }, &[0x49, 0x8b, 0x07]),
        ("mov rax, qword ptr [r15 + 1]", |a| { a.ld(RAX, mem(R15, 1)); }, &[0x49, 0x8b, 0x47, 0x01]),
        ("mov rax, qword ptr [r15 - 1]", |a| { a.ld(RAX, mem(R15, -1)); }, &[0x49, 0x8b, 0x47, 0xff]),
        ("mov rax, qword ptr [r15 + 127]", |a| { a.ld(RAX, mem(R15, 127)); }, &[0x49, 0x8b, 0x47, 0x7f]),
        ("mov rax, qword ptr [r15 + 128]", |a| { a.ld(RAX, mem(R15, 128)); }, &[0x49, 0x8b, 0x87, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 - 128]", |a| { a.ld(RAX, mem(R15, -128)); }, &[0x49, 0x8b, 0x47, 0x80]),
        ("mov rax, qword ptr [r15 - 129]", |a| { a.ld(RAX, mem(R15, -129)); }, &[0x49, 0x8b, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [r15 + 305419896]", |a| { a.ld(RAX, mem(R15, 305419896)); }, &[0x49, 0x8b, 0x87, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [r15 - 305419896]", |a| { a.ld(RAX, mem(R15, -305419896)); }, &[0x49, 0x8b, 0x87, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [rbx]", |a| { a.ld(RAX, mem(RBX, 0)); }, &[0x48, 0x8b, 0x03]),
        ("mov rax, qword ptr [rbx + 1]", |a| { a.ld(RAX, mem(RBX, 1)); }, &[0x48, 0x8b, 0x43, 0x01]),
        ("mov rax, qword ptr [rbx - 1]", |a| { a.ld(RAX, mem(RBX, -1)); }, &[0x48, 0x8b, 0x43, 0xff]),
        ("mov rax, qword ptr [rbx + 127]", |a| { a.ld(RAX, mem(RBX, 127)); }, &[0x48, 0x8b, 0x43, 0x7f]),
        ("mov rax, qword ptr [rbx + 128]", |a| { a.ld(RAX, mem(RBX, 128)); }, &[0x48, 0x8b, 0x83, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [rbx - 128]", |a| { a.ld(RAX, mem(RBX, -128)); }, &[0x48, 0x8b, 0x43, 0x80]),
        ("mov rax, qword ptr [rbx - 129]", |a| { a.ld(RAX, mem(RBX, -129)); }, &[0x48, 0x8b, 0x83, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [rbx + 305419896]", |a| { a.ld(RAX, mem(RBX, 305419896)); }, &[0x48, 0x8b, 0x83, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [rbx - 305419896]", |a| { a.ld(RAX, mem(RBX, -305419896)); }, &[0x48, 0x8b, 0x83, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [rax + rax*1]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x00]),
        ("mov rax, qword ptr [rax + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x00, 0x08]),
        ("mov rax, qword ptr [rax + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rax*8]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xc0]),
        ("mov rax, qword ptr [rax + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xc0, 0x08]),
        ("mov rax, qword ptr [rax + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RAX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rcx*1]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x08]),
        ("mov rax, qword ptr [rax + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x08, 0x08]),
        ("mov rax, qword ptr [rax + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x08, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rcx*8]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xc8]),
        ("mov rax, qword ptr [rax + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xc8, 0x08]),
        ("mov rax, qword ptr [rax + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RCX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rbp*1]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x28]),
        ("mov rax, qword ptr [rax + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x28, 0x08]),
        ("mov rax, qword ptr [rax + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rbp*8]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xe8]),
        ("mov rax, qword ptr [rax + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xe8, 0x08]),
        ("mov rax, qword ptr [rax + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RBP, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rsi*1]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x30]),
        ("mov rax, qword ptr [rax + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x30, 0x08]),
        ("mov rax, qword ptr [rax + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x30, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + rsi*8]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xf0]),
        ("mov rax, qword ptr [rax + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xf0, 0x08]),
        ("mov rax, qword ptr [rax + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, RSI, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r8*1]", |a| { a.ld(RAX, mem_idx(RAX, R8, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x00]),
        ("mov rax, qword ptr [rax + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R8, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x00, 0x08]),
        ("mov rax, qword ptr [rax + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R8, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r8*8]", |a| { a.ld(RAX, mem_idx(RAX, R8, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xc0]),
        ("mov rax, qword ptr [rax + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R8, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xc0, 0x08]),
        ("mov rax, qword ptr [rax + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R8, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r12*1]", |a| { a.ld(RAX, mem_idx(RAX, R12, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x20]),
        ("mov rax, qword ptr [rax + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R12, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x20, 0x08]),
        ("mov rax, qword ptr [rax + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R12, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x20, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r12*8]", |a| { a.ld(RAX, mem_idx(RAX, R12, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xe0]),
        ("mov rax, qword ptr [rax + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R12, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xe0, 0x08]),
        ("mov rax, qword ptr [rax + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R12, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r13*1]", |a| { a.ld(RAX, mem_idx(RAX, R13, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x28]),
        ("mov rax, qword ptr [rax + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R13, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x28, 0x08]),
        ("mov rax, qword ptr [rax + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R13, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r13*8]", |a| { a.ld(RAX, mem_idx(RAX, R13, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xe8]),
        ("mov rax, qword ptr [rax + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R13, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xe8, 0x08]),
        ("mov rax, qword ptr [rax + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R13, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r15*1]", |a| { a.ld(RAX, mem_idx(RAX, R15, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x38]),
        ("mov rax, qword ptr [rax + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R15, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x38, 0x08]),
        ("mov rax, qword ptr [rax + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R15, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x38, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rax + r15*8]", |a| { a.ld(RAX, mem_idx(RAX, R15, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xf8]),
        ("mov rax, qword ptr [rax + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(RAX, R15, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xf8, 0x08]),
        ("mov rax, qword ptr [rax + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(RAX, R15, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rax*1]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [rsp + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x04, 0x08]),
        ("mov rax, qword ptr [rsp + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rax*8]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xc4]),
        ("mov rax, qword ptr [rsp + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xc4, 0x08]),
        ("mov rax, qword ptr [rsp + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RAX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rcx*1]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x0c]),
        ("mov rax, qword ptr [rsp + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x0c, 0x08]),
        ("mov rax, qword ptr [rsp + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rcx*8]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xcc]),
        ("mov rax, qword ptr [rsp + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xcc, 0x08]),
        ("mov rax, qword ptr [rsp + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RCX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rbp*1]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x2c]),
        ("mov rax, qword ptr [rsp + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x2c, 0x08]),
        ("mov rax, qword ptr [rsp + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rbp*8]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xec]),
        ("mov rax, qword ptr [rsp + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xec, 0x08]),
        ("mov rax, qword ptr [rsp + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RBP, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rsi*1]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 0, 0)); }, &[0x48, 0x8b, 0x04, 0x34]),
        ("mov rax, qword ptr [rsp + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x34, 0x08]),
        ("mov rax, qword ptr [rsp + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + rsi*8]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 3, 0)); }, &[0x48, 0x8b, 0x04, 0xf4]),
        ("mov rax, qword ptr [rsp + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xf4, 0x08]),
        ("mov rax, qword ptr [rsp + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, RSI, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r8*1]", |a| { a.ld(RAX, mem_idx(RSP, R8, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [rsp + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R8, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x04, 0x08]),
        ("mov rax, qword ptr [rsp + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R8, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r8*8]", |a| { a.ld(RAX, mem_idx(RSP, R8, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xc4]),
        ("mov rax, qword ptr [rsp + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R8, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xc4, 0x08]),
        ("mov rax, qword ptr [rsp + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R8, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r12*1]", |a| { a.ld(RAX, mem_idx(RSP, R12, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x24]),
        ("mov rax, qword ptr [rsp + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R12, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x24, 0x08]),
        ("mov rax, qword ptr [rsp + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R12, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r12*8]", |a| { a.ld(RAX, mem_idx(RSP, R12, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xe4]),
        ("mov rax, qword ptr [rsp + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R12, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xe4, 0x08]),
        ("mov rax, qword ptr [rsp + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R12, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r13*1]", |a| { a.ld(RAX, mem_idx(RSP, R13, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x2c]),
        ("mov rax, qword ptr [rsp + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R13, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x2c, 0x08]),
        ("mov rax, qword ptr [rsp + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R13, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r13*8]", |a| { a.ld(RAX, mem_idx(RSP, R13, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xec]),
        ("mov rax, qword ptr [rsp + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R13, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xec, 0x08]),
        ("mov rax, qword ptr [rsp + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R13, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r15*1]", |a| { a.ld(RAX, mem_idx(RSP, R15, 0, 0)); }, &[0x4a, 0x8b, 0x04, 0x3c]),
        ("mov rax, qword ptr [rsp + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R15, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x3c, 0x08]),
        ("mov rax, qword ptr [rsp + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R15, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rsp + r15*8]", |a| { a.ld(RAX, mem_idx(RSP, R15, 3, 0)); }, &[0x4a, 0x8b, 0x04, 0xfc]),
        ("mov rax, qword ptr [rsp + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(RSP, R15, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xfc, 0x08]),
        ("mov rax, qword ptr [rsp + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(RSP, R15, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rax*1]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 0, 0)); }, &[0x48, 0x8b, 0x44, 0x05, 0x00]),
        ("mov rax, qword ptr [rbp + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x05, 0x08]),
        ("mov rax, qword ptr [rbp + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rax*8]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 3, 0)); }, &[0x48, 0x8b, 0x44, 0xc5, 0x00]),
        ("mov rax, qword ptr [rbp + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xc5, 0x08]),
        ("mov rax, qword ptr [rbp + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RAX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rcx*1]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 0, 0)); }, &[0x48, 0x8b, 0x44, 0x0d, 0x00]),
        ("mov rax, qword ptr [rbp + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x0d, 0x08]),
        ("mov rax, qword ptr [rbp + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rcx*8]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 3, 0)); }, &[0x48, 0x8b, 0x44, 0xcd, 0x00]),
        ("mov rax, qword ptr [rbp + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xcd, 0x08]),
        ("mov rax, qword ptr [rbp + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rbp*1]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 0, 0)); }, &[0x48, 0x8b, 0x44, 0x2d, 0x00]),
        ("mov rax, qword ptr [rbp + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x2d, 0x08]),
        ("mov rax, qword ptr [rbp + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rbp*8]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 3, 0)); }, &[0x48, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [rbp + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xed, 0x08]),
        ("mov rax, qword ptr [rbp + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RBP, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rsi*1]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 0, 0)); }, &[0x48, 0x8b, 0x44, 0x35, 0x00]),
        ("mov rax, qword ptr [rbp + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 0, 8)); }, &[0x48, 0x8b, 0x44, 0x35, 0x08]),
        ("mov rax, qword ptr [rbp + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 0, 4096)); }, &[0x48, 0x8b, 0x84, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + rsi*8]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 3, 0)); }, &[0x48, 0x8b, 0x44, 0xf5, 0x00]),
        ("mov rax, qword ptr [rbp + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 3, 8)); }, &[0x48, 0x8b, 0x44, 0xf5, 0x08]),
        ("mov rax, qword ptr [rbp + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, RSI, 3, 4096)); }, &[0x48, 0x8b, 0x84, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r8*1]", |a| { a.ld(RAX, mem_idx(RBP, R8, 0, 0)); }, &[0x4a, 0x8b, 0x44, 0x05, 0x00]),
        ("mov rax, qword ptr [rbp + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R8, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x05, 0x08]),
        ("mov rax, qword ptr [rbp + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R8, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r8*8]", |a| { a.ld(RAX, mem_idx(RBP, R8, 3, 0)); }, &[0x4a, 0x8b, 0x44, 0xc5, 0x00]),
        ("mov rax, qword ptr [rbp + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R8, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xc5, 0x08]),
        ("mov rax, qword ptr [rbp + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R8, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r12*1]", |a| { a.ld(RAX, mem_idx(RBP, R12, 0, 0)); }, &[0x4a, 0x8b, 0x44, 0x25, 0x00]),
        ("mov rax, qword ptr [rbp + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R12, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x25, 0x08]),
        ("mov rax, qword ptr [rbp + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R12, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r12*8]", |a| { a.ld(RAX, mem_idx(RBP, R12, 3, 0)); }, &[0x4a, 0x8b, 0x44, 0xe5, 0x00]),
        ("mov rax, qword ptr [rbp + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R12, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xe5, 0x08]),
        ("mov rax, qword ptr [rbp + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R12, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r13*1]", |a| { a.ld(RAX, mem_idx(RBP, R13, 0, 0)); }, &[0x4a, 0x8b, 0x44, 0x2d, 0x00]),
        ("mov rax, qword ptr [rbp + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R13, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x2d, 0x08]),
        ("mov rax, qword ptr [rbp + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R13, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r13*8]", |a| { a.ld(RAX, mem_idx(RBP, R13, 3, 0)); }, &[0x4a, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [rbp + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R13, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xed, 0x08]),
        ("mov rax, qword ptr [rbp + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R13, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r15*1]", |a| { a.ld(RAX, mem_idx(RBP, R15, 0, 0)); }, &[0x4a, 0x8b, 0x44, 0x3d, 0x00]),
        ("mov rax, qword ptr [rbp + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R15, 0, 8)); }, &[0x4a, 0x8b, 0x44, 0x3d, 0x08]),
        ("mov rax, qword ptr [rbp + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R15, 0, 4096)); }, &[0x4a, 0x8b, 0x84, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbp + r15*8]", |a| { a.ld(RAX, mem_idx(RBP, R15, 3, 0)); }, &[0x4a, 0x8b, 0x44, 0xfd, 0x00]),
        ("mov rax, qword ptr [rbp + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(RBP, R15, 3, 8)); }, &[0x4a, 0x8b, 0x44, 0xfd, 0x08]),
        ("mov rax, qword ptr [rbp + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(RBP, R15, 3, 4096)); }, &[0x4a, 0x8b, 0x84, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rax*1]", |a| { a.ld(RAX, mem_idx(R12, RAX, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [r12 + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, RAX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x04, 0x08]),
        ("mov rax, qword ptr [r12 + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RAX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rax*8]", |a| { a.ld(RAX, mem_idx(R12, RAX, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xc4]),
        ("mov rax, qword ptr [r12 + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, RAX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xc4, 0x08]),
        ("mov rax, qword ptr [r12 + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RAX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rcx*1]", |a| { a.ld(RAX, mem_idx(R12, RCX, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x0c]),
        ("mov rax, qword ptr [r12 + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, RCX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x0c, 0x08]),
        ("mov rax, qword ptr [r12 + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RCX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rcx*8]", |a| { a.ld(RAX, mem_idx(R12, RCX, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xcc]),
        ("mov rax, qword ptr [r12 + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, RCX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xcc, 0x08]),
        ("mov rax, qword ptr [r12 + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RCX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rbp*1]", |a| { a.ld(RAX, mem_idx(R12, RBP, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x2c]),
        ("mov rax, qword ptr [r12 + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, RBP, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x2c, 0x08]),
        ("mov rax, qword ptr [r12 + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RBP, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rbp*8]", |a| { a.ld(RAX, mem_idx(R12, RBP, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xec]),
        ("mov rax, qword ptr [r12 + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, RBP, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xec, 0x08]),
        ("mov rax, qword ptr [r12 + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RBP, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rsi*1]", |a| { a.ld(RAX, mem_idx(R12, RSI, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x34]),
        ("mov rax, qword ptr [r12 + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, RSI, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x34, 0x08]),
        ("mov rax, qword ptr [r12 + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RSI, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + rsi*8]", |a| { a.ld(RAX, mem_idx(R12, RSI, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xf4]),
        ("mov rax, qword ptr [r12 + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, RSI, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xf4, 0x08]),
        ("mov rax, qword ptr [r12 + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, RSI, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r8*1]", |a| { a.ld(RAX, mem_idx(R12, R8, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [r12 + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, R8, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x04, 0x08]),
        ("mov rax, qword ptr [r12 + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R8, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r8*8]", |a| { a.ld(RAX, mem_idx(R12, R8, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xc4]),
        ("mov rax, qword ptr [r12 + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, R8, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xc4, 0x08]),
        ("mov rax, qword ptr [r12 + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R8, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r12*1]", |a| { a.ld(RAX, mem_idx(R12, R12, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x24]),
        ("mov rax, qword ptr [r12 + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, R12, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x24, 0x08]),
        ("mov rax, qword ptr [r12 + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R12, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r12*8]", |a| { a.ld(RAX, mem_idx(R12, R12, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xe4]),
        ("mov rax, qword ptr [r12 + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, R12, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xe4, 0x08]),
        ("mov rax, qword ptr [r12 + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R12, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r13*1]", |a| { a.ld(RAX, mem_idx(R12, R13, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x2c]),
        ("mov rax, qword ptr [r12 + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, R13, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x2c, 0x08]),
        ("mov rax, qword ptr [r12 + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R13, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r13*8]", |a| { a.ld(RAX, mem_idx(R12, R13, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xec]),
        ("mov rax, qword ptr [r12 + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, R13, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xec, 0x08]),
        ("mov rax, qword ptr [r12 + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R13, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r15*1]", |a| { a.ld(RAX, mem_idx(R12, R15, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x3c]),
        ("mov rax, qword ptr [r12 + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(R12, R15, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x3c, 0x08]),
        ("mov rax, qword ptr [r12 + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R15, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 + r15*8]", |a| { a.ld(RAX, mem_idx(R12, R15, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xfc]),
        ("mov rax, qword ptr [r12 + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(R12, R15, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xfc, 0x08]),
        ("mov rax, qword ptr [r12 + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(R12, R15, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rax*1]", |a| { a.ld(RAX, mem_idx(R13, RAX, 0, 0)); }, &[0x49, 0x8b, 0x44, 0x05, 0x00]),
        ("mov rax, qword ptr [r13 + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, RAX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x05, 0x08]),
        ("mov rax, qword ptr [r13 + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RAX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rax*8]", |a| { a.ld(RAX, mem_idx(R13, RAX, 3, 0)); }, &[0x49, 0x8b, 0x44, 0xc5, 0x00]),
        ("mov rax, qword ptr [r13 + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, RAX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xc5, 0x08]),
        ("mov rax, qword ptr [r13 + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RAX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rcx*1]", |a| { a.ld(RAX, mem_idx(R13, RCX, 0, 0)); }, &[0x49, 0x8b, 0x44, 0x0d, 0x00]),
        ("mov rax, qword ptr [r13 + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, RCX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x0d, 0x08]),
        ("mov rax, qword ptr [r13 + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RCX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rcx*8]", |a| { a.ld(RAX, mem_idx(R13, RCX, 3, 0)); }, &[0x49, 0x8b, 0x44, 0xcd, 0x00]),
        ("mov rax, qword ptr [r13 + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, RCX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xcd, 0x08]),
        ("mov rax, qword ptr [r13 + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RCX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rbp*1]", |a| { a.ld(RAX, mem_idx(R13, RBP, 0, 0)); }, &[0x49, 0x8b, 0x44, 0x2d, 0x00]),
        ("mov rax, qword ptr [r13 + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, RBP, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x2d, 0x08]),
        ("mov rax, qword ptr [r13 + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RBP, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rbp*8]", |a| { a.ld(RAX, mem_idx(R13, RBP, 3, 0)); }, &[0x49, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [r13 + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, RBP, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xed, 0x08]),
        ("mov rax, qword ptr [r13 + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RBP, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rsi*1]", |a| { a.ld(RAX, mem_idx(R13, RSI, 0, 0)); }, &[0x49, 0x8b, 0x44, 0x35, 0x00]),
        ("mov rax, qword ptr [r13 + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, RSI, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x35, 0x08]),
        ("mov rax, qword ptr [r13 + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RSI, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + rsi*8]", |a| { a.ld(RAX, mem_idx(R13, RSI, 3, 0)); }, &[0x49, 0x8b, 0x44, 0xf5, 0x00]),
        ("mov rax, qword ptr [r13 + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, RSI, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xf5, 0x08]),
        ("mov rax, qword ptr [r13 + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, RSI, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r8*1]", |a| { a.ld(RAX, mem_idx(R13, R8, 0, 0)); }, &[0x4b, 0x8b, 0x44, 0x05, 0x00]),
        ("mov rax, qword ptr [r13 + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, R8, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x05, 0x08]),
        ("mov rax, qword ptr [r13 + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R8, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r8*8]", |a| { a.ld(RAX, mem_idx(R13, R8, 3, 0)); }, &[0x4b, 0x8b, 0x44, 0xc5, 0x00]),
        ("mov rax, qword ptr [r13 + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, R8, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xc5, 0x08]),
        ("mov rax, qword ptr [r13 + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R8, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r12*1]", |a| { a.ld(RAX, mem_idx(R13, R12, 0, 0)); }, &[0x4b, 0x8b, 0x44, 0x25, 0x00]),
        ("mov rax, qword ptr [r13 + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, R12, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x25, 0x08]),
        ("mov rax, qword ptr [r13 + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R12, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r12*8]", |a| { a.ld(RAX, mem_idx(R13, R12, 3, 0)); }, &[0x4b, 0x8b, 0x44, 0xe5, 0x00]),
        ("mov rax, qword ptr [r13 + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, R12, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xe5, 0x08]),
        ("mov rax, qword ptr [r13 + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R12, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r13*1]", |a| { a.ld(RAX, mem_idx(R13, R13, 0, 0)); }, &[0x4b, 0x8b, 0x44, 0x2d, 0x00]),
        ("mov rax, qword ptr [r13 + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, R13, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x2d, 0x08]),
        ("mov rax, qword ptr [r13 + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R13, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r13*8]", |a| { a.ld(RAX, mem_idx(R13, R13, 3, 0)); }, &[0x4b, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [r13 + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, R13, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xed, 0x08]),
        ("mov rax, qword ptr [r13 + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R13, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r15*1]", |a| { a.ld(RAX, mem_idx(R13, R15, 0, 0)); }, &[0x4b, 0x8b, 0x44, 0x3d, 0x00]),
        ("mov rax, qword ptr [r13 + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(R13, R15, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x3d, 0x08]),
        ("mov rax, qword ptr [r13 + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R15, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r13 + r15*8]", |a| { a.ld(RAX, mem_idx(R13, R15, 3, 0)); }, &[0x4b, 0x8b, 0x44, 0xfd, 0x00]),
        ("mov rax, qword ptr [r13 + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(R13, R15, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xfd, 0x08]),
        ("mov rax, qword ptr [r13 + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(R13, R15, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rax*1]", |a| { a.ld(RAX, mem_idx(R15, RAX, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x07]),
        ("mov rax, qword ptr [r15 + rax*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, RAX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x07, 0x08]),
        ("mov rax, qword ptr [r15 + rax*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RAX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rax*8]", |a| { a.ld(RAX, mem_idx(R15, RAX, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xc7]),
        ("mov rax, qword ptr [r15 + rax*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, RAX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xc7, 0x08]),
        ("mov rax, qword ptr [r15 + rax*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RAX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rcx*1]", |a| { a.ld(RAX, mem_idx(R15, RCX, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x0f]),
        ("mov rax, qword ptr [r15 + rcx*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, RCX, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x0f, 0x08]),
        ("mov rax, qword ptr [r15 + rcx*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RCX, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x0f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rcx*8]", |a| { a.ld(RAX, mem_idx(R15, RCX, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xcf]),
        ("mov rax, qword ptr [r15 + rcx*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, RCX, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xcf, 0x08]),
        ("mov rax, qword ptr [r15 + rcx*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RCX, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rbp*1]", |a| { a.ld(RAX, mem_idx(R15, RBP, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x2f]),
        ("mov rax, qword ptr [r15 + rbp*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, RBP, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x2f, 0x08]),
        ("mov rax, qword ptr [r15 + rbp*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RBP, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rbp*8]", |a| { a.ld(RAX, mem_idx(R15, RBP, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xef]),
        ("mov rax, qword ptr [r15 + rbp*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, RBP, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xef, 0x08]),
        ("mov rax, qword ptr [r15 + rbp*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RBP, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rsi*1]", |a| { a.ld(RAX, mem_idx(R15, RSI, 0, 0)); }, &[0x49, 0x8b, 0x04, 0x37]),
        ("mov rax, qword ptr [r15 + rsi*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, RSI, 0, 8)); }, &[0x49, 0x8b, 0x44, 0x37, 0x08]),
        ("mov rax, qword ptr [r15 + rsi*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RSI, 0, 4096)); }, &[0x49, 0x8b, 0x84, 0x37, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + rsi*8]", |a| { a.ld(RAX, mem_idx(R15, RSI, 3, 0)); }, &[0x49, 0x8b, 0x04, 0xf7]),
        ("mov rax, qword ptr [r15 + rsi*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, RSI, 3, 8)); }, &[0x49, 0x8b, 0x44, 0xf7, 0x08]),
        ("mov rax, qword ptr [r15 + rsi*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, RSI, 3, 4096)); }, &[0x49, 0x8b, 0x84, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r8*1]", |a| { a.ld(RAX, mem_idx(R15, R8, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x07]),
        ("mov rax, qword ptr [r15 + r8*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, R8, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x07, 0x08]),
        ("mov rax, qword ptr [r15 + r8*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R8, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r8*8]", |a| { a.ld(RAX, mem_idx(R15, R8, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xc7]),
        ("mov rax, qword ptr [r15 + r8*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, R8, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xc7, 0x08]),
        ("mov rax, qword ptr [r15 + r8*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R8, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r12*1]", |a| { a.ld(RAX, mem_idx(R15, R12, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x27]),
        ("mov rax, qword ptr [r15 + r12*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, R12, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x27, 0x08]),
        ("mov rax, qword ptr [r15 + r12*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R12, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x27, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r12*8]", |a| { a.ld(RAX, mem_idx(R15, R12, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xe7]),
        ("mov rax, qword ptr [r15 + r12*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, R12, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xe7, 0x08]),
        ("mov rax, qword ptr [r15 + r12*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R12, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r13*1]", |a| { a.ld(RAX, mem_idx(R15, R13, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x2f]),
        ("mov rax, qword ptr [r15 + r13*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, R13, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x2f, 0x08]),
        ("mov rax, qword ptr [r15 + r13*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R13, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r13*8]", |a| { a.ld(RAX, mem_idx(R15, R13, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xef]),
        ("mov rax, qword ptr [r15 + r13*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, R13, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xef, 0x08]),
        ("mov rax, qword ptr [r15 + r13*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R13, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r15*1]", |a| { a.ld(RAX, mem_idx(R15, R15, 0, 0)); }, &[0x4b, 0x8b, 0x04, 0x3f]),
        ("mov rax, qword ptr [r15 + r15*1 + 8]", |a| { a.ld(RAX, mem_idx(R15, R15, 0, 8)); }, &[0x4b, 0x8b, 0x44, 0x3f, 0x08]),
        ("mov rax, qword ptr [r15 + r15*1 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R15, 0, 4096)); }, &[0x4b, 0x8b, 0x84, 0x3f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [r15 + r15*8]", |a| { a.ld(RAX, mem_idx(R15, R15, 3, 0)); }, &[0x4b, 0x8b, 0x04, 0xff]),
        ("mov rax, qword ptr [r15 + r15*8 + 8]", |a| { a.ld(RAX, mem_idx(R15, R15, 3, 8)); }, &[0x4b, 0x8b, 0x44, 0xff, 0x08]),
        ("mov rax, qword ptr [r15 + r15*8 + 4096]", |a| { a.ld(RAX, mem_idx(R15, R15, 3, 4096)); }, &[0x4b, 0x8b, 0x84, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("mov rax, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld(RAX, mem_idx(RBX, RCX, 1, 32)); }, &[0x48, 0x8b, 0x44, 0x4b, 0x20]),
        ("mov rax, qword ptr [rbx + r12*2 + 32]", |a| { a.ld(RAX, mem_idx(RBX, R12, 1, 32)); }, &[0x4a, 0x8b, 0x44, 0x63, 0x20]),
        ("mov rax, qword ptr [rbp + rcx*2 + 32]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 1, 32)); }, &[0x48, 0x8b, 0x44, 0x4d, 0x20]),
        ("mov rax, qword ptr [rbp + r12*2 + 32]", |a| { a.ld(RAX, mem_idx(RBP, R12, 1, 32)); }, &[0x4a, 0x8b, 0x44, 0x65, 0x20]),
        ("mov rax, qword ptr [rbx + rcx*4 + 32]", |a| { a.ld(RAX, mem_idx(RBX, RCX, 2, 32)); }, &[0x48, 0x8b, 0x44, 0x8b, 0x20]),
        ("mov rax, qword ptr [rbx + r12*4 + 32]", |a| { a.ld(RAX, mem_idx(RBX, R12, 2, 32)); }, &[0x4a, 0x8b, 0x44, 0xa3, 0x20]),
        ("mov rax, qword ptr [rbp + rcx*4 + 32]", |a| { a.ld(RAX, mem_idx(RBP, RCX, 2, 32)); }, &[0x48, 0x8b, 0x44, 0x8d, 0x20]),
        ("mov rax, qword ptr [rbp + r12*4 + 32]", |a| { a.ld(RAX, mem_idx(RBP, R12, 2, 32)); }, &[0x4a, 0x8b, 0x44, 0xa5, 0x20]),
        ("mov r9, qword ptr [rax]", |a| { a.ld(R9, mem(RAX, 0)); }, &[0x4c, 0x8b, 0x08]),
        ("mov r9, qword ptr [rax + 1]", |a| { a.ld(R9, mem(RAX, 1)); }, &[0x4c, 0x8b, 0x48, 0x01]),
        ("mov r9, qword ptr [rax - 1]", |a| { a.ld(R9, mem(RAX, -1)); }, &[0x4c, 0x8b, 0x48, 0xff]),
        ("mov r9, qword ptr [rax + 127]", |a| { a.ld(R9, mem(RAX, 127)); }, &[0x4c, 0x8b, 0x48, 0x7f]),
        ("mov r9, qword ptr [rax + 128]", |a| { a.ld(R9, mem(RAX, 128)); }, &[0x4c, 0x8b, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [rax - 128]", |a| { a.ld(R9, mem(RAX, -128)); }, &[0x4c, 0x8b, 0x48, 0x80]),
        ("mov r9, qword ptr [rax - 129]", |a| { a.ld(R9, mem(RAX, -129)); }, &[0x4c, 0x8b, 0x88, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [rax + 305419896]", |a| { a.ld(R9, mem(RAX, 305419896)); }, &[0x4c, 0x8b, 0x88, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [rax - 305419896]", |a| { a.ld(R9, mem(RAX, -305419896)); }, &[0x4c, 0x8b, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [rsp]", |a| { a.ld(R9, mem(RSP, 0)); }, &[0x4c, 0x8b, 0x0c, 0x24]),
        ("mov r9, qword ptr [rsp + 1]", |a| { a.ld(R9, mem(RSP, 1)); }, &[0x4c, 0x8b, 0x4c, 0x24, 0x01]),
        ("mov r9, qword ptr [rsp - 1]", |a| { a.ld(R9, mem(RSP, -1)); }, &[0x4c, 0x8b, 0x4c, 0x24, 0xff]),
        ("mov r9, qword ptr [rsp + 127]", |a| { a.ld(R9, mem(RSP, 127)); }, &[0x4c, 0x8b, 0x4c, 0x24, 0x7f]),
        ("mov r9, qword ptr [rsp + 128]", |a| { a.ld(R9, mem(RSP, 128)); }, &[0x4c, 0x8b, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp - 128]", |a| { a.ld(R9, mem(RSP, -128)); }, &[0x4c, 0x8b, 0x4c, 0x24, 0x80]),
        ("mov r9, qword ptr [rsp - 129]", |a| { a.ld(R9, mem(RSP, -129)); }, &[0x4c, 0x8b, 0x8c, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [rsp + 305419896]", |a| { a.ld(R9, mem(RSP, 305419896)); }, &[0x4c, 0x8b, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [rsp - 305419896]", |a| { a.ld(R9, mem(RSP, -305419896)); }, &[0x4c, 0x8b, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [rbp]", |a| { a.ld(R9, mem(RBP, 0)); }, &[0x4c, 0x8b, 0x4d, 0x00]),
        ("mov r9, qword ptr [rbp + 1]", |a| { a.ld(R9, mem(RBP, 1)); }, &[0x4c, 0x8b, 0x4d, 0x01]),
        ("mov r9, qword ptr [rbp - 1]", |a| { a.ld(R9, mem(RBP, -1)); }, &[0x4c, 0x8b, 0x4d, 0xff]),
        ("mov r9, qword ptr [rbp + 127]", |a| { a.ld(R9, mem(RBP, 127)); }, &[0x4c, 0x8b, 0x4d, 0x7f]),
        ("mov r9, qword ptr [rbp + 128]", |a| { a.ld(R9, mem(RBP, 128)); }, &[0x4c, 0x8b, 0x8d, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp - 128]", |a| { a.ld(R9, mem(RBP, -128)); }, &[0x4c, 0x8b, 0x4d, 0x80]),
        ("mov r9, qword ptr [rbp - 129]", |a| { a.ld(R9, mem(RBP, -129)); }, &[0x4c, 0x8b, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [rbp + 305419896]", |a| { a.ld(R9, mem(RBP, 305419896)); }, &[0x4c, 0x8b, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [rbp - 305419896]", |a| { a.ld(R9, mem(RBP, -305419896)); }, &[0x4c, 0x8b, 0x8d, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [r8]", |a| { a.ld(R9, mem(R8, 0)); }, &[0x4d, 0x8b, 0x08]),
        ("mov r9, qword ptr [r8 + 1]", |a| { a.ld(R9, mem(R8, 1)); }, &[0x4d, 0x8b, 0x48, 0x01]),
        ("mov r9, qword ptr [r8 - 1]", |a| { a.ld(R9, mem(R8, -1)); }, &[0x4d, 0x8b, 0x48, 0xff]),
        ("mov r9, qword ptr [r8 + 127]", |a| { a.ld(R9, mem(R8, 127)); }, &[0x4d, 0x8b, 0x48, 0x7f]),
        ("mov r9, qword ptr [r8 + 128]", |a| { a.ld(R9, mem(R8, 128)); }, &[0x4d, 0x8b, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [r8 - 128]", |a| { a.ld(R9, mem(R8, -128)); }, &[0x4d, 0x8b, 0x48, 0x80]),
        ("mov r9, qword ptr [r8 - 129]", |a| { a.ld(R9, mem(R8, -129)); }, &[0x4d, 0x8b, 0x88, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [r8 + 305419896]", |a| { a.ld(R9, mem(R8, 305419896)); }, &[0x4d, 0x8b, 0x88, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [r8 - 305419896]", |a| { a.ld(R9, mem(R8, -305419896)); }, &[0x4d, 0x8b, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [r12]", |a| { a.ld(R9, mem(R12, 0)); }, &[0x4d, 0x8b, 0x0c, 0x24]),
        ("mov r9, qword ptr [r12 + 1]", |a| { a.ld(R9, mem(R12, 1)); }, &[0x4d, 0x8b, 0x4c, 0x24, 0x01]),
        ("mov r9, qword ptr [r12 - 1]", |a| { a.ld(R9, mem(R12, -1)); }, &[0x4d, 0x8b, 0x4c, 0x24, 0xff]),
        ("mov r9, qword ptr [r12 + 127]", |a| { a.ld(R9, mem(R12, 127)); }, &[0x4d, 0x8b, 0x4c, 0x24, 0x7f]),
        ("mov r9, qword ptr [r12 + 128]", |a| { a.ld(R9, mem(R12, 128)); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 - 128]", |a| { a.ld(R9, mem(R12, -128)); }, &[0x4d, 0x8b, 0x4c, 0x24, 0x80]),
        ("mov r9, qword ptr [r12 - 129]", |a| { a.ld(R9, mem(R12, -129)); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [r12 + 305419896]", |a| { a.ld(R9, mem(R12, 305419896)); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [r12 - 305419896]", |a| { a.ld(R9, mem(R12, -305419896)); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [r13]", |a| { a.ld(R9, mem(R13, 0)); }, &[0x4d, 0x8b, 0x4d, 0x00]),
        ("mov r9, qword ptr [r13 + 1]", |a| { a.ld(R9, mem(R13, 1)); }, &[0x4d, 0x8b, 0x4d, 0x01]),
        ("mov r9, qword ptr [r13 - 1]", |a| { a.ld(R9, mem(R13, -1)); }, &[0x4d, 0x8b, 0x4d, 0xff]),
        ("mov r9, qword ptr [r13 + 127]", |a| { a.ld(R9, mem(R13, 127)); }, &[0x4d, 0x8b, 0x4d, 0x7f]),
        ("mov r9, qword ptr [r13 + 128]", |a| { a.ld(R9, mem(R13, 128)); }, &[0x4d, 0x8b, 0x8d, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 - 128]", |a| { a.ld(R9, mem(R13, -128)); }, &[0x4d, 0x8b, 0x4d, 0x80]),
        ("mov r9, qword ptr [r13 - 129]", |a| { a.ld(R9, mem(R13, -129)); }, &[0x4d, 0x8b, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [r13 + 305419896]", |a| { a.ld(R9, mem(R13, 305419896)); }, &[0x4d, 0x8b, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [r13 - 305419896]", |a| { a.ld(R9, mem(R13, -305419896)); }, &[0x4d, 0x8b, 0x8d, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [r15]", |a| { a.ld(R9, mem(R15, 0)); }, &[0x4d, 0x8b, 0x0f]),
        ("mov r9, qword ptr [r15 + 1]", |a| { a.ld(R9, mem(R15, 1)); }, &[0x4d, 0x8b, 0x4f, 0x01]),
        ("mov r9, qword ptr [r15 - 1]", |a| { a.ld(R9, mem(R15, -1)); }, &[0x4d, 0x8b, 0x4f, 0xff]),
        ("mov r9, qword ptr [r15 + 127]", |a| { a.ld(R9, mem(R15, 127)); }, &[0x4d, 0x8b, 0x4f, 0x7f]),
        ("mov r9, qword ptr [r15 + 128]", |a| { a.ld(R9, mem(R15, 128)); }, &[0x4d, 0x8b, 0x8f, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 - 128]", |a| { a.ld(R9, mem(R15, -128)); }, &[0x4d, 0x8b, 0x4f, 0x80]),
        ("mov r9, qword ptr [r15 - 129]", |a| { a.ld(R9, mem(R15, -129)); }, &[0x4d, 0x8b, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [r15 + 305419896]", |a| { a.ld(R9, mem(R15, 305419896)); }, &[0x4d, 0x8b, 0x8f, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [r15 - 305419896]", |a| { a.ld(R9, mem(R15, -305419896)); }, &[0x4d, 0x8b, 0x8f, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [rbx]", |a| { a.ld(R9, mem(RBX, 0)); }, &[0x4c, 0x8b, 0x0b]),
        ("mov r9, qword ptr [rbx + 1]", |a| { a.ld(R9, mem(RBX, 1)); }, &[0x4c, 0x8b, 0x4b, 0x01]),
        ("mov r9, qword ptr [rbx - 1]", |a| { a.ld(R9, mem(RBX, -1)); }, &[0x4c, 0x8b, 0x4b, 0xff]),
        ("mov r9, qword ptr [rbx + 127]", |a| { a.ld(R9, mem(RBX, 127)); }, &[0x4c, 0x8b, 0x4b, 0x7f]),
        ("mov r9, qword ptr [rbx + 128]", |a| { a.ld(R9, mem(RBX, 128)); }, &[0x4c, 0x8b, 0x8b, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [rbx - 128]", |a| { a.ld(R9, mem(RBX, -128)); }, &[0x4c, 0x8b, 0x4b, 0x80]),
        ("mov r9, qword ptr [rbx - 129]", |a| { a.ld(R9, mem(RBX, -129)); }, &[0x4c, 0x8b, 0x8b, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [rbx + 305419896]", |a| { a.ld(R9, mem(RBX, 305419896)); }, &[0x4c, 0x8b, 0x8b, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [rbx - 305419896]", |a| { a.ld(R9, mem(RBX, -305419896)); }, &[0x4c, 0x8b, 0x8b, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [rax + rax*1]", |a| { a.ld(R9, mem_idx(RAX, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x00]),
        ("mov r9, qword ptr [rax + rax*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x00, 0x08]),
        ("mov r9, qword ptr [rax + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rax*8]", |a| { a.ld(R9, mem_idx(RAX, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xc0]),
        ("mov r9, qword ptr [rax + rax*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xc0, 0x08]),
        ("mov r9, qword ptr [rax + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rcx*1]", |a| { a.ld(R9, mem_idx(RAX, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x08]),
        ("mov r9, qword ptr [rax + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x08, 0x08]),
        ("mov r9, qword ptr [rax + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x08, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rcx*8]", |a| { a.ld(R9, mem_idx(RAX, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xc8]),
        ("mov r9, qword ptr [rax + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xc8, 0x08]),
        ("mov r9, qword ptr [rax + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rbp*1]", |a| { a.ld(R9, mem_idx(RAX, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x28]),
        ("mov r9, qword ptr [rax + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x28, 0x08]),
        ("mov r9, qword ptr [rax + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rbp*8]", |a| { a.ld(R9, mem_idx(RAX, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xe8]),
        ("mov r9, qword ptr [rax + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xe8, 0x08]),
        ("mov r9, qword ptr [rax + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rsi*1]", |a| { a.ld(R9, mem_idx(RAX, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x30]),
        ("mov r9, qword ptr [rax + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x30, 0x08]),
        ("mov r9, qword ptr [rax + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x30, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + rsi*8]", |a| { a.ld(R9, mem_idx(RAX, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xf0]),
        ("mov r9, qword ptr [rax + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xf0, 0x08]),
        ("mov r9, qword ptr [rax + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r8*1]", |a| { a.ld(R9, mem_idx(RAX, R8, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x00]),
        ("mov r9, qword ptr [rax + r8*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, R8, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x00, 0x08]),
        ("mov r9, qword ptr [rax + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R8, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r8*8]", |a| { a.ld(R9, mem_idx(RAX, R8, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xc0]),
        ("mov r9, qword ptr [rax + r8*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, R8, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xc0, 0x08]),
        ("mov r9, qword ptr [rax + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R8, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r12*1]", |a| { a.ld(R9, mem_idx(RAX, R12, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x20]),
        ("mov r9, qword ptr [rax + r12*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, R12, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x20, 0x08]),
        ("mov r9, qword ptr [rax + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R12, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x20, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r12*8]", |a| { a.ld(R9, mem_idx(RAX, R12, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xe0]),
        ("mov r9, qword ptr [rax + r12*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, R12, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xe0, 0x08]),
        ("mov r9, qword ptr [rax + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R12, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r13*1]", |a| { a.ld(R9, mem_idx(RAX, R13, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x28]),
        ("mov r9, qword ptr [rax + r13*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, R13, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x28, 0x08]),
        ("mov r9, qword ptr [rax + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R13, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r13*8]", |a| { a.ld(R9, mem_idx(RAX, R13, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xe8]),
        ("mov r9, qword ptr [rax + r13*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, R13, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xe8, 0x08]),
        ("mov r9, qword ptr [rax + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R13, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r15*1]", |a| { a.ld(R9, mem_idx(RAX, R15, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x38]),
        ("mov r9, qword ptr [rax + r15*1 + 8]", |a| { a.ld(R9, mem_idx(RAX, R15, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x38, 0x08]),
        ("mov r9, qword ptr [rax + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R15, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x38, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rax + r15*8]", |a| { a.ld(R9, mem_idx(RAX, R15, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xf8]),
        ("mov r9, qword ptr [rax + r15*8 + 8]", |a| { a.ld(R9, mem_idx(RAX, R15, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xf8, 0x08]),
        ("mov r9, qword ptr [rax + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(RAX, R15, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rax*1]", |a| { a.ld(R9, mem_idx(RSP, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [rsp + rax*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x04, 0x08]),
        ("mov r9, qword ptr [rsp + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rax*8]", |a| { a.ld(R9, mem_idx(RSP, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xc4]),
        ("mov r9, qword ptr [rsp + rax*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xc4, 0x08]),
        ("mov r9, qword ptr [rsp + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rcx*1]", |a| { a.ld(R9, mem_idx(RSP, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x0c]),
        ("mov r9, qword ptr [rsp + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x0c, 0x08]),
        ("mov r9, qword ptr [rsp + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rcx*8]", |a| { a.ld(R9, mem_idx(RSP, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xcc]),
        ("mov r9, qword ptr [rsp + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xcc, 0x08]),
        ("mov r9, qword ptr [rsp + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rbp*1]", |a| { a.ld(R9, mem_idx(RSP, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x2c]),
        ("mov r9, qword ptr [rsp + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x2c, 0x08]),
        ("mov r9, qword ptr [rsp + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rbp*8]", |a| { a.ld(R9, mem_idx(RSP, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xec]),
        ("mov r9, qword ptr [rsp + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xec, 0x08]),
        ("mov r9, qword ptr [rsp + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rsi*1]", |a| { a.ld(R9, mem_idx(RSP, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x0c, 0x34]),
        ("mov r9, qword ptr [rsp + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x34, 0x08]),
        ("mov r9, qword ptr [rsp + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + rsi*8]", |a| { a.ld(R9, mem_idx(RSP, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x0c, 0xf4]),
        ("mov r9, qword ptr [rsp + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xf4, 0x08]),
        ("mov r9, qword ptr [rsp + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r8*1]", |a| { a.ld(R9, mem_idx(RSP, R8, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [rsp + r8*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, R8, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x04, 0x08]),
        ("mov r9, qword ptr [rsp + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R8, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r8*8]", |a| { a.ld(R9, mem_idx(RSP, R8, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xc4]),
        ("mov r9, qword ptr [rsp + r8*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, R8, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xc4, 0x08]),
        ("mov r9, qword ptr [rsp + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R8, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r12*1]", |a| { a.ld(R9, mem_idx(RSP, R12, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x24]),
        ("mov r9, qword ptr [rsp + r12*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, R12, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x24, 0x08]),
        ("mov r9, qword ptr [rsp + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R12, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r12*8]", |a| { a.ld(R9, mem_idx(RSP, R12, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xe4]),
        ("mov r9, qword ptr [rsp + r12*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, R12, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xe4, 0x08]),
        ("mov r9, qword ptr [rsp + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R12, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r13*1]", |a| { a.ld(R9, mem_idx(RSP, R13, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x2c]),
        ("mov r9, qword ptr [rsp + r13*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, R13, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x2c, 0x08]),
        ("mov r9, qword ptr [rsp + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R13, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r13*8]", |a| { a.ld(R9, mem_idx(RSP, R13, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xec]),
        ("mov r9, qword ptr [rsp + r13*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, R13, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xec, 0x08]),
        ("mov r9, qword ptr [rsp + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R13, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r15*1]", |a| { a.ld(R9, mem_idx(RSP, R15, 0, 0)); }, &[0x4e, 0x8b, 0x0c, 0x3c]),
        ("mov r9, qword ptr [rsp + r15*1 + 8]", |a| { a.ld(R9, mem_idx(RSP, R15, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x3c, 0x08]),
        ("mov r9, qword ptr [rsp + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R15, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rsp + r15*8]", |a| { a.ld(R9, mem_idx(RSP, R15, 3, 0)); }, &[0x4e, 0x8b, 0x0c, 0xfc]),
        ("mov r9, qword ptr [rsp + r15*8 + 8]", |a| { a.ld(R9, mem_idx(RSP, R15, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xfc, 0x08]),
        ("mov r9, qword ptr [rsp + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(RSP, R15, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rax*1]", |a| { a.ld(R9, mem_idx(RBP, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9, qword ptr [rbp + rax*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x05, 0x08]),
        ("mov r9, qword ptr [rbp + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rax*8]", |a| { a.ld(R9, mem_idx(RBP, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9, qword ptr [rbp + rax*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xc5, 0x08]),
        ("mov r9, qword ptr [rbp + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rcx*1]", |a| { a.ld(R9, mem_idx(RBP, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x4c, 0x0d, 0x00]),
        ("mov r9, qword ptr [rbp + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x0d, 0x08]),
        ("mov r9, qword ptr [rbp + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rcx*8]", |a| { a.ld(R9, mem_idx(RBP, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x4c, 0xcd, 0x00]),
        ("mov r9, qword ptr [rbp + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xcd, 0x08]),
        ("mov r9, qword ptr [rbp + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rbp*1]", |a| { a.ld(R9, mem_idx(RBP, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9, qword ptr [rbp + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x2d, 0x08]),
        ("mov r9, qword ptr [rbp + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rbp*8]", |a| { a.ld(R9, mem_idx(RBP, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [rbp + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xed, 0x08]),
        ("mov r9, qword ptr [rbp + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rsi*1]", |a| { a.ld(R9, mem_idx(RBP, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x4c, 0x35, 0x00]),
        ("mov r9, qword ptr [rbp + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x4c, 0x35, 0x08]),
        ("mov r9, qword ptr [rbp + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0x8c, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + rsi*8]", |a| { a.ld(R9, mem_idx(RBP, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x4c, 0xf5, 0x00]),
        ("mov r9, qword ptr [rbp + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x4c, 0xf5, 0x08]),
        ("mov r9, qword ptr [rbp + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0x8c, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r8*1]", |a| { a.ld(R9, mem_idx(RBP, R8, 0, 0)); }, &[0x4e, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9, qword ptr [rbp + r8*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, R8, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x05, 0x08]),
        ("mov r9, qword ptr [rbp + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R8, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r8*8]", |a| { a.ld(R9, mem_idx(RBP, R8, 3, 0)); }, &[0x4e, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9, qword ptr [rbp + r8*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, R8, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xc5, 0x08]),
        ("mov r9, qword ptr [rbp + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R8, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r12*1]", |a| { a.ld(R9, mem_idx(RBP, R12, 0, 0)); }, &[0x4e, 0x8b, 0x4c, 0x25, 0x00]),
        ("mov r9, qword ptr [rbp + r12*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, R12, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x25, 0x08]),
        ("mov r9, qword ptr [rbp + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R12, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r12*8]", |a| { a.ld(R9, mem_idx(RBP, R12, 3, 0)); }, &[0x4e, 0x8b, 0x4c, 0xe5, 0x00]),
        ("mov r9, qword ptr [rbp + r12*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, R12, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xe5, 0x08]),
        ("mov r9, qword ptr [rbp + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R12, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r13*1]", |a| { a.ld(R9, mem_idx(RBP, R13, 0, 0)); }, &[0x4e, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9, qword ptr [rbp + r13*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, R13, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x2d, 0x08]),
        ("mov r9, qword ptr [rbp + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R13, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r13*8]", |a| { a.ld(R9, mem_idx(RBP, R13, 3, 0)); }, &[0x4e, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [rbp + r13*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, R13, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xed, 0x08]),
        ("mov r9, qword ptr [rbp + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R13, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r15*1]", |a| { a.ld(R9, mem_idx(RBP, R15, 0, 0)); }, &[0x4e, 0x8b, 0x4c, 0x3d, 0x00]),
        ("mov r9, qword ptr [rbp + r15*1 + 8]", |a| { a.ld(R9, mem_idx(RBP, R15, 0, 8)); }, &[0x4e, 0x8b, 0x4c, 0x3d, 0x08]),
        ("mov r9, qword ptr [rbp + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R15, 0, 4096)); }, &[0x4e, 0x8b, 0x8c, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbp + r15*8]", |a| { a.ld(R9, mem_idx(RBP, R15, 3, 0)); }, &[0x4e, 0x8b, 0x4c, 0xfd, 0x00]),
        ("mov r9, qword ptr [rbp + r15*8 + 8]", |a| { a.ld(R9, mem_idx(RBP, R15, 3, 8)); }, &[0x4e, 0x8b, 0x4c, 0xfd, 0x08]),
        ("mov r9, qword ptr [rbp + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(RBP, R15, 3, 4096)); }, &[0x4e, 0x8b, 0x8c, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rax*1]", |a| { a.ld(R9, mem_idx(R12, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [r12 + rax*1 + 8]", |a| { a.ld(R9, mem_idx(R12, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x04, 0x08]),
        ("mov r9, qword ptr [r12 + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rax*8]", |a| { a.ld(R9, mem_idx(R12, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xc4]),
        ("mov r9, qword ptr [r12 + rax*8 + 8]", |a| { a.ld(R9, mem_idx(R12, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xc4, 0x08]),
        ("mov r9, qword ptr [r12 + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rcx*1]", |a| { a.ld(R9, mem_idx(R12, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x0c]),
        ("mov r9, qword ptr [r12 + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(R12, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x0c, 0x08]),
        ("mov r9, qword ptr [r12 + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rcx*8]", |a| { a.ld(R9, mem_idx(R12, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xcc]),
        ("mov r9, qword ptr [r12 + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(R12, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xcc, 0x08]),
        ("mov r9, qword ptr [r12 + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rbp*1]", |a| { a.ld(R9, mem_idx(R12, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x2c]),
        ("mov r9, qword ptr [r12 + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(R12, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x2c, 0x08]),
        ("mov r9, qword ptr [r12 + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rbp*8]", |a| { a.ld(R9, mem_idx(R12, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xec]),
        ("mov r9, qword ptr [r12 + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(R12, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xec, 0x08]),
        ("mov r9, qword ptr [r12 + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rsi*1]", |a| { a.ld(R9, mem_idx(R12, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x34]),
        ("mov r9, qword ptr [r12 + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(R12, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x34, 0x08]),
        ("mov r9, qword ptr [r12 + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + rsi*8]", |a| { a.ld(R9, mem_idx(R12, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xf4]),
        ("mov r9, qword ptr [r12 + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(R12, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xf4, 0x08]),
        ("mov r9, qword ptr [r12 + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r8*1]", |a| { a.ld(R9, mem_idx(R12, R8, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [r12 + r8*1 + 8]", |a| { a.ld(R9, mem_idx(R12, R8, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x04, 0x08]),
        ("mov r9, qword ptr [r12 + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, R8, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r8*8]", |a| { a.ld(R9, mem_idx(R12, R8, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xc4]),
        ("mov r9, qword ptr [r12 + r8*8 + 8]", |a| { a.ld(R9, mem_idx(R12, R8, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xc4, 0x08]),
        ("mov r9, qword ptr [r12 + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, R8, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r12*1]", |a| { a.ld(R9, mem_idx(R12, R12, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x24]),
        ("mov r9, qword ptr [r12 + r12*1 + 8]", |a| { a.ld(R9, mem_idx(R12, R12, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x24, 0x08]),
        ("mov r9, qword ptr [r12 + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, R12, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r12*8]", |a| { a.ld(R9, mem_idx(R12, R12, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xe4]),
        ("mov r9, qword ptr [r12 + r12*8 + 8]", |a| { a.ld(R9, mem_idx(R12, R12, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xe4, 0x08]),
        ("mov r9, qword ptr [r12 + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, R12, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r13*1]", |a| { a.ld(R9, mem_idx(R12, R13, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x2c]),
        ("mov r9, qword ptr [r12 + r13*1 + 8]", |a| { a.ld(R9, mem_idx(R12, R13, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x2c, 0x08]),
        ("mov r9, qword ptr [r12 + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, R13, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r13*8]", |a| { a.ld(R9, mem_idx(R12, R13, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xec]),
        ("mov r9, qword ptr [r12 + r13*8 + 8]", |a| { a.ld(R9, mem_idx(R12, R13, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xec, 0x08]),
        ("mov r9, qword ptr [r12 + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, R13, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r15*1]", |a| { a.ld(R9, mem_idx(R12, R15, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x3c]),
        ("mov r9, qword ptr [r12 + r15*1 + 8]", |a| { a.ld(R9, mem_idx(R12, R15, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x3c, 0x08]),
        ("mov r9, qword ptr [r12 + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(R12, R15, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 + r15*8]", |a| { a.ld(R9, mem_idx(R12, R15, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xfc]),
        ("mov r9, qword ptr [r12 + r15*8 + 8]", |a| { a.ld(R9, mem_idx(R12, R15, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xfc, 0x08]),
        ("mov r9, qword ptr [r12 + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(R12, R15, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rax*1]", |a| { a.ld(R9, mem_idx(R13, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9, qword ptr [r13 + rax*1 + 8]", |a| { a.ld(R9, mem_idx(R13, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x05, 0x08]),
        ("mov r9, qword ptr [r13 + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rax*8]", |a| { a.ld(R9, mem_idx(R13, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9, qword ptr [r13 + rax*8 + 8]", |a| { a.ld(R9, mem_idx(R13, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xc5, 0x08]),
        ("mov r9, qword ptr [r13 + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rcx*1]", |a| { a.ld(R9, mem_idx(R13, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x4c, 0x0d, 0x00]),
        ("mov r9, qword ptr [r13 + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(R13, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x0d, 0x08]),
        ("mov r9, qword ptr [r13 + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rcx*8]", |a| { a.ld(R9, mem_idx(R13, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x4c, 0xcd, 0x00]),
        ("mov r9, qword ptr [r13 + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(R13, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xcd, 0x08]),
        ("mov r9, qword ptr [r13 + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rbp*1]", |a| { a.ld(R9, mem_idx(R13, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9, qword ptr [r13 + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(R13, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x2d, 0x08]),
        ("mov r9, qword ptr [r13 + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rbp*8]", |a| { a.ld(R9, mem_idx(R13, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [r13 + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(R13, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xed, 0x08]),
        ("mov r9, qword ptr [r13 + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rsi*1]", |a| { a.ld(R9, mem_idx(R13, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x4c, 0x35, 0x00]),
        ("mov r9, qword ptr [r13 + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(R13, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x35, 0x08]),
        ("mov r9, qword ptr [r13 + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + rsi*8]", |a| { a.ld(R9, mem_idx(R13, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x4c, 0xf5, 0x00]),
        ("mov r9, qword ptr [r13 + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(R13, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xf5, 0x08]),
        ("mov r9, qword ptr [r13 + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r8*1]", |a| { a.ld(R9, mem_idx(R13, R8, 0, 0)); }, &[0x4f, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9, qword ptr [r13 + r8*1 + 8]", |a| { a.ld(R9, mem_idx(R13, R8, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x05, 0x08]),
        ("mov r9, qword ptr [r13 + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, R8, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r8*8]", |a| { a.ld(R9, mem_idx(R13, R8, 3, 0)); }, &[0x4f, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9, qword ptr [r13 + r8*8 + 8]", |a| { a.ld(R9, mem_idx(R13, R8, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xc5, 0x08]),
        ("mov r9, qword ptr [r13 + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, R8, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r12*1]", |a| { a.ld(R9, mem_idx(R13, R12, 0, 0)); }, &[0x4f, 0x8b, 0x4c, 0x25, 0x00]),
        ("mov r9, qword ptr [r13 + r12*1 + 8]", |a| { a.ld(R9, mem_idx(R13, R12, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x25, 0x08]),
        ("mov r9, qword ptr [r13 + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, R12, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r12*8]", |a| { a.ld(R9, mem_idx(R13, R12, 3, 0)); }, &[0x4f, 0x8b, 0x4c, 0xe5, 0x00]),
        ("mov r9, qword ptr [r13 + r12*8 + 8]", |a| { a.ld(R9, mem_idx(R13, R12, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xe5, 0x08]),
        ("mov r9, qword ptr [r13 + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, R12, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r13*1]", |a| { a.ld(R9, mem_idx(R13, R13, 0, 0)); }, &[0x4f, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9, qword ptr [r13 + r13*1 + 8]", |a| { a.ld(R9, mem_idx(R13, R13, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x2d, 0x08]),
        ("mov r9, qword ptr [r13 + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, R13, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r13*8]", |a| { a.ld(R9, mem_idx(R13, R13, 3, 0)); }, &[0x4f, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [r13 + r13*8 + 8]", |a| { a.ld(R9, mem_idx(R13, R13, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xed, 0x08]),
        ("mov r9, qword ptr [r13 + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, R13, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r15*1]", |a| { a.ld(R9, mem_idx(R13, R15, 0, 0)); }, &[0x4f, 0x8b, 0x4c, 0x3d, 0x00]),
        ("mov r9, qword ptr [r13 + r15*1 + 8]", |a| { a.ld(R9, mem_idx(R13, R15, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x3d, 0x08]),
        ("mov r9, qword ptr [r13 + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(R13, R15, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r13 + r15*8]", |a| { a.ld(R9, mem_idx(R13, R15, 3, 0)); }, &[0x4f, 0x8b, 0x4c, 0xfd, 0x00]),
        ("mov r9, qword ptr [r13 + r15*8 + 8]", |a| { a.ld(R9, mem_idx(R13, R15, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xfd, 0x08]),
        ("mov r9, qword ptr [r13 + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(R13, R15, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rax*1]", |a| { a.ld(R9, mem_idx(R15, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x07]),
        ("mov r9, qword ptr [r15 + rax*1 + 8]", |a| { a.ld(R9, mem_idx(R15, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x07, 0x08]),
        ("mov r9, qword ptr [r15 + rax*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rax*8]", |a| { a.ld(R9, mem_idx(R15, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xc7]),
        ("mov r9, qword ptr [r15 + rax*8 + 8]", |a| { a.ld(R9, mem_idx(R15, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xc7, 0x08]),
        ("mov r9, qword ptr [r15 + rax*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rcx*1]", |a| { a.ld(R9, mem_idx(R15, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x0f]),
        ("mov r9, qword ptr [r15 + rcx*1 + 8]", |a| { a.ld(R9, mem_idx(R15, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x0f, 0x08]),
        ("mov r9, qword ptr [r15 + rcx*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x0f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rcx*8]", |a| { a.ld(R9, mem_idx(R15, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xcf]),
        ("mov r9, qword ptr [r15 + rcx*8 + 8]", |a| { a.ld(R9, mem_idx(R15, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xcf, 0x08]),
        ("mov r9, qword ptr [r15 + rcx*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rbp*1]", |a| { a.ld(R9, mem_idx(R15, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x2f]),
        ("mov r9, qword ptr [r15 + rbp*1 + 8]", |a| { a.ld(R9, mem_idx(R15, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x2f, 0x08]),
        ("mov r9, qword ptr [r15 + rbp*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rbp*8]", |a| { a.ld(R9, mem_idx(R15, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xef]),
        ("mov r9, qword ptr [r15 + rbp*8 + 8]", |a| { a.ld(R9, mem_idx(R15, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xef, 0x08]),
        ("mov r9, qword ptr [r15 + rbp*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rsi*1]", |a| { a.ld(R9, mem_idx(R15, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x0c, 0x37]),
        ("mov r9, qword ptr [r15 + rsi*1 + 8]", |a| { a.ld(R9, mem_idx(R15, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x4c, 0x37, 0x08]),
        ("mov r9, qword ptr [r15 + rsi*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0x8c, 0x37, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + rsi*8]", |a| { a.ld(R9, mem_idx(R15, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x0c, 0xf7]),
        ("mov r9, qword ptr [r15 + rsi*8 + 8]", |a| { a.ld(R9, mem_idx(R15, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x4c, 0xf7, 0x08]),
        ("mov r9, qword ptr [r15 + rsi*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0x8c, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r8*1]", |a| { a.ld(R9, mem_idx(R15, R8, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x07]),
        ("mov r9, qword ptr [r15 + r8*1 + 8]", |a| { a.ld(R9, mem_idx(R15, R8, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x07, 0x08]),
        ("mov r9, qword ptr [r15 + r8*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, R8, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r8*8]", |a| { a.ld(R9, mem_idx(R15, R8, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xc7]),
        ("mov r9, qword ptr [r15 + r8*8 + 8]", |a| { a.ld(R9, mem_idx(R15, R8, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xc7, 0x08]),
        ("mov r9, qword ptr [r15 + r8*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, R8, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r12*1]", |a| { a.ld(R9, mem_idx(R15, R12, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x27]),
        ("mov r9, qword ptr [r15 + r12*1 + 8]", |a| { a.ld(R9, mem_idx(R15, R12, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x27, 0x08]),
        ("mov r9, qword ptr [r15 + r12*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, R12, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x27, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r12*8]", |a| { a.ld(R9, mem_idx(R15, R12, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xe7]),
        ("mov r9, qword ptr [r15 + r12*8 + 8]", |a| { a.ld(R9, mem_idx(R15, R12, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xe7, 0x08]),
        ("mov r9, qword ptr [r15 + r12*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, R12, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r13*1]", |a| { a.ld(R9, mem_idx(R15, R13, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x2f]),
        ("mov r9, qword ptr [r15 + r13*1 + 8]", |a| { a.ld(R9, mem_idx(R15, R13, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x2f, 0x08]),
        ("mov r9, qword ptr [r15 + r13*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, R13, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r13*8]", |a| { a.ld(R9, mem_idx(R15, R13, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xef]),
        ("mov r9, qword ptr [r15 + r13*8 + 8]", |a| { a.ld(R9, mem_idx(R15, R13, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xef, 0x08]),
        ("mov r9, qword ptr [r15 + r13*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, R13, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r15*1]", |a| { a.ld(R9, mem_idx(R15, R15, 0, 0)); }, &[0x4f, 0x8b, 0x0c, 0x3f]),
        ("mov r9, qword ptr [r15 + r15*1 + 8]", |a| { a.ld(R9, mem_idx(R15, R15, 0, 8)); }, &[0x4f, 0x8b, 0x4c, 0x3f, 0x08]),
        ("mov r9, qword ptr [r15 + r15*1 + 4096]", |a| { a.ld(R9, mem_idx(R15, R15, 0, 4096)); }, &[0x4f, 0x8b, 0x8c, 0x3f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [r15 + r15*8]", |a| { a.ld(R9, mem_idx(R15, R15, 3, 0)); }, &[0x4f, 0x8b, 0x0c, 0xff]),
        ("mov r9, qword ptr [r15 + r15*8 + 8]", |a| { a.ld(R9, mem_idx(R15, R15, 3, 8)); }, &[0x4f, 0x8b, 0x4c, 0xff, 0x08]),
        ("mov r9, qword ptr [r15 + r15*8 + 4096]", |a| { a.ld(R9, mem_idx(R15, R15, 3, 4096)); }, &[0x4f, 0x8b, 0x8c, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("mov r9, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld(R9, mem_idx(RBX, RCX, 1, 32)); }, &[0x4c, 0x8b, 0x4c, 0x4b, 0x20]),
        ("mov r9, qword ptr [rbx + r12*2 + 32]", |a| { a.ld(R9, mem_idx(RBX, R12, 1, 32)); }, &[0x4e, 0x8b, 0x4c, 0x63, 0x20]),
        ("mov r9, qword ptr [rbp + rcx*2 + 32]", |a| { a.ld(R9, mem_idx(RBP, RCX, 1, 32)); }, &[0x4c, 0x8b, 0x4c, 0x4d, 0x20]),
        ("mov r9, qword ptr [rbp + r12*2 + 32]", |a| { a.ld(R9, mem_idx(RBP, R12, 1, 32)); }, &[0x4e, 0x8b, 0x4c, 0x65, 0x20]),
        ("mov r9, qword ptr [rbx + rcx*4 + 32]", |a| { a.ld(R9, mem_idx(RBX, RCX, 2, 32)); }, &[0x4c, 0x8b, 0x4c, 0x8b, 0x20]),
        ("mov r9, qword ptr [rbx + r12*4 + 32]", |a| { a.ld(R9, mem_idx(RBX, R12, 2, 32)); }, &[0x4e, 0x8b, 0x4c, 0xa3, 0x20]),
        ("mov r9, qword ptr [rbp + rcx*4 + 32]", |a| { a.ld(R9, mem_idx(RBP, RCX, 2, 32)); }, &[0x4c, 0x8b, 0x4c, 0x8d, 0x20]),
        ("mov r9, qword ptr [rbp + r12*4 + 32]", |a| { a.ld(R9, mem_idx(RBP, R12, 2, 32)); }, &[0x4e, 0x8b, 0x4c, 0xa5, 0x20]),
        ("mov rsp, qword ptr [rax]", |a| { a.ld(RSP, mem(RAX, 0)); }, &[0x48, 0x8b, 0x20]),
        ("mov rsp, qword ptr [rax + 1]", |a| { a.ld(RSP, mem(RAX, 1)); }, &[0x48, 0x8b, 0x60, 0x01]),
        ("mov rsp, qword ptr [rax - 1]", |a| { a.ld(RSP, mem(RAX, -1)); }, &[0x48, 0x8b, 0x60, 0xff]),
        ("mov rsp, qword ptr [rax + 127]", |a| { a.ld(RSP, mem(RAX, 127)); }, &[0x48, 0x8b, 0x60, 0x7f]),
        ("mov rsp, qword ptr [rax + 128]", |a| { a.ld(RSP, mem(RAX, 128)); }, &[0x48, 0x8b, 0xa0, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax - 128]", |a| { a.ld(RSP, mem(RAX, -128)); }, &[0x48, 0x8b, 0x60, 0x80]),
        ("mov rsp, qword ptr [rax - 129]", |a| { a.ld(RSP, mem(RAX, -129)); }, &[0x48, 0x8b, 0xa0, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [rax + 305419896]", |a| { a.ld(RSP, mem(RAX, 305419896)); }, &[0x48, 0x8b, 0xa0, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [rax - 305419896]", |a| { a.ld(RSP, mem(RAX, -305419896)); }, &[0x48, 0x8b, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [rsp]", |a| { a.ld(RSP, mem(RSP, 0)); }, &[0x48, 0x8b, 0x24, 0x24]),
        ("mov rsp, qword ptr [rsp + 1]", |a| { a.ld(RSP, mem(RSP, 1)); }, &[0x48, 0x8b, 0x64, 0x24, 0x01]),
        ("mov rsp, qword ptr [rsp - 1]", |a| { a.ld(RSP, mem(RSP, -1)); }, &[0x48, 0x8b, 0x64, 0x24, 0xff]),
        ("mov rsp, qword ptr [rsp + 127]", |a| { a.ld(RSP, mem(RSP, 127)); }, &[0x48, 0x8b, 0x64, 0x24, 0x7f]),
        ("mov rsp, qword ptr [rsp + 128]", |a| { a.ld(RSP, mem(RSP, 128)); }, &[0x48, 0x8b, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp - 128]", |a| { a.ld(RSP, mem(RSP, -128)); }, &[0x48, 0x8b, 0x64, 0x24, 0x80]),
        ("mov rsp, qword ptr [rsp - 129]", |a| { a.ld(RSP, mem(RSP, -129)); }, &[0x48, 0x8b, 0xa4, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [rsp + 305419896]", |a| { a.ld(RSP, mem(RSP, 305419896)); }, &[0x48, 0x8b, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [rsp - 305419896]", |a| { a.ld(RSP, mem(RSP, -305419896)); }, &[0x48, 0x8b, 0xa4, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [rbp]", |a| { a.ld(RSP, mem(RBP, 0)); }, &[0x48, 0x8b, 0x65, 0x00]),
        ("mov rsp, qword ptr [rbp + 1]", |a| { a.ld(RSP, mem(RBP, 1)); }, &[0x48, 0x8b, 0x65, 0x01]),
        ("mov rsp, qword ptr [rbp - 1]", |a| { a.ld(RSP, mem(RBP, -1)); }, &[0x48, 0x8b, 0x65, 0xff]),
        ("mov rsp, qword ptr [rbp + 127]", |a| { a.ld(RSP, mem(RBP, 127)); }, &[0x48, 0x8b, 0x65, 0x7f]),
        ("mov rsp, qword ptr [rbp + 128]", |a| { a.ld(RSP, mem(RBP, 128)); }, &[0x48, 0x8b, 0xa5, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp - 128]", |a| { a.ld(RSP, mem(RBP, -128)); }, &[0x48, 0x8b, 0x65, 0x80]),
        ("mov rsp, qword ptr [rbp - 129]", |a| { a.ld(RSP, mem(RBP, -129)); }, &[0x48, 0x8b, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [rbp + 305419896]", |a| { a.ld(RSP, mem(RBP, 305419896)); }, &[0x48, 0x8b, 0xa5, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [rbp - 305419896]", |a| { a.ld(RSP, mem(RBP, -305419896)); }, &[0x48, 0x8b, 0xa5, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [r8]", |a| { a.ld(RSP, mem(R8, 0)); }, &[0x49, 0x8b, 0x20]),
        ("mov rsp, qword ptr [r8 + 1]", |a| { a.ld(RSP, mem(R8, 1)); }, &[0x49, 0x8b, 0x60, 0x01]),
        ("mov rsp, qword ptr [r8 - 1]", |a| { a.ld(RSP, mem(R8, -1)); }, &[0x49, 0x8b, 0x60, 0xff]),
        ("mov rsp, qword ptr [r8 + 127]", |a| { a.ld(RSP, mem(R8, 127)); }, &[0x49, 0x8b, 0x60, 0x7f]),
        ("mov rsp, qword ptr [r8 + 128]", |a| { a.ld(RSP, mem(R8, 128)); }, &[0x49, 0x8b, 0xa0, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [r8 - 128]", |a| { a.ld(RSP, mem(R8, -128)); }, &[0x49, 0x8b, 0x60, 0x80]),
        ("mov rsp, qword ptr [r8 - 129]", |a| { a.ld(RSP, mem(R8, -129)); }, &[0x49, 0x8b, 0xa0, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [r8 + 305419896]", |a| { a.ld(RSP, mem(R8, 305419896)); }, &[0x49, 0x8b, 0xa0, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [r8 - 305419896]", |a| { a.ld(RSP, mem(R8, -305419896)); }, &[0x49, 0x8b, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [r12]", |a| { a.ld(RSP, mem(R12, 0)); }, &[0x49, 0x8b, 0x24, 0x24]),
        ("mov rsp, qword ptr [r12 + 1]", |a| { a.ld(RSP, mem(R12, 1)); }, &[0x49, 0x8b, 0x64, 0x24, 0x01]),
        ("mov rsp, qword ptr [r12 - 1]", |a| { a.ld(RSP, mem(R12, -1)); }, &[0x49, 0x8b, 0x64, 0x24, 0xff]),
        ("mov rsp, qword ptr [r12 + 127]", |a| { a.ld(RSP, mem(R12, 127)); }, &[0x49, 0x8b, 0x64, 0x24, 0x7f]),
        ("mov rsp, qword ptr [r12 + 128]", |a| { a.ld(RSP, mem(R12, 128)); }, &[0x49, 0x8b, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 - 128]", |a| { a.ld(RSP, mem(R12, -128)); }, &[0x49, 0x8b, 0x64, 0x24, 0x80]),
        ("mov rsp, qword ptr [r12 - 129]", |a| { a.ld(RSP, mem(R12, -129)); }, &[0x49, 0x8b, 0xa4, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [r12 + 305419896]", |a| { a.ld(RSP, mem(R12, 305419896)); }, &[0x49, 0x8b, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [r12 - 305419896]", |a| { a.ld(RSP, mem(R12, -305419896)); }, &[0x49, 0x8b, 0xa4, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [r13]", |a| { a.ld(RSP, mem(R13, 0)); }, &[0x49, 0x8b, 0x65, 0x00]),
        ("mov rsp, qword ptr [r13 + 1]", |a| { a.ld(RSP, mem(R13, 1)); }, &[0x49, 0x8b, 0x65, 0x01]),
        ("mov rsp, qword ptr [r13 - 1]", |a| { a.ld(RSP, mem(R13, -1)); }, &[0x49, 0x8b, 0x65, 0xff]),
        ("mov rsp, qword ptr [r13 + 127]", |a| { a.ld(RSP, mem(R13, 127)); }, &[0x49, 0x8b, 0x65, 0x7f]),
        ("mov rsp, qword ptr [r13 + 128]", |a| { a.ld(RSP, mem(R13, 128)); }, &[0x49, 0x8b, 0xa5, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 - 128]", |a| { a.ld(RSP, mem(R13, -128)); }, &[0x49, 0x8b, 0x65, 0x80]),
        ("mov rsp, qword ptr [r13 - 129]", |a| { a.ld(RSP, mem(R13, -129)); }, &[0x49, 0x8b, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [r13 + 305419896]", |a| { a.ld(RSP, mem(R13, 305419896)); }, &[0x49, 0x8b, 0xa5, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [r13 - 305419896]", |a| { a.ld(RSP, mem(R13, -305419896)); }, &[0x49, 0x8b, 0xa5, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [r15]", |a| { a.ld(RSP, mem(R15, 0)); }, &[0x49, 0x8b, 0x27]),
        ("mov rsp, qword ptr [r15 + 1]", |a| { a.ld(RSP, mem(R15, 1)); }, &[0x49, 0x8b, 0x67, 0x01]),
        ("mov rsp, qword ptr [r15 - 1]", |a| { a.ld(RSP, mem(R15, -1)); }, &[0x49, 0x8b, 0x67, 0xff]),
        ("mov rsp, qword ptr [r15 + 127]", |a| { a.ld(RSP, mem(R15, 127)); }, &[0x49, 0x8b, 0x67, 0x7f]),
        ("mov rsp, qword ptr [r15 + 128]", |a| { a.ld(RSP, mem(R15, 128)); }, &[0x49, 0x8b, 0xa7, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 - 128]", |a| { a.ld(RSP, mem(R15, -128)); }, &[0x49, 0x8b, 0x67, 0x80]),
        ("mov rsp, qword ptr [r15 - 129]", |a| { a.ld(RSP, mem(R15, -129)); }, &[0x49, 0x8b, 0xa7, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [r15 + 305419896]", |a| { a.ld(RSP, mem(R15, 305419896)); }, &[0x49, 0x8b, 0xa7, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [r15 - 305419896]", |a| { a.ld(RSP, mem(R15, -305419896)); }, &[0x49, 0x8b, 0xa7, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [rbx]", |a| { a.ld(RSP, mem(RBX, 0)); }, &[0x48, 0x8b, 0x23]),
        ("mov rsp, qword ptr [rbx + 1]", |a| { a.ld(RSP, mem(RBX, 1)); }, &[0x48, 0x8b, 0x63, 0x01]),
        ("mov rsp, qword ptr [rbx - 1]", |a| { a.ld(RSP, mem(RBX, -1)); }, &[0x48, 0x8b, 0x63, 0xff]),
        ("mov rsp, qword ptr [rbx + 127]", |a| { a.ld(RSP, mem(RBX, 127)); }, &[0x48, 0x8b, 0x63, 0x7f]),
        ("mov rsp, qword ptr [rbx + 128]", |a| { a.ld(RSP, mem(RBX, 128)); }, &[0x48, 0x8b, 0xa3, 0x80, 0x00, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbx - 128]", |a| { a.ld(RSP, mem(RBX, -128)); }, &[0x48, 0x8b, 0x63, 0x80]),
        ("mov rsp, qword ptr [rbx - 129]", |a| { a.ld(RSP, mem(RBX, -129)); }, &[0x48, 0x8b, 0xa3, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rsp, qword ptr [rbx + 305419896]", |a| { a.ld(RSP, mem(RBX, 305419896)); }, &[0x48, 0x8b, 0xa3, 0x78, 0x56, 0x34, 0x12]),
        ("mov rsp, qword ptr [rbx - 305419896]", |a| { a.ld(RSP, mem(RBX, -305419896)); }, &[0x48, 0x8b, 0xa3, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rsp, qword ptr [rax + rax*1]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x00]),
        ("mov rsp, qword ptr [rax + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x00, 0x08]),
        ("mov rsp, qword ptr [rax + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rax*8]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xc0]),
        ("mov rsp, qword ptr [rax + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xc0, 0x08]),
        ("mov rsp, qword ptr [rax + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RAX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rcx*1]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x08]),
        ("mov rsp, qword ptr [rax + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x08, 0x08]),
        ("mov rsp, qword ptr [rax + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x08, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rcx*8]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xc8]),
        ("mov rsp, qword ptr [rax + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xc8, 0x08]),
        ("mov rsp, qword ptr [rax + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RCX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rbp*1]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x28]),
        ("mov rsp, qword ptr [rax + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x28, 0x08]),
        ("mov rsp, qword ptr [rax + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rbp*8]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xe8]),
        ("mov rsp, qword ptr [rax + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xe8, 0x08]),
        ("mov rsp, qword ptr [rax + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RBP, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rsi*1]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x30]),
        ("mov rsp, qword ptr [rax + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x30, 0x08]),
        ("mov rsp, qword ptr [rax + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x30, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + rsi*8]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xf0]),
        ("mov rsp, qword ptr [rax + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xf0, 0x08]),
        ("mov rsp, qword ptr [rax + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, RSI, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r8*1]", |a| { a.ld(RSP, mem_idx(RAX, R8, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x00]),
        ("mov rsp, qword ptr [rax + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R8, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x00, 0x08]),
        ("mov rsp, qword ptr [rax + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R8, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r8*8]", |a| { a.ld(RSP, mem_idx(RAX, R8, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xc0]),
        ("mov rsp, qword ptr [rax + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R8, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xc0, 0x08]),
        ("mov rsp, qword ptr [rax + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R8, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r12*1]", |a| { a.ld(RSP, mem_idx(RAX, R12, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x20]),
        ("mov rsp, qword ptr [rax + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R12, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x20, 0x08]),
        ("mov rsp, qword ptr [rax + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R12, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x20, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r12*8]", |a| { a.ld(RSP, mem_idx(RAX, R12, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xe0]),
        ("mov rsp, qword ptr [rax + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R12, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xe0, 0x08]),
        ("mov rsp, qword ptr [rax + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R12, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r13*1]", |a| { a.ld(RSP, mem_idx(RAX, R13, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x28]),
        ("mov rsp, qword ptr [rax + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R13, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x28, 0x08]),
        ("mov rsp, qword ptr [rax + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R13, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r13*8]", |a| { a.ld(RSP, mem_idx(RAX, R13, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xe8]),
        ("mov rsp, qword ptr [rax + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R13, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xe8, 0x08]),
        ("mov rsp, qword ptr [rax + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R13, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r15*1]", |a| { a.ld(RSP, mem_idx(RAX, R15, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x38]),
        ("mov rsp, qword ptr [rax + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R15, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x38, 0x08]),
        ("mov rsp, qword ptr [rax + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R15, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x38, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rax + r15*8]", |a| { a.ld(RSP, mem_idx(RAX, R15, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xf8]),
        ("mov rsp, qword ptr [rax + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(RAX, R15, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xf8, 0x08]),
        ("mov rsp, qword ptr [rax + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(RAX, R15, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rax*1]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x04]),
        ("mov rsp, qword ptr [rsp + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x04, 0x08]),
        ("mov rsp, qword ptr [rsp + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rax*8]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xc4]),
        ("mov rsp, qword ptr [rsp + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xc4, 0x08]),
        ("mov rsp, qword ptr [rsp + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RAX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rcx*1]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x0c]),
        ("mov rsp, qword ptr [rsp + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x0c, 0x08]),
        ("mov rsp, qword ptr [rsp + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rcx*8]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xcc]),
        ("mov rsp, qword ptr [rsp + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xcc, 0x08]),
        ("mov rsp, qword ptr [rsp + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RCX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rbp*1]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x2c]),
        ("mov rsp, qword ptr [rsp + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x2c, 0x08]),
        ("mov rsp, qword ptr [rsp + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rbp*8]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xec]),
        ("mov rsp, qword ptr [rsp + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xec, 0x08]),
        ("mov rsp, qword ptr [rsp + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RBP, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rsi*1]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 0, 0)); }, &[0x48, 0x8b, 0x24, 0x34]),
        ("mov rsp, qword ptr [rsp + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x34, 0x08]),
        ("mov rsp, qword ptr [rsp + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + rsi*8]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 3, 0)); }, &[0x48, 0x8b, 0x24, 0xf4]),
        ("mov rsp, qword ptr [rsp + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xf4, 0x08]),
        ("mov rsp, qword ptr [rsp + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, RSI, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r8*1]", |a| { a.ld(RSP, mem_idx(RSP, R8, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x04]),
        ("mov rsp, qword ptr [rsp + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R8, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x04, 0x08]),
        ("mov rsp, qword ptr [rsp + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R8, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r8*8]", |a| { a.ld(RSP, mem_idx(RSP, R8, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xc4]),
        ("mov rsp, qword ptr [rsp + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R8, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xc4, 0x08]),
        ("mov rsp, qword ptr [rsp + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R8, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r12*1]", |a| { a.ld(RSP, mem_idx(RSP, R12, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x24]),
        ("mov rsp, qword ptr [rsp + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R12, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x24, 0x08]),
        ("mov rsp, qword ptr [rsp + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R12, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r12*8]", |a| { a.ld(RSP, mem_idx(RSP, R12, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xe4]),
        ("mov rsp, qword ptr [rsp + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R12, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xe4, 0x08]),
        ("mov rsp, qword ptr [rsp + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R12, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r13*1]", |a| { a.ld(RSP, mem_idx(RSP, R13, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x2c]),
        ("mov rsp, qword ptr [rsp + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R13, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x2c, 0x08]),
        ("mov rsp, qword ptr [rsp + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R13, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r13*8]", |a| { a.ld(RSP, mem_idx(RSP, R13, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xec]),
        ("mov rsp, qword ptr [rsp + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R13, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xec, 0x08]),
        ("mov rsp, qword ptr [rsp + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R13, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r15*1]", |a| { a.ld(RSP, mem_idx(RSP, R15, 0, 0)); }, &[0x4a, 0x8b, 0x24, 0x3c]),
        ("mov rsp, qword ptr [rsp + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R15, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x3c, 0x08]),
        ("mov rsp, qword ptr [rsp + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R15, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rsp + r15*8]", |a| { a.ld(RSP, mem_idx(RSP, R15, 3, 0)); }, &[0x4a, 0x8b, 0x24, 0xfc]),
        ("mov rsp, qword ptr [rsp + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(RSP, R15, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xfc, 0x08]),
        ("mov rsp, qword ptr [rsp + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(RSP, R15, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rax*1]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 0, 0)); }, &[0x48, 0x8b, 0x64, 0x05, 0x00]),
        ("mov rsp, qword ptr [rbp + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x05, 0x08]),
        ("mov rsp, qword ptr [rbp + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rax*8]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 3, 0)); }, &[0x48, 0x8b, 0x64, 0xc5, 0x00]),
        ("mov rsp, qword ptr [rbp + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xc5, 0x08]),
        ("mov rsp, qword ptr [rbp + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RAX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rcx*1]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 0, 0)); }, &[0x48, 0x8b, 0x64, 0x0d, 0x00]),
        ("mov rsp, qword ptr [rbp + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x0d, 0x08]),
        ("mov rsp, qword ptr [rbp + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rcx*8]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 3, 0)); }, &[0x48, 0x8b, 0x64, 0xcd, 0x00]),
        ("mov rsp, qword ptr [rbp + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xcd, 0x08]),
        ("mov rsp, qword ptr [rbp + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rbp*1]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 0, 0)); }, &[0x48, 0x8b, 0x64, 0x2d, 0x00]),
        ("mov rsp, qword ptr [rbp + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x2d, 0x08]),
        ("mov rsp, qword ptr [rbp + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rbp*8]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 3, 0)); }, &[0x48, 0x8b, 0x64, 0xed, 0x00]),
        ("mov rsp, qword ptr [rbp + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xed, 0x08]),
        ("mov rsp, qword ptr [rbp + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RBP, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rsi*1]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 0, 0)); }, &[0x48, 0x8b, 0x64, 0x35, 0x00]),
        ("mov rsp, qword ptr [rbp + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 0, 8)); }, &[0x48, 0x8b, 0x64, 0x35, 0x08]),
        ("mov rsp, qword ptr [rbp + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 0, 4096)); }, &[0x48, 0x8b, 0xa4, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + rsi*8]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 3, 0)); }, &[0x48, 0x8b, 0x64, 0xf5, 0x00]),
        ("mov rsp, qword ptr [rbp + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 3, 8)); }, &[0x48, 0x8b, 0x64, 0xf5, 0x08]),
        ("mov rsp, qword ptr [rbp + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, RSI, 3, 4096)); }, &[0x48, 0x8b, 0xa4, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r8*1]", |a| { a.ld(RSP, mem_idx(RBP, R8, 0, 0)); }, &[0x4a, 0x8b, 0x64, 0x05, 0x00]),
        ("mov rsp, qword ptr [rbp + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R8, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x05, 0x08]),
        ("mov rsp, qword ptr [rbp + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R8, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r8*8]", |a| { a.ld(RSP, mem_idx(RBP, R8, 3, 0)); }, &[0x4a, 0x8b, 0x64, 0xc5, 0x00]),
        ("mov rsp, qword ptr [rbp + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R8, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xc5, 0x08]),
        ("mov rsp, qword ptr [rbp + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R8, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r12*1]", |a| { a.ld(RSP, mem_idx(RBP, R12, 0, 0)); }, &[0x4a, 0x8b, 0x64, 0x25, 0x00]),
        ("mov rsp, qword ptr [rbp + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R12, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x25, 0x08]),
        ("mov rsp, qword ptr [rbp + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R12, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r12*8]", |a| { a.ld(RSP, mem_idx(RBP, R12, 3, 0)); }, &[0x4a, 0x8b, 0x64, 0xe5, 0x00]),
        ("mov rsp, qword ptr [rbp + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R12, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xe5, 0x08]),
        ("mov rsp, qword ptr [rbp + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R12, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r13*1]", |a| { a.ld(RSP, mem_idx(RBP, R13, 0, 0)); }, &[0x4a, 0x8b, 0x64, 0x2d, 0x00]),
        ("mov rsp, qword ptr [rbp + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R13, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x2d, 0x08]),
        ("mov rsp, qword ptr [rbp + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R13, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r13*8]", |a| { a.ld(RSP, mem_idx(RBP, R13, 3, 0)); }, &[0x4a, 0x8b, 0x64, 0xed, 0x00]),
        ("mov rsp, qword ptr [rbp + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R13, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xed, 0x08]),
        ("mov rsp, qword ptr [rbp + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R13, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r15*1]", |a| { a.ld(RSP, mem_idx(RBP, R15, 0, 0)); }, &[0x4a, 0x8b, 0x64, 0x3d, 0x00]),
        ("mov rsp, qword ptr [rbp + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R15, 0, 8)); }, &[0x4a, 0x8b, 0x64, 0x3d, 0x08]),
        ("mov rsp, qword ptr [rbp + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R15, 0, 4096)); }, &[0x4a, 0x8b, 0xa4, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbp + r15*8]", |a| { a.ld(RSP, mem_idx(RBP, R15, 3, 0)); }, &[0x4a, 0x8b, 0x64, 0xfd, 0x00]),
        ("mov rsp, qword ptr [rbp + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(RBP, R15, 3, 8)); }, &[0x4a, 0x8b, 0x64, 0xfd, 0x08]),
        ("mov rsp, qword ptr [rbp + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(RBP, R15, 3, 4096)); }, &[0x4a, 0x8b, 0xa4, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rax*1]", |a| { a.ld(RSP, mem_idx(R12, RAX, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x04]),
        ("mov rsp, qword ptr [r12 + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, RAX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x04, 0x08]),
        ("mov rsp, qword ptr [r12 + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RAX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rax*8]", |a| { a.ld(RSP, mem_idx(R12, RAX, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xc4]),
        ("mov rsp, qword ptr [r12 + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, RAX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xc4, 0x08]),
        ("mov rsp, qword ptr [r12 + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RAX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rcx*1]", |a| { a.ld(RSP, mem_idx(R12, RCX, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x0c]),
        ("mov rsp, qword ptr [r12 + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, RCX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x0c, 0x08]),
        ("mov rsp, qword ptr [r12 + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RCX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rcx*8]", |a| { a.ld(RSP, mem_idx(R12, RCX, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xcc]),
        ("mov rsp, qword ptr [r12 + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, RCX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xcc, 0x08]),
        ("mov rsp, qword ptr [r12 + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RCX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rbp*1]", |a| { a.ld(RSP, mem_idx(R12, RBP, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x2c]),
        ("mov rsp, qword ptr [r12 + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, RBP, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x2c, 0x08]),
        ("mov rsp, qword ptr [r12 + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RBP, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rbp*8]", |a| { a.ld(RSP, mem_idx(R12, RBP, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xec]),
        ("mov rsp, qword ptr [r12 + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, RBP, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xec, 0x08]),
        ("mov rsp, qword ptr [r12 + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RBP, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rsi*1]", |a| { a.ld(RSP, mem_idx(R12, RSI, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x34]),
        ("mov rsp, qword ptr [r12 + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, RSI, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x34, 0x08]),
        ("mov rsp, qword ptr [r12 + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RSI, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + rsi*8]", |a| { a.ld(RSP, mem_idx(R12, RSI, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xf4]),
        ("mov rsp, qword ptr [r12 + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, RSI, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xf4, 0x08]),
        ("mov rsp, qword ptr [r12 + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, RSI, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r8*1]", |a| { a.ld(RSP, mem_idx(R12, R8, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x04]),
        ("mov rsp, qword ptr [r12 + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, R8, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x04, 0x08]),
        ("mov rsp, qword ptr [r12 + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R8, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r8*8]", |a| { a.ld(RSP, mem_idx(R12, R8, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xc4]),
        ("mov rsp, qword ptr [r12 + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, R8, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xc4, 0x08]),
        ("mov rsp, qword ptr [r12 + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R8, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r12*1]", |a| { a.ld(RSP, mem_idx(R12, R12, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x24]),
        ("mov rsp, qword ptr [r12 + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, R12, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x24, 0x08]),
        ("mov rsp, qword ptr [r12 + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R12, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r12*8]", |a| { a.ld(RSP, mem_idx(R12, R12, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xe4]),
        ("mov rsp, qword ptr [r12 + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, R12, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xe4, 0x08]),
        ("mov rsp, qword ptr [r12 + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R12, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r13*1]", |a| { a.ld(RSP, mem_idx(R12, R13, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x2c]),
        ("mov rsp, qword ptr [r12 + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, R13, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x2c, 0x08]),
        ("mov rsp, qword ptr [r12 + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R13, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r13*8]", |a| { a.ld(RSP, mem_idx(R12, R13, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xec]),
        ("mov rsp, qword ptr [r12 + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, R13, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xec, 0x08]),
        ("mov rsp, qword ptr [r12 + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R13, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r15*1]", |a| { a.ld(RSP, mem_idx(R12, R15, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x3c]),
        ("mov rsp, qword ptr [r12 + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(R12, R15, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x3c, 0x08]),
        ("mov rsp, qword ptr [r12 + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R15, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r12 + r15*8]", |a| { a.ld(RSP, mem_idx(R12, R15, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xfc]),
        ("mov rsp, qword ptr [r12 + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(R12, R15, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xfc, 0x08]),
        ("mov rsp, qword ptr [r12 + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(R12, R15, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rax*1]", |a| { a.ld(RSP, mem_idx(R13, RAX, 0, 0)); }, &[0x49, 0x8b, 0x64, 0x05, 0x00]),
        ("mov rsp, qword ptr [r13 + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, RAX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x05, 0x08]),
        ("mov rsp, qword ptr [r13 + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RAX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rax*8]", |a| { a.ld(RSP, mem_idx(R13, RAX, 3, 0)); }, &[0x49, 0x8b, 0x64, 0xc5, 0x00]),
        ("mov rsp, qword ptr [r13 + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, RAX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xc5, 0x08]),
        ("mov rsp, qword ptr [r13 + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RAX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rcx*1]", |a| { a.ld(RSP, mem_idx(R13, RCX, 0, 0)); }, &[0x49, 0x8b, 0x64, 0x0d, 0x00]),
        ("mov rsp, qword ptr [r13 + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, RCX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x0d, 0x08]),
        ("mov rsp, qword ptr [r13 + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RCX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rcx*8]", |a| { a.ld(RSP, mem_idx(R13, RCX, 3, 0)); }, &[0x49, 0x8b, 0x64, 0xcd, 0x00]),
        ("mov rsp, qword ptr [r13 + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, RCX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xcd, 0x08]),
        ("mov rsp, qword ptr [r13 + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RCX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rbp*1]", |a| { a.ld(RSP, mem_idx(R13, RBP, 0, 0)); }, &[0x49, 0x8b, 0x64, 0x2d, 0x00]),
        ("mov rsp, qword ptr [r13 + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, RBP, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x2d, 0x08]),
        ("mov rsp, qword ptr [r13 + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RBP, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rbp*8]", |a| { a.ld(RSP, mem_idx(R13, RBP, 3, 0)); }, &[0x49, 0x8b, 0x64, 0xed, 0x00]),
        ("mov rsp, qword ptr [r13 + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, RBP, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xed, 0x08]),
        ("mov rsp, qword ptr [r13 + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RBP, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rsi*1]", |a| { a.ld(RSP, mem_idx(R13, RSI, 0, 0)); }, &[0x49, 0x8b, 0x64, 0x35, 0x00]),
        ("mov rsp, qword ptr [r13 + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, RSI, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x35, 0x08]),
        ("mov rsp, qword ptr [r13 + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RSI, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + rsi*8]", |a| { a.ld(RSP, mem_idx(R13, RSI, 3, 0)); }, &[0x49, 0x8b, 0x64, 0xf5, 0x00]),
        ("mov rsp, qword ptr [r13 + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, RSI, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xf5, 0x08]),
        ("mov rsp, qword ptr [r13 + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, RSI, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r8*1]", |a| { a.ld(RSP, mem_idx(R13, R8, 0, 0)); }, &[0x4b, 0x8b, 0x64, 0x05, 0x00]),
        ("mov rsp, qword ptr [r13 + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, R8, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x05, 0x08]),
        ("mov rsp, qword ptr [r13 + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R8, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r8*8]", |a| { a.ld(RSP, mem_idx(R13, R8, 3, 0)); }, &[0x4b, 0x8b, 0x64, 0xc5, 0x00]),
        ("mov rsp, qword ptr [r13 + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, R8, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xc5, 0x08]),
        ("mov rsp, qword ptr [r13 + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R8, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r12*1]", |a| { a.ld(RSP, mem_idx(R13, R12, 0, 0)); }, &[0x4b, 0x8b, 0x64, 0x25, 0x00]),
        ("mov rsp, qword ptr [r13 + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, R12, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x25, 0x08]),
        ("mov rsp, qword ptr [r13 + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R12, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r12*8]", |a| { a.ld(RSP, mem_idx(R13, R12, 3, 0)); }, &[0x4b, 0x8b, 0x64, 0xe5, 0x00]),
        ("mov rsp, qword ptr [r13 + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, R12, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xe5, 0x08]),
        ("mov rsp, qword ptr [r13 + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R12, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r13*1]", |a| { a.ld(RSP, mem_idx(R13, R13, 0, 0)); }, &[0x4b, 0x8b, 0x64, 0x2d, 0x00]),
        ("mov rsp, qword ptr [r13 + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, R13, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x2d, 0x08]),
        ("mov rsp, qword ptr [r13 + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R13, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r13*8]", |a| { a.ld(RSP, mem_idx(R13, R13, 3, 0)); }, &[0x4b, 0x8b, 0x64, 0xed, 0x00]),
        ("mov rsp, qword ptr [r13 + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, R13, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xed, 0x08]),
        ("mov rsp, qword ptr [r13 + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R13, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r15*1]", |a| { a.ld(RSP, mem_idx(R13, R15, 0, 0)); }, &[0x4b, 0x8b, 0x64, 0x3d, 0x00]),
        ("mov rsp, qword ptr [r13 + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(R13, R15, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x3d, 0x08]),
        ("mov rsp, qword ptr [r13 + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R15, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r13 + r15*8]", |a| { a.ld(RSP, mem_idx(R13, R15, 3, 0)); }, &[0x4b, 0x8b, 0x64, 0xfd, 0x00]),
        ("mov rsp, qword ptr [r13 + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(R13, R15, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xfd, 0x08]),
        ("mov rsp, qword ptr [r13 + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(R13, R15, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rax*1]", |a| { a.ld(RSP, mem_idx(R15, RAX, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x07]),
        ("mov rsp, qword ptr [r15 + rax*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, RAX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x07, 0x08]),
        ("mov rsp, qword ptr [r15 + rax*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RAX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rax*8]", |a| { a.ld(RSP, mem_idx(R15, RAX, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xc7]),
        ("mov rsp, qword ptr [r15 + rax*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, RAX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xc7, 0x08]),
        ("mov rsp, qword ptr [r15 + rax*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RAX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rcx*1]", |a| { a.ld(RSP, mem_idx(R15, RCX, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x0f]),
        ("mov rsp, qword ptr [r15 + rcx*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, RCX, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x0f, 0x08]),
        ("mov rsp, qword ptr [r15 + rcx*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RCX, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x0f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rcx*8]", |a| { a.ld(RSP, mem_idx(R15, RCX, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xcf]),
        ("mov rsp, qword ptr [r15 + rcx*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, RCX, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xcf, 0x08]),
        ("mov rsp, qword ptr [r15 + rcx*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RCX, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rbp*1]", |a| { a.ld(RSP, mem_idx(R15, RBP, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x2f]),
        ("mov rsp, qword ptr [r15 + rbp*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, RBP, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x2f, 0x08]),
        ("mov rsp, qword ptr [r15 + rbp*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RBP, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rbp*8]", |a| { a.ld(RSP, mem_idx(R15, RBP, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xef]),
        ("mov rsp, qword ptr [r15 + rbp*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, RBP, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xef, 0x08]),
        ("mov rsp, qword ptr [r15 + rbp*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RBP, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rsi*1]", |a| { a.ld(RSP, mem_idx(R15, RSI, 0, 0)); }, &[0x49, 0x8b, 0x24, 0x37]),
        ("mov rsp, qword ptr [r15 + rsi*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, RSI, 0, 8)); }, &[0x49, 0x8b, 0x64, 0x37, 0x08]),
        ("mov rsp, qword ptr [r15 + rsi*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RSI, 0, 4096)); }, &[0x49, 0x8b, 0xa4, 0x37, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + rsi*8]", |a| { a.ld(RSP, mem_idx(R15, RSI, 3, 0)); }, &[0x49, 0x8b, 0x24, 0xf7]),
        ("mov rsp, qword ptr [r15 + rsi*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, RSI, 3, 8)); }, &[0x49, 0x8b, 0x64, 0xf7, 0x08]),
        ("mov rsp, qword ptr [r15 + rsi*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, RSI, 3, 4096)); }, &[0x49, 0x8b, 0xa4, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r8*1]", |a| { a.ld(RSP, mem_idx(R15, R8, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x07]),
        ("mov rsp, qword ptr [r15 + r8*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, R8, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x07, 0x08]),
        ("mov rsp, qword ptr [r15 + r8*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R8, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r8*8]", |a| { a.ld(RSP, mem_idx(R15, R8, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xc7]),
        ("mov rsp, qword ptr [r15 + r8*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, R8, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xc7, 0x08]),
        ("mov rsp, qword ptr [r15 + r8*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R8, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r12*1]", |a| { a.ld(RSP, mem_idx(R15, R12, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x27]),
        ("mov rsp, qword ptr [r15 + r12*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, R12, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x27, 0x08]),
        ("mov rsp, qword ptr [r15 + r12*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R12, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x27, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r12*8]", |a| { a.ld(RSP, mem_idx(R15, R12, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xe7]),
        ("mov rsp, qword ptr [r15 + r12*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, R12, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xe7, 0x08]),
        ("mov rsp, qword ptr [r15 + r12*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R12, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r13*1]", |a| { a.ld(RSP, mem_idx(R15, R13, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x2f]),
        ("mov rsp, qword ptr [r15 + r13*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, R13, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x2f, 0x08]),
        ("mov rsp, qword ptr [r15 + r13*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R13, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r13*8]", |a| { a.ld(RSP, mem_idx(R15, R13, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xef]),
        ("mov rsp, qword ptr [r15 + r13*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, R13, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xef, 0x08]),
        ("mov rsp, qword ptr [r15 + r13*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R13, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r15*1]", |a| { a.ld(RSP, mem_idx(R15, R15, 0, 0)); }, &[0x4b, 0x8b, 0x24, 0x3f]),
        ("mov rsp, qword ptr [r15 + r15*1 + 8]", |a| { a.ld(RSP, mem_idx(R15, R15, 0, 8)); }, &[0x4b, 0x8b, 0x64, 0x3f, 0x08]),
        ("mov rsp, qword ptr [r15 + r15*1 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R15, 0, 4096)); }, &[0x4b, 0x8b, 0xa4, 0x3f, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [r15 + r15*8]", |a| { a.ld(RSP, mem_idx(R15, R15, 3, 0)); }, &[0x4b, 0x8b, 0x24, 0xff]),
        ("mov rsp, qword ptr [r15 + r15*8 + 8]", |a| { a.ld(RSP, mem_idx(R15, R15, 3, 8)); }, &[0x4b, 0x8b, 0x64, 0xff, 0x08]),
        ("mov rsp, qword ptr [r15 + r15*8 + 4096]", |a| { a.ld(RSP, mem_idx(R15, R15, 3, 4096)); }, &[0x4b, 0x8b, 0xa4, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("mov rsp, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld(RSP, mem_idx(RBX, RCX, 1, 32)); }, &[0x48, 0x8b, 0x64, 0x4b, 0x20]),
        ("mov rsp, qword ptr [rbx + r12*2 + 32]", |a| { a.ld(RSP, mem_idx(RBX, R12, 1, 32)); }, &[0x4a, 0x8b, 0x64, 0x63, 0x20]),
        ("mov rsp, qword ptr [rbp + rcx*2 + 32]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 1, 32)); }, &[0x48, 0x8b, 0x64, 0x4d, 0x20]),
        ("mov rsp, qword ptr [rbp + r12*2 + 32]", |a| { a.ld(RSP, mem_idx(RBP, R12, 1, 32)); }, &[0x4a, 0x8b, 0x64, 0x65, 0x20]),
        ("mov rsp, qword ptr [rbx + rcx*4 + 32]", |a| { a.ld(RSP, mem_idx(RBX, RCX, 2, 32)); }, &[0x48, 0x8b, 0x64, 0x8b, 0x20]),
        ("mov rsp, qword ptr [rbx + r12*4 + 32]", |a| { a.ld(RSP, mem_idx(RBX, R12, 2, 32)); }, &[0x4a, 0x8b, 0x64, 0xa3, 0x20]),
        ("mov rsp, qword ptr [rbp + rcx*4 + 32]", |a| { a.ld(RSP, mem_idx(RBP, RCX, 2, 32)); }, &[0x48, 0x8b, 0x64, 0x8d, 0x20]),
        ("mov rsp, qword ptr [rbp + r12*4 + 32]", |a| { a.ld(RSP, mem_idx(RBP, R12, 2, 32)); }, &[0x4a, 0x8b, 0x64, 0xa5, 0x20]),
        ("mov r15, qword ptr [rax]", |a| { a.ld(R15, mem(RAX, 0)); }, &[0x4c, 0x8b, 0x38]),
        ("mov r15, qword ptr [rax + 1]", |a| { a.ld(R15, mem(RAX, 1)); }, &[0x4c, 0x8b, 0x78, 0x01]),
        ("mov r15, qword ptr [rax - 1]", |a| { a.ld(R15, mem(RAX, -1)); }, &[0x4c, 0x8b, 0x78, 0xff]),
        ("mov r15, qword ptr [rax + 127]", |a| { a.ld(R15, mem(RAX, 127)); }, &[0x4c, 0x8b, 0x78, 0x7f]),
        ("mov r15, qword ptr [rax + 128]", |a| { a.ld(R15, mem(RAX, 128)); }, &[0x4c, 0x8b, 0xb8, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [rax - 128]", |a| { a.ld(R15, mem(RAX, -128)); }, &[0x4c, 0x8b, 0x78, 0x80]),
        ("mov r15, qword ptr [rax - 129]", |a| { a.ld(R15, mem(RAX, -129)); }, &[0x4c, 0x8b, 0xb8, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [rax + 305419896]", |a| { a.ld(R15, mem(RAX, 305419896)); }, &[0x4c, 0x8b, 0xb8, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [rax - 305419896]", |a| { a.ld(R15, mem(RAX, -305419896)); }, &[0x4c, 0x8b, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [rsp]", |a| { a.ld(R15, mem(RSP, 0)); }, &[0x4c, 0x8b, 0x3c, 0x24]),
        ("mov r15, qword ptr [rsp + 1]", |a| { a.ld(R15, mem(RSP, 1)); }, &[0x4c, 0x8b, 0x7c, 0x24, 0x01]),
        ("mov r15, qword ptr [rsp - 1]", |a| { a.ld(R15, mem(RSP, -1)); }, &[0x4c, 0x8b, 0x7c, 0x24, 0xff]),
        ("mov r15, qword ptr [rsp + 127]", |a| { a.ld(R15, mem(RSP, 127)); }, &[0x4c, 0x8b, 0x7c, 0x24, 0x7f]),
        ("mov r15, qword ptr [rsp + 128]", |a| { a.ld(R15, mem(RSP, 128)); }, &[0x4c, 0x8b, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp - 128]", |a| { a.ld(R15, mem(RSP, -128)); }, &[0x4c, 0x8b, 0x7c, 0x24, 0x80]),
        ("mov r15, qword ptr [rsp - 129]", |a| { a.ld(R15, mem(RSP, -129)); }, &[0x4c, 0x8b, 0xbc, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [rsp + 305419896]", |a| { a.ld(R15, mem(RSP, 305419896)); }, &[0x4c, 0x8b, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [rsp - 305419896]", |a| { a.ld(R15, mem(RSP, -305419896)); }, &[0x4c, 0x8b, 0xbc, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [rbp]", |a| { a.ld(R15, mem(RBP, 0)); }, &[0x4c, 0x8b, 0x7d, 0x00]),
        ("mov r15, qword ptr [rbp + 1]", |a| { a.ld(R15, mem(RBP, 1)); }, &[0x4c, 0x8b, 0x7d, 0x01]),
        ("mov r15, qword ptr [rbp - 1]", |a| { a.ld(R15, mem(RBP, -1)); }, &[0x4c, 0x8b, 0x7d, 0xff]),
        ("mov r15, qword ptr [rbp + 127]", |a| { a.ld(R15, mem(RBP, 127)); }, &[0x4c, 0x8b, 0x7d, 0x7f]),
        ("mov r15, qword ptr [rbp + 128]", |a| { a.ld(R15, mem(RBP, 128)); }, &[0x4c, 0x8b, 0xbd, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp - 128]", |a| { a.ld(R15, mem(RBP, -128)); }, &[0x4c, 0x8b, 0x7d, 0x80]),
        ("mov r15, qword ptr [rbp - 129]", |a| { a.ld(R15, mem(RBP, -129)); }, &[0x4c, 0x8b, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [rbp + 305419896]", |a| { a.ld(R15, mem(RBP, 305419896)); }, &[0x4c, 0x8b, 0xbd, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [rbp - 305419896]", |a| { a.ld(R15, mem(RBP, -305419896)); }, &[0x4c, 0x8b, 0xbd, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [r8]", |a| { a.ld(R15, mem(R8, 0)); }, &[0x4d, 0x8b, 0x38]),
        ("mov r15, qword ptr [r8 + 1]", |a| { a.ld(R15, mem(R8, 1)); }, &[0x4d, 0x8b, 0x78, 0x01]),
        ("mov r15, qword ptr [r8 - 1]", |a| { a.ld(R15, mem(R8, -1)); }, &[0x4d, 0x8b, 0x78, 0xff]),
        ("mov r15, qword ptr [r8 + 127]", |a| { a.ld(R15, mem(R8, 127)); }, &[0x4d, 0x8b, 0x78, 0x7f]),
        ("mov r15, qword ptr [r8 + 128]", |a| { a.ld(R15, mem(R8, 128)); }, &[0x4d, 0x8b, 0xb8, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [r8 - 128]", |a| { a.ld(R15, mem(R8, -128)); }, &[0x4d, 0x8b, 0x78, 0x80]),
        ("mov r15, qword ptr [r8 - 129]", |a| { a.ld(R15, mem(R8, -129)); }, &[0x4d, 0x8b, 0xb8, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [r8 + 305419896]", |a| { a.ld(R15, mem(R8, 305419896)); }, &[0x4d, 0x8b, 0xb8, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [r8 - 305419896]", |a| { a.ld(R15, mem(R8, -305419896)); }, &[0x4d, 0x8b, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [r12]", |a| { a.ld(R15, mem(R12, 0)); }, &[0x4d, 0x8b, 0x3c, 0x24]),
        ("mov r15, qword ptr [r12 + 1]", |a| { a.ld(R15, mem(R12, 1)); }, &[0x4d, 0x8b, 0x7c, 0x24, 0x01]),
        ("mov r15, qword ptr [r12 - 1]", |a| { a.ld(R15, mem(R12, -1)); }, &[0x4d, 0x8b, 0x7c, 0x24, 0xff]),
        ("mov r15, qword ptr [r12 + 127]", |a| { a.ld(R15, mem(R12, 127)); }, &[0x4d, 0x8b, 0x7c, 0x24, 0x7f]),
        ("mov r15, qword ptr [r12 + 128]", |a| { a.ld(R15, mem(R12, 128)); }, &[0x4d, 0x8b, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 - 128]", |a| { a.ld(R15, mem(R12, -128)); }, &[0x4d, 0x8b, 0x7c, 0x24, 0x80]),
        ("mov r15, qword ptr [r12 - 129]", |a| { a.ld(R15, mem(R12, -129)); }, &[0x4d, 0x8b, 0xbc, 0x24, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [r12 + 305419896]", |a| { a.ld(R15, mem(R12, 305419896)); }, &[0x4d, 0x8b, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [r12 - 305419896]", |a| { a.ld(R15, mem(R12, -305419896)); }, &[0x4d, 0x8b, 0xbc, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [r13]", |a| { a.ld(R15, mem(R13, 0)); }, &[0x4d, 0x8b, 0x7d, 0x00]),
        ("mov r15, qword ptr [r13 + 1]", |a| { a.ld(R15, mem(R13, 1)); }, &[0x4d, 0x8b, 0x7d, 0x01]),
        ("mov r15, qword ptr [r13 - 1]", |a| { a.ld(R15, mem(R13, -1)); }, &[0x4d, 0x8b, 0x7d, 0xff]),
        ("mov r15, qword ptr [r13 + 127]", |a| { a.ld(R15, mem(R13, 127)); }, &[0x4d, 0x8b, 0x7d, 0x7f]),
        ("mov r15, qword ptr [r13 + 128]", |a| { a.ld(R15, mem(R13, 128)); }, &[0x4d, 0x8b, 0xbd, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 - 128]", |a| { a.ld(R15, mem(R13, -128)); }, &[0x4d, 0x8b, 0x7d, 0x80]),
        ("mov r15, qword ptr [r13 - 129]", |a| { a.ld(R15, mem(R13, -129)); }, &[0x4d, 0x8b, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [r13 + 305419896]", |a| { a.ld(R15, mem(R13, 305419896)); }, &[0x4d, 0x8b, 0xbd, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [r13 - 305419896]", |a| { a.ld(R15, mem(R13, -305419896)); }, &[0x4d, 0x8b, 0xbd, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [r15]", |a| { a.ld(R15, mem(R15, 0)); }, &[0x4d, 0x8b, 0x3f]),
        ("mov r15, qword ptr [r15 + 1]", |a| { a.ld(R15, mem(R15, 1)); }, &[0x4d, 0x8b, 0x7f, 0x01]),
        ("mov r15, qword ptr [r15 - 1]", |a| { a.ld(R15, mem(R15, -1)); }, &[0x4d, 0x8b, 0x7f, 0xff]),
        ("mov r15, qword ptr [r15 + 127]", |a| { a.ld(R15, mem(R15, 127)); }, &[0x4d, 0x8b, 0x7f, 0x7f]),
        ("mov r15, qword ptr [r15 + 128]", |a| { a.ld(R15, mem(R15, 128)); }, &[0x4d, 0x8b, 0xbf, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 - 128]", |a| { a.ld(R15, mem(R15, -128)); }, &[0x4d, 0x8b, 0x7f, 0x80]),
        ("mov r15, qword ptr [r15 - 129]", |a| { a.ld(R15, mem(R15, -129)); }, &[0x4d, 0x8b, 0xbf, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [r15 + 305419896]", |a| { a.ld(R15, mem(R15, 305419896)); }, &[0x4d, 0x8b, 0xbf, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [r15 - 305419896]", |a| { a.ld(R15, mem(R15, -305419896)); }, &[0x4d, 0x8b, 0xbf, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [rbx]", |a| { a.ld(R15, mem(RBX, 0)); }, &[0x4c, 0x8b, 0x3b]),
        ("mov r15, qword ptr [rbx + 1]", |a| { a.ld(R15, mem(RBX, 1)); }, &[0x4c, 0x8b, 0x7b, 0x01]),
        ("mov r15, qword ptr [rbx - 1]", |a| { a.ld(R15, mem(RBX, -1)); }, &[0x4c, 0x8b, 0x7b, 0xff]),
        ("mov r15, qword ptr [rbx + 127]", |a| { a.ld(R15, mem(RBX, 127)); }, &[0x4c, 0x8b, 0x7b, 0x7f]),
        ("mov r15, qword ptr [rbx + 128]", |a| { a.ld(R15, mem(RBX, 128)); }, &[0x4c, 0x8b, 0xbb, 0x80, 0x00, 0x00, 0x00]),
        ("mov r15, qword ptr [rbx - 128]", |a| { a.ld(R15, mem(RBX, -128)); }, &[0x4c, 0x8b, 0x7b, 0x80]),
        ("mov r15, qword ptr [rbx - 129]", |a| { a.ld(R15, mem(RBX, -129)); }, &[0x4c, 0x8b, 0xbb, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r15, qword ptr [rbx + 305419896]", |a| { a.ld(R15, mem(RBX, 305419896)); }, &[0x4c, 0x8b, 0xbb, 0x78, 0x56, 0x34, 0x12]),
        ("mov r15, qword ptr [rbx - 305419896]", |a| { a.ld(R15, mem(RBX, -305419896)); }, &[0x4c, 0x8b, 0xbb, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r15, qword ptr [rax + rax*1]", |a| { a.ld(R15, mem_idx(RAX, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x00]),
        ("mov r15, qword ptr [rax + rax*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x00, 0x08]),
        ("mov r15, qword ptr [rax + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rax*8]", |a| { a.ld(R15, mem_idx(RAX, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xc0]),
        ("mov r15, qword ptr [rax + rax*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xc0, 0x08]),
        ("mov r15, qword ptr [rax + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rcx*1]", |a| { a.ld(R15, mem_idx(RAX, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x08]),
        ("mov r15, qword ptr [rax + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x08, 0x08]),
        ("mov r15, qword ptr [rax + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x08, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rcx*8]", |a| { a.ld(R15, mem_idx(RAX, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xc8]),
        ("mov r15, qword ptr [rax + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xc8, 0x08]),
        ("mov r15, qword ptr [rax + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rbp*1]", |a| { a.ld(R15, mem_idx(RAX, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x28]),
        ("mov r15, qword ptr [rax + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x28, 0x08]),
        ("mov r15, qword ptr [rax + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rbp*8]", |a| { a.ld(R15, mem_idx(RAX, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xe8]),
        ("mov r15, qword ptr [rax + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xe8, 0x08]),
        ("mov r15, qword ptr [rax + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rsi*1]", |a| { a.ld(R15, mem_idx(RAX, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x30]),
        ("mov r15, qword ptr [rax + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x30, 0x08]),
        ("mov r15, qword ptr [rax + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x30, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + rsi*8]", |a| { a.ld(R15, mem_idx(RAX, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xf0]),
        ("mov r15, qword ptr [rax + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xf0, 0x08]),
        ("mov r15, qword ptr [rax + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r8*1]", |a| { a.ld(R15, mem_idx(RAX, R8, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x00]),
        ("mov r15, qword ptr [rax + r8*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, R8, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x00, 0x08]),
        ("mov r15, qword ptr [rax + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R8, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x00, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r8*8]", |a| { a.ld(R15, mem_idx(RAX, R8, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xc0]),
        ("mov r15, qword ptr [rax + r8*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, R8, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xc0, 0x08]),
        ("mov r15, qword ptr [rax + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R8, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r12*1]", |a| { a.ld(R15, mem_idx(RAX, R12, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x20]),
        ("mov r15, qword ptr [rax + r12*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, R12, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x20, 0x08]),
        ("mov r15, qword ptr [rax + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R12, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x20, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r12*8]", |a| { a.ld(R15, mem_idx(RAX, R12, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xe0]),
        ("mov r15, qword ptr [rax + r12*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, R12, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xe0, 0x08]),
        ("mov r15, qword ptr [rax + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R12, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r13*1]", |a| { a.ld(R15, mem_idx(RAX, R13, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x28]),
        ("mov r15, qword ptr [rax + r13*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, R13, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x28, 0x08]),
        ("mov r15, qword ptr [rax + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R13, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x28, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r13*8]", |a| { a.ld(R15, mem_idx(RAX, R13, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xe8]),
        ("mov r15, qword ptr [rax + r13*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, R13, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xe8, 0x08]),
        ("mov r15, qword ptr [rax + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R13, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r15*1]", |a| { a.ld(R15, mem_idx(RAX, R15, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x38]),
        ("mov r15, qword ptr [rax + r15*1 + 8]", |a| { a.ld(R15, mem_idx(RAX, R15, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x38, 0x08]),
        ("mov r15, qword ptr [rax + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R15, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x38, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rax + r15*8]", |a| { a.ld(R15, mem_idx(RAX, R15, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xf8]),
        ("mov r15, qword ptr [rax + r15*8 + 8]", |a| { a.ld(R15, mem_idx(RAX, R15, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xf8, 0x08]),
        ("mov r15, qword ptr [rax + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(RAX, R15, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rax*1]", |a| { a.ld(R15, mem_idx(RSP, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x04]),
        ("mov r15, qword ptr [rsp + rax*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x04, 0x08]),
        ("mov r15, qword ptr [rsp + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rax*8]", |a| { a.ld(R15, mem_idx(RSP, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xc4]),
        ("mov r15, qword ptr [rsp + rax*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xc4, 0x08]),
        ("mov r15, qword ptr [rsp + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rcx*1]", |a| { a.ld(R15, mem_idx(RSP, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x0c]),
        ("mov r15, qword ptr [rsp + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x0c, 0x08]),
        ("mov r15, qword ptr [rsp + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rcx*8]", |a| { a.ld(R15, mem_idx(RSP, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xcc]),
        ("mov r15, qword ptr [rsp + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xcc, 0x08]),
        ("mov r15, qword ptr [rsp + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rbp*1]", |a| { a.ld(R15, mem_idx(RSP, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x2c]),
        ("mov r15, qword ptr [rsp + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x2c, 0x08]),
        ("mov r15, qword ptr [rsp + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rbp*8]", |a| { a.ld(R15, mem_idx(RSP, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xec]),
        ("mov r15, qword ptr [rsp + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xec, 0x08]),
        ("mov r15, qword ptr [rsp + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rsi*1]", |a| { a.ld(R15, mem_idx(RSP, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x3c, 0x34]),
        ("mov r15, qword ptr [rsp + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x34, 0x08]),
        ("mov r15, qword ptr [rsp + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + rsi*8]", |a| { a.ld(R15, mem_idx(RSP, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x3c, 0xf4]),
        ("mov r15, qword ptr [rsp + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xf4, 0x08]),
        ("mov r15, qword ptr [rsp + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r8*1]", |a| { a.ld(R15, mem_idx(RSP, R8, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x04]),
        ("mov r15, qword ptr [rsp + r8*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, R8, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x04, 0x08]),
        ("mov r15, qword ptr [rsp + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R8, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r8*8]", |a| { a.ld(R15, mem_idx(RSP, R8, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xc4]),
        ("mov r15, qword ptr [rsp + r8*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, R8, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xc4, 0x08]),
        ("mov r15, qword ptr [rsp + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R8, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r12*1]", |a| { a.ld(R15, mem_idx(RSP, R12, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x24]),
        ("mov r15, qword ptr [rsp + r12*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, R12, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x24, 0x08]),
        ("mov r15, qword ptr [rsp + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R12, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r12*8]", |a| { a.ld(R15, mem_idx(RSP, R12, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xe4]),
        ("mov r15, qword ptr [rsp + r12*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, R12, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xe4, 0x08]),
        ("mov r15, qword ptr [rsp + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R12, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r13*1]", |a| { a.ld(R15, mem_idx(RSP, R13, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x2c]),
        ("mov r15, qword ptr [rsp + r13*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, R13, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x2c, 0x08]),
        ("mov r15, qword ptr [rsp + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R13, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r13*8]", |a| { a.ld(R15, mem_idx(RSP, R13, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xec]),
        ("mov r15, qword ptr [rsp + r13*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, R13, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xec, 0x08]),
        ("mov r15, qword ptr [rsp + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R13, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r15*1]", |a| { a.ld(R15, mem_idx(RSP, R15, 0, 0)); }, &[0x4e, 0x8b, 0x3c, 0x3c]),
        ("mov r15, qword ptr [rsp + r15*1 + 8]", |a| { a.ld(R15, mem_idx(RSP, R15, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x3c, 0x08]),
        ("mov r15, qword ptr [rsp + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R15, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rsp + r15*8]", |a| { a.ld(R15, mem_idx(RSP, R15, 3, 0)); }, &[0x4e, 0x8b, 0x3c, 0xfc]),
        ("mov r15, qword ptr [rsp + r15*8 + 8]", |a| { a.ld(R15, mem_idx(RSP, R15, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xfc, 0x08]),
        ("mov r15, qword ptr [rsp + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(RSP, R15, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rax*1]", |a| { a.ld(R15, mem_idx(RBP, RAX, 0, 0)); }, &[0x4c, 0x8b, 0x7c, 0x05, 0x00]),
        ("mov r15, qword ptr [rbp + rax*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, RAX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x05, 0x08]),
        ("mov r15, qword ptr [rbp + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RAX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rax*8]", |a| { a.ld(R15, mem_idx(RBP, RAX, 3, 0)); }, &[0x4c, 0x8b, 0x7c, 0xc5, 0x00]),
        ("mov r15, qword ptr [rbp + rax*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, RAX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xc5, 0x08]),
        ("mov r15, qword ptr [rbp + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RAX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rcx*1]", |a| { a.ld(R15, mem_idx(RBP, RCX, 0, 0)); }, &[0x4c, 0x8b, 0x7c, 0x0d, 0x00]),
        ("mov r15, qword ptr [rbp + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, RCX, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x0d, 0x08]),
        ("mov r15, qword ptr [rbp + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RCX, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rcx*8]", |a| { a.ld(R15, mem_idx(RBP, RCX, 3, 0)); }, &[0x4c, 0x8b, 0x7c, 0xcd, 0x00]),
        ("mov r15, qword ptr [rbp + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, RCX, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xcd, 0x08]),
        ("mov r15, qword ptr [rbp + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RCX, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rbp*1]", |a| { a.ld(R15, mem_idx(RBP, RBP, 0, 0)); }, &[0x4c, 0x8b, 0x7c, 0x2d, 0x00]),
        ("mov r15, qword ptr [rbp + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, RBP, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x2d, 0x08]),
        ("mov r15, qword ptr [rbp + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RBP, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rbp*8]", |a| { a.ld(R15, mem_idx(RBP, RBP, 3, 0)); }, &[0x4c, 0x8b, 0x7c, 0xed, 0x00]),
        ("mov r15, qword ptr [rbp + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, RBP, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xed, 0x08]),
        ("mov r15, qword ptr [rbp + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RBP, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rsi*1]", |a| { a.ld(R15, mem_idx(RBP, RSI, 0, 0)); }, &[0x4c, 0x8b, 0x7c, 0x35, 0x00]),
        ("mov r15, qword ptr [rbp + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, RSI, 0, 8)); }, &[0x4c, 0x8b, 0x7c, 0x35, 0x08]),
        ("mov r15, qword ptr [rbp + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RSI, 0, 4096)); }, &[0x4c, 0x8b, 0xbc, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + rsi*8]", |a| { a.ld(R15, mem_idx(RBP, RSI, 3, 0)); }, &[0x4c, 0x8b, 0x7c, 0xf5, 0x00]),
        ("mov r15, qword ptr [rbp + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, RSI, 3, 8)); }, &[0x4c, 0x8b, 0x7c, 0xf5, 0x08]),
        ("mov r15, qword ptr [rbp + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, RSI, 3, 4096)); }, &[0x4c, 0x8b, 0xbc, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r8*1]", |a| { a.ld(R15, mem_idx(RBP, R8, 0, 0)); }, &[0x4e, 0x8b, 0x7c, 0x05, 0x00]),
        ("mov r15, qword ptr [rbp + r8*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, R8, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x05, 0x08]),
        ("mov r15, qword ptr [rbp + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R8, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r8*8]", |a| { a.ld(R15, mem_idx(RBP, R8, 3, 0)); }, &[0x4e, 0x8b, 0x7c, 0xc5, 0x00]),
        ("mov r15, qword ptr [rbp + r8*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, R8, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xc5, 0x08]),
        ("mov r15, qword ptr [rbp + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R8, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r12*1]", |a| { a.ld(R15, mem_idx(RBP, R12, 0, 0)); }, &[0x4e, 0x8b, 0x7c, 0x25, 0x00]),
        ("mov r15, qword ptr [rbp + r12*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, R12, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x25, 0x08]),
        ("mov r15, qword ptr [rbp + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R12, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r12*8]", |a| { a.ld(R15, mem_idx(RBP, R12, 3, 0)); }, &[0x4e, 0x8b, 0x7c, 0xe5, 0x00]),
        ("mov r15, qword ptr [rbp + r12*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, R12, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xe5, 0x08]),
        ("mov r15, qword ptr [rbp + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R12, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r13*1]", |a| { a.ld(R15, mem_idx(RBP, R13, 0, 0)); }, &[0x4e, 0x8b, 0x7c, 0x2d, 0x00]),
        ("mov r15, qword ptr [rbp + r13*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, R13, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x2d, 0x08]),
        ("mov r15, qword ptr [rbp + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R13, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r13*8]", |a| { a.ld(R15, mem_idx(RBP, R13, 3, 0)); }, &[0x4e, 0x8b, 0x7c, 0xed, 0x00]),
        ("mov r15, qword ptr [rbp + r13*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, R13, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xed, 0x08]),
        ("mov r15, qword ptr [rbp + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R13, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r15*1]", |a| { a.ld(R15, mem_idx(RBP, R15, 0, 0)); }, &[0x4e, 0x8b, 0x7c, 0x3d, 0x00]),
        ("mov r15, qword ptr [rbp + r15*1 + 8]", |a| { a.ld(R15, mem_idx(RBP, R15, 0, 8)); }, &[0x4e, 0x8b, 0x7c, 0x3d, 0x08]),
        ("mov r15, qword ptr [rbp + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R15, 0, 4096)); }, &[0x4e, 0x8b, 0xbc, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbp + r15*8]", |a| { a.ld(R15, mem_idx(RBP, R15, 3, 0)); }, &[0x4e, 0x8b, 0x7c, 0xfd, 0x00]),
        ("mov r15, qword ptr [rbp + r15*8 + 8]", |a| { a.ld(R15, mem_idx(RBP, R15, 3, 8)); }, &[0x4e, 0x8b, 0x7c, 0xfd, 0x08]),
        ("mov r15, qword ptr [rbp + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(RBP, R15, 3, 4096)); }, &[0x4e, 0x8b, 0xbc, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rax*1]", |a| { a.ld(R15, mem_idx(R12, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x04]),
        ("mov r15, qword ptr [r12 + rax*1 + 8]", |a| { a.ld(R15, mem_idx(R12, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x04, 0x08]),
        ("mov r15, qword ptr [r12 + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rax*8]", |a| { a.ld(R15, mem_idx(R12, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xc4]),
        ("mov r15, qword ptr [r12 + rax*8 + 8]", |a| { a.ld(R15, mem_idx(R12, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xc4, 0x08]),
        ("mov r15, qword ptr [r12 + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rcx*1]", |a| { a.ld(R15, mem_idx(R12, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x0c]),
        ("mov r15, qword ptr [r12 + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(R12, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x0c, 0x08]),
        ("mov r15, qword ptr [r12 + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x0c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rcx*8]", |a| { a.ld(R15, mem_idx(R12, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xcc]),
        ("mov r15, qword ptr [r12 + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(R12, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xcc, 0x08]),
        ("mov r15, qword ptr [r12 + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rbp*1]", |a| { a.ld(R15, mem_idx(R12, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x2c]),
        ("mov r15, qword ptr [r12 + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(R12, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x2c, 0x08]),
        ("mov r15, qword ptr [r12 + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rbp*8]", |a| { a.ld(R15, mem_idx(R12, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xec]),
        ("mov r15, qword ptr [r12 + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(R12, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xec, 0x08]),
        ("mov r15, qword ptr [r12 + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rsi*1]", |a| { a.ld(R15, mem_idx(R12, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x34]),
        ("mov r15, qword ptr [r12 + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(R12, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x34, 0x08]),
        ("mov r15, qword ptr [r12 + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x34, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + rsi*8]", |a| { a.ld(R15, mem_idx(R12, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xf4]),
        ("mov r15, qword ptr [r12 + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(R12, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xf4, 0x08]),
        ("mov r15, qword ptr [r12 + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r8*1]", |a| { a.ld(R15, mem_idx(R12, R8, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x04]),
        ("mov r15, qword ptr [r12 + r8*1 + 8]", |a| { a.ld(R15, mem_idx(R12, R8, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x04, 0x08]),
        ("mov r15, qword ptr [r12 + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, R8, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x04, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r8*8]", |a| { a.ld(R15, mem_idx(R12, R8, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xc4]),
        ("mov r15, qword ptr [r12 + r8*8 + 8]", |a| { a.ld(R15, mem_idx(R12, R8, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xc4, 0x08]),
        ("mov r15, qword ptr [r12 + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, R8, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r12*1]", |a| { a.ld(R15, mem_idx(R12, R12, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x24]),
        ("mov r15, qword ptr [r12 + r12*1 + 8]", |a| { a.ld(R15, mem_idx(R12, R12, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x24, 0x08]),
        ("mov r15, qword ptr [r12 + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, R12, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x24, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r12*8]", |a| { a.ld(R15, mem_idx(R12, R12, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xe4]),
        ("mov r15, qword ptr [r12 + r12*8 + 8]", |a| { a.ld(R15, mem_idx(R12, R12, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xe4, 0x08]),
        ("mov r15, qword ptr [r12 + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, R12, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r13*1]", |a| { a.ld(R15, mem_idx(R12, R13, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x2c]),
        ("mov r15, qword ptr [r12 + r13*1 + 8]", |a| { a.ld(R15, mem_idx(R12, R13, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x2c, 0x08]),
        ("mov r15, qword ptr [r12 + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, R13, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x2c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r13*8]", |a| { a.ld(R15, mem_idx(R12, R13, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xec]),
        ("mov r15, qword ptr [r12 + r13*8 + 8]", |a| { a.ld(R15, mem_idx(R12, R13, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xec, 0x08]),
        ("mov r15, qword ptr [r12 + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, R13, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r15*1]", |a| { a.ld(R15, mem_idx(R12, R15, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x3c]),
        ("mov r15, qword ptr [r12 + r15*1 + 8]", |a| { a.ld(R15, mem_idx(R12, R15, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x3c, 0x08]),
        ("mov r15, qword ptr [r12 + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(R12, R15, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x3c, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r12 + r15*8]", |a| { a.ld(R15, mem_idx(R12, R15, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xfc]),
        ("mov r15, qword ptr [r12 + r15*8 + 8]", |a| { a.ld(R15, mem_idx(R12, R15, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xfc, 0x08]),
        ("mov r15, qword ptr [r12 + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(R12, R15, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rax*1]", |a| { a.ld(R15, mem_idx(R13, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x7c, 0x05, 0x00]),
        ("mov r15, qword ptr [r13 + rax*1 + 8]", |a| { a.ld(R15, mem_idx(R13, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x05, 0x08]),
        ("mov r15, qword ptr [r13 + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rax*8]", |a| { a.ld(R15, mem_idx(R13, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x7c, 0xc5, 0x00]),
        ("mov r15, qword ptr [r13 + rax*8 + 8]", |a| { a.ld(R15, mem_idx(R13, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xc5, 0x08]),
        ("mov r15, qword ptr [r13 + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rcx*1]", |a| { a.ld(R15, mem_idx(R13, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x7c, 0x0d, 0x00]),
        ("mov r15, qword ptr [r13 + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(R13, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x0d, 0x08]),
        ("mov r15, qword ptr [r13 + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x0d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rcx*8]", |a| { a.ld(R15, mem_idx(R13, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x7c, 0xcd, 0x00]),
        ("mov r15, qword ptr [r13 + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(R13, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xcd, 0x08]),
        ("mov r15, qword ptr [r13 + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rbp*1]", |a| { a.ld(R15, mem_idx(R13, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x7c, 0x2d, 0x00]),
        ("mov r15, qword ptr [r13 + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(R13, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x2d, 0x08]),
        ("mov r15, qword ptr [r13 + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rbp*8]", |a| { a.ld(R15, mem_idx(R13, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x7c, 0xed, 0x00]),
        ("mov r15, qword ptr [r13 + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(R13, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xed, 0x08]),
        ("mov r15, qword ptr [r13 + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rsi*1]", |a| { a.ld(R15, mem_idx(R13, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x7c, 0x35, 0x00]),
        ("mov r15, qword ptr [r13 + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(R13, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x35, 0x08]),
        ("mov r15, qword ptr [r13 + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x35, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + rsi*8]", |a| { a.ld(R15, mem_idx(R13, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x7c, 0xf5, 0x00]),
        ("mov r15, qword ptr [r13 + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(R13, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xf5, 0x08]),
        ("mov r15, qword ptr [r13 + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r8*1]", |a| { a.ld(R15, mem_idx(R13, R8, 0, 0)); }, &[0x4f, 0x8b, 0x7c, 0x05, 0x00]),
        ("mov r15, qword ptr [r13 + r8*1 + 8]", |a| { a.ld(R15, mem_idx(R13, R8, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x05, 0x08]),
        ("mov r15, qword ptr [r13 + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, R8, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x05, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r8*8]", |a| { a.ld(R15, mem_idx(R13, R8, 3, 0)); }, &[0x4f, 0x8b, 0x7c, 0xc5, 0x00]),
        ("mov r15, qword ptr [r13 + r8*8 + 8]", |a| { a.ld(R15, mem_idx(R13, R8, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xc5, 0x08]),
        ("mov r15, qword ptr [r13 + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, R8, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r12*1]", |a| { a.ld(R15, mem_idx(R13, R12, 0, 0)); }, &[0x4f, 0x8b, 0x7c, 0x25, 0x00]),
        ("mov r15, qword ptr [r13 + r12*1 + 8]", |a| { a.ld(R15, mem_idx(R13, R12, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x25, 0x08]),
        ("mov r15, qword ptr [r13 + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, R12, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x25, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r12*8]", |a| { a.ld(R15, mem_idx(R13, R12, 3, 0)); }, &[0x4f, 0x8b, 0x7c, 0xe5, 0x00]),
        ("mov r15, qword ptr [r13 + r12*8 + 8]", |a| { a.ld(R15, mem_idx(R13, R12, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xe5, 0x08]),
        ("mov r15, qword ptr [r13 + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, R12, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r13*1]", |a| { a.ld(R15, mem_idx(R13, R13, 0, 0)); }, &[0x4f, 0x8b, 0x7c, 0x2d, 0x00]),
        ("mov r15, qword ptr [r13 + r13*1 + 8]", |a| { a.ld(R15, mem_idx(R13, R13, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x2d, 0x08]),
        ("mov r15, qword ptr [r13 + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, R13, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x2d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r13*8]", |a| { a.ld(R15, mem_idx(R13, R13, 3, 0)); }, &[0x4f, 0x8b, 0x7c, 0xed, 0x00]),
        ("mov r15, qword ptr [r13 + r13*8 + 8]", |a| { a.ld(R15, mem_idx(R13, R13, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xed, 0x08]),
        ("mov r15, qword ptr [r13 + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, R13, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r15*1]", |a| { a.ld(R15, mem_idx(R13, R15, 0, 0)); }, &[0x4f, 0x8b, 0x7c, 0x3d, 0x00]),
        ("mov r15, qword ptr [r13 + r15*1 + 8]", |a| { a.ld(R15, mem_idx(R13, R15, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x3d, 0x08]),
        ("mov r15, qword ptr [r13 + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(R13, R15, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x3d, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r13 + r15*8]", |a| { a.ld(R15, mem_idx(R13, R15, 3, 0)); }, &[0x4f, 0x8b, 0x7c, 0xfd, 0x00]),
        ("mov r15, qword ptr [r13 + r15*8 + 8]", |a| { a.ld(R15, mem_idx(R13, R15, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xfd, 0x08]),
        ("mov r15, qword ptr [r13 + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(R13, R15, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rax*1]", |a| { a.ld(R15, mem_idx(R15, RAX, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x07]),
        ("mov r15, qword ptr [r15 + rax*1 + 8]", |a| { a.ld(R15, mem_idx(R15, RAX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x07, 0x08]),
        ("mov r15, qword ptr [r15 + rax*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, RAX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rax*8]", |a| { a.ld(R15, mem_idx(R15, RAX, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xc7]),
        ("mov r15, qword ptr [r15 + rax*8 + 8]", |a| { a.ld(R15, mem_idx(R15, RAX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xc7, 0x08]),
        ("mov r15, qword ptr [r15 + rax*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, RAX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rcx*1]", |a| { a.ld(R15, mem_idx(R15, RCX, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x0f]),
        ("mov r15, qword ptr [r15 + rcx*1 + 8]", |a| { a.ld(R15, mem_idx(R15, RCX, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x0f, 0x08]),
        ("mov r15, qword ptr [r15 + rcx*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, RCX, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x0f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rcx*8]", |a| { a.ld(R15, mem_idx(R15, RCX, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xcf]),
        ("mov r15, qword ptr [r15 + rcx*8 + 8]", |a| { a.ld(R15, mem_idx(R15, RCX, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xcf, 0x08]),
        ("mov r15, qword ptr [r15 + rcx*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, RCX, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rbp*1]", |a| { a.ld(R15, mem_idx(R15, RBP, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x2f]),
        ("mov r15, qword ptr [r15 + rbp*1 + 8]", |a| { a.ld(R15, mem_idx(R15, RBP, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x2f, 0x08]),
        ("mov r15, qword ptr [r15 + rbp*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, RBP, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rbp*8]", |a| { a.ld(R15, mem_idx(R15, RBP, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xef]),
        ("mov r15, qword ptr [r15 + rbp*8 + 8]", |a| { a.ld(R15, mem_idx(R15, RBP, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xef, 0x08]),
        ("mov r15, qword ptr [r15 + rbp*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, RBP, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rsi*1]", |a| { a.ld(R15, mem_idx(R15, RSI, 0, 0)); }, &[0x4d, 0x8b, 0x3c, 0x37]),
        ("mov r15, qword ptr [r15 + rsi*1 + 8]", |a| { a.ld(R15, mem_idx(R15, RSI, 0, 8)); }, &[0x4d, 0x8b, 0x7c, 0x37, 0x08]),
        ("mov r15, qword ptr [r15 + rsi*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, RSI, 0, 4096)); }, &[0x4d, 0x8b, 0xbc, 0x37, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + rsi*8]", |a| { a.ld(R15, mem_idx(R15, RSI, 3, 0)); }, &[0x4d, 0x8b, 0x3c, 0xf7]),
        ("mov r15, qword ptr [r15 + rsi*8 + 8]", |a| { a.ld(R15, mem_idx(R15, RSI, 3, 8)); }, &[0x4d, 0x8b, 0x7c, 0xf7, 0x08]),
        ("mov r15, qword ptr [r15 + rsi*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, RSI, 3, 4096)); }, &[0x4d, 0x8b, 0xbc, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r8*1]", |a| { a.ld(R15, mem_idx(R15, R8, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x07]),
        ("mov r15, qword ptr [r15 + r8*1 + 8]", |a| { a.ld(R15, mem_idx(R15, R8, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x07, 0x08]),
        ("mov r15, qword ptr [r15 + r8*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, R8, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x07, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r8*8]", |a| { a.ld(R15, mem_idx(R15, R8, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xc7]),
        ("mov r15, qword ptr [r15 + r8*8 + 8]", |a| { a.ld(R15, mem_idx(R15, R8, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xc7, 0x08]),
        ("mov r15, qword ptr [r15 + r8*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, R8, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r12*1]", |a| { a.ld(R15, mem_idx(R15, R12, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x27]),
        ("mov r15, qword ptr [r15 + r12*1 + 8]", |a| { a.ld(R15, mem_idx(R15, R12, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x27, 0x08]),
        ("mov r15, qword ptr [r15 + r12*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, R12, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x27, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r12*8]", |a| { a.ld(R15, mem_idx(R15, R12, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xe7]),
        ("mov r15, qword ptr [r15 + r12*8 + 8]", |a| { a.ld(R15, mem_idx(R15, R12, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xe7, 0x08]),
        ("mov r15, qword ptr [r15 + r12*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, R12, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r13*1]", |a| { a.ld(R15, mem_idx(R15, R13, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x2f]),
        ("mov r15, qword ptr [r15 + r13*1 + 8]", |a| { a.ld(R15, mem_idx(R15, R13, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x2f, 0x08]),
        ("mov r15, qword ptr [r15 + r13*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, R13, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x2f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r13*8]", |a| { a.ld(R15, mem_idx(R15, R13, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xef]),
        ("mov r15, qword ptr [r15 + r13*8 + 8]", |a| { a.ld(R15, mem_idx(R15, R13, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xef, 0x08]),
        ("mov r15, qword ptr [r15 + r13*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, R13, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r15*1]", |a| { a.ld(R15, mem_idx(R15, R15, 0, 0)); }, &[0x4f, 0x8b, 0x3c, 0x3f]),
        ("mov r15, qword ptr [r15 + r15*1 + 8]", |a| { a.ld(R15, mem_idx(R15, R15, 0, 8)); }, &[0x4f, 0x8b, 0x7c, 0x3f, 0x08]),
        ("mov r15, qword ptr [r15 + r15*1 + 4096]", |a| { a.ld(R15, mem_idx(R15, R15, 0, 4096)); }, &[0x4f, 0x8b, 0xbc, 0x3f, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [r15 + r15*8]", |a| { a.ld(R15, mem_idx(R15, R15, 3, 0)); }, &[0x4f, 0x8b, 0x3c, 0xff]),
        ("mov r15, qword ptr [r15 + r15*8 + 8]", |a| { a.ld(R15, mem_idx(R15, R15, 3, 8)); }, &[0x4f, 0x8b, 0x7c, 0xff, 0x08]),
        ("mov r15, qword ptr [r15 + r15*8 + 4096]", |a| { a.ld(R15, mem_idx(R15, R15, 3, 4096)); }, &[0x4f, 0x8b, 0xbc, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("mov r15, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld(R15, mem_idx(RBX, RCX, 1, 32)); }, &[0x4c, 0x8b, 0x7c, 0x4b, 0x20]),
        ("mov r15, qword ptr [rbx + r12*2 + 32]", |a| { a.ld(R15, mem_idx(RBX, R12, 1, 32)); }, &[0x4e, 0x8b, 0x7c, 0x63, 0x20]),
        ("mov r15, qword ptr [rbp + rcx*2 + 32]", |a| { a.ld(R15, mem_idx(RBP, RCX, 1, 32)); }, &[0x4c, 0x8b, 0x7c, 0x4d, 0x20]),
        ("mov r15, qword ptr [rbp + r12*2 + 32]", |a| { a.ld(R15, mem_idx(RBP, R12, 1, 32)); }, &[0x4e, 0x8b, 0x7c, 0x65, 0x20]),
        ("mov r15, qword ptr [rbx + rcx*4 + 32]", |a| { a.ld(R15, mem_idx(RBX, RCX, 2, 32)); }, &[0x4c, 0x8b, 0x7c, 0x8b, 0x20]),
        ("mov r15, qword ptr [rbx + r12*4 + 32]", |a| { a.ld(R15, mem_idx(RBX, R12, 2, 32)); }, &[0x4e, 0x8b, 0x7c, 0xa3, 0x20]),
        ("mov r15, qword ptr [rbp + rcx*4 + 32]", |a| { a.ld(R15, mem_idx(RBP, RCX, 2, 32)); }, &[0x4c, 0x8b, 0x7c, 0x8d, 0x20]),
        ("mov r15, qword ptr [rbp + r12*4 + 32]", |a| { a.ld(R15, mem_idx(RBP, R12, 2, 32)); }, &[0x4e, 0x8b, 0x7c, 0xa5, 0x20]),
    ];
    check_encodings(cases);
}

    /// Movzx/movsx/mov/movsxd from memory in every size.
#[rustfmt::skip]
#[test]
fn extending_loads_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("movzx eax, byte ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 1, false); }, &[0x0f, 0xb6, 0x00]),
        ("movzx eax, byte ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 1, false); }, &[0x0f, 0xb6, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("movzx eax, byte ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 1, false); }, &[0x0f, 0xb6, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx eax, byte ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 1, false); }, &[0x0f, 0xb6, 0x44, 0x24, 0x7f]),
        ("movzx eax, byte ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 1, false); }, &[0x0f, 0xb6, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movzx eax, byte ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 1, false); }, &[0x0f, 0xb6, 0x45, 0xff]),
        ("movzx eax, byte ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 1, false); }, &[0x0f, 0xb6, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx eax, byte ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 1, false); }, &[0x41, 0x0f, 0xb6, 0x40, 0x01]),
        ("movzx eax, byte ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 1, false); }, &[0x41, 0x0f, 0xb6, 0x40, 0x80]),
        ("movzx eax, byte ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0x24]),
        ("movzx eax, byte ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 1, false); }, &[0x41, 0x0f, 0xb6, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movzx eax, byte ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 1, false); }, &[0x41, 0x0f, 0xb6, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx eax, byte ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 1, false); }, &[0x41, 0x0f, 0xb6, 0x45, 0x7f]),
        ("movzx eax, byte ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 1, false); }, &[0x41, 0x0f, 0xb6, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("movzx eax, byte ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 1, false); }, &[0x41, 0x0f, 0xb6, 0x47, 0xff]),
        ("movzx eax, byte ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 1, false); }, &[0x41, 0x0f, 0xb6, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx eax, byte ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 1, false); }, &[0x0f, 0xb6, 0x43, 0x01]),
        ("movzx eax, byte ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 1, false); }, &[0x0f, 0xb6, 0x43, 0x80]),
        ("movzx eax, byte ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 1, false); }, &[0x0f, 0xb6, 0x04, 0x00]),
        ("movzx eax, byte ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 1, false); }, &[0x0f, 0xb6, 0x04, 0xc8]),
        ("movzx eax, byte ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 1, false); }, &[0x0f, 0xb6, 0x04, 0x30]),
        ("movzx eax, byte ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0xc0]),
        ("movzx eax, byte ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0x28]),
        ("movzx eax, byte ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0xf8]),
        ("movzx eax, byte ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 1, false); }, &[0x0f, 0xb6, 0x04, 0x0c]),
        ("movzx eax, byte ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 1, false); }, &[0x0f, 0xb6, 0x04, 0xec]),
        ("movzx eax, byte ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0x04]),
        ("movzx eax, byte ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0xe4]),
        ("movzx eax, byte ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x04, 0x3c]),
        ("movzx eax, byte ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 1, false); }, &[0x0f, 0xb6, 0x44, 0xc5, 0x00]),
        ("movzx eax, byte ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 1, false); }, &[0x0f, 0xb6, 0x44, 0x2d, 0x00]),
        ("movzx eax, byte ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 1, false); }, &[0x0f, 0xb6, 0x44, 0xf5, 0x00]),
        ("movzx eax, byte ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x44, 0x25, 0x00]),
        ("movzx eax, byte ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 1, false); }, &[0x42, 0x0f, 0xb6, 0x44, 0xed, 0x00]),
        ("movzx eax, byte ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0x04]),
        ("movzx eax, byte ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0xcc]),
        ("movzx eax, byte ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0x34]),
        ("movzx eax, byte ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x04, 0xc4]),
        ("movzx eax, byte ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x04, 0x2c]),
        ("movzx eax, byte ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x04, 0xfc]),
        ("movzx eax, byte ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x44, 0x0d, 0x00]),
        ("movzx eax, byte ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x44, 0xed, 0x00]),
        ("movzx eax, byte ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x44, 0x05, 0x00]),
        ("movzx eax, byte ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x44, 0xe5, 0x00]),
        ("movzx eax, byte ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x44, 0x3d, 0x00]),
        ("movzx eax, byte ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0xc7]),
        ("movzx eax, byte ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0x2f]),
        ("movzx eax, byte ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 1, false); }, &[0x41, 0x0f, 0xb6, 0x04, 0xf7]),
        ("movzx eax, byte ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x04, 0x27]),
        ("movzx eax, byte ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 1, false); }, &[0x43, 0x0f, 0xb6, 0x04, 0xef]),
        ("movzx eax, byte ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 1, false); }, &[0x0f, 0xb6, 0x44, 0x4b, 0x20]),
        ("movzx r9d, byte ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x08]),
        ("movzx r9d, byte ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 1, false); }, &[0x44, 0x0f, 0xb6, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("movzx r9d, byte ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 1, false); }, &[0x44, 0x0f, 0xb6, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx r9d, byte ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4c, 0x24, 0x7f]),
        ("movzx r9d, byte ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 1, false); }, &[0x44, 0x0f, 0xb6, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movzx r9d, byte ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4d, 0xff]),
        ("movzx r9d, byte ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 1, false); }, &[0x44, 0x0f, 0xb6, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx r9d, byte ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 1, false); }, &[0x45, 0x0f, 0xb6, 0x48, 0x01]),
        ("movzx r9d, byte ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 1, false); }, &[0x45, 0x0f, 0xb6, 0x48, 0x80]),
        ("movzx r9d, byte ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0x24]),
        ("movzx r9d, byte ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 1, false); }, &[0x45, 0x0f, 0xb6, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movzx r9d, byte ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 1, false); }, &[0x45, 0x0f, 0xb6, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx r9d, byte ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 1, false); }, &[0x45, 0x0f, 0xb6, 0x4d, 0x7f]),
        ("movzx r9d, byte ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 1, false); }, &[0x45, 0x0f, 0xb6, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("movzx r9d, byte ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 1, false); }, &[0x45, 0x0f, 0xb6, 0x4f, 0xff]),
        ("movzx r9d, byte ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 1, false); }, &[0x45, 0x0f, 0xb6, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx r9d, byte ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4b, 0x01]),
        ("movzx r9d, byte ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4b, 0x80]),
        ("movzx r9d, byte ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x0c, 0x00]),
        ("movzx r9d, byte ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x0c, 0xc8]),
        ("movzx r9d, byte ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x0c, 0x30]),
        ("movzx r9d, byte ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0xc0]),
        ("movzx r9d, byte ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0x28]),
        ("movzx r9d, byte ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0xf8]),
        ("movzx r9d, byte ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x0c, 0x0c]),
        ("movzx r9d, byte ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x0c, 0xec]),
        ("movzx r9d, byte ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0x04]),
        ("movzx r9d, byte ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0xe4]),
        ("movzx r9d, byte ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x0c, 0x3c]),
        ("movzx r9d, byte ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4c, 0xc5, 0x00]),
        ("movzx r9d, byte ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4c, 0x2d, 0x00]),
        ("movzx r9d, byte ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4c, 0xf5, 0x00]),
        ("movzx r9d, byte ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x4c, 0x25, 0x00]),
        ("movzx r9d, byte ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 1, false); }, &[0x46, 0x0f, 0xb6, 0x4c, 0xed, 0x00]),
        ("movzx r9d, byte ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0x04]),
        ("movzx r9d, byte ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0xcc]),
        ("movzx r9d, byte ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0x34]),
        ("movzx r9d, byte ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x0c, 0xc4]),
        ("movzx r9d, byte ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x0c, 0x2c]),
        ("movzx r9d, byte ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x0c, 0xfc]),
        ("movzx r9d, byte ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x4c, 0x0d, 0x00]),
        ("movzx r9d, byte ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x4c, 0xed, 0x00]),
        ("movzx r9d, byte ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x4c, 0x05, 0x00]),
        ("movzx r9d, byte ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x4c, 0xe5, 0x00]),
        ("movzx r9d, byte ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x4c, 0x3d, 0x00]),
        ("movzx r9d, byte ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0xc7]),
        ("movzx r9d, byte ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0x2f]),
        ("movzx r9d, byte ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 1, false); }, &[0x45, 0x0f, 0xb6, 0x0c, 0xf7]),
        ("movzx r9d, byte ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x0c, 0x27]),
        ("movzx r9d, byte ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 1, false); }, &[0x47, 0x0f, 0xb6, 0x0c, 0xef]),
        ("movzx r9d, byte ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 1, false); }, &[0x44, 0x0f, 0xb6, 0x4c, 0x4b, 0x20]),
        ("movsx rax, byte ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x00]),
        ("movsx rax, byte ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 1, true); }, &[0x48, 0x0f, 0xbe, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("movsx rax, byte ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 1, true); }, &[0x48, 0x0f, 0xbe, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx rax, byte ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 1, true); }, &[0x48, 0x0f, 0xbe, 0x44, 0x24, 0x7f]),
        ("movsx rax, byte ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 1, true); }, &[0x48, 0x0f, 0xbe, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsx rax, byte ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 1, true); }, &[0x48, 0x0f, 0xbe, 0x45, 0xff]),
        ("movsx rax, byte ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 1, true); }, &[0x48, 0x0f, 0xbe, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx rax, byte ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 1, true); }, &[0x49, 0x0f, 0xbe, 0x40, 0x01]),
        ("movsx rax, byte ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 1, true); }, &[0x49, 0x0f, 0xbe, 0x40, 0x80]),
        ("movsx rax, byte ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0x24]),
        ("movsx rax, byte ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 1, true); }, &[0x49, 0x0f, 0xbe, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsx rax, byte ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 1, true); }, &[0x49, 0x0f, 0xbe, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx rax, byte ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 1, true); }, &[0x49, 0x0f, 0xbe, 0x45, 0x7f]),
        ("movsx rax, byte ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 1, true); }, &[0x49, 0x0f, 0xbe, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("movsx rax, byte ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 1, true); }, &[0x49, 0x0f, 0xbe, 0x47, 0xff]),
        ("movsx rax, byte ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 1, true); }, &[0x49, 0x0f, 0xbe, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx rax, byte ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 1, true); }, &[0x48, 0x0f, 0xbe, 0x43, 0x01]),
        ("movsx rax, byte ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 1, true); }, &[0x48, 0x0f, 0xbe, 0x43, 0x80]),
        ("movsx rax, byte ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x04, 0x00]),
        ("movsx rax, byte ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x04, 0xc8]),
        ("movsx rax, byte ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x04, 0x30]),
        ("movsx rax, byte ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0xc0]),
        ("movsx rax, byte ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0x28]),
        ("movsx rax, byte ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0xf8]),
        ("movsx rax, byte ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x04, 0x0c]),
        ("movsx rax, byte ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x04, 0xec]),
        ("movsx rax, byte ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0x04]),
        ("movsx rax, byte ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0xe4]),
        ("movsx rax, byte ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x04, 0x3c]),
        ("movsx rax, byte ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x44, 0xc5, 0x00]),
        ("movsx rax, byte ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x44, 0x2d, 0x00]),
        ("movsx rax, byte ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 1, true); }, &[0x48, 0x0f, 0xbe, 0x44, 0xf5, 0x00]),
        ("movsx rax, byte ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x44, 0x25, 0x00]),
        ("movsx rax, byte ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 1, true); }, &[0x4a, 0x0f, 0xbe, 0x44, 0xed, 0x00]),
        ("movsx rax, byte ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0x04]),
        ("movsx rax, byte ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0xcc]),
        ("movsx rax, byte ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0x34]),
        ("movsx rax, byte ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x04, 0xc4]),
        ("movsx rax, byte ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x04, 0x2c]),
        ("movsx rax, byte ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x04, 0xfc]),
        ("movsx rax, byte ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x44, 0x0d, 0x00]),
        ("movsx rax, byte ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x44, 0xed, 0x00]),
        ("movsx rax, byte ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x44, 0x05, 0x00]),
        ("movsx rax, byte ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x44, 0xe5, 0x00]),
        ("movsx rax, byte ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x44, 0x3d, 0x00]),
        ("movsx rax, byte ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0xc7]),
        ("movsx rax, byte ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0x2f]),
        ("movsx rax, byte ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 1, true); }, &[0x49, 0x0f, 0xbe, 0x04, 0xf7]),
        ("movsx rax, byte ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x04, 0x27]),
        ("movsx rax, byte ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 1, true); }, &[0x4b, 0x0f, 0xbe, 0x04, 0xef]),
        ("movsx rax, byte ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 1, true); }, &[0x48, 0x0f, 0xbe, 0x44, 0x4b, 0x20]),
        ("movsx r9, byte ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x08]),
        ("movsx r9, byte ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("movsx r9, byte ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx r9, byte ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4c, 0x24, 0x7f]),
        ("movsx r9, byte ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsx r9, byte ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4d, 0xff]),
        ("movsx r9, byte ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx r9, byte ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x48, 0x01]),
        ("movsx r9, byte ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x48, 0x80]),
        ("movsx r9, byte ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0x24]),
        ("movsx r9, byte ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsx r9, byte ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx r9, byte ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x4d, 0x7f]),
        ("movsx r9, byte ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("movsx r9, byte ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x4f, 0xff]),
        ("movsx r9, byte ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx r9, byte ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4b, 0x01]),
        ("movsx r9, byte ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4b, 0x80]),
        ("movsx r9, byte ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x0c, 0x00]),
        ("movsx r9, byte ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x0c, 0xc8]),
        ("movsx r9, byte ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x0c, 0x30]),
        ("movsx r9, byte ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0xc0]),
        ("movsx r9, byte ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0x28]),
        ("movsx r9, byte ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0xf8]),
        ("movsx r9, byte ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x0c, 0x0c]),
        ("movsx r9, byte ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x0c, 0xec]),
        ("movsx r9, byte ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0x04]),
        ("movsx r9, byte ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0xe4]),
        ("movsx r9, byte ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x0c, 0x3c]),
        ("movsx r9, byte ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4c, 0xc5, 0x00]),
        ("movsx r9, byte ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4c, 0x2d, 0x00]),
        ("movsx r9, byte ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4c, 0xf5, 0x00]),
        ("movsx r9, byte ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x4c, 0x25, 0x00]),
        ("movsx r9, byte ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 1, true); }, &[0x4e, 0x0f, 0xbe, 0x4c, 0xed, 0x00]),
        ("movsx r9, byte ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0x04]),
        ("movsx r9, byte ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0xcc]),
        ("movsx r9, byte ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0x34]),
        ("movsx r9, byte ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x0c, 0xc4]),
        ("movsx r9, byte ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x0c, 0x2c]),
        ("movsx r9, byte ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x0c, 0xfc]),
        ("movsx r9, byte ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x4c, 0x0d, 0x00]),
        ("movsx r9, byte ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x4c, 0xed, 0x00]),
        ("movsx r9, byte ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x4c, 0x05, 0x00]),
        ("movsx r9, byte ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x4c, 0xe5, 0x00]),
        ("movsx r9, byte ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x4c, 0x3d, 0x00]),
        ("movsx r9, byte ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0xc7]),
        ("movsx r9, byte ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0x2f]),
        ("movsx r9, byte ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 1, true); }, &[0x4d, 0x0f, 0xbe, 0x0c, 0xf7]),
        ("movsx r9, byte ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x0c, 0x27]),
        ("movsx r9, byte ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 1, true); }, &[0x4f, 0x0f, 0xbe, 0x0c, 0xef]),
        ("movsx r9, byte ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 1, true); }, &[0x4c, 0x0f, 0xbe, 0x4c, 0x4b, 0x20]),
        ("movzx eax, word ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 2, false); }, &[0x0f, 0xb7, 0x00]),
        ("movzx eax, word ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 2, false); }, &[0x0f, 0xb7, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("movzx eax, word ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 2, false); }, &[0x0f, 0xb7, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx eax, word ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 2, false); }, &[0x0f, 0xb7, 0x44, 0x24, 0x7f]),
        ("movzx eax, word ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 2, false); }, &[0x0f, 0xb7, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movzx eax, word ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 2, false); }, &[0x0f, 0xb7, 0x45, 0xff]),
        ("movzx eax, word ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 2, false); }, &[0x0f, 0xb7, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx eax, word ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 2, false); }, &[0x41, 0x0f, 0xb7, 0x40, 0x01]),
        ("movzx eax, word ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 2, false); }, &[0x41, 0x0f, 0xb7, 0x40, 0x80]),
        ("movzx eax, word ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0x24]),
        ("movzx eax, word ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 2, false); }, &[0x41, 0x0f, 0xb7, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movzx eax, word ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 2, false); }, &[0x41, 0x0f, 0xb7, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx eax, word ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 2, false); }, &[0x41, 0x0f, 0xb7, 0x45, 0x7f]),
        ("movzx eax, word ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 2, false); }, &[0x41, 0x0f, 0xb7, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("movzx eax, word ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 2, false); }, &[0x41, 0x0f, 0xb7, 0x47, 0xff]),
        ("movzx eax, word ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 2, false); }, &[0x41, 0x0f, 0xb7, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx eax, word ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 2, false); }, &[0x0f, 0xb7, 0x43, 0x01]),
        ("movzx eax, word ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 2, false); }, &[0x0f, 0xb7, 0x43, 0x80]),
        ("movzx eax, word ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 2, false); }, &[0x0f, 0xb7, 0x04, 0x00]),
        ("movzx eax, word ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 2, false); }, &[0x0f, 0xb7, 0x04, 0xc8]),
        ("movzx eax, word ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 2, false); }, &[0x0f, 0xb7, 0x04, 0x30]),
        ("movzx eax, word ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0xc0]),
        ("movzx eax, word ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0x28]),
        ("movzx eax, word ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0xf8]),
        ("movzx eax, word ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 2, false); }, &[0x0f, 0xb7, 0x04, 0x0c]),
        ("movzx eax, word ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 2, false); }, &[0x0f, 0xb7, 0x04, 0xec]),
        ("movzx eax, word ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0x04]),
        ("movzx eax, word ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0xe4]),
        ("movzx eax, word ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x04, 0x3c]),
        ("movzx eax, word ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 2, false); }, &[0x0f, 0xb7, 0x44, 0xc5, 0x00]),
        ("movzx eax, word ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 2, false); }, &[0x0f, 0xb7, 0x44, 0x2d, 0x00]),
        ("movzx eax, word ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 2, false); }, &[0x0f, 0xb7, 0x44, 0xf5, 0x00]),
        ("movzx eax, word ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x44, 0x25, 0x00]),
        ("movzx eax, word ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 2, false); }, &[0x42, 0x0f, 0xb7, 0x44, 0xed, 0x00]),
        ("movzx eax, word ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0x04]),
        ("movzx eax, word ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0xcc]),
        ("movzx eax, word ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0x34]),
        ("movzx eax, word ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x04, 0xc4]),
        ("movzx eax, word ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x04, 0x2c]),
        ("movzx eax, word ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x04, 0xfc]),
        ("movzx eax, word ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x44, 0x0d, 0x00]),
        ("movzx eax, word ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x44, 0xed, 0x00]),
        ("movzx eax, word ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x44, 0x05, 0x00]),
        ("movzx eax, word ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x44, 0xe5, 0x00]),
        ("movzx eax, word ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x44, 0x3d, 0x00]),
        ("movzx eax, word ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0xc7]),
        ("movzx eax, word ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0x2f]),
        ("movzx eax, word ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 2, false); }, &[0x41, 0x0f, 0xb7, 0x04, 0xf7]),
        ("movzx eax, word ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x04, 0x27]),
        ("movzx eax, word ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 2, false); }, &[0x43, 0x0f, 0xb7, 0x04, 0xef]),
        ("movzx eax, word ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 2, false); }, &[0x0f, 0xb7, 0x44, 0x4b, 0x20]),
        ("movzx r9d, word ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x08]),
        ("movzx r9d, word ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 2, false); }, &[0x44, 0x0f, 0xb7, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("movzx r9d, word ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 2, false); }, &[0x44, 0x0f, 0xb7, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx r9d, word ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4c, 0x24, 0x7f]),
        ("movzx r9d, word ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 2, false); }, &[0x44, 0x0f, 0xb7, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movzx r9d, word ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4d, 0xff]),
        ("movzx r9d, word ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 2, false); }, &[0x44, 0x0f, 0xb7, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx r9d, word ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 2, false); }, &[0x45, 0x0f, 0xb7, 0x48, 0x01]),
        ("movzx r9d, word ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 2, false); }, &[0x45, 0x0f, 0xb7, 0x48, 0x80]),
        ("movzx r9d, word ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0x24]),
        ("movzx r9d, word ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 2, false); }, &[0x45, 0x0f, 0xb7, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movzx r9d, word ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 2, false); }, &[0x45, 0x0f, 0xb7, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movzx r9d, word ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 2, false); }, &[0x45, 0x0f, 0xb7, 0x4d, 0x7f]),
        ("movzx r9d, word ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 2, false); }, &[0x45, 0x0f, 0xb7, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("movzx r9d, word ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 2, false); }, &[0x45, 0x0f, 0xb7, 0x4f, 0xff]),
        ("movzx r9d, word ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 2, false); }, &[0x45, 0x0f, 0xb7, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("movzx r9d, word ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4b, 0x01]),
        ("movzx r9d, word ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4b, 0x80]),
        ("movzx r9d, word ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x0c, 0x00]),
        ("movzx r9d, word ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x0c, 0xc8]),
        ("movzx r9d, word ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x0c, 0x30]),
        ("movzx r9d, word ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0xc0]),
        ("movzx r9d, word ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0x28]),
        ("movzx r9d, word ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0xf8]),
        ("movzx r9d, word ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x0c, 0x0c]),
        ("movzx r9d, word ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x0c, 0xec]),
        ("movzx r9d, word ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0x04]),
        ("movzx r9d, word ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0xe4]),
        ("movzx r9d, word ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x0c, 0x3c]),
        ("movzx r9d, word ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4c, 0xc5, 0x00]),
        ("movzx r9d, word ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4c, 0x2d, 0x00]),
        ("movzx r9d, word ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4c, 0xf5, 0x00]),
        ("movzx r9d, word ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x4c, 0x25, 0x00]),
        ("movzx r9d, word ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 2, false); }, &[0x46, 0x0f, 0xb7, 0x4c, 0xed, 0x00]),
        ("movzx r9d, word ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0x04]),
        ("movzx r9d, word ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0xcc]),
        ("movzx r9d, word ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0x34]),
        ("movzx r9d, word ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x0c, 0xc4]),
        ("movzx r9d, word ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x0c, 0x2c]),
        ("movzx r9d, word ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x0c, 0xfc]),
        ("movzx r9d, word ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x4c, 0x0d, 0x00]),
        ("movzx r9d, word ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x4c, 0xed, 0x00]),
        ("movzx r9d, word ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x4c, 0x05, 0x00]),
        ("movzx r9d, word ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x4c, 0xe5, 0x00]),
        ("movzx r9d, word ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x4c, 0x3d, 0x00]),
        ("movzx r9d, word ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0xc7]),
        ("movzx r9d, word ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0x2f]),
        ("movzx r9d, word ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 2, false); }, &[0x45, 0x0f, 0xb7, 0x0c, 0xf7]),
        ("movzx r9d, word ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x0c, 0x27]),
        ("movzx r9d, word ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 2, false); }, &[0x47, 0x0f, 0xb7, 0x0c, 0xef]),
        ("movzx r9d, word ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 2, false); }, &[0x44, 0x0f, 0xb7, 0x4c, 0x4b, 0x20]),
        ("movsx rax, word ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x00]),
        ("movsx rax, word ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 2, true); }, &[0x48, 0x0f, 0xbf, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("movsx rax, word ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 2, true); }, &[0x48, 0x0f, 0xbf, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx rax, word ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 2, true); }, &[0x48, 0x0f, 0xbf, 0x44, 0x24, 0x7f]),
        ("movsx rax, word ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 2, true); }, &[0x48, 0x0f, 0xbf, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsx rax, word ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 2, true); }, &[0x48, 0x0f, 0xbf, 0x45, 0xff]),
        ("movsx rax, word ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 2, true); }, &[0x48, 0x0f, 0xbf, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx rax, word ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 2, true); }, &[0x49, 0x0f, 0xbf, 0x40, 0x01]),
        ("movsx rax, word ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 2, true); }, &[0x49, 0x0f, 0xbf, 0x40, 0x80]),
        ("movsx rax, word ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0x24]),
        ("movsx rax, word ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 2, true); }, &[0x49, 0x0f, 0xbf, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsx rax, word ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 2, true); }, &[0x49, 0x0f, 0xbf, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx rax, word ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 2, true); }, &[0x49, 0x0f, 0xbf, 0x45, 0x7f]),
        ("movsx rax, word ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 2, true); }, &[0x49, 0x0f, 0xbf, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("movsx rax, word ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 2, true); }, &[0x49, 0x0f, 0xbf, 0x47, 0xff]),
        ("movsx rax, word ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 2, true); }, &[0x49, 0x0f, 0xbf, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx rax, word ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 2, true); }, &[0x48, 0x0f, 0xbf, 0x43, 0x01]),
        ("movsx rax, word ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 2, true); }, &[0x48, 0x0f, 0xbf, 0x43, 0x80]),
        ("movsx rax, word ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x04, 0x00]),
        ("movsx rax, word ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x04, 0xc8]),
        ("movsx rax, word ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x04, 0x30]),
        ("movsx rax, word ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0xc0]),
        ("movsx rax, word ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0x28]),
        ("movsx rax, word ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0xf8]),
        ("movsx rax, word ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x04, 0x0c]),
        ("movsx rax, word ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x04, 0xec]),
        ("movsx rax, word ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0x04]),
        ("movsx rax, word ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0xe4]),
        ("movsx rax, word ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x04, 0x3c]),
        ("movsx rax, word ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x44, 0xc5, 0x00]),
        ("movsx rax, word ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x44, 0x2d, 0x00]),
        ("movsx rax, word ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 2, true); }, &[0x48, 0x0f, 0xbf, 0x44, 0xf5, 0x00]),
        ("movsx rax, word ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x44, 0x25, 0x00]),
        ("movsx rax, word ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 2, true); }, &[0x4a, 0x0f, 0xbf, 0x44, 0xed, 0x00]),
        ("movsx rax, word ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0x04]),
        ("movsx rax, word ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0xcc]),
        ("movsx rax, word ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0x34]),
        ("movsx rax, word ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x04, 0xc4]),
        ("movsx rax, word ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x04, 0x2c]),
        ("movsx rax, word ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x04, 0xfc]),
        ("movsx rax, word ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x44, 0x0d, 0x00]),
        ("movsx rax, word ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x44, 0xed, 0x00]),
        ("movsx rax, word ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x44, 0x05, 0x00]),
        ("movsx rax, word ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x44, 0xe5, 0x00]),
        ("movsx rax, word ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x44, 0x3d, 0x00]),
        ("movsx rax, word ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0xc7]),
        ("movsx rax, word ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0x2f]),
        ("movsx rax, word ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 2, true); }, &[0x49, 0x0f, 0xbf, 0x04, 0xf7]),
        ("movsx rax, word ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x04, 0x27]),
        ("movsx rax, word ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 2, true); }, &[0x4b, 0x0f, 0xbf, 0x04, 0xef]),
        ("movsx rax, word ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 2, true); }, &[0x48, 0x0f, 0xbf, 0x44, 0x4b, 0x20]),
        ("movsx r9, word ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x08]),
        ("movsx r9, word ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("movsx r9, word ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx r9, word ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4c, 0x24, 0x7f]),
        ("movsx r9, word ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsx r9, word ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4d, 0xff]),
        ("movsx r9, word ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx r9, word ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x48, 0x01]),
        ("movsx r9, word ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x48, 0x80]),
        ("movsx r9, word ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0x24]),
        ("movsx r9, word ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsx r9, word ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsx r9, word ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x4d, 0x7f]),
        ("movsx r9, word ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("movsx r9, word ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x4f, 0xff]),
        ("movsx r9, word ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("movsx r9, word ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4b, 0x01]),
        ("movsx r9, word ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4b, 0x80]),
        ("movsx r9, word ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x0c, 0x00]),
        ("movsx r9, word ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x0c, 0xc8]),
        ("movsx r9, word ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x0c, 0x30]),
        ("movsx r9, word ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0xc0]),
        ("movsx r9, word ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0x28]),
        ("movsx r9, word ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0xf8]),
        ("movsx r9, word ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x0c, 0x0c]),
        ("movsx r9, word ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x0c, 0xec]),
        ("movsx r9, word ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0x04]),
        ("movsx r9, word ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0xe4]),
        ("movsx r9, word ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x0c, 0x3c]),
        ("movsx r9, word ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4c, 0xc5, 0x00]),
        ("movsx r9, word ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4c, 0x2d, 0x00]),
        ("movsx r9, word ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4c, 0xf5, 0x00]),
        ("movsx r9, word ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x4c, 0x25, 0x00]),
        ("movsx r9, word ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 2, true); }, &[0x4e, 0x0f, 0xbf, 0x4c, 0xed, 0x00]),
        ("movsx r9, word ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0x04]),
        ("movsx r9, word ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0xcc]),
        ("movsx r9, word ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0x34]),
        ("movsx r9, word ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x0c, 0xc4]),
        ("movsx r9, word ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x0c, 0x2c]),
        ("movsx r9, word ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x0c, 0xfc]),
        ("movsx r9, word ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x4c, 0x0d, 0x00]),
        ("movsx r9, word ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x4c, 0xed, 0x00]),
        ("movsx r9, word ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x4c, 0x05, 0x00]),
        ("movsx r9, word ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x4c, 0xe5, 0x00]),
        ("movsx r9, word ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x4c, 0x3d, 0x00]),
        ("movsx r9, word ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0xc7]),
        ("movsx r9, word ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0x2f]),
        ("movsx r9, word ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 2, true); }, &[0x4d, 0x0f, 0xbf, 0x0c, 0xf7]),
        ("movsx r9, word ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x0c, 0x27]),
        ("movsx r9, word ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 2, true); }, &[0x4f, 0x0f, 0xbf, 0x0c, 0xef]),
        ("movsx r9, word ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 2, true); }, &[0x4c, 0x0f, 0xbf, 0x4c, 0x4b, 0x20]),
        ("mov eax, dword ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 4, false); }, &[0x8b, 0x00]),
        ("mov eax, dword ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 4, false); }, &[0x8b, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("mov eax, dword ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 4, false); }, &[0x8b, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov eax, dword ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 4, false); }, &[0x8b, 0x44, 0x24, 0x7f]),
        ("mov eax, dword ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 4, false); }, &[0x8b, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov eax, dword ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 4, false); }, &[0x8b, 0x45, 0xff]),
        ("mov eax, dword ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 4, false); }, &[0x8b, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov eax, dword ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 4, false); }, &[0x41, 0x8b, 0x40, 0x01]),
        ("mov eax, dword ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 4, false); }, &[0x41, 0x8b, 0x40, 0x80]),
        ("mov eax, dword ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0x24]),
        ("mov eax, dword ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 4, false); }, &[0x41, 0x8b, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov eax, dword ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 4, false); }, &[0x41, 0x8b, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov eax, dword ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 4, false); }, &[0x41, 0x8b, 0x45, 0x7f]),
        ("mov eax, dword ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 4, false); }, &[0x41, 0x8b, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("mov eax, dword ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 4, false); }, &[0x41, 0x8b, 0x47, 0xff]),
        ("mov eax, dword ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 4, false); }, &[0x41, 0x8b, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("mov eax, dword ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 4, false); }, &[0x8b, 0x43, 0x01]),
        ("mov eax, dword ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 4, false); }, &[0x8b, 0x43, 0x80]),
        ("mov eax, dword ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 4, false); }, &[0x8b, 0x04, 0x00]),
        ("mov eax, dword ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 4, false); }, &[0x8b, 0x04, 0xc8]),
        ("mov eax, dword ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 4, false); }, &[0x8b, 0x04, 0x30]),
        ("mov eax, dword ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0xc0]),
        ("mov eax, dword ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0x28]),
        ("mov eax, dword ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0xf8]),
        ("mov eax, dword ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 4, false); }, &[0x8b, 0x04, 0x0c]),
        ("mov eax, dword ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 4, false); }, &[0x8b, 0x04, 0xec]),
        ("mov eax, dword ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0x04]),
        ("mov eax, dword ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0xe4]),
        ("mov eax, dword ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 4, false); }, &[0x42, 0x8b, 0x04, 0x3c]),
        ("mov eax, dword ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 4, false); }, &[0x8b, 0x44, 0xc5, 0x00]),
        ("mov eax, dword ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 4, false); }, &[0x8b, 0x44, 0x2d, 0x00]),
        ("mov eax, dword ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 4, false); }, &[0x8b, 0x44, 0xf5, 0x00]),
        ("mov eax, dword ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 4, false); }, &[0x42, 0x8b, 0x44, 0x25, 0x00]),
        ("mov eax, dword ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 4, false); }, &[0x42, 0x8b, 0x44, 0xed, 0x00]),
        ("mov eax, dword ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0x04]),
        ("mov eax, dword ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0xcc]),
        ("mov eax, dword ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0x34]),
        ("mov eax, dword ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 4, false); }, &[0x43, 0x8b, 0x04, 0xc4]),
        ("mov eax, dword ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 4, false); }, &[0x43, 0x8b, 0x04, 0x2c]),
        ("mov eax, dword ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 4, false); }, &[0x43, 0x8b, 0x04, 0xfc]),
        ("mov eax, dword ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 4, false); }, &[0x41, 0x8b, 0x44, 0x0d, 0x00]),
        ("mov eax, dword ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 4, false); }, &[0x41, 0x8b, 0x44, 0xed, 0x00]),
        ("mov eax, dword ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 4, false); }, &[0x43, 0x8b, 0x44, 0x05, 0x00]),
        ("mov eax, dword ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 4, false); }, &[0x43, 0x8b, 0x44, 0xe5, 0x00]),
        ("mov eax, dword ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 4, false); }, &[0x43, 0x8b, 0x44, 0x3d, 0x00]),
        ("mov eax, dword ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0xc7]),
        ("mov eax, dword ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0x2f]),
        ("mov eax, dword ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 4, false); }, &[0x41, 0x8b, 0x04, 0xf7]),
        ("mov eax, dword ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 4, false); }, &[0x43, 0x8b, 0x04, 0x27]),
        ("mov eax, dword ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 4, false); }, &[0x43, 0x8b, 0x04, 0xef]),
        ("mov eax, dword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 4, false); }, &[0x8b, 0x44, 0x4b, 0x20]),
        ("mov r9d, dword ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 4, false); }, &[0x44, 0x8b, 0x08]),
        ("mov r9d, dword ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 4, false); }, &[0x44, 0x8b, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9d, dword ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 4, false); }, &[0x44, 0x8b, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9d, dword ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 4, false); }, &[0x44, 0x8b, 0x4c, 0x24, 0x7f]),
        ("mov r9d, dword ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 4, false); }, &[0x44, 0x8b, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9d, dword ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 4, false); }, &[0x44, 0x8b, 0x4d, 0xff]),
        ("mov r9d, dword ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 4, false); }, &[0x44, 0x8b, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9d, dword ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 4, false); }, &[0x45, 0x8b, 0x48, 0x01]),
        ("mov r9d, dword ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 4, false); }, &[0x45, 0x8b, 0x48, 0x80]),
        ("mov r9d, dword ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0x24]),
        ("mov r9d, dword ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 4, false); }, &[0x45, 0x8b, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9d, dword ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 4, false); }, &[0x45, 0x8b, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9d, dword ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 4, false); }, &[0x45, 0x8b, 0x4d, 0x7f]),
        ("mov r9d, dword ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 4, false); }, &[0x45, 0x8b, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9d, dword ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 4, false); }, &[0x45, 0x8b, 0x4f, 0xff]),
        ("mov r9d, dword ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 4, false); }, &[0x45, 0x8b, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9d, dword ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 4, false); }, &[0x44, 0x8b, 0x4b, 0x01]),
        ("mov r9d, dword ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 4, false); }, &[0x44, 0x8b, 0x4b, 0x80]),
        ("mov r9d, dword ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 4, false); }, &[0x44, 0x8b, 0x0c, 0x00]),
        ("mov r9d, dword ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 4, false); }, &[0x44, 0x8b, 0x0c, 0xc8]),
        ("mov r9d, dword ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 4, false); }, &[0x44, 0x8b, 0x0c, 0x30]),
        ("mov r9d, dword ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0xc0]),
        ("mov r9d, dword ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0x28]),
        ("mov r9d, dword ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0xf8]),
        ("mov r9d, dword ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 4, false); }, &[0x44, 0x8b, 0x0c, 0x0c]),
        ("mov r9d, dword ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 4, false); }, &[0x44, 0x8b, 0x0c, 0xec]),
        ("mov r9d, dword ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0x04]),
        ("mov r9d, dword ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0xe4]),
        ("mov r9d, dword ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 4, false); }, &[0x46, 0x8b, 0x0c, 0x3c]),
        ("mov r9d, dword ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 4, false); }, &[0x44, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9d, dword ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 4, false); }, &[0x44, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9d, dword ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 4, false); }, &[0x44, 0x8b, 0x4c, 0xf5, 0x00]),
        ("mov r9d, dword ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 4, false); }, &[0x46, 0x8b, 0x4c, 0x25, 0x00]),
        ("mov r9d, dword ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 4, false); }, &[0x46, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9d, dword ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0x04]),
        ("mov r9d, dword ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0xcc]),
        ("mov r9d, dword ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0x34]),
        ("mov r9d, dword ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 4, false); }, &[0x47, 0x8b, 0x0c, 0xc4]),
        ("mov r9d, dword ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 4, false); }, &[0x47, 0x8b, 0x0c, 0x2c]),
        ("mov r9d, dword ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 4, false); }, &[0x47, 0x8b, 0x0c, 0xfc]),
        ("mov r9d, dword ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 4, false); }, &[0x45, 0x8b, 0x4c, 0x0d, 0x00]),
        ("mov r9d, dword ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 4, false); }, &[0x45, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9d, dword ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 4, false); }, &[0x47, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9d, dword ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 4, false); }, &[0x47, 0x8b, 0x4c, 0xe5, 0x00]),
        ("mov r9d, dword ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 4, false); }, &[0x47, 0x8b, 0x4c, 0x3d, 0x00]),
        ("mov r9d, dword ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0xc7]),
        ("mov r9d, dword ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0x2f]),
        ("mov r9d, dword ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 4, false); }, &[0x45, 0x8b, 0x0c, 0xf7]),
        ("mov r9d, dword ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 4, false); }, &[0x47, 0x8b, 0x0c, 0x27]),
        ("mov r9d, dword ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 4, false); }, &[0x47, 0x8b, 0x0c, 0xef]),
        ("mov r9d, dword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 4, false); }, &[0x44, 0x8b, 0x4c, 0x4b, 0x20]),
        ("movsxd rax, dword ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 4, true); }, &[0x48, 0x63, 0x00]),
        ("movsxd rax, dword ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 4, true); }, &[0x48, 0x63, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("movsxd rax, dword ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 4, true); }, &[0x48, 0x63, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsxd rax, dword ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 4, true); }, &[0x48, 0x63, 0x44, 0x24, 0x7f]),
        ("movsxd rax, dword ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 4, true); }, &[0x48, 0x63, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsxd rax, dword ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 4, true); }, &[0x48, 0x63, 0x45, 0xff]),
        ("movsxd rax, dword ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 4, true); }, &[0x48, 0x63, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("movsxd rax, dword ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 4, true); }, &[0x49, 0x63, 0x40, 0x01]),
        ("movsxd rax, dword ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 4, true); }, &[0x49, 0x63, 0x40, 0x80]),
        ("movsxd rax, dword ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 4, true); }, &[0x49, 0x63, 0x04, 0x24]),
        ("movsxd rax, dword ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 4, true); }, &[0x49, 0x63, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsxd rax, dword ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 4, true); }, &[0x49, 0x63, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsxd rax, dword ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 4, true); }, &[0x49, 0x63, 0x45, 0x7f]),
        ("movsxd rax, dword ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 4, true); }, &[0x49, 0x63, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("movsxd rax, dword ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 4, true); }, &[0x49, 0x63, 0x47, 0xff]),
        ("movsxd rax, dword ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 4, true); }, &[0x49, 0x63, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("movsxd rax, dword ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 4, true); }, &[0x48, 0x63, 0x43, 0x01]),
        ("movsxd rax, dword ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 4, true); }, &[0x48, 0x63, 0x43, 0x80]),
        ("movsxd rax, dword ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 4, true); }, &[0x48, 0x63, 0x04, 0x00]),
        ("movsxd rax, dword ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 4, true); }, &[0x48, 0x63, 0x04, 0xc8]),
        ("movsxd rax, dword ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 4, true); }, &[0x48, 0x63, 0x04, 0x30]),
        ("movsxd rax, dword ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0xc0]),
        ("movsxd rax, dword ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0x28]),
        ("movsxd rax, dword ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0xf8]),
        ("movsxd rax, dword ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 4, true); }, &[0x48, 0x63, 0x04, 0x0c]),
        ("movsxd rax, dword ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 4, true); }, &[0x48, 0x63, 0x04, 0xec]),
        ("movsxd rax, dword ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0x04]),
        ("movsxd rax, dword ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0xe4]),
        ("movsxd rax, dword ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 4, true); }, &[0x4a, 0x63, 0x04, 0x3c]),
        ("movsxd rax, dword ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 4, true); }, &[0x48, 0x63, 0x44, 0xc5, 0x00]),
        ("movsxd rax, dword ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 4, true); }, &[0x48, 0x63, 0x44, 0x2d, 0x00]),
        ("movsxd rax, dword ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 4, true); }, &[0x48, 0x63, 0x44, 0xf5, 0x00]),
        ("movsxd rax, dword ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 4, true); }, &[0x4a, 0x63, 0x44, 0x25, 0x00]),
        ("movsxd rax, dword ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 4, true); }, &[0x4a, 0x63, 0x44, 0xed, 0x00]),
        ("movsxd rax, dword ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 4, true); }, &[0x49, 0x63, 0x04, 0x04]),
        ("movsxd rax, dword ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 4, true); }, &[0x49, 0x63, 0x04, 0xcc]),
        ("movsxd rax, dword ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 4, true); }, &[0x49, 0x63, 0x04, 0x34]),
        ("movsxd rax, dword ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 4, true); }, &[0x4b, 0x63, 0x04, 0xc4]),
        ("movsxd rax, dword ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 4, true); }, &[0x4b, 0x63, 0x04, 0x2c]),
        ("movsxd rax, dword ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 4, true); }, &[0x4b, 0x63, 0x04, 0xfc]),
        ("movsxd rax, dword ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 4, true); }, &[0x49, 0x63, 0x44, 0x0d, 0x00]),
        ("movsxd rax, dword ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 4, true); }, &[0x49, 0x63, 0x44, 0xed, 0x00]),
        ("movsxd rax, dword ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 4, true); }, &[0x4b, 0x63, 0x44, 0x05, 0x00]),
        ("movsxd rax, dword ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 4, true); }, &[0x4b, 0x63, 0x44, 0xe5, 0x00]),
        ("movsxd rax, dword ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 4, true); }, &[0x4b, 0x63, 0x44, 0x3d, 0x00]),
        ("movsxd rax, dword ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 4, true); }, &[0x49, 0x63, 0x04, 0xc7]),
        ("movsxd rax, dword ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 4, true); }, &[0x49, 0x63, 0x04, 0x2f]),
        ("movsxd rax, dword ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 4, true); }, &[0x49, 0x63, 0x04, 0xf7]),
        ("movsxd rax, dword ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 4, true); }, &[0x4b, 0x63, 0x04, 0x27]),
        ("movsxd rax, dword ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 4, true); }, &[0x4b, 0x63, 0x04, 0xef]),
        ("movsxd rax, dword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 4, true); }, &[0x48, 0x63, 0x44, 0x4b, 0x20]),
        ("movsxd r9, dword ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 4, true); }, &[0x4c, 0x63, 0x08]),
        ("movsxd r9, dword ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 4, true); }, &[0x4c, 0x63, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("movsxd r9, dword ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 4, true); }, &[0x4c, 0x63, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsxd r9, dword ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 4, true); }, &[0x4c, 0x63, 0x4c, 0x24, 0x7f]),
        ("movsxd r9, dword ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 4, true); }, &[0x4c, 0x63, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("movsxd r9, dword ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 4, true); }, &[0x4c, 0x63, 0x4d, 0xff]),
        ("movsxd r9, dword ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 4, true); }, &[0x4c, 0x63, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("movsxd r9, dword ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 4, true); }, &[0x4d, 0x63, 0x48, 0x01]),
        ("movsxd r9, dword ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 4, true); }, &[0x4d, 0x63, 0x48, 0x80]),
        ("movsxd r9, dword ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0x24]),
        ("movsxd r9, dword ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 4, true); }, &[0x4d, 0x63, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("movsxd r9, dword ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 4, true); }, &[0x4d, 0x63, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("movsxd r9, dword ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 4, true); }, &[0x4d, 0x63, 0x4d, 0x7f]),
        ("movsxd r9, dword ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 4, true); }, &[0x4d, 0x63, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("movsxd r9, dword ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 4, true); }, &[0x4d, 0x63, 0x4f, 0xff]),
        ("movsxd r9, dword ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 4, true); }, &[0x4d, 0x63, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("movsxd r9, dword ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 4, true); }, &[0x4c, 0x63, 0x4b, 0x01]),
        ("movsxd r9, dword ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 4, true); }, &[0x4c, 0x63, 0x4b, 0x80]),
        ("movsxd r9, dword ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 4, true); }, &[0x4c, 0x63, 0x0c, 0x00]),
        ("movsxd r9, dword ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 4, true); }, &[0x4c, 0x63, 0x0c, 0xc8]),
        ("movsxd r9, dword ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 4, true); }, &[0x4c, 0x63, 0x0c, 0x30]),
        ("movsxd r9, dword ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0xc0]),
        ("movsxd r9, dword ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0x28]),
        ("movsxd r9, dword ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0xf8]),
        ("movsxd r9, dword ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 4, true); }, &[0x4c, 0x63, 0x0c, 0x0c]),
        ("movsxd r9, dword ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 4, true); }, &[0x4c, 0x63, 0x0c, 0xec]),
        ("movsxd r9, dword ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0x04]),
        ("movsxd r9, dword ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0xe4]),
        ("movsxd r9, dword ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 4, true); }, &[0x4e, 0x63, 0x0c, 0x3c]),
        ("movsxd r9, dword ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 4, true); }, &[0x4c, 0x63, 0x4c, 0xc5, 0x00]),
        ("movsxd r9, dword ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 4, true); }, &[0x4c, 0x63, 0x4c, 0x2d, 0x00]),
        ("movsxd r9, dword ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 4, true); }, &[0x4c, 0x63, 0x4c, 0xf5, 0x00]),
        ("movsxd r9, dword ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 4, true); }, &[0x4e, 0x63, 0x4c, 0x25, 0x00]),
        ("movsxd r9, dword ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 4, true); }, &[0x4e, 0x63, 0x4c, 0xed, 0x00]),
        ("movsxd r9, dword ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0x04]),
        ("movsxd r9, dword ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0xcc]),
        ("movsxd r9, dword ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0x34]),
        ("movsxd r9, dword ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 4, true); }, &[0x4f, 0x63, 0x0c, 0xc4]),
        ("movsxd r9, dword ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 4, true); }, &[0x4f, 0x63, 0x0c, 0x2c]),
        ("movsxd r9, dword ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 4, true); }, &[0x4f, 0x63, 0x0c, 0xfc]),
        ("movsxd r9, dword ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 4, true); }, &[0x4d, 0x63, 0x4c, 0x0d, 0x00]),
        ("movsxd r9, dword ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 4, true); }, &[0x4d, 0x63, 0x4c, 0xed, 0x00]),
        ("movsxd r9, dword ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 4, true); }, &[0x4f, 0x63, 0x4c, 0x05, 0x00]),
        ("movsxd r9, dword ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 4, true); }, &[0x4f, 0x63, 0x4c, 0xe5, 0x00]),
        ("movsxd r9, dword ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 4, true); }, &[0x4f, 0x63, 0x4c, 0x3d, 0x00]),
        ("movsxd r9, dword ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0xc7]),
        ("movsxd r9, dword ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0x2f]),
        ("movsxd r9, dword ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 4, true); }, &[0x4d, 0x63, 0x0c, 0xf7]),
        ("movsxd r9, dword ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 4, true); }, &[0x4f, 0x63, 0x0c, 0x27]),
        ("movsxd r9, dword ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 4, true); }, &[0x4f, 0x63, 0x0c, 0xef]),
        ("movsxd r9, dword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 4, true); }, &[0x4c, 0x63, 0x4c, 0x4b, 0x20]),
        ("mov rax, qword ptr [rax]", |a| { a.ld_ext(RAX, mem(RAX, 0), 8, false); }, &[0x48, 0x8b, 0x00]),
        ("mov rax, qword ptr [rax + 128]", |a| { a.ld_ext(RAX, mem(RAX, 128), 8, false); }, &[0x48, 0x8b, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [rax - 305419896]", |a| { a.ld_ext(RAX, mem(RAX, -305419896), 8, false); }, &[0x48, 0x8b, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [rsp + 127]", |a| { a.ld_ext(RAX, mem(RSP, 127), 8, false); }, &[0x48, 0x8b, 0x44, 0x24, 0x7f]),
        ("mov rax, qword ptr [rsp + 305419896]", |a| { a.ld_ext(RAX, mem(RSP, 305419896), 8, false); }, &[0x48, 0x8b, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [rbp - 1]", |a| { a.ld_ext(RAX, mem(RBP, -1), 8, false); }, &[0x48, 0x8b, 0x45, 0xff]),
        ("mov rax, qword ptr [rbp - 129]", |a| { a.ld_ext(RAX, mem(RBP, -129), 8, false); }, &[0x48, 0x8b, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [r8 + 1]", |a| { a.ld_ext(RAX, mem(R8, 1), 8, false); }, &[0x49, 0x8b, 0x40, 0x01]),
        ("mov rax, qword ptr [r8 - 128]", |a| { a.ld_ext(RAX, mem(R8, -128), 8, false); }, &[0x49, 0x8b, 0x40, 0x80]),
        ("mov rax, qword ptr [r12]", |a| { a.ld_ext(RAX, mem(R12, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0x24]),
        ("mov rax, qword ptr [r12 + 128]", |a| { a.ld_ext(RAX, mem(R12, 128), 8, false); }, &[0x49, 0x8b, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov rax, qword ptr [r12 - 305419896]", |a| { a.ld_ext(RAX, mem(R12, -305419896), 8, false); }, &[0x49, 0x8b, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov rax, qword ptr [r13 + 127]", |a| { a.ld_ext(RAX, mem(R13, 127), 8, false); }, &[0x49, 0x8b, 0x45, 0x7f]),
        ("mov rax, qword ptr [r13 + 305419896]", |a| { a.ld_ext(RAX, mem(R13, 305419896), 8, false); }, &[0x49, 0x8b, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("mov rax, qword ptr [r15 - 1]", |a| { a.ld_ext(RAX, mem(R15, -1), 8, false); }, &[0x49, 0x8b, 0x47, 0xff]),
        ("mov rax, qword ptr [r15 - 129]", |a| { a.ld_ext(RAX, mem(R15, -129), 8, false); }, &[0x49, 0x8b, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("mov rax, qword ptr [rbx + 1]", |a| { a.ld_ext(RAX, mem(RBX, 1), 8, false); }, &[0x48, 0x8b, 0x43, 0x01]),
        ("mov rax, qword ptr [rbx - 128]", |a| { a.ld_ext(RAX, mem(RBX, -128), 8, false); }, &[0x48, 0x8b, 0x43, 0x80]),
        ("mov rax, qword ptr [rax + rax*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RAX, 0, 0), 8, false); }, &[0x48, 0x8b, 0x04, 0x00]),
        ("mov rax, qword ptr [rax + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, RCX, 3, 0), 8, false); }, &[0x48, 0x8b, 0x04, 0xc8]),
        ("mov rax, qword ptr [rax + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, RSI, 0, 0), 8, false); }, &[0x48, 0x8b, 0x04, 0x30]),
        ("mov rax, qword ptr [rax + r8*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R8, 3, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0xc0]),
        ("mov rax, qword ptr [rax + r13*1]", |a| { a.ld_ext(RAX, mem_idx(RAX, R13, 0, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0x28]),
        ("mov rax, qword ptr [rax + r15*8]", |a| { a.ld_ext(RAX, mem_idx(RAX, R15, 3, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0xf8]),
        ("mov rax, qword ptr [rsp + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, RCX, 0, 0), 8, false); }, &[0x48, 0x8b, 0x04, 0x0c]),
        ("mov rax, qword ptr [rsp + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, RBP, 3, 0), 8, false); }, &[0x48, 0x8b, 0x04, 0xec]),
        ("mov rax, qword ptr [rsp + r8*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R8, 0, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [rsp + r12*8]", |a| { a.ld_ext(RAX, mem_idx(RSP, R12, 3, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0xe4]),
        ("mov rax, qword ptr [rsp + r15*1]", |a| { a.ld_ext(RAX, mem_idx(RSP, R15, 0, 0), 8, false); }, &[0x4a, 0x8b, 0x04, 0x3c]),
        ("mov rax, qword ptr [rbp + rax*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RAX, 3, 0), 8, false); }, &[0x48, 0x8b, 0x44, 0xc5, 0x00]),
        ("mov rax, qword ptr [rbp + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, RBP, 0, 0), 8, false); }, &[0x48, 0x8b, 0x44, 0x2d, 0x00]),
        ("mov rax, qword ptr [rbp + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, RSI, 3, 0), 8, false); }, &[0x48, 0x8b, 0x44, 0xf5, 0x00]),
        ("mov rax, qword ptr [rbp + r12*1]", |a| { a.ld_ext(RAX, mem_idx(RBP, R12, 0, 0), 8, false); }, &[0x4a, 0x8b, 0x44, 0x25, 0x00]),
        ("mov rax, qword ptr [rbp + r13*8]", |a| { a.ld_ext(RAX, mem_idx(RBP, R13, 3, 0), 8, false); }, &[0x4a, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [r12 + rax*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RAX, 0, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0x04]),
        ("mov rax, qword ptr [r12 + rcx*8]", |a| { a.ld_ext(RAX, mem_idx(R12, RCX, 3, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0xcc]),
        ("mov rax, qword ptr [r12 + rsi*1]", |a| { a.ld_ext(RAX, mem_idx(R12, RSI, 0, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0x34]),
        ("mov rax, qword ptr [r12 + r8*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R8, 3, 0), 8, false); }, &[0x4b, 0x8b, 0x04, 0xc4]),
        ("mov rax, qword ptr [r12 + r13*1]", |a| { a.ld_ext(RAX, mem_idx(R12, R13, 0, 0), 8, false); }, &[0x4b, 0x8b, 0x04, 0x2c]),
        ("mov rax, qword ptr [r12 + r15*8]", |a| { a.ld_ext(RAX, mem_idx(R12, R15, 3, 0), 8, false); }, &[0x4b, 0x8b, 0x04, 0xfc]),
        ("mov rax, qword ptr [r13 + rcx*1]", |a| { a.ld_ext(RAX, mem_idx(R13, RCX, 0, 0), 8, false); }, &[0x49, 0x8b, 0x44, 0x0d, 0x00]),
        ("mov rax, qword ptr [r13 + rbp*8]", |a| { a.ld_ext(RAX, mem_idx(R13, RBP, 3, 0), 8, false); }, &[0x49, 0x8b, 0x44, 0xed, 0x00]),
        ("mov rax, qword ptr [r13 + r8*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R8, 0, 0), 8, false); }, &[0x4b, 0x8b, 0x44, 0x05, 0x00]),
        ("mov rax, qword ptr [r13 + r12*8]", |a| { a.ld_ext(RAX, mem_idx(R13, R12, 3, 0), 8, false); }, &[0x4b, 0x8b, 0x44, 0xe5, 0x00]),
        ("mov rax, qword ptr [r13 + r15*1]", |a| { a.ld_ext(RAX, mem_idx(R13, R15, 0, 0), 8, false); }, &[0x4b, 0x8b, 0x44, 0x3d, 0x00]),
        ("mov rax, qword ptr [r15 + rax*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RAX, 3, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0xc7]),
        ("mov rax, qword ptr [r15 + rbp*1]", |a| { a.ld_ext(RAX, mem_idx(R15, RBP, 0, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0x2f]),
        ("mov rax, qword ptr [r15 + rsi*8]", |a| { a.ld_ext(RAX, mem_idx(R15, RSI, 3, 0), 8, false); }, &[0x49, 0x8b, 0x04, 0xf7]),
        ("mov rax, qword ptr [r15 + r12*1]", |a| { a.ld_ext(RAX, mem_idx(R15, R12, 0, 0), 8, false); }, &[0x4b, 0x8b, 0x04, 0x27]),
        ("mov rax, qword ptr [r15 + r13*8]", |a| { a.ld_ext(RAX, mem_idx(R15, R13, 3, 0), 8, false); }, &[0x4b, 0x8b, 0x04, 0xef]),
        ("mov rax, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(RAX, mem_idx(RBX, RCX, 1, 32), 8, false); }, &[0x48, 0x8b, 0x44, 0x4b, 0x20]),
        ("mov r9, qword ptr [rax]", |a| { a.ld_ext(R9, mem(RAX, 0), 8, false); }, &[0x4c, 0x8b, 0x08]),
        ("mov r9, qword ptr [rax + 128]", |a| { a.ld_ext(R9, mem(RAX, 128), 8, false); }, &[0x4c, 0x8b, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [rax - 305419896]", |a| { a.ld_ext(R9, mem(RAX, -305419896), 8, false); }, &[0x4c, 0x8b, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [rsp + 127]", |a| { a.ld_ext(R9, mem(RSP, 127), 8, false); }, &[0x4c, 0x8b, 0x4c, 0x24, 0x7f]),
        ("mov r9, qword ptr [rsp + 305419896]", |a| { a.ld_ext(R9, mem(RSP, 305419896), 8, false); }, &[0x4c, 0x8b, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [rbp - 1]", |a| { a.ld_ext(R9, mem(RBP, -1), 8, false); }, &[0x4c, 0x8b, 0x4d, 0xff]),
        ("mov r9, qword ptr [rbp - 129]", |a| { a.ld_ext(R9, mem(RBP, -129), 8, false); }, &[0x4c, 0x8b, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [r8 + 1]", |a| { a.ld_ext(R9, mem(R8, 1), 8, false); }, &[0x4d, 0x8b, 0x48, 0x01]),
        ("mov r9, qword ptr [r8 - 128]", |a| { a.ld_ext(R9, mem(R8, -128), 8, false); }, &[0x4d, 0x8b, 0x48, 0x80]),
        ("mov r9, qword ptr [r12]", |a| { a.ld_ext(R9, mem(R12, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0x24]),
        ("mov r9, qword ptr [r12 + 128]", |a| { a.ld_ext(R9, mem(R12, 128), 8, false); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov r9, qword ptr [r12 - 305419896]", |a| { a.ld_ext(R9, mem(R12, -305419896), 8, false); }, &[0x4d, 0x8b, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r9, qword ptr [r13 + 127]", |a| { a.ld_ext(R9, mem(R13, 127), 8, false); }, &[0x4d, 0x8b, 0x4d, 0x7f]),
        ("mov r9, qword ptr [r13 + 305419896]", |a| { a.ld_ext(R9, mem(R13, 305419896), 8, false); }, &[0x4d, 0x8b, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("mov r9, qword ptr [r15 - 1]", |a| { a.ld_ext(R9, mem(R15, -1), 8, false); }, &[0x4d, 0x8b, 0x4f, 0xff]),
        ("mov r9, qword ptr [r15 - 129]", |a| { a.ld_ext(R9, mem(R15, -129), 8, false); }, &[0x4d, 0x8b, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("mov r9, qword ptr [rbx + 1]", |a| { a.ld_ext(R9, mem(RBX, 1), 8, false); }, &[0x4c, 0x8b, 0x4b, 0x01]),
        ("mov r9, qword ptr [rbx - 128]", |a| { a.ld_ext(R9, mem(RBX, -128), 8, false); }, &[0x4c, 0x8b, 0x4b, 0x80]),
        ("mov r9, qword ptr [rax + rax*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RAX, 0, 0), 8, false); }, &[0x4c, 0x8b, 0x0c, 0x00]),
        ("mov r9, qword ptr [rax + rcx*8]", |a| { a.ld_ext(R9, mem_idx(RAX, RCX, 3, 0), 8, false); }, &[0x4c, 0x8b, 0x0c, 0xc8]),
        ("mov r9, qword ptr [rax + rsi*1]", |a| { a.ld_ext(R9, mem_idx(RAX, RSI, 0, 0), 8, false); }, &[0x4c, 0x8b, 0x0c, 0x30]),
        ("mov r9, qword ptr [rax + r8*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R8, 3, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0xc0]),
        ("mov r9, qword ptr [rax + r13*1]", |a| { a.ld_ext(R9, mem_idx(RAX, R13, 0, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0x28]),
        ("mov r9, qword ptr [rax + r15*8]", |a| { a.ld_ext(R9, mem_idx(RAX, R15, 3, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0xf8]),
        ("mov r9, qword ptr [rsp + rcx*1]", |a| { a.ld_ext(R9, mem_idx(RSP, RCX, 0, 0), 8, false); }, &[0x4c, 0x8b, 0x0c, 0x0c]),
        ("mov r9, qword ptr [rsp + rbp*8]", |a| { a.ld_ext(R9, mem_idx(RSP, RBP, 3, 0), 8, false); }, &[0x4c, 0x8b, 0x0c, 0xec]),
        ("mov r9, qword ptr [rsp + r8*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R8, 0, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [rsp + r12*8]", |a| { a.ld_ext(R9, mem_idx(RSP, R12, 3, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0xe4]),
        ("mov r9, qword ptr [rsp + r15*1]", |a| { a.ld_ext(R9, mem_idx(RSP, R15, 0, 0), 8, false); }, &[0x4e, 0x8b, 0x0c, 0x3c]),
        ("mov r9, qword ptr [rbp + rax*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RAX, 3, 0), 8, false); }, &[0x4c, 0x8b, 0x4c, 0xc5, 0x00]),
        ("mov r9, qword ptr [rbp + rbp*1]", |a| { a.ld_ext(R9, mem_idx(RBP, RBP, 0, 0), 8, false); }, &[0x4c, 0x8b, 0x4c, 0x2d, 0x00]),
        ("mov r9, qword ptr [rbp + rsi*8]", |a| { a.ld_ext(R9, mem_idx(RBP, RSI, 3, 0), 8, false); }, &[0x4c, 0x8b, 0x4c, 0xf5, 0x00]),
        ("mov r9, qword ptr [rbp + r12*1]", |a| { a.ld_ext(R9, mem_idx(RBP, R12, 0, 0), 8, false); }, &[0x4e, 0x8b, 0x4c, 0x25, 0x00]),
        ("mov r9, qword ptr [rbp + r13*8]", |a| { a.ld_ext(R9, mem_idx(RBP, R13, 3, 0), 8, false); }, &[0x4e, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [r12 + rax*1]", |a| { a.ld_ext(R9, mem_idx(R12, RAX, 0, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0x04]),
        ("mov r9, qword ptr [r12 + rcx*8]", |a| { a.ld_ext(R9, mem_idx(R12, RCX, 3, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0xcc]),
        ("mov r9, qword ptr [r12 + rsi*1]", |a| { a.ld_ext(R9, mem_idx(R12, RSI, 0, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0x34]),
        ("mov r9, qword ptr [r12 + r8*8]", |a| { a.ld_ext(R9, mem_idx(R12, R8, 3, 0), 8, false); }, &[0x4f, 0x8b, 0x0c, 0xc4]),
        ("mov r9, qword ptr [r12 + r13*1]", |a| { a.ld_ext(R9, mem_idx(R12, R13, 0, 0), 8, false); }, &[0x4f, 0x8b, 0x0c, 0x2c]),
        ("mov r9, qword ptr [r12 + r15*8]", |a| { a.ld_ext(R9, mem_idx(R12, R15, 3, 0), 8, false); }, &[0x4f, 0x8b, 0x0c, 0xfc]),
        ("mov r9, qword ptr [r13 + rcx*1]", |a| { a.ld_ext(R9, mem_idx(R13, RCX, 0, 0), 8, false); }, &[0x4d, 0x8b, 0x4c, 0x0d, 0x00]),
        ("mov r9, qword ptr [r13 + rbp*8]", |a| { a.ld_ext(R9, mem_idx(R13, RBP, 3, 0), 8, false); }, &[0x4d, 0x8b, 0x4c, 0xed, 0x00]),
        ("mov r9, qword ptr [r13 + r8*1]", |a| { a.ld_ext(R9, mem_idx(R13, R8, 0, 0), 8, false); }, &[0x4f, 0x8b, 0x4c, 0x05, 0x00]),
        ("mov r9, qword ptr [r13 + r12*8]", |a| { a.ld_ext(R9, mem_idx(R13, R12, 3, 0), 8, false); }, &[0x4f, 0x8b, 0x4c, 0xe5, 0x00]),
        ("mov r9, qword ptr [r13 + r15*1]", |a| { a.ld_ext(R9, mem_idx(R13, R15, 0, 0), 8, false); }, &[0x4f, 0x8b, 0x4c, 0x3d, 0x00]),
        ("mov r9, qword ptr [r15 + rax*8]", |a| { a.ld_ext(R9, mem_idx(R15, RAX, 3, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0xc7]),
        ("mov r9, qword ptr [r15 + rbp*1]", |a| { a.ld_ext(R9, mem_idx(R15, RBP, 0, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0x2f]),
        ("mov r9, qword ptr [r15 + rsi*8]", |a| { a.ld_ext(R9, mem_idx(R15, RSI, 3, 0), 8, false); }, &[0x4d, 0x8b, 0x0c, 0xf7]),
        ("mov r9, qword ptr [r15 + r12*1]", |a| { a.ld_ext(R9, mem_idx(R15, R12, 0, 0), 8, false); }, &[0x4f, 0x8b, 0x0c, 0x27]),
        ("mov r9, qword ptr [r15 + r13*8]", |a| { a.ld_ext(R9, mem_idx(R15, R13, 3, 0), 8, false); }, &[0x4f, 0x8b, 0x0c, 0xef]),
        ("mov r9, qword ptr [rbx + rcx*2 + 32]", |a| { a.ld_ext(R9, mem_idx(RBX, RCX, 1, 32), 8, false); }, &[0x4c, 0x8b, 0x4c, 0x4b, 0x20]),
    ];
    check_encodings(cases);
}

    /// Stores of every size from registers including spl/bpl/sil/dil, which need a REX prefix.
#[rustfmt::skip]
#[test]
fn stores_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("mov byte ptr [rax], al", |a| { a.st(mem(RAX, 0), RAX, 1); }, &[0x88, 0x00]),
        ("mov byte ptr [rax - 305419896], al", |a| { a.st(mem(RAX, -305419896), RAX, 1); }, &[0x88, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], al", |a| { a.st(mem(RSP, 305419896), RAX, 1); }, &[0x88, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], al", |a| { a.st(mem(RBP, -129), RAX, 1); }, &[0x88, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], al", |a| { a.st(mem(R8, -128), RAX, 1); }, &[0x41, 0x88, 0x40, 0x80]),
        ("mov byte ptr [r12 + 128], al", |a| { a.st(mem(R12, 128), RAX, 1); }, &[0x41, 0x88, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], al", |a| { a.st(mem(R13, 127), RAX, 1); }, &[0x41, 0x88, 0x45, 0x7f]),
        ("mov byte ptr [r15 - 1], al", |a| { a.st(mem(R15, -1), RAX, 1); }, &[0x41, 0x88, 0x47, 0xff]),
        ("mov byte ptr [rbx + 1], al", |a| { a.st(mem(RBX, 1), RAX, 1); }, &[0x88, 0x43, 0x01]),
        ("mov byte ptr [rax + rax*1], al", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RAX, 1); }, &[0x88, 0x04, 0x00]),
        ("mov byte ptr [rax + rsi*1], al", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RAX, 1); }, &[0x88, 0x04, 0x30]),
        ("mov byte ptr [rax + r13*1], al", |a| { a.st(mem_idx(RAX, R13, 0, 0), RAX, 1); }, &[0x42, 0x88, 0x04, 0x28]),
        ("mov byte ptr [rsp + rcx*1], al", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RAX, 1); }, &[0x88, 0x04, 0x0c]),
        ("mov byte ptr [rsp + r8*1], al", |a| { a.st(mem_idx(RSP, R8, 0, 0), RAX, 1); }, &[0x42, 0x88, 0x04, 0x04]),
        ("mov byte ptr [rsp + r15*1], al", |a| { a.st(mem_idx(RSP, R15, 0, 0), RAX, 1); }, &[0x42, 0x88, 0x04, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], al", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RAX, 1); }, &[0x88, 0x44, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], al", |a| { a.st(mem_idx(RBP, R12, 0, 0), RAX, 1); }, &[0x42, 0x88, 0x44, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], al", |a| { a.st(mem_idx(R12, RAX, 0, 0), RAX, 1); }, &[0x41, 0x88, 0x04, 0x04]),
        ("mov byte ptr [r12 + rsi*1], al", |a| { a.st(mem_idx(R12, RSI, 0, 0), RAX, 1); }, &[0x41, 0x88, 0x04, 0x34]),
        ("mov byte ptr [r12 + r13*1], al", |a| { a.st(mem_idx(R12, R13, 0, 0), RAX, 1); }, &[0x43, 0x88, 0x04, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], al", |a| { a.st(mem_idx(R13, RCX, 0, 0), RAX, 1); }, &[0x41, 0x88, 0x44, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], al", |a| { a.st(mem_idx(R13, R8, 0, 0), RAX, 1); }, &[0x43, 0x88, 0x44, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], al", |a| { a.st(mem_idx(R13, R15, 0, 0), RAX, 1); }, &[0x43, 0x88, 0x44, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], al", |a| { a.st(mem_idx(R15, RBP, 0, 0), RAX, 1); }, &[0x41, 0x88, 0x04, 0x2f]),
        ("mov byte ptr [r15 + r12*1], al", |a| { a.st(mem_idx(R15, R12, 0, 0), RAX, 1); }, &[0x43, 0x88, 0x04, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], al", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RAX, 1); }, &[0x88, 0x44, 0x4b, 0x20]),
        ("mov byte ptr [rax], sil", |a| { a.st(mem(RAX, 0), RSI, 1); }, &[0x40, 0x88, 0x30]),
        ("mov byte ptr [rax - 305419896], sil", |a| { a.st(mem(RAX, -305419896), RSI, 1); }, &[0x40, 0x88, 0xb0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], sil", |a| { a.st(mem(RSP, 305419896), RSI, 1); }, &[0x40, 0x88, 0xb4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], sil", |a| { a.st(mem(RBP, -129), RSI, 1); }, &[0x40, 0x88, 0xb5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], sil", |a| { a.st(mem(R8, -128), RSI, 1); }, &[0x41, 0x88, 0x70, 0x80]),
        ("mov byte ptr [r12 + 128], sil", |a| { a.st(mem(R12, 128), RSI, 1); }, &[0x41, 0x88, 0xb4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], sil", |a| { a.st(mem(R13, 127), RSI, 1); }, &[0x41, 0x88, 0x75, 0x7f]),
        ("mov byte ptr [r15 - 1], sil", |a| { a.st(mem(R15, -1), RSI, 1); }, &[0x41, 0x88, 0x77, 0xff]),
        ("mov byte ptr [rbx + 1], sil", |a| { a.st(mem(RBX, 1), RSI, 1); }, &[0x40, 0x88, 0x73, 0x01]),
        ("mov byte ptr [rax + rax*1], sil", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSI, 1); }, &[0x40, 0x88, 0x34, 0x00]),
        ("mov byte ptr [rax + rsi*1], sil", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSI, 1); }, &[0x40, 0x88, 0x34, 0x30]),
        ("mov byte ptr [rax + r13*1], sil", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSI, 1); }, &[0x42, 0x88, 0x34, 0x28]),
        ("mov byte ptr [rsp + rcx*1], sil", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSI, 1); }, &[0x40, 0x88, 0x34, 0x0c]),
        ("mov byte ptr [rsp + r8*1], sil", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSI, 1); }, &[0x42, 0x88, 0x34, 0x04]),
        ("mov byte ptr [rsp + r15*1], sil", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSI, 1); }, &[0x42, 0x88, 0x34, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], sil", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSI, 1); }, &[0x40, 0x88, 0x74, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], sil", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSI, 1); }, &[0x42, 0x88, 0x74, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], sil", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSI, 1); }, &[0x41, 0x88, 0x34, 0x04]),
        ("mov byte ptr [r12 + rsi*1], sil", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSI, 1); }, &[0x41, 0x88, 0x34, 0x34]),
        ("mov byte ptr [r12 + r13*1], sil", |a| { a.st(mem_idx(R12, R13, 0, 0), RSI, 1); }, &[0x43, 0x88, 0x34, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], sil", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSI, 1); }, &[0x41, 0x88, 0x74, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], sil", |a| { a.st(mem_idx(R13, R8, 0, 0), RSI, 1); }, &[0x43, 0x88, 0x74, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], sil", |a| { a.st(mem_idx(R13, R15, 0, 0), RSI, 1); }, &[0x43, 0x88, 0x74, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], sil", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSI, 1); }, &[0x41, 0x88, 0x34, 0x2f]),
        ("mov byte ptr [r15 + r12*1], sil", |a| { a.st(mem_idx(R15, R12, 0, 0), RSI, 1); }, &[0x43, 0x88, 0x34, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], sil", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSI, 1); }, &[0x40, 0x88, 0x74, 0x4b, 0x20]),
        ("mov byte ptr [rax], dil", |a| { a.st(mem(RAX, 0), RDI, 1); }, &[0x40, 0x88, 0x38]),
        ("mov byte ptr [rax - 305419896], dil", |a| { a.st(mem(RAX, -305419896), RDI, 1); }, &[0x40, 0x88, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], dil", |a| { a.st(mem(RSP, 305419896), RDI, 1); }, &[0x40, 0x88, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], dil", |a| { a.st(mem(RBP, -129), RDI, 1); }, &[0x40, 0x88, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], dil", |a| { a.st(mem(R8, -128), RDI, 1); }, &[0x41, 0x88, 0x78, 0x80]),
        ("mov byte ptr [r12 + 128], dil", |a| { a.st(mem(R12, 128), RDI, 1); }, &[0x41, 0x88, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], dil", |a| { a.st(mem(R13, 127), RDI, 1); }, &[0x41, 0x88, 0x7d, 0x7f]),
        ("mov byte ptr [r15 - 1], dil", |a| { a.st(mem(R15, -1), RDI, 1); }, &[0x41, 0x88, 0x7f, 0xff]),
        ("mov byte ptr [rbx + 1], dil", |a| { a.st(mem(RBX, 1), RDI, 1); }, &[0x40, 0x88, 0x7b, 0x01]),
        ("mov byte ptr [rax + rax*1], dil", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RDI, 1); }, &[0x40, 0x88, 0x3c, 0x00]),
        ("mov byte ptr [rax + rsi*1], dil", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RDI, 1); }, &[0x40, 0x88, 0x3c, 0x30]),
        ("mov byte ptr [rax + r13*1], dil", |a| { a.st(mem_idx(RAX, R13, 0, 0), RDI, 1); }, &[0x42, 0x88, 0x3c, 0x28]),
        ("mov byte ptr [rsp + rcx*1], dil", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RDI, 1); }, &[0x40, 0x88, 0x3c, 0x0c]),
        ("mov byte ptr [rsp + r8*1], dil", |a| { a.st(mem_idx(RSP, R8, 0, 0), RDI, 1); }, &[0x42, 0x88, 0x3c, 0x04]),
        ("mov byte ptr [rsp + r15*1], dil", |a| { a.st(mem_idx(RSP, R15, 0, 0), RDI, 1); }, &[0x42, 0x88, 0x3c, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], dil", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RDI, 1); }, &[0x40, 0x88, 0x7c, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], dil", |a| { a.st(mem_idx(RBP, R12, 0, 0), RDI, 1); }, &[0x42, 0x88, 0x7c, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], dil", |a| { a.st(mem_idx(R12, RAX, 0, 0), RDI, 1); }, &[0x41, 0x88, 0x3c, 0x04]),
        ("mov byte ptr [r12 + rsi*1], dil", |a| { a.st(mem_idx(R12, RSI, 0, 0), RDI, 1); }, &[0x41, 0x88, 0x3c, 0x34]),
        ("mov byte ptr [r12 + r13*1], dil", |a| { a.st(mem_idx(R12, R13, 0, 0), RDI, 1); }, &[0x43, 0x88, 0x3c, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], dil", |a| { a.st(mem_idx(R13, RCX, 0, 0), RDI, 1); }, &[0x41, 0x88, 0x7c, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], dil", |a| { a.st(mem_idx(R13, R8, 0, 0), RDI, 1); }, &[0x43, 0x88, 0x7c, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], dil", |a| { a.st(mem_idx(R13, R15, 0, 0), RDI, 1); }, &[0x43, 0x88, 0x7c, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], dil", |a| { a.st(mem_idx(R15, RBP, 0, 0), RDI, 1); }, &[0x41, 0x88, 0x3c, 0x2f]),
        ("mov byte ptr [r15 + r12*1], dil", |a| { a.st(mem_idx(R15, R12, 0, 0), RDI, 1); }, &[0x43, 0x88, 0x3c, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], dil", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RDI, 1); }, &[0x40, 0x88, 0x7c, 0x4b, 0x20]),
        ("mov byte ptr [rax], spl", |a| { a.st(mem(RAX, 0), RSP, 1); }, &[0x40, 0x88, 0x20]),
        ("mov byte ptr [rax - 305419896], spl", |a| { a.st(mem(RAX, -305419896), RSP, 1); }, &[0x40, 0x88, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], spl", |a| { a.st(mem(RSP, 305419896), RSP, 1); }, &[0x40, 0x88, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], spl", |a| { a.st(mem(RBP, -129), RSP, 1); }, &[0x40, 0x88, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], spl", |a| { a.st(mem(R8, -128), RSP, 1); }, &[0x41, 0x88, 0x60, 0x80]),
        ("mov byte ptr [r12 + 128], spl", |a| { a.st(mem(R12, 128), RSP, 1); }, &[0x41, 0x88, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], spl", |a| { a.st(mem(R13, 127), RSP, 1); }, &[0x41, 0x88, 0x65, 0x7f]),
        ("mov byte ptr [r15 - 1], spl", |a| { a.st(mem(R15, -1), RSP, 1); }, &[0x41, 0x88, 0x67, 0xff]),
        ("mov byte ptr [rbx + 1], spl", |a| { a.st(mem(RBX, 1), RSP, 1); }, &[0x40, 0x88, 0x63, 0x01]),
        ("mov byte ptr [rax + rax*1], spl", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSP, 1); }, &[0x40, 0x88, 0x24, 0x00]),
        ("mov byte ptr [rax + rsi*1], spl", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSP, 1); }, &[0x40, 0x88, 0x24, 0x30]),
        ("mov byte ptr [rax + r13*1], spl", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSP, 1); }, &[0x42, 0x88, 0x24, 0x28]),
        ("mov byte ptr [rsp + rcx*1], spl", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSP, 1); }, &[0x40, 0x88, 0x24, 0x0c]),
        ("mov byte ptr [rsp + r8*1], spl", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSP, 1); }, &[0x42, 0x88, 0x24, 0x04]),
        ("mov byte ptr [rsp + r15*1], spl", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSP, 1); }, &[0x42, 0x88, 0x24, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], spl", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSP, 1); }, &[0x40, 0x88, 0x64, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], spl", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSP, 1); }, &[0x42, 0x88, 0x64, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], spl", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSP, 1); }, &[0x41, 0x88, 0x24, 0x04]),
        ("mov byte ptr [r12 + rsi*1], spl", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSP, 1); }, &[0x41, 0x88, 0x24, 0x34]),
        ("mov byte ptr [r12 + r13*1], spl", |a| { a.st(mem_idx(R12, R13, 0, 0), RSP, 1); }, &[0x43, 0x88, 0x24, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], spl", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSP, 1); }, &[0x41, 0x88, 0x64, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], spl", |a| { a.st(mem_idx(R13, R8, 0, 0), RSP, 1); }, &[0x43, 0x88, 0x64, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], spl", |a| { a.st(mem_idx(R13, R15, 0, 0), RSP, 1); }, &[0x43, 0x88, 0x64, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], spl", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSP, 1); }, &[0x41, 0x88, 0x24, 0x2f]),
        ("mov byte ptr [r15 + r12*1], spl", |a| { a.st(mem_idx(R15, R12, 0, 0), RSP, 1); }, &[0x43, 0x88, 0x24, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], spl", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSP, 1); }, &[0x40, 0x88, 0x64, 0x4b, 0x20]),
        ("mov byte ptr [rax], bpl", |a| { a.st(mem(RAX, 0), RBP, 1); }, &[0x40, 0x88, 0x28]),
        ("mov byte ptr [rax - 305419896], bpl", |a| { a.st(mem(RAX, -305419896), RBP, 1); }, &[0x40, 0x88, 0xa8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], bpl", |a| { a.st(mem(RSP, 305419896), RBP, 1); }, &[0x40, 0x88, 0xac, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], bpl", |a| { a.st(mem(RBP, -129), RBP, 1); }, &[0x40, 0x88, 0xad, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], bpl", |a| { a.st(mem(R8, -128), RBP, 1); }, &[0x41, 0x88, 0x68, 0x80]),
        ("mov byte ptr [r12 + 128], bpl", |a| { a.st(mem(R12, 128), RBP, 1); }, &[0x41, 0x88, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], bpl", |a| { a.st(mem(R13, 127), RBP, 1); }, &[0x41, 0x88, 0x6d, 0x7f]),
        ("mov byte ptr [r15 - 1], bpl", |a| { a.st(mem(R15, -1), RBP, 1); }, &[0x41, 0x88, 0x6f, 0xff]),
        ("mov byte ptr [rbx + 1], bpl", |a| { a.st(mem(RBX, 1), RBP, 1); }, &[0x40, 0x88, 0x6b, 0x01]),
        ("mov byte ptr [rax + rax*1], bpl", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RBP, 1); }, &[0x40, 0x88, 0x2c, 0x00]),
        ("mov byte ptr [rax + rsi*1], bpl", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RBP, 1); }, &[0x40, 0x88, 0x2c, 0x30]),
        ("mov byte ptr [rax + r13*1], bpl", |a| { a.st(mem_idx(RAX, R13, 0, 0), RBP, 1); }, &[0x42, 0x88, 0x2c, 0x28]),
        ("mov byte ptr [rsp + rcx*1], bpl", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RBP, 1); }, &[0x40, 0x88, 0x2c, 0x0c]),
        ("mov byte ptr [rsp + r8*1], bpl", |a| { a.st(mem_idx(RSP, R8, 0, 0), RBP, 1); }, &[0x42, 0x88, 0x2c, 0x04]),
        ("mov byte ptr [rsp + r15*1], bpl", |a| { a.st(mem_idx(RSP, R15, 0, 0), RBP, 1); }, &[0x42, 0x88, 0x2c, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], bpl", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RBP, 1); }, &[0x40, 0x88, 0x6c, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], bpl", |a| { a.st(mem_idx(RBP, R12, 0, 0), RBP, 1); }, &[0x42, 0x88, 0x6c, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], bpl", |a| { a.st(mem_idx(R12, RAX, 0, 0), RBP, 1); }, &[0x41, 0x88, 0x2c, 0x04]),
        ("mov byte ptr [r12 + rsi*1], bpl", |a| { a.st(mem_idx(R12, RSI, 0, 0), RBP, 1); }, &[0x41, 0x88, 0x2c, 0x34]),
        ("mov byte ptr [r12 + r13*1], bpl", |a| { a.st(mem_idx(R12, R13, 0, 0), RBP, 1); }, &[0x43, 0x88, 0x2c, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], bpl", |a| { a.st(mem_idx(R13, RCX, 0, 0), RBP, 1); }, &[0x41, 0x88, 0x6c, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], bpl", |a| { a.st(mem_idx(R13, R8, 0, 0), RBP, 1); }, &[0x43, 0x88, 0x6c, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], bpl", |a| { a.st(mem_idx(R13, R15, 0, 0), RBP, 1); }, &[0x43, 0x88, 0x6c, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], bpl", |a| { a.st(mem_idx(R15, RBP, 0, 0), RBP, 1); }, &[0x41, 0x88, 0x2c, 0x2f]),
        ("mov byte ptr [r15 + r12*1], bpl", |a| { a.st(mem_idx(R15, R12, 0, 0), RBP, 1); }, &[0x43, 0x88, 0x2c, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], bpl", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RBP, 1); }, &[0x40, 0x88, 0x6c, 0x4b, 0x20]),
        ("mov byte ptr [rax], r8b", |a| { a.st(mem(RAX, 0), R8, 1); }, &[0x44, 0x88, 0x00]),
        ("mov byte ptr [rax - 305419896], r8b", |a| { a.st(mem(RAX, -305419896), R8, 1); }, &[0x44, 0x88, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], r8b", |a| { a.st(mem(RSP, 305419896), R8, 1); }, &[0x44, 0x88, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], r8b", |a| { a.st(mem(RBP, -129), R8, 1); }, &[0x44, 0x88, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], r8b", |a| { a.st(mem(R8, -128), R8, 1); }, &[0x45, 0x88, 0x40, 0x80]),
        ("mov byte ptr [r12 + 128], r8b", |a| { a.st(mem(R12, 128), R8, 1); }, &[0x45, 0x88, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], r8b", |a| { a.st(mem(R13, 127), R8, 1); }, &[0x45, 0x88, 0x45, 0x7f]),
        ("mov byte ptr [r15 - 1], r8b", |a| { a.st(mem(R15, -1), R8, 1); }, &[0x45, 0x88, 0x47, 0xff]),
        ("mov byte ptr [rbx + 1], r8b", |a| { a.st(mem(RBX, 1), R8, 1); }, &[0x44, 0x88, 0x43, 0x01]),
        ("mov byte ptr [rax + rax*1], r8b", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R8, 1); }, &[0x44, 0x88, 0x04, 0x00]),
        ("mov byte ptr [rax + rsi*1], r8b", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R8, 1); }, &[0x44, 0x88, 0x04, 0x30]),
        ("mov byte ptr [rax + r13*1], r8b", |a| { a.st(mem_idx(RAX, R13, 0, 0), R8, 1); }, &[0x46, 0x88, 0x04, 0x28]),
        ("mov byte ptr [rsp + rcx*1], r8b", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R8, 1); }, &[0x44, 0x88, 0x04, 0x0c]),
        ("mov byte ptr [rsp + r8*1], r8b", |a| { a.st(mem_idx(RSP, R8, 0, 0), R8, 1); }, &[0x46, 0x88, 0x04, 0x04]),
        ("mov byte ptr [rsp + r15*1], r8b", |a| { a.st(mem_idx(RSP, R15, 0, 0), R8, 1); }, &[0x46, 0x88, 0x04, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], r8b", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R8, 1); }, &[0x44, 0x88, 0x44, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], r8b", |a| { a.st(mem_idx(RBP, R12, 0, 0), R8, 1); }, &[0x46, 0x88, 0x44, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], r8b", |a| { a.st(mem_idx(R12, RAX, 0, 0), R8, 1); }, &[0x45, 0x88, 0x04, 0x04]),
        ("mov byte ptr [r12 + rsi*1], r8b", |a| { a.st(mem_idx(R12, RSI, 0, 0), R8, 1); }, &[0x45, 0x88, 0x04, 0x34]),
        ("mov byte ptr [r12 + r13*1], r8b", |a| { a.st(mem_idx(R12, R13, 0, 0), R8, 1); }, &[0x47, 0x88, 0x04, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], r8b", |a| { a.st(mem_idx(R13, RCX, 0, 0), R8, 1); }, &[0x45, 0x88, 0x44, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], r8b", |a| { a.st(mem_idx(R13, R8, 0, 0), R8, 1); }, &[0x47, 0x88, 0x44, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], r8b", |a| { a.st(mem_idx(R13, R15, 0, 0), R8, 1); }, &[0x47, 0x88, 0x44, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], r8b", |a| { a.st(mem_idx(R15, RBP, 0, 0), R8, 1); }, &[0x45, 0x88, 0x04, 0x2f]),
        ("mov byte ptr [r15 + r12*1], r8b", |a| { a.st(mem_idx(R15, R12, 0, 0), R8, 1); }, &[0x47, 0x88, 0x04, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], r8b", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R8, 1); }, &[0x44, 0x88, 0x44, 0x4b, 0x20]),
        ("mov byte ptr [rax], r15b", |a| { a.st(mem(RAX, 0), R15, 1); }, &[0x44, 0x88, 0x38]),
        ("mov byte ptr [rax - 305419896], r15b", |a| { a.st(mem(RAX, -305419896), R15, 1); }, &[0x44, 0x88, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], r15b", |a| { a.st(mem(RSP, 305419896), R15, 1); }, &[0x44, 0x88, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], r15b", |a| { a.st(mem(RBP, -129), R15, 1); }, &[0x44, 0x88, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], r15b", |a| { a.st(mem(R8, -128), R15, 1); }, &[0x45, 0x88, 0x78, 0x80]),
        ("mov byte ptr [r12 + 128], r15b", |a| { a.st(mem(R12, 128), R15, 1); }, &[0x45, 0x88, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], r15b", |a| { a.st(mem(R13, 127), R15, 1); }, &[0x45, 0x88, 0x7d, 0x7f]),
        ("mov byte ptr [r15 - 1], r15b", |a| { a.st(mem(R15, -1), R15, 1); }, &[0x45, 0x88, 0x7f, 0xff]),
        ("mov byte ptr [rbx + 1], r15b", |a| { a.st(mem(RBX, 1), R15, 1); }, &[0x44, 0x88, 0x7b, 0x01]),
        ("mov byte ptr [rax + rax*1], r15b", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R15, 1); }, &[0x44, 0x88, 0x3c, 0x00]),
        ("mov byte ptr [rax + rsi*1], r15b", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R15, 1); }, &[0x44, 0x88, 0x3c, 0x30]),
        ("mov byte ptr [rax + r13*1], r15b", |a| { a.st(mem_idx(RAX, R13, 0, 0), R15, 1); }, &[0x46, 0x88, 0x3c, 0x28]),
        ("mov byte ptr [rsp + rcx*1], r15b", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R15, 1); }, &[0x44, 0x88, 0x3c, 0x0c]),
        ("mov byte ptr [rsp + r8*1], r15b", |a| { a.st(mem_idx(RSP, R8, 0, 0), R15, 1); }, &[0x46, 0x88, 0x3c, 0x04]),
        ("mov byte ptr [rsp + r15*1], r15b", |a| { a.st(mem_idx(RSP, R15, 0, 0), R15, 1); }, &[0x46, 0x88, 0x3c, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], r15b", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R15, 1); }, &[0x44, 0x88, 0x7c, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], r15b", |a| { a.st(mem_idx(RBP, R12, 0, 0), R15, 1); }, &[0x46, 0x88, 0x7c, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], r15b", |a| { a.st(mem_idx(R12, RAX, 0, 0), R15, 1); }, &[0x45, 0x88, 0x3c, 0x04]),
        ("mov byte ptr [r12 + rsi*1], r15b", |a| { a.st(mem_idx(R12, RSI, 0, 0), R15, 1); }, &[0x45, 0x88, 0x3c, 0x34]),
        ("mov byte ptr [r12 + r13*1], r15b", |a| { a.st(mem_idx(R12, R13, 0, 0), R15, 1); }, &[0x47, 0x88, 0x3c, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], r15b", |a| { a.st(mem_idx(R13, RCX, 0, 0), R15, 1); }, &[0x45, 0x88, 0x7c, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], r15b", |a| { a.st(mem_idx(R13, R8, 0, 0), R15, 1); }, &[0x47, 0x88, 0x7c, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], r15b", |a| { a.st(mem_idx(R13, R15, 0, 0), R15, 1); }, &[0x47, 0x88, 0x7c, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], r15b", |a| { a.st(mem_idx(R15, RBP, 0, 0), R15, 1); }, &[0x45, 0x88, 0x3c, 0x2f]),
        ("mov byte ptr [r15 + r12*1], r15b", |a| { a.st(mem_idx(R15, R12, 0, 0), R15, 1); }, &[0x47, 0x88, 0x3c, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], r15b", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R15, 1); }, &[0x44, 0x88, 0x7c, 0x4b, 0x20]),
        ("mov byte ptr [rax], r12b", |a| { a.st(mem(RAX, 0), R12, 1); }, &[0x44, 0x88, 0x20]),
        ("mov byte ptr [rax - 305419896], r12b", |a| { a.st(mem(RAX, -305419896), R12, 1); }, &[0x44, 0x88, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov byte ptr [rsp + 305419896], r12b", |a| { a.st(mem(RSP, 305419896), R12, 1); }, &[0x44, 0x88, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov byte ptr [rbp - 129], r12b", |a| { a.st(mem(RBP, -129), R12, 1); }, &[0x44, 0x88, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov byte ptr [r8 - 128], r12b", |a| { a.st(mem(R8, -128), R12, 1); }, &[0x45, 0x88, 0x60, 0x80]),
        ("mov byte ptr [r12 + 128], r12b", |a| { a.st(mem(R12, 128), R12, 1); }, &[0x45, 0x88, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov byte ptr [r13 + 127], r12b", |a| { a.st(mem(R13, 127), R12, 1); }, &[0x45, 0x88, 0x65, 0x7f]),
        ("mov byte ptr [r15 - 1], r12b", |a| { a.st(mem(R15, -1), R12, 1); }, &[0x45, 0x88, 0x67, 0xff]),
        ("mov byte ptr [rbx + 1], r12b", |a| { a.st(mem(RBX, 1), R12, 1); }, &[0x44, 0x88, 0x63, 0x01]),
        ("mov byte ptr [rax + rax*1], r12b", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R12, 1); }, &[0x44, 0x88, 0x24, 0x00]),
        ("mov byte ptr [rax + rsi*1], r12b", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R12, 1); }, &[0x44, 0x88, 0x24, 0x30]),
        ("mov byte ptr [rax + r13*1], r12b", |a| { a.st(mem_idx(RAX, R13, 0, 0), R12, 1); }, &[0x46, 0x88, 0x24, 0x28]),
        ("mov byte ptr [rsp + rcx*1], r12b", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R12, 1); }, &[0x44, 0x88, 0x24, 0x0c]),
        ("mov byte ptr [rsp + r8*1], r12b", |a| { a.st(mem_idx(RSP, R8, 0, 0), R12, 1); }, &[0x46, 0x88, 0x24, 0x04]),
        ("mov byte ptr [rsp + r15*1], r12b", |a| { a.st(mem_idx(RSP, R15, 0, 0), R12, 1); }, &[0x46, 0x88, 0x24, 0x3c]),
        ("mov byte ptr [rbp + rbp*1], r12b", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R12, 1); }, &[0x44, 0x88, 0x64, 0x2d, 0x00]),
        ("mov byte ptr [rbp + r12*1], r12b", |a| { a.st(mem_idx(RBP, R12, 0, 0), R12, 1); }, &[0x46, 0x88, 0x64, 0x25, 0x00]),
        ("mov byte ptr [r12 + rax*1], r12b", |a| { a.st(mem_idx(R12, RAX, 0, 0), R12, 1); }, &[0x45, 0x88, 0x24, 0x04]),
        ("mov byte ptr [r12 + rsi*1], r12b", |a| { a.st(mem_idx(R12, RSI, 0, 0), R12, 1); }, &[0x45, 0x88, 0x24, 0x34]),
        ("mov byte ptr [r12 + r13*1], r12b", |a| { a.st(mem_idx(R12, R13, 0, 0), R12, 1); }, &[0x47, 0x88, 0x24, 0x2c]),
        ("mov byte ptr [r13 + rcx*1], r12b", |a| { a.st(mem_idx(R13, RCX, 0, 0), R12, 1); }, &[0x45, 0x88, 0x64, 0x0d, 0x00]),
        ("mov byte ptr [r13 + r8*1], r12b", |a| { a.st(mem_idx(R13, R8, 0, 0), R12, 1); }, &[0x47, 0x88, 0x64, 0x05, 0x00]),
        ("mov byte ptr [r13 + r15*1], r12b", |a| { a.st(mem_idx(R13, R15, 0, 0), R12, 1); }, &[0x47, 0x88, 0x64, 0x3d, 0x00]),
        ("mov byte ptr [r15 + rbp*1], r12b", |a| { a.st(mem_idx(R15, RBP, 0, 0), R12, 1); }, &[0x45, 0x88, 0x24, 0x2f]),
        ("mov byte ptr [r15 + r12*1], r12b", |a| { a.st(mem_idx(R15, R12, 0, 0), R12, 1); }, &[0x47, 0x88, 0x24, 0x27]),
        ("mov byte ptr [rbx + rcx*2 + 32], r12b", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R12, 1); }, &[0x44, 0x88, 0x64, 0x4b, 0x20]),
        ("mov word ptr [rax], ax", |a| { a.st(mem(RAX, 0), RAX, 2); }, &[0x66, 0x89, 0x00]),
        ("mov word ptr [rax - 305419896], ax", |a| { a.st(mem(RAX, -305419896), RAX, 2); }, &[0x66, 0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], ax", |a| { a.st(mem(RSP, 305419896), RAX, 2); }, &[0x66, 0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], ax", |a| { a.st(mem(RBP, -129), RAX, 2); }, &[0x66, 0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], ax", |a| { a.st(mem(R8, -128), RAX, 2); }, &[0x66, 0x41, 0x89, 0x40, 0x80]),
        ("mov word ptr [r12 + 128], ax", |a| { a.st(mem(R12, 128), RAX, 2); }, &[0x66, 0x41, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], ax", |a| { a.st(mem(R13, 127), RAX, 2); }, &[0x66, 0x41, 0x89, 0x45, 0x7f]),
        ("mov word ptr [r15 - 1], ax", |a| { a.st(mem(R15, -1), RAX, 2); }, &[0x66, 0x41, 0x89, 0x47, 0xff]),
        ("mov word ptr [rbx + 1], ax", |a| { a.st(mem(RBX, 1), RAX, 2); }, &[0x66, 0x89, 0x43, 0x01]),
        ("mov word ptr [rax + rax*1], ax", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RAX, 2); }, &[0x66, 0x89, 0x04, 0x00]),
        ("mov word ptr [rax + rsi*1], ax", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RAX, 2); }, &[0x66, 0x89, 0x04, 0x30]),
        ("mov word ptr [rax + r13*1], ax", |a| { a.st(mem_idx(RAX, R13, 0, 0), RAX, 2); }, &[0x66, 0x42, 0x89, 0x04, 0x28]),
        ("mov word ptr [rsp + rcx*1], ax", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RAX, 2); }, &[0x66, 0x89, 0x04, 0x0c]),
        ("mov word ptr [rsp + r8*1], ax", |a| { a.st(mem_idx(RSP, R8, 0, 0), RAX, 2); }, &[0x66, 0x42, 0x89, 0x04, 0x04]),
        ("mov word ptr [rsp + r15*1], ax", |a| { a.st(mem_idx(RSP, R15, 0, 0), RAX, 2); }, &[0x66, 0x42, 0x89, 0x04, 0x3c]),
        ("mov word ptr [rbp + rbp*1], ax", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RAX, 2); }, &[0x66, 0x89, 0x44, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], ax", |a| { a.st(mem_idx(RBP, R12, 0, 0), RAX, 2); }, &[0x66, 0x42, 0x89, 0x44, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], ax", |a| { a.st(mem_idx(R12, RAX, 0, 0), RAX, 2); }, &[0x66, 0x41, 0x89, 0x04, 0x04]),
        ("mov word ptr [r12 + rsi*1], ax", |a| { a.st(mem_idx(R12, RSI, 0, 0), RAX, 2); }, &[0x66, 0x41, 0x89, 0x04, 0x34]),
        ("mov word ptr [r12 + r13*1], ax", |a| { a.st(mem_idx(R12, R13, 0, 0), RAX, 2); }, &[0x66, 0x43, 0x89, 0x04, 0x2c]),
        ("mov word ptr [r13 + rcx*1], ax", |a| { a.st(mem_idx(R13, RCX, 0, 0), RAX, 2); }, &[0x66, 0x41, 0x89, 0x44, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], ax", |a| { a.st(mem_idx(R13, R8, 0, 0), RAX, 2); }, &[0x66, 0x43, 0x89, 0x44, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], ax", |a| { a.st(mem_idx(R13, R15, 0, 0), RAX, 2); }, &[0x66, 0x43, 0x89, 0x44, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], ax", |a| { a.st(mem_idx(R15, RBP, 0, 0), RAX, 2); }, &[0x66, 0x41, 0x89, 0x04, 0x2f]),
        ("mov word ptr [r15 + r12*1], ax", |a| { a.st(mem_idx(R15, R12, 0, 0), RAX, 2); }, &[0x66, 0x43, 0x89, 0x04, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], ax", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RAX, 2); }, &[0x66, 0x89, 0x44, 0x4b, 0x20]),
        ("mov word ptr [rax], si", |a| { a.st(mem(RAX, 0), RSI, 2); }, &[0x66, 0x89, 0x30]),
        ("mov word ptr [rax - 305419896], si", |a| { a.st(mem(RAX, -305419896), RSI, 2); }, &[0x66, 0x89, 0xb0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], si", |a| { a.st(mem(RSP, 305419896), RSI, 2); }, &[0x66, 0x89, 0xb4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], si", |a| { a.st(mem(RBP, -129), RSI, 2); }, &[0x66, 0x89, 0xb5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], si", |a| { a.st(mem(R8, -128), RSI, 2); }, &[0x66, 0x41, 0x89, 0x70, 0x80]),
        ("mov word ptr [r12 + 128], si", |a| { a.st(mem(R12, 128), RSI, 2); }, &[0x66, 0x41, 0x89, 0xb4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], si", |a| { a.st(mem(R13, 127), RSI, 2); }, &[0x66, 0x41, 0x89, 0x75, 0x7f]),
        ("mov word ptr [r15 - 1], si", |a| { a.st(mem(R15, -1), RSI, 2); }, &[0x66, 0x41, 0x89, 0x77, 0xff]),
        ("mov word ptr [rbx + 1], si", |a| { a.st(mem(RBX, 1), RSI, 2); }, &[0x66, 0x89, 0x73, 0x01]),
        ("mov word ptr [rax + rax*1], si", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSI, 2); }, &[0x66, 0x89, 0x34, 0x00]),
        ("mov word ptr [rax + rsi*1], si", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSI, 2); }, &[0x66, 0x89, 0x34, 0x30]),
        ("mov word ptr [rax + r13*1], si", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSI, 2); }, &[0x66, 0x42, 0x89, 0x34, 0x28]),
        ("mov word ptr [rsp + rcx*1], si", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSI, 2); }, &[0x66, 0x89, 0x34, 0x0c]),
        ("mov word ptr [rsp + r8*1], si", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSI, 2); }, &[0x66, 0x42, 0x89, 0x34, 0x04]),
        ("mov word ptr [rsp + r15*1], si", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSI, 2); }, &[0x66, 0x42, 0x89, 0x34, 0x3c]),
        ("mov word ptr [rbp + rbp*1], si", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSI, 2); }, &[0x66, 0x89, 0x74, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], si", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSI, 2); }, &[0x66, 0x42, 0x89, 0x74, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], si", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSI, 2); }, &[0x66, 0x41, 0x89, 0x34, 0x04]),
        ("mov word ptr [r12 + rsi*1], si", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSI, 2); }, &[0x66, 0x41, 0x89, 0x34, 0x34]),
        ("mov word ptr [r12 + r13*1], si", |a| { a.st(mem_idx(R12, R13, 0, 0), RSI, 2); }, &[0x66, 0x43, 0x89, 0x34, 0x2c]),
        ("mov word ptr [r13 + rcx*1], si", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSI, 2); }, &[0x66, 0x41, 0x89, 0x74, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], si", |a| { a.st(mem_idx(R13, R8, 0, 0), RSI, 2); }, &[0x66, 0x43, 0x89, 0x74, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], si", |a| { a.st(mem_idx(R13, R15, 0, 0), RSI, 2); }, &[0x66, 0x43, 0x89, 0x74, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], si", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSI, 2); }, &[0x66, 0x41, 0x89, 0x34, 0x2f]),
        ("mov word ptr [r15 + r12*1], si", |a| { a.st(mem_idx(R15, R12, 0, 0), RSI, 2); }, &[0x66, 0x43, 0x89, 0x34, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], si", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSI, 2); }, &[0x66, 0x89, 0x74, 0x4b, 0x20]),
        ("mov word ptr [rax], di", |a| { a.st(mem(RAX, 0), RDI, 2); }, &[0x66, 0x89, 0x38]),
        ("mov word ptr [rax - 305419896], di", |a| { a.st(mem(RAX, -305419896), RDI, 2); }, &[0x66, 0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], di", |a| { a.st(mem(RSP, 305419896), RDI, 2); }, &[0x66, 0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], di", |a| { a.st(mem(RBP, -129), RDI, 2); }, &[0x66, 0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], di", |a| { a.st(mem(R8, -128), RDI, 2); }, &[0x66, 0x41, 0x89, 0x78, 0x80]),
        ("mov word ptr [r12 + 128], di", |a| { a.st(mem(R12, 128), RDI, 2); }, &[0x66, 0x41, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], di", |a| { a.st(mem(R13, 127), RDI, 2); }, &[0x66, 0x41, 0x89, 0x7d, 0x7f]),
        ("mov word ptr [r15 - 1], di", |a| { a.st(mem(R15, -1), RDI, 2); }, &[0x66, 0x41, 0x89, 0x7f, 0xff]),
        ("mov word ptr [rbx + 1], di", |a| { a.st(mem(RBX, 1), RDI, 2); }, &[0x66, 0x89, 0x7b, 0x01]),
        ("mov word ptr [rax + rax*1], di", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RDI, 2); }, &[0x66, 0x89, 0x3c, 0x00]),
        ("mov word ptr [rax + rsi*1], di", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RDI, 2); }, &[0x66, 0x89, 0x3c, 0x30]),
        ("mov word ptr [rax + r13*1], di", |a| { a.st(mem_idx(RAX, R13, 0, 0), RDI, 2); }, &[0x66, 0x42, 0x89, 0x3c, 0x28]),
        ("mov word ptr [rsp + rcx*1], di", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RDI, 2); }, &[0x66, 0x89, 0x3c, 0x0c]),
        ("mov word ptr [rsp + r8*1], di", |a| { a.st(mem_idx(RSP, R8, 0, 0), RDI, 2); }, &[0x66, 0x42, 0x89, 0x3c, 0x04]),
        ("mov word ptr [rsp + r15*1], di", |a| { a.st(mem_idx(RSP, R15, 0, 0), RDI, 2); }, &[0x66, 0x42, 0x89, 0x3c, 0x3c]),
        ("mov word ptr [rbp + rbp*1], di", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RDI, 2); }, &[0x66, 0x89, 0x7c, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], di", |a| { a.st(mem_idx(RBP, R12, 0, 0), RDI, 2); }, &[0x66, 0x42, 0x89, 0x7c, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], di", |a| { a.st(mem_idx(R12, RAX, 0, 0), RDI, 2); }, &[0x66, 0x41, 0x89, 0x3c, 0x04]),
        ("mov word ptr [r12 + rsi*1], di", |a| { a.st(mem_idx(R12, RSI, 0, 0), RDI, 2); }, &[0x66, 0x41, 0x89, 0x3c, 0x34]),
        ("mov word ptr [r12 + r13*1], di", |a| { a.st(mem_idx(R12, R13, 0, 0), RDI, 2); }, &[0x66, 0x43, 0x89, 0x3c, 0x2c]),
        ("mov word ptr [r13 + rcx*1], di", |a| { a.st(mem_idx(R13, RCX, 0, 0), RDI, 2); }, &[0x66, 0x41, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], di", |a| { a.st(mem_idx(R13, R8, 0, 0), RDI, 2); }, &[0x66, 0x43, 0x89, 0x7c, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], di", |a| { a.st(mem_idx(R13, R15, 0, 0), RDI, 2); }, &[0x66, 0x43, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], di", |a| { a.st(mem_idx(R15, RBP, 0, 0), RDI, 2); }, &[0x66, 0x41, 0x89, 0x3c, 0x2f]),
        ("mov word ptr [r15 + r12*1], di", |a| { a.st(mem_idx(R15, R12, 0, 0), RDI, 2); }, &[0x66, 0x43, 0x89, 0x3c, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], di", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RDI, 2); }, &[0x66, 0x89, 0x7c, 0x4b, 0x20]),
        ("mov word ptr [rax], sp", |a| { a.st(mem(RAX, 0), RSP, 2); }, &[0x66, 0x89, 0x20]),
        ("mov word ptr [rax - 305419896], sp", |a| { a.st(mem(RAX, -305419896), RSP, 2); }, &[0x66, 0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], sp", |a| { a.st(mem(RSP, 305419896), RSP, 2); }, &[0x66, 0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], sp", |a| { a.st(mem(RBP, -129), RSP, 2); }, &[0x66, 0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], sp", |a| { a.st(mem(R8, -128), RSP, 2); }, &[0x66, 0x41, 0x89, 0x60, 0x80]),
        ("mov word ptr [r12 + 128], sp", |a| { a.st(mem(R12, 128), RSP, 2); }, &[0x66, 0x41, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], sp", |a| { a.st(mem(R13, 127), RSP, 2); }, &[0x66, 0x41, 0x89, 0x65, 0x7f]),
        ("mov word ptr [r15 - 1], sp", |a| { a.st(mem(R15, -1), RSP, 2); }, &[0x66, 0x41, 0x89, 0x67, 0xff]),
        ("mov word ptr [rbx + 1], sp", |a| { a.st(mem(RBX, 1), RSP, 2); }, &[0x66, 0x89, 0x63, 0x01]),
        ("mov word ptr [rax + rax*1], sp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSP, 2); }, &[0x66, 0x89, 0x24, 0x00]),
        ("mov word ptr [rax + rsi*1], sp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSP, 2); }, &[0x66, 0x89, 0x24, 0x30]),
        ("mov word ptr [rax + r13*1], sp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSP, 2); }, &[0x66, 0x42, 0x89, 0x24, 0x28]),
        ("mov word ptr [rsp + rcx*1], sp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSP, 2); }, &[0x66, 0x89, 0x24, 0x0c]),
        ("mov word ptr [rsp + r8*1], sp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSP, 2); }, &[0x66, 0x42, 0x89, 0x24, 0x04]),
        ("mov word ptr [rsp + r15*1], sp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSP, 2); }, &[0x66, 0x42, 0x89, 0x24, 0x3c]),
        ("mov word ptr [rbp + rbp*1], sp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSP, 2); }, &[0x66, 0x89, 0x64, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], sp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSP, 2); }, &[0x66, 0x42, 0x89, 0x64, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], sp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSP, 2); }, &[0x66, 0x41, 0x89, 0x24, 0x04]),
        ("mov word ptr [r12 + rsi*1], sp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSP, 2); }, &[0x66, 0x41, 0x89, 0x24, 0x34]),
        ("mov word ptr [r12 + r13*1], sp", |a| { a.st(mem_idx(R12, R13, 0, 0), RSP, 2); }, &[0x66, 0x43, 0x89, 0x24, 0x2c]),
        ("mov word ptr [r13 + rcx*1], sp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSP, 2); }, &[0x66, 0x41, 0x89, 0x64, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], sp", |a| { a.st(mem_idx(R13, R8, 0, 0), RSP, 2); }, &[0x66, 0x43, 0x89, 0x64, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], sp", |a| { a.st(mem_idx(R13, R15, 0, 0), RSP, 2); }, &[0x66, 0x43, 0x89, 0x64, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], sp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSP, 2); }, &[0x66, 0x41, 0x89, 0x24, 0x2f]),
        ("mov word ptr [r15 + r12*1], sp", |a| { a.st(mem_idx(R15, R12, 0, 0), RSP, 2); }, &[0x66, 0x43, 0x89, 0x24, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], sp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSP, 2); }, &[0x66, 0x89, 0x64, 0x4b, 0x20]),
        ("mov word ptr [rax], bp", |a| { a.st(mem(RAX, 0), RBP, 2); }, &[0x66, 0x89, 0x28]),
        ("mov word ptr [rax - 305419896], bp", |a| { a.st(mem(RAX, -305419896), RBP, 2); }, &[0x66, 0x89, 0xa8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], bp", |a| { a.st(mem(RSP, 305419896), RBP, 2); }, &[0x66, 0x89, 0xac, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], bp", |a| { a.st(mem(RBP, -129), RBP, 2); }, &[0x66, 0x89, 0xad, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], bp", |a| { a.st(mem(R8, -128), RBP, 2); }, &[0x66, 0x41, 0x89, 0x68, 0x80]),
        ("mov word ptr [r12 + 128], bp", |a| { a.st(mem(R12, 128), RBP, 2); }, &[0x66, 0x41, 0x89, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], bp", |a| { a.st(mem(R13, 127), RBP, 2); }, &[0x66, 0x41, 0x89, 0x6d, 0x7f]),
        ("mov word ptr [r15 - 1], bp", |a| { a.st(mem(R15, -1), RBP, 2); }, &[0x66, 0x41, 0x89, 0x6f, 0xff]),
        ("mov word ptr [rbx + 1], bp", |a| { a.st(mem(RBX, 1), RBP, 2); }, &[0x66, 0x89, 0x6b, 0x01]),
        ("mov word ptr [rax + rax*1], bp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RBP, 2); }, &[0x66, 0x89, 0x2c, 0x00]),
        ("mov word ptr [rax + rsi*1], bp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RBP, 2); }, &[0x66, 0x89, 0x2c, 0x30]),
        ("mov word ptr [rax + r13*1], bp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RBP, 2); }, &[0x66, 0x42, 0x89, 0x2c, 0x28]),
        ("mov word ptr [rsp + rcx*1], bp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RBP, 2); }, &[0x66, 0x89, 0x2c, 0x0c]),
        ("mov word ptr [rsp + r8*1], bp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RBP, 2); }, &[0x66, 0x42, 0x89, 0x2c, 0x04]),
        ("mov word ptr [rsp + r15*1], bp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RBP, 2); }, &[0x66, 0x42, 0x89, 0x2c, 0x3c]),
        ("mov word ptr [rbp + rbp*1], bp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RBP, 2); }, &[0x66, 0x89, 0x6c, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], bp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RBP, 2); }, &[0x66, 0x42, 0x89, 0x6c, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], bp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RBP, 2); }, &[0x66, 0x41, 0x89, 0x2c, 0x04]),
        ("mov word ptr [r12 + rsi*1], bp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RBP, 2); }, &[0x66, 0x41, 0x89, 0x2c, 0x34]),
        ("mov word ptr [r12 + r13*1], bp", |a| { a.st(mem_idx(R12, R13, 0, 0), RBP, 2); }, &[0x66, 0x43, 0x89, 0x2c, 0x2c]),
        ("mov word ptr [r13 + rcx*1], bp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RBP, 2); }, &[0x66, 0x41, 0x89, 0x6c, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], bp", |a| { a.st(mem_idx(R13, R8, 0, 0), RBP, 2); }, &[0x66, 0x43, 0x89, 0x6c, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], bp", |a| { a.st(mem_idx(R13, R15, 0, 0), RBP, 2); }, &[0x66, 0x43, 0x89, 0x6c, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], bp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RBP, 2); }, &[0x66, 0x41, 0x89, 0x2c, 0x2f]),
        ("mov word ptr [r15 + r12*1], bp", |a| { a.st(mem_idx(R15, R12, 0, 0), RBP, 2); }, &[0x66, 0x43, 0x89, 0x2c, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], bp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RBP, 2); }, &[0x66, 0x89, 0x6c, 0x4b, 0x20]),
        ("mov word ptr [rax], r8w", |a| { a.st(mem(RAX, 0), R8, 2); }, &[0x66, 0x44, 0x89, 0x00]),
        ("mov word ptr [rax - 305419896], r8w", |a| { a.st(mem(RAX, -305419896), R8, 2); }, &[0x66, 0x44, 0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], r8w", |a| { a.st(mem(RSP, 305419896), R8, 2); }, &[0x66, 0x44, 0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], r8w", |a| { a.st(mem(RBP, -129), R8, 2); }, &[0x66, 0x44, 0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], r8w", |a| { a.st(mem(R8, -128), R8, 2); }, &[0x66, 0x45, 0x89, 0x40, 0x80]),
        ("mov word ptr [r12 + 128], r8w", |a| { a.st(mem(R12, 128), R8, 2); }, &[0x66, 0x45, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], r8w", |a| { a.st(mem(R13, 127), R8, 2); }, &[0x66, 0x45, 0x89, 0x45, 0x7f]),
        ("mov word ptr [r15 - 1], r8w", |a| { a.st(mem(R15, -1), R8, 2); }, &[0x66, 0x45, 0x89, 0x47, 0xff]),
        ("mov word ptr [rbx + 1], r8w", |a| { a.st(mem(RBX, 1), R8, 2); }, &[0x66, 0x44, 0x89, 0x43, 0x01]),
        ("mov word ptr [rax + rax*1], r8w", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R8, 2); }, &[0x66, 0x44, 0x89, 0x04, 0x00]),
        ("mov word ptr [rax + rsi*1], r8w", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R8, 2); }, &[0x66, 0x44, 0x89, 0x04, 0x30]),
        ("mov word ptr [rax + r13*1], r8w", |a| { a.st(mem_idx(RAX, R13, 0, 0), R8, 2); }, &[0x66, 0x46, 0x89, 0x04, 0x28]),
        ("mov word ptr [rsp + rcx*1], r8w", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R8, 2); }, &[0x66, 0x44, 0x89, 0x04, 0x0c]),
        ("mov word ptr [rsp + r8*1], r8w", |a| { a.st(mem_idx(RSP, R8, 0, 0), R8, 2); }, &[0x66, 0x46, 0x89, 0x04, 0x04]),
        ("mov word ptr [rsp + r15*1], r8w", |a| { a.st(mem_idx(RSP, R15, 0, 0), R8, 2); }, &[0x66, 0x46, 0x89, 0x04, 0x3c]),
        ("mov word ptr [rbp + rbp*1], r8w", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R8, 2); }, &[0x66, 0x44, 0x89, 0x44, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], r8w", |a| { a.st(mem_idx(RBP, R12, 0, 0), R8, 2); }, &[0x66, 0x46, 0x89, 0x44, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], r8w", |a| { a.st(mem_idx(R12, RAX, 0, 0), R8, 2); }, &[0x66, 0x45, 0x89, 0x04, 0x04]),
        ("mov word ptr [r12 + rsi*1], r8w", |a| { a.st(mem_idx(R12, RSI, 0, 0), R8, 2); }, &[0x66, 0x45, 0x89, 0x04, 0x34]),
        ("mov word ptr [r12 + r13*1], r8w", |a| { a.st(mem_idx(R12, R13, 0, 0), R8, 2); }, &[0x66, 0x47, 0x89, 0x04, 0x2c]),
        ("mov word ptr [r13 + rcx*1], r8w", |a| { a.st(mem_idx(R13, RCX, 0, 0), R8, 2); }, &[0x66, 0x45, 0x89, 0x44, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], r8w", |a| { a.st(mem_idx(R13, R8, 0, 0), R8, 2); }, &[0x66, 0x47, 0x89, 0x44, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], r8w", |a| { a.st(mem_idx(R13, R15, 0, 0), R8, 2); }, &[0x66, 0x47, 0x89, 0x44, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], r8w", |a| { a.st(mem_idx(R15, RBP, 0, 0), R8, 2); }, &[0x66, 0x45, 0x89, 0x04, 0x2f]),
        ("mov word ptr [r15 + r12*1], r8w", |a| { a.st(mem_idx(R15, R12, 0, 0), R8, 2); }, &[0x66, 0x47, 0x89, 0x04, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], r8w", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R8, 2); }, &[0x66, 0x44, 0x89, 0x44, 0x4b, 0x20]),
        ("mov word ptr [rax], r15w", |a| { a.st(mem(RAX, 0), R15, 2); }, &[0x66, 0x44, 0x89, 0x38]),
        ("mov word ptr [rax - 305419896], r15w", |a| { a.st(mem(RAX, -305419896), R15, 2); }, &[0x66, 0x44, 0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], r15w", |a| { a.st(mem(RSP, 305419896), R15, 2); }, &[0x66, 0x44, 0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], r15w", |a| { a.st(mem(RBP, -129), R15, 2); }, &[0x66, 0x44, 0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], r15w", |a| { a.st(mem(R8, -128), R15, 2); }, &[0x66, 0x45, 0x89, 0x78, 0x80]),
        ("mov word ptr [r12 + 128], r15w", |a| { a.st(mem(R12, 128), R15, 2); }, &[0x66, 0x45, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], r15w", |a| { a.st(mem(R13, 127), R15, 2); }, &[0x66, 0x45, 0x89, 0x7d, 0x7f]),
        ("mov word ptr [r15 - 1], r15w", |a| { a.st(mem(R15, -1), R15, 2); }, &[0x66, 0x45, 0x89, 0x7f, 0xff]),
        ("mov word ptr [rbx + 1], r15w", |a| { a.st(mem(RBX, 1), R15, 2); }, &[0x66, 0x44, 0x89, 0x7b, 0x01]),
        ("mov word ptr [rax + rax*1], r15w", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R15, 2); }, &[0x66, 0x44, 0x89, 0x3c, 0x00]),
        ("mov word ptr [rax + rsi*1], r15w", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R15, 2); }, &[0x66, 0x44, 0x89, 0x3c, 0x30]),
        ("mov word ptr [rax + r13*1], r15w", |a| { a.st(mem_idx(RAX, R13, 0, 0), R15, 2); }, &[0x66, 0x46, 0x89, 0x3c, 0x28]),
        ("mov word ptr [rsp + rcx*1], r15w", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R15, 2); }, &[0x66, 0x44, 0x89, 0x3c, 0x0c]),
        ("mov word ptr [rsp + r8*1], r15w", |a| { a.st(mem_idx(RSP, R8, 0, 0), R15, 2); }, &[0x66, 0x46, 0x89, 0x3c, 0x04]),
        ("mov word ptr [rsp + r15*1], r15w", |a| { a.st(mem_idx(RSP, R15, 0, 0), R15, 2); }, &[0x66, 0x46, 0x89, 0x3c, 0x3c]),
        ("mov word ptr [rbp + rbp*1], r15w", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R15, 2); }, &[0x66, 0x44, 0x89, 0x7c, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], r15w", |a| { a.st(mem_idx(RBP, R12, 0, 0), R15, 2); }, &[0x66, 0x46, 0x89, 0x7c, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], r15w", |a| { a.st(mem_idx(R12, RAX, 0, 0), R15, 2); }, &[0x66, 0x45, 0x89, 0x3c, 0x04]),
        ("mov word ptr [r12 + rsi*1], r15w", |a| { a.st(mem_idx(R12, RSI, 0, 0), R15, 2); }, &[0x66, 0x45, 0x89, 0x3c, 0x34]),
        ("mov word ptr [r12 + r13*1], r15w", |a| { a.st(mem_idx(R12, R13, 0, 0), R15, 2); }, &[0x66, 0x47, 0x89, 0x3c, 0x2c]),
        ("mov word ptr [r13 + rcx*1], r15w", |a| { a.st(mem_idx(R13, RCX, 0, 0), R15, 2); }, &[0x66, 0x45, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], r15w", |a| { a.st(mem_idx(R13, R8, 0, 0), R15, 2); }, &[0x66, 0x47, 0x89, 0x7c, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], r15w", |a| { a.st(mem_idx(R13, R15, 0, 0), R15, 2); }, &[0x66, 0x47, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], r15w", |a| { a.st(mem_idx(R15, RBP, 0, 0), R15, 2); }, &[0x66, 0x45, 0x89, 0x3c, 0x2f]),
        ("mov word ptr [r15 + r12*1], r15w", |a| { a.st(mem_idx(R15, R12, 0, 0), R15, 2); }, &[0x66, 0x47, 0x89, 0x3c, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], r15w", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R15, 2); }, &[0x66, 0x44, 0x89, 0x7c, 0x4b, 0x20]),
        ("mov word ptr [rax], r12w", |a| { a.st(mem(RAX, 0), R12, 2); }, &[0x66, 0x44, 0x89, 0x20]),
        ("mov word ptr [rax - 305419896], r12w", |a| { a.st(mem(RAX, -305419896), R12, 2); }, &[0x66, 0x44, 0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov word ptr [rsp + 305419896], r12w", |a| { a.st(mem(RSP, 305419896), R12, 2); }, &[0x66, 0x44, 0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov word ptr [rbp - 129], r12w", |a| { a.st(mem(RBP, -129), R12, 2); }, &[0x66, 0x44, 0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov word ptr [r8 - 128], r12w", |a| { a.st(mem(R8, -128), R12, 2); }, &[0x66, 0x45, 0x89, 0x60, 0x80]),
        ("mov word ptr [r12 + 128], r12w", |a| { a.st(mem(R12, 128), R12, 2); }, &[0x66, 0x45, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov word ptr [r13 + 127], r12w", |a| { a.st(mem(R13, 127), R12, 2); }, &[0x66, 0x45, 0x89, 0x65, 0x7f]),
        ("mov word ptr [r15 - 1], r12w", |a| { a.st(mem(R15, -1), R12, 2); }, &[0x66, 0x45, 0x89, 0x67, 0xff]),
        ("mov word ptr [rbx + 1], r12w", |a| { a.st(mem(RBX, 1), R12, 2); }, &[0x66, 0x44, 0x89, 0x63, 0x01]),
        ("mov word ptr [rax + rax*1], r12w", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R12, 2); }, &[0x66, 0x44, 0x89, 0x24, 0x00]),
        ("mov word ptr [rax + rsi*1], r12w", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R12, 2); }, &[0x66, 0x44, 0x89, 0x24, 0x30]),
        ("mov word ptr [rax + r13*1], r12w", |a| { a.st(mem_idx(RAX, R13, 0, 0), R12, 2); }, &[0x66, 0x46, 0x89, 0x24, 0x28]),
        ("mov word ptr [rsp + rcx*1], r12w", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R12, 2); }, &[0x66, 0x44, 0x89, 0x24, 0x0c]),
        ("mov word ptr [rsp + r8*1], r12w", |a| { a.st(mem_idx(RSP, R8, 0, 0), R12, 2); }, &[0x66, 0x46, 0x89, 0x24, 0x04]),
        ("mov word ptr [rsp + r15*1], r12w", |a| { a.st(mem_idx(RSP, R15, 0, 0), R12, 2); }, &[0x66, 0x46, 0x89, 0x24, 0x3c]),
        ("mov word ptr [rbp + rbp*1], r12w", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R12, 2); }, &[0x66, 0x44, 0x89, 0x64, 0x2d, 0x00]),
        ("mov word ptr [rbp + r12*1], r12w", |a| { a.st(mem_idx(RBP, R12, 0, 0), R12, 2); }, &[0x66, 0x46, 0x89, 0x64, 0x25, 0x00]),
        ("mov word ptr [r12 + rax*1], r12w", |a| { a.st(mem_idx(R12, RAX, 0, 0), R12, 2); }, &[0x66, 0x45, 0x89, 0x24, 0x04]),
        ("mov word ptr [r12 + rsi*1], r12w", |a| { a.st(mem_idx(R12, RSI, 0, 0), R12, 2); }, &[0x66, 0x45, 0x89, 0x24, 0x34]),
        ("mov word ptr [r12 + r13*1], r12w", |a| { a.st(mem_idx(R12, R13, 0, 0), R12, 2); }, &[0x66, 0x47, 0x89, 0x24, 0x2c]),
        ("mov word ptr [r13 + rcx*1], r12w", |a| { a.st(mem_idx(R13, RCX, 0, 0), R12, 2); }, &[0x66, 0x45, 0x89, 0x64, 0x0d, 0x00]),
        ("mov word ptr [r13 + r8*1], r12w", |a| { a.st(mem_idx(R13, R8, 0, 0), R12, 2); }, &[0x66, 0x47, 0x89, 0x64, 0x05, 0x00]),
        ("mov word ptr [r13 + r15*1], r12w", |a| { a.st(mem_idx(R13, R15, 0, 0), R12, 2); }, &[0x66, 0x47, 0x89, 0x64, 0x3d, 0x00]),
        ("mov word ptr [r15 + rbp*1], r12w", |a| { a.st(mem_idx(R15, RBP, 0, 0), R12, 2); }, &[0x66, 0x45, 0x89, 0x24, 0x2f]),
        ("mov word ptr [r15 + r12*1], r12w", |a| { a.st(mem_idx(R15, R12, 0, 0), R12, 2); }, &[0x66, 0x47, 0x89, 0x24, 0x27]),
        ("mov word ptr [rbx + rcx*2 + 32], r12w", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R12, 2); }, &[0x66, 0x44, 0x89, 0x64, 0x4b, 0x20]),
        ("mov dword ptr [rax], eax", |a| { a.st(mem(RAX, 0), RAX, 4); }, &[0x89, 0x00]),
        ("mov dword ptr [rax - 305419896], eax", |a| { a.st(mem(RAX, -305419896), RAX, 4); }, &[0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], eax", |a| { a.st(mem(RSP, 305419896), RAX, 4); }, &[0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], eax", |a| { a.st(mem(RBP, -129), RAX, 4); }, &[0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], eax", |a| { a.st(mem(R8, -128), RAX, 4); }, &[0x41, 0x89, 0x40, 0x80]),
        ("mov dword ptr [r12 + 128], eax", |a| { a.st(mem(R12, 128), RAX, 4); }, &[0x41, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], eax", |a| { a.st(mem(R13, 127), RAX, 4); }, &[0x41, 0x89, 0x45, 0x7f]),
        ("mov dword ptr [r15 - 1], eax", |a| { a.st(mem(R15, -1), RAX, 4); }, &[0x41, 0x89, 0x47, 0xff]),
        ("mov dword ptr [rbx + 1], eax", |a| { a.st(mem(RBX, 1), RAX, 4); }, &[0x89, 0x43, 0x01]),
        ("mov dword ptr [rax + rax*1], eax", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RAX, 4); }, &[0x89, 0x04, 0x00]),
        ("mov dword ptr [rax + rsi*1], eax", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RAX, 4); }, &[0x89, 0x04, 0x30]),
        ("mov dword ptr [rax + r13*1], eax", |a| { a.st(mem_idx(RAX, R13, 0, 0), RAX, 4); }, &[0x42, 0x89, 0x04, 0x28]),
        ("mov dword ptr [rsp + rcx*1], eax", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RAX, 4); }, &[0x89, 0x04, 0x0c]),
        ("mov dword ptr [rsp + r8*1], eax", |a| { a.st(mem_idx(RSP, R8, 0, 0), RAX, 4); }, &[0x42, 0x89, 0x04, 0x04]),
        ("mov dword ptr [rsp + r15*1], eax", |a| { a.st(mem_idx(RSP, R15, 0, 0), RAX, 4); }, &[0x42, 0x89, 0x04, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], eax", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RAX, 4); }, &[0x89, 0x44, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], eax", |a| { a.st(mem_idx(RBP, R12, 0, 0), RAX, 4); }, &[0x42, 0x89, 0x44, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], eax", |a| { a.st(mem_idx(R12, RAX, 0, 0), RAX, 4); }, &[0x41, 0x89, 0x04, 0x04]),
        ("mov dword ptr [r12 + rsi*1], eax", |a| { a.st(mem_idx(R12, RSI, 0, 0), RAX, 4); }, &[0x41, 0x89, 0x04, 0x34]),
        ("mov dword ptr [r12 + r13*1], eax", |a| { a.st(mem_idx(R12, R13, 0, 0), RAX, 4); }, &[0x43, 0x89, 0x04, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], eax", |a| { a.st(mem_idx(R13, RCX, 0, 0), RAX, 4); }, &[0x41, 0x89, 0x44, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], eax", |a| { a.st(mem_idx(R13, R8, 0, 0), RAX, 4); }, &[0x43, 0x89, 0x44, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], eax", |a| { a.st(mem_idx(R13, R15, 0, 0), RAX, 4); }, &[0x43, 0x89, 0x44, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], eax", |a| { a.st(mem_idx(R15, RBP, 0, 0), RAX, 4); }, &[0x41, 0x89, 0x04, 0x2f]),
        ("mov dword ptr [r15 + r12*1], eax", |a| { a.st(mem_idx(R15, R12, 0, 0), RAX, 4); }, &[0x43, 0x89, 0x04, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], eax", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RAX, 4); }, &[0x89, 0x44, 0x4b, 0x20]),
        ("mov dword ptr [rax], esi", |a| { a.st(mem(RAX, 0), RSI, 4); }, &[0x89, 0x30]),
        ("mov dword ptr [rax - 305419896], esi", |a| { a.st(mem(RAX, -305419896), RSI, 4); }, &[0x89, 0xb0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], esi", |a| { a.st(mem(RSP, 305419896), RSI, 4); }, &[0x89, 0xb4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], esi", |a| { a.st(mem(RBP, -129), RSI, 4); }, &[0x89, 0xb5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], esi", |a| { a.st(mem(R8, -128), RSI, 4); }, &[0x41, 0x89, 0x70, 0x80]),
        ("mov dword ptr [r12 + 128], esi", |a| { a.st(mem(R12, 128), RSI, 4); }, &[0x41, 0x89, 0xb4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], esi", |a| { a.st(mem(R13, 127), RSI, 4); }, &[0x41, 0x89, 0x75, 0x7f]),
        ("mov dword ptr [r15 - 1], esi", |a| { a.st(mem(R15, -1), RSI, 4); }, &[0x41, 0x89, 0x77, 0xff]),
        ("mov dword ptr [rbx + 1], esi", |a| { a.st(mem(RBX, 1), RSI, 4); }, &[0x89, 0x73, 0x01]),
        ("mov dword ptr [rax + rax*1], esi", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSI, 4); }, &[0x89, 0x34, 0x00]),
        ("mov dword ptr [rax + rsi*1], esi", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSI, 4); }, &[0x89, 0x34, 0x30]),
        ("mov dword ptr [rax + r13*1], esi", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSI, 4); }, &[0x42, 0x89, 0x34, 0x28]),
        ("mov dword ptr [rsp + rcx*1], esi", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSI, 4); }, &[0x89, 0x34, 0x0c]),
        ("mov dword ptr [rsp + r8*1], esi", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSI, 4); }, &[0x42, 0x89, 0x34, 0x04]),
        ("mov dword ptr [rsp + r15*1], esi", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSI, 4); }, &[0x42, 0x89, 0x34, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], esi", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSI, 4); }, &[0x89, 0x74, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], esi", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSI, 4); }, &[0x42, 0x89, 0x74, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], esi", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSI, 4); }, &[0x41, 0x89, 0x34, 0x04]),
        ("mov dword ptr [r12 + rsi*1], esi", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSI, 4); }, &[0x41, 0x89, 0x34, 0x34]),
        ("mov dword ptr [r12 + r13*1], esi", |a| { a.st(mem_idx(R12, R13, 0, 0), RSI, 4); }, &[0x43, 0x89, 0x34, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], esi", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSI, 4); }, &[0x41, 0x89, 0x74, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], esi", |a| { a.st(mem_idx(R13, R8, 0, 0), RSI, 4); }, &[0x43, 0x89, 0x74, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], esi", |a| { a.st(mem_idx(R13, R15, 0, 0), RSI, 4); }, &[0x43, 0x89, 0x74, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], esi", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSI, 4); }, &[0x41, 0x89, 0x34, 0x2f]),
        ("mov dword ptr [r15 + r12*1], esi", |a| { a.st(mem_idx(R15, R12, 0, 0), RSI, 4); }, &[0x43, 0x89, 0x34, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], esi", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSI, 4); }, &[0x89, 0x74, 0x4b, 0x20]),
        ("mov dword ptr [rax], edi", |a| { a.st(mem(RAX, 0), RDI, 4); }, &[0x89, 0x38]),
        ("mov dword ptr [rax - 305419896], edi", |a| { a.st(mem(RAX, -305419896), RDI, 4); }, &[0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], edi", |a| { a.st(mem(RSP, 305419896), RDI, 4); }, &[0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], edi", |a| { a.st(mem(RBP, -129), RDI, 4); }, &[0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], edi", |a| { a.st(mem(R8, -128), RDI, 4); }, &[0x41, 0x89, 0x78, 0x80]),
        ("mov dword ptr [r12 + 128], edi", |a| { a.st(mem(R12, 128), RDI, 4); }, &[0x41, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], edi", |a| { a.st(mem(R13, 127), RDI, 4); }, &[0x41, 0x89, 0x7d, 0x7f]),
        ("mov dword ptr [r15 - 1], edi", |a| { a.st(mem(R15, -1), RDI, 4); }, &[0x41, 0x89, 0x7f, 0xff]),
        ("mov dword ptr [rbx + 1], edi", |a| { a.st(mem(RBX, 1), RDI, 4); }, &[0x89, 0x7b, 0x01]),
        ("mov dword ptr [rax + rax*1], edi", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RDI, 4); }, &[0x89, 0x3c, 0x00]),
        ("mov dword ptr [rax + rsi*1], edi", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RDI, 4); }, &[0x89, 0x3c, 0x30]),
        ("mov dword ptr [rax + r13*1], edi", |a| { a.st(mem_idx(RAX, R13, 0, 0), RDI, 4); }, &[0x42, 0x89, 0x3c, 0x28]),
        ("mov dword ptr [rsp + rcx*1], edi", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RDI, 4); }, &[0x89, 0x3c, 0x0c]),
        ("mov dword ptr [rsp + r8*1], edi", |a| { a.st(mem_idx(RSP, R8, 0, 0), RDI, 4); }, &[0x42, 0x89, 0x3c, 0x04]),
        ("mov dword ptr [rsp + r15*1], edi", |a| { a.st(mem_idx(RSP, R15, 0, 0), RDI, 4); }, &[0x42, 0x89, 0x3c, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], edi", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RDI, 4); }, &[0x89, 0x7c, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], edi", |a| { a.st(mem_idx(RBP, R12, 0, 0), RDI, 4); }, &[0x42, 0x89, 0x7c, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], edi", |a| { a.st(mem_idx(R12, RAX, 0, 0), RDI, 4); }, &[0x41, 0x89, 0x3c, 0x04]),
        ("mov dword ptr [r12 + rsi*1], edi", |a| { a.st(mem_idx(R12, RSI, 0, 0), RDI, 4); }, &[0x41, 0x89, 0x3c, 0x34]),
        ("mov dword ptr [r12 + r13*1], edi", |a| { a.st(mem_idx(R12, R13, 0, 0), RDI, 4); }, &[0x43, 0x89, 0x3c, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], edi", |a| { a.st(mem_idx(R13, RCX, 0, 0), RDI, 4); }, &[0x41, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], edi", |a| { a.st(mem_idx(R13, R8, 0, 0), RDI, 4); }, &[0x43, 0x89, 0x7c, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], edi", |a| { a.st(mem_idx(R13, R15, 0, 0), RDI, 4); }, &[0x43, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], edi", |a| { a.st(mem_idx(R15, RBP, 0, 0), RDI, 4); }, &[0x41, 0x89, 0x3c, 0x2f]),
        ("mov dword ptr [r15 + r12*1], edi", |a| { a.st(mem_idx(R15, R12, 0, 0), RDI, 4); }, &[0x43, 0x89, 0x3c, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], edi", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RDI, 4); }, &[0x89, 0x7c, 0x4b, 0x20]),
        ("mov dword ptr [rax], esp", |a| { a.st(mem(RAX, 0), RSP, 4); }, &[0x89, 0x20]),
        ("mov dword ptr [rax - 305419896], esp", |a| { a.st(mem(RAX, -305419896), RSP, 4); }, &[0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], esp", |a| { a.st(mem(RSP, 305419896), RSP, 4); }, &[0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], esp", |a| { a.st(mem(RBP, -129), RSP, 4); }, &[0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], esp", |a| { a.st(mem(R8, -128), RSP, 4); }, &[0x41, 0x89, 0x60, 0x80]),
        ("mov dword ptr [r12 + 128], esp", |a| { a.st(mem(R12, 128), RSP, 4); }, &[0x41, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], esp", |a| { a.st(mem(R13, 127), RSP, 4); }, &[0x41, 0x89, 0x65, 0x7f]),
        ("mov dword ptr [r15 - 1], esp", |a| { a.st(mem(R15, -1), RSP, 4); }, &[0x41, 0x89, 0x67, 0xff]),
        ("mov dword ptr [rbx + 1], esp", |a| { a.st(mem(RBX, 1), RSP, 4); }, &[0x89, 0x63, 0x01]),
        ("mov dword ptr [rax + rax*1], esp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSP, 4); }, &[0x89, 0x24, 0x00]),
        ("mov dword ptr [rax + rsi*1], esp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSP, 4); }, &[0x89, 0x24, 0x30]),
        ("mov dword ptr [rax + r13*1], esp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSP, 4); }, &[0x42, 0x89, 0x24, 0x28]),
        ("mov dword ptr [rsp + rcx*1], esp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSP, 4); }, &[0x89, 0x24, 0x0c]),
        ("mov dword ptr [rsp + r8*1], esp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSP, 4); }, &[0x42, 0x89, 0x24, 0x04]),
        ("mov dword ptr [rsp + r15*1], esp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSP, 4); }, &[0x42, 0x89, 0x24, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], esp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSP, 4); }, &[0x89, 0x64, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], esp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSP, 4); }, &[0x42, 0x89, 0x64, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], esp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSP, 4); }, &[0x41, 0x89, 0x24, 0x04]),
        ("mov dword ptr [r12 + rsi*1], esp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSP, 4); }, &[0x41, 0x89, 0x24, 0x34]),
        ("mov dword ptr [r12 + r13*1], esp", |a| { a.st(mem_idx(R12, R13, 0, 0), RSP, 4); }, &[0x43, 0x89, 0x24, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], esp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSP, 4); }, &[0x41, 0x89, 0x64, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], esp", |a| { a.st(mem_idx(R13, R8, 0, 0), RSP, 4); }, &[0x43, 0x89, 0x64, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], esp", |a| { a.st(mem_idx(R13, R15, 0, 0), RSP, 4); }, &[0x43, 0x89, 0x64, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], esp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSP, 4); }, &[0x41, 0x89, 0x24, 0x2f]),
        ("mov dword ptr [r15 + r12*1], esp", |a| { a.st(mem_idx(R15, R12, 0, 0), RSP, 4); }, &[0x43, 0x89, 0x24, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], esp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSP, 4); }, &[0x89, 0x64, 0x4b, 0x20]),
        ("mov dword ptr [rax], ebp", |a| { a.st(mem(RAX, 0), RBP, 4); }, &[0x89, 0x28]),
        ("mov dword ptr [rax - 305419896], ebp", |a| { a.st(mem(RAX, -305419896), RBP, 4); }, &[0x89, 0xa8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], ebp", |a| { a.st(mem(RSP, 305419896), RBP, 4); }, &[0x89, 0xac, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], ebp", |a| { a.st(mem(RBP, -129), RBP, 4); }, &[0x89, 0xad, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], ebp", |a| { a.st(mem(R8, -128), RBP, 4); }, &[0x41, 0x89, 0x68, 0x80]),
        ("mov dword ptr [r12 + 128], ebp", |a| { a.st(mem(R12, 128), RBP, 4); }, &[0x41, 0x89, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], ebp", |a| { a.st(mem(R13, 127), RBP, 4); }, &[0x41, 0x89, 0x6d, 0x7f]),
        ("mov dword ptr [r15 - 1], ebp", |a| { a.st(mem(R15, -1), RBP, 4); }, &[0x41, 0x89, 0x6f, 0xff]),
        ("mov dword ptr [rbx + 1], ebp", |a| { a.st(mem(RBX, 1), RBP, 4); }, &[0x89, 0x6b, 0x01]),
        ("mov dword ptr [rax + rax*1], ebp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RBP, 4); }, &[0x89, 0x2c, 0x00]),
        ("mov dword ptr [rax + rsi*1], ebp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RBP, 4); }, &[0x89, 0x2c, 0x30]),
        ("mov dword ptr [rax + r13*1], ebp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RBP, 4); }, &[0x42, 0x89, 0x2c, 0x28]),
        ("mov dword ptr [rsp + rcx*1], ebp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RBP, 4); }, &[0x89, 0x2c, 0x0c]),
        ("mov dword ptr [rsp + r8*1], ebp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RBP, 4); }, &[0x42, 0x89, 0x2c, 0x04]),
        ("mov dword ptr [rsp + r15*1], ebp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RBP, 4); }, &[0x42, 0x89, 0x2c, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], ebp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RBP, 4); }, &[0x89, 0x6c, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], ebp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RBP, 4); }, &[0x42, 0x89, 0x6c, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], ebp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RBP, 4); }, &[0x41, 0x89, 0x2c, 0x04]),
        ("mov dword ptr [r12 + rsi*1], ebp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RBP, 4); }, &[0x41, 0x89, 0x2c, 0x34]),
        ("mov dword ptr [r12 + r13*1], ebp", |a| { a.st(mem_idx(R12, R13, 0, 0), RBP, 4); }, &[0x43, 0x89, 0x2c, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], ebp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RBP, 4); }, &[0x41, 0x89, 0x6c, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], ebp", |a| { a.st(mem_idx(R13, R8, 0, 0), RBP, 4); }, &[0x43, 0x89, 0x6c, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], ebp", |a| { a.st(mem_idx(R13, R15, 0, 0), RBP, 4); }, &[0x43, 0x89, 0x6c, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], ebp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RBP, 4); }, &[0x41, 0x89, 0x2c, 0x2f]),
        ("mov dword ptr [r15 + r12*1], ebp", |a| { a.st(mem_idx(R15, R12, 0, 0), RBP, 4); }, &[0x43, 0x89, 0x2c, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], ebp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RBP, 4); }, &[0x89, 0x6c, 0x4b, 0x20]),
        ("mov dword ptr [rax], r8d", |a| { a.st(mem(RAX, 0), R8, 4); }, &[0x44, 0x89, 0x00]),
        ("mov dword ptr [rax - 305419896], r8d", |a| { a.st(mem(RAX, -305419896), R8, 4); }, &[0x44, 0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], r8d", |a| { a.st(mem(RSP, 305419896), R8, 4); }, &[0x44, 0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], r8d", |a| { a.st(mem(RBP, -129), R8, 4); }, &[0x44, 0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], r8d", |a| { a.st(mem(R8, -128), R8, 4); }, &[0x45, 0x89, 0x40, 0x80]),
        ("mov dword ptr [r12 + 128], r8d", |a| { a.st(mem(R12, 128), R8, 4); }, &[0x45, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], r8d", |a| { a.st(mem(R13, 127), R8, 4); }, &[0x45, 0x89, 0x45, 0x7f]),
        ("mov dword ptr [r15 - 1], r8d", |a| { a.st(mem(R15, -1), R8, 4); }, &[0x45, 0x89, 0x47, 0xff]),
        ("mov dword ptr [rbx + 1], r8d", |a| { a.st(mem(RBX, 1), R8, 4); }, &[0x44, 0x89, 0x43, 0x01]),
        ("mov dword ptr [rax + rax*1], r8d", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R8, 4); }, &[0x44, 0x89, 0x04, 0x00]),
        ("mov dword ptr [rax + rsi*1], r8d", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R8, 4); }, &[0x44, 0x89, 0x04, 0x30]),
        ("mov dword ptr [rax + r13*1], r8d", |a| { a.st(mem_idx(RAX, R13, 0, 0), R8, 4); }, &[0x46, 0x89, 0x04, 0x28]),
        ("mov dword ptr [rsp + rcx*1], r8d", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R8, 4); }, &[0x44, 0x89, 0x04, 0x0c]),
        ("mov dword ptr [rsp + r8*1], r8d", |a| { a.st(mem_idx(RSP, R8, 0, 0), R8, 4); }, &[0x46, 0x89, 0x04, 0x04]),
        ("mov dword ptr [rsp + r15*1], r8d", |a| { a.st(mem_idx(RSP, R15, 0, 0), R8, 4); }, &[0x46, 0x89, 0x04, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], r8d", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R8, 4); }, &[0x44, 0x89, 0x44, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], r8d", |a| { a.st(mem_idx(RBP, R12, 0, 0), R8, 4); }, &[0x46, 0x89, 0x44, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], r8d", |a| { a.st(mem_idx(R12, RAX, 0, 0), R8, 4); }, &[0x45, 0x89, 0x04, 0x04]),
        ("mov dword ptr [r12 + rsi*1], r8d", |a| { a.st(mem_idx(R12, RSI, 0, 0), R8, 4); }, &[0x45, 0x89, 0x04, 0x34]),
        ("mov dword ptr [r12 + r13*1], r8d", |a| { a.st(mem_idx(R12, R13, 0, 0), R8, 4); }, &[0x47, 0x89, 0x04, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], r8d", |a| { a.st(mem_idx(R13, RCX, 0, 0), R8, 4); }, &[0x45, 0x89, 0x44, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], r8d", |a| { a.st(mem_idx(R13, R8, 0, 0), R8, 4); }, &[0x47, 0x89, 0x44, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], r8d", |a| { a.st(mem_idx(R13, R15, 0, 0), R8, 4); }, &[0x47, 0x89, 0x44, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], r8d", |a| { a.st(mem_idx(R15, RBP, 0, 0), R8, 4); }, &[0x45, 0x89, 0x04, 0x2f]),
        ("mov dword ptr [r15 + r12*1], r8d", |a| { a.st(mem_idx(R15, R12, 0, 0), R8, 4); }, &[0x47, 0x89, 0x04, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], r8d", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R8, 4); }, &[0x44, 0x89, 0x44, 0x4b, 0x20]),
        ("mov dword ptr [rax], r15d", |a| { a.st(mem(RAX, 0), R15, 4); }, &[0x44, 0x89, 0x38]),
        ("mov dword ptr [rax - 305419896], r15d", |a| { a.st(mem(RAX, -305419896), R15, 4); }, &[0x44, 0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], r15d", |a| { a.st(mem(RSP, 305419896), R15, 4); }, &[0x44, 0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], r15d", |a| { a.st(mem(RBP, -129), R15, 4); }, &[0x44, 0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], r15d", |a| { a.st(mem(R8, -128), R15, 4); }, &[0x45, 0x89, 0x78, 0x80]),
        ("mov dword ptr [r12 + 128], r15d", |a| { a.st(mem(R12, 128), R15, 4); }, &[0x45, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], r15d", |a| { a.st(mem(R13, 127), R15, 4); }, &[0x45, 0x89, 0x7d, 0x7f]),
        ("mov dword ptr [r15 - 1], r15d", |a| { a.st(mem(R15, -1), R15, 4); }, &[0x45, 0x89, 0x7f, 0xff]),
        ("mov dword ptr [rbx + 1], r15d", |a| { a.st(mem(RBX, 1), R15, 4); }, &[0x44, 0x89, 0x7b, 0x01]),
        ("mov dword ptr [rax + rax*1], r15d", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R15, 4); }, &[0x44, 0x89, 0x3c, 0x00]),
        ("mov dword ptr [rax + rsi*1], r15d", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R15, 4); }, &[0x44, 0x89, 0x3c, 0x30]),
        ("mov dword ptr [rax + r13*1], r15d", |a| { a.st(mem_idx(RAX, R13, 0, 0), R15, 4); }, &[0x46, 0x89, 0x3c, 0x28]),
        ("mov dword ptr [rsp + rcx*1], r15d", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R15, 4); }, &[0x44, 0x89, 0x3c, 0x0c]),
        ("mov dword ptr [rsp + r8*1], r15d", |a| { a.st(mem_idx(RSP, R8, 0, 0), R15, 4); }, &[0x46, 0x89, 0x3c, 0x04]),
        ("mov dword ptr [rsp + r15*1], r15d", |a| { a.st(mem_idx(RSP, R15, 0, 0), R15, 4); }, &[0x46, 0x89, 0x3c, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], r15d", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R15, 4); }, &[0x44, 0x89, 0x7c, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], r15d", |a| { a.st(mem_idx(RBP, R12, 0, 0), R15, 4); }, &[0x46, 0x89, 0x7c, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], r15d", |a| { a.st(mem_idx(R12, RAX, 0, 0), R15, 4); }, &[0x45, 0x89, 0x3c, 0x04]),
        ("mov dword ptr [r12 + rsi*1], r15d", |a| { a.st(mem_idx(R12, RSI, 0, 0), R15, 4); }, &[0x45, 0x89, 0x3c, 0x34]),
        ("mov dword ptr [r12 + r13*1], r15d", |a| { a.st(mem_idx(R12, R13, 0, 0), R15, 4); }, &[0x47, 0x89, 0x3c, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], r15d", |a| { a.st(mem_idx(R13, RCX, 0, 0), R15, 4); }, &[0x45, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], r15d", |a| { a.st(mem_idx(R13, R8, 0, 0), R15, 4); }, &[0x47, 0x89, 0x7c, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], r15d", |a| { a.st(mem_idx(R13, R15, 0, 0), R15, 4); }, &[0x47, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], r15d", |a| { a.st(mem_idx(R15, RBP, 0, 0), R15, 4); }, &[0x45, 0x89, 0x3c, 0x2f]),
        ("mov dword ptr [r15 + r12*1], r15d", |a| { a.st(mem_idx(R15, R12, 0, 0), R15, 4); }, &[0x47, 0x89, 0x3c, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], r15d", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R15, 4); }, &[0x44, 0x89, 0x7c, 0x4b, 0x20]),
        ("mov dword ptr [rax], r12d", |a| { a.st(mem(RAX, 0), R12, 4); }, &[0x44, 0x89, 0x20]),
        ("mov dword ptr [rax - 305419896], r12d", |a| { a.st(mem(RAX, -305419896), R12, 4); }, &[0x44, 0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov dword ptr [rsp + 305419896], r12d", |a| { a.st(mem(RSP, 305419896), R12, 4); }, &[0x44, 0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov dword ptr [rbp - 129], r12d", |a| { a.st(mem(RBP, -129), R12, 4); }, &[0x44, 0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov dword ptr [r8 - 128], r12d", |a| { a.st(mem(R8, -128), R12, 4); }, &[0x45, 0x89, 0x60, 0x80]),
        ("mov dword ptr [r12 + 128], r12d", |a| { a.st(mem(R12, 128), R12, 4); }, &[0x45, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov dword ptr [r13 + 127], r12d", |a| { a.st(mem(R13, 127), R12, 4); }, &[0x45, 0x89, 0x65, 0x7f]),
        ("mov dword ptr [r15 - 1], r12d", |a| { a.st(mem(R15, -1), R12, 4); }, &[0x45, 0x89, 0x67, 0xff]),
        ("mov dword ptr [rbx + 1], r12d", |a| { a.st(mem(RBX, 1), R12, 4); }, &[0x44, 0x89, 0x63, 0x01]),
        ("mov dword ptr [rax + rax*1], r12d", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R12, 4); }, &[0x44, 0x89, 0x24, 0x00]),
        ("mov dword ptr [rax + rsi*1], r12d", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R12, 4); }, &[0x44, 0x89, 0x24, 0x30]),
        ("mov dword ptr [rax + r13*1], r12d", |a| { a.st(mem_idx(RAX, R13, 0, 0), R12, 4); }, &[0x46, 0x89, 0x24, 0x28]),
        ("mov dword ptr [rsp + rcx*1], r12d", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R12, 4); }, &[0x44, 0x89, 0x24, 0x0c]),
        ("mov dword ptr [rsp + r8*1], r12d", |a| { a.st(mem_idx(RSP, R8, 0, 0), R12, 4); }, &[0x46, 0x89, 0x24, 0x04]),
        ("mov dword ptr [rsp + r15*1], r12d", |a| { a.st(mem_idx(RSP, R15, 0, 0), R12, 4); }, &[0x46, 0x89, 0x24, 0x3c]),
        ("mov dword ptr [rbp + rbp*1], r12d", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R12, 4); }, &[0x44, 0x89, 0x64, 0x2d, 0x00]),
        ("mov dword ptr [rbp + r12*1], r12d", |a| { a.st(mem_idx(RBP, R12, 0, 0), R12, 4); }, &[0x46, 0x89, 0x64, 0x25, 0x00]),
        ("mov dword ptr [r12 + rax*1], r12d", |a| { a.st(mem_idx(R12, RAX, 0, 0), R12, 4); }, &[0x45, 0x89, 0x24, 0x04]),
        ("mov dword ptr [r12 + rsi*1], r12d", |a| { a.st(mem_idx(R12, RSI, 0, 0), R12, 4); }, &[0x45, 0x89, 0x24, 0x34]),
        ("mov dword ptr [r12 + r13*1], r12d", |a| { a.st(mem_idx(R12, R13, 0, 0), R12, 4); }, &[0x47, 0x89, 0x24, 0x2c]),
        ("mov dword ptr [r13 + rcx*1], r12d", |a| { a.st(mem_idx(R13, RCX, 0, 0), R12, 4); }, &[0x45, 0x89, 0x64, 0x0d, 0x00]),
        ("mov dword ptr [r13 + r8*1], r12d", |a| { a.st(mem_idx(R13, R8, 0, 0), R12, 4); }, &[0x47, 0x89, 0x64, 0x05, 0x00]),
        ("mov dword ptr [r13 + r15*1], r12d", |a| { a.st(mem_idx(R13, R15, 0, 0), R12, 4); }, &[0x47, 0x89, 0x64, 0x3d, 0x00]),
        ("mov dword ptr [r15 + rbp*1], r12d", |a| { a.st(mem_idx(R15, RBP, 0, 0), R12, 4); }, &[0x45, 0x89, 0x24, 0x2f]),
        ("mov dword ptr [r15 + r12*1], r12d", |a| { a.st(mem_idx(R15, R12, 0, 0), R12, 4); }, &[0x47, 0x89, 0x24, 0x27]),
        ("mov dword ptr [rbx + rcx*2 + 32], r12d", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R12, 4); }, &[0x44, 0x89, 0x64, 0x4b, 0x20]),
        ("mov qword ptr [rax], rax", |a| { a.st(mem(RAX, 0), RAX, 8); }, &[0x48, 0x89, 0x00]),
        ("mov qword ptr [rax - 305419896], rax", |a| { a.st(mem(RAX, -305419896), RAX, 8); }, &[0x48, 0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], rax", |a| { a.st(mem(RSP, 305419896), RAX, 8); }, &[0x48, 0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], rax", |a| { a.st(mem(RBP, -129), RAX, 8); }, &[0x48, 0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], rax", |a| { a.st(mem(R8, -128), RAX, 8); }, &[0x49, 0x89, 0x40, 0x80]),
        ("mov qword ptr [r12 + 128], rax", |a| { a.st(mem(R12, 128), RAX, 8); }, &[0x49, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], rax", |a| { a.st(mem(R13, 127), RAX, 8); }, &[0x49, 0x89, 0x45, 0x7f]),
        ("mov qword ptr [r15 - 1], rax", |a| { a.st(mem(R15, -1), RAX, 8); }, &[0x49, 0x89, 0x47, 0xff]),
        ("mov qword ptr [rbx + 1], rax", |a| { a.st(mem(RBX, 1), RAX, 8); }, &[0x48, 0x89, 0x43, 0x01]),
        ("mov qword ptr [rax + rax*1], rax", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RAX, 8); }, &[0x48, 0x89, 0x04, 0x00]),
        ("mov qword ptr [rax + rsi*1], rax", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RAX, 8); }, &[0x48, 0x89, 0x04, 0x30]),
        ("mov qword ptr [rax + r13*1], rax", |a| { a.st(mem_idx(RAX, R13, 0, 0), RAX, 8); }, &[0x4a, 0x89, 0x04, 0x28]),
        ("mov qword ptr [rsp + rcx*1], rax", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RAX, 8); }, &[0x48, 0x89, 0x04, 0x0c]),
        ("mov qword ptr [rsp + r8*1], rax", |a| { a.st(mem_idx(RSP, R8, 0, 0), RAX, 8); }, &[0x4a, 0x89, 0x04, 0x04]),
        ("mov qword ptr [rsp + r15*1], rax", |a| { a.st(mem_idx(RSP, R15, 0, 0), RAX, 8); }, &[0x4a, 0x89, 0x04, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], rax", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RAX, 8); }, &[0x48, 0x89, 0x44, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], rax", |a| { a.st(mem_idx(RBP, R12, 0, 0), RAX, 8); }, &[0x4a, 0x89, 0x44, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], rax", |a| { a.st(mem_idx(R12, RAX, 0, 0), RAX, 8); }, &[0x49, 0x89, 0x04, 0x04]),
        ("mov qword ptr [r12 + rsi*1], rax", |a| { a.st(mem_idx(R12, RSI, 0, 0), RAX, 8); }, &[0x49, 0x89, 0x04, 0x34]),
        ("mov qword ptr [r12 + r13*1], rax", |a| { a.st(mem_idx(R12, R13, 0, 0), RAX, 8); }, &[0x4b, 0x89, 0x04, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], rax", |a| { a.st(mem_idx(R13, RCX, 0, 0), RAX, 8); }, &[0x49, 0x89, 0x44, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], rax", |a| { a.st(mem_idx(R13, R8, 0, 0), RAX, 8); }, &[0x4b, 0x89, 0x44, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], rax", |a| { a.st(mem_idx(R13, R15, 0, 0), RAX, 8); }, &[0x4b, 0x89, 0x44, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], rax", |a| { a.st(mem_idx(R15, RBP, 0, 0), RAX, 8); }, &[0x49, 0x89, 0x04, 0x2f]),
        ("mov qword ptr [r15 + r12*1], rax", |a| { a.st(mem_idx(R15, R12, 0, 0), RAX, 8); }, &[0x4b, 0x89, 0x04, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], rax", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RAX, 8); }, &[0x48, 0x89, 0x44, 0x4b, 0x20]),
        ("mov qword ptr [rax], rsi", |a| { a.st(mem(RAX, 0), RSI, 8); }, &[0x48, 0x89, 0x30]),
        ("mov qword ptr [rax - 305419896], rsi", |a| { a.st(mem(RAX, -305419896), RSI, 8); }, &[0x48, 0x89, 0xb0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], rsi", |a| { a.st(mem(RSP, 305419896), RSI, 8); }, &[0x48, 0x89, 0xb4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], rsi", |a| { a.st(mem(RBP, -129), RSI, 8); }, &[0x48, 0x89, 0xb5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], rsi", |a| { a.st(mem(R8, -128), RSI, 8); }, &[0x49, 0x89, 0x70, 0x80]),
        ("mov qword ptr [r12 + 128], rsi", |a| { a.st(mem(R12, 128), RSI, 8); }, &[0x49, 0x89, 0xb4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], rsi", |a| { a.st(mem(R13, 127), RSI, 8); }, &[0x49, 0x89, 0x75, 0x7f]),
        ("mov qword ptr [r15 - 1], rsi", |a| { a.st(mem(R15, -1), RSI, 8); }, &[0x49, 0x89, 0x77, 0xff]),
        ("mov qword ptr [rbx + 1], rsi", |a| { a.st(mem(RBX, 1), RSI, 8); }, &[0x48, 0x89, 0x73, 0x01]),
        ("mov qword ptr [rax + rax*1], rsi", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSI, 8); }, &[0x48, 0x89, 0x34, 0x00]),
        ("mov qword ptr [rax + rsi*1], rsi", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSI, 8); }, &[0x48, 0x89, 0x34, 0x30]),
        ("mov qword ptr [rax + r13*1], rsi", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSI, 8); }, &[0x4a, 0x89, 0x34, 0x28]),
        ("mov qword ptr [rsp + rcx*1], rsi", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSI, 8); }, &[0x48, 0x89, 0x34, 0x0c]),
        ("mov qword ptr [rsp + r8*1], rsi", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSI, 8); }, &[0x4a, 0x89, 0x34, 0x04]),
        ("mov qword ptr [rsp + r15*1], rsi", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSI, 8); }, &[0x4a, 0x89, 0x34, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], rsi", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSI, 8); }, &[0x48, 0x89, 0x74, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], rsi", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSI, 8); }, &[0x4a, 0x89, 0x74, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], rsi", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSI, 8); }, &[0x49, 0x89, 0x34, 0x04]),
        ("mov qword ptr [r12 + rsi*1], rsi", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSI, 8); }, &[0x49, 0x89, 0x34, 0x34]),
        ("mov qword ptr [r12 + r13*1], rsi", |a| { a.st(mem_idx(R12, R13, 0, 0), RSI, 8); }, &[0x4b, 0x89, 0x34, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], rsi", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSI, 8); }, &[0x49, 0x89, 0x74, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], rsi", |a| { a.st(mem_idx(R13, R8, 0, 0), RSI, 8); }, &[0x4b, 0x89, 0x74, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], rsi", |a| { a.st(mem_idx(R13, R15, 0, 0), RSI, 8); }, &[0x4b, 0x89, 0x74, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], rsi", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSI, 8); }, &[0x49, 0x89, 0x34, 0x2f]),
        ("mov qword ptr [r15 + r12*1], rsi", |a| { a.st(mem_idx(R15, R12, 0, 0), RSI, 8); }, &[0x4b, 0x89, 0x34, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], rsi", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSI, 8); }, &[0x48, 0x89, 0x74, 0x4b, 0x20]),
        ("mov qword ptr [rax], rdi", |a| { a.st(mem(RAX, 0), RDI, 8); }, &[0x48, 0x89, 0x38]),
        ("mov qword ptr [rax - 305419896], rdi", |a| { a.st(mem(RAX, -305419896), RDI, 8); }, &[0x48, 0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], rdi", |a| { a.st(mem(RSP, 305419896), RDI, 8); }, &[0x48, 0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], rdi", |a| { a.st(mem(RBP, -129), RDI, 8); }, &[0x48, 0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], rdi", |a| { a.st(mem(R8, -128), RDI, 8); }, &[0x49, 0x89, 0x78, 0x80]),
        ("mov qword ptr [r12 + 128], rdi", |a| { a.st(mem(R12, 128), RDI, 8); }, &[0x49, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], rdi", |a| { a.st(mem(R13, 127), RDI, 8); }, &[0x49, 0x89, 0x7d, 0x7f]),
        ("mov qword ptr [r15 - 1], rdi", |a| { a.st(mem(R15, -1), RDI, 8); }, &[0x49, 0x89, 0x7f, 0xff]),
        ("mov qword ptr [rbx + 1], rdi", |a| { a.st(mem(RBX, 1), RDI, 8); }, &[0x48, 0x89, 0x7b, 0x01]),
        ("mov qword ptr [rax + rax*1], rdi", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RDI, 8); }, &[0x48, 0x89, 0x3c, 0x00]),
        ("mov qword ptr [rax + rsi*1], rdi", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RDI, 8); }, &[0x48, 0x89, 0x3c, 0x30]),
        ("mov qword ptr [rax + r13*1], rdi", |a| { a.st(mem_idx(RAX, R13, 0, 0), RDI, 8); }, &[0x4a, 0x89, 0x3c, 0x28]),
        ("mov qword ptr [rsp + rcx*1], rdi", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RDI, 8); }, &[0x48, 0x89, 0x3c, 0x0c]),
        ("mov qword ptr [rsp + r8*1], rdi", |a| { a.st(mem_idx(RSP, R8, 0, 0), RDI, 8); }, &[0x4a, 0x89, 0x3c, 0x04]),
        ("mov qword ptr [rsp + r15*1], rdi", |a| { a.st(mem_idx(RSP, R15, 0, 0), RDI, 8); }, &[0x4a, 0x89, 0x3c, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], rdi", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RDI, 8); }, &[0x48, 0x89, 0x7c, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], rdi", |a| { a.st(mem_idx(RBP, R12, 0, 0), RDI, 8); }, &[0x4a, 0x89, 0x7c, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], rdi", |a| { a.st(mem_idx(R12, RAX, 0, 0), RDI, 8); }, &[0x49, 0x89, 0x3c, 0x04]),
        ("mov qword ptr [r12 + rsi*1], rdi", |a| { a.st(mem_idx(R12, RSI, 0, 0), RDI, 8); }, &[0x49, 0x89, 0x3c, 0x34]),
        ("mov qword ptr [r12 + r13*1], rdi", |a| { a.st(mem_idx(R12, R13, 0, 0), RDI, 8); }, &[0x4b, 0x89, 0x3c, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], rdi", |a| { a.st(mem_idx(R13, RCX, 0, 0), RDI, 8); }, &[0x49, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], rdi", |a| { a.st(mem_idx(R13, R8, 0, 0), RDI, 8); }, &[0x4b, 0x89, 0x7c, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], rdi", |a| { a.st(mem_idx(R13, R15, 0, 0), RDI, 8); }, &[0x4b, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], rdi", |a| { a.st(mem_idx(R15, RBP, 0, 0), RDI, 8); }, &[0x49, 0x89, 0x3c, 0x2f]),
        ("mov qword ptr [r15 + r12*1], rdi", |a| { a.st(mem_idx(R15, R12, 0, 0), RDI, 8); }, &[0x4b, 0x89, 0x3c, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], rdi", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RDI, 8); }, &[0x48, 0x89, 0x7c, 0x4b, 0x20]),
        ("mov qword ptr [rax], rsp", |a| { a.st(mem(RAX, 0), RSP, 8); }, &[0x48, 0x89, 0x20]),
        ("mov qword ptr [rax - 305419896], rsp", |a| { a.st(mem(RAX, -305419896), RSP, 8); }, &[0x48, 0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], rsp", |a| { a.st(mem(RSP, 305419896), RSP, 8); }, &[0x48, 0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], rsp", |a| { a.st(mem(RBP, -129), RSP, 8); }, &[0x48, 0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], rsp", |a| { a.st(mem(R8, -128), RSP, 8); }, &[0x49, 0x89, 0x60, 0x80]),
        ("mov qword ptr [r12 + 128], rsp", |a| { a.st(mem(R12, 128), RSP, 8); }, &[0x49, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], rsp", |a| { a.st(mem(R13, 127), RSP, 8); }, &[0x49, 0x89, 0x65, 0x7f]),
        ("mov qword ptr [r15 - 1], rsp", |a| { a.st(mem(R15, -1), RSP, 8); }, &[0x49, 0x89, 0x67, 0xff]),
        ("mov qword ptr [rbx + 1], rsp", |a| { a.st(mem(RBX, 1), RSP, 8); }, &[0x48, 0x89, 0x63, 0x01]),
        ("mov qword ptr [rax + rax*1], rsp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RSP, 8); }, &[0x48, 0x89, 0x24, 0x00]),
        ("mov qword ptr [rax + rsi*1], rsp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RSP, 8); }, &[0x48, 0x89, 0x24, 0x30]),
        ("mov qword ptr [rax + r13*1], rsp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RSP, 8); }, &[0x4a, 0x89, 0x24, 0x28]),
        ("mov qword ptr [rsp + rcx*1], rsp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RSP, 8); }, &[0x48, 0x89, 0x24, 0x0c]),
        ("mov qword ptr [rsp + r8*1], rsp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RSP, 8); }, &[0x4a, 0x89, 0x24, 0x04]),
        ("mov qword ptr [rsp + r15*1], rsp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RSP, 8); }, &[0x4a, 0x89, 0x24, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], rsp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RSP, 8); }, &[0x48, 0x89, 0x64, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], rsp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RSP, 8); }, &[0x4a, 0x89, 0x64, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], rsp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RSP, 8); }, &[0x49, 0x89, 0x24, 0x04]),
        ("mov qword ptr [r12 + rsi*1], rsp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RSP, 8); }, &[0x49, 0x89, 0x24, 0x34]),
        ("mov qword ptr [r12 + r13*1], rsp", |a| { a.st(mem_idx(R12, R13, 0, 0), RSP, 8); }, &[0x4b, 0x89, 0x24, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], rsp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RSP, 8); }, &[0x49, 0x89, 0x64, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], rsp", |a| { a.st(mem_idx(R13, R8, 0, 0), RSP, 8); }, &[0x4b, 0x89, 0x64, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], rsp", |a| { a.st(mem_idx(R13, R15, 0, 0), RSP, 8); }, &[0x4b, 0x89, 0x64, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], rsp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RSP, 8); }, &[0x49, 0x89, 0x24, 0x2f]),
        ("mov qword ptr [r15 + r12*1], rsp", |a| { a.st(mem_idx(R15, R12, 0, 0), RSP, 8); }, &[0x4b, 0x89, 0x24, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], rsp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RSP, 8); }, &[0x48, 0x89, 0x64, 0x4b, 0x20]),
        ("mov qword ptr [rax], rbp", |a| { a.st(mem(RAX, 0), RBP, 8); }, &[0x48, 0x89, 0x28]),
        ("mov qword ptr [rax - 305419896], rbp", |a| { a.st(mem(RAX, -305419896), RBP, 8); }, &[0x48, 0x89, 0xa8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], rbp", |a| { a.st(mem(RSP, 305419896), RBP, 8); }, &[0x48, 0x89, 0xac, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], rbp", |a| { a.st(mem(RBP, -129), RBP, 8); }, &[0x48, 0x89, 0xad, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], rbp", |a| { a.st(mem(R8, -128), RBP, 8); }, &[0x49, 0x89, 0x68, 0x80]),
        ("mov qword ptr [r12 + 128], rbp", |a| { a.st(mem(R12, 128), RBP, 8); }, &[0x49, 0x89, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], rbp", |a| { a.st(mem(R13, 127), RBP, 8); }, &[0x49, 0x89, 0x6d, 0x7f]),
        ("mov qword ptr [r15 - 1], rbp", |a| { a.st(mem(R15, -1), RBP, 8); }, &[0x49, 0x89, 0x6f, 0xff]),
        ("mov qword ptr [rbx + 1], rbp", |a| { a.st(mem(RBX, 1), RBP, 8); }, &[0x48, 0x89, 0x6b, 0x01]),
        ("mov qword ptr [rax + rax*1], rbp", |a| { a.st(mem_idx(RAX, RAX, 0, 0), RBP, 8); }, &[0x48, 0x89, 0x2c, 0x00]),
        ("mov qword ptr [rax + rsi*1], rbp", |a| { a.st(mem_idx(RAX, RSI, 0, 0), RBP, 8); }, &[0x48, 0x89, 0x2c, 0x30]),
        ("mov qword ptr [rax + r13*1], rbp", |a| { a.st(mem_idx(RAX, R13, 0, 0), RBP, 8); }, &[0x4a, 0x89, 0x2c, 0x28]),
        ("mov qword ptr [rsp + rcx*1], rbp", |a| { a.st(mem_idx(RSP, RCX, 0, 0), RBP, 8); }, &[0x48, 0x89, 0x2c, 0x0c]),
        ("mov qword ptr [rsp + r8*1], rbp", |a| { a.st(mem_idx(RSP, R8, 0, 0), RBP, 8); }, &[0x4a, 0x89, 0x2c, 0x04]),
        ("mov qword ptr [rsp + r15*1], rbp", |a| { a.st(mem_idx(RSP, R15, 0, 0), RBP, 8); }, &[0x4a, 0x89, 0x2c, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], rbp", |a| { a.st(mem_idx(RBP, RBP, 0, 0), RBP, 8); }, &[0x48, 0x89, 0x6c, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], rbp", |a| { a.st(mem_idx(RBP, R12, 0, 0), RBP, 8); }, &[0x4a, 0x89, 0x6c, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], rbp", |a| { a.st(mem_idx(R12, RAX, 0, 0), RBP, 8); }, &[0x49, 0x89, 0x2c, 0x04]),
        ("mov qword ptr [r12 + rsi*1], rbp", |a| { a.st(mem_idx(R12, RSI, 0, 0), RBP, 8); }, &[0x49, 0x89, 0x2c, 0x34]),
        ("mov qword ptr [r12 + r13*1], rbp", |a| { a.st(mem_idx(R12, R13, 0, 0), RBP, 8); }, &[0x4b, 0x89, 0x2c, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], rbp", |a| { a.st(mem_idx(R13, RCX, 0, 0), RBP, 8); }, &[0x49, 0x89, 0x6c, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], rbp", |a| { a.st(mem_idx(R13, R8, 0, 0), RBP, 8); }, &[0x4b, 0x89, 0x6c, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], rbp", |a| { a.st(mem_idx(R13, R15, 0, 0), RBP, 8); }, &[0x4b, 0x89, 0x6c, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], rbp", |a| { a.st(mem_idx(R15, RBP, 0, 0), RBP, 8); }, &[0x49, 0x89, 0x2c, 0x2f]),
        ("mov qword ptr [r15 + r12*1], rbp", |a| { a.st(mem_idx(R15, R12, 0, 0), RBP, 8); }, &[0x4b, 0x89, 0x2c, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], rbp", |a| { a.st(mem_idx(RBX, RCX, 1, 32), RBP, 8); }, &[0x48, 0x89, 0x6c, 0x4b, 0x20]),
        ("mov qword ptr [rax], r8", |a| { a.st(mem(RAX, 0), R8, 8); }, &[0x4c, 0x89, 0x00]),
        ("mov qword ptr [rax - 305419896], r8", |a| { a.st(mem(RAX, -305419896), R8, 8); }, &[0x4c, 0x89, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], r8", |a| { a.st(mem(RSP, 305419896), R8, 8); }, &[0x4c, 0x89, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], r8", |a| { a.st(mem(RBP, -129), R8, 8); }, &[0x4c, 0x89, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], r8", |a| { a.st(mem(R8, -128), R8, 8); }, &[0x4d, 0x89, 0x40, 0x80]),
        ("mov qword ptr [r12 + 128], r8", |a| { a.st(mem(R12, 128), R8, 8); }, &[0x4d, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], r8", |a| { a.st(mem(R13, 127), R8, 8); }, &[0x4d, 0x89, 0x45, 0x7f]),
        ("mov qword ptr [r15 - 1], r8", |a| { a.st(mem(R15, -1), R8, 8); }, &[0x4d, 0x89, 0x47, 0xff]),
        ("mov qword ptr [rbx + 1], r8", |a| { a.st(mem(RBX, 1), R8, 8); }, &[0x4c, 0x89, 0x43, 0x01]),
        ("mov qword ptr [rax + rax*1], r8", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R8, 8); }, &[0x4c, 0x89, 0x04, 0x00]),
        ("mov qword ptr [rax + rsi*1], r8", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R8, 8); }, &[0x4c, 0x89, 0x04, 0x30]),
        ("mov qword ptr [rax + r13*1], r8", |a| { a.st(mem_idx(RAX, R13, 0, 0), R8, 8); }, &[0x4e, 0x89, 0x04, 0x28]),
        ("mov qword ptr [rsp + rcx*1], r8", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R8, 8); }, &[0x4c, 0x89, 0x04, 0x0c]),
        ("mov qword ptr [rsp + r8*1], r8", |a| { a.st(mem_idx(RSP, R8, 0, 0), R8, 8); }, &[0x4e, 0x89, 0x04, 0x04]),
        ("mov qword ptr [rsp + r15*1], r8", |a| { a.st(mem_idx(RSP, R15, 0, 0), R8, 8); }, &[0x4e, 0x89, 0x04, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], r8", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R8, 8); }, &[0x4c, 0x89, 0x44, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], r8", |a| { a.st(mem_idx(RBP, R12, 0, 0), R8, 8); }, &[0x4e, 0x89, 0x44, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], r8", |a| { a.st(mem_idx(R12, RAX, 0, 0), R8, 8); }, &[0x4d, 0x89, 0x04, 0x04]),
        ("mov qword ptr [r12 + rsi*1], r8", |a| { a.st(mem_idx(R12, RSI, 0, 0), R8, 8); }, &[0x4d, 0x89, 0x04, 0x34]),
        ("mov qword ptr [r12 + r13*1], r8", |a| { a.st(mem_idx(R12, R13, 0, 0), R8, 8); }, &[0x4f, 0x89, 0x04, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], r8", |a| { a.st(mem_idx(R13, RCX, 0, 0), R8, 8); }, &[0x4d, 0x89, 0x44, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], r8", |a| { a.st(mem_idx(R13, R8, 0, 0), R8, 8); }, &[0x4f, 0x89, 0x44, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], r8", |a| { a.st(mem_idx(R13, R15, 0, 0), R8, 8); }, &[0x4f, 0x89, 0x44, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], r8", |a| { a.st(mem_idx(R15, RBP, 0, 0), R8, 8); }, &[0x4d, 0x89, 0x04, 0x2f]),
        ("mov qword ptr [r15 + r12*1], r8", |a| { a.st(mem_idx(R15, R12, 0, 0), R8, 8); }, &[0x4f, 0x89, 0x04, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], r8", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R8, 8); }, &[0x4c, 0x89, 0x44, 0x4b, 0x20]),
        ("mov qword ptr [rax], r15", |a| { a.st(mem(RAX, 0), R15, 8); }, &[0x4c, 0x89, 0x38]),
        ("mov qword ptr [rax - 305419896], r15", |a| { a.st(mem(RAX, -305419896), R15, 8); }, &[0x4c, 0x89, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], r15", |a| { a.st(mem(RSP, 305419896), R15, 8); }, &[0x4c, 0x89, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], r15", |a| { a.st(mem(RBP, -129), R15, 8); }, &[0x4c, 0x89, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], r15", |a| { a.st(mem(R8, -128), R15, 8); }, &[0x4d, 0x89, 0x78, 0x80]),
        ("mov qword ptr [r12 + 128], r15", |a| { a.st(mem(R12, 128), R15, 8); }, &[0x4d, 0x89, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], r15", |a| { a.st(mem(R13, 127), R15, 8); }, &[0x4d, 0x89, 0x7d, 0x7f]),
        ("mov qword ptr [r15 - 1], r15", |a| { a.st(mem(R15, -1), R15, 8); }, &[0x4d, 0x89, 0x7f, 0xff]),
        ("mov qword ptr [rbx + 1], r15", |a| { a.st(mem(RBX, 1), R15, 8); }, &[0x4c, 0x89, 0x7b, 0x01]),
        ("mov qword ptr [rax + rax*1], r15", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R15, 8); }, &[0x4c, 0x89, 0x3c, 0x00]),
        ("mov qword ptr [rax + rsi*1], r15", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R15, 8); }, &[0x4c, 0x89, 0x3c, 0x30]),
        ("mov qword ptr [rax + r13*1], r15", |a| { a.st(mem_idx(RAX, R13, 0, 0), R15, 8); }, &[0x4e, 0x89, 0x3c, 0x28]),
        ("mov qword ptr [rsp + rcx*1], r15", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R15, 8); }, &[0x4c, 0x89, 0x3c, 0x0c]),
        ("mov qword ptr [rsp + r8*1], r15", |a| { a.st(mem_idx(RSP, R8, 0, 0), R15, 8); }, &[0x4e, 0x89, 0x3c, 0x04]),
        ("mov qword ptr [rsp + r15*1], r15", |a| { a.st(mem_idx(RSP, R15, 0, 0), R15, 8); }, &[0x4e, 0x89, 0x3c, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], r15", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R15, 8); }, &[0x4c, 0x89, 0x7c, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], r15", |a| { a.st(mem_idx(RBP, R12, 0, 0), R15, 8); }, &[0x4e, 0x89, 0x7c, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], r15", |a| { a.st(mem_idx(R12, RAX, 0, 0), R15, 8); }, &[0x4d, 0x89, 0x3c, 0x04]),
        ("mov qword ptr [r12 + rsi*1], r15", |a| { a.st(mem_idx(R12, RSI, 0, 0), R15, 8); }, &[0x4d, 0x89, 0x3c, 0x34]),
        ("mov qword ptr [r12 + r13*1], r15", |a| { a.st(mem_idx(R12, R13, 0, 0), R15, 8); }, &[0x4f, 0x89, 0x3c, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], r15", |a| { a.st(mem_idx(R13, RCX, 0, 0), R15, 8); }, &[0x4d, 0x89, 0x7c, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], r15", |a| { a.st(mem_idx(R13, R8, 0, 0), R15, 8); }, &[0x4f, 0x89, 0x7c, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], r15", |a| { a.st(mem_idx(R13, R15, 0, 0), R15, 8); }, &[0x4f, 0x89, 0x7c, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], r15", |a| { a.st(mem_idx(R15, RBP, 0, 0), R15, 8); }, &[0x4d, 0x89, 0x3c, 0x2f]),
        ("mov qword ptr [r15 + r12*1], r15", |a| { a.st(mem_idx(R15, R12, 0, 0), R15, 8); }, &[0x4f, 0x89, 0x3c, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], r15", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R15, 8); }, &[0x4c, 0x89, 0x7c, 0x4b, 0x20]),
        ("mov qword ptr [rax], r12", |a| { a.st(mem(RAX, 0), R12, 8); }, &[0x4c, 0x89, 0x20]),
        ("mov qword ptr [rax - 305419896], r12", |a| { a.st(mem(RAX, -305419896), R12, 8); }, &[0x4c, 0x89, 0xa0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov qword ptr [rsp + 305419896], r12", |a| { a.st(mem(RSP, 305419896), R12, 8); }, &[0x4c, 0x89, 0xa4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("mov qword ptr [rbp - 129], r12", |a| { a.st(mem(RBP, -129), R12, 8); }, &[0x4c, 0x89, 0xa5, 0x7f, 0xff, 0xff, 0xff]),
        ("mov qword ptr [r8 - 128], r12", |a| { a.st(mem(R8, -128), R12, 8); }, &[0x4d, 0x89, 0x60, 0x80]),
        ("mov qword ptr [r12 + 128], r12", |a| { a.st(mem(R12, 128), R12, 8); }, &[0x4d, 0x89, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("mov qword ptr [r13 + 127], r12", |a| { a.st(mem(R13, 127), R12, 8); }, &[0x4d, 0x89, 0x65, 0x7f]),
        ("mov qword ptr [r15 - 1], r12", |a| { a.st(mem(R15, -1), R12, 8); }, &[0x4d, 0x89, 0x67, 0xff]),
        ("mov qword ptr [rbx + 1], r12", |a| { a.st(mem(RBX, 1), R12, 8); }, &[0x4c, 0x89, 0x63, 0x01]),
        ("mov qword ptr [rax + rax*1], r12", |a| { a.st(mem_idx(RAX, RAX, 0, 0), R12, 8); }, &[0x4c, 0x89, 0x24, 0x00]),
        ("mov qword ptr [rax + rsi*1], r12", |a| { a.st(mem_idx(RAX, RSI, 0, 0), R12, 8); }, &[0x4c, 0x89, 0x24, 0x30]),
        ("mov qword ptr [rax + r13*1], r12", |a| { a.st(mem_idx(RAX, R13, 0, 0), R12, 8); }, &[0x4e, 0x89, 0x24, 0x28]),
        ("mov qword ptr [rsp + rcx*1], r12", |a| { a.st(mem_idx(RSP, RCX, 0, 0), R12, 8); }, &[0x4c, 0x89, 0x24, 0x0c]),
        ("mov qword ptr [rsp + r8*1], r12", |a| { a.st(mem_idx(RSP, R8, 0, 0), R12, 8); }, &[0x4e, 0x89, 0x24, 0x04]),
        ("mov qword ptr [rsp + r15*1], r12", |a| { a.st(mem_idx(RSP, R15, 0, 0), R12, 8); }, &[0x4e, 0x89, 0x24, 0x3c]),
        ("mov qword ptr [rbp + rbp*1], r12", |a| { a.st(mem_idx(RBP, RBP, 0, 0), R12, 8); }, &[0x4c, 0x89, 0x64, 0x2d, 0x00]),
        ("mov qword ptr [rbp + r12*1], r12", |a| { a.st(mem_idx(RBP, R12, 0, 0), R12, 8); }, &[0x4e, 0x89, 0x64, 0x25, 0x00]),
        ("mov qword ptr [r12 + rax*1], r12", |a| { a.st(mem_idx(R12, RAX, 0, 0), R12, 8); }, &[0x4d, 0x89, 0x24, 0x04]),
        ("mov qword ptr [r12 + rsi*1], r12", |a| { a.st(mem_idx(R12, RSI, 0, 0), R12, 8); }, &[0x4d, 0x89, 0x24, 0x34]),
        ("mov qword ptr [r12 + r13*1], r12", |a| { a.st(mem_idx(R12, R13, 0, 0), R12, 8); }, &[0x4f, 0x89, 0x24, 0x2c]),
        ("mov qword ptr [r13 + rcx*1], r12", |a| { a.st(mem_idx(R13, RCX, 0, 0), R12, 8); }, &[0x4d, 0x89, 0x64, 0x0d, 0x00]),
        ("mov qword ptr [r13 + r8*1], r12", |a| { a.st(mem_idx(R13, R8, 0, 0), R12, 8); }, &[0x4f, 0x89, 0x64, 0x05, 0x00]),
        ("mov qword ptr [r13 + r15*1], r12", |a| { a.st(mem_idx(R13, R15, 0, 0), R12, 8); }, &[0x4f, 0x89, 0x64, 0x3d, 0x00]),
        ("mov qword ptr [r15 + rbp*1], r12", |a| { a.st(mem_idx(R15, RBP, 0, 0), R12, 8); }, &[0x4d, 0x89, 0x24, 0x2f]),
        ("mov qword ptr [r15 + r12*1], r12", |a| { a.st(mem_idx(R15, R12, 0, 0), R12, 8); }, &[0x4f, 0x89, 0x24, 0x27]),
        ("mov qword ptr [rbx + rcx*2 + 32], r12", |a| { a.st(mem_idx(RBX, RCX, 1, 32), R12, 8); }, &[0x4c, 0x89, 0x64, 0x4b, 0x20]),
    ];
    check_encodings(cases);
}

    /// Lea with every addressing shape.
#[rustfmt::skip]
#[test]
fn sse_forms_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("movdqu xmm0, [rbx + 1296]", |a| { a.xmm_ld(0, mem(RBX, 1296)); }, &[0xf3, 0x0f, 0x6f, 0x83, 0x10, 0x05, 0x00, 0x00]),
        ("movdqu xmm1, [rbx + 8]", |a| { a.xmm_ld(1, mem(RBX, 8)); }, &[0xf3, 0x0f, 0x6f, 0x4b, 0x08]),
        ("movdqu xmm1, [rdx + rdi]", |a| { a.xmm_ld(1, mem_idx(RDX, RDI, 0, 0)); }, &[0xf3, 0x0f, 0x6f, 0x0c, 0x3a]),
        ("movdqu [rbx + 1296], xmm0", |a| { a.xmm_st(mem(RBX, 1296), 0); }, &[0xf3, 0x0f, 0x7f, 0x83, 0x10, 0x05, 0x00, 0x00]),
        ("movdqu [rdx + rdi + 8], xmm1", |a| { a.xmm_st(mem_idx(RDX, RDI, 0, 8), 1); }, &[0xf3, 0x0f, 0x7f, 0x4c, 0x3a, 0x08]),
        ("pxor xmm0, xmm1", |a| { a.xmm_op(0xef, 0, 1); }, &[0x66, 0x0f, 0xef, 0xc1]),
        ("pcmpeqb xmm0, xmm1", |a| { a.xmm_op(0x74, 0, 1); }, &[0x66, 0x0f, 0x74, 0xc1]),
        ("psubq xmm0, xmm1", |a| { a.xmm_op(0xfb, 0, 1); }, &[0x66, 0x0f, 0xfb, 0xc1]),
        ("pmovmskb r9d, xmm1", |a| { a.pmovmskb(R9, 1); }, &[0x66, 0x44, 0x0f, 0xd7, 0xc9]),
        ("pmovmskb eax, xmm1", |a| { a.pmovmskb(RAX, 1); }, &[0x66, 0x0f, 0xd7, 0xc1]),
        ("pshufd xmm0, xmm1, 0x1b", |a| { a.pshufd(0, 1, 0x1b); }, &[0x66, 0x0f, 0x70, 0xc1, 0x1b]),
        ("pshufd xmm9, xmm12, 0xe4", |a| { a.pshufd(9, 12, 0xe4); }, &[0x66, 0x45, 0x0f, 0x70, 0xcc, 0xe4]),
    ];
    check_encodings(cases);
}

    #[rustfmt::skip]
#[test]
fn lea_matches_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("lea rax, [rax]", |a| { a.lea(RAX, mem(RAX, 0)); }, &[0x48, 0x8d, 0x00]),
        ("lea rax, [rax + 128]", |a| { a.lea(RAX, mem(RAX, 128)); }, &[0x48, 0x8d, 0x80, 0x80, 0x00, 0x00, 0x00]),
        ("lea rax, [rax - 305419896]", |a| { a.lea(RAX, mem(RAX, -305419896)); }, &[0x48, 0x8d, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea rax, [rsp + 127]", |a| { a.lea(RAX, mem(RSP, 127)); }, &[0x48, 0x8d, 0x44, 0x24, 0x7f]),
        ("lea rax, [rsp + 305419896]", |a| { a.lea(RAX, mem(RSP, 305419896)); }, &[0x48, 0x8d, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("lea rax, [rbp - 1]", |a| { a.lea(RAX, mem(RBP, -1)); }, &[0x48, 0x8d, 0x45, 0xff]),
        ("lea rax, [rbp - 129]", |a| { a.lea(RAX, mem(RBP, -129)); }, &[0x48, 0x8d, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("lea rax, [r8 + 1]", |a| { a.lea(RAX, mem(R8, 1)); }, &[0x49, 0x8d, 0x40, 0x01]),
        ("lea rax, [r8 - 128]", |a| { a.lea(RAX, mem(R8, -128)); }, &[0x49, 0x8d, 0x40, 0x80]),
        ("lea rax, [r12]", |a| { a.lea(RAX, mem(R12, 0)); }, &[0x49, 0x8d, 0x04, 0x24]),
        ("lea rax, [r12 + 128]", |a| { a.lea(RAX, mem(R12, 128)); }, &[0x49, 0x8d, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("lea rax, [r12 - 305419896]", |a| { a.lea(RAX, mem(R12, -305419896)); }, &[0x49, 0x8d, 0x84, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea rax, [r13 + 127]", |a| { a.lea(RAX, mem(R13, 127)); }, &[0x49, 0x8d, 0x45, 0x7f]),
        ("lea rax, [r13 + 305419896]", |a| { a.lea(RAX, mem(R13, 305419896)); }, &[0x49, 0x8d, 0x85, 0x78, 0x56, 0x34, 0x12]),
        ("lea rax, [r15 - 1]", |a| { a.lea(RAX, mem(R15, -1)); }, &[0x49, 0x8d, 0x47, 0xff]),
        ("lea rax, [r15 - 129]", |a| { a.lea(RAX, mem(R15, -129)); }, &[0x49, 0x8d, 0x87, 0x7f, 0xff, 0xff, 0xff]),
        ("lea rax, [rbx + 1]", |a| { a.lea(RAX, mem(RBX, 1)); }, &[0x48, 0x8d, 0x43, 0x01]),
        ("lea rax, [rbx - 128]", |a| { a.lea(RAX, mem(RBX, -128)); }, &[0x48, 0x8d, 0x43, 0x80]),
        ("lea rax, [rax + rax*1]", |a| { a.lea(RAX, mem_idx(RAX, RAX, 0, 0)); }, &[0x48, 0x8d, 0x04, 0x00]),
        ("lea rax, [rax + rcx*8]", |a| { a.lea(RAX, mem_idx(RAX, RCX, 3, 0)); }, &[0x48, 0x8d, 0x04, 0xc8]),
        ("lea rax, [rax + rsi*1]", |a| { a.lea(RAX, mem_idx(RAX, RSI, 0, 0)); }, &[0x48, 0x8d, 0x04, 0x30]),
        ("lea rax, [rax + r8*8]", |a| { a.lea(RAX, mem_idx(RAX, R8, 3, 0)); }, &[0x4a, 0x8d, 0x04, 0xc0]),
        ("lea rax, [rax + r13*1]", |a| { a.lea(RAX, mem_idx(RAX, R13, 0, 0)); }, &[0x4a, 0x8d, 0x04, 0x28]),
        ("lea rax, [rax + r15*8]", |a| { a.lea(RAX, mem_idx(RAX, R15, 3, 0)); }, &[0x4a, 0x8d, 0x04, 0xf8]),
        ("lea rax, [rsp + rcx*1]", |a| { a.lea(RAX, mem_idx(RSP, RCX, 0, 0)); }, &[0x48, 0x8d, 0x04, 0x0c]),
        ("lea rax, [rsp + rbp*8]", |a| { a.lea(RAX, mem_idx(RSP, RBP, 3, 0)); }, &[0x48, 0x8d, 0x04, 0xec]),
        ("lea rax, [rsp + r8*1]", |a| { a.lea(RAX, mem_idx(RSP, R8, 0, 0)); }, &[0x4a, 0x8d, 0x04, 0x04]),
        ("lea rax, [rsp + r12*8]", |a| { a.lea(RAX, mem_idx(RSP, R12, 3, 0)); }, &[0x4a, 0x8d, 0x04, 0xe4]),
        ("lea rax, [rsp + r15*1]", |a| { a.lea(RAX, mem_idx(RSP, R15, 0, 0)); }, &[0x4a, 0x8d, 0x04, 0x3c]),
        ("lea rax, [rbp + rax*8]", |a| { a.lea(RAX, mem_idx(RBP, RAX, 3, 0)); }, &[0x48, 0x8d, 0x44, 0xc5, 0x00]),
        ("lea rax, [rbp + rbp*1]", |a| { a.lea(RAX, mem_idx(RBP, RBP, 0, 0)); }, &[0x48, 0x8d, 0x44, 0x2d, 0x00]),
        ("lea rax, [rbp + rsi*8]", |a| { a.lea(RAX, mem_idx(RBP, RSI, 3, 0)); }, &[0x48, 0x8d, 0x44, 0xf5, 0x00]),
        ("lea rax, [rbp + r12*1]", |a| { a.lea(RAX, mem_idx(RBP, R12, 0, 0)); }, &[0x4a, 0x8d, 0x44, 0x25, 0x00]),
        ("lea rax, [rbp + r13*8]", |a| { a.lea(RAX, mem_idx(RBP, R13, 3, 0)); }, &[0x4a, 0x8d, 0x44, 0xed, 0x00]),
        ("lea rax, [r12 + rax*1]", |a| { a.lea(RAX, mem_idx(R12, RAX, 0, 0)); }, &[0x49, 0x8d, 0x04, 0x04]),
        ("lea rax, [r12 + rcx*8]", |a| { a.lea(RAX, mem_idx(R12, RCX, 3, 0)); }, &[0x49, 0x8d, 0x04, 0xcc]),
        ("lea rax, [r12 + rsi*1]", |a| { a.lea(RAX, mem_idx(R12, RSI, 0, 0)); }, &[0x49, 0x8d, 0x04, 0x34]),
        ("lea rax, [r12 + r8*8]", |a| { a.lea(RAX, mem_idx(R12, R8, 3, 0)); }, &[0x4b, 0x8d, 0x04, 0xc4]),
        ("lea rax, [r12 + r13*1]", |a| { a.lea(RAX, mem_idx(R12, R13, 0, 0)); }, &[0x4b, 0x8d, 0x04, 0x2c]),
        ("lea rax, [r12 + r15*8]", |a| { a.lea(RAX, mem_idx(R12, R15, 3, 0)); }, &[0x4b, 0x8d, 0x04, 0xfc]),
        ("lea rax, [r13 + rcx*1]", |a| { a.lea(RAX, mem_idx(R13, RCX, 0, 0)); }, &[0x49, 0x8d, 0x44, 0x0d, 0x00]),
        ("lea rax, [r13 + rbp*8]", |a| { a.lea(RAX, mem_idx(R13, RBP, 3, 0)); }, &[0x49, 0x8d, 0x44, 0xed, 0x00]),
        ("lea rax, [r13 + r8*1]", |a| { a.lea(RAX, mem_idx(R13, R8, 0, 0)); }, &[0x4b, 0x8d, 0x44, 0x05, 0x00]),
        ("lea rax, [r13 + r12*8]", |a| { a.lea(RAX, mem_idx(R13, R12, 3, 0)); }, &[0x4b, 0x8d, 0x44, 0xe5, 0x00]),
        ("lea rax, [r13 + r15*1]", |a| { a.lea(RAX, mem_idx(R13, R15, 0, 0)); }, &[0x4b, 0x8d, 0x44, 0x3d, 0x00]),
        ("lea rax, [r15 + rax*8]", |a| { a.lea(RAX, mem_idx(R15, RAX, 3, 0)); }, &[0x49, 0x8d, 0x04, 0xc7]),
        ("lea rax, [r15 + rbp*1]", |a| { a.lea(RAX, mem_idx(R15, RBP, 0, 0)); }, &[0x49, 0x8d, 0x04, 0x2f]),
        ("lea rax, [r15 + rsi*8]", |a| { a.lea(RAX, mem_idx(R15, RSI, 3, 0)); }, &[0x49, 0x8d, 0x04, 0xf7]),
        ("lea rax, [r15 + r12*1]", |a| { a.lea(RAX, mem_idx(R15, R12, 0, 0)); }, &[0x4b, 0x8d, 0x04, 0x27]),
        ("lea rax, [r15 + r13*8]", |a| { a.lea(RAX, mem_idx(R15, R13, 3, 0)); }, &[0x4b, 0x8d, 0x04, 0xef]),
        ("lea rax, [rbx + rcx*2 + 32]", |a| { a.lea(RAX, mem_idx(RBX, RCX, 1, 32)); }, &[0x48, 0x8d, 0x44, 0x4b, 0x20]),
        ("lea r9, [rax]", |a| { a.lea(R9, mem(RAX, 0)); }, &[0x4c, 0x8d, 0x08]),
        ("lea r9, [rax + 128]", |a| { a.lea(R9, mem(RAX, 128)); }, &[0x4c, 0x8d, 0x88, 0x80, 0x00, 0x00, 0x00]),
        ("lea r9, [rax - 305419896]", |a| { a.lea(R9, mem(RAX, -305419896)); }, &[0x4c, 0x8d, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea r9, [rsp + 127]", |a| { a.lea(R9, mem(RSP, 127)); }, &[0x4c, 0x8d, 0x4c, 0x24, 0x7f]),
        ("lea r9, [rsp + 305419896]", |a| { a.lea(R9, mem(RSP, 305419896)); }, &[0x4c, 0x8d, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("lea r9, [rbp - 1]", |a| { a.lea(R9, mem(RBP, -1)); }, &[0x4c, 0x8d, 0x4d, 0xff]),
        ("lea r9, [rbp - 129]", |a| { a.lea(R9, mem(RBP, -129)); }, &[0x4c, 0x8d, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("lea r9, [r8 + 1]", |a| { a.lea(R9, mem(R8, 1)); }, &[0x4d, 0x8d, 0x48, 0x01]),
        ("lea r9, [r8 - 128]", |a| { a.lea(R9, mem(R8, -128)); }, &[0x4d, 0x8d, 0x48, 0x80]),
        ("lea r9, [r12]", |a| { a.lea(R9, mem(R12, 0)); }, &[0x4d, 0x8d, 0x0c, 0x24]),
        ("lea r9, [r12 + 128]", |a| { a.lea(R9, mem(R12, 128)); }, &[0x4d, 0x8d, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("lea r9, [r12 - 305419896]", |a| { a.lea(R9, mem(R12, -305419896)); }, &[0x4d, 0x8d, 0x8c, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea r9, [r13 + 127]", |a| { a.lea(R9, mem(R13, 127)); }, &[0x4d, 0x8d, 0x4d, 0x7f]),
        ("lea r9, [r13 + 305419896]", |a| { a.lea(R9, mem(R13, 305419896)); }, &[0x4d, 0x8d, 0x8d, 0x78, 0x56, 0x34, 0x12]),
        ("lea r9, [r15 - 1]", |a| { a.lea(R9, mem(R15, -1)); }, &[0x4d, 0x8d, 0x4f, 0xff]),
        ("lea r9, [r15 - 129]", |a| { a.lea(R9, mem(R15, -129)); }, &[0x4d, 0x8d, 0x8f, 0x7f, 0xff, 0xff, 0xff]),
        ("lea r9, [rbx + 1]", |a| { a.lea(R9, mem(RBX, 1)); }, &[0x4c, 0x8d, 0x4b, 0x01]),
        ("lea r9, [rbx - 128]", |a| { a.lea(R9, mem(RBX, -128)); }, &[0x4c, 0x8d, 0x4b, 0x80]),
        ("lea r9, [rax + rax*1]", |a| { a.lea(R9, mem_idx(RAX, RAX, 0, 0)); }, &[0x4c, 0x8d, 0x0c, 0x00]),
        ("lea r9, [rax + rcx*8]", |a| { a.lea(R9, mem_idx(RAX, RCX, 3, 0)); }, &[0x4c, 0x8d, 0x0c, 0xc8]),
        ("lea r9, [rax + rsi*1]", |a| { a.lea(R9, mem_idx(RAX, RSI, 0, 0)); }, &[0x4c, 0x8d, 0x0c, 0x30]),
        ("lea r9, [rax + r8*8]", |a| { a.lea(R9, mem_idx(RAX, R8, 3, 0)); }, &[0x4e, 0x8d, 0x0c, 0xc0]),
        ("lea r9, [rax + r13*1]", |a| { a.lea(R9, mem_idx(RAX, R13, 0, 0)); }, &[0x4e, 0x8d, 0x0c, 0x28]),
        ("lea r9, [rax + r15*8]", |a| { a.lea(R9, mem_idx(RAX, R15, 3, 0)); }, &[0x4e, 0x8d, 0x0c, 0xf8]),
        ("lea r9, [rsp + rcx*1]", |a| { a.lea(R9, mem_idx(RSP, RCX, 0, 0)); }, &[0x4c, 0x8d, 0x0c, 0x0c]),
        ("lea r9, [rsp + rbp*8]", |a| { a.lea(R9, mem_idx(RSP, RBP, 3, 0)); }, &[0x4c, 0x8d, 0x0c, 0xec]),
        ("lea r9, [rsp + r8*1]", |a| { a.lea(R9, mem_idx(RSP, R8, 0, 0)); }, &[0x4e, 0x8d, 0x0c, 0x04]),
        ("lea r9, [rsp + r12*8]", |a| { a.lea(R9, mem_idx(RSP, R12, 3, 0)); }, &[0x4e, 0x8d, 0x0c, 0xe4]),
        ("lea r9, [rsp + r15*1]", |a| { a.lea(R9, mem_idx(RSP, R15, 0, 0)); }, &[0x4e, 0x8d, 0x0c, 0x3c]),
        ("lea r9, [rbp + rax*8]", |a| { a.lea(R9, mem_idx(RBP, RAX, 3, 0)); }, &[0x4c, 0x8d, 0x4c, 0xc5, 0x00]),
        ("lea r9, [rbp + rbp*1]", |a| { a.lea(R9, mem_idx(RBP, RBP, 0, 0)); }, &[0x4c, 0x8d, 0x4c, 0x2d, 0x00]),
        ("lea r9, [rbp + rsi*8]", |a| { a.lea(R9, mem_idx(RBP, RSI, 3, 0)); }, &[0x4c, 0x8d, 0x4c, 0xf5, 0x00]),
        ("lea r9, [rbp + r12*1]", |a| { a.lea(R9, mem_idx(RBP, R12, 0, 0)); }, &[0x4e, 0x8d, 0x4c, 0x25, 0x00]),
        ("lea r9, [rbp + r13*8]", |a| { a.lea(R9, mem_idx(RBP, R13, 3, 0)); }, &[0x4e, 0x8d, 0x4c, 0xed, 0x00]),
        ("lea r9, [r12 + rax*1]", |a| { a.lea(R9, mem_idx(R12, RAX, 0, 0)); }, &[0x4d, 0x8d, 0x0c, 0x04]),
        ("lea r9, [r12 + rcx*8]", |a| { a.lea(R9, mem_idx(R12, RCX, 3, 0)); }, &[0x4d, 0x8d, 0x0c, 0xcc]),
        ("lea r9, [r12 + rsi*1]", |a| { a.lea(R9, mem_idx(R12, RSI, 0, 0)); }, &[0x4d, 0x8d, 0x0c, 0x34]),
        ("lea r9, [r12 + r8*8]", |a| { a.lea(R9, mem_idx(R12, R8, 3, 0)); }, &[0x4f, 0x8d, 0x0c, 0xc4]),
        ("lea r9, [r12 + r13*1]", |a| { a.lea(R9, mem_idx(R12, R13, 0, 0)); }, &[0x4f, 0x8d, 0x0c, 0x2c]),
        ("lea r9, [r12 + r15*8]", |a| { a.lea(R9, mem_idx(R12, R15, 3, 0)); }, &[0x4f, 0x8d, 0x0c, 0xfc]),
        ("lea r9, [r13 + rcx*1]", |a| { a.lea(R9, mem_idx(R13, RCX, 0, 0)); }, &[0x4d, 0x8d, 0x4c, 0x0d, 0x00]),
        ("lea r9, [r13 + rbp*8]", |a| { a.lea(R9, mem_idx(R13, RBP, 3, 0)); }, &[0x4d, 0x8d, 0x4c, 0xed, 0x00]),
        ("lea r9, [r13 + r8*1]", |a| { a.lea(R9, mem_idx(R13, R8, 0, 0)); }, &[0x4f, 0x8d, 0x4c, 0x05, 0x00]),
        ("lea r9, [r13 + r12*8]", |a| { a.lea(R9, mem_idx(R13, R12, 3, 0)); }, &[0x4f, 0x8d, 0x4c, 0xe5, 0x00]),
        ("lea r9, [r13 + r15*1]", |a| { a.lea(R9, mem_idx(R13, R15, 0, 0)); }, &[0x4f, 0x8d, 0x4c, 0x3d, 0x00]),
        ("lea r9, [r15 + rax*8]", |a| { a.lea(R9, mem_idx(R15, RAX, 3, 0)); }, &[0x4d, 0x8d, 0x0c, 0xc7]),
        ("lea r9, [r15 + rbp*1]", |a| { a.lea(R9, mem_idx(R15, RBP, 0, 0)); }, &[0x4d, 0x8d, 0x0c, 0x2f]),
        ("lea r9, [r15 + rsi*8]", |a| { a.lea(R9, mem_idx(R15, RSI, 3, 0)); }, &[0x4d, 0x8d, 0x0c, 0xf7]),
        ("lea r9, [r15 + r12*1]", |a| { a.lea(R9, mem_idx(R15, R12, 0, 0)); }, &[0x4f, 0x8d, 0x0c, 0x27]),
        ("lea r9, [r15 + r13*8]", |a| { a.lea(R9, mem_idx(R15, R13, 3, 0)); }, &[0x4f, 0x8d, 0x0c, 0xef]),
        ("lea r9, [rbx + rcx*2 + 32]", |a| { a.lea(R9, mem_idx(RBX, RCX, 1, 32)); }, &[0x4c, 0x8d, 0x4c, 0x4b, 0x20]),
        ("lea r15, [rax]", |a| { a.lea(R15, mem(RAX, 0)); }, &[0x4c, 0x8d, 0x38]),
        ("lea r15, [rax + 128]", |a| { a.lea(R15, mem(RAX, 128)); }, &[0x4c, 0x8d, 0xb8, 0x80, 0x00, 0x00, 0x00]),
        ("lea r15, [rax - 305419896]", |a| { a.lea(R15, mem(RAX, -305419896)); }, &[0x4c, 0x8d, 0xb8, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea r15, [rsp + 127]", |a| { a.lea(R15, mem(RSP, 127)); }, &[0x4c, 0x8d, 0x7c, 0x24, 0x7f]),
        ("lea r15, [rsp + 305419896]", |a| { a.lea(R15, mem(RSP, 305419896)); }, &[0x4c, 0x8d, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("lea r15, [rbp - 1]", |a| { a.lea(R15, mem(RBP, -1)); }, &[0x4c, 0x8d, 0x7d, 0xff]),
        ("lea r15, [rbp - 129]", |a| { a.lea(R15, mem(RBP, -129)); }, &[0x4c, 0x8d, 0xbd, 0x7f, 0xff, 0xff, 0xff]),
        ("lea r15, [r8 + 1]", |a| { a.lea(R15, mem(R8, 1)); }, &[0x4d, 0x8d, 0x78, 0x01]),
        ("lea r15, [r8 - 128]", |a| { a.lea(R15, mem(R8, -128)); }, &[0x4d, 0x8d, 0x78, 0x80]),
        ("lea r15, [r12]", |a| { a.lea(R15, mem(R12, 0)); }, &[0x4d, 0x8d, 0x3c, 0x24]),
        ("lea r15, [r12 + 128]", |a| { a.lea(R15, mem(R12, 128)); }, &[0x4d, 0x8d, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("lea r15, [r12 - 305419896]", |a| { a.lea(R15, mem(R12, -305419896)); }, &[0x4d, 0x8d, 0xbc, 0x24, 0x88, 0xa9, 0xcb, 0xed]),
        ("lea r15, [r13 + 127]", |a| { a.lea(R15, mem(R13, 127)); }, &[0x4d, 0x8d, 0x7d, 0x7f]),
        ("lea r15, [r13 + 305419896]", |a| { a.lea(R15, mem(R13, 305419896)); }, &[0x4d, 0x8d, 0xbd, 0x78, 0x56, 0x34, 0x12]),
        ("lea r15, [r15 - 1]", |a| { a.lea(R15, mem(R15, -1)); }, &[0x4d, 0x8d, 0x7f, 0xff]),
        ("lea r15, [r15 - 129]", |a| { a.lea(R15, mem(R15, -129)); }, &[0x4d, 0x8d, 0xbf, 0x7f, 0xff, 0xff, 0xff]),
        ("lea r15, [rbx + 1]", |a| { a.lea(R15, mem(RBX, 1)); }, &[0x4c, 0x8d, 0x7b, 0x01]),
        ("lea r15, [rbx - 128]", |a| { a.lea(R15, mem(RBX, -128)); }, &[0x4c, 0x8d, 0x7b, 0x80]),
        ("lea r15, [rax + rax*1]", |a| { a.lea(R15, mem_idx(RAX, RAX, 0, 0)); }, &[0x4c, 0x8d, 0x3c, 0x00]),
        ("lea r15, [rax + rcx*8]", |a| { a.lea(R15, mem_idx(RAX, RCX, 3, 0)); }, &[0x4c, 0x8d, 0x3c, 0xc8]),
        ("lea r15, [rax + rsi*1]", |a| { a.lea(R15, mem_idx(RAX, RSI, 0, 0)); }, &[0x4c, 0x8d, 0x3c, 0x30]),
        ("lea r15, [rax + r8*8]", |a| { a.lea(R15, mem_idx(RAX, R8, 3, 0)); }, &[0x4e, 0x8d, 0x3c, 0xc0]),
        ("lea r15, [rax + r13*1]", |a| { a.lea(R15, mem_idx(RAX, R13, 0, 0)); }, &[0x4e, 0x8d, 0x3c, 0x28]),
        ("lea r15, [rax + r15*8]", |a| { a.lea(R15, mem_idx(RAX, R15, 3, 0)); }, &[0x4e, 0x8d, 0x3c, 0xf8]),
        ("lea r15, [rsp + rcx*1]", |a| { a.lea(R15, mem_idx(RSP, RCX, 0, 0)); }, &[0x4c, 0x8d, 0x3c, 0x0c]),
        ("lea r15, [rsp + rbp*8]", |a| { a.lea(R15, mem_idx(RSP, RBP, 3, 0)); }, &[0x4c, 0x8d, 0x3c, 0xec]),
        ("lea r15, [rsp + r8*1]", |a| { a.lea(R15, mem_idx(RSP, R8, 0, 0)); }, &[0x4e, 0x8d, 0x3c, 0x04]),
        ("lea r15, [rsp + r12*8]", |a| { a.lea(R15, mem_idx(RSP, R12, 3, 0)); }, &[0x4e, 0x8d, 0x3c, 0xe4]),
        ("lea r15, [rsp + r15*1]", |a| { a.lea(R15, mem_idx(RSP, R15, 0, 0)); }, &[0x4e, 0x8d, 0x3c, 0x3c]),
        ("lea r15, [rbp + rax*8]", |a| { a.lea(R15, mem_idx(RBP, RAX, 3, 0)); }, &[0x4c, 0x8d, 0x7c, 0xc5, 0x00]),
        ("lea r15, [rbp + rbp*1]", |a| { a.lea(R15, mem_idx(RBP, RBP, 0, 0)); }, &[0x4c, 0x8d, 0x7c, 0x2d, 0x00]),
        ("lea r15, [rbp + rsi*8]", |a| { a.lea(R15, mem_idx(RBP, RSI, 3, 0)); }, &[0x4c, 0x8d, 0x7c, 0xf5, 0x00]),
        ("lea r15, [rbp + r12*1]", |a| { a.lea(R15, mem_idx(RBP, R12, 0, 0)); }, &[0x4e, 0x8d, 0x7c, 0x25, 0x00]),
        ("lea r15, [rbp + r13*8]", |a| { a.lea(R15, mem_idx(RBP, R13, 3, 0)); }, &[0x4e, 0x8d, 0x7c, 0xed, 0x00]),
        ("lea r15, [r12 + rax*1]", |a| { a.lea(R15, mem_idx(R12, RAX, 0, 0)); }, &[0x4d, 0x8d, 0x3c, 0x04]),
        ("lea r15, [r12 + rcx*8]", |a| { a.lea(R15, mem_idx(R12, RCX, 3, 0)); }, &[0x4d, 0x8d, 0x3c, 0xcc]),
        ("lea r15, [r12 + rsi*1]", |a| { a.lea(R15, mem_idx(R12, RSI, 0, 0)); }, &[0x4d, 0x8d, 0x3c, 0x34]),
        ("lea r15, [r12 + r8*8]", |a| { a.lea(R15, mem_idx(R12, R8, 3, 0)); }, &[0x4f, 0x8d, 0x3c, 0xc4]),
        ("lea r15, [r12 + r13*1]", |a| { a.lea(R15, mem_idx(R12, R13, 0, 0)); }, &[0x4f, 0x8d, 0x3c, 0x2c]),
        ("lea r15, [r12 + r15*8]", |a| { a.lea(R15, mem_idx(R12, R15, 3, 0)); }, &[0x4f, 0x8d, 0x3c, 0xfc]),
        ("lea r15, [r13 + rcx*1]", |a| { a.lea(R15, mem_idx(R13, RCX, 0, 0)); }, &[0x4d, 0x8d, 0x7c, 0x0d, 0x00]),
        ("lea r15, [r13 + rbp*8]", |a| { a.lea(R15, mem_idx(R13, RBP, 3, 0)); }, &[0x4d, 0x8d, 0x7c, 0xed, 0x00]),
        ("lea r15, [r13 + r8*1]", |a| { a.lea(R15, mem_idx(R13, R8, 0, 0)); }, &[0x4f, 0x8d, 0x7c, 0x05, 0x00]),
        ("lea r15, [r13 + r12*8]", |a| { a.lea(R15, mem_idx(R13, R12, 3, 0)); }, &[0x4f, 0x8d, 0x7c, 0xe5, 0x00]),
        ("lea r15, [r13 + r15*1]", |a| { a.lea(R15, mem_idx(R13, R15, 0, 0)); }, &[0x4f, 0x8d, 0x7c, 0x3d, 0x00]),
        ("lea r15, [r15 + rax*8]", |a| { a.lea(R15, mem_idx(R15, RAX, 3, 0)); }, &[0x4d, 0x8d, 0x3c, 0xc7]),
        ("lea r15, [r15 + rbp*1]", |a| { a.lea(R15, mem_idx(R15, RBP, 0, 0)); }, &[0x4d, 0x8d, 0x3c, 0x2f]),
        ("lea r15, [r15 + rsi*8]", |a| { a.lea(R15, mem_idx(R15, RSI, 3, 0)); }, &[0x4d, 0x8d, 0x3c, 0xf7]),
        ("lea r15, [r15 + r12*1]", |a| { a.lea(R15, mem_idx(R15, R12, 0, 0)); }, &[0x4f, 0x8d, 0x3c, 0x27]),
        ("lea r15, [r15 + r13*8]", |a| { a.lea(R15, mem_idx(R15, R13, 3, 0)); }, &[0x4f, 0x8d, 0x3c, 0xef]),
        ("lea r15, [rbx + rcx*2 + 32]", |a| { a.lea(R15, mem_idx(RBX, RCX, 1, 32)); }, &[0x4c, 0x8d, 0x7c, 0x4b, 0x20]),
    ];
    check_encodings(cases);
}

    /// Cmp [m], r, cmp byte [m], imm8 and mov word [m], imm16.
#[rustfmt::skip]
#[test]
fn compare_and_immediate_store_to_memory_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("cmp qword ptr [rax], rax", |a| { a.cmp_mr(mem(RAX, 0), RAX); }, &[0x48, 0x39, 0x00]),
        ("cmp qword ptr [rax - 305419896], rax", |a| { a.cmp_mr(mem(RAX, -305419896), RAX); }, &[0x48, 0x39, 0x80, 0x88, 0xa9, 0xcb, 0xed]),
        ("cmp qword ptr [rsp + 305419896], rax", |a| { a.cmp_mr(mem(RSP, 305419896), RAX); }, &[0x48, 0x39, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("cmp qword ptr [rbp - 129], rax", |a| { a.cmp_mr(mem(RBP, -129), RAX); }, &[0x48, 0x39, 0x85, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp qword ptr [r8 - 128], rax", |a| { a.cmp_mr(mem(R8, -128), RAX); }, &[0x49, 0x39, 0x40, 0x80]),
        ("cmp qword ptr [r12 + 128], rax", |a| { a.cmp_mr(mem(R12, 128), RAX); }, &[0x49, 0x39, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("cmp qword ptr [r13 + 127], rax", |a| { a.cmp_mr(mem(R13, 127), RAX); }, &[0x49, 0x39, 0x45, 0x7f]),
        ("cmp qword ptr [r15 - 1], rax", |a| { a.cmp_mr(mem(R15, -1), RAX); }, &[0x49, 0x39, 0x47, 0xff]),
        ("cmp qword ptr [rbx + 1], rax", |a| { a.cmp_mr(mem(RBX, 1), RAX); }, &[0x48, 0x39, 0x43, 0x01]),
        ("cmp qword ptr [rax + rax*1], rax", |a| { a.cmp_mr(mem_idx(RAX, RAX, 0, 0), RAX); }, &[0x48, 0x39, 0x04, 0x00]),
        ("cmp qword ptr [rax + rsi*1], rax", |a| { a.cmp_mr(mem_idx(RAX, RSI, 0, 0), RAX); }, &[0x48, 0x39, 0x04, 0x30]),
        ("cmp qword ptr [rax + r13*1], rax", |a| { a.cmp_mr(mem_idx(RAX, R13, 0, 0), RAX); }, &[0x4a, 0x39, 0x04, 0x28]),
        ("cmp qword ptr [rsp + rcx*1], rax", |a| { a.cmp_mr(mem_idx(RSP, RCX, 0, 0), RAX); }, &[0x48, 0x39, 0x04, 0x0c]),
        ("cmp qword ptr [rsp + r8*1], rax", |a| { a.cmp_mr(mem_idx(RSP, R8, 0, 0), RAX); }, &[0x4a, 0x39, 0x04, 0x04]),
        ("cmp qword ptr [rsp + r15*1], rax", |a| { a.cmp_mr(mem_idx(RSP, R15, 0, 0), RAX); }, &[0x4a, 0x39, 0x04, 0x3c]),
        ("cmp qword ptr [rbp + rbp*1], rax", |a| { a.cmp_mr(mem_idx(RBP, RBP, 0, 0), RAX); }, &[0x48, 0x39, 0x44, 0x2d, 0x00]),
        ("cmp qword ptr [rbp + r12*1], rax", |a| { a.cmp_mr(mem_idx(RBP, R12, 0, 0), RAX); }, &[0x4a, 0x39, 0x44, 0x25, 0x00]),
        ("cmp qword ptr [r12 + rax*1], rax", |a| { a.cmp_mr(mem_idx(R12, RAX, 0, 0), RAX); }, &[0x49, 0x39, 0x04, 0x04]),
        ("cmp qword ptr [r12 + rsi*1], rax", |a| { a.cmp_mr(mem_idx(R12, RSI, 0, 0), RAX); }, &[0x49, 0x39, 0x04, 0x34]),
        ("cmp qword ptr [r12 + r13*1], rax", |a| { a.cmp_mr(mem_idx(R12, R13, 0, 0), RAX); }, &[0x4b, 0x39, 0x04, 0x2c]),
        ("cmp qword ptr [r13 + rcx*1], rax", |a| { a.cmp_mr(mem_idx(R13, RCX, 0, 0), RAX); }, &[0x49, 0x39, 0x44, 0x0d, 0x00]),
        ("cmp qword ptr [r13 + r8*1], rax", |a| { a.cmp_mr(mem_idx(R13, R8, 0, 0), RAX); }, &[0x4b, 0x39, 0x44, 0x05, 0x00]),
        ("cmp qword ptr [r13 + r15*1], rax", |a| { a.cmp_mr(mem_idx(R13, R15, 0, 0), RAX); }, &[0x4b, 0x39, 0x44, 0x3d, 0x00]),
        ("cmp qword ptr [r15 + rbp*1], rax", |a| { a.cmp_mr(mem_idx(R15, RBP, 0, 0), RAX); }, &[0x49, 0x39, 0x04, 0x2f]),
        ("cmp qword ptr [r15 + r12*1], rax", |a| { a.cmp_mr(mem_idx(R15, R12, 0, 0), RAX); }, &[0x4b, 0x39, 0x04, 0x27]),
        ("cmp qword ptr [rbx + rcx*2 + 32], rax", |a| { a.cmp_mr(mem_idx(RBX, RCX, 1, 32), RAX); }, &[0x48, 0x39, 0x44, 0x4b, 0x20]),
        ("cmp qword ptr [rax], rsi", |a| { a.cmp_mr(mem(RAX, 0), RSI); }, &[0x48, 0x39, 0x30]),
        ("cmp qword ptr [rax - 305419896], rsi", |a| { a.cmp_mr(mem(RAX, -305419896), RSI); }, &[0x48, 0x39, 0xb0, 0x88, 0xa9, 0xcb, 0xed]),
        ("cmp qword ptr [rsp + 305419896], rsi", |a| { a.cmp_mr(mem(RSP, 305419896), RSI); }, &[0x48, 0x39, 0xb4, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("cmp qword ptr [rbp - 129], rsi", |a| { a.cmp_mr(mem(RBP, -129), RSI); }, &[0x48, 0x39, 0xb5, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp qword ptr [r8 - 128], rsi", |a| { a.cmp_mr(mem(R8, -128), RSI); }, &[0x49, 0x39, 0x70, 0x80]),
        ("cmp qword ptr [r12 + 128], rsi", |a| { a.cmp_mr(mem(R12, 128), RSI); }, &[0x49, 0x39, 0xb4, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("cmp qword ptr [r13 + 127], rsi", |a| { a.cmp_mr(mem(R13, 127), RSI); }, &[0x49, 0x39, 0x75, 0x7f]),
        ("cmp qword ptr [r15 - 1], rsi", |a| { a.cmp_mr(mem(R15, -1), RSI); }, &[0x49, 0x39, 0x77, 0xff]),
        ("cmp qword ptr [rbx + 1], rsi", |a| { a.cmp_mr(mem(RBX, 1), RSI); }, &[0x48, 0x39, 0x73, 0x01]),
        ("cmp qword ptr [rax + rax*1], rsi", |a| { a.cmp_mr(mem_idx(RAX, RAX, 0, 0), RSI); }, &[0x48, 0x39, 0x34, 0x00]),
        ("cmp qword ptr [rax + rsi*1], rsi", |a| { a.cmp_mr(mem_idx(RAX, RSI, 0, 0), RSI); }, &[0x48, 0x39, 0x34, 0x30]),
        ("cmp qword ptr [rax + r13*1], rsi", |a| { a.cmp_mr(mem_idx(RAX, R13, 0, 0), RSI); }, &[0x4a, 0x39, 0x34, 0x28]),
        ("cmp qword ptr [rsp + rcx*1], rsi", |a| { a.cmp_mr(mem_idx(RSP, RCX, 0, 0), RSI); }, &[0x48, 0x39, 0x34, 0x0c]),
        ("cmp qword ptr [rsp + r8*1], rsi", |a| { a.cmp_mr(mem_idx(RSP, R8, 0, 0), RSI); }, &[0x4a, 0x39, 0x34, 0x04]),
        ("cmp qword ptr [rsp + r15*1], rsi", |a| { a.cmp_mr(mem_idx(RSP, R15, 0, 0), RSI); }, &[0x4a, 0x39, 0x34, 0x3c]),
        ("cmp qword ptr [rbp + rbp*1], rsi", |a| { a.cmp_mr(mem_idx(RBP, RBP, 0, 0), RSI); }, &[0x48, 0x39, 0x74, 0x2d, 0x00]),
        ("cmp qword ptr [rbp + r12*1], rsi", |a| { a.cmp_mr(mem_idx(RBP, R12, 0, 0), RSI); }, &[0x4a, 0x39, 0x74, 0x25, 0x00]),
        ("cmp qword ptr [r12 + rax*1], rsi", |a| { a.cmp_mr(mem_idx(R12, RAX, 0, 0), RSI); }, &[0x49, 0x39, 0x34, 0x04]),
        ("cmp qword ptr [r12 + rsi*1], rsi", |a| { a.cmp_mr(mem_idx(R12, RSI, 0, 0), RSI); }, &[0x49, 0x39, 0x34, 0x34]),
        ("cmp qword ptr [r12 + r13*1], rsi", |a| { a.cmp_mr(mem_idx(R12, R13, 0, 0), RSI); }, &[0x4b, 0x39, 0x34, 0x2c]),
        ("cmp qword ptr [r13 + rcx*1], rsi", |a| { a.cmp_mr(mem_idx(R13, RCX, 0, 0), RSI); }, &[0x49, 0x39, 0x74, 0x0d, 0x00]),
        ("cmp qword ptr [r13 + r8*1], rsi", |a| { a.cmp_mr(mem_idx(R13, R8, 0, 0), RSI); }, &[0x4b, 0x39, 0x74, 0x05, 0x00]),
        ("cmp qword ptr [r13 + r15*1], rsi", |a| { a.cmp_mr(mem_idx(R13, R15, 0, 0), RSI); }, &[0x4b, 0x39, 0x74, 0x3d, 0x00]),
        ("cmp qword ptr [r15 + rbp*1], rsi", |a| { a.cmp_mr(mem_idx(R15, RBP, 0, 0), RSI); }, &[0x49, 0x39, 0x34, 0x2f]),
        ("cmp qword ptr [r15 + r12*1], rsi", |a| { a.cmp_mr(mem_idx(R15, R12, 0, 0), RSI); }, &[0x4b, 0x39, 0x34, 0x27]),
        ("cmp qword ptr [rbx + rcx*2 + 32], rsi", |a| { a.cmp_mr(mem_idx(RBX, RCX, 1, 32), RSI); }, &[0x48, 0x39, 0x74, 0x4b, 0x20]),
        ("cmp qword ptr [rax], r9", |a| { a.cmp_mr(mem(RAX, 0), R9); }, &[0x4c, 0x39, 0x08]),
        ("cmp qword ptr [rax - 305419896], r9", |a| { a.cmp_mr(mem(RAX, -305419896), R9); }, &[0x4c, 0x39, 0x88, 0x88, 0xa9, 0xcb, 0xed]),
        ("cmp qword ptr [rsp + 305419896], r9", |a| { a.cmp_mr(mem(RSP, 305419896), R9); }, &[0x4c, 0x39, 0x8c, 0x24, 0x78, 0x56, 0x34, 0x12]),
        ("cmp qword ptr [rbp - 129], r9", |a| { a.cmp_mr(mem(RBP, -129), R9); }, &[0x4c, 0x39, 0x8d, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp qword ptr [r8 - 128], r9", |a| { a.cmp_mr(mem(R8, -128), R9); }, &[0x4d, 0x39, 0x48, 0x80]),
        ("cmp qword ptr [r12 + 128], r9", |a| { a.cmp_mr(mem(R12, 128), R9); }, &[0x4d, 0x39, 0x8c, 0x24, 0x80, 0x00, 0x00, 0x00]),
        ("cmp qword ptr [r13 + 127], r9", |a| { a.cmp_mr(mem(R13, 127), R9); }, &[0x4d, 0x39, 0x4d, 0x7f]),
        ("cmp qword ptr [r15 - 1], r9", |a| { a.cmp_mr(mem(R15, -1), R9); }, &[0x4d, 0x39, 0x4f, 0xff]),
        ("cmp qword ptr [rbx + 1], r9", |a| { a.cmp_mr(mem(RBX, 1), R9); }, &[0x4c, 0x39, 0x4b, 0x01]),
        ("cmp qword ptr [rax + rax*1], r9", |a| { a.cmp_mr(mem_idx(RAX, RAX, 0, 0), R9); }, &[0x4c, 0x39, 0x0c, 0x00]),
        ("cmp qword ptr [rax + rsi*1], r9", |a| { a.cmp_mr(mem_idx(RAX, RSI, 0, 0), R9); }, &[0x4c, 0x39, 0x0c, 0x30]),
        ("cmp qword ptr [rax + r13*1], r9", |a| { a.cmp_mr(mem_idx(RAX, R13, 0, 0), R9); }, &[0x4e, 0x39, 0x0c, 0x28]),
        ("cmp qword ptr [rsp + rcx*1], r9", |a| { a.cmp_mr(mem_idx(RSP, RCX, 0, 0), R9); }, &[0x4c, 0x39, 0x0c, 0x0c]),
        ("cmp qword ptr [rsp + r8*1], r9", |a| { a.cmp_mr(mem_idx(RSP, R8, 0, 0), R9); }, &[0x4e, 0x39, 0x0c, 0x04]),
        ("cmp qword ptr [rsp + r15*1], r9", |a| { a.cmp_mr(mem_idx(RSP, R15, 0, 0), R9); }, &[0x4e, 0x39, 0x0c, 0x3c]),
        ("cmp qword ptr [rbp + rbp*1], r9", |a| { a.cmp_mr(mem_idx(RBP, RBP, 0, 0), R9); }, &[0x4c, 0x39, 0x4c, 0x2d, 0x00]),
        ("cmp qword ptr [rbp + r12*1], r9", |a| { a.cmp_mr(mem_idx(RBP, R12, 0, 0), R9); }, &[0x4e, 0x39, 0x4c, 0x25, 0x00]),
        ("cmp qword ptr [r12 + rax*1], r9", |a| { a.cmp_mr(mem_idx(R12, RAX, 0, 0), R9); }, &[0x4d, 0x39, 0x0c, 0x04]),
        ("cmp qword ptr [r12 + rsi*1], r9", |a| { a.cmp_mr(mem_idx(R12, RSI, 0, 0), R9); }, &[0x4d, 0x39, 0x0c, 0x34]),
        ("cmp qword ptr [r12 + r13*1], r9", |a| { a.cmp_mr(mem_idx(R12, R13, 0, 0), R9); }, &[0x4f, 0x39, 0x0c, 0x2c]),
        ("cmp qword ptr [r13 + rcx*1], r9", |a| { a.cmp_mr(mem_idx(R13, RCX, 0, 0), R9); }, &[0x4d, 0x39, 0x4c, 0x0d, 0x00]),
        ("cmp qword ptr [r13 + r8*1], r9", |a| { a.cmp_mr(mem_idx(R13, R8, 0, 0), R9); }, &[0x4f, 0x39, 0x4c, 0x05, 0x00]),
        ("cmp qword ptr [r13 + r15*1], r9", |a| { a.cmp_mr(mem_idx(R13, R15, 0, 0), R9); }, &[0x4f, 0x39, 0x4c, 0x3d, 0x00]),
        ("cmp qword ptr [r15 + rbp*1], r9", |a| { a.cmp_mr(mem_idx(R15, RBP, 0, 0), R9); }, &[0x4d, 0x39, 0x0c, 0x2f]),
        ("cmp qword ptr [r15 + r12*1], r9", |a| { a.cmp_mr(mem_idx(R15, R12, 0, 0), R9); }, &[0x4f, 0x39, 0x0c, 0x27]),
        ("cmp qword ptr [rbx + rcx*2 + 32], r9", |a| { a.cmp_mr(mem_idx(RBX, RCX, 1, 32), R9); }, &[0x4c, 0x39, 0x4c, 0x4b, 0x20]),
        ("cmp byte ptr [rax], 0", |a| { a.cmp_m8_imm(mem(RAX, 0), 0); }, &[0x80, 0x38, 0x00]),
        ("cmp byte ptr [rax], 0x7f", |a| { a.cmp_m8_imm(mem(RAX, 0), 0x7f); }, &[0x80, 0x38, 0x7f]),
        ("mov word ptr [rax], 0x1234", |a| { a.st_imm16(mem(RAX, 0), 0x1234); }, &[0x66, 0xc7, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [rax - 305419896], 0", |a| { a.cmp_m8_imm(mem(RAX, -305419896), 0); }, &[0x80, 0xb8, 0x88, 0xa9, 0xcb, 0xed, 0x00]),
        ("cmp byte ptr [rax - 305419896], 0x7f", |a| { a.cmp_m8_imm(mem(RAX, -305419896), 0x7f); }, &[0x80, 0xb8, 0x88, 0xa9, 0xcb, 0xed, 0x7f]),
        ("mov word ptr [rax - 305419896], 0x1234", |a| { a.st_imm16(mem(RAX, -305419896), 0x1234); }, &[0x66, 0xc7, 0x80, 0x88, 0xa9, 0xcb, 0xed, 0x34, 0x12]),
        ("cmp byte ptr [rsp + 305419896], 0", |a| { a.cmp_m8_imm(mem(RSP, 305419896), 0); }, &[0x80, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12, 0x00]),
        ("cmp byte ptr [rsp + 305419896], 0x7f", |a| { a.cmp_m8_imm(mem(RSP, 305419896), 0x7f); }, &[0x80, 0xbc, 0x24, 0x78, 0x56, 0x34, 0x12, 0x7f]),
        ("mov word ptr [rsp + 305419896], 0x1234", |a| { a.st_imm16(mem(RSP, 305419896), 0x1234); }, &[0x66, 0xc7, 0x84, 0x24, 0x78, 0x56, 0x34, 0x12, 0x34, 0x12]),
        ("cmp byte ptr [rbp - 129], 0", |a| { a.cmp_m8_imm(mem(RBP, -129), 0); }, &[0x80, 0xbd, 0x7f, 0xff, 0xff, 0xff, 0x00]),
        ("cmp byte ptr [rbp - 129], 0x7f", |a| { a.cmp_m8_imm(mem(RBP, -129), 0x7f); }, &[0x80, 0xbd, 0x7f, 0xff, 0xff, 0xff, 0x7f]),
        ("mov word ptr [rbp - 129], 0x1234", |a| { a.st_imm16(mem(RBP, -129), 0x1234); }, &[0x66, 0xc7, 0x85, 0x7f, 0xff, 0xff, 0xff, 0x34, 0x12]),
        ("cmp byte ptr [r8 - 128], 0", |a| { a.cmp_m8_imm(mem(R8, -128), 0); }, &[0x41, 0x80, 0x78, 0x80, 0x00]),
        ("cmp byte ptr [r8 - 128], 0x7f", |a| { a.cmp_m8_imm(mem(R8, -128), 0x7f); }, &[0x41, 0x80, 0x78, 0x80, 0x7f]),
        ("mov word ptr [r8 - 128], 0x1234", |a| { a.st_imm16(mem(R8, -128), 0x1234); }, &[0x66, 0x41, 0xc7, 0x40, 0x80, 0x34, 0x12]),
        ("cmp byte ptr [r12 + 128], 0", |a| { a.cmp_m8_imm(mem(R12, 128), 0); }, &[0x41, 0x80, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00, 0x00]),
        ("cmp byte ptr [r12 + 128], 0x7f", |a| { a.cmp_m8_imm(mem(R12, 128), 0x7f); }, &[0x41, 0x80, 0xbc, 0x24, 0x80, 0x00, 0x00, 0x00, 0x7f]),
        ("mov word ptr [r12 + 128], 0x1234", |a| { a.st_imm16(mem(R12, 128), 0x1234); }, &[0x66, 0x41, 0xc7, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [r13 + 127], 0", |a| { a.cmp_m8_imm(mem(R13, 127), 0); }, &[0x41, 0x80, 0x7d, 0x7f, 0x00]),
        ("cmp byte ptr [r13 + 127], 0x7f", |a| { a.cmp_m8_imm(mem(R13, 127), 0x7f); }, &[0x41, 0x80, 0x7d, 0x7f, 0x7f]),
        ("mov word ptr [r13 + 127], 0x1234", |a| { a.st_imm16(mem(R13, 127), 0x1234); }, &[0x66, 0x41, 0xc7, 0x45, 0x7f, 0x34, 0x12]),
        ("cmp byte ptr [r15 - 1], 0", |a| { a.cmp_m8_imm(mem(R15, -1), 0); }, &[0x41, 0x80, 0x7f, 0xff, 0x00]),
        ("cmp byte ptr [r15 - 1], 0x7f", |a| { a.cmp_m8_imm(mem(R15, -1), 0x7f); }, &[0x41, 0x80, 0x7f, 0xff, 0x7f]),
        ("mov word ptr [r15 - 1], 0x1234", |a| { a.st_imm16(mem(R15, -1), 0x1234); }, &[0x66, 0x41, 0xc7, 0x47, 0xff, 0x34, 0x12]),
        ("cmp byte ptr [rbx + 1], 0", |a| { a.cmp_m8_imm(mem(RBX, 1), 0); }, &[0x80, 0x7b, 0x01, 0x00]),
        ("cmp byte ptr [rbx + 1], 0x7f", |a| { a.cmp_m8_imm(mem(RBX, 1), 0x7f); }, &[0x80, 0x7b, 0x01, 0x7f]),
        ("mov word ptr [rbx + 1], 0x1234", |a| { a.st_imm16(mem(RBX, 1), 0x1234); }, &[0x66, 0xc7, 0x43, 0x01, 0x34, 0x12]),
        ("cmp byte ptr [rax + rax*1], 0", |a| { a.cmp_m8_imm(mem_idx(RAX, RAX, 0, 0), 0); }, &[0x80, 0x3c, 0x00, 0x00]),
        ("cmp byte ptr [rax + rax*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RAX, RAX, 0, 0), 0x7f); }, &[0x80, 0x3c, 0x00, 0x7f]),
        ("mov word ptr [rax + rax*1], 0x1234", |a| { a.st_imm16(mem_idx(RAX, RAX, 0, 0), 0x1234); }, &[0x66, 0xc7, 0x04, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [rax + rsi*1], 0", |a| { a.cmp_m8_imm(mem_idx(RAX, RSI, 0, 0), 0); }, &[0x80, 0x3c, 0x30, 0x00]),
        ("cmp byte ptr [rax + rsi*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RAX, RSI, 0, 0), 0x7f); }, &[0x80, 0x3c, 0x30, 0x7f]),
        ("mov word ptr [rax + rsi*1], 0x1234", |a| { a.st_imm16(mem_idx(RAX, RSI, 0, 0), 0x1234); }, &[0x66, 0xc7, 0x04, 0x30, 0x34, 0x12]),
        ("cmp byte ptr [rax + r13*1], 0", |a| { a.cmp_m8_imm(mem_idx(RAX, R13, 0, 0), 0); }, &[0x42, 0x80, 0x3c, 0x28, 0x00]),
        ("cmp byte ptr [rax + r13*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RAX, R13, 0, 0), 0x7f); }, &[0x42, 0x80, 0x3c, 0x28, 0x7f]),
        ("mov word ptr [rax + r13*1], 0x1234", |a| { a.st_imm16(mem_idx(RAX, R13, 0, 0), 0x1234); }, &[0x66, 0x42, 0xc7, 0x04, 0x28, 0x34, 0x12]),
        ("cmp byte ptr [rsp + rcx*1], 0", |a| { a.cmp_m8_imm(mem_idx(RSP, RCX, 0, 0), 0); }, &[0x80, 0x3c, 0x0c, 0x00]),
        ("cmp byte ptr [rsp + rcx*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RSP, RCX, 0, 0), 0x7f); }, &[0x80, 0x3c, 0x0c, 0x7f]),
        ("mov word ptr [rsp + rcx*1], 0x1234", |a| { a.st_imm16(mem_idx(RSP, RCX, 0, 0), 0x1234); }, &[0x66, 0xc7, 0x04, 0x0c, 0x34, 0x12]),
        ("cmp byte ptr [rsp + r8*1], 0", |a| { a.cmp_m8_imm(mem_idx(RSP, R8, 0, 0), 0); }, &[0x42, 0x80, 0x3c, 0x04, 0x00]),
        ("cmp byte ptr [rsp + r8*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RSP, R8, 0, 0), 0x7f); }, &[0x42, 0x80, 0x3c, 0x04, 0x7f]),
        ("mov word ptr [rsp + r8*1], 0x1234", |a| { a.st_imm16(mem_idx(RSP, R8, 0, 0), 0x1234); }, &[0x66, 0x42, 0xc7, 0x04, 0x04, 0x34, 0x12]),
        ("cmp byte ptr [rsp + r15*1], 0", |a| { a.cmp_m8_imm(mem_idx(RSP, R15, 0, 0), 0); }, &[0x42, 0x80, 0x3c, 0x3c, 0x00]),
        ("cmp byte ptr [rsp + r15*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RSP, R15, 0, 0), 0x7f); }, &[0x42, 0x80, 0x3c, 0x3c, 0x7f]),
        ("mov word ptr [rsp + r15*1], 0x1234", |a| { a.st_imm16(mem_idx(RSP, R15, 0, 0), 0x1234); }, &[0x66, 0x42, 0xc7, 0x04, 0x3c, 0x34, 0x12]),
        ("cmp byte ptr [rbp + rbp*1], 0", |a| { a.cmp_m8_imm(mem_idx(RBP, RBP, 0, 0), 0); }, &[0x80, 0x7c, 0x2d, 0x00, 0x00]),
        ("cmp byte ptr [rbp + rbp*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RBP, RBP, 0, 0), 0x7f); }, &[0x80, 0x7c, 0x2d, 0x00, 0x7f]),
        ("mov word ptr [rbp + rbp*1], 0x1234", |a| { a.st_imm16(mem_idx(RBP, RBP, 0, 0), 0x1234); }, &[0x66, 0xc7, 0x44, 0x2d, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [rbp + r12*1], 0", |a| { a.cmp_m8_imm(mem_idx(RBP, R12, 0, 0), 0); }, &[0x42, 0x80, 0x7c, 0x25, 0x00, 0x00]),
        ("cmp byte ptr [rbp + r12*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RBP, R12, 0, 0), 0x7f); }, &[0x42, 0x80, 0x7c, 0x25, 0x00, 0x7f]),
        ("mov word ptr [rbp + r12*1], 0x1234", |a| { a.st_imm16(mem_idx(RBP, R12, 0, 0), 0x1234); }, &[0x66, 0x42, 0xc7, 0x44, 0x25, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [r12 + rax*1], 0", |a| { a.cmp_m8_imm(mem_idx(R12, RAX, 0, 0), 0); }, &[0x41, 0x80, 0x3c, 0x04, 0x00]),
        ("cmp byte ptr [r12 + rax*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R12, RAX, 0, 0), 0x7f); }, &[0x41, 0x80, 0x3c, 0x04, 0x7f]),
        ("mov word ptr [r12 + rax*1], 0x1234", |a| { a.st_imm16(mem_idx(R12, RAX, 0, 0), 0x1234); }, &[0x66, 0x41, 0xc7, 0x04, 0x04, 0x34, 0x12]),
        ("cmp byte ptr [r12 + rsi*1], 0", |a| { a.cmp_m8_imm(mem_idx(R12, RSI, 0, 0), 0); }, &[0x41, 0x80, 0x3c, 0x34, 0x00]),
        ("cmp byte ptr [r12 + rsi*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R12, RSI, 0, 0), 0x7f); }, &[0x41, 0x80, 0x3c, 0x34, 0x7f]),
        ("mov word ptr [r12 + rsi*1], 0x1234", |a| { a.st_imm16(mem_idx(R12, RSI, 0, 0), 0x1234); }, &[0x66, 0x41, 0xc7, 0x04, 0x34, 0x34, 0x12]),
        ("cmp byte ptr [r12 + r13*1], 0", |a| { a.cmp_m8_imm(mem_idx(R12, R13, 0, 0), 0); }, &[0x43, 0x80, 0x3c, 0x2c, 0x00]),
        ("cmp byte ptr [r12 + r13*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R12, R13, 0, 0), 0x7f); }, &[0x43, 0x80, 0x3c, 0x2c, 0x7f]),
        ("mov word ptr [r12 + r13*1], 0x1234", |a| { a.st_imm16(mem_idx(R12, R13, 0, 0), 0x1234); }, &[0x66, 0x43, 0xc7, 0x04, 0x2c, 0x34, 0x12]),
        ("cmp byte ptr [r13 + rcx*1], 0", |a| { a.cmp_m8_imm(mem_idx(R13, RCX, 0, 0), 0); }, &[0x41, 0x80, 0x7c, 0x0d, 0x00, 0x00]),
        ("cmp byte ptr [r13 + rcx*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R13, RCX, 0, 0), 0x7f); }, &[0x41, 0x80, 0x7c, 0x0d, 0x00, 0x7f]),
        ("mov word ptr [r13 + rcx*1], 0x1234", |a| { a.st_imm16(mem_idx(R13, RCX, 0, 0), 0x1234); }, &[0x66, 0x41, 0xc7, 0x44, 0x0d, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [r13 + r8*1], 0", |a| { a.cmp_m8_imm(mem_idx(R13, R8, 0, 0), 0); }, &[0x43, 0x80, 0x7c, 0x05, 0x00, 0x00]),
        ("cmp byte ptr [r13 + r8*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R13, R8, 0, 0), 0x7f); }, &[0x43, 0x80, 0x7c, 0x05, 0x00, 0x7f]),
        ("mov word ptr [r13 + r8*1], 0x1234", |a| { a.st_imm16(mem_idx(R13, R8, 0, 0), 0x1234); }, &[0x66, 0x43, 0xc7, 0x44, 0x05, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [r13 + r15*1], 0", |a| { a.cmp_m8_imm(mem_idx(R13, R15, 0, 0), 0); }, &[0x43, 0x80, 0x7c, 0x3d, 0x00, 0x00]),
        ("cmp byte ptr [r13 + r15*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R13, R15, 0, 0), 0x7f); }, &[0x43, 0x80, 0x7c, 0x3d, 0x00, 0x7f]),
        ("mov word ptr [r13 + r15*1], 0x1234", |a| { a.st_imm16(mem_idx(R13, R15, 0, 0), 0x1234); }, &[0x66, 0x43, 0xc7, 0x44, 0x3d, 0x00, 0x34, 0x12]),
        ("cmp byte ptr [r15 + rbp*1], 0", |a| { a.cmp_m8_imm(mem_idx(R15, RBP, 0, 0), 0); }, &[0x41, 0x80, 0x3c, 0x2f, 0x00]),
        ("cmp byte ptr [r15 + rbp*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R15, RBP, 0, 0), 0x7f); }, &[0x41, 0x80, 0x3c, 0x2f, 0x7f]),
        ("mov word ptr [r15 + rbp*1], 0x1234", |a| { a.st_imm16(mem_idx(R15, RBP, 0, 0), 0x1234); }, &[0x66, 0x41, 0xc7, 0x04, 0x2f, 0x34, 0x12]),
        ("cmp byte ptr [r15 + r12*1], 0", |a| { a.cmp_m8_imm(mem_idx(R15, R12, 0, 0), 0); }, &[0x43, 0x80, 0x3c, 0x27, 0x00]),
        ("cmp byte ptr [r15 + r12*1], 0x7f", |a| { a.cmp_m8_imm(mem_idx(R15, R12, 0, 0), 0x7f); }, &[0x43, 0x80, 0x3c, 0x27, 0x7f]),
        ("mov word ptr [r15 + r12*1], 0x1234", |a| { a.st_imm16(mem_idx(R15, R12, 0, 0), 0x1234); }, &[0x66, 0x43, 0xc7, 0x04, 0x27, 0x34, 0x12]),
        ("cmp byte ptr [rbx + rcx*2 + 32], 0", |a| { a.cmp_m8_imm(mem_idx(RBX, RCX, 1, 32), 0); }, &[0x80, 0x7c, 0x4b, 0x20, 0x00]),
        ("cmp byte ptr [rbx + rcx*2 + 32], 0x7f", |a| { a.cmp_m8_imm(mem_idx(RBX, RCX, 1, 32), 0x7f); }, &[0x80, 0x7c, 0x4b, 0x20, 0x7f]),
        ("mov word ptr [rbx + rcx*2 + 32], 0x1234", |a| { a.st_imm16(mem_idx(RBX, RCX, 1, 32), 0x1234); }, &[0x66, 0xc7, 0x44, 0x4b, 0x20, 0x34, 0x12]),
    ];
    check_encodings(cases);
}

    /// Mov, the six group-1 operations and test, 64- and 32-bit, with the corner registers.
#[rustfmt::skip]
#[test]
fn register_to_register_forms_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("mov rax, rcx", |a| { a.mov_rr(RAX, RCX, true); }, &[0x48, 0x89, 0xc8]),
        ("add rax, rcx", |a| { a.alu_rr(ADD, RAX, RCX, true); }, &[0x48, 0x01, 0xc8]),
        ("or rax, rcx", |a| { a.alu_rr(OR, RAX, RCX, true); }, &[0x48, 0x09, 0xc8]),
        ("and rax, rcx", |a| { a.alu_rr(AND, RAX, RCX, true); }, &[0x48, 0x21, 0xc8]),
        ("sub rax, rcx", |a| { a.alu_rr(SUB, RAX, RCX, true); }, &[0x48, 0x29, 0xc8]),
        ("xor rax, rcx", |a| { a.alu_rr(XOR, RAX, RCX, true); }, &[0x48, 0x31, 0xc8]),
        ("cmp rax, rcx", |a| { a.alu_rr(CMP, RAX, RCX, true); }, &[0x48, 0x39, 0xc8]),
        ("test rax, rcx", |a| { a.test_rr(RAX, RCX, true); }, &[0x48, 0x85, 0xc8]),
        ("mov rsp, rbp", |a| { a.mov_rr(RSP, RBP, true); }, &[0x48, 0x89, 0xec]),
        ("add rsp, rbp", |a| { a.alu_rr(ADD, RSP, RBP, true); }, &[0x48, 0x01, 0xec]),
        ("or rsp, rbp", |a| { a.alu_rr(OR, RSP, RBP, true); }, &[0x48, 0x09, 0xec]),
        ("and rsp, rbp", |a| { a.alu_rr(AND, RSP, RBP, true); }, &[0x48, 0x21, 0xec]),
        ("sub rsp, rbp", |a| { a.alu_rr(SUB, RSP, RBP, true); }, &[0x48, 0x29, 0xec]),
        ("xor rsp, rbp", |a| { a.alu_rr(XOR, RSP, RBP, true); }, &[0x48, 0x31, 0xec]),
        ("cmp rsp, rbp", |a| { a.alu_rr(CMP, RSP, RBP, true); }, &[0x48, 0x39, 0xec]),
        ("test rsp, rbp", |a| { a.test_rr(RSP, RBP, true); }, &[0x48, 0x85, 0xec]),
        ("mov r12, r13", |a| { a.mov_rr(R12, R13, true); }, &[0x4d, 0x89, 0xec]),
        ("add r12, r13", |a| { a.alu_rr(ADD, R12, R13, true); }, &[0x4d, 0x01, 0xec]),
        ("or r12, r13", |a| { a.alu_rr(OR, R12, R13, true); }, &[0x4d, 0x09, 0xec]),
        ("and r12, r13", |a| { a.alu_rr(AND, R12, R13, true); }, &[0x4d, 0x21, 0xec]),
        ("sub r12, r13", |a| { a.alu_rr(SUB, R12, R13, true); }, &[0x4d, 0x29, 0xec]),
        ("xor r12, r13", |a| { a.alu_rr(XOR, R12, R13, true); }, &[0x4d, 0x31, 0xec]),
        ("cmp r12, r13", |a| { a.alu_rr(CMP, R12, R13, true); }, &[0x4d, 0x39, 0xec]),
        ("test r12, r13", |a| { a.test_rr(R12, R13, true); }, &[0x4d, 0x85, 0xec]),
        ("mov r8, r15", |a| { a.mov_rr(R8, R15, true); }, &[0x4d, 0x89, 0xf8]),
        ("add r8, r15", |a| { a.alu_rr(ADD, R8, R15, true); }, &[0x4d, 0x01, 0xf8]),
        ("or r8, r15", |a| { a.alu_rr(OR, R8, R15, true); }, &[0x4d, 0x09, 0xf8]),
        ("and r8, r15", |a| { a.alu_rr(AND, R8, R15, true); }, &[0x4d, 0x21, 0xf8]),
        ("sub r8, r15", |a| { a.alu_rr(SUB, R8, R15, true); }, &[0x4d, 0x29, 0xf8]),
        ("xor r8, r15", |a| { a.alu_rr(XOR, R8, R15, true); }, &[0x4d, 0x31, 0xf8]),
        ("cmp r8, r15", |a| { a.alu_rr(CMP, R8, R15, true); }, &[0x4d, 0x39, 0xf8]),
        ("test r8, r15", |a| { a.test_rr(R8, R15, true); }, &[0x4d, 0x85, 0xf8]),
        ("mov r15, r8", |a| { a.mov_rr(R15, R8, true); }, &[0x4d, 0x89, 0xc7]),
        ("add r15, r8", |a| { a.alu_rr(ADD, R15, R8, true); }, &[0x4d, 0x01, 0xc7]),
        ("or r15, r8", |a| { a.alu_rr(OR, R15, R8, true); }, &[0x4d, 0x09, 0xc7]),
        ("and r15, r8", |a| { a.alu_rr(AND, R15, R8, true); }, &[0x4d, 0x21, 0xc7]),
        ("sub r15, r8", |a| { a.alu_rr(SUB, R15, R8, true); }, &[0x4d, 0x29, 0xc7]),
        ("xor r15, r8", |a| { a.alu_rr(XOR, R15, R8, true); }, &[0x4d, 0x31, 0xc7]),
        ("cmp r15, r8", |a| { a.alu_rr(CMP, R15, R8, true); }, &[0x4d, 0x39, 0xc7]),
        ("test r15, r8", |a| { a.test_rr(R15, R8, true); }, &[0x4d, 0x85, 0xc7]),
        ("mov rbp, rsp", |a| { a.mov_rr(RBP, RSP, true); }, &[0x48, 0x89, 0xe5]),
        ("add rbp, rsp", |a| { a.alu_rr(ADD, RBP, RSP, true); }, &[0x48, 0x01, 0xe5]),
        ("or rbp, rsp", |a| { a.alu_rr(OR, RBP, RSP, true); }, &[0x48, 0x09, 0xe5]),
        ("and rbp, rsp", |a| { a.alu_rr(AND, RBP, RSP, true); }, &[0x48, 0x21, 0xe5]),
        ("sub rbp, rsp", |a| { a.alu_rr(SUB, RBP, RSP, true); }, &[0x48, 0x29, 0xe5]),
        ("xor rbp, rsp", |a| { a.alu_rr(XOR, RBP, RSP, true); }, &[0x48, 0x31, 0xe5]),
        ("cmp rbp, rsp", |a| { a.alu_rr(CMP, RBP, RSP, true); }, &[0x48, 0x39, 0xe5]),
        ("test rbp, rsp", |a| { a.test_rr(RBP, RSP, true); }, &[0x48, 0x85, 0xe5]),
        ("mov r13, r12", |a| { a.mov_rr(R13, R12, true); }, &[0x4d, 0x89, 0xe5]),
        ("add r13, r12", |a| { a.alu_rr(ADD, R13, R12, true); }, &[0x4d, 0x01, 0xe5]),
        ("or r13, r12", |a| { a.alu_rr(OR, R13, R12, true); }, &[0x4d, 0x09, 0xe5]),
        ("and r13, r12", |a| { a.alu_rr(AND, R13, R12, true); }, &[0x4d, 0x21, 0xe5]),
        ("sub r13, r12", |a| { a.alu_rr(SUB, R13, R12, true); }, &[0x4d, 0x29, 0xe5]),
        ("xor r13, r12", |a| { a.alu_rr(XOR, R13, R12, true); }, &[0x4d, 0x31, 0xe5]),
        ("cmp r13, r12", |a| { a.alu_rr(CMP, R13, R12, true); }, &[0x4d, 0x39, 0xe5]),
        ("test r13, r12", |a| { a.test_rr(R13, R12, true); }, &[0x4d, 0x85, 0xe5]),
        ("mov rdi, rax", |a| { a.mov_rr(RDI, RAX, true); }, &[0x48, 0x89, 0xc7]),
        ("add rdi, rax", |a| { a.alu_rr(ADD, RDI, RAX, true); }, &[0x48, 0x01, 0xc7]),
        ("or rdi, rax", |a| { a.alu_rr(OR, RDI, RAX, true); }, &[0x48, 0x09, 0xc7]),
        ("and rdi, rax", |a| { a.alu_rr(AND, RDI, RAX, true); }, &[0x48, 0x21, 0xc7]),
        ("sub rdi, rax", |a| { a.alu_rr(SUB, RDI, RAX, true); }, &[0x48, 0x29, 0xc7]),
        ("xor rdi, rax", |a| { a.alu_rr(XOR, RDI, RAX, true); }, &[0x48, 0x31, 0xc7]),
        ("cmp rdi, rax", |a| { a.alu_rr(CMP, RDI, RAX, true); }, &[0x48, 0x39, 0xc7]),
        ("test rdi, rax", |a| { a.test_rr(RDI, RAX, true); }, &[0x48, 0x85, 0xc7]),
        ("mov rax, rdi", |a| { a.mov_rr(RAX, RDI, true); }, &[0x48, 0x89, 0xf8]),
        ("add rax, rdi", |a| { a.alu_rr(ADD, RAX, RDI, true); }, &[0x48, 0x01, 0xf8]),
        ("or rax, rdi", |a| { a.alu_rr(OR, RAX, RDI, true); }, &[0x48, 0x09, 0xf8]),
        ("and rax, rdi", |a| { a.alu_rr(AND, RAX, RDI, true); }, &[0x48, 0x21, 0xf8]),
        ("sub rax, rdi", |a| { a.alu_rr(SUB, RAX, RDI, true); }, &[0x48, 0x29, 0xf8]),
        ("xor rax, rdi", |a| { a.alu_rr(XOR, RAX, RDI, true); }, &[0x48, 0x31, 0xf8]),
        ("cmp rax, rdi", |a| { a.alu_rr(CMP, RAX, RDI, true); }, &[0x48, 0x39, 0xf8]),
        ("test rax, rdi", |a| { a.test_rr(RAX, RDI, true); }, &[0x48, 0x85, 0xf8]),
        ("mov eax, ecx", |a| { a.mov_rr(RAX, RCX, false); }, &[0x89, 0xc8]),
        ("add eax, ecx", |a| { a.alu_rr(ADD, RAX, RCX, false); }, &[0x01, 0xc8]),
        ("or eax, ecx", |a| { a.alu_rr(OR, RAX, RCX, false); }, &[0x09, 0xc8]),
        ("and eax, ecx", |a| { a.alu_rr(AND, RAX, RCX, false); }, &[0x21, 0xc8]),
        ("sub eax, ecx", |a| { a.alu_rr(SUB, RAX, RCX, false); }, &[0x29, 0xc8]),
        ("xor eax, ecx", |a| { a.alu_rr(XOR, RAX, RCX, false); }, &[0x31, 0xc8]),
        ("cmp eax, ecx", |a| { a.alu_rr(CMP, RAX, RCX, false); }, &[0x39, 0xc8]),
        ("test eax, ecx", |a| { a.test_rr(RAX, RCX, false); }, &[0x85, 0xc8]),
        ("mov esp, ebp", |a| { a.mov_rr(RSP, RBP, false); }, &[0x89, 0xec]),
        ("add esp, ebp", |a| { a.alu_rr(ADD, RSP, RBP, false); }, &[0x01, 0xec]),
        ("or esp, ebp", |a| { a.alu_rr(OR, RSP, RBP, false); }, &[0x09, 0xec]),
        ("and esp, ebp", |a| { a.alu_rr(AND, RSP, RBP, false); }, &[0x21, 0xec]),
        ("sub esp, ebp", |a| { a.alu_rr(SUB, RSP, RBP, false); }, &[0x29, 0xec]),
        ("xor esp, ebp", |a| { a.alu_rr(XOR, RSP, RBP, false); }, &[0x31, 0xec]),
        ("cmp esp, ebp", |a| { a.alu_rr(CMP, RSP, RBP, false); }, &[0x39, 0xec]),
        ("test esp, ebp", |a| { a.test_rr(RSP, RBP, false); }, &[0x85, 0xec]),
        ("mov r12d, r13d", |a| { a.mov_rr(R12, R13, false); }, &[0x45, 0x89, 0xec]),
        ("add r12d, r13d", |a| { a.alu_rr(ADD, R12, R13, false); }, &[0x45, 0x01, 0xec]),
        ("or r12d, r13d", |a| { a.alu_rr(OR, R12, R13, false); }, &[0x45, 0x09, 0xec]),
        ("and r12d, r13d", |a| { a.alu_rr(AND, R12, R13, false); }, &[0x45, 0x21, 0xec]),
        ("sub r12d, r13d", |a| { a.alu_rr(SUB, R12, R13, false); }, &[0x45, 0x29, 0xec]),
        ("xor r12d, r13d", |a| { a.alu_rr(XOR, R12, R13, false); }, &[0x45, 0x31, 0xec]),
        ("cmp r12d, r13d", |a| { a.alu_rr(CMP, R12, R13, false); }, &[0x45, 0x39, 0xec]),
        ("test r12d, r13d", |a| { a.test_rr(R12, R13, false); }, &[0x45, 0x85, 0xec]),
        ("mov r8d, r15d", |a| { a.mov_rr(R8, R15, false); }, &[0x45, 0x89, 0xf8]),
        ("add r8d, r15d", |a| { a.alu_rr(ADD, R8, R15, false); }, &[0x45, 0x01, 0xf8]),
        ("or r8d, r15d", |a| { a.alu_rr(OR, R8, R15, false); }, &[0x45, 0x09, 0xf8]),
        ("and r8d, r15d", |a| { a.alu_rr(AND, R8, R15, false); }, &[0x45, 0x21, 0xf8]),
        ("sub r8d, r15d", |a| { a.alu_rr(SUB, R8, R15, false); }, &[0x45, 0x29, 0xf8]),
        ("xor r8d, r15d", |a| { a.alu_rr(XOR, R8, R15, false); }, &[0x45, 0x31, 0xf8]),
        ("cmp r8d, r15d", |a| { a.alu_rr(CMP, R8, R15, false); }, &[0x45, 0x39, 0xf8]),
        ("test r8d, r15d", |a| { a.test_rr(R8, R15, false); }, &[0x45, 0x85, 0xf8]),
        ("mov r15d, r8d", |a| { a.mov_rr(R15, R8, false); }, &[0x45, 0x89, 0xc7]),
        ("add r15d, r8d", |a| { a.alu_rr(ADD, R15, R8, false); }, &[0x45, 0x01, 0xc7]),
        ("or r15d, r8d", |a| { a.alu_rr(OR, R15, R8, false); }, &[0x45, 0x09, 0xc7]),
        ("and r15d, r8d", |a| { a.alu_rr(AND, R15, R8, false); }, &[0x45, 0x21, 0xc7]),
        ("sub r15d, r8d", |a| { a.alu_rr(SUB, R15, R8, false); }, &[0x45, 0x29, 0xc7]),
        ("xor r15d, r8d", |a| { a.alu_rr(XOR, R15, R8, false); }, &[0x45, 0x31, 0xc7]),
        ("cmp r15d, r8d", |a| { a.alu_rr(CMP, R15, R8, false); }, &[0x45, 0x39, 0xc7]),
        ("test r15d, r8d", |a| { a.test_rr(R15, R8, false); }, &[0x45, 0x85, 0xc7]),
        ("mov ebp, esp", |a| { a.mov_rr(RBP, RSP, false); }, &[0x89, 0xe5]),
        ("add ebp, esp", |a| { a.alu_rr(ADD, RBP, RSP, false); }, &[0x01, 0xe5]),
        ("or ebp, esp", |a| { a.alu_rr(OR, RBP, RSP, false); }, &[0x09, 0xe5]),
        ("and ebp, esp", |a| { a.alu_rr(AND, RBP, RSP, false); }, &[0x21, 0xe5]),
        ("sub ebp, esp", |a| { a.alu_rr(SUB, RBP, RSP, false); }, &[0x29, 0xe5]),
        ("xor ebp, esp", |a| { a.alu_rr(XOR, RBP, RSP, false); }, &[0x31, 0xe5]),
        ("cmp ebp, esp", |a| { a.alu_rr(CMP, RBP, RSP, false); }, &[0x39, 0xe5]),
        ("test ebp, esp", |a| { a.test_rr(RBP, RSP, false); }, &[0x85, 0xe5]),
        ("mov r13d, r12d", |a| { a.mov_rr(R13, R12, false); }, &[0x45, 0x89, 0xe5]),
        ("add r13d, r12d", |a| { a.alu_rr(ADD, R13, R12, false); }, &[0x45, 0x01, 0xe5]),
        ("or r13d, r12d", |a| { a.alu_rr(OR, R13, R12, false); }, &[0x45, 0x09, 0xe5]),
        ("and r13d, r12d", |a| { a.alu_rr(AND, R13, R12, false); }, &[0x45, 0x21, 0xe5]),
        ("sub r13d, r12d", |a| { a.alu_rr(SUB, R13, R12, false); }, &[0x45, 0x29, 0xe5]),
        ("xor r13d, r12d", |a| { a.alu_rr(XOR, R13, R12, false); }, &[0x45, 0x31, 0xe5]),
        ("cmp r13d, r12d", |a| { a.alu_rr(CMP, R13, R12, false); }, &[0x45, 0x39, 0xe5]),
        ("test r13d, r12d", |a| { a.test_rr(R13, R12, false); }, &[0x45, 0x85, 0xe5]),
        ("mov edi, eax", |a| { a.mov_rr(RDI, RAX, false); }, &[0x89, 0xc7]),
        ("add edi, eax", |a| { a.alu_rr(ADD, RDI, RAX, false); }, &[0x01, 0xc7]),
        ("or edi, eax", |a| { a.alu_rr(OR, RDI, RAX, false); }, &[0x09, 0xc7]),
        ("and edi, eax", |a| { a.alu_rr(AND, RDI, RAX, false); }, &[0x21, 0xc7]),
        ("sub edi, eax", |a| { a.alu_rr(SUB, RDI, RAX, false); }, &[0x29, 0xc7]),
        ("xor edi, eax", |a| { a.alu_rr(XOR, RDI, RAX, false); }, &[0x31, 0xc7]),
        ("cmp edi, eax", |a| { a.alu_rr(CMP, RDI, RAX, false); }, &[0x39, 0xc7]),
        ("test edi, eax", |a| { a.test_rr(RDI, RAX, false); }, &[0x85, 0xc7]),
        ("mov eax, edi", |a| { a.mov_rr(RAX, RDI, false); }, &[0x89, 0xf8]),
        ("add eax, edi", |a| { a.alu_rr(ADD, RAX, RDI, false); }, &[0x01, 0xf8]),
        ("or eax, edi", |a| { a.alu_rr(OR, RAX, RDI, false); }, &[0x09, 0xf8]),
        ("and eax, edi", |a| { a.alu_rr(AND, RAX, RDI, false); }, &[0x21, 0xf8]),
        ("sub eax, edi", |a| { a.alu_rr(SUB, RAX, RDI, false); }, &[0x29, 0xf8]),
        ("xor eax, edi", |a| { a.alu_rr(XOR, RAX, RDI, false); }, &[0x31, 0xf8]),
        ("cmp eax, edi", |a| { a.alu_rr(CMP, RAX, RDI, false); }, &[0x39, 0xf8]),
        ("test eax, edi", |a| { a.test_rr(RAX, RDI, false); }, &[0x85, 0xf8]),
    ];
    check_encodings(cases);
}

    /// Imm8 for values that fit, imm32 otherwise (registers other than rax, which has a shorter dedicated encoding).
#[rustfmt::skip]
#[test]
fn immediate_arithmetic_matches_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add rcx, 5", |a| { a.alu_ri(ADD, RCX, 5, true); }, &[0x48, 0x83, 0xc1, 0x05]),
        ("add rcx, -1", |a| { a.alu_ri(ADD, RCX, -1, true); }, &[0x48, 0x83, 0xc1, 0xff]),
        ("add rcx, 127", |a| { a.alu_ri(ADD, RCX, 127, true); }, &[0x48, 0x83, 0xc1, 0x7f]),
        ("add rcx, 128", |a| { a.alu_ri(ADD, RCX, 128, true); }, &[0x48, 0x81, 0xc1, 0x80, 0x00, 0x00, 0x00]),
        ("add rcx, -128", |a| { a.alu_ri(ADD, RCX, -128, true); }, &[0x48, 0x83, 0xc1, 0x80]),
        ("add rcx, -129", |a| { a.alu_ri(ADD, RCX, -129, true); }, &[0x48, 0x81, 0xc1, 0x7f, 0xff, 0xff, 0xff]),
        ("add rcx, 4096", |a| { a.alu_ri(ADD, RCX, 4096, true); }, &[0x48, 0x81, 0xc1, 0x00, 0x10, 0x00, 0x00]),
        ("add rcx, 2147483647", |a| { a.alu_ri(ADD, RCX, 2147483647, true); }, &[0x48, 0x81, 0xc1, 0xff, 0xff, 0xff, 0x7f]),
        ("add rcx, -2147483648", |a| { a.alu_ri(ADD, RCX, -2147483648, true); }, &[0x48, 0x81, 0xc1, 0x00, 0x00, 0x00, 0x80]),
        ("or rcx, 5", |a| { a.alu_ri(OR, RCX, 5, true); }, &[0x48, 0x83, 0xc9, 0x05]),
        ("or rcx, -1", |a| { a.alu_ri(OR, RCX, -1, true); }, &[0x48, 0x83, 0xc9, 0xff]),
        ("or rcx, 127", |a| { a.alu_ri(OR, RCX, 127, true); }, &[0x48, 0x83, 0xc9, 0x7f]),
        ("or rcx, 128", |a| { a.alu_ri(OR, RCX, 128, true); }, &[0x48, 0x81, 0xc9, 0x80, 0x00, 0x00, 0x00]),
        ("or rcx, -128", |a| { a.alu_ri(OR, RCX, -128, true); }, &[0x48, 0x83, 0xc9, 0x80]),
        ("or rcx, -129", |a| { a.alu_ri(OR, RCX, -129, true); }, &[0x48, 0x81, 0xc9, 0x7f, 0xff, 0xff, 0xff]),
        ("or rcx, 4096", |a| { a.alu_ri(OR, RCX, 4096, true); }, &[0x48, 0x81, 0xc9, 0x00, 0x10, 0x00, 0x00]),
        ("or rcx, 2147483647", |a| { a.alu_ri(OR, RCX, 2147483647, true); }, &[0x48, 0x81, 0xc9, 0xff, 0xff, 0xff, 0x7f]),
        ("or rcx, -2147483648", |a| { a.alu_ri(OR, RCX, -2147483648, true); }, &[0x48, 0x81, 0xc9, 0x00, 0x00, 0x00, 0x80]),
        ("and rcx, 5", |a| { a.alu_ri(AND, RCX, 5, true); }, &[0x48, 0x83, 0xe1, 0x05]),
        ("and rcx, -1", |a| { a.alu_ri(AND, RCX, -1, true); }, &[0x48, 0x83, 0xe1, 0xff]),
        ("and rcx, 127", |a| { a.alu_ri(AND, RCX, 127, true); }, &[0x48, 0x83, 0xe1, 0x7f]),
        ("and rcx, 128", |a| { a.alu_ri(AND, RCX, 128, true); }, &[0x48, 0x81, 0xe1, 0x80, 0x00, 0x00, 0x00]),
        ("and rcx, -128", |a| { a.alu_ri(AND, RCX, -128, true); }, &[0x48, 0x83, 0xe1, 0x80]),
        ("and rcx, -129", |a| { a.alu_ri(AND, RCX, -129, true); }, &[0x48, 0x81, 0xe1, 0x7f, 0xff, 0xff, 0xff]),
        ("and rcx, 4096", |a| { a.alu_ri(AND, RCX, 4096, true); }, &[0x48, 0x81, 0xe1, 0x00, 0x10, 0x00, 0x00]),
        ("and rcx, 2147483647", |a| { a.alu_ri(AND, RCX, 2147483647, true); }, &[0x48, 0x81, 0xe1, 0xff, 0xff, 0xff, 0x7f]),
        ("and rcx, -2147483648", |a| { a.alu_ri(AND, RCX, -2147483648, true); }, &[0x48, 0x81, 0xe1, 0x00, 0x00, 0x00, 0x80]),
        ("sub rcx, 5", |a| { a.alu_ri(SUB, RCX, 5, true); }, &[0x48, 0x83, 0xe9, 0x05]),
        ("sub rcx, -1", |a| { a.alu_ri(SUB, RCX, -1, true); }, &[0x48, 0x83, 0xe9, 0xff]),
        ("sub rcx, 127", |a| { a.alu_ri(SUB, RCX, 127, true); }, &[0x48, 0x83, 0xe9, 0x7f]),
        ("sub rcx, 128", |a| { a.alu_ri(SUB, RCX, 128, true); }, &[0x48, 0x81, 0xe9, 0x80, 0x00, 0x00, 0x00]),
        ("sub rcx, -128", |a| { a.alu_ri(SUB, RCX, -128, true); }, &[0x48, 0x83, 0xe9, 0x80]),
        ("sub rcx, -129", |a| { a.alu_ri(SUB, RCX, -129, true); }, &[0x48, 0x81, 0xe9, 0x7f, 0xff, 0xff, 0xff]),
        ("sub rcx, 4096", |a| { a.alu_ri(SUB, RCX, 4096, true); }, &[0x48, 0x81, 0xe9, 0x00, 0x10, 0x00, 0x00]),
        ("sub rcx, 2147483647", |a| { a.alu_ri(SUB, RCX, 2147483647, true); }, &[0x48, 0x81, 0xe9, 0xff, 0xff, 0xff, 0x7f]),
        ("sub rcx, -2147483648", |a| { a.alu_ri(SUB, RCX, -2147483648, true); }, &[0x48, 0x81, 0xe9, 0x00, 0x00, 0x00, 0x80]),
        ("xor rcx, 5", |a| { a.alu_ri(XOR, RCX, 5, true); }, &[0x48, 0x83, 0xf1, 0x05]),
        ("xor rcx, -1", |a| { a.alu_ri(XOR, RCX, -1, true); }, &[0x48, 0x83, 0xf1, 0xff]),
        ("xor rcx, 127", |a| { a.alu_ri(XOR, RCX, 127, true); }, &[0x48, 0x83, 0xf1, 0x7f]),
        ("xor rcx, 128", |a| { a.alu_ri(XOR, RCX, 128, true); }, &[0x48, 0x81, 0xf1, 0x80, 0x00, 0x00, 0x00]),
        ("xor rcx, -128", |a| { a.alu_ri(XOR, RCX, -128, true); }, &[0x48, 0x83, 0xf1, 0x80]),
        ("xor rcx, -129", |a| { a.alu_ri(XOR, RCX, -129, true); }, &[0x48, 0x81, 0xf1, 0x7f, 0xff, 0xff, 0xff]),
        ("xor rcx, 4096", |a| { a.alu_ri(XOR, RCX, 4096, true); }, &[0x48, 0x81, 0xf1, 0x00, 0x10, 0x00, 0x00]),
        ("xor rcx, 2147483647", |a| { a.alu_ri(XOR, RCX, 2147483647, true); }, &[0x48, 0x81, 0xf1, 0xff, 0xff, 0xff, 0x7f]),
        ("xor rcx, -2147483648", |a| { a.alu_ri(XOR, RCX, -2147483648, true); }, &[0x48, 0x81, 0xf1, 0x00, 0x00, 0x00, 0x80]),
        ("cmp rcx, 5", |a| { a.alu_ri(CMP, RCX, 5, true); }, &[0x48, 0x83, 0xf9, 0x05]),
        ("cmp rcx, -1", |a| { a.alu_ri(CMP, RCX, -1, true); }, &[0x48, 0x83, 0xf9, 0xff]),
        ("cmp rcx, 127", |a| { a.alu_ri(CMP, RCX, 127, true); }, &[0x48, 0x83, 0xf9, 0x7f]),
        ("cmp rcx, 128", |a| { a.alu_ri(CMP, RCX, 128, true); }, &[0x48, 0x81, 0xf9, 0x80, 0x00, 0x00, 0x00]),
        ("cmp rcx, -128", |a| { a.alu_ri(CMP, RCX, -128, true); }, &[0x48, 0x83, 0xf9, 0x80]),
        ("cmp rcx, -129", |a| { a.alu_ri(CMP, RCX, -129, true); }, &[0x48, 0x81, 0xf9, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp rcx, 4096", |a| { a.alu_ri(CMP, RCX, 4096, true); }, &[0x48, 0x81, 0xf9, 0x00, 0x10, 0x00, 0x00]),
        ("cmp rcx, 2147483647", |a| { a.alu_ri(CMP, RCX, 2147483647, true); }, &[0x48, 0x81, 0xf9, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp rcx, -2147483648", |a| { a.alu_ri(CMP, RCX, -2147483648, true); }, &[0x48, 0x81, 0xf9, 0x00, 0x00, 0x00, 0x80]),
        ("add rdx, 5", |a| { a.alu_ri(ADD, RDX, 5, true); }, &[0x48, 0x83, 0xc2, 0x05]),
        ("add rdx, -1", |a| { a.alu_ri(ADD, RDX, -1, true); }, &[0x48, 0x83, 0xc2, 0xff]),
        ("add rdx, 127", |a| { a.alu_ri(ADD, RDX, 127, true); }, &[0x48, 0x83, 0xc2, 0x7f]),
        ("add rdx, 128", |a| { a.alu_ri(ADD, RDX, 128, true); }, &[0x48, 0x81, 0xc2, 0x80, 0x00, 0x00, 0x00]),
        ("add rdx, -128", |a| { a.alu_ri(ADD, RDX, -128, true); }, &[0x48, 0x83, 0xc2, 0x80]),
        ("add rdx, -129", |a| { a.alu_ri(ADD, RDX, -129, true); }, &[0x48, 0x81, 0xc2, 0x7f, 0xff, 0xff, 0xff]),
        ("add rdx, 4096", |a| { a.alu_ri(ADD, RDX, 4096, true); }, &[0x48, 0x81, 0xc2, 0x00, 0x10, 0x00, 0x00]),
        ("add rdx, 2147483647", |a| { a.alu_ri(ADD, RDX, 2147483647, true); }, &[0x48, 0x81, 0xc2, 0xff, 0xff, 0xff, 0x7f]),
        ("add rdx, -2147483648", |a| { a.alu_ri(ADD, RDX, -2147483648, true); }, &[0x48, 0x81, 0xc2, 0x00, 0x00, 0x00, 0x80]),
        ("or rdx, 5", |a| { a.alu_ri(OR, RDX, 5, true); }, &[0x48, 0x83, 0xca, 0x05]),
        ("or rdx, -1", |a| { a.alu_ri(OR, RDX, -1, true); }, &[0x48, 0x83, 0xca, 0xff]),
        ("or rdx, 127", |a| { a.alu_ri(OR, RDX, 127, true); }, &[0x48, 0x83, 0xca, 0x7f]),
        ("or rdx, 128", |a| { a.alu_ri(OR, RDX, 128, true); }, &[0x48, 0x81, 0xca, 0x80, 0x00, 0x00, 0x00]),
        ("or rdx, -128", |a| { a.alu_ri(OR, RDX, -128, true); }, &[0x48, 0x83, 0xca, 0x80]),
        ("or rdx, -129", |a| { a.alu_ri(OR, RDX, -129, true); }, &[0x48, 0x81, 0xca, 0x7f, 0xff, 0xff, 0xff]),
        ("or rdx, 4096", |a| { a.alu_ri(OR, RDX, 4096, true); }, &[0x48, 0x81, 0xca, 0x00, 0x10, 0x00, 0x00]),
        ("or rdx, 2147483647", |a| { a.alu_ri(OR, RDX, 2147483647, true); }, &[0x48, 0x81, 0xca, 0xff, 0xff, 0xff, 0x7f]),
        ("or rdx, -2147483648", |a| { a.alu_ri(OR, RDX, -2147483648, true); }, &[0x48, 0x81, 0xca, 0x00, 0x00, 0x00, 0x80]),
        ("and rdx, 5", |a| { a.alu_ri(AND, RDX, 5, true); }, &[0x48, 0x83, 0xe2, 0x05]),
        ("and rdx, -1", |a| { a.alu_ri(AND, RDX, -1, true); }, &[0x48, 0x83, 0xe2, 0xff]),
        ("and rdx, 127", |a| { a.alu_ri(AND, RDX, 127, true); }, &[0x48, 0x83, 0xe2, 0x7f]),
        ("and rdx, 128", |a| { a.alu_ri(AND, RDX, 128, true); }, &[0x48, 0x81, 0xe2, 0x80, 0x00, 0x00, 0x00]),
        ("and rdx, -128", |a| { a.alu_ri(AND, RDX, -128, true); }, &[0x48, 0x83, 0xe2, 0x80]),
        ("and rdx, -129", |a| { a.alu_ri(AND, RDX, -129, true); }, &[0x48, 0x81, 0xe2, 0x7f, 0xff, 0xff, 0xff]),
        ("and rdx, 4096", |a| { a.alu_ri(AND, RDX, 4096, true); }, &[0x48, 0x81, 0xe2, 0x00, 0x10, 0x00, 0x00]),
        ("and rdx, 2147483647", |a| { a.alu_ri(AND, RDX, 2147483647, true); }, &[0x48, 0x81, 0xe2, 0xff, 0xff, 0xff, 0x7f]),
        ("and rdx, -2147483648", |a| { a.alu_ri(AND, RDX, -2147483648, true); }, &[0x48, 0x81, 0xe2, 0x00, 0x00, 0x00, 0x80]),
        ("sub rdx, 5", |a| { a.alu_ri(SUB, RDX, 5, true); }, &[0x48, 0x83, 0xea, 0x05]),
        ("sub rdx, -1", |a| { a.alu_ri(SUB, RDX, -1, true); }, &[0x48, 0x83, 0xea, 0xff]),
        ("sub rdx, 127", |a| { a.alu_ri(SUB, RDX, 127, true); }, &[0x48, 0x83, 0xea, 0x7f]),
        ("sub rdx, 128", |a| { a.alu_ri(SUB, RDX, 128, true); }, &[0x48, 0x81, 0xea, 0x80, 0x00, 0x00, 0x00]),
        ("sub rdx, -128", |a| { a.alu_ri(SUB, RDX, -128, true); }, &[0x48, 0x83, 0xea, 0x80]),
        ("sub rdx, -129", |a| { a.alu_ri(SUB, RDX, -129, true); }, &[0x48, 0x81, 0xea, 0x7f, 0xff, 0xff, 0xff]),
        ("sub rdx, 4096", |a| { a.alu_ri(SUB, RDX, 4096, true); }, &[0x48, 0x81, 0xea, 0x00, 0x10, 0x00, 0x00]),
        ("sub rdx, 2147483647", |a| { a.alu_ri(SUB, RDX, 2147483647, true); }, &[0x48, 0x81, 0xea, 0xff, 0xff, 0xff, 0x7f]),
        ("sub rdx, -2147483648", |a| { a.alu_ri(SUB, RDX, -2147483648, true); }, &[0x48, 0x81, 0xea, 0x00, 0x00, 0x00, 0x80]),
        ("xor rdx, 5", |a| { a.alu_ri(XOR, RDX, 5, true); }, &[0x48, 0x83, 0xf2, 0x05]),
        ("xor rdx, -1", |a| { a.alu_ri(XOR, RDX, -1, true); }, &[0x48, 0x83, 0xf2, 0xff]),
        ("xor rdx, 127", |a| { a.alu_ri(XOR, RDX, 127, true); }, &[0x48, 0x83, 0xf2, 0x7f]),
        ("xor rdx, 128", |a| { a.alu_ri(XOR, RDX, 128, true); }, &[0x48, 0x81, 0xf2, 0x80, 0x00, 0x00, 0x00]),
        ("xor rdx, -128", |a| { a.alu_ri(XOR, RDX, -128, true); }, &[0x48, 0x83, 0xf2, 0x80]),
        ("xor rdx, -129", |a| { a.alu_ri(XOR, RDX, -129, true); }, &[0x48, 0x81, 0xf2, 0x7f, 0xff, 0xff, 0xff]),
        ("xor rdx, 4096", |a| { a.alu_ri(XOR, RDX, 4096, true); }, &[0x48, 0x81, 0xf2, 0x00, 0x10, 0x00, 0x00]),
        ("xor rdx, 2147483647", |a| { a.alu_ri(XOR, RDX, 2147483647, true); }, &[0x48, 0x81, 0xf2, 0xff, 0xff, 0xff, 0x7f]),
        ("xor rdx, -2147483648", |a| { a.alu_ri(XOR, RDX, -2147483648, true); }, &[0x48, 0x81, 0xf2, 0x00, 0x00, 0x00, 0x80]),
        ("cmp rdx, 5", |a| { a.alu_ri(CMP, RDX, 5, true); }, &[0x48, 0x83, 0xfa, 0x05]),
        ("cmp rdx, -1", |a| { a.alu_ri(CMP, RDX, -1, true); }, &[0x48, 0x83, 0xfa, 0xff]),
        ("cmp rdx, 127", |a| { a.alu_ri(CMP, RDX, 127, true); }, &[0x48, 0x83, 0xfa, 0x7f]),
        ("cmp rdx, 128", |a| { a.alu_ri(CMP, RDX, 128, true); }, &[0x48, 0x81, 0xfa, 0x80, 0x00, 0x00, 0x00]),
        ("cmp rdx, -128", |a| { a.alu_ri(CMP, RDX, -128, true); }, &[0x48, 0x83, 0xfa, 0x80]),
        ("cmp rdx, -129", |a| { a.alu_ri(CMP, RDX, -129, true); }, &[0x48, 0x81, 0xfa, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp rdx, 4096", |a| { a.alu_ri(CMP, RDX, 4096, true); }, &[0x48, 0x81, 0xfa, 0x00, 0x10, 0x00, 0x00]),
        ("cmp rdx, 2147483647", |a| { a.alu_ri(CMP, RDX, 2147483647, true); }, &[0x48, 0x81, 0xfa, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp rdx, -2147483648", |a| { a.alu_ri(CMP, RDX, -2147483648, true); }, &[0x48, 0x81, 0xfa, 0x00, 0x00, 0x00, 0x80]),
        ("add rsp, 5", |a| { a.alu_ri(ADD, RSP, 5, true); }, &[0x48, 0x83, 0xc4, 0x05]),
        ("add rsp, -1", |a| { a.alu_ri(ADD, RSP, -1, true); }, &[0x48, 0x83, 0xc4, 0xff]),
        ("add rsp, 127", |a| { a.alu_ri(ADD, RSP, 127, true); }, &[0x48, 0x83, 0xc4, 0x7f]),
        ("add rsp, 128", |a| { a.alu_ri(ADD, RSP, 128, true); }, &[0x48, 0x81, 0xc4, 0x80, 0x00, 0x00, 0x00]),
        ("add rsp, -128", |a| { a.alu_ri(ADD, RSP, -128, true); }, &[0x48, 0x83, 0xc4, 0x80]),
        ("add rsp, -129", |a| { a.alu_ri(ADD, RSP, -129, true); }, &[0x48, 0x81, 0xc4, 0x7f, 0xff, 0xff, 0xff]),
        ("add rsp, 4096", |a| { a.alu_ri(ADD, RSP, 4096, true); }, &[0x48, 0x81, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("add rsp, 2147483647", |a| { a.alu_ri(ADD, RSP, 2147483647, true); }, &[0x48, 0x81, 0xc4, 0xff, 0xff, 0xff, 0x7f]),
        ("add rsp, -2147483648", |a| { a.alu_ri(ADD, RSP, -2147483648, true); }, &[0x48, 0x81, 0xc4, 0x00, 0x00, 0x00, 0x80]),
        ("or rsp, 5", |a| { a.alu_ri(OR, RSP, 5, true); }, &[0x48, 0x83, 0xcc, 0x05]),
        ("or rsp, -1", |a| { a.alu_ri(OR, RSP, -1, true); }, &[0x48, 0x83, 0xcc, 0xff]),
        ("or rsp, 127", |a| { a.alu_ri(OR, RSP, 127, true); }, &[0x48, 0x83, 0xcc, 0x7f]),
        ("or rsp, 128", |a| { a.alu_ri(OR, RSP, 128, true); }, &[0x48, 0x81, 0xcc, 0x80, 0x00, 0x00, 0x00]),
        ("or rsp, -128", |a| { a.alu_ri(OR, RSP, -128, true); }, &[0x48, 0x83, 0xcc, 0x80]),
        ("or rsp, -129", |a| { a.alu_ri(OR, RSP, -129, true); }, &[0x48, 0x81, 0xcc, 0x7f, 0xff, 0xff, 0xff]),
        ("or rsp, 4096", |a| { a.alu_ri(OR, RSP, 4096, true); }, &[0x48, 0x81, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("or rsp, 2147483647", |a| { a.alu_ri(OR, RSP, 2147483647, true); }, &[0x48, 0x81, 0xcc, 0xff, 0xff, 0xff, 0x7f]),
        ("or rsp, -2147483648", |a| { a.alu_ri(OR, RSP, -2147483648, true); }, &[0x48, 0x81, 0xcc, 0x00, 0x00, 0x00, 0x80]),
        ("and rsp, 5", |a| { a.alu_ri(AND, RSP, 5, true); }, &[0x48, 0x83, 0xe4, 0x05]),
        ("and rsp, -1", |a| { a.alu_ri(AND, RSP, -1, true); }, &[0x48, 0x83, 0xe4, 0xff]),
        ("and rsp, 127", |a| { a.alu_ri(AND, RSP, 127, true); }, &[0x48, 0x83, 0xe4, 0x7f]),
        ("and rsp, 128", |a| { a.alu_ri(AND, RSP, 128, true); }, &[0x48, 0x81, 0xe4, 0x80, 0x00, 0x00, 0x00]),
        ("and rsp, -128", |a| { a.alu_ri(AND, RSP, -128, true); }, &[0x48, 0x83, 0xe4, 0x80]),
        ("and rsp, -129", |a| { a.alu_ri(AND, RSP, -129, true); }, &[0x48, 0x81, 0xe4, 0x7f, 0xff, 0xff, 0xff]),
        ("and rsp, 4096", |a| { a.alu_ri(AND, RSP, 4096, true); }, &[0x48, 0x81, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("and rsp, 2147483647", |a| { a.alu_ri(AND, RSP, 2147483647, true); }, &[0x48, 0x81, 0xe4, 0xff, 0xff, 0xff, 0x7f]),
        ("and rsp, -2147483648", |a| { a.alu_ri(AND, RSP, -2147483648, true); }, &[0x48, 0x81, 0xe4, 0x00, 0x00, 0x00, 0x80]),
        ("sub rsp, 5", |a| { a.alu_ri(SUB, RSP, 5, true); }, &[0x48, 0x83, 0xec, 0x05]),
        ("sub rsp, -1", |a| { a.alu_ri(SUB, RSP, -1, true); }, &[0x48, 0x83, 0xec, 0xff]),
        ("sub rsp, 127", |a| { a.alu_ri(SUB, RSP, 127, true); }, &[0x48, 0x83, 0xec, 0x7f]),
        ("sub rsp, 128", |a| { a.alu_ri(SUB, RSP, 128, true); }, &[0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00]),
        ("sub rsp, -128", |a| { a.alu_ri(SUB, RSP, -128, true); }, &[0x48, 0x83, 0xec, 0x80]),
        ("sub rsp, -129", |a| { a.alu_ri(SUB, RSP, -129, true); }, &[0x48, 0x81, 0xec, 0x7f, 0xff, 0xff, 0xff]),
        ("sub rsp, 4096", |a| { a.alu_ri(SUB, RSP, 4096, true); }, &[0x48, 0x81, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("sub rsp, 2147483647", |a| { a.alu_ri(SUB, RSP, 2147483647, true); }, &[0x48, 0x81, 0xec, 0xff, 0xff, 0xff, 0x7f]),
        ("sub rsp, -2147483648", |a| { a.alu_ri(SUB, RSP, -2147483648, true); }, &[0x48, 0x81, 0xec, 0x00, 0x00, 0x00, 0x80]),
        ("xor rsp, 5", |a| { a.alu_ri(XOR, RSP, 5, true); }, &[0x48, 0x83, 0xf4, 0x05]),
        ("xor rsp, -1", |a| { a.alu_ri(XOR, RSP, -1, true); }, &[0x48, 0x83, 0xf4, 0xff]),
        ("xor rsp, 127", |a| { a.alu_ri(XOR, RSP, 127, true); }, &[0x48, 0x83, 0xf4, 0x7f]),
        ("xor rsp, 128", |a| { a.alu_ri(XOR, RSP, 128, true); }, &[0x48, 0x81, 0xf4, 0x80, 0x00, 0x00, 0x00]),
        ("xor rsp, -128", |a| { a.alu_ri(XOR, RSP, -128, true); }, &[0x48, 0x83, 0xf4, 0x80]),
        ("xor rsp, -129", |a| { a.alu_ri(XOR, RSP, -129, true); }, &[0x48, 0x81, 0xf4, 0x7f, 0xff, 0xff, 0xff]),
        ("xor rsp, 4096", |a| { a.alu_ri(XOR, RSP, 4096, true); }, &[0x48, 0x81, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("xor rsp, 2147483647", |a| { a.alu_ri(XOR, RSP, 2147483647, true); }, &[0x48, 0x81, 0xf4, 0xff, 0xff, 0xff, 0x7f]),
        ("xor rsp, -2147483648", |a| { a.alu_ri(XOR, RSP, -2147483648, true); }, &[0x48, 0x81, 0xf4, 0x00, 0x00, 0x00, 0x80]),
        ("cmp rsp, 5", |a| { a.alu_ri(CMP, RSP, 5, true); }, &[0x48, 0x83, 0xfc, 0x05]),
        ("cmp rsp, -1", |a| { a.alu_ri(CMP, RSP, -1, true); }, &[0x48, 0x83, 0xfc, 0xff]),
        ("cmp rsp, 127", |a| { a.alu_ri(CMP, RSP, 127, true); }, &[0x48, 0x83, 0xfc, 0x7f]),
        ("cmp rsp, 128", |a| { a.alu_ri(CMP, RSP, 128, true); }, &[0x48, 0x81, 0xfc, 0x80, 0x00, 0x00, 0x00]),
        ("cmp rsp, -128", |a| { a.alu_ri(CMP, RSP, -128, true); }, &[0x48, 0x83, 0xfc, 0x80]),
        ("cmp rsp, -129", |a| { a.alu_ri(CMP, RSP, -129, true); }, &[0x48, 0x81, 0xfc, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp rsp, 4096", |a| { a.alu_ri(CMP, RSP, 4096, true); }, &[0x48, 0x81, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("cmp rsp, 2147483647", |a| { a.alu_ri(CMP, RSP, 2147483647, true); }, &[0x48, 0x81, 0xfc, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp rsp, -2147483648", |a| { a.alu_ri(CMP, RSP, -2147483648, true); }, &[0x48, 0x81, 0xfc, 0x00, 0x00, 0x00, 0x80]),
        ("add rbp, 5", |a| { a.alu_ri(ADD, RBP, 5, true); }, &[0x48, 0x83, 0xc5, 0x05]),
        ("add rbp, -1", |a| { a.alu_ri(ADD, RBP, -1, true); }, &[0x48, 0x83, 0xc5, 0xff]),
        ("add rbp, 127", |a| { a.alu_ri(ADD, RBP, 127, true); }, &[0x48, 0x83, 0xc5, 0x7f]),
        ("add rbp, 128", |a| { a.alu_ri(ADD, RBP, 128, true); }, &[0x48, 0x81, 0xc5, 0x80, 0x00, 0x00, 0x00]),
        ("add rbp, -128", |a| { a.alu_ri(ADD, RBP, -128, true); }, &[0x48, 0x83, 0xc5, 0x80]),
        ("add rbp, -129", |a| { a.alu_ri(ADD, RBP, -129, true); }, &[0x48, 0x81, 0xc5, 0x7f, 0xff, 0xff, 0xff]),
        ("add rbp, 4096", |a| { a.alu_ri(ADD, RBP, 4096, true); }, &[0x48, 0x81, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("add rbp, 2147483647", |a| { a.alu_ri(ADD, RBP, 2147483647, true); }, &[0x48, 0x81, 0xc5, 0xff, 0xff, 0xff, 0x7f]),
        ("add rbp, -2147483648", |a| { a.alu_ri(ADD, RBP, -2147483648, true); }, &[0x48, 0x81, 0xc5, 0x00, 0x00, 0x00, 0x80]),
        ("or rbp, 5", |a| { a.alu_ri(OR, RBP, 5, true); }, &[0x48, 0x83, 0xcd, 0x05]),
        ("or rbp, -1", |a| { a.alu_ri(OR, RBP, -1, true); }, &[0x48, 0x83, 0xcd, 0xff]),
        ("or rbp, 127", |a| { a.alu_ri(OR, RBP, 127, true); }, &[0x48, 0x83, 0xcd, 0x7f]),
        ("or rbp, 128", |a| { a.alu_ri(OR, RBP, 128, true); }, &[0x48, 0x81, 0xcd, 0x80, 0x00, 0x00, 0x00]),
        ("or rbp, -128", |a| { a.alu_ri(OR, RBP, -128, true); }, &[0x48, 0x83, 0xcd, 0x80]),
        ("or rbp, -129", |a| { a.alu_ri(OR, RBP, -129, true); }, &[0x48, 0x81, 0xcd, 0x7f, 0xff, 0xff, 0xff]),
        ("or rbp, 4096", |a| { a.alu_ri(OR, RBP, 4096, true); }, &[0x48, 0x81, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("or rbp, 2147483647", |a| { a.alu_ri(OR, RBP, 2147483647, true); }, &[0x48, 0x81, 0xcd, 0xff, 0xff, 0xff, 0x7f]),
        ("or rbp, -2147483648", |a| { a.alu_ri(OR, RBP, -2147483648, true); }, &[0x48, 0x81, 0xcd, 0x00, 0x00, 0x00, 0x80]),
        ("and rbp, 5", |a| { a.alu_ri(AND, RBP, 5, true); }, &[0x48, 0x83, 0xe5, 0x05]),
        ("and rbp, -1", |a| { a.alu_ri(AND, RBP, -1, true); }, &[0x48, 0x83, 0xe5, 0xff]),
        ("and rbp, 127", |a| { a.alu_ri(AND, RBP, 127, true); }, &[0x48, 0x83, 0xe5, 0x7f]),
        ("and rbp, 128", |a| { a.alu_ri(AND, RBP, 128, true); }, &[0x48, 0x81, 0xe5, 0x80, 0x00, 0x00, 0x00]),
        ("and rbp, -128", |a| { a.alu_ri(AND, RBP, -128, true); }, &[0x48, 0x83, 0xe5, 0x80]),
        ("and rbp, -129", |a| { a.alu_ri(AND, RBP, -129, true); }, &[0x48, 0x81, 0xe5, 0x7f, 0xff, 0xff, 0xff]),
        ("and rbp, 4096", |a| { a.alu_ri(AND, RBP, 4096, true); }, &[0x48, 0x81, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("and rbp, 2147483647", |a| { a.alu_ri(AND, RBP, 2147483647, true); }, &[0x48, 0x81, 0xe5, 0xff, 0xff, 0xff, 0x7f]),
        ("and rbp, -2147483648", |a| { a.alu_ri(AND, RBP, -2147483648, true); }, &[0x48, 0x81, 0xe5, 0x00, 0x00, 0x00, 0x80]),
        ("sub rbp, 5", |a| { a.alu_ri(SUB, RBP, 5, true); }, &[0x48, 0x83, 0xed, 0x05]),
        ("sub rbp, -1", |a| { a.alu_ri(SUB, RBP, -1, true); }, &[0x48, 0x83, 0xed, 0xff]),
        ("sub rbp, 127", |a| { a.alu_ri(SUB, RBP, 127, true); }, &[0x48, 0x83, 0xed, 0x7f]),
        ("sub rbp, 128", |a| { a.alu_ri(SUB, RBP, 128, true); }, &[0x48, 0x81, 0xed, 0x80, 0x00, 0x00, 0x00]),
        ("sub rbp, -128", |a| { a.alu_ri(SUB, RBP, -128, true); }, &[0x48, 0x83, 0xed, 0x80]),
        ("sub rbp, -129", |a| { a.alu_ri(SUB, RBP, -129, true); }, &[0x48, 0x81, 0xed, 0x7f, 0xff, 0xff, 0xff]),
        ("sub rbp, 4096", |a| { a.alu_ri(SUB, RBP, 4096, true); }, &[0x48, 0x81, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("sub rbp, 2147483647", |a| { a.alu_ri(SUB, RBP, 2147483647, true); }, &[0x48, 0x81, 0xed, 0xff, 0xff, 0xff, 0x7f]),
        ("sub rbp, -2147483648", |a| { a.alu_ri(SUB, RBP, -2147483648, true); }, &[0x48, 0x81, 0xed, 0x00, 0x00, 0x00, 0x80]),
        ("xor rbp, 5", |a| { a.alu_ri(XOR, RBP, 5, true); }, &[0x48, 0x83, 0xf5, 0x05]),
        ("xor rbp, -1", |a| { a.alu_ri(XOR, RBP, -1, true); }, &[0x48, 0x83, 0xf5, 0xff]),
        ("xor rbp, 127", |a| { a.alu_ri(XOR, RBP, 127, true); }, &[0x48, 0x83, 0xf5, 0x7f]),
        ("xor rbp, 128", |a| { a.alu_ri(XOR, RBP, 128, true); }, &[0x48, 0x81, 0xf5, 0x80, 0x00, 0x00, 0x00]),
        ("xor rbp, -128", |a| { a.alu_ri(XOR, RBP, -128, true); }, &[0x48, 0x83, 0xf5, 0x80]),
        ("xor rbp, -129", |a| { a.alu_ri(XOR, RBP, -129, true); }, &[0x48, 0x81, 0xf5, 0x7f, 0xff, 0xff, 0xff]),
        ("xor rbp, 4096", |a| { a.alu_ri(XOR, RBP, 4096, true); }, &[0x48, 0x81, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("xor rbp, 2147483647", |a| { a.alu_ri(XOR, RBP, 2147483647, true); }, &[0x48, 0x81, 0xf5, 0xff, 0xff, 0xff, 0x7f]),
        ("xor rbp, -2147483648", |a| { a.alu_ri(XOR, RBP, -2147483648, true); }, &[0x48, 0x81, 0xf5, 0x00, 0x00, 0x00, 0x80]),
        ("cmp rbp, 5", |a| { a.alu_ri(CMP, RBP, 5, true); }, &[0x48, 0x83, 0xfd, 0x05]),
        ("cmp rbp, -1", |a| { a.alu_ri(CMP, RBP, -1, true); }, &[0x48, 0x83, 0xfd, 0xff]),
        ("cmp rbp, 127", |a| { a.alu_ri(CMP, RBP, 127, true); }, &[0x48, 0x83, 0xfd, 0x7f]),
        ("cmp rbp, 128", |a| { a.alu_ri(CMP, RBP, 128, true); }, &[0x48, 0x81, 0xfd, 0x80, 0x00, 0x00, 0x00]),
        ("cmp rbp, -128", |a| { a.alu_ri(CMP, RBP, -128, true); }, &[0x48, 0x83, 0xfd, 0x80]),
        ("cmp rbp, -129", |a| { a.alu_ri(CMP, RBP, -129, true); }, &[0x48, 0x81, 0xfd, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp rbp, 4096", |a| { a.alu_ri(CMP, RBP, 4096, true); }, &[0x48, 0x81, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("cmp rbp, 2147483647", |a| { a.alu_ri(CMP, RBP, 2147483647, true); }, &[0x48, 0x81, 0xfd, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp rbp, -2147483648", |a| { a.alu_ri(CMP, RBP, -2147483648, true); }, &[0x48, 0x81, 0xfd, 0x00, 0x00, 0x00, 0x80]),
        ("add r8, 5", |a| { a.alu_ri(ADD, R8, 5, true); }, &[0x49, 0x83, 0xc0, 0x05]),
        ("add r8, -1", |a| { a.alu_ri(ADD, R8, -1, true); }, &[0x49, 0x83, 0xc0, 0xff]),
        ("add r8, 127", |a| { a.alu_ri(ADD, R8, 127, true); }, &[0x49, 0x83, 0xc0, 0x7f]),
        ("add r8, 128", |a| { a.alu_ri(ADD, R8, 128, true); }, &[0x49, 0x81, 0xc0, 0x80, 0x00, 0x00, 0x00]),
        ("add r8, -128", |a| { a.alu_ri(ADD, R8, -128, true); }, &[0x49, 0x83, 0xc0, 0x80]),
        ("add r8, -129", |a| { a.alu_ri(ADD, R8, -129, true); }, &[0x49, 0x81, 0xc0, 0x7f, 0xff, 0xff, 0xff]),
        ("add r8, 4096", |a| { a.alu_ri(ADD, R8, 4096, true); }, &[0x49, 0x81, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("add r8, 2147483647", |a| { a.alu_ri(ADD, R8, 2147483647, true); }, &[0x49, 0x81, 0xc0, 0xff, 0xff, 0xff, 0x7f]),
        ("add r8, -2147483648", |a| { a.alu_ri(ADD, R8, -2147483648, true); }, &[0x49, 0x81, 0xc0, 0x00, 0x00, 0x00, 0x80]),
        ("or r8, 5", |a| { a.alu_ri(OR, R8, 5, true); }, &[0x49, 0x83, 0xc8, 0x05]),
        ("or r8, -1", |a| { a.alu_ri(OR, R8, -1, true); }, &[0x49, 0x83, 0xc8, 0xff]),
        ("or r8, 127", |a| { a.alu_ri(OR, R8, 127, true); }, &[0x49, 0x83, 0xc8, 0x7f]),
        ("or r8, 128", |a| { a.alu_ri(OR, R8, 128, true); }, &[0x49, 0x81, 0xc8, 0x80, 0x00, 0x00, 0x00]),
        ("or r8, -128", |a| { a.alu_ri(OR, R8, -128, true); }, &[0x49, 0x83, 0xc8, 0x80]),
        ("or r8, -129", |a| { a.alu_ri(OR, R8, -129, true); }, &[0x49, 0x81, 0xc8, 0x7f, 0xff, 0xff, 0xff]),
        ("or r8, 4096", |a| { a.alu_ri(OR, R8, 4096, true); }, &[0x49, 0x81, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("or r8, 2147483647", |a| { a.alu_ri(OR, R8, 2147483647, true); }, &[0x49, 0x81, 0xc8, 0xff, 0xff, 0xff, 0x7f]),
        ("or r8, -2147483648", |a| { a.alu_ri(OR, R8, -2147483648, true); }, &[0x49, 0x81, 0xc8, 0x00, 0x00, 0x00, 0x80]),
        ("and r8, 5", |a| { a.alu_ri(AND, R8, 5, true); }, &[0x49, 0x83, 0xe0, 0x05]),
        ("and r8, -1", |a| { a.alu_ri(AND, R8, -1, true); }, &[0x49, 0x83, 0xe0, 0xff]),
        ("and r8, 127", |a| { a.alu_ri(AND, R8, 127, true); }, &[0x49, 0x83, 0xe0, 0x7f]),
        ("and r8, 128", |a| { a.alu_ri(AND, R8, 128, true); }, &[0x49, 0x81, 0xe0, 0x80, 0x00, 0x00, 0x00]),
        ("and r8, -128", |a| { a.alu_ri(AND, R8, -128, true); }, &[0x49, 0x83, 0xe0, 0x80]),
        ("and r8, -129", |a| { a.alu_ri(AND, R8, -129, true); }, &[0x49, 0x81, 0xe0, 0x7f, 0xff, 0xff, 0xff]),
        ("and r8, 4096", |a| { a.alu_ri(AND, R8, 4096, true); }, &[0x49, 0x81, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("and r8, 2147483647", |a| { a.alu_ri(AND, R8, 2147483647, true); }, &[0x49, 0x81, 0xe0, 0xff, 0xff, 0xff, 0x7f]),
        ("and r8, -2147483648", |a| { a.alu_ri(AND, R8, -2147483648, true); }, &[0x49, 0x81, 0xe0, 0x00, 0x00, 0x00, 0x80]),
        ("sub r8, 5", |a| { a.alu_ri(SUB, R8, 5, true); }, &[0x49, 0x83, 0xe8, 0x05]),
        ("sub r8, -1", |a| { a.alu_ri(SUB, R8, -1, true); }, &[0x49, 0x83, 0xe8, 0xff]),
        ("sub r8, 127", |a| { a.alu_ri(SUB, R8, 127, true); }, &[0x49, 0x83, 0xe8, 0x7f]),
        ("sub r8, 128", |a| { a.alu_ri(SUB, R8, 128, true); }, &[0x49, 0x81, 0xe8, 0x80, 0x00, 0x00, 0x00]),
        ("sub r8, -128", |a| { a.alu_ri(SUB, R8, -128, true); }, &[0x49, 0x83, 0xe8, 0x80]),
        ("sub r8, -129", |a| { a.alu_ri(SUB, R8, -129, true); }, &[0x49, 0x81, 0xe8, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r8, 4096", |a| { a.alu_ri(SUB, R8, 4096, true); }, &[0x49, 0x81, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("sub r8, 2147483647", |a| { a.alu_ri(SUB, R8, 2147483647, true); }, &[0x49, 0x81, 0xe8, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r8, -2147483648", |a| { a.alu_ri(SUB, R8, -2147483648, true); }, &[0x49, 0x81, 0xe8, 0x00, 0x00, 0x00, 0x80]),
        ("xor r8, 5", |a| { a.alu_ri(XOR, R8, 5, true); }, &[0x49, 0x83, 0xf0, 0x05]),
        ("xor r8, -1", |a| { a.alu_ri(XOR, R8, -1, true); }, &[0x49, 0x83, 0xf0, 0xff]),
        ("xor r8, 127", |a| { a.alu_ri(XOR, R8, 127, true); }, &[0x49, 0x83, 0xf0, 0x7f]),
        ("xor r8, 128", |a| { a.alu_ri(XOR, R8, 128, true); }, &[0x49, 0x81, 0xf0, 0x80, 0x00, 0x00, 0x00]),
        ("xor r8, -128", |a| { a.alu_ri(XOR, R8, -128, true); }, &[0x49, 0x83, 0xf0, 0x80]),
        ("xor r8, -129", |a| { a.alu_ri(XOR, R8, -129, true); }, &[0x49, 0x81, 0xf0, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r8, 4096", |a| { a.alu_ri(XOR, R8, 4096, true); }, &[0x49, 0x81, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("xor r8, 2147483647", |a| { a.alu_ri(XOR, R8, 2147483647, true); }, &[0x49, 0x81, 0xf0, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r8, -2147483648", |a| { a.alu_ri(XOR, R8, -2147483648, true); }, &[0x49, 0x81, 0xf0, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r8, 5", |a| { a.alu_ri(CMP, R8, 5, true); }, &[0x49, 0x83, 0xf8, 0x05]),
        ("cmp r8, -1", |a| { a.alu_ri(CMP, R8, -1, true); }, &[0x49, 0x83, 0xf8, 0xff]),
        ("cmp r8, 127", |a| { a.alu_ri(CMP, R8, 127, true); }, &[0x49, 0x83, 0xf8, 0x7f]),
        ("cmp r8, 128", |a| { a.alu_ri(CMP, R8, 128, true); }, &[0x49, 0x81, 0xf8, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r8, -128", |a| { a.alu_ri(CMP, R8, -128, true); }, &[0x49, 0x83, 0xf8, 0x80]),
        ("cmp r8, -129", |a| { a.alu_ri(CMP, R8, -129, true); }, &[0x49, 0x81, 0xf8, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r8, 4096", |a| { a.alu_ri(CMP, R8, 4096, true); }, &[0x49, 0x81, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r8, 2147483647", |a| { a.alu_ri(CMP, R8, 2147483647, true); }, &[0x49, 0x81, 0xf8, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r8, -2147483648", |a| { a.alu_ri(CMP, R8, -2147483648, true); }, &[0x49, 0x81, 0xf8, 0x00, 0x00, 0x00, 0x80]),
        ("add r12, 5", |a| { a.alu_ri(ADD, R12, 5, true); }, &[0x49, 0x83, 0xc4, 0x05]),
        ("add r12, -1", |a| { a.alu_ri(ADD, R12, -1, true); }, &[0x49, 0x83, 0xc4, 0xff]),
        ("add r12, 127", |a| { a.alu_ri(ADD, R12, 127, true); }, &[0x49, 0x83, 0xc4, 0x7f]),
        ("add r12, 128", |a| { a.alu_ri(ADD, R12, 128, true); }, &[0x49, 0x81, 0xc4, 0x80, 0x00, 0x00, 0x00]),
        ("add r12, -128", |a| { a.alu_ri(ADD, R12, -128, true); }, &[0x49, 0x83, 0xc4, 0x80]),
        ("add r12, -129", |a| { a.alu_ri(ADD, R12, -129, true); }, &[0x49, 0x81, 0xc4, 0x7f, 0xff, 0xff, 0xff]),
        ("add r12, 4096", |a| { a.alu_ri(ADD, R12, 4096, true); }, &[0x49, 0x81, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("add r12, 2147483647", |a| { a.alu_ri(ADD, R12, 2147483647, true); }, &[0x49, 0x81, 0xc4, 0xff, 0xff, 0xff, 0x7f]),
        ("add r12, -2147483648", |a| { a.alu_ri(ADD, R12, -2147483648, true); }, &[0x49, 0x81, 0xc4, 0x00, 0x00, 0x00, 0x80]),
        ("or r12, 5", |a| { a.alu_ri(OR, R12, 5, true); }, &[0x49, 0x83, 0xcc, 0x05]),
        ("or r12, -1", |a| { a.alu_ri(OR, R12, -1, true); }, &[0x49, 0x83, 0xcc, 0xff]),
        ("or r12, 127", |a| { a.alu_ri(OR, R12, 127, true); }, &[0x49, 0x83, 0xcc, 0x7f]),
        ("or r12, 128", |a| { a.alu_ri(OR, R12, 128, true); }, &[0x49, 0x81, 0xcc, 0x80, 0x00, 0x00, 0x00]),
        ("or r12, -128", |a| { a.alu_ri(OR, R12, -128, true); }, &[0x49, 0x83, 0xcc, 0x80]),
        ("or r12, -129", |a| { a.alu_ri(OR, R12, -129, true); }, &[0x49, 0x81, 0xcc, 0x7f, 0xff, 0xff, 0xff]),
        ("or r12, 4096", |a| { a.alu_ri(OR, R12, 4096, true); }, &[0x49, 0x81, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("or r12, 2147483647", |a| { a.alu_ri(OR, R12, 2147483647, true); }, &[0x49, 0x81, 0xcc, 0xff, 0xff, 0xff, 0x7f]),
        ("or r12, -2147483648", |a| { a.alu_ri(OR, R12, -2147483648, true); }, &[0x49, 0x81, 0xcc, 0x00, 0x00, 0x00, 0x80]),
        ("and r12, 5", |a| { a.alu_ri(AND, R12, 5, true); }, &[0x49, 0x83, 0xe4, 0x05]),
        ("and r12, -1", |a| { a.alu_ri(AND, R12, -1, true); }, &[0x49, 0x83, 0xe4, 0xff]),
        ("and r12, 127", |a| { a.alu_ri(AND, R12, 127, true); }, &[0x49, 0x83, 0xe4, 0x7f]),
        ("and r12, 128", |a| { a.alu_ri(AND, R12, 128, true); }, &[0x49, 0x81, 0xe4, 0x80, 0x00, 0x00, 0x00]),
        ("and r12, -128", |a| { a.alu_ri(AND, R12, -128, true); }, &[0x49, 0x83, 0xe4, 0x80]),
        ("and r12, -129", |a| { a.alu_ri(AND, R12, -129, true); }, &[0x49, 0x81, 0xe4, 0x7f, 0xff, 0xff, 0xff]),
        ("and r12, 4096", |a| { a.alu_ri(AND, R12, 4096, true); }, &[0x49, 0x81, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("and r12, 2147483647", |a| { a.alu_ri(AND, R12, 2147483647, true); }, &[0x49, 0x81, 0xe4, 0xff, 0xff, 0xff, 0x7f]),
        ("and r12, -2147483648", |a| { a.alu_ri(AND, R12, -2147483648, true); }, &[0x49, 0x81, 0xe4, 0x00, 0x00, 0x00, 0x80]),
        ("sub r12, 5", |a| { a.alu_ri(SUB, R12, 5, true); }, &[0x49, 0x83, 0xec, 0x05]),
        ("sub r12, -1", |a| { a.alu_ri(SUB, R12, -1, true); }, &[0x49, 0x83, 0xec, 0xff]),
        ("sub r12, 127", |a| { a.alu_ri(SUB, R12, 127, true); }, &[0x49, 0x83, 0xec, 0x7f]),
        ("sub r12, 128", |a| { a.alu_ri(SUB, R12, 128, true); }, &[0x49, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00]),
        ("sub r12, -128", |a| { a.alu_ri(SUB, R12, -128, true); }, &[0x49, 0x83, 0xec, 0x80]),
        ("sub r12, -129", |a| { a.alu_ri(SUB, R12, -129, true); }, &[0x49, 0x81, 0xec, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r12, 4096", |a| { a.alu_ri(SUB, R12, 4096, true); }, &[0x49, 0x81, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("sub r12, 2147483647", |a| { a.alu_ri(SUB, R12, 2147483647, true); }, &[0x49, 0x81, 0xec, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r12, -2147483648", |a| { a.alu_ri(SUB, R12, -2147483648, true); }, &[0x49, 0x81, 0xec, 0x00, 0x00, 0x00, 0x80]),
        ("xor r12, 5", |a| { a.alu_ri(XOR, R12, 5, true); }, &[0x49, 0x83, 0xf4, 0x05]),
        ("xor r12, -1", |a| { a.alu_ri(XOR, R12, -1, true); }, &[0x49, 0x83, 0xf4, 0xff]),
        ("xor r12, 127", |a| { a.alu_ri(XOR, R12, 127, true); }, &[0x49, 0x83, 0xf4, 0x7f]),
        ("xor r12, 128", |a| { a.alu_ri(XOR, R12, 128, true); }, &[0x49, 0x81, 0xf4, 0x80, 0x00, 0x00, 0x00]),
        ("xor r12, -128", |a| { a.alu_ri(XOR, R12, -128, true); }, &[0x49, 0x83, 0xf4, 0x80]),
        ("xor r12, -129", |a| { a.alu_ri(XOR, R12, -129, true); }, &[0x49, 0x81, 0xf4, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r12, 4096", |a| { a.alu_ri(XOR, R12, 4096, true); }, &[0x49, 0x81, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("xor r12, 2147483647", |a| { a.alu_ri(XOR, R12, 2147483647, true); }, &[0x49, 0x81, 0xf4, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r12, -2147483648", |a| { a.alu_ri(XOR, R12, -2147483648, true); }, &[0x49, 0x81, 0xf4, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r12, 5", |a| { a.alu_ri(CMP, R12, 5, true); }, &[0x49, 0x83, 0xfc, 0x05]),
        ("cmp r12, -1", |a| { a.alu_ri(CMP, R12, -1, true); }, &[0x49, 0x83, 0xfc, 0xff]),
        ("cmp r12, 127", |a| { a.alu_ri(CMP, R12, 127, true); }, &[0x49, 0x83, 0xfc, 0x7f]),
        ("cmp r12, 128", |a| { a.alu_ri(CMP, R12, 128, true); }, &[0x49, 0x81, 0xfc, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r12, -128", |a| { a.alu_ri(CMP, R12, -128, true); }, &[0x49, 0x83, 0xfc, 0x80]),
        ("cmp r12, -129", |a| { a.alu_ri(CMP, R12, -129, true); }, &[0x49, 0x81, 0xfc, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r12, 4096", |a| { a.alu_ri(CMP, R12, 4096, true); }, &[0x49, 0x81, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r12, 2147483647", |a| { a.alu_ri(CMP, R12, 2147483647, true); }, &[0x49, 0x81, 0xfc, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r12, -2147483648", |a| { a.alu_ri(CMP, R12, -2147483648, true); }, &[0x49, 0x81, 0xfc, 0x00, 0x00, 0x00, 0x80]),
        ("add r13, 5", |a| { a.alu_ri(ADD, R13, 5, true); }, &[0x49, 0x83, 0xc5, 0x05]),
        ("add r13, -1", |a| { a.alu_ri(ADD, R13, -1, true); }, &[0x49, 0x83, 0xc5, 0xff]),
        ("add r13, 127", |a| { a.alu_ri(ADD, R13, 127, true); }, &[0x49, 0x83, 0xc5, 0x7f]),
        ("add r13, 128", |a| { a.alu_ri(ADD, R13, 128, true); }, &[0x49, 0x81, 0xc5, 0x80, 0x00, 0x00, 0x00]),
        ("add r13, -128", |a| { a.alu_ri(ADD, R13, -128, true); }, &[0x49, 0x83, 0xc5, 0x80]),
        ("add r13, -129", |a| { a.alu_ri(ADD, R13, -129, true); }, &[0x49, 0x81, 0xc5, 0x7f, 0xff, 0xff, 0xff]),
        ("add r13, 4096", |a| { a.alu_ri(ADD, R13, 4096, true); }, &[0x49, 0x81, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("add r13, 2147483647", |a| { a.alu_ri(ADD, R13, 2147483647, true); }, &[0x49, 0x81, 0xc5, 0xff, 0xff, 0xff, 0x7f]),
        ("add r13, -2147483648", |a| { a.alu_ri(ADD, R13, -2147483648, true); }, &[0x49, 0x81, 0xc5, 0x00, 0x00, 0x00, 0x80]),
        ("or r13, 5", |a| { a.alu_ri(OR, R13, 5, true); }, &[0x49, 0x83, 0xcd, 0x05]),
        ("or r13, -1", |a| { a.alu_ri(OR, R13, -1, true); }, &[0x49, 0x83, 0xcd, 0xff]),
        ("or r13, 127", |a| { a.alu_ri(OR, R13, 127, true); }, &[0x49, 0x83, 0xcd, 0x7f]),
        ("or r13, 128", |a| { a.alu_ri(OR, R13, 128, true); }, &[0x49, 0x81, 0xcd, 0x80, 0x00, 0x00, 0x00]),
        ("or r13, -128", |a| { a.alu_ri(OR, R13, -128, true); }, &[0x49, 0x83, 0xcd, 0x80]),
        ("or r13, -129", |a| { a.alu_ri(OR, R13, -129, true); }, &[0x49, 0x81, 0xcd, 0x7f, 0xff, 0xff, 0xff]),
        ("or r13, 4096", |a| { a.alu_ri(OR, R13, 4096, true); }, &[0x49, 0x81, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("or r13, 2147483647", |a| { a.alu_ri(OR, R13, 2147483647, true); }, &[0x49, 0x81, 0xcd, 0xff, 0xff, 0xff, 0x7f]),
        ("or r13, -2147483648", |a| { a.alu_ri(OR, R13, -2147483648, true); }, &[0x49, 0x81, 0xcd, 0x00, 0x00, 0x00, 0x80]),
        ("and r13, 5", |a| { a.alu_ri(AND, R13, 5, true); }, &[0x49, 0x83, 0xe5, 0x05]),
        ("and r13, -1", |a| { a.alu_ri(AND, R13, -1, true); }, &[0x49, 0x83, 0xe5, 0xff]),
        ("and r13, 127", |a| { a.alu_ri(AND, R13, 127, true); }, &[0x49, 0x83, 0xe5, 0x7f]),
        ("and r13, 128", |a| { a.alu_ri(AND, R13, 128, true); }, &[0x49, 0x81, 0xe5, 0x80, 0x00, 0x00, 0x00]),
        ("and r13, -128", |a| { a.alu_ri(AND, R13, -128, true); }, &[0x49, 0x83, 0xe5, 0x80]),
        ("and r13, -129", |a| { a.alu_ri(AND, R13, -129, true); }, &[0x49, 0x81, 0xe5, 0x7f, 0xff, 0xff, 0xff]),
        ("and r13, 4096", |a| { a.alu_ri(AND, R13, 4096, true); }, &[0x49, 0x81, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("and r13, 2147483647", |a| { a.alu_ri(AND, R13, 2147483647, true); }, &[0x49, 0x81, 0xe5, 0xff, 0xff, 0xff, 0x7f]),
        ("and r13, -2147483648", |a| { a.alu_ri(AND, R13, -2147483648, true); }, &[0x49, 0x81, 0xe5, 0x00, 0x00, 0x00, 0x80]),
        ("sub r13, 5", |a| { a.alu_ri(SUB, R13, 5, true); }, &[0x49, 0x83, 0xed, 0x05]),
        ("sub r13, -1", |a| { a.alu_ri(SUB, R13, -1, true); }, &[0x49, 0x83, 0xed, 0xff]),
        ("sub r13, 127", |a| { a.alu_ri(SUB, R13, 127, true); }, &[0x49, 0x83, 0xed, 0x7f]),
        ("sub r13, 128", |a| { a.alu_ri(SUB, R13, 128, true); }, &[0x49, 0x81, 0xed, 0x80, 0x00, 0x00, 0x00]),
        ("sub r13, -128", |a| { a.alu_ri(SUB, R13, -128, true); }, &[0x49, 0x83, 0xed, 0x80]),
        ("sub r13, -129", |a| { a.alu_ri(SUB, R13, -129, true); }, &[0x49, 0x81, 0xed, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r13, 4096", |a| { a.alu_ri(SUB, R13, 4096, true); }, &[0x49, 0x81, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("sub r13, 2147483647", |a| { a.alu_ri(SUB, R13, 2147483647, true); }, &[0x49, 0x81, 0xed, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r13, -2147483648", |a| { a.alu_ri(SUB, R13, -2147483648, true); }, &[0x49, 0x81, 0xed, 0x00, 0x00, 0x00, 0x80]),
        ("xor r13, 5", |a| { a.alu_ri(XOR, R13, 5, true); }, &[0x49, 0x83, 0xf5, 0x05]),
        ("xor r13, -1", |a| { a.alu_ri(XOR, R13, -1, true); }, &[0x49, 0x83, 0xf5, 0xff]),
        ("xor r13, 127", |a| { a.alu_ri(XOR, R13, 127, true); }, &[0x49, 0x83, 0xf5, 0x7f]),
        ("xor r13, 128", |a| { a.alu_ri(XOR, R13, 128, true); }, &[0x49, 0x81, 0xf5, 0x80, 0x00, 0x00, 0x00]),
        ("xor r13, -128", |a| { a.alu_ri(XOR, R13, -128, true); }, &[0x49, 0x83, 0xf5, 0x80]),
        ("xor r13, -129", |a| { a.alu_ri(XOR, R13, -129, true); }, &[0x49, 0x81, 0xf5, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r13, 4096", |a| { a.alu_ri(XOR, R13, 4096, true); }, &[0x49, 0x81, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("xor r13, 2147483647", |a| { a.alu_ri(XOR, R13, 2147483647, true); }, &[0x49, 0x81, 0xf5, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r13, -2147483648", |a| { a.alu_ri(XOR, R13, -2147483648, true); }, &[0x49, 0x81, 0xf5, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r13, 5", |a| { a.alu_ri(CMP, R13, 5, true); }, &[0x49, 0x83, 0xfd, 0x05]),
        ("cmp r13, -1", |a| { a.alu_ri(CMP, R13, -1, true); }, &[0x49, 0x83, 0xfd, 0xff]),
        ("cmp r13, 127", |a| { a.alu_ri(CMP, R13, 127, true); }, &[0x49, 0x83, 0xfd, 0x7f]),
        ("cmp r13, 128", |a| { a.alu_ri(CMP, R13, 128, true); }, &[0x49, 0x81, 0xfd, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r13, -128", |a| { a.alu_ri(CMP, R13, -128, true); }, &[0x49, 0x83, 0xfd, 0x80]),
        ("cmp r13, -129", |a| { a.alu_ri(CMP, R13, -129, true); }, &[0x49, 0x81, 0xfd, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r13, 4096", |a| { a.alu_ri(CMP, R13, 4096, true); }, &[0x49, 0x81, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r13, 2147483647", |a| { a.alu_ri(CMP, R13, 2147483647, true); }, &[0x49, 0x81, 0xfd, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r13, -2147483648", |a| { a.alu_ri(CMP, R13, -2147483648, true); }, &[0x49, 0x81, 0xfd, 0x00, 0x00, 0x00, 0x80]),
        ("add r15, 5", |a| { a.alu_ri(ADD, R15, 5, true); }, &[0x49, 0x83, 0xc7, 0x05]),
        ("add r15, -1", |a| { a.alu_ri(ADD, R15, -1, true); }, &[0x49, 0x83, 0xc7, 0xff]),
        ("add r15, 127", |a| { a.alu_ri(ADD, R15, 127, true); }, &[0x49, 0x83, 0xc7, 0x7f]),
        ("add r15, 128", |a| { a.alu_ri(ADD, R15, 128, true); }, &[0x49, 0x81, 0xc7, 0x80, 0x00, 0x00, 0x00]),
        ("add r15, -128", |a| { a.alu_ri(ADD, R15, -128, true); }, &[0x49, 0x83, 0xc7, 0x80]),
        ("add r15, -129", |a| { a.alu_ri(ADD, R15, -129, true); }, &[0x49, 0x81, 0xc7, 0x7f, 0xff, 0xff, 0xff]),
        ("add r15, 4096", |a| { a.alu_ri(ADD, R15, 4096, true); }, &[0x49, 0x81, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("add r15, 2147483647", |a| { a.alu_ri(ADD, R15, 2147483647, true); }, &[0x49, 0x81, 0xc7, 0xff, 0xff, 0xff, 0x7f]),
        ("add r15, -2147483648", |a| { a.alu_ri(ADD, R15, -2147483648, true); }, &[0x49, 0x81, 0xc7, 0x00, 0x00, 0x00, 0x80]),
        ("or r15, 5", |a| { a.alu_ri(OR, R15, 5, true); }, &[0x49, 0x83, 0xcf, 0x05]),
        ("or r15, -1", |a| { a.alu_ri(OR, R15, -1, true); }, &[0x49, 0x83, 0xcf, 0xff]),
        ("or r15, 127", |a| { a.alu_ri(OR, R15, 127, true); }, &[0x49, 0x83, 0xcf, 0x7f]),
        ("or r15, 128", |a| { a.alu_ri(OR, R15, 128, true); }, &[0x49, 0x81, 0xcf, 0x80, 0x00, 0x00, 0x00]),
        ("or r15, -128", |a| { a.alu_ri(OR, R15, -128, true); }, &[0x49, 0x83, 0xcf, 0x80]),
        ("or r15, -129", |a| { a.alu_ri(OR, R15, -129, true); }, &[0x49, 0x81, 0xcf, 0x7f, 0xff, 0xff, 0xff]),
        ("or r15, 4096", |a| { a.alu_ri(OR, R15, 4096, true); }, &[0x49, 0x81, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("or r15, 2147483647", |a| { a.alu_ri(OR, R15, 2147483647, true); }, &[0x49, 0x81, 0xcf, 0xff, 0xff, 0xff, 0x7f]),
        ("or r15, -2147483648", |a| { a.alu_ri(OR, R15, -2147483648, true); }, &[0x49, 0x81, 0xcf, 0x00, 0x00, 0x00, 0x80]),
        ("and r15, 5", |a| { a.alu_ri(AND, R15, 5, true); }, &[0x49, 0x83, 0xe7, 0x05]),
        ("and r15, -1", |a| { a.alu_ri(AND, R15, -1, true); }, &[0x49, 0x83, 0xe7, 0xff]),
        ("and r15, 127", |a| { a.alu_ri(AND, R15, 127, true); }, &[0x49, 0x83, 0xe7, 0x7f]),
        ("and r15, 128", |a| { a.alu_ri(AND, R15, 128, true); }, &[0x49, 0x81, 0xe7, 0x80, 0x00, 0x00, 0x00]),
        ("and r15, -128", |a| { a.alu_ri(AND, R15, -128, true); }, &[0x49, 0x83, 0xe7, 0x80]),
        ("and r15, -129", |a| { a.alu_ri(AND, R15, -129, true); }, &[0x49, 0x81, 0xe7, 0x7f, 0xff, 0xff, 0xff]),
        ("and r15, 4096", |a| { a.alu_ri(AND, R15, 4096, true); }, &[0x49, 0x81, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("and r15, 2147483647", |a| { a.alu_ri(AND, R15, 2147483647, true); }, &[0x49, 0x81, 0xe7, 0xff, 0xff, 0xff, 0x7f]),
        ("and r15, -2147483648", |a| { a.alu_ri(AND, R15, -2147483648, true); }, &[0x49, 0x81, 0xe7, 0x00, 0x00, 0x00, 0x80]),
        ("sub r15, 5", |a| { a.alu_ri(SUB, R15, 5, true); }, &[0x49, 0x83, 0xef, 0x05]),
        ("sub r15, -1", |a| { a.alu_ri(SUB, R15, -1, true); }, &[0x49, 0x83, 0xef, 0xff]),
        ("sub r15, 127", |a| { a.alu_ri(SUB, R15, 127, true); }, &[0x49, 0x83, 0xef, 0x7f]),
        ("sub r15, 128", |a| { a.alu_ri(SUB, R15, 128, true); }, &[0x49, 0x81, 0xef, 0x80, 0x00, 0x00, 0x00]),
        ("sub r15, -128", |a| { a.alu_ri(SUB, R15, -128, true); }, &[0x49, 0x83, 0xef, 0x80]),
        ("sub r15, -129", |a| { a.alu_ri(SUB, R15, -129, true); }, &[0x49, 0x81, 0xef, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r15, 4096", |a| { a.alu_ri(SUB, R15, 4096, true); }, &[0x49, 0x81, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("sub r15, 2147483647", |a| { a.alu_ri(SUB, R15, 2147483647, true); }, &[0x49, 0x81, 0xef, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r15, -2147483648", |a| { a.alu_ri(SUB, R15, -2147483648, true); }, &[0x49, 0x81, 0xef, 0x00, 0x00, 0x00, 0x80]),
        ("xor r15, 5", |a| { a.alu_ri(XOR, R15, 5, true); }, &[0x49, 0x83, 0xf7, 0x05]),
        ("xor r15, -1", |a| { a.alu_ri(XOR, R15, -1, true); }, &[0x49, 0x83, 0xf7, 0xff]),
        ("xor r15, 127", |a| { a.alu_ri(XOR, R15, 127, true); }, &[0x49, 0x83, 0xf7, 0x7f]),
        ("xor r15, 128", |a| { a.alu_ri(XOR, R15, 128, true); }, &[0x49, 0x81, 0xf7, 0x80, 0x00, 0x00, 0x00]),
        ("xor r15, -128", |a| { a.alu_ri(XOR, R15, -128, true); }, &[0x49, 0x83, 0xf7, 0x80]),
        ("xor r15, -129", |a| { a.alu_ri(XOR, R15, -129, true); }, &[0x49, 0x81, 0xf7, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r15, 4096", |a| { a.alu_ri(XOR, R15, 4096, true); }, &[0x49, 0x81, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("xor r15, 2147483647", |a| { a.alu_ri(XOR, R15, 2147483647, true); }, &[0x49, 0x81, 0xf7, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r15, -2147483648", |a| { a.alu_ri(XOR, R15, -2147483648, true); }, &[0x49, 0x81, 0xf7, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r15, 5", |a| { a.alu_ri(CMP, R15, 5, true); }, &[0x49, 0x83, 0xff, 0x05]),
        ("cmp r15, -1", |a| { a.alu_ri(CMP, R15, -1, true); }, &[0x49, 0x83, 0xff, 0xff]),
        ("cmp r15, 127", |a| { a.alu_ri(CMP, R15, 127, true); }, &[0x49, 0x83, 0xff, 0x7f]),
        ("cmp r15, 128", |a| { a.alu_ri(CMP, R15, 128, true); }, &[0x49, 0x81, 0xff, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r15, -128", |a| { a.alu_ri(CMP, R15, -128, true); }, &[0x49, 0x83, 0xff, 0x80]),
        ("cmp r15, -129", |a| { a.alu_ri(CMP, R15, -129, true); }, &[0x49, 0x81, 0xff, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r15, 4096", |a| { a.alu_ri(CMP, R15, 4096, true); }, &[0x49, 0x81, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r15, 2147483647", |a| { a.alu_ri(CMP, R15, 2147483647, true); }, &[0x49, 0x81, 0xff, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r15, -2147483648", |a| { a.alu_ri(CMP, R15, -2147483648, true); }, &[0x49, 0x81, 0xff, 0x00, 0x00, 0x00, 0x80]),
        ("add ecx, 5", |a| { a.alu_ri(ADD, RCX, 5, false); }, &[0x83, 0xc1, 0x05]),
        ("add ecx, -1", |a| { a.alu_ri(ADD, RCX, -1, false); }, &[0x83, 0xc1, 0xff]),
        ("add ecx, 127", |a| { a.alu_ri(ADD, RCX, 127, false); }, &[0x83, 0xc1, 0x7f]),
        ("add ecx, 128", |a| { a.alu_ri(ADD, RCX, 128, false); }, &[0x81, 0xc1, 0x80, 0x00, 0x00, 0x00]),
        ("add ecx, -128", |a| { a.alu_ri(ADD, RCX, -128, false); }, &[0x83, 0xc1, 0x80]),
        ("add ecx, -129", |a| { a.alu_ri(ADD, RCX, -129, false); }, &[0x81, 0xc1, 0x7f, 0xff, 0xff, 0xff]),
        ("add ecx, 4096", |a| { a.alu_ri(ADD, RCX, 4096, false); }, &[0x81, 0xc1, 0x00, 0x10, 0x00, 0x00]),
        ("add ecx, 2147483647", |a| { a.alu_ri(ADD, RCX, 2147483647, false); }, &[0x81, 0xc1, 0xff, 0xff, 0xff, 0x7f]),
        ("add ecx, -2147483648", |a| { a.alu_ri(ADD, RCX, -2147483648, false); }, &[0x81, 0xc1, 0x00, 0x00, 0x00, 0x80]),
        ("or ecx, 5", |a| { a.alu_ri(OR, RCX, 5, false); }, &[0x83, 0xc9, 0x05]),
        ("or ecx, -1", |a| { a.alu_ri(OR, RCX, -1, false); }, &[0x83, 0xc9, 0xff]),
        ("or ecx, 127", |a| { a.alu_ri(OR, RCX, 127, false); }, &[0x83, 0xc9, 0x7f]),
        ("or ecx, 128", |a| { a.alu_ri(OR, RCX, 128, false); }, &[0x81, 0xc9, 0x80, 0x00, 0x00, 0x00]),
        ("or ecx, -128", |a| { a.alu_ri(OR, RCX, -128, false); }, &[0x83, 0xc9, 0x80]),
        ("or ecx, -129", |a| { a.alu_ri(OR, RCX, -129, false); }, &[0x81, 0xc9, 0x7f, 0xff, 0xff, 0xff]),
        ("or ecx, 4096", |a| { a.alu_ri(OR, RCX, 4096, false); }, &[0x81, 0xc9, 0x00, 0x10, 0x00, 0x00]),
        ("or ecx, 2147483647", |a| { a.alu_ri(OR, RCX, 2147483647, false); }, &[0x81, 0xc9, 0xff, 0xff, 0xff, 0x7f]),
        ("or ecx, -2147483648", |a| { a.alu_ri(OR, RCX, -2147483648, false); }, &[0x81, 0xc9, 0x00, 0x00, 0x00, 0x80]),
        ("and ecx, 5", |a| { a.alu_ri(AND, RCX, 5, false); }, &[0x83, 0xe1, 0x05]),
        ("and ecx, -1", |a| { a.alu_ri(AND, RCX, -1, false); }, &[0x83, 0xe1, 0xff]),
        ("and ecx, 127", |a| { a.alu_ri(AND, RCX, 127, false); }, &[0x83, 0xe1, 0x7f]),
        ("and ecx, 128", |a| { a.alu_ri(AND, RCX, 128, false); }, &[0x81, 0xe1, 0x80, 0x00, 0x00, 0x00]),
        ("and ecx, -128", |a| { a.alu_ri(AND, RCX, -128, false); }, &[0x83, 0xe1, 0x80]),
        ("and ecx, -129", |a| { a.alu_ri(AND, RCX, -129, false); }, &[0x81, 0xe1, 0x7f, 0xff, 0xff, 0xff]),
        ("and ecx, 4096", |a| { a.alu_ri(AND, RCX, 4096, false); }, &[0x81, 0xe1, 0x00, 0x10, 0x00, 0x00]),
        ("and ecx, 2147483647", |a| { a.alu_ri(AND, RCX, 2147483647, false); }, &[0x81, 0xe1, 0xff, 0xff, 0xff, 0x7f]),
        ("and ecx, -2147483648", |a| { a.alu_ri(AND, RCX, -2147483648, false); }, &[0x81, 0xe1, 0x00, 0x00, 0x00, 0x80]),
        ("sub ecx, 5", |a| { a.alu_ri(SUB, RCX, 5, false); }, &[0x83, 0xe9, 0x05]),
        ("sub ecx, -1", |a| { a.alu_ri(SUB, RCX, -1, false); }, &[0x83, 0xe9, 0xff]),
        ("sub ecx, 127", |a| { a.alu_ri(SUB, RCX, 127, false); }, &[0x83, 0xe9, 0x7f]),
        ("sub ecx, 128", |a| { a.alu_ri(SUB, RCX, 128, false); }, &[0x81, 0xe9, 0x80, 0x00, 0x00, 0x00]),
        ("sub ecx, -128", |a| { a.alu_ri(SUB, RCX, -128, false); }, &[0x83, 0xe9, 0x80]),
        ("sub ecx, -129", |a| { a.alu_ri(SUB, RCX, -129, false); }, &[0x81, 0xe9, 0x7f, 0xff, 0xff, 0xff]),
        ("sub ecx, 4096", |a| { a.alu_ri(SUB, RCX, 4096, false); }, &[0x81, 0xe9, 0x00, 0x10, 0x00, 0x00]),
        ("sub ecx, 2147483647", |a| { a.alu_ri(SUB, RCX, 2147483647, false); }, &[0x81, 0xe9, 0xff, 0xff, 0xff, 0x7f]),
        ("sub ecx, -2147483648", |a| { a.alu_ri(SUB, RCX, -2147483648, false); }, &[0x81, 0xe9, 0x00, 0x00, 0x00, 0x80]),
        ("xor ecx, 5", |a| { a.alu_ri(XOR, RCX, 5, false); }, &[0x83, 0xf1, 0x05]),
        ("xor ecx, -1", |a| { a.alu_ri(XOR, RCX, -1, false); }, &[0x83, 0xf1, 0xff]),
        ("xor ecx, 127", |a| { a.alu_ri(XOR, RCX, 127, false); }, &[0x83, 0xf1, 0x7f]),
        ("xor ecx, 128", |a| { a.alu_ri(XOR, RCX, 128, false); }, &[0x81, 0xf1, 0x80, 0x00, 0x00, 0x00]),
        ("xor ecx, -128", |a| { a.alu_ri(XOR, RCX, -128, false); }, &[0x83, 0xf1, 0x80]),
        ("xor ecx, -129", |a| { a.alu_ri(XOR, RCX, -129, false); }, &[0x81, 0xf1, 0x7f, 0xff, 0xff, 0xff]),
        ("xor ecx, 4096", |a| { a.alu_ri(XOR, RCX, 4096, false); }, &[0x81, 0xf1, 0x00, 0x10, 0x00, 0x00]),
        ("xor ecx, 2147483647", |a| { a.alu_ri(XOR, RCX, 2147483647, false); }, &[0x81, 0xf1, 0xff, 0xff, 0xff, 0x7f]),
        ("xor ecx, -2147483648", |a| { a.alu_ri(XOR, RCX, -2147483648, false); }, &[0x81, 0xf1, 0x00, 0x00, 0x00, 0x80]),
        ("cmp ecx, 5", |a| { a.alu_ri(CMP, RCX, 5, false); }, &[0x83, 0xf9, 0x05]),
        ("cmp ecx, -1", |a| { a.alu_ri(CMP, RCX, -1, false); }, &[0x83, 0xf9, 0xff]),
        ("cmp ecx, 127", |a| { a.alu_ri(CMP, RCX, 127, false); }, &[0x83, 0xf9, 0x7f]),
        ("cmp ecx, 128", |a| { a.alu_ri(CMP, RCX, 128, false); }, &[0x81, 0xf9, 0x80, 0x00, 0x00, 0x00]),
        ("cmp ecx, -128", |a| { a.alu_ri(CMP, RCX, -128, false); }, &[0x83, 0xf9, 0x80]),
        ("cmp ecx, -129", |a| { a.alu_ri(CMP, RCX, -129, false); }, &[0x81, 0xf9, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp ecx, 4096", |a| { a.alu_ri(CMP, RCX, 4096, false); }, &[0x81, 0xf9, 0x00, 0x10, 0x00, 0x00]),
        ("cmp ecx, 2147483647", |a| { a.alu_ri(CMP, RCX, 2147483647, false); }, &[0x81, 0xf9, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp ecx, -2147483648", |a| { a.alu_ri(CMP, RCX, -2147483648, false); }, &[0x81, 0xf9, 0x00, 0x00, 0x00, 0x80]),
        ("add edx, 5", |a| { a.alu_ri(ADD, RDX, 5, false); }, &[0x83, 0xc2, 0x05]),
        ("add edx, -1", |a| { a.alu_ri(ADD, RDX, -1, false); }, &[0x83, 0xc2, 0xff]),
        ("add edx, 127", |a| { a.alu_ri(ADD, RDX, 127, false); }, &[0x83, 0xc2, 0x7f]),
        ("add edx, 128", |a| { a.alu_ri(ADD, RDX, 128, false); }, &[0x81, 0xc2, 0x80, 0x00, 0x00, 0x00]),
        ("add edx, -128", |a| { a.alu_ri(ADD, RDX, -128, false); }, &[0x83, 0xc2, 0x80]),
        ("add edx, -129", |a| { a.alu_ri(ADD, RDX, -129, false); }, &[0x81, 0xc2, 0x7f, 0xff, 0xff, 0xff]),
        ("add edx, 4096", |a| { a.alu_ri(ADD, RDX, 4096, false); }, &[0x81, 0xc2, 0x00, 0x10, 0x00, 0x00]),
        ("add edx, 2147483647", |a| { a.alu_ri(ADD, RDX, 2147483647, false); }, &[0x81, 0xc2, 0xff, 0xff, 0xff, 0x7f]),
        ("add edx, -2147483648", |a| { a.alu_ri(ADD, RDX, -2147483648, false); }, &[0x81, 0xc2, 0x00, 0x00, 0x00, 0x80]),
        ("or edx, 5", |a| { a.alu_ri(OR, RDX, 5, false); }, &[0x83, 0xca, 0x05]),
        ("or edx, -1", |a| { a.alu_ri(OR, RDX, -1, false); }, &[0x83, 0xca, 0xff]),
        ("or edx, 127", |a| { a.alu_ri(OR, RDX, 127, false); }, &[0x83, 0xca, 0x7f]),
        ("or edx, 128", |a| { a.alu_ri(OR, RDX, 128, false); }, &[0x81, 0xca, 0x80, 0x00, 0x00, 0x00]),
        ("or edx, -128", |a| { a.alu_ri(OR, RDX, -128, false); }, &[0x83, 0xca, 0x80]),
        ("or edx, -129", |a| { a.alu_ri(OR, RDX, -129, false); }, &[0x81, 0xca, 0x7f, 0xff, 0xff, 0xff]),
        ("or edx, 4096", |a| { a.alu_ri(OR, RDX, 4096, false); }, &[0x81, 0xca, 0x00, 0x10, 0x00, 0x00]),
        ("or edx, 2147483647", |a| { a.alu_ri(OR, RDX, 2147483647, false); }, &[0x81, 0xca, 0xff, 0xff, 0xff, 0x7f]),
        ("or edx, -2147483648", |a| { a.alu_ri(OR, RDX, -2147483648, false); }, &[0x81, 0xca, 0x00, 0x00, 0x00, 0x80]),
        ("and edx, 5", |a| { a.alu_ri(AND, RDX, 5, false); }, &[0x83, 0xe2, 0x05]),
        ("and edx, -1", |a| { a.alu_ri(AND, RDX, -1, false); }, &[0x83, 0xe2, 0xff]),
        ("and edx, 127", |a| { a.alu_ri(AND, RDX, 127, false); }, &[0x83, 0xe2, 0x7f]),
        ("and edx, 128", |a| { a.alu_ri(AND, RDX, 128, false); }, &[0x81, 0xe2, 0x80, 0x00, 0x00, 0x00]),
        ("and edx, -128", |a| { a.alu_ri(AND, RDX, -128, false); }, &[0x83, 0xe2, 0x80]),
        ("and edx, -129", |a| { a.alu_ri(AND, RDX, -129, false); }, &[0x81, 0xe2, 0x7f, 0xff, 0xff, 0xff]),
        ("and edx, 4096", |a| { a.alu_ri(AND, RDX, 4096, false); }, &[0x81, 0xe2, 0x00, 0x10, 0x00, 0x00]),
        ("and edx, 2147483647", |a| { a.alu_ri(AND, RDX, 2147483647, false); }, &[0x81, 0xe2, 0xff, 0xff, 0xff, 0x7f]),
        ("and edx, -2147483648", |a| { a.alu_ri(AND, RDX, -2147483648, false); }, &[0x81, 0xe2, 0x00, 0x00, 0x00, 0x80]),
        ("sub edx, 5", |a| { a.alu_ri(SUB, RDX, 5, false); }, &[0x83, 0xea, 0x05]),
        ("sub edx, -1", |a| { a.alu_ri(SUB, RDX, -1, false); }, &[0x83, 0xea, 0xff]),
        ("sub edx, 127", |a| { a.alu_ri(SUB, RDX, 127, false); }, &[0x83, 0xea, 0x7f]),
        ("sub edx, 128", |a| { a.alu_ri(SUB, RDX, 128, false); }, &[0x81, 0xea, 0x80, 0x00, 0x00, 0x00]),
        ("sub edx, -128", |a| { a.alu_ri(SUB, RDX, -128, false); }, &[0x83, 0xea, 0x80]),
        ("sub edx, -129", |a| { a.alu_ri(SUB, RDX, -129, false); }, &[0x81, 0xea, 0x7f, 0xff, 0xff, 0xff]),
        ("sub edx, 4096", |a| { a.alu_ri(SUB, RDX, 4096, false); }, &[0x81, 0xea, 0x00, 0x10, 0x00, 0x00]),
        ("sub edx, 2147483647", |a| { a.alu_ri(SUB, RDX, 2147483647, false); }, &[0x81, 0xea, 0xff, 0xff, 0xff, 0x7f]),
        ("sub edx, -2147483648", |a| { a.alu_ri(SUB, RDX, -2147483648, false); }, &[0x81, 0xea, 0x00, 0x00, 0x00, 0x80]),
        ("xor edx, 5", |a| { a.alu_ri(XOR, RDX, 5, false); }, &[0x83, 0xf2, 0x05]),
        ("xor edx, -1", |a| { a.alu_ri(XOR, RDX, -1, false); }, &[0x83, 0xf2, 0xff]),
        ("xor edx, 127", |a| { a.alu_ri(XOR, RDX, 127, false); }, &[0x83, 0xf2, 0x7f]),
        ("xor edx, 128", |a| { a.alu_ri(XOR, RDX, 128, false); }, &[0x81, 0xf2, 0x80, 0x00, 0x00, 0x00]),
        ("xor edx, -128", |a| { a.alu_ri(XOR, RDX, -128, false); }, &[0x83, 0xf2, 0x80]),
        ("xor edx, -129", |a| { a.alu_ri(XOR, RDX, -129, false); }, &[0x81, 0xf2, 0x7f, 0xff, 0xff, 0xff]),
        ("xor edx, 4096", |a| { a.alu_ri(XOR, RDX, 4096, false); }, &[0x81, 0xf2, 0x00, 0x10, 0x00, 0x00]),
        ("xor edx, 2147483647", |a| { a.alu_ri(XOR, RDX, 2147483647, false); }, &[0x81, 0xf2, 0xff, 0xff, 0xff, 0x7f]),
        ("xor edx, -2147483648", |a| { a.alu_ri(XOR, RDX, -2147483648, false); }, &[0x81, 0xf2, 0x00, 0x00, 0x00, 0x80]),
        ("cmp edx, 5", |a| { a.alu_ri(CMP, RDX, 5, false); }, &[0x83, 0xfa, 0x05]),
        ("cmp edx, -1", |a| { a.alu_ri(CMP, RDX, -1, false); }, &[0x83, 0xfa, 0xff]),
        ("cmp edx, 127", |a| { a.alu_ri(CMP, RDX, 127, false); }, &[0x83, 0xfa, 0x7f]),
        ("cmp edx, 128", |a| { a.alu_ri(CMP, RDX, 128, false); }, &[0x81, 0xfa, 0x80, 0x00, 0x00, 0x00]),
        ("cmp edx, -128", |a| { a.alu_ri(CMP, RDX, -128, false); }, &[0x83, 0xfa, 0x80]),
        ("cmp edx, -129", |a| { a.alu_ri(CMP, RDX, -129, false); }, &[0x81, 0xfa, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp edx, 4096", |a| { a.alu_ri(CMP, RDX, 4096, false); }, &[0x81, 0xfa, 0x00, 0x10, 0x00, 0x00]),
        ("cmp edx, 2147483647", |a| { a.alu_ri(CMP, RDX, 2147483647, false); }, &[0x81, 0xfa, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp edx, -2147483648", |a| { a.alu_ri(CMP, RDX, -2147483648, false); }, &[0x81, 0xfa, 0x00, 0x00, 0x00, 0x80]),
        ("add esp, 5", |a| { a.alu_ri(ADD, RSP, 5, false); }, &[0x83, 0xc4, 0x05]),
        ("add esp, -1", |a| { a.alu_ri(ADD, RSP, -1, false); }, &[0x83, 0xc4, 0xff]),
        ("add esp, 127", |a| { a.alu_ri(ADD, RSP, 127, false); }, &[0x83, 0xc4, 0x7f]),
        ("add esp, 128", |a| { a.alu_ri(ADD, RSP, 128, false); }, &[0x81, 0xc4, 0x80, 0x00, 0x00, 0x00]),
        ("add esp, -128", |a| { a.alu_ri(ADD, RSP, -128, false); }, &[0x83, 0xc4, 0x80]),
        ("add esp, -129", |a| { a.alu_ri(ADD, RSP, -129, false); }, &[0x81, 0xc4, 0x7f, 0xff, 0xff, 0xff]),
        ("add esp, 4096", |a| { a.alu_ri(ADD, RSP, 4096, false); }, &[0x81, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("add esp, 2147483647", |a| { a.alu_ri(ADD, RSP, 2147483647, false); }, &[0x81, 0xc4, 0xff, 0xff, 0xff, 0x7f]),
        ("add esp, -2147483648", |a| { a.alu_ri(ADD, RSP, -2147483648, false); }, &[0x81, 0xc4, 0x00, 0x00, 0x00, 0x80]),
        ("or esp, 5", |a| { a.alu_ri(OR, RSP, 5, false); }, &[0x83, 0xcc, 0x05]),
        ("or esp, -1", |a| { a.alu_ri(OR, RSP, -1, false); }, &[0x83, 0xcc, 0xff]),
        ("or esp, 127", |a| { a.alu_ri(OR, RSP, 127, false); }, &[0x83, 0xcc, 0x7f]),
        ("or esp, 128", |a| { a.alu_ri(OR, RSP, 128, false); }, &[0x81, 0xcc, 0x80, 0x00, 0x00, 0x00]),
        ("or esp, -128", |a| { a.alu_ri(OR, RSP, -128, false); }, &[0x83, 0xcc, 0x80]),
        ("or esp, -129", |a| { a.alu_ri(OR, RSP, -129, false); }, &[0x81, 0xcc, 0x7f, 0xff, 0xff, 0xff]),
        ("or esp, 4096", |a| { a.alu_ri(OR, RSP, 4096, false); }, &[0x81, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("or esp, 2147483647", |a| { a.alu_ri(OR, RSP, 2147483647, false); }, &[0x81, 0xcc, 0xff, 0xff, 0xff, 0x7f]),
        ("or esp, -2147483648", |a| { a.alu_ri(OR, RSP, -2147483648, false); }, &[0x81, 0xcc, 0x00, 0x00, 0x00, 0x80]),
        ("and esp, 5", |a| { a.alu_ri(AND, RSP, 5, false); }, &[0x83, 0xe4, 0x05]),
        ("and esp, -1", |a| { a.alu_ri(AND, RSP, -1, false); }, &[0x83, 0xe4, 0xff]),
        ("and esp, 127", |a| { a.alu_ri(AND, RSP, 127, false); }, &[0x83, 0xe4, 0x7f]),
        ("and esp, 128", |a| { a.alu_ri(AND, RSP, 128, false); }, &[0x81, 0xe4, 0x80, 0x00, 0x00, 0x00]),
        ("and esp, -128", |a| { a.alu_ri(AND, RSP, -128, false); }, &[0x83, 0xe4, 0x80]),
        ("and esp, -129", |a| { a.alu_ri(AND, RSP, -129, false); }, &[0x81, 0xe4, 0x7f, 0xff, 0xff, 0xff]),
        ("and esp, 4096", |a| { a.alu_ri(AND, RSP, 4096, false); }, &[0x81, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("and esp, 2147483647", |a| { a.alu_ri(AND, RSP, 2147483647, false); }, &[0x81, 0xe4, 0xff, 0xff, 0xff, 0x7f]),
        ("and esp, -2147483648", |a| { a.alu_ri(AND, RSP, -2147483648, false); }, &[0x81, 0xe4, 0x00, 0x00, 0x00, 0x80]),
        ("sub esp, 5", |a| { a.alu_ri(SUB, RSP, 5, false); }, &[0x83, 0xec, 0x05]),
        ("sub esp, -1", |a| { a.alu_ri(SUB, RSP, -1, false); }, &[0x83, 0xec, 0xff]),
        ("sub esp, 127", |a| { a.alu_ri(SUB, RSP, 127, false); }, &[0x83, 0xec, 0x7f]),
        ("sub esp, 128", |a| { a.alu_ri(SUB, RSP, 128, false); }, &[0x81, 0xec, 0x80, 0x00, 0x00, 0x00]),
        ("sub esp, -128", |a| { a.alu_ri(SUB, RSP, -128, false); }, &[0x83, 0xec, 0x80]),
        ("sub esp, -129", |a| { a.alu_ri(SUB, RSP, -129, false); }, &[0x81, 0xec, 0x7f, 0xff, 0xff, 0xff]),
        ("sub esp, 4096", |a| { a.alu_ri(SUB, RSP, 4096, false); }, &[0x81, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("sub esp, 2147483647", |a| { a.alu_ri(SUB, RSP, 2147483647, false); }, &[0x81, 0xec, 0xff, 0xff, 0xff, 0x7f]),
        ("sub esp, -2147483648", |a| { a.alu_ri(SUB, RSP, -2147483648, false); }, &[0x81, 0xec, 0x00, 0x00, 0x00, 0x80]),
        ("xor esp, 5", |a| { a.alu_ri(XOR, RSP, 5, false); }, &[0x83, 0xf4, 0x05]),
        ("xor esp, -1", |a| { a.alu_ri(XOR, RSP, -1, false); }, &[0x83, 0xf4, 0xff]),
        ("xor esp, 127", |a| { a.alu_ri(XOR, RSP, 127, false); }, &[0x83, 0xf4, 0x7f]),
        ("xor esp, 128", |a| { a.alu_ri(XOR, RSP, 128, false); }, &[0x81, 0xf4, 0x80, 0x00, 0x00, 0x00]),
        ("xor esp, -128", |a| { a.alu_ri(XOR, RSP, -128, false); }, &[0x83, 0xf4, 0x80]),
        ("xor esp, -129", |a| { a.alu_ri(XOR, RSP, -129, false); }, &[0x81, 0xf4, 0x7f, 0xff, 0xff, 0xff]),
        ("xor esp, 4096", |a| { a.alu_ri(XOR, RSP, 4096, false); }, &[0x81, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("xor esp, 2147483647", |a| { a.alu_ri(XOR, RSP, 2147483647, false); }, &[0x81, 0xf4, 0xff, 0xff, 0xff, 0x7f]),
        ("xor esp, -2147483648", |a| { a.alu_ri(XOR, RSP, -2147483648, false); }, &[0x81, 0xf4, 0x00, 0x00, 0x00, 0x80]),
        ("cmp esp, 5", |a| { a.alu_ri(CMP, RSP, 5, false); }, &[0x83, 0xfc, 0x05]),
        ("cmp esp, -1", |a| { a.alu_ri(CMP, RSP, -1, false); }, &[0x83, 0xfc, 0xff]),
        ("cmp esp, 127", |a| { a.alu_ri(CMP, RSP, 127, false); }, &[0x83, 0xfc, 0x7f]),
        ("cmp esp, 128", |a| { a.alu_ri(CMP, RSP, 128, false); }, &[0x81, 0xfc, 0x80, 0x00, 0x00, 0x00]),
        ("cmp esp, -128", |a| { a.alu_ri(CMP, RSP, -128, false); }, &[0x83, 0xfc, 0x80]),
        ("cmp esp, -129", |a| { a.alu_ri(CMP, RSP, -129, false); }, &[0x81, 0xfc, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp esp, 4096", |a| { a.alu_ri(CMP, RSP, 4096, false); }, &[0x81, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("cmp esp, 2147483647", |a| { a.alu_ri(CMP, RSP, 2147483647, false); }, &[0x81, 0xfc, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp esp, -2147483648", |a| { a.alu_ri(CMP, RSP, -2147483648, false); }, &[0x81, 0xfc, 0x00, 0x00, 0x00, 0x80]),
        ("add ebp, 5", |a| { a.alu_ri(ADD, RBP, 5, false); }, &[0x83, 0xc5, 0x05]),
        ("add ebp, -1", |a| { a.alu_ri(ADD, RBP, -1, false); }, &[0x83, 0xc5, 0xff]),
        ("add ebp, 127", |a| { a.alu_ri(ADD, RBP, 127, false); }, &[0x83, 0xc5, 0x7f]),
        ("add ebp, 128", |a| { a.alu_ri(ADD, RBP, 128, false); }, &[0x81, 0xc5, 0x80, 0x00, 0x00, 0x00]),
        ("add ebp, -128", |a| { a.alu_ri(ADD, RBP, -128, false); }, &[0x83, 0xc5, 0x80]),
        ("add ebp, -129", |a| { a.alu_ri(ADD, RBP, -129, false); }, &[0x81, 0xc5, 0x7f, 0xff, 0xff, 0xff]),
        ("add ebp, 4096", |a| { a.alu_ri(ADD, RBP, 4096, false); }, &[0x81, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("add ebp, 2147483647", |a| { a.alu_ri(ADD, RBP, 2147483647, false); }, &[0x81, 0xc5, 0xff, 0xff, 0xff, 0x7f]),
        ("add ebp, -2147483648", |a| { a.alu_ri(ADD, RBP, -2147483648, false); }, &[0x81, 0xc5, 0x00, 0x00, 0x00, 0x80]),
        ("or ebp, 5", |a| { a.alu_ri(OR, RBP, 5, false); }, &[0x83, 0xcd, 0x05]),
        ("or ebp, -1", |a| { a.alu_ri(OR, RBP, -1, false); }, &[0x83, 0xcd, 0xff]),
        ("or ebp, 127", |a| { a.alu_ri(OR, RBP, 127, false); }, &[0x83, 0xcd, 0x7f]),
        ("or ebp, 128", |a| { a.alu_ri(OR, RBP, 128, false); }, &[0x81, 0xcd, 0x80, 0x00, 0x00, 0x00]),
        ("or ebp, -128", |a| { a.alu_ri(OR, RBP, -128, false); }, &[0x83, 0xcd, 0x80]),
        ("or ebp, -129", |a| { a.alu_ri(OR, RBP, -129, false); }, &[0x81, 0xcd, 0x7f, 0xff, 0xff, 0xff]),
        ("or ebp, 4096", |a| { a.alu_ri(OR, RBP, 4096, false); }, &[0x81, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("or ebp, 2147483647", |a| { a.alu_ri(OR, RBP, 2147483647, false); }, &[0x81, 0xcd, 0xff, 0xff, 0xff, 0x7f]),
        ("or ebp, -2147483648", |a| { a.alu_ri(OR, RBP, -2147483648, false); }, &[0x81, 0xcd, 0x00, 0x00, 0x00, 0x80]),
        ("and ebp, 5", |a| { a.alu_ri(AND, RBP, 5, false); }, &[0x83, 0xe5, 0x05]),
        ("and ebp, -1", |a| { a.alu_ri(AND, RBP, -1, false); }, &[0x83, 0xe5, 0xff]),
        ("and ebp, 127", |a| { a.alu_ri(AND, RBP, 127, false); }, &[0x83, 0xe5, 0x7f]),
        ("and ebp, 128", |a| { a.alu_ri(AND, RBP, 128, false); }, &[0x81, 0xe5, 0x80, 0x00, 0x00, 0x00]),
        ("and ebp, -128", |a| { a.alu_ri(AND, RBP, -128, false); }, &[0x83, 0xe5, 0x80]),
        ("and ebp, -129", |a| { a.alu_ri(AND, RBP, -129, false); }, &[0x81, 0xe5, 0x7f, 0xff, 0xff, 0xff]),
        ("and ebp, 4096", |a| { a.alu_ri(AND, RBP, 4096, false); }, &[0x81, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("and ebp, 2147483647", |a| { a.alu_ri(AND, RBP, 2147483647, false); }, &[0x81, 0xe5, 0xff, 0xff, 0xff, 0x7f]),
        ("and ebp, -2147483648", |a| { a.alu_ri(AND, RBP, -2147483648, false); }, &[0x81, 0xe5, 0x00, 0x00, 0x00, 0x80]),
        ("sub ebp, 5", |a| { a.alu_ri(SUB, RBP, 5, false); }, &[0x83, 0xed, 0x05]),
        ("sub ebp, -1", |a| { a.alu_ri(SUB, RBP, -1, false); }, &[0x83, 0xed, 0xff]),
        ("sub ebp, 127", |a| { a.alu_ri(SUB, RBP, 127, false); }, &[0x83, 0xed, 0x7f]),
        ("sub ebp, 128", |a| { a.alu_ri(SUB, RBP, 128, false); }, &[0x81, 0xed, 0x80, 0x00, 0x00, 0x00]),
        ("sub ebp, -128", |a| { a.alu_ri(SUB, RBP, -128, false); }, &[0x83, 0xed, 0x80]),
        ("sub ebp, -129", |a| { a.alu_ri(SUB, RBP, -129, false); }, &[0x81, 0xed, 0x7f, 0xff, 0xff, 0xff]),
        ("sub ebp, 4096", |a| { a.alu_ri(SUB, RBP, 4096, false); }, &[0x81, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("sub ebp, 2147483647", |a| { a.alu_ri(SUB, RBP, 2147483647, false); }, &[0x81, 0xed, 0xff, 0xff, 0xff, 0x7f]),
        ("sub ebp, -2147483648", |a| { a.alu_ri(SUB, RBP, -2147483648, false); }, &[0x81, 0xed, 0x00, 0x00, 0x00, 0x80]),
        ("xor ebp, 5", |a| { a.alu_ri(XOR, RBP, 5, false); }, &[0x83, 0xf5, 0x05]),
        ("xor ebp, -1", |a| { a.alu_ri(XOR, RBP, -1, false); }, &[0x83, 0xf5, 0xff]),
        ("xor ebp, 127", |a| { a.alu_ri(XOR, RBP, 127, false); }, &[0x83, 0xf5, 0x7f]),
        ("xor ebp, 128", |a| { a.alu_ri(XOR, RBP, 128, false); }, &[0x81, 0xf5, 0x80, 0x00, 0x00, 0x00]),
        ("xor ebp, -128", |a| { a.alu_ri(XOR, RBP, -128, false); }, &[0x83, 0xf5, 0x80]),
        ("xor ebp, -129", |a| { a.alu_ri(XOR, RBP, -129, false); }, &[0x81, 0xf5, 0x7f, 0xff, 0xff, 0xff]),
        ("xor ebp, 4096", |a| { a.alu_ri(XOR, RBP, 4096, false); }, &[0x81, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("xor ebp, 2147483647", |a| { a.alu_ri(XOR, RBP, 2147483647, false); }, &[0x81, 0xf5, 0xff, 0xff, 0xff, 0x7f]),
        ("xor ebp, -2147483648", |a| { a.alu_ri(XOR, RBP, -2147483648, false); }, &[0x81, 0xf5, 0x00, 0x00, 0x00, 0x80]),
        ("cmp ebp, 5", |a| { a.alu_ri(CMP, RBP, 5, false); }, &[0x83, 0xfd, 0x05]),
        ("cmp ebp, -1", |a| { a.alu_ri(CMP, RBP, -1, false); }, &[0x83, 0xfd, 0xff]),
        ("cmp ebp, 127", |a| { a.alu_ri(CMP, RBP, 127, false); }, &[0x83, 0xfd, 0x7f]),
        ("cmp ebp, 128", |a| { a.alu_ri(CMP, RBP, 128, false); }, &[0x81, 0xfd, 0x80, 0x00, 0x00, 0x00]),
        ("cmp ebp, -128", |a| { a.alu_ri(CMP, RBP, -128, false); }, &[0x83, 0xfd, 0x80]),
        ("cmp ebp, -129", |a| { a.alu_ri(CMP, RBP, -129, false); }, &[0x81, 0xfd, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp ebp, 4096", |a| { a.alu_ri(CMP, RBP, 4096, false); }, &[0x81, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("cmp ebp, 2147483647", |a| { a.alu_ri(CMP, RBP, 2147483647, false); }, &[0x81, 0xfd, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp ebp, -2147483648", |a| { a.alu_ri(CMP, RBP, -2147483648, false); }, &[0x81, 0xfd, 0x00, 0x00, 0x00, 0x80]),
        ("add r8d, 5", |a| { a.alu_ri(ADD, R8, 5, false); }, &[0x41, 0x83, 0xc0, 0x05]),
        ("add r8d, -1", |a| { a.alu_ri(ADD, R8, -1, false); }, &[0x41, 0x83, 0xc0, 0xff]),
        ("add r8d, 127", |a| { a.alu_ri(ADD, R8, 127, false); }, &[0x41, 0x83, 0xc0, 0x7f]),
        ("add r8d, 128", |a| { a.alu_ri(ADD, R8, 128, false); }, &[0x41, 0x81, 0xc0, 0x80, 0x00, 0x00, 0x00]),
        ("add r8d, -128", |a| { a.alu_ri(ADD, R8, -128, false); }, &[0x41, 0x83, 0xc0, 0x80]),
        ("add r8d, -129", |a| { a.alu_ri(ADD, R8, -129, false); }, &[0x41, 0x81, 0xc0, 0x7f, 0xff, 0xff, 0xff]),
        ("add r8d, 4096", |a| { a.alu_ri(ADD, R8, 4096, false); }, &[0x41, 0x81, 0xc0, 0x00, 0x10, 0x00, 0x00]),
        ("add r8d, 2147483647", |a| { a.alu_ri(ADD, R8, 2147483647, false); }, &[0x41, 0x81, 0xc0, 0xff, 0xff, 0xff, 0x7f]),
        ("add r8d, -2147483648", |a| { a.alu_ri(ADD, R8, -2147483648, false); }, &[0x41, 0x81, 0xc0, 0x00, 0x00, 0x00, 0x80]),
        ("or r8d, 5", |a| { a.alu_ri(OR, R8, 5, false); }, &[0x41, 0x83, 0xc8, 0x05]),
        ("or r8d, -1", |a| { a.alu_ri(OR, R8, -1, false); }, &[0x41, 0x83, 0xc8, 0xff]),
        ("or r8d, 127", |a| { a.alu_ri(OR, R8, 127, false); }, &[0x41, 0x83, 0xc8, 0x7f]),
        ("or r8d, 128", |a| { a.alu_ri(OR, R8, 128, false); }, &[0x41, 0x81, 0xc8, 0x80, 0x00, 0x00, 0x00]),
        ("or r8d, -128", |a| { a.alu_ri(OR, R8, -128, false); }, &[0x41, 0x83, 0xc8, 0x80]),
        ("or r8d, -129", |a| { a.alu_ri(OR, R8, -129, false); }, &[0x41, 0x81, 0xc8, 0x7f, 0xff, 0xff, 0xff]),
        ("or r8d, 4096", |a| { a.alu_ri(OR, R8, 4096, false); }, &[0x41, 0x81, 0xc8, 0x00, 0x10, 0x00, 0x00]),
        ("or r8d, 2147483647", |a| { a.alu_ri(OR, R8, 2147483647, false); }, &[0x41, 0x81, 0xc8, 0xff, 0xff, 0xff, 0x7f]),
        ("or r8d, -2147483648", |a| { a.alu_ri(OR, R8, -2147483648, false); }, &[0x41, 0x81, 0xc8, 0x00, 0x00, 0x00, 0x80]),
        ("and r8d, 5", |a| { a.alu_ri(AND, R8, 5, false); }, &[0x41, 0x83, 0xe0, 0x05]),
        ("and r8d, -1", |a| { a.alu_ri(AND, R8, -1, false); }, &[0x41, 0x83, 0xe0, 0xff]),
        ("and r8d, 127", |a| { a.alu_ri(AND, R8, 127, false); }, &[0x41, 0x83, 0xe0, 0x7f]),
        ("and r8d, 128", |a| { a.alu_ri(AND, R8, 128, false); }, &[0x41, 0x81, 0xe0, 0x80, 0x00, 0x00, 0x00]),
        ("and r8d, -128", |a| { a.alu_ri(AND, R8, -128, false); }, &[0x41, 0x83, 0xe0, 0x80]),
        ("and r8d, -129", |a| { a.alu_ri(AND, R8, -129, false); }, &[0x41, 0x81, 0xe0, 0x7f, 0xff, 0xff, 0xff]),
        ("and r8d, 4096", |a| { a.alu_ri(AND, R8, 4096, false); }, &[0x41, 0x81, 0xe0, 0x00, 0x10, 0x00, 0x00]),
        ("and r8d, 2147483647", |a| { a.alu_ri(AND, R8, 2147483647, false); }, &[0x41, 0x81, 0xe0, 0xff, 0xff, 0xff, 0x7f]),
        ("and r8d, -2147483648", |a| { a.alu_ri(AND, R8, -2147483648, false); }, &[0x41, 0x81, 0xe0, 0x00, 0x00, 0x00, 0x80]),
        ("sub r8d, 5", |a| { a.alu_ri(SUB, R8, 5, false); }, &[0x41, 0x83, 0xe8, 0x05]),
        ("sub r8d, -1", |a| { a.alu_ri(SUB, R8, -1, false); }, &[0x41, 0x83, 0xe8, 0xff]),
        ("sub r8d, 127", |a| { a.alu_ri(SUB, R8, 127, false); }, &[0x41, 0x83, 0xe8, 0x7f]),
        ("sub r8d, 128", |a| { a.alu_ri(SUB, R8, 128, false); }, &[0x41, 0x81, 0xe8, 0x80, 0x00, 0x00, 0x00]),
        ("sub r8d, -128", |a| { a.alu_ri(SUB, R8, -128, false); }, &[0x41, 0x83, 0xe8, 0x80]),
        ("sub r8d, -129", |a| { a.alu_ri(SUB, R8, -129, false); }, &[0x41, 0x81, 0xe8, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r8d, 4096", |a| { a.alu_ri(SUB, R8, 4096, false); }, &[0x41, 0x81, 0xe8, 0x00, 0x10, 0x00, 0x00]),
        ("sub r8d, 2147483647", |a| { a.alu_ri(SUB, R8, 2147483647, false); }, &[0x41, 0x81, 0xe8, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r8d, -2147483648", |a| { a.alu_ri(SUB, R8, -2147483648, false); }, &[0x41, 0x81, 0xe8, 0x00, 0x00, 0x00, 0x80]),
        ("xor r8d, 5", |a| { a.alu_ri(XOR, R8, 5, false); }, &[0x41, 0x83, 0xf0, 0x05]),
        ("xor r8d, -1", |a| { a.alu_ri(XOR, R8, -1, false); }, &[0x41, 0x83, 0xf0, 0xff]),
        ("xor r8d, 127", |a| { a.alu_ri(XOR, R8, 127, false); }, &[0x41, 0x83, 0xf0, 0x7f]),
        ("xor r8d, 128", |a| { a.alu_ri(XOR, R8, 128, false); }, &[0x41, 0x81, 0xf0, 0x80, 0x00, 0x00, 0x00]),
        ("xor r8d, -128", |a| { a.alu_ri(XOR, R8, -128, false); }, &[0x41, 0x83, 0xf0, 0x80]),
        ("xor r8d, -129", |a| { a.alu_ri(XOR, R8, -129, false); }, &[0x41, 0x81, 0xf0, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r8d, 4096", |a| { a.alu_ri(XOR, R8, 4096, false); }, &[0x41, 0x81, 0xf0, 0x00, 0x10, 0x00, 0x00]),
        ("xor r8d, 2147483647", |a| { a.alu_ri(XOR, R8, 2147483647, false); }, &[0x41, 0x81, 0xf0, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r8d, -2147483648", |a| { a.alu_ri(XOR, R8, -2147483648, false); }, &[0x41, 0x81, 0xf0, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r8d, 5", |a| { a.alu_ri(CMP, R8, 5, false); }, &[0x41, 0x83, 0xf8, 0x05]),
        ("cmp r8d, -1", |a| { a.alu_ri(CMP, R8, -1, false); }, &[0x41, 0x83, 0xf8, 0xff]),
        ("cmp r8d, 127", |a| { a.alu_ri(CMP, R8, 127, false); }, &[0x41, 0x83, 0xf8, 0x7f]),
        ("cmp r8d, 128", |a| { a.alu_ri(CMP, R8, 128, false); }, &[0x41, 0x81, 0xf8, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r8d, -128", |a| { a.alu_ri(CMP, R8, -128, false); }, &[0x41, 0x83, 0xf8, 0x80]),
        ("cmp r8d, -129", |a| { a.alu_ri(CMP, R8, -129, false); }, &[0x41, 0x81, 0xf8, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r8d, 4096", |a| { a.alu_ri(CMP, R8, 4096, false); }, &[0x41, 0x81, 0xf8, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r8d, 2147483647", |a| { a.alu_ri(CMP, R8, 2147483647, false); }, &[0x41, 0x81, 0xf8, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r8d, -2147483648", |a| { a.alu_ri(CMP, R8, -2147483648, false); }, &[0x41, 0x81, 0xf8, 0x00, 0x00, 0x00, 0x80]),
        ("add r12d, 5", |a| { a.alu_ri(ADD, R12, 5, false); }, &[0x41, 0x83, 0xc4, 0x05]),
        ("add r12d, -1", |a| { a.alu_ri(ADD, R12, -1, false); }, &[0x41, 0x83, 0xc4, 0xff]),
        ("add r12d, 127", |a| { a.alu_ri(ADD, R12, 127, false); }, &[0x41, 0x83, 0xc4, 0x7f]),
        ("add r12d, 128", |a| { a.alu_ri(ADD, R12, 128, false); }, &[0x41, 0x81, 0xc4, 0x80, 0x00, 0x00, 0x00]),
        ("add r12d, -128", |a| { a.alu_ri(ADD, R12, -128, false); }, &[0x41, 0x83, 0xc4, 0x80]),
        ("add r12d, -129", |a| { a.alu_ri(ADD, R12, -129, false); }, &[0x41, 0x81, 0xc4, 0x7f, 0xff, 0xff, 0xff]),
        ("add r12d, 4096", |a| { a.alu_ri(ADD, R12, 4096, false); }, &[0x41, 0x81, 0xc4, 0x00, 0x10, 0x00, 0x00]),
        ("add r12d, 2147483647", |a| { a.alu_ri(ADD, R12, 2147483647, false); }, &[0x41, 0x81, 0xc4, 0xff, 0xff, 0xff, 0x7f]),
        ("add r12d, -2147483648", |a| { a.alu_ri(ADD, R12, -2147483648, false); }, &[0x41, 0x81, 0xc4, 0x00, 0x00, 0x00, 0x80]),
        ("or r12d, 5", |a| { a.alu_ri(OR, R12, 5, false); }, &[0x41, 0x83, 0xcc, 0x05]),
        ("or r12d, -1", |a| { a.alu_ri(OR, R12, -1, false); }, &[0x41, 0x83, 0xcc, 0xff]),
        ("or r12d, 127", |a| { a.alu_ri(OR, R12, 127, false); }, &[0x41, 0x83, 0xcc, 0x7f]),
        ("or r12d, 128", |a| { a.alu_ri(OR, R12, 128, false); }, &[0x41, 0x81, 0xcc, 0x80, 0x00, 0x00, 0x00]),
        ("or r12d, -128", |a| { a.alu_ri(OR, R12, -128, false); }, &[0x41, 0x83, 0xcc, 0x80]),
        ("or r12d, -129", |a| { a.alu_ri(OR, R12, -129, false); }, &[0x41, 0x81, 0xcc, 0x7f, 0xff, 0xff, 0xff]),
        ("or r12d, 4096", |a| { a.alu_ri(OR, R12, 4096, false); }, &[0x41, 0x81, 0xcc, 0x00, 0x10, 0x00, 0x00]),
        ("or r12d, 2147483647", |a| { a.alu_ri(OR, R12, 2147483647, false); }, &[0x41, 0x81, 0xcc, 0xff, 0xff, 0xff, 0x7f]),
        ("or r12d, -2147483648", |a| { a.alu_ri(OR, R12, -2147483648, false); }, &[0x41, 0x81, 0xcc, 0x00, 0x00, 0x00, 0x80]),
        ("and r12d, 5", |a| { a.alu_ri(AND, R12, 5, false); }, &[0x41, 0x83, 0xe4, 0x05]),
        ("and r12d, -1", |a| { a.alu_ri(AND, R12, -1, false); }, &[0x41, 0x83, 0xe4, 0xff]),
        ("and r12d, 127", |a| { a.alu_ri(AND, R12, 127, false); }, &[0x41, 0x83, 0xe4, 0x7f]),
        ("and r12d, 128", |a| { a.alu_ri(AND, R12, 128, false); }, &[0x41, 0x81, 0xe4, 0x80, 0x00, 0x00, 0x00]),
        ("and r12d, -128", |a| { a.alu_ri(AND, R12, -128, false); }, &[0x41, 0x83, 0xe4, 0x80]),
        ("and r12d, -129", |a| { a.alu_ri(AND, R12, -129, false); }, &[0x41, 0x81, 0xe4, 0x7f, 0xff, 0xff, 0xff]),
        ("and r12d, 4096", |a| { a.alu_ri(AND, R12, 4096, false); }, &[0x41, 0x81, 0xe4, 0x00, 0x10, 0x00, 0x00]),
        ("and r12d, 2147483647", |a| { a.alu_ri(AND, R12, 2147483647, false); }, &[0x41, 0x81, 0xe4, 0xff, 0xff, 0xff, 0x7f]),
        ("and r12d, -2147483648", |a| { a.alu_ri(AND, R12, -2147483648, false); }, &[0x41, 0x81, 0xe4, 0x00, 0x00, 0x00, 0x80]),
        ("sub r12d, 5", |a| { a.alu_ri(SUB, R12, 5, false); }, &[0x41, 0x83, 0xec, 0x05]),
        ("sub r12d, -1", |a| { a.alu_ri(SUB, R12, -1, false); }, &[0x41, 0x83, 0xec, 0xff]),
        ("sub r12d, 127", |a| { a.alu_ri(SUB, R12, 127, false); }, &[0x41, 0x83, 0xec, 0x7f]),
        ("sub r12d, 128", |a| { a.alu_ri(SUB, R12, 128, false); }, &[0x41, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00]),
        ("sub r12d, -128", |a| { a.alu_ri(SUB, R12, -128, false); }, &[0x41, 0x83, 0xec, 0x80]),
        ("sub r12d, -129", |a| { a.alu_ri(SUB, R12, -129, false); }, &[0x41, 0x81, 0xec, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r12d, 4096", |a| { a.alu_ri(SUB, R12, 4096, false); }, &[0x41, 0x81, 0xec, 0x00, 0x10, 0x00, 0x00]),
        ("sub r12d, 2147483647", |a| { a.alu_ri(SUB, R12, 2147483647, false); }, &[0x41, 0x81, 0xec, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r12d, -2147483648", |a| { a.alu_ri(SUB, R12, -2147483648, false); }, &[0x41, 0x81, 0xec, 0x00, 0x00, 0x00, 0x80]),
        ("xor r12d, 5", |a| { a.alu_ri(XOR, R12, 5, false); }, &[0x41, 0x83, 0xf4, 0x05]),
        ("xor r12d, -1", |a| { a.alu_ri(XOR, R12, -1, false); }, &[0x41, 0x83, 0xf4, 0xff]),
        ("xor r12d, 127", |a| { a.alu_ri(XOR, R12, 127, false); }, &[0x41, 0x83, 0xf4, 0x7f]),
        ("xor r12d, 128", |a| { a.alu_ri(XOR, R12, 128, false); }, &[0x41, 0x81, 0xf4, 0x80, 0x00, 0x00, 0x00]),
        ("xor r12d, -128", |a| { a.alu_ri(XOR, R12, -128, false); }, &[0x41, 0x83, 0xf4, 0x80]),
        ("xor r12d, -129", |a| { a.alu_ri(XOR, R12, -129, false); }, &[0x41, 0x81, 0xf4, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r12d, 4096", |a| { a.alu_ri(XOR, R12, 4096, false); }, &[0x41, 0x81, 0xf4, 0x00, 0x10, 0x00, 0x00]),
        ("xor r12d, 2147483647", |a| { a.alu_ri(XOR, R12, 2147483647, false); }, &[0x41, 0x81, 0xf4, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r12d, -2147483648", |a| { a.alu_ri(XOR, R12, -2147483648, false); }, &[0x41, 0x81, 0xf4, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r12d, 5", |a| { a.alu_ri(CMP, R12, 5, false); }, &[0x41, 0x83, 0xfc, 0x05]),
        ("cmp r12d, -1", |a| { a.alu_ri(CMP, R12, -1, false); }, &[0x41, 0x83, 0xfc, 0xff]),
        ("cmp r12d, 127", |a| { a.alu_ri(CMP, R12, 127, false); }, &[0x41, 0x83, 0xfc, 0x7f]),
        ("cmp r12d, 128", |a| { a.alu_ri(CMP, R12, 128, false); }, &[0x41, 0x81, 0xfc, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r12d, -128", |a| { a.alu_ri(CMP, R12, -128, false); }, &[0x41, 0x83, 0xfc, 0x80]),
        ("cmp r12d, -129", |a| { a.alu_ri(CMP, R12, -129, false); }, &[0x41, 0x81, 0xfc, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r12d, 4096", |a| { a.alu_ri(CMP, R12, 4096, false); }, &[0x41, 0x81, 0xfc, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r12d, 2147483647", |a| { a.alu_ri(CMP, R12, 2147483647, false); }, &[0x41, 0x81, 0xfc, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r12d, -2147483648", |a| { a.alu_ri(CMP, R12, -2147483648, false); }, &[0x41, 0x81, 0xfc, 0x00, 0x00, 0x00, 0x80]),
        ("add r13d, 5", |a| { a.alu_ri(ADD, R13, 5, false); }, &[0x41, 0x83, 0xc5, 0x05]),
        ("add r13d, -1", |a| { a.alu_ri(ADD, R13, -1, false); }, &[0x41, 0x83, 0xc5, 0xff]),
        ("add r13d, 127", |a| { a.alu_ri(ADD, R13, 127, false); }, &[0x41, 0x83, 0xc5, 0x7f]),
        ("add r13d, 128", |a| { a.alu_ri(ADD, R13, 128, false); }, &[0x41, 0x81, 0xc5, 0x80, 0x00, 0x00, 0x00]),
        ("add r13d, -128", |a| { a.alu_ri(ADD, R13, -128, false); }, &[0x41, 0x83, 0xc5, 0x80]),
        ("add r13d, -129", |a| { a.alu_ri(ADD, R13, -129, false); }, &[0x41, 0x81, 0xc5, 0x7f, 0xff, 0xff, 0xff]),
        ("add r13d, 4096", |a| { a.alu_ri(ADD, R13, 4096, false); }, &[0x41, 0x81, 0xc5, 0x00, 0x10, 0x00, 0x00]),
        ("add r13d, 2147483647", |a| { a.alu_ri(ADD, R13, 2147483647, false); }, &[0x41, 0x81, 0xc5, 0xff, 0xff, 0xff, 0x7f]),
        ("add r13d, -2147483648", |a| { a.alu_ri(ADD, R13, -2147483648, false); }, &[0x41, 0x81, 0xc5, 0x00, 0x00, 0x00, 0x80]),
        ("or r13d, 5", |a| { a.alu_ri(OR, R13, 5, false); }, &[0x41, 0x83, 0xcd, 0x05]),
        ("or r13d, -1", |a| { a.alu_ri(OR, R13, -1, false); }, &[0x41, 0x83, 0xcd, 0xff]),
        ("or r13d, 127", |a| { a.alu_ri(OR, R13, 127, false); }, &[0x41, 0x83, 0xcd, 0x7f]),
        ("or r13d, 128", |a| { a.alu_ri(OR, R13, 128, false); }, &[0x41, 0x81, 0xcd, 0x80, 0x00, 0x00, 0x00]),
        ("or r13d, -128", |a| { a.alu_ri(OR, R13, -128, false); }, &[0x41, 0x83, 0xcd, 0x80]),
        ("or r13d, -129", |a| { a.alu_ri(OR, R13, -129, false); }, &[0x41, 0x81, 0xcd, 0x7f, 0xff, 0xff, 0xff]),
        ("or r13d, 4096", |a| { a.alu_ri(OR, R13, 4096, false); }, &[0x41, 0x81, 0xcd, 0x00, 0x10, 0x00, 0x00]),
        ("or r13d, 2147483647", |a| { a.alu_ri(OR, R13, 2147483647, false); }, &[0x41, 0x81, 0xcd, 0xff, 0xff, 0xff, 0x7f]),
        ("or r13d, -2147483648", |a| { a.alu_ri(OR, R13, -2147483648, false); }, &[0x41, 0x81, 0xcd, 0x00, 0x00, 0x00, 0x80]),
        ("and r13d, 5", |a| { a.alu_ri(AND, R13, 5, false); }, &[0x41, 0x83, 0xe5, 0x05]),
        ("and r13d, -1", |a| { a.alu_ri(AND, R13, -1, false); }, &[0x41, 0x83, 0xe5, 0xff]),
        ("and r13d, 127", |a| { a.alu_ri(AND, R13, 127, false); }, &[0x41, 0x83, 0xe5, 0x7f]),
        ("and r13d, 128", |a| { a.alu_ri(AND, R13, 128, false); }, &[0x41, 0x81, 0xe5, 0x80, 0x00, 0x00, 0x00]),
        ("and r13d, -128", |a| { a.alu_ri(AND, R13, -128, false); }, &[0x41, 0x83, 0xe5, 0x80]),
        ("and r13d, -129", |a| { a.alu_ri(AND, R13, -129, false); }, &[0x41, 0x81, 0xe5, 0x7f, 0xff, 0xff, 0xff]),
        ("and r13d, 4096", |a| { a.alu_ri(AND, R13, 4096, false); }, &[0x41, 0x81, 0xe5, 0x00, 0x10, 0x00, 0x00]),
        ("and r13d, 2147483647", |a| { a.alu_ri(AND, R13, 2147483647, false); }, &[0x41, 0x81, 0xe5, 0xff, 0xff, 0xff, 0x7f]),
        ("and r13d, -2147483648", |a| { a.alu_ri(AND, R13, -2147483648, false); }, &[0x41, 0x81, 0xe5, 0x00, 0x00, 0x00, 0x80]),
        ("sub r13d, 5", |a| { a.alu_ri(SUB, R13, 5, false); }, &[0x41, 0x83, 0xed, 0x05]),
        ("sub r13d, -1", |a| { a.alu_ri(SUB, R13, -1, false); }, &[0x41, 0x83, 0xed, 0xff]),
        ("sub r13d, 127", |a| { a.alu_ri(SUB, R13, 127, false); }, &[0x41, 0x83, 0xed, 0x7f]),
        ("sub r13d, 128", |a| { a.alu_ri(SUB, R13, 128, false); }, &[0x41, 0x81, 0xed, 0x80, 0x00, 0x00, 0x00]),
        ("sub r13d, -128", |a| { a.alu_ri(SUB, R13, -128, false); }, &[0x41, 0x83, 0xed, 0x80]),
        ("sub r13d, -129", |a| { a.alu_ri(SUB, R13, -129, false); }, &[0x41, 0x81, 0xed, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r13d, 4096", |a| { a.alu_ri(SUB, R13, 4096, false); }, &[0x41, 0x81, 0xed, 0x00, 0x10, 0x00, 0x00]),
        ("sub r13d, 2147483647", |a| { a.alu_ri(SUB, R13, 2147483647, false); }, &[0x41, 0x81, 0xed, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r13d, -2147483648", |a| { a.alu_ri(SUB, R13, -2147483648, false); }, &[0x41, 0x81, 0xed, 0x00, 0x00, 0x00, 0x80]),
        ("xor r13d, 5", |a| { a.alu_ri(XOR, R13, 5, false); }, &[0x41, 0x83, 0xf5, 0x05]),
        ("xor r13d, -1", |a| { a.alu_ri(XOR, R13, -1, false); }, &[0x41, 0x83, 0xf5, 0xff]),
        ("xor r13d, 127", |a| { a.alu_ri(XOR, R13, 127, false); }, &[0x41, 0x83, 0xf5, 0x7f]),
        ("xor r13d, 128", |a| { a.alu_ri(XOR, R13, 128, false); }, &[0x41, 0x81, 0xf5, 0x80, 0x00, 0x00, 0x00]),
        ("xor r13d, -128", |a| { a.alu_ri(XOR, R13, -128, false); }, &[0x41, 0x83, 0xf5, 0x80]),
        ("xor r13d, -129", |a| { a.alu_ri(XOR, R13, -129, false); }, &[0x41, 0x81, 0xf5, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r13d, 4096", |a| { a.alu_ri(XOR, R13, 4096, false); }, &[0x41, 0x81, 0xf5, 0x00, 0x10, 0x00, 0x00]),
        ("xor r13d, 2147483647", |a| { a.alu_ri(XOR, R13, 2147483647, false); }, &[0x41, 0x81, 0xf5, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r13d, -2147483648", |a| { a.alu_ri(XOR, R13, -2147483648, false); }, &[0x41, 0x81, 0xf5, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r13d, 5", |a| { a.alu_ri(CMP, R13, 5, false); }, &[0x41, 0x83, 0xfd, 0x05]),
        ("cmp r13d, -1", |a| { a.alu_ri(CMP, R13, -1, false); }, &[0x41, 0x83, 0xfd, 0xff]),
        ("cmp r13d, 127", |a| { a.alu_ri(CMP, R13, 127, false); }, &[0x41, 0x83, 0xfd, 0x7f]),
        ("cmp r13d, 128", |a| { a.alu_ri(CMP, R13, 128, false); }, &[0x41, 0x81, 0xfd, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r13d, -128", |a| { a.alu_ri(CMP, R13, -128, false); }, &[0x41, 0x83, 0xfd, 0x80]),
        ("cmp r13d, -129", |a| { a.alu_ri(CMP, R13, -129, false); }, &[0x41, 0x81, 0xfd, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r13d, 4096", |a| { a.alu_ri(CMP, R13, 4096, false); }, &[0x41, 0x81, 0xfd, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r13d, 2147483647", |a| { a.alu_ri(CMP, R13, 2147483647, false); }, &[0x41, 0x81, 0xfd, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r13d, -2147483648", |a| { a.alu_ri(CMP, R13, -2147483648, false); }, &[0x41, 0x81, 0xfd, 0x00, 0x00, 0x00, 0x80]),
        ("add r15d, 5", |a| { a.alu_ri(ADD, R15, 5, false); }, &[0x41, 0x83, 0xc7, 0x05]),
        ("add r15d, -1", |a| { a.alu_ri(ADD, R15, -1, false); }, &[0x41, 0x83, 0xc7, 0xff]),
        ("add r15d, 127", |a| { a.alu_ri(ADD, R15, 127, false); }, &[0x41, 0x83, 0xc7, 0x7f]),
        ("add r15d, 128", |a| { a.alu_ri(ADD, R15, 128, false); }, &[0x41, 0x81, 0xc7, 0x80, 0x00, 0x00, 0x00]),
        ("add r15d, -128", |a| { a.alu_ri(ADD, R15, -128, false); }, &[0x41, 0x83, 0xc7, 0x80]),
        ("add r15d, -129", |a| { a.alu_ri(ADD, R15, -129, false); }, &[0x41, 0x81, 0xc7, 0x7f, 0xff, 0xff, 0xff]),
        ("add r15d, 4096", |a| { a.alu_ri(ADD, R15, 4096, false); }, &[0x41, 0x81, 0xc7, 0x00, 0x10, 0x00, 0x00]),
        ("add r15d, 2147483647", |a| { a.alu_ri(ADD, R15, 2147483647, false); }, &[0x41, 0x81, 0xc7, 0xff, 0xff, 0xff, 0x7f]),
        ("add r15d, -2147483648", |a| { a.alu_ri(ADD, R15, -2147483648, false); }, &[0x41, 0x81, 0xc7, 0x00, 0x00, 0x00, 0x80]),
        ("or r15d, 5", |a| { a.alu_ri(OR, R15, 5, false); }, &[0x41, 0x83, 0xcf, 0x05]),
        ("or r15d, -1", |a| { a.alu_ri(OR, R15, -1, false); }, &[0x41, 0x83, 0xcf, 0xff]),
        ("or r15d, 127", |a| { a.alu_ri(OR, R15, 127, false); }, &[0x41, 0x83, 0xcf, 0x7f]),
        ("or r15d, 128", |a| { a.alu_ri(OR, R15, 128, false); }, &[0x41, 0x81, 0xcf, 0x80, 0x00, 0x00, 0x00]),
        ("or r15d, -128", |a| { a.alu_ri(OR, R15, -128, false); }, &[0x41, 0x83, 0xcf, 0x80]),
        ("or r15d, -129", |a| { a.alu_ri(OR, R15, -129, false); }, &[0x41, 0x81, 0xcf, 0x7f, 0xff, 0xff, 0xff]),
        ("or r15d, 4096", |a| { a.alu_ri(OR, R15, 4096, false); }, &[0x41, 0x81, 0xcf, 0x00, 0x10, 0x00, 0x00]),
        ("or r15d, 2147483647", |a| { a.alu_ri(OR, R15, 2147483647, false); }, &[0x41, 0x81, 0xcf, 0xff, 0xff, 0xff, 0x7f]),
        ("or r15d, -2147483648", |a| { a.alu_ri(OR, R15, -2147483648, false); }, &[0x41, 0x81, 0xcf, 0x00, 0x00, 0x00, 0x80]),
        ("and r15d, 5", |a| { a.alu_ri(AND, R15, 5, false); }, &[0x41, 0x83, 0xe7, 0x05]),
        ("and r15d, -1", |a| { a.alu_ri(AND, R15, -1, false); }, &[0x41, 0x83, 0xe7, 0xff]),
        ("and r15d, 127", |a| { a.alu_ri(AND, R15, 127, false); }, &[0x41, 0x83, 0xe7, 0x7f]),
        ("and r15d, 128", |a| { a.alu_ri(AND, R15, 128, false); }, &[0x41, 0x81, 0xe7, 0x80, 0x00, 0x00, 0x00]),
        ("and r15d, -128", |a| { a.alu_ri(AND, R15, -128, false); }, &[0x41, 0x83, 0xe7, 0x80]),
        ("and r15d, -129", |a| { a.alu_ri(AND, R15, -129, false); }, &[0x41, 0x81, 0xe7, 0x7f, 0xff, 0xff, 0xff]),
        ("and r15d, 4096", |a| { a.alu_ri(AND, R15, 4096, false); }, &[0x41, 0x81, 0xe7, 0x00, 0x10, 0x00, 0x00]),
        ("and r15d, 2147483647", |a| { a.alu_ri(AND, R15, 2147483647, false); }, &[0x41, 0x81, 0xe7, 0xff, 0xff, 0xff, 0x7f]),
        ("and r15d, -2147483648", |a| { a.alu_ri(AND, R15, -2147483648, false); }, &[0x41, 0x81, 0xe7, 0x00, 0x00, 0x00, 0x80]),
        ("sub r15d, 5", |a| { a.alu_ri(SUB, R15, 5, false); }, &[0x41, 0x83, 0xef, 0x05]),
        ("sub r15d, -1", |a| { a.alu_ri(SUB, R15, -1, false); }, &[0x41, 0x83, 0xef, 0xff]),
        ("sub r15d, 127", |a| { a.alu_ri(SUB, R15, 127, false); }, &[0x41, 0x83, 0xef, 0x7f]),
        ("sub r15d, 128", |a| { a.alu_ri(SUB, R15, 128, false); }, &[0x41, 0x81, 0xef, 0x80, 0x00, 0x00, 0x00]),
        ("sub r15d, -128", |a| { a.alu_ri(SUB, R15, -128, false); }, &[0x41, 0x83, 0xef, 0x80]),
        ("sub r15d, -129", |a| { a.alu_ri(SUB, R15, -129, false); }, &[0x41, 0x81, 0xef, 0x7f, 0xff, 0xff, 0xff]),
        ("sub r15d, 4096", |a| { a.alu_ri(SUB, R15, 4096, false); }, &[0x41, 0x81, 0xef, 0x00, 0x10, 0x00, 0x00]),
        ("sub r15d, 2147483647", |a| { a.alu_ri(SUB, R15, 2147483647, false); }, &[0x41, 0x81, 0xef, 0xff, 0xff, 0xff, 0x7f]),
        ("sub r15d, -2147483648", |a| { a.alu_ri(SUB, R15, -2147483648, false); }, &[0x41, 0x81, 0xef, 0x00, 0x00, 0x00, 0x80]),
        ("xor r15d, 5", |a| { a.alu_ri(XOR, R15, 5, false); }, &[0x41, 0x83, 0xf7, 0x05]),
        ("xor r15d, -1", |a| { a.alu_ri(XOR, R15, -1, false); }, &[0x41, 0x83, 0xf7, 0xff]),
        ("xor r15d, 127", |a| { a.alu_ri(XOR, R15, 127, false); }, &[0x41, 0x83, 0xf7, 0x7f]),
        ("xor r15d, 128", |a| { a.alu_ri(XOR, R15, 128, false); }, &[0x41, 0x81, 0xf7, 0x80, 0x00, 0x00, 0x00]),
        ("xor r15d, -128", |a| { a.alu_ri(XOR, R15, -128, false); }, &[0x41, 0x83, 0xf7, 0x80]),
        ("xor r15d, -129", |a| { a.alu_ri(XOR, R15, -129, false); }, &[0x41, 0x81, 0xf7, 0x7f, 0xff, 0xff, 0xff]),
        ("xor r15d, 4096", |a| { a.alu_ri(XOR, R15, 4096, false); }, &[0x41, 0x81, 0xf7, 0x00, 0x10, 0x00, 0x00]),
        ("xor r15d, 2147483647", |a| { a.alu_ri(XOR, R15, 2147483647, false); }, &[0x41, 0x81, 0xf7, 0xff, 0xff, 0xff, 0x7f]),
        ("xor r15d, -2147483648", |a| { a.alu_ri(XOR, R15, -2147483648, false); }, &[0x41, 0x81, 0xf7, 0x00, 0x00, 0x00, 0x80]),
        ("cmp r15d, 5", |a| { a.alu_ri(CMP, R15, 5, false); }, &[0x41, 0x83, 0xff, 0x05]),
        ("cmp r15d, -1", |a| { a.alu_ri(CMP, R15, -1, false); }, &[0x41, 0x83, 0xff, 0xff]),
        ("cmp r15d, 127", |a| { a.alu_ri(CMP, R15, 127, false); }, &[0x41, 0x83, 0xff, 0x7f]),
        ("cmp r15d, 128", |a| { a.alu_ri(CMP, R15, 128, false); }, &[0x41, 0x81, 0xff, 0x80, 0x00, 0x00, 0x00]),
        ("cmp r15d, -128", |a| { a.alu_ri(CMP, R15, -128, false); }, &[0x41, 0x83, 0xff, 0x80]),
        ("cmp r15d, -129", |a| { a.alu_ri(CMP, R15, -129, false); }, &[0x41, 0x81, 0xff, 0x7f, 0xff, 0xff, 0xff]),
        ("cmp r15d, 4096", |a| { a.alu_ri(CMP, R15, 4096, false); }, &[0x41, 0x81, 0xff, 0x00, 0x10, 0x00, 0x00]),
        ("cmp r15d, 2147483647", |a| { a.alu_ri(CMP, R15, 2147483647, false); }, &[0x41, 0x81, 0xff, 0xff, 0xff, 0xff, 0x7f]),
        ("cmp r15d, -2147483648", |a| { a.alu_ri(CMP, R15, -2147483648, false); }, &[0x41, 0x81, 0xff, 0x00, 0x00, 0x00, 0x80]),
    ];
    check_encodings(cases);
}

    /// Shl/shr by an immediate above one (a count of one has its own shorter encoding).
#[rustfmt::skip]
#[test]
fn immediate_shifts_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("shl rcx, 2", |a| { a.shift_ri(SHL, RCX, 2, true); }, &[0x48, 0xc1, 0xe1, 0x02]),
        ("shl rcx, 4", |a| { a.shift_ri(SHL, RCX, 4, true); }, &[0x48, 0xc1, 0xe1, 0x04]),
        ("shl rcx, 12", |a| { a.shift_ri(SHL, RCX, 12, true); }, &[0x48, 0xc1, 0xe1, 0x0c]),
        ("shl rcx, 31", |a| { a.shift_ri(SHL, RCX, 31, true); }, &[0x48, 0xc1, 0xe1, 0x1f]),
        ("shl rcx, 63", |a| { a.shift_ri(SHL, RCX, 63, true); }, &[0x48, 0xc1, 0xe1, 0x3f]),
        ("shr rcx, 2", |a| { a.shift_ri(SHR, RCX, 2, true); }, &[0x48, 0xc1, 0xe9, 0x02]),
        ("shr rcx, 4", |a| { a.shift_ri(SHR, RCX, 4, true); }, &[0x48, 0xc1, 0xe9, 0x04]),
        ("shr rcx, 12", |a| { a.shift_ri(SHR, RCX, 12, true); }, &[0x48, 0xc1, 0xe9, 0x0c]),
        ("shr rcx, 31", |a| { a.shift_ri(SHR, RCX, 31, true); }, &[0x48, 0xc1, 0xe9, 0x1f]),
        ("shr rcx, 63", |a| { a.shift_ri(SHR, RCX, 63, true); }, &[0x48, 0xc1, 0xe9, 0x3f]),
        ("shl rsp, 2", |a| { a.shift_ri(SHL, RSP, 2, true); }, &[0x48, 0xc1, 0xe4, 0x02]),
        ("shl rsp, 4", |a| { a.shift_ri(SHL, RSP, 4, true); }, &[0x48, 0xc1, 0xe4, 0x04]),
        ("shl rsp, 12", |a| { a.shift_ri(SHL, RSP, 12, true); }, &[0x48, 0xc1, 0xe4, 0x0c]),
        ("shl rsp, 31", |a| { a.shift_ri(SHL, RSP, 31, true); }, &[0x48, 0xc1, 0xe4, 0x1f]),
        ("shl rsp, 63", |a| { a.shift_ri(SHL, RSP, 63, true); }, &[0x48, 0xc1, 0xe4, 0x3f]),
        ("shr rsp, 2", |a| { a.shift_ri(SHR, RSP, 2, true); }, &[0x48, 0xc1, 0xec, 0x02]),
        ("shr rsp, 4", |a| { a.shift_ri(SHR, RSP, 4, true); }, &[0x48, 0xc1, 0xec, 0x04]),
        ("shr rsp, 12", |a| { a.shift_ri(SHR, RSP, 12, true); }, &[0x48, 0xc1, 0xec, 0x0c]),
        ("shr rsp, 31", |a| { a.shift_ri(SHR, RSP, 31, true); }, &[0x48, 0xc1, 0xec, 0x1f]),
        ("shr rsp, 63", |a| { a.shift_ri(SHR, RSP, 63, true); }, &[0x48, 0xc1, 0xec, 0x3f]),
        ("shl rbp, 2", |a| { a.shift_ri(SHL, RBP, 2, true); }, &[0x48, 0xc1, 0xe5, 0x02]),
        ("shl rbp, 4", |a| { a.shift_ri(SHL, RBP, 4, true); }, &[0x48, 0xc1, 0xe5, 0x04]),
        ("shl rbp, 12", |a| { a.shift_ri(SHL, RBP, 12, true); }, &[0x48, 0xc1, 0xe5, 0x0c]),
        ("shl rbp, 31", |a| { a.shift_ri(SHL, RBP, 31, true); }, &[0x48, 0xc1, 0xe5, 0x1f]),
        ("shl rbp, 63", |a| { a.shift_ri(SHL, RBP, 63, true); }, &[0x48, 0xc1, 0xe5, 0x3f]),
        ("shr rbp, 2", |a| { a.shift_ri(SHR, RBP, 2, true); }, &[0x48, 0xc1, 0xed, 0x02]),
        ("shr rbp, 4", |a| { a.shift_ri(SHR, RBP, 4, true); }, &[0x48, 0xc1, 0xed, 0x04]),
        ("shr rbp, 12", |a| { a.shift_ri(SHR, RBP, 12, true); }, &[0x48, 0xc1, 0xed, 0x0c]),
        ("shr rbp, 31", |a| { a.shift_ri(SHR, RBP, 31, true); }, &[0x48, 0xc1, 0xed, 0x1f]),
        ("shr rbp, 63", |a| { a.shift_ri(SHR, RBP, 63, true); }, &[0x48, 0xc1, 0xed, 0x3f]),
        ("shl r9, 2", |a| { a.shift_ri(SHL, R9, 2, true); }, &[0x49, 0xc1, 0xe1, 0x02]),
        ("shl r9, 4", |a| { a.shift_ri(SHL, R9, 4, true); }, &[0x49, 0xc1, 0xe1, 0x04]),
        ("shl r9, 12", |a| { a.shift_ri(SHL, R9, 12, true); }, &[0x49, 0xc1, 0xe1, 0x0c]),
        ("shl r9, 31", |a| { a.shift_ri(SHL, R9, 31, true); }, &[0x49, 0xc1, 0xe1, 0x1f]),
        ("shl r9, 63", |a| { a.shift_ri(SHL, R9, 63, true); }, &[0x49, 0xc1, 0xe1, 0x3f]),
        ("shr r9, 2", |a| { a.shift_ri(SHR, R9, 2, true); }, &[0x49, 0xc1, 0xe9, 0x02]),
        ("shr r9, 4", |a| { a.shift_ri(SHR, R9, 4, true); }, &[0x49, 0xc1, 0xe9, 0x04]),
        ("shr r9, 12", |a| { a.shift_ri(SHR, R9, 12, true); }, &[0x49, 0xc1, 0xe9, 0x0c]),
        ("shr r9, 31", |a| { a.shift_ri(SHR, R9, 31, true); }, &[0x49, 0xc1, 0xe9, 0x1f]),
        ("shr r9, 63", |a| { a.shift_ri(SHR, R9, 63, true); }, &[0x49, 0xc1, 0xe9, 0x3f]),
        ("shl r13, 2", |a| { a.shift_ri(SHL, R13, 2, true); }, &[0x49, 0xc1, 0xe5, 0x02]),
        ("shl r13, 4", |a| { a.shift_ri(SHL, R13, 4, true); }, &[0x49, 0xc1, 0xe5, 0x04]),
        ("shl r13, 12", |a| { a.shift_ri(SHL, R13, 12, true); }, &[0x49, 0xc1, 0xe5, 0x0c]),
        ("shl r13, 31", |a| { a.shift_ri(SHL, R13, 31, true); }, &[0x49, 0xc1, 0xe5, 0x1f]),
        ("shl r13, 63", |a| { a.shift_ri(SHL, R13, 63, true); }, &[0x49, 0xc1, 0xe5, 0x3f]),
        ("shr r13, 2", |a| { a.shift_ri(SHR, R13, 2, true); }, &[0x49, 0xc1, 0xed, 0x02]),
        ("shr r13, 4", |a| { a.shift_ri(SHR, R13, 4, true); }, &[0x49, 0xc1, 0xed, 0x04]),
        ("shr r13, 12", |a| { a.shift_ri(SHR, R13, 12, true); }, &[0x49, 0xc1, 0xed, 0x0c]),
        ("shr r13, 31", |a| { a.shift_ri(SHR, R13, 31, true); }, &[0x49, 0xc1, 0xed, 0x1f]),
        ("shr r13, 63", |a| { a.shift_ri(SHR, R13, 63, true); }, &[0x49, 0xc1, 0xed, 0x3f]),
        ("shl ecx, 2", |a| { a.shift_ri(SHL, RCX, 2, false); }, &[0xc1, 0xe1, 0x02]),
        ("shl ecx, 4", |a| { a.shift_ri(SHL, RCX, 4, false); }, &[0xc1, 0xe1, 0x04]),
        ("shl ecx, 12", |a| { a.shift_ri(SHL, RCX, 12, false); }, &[0xc1, 0xe1, 0x0c]),
        ("shl ecx, 31", |a| { a.shift_ri(SHL, RCX, 31, false); }, &[0xc1, 0xe1, 0x1f]),
        ("shr ecx, 2", |a| { a.shift_ri(SHR, RCX, 2, false); }, &[0xc1, 0xe9, 0x02]),
        ("shr ecx, 4", |a| { a.shift_ri(SHR, RCX, 4, false); }, &[0xc1, 0xe9, 0x04]),
        ("shr ecx, 12", |a| { a.shift_ri(SHR, RCX, 12, false); }, &[0xc1, 0xe9, 0x0c]),
        ("shr ecx, 31", |a| { a.shift_ri(SHR, RCX, 31, false); }, &[0xc1, 0xe9, 0x1f]),
        ("shl esp, 2", |a| { a.shift_ri(SHL, RSP, 2, false); }, &[0xc1, 0xe4, 0x02]),
        ("shl esp, 4", |a| { a.shift_ri(SHL, RSP, 4, false); }, &[0xc1, 0xe4, 0x04]),
        ("shl esp, 12", |a| { a.shift_ri(SHL, RSP, 12, false); }, &[0xc1, 0xe4, 0x0c]),
        ("shl esp, 31", |a| { a.shift_ri(SHL, RSP, 31, false); }, &[0xc1, 0xe4, 0x1f]),
        ("shr esp, 2", |a| { a.shift_ri(SHR, RSP, 2, false); }, &[0xc1, 0xec, 0x02]),
        ("shr esp, 4", |a| { a.shift_ri(SHR, RSP, 4, false); }, &[0xc1, 0xec, 0x04]),
        ("shr esp, 12", |a| { a.shift_ri(SHR, RSP, 12, false); }, &[0xc1, 0xec, 0x0c]),
        ("shr esp, 31", |a| { a.shift_ri(SHR, RSP, 31, false); }, &[0xc1, 0xec, 0x1f]),
        ("shl ebp, 2", |a| { a.shift_ri(SHL, RBP, 2, false); }, &[0xc1, 0xe5, 0x02]),
        ("shl ebp, 4", |a| { a.shift_ri(SHL, RBP, 4, false); }, &[0xc1, 0xe5, 0x04]),
        ("shl ebp, 12", |a| { a.shift_ri(SHL, RBP, 12, false); }, &[0xc1, 0xe5, 0x0c]),
        ("shl ebp, 31", |a| { a.shift_ri(SHL, RBP, 31, false); }, &[0xc1, 0xe5, 0x1f]),
        ("shr ebp, 2", |a| { a.shift_ri(SHR, RBP, 2, false); }, &[0xc1, 0xed, 0x02]),
        ("shr ebp, 4", |a| { a.shift_ri(SHR, RBP, 4, false); }, &[0xc1, 0xed, 0x04]),
        ("shr ebp, 12", |a| { a.shift_ri(SHR, RBP, 12, false); }, &[0xc1, 0xed, 0x0c]),
        ("shr ebp, 31", |a| { a.shift_ri(SHR, RBP, 31, false); }, &[0xc1, 0xed, 0x1f]),
        ("shl r9d, 2", |a| { a.shift_ri(SHL, R9, 2, false); }, &[0x41, 0xc1, 0xe1, 0x02]),
        ("shl r9d, 4", |a| { a.shift_ri(SHL, R9, 4, false); }, &[0x41, 0xc1, 0xe1, 0x04]),
        ("shl r9d, 12", |a| { a.shift_ri(SHL, R9, 12, false); }, &[0x41, 0xc1, 0xe1, 0x0c]),
        ("shl r9d, 31", |a| { a.shift_ri(SHL, R9, 31, false); }, &[0x41, 0xc1, 0xe1, 0x1f]),
        ("shr r9d, 2", |a| { a.shift_ri(SHR, R9, 2, false); }, &[0x41, 0xc1, 0xe9, 0x02]),
        ("shr r9d, 4", |a| { a.shift_ri(SHR, R9, 4, false); }, &[0x41, 0xc1, 0xe9, 0x04]),
        ("shr r9d, 12", |a| { a.shift_ri(SHR, R9, 12, false); }, &[0x41, 0xc1, 0xe9, 0x0c]),
        ("shr r9d, 31", |a| { a.shift_ri(SHR, R9, 31, false); }, &[0x41, 0xc1, 0xe9, 0x1f]),
        ("shl r13d, 2", |a| { a.shift_ri(SHL, R13, 2, false); }, &[0x41, 0xc1, 0xe5, 0x02]),
        ("shl r13d, 4", |a| { a.shift_ri(SHL, R13, 4, false); }, &[0x41, 0xc1, 0xe5, 0x04]),
        ("shl r13d, 12", |a| { a.shift_ri(SHL, R13, 12, false); }, &[0x41, 0xc1, 0xe5, 0x0c]),
        ("shl r13d, 31", |a| { a.shift_ri(SHL, R13, 31, false); }, &[0x41, 0xc1, 0xe5, 0x1f]),
        ("shr r13d, 2", |a| { a.shift_ri(SHR, R13, 2, false); }, &[0x41, 0xc1, 0xed, 0x02]),
        ("shr r13d, 4", |a| { a.shift_ri(SHR, R13, 4, false); }, &[0x41, 0xc1, 0xed, 0x04]),
        ("shr r13d, 12", |a| { a.shift_ri(SHR, R13, 12, false); }, &[0x41, 0xc1, 0xed, 0x0c]),
        ("shr r13d, 31", |a| { a.shift_ri(SHR, R13, 31, false); }, &[0x41, 0xc1, 0xed, 0x1f]),
    ];
    check_encodings(cases);
}

    /// Mov r32, imm32 for 0..=u32::MAX, mov r64, imm32 (sign-extended) for negative 32-bit values, movabs otherwise.
#[rustfmt::skip]
#[test]
fn constant_loads_pick_the_shortest_encoding_like_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("mov eax, 0x0", |a| { a.mov_imm(RAX, 0x0); }, &[0xb8, 0x00, 0x00, 0x00, 0x00]),
        ("mov ecx, 0x0", |a| { a.mov_imm(RCX, 0x0); }, &[0xb9, 0x00, 0x00, 0x00, 0x00]),
        ("mov esp, 0x0", |a| { a.mov_imm(RSP, 0x0); }, &[0xbc, 0x00, 0x00, 0x00, 0x00]),
        ("mov ebp, 0x0", |a| { a.mov_imm(RBP, 0x0); }, &[0xbd, 0x00, 0x00, 0x00, 0x00]),
        ("mov r8d, 0x0", |a| { a.mov_imm(R8, 0x0); }, &[0x41, 0xb8, 0x00, 0x00, 0x00, 0x00]),
        ("mov r12d, 0x0", |a| { a.mov_imm(R12, 0x0); }, &[0x41, 0xbc, 0x00, 0x00, 0x00, 0x00]),
        ("mov r13d, 0x0", |a| { a.mov_imm(R13, 0x0); }, &[0x41, 0xbd, 0x00, 0x00, 0x00, 0x00]),
        ("mov r15d, 0x0", |a| { a.mov_imm(R15, 0x0); }, &[0x41, 0xbf, 0x00, 0x00, 0x00, 0x00]),
        ("mov eax, 0x1", |a| { a.mov_imm(RAX, 0x1); }, &[0xb8, 0x01, 0x00, 0x00, 0x00]),
        ("mov ecx, 0x1", |a| { a.mov_imm(RCX, 0x1); }, &[0xb9, 0x01, 0x00, 0x00, 0x00]),
        ("mov esp, 0x1", |a| { a.mov_imm(RSP, 0x1); }, &[0xbc, 0x01, 0x00, 0x00, 0x00]),
        ("mov ebp, 0x1", |a| { a.mov_imm(RBP, 0x1); }, &[0xbd, 0x01, 0x00, 0x00, 0x00]),
        ("mov r8d, 0x1", |a| { a.mov_imm(R8, 0x1); }, &[0x41, 0xb8, 0x01, 0x00, 0x00, 0x00]),
        ("mov r12d, 0x1", |a| { a.mov_imm(R12, 0x1); }, &[0x41, 0xbc, 0x01, 0x00, 0x00, 0x00]),
        ("mov r13d, 0x1", |a| { a.mov_imm(R13, 0x1); }, &[0x41, 0xbd, 0x01, 0x00, 0x00, 0x00]),
        ("mov r15d, 0x1", |a| { a.mov_imm(R15, 0x1); }, &[0x41, 0xbf, 0x01, 0x00, 0x00, 0x00]),
        ("mov eax, 0x7fffffff", |a| { a.mov_imm(RAX, 0x7fffffff); }, &[0xb8, 0xff, 0xff, 0xff, 0x7f]),
        ("mov ecx, 0x7fffffff", |a| { a.mov_imm(RCX, 0x7fffffff); }, &[0xb9, 0xff, 0xff, 0xff, 0x7f]),
        ("mov esp, 0x7fffffff", |a| { a.mov_imm(RSP, 0x7fffffff); }, &[0xbc, 0xff, 0xff, 0xff, 0x7f]),
        ("mov ebp, 0x7fffffff", |a| { a.mov_imm(RBP, 0x7fffffff); }, &[0xbd, 0xff, 0xff, 0xff, 0x7f]),
        ("mov r8d, 0x7fffffff", |a| { a.mov_imm(R8, 0x7fffffff); }, &[0x41, 0xb8, 0xff, 0xff, 0xff, 0x7f]),
        ("mov r12d, 0x7fffffff", |a| { a.mov_imm(R12, 0x7fffffff); }, &[0x41, 0xbc, 0xff, 0xff, 0xff, 0x7f]),
        ("mov r13d, 0x7fffffff", |a| { a.mov_imm(R13, 0x7fffffff); }, &[0x41, 0xbd, 0xff, 0xff, 0xff, 0x7f]),
        ("mov r15d, 0x7fffffff", |a| { a.mov_imm(R15, 0x7fffffff); }, &[0x41, 0xbf, 0xff, 0xff, 0xff, 0x7f]),
        ("mov eax, 0xffffffff", |a| { a.mov_imm(RAX, 0xffffffff); }, &[0xb8, 0xff, 0xff, 0xff, 0xff]),
        ("mov ecx, 0xffffffff", |a| { a.mov_imm(RCX, 0xffffffff); }, &[0xb9, 0xff, 0xff, 0xff, 0xff]),
        ("mov esp, 0xffffffff", |a| { a.mov_imm(RSP, 0xffffffff); }, &[0xbc, 0xff, 0xff, 0xff, 0xff]),
        ("mov ebp, 0xffffffff", |a| { a.mov_imm(RBP, 0xffffffff); }, &[0xbd, 0xff, 0xff, 0xff, 0xff]),
        ("mov r8d, 0xffffffff", |a| { a.mov_imm(R8, 0xffffffff); }, &[0x41, 0xb8, 0xff, 0xff, 0xff, 0xff]),
        ("mov r12d, 0xffffffff", |a| { a.mov_imm(R12, 0xffffffff); }, &[0x41, 0xbc, 0xff, 0xff, 0xff, 0xff]),
        ("mov r13d, 0xffffffff", |a| { a.mov_imm(R13, 0xffffffff); }, &[0x41, 0xbd, 0xff, 0xff, 0xff, 0xff]),
        ("mov r15d, 0xffffffff", |a| { a.mov_imm(R15, 0xffffffff); }, &[0x41, 0xbf, 0xff, 0xff, 0xff, 0xff]),
        ("mov rax, -1", |a| { a.mov_imm(RAX, 0xffffffffffffffff); }, &[0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]),
        ("mov r8, -1", |a| { a.mov_imm(R8, 0xffffffffffffffff); }, &[0x49, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]),
        ("mov r13, -1", |a| { a.mov_imm(R13, 0xffffffffffffffff); }, &[0x49, 0xc7, 0xc5, 0xff, 0xff, 0xff, 0xff]),
        ("mov rax, -2", |a| { a.mov_imm(RAX, 0xfffffffffffffffe); }, &[0x48, 0xc7, 0xc0, 0xfe, 0xff, 0xff, 0xff]),
        ("mov r8, -2", |a| { a.mov_imm(R8, 0xfffffffffffffffe); }, &[0x49, 0xc7, 0xc0, 0xfe, 0xff, 0xff, 0xff]),
        ("mov r13, -2", |a| { a.mov_imm(R13, 0xfffffffffffffffe); }, &[0x49, 0xc7, 0xc5, 0xfe, 0xff, 0xff, 0xff]),
        ("mov rax, -2147483648", |a| { a.mov_imm(RAX, 0xffffffff80000000); }, &[0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x80]),
        ("mov r8, -2147483648", |a| { a.mov_imm(R8, 0xffffffff80000000); }, &[0x49, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x80]),
        ("mov r13, -2147483648", |a| { a.mov_imm(R13, 0xffffffff80000000); }, &[0x49, 0xc7, 0xc5, 0x00, 0x00, 0x00, 0x80]),
        ("mov rax, -305419896", |a| { a.mov_imm(RAX, 0xffffffffedcba988); }, &[0x48, 0xc7, 0xc0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r8, -305419896", |a| { a.mov_imm(R8, 0xffffffffedcba988); }, &[0x49, 0xc7, 0xc0, 0x88, 0xa9, 0xcb, 0xed]),
        ("mov r13, -305419896", |a| { a.mov_imm(R13, 0xffffffffedcba988); }, &[0x49, 0xc7, 0xc5, 0x88, 0xa9, 0xcb, 0xed]),
        ("movabs rax, 0x100000000", |a| { a.mov_imm(RAX, 0x100000000); }, &[0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
        ("movabs r9, 0x100000000", |a| { a.mov_imm(R9, 0x100000000); }, &[0x49, 0xb9, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
        ("movabs r15, 0x100000000", |a| { a.mov_imm(R15, 0x100000000); }, &[0x49, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]),
        ("movabs rax, 0xffffffff7fffffff", |a| { a.mov_imm(RAX, 0xffffffff7fffffff); }, &[0x48, 0xb8, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff]),
        ("movabs r9, 0xffffffff7fffffff", |a| { a.mov_imm(R9, 0xffffffff7fffffff); }, &[0x49, 0xb9, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff]),
        ("movabs r15, 0xffffffff7fffffff", |a| { a.mov_imm(R15, 0xffffffff7fffffff); }, &[0x49, 0xbf, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff]),
        ("movabs rax, 0x123456789abcdef0", |a| { a.mov_imm(RAX, 0x123456789abcdef0); }, &[0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]),
        ("movabs r9, 0x123456789abcdef0", |a| { a.mov_imm(R9, 0x123456789abcdef0); }, &[0x49, 0xb9, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]),
        ("movabs r15, 0x123456789abcdef0", |a| { a.mov_imm(R15, 0x123456789abcdef0); }, &[0x49, 0xbf, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]),
        ("movabs rax, 0x8000000000000000", |a| { a.mov_imm(RAX, 0x8000000000000000); }, &[0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]),
        ("movabs r9, 0x8000000000000000", |a| { a.mov_imm(R9, 0x8000000000000000); }, &[0x49, 0xb9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]),
        ("movabs r15, 0x8000000000000000", |a| { a.mov_imm(R15, 0x8000000000000000); }, &[0x49, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]),
    ];
    check_encodings(cases);
}

    /// Movzx/movsx/mov/movsxd between registers; byte sources numbered 4..=7 get a REX prefix (sil/dil, not dh/bh).
#[rustfmt::skip]
#[test]
fn register_extensions_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("movzx eax, cl", |a| { a.ext_rr(RAX, RCX, 1, false); }, &[0x0f, 0xb6, 0xc1]),
        ("movsx rax, cl", |a| { a.ext_rr(RAX, RCX, 1, true); }, &[0x48, 0x0f, 0xbe, 0xc1]),
        ("movzx eax, cx", |a| { a.ext_rr(RAX, RCX, 2, false); }, &[0x0f, 0xb7, 0xc1]),
        ("movsx rax, cx", |a| { a.ext_rr(RAX, RCX, 2, true); }, &[0x48, 0x0f, 0xbf, 0xc1]),
        ("mov eax, ecx", |a| { a.ext_rr(RAX, RCX, 4, false); }, &[0x89, 0xc8]),
        ("movsxd rax, ecx", |a| { a.ext_rr(RAX, RCX, 4, true); }, &[0x48, 0x63, 0xc1]),
        ("mov rax, rcx", |a| { a.ext_rr(RAX, RCX, 8, false); }, &[0x48, 0x89, 0xc8]),
        ("movzx eax, sil", |a| { a.ext_rr(RAX, RSI, 1, false); }, &[0x40, 0x0f, 0xb6, 0xc6]),
        ("movsx rax, sil", |a| { a.ext_rr(RAX, RSI, 1, true); }, &[0x48, 0x0f, 0xbe, 0xc6]),
        ("movzx eax, si", |a| { a.ext_rr(RAX, RSI, 2, false); }, &[0x0f, 0xb7, 0xc6]),
        ("movsx rax, si", |a| { a.ext_rr(RAX, RSI, 2, true); }, &[0x48, 0x0f, 0xbf, 0xc6]),
        ("mov eax, esi", |a| { a.ext_rr(RAX, RSI, 4, false); }, &[0x89, 0xf0]),
        ("movsxd rax, esi", |a| { a.ext_rr(RAX, RSI, 4, true); }, &[0x48, 0x63, 0xc6]),
        ("mov rax, rsi", |a| { a.ext_rr(RAX, RSI, 8, false); }, &[0x48, 0x89, 0xf0]),
        ("movzx r9d, dil", |a| { a.ext_rr(R9, RDI, 1, false); }, &[0x44, 0x0f, 0xb6, 0xcf]),
        ("movsx r9, dil", |a| { a.ext_rr(R9, RDI, 1, true); }, &[0x4c, 0x0f, 0xbe, 0xcf]),
        ("movzx r9d, di", |a| { a.ext_rr(R9, RDI, 2, false); }, &[0x44, 0x0f, 0xb7, 0xcf]),
        ("movsx r9, di", |a| { a.ext_rr(R9, RDI, 2, true); }, &[0x4c, 0x0f, 0xbf, 0xcf]),
        ("mov r9d, edi", |a| { a.ext_rr(R9, RDI, 4, false); }, &[0x41, 0x89, 0xf9]),
        ("movsxd r9, edi", |a| { a.ext_rr(R9, RDI, 4, true); }, &[0x4c, 0x63, 0xcf]),
        ("mov r9, rdi", |a| { a.ext_rr(R9, RDI, 8, false); }, &[0x49, 0x89, 0xf9]),
        ("movzx r15d, spl", |a| { a.ext_rr(R15, RSP, 1, false); }, &[0x44, 0x0f, 0xb6, 0xfc]),
        ("movsx r15, spl", |a| { a.ext_rr(R15, RSP, 1, true); }, &[0x4c, 0x0f, 0xbe, 0xfc]),
        ("movzx r15d, sp", |a| { a.ext_rr(R15, RSP, 2, false); }, &[0x44, 0x0f, 0xb7, 0xfc]),
        ("movsx r15, sp", |a| { a.ext_rr(R15, RSP, 2, true); }, &[0x4c, 0x0f, 0xbf, 0xfc]),
        ("mov r15d, esp", |a| { a.ext_rr(R15, RSP, 4, false); }, &[0x41, 0x89, 0xe7]),
        ("movsxd r15, esp", |a| { a.ext_rr(R15, RSP, 4, true); }, &[0x4c, 0x63, 0xfc]),
        ("mov r15, rsp", |a| { a.ext_rr(R15, RSP, 8, false); }, &[0x49, 0x89, 0xe7]),
        ("movzx ebx, bpl", |a| { a.ext_rr(RBX, RBP, 1, false); }, &[0x40, 0x0f, 0xb6, 0xdd]),
        ("movsx rbx, bpl", |a| { a.ext_rr(RBX, RBP, 1, true); }, &[0x48, 0x0f, 0xbe, 0xdd]),
        ("movzx ebx, bp", |a| { a.ext_rr(RBX, RBP, 2, false); }, &[0x0f, 0xb7, 0xdd]),
        ("movsx rbx, bp", |a| { a.ext_rr(RBX, RBP, 2, true); }, &[0x48, 0x0f, 0xbf, 0xdd]),
        ("mov ebx, ebp", |a| { a.ext_rr(RBX, RBP, 4, false); }, &[0x89, 0xeb]),
        ("movsxd rbx, ebp", |a| { a.ext_rr(RBX, RBP, 4, true); }, &[0x48, 0x63, 0xdd]),
        ("mov rbx, rbp", |a| { a.ext_rr(RBX, RBP, 8, false); }, &[0x48, 0x89, 0xeb]),
        ("movzx r12d, r13b", |a| { a.ext_rr(R12, R13, 1, false); }, &[0x45, 0x0f, 0xb6, 0xe5]),
        ("movsx r12, r13b", |a| { a.ext_rr(R12, R13, 1, true); }, &[0x4d, 0x0f, 0xbe, 0xe5]),
        ("movzx r12d, r13w", |a| { a.ext_rr(R12, R13, 2, false); }, &[0x45, 0x0f, 0xb7, 0xe5]),
        ("movsx r12, r13w", |a| { a.ext_rr(R12, R13, 2, true); }, &[0x4d, 0x0f, 0xbf, 0xe5]),
        ("mov r12d, r13d", |a| { a.ext_rr(R12, R13, 4, false); }, &[0x45, 0x89, 0xec]),
        ("movsxd r12, r13d", |a| { a.ext_rr(R12, R13, 4, true); }, &[0x4d, 0x63, 0xe5]),
        ("mov r12, r13", |a| { a.ext_rr(R12, R13, 8, false); }, &[0x4d, 0x89, 0xec]),
    ];
    check_encodings(cases);
}

    /// Push/pop/call/jmp through every register, sete into every byte register, ret.
#[rustfmt::skip]
#[test]
fn stack_calls_jumps_and_setcc_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("push rax", |a| { a.push(RAX); }, &[0x50]),
        ("pop rax", |a| { a.pop(RAX); }, &[0x58]),
        ("call rax", |a| { a.call_reg(RAX); }, &[0xff, 0xd0]),
        ("jmp rax", |a| { a.jmp_reg(RAX); }, &[0xff, 0xe0]),
        ("sete al", |a| { a.sete(RAX); }, &[0x0f, 0x94, 0xc0]),
        ("push rcx", |a| { a.push(RCX); }, &[0x51]),
        ("pop rcx", |a| { a.pop(RCX); }, &[0x59]),
        ("call rcx", |a| { a.call_reg(RCX); }, &[0xff, 0xd1]),
        ("jmp rcx", |a| { a.jmp_reg(RCX); }, &[0xff, 0xe1]),
        ("sete cl", |a| { a.sete(RCX); }, &[0x0f, 0x94, 0xc1]),
        ("push rdx", |a| { a.push(RDX); }, &[0x52]),
        ("pop rdx", |a| { a.pop(RDX); }, &[0x5a]),
        ("call rdx", |a| { a.call_reg(RDX); }, &[0xff, 0xd2]),
        ("jmp rdx", |a| { a.jmp_reg(RDX); }, &[0xff, 0xe2]),
        ("sete dl", |a| { a.sete(RDX); }, &[0x0f, 0x94, 0xc2]),
        ("push rbx", |a| { a.push(RBX); }, &[0x53]),
        ("pop rbx", |a| { a.pop(RBX); }, &[0x5b]),
        ("call rbx", |a| { a.call_reg(RBX); }, &[0xff, 0xd3]),
        ("jmp rbx", |a| { a.jmp_reg(RBX); }, &[0xff, 0xe3]),
        ("sete bl", |a| { a.sete(RBX); }, &[0x0f, 0x94, 0xc3]),
        ("push rsp", |a| { a.push(RSP); }, &[0x54]),
        ("pop rsp", |a| { a.pop(RSP); }, &[0x5c]),
        ("call rsp", |a| { a.call_reg(RSP); }, &[0xff, 0xd4]),
        ("jmp rsp", |a| { a.jmp_reg(RSP); }, &[0xff, 0xe4]),
        ("sete spl", |a| { a.sete(RSP); }, &[0x40, 0x0f, 0x94, 0xc4]),
        ("push rbp", |a| { a.push(RBP); }, &[0x55]),
        ("pop rbp", |a| { a.pop(RBP); }, &[0x5d]),
        ("call rbp", |a| { a.call_reg(RBP); }, &[0xff, 0xd5]),
        ("jmp rbp", |a| { a.jmp_reg(RBP); }, &[0xff, 0xe5]),
        ("sete bpl", |a| { a.sete(RBP); }, &[0x40, 0x0f, 0x94, 0xc5]),
        ("push rsi", |a| { a.push(RSI); }, &[0x56]),
        ("pop rsi", |a| { a.pop(RSI); }, &[0x5e]),
        ("call rsi", |a| { a.call_reg(RSI); }, &[0xff, 0xd6]),
        ("jmp rsi", |a| { a.jmp_reg(RSI); }, &[0xff, 0xe6]),
        ("sete sil", |a| { a.sete(RSI); }, &[0x40, 0x0f, 0x94, 0xc6]),
        ("push rdi", |a| { a.push(RDI); }, &[0x57]),
        ("pop rdi", |a| { a.pop(RDI); }, &[0x5f]),
        ("call rdi", |a| { a.call_reg(RDI); }, &[0xff, 0xd7]),
        ("jmp rdi", |a| { a.jmp_reg(RDI); }, &[0xff, 0xe7]),
        ("sete dil", |a| { a.sete(RDI); }, &[0x40, 0x0f, 0x94, 0xc7]),
        ("push r8", |a| { a.push(R8); }, &[0x41, 0x50]),
        ("pop r8", |a| { a.pop(R8); }, &[0x41, 0x58]),
        ("call r8", |a| { a.call_reg(R8); }, &[0x41, 0xff, 0xd0]),
        ("jmp r8", |a| { a.jmp_reg(R8); }, &[0x41, 0xff, 0xe0]),
        ("sete r8b", |a| { a.sete(R8); }, &[0x41, 0x0f, 0x94, 0xc0]),
        ("push r9", |a| { a.push(R9); }, &[0x41, 0x51]),
        ("pop r9", |a| { a.pop(R9); }, &[0x41, 0x59]),
        ("call r9", |a| { a.call_reg(R9); }, &[0x41, 0xff, 0xd1]),
        ("jmp r9", |a| { a.jmp_reg(R9); }, &[0x41, 0xff, 0xe1]),
        ("sete r9b", |a| { a.sete(R9); }, &[0x41, 0x0f, 0x94, 0xc1]),
        ("push r10", |a| { a.push(R10); }, &[0x41, 0x52]),
        ("pop r10", |a| { a.pop(R10); }, &[0x41, 0x5a]),
        ("call r10", |a| { a.call_reg(R10); }, &[0x41, 0xff, 0xd2]),
        ("jmp r10", |a| { a.jmp_reg(R10); }, &[0x41, 0xff, 0xe2]),
        ("sete r10b", |a| { a.sete(R10); }, &[0x41, 0x0f, 0x94, 0xc2]),
        ("push r11", |a| { a.push(R11); }, &[0x41, 0x53]),
        ("pop r11", |a| { a.pop(R11); }, &[0x41, 0x5b]),
        ("call r11", |a| { a.call_reg(R11); }, &[0x41, 0xff, 0xd3]),
        ("jmp r11", |a| { a.jmp_reg(R11); }, &[0x41, 0xff, 0xe3]),
        ("sete r11b", |a| { a.sete(R11); }, &[0x41, 0x0f, 0x94, 0xc3]),
        ("push r12", |a| { a.push(R12); }, &[0x41, 0x54]),
        ("pop r12", |a| { a.pop(R12); }, &[0x41, 0x5c]),
        ("call r12", |a| { a.call_reg(R12); }, &[0x41, 0xff, 0xd4]),
        ("jmp r12", |a| { a.jmp_reg(R12); }, &[0x41, 0xff, 0xe4]),
        ("sete r12b", |a| { a.sete(R12); }, &[0x41, 0x0f, 0x94, 0xc4]),
        ("push r13", |a| { a.push(R13); }, &[0x41, 0x55]),
        ("pop r13", |a| { a.pop(R13); }, &[0x41, 0x5d]),
        ("call r13", |a| { a.call_reg(R13); }, &[0x41, 0xff, 0xd5]),
        ("jmp r13", |a| { a.jmp_reg(R13); }, &[0x41, 0xff, 0xe5]),
        ("sete r13b", |a| { a.sete(R13); }, &[0x41, 0x0f, 0x94, 0xc5]),
        ("push r14", |a| { a.push(R14); }, &[0x41, 0x56]),
        ("pop r14", |a| { a.pop(R14); }, &[0x41, 0x5e]),
        ("call r14", |a| { a.call_reg(R14); }, &[0x41, 0xff, 0xd6]),
        ("jmp r14", |a| { a.jmp_reg(R14); }, &[0x41, 0xff, 0xe6]),
        ("sete r14b", |a| { a.sete(R14); }, &[0x41, 0x0f, 0x94, 0xc6]),
        ("push r15", |a| { a.push(R15); }, &[0x41, 0x57]),
        ("pop r15", |a| { a.pop(R15); }, &[0x41, 0x5f]),
        ("call r15", |a| { a.call_reg(R15); }, &[0x41, 0xff, 0xd7]),
        ("jmp r15", |a| { a.jmp_reg(R15); }, &[0x41, 0xff, 0xe7]),
        ("sete r15b", |a| { a.sete(R15); }, &[0x41, 0x0f, 0x94, 0xc7]),
        ("ret", |a| { a.ret(); }, &[0xc3]),
    ];
    check_encodings(cases);
}

    #[test]
    fn forward_conditional_jump_is_patched_with_the_distance_in_bytes() {
        // {disp32} jne 1f; nop; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x85, 0x02, 0x00, 0x00, 0x00, 0x90, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(NE, l);
                nop(a);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn backward_unconditional_jump_is_patched_with_a_negative_distance() {
        // 1: nop; {disp32} jmp 1b
        let expected: &[u8] = &[0x90, 0xe9, 0xfa, 0xff, 0xff, 0xff];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.bind(l);
                nop(a);
                a.jmp(l);
            }),
            expected
        );
    }

    #[test]
    fn forward_unconditional_jump_to_the_next_instruction_has_distance_zero() {
        // {disp32} jmp 1f; 1: ret
        let expected: &[u8] = &[0xe9, 0x00, 0x00, 0x00, 0x00, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jmp(l);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn two_conditional_jumps_can_share_a_label() {
        // {disp32} je 1f; {disp32} jb 1f; 1:
        let expected: &[u8] = &[
            0x0f, 0x84, 0x06, 0x00, 0x00, 0x00, 0x0f, 0x82, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(E, l);
                a.jcc(2, l);
                a.bind(l);
            }),
            expected
        );
    }

    #[test]
    fn a_jump_to_itself_has_a_distance_of_minus_the_instruction_length() {
        // 1: {disp32} jne 1b
        let expected: &[u8] = &[0x0f, 0x85, 0xfa, 0xff, 0xff, 0xff];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.bind(l);
                a.jcc(NE, l);
            }),
            expected
        );
    }

    #[test]
    fn a_label_bound_at_the_end_of_the_code_is_the_end_of_the_code() {
        // {disp32} jmp 1f; nop; 1:
        let expected: &[u8] = &[0xe9, 0x01, 0x00, 0x00, 0x00, 0x90];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jmp(l);
                nop(a);
                a.bind(l);
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_o_uses_condition_code_0() {
        // {disp32} jo 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x80, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(0, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_no_uses_condition_code_1() {
        // {disp32} jno 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x81, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(1, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_b_uses_condition_code_2() {
        // {disp32} jb 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x82, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(2, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_ae_uses_condition_code_3() {
        // {disp32} jae 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x83, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(3, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_e_uses_condition_code_4() {
        // {disp32} je 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x84, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(4, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_ne_uses_condition_code_5() {
        // {disp32} jne 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x85, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(5, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_be_uses_condition_code_6() {
        // {disp32} jbe 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x86, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(6, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_a_uses_condition_code_7() {
        // {disp32} ja 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x87, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(7, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_s_uses_condition_code_8() {
        // {disp32} js 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x88, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(8, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_ns_uses_condition_code_9() {
        // {disp32} jns 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x89, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(9, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_p_uses_condition_code_10() {
        // {disp32} jp 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8a, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(10, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_np_uses_condition_code_11() {
        // {disp32} jnp 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(11, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_l_uses_condition_code_12() {
        // {disp32} jl 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8c, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(12, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_ge_uses_condition_code_13() {
        // {disp32} jge 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8d, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(13, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_le_uses_condition_code_14() {
        // {disp32} jle 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8e, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(14, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn conditional_jump_g_uses_condition_code_15() {
        // {disp32} jg 1f; nop; 1: ret
        let expected: &[u8] = &[0x0f, 0x8f, 0x01, 0x00, 0x00, 0x00, 0x90, 0xc3];
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jcc(15, l);
                nop(a);
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn a_forward_jump_over_a_thousand_bytes() {
        // {disp32} jmp 1f; 1000 x nop; 1: ret
        let expected: Vec<u8> = [
            &[0xe9, 0xe8, 0x03, 0x00, 0x00][..],
            &[NOP_BYTE; 1000][..],
            &[0xc3][..],
        ]
        .concat();
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.jmp(l);
                for _ in 0..1000 {
                    nop(a);
                }
                a.bind(l);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn a_backward_conditional_jump_over_a_thousand_bytes() {
        // 1: 1000 x nop; {disp32} jae 1b
        let expected: Vec<u8> = [
            &[][..],
            &[NOP_BYTE; 1000][..],
            &[0x0f, 0x83, 0x12, 0xfc, 0xff, 0xff][..],
        ]
        .concat();
        assert_eq!(
            assemble(|a| {
                let l = a.label();
                a.bind(l);
                for _ in 0..1000 {
                    nop(a);
                }
                a.jcc(3, l);
            }),
            expected
        );
    }

    #[test]
    fn condition_code_constants_match_the_jcc_encoding() {
        // The JIT passes guest condition codes (the low nibble of 0x70..=0x7f) through unchanged.
        assert_eq!(E, 4);
        assert_eq!(NE, 5);
    }

    #[test]
    fn group_one_operation_numbers_are_the_opcode_extension_digits() {
        assert_eq!((ADD, OR, AND, SUB, XOR, CMP), (0, 1, 4, 5, 6, 7));
        assert_eq!((SHL, SHR), (4, 5));
    }

    #[cfg(windows)]
    #[test]
    fn win64_argument_registers() {
        assert_eq!(ARGS, [RCX, RDX, R8, R9]);
    }

    #[cfg(not(windows))]
    #[test]
    fn system_v_argument_registers() {
        assert_eq!(ARGS, [RDI, RSI, RDX, RCX]);
    }

    #[test]
    fn register_numbers_are_the_architectural_ones() {
        let all = [
            RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
        ];
        for (number, register) in all.iter().enumerate() {
            assert_eq!(*register as usize, number);
        }
    }

    #[test]
    fn here_counts_bytes_from_the_base() {
        let mut a = Asm::new(BASE_ADDRESS);
        assert_eq!(a.here(), BASE_ADDRESS);
        a.ret();
        nop(&mut a);
        assert_eq!(a.here(), BASE_ADDRESS + 2);
    }

    #[test]
    fn absolute_jump_encodes_the_distance_from_the_end_of_the_instruction() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.jmp_abs(BASE_ADDRESS + 0x105);
        assert_eq!(a.code, [0xe9, 0x00, 0x01, 0x00, 0x00]);
        let mut a = Asm::new(BASE_ADDRESS);
        a.jmp_abs(BASE_ADDRESS);
        assert_eq!(
            a.code,
            [0xe9, 0xfb, 0xff, 0xff, 0xff],
            "jumping to itself is -5"
        );
        let mut a = Asm::new(BASE_ADDRESS);
        nop(&mut a);
        a.jmp_abs(BASE_ADDRESS - 0x1000);
        assert_eq!(&a.code[1..], &[0xe9, 0xfa, 0xef, 0xff, 0xff]);
    }

    #[test]
    fn absolute_jump_reaches_the_full_rel32_range() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.jmp_abs(BASE_ADDRESS + 5 + i32::MAX as u64);
        assert_eq!(&a.code[1..], &i32::MAX.to_le_bytes());
        let mut a = Asm::new(BASE_ADDRESS);
        a.jmp_abs((BASE_ADDRESS as i64 + 5 + i32::MIN as i64) as u64);
        assert_eq!(&a.code[1..], &i32::MIN.to_le_bytes());
    }

    #[test]
    #[should_panic(expected = "jump out of range")]
    fn absolute_jump_beyond_rel32_forward_panics() {
        Asm::new(BASE_ADDRESS).jmp_abs(BASE_ADDRESS + 6 + i32::MAX as u64);
    }

    #[test]
    #[should_panic(expected = "jump out of range")]
    fn absolute_jump_beyond_rel32_backward_panics() {
        Asm::new(BASE_ADDRESS + (1 << 33)).jmp_abs(BASE_ADDRESS);
    }

    #[test]
    #[should_panic(expected = "unbound label")]
    fn finishing_with_a_jump_to_an_unbound_label_panics() {
        let mut a = Asm::new(BASE_ADDRESS);
        let never_bound = a.label();
        a.jmp(never_bound);
        a.finish();
    }

    #[test]
    fn finish_is_idempotent() {
        let mut a = Asm::new(BASE_ADDRESS);
        let target = a.label();
        a.jcc(E, target);
        a.bind(target);
        a.finish();
        let patched = a.code.clone();
        a.finish();
        assert_eq!(a.code, patched);
    }

    #[test]
    fn an_unreferenced_label_needs_no_binding() {
        let mut a = Asm::new(BASE_ADDRESS);
        let _unused = a.label();
        a.ret();
        a.finish();
        assert_eq!(a.code, [0xc3]);
    }

    #[test]
    fn rex_is_omitted_for_wide_forms_without_extension_bits_and_kept_for_byte_registers() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.mov_rr(RAX, RCX, false);
        assert_eq!(
            a.code,
            [0x89, 0xc8],
            "no REX for 32-bit moves between low registers"
        );
        let mut a = Asm::new(BASE_ADDRESS);
        a.mov_rr(RAX, RCX, true);
        assert_eq!(a.code, [0x48, 0x89, 0xc8]);
        let mut a = Asm::new(BASE_ADDRESS);
        a.sete(RAX);
        assert_eq!(
            a.code,
            [0x40, 0x0f, 0x94, 0xc0],
            "byte operations always carry a REX prefix"
        );
        let mut a = Asm::new(BASE_ADDRESS);
        a.sete(RSI);
        assert_eq!(
            a.code,
            [0x40, 0x0f, 0x94, 0xc6],
            "sil needs an empty REX, or it would be dh"
        );
    }

    #[cfg(debug_assertions)]
    mod register_range_checks {
        use super::*;

        #[test]
        #[should_panic]
        fn mov_imm_rejects_a_register_number_of_16() {
            Asm::new(BASE_ADDRESS).mov_imm(16, 0);
        }

        #[test]
        #[should_panic]
        fn alu_ri_rejects_a_register_number_of_16() {
            Asm::new(BASE_ADDRESS).alu_ri(ADD, 16, 1, true);
        }

        #[test]
        #[should_panic]
        fn mov_rr_rejects_an_out_of_range_source() {
            Asm::new(BASE_ADDRESS).mov_rr(RAX, 17, true);
        }

        #[test]
        #[should_panic]
        fn memory_operands_reject_an_out_of_range_base() {
            Asm::new(BASE_ADDRESS).ld(RAX, mem(32, 0));
        }

        #[test]
        #[should_panic]
        fn rsp_cannot_be_an_index() {
            let _ = mem_idx(RAX, RSP, 0, 0);
        }

        #[test]
        #[should_panic]
        fn scales_above_eight_are_rejected() {
            let _ = mem_idx(RAX, RCX, 4, 0);
        }

        #[test]
        fn r12_can_be_an_index_because_rex_x_distinguishes_it() {
            let mut a = Asm::new(BASE_ADDRESS);
            a.ld(RAX, mem_idx(RAX, R12, 0, 0));
            assert_eq!(a.code, [0x4a, 0x8b, 0x04, 0x20]);
        }
    }

    mod sized_forms {
        //! Encoder tests for the sized, unary and conditional forms the block generator uses at every
        //! operand size. Expected bytes come from `hack/asm.sh` (the assembly is in each case). Nested
        //! in `encoding_tests`, which has the case runner and `run`.

        use super::*;

        /// Condition code 6 (BE); the generator never names it, so the encoder has no constant.
        const CONDITION_BE: u8 = 6;
        const CONDITION_L: u8 = 0xc;
        const CONDITION_G: u8 = 0xf;

        #[rustfmt::skip]
#[test]
fn sized_unary_and_conditional_forms_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add al, cl", |a| a.alu_rr_sized(ADD, RAX, RCX, 1), &[0x00, 0xc8]),
        ("add ax, cx", |a| a.alu_rr_sized(ADD, RAX, RCX, 2), &[0x66, 0x01, 0xc8]),
        ("add eax, ecx", |a| a.alu_rr_sized(ADD, RAX, RCX, 4), &[0x01, 0xc8]),
        ("add rax, rcx", |a| a.alu_rr_sized(ADD, RAX, RCX, 8), &[0x48, 0x01, 0xc8]),
        ("sub sil, dil", |a| a.alu_rr_sized(SUB, RSI, RDI, 1), &[0x40, 0x28, 0xfe]),
        ("xor r9b, r10b", |a| a.alu_rr_sized(XOR, R9, R10, 1), &[0x45, 0x30, 0xd1]),
        ("cmp r8w, r9w", |a| a.alu_rr_sized(CMP, R8, R9, 2), &[0x66, 0x45, 0x39, 0xc8]),
        ("and r9d, r10d", |a| a.alu_rr_sized(AND, R9, R10, 4), &[0x45, 0x21, 0xd1]),
        ("or r11, r12", |a| a.alu_rr_sized(OR, R11, R12, 8), &[0x4d, 0x09, 0xe3]),
        ("test ecx, 4", |a| a.test_ri(RCX, 4, false), &[0xf7, 0xc1, 0x04, 0x00, 0x00, 0x00]),
        ("test rcx, 4", |a| a.test_ri(RCX, 4, true), &[0x48, 0xf7, 0xc1, 0x04, 0x00, 0x00, 0x00]),
        ("test r9d, 0x12345678", |a| a.test_ri(R9, 0x1234_5678, false), &[0x41, 0xf7, 0xc1, 0x78, 0x56, 0x34, 0x12]),
        ("not al", |a| a.not_r(RAX, 1), &[0xf6, 0xd0]),
        ("not cx", |a| a.not_r(RCX, 2), &[0x66, 0xf7, 0xd1]),
        ("not ecx", |a| a.not_r(RCX, 4), &[0xf7, 0xd1]),
        ("not r9", |a| a.not_r(R9, 8), &[0x49, 0xf7, 0xd1]),
        ("neg sil", |a| a.neg_r(RSI, 1), &[0x40, 0xf6, 0xde]),
        ("neg ax", |a| a.neg_r(RAX, 2), &[0x66, 0xf7, 0xd8]),
        ("neg r10d", |a| a.neg_r(R10, 4), &[0x41, 0xf7, 0xda]),
        ("neg rdx", |a| a.neg_r(RDX, 8), &[0x48, 0xf7, 0xda]),
        ("inc al", |a| a.inc_r(RAX, 1), &[0xfe, 0xc0]),
        ("inc cx", |a| a.inc_r(RCX, 2), &[0x66, 0xff, 0xc1]),
        ("inc r9d", |a| a.inc_r(R9, 4), &[0x41, 0xff, 0xc1]),
        ("inc rax", |a| a.inc_r(RAX, 8), &[0x48, 0xff, 0xc0]),
        ("dec dil", |a| a.dec_r(RDI, 1), &[0x40, 0xfe, 0xcf]),
        ("dec r8w", |a| a.dec_r(R8, 2), &[0x66, 0x41, 0xff, 0xc8]),
        ("dec ecx", |a| a.dec_r(RCX, 4), &[0xff, 0xc9]),
        ("dec r10", |a| a.dec_r(R10, 8), &[0x49, 0xff, 0xca]),
        ("bt ecx, 0", |a| a.bt_ri(RCX, 0, false), &[0x0f, 0xba, 0xe1, 0x00]),
        ("bt r9, 5", |a| a.bt_ri(R9, 5, true), &[0x49, 0x0f, 0xba, 0xe1, 0x05]),
        ("imul eax, ecx", |a| a.imul_rr(RAX, RCX, false), &[0x0f, 0xaf, 0xc1]),
        ("imul rax, r9", |a| a.imul_rr(RAX, R9, true), &[0x49, 0x0f, 0xaf, 0xc1]),
        ("imul r10d, edx", |a| a.imul_rr(R10, RDX, false), &[0x44, 0x0f, 0xaf, 0xd2]),
        ("shld rax, rcx, 5", |a| a.double_shift_ri(true, RAX, RCX, 5, true), &[0x48, 0x0f, 0xa4, 0xc8, 0x05]),
        ("shrd rax, rcx, 5", |a| a.double_shift_ri(false, RAX, RCX, 5, true), &[0x48, 0x0f, 0xac, 0xc8, 0x05]),
        ("shld eax, ecx, 5", |a| a.double_shift_ri(true, RAX, RCX, 5, false), &[0x0f, 0xa4, 0xc8, 0x05]),
        ("shld r9, r10, 5", |a| a.double_shift_ri(true, R9, R10, 5, true), &[0x4d, 0x0f, 0xa4, 0xd1, 0x05]),
        ("movq xmm2, r9", |a| a.movq_xmm_gpr(2, R9), &[0x66, 0x49, 0x0f, 0x6e, 0xd1]),
        ("movq xmm0, rax", |a| a.movq_xmm_gpr(0, RAX), &[0x66, 0x48, 0x0f, 0x6e, 0xc0]),
        ("punpcklqdq xmm0, xmm2", |a| a.xmm_op(OPCODE_PUNPCKLQDQ, 0, 2), &[0x66, 0x0f, 0x6c, 0xc2]),
        ("pandn xmm2, xmm3", |a| a.xmm_op(OPCODE_PANDN, 2, 3), &[0x66, 0x0f, 0xdf, 0xd3]),
        ("por xmm0, xmm2", |a| a.xmm_op(OPCODE_POR, 0, 2), &[0x66, 0x0f, 0xeb, 0xc2]),
        ("mul rcx", |a| a.mul_r(RCX, true), &[0x48, 0xf7, 0xe1]),
        ("imul ecx", |a| a.imul_r(RCX, false), &[0xf7, 0xe9]),
        ("mul r9", |a| a.mul_r(R9, true), &[0x49, 0xf7, 0xe1]),
        ("imul r9d", |a| a.imul_r(R9, false), &[0x41, 0xf7, 0xe9]),
        ("bsf rax, r9", |a| a.bsf_rr(RAX, R9, true), &[0x49, 0x0f, 0xbc, 0xc1]),
        ("bsf r10d, edx", |a| a.bsf_rr(R10, RDX, false), &[0x44, 0x0f, 0xbc, 0xd2]),
        ("bsr rax, r9", |a| a.bsr_rr(RAX, R9, true), &[0x49, 0x0f, 0xbd, 0xc1]),
        ("bsr r10d, edx", |a| a.bsr_rr(R10, RDX, false), &[0x44, 0x0f, 0xbd, 0xd2]),
        ("cmove eax, ecx", |a| a.cmov_rr(E, RAX, RCX, false), &[0x0f, 0x44, 0xc1]),
        ("cmovne rax, r9", |a| a.cmov_rr(NE, RAX, R9, true), &[0x49, 0x0f, 0x45, 0xc1]),
        ("cmovl r10d, edx", |a| a.cmov_rr(CONDITION_L, R10, RDX, false), &[0x44, 0x0f, 0x4c, 0xd2]),
        ("cmovg r8, r15", |a| a.cmov_rr(CONDITION_G, R8, R15, true), &[0x4d, 0x0f, 0x4f, 0xc7]),
        ("shl eax, cl", |a| a.shift_cl(SHL, RAX, false), &[0xd3, 0xe0]),
        ("shr rdx, cl", |a| a.shift_cl(SHR, RDX, true), &[0x48, 0xd3, 0xea]),
        ("sar r9d, cl", |a| a.shift_cl(SAR, R9, false), &[0x41, 0xd3, 0xf9]),
        ("sar r10, cl", |a| a.shift_cl(SAR, R10, true), &[0x49, 0xd3, 0xfa]),
        ("sar eax, 31", |a| a.shift_ri(SAR, RAX, 31, false), &[0xc1, 0xf8, 0x1f]),
        ("sar rdx, 63", |a| a.shift_ri(SAR, RDX, 63, true), &[0x48, 0xc1, 0xfa, 0x3f]),
        ("seto r8b", |a| a.setcc(O, R8), &[0x41, 0x0f, 0x90, 0xc0]),
        ("setb sil", |a| a.setcc(B, RSI), &[0x40, 0x0f, 0x92, 0xc6]),
        ("setae al", |a| a.setcc(AE, RAX), &[0x0f, 0x93, 0xc0]),
        ("setne r10b", |a| a.setcc(NE, R10), &[0x41, 0x0f, 0x95, 0xc2]),
        ("setbe dil", |a| a.setcc(CONDITION_BE, RDI), &[0x40, 0x0f, 0x96, 0xc7]),
        ("cdq", |a| a.cdq(false), &[0x99]),
        ("cqo", |a| a.cdq(true), &[0x48, 0x99]),
    ];
    check_encodings(cases);
}

        #[test]
        fn sized_operations_compute_at_their_width_and_set_that_widths_flags() {
            // a byte add that overflows leaves the upper bits of rax alone
            assert_eq!(
                run(|a| a.alu_rr_sized(ADD, RAX, RCX, 1), 0x1234_5680, 0x80),
                0x1234_5600
            );
            assert_eq!(
                run(|a| a.alu_rr_sized(ADD, RAX, RCX, 2), 0x1234_ffff, 2),
                0x1234_0001
            );
            // and sets CF for the byte, not for the register
            assert_eq!(
                run(
                    |a| {
                        a.alu_rr_sized(ADD, RAX, RCX, 1);
                        a.setcc(B, RDX);
                        a.ext_rr(RAX, RDX, 1, false);
                    },
                    0x80,
                    0x80
                ),
                1
            );
            assert_eq!(run(|a| a.neg_r(RAX, 4), 1, 0), 0xffff_ffff);
            assert_eq!(run(|a| a.not_r(RAX, 1), 0xff00, 0), 0xffff);
            assert_eq!(run(|a| a.inc_r(RAX, 1), 0xff, 0), 0);
            assert_eq!(run(|a| a.dec_r(RAX, 2), 0x1_0000, 0), 0x1_ffff);
            assert_eq!(
                run(|a| a.imul_rr(RAX, RCX, false), 0xffff_fffe, 3),
                0xffff_fffa
            );
            assert_eq!(
                run(|a| a.shift_cl(SAR, RAX, true), !0u64 << 8, 4),
                !0u64 << 4
            );
        }

        #[test]
        fn inc_keeps_the_carry_that_bt_put_in() {
            let carry_after = |value: u64| {
                run(
                    |a| {
                        a.bt_ri(RCX, 0, false);
                        a.inc_r(RAX, 4);
                        a.setcc(B, RDX);
                        a.ext_rr(RAX, RDX, 1, false);
                    },
                    0,
                    value,
                )
            };
            assert_eq!(carry_after(1), 1);
            assert_eq!(carry_after(2), 0);
        }

        #[test]
        fn cmov_moves_only_when_the_condition_holds_and_zero_extends_either_way() {
            let cmov = |condition: u8| {
                run(
                    |a| {
                        a.bt_ri(RCX, 0, false);
                        a.cmov_rr(condition, RAX, RCX, false);
                    },
                    0xffff_ffff_0000_0005,
                    1,
                )
            };
            assert_eq!(cmov(B), 1, "moved");
            assert_eq!(cmov(AE), 5, "not moved, upper half still cleared");
        }

        #[test]
        fn cdq_and_cqo_sign_extend_into_rdx() {
            let rdx_after = |w: bool, value: u64| {
                run(
                    |a| {
                        a.mov_rr(RDX, RCX, true);
                        a.cdq(w);
                        a.mov_rr(RAX, RDX, true);
                    },
                    value,
                    u64::MAX,
                )
            };
            assert_eq!(rdx_after(false, 0x8000_0000), 0xffff_ffff);
            assert_eq!(
                rdx_after(false, 0x7fff_ffff),
                0,
                "the upper half is cleared"
            );
            assert_eq!(rdx_after(true, 1 << 63), u64::MAX);
            assert_eq!(rdx_after(true, 1 << 62), 0);
        }

        #[test]
        fn cdq_does_not_touch_the_flags() {
            for w in [false, true] {
                let zero_flag = run(
                    |a| {
                        a.alu_rr(CMP, RAX, RAX, true);
                        a.cdq(w);
                        a.mov_imm(RCX, 0);
                        a.setcc(E, RCX);
                        a.mov_rr(RAX, RCX, true);
                    },
                    5,
                    0,
                );
                assert_eq!(zero_flag, 1);
            }
        }
    }

    fn run(build: impl FnOnce(&mut Asm), arg0: u64, arg1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let memory = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(memory.base());
        a.mov_rr(RAX, ARGS[0], true);
        a.mov_rr(RCX, ARGS[1], true);
        build(&mut a);
        a.ret();
        a.finish();
        memory.write_bytes(0, &a.code);
        let function: extern "C" fn(u64, u64) -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        function(arg0, arg1)
    }

    #[test]
    fn an_absolute_jump_lands_on_its_target() {
        let _guard = crate::jit::serialize_jit_tests();
        const STUB_OFFSET: usize = 0x40;
        let memory = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(memory.base());
        a.mov_imm(RAX, 1);
        a.jmp_abs(memory.base() + STUB_OFFSET as u64);
        a.mov_imm(RAX, 2);
        a.ret();
        a.code.resize(STUB_OFFSET, 0xcc);
        a.mov_imm(RAX, 3);
        a.ret();
        memory.write_bytes(0, &a.code);
        let function: extern "C" fn() -> u64 =
            unsafe { std::mem::transmute(memory.base() as usize) };
        assert_eq!(function(), 3);
    }

    #[test]
    fn conditional_jumps_follow_the_host_flags_for_every_condition_code() {
        // cmp rax, rcx; mov eax, 0; j<cc> taken; ret; taken: mov eax, 1
        for (x, y) in [
            (1u64, 2u64),
            (2, 1),
            (5, 5),
            (0, u64::MAX),
            (u64::MAX, 0),
            (1 << 63, 1),
            (i64::MAX as u64, u64::MAX),
        ] {
            let (xi, yi) = (x as i64, y as i64);
            let difference = x.wrapping_sub(y);
            let parity_even = (difference as u8).count_ones().is_multiple_of(2);
            let overflow = xi.overflowing_sub(yi).1;
            let sign = (difference as i64) < 0;
            let zero = x == y;
            let carry = x < y;
            let holds = [
                overflow,
                !overflow,
                carry,
                !carry,
                zero,
                !zero,
                carry || zero,
                !(carry || zero),
                sign,
                !sign,
                parity_even,
                !parity_even,
                sign != overflow,
                sign == overflow,
                zero || sign != overflow,
                !zero && sign == overflow,
            ];
            for (condition, expected) in holds.iter().enumerate() {
                let result = run(
                    |a| {
                        let taken = a.label();
                        a.alu_rr(CMP, RAX, RCX, true);
                        a.mov_imm(RAX, 0);
                        a.jcc(condition as u8, taken);
                        a.ret();
                        a.bind(taken);
                        a.mov_imm(RAX, 1);
                    },
                    x,
                    y,
                );
                assert_eq!(
                    result, *expected as u64,
                    "cmp {x:#x}, {y:#x}: condition {condition}"
                );
            }
        }
    }

    #[test]
    fn byte_registers_sil_dil_spl_and_bpl_are_addressable_for_stores() {
        let mut buffer = [0u8; 8];
        let base = buffer.as_mut_ptr() as u64;
        for (register, slot) in [(RSI, 0usize), (RDI, 1), (RBP, 2)] {
            run(
                |a| {
                    a.push(register);
                    a.mov_imm(register, 0x1111_2222_3333_44a0 | slot as u64);
                    a.st(mem(RAX, slot as i32), register, 1);
                    a.pop(register);
                },
                base,
                0,
            );
        }
        assert_eq!(&buffer[..3], &[0xa0, 0xa1, 0xa2]);
    }

    #[test]
    fn indexed_addressing_with_every_scale_reaches_the_right_element() {
        let buffer: [u64; 32] = std::array::from_fn(|index| 0x100 + index as u64);
        let base = buffer.as_ptr() as u64;
        for (scale, bytes_per_step) in [(0u8, 1u64), (1, 2), (2, 4), (3, 8)] {
            let value = run(
                |a| {
                    a.mov_imm(RDX, 8);
                    a.ld(RAX, mem_idx(RAX, RDX, scale, 0));
                },
                base,
                0,
            );
            let offset = 8 * bytes_per_step as usize;
            let expected = u64::from_le_bytes(
                unsafe { std::slice::from_raw_parts((base as usize + offset) as *const u8, 8) }
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(value, expected, "scale {}", 1 << scale);
        }
    }

    #[test]
    fn r12_and_r13_bases_load_from_the_right_address() {
        let buffer = [0x11u64, 0x22, 0x33, 0x44];
        let base = buffer.as_ptr() as u64;
        for register in [R12, R13] {
            for (displacement, expected) in [(0, 0x11), (8, 0x22), (16, 0x33), (24, 0x44)] {
                let value = run(
                    |a| {
                        a.push(register);
                        a.mov_rr(register, RAX, true);
                        a.ld(RAX, mem(register, displacement));
                        a.pop(register);
                    },
                    base,
                    0,
                );
                assert_eq!(value, expected, "base {register} disp {displacement}");
            }
        }
    }
}
