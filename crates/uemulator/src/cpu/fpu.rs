use super::decode::*;
use super::exception::Exception;
use super::exec::Loc;
use super::flags::*;
use super::sys::{CR0_EM, CR0_TS};
use super::*;
#[cfg(freestanding)]
use crate::math::{Float64Ext, FloatExt};

// x87 registers are kept as raw 80-bit values (`Cpu::st`, 10 bytes little-endian: 64-bit
// significand with an explicit integer bit, then 15-bit exponent and sign), but every
// arithmetic operation converts to f64, computes there and converts back. Results therefore
// have 53-bit precision regardless of FCW.PC, and rounding control only affects frndint and
// the fist family. Only moves (fld/fst m80, fxch, fld st(i), fxsave) preserve all 64 bits.
// No x87 exception is ever raised: FSW exception flags are never set and FCW masks are ignored.

/// Number of x87 registers, and the mask that wraps a stack position into them.
const STACK_DEPTH: usize = 8;
const STACK_MASK: usize = STACK_DEPTH - 1;

const EXT_EXPONENT_BIAS: i32 = 16383;
/// Exponent field value for infinity and NaN.
const EXT_EXPONENT_SPECIAL: u16 = 0x7fff;
const EXT_SIGN: u16 = 0x8000;
/// Index of the byte holding the sign and the high exponent bits, and the sign bit within it.
const EXT_SIGN_BYTE: usize = 9;
const EXT_SIGN_BYTE_BIT: u8 = 0x80;
/// Bit 63 of the significand: the explicit integer bit, set in every normalised value.
const EXT_INTEGER_BIT: u64 = 1 << 63;
/// Distance from the f64 fraction (52 bits) up to the extended fraction (63 bits).
const F64_FRACTION_TO_EXT_SHIFT: u32 = 11;
const F64_EXPONENT_BIAS: i32 = 1023;
const F64_EXPONENT_SPECIAL: i32 = 0x7ff;
const F64_FRACTION_BITS: u32 = 52;

/// FCW layout: exception masks in bits 0-5, precision control in 8-9, rounding control in 10-11.
/// The power-on/FINIT value is `FCW_RESET` in cpu/mod.rs.
const FCW_EXCEPTION_MASKS: u16 = 0x3f;
const FCW_ROUNDING_CONTROL_SHIFT: u32 = 10;
const ROUNDING_CONTROL_MASK: u16 = 3;
const ROUND_NEAREST_EVEN: u16 = 0;
const ROUND_DOWN: u16 = 1;
const ROUND_UP: u16 = 2;
const ROUND_TOWARD_ZERO: u16 = 3;

/// FSW: TOP in bits 11-13, condition codes C0 (8), C1 (9), C2 (10), C3 (14).
const FSW_TOP_SHIFT: u32 = 11;
const FSW_TOP_MASK: u16 = 7 << FSW_TOP_SHIFT;
const FSW_C0: u16 = 1 << 8;
const FSW_C1: u16 = 1 << 9;
const FSW_C2: u16 = 1 << 10;
const FSW_C3: u16 = 1 << 14;
/// C1 is excluded: fxam and fprem set it themselves and other compares leave it alone.
const FSW_CONDITION_MASK: u16 = FSW_C0 | FSW_C2 | FSW_C3;
/// fnclex keeps TOP and C0-C3; it clears the exception flags (0-7) and the busy bit (15).
const FSW_KEPT_BY_FNCLEX: u16 = 0x7f00;

/// Two-bit tag word values, as stored by fnstenv.
const TAG_VALID: u16 = 0;
const TAG_ZERO: u16 = 1;
const TAG_SPECIAL: u16 = 2;
const TAG_EMPTY: u16 = 3;

/// 32-bit protected-mode FNSTENV/FLDENV image: FCW, FSW, FTW as 32-bit words, then the
/// FIP/FCS/FDP/FDS pointer words, which are not tracked and always written as zero.
/// FNSAVE/FRSTOR append the eight 10-byte registers in stack order right after it. The 16-bit
/// formats selected by a 66 prefix or real mode are not implemented.
/// The environment image is seven words (control, status, tag, then FIP, FCS/opcode, FDP, FDS).
const ENV_FSW_INDEX: u64 = 1;
const ENV_FTW_INDEX: u64 = 2;
const ENV_POINTER_FIRST_INDEX: u64 = 3;
const ENV_WORDS: u64 = 7;
const EXT_REG_SIZE: u64 = 10;

/// FXSAVE image (SDM layout, 512 bytes). FOP/FIP/FDP at 6..24 are not written.
pub(super) const FXSAVE_AREA_SIZE: u64 = 512;
/// The area must be 16-byte aligned or the instruction is #GP(0).
pub(super) const FXSAVE_ALIGNMENT: u64 = 16;
pub(super) const FXSAVE_FSW_OFFSET: u64 = 2;
/// Abridged tag word: one bit per physical register, 1 = not empty.
pub(super) const FXSAVE_FTW_OFFSET: u64 = 4;
pub(super) const FXSAVE_MXCSR_OFFSET: u64 = 24;
pub(super) const FXSAVE_MXCSR_MASK_OFFSET: u64 = 28;
pub(super) const FXSAVE_ST_OFFSET: u64 = 32;
pub(super) const FXSAVE_ST_STRIDE: u64 = 16;
pub(super) const FXSAVE_XMM_OFFSET: u64 = 160;
pub(super) const FXSAVE_XMM_STRIDE: u64 = 16;

/// MXCSR.MM, misaligned exception mask (AMD misaligned SSE mode, leaf 0x8000_0001 ECX bit 7):
/// legacy SSE memory operands no longer need 16-byte alignment, except those of the explicitly
/// aligned moves (`sse.rs`).
pub(crate) const MXCSR_MISALIGNED_MASK: u32 = 1 << 17;
/// MXCSR bits that may be set by ldmxcsr/fxrstor; writing any other is #GP. Also reported as
/// MXCSR_MASK in the FXSAVE image, which is how Linux learns that MM may be set.
pub(crate) const MXCSR_VALID_BITS: u32 = 0xffff | MXCSR_MISALIGNED_MASK;

pub(crate) fn ext_to_f64(b: [u8; 10]) -> f64 {
    let significand = u64::from_le_bytes(b[..8].try_into().unwrap());
    let sign_exponent = u16::from_le_bytes([b[8], b[9]]);
    let neg = sign_exponent & EXT_SIGN != 0;
    let exponent = (sign_exponent & !EXT_SIGN) as i32;
    let v = if exponent == EXT_EXPONENT_SPECIAL as i32 {
        // integer bit is ignored when telling infinity from NaN
        if significand << 1 == 0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else if significand == 0 {
        0.0
    } else {
        // value = significand * 2^(exponent - bias - 63); split the power so each half stays
        // representable when the full scale factor would overflow or underflow f64
        let scale = exponent - EXT_EXPONENT_BIAS - 63;
        let (first, second) = (scale / 2, scale - scale / 2);
        (significand as f64) * 2f64.powi(first) * 2f64.powi(second)
    };
    if neg { -v } else { v }
}

pub(crate) fn f64_to_ext(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let neg = bits >> 63 != 0;
    let exponent = ((bits >> F64_FRACTION_BITS) & F64_EXPONENT_SPECIAL as u64) as i32;
    let fraction = bits & ((1 << F64_FRACTION_BITS) - 1);
    let (sign_exponent, significand): (u16, u64) = if exponent == F64_EXPONENT_SPECIAL {
        (
            EXT_EXPONENT_SPECIAL,
            EXT_INTEGER_BIT | (fraction << F64_FRACTION_TO_EXT_SHIFT),
        )
    } else if exponent == 0 {
        if fraction == 0 {
            (0, 0)
        } else {
            // f64 subnormals are normalised: extended has the range to hold them
            let shift = fraction.leading_zeros();
            (
                ((EXT_EXPONENT_BIAS - F64_EXPONENT_BIAS + 1
                    - (shift as i32 - F64_FRACTION_TO_EXT_SHIFT as i32)) as u16),
                fraction << shift,
            )
        }
    } else {
        (
            (exponent - F64_EXPONENT_BIAS + EXT_EXPONENT_BIAS) as u16,
            EXT_INTEGER_BIT | (fraction << F64_FRACTION_TO_EXT_SHIFT),
        )
    };
    let mut out = [0u8; 10];
    out[..8].copy_from_slice(&significand.to_le_bytes());
    out[8..].copy_from_slice(&(sign_exponent | if neg { EXT_SIGN } else { 0 }).to_le_bytes());
    out
}

/// What an 80-bit register holds, as FXAM reports it and as the tag word records it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExtClass {
    /// Unnormals, pseudo-infinities and pseudo-NaNs: formats the 387 and later reject.
    Unsupported,
    Nan,
    Normal,
    Infinity,
    Zero,
    /// Exponent 0 with a nonzero significand. A pseudo-denormal (integer bit set) is included.
    Denormal,
}

/// Classifies the raw bits, not an f64 conversion: an extended denormal underflows f64 to zero
/// and would otherwise look like one.
fn classify(raw: [u8; 10]) -> ExtClass {
    let significand = u64::from_le_bytes(raw[..8].try_into().unwrap());
    let exponent = u16::from_le_bytes([raw[8], raw[9]]) & !EXT_SIGN;
    let integer_bit_set = significand & EXT_INTEGER_BIT != 0;
    match exponent {
        0 if significand == 0 => ExtClass::Zero,
        0 => ExtClass::Denormal,
        EXT_EXPONENT_SPECIAL if !integer_bit_set => ExtClass::Unsupported,
        EXT_EXPONENT_SPECIAL if significand << 1 == 0 => ExtClass::Infinity,
        EXT_EXPONENT_SPECIAL => ExtClass::Nan,
        _ if integer_bit_set => ExtClass::Normal,
        _ => ExtClass::Unsupported,
    }
}

/// Rounds to an integral value according to an FCW.RC field value (bits above 1:0 are ignored).
fn round_to_integral(v: f64, rounding_control: u16) -> f64 {
    match rounding_control & ROUNDING_CONTROL_MASK {
        ROUND_NEAREST_EVEN => v.round_ties_even(),
        ROUND_DOWN => v.floor(),
        ROUND_UP => v.ceil(),
        _ => v.trunc(),
    }
}

/// Width in bytes of the words of the fnstenv/fnsave image: 2 with a 16-bit operand size.
fn env_word(i: &Insn) -> u8 {
    if i.opsize == 2 { 2 } else { 4 }
}

impl Cpu {
    /// #NM when CR0.EM or CR0.TS is set (the kernel's lazy FPU switching relies on the TS case).
    pub fn fpu_check(&self) -> ExecResult<()> {
        if self.cr0 & (CR0_EM | CR0_TS) != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    fn top(&self) -> usize {
        ((self.fsw & FSW_TOP_MASK) >> FSW_TOP_SHIFT) as usize
    }

    fn set_top(&mut self, top: usize) {
        self.fsw = (self.fsw & !FSW_TOP_MASK) | ((top as u16 & STACK_MASK as u16) << FSW_TOP_SHIFT);
    }

    /// Raw contents of ST(i); `i` is relative to TOP.
    fn st_raw(&self, i: usize) -> [u8; 10] {
        self.st[(self.top() + i) & STACK_MASK]
    }

    fn st_val(&self, i: usize) -> f64 {
        ext_to_f64(self.st_raw(i))
    }

    /// Writes ST(i) and marks the physical register non-empty. `ftw` is indexed by physical
    /// register (like the FXSAVE abridged tag), not by stack position.
    fn set_st_raw(&mut self, i: usize, v: [u8; 10]) {
        let physical = (self.top() + i) & STACK_MASK;
        self.st[physical] = v;
        self.ftw |= 1 << physical;
    }

    fn set_st(&mut self, i: usize, v: f64) {
        self.set_st_raw(i, f64_to_ext(v));
    }

    fn fpush_raw(&mut self, v: [u8; 10]) {
        let new_top = self.top().wrapping_sub(1) & STACK_MASK;
        self.set_top(new_top);
        self.st[new_top] = v;
        self.ftw |= 1 << new_top;
    }

    fn fpush(&mut self, v: f64) {
        self.fpush_raw(f64_to_ext(v));
    }

    fn fpop(&mut self) {
        let top = self.top();
        self.ftw &= !(1 << top);
        self.set_top((top + 1) & STACK_MASK);
    }

    fn set_condition_codes(&mut self, c0: bool, c2: bool, c3: bool) {
        self.fsw = (self.fsw & !FSW_CONDITION_MASK)
            | (c0 as u16 * FSW_C0)
            | (c2 as u16 * FSW_C2)
            | (c3 as u16 * FSW_C3);
    }

    /// fcom/fucom: C3,C2,C0 = 111 unordered, 000 greater, 001 less, 100 equal. fucom is not
    /// distinguished from fcom (no invalid-operation signalling either way).
    fn fcom_val(&mut self, a: f64, b: f64) {
        if a.is_nan() || b.is_nan() {
            self.set_condition_codes(true, true, true);
        } else if a > b {
            self.set_condition_codes(false, false, false);
        } else if a < b {
            self.set_condition_codes(true, false, false);
        } else {
            self.set_condition_codes(false, false, true);
        }
    }

    /// fcomi/fucomi: same outcomes as fcom but reported in ZF, PF, CF.
    fn fcomi_val(&mut self, a: f64, b: f64) {
        let flags = if a.is_nan() || b.is_nan() {
            ZF | PF | CF
        } else if a > b {
            0
        } else if a < b {
            CF
        } else {
            ZF
        };
        let current = self.eflags_full();
        self.set_eflags_arith((current & !(ZF | PF | CF)) | flags);
    }

    /// Arithmetic selected by ModRM.reg of the D8/DC/DE/DA groups: 0 add, 1 mul, 4 sub, 5 subr,
    /// 6 div, 7 divr (2 and 3 are the compares, handled by callers).
    fn arith(reg: u8, a: f64, b: f64) -> f64 {
        match reg & 7 {
            0 => a + b,
            1 => a * b,
            4 => a - b,
            5 => b - a,
            6 => a / b,
            _ => b / a,
        }
    }

    /// Converts to a signed integer of `bits` width using FCW.RC. Out-of-range and NaN give the
    /// "integer indefinite" value (only the sign bit set) without signalling #IA.
    fn to_int(&self, v: f64, bits: u32) -> i64 {
        self.to_int_rounded(v, bits, self.fcw >> FCW_ROUNDING_CONTROL_SHIFT)
    }

    /// `to_int` with an explicit rounding control instead of FCW.RC (fisttp always truncates).
    fn to_int_rounded(&self, v: f64, bits: u32, rounding_control: u16) -> i64 {
        let rounded = round_to_integral(v, rounding_control);
        let limit = (1u128 << (bits - 1)) as f64;
        if rounded.is_nan() || rounded >= limit || rounded < -limit {
            i64::MIN >> (64 - bits)
        } else {
            rounded as i64
        }
    }

    /// Opcodes d8-df. Register forms (mod == 3) are selected by opcode and the whole ModRM
    /// byte or ModRM.reg; memory forms by opcode and ModRM.reg.
    pub fn x87(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.fpu_check()?;
        let op = (i.op & 0xff) as u8;
        let r = i.reg & 7;
        let sti = (i.rm & 7) as usize;
        if i.md == 3 {
            match op {
                0xd8 => {
                    // st0 = st0 op st(i); /2 fcom, /3 fcomp
                    if r == 2 || r == 3 {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 3 {
                            self.fpop();
                        }
                    } else {
                        let v = Self::arith(r, self.st_val(0), self.st_val(sti));
                        self.set_st(0, v);
                    }
                }
                0xd9 => match i.modrm {
                    // fld st(i)
                    0xc0..=0xc7 => {
                        let v = self.st_raw(sti);
                        self.fpush_raw(v);
                    }
                    // fxch
                    0xc8..=0xcf => {
                        let (a, b) = (self.st_raw(0), self.st_raw(sti));
                        self.set_st_raw(0, b);
                        self.set_st_raw(sti, a);
                    }
                    // fnop
                    0xd0 => {}
                    // fchs
                    0xe0 => {
                        let mut v = self.st_raw(0);
                        v[EXT_SIGN_BYTE] ^= EXT_SIGN_BYTE_BIT;
                        self.set_st_raw(0, v);
                    }
                    // fabs
                    0xe1 => {
                        let mut v = self.st_raw(0);
                        v[EXT_SIGN_BYTE] &= !EXT_SIGN_BYTE_BIT;
                        self.set_st_raw(0, v);
                    }
                    // ftst
                    0xe4 => {
                        let a = self.st_val(0);
                        self.fcom_val(a, 0.0);
                    }
                    // fxam: C3,C2,C0 = 101 empty, 001 NaN, 010 normal, 011 infinity, 100 zero,
                    // 110 denormal, 000 unsupported format. C1 is the sign of ST(0), even when
                    // the register is empty.
                    0xe5 => {
                        let raw = self.st_raw(0);
                        let neg = raw[EXT_SIGN_BYTE] & EXT_SIGN_BYTE_BIT != 0;
                        let empty = self.ftw & (1 << self.top()) == 0;
                        let (c3, c2, c0) = if empty {
                            (true, false, true)
                        } else {
                            match classify(raw) {
                                ExtClass::Unsupported => (false, false, false),
                                ExtClass::Nan => (false, false, true),
                                ExtClass::Normal => (false, true, false),
                                ExtClass::Infinity => (false, true, true),
                                ExtClass::Zero => (true, false, false),
                                ExtClass::Denormal => (true, true, false),
                            }
                        };
                        self.set_condition_codes(c0, c2, c3);
                        self.fsw = (self.fsw & !FSW_C1) | (neg as u16 * FSW_C1);
                    }
                    // Constants are rounded to f64, not to 64 significand bits.
                    // fld1, fldl2t, fldl2e, fldpi, fldlg2, fldln2, fldz
                    0xe8 => self.fpush(1.0),
                    0xe9 => self.fpush(core::f64::consts::LOG2_10),
                    0xea => self.fpush(core::f64::consts::LOG2_E),
                    0xeb => self.fpush(core::f64::consts::PI),
                    0xec => self.fpush(core::f64::consts::LOG10_2),
                    0xed => self.fpush(core::f64::consts::LN_2),
                    0xee => self.fpush(0.0),
                    // f2xm1
                    0xf0 => {
                        let x = self.st_val(0);
                        self.set_st(0, (x * core::f64::consts::LN_2).exp_m1());
                    }
                    // fyl2x
                    0xf1 => {
                        let v = self.st_val(1) * self.st_val(0).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fptan: C2 = 0 reports the operand as in range
                    0xf2 => {
                        let x = self.st_val(0).tan();
                        self.set_st(0, x);
                        self.fpush(1.0);
                        self.set_condition_codes(false, false, false);
                    }
                    // fpatan
                    0xf3 => {
                        let v = self.st_val(1).atan2(self.st_val(0));
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fprem1 (quotient rounded to nearest) and fprem (truncated). The remainder is
                    // always computed in one step, so C2 stays clear; C0,C3,C1 = quotient bits 2,1,0.
                    0xf5 | 0xf8 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        let q = if i.modrm == 0xf8 {
                            (a / b).trunc()
                        } else {
                            (a / b).round_ties_even()
                        };
                        self.set_st(0, a - q * b);
                        let qi = q.abs() as u64;
                        self.set_condition_codes(qi & 4 != 0, false, qi & 2 != 0);
                        self.fsw = (self.fsw & !FSW_C1) | ((qi & 1) as u16 * FSW_C1);
                    }
                    // fdecstp, fincstp: move TOP only, tags are left alone
                    0xf6 => {
                        let top = self.top();
                        self.set_top(top.wrapping_sub(1) & STACK_MASK);
                    }
                    0xf7 => {
                        let top = self.top();
                        self.set_top((top + 1) & STACK_MASK);
                    }
                    // fyl2xp1
                    0xf9 => {
                        let v = self.st_val(1) * (1.0 + self.st_val(0)).log2();
                        self.set_st(1, v);
                        self.fpop();
                    }
                    // fsqrt
                    0xfa => {
                        let v = self.st_val(0).sqrt();
                        self.set_st(0, v);
                    }
                    // fsincos
                    0xfb => {
                        let x = self.st_val(0);
                        self.set_st(0, x.sin());
                        self.fpush(x.cos());
                        self.set_condition_codes(false, false, false);
                    }
                    // frndint
                    0xfc => {
                        let v = round_to_integral(
                            self.st_val(0),
                            self.fcw >> FCW_ROUNDING_CONTROL_SHIFT,
                        );
                        self.set_st(0, v);
                    }
                    // fscale
                    0xfd => {
                        let v = self.st_val(0) * 2f64.powf(self.st_val(1).trunc());
                        self.set_st(0, v);
                    }
                    // fsin, fcos
                    0xfe => {
                        let v = self.st_val(0).sin();
                        self.set_st(0, v);
                        self.set_condition_codes(false, false, false);
                    }
                    0xff => {
                        let v = self.st_val(0).cos();
                        self.set_st(0, v);
                        self.set_condition_codes(false, false, false);
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                0xda => match i.modrm {
                    // fcmovb, fcmove, fcmovbe, fcmovu
                    0xc0..=0xdf => {
                        let flags = self.eflags_full();
                        let condition = match r {
                            0 => flags & CF != 0,
                            1 => flags & ZF != 0,
                            2 => flags & (CF | ZF) != 0,
                            _ => flags & PF != 0,
                        };
                        if condition {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    // fucompp
                    0xe9 => {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                        self.fpop();
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                0xdb => match i.modrm {
                    // fcmovnb, fcmovne, fcmovnbe, fcmovnu
                    0xc0..=0xdf => {
                        let flags = self.eflags_full();
                        let condition = match r {
                            0 => flags & CF == 0,
                            1 => flags & ZF == 0,
                            2 => flags & (CF | ZF) == 0,
                            _ => flags & PF == 0,
                        };
                        if condition {
                            let v = self.st_raw(sti);
                            self.set_st_raw(0, v);
                        }
                    }
                    // fnclex
                    0xe2 => self.fsw &= FSW_KEPT_BY_FNCLEX,
                    // fninit
                    0xe3 => {
                        self.fcw = FCW_RESET;
                        self.fsw = 0;
                        self.ftw = 0;
                    }
                    // fucomi (e8-ef), fcomi (f0-f7)
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                // st(i) = st(i) op st0. The sub/subr and div/divr encodings are swapped relative
                // to d8, which is why /4 computes b - a and /5 computes a - b below.
                0xdc => {
                    if r == 2 || r == 3 {
                        return Err(self.unimplemented_opcode(i));
                    }
                    let (a, b) = (self.st_val(sti), self.st_val(0));
                    let v = match r {
                        4 => b - a,
                        5 => a - b,
                        6 => b / a,
                        7 => a / b,
                        _ => Self::arith(r, a, b),
                    };
                    self.set_st(sti, v);
                }
                0xdd => match r {
                    // ffree st(i)
                    0 => self.ftw &= !(1 << ((self.top() + sti) & STACK_MASK)),
                    // fst/fstp st(i)
                    2 | 3 => {
                        let v = self.st_raw(0);
                        self.set_st_raw(sti, v);
                        if r == 3 {
                            self.fpop();
                        }
                    }
                    // fucom/fucomp
                    4 | 5 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcom_val(a, b);
                        if r == 5 {
                            self.fpop();
                        }
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
                // faddp/fmulp/fsubrp/fsubp/fdivrp/fdivp (same swapped encodings as dc), fcompp
                0xde => {
                    if i.modrm == 0xd9 {
                        let (a, b) = (self.st_val(0), self.st_val(1));
                        self.fcom_val(a, b);
                        self.fpop();
                    } else if r == 2 || r == 3 {
                        return Err(self.unimplemented_opcode(i));
                    } else {
                        let (a, b) = (self.st_val(sti), self.st_val(0));
                        let v = match r {
                            4 => b - a,
                            5 => a - b,
                            6 => b / a,
                            7 => a / b,
                            _ => Self::arith(r, a, b),
                        };
                        self.set_st(sti, v);
                    }
                    self.fpop();
                }
                // df
                _ => match i.modrm {
                    // fnstsw ax
                    0xe0 => {
                        let v = self.fsw as u64;
                        self.write_gpr(0, 2, true, v);
                    }
                    // fucomip (e8-ef), fcomip (f0-f7)
                    0xe8..=0xf7 => {
                        let (a, b) = (self.st_val(0), self.st_val(sti));
                        self.fcomi_val(a, b);
                        self.fpop();
                    }
                    // ffreep
                    0xc0..=0xc7 => {
                        self.ftw &= !(1 << ((self.top() + sti) & STACK_MASK));
                        self.fpop();
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                },
            }
            self.rip = next;
            return Ok(());
        }
        let a = match self.modrm_location(i, next)? {
            Loc::Mem(a) => a,
            _ => unreachable!(),
        };
        match op {
            0xd8 | 0xdc => {
                let v = if op == 0xd8 {
                    f32::from_bits(self.read(a, 4)? as u32) as f64
                } else {
                    f64::from_bits(self.read(a, 8)?)
                };
                self.arith_mem(r, v);
            }
            0xda | 0xde => {
                let v = if op == 0xda {
                    self.read(a, 4)? as u32 as i32 as f64
                } else {
                    self.read(a, 2)? as u16 as i16 as f64
                };
                self.arith_mem(r, v);
            }
            0xd9 => match r {
                0 => {
                    let v = f32::from_bits(self.read(a, 4)? as u32) as f64;
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0) as f32;
                    self.write(a, 4, v.to_bits() as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                4 => self.fldenv(a, env_word(i))?,
                // fldcw: no exception unmasking is modelled
                5 => self.fcw = self.read(a, 2)? as u16,
                6 => self.fnstenv(a, env_word(i))?,
                7 => self.write(a, 2, self.fcw as u64)?,
                // /1 is not an instruction
                _ => return Err(Exception::invalid_opcode()),
            },
            0xdb => match r {
                0 => {
                    let v = self.read(a, 4)? as u32 as i32 as f64;
                    self.fpush(v);
                }
                // fisttp m32int (SSE3): truncates whatever FCW.RC says, and always pops
                1 => {
                    let v = self.to_int_rounded(self.st_val(0), 32, ROUND_TOWARD_ZERO) as u32;
                    self.write(a, 4, v as u64)?;
                    self.fpop();
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 32) as u32;
                    self.write(a, 4, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                // fld m80 / fstp m80
                5 => {
                    let mut raw = [0u8; 10];
                    for (offset, byte) in raw.iter_mut().enumerate() {
                        *byte = self.read(a + offset as u64, 1)? as u8;
                    }
                    self.fpush_raw(raw);
                }
                7 => {
                    let raw = self.st_raw(0);
                    for (offset, byte) in raw.iter().enumerate() {
                        self.write(a + offset as u64, 1, *byte as u64)?;
                    }
                    self.fpop();
                }
                _ => return Err(self.unimplemented_opcode(i)),
            },
            0xdd => match r {
                0 => {
                    let v = f64::from_bits(self.read(a, 8)?);
                    self.fpush(v);
                }
                2 | 3 => {
                    let v = self.st_val(0);
                    self.write(a, 8, v.to_bits())?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                // fisttp m64int
                1 => {
                    let v = self.to_int_rounded(self.st_val(0), 64, ROUND_TOWARD_ZERO);
                    self.write(a, 8, v as u64)?;
                    self.fpop();
                }
                4 => self.frstor(a, env_word(i))?,
                6 => self.fnsave(a, env_word(i))?,
                7 => self.write(a, 2, self.fsw as u64)?,
                _ => return Err(self.unimplemented_opcode(i)),
            },
            // df: fild/fist/fistp/fisttp m16, fild/fistp m64 (bcd forms are not implemented)
            _ => match r {
                0 => {
                    let v = self.read(a, 2)? as u16 as i16 as f64;
                    self.fpush(v);
                }
                // fisttp m16int
                1 => {
                    let v = self.to_int_rounded(self.st_val(0), 16, ROUND_TOWARD_ZERO) as u16;
                    self.write(a, 2, v as u64)?;
                    self.fpop();
                }
                2 | 3 => {
                    let v = self.to_int(self.st_val(0), 16) as u16;
                    self.write(a, 2, v as u64)?;
                    if r == 3 {
                        self.fpop();
                    }
                }
                5 => {
                    let v = self.read(a, 8)? as i64 as f64;
                    self.fpush(v);
                }
                7 => {
                    let v = self.to_int(self.st_val(0), 64);
                    self.write(a, 8, v as u64)?;
                    self.fpop();
                }
                _ => return Err(self.unimplemented_opcode(i)),
            },
        }
        self.rip = next;
        Ok(())
    }

    /// Memory-operand forms of the d8/da/dc/de groups: st0 = st0 op v, or compare with v.
    fn arith_mem(&mut self, reg: u8, v: f64) {
        let a = self.st_val(0);
        if reg == 2 || reg == 3 {
            self.fcom_val(a, v);
            if reg == 3 {
                self.fpop();
            }
        } else {
            let res = Self::arith(reg, a, v);
            self.set_st(0, res);
        }
    }

    /// Builds the two-bit-per-register tag word for fnstenv from `ftw` and the register
    /// contents. Tags are indexed by physical register.
    fn full_tag(&self) -> u16 {
        let mut tag_word = 0u16;
        for physical in 0..STACK_DEPTH {
            let tag = if self.ftw & (1 << physical) == 0 {
                TAG_EMPTY
            } else {
                match classify(self.st[physical]) {
                    ExtClass::Normal => TAG_VALID,
                    ExtClass::Zero => TAG_ZERO,
                    ExtClass::Nan
                    | ExtClass::Infinity
                    | ExtClass::Denormal
                    | ExtClass::Unsupported => TAG_SPECIAL,
                }
            };
            tag_word |= tag << (2 * physical);
        }
        tag_word
    }

    /// The environment image: control, status and tag words, then the four pointer words, each
    /// `word` bytes wide (4 normally, 2 for the 16-bit operand-size format). The pointers are
    /// always zero.
    fn fnstenv(&mut self, a: u64, word: u8) -> ExecResult<()> {
        let step = word as u64;
        self.write(a, word, self.fcw as u64)?;
        self.write(a + ENV_FSW_INDEX * step, word, self.fsw as u64)?;
        let tag_word = self.full_tag();
        self.write(a + ENV_FTW_INDEX * step, word, tag_word as u64)?;
        for index in ENV_POINTER_FIRST_INDEX..ENV_WORDS {
            self.write(a + index * step, word, 0)?;
        }
        // fnstenv masks all exceptions after saving the environment
        self.fcw |= FCW_EXCEPTION_MASKS;
        Ok(())
    }

    /// Only a 2-bit "empty or not" distinction survives into `ftw`.
    fn fldenv(&mut self, a: u64, word: u8) -> ExecResult<()> {
        let step = word as u64;
        self.fcw = self.read(a, 2)? as u16;
        self.fsw = self.read(a + ENV_FSW_INDEX * step, 2)? as u16;
        let tag_word = self.read(a + ENV_FTW_INDEX * step, 2)? as u16;
        self.ftw = (0..STACK_DEPTH)
            .filter(|physical| (tag_word >> (2 * physical)) & 3 != TAG_EMPTY)
            .fold(0, |mask, physical| mask | (1 << physical));
        Ok(())
    }

    fn fnsave(&mut self, a: u64, word: u8) -> ExecResult<()> {
        self.fnstenv(a, word)?;
        let registers = a + ENV_WORDS * word as u64;
        for stack_pos in 0..STACK_DEPTH {
            let raw = self.st_raw(stack_pos);
            for (offset, byte) in raw.iter().enumerate() {
                let at = registers + stack_pos as u64 * EXT_REG_SIZE + offset as u64;
                self.write(at, 1, *byte as u64)?;
            }
        }
        // fnsave reinitialises the FPU, same state as fninit
        self.fcw = FCW_RESET;
        self.fsw = 0;
        self.ftw = 0;
        Ok(())
    }

    fn frstor(&mut self, a: u64, word: u8) -> ExecResult<()> {
        self.fldenv(a, word)?;
        let registers = a + ENV_WORDS * word as u64;
        for stack_pos in 0..STACK_DEPTH {
            let mut raw = [0u8; 10];
            for (offset, byte) in raw.iter_mut().enumerate() {
                let at = registers + stack_pos as u64 * EXT_REG_SIZE + offset as u64;
                *byte = self.read(at, 1)? as u8;
            }
            let physical = (self.top() + stack_pos) & STACK_MASK;
            self.st[physical] = raw;
        }
        Ok(())
    }

    /// #GP(0) unless the 512-byte area is 16-byte aligned, checked on the linear address before
    /// anything is read or written.
    pub(super) fn check_fxsave_alignment(&self, a: u64) -> ExecResult<()> {
        if a & (FXSAVE_ALIGNMENT - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    pub fn fxsave(&mut self, a: u64) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 {
            return Err(Exception::invalid_opcode());
        }
        self.fpu_check()?;
        self.check_fxsave_alignment(a)?;
        let fast = self.fast_fxsave();
        self.save_legacy_region(a, true, !fast)?;
        if fast {
            // The region is probed whole, so these cannot fault part way.
            self.write(a + FXSAVE_MXCSR_OFFSET, 4, self.mxcsr as u64)?;
            self.write(a + FXSAVE_MXCSR_MASK_OFFSET, 4, MXCSR_VALID_BITS as u64)?;
        }
        Ok(())
    }

    /// EFER.FFXSR in 64-bit mode at CPL 0: fxsave and fxrstor leave XMM0-15 out of the image (MXCSR
    /// stays in), which is how a kernel saves time when it already knows the vector registers
    /// are not live (APM vol. 2 3.1.7). Outside that mode the bit changes nothing.
    fn fast_fxsave(&self) -> bool {
        self.efer & super::sys::EFER_FFXSR != 0
            && self.efer & EFER_LMA != 0
            && self.seg[SEG_CS].long()
            && self.cpl == 0
    }

    /// Writes the x87 part (control/status/tag words and the eight registers) and/or the SSE part
    /// (MXCSR, its mask and XMM0-15) of the 512-byte legacy image. Shared by fxsave and xsave,
    /// whose first 512 bytes are the same image; a component that is not saved leaves its bytes
    /// untouched, as xsave with a partial mask requires.
    pub(super) fn save_legacy_region(&mut self, a: u64, x87: bool, sse: bool) -> ExecResult<()> {
        // Probe both ends first so a page fault is raised before any byte is written.
        self.translate(a, mem::AccessKind::Write, false)?;
        self.translate(a + FXSAVE_AREA_SIZE - 1, mem::AccessKind::Write, false)?;
        if x87 {
            self.write(a, 2, self.fcw as u64)?;
            self.write(a + FXSAVE_FSW_OFFSET, 2, self.fsw as u64)?;
            self.write(a + FXSAVE_FTW_OFFSET, 1, self.ftw as u64 & 0xff)?;
            for stack_pos in 0..STACK_DEPTH {
                let raw = self.st_raw(stack_pos);
                for (offset, byte) in raw.iter().enumerate() {
                    let at =
                        a + FXSAVE_ST_OFFSET + stack_pos as u64 * FXSAVE_ST_STRIDE + offset as u64;
                    self.write(at, 1, *byte as u64)?;
                }
            }
        }
        if sse {
            self.write(a + FXSAVE_MXCSR_OFFSET, 4, self.mxcsr as u64)?;
            self.write(a + FXSAVE_MXCSR_MASK_OFFSET, 4, MXCSR_VALID_BITS as u64)?;
            for k in 0..self.xmm.len() {
                self.write128(
                    a + FXSAVE_XMM_OFFSET + k as u64 * FXSAVE_XMM_STRIDE,
                    self.xmm[k],
                )?;
            }
        }
        Ok(())
    }

    /// Reads and validates the whole image before changing any register, so a fault or a
    /// reserved MXCSR bit leaves the x87 and SSE state exactly as it was.
    pub fn fxrstor(&mut self, a: u64) -> ExecResult<()> {
        if self.cr0 & CR0_EM != 0 {
            return Err(Exception::invalid_opcode());
        }
        self.fpu_check()?;
        self.check_fxsave_alignment(a)?;
        let sse = if self.fast_fxsave() {
            LegacyRestore::MxcsrOnly
        } else {
            LegacyRestore::Load
        };
        self.restore_legacy_region(a, LegacyRestore::Load, sse)
    }

    /// The load half shared by fxrstor and xrstor. `Init` puts a component in its reset state
    /// (what xrstor does when the header's XSTATE_BV bit is clear) without reading its bytes.
    /// MXCSR is read whenever the SSE component is not skipped, as the SDM specifies for xrstor,
    /// and validated before anything is committed.
    pub(super) fn restore_legacy_region(
        &mut self,
        a: u64,
        x87: LegacyRestore,
        sse: LegacyRestore,
    ) -> ExecResult<()> {
        let mut fcw = self.fcw;
        let mut fsw = self.fsw;
        let mut ftw = self.ftw;
        let mut st = self.st;
        match x87 {
            // MxcsrOnly only describes the SSE component
            LegacyRestore::Skip | LegacyRestore::MxcsrOnly => {}
            LegacyRestore::Init => {
                fcw = FCW_RESET;
                fsw = 0;
                ftw = 0;
                st = [[0u8; 10]; STACK_DEPTH];
            }
            LegacyRestore::Load => {
                fcw = self.read(a, 2)? as u16;
                fsw = self.read(a + FXSAVE_FSW_OFFSET, 2)? as u16;
                ftw = self.read(a + FXSAVE_FTW_OFFSET, 1)? as u16;
                st = [[0u8; 10]; STACK_DEPTH];
                let top = ((fsw & FSW_TOP_MASK) >> FSW_TOP_SHIFT) as usize;
                // the image stores registers in stack order; place them by physical register
                for stack_pos in 0..STACK_DEPTH {
                    for (offset, byte) in st[(top + stack_pos) & STACK_MASK].iter_mut().enumerate()
                    {
                        let at = a
                            + FXSAVE_ST_OFFSET
                            + stack_pos as u64 * FXSAVE_ST_STRIDE
                            + offset as u64;
                        *byte = self.read(at, 1)? as u8;
                    }
                }
            }
        }
        let mut mxcsr = self.mxcsr;
        let mut xmm = self.xmm;
        if sse != LegacyRestore::Skip {
            mxcsr = self.read(a + FXSAVE_MXCSR_OFFSET, 4)? as u32;
            for (k, register) in xmm.iter_mut().enumerate() {
                *register = match sse {
                    LegacyRestore::Load => {
                        self.read128(a + FXSAVE_XMM_OFFSET + k as u64 * FXSAVE_XMM_STRIDE)?
                    }
                    LegacyRestore::MxcsrOnly => *register,
                    _ => 0,
                };
            }
            if mxcsr & !MXCSR_VALID_BITS != 0 {
                return Err(Exception::general_protection(0));
            }
        }
        self.st = st;
        self.xmm = xmm;
        self.fcw = fcw;
        self.fsw = fsw;
        self.ftw = ftw;
        self.mxcsr = mxcsr;
        Ok(())
    }
}

/// What xrstor (or fxrstor) does with one legacy-image component.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LegacyRestore {
    Skip,
    Init,
    Load,
    /// The SSE component of a fast fxrstor: MXCSR only, the XMM registers keep their values.
    MxcsrOnly,
}

#[cfg(test)]
mod fpu_tests {
    use super::*;
    use crate::cpu::exception::{
        VECTOR_DEVICE_NOT_AVAILABLE, VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE,
        VECTOR_PAGE_FAULT,
    };
    use crate::testkit::{CODE_BASE, HLT, Harness, RDI, RSI, RSP, SCRATCH, Stop};

    const RAX: usize = 0;
    const DATA: u64 = SCRATCH + 0x1000;
    const OUT: u64 = SCRATCH + 0x2000;
    /// A 2 MiB region of the boot identity map that tests unmap to provoke page faults.
    const UNMAPPED_REGION: u64 = 0xa0_0000;

    const C0: u16 = 1 << 8;
    const C1: u16 = 1 << 9;
    const C2: u16 = 1 << 10;
    const C3: u16 = 1 << 14;
    const FSW_EXCEPTION_FLAGS: u16 = 0x00ff;
    const RC_NEAREST: u16 = 0x037f;
    const RC_DOWN: u16 = 0x077f;
    const RC_UP: u16 = 0x0b7f;
    const RC_TRUNCATE: u16 = 0x0f7f;

    /// Raw 80-bit patterns, little-endian: 64-bit significand then sign and 15-bit exponent.
    const EXT_ZERO: [u8; 10] = [0; 10];
    const EXT_NEG_ZERO: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80];
    const EXT_ONE: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
    const EXT_NEG_TWO: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0x00, 0xc0];
    const EXT_THREE: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0xc0, 0x00, 0x40];
    /// pi with the full 64-bit significand 0xc90fdaa22168c235, exponent 0x4000.
    const EXT_PI_FULL: [u8; 10] = [0x35, 0xc2, 0x68, 0x21, 0xa2, 0xda, 0x0f, 0xc9, 0x00, 0x40];
    /// 1 + 2^-63: the lowest significand bit set, which no f64 can hold.
    const EXT_ONE_PLUS_ULP: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
    const EXT_INFINITY: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x7f];
    const EXT_NEG_INFINITY: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0xff];
    const EXT_QUIET_NAN: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0x7f];
    const EXT_QUIET_NAN_PAYLOAD: [u8; 10] = [0x23, 0x01, 0, 0, 0, 0, 0, 0xc0, 0xff, 0x7f];
    const EXT_SIGNALING_NAN: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x7f];
    /// Exponent 0, integer bit clear: a true denormal.
    const EXT_DENORMAL: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const EXT_NEG_DENORMAL: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0x80];
    /// Exponent 0 with the integer bit set: a pseudo-denormal.
    const EXT_PSEUDO_DENORMAL: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0, 0];
    /// Exponent 0x3fff with the integer bit clear: an unnormal.
    const EXT_UNNORMAL: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0, 0xff, 0x3f];
    /// Exponent 0x7fff with the integer bit clear: pseudo-infinity and pseudo-NaN.
    const EXT_PSEUDO_INFINITY: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0x7f];
    const EXT_PSEUDO_NAN: [u8; 10] = [1, 0, 0, 0, 0, 0, 0, 0, 0xff, 0x7f];

    // fld qword ptr [rsi]; fld qword ptr [rsi + 8]
    const LOAD_A_THEN_B: &[u8] = &[0xdd, 0x06, 0xdd, 0x46, 0x08];
    // fld tbyte ptr [rsi]
    const LOAD_RAW_FROM_DATA: &[u8] = &[0xdb, 0x2e];

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    fn device_not_available() -> Stop {
        fault(VECTOR_DEVICE_NOT_AVAILABLE, None)
    }

    fn invalid_opcode() -> Stop {
        fault(VECTOR_INVALID_OPCODE, None)
    }

    fn general_protection_zero() -> Stop {
        fault(VECTOR_GENERAL_PROTECTION, Some(0))
    }

    fn page_fault_not_present_read() -> Stop {
        fault(VECTOR_PAGE_FAULT, Some(0))
    }

    /// Concatenates instruction fragments into one snippet, ending in `hlt`.
    fn program(parts: &[&[u8]]) -> Vec<u8> {
        let mut bytes: Vec<u8> = parts.concat();
        bytes.push(HLT);
        bytes
    }

    /// Points rsi at the input area and rdi at the output area.
    fn init(harness: &mut Harness) {
        harness.cpu.gpr[RSI] = DATA;
        harness.cpu.gpr[RDI] = OUT;
    }

    fn put_f64s(harness: &mut Harness, values: &[f64]) {
        for (index, value) in values.iter().enumerate() {
            harness.write_u64(DATA + 8 * index as u64, value.to_bits());
        }
    }

    fn run_with(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
        let mut harness = Harness::new(code);
        init(&mut harness);
        setup(&mut harness);
        let stop = harness.run(1000);
        (harness, stop)
    }

    #[track_caller]
    fn run_ok(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
        let (harness, stop) = run_with(code, setup);
        assert_eq!(stop, Stop::Halted, "{}", harness.cpu.dump_regs());
        harness
    }

    /// Runs an op on top of two pushed inputs: ST(1) = `a`, ST(0) = `b`.
    #[track_caller]
    fn run_binary(op: &[u8], a: f64, b: f64) -> Harness {
        run_ok(&program(&[LOAD_A_THEN_B, op]), |h| put_f64s(h, &[a, b]))
    }

    /// Runs a snippet after placing a raw 80-bit value at `DATA`.
    #[track_caller]
    fn run_with_raw(raw: [u8; 10], op: &[u8]) -> Harness {
        run_ok(&program(&[LOAD_RAW_FROM_DATA, op]), |h| h.write(DATA, &raw))
    }

    /// Runs `code` whose first instruction must fault with `expected`, leaving rip on it.
    #[track_caller]
    fn assert_faults(code: &[u8], setup: impl FnOnce(&mut Harness), expected: Stop) -> Harness {
        let (harness, stop) = run_with(code, setup);
        assert_eq!(stop, expected, "{}", harness.cpu.dump_regs());
        assert_eq!(
            harness.cpu.rip, CODE_BASE,
            "a fault must leave rip on the instruction"
        );
        harness
    }

    /// Makes the 2 MiB region of the boot identity map containing `linear` not present.
    fn unmap_region(harness: &mut Harness, linear: u64) {
        const ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
        let read = |address: u64| harness.read_u64(address) & ADDRESS_MASK;
        let pml4 = harness.cpu.cr3 & ADDRESS_MASK;
        let pdpt = read(pml4 + ((linear >> 39) & 0x1ff) * 8);
        let page_directory = read(pdpt + ((linear >> 30) & 0x1ff) * 8);
        harness.write_u64(page_directory + ((linear >> 21) & 0x1ff) * 8, 0);
        harness.cpu.flush_tlb();
    }

    fn top(harness: &Harness) -> usize {
        ((harness.cpu.fsw >> 11) & 7) as usize
    }

    fn st_raw(harness: &Harness, index: usize) -> [u8; 10] {
        harness.cpu.st[(top(harness) + index) & 7]
    }

    fn st(harness: &Harness, index: usize) -> f64 {
        ext_to_f64(st_raw(harness, index))
    }

    fn is_empty(harness: &Harness, physical: usize) -> bool {
        harness.cpu.ftw & (1 << physical) == 0
    }

    fn condition_codes(harness: &Harness) -> u16 {
        harness.cpu.fsw & (C0 | C2 | C3)
    }

    fn out_f64(harness: &Harness) -> f64 {
        f64::from_bits(harness.read_u64(OUT))
    }

    fn set_control_word(harness: &mut Harness, value: u16) {
        harness.cpu.fcw = value;
    }

    #[track_caller]
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= expected.abs() * 1e-14 + 1e-300,
            "{actual:e} is not close to {expected:e}"
        );
    }

    mod conversions {
        use super::*;

        #[test]
        fn one_converts_to_the_canonical_extended_encoding() {
            assert_eq!(f64_to_ext(1.0), EXT_ONE);
            assert_eq!(ext_to_f64(EXT_ONE), 1.0);
        }

        #[test]
        fn small_integers_have_the_expected_exponents_and_significands() {
            assert_eq!(f64_to_ext(-2.0), EXT_NEG_TWO);
            assert_eq!(f64_to_ext(3.0), EXT_THREE);
            // 0.5: exponent 0x3ffe
            assert_eq!(f64_to_ext(0.5), [0, 0, 0, 0, 0, 0, 0, 0x80, 0xfe, 0x3f]);
            // 1024.0: exponent 0x3fff + 10 = 0x4009
            assert_eq!(f64_to_ext(1024.0), [0, 0, 0, 0, 0, 0, 0, 0x80, 0x09, 0x40]);
        }

        #[test]
        fn signed_zeros_keep_their_sign() {
            assert_eq!(f64_to_ext(0.0), EXT_ZERO);
            assert_eq!(f64_to_ext(-0.0), EXT_NEG_ZERO);
            assert!(ext_to_f64(EXT_NEG_ZERO).is_sign_negative());
            assert_eq!(ext_to_f64(EXT_NEG_ZERO), 0.0);
            assert!(ext_to_f64(EXT_ZERO).is_sign_positive());
        }

        #[test]
        fn infinities_use_the_all_ones_exponent_with_the_integer_bit() {
            assert_eq!(f64_to_ext(f64::INFINITY), EXT_INFINITY);
            assert_eq!(f64_to_ext(f64::NEG_INFINITY), EXT_NEG_INFINITY);
            assert_eq!(ext_to_f64(EXT_INFINITY), f64::INFINITY);
            assert_eq!(ext_to_f64(EXT_NEG_INFINITY), f64::NEG_INFINITY);
        }

        #[test]
        fn quiet_nans_convert_both_ways() {
            assert_eq!(
                f64_to_ext(f64::from_bits(0x7ff8_0000_0000_0000)),
                EXT_QUIET_NAN
            );
            assert!(ext_to_f64(EXT_QUIET_NAN).is_nan());
            assert!(ext_to_f64(EXT_QUIET_NAN_PAYLOAD).is_nan());
            assert!(ext_to_f64(EXT_SIGNALING_NAN).is_nan());
        }

        #[test]
        fn the_largest_f64_has_a_full_53_bit_significand() {
            let ext = f64_to_ext(f64::MAX);
            assert_eq!(&ext[..8], &0xffff_ffff_ffff_f800u64.to_le_bytes());
            assert_eq!(u16::from_le_bytes([ext[8], ext[9]]), 1023 + 16383);
        }

        #[test]
        fn f64_denormals_are_normalised_in_extended_precision() {
            let smallest = f64::from_bits(1);
            let ext = f64_to_ext(smallest);
            assert_eq!(
                u64::from_le_bytes(ext[..8].try_into().unwrap()),
                1 << 63,
                "integer bit set"
            );
            assert_eq!(u16::from_le_bytes([ext[8], ext[9]]), 16383 - 1074);
            assert_eq!(ext_to_f64(ext), smallest);
            let largest_denormal = f64::from_bits(0x000f_ffff_ffff_ffff);
            assert_eq!(ext_to_f64(f64_to_ext(largest_denormal)), largest_denormal);
        }

        #[test]
        fn every_f64_class_round_trips_exactly() {
            for value in [
                1.0,
                -1.0,
                0.1,
                -0.1,
                core::f64::consts::PI,
                1e-300,
                1e300,
                f64::MAX,
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                f64::from_bits(0x000f_ffff_ffff_ffff),
                -f64::from_bits(0x0008_0000_0000_0000),
                2f64.powi(-1022),
                2f64.powi(1023),
                123_456_789.123_456_79,
            ] {
                let back = ext_to_f64(f64_to_ext(value));
                assert_eq!(back.to_bits(), value.to_bits(), "{value:e}");
            }
        }

        #[test]
        fn extended_values_beyond_the_f64_range_saturate() {
            // 2^16000 has no f64 representation, nor does 2^-16000.
            let huge = [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7f, 0x7e];
            assert_eq!(ext_to_f64(huge), f64::INFINITY);
            let tiny = [0, 0, 0, 0, 0, 0, 0, 0x80, 0x7f, 0x01];
            assert_eq!(ext_to_f64(tiny), 0.0);
        }

        #[test]
        fn extra_extended_precision_rounds_to_nearest_even_when_narrowed() {
            // 1 + 2^-63 narrows to 1.0
            assert_eq!(ext_to_f64(EXT_ONE_PLUS_ULP), 1.0);
            // 1 + 2^-53 is exactly halfway between 1.0 and 1 + 2^-52: ties to even gives 1.0
            let halfway = [0, 0x04, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
            assert_eq!(ext_to_f64(halfway), 1.0);
            // 1 + 2^-53 + 2^-63 is above halfway and rounds up
            let above = [1, 0x04, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f];
            assert_eq!(ext_to_f64(above), 1.0 + f64::EPSILON);
        }

        #[test]
        fn round_to_integral_follows_the_rounding_control_field() {
            let table: [(f64, [f64; 4]); 8] = [
                (2.5, [2.0, 2.0, 3.0, 2.0]),
                (3.5, [4.0, 3.0, 4.0, 3.0]),
                (-2.5, [-2.0, -3.0, -2.0, -2.0]),
                (-3.5, [-4.0, -4.0, -3.0, -3.0]),
                (0.5, [0.0, 0.0, 1.0, 0.0]),
                (2.7, [3.0, 2.0, 3.0, 2.0]),
                (-2.7, [-3.0, -3.0, -2.0, -2.0]),
                (5.0, [5.0, 5.0, 5.0, 5.0]),
            ];
            for (value, expected) in table {
                for (rounding_control, want) in expected.into_iter().enumerate() {
                    assert_eq!(
                        round_to_integral(value, rounding_control as u16),
                        want,
                        "{value} rc={rounding_control}"
                    );
                }
            }
        }

        #[test]
        fn round_to_integral_ignores_bits_above_the_field() {
            assert_eq!(round_to_integral(2.5, 0b100), 2.0);
            assert_eq!(round_to_integral(2.5, 0b110), 3.0);
            assert_eq!(
                round_to_integral(2.5, 0xffff),
                2.0,
                "all ones is RC=3, truncate"
            );
        }

        #[test]
        fn round_to_integral_keeps_the_sign_of_a_zero_result() {
            assert!(round_to_integral(-0.25, 0).is_sign_negative());
            assert!(round_to_integral(-0.25, 3).is_sign_negative());
            assert!(round_to_integral(0.25, 0).is_sign_positive());
        }
    }

    mod classification {
        use super::*;

        #[test]
        fn classify_recognises_every_class() {
            let cases: [([u8; 10], ExtClass); 17] = [
                (EXT_ZERO, ExtClass::Zero),
                (EXT_NEG_ZERO, ExtClass::Zero),
                (EXT_ONE, ExtClass::Normal),
                (EXT_NEG_TWO, ExtClass::Normal),
                (EXT_PI_FULL, ExtClass::Normal),
                (EXT_ONE_PLUS_ULP, ExtClass::Normal),
                (EXT_INFINITY, ExtClass::Infinity),
                (EXT_NEG_INFINITY, ExtClass::Infinity),
                (EXT_QUIET_NAN, ExtClass::Nan),
                (EXT_QUIET_NAN_PAYLOAD, ExtClass::Nan),
                (EXT_SIGNALING_NAN, ExtClass::Nan),
                (EXT_DENORMAL, ExtClass::Denormal),
                (EXT_NEG_DENORMAL, ExtClass::Denormal),
                (EXT_PSEUDO_DENORMAL, ExtClass::Denormal),
                (EXT_UNNORMAL, ExtClass::Unsupported),
                (EXT_PSEUDO_INFINITY, ExtClass::Unsupported),
                (EXT_PSEUDO_NAN, ExtClass::Unsupported),
            ];
            for (raw, expected) in cases {
                assert_eq!(classify(raw), expected, "{raw:02x?}");
            }
        }

        #[test]
        fn the_largest_denormal_and_smallest_normal_are_told_apart() {
            // exponent 0 with all significand bits but the integer bit: still a denormal
            let largest_denormal = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0, 0];
            assert_eq!(classify(largest_denormal), ExtClass::Denormal);
            let smallest_normal = [0, 0, 0, 0, 0, 0, 0, 0x80, 1, 0];
            assert_eq!(classify(smallest_normal), ExtClass::Normal);
            let largest_normal = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x7f];
            assert_eq!(classify(largest_normal), ExtClass::Normal);
        }
    }

    mod loads_and_stores {
        use super::*;

        #[test]
        fn fld_m32_converts_to_extended() {
            // fld dword ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0x06]]), |h| {
                h.write(DATA, &1.5f32.to_le_bytes());
            });
            assert_eq!(st(&h, 0), 1.5);
            assert_eq!(st_raw(&h, 0), f64_to_ext(1.5));
            assert_eq!(top(&h), 7);
        }

        #[test]
        fn fld_m64_converts_to_extended() {
            // fld qword ptr [rsi]
            let h = run_ok(&program(&[&[0xdd, 0x06]]), |h| put_f64s(h, &[-6.25]));
            assert_eq!(st(&h, 0), -6.25);
        }

        #[test]
        fn fld_m32_and_m64_denormals_become_normal_extended_values() {
            // The smallest f32 denormal (2^-149) is a normal number in extended precision.
            // fld dword ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0x06]]), |h| {
                h.write(DATA, &1u32.to_le_bytes());
            });
            assert_eq!(classify(st_raw(&h, 0)), ExtClass::Normal);
            assert_eq!(st(&h, 0), f32::from_bits(1) as f64);
            // fld qword ptr [rsi]
            let h = run_ok(&program(&[&[0xdd, 0x06]]), |h| {
                put_f64s(h, &[f64::from_bits(1)])
            });
            assert_eq!(classify(st_raw(&h, 0)), ExtClass::Normal);
        }

        #[test]
        fn fld_m80_preserves_every_bit() {
            for raw in [
                EXT_PI_FULL,
                EXT_ONE_PLUS_ULP,
                EXT_NEG_ZERO,
                EXT_DENORMAL,
                EXT_PSEUDO_DENORMAL,
                EXT_UNNORMAL,
                EXT_QUIET_NAN_PAYLOAD,
                EXT_SIGNALING_NAN,
                EXT_PSEUDO_NAN,
                EXT_NEG_INFINITY,
            ] {
                // fstp tbyte ptr [rdi]
                let h = run_ok(&program(&[LOAD_RAW_FROM_DATA, &[0xdb, 0x3f]]), |h| {
                    h.write(DATA, &raw);
                    h.write(OUT, &[0xaa; 12]);
                });
                assert_eq!(h.read(OUT, 10), raw, "{raw:02x?}");
                assert_eq!(
                    h.read(OUT + 10, 2),
                    [0xaa, 0xaa],
                    "fstp m80 writes exactly 10 bytes"
                );
            }
        }

        #[test]
        fn fld_m80_leaves_the_register_raw() {
            let h = run_with_raw(EXT_ONE_PLUS_ULP, &[]);
            assert_eq!(st_raw(&h, 0), EXT_ONE_PLUS_ULP);
        }

        #[test]
        fn fld_st_i_pushes_a_bit_exact_copy() {
            let h = run_ok(
                // fld1
                // fld st(1)
                &program(&[LOAD_RAW_FROM_DATA, &[0xd9, 0xe8], &[0xd9, 0xc1]]),
                |h| h.write(DATA, &EXT_ONE_PLUS_ULP),
            );
            assert_eq!(st_raw(&h, 0), EXT_ONE_PLUS_ULP);
            assert_eq!(st_raw(&h, 1), EXT_ONE);
            assert_eq!(st_raw(&h, 2), EXT_ONE_PLUS_ULP);
            assert_eq!(top(&h), 5);
        }

        #[test]
        fn fst_m32_rounds_to_nearest_even() {
            let h = run_ok(
                // fld qword ptr [rsi]
                // fst dword ptr [rdi]
                &program(&[&[0xdd, 0x06], &[0xd9, 0x17]]),
                |h| put_f64s(h, &[1.1]),
            );
            assert_eq!(h.read(OUT, 4), 1.1f32.to_le_bytes());
            assert_eq!(top(&h), 7, "fst does not pop");
        }

        #[test]
        fn fst_m32_overflow_gives_infinity_and_underflow_zero() {
            let h = run_ok(
                // fld qword ptr [rsi]
                // fstp dword ptr [rdi]
                &program(&[&[0xdd, 0x06], &[0xd9, 0x1f]]),
                |h| put_f64s(h, &[1e300]),
            );
            assert_eq!(h.read(OUT, 4), f32::INFINITY.to_le_bytes());
            let h = run_ok(
                // fld qword ptr [rsi]
                // fstp dword ptr [rdi]
                &program(&[&[0xdd, 0x06], &[0xd9, 0x1f]]),
                |h| put_f64s(h, &[-1e-300]),
            );
            assert_eq!(h.read(OUT, 4), (-0.0f32).to_le_bytes());
        }

        #[test]
        fn fstp_pops_and_fst_does_not() {
            let h = run_ok(
                // fld1
                // fst qword ptr [rdi]
                // fstp qword ptr [rdi + 8]
                &program(&[&[0xd9, 0xe8], &[0xdd, 0x17], &[0xdd, 0x5f, 0x08]]),
                |_| {},
            );
            assert_eq!(top(&h), 0);
            assert!(is_empty(&h, 7));
            assert_eq!(f64::from_bits(h.read_u64(OUT)), 1.0);
            assert_eq!(f64::from_bits(h.read_u64(OUT + 8)), 1.0);
        }

        #[test]
        fn fst_m64_narrows_extra_precision_with_round_to_nearest() {
            // fstp qword ptr [rdi]
            let h = run_ok(&program(&[LOAD_RAW_FROM_DATA, &[0xdd, 0x1f]]), |h| {
                h.write(DATA, &EXT_ONE_PLUS_ULP);
            });
            assert_eq!(out_f64(&h), 1.0);
        }

        #[test]
        fn fst_m80_of_converted_values_round_trips_f64() {
            for value in [0.1f64, -123456.789, 1e-310, 6.02214076e23] {
                let h = run_ok(
                    // fld qword ptr [rsi]
                    // fstp tbyte ptr [rdi]
                    // fld tbyte ptr [rdi]
                    // fstp qword ptr [rdi + 16]
                    &program(&[
                        &[0xdd, 0x06],
                        &[0xdb, 0x3f],
                        &[0xdb, 0x2f],
                        &[0xdd, 0x5f, 0x10],
                    ]),
                    |h| put_f64s(h, &[value]),
                );
                assert_eq!(
                    f64::from_bits(h.read_u64(OUT + 16)).to_bits(),
                    value.to_bits()
                );
            }
        }

        #[test]
        fn fst_st_i_copies_without_popping() {
            let h = run_ok(
                // fld1
                // fldz
                // fst st(1)
                &program(&[&[0xd9, 0xe8], &[0xd9, 0xee], &[0xdd, 0xd1]]),
                |_| {},
            );
            assert_eq!(st(&h, 0), 0.0);
            assert_eq!(st(&h, 1), 0.0);
            assert_eq!(top(&h), 6);
        }

        #[test]
        fn fstp_st_i_copies_then_pops() {
            let h = run_ok(
                // fld1
                // fldz
                // fstp st(1)
                &program(&[&[0xd9, 0xe8], &[0xd9, 0xee], &[0xdd, 0xd9]]),
                |_| {},
            );
            assert_eq!(top(&h), 7);
            assert_eq!(
                st(&h, 0),
                0.0,
                "the old ST(0) landed in ST(1) and became ST(0) after the pop"
            );
            assert!(!is_empty(&h, 7));
            assert!(is_empty(&h, 6));
        }

        #[test]
        fn fxch_swaps_raw_values() {
            let h = run_ok(
                // fld1
                // fxch st(1)
                &program(&[LOAD_RAW_FROM_DATA, &[0xd9, 0xe8], &[0xd9, 0xc9]]),
                |h| h.write(DATA, &EXT_ONE_PLUS_ULP),
            );
            assert_eq!(st_raw(&h, 0), EXT_ONE_PLUS_ULP);
            assert_eq!(st_raw(&h, 1), EXT_ONE);
            assert_eq!(top(&h), 6);
        }

        #[test]
        fn fxch_with_a_distant_register() {
            let h = run_ok(
                // fld1
                // fldz
                // fldz
                // fldz
                // fxch st(3)
                &program(&[
                    &[0xd9, 0xe8],
                    &[0xd9, 0xee],
                    &[0xd9, 0xee],
                    &[0xd9, 0xee],
                    &[0xd9, 0xcb],
                ]),
                |_| {},
            );
            assert_eq!(st(&h, 0), 1.0);
            assert_eq!(st(&h, 3), 0.0);
        }

        #[test]
        fn eight_pushes_fill_the_register_stack() {
            let h = run_ok(
                // fld1; fld1; fld1; fld1; fld1; fld1; fld1; fld1
                &program(&[&[
                    0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9,
                    0xe8, 0xd9, 0xe8,
                ]]),
                |_| {},
            );
            assert_eq!(top(&h), 0);
            assert_eq!(h.cpu.ftw, 0xff);
        }

        #[test]
        fn pushing_decrements_top_and_marks_the_register_in_use() {
            // fld1; fld1; fld1
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8]]), |_| {});
            assert_eq!(top(&h), 5);
            assert_eq!(h.cpu.ftw, 0b1110_0000);
            assert_eq!(h.cpu.fsw & 0x3800, 5 << 11);
        }

        #[test]
        fn popping_increments_top_and_frees_the_register() {
            // fld1; fld1; fld1; fstp st(0)
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xdd, 0xd8]]),
                |_| {},
            );
            assert_eq!(top(&h), 6);
            assert_eq!(h.cpu.ftw, 0b1100_0000);
        }

        #[test]
        fn ffree_frees_without_moving_top() {
            // fld1; fld1; fld1; ffree st(1)
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xdd, 0xc1]]),
                |_| {},
            );
            assert_eq!(top(&h), 5);
            assert_eq!(
                h.cpu.ftw, 0b1010_0000,
                "physical register 6 (ST(1)) is now empty"
            );
        }

        #[test]
        fn ffreep_frees_and_pops() {
            // fld1; fld1; ffreep st(0)
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xdf, 0xc0]]), |_| {});
            assert_eq!(top(&h), 7);
            assert_eq!(h.cpu.ftw, 0b1000_0000);
        }

        #[test]
        fn fdecstp_and_fincstp_move_top_without_touching_tags() {
            // fld1; fdecstp
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xf6]]), |_| {});
            assert_eq!(top(&h), 6);
            assert_eq!(h.cpu.ftw, 0b1000_0000);
            // fld1; fincstp
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xf7]]), |_| {});
            assert_eq!(top(&h), 0);
            assert_eq!(h.cpu.ftw, 0b1000_0000);
        }

        #[test]
        fn top_wraps_around_in_both_directions() {
            // fincstp
            let h = run_ok(&program(&[&[0xd9, 0xf7]]), |_| {});
            assert_eq!(top(&h), 1);
            // fdecstp
            let h = run_ok(&program(&[&[0xd9, 0xf6]]), |_| {});
            assert_eq!(top(&h), 7);
            // fdecstp; fincstp
            let h = run_ok(&program(&[&[0xd9, 0xf6, 0xd9, 0xf7]]), |_| {});
            assert_eq!(top(&h), 0);
        }

        #[test]
        fn fnop_changes_nothing() {
            // fld1; fnop
            let mut h = Harness::new(&program(&[&[0xd9, 0xe8, 0xd9, 0xd0]]));
            assert_eq!(h.run(1), Stop::Limit);
            let before = (
                h.cpu.fcw,
                h.cpu.fsw,
                h.cpu.ftw,
                h.cpu.st,
                h.cpu.gpr,
                h.cpu.eflags_full(),
            );
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.rip, CODE_BASE + 4);
            assert_eq!(
                (
                    h.cpu.fcw,
                    h.cpu.fsw,
                    h.cpu.ftw,
                    h.cpu.st,
                    h.cpu.gpr,
                    h.cpu.eflags_full()
                ),
                before
            );
        }

        #[test]
        fn ninth_push_wraps_without_a_stack_fault_documented_limitation() {
            // Hardware would flag stack overflow (IE and SF, C1) and push the indefinite NaN.
            let h = run_ok(
                // fld1; fld1; fld1; fld1; fld1; fld1; fld1; fld1; fldz
                &program(&[&[
                    0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0xd9,
                    0xe8, 0xd9, 0xe8, 0xd9, 0xee,
                ]]),
                |_| {},
            );
            assert_eq!(top(&h), 7);
            assert_eq!(st(&h, 0), 0.0);
            assert_eq!(h.cpu.ftw, 0xff);
            assert_eq!(h.cpu.fsw & FSW_EXCEPTION_FLAGS, 0);
        }

        #[test]
        fn popping_an_empty_stack_just_moves_top_documented_limitation() {
            // fstp st(0)
            let h = run_ok(&program(&[&[0xdd, 0xd8]]), |_| {});
            assert_eq!(top(&h), 1);
            assert_eq!(h.cpu.fsw & FSW_EXCEPTION_FLAGS, 0);
        }

        #[test]
        fn memory_load_faults_leave_the_stack_untouched() {
            // fld qword ptr [rsi]
            let (h, stop) = run_with(&program(&[&[0xdd, 0x06]]), |h| {
                unmap_region(h, UNMAPPED_REGION);
                h.cpu.gpr[RSI] = UNMAPPED_REGION;
            });
            assert_eq!(stop, page_fault_not_present_read());
            assert_eq!(top(&h), 0);
            assert_eq!(h.cpu.ftw, 0);
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn jit_and_interpreter_agree_on_loads_stores_and_arithmetic() {
            crate::testkit::assert_jit_matches_interpreter(
                // fld qword ptr [rsi]; fld qword ptr [rsi + 8]; fadd st, st(1); fmul st(1), st; fxch st(1); fstp qword ptr [rdi]; fst dword ptr [rdi + 8]
                &program(&[&[
                    0xdd, 0x06, 0xdd, 0x46, 0x08, 0xd8, 0xc1, 0xdc, 0xc9, 0xd9, 0xc9, 0xdd, 0x1f,
                    0xd9, 0x57, 0x08,
                ]]),
                |h| {
                    init(h);
                    put_f64s(h, &[1.5, 2.25]);
                },
            );
        }
    }

    mod integer_conversions {
        use super::*;

        // fild word ptr [rsi]
        const FILD_M16: &[u8] = &[0xdf, 0x06];
        // fild dword ptr [rsi]
        const FILD_M32: &[u8] = &[0xdb, 0x06];
        // fild qword ptr [rsi]
        const FILD_M64: &[u8] = &[0xdf, 0x2e];

        fn convert(store: &[u8], value: f64, control_word: u16) -> Harness {
            // fld qword ptr [rsi]
            run_ok(&program(&[&[0xdd, 0x06], store]), |h| {
                put_f64s(h, &[value]);
                set_control_word(h, control_word);
            })
        }

        #[test]
        fn fild_m16_sign_extends() {
            for (value, expected) in [
                (0i16, 0.0),
                (1, 1.0),
                (-1, -1.0),
                (i16::MAX, 32767.0),
                (i16::MIN, -32768.0),
            ] {
                let h = run_ok(&program(&[FILD_M16]), |h| {
                    h.write(DATA, &value.to_le_bytes())
                });
                assert_eq!(st(&h, 0), expected, "{value}");
            }
        }

        #[test]
        fn fild_m32_sign_extends() {
            for (value, expected) in [
                (0i32, 0.0),
                (-5, -5.0),
                (i32::MAX, 2147483647.0),
                (i32::MIN, -2147483648.0),
            ] {
                let h = run_ok(&program(&[FILD_M32]), |h| {
                    h.write(DATA, &value.to_le_bytes())
                });
                assert_eq!(st(&h, 0), expected, "{value}");
            }
        }

        #[test]
        fn fild_m64_loads_signed_64_bit_integers() {
            for (value, expected) in [
                (0i64, 0.0),
                (-1, -1.0),
                (1 << 40, 1099511627776.0),
                (i64::MIN, -9223372036854775808.0),
            ] {
                let h = run_ok(&program(&[FILD_M64]), |h| {
                    h.write(DATA, &value.to_le_bytes())
                });
                assert_eq!(st(&h, 0), expected, "{value}");
            }
        }

        #[test]
        fn fild_m64_beyond_53_bits_loses_precision_documented_limitation() {
            // The extended format would hold 2^53 + 1 exactly; the f64 detour rounds it.
            let h = run_ok(&program(&[FILD_M64]), |h| {
                h.write(DATA, &((1i64 << 53) + 1).to_le_bytes())
            });
            assert_eq!(st(&h, 0), 9007199254740992.0);
        }

        #[test]
        fn fild_m16_leaves_a_zero_with_the_zero_tag() {
            // fnstenv [rdi]
            let h = run_ok(&program(&[FILD_M16, &[0xd9, 0x37]]), |h| {
                h.write(DATA, &0i16.to_le_bytes())
            });
            assert_eq!(
                h.read(OUT + 8, 2),
                [0xff, 0x7f],
                "ST(0) is zero: tag 01, the rest empty"
            );
        }

        #[test]
        fn rounding_to_integers_follows_fcw_rc_for_m16() {
            let table: [(f64, [i16; 4]); 8] = [
                (0.5, [0, 0, 1, 0]),
                (1.5, [2, 1, 2, 1]),
                (2.5, [2, 2, 3, 2]),
                (-0.5, [0, -1, 0, 0]),
                (-1.5, [-2, -2, -1, -1]),
                (-2.5, [-2, -3, -2, -2]),
                (2.7, [3, 2, 3, 2]),
                (-2.7, [-3, -3, -2, -2]),
            ];
            for (value, expected) in table {
                for (control, want) in [RC_NEAREST, RC_DOWN, RC_UP, RC_TRUNCATE]
                    .into_iter()
                    .zip(expected)
                {
                    // fistp word ptr [rdi]
                    let h = convert(&[0xdf, 0x1f], value, control);
                    assert_eq!(
                        i16::from_le_bytes(h.read(OUT, 2).try_into().unwrap()),
                        want,
                        "{value} fcw={control:#x}"
                    );
                }
            }
        }

        #[test]
        fn rounding_to_integers_follows_fcw_rc_for_m32_and_m64() {
            for (value, control, want) in [
                (2.5, RC_NEAREST, 2),
                (3.5, RC_NEAREST, 4),
                (-2.5, RC_DOWN, -3),
                (2.1, RC_UP, 3),
                (-2.9, RC_TRUNCATE, -2),
                (100000.5, RC_NEAREST, 100000),
                (-100000.5, RC_UP, -100000),
            ] {
                // fistp dword ptr [rdi]
                let h = convert(&[0xdb, 0x1f], value, control);
                assert_eq!(
                    i32::from_le_bytes(h.read(OUT, 4).try_into().unwrap()),
                    want,
                    "m32 {value} {control:#x}"
                );
                // fistp qword ptr [rdi]
                let h = convert(&[0xdf, 0x3f], value, control);
                assert_eq!(
                    i64::from_le_bytes(h.read(OUT, 8).try_into().unwrap()),
                    want as i64,
                    "m64 {value} {control:#x}"
                );
            }
        }

        #[test]
        fn fist_without_pop_keeps_the_value_on_the_stack() {
            // fist dword ptr [rdi]
            let h = convert(&[0xdb, 0x17], 7.0, RC_NEAREST);
            assert_eq!(top(&h), 7);
            assert_eq!(st(&h, 0), 7.0);
            // fist word ptr [rdi]
            let h = convert(&[0xdf, 0x17], 7.0, RC_NEAREST);
            assert_eq!(top(&h), 7);
        }

        #[test]
        fn fistp_pops() {
            // fistp dword ptr [rdi]
            let h = convert(&[0xdb, 0x1f], 7.0, RC_NEAREST);
            assert_eq!(top(&h), 0);
            assert!(is_empty(&h, 7));
        }

        #[test]
        fn the_fcw_rounding_control_set_by_fldcw_is_honoured() {
            let h = run_ok(
                // fldcw [rsi + 16]; fld qword ptr [rsi]; fistp dword ptr [rdi]
                &program(&[&[0xd9, 0x6e, 0x10, 0xdd, 0x06, 0xdb, 0x1f]]),
                |h| {
                    put_f64s(h, &[-2.5]);
                    h.write(DATA + 16, &RC_DOWN.to_le_bytes());
                },
            );
            assert_eq!(i32::from_le_bytes(h.read(OUT, 4).try_into().unwrap()), -3);
        }

        #[test]
        fn out_of_range_values_store_the_integer_indefinite() {
            let indefinite16 = i16::MIN.to_le_bytes();
            for value in [
                32768.0,
                -32769.0,
                1e10,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                // fistp word ptr [rdi]
                let h = convert(&[0xdf, 0x1f], value, RC_NEAREST);
                assert_eq!(h.read(OUT, 2), indefinite16, "{value}");
            }
            let indefinite32 = i32::MIN.to_le_bytes();
            for value in [2147483648.0, -2147483649.0, 1e18, f64::INFINITY, f64::NAN] {
                // fistp dword ptr [rdi]
                let h = convert(&[0xdb, 0x1f], value, RC_NEAREST);
                assert_eq!(h.read(OUT, 4), indefinite32, "{value}");
            }
            let indefinite64 = i64::MIN.to_le_bytes();
            for value in [
                9223372036854775808.0,
                -9.3e18,
                1e30,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                // fistp qword ptr [rdi]
                let h = convert(&[0xdf, 0x3f], value, RC_NEAREST);
                assert_eq!(h.read(OUT, 8), indefinite64, "{value}");
            }
        }

        #[test]
        fn the_extreme_in_range_values_convert_exactly() {
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], 32767.0, RC_NEAREST);
            assert_eq!(h.read(OUT, 2), 32767i16.to_le_bytes());
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], -32768.0, RC_NEAREST);
            assert_eq!(h.read(OUT, 2), i16::MIN.to_le_bytes());
            // fistp dword ptr [rdi]
            let h = convert(&[0xdb, 0x1f], 2147483647.0, RC_NEAREST);
            assert_eq!(h.read(OUT, 4), i32::MAX.to_le_bytes());
            // fistp qword ptr [rdi]
            let h = convert(&[0xdf, 0x3f], -9223372036854775808.0, RC_NEAREST);
            assert_eq!(h.read(OUT, 8), i64::MIN.to_le_bytes());
        }

        #[test]
        fn rounding_happens_before_the_range_check() {
            // 32767.5 rounds to 32768 (ties to even) and overflows; 32767.4 rounds down and fits.
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], 32767.5, RC_NEAREST);
            assert_eq!(h.read(OUT, 2), i16::MIN.to_le_bytes());
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], 32767.4, RC_NEAREST);
            assert_eq!(h.read(OUT, 2), 32767i16.to_le_bytes());
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], -32768.4, RC_NEAREST);
            assert_eq!(
                h.read(OUT, 2),
                i16::MIN.to_le_bytes(),
                "-32768.4 rounds to -32768 and fits"
            );
            // fistp word ptr [rdi]
            let h = convert(&[0xdf, 0x1f], -32768.6, RC_NEAREST);
            assert_eq!(
                h.read(OUT, 2),
                i16::MIN.to_le_bytes(),
                "-32769 overflows to the indefinite"
            );
        }

        #[test]
        fn fisttp_truncates_whatever_the_rounding_control_says_and_pops() {
            for control in [RC_NEAREST, RC_DOWN, RC_UP, RC_TRUNCATE] {
                for (value, want) in [
                    (2.9, 2),
                    (-2.9, -2),
                    (0.99, 0),
                    (-0.5, 0),
                    (100000.99, 100000),
                    (-100000.99, -100000),
                ] {
                    // fisttp dword ptr [rdi]
                    let h = convert(&[0xdb, 0x0f], value, control);
                    assert_eq!(
                        i32::from_le_bytes(h.read(OUT, 4).try_into().unwrap()),
                        want,
                        "m32 {value} fcw={control:#x}"
                    );
                    assert_eq!(top(&h), 0, "fisttp pops");
                    assert!(is_empty(&h, 7));
                    // fisttp qword ptr [rdi]
                    let h = convert(&[0xdd, 0x0f], value, control);
                    assert_eq!(
                        i64::from_le_bytes(h.read(OUT, 8).try_into().unwrap()),
                        want as i64,
                        "m64 {value} fcw={control:#x}"
                    );
                    assert_eq!(top(&h), 0);
                    if value.abs() < 30000.0 {
                        // fisttp word ptr [rdi]
                        let h = convert(&[0xdf, 0x0f], value, control);
                        assert_eq!(
                            i16::from_le_bytes(h.read(OUT, 2).try_into().unwrap()),
                            want as i16,
                            "m16 {value} fcw={control:#x}"
                        );
                        assert_eq!(top(&h), 0);
                    }
                }
            }
        }

        #[test]
        fn fisttp_out_of_range_and_nan_store_the_integer_indefinite() {
            for (value, m16, m32, m64) in [
                (40000.5, i16::MIN as i64, 40000, 40000),
                (3e9, i16::MIN as i64, i32::MIN as i64, 3_000_000_000),
                (-3e9, i16::MIN as i64, i32::MIN as i64, -3_000_000_000),
                (1e30, i16::MIN as i64, i32::MIN as i64, i64::MIN),
                (f64::NAN, i16::MIN as i64, i32::MIN as i64, i64::MIN),
                (
                    f64::NEG_INFINITY,
                    i16::MIN as i64,
                    i32::MIN as i64,
                    i64::MIN,
                ),
                // truncation, not rounding, decides whether the value is in range
                (32767.9, 32767, 32767, 32767),
                (-32768.9, -32768, -32768, -32768),
            ] {
                let h = convert(&[0xdf, 0x0f], value, RC_NEAREST);
                assert_eq!(
                    i16::from_le_bytes(h.read(OUT, 2).try_into().unwrap()) as i64,
                    m16,
                    "m16 {value}"
                );
                let h = convert(&[0xdb, 0x0f], value, RC_NEAREST);
                assert_eq!(
                    i32::from_le_bytes(h.read(OUT, 4).try_into().unwrap()) as i64,
                    m32,
                    "m32 {value}"
                );
                let h = convert(&[0xdd, 0x0f], value, RC_NEAREST);
                assert_eq!(
                    i64::from_le_bytes(h.read(OUT, 8).try_into().unwrap()),
                    m64,
                    "m64 {value}"
                );
            }
        }

        #[test]
        fn dd_and_df_slash_1_register_encodings_stay_invalid() {
            // fld1, then DD C8 and DF C8. fisttp has no register form; hardware treats these as
            // undocumented FXCH aliases, which this emulator does not implement (it raises #UD), and
            // this pins that rather than architectural behavior.
            for opcode in [0xdd, 0xdf] {
                let code = program(&[&[0xd9, 0xe8], &[opcode, 0xc8]]);
                let (_, stop) = run_with(&code, |_| {});
                assert_eq!(stop, invalid_opcode(), "{opcode:#x}");
            }
        }

        #[test]
        fn fisttp_matches_the_jit_helper() {
            // fld qword ptr [rsi]; fisttp dword ptr [rdi]
            let code = program(&[&[0xdd, 0x06], &[0xdb, 0x0f]]);
            crate::testkit::assert_jit_matches_interpreter(&code, |h| {
                init(h);
                put_f64s(h, &[-7.75]);
            });
        }

        #[test]
        fn bcd_loads_and_stores_are_not_implemented_documented_limitation() {
            // fbld [rsi]
            // fld1
            // fbstp [rdi]
            for code in [
                program(&[&[0xdf, 0x26]]),
                program(&[&[0xd9, 0xe8], &[0xdf, 0x37]]),
            ] {
                let (_, stop) = run_with(&code, |_| {});
                assert_eq!(stop, invalid_opcode());
            }
        }
    }

    mod arithmetic {
        use super::*;

        // With a = 12 loaded first and b = 3 on top: ST(1) = 12, ST(0) = 3.
        const A: f64 = 12.0;
        const B: f64 = 3.0;

        /// Register forms that write ST(0) = ST(0) op ST(1) (opcode d8).
        #[test]
        fn d8_register_forms_compute_st0_op_st1() {
            let cases: [(&[u8], f64); 6] = [
                // fadd st, st(1)
                (&[0xd8, 0xc1], B + A),
                // fmul st, st(1)
                (&[0xd8, 0xc9], B * A),
                // fsub st, st(1)
                (&[0xd8, 0xe1], B - A),
                // fsubr st, st(1)
                (&[0xd8, 0xe9], A - B),
                // fdiv st, st(1)
                (&[0xd8, 0xf1], B / A),
                // fdivr st, st(1)
                (&[0xd8, 0xf9], A / B),
            ];
            for (op, expected) in cases {
                let h = run_binary(op, A, B);
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
                assert_eq!(st(&h, 1), A, "{op:02x?}: ST(1) is untouched");
                assert_eq!(top(&h), 6, "{op:02x?}: nothing is popped");
            }
        }

        /// Register forms that write ST(i) = ST(i) op ST(0) (opcode dc).
        #[test]
        fn dc_register_forms_compute_sti_op_st0() {
            let cases: [(&[u8], f64); 6] = [
                // fadd st(1), st
                (&[0xdc, 0xc1], A + B),
                // fmul st(1), st
                (&[0xdc, 0xc9], A * B),
                // fsub st(1), st
                (&[0xdc, 0xe9], A - B),
                // fsubr st(1), st
                (&[0xdc, 0xe1], B - A),
                // fdiv st(1), st
                (&[0xdc, 0xf9], A / B),
                // fdivr st(1), st
                (&[0xdc, 0xf1], B / A),
            ];
            for (op, expected) in cases {
                let h = run_binary(op, A, B);
                assert_eq!(st(&h, 1), expected, "{op:02x?}");
                assert_eq!(st(&h, 0), B, "{op:02x?}: ST(0) is untouched");
                assert_eq!(top(&h), 6);
            }
        }

        /// The same results as dc, with a pop: the result ends up in ST(0).
        #[test]
        fn de_register_forms_pop_after_computing_sti_op_st0() {
            let cases: [(&[u8], f64); 6] = [
                // faddp st(1), st
                (&[0xde, 0xc1], A + B),
                // fmulp st(1), st
                (&[0xde, 0xc9], A * B),
                // fsubp st(1), st
                (&[0xde, 0xe9], A - B),
                // fsubrp st(1), st
                (&[0xde, 0xe1], B - A),
                // fdivp st(1), st
                (&[0xde, 0xf9], A / B),
                // fdivrp st(1), st
                (&[0xde, 0xf1], B / A),
            ];
            for (op, expected) in cases {
                let h = run_binary(op, A, B);
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
                assert_eq!(top(&h), 7, "{op:02x?}: one pop");
                assert!(
                    is_empty(&h, 6),
                    "{op:02x?}: the old ST(0) register is freed"
                );
            }
        }

        #[test]
        fn register_forms_with_a_distant_operand() {
            // ST(2) = 100, ST(1) = 7, ST(0) = 5
            let program_with = |op: &[u8]| {
                run_ok(
                    // fld qword ptr [rsi]; fld qword ptr [rsi + 8]; fld qword ptr [rsi + 16]
                    &program(&[&[0xdd, 0x06, 0xdd, 0x46, 0x08, 0xdd, 0x46, 0x10], op]),
                    |h| put_f64s(h, &[100.0, 7.0, 5.0]),
                )
            };
            // fsub st, st(2)
            let h = program_with(&[0xd8, 0xe2]);
            assert_eq!(st(&h, 0), 5.0 - 100.0);
            // fdiv st(2), st
            let h = program_with(&[0xdc, 0xfa]);
            assert_eq!(st(&h, 2), 20.0);
            // faddp st(2), st
            let h = program_with(&[0xde, 0xc2]);
            assert_eq!(st(&h, 1), 105.0);
            assert_eq!(st(&h, 0), 7.0);
        }

        #[test]
        fn adding_a_register_to_itself_doubles_it() {
            // fadd st, st(0)
            let h = run_binary(&[0xd8, 0xc0], A, B);
            assert_eq!(st(&h, 0), 6.0);
            // fmul st, st(0)
            let h = run_binary(&[0xd8, 0xc8], A, B);
            assert_eq!(st(&h, 0), 9.0);
        }

        #[test]
        fn memory_operand_forms_with_m64() {
            let cases: [(&[u8], f64); 6] = [
                // fadd qword ptr [rsi + 16]
                (&[0xdc, 0x46, 0x10], 15.0),
                // fmul qword ptr [rsi + 16]
                (&[0xdc, 0x4e, 0x10], 36.0),
                // fsub qword ptr [rsi + 16]
                (&[0xdc, 0x66, 0x10], 9.0),
                // fsubr qword ptr [rsi + 16]
                (&[0xdc, 0x6e, 0x10], -9.0),
                // fdiv qword ptr [rsi + 16]
                (&[0xdc, 0x76, 0x10], 4.0),
                // fdivr qword ptr [rsi + 16]
                (&[0xdc, 0x7e, 0x10], 0.25),
            ];
            for (op, expected) in cases {
                // fld qword ptr [rsi]
                let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                    put_f64s(h, &[12.0, 0.0, 3.0])
                });
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
                assert_eq!(top(&h), 7);
            }
        }

        #[test]
        fn memory_operand_forms_with_m32() {
            let cases: [(&[u8], f64); 6] = [
                // fadd dword ptr [rsi + 16]
                (&[0xd8, 0x46, 0x10], 15.0),
                // fmul dword ptr [rsi + 16]
                (&[0xd8, 0x4e, 0x10], 36.0),
                // fsub dword ptr [rsi + 16]
                (&[0xd8, 0x66, 0x10], 9.0),
                // fsubr dword ptr [rsi + 16]
                (&[0xd8, 0x6e, 0x10], -9.0),
                // fdiv dword ptr [rsi + 16]
                (&[0xd8, 0x76, 0x10], 4.0),
                // fdivr dword ptr [rsi + 16]
                (&[0xd8, 0x7e, 0x10], 0.25),
            ];
            for (op, expected) in cases {
                // fld qword ptr [rsi]
                let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                    put_f64s(h, &[12.0]);
                    h.write(DATA + 16, &3.0f32.to_le_bytes());
                });
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
            }
        }

        #[test]
        fn integer_memory_operand_forms_with_m16() {
            let cases: [(&[u8], f64); 6] = [
                // fiadd word ptr [rsi + 16]
                (&[0xde, 0x46, 0x10], 3.0),
                // fimul word ptr [rsi + 16]
                (&[0xde, 0x4e, 0x10], -18.0),
                // fisub word ptr [rsi + 16]
                (&[0xde, 0x66, 0x10], 9.0),
                // fisubr word ptr [rsi + 16]
                (&[0xde, 0x6e, 0x10], -9.0),
                // fidiv word ptr [rsi + 16]
                (&[0xde, 0x76, 0x10], -2.0),
                // fidivr word ptr [rsi + 16]
                (&[0xde, 0x7e, 0x10], -0.5),
            ];
            for (op, expected) in cases {
                // fld qword ptr [rsi]
                let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                    put_f64s(h, &[6.0]);
                    h.write(DATA + 16, &(-3i16).to_le_bytes());
                });
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
            }
        }

        #[test]
        fn integer_memory_operand_forms_with_m32() {
            let cases: [(&[u8], f64); 6] = [
                // fiadd dword ptr [rsi + 16]
                (&[0xda, 0x46, 0x10], 100_006.0),
                // fimul dword ptr [rsi + 16]
                (&[0xda, 0x4e, 0x10], 600_000.0),
                // fisub dword ptr [rsi + 16]
                (&[0xda, 0x66, 0x10], -99_994.0),
                // fisubr dword ptr [rsi + 16]
                (&[0xda, 0x6e, 0x10], 99_994.0),
                // fidiv dword ptr [rsi + 16]
                (&[0xda, 0x76, 0x10], 6.0 / 100_000.0),
                // fidivr dword ptr [rsi + 16]
                (&[0xda, 0x7e, 0x10], 100_000.0 / 6.0),
            ];
            for (op, expected) in cases {
                // fld qword ptr [rsi]
                let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                    put_f64s(h, &[6.0]);
                    h.write(DATA + 16, &100_000i32.to_le_bytes());
                });
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
            }
        }

        #[test]
        fn integer_operands_are_sign_extended() {
            // fld1
            // fiadd word ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0xe8], &[0xde, 0x06]]), |h| {
                h.write(DATA, &0x8000u16.to_le_bytes())
            });
            assert_eq!(st(&h, 0), 1.0 - 32768.0);
            // fld1
            // fiadd dword ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0xe8], &[0xda, 0x06]]), |h| {
                h.write(DATA, &0x8000_0000u32.to_le_bytes())
            });
            assert_eq!(st(&h, 0), 1.0 - 2147483648.0);
        }

        #[test]
        fn m32_operands_are_widened_before_use() {
            // fldz
            // fadd dword ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0xee], &[0xd8, 0x06]]), |h| {
                h.write(DATA, &0.1f32.to_le_bytes())
            });
            assert_eq!(st(&h, 0), 0.1f32 as f64);
        }

        #[test]
        fn division_by_zero_gives_infinity_without_signalling_documented_limitation() {
            // Hardware also sets FSW.ZE (bit 2) even when the exception is masked.
            // fdiv st, st(1)
            let h = run_binary(&[0xd8, 0xf1], 0.0, 1.0);
            assert_eq!(st(&h, 0), f64::INFINITY);
            assert_eq!(h.cpu.fsw & FSW_EXCEPTION_FLAGS, 0);
            // fdiv st, st(1)
            let h = run_binary(&[0xd8, 0xf1], 0.0, -1.0);
            assert_eq!(st(&h, 0), f64::NEG_INFINITY);
        }

        #[test]
        fn zero_over_zero_is_a_nan() {
            // fdiv st, st(1)
            let h = run_binary(&[0xd8, 0xf1], 0.0, 0.0);
            assert!(st(&h, 0).is_nan());
        }

        #[test]
        fn infinity_arithmetic() {
            // fadd st, st(1)
            let h = run_binary(&[0xd8, 0xc1], f64::INFINITY, 1.0);
            assert_eq!(st(&h, 0), f64::INFINITY);
            // fsub st, st(1)
            let h = run_binary(&[0xd8, 0xe1], f64::INFINITY, f64::INFINITY);
            assert!(st(&h, 0).is_nan(), "inf - inf");
            // fmul st, st(1)
            let h = run_binary(&[0xd8, 0xc9], f64::INFINITY, 0.0);
            assert!(st(&h, 0).is_nan(), "inf * 0");
        }

        #[test]
        fn results_are_rounded_to_double_precision_documented_limitation() {
            // 1 + 2^-60 fits the 64-bit extended significand, so hardware keeps it; the f64 detour
            // rounds it back to exactly 1.
            let two_to_the_minus_60 = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xc3, 0x3f];
            // fld1
            // fadd st, st(1)
            let h = run_ok(
                &program(&[LOAD_RAW_FROM_DATA, &[0xd9, 0xe8], &[0xd8, 0xc1]]),
                |h| {
                    h.write(DATA, &two_to_the_minus_60);
                },
            );
            assert_eq!(st_raw(&h, 0), EXT_ONE);
        }

        #[test]
        fn memory_operand_faults_leave_the_stack_alone() {
            // fld1
            // fadd qword ptr [rsi]
            let code = program(&[&[0xd9, 0xe8], &[0xdc, 0x06]]);
            let (h, stop) = run_with(&code, |h| {
                unmap_region(h, UNMAPPED_REGION);
                h.cpu.gpr[RSI] = UNMAPPED_REGION;
            });
            assert_eq!(stop, page_fault_not_present_read());
            assert_eq!(st(&h, 0), 1.0);
            assert_eq!(top(&h), 7);
            assert_eq!(h.cpu.rip, CODE_BASE + 2);
        }

        #[test]
        fn jit_and_interpreter_agree_on_every_arithmetic_form() {
            let ops: [&[u8]; 5] = [
                // fadd st, st(1)
                &[0xd8, 0xc1],
                // fsubr st(1), st
                &[0xdc, 0xe1],
                // fdivp st(1), st
                &[0xde, 0xf9],
                // fmul qword ptr [rsi + 16]
                &[0xdc, 0x4e, 0x10],
                // fisub dword ptr [rsi + 16]
                &[0xda, 0x66, 0x10],
            ];
            for op in ops {
                crate::testkit::assert_jit_matches_interpreter(
                    &program(&[LOAD_A_THEN_B, op]),
                    |h| {
                        init(h);
                        put_f64s(h, &[A, B, 5.0]);
                    },
                );
            }
        }
    }

    mod comparisons {
        use super::*;

        const GREATER: u16 = 0;
        const LESS: u16 = C0;
        const EQUAL: u16 = C3;
        const UNORDERED: u16 = C0 | C2 | C3;

        // a = ST(1), b = ST(0): the compare asks how ST(0) relates to ST(i).
        // fcom st(1)
        const FCOM_ST1: &[u8] = &[0xd8, 0xd1];
        // fcomp st(1)
        const FCOMP_ST1: &[u8] = &[0xd8, 0xd9];
        // fucom st(1)
        const FUCOM_ST1: &[u8] = &[0xdd, 0xe1];
        // fucomp st(1)
        const FUCOMP_ST1: &[u8] = &[0xdd, 0xe9];

        #[test]
        fn fcom_reports_st0_relative_to_the_operand_in_the_condition_codes() {
            for op in [FCOM_ST1, FCOMP_ST1, FUCOM_ST1, FUCOMP_ST1] {
                for (a, b, expected) in [
                    (1.0, 2.0, GREATER),
                    (2.0, 1.0, LESS),
                    (2.0, 2.0, EQUAL),
                    (-1.0, 1.0, GREATER),
                    (-0.0, 0.0, EQUAL),
                    (f64::NAN, 1.0, UNORDERED),
                    (1.0, f64::NAN, UNORDERED),
                    (f64::INFINITY, 1.0, LESS),
                    (f64::NEG_INFINITY, f64::NEG_INFINITY, EQUAL),
                ] {
                    let h = run_binary(op, a, b);
                    assert_eq!(condition_codes(&h), expected, "{op:02x?} st1={a} st0={b}");
                }
            }
        }

        #[test]
        fn fcom_leaves_c1_alone() {
            let mut h = Harness::new(&program(&[LOAD_A_THEN_B, FCOM_ST1]));
            init(&mut h);
            put_f64s(&mut h, &[1.0, 2.0]);
            h.cpu.fsw |= C1;
            h.run_to_halt();
            assert_ne!(h.cpu.fsw & C1, 0);
        }

        #[test]
        fn fcom_does_not_pop_but_fcomp_does() {
            assert_eq!(top(&run_binary(FCOM_ST1, 1.0, 2.0)), 6);
            assert_eq!(top(&run_binary(FUCOM_ST1, 1.0, 2.0)), 6);
            assert_eq!(top(&run_binary(FCOMP_ST1, 1.0, 2.0)), 7);
            assert_eq!(top(&run_binary(FUCOMP_ST1, 1.0, 2.0)), 7);
        }

        #[test]
        fn fcompp_and_fucompp_pop_twice() {
            // fcompp
            // fucompp
            for op in [&[0xde, 0xd9], &[0xda, 0xe9]] {
                let h = run_binary(op, 5.0, 2.0);
                assert_eq!(top(&h), 0, "{op:02x?}");
                assert_eq!(h.cpu.ftw, 0);
                assert_eq!(
                    condition_codes(&h),
                    LESS,
                    "{op:02x?}: ST(0)=2 is below ST(1)=5"
                );
            }
        }

        #[test]
        fn fcompp_compares_st0_with_st1() {
            // fcompp
            let h = run_binary(&[0xde, 0xd9], 2.0, 5.0);
            assert_eq!(condition_codes(&h), GREATER, "ST(0)=5 is above ST(1)=2");
        }

        #[test]
        fn fcom_with_a_distant_register() {
            let h = run_ok(
                // fld qword ptr [rsi]; fld qword ptr [rsi + 8]; fld qword ptr [rsi + 16]
                // fcom st(2)
                &program(&[
                    &[0xdd, 0x06, 0xdd, 0x46, 0x08, 0xdd, 0x46, 0x10],
                    &[0xd8, 0xd2],
                ]),
                |h| put_f64s(h, &[1.0, 2.0, 3.0]),
            );
            assert_eq!(condition_codes(&h), GREATER, "ST(0)=3 vs ST(2)=1");
        }

        #[test]
        fn fcom_memory_forms() {
            let forms: [(&[u8], Vec<u8>); 2] = [
                // fcom dword ptr [rsi + 16]
                (&[0xd8, 0x56, 0x10], 2.0f32.to_le_bytes().to_vec()),
                // fcom qword ptr [rsi + 16]
                (&[0xdc, 0x56, 0x10], 2.0f64.to_le_bytes().to_vec()),
            ];
            for (op, memory) in forms {
                for (value, expected) in [(1.0, LESS), (3.0, GREATER), (2.0, EQUAL)] {
                    // fld qword ptr [rsi]
                    let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                        put_f64s(h, &[value]);
                        h.write(DATA + 16, &memory);
                    });
                    assert_eq!(condition_codes(&h), expected, "{op:02x?} {value}");
                    assert_eq!(top(&h), 7, "fcom does not pop");
                }
            }
        }

        #[test]
        fn fcomp_memory_forms_pop() {
            // fcomp dword ptr [rsi + 16]
            // fcomp qword ptr [rsi + 16]
            for op in [&[0xd8, 0x5e, 0x10], &[0xdc, 0x5e, 0x10]] {
                // fld qword ptr [rsi]
                let h = run_ok(&program(&[&[0xdd, 0x06], op]), |h| {
                    put_f64s(h, &[1.0]);
                    h.write(DATA + 16, &[0; 8]);
                });
                assert_eq!(top(&h), 0);
            }
        }

        #[test]
        fn ficom_and_ficomp_compare_with_integers() {
            // fld qword ptr [rsi]
            // ficom word ptr [rsi + 16]
            let h = run_ok(&program(&[&[0xdd, 0x06], &[0xde, 0x56, 0x10]]), |h| {
                put_f64s(h, &[-5.0]);
                h.write(DATA + 16, &(-7i16).to_le_bytes());
            });
            assert_eq!(condition_codes(&h), GREATER, "-5 > -7");
            assert_eq!(top(&h), 7);
            // fld qword ptr [rsi]
            // ficomp dword ptr [rsi + 16]
            let h = run_ok(&program(&[&[0xdd, 0x06], &[0xda, 0x5e, 0x10]]), |h| {
                put_f64s(h, &[100000.0]);
                h.write(DATA + 16, &100_000i32.to_le_bytes());
            });
            assert_eq!(condition_codes(&h), EQUAL);
            assert_eq!(top(&h), 0, "ficomp pops");
        }

        #[test]
        fn ftst_compares_with_zero() {
            for (value, expected) in [
                (1.0, GREATER),
                (-1.0, LESS),
                (0.0, EQUAL),
                (-0.0, EQUAL),
                (f64::NAN, UNORDERED),
                (f64::INFINITY, GREATER),
                (f64::NEG_INFINITY, LESS),
            ] {
                // fld qword ptr [rsi]
                // ftst
                let h = run_ok(&program(&[&[0xdd, 0x06], &[0xd9, 0xe4]]), |h| {
                    put_f64s(h, &[value])
                });
                assert_eq!(condition_codes(&h), expected, "{value}");
                assert_eq!(top(&h), 7, "ftst does not pop");
            }
        }

        // fcomi st, st(1)
        const FCOMI_ST1: &[u8] = &[0xdb, 0xf1];
        // fucomi st, st(1)
        const FUCOMI_ST1: &[u8] = &[0xdb, 0xe9];
        // fcomip st, st(1)
        const FCOMIP_ST1: &[u8] = &[0xdf, 0xf1];
        // fucomip st, st(1)
        const FUCOMIP_ST1: &[u8] = &[0xdf, 0xe9];

        #[test]
        fn fcomi_family_reports_in_zf_pf_cf() {
            for op in [FCOMI_ST1, FUCOMI_ST1, FCOMIP_ST1, FUCOMIP_ST1] {
                for (a, b, expected) in [
                    (1.0, 2.0, 0),
                    (2.0, 1.0, CF),
                    (2.0, 2.0, ZF),
                    (f64::NAN, 1.0, ZF | PF | CF),
                    (1.0, f64::NAN, ZF | PF | CF),
                    (-0.0, 0.0, ZF),
                ] {
                    let h = run_binary(op, a, b);
                    assert_eq!(
                        h.cpu.eflags_full() & (ZF | PF | CF),
                        expected,
                        "{op:02x?} st1={a} st0={b}"
                    );
                }
            }
        }

        #[test]
        fn fcomi_family_leaves_the_condition_codes_alone() {
            let mut h = Harness::new(&program(&[LOAD_A_THEN_B, FCOMI_ST1]));
            init(&mut h);
            put_f64s(&mut h, &[1.0, 2.0]);
            h.cpu.fsw |= C0 | C2 | C3;
            h.run_to_halt();
            assert_eq!(h.cpu.fsw & (C0 | C2 | C3), C0 | C2 | C3);
        }

        #[test]
        fn fcomip_and_fucomip_pop_once_and_fcomi_does_not() {
            assert_eq!(top(&run_binary(FCOMI_ST1, 1.0, 2.0)), 6);
            assert_eq!(top(&run_binary(FUCOMI_ST1, 1.0, 2.0)), 6);
            assert_eq!(top(&run_binary(FCOMIP_ST1, 1.0, 2.0)), 7);
            assert_eq!(top(&run_binary(FUCOMIP_ST1, 1.0, 2.0)), 7);
        }

        #[test]
        fn fcomi_feeds_a_conditional_jump() {
            // fcomi then ja: taken only when ST(0) > ST(1).
            // ja skip; mov eax, 1; skip: nop
            let code = program(&[
                LOAD_A_THEN_B,
                FCOMI_ST1,
                &[0x77, 0x05, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x90],
            ]);
            let h = run_ok(&code, |h| put_f64s(h, &[1.0, 2.0]));
            assert_eq!(h.cpu.gpr[RAX], 0, "ST(0)=2 > ST(1)=1: ja taken");
            let h = run_ok(&code, |h| put_f64s(h, &[2.0, 1.0]));
            assert_eq!(h.cpu.gpr[RAX], 1, "ST(0)=1 < ST(1)=2: ja not taken");
        }

        #[test]
        fn fcomi_with_a_distant_register() {
            let h = run_ok(
                // fld qword ptr [rsi]; fld qword ptr [rsi + 8]; fld qword ptr [rsi + 16]
                // fcomi st, st(2)
                &program(&[
                    &[0xdd, 0x06, 0xdd, 0x46, 0x08, 0xdd, 0x46, 0x10],
                    &[0xdb, 0xf2],
                ]),
                |h| put_f64s(h, &[3.0, 2.0, 3.0]),
            );
            assert_eq!(h.cpu.eflags_full() & (ZF | PF | CF), ZF);
        }

        #[test]
        fn fcmov_moves_st_i_into_st0_when_the_flag_condition_holds() {
            let cases: [(&[u8], u64, bool); 14] = [
                // fcmovb st, st(1)
                (&[0xda, 0xc1], CF, true),
                // fcmovb st, st(1)
                (&[0xda, 0xc1], 0, false),
                // fcmove st, st(1)
                (&[0xda, 0xc9], ZF, true),
                // fcmove st, st(1)
                (&[0xda, 0xc9], CF, false),
                // fcmovbe st, st(1)
                (&[0xda, 0xd1], CF, true),
                // fcmovbe st, st(1)
                (&[0xda, 0xd1], ZF, true),
                // fcmovbe st, st(1)
                (&[0xda, 0xd1], PF, false),
                // fcmovu st, st(1)
                (&[0xda, 0xd9], PF, true),
                // fcmovu st, st(1)
                (&[0xda, 0xd9], CF | ZF, false),
                // fcmovnb st, st(1)
                (&[0xdb, 0xc1], 0, true),
                // fcmovnb st, st(1)
                (&[0xdb, 0xc1], CF, false),
                // fcmovne st, st(1)
                (&[0xdb, 0xc9], 0, true),
                // fcmovnbe st, st(1)
                (&[0xdb, 0xd1], PF, true),
                // fcmovnu st, st(1)
                (&[0xdb, 0xd9], PF, false),
            ];
            for (op, flags, moves) in cases {
                let mut h = Harness::new(&program(&[LOAD_A_THEN_B, op]));
                init(&mut h);
                put_f64s(&mut h, &[7.0, 9.0]);
                h.cpu.set_eflags_arith(flags);
                h.run_to_halt();
                assert_eq!(
                    st(&h, 0),
                    if moves { 7.0 } else { 9.0 },
                    "{op:02x?} flags={flags:#x}"
                );
                assert_eq!(st(&h, 1), 7.0, "the source is unchanged");
                assert_eq!(top(&h), 6, "no pop");
            }
        }

        #[test]
        fn fcmovnbe_and_fcmovne_the_remaining_combinations() {
            for (op, flags, moves) in [
                // fcmovnbe st, st(1)
                (&[0xdb, 0xd1], CF, false),
                // fcmovnbe st, st(1)
                (&[0xdb, 0xd1], ZF, false),
                // fcmovne st, st(1)
                (&[0xdb, 0xc9], ZF, false),
                // fcmovnu st, st(1)
                (&[0xdb, 0xd9], 0, true),
            ] {
                let mut h = Harness::new(&program(&[LOAD_A_THEN_B, op]));
                init(&mut h);
                put_f64s(&mut h, &[7.0, 9.0]);
                h.cpu.set_eflags_arith(flags);
                h.run_to_halt();
                assert_eq!(
                    st(&h, 0),
                    if moves { 7.0 } else { 9.0 },
                    "{op:02x?} flags={flags:#x}"
                );
            }
        }
    }

    mod unary_operations {
        use super::*;

        fn run_unary(raw: [u8; 10], op: &[u8]) -> Harness {
            run_with_raw(raw, op)
        }

        fn run_unary_f64(value: f64, op: &[u8]) -> Harness {
            // fld qword ptr [rsi]
            run_ok(&program(&[&[0xdd, 0x06], op]), |h| put_f64s(h, &[value]))
        }

        #[test]
        fn fchs_flips_only_the_sign_bit_of_the_raw_value() {
            for (raw, expected) in [
                (EXT_ONE, [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0xbf]),
                (EXT_ZERO, EXT_NEG_ZERO),
                (EXT_NEG_ZERO, EXT_ZERO),
                (EXT_INFINITY, EXT_NEG_INFINITY),
                (
                    EXT_QUIET_NAN_PAYLOAD,
                    [0x23, 0x01, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xff],
                ),
                (EXT_DENORMAL, EXT_NEG_DENORMAL),
            ] {
                // fchs
                let h = run_unary(raw, &[0xd9, 0xe0]);
                assert_eq!(st_raw(&h, 0), expected, "{raw:02x?}");
                assert_eq!(top(&h), 7);
            }
        }

        #[test]
        fn fabs_clears_the_sign_bit_of_the_raw_value() {
            for (raw, expected) in [
                (EXT_NEG_TWO, [0, 0, 0, 0, 0, 0, 0, 0x80, 0x00, 0x40]),
                (EXT_NEG_ZERO, EXT_ZERO),
                (EXT_NEG_INFINITY, EXT_INFINITY),
                (EXT_NEG_DENORMAL, EXT_DENORMAL),
                (EXT_ONE, EXT_ONE),
            ] {
                // fabs
                let h = run_unary(raw, &[0xd9, 0xe1]);
                assert_eq!(st_raw(&h, 0), expected, "{raw:02x?}");
            }
        }

        #[test]
        fn fchs_and_fabs_do_not_round_extended_precision_values() {
            // fchs
            let h = run_unary(EXT_ONE_PLUS_ULP, &[0xd9, 0xe0]);
            assert_eq!(st_raw(&h, 0)[..8], EXT_ONE_PLUS_ULP[..8]);
            // fabs
            let h = run_unary(EXT_PI_FULL, &[0xd9, 0xe1]);
            assert_eq!(st_raw(&h, 0), EXT_PI_FULL);
        }

        #[test]
        fn fsqrt_of_perfect_squares_is_exact() {
            for (value, expected) in [
                (16.0, 4.0),
                (0.25, 0.5),
                (1.0, 1.0),
                (0.0, 0.0),
                (f64::INFINITY, f64::INFINITY),
            ] {
                // fsqrt
                let h = run_unary_f64(value, &[0xd9, 0xfa]);
                assert_eq!(st(&h, 0), expected, "{value}");
            }
        }

        #[test]
        fn fsqrt_of_two_matches_the_double_precision_result() {
            // fsqrt
            let h = run_unary_f64(2.0, &[0xd9, 0xfa]);
            assert_eq!(st(&h, 0), core::f64::consts::SQRT_2);
        }

        #[test]
        fn fsqrt_of_a_negative_number_is_a_nan_and_of_negative_zero_is_negative_zero() {
            // fsqrt
            let h = run_unary_f64(-1.0, &[0xd9, 0xfa]);
            assert!(st(&h, 0).is_nan());
            // fsqrt
            let h = run_unary_f64(-0.0, &[0xd9, 0xfa]);
            assert!(st(&h, 0).is_sign_negative() && st(&h, 0) == 0.0);
        }

        #[test]
        fn frndint_follows_the_rounding_control_field() {
            let table: [(f64, [f64; 4]); 6] = [
                (2.5, [2.0, 2.0, 3.0, 2.0]),
                (3.5, [4.0, 3.0, 4.0, 3.0]),
                (-2.5, [-2.0, -3.0, -2.0, -2.0]),
                (2.7, [3.0, 2.0, 3.0, 2.0]),
                (-2.7, [-3.0, -3.0, -2.0, -2.0]),
                (1e20, [1e20, 1e20, 1e20, 1e20]),
            ];
            for (value, expected) in table {
                for (control, want) in [RC_NEAREST, RC_DOWN, RC_UP, RC_TRUNCATE]
                    .into_iter()
                    .zip(expected)
                {
                    // fld qword ptr [rsi]; frndint
                    let h = run_ok(&program(&[&[0xdd, 0x06, 0xd9, 0xfc]]), |h| {
                        put_f64s(h, &[value]);
                        set_control_word(h, control);
                    });
                    assert_eq!(st(&h, 0), want, "{value} fcw={control:#x}");
                }
            }
        }

        #[test]
        fn frndint_of_a_small_negative_value_keeps_the_negative_sign_on_zero() {
            // frndint
            let h = run_unary_f64(-0.25, &[0xd9, 0xfc]);
            assert!(st(&h, 0).is_sign_negative() && st(&h, 0) == 0.0);
        }

        #[test]
        fn fscale_multiplies_by_two_to_the_truncated_st1() {
            // ST(1) = scale, ST(0) = value
            for (value, scale, expected) in [
                (3.0, 4.0, 48.0),
                (3.0, -2.0, 0.75),
                (3.0, 2.7, 12.0),
                (3.0, -2.7, 0.75),
                (3.0, 0.9, 3.0),
                (3.0, 0.0, 3.0),
                (-1.5, 3.0, -12.0),
            ] {
                // fscale
                let h = run_binary(&[0xd9, 0xfd], scale, value);
                assert_eq!(st(&h, 0), expected, "{value} * 2^trunc({scale})");
                assert_eq!(st(&h, 1), scale, "ST(1) is unchanged");
            }
        }

        #[test]
        fn fprem_leaves_the_truncated_remainder_and_the_low_quotient_bits() {
            // ST(1) = divisor, ST(0) = dividend; C0, C3, C1 = Q2, Q1, Q0 and C2 = 0 when complete.
            for (dividend, divisor, remainder, quotient) in [
                (5.5, 2.0, 1.5, 2u8),
                (-5.5, 2.0, -1.5, 2),
                (7.0, 2.0, 1.0, 3),
                (15.0, 2.0, 1.0, 7),
                (1.0, 4.0, 1.0, 0),
                (8.5, 2.0, 0.5, 4),
            ] {
                // fprem
                let h = run_binary(&[0xd9, 0xf8], divisor, dividend);
                assert_eq!(st(&h, 0), remainder, "{dividend} rem {divisor}");
                let q2 = (quotient >> 2) & 1 != 0;
                let q1 = (quotient >> 1) & 1 != 0;
                let q0 = quotient & 1 != 0;
                assert_eq!(h.cpu.fsw & C0 != 0, q2, "C0 = Q2 for quotient {quotient}");
                assert_eq!(h.cpu.fsw & C3 != 0, q1, "C3 = Q1 for quotient {quotient}");
                assert_eq!(h.cpu.fsw & C1 != 0, q0, "C1 = Q0 for quotient {quotient}");
                assert_eq!(
                    h.cpu.fsw & C2,
                    0,
                    "the reduction always completes in one step"
                );
            }
        }

        #[test]
        fn fprem1_rounds_the_quotient_to_nearest() {
            for (dividend, divisor, remainder, quotient) in [
                (5.5, 2.0, -0.5, 3u8),
                (7.0, 2.0, -1.0, 4),
                (4.0, 2.0, 0.0, 2),
                (5.0, 2.0, 1.0, 2),
            ] {
                // fprem1
                let h = run_binary(&[0xd9, 0xf5], divisor, dividend);
                assert_eq!(st(&h, 0), remainder, "{dividend} rem1 {divisor}");
                assert_eq!(h.cpu.fsw & C0 != 0, (quotient >> 2) & 1 != 0);
                assert_eq!(h.cpu.fsw & C3 != 0, (quotient >> 1) & 1 != 0);
                assert_eq!(h.cpu.fsw & C1 != 0, quotient & 1 != 0);
            }
        }

        #[test]
        fn fprem_does_not_pop_or_change_st1() {
            // fprem
            let h = run_binary(&[0xd9, 0xf8], 2.0, 5.5);
            assert_eq!(st(&h, 1), 2.0);
            assert_eq!(top(&h), 6);
        }

        #[test]
        fn fxtract_is_not_implemented_documented_limitation() {
            // fld1
            // fxtract
            let (_, stop) = run_with(&program(&[&[0xd9, 0xe8], &[0xd9, 0xf4]]), |_| {});
            assert_eq!(stop, invalid_opcode());
        }

        #[test]
        fn transcendentals_at_their_exact_points() {
            // f2xm1
            let h = run_unary_f64(0.0, &[0xd9, 0xf0]);
            assert_eq!(st(&h, 0), 0.0);
            // fsin
            let h = run_unary_f64(0.0, &[0xd9, 0xfe]);
            assert_eq!(st(&h, 0), 0.0);
            // fcos
            let h = run_unary_f64(0.0, &[0xd9, 0xff]);
            assert_eq!(st(&h, 0), 1.0);
            // fyl2x: ST(1) * log2(ST(0)), pops. 3 * log2(8) = 9
            // fyl2x
            let h = run_binary(&[0xd9, 0xf1], 3.0, 8.0);
            assert_eq!(st(&h, 0), 9.0);
            assert_eq!(top(&h), 7, "fyl2x pops");
            // fyl2xp1: ST(1) * log2(ST(0) + 1). 2 * log2(1 + 1) = 2
            // fyl2xp1
            let h = run_binary(&[0xd9, 0xf9], 2.0, 1.0);
            assert_eq!(st(&h, 0), 2.0);
            assert_eq!(top(&h), 7);
            // fpatan: atan2(ST(1), ST(0)), pops. atan2(0, 1) = 0
            // fpatan
            let h = run_binary(&[0xd9, 0xf3], 0.0, 1.0);
            assert_eq!(st(&h, 0), 0.0);
            assert_eq!(top(&h), 7);
        }

        #[test]
        fn transcendentals_agree_with_the_host_math_library_to_double_precision() {
            // f2xm1
            let h = run_unary_f64(0.5, &[0xd9, 0xf0]);
            assert_close(st(&h, 0), 2f64.powf(0.5) - 1.0);
            // fsin
            let h = run_unary_f64(1.0, &[0xd9, 0xfe]);
            assert_close(st(&h, 0), 1f64.sin());
            // fcos
            let h = run_unary_f64(1.0, &[0xd9, 0xff]);
            assert_close(st(&h, 0), 1f64.cos());
            // fpatan
            let h = run_binary(&[0xd9, 0xf3], 1.0, 1.0);
            assert_close(st(&h, 0), core::f64::consts::FRAC_PI_4);
            // fpatan
            let h = run_binary(&[0xd9, 0xf3], 1.0, -1.0);
            assert_close(st(&h, 0), 3.0 * core::f64::consts::FRAC_PI_4);
            // fyl2x
            let h = run_binary(&[0xd9, 0xf1], 1.0, 10.0);
            assert_close(st(&h, 0), core::f64::consts::LOG2_10);
        }

        #[test]
        fn fptan_replaces_st0_with_the_tangent_and_pushes_one() {
            // fptan
            let h = run_unary_f64(0.0, &[0xd9, 0xf2]);
            assert_eq!(st(&h, 0), 1.0);
            assert_eq!(st(&h, 1), 0.0);
            assert_eq!(top(&h), 6);
            assert_eq!(h.cpu.fsw & C2, 0, "C2 = 0: the operand was in range");
            // fptan
            let h = run_unary_f64(1.0, &[0xd9, 0xf2]);
            assert_close(st(&h, 1), 1f64.tan());
        }

        #[test]
        fn fsincos_leaves_the_cosine_on_top_of_the_sine() {
            // fsincos
            let h = run_unary_f64(0.0, &[0xd9, 0xfb]);
            assert_eq!(st(&h, 0), 1.0, "cos");
            assert_eq!(st(&h, 1), 0.0, "sin");
            assert_eq!(top(&h), 6);
            // fsincos
            let h = run_unary_f64(1.0, &[0xd9, 0xfb]);
            assert_close(st(&h, 0), 1f64.cos());
            assert_close(st(&h, 1), 1f64.sin());
        }

        #[test]
        fn trigonometric_instructions_clear_c2() {
            // fsin
            // fcos
            // fsincos
            // fptan
            for op in [&[0xd9, 0xfe], &[0xd9, 0xff], &[0xd9, 0xfb], &[0xd9, 0xf2]] {
                // fld1
                let mut h = Harness::new(&program(&[&[0xd9, 0xe8], op]));
                h.cpu.fsw |= C2;
                h.run_to_halt();
                assert_eq!(h.cpu.fsw & C2, 0, "{op:02x?}");
            }
        }

        #[test]
        fn constants_push_their_values() {
            let table: [(&[u8], f64); 7] = [
                // fld1
                (&[0xd9, 0xe8], 1.0),
                // fldz
                (&[0xd9, 0xee], 0.0),
                // fldpi
                (&[0xd9, 0xeb], core::f64::consts::PI),
                // fldl2e
                (&[0xd9, 0xea], core::f64::consts::LOG2_E),
                // fldl2t
                (&[0xd9, 0xe9], core::f64::consts::LOG2_10),
                // fldlg2
                (&[0xd9, 0xec], core::f64::consts::LOG10_2),
                // fldln2
                (&[0xd9, 0xed], core::f64::consts::LN_2),
            ];
            for (op, expected) in table {
                let h = run_ok(&program(&[op]), |_| {});
                assert_eq!(st(&h, 0), expected, "{op:02x?}");
                assert_eq!(top(&h), 7);
                assert!(!is_empty(&h, 7));
            }
        }

        #[test]
        fn fld1_and_fldz_are_exact_and_fldz_is_tagged_zero() {
            // fld1
            let h = run_ok(&program(&[&[0xd9, 0xe8]]), |_| {});
            assert_eq!(st_raw(&h, 0), EXT_ONE);
            // fldz
            let h = run_ok(&program(&[&[0xd9, 0xee]]), |_| {});
            assert_eq!(st_raw(&h, 0), EXT_ZERO);
        }

        #[test]
        fn constants_are_rounded_to_double_precision_documented_limitation() {
            // The architectural fldpi significand is 0xc90fdaa22168c235; here the low 11 bits are
            // zero because the value went through an f64.
            // fldpi
            let h = run_ok(&program(&[&[0xd9, 0xeb]]), |_| {});
            let significand = u64::from_le_bytes(st_raw(&h, 0)[..8].try_into().unwrap());
            assert_eq!(significand, 0xc90f_daa2_2168_c000);
            assert_ne!(significand, 0xc90f_daa2_2168_c235);
        }
    }

    mod examine_and_tags {
        use super::*;

        // fxam
        const FXAM: &[u8] = &[0xd9, 0xe5];

        fn examine(raw: [u8; 10]) -> u16 {
            run_with_raw(raw, FXAM).cpu.fsw & (C3 | C2 | C1 | C0)
        }

        const NORMAL: u16 = C2;
        const ZERO: u16 = C3;
        const NAN: u16 = C0;
        const INFINITY: u16 = C2 | C0;
        const DENORMAL: u16 = C3 | C2;
        const EMPTY: u16 = C3 | C0;
        const UNSUPPORTED: u16 = 0;

        #[test]
        fn fxam_classifies_every_format_for_positive_values() {
            for (raw, expected, name) in [
                (EXT_ONE, NORMAL, "normal"),
                (EXT_PI_FULL, NORMAL, "normal"),
                (EXT_ZERO, ZERO, "zero"),
                (EXT_INFINITY, INFINITY, "infinity"),
                (EXT_QUIET_NAN, NAN, "quiet NaN"),
                (EXT_SIGNALING_NAN, NAN, "signaling NaN"),
                (EXT_QUIET_NAN_PAYLOAD, NAN, "NaN with payload"),
                (EXT_DENORMAL, DENORMAL, "denormal"),
                (EXT_UNNORMAL, UNSUPPORTED, "unnormal"),
                (EXT_PSEUDO_INFINITY, UNSUPPORTED, "pseudo-infinity"),
                (EXT_PSEUDO_NAN, UNSUPPORTED, "pseudo-NaN"),
            ] {
                assert_eq!(examine(raw), expected, "{name}");
            }
        }

        #[test]
        fn fxam_reports_the_sign_in_c1() {
            for (raw, expected) in [
                (EXT_NEG_TWO, NORMAL | C1),
                (EXT_NEG_ZERO, ZERO | C1),
                (EXT_NEG_INFINITY, INFINITY | C1),
                (EXT_NEG_DENORMAL, DENORMAL | C1),
                ([0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0xff], NAN | C1),
            ] {
                assert_eq!(examine(raw), expected, "{raw:02x?}");
            }
        }

        #[test]
        fn fxam_of_an_empty_register_is_c3_c0() {
            let h = run_ok(&program(&[FXAM]), |_| {});
            assert_eq!(h.cpu.fsw & (C3 | C2 | C0), EMPTY);
            // fld1; fstp st(0)
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xdd, 0xd8], FXAM]), |_| {});
            assert_eq!(
                h.cpu.fsw & (C3 | C2 | C0),
                EMPTY,
                "a popped register is empty again"
            );
        }

        #[test]
        fn fxam_of_an_empty_register_still_reports_the_stale_sign() {
            let mut h = Harness::new(&program(&[FXAM]));
            h.cpu.st[0] = EXT_NEG_TWO;
            h.run_to_halt();
            assert_eq!(h.cpu.fsw & (C3 | C2 | C1 | C0), EMPTY | C1);
        }

        #[test]
        fn fxam_treats_a_denormal_loaded_from_a_32_or_64_bit_operand_as_normal() {
            // fld dword ptr [rsi]
            let h = run_ok(&program(&[&[0xd9, 0x06], FXAM]), |h| {
                h.write(DATA, &1u32.to_le_bytes())
            });
            assert_eq!(h.cpu.fsw & (C3 | C2 | C0), NORMAL);
            // fld qword ptr [rsi]
            let h = run_ok(&program(&[&[0xdd, 0x06], FXAM]), |h| {
                put_f64s(h, &[f64::from_bits(1)])
            });
            assert_eq!(h.cpu.fsw & (C3 | C2 | C0), NORMAL);
        }

        #[test]
        fn fxam_looks_at_st0_only_and_pops_nothing() {
            // fldz; fld1
            let h = run_ok(&program(&[&[0xd9, 0xee, 0xd9, 0xe8], FXAM]), |_| {});
            assert_eq!(h.cpu.fsw & (C3 | C2 | C0), NORMAL);
            assert_eq!(top(&h), 6);
        }

        #[test]
        fn fxam_overwrites_the_previous_condition_codes() {
            let mut h = Harness::new(&program(&[LOAD_RAW_FROM_DATA, FXAM]));
            init(&mut h);
            h.write(DATA, &EXT_ONE);
            h.cpu.fsw |= C0 | C3;
            h.run_to_halt();
            assert_eq!(h.cpu.fsw & (C3 | C2 | C1 | C0), NORMAL);
        }

        fn tag_word_after_loading(raw: [u8; 10]) -> u16 {
            // fnstenv [rdi]
            let h = run_ok(&program(&[LOAD_RAW_FROM_DATA, &[0xd9, 0x37]]), |h| {
                h.write(DATA, &raw)
            });
            u16::from_le_bytes(h.read(OUT + 8, 2).try_into().unwrap())
        }

        // The value lands in physical register 7 (bits 15:14); the other seven are empty (11).
        const OTHERS_EMPTY: u16 = 0x3fff;

        #[test]
        fn the_tag_of_a_normal_value_is_valid() {
            for raw in [EXT_ONE, EXT_PI_FULL, EXT_NEG_TWO, EXT_ONE_PLUS_ULP] {
                assert_eq!(tag_word_after_loading(raw), OTHERS_EMPTY, "{raw:02x?}");
            }
        }

        #[test]
        fn the_tag_of_a_zero_is_zero() {
            for raw in [EXT_ZERO, EXT_NEG_ZERO] {
                assert_eq!(tag_word_after_loading(raw), OTHERS_EMPTY | 0b01 << 14);
            }
        }

        #[test]
        fn the_tag_of_special_values_is_special() {
            for (raw, name) in [
                (EXT_INFINITY, "infinity"),
                (EXT_NEG_INFINITY, "-infinity"),
                (EXT_QUIET_NAN, "NaN"),
                (EXT_SIGNALING_NAN, "sNaN"),
                (EXT_DENORMAL, "denormal"),
                (EXT_NEG_DENORMAL, "-denormal"),
                (EXT_PSEUDO_DENORMAL, "pseudo-denormal"),
                (EXT_UNNORMAL, "unnormal"),
                (EXT_PSEUDO_NAN, "pseudo-NaN"),
            ] {
                assert_eq!(
                    tag_word_after_loading(raw),
                    OTHERS_EMPTY | 0b10 << 14,
                    "{name}"
                );
            }
        }

        #[test]
        fn the_tag_word_of_an_empty_stack_is_all_ones() {
            // fnstenv [rdi]
            let h = run_ok(&program(&[&[0xd9, 0x37]]), |_| {});
            assert_eq!(h.read(OUT + 8, 2), [0xff, 0xff]);
        }

        #[test]
        fn tags_are_indexed_by_physical_register() {
            // fld1 -> physical 7 (valid), fldz -> physical 6 (zero); the rest are empty.
            // fld1; fldz; fnstenv [rdi]
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xee, 0xd9, 0x37]]), |_| {});
            assert_eq!(
                u16::from_le_bytes(h.read(OUT + 8, 2).try_into().unwrap()),
                0b00_01_11_11_11_11_11_11
            );
        }

        #[test]
        fn a_freed_register_is_tagged_empty() {
            // fld1; fld1; ffree st(1); fnstenv [rdi]
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xdd, 0xc1, 0xd9, 0x37]]),
                |_| {},
            );
            // TOP = 6, so ST(1) is physical register 7: freed (11); physical 6 stays valid (00)
            assert_eq!(
                u16::from_le_bytes(h.read(OUT + 8, 2).try_into().unwrap()),
                0b11_00_11_11_11_11_11_11
            );
        }
    }

    mod control_and_status {
        use super::*;

        #[test]
        fn fnstcw_stores_the_power_on_control_word() {
            // fnstcw [rdi]
            let h = run_ok(&program(&[&[0xd9, 0x3f]]), |h| h.write(OUT, &[0xaa; 4]));
            assert_eq!(
                h.read(OUT, 4),
                [0x7f, 0x03, 0xaa, 0xaa],
                "exactly two bytes"
            );
        }

        #[test]
        fn fldcw_then_fnstcw_round_trips_any_value() {
            for value in [0x037fu16, 0x027f, 0x0f7f, 0x1234, 0x0000, 0xffff] {
                // fldcw [rsi]; fnstcw [rdi]
                let h = run_ok(&program(&[&[0xd9, 0x2e, 0xd9, 0x3f]]), |h| {
                    h.write(DATA, &value.to_le_bytes())
                });
                assert_eq!(h.read(OUT, 2), value.to_le_bytes());
                assert_eq!(h.cpu.fcw, value);
            }
        }

        #[test]
        fn fldcw_does_not_touch_the_status_word_or_the_stack() {
            // fld1; fldcw [rsi]
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0x2e]]), |h| {
                h.write(DATA, &0x0c7fu16.to_le_bytes())
            });
            assert_eq!(top(&h), 7);
            assert_eq!(h.cpu.ftw, 0x80);
        }

        #[test]
        fn fnstsw_to_ax_keeps_the_upper_bits_of_rax() {
            // fld1; fld1; fnstsw ax
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xdf, 0xe0]]), |h| {
                h.cpu.gpr[RAX] = 0xdead_beef_0000_0000
            });
            assert_eq!(h.cpu.gpr[RAX], 0xdead_beef_0000_0000 | (6 << 11));
        }

        #[test]
        fn fnstsw_to_memory_stores_two_bytes() {
            // fld1; fnstsw [rdi]
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xdd, 0x3f]]), |h| {
                h.write(OUT, &[0xaa; 4])
            });
            assert_eq!(
                h.read(OUT, 4),
                [0x00, 0x38, 0xaa, 0xaa],
                "TOP = 7 is 0x3800"
            );
        }

        #[test]
        fn fnstsw_reports_condition_codes_and_top() {
            // fld1; fld1; fcom st(1); fnstsw ax
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd8, 0xd1, 0xdf, 0xe0]]),
                |_| {},
            );
            assert_eq!(
                h.cpu.gpr[RAX] as u16,
                (6 << 11) | C3,
                "equal: C3, with TOP = 6"
            );
        }

        #[test]
        fn fnclex_clears_exception_flags_and_busy_but_keeps_top_and_condition_codes() {
            // fnclex
            let mut h = Harness::new(&program(&[&[0xdb, 0xe2]]));
            h.cpu.fsw = 0xffff;
            h.run_to_halt();
            assert_eq!(h.cpu.fsw, 0x7f00);
        }

        #[test]
        fn fninit_resets_control_status_and_tags() {
            // fninit
            let mut h = Harness::new(&program(&[&[0xdb, 0xe3]]));
            h.cpu.fcw = 0x0c00;
            h.cpu.fsw = 0xffff;
            h.cpu.ftw = 0xff;
            h.run_to_halt();
            assert_eq!((h.cpu.fcw, h.cpu.fsw, h.cpu.ftw), (0x037f, 0, 0));
        }

        #[test]
        fn fninit_leaves_the_register_contents_alone() {
            // fninit
            let mut h = Harness::new(&program(&[&[0xdb, 0xe3]]));
            h.cpu.st[3] = EXT_PI_FULL;
            h.run_to_halt();
            assert_eq!(h.cpu.st[3], EXT_PI_FULL);
        }

        #[test]
        fn emms_empties_all_tags_without_moving_top() {
            // fld1; fld1; fld1; emms
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd9, 0xe8, 0x0f, 0x77]]),
                |_| {},
            );
            assert_eq!(h.cpu.ftw, 0);
            assert_eq!(top(&h), 5);
        }

        #[test]
        fn emms_keeps_control_word_and_status_flags() {
            // emms
            let mut h = Harness::new(&program(&[&[0x0f, 0x77]]));
            h.cpu.fcw = 0x0c7f;
            h.cpu.fsw = C2;
            h.run_to_halt();
            assert_eq!((h.cpu.fcw, h.cpu.fsw), (0x0c7f, C2));
        }

        #[test]
        fn mmx_register_forms_execute_and_put_the_stack_in_mmx_state() {
            // Detailed MMX coverage is in the tests in cpu/sse.rs; this pins the x87-visible effects.
            // movd mm0, eax; paddb mm0, mm0; hlt
            let h = run_ok(&program(&[&[0x0f, 0x6e, 0xc0, 0x0f, 0xfc, 0xc0]]), |h| {
                h.cpu.gpr[0] = 0x0101_0101;
                h.cpu.fsw = 5 << 11;
            });
            assert_eq!(
                u64::from_le_bytes(h.cpu.st[0][..8].try_into().unwrap()),
                0x0202_0202
            );
            assert_eq!(
                h.cpu.st[0][8..10],
                [0xff, 0xff],
                "an MMX write sets the exponent field"
            );
            assert_eq!(h.cpu.fsw & (7 << 11), 0, "TOP is 0");
            assert_eq!(h.cpu.ftw, 0xff, "every register is tagged valid");
        }
    }

    mod environment_and_state_images {
        use super::*;

        #[test]
        fn fnstenv_writes_a_28_byte_protected_mode_image() {
            // fld1; fldz; fnstenv [rdi]
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xee, 0xd9, 0x37]]), |h| {
                h.write(OUT, &[0xaa; 40])
            });
            assert_eq!(h.read(OUT, 2), 0x037fu16.to_le_bytes(), "control word");
            assert_eq!(
                h.read(OUT + 4, 2),
                (6u16 << 11).to_le_bytes(),
                "status word with TOP = 6"
            );
            assert_eq!(h.read(OUT + 8, 2), 0x1fffu16.to_le_bytes(), "tag word");
            assert_eq!(
                h.read(OUT + 12, 16),
                [0; 16],
                "instruction and data pointers are not tracked"
            );
            assert_eq!(
                h.read(OUT + 28, 12),
                [0xaa; 12],
                "exactly 28 bytes are written"
            );
        }

        #[test]
        fn fnstenv_masks_all_exceptions_afterwards() {
            // fldcw [rsi]; fnstenv [rdi]
            let h = run_ok(&program(&[&[0xd9, 0x2e, 0xd9, 0x37]]), |h| {
                h.write(DATA, &0x0340u16.to_le_bytes())
            });
            assert_eq!(
                h.read(OUT, 2),
                0x0340u16.to_le_bytes(),
                "the image has the unmasked word"
            );
            assert_eq!(h.cpu.fcw, 0x037f, "all six masks set afterwards");
        }

        #[test]
        fn fnstenv_includes_the_condition_codes() {
            // fld1; fld1; fcom st(1); fnstenv [rdi]
            let h = run_ok(
                &program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xd8, 0xd1, 0xd9, 0x37]]),
                |_| {},
            );
            assert_eq!(h.read(OUT + 4, 2), ((6u16 << 11) | C3).to_le_bytes());
        }

        #[test]
        fn fldenv_loads_control_status_and_tags() {
            // TOP = 5; physical 5 valid, 6 zero, 7 special, the rest empty.
            let tag_word: u16 = 0b10_01_00_11_11_11_11_11;
            // fldenv [rsi]
            let h = run_ok(&program(&[&[0xd9, 0x26]]), |h| {
                h.write(DATA, &0x027fu32.to_le_bytes());
                h.write(DATA + 4, &(((5u16 << 11) | C0) as u32).to_le_bytes());
                h.write(DATA + 8, &(tag_word as u32).to_le_bytes());
            });
            assert_eq!(h.cpu.fcw, 0x027f);
            assert_eq!(h.cpu.fsw, (5 << 11) | C0);
            assert_eq!(
                h.cpu.ftw, 0b1110_0000,
                "only emptiness survives into the abridged tags"
            );
            assert_eq!(top(&h), 5);
        }

        #[test]
        fn fldenv_does_not_load_register_contents() {
            // fldenv [rsi]
            let mut h = Harness::new(&program(&[&[0xd9, 0x26]]));
            init(&mut h);
            h.cpu.st[7] = EXT_PI_FULL;
            h.write(DATA, &[0xaa; 28]);
            h.run_to_halt();
            assert_eq!(h.cpu.st[7], EXT_PI_FULL);
        }

        #[test]
        fn fnstenv_then_fldenv_round_trips_the_environment() {
            let h = run_ok(
                // fld1; fldz; fld1; fcom st(2); fnstenv [rdi]; fninit; fldenv [rdi]
                &program(&[&[
                    0xd9, 0xe8, 0xd9, 0xee, 0xd9, 0xe8, 0xd8, 0xd2, 0xd9, 0x37, 0xdb, 0xe3, 0xd9,
                    0x27,
                ]]),
                |_| {},
            );
            assert_eq!(h.cpu.fcw, 0x037f);
            assert_eq!(top(&h), 5);
            assert_eq!(h.cpu.fsw & (C0 | C2 | C3), C3);
            assert_eq!(h.cpu.ftw, 0b1110_0000);
        }

        #[test]
        fn fnsave_writes_the_environment_then_eight_registers_in_stack_order() {
            let h = run_ok(
                // fld1; fldz; fld qword ptr [rsi]; fnsave [rdi]
                &program(&[&[0xd9, 0xe8, 0xd9, 0xee, 0xdd, 0x06, 0xdd, 0x37]]),
                |h| {
                    put_f64s(h, &[-2.0]);
                    h.write(OUT, &[0xaa; 120]);
                },
            );
            assert_eq!(h.read(OUT, 2), 0x037fu16.to_le_bytes());
            assert_eq!(h.read(OUT + 4, 2), (5u16 << 11).to_le_bytes());
            assert_eq!(
                h.read(OUT + 8, 2),
                0b00_01_00_11_11_11_11_11u16.to_le_bytes()
            );
            assert_eq!(h.read(OUT + 28, 10), EXT_NEG_TWO, "ST(0)");
            assert_eq!(h.read(OUT + 38, 10), EXT_ZERO, "ST(1)");
            assert_eq!(h.read(OUT + 48, 10), EXT_ONE, "ST(2)");
            assert_eq!(h.read(OUT + 108, 12), [0xaa; 12], "the image is 108 bytes");
        }

        #[test]
        fn fnsave_reinitialises_the_fpu() {
            // fldcw [rsi]; fld1; fld1; fnsave [rdi]
            let h = run_ok(
                &program(&[&[0xd9, 0x2e, 0xd9, 0xe8, 0xd9, 0xe8, 0xdd, 0x37]]),
                |h| {
                    h.write(DATA, &0x0c00u16.to_le_bytes());
                },
            );
            assert_eq!((h.cpu.fcw, h.cpu.fsw, h.cpu.ftw), (0x037f, 0, 0));
            assert_eq!(
                h.read(OUT, 2),
                0x0c00u16.to_le_bytes(),
                "the image has the old control word"
            );
        }

        #[test]
        fn frstor_restores_environment_and_registers() {
            let h = run_ok(
                // fld1; fldz; fld qword ptr [rsi]; fnsave [rdi]; frstor [rdi]
                &program(&[&[0xd9, 0xe8, 0xd9, 0xee, 0xdd, 0x06, 0xdd, 0x37, 0xdd, 0x27]]),
                |h| put_f64s(h, &[-2.0]),
            );
            assert_eq!(top(&h), 5);
            assert_eq!(h.cpu.ftw, 0b1110_0000);
            assert_eq!(st_raw(&h, 0), EXT_NEG_TWO);
            assert_eq!(st_raw(&h, 1), EXT_ZERO);
            assert_eq!(st_raw(&h, 2), EXT_ONE);
            assert_eq!(h.cpu.fcw, 0x037f);
        }

        #[test]
        fn frstor_places_registers_by_the_restored_top() {
            // A hand-built image with TOP = 3: ST(0) lives in physical register 3.
            let mut image = vec![0u8; 108];
            image[0..2].copy_from_slice(&0x037fu16.to_le_bytes());
            image[4..6].copy_from_slice(&(3u16 << 11).to_le_bytes());
            // physical registers 3 and 4 in use (tag 00), the rest empty (tag 11)
            image[8..10].copy_from_slice(&0xfc3fu16.to_le_bytes());
            image[28..38].copy_from_slice(&EXT_THREE);
            image[38..48].copy_from_slice(&EXT_ONE);
            // frstor [rsi]
            let h = run_ok(&program(&[&[0xdd, 0x26]]), |h| h.write(DATA, &image));
            assert_eq!(h.cpu.st[3], EXT_THREE);
            assert_eq!(h.cpu.st[4], EXT_ONE);
            assert_eq!(h.cpu.ftw, 0b0001_1000, "physical 3 and 4 are in use");
        }

        #[test]
        fn frstor_preserves_extended_precision_bits() {
            let h = run_ok(
                // fld tbyte ptr [rsi]; fnsave [rdi]; fninit; frstor [rdi]
                &program(&[&[0xdb, 0x2e, 0xdd, 0x37, 0xdb, 0xe3, 0xdd, 0x27]]),
                |h| h.write(DATA, &EXT_ONE_PLUS_ULP),
            );
            assert_eq!(st_raw(&h, 0), EXT_ONE_PLUS_ULP);
        }

        #[test]
        fn state_image_loads_report_page_faults() {
            // fldenv [rsi]
            // frstor [rsi]
            for code in [program(&[&[0xd9, 0x26]]), program(&[&[0xdd, 0x26]])] {
                let (h, stop) = run_with(&code, |h| {
                    unmap_region(h, UNMAPPED_REGION);
                    h.cpu.gpr[RSI] = UNMAPPED_REGION;
                });
                assert_eq!(stop, page_fault_not_present_read());
                assert_eq!(h.cpu.rip, CODE_BASE);
                assert_eq!(h.cpu.fcw, 0x037f, "nothing was loaded");
            }
        }

        #[test]
        fn state_image_stores_report_page_faults_and_change_nothing() {
            const NOT_PRESENT_WRITE: u32 = 2;
            // fnstenv [rdi]
            // fnsave [rdi]
            for code in [program(&[&[0xd9, 0x37]]), program(&[&[0xdd, 0x37]])] {
                let (h, stop) = run_with(&code, |h| {
                    unmap_region(h, UNMAPPED_REGION);
                    h.cpu.gpr[RDI] = UNMAPPED_REGION;
                    h.cpu.fcw = 0x0340;
                });
                assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(NOT_PRESENT_WRITE)));
                assert_eq!(h.cpu.rip, CODE_BASE);
                assert_eq!(
                    h.cpu.fcw, 0x0340,
                    "the masks are set only after the image was stored"
                );
            }
        }
    }

    mod fxsave_and_fxrstor {
        use super::*;

        // fxsave [rdi]
        const FXSAVE: &[u8] = &[0x0f, 0xae, 0x07];
        // fxrstor [rsi]
        const FXRSTOR: &[u8] = &[0x0f, 0xae, 0x0e];
        // fxsave64 [rdi]
        const FXSAVE64: &[u8] = &[0x48, 0x0f, 0xae, 0x07];
        // fxrstor64 [rsi]
        const FXRSTOR64: &[u8] = &[0x48, 0x0f, 0xae, 0x0e];
        const IMAGE_BYTES: usize = 512;

        fn xmm_pattern(index: usize) -> u128 {
            (0x1111_2222_3333_4444u128 << 64) | ((0xa0 + index as u128) << 8) | index as u128
        }

        /// FPU state with a few registers pushed, distinctive XMM registers and a non-default MXCSR.
        fn dirty_state(harness: &mut Harness) {
            for index in 0..16 {
                harness.cpu.xmm[index] = xmm_pattern(index);
            }
            harness.cpu.mxcsr = 0x1f80 | 0x2000;
        }

        // fld1; fldz; fld qword ptr [rsi]
        const PUSH_THREE: &[u8] = &[0xd9, 0xe8, 0xd9, 0xee, 0xdd, 0x06];

        fn saved_image(extra_setup: impl FnOnce(&mut Harness)) -> (Harness, Vec<u8>) {
            let h = run_ok(&program(&[PUSH_THREE, FXSAVE]), |h| {
                put_f64s(h, &[-2.0]);
                h.write(OUT, &[0xaa; IMAGE_BYTES]);
                dirty_state(h);
                extra_setup(h);
            });
            let image = h.read(OUT, IMAGE_BYTES);
            (h, image)
        }

        fn u16_at(image: &[u8], offset: usize) -> u16 {
            u16::from_le_bytes(image[offset..offset + 2].try_into().unwrap())
        }

        fn u32_at(image: &[u8], offset: usize) -> u32 {
            u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap())
        }

        #[test]
        fn fxsave_writes_the_control_and_status_words() {
            let (_, image) = saved_image(|_| {});
            assert_eq!(u16_at(&image, 0), 0x037f, "FCW");
            assert_eq!(u16_at(&image, 2), 5 << 11, "FSW with TOP = 5");
        }

        #[test]
        fn fxsave_writes_the_abridged_tag_word_by_physical_register() {
            let (_, image) = saved_image(|_| {});
            // physical registers 5, 6, 7 hold values
            assert_eq!(image[4], 0b1110_0000);
        }

        #[test]
        fn fxsave_writes_mxcsr_and_the_mxcsr_mask() {
            let (_, image) = saved_image(|_| {});
            assert_eq!(u32_at(&image, 24), 0x3f80, "MXCSR");
            assert_eq!(
                u32_at(&image, 28),
                0x0002_ffff,
                "MXCSR_MASK: the writable bits, MM (17) among them"
            );
        }

        #[test]
        fn fxsave_writes_the_stack_registers_in_st_order_at_16_byte_strides() {
            let (_, image) = saved_image(|_| {});
            assert_eq!(image[32..42], EXT_NEG_TWO, "ST(0) at offset 32");
            assert_eq!(image[48..58], EXT_ZERO, "ST(1) at offset 48");
            assert_eq!(image[64..74], EXT_ONE, "ST(2) at offset 64");
        }

        #[test]
        fn fxsave_writes_each_xmm_register_at_offset_160_plus_16_k() {
            let (_, image) = saved_image(|_| {});
            for index in 0..16 {
                let start = 160 + 16 * index;
                assert_eq!(
                    u128::from_le_bytes(image[start..start + 16].try_into().unwrap()),
                    xmm_pattern(index),
                    "xmm{index}"
                );
            }
        }

        #[test]
        fn fxsave_does_not_write_past_the_512_byte_area() {
            let h = run_ok(&program(&[FXSAVE]), |h| {
                h.write(OUT + IMAGE_BYTES as u64 - 16, &[0xaa; 48])
            });
            assert_eq!(h.read(OUT + IMAGE_BYTES as u64, 32), [0xaa; 32]);
        }

        #[test]
        fn fxsave_leaves_the_fpu_state_unchanged() {
            let (h, _) = saved_image(|_| {});
            assert_eq!(h.cpu.fcw, 0x037f);
            assert_eq!(top(&h), 5);
            assert_eq!(h.cpu.ftw, 0b1110_0000);
            assert_eq!(st(&h, 0), -2.0);
        }

        #[test]
        fn fxsave_and_fxsave64_produce_the_same_image() {
            let (_, narrow) = saved_image(|_| {});
            let h = run_ok(&program(&[PUSH_THREE, FXSAVE64]), |h| {
                put_f64s(h, &[-2.0]);
                h.write(OUT, &[0xaa; IMAGE_BYTES]);
                dirty_state(h);
            });
            assert_eq!(h.read(OUT, IMAGE_BYTES), narrow);
        }

        // fxrstor [rdi]
        const FXRSTOR_FROM_RDI: &[u8] = &[0x0f, 0xae, 0x0f];

        /// The architectural state fxsave/fxrstor cover.
        fn fpu_state(h: &Harness) -> (u16, u16, u16, [[u8; 10]; 8], [u128; 16], u32) {
            (
                h.cpu.fcw,
                h.cpu.fsw,
                h.cpu.ftw,
                h.cpu.st,
                h.cpu.xmm,
                h.cpu.mxcsr,
            )
        }

        #[test]
        fn fxrstor_restores_everything_fxsave_wrote() {
            // fninit
            let mut h = Harness::new(&program(&[
                PUSH_THREE,
                FXSAVE,
                &[0xdb, 0xe3],
                FXRSTOR_FROM_RDI,
            ]));
            init(&mut h);
            put_f64s(&mut h, &[-2.0]);
            dirty_state(&mut h);
            // fld, fld, fld, fxsave
            assert_eq!(h.run(4), Stop::Limit);
            let saved = fpu_state(&h);
            h.cpu.xmm = [0; 16];
            h.cpu.mxcsr = 0x1f80;
            h.cpu.st = [[0; 10]; 8];
            h.cpu.fcw = 0x0c00;
            h.cpu.fsw = C2;
            h.cpu.ftw = 0;
            h.run_to_halt();
            assert_eq!(fpu_state(&h), saved);
        }

        #[test]
        fn fxrstor_loads_a_hand_built_image() {
            let mut image = vec![0u8; IMAGE_BYTES];
            image[0..2].copy_from_slice(&0x027fu16.to_le_bytes());
            image[2..4].copy_from_slice(&((2u16 << 11) | C1).to_le_bytes());
            image[4] = 0b0000_0110;
            image[24..28].copy_from_slice(&0x1fa0u32.to_le_bytes());
            image[32..42].copy_from_slice(&EXT_THREE);
            image[48..58].copy_from_slice(&EXT_ONE);
            image[160..176]
                .copy_from_slice(&0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128.to_le_bytes());
            image[160 + 15 * 16..160 + 16 * 16].copy_from_slice(&u128::MAX.to_le_bytes());
            let h = run_ok(&program(&[FXRSTOR]), |h| h.write(DATA, &image));
            assert_eq!(h.cpu.fcw, 0x027f);
            assert_eq!(h.cpu.fsw, (2 << 11) | C1);
            assert_eq!(h.cpu.ftw, 0b0000_0110);
            assert_eq!(h.cpu.mxcsr, 0x1fa0);
            assert_eq!(
                h.cpu.st[2], EXT_THREE,
                "ST(0) goes to physical register TOP = 2"
            );
            assert_eq!(h.cpu.st[3], EXT_ONE);
            assert_eq!(h.cpu.xmm[0], 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
            assert_eq!(h.cpu.xmm[15], u128::MAX);
            assert_eq!(h.cpu.xmm[1], 0);
        }

        #[test]
        fn fxrstor64_loads_the_same_layout() {
            let mut image = vec![0u8; IMAGE_BYTES];
            image[0..2].copy_from_slice(&0x037fu16.to_le_bytes());
            image[24..28].copy_from_slice(&0x1f80u32.to_le_bytes());
            image[160..176].copy_from_slice(&7u128.to_le_bytes());
            let h = run_ok(&program(&[FXRSTOR64]), |h| h.write(DATA, &image));
            assert_eq!(h.cpu.xmm[0], 7);
        }

        #[test]
        fn fxrstor_accepts_every_mxcsr_bit_below_16() {
            for bit in 0..16 {
                let mut image = vec![0u8; IMAGE_BYTES];
                image[24..28].copy_from_slice(&(1u32 << bit).to_le_bytes());
                let h = run_ok(&program(&[FXRSTOR]), |h| h.write(DATA, &image));
                assert_eq!(h.cpu.mxcsr, 1 << bit);
            }
        }

        #[test]
        fn fxrstor_with_reserved_mxcsr_bits_is_gp0_and_commits_nothing() {
            // bit 17 is MM, misaligned SSE mode
            for bit in (16..32).filter(|&bit| bit != 17) {
                let mut image = vec![0u8; IMAGE_BYTES];
                image[0..2].copy_from_slice(&0x0c00u16.to_le_bytes());
                image[24..28].copy_from_slice(&(0x1f80u32 | 1 << bit).to_le_bytes());
                image[160..176].copy_from_slice(&0xdeadu128.to_le_bytes());
                let mut before = None;
                let h = assert_faults(
                    &program(&[FXRSTOR]),
                    |h| {
                        h.write(DATA, &image);
                        dirty_state(h);
                        before = Some(fpu_state(h));
                    },
                    general_protection_zero(),
                );
                assert_eq!(
                    Some(fpu_state(&h)),
                    before,
                    "bit {bit}: no register may change"
                );
            }
        }

        #[test]
        fn misaligned_fxsave_and_fxrstor_are_gp0_for_every_offset() {
            for code in [FXSAVE, FXSAVE64] {
                for misalignment in 1..16 {
                    let h = assert_faults(
                        &program(&[code]),
                        |h| {
                            h.write(OUT, &[0xaa; 2 * IMAGE_BYTES]);
                            h.cpu.gpr[RDI] = OUT + misalignment;
                        },
                        general_protection_zero(),
                    );
                    assert_eq!(
                        h.read(OUT, 2 * IMAGE_BYTES),
                        [0xaa; 2 * IMAGE_BYTES],
                        "offset {misalignment}: nothing may be written"
                    );
                }
            }
            for code in [FXRSTOR, FXRSTOR64] {
                for misalignment in 1..16 {
                    let mut before = None;
                    let h = assert_faults(
                        &program(&[code]),
                        |h| {
                            dirty_state(h);
                            h.cpu.gpr[RSI] = DATA + misalignment;
                            before = Some(fpu_state(h));
                        },
                        general_protection_zero(),
                    );
                    assert_eq!(Some(fpu_state(&h)), before, "offset {misalignment}");
                }
            }
        }

        #[test]
        fn aligned_addresses_other_than_page_start_work() {
            for offset in [16u64, 48, 0x70, 0xe00] {
                let h = run_ok(&program(&[FXSAVE]), |h| h.cpu.gpr[RDI] = OUT + offset);
                assert_eq!(
                    u16_at(&h.read(OUT + offset, 4), 0),
                    0x037f,
                    "offset {offset:#x}"
                );
            }
        }

        #[test]
        fn the_alignment_check_looks_at_the_linear_address_including_a_segment_base() {
            // fxsave gs:[rdi]
            let h = assert_faults(
                &program(&[&[0x65, 0x0f, 0xae, 0x07]]),
                |h| {
                    h.cpu.seg[SEG_GS].base = 8;
                    h.cpu.gpr[RDI] = OUT;
                },
                general_protection_zero(),
            );
            assert_eq!(h.read(OUT, 4), [0, 0, 0, 0]);
            // fxsave gs:[rdi]
            let h = run_ok(&program(&[&[0x65, 0x0f, 0xae, 0x07]]), |h| {
                h.cpu.seg[SEG_GS].base = 16;
                h.cpu.gpr[RDI] = OUT;
            });
            assert_eq!(u16_at(&h.read(OUT + 16, 2), 0), 0x037f);
        }

        #[test]
        fn fxsave_and_fxrstor_with_ts_set_are_device_not_available() {
            for code in [FXSAVE, FXRSTOR] {
                assert_faults(
                    &program(&[code]),
                    |h| h.cpu.cr0 |= CR0_TS,
                    device_not_available(),
                );
            }
        }

        #[test]
        fn fxsave_page_fault_in_the_second_half_writes_nothing() {
            // The 512-byte area starts 256 bytes below an unmapped region.
            let start = UNMAPPED_REGION - 256;
            let h = assert_faults(
                &program(&[FXSAVE]),
                |h| {
                    unmap_region(h, UNMAPPED_REGION);
                    h.write(start, &[0xaa; 256]);
                    h.cpu.gpr[RDI] = start;
                },
                fault(VECTOR_PAGE_FAULT, Some(2)),
            );
            assert_eq!(
                h.read(start, 256),
                [0xaa; 256],
                "the probe faults before any byte is stored"
            );
        }

        #[test]
        fn fxrstor_page_fault_midway_leaves_every_register_untouched() {
            // fcw..xmm5 (the first 256 bytes) are readable, xmm6 is not: nothing may be committed.
            let start = UNMAPPED_REGION - 256;
            let mut before = None;
            let h = assert_faults(
                &program(&[FXRSTOR]),
                |h| {
                    unmap_region(h, UNMAPPED_REGION);
                    let mut image = vec![0u8; 256];
                    image[0..2].copy_from_slice(&0x0c00u16.to_le_bytes());
                    image[24..28].copy_from_slice(&0x1fa0u32.to_le_bytes());
                    for index in 0..6 {
                        let at = 160 + 16 * index;
                        image[at..at + 16].copy_from_slice(&(0x1000 + index as u128).to_le_bytes());
                    }
                    h.write(start, &image);
                    h.cpu.gpr[RSI] = start;
                    dirty_state(h);
                    before = Some(fpu_state(h));
                },
                fault(VECTOR_PAGE_FAULT, Some(0)),
            );
            assert_eq!(Some(fpu_state(&h)), before);
            assert_eq!(h.cpu.cr2, UNMAPPED_REGION);
        }

        #[test]
        fn fxsave_and_fxrstor_are_allowed_at_cpl3() {
            let mut h = Harness::new(&program(&[FXSAVE, FXRSTOR_FROM_RDI]));
            init(&mut h);
            enter_user_mode_for_fpu(&mut h);
            assert_eq!(h.run(2), Stop::Limit);
            assert_eq!(h.cpu.rip, CODE_BASE + 6);
        }

        /// Makes the identity map user-accessible and switches to CPL 3.
        fn enter_user_mode_for_fpu(harness: &mut Harness) {
            const PTE_USER: u64 = 4;
            const ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
            let pml4 = harness.cpu.cr3 & ADDRESS_MASK;
            let pml4_entry = harness.read_u64(pml4);
            harness.write_u64(pml4, pml4_entry | PTE_USER);
            let pdpt = pml4_entry & ADDRESS_MASK;
            for gib in 0..4 {
                let pdpt_entry = harness.read_u64(pdpt + gib * 8);
                harness.write_u64(pdpt + gib * 8, pdpt_entry | PTE_USER);
                let page_directory = pdpt_entry & ADDRESS_MASK;
                for slot in 0..512 {
                    let entry = harness.read_u64(page_directory + slot * 8);
                    harness.write_u64(page_directory + slot * 8, entry | PTE_USER);
                }
            }
            harness.cpu.flush_tlb();
            harness.cpu.set_cpl(3);
        }

        #[test]
        fn jit_and_interpreter_agree_on_fxsave_and_fxrstor() {
            crate::testkit::assert_jit_matches_interpreter(
                // fninit
                &program(&[PUSH_THREE, FXSAVE, &[0xdb, 0xe3], FXRSTOR_FROM_RDI]),
                |h| {
                    init(h);
                    put_f64s(h, &[-2.0]);
                    dirty_state(h);
                },
            );
        }
    }

    mod device_not_available {
        use super::*;

        // One instruction of every x87 family; each must raise #NM when CR0.TS or CR0.EM is set.
        fn x87_samples() -> Vec<(&'static str, Vec<u8>)> {
            let samples: [(&'static str, &[u8]); 24] = [
                // fld1
                ("fld1", &[0xd9, 0xe8]),
                // fldz
                ("fldz", &[0xd9, 0xee]),
                // fld dword ptr [rsi]
                ("fld m32", &[0xd9, 0x06]),
                // fld qword ptr [rsi]
                ("fld m64", &[0xdd, 0x06]),
                // fld tbyte ptr [rsi]
                ("fld m80", &[0xdb, 0x2e]),
                // fild word ptr [rsi]
                ("fild m16", &[0xdf, 0x06]),
                // fstp qword ptr [rdi]
                ("fstp m64", &[0xdd, 0x1f]),
                // fadd st, st(1)
                ("fadd st, st(1)", &[0xd8, 0xc1]),
                // faddp st(1), st
                ("faddp", &[0xde, 0xc1]),
                // fadd qword ptr [rsi]
                ("fadd m64", &[0xdc, 0x06]),
                // fxch st(1)
                ("fxch", &[0xd9, 0xc9]),
                // fcom st(1)
                ("fcom", &[0xd8, 0xd1]),
                // fucomi st, st(1)
                ("fucomi", &[0xdb, 0xe9]),
                // fchs
                ("fchs", &[0xd9, 0xe0]),
                // fxam
                ("fxam", &[0xd9, 0xe5]),
                // fsqrt
                ("fsqrt", &[0xd9, 0xfa]),
                // fnstcw [rdi]
                ("fnstcw", &[0xd9, 0x3f]),
                // fldcw [rsi]
                ("fldcw", &[0xd9, 0x2e]),
                // fnstsw ax
                ("fnstsw ax", &[0xdf, 0xe0]),
                // fnstsw [rdi]
                ("fnstsw m16", &[0xdd, 0x3f]),
                // fnclex
                ("fnclex", &[0xdb, 0xe2]),
                // fninit
                ("fninit", &[0xdb, 0xe3]),
                // fnstenv [rdi]
                ("fnstenv", &[0xd9, 0x37]),
                // fnsave [rdi]
                ("fnsave", &[0xdd, 0x37]),
            ];
            samples
                .into_iter()
                .map(|(name, bytes)| (name, program(&[bytes])))
                .collect()
        }

        #[test]
        fn every_x87_family_raises_nm_when_cr0_ts_is_set() {
            for (name, code) in x87_samples() {
                let h = assert_faults(&code, |h| h.cpu.cr0 |= CR0_TS, device_not_available());
                assert_eq!(h.cpu.ftw, 0, "{name}: the FPU state is untouched");
                assert_eq!(h.cpu.fsw, 0, "{name}");
                assert_eq!(h.cpu.fcw, 0x037f, "{name}");
            }
        }

        #[test]
        fn every_x87_family_raises_nm_when_cr0_em_is_set() {
            for (name, code) in x87_samples() {
                let h = assert_faults(&code, |h| h.cpu.cr0 |= CR0_EM, device_not_available());
                assert_eq!(h.cpu.ftw, 0, "{name}");
            }
        }

        #[test]
        fn nm_carries_no_error_code() {
            // fld1
            let (_, stop) = run_with(&program(&[&[0xd9, 0xe8]]), |h| h.cpu.cr0 |= CR0_TS);
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: 7,
                    error_code: None,
                    software: false
                }
            );
        }

        #[test]
        fn emms_raises_nm_with_ts_and_ud_with_em() {
            // MMX instructions: #UD when CR0.EM is set, otherwise #NM when CR0.TS is set.
            for (bit, expected) in [(CR0_TS, device_not_available()), (CR0_EM, invalid_opcode())] {
                // emms
                let h = assert_faults(
                    &program(&[&[0x0f, 0x77]]),
                    |h| {
                        h.cpu.cr0 |= bit;
                        h.cpu.ftw = 0xff;
                    },
                    expected,
                );
                assert_eq!(h.cpu.ftw, 0xff, "the tags are not cleared");
            }
        }

        #[test]
        fn clts_makes_x87_work_again() {
            // clts; fld1
            let h = run_ok(&program(&[&[0x0f, 0x06, 0xd9, 0xe8]]), |h| {
                h.cpu.cr0 |= CR0_TS
            });
            assert_eq!(st(&h, 0), 1.0);
        }

        #[test]
        fn the_nm_check_comes_before_the_memory_access() {
            // fld qword ptr [rsi]
            let (_, stop) = run_with(&program(&[&[0xdd, 0x06]]), |h| {
                h.cpu.cr0 |= CR0_TS;
                unmap_region(h, UNMAPPED_REGION);
                h.cpu.gpr[RSI] = UNMAPPED_REGION;
            });
            assert_eq!(stop, device_not_available());
        }
    }

    mod invalid_encodings {
        use super::*;

        #[track_caller]
        fn assert_ud(code: &[u8]) {
            let h = assert_faults(&program(&[code]), |_| {}, invalid_opcode());
            assert_eq!(top(&h), 0, "{code:02x?}: the stack is untouched");
            assert_eq!(h.cpu.ftw, 0, "{code:02x?}");
            assert_eq!(h.cpu.fcw, 0x037f, "{code:02x?}");
        }

        #[test]
        fn d9_slash_1_is_ud_and_does_not_store_the_control_word() {
            // d9 /1 has no meaning; it must not execute as fnstcw (d9 /7).
            // .byte 0xd9, 0x0f
            let (h, stop) = run_with(&program(&[&[0xd9, 0x0f]]), |h| h.write(OUT, &[0xaa; 4]));
            assert_eq!(stop, invalid_opcode());
            assert_eq!(h.read(OUT, 4), [0xaa; 4], "nothing may be written");
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn d9_slash_1_is_ud_for_every_memory_addressing_form() {
            let codes: [&[u8]; 4] = [
                // .byte 0xd9, 0x0f
                &[0xd9, 0x0f],
                // .byte 0xd9, 0x4f, 0x08
                &[0xd9, 0x4f, 0x08],
                // .byte 0xd9, 0x8f, 0x00, 0x01, 0x00, 0x00
                &[0xd9, 0x8f, 0x00, 0x01, 0x00, 0x00],
                // .byte 0xd9, 0x0c, 0x24
                &[0xd9, 0x0c, 0x24],
            ];
            for code in codes {
                let mut h = Harness::new(&program(&[code]));
                init(&mut h);
                h.write(OUT, &[0xaa; 4]);
                h.cpu.gpr[RSP] = OUT + 8;
                assert_eq!(h.run(5), invalid_opcode(), "{code:02x?}");
                assert_eq!(h.read(OUT, 4), [0xaa; 4]);
            }
        }

        #[test]
        fn the_valid_neighbours_of_d9_slash_1_still_work() {
            // fnstcw [rdi]
            let h = run_ok(&program(&[&[0xd9, 0x3f]]), |_| {});
            assert_eq!(h.read(OUT, 2), 0x037fu16.to_le_bytes());
            // fldcw [rsi]
            let h = run_ok(&program(&[&[0xd9, 0x2e]]), |h| {
                h.write(DATA, &0x027fu16.to_le_bytes())
            });
            assert_eq!(h.cpu.fcw, 0x027f);
        }

        #[test]
        fn invalid_d9_register_encodings() {
            for byte in [
                0xd1u8, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xe2, 0xe3, 0xe6, 0xe7, 0xef,
            ] {
                assert_ud(&[0xd9, byte]);
            }
        }

        #[test]
        fn invalid_da_register_encodings() {
            for byte in [
                0xe0u8, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xea, 0xeb, 0xec, 0xed,
                0xee, 0xef, 0xf0, 0xff,
            ] {
                assert_ud(&[0xda, byte]);
            }
        }

        #[test]
        fn invalid_db_register_encodings() {
            for byte in [0xe5u8, 0xe6, 0xe7, 0xf8, 0xff] {
                assert_ud(&[0xdb, byte]);
            }
        }

        #[test]
        fn invalid_memory_slashes_of_db_dd_and_df() {
            // db /4, db /6 and dd /5 are reserved encodings.
            let codes: [&[u8]; 3] = [
                // .byte 0xdb, 0x26
                &[0xdb, 0x26],
                // .byte 0xdb, 0x36
                &[0xdb, 0x36],
                // .byte 0xdd, 0x2e
                &[0xdd, 0x2e],
            ];
            for code in codes {
                let mut h = Harness::new(&program(&[code]));
                init(&mut h);
                h.write(OUT, &[0xaa; 16]);
                assert_eq!(h.run(5), invalid_opcode(), "{code:02x?}");
            }
        }

        #[test]
        fn de_d9_is_fcompp_and_other_de_d8_to_df_are_invalid() {
            for byte in [0xd8u8, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf] {
                assert_ud(&[0xde, byte]);
            }
            // fld1; fld1; fcompp
            let h = run_ok(&program(&[&[0xd9, 0xe8, 0xd9, 0xe8, 0xde, 0xd9]]), |_| {});
            assert_eq!(top(&h), 0);
        }

        #[test]
        fn df_register_slots_that_are_not_defined() {
            for byte in [0xe1u8, 0xe2, 0xe7, 0xf8, 0xff] {
                assert_ud(&[0xdf, byte]);
            }
        }
    }
}
