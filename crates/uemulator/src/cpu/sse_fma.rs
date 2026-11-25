//! FMA3, F16C and the AVX compare predicates: the parts of AVX that need the soft-float engine's
//! internals. A child of `sse` so it can use `FpEnv`, `unpack`, `round_pack`, `add_terms` and the
//! lane helpers directly.

use super::*;
use crate::cpu::avx_special::{LANE_BYTES, YMM_BYTES};
use crate::cpu::decode::Insn;

/// Which source plays which role in `x * y + z`. Operand 1 is ModRM.reg (also the destination),
/// operand 2 is VEX.vvvv and operand 3 is ModRM.rm; the digits of the mnemonic name the order of
/// the multiplicands and the addend.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FmaForm {
    /// `op1 * op3 + op2`
    Form132,
    /// `op2 * op1 + op3`
    Form213,
    /// `op2 * op3 + op1`
    Form231,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FmaSign {
    /// Add the addend.
    Add,
    /// Subtract the addend (fmsub).
    Subtract,
}

/// One fused lane. `sources` are the raw operands in instruction order (op1, op2, op3); NaN
/// selection follows that order: the first NaN operand wins, quieted, and any signaling NaN raises
/// invalid. `negate_product` is the fnm* variants, `sign` the msub ones.
pub(super) fn fused_lane(
    env: &mut FpEnv,
    format: Format,
    sources: [u64; 3],
    form: FmaForm,
    negate_product: bool,
    sign: FmaSign,
) -> u64 {
    let [op1, op2, op3] = sources;
    let (x, y, z) = match form {
        FmaForm::Form132 => (op1, op3, op2),
        FmaForm::Form213 => (op2, op1, op3),
        FmaForm::Form231 => (op2, op3, op1),
    };
    // Negating x negates the product; neither negation rounds.
    let x = if negate_product {
        x ^ format.sign_bit()
    } else {
        x
    };
    let z = if sign == FmaSign::Subtract {
        z ^ format.sign_bit()
    } else {
        z
    };
    super::host::fused(env, format, x, y, z)
        .unwrap_or_else(|| soft_fused_lane(env, format, sources, form, negate_product, sign))
}

/// `fused_lane` on the soft engine alone: the oracle.
pub(super) fn soft_fused_lane(
    env: &mut FpEnv,
    format: Format,
    sources: [u64; 3],
    form: FmaForm,
    negate_product: bool,
    sign: FmaSign,
) -> u64 {
    let unpacked = sources.map(|bits| unpack(env, format, bits));
    if unpacked.iter().any(Unpacked::is_nan) {
        if unpacked.iter().any(|u| u.kind == Kind::SignalingNan) {
            env.flags |= FLAG_INVALID;
        }
        let first = unpacked.iter().find(|u| u.is_nan()).unwrap();
        return first.bits | format.quiet_bit();
    }
    let [op1, op2, op3] = unpacked;
    let (x, y, z) = match form {
        FmaForm::Form132 => (op1, op3, op2),
        FmaForm::Form213 => (op2, op1, op3),
        FmaForm::Form231 => (op2, op3, op1),
    };
    let product_sign = (x.sign != y.sign) != negate_product;
    let addend_sign = z.sign != (sign == FmaSign::Subtract);
    let product_is_infinite = x.kind == Kind::Infinity || y.kind == Kind::Infinity;
    if (x.kind == Kind::Infinity && y.is_zero()) || (x.is_zero() && y.kind == Kind::Infinity) {
        return invalid(env, format);
    }
    if product_is_infinite {
        if z.kind == Kind::Infinity && addend_sign != product_sign {
            return invalid(env, format);
        }
        return format.infinity(product_sign);
    }
    if z.kind == Kind::Infinity {
        return format.infinity(addend_sign);
    }
    let addend_term = if z.is_zero() {
        (addend_sign, i32::MIN, 0)
    } else {
        (addend_sign, z.exponent, widen(format, &z))
    };
    if x.is_zero() || y.is_zero() {
        // An exact zero product: the sum is the addend, or a signed zero when both are zero.
        if z.is_zero() {
            let sign = if product_sign == addend_sign {
                product_sign
            } else {
                env.rounding == ROUND_DOWN
            };
            return format.zero(sign);
        }
        return add_terms(env, format, (product_sign, i32::MIN, 0), addend_term);
    }
    // The exact product of two significands (at most 53 bits each) fits in 106 bits, so it is one
    // term of `add_terms` with nothing rounded yet: that is what makes the operation fused.
    let product = x.significand as u128 * y.significand as u128;
    let leading = leading_bit(product);
    let exponent = x.exponent + y.exponent + leading as i32 - 2 * format.fraction_bits as i32;
    let product_term = (
        product_sign,
        exponent,
        product << (WIDE_LEADING_BIT - leading),
    );
    add_terms(env, format, product_term, addend_term)
}

/// Single-precision NaN with the quiet bit, from half-precision fraction bits.
const HALF_EXPONENT_MAX: u32 = 0x1f;
const HALF_FRACTION_BITS: u32 = 10;
const SINGLE_FRACTION_BITS: u32 = 23;
const SINGLE_QUIET_BIT: u32 = 1 << 22;
const HALF_QUIET_BIT: u32 = 1 << 9;

/// Exact half to single. Returns the single and whether the input was a signaling NaN.
pub(super) fn half_to_single(half: u16) -> (u32, bool) {
    let sign = ((half as u32) >> 15) << 31;
    let exponent = (half as u32 >> HALF_FRACTION_BITS) & HALF_EXPONENT_MAX;
    let fraction = half as u32 & ((1 << HALF_FRACTION_BITS) - 1);
    if exponent == HALF_EXPONENT_MAX {
        if fraction == 0 {
            return (sign | 0x7f80_0000, false);
        }
        let signaling = fraction & HALF_QUIET_BIT == 0;
        return (
            sign | 0x7f80_0000
                | SINGLE_QUIET_BIT
                | (fraction << (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS)),
            signaling,
        );
    }
    if exponent == 0 {
        if fraction == 0 {
            return (sign, false);
        }
        // A half subnormal is fraction * 2^-24, exactly representable as a normal single.
        let shift = fraction.leading_zeros() - (32 - HALF_FRACTION_BITS - 1);
        let normalised = (fraction << shift) & ((1 << HALF_FRACTION_BITS) - 1);
        let single_exponent = 127 - 14 - shift;
        return (
            sign | (single_exponent << SINGLE_FRACTION_BITS)
                | (normalised << (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS)),
            false,
        );
    }
    (
        sign | ((exponent + 127 - 15) << SINGLE_FRACTION_BITS)
            | (fraction << (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS)),
        false,
    )
}

/// Single to half under `rounding` (an MXCSR.RC value). Returns the half and the exception flags.
/// Denormal inputs are not flushed (DAZ and FTZ do not apply to F16C).
pub(super) fn single_to_half(single: u32, rounding: u32) -> (u16, u32) {
    let sign = ((single >> 31) as u16) << 15;
    let negative = single >> 31 != 0;
    let exponent_field = (single >> SINGLE_FRACTION_BITS) & 0xff;
    let fraction = single & ((1 << SINGLE_FRACTION_BITS) - 1);
    if exponent_field == 0xff {
        if fraction == 0 {
            return (sign | 0x7c00, 0);
        }
        let flags = if fraction & SINGLE_QUIET_BIT == 0 {
            FLAG_INVALID
        } else {
            0
        };
        let payload = (fraction >> (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS)) as u16;
        return (sign | 0x7c00 | HALF_QUIET_BIT as u16 | payload, flags);
    }
    // value = mantissa * 2^(exponent - 23)
    let (mantissa, exponent) = if exponent_field == 0 {
        (fraction as u64, -126)
    } else {
        (
            (fraction | (1 << SINGLE_FRACTION_BITS)) as u64,
            exponent_field as i32 - 127,
        )
    };
    if mantissa == 0 {
        return (sign, 0);
    }
    let half_min_exponent = -14;
    let tiny = exponent < half_min_exponent;
    let shift = (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS) as i32
        + if tiny {
            half_min_exponent - exponent
        } else {
            0
        };
    let (kept, remainder, half) = if shift >= 63 {
        (0u64, mantissa, 1u64 << 62)
    } else {
        (
            mantissa >> shift,
            mantissa & ((1u64 << shift) - 1),
            1u64 << (shift - 1),
        )
    };
    let inexact = remainder != 0;
    let round_up = match rounding {
        ROUND_NEAREST_EVEN => remainder > half || (remainder == half && kept & 1 == 1),
        ROUND_DOWN => negative && inexact,
        ROUND_UP => !negative && inexact,
        _ => false,
    };
    let kept = kept + round_up as u64;
    let mut flags = if inexact { FLAG_PRECISION } else { 0 };
    let bits = if tiny {
        if inexact {
            flags |= FLAG_UNDERFLOW;
        }
        // kept == 0x400 is the smallest normal: the same bits as exponent field 1
        kept
    } else {
        ((exponent + 15) as u64) * (1 << HALF_FRACTION_BITS) + kept - (1 << HALF_FRACTION_BITS)
    };
    if bits >= 0x7c00 {
        flags |= FLAG_OVERFLOW | FLAG_PRECISION;
        let to_infinity = match rounding {
            ROUND_NEAREST_EVEN => true,
            ROUND_DOWN => negative,
            ROUND_UP => !negative,
            _ => false,
        };
        return (sign | if to_infinity { 0x7c00 } else { 0x7bff }, flags);
    }
    (sign | bits as u16, flags)
}

/// The 32 VEX compare predicates beyond the legacy eight: `(ordered result, unordered result)`
/// is derived from the predicate's mnemonic (SDM table 3-?). `signaling` marks the `_S` forms,
/// which raise invalid for quiet NaNs too.
pub(super) fn compare_predicate_avx(
    env: &mut FpEnv,
    format: Format,
    predicate: u8,
    dest: u64,
    source: u64,
) -> bool {
    use core::cmp::Ordering::{Equal, Greater, Less};
    const SIGNALING: u32 = 0x99996666;
    let signaling = SIGNALING >> predicate & 1 != 0;
    let ordering = compare(env, format, dest, source, signaling);
    // (result for each ordering when ordered, result when unordered)
    let (ordered, unordered): (fn(core::cmp::Ordering) -> bool, bool) = match predicate {
        0 | 16 => (|o| o == Equal, false),
        8 | 24 => (|o| o == Equal, true),
        1 | 17 => (|o| o == Less, false),
        2 | 18 => (|o| o != Greater, false),
        3 | 19 => (|_| false, true),
        4 | 20 => (|o| o != Equal, true),
        12 | 28 => (|o| o != Equal, false),
        5 | 21 => (|o| o != Less, true),
        6 | 22 => (|o| o == Greater, true),
        7 | 23 => (|_| true, false),
        9 | 25 => (|o| o == Less, true),
        10 | 26 => (|o| o != Greater, true),
        11 | 27 => (|_| false, false),
        13 | 29 => (|o| o != Less, false),
        14 | 30 => (|o| o == Greater, false),
        _ => (|_| true, true),
    };
    match ordering {
        Some(o) => ordered(o),
        None => unordered,
    }
}

impl Cpu {
    /// vfmadd/vfmsub/vfnmadd/vfnmsub (132/213/231) and vfmaddsub/vfmsubadd, packed and scalar.
    pub(in crate::cpu) fn exec_fma(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.vex_gate()?;
        let opcode = (i.op & 0xff) as u8;
        let double = i.opsize == 8;
        let format = if double { DOUBLE } else { SINGLE };
        let form = match opcode >> 4 {
            0x9 => FmaForm::Form132,
            0xa => FmaForm::Form213,
            _ => FmaForm::Form231,
        };
        let kind = opcode & 0xf;
        let scalar = matches!(kind, 0x9 | 0xb | 0xd | 0xf);
        let negate_product = matches!(kind, 0xc..=0xf);
        let subtract = matches!(kind, 0xa | 0xb | 0xe | 0xf);
        let alternate = matches!(kind, 0x6 | 0x7);
        let wide = i.vex_l && !scalar;
        let (reg, vvvv) = (i.reg as usize, i.vex_v as usize);
        let bytes = if scalar {
            format.bytes() as usize
        } else if wide {
            YMM_BYTES
        } else {
            LANE_BYTES
        };
        let (c_low, c_high) = self.vector_operand(i, next, bytes)?;
        let operands = [
            (self.xmm[reg], self.ymm_hi[reg]),
            (self.xmm[vvvv], self.ymm_hi[vvvv]),
            (c_low, c_high),
        ];
        let mut env = FpEnv::new(self.mxcsr);
        // `lane_put` ORs into its target, so the halves start empty
        let mut halves = [0u128; 2];
        let half_count = if wide { 2 } else { 1 };
        let lanes = if scalar { 1 } else { format.lanes() };
        for (half, slot) in halves.iter_mut().enumerate().take(half_count) {
            for lane in 0..lanes {
                let pick = |which: usize| {
                    let value = if half == 0 {
                        operands[which].0
                    } else {
                        operands[which].1
                    };
                    lane_get(value, format, lane)
                };
                let sign = if alternate {
                    // vfmaddsub: even lanes subtract; vfmsubadd: even lanes add
                    if (lane % 2 == 0) == (kind == 0x6) {
                        FmaSign::Subtract
                    } else {
                        FmaSign::Add
                    }
                } else if subtract {
                    FmaSign::Subtract
                } else {
                    FmaSign::Add
                };
                let result = fused_lane(
                    &mut env,
                    format,
                    [pick(0), pick(1), pick(2)],
                    form,
                    negate_product,
                    sign,
                );
                lane_put(slot, format, lane, result);
            }
        }
        self.fp_commit(&env)?;
        if scalar {
            // the upper bits of the destination register are kept (operand 1 is the destination)
            let kept = self.xmm[reg] & !format.lane_mask();
            self.set_vector(reg, kept | (halves[0] & format.lane_mask()), 0, false);
        } else {
            self.set_vector(reg, halves[0], halves[1], wide);
        }
        Ok(())
    }

    /// vcvtph2ps (0F38 13) and vcvtps2ph (0F3A 1D).
    pub(in crate::cpu) fn exec_f16c(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.vex_gate()?;
        if i.opsize == 8 {
            return Err(Exception::invalid_opcode());
        }
        let wide = i.vex_l;
        let to_single = (i.op & 0xff) == 0x13;
        let mut env = FpEnv::new(self.mxcsr);
        if to_single {
            if i.vex_v != 0 {
                return Err(Exception::invalid_opcode());
            }
            let (source, _) = self.vector_operand(i, next, if wide { LANE_BYTES } else { 8 })?;
            let mut halves = [0u128; 2];
            for k in 0..if wide { 8 } else { 4 } {
                let (single, signaling) = half_to_single((source >> (16 * k)) as u16);
                if signaling {
                    env.flags |= FLAG_INVALID;
                }
                halves[k / 4] |= (single as u128) << (32 * (k % 4));
            }
            self.fp_commit(&env)?;
            self.set_vector(i.reg as usize, halves[0], halves[1], wide);
        } else {
            if i.vex_v != 0 {
                return Err(Exception::invalid_opcode());
            }
            let selector = i.imm as u8;
            // imm8[2] defers to MXCSR.RC, otherwise imm8[1:0] is the rounding mode
            let rounding = if selector & 4 != 0 {
                env.rounding
            } else {
                (selector & 3) as u32
            };
            let (reg, rm) = (i.reg as usize, i.rm as usize);
            let sources = [self.xmm[reg], self.ymm_hi[reg]];
            let mut packed = 0u128;
            let mut flags = 0;
            for k in 0..if wide { 8 } else { 4 } {
                let single = (sources[k / 4] >> (32 * (k % 4))) as u32;
                let (half, raised) = single_to_half(single, rounding);
                packed |= (half as u128) << (16 * k);
                flags |= raised;
            }
            env.flags = flags;
            self.fp_commit(&env)?;
            if i.md == 3 {
                self.set_vector(rm, packed, 0, false);
            } else {
                let address = self.vex_address(i, next)?;
                if wide {
                    self.write128(address, packed)?;
                } else {
                    self.write(address, 8, packed as u64)?;
                }
            }
        }
        Ok(())
    }
}
