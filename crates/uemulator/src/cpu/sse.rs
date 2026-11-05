//! SSE and SSE2 for the interpreter (the JIT runs these through `helper_exec`).
//!
//! Floating-point arithmetic and conversions never touch the host FPU: x86 NaN selection, the
//! default NaN, DAZ/FTZ, MXCSR.RC, the six exception flags and the unmasked-exception fault have to
//! be identical on aarch64 and x86_64 hosts, so a small soft-float engine on raw bits does the work
//! and the instruction arms only pick lanes. An unmasked exception sets its MXCSR flag, then
//! faults (#XM, or #UD without CR4.OSXMMEXCPT) before the destination is written.
//!
//! Known limits:
//! - rcpps/rsqrtps (and the scalar forms) return the correctly rounded result, where hardware
//!   returns an approximation with ~12 bits, so low mantissa bits differ.
//! - Tininess is detected after rounding to the subnormal grid (see `round_pack`), which differs
//!   from x86 only for values within half a subnormal ulp of the smallest normal number.
//! - A subnormal operand raises the denormal flag even when the other operand is a NaN, as in
//!   QEMU.
//! - SSE3 and later live in `sse_ext.rs` (a child module). No maskmovdqu.
//! - MMX: mmN is the low 64 bits of physical x87 register N (`Cpu::st`); see `Cpu::mmx`. Pending
//!   x87 exceptions (#MF) are not modelled.

use super::decode::*;
use super::exception::Exception;
use super::flags::*;
use super::sys::{CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXMMEXCPT};
use super::*;

/// A vector register viewed as `N` lanes of this type; lane 0 is the least significant.
trait Lane: Copy {
    const N: usize;
    fn get(v: u128, i: usize) -> Self;
    /// ORs into the lane, so `v` must start zeroed (or the lane must be known to be clear).
    fn put(v: &mut u128, i: usize, x: Self);
}

macro_rules! lane {
    ($t:ty, $n:expr) => {
        impl Lane for $t {
            const N: usize = $n;
            fn get(v: u128, i: usize) -> Self {
                (v >> (i * (128 / $n))) as $t
            }
            fn put(v: &mut u128, i: usize, x: Self) {
                *v |= (x as u128) << (i * (128 / $n));
            }
        }
    };
}
lane!(u8, 16);
lane!(u16, 8);
lane!(u32, 4);
lane!(u64, 2);

fn zip_lanes<T: Lane>(a: u128, b: u128, f: impl Fn(T, T) -> T) -> u128 {
    let mut r = 0;
    for i in 0..T::N {
        T::put(&mut r, i, f(T::get(a, i), T::get(b, i)));
    }
    r
}

/// punpckl*/punpckh*: alternates lanes of `a` and `b` taken from the low or high half of each.
fn interleave<T: Lane>(a: u128, b: u128, hi: bool) -> u128 {
    let mut r = 0;
    let off = if hi { T::N / 2 } else { 0 };
    for i in 0..T::N / 2 {
        T::put(&mut r, 2 * i, T::get(a, off + i));
        T::put(&mut r, 2 * i + 1, T::get(b, off + i));
    }
    r
}

/// packsswb/packuswb (`source_bits` 16) and packssdw (`source_bits` 32): saturating narrow of the
/// signed lanes of `a` into the low half of the result, then those of `b` into the high half.
fn pack_sat(a: u128, b: u128, source_bits: usize, signed_out: bool) -> u128 {
    let mut r = 0u128;
    let get = |v: u128, i: usize| -> i32 {
        if source_bits == 16 {
            u16::get(v, i) as i16 as i32
        } else {
            u32::get(v, i) as i32
        }
    };
    let n = 128 / source_bits;
    for k in 0..2 * n {
        let x = get(if k < n { a } else { b }, k % n);
        if source_bits == 16 {
            let y = if signed_out {
                x.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8 as u8
            } else {
                x.clamp(0, i32::from(u8::MAX)) as u8
            };
            u8::put(&mut r, k, y);
        } else {
            let y = if signed_out {
                x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 as u16
            } else {
                x.clamp(0, i32::from(u16::MAX)) as u16
            };
            u16::put(&mut r, k, y);
        }
    }
    r
}

/// MXCSR exception flag bits (0-5); the matching mask bits are the same positions shifted by
/// `MXCSR_MASK_SHIFT`.
const FLAG_INVALID: u32 = 1 << 0;
const FLAG_DENORMAL: u32 = 1 << 1;
const FLAG_DIVIDE_BY_ZERO: u32 = 1 << 2;
const FLAG_OVERFLOW: u32 = 1 << 3;
const FLAG_UNDERFLOW: u32 = 1 << 4;
const FLAG_PRECISION: u32 = 1 << 5;
const MXCSR_EXCEPTION_FLAGS: u32 = 0x3f;
const MXCSR_MASK_SHIFT: u32 = 7;
const MXCSR_DAZ: u32 = 1 << 6;
const MXCSR_UNDERFLOW_MASK: u32 = 1 << 11;
const MXCSR_FTZ: u32 = 1 << 15;
/// MXCSR.RC field, bits 14:13.
const MXCSR_RC_SHIFT: u32 = 13;
const MXCSR_RC_MASK: u32 = 3;
const ROUND_NEAREST_EVEN: u32 = 0;
const ROUND_DOWN: u32 = 1;
const ROUND_UP: u32 = 2;
const ROUND_TOWARD_ZERO: u32 = 3;
/// #XM, raised for an unmasked SIMD floating-point exception when CR4.OSXMMEXCPT is set.
const VECTOR_SIMD_FLOATING_POINT: u8 = 19;

/// Rounding mode, DAZ/FTZ and the exception flags one instruction raises. The flags are merged
/// into MXCSR (and checked against the masks) by `Cpu::fp_commit` only after every lane is done,
/// so an unmasked exception leaves the destination untouched.
struct FpEnv {
    rounding: u32,
    denormals_are_zero: bool,
    flush_to_zero: bool,
    underflow_masked: bool,
    flags: u32,
}

impl FpEnv {
    fn new(mxcsr: u32) -> Self {
        FpEnv {
            rounding: (mxcsr >> MXCSR_RC_SHIFT) & MXCSR_RC_MASK,
            denormals_are_zero: mxcsr & MXCSR_DAZ != 0,
            flush_to_zero: mxcsr & MXCSR_FTZ != 0,
            underflow_masked: mxcsr & MXCSR_UNDERFLOW_MASK != 0,
            flags: 0,
        }
    }
}

/// An IEEE binary format, parameterised so one set of arithmetic serves f32 and f64. Values travel
/// as raw bits in a u64: the host FPU never sees them, because x86 NaN selection, the default NaN,
/// DAZ/FTZ, directed rounding and the exception flags cannot be taken from it portably.
#[derive(Clone, Copy)]
struct Format {
    exponent_bits: u32,
    fraction_bits: u32,
}

/// Width of an XMM register.
const VECTOR_BITS: u32 = 128;
/// SSE memory operands of 16 bytes or more must be aligned to this many bytes (#GP(0) otherwise).
const VECTOR_ALIGNMENT: u64 = 16;

const SINGLE: Format = Format {
    exponent_bits: 8,
    fraction_bits: 23,
};
const DOUBLE: Format = Format {
    exponent_bits: 11,
    fraction_bits: 52,
};

impl Format {
    fn width(self) -> u32 {
        1 + self.exponent_bits + self.fraction_bits
    }
    fn bytes(self) -> u8 {
        (self.width() / 8) as u8
    }
    fn lanes(self) -> usize {
        (VECTOR_BITS / self.width()) as usize
    }
    fn exponent_mask(self) -> u64 {
        (1 << self.exponent_bits) - 1
    }
    fn fraction_mask(self) -> u64 {
        (1 << self.fraction_bits) - 1
    }
    fn lane_mask(self) -> u128 {
        (1u128 << self.width()) - 1
    }
    fn bias(self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }
    /// Unbiased exponent of the smallest normal number.
    fn min_exponent(self) -> i32 {
        1 - self.bias()
    }
    /// Unbiased exponent of the largest finite number.
    fn max_exponent(self) -> i32 {
        self.bias()
    }
    fn quiet_bit(self) -> u64 {
        1 << (self.fraction_bits - 1)
    }
    fn sign_bit(self) -> u64 {
        1 << (self.exponent_bits + self.fraction_bits)
    }
    fn infinity(self, sign: bool) -> u64 {
        (self.exponent_mask() << self.fraction_bits) | if sign { self.sign_bit() } else { 0 }
    }
    fn zero(self, sign: bool) -> u64 {
        if sign { self.sign_bit() } else { 0 }
    }
    fn max_finite(self, sign: bool) -> u64 {
        ((self.exponent_mask() - 1) << self.fraction_bits)
            | self.fraction_mask()
            | if sign { self.sign_bit() } else { 0 }
    }
    /// The "real indefinite" QNaN that invalid operations produce: negative, quiet, empty
    /// payload (0xFFC00000 / 0xFFF8000000000000).
    fn indefinite(self) -> u64 {
        self.infinity(true) | self.quiet_bit()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Zero,
    Finite,
    Infinity,
    QuietNan,
    SignalingNan,
}

/// A decoded operand. Finite values (subnormals included) are normalised: `significand` has its
/// leading one at bit `fraction_bits` and `exponent` is the unbiased exponent of that bit.
/// `bits` is the operand as the instruction sees it, i.e. with DAZ applied.
#[derive(Clone, Copy)]
struct Unpacked {
    sign: bool,
    kind: Kind,
    exponent: i32,
    significand: u64,
    bits: u64,
}

impl Unpacked {
    fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::QuietNan | Kind::SignalingNan)
    }
    fn is_zero(&self) -> bool {
        self.kind == Kind::Zero
    }
}

/// Decodes `bits`. A subnormal source raises the denormal-operand flag, unless DAZ turns it into a
/// zero of the same sign.
fn unpack(env: &mut FpEnv, format: Format, bits: u64) -> Unpacked {
    let sign = bits & format.sign_bit() != 0;
    let exponent_field = (bits >> format.fraction_bits) & format.exponent_mask();
    let fraction = bits & format.fraction_mask();
    let mut value = Unpacked {
        sign,
        kind: Kind::Finite,
        exponent: 0,
        significand: 0,
        bits,
    };
    if exponent_field == format.exponent_mask() {
        value.kind = if fraction == 0 {
            Kind::Infinity
        } else if fraction & format.quiet_bit() != 0 {
            Kind::QuietNan
        } else {
            Kind::SignalingNan
        };
    } else if exponent_field == 0 {
        if fraction == 0 || env.denormals_are_zero {
            value.kind = Kind::Zero;
            value.bits = format.zero(sign);
        } else {
            env.flags |= FLAG_DENORMAL;
            let shift = format.fraction_bits - (u64::BITS - 1 - fraction.leading_zeros());
            value.significand = fraction << shift;
            value.exponent = format.min_exponent() - shift as i32;
        }
    } else {
        value.significand = fraction | (1 << format.fraction_bits);
        value.exponent = exponent_field as i32 - format.bias();
    }
    value
}

/// Width of the intermediate significand every operation hands to `round_pack`.
const WIDE_BITS: u32 = 128;
/// Leading-one position of that intermediate: one below the top so that adding two aligned
/// significands cannot overflow.
const WIDE_LEADING_BIT: u32 = WIDE_BITS - 2;
/// Where the leading one of the integer square root's radicand is placed: high enough that the
/// root has well over the 55 bits the widest format needs for round and sticky.
const RADICAND_LEADING_BIT: u32 = 120;

/// Position of the highest set bit of a nonzero value.
fn leading_bit(value: u128) -> u32 {
    WIDE_BITS - 1 - value.leading_zeros()
}

fn widen(format: Format, value: &Unpacked) -> u128 {
    if value.kind == Kind::Finite {
        (value.significand as u128) << (WIDE_LEADING_BIT - format.fraction_bits)
    } else {
        0
    }
}

/// Right shift that ORs everything shifted out into bit 0, so later rounding still sees it.
fn shift_right_sticky(value: u128, count: u32) -> u128 {
    if count == 0 {
        value
    } else if count >= WIDE_BITS {
        (value != 0) as u128
    } else {
        (value >> count) | ((value & ((1u128 << count) - 1) != 0) as u128)
    }
}

/// Whether magnitude `kept` (with `remainder` of `half * 2` dropped below it) rounds up.
fn rounds_up(rounding: u32, sign: bool, kept: u128, remainder: u128, half: u128) -> bool {
    match rounding {
        ROUND_NEAREST_EVEN => remainder > half || (remainder == half && kept & 1 == 1),
        ROUND_DOWN => sign && remainder != 0,
        ROUND_UP => !sign && remainder != 0,
        _ => false,
    }
}

/// Splits `significand` (a value in [2^126, 2^127)) at `shift` bits, giving the kept part, the
/// dropped part and half of one kept-part unit.
fn split_significand(significand: u128, shift: u32) -> (u128, u128, u128) {
    if shift >= WIDE_BITS {
        (0, significand, 1 << (WIDE_BITS - 1))
    } else {
        (
            significand >> shift,
            significand & ((1u128 << shift) - 1),
            1 << (shift - 1),
        )
    }
}

/// Rounds `significand` x 2^(exponent - 126) to `format` under the current rounding mode and
/// returns the packed result, raising overflow, underflow and precision as needed. The caller
/// guarantees bit 126 is set and that bit 0 carries the sticky of anything below it.
/// Tininess is detected after rounding as on x86, with one simplification: rounding is done to the
/// subnormal grid, so a value that only rounds up to the smallest normal number there is treated
/// as not tiny. FTZ (with underflow masked) replaces any tiny result, even an exact one, by a
/// signed zero and reports underflow and precision, as hardware does.
fn round_pack(
    env: &mut FpEnv,
    format: Format,
    sign: bool,
    exponent: i32,
    significand: u128,
) -> u64 {
    let fraction_bits = format.fraction_bits;
    let in_subnormal_range = exponent < format.min_exponent();
    let mut shift = WIDE_LEADING_BIT - fraction_bits;
    if in_subnormal_range {
        shift = shift.saturating_add((format.min_exponent() - exponent) as u32);
    }
    let (mut kept, remainder, half) = split_significand(significand, shift);
    let inexact = remainder != 0;
    if rounds_up(env.rounding, sign, kept, remainder, half) {
        kept += 1;
    }
    let mut kept = kept as u64;
    let mut exponent = exponent;
    let sign_bits = if sign { format.sign_bit() } else { 0 };
    if inexact {
        env.flags |= FLAG_PRECISION;
    }
    if in_subnormal_range {
        let tiny = kept < (1 << fraction_bits);
        if tiny {
            if env.flush_to_zero && env.underflow_masked {
                env.flags |= FLAG_UNDERFLOW | FLAG_PRECISION;
                return format.zero(sign);
            }
            if inexact || !env.underflow_masked {
                env.flags |= FLAG_UNDERFLOW;
            }
        }
        // kept == 2^fraction_bits is the smallest normal number: the same bits as exponent field 1.
        return sign_bits | kept;
    }
    if kept >> (fraction_bits + 1) != 0 {
        kept >>= 1;
        exponent += 1;
    }
    if exponent > format.max_exponent() {
        env.flags |= FLAG_OVERFLOW | FLAG_PRECISION;
        let to_infinity = match env.rounding {
            ROUND_NEAREST_EVEN => true,
            ROUND_DOWN => sign,
            ROUND_UP => !sign,
            _ => false,
        };
        return if to_infinity {
            format.infinity(sign)
        } else {
            format.max_finite(sign)
        };
    }
    let exponent_field = (exponent + format.bias()) as u64;
    sign_bits | (exponent_field << fraction_bits) | (kept & format.fraction_mask())
}

fn invalid(env: &mut FpEnv, format: Format) -> u64 {
    env.flags |= FLAG_INVALID;
    format.indefinite()
}

/// x86 NaN selection for a two-operand instruction: the destination operand's NaN wins when both
/// are NaN; the result is always quieted, and any signaling NaN raises invalid.
fn propagate_nan(
    env: &mut FpEnv,
    format: Format,
    dest: &Unpacked,
    source: &Unpacked,
) -> Option<u64> {
    if !dest.is_nan() && !source.is_nan() {
        return None;
    }
    if dest.kind == Kind::SignalingNan || source.kind == Kind::SignalingNan {
        env.flags |= FLAG_INVALID;
    }
    let chosen = if dest.is_nan() { dest } else { source };
    Some(chosen.bits | format.quiet_bit())
}

#[derive(Clone, Copy, PartialEq)]
enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
}

/// One lane of add/sub/mul/div/sqrt, `dest` being the destination operand and `source` the
/// ModRM.rm one (sqrt only reads `source`).
fn arithmetic(
    env: &mut FpEnv,
    format: Format,
    operation: Arithmetic,
    dest: u64,
    source: u64,
) -> u64 {
    let b = unpack(env, format, source);
    if operation == Arithmetic::SquareRoot {
        if b.is_nan() {
            return propagate_nan(env, format, &b, &b).unwrap();
        }
        return square_root(env, format, &b);
    }
    let a = unpack(env, format, dest);
    if let Some(nan) = propagate_nan(env, format, &a, &b) {
        return nan;
    }
    match operation {
        Arithmetic::Add => add(env, format, &a, &b, false),
        Arithmetic::Subtract => add(env, format, &a, &b, true),
        Arithmetic::Multiply => multiply(env, format, &a, &b),
        _ => divide(env, format, &a, &b),
    }
}

fn add(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked, subtract: bool) -> u64 {
    let b_sign = b.sign != subtract;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Infinity) => {
            return if a.sign == b_sign {
                format.infinity(a.sign)
            } else {
                invalid(env, format)
            };
        }
        (Kind::Infinity, _) => return format.infinity(a.sign),
        (_, Kind::Infinity) => return format.infinity(b_sign),
        // Exact zero sum: -0 only when both are -0, or in round-down mode for opposite signs.
        (Kind::Zero, Kind::Zero) => {
            let sign = if a.sign == b_sign {
                a.sign
            } else {
                env.rounding == ROUND_DOWN
            };
            return format.zero(sign);
        }
        _ => {}
    }
    // Zero operands get an exponent below every finite one, so the ordering swap below puts the
    // zero second.
    let term = |value: &Unpacked, sign: bool| {
        let exponent = if value.is_zero() {
            i32::MIN
        } else {
            value.exponent
        };
        (sign, exponent, widen(format, value))
    };
    add_terms(env, format, term(a, a.sign), term(b, b_sign))
}

/// Sum of two finite, normalised terms `(sign, exponent, significand)` whose significands have
/// their leading one at `WIDE_LEADING_BIT` (a zero term has significand 0 and exponent
/// `i32::MIN`), rounded once. Also used by the fused multiply-add, whose exact product is one term.
fn add_terms(
    env: &mut FpEnv,
    format: Format,
    first: (bool, i32, u128),
    second: (bool, i32, u128),
) -> u64 {
    let mut x = first;
    let mut y = second;
    if (y.1, y.2) > (x.1, x.2) {
        core::mem::swap(&mut x, &mut y);
    }
    if y.2 == 0 {
        return round_pack(env, format, x.0, x.1, x.2);
    }
    let aligned = shift_right_sticky(y.2, (x.1 - y.1) as u32);
    if x.0 == y.0 {
        let mut sum = x.2 + aligned;
        let mut exponent = x.1;
        if sum >> (WIDE_BITS - 1) != 0 {
            sum = shift_right_sticky(sum, 1);
            exponent += 1;
        }
        round_pack(env, format, x.0, exponent, sum)
    } else {
        let difference = x.2 - aligned;
        if difference == 0 {
            return format.zero(env.rounding == ROUND_DOWN);
        }
        let shift = WIDE_LEADING_BIT - leading_bit(difference);
        round_pack(env, format, x.0, x.1 - shift as i32, difference << shift)
    }
}

fn multiply(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked) -> u64 {
    let sign = a.sign != b.sign;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Zero) | (Kind::Zero, Kind::Infinity) => {
            return invalid(env, format);
        }
        (Kind::Infinity, _) | (_, Kind::Infinity) => return format.infinity(sign),
        (Kind::Zero, _) | (_, Kind::Zero) => return format.zero(sign),
        _ => {}
    }
    let product = a.significand as u128 * b.significand as u128;
    let leading = leading_bit(product);
    // The product of two significands in [1, 2) in 2 x fraction_bits fixed point is in [1, 4).
    let exponent = a.exponent + b.exponent + leading as i32 - 2 * format.fraction_bits as i32;
    round_pack(
        env,
        format,
        sign,
        exponent,
        product << (WIDE_LEADING_BIT - leading),
    )
}

/// Fractional bits of the quotient computed by `divide`; far more than any format needs, so the
/// remainder only has to supply the sticky bit.
const QUOTIENT_FRACTION_BITS: u32 = 64;

fn divide(env: &mut FpEnv, format: Format, a: &Unpacked, b: &Unpacked) -> u64 {
    let sign = a.sign != b.sign;
    match (a.kind, b.kind) {
        (Kind::Infinity, Kind::Infinity) | (Kind::Zero, Kind::Zero) => {
            return invalid(env, format);
        }
        (Kind::Infinity, _) => return format.infinity(sign),
        (_, Kind::Infinity) => return format.zero(sign),
        (Kind::Zero, _) => return format.zero(sign),
        (_, Kind::Zero) => {
            env.flags |= FLAG_DIVIDE_BY_ZERO;
            return format.infinity(sign);
        }
        _ => {}
    }
    let numerator = (a.significand as u128) << QUOTIENT_FRACTION_BITS;
    let divisor = b.significand as u128;
    let quotient = numerator / divisor;
    let sticky = !numerator.is_multiple_of(divisor) as u128;
    let leading = leading_bit(quotient);
    let exponent = a.exponent - b.exponent + leading as i32 - QUOTIENT_FRACTION_BITS as i32;
    round_pack(
        env,
        format,
        sign,
        exponent,
        (quotient << (WIDE_LEADING_BIT - leading)) | sticky,
    )
}

/// Floor of the square root of `n`.
fn integer_square_root(mut n: u128) -> u128 {
    let mut root = 0u128;
    let mut bit = 1u128 << WIDE_LEADING_BIT;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if n >= root + bit {
            n -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

fn square_root(env: &mut FpEnv, format: Format, value: &Unpacked) -> u64 {
    match value.kind {
        Kind::Zero => return value.bits,
        Kind::Infinity if !value.sign => return format.infinity(false),
        _ if value.sign => return invalid(env, format),
        _ => {}
    }
    // value = m * 2^e with e even, then scaled up so the integer root has plenty of bits.
    let mut m = value.significand as u128;
    let mut e = value.exponent - format.fraction_bits as i32;
    if e & 1 != 0 {
        m <<= 1;
        e -= 1;
    }
    let scale = (RADICAND_LEADING_BIT - leading_bit(m)) & !1;
    let radicand = m << scale;
    let root = integer_square_root(radicand);
    let sticky = (root * root != radicand) as u128;
    let leading = leading_bit(root);
    let exponent = (e - scale as i32) / 2 + leading as i32;
    round_pack(
        env,
        format,
        false,
        exponent,
        (root << (WIDE_LEADING_BIT - leading)) | sticky,
    )
}

/// Orders two non-NaN operands numerically; zeros of either sign are equal.
fn order(a: &Unpacked, b: &Unpacked) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let negative = |v: &Unpacked| v.sign && !v.is_zero();
    let magnitude = |v: &Unpacked| match v.kind {
        Kind::Zero => (0, 0, 0),
        Kind::Infinity => (2, 0, 0),
        _ => (1, v.exponent, v.significand),
    };
    match (negative(a), negative(b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => magnitude(b).cmp(&magnitude(a)),
        (false, false) => magnitude(a).cmp(&magnitude(b)),
    }
}

/// minps/maxps and the scalar forms: when either operand is a NaN (any kind, which raises invalid)
/// or both are zero, the second (source) operand is returned unchanged.
fn minimum_maximum(env: &mut FpEnv, format: Format, maximum: bool, dest: u64, source: u64) -> u64 {
    let a = unpack(env, format, dest);
    let b = unpack(env, format, source);
    if a.is_nan() || b.is_nan() {
        env.flags |= FLAG_INVALID;
        return source;
    }
    if a.is_zero() && b.is_zero() {
        return b.bits;
    }
    let ordering = order(&a, &b);
    let first = if maximum {
        ordering == core::cmp::Ordering::Greater
    } else {
        ordering == core::cmp::Ordering::Less
    };
    if first { a.bits } else { b.bits }
}

/// Orders two operands for the compare instructions, `None` when unordered. `signal_quiet_nan`
/// selects whether a QNaN (not just an SNaN) raises invalid.
fn compare(
    env: &mut FpEnv,
    format: Format,
    dest: u64,
    source: u64,
    signal_quiet_nan: bool,
) -> Option<core::cmp::Ordering> {
    let a = unpack(env, format, dest);
    let b = unpack(env, format, source);
    if a.is_nan() || b.is_nan() {
        let signaling = a.kind == Kind::SignalingNan || b.kind == Kind::SignalingNan;
        if signaling || signal_quiet_nan {
            env.flags |= FLAG_INVALID;
        }
        return None;
    }
    Some(order(&a, &b))
}

/// cmpps/cmppd/cmpss/cmpsd with imm8 predicates 0-7 (eq, lt, le, unord, neq, nlt, nle, ord; only
/// the low three bits exist before AVX). lt, le, nlt and nle also raise invalid for a QNaN.
fn compare_predicate(
    env: &mut FpEnv,
    format: Format,
    predicate: u8,
    dest: u64,
    source: u64,
    extended: bool,
) -> bool {
    use core::cmp::Ordering::{Equal, Less};
    if extended {
        return fma::compare_predicate_avx(env, format, predicate & 0x1f, dest, source);
    }
    let predicate = predicate & 7;
    let signaling = matches!(predicate, 1 | 2 | 5 | 6);
    let ordering = compare(env, format, dest, source, signaling);
    match predicate {
        0 => ordering == Some(Equal),
        1 => ordering == Some(Less),
        2 => matches!(ordering, Some(Less | Equal)),
        3 => ordering.is_none(),
        4 => ordering != Some(Equal),
        5 => ordering != Some(Less),
        6 => !matches!(ordering, Some(Less | Equal)),
        _ => ordering.is_some(),
    }
}

/// (u)comiss/(u)comisd result: ZF, PF, CF encode unordered / less / equal; the other status
/// flags are cleared.
fn compare_eflags(ordering: Option<core::cmp::Ordering>) -> u64 {
    use core::cmp::Ordering::{Equal, Greater, Less};
    match ordering {
        None => ZF | PF | CF,
        Some(Greater) => 0,
        Some(Less) => CF,
        Some(Equal) => ZF,
    }
}

/// Signed integer to float (cvtsi2ss/sd, cvtdq2ps/pd), rounded once under the current mode.
fn int_to_float(env: &mut FpEnv, format: Format, value: i64) -> u64 {
    if value == 0 {
        return 0;
    }
    let magnitude = value.unsigned_abs() as u128;
    let leading = leading_bit(magnitude);
    round_pack(
        env,
        format,
        value < 0,
        leading as i32,
        magnitude << (WIDE_LEADING_BIT - leading),
    )
}

/// cvt(t)ss2si/cvt(t)sd2si/cvt(t)ps2dq/cvt(t)pd2dq to a `width`-bit signed result (returned as
/// raw bits, zero-extended to 64). `truncate` selects the cvtt* forms; the others round under
/// MXCSR.RC. NaN, infinity and out-of-range values give the "integer indefinite" (the minimum)
/// and raise invalid; an inexact conversion raises precision.
fn float_to_int(env: &mut FpEnv, format: Format, bits: u64, width: u32, truncate: bool) -> u64 {
    let indefinite = 1u64 << (width - 1);
    let value = unpack(env, format, bits);
    match value.kind {
        Kind::Zero => return 0,
        Kind::Finite => {}
        _ => {
            env.flags |= FLAG_INVALID;
            return indefinite;
        }
    }
    // Values of 2^64 or more are out of range for either width; this also keeps the shift below
    // positive.
    if value.exponent >= 64 {
        env.flags |= FLAG_INVALID;
        return indefinite;
    }
    let rounding = if truncate {
        ROUND_TOWARD_ZERO
    } else {
        env.rounding
    };
    let (mut magnitude, remainder, half) = split_significand(
        widen(format, &value),
        (WIDE_LEADING_BIT as i32 - value.exponent) as u32,
    );
    if rounds_up(rounding, value.sign, magnitude, remainder, half) {
        magnitude += 1;
    }
    let limit = if value.sign {
        1u128 << (width - 1)
    } else {
        (1u128 << (width - 1)) - 1
    };
    if magnitude > limit {
        env.flags |= FLAG_INVALID;
        return indefinite;
    }
    if remainder != 0 {
        env.flags |= FLAG_PRECISION;
    }
    let result = if value.sign {
        (magnitude as u64).wrapping_neg()
    } else {
        magnitude as u64
    };
    if width == 32 {
        result & 0xffff_ffff
    } else {
        result
    }
}

/// cvtss2sd, cvtsd2ss, cvtps2pd, cvtpd2ps for one lane. NaNs keep their payload (shifted to the
/// new width) and are quieted.
fn convert_float(env: &mut FpEnv, from: Format, to: Format, bits: u64) -> u64 {
    let value = unpack(env, from, bits);
    match value.kind {
        Kind::QuietNan | Kind::SignalingNan => {
            if value.kind == Kind::SignalingNan {
                env.flags |= FLAG_INVALID;
            }
            let payload = bits & from.fraction_mask();
            let payload = if to.fraction_bits > from.fraction_bits {
                payload << (to.fraction_bits - from.fraction_bits)
            } else {
                payload >> (from.fraction_bits - to.fraction_bits)
            };
            let sign = if value.sign { to.sign_bit() } else { 0 };
            sign | to.infinity(false) | to.quiet_bit() | payload
        }
        Kind::Infinity => to.infinity(value.sign),
        Kind::Zero => to.zero(value.sign),
        Kind::Finite => round_pack(env, to, value.sign, value.exponent, widen(from, &value)),
    }
}

/// rcpps/rcpss and rsqrtps/rsqrtss. Hardware returns a ~12-bit approximation; this computes the
/// correctly rounded reciprocal (of the square root), so the low mantissa bits differ. These
/// instructions never touch MXCSR flags and always treat subnormal inputs and tiny results as
/// zero, which a scratch environment with DAZ and FTZ forced on reproduces.
fn reciprocal(format: Format, bits: u64, square_root: bool) -> u64 {
    let mut scratch = FpEnv {
        rounding: ROUND_NEAREST_EVEN,
        denormals_are_zero: true,
        flush_to_zero: true,
        underflow_masked: true,
        flags: 0,
    };
    let one = (format.bias() as u64) << format.fraction_bits;
    let operand = if square_root {
        arithmetic(&mut scratch, format, Arithmetic::SquareRoot, 0, bits)
    } else {
        bits
    };
    arithmetic(&mut scratch, format, Arithmetic::Divide, one, operand)
}

fn lane_get(value: u128, format: Format, lane: usize) -> u64 {
    (value >> (lane as u32 * format.width())) as u64 & format.lane_mask() as u64
}

fn lane_put(result: &mut u128, format: Format, lane: usize, bits: u64) {
    *result |= (bits as u128) << (lane as u32 * format.width());
}

/// One lane of the sqrt/add/mul/sub/min/div/max group, selected by opcode.
fn binary_operation(env: &mut FpEnv, format: Format, opcode: u8, dest: u64, source: u64) -> u64 {
    let operation = match opcode {
        0x51 => Arithmetic::SquareRoot,
        0x58 => Arithmetic::Add,
        0x59 => Arithmetic::Multiply,
        0x5c => Arithmetic::Subtract,
        0x5e => Arithmetic::Divide,
        _ => return minimum_maximum(env, format, opcode == 0x5f, dest, source),
    };
    arithmetic(env, format, operation, dest, source)
}

/// The mandatory prefix that selects between the packed-single, packed-double, scalar-single and
/// scalar-double (or MMX vs. XMM integer) forms of an 0F opcode. `Np` = none, `P66` = operand-size
/// prefix, `F3`/`F2` = rep/repne; if several are present F3/F2 win (see `Insn::rep`).
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Prefix {
    Np,
    P66,
    F3,
    F2,
}

/// `Insn::op` is `(map << 8) | opcode_byte`; map 1 is the `0f xx` two-byte map.
const OPCODE_MAP_0F: u16 = 1;
const MODRM_MOD_REGISTER: u8 = 3;
/// MMX registers mm0-mm7 are the low 64 bits of the *physical* x87 registers, whatever TOP is.
const MMX_REGISTER_MASK: u8 = 7;
const MMX_QWORD_BYTES: usize = 8;
/// The sign and exponent field an MMX write leaves in the x87 register: all ones.
const MMX_EXPONENT_FIELD: u16 = 0xffff;
/// FSW.TOP, bits 13:11.
const FSW_TOP_MASK: u16 = 7 << 11;
/// `Cpu::ftw` holds one bit per physical x87 register, set when it is not empty.
const ALL_X87_REGISTERS_VALID: u16 = 0xff;
const MMX_WORD_COUNT: usize = 4;
const GPR_RDI: usize = 7;
const LOW_32: u128 = 0xffff_ffff;
const LOW_64: u128 = u64::MAX as u128;

impl Cpu {
    /// SDM SSE exception rules: #UD if CR0.EM is set or the OS has not enabled SSE (CR4.OSFXSR),
    /// #NM if CR0.TS is set (lazy FPU switching). #UD takes priority over #NM.
    pub(super) fn sse_gate(&self) -> ExecResult<()> {
        // A VEX instruction did its own checks (XCR0/OSXSAVE instead of OSFXSR) in `avx.rs`.
        if self.vex_active {
            return Ok(());
        }
        if self.cr0 & CR0_EM != 0 || self.cr4 & CR4_OSFXSR == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    /// Merges the exception flags an instruction raised into MXCSR. An unmasked one is a fault
    /// (#XM, or #UD when the OS has not set CR4.OSXMMEXCPT) and the caller must not have written
    /// its destination yet. The flags stay set in MXCSR either way, as on hardware.
    fn fp_commit(&mut self, env: &FpEnv) -> ExecResult<()> {
        self.mxcsr |= env.flags;
        let unmasked = env.flags & !(self.mxcsr >> MXCSR_MASK_SHIFT) & MXCSR_EXCEPTION_FLAGS;
        if unmasked == 0 {
            Ok(())
        } else if self.cr4 & CR4_OSXMMEXCPT != 0 {
            Err(Exception::fault(VECTOR_SIMD_FLOATING_POINT, None))
        } else {
            Err(Exception::invalid_opcode())
        }
    }

    fn mmx_enter(&mut self) {
        self.fsw &= !FSW_TOP_MASK;
        self.ftw = ALL_X87_REGISTERS_VALID;
    }

    fn mm_get(&self, index: u8) -> u64 {
        let raw = &self.st[(index & MMX_REGISTER_MASK) as usize];
        u64::from_le_bytes(raw[..MMX_QWORD_BYTES].try_into().unwrap())
    }

    fn mm_put(&mut self, index: u8, value: u64) {
        let raw = &mut self.st[(index & MMX_REGISTER_MASK) as usize];
        raw[..MMX_QWORD_BYTES].copy_from_slice(&value.to_le_bytes());
        raw[MMX_QWORD_BYTES..].copy_from_slice(&MMX_EXPONENT_FIELD.to_le_bytes());
    }

    /// The MMX instructions (see `mmx_form`). Gating: CR0.EM is #UD and CR0.TS is #NM; CR4.OSFXSR
    /// only matters for the forms that also touch an xmm register. Any of them except emms puts
    /// the x87 stack in MMX state (TOP = 0, every tag valid) and a write to mmN also fills the
    /// register's sign/exponent field with ones. Memory operands are read before any state
    /// changes, so a fault leaves the machine as it was.
    fn mmx(&mut self, i: &Insn, next: u64, opcode: u8, prefix: Prefix) -> ExecResult<()> {
        if matches!(opcode, 0x2a | 0x2c | 0x2d | 0xd6) {
            self.sse_gate()?;
        } else {
            if self.cr0 & CR0_EM != 0 {
                return Err(Exception::invalid_opcode());
            }
            self.fpu_check()?;
        }
        let reg = i.reg;
        let rm = i.rm;
        let mem_addr = if i.md == MODRM_MOD_REGISTER {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next))?)
        };
        let operand_size = if i.opsize == 8 { 8 } else { 4 };
        match opcode {
            // movd/movq mm, r/m
            0x6e => {
                let v = match mem_addr {
                    None => self.read_gpr(rm, operand_size, true),
                    Some(addr) => self.read(addr, operand_size)?,
                } & mask(operand_size);
                self.mmx_enter();
                self.mm_put(reg, v);
            }
            // movd/movq r/m, mm
            0x7e => {
                let v = self.mm_get(reg) & mask(operand_size);
                match mem_addr {
                    None => self.write_gpr(rm, operand_size, true, v),
                    Some(addr) => self.write(addr, operand_size, v)?,
                }
                self.mmx_enter();
            }
            // movq mm, mm/m64
            0x6f => {
                let v = self.mm_or_m64(rm, mem_addr)?;
                self.mmx_enter();
                self.mm_put(reg, v);
            }
            // movq mm/m64, mm
            0x7f => {
                let v = self.mm_get(reg);
                match mem_addr {
                    None => {
                        self.mmx_enter();
                        self.mm_put(rm, v);
                    }
                    Some(addr) => {
                        self.write(addr, MMX_QWORD_BYTES as u8, v)?;
                        self.mmx_enter();
                    }
                }
            }
            // movntq m64, mm (the memory-only form was checked by `form_valid`)
            0xe7 => {
                let v = self.mm_get(reg);
                if let Some(addr) = mem_addr {
                    self.write(addr, MMX_QWORD_BYTES as u8, v)?;
                }
                self.mmx_enter();
            }
            // pshufw: the pshuflw word selection applied to the whole register
            0x70 => {
                let source = self.mm_or_m64(rm, mem_addr)?;
                let imm8 = i.imm as u8;
                let mut r = 0u64;
                for k in 0..MMX_WORD_COUNT {
                    let word = (source >> (16 * ((imm8 >> (2 * k)) & 3))) & 0xffff;
                    r |= word << (16 * k);
                }
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // shifts by immediate; the register is ModRM.rm
            0x71..=0x73 => {
                let x = self.mm_get(rm) as u128;
                let count = i.imm as u8 as u64;
                let Some(r) = shift_immediate(opcode, i.reg & 7, x, count) else {
                    return Err(self.unimplemented_opcode(i));
                };
                self.mmx_enter();
                self.mm_put(rm, r as u64);
            }
            // pinsrw mm, r32/m16: imm8 & 3 selects the word
            0xc4 => {
                let word = match mem_addr {
                    None => self.read_gpr(rm, 2, true),
                    Some(addr) => self.read(addr, 2)?,
                } & 0xffff;
                let shift = 16 * (i.imm as u64 & (MMX_WORD_COUNT as u64 - 1));
                let r = (self.mm_get(reg) & !(0xffff << shift)) | (word << shift);
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // pextrw r32, mm, imm8
            0xc5 => {
                let shift = 16 * (i.imm as u64 & (MMX_WORD_COUNT as u64 - 1));
                self.gpr[reg as usize] = (self.mm_get(rm) >> shift) & 0xffff;
                self.mmx_enter();
            }
            // pmovmskb r32, mm
            0xd7 => {
                let x = self.mm_get(rm);
                self.gpr[reg as usize] =
                    (0..MMX_QWORD_BYTES).fold(0, |m, k| m | (((x >> (8 * k + 7)) & 1) << k));
                self.mmx_enter();
            }
            // maskmovq mm1, mm2: stores the bytes of mm1 whose mm2 byte has its top bit set at
            // DS:rdi. Bytes that are not selected are not accessed.
            0xf7 => {
                let data = self.mm_get(reg);
                let selector = self.mm_get(rm);
                for k in 0..MMX_QWORD_BYTES as u64 {
                    if (selector >> (8 * k + 7)) & 1 != 0 {
                        let addr = self.linear_address(i, self.gpr[GPR_RDI].wrapping_add(k))?;
                        self.write(addr, 1, (data >> (8 * k)) & 0xff)?;
                    }
                }
                self.mmx_enter();
            }
            // cvtpi2ps xmm, mm/m64 (Np) and cvtpi2pd xmm, mm/m64 (66)
            0x2a => {
                let source = self.mm_or_m64(rm, mem_addr)?;
                let mut fp = FpEnv::new(self.mxcsr);
                let dst = self.xmm[reg as usize];
                let int = |k: u32| (source >> (32 * k)) as u32 as i32 as i64;
                let r = if prefix == Prefix::Np {
                    let mut r = 0;
                    for k in 0..2 {
                        lane_put(
                            &mut r,
                            SINGLE,
                            k,
                            int_to_float(&mut fp, SINGLE, int(k as u32)),
                        );
                    }
                    (dst & !LOW_64) | r
                } else {
                    let mut r = 0;
                    for k in 0..2 {
                        lane_put(
                            &mut r,
                            DOUBLE,
                            k,
                            int_to_float(&mut fp, DOUBLE, int(k as u32)),
                        );
                    }
                    r
                };
                self.fp_commit(&fp)?;
                self.mmx_enter();
                self.xmm[reg as usize] = r;
            }
            // cvtps2pi/cvttps2pi mm, xmm/m64 (Np) and cvtpd2pi/cvttpd2pi mm, xmm/m128 (66)
            0x2c | 0x2d => {
                let truncate = opcode == 0x2c;
                let mut fp = FpEnv::new(self.mxcsr);
                let format = if prefix == Prefix::Np { SINGLE } else { DOUBLE };
                let source = match mem_addr {
                    None => self.xmm[rm as usize],
                    Some(addr) if prefix == Prefix::Np => self.read(addr, 8)? as u128,
                    Some(addr) => {
                        if addr & (VECTOR_ALIGNMENT - 1) != 0 {
                            return Err(Exception::general_protection(0));
                        }
                        self.read128(addr)?
                    }
                };
                let mut r = 0u64;
                for k in 0..2 {
                    let lane = lane_get(source, format, k);
                    let integer = float_to_int(&mut fp, format, lane, 32, truncate);
                    r |= integer << (32 * k);
                }
                self.fp_commit(&fp)?;
                self.mmx_enter();
                self.mm_put(reg, r);
            }
            // F3: movq2dq xmm, mm; F2: movdq2q mm, xmm (register forms only)
            0xd6 => {
                if prefix == Prefix::F3 {
                    let v = self.mm_get(rm);
                    self.mmx_enter();
                    self.xmm[reg as usize] = v as u128;
                } else {
                    let v = self.xmm[rm as usize] as u64;
                    self.mmx_enter();
                    self.mm_put(reg, v);
                }
            }
            // everything else is a two-operand packed integer operation on mm and mm/m64
            _ => {
                // punpckl* read only 32 bits of a memory operand (mm/m32)
                let size = if matches!(opcode, 0x60..=0x62) {
                    4
                } else {
                    MMX_QWORD_BYTES as u8
                };
                let source = match mem_addr {
                    None => self.mm_get(rm),
                    Some(addr) => self.read(addr, size)?,
                };
                let r = mmx_binary(opcode, self.mm_get(reg), source);
                self.mmx_enter();
                self.mm_put(reg, r);
            }
        }
        Ok(())
    }

    /// The mm/m64 operand: register mm`rm`, or 8 bytes at `mem_addr`.
    fn mm_or_m64(&mut self, rm: u8, mem_addr: Option<u64>) -> ExecResult<u64> {
        match mem_addr {
            None => Ok(self.mm_get(rm)),
            Some(addr) => self.read(addr, MMX_QWORD_BYTES as u8),
        }
    }

    /// Executes an SSE/SSE2 instruction from the `0f xx` map. Returns `Ok(false)` for anything
    /// else, which the caller turns into #UD; on `Ok(true)` the caller sets rip to `next`.
    /// CPUID (`cpu/sys.rs`) advertises MMX, SSE and SSE2 and this covers those (the MMX register
    /// forms are in `mmx`). SSE3 and later, which have their own opcodes and mandatory prefixes,
    /// are in `sse_ext`; there is no VEX.
    ///
    /// Vector registers are `u128` in `self.xmm`. The `src!` operand is the whole register in the
    /// register form, so scalar arms truncate it to the lane they use. Every instruction that can
    /// raise an FP exception collects its flags in `fp` and commits them (possibly faulting)
    /// before the destination is written.
    pub fn sse(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let map = i.op >> 8;
        if map != OPCODE_MAP_0F {
            return self.sse_ext(i, next);
        }
        let opcode = (i.op & 0xff) as u8;
        let prefix = match (i.rep, i.p66) {
            (1, _) => Prefix::F3,
            (2, _) => Prefix::F2,
            (_, true) => Prefix::P66,
            _ => Prefix::Np,
        };
        if !allowed(opcode, prefix) {
            return self.sse_ext(i, next);
        }
        // An encoding that is invalid for this ModRM form is #UD even with CR0.TS set (#UD has
        // priority over #NM, as in QEMU).
        if !form_valid(opcode, prefix, i.md == MODRM_MOD_REGISTER, i.reg & 7) {
            return Err(Exception::invalid_opcode());
        }
        // emms is an MMX instruction: CR0.EM is #UD and CR0.TS is #NM, and CR4.OSFXSR does not
        // matter. ftw is the abridged tag word of FXSAVE: 0 marks all eight registers empty.
        if opcode == 0x77 {
            if self.cr0 & CR0_EM != 0 {
                return Err(Exception::invalid_opcode());
            }
            self.fpu_check()?;
            self.ftw = 0;
            return Ok(true);
        }
        if mmx_form(opcode, prefix) {
            self.mmx(i, next, opcode, prefix)?;
            return Ok(true);
        }
        self.sse_gate()?;
        // ModRM.reg and .rm already include REX.R/REX.B. `dst_reg` is a general register for the
        // instructions that write one (cvt*2si, movmskp*, pmovmskb, pextrw).
        let dst_reg = i.reg as usize;
        let rm_reg = i.rm as usize;
        // Linear address of the memory operand, `None` for the register form (ModRM.mod == 3).
        let mem_addr = if i.md == MODRM_MOD_REGISTER {
            None
        } else {
            Some(self.linear_address(i, self.effective_address(i, next))?)
        };
        // Every 128-bit memory operand of a legacy-SSE instruction must be 16-byte aligned (#GP(0))
        // except for the explicitly unaligned forms: movups, movupd and movdqu loads.
        let unaligned_ok = matches!(
            (opcode, prefix),
            (0x10, Prefix::Np | Prefix::P66) | (0x6f, Prefix::F3)
        ) || (self.vex_active
            // VEX drops the alignment rule for everything except the explicitly aligned
            // vmovaps/vmovapd/vmovdqa loads (stores are checked by `aligned!`)
            && !matches!((opcode, prefix), (0x28, _) | (0x6f, Prefix::P66)));
        macro_rules! src {
            ($size:expr) => {
                match mem_addr {
                    None => self.xmm[rm_reg],
                    Some(addr) => {
                        if $size == 16 {
                            if addr & (VECTOR_ALIGNMENT - 1) != 0 && !unaligned_ok {
                                return Err(Exception::general_protection(0));
                            }
                            self.read128(addr)?
                        } else {
                            self.read(addr, $size)? as u128
                        }
                    }
                }
            };
        }
        // #GP(0) for a misaligned 16-byte store destination; loads check in `src!`.
        macro_rules! aligned {
            () => {
                if let Some(addr) = mem_addr {
                    if addr & (VECTOR_ALIGNMENT - 1) != 0 {
                        return Err(Exception::general_protection(0));
                    }
                }
            };
        }
        let dst = self.xmm[dst_reg];
        let mut fp = FpEnv::new(self.mxcsr);
        let result: Option<u128> = match opcode {
            // movups, movupd, movss, movsd load. The memory form of movss/movsd zeroes the rest of
            // the register; the register form merges into the low lane.
            0x10 => Some(match prefix {
                Prefix::Np | Prefix::P66 => src!(16),
                Prefix::F3 => {
                    if mem_addr.is_some() {
                        src!(4)
                    } else {
                        (dst & !LOW_32) | (self.xmm[rm_reg] & LOW_32)
                    }
                }
                Prefix::F2 => {
                    if mem_addr.is_some() {
                        src!(8)
                    } else {
                        (dst & !LOW_64) | (self.xmm[rm_reg] & LOW_64)
                    }
                }
            }),
            // stores: movups/movupd/movss/movsd (0x11), movaps/movapd (0x29), movntps/movntpd
            // (0x2b), movdqa/movdqu (0x7f), movntdq (0xe7). Scalar register destinations merge into
            // the low lane.
            0x11 | 0x29 | 0x2b | 0x7f | 0xe7 => {
                if matches!(opcode, 0x29 | 0x2b)
                    || (opcode == 0x7f && prefix == Prefix::P66)
                    || opcode == 0xe7
                {
                    aligned!();
                }
                let size = match (opcode, prefix) {
                    (0x11, Prefix::F3) => 4,
                    (0x11, Prefix::F2) => 8,
                    _ => 16,
                };
                match mem_addr {
                    Some(addr) => match size {
                        16 => self.write128(addr, dst)?,
                        s => self.write(addr, s, dst as u64)?,
                    },
                    None => {
                        let cur = self.xmm[rm_reg];
                        self.xmm[rm_reg] = match size {
                            4 => (cur & !LOW_32) | (dst & LOW_32),
                            8 => (cur & !LOW_64) | (dst & LOW_64),
                            _ => dst,
                        };
                    }
                }
                None
            }
            // movlps/movlpd (0x12) and movhps/movhpd (0x16) loads; the register forms are movhlps
            // and movlhps.
            0x12 | 0x16 => {
                let low_half = opcode == 0x12;
                let v = if mem_addr.is_none() {
                    if low_half {
                        self.xmm[rm_reg] >> 64
                    } else {
                        self.xmm[rm_reg] & LOW_64
                    }
                } else {
                    src!(8)
                } & LOW_64;
                Some(if low_half {
                    (dst & !LOW_64) | v
                } else {
                    (dst & LOW_64) | (v << 64)
                })
            }
            // movlps/movlpd and movhps/movhpd stores; memory destination only.
            0x13 | 0x17 => {
                let addr = mem_addr.ok_or_else(Exception::invalid_opcode)?;
                let v = if opcode == 0x13 {
                    dst as u64
                } else {
                    (dst >> 64) as u64
                };
                self.write(addr, 8, v)?;
                None
            }
            // unpcklps/unpcklpd (0x14), unpckhps/unpckhpd (0x15)
            0x14 | 0x15 => {
                let s = src!(16);
                let hi = opcode == 0x15;
                Some(if prefix == Prefix::Np {
                    interleave::<u32>(dst, s, hi)
                } else {
                    interleave::<u64>(dst, s, hi)
                })
            }
            // movaps/movapd load
            0x28 => Some(src!(16)),
            // cvtsi2ss/cvtsi2sd from a 32-bit or (REX.W) 64-bit integer, rounded once under MXCSR.RC
            0x2a => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                let value = if size == 8 {
                    v as i64
                } else {
                    v as u32 as i32 as i64
                };
                let format = if prefix == Prefix::F3 { SINGLE } else { DOUBLE };
                let converted = int_to_float(&mut fp, format, value) as u128;
                Some((dst & !format.lane_mask()) | converted)
            }
            // cvttss2si/cvttsd2si (0x2c, truncates) and cvtss2si/cvtsd2si (0x2d, rounds per
            // MXCSR.RC)
            0x2c | 0x2d => {
                let format = if prefix == Prefix::F3 { SINGLE } else { DOUBLE };
                let size = if prefix == Prefix::F3 { 4 } else { 8 };
                let v = lane_get(src!(size), format, 0);
                let width = if i.opsize == 8 { 64 } else { 32 };
                let converted = float_to_int(&mut fp, format, v, width, opcode == 0x2c);
                self.fp_commit(&fp)?;
                self.gpr[dst_reg] = converted;
                None
            }
            // ucomiss/ucomisd (0x2e) and comiss/comisd (0x2f). comis also raises invalid for a
            // QNaN.
            0x2e | 0x2f => {
                let (format, size) = if prefix == Prefix::Np {
                    (SINGLE, 4)
                } else {
                    (DOUBLE, 8)
                };
                let s = lane_get(src!(size), format, 0);
                let ordering =
                    compare(&mut fp, format, lane_get(dst, format, 0), s, opcode == 0x2f);
                self.fp_commit(&fp)?;
                self.set_eflags_arith(compare_eflags(ordering));
                None
            }
            // movmskps/movmskpd: sign bits into a general register
            0x50 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] = if prefix == Prefix::Np {
                    (0..4).fold(0, |m, k| m | (((u32::get(x, k) >> 31) as u64) << k))
                } else {
                    (0..2).fold(0, |m, k| m | ((u64::get(x, k) >> 63) << k))
                };
                None
            }
            // sqrt, add, mul, sub, min, div, max in packed and scalar forms; scalar forms merge
            // into the low lane.
            0x51 | 0x58 | 0x59 | 0x5c | 0x5d | 0x5e | 0x5f => {
                let format = if matches!(prefix, Prefix::Np | Prefix::F3) {
                    SINGLE
                } else {
                    DOUBLE
                };
                if matches!(prefix, Prefix::Np | Prefix::P66) {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..format.lanes() {
                        let lane = binary_operation(
                            &mut fp,
                            format,
                            opcode,
                            lane_get(dst, format, k),
                            lane_get(s, format, k),
                        );
                        lane_put(&mut r, format, k, lane);
                    }
                    Some(r)
                } else {
                    let s = lane_get(src!(format.bytes()), format, 0);
                    let lane =
                        binary_operation(&mut fp, format, opcode, lane_get(dst, format, 0), s);
                    Some((dst & !format.lane_mask()) | lane as u128)
                }
            }
            // rsqrtps/rsqrtss (0x52), rcpps/rcpss (0x53)
            0x52 | 0x53 => {
                let square_root = opcode == 0x52;
                Some(if prefix == Prefix::F3 {
                    let s = lane_get(src!(4), SINGLE, 0);
                    (dst & !SINGLE.lane_mask()) | reciprocal(SINGLE, s, square_root) as u128
                } else {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..SINGLE.lanes() {
                        lane_put(
                            &mut r,
                            SINGLE,
                            k,
                            reciprocal(SINGLE, lane_get(s, SINGLE, k), square_root),
                        );
                    }
                    r
                })
            }
            // andps/andpd, andnps/andnpd, orps/orpd, xorps/xorpd
            0x54..=0x57 => {
                let s = src!(16);
                Some(match opcode {
                    0x54 => dst & s,
                    0x55 => !dst & s,
                    0x56 => dst | s,
                    _ => dst ^ s,
                })
            }
            // cvtps2pd, cvtpd2ps, cvtss2sd, cvtsd2ss
            0x5a => Some(match prefix {
                Prefix::Np => {
                    let s = src!(8);
                    let mut r = 0;
                    for k in 0..DOUBLE.lanes() {
                        let lane = convert_float(&mut fp, SINGLE, DOUBLE, lane_get(s, SINGLE, k));
                        lane_put(&mut r, DOUBLE, k, lane);
                    }
                    r
                }
                Prefix::P66 => {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..DOUBLE.lanes() {
                        let lane = convert_float(&mut fp, DOUBLE, SINGLE, lane_get(s, DOUBLE, k));
                        lane_put(&mut r, SINGLE, k, lane);
                    }
                    r
                }
                Prefix::F3 => {
                    let s = lane_get(src!(4), SINGLE, 0);
                    (dst & !DOUBLE.lane_mask()) | convert_float(&mut fp, SINGLE, DOUBLE, s) as u128
                }
                Prefix::F2 => {
                    let s = lane_get(src!(8), DOUBLE, 0);
                    (dst & !SINGLE.lane_mask()) | convert_float(&mut fp, DOUBLE, SINGLE, s) as u128
                }
            }),
            // cvtdq2ps, cvtps2dq (rounds per MXCSR.RC), cvttps2dq (truncates)
            0x5b => {
                let s = src!(16);
                let mut r = 0;
                for k in 0..SINGLE.lanes() {
                    let lane = lane_get(s, SINGLE, k);
                    let converted = if prefix == Prefix::Np {
                        int_to_float(&mut fp, SINGLE, lane as u32 as i32 as i64)
                    } else {
                        float_to_int(&mut fp, SINGLE, lane, 32, prefix == Prefix::F3)
                    };
                    lane_put(&mut r, SINGLE, k, converted);
                }
                Some(r)
            }
            // 0x60-0x6d unpack/pack/compare, 0x74-0x76 pcmpeq, shifts by an xmm/m128 count and
            // the rest of the SSE2 integer arithmetic and logic: see `integer_op`.
            0x60..=0x6d
            | 0x74..=0x76
            | 0xd1..=0xd5
            | 0xd8..=0xdf
            | 0xe0..=0xe5
            | 0xe8..=0xef
            | 0xf1..=0xf6
            | 0xf8..=0xfe => Some(integer_op(opcode, dst, src!(16))),
            // movd/movq xmm, r/m; zero-extends into the whole register
            0x6e => {
                let size = if i.opsize == 8 { 8 } else { 4 };
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, size, true),
                    Some(addr) => self.read(addr, size)?,
                };
                Some(v as u128)
            }
            // movdqa and movdqu load
            0x6f => Some(src!(16)),
            // pshufd, pshuflw, pshufhw: imm8 holds four 2-bit source lane indices, lowest field
            // first; pshuflw/pshufhw copy the other half of the register unchanged.
            0x70 => {
                let s = src!(16);
                let imm8 = i.imm as u8;
                Some(match prefix {
                    Prefix::P66 => shuffle_dwords(s, imm8),
                    Prefix::F2 => {
                        let mut r = s & !LOW_64;
                        for k in 0..4 {
                            u16::put(&mut r, k, u16::get(s, ((imm8 >> (2 * k)) & 3) as usize));
                        }
                        r
                    }
                    _ => {
                        let mut r = s & LOW_64;
                        for k in 0..4 {
                            u16::put(
                                &mut r,
                                4 + k,
                                u16::get(s, 4 + ((imm8 >> (2 * k)) & 3) as usize),
                            );
                        }
                        r
                    }
                })
            }
            // Shifts by immediate; ModRM.reg selects the operation (see `shift_immediate`)
            0x71..=0x73 => {
                let count = i.imm as u8 as u64;
                let Some(r) = shift_immediate(opcode, i.reg & 7, self.xmm[rm_reg], count) else {
                    // unassigned /reg values were already rejected by `form_valid`
                    return Err(self.unimplemented_opcode(i));
                };
                self.xmm[rm_reg] = r;
                None
            }
            // movq xmm, xmm/m64 (F3, zero-extends) and movd/movq r/m, xmm (66)
            0x7e => {
                if prefix == Prefix::F3 {
                    Some(src!(8) & LOW_64)
                } else {
                    let size = if i.opsize == 8 { 8 } else { 4 };
                    let v = dst as u64 & if size == 8 { u64::MAX } else { 0xffff_ffff };
                    match mem_addr {
                        None => self.write_gpr(i.rm, size, true, v),
                        Some(addr) => self.write(addr, size, v)?,
                    }
                    None
                }
            }
            // cmpps/cmppd/cmpss/cmpsd: all ones or zero per lane
            0xc2 => {
                let predicate = i.imm as u8;
                let format = if matches!(prefix, Prefix::Np | Prefix::F3) {
                    SINGLE
                } else {
                    DOUBLE
                };
                if matches!(prefix, Prefix::Np | Prefix::P66) {
                    let s = src!(16);
                    let mut r = 0;
                    for k in 0..format.lanes() {
                        let (a, b) = (lane_get(dst, format, k), lane_get(s, format, k));
                        if compare_predicate(&mut fp, format, predicate, a, b, self.vex_active) {
                            lane_put(&mut r, format, k, format.lane_mask() as u64);
                        }
                    }
                    Some(r)
                } else {
                    let s = lane_get(src!(format.bytes()), format, 0);
                    let a = lane_get(dst, format, 0);
                    let matched =
                        compare_predicate(&mut fp, format, predicate, a, s, self.vex_active);
                    Some((dst & !format.lane_mask()) | if matched { format.lane_mask() } else { 0 })
                }
            }
            // pinsrw: imm8 & 7 selects the word
            0xc4 => {
                let v = match mem_addr {
                    None => self.read_gpr(i.rm, 2, true),
                    Some(addr) => self.read(addr, 2)?,
                } as u16;
                let k = (i.imm as usize) & 7;
                let sh = k * 16;
                Some((dst & !(0xffffu128 << sh)) | ((v as u128) << sh))
            }
            // pextrw: register source only; zero-extends into the general register
            0xc5 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let k = (i.imm as usize) & 7;
                self.gpr[dst_reg] = u16::get(self.xmm[rm_reg], k) as u64;
                None
            }
            // shufps: result lanes 0-1 come from dst, 2-3 from the source. shufpd: imm8 bit 0 picks
            // the low lane from dst, bit 1 the high lane from the source.
            0xc6 => {
                let s = src!(16);
                let imm8 = i.imm as u8 as usize;
                Some(if prefix == Prefix::Np {
                    let mut r = 0;
                    u32::put(&mut r, 0, u32::get(dst, imm8 & 3));
                    u32::put(&mut r, 1, u32::get(dst, (imm8 >> 2) & 3));
                    u32::put(&mut r, 2, u32::get(s, (imm8 >> 4) & 3));
                    u32::put(&mut r, 3, u32::get(s, (imm8 >> 6) & 3));
                    r
                } else {
                    u64::get(dst, imm8 & 1) as u128 | ((u64::get(s, (imm8 >> 1) & 1) as u128) << 64)
                })
            }
            // movq xmm/m64, xmm; the register form zeroes the upper half
            0xd6 => {
                let v = dst & LOW_64;
                match mem_addr {
                    None => self.xmm[rm_reg] = v,
                    Some(addr) => self.write(addr, 8, v as u64)?,
                }
                None
            }
            // pmovmskb
            0xd7 => {
                if mem_addr.is_some() {
                    return Err(Exception::invalid_opcode());
                }
                let x = self.xmm[rm_reg];
                self.gpr[dst_reg] =
                    (0..16).fold(0, |m, k| m | (((u8::get(x, k) >> 7) as u64) << k));
                None
            }
            // cvtdq2pd (F3), cvtpd2dq (F2, MXCSR.RC), cvttpd2dq (66, truncates). The integer forms
            // fill only the low two dwords and zero the upper half.
            0xe6 => {
                let truncate = prefix == Prefix::P66;
                let s = src!(if prefix == Prefix::F3 { 8 } else { 16 });
                let mut r = 0;
                for k in 0..DOUBLE.lanes() {
                    if prefix == Prefix::F3 {
                        let integer = lane_get(s, SINGLE, k) as u32 as i32 as i64;
                        lane_put(&mut r, DOUBLE, k, int_to_float(&mut fp, DOUBLE, integer));
                    } else {
                        let lane =
                            float_to_int(&mut fp, DOUBLE, lane_get(s, DOUBLE, k), 32, truncate);
                        lane_put(&mut r, SINGLE, k, lane);
                    }
                }
                Some(r)
            }
            _ => return Ok(false),
        };
        if fp.flags != 0 {
            self.fp_commit(&fp)?;
        }
        if let Some(v) = result {
            self.xmm[dst_reg] = v;
        }
        Ok(true)
    }
}

/// Opcodes (with this prefix) whose operands are MMX registers, or for the conversions and
/// movq2dq/movdq2q a mix of MMX and xmm registers. The same opcodes with 66 are the xmm forms.
fn mmx_form(opcode: u8, prefix: Prefix) -> bool {
    match opcode {
        0x2a | 0x2c | 0x2d => matches!(prefix, Prefix::Np | Prefix::P66),
        0xd6 => matches!(prefix, Prefix::F3 | Prefix::F2),
        0x60..=0x6b
        | 0x6e
        | 0x6f
        | 0x70
        | 0x71..=0x76
        | 0x7e
        | 0x7f
        | 0xc4
        | 0xc5
        | 0xd1..=0xd5
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf7
        | 0xf8..=0xfe => prefix == Prefix::Np,
        _ => false,
    }
}

/// The packed integer instructions on 64-bit mm operands. Most are `integer_op` on a register
/// whose upper half is zero; unpack-high, pack and the other cross-lane ones need their inputs
/// rearranged because an mm register has half as many lanes as an xmm one.
fn mmx_binary(opcode: u8, dst: u64, s: u64) -> u64 {
    let (a, b) = (dst as u128, s as u128);
    let r = match opcode {
        // packs read the low half of dst then the low half of s, the layout of xmm lanes 0-7
        0x63 | 0x67 => pack_sat(a | (b << 64), 0, 16, opcode == 0x63),
        0x6b => pack_sat(a | (b << 64), 0, 32, true),
        // punpckh* of an mm register interleaves its upper 32 bits: the low-unpack of those
        0x68..=0x6a => integer_op(opcode - 8, a >> 32, b >> 32),
        _ => integer_op(opcode, a, b),
    };
    r as u64
}

/// The 0F 71/72/73 shift-by-immediate groups: /2 psrl, /4 psra, /6 psll, and for 0x73 /3 psrldq
/// and /7 pslldq (whole-register byte shifts; a count above 15 clears the register). `None` for an
/// unassigned (opcode, /reg).
fn shift_immediate(opcode: u8, group: u8, x: u128, count: u64) -> Option<u128> {
    Some(match (opcode, group) {
        (0x71, 2) => shift_logical::<u16>(x, count, false, 16),
        (0x71, 4) => sra::<u16>(x, count, 16),
        (0x71, 6) => shift_logical::<u16>(x, count, true, 16),
        (0x72, 2) => shift_logical::<u32>(x, count, false, 32),
        (0x72, 4) => sra::<u32>(x, count, 32),
        (0x72, 6) => shift_logical::<u32>(x, count, true, 32),
        (0x73, 2) => shift_logical::<u64>(x, count, false, 64),
        (0x73, 6) => shift_logical::<u64>(x, count, true, 64),
        (0x73, 3) => {
            if count > 15 {
                0
            } else {
                x >> (count * 8)
            }
        }
        (0x73, 7) => {
            if count > 15 {
                0
            } else {
                x << (count * 8)
            }
        }
        _ => return None,
    })
}

/// The two-operand packed integer instructions shared by the xmm (66) forms: unpack, pack,
/// compare, shifts by a register count (only the low 64 bits of `s` count) and arithmetic.
/// `opcode` is the 0F-map opcode byte.
fn integer_op(opcode: u8, dst: u128, s: u128) -> u128 {
    // shift count: the low quadword only
    let count = s as u64;
    match opcode {
        0x60 => interleave::<u8>(dst, s, false),
        0x61 => interleave::<u16>(dst, s, false),
        0x62 => interleave::<u32>(dst, s, false),
        0x63 => pack_sat(dst, s, 16, true),
        0x64 => zip_lanes::<u8>(dst, s, |a, b| if (a as i8) > (b as i8) { 0xff } else { 0 }),
        0x65 => zip_lanes::<u16>(
            dst,
            s,
            |a, b| if (a as i16) > (b as i16) { 0xffff } else { 0 },
        ),
        0x66 => zip_lanes::<u32>(
            dst,
            s,
            |a, b| if (a as i32) > (b as i32) { u32::MAX } else { 0 },
        ),
        0x67 => pack_sat(dst, s, 16, false),
        0x68 => interleave::<u8>(dst, s, true),
        0x69 => interleave::<u16>(dst, s, true),
        0x6a => interleave::<u32>(dst, s, true),
        0x6b => pack_sat(dst, s, 32, true),
        0x6c => interleave::<u64>(dst, s, false),
        0x6d => interleave::<u64>(dst, s, true),
        0x74 => zip_lanes::<u8>(dst, s, |a, b| if a == b { 0xff } else { 0 }),
        0x75 => zip_lanes::<u16>(dst, s, |a, b| if a == b { 0xffff } else { 0 }),
        0x76 => zip_lanes::<u32>(dst, s, |a, b| if a == b { u32::MAX } else { 0 }),
        0xd1 => shift_logical::<u16>(dst, count, false, 16),
        0xd2 => shift_logical::<u32>(dst, count, false, 32),
        0xd3 => shift_logical::<u64>(dst, count, false, 64),
        0xe1 => sra::<u16>(dst, count, 16),
        0xe2 => sra::<u32>(dst, count, 32),
        0xf1 => shift_logical::<u16>(dst, count, true, 16),
        0xf2 => shift_logical::<u32>(dst, count, true, 32),
        0xf3 => shift_logical::<u64>(dst, count, true, 64),
        0xd4 => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_add(b)),
        0xd5 => zip_lanes::<u16>(dst, s, |a, b| (a as i16).wrapping_mul(b as i16) as u16),
        0xd8 => zip_lanes::<u8>(dst, s, |a, b| a.saturating_sub(b)),
        0xd9 => zip_lanes::<u16>(dst, s, |a, b| a.saturating_sub(b)),
        0xda => zip_lanes::<u8>(dst, s, |a, b| a.min(b)),
        0xdb => dst & s,
        0xdc => zip_lanes::<u8>(dst, s, |a, b| a.saturating_add(b)),
        0xdd => zip_lanes::<u16>(dst, s, |a, b| a.saturating_add(b)),
        0xde => zip_lanes::<u8>(dst, s, |a, b| a.max(b)),
        0xdf => !dst & s,
        0xe0 => zip_lanes::<u8>(dst, s, |a, b| ((a as u16 + b as u16 + 1) >> 1) as u8),
        0xe3 => zip_lanes::<u16>(dst, s, |a, b| ((a as u32 + b as u32 + 1) >> 1) as u16),
        0xe4 => zip_lanes::<u16>(dst, s, |a, b| ((a as u32 * b as u32) >> 16) as u16),
        0xe5 => zip_lanes::<u16>(dst, s, |a, b| {
            ((a as i16 as i32 * b as i16 as i32) >> 16) as u16
        }),
        0xe8 => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_sub(b as i8) as u8),
        0xe9 => zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_sub(b as i16) as u16),
        0xea => zip_lanes::<u16>(dst, s, |a, b| (a as i16).min(b as i16) as u16),
        0xeb => dst | s,
        0xec => zip_lanes::<u8>(dst, s, |a, b| (a as i8).saturating_add(b as i8) as u8),
        0xed => zip_lanes::<u16>(dst, s, |a, b| (a as i16).saturating_add(b as i16) as u16),
        0xee => zip_lanes::<u16>(dst, s, |a, b| (a as i16).max(b as i16) as u16),
        0xef => dst ^ s,
        0xf4 => {
            let mut r = 0;
            for k in 0..2 {
                u64::put(
                    &mut r,
                    k,
                    (u32::get(dst, 2 * k) as u64) * (u32::get(s, 2 * k) as u64),
                );
            }
            r
        }
        0xf5 => {
            let mut r = 0;
            for k in 0..4 {
                let lo = (u16::get(dst, 2 * k) as i16 as i32) * (u16::get(s, 2 * k) as i16 as i32);
                let hi = (u16::get(dst, 2 * k + 1) as i16 as i32)
                    * (u16::get(s, 2 * k + 1) as i16 as i32);
                u32::put(&mut r, k, lo.wrapping_add(hi) as u32);
            }
            r
        }
        0xf6 => {
            let mut r = 0;
            for k in 0..2 {
                let sum: u64 = (0..8)
                    .map(|j| {
                        (u8::get(dst, 8 * k + j) as i32 - u8::get(s, 8 * k + j) as i32)
                            .unsigned_abs() as u64
                    })
                    .sum();
                u64::put(&mut r, k, sum);
            }
            r
        }
        0xf8 => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_sub(b)),
        0xf9 => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfa => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfb => zip_lanes::<u64>(dst, s, |a, b| a.wrapping_sub(b)),
        0xfc => zip_lanes::<u8>(dst, s, |a, b| a.wrapping_add(b)),
        0xfd => zip_lanes::<u16>(dst, s, |a, b| a.wrapping_add(b)),
        _ => zip_lanes::<u32>(dst, s, |a, b| a.wrapping_add(b)),
    }
}

/// ModRM forms that are invalid encodings: memory-only instructions with mod == 3 and
/// register-only ones with a memory operand, plus the unassigned /reg values of the shift groups
/// 0x71-0x73. `group` is ModRM.reg & 7.
fn form_valid(opcode: u8, prefix: Prefix, is_register: bool, group: u8) -> bool {
    match opcode {
        0x13 | 0x17 | 0x2b | 0xe7 => !is_register,
        // movlps/movhps have register forms (movhlps/movlhps); movlpd/movhpd do not
        0x12 | 0x16 => prefix == Prefix::Np || !is_register,
        0x50 | 0xc5 | 0xd7 | 0xf7 => is_register,
        // movq2dq/movdq2q exist only as register forms; 66 0F D6 is movq xmm/m64, xmm
        0xd6 => prefix == Prefix::P66 || is_register,
        0x71 | 0x72 => is_register && matches!(group, 2 | 4 | 6),
        // the byte shifts /3 and /7 exist only for xmm registers
        0x73 => {
            is_register
                && if prefix == Prefix::Np {
                    matches!(group, 2 | 6)
                } else {
                    matches!(group, 2 | 3 | 6 | 7)
                }
        }
        _ => true,
    }
}

/// Which mandatory prefixes each implemented opcode accepts. This is the only prefix validation:
/// `sse` assumes any (opcode, prefix) pair that gets past here is handled by its arm.
fn allowed(opcode: u8, prefix: Prefix) -> bool {
    use Prefix::*;
    if mmx_form(opcode, prefix) {
        return true;
    }
    match opcode {
        // movups/movss/movupd/movsd (load and store), sqrt, add, mul, cvtps2pd family, sub, min,
        // div, max, cmp: all four prefixes
        0x10 | 0x11 | 0x51 | 0x58 | 0x59 | 0x5a | 0x5c..=0x5f | 0xc2 => true,
        // movlps/movlpd, movhps/movhpd (and movhlps/movlhps), unpck, movaps/movapd, movnt,
        // (u)comiss/sd, movmsk, and/andn/or/xor, shufps/shufpd: packed only
        0x12
        | 0x13
        | 0x14
        | 0x15
        | 0x16
        | 0x17
        | 0x28
        | 0x29
        | 0x2b
        | 0x2e
        | 0x2f
        | 0x50
        | 0x54..=0x57
        | 0xc6 => {
            matches!(prefix, Np | P66)
        }
        // cvtsi2ss/sd, cvttss2si/cvttsd2si, cvtss2si/cvtsd2si: scalar only
        0x2a | 0x2c | 0x2d => matches!(prefix, F3 | F2),
        // rsqrtps/rsqrtss, rcpps/rcpss
        0x52 | 0x53 => matches!(prefix, Np | F3),
        // cvtdq2ps, cvtps2dq, cvttps2dq
        0x5b => prefix != F2,
        // movdqa/movdqu load, movdqa/movdqu store, movd/movq store and movq load
        0x6f | 0x7f | 0x7e => matches!(prefix, P66 | F3),
        // pshufd/pshufhw/pshuflw, cvtdq2pd/cvtpd2dq/cvttpd2dq
        0x70 | 0xe6 => prefix != Np,
        // emms (the only MMX instruction implemented)
        0x77 => prefix == Np,
        // SSE2 integer instructions, 66-prefixed XMM forms only (the unprefixed ones are MMX)
        0x60..=0x6e
        | 0x71..=0x76
        | 0xc4
        | 0xc5
        | 0xd1..=0xd6
        | 0xd7..=0xdf
        | 0xe0..=0xe5
        | 0xe7..=0xef
        | 0xf1..=0xf6
        | 0xf8..=0xfe => prefix == P66,
        _ => false,
    }
}

/// psrl*/psll* on every lane of type `T` (`width` bits). Counts of `width` or more clear the
/// register, as the SDM specifies; Rust's own shifts would panic or wrap there.
fn shift_logical<T: Lane>(x: u128, count: u64, left: bool, width: u32) -> u128 {
    if count >= width as u64 {
        return 0;
    }
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        let shifted = if left { v << count } else { v >> count };
        put_u64::<T>(&mut r, k, shifted, width);
    }
    r
}

/// psra*: arithmetic right shift. Counts of `width` or more behave as `width - 1` (all sign bits).
fn sra<T: Lane>(x: u128, count: u64, width: u32) -> u128 {
    let count = count.min(width as u64 - 1);
    let mut r = 0;
    for k in 0..T::N {
        let v = lane_u64::<T>(x, k);
        // Left-align the lane in an i64 so the arithmetic shift sign-extends it.
        let align = 64 - width;
        let shifted = (((v << align) as i64) >> align) >> count;
        put_u64::<T>(&mut r, k, shifted as u64, width);
    }
    r
}

/// Lane `k` zero-extended to u64; `Lane::get` returns the lane type itself, which the generic
/// shift code cannot do arithmetic on.
fn lane_u64<T: Lane>(x: u128, k: usize) -> u64 {
    let w = 128 / T::N;
    ((x >> (k * w)) as u64) & if w == 64 { u64::MAX } else { (1u64 << w) - 1 }
}

/// ORs the low `width` bits of `v` into lane `k` of `r`.
fn put_u64<T: Lane>(r: &mut u128, k: usize, v: u64, width: u32) {
    let m = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    *r |= ((v & m) as u128) << (k * width as usize);
}

/// pshufd: result dword `k` is source dword `order[2k+1:2k]`.
pub(crate) fn shuffle_dwords(source: u128, order: u8) -> u128 {
    const DWORDS: usize = 4;
    const SELECTOR_MASK: u8 = 3;
    let mut result = 0;
    for k in 0..DWORDS {
        let selected = (order >> (2 * k)) & SELECTOR_MASK;
        u32::put(&mut result, k, u32::get(source, selected as usize));
    }
    result
}

#[path = "evex.rs"]
pub(in crate::cpu) mod evex;
#[path = "sse_ext.rs"]
pub(in crate::cpu) mod ext;
#[path = "sse_fma.rs"]
pub(in crate::cpu) mod fma;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::fpu::MXCSR_VALID_BITS;
    use crate::testkit::*;

    const MXCSR_RESET: u32 = 0x1f80;
    const ALL_MASKS: u32 = 0x3f << MXCSR_MASK_SHIFT;
    /// Where a snippet's `[rsi]` points unless a test moves it: 16-byte aligned scratch RAM.
    const DATA: u64 = SCRATCH;
    const SINGLE_NEGATIVE_ZERO: u32 = 0x8000_0000;
    const DEFAULT_NAN_SINGLE: u32 = 0xffc0_0000;
    const DEFAULT_NAN_DOUBLE: u64 = 0xfff8_0000_0000_0000;
    const SNAN_SINGLE: u32 = 0x7f80_0001;
    const QNAN_SINGLE: u32 = 0x7fc0_0001;
    const SNAN_DOUBLE: u64 = 0x7ff0_0000_0000_0001;
    const QNAN_DOUBLE: u64 = 0x7ff8_0000_0000_0001;
    const INFINITY_SINGLE: u32 = 0x7f80_0000;
    const INFINITY_DOUBLE: u64 = 0x7ff0_0000_0000_0000;
    const MAX_SINGLE: u32 = 0x7f7f_ffff;
    const MIN_SUBNORMAL_SINGLE: u32 = 1;
    const MIN_NORMAL_SINGLE: u32 = 0x0080_0000;
    const MIN_SUBNORMAL_DOUBLE: u64 = 1;
    const MIN_NORMAL_DOUBLE: u64 = 0x0010_0000_0000_0000;
    const ONE_SINGLE: u32 = 0x3f80_0000;
    const ONE_DOUBLE: u64 = 0x3ff0_0000_0000_0000;
    const ARITH_FLAGS: u64 = ARITH;
    const SIGN_SINGLE: u32 = 1 << 31;
    const SIGN_DOUBLE: u64 = 1 << 63;
    const QUIET_BIT_SINGLE: u32 = 0x0040_0000;
    const QUIET_BIT_DOUBLE: u64 = 0x0008_0000_0000_0000;
    const HALF_SINGLE: u32 = 0x3f00_0000;
    const TWO_SINGLE: u32 = 0x4000_0000;
    const THREE_SINGLE: u32 = 0x4040_0000;
    const TWO_DOUBLE: u64 = 0x4000_0000_0000_0000;
    const THREE_DOUBLE: u64 = 0x4008_0000_0000_0000;
    /// 2^-24: exactly half an ulp of 1.0 in single precision, so 1.0 + this is a rounding tie.
    const TWO_POW_MINUS_24_SINGLE: u32 = 0x3380_0000;
    /// 1/3 correctly rounded to single precision, and the single just below it (1/3 is not exact, so
    /// each directed mode picks one of these).
    const ONE_THIRD_SINGLE_UP: u32 = 0x3eaa_aaab;
    const ONE_THIRD_SINGLE_DOWN: u32 = 0x3eaa_aaaa;
    const ONE_THIRD_DOUBLE_DOWN: u64 = 0x3fd5_5555_5555_5555;
    const ONE_THIRD_DOUBLE_UP: u64 = 0x3fd5_5555_5555_5556;
    /// 2^24 = 16777216 and the next single above it: 2^24 + 1 is the first integer that is not exact.
    const TWO_POW_24_SINGLE: u32 = 0x4b80_0000;
    const TWO_POW_24_NEXT_SINGLE: u32 = 0x4b80_0001;
    const FIRST_INEXACT_INTEGER: u32 = (1 << 24) + 1;

    fn code(asm: &str) -> &'static [u8] {
        SNIPPETS
            .iter()
            .find(|(key, _)| *key == asm)
            .unwrap_or_else(|| panic!("no assembled snippet for {asm:?}"))
            .1
    }

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }
    fn ud() -> Stop {
        fault(6, None)
    }
    fn nm() -> Stop {
        fault(7, None)
    }
    fn gp0() -> Stop {
        fault(13, Some(0))
    }
    fn xm() -> Stop {
        fault(VECTOR_SIMD_FLOATING_POINT, None)
    }

    fn lanes8(v: [u8; 16]) -> u128 {
        u128::from_le_bytes(v)
    }
    fn lanes16(v: [u16; 8]) -> u128 {
        v.iter().rev().fold(0, |acc, &x| (acc << 16) | x as u128)
    }
    fn lanes32(v: [u32; 4]) -> u128 {
        v.iter().rev().fold(0, |acc, &x| (acc << 32) | x as u128)
    }
    fn lanes64(v: [u64; 2]) -> u128 {
        ((v[1] as u128) << 64) | v[0] as u128
    }
    fn bytes(v: u128) -> [u8; 16] {
        v.to_le_bytes()
    }
    fn words(v: u128) -> [u16; 8] {
        std::array::from_fn(|k| (v >> (16 * k)) as u16)
    }
    fn dwords(v: u128) -> [u32; 4] {
        std::array::from_fn(|k| (v >> (32 * k)) as u32)
    }
    fn qwords(v: u128) -> [u64; 2] {
        [v as u64, (v >> 64) as u64]
    }
    fn singles(v: [f32; 4]) -> u128 {
        lanes32(v.map(f32::to_bits))
    }
    fn doubles(v: [f64; 2]) -> u128 {
        lanes64(v.map(f64::to_bits))
    }

    /// One CPU reused across many runs, so table-driven tests do not pay for a fresh 32 MiB RAM each
    /// time. `run` puts the machine back into the state a fresh harness has, with SSE enabled.
    struct Rig {
        h: Harness,
        cr0: u64,
        cr4: u64,
    }

    impl Rig {
        fn new() -> Rig {
            let mut h = Harness::new(&[HLT]);
            h.cpu.cr4 |= CR4_OSFXSR;
            let (cr0, cr4) = (h.cpu.cr0, h.cpu.cr4);
            Rig { h, cr0, cr4 }
        }

        fn reset(&mut self) {
            let cpu = &mut self.h.cpu;
            cpu.cr0 = self.cr0;
            cpu.cr4 = self.cr4;
            cpu.mxcsr = MXCSR_RESET;
            cpu.halted = false;
            cpu.gpr[RSI] = DATA;
            cpu.gpr[RSP] = STACK_TOP;
            cpu.set_eflags_arith(0);
            cpu.decode_cache_generation = cpu.decode_cache_generation.wrapping_add(1);
        }

        /// Runs `asm` from the current register state with the harness defaults restored first; the
        /// vector registers and general registers keep whatever the test put there.
        fn run(&mut self, asm: &str) -> Stop {
            self.reset();
            self.exec(asm)
        }

        /// Like `run` but keeps CR0, CR4 and MXCSR as the test left them.
        fn exec(&mut self, asm: &str) -> Stop {
            let bytes = code(asm);
            self.h.write(CODE_BASE, bytes);
            self.h.cpu.rip = CODE_BASE;
            self.h.cpu.halted = false;
            self.h.cpu.decode_cache_generation = self.h.cpu.decode_cache_generation.wrapping_add(1);
            self.h.run(100)
        }

        fn run_ok(&mut self, asm: &str) -> &mut Self {
            let stop = self.run(asm);
            assert_eq!(stop, Stop::Halted, "{asm}: {}", self.h.cpu.dump_regs());
            self
        }

        /// xmm0 = `dst`, xmm1 = `src`; returns xmm0 after `asm`.
        fn op(&mut self, asm: &str, dst: u128, src: u128) -> u128 {
            self.h.cpu.xmm[0] = dst;
            self.h.cpu.xmm[1] = src;
            self.run_ok(asm);
            self.h.cpu.xmm[0]
        }

        /// xmm0 = `dst`, memory at `DATA` = `src` (16 bytes); returns xmm0 after `asm`.
        fn op_mem(&mut self, asm: &str, dst: u128, src: u128) -> u128 {
            self.h.cpu.xmm[0] = dst;
            self.h.write(DATA, &src.to_le_bytes());
            self.run_ok(asm);
            self.h.cpu.xmm[0]
        }

        fn xmm(&self, index: usize) -> u128 {
            self.h.cpu.xmm[index]
        }
        fn mxcsr(&self) -> u32 {
            self.h.cpu.mxcsr
        }
        fn flags(&self) -> u64 {
            self.h.cpu.eflags_full() & ARITH_FLAGS
        }
        fn gpr(&self, index: usize) -> u64 {
            self.h.cpu.gpr[index]
        }
        fn memory(&self, address: u64) -> u128 {
            u128::from_le_bytes(self.h.read(address, 16).try_into().unwrap())
        }
    }

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
    /// Every snippet the tests run, keyed by its Intel-syntax source (without the trailing `hlt`
    /// that each one ends with). Generated with `hack/asm.sh`; `.byte` entries are encodings the
    /// assembler will not emit (invalid forms, store-direction register forms).
    const SNIPPETS: &[(&str, &[u8])] = &[
        (".byte 0x0f, 0x11, 0xc8", &[0x0f, 0x11, 0xc8, 0xf4]),
        (".byte 0x0f, 0x13, 0xc1", &[0x0f, 0x13, 0xc1, 0xf4]),
        (".byte 0x0f, 0x17, 0xc1", &[0x0f, 0x17, 0xc1, 0xf4]),
        (".byte 0x0f, 0x29, 0xc8", &[0x0f, 0x29, 0xc8, 0xf4]),
        (".byte 0x0f, 0x2b, 0xc1", &[0x0f, 0x2b, 0xc1, 0xf4]),
        (".byte 0x0f, 0x50, 0x06", &[0x0f, 0x50, 0x06, 0xf4]),
        (".byte 0x0f, 0x6c, 0xc1", &[0x0f, 0x6c, 0xc1, 0xf4]),
        (".byte 0x0f, 0x6d, 0xc1", &[0x0f, 0x6d, 0xc1, 0xf4]),
        (
            ".byte 0x0f, 0x71, 0x16, 0x01",
            &[0x0f, 0x71, 0x16, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x71, 0xc0, 0x01",
            &[0x0f, 0x71, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x71, 0xc8, 0x01",
            &[0x0f, 0x71, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x71, 0xd8, 0x01",
            &[0x0f, 0x71, 0xd8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x71, 0xe8, 0x01",
            &[0x0f, 0x71, 0xe8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x71, 0xf8, 0x01",
            &[0x0f, 0x71, 0xf8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x72, 0xc0, 0x01",
            &[0x0f, 0x72, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x72, 0xc8, 0x01",
            &[0x0f, 0x72, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x72, 0xd8, 0x01",
            &[0x0f, 0x72, 0xd8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x72, 0xe8, 0x01",
            &[0x0f, 0x72, 0xe8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x72, 0xf8, 0x01",
            &[0x0f, 0x72, 0xf8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0x16, 0x01",
            &[0x0f, 0x73, 0x16, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xc0, 0x01",
            &[0x0f, 0x73, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xc8, 0x01",
            &[0x0f, 0x73, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xd8, 0x01",
            &[0x0f, 0x73, 0xd8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xe0, 0x01",
            &[0x0f, 0x73, 0xe0, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xe8, 0x01",
            &[0x0f, 0x73, 0xe8, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0x73, 0xf8, 0x01",
            &[0x0f, 0x73, 0xf8, 0x01, 0xf4],
        ),
        (".byte 0x0f, 0x7f, 0xc8", &[0x0f, 0x7f, 0xc8, 0xf4]),
        (".byte 0x0f, 0xae, 0x26", &[0x0f, 0xae, 0x26, 0xf4]),
        (".byte 0x0f, 0xae, 0x2e", &[0x0f, 0xae, 0x2e, 0xf4]),
        (".byte 0x0f, 0xae, 0x36", &[0x0f, 0xae, 0x36, 0xf4]),
        (".byte 0x0f, 0xae, 0xc0", &[0x0f, 0xae, 0xc0, 0xf4]),
        (
            ".byte 0x0f, 0xae, 0xc8, 0x90",
            &[0x0f, 0xae, 0xc8, 0x90, 0xf4],
        ),
        (".byte 0x0f, 0xae, 0xe8", &[0x0f, 0xae, 0xe8, 0xf4]),
        (
            ".byte 0x0f, 0xc5, 0x06, 0x01",
            &[0x0f, 0xc5, 0x06, 0x01, 0xf4],
        ),
        (
            ".byte 0x0f, 0xc6, 0xc1, 0x00",
            &[0x0f, 0xc6, 0xc1, 0x00, 0xf4],
        ),
        (".byte 0x0f, 0xd6, 0xc1", &[0x0f, 0xd6, 0xc1, 0xf4]),
        (".byte 0x0f, 0xd7, 0x06", &[0x0f, 0xd7, 0x06, 0xf4]),
        (".byte 0x0f, 0xe6, 0xc1", &[0x0f, 0xe6, 0xc1, 0xf4]),
        (".byte 0x0f, 0xe7, 0xc1", &[0x0f, 0xe7, 0xc1, 0xf4]),
        (".byte 0x0f, 0xf7, 0x06", &[0x0f, 0xf7, 0x06, 0xf4]),
        (
            ".byte 0x66, 0x0f, 0x11, 0xc8",
            &[0x66, 0x0f, 0x11, 0xc8, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x12, 0xc1",
            &[0x66, 0x0f, 0x12, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x13, 0xc1",
            &[0x66, 0x0f, 0x13, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x16, 0xc1",
            &[0x66, 0x0f, 0x16, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x17, 0xc1",
            &[0x66, 0x0f, 0x17, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x29, 0xc8",
            &[0x66, 0x0f, 0x29, 0xc8, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x2b, 0xc1",
            &[0x66, 0x0f, 0x2b, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x38, 0x00, 0xc1",
            &[0x66, 0x0f, 0x38, 0x00, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x50, 0x06",
            &[0x66, 0x0f, 0x50, 0x06, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x52, 0xc1",
            &[0x66, 0x0f, 0x52, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x6e, 0xc0",
            &[0x66, 0x0f, 0x6e, 0xc0, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0x16, 0x01",
            &[0x66, 0x0f, 0x71, 0x16, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0xc0, 0x01",
            &[0x66, 0x0f, 0x71, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0xc8, 0x01",
            &[0x66, 0x0f, 0x71, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0xd8, 0x01",
            &[0x66, 0x0f, 0x71, 0xd8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0xe8, 0x01",
            &[0x66, 0x0f, 0x71, 0xe8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x71, 0xf8, 0x01",
            &[0x66, 0x0f, 0x71, 0xf8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x72, 0xc0, 0x01",
            &[0x66, 0x0f, 0x72, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x72, 0xc8, 0x01",
            &[0x66, 0x0f, 0x72, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x72, 0xd8, 0x01",
            &[0x66, 0x0f, 0x72, 0xd8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x72, 0xe8, 0x01",
            &[0x66, 0x0f, 0x72, 0xe8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x72, 0xf8, 0x01",
            &[0x66, 0x0f, 0x72, 0xf8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x73, 0x1e, 0x01",
            &[0x66, 0x0f, 0x73, 0x1e, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x73, 0xc0, 0x01",
            &[0x66, 0x0f, 0x73, 0xc0, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x73, 0xc8, 0x01",
            &[0x66, 0x0f, 0x73, 0xc8, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x73, 0xe0, 0x01",
            &[0x66, 0x0f, 0x73, 0xe0, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x73, 0xe8, 0x01",
            &[0x66, 0x0f, 0x73, 0xe8, 0x01, 0xf4],
        ),
        (".byte 0x66, 0x0f, 0x77", &[0x66, 0x0f, 0x77, 0xf4]),
        (
            ".byte 0x66, 0x0f, 0x78, 0xc1",
            &[0x66, 0x0f, 0x78, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x7c, 0xc1",
            &[0x66, 0x0f, 0x7c, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0x7f, 0xc8",
            &[0x66, 0x0f, 0x7f, 0xc8, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xc5, 0x06, 0x01",
            &[0x66, 0x0f, 0xc5, 0x06, 0x01, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xd0, 0xc1",
            &[0x66, 0x0f, 0xd0, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xd6, 0xc8",
            &[0x66, 0x0f, 0xd6, 0xc8, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xd7, 0x06",
            &[0x66, 0x0f, 0xd7, 0x06, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xe7, 0xc1",
            &[0x66, 0x0f, 0xe7, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0x0f, 0xf7, 0xc1",
            &[0x66, 0x0f, 0xf7, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0xf2, 0x0f, 0x58, 0xc1",
            &[0x66, 0xf2, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        (
            ".byte 0x66, 0xf3, 0x0f, 0x58, 0xc1",
            &[0x66, 0xf3, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x11, 0xc8",
            &[0xf2, 0x0f, 0x11, 0xc8, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x12, 0xc1",
            &[0xf2, 0x0f, 0x12, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x14, 0xc1",
            &[0xf2, 0x0f, 0x14, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x28, 0xc1",
            &[0xf2, 0x0f, 0x28, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x2e, 0xc1",
            &[0xf2, 0x0f, 0x2e, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x52, 0xc1",
            &[0xf2, 0x0f, 0x52, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x54, 0xc1",
            &[0xf2, 0x0f, 0x54, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x5b, 0xc1",
            &[0xf2, 0x0f, 0x5b, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x6f, 0xc1",
            &[0xf2, 0x0f, 0x6f, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x74, 0xc1",
            &[0xf2, 0x0f, 0x74, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x7e, 0xc1",
            &[0xf2, 0x0f, 0x7e, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0x7f, 0xc1",
            &[0xf2, 0x0f, 0x7f, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0xd0, 0xc1",
            &[0xf2, 0x0f, 0xd0, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0x0f, 0xfc, 0xc1",
            &[0xf2, 0x0f, 0xfc, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf2, 0xf3, 0x0f, 0x58, 0xc1",
            &[0xf2, 0xf3, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x11, 0xc8",
            &[0xf3, 0x0f, 0x11, 0xc8, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x12, 0xc1",
            &[0xf3, 0x0f, 0x12, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x14, 0xc1",
            &[0xf3, 0x0f, 0x14, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x16, 0xc1",
            &[0xf3, 0x0f, 0x16, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x28, 0xc1",
            &[0xf3, 0x0f, 0x28, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x2e, 0xc1",
            &[0xf3, 0x0f, 0x2e, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x54, 0xc1",
            &[0xf3, 0x0f, 0x54, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x58, 0xc1",
            &[0xf3, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x60, 0xc1",
            &[0xf3, 0x0f, 0x60, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x70, 0xc1, 0x00",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x00, 0xf4],
        ),
        (".byte 0xf3, 0x0f, 0x77", &[0xf3, 0x0f, 0x77, 0xf4]),
        (
            ".byte 0xf3, 0x0f, 0x7e, 0x06",
            &[0xf3, 0x0f, 0x7e, 0x06, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x7e, 0xc1",
            &[0xf3, 0x0f, 0x7e, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0x7f, 0xc8",
            &[0xf3, 0x0f, 0x7f, 0xc8, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0xae, 0xe8",
            &[0xf3, 0x0f, 0xae, 0xe8, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0xc6, 0xc1, 0x00",
            &[0xf3, 0x0f, 0xc6, 0xc1, 0x00, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0xd7, 0xc1",
            &[0xf3, 0x0f, 0xd7, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x0f, 0xfc, 0xc1",
            &[0xf3, 0x0f, 0xfc, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0x66, 0x0f, 0x58, 0xc1",
            &[0xf3, 0x66, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        (
            ".byte 0xf3, 0xf2, 0x0f, 0x58, 0xc1",
            &[0xf3, 0xf2, 0x0f, 0x58, 0xc1, 0xf4],
        ),
        ("addpd xmm0, xmm1", &[0x66, 0x0f, 0x58, 0xc1, 0xf4]),
        (
            "addpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x58, 0x06, 0xf4],
        ),
        ("addps xmm0, xmm1", &[0x0f, 0x58, 0xc1, 0xf4]),
        ("addps xmm0, xmmword ptr [rsi]", &[0x0f, 0x58, 0x06, 0xf4]),
        ("addps xmm9, xmm10", &[0x45, 0x0f, 0x58, 0xca, 0xf4]),
        (
            "addsd xmm0, qword ptr [rsi + rdx*8 + 8]",
            &[0xf2, 0x0f, 0x58, 0x44, 0xd6, 0x08, 0xf4],
        ),
        (
            "addsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x58, 0x06, 0xf4],
        ),
        ("addsd xmm0, xmm1", &[0xf2, 0x0f, 0x58, 0xc1, 0xf4]),
        ("addsd xmm15, xmm8", &[0xf2, 0x45, 0x0f, 0x58, 0xf8, 0xf4]),
        (
            "addss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x58, 0x06, 0xf4],
        ),
        ("addss xmm0, xmm1", &[0xf3, 0x0f, 0x58, 0xc1, 0xf4]),
        ("andnpd xmm0, xmm1", &[0x66, 0x0f, 0x55, 0xc1, 0xf4]),
        (
            "andnpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x55, 0x06, 0xf4],
        ),
        ("andnps xmm0, xmm1", &[0x0f, 0x55, 0xc1, 0xf4]),
        ("andnps xmm0, xmmword ptr [rsi]", &[0x0f, 0x55, 0x06, 0xf4]),
        ("andpd xmm0, xmm1", &[0x66, 0x0f, 0x54, 0xc1, 0xf4]),
        (
            "andpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x54, 0x06, 0xf4],
        ),
        ("andps xmm0, xmm1", &[0x0f, 0x54, 0xc1, 0xf4]),
        ("andps xmm0, xmmword ptr [rsi]", &[0x0f, 0x54, 0x06, 0xf4]),
        ("clflush byte ptr [rsi]", &[0x0f, 0xae, 0x3e, 0xf4]),
        ("cmppd xmm0, xmm1, 0", &[0x66, 0x0f, 0xc2, 0xc1, 0x00, 0xf4]),
        ("cmppd xmm0, xmm1, 1", &[0x66, 0x0f, 0xc2, 0xc1, 0x01, 0xf4]),
        ("cmppd xmm0, xmm1, 2", &[0x66, 0x0f, 0xc2, 0xc1, 0x02, 0xf4]),
        ("cmppd xmm0, xmm1, 3", &[0x66, 0x0f, 0xc2, 0xc1, 0x03, 0xf4]),
        ("cmppd xmm0, xmm1, 4", &[0x66, 0x0f, 0xc2, 0xc1, 0x04, 0xf4]),
        ("cmppd xmm0, xmm1, 5", &[0x66, 0x0f, 0xc2, 0xc1, 0x05, 0xf4]),
        ("cmppd xmm0, xmm1, 6", &[0x66, 0x0f, 0xc2, 0xc1, 0x06, 0xf4]),
        ("cmppd xmm0, xmm1, 7", &[0x66, 0x0f, 0xc2, 0xc1, 0x07, 0xf4]),
        (
            "cmppd xmm0, xmmword ptr [rsi], 1",
            &[0x66, 0x0f, 0xc2, 0x06, 0x01, 0xf4],
        ),
        ("cmpps xmm0, xmm1, 0", &[0x0f, 0xc2, 0xc1, 0x00, 0xf4]),
        ("cmpps xmm0, xmm1, 1", &[0x0f, 0xc2, 0xc1, 0x01, 0xf4]),
        ("cmpps xmm0, xmm1, 2", &[0x0f, 0xc2, 0xc1, 0x02, 0xf4]),
        ("cmpps xmm0, xmm1, 3", &[0x0f, 0xc2, 0xc1, 0x03, 0xf4]),
        ("cmpps xmm0, xmm1, 4", &[0x0f, 0xc2, 0xc1, 0x04, 0xf4]),
        ("cmpps xmm0, xmm1, 5", &[0x0f, 0xc2, 0xc1, 0x05, 0xf4]),
        ("cmpps xmm0, xmm1, 6", &[0x0f, 0xc2, 0xc1, 0x06, 0xf4]),
        ("cmpps xmm0, xmm1, 7", &[0x0f, 0xc2, 0xc1, 0x07, 0xf4]),
        ("cmpps xmm0, xmm1, 8", &[0x0f, 0xc2, 0xc1, 0x08, 0xf4]),
        (
            "cmpps xmm0, xmmword ptr [rsi], 1",
            &[0x0f, 0xc2, 0x06, 0x01, 0xf4],
        ),
        (
            "cmpsd xmm0, qword ptr [rsi], 1",
            &[0xf2, 0x0f, 0xc2, 0x06, 0x01, 0xf4],
        ),
        ("cmpsd xmm0, xmm1, 0", &[0xf2, 0x0f, 0xc2, 0xc1, 0x00, 0xf4]),
        ("cmpsd xmm0, xmm1, 1", &[0xf2, 0x0f, 0xc2, 0xc1, 0x01, 0xf4]),
        ("cmpsd xmm0, xmm1, 2", &[0xf2, 0x0f, 0xc2, 0xc1, 0x02, 0xf4]),
        ("cmpsd xmm0, xmm1, 3", &[0xf2, 0x0f, 0xc2, 0xc1, 0x03, 0xf4]),
        ("cmpsd xmm0, xmm1, 4", &[0xf2, 0x0f, 0xc2, 0xc1, 0x04, 0xf4]),
        ("cmpsd xmm0, xmm1, 5", &[0xf2, 0x0f, 0xc2, 0xc1, 0x05, 0xf4]),
        ("cmpsd xmm0, xmm1, 6", &[0xf2, 0x0f, 0xc2, 0xc1, 0x06, 0xf4]),
        ("cmpsd xmm0, xmm1, 7", &[0xf2, 0x0f, 0xc2, 0xc1, 0x07, 0xf4]),
        (
            "cmpss xmm0, dword ptr [rsi], 1",
            &[0xf3, 0x0f, 0xc2, 0x06, 0x01, 0xf4],
        ),
        ("cmpss xmm0, xmm1, 0", &[0xf3, 0x0f, 0xc2, 0xc1, 0x00, 0xf4]),
        ("cmpss xmm0, xmm1, 1", &[0xf3, 0x0f, 0xc2, 0xc1, 0x01, 0xf4]),
        ("cmpss xmm0, xmm1, 2", &[0xf3, 0x0f, 0xc2, 0xc1, 0x02, 0xf4]),
        ("cmpss xmm0, xmm1, 3", &[0xf3, 0x0f, 0xc2, 0xc1, 0x03, 0xf4]),
        ("cmpss xmm0, xmm1, 4", &[0xf3, 0x0f, 0xc2, 0xc1, 0x04, 0xf4]),
        ("cmpss xmm0, xmm1, 5", &[0xf3, 0x0f, 0xc2, 0xc1, 0x05, 0xf4]),
        ("cmpss xmm0, xmm1, 6", &[0xf3, 0x0f, 0xc2, 0xc1, 0x06, 0xf4]),
        ("cmpss xmm0, xmm1, 7", &[0xf3, 0x0f, 0xc2, 0xc1, 0x07, 0xf4]),
        (
            "comisd xmm0, qword ptr [rsi]",
            &[0x66, 0x0f, 0x2f, 0x06, 0xf4],
        ),
        ("comisd xmm0, xmm1", &[0x66, 0x0f, 0x2f, 0xc1, 0xf4]),
        ("comiss xmm0, dword ptr [rsi]", &[0x0f, 0x2f, 0x06, 0xf4]),
        ("comiss xmm0, xmm1", &[0x0f, 0x2f, 0xc1, 0xf4]),
        (
            "cvtdq2pd xmm0, qword ptr [rsi]",
            &[0xf3, 0x0f, 0xe6, 0x06, 0xf4],
        ),
        ("cvtdq2pd xmm0, xmm1", &[0xf3, 0x0f, 0xe6, 0xc1, 0xf4]),
        ("cvtdq2ps xmm0, xmm1", &[0x0f, 0x5b, 0xc1, 0xf4]),
        (
            "cvtdq2ps xmm0, xmmword ptr [rsi]",
            &[0x0f, 0x5b, 0x06, 0xf4],
        ),
        ("cvtpd2dq xmm0, xmm1", &[0xf2, 0x0f, 0xe6, 0xc1, 0xf4]),
        (
            "cvtpd2dq xmm0, xmmword ptr [rsi]",
            &[0xf2, 0x0f, 0xe6, 0x06, 0xf4],
        ),
        ("cvtpd2pi mm0, xmm1", &[0x66, 0x0f, 0x2d, 0xc1, 0xf4]),
        (
            "cvtpd2pi mm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x2d, 0x06, 0xf4],
        ),
        ("cvtpd2ps xmm0, xmm1", &[0x66, 0x0f, 0x5a, 0xc1, 0xf4]),
        (
            "cvtpd2ps xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5a, 0x06, 0xf4],
        ),
        ("cvtpi2pd xmm0, mm1", &[0x66, 0x0f, 0x2a, 0xc1, 0xf4]),
        (
            "cvtpi2pd xmm0, qword ptr [rsi]",
            &[0x66, 0x0f, 0x2a, 0x06, 0xf4],
        ),
        ("cvtpi2ps xmm0, mm1", &[0x0f, 0x2a, 0xc1, 0xf4]),
        ("cvtpi2ps xmm0, qword ptr [rsi]", &[0x0f, 0x2a, 0x06, 0xf4]),
        ("cvtpi2ps xmm9, mm3", &[0x44, 0x0f, 0x2a, 0xcb, 0xf4]),
        ("cvtps2dq xmm0, xmm1", &[0x66, 0x0f, 0x5b, 0xc1, 0xf4]),
        (
            "cvtps2dq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5b, 0x06, 0xf4],
        ),
        ("cvtps2pd xmm0, qword ptr [rsi]", &[0x0f, 0x5a, 0x06, 0xf4]),
        ("cvtps2pd xmm0, xmm1", &[0x0f, 0x5a, 0xc1, 0xf4]),
        ("cvtps2pi mm0, qword ptr [rsi]", &[0x0f, 0x2d, 0x06, 0xf4]),
        ("cvtps2pi mm0, xmm1", &[0x0f, 0x2d, 0xc1, 0xf4]),
        (
            "cvtsd2si eax, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x2d, 0x06, 0xf4],
        ),
        ("cvtsd2si eax, xmm1", &[0xf2, 0x0f, 0x2d, 0xc1, 0xf4]),
        (
            "cvtsd2si rax, qword ptr [rsi]",
            &[0xf2, 0x48, 0x0f, 0x2d, 0x06, 0xf4],
        ),
        ("cvtsd2si rax, xmm1", &[0xf2, 0x48, 0x0f, 0x2d, 0xc1, 0xf4]),
        (
            "cvtsd2ss xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x5a, 0x06, 0xf4],
        ),
        ("cvtsd2ss xmm0, xmm1", &[0xf2, 0x0f, 0x5a, 0xc1, 0xf4]),
        (
            "cvtsi2sd xmm0, dword ptr [rsi]",
            &[0xf2, 0x0f, 0x2a, 0x06, 0xf4],
        ),
        ("cvtsi2sd xmm0, eax", &[0xf2, 0x0f, 0x2a, 0xc0, 0xf4]),
        (
            "cvtsi2sd xmm0, qword ptr [rsi]",
            &[0xf2, 0x48, 0x0f, 0x2a, 0x06, 0xf4],
        ),
        ("cvtsi2sd xmm0, rax", &[0xf2, 0x48, 0x0f, 0x2a, 0xc0, 0xf4]),
        ("cvtsi2sd xmm9, r10", &[0xf2, 0x4d, 0x0f, 0x2a, 0xca, 0xf4]),
        (
            "cvtsi2ss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x2a, 0x06, 0xf4],
        ),
        ("cvtsi2ss xmm0, eax", &[0xf3, 0x0f, 0x2a, 0xc0, 0xf4]),
        (
            "cvtsi2ss xmm0, qword ptr [rsi]",
            &[0xf3, 0x48, 0x0f, 0x2a, 0x06, 0xf4],
        ),
        ("cvtsi2ss xmm0, rax", &[0xf3, 0x48, 0x0f, 0x2a, 0xc0, 0xf4]),
        (
            "cvtss2sd xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x5a, 0x06, 0xf4],
        ),
        ("cvtss2sd xmm0, xmm1", &[0xf3, 0x0f, 0x5a, 0xc1, 0xf4]),
        (
            "cvtss2si eax, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x2d, 0x06, 0xf4],
        ),
        ("cvtss2si eax, xmm1", &[0xf3, 0x0f, 0x2d, 0xc1, 0xf4]),
        (
            "cvtss2si rax, dword ptr [rsi]",
            &[0xf3, 0x48, 0x0f, 0x2d, 0x06, 0xf4],
        ),
        ("cvtss2si rax, xmm1", &[0xf3, 0x48, 0x0f, 0x2d, 0xc1, 0xf4]),
        ("cvttpd2dq xmm0, xmm1", &[0x66, 0x0f, 0xe6, 0xc1, 0xf4]),
        (
            "cvttpd2dq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe6, 0x06, 0xf4],
        ),
        ("cvttpd2pi mm0, xmm1", &[0x66, 0x0f, 0x2c, 0xc1, 0xf4]),
        (
            "cvttpd2pi mm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x2c, 0x06, 0xf4],
        ),
        ("cvttps2dq xmm0, xmm1", &[0xf3, 0x0f, 0x5b, 0xc1, 0xf4]),
        (
            "cvttps2dq xmm0, xmmword ptr [rsi]",
            &[0xf3, 0x0f, 0x5b, 0x06, 0xf4],
        ),
        ("cvttps2pi mm0, qword ptr [rsi]", &[0x0f, 0x2c, 0x06, 0xf4]),
        ("cvttps2pi mm0, xmm1", &[0x0f, 0x2c, 0xc1, 0xf4]),
        ("cvttps2pi mm5, xmm10", &[0x41, 0x0f, 0x2c, 0xea, 0xf4]),
        (
            "cvttsd2si eax, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x2c, 0x06, 0xf4],
        ),
        ("cvttsd2si eax, xmm1", &[0xf2, 0x0f, 0x2c, 0xc1, 0xf4]),
        (
            "cvttsd2si r9d, xmm10",
            &[0xf2, 0x45, 0x0f, 0x2c, 0xca, 0xf4],
        ),
        (
            "cvttsd2si rax, qword ptr [rsi]",
            &[0xf2, 0x48, 0x0f, 0x2c, 0x06, 0xf4],
        ),
        ("cvttsd2si rax, xmm1", &[0xf2, 0x48, 0x0f, 0x2c, 0xc1, 0xf4]),
        (
            "cvttss2si eax, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x2c, 0x06, 0xf4],
        ),
        ("cvttss2si eax, xmm1", &[0xf3, 0x0f, 0x2c, 0xc1, 0xf4]),
        (
            "cvttss2si rax, dword ptr [rsi]",
            &[0xf3, 0x48, 0x0f, 0x2c, 0x06, 0xf4],
        ),
        ("cvttss2si rax, xmm1", &[0xf3, 0x48, 0x0f, 0x2c, 0xc1, 0xf4]),
        ("divpd xmm0, xmm1", &[0x66, 0x0f, 0x5e, 0xc1, 0xf4]),
        (
            "divpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5e, 0x06, 0xf4],
        ),
        ("divps xmm0, xmm1", &[0x0f, 0x5e, 0xc1, 0xf4]),
        ("divps xmm0, xmmword ptr [rsi]", &[0x0f, 0x5e, 0x06, 0xf4]),
        (
            "divsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x5e, 0x06, 0xf4],
        ),
        ("divsd xmm0, xmm1", &[0xf2, 0x0f, 0x5e, 0xc1, 0xf4]),
        (
            "divss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x5e, 0x06, 0xf4],
        ),
        ("divss xmm0, xmm1", &[0xf3, 0x0f, 0x5e, 0xc1, 0xf4]),
        ("emms", &[0x0f, 0x77, 0xf4]),
        (
            "fld1; fld1; movq mm0, mm1; emms",
            &[0xd9, 0xe8, 0xd9, 0xe8, 0x0f, 0x6f, 0xc1, 0x0f, 0x77, 0xf4],
        ),
        (
            "fld1; movq rax, mm7",
            &[0xd9, 0xe8, 0x48, 0x0f, 0x7e, 0xf8, 0xf4],
        ),
        (
            "fldz; fld1; movq mm2, mm0; movq mm3, mm1; emms",
            &[
                0xd9, 0xee, 0xd9, 0xe8, 0x0f, 0x6f, 0xd0, 0x0f, 0x6f, 0xd9, 0x0f, 0x77, 0xf4,
            ],
        ),
        ("fxrstor [rsi]", &[0x0f, 0xae, 0x0e, 0xf4]),
        ("fxsave [rsi]", &[0x0f, 0xae, 0x06, 0xf4]),
        ("ldmxcsr dword ptr [rsi]", &[0x0f, 0xae, 0x16, 0xf4]),
        (
            "ldmxcsr dword ptr [rsi]; stmxcsr dword ptr [rsi+8]",
            &[0x0f, 0xae, 0x16, 0x0f, 0xae, 0x5e, 0x08, 0xf4],
        ),
        ("lfence", &[0x0f, 0xae, 0xe8, 0xf4]),
        ("maskmovq mm0, mm1", &[0x0f, 0xf7, 0xc1, 0xf4]),
        ("maxpd xmm0, xmm1", &[0x66, 0x0f, 0x5f, 0xc1, 0xf4]),
        (
            "maxpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5f, 0x06, 0xf4],
        ),
        ("maxps xmm0, xmm1", &[0x0f, 0x5f, 0xc1, 0xf4]),
        ("maxps xmm0, xmmword ptr [rsi]", &[0x0f, 0x5f, 0x06, 0xf4]),
        (
            "maxsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x5f, 0x06, 0xf4],
        ),
        ("maxsd xmm0, xmm1", &[0xf2, 0x0f, 0x5f, 0xc1, 0xf4]),
        (
            "maxss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x5f, 0x06, 0xf4],
        ),
        ("maxss xmm0, xmm1", &[0xf3, 0x0f, 0x5f, 0xc1, 0xf4]),
        ("mfence", &[0x0f, 0xae, 0xf0, 0xf4]),
        ("minpd xmm0, xmm1", &[0x66, 0x0f, 0x5d, 0xc1, 0xf4]),
        (
            "minpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5d, 0x06, 0xf4],
        ),
        ("minps xmm0, xmm1", &[0x0f, 0x5d, 0xc1, 0xf4]),
        ("minps xmm0, xmmword ptr [rsi]", &[0x0f, 0x5d, 0x06, 0xf4]),
        (
            "minsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x5d, 0x06, 0xf4],
        ),
        ("minsd xmm0, xmm1", &[0xf2, 0x0f, 0x5d, 0xc1, 0xf4]),
        (
            "minss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x5d, 0x06, 0xf4],
        ),
        ("minss xmm0, xmm1", &[0xf3, 0x0f, 0x5d, 0xc1, 0xf4]),
        ("movapd xmm0, xmm1", &[0x66, 0x0f, 0x28, 0xc1, 0xf4]),
        (
            "movapd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x28, 0x06, 0xf4],
        ),
        (
            "movapd xmmword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x29, 0x06, 0xf4],
        ),
        ("movaps xmm0, xmm1", &[0x0f, 0x28, 0xc1, 0xf4]),
        (
            "movaps xmm0, xmmword ptr [rip + 10]",
            &[0x0f, 0x28, 0x05, 0x0a, 0x00, 0x00, 0x00, 0xf4],
        ),
        (
            "movaps xmm0, xmmword ptr [rip + 9]",
            &[0x0f, 0x28, 0x05, 0x09, 0x00, 0x00, 0x00, 0xf4],
        ),
        (
            "movaps xmm0, xmmword ptr [rsi + rdx*8 + 16]",
            &[0x0f, 0x28, 0x44, 0xd6, 0x10, 0xf4],
        ),
        ("movaps xmm0, xmmword ptr [rsi]", &[0x0f, 0x28, 0x06, 0xf4]),
        (
            "movaps xmm0, xmmword ptr fs:[rsi]",
            &[0x64, 0x0f, 0x28, 0x06, 0xf4],
        ),
        (
            "movaps xmmword ptr [rsi + rdx*8 + 16], xmm0",
            &[0x0f, 0x29, 0x44, 0xd6, 0x10, 0xf4],
        ),
        ("movaps xmmword ptr [rsi], xmm0", &[0x0f, 0x29, 0x06, 0xf4]),
        ("movd dword ptr [rsi], mm0", &[0x0f, 0x7e, 0x06, 0xf4]),
        (
            "movd dword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x7e, 0x06, 0xf4],
        ),
        ("movd eax, mm0", &[0x0f, 0x7e, 0xc0, 0xf4]),
        ("movd eax, xmm0", &[0x66, 0x0f, 0x7e, 0xc0, 0xf4]),
        ("movd mm0, dword ptr [rsi]", &[0x0f, 0x6e, 0x06, 0xf4]),
        ("movd mm0, eax", &[0x0f, 0x6e, 0xc0, 0xf4]),
        ("movd mm3, r8d", &[0x41, 0x0f, 0x6e, 0xd8, 0xf4]),
        ("movd r9d, mm3", &[0x41, 0x0f, 0x7e, 0xd9, 0xf4]),
        (
            "movd xmm0, dword ptr [rsi]",
            &[0x66, 0x0f, 0x6e, 0x06, 0xf4],
        ),
        ("movd xmm0, eax", &[0x66, 0x0f, 0x6e, 0xc0, 0xf4]),
        ("movd xmm9, r8d", &[0x66, 0x45, 0x0f, 0x6e, 0xc8, 0xf4]),
        ("movdq2q mm0, xmm1", &[0xf2, 0x0f, 0xd6, 0xc1, 0xf4]),
        ("movdq2q mm5, xmm10", &[0xf2, 0x41, 0x0f, 0xd6, 0xea, 0xf4]),
        ("movdqa xmm0, xmm1", &[0x66, 0x0f, 0x6f, 0xc1, 0xf4]),
        (
            "movdqa xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x6f, 0x06, 0xf4],
        ),
        (
            "movdqa xmmword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x7f, 0x06, 0xf4],
        ),
        ("movdqu xmm0, xmm1", &[0xf3, 0x0f, 0x6f, 0xc1, 0xf4]),
        (
            "movdqu xmm0, xmmword ptr [rsi]",
            &[0xf3, 0x0f, 0x6f, 0x06, 0xf4],
        ),
        (
            "movdqu xmmword ptr [rsi], xmm0",
            &[0xf3, 0x0f, 0x7f, 0x06, 0xf4],
        ),
        ("movhlps xmm0, xmm1", &[0x0f, 0x12, 0xc1, 0xf4]),
        (
            "movhpd qword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x17, 0x06, 0xf4],
        ),
        (
            "movhpd xmm0, qword ptr [rsi]",
            &[0x66, 0x0f, 0x16, 0x06, 0xf4],
        ),
        ("movhps qword ptr [rsi], xmm0", &[0x0f, 0x17, 0x06, 0xf4]),
        ("movhps xmm0, qword ptr [rsi]", &[0x0f, 0x16, 0x06, 0xf4]),
        ("movlhps xmm0, xmm1", &[0x0f, 0x16, 0xc1, 0xf4]),
        (
            "movlpd qword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x13, 0x06, 0xf4],
        ),
        (
            "movlpd xmm0, qword ptr [rsi]",
            &[0x66, 0x0f, 0x12, 0x06, 0xf4],
        ),
        ("movlps qword ptr [rsi], xmm0", &[0x0f, 0x13, 0x06, 0xf4]),
        ("movlps xmm0, qword ptr [rsi]", &[0x0f, 0x12, 0x06, 0xf4]),
        ("movmskpd eax, xmm1", &[0x66, 0x0f, 0x50, 0xc1, 0xf4]),
        ("movmskps eax, xmm1", &[0x0f, 0x50, 0xc1, 0xf4]),
        ("movmskps r9d, xmm10", &[0x45, 0x0f, 0x50, 0xca, 0xf4]),
        ("movmskps rax, xmm1", &[0x0f, 0x50, 0xc1, 0xf4]),
        (
            "movntdq xmmword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0xe7, 0x06, 0xf4],
        ),
        (
            "movntpd xmmword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x2b, 0x06, 0xf4],
        ),
        ("movntps xmmword ptr [rsi], xmm0", &[0x0f, 0x2b, 0x06, 0xf4]),
        ("movntq qword ptr [rsi], mm0", &[0x0f, 0xe7, 0x06, 0xf4]),
        ("movq mm0, mm1", &[0x0f, 0x6f, 0xc1, 0xf4]),
        ("movq mm0, qword ptr [rsi]", &[0x0f, 0x6f, 0x06, 0xf4]),
        (
            "movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]",
            &[
                0x0f, 0x6f, 0x06, 0x0f, 0xae, 0x86, 0x00, 0x01, 0x00, 0x00, 0xf4,
            ],
        ),
        (
            "movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]; emms; fxrstor [rsi + 0x100]",
            &[
                0x0f, 0x6f, 0x06, 0x0f, 0xae, 0x86, 0x00, 0x01, 0x00, 0x00, 0x0f, 0x77, 0x0f, 0xae,
                0x8e, 0x00, 0x01, 0x00, 0x00, 0xf4,
            ],
        ),
        (
            "movq mm0, qword ptr [rsi]; movq mm1, qword ptr [rsi + 8]; paddb mm0, mm1; movq qword ptr [rsi + 16], mm0; emms",
            &[
                0x0f, 0x6f, 0x06, 0x0f, 0x6f, 0x4e, 0x08, 0x0f, 0xfc, 0xc1, 0x0f, 0x7f, 0x46, 0x10,
                0x0f, 0x77, 0xf4,
            ],
        ),
        ("movq mm0, rax", &[0x48, 0x0f, 0x6e, 0xc0, 0xf4]),
        ("movq mm3, r9", &[0x49, 0x0f, 0x6e, 0xd9, 0xf4]),
        ("movq mm7, mm2", &[0x0f, 0x6f, 0xfa, 0xf4]),
        ("movq qword ptr [rsi], mm0", &[0x0f, 0x7f, 0x06, 0xf4]),
        (
            "movq qword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0xd6, 0x06, 0xf4],
        ),
        ("movq r10, mm3", &[0x49, 0x0f, 0x7e, 0xda, 0xf4]),
        ("movq r9, xmm10", &[0x66, 0x4d, 0x0f, 0x7e, 0xd1, 0xf4]),
        ("movq rax, mm0", &[0x48, 0x0f, 0x7e, 0xc0, 0xf4]),
        ("movq rax, mm3", &[0x48, 0x0f, 0x7e, 0xd8, 0xf4]),
        ("movq rax, xmm0", &[0x66, 0x48, 0x0f, 0x7e, 0xc0, 0xf4]),
        (
            "movq xmm0, qword ptr [rsi]",
            &[0xf3, 0x0f, 0x7e, 0x06, 0xf4],
        ),
        ("movq xmm0, rax", &[0x66, 0x48, 0x0f, 0x6e, 0xc0, 0xf4]),
        ("movq xmm0, xmm1", &[0xf3, 0x0f, 0x7e, 0xc1, 0xf4]),
        ("movq2dq xmm0, mm1", &[0xf3, 0x0f, 0xd6, 0xc1, 0xf4]),
        ("movq2dq xmm9, mm3", &[0xf3, 0x44, 0x0f, 0xd6, 0xcb, 0xf4]),
        (
            "movsd qword ptr [rsi], xmm0",
            &[0xf2, 0x0f, 0x11, 0x06, 0xf4],
        ),
        (
            "movsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x10, 0x06, 0xf4],
        ),
        ("movsd xmm0, xmm1", &[0xf2, 0x0f, 0x10, 0xc1, 0xf4]),
        (
            "movss dword ptr [rsi], xmm0",
            &[0xf3, 0x0f, 0x11, 0x06, 0xf4],
        ),
        (
            "movss xmm0, dword ptr [rsi + 0x100]",
            &[0xf3, 0x0f, 0x10, 0x86, 0x00, 0x01, 0x00, 0x00, 0xf4],
        ),
        (
            "movss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x10, 0x06, 0xf4],
        ),
        ("movss xmm0, xmm1", &[0xf3, 0x0f, 0x10, 0xc1, 0xf4]),
        ("movupd xmm0, xmm1", &[0x66, 0x0f, 0x10, 0xc1, 0xf4]),
        (
            "movupd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x10, 0x06, 0xf4],
        ),
        (
            "movupd xmmword ptr [rsi], xmm0",
            &[0x66, 0x0f, 0x11, 0x06, 0xf4],
        ),
        ("movups xmm0, xmm1", &[0x0f, 0x10, 0xc1, 0xf4]),
        ("movups xmm0, xmmword ptr [rsi]", &[0x0f, 0x10, 0x06, 0xf4]),
        (
            "movups xmm0, xmmword ptr [rsi]; movups xmm1, xmmword ptr [rsi + 16]",
            &[0x0f, 0x10, 0x06, 0x0f, 0x10, 0x4e, 0x10, 0xf4],
        ),
        (
            "movups xmm0, xmmword ptr fs:[rsi]",
            &[0x64, 0x0f, 0x10, 0x06, 0xf4],
        ),
        ("movups xmm9, xmm10", &[0x45, 0x0f, 0x10, 0xca, 0xf4]),
        ("movups xmmword ptr [rsi], xmm0", &[0x0f, 0x11, 0x06, 0xf4]),
        ("mulpd xmm0, xmm1", &[0x66, 0x0f, 0x59, 0xc1, 0xf4]),
        (
            "mulpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x59, 0x06, 0xf4],
        ),
        ("mulps xmm0, xmm1", &[0x0f, 0x59, 0xc1, 0xf4]),
        ("mulps xmm0, xmmword ptr [rsi]", &[0x0f, 0x59, 0x06, 0xf4]),
        (
            "mulsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x59, 0x06, 0xf4],
        ),
        ("mulsd xmm0, xmm1", &[0xf2, 0x0f, 0x59, 0xc1, 0xf4]),
        (
            "mulss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x59, 0x06, 0xf4],
        ),
        ("mulss xmm0, xmm1", &[0xf3, 0x0f, 0x59, 0xc1, 0xf4]),
        ("orpd xmm0, xmm1", &[0x66, 0x0f, 0x56, 0xc1, 0xf4]),
        (
            "orpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x56, 0x06, 0xf4],
        ),
        ("orps xmm0, xmm1", &[0x0f, 0x56, 0xc1, 0xf4]),
        ("orps xmm0, xmmword ptr [rsi]", &[0x0f, 0x56, 0x06, 0xf4]),
        ("packssdw mm0, mm0", &[0x0f, 0x6b, 0xc0, 0xf4]),
        ("packssdw mm0, mm1", &[0x0f, 0x6b, 0xc1, 0xf4]),
        ("packssdw mm0, qword ptr [rsi]", &[0x0f, 0x6b, 0x06, 0xf4]),
        ("packssdw xmm0, xmm1", &[0x66, 0x0f, 0x6b, 0xc1, 0xf4]),
        (
            "packssdw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x6b, 0x06, 0xf4],
        ),
        ("packsswb mm0, mm0", &[0x0f, 0x63, 0xc0, 0xf4]),
        ("packsswb mm0, mm1", &[0x0f, 0x63, 0xc1, 0xf4]),
        ("packsswb mm0, qword ptr [rsi]", &[0x0f, 0x63, 0x06, 0xf4]),
        ("packsswb xmm0, xmm1", &[0x66, 0x0f, 0x63, 0xc1, 0xf4]),
        (
            "packsswb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x63, 0x06, 0xf4],
        ),
        ("packuswb mm0, mm0", &[0x0f, 0x67, 0xc0, 0xf4]),
        ("packuswb mm0, mm1", &[0x0f, 0x67, 0xc1, 0xf4]),
        ("packuswb mm0, qword ptr [rsi]", &[0x0f, 0x67, 0x06, 0xf4]),
        ("packuswb xmm0, xmm1", &[0x66, 0x0f, 0x67, 0xc1, 0xf4]),
        (
            "packuswb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x67, 0x06, 0xf4],
        ),
        ("paddb mm0, mm0", &[0x0f, 0xfc, 0xc0, 0xf4]),
        ("paddb mm0, mm1", &[0x0f, 0xfc, 0xc1, 0xf4]),
        (
            "paddb mm0, mm1; emms",
            &[0x0f, 0xfc, 0xc1, 0x0f, 0x77, 0xf4],
        ),
        (
            "paddb mm0, mm1; emms; fld1",
            &[0x0f, 0xfc, 0xc1, 0x0f, 0x77, 0xd9, 0xe8, 0xf4],
        ),
        (
            "paddb mm0, mm1; fld1",
            &[0x0f, 0xfc, 0xc1, 0xd9, 0xe8, 0xf4],
        ),
        ("paddb mm0, qword ptr [rsi]", &[0x0f, 0xfc, 0x06, 0xf4]),
        ("paddb mm7, mm3", &[0x0f, 0xfc, 0xfb, 0xf4]),
        ("paddb xmm0, xmm1", &[0x66, 0x0f, 0xfc, 0xc1, 0xf4]),
        (
            "paddb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xfc, 0x06, 0xf4],
        ),
        ("paddb xmm9, xmm10", &[0x66, 0x45, 0x0f, 0xfc, 0xca, 0xf4]),
        ("paddd mm0, mm0", &[0x0f, 0xfe, 0xc0, 0xf4]),
        ("paddd mm0, mm1", &[0x0f, 0xfe, 0xc1, 0xf4]),
        ("paddd mm0, qword ptr [rsi]", &[0x0f, 0xfe, 0x06, 0xf4]),
        ("paddd xmm0, xmm1", &[0x66, 0x0f, 0xfe, 0xc1, 0xf4]),
        (
            "paddd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xfe, 0x06, 0xf4],
        ),
        ("paddq mm0, mm0", &[0x0f, 0xd4, 0xc0, 0xf4]),
        ("paddq mm0, mm1", &[0x0f, 0xd4, 0xc1, 0xf4]),
        ("paddq mm0, qword ptr [rsi]", &[0x0f, 0xd4, 0x06, 0xf4]),
        ("paddq xmm0, xmm1", &[0x66, 0x0f, 0xd4, 0xc1, 0xf4]),
        (
            "paddq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd4, 0x06, 0xf4],
        ),
        ("paddsb mm0, mm0", &[0x0f, 0xec, 0xc0, 0xf4]),
        ("paddsb mm0, mm1", &[0x0f, 0xec, 0xc1, 0xf4]),
        ("paddsb mm0, qword ptr [rsi]", &[0x0f, 0xec, 0x06, 0xf4]),
        ("paddsb xmm0, xmm1", &[0x66, 0x0f, 0xec, 0xc1, 0xf4]),
        (
            "paddsb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xec, 0x06, 0xf4],
        ),
        ("paddsw mm0, mm0", &[0x0f, 0xed, 0xc0, 0xf4]),
        ("paddsw mm0, mm1", &[0x0f, 0xed, 0xc1, 0xf4]),
        ("paddsw mm0, qword ptr [rsi]", &[0x0f, 0xed, 0x06, 0xf4]),
        ("paddsw xmm0, xmm1", &[0x66, 0x0f, 0xed, 0xc1, 0xf4]),
        (
            "paddsw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xed, 0x06, 0xf4],
        ),
        ("paddusb mm0, mm0", &[0x0f, 0xdc, 0xc0, 0xf4]),
        ("paddusb mm0, mm1", &[0x0f, 0xdc, 0xc1, 0xf4]),
        ("paddusb mm0, qword ptr [rsi]", &[0x0f, 0xdc, 0x06, 0xf4]),
        ("paddusb xmm0, xmm1", &[0x66, 0x0f, 0xdc, 0xc1, 0xf4]),
        (
            "paddusb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xdc, 0x06, 0xf4],
        ),
        ("paddusw mm0, mm0", &[0x0f, 0xdd, 0xc0, 0xf4]),
        ("paddusw mm0, mm1", &[0x0f, 0xdd, 0xc1, 0xf4]),
        ("paddusw mm0, qword ptr [rsi]", &[0x0f, 0xdd, 0x06, 0xf4]),
        ("paddusw xmm0, xmm1", &[0x66, 0x0f, 0xdd, 0xc1, 0xf4]),
        (
            "paddusw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xdd, 0x06, 0xf4],
        ),
        ("paddw mm0, mm0", &[0x0f, 0xfd, 0xc0, 0xf4]),
        ("paddw mm0, mm1", &[0x0f, 0xfd, 0xc1, 0xf4]),
        ("paddw mm0, qword ptr [rsi]", &[0x0f, 0xfd, 0x06, 0xf4]),
        ("paddw xmm0, xmm1", &[0x66, 0x0f, 0xfd, 0xc1, 0xf4]),
        (
            "paddw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xfd, 0x06, 0xf4],
        ),
        ("pand mm0, mm0", &[0x0f, 0xdb, 0xc0, 0xf4]),
        ("pand mm0, mm1", &[0x0f, 0xdb, 0xc1, 0xf4]),
        ("pand mm0, qword ptr [rsi]", &[0x0f, 0xdb, 0x06, 0xf4]),
        ("pand xmm0, xmm1", &[0x66, 0x0f, 0xdb, 0xc1, 0xf4]),
        (
            "pand xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xdb, 0x06, 0xf4],
        ),
        ("pandn mm0, mm0", &[0x0f, 0xdf, 0xc0, 0xf4]),
        ("pandn mm0, mm1", &[0x0f, 0xdf, 0xc1, 0xf4]),
        ("pandn mm0, qword ptr [rsi]", &[0x0f, 0xdf, 0x06, 0xf4]),
        ("pandn xmm0, xmm1", &[0x66, 0x0f, 0xdf, 0xc1, 0xf4]),
        (
            "pandn xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xdf, 0x06, 0xf4],
        ),
        ("pause", &[0xf3, 0x90, 0xf4]),
        ("pavgb mm0, mm0", &[0x0f, 0xe0, 0xc0, 0xf4]),
        ("pavgb mm0, mm1", &[0x0f, 0xe0, 0xc1, 0xf4]),
        ("pavgb mm0, qword ptr [rsi]", &[0x0f, 0xe0, 0x06, 0xf4]),
        ("pavgb xmm0, xmm1", &[0x66, 0x0f, 0xe0, 0xc1, 0xf4]),
        (
            "pavgb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe0, 0x06, 0xf4],
        ),
        ("pavgw mm0, mm0", &[0x0f, 0xe3, 0xc0, 0xf4]),
        ("pavgw mm0, mm1", &[0x0f, 0xe3, 0xc1, 0xf4]),
        ("pavgw mm0, qword ptr [rsi]", &[0x0f, 0xe3, 0x06, 0xf4]),
        ("pavgw xmm0, xmm1", &[0x66, 0x0f, 0xe3, 0xc1, 0xf4]),
        (
            "pavgw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe3, 0x06, 0xf4],
        ),
        ("pcmpeqb mm0, mm0", &[0x0f, 0x74, 0xc0, 0xf4]),
        ("pcmpeqb mm0, mm1", &[0x0f, 0x74, 0xc1, 0xf4]),
        ("pcmpeqb mm0, qword ptr [rsi]", &[0x0f, 0x74, 0x06, 0xf4]),
        ("pcmpeqb mm5, mm6", &[0x0f, 0x74, 0xee, 0xf4]),
        ("pcmpeqb xmm0, xmm1", &[0x66, 0x0f, 0x74, 0xc1, 0xf4]),
        (
            "pcmpeqb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x74, 0x06, 0xf4],
        ),
        ("pcmpeqd mm0, mm0", &[0x0f, 0x76, 0xc0, 0xf4]),
        ("pcmpeqd mm0, mm1", &[0x0f, 0x76, 0xc1, 0xf4]),
        ("pcmpeqd mm0, qword ptr [rsi]", &[0x0f, 0x76, 0x06, 0xf4]),
        ("pcmpeqd xmm0, xmm1", &[0x66, 0x0f, 0x76, 0xc1, 0xf4]),
        (
            "pcmpeqd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x76, 0x06, 0xf4],
        ),
        ("pcmpeqw mm0, mm0", &[0x0f, 0x75, 0xc0, 0xf4]),
        ("pcmpeqw mm0, mm1", &[0x0f, 0x75, 0xc1, 0xf4]),
        ("pcmpeqw mm0, qword ptr [rsi]", &[0x0f, 0x75, 0x06, 0xf4]),
        ("pcmpeqw xmm0, xmm1", &[0x66, 0x0f, 0x75, 0xc1, 0xf4]),
        (
            "pcmpeqw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x75, 0x06, 0xf4],
        ),
        ("pcmpgtb mm0, mm0", &[0x0f, 0x64, 0xc0, 0xf4]),
        ("pcmpgtb mm0, mm1", &[0x0f, 0x64, 0xc1, 0xf4]),
        ("pcmpgtb mm0, qword ptr [rsi]", &[0x0f, 0x64, 0x06, 0xf4]),
        ("pcmpgtb xmm0, xmm1", &[0x66, 0x0f, 0x64, 0xc1, 0xf4]),
        (
            "pcmpgtb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x64, 0x06, 0xf4],
        ),
        ("pcmpgtd mm0, mm0", &[0x0f, 0x66, 0xc0, 0xf4]),
        ("pcmpgtd mm0, mm1", &[0x0f, 0x66, 0xc1, 0xf4]),
        ("pcmpgtd mm0, qword ptr [rsi]", &[0x0f, 0x66, 0x06, 0xf4]),
        ("pcmpgtd xmm0, xmm1", &[0x66, 0x0f, 0x66, 0xc1, 0xf4]),
        (
            "pcmpgtd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x66, 0x06, 0xf4],
        ),
        ("pcmpgtw mm0, mm0", &[0x0f, 0x65, 0xc0, 0xf4]),
        ("pcmpgtw mm0, mm1", &[0x0f, 0x65, 0xc1, 0xf4]),
        ("pcmpgtw mm0, qword ptr [rsi]", &[0x0f, 0x65, 0x06, 0xf4]),
        ("pcmpgtw xmm0, xmm1", &[0x66, 0x0f, 0x65, 0xc1, 0xf4]),
        (
            "pcmpgtw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x65, 0x06, 0xf4],
        ),
        ("pextrw eax, mm0, 0", &[0x0f, 0xc5, 0xc0, 0x00, 0xf4]),
        ("pextrw eax, mm1, 0", &[0x0f, 0xc5, 0xc1, 0x00, 0xf4]),
        ("pextrw eax, mm1, 1", &[0x0f, 0xc5, 0xc1, 0x01, 0xf4]),
        ("pextrw eax, mm1, 2", &[0x0f, 0xc5, 0xc1, 0x02, 0xf4]),
        ("pextrw eax, mm1, 3", &[0x0f, 0xc5, 0xc1, 0x03, 0xf4]),
        ("pextrw eax, mm1, 4", &[0x0f, 0xc5, 0xc1, 0x04, 0xf4]),
        ("pextrw eax, mm1, 7", &[0x0f, 0xc5, 0xc1, 0x07, 0xf4]),
        ("pextrw eax, xmm1, 0", &[0x66, 0x0f, 0xc5, 0xc1, 0x00, 0xf4]),
        ("pextrw eax, xmm1, 1", &[0x66, 0x0f, 0xc5, 0xc1, 0x01, 0xf4]),
        (
            "pextrw eax, xmm1, 15",
            &[0x66, 0x0f, 0xc5, 0xc1, 0x0f, 0xf4],
        ),
        ("pextrw eax, xmm1, 2", &[0x66, 0x0f, 0xc5, 0xc1, 0x02, 0xf4]),
        ("pextrw eax, xmm1, 3", &[0x66, 0x0f, 0xc5, 0xc1, 0x03, 0xf4]),
        ("pextrw eax, xmm1, 4", &[0x66, 0x0f, 0xc5, 0xc1, 0x04, 0xf4]),
        ("pextrw eax, xmm1, 5", &[0x66, 0x0f, 0xc5, 0xc1, 0x05, 0xf4]),
        ("pextrw eax, xmm1, 6", &[0x66, 0x0f, 0xc5, 0xc1, 0x06, 0xf4]),
        ("pextrw eax, xmm1, 7", &[0x66, 0x0f, 0xc5, 0xc1, 0x07, 0xf4]),
        ("pextrw eax, xmm1, 8", &[0x66, 0x0f, 0xc5, 0xc1, 0x08, 0xf4]),
        ("pextrw r9d, mm1, 1", &[0x44, 0x0f, 0xc5, 0xc9, 0x01, 0xf4]),
        (
            "pextrw r9d, xmm10, 1",
            &[0x66, 0x45, 0x0f, 0xc5, 0xca, 0x01, 0xf4],
        ),
        ("pinsrw mm0, eax, 0", &[0x0f, 0xc4, 0xc0, 0x00, 0xf4]),
        ("pinsrw mm0, eax, 1", &[0x0f, 0xc4, 0xc0, 0x01, 0xf4]),
        ("pinsrw mm0, eax, 2", &[0x0f, 0xc4, 0xc0, 0x02, 0xf4]),
        ("pinsrw mm0, eax, 3", &[0x0f, 0xc4, 0xc0, 0x03, 0xf4]),
        ("pinsrw mm0, eax, 4", &[0x0f, 0xc4, 0xc0, 0x04, 0xf4]),
        ("pinsrw mm0, eax, 7", &[0x0f, 0xc4, 0xc0, 0x07, 0xf4]),
        (
            "pinsrw mm0, word ptr [rsi], 2",
            &[0x0f, 0xc4, 0x06, 0x02, 0xf4],
        ),
        ("pinsrw mm3, r8d, 3", &[0x41, 0x0f, 0xc4, 0xd8, 0x03, 0xf4]),
        ("pinsrw xmm0, eax, 0", &[0x66, 0x0f, 0xc4, 0xc0, 0x00, 0xf4]),
        ("pinsrw xmm0, eax, 1", &[0x66, 0x0f, 0xc4, 0xc0, 0x01, 0xf4]),
        (
            "pinsrw xmm0, eax, 15",
            &[0x66, 0x0f, 0xc4, 0xc0, 0x0f, 0xf4],
        ),
        ("pinsrw xmm0, eax, 2", &[0x66, 0x0f, 0xc4, 0xc0, 0x02, 0xf4]),
        ("pinsrw xmm0, eax, 3", &[0x66, 0x0f, 0xc4, 0xc0, 0x03, 0xf4]),
        ("pinsrw xmm0, eax, 4", &[0x66, 0x0f, 0xc4, 0xc0, 0x04, 0xf4]),
        ("pinsrw xmm0, eax, 5", &[0x66, 0x0f, 0xc4, 0xc0, 0x05, 0xf4]),
        ("pinsrw xmm0, eax, 6", &[0x66, 0x0f, 0xc4, 0xc0, 0x06, 0xf4]),
        ("pinsrw xmm0, eax, 7", &[0x66, 0x0f, 0xc4, 0xc0, 0x07, 0xf4]),
        ("pinsrw xmm0, eax, 8", &[0x66, 0x0f, 0xc4, 0xc0, 0x08, 0xf4]),
        (
            "pinsrw xmm0, word ptr [rsi], 2",
            &[0x66, 0x0f, 0xc4, 0x06, 0x02, 0xf4],
        ),
        (
            "pinsrw xmm9, r8d, 5",
            &[0x66, 0x45, 0x0f, 0xc4, 0xc8, 0x05, 0xf4],
        ),
        ("pmaddwd mm0, mm0", &[0x0f, 0xf5, 0xc0, 0xf4]),
        ("pmaddwd mm0, mm1", &[0x0f, 0xf5, 0xc1, 0xf4]),
        ("pmaddwd mm0, qword ptr [rsi]", &[0x0f, 0xf5, 0x06, 0xf4]),
        ("pmaddwd xmm0, xmm1", &[0x66, 0x0f, 0xf5, 0xc1, 0xf4]),
        (
            "pmaddwd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf5, 0x06, 0xf4],
        ),
        ("pmaxsw mm0, mm0", &[0x0f, 0xee, 0xc0, 0xf4]),
        ("pmaxsw mm0, mm1", &[0x0f, 0xee, 0xc1, 0xf4]),
        ("pmaxsw mm0, qword ptr [rsi]", &[0x0f, 0xee, 0x06, 0xf4]),
        ("pmaxsw xmm0, xmm1", &[0x66, 0x0f, 0xee, 0xc1, 0xf4]),
        (
            "pmaxsw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xee, 0x06, 0xf4],
        ),
        ("pmaxub mm0, mm0", &[0x0f, 0xde, 0xc0, 0xf4]),
        ("pmaxub mm0, mm1", &[0x0f, 0xde, 0xc1, 0xf4]),
        ("pmaxub mm0, qword ptr [rsi]", &[0x0f, 0xde, 0x06, 0xf4]),
        ("pmaxub xmm0, xmm1", &[0x66, 0x0f, 0xde, 0xc1, 0xf4]),
        (
            "pmaxub xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xde, 0x06, 0xf4],
        ),
        ("pminsw mm0, mm0", &[0x0f, 0xea, 0xc0, 0xf4]),
        ("pminsw mm0, mm1", &[0x0f, 0xea, 0xc1, 0xf4]),
        ("pminsw mm0, qword ptr [rsi]", &[0x0f, 0xea, 0x06, 0xf4]),
        ("pminsw xmm0, xmm1", &[0x66, 0x0f, 0xea, 0xc1, 0xf4]),
        (
            "pminsw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xea, 0x06, 0xf4],
        ),
        ("pminub mm0, mm0", &[0x0f, 0xda, 0xc0, 0xf4]),
        ("pminub mm0, mm1", &[0x0f, 0xda, 0xc1, 0xf4]),
        ("pminub mm0, qword ptr [rsi]", &[0x0f, 0xda, 0x06, 0xf4]),
        ("pminub xmm0, xmm1", &[0x66, 0x0f, 0xda, 0xc1, 0xf4]),
        (
            "pminub xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xda, 0x06, 0xf4],
        ),
        ("pmovmskb eax, mm1", &[0x0f, 0xd7, 0xc1, 0xf4]),
        ("pmovmskb eax, xmm1", &[0x66, 0x0f, 0xd7, 0xc1, 0xf4]),
        ("pmovmskb r9d, mm1", &[0x44, 0x0f, 0xd7, 0xc9, 0xf4]),
        ("pmovmskb rax, xmm1", &[0x66, 0x0f, 0xd7, 0xc1, 0xf4]),
        ("pmulhuw mm0, mm0", &[0x0f, 0xe4, 0xc0, 0xf4]),
        ("pmulhuw mm0, mm1", &[0x0f, 0xe4, 0xc1, 0xf4]),
        ("pmulhuw mm0, qword ptr [rsi]", &[0x0f, 0xe4, 0x06, 0xf4]),
        ("pmulhuw xmm0, xmm1", &[0x66, 0x0f, 0xe4, 0xc1, 0xf4]),
        (
            "pmulhuw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe4, 0x06, 0xf4],
        ),
        ("pmulhw mm0, mm0", &[0x0f, 0xe5, 0xc0, 0xf4]),
        ("pmulhw mm0, mm1", &[0x0f, 0xe5, 0xc1, 0xf4]),
        ("pmulhw mm0, qword ptr [rsi]", &[0x0f, 0xe5, 0x06, 0xf4]),
        ("pmulhw xmm0, xmm1", &[0x66, 0x0f, 0xe5, 0xc1, 0xf4]),
        (
            "pmulhw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe5, 0x06, 0xf4],
        ),
        ("pmullw mm0, mm0", &[0x0f, 0xd5, 0xc0, 0xf4]),
        ("pmullw mm0, mm1", &[0x0f, 0xd5, 0xc1, 0xf4]),
        ("pmullw mm0, qword ptr [rsi]", &[0x0f, 0xd5, 0x06, 0xf4]),
        ("pmullw xmm0, xmm1", &[0x66, 0x0f, 0xd5, 0xc1, 0xf4]),
        (
            "pmullw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd5, 0x06, 0xf4],
        ),
        ("pmuludq mm0, mm0", &[0x0f, 0xf4, 0xc0, 0xf4]),
        ("pmuludq mm0, mm1", &[0x0f, 0xf4, 0xc1, 0xf4]),
        ("pmuludq mm0, qword ptr [rsi]", &[0x0f, 0xf4, 0x06, 0xf4]),
        ("pmuludq xmm0, xmm1", &[0x66, 0x0f, 0xf4, 0xc1, 0xf4]),
        (
            "pmuludq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf4, 0x06, 0xf4],
        ),
        ("por mm0, mm0", &[0x0f, 0xeb, 0xc0, 0xf4]),
        ("por mm0, mm1", &[0x0f, 0xeb, 0xc1, 0xf4]),
        ("por mm0, qword ptr [rsi]", &[0x0f, 0xeb, 0x06, 0xf4]),
        ("por xmm0, xmm1", &[0x66, 0x0f, 0xeb, 0xc1, 0xf4]),
        (
            "por xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xeb, 0x06, 0xf4],
        ),
        ("prefetchnta byte ptr [rsi]", &[0x0f, 0x18, 0x06, 0xf4]),
        ("prefetcht0 byte ptr [rsi]", &[0x0f, 0x18, 0x0e, 0xf4]),
        ("prefetcht1 byte ptr [rsi]", &[0x0f, 0x18, 0x16, 0xf4]),
        ("prefetcht2 byte ptr [rsi]", &[0x0f, 0x18, 0x1e, 0xf4]),
        ("prefetchw byte ptr [rsi]", &[0x0f, 0x0d, 0x0e, 0xf4]),
        ("psadbw mm0, mm0", &[0x0f, 0xf6, 0xc0, 0xf4]),
        ("psadbw mm0, mm1", &[0x0f, 0xf6, 0xc1, 0xf4]),
        ("psadbw mm0, qword ptr [rsi]", &[0x0f, 0xf6, 0x06, 0xf4]),
        ("psadbw xmm0, xmm1", &[0x66, 0x0f, 0xf6, 0xc1, 0xf4]),
        (
            "psadbw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf6, 0x06, 0xf4],
        ),
        (
            "pshufd xmm0, xmm0, 0x1b",
            &[0x66, 0x0f, 0x70, 0xc0, 0x1b, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0",
            &[0x66, 0x0f, 0x70, 0xc1, 0x00, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0x1b",
            &[0x66, 0x0f, 0x70, 0xc1, 0x1b, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0x39",
            &[0x66, 0x0f, 0x70, 0xc1, 0x39, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0x4e",
            &[0x66, 0x0f, 0x70, 0xc1, 0x4e, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0x93",
            &[0x66, 0x0f, 0x70, 0xc1, 0x93, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0xe4",
            &[0x66, 0x0f, 0x70, 0xc1, 0xe4, 0xf4],
        ),
        (
            "pshufd xmm0, xmm1, 0xff",
            &[0x66, 0x0f, 0x70, 0xc1, 0xff, 0xf4],
        ),
        (
            "pshufd xmm0, xmmword ptr [rsi], 0x1b",
            &[0x66, 0x0f, 0x70, 0x06, 0x1b, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x00, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0x1b",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x1b, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0x39",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x39, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0x4e",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x4e, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0x93",
            &[0xf3, 0x0f, 0x70, 0xc1, 0x93, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0xe4",
            &[0xf3, 0x0f, 0x70, 0xc1, 0xe4, 0xf4],
        ),
        (
            "pshufhw xmm0, xmm1, 0xff",
            &[0xf3, 0x0f, 0x70, 0xc1, 0xff, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0",
            &[0xf2, 0x0f, 0x70, 0xc1, 0x00, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0x1b",
            &[0xf2, 0x0f, 0x70, 0xc1, 0x1b, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0x39",
            &[0xf2, 0x0f, 0x70, 0xc1, 0x39, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0x4e",
            &[0xf2, 0x0f, 0x70, 0xc1, 0x4e, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0x93",
            &[0xf2, 0x0f, 0x70, 0xc1, 0x93, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0xe4",
            &[0xf2, 0x0f, 0x70, 0xc1, 0xe4, 0xf4],
        ),
        (
            "pshuflw xmm0, xmm1, 0xff",
            &[0xf2, 0x0f, 0x70, 0xc1, 0xff, 0xf4],
        ),
        ("pshufw mm0, mm1, 0", &[0x0f, 0x70, 0xc1, 0x00, 0xf4]),
        ("pshufw mm0, mm1, 0x1b", &[0x0f, 0x70, 0xc1, 0x1b, 0xf4]),
        ("pshufw mm0, mm1, 0x39", &[0x0f, 0x70, 0xc1, 0x39, 0xf4]),
        ("pshufw mm0, mm1, 0x4e", &[0x0f, 0x70, 0xc1, 0x4e, 0xf4]),
        ("pshufw mm0, mm1, 0x93", &[0x0f, 0x70, 0xc1, 0x93, 0xf4]),
        ("pshufw mm0, mm1, 0xe4", &[0x0f, 0x70, 0xc1, 0xe4, 0xf4]),
        ("pshufw mm0, mm1, 0xff", &[0x0f, 0x70, 0xc1, 0xff, 0xf4]),
        (
            "pshufw mm0, qword ptr [rsi], 0x1b",
            &[0x0f, 0x70, 0x06, 0x1b, 0xf4],
        ),
        ("pslld mm0, 0", &[0x0f, 0x72, 0xf0, 0x00, 0xf4]),
        ("pslld mm0, 1", &[0x0f, 0x72, 0xf0, 0x01, 0xf4]),
        ("pslld mm0, 15", &[0x0f, 0x72, 0xf0, 0x0f, 0xf4]),
        ("pslld mm0, 16", &[0x0f, 0x72, 0xf0, 0x10, 0xf4]),
        ("pslld mm0, 255", &[0x0f, 0x72, 0xf0, 0xff, 0xf4]),
        ("pslld mm0, 3", &[0x0f, 0x72, 0xf0, 0x03, 0xf4]),
        ("pslld mm0, 31", &[0x0f, 0x72, 0xf0, 0x1f, 0xf4]),
        ("pslld mm0, 32", &[0x0f, 0x72, 0xf0, 0x20, 0xf4]),
        ("pslld mm0, 33", &[0x0f, 0x72, 0xf0, 0x21, 0xf4]),
        ("pslld mm0, 63", &[0x0f, 0x72, 0xf0, 0x3f, 0xf4]),
        ("pslld mm0, 64", &[0x0f, 0x72, 0xf0, 0x40, 0xf4]),
        ("pslld mm0, 65", &[0x0f, 0x72, 0xf0, 0x41, 0xf4]),
        ("pslld mm0, 7", &[0x0f, 0x72, 0xf0, 0x07, 0xf4]),
        ("pslld mm0, 8", &[0x0f, 0x72, 0xf0, 0x08, 0xf4]),
        ("pslld mm0, mm0", &[0x0f, 0xf2, 0xc0, 0xf4]),
        ("pslld mm0, mm1", &[0x0f, 0xf2, 0xc1, 0xf4]),
        ("pslld mm0, qword ptr [rsi]", &[0x0f, 0xf2, 0x06, 0xf4]),
        ("pslld xmm0, 0", &[0x66, 0x0f, 0x72, 0xf0, 0x00, 0xf4]),
        ("pslld xmm0, 1", &[0x66, 0x0f, 0x72, 0xf0, 0x01, 0xf4]),
        ("pslld xmm0, 15", &[0x66, 0x0f, 0x72, 0xf0, 0x0f, 0xf4]),
        ("pslld xmm0, 16", &[0x66, 0x0f, 0x72, 0xf0, 0x10, 0xf4]),
        ("pslld xmm0, 255", &[0x66, 0x0f, 0x72, 0xf0, 0xff, 0xf4]),
        ("pslld xmm0, 3", &[0x66, 0x0f, 0x72, 0xf0, 0x03, 0xf4]),
        ("pslld xmm0, 31", &[0x66, 0x0f, 0x72, 0xf0, 0x1f, 0xf4]),
        ("pslld xmm0, 32", &[0x66, 0x0f, 0x72, 0xf0, 0x20, 0xf4]),
        ("pslld xmm0, 33", &[0x66, 0x0f, 0x72, 0xf0, 0x21, 0xf4]),
        ("pslld xmm0, 63", &[0x66, 0x0f, 0x72, 0xf0, 0x3f, 0xf4]),
        ("pslld xmm0, 64", &[0x66, 0x0f, 0x72, 0xf0, 0x40, 0xf4]),
        ("pslld xmm0, 65", &[0x66, 0x0f, 0x72, 0xf0, 0x41, 0xf4]),
        ("pslld xmm0, 7", &[0x66, 0x0f, 0x72, 0xf0, 0x07, 0xf4]),
        ("pslld xmm0, 8", &[0x66, 0x0f, 0x72, 0xf0, 0x08, 0xf4]),
        ("pslld xmm0, xmm1", &[0x66, 0x0f, 0xf2, 0xc1, 0xf4]),
        (
            "pslld xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf2, 0x06, 0xf4],
        ),
        ("pslldq xmm0, 0", &[0x66, 0x0f, 0x73, 0xf8, 0x00, 0xf4]),
        ("pslldq xmm0, 1", &[0x66, 0x0f, 0x73, 0xf8, 0x01, 0xf4]),
        ("pslldq xmm0, 15", &[0x66, 0x0f, 0x73, 0xf8, 0x0f, 0xf4]),
        ("pslldq xmm0, 16", &[0x66, 0x0f, 0x73, 0xf8, 0x10, 0xf4]),
        ("pslldq xmm0, 17", &[0x66, 0x0f, 0x73, 0xf8, 0x11, 0xf4]),
        ("pslldq xmm0, 255", &[0x66, 0x0f, 0x73, 0xf8, 0xff, 0xf4]),
        ("pslldq xmm0, 4", &[0x66, 0x0f, 0x73, 0xf8, 0x04, 0xf4]),
        ("pslldq xmm0, 8", &[0x66, 0x0f, 0x73, 0xf8, 0x08, 0xf4]),
        (
            "pslldq xmm9, 4",
            &[0x66, 0x41, 0x0f, 0x73, 0xf9, 0x04, 0xf4],
        ),
        ("psllq mm0, 0", &[0x0f, 0x73, 0xf0, 0x00, 0xf4]),
        ("psllq mm0, 1", &[0x0f, 0x73, 0xf0, 0x01, 0xf4]),
        ("psllq mm0, 15", &[0x0f, 0x73, 0xf0, 0x0f, 0xf4]),
        ("psllq mm0, 16", &[0x0f, 0x73, 0xf0, 0x10, 0xf4]),
        ("psllq mm0, 255", &[0x0f, 0x73, 0xf0, 0xff, 0xf4]),
        ("psllq mm0, 3", &[0x0f, 0x73, 0xf0, 0x03, 0xf4]),
        ("psllq mm0, 31", &[0x0f, 0x73, 0xf0, 0x1f, 0xf4]),
        ("psllq mm0, 32", &[0x0f, 0x73, 0xf0, 0x20, 0xf4]),
        ("psllq mm0, 33", &[0x0f, 0x73, 0xf0, 0x21, 0xf4]),
        ("psllq mm0, 63", &[0x0f, 0x73, 0xf0, 0x3f, 0xf4]),
        ("psllq mm0, 64", &[0x0f, 0x73, 0xf0, 0x40, 0xf4]),
        ("psllq mm0, 65", &[0x0f, 0x73, 0xf0, 0x41, 0xf4]),
        ("psllq mm0, 7", &[0x0f, 0x73, 0xf0, 0x07, 0xf4]),
        ("psllq mm0, 8", &[0x0f, 0x73, 0xf0, 0x08, 0xf4]),
        ("psllq mm0, mm0", &[0x0f, 0xf3, 0xc0, 0xf4]),
        ("psllq mm0, mm1", &[0x0f, 0xf3, 0xc1, 0xf4]),
        ("psllq mm0, qword ptr [rsi]", &[0x0f, 0xf3, 0x06, 0xf4]),
        ("psllq xmm0, 0", &[0x66, 0x0f, 0x73, 0xf0, 0x00, 0xf4]),
        ("psllq xmm0, 1", &[0x66, 0x0f, 0x73, 0xf0, 0x01, 0xf4]),
        ("psllq xmm0, 15", &[0x66, 0x0f, 0x73, 0xf0, 0x0f, 0xf4]),
        ("psllq xmm0, 16", &[0x66, 0x0f, 0x73, 0xf0, 0x10, 0xf4]),
        ("psllq xmm0, 255", &[0x66, 0x0f, 0x73, 0xf0, 0xff, 0xf4]),
        ("psllq xmm0, 3", &[0x66, 0x0f, 0x73, 0xf0, 0x03, 0xf4]),
        ("psllq xmm0, 31", &[0x66, 0x0f, 0x73, 0xf0, 0x1f, 0xf4]),
        ("psllq xmm0, 32", &[0x66, 0x0f, 0x73, 0xf0, 0x20, 0xf4]),
        ("psllq xmm0, 33", &[0x66, 0x0f, 0x73, 0xf0, 0x21, 0xf4]),
        ("psllq xmm0, 63", &[0x66, 0x0f, 0x73, 0xf0, 0x3f, 0xf4]),
        ("psllq xmm0, 64", &[0x66, 0x0f, 0x73, 0xf0, 0x40, 0xf4]),
        ("psllq xmm0, 65", &[0x66, 0x0f, 0x73, 0xf0, 0x41, 0xf4]),
        ("psllq xmm0, 7", &[0x66, 0x0f, 0x73, 0xf0, 0x07, 0xf4]),
        ("psllq xmm0, 8", &[0x66, 0x0f, 0x73, 0xf0, 0x08, 0xf4]),
        ("psllq xmm0, xmm1", &[0x66, 0x0f, 0xf3, 0xc1, 0xf4]),
        (
            "psllq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf3, 0x06, 0xf4],
        ),
        ("psllw mm0, 0", &[0x0f, 0x71, 0xf0, 0x00, 0xf4]),
        ("psllw mm0, 1", &[0x0f, 0x71, 0xf0, 0x01, 0xf4]),
        ("psllw mm0, 15", &[0x0f, 0x71, 0xf0, 0x0f, 0xf4]),
        ("psllw mm0, 16", &[0x0f, 0x71, 0xf0, 0x10, 0xf4]),
        ("psllw mm0, 255", &[0x0f, 0x71, 0xf0, 0xff, 0xf4]),
        ("psllw mm0, 3", &[0x0f, 0x71, 0xf0, 0x03, 0xf4]),
        ("psllw mm0, 31", &[0x0f, 0x71, 0xf0, 0x1f, 0xf4]),
        ("psllw mm0, 32", &[0x0f, 0x71, 0xf0, 0x20, 0xf4]),
        ("psllw mm0, 33", &[0x0f, 0x71, 0xf0, 0x21, 0xf4]),
        ("psllw mm0, 63", &[0x0f, 0x71, 0xf0, 0x3f, 0xf4]),
        ("psllw mm0, 64", &[0x0f, 0x71, 0xf0, 0x40, 0xf4]),
        ("psllw mm0, 65", &[0x0f, 0x71, 0xf0, 0x41, 0xf4]),
        ("psllw mm0, 7", &[0x0f, 0x71, 0xf0, 0x07, 0xf4]),
        ("psllw mm0, 8", &[0x0f, 0x71, 0xf0, 0x08, 0xf4]),
        ("psllw mm0, mm0", &[0x0f, 0xf1, 0xc0, 0xf4]),
        ("psllw mm0, mm1", &[0x0f, 0xf1, 0xc1, 0xf4]),
        ("psllw mm0, qword ptr [rsi]", &[0x0f, 0xf1, 0x06, 0xf4]),
        ("psllw mm7, 3", &[0x0f, 0x71, 0xf7, 0x03, 0xf4]),
        ("psllw xmm0, 0", &[0x66, 0x0f, 0x71, 0xf0, 0x00, 0xf4]),
        ("psllw xmm0, 1", &[0x66, 0x0f, 0x71, 0xf0, 0x01, 0xf4]),
        ("psllw xmm0, 15", &[0x66, 0x0f, 0x71, 0xf0, 0x0f, 0xf4]),
        ("psllw xmm0, 16", &[0x66, 0x0f, 0x71, 0xf0, 0x10, 0xf4]),
        ("psllw xmm0, 255", &[0x66, 0x0f, 0x71, 0xf0, 0xff, 0xf4]),
        ("psllw xmm0, 3", &[0x66, 0x0f, 0x71, 0xf0, 0x03, 0xf4]),
        ("psllw xmm0, 31", &[0x66, 0x0f, 0x71, 0xf0, 0x1f, 0xf4]),
        ("psllw xmm0, 32", &[0x66, 0x0f, 0x71, 0xf0, 0x20, 0xf4]),
        ("psllw xmm0, 33", &[0x66, 0x0f, 0x71, 0xf0, 0x21, 0xf4]),
        ("psllw xmm0, 63", &[0x66, 0x0f, 0x71, 0xf0, 0x3f, 0xf4]),
        ("psllw xmm0, 64", &[0x66, 0x0f, 0x71, 0xf0, 0x40, 0xf4]),
        ("psllw xmm0, 65", &[0x66, 0x0f, 0x71, 0xf0, 0x41, 0xf4]),
        ("psllw xmm0, 7", &[0x66, 0x0f, 0x71, 0xf0, 0x07, 0xf4]),
        ("psllw xmm0, 8", &[0x66, 0x0f, 0x71, 0xf0, 0x08, 0xf4]),
        ("psllw xmm0, xmm1", &[0x66, 0x0f, 0xf1, 0xc1, 0xf4]),
        (
            "psllw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf1, 0x06, 0xf4],
        ),
        ("psllw xmm9, 3", &[0x66, 0x41, 0x0f, 0x71, 0xf1, 0x03, 0xf4]),
        ("psrad mm0, 0", &[0x0f, 0x72, 0xe0, 0x00, 0xf4]),
        ("psrad mm0, 1", &[0x0f, 0x72, 0xe0, 0x01, 0xf4]),
        ("psrad mm0, 15", &[0x0f, 0x72, 0xe0, 0x0f, 0xf4]),
        ("psrad mm0, 16", &[0x0f, 0x72, 0xe0, 0x10, 0xf4]),
        ("psrad mm0, 255", &[0x0f, 0x72, 0xe0, 0xff, 0xf4]),
        ("psrad mm0, 3", &[0x0f, 0x72, 0xe0, 0x03, 0xf4]),
        ("psrad mm0, 31", &[0x0f, 0x72, 0xe0, 0x1f, 0xf4]),
        ("psrad mm0, 32", &[0x0f, 0x72, 0xe0, 0x20, 0xf4]),
        ("psrad mm0, 33", &[0x0f, 0x72, 0xe0, 0x21, 0xf4]),
        ("psrad mm0, 63", &[0x0f, 0x72, 0xe0, 0x3f, 0xf4]),
        ("psrad mm0, 64", &[0x0f, 0x72, 0xe0, 0x40, 0xf4]),
        ("psrad mm0, 65", &[0x0f, 0x72, 0xe0, 0x41, 0xf4]),
        ("psrad mm0, 7", &[0x0f, 0x72, 0xe0, 0x07, 0xf4]),
        ("psrad mm0, 8", &[0x0f, 0x72, 0xe0, 0x08, 0xf4]),
        ("psrad mm0, mm0", &[0x0f, 0xe2, 0xc0, 0xf4]),
        ("psrad mm0, mm1", &[0x0f, 0xe2, 0xc1, 0xf4]),
        ("psrad mm0, qword ptr [rsi]", &[0x0f, 0xe2, 0x06, 0xf4]),
        ("psrad xmm0, 0", &[0x66, 0x0f, 0x72, 0xe0, 0x00, 0xf4]),
        ("psrad xmm0, 1", &[0x66, 0x0f, 0x72, 0xe0, 0x01, 0xf4]),
        ("psrad xmm0, 15", &[0x66, 0x0f, 0x72, 0xe0, 0x0f, 0xf4]),
        ("psrad xmm0, 16", &[0x66, 0x0f, 0x72, 0xe0, 0x10, 0xf4]),
        ("psrad xmm0, 255", &[0x66, 0x0f, 0x72, 0xe0, 0xff, 0xf4]),
        ("psrad xmm0, 3", &[0x66, 0x0f, 0x72, 0xe0, 0x03, 0xf4]),
        ("psrad xmm0, 31", &[0x66, 0x0f, 0x72, 0xe0, 0x1f, 0xf4]),
        ("psrad xmm0, 32", &[0x66, 0x0f, 0x72, 0xe0, 0x20, 0xf4]),
        ("psrad xmm0, 33", &[0x66, 0x0f, 0x72, 0xe0, 0x21, 0xf4]),
        ("psrad xmm0, 63", &[0x66, 0x0f, 0x72, 0xe0, 0x3f, 0xf4]),
        ("psrad xmm0, 64", &[0x66, 0x0f, 0x72, 0xe0, 0x40, 0xf4]),
        ("psrad xmm0, 65", &[0x66, 0x0f, 0x72, 0xe0, 0x41, 0xf4]),
        ("psrad xmm0, 7", &[0x66, 0x0f, 0x72, 0xe0, 0x07, 0xf4]),
        ("psrad xmm0, 8", &[0x66, 0x0f, 0x72, 0xe0, 0x08, 0xf4]),
        ("psrad xmm0, xmm1", &[0x66, 0x0f, 0xe2, 0xc1, 0xf4]),
        (
            "psrad xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe2, 0x06, 0xf4],
        ),
        ("psraw mm0, 0", &[0x0f, 0x71, 0xe0, 0x00, 0xf4]),
        ("psraw mm0, 1", &[0x0f, 0x71, 0xe0, 0x01, 0xf4]),
        ("psraw mm0, 15", &[0x0f, 0x71, 0xe0, 0x0f, 0xf4]),
        ("psraw mm0, 16", &[0x0f, 0x71, 0xe0, 0x10, 0xf4]),
        ("psraw mm0, 255", &[0x0f, 0x71, 0xe0, 0xff, 0xf4]),
        ("psraw mm0, 3", &[0x0f, 0x71, 0xe0, 0x03, 0xf4]),
        ("psraw mm0, 31", &[0x0f, 0x71, 0xe0, 0x1f, 0xf4]),
        ("psraw mm0, 32", &[0x0f, 0x71, 0xe0, 0x20, 0xf4]),
        ("psraw mm0, 33", &[0x0f, 0x71, 0xe0, 0x21, 0xf4]),
        ("psraw mm0, 63", &[0x0f, 0x71, 0xe0, 0x3f, 0xf4]),
        ("psraw mm0, 64", &[0x0f, 0x71, 0xe0, 0x40, 0xf4]),
        ("psraw mm0, 65", &[0x0f, 0x71, 0xe0, 0x41, 0xf4]),
        ("psraw mm0, 7", &[0x0f, 0x71, 0xe0, 0x07, 0xf4]),
        ("psraw mm0, 8", &[0x0f, 0x71, 0xe0, 0x08, 0xf4]),
        ("psraw mm0, mm0", &[0x0f, 0xe1, 0xc0, 0xf4]),
        ("psraw mm0, mm1", &[0x0f, 0xe1, 0xc1, 0xf4]),
        ("psraw mm0, qword ptr [rsi]", &[0x0f, 0xe1, 0x06, 0xf4]),
        ("psraw xmm0, 0", &[0x66, 0x0f, 0x71, 0xe0, 0x00, 0xf4]),
        ("psraw xmm0, 1", &[0x66, 0x0f, 0x71, 0xe0, 0x01, 0xf4]),
        ("psraw xmm0, 15", &[0x66, 0x0f, 0x71, 0xe0, 0x0f, 0xf4]),
        ("psraw xmm0, 16", &[0x66, 0x0f, 0x71, 0xe0, 0x10, 0xf4]),
        ("psraw xmm0, 255", &[0x66, 0x0f, 0x71, 0xe0, 0xff, 0xf4]),
        ("psraw xmm0, 3", &[0x66, 0x0f, 0x71, 0xe0, 0x03, 0xf4]),
        ("psraw xmm0, 31", &[0x66, 0x0f, 0x71, 0xe0, 0x1f, 0xf4]),
        ("psraw xmm0, 32", &[0x66, 0x0f, 0x71, 0xe0, 0x20, 0xf4]),
        ("psraw xmm0, 33", &[0x66, 0x0f, 0x71, 0xe0, 0x21, 0xf4]),
        ("psraw xmm0, 63", &[0x66, 0x0f, 0x71, 0xe0, 0x3f, 0xf4]),
        ("psraw xmm0, 64", &[0x66, 0x0f, 0x71, 0xe0, 0x40, 0xf4]),
        ("psraw xmm0, 65", &[0x66, 0x0f, 0x71, 0xe0, 0x41, 0xf4]),
        ("psraw xmm0, 7", &[0x66, 0x0f, 0x71, 0xe0, 0x07, 0xf4]),
        ("psraw xmm0, 8", &[0x66, 0x0f, 0x71, 0xe0, 0x08, 0xf4]),
        ("psraw xmm0, xmm1", &[0x66, 0x0f, 0xe1, 0xc1, 0xf4]),
        (
            "psraw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe1, 0x06, 0xf4],
        ),
        ("psrld mm0, 0", &[0x0f, 0x72, 0xd0, 0x00, 0xf4]),
        ("psrld mm0, 1", &[0x0f, 0x72, 0xd0, 0x01, 0xf4]),
        ("psrld mm0, 15", &[0x0f, 0x72, 0xd0, 0x0f, 0xf4]),
        ("psrld mm0, 16", &[0x0f, 0x72, 0xd0, 0x10, 0xf4]),
        ("psrld mm0, 255", &[0x0f, 0x72, 0xd0, 0xff, 0xf4]),
        ("psrld mm0, 3", &[0x0f, 0x72, 0xd0, 0x03, 0xf4]),
        ("psrld mm0, 31", &[0x0f, 0x72, 0xd0, 0x1f, 0xf4]),
        ("psrld mm0, 32", &[0x0f, 0x72, 0xd0, 0x20, 0xf4]),
        ("psrld mm0, 33", &[0x0f, 0x72, 0xd0, 0x21, 0xf4]),
        ("psrld mm0, 63", &[0x0f, 0x72, 0xd0, 0x3f, 0xf4]),
        ("psrld mm0, 64", &[0x0f, 0x72, 0xd0, 0x40, 0xf4]),
        ("psrld mm0, 65", &[0x0f, 0x72, 0xd0, 0x41, 0xf4]),
        ("psrld mm0, 7", &[0x0f, 0x72, 0xd0, 0x07, 0xf4]),
        ("psrld mm0, 8", &[0x0f, 0x72, 0xd0, 0x08, 0xf4]),
        ("psrld mm0, mm0", &[0x0f, 0xd2, 0xc0, 0xf4]),
        ("psrld mm0, mm1", &[0x0f, 0xd2, 0xc1, 0xf4]),
        ("psrld mm0, qword ptr [rsi]", &[0x0f, 0xd2, 0x06, 0xf4]),
        ("psrld xmm0, 0", &[0x66, 0x0f, 0x72, 0xd0, 0x00, 0xf4]),
        ("psrld xmm0, 1", &[0x66, 0x0f, 0x72, 0xd0, 0x01, 0xf4]),
        ("psrld xmm0, 15", &[0x66, 0x0f, 0x72, 0xd0, 0x0f, 0xf4]),
        ("psrld xmm0, 16", &[0x66, 0x0f, 0x72, 0xd0, 0x10, 0xf4]),
        ("psrld xmm0, 255", &[0x66, 0x0f, 0x72, 0xd0, 0xff, 0xf4]),
        ("psrld xmm0, 3", &[0x66, 0x0f, 0x72, 0xd0, 0x03, 0xf4]),
        ("psrld xmm0, 31", &[0x66, 0x0f, 0x72, 0xd0, 0x1f, 0xf4]),
        ("psrld xmm0, 32", &[0x66, 0x0f, 0x72, 0xd0, 0x20, 0xf4]),
        ("psrld xmm0, 33", &[0x66, 0x0f, 0x72, 0xd0, 0x21, 0xf4]),
        ("psrld xmm0, 63", &[0x66, 0x0f, 0x72, 0xd0, 0x3f, 0xf4]),
        ("psrld xmm0, 64", &[0x66, 0x0f, 0x72, 0xd0, 0x40, 0xf4]),
        ("psrld xmm0, 65", &[0x66, 0x0f, 0x72, 0xd0, 0x41, 0xf4]),
        ("psrld xmm0, 7", &[0x66, 0x0f, 0x72, 0xd0, 0x07, 0xf4]),
        ("psrld xmm0, 8", &[0x66, 0x0f, 0x72, 0xd0, 0x08, 0xf4]),
        ("psrld xmm0, xmm1", &[0x66, 0x0f, 0xd2, 0xc1, 0xf4]),
        (
            "psrld xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd2, 0x06, 0xf4],
        ),
        ("psrldq xmm0, 0", &[0x66, 0x0f, 0x73, 0xd8, 0x00, 0xf4]),
        ("psrldq xmm0, 1", &[0x66, 0x0f, 0x73, 0xd8, 0x01, 0xf4]),
        ("psrldq xmm0, 15", &[0x66, 0x0f, 0x73, 0xd8, 0x0f, 0xf4]),
        ("psrldq xmm0, 16", &[0x66, 0x0f, 0x73, 0xd8, 0x10, 0xf4]),
        ("psrldq xmm0, 17", &[0x66, 0x0f, 0x73, 0xd8, 0x11, 0xf4]),
        ("psrldq xmm0, 255", &[0x66, 0x0f, 0x73, 0xd8, 0xff, 0xf4]),
        ("psrldq xmm0, 4", &[0x66, 0x0f, 0x73, 0xd8, 0x04, 0xf4]),
        ("psrldq xmm0, 8", &[0x66, 0x0f, 0x73, 0xd8, 0x08, 0xf4]),
        ("psrlq mm0, 0", &[0x0f, 0x73, 0xd0, 0x00, 0xf4]),
        ("psrlq mm0, 1", &[0x0f, 0x73, 0xd0, 0x01, 0xf4]),
        ("psrlq mm0, 15", &[0x0f, 0x73, 0xd0, 0x0f, 0xf4]),
        ("psrlq mm0, 16", &[0x0f, 0x73, 0xd0, 0x10, 0xf4]),
        ("psrlq mm0, 255", &[0x0f, 0x73, 0xd0, 0xff, 0xf4]),
        ("psrlq mm0, 3", &[0x0f, 0x73, 0xd0, 0x03, 0xf4]),
        ("psrlq mm0, 31", &[0x0f, 0x73, 0xd0, 0x1f, 0xf4]),
        ("psrlq mm0, 32", &[0x0f, 0x73, 0xd0, 0x20, 0xf4]),
        ("psrlq mm0, 33", &[0x0f, 0x73, 0xd0, 0x21, 0xf4]),
        ("psrlq mm0, 63", &[0x0f, 0x73, 0xd0, 0x3f, 0xf4]),
        ("psrlq mm0, 64", &[0x0f, 0x73, 0xd0, 0x40, 0xf4]),
        ("psrlq mm0, 65", &[0x0f, 0x73, 0xd0, 0x41, 0xf4]),
        ("psrlq mm0, 7", &[0x0f, 0x73, 0xd0, 0x07, 0xf4]),
        ("psrlq mm0, 8", &[0x0f, 0x73, 0xd0, 0x08, 0xf4]),
        ("psrlq mm0, mm0", &[0x0f, 0xd3, 0xc0, 0xf4]),
        ("psrlq mm0, mm1", &[0x0f, 0xd3, 0xc1, 0xf4]),
        ("psrlq mm0, qword ptr [rsi]", &[0x0f, 0xd3, 0x06, 0xf4]),
        ("psrlq xmm0, 0", &[0x66, 0x0f, 0x73, 0xd0, 0x00, 0xf4]),
        ("psrlq xmm0, 1", &[0x66, 0x0f, 0x73, 0xd0, 0x01, 0xf4]),
        ("psrlq xmm0, 15", &[0x66, 0x0f, 0x73, 0xd0, 0x0f, 0xf4]),
        ("psrlq xmm0, 16", &[0x66, 0x0f, 0x73, 0xd0, 0x10, 0xf4]),
        ("psrlq xmm0, 255", &[0x66, 0x0f, 0x73, 0xd0, 0xff, 0xf4]),
        ("psrlq xmm0, 3", &[0x66, 0x0f, 0x73, 0xd0, 0x03, 0xf4]),
        ("psrlq xmm0, 31", &[0x66, 0x0f, 0x73, 0xd0, 0x1f, 0xf4]),
        ("psrlq xmm0, 32", &[0x66, 0x0f, 0x73, 0xd0, 0x20, 0xf4]),
        ("psrlq xmm0, 33", &[0x66, 0x0f, 0x73, 0xd0, 0x21, 0xf4]),
        ("psrlq xmm0, 63", &[0x66, 0x0f, 0x73, 0xd0, 0x3f, 0xf4]),
        ("psrlq xmm0, 64", &[0x66, 0x0f, 0x73, 0xd0, 0x40, 0xf4]),
        ("psrlq xmm0, 65", &[0x66, 0x0f, 0x73, 0xd0, 0x41, 0xf4]),
        ("psrlq xmm0, 7", &[0x66, 0x0f, 0x73, 0xd0, 0x07, 0xf4]),
        ("psrlq xmm0, 8", &[0x66, 0x0f, 0x73, 0xd0, 0x08, 0xf4]),
        ("psrlq xmm0, xmm1", &[0x66, 0x0f, 0xd3, 0xc1, 0xf4]),
        (
            "psrlq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd3, 0x06, 0xf4],
        ),
        ("psrlw mm0, 0", &[0x0f, 0x71, 0xd0, 0x00, 0xf4]),
        ("psrlw mm0, 1", &[0x0f, 0x71, 0xd0, 0x01, 0xf4]),
        ("psrlw mm0, 15", &[0x0f, 0x71, 0xd0, 0x0f, 0xf4]),
        ("psrlw mm0, 16", &[0x0f, 0x71, 0xd0, 0x10, 0xf4]),
        ("psrlw mm0, 255", &[0x0f, 0x71, 0xd0, 0xff, 0xf4]),
        ("psrlw mm0, 3", &[0x0f, 0x71, 0xd0, 0x03, 0xf4]),
        ("psrlw mm0, 31", &[0x0f, 0x71, 0xd0, 0x1f, 0xf4]),
        ("psrlw mm0, 32", &[0x0f, 0x71, 0xd0, 0x20, 0xf4]),
        ("psrlw mm0, 33", &[0x0f, 0x71, 0xd0, 0x21, 0xf4]),
        ("psrlw mm0, 63", &[0x0f, 0x71, 0xd0, 0x3f, 0xf4]),
        ("psrlw mm0, 64", &[0x0f, 0x71, 0xd0, 0x40, 0xf4]),
        ("psrlw mm0, 65", &[0x0f, 0x71, 0xd0, 0x41, 0xf4]),
        ("psrlw mm0, 7", &[0x0f, 0x71, 0xd0, 0x07, 0xf4]),
        ("psrlw mm0, 8", &[0x0f, 0x71, 0xd0, 0x08, 0xf4]),
        ("psrlw mm0, mm0", &[0x0f, 0xd1, 0xc0, 0xf4]),
        ("psrlw mm0, mm1", &[0x0f, 0xd1, 0xc1, 0xf4]),
        ("psrlw mm0, qword ptr [rsi]", &[0x0f, 0xd1, 0x06, 0xf4]),
        ("psrlw xmm0, 0", &[0x66, 0x0f, 0x71, 0xd0, 0x00, 0xf4]),
        ("psrlw xmm0, 1", &[0x66, 0x0f, 0x71, 0xd0, 0x01, 0xf4]),
        ("psrlw xmm0, 15", &[0x66, 0x0f, 0x71, 0xd0, 0x0f, 0xf4]),
        ("psrlw xmm0, 16", &[0x66, 0x0f, 0x71, 0xd0, 0x10, 0xf4]),
        ("psrlw xmm0, 255", &[0x66, 0x0f, 0x71, 0xd0, 0xff, 0xf4]),
        ("psrlw xmm0, 3", &[0x66, 0x0f, 0x71, 0xd0, 0x03, 0xf4]),
        ("psrlw xmm0, 31", &[0x66, 0x0f, 0x71, 0xd0, 0x1f, 0xf4]),
        ("psrlw xmm0, 32", &[0x66, 0x0f, 0x71, 0xd0, 0x20, 0xf4]),
        ("psrlw xmm0, 33", &[0x66, 0x0f, 0x71, 0xd0, 0x21, 0xf4]),
        ("psrlw xmm0, 63", &[0x66, 0x0f, 0x71, 0xd0, 0x3f, 0xf4]),
        ("psrlw xmm0, 64", &[0x66, 0x0f, 0x71, 0xd0, 0x40, 0xf4]),
        ("psrlw xmm0, 65", &[0x66, 0x0f, 0x71, 0xd0, 0x41, 0xf4]),
        ("psrlw xmm0, 7", &[0x66, 0x0f, 0x71, 0xd0, 0x07, 0xf4]),
        ("psrlw xmm0, 8", &[0x66, 0x0f, 0x71, 0xd0, 0x08, 0xf4]),
        ("psrlw xmm0, xmm1", &[0x66, 0x0f, 0xd1, 0xc1, 0xf4]),
        (
            "psrlw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd1, 0x06, 0xf4],
        ),
        ("psubb mm0, mm0", &[0x0f, 0xf8, 0xc0, 0xf4]),
        ("psubb mm0, mm1", &[0x0f, 0xf8, 0xc1, 0xf4]),
        ("psubb mm0, qword ptr [rsi]", &[0x0f, 0xf8, 0x06, 0xf4]),
        ("psubb xmm0, xmm1", &[0x66, 0x0f, 0xf8, 0xc1, 0xf4]),
        (
            "psubb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf8, 0x06, 0xf4],
        ),
        ("psubd mm0, mm0", &[0x0f, 0xfa, 0xc0, 0xf4]),
        ("psubd mm0, mm1", &[0x0f, 0xfa, 0xc1, 0xf4]),
        ("psubd mm0, qword ptr [rsi]", &[0x0f, 0xfa, 0x06, 0xf4]),
        ("psubd xmm0, xmm1", &[0x66, 0x0f, 0xfa, 0xc1, 0xf4]),
        (
            "psubd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xfa, 0x06, 0xf4],
        ),
        ("psubq mm0, mm0", &[0x0f, 0xfb, 0xc0, 0xf4]),
        ("psubq mm0, mm1", &[0x0f, 0xfb, 0xc1, 0xf4]),
        ("psubq mm0, qword ptr [rsi]", &[0x0f, 0xfb, 0x06, 0xf4]),
        ("psubq xmm0, xmm1", &[0x66, 0x0f, 0xfb, 0xc1, 0xf4]),
        (
            "psubq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xfb, 0x06, 0xf4],
        ),
        ("psubsb mm0, mm0", &[0x0f, 0xe8, 0xc0, 0xf4]),
        ("psubsb mm0, mm1", &[0x0f, 0xe8, 0xc1, 0xf4]),
        ("psubsb mm0, qword ptr [rsi]", &[0x0f, 0xe8, 0x06, 0xf4]),
        ("psubsb xmm0, xmm1", &[0x66, 0x0f, 0xe8, 0xc1, 0xf4]),
        (
            "psubsb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe8, 0x06, 0xf4],
        ),
        ("psubsw mm0, mm0", &[0x0f, 0xe9, 0xc0, 0xf4]),
        ("psubsw mm0, mm1", &[0x0f, 0xe9, 0xc1, 0xf4]),
        ("psubsw mm0, qword ptr [rsi]", &[0x0f, 0xe9, 0x06, 0xf4]),
        ("psubsw xmm0, xmm1", &[0x66, 0x0f, 0xe9, 0xc1, 0xf4]),
        (
            "psubsw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xe9, 0x06, 0xf4],
        ),
        ("psubusb mm0, mm0", &[0x0f, 0xd8, 0xc0, 0xf4]),
        ("psubusb mm0, mm1", &[0x0f, 0xd8, 0xc1, 0xf4]),
        ("psubusb mm0, qword ptr [rsi]", &[0x0f, 0xd8, 0x06, 0xf4]),
        ("psubusb xmm0, xmm1", &[0x66, 0x0f, 0xd8, 0xc1, 0xf4]),
        (
            "psubusb xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd8, 0x06, 0xf4],
        ),
        ("psubusw mm0, mm0", &[0x0f, 0xd9, 0xc0, 0xf4]),
        ("psubusw mm0, mm1", &[0x0f, 0xd9, 0xc1, 0xf4]),
        ("psubusw mm0, qword ptr [rsi]", &[0x0f, 0xd9, 0x06, 0xf4]),
        ("psubusw xmm0, xmm1", &[0x66, 0x0f, 0xd9, 0xc1, 0xf4]),
        (
            "psubusw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xd9, 0x06, 0xf4],
        ),
        ("psubw mm0, mm0", &[0x0f, 0xf9, 0xc0, 0xf4]),
        ("psubw mm0, mm1", &[0x0f, 0xf9, 0xc1, 0xf4]),
        ("psubw mm0, qword ptr [rsi]", &[0x0f, 0xf9, 0x06, 0xf4]),
        ("psubw xmm0, xmm1", &[0x66, 0x0f, 0xf9, 0xc1, 0xf4]),
        (
            "psubw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xf9, 0x06, 0xf4],
        ),
        ("punpckhbw mm0, mm0", &[0x0f, 0x68, 0xc0, 0xf4]),
        ("punpckhbw mm0, mm1", &[0x0f, 0x68, 0xc1, 0xf4]),
        ("punpckhbw mm0, qword ptr [rsi]", &[0x0f, 0x68, 0x06, 0xf4]),
        ("punpckhbw xmm0, xmm1", &[0x66, 0x0f, 0x68, 0xc1, 0xf4]),
        (
            "punpckhbw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x68, 0x06, 0xf4],
        ),
        ("punpckhdq mm0, mm0", &[0x0f, 0x6a, 0xc0, 0xf4]),
        ("punpckhdq mm0, mm1", &[0x0f, 0x6a, 0xc1, 0xf4]),
        ("punpckhdq mm0, qword ptr [rsi]", &[0x0f, 0x6a, 0x06, 0xf4]),
        ("punpckhdq xmm0, xmm1", &[0x66, 0x0f, 0x6a, 0xc1, 0xf4]),
        (
            "punpckhdq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x6a, 0x06, 0xf4],
        ),
        ("punpckhqdq xmm0, xmm1", &[0x66, 0x0f, 0x6d, 0xc1, 0xf4]),
        (
            "punpckhqdq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x6d, 0x06, 0xf4],
        ),
        ("punpckhwd mm0, mm0", &[0x0f, 0x69, 0xc0, 0xf4]),
        ("punpckhwd mm0, mm1", &[0x0f, 0x69, 0xc1, 0xf4]),
        ("punpckhwd mm0, qword ptr [rsi]", &[0x0f, 0x69, 0x06, 0xf4]),
        ("punpckhwd xmm0, xmm1", &[0x66, 0x0f, 0x69, 0xc1, 0xf4]),
        (
            "punpckhwd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x69, 0x06, 0xf4],
        ),
        ("punpcklbw mm0, dword ptr [rsi]", &[0x0f, 0x60, 0x06, 0xf4]),
        ("punpcklbw mm0, mm0", &[0x0f, 0x60, 0xc0, 0xf4]),
        ("punpcklbw mm0, mm1", &[0x0f, 0x60, 0xc1, 0xf4]),
        ("punpcklbw xmm0, xmm1", &[0x66, 0x0f, 0x60, 0xc1, 0xf4]),
        (
            "punpcklbw xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x60, 0x06, 0xf4],
        ),
        ("punpckldq mm0, dword ptr [rsi]", &[0x0f, 0x62, 0x06, 0xf4]),
        ("punpckldq mm0, mm0", &[0x0f, 0x62, 0xc0, 0xf4]),
        ("punpckldq mm0, mm1", &[0x0f, 0x62, 0xc1, 0xf4]),
        ("punpckldq xmm0, xmm1", &[0x66, 0x0f, 0x62, 0xc1, 0xf4]),
        (
            "punpckldq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x62, 0x06, 0xf4],
        ),
        ("punpcklqdq xmm0, xmm1", &[0x66, 0x0f, 0x6c, 0xc1, 0xf4]),
        (
            "punpcklqdq xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x6c, 0x06, 0xf4],
        ),
        ("punpcklwd mm0, dword ptr [rsi]", &[0x0f, 0x61, 0x06, 0xf4]),
        ("punpcklwd mm0, mm0", &[0x0f, 0x61, 0xc0, 0xf4]),
        ("punpcklwd mm0, mm1", &[0x0f, 0x61, 0xc1, 0xf4]),
        ("punpcklwd xmm0, xmm1", &[0x66, 0x0f, 0x61, 0xc1, 0xf4]),
        (
            "punpcklwd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x61, 0x06, 0xf4],
        ),
        ("pxor mm0, mm0", &[0x0f, 0xef, 0xc0, 0xf4]),
        ("pxor mm0, mm1", &[0x0f, 0xef, 0xc1, 0xf4]),
        ("pxor mm0, qword ptr [rsi]", &[0x0f, 0xef, 0x06, 0xf4]),
        ("pxor xmm0, xmm1", &[0x66, 0x0f, 0xef, 0xc1, 0xf4]),
        (
            "pxor xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0xef, 0x06, 0xf4],
        ),
        ("rcpps xmm0, xmm1", &[0x0f, 0x53, 0xc1, 0xf4]),
        ("rcpps xmm0, xmmword ptr [rsi]", &[0x0f, 0x53, 0x06, 0xf4]),
        (
            "rcpss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x53, 0x06, 0xf4],
        ),
        ("rcpss xmm0, xmm1", &[0xf3, 0x0f, 0x53, 0xc1, 0xf4]),
        ("rsqrtps xmm0, xmm1", &[0x0f, 0x52, 0xc1, 0xf4]),
        ("rsqrtps xmm0, xmmword ptr [rsi]", &[0x0f, 0x52, 0x06, 0xf4]),
        (
            "rsqrtss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x52, 0x06, 0xf4],
        ),
        ("rsqrtss xmm0, xmm1", &[0xf3, 0x0f, 0x52, 0xc1, 0xf4]),
        ("sfence", &[0x0f, 0xae, 0xf8, 0xf4]),
        (
            "shufpd xmm0, xmm1, 0",
            &[0x66, 0x0f, 0xc6, 0xc1, 0x00, 0xf4],
        ),
        (
            "shufpd xmm0, xmm1, 1",
            &[0x66, 0x0f, 0xc6, 0xc1, 0x01, 0xf4],
        ),
        (
            "shufpd xmm0, xmm1, 2",
            &[0x66, 0x0f, 0xc6, 0xc1, 0x02, 0xf4],
        ),
        (
            "shufpd xmm0, xmm1, 3",
            &[0x66, 0x0f, 0xc6, 0xc1, 0x03, 0xf4],
        ),
        (
            "shufpd xmm0, xmmword ptr [rsi], 1",
            &[0x66, 0x0f, 0xc6, 0x06, 0x01, 0xf4],
        ),
        ("shufps xmm0, xmm1, 0", &[0x0f, 0xc6, 0xc1, 0x00, 0xf4]),
        ("shufps xmm0, xmm1, 0x1b", &[0x0f, 0xc6, 0xc1, 0x1b, 0xf4]),
        ("shufps xmm0, xmm1, 0x39", &[0x0f, 0xc6, 0xc1, 0x39, 0xf4]),
        ("shufps xmm0, xmm1, 0x4e", &[0x0f, 0xc6, 0xc1, 0x4e, 0xf4]),
        ("shufps xmm0, xmm1, 0x93", &[0x0f, 0xc6, 0xc1, 0x93, 0xf4]),
        ("shufps xmm0, xmm1, 0xe4", &[0x0f, 0xc6, 0xc1, 0xe4, 0xf4]),
        ("shufps xmm0, xmm1, 0xff", &[0x0f, 0xc6, 0xc1, 0xff, 0xf4]),
        (
            "shufps xmm0, xmmword ptr [rsi], 0x1b",
            &[0x0f, 0xc6, 0x06, 0x1b, 0xf4],
        ),
        ("sqrtpd xmm0, xmm1", &[0x66, 0x0f, 0x51, 0xc1, 0xf4]),
        (
            "sqrtpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x51, 0x06, 0xf4],
        ),
        ("sqrtps xmm0, xmm1", &[0x0f, 0x51, 0xc1, 0xf4]),
        ("sqrtps xmm0, xmmword ptr [rsi]", &[0x0f, 0x51, 0x06, 0xf4]),
        (
            "sqrtsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x51, 0x06, 0xf4],
        ),
        ("sqrtsd xmm0, xmm1", &[0xf2, 0x0f, 0x51, 0xc1, 0xf4]),
        ("sqrtsd xmm3, xmm3", &[0xf2, 0x0f, 0x51, 0xdb, 0xf4]),
        (
            "sqrtss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x51, 0x06, 0xf4],
        ),
        ("sqrtss xmm0, xmm1", &[0xf3, 0x0f, 0x51, 0xc1, 0xf4]),
        ("stmxcsr dword ptr [rsi]", &[0x0f, 0xae, 0x1e, 0xf4]),
        ("subpd xmm0, xmm1", &[0x66, 0x0f, 0x5c, 0xc1, 0xf4]),
        (
            "subpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x5c, 0x06, 0xf4],
        ),
        ("subps xmm0, xmm1", &[0x0f, 0x5c, 0xc1, 0xf4]),
        ("subps xmm0, xmmword ptr [rsi]", &[0x0f, 0x5c, 0x06, 0xf4]),
        (
            "subsd xmm0, qword ptr [rsi]",
            &[0xf2, 0x0f, 0x5c, 0x06, 0xf4],
        ),
        ("subsd xmm0, xmm1", &[0xf2, 0x0f, 0x5c, 0xc1, 0xf4]),
        (
            "subss xmm0, dword ptr [rsi]",
            &[0xf3, 0x0f, 0x5c, 0x06, 0xf4],
        ),
        ("subss xmm0, xmm1", &[0xf3, 0x0f, 0x5c, 0xc1, 0xf4]),
        (
            "ucomisd xmm0, qword ptr [rsi]",
            &[0x66, 0x0f, 0x2e, 0x06, 0xf4],
        ),
        ("ucomisd xmm0, xmm1", &[0x66, 0x0f, 0x2e, 0xc1, 0xf4]),
        ("ucomiss xmm0, dword ptr [rsi]", &[0x0f, 0x2e, 0x06, 0xf4]),
        ("ucomiss xmm0, xmm1", &[0x0f, 0x2e, 0xc1, 0xf4]),
        ("unpckhpd xmm0, xmm1", &[0x66, 0x0f, 0x15, 0xc1, 0xf4]),
        (
            "unpckhpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x15, 0x06, 0xf4],
        ),
        ("unpckhps xmm0, xmm1", &[0x0f, 0x15, 0xc1, 0xf4]),
        (
            "unpckhps xmm0, xmmword ptr [rsi]",
            &[0x0f, 0x15, 0x06, 0xf4],
        ),
        ("unpcklpd xmm0, xmm1", &[0x66, 0x0f, 0x14, 0xc1, 0xf4]),
        (
            "unpcklpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x14, 0x06, 0xf4],
        ),
        ("unpcklps xmm0, xmm1", &[0x0f, 0x14, 0xc1, 0xf4]),
        (
            "unpcklps xmm0, xmmword ptr [rsi]",
            &[0x0f, 0x14, 0x06, 0xf4],
        ),
        ("xorpd xmm0, xmm1", &[0x66, 0x0f, 0x57, 0xc1, 0xf4]),
        (
            "xorpd xmm0, xmmword ptr [rsi]",
            &[0x66, 0x0f, 0x57, 0x06, 0xf4],
        ),
        ("xorps xmm0, xmm1", &[0x0f, 0x57, 0xc1, 0xf4]),
        ("xorps xmm0, xmmword ptr [rsi]", &[0x0f, 0x57, 0x06, 0xf4]),
    ];

    const RC_NEAREST: u32 = ROUND_NEAREST_EVEN;
    const RC_DOWN: u32 = ROUND_DOWN;
    const RC_UP: u32 = ROUND_UP;
    const RC_ZERO: u32 = ROUND_TOWARD_ZERO;
    const ALL_ROUNDING_MODES: [u32; 4] = [RC_NEAREST, RC_DOWN, RC_UP, RC_ZERO];
    const ARITHMETIC_OPERATIONS: [Arithmetic; 5] = [
        Arithmetic::Add,
        Arithmetic::Subtract,
        Arithmetic::Multiply,
        Arithmetic::Divide,
        Arithmetic::SquareRoot,
    ];

    fn env_with(mxcsr_bits: u32) -> FpEnv {
        FpEnv::new(MXCSR_RESET | mxcsr_bits)
    }
    fn env_rc(rounding: u32) -> FpEnv {
        env_with(rounding << MXCSR_RC_SHIFT)
    }
    fn default_env() -> FpEnv {
        FpEnv::new(MXCSR_RESET)
    }

    fn single(env: &mut FpEnv, op: Arithmetic, a: u32, b: u32) -> u32 {
        arithmetic(env, SINGLE, op, a as u64, b as u64) as u32
    }
    fn double(env: &mut FpEnv, op: Arithmetic, a: u64, b: u64) -> u64 {
        arithmetic(env, DOUBLE, op, a, b)
    }

    fn host_single(op: Arithmetic, a: f32, b: f32) -> f32 {
        match op {
            Arithmetic::Add => a + b,
            Arithmetic::Subtract => a - b,
            Arithmetic::Multiply => a * b,
            Arithmetic::Divide => a / b,
            Arithmetic::SquareRoot => b.sqrt(),
        }
    }
    fn host_double(op: Arithmetic, a: f64, b: f64) -> f64 {
        match op {
            Arithmetic::Add => a + b,
            Arithmetic::Subtract => a - b,
            Arithmetic::Multiply => a * b,
            Arithmetic::Divide => a / b,
            Arithmetic::SquareRoot => b.sqrt(),
        }
    }

    const SPECIAL_DOUBLES: [u64; 14] = [
        0,
        1 << 63,
        ONE_DOUBLE,
        ONE_DOUBLE | SIGN_DOUBLE,
        INFINITY_DOUBLE,
        INFINITY_DOUBLE | SIGN_DOUBLE,
        MIN_SUBNORMAL_DOUBLE,
        MIN_NORMAL_DOUBLE,
        MIN_NORMAL_DOUBLE - 1,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        TWO_DOUBLE,
        0x3fe0_0000_0000_0000,
        0x3ff0_0000_0000_0001,
    ];
    const SPECIAL_SINGLES: [u32; 14] = [
        0,
        1 << 31,
        ONE_SINGLE,
        ONE_SINGLE | SIGN_SINGLE,
        INFINITY_SINGLE,
        INFINITY_SINGLE | SIGN_SINGLE,
        MIN_SUBNORMAL_SINGLE,
        MIN_NORMAL_SINGLE,
        MIN_NORMAL_SINGLE - 1,
        MAX_SINGLE,
        MAX_SINGLE | SIGN_SINGLE,
        TWO_SINGLE,
        HALF_SINGLE,
        0x3f80_0001,
    ];

    /// Operands that hit ties, subnormals and the exponent extremes as often as plain values do.
    fn random_double(rng: &mut XorShift) -> u64 {
        let sign = (rng.next() & 1) << 63;
        sign | match rng.next() % 8 {
            0 => rng.next() & !SIGN_DOUBLE,
            1..=3 => {
                let exponent = 1023 - 40 + rng.next() % 80;
                (rng.next() & 0x000f_ffff_ffff_ffff) | (exponent << 52)
            }
            4 => rng.next() & 0x000f_ffff_ffff_ffff,
            5 => SPECIAL_DOUBLES[(rng.next() % 14) as usize] & !SIGN_DOUBLE,
            6 => {
                let exponent = 1023 - 20 + rng.next() % 40;
                ((rng.next() & 0xff) << (rng.next() % 45)) & 0x000f_ffff_ffff_ffff
                    | (exponent << 52)
            }
            _ => {
                let exponent = rng.next() % 0x7ff;
                (rng.next() & 0x000f_ffff_ffff_ffff) | (exponent << 52)
            }
        }
    }

    fn random_single(rng: &mut XorShift) -> u32 {
        let sign = ((rng.next() & 1) as u32) << 31;
        sign | match rng.next() % 8 {
            0 => rng.next() as u32 & 0x7fff_ffff,
            1..=3 => {
                let exponent = 127 - 20 + (rng.next() % 40) as u32;
                (rng.next() as u32 & 0x007f_ffff) | (exponent << 23)
            }
            4 => rng.next() as u32 & 0x007f_ffff,
            5 => SPECIAL_SINGLES[(rng.next() % 14) as usize] & 0x7fff_ffff,
            6 => {
                let exponent = 127 - 10 + (rng.next() % 20) as u32;
                (((rng.next() as u32 & 0xff) << (rng.next() % 16)) & 0x007f_ffff) | (exponent << 23)
            }
            _ => {
                let exponent = (rng.next() % 0xff) as u32;
                (rng.next() as u32 & 0x007f_ffff) | (exponent << 23)
            }
        }
    }

    /// Sign (-1, 0, 1) of the exact result minus the nearest result `r`, from error-free transforms
    /// that are exact while nothing overflows or underflows.
    fn error_sign_double(op: Arithmetic, a: f64, b: f64, r: f64) -> i32 {
        let error = match op {
            Arithmetic::Add => {
                let bb = r - a;
                (a - (r - bb)) + (b - bb)
            }
            Arithmetic::Subtract => {
                let b = -b;
                let bb = r - a;
                (a - (r - bb)) + (b - bb)
            }
            Arithmetic::Multiply => a.mul_add(b, -r),
            Arithmetic::Divide => {
                let residual = (-r).mul_add(b, a);
                if b < 0.0 { -residual } else { residual }
            }
            Arithmetic::SquareRoot => (-r).mul_add(r, b),
        };
        if error > 0.0 {
            1
        } else if error < 0.0 {
            -1
        } else {
            0
        }
    }

    fn error_sign_single(op: Arithmetic, a: f32, b: f32, r: f32) -> i32 {
        let error = match op {
            Arithmetic::Add => {
                let bb = r - a;
                (a - (r - bb)) + (b - bb)
            }
            Arithmetic::Subtract => {
                let b = -b;
                let bb = r - a;
                (a - (r - bb)) + (b - bb)
            }
            Arithmetic::Multiply => a.mul_add(b, -r),
            Arithmetic::Divide => {
                let residual = (-r).mul_add(b, a);
                if b < 0.0 { -residual } else { residual }
            }
            Arithmetic::SquareRoot => (-r).mul_add(r, b),
        };
        if error > 0.0 {
            1
        } else if error < 0.0 {
            -1
        } else {
            0
        }
    }

    fn directed_double(rounding: u32, r: f64, error_sign: i32) -> f64 {
        let toward_zero_is_down = !r.is_sign_negative();
        let down = match rounding {
            RC_DOWN => true,
            RC_ZERO => toward_zero_is_down,
            _ => false,
        };
        let up = match rounding {
            RC_UP => true,
            RC_ZERO => !toward_zero_is_down,
            _ => false,
        };
        if down && error_sign < 0 {
            r.next_down()
        } else if up && error_sign > 0 {
            r.next_up()
        } else {
            r
        }
    }

    fn directed_single(rounding: u32, r: f32, error_sign: i32) -> f32 {
        let toward_zero_is_down = !r.is_sign_negative();
        let down = match rounding {
            RC_DOWN => true,
            RC_ZERO => toward_zero_is_down,
            _ => false,
        };
        let up = match rounding {
            RC_UP => true,
            RC_ZERO => !toward_zero_is_down,
            _ => false,
        };
        if down && error_sign < 0 {
            r.next_down()
        } else if up && error_sign > 0 {
            r.next_up()
        } else {
            r
        }
    }

    const RANDOM_CASES: usize = 40_000;

    #[test]
    fn double_arithmetic_matches_the_host_for_nearest_even() {
        let mut rng = XorShift(0x1234_5678_9abc_def1);
        for _ in 0..RANDOM_CASES {
            let (a, b) = (random_double(&mut rng), random_double(&mut rng));
            if f64::from_bits(a).is_nan() || f64::from_bits(b).is_nan() {
                continue;
            }
            for op in ARITHMETIC_OPERATIONS {
                let host = host_double(op, f64::from_bits(a), f64::from_bits(b));
                let got = double(&mut default_env(), op, a, b);
                let expected = if host.is_nan() {
                    DEFAULT_NAN_DOUBLE
                } else {
                    host.to_bits()
                };
                assert_eq!(got, expected, "{a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn single_arithmetic_matches_the_host_for_nearest_even() {
        let mut rng = XorShift(0x0f0f_1234_5555_aaa1);
        for _ in 0..RANDOM_CASES {
            let (a, b) = (random_single(&mut rng), random_single(&mut rng));
            if f32::from_bits(a).is_nan() || f32::from_bits(b).is_nan() {
                continue;
            }
            for op in ARITHMETIC_OPERATIONS {
                let host = host_single(op, f32::from_bits(a), f32::from_bits(b));
                let got = single(&mut default_env(), op, a, b);
                let expected = if host.is_nan() {
                    DEFAULT_NAN_SINGLE
                } else {
                    host.to_bits()
                };
                assert_eq!(got, expected, "{a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn double_directed_rounding_and_precision_flag_match_error_free_transforms() {
        let mut rng = XorShift(0xdead_beef_0bad_f00d);
        for _ in 0..RANDOM_CASES / 2 {
            // exponents within +-40 of 1: no overflow or underflow, so the transforms are exact
            let exponent = |rng: &mut XorShift| 1023 - 40 + rng.next() % 80;
            let make = |rng: &mut XorShift| {
                let e = exponent(rng);
                let m = if rng.next().is_multiple_of(3) {
                    (rng.next() & 0xff) << (rng.next() % 44)
                } else {
                    rng.next() & 0x000f_ffff_ffff_ffff
                };
                f64::from_bits(((rng.next() & 1) << 63) | (e << 52) | (m & 0x000f_ffff_ffff_ffff))
            };
            let (a, b) = (make(&mut rng), make(&mut rng));
            for op in ARITHMETIC_OPERATIONS {
                if op == Arithmetic::SquareRoot && b.is_sign_negative() {
                    continue;
                }
                let nearest = host_double(op, a, b);
                let error = error_sign_double(op, a, b, nearest);
                for rounding in ALL_ROUNDING_MODES {
                    let mut env = env_rc(rounding);
                    let got = double(&mut env, op, a.to_bits(), b.to_bits());
                    let expected = directed_double(rounding, nearest, error);
                    assert_eq!(got, expected.to_bits(), "rc {rounding} {a:e} {b:e}");
                    assert_eq!(
                        env.flags,
                        if error != 0 { FLAG_PRECISION } else { 0 },
                        "rc {rounding} {a:e} {b:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn single_directed_rounding_and_precision_flag_match_error_free_transforms() {
        let mut rng = XorShift(0x7777_1111_3333_5555);
        for _ in 0..RANDOM_CASES / 2 {
            let make = |rng: &mut XorShift| {
                let exponent = 127 - 20 + (rng.next() % 40) as u32;
                let m = if rng.next().is_multiple_of(3) {
                    (rng.next() as u32 & 0xff) << (rng.next() % 15)
                } else {
                    rng.next() as u32
                };
                f32::from_bits(
                    (((rng.next() & 1) as u32) << 31) | (exponent << 23) | (m & 0x007f_ffff),
                )
            };
            let (a, b) = (make(&mut rng), make(&mut rng));
            for op in ARITHMETIC_OPERATIONS {
                if op == Arithmetic::SquareRoot && b.is_sign_negative() {
                    continue;
                }
                let nearest = host_single(op, a, b);
                let error = error_sign_single(op, a, b, nearest);
                for rounding in ALL_ROUNDING_MODES {
                    let mut env = env_rc(rounding);
                    let got = single(&mut env, op, a.to_bits(), b.to_bits());
                    let expected = directed_single(rounding, nearest, error);
                    assert_eq!(got, expected.to_bits(), "rc {rounding} {a:e} {b:e}");
                    assert_eq!(
                        env.flags,
                        if error != 0 { FLAG_PRECISION } else { 0 },
                        "rc {rounding} {a:e} {b:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn one_third_rounds_per_mode_in_single_precision() {
        let three = THREE_SINGLE;
        for (rounding, expected) in [
            (RC_NEAREST, ONE_THIRD_SINGLE_UP),
            (RC_DOWN, ONE_THIRD_SINGLE_DOWN),
            (RC_UP, ONE_THIRD_SINGLE_UP),
            (RC_ZERO, ONE_THIRD_SINGLE_DOWN),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                single(&mut env, Arithmetic::Divide, ONE_SINGLE, three),
                expected
            );
            assert_eq!(env.flags, FLAG_PRECISION);
        }
    }

    #[test]
    fn negative_one_third_rounds_away_from_zero_when_rounding_down() {
        let minus_three = 0xc040_0000;
        for (rounding, expected) in [
            (RC_NEAREST, (ONE_THIRD_SINGLE_UP | SIGN_SINGLE)),
            (RC_DOWN, (ONE_THIRD_SINGLE_UP | SIGN_SINGLE)),
            (RC_UP, (ONE_THIRD_SINGLE_DOWN | SIGN_SINGLE)),
            (RC_ZERO, (ONE_THIRD_SINGLE_DOWN | SIGN_SINGLE)),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                single(&mut env, Arithmetic::Divide, ONE_SINGLE, minus_three),
                expected
            );
        }
    }

    #[test]
    fn one_third_rounds_per_mode_in_double_precision() {
        let three = THREE_DOUBLE;
        for (rounding, expected) in [
            (RC_NEAREST, ONE_THIRD_DOUBLE_DOWN),
            (RC_DOWN, ONE_THIRD_DOUBLE_DOWN),
            (RC_UP, ONE_THIRD_DOUBLE_UP),
            (RC_ZERO, ONE_THIRD_DOUBLE_DOWN),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                double(&mut env, Arithmetic::Divide, ONE_DOUBLE, three),
                expected
            );
        }
    }

    #[test]
    fn sqrt_two_rounds_per_mode_in_double_precision() {
        let two = TWO_DOUBLE;
        // The correctly rounded root is 1.41421356237309514547..., just above the true value.
        for (rounding, expected) in [
            (RC_NEAREST, 0x3ff6_a09e_667f_3bcd),
            (RC_DOWN, 0x3ff6_a09e_667f_3bcc),
            (RC_UP, 0x3ff6_a09e_667f_3bcd),
            (RC_ZERO, 0x3ff6_a09e_667f_3bcc),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(double(&mut env, Arithmetic::SquareRoot, 0, two), expected);
            assert_eq!(env.flags, FLAG_PRECISION);
        }
    }

    #[test]
    fn exact_square_root_raises_no_flags() {
        let mut env = default_env();
        assert_eq!(
            double(&mut env, Arithmetic::SquareRoot, 0, 0x4010_0000_0000_0000),
            TWO_DOUBLE
        );
        assert_eq!(
            single(&mut env, Arithmetic::SquareRoot, 0, 0x4110_0000),
            THREE_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn half_ulp_tie_rounds_to_even_and_raises_precision() {
        // 1 + 2^-24 is exactly halfway between 1.0 and the next single; 1.0 has an even mantissa.
        let mut env = default_env();
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Add,
                ONE_SINGLE,
                TWO_POW_MINUS_24_SINGLE
            ),
            ONE_SINGLE
        );
        assert_eq!(env.flags, FLAG_PRECISION);
        // 1 + 3 * 2^-24 is halfway between ...01 (odd) and ...02 (even): rounds up to even.
        let mut env = default_env();
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Add,
                0x3f80_0001,
                TWO_POW_MINUS_24_SINGLE
            ),
            0x3f80_0002
        );
    }

    #[test]
    fn addition_of_exact_operands_raises_no_flags() {
        let mut env = default_env();
        let sum = double(&mut env, Arithmetic::Add, ONE_DOUBLE, ONE_DOUBLE);
        assert_eq!(sum, TWO_DOUBLE);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn overflow_result_depends_on_rounding_mode_and_sign() {
        let negative_max = MAX_SINGLE | SIGN_SINGLE;
        for (rounding, positive, negative) in [
            (RC_NEAREST, INFINITY_SINGLE, INFINITY_SINGLE | SIGN_SINGLE),
            (RC_DOWN, MAX_SINGLE, INFINITY_SINGLE | SIGN_SINGLE),
            (RC_UP, INFINITY_SINGLE, negative_max),
            (RC_ZERO, MAX_SINGLE, negative_max),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                single(&mut env, Arithmetic::Add, MAX_SINGLE, MAX_SINGLE),
                positive,
                "rc {rounding}"
            );
            assert_eq!(env.flags, FLAG_OVERFLOW | FLAG_PRECISION);
            let mut env = env_rc(rounding);
            assert_eq!(
                single(&mut env, Arithmetic::Add, negative_max, negative_max),
                negative,
                "rc {rounding}"
            );
        }
    }

    #[test]
    fn double_overflow_by_multiplication_raises_overflow_and_precision() {
        let mut env = default_env();
        let big = 0x7fe0_0000_0000_0000;
        assert_eq!(
            double(&mut env, Arithmetic::Multiply, big, big),
            INFINITY_DOUBLE
        );
        assert_eq!(env.flags, FLAG_OVERFLOW | FLAG_PRECISION);
    }

    #[test]
    fn rounding_up_to_the_next_binade_is_overflow_only_at_the_top() {
        // MAX + half an ulp rounds to even, which is infinity (the next binade).
        let mut env = default_env();
        let half_ulp_of_max = 0x7300_0000;
        assert_eq!(
            single(&mut env, Arithmetic::Add, MAX_SINGLE, half_ulp_of_max),
            INFINITY_SINGLE
        );
        assert_eq!(env.flags, FLAG_OVERFLOW | FLAG_PRECISION);
        // A quarter ulp does not.
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, MAX_SINGLE, 0x7280_0000),
            MAX_SINGLE
        );
        assert_eq!(env.flags, FLAG_PRECISION);
    }

    #[test]
    fn subnormal_exact_result_raises_no_flags_when_underflow_is_masked() {
        // 2^-126 * 0.5 = 2^-127 is representable as a subnormal: tiny but exact.
        let mut env = default_env();
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_NORMAL_SINGLE,
                HALF_SINGLE
            ),
            QUIET_BIT_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn subnormal_inexact_result_raises_underflow_and_precision() {
        // The smallest subnormal times 0.75 = 0.75 * 2^-149 rounds up to 2^-149.
        let mut env = default_env();
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_SUBNORMAL_SINGLE,
                0x3f40_0000
            ),
            MIN_SUBNORMAL_SINGLE
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION | FLAG_DENORMAL);
    }

    #[test]
    fn halfway_to_the_smallest_subnormal_rounds_to_even_zero() {
        let mut env = default_env();
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_SUBNORMAL_SINGLE,
                HALF_SINGLE
            ),
            0
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION | FLAG_DENORMAL);
        // Rounding up instead gives the smallest subnormal, rounding down gives zero.
        let mut env = env_rc(RC_UP);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_SUBNORMAL_SINGLE,
                HALF_SINGLE
            ),
            MIN_SUBNORMAL_SINGLE
        );
        let mut env = env_rc(RC_DOWN);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_SUBNORMAL_SINGLE,
                HALF_SINGLE
            ),
            0
        );
    }

    #[test]
    fn tiny_negative_result_rounds_to_negative_zero_or_to_the_smallest_negative_subnormal() {
        let negative_tiny = MIN_SUBNORMAL_SINGLE | SIGN_SINGLE;
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Multiply, negative_tiny, HALF_SINGLE),
            SINGLE_NEGATIVE_ZERO
        );
        let mut env = env_rc(RC_DOWN);
        assert_eq!(
            single(&mut env, Arithmetic::Multiply, negative_tiny, HALF_SINGLE),
            negative_tiny
        );
    }

    #[test]
    fn flush_to_zero_replaces_a_tiny_result_by_signed_zero() {
        let mut env = env_with(MXCSR_FTZ);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_NORMAL_SINGLE,
                HALF_SINGLE
            ),
            0
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION);
        let mut env = env_with(MXCSR_FTZ);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_NORMAL_SINGLE | SIGN_SINGLE,
                HALF_SINGLE
            ),
            SINGLE_NEGATIVE_ZERO
        );
    }

    #[test]
    fn flush_to_zero_leaves_normal_results_alone() {
        let mut env = env_with(MXCSR_FTZ);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_NORMAL_SINGLE,
                ONE_SINGLE
            ),
            MIN_NORMAL_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn flush_to_zero_does_nothing_while_underflow_is_unmasked() {
        let mut env = FpEnv::new((MXCSR_RESET | MXCSR_FTZ) & !MXCSR_UNDERFLOW_MASK);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                MIN_NORMAL_SINGLE,
                HALF_SINGLE
            ),
            QUIET_BIT_SINGLE
        );
        // Unmasked underflow is reported for any tiny result, exact or not.
        assert_eq!(env.flags, FLAG_UNDERFLOW);
    }

    #[test]
    fn flush_to_zero_applies_to_the_double_format_too() {
        let mut env = env_with(MXCSR_FTZ);
        assert_eq!(
            double(&mut env, Arithmetic::Divide, MIN_NORMAL_DOUBLE, TWO_DOUBLE),
            0
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION);
    }

    #[test]
    fn subnormal_operand_raises_denormal_flag_unless_daz() {
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, MIN_SUBNORMAL_SINGLE, ONE_SINGLE),
            ONE_SINGLE
        );
        assert_eq!(env.flags, FLAG_DENORMAL | FLAG_PRECISION);
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(
            single(&mut env, Arithmetic::Add, MIN_SUBNORMAL_SINGLE, ONE_SINGLE),
            ONE_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn daz_treats_subnormals_as_zeros_of_the_same_sign() {
        let negative_subnormal = 0x8000_0001;
        let mut env = env_with(MXCSR_DAZ);
        // -denormal * 5 = -0 under DAZ
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Multiply,
                negative_subnormal,
                0x40a0_0000
            ),
            SINGLE_NEGATIVE_ZERO
        );
        // x / denormal = x / 0 = infinity with divide-by-zero
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(
            single(
                &mut env,
                Arithmetic::Divide,
                ONE_SINGLE,
                MIN_SUBNORMAL_SINGLE
            ),
            INFINITY_SINGLE
        );
        assert_eq!(env.flags, FLAG_DIVIDE_BY_ZERO);
        // sqrt of a subnormal is sqrt(+0) = +0
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(
            single(&mut env, Arithmetic::SquareRoot, 0, MIN_SUBNORMAL_SINGLE),
            0
        );
    }

    #[test]
    fn subnormal_operands_are_computed_exactly_without_daz() {
        // 2^-149 + 2^-149 = 2^-148
        let mut env = default_env();
        assert_eq!(single(&mut env, Arithmetic::Add, 1, 1), 2,);
        assert_eq!(env.flags, FLAG_DENORMAL);
        // 2^-1074 * 2^1023 = 2^-51 exactly; the subnormal operand is normalised first
        let mut env = default_env();
        assert_eq!(
            double(
                &mut env,
                Arithmetic::Multiply,
                MIN_SUBNORMAL_DOUBLE,
                0x7fe0_0000_0000_0000
            ),
            0x3cc0_0000_0000_0000
        );
        assert_eq!(env.flags, FLAG_DENORMAL);
    }

    #[test]
    fn division_by_zero_gives_signed_infinity_and_divide_by_zero_flag() {
        for (a, b, expected) in [
            (ONE_SINGLE, 0, INFINITY_SINGLE),
            (ONE_SINGLE, 1 << 31, INFINITY_SINGLE | SIGN_SINGLE),
            (ONE_SINGLE | SIGN_SINGLE, 0, INFINITY_SINGLE | SIGN_SINGLE),
            (ONE_SINGLE | SIGN_SINGLE, 1 << 31, INFINITY_SINGLE),
        ] {
            let mut env = default_env();
            assert_eq!(single(&mut env, Arithmetic::Divide, a, b), expected);
            assert_eq!(env.flags, FLAG_DIVIDE_BY_ZERO);
        }
    }

    #[test]
    fn invalid_operations_give_the_default_nan() {
        let negative_infinity = INFINITY_SINGLE | SIGN_SINGLE;
        let cases = [
            (Arithmetic::Divide, 0, 0),
            (Arithmetic::Divide, INFINITY_SINGLE, INFINITY_SINGLE),
            (Arithmetic::Multiply, 0, INFINITY_SINGLE),
            (Arithmetic::Multiply, negative_infinity, 1 << 31),
            (Arithmetic::Add, INFINITY_SINGLE, negative_infinity),
            (Arithmetic::Subtract, INFINITY_SINGLE, INFINITY_SINGLE),
            (Arithmetic::SquareRoot, 0, 0xbf80_0000),
            (Arithmetic::SquareRoot, 0, negative_infinity),
        ];
        for (op, a, b) in cases {
            let mut env = default_env();
            assert_eq!(single(&mut env, op, a, b), DEFAULT_NAN_SINGLE);
            assert_eq!(env.flags, FLAG_INVALID);
        }
        let mut env = default_env();
        assert_eq!(
            double(&mut env, Arithmetic::Divide, 0, 0),
            DEFAULT_NAN_DOUBLE
        );
    }

    #[test]
    fn infinity_arithmetic_without_invalid_operations() {
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, INFINITY_SINGLE, ONE_SINGLE),
            INFINITY_SINGLE
        );
        assert_eq!(
            single(&mut env, Arithmetic::Subtract, ONE_SINGLE, INFINITY_SINGLE),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(
            single(&mut env, Arithmetic::Multiply, INFINITY_SINGLE, 0xc000_0000),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(
            single(&mut env, Arithmetic::Divide, ONE_SINGLE, INFINITY_SINGLE),
            0
        );
        assert_eq!(
            single(&mut env, Arithmetic::Divide, INFINITY_SINGLE, TWO_SINGLE),
            INFINITY_SINGLE
        );
        assert_eq!(
            single(&mut env, Arithmetic::SquareRoot, 0, INFINITY_SINGLE),
            INFINITY_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn signed_zero_rules() {
        let negative_zero = 1u32 << 31;
        // x - x is +0, except when rounding down
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Subtract, ONE_SINGLE, ONE_SINGLE),
            0
        );
        let mut env = env_rc(RC_DOWN);
        assert_eq!(
            single(&mut env, Arithmetic::Subtract, ONE_SINGLE, ONE_SINGLE),
            negative_zero
        );
        // (+0) + (-0) is +0, or -0 when rounding down; (-0) + (-0) is always -0
        for (rounding, expected) in [
            (RC_NEAREST, 0),
            (RC_DOWN, negative_zero),
            (RC_UP, 0),
            (RC_ZERO, 0),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                single(&mut env, Arithmetic::Add, 0, negative_zero),
                expected
            );
            assert_eq!(
                single(&mut env, Arithmetic::Add, negative_zero, negative_zero),
                negative_zero
            );
            assert_eq!(
                single(&mut env, Arithmetic::Subtract, 0, 0),
                if rounding == RC_DOWN {
                    negative_zero
                } else {
                    0
                }
            );
            assert_eq!(
                single(&mut env, Arithmetic::Subtract, negative_zero, 0),
                negative_zero
            );
        }
        // zero times anything keeps the xor of the signs; zero divided likewise
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Multiply, negative_zero, TWO_SINGLE),
            negative_zero
        );
        assert_eq!(
            single(&mut env, Arithmetic::Multiply, 0, 0xc000_0000),
            negative_zero
        );
        assert_eq!(
            single(&mut env, Arithmetic::Divide, negative_zero, 0xc000_0000),
            0
        );
        assert_eq!(
            single(&mut env, Arithmetic::SquareRoot, 0, negative_zero),
            negative_zero
        );
        assert_eq!(single(&mut env, Arithmetic::SquareRoot, 0, 0), 0);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn adding_zero_to_a_value_is_exact() {
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, 0x4049_0fdb, 0),
            0x4049_0fdb
        );
        assert_eq!(
            single(&mut env, Arithmetic::Add, 0, 0xc049_0fdb),
            0xc049_0fdb
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn nan_operand_is_returned_quieted_and_signaling_raises_invalid() {
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, ONE_SINGLE, QNAN_SINGLE),
            QNAN_SINGLE
        );
        assert_eq!(env.flags, 0);
        assert_eq!(
            single(&mut env, Arithmetic::Add, SNAN_SINGLE, ONE_SINGLE),
            SNAN_SINGLE | QUIET_BIT_SINGLE
        );
        assert_eq!(env.flags, FLAG_INVALID);
        let mut env = default_env();
        assert_eq!(
            double(&mut env, Arithmetic::Multiply, ONE_DOUBLE, SNAN_DOUBLE),
            SNAN_DOUBLE | QUIET_BIT_DOUBLE
        );
        assert_eq!(env.flags, FLAG_INVALID);
    }

    #[test]
    fn destination_nan_wins_when_both_operands_are_nan() {
        let other_qnan = 0x7fc0_0002;
        let other_snan = 0x7f80_0002;
        let cases = [
            (QNAN_SINGLE, other_qnan, QNAN_SINGLE, 0),
            (other_qnan, QNAN_SINGLE, other_qnan, 0),
            (
                SNAN_SINGLE,
                other_snan,
                SNAN_SINGLE | QUIET_BIT_SINGLE,
                FLAG_INVALID,
            ),
            // QNaN destination beats an SNaN source, but the SNaN still raises invalid
            (QNAN_SINGLE, other_snan, QNAN_SINGLE, FLAG_INVALID),
            (
                SNAN_SINGLE,
                other_qnan,
                SNAN_SINGLE | QUIET_BIT_SINGLE,
                FLAG_INVALID,
            ),
        ];
        for (a, b, expected, flags) in cases {
            for op in [
                Arithmetic::Add,
                Arithmetic::Subtract,
                Arithmetic::Multiply,
                Arithmetic::Divide,
            ] {
                let mut env = default_env();
                assert_eq!(single(&mut env, op, a, b), expected, "{a:#x} {b:#x}");
                assert_eq!(env.flags, flags);
            }
        }
    }

    #[test]
    fn nan_sign_is_preserved_when_propagated() {
        let negative_qnan = QNAN_SINGLE | SIGN_SINGLE;
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::Add, negative_qnan, ONE_SINGLE),
            negative_qnan
        );
    }

    #[test]
    fn square_root_of_a_nan_returns_the_quieted_nan() {
        let mut env = default_env();
        assert_eq!(
            single(&mut env, Arithmetic::SquareRoot, 0, SNAN_SINGLE),
            SNAN_SINGLE | QUIET_BIT_SINGLE
        );
        assert_eq!(env.flags, FLAG_INVALID);
    }

    #[test]
    fn nan_with_a_subnormal_operand_still_reports_the_denormal() {
        // Matches QEMU, which records the denormal operand whatever the other operand is.
        let mut env = default_env();
        single(&mut env, Arithmetic::Add, MIN_SUBNORMAL_SINGLE, QNAN_SINGLE);
        assert_eq!(env.flags, FLAG_DENORMAL);
    }

    #[test]
    fn sqrt_ignores_the_destination_operand() {
        // A garbage (subnormal, NaN) destination must neither raise flags nor change the result.
        let mut env = default_env();
        assert_eq!(
            arithmetic(
                &mut env,
                SINGLE,
                Arithmetic::SquareRoot,
                SNAN_SINGLE as u64,
                0x4080_0000
            ),
            TWO_SINGLE as u64
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn square_root_matches_exhaustive_perfect_squares() {
        for n in 0u32..=2000 {
            let square = (n * n) as f32;
            let mut env = default_env();
            let got = single(&mut env, Arithmetic::SquareRoot, 0, square.to_bits());
            assert_eq!(got, (n as f32).to_bits(), "{n}");
            assert_eq!(env.flags, 0, "{n}");
        }
    }

    #[test]
    fn integer_square_root_is_floor() {
        for n in [
            0u128,
            1,
            2,
            3,
            4,
            8,
            9,
            15,
            16,
            17,
            u64::MAX as u128,
            1 << 100,
            (1 << 100) - 1,
        ] {
            let root = integer_square_root(n);
            assert!(root * root <= n && (root + 1) * (root + 1) > n, "{n}");
        }
    }

    #[test]
    fn shift_right_sticky_keeps_dropped_bits_as_one_bit() {
        assert_eq!(shift_right_sticky(0b1000, 0), 0b1000);
        assert_eq!(shift_right_sticky(0b1000, 3), 1);
        assert_eq!(shift_right_sticky(0b10001, 3), 0b11);
        assert_eq!(shift_right_sticky(1, 1), 1);
        assert_eq!(shift_right_sticky(0, 200), 0);
        assert_eq!(shift_right_sticky(5, 128), 1);
        assert_eq!(shift_right_sticky(u128::MAX, 127), 1);
    }

    #[test]
    fn rounding_increment_by_mode() {
        // 0.25, 0.5, 0.75 of a unit, positive and negative, odd and even kept part
        let half = 8u128;
        for (rounding, sign, kept, remainder, expected) in [
            (RC_NEAREST, false, 0, 4, false),
            (RC_NEAREST, false, 0, 8, false),
            (RC_NEAREST, false, 1, 8, true),
            (RC_NEAREST, true, 1, 12, true),
            (RC_DOWN, false, 0, 4, false),
            (RC_DOWN, true, 0, 4, true),
            (RC_DOWN, true, 0, 0, false),
            (RC_UP, false, 0, 4, true),
            (RC_UP, true, 0, 4, false),
            (RC_UP, false, 0, 0, false),
            (RC_ZERO, false, 0, 15, false),
            (RC_ZERO, true, 0, 15, false),
        ] {
            assert_eq!(
                rounds_up(rounding, sign, kept, remainder, half),
                expected,
                "rc {rounding} sign {sign} kept {kept} rem {remainder}"
            );
        }
    }

    fn min_single(env: &mut FpEnv, a: u32, b: u32) -> u32 {
        minimum_maximum(env, SINGLE, false, a as u64, b as u64) as u32
    }
    fn max_single(env: &mut FpEnv, a: u32, b: u32) -> u32 {
        minimum_maximum(env, SINGLE, true, a as u64, b as u64) as u32
    }

    #[test]
    fn min_max_pick_by_numeric_order() {
        let mut env = default_env();
        let (two, three) = (TWO_SINGLE, THREE_SINGLE);
        assert_eq!(min_single(&mut env, two, three), two);
        assert_eq!(min_single(&mut env, three, two), two);
        assert_eq!(max_single(&mut env, two, three), three);
        assert_eq!(max_single(&mut env, three, two), three);
        let (minus_two, minus_three) = (0xc000_0000, 0xc040_0000);
        assert_eq!(min_single(&mut env, minus_two, minus_three), minus_three);
        assert_eq!(max_single(&mut env, minus_two, minus_three), minus_two);
        assert_eq!(min_single(&mut env, minus_two, two), minus_two);
        assert_eq!(max_single(&mut env, minus_two, two), two);
        assert_eq!(
            min_single(&mut env, INFINITY_SINGLE | SIGN_SINGLE, minus_two),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(
            max_single(&mut env, INFINITY_SINGLE, three),
            INFINITY_SINGLE
        );
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn min_max_of_equal_values_return_the_source_operand() {
        // 2.0 and 2.0 are indistinguishable, but +0 and -0 are not: equality returns the source.
        let mut env = default_env();
        assert_eq!(min_single(&mut env, 0, 1 << 31), 1 << 31);
        assert_eq!(min_single(&mut env, 1 << 31, 0), 0);
        assert_eq!(max_single(&mut env, 0, 1 << 31), 1 << 31);
        assert_eq!(max_single(&mut env, 1 << 31, 0), 0);
        assert_eq!(min_single(&mut env, 0, 0), 0);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn min_max_return_the_source_operand_unchanged_when_either_is_nan() {
        for op in [min_single, max_single] {
            let mut env = default_env();
            assert_eq!(op(&mut env, QNAN_SINGLE, ONE_SINGLE), ONE_SINGLE);
            assert_eq!(env.flags, FLAG_INVALID, "a QNaN raises invalid too");
            let mut env = default_env();
            assert_eq!(op(&mut env, ONE_SINGLE, QNAN_SINGLE), QNAN_SINGLE);
            assert_eq!(env.flags, FLAG_INVALID);
            // an SNaN source comes back as it is, not quieted
            let mut env = default_env();
            assert_eq!(op(&mut env, ONE_SINGLE, SNAN_SINGLE), SNAN_SINGLE);
            let mut env = default_env();
            assert_eq!(op(&mut env, SNAN_SINGLE, QNAN_SINGLE), QNAN_SINGLE);
            let mut env = default_env();
            assert_eq!(op(&mut env, QNAN_SINGLE, SNAN_SINGLE), SNAN_SINGLE);
        }
    }

    #[test]
    fn min_max_with_daz_return_a_zero_for_a_subnormal_operand() {
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(
            max_single(&mut env, MIN_SUBNORMAL_SINGLE, 0x8000_0000),
            0x8000_0000
        );
        assert_eq!(min_single(&mut env, ONE_SINGLE, 0x0000_0005), 0);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn min_max_flag_subnormal_operands_without_daz() {
        let mut env = default_env();
        assert_eq!(
            min_single(&mut env, MIN_SUBNORMAL_SINGLE, ONE_SINGLE),
            MIN_SUBNORMAL_SINGLE
        );
        assert_eq!(env.flags, FLAG_DENORMAL);
    }

    #[test]
    fn compare_orders_all_classes_of_values() {
        use core::cmp::Ordering::*;
        let values = [
            INFINITY_SINGLE | SIGN_SINGLE,
            MAX_SINGLE | SIGN_SINGLE,
            0xc000_0000,
            ONE_SINGLE | SIGN_SINGLE,
            0x8000_0001,
            0,
            MIN_SUBNORMAL_SINGLE,
            MIN_NORMAL_SINGLE,
            ONE_SINGLE,
            TWO_SINGLE,
            MAX_SINGLE,
            INFINITY_SINGLE,
        ];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                let mut env = default_env();
                let got = compare(&mut env, SINGLE, a as u64, b as u64, false);
                // index 4 (-subnormal) and 5 (+0)... the negative zero is not in the list, but the
                // subnormal sits strictly between -1 and +0
                assert_eq!(got, Some(i.cmp(&j)), "{a:#x} vs {b:#x}");
                let _ = (Less, Equal, Greater);
            }
        }
    }

    #[test]
    fn compare_treats_both_zeros_as_equal() {
        let mut env = default_env();
        for (a, b) in [(0, 1u64 << 31), (1 << 31, 0), (0, 0), (1 << 31, 1 << 31)] {
            assert_eq!(
                compare(&mut env, SINGLE, a, b, true),
                Some(core::cmp::Ordering::Equal)
            );
        }
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn compare_nan_is_unordered_and_raises_invalid_per_kind() {
        let mut env = default_env();
        assert_eq!(
            compare(&mut env, SINGLE, QNAN_SINGLE as u64, 0, false),
            None
        );
        assert_eq!(env.flags, 0);
        assert_eq!(compare(&mut env, SINGLE, 0, QNAN_SINGLE as u64, true), None);
        assert_eq!(env.flags, FLAG_INVALID);
        let mut env = default_env();
        assert_eq!(
            compare(&mut env, SINGLE, SNAN_SINGLE as u64, 0, false),
            None
        );
        assert_eq!(env.flags, FLAG_INVALID);
        let mut env = default_env();
        assert_eq!(compare(&mut env, DOUBLE, 0, SNAN_DOUBLE, false), None);
        assert_eq!(env.flags, FLAG_INVALID);
    }

    #[test]
    fn compare_eflags_encoding() {
        use core::cmp::Ordering::*;
        assert_eq!(compare_eflags(None), ZF | PF | CF);
        assert_eq!(compare_eflags(Some(Greater)), 0);
        assert_eq!(compare_eflags(Some(Less)), CF);
        assert_eq!(compare_eflags(Some(Equal)), ZF);
    }

    #[test]
    fn compare_predicates_truth_table() {
        // columns: less, equal, greater, unordered
        let table: [[bool; 4]; 8] = [
            [false, true, false, false], // eq
            [true, false, false, false], // lt
            [true, true, false, false],  // le
            [false, false, false, true], // unord
            [true, false, true, true],   // neq
            [false, true, true, true],   // nlt
            [false, false, true, true],  // nle
            [true, true, true, false],   // ord
        ];
        let operands: [(u64, u64); 4] = [
            (ONE_SINGLE as u64, TWO_SINGLE as u64),
            (ONE_SINGLE as u64, ONE_SINGLE as u64),
            (TWO_SINGLE as u64, ONE_SINGLE as u64),
            (QNAN_SINGLE as u64, ONE_SINGLE as u64),
        ];
        for (predicate, row) in table.iter().enumerate() {
            for (column, (a, b)) in operands.iter().enumerate() {
                let mut env = default_env();
                assert_eq!(
                    compare_predicate(&mut env, SINGLE, predicate as u8, *a, *b, false),
                    row[column],
                    "predicate {predicate} column {column}"
                );
            }
        }
    }

    #[test]
    fn compare_predicate_ignores_bits_above_the_low_three() {
        let mut env = default_env();
        assert!(compare_predicate(
            &mut env,
            SINGLE,
            8,
            ONE_SINGLE as u64,
            ONE_SINGLE as u64,
            false
        ));
        assert!(!compare_predicate(
            &mut env,
            SINGLE,
            0xf9,
            ONE_SINGLE as u64,
            ONE_SINGLE as u64,
            false
        ));
    }

    #[test]
    fn compare_predicate_invalid_flag_for_quiet_nan_only_on_signaling_predicates() {
        for predicate in 0u8..8 {
            let mut env = default_env();
            compare_predicate(
                &mut env,
                SINGLE,
                predicate,
                QNAN_SINGLE as u64,
                ONE_SINGLE as u64,
                false,
            );
            let expected = if matches!(predicate, 1 | 2 | 5 | 6) {
                FLAG_INVALID
            } else {
                0
            };
            assert_eq!(env.flags, expected, "predicate {predicate}");
            let mut env = default_env();
            compare_predicate(
                &mut env,
                SINGLE,
                predicate,
                ONE_SINGLE as u64,
                SNAN_SINGLE as u64,
                false,
            );
            assert_eq!(
                env.flags, FLAG_INVALID,
                "predicate {predicate} with an SNaN"
            );
        }
    }

    #[test]
    fn integer_to_float_matches_the_host_for_nearest_even() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        let mut values = vec![
            0i64,
            1,
            -1,
            i64::MAX,
            i64::MIN,
            1 << 53,
            (1 << 53) + 1,
            (1 << 24) + 1,
            -(1 << 24) - 1,
        ];
        for _ in 0..20_000 {
            let shift = rng.next() % 64;
            values.push((rng.next() as i64) >> shift);
        }
        for value in values {
            let mut env = default_env();
            assert_eq!(
                int_to_float(&mut env, SINGLE, value) as u32,
                (value as f32).to_bits(),
                "{value}"
            );
            assert_eq!(
                int_to_float(&mut env, DOUBLE, value),
                (value as f64).to_bits(),
                "{value}"
            );
        }
    }

    #[test]
    fn integer_to_float_rounds_once_for_64_bit_sources() {
        // 2^60 + 2^36 + 1: above the halfway point of the single ulp 2^37 only because of the +1.
        // Going through f64 would lose the +1 and tie-round down to 2^60.
        let value = 0x1000_0010_0000_0001i64;
        let mut env = default_env();
        assert_eq!(int_to_float(&mut env, SINGLE, value) as u32, 0x5d80_0001);
        assert_eq!(env.flags, FLAG_PRECISION);
        assert_eq!(
            ((value as f64) as f32).to_bits(),
            0x5d80_0000,
            "the double rounding this avoids"
        );
    }

    #[test]
    fn integer_to_float_directed_rounding() {
        // 16777217 = 2^24 + 1 is not a single
        for (rounding, positive, negative) in [
            (
                RC_NEAREST,
                TWO_POW_24_SINGLE,
                (TWO_POW_24_SINGLE | SIGN_SINGLE),
            ),
            (
                RC_DOWN,
                TWO_POW_24_SINGLE,
                (TWO_POW_24_NEXT_SINGLE | SIGN_SINGLE),
            ),
            (
                RC_UP,
                TWO_POW_24_NEXT_SINGLE,
                (TWO_POW_24_SINGLE | SIGN_SINGLE),
            ),
            (
                RC_ZERO,
                TWO_POW_24_SINGLE,
                (TWO_POW_24_SINGLE | SIGN_SINGLE),
            ),
        ] {
            let mut env = env_rc(rounding);
            assert_eq!(
                int_to_float(&mut env, SINGLE, FIRST_INEXACT_INTEGER as i64) as u32,
                positive,
                "rc {rounding}"
            );
            assert_eq!(
                int_to_float(&mut env, SINGLE, -(FIRST_INEXACT_INTEGER as i64)) as u32,
                negative,
                "rc {rounding}"
            );
        }
        let mut env = env_rc(RC_DOWN);
        assert_eq!(
            int_to_float(&mut env, DOUBLE, i64::MAX),
            0x43df_ffff_ffff_ffff
        );
        let mut env = env_rc(RC_UP);
        assert_eq!(
            int_to_float(&mut env, DOUBLE, i64::MAX),
            0x43e0_0000_0000_0000
        );
    }

    #[test]
    fn integer_to_float_exact_values_raise_no_flags() {
        for rounding in ALL_ROUNDING_MODES {
            let mut env = env_rc(rounding);
            assert_eq!(int_to_float(&mut env, SINGLE, 0), 0);
            assert_eq!(int_to_float(&mut env, SINGLE, -5) as u32, 0xc0a0_0000);
            assert_eq!(
                int_to_float(&mut env, DOUBLE, i64::MIN),
                0xc3e0_0000_0000_0000
            );
            assert_eq!(
                int_to_float(&mut env, SINGLE, 1 << 24) as u32,
                TWO_POW_24_SINGLE
            );
            assert_eq!(env.flags, 0);
        }
    }

    fn to_i32(env: &mut FpEnv, value: f64, truncate: bool) -> u64 {
        float_to_int(env, DOUBLE, value.to_bits(), 32, truncate)
    }
    fn to_i64(env: &mut FpEnv, value: f64, truncate: bool) -> u64 {
        float_to_int(env, DOUBLE, value.to_bits(), 64, truncate)
    }

    const INDEFINITE_32: u64 = 0x8000_0000;
    const INDEFINITE_64: u64 = 0x8000_0000_0000_0000;

    #[test]
    fn float_to_integer_rounding_table() {
        // value, then results for nearest-even, down, up, toward zero
        let table: [(f64, [i32; 4]); 12] = [
            (2.5, [2, 2, 3, 2]),
            (-2.5, [-2, -3, -2, -2]),
            (3.5, [4, 3, 4, 3]),
            (-3.5, [-4, -4, -3, -3]),
            (0.5, [0, 0, 1, 0]),
            (-0.5, [0, -1, 0, 0]),
            (1.5, [2, 1, 2, 1]),
            (0.4999999, [0, 0, 1, 0]),
            (-0.4999999, [0, -1, 0, 0]),
            (0.75, [1, 0, 1, 0]),
            (7.0, [7, 7, 7, 7]),
            (-7.0, [-7, -7, -7, -7]),
        ];
        for (value, results) in table {
            for (rounding, expected) in ALL_ROUNDING_MODES.iter().zip(results) {
                let mut env = env_rc(*rounding);
                assert_eq!(
                    to_i32(&mut env, value, false),
                    expected as u32 as u64,
                    "{value} rc {rounding}"
                );
                assert_eq!(
                    env.flags,
                    if value.fract() != 0.0 {
                        FLAG_PRECISION
                    } else {
                        0
                    },
                    "{value} rc {rounding}"
                );
                assert_eq!(
                    to_i64(&mut env_rc(*rounding), value, false),
                    expected as i64 as u64
                );
            }
            // the truncating forms ignore MXCSR.RC
            let mut env = env_rc(RC_UP);
            assert_eq!(
                to_i32(&mut env, value, true),
                results[3] as u32 as u64,
                "{value}"
            );
        }
    }

    #[test]
    fn float_to_integer_matches_the_host_rounding_for_random_values() {
        let mut rng = XorShift(0xabcd_ef01_2345_6789);
        for _ in 0..30_000 {
            let value = f64::from_bits(random_double(&mut rng));
            if value.is_nan() {
                continue;
            }
            let rounded = [
                value.round_ties_even(),
                value.floor(),
                value.ceil(),
                value.trunc(),
            ];
            for (rounding, r) in ALL_ROUNDING_MODES.iter().zip(rounded) {
                let expected32 = if (-2147483648.0..2147483648.0).contains(&r) {
                    r as i32 as u32 as u64
                } else {
                    INDEFINITE_32
                };
                let expected64 = if (-9223372036854775808.0..9223372036854775808.0).contains(&r) {
                    r as i64 as u64
                } else {
                    INDEFINITE_64
                };
                assert_eq!(
                    to_i32(&mut env_rc(*rounding), value, false),
                    expected32,
                    "{value:e} rc {rounding}"
                );
                assert_eq!(
                    to_i64(&mut env_rc(*rounding), value, false),
                    expected64,
                    "{value:e} rc {rounding}"
                );
            }
            let truncated = value.trunc();
            let expected = if (-2147483648.0..2147483648.0).contains(&truncated) {
                truncated as i32 as u32 as u64
            } else {
                INDEFINITE_32
            };
            assert_eq!(
                to_i32(&mut env_rc(RC_UP), value, true),
                expected,
                "{value:e}"
            );
        }
    }

    #[test]
    fn float_to_integer_boundaries_32() {
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, 2147483647.0, false), 0x7fff_ffff);
        assert_eq!(
            to_i32(&mut env, -2147483648.0, false),
            INDEFINITE_32,
            "the minimum itself is valid and equals the indefinite value"
        );
        assert_eq!(env.flags, 0);
        // 2^31 - 0.5 rounds to even, which is 2^31: out of range
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, 2147483647.5, false), INDEFINITE_32);
        assert_eq!(env.flags, FLAG_INVALID, "no precision flag when invalid");
        // truncation keeps it in range
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, 2147483647.5, true), 0x7fff_ffff);
        assert_eq!(env.flags, FLAG_PRECISION);
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, 2147483648.0, false), INDEFINITE_32);
        assert_eq!(env.flags, FLAG_INVALID);
        // -2^31 - 0.5 rounds to -2^31 (even): valid but inexact; rounding down gives -2^31 - 1
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, -2147483648.5, false), INDEFINITE_32);
        assert_eq!(env.flags, FLAG_PRECISION);
        let mut env = env_rc(RC_DOWN);
        assert_eq!(to_i32(&mut env, -2147483648.5, false), INDEFINITE_32);
        assert_eq!(env.flags, FLAG_INVALID);
        let mut env = default_env();
        assert_eq!(to_i32(&mut env, -2147483647.0, false), 0x8000_0001);
        assert_eq!(to_i32(&mut env, -2147483648.9, true), 0x8000_0000);
    }

    #[test]
    fn float_to_integer_boundaries_64() {
        let mut env = default_env();
        assert_eq!(
            to_i64(&mut env, 9223372036854774784.0, false),
            0x7fff_ffff_ffff_fc00
        );
        assert_eq!(env.flags, 0);
        assert_eq!(
            to_i64(&mut env, 9223372036854775808.0, false),
            INDEFINITE_64
        );
        assert_eq!(env.flags, FLAG_INVALID);
        let mut env = default_env();
        assert_eq!(
            to_i64(&mut env, -9223372036854775808.0, false),
            INDEFINITE_64
        );
        assert_eq!(env.flags, 0, "the minimum is a valid result");
        assert_eq!(to_i64(&mut env, -1.0, false), u64::MAX);
        // 2^64 and above never fit
        let mut env = default_env();
        assert_eq!(to_i64(&mut env, 1.8446744073709552e19, true), INDEFINITE_64);
        assert_eq!(to_i64(&mut env, -1e300, false), INDEFINITE_64);
        assert_eq!(env.flags, FLAG_INVALID);
    }

    #[test]
    fn float_to_integer_special_values_give_the_indefinite_and_invalid() {
        for bits in [
            QNAN_DOUBLE,
            SNAN_DOUBLE,
            DEFAULT_NAN_DOUBLE,
            INFINITY_DOUBLE,
            INFINITY_DOUBLE | SIGN_DOUBLE,
        ] {
            for width in [32, 64] {
                for truncate in [false, true] {
                    let mut env = default_env();
                    assert_eq!(
                        float_to_int(&mut env, DOUBLE, bits, width, truncate),
                        1u64 << (width - 1)
                    );
                    assert_eq!(env.flags, FLAG_INVALID);
                }
            }
        }
        let mut env = default_env();
        assert_eq!(
            float_to_int(&mut env, SINGLE, QNAN_SINGLE as u64, 32, false),
            INDEFINITE_32
        );
    }

    #[test]
    fn float_to_integer_zero_is_exact() {
        for bits in [0u64, 1 << 63] {
            let mut env = default_env();
            assert_eq!(float_to_int(&mut env, DOUBLE, bits, 32, false), 0);
            assert_eq!(float_to_int(&mut env, DOUBLE, bits, 64, true), 0);
            assert_eq!(env.flags, 0);
        }
    }

    #[test]
    fn float_to_integer_subnormal_is_inexact_zero_or_one() {
        let mut env = default_env();
        assert_eq!(float_to_int(&mut env, SINGLE, 1, 32, false), 0);
        assert_eq!(env.flags, FLAG_DENORMAL | FLAG_PRECISION);
        let mut env = env_rc(RC_UP);
        assert_eq!(float_to_int(&mut env, SINGLE, 1, 32, false), 1);
        let mut env = env_rc(RC_DOWN);
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0x8000_0001, 32, false),
            0xffff_ffff
        );
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(float_to_int(&mut env, SINGLE, 1, 32, false), 0);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn float_to_integer_single_source() {
        let mut env = default_env();
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0x4b7f_ffff, 32, false),
            16777215
        );
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0xcf00_0000, 32, false),
            INDEFINITE_32
        );
        assert_eq!(env.flags, 0);
        // 2^31 as a single is out of range for 32 bits but fine for 64
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0x4f00_0000, 32, false),
            INDEFINITE_32
        );
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0x4f00_0000, 64, false),
            1 << 31
        );
        assert_eq!(
            float_to_int(&mut env, SINGLE, 0x5f00_0000, 64, false),
            INDEFINITE_64
        );
    }

    #[test]
    fn widening_single_to_double_is_exact_and_matches_the_host() {
        let mut rng = XorShift(0x5151_5151_a1a1_a1a1);
        for _ in 0..20_000 {
            let bits = random_single(&mut rng);
            let value = f32::from_bits(bits);
            if value.is_nan() {
                continue;
            }
            let mut env = default_env();
            assert_eq!(
                convert_float(&mut env, SINGLE, DOUBLE, bits as u64),
                (value as f64).to_bits(),
                "{bits:#x}"
            );
            assert_eq!(env.flags & !FLAG_DENORMAL, 0);
        }
    }

    #[test]
    fn narrowing_double_to_single_matches_the_host_for_nearest_even() {
        let mut rng = XorShift(0x2468_ace0_1357_9bdf);
        for _ in 0..30_000 {
            let bits = random_double(&mut rng);
            let value = f64::from_bits(bits);
            if value.is_nan() {
                continue;
            }
            let mut env = default_env();
            assert_eq!(
                convert_float(&mut env, DOUBLE, SINGLE, bits) as u32,
                (value as f32).to_bits(),
                "{bits:#x}"
            );
        }
    }

    #[test]
    fn narrowing_directed_rounding_brackets_the_value() {
        let mut rng = XorShift(0x1357_9bdf_2468_ace1);
        for _ in 0..20_000 {
            let bits = random_double(&mut rng);
            let value = f64::from_bits(bits);
            if value.is_nan() || !value.is_finite() {
                continue;
            }
            let nearest = value as f32;
            let (down, up) = {
                let mut env = env_rc(RC_DOWN);
                let down = f32::from_bits(convert_float(&mut env, DOUBLE, SINGLE, bits) as u32);
                let mut env = env_rc(RC_UP);
                let up = f32::from_bits(convert_float(&mut env, DOUBLE, SINGLE, bits) as u32);
                (down, up)
            };
            // the exact value is between them (infinity counts as the bound past MAX)
            assert!((down as f64) <= value && value <= (up as f64), "{value:e}");
            assert!(nearest == down || nearest == up, "{value:e}");
            assert!(down == up || down.next_up() == up, "{value:e}: adjacent");
            assert_eq!(
                down == up,
                (nearest as f64) == value,
                "{value:e}: exact iff equal"
            );
            let mut env = env_rc(RC_ZERO);
            let toward_zero = f32::from_bits(convert_float(&mut env, DOUBLE, SINGLE, bits) as u32);
            assert_eq!(
                toward_zero,
                if value > 0.0 { down } else { up },
                "{value:e}"
            );
        }
    }

    #[test]
    fn narrowing_overflow_and_underflow() {
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, 1e300f64.to_bits()) as u32,
            INFINITY_SINGLE
        );
        assert_eq!(env.flags, FLAG_OVERFLOW | FLAG_PRECISION);
        let mut env = env_rc(RC_ZERO);
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, 1e300f64.to_bits()) as u32,
            MAX_SINGLE
        );
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, 1e-60f64.to_bits()) as u32,
            0
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION);
        // 2^-140 is a single subnormal: exact, no flags
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, (2.0f64).powi(-140).to_bits()) as u32,
            1 << 9
        );
        assert_eq!(env.flags, 0);
        // FTZ flushes that exact subnormal
        let mut env = env_with(MXCSR_FTZ);
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, (2.0f64).powi(-140).to_bits()) as u32,
            0
        );
        assert_eq!(env.flags, FLAG_UNDERFLOW | FLAG_PRECISION);
    }

    #[test]
    fn narrowing_overflow_boundary_is_halfway_above_max() {
        // MAX + half ulp = 2^128 - 2^103 rounds to even = 2^128 (overflow); one double below stays.
        let halfway = f64::from_bits(0x47ef_ffff_f000_0000);
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, halfway.to_bits()) as u32,
            INFINITY_SINGLE
        );
        let below = halfway.next_down();
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, below.to_bits()) as u32,
            MAX_SINGLE
        );
        assert_eq!(env.flags, FLAG_PRECISION);
    }

    #[test]
    fn convert_float_nan_payloads_shift_with_the_format() {
        let mut env = default_env();
        // single QNaN with payload -> double: fraction bits move up by 29
        assert_eq!(
            convert_float(&mut env, SINGLE, DOUBLE, 0x7fc1_2345),
            0x7ff0_0000_0000_0000 | (0x41_2345u64 << 29)
        );
        assert_eq!(env.flags, 0);
        // negative sign survives
        assert_eq!(
            convert_float(&mut env, SINGLE, DOUBLE, 0xffc0_0000),
            0xfff8_0000_0000_0000
        );
        // SNaN single -> quiet double, invalid
        assert_eq!(
            convert_float(&mut env, SINGLE, DOUBLE, SNAN_SINGLE as u64),
            0x7ff8_0000_2000_0000
        );
        assert_eq!(env.flags, FLAG_INVALID);
        // double -> single drops the low 29 fraction bits and sets the quiet bit
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, 0x7ff8_2468_a000_0000) as u32,
            0x7fc1_2345
        );
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, SNAN_DOUBLE) as u32,
            0x7fc0_0000
        );
        assert_eq!(env.flags, FLAG_INVALID);
    }

    #[test]
    fn convert_float_specials() {
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, DOUBLE, SINGLE, INFINITY_DOUBLE | SIGN_DOUBLE) as u32,
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(convert_float(&mut env, SINGLE, DOUBLE, 1 << 31), 1 << 63);
        assert_eq!(convert_float(&mut env, DOUBLE, SINGLE, 0) as u32, 0);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn widening_a_subnormal_single_gives_a_normal_double() {
        let mut env = default_env();
        assert_eq!(
            convert_float(&mut env, SINGLE, DOUBLE, 1),
            0x36a0_0000_0000_0000
        );
        assert_eq!(env.flags, FLAG_DENORMAL);
        let mut env = env_with(MXCSR_DAZ);
        assert_eq!(convert_float(&mut env, SINGLE, DOUBLE, 1), 0);
        assert_eq!(env.flags, 0);
    }

    fn reciprocal_single(bits: u32) -> u32 {
        reciprocal(SINGLE, bits as u64, false) as u32
    }
    fn reciprocal_root_single(bits: u32) -> u32 {
        reciprocal(SINGLE, bits as u64, true) as u32
    }

    #[test]
    fn reciprocal_of_ordinary_values() {
        assert_eq!(reciprocal_single(TWO_SINGLE), HALF_SINGLE);
        assert_eq!(reciprocal_single(0xc080_0000), 0xbe80_0000);
        assert_eq!(reciprocal_single(THREE_SINGLE), ONE_THIRD_SINGLE_UP);
    }

    #[test]
    fn reciprocal_of_zero_and_infinity() {
        assert_eq!(reciprocal_single(0), INFINITY_SINGLE);
        assert_eq!(
            reciprocal_single(SIGN_SINGLE),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(reciprocal_single(INFINITY_SINGLE), 0);
        assert_eq!(reciprocal_single(INFINITY_SINGLE | SIGN_SINGLE), 1 << 31);
    }

    #[test]
    fn reciprocal_treats_subnormal_inputs_as_zero() {
        assert_eq!(reciprocal_single(MIN_SUBNORMAL_SINGLE), INFINITY_SINGLE);
        assert_eq!(
            reciprocal_single(0x8000_0001),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(
            reciprocal_root_single(MIN_SUBNORMAL_SINGLE),
            INFINITY_SINGLE
        );
    }

    #[test]
    fn reciprocal_flushes_tiny_results_to_signed_zero() {
        // 1 / 2^127 = 2^-127 would be subnormal
        assert_eq!(reciprocal_single(0x7f00_0000), 0);
        assert_eq!(reciprocal_single(0xff00_0000), 1 << 31);
    }

    #[test]
    fn reciprocal_square_root_cases() {
        assert_eq!(reciprocal_root_single(0x4080_0000), HALF_SINGLE);
        assert_eq!(reciprocal_root_single(ONE_SINGLE), ONE_SINGLE);
        assert_eq!(reciprocal_root_single(0), INFINITY_SINGLE);
        assert_eq!(
            reciprocal_root_single(SIGN_SINGLE),
            INFINITY_SINGLE | SIGN_SINGLE
        );
        assert_eq!(reciprocal_root_single(INFINITY_SINGLE), 0);
        assert_eq!(reciprocal_root_single(0xbf80_0000), DEFAULT_NAN_SINGLE);
    }

    #[test]
    fn reciprocal_of_nan_returns_the_quieted_nan() {
        assert_eq!(reciprocal_single(QNAN_SINGLE), QNAN_SINGLE);
        assert_eq!(
            reciprocal_single(SNAN_SINGLE),
            SNAN_SINGLE | QUIET_BIT_SINGLE
        );
        assert_eq!(reciprocal_root_single(QNAN_SINGLE), QNAN_SINGLE);
    }

    #[test]
    fn reciprocal_is_within_hardware_error_of_the_exact_value() {
        // Hardware guarantees a relative error under 1.5 * 2^-12; the exact result trivially is.
        let mut rng = XorShift(0x1111_2222_3333_4444);
        for _ in 0..5_000 {
            let bits = random_single(&mut rng);
            let value = f32::from_bits(bits);
            if !value.is_normal() || value.abs() > 1e30 || value.abs() < 1e-30 {
                continue;
            }
            let got = f32::from_bits(reciprocal_single(bits));
            let relative = ((got as f64) * (value as f64) - 1.0).abs();
            assert!(relative < 1.5 / 4096.0, "{value:e}");
        }
    }

    #[test]
    fn format_constants() {
        assert_eq!(SINGLE.width(), 32);
        assert_eq!(DOUBLE.width(), 64);
        assert_eq!(SINGLE.lanes(), 4);
        assert_eq!(DOUBLE.lanes(), 2);
        assert_eq!(SINGLE.bytes(), 4);
        assert_eq!(DOUBLE.bytes(), 8);
        assert_eq!(SINGLE.bias(), 127);
        assert_eq!(DOUBLE.bias(), 1023);
        assert_eq!(SINGLE.indefinite() as u32, DEFAULT_NAN_SINGLE);
        assert_eq!(DOUBLE.indefinite(), DEFAULT_NAN_DOUBLE);
        assert_eq!(SINGLE.max_finite(false) as u32, MAX_SINGLE);
        assert_eq!(DOUBLE.max_finite(true), 0xffef_ffff_ffff_ffff);
        assert_eq!(SINGLE.infinity(false) as u32, INFINITY_SINGLE);
        assert_eq!(DOUBLE.zero(true), 1 << 63);
        assert_eq!(SINGLE.lane_mask(), 0xffff_ffff);
        assert_eq!(DOUBLE.lane_mask(), u64::MAX as u128);
    }

    #[test]
    fn lane_helpers_round_trip() {
        let v = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        let mut rebuilt = 0;
        for k in 0..4 {
            lane_put(&mut rebuilt, SINGLE, k, lane_get(v, SINGLE, k));
        }
        assert_eq!(rebuilt, v);
        assert_eq!(lane_get(v, DOUBLE, 1), 0x0123_4567_89ab_cdef);
        assert_eq!(lane_get(v, DOUBLE, 0), 0xfedc_ba98_7654_3210);
        assert_eq!(lane_get(v, SINGLE, 3), 0x0123_4567);
    }

    const ARITHMETIC_MNEMONICS: [(&str, u8); 7] = [
        ("add", 0x58),
        ("sub", 0x5c),
        ("mul", 0x59),
        ("div", 0x5e),
        ("sqrt", 0x51),
        ("min", 0x5d),
        ("max", 0x5f),
    ];

    fn host_lane_single(mnemonic: &str, a: f32, b: f32) -> f32 {
        match mnemonic {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => a / b,
            "sqrt" => b.sqrt(),
            "min" => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            _ => {
                if a > b {
                    a
                } else {
                    b
                }
            }
        }
    }

    fn host_lane_double(mnemonic: &str, a: f64, b: f64) -> f64 {
        match mnemonic {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => a / b,
            "sqrt" => b.sqrt(),
            "min" => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            _ => {
                if a > b {
                    a
                } else {
                    b
                }
            }
        }
    }

    const DEST_SINGLES: [f32; 4] = [1.0, 2.0, -3.0, 4.5];
    const SOURCE_SINGLES: [f32; 4] = [0.25, 4.0, 2.0, 9.0];
    const DEST_DOUBLES: [f64; 2] = [1.5, -2.0];
    const SOURCE_DOUBLES: [f64; 2] = [0.25, 8.0];

    #[test]
    fn packed_single_arithmetic_computes_every_lane() {
        let mut rig = Rig::new();
        for (name, _) in ARITHMETIC_MNEMONICS {
            let expected: [f32; 4] =
                std::array::from_fn(|k| host_lane_single(name, DEST_SINGLES[k], SOURCE_SINGLES[k]));
            let register_form = rig.op(
                &format!("{name}ps xmm0, xmm1"),
                singles(DEST_SINGLES),
                singles(SOURCE_SINGLES),
            );
            assert_eq!(register_form, singles(expected), "{name}ps");
            assert_eq!(
                rig.xmm(1),
                singles(SOURCE_SINGLES),
                "the source is untouched"
            );
            let memory_form = rig.op_mem(
                &format!("{name}ps xmm0, xmmword ptr [rsi]"),
                singles(DEST_SINGLES),
                singles(SOURCE_SINGLES),
            );
            assert_eq!(memory_form, singles(expected), "{name}ps memory");
        }
    }

    #[test]
    fn packed_double_arithmetic_computes_both_lanes() {
        let mut rig = Rig::new();
        for (name, _) in ARITHMETIC_MNEMONICS {
            let expected: [f64; 2] =
                std::array::from_fn(|k| host_lane_double(name, DEST_DOUBLES[k], SOURCE_DOUBLES[k]));
            let register_form = rig.op(
                &format!("{name}pd xmm0, xmm1"),
                doubles(DEST_DOUBLES),
                doubles(SOURCE_DOUBLES),
            );
            assert_eq!(register_form, doubles(expected), "{name}pd");
            let memory_form = rig.op_mem(
                &format!("{name}pd xmm0, xmmword ptr [rsi]"),
                doubles(DEST_DOUBLES),
                doubles(SOURCE_DOUBLES),
            );
            assert_eq!(memory_form, doubles(expected), "{name}pd memory");
        }
    }

    #[test]
    fn scalar_single_arithmetic_changes_only_the_low_lane() {
        let mut rig = Rig::new();
        for (name, _) in ARITHMETIC_MNEMONICS {
            let expected = host_lane_single(name, DEST_SINGLES[0], SOURCE_SINGLES[0]);
            let mut want = DEST_SINGLES;
            want[0] = expected;
            let got = rig.op(
                &format!("{name}ss xmm0, xmm1"),
                singles(DEST_SINGLES),
                singles(SOURCE_SINGLES),
            );
            assert_eq!(got, singles(want), "{name}ss");
            let got = rig.op_mem(
                &format!("{name}ss xmm0, dword ptr [rsi]"),
                singles(DEST_SINGLES),
                singles(SOURCE_SINGLES),
            );
            assert_eq!(got, singles(want), "{name}ss memory");
        }
    }

    #[test]
    fn scalar_double_arithmetic_changes_only_the_low_lane() {
        let mut rig = Rig::new();
        for (name, _) in ARITHMETIC_MNEMONICS {
            let expected = host_lane_double(name, DEST_DOUBLES[0], SOURCE_DOUBLES[0]);
            let want = [expected, DEST_DOUBLES[1]];
            let got = rig.op(
                &format!("{name}sd xmm0, xmm1"),
                doubles(DEST_DOUBLES),
                doubles(SOURCE_DOUBLES),
            );
            assert_eq!(got, doubles(want), "{name}sd");
            let got = rig.op_mem(
                &format!("{name}sd xmm0, qword ptr [rsi]"),
                doubles(DEST_DOUBLES),
                doubles(SOURCE_DOUBLES),
            );
            assert_eq!(got, doubles(want), "{name}sd memory");
        }
    }

    #[test]
    fn scalar_arithmetic_reads_only_the_low_lane_of_the_source() {
        let mut rig = Rig::new();
        let source = lanes32([ONE_SINGLE, QNAN_SINGLE, SNAN_SINGLE, INFINITY_SINGLE]);
        let got = rig.op("addss xmm0, xmm1", singles([2.0, 0.0, 0.0, 0.0]), source);
        assert_eq!(dwords(got)[0], THREE_SINGLE);
        assert_eq!(rig.mxcsr(), MXCSR_RESET, "upper source lanes raise nothing");
    }

    #[test]
    fn arithmetic_with_the_same_register_for_both_operands() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[3] = singles([2.0, 3.0, 4.0, 5.0]);
        rig.run_ok("sqrtsd xmm3, xmm3");
        assert_eq!(
            dwords(rig.xmm(3))[2..],
            dwords(singles([2.0, 3.0, 4.0, 5.0]))[2..]
        );
        let low = f64::from_bits(qwords(rig.xmm(3))[0]);
        let original = f64::from_bits(qwords(singles([2.0, 3.0, 4.0, 5.0]))[0]);
        assert_eq!(low, original.sqrt());
    }

    #[test]
    fn arithmetic_on_extended_registers() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[9] = singles([1.0, 2.0, 3.0, 4.0]);
        rig.h.cpu.xmm[10] = singles([10.0, 20.0, 30.0, 40.0]);
        rig.run_ok("addps xmm9, xmm10");
        assert_eq!(rig.xmm(9), singles([11.0, 22.0, 33.0, 44.0]));
        assert_eq!(rig.xmm(10), singles([10.0, 20.0, 30.0, 40.0]));
        rig.h.cpu.xmm[15] = doubles([1.0, 2.0]);
        rig.h.cpu.xmm[8] = doubles([0.5, 100.0]);
        rig.run_ok("addsd xmm15, xmm8");
        assert_eq!(rig.xmm(15), doubles([1.5, 2.0]));
    }

    #[test]
    fn arithmetic_rounds_per_mxcsr_rounding_control() {
        let mut rig = Rig::new();
        let third_inputs = (singles([1.0, 1.0, 1.0, 1.0]), singles([3.0, 3.0, 3.0, 3.0]));
        for (rounding, expected) in [
            (RC_NEAREST, ONE_THIRD_SINGLE_UP),
            (RC_DOWN, ONE_THIRD_SINGLE_DOWN),
            (RC_UP, ONE_THIRD_SINGLE_UP),
            (RC_ZERO, ONE_THIRD_SINGLE_DOWN),
        ] {
            rig.h.cpu.xmm[0] = third_inputs.0;
            rig.h.cpu.xmm[1] = third_inputs.1;
            rig.reset();
            rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
            assert_eq!(rig.exec("divps xmm0, xmm1"), Stop::Halted);
            assert_eq!(dwords(rig.xmm(0)), [expected; 4], "rc {rounding}");
            assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_PRECISION);
        }
    }

    #[test]
    fn arithmetic_results_with_nans_follow_the_x86_rules() {
        let mut rig = Rig::new();
        let negative_infinity = INFINITY_SINGLE | SIGN_SINGLE;
        let dst = lanes32([QNAN_SINGLE, ONE_SINGLE, INFINITY_SINGLE, 0]);
        let src = lanes32([0x7fc0_0002, SNAN_SINGLE, negative_infinity, INFINITY_SINGLE]);
        // lane 0: dest QNaN wins; lane 1: SNaN source quieted; lane 2: inf + -inf = default NaN;
        // lane 3: 0 + inf = inf
        let got = rig.op("addps xmm0, xmm1", dst, src);
        assert_eq!(
            dwords(got),
            [
                QNAN_SINGLE,
                SNAN_SINGLE | QUIET_BIT_SINGLE,
                DEFAULT_NAN_SINGLE,
                INFINITY_SINGLE
            ]
        );
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
    }

    #[test]
    fn default_nan_is_negative_for_double_precision_as_well() {
        let mut rig = Rig::new();
        let got = rig.op(
            "mulpd xmm0, xmm1",
            doubles([0.0, 1.0]),
            lanes64([INFINITY_DOUBLE, ONE_DOUBLE]),
        );
        assert_eq!(qwords(got), [DEFAULT_NAN_DOUBLE, ONE_DOUBLE]);
        let got = rig.op("sqrtsd xmm0, xmm1", 0, doubles([-4.0, 0.0]));
        assert_eq!(qwords(got)[0], DEFAULT_NAN_DOUBLE);
    }

    #[test]
    fn min_max_instructions_return_the_second_operand_for_nan_and_equal_zeros() {
        let mut rig = Rig::new();
        let negative_zero = 1u32 << 31;
        let dst = lanes32([QNAN_SINGLE, ONE_SINGLE, 0, negative_zero]);
        let src = lanes32([ONE_SINGLE, QNAN_SINGLE, negative_zero, 0]);
        for asm in ["minps xmm0, xmm1", "maxps xmm0, xmm1"] {
            let got = rig.op(asm, dst, src);
            assert_eq!(
                dwords(got),
                [ONE_SINGLE, QNAN_SINGLE, negative_zero, 0],
                "{asm}"
            );
            assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID, "{asm}");
        }
    }

    #[test]
    fn min_max_ordinary_values_instruction() {
        let mut rig = Rig::new();
        let dst = singles([1.0, -2.0, 5.0, 0.0]);
        let src = singles([2.0, -3.0, 5.0, -1.0]);
        assert_eq!(
            rig.op("minps xmm0, xmm1", dst, src),
            singles([1.0, -3.0, 5.0, -1.0])
        );
        assert_eq!(
            rig.op("maxps xmm0, xmm1", dst, src),
            singles([2.0, -2.0, 5.0, 0.0])
        );
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    #[test]
    fn scalar_min_max_with_an_snan_source_returns_it_unquieted() {
        let mut rig = Rig::new();
        let got = rig.op(
            "minss xmm0, xmm1",
            singles([1.0, 2.0, 3.0, 4.0]),
            lanes32([SNAN_SINGLE, 0, 0, 0]),
        );
        assert_eq!(
            dwords(got),
            [SNAN_SINGLE, TWO_SINGLE, THREE_SINGLE, 0x4080_0000]
        );
    }

    fn expect_flag(rig: &Rig, flag: u32) {
        assert_eq!(
            rig.mxcsr() & MXCSR_EXCEPTION_FLAGS,
            flag,
            "mxcsr {:#x}",
            rig.mxcsr()
        );
    }

    #[test]
    fn inexact_result_sets_the_sticky_precision_flag() {
        let mut rig = Rig::new();
        rig.op(
            "addss xmm0, xmm1",
            singles([1.0; 4]),
            lanes32([TWO_POW_MINUS_24_SINGLE, 0, 0, 0]),
        );
        expect_flag(&rig, FLAG_PRECISION);
    }

    #[test]
    fn exception_flags_are_sticky_across_instructions() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([1.0; 4]);
        rig.h.cpu.xmm[1] = singles([0.0; 4]);
        rig.run_ok("divss xmm0, xmm1");
        expect_flag(&rig, FLAG_DIVIDE_BY_ZERO);
        // A later instruction raising nothing leaves the flag; `exec` keeps MXCSR as it is.
        rig.h.cpu.xmm[0] = singles([1.0; 4]);
        rig.h.cpu.xmm[1] = singles([1.0; 4]);
        assert_eq!(rig.exec("addss xmm0, xmm1"), Stop::Halted);
        expect_flag(&rig, FLAG_DIVIDE_BY_ZERO);
        // and a different exception accumulates
        rig.h.cpu.xmm[1] = lanes32([TWO_POW_MINUS_24_SINGLE, 0, 0, 0]);
        assert_eq!(rig.exec("addss xmm0, xmm1"), Stop::Halted);
        expect_flag(&rig, FLAG_DIVIDE_BY_ZERO | FLAG_PRECISION);
    }

    #[test]
    fn flags_of_all_lanes_are_merged() {
        let mut rig = Rig::new();
        let dst = singles([1.0, 0.0, 1.0, MAX_SINGLE as f32 * 0.0 + 1.0]);
        let src = lanes32([THREE_SINGLE, 0, TWO_POW_MINUS_24_SINGLE, ONE_SINGLE]);
        // lane 0: 1/3 inexact; lane 1: 0/0 invalid; lane 2: 1/2^-24 exact; lane 3: 1/1 exact
        rig.op("divps xmm0, xmm1", dst, src);
        expect_flag(&rig, FLAG_PRECISION | FLAG_INVALID);
    }

    #[test]
    fn masked_exceptions_deliver_the_default_results() {
        let mut rig = Rig::new();
        let negative_one = (-1.0f32).to_bits();
        // masked divide by zero: infinity
        let got = rig.op(
            "divps xmm0, xmm1",
            singles([1.0, -1.0, 0.0, 1.0]),
            singles([0.0, 0.0, 0.0, 1.0]),
        );
        assert_eq!(
            dwords(got),
            [
                INFINITY_SINGLE,
                INFINITY_SINGLE | SIGN_SINGLE,
                DEFAULT_NAN_SINGLE,
                ONE_SINGLE
            ]
        );
        expect_flag(&rig, FLAG_DIVIDE_BY_ZERO | FLAG_INVALID);
        // masked overflow
        let got = rig.op(
            "mulss xmm0, xmm1",
            lanes32([MAX_SINGLE, 7, 8, 9]),
            singles([2.0, 0.0, 0.0, 0.0]),
        );
        assert_eq!(dwords(got), [INFINITY_SINGLE, 7, 8, 9]);
        expect_flag(&rig, FLAG_OVERFLOW | FLAG_PRECISION);
        // masked invalid
        let got = rig.op("sqrtss xmm0, xmm1", 0, lanes32([negative_one, 0, 0, 0]));
        assert_eq!(dwords(got)[0], DEFAULT_NAN_SINGLE);
        expect_flag(&rig, FLAG_INVALID);
    }

    #[test]
    fn denormal_operand_flag_and_daz_through_the_instruction() {
        let mut rig = Rig::new();
        rig.op(
            "addss xmm0, xmm1",
            lanes32([MIN_SUBNORMAL_SINGLE, 0, 0, 0]),
            singles([1.0, 0.0, 0.0, 0.0]),
        );
        expect_flag(&rig, FLAG_DENORMAL | FLAG_PRECISION);
        rig.h.cpu.xmm[0] = lanes32([MIN_SUBNORMAL_SINGLE, 0, 0, 0]);
        rig.h.cpu.xmm[1] = singles([1.0, 0.0, 0.0, 0.0]);
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | MXCSR_DAZ;
        assert_eq!(rig.exec("addss xmm0, xmm1"), Stop::Halted);
        assert_eq!(
            rig.mxcsr(),
            MXCSR_RESET | MXCSR_DAZ,
            "DAZ hides the denormal operand"
        );
        assert_eq!(dwords(rig.xmm(0))[0], ONE_SINGLE);
    }

    #[test]
    fn daz_makes_a_subnormal_multiplier_a_zero_through_the_instruction() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([5.0; 4]);
        rig.h.cpu.xmm[1] = lanes32([MIN_SUBNORMAL_SINGLE, 0x8000_0001, 0x0000_ffff, 0x007f_ffff]);
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | MXCSR_DAZ;
        assert_eq!(rig.exec("mulps xmm0, xmm1"), Stop::Halted);
        assert_eq!(dwords(rig.xmm(0)), [0, 0x8000_0000, 0, 0]);
    }

    #[test]
    fn ftz_flushes_a_tiny_result_through_the_instruction() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([f32::from_bits(MIN_NORMAL_SINGLE); 4]);
        rig.h.cpu.xmm[1] = singles([0.5; 4]);
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | MXCSR_FTZ;
        assert_eq!(rig.exec("mulps xmm0, xmm1"), Stop::Halted);
        assert_eq!(dwords(rig.xmm(0)), [0; 4]);
        assert_eq!(
            rig.mxcsr() & MXCSR_EXCEPTION_FLAGS,
            FLAG_UNDERFLOW | FLAG_PRECISION
        );
        // without FTZ the exact subnormal survives and nothing is flagged
        rig.h.cpu.xmm[0] = singles([f32::from_bits(MIN_NORMAL_SINGLE); 4]);
        rig.h.cpu.xmm[1] = singles([0.5; 4]);
        rig.run_ok("mulps xmm0, xmm1");
        assert_eq!(dwords(rig.xmm(0)), [QUIET_BIT_SINGLE; 4]);
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    /// An instruction and operands that raise exactly one exception, plus its flag and mask bit.
    struct Trap {
        name: &'static str,
        asm: &'static str,
        dest: u128,
        source: u128,
        flag: u32,
    }

    fn traps() -> Vec<Trap> {
        vec![
            Trap {
                name: "invalid",
                asm: "sqrtss xmm0, xmm1",
                dest: 0,
                source: singles([-1.0, 0.0, 0.0, 0.0]),
                flag: FLAG_INVALID,
            },
            Trap {
                name: "denormal",
                asm: "addss xmm0, xmm1",
                dest: lanes32([MIN_SUBNORMAL_SINGLE, 0, 0, 0]),
                source: 0,
                flag: FLAG_DENORMAL,
            },
            Trap {
                name: "divide by zero",
                asm: "divss xmm0, xmm1",
                dest: singles([1.0, 0.0, 0.0, 0.0]),
                source: 0,
                flag: FLAG_DIVIDE_BY_ZERO,
            },
            Trap {
                name: "overflow",
                asm: "mulss xmm0, xmm1",
                dest: lanes32([MAX_SINGLE, 0, 0, 0]),
                source: singles([2.0, 0.0, 0.0, 0.0]),
                flag: FLAG_OVERFLOW,
            },
            Trap {
                name: "underflow",
                asm: "mulss xmm0, xmm1",
                dest: lanes32([MIN_NORMAL_SINGLE, 0, 0, 0]),
                source: singles([0.5, 0.0, 0.0, 0.0]),
                flag: FLAG_UNDERFLOW,
            },
            Trap {
                name: "precision",
                asm: "addss xmm0, xmm1",
                dest: singles([1.0, 0.0, 0.0, 0.0]),
                source: lanes32([TWO_POW_MINUS_24_SINGLE, 0, 0, 0]),
                flag: FLAG_PRECISION,
            },
        ]
    }

    #[test]
    fn unmasked_exception_raises_xm_and_leaves_the_destination_untouched() {
        let mut rig = Rig::new();
        for trap in traps() {
            let mask = trap.flag << MXCSR_MASK_SHIFT;
            rig.h.cpu.xmm[0] = trap.dest;
            rig.h.cpu.xmm[1] = trap.source;
            rig.reset();
            rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
            rig.h.cpu.mxcsr = MXCSR_RESET & !mask;
            let stop = rig.exec(trap.asm);
            assert_eq!(stop, xm(), "{}", trap.name);
            assert_eq!(
                rig.h.cpu.rip, CODE_BASE,
                "{}: the fault points at the instruction",
                trap.name
            );
            assert_eq!(
                rig.xmm(0),
                trap.dest,
                "{}: destination unchanged",
                trap.name
            );
            assert_eq!(
                rig.mxcsr() & MXCSR_EXCEPTION_FLAGS & trap.flag,
                trap.flag,
                "{}: the flag is set even though the exception is taken",
                trap.name
            );
        }
    }

    #[test]
    fn unmasked_exception_without_osxmmexcpt_is_an_invalid_opcode() {
        let mut rig = Rig::new();
        for trap in traps() {
            rig.h.cpu.xmm[0] = trap.dest;
            rig.h.cpu.xmm[1] = trap.source;
            rig.reset();
            rig.h.cpu.mxcsr = MXCSR_RESET & !(trap.flag << MXCSR_MASK_SHIFT);
            assert_eq!(rig.exec(trap.asm), ud(), "{}", trap.name);
            assert_eq!(rig.xmm(0), trap.dest, "{}", trap.name);
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
        }
    }

    #[test]
    fn masking_one_exception_does_not_mask_another() {
        let mut rig = Rig::new();
        let divide = &traps()[2];
        rig.h.cpu.xmm[0] = divide.dest;
        rig.h.cpu.xmm[1] = divide.source;
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        // everything except the invalid mask is set, the divide-by-zero is masked
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_INVALID << MXCSR_MASK_SHIFT);
        assert_eq!(rig.exec(divide.asm), Stop::Halted);
        assert_eq!(dwords(rig.xmm(0))[0], INFINITY_SINGLE);
    }

    #[test]
    fn stale_flags_do_not_trigger_an_unmasked_exception() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([1.0; 4]);
        rig.h.cpu.xmm[1] = singles([2.0; 4]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        // every flag already set, every exception unmasked: an exact add still runs
        rig.h.cpu.mxcsr = MXCSR_EXCEPTION_FLAGS;
        assert_eq!(rig.exec("addps xmm0, xmm1"), Stop::Halted);
        assert_eq!(rig.xmm(0), singles([3.0; 4]));
    }

    #[test]
    fn an_unmasked_exception_in_one_lane_leaves_all_lanes_unwritten() {
        let mut rig = Rig::new();
        let dest = singles([1.0, 2.0, 3.0, 4.0]);
        rig.h.cpu.xmm[0] = dest;
        rig.h.cpu.xmm[1] = singles([2.0, 2.0, 0.0, 2.0]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_DIVIDE_BY_ZERO << MXCSR_MASK_SHIFT);
        assert_eq!(rig.exec("divps xmm0, xmm1"), xm());
        assert_eq!(rig.xmm(0), dest);
    }

    #[test]
    fn all_masks_cleared_still_lets_exact_operations_through() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = doubles([1.0, 2.0]);
        rig.h.cpu.xmm[1] = doubles([4.0, 8.0]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = 0;
        assert_eq!(rig.exec("mulpd xmm0, xmm1"), Stop::Halted);
        assert_eq!(rig.xmm(0), doubles([4.0, 16.0]));
        assert_eq!(rig.mxcsr(), 0);
    }

    #[test]
    fn rcp_and_rsqrt_never_set_exception_flags() {
        let mut rig = Rig::new();
        for asm in ["rcpps xmm0, xmm1", "rsqrtps xmm0, xmm1"] {
            rig.op(
                asm,
                0,
                lanes32([0, 1 << 31, SNAN_SINGLE, (-1.0f32).to_bits()]),
            );
            assert_eq!(rig.mxcsr(), MXCSR_RESET, "{asm}");
        }
    }

    #[test]
    fn rcp_and_rsqrt_instructions() {
        let mut rig = Rig::new();
        let got = rig.op("rcpps xmm0, xmm1", 0, singles([2.0, 4.0, -0.5, 0.0]));
        assert_eq!(
            dwords(got),
            [HALF_SINGLE, 0x3e80_0000, 0xc000_0000, INFINITY_SINGLE]
        );
        let got = rig.op("rsqrtps xmm0, xmm1", 0, singles([4.0, 1.0, 0.25, -1.0]));
        assert_eq!(
            dwords(got),
            [HALF_SINGLE, ONE_SINGLE, TWO_SINGLE, DEFAULT_NAN_SINGLE]
        );
        let got = rig.op(
            "rcpss xmm0, xmm1",
            singles([9.0, 8.0, 7.0, 6.0]),
            singles([8.0, 0.0, 0.0, 0.0]),
        );
        assert_eq!(
            dwords(got),
            [0x3e00_0000, 0x4100_0000, 0x40e0_0000, 0x40c0_0000]
        );
        let got = rig.op(
            "rsqrtss xmm0, xmm1",
            singles([9.0, 8.0, 7.0, 6.0]),
            singles([16.0, 0.0, 0.0, 0.0]),
        );
        assert_eq!(
            dwords(got),
            [0x3e80_0000, 0x4100_0000, 0x40e0_0000, 0x40c0_0000]
        );
        let got = rig.op_mem(
            "rcpps xmm0, xmmword ptr [rsi]",
            0,
            singles([2.0, 2.0, 2.0, 2.0]),
        );
        assert_eq!(dwords(got), [HALF_SINGLE; 4]);
        let got = rig.op_mem(
            "rsqrtss xmm0, dword ptr [rsi]",
            singles([1.0, 2.0, 3.0, 4.0]),
            singles([4.0, 0.0, 0.0, 0.0]),
        );
        assert_eq!(dwords(got)[0], HALF_SINGLE);
    }

    #[test]
    fn double_precision_directed_rounding_through_cvt_and_arithmetic() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = doubles([1.0, 1.0]);
        rig.h.cpu.xmm[1] = doubles([3.0, 3.0]);
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | (RC_UP << MXCSR_RC_SHIFT);
        assert_eq!(rig.exec("divpd xmm0, xmm1"), Stop::Halted);
        assert_eq!(qwords(rig.xmm(0)), [ONE_THIRD_DOUBLE_UP; 2]);
    }

    #[test]
    fn logical_instructions() {
        let mut rig = Rig::new();
        let a = 0xf0f0_f0f0_0f0f_0f0f_ffff_0000_1234_5678u128;
        let b = 0xff00_ff00_ff00_ff00_0ff0_0ff0_aaaa_5555u128;
        for suffix in ["ps", "pd"] {
            assert_eq!(rig.op(&format!("and{suffix} xmm0, xmm1"), a, b), a & b);
            assert_eq!(rig.op(&format!("andn{suffix} xmm0, xmm1"), a, b), !a & b);
            assert_eq!(rig.op(&format!("or{suffix} xmm0, xmm1"), a, b), a | b);
            assert_eq!(rig.op(&format!("xor{suffix} xmm0, xmm1"), a, b), a ^ b);
            assert_eq!(
                rig.op_mem(&format!("and{suffix} xmm0, xmmword ptr [rsi]"), a, b),
                a & b
            );
            assert_eq!(
                rig.op_mem(&format!("andn{suffix} xmm0, xmmword ptr [rsi]"), a, b),
                !a & b
            );
            assert_eq!(
                rig.op_mem(&format!("or{suffix} xmm0, xmmword ptr [rsi]"), a, b),
                a | b
            );
            assert_eq!(
                rig.op_mem(&format!("xor{suffix} xmm0, xmmword ptr [rsi]"), a, b),
                a ^ b
            );
        }
    }

    #[test]
    fn logical_instructions_never_touch_mxcsr_even_for_nans() {
        let mut rig = Rig::new();
        rig.op(
            "andps xmm0, xmm1",
            lanes32([SNAN_SINGLE; 4]),
            lanes32([QNAN_SINGLE; 4]),
        );
        let same = lanes64([SNAN_DOUBLE; 2]);
        assert_eq!(rig.op("xorpd xmm0, xmm1", same, same), 0);
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    #[test]
    fn andnps_complements_the_destination_not_the_source() {
        let mut rig = Rig::new();
        assert_eq!(rig.op("andnps xmm0, xmm1", u128::MAX, 0x1234), 0);
        assert_eq!(rig.op("andnps xmm0, xmm1", 0, 0x1234), 0x1234);
    }

    #[test]
    fn sign_bit_tricks_with_logicals() {
        // The common fabs/fneg idioms: and with 0x7fffffff, xor with 0x80000000.
        let mut rig = Rig::new();
        let values = singles([-1.5, 2.5, -0.0, 3.0]);
        let abs_mask = lanes32([0x7fff_ffff; 4]);
        let sign_mask = lanes32([0x8000_0000; 4]);
        assert_eq!(
            rig.op("andps xmm0, xmm1", values, abs_mask),
            singles([1.5, 2.5, 0.0, 3.0])
        );
        assert_eq!(
            rig.op("xorps xmm0, xmm1", values, sign_mask),
            singles([1.5, -2.5, 0.0, -3.0])
        );
    }

    const A: u128 = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff;
    const B: u128 = 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef;
    const LOW_32: u128 = 0xffff_ffff;
    const LOW_64: u128 = u64::MAX as u128;
    const SENTINEL: u8 = 0xa5;
    /// Bytes of scratch filled with `SENTINEL` around a store, to see exactly how many it wrote.
    const GUARD_BYTES: usize = 48;

    fn fill_sentinel(rig: &mut Rig, address: u64) {
        rig.h.write(address - 16, &[SENTINEL; GUARD_BYTES]);
    }

    /// Number of bytes of the guard region around `address` that no longer hold the sentinel, and the
    /// offset of the first one.
    fn changed_bytes(rig: &Rig, address: u64) -> Vec<usize> {
        rig.h
            .read(address - 16, GUARD_BYTES)
            .iter()
            .enumerate()
            .filter(|(_, b)| **b != SENTINEL)
            .map(|(k, _)| k)
            .collect()
    }

    #[test]
    fn full_register_moves_copy_128_bits() {
        let mut rig = Rig::new();
        for name in ["movups", "movupd", "movaps", "movapd", "movdqa", "movdqu"] {
            assert_eq!(rig.op(&format!("{name} xmm0, xmm1"), A, B), B, "{name}");
            assert_eq!(rig.xmm(1), B, "{name} leaves the source");
            assert_eq!(
                rig.op_mem(&format!("{name} xmm0, xmmword ptr [rsi]"), A, B),
                B,
                "{name} load"
            );
        }
        rig.h.cpu.xmm[10] = B;
        rig.run_ok("movups xmm9, xmm10");
        assert_eq!(rig.xmm(9), B);
    }

    #[test]
    fn full_register_stores_write_exactly_16_bytes() {
        let mut rig = Rig::new();
        for name in ["movups", "movupd", "movaps", "movapd", "movdqa", "movdqu"] {
            fill_sentinel(&mut rig, DATA);
            rig.h.cpu.xmm[0] = A;
            rig.run_ok(&format!("{name} xmmword ptr [rsi], xmm0"));
            assert_eq!(rig.memory(DATA), A, "{name}");
            assert_eq!(
                changed_bytes(&rig, DATA),
                (16..32).collect::<Vec<_>>(),
                "{name}"
            );
            assert_eq!(rig.xmm(0), A);
        }
    }

    #[test]
    fn register_to_register_store_forms_copy_reg_into_rm() {
        // 0F 11 /r, 0F 29 /r, 0F 7F /r with ModRM mod == 3: xmm1 (reg) is stored into xmm0 (rm).
        let mut rig = Rig::new();
        for key in [
            ".byte 0x0f, 0x11, 0xc8",
            ".byte 0x66, 0x0f, 0x11, 0xc8",
            ".byte 0x0f, 0x29, 0xc8",
            ".byte 0x66, 0x0f, 0x29, 0xc8",
            ".byte 0x66, 0x0f, 0x7f, 0xc8",
            ".byte 0xf3, 0x0f, 0x7f, 0xc8",
        ] {
            assert_eq!(rig.op(key, A, B), B, "{key}");
            assert_eq!(rig.xmm(1), B, "{key}");
        }
    }

    #[test]
    fn movss_register_form_merges_the_low_dword() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op("movss xmm0, xmm1", A, B),
            (A & !LOW_32) | (B & LOW_32)
        );
    }

    #[test]
    fn movss_load_zeroes_the_upper_96_bits() {
        let mut rig = Rig::new();
        assert_eq!(rig.op_mem("movss xmm0, dword ptr [rsi]", A, B), B & LOW_32);
    }

    #[test]
    fn movss_store_writes_four_bytes() {
        let mut rig = Rig::new();
        fill_sentinel(&mut rig, DATA);
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("movss dword ptr [rsi], xmm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..20).collect::<Vec<_>>());
        assert_eq!(rig.h.read(DATA, 4), (A as u32).to_le_bytes());
    }

    #[test]
    fn movss_store_register_form_merges_into_the_destination() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op(".byte 0xf3, 0x0f, 0x11, 0xc8", A, B),
            (A & !LOW_32) | (B & LOW_32)
        );
    }

    #[test]
    fn movsd_forms() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op("movsd xmm0, xmm1", A, B),
            (A & !LOW_64) | (B & LOW_64)
        );
        assert_eq!(rig.op_mem("movsd xmm0, qword ptr [rsi]", A, B), B & LOW_64);
        assert_eq!(
            rig.op(".byte 0xf2, 0x0f, 0x11, 0xc8", A, B),
            (A & !LOW_64) | (B & LOW_64)
        );
        fill_sentinel(&mut rig, DATA);
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("movsd qword ptr [rsi], xmm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..24).collect::<Vec<_>>());
        assert_eq!(rig.h.read_u64(DATA), A as u64);
    }

    #[test]
    fn movlps_and_movlpd_load_the_low_half_and_keep_the_high_half() {
        let mut rig = Rig::new();
        for name in ["movlps", "movlpd"] {
            let got = rig.op_mem(&format!("{name} xmm0, qword ptr [rsi]"), A, B);
            assert_eq!(got, (A & !LOW_64) | (B & LOW_64), "{name}");
        }
    }

    #[test]
    fn movhps_and_movhpd_load_the_high_half_and_keep_the_low_half() {
        let mut rig = Rig::new();
        for name in ["movhps", "movhpd"] {
            let got = rig.op_mem(&format!("{name} xmm0, qword ptr [rsi]"), A, B);
            assert_eq!(got, (A & LOW_64) | ((B & LOW_64) << 64), "{name}");
        }
    }

    #[test]
    fn movlps_movhps_stores_write_eight_bytes() {
        let mut rig = Rig::new();
        for (name, expected) in [
            ("movlps", A as u64),
            ("movlpd", A as u64),
            ("movhps", (A >> 64) as u64),
            ("movhpd", (A >> 64) as u64),
        ] {
            fill_sentinel(&mut rig, DATA);
            rig.h.cpu.xmm[0] = A;
            rig.run_ok(&format!("{name} qword ptr [rsi], xmm0"));
            assert_eq!(rig.h.read_u64(DATA), expected, "{name}");
            assert_eq!(
                changed_bytes(&rig, DATA),
                (16..24).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    #[test]
    fn movhlps_copies_the_source_high_half_to_the_destination_low_half() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op("movhlps xmm0, xmm1", A, B),
            (A & !LOW_64) | (B >> 64)
        );
    }

    #[test]
    fn movlhps_copies_the_source_low_half_to_the_destination_high_half() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op("movlhps xmm0, xmm1", A, B),
            (A & LOW_64) | ((B & LOW_64) << 64)
        );
    }

    #[test]
    fn movlpd_and_movhpd_register_forms_are_invalid() {
        let mut rig = Rig::new();
        for key in [
            ".byte 0x66, 0x0f, 0x12, 0xc1",
            ".byte 0x66, 0x0f, 0x16, 0xc1",
        ] {
            rig.h.cpu.xmm[0] = A;
            rig.h.cpu.xmm[1] = B;
            assert_eq!(rig.run(key), ud(), "{key}");
            assert_eq!(rig.xmm(0), A, "{key}: nothing written");
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
        }
    }

    #[test]
    fn movlpd_movhpd_register_form_is_ud_even_with_ts_set() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.cr0 |= CR0_TS;
        assert_eq!(rig.exec(".byte 0x66, 0x0f, 0x12, 0xc1"), ud());
    }

    #[test]
    fn memory_only_stores_with_a_register_operand_are_invalid() {
        let mut rig = Rig::new();
        for key in [
            ".byte 0x0f, 0x13, 0xc1",
            ".byte 0x66, 0x0f, 0x13, 0xc1",
            ".byte 0x0f, 0x17, 0xc1",
            ".byte 0x66, 0x0f, 0x17, 0xc1",
            ".byte 0x0f, 0x2b, 0xc1",
            ".byte 0x66, 0x0f, 0x2b, 0xc1",
            ".byte 0x66, 0x0f, 0xe7, 0xc1",
        ] {
            rig.h.cpu.xmm[0] = A;
            rig.h.cpu.xmm[1] = B;
            assert_eq!(rig.run(key), ud(), "{key}");
            assert_eq!((rig.xmm(0), rig.xmm(1)), (A, B), "{key}: nothing written");
        }
    }

    #[test]
    fn non_temporal_stores_write_16_aligned_bytes() {
        let mut rig = Rig::new();
        for name in ["movntps", "movntpd", "movntdq"] {
            fill_sentinel(&mut rig, DATA);
            rig.h.cpu.xmm[0] = A;
            rig.run_ok(&format!("{name} xmmword ptr [rsi], xmm0"));
            assert_eq!(rig.memory(DATA), A, "{name}");
            assert_eq!(
                changed_bytes(&rig, DATA),
                (16..32).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    #[test]
    fn movmskps_collects_the_four_sign_bits() {
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.h.cpu.xmm[1] = lanes32([0x8000_0000, 0, 0xffff_ffff, 0x7fff_ffff]);
        rig.run_ok("movmskps eax, xmm1");
        assert_eq!(rig.gpr(RAX), 0b0101, "the upper half of rax is zeroed");
        rig.h.cpu.xmm[1] = lanes32([0, 0x8000_0000, 0, 0xf000_0000]);
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("movmskps rax, xmm1");
        assert_eq!(rig.gpr(RAX), 0b1010);
        rig.h.cpu.xmm[10] = lanes32([u32::MAX; 4]);
        rig.run_ok("movmskps r9d, xmm10");
        assert_eq!(rig.gpr(9), 0b1111);
    }

    #[test]
    fn movmskpd_collects_the_two_sign_bits() {
        let mut rig = Rig::new();
        for (value, expected) in [
            (lanes64([1 << 63, 0]), 1u64),
            (lanes64([0, u64::MAX]), 2),
            (lanes64([u64::MAX, 1 << 63]), 3),
            (lanes64([0x7fff_ffff_ffff_ffff, 1]), 0),
        ] {
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.h.cpu.xmm[1] = value;
            rig.run_ok("movmskpd eax, xmm1");
            assert_eq!(rig.gpr(RAX), expected);
        }
    }

    #[test]
    fn pmovmskb_collects_sixteen_sign_bits() {
        let mut rig = Rig::new();
        let value = lanes8([
            0x80, 0, 0xff, 0x7f, 0x81, 0, 0, 0, 0, 0, 0, 0x80, 0, 0, 0, 0xff,
        ]);
        let expected = bytes(value)
            .iter()
            .enumerate()
            .fold(0u64, |mask, (k, b)| mask | (((b >> 7) as u64) << k));
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.h.cpu.xmm[1] = value;
        rig.run_ok("pmovmskb eax, xmm1");
        assert_eq!(rig.gpr(RAX), expected);
        assert_eq!(expected, 0b1000_1000_0001_0101);
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("pmovmskb rax, xmm1");
        assert_eq!(rig.gpr(RAX), expected);
        rig.h.cpu.xmm[1] = u128::MAX;
        rig.run_ok("pmovmskb eax, xmm1");
        assert_eq!(rig.gpr(RAX), 0xffff);
    }

    #[test]
    fn movmsk_with_a_memory_operand_is_invalid() {
        let mut rig = Rig::new();
        for key in [
            ".byte 0x0f, 0x50, 0x06",
            ".byte 0x66, 0x0f, 0x50, 0x06",
            ".byte 0x66, 0x0f, 0xd7, 0x06",
        ] {
            assert_eq!(rig.run(key), ud(), "{key}");
        }
    }

    #[test]
    fn movd_and_movq_into_a_vector_register_zero_extend() {
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = 0xdead_beef_1234_5678;
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("movd xmm0, eax");
        assert_eq!(rig.xmm(0), 0x1234_5678);
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("movq xmm0, rax");
        assert_eq!(rig.xmm(0), 0xdead_beef_1234_5678);
        assert_eq!(rig.op_mem("movd xmm0, dword ptr [rsi]", A, B), B & LOW_32);
        assert_eq!(rig.op_mem("movq xmm0, qword ptr [rsi]", A, B), B & LOW_64);
        rig.h.cpu.gpr[8] = 0xffff_ffff_8000_0001;
        rig.run_ok("movd xmm9, r8d");
        assert_eq!(rig.xmm(9), 0x8000_0001);
    }

    #[test]
    fn movd_and_movq_out_of_a_vector_register() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = A;
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("movd eax, xmm0");
        assert_eq!(
            rig.gpr(RAX),
            (A & LOW_32) as u64,
            "movd to a 32-bit register zero-extends"
        );
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("movq rax, xmm0");
        assert_eq!(rig.gpr(RAX) as u128, A & LOW_64);
        rig.h.cpu.xmm[10] = B;
        rig.run_ok("movq r9, xmm10");
        assert_eq!(rig.gpr(9) as u128, B & LOW_64);
        fill_sentinel(&mut rig, DATA);
        rig.run_ok("movd dword ptr [rsi], xmm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..20).collect::<Vec<_>>());
        assert_eq!(rig.h.read(DATA, 4), (A as u32).to_le_bytes());
        fill_sentinel(&mut rig, DATA);
        rig.run_ok("movq qword ptr [rsi], xmm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..24).collect::<Vec<_>>());
        assert_eq!(rig.h.read_u64(DATA), A as u64);
    }

    #[test]
    fn movq_between_vector_registers_zeroes_the_upper_half() {
        let mut rig = Rig::new();
        for key in [
            "movq xmm0, xmm1",
            ".byte 0xf3, 0x0f, 0x7e, 0xc1",
            ".byte 0x66, 0x0f, 0xd6, 0xc8",
        ] {
            assert_eq!(rig.op(key, A, B), B & LOW_64, "{key}");
        }
        assert_eq!(rig.op_mem(".byte 0xf3, 0x0f, 0x7e, 0x06", A, B), B & LOW_64);
    }

    #[test]
    fn zero_idiom_pxor_clears_a_register() {
        let mut rig = Rig::new();
        assert_eq!(rig.op("pxor xmm0, xmm1", A, A), 0);
    }

    fn all_keys_with(needle: &str) -> Vec<&'static str> {
        SNIPPETS
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| key.contains(needle))
            .collect()
    }

    /// Loads and stores that explicitly allow any 16-byte alignment.
    fn allows_unaligned(key: &str) -> bool {
        ["movups ", "movupd ", "movdqu "]
            .iter()
            .any(|prefix| key.starts_with(prefix))
    }

    #[test]
    fn every_aligned_form_with_a_misaligned_128_bit_operand_raises_gp0() {
        let mut rig = Rig::new();
        let keys = all_keys_with("xmmword ptr [rsi]");
        assert!(keys.len() > 100, "the table covers the m128 forms");
        for key in keys {
            for offset in [1u64, 4, 8, 15] {
                rig.h.cpu.xmm[0] = A;
                rig.h.write(DATA - 16, &[SENTINEL; 64]);
                rig.reset();
                rig.h.cpu.gpr[RSI] = DATA + offset;
                let stop = rig.exec(key);
                if allows_unaligned(key) {
                    assert_eq!(stop, Stop::Halted, "{key} at +{offset}");
                } else {
                    assert_eq!(stop, gp0(), "{key} at +{offset}");
                    assert_eq!(rig.h.cpu.rip, CODE_BASE, "{key}");
                    assert_eq!(rig.xmm(0), A, "{key}: register untouched");
                    assert!(
                        changed_bytes(&rig, DATA).is_empty(),
                        "{key}: memory untouched"
                    );
                }
            }
        }
    }

    #[test]
    fn every_128_bit_memory_form_works_at_an_aligned_address() {
        let mut rig = Rig::new();
        let keys = all_keys_with("xmmword ptr [rsi]");
        for key in keys {
            for offset in [0u64, 16, 48] {
                rig.h.cpu.xmm[0] = A;
                rig.reset();
                rig.h.cpu.gpr[RSI] = DATA + offset;
                assert_eq!(rig.exec(key), Stop::Halted, "{key} at +{offset}");
            }
        }
    }

    #[test]
    fn movups_movupd_movdqu_access_any_address() {
        let mut rig = Rig::new();
        let source: Vec<u8> = (0u8..40).collect();
        rig.h.write(DATA, &source);
        for name in ["movups", "movupd", "movdqu"] {
            for offset in 1..16u64 {
                rig.h.cpu.xmm[0] = 0;
                rig.reset();
                rig.h.cpu.gpr[RSI] = DATA + offset;
                assert_eq!(
                    rig.exec(&format!("{name} xmm0, xmmword ptr [rsi]")),
                    Stop::Halted
                );
                let expected = u128::from_le_bytes(
                    source[offset as usize..offset as usize + 16]
                        .try_into()
                        .unwrap(),
                );
                assert_eq!(rig.xmm(0), expected, "{name} +{offset}");
            }
            rig.h.cpu.xmm[0] = A;
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 0x103;
            fill_sentinel(&mut rig, DATA + 0x103);
            assert_eq!(
                rig.exec(&format!("{name} xmmword ptr [rsi], xmm0")),
                Stop::Halted
            );
            assert_eq!(
                rig.h.read(DATA + 0x103, 16),
                A.to_le_bytes(),
                "{name} store"
            );
        }
    }

    #[test]
    fn scalar_and_64_bit_memory_operands_need_no_alignment() {
        let mut rig = Rig::new();
        let mut checked = 0;
        for (key, _) in SNIPPETS {
            let narrow = key.contains("dword ptr [rsi]")
                || key.contains("qword ptr [rsi]")
                || key.contains("word ptr [rsi]");
            if !narrow || key.contains("xmmword") || key.contains("fxsave") {
                continue;
            }
            // ldmxcsr reads zeros there, which is a valid MXCSR
            rig.h.cpu.xmm[0] = A;
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + 1;
            assert_eq!(rig.exec(key), Stop::Halted, "{key}");
            checked += 1;
        }
        assert!(checked > 40);
    }

    #[test]
    fn alignment_fault_leaves_rip_and_rsp_untouched() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = A;
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        let rsp = rig.h.cpu.gpr[RSP];
        assert_eq!(rig.exec("addps xmm0, xmmword ptr [rsi]"), gp0());
        assert_eq!(rig.h.cpu.rip, CODE_BASE);
        assert_eq!(rig.h.cpu.gpr[RSP], rsp);
    }

    #[test]
    fn store_alignment_is_checked_for_aligned_stores_only() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = A;
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 4;
        for key in [
            "movaps xmmword ptr [rsi], xmm0",
            "movapd xmmword ptr [rsi], xmm0",
            "movdqa xmmword ptr [rsi], xmm0",
            "movntps xmmword ptr [rsi], xmm0",
            "movntpd xmmword ptr [rsi], xmm0",
            "movntdq xmmword ptr [rsi], xmm0",
        ] {
            assert_eq!(rig.exec(key), gp0(), "{key}");
        }
        for key in [
            "movups xmmword ptr [rsi], xmm0",
            "movupd xmmword ptr [rsi], xmm0",
            "movdqu xmmword ptr [rsi], xmm0",
            "movss dword ptr [rsi], xmm0",
            "movsd qword ptr [rsi], xmm0",
            "movlps qword ptr [rsi], xmm0",
            "movhpd qword ptr [rsi], xmm0",
            "movq qword ptr [rsi], xmm0",
            "movd dword ptr [rsi], xmm0",
        ] {
            assert_eq!(rig.exec(key), Stop::Halted, "{key}");
        }
    }

    /// Snippets whose operands are all MMX registers: gated by CR0.EM/TS only, not by CR4.OSFXSR.
    fn is_mmx_only(key: &str) -> bool {
        (key.contains(" mm") && !key.contains("xmm") && !key.starts_with("cvt"))
            || key == ".byte 0x0f, 0x7f, 0xc8"
    }

    fn is_sse_instruction_for_gating(key: &str) -> bool {
        const NOT_SSE: [&str; 11] = [
            "prefetch", "clflush", "sfence", "lfence", "mfence", "pause", "emms", "ldmxcsr",
            "stmxcsr", "fxsave", "fxrstor",
        ];
        if NOT_SSE.iter().any(|p| key.starts_with(p))
            || key.contains("0xae")
            || key.contains("0x77")
            || is_mmx_only(key)
        {
            return false;
        }
        !(key.contains("; "))
    }

    fn valid_sse_keys(rig: &mut Rig) -> Vec<&'static str> {
        SNIPPETS
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| is_sse_instruction_for_gating(key))
            .filter(|key| rig.run(key) == Stop::Halted)
            .collect()
    }

    #[test]
    fn cr0_em_makes_every_sse_instruction_invalid() {
        let mut rig = Rig::new();
        let keys = valid_sse_keys(&mut rig);
        assert!(keys.len() > 500, "{}", keys.len());
        for key in keys {
            rig.h.cpu.xmm[0] = A;
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM;
            assert_eq!(rig.exec(key), ud(), "{key}");
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
            assert_eq!(rig.xmm(0), A);
        }
    }

    #[test]
    fn cr0_ts_makes_every_sse_instruction_raise_device_not_available() {
        let mut rig = Rig::new();
        for key in valid_sse_keys(&mut rig) {
            rig.h.cpu.xmm[0] = A;
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(key), nm(), "{key}");
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
            assert_eq!(rig.xmm(0), A);
        }
    }

    #[test]
    fn cr0_em_takes_priority_over_ts() {
        let mut rig = Rig::new();
        for key in valid_sse_keys(&mut rig) {
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM | CR0_TS;
            assert_eq!(rig.exec(key), ud(), "{key}");
        }
    }

    #[test]
    fn cr4_osfxsr_clear_makes_every_sse_instruction_invalid() {
        let mut rig = Rig::new();
        for key in valid_sse_keys(&mut rig) {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(key), ud(), "{key}");
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(key), ud(), "{key}: #UD beats #NM");
        }
    }

    #[test]
    fn sse_gating_is_checked_before_the_memory_operand_is_touched() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.cr0 |= CR0_TS;
        rig.h.cpu.gpr[RSI] = 0xffff_ffff_ffff_f000;
        assert_eq!(rig.exec("addps xmm0, xmmword ptr [rsi]"), nm());
    }

    #[test]
    fn emms_uses_the_mmx_rules() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.ftw = 0xff;
        rig.h.cpu.cr4 &= !CR4_OSFXSR;
        assert_eq!(
            rig.exec("emms"),
            Stop::Halted,
            "emms does not need CR4.OSFXSR"
        );
        assert_eq!(rig.h.cpu.ftw, 0, "all x87 registers marked empty");
        rig.reset();
        rig.h.cpu.cr0 |= CR0_TS;
        rig.h.cpu.ftw = 0xff;
        assert_eq!(rig.exec("emms"), nm());
        assert_eq!(rig.h.cpu.ftw, 0xff);
        rig.reset();
        rig.h.cpu.cr0 |= CR0_EM;
        assert_eq!(
            rig.exec("emms"),
            ud(),
            "CR0.EM is #UD for MMX instructions, not #NM"
        );
        rig.reset();
        rig.h.cpu.cr0 |= CR0_EM | CR0_TS;
        assert_eq!(rig.exec("emms"), ud());
    }

    #[test]
    fn emms_with_a_mandatory_prefix_is_invalid() {
        let mut rig = Rig::new();
        assert_eq!(rig.run(".byte 0x66, 0x0f, 0x77"), ud());
        assert_eq!(rig.run(".byte 0xf3, 0x0f, 0x77"), ud());
    }

    /// (opcode, prefixes that select an implemented instruction): n = none, 6 = 66, 3 = F3, 2 = F2.
    /// Written out from the SDM opcode map for the SSE/SSE2 instructions implemented; everything else
    /// in the 0F map (SSE3 and later, MMX forms, x87 and system opcodes) must be rejected.
    const IMPLEMENTED: &[(u8, &str)] = &[
        (0x10, "n632"),
        (0x11, "n632"),
        (0x12, "n6"),
        (0x13, "n6"),
        (0x14, "n6"),
        (0x15, "n6"),
        (0x16, "n6"),
        (0x17, "n6"),
        (0x28, "n6"),
        (0x29, "n6"),
        (0x2a, "n632"),
        (0x2b, "n6"),
        (0x2c, "n632"),
        (0x2d, "n632"),
        (0x2e, "n6"),
        (0x2f, "n6"),
        (0x50, "n6"),
        (0x51, "n632"),
        (0x52, "n3"),
        (0x53, "n3"),
        (0x54, "n6"),
        (0x55, "n6"),
        (0x56, "n6"),
        (0x57, "n6"),
        (0x58, "n632"),
        (0x59, "n632"),
        (0x5a, "n632"),
        (0x5b, "n63"),
        (0x5c, "n632"),
        (0x5d, "n632"),
        (0x5e, "n632"),
        (0x5f, "n632"),
        (0x60, "n6"),
        (0x61, "n6"),
        (0x62, "n6"),
        (0x63, "n6"),
        (0x64, "n6"),
        (0x65, "n6"),
        (0x66, "n6"),
        (0x67, "n6"),
        (0x68, "n6"),
        (0x69, "n6"),
        (0x6a, "n6"),
        (0x6b, "n6"),
        (0x6c, "6"),
        (0x6d, "6"),
        (0x6e, "n6"),
        (0x6f, "n63"),
        (0x70, "n632"),
        (0x71, "n6"),
        (0x72, "n6"),
        (0x73, "n6"),
        (0x74, "n6"),
        (0x75, "n6"),
        (0x76, "n6"),
        (0x77, "n"),
        (0x7e, "n63"),
        (0x7f, "n63"),
        (0xc2, "n632"),
        (0xc4, "n6"),
        (0xc5, "n6"),
        (0xc6, "n6"),
        (0xd1, "n6"),
        (0xd2, "n6"),
        (0xd3, "n6"),
        (0xd4, "n6"),
        (0xd5, "n6"),
        (0xd6, "632"),
        (0xd7, "n6"),
        (0xd8, "n6"),
        (0xd9, "n6"),
        (0xda, "n6"),
        (0xdb, "n6"),
        (0xdc, "n6"),
        (0xdd, "n6"),
        (0xde, "n6"),
        (0xdf, "n6"),
        (0xe0, "n6"),
        (0xe1, "n6"),
        (0xe2, "n6"),
        (0xe3, "n6"),
        (0xe4, "n6"),
        (0xe5, "n6"),
        (0xe6, "632"),
        (0xe7, "n6"),
        (0xe8, "n6"),
        (0xe9, "n6"),
        (0xea, "n6"),
        (0xeb, "n6"),
        (0xec, "n6"),
        (0xed, "n6"),
        (0xee, "n6"),
        (0xef, "n6"),
        (0xf1, "n6"),
        (0xf2, "n6"),
        (0xf3, "n6"),
        (0xf4, "n6"),
        (0xf5, "n6"),
        (0xf6, "n6"),
        (0xf7, "n"),
        (0xf8, "n6"),
        (0xf9, "n6"),
        (0xfa, "n6"),
        (0xfb, "n6"),
        (0xfc, "n6"),
        (0xfd, "n6"),
        (0xfe, "n6"),
    ];

    #[test]
    fn opcode_map_accepts_exactly_the_implemented_opcode_prefix_pairs() {
        for opcode in 0..=255u8 {
            let accepted = IMPLEMENTED
                .iter()
                .find(|(o, _)| *o == opcode)
                .map_or("", |(_, p)| *p);
            for (prefix, letter) in [
                (Prefix::Np, 'n'),
                (Prefix::P66, '6'),
                (Prefix::F3, '3'),
                (Prefix::F2, '2'),
            ] {
                assert_eq!(
                    allowed(opcode, prefix),
                    accepted.contains(letter),
                    "opcode {opcode:#04x} prefix {letter}"
                );
            }
        }
    }

    #[test]
    fn form_validity_table() {
        use Prefix::*;
        // memory-only
        for opcode in [0x13u8, 0x17, 0x2b, 0xe7] {
            assert!(form_valid(opcode, P66, false, 0), "{opcode:#x} memory");
            assert!(!form_valid(opcode, P66, true, 0), "{opcode:#x} register");
            assert!(!form_valid(opcode, Np, true, 0), "{opcode:#x} register");
        }
        // register-only
        for opcode in [0x50u8, 0xc5, 0xd7] {
            assert!(form_valid(opcode, P66, true, 0), "{opcode:#x} register");
            assert!(!form_valid(opcode, P66, false, 0), "{opcode:#x} memory");
        }
        // movlps/movhps have register forms (movhlps/movlhps); the pd forms do not
        for opcode in [0x12u8, 0x16] {
            assert!(form_valid(opcode, Np, true, 0));
            assert!(form_valid(opcode, Np, false, 0));
            assert!(form_valid(opcode, P66, false, 0));
            assert!(!form_valid(opcode, P66, true, 0));
        }
        // shift groups: register only, and only the assigned /reg values
        for (opcode, valid) in [
            (0x71u8, [2u8, 4, 6].as_slice()),
            (0x72, &[2, 4, 6]),
            (0x73, &[2, 3, 6, 7]),
        ] {
            for group in 0..8u8 {
                assert_eq!(
                    form_valid(opcode, P66, true, group),
                    valid.contains(&group),
                    "{opcode:#x} /{group}"
                );
                assert!(
                    !form_valid(opcode, P66, false, group),
                    "{opcode:#x} /{group} memory"
                );
            }
        }
        // everything else accepts either form
        for opcode in [0x10u8, 0x58, 0x6f, 0xd4, 0xc6] {
            assert!(form_valid(opcode, P66, true, 0));
            assert!(form_valid(opcode, P66, false, 0));
        }
    }

    #[test]
    fn invalid_prefix_combinations_raise_invalid_opcode() {
        let mut rig = Rig::new();
        let invalid = [
            ".byte 0xf3, 0x0f, 0xfc, 0xc1",
            ".byte 0xf2, 0x0f, 0xfc, 0xc1",
            ".byte 0xf2, 0x0f, 0x6f, 0xc1",
            ".byte 0xf2, 0x0f, 0x14, 0xc1",
            ".byte 0xf3, 0x0f, 0x14, 0xc1",
            ".byte 0xf3, 0x0f, 0x28, 0xc1",
            ".byte 0xf2, 0x0f, 0x28, 0xc1",
            ".byte 0xf3, 0x0f, 0x2e, 0xc1",
            ".byte 0xf2, 0x0f, 0x2e, 0xc1",
            ".byte 0xf3, 0x0f, 0x54, 0xc1",
            ".byte 0xf2, 0x0f, 0x54, 0xc1",
            ".byte 0xf2, 0x0f, 0x52, 0xc1",
            ".byte 0x66, 0x0f, 0x52, 0xc1",
            ".byte 0xf2, 0x0f, 0x5b, 0xc1",
            ".byte 0x0f, 0xe6, 0xc1",
            ".byte 0xf2, 0x0f, 0x7e, 0xc1",
            ".byte 0xf2, 0x0f, 0x7f, 0xc1",
            ".byte 0x0f, 0xd6, 0xc1",
            ".byte 0xf3, 0x0f, 0xc6, 0xc1, 0x00",
            ".byte 0x66, 0x0f, 0xf7, 0xc1",
            ".byte 0x66, 0x0f, 0x78, 0xc1",
        ];
        for key in invalid {
            rig.h.cpu.xmm[0] = A;
            rig.h.cpu.xmm[1] = B;
            assert_eq!(rig.run(key), ud(), "{key}");
            assert_eq!((rig.xmm(0), rig.xmm(1)), (A, B), "{key}");
            assert_eq!(rig.h.cpu.rip, CODE_BASE, "{key}");
        }
    }

    #[test]
    fn invalid_shift_group_encodings_raise_invalid_opcode() {
        let mut rig = Rig::new();
        for opcode in ["0x71", "0x72"] {
            for group in [0u8, 1, 3, 5, 7] {
                let modrm = 0xc0 + group * 8;
                let key = format!(".byte 0x66, 0x0f, {opcode}, {modrm:#04x}, 0x01");
                assert_eq!(rig.run(&key), ud(), "{key}");
            }
        }
        for group in [0u8, 1, 4, 5] {
            let modrm = 0xc0 + group * 8;
            let key = format!(".byte 0x66, 0x0f, 0x73, {modrm:#04x}, 0x01");
            assert_eq!(rig.run(&key), ud(), "{key}");
        }
        assert_eq!(
            rig.run(".byte 0x66, 0x0f, 0x71, 0x16, 0x01"),
            ud(),
            "psrlw with a memory operand"
        );
        assert_eq!(
            rig.run(".byte 0x66, 0x0f, 0x73, 0x1e, 0x01"),
            ud(),
            "psrldq with a memory operand"
        );
    }

    #[test]
    fn invalid_encoding_is_ud_even_with_cr0_ts_set() {
        let mut rig = Rig::new();
        for key in [
            ".byte 0x0f, 0x13, 0xc1",
            ".byte 0x66, 0x0f, 0x71, 0xc0, 0x01",
            ".byte 0x66, 0x0f, 0xd7, 0x06",
            ".byte 0xf2, 0x0f, 0x54, 0xc1",
        ] {
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(key), ud(), "{key}");
        }
    }

    fn set_status_flags(rig: &mut Rig, flags: u64) {
        rig.reset();
        rig.h.cpu.set_eflags_arith(flags);
    }

    /// Runs a (u)comis instruction with every status flag preset to `preset` and returns the status
    /// flags afterwards.
    fn comis_flags(rig: &mut Rig, asm: &str, dst: u128, src: u128, preset: u64) -> u64 {
        rig.h.cpu.xmm[0] = dst;
        rig.h.cpu.xmm[1] = src;
        set_status_flags(rig, preset);
        assert_eq!(rig.exec(asm), Stop::Halted, "{asm}");
        rig.flags()
    }

    #[test]
    fn ucomiss_sets_zf_pf_cf_and_clears_the_other_status_flags() {
        let mut rig = Rig::new();
        let single = |v: f32| lanes32([v.to_bits(), 0, 0, 0]);
        for (a, b, expected) in [
            (2.0, 1.0, 0),
            (1.0, 2.0, CF),
            (1.0, 1.0, ZF),
            (0.0, -0.0, ZF),
            (f32::INFINITY, f32::MAX, 0),
            (f32::NEG_INFINITY, f32::MIN, CF),
            (f32::INFINITY, f32::INFINITY, ZF),
        ] {
            for asm in ["ucomiss xmm0, xmm1", "comiss xmm0, xmm1"] {
                // all six status flags set beforehand: OF, SF and AF must be cleared
                assert_eq!(
                    comis_flags(&mut rig, asm, single(a), single(b), ARITH),
                    expected,
                    "{asm} {a} {b}"
                );
            }
        }
    }

    #[test]
    fn comisd_and_ucomisd_compare_the_low_double() {
        let mut rig = Rig::new();
        let double = |v: f64| lanes64([v.to_bits(), (99.0f64).to_bits()]);
        for (a, b, expected) in [
            (2.0, 1.0, 0),
            (1.0, 2.0, CF),
            (1.0, 1.0, ZF),
            (-0.0, 0.0, ZF),
            (1e300, 1e-300, 0),
        ] {
            for asm in ["ucomisd xmm0, xmm1", "comisd xmm0, xmm1"] {
                assert_eq!(
                    comis_flags(&mut rig, asm, double(a), double(b), ARITH),
                    expected,
                    "{asm}"
                );
            }
        }
    }

    #[test]
    fn comis_with_a_nan_is_unordered() {
        let mut rig = Rig::new();
        let nan_a = lanes32([QNAN_SINGLE, 0, 0, 0]);
        let one = singles([1.0, 0.0, 0.0, 0.0]);
        for asm in ["ucomiss xmm0, xmm1", "comiss xmm0, xmm1"] {
            assert_eq!(
                comis_flags(&mut rig, asm, nan_a, one, 0),
                ZF | PF | CF,
                "{asm}"
            );
            assert_eq!(
                comis_flags(&mut rig, asm, one, nan_a, 0),
                ZF | PF | CF,
                "{asm}"
            );
            assert_eq!(
                comis_flags(&mut rig, asm, nan_a, nan_a, 0),
                ZF | PF | CF,
                "{asm}"
            );
        }
        let nan_d = lanes64([QNAN_DOUBLE, 0]);
        let one_d = doubles([1.0, 0.0]);
        for asm in ["ucomisd xmm0, xmm1", "comisd xmm0, xmm1"] {
            assert_eq!(
                comis_flags(&mut rig, asm, nan_d, one_d, 0),
                ZF | PF | CF,
                "{asm}"
            );
        }
    }

    #[test]
    fn ucomis_flags_invalid_only_for_signaling_nans_comis_for_any_nan() {
        let mut rig = Rig::new();
        let qnan = lanes32([QNAN_SINGLE, 0, 0, 0]);
        let snan = lanes32([SNAN_SINGLE, 0, 0, 0]);
        let one = singles([1.0, 0.0, 0.0, 0.0]);
        comis_flags(&mut rig, "ucomiss xmm0, xmm1", qnan, one, 0);
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, 0);
        comis_flags(&mut rig, "ucomiss xmm0, xmm1", one, snan, 0);
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
        comis_flags(&mut rig, "comiss xmm0, xmm1", qnan, one, 0);
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
        comis_flags(
            &mut rig,
            "ucomisd xmm0, xmm1",
            lanes64([QNAN_DOUBLE, 0]),
            doubles([1.0, 0.0]),
            0,
        );
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, 0);
        comis_flags(
            &mut rig,
            "comisd xmm0, xmm1",
            lanes64([QNAN_DOUBLE, 0]),
            doubles([1.0, 0.0]),
            0,
        );
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_INVALID);
    }

    #[test]
    fn comis_memory_forms_read_only_the_scalar() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([1.0, 0.0, 0.0, 0.0]);
        rig.h.write(DATA, &2.0f32.to_le_bytes());
        set_status_flags(&mut rig, 0);
        assert_eq!(rig.exec("comiss xmm0, dword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.flags(), CF);
        rig.h.cpu.xmm[0] = doubles([2.0, 0.0]);
        rig.h.write(DATA, &1.0f64.to_le_bytes());
        set_status_flags(&mut rig, 0);
        assert_eq!(rig.exec("ucomisd xmm0, qword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.flags(), 0);
    }

    #[test]
    fn comis_does_not_modify_the_vector_registers() {
        let mut rig = Rig::new();
        comis_flags(&mut rig, "comiss xmm0, xmm1", A, B, 0);
        assert_eq!((rig.xmm(0), rig.xmm(1)), (A, B));
    }

    #[test]
    fn comis_unmasked_invalid_faults_before_changing_eflags() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = lanes32([QNAN_SINGLE, 0, 0, 0]);
        rig.h.cpu.xmm[1] = singles([1.0, 0.0, 0.0, 0.0]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_INVALID << MXCSR_MASK_SHIFT);
        rig.h.cpu.set_eflags_arith(CF);
        assert_eq!(rig.exec("comiss xmm0, xmm1"), xm());
        assert_eq!(rig.flags(), CF);
        // ucomiss does not raise it for a QNaN
        assert_eq!(rig.exec("ucomiss xmm0, xmm1"), Stop::Halted);
        assert_eq!(rig.flags(), ZF | PF | CF);
    }

    const COMPARE_DEST_SINGLES: [u32; 4] = [ONE_SINGLE, ONE_SINGLE, TWO_SINGLE, QNAN_SINGLE];
    const COMPARE_SOURCE_SINGLES: [u32; 4] = [TWO_SINGLE, ONE_SINGLE, ONE_SINGLE, ONE_SINGLE];
    /// For each predicate 0-7: result for less, equal, greater, unordered.
    const PREDICATE_TRUTH: [[bool; 4]; 8] = [
        [false, true, false, false],
        [true, false, false, false],
        [true, true, false, false],
        [false, false, false, true],
        [true, false, true, true],
        [false, true, true, true],
        [false, false, true, true],
        [true, true, true, false],
    ];

    fn mask32(set: bool) -> u32 {
        if set { u32::MAX } else { 0 }
    }
    fn mask64(set: bool) -> u64 {
        if set { u64::MAX } else { 0 }
    }

    #[test]
    fn cmpps_produces_all_ones_or_zero_per_lane_for_every_predicate() {
        let mut rig = Rig::new();
        for (predicate, truth) in PREDICATE_TRUTH.iter().enumerate() {
            let got = rig.op(
                &format!("cmpps xmm0, xmm1, {predicate}"),
                lanes32(COMPARE_DEST_SINGLES),
                lanes32(COMPARE_SOURCE_SINGLES),
            );
            assert_eq!(dwords(got), truth.map(mask32), "predicate {predicate}");
        }
    }

    #[test]
    fn cmppd_produces_all_ones_or_zero_per_lane_for_every_predicate() {
        let mut rig = Rig::new();
        let dest = [
            [1.0f64.to_bits(), 1.0f64.to_bits()],
            [2.0f64.to_bits(), QNAN_DOUBLE],
        ];
        let source = [
            [2.0f64.to_bits(), 1.0f64.to_bits()],
            [1.0f64.to_bits(), 1.0f64.to_bits()],
        ];
        for (predicate, truth) in PREDICATE_TRUTH.iter().enumerate() {
            for half in 0..2 {
                let got = rig.op(
                    &format!("cmppd xmm0, xmm1, {predicate}"),
                    lanes64(dest[half]),
                    lanes64(source[half]),
                );
                assert_eq!(
                    qwords(got),
                    [mask64(truth[2 * half]), mask64(truth[2 * half + 1])],
                    "predicate {predicate}"
                );
            }
        }
    }

    #[test]
    fn cmpss_and_cmpsd_compare_only_the_low_lane_and_keep_the_rest() {
        let mut rig = Rig::new();
        for (predicate, truth) in PREDICATE_TRUTH.iter().enumerate() {
            for (column, (a, b)) in COMPARE_DEST_SINGLES
                .iter()
                .zip(COMPARE_SOURCE_SINGLES)
                .enumerate()
            {
                let dst = lanes32([*a, 0x1111_1111, 0x2222_2222, 0x3333_3333]);
                let src = lanes32([b, 0, 0, 0]);
                let got = rig.op(&format!("cmpss xmm0, xmm1, {predicate}"), dst, src);
                assert_eq!(
                    dwords(got),
                    [mask32(truth[column]), 0x1111_1111, 0x2222_2222, 0x3333_3333]
                );
            }
            let dest_doubles = [
                1.0f64.to_bits(),
                1.0f64.to_bits(),
                2.0f64.to_bits(),
                QNAN_DOUBLE,
            ];
            let source_doubles = [
                2.0f64.to_bits(),
                1.0f64.to_bits(),
                1.0f64.to_bits(),
                1.0f64.to_bits(),
            ];
            for column in 0..4 {
                let got = rig.op(
                    &format!("cmpsd xmm0, xmm1, {predicate}"),
                    lanes64([dest_doubles[column], 0x4444_4444_4444_4444]),
                    lanes64([source_doubles[column], 0]),
                );
                assert_eq!(qwords(got), [mask64(truth[column]), 0x4444_4444_4444_4444]);
            }
        }
    }

    #[test]
    fn cmp_memory_forms() {
        let mut rig = Rig::new();
        let got = rig.op_mem(
            "cmpps xmm0, xmmword ptr [rsi], 1",
            lanes32(COMPARE_DEST_SINGLES),
            lanes32(COMPARE_SOURCE_SINGLES),
        );
        assert_eq!(dwords(got), PREDICATE_TRUTH[1].map(mask32));
        let got = rig.op_mem(
            "cmpss xmm0, dword ptr [rsi], 1",
            lanes32(COMPARE_DEST_SINGLES),
            lanes32(COMPARE_SOURCE_SINGLES),
        );
        assert_eq!(dwords(got)[..2], [mask32(true), ONE_SINGLE]);
        let got = rig.op_mem(
            "cmpsd xmm0, qword ptr [rsi], 1",
            doubles([1.0, 5.0]),
            doubles([2.0, 0.0]),
        );
        assert_eq!(qwords(got), [u64::MAX, 5.0f64.to_bits()]);
        let got = rig.op_mem(
            "cmppd xmm0, xmmword ptr [rsi], 1",
            doubles([1.0, 5.0]),
            doubles([2.0, 0.0]),
        );
        assert_eq!(qwords(got), [u64::MAX, 0]);
    }

    #[test]
    fn cmp_predicate_immediate_above_seven_wraps_for_sse() {
        let mut rig = Rig::new();
        let got = rig.op(
            "cmpps xmm0, xmm1, 8",
            lanes32(COMPARE_DEST_SINGLES),
            lanes32(COMPARE_SOURCE_SINGLES),
        );
        assert_eq!(dwords(got), PREDICATE_TRUTH[0].map(mask32));
    }

    #[test]
    fn cmp_flags_invalid_only_for_signaling_predicates_with_quiet_nans() {
        let mut rig = Rig::new();
        for predicate in 0..8u32 {
            rig.op(
                &format!("cmpps xmm0, xmm1, {predicate}"),
                lanes32([QNAN_SINGLE; 4]),
                singles([1.0; 4]),
            );
            let expected = if matches!(predicate, 1 | 2 | 5 | 6) {
                FLAG_INVALID
            } else {
                0
            };
            assert_eq!(
                rig.mxcsr() & MXCSR_EXCEPTION_FLAGS,
                expected,
                "predicate {predicate}"
            );
        }
    }

    #[test]
    fn cmp_with_negative_zero_equals_positive_zero() {
        let mut rig = Rig::new();
        let got = rig.op(
            "cmpps xmm0, xmm1, 0",
            lanes32([0, 1 << 31, 0, 1 << 31]),
            lanes32([1 << 31, 0, 0, 1 << 31]),
        );
        assert_eq!(dwords(got), [u32::MAX; 4]);
    }

    #[test]
    fn cvtps2pd_widens_the_two_low_singles() {
        let mut rig = Rig::new();
        let got = rig.op("cvtps2pd xmm0, xmm1", A, singles([1.5, -2.25, 9.0, 9.0]));
        assert_eq!(got, doubles([1.5, -2.25]));
        let got = rig.op_mem(
            "cvtps2pd xmm0, qword ptr [rsi]",
            A,
            singles([0.1, 1e10, 5.0, 5.0]),
        );
        assert_eq!(got, doubles([0.1f32 as f64, 1e10f32 as f64]));
    }

    #[test]
    fn cvtpd2ps_narrows_and_zeroes_the_upper_half() {
        let mut rig = Rig::new();
        let got = rig.op("cvtpd2ps xmm0, xmm1", A, doubles([1.5, -2.25]));
        assert_eq!(got, singles([1.5, -2.25, 0.0, 0.0]));
        let got = rig.op_mem("cvtpd2ps xmm0, xmmword ptr [rsi]", A, doubles([0.1, 1e300]));
        assert_eq!(
            dwords(got),
            [(0.1f64 as f32).to_bits(), INFINITY_SINGLE, 0, 0]
        );
        expect_flag(&rig, FLAG_OVERFLOW | FLAG_PRECISION);
    }

    #[test]
    fn cvtss2sd_and_cvtsd2ss_merge_into_the_destination() {
        let mut rig = Rig::new();
        let got = rig.op("cvtss2sd xmm0, xmm1", A, singles([2.5, 7.0, 7.0, 7.0]));
        assert_eq!(got, (A & !LOW_64) | (2.5f64.to_bits() as u128));
        let got = rig.op_mem(
            "cvtss2sd xmm0, dword ptr [rsi]",
            A,
            singles([-0.5, 7.0, 7.0, 7.0]),
        );
        assert_eq!(got, (A & !LOW_64) | ((-0.5f64).to_bits() as u128));
        let got = rig.op("cvtsd2ss xmm0, xmm1", A, doubles([2.5, 7.0]));
        assert_eq!(got, (A & !LOW_32) | (2.5f32.to_bits() as u128));
        let got = rig.op_mem("cvtsd2ss xmm0, qword ptr [rsi]", A, doubles([0.1, 7.0]));
        assert_eq!(got, (A & !LOW_32) | ((0.1f64 as f32).to_bits() as u128));
    }

    #[test]
    fn cvt_between_float_formats_propagates_and_quiets_nans() {
        let mut rig = Rig::new();
        let got = rig.op("cvtss2sd xmm0, xmm1", 0, lanes32([SNAN_SINGLE, 0, 0, 0]));
        assert_eq!(qwords(got)[0], 0x7ff8_0000_2000_0000);
        expect_flag(&rig, FLAG_INVALID);
        let got = rig.op("cvtsd2ss xmm0, xmm1", 0, lanes64([SNAN_DOUBLE, 0]));
        assert_eq!(dwords(got)[0], 0x7fc0_0000);
        let got = rig.op(
            "cvtps2pd xmm0, xmm1",
            0,
            lanes32([QNAN_SINGLE, 0xffc0_0000, 0, 0]),
        );
        assert_eq!(
            qwords(got),
            [0x7ff8_0000_0000_0000 | (1 << 29), 0xfff8_0000_0000_0000]
        );
    }

    #[test]
    fn cvtdq2ps_converts_signed_dwords_and_rounds_once() {
        let mut rig = Rig::new();
        let got = rig.op(
            "cvtdq2ps xmm0, xmm1",
            A,
            lanes32([1, u32::MAX, FIRST_INEXACT_INTEGER, 0x8000_0000]),
        );
        assert_eq!(got, singles([1.0, -1.0, 16_777_216.0, -2147483648.0]));
        expect_flag(&rig, FLAG_PRECISION);
        let got = rig.op_mem(
            "cvtdq2ps xmm0, xmmword ptr [rsi]",
            A,
            lanes32([0, 2, 3, 0x7fff_ffff]),
        );
        assert_eq!(got, singles([0.0, 2.0, 3.0, 2147483648.0]));
    }

    #[test]
    fn cvtdq2ps_honors_the_rounding_control() {
        let mut rig = Rig::new();
        for (rounding, expected) in [
            (
                RC_NEAREST,
                [TWO_POW_24_SINGLE, (TWO_POW_24_SINGLE | SIGN_SINGLE)],
            ),
            (
                RC_DOWN,
                [TWO_POW_24_SINGLE, (TWO_POW_24_NEXT_SINGLE | SIGN_SINGLE)],
            ),
            (
                RC_UP,
                [TWO_POW_24_NEXT_SINGLE, (TWO_POW_24_SINGLE | SIGN_SINGLE)],
            ),
            (
                RC_ZERO,
                [TWO_POW_24_SINGLE, (TWO_POW_24_SINGLE | SIGN_SINGLE)],
            ),
        ] {
            rig.h.cpu.xmm[1] = lanes32([
                FIRST_INEXACT_INTEGER,
                FIRST_INEXACT_INTEGER.wrapping_neg(),
                0,
                0,
            ]);
            rig.reset();
            rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
            assert_eq!(rig.exec("cvtdq2ps xmm0, xmm1"), Stop::Halted);
            assert_eq!(dwords(rig.xmm(0))[..2], expected, "rc {rounding}");
        }
    }

    #[test]
    fn cvtdq2pd_converts_the_two_low_dwords_exactly() {
        let mut rig = Rig::new();
        let got = rig.op(
            "cvtdq2pd xmm0, xmm1",
            A,
            lanes32([(-5i32) as u32, 7, 99, 99]),
        );
        assert_eq!(got, doubles([-5.0, 7.0]));
        let got = rig.op_mem(
            "cvtdq2pd xmm0, qword ptr [rsi]",
            A,
            lanes32([0x8000_0000, 0x7fff_ffff, 99, 99]),
        );
        assert_eq!(got, doubles([-2147483648.0, 2147483647.0]));
        assert_eq!(rig.mxcsr(), MXCSR_RESET, "always exact");
    }

    fn run_with_rc(rig: &mut Rig, asm: &str, rounding: u32, dst: u128, src: u128) -> u128 {
        rig.h.cpu.xmm[0] = dst;
        rig.h.cpu.xmm[1] = src;
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
        assert_eq!(rig.exec(asm), Stop::Halted, "{asm}");
        rig.xmm(0)
    }

    #[test]
    fn cvtps2dq_rounds_per_mxcsr_and_cvttps2dq_truncates() {
        let mut rig = Rig::new();
        let input = singles([1.5, 2.5, -1.5, -2.5]);
        for (rounding, expected) in [
            (RC_NEAREST, [2i32, 2, -2, -2]),
            (RC_DOWN, [1, 2, -2, -3]),
            (RC_UP, [2, 3, -1, -2]),
            (RC_ZERO, [1, 2, -1, -2]),
        ] {
            let got = run_with_rc(&mut rig, "cvtps2dq xmm0, xmm1", rounding, A, input);
            assert_eq!(
                dwords(got),
                expected.map(|v| v as u32),
                "cvtps2dq rc {rounding}"
            );
            let got = run_with_rc(&mut rig, "cvttps2dq xmm0, xmm1", rounding, A, input);
            assert_eq!(
                dwords(got),
                [1, 2, -1i32 as u32, -2i32 as u32],
                "cvttps2dq rc {rounding}"
            );
        }
    }

    #[test]
    fn cvtps2dq_out_of_range_and_nan_give_the_integer_indefinite() {
        let mut rig = Rig::new();
        let input = lanes32([
            f32::MAX.to_bits(),
            QNAN_SINGLE,
            INFINITY_SINGLE | SIGN_SINGLE,
            0xcf00_0000,
        ]);
        let got = rig.op("cvtps2dq xmm0, xmm1", A, input);
        assert_eq!(
            dwords(got),
            [
                0x8000_0000,
                0x8000_0000,
                0x8000_0000,
                0x8000_0000 /* -2^31 is valid */
            ]
        );
        expect_flag(&rig, FLAG_INVALID);
        let got = rig.op(
            "cvttps2dq xmm0, xmm1",
            A,
            singles([2147483648.0, -2147483904.0, 2147483520.0, -2147483648.0]),
        );
        assert_eq!(
            dwords(got),
            [0x8000_0000, 0x8000_0000, 2147483520, 0x8000_0000]
        );
    }

    #[test]
    fn cvtps2dq_precision_flag_only_for_inexact_in_range_lanes() {
        let mut rig = Rig::new();
        rig.op("cvtps2dq xmm0, xmm1", A, singles([1.0, 2.0, -3.0, 4.0]));
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, 0);
        rig.op("cvtps2dq xmm0, xmm1", A, singles([1.0, 2.5, -3.0, 4.0]));
        assert_eq!(rig.mxcsr() & MXCSR_EXCEPTION_FLAGS, FLAG_PRECISION);
    }

    #[test]
    fn cvtpd2dq_and_cvttpd2dq_fill_the_low_two_dwords_and_zero_the_rest() {
        let mut rig = Rig::new();
        let input = doubles([2.5, -2.5]);
        for (rounding, expected) in [
            (RC_NEAREST, [2i32, -2]),
            (RC_DOWN, [2, -3]),
            (RC_UP, [3, -2]),
            (RC_ZERO, [2, -2]),
        ] {
            let got = run_with_rc(&mut rig, "cvtpd2dq xmm0, xmm1", rounding, A, input);
            assert_eq!(
                dwords(got),
                [expected[0] as u32, expected[1] as u32, 0, 0],
                "rc {rounding}"
            );
            let got = run_with_rc(&mut rig, "cvttpd2dq xmm0, xmm1", rounding, A, input);
            assert_eq!(
                dwords(got),
                [2, -2i32 as u32, 0, 0],
                "cvttpd2dq rc {rounding}"
            );
        }
        let got = rig.op("cvtpd2dq xmm0, xmm1", A, doubles([1e10, f64::NAN]));
        assert_eq!(dwords(got), [0x8000_0000, 0x8000_0000, 0, 0]);
        let got = rig.op_mem(
            "cvttpd2dq xmm0, xmmword ptr [rsi]",
            A,
            doubles([-1e10, 7.9]),
        );
        assert_eq!(dwords(got), [0x8000_0000, 7, 0, 0]);
    }

    #[test]
    fn cvtsi2ss_and_cvtsi2sd_convert_signed_integers() {
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = 0xffff_ffff_ffff_fffd;
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtsi2ss xmm0, eax");
        assert_eq!(
            rig.xmm(0),
            (A & !LOW_32) | ((-3.0f32).to_bits() as u128),
            "the 32-bit form sign-extends eax only"
        );
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtsi2sd xmm0, eax");
        assert_eq!(rig.xmm(0), (A & !LOW_64) | ((-3.0f64).to_bits() as u128));
        rig.h.cpu.gpr[RAX] = 0x1_0000_0000;
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtsi2sd xmm0, rax");
        assert_eq!(
            rig.xmm(0),
            (A & !LOW_64) | ((4294967296.0f64).to_bits() as u128)
        );
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtsi2ss xmm0, rax");
        assert_eq!(
            rig.xmm(0),
            (A & !LOW_32) | ((4294967296.0f32).to_bits() as u128)
        );
        rig.h.cpu.gpr[10] = (-7i64) as u64;
        rig.h.cpu.xmm[9] = A;
        rig.run_ok("cvtsi2sd xmm9, r10");
        assert_eq!(rig.xmm(9), (A & !LOW_64) | ((-7.0f64).to_bits() as u128));
    }

    #[test]
    fn cvtsi2ss_memory_forms() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &(-9i32).to_le_bytes());
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtsi2ss xmm0, dword ptr [rsi]");
        assert_eq!(rig.xmm(0) & LOW_32, (-9.0f32).to_bits() as u128);
        rig.h.write(DATA, &(1i64 << 40).to_le_bytes());
        rig.run_ok("cvtsi2sd xmm0, qword ptr [rsi]");
        assert_eq!(qwords(rig.xmm(0))[0], (1099511627776.0f64).to_bits());
        rig.h.write(DATA, &(-9i32).to_le_bytes());
        rig.run_ok("cvtsi2sd xmm0, dword ptr [rsi]");
        assert_eq!(qwords(rig.xmm(0))[0], (-9.0f64).to_bits());
        rig.h.write(DATA, &(1i64 << 40).to_le_bytes());
        rig.run_ok("cvtsi2ss xmm0, qword ptr [rsi]");
        assert_eq!(dwords(rig.xmm(0))[0], (1099511627776.0f32).to_bits());
    }

    #[test]
    fn cvtsi2ss_of_a_64_bit_integer_rounds_once() {
        // 2^60 + 2^36 + 1: via f64 the +1 is lost and the tie rounds down to 2^60.
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = 0x1000_0010_0000_0001;
        rig.h.cpu.xmm[0] = 0;
        rig.run_ok("cvtsi2ss xmm0, rax");
        assert_eq!(dwords(rig.xmm(0))[0], 0x5d80_0001);
        expect_flag(&rig, FLAG_PRECISION);
    }

    #[test]
    fn cvtsi2sd_of_a_64_bit_integer_rounds_once_per_mode() {
        let mut rig = Rig::new();
        for (rounding, expected) in [
            (RC_NEAREST, 0x43e0_0000_0000_0000u64),
            (RC_DOWN, 0x43df_ffff_ffff_ffff),
            (RC_UP, 0x43e0_0000_0000_0000),
            (RC_ZERO, 0x43df_ffff_ffff_ffff),
        ] {
            rig.h.cpu.gpr[RAX] = i64::MAX as u64;
            rig.reset();
            rig.h.cpu.gpr[RAX] = i64::MAX as u64;
            rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
            assert_eq!(rig.exec("cvtsi2sd xmm0, rax"), Stop::Halted);
            assert_eq!(qwords(rig.xmm(0))[0], expected, "rc {rounding}");
        }
    }

    #[test]
    fn cvtsi2ss_32_bit_rounds_per_mxcsr() {
        let mut rig = Rig::new();
        for (rounding, expected) in [
            (RC_NEAREST, TWO_POW_24_SINGLE),
            (RC_DOWN, TWO_POW_24_SINGLE),
            (RC_UP, TWO_POW_24_NEXT_SINGLE),
        ] {
            rig.reset();
            rig.h.cpu.gpr[RAX] = FIRST_INEXACT_INTEGER as u64;
            rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
            assert_eq!(rig.exec("cvtsi2ss xmm0, eax"), Stop::Halted);
            assert_eq!(dwords(rig.xmm(0))[0], expected, "rc {rounding}");
        }
    }

    #[test]
    fn cvtsi2sd_is_always_exact_for_32_bit_sources() {
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = i32::MIN as u32 as u64;
        rig.run_ok("cvtsi2sd xmm0, eax");
        assert_eq!(qwords(rig.xmm(0))[0], (-2147483648.0f64).to_bits());
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    fn cvt_to_gpr(rig: &mut Rig, asm: &str, rounding: u32, source: u128) -> u64 {
        rig.h.cpu.xmm[1] = source;
        rig.reset();
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
        assert_eq!(rig.exec(asm), Stop::Halted, "{asm}");
        rig.gpr(RAX)
    }

    #[test]
    fn cvtss2si_rounds_per_mxcsr_and_cvttss2si_truncates() {
        let mut rig = Rig::new();
        for (value, nearest, down, up, truncated) in [
            (2.5f32, 2i64, 2i64, 3i64, 2i64),
            (-2.5, -2, -3, -2, -2),
            (3.5, 4, 3, 4, 3),
            (0.75, 1, 0, 1, 0),
            (-0.75, -1, -1, 0, 0),
            (7.0, 7, 7, 7, 7),
        ] {
            let source = singles([value, 0.0, 0.0, 0.0]);
            for (rounding, expected) in [
                (RC_NEAREST, nearest),
                (RC_DOWN, down),
                (RC_UP, up),
                (RC_ZERO, truncated),
            ] {
                assert_eq!(
                    cvt_to_gpr(&mut rig, "cvtss2si eax, xmm1", rounding, source),
                    expected as i32 as u32 as u64,
                    "{value} rc {rounding}"
                );
                assert_eq!(
                    cvt_to_gpr(&mut rig, "cvtss2si rax, xmm1", rounding, source),
                    expected as u64,
                    "{value} rc {rounding}"
                );
                assert_eq!(
                    cvt_to_gpr(&mut rig, "cvttss2si eax, xmm1", rounding, source),
                    truncated as i32 as u32 as u64,
                    "{value} rc {rounding}"
                );
                assert_eq!(
                    cvt_to_gpr(&mut rig, "cvttss2si rax, xmm1", rounding, source),
                    truncated as u64
                );
            }
        }
    }

    #[test]
    fn cvtsd2si_rounds_per_mxcsr_and_cvttsd2si_truncates() {
        let mut rig = Rig::new();
        let source = doubles([-2.5, 0.0]);
        for (rounding, expected) in [
            (RC_NEAREST, -2i64),
            (RC_DOWN, -3),
            (RC_UP, -2),
            (RC_ZERO, -2),
        ] {
            assert_eq!(
                cvt_to_gpr(&mut rig, "cvtsd2si eax, xmm1", rounding, source),
                expected as i32 as u32 as u64
            );
            assert_eq!(
                cvt_to_gpr(&mut rig, "cvtsd2si rax, xmm1", rounding, source),
                expected as u64
            );
        }
        assert_eq!(
            cvt_to_gpr(&mut rig, "cvttsd2si rax, xmm1", RC_UP, doubles([1e18, 0.0])),
            1_000_000_000_000_000_000
        );
        assert_eq!(
            cvt_to_gpr(
                &mut rig,
                "cvttsd2si rax, xmm1",
                RC_UP,
                doubles([-1e18, 0.0])
            ),
            (-1_000_000_000_000_000_000i64) as u64
        );
    }

    #[test]
    fn float_to_int_conversions_write_the_indefinite_for_nan_and_overflow() {
        let mut rig = Rig::new();
        for source in [
            singles([f32::NAN, 0.0, 0.0, 0.0]),
            singles([3e9, 0.0, 0.0, 0.0]),
            singles([f32::NEG_INFINITY, 0.0, 0.0, 0.0]),
        ] {
            for asm in ["cvtss2si eax, xmm1", "cvttss2si eax, xmm1"] {
                assert_eq!(
                    cvt_to_gpr(&mut rig, asm, RC_NEAREST, source),
                    0x8000_0000,
                    "{asm}: written as a 32-bit value"
                );
                expect_flag(&rig, FLAG_INVALID);
            }
            for asm in ["cvtss2si rax, xmm1", "cvttss2si rax, xmm1"] {
                let expected = if f32::from_bits(dwords(source)[0]) == 3e9 {
                    3_000_000_000
                } else {
                    0x8000_0000_0000_0000
                };
                assert_eq!(
                    cvt_to_gpr(&mut rig, asm, RC_NEAREST, source),
                    expected,
                    "{asm}"
                );
            }
        }
        for asm in ["cvtsd2si eax, xmm1", "cvttsd2si eax, xmm1"] {
            assert_eq!(
                cvt_to_gpr(&mut rig, asm, RC_NEAREST, doubles([2147483648.0, 0.0])),
                0x8000_0000,
                "{asm}"
            );
        }
        for asm in ["cvtsd2si rax, xmm1", "cvttsd2si rax, xmm1"] {
            assert_eq!(
                cvt_to_gpr(
                    &mut rig,
                    asm,
                    RC_NEAREST,
                    doubles([9223372036854775808.0, 0.0])
                ),
                1 << 63,
                "{asm}"
            );
            assert_eq!(
                cvt_to_gpr(&mut rig, asm, RC_NEAREST, lanes64([QNAN_DOUBLE, 0])),
                1 << 63,
                "{asm}"
            );
        }
    }

    #[test]
    fn cvt_to_gpr_memory_and_extended_register_forms() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &(-3.75f32).to_le_bytes());
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("cvttss2si eax, dword ptr [rsi]");
        assert_eq!(rig.gpr(RAX), (-3i32) as u32 as u64);
        rig.run_ok("cvtss2si rax, dword ptr [rsi]");
        assert_eq!(rig.gpr(RAX), (-4i64) as u64);
        rig.h.write(DATA, &(1e12f64).to_le_bytes());
        rig.run_ok("cvtsd2si rax, qword ptr [rsi]");
        assert_eq!(rig.gpr(RAX), 1_000_000_000_000);
        rig.run_ok("cvtsd2si eax, qword ptr [rsi]");
        assert_eq!(rig.gpr(RAX), 0x8000_0000);
        rig.h.cpu.xmm[10] = doubles([9.99, 0.0]);
        rig.h.cpu.gpr[9] = u64::MAX;
        rig.run_ok("cvttsd2si r9d, xmm10");
        assert_eq!(rig.gpr(9), 9);
    }

    #[test]
    fn cvt_to_gpr_unmasked_precision_faults_without_writing_the_register() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[1] = singles([2.5, 0.0, 0.0, 0.0]);
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_PRECISION << MXCSR_MASK_SHIFT);
        rig.h.cpu.gpr[RAX] = 0x1234;
        assert_eq!(rig.exec("cvtss2si eax, xmm1"), xm());
        assert_eq!(rig.gpr(RAX), 0x1234);
        // an exact value does not trap
        rig.h.cpu.xmm[1] = singles([2.0, 0.0, 0.0, 0.0]);
        assert_eq!(rig.exec("cvtss2si eax, xmm1"), Stop::Halted);
        assert_eq!(rig.gpr(RAX), 2);
    }

    #[test]
    fn cvtsi2ss_unmasked_precision_faults_without_writing_xmm() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_PRECISION << MXCSR_MASK_SHIFT);
        rig.h.cpu.gpr[RAX] = FIRST_INEXACT_INTEGER as u64;
        rig.h.cpu.xmm[0] = A;
        assert_eq!(rig.exec("cvtsi2ss xmm0, eax"), xm());
        assert_eq!(rig.xmm(0), A);
        expect_flag(&rig, FLAG_PRECISION);
    }

    #[test]
    fn shufps_selects_two_lanes_from_each_operand() {
        let mut rig = Rig::new();
        let a = [0xa0a0_a0a0u32, 0xa1a1_a1a1, 0xa2a2_a2a2, 0xa3a3_a3a3];
        let b = [0xb0b0_b0b0u32, 0xb1b1_b1b1, 0xb2b2_b2b2, 0xb3b3_b3b3];
        for imm in [0u8, 0x1b, 0xe4, 0xff, 0x4e, 0x39, 0x93] {
            let expected = [
                a[(imm & 3) as usize],
                a[((imm >> 2) & 3) as usize],
                b[((imm >> 4) & 3) as usize],
                b[((imm >> 6) & 3) as usize],
            ];
            let got = rig.op(
                &format!("shufps xmm0, xmm1, {}", imm_text(imm)),
                lanes32(a),
                lanes32(b),
            );
            assert_eq!(dwords(got), expected, "imm {imm:#x}");
        }
        let got = rig.op_mem(
            "shufps xmm0, xmmword ptr [rsi], 0x1b",
            lanes32(a),
            lanes32(b),
        );
        assert_eq!(dwords(got), [a[3], a[2], b[1], b[0]]);
    }

    fn imm_text(imm: u8) -> String {
        if imm == 0 {
            "0".into()
        } else {
            format!("{imm:#x}")
        }
    }

    #[test]
    fn shufpd_selects_one_double_from_each_operand() {
        let mut rig = Rig::new();
        let a = [0xaaaa_0000_0000_0001u64, 0xaaaa_0000_0000_0002];
        let b = [0xbbbb_0000_0000_0001u64, 0xbbbb_0000_0000_0002];
        for imm in 0..4usize {
            let got = rig.op(&format!("shufpd xmm0, xmm1, {imm}"), lanes64(a), lanes64(b));
            assert_eq!(qwords(got), [a[imm & 1], b[(imm >> 1) & 1]], "imm {imm}");
        }
        let got = rig.op_mem("shufpd xmm0, xmmword ptr [rsi], 1", lanes64(a), lanes64(b));
        assert_eq!(qwords(got), [a[1], b[0]]);
    }

    #[test]
    fn pshufd_permutes_dwords_of_the_source() {
        let mut rig = Rig::new();
        let source = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        for imm in [0u8, 0x1b, 0xe4, 0xff, 0x4e, 0x39, 0x93] {
            let expected: [u32; 4] =
                std::array::from_fn(|k| source[((imm >> (2 * k)) & 3) as usize]);
            let got = rig.op(
                &format!("pshufd xmm0, xmm1, {}", imm_text(imm)),
                A,
                lanes32(source),
            );
            assert_eq!(dwords(got), expected, "imm {imm:#x}");
        }
        let got = rig.op_mem("pshufd xmm0, xmmword ptr [rsi], 0x1b", A, lanes32(source));
        assert_eq!(dwords(got), [source[3], source[2], source[1], source[0]]);
    }

    #[test]
    fn pshufd_with_the_same_register_for_source_and_destination() {
        let mut rig = Rig::new();
        let source = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        rig.h.cpu.xmm[0] = lanes32(source);
        rig.run_ok("pshufd xmm0, xmm0, 0x1b");
        assert_eq!(
            dwords(rig.xmm(0)),
            [source[3], source[2], source[1], source[0]]
        );
    }

    #[test]
    fn pshuflw_permutes_the_low_words_and_copies_the_high_quadword() {
        let mut rig = Rig::new();
        let source = [
            0x0001u16, 0x0002, 0x0003, 0x0004, 0x1005, 0x1006, 0x1007, 0x1008,
        ];
        for imm in [0u8, 0x1b, 0xe4, 0xff, 0x4e, 0x39, 0x93] {
            let mut expected = source;
            for k in 0..4 {
                expected[k] = source[((imm >> (2 * k)) & 3) as usize];
            }
            let got = rig.op(
                &format!("pshuflw xmm0, xmm1, {}", imm_text(imm)),
                A,
                lanes16(source),
            );
            assert_eq!(words(got), expected, "imm {imm:#x}");
        }
    }

    #[test]
    fn pshufhw_permutes_the_high_words_and_copies_the_low_quadword() {
        let mut rig = Rig::new();
        let source = [
            0x0001u16, 0x0002, 0x0003, 0x0004, 0x1005, 0x1006, 0x1007, 0x1008,
        ];
        for imm in [0u8, 0x1b, 0xe4, 0xff, 0x4e, 0x39, 0x93] {
            let mut expected = source;
            for k in 0..4 {
                expected[4 + k] = source[4 + ((imm >> (2 * k)) & 3) as usize];
            }
            let got = rig.op(
                &format!("pshufhw xmm0, xmm1, {}", imm_text(imm)),
                A,
                lanes16(source),
            );
            assert_eq!(words(got), expected, "imm {imm:#x}");
        }
    }

    #[test]
    fn unpck_interleaves_singles_and_doubles() {
        let mut rig = Rig::new();
        let a = [0xa0u32, 0xa1, 0xa2, 0xa3];
        let b = [0xb0u32, 0xb1, 0xb2, 0xb3];
        assert_eq!(
            dwords(rig.op("unpcklps xmm0, xmm1", lanes32(a), lanes32(b))),
            [0xa0, 0xb0, 0xa1, 0xb1]
        );
        assert_eq!(
            dwords(rig.op("unpckhps xmm0, xmm1", lanes32(a), lanes32(b))),
            [0xa2, 0xb2, 0xa3, 0xb3]
        );
        assert_eq!(
            dwords(rig.op_mem("unpcklps xmm0, xmmword ptr [rsi]", lanes32(a), lanes32(b))),
            [0xa0, 0xb0, 0xa1, 0xb1]
        );
        assert_eq!(
            dwords(rig.op_mem("unpckhps xmm0, xmmword ptr [rsi]", lanes32(a), lanes32(b))),
            [0xa2, 0xb2, 0xa3, 0xb3]
        );
        let a = [0xa000u64, 0xa111];
        let b = [0xb000u64, 0xb111];
        assert_eq!(
            qwords(rig.op("unpcklpd xmm0, xmm1", lanes64(a), lanes64(b))),
            [0xa000, 0xb000]
        );
        assert_eq!(
            qwords(rig.op("unpckhpd xmm0, xmm1", lanes64(a), lanes64(b))),
            [0xa111, 0xb111]
        );
    }

    #[test]
    fn punpck_integer_interleaves() {
        let mut rig = Rig::new();
        let a: [u8; 16] = std::array::from_fn(|k| 0xa0 + k as u8);
        let b: [u8; 16] = std::array::from_fn(|k| 0xb0 + k as u8);
        let (va, vb) = (lanes8(a), lanes8(b));
        for (name, hi) in [("punpcklbw", false), ("punpckhbw", true)] {
            let base = if hi { 8 } else { 0 };
            let expected: [u8; 16] = std::array::from_fn(|k| {
                if k % 2 == 0 {
                    a[base + k / 2]
                } else {
                    b[base + k / 2]
                }
            });
            assert_eq!(
                bytes(rig.op(&format!("{name} xmm0, xmm1"), va, vb)),
                expected,
                "{name}"
            );
        }
        let (aw, bw) = (words(va), words(vb));
        for (name, hi) in [("punpcklwd", false), ("punpckhwd", true)] {
            let base = if hi { 4 } else { 0 };
            let expected: [u16; 8] = std::array::from_fn(|k| {
                if k % 2 == 0 {
                    aw[base + k / 2]
                } else {
                    bw[base + k / 2]
                }
            });
            assert_eq!(
                words(rig.op(&format!("{name} xmm0, xmm1"), va, vb)),
                expected,
                "{name}"
            );
        }
        let (ad, bd) = (dwords(va), dwords(vb));
        assert_eq!(
            dwords(rig.op("punpckldq xmm0, xmm1", va, vb)),
            [ad[0], bd[0], ad[1], bd[1]]
        );
        assert_eq!(
            dwords(rig.op("punpckhdq xmm0, xmm1", va, vb)),
            [ad[2], bd[2], ad[3], bd[3]]
        );
        let (aq, bq) = (qwords(va), qwords(vb));
        assert_eq!(
            qwords(rig.op("punpcklqdq xmm0, xmm1", va, vb)),
            [aq[0], bq[0]]
        );
        assert_eq!(
            qwords(rig.op("punpckhqdq xmm0, xmm1", va, vb)),
            [aq[1], bq[1]]
        );
        assert_eq!(
            qwords(rig.op_mem("punpckhqdq xmm0, xmmword ptr [rsi]", va, vb)),
            [aq[1], bq[1]]
        );
    }

    /// Vectors that exercise lane boundaries: all zero, all one, sign bits, the largest positive
    /// values, alternating bytes, and two irregular patterns.
    const INTERESTING: [u128; 8] = [
        0,
        u128::MAX,
        0x8080_8080_8080_8080_8080_8080_8080_8080,
        0x7f7f_7f7f_7f7f_7f7f_7f7f_7f7f_7f7f_7f7f,
        0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff,
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        0x8000_7fff_ffff_0001_0002_fffe_8001_7ffe,
        0x0000_0001_ffff_ffff_8000_0000_7fff_ffff,
    ];

    fn each_pair(mut check: impl FnMut(u128, u128)) {
        for a in INTERESTING {
            for b in INTERESTING {
                check(a, b);
            }
        }
    }

    fn from_byte_fn(f: impl Fn(usize) -> u8) -> u128 {
        lanes8(std::array::from_fn(f))
    }
    fn from_word_fn(f: impl Fn(usize) -> u16) -> u128 {
        lanes16(std::array::from_fn(f))
    }
    fn from_dword_fn(f: impl Fn(usize) -> u32) -> u128 {
        lanes32(std::array::from_fn(f))
    }
    fn from_qword_fn(f: impl Fn(usize) -> u64) -> u128 {
        lanes64(std::array::from_fn(f))
    }

    fn saturate_signed(value: i64, bits: u32) -> i64 {
        value.clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    }
    fn saturate_unsigned(value: i64, bits: u32) -> i64 {
        value.clamp(0, (1 << bits) - 1)
    }

    /// Every two-operand integer instruction paired with an independent lane-by-lane model.
    type LaneModel = Box<dyn Fn(u128, u128) -> u128>;

    fn integer_models() -> Vec<(&'static str, LaneModel)> {
        let mut models: Vec<(&'static str, LaneModel)> = Vec::new();
        macro_rules! model {
            ($name:expr, $f:expr) => {
                models.push(($name, Box::new($f)));
            };
        }
        model!("paddb", |a, b| from_byte_fn(
            |k| bytes(a)[k].wrapping_add(bytes(b)[k])
        ));
        model!("paddw", |a, b| from_word_fn(
            |k| words(a)[k].wrapping_add(words(b)[k])
        ));
        model!("paddd", |a, b| from_dword_fn(
            |k| dwords(a)[k].wrapping_add(dwords(b)[k])
        ));
        model!("paddq", |a, b| from_qword_fn(
            |k| qwords(a)[k].wrapping_add(qwords(b)[k])
        ));
        model!("psubb", |a, b| from_byte_fn(
            |k| bytes(a)[k].wrapping_sub(bytes(b)[k])
        ));
        model!("psubw", |a, b| from_word_fn(
            |k| words(a)[k].wrapping_sub(words(b)[k])
        ));
        model!("psubd", |a, b| from_dword_fn(
            |k| dwords(a)[k].wrapping_sub(dwords(b)[k])
        ));
        model!("psubq", |a, b| from_qword_fn(
            |k| qwords(a)[k].wrapping_sub(qwords(b)[k])
        ));
        model!("paddsb", |a, b| from_byte_fn(|k| saturate_signed(
            bytes(a)[k] as i8 as i64 + bytes(b)[k] as i8 as i64,
            8
        ) as i8 as u8));
        model!("paddsw", |a, b| from_word_fn(|k| saturate_signed(
            words(a)[k] as i16 as i64 + words(b)[k] as i16 as i64,
            16
        ) as i16 as u16));
        model!("paddusb", |a, b| from_byte_fn(|k| saturate_unsigned(
            bytes(a)[k] as i64 + bytes(b)[k] as i64,
            8
        ) as u8));
        model!("paddusw", |a, b| from_word_fn(|k| saturate_unsigned(
            words(a)[k] as i64 + words(b)[k] as i64,
            16
        ) as u16));
        model!("psubsb", |a, b| from_byte_fn(|k| saturate_signed(
            bytes(a)[k] as i8 as i64 - bytes(b)[k] as i8 as i64,
            8
        ) as i8 as u8));
        model!("psubsw", |a, b| from_word_fn(|k| saturate_signed(
            words(a)[k] as i16 as i64 - words(b)[k] as i16 as i64,
            16
        ) as i16 as u16));
        model!("psubusb", |a, b| from_byte_fn(|k| saturate_unsigned(
            bytes(a)[k] as i64 - bytes(b)[k] as i64,
            8
        ) as u8));
        model!("psubusw", |a, b| from_word_fn(|k| saturate_unsigned(
            words(a)[k] as i64 - words(b)[k] as i64,
            16
        ) as u16));
        model!("pmullw", |a, b| from_word_fn(
            |k| ((words(a)[k] as i16 as i32 * words(b)[k] as i16 as i32) & 0xffff) as u16
        ));
        model!("pmulhw", |a, b| from_word_fn(
            |k| ((words(a)[k] as i16 as i32 * words(b)[k] as i16 as i32) >> 16) as u16
        ));
        model!("pmulhuw", |a, b| from_word_fn(
            |k| ((words(a)[k] as u32 * words(b)[k] as u32) >> 16) as u16
        ));
        model!("pmuludq", |a, b| from_qword_fn(
            |k| dwords(a)[2 * k] as u64 * dwords(b)[2 * k] as u64
        ));
        model!("pmaddwd", |a, b| from_dword_fn(|k| {
            let products = [0, 1]
                .map(|j| words(a)[2 * k + j] as i16 as i64 * words(b)[2 * k + j] as i16 as i64);
            ((products[0] + products[1]) & 0xffff_ffff) as u32
        }));
        model!("pavgb", |a, b| from_byte_fn(
            |k| (bytes(a)[k] as u32 + bytes(b)[k] as u32).div_ceil(2) as u8
        ));
        model!("pavgw", |a, b| from_word_fn(
            |k| (words(a)[k] as u32 + words(b)[k] as u32).div_ceil(2) as u16
        ));
        model!("pminub", |a, b| from_byte_fn(
            |k| bytes(a)[k].min(bytes(b)[k])
        ));
        model!("pmaxub", |a, b| from_byte_fn(
            |k| bytes(a)[k].max(bytes(b)[k])
        ));
        model!("pminsw", |a, b| from_word_fn(
            |k| (words(a)[k] as i16).min(words(b)[k] as i16) as u16
        ));
        model!("pmaxsw", |a, b| from_word_fn(
            |k| (words(a)[k] as i16).max(words(b)[k] as i16) as u16
        ));
        model!("psadbw", |a, b| from_qword_fn(|k| (0..8)
            .map(
                |j| (bytes(a)[8 * k + j] as i32 - bytes(b)[8 * k + j] as i32).unsigned_abs() as u64
            )
            .sum()));
        model!("pand", |a, b| a & b);
        model!("pandn", |a, b| !a & b);
        model!("por", |a, b| a | b);
        model!("pxor", |a, b| a ^ b);
        model!("pcmpeqb", |a, b| from_byte_fn(|k| {
            if bytes(a)[k] == bytes(b)[k] { 0xff } else { 0 }
        }));
        model!("pcmpeqw", |a, b| from_word_fn(|k| {
            if words(a)[k] == words(b)[k] {
                0xffff
            } else {
                0
            }
        }));
        model!("pcmpeqd", |a, b| from_dword_fn(|k| {
            if dwords(a)[k] == dwords(b)[k] {
                u32::MAX
            } else {
                0
            }
        }));
        model!("pcmpgtb", |a, b| from_byte_fn(|k| {
            if (bytes(a)[k] as i8) > (bytes(b)[k] as i8) {
                0xff
            } else {
                0
            }
        }));
        model!("pcmpgtw", |a, b| from_word_fn(|k| {
            if (words(a)[k] as i16) > (words(b)[k] as i16) {
                0xffff
            } else {
                0
            }
        }));
        model!("pcmpgtd", |a, b| from_dword_fn(|k| {
            if (dwords(a)[k] as i32) > (dwords(b)[k] as i32) {
                u32::MAX
            } else {
                0
            }
        }));
        model!("packsswb", |a, b| from_byte_fn(|k| {
            let source = if k < 8 { words(a)[k] } else { words(b)[k - 8] };
            saturate_signed(source as i16 as i64, 8) as i8 as u8
        }));
        model!("packuswb", |a, b| from_byte_fn(|k| {
            let source = if k < 8 { words(a)[k] } else { words(b)[k - 8] };
            saturate_unsigned(source as i16 as i64, 8) as u8
        }));
        model!("packssdw", |a, b| from_word_fn(|k| {
            let source = if k < 4 {
                dwords(a)[k]
            } else {
                dwords(b)[k - 4]
            };
            saturate_signed(source as i32 as i64, 16) as i16 as u16
        }));
        models
    }

    #[test]
    fn integer_instructions_match_their_lane_models_for_register_operands() {
        let mut rig = Rig::new();
        for (name, model) in integer_models() {
            let asm = format!("{name} xmm0, xmm1");
            each_pair(|a, b| {
                assert_eq!(
                    rig.op(&asm, a, b),
                    model(a, b),
                    "{name} {a:#034x} {b:#034x}"
                );
                assert_eq!(rig.xmm(1), b, "{name} leaves the source");
            });
        }
    }

    #[test]
    fn integer_instructions_match_their_lane_models_for_memory_operands() {
        let mut rig = Rig::new();
        for (name, model) in integer_models() {
            let asm = format!("{name} xmm0, xmmword ptr [rsi]");
            each_pair(|a, b| {
                assert_eq!(
                    rig.op_mem(&asm, a, b),
                    model(a, b),
                    "{name} {a:#034x} {b:#034x}"
                );
            });
        }
    }

    #[test]
    fn integer_instructions_on_extended_registers() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[9] = A;
        rig.h.cpu.xmm[10] = B;
        rig.run_ok("paddb xmm9, xmm10");
        assert_eq!(
            rig.xmm(9),
            from_byte_fn(|k| bytes(A)[k].wrapping_add(bytes(B)[k]))
        );
    }

    #[test]
    fn integer_arithmetic_boundary_cases() {
        let mut rig = Rig::new();
        let lane = |v: u16| lanes16([v; 8]);
        assert_eq!(
            rig.op("paddsw xmm0, xmm1", lane(0x7fff), lane(1)),
            lane(0x7fff)
        );
        assert_eq!(
            rig.op("paddsw xmm0, xmm1", lane(0x8000), lane(0xffff)),
            lane(0x8000)
        );
        assert_eq!(
            rig.op("paddusw xmm0, xmm1", lane(0xffff), lane(1)),
            lane(0xffff)
        );
        assert_eq!(rig.op("psubusw xmm0, xmm1", lane(0), lane(1)), lane(0));
        assert_eq!(
            rig.op("psubsw xmm0, xmm1", lane(0x8000), lane(1)),
            lane(0x8000)
        );
        assert_eq!(
            rig.op("psubsw xmm0, xmm1", lane(0x7fff), lane(0xffff)),
            lane(0x7fff)
        );
        assert_eq!(rig.op("paddw xmm0, xmm1", lane(0xffff), lane(1)), lane(0));
        assert_eq!(
            rig.op("pmulhw xmm0, xmm1", lane(0x8000), lane(0x8000)),
            lane(0x4000)
        );
        assert_eq!(
            rig.op("pmulhuw xmm0, xmm1", lane(0xffff), lane(0xffff)),
            lane(0xfffe)
        );
        assert_eq!(
            rig.op("pmullw xmm0, xmm1", lane(0xffff), lane(0xffff)),
            lane(1)
        );
        assert_eq!(
            rig.op("pavgw xmm0, xmm1", lane(0xffff), lane(0xffff)),
            lane(0xffff)
        );
        assert_eq!(
            rig.op("pavgw xmm0, xmm1", lane(0), lane(1)),
            lane(1),
            "averages round up"
        );
        assert_eq!(rig.op("pavgb xmm0, xmm1", u128::MAX, u128::MAX), u128::MAX);
    }

    #[test]
    fn pmaddwd_overflow_wraps_only_for_the_most_negative_pair() {
        let mut rig = Rig::new();
        let got = rig.op(
            "pmaddwd xmm0, xmm1",
            lanes16([0x8000; 8]),
            lanes16([0x8000; 8]),
        );
        assert_eq!(dwords(got), [0x8000_0000; 4]);
        let got = rig.op(
            "pmaddwd xmm0, xmm1",
            lanes16([1, 2, 3, 4, 5, 6, 7, 8]),
            lanes16([10, 20, 30, 40, 50, 60, 70, 80]),
        );
        assert_eq!(dwords(got), [50, 250, 610, 1130]);
    }

    #[test]
    fn pmuludq_uses_only_the_even_dwords() {
        let mut rig = Rig::new();
        let got = rig.op(
            "pmuludq xmm0, xmm1",
            lanes32([0xffff_ffff, 99, 3, 99]),
            lanes32([0xffff_ffff, 77, 5, 77]),
        );
        assert_eq!(qwords(got), [0xffff_fffe_0000_0001, 15]);
    }

    #[test]
    fn psadbw_sums_absolute_byte_differences_per_quadword() {
        let mut rig = Rig::new();
        let a = lanes8([
            10, 20, 30, 40, 50, 60, 70, 80, 255, 0, 255, 0, 255, 0, 255, 0,
        ]);
        let b = lanes8([
            15, 15, 35, 35, 55, 55, 75, 75, 0, 255, 0, 255, 0, 255, 0, 255,
        ]);
        let got = rig.op("psadbw xmm0, xmm1", a, b);
        assert_eq!(qwords(got), [40, 8 * 255]);
    }

    #[test]
    fn pack_instructions_saturate_signed_words_and_dwords() {
        let mut rig = Rig::new();
        let a = lanes16([0, 1, 127, 128, 0xff80, 0xff7f, 0x7fff, 0x8000]);
        let b = lanes16([255, 256, 0xffff, 0xff00, 5, 6, 7, 8]);
        assert_eq!(
            bytes(rig.op("packsswb xmm0, xmm1", a, b)),
            [
                0, 1, 127, 127, 0x80, 0x80, 127, 0x80, 127, 127, 0xff, 0x80, 5, 6, 7, 8
            ]
        );
        assert_eq!(
            bytes(rig.op("packuswb xmm0, xmm1", a, b)),
            [0, 1, 127, 128, 0, 0, 255, 0, 255, 255, 0, 0, 5, 6, 7, 8]
        );
        let c = lanes32([0, 32767, 32768, 0xffff_8000]);
        let d = lanes32([0xffff_7fff, 0x7fff_ffff, 0x8000_0000, 1]);
        assert_eq!(
            words(rig.op("packssdw xmm0, xmm1", c, d)),
            [0, 0x7fff, 0x7fff, 0x8000, 0x8000, 0x7fff, 0x8000, 1]
        );
    }

    #[test]
    fn compare_instructions_produce_masks() {
        let mut rig = Rig::new();
        let got = rig.op(
            "pcmpgtb xmm0, xmm1",
            lanes8([0x7f, 0x80, 1, 0, 0xff, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            lanes8([0x80, 0x7f, 0, 1, 0, 5, 4, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]),
        );
        assert_eq!(
            &bytes(got)[..8],
            &[0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff],
            "compares are signed"
        );
        let got = rig.op(
            "pcmpeqd xmm0, xmm1",
            lanes32([1, 2, 3, 4]),
            lanes32([1, 0, 3, 0]),
        );
        assert_eq!(dwords(got), [u32::MAX, 0, u32::MAX, 0]);
        let got = rig.op(
            "pcmpgtw xmm0, xmm1",
            lanes16([0x8000, 1, 0, 0xffff, 0, 0, 0, 0]),
            lanes16([0x7fff, 0, 0xffff, 0x8000, 0, 0, 0, 0]),
        );
        assert_eq!(&words(got)[..4], &[0, 0xffff, 0xffff, 0xffff]);
        let got = rig.op(
            "pcmpgtd xmm0, xmm1",
            lanes32([0x8000_0000, 1, 0, 0]),
            lanes32([0, 0, 0xffff_ffff, 0]),
        );
        assert_eq!(dwords(got), [0, u32::MAX, u32::MAX, 0]);
    }

    #[test]
    fn compare_with_self_is_all_ones_for_equality() {
        let mut rig = Rig::new();
        assert_eq!(rig.op("pcmpeqb xmm0, xmm1", A, A), u128::MAX);
        assert_eq!(rig.op("pcmpgtb xmm0, xmm1", A, A), 0);
    }

    #[test]
    fn integer_instructions_leave_mxcsr_alone() {
        let mut rig = Rig::new();
        rig.op(
            "paddd xmm0, xmm1",
            lanes32([SNAN_SINGLE; 4]),
            lanes32([QNAN_SINGLE; 4]),
        );
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    const SHIFT_SAMPLE: u128 = 0x8001_7ffe_0123_fedc_f00d_8000_0001_ff00;
    const SHIFT_COUNTS: [u8; 14] = [0, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255];

    #[derive(Clone, Copy)]
    enum ShiftKind {
        Left,
        LogicalRight,
        ArithmeticRight,
    }

    /// Lane-by-lane model for one shift instruction, written with i128 so no Rust shift can overflow.
    fn shift_model(value: u128, width: u32, kind: ShiftKind, count: u64) -> u128 {
        let lanes = 128 / width;
        let mut result = 0u128;
        for k in 0..lanes {
            let lane = (value >> (k * width)) & ((1u128 << width) - 1);
            let signed = (lane as i128)
                - if lane >> (width - 1) == 1 {
                    1i128 << width
                } else {
                    0
                };
            let shifted: i128 = match kind {
                ShiftKind::Left => {
                    if count >= width as u64 {
                        0
                    } else {
                        (lane << count) as i128
                    }
                }
                ShiftKind::LogicalRight => {
                    if count >= width as u64 {
                        0
                    } else {
                        (lane >> count) as i128
                    }
                }
                ShiftKind::ArithmeticRight => signed >> count.min(width as u64 - 1),
            };
            result |= ((shifted as u128) & ((1u128 << width) - 1)) << (k * width);
        }
        result
    }

    const SHIFT_INSTRUCTIONS: [(&str, u32, ShiftKind); 8] = [
        ("psllw", 16, ShiftKind::Left),
        ("pslld", 32, ShiftKind::Left),
        ("psllq", 64, ShiftKind::Left),
        ("psrlw", 16, ShiftKind::LogicalRight),
        ("psrld", 32, ShiftKind::LogicalRight),
        ("psrlq", 64, ShiftKind::LogicalRight),
        ("psraw", 16, ShiftKind::ArithmeticRight),
        ("psrad", 32, ShiftKind::ArithmeticRight),
    ];

    #[test]
    fn shifts_by_immediate_match_the_lane_model_including_out_of_range_counts() {
        let mut rig = Rig::new();
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            for count in [0u8, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255] {
                rig.h.cpu.xmm[0] = SHIFT_SAMPLE;
                rig.run_ok(&format!("{name} xmm0, {count}"));
                assert_eq!(
                    rig.xmm(0),
                    shift_model(SHIFT_SAMPLE, width, kind, count as u64),
                    "{name} {count}"
                );
            }
        }
    }

    #[test]
    fn shifts_by_register_use_the_low_quadword_as_the_count() {
        let mut rig = Rig::new();
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            for count in SHIFT_COUNTS.iter().map(|c| *c as u64).chain([
                0x1_0000_0000,
                u64::MAX,
                0x8000_0000_0000_0000,
            ]) {
                let got = rig.op(&format!("{name} xmm0, xmm1"), SHIFT_SAMPLE, count as u128);
                assert_eq!(
                    got,
                    shift_model(SHIFT_SAMPLE, width, kind, count),
                    "{name} {count:#x}"
                );
                let got = rig.op_mem(
                    &format!("{name} xmm0, xmmword ptr [rsi]"),
                    SHIFT_SAMPLE,
                    count as u128,
                );
                assert_eq!(
                    got,
                    shift_model(SHIFT_SAMPLE, width, kind, count),
                    "{name} memory {count:#x}"
                );
            }
        }
    }

    #[test]
    fn shift_count_ignores_the_upper_quadword_of_the_count_register() {
        // The count is the low 64 bits only; garbage above them must not turn a small count into an
        // out-of-range one.
        let mut rig = Rig::new();
        let count = (0x1234_5678_9abc_def0u128 << 64) | 3;
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            assert_eq!(
                rig.op(&format!("{name} xmm0, xmm1"), SHIFT_SAMPLE, count),
                shift_model(SHIFT_SAMPLE, width, kind, 3),
                "{name}"
            );
            assert_eq!(
                rig.op_mem(
                    &format!("{name} xmm0, xmmword ptr [rsi]"),
                    SHIFT_SAMPLE,
                    count
                ),
                shift_model(SHIFT_SAMPLE, width, kind, 3),
                "{name} memory"
            );
        }
    }

    #[test]
    fn arithmetic_shift_with_a_huge_count_fills_with_the_sign() {
        let mut rig = Rig::new();
        let got = rig.op(
            "psraw xmm0, xmm1",
            lanes16([0x8000, 0x7fff, 0xffff, 1, 0x8000, 0, 0, 0]),
            u64::MAX as u128,
        );
        assert_eq!(words(got), [0xffff, 0, 0xffff, 0, 0xffff, 0, 0, 0]);
        let got = rig.op(
            "psrad xmm0, xmm1",
            lanes32([0x8000_0000, 0x7fff_ffff, 1, u32::MAX]),
            32,
        );
        assert_eq!(dwords(got), [u32::MAX, 0, 0, u32::MAX]);
    }

    #[test]
    fn logical_shift_with_count_equal_to_the_lane_width_clears() {
        let mut rig = Rig::new();
        assert_eq!(rig.op("psllw xmm0, xmm1", u128::MAX, 16), 0);
        assert_eq!(
            rig.op("psllw xmm0, xmm1", u128::MAX, 15),
            lanes16([0x8000; 8])
        );
        assert_eq!(rig.op("psrlq xmm0, xmm1", u128::MAX, 64), 0);
        assert_eq!(rig.op("psrlq xmm0, xmm1", u128::MAX, 63), lanes64([1, 1]));
        assert_eq!(
            rig.op("pslld xmm0, xmm1", u128::MAX, 31),
            lanes32([0x8000_0000; 4])
        );
    }

    #[test]
    fn shifts_by_immediate_on_extended_registers_and_the_rm_destination() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[9] = SHIFT_SAMPLE;
        rig.run_ok("psllw xmm9, 3");
        assert_eq!(
            rig.xmm(9),
            shift_model(SHIFT_SAMPLE, 16, ShiftKind::Left, 3)
        );
        assert_eq!(rig.xmm(0), 0, "other registers are untouched");
    }

    #[test]
    fn byte_shifts_move_the_whole_register() {
        let mut rig = Rig::new();
        let value = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100u128;
        for count in [0u32, 1, 4, 8, 15, 16, 17, 255] {
            rig.h.cpu.xmm[0] = value;
            rig.run_ok(&format!("pslldq xmm0, {count}"));
            let expected_left = if count > 15 { 0 } else { value << (8 * count) };
            assert_eq!(rig.xmm(0), expected_left, "pslldq {count}");
            rig.h.cpu.xmm[0] = value;
            rig.run_ok(&format!("psrldq xmm0, {count}"));
            let expected_right = if count > 15 { 0 } else { value >> (8 * count) };
            assert_eq!(rig.xmm(0), expected_right, "psrldq {count}");
        }
        rig.h.cpu.xmm[9] = value;
        rig.run_ok("pslldq xmm9, 4");
        assert_eq!(rig.xmm(9), value << 32);
    }

    #[test]
    fn pextrw_extracts_a_zero_extended_word() {
        let mut rig = Rig::new();
        let value = lanes16([
            0x1000, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0xf777,
        ]);
        for index in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 15] {
            rig.h.cpu.xmm[1] = value;
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.run_ok(&format!("pextrw eax, xmm1, {index}"));
            assert_eq!(
                rig.gpr(RAX),
                words(value)[index & 7] as u64,
                "index {index}: only the low three bits select"
            );
        }
        rig.h.cpu.xmm[10] = value;
        rig.h.cpu.gpr[9] = u64::MAX;
        rig.run_ok("pextrw r9d, xmm10, 1");
        assert_eq!(rig.gpr(9), 0x1111);
    }

    #[test]
    fn pextrw_with_a_memory_operand_is_invalid() {
        let mut rig = Rig::new();
        assert_eq!(rig.run(".byte 0x66, 0x0f, 0xc5, 0x06, 0x01"), ud());
    }

    #[test]
    fn pinsrw_replaces_one_word() {
        let mut rig = Rig::new();
        let base = lanes16([
            0x1000, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777,
        ]);
        for index in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 15] {
            rig.h.cpu.gpr[RAX] = 0xffff_ffff_dead_beef;
            let got = rig.op(&format!("pinsrw xmm0, eax, {index}"), base, 0);
            let mut expected = words(base);
            expected[index & 7] = 0xbeef;
            assert_eq!(words(got), expected, "index {index}");
        }
        rig.h.cpu.xmm[0] = base;
        rig.h.write(DATA, &[0x34, 0x12]);
        rig.run_ok("pinsrw xmm0, word ptr [rsi], 2");
        let got = rig.xmm(0);
        let mut expected = words(base);
        expected[2] = 0x1234;
        assert_eq!(words(got), expected);
        rig.h.cpu.gpr[8] = 0xabcd;
        rig.h.cpu.xmm[9] = base;
        rig.run_ok("pinsrw xmm9, r8d, 5");
        let mut expected = words(base);
        expected[5] = 0xabcd;
        assert_eq!(words(rig.xmm(9)), expected);
    }

    const FXSAVE_FCW_OFFSET: u64 = 0;
    const FXSAVE_FSW_OFFSET: u64 = 2;
    const FXSAVE_FTW_OFFSET: u64 = 4;
    const FXSAVE_MXCSR_OFFSET: u64 = 24;
    const FXSAVE_MXCSR_MASK_OFFSET: u64 = 28;
    const FXSAVE_XMM_OFFSET: u64 = 160;
    const FXSAVE_AREA_BYTES: usize = 512;

    #[test]
    fn ldmxcsr_loads_any_value_without_reserved_bits() {
        let mut rig = Rig::new();
        for value in [
            0u32,
            MXCSR_RESET,
            0xffff,
            0x1fbf,
            0x8040,
            0x6000,
            0x0001,
            0x1f80 | (RC_UP << MXCSR_RC_SHIFT),
        ] {
            rig.h.write(DATA, &value.to_le_bytes());
            rig.run_ok("ldmxcsr dword ptr [rsi]");
            assert_eq!(rig.mxcsr(), value);
        }
    }

    #[test]
    fn ldmxcsr_with_a_reserved_bit_raises_gp0_and_keeps_mxcsr() {
        let mut rig = Rig::new();
        for bit in 16..32 {
            rig.h.write(DATA, &(MXCSR_RESET | 1 << bit).to_le_bytes());
            assert_eq!(rig.run("ldmxcsr dword ptr [rsi]"), gp0(), "bit {bit}");
            assert_eq!(rig.mxcsr(), MXCSR_RESET, "bit {bit}");
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
        }
        assert_eq!(MXCSR_VALID_BITS, 0xffff);
    }

    #[test]
    fn ldmxcsr_accepts_a_misaligned_operand() {
        let mut rig = Rig::new();
        rig.h.write(DATA + 3, &0x1f81u32.to_le_bytes());
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 3;
        assert_eq!(rig.exec("ldmxcsr dword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.mxcsr(), 0x1f81);
    }

    #[test]
    fn stmxcsr_stores_the_current_mxcsr() {
        let mut rig = Rig::new();
        fill_sentinel(&mut rig, DATA);
        rig.reset();
        rig.h.cpu.mxcsr = 0x1234;
        assert_eq!(rig.exec("stmxcsr dword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.h.read(DATA, 4), 0x1234u32.to_le_bytes());
        assert_eq!(changed_bytes(&rig, DATA), (16..20).collect::<Vec<_>>());
        assert_eq!(rig.mxcsr(), 0x1234);
    }

    #[test]
    fn ldmxcsr_then_stmxcsr_round_trips_and_changes_rounding() {
        let mut rig = Rig::new();
        let mxcsr = MXCSR_RESET | (RC_DOWN << MXCSR_RC_SHIFT);
        rig.h.write(DATA, &mxcsr.to_le_bytes());
        rig.h.cpu.xmm[0] = singles([1.0; 4]);
        rig.h.cpu.xmm[1] = singles([3.0; 4]);
        rig.reset();
        assert_eq!(
            rig.exec("ldmxcsr dword ptr [rsi]; stmxcsr dword ptr [rsi+8]"),
            Stop::Halted
        );
        assert_eq!(rig.h.read(DATA + 8, 4), mxcsr.to_le_bytes());
        assert_eq!(rig.exec("divps xmm0, xmm1"), Stop::Halted);
        assert_eq!(
            dwords(rig.xmm(0)),
            [ONE_THIRD_SINGLE_DOWN; 4],
            "the loaded rounding mode applies"
        );
    }

    #[test]
    fn mxcsr_flags_can_be_cleared_by_ldmxcsr() {
        let mut rig = Rig::new();
        rig.op("divss xmm0, xmm1", singles([1.0; 4]), singles([0.0; 4]));
        expect_flag(&rig, FLAG_DIVIDE_BY_ZERO);
        rig.h.write(DATA, &MXCSR_RESET.to_le_bytes());
        assert_eq!(rig.exec("ldmxcsr dword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
    }

    #[test]
    fn clearing_a_mask_through_ldmxcsr_enables_the_exception() {
        let mut rig = Rig::new();
        rig.h.cpu.xmm[0] = singles([1.0; 4]);
        rig.h.cpu.xmm[1] = singles([0.0; 4]);
        rig.h.write(
            DATA,
            &(MXCSR_RESET & !(FLAG_DIVIDE_BY_ZERO << MXCSR_MASK_SHIFT)).to_le_bytes(),
        );
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        assert_eq!(rig.exec("ldmxcsr dword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.exec("divss xmm0, xmm1"), xm());
    }

    #[test]
    fn fxsave_stores_mxcsr_and_the_sixteen_vector_registers() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &[SENTINEL; FXSAVE_AREA_BYTES + 16]);
        for k in 0..16 {
            rig.h.cpu.xmm[k] = A.rotate_left(8 * k as u32) ^ k as u128;
        }
        rig.reset();
        rig.h.cpu.mxcsr = 0x1fa5;
        rig.h.cpu.fcw = 0x27f;
        rig.h.cpu.fsw = 0x0800;
        rig.h.cpu.ftw = 0xff;
        assert_eq!(rig.exec("fxsave [rsi]"), Stop::Halted);
        let image = rig.h.read(DATA, FXSAVE_AREA_BYTES + 16);
        assert_eq!(
            &image[FXSAVE_FCW_OFFSET as usize..][..2],
            0x27fu16.to_le_bytes()
        );
        assert_eq!(
            &image[FXSAVE_FSW_OFFSET as usize..][..2],
            0x0800u16.to_le_bytes()
        );
        assert_eq!(image[FXSAVE_FTW_OFFSET as usize], 0xff);
        assert_eq!(
            &image[FXSAVE_MXCSR_OFFSET as usize..][..4],
            0x1fa5u32.to_le_bytes()
        );
        assert_eq!(
            &image[FXSAVE_MXCSR_MASK_OFFSET as usize..][..4],
            MXCSR_VALID_BITS.to_le_bytes()
        );
        for k in 0..16u64 {
            let at = (FXSAVE_XMM_OFFSET + 16 * k) as usize;
            assert_eq!(
                &image[at..at + 16],
                rig.xmm(k as usize).to_le_bytes(),
                "xmm{k}"
            );
        }
        assert!(
            image[FXSAVE_AREA_BYTES..].iter().all(|b| *b == SENTINEL),
            "no write past 512 bytes"
        );
    }

    #[test]
    fn fxrstor_restores_mxcsr_and_the_vector_registers() {
        let mut rig = Rig::new();
        for k in 0..16 {
            rig.h.cpu.xmm[k] = A.rotate_left(8 * k as u32) ^ k as u128;
        }
        rig.reset();
        rig.h.cpu.mxcsr = 0x1fa5;
        assert_eq!(rig.exec("fxsave [rsi]"), Stop::Halted);
        let saved: Vec<u128> = (0..16).map(|k| rig.xmm(k)).collect();
        for k in 0..16 {
            rig.h.cpu.xmm[k] = 0;
        }
        rig.h.cpu.mxcsr = MXCSR_RESET;
        assert_eq!(rig.exec("fxrstor [rsi]"), Stop::Halted);
        assert_eq!(rig.mxcsr(), 0x1fa5);
        for (k, expected) in saved.iter().enumerate() {
            assert_eq!(rig.xmm(k), *expected, "xmm{k}");
        }
    }

    #[test]
    fn fxrstor_with_reserved_mxcsr_bits_faults_and_changes_nothing() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &[0; FXSAVE_AREA_BYTES]);
        rig.h
            .write(DATA + FXSAVE_MXCSR_OFFSET, &0x0001_1f80u32.to_le_bytes());
        rig.h.write(DATA + FXSAVE_XMM_OFFSET, &B.to_le_bytes());
        rig.h.cpu.xmm[0] = A;
        assert_eq!(rig.run("fxrstor [rsi]"), gp0());
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
        assert_eq!(rig.xmm(0), A);
    }

    #[test]
    fn fences_pause_and_prefetch_change_no_architectural_state() {
        let mut rig = Rig::new();
        for key in [
            "sfence",
            "lfence",
            "mfence",
            "pause",
            "prefetcht0 byte ptr [rsi]",
            "prefetcht1 byte ptr [rsi]",
            "prefetcht2 byte ptr [rsi]",
            "prefetchnta byte ptr [rsi]",
            "prefetchw byte ptr [rsi]",
            "clflush byte ptr [rsi]",
        ] {
            for k in 0..16 {
                rig.h.cpu.xmm[k] = A + k as u128;
                rig.h.cpu.gpr[k] = 0x1111_1111_1111_1111u64.wrapping_mul(k as u64 + 1);
            }
            rig.reset();
            rig.h.cpu.set_eflags_arith(CF | ZF);
            let before = (rig.h.cpu.gpr, rig.h.cpu.xmm, rig.mxcsr(), rig.flags());
            assert_eq!(rig.exec(key), Stop::Halted, "{key}");
            let mut after = (rig.h.cpu.gpr, rig.h.cpu.xmm, rig.mxcsr(), rig.flags());
            after.0[RAX] = before.0[RAX];
            assert_eq!(before, after, "{key}");
        }
    }

    #[test]
    fn prefetch_never_faults_even_for_bad_addresses() {
        let mut rig = Rig::new();
        for address in [
            0u64,
            0xdead_0000_0000_0000,
            0xffff_ffff_ffff_f000,
            0x7fff_ffff_f000,
        ] {
            for key in [
                "prefetcht0 byte ptr [rsi]",
                "prefetchnta byte ptr [rsi]",
                "prefetchw byte ptr [rsi]",
            ] {
                rig.reset();
                rig.h.cpu.gpr[RSI] = address;
                assert_eq!(rig.exec(key), Stop::Halted, "{key} {address:#x}");
            }
        }
    }

    #[test]
    fn fences_and_prefetch_are_not_gated_by_the_sse_enable_bits() {
        let mut rig = Rig::new();
        for key in [
            "sfence",
            "lfence",
            "mfence",
            "pause",
            "prefetcht0 byte ptr [rsi]",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            rig.h.cpu.cr0 |= CR0_EM | CR0_TS;
            assert_eq!(rig.exec(key), Stop::Halted, "{key}");
        }
    }

    #[test]
    fn group_15_register_forms_other_than_the_fences_are_invalid() {
        let mut rig = Rig::new();
        // /0..3 with mod == 3 (rdfsbase and friends need F3 and are not implemented), and the same
        // encoding with a rep prefix on a fence
        for key in [
            ".byte 0x0f, 0xae, 0xc0",
            ".byte 0x0f, 0xae, 0xc8, 0x90",
            ".byte 0xf3, 0x0f, 0xae, 0xe8",
        ] {
            assert_eq!(rig.run(key), ud(), "{key}");
        }
        // /4, /5, /6 with a memory operand (xsave and reserved encodings) are #UD here
        for key in [
            ".byte 0x0f, 0xae, 0x26",
            ".byte 0x0f, 0xae, 0x2e",
            ".byte 0x0f, 0xae, 0x36",
        ] {
            assert_eq!(rig.run(key), ud(), "{key}");
        }
        assert_eq!(rig.run(".byte 0x0f, 0xae, 0xe8"), Stop::Halted, "lfence");
    }

    /// Operand state for the differential runs: distinct, awkward values in every register and
    /// memory, so all lanes matter.
    fn differential_state(h: &mut Harness, mxcsr: u32, cr4_extra: u64) {
        for k in 0..16 {
            h.cpu.xmm[k] = INTERESTING[k % INTERESTING.len()] ^ ((k as u128) << 100);
        }
        for k in 0..4 {
            h.write(
                DATA + 16 * k,
                &INTERESTING[(k as usize + 3) % INTERESTING.len()].to_le_bytes(),
            );
        }
        for k in 0..8 {
            let bits = INTERESTING[(k + 2) % INTERESTING.len()] ^ ((k as u128) << 70);
            h.cpu.st[k].copy_from_slice(&bits.to_le_bytes()[..10]);
        }
        h.cpu.fsw = 3 << 11;
        h.cpu.ftw = 0x1b;
        h.cpu.gpr[RAX] = 0x1234_5678_9abc_def0;
        h.cpu.gpr[8] = 0xffff_ffff_8000_0001;
        h.cpu.gpr[9] = 5;
        h.cpu.gpr[10] = 0x0fff_ffff_ffff_ffff;
        h.cpu.mxcsr = mxcsr;
        h.cpu.cr4 |= cr4_extra;
    }

    #[test]
    fn jit_and_interpreter_agree_on_every_snippet_with_default_mxcsr() {
        for (key, bytes) in SNIPPETS {
            assert_jit_matches_interpreter(bytes, |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                differential_state(h, MXCSR_RESET, 0);
            });
            let _ = key;
        }
    }

    #[test]
    fn jit_and_interpreter_agree_on_every_snippet_with_all_exceptions_unmasked_and_daz_ftz() {
        for (key, bytes) in SNIPPETS {
            let _ = key;
            assert_jit_matches_interpreter(bytes, |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                h.cpu.gpr[RSI] = DATA;
                differential_state(
                    h,
                    MXCSR_DAZ | MXCSR_FTZ | (RC_UP << MXCSR_RC_SHIFT),
                    CR4_OSXMMEXCPT,
                );
            });
        }
    }

    #[test]
    fn jit_and_interpreter_agree_on_misaligned_memory_operands() {
        for (key, bytes) in SNIPPETS {
            if !key.contains("ptr [rsi]") {
                continue;
            }
            assert_jit_matches_interpreter(bytes, |h| {
                h.cpu.cr4 |= CR4_OSFXSR;
                differential_state(h, MXCSR_RESET, 0);
                h.cpu.gpr[RSI] = DATA + 8;
            });
        }
    }

    #[test]
    fn jit_and_interpreter_agree_when_sse_is_gated_off() {
        for (key, bytes) in SNIPPETS.iter().step_by(7) {
            let _ = key;
            for (cr0_bits, cr4_clear) in [(CR0_EM, 0), (CR0_TS, 0), (0, CR4_OSFXSR)] {
                assert_jit_matches_interpreter(bytes, |h| {
                    h.cpu.cr4 |= CR4_OSFXSR;
                    h.cpu.gpr[RSI] = DATA;
                    differential_state(h, MXCSR_RESET, 0);
                    h.cpu.cr0 |= cr0_bits;
                    h.cpu.cr4 &= !cr4_clear;
                });
            }
        }
    }

    #[test]
    fn jit_runs_an_sse_loop_like_the_interpreter() {
        // xmm0 = xmm0 + xmm1 repeated by a counted loop: the SSE instruction sits inside a block that
        // the JIT compiles and re-executes.
        // mov ecx, 5; top: addps xmm0, xmm1; mulps xmm1, xmm1; dec ecx; jnz top; hlt
        let code = [
            0xb9, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x58, 0xc1, 0x0f, 0x59, 0xc9, 0xff, 0xc9, 0x75,
            0xf6, 0xf4,
        ];
        assert_jit_matches_interpreter(&code, |h| {
            h.cpu.cr4 |= CR4_OSFXSR;
            h.cpu.xmm[0] = singles([1.0, 2.0, 3.0, 4.0]);
            h.cpu.xmm[1] = singles([1.5, 0.5, 2.0, -1.0]);
        });
        let mut h = Harness::new(&code);
        h.cpu.cr4 |= CR4_OSFXSR;
        h.cpu.xmm[0] = singles([1.0, 2.0, 3.0, 4.0]);
        h.cpu.xmm[1] = singles([1.5, 0.5, 2.0, -1.0]);
        h.run_to_halt();
        let mut a = [1.0f32, 2.0, 3.0, 4.0];
        let mut b = [1.5f32, 0.5, 2.0, -1.0];
        for _ in 0..5 {
            for k in 0..4 {
                a[k] += b[k];
                b[k] *= b[k];
            }
        }
        assert_eq!(h.cpu.xmm[0], singles(a));
    }

    /// Canonical but unmapped: the boot identity map only covers the first 4 GiB.
    const UNMAPPED: u64 = 0x80_0000_0000;
    const PAGE_FAULT_VECTOR: u8 = 14;
    const PAGE_FAULT_WRITE: u32 = 1 << 1;

    #[test]
    fn loads_from_unmapped_memory_raise_a_page_fault_and_leave_the_register() {
        let mut rig = Rig::new();
        for key in [
            "movups xmm0, xmmword ptr [rsi]",
            "addsd xmm0, qword ptr [rsi]",
            "movss xmm0, dword ptr [rsi]",
            "cvtdq2pd xmm0, qword ptr [rsi]",
        ] {
            rig.h.cpu.xmm[0] = A;
            rig.reset();
            rig.h.cpu.gpr[RSI] = UNMAPPED;
            assert_eq!(rig.exec(key), fault(PAGE_FAULT_VECTOR, Some(0)), "{key}");
            assert_eq!(rig.h.cpu.cr2, UNMAPPED, "{key}");
            assert_eq!(rig.xmm(0), A, "{key}");
            assert_eq!(rig.h.cpu.rip, CODE_BASE);
        }
    }

    #[test]
    fn stores_to_unmapped_memory_raise_a_write_page_fault() {
        let mut rig = Rig::new();
        for key in [
            "movups xmmword ptr [rsi], xmm0",
            "movlps qword ptr [rsi], xmm0",
            "movd dword ptr [rsi], xmm0",
            "stmxcsr dword ptr [rsi]",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = UNMAPPED;
            assert_eq!(
                rig.exec(key),
                fault(PAGE_FAULT_VECTOR, Some(PAGE_FAULT_WRITE)),
                "{key}"
            );
            assert_eq!(rig.h.cpu.cr2, UNMAPPED, "{key}");
        }
    }

    #[test]
    fn non_canonical_addresses_raise_gp0() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.gpr[RSI] = 0x0001_0000_0000_0000;
        assert_eq!(rig.exec("movups xmm0, xmmword ptr [rsi]"), gp0());
    }

    #[test]
    fn alignment_is_checked_on_the_linear_address_including_the_segment_base() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.seg[SEG_FS].base = 8;
        assert_eq!(
            rig.exec("movaps xmm0, xmmword ptr fs:[rsi]"),
            gp0(),
            "rsi is aligned but fs:rsi is not"
        );
        assert_eq!(rig.exec("movups xmm0, xmmword ptr fs:[rsi]"), Stop::Halted);
        rig.h.cpu.seg[SEG_FS].base = 16;
        assert_eq!(rig.exec("movaps xmm0, xmmword ptr fs:[rsi]"), Stop::Halted);
    }

    #[test]
    fn scaled_index_addressing_selects_the_operand_and_its_alignment() {
        let mut rig = Rig::new();
        rig.h.write(DATA + 32, &B.to_le_bytes());
        rig.h.cpu.xmm[0] = A;
        rig.reset();
        rig.h.cpu.gpr[RDX] = 2;
        assert_eq!(
            rig.exec("movaps xmm0, xmmword ptr [rsi + rdx*8 + 16]"),
            Stop::Halted
        );
        assert_eq!(rig.xmm(0), B);
        rig.h.cpu.gpr[RDX] = 1;
        rig.h.cpu.xmm[0] = A;
        assert_eq!(
            rig.exec("movaps xmm0, xmmword ptr [rsi + rdx*8 + 16]"),
            gp0()
        );
        assert_eq!(rig.xmm(0), A);
        rig.h.write(DATA + 8 + 3 * 8, &2.5f64.to_le_bytes());
        rig.h.cpu.gpr[RDX] = 3;
        rig.h.cpu.xmm[0] = doubles([1.0, 9.0]);
        assert_eq!(
            rig.exec("addsd xmm0, qword ptr [rsi + rdx*8 + 8]"),
            Stop::Halted
        );
        assert_eq!(rig.xmm(0), doubles([3.5, 9.0]));
    }

    #[test]
    fn rip_relative_operands_are_relative_to_the_next_instruction() {
        // movaps xmm0, [rip + 9] is 7 bytes then a 1-byte hlt, so the operand is at CODE_BASE + 16.
        let mut rig = Rig::new();
        rig.h.write(CODE_BASE + 16, &B.to_le_bytes());
        rig.h.cpu.xmm[0] = A;
        assert_eq!(rig.run("movaps xmm0, xmmword ptr [rip + 9]"), Stop::Halted);
        assert_eq!(rig.xmm(0), B);
        rig.h.cpu.xmm[0] = A;
        assert_eq!(rig.run("movaps xmm0, xmmword ptr [rip + 10]"), gp0());
        assert_eq!(rig.xmm(0), A);
    }

    #[test]
    fn two_sse_instructions_in_a_row() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &A.to_le_bytes());
        rig.h.write(DATA + 16, &B.to_le_bytes());
        rig.run_ok("movups xmm0, xmmword ptr [rsi]; movups xmm1, xmmword ptr [rsi + 16]");
        assert_eq!((rig.xmm(0), rig.xmm(1)), (A, B));
    }

    #[test]
    fn rep_prefixes_win_over_the_operand_size_prefix_and_the_last_rep_wins() {
        let mut rig = Rig::new();
        let (dst, src) = (
            singles([1.0, 2.0, 3.0, 4.0]),
            singles([10.0, 20.0, 30.0, 40.0]),
        );
        let addss = singles([11.0, 2.0, 3.0, 4.0]);
        for key in [
            ".byte 0x66, 0xf3, 0x0f, 0x58, 0xc1",
            ".byte 0xf3, 0x66, 0x0f, 0x58, 0xc1",
            ".byte 0xf2, 0xf3, 0x0f, 0x58, 0xc1",
        ] {
            assert_eq!(rig.op(key, dst, src), addss, "{key}");
        }
        // F3 then F2: the F2 (addsd) wins, as does F2 with 66
        let addsd = rig.op("addsd xmm0, xmm1", dst, src);
        for key in [
            ".byte 0xf3, 0xf2, 0x0f, 0x58, 0xc1",
            ".byte 0x66, 0xf2, 0x0f, 0x58, 0xc1",
        ] {
            assert_eq!(rig.op(key, dst, src), addsd, "{key}");
        }
    }

    fn insn(op: u16, rep: u8, p66: bool) -> Insn {
        Insn {
            op,
            rep,
            p66,
            md: MODRM_MOD_REGISTER,
            has_modrm: true,
            modrm: 0xc1,
            rm: 1,
            len: 4,
            ..Insn::default()
        }
    }

    #[test]
    fn only_the_two_byte_map_is_handled_here() {
        let mut rig = Rig::new();
        rig.reset();
        // map 0 (one-byte), 2 (0F 38) and 3 (0F 3A) opcodes that share a low byte with real SSE ones
        for map in [0u16, 2, 3] {
            let result = rig
                .h
                .cpu
                .sse(&insn((map << 8) | 0x58, 0, false), CODE_BASE + 4);
            assert!(matches!(result, Ok(false)), "map {map}");
        }
        let result = rig
            .h
            .cpu
            .sse(&insn((OPCODE_MAP_0F << 8) | 0x58, 0, false), CODE_BASE + 4);
        assert!(matches!(result, Ok(true)));
    }

    #[test]
    fn unassigned_opcodes_are_reported_as_not_handled() {
        let mut rig = Rig::new();
        rig.reset();
        for opcode in [0x00u16, 0x0b, 0x38, 0x3a, 0xf0, 0xf7, 0xff] {
            let result = rig
                .h
                .cpu
                .sse(&insn((OPCODE_MAP_0F << 8) | opcode, 0, true), CODE_BASE + 4);
            assert!(matches!(result, Ok(false)), "{opcode:#x}");
        }
    }

    #[test]
    fn sse_gate_checks_em_then_osfxsr_then_ts() {
        let mut rig = Rig::new();
        rig.reset();
        assert!(rig.h.cpu.sse_gate().is_ok());
        rig.h.cpu.cr0 |= CR0_TS;
        assert!(matches!(
            rig.h.cpu.sse_gate(),
            Err(Exception::Fault { vector: 7, .. })
        ));
        rig.h.cpu.cr4 &= !CR4_OSFXSR;
        assert!(matches!(
            rig.h.cpu.sse_gate(),
            Err(Exception::Fault { vector: 6, .. })
        ));
        rig.h.cpu.cr4 |= CR4_OSFXSR;
        rig.h.cpu.cr0 |= CR0_EM;
        assert!(matches!(
            rig.h.cpu.sse_gate(),
            Err(Exception::Fault { vector: 6, .. })
        ));
    }

    #[test]
    fn fp_commit_sets_flags_and_checks_the_masks() {
        let mut rig = Rig::new();
        rig.reset();
        let mut env = default_env();
        env.flags = FLAG_PRECISION | FLAG_INVALID;
        assert!(rig.h.cpu.fp_commit(&env).is_ok());
        assert_eq!(rig.h.cpu.mxcsr, MXCSR_RESET | FLAG_PRECISION | FLAG_INVALID);
        // unmask divide-by-zero only: other flags still pass
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_DIVIDE_BY_ZERO << MXCSR_MASK_SHIFT);
        assert!(rig.h.cpu.fp_commit(&env).is_ok());
        env.flags = FLAG_DIVIDE_BY_ZERO;
        assert!(matches!(
            rig.h.cpu.fp_commit(&env),
            Err(Exception::Fault { vector: 6, .. })
        ));
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        assert!(matches!(
            rig.h.cpu.fp_commit(&env),
            Err(Exception::Fault {
                vector: VECTOR_SIMD_FLOATING_POINT,
                error_code: None,
                ..
            })
        ));
        assert_eq!(rig.h.cpu.mxcsr & FLAG_DIVIDE_BY_ZERO, FLAG_DIVIDE_BY_ZERO);
    }

    #[test]
    fn fp_env_decodes_mxcsr() {
        let env = FpEnv::new(MXCSR_DAZ | MXCSR_FTZ | (RC_ZERO << MXCSR_RC_SHIFT));
        assert!(env.denormals_are_zero && env.flush_to_zero && !env.underflow_masked);
        assert_eq!(env.rounding, RC_ZERO);
        let env = FpEnv::new(MXCSR_RESET);
        assert!(!env.denormals_are_zero && !env.flush_to_zero && env.underflow_masked);
        assert_eq!(env.rounding, RC_NEAREST);
        assert_eq!(env.flags, 0);
    }

    #[test]
    fn mxcsr_reset_value_masks_everything() {
        assert_eq!(MXCSR_RESET & ALL_MASKS, ALL_MASKS);
        assert_eq!(MXCSR_RESET & MXCSR_EXCEPTION_FLAGS, 0);
        assert_eq!(Rig::new().h.cpu.mxcsr, MXCSR_RESET);
    }

    #[test]
    fn ldmxcsr_and_stmxcsr_follow_the_sse_gating_rules() {
        let mut rig = Rig::new();
        for key in ["ldmxcsr dword ptr [rsi]", "stmxcsr dword ptr [rsi]"] {
            rig.reset();
            rig.h.cpu.cr0 |= CR0_EM;
            assert_eq!(rig.exec(key), ud(), "{key} with CR0.EM");
            rig.reset();
            rig.h.cpu.cr0 |= CR0_TS;
            assert_eq!(rig.exec(key), nm(), "{key} with CR0.TS");
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(key), ud(), "{key} without CR4.OSFXSR");
        }
    }

    #[test]
    fn fxsave_and_fxrstor_with_a_misaligned_area_raise_gp0() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        assert_eq!(rig.exec("fxsave [rsi]"), gp0());
        assert_eq!(rig.exec("fxrstor [rsi]"), gp0());
    }

    const MMX_TOP_MASK: u16 = 7 << 11;
    const MMX_EXPONENT_MARKER: [u8; 2] = [0x34, 0x12];
    const ALL_TAGS_VALID: u16 = 0xff;
    const ALL_TAGS_EMPTY: u16 = 0;
    /// Words of FSW that are not TOP: a stack-fault flag, used to see that MMX leaves them alone.
    const FSW_INVALID_OPERATION: u16 = 1;
    const X87_ONE_MANTISSA: u64 = 0x8000_0000_0000_0000;

    fn mm(rig: &Rig, index: usize) -> u64 {
        u64::from_le_bytes(rig.h.cpu.st[index][..8].try_into().unwrap())
    }

    fn mm_exponent(rig: &Rig, index: usize) -> [u8; 2] {
        rig.h.cpu.st[index][8..10].try_into().unwrap()
    }

    /// Sets mmN's mantissa and gives the register a recognisable (non-all-ones) exponent field, so a
    /// test can tell a read (field untouched) from a write (field becomes all ones).
    fn set_mm(rig: &mut Rig, index: usize, value: u64) {
        rig.h.cpu.st[index][..8].copy_from_slice(&value.to_le_bytes());
        rig.h.cpu.st[index][8..10].copy_from_slice(&MMX_EXPONENT_MARKER);
    }

    fn mm_op(rig: &mut Rig, asm: &str, dst: u64, src: u64) -> u64 {
        set_mm(rig, 0, dst);
        set_mm(rig, 1, src);
        rig.run_ok(asm);
        mm(rig, 0)
    }

    fn mm_op_mem(rig: &mut Rig, asm: &str, dst: u64, src: u64) -> u64 {
        set_mm(rig, 0, dst);
        rig.h.write(DATA, &src.to_le_bytes());
        rig.run_ok(asm);
        mm(rig, 0)
    }

    fn mm_bytes(v: u64) -> [u8; 8] {
        v.to_le_bytes()
    }
    fn mm_words(v: u64) -> [u16; 4] {
        std::array::from_fn(|k| (v >> (16 * k)) as u16)
    }
    fn mm_dwords(v: u64) -> [u32; 2] {
        [v as u32, (v >> 32) as u32]
    }
    fn from_mm_bytes(b: [u8; 8]) -> u64 {
        u64::from_le_bytes(b)
    }
    fn from_mm_words(w: [u16; 4]) -> u64 {
        w.iter().rev().fold(0, |acc, &x| (acc << 16) | x as u64)
    }
    fn from_mm_dwords(d: [u32; 2]) -> u64 {
        ((d[1] as u64) << 32) | d[0] as u64
    }

    /// Operands for the 64-bit tests: both halves of every interesting 128-bit pattern.
    fn interesting_qwords() -> Vec<u64> {
        let mut values: Vec<u64> = INTERESTING
            .iter()
            .flat_map(|v| [*v as u64, (*v >> 64) as u64])
            .collect();
        values.sort_unstable();
        values.dedup();
        values
    }

    fn each_qword_pair(mut check: impl FnMut(u64, u64)) {
        let values = interesting_qwords();
        for &a in &values {
            for &b in &values {
                check(a, b);
            }
        }
    }

    fn expect_mmx_state(rig: &Rig) {
        assert_eq!(rig.h.cpu.fsw & MMX_TOP_MASK, 0, "TOP is 0");
        assert_eq!(rig.h.cpu.ftw, ALL_TAGS_VALID, "every tag is valid");
    }

    #[test]
    fn mmx_lane_wise_integer_instructions_match_the_xmm_models_on_64_bits() {
        let mut rig = Rig::new();
        for (name, model) in integer_models() {
            if name.starts_with("pack") {
                continue;
            }
            let register_form = format!("{name} mm0, mm1");
            let memory_form = format!("{name} mm0, qword ptr [rsi]");
            each_qword_pair(|a, b| {
                let expected = model(a as u128, b as u128) as u64;
                assert_eq!(
                    mm_op(&mut rig, &register_form, a, b),
                    expected,
                    "{name} {a:#x} {b:#x}"
                );
                assert_eq!(mm(&rig, 1), b, "{name}: the source is untouched");
                assert_eq!(
                    mm_op_mem(&mut rig, &memory_form, a, b),
                    expected,
                    "{name} memory {a:#x} {b:#x}"
                );
            });
        }
    }

    #[test]
    fn mmx_instructions_write_all_ones_to_the_destination_exponent_only() {
        let mut rig = Rig::new();
        mm_op(&mut rig, "paddb mm0, mm1", 1, 2);
        assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
        assert_eq!(
            mm_exponent(&rig, 1),
            MMX_EXPONENT_MARKER,
            "reads leave the source alone"
        );
        for k in 2..8 {
            assert_eq!(rig.h.cpu.st[k], [0; 10], "other registers are untouched");
        }
    }

    #[test]
    fn mmx_reads_ignore_the_exponent_field() {
        let mut rig = Rig::new();
        for exponent in [
            [0u8, 0],
            [0xff, 0xff],
            [0xff, 0x7f],
            [0, 0x80],
            MMX_EXPONENT_MARKER,
        ] {
            set_mm(&mut rig, 0, 5);
            set_mm(&mut rig, 1, 7);
            rig.h.cpu.st[1][8..10].copy_from_slice(&exponent);
            rig.run_ok("paddd mm0, mm1");
            assert_eq!(mm(&rig, 0), 12);
        }
    }

    #[test]
    fn mmx_packs_saturate_the_low_halves_of_both_operands() {
        let mut rig = Rig::new();
        let saturate_i8 = |w: u16| saturate_signed(w as i16 as i64, 8) as i8 as u8;
        let saturate_u8 = |w: u16| saturate_unsigned(w as i16 as i64, 8) as u8;
        let saturate_i16 = |d: u32| saturate_signed(d as i32 as i64, 16) as i16 as u16;
        each_qword_pair(|a, b| {
            let (aw, bw) = (mm_words(a), mm_words(b));
            let signed: [u8; 8] = std::array::from_fn(|k| {
                if k < 4 {
                    saturate_i8(aw[k])
                } else {
                    saturate_i8(bw[k - 4])
                }
            });
            let unsigned: [u8; 8] = std::array::from_fn(|k| {
                if k < 4 {
                    saturate_u8(aw[k])
                } else {
                    saturate_u8(bw[k - 4])
                }
            });
            assert_eq!(
                mm_op(&mut rig, "packsswb mm0, mm1", a, b),
                from_mm_bytes(signed),
                "{a:#x} {b:#x}"
            );
            assert_eq!(
                mm_op(&mut rig, "packuswb mm0, mm1", a, b),
                from_mm_bytes(unsigned),
                "{a:#x} {b:#x}"
            );
            let (ad, bd) = (mm_dwords(a), mm_dwords(b));
            let dwords_packed: [u16; 4] = [
                saturate_i16(ad[0]),
                saturate_i16(ad[1]),
                saturate_i16(bd[0]),
                saturate_i16(bd[1]),
            ];
            assert_eq!(
                mm_op(&mut rig, "packssdw mm0, mm1", a, b),
                from_mm_words(dwords_packed),
                "{a:#x} {b:#x}"
            );
            assert_eq!(
                mm_op_mem(&mut rig, "packssdw mm0, qword ptr [rsi]", a, b),
                from_mm_words(dwords_packed)
            );
        });
    }

    #[test]
    fn mmx_pack_boundary_values() {
        let mut rig = Rig::new();
        let a = from_mm_words([0x7f, 0x80, 0xff7f, 0xff80]);
        let b = from_mm_words([0xff, 0x100, 0, 0x8000]);
        assert_eq!(
            mm_bytes(mm_op(&mut rig, "packsswb mm0, mm1", a, b)),
            [0x7f, 0x7f, 0x80, 0x80, 0x7f, 0x7f, 0, 0x80]
        );
        assert_eq!(
            mm_bytes(mm_op(&mut rig, "packuswb mm0, mm1", a, b)),
            [0x7f, 0x80, 0, 0, 0xff, 0xff, 0, 0]
        );
    }

    #[test]
    fn mmx_unpacks_interleave_the_low_or_high_halves() {
        let mut rig = Rig::new();
        each_qword_pair(|a, b| {
            let (ab, bb) = (mm_bytes(a), mm_bytes(b));
            let low_bytes: [u8; 8] =
                std::array::from_fn(|k| if k % 2 == 0 { ab[k / 2] } else { bb[k / 2] });
            let high_bytes: [u8; 8] = std::array::from_fn(|k| {
                if k % 2 == 0 {
                    ab[4 + k / 2]
                } else {
                    bb[4 + k / 2]
                }
            });
            assert_eq!(
                mm_op(&mut rig, "punpcklbw mm0, mm1", a, b),
                from_mm_bytes(low_bytes),
                "{a:#x} {b:#x}"
            );
            assert_eq!(
                mm_op(&mut rig, "punpckhbw mm0, mm1", a, b),
                from_mm_bytes(high_bytes),
                "{a:#x} {b:#x}"
            );
            let (aw, bw) = (mm_words(a), mm_words(b));
            let low_words = [aw[0], bw[0], aw[1], bw[1]];
            let high_words = [aw[2], bw[2], aw[3], bw[3]];
            assert_eq!(
                mm_op(&mut rig, "punpcklwd mm0, mm1", a, b),
                from_mm_words(low_words)
            );
            assert_eq!(
                mm_op(&mut rig, "punpckhwd mm0, mm1", a, b),
                from_mm_words(high_words)
            );
            let (ad, bd) = (mm_dwords(a), mm_dwords(b));
            assert_eq!(
                mm_op(&mut rig, "punpckldq mm0, mm1", a, b),
                from_mm_dwords([ad[0], bd[0]])
            );
            assert_eq!(
                mm_op(&mut rig, "punpckhdq mm0, mm1", a, b),
                from_mm_dwords([ad[1], bd[1]])
            );
            assert_eq!(
                mm_op_mem(&mut rig, "punpckhbw mm0, qword ptr [rsi]", a, b),
                from_mm_bytes(high_bytes)
            );
            assert_eq!(
                mm_op_mem(&mut rig, "punpcklbw mm0, dword ptr [rsi]", a, b),
                from_mm_bytes(low_bytes)
            );
            assert_eq!(
                mm_op_mem(&mut rig, "punpcklwd mm0, dword ptr [rsi]", a, b),
                from_mm_words(low_words)
            );
            assert_eq!(
                mm_op_mem(&mut rig, "punpckldq mm0, dword ptr [rsi]", a, b),
                from_mm_dwords([ad[0], bd[0]])
            );
        });
    }

    #[test]
    fn mmx_unpack_hand_checked() {
        let mut rig = Rig::new();
        let a = 0x0706_0504_0302_0100;
        let b = 0x1716_1514_1312_1110;
        assert_eq!(
            mm_op(&mut rig, "punpcklbw mm0, mm1", a, b),
            0x1303_1202_1101_1000
        );
        assert_eq!(
            mm_op(&mut rig, "punpckhbw mm0, mm1", a, b),
            0x1707_1606_1505_1404
        );
        assert_eq!(
            mm_op(&mut rig, "punpcklwd mm0, mm1", a, b),
            0x1312_0302_1110_0100
        );
        assert_eq!(
            mm_op(&mut rig, "punpckhwd mm0, mm1", a, b),
            0x1716_0706_1514_0504
        );
        assert_eq!(
            mm_op(&mut rig, "punpckldq mm0, mm1", a, b),
            0x1312_1110_0302_0100
        );
        assert_eq!(
            mm_op(&mut rig, "punpckhdq mm0, mm1", a, b),
            0x1716_1514_0706_0504
        );
    }

    #[test]
    fn mmx_instruction_with_the_same_register_for_both_operands() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 0, 0x0102_0304_0506_0708);
        rig.run_ok("paddb mm0, mm0");
        assert_eq!(mm(&rig, 0), 0x0204_0608_0a0c_0e10);
        set_mm(&mut rig, 0, 0x0102_0304_0506_0708);
        rig.run_ok("pxor mm0, mm0");
        assert_eq!(mm(&rig, 0), 0);
    }

    #[test]
    fn mmx_instructions_on_the_other_registers() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 7, 0x0101_0101_0101_0101);
        set_mm(&mut rig, 3, 0x0202_0202_0202_0202);
        rig.run_ok("paddb mm7, mm3");
        assert_eq!(mm(&rig, 7), 0x0303_0303_0303_0303);
        assert_eq!(mm_exponent(&rig, 7), [0xff, 0xff]);
        set_mm(&mut rig, 5, 0x1234);
        set_mm(&mut rig, 6, 0x1234);
        rig.run_ok("pcmpeqb mm5, mm6");
        assert_eq!(mm(&rig, 5), 0xffff_ffff_ffff_ffff);
    }

    const MMX_SHIFT_SAMPLES: [u64; 3] = [
        0x8001_7ffe_0123_fedc,
        0xffff_ffff_ffff_ffff,
        0x0123_4567_89ab_cdef,
    ];

    #[test]
    fn mmx_shifts_by_register_count_match_the_lane_model() {
        let mut rig = Rig::new();
        let counts = [
            0u64,
            1,
            7,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            255,
            0x1_0000_0000,
            u64::MAX,
            1 << 63,
        ];
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            for sample in MMX_SHIFT_SAMPLES {
                for count in counts {
                    let expected = shift_model(sample as u128, width, kind, count) as u64;
                    assert_eq!(
                        mm_op(&mut rig, &format!("{name} mm0, mm1"), sample, count),
                        expected,
                        "{name} {sample:#x} by {count:#x}"
                    );
                    assert_eq!(
                        mm_op_mem(
                            &mut rig,
                            &format!("{name} mm0, qword ptr [rsi]"),
                            sample,
                            count
                        ),
                        expected,
                        "{name} memory {sample:#x} by {count:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn mmx_shift_by_itself_uses_the_register_value_as_the_count() {
        let mut rig = Rig::new();
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            for value in [3u64, 0x8000_0000_0000_0003, 0xffff_ffff_ffff_ffff, 17] {
                set_mm(&mut rig, 0, value);
                rig.run_ok(&format!("{name} mm0, mm0"));
                assert_eq!(
                    mm(&rig, 0),
                    shift_model(value as u128, width, kind, value) as u64,
                    "{name} {value:#x}"
                );
            }
        }
    }

    #[test]
    fn mmx_shifts_by_immediate_match_the_lane_model() {
        let mut rig = Rig::new();
        for (name, width, kind) in SHIFT_INSTRUCTIONS {
            for count in [0u8, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255] {
                for sample in MMX_SHIFT_SAMPLES {
                    set_mm(&mut rig, 0, sample);
                    rig.run_ok(&format!("{name} mm0, {count}"));
                    assert_eq!(
                        mm(&rig, 0),
                        shift_model(sample as u128, width, kind, count as u64) as u64,
                        "{name} {sample:#x} by {count}"
                    );
                    assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
                }
            }
        }
    }

    #[test]
    fn mmx_shift_by_immediate_on_a_high_register() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 7, 0x00ff_00ff_00ff_00ff);
        rig.run_ok("psllw mm7, 3");
        assert_eq!(mm(&rig, 7), 0x07f8_07f8_07f8_07f8);
        assert_eq!(mm(&rig, 0), 0);
    }

    #[test]
    fn mmx_hand_checked_shift_values() {
        let mut rig = Rig::new();
        assert_eq!(
            mm_op(&mut rig, "psrlw mm0, mm1", 0xffff_ffff_ffff_ffff, 4),
            0x0fff_0fff_0fff_0fff
        );
        assert_eq!(
            mm_op(&mut rig, "psraw mm0, mm1", 0x8000_8000_8000_8000, 4),
            0xf800_f800_f800_f800
        );
        assert_eq!(
            mm_op(&mut rig, "psraw mm0, mm1", 0x8000_7fff_0001_ffff, 100),
            0xffff_0000_0000_ffff
        );
        assert_eq!(mm_op(&mut rig, "psllq mm0, mm1", 1, 63), 1 << 63);
        assert_eq!(mm_op(&mut rig, "psllq mm0, mm1", 1, 64), 0);
        assert_eq!(
            mm_op(&mut rig, "psrad mm0, mm1", 0x8000_0000_4000_0000, 31),
            0xffff_ffff_0000_0000
        );
    }

    #[test]
    fn movd_and_movq_into_an_mmx_register_zero_extend() {
        let mut rig = Rig::new();
        rig.h.cpu.gpr[RAX] = 0xdead_beef_1234_5678;
        set_mm(&mut rig, 0, u64::MAX);
        rig.run_ok("movd mm0, eax");
        assert_eq!(mm(&rig, 0), 0x1234_5678);
        assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
        rig.run_ok("movq mm0, rax");
        assert_eq!(mm(&rig, 0), 0xdead_beef_1234_5678);
        rig.h.write(DATA, &0x8899_aabb_ccdd_eeffu64.to_le_bytes());
        rig.run_ok("movd mm0, dword ptr [rsi]");
        assert_eq!(mm(&rig, 0), 0xccdd_eeff);
        rig.run_ok("movq mm0, qword ptr [rsi]");
        assert_eq!(mm(&rig, 0), 0x8899_aabb_ccdd_eeff);
        rig.h.cpu.gpr[8] = 0xffff_ffff_8000_0001;
        rig.run_ok("movd mm3, r8d");
        assert_eq!(mm(&rig, 3), 0x8000_0001);
        rig.h.cpu.gpr[9] = 0x0102_0304_0506_0708;
        rig.run_ok("movq mm3, r9");
        assert_eq!(mm(&rig, 3), 0x0102_0304_0506_0708);
    }

    #[test]
    fn movd_and_movq_out_of_an_mmx_register() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 0, 0x8899_aabb_ccdd_eeff);
        rig.h.cpu.gpr[RAX] = u64::MAX;
        rig.run_ok("movd eax, mm0");
        assert_eq!(
            rig.gpr(RAX),
            0xccdd_eeff,
            "a 32-bit destination zero-extends"
        );
        rig.run_ok("movq rax, mm0");
        assert_eq!(rig.gpr(RAX), 0x8899_aabb_ccdd_eeff);
        assert_eq!(
            mm_exponent(&rig, 0),
            MMX_EXPONENT_MARKER,
            "reading mm0 leaves its exponent"
        );
        set_mm(&mut rig, 3, 0x1111_2222_3333_4444);
        rig.h.cpu.gpr[9] = u64::MAX;
        rig.h.cpu.gpr[10] = 0;
        rig.run_ok("movd r9d, mm3");
        rig.run_ok("movq r10, mm3");
        assert_eq!(rig.gpr(9), 0x3333_4444);
        assert_eq!(rig.gpr(10), 0x1111_2222_3333_4444);
        fill_sentinel(&mut rig, DATA);
        rig.run_ok("movd dword ptr [rsi], mm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..20).collect::<Vec<_>>());
        assert_eq!(rig.h.read(DATA, 4), 0xccdd_eeffu32.to_le_bytes());
        fill_sentinel(&mut rig, DATA);
        rig.run_ok("movq qword ptr [rsi], mm0");
        assert_eq!(changed_bytes(&rig, DATA), (16..24).collect::<Vec<_>>());
        assert_eq!(rig.h.read_u64(DATA), 0x8899_aabb_ccdd_eeff);
    }

    #[test]
    fn movq_between_mmx_registers_and_the_store_form() {
        let mut rig = Rig::new();
        assert_eq!(
            mm_op(&mut rig, "movq mm0, mm1", 0x1111, 0x2222_3333_4444_5555),
            0x2222_3333_4444_5555
        );
        // 0F 7F /r with mod == 3 stores mm1 (reg) into mm0 (rm)
        assert_eq!(
            mm_op(
                &mut rig,
                ".byte 0x0f, 0x7f, 0xc8",
                0x1111,
                0x2222_3333_4444_5555
            ),
            0x2222_3333_4444_5555
        );
        assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
        set_mm(&mut rig, 2, 0x77);
        rig.run_ok("movq mm7, mm2");
        assert_eq!(mm(&rig, 7), 0x77);
        assert_eq!(
            mm_op_mem(
                &mut rig,
                "movq mm0, qword ptr [rsi]",
                0,
                0xfeed_face_cafe_beef
            ),
            0xfeed_face_cafe_beef
        );
    }

    #[test]
    fn movntq_stores_eight_bytes() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 0, 0x0123_4567_89ab_cdef);
        fill_sentinel(&mut rig, DATA);
        rig.run_ok("movntq qword ptr [rsi], mm0");
        assert_eq!(rig.h.read_u64(DATA), 0x0123_4567_89ab_cdef);
        assert_eq!(changed_bytes(&rig, DATA), (16..24).collect::<Vec<_>>());
        expect_mmx_state(&rig);
    }

    #[test]
    fn pshufw_selects_words_by_the_immediate() {
        let mut rig = Rig::new();
        let source = from_mm_words([0x1111, 0x2222, 0x3333, 0x4444]);
        for (imm, text) in [
            (0u8, "0"),
            (0x1b, "0x1b"),
            (0xe4, "0xe4"),
            (0xff, "0xff"),
            (0x4e, "0x4e"),
            (0x39, "0x39"),
            (0x93, "0x93"),
        ] {
            let words = mm_words(source);
            let expected: [u16; 4] =
                std::array::from_fn(|k| words[((imm >> (2 * k)) & 3) as usize]);
            assert_eq!(
                mm_op(
                    &mut rig,
                    &format!("pshufw mm0, mm1, {text}"),
                    0xaaaa,
                    source
                ),
                from_mm_words(expected),
                "imm {imm:#x}"
            );
        }
        assert_eq!(
            mm_op_mem(&mut rig, "pshufw mm0, qword ptr [rsi], 0x1b", 0, source),
            from_mm_words([0x4444, 0x3333, 0x2222, 0x1111])
        );
    }

    #[test]
    fn pextrw_and_pinsrw_use_two_index_bits_on_mmx_registers() {
        let mut rig = Rig::new();
        let value = from_mm_words([0x1000, 0x1111, 0x2222, 0xf333]);
        for index in [0usize, 1, 2, 3, 4, 7] {
            set_mm(&mut rig, 1, value);
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.run_ok(&format!("pextrw eax, mm1, {index}"));
            assert_eq!(
                rig.gpr(RAX),
                mm_words(value)[index & 3] as u64,
                "index {index}"
            );
            rig.h.cpu.gpr[RAX] = 0xffff_ffff_dead_beef;
            set_mm(&mut rig, 0, value);
            rig.run_ok(&format!("pinsrw mm0, eax, {index}"));
            let mut expected = mm_words(value);
            expected[index & 3] = 0xbeef;
            assert_eq!(mm(&rig, 0), from_mm_words(expected), "index {index}");
        }
        set_mm(&mut rig, 0, value);
        rig.h.write(DATA, &[0x34, 0x12]);
        rig.run_ok("pinsrw mm0, word ptr [rsi], 2");
        assert_eq!(mm(&rig, 0), from_mm_words([0x1000, 0x1111, 0x1234, 0xf333]));
        set_mm(&mut rig, 1, value);
        rig.h.cpu.gpr[9] = u64::MAX;
        rig.run_ok("pextrw r9d, mm1, 1");
        assert_eq!(rig.gpr(9), 0x1111);
        rig.h.cpu.gpr[8] = 0xabcd;
        set_mm(&mut rig, 3, value);
        rig.run_ok("pinsrw mm3, r8d, 3");
        assert_eq!(mm(&rig, 3), from_mm_words([0x1000, 0x1111, 0x2222, 0xabcd]));
    }

    #[test]
    fn pmovmskb_collects_eight_sign_bits_from_an_mmx_register() {
        let mut rig = Rig::new();
        each_qword_pair(|value, _| {
            let expected = mm_bytes(value)
                .iter()
                .enumerate()
                .fold(0u64, |mask, (k, b)| mask | (((b >> 7) as u64) << k));
            set_mm(&mut rig, 1, value);
            rig.h.cpu.gpr[RAX] = u64::MAX;
            rig.run_ok("pmovmskb eax, mm1");
            assert_eq!(rig.gpr(RAX), expected, "{value:#x}");
            rig.h.cpu.gpr[9] = u64::MAX;
            rig.run_ok("pmovmskb r9d, mm1");
            assert_eq!(rig.gpr(9), expected);
        });
    }

    #[test]
    fn maskmovq_stores_only_the_selected_bytes_at_rdi() {
        let mut rig = Rig::new();
        let data = 0x8877_6655_4433_2211u64;
        let selector = 0x0080_0080_8000_0080u64;
        set_mm(&mut rig, 0, data);
        set_mm(&mut rig, 1, selector);
        fill_sentinel(&mut rig, DATA);
        rig.reset();
        rig.h.cpu.gpr[RDI] = DATA + 3;
        assert_eq!(rig.exec("maskmovq mm0, mm1"), Stop::Halted);
        let written = rig.h.read(DATA - 16, GUARD_BYTES);
        for k in 0..8usize {
            let at = 16 + 3 + k;
            let expected = if (selector >> (8 * k + 7)) & 1 != 0 {
                (data >> (8 * k)) as u8
            } else {
                SENTINEL
            };
            assert_eq!(written[at], expected, "byte {k}");
        }
        let selected = (0..8)
            .filter(|k| (selector >> (8 * k + 7)) & 1 != 0)
            .count();
        assert_eq!(changed_bytes(&rig, DATA).len(), selected);
        expect_mmx_state(&rig);
    }

    #[test]
    fn maskmovq_with_an_empty_selector_touches_no_memory() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 0, u64::MAX);
        set_mm(&mut rig, 1, 0x7f7f_7f7f_7f7f_7f7f);
        rig.reset();
        rig.h.cpu.gpr[RDI] = UNMAPPED;
        assert_eq!(rig.exec("maskmovq mm0, mm1"), Stop::Halted);
        set_mm(&mut rig, 1, 0x80);
        assert_eq!(
            rig.exec("maskmovq mm0, mm1"),
            fault(PAGE_FAULT_VECTOR, Some(PAGE_FAULT_WRITE))
        );
        assert_eq!(rig.h.cpu.cr2, UNMAPPED);
    }

    #[test]
    fn cvtpi2ps_converts_two_integers_into_the_low_singles() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 1, from_mm_dwords([1, (-2i32) as u32]));
        rig.h.cpu.xmm[0] = lanes32([0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444]);
        rig.run_ok("cvtpi2ps xmm0, mm1");
        assert_eq!(
            dwords(rig.xmm(0)),
            [ONE_SINGLE, (-2.0f32).to_bits(), 0x3333_3333, 0x4444_4444]
        );
        expect_mmx_state(&rig);
        rig.h.write(
            DATA,
            &from_mm_dwords([0x7fff_ffff, 0x8000_0000]).to_le_bytes(),
        );
        rig.run_ok("cvtpi2ps xmm0, qword ptr [rsi]");
        assert_eq!(
            dwords(rig.xmm(0))[..2],
            [2147483648.0f32.to_bits(), (-2147483648.0f32).to_bits()]
        );
        expect_flag(&rig, FLAG_PRECISION);
        set_mm(&mut rig, 3, from_mm_dwords([5, 6]));
        rig.run_ok("cvtpi2ps xmm9, mm3");
        assert_eq!(
            dwords(rig.xmm(9))[..2],
            [5.0f32.to_bits(), 6.0f32.to_bits()]
        );
    }

    #[test]
    fn cvtpi2ps_honors_the_rounding_control() {
        let mut rig = Rig::new();
        for (rounding, expected) in [
            (
                RC_NEAREST,
                [TWO_POW_24_SINGLE, TWO_POW_24_SINGLE | SIGN_SINGLE],
            ),
            (
                RC_DOWN,
                [TWO_POW_24_SINGLE, TWO_POW_24_NEXT_SINGLE | SIGN_SINGLE],
            ),
            (
                RC_UP,
                [TWO_POW_24_NEXT_SINGLE, TWO_POW_24_SINGLE | SIGN_SINGLE],
            ),
            (
                RC_ZERO,
                [TWO_POW_24_SINGLE, TWO_POW_24_SINGLE | SIGN_SINGLE],
            ),
        ] {
            set_mm(
                &mut rig,
                1,
                from_mm_dwords([FIRST_INEXACT_INTEGER, FIRST_INEXACT_INTEGER.wrapping_neg()]),
            );
            rig.reset();
            rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
            assert_eq!(rig.exec("cvtpi2ps xmm0, mm1"), Stop::Halted);
            assert_eq!(dwords(rig.xmm(0))[..2], expected, "rc {rounding}");
        }
    }

    #[test]
    fn cvtpi2pd_converts_two_integers_exactly() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 1, from_mm_dwords([(-5i32) as u32, 7]));
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("cvtpi2pd xmm0, mm1");
        assert_eq!(rig.xmm(0), doubles([-5.0, 7.0]));
        rig.h.write(
            DATA,
            &from_mm_dwords([0x8000_0000, 0x7fff_ffff]).to_le_bytes(),
        );
        rig.run_ok("cvtpi2pd xmm0, qword ptr [rsi]");
        assert_eq!(rig.xmm(0), doubles([-2147483648.0, 2147483647.0]));
        assert_eq!(rig.mxcsr(), MXCSR_RESET);
        expect_mmx_state(&rig);
    }

    fn cvt_to_mm(rig: &mut Rig, asm: &str, rounding: u32, source: u128) -> u64 {
        rig.h.cpu.xmm[1] = source;
        set_mm(rig, 0, 0x1234);
        rig.reset();
        rig.h.cpu.mxcsr = MXCSR_RESET | (rounding << MXCSR_RC_SHIFT);
        assert_eq!(rig.exec(asm), Stop::Halted, "{asm}");
        mm(rig, 0)
    }

    #[test]
    fn cvtps2pi_rounds_per_mxcsr_and_cvttps2pi_truncates() {
        let mut rig = Rig::new();
        for (value, nearest, down, up, truncated) in [
            ([2.5f32, -2.5], [2i32, -2], [2, -3], [3, -2], [2, -2]),
            ([1.5, 3.5], [2, 4], [1, 3], [2, 4], [1, 3]),
            ([0.75, -0.75], [1, -1], [0, -1], [1, 0], [0, 0]),
        ] {
            let source = singles([value[0], value[1], 9.0, 9.0]);
            for (rounding, expected) in [(RC_NEAREST, nearest), (RC_DOWN, down), (RC_UP, up)] {
                let packed = from_mm_dwords([expected[0] as u32, expected[1] as u32]);
                assert_eq!(
                    cvt_to_mm(&mut rig, "cvtps2pi mm0, xmm1", rounding, source),
                    packed,
                    "{value:?} rc {rounding}"
                );
            }
            let packed = from_mm_dwords([truncated[0] as u32, truncated[1] as u32]);
            for rounding in ALL_ROUNDING_MODES {
                assert_eq!(
                    cvt_to_mm(&mut rig, "cvttps2pi mm0, xmm1", rounding, source),
                    packed,
                    "{value:?} rc {rounding}"
                );
            }
            assert_eq!(
                cvt_to_mm(&mut rig, "cvtps2pi mm0, xmm1", RC_ZERO, source),
                packed
            );
        }
    }

    #[test]
    fn cvtps2pi_out_of_range_gives_the_integer_indefinite() {
        let mut rig = Rig::new();
        let source = lanes32([f32::NAN.to_bits(), 3e9f32.to_bits(), 0, 0]);
        for asm in ["cvtps2pi mm0, xmm1", "cvttps2pi mm0, xmm1"] {
            assert_eq!(
                cvt_to_mm(&mut rig, asm, RC_NEAREST, source),
                from_mm_dwords([0x8000_0000, 0x8000_0000])
            );
            expect_flag(&rig, FLAG_INVALID);
        }
        rig.h.write(
            DATA,
            &from_mm_dwords([(-3.75f32).to_bits(), 1e10f32.to_bits()]).to_le_bytes(),
        );
        set_mm(&mut rig, 0, 0);
        rig.run_ok("cvttps2pi mm0, qword ptr [rsi]");
        assert_eq!(mm(&rig, 0), from_mm_dwords([(-3i32) as u32, 0x8000_0000]));
    }

    #[test]
    fn cvtpd2pi_and_cvttpd2pi_convert_two_doubles() {
        let mut rig = Rig::new();
        let source = doubles([-2.5, 7.9]);
        assert_eq!(
            cvt_to_mm(&mut rig, "cvtpd2pi mm0, xmm1", RC_NEAREST, source),
            from_mm_dwords([(-2i32) as u32, 8])
        );
        assert_eq!(
            cvt_to_mm(&mut rig, "cvtpd2pi mm0, xmm1", RC_DOWN, source),
            from_mm_dwords([(-3i32) as u32, 7])
        );
        assert_eq!(
            cvt_to_mm(&mut rig, "cvttpd2pi mm0, xmm1", RC_UP, source),
            from_mm_dwords([(-2i32) as u32, 7])
        );
        assert_eq!(
            cvt_to_mm(
                &mut rig,
                "cvtpd2pi mm0, xmm1",
                RC_NEAREST,
                doubles([1e10, f64::NAN])
            ),
            from_mm_dwords([0x8000_0000, 0x8000_0000])
        );
        rig.h.write(DATA, &source.to_le_bytes());
        rig.run_ok("cvttpd2pi mm0, xmmword ptr [rsi]");
        assert_eq!(mm(&rig, 0), from_mm_dwords([(-2i32) as u32, 7]));
        rig.h.cpu.xmm[10] = doubles([4.0, 5.0]);
        rig.run_ok("cvttps2pi mm5, xmm10");
        // the same register read as singles: the double 4.0 is 0x4010_0000_0000_0000, so the two
        // singles are 0.0 and 0x4010_0000 = 2.25
        assert_eq!(mm(&rig, 5), from_mm_dwords([0, 2]));
    }

    #[test]
    fn cvtpd2pi_with_a_misaligned_m128_operand_raises_gp0_but_m64_forms_do_not() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.gpr[RSI] = DATA + 8;
        set_mm(&mut rig, 0, 0x55);
        rig.h.cpu.fsw = 5 << 11;
        rig.h.cpu.ftw = 0x0f;
        assert_eq!(rig.exec("cvtpd2pi mm0, xmmword ptr [rsi]"), gp0());
        assert_eq!(mm(&rig, 0), 0x55);
        assert_eq!(
            (rig.h.cpu.fsw, rig.h.cpu.ftw),
            (5 << 11, 0x0f),
            "a fault leaves the x87 state alone"
        );
        assert_eq!(rig.exec("cvtps2pi mm0, qword ptr [rsi]"), Stop::Halted);
        assert_eq!(rig.exec("cvtpi2pd xmm0, qword ptr [rsi]"), Stop::Halted);
    }

    #[test]
    fn mmx_conversions_with_an_unmasked_exception_fault_before_any_write() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 1, from_mm_dwords([FIRST_INEXACT_INTEGER, 0]));
        rig.h.cpu.xmm[0] = A;
        rig.reset();
        rig.h.cpu.cr4 |= CR4_OSXMMEXCPT;
        rig.h.cpu.mxcsr = MXCSR_RESET & !(FLAG_PRECISION << MXCSR_MASK_SHIFT);
        rig.h.cpu.fsw = 5 << 11;
        rig.h.cpu.ftw = 0x0f;
        assert_eq!(rig.exec("cvtpi2ps xmm0, mm1"), xm());
        assert_eq!(rig.xmm(0), A);
        assert_eq!((rig.h.cpu.fsw, rig.h.cpu.ftw), (5 << 11, 0x0f));
        rig.h.cpu.xmm[1] = singles([2.5, 0.0, 0.0, 0.0]);
        set_mm(&mut rig, 0, 0x99);
        assert_eq!(rig.exec("cvtps2pi mm0, xmm1"), xm());
        assert_eq!(mm(&rig, 0), 0x99);
    }

    #[test]
    fn movq2dq_and_movdq2q_move_a_quadword_between_register_files() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 1, 0x1122_3344_5566_7788);
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("movq2dq xmm0, mm1");
        assert_eq!(
            rig.xmm(0),
            0x1122_3344_5566_7788,
            "the upper half is zeroed"
        );
        assert_eq!(mm_exponent(&rig, 1), MMX_EXPONENT_MARKER);
        expect_mmx_state(&rig);
        rig.h.cpu.xmm[1] = B;
        set_mm(&mut rig, 0, 0);
        rig.run_ok("movdq2q mm0, xmm1");
        assert_eq!(mm(&rig, 0), B as u64);
        assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
        set_mm(&mut rig, 3, 0xabcd);
        rig.run_ok("movq2dq xmm9, mm3");
        assert_eq!(rig.xmm(9), 0xabcd);
        rig.h.cpu.xmm[10] = A;
        rig.run_ok("movdq2q mm5, xmm10");
        assert_eq!(mm(&rig, 5), A as u64);
    }

    #[test]
    fn mixed_mmx_and_xmm_forms_need_osfxsr_but_pure_mmx_forms_do_not() {
        let mut rig = Rig::new();
        for key in [
            "cvtpi2ps xmm0, mm1",
            "cvtps2pi mm0, xmm1",
            "movq2dq xmm0, mm1",
            "movdq2q mm0, xmm1",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            rig.h.cpu.ftw = 0x0f;
            assert_eq!(rig.exec(key), ud(), "{key}");
            assert_eq!(rig.h.cpu.ftw, 0x0f, "{key}: a fault leaves the tags alone");
        }
        for key in [
            "paddb mm0, mm1",
            "movd mm0, eax",
            "pmovmskb eax, mm1",
            "emms",
        ] {
            rig.reset();
            rig.h.cpu.cr4 &= !CR4_OSFXSR;
            assert_eq!(rig.exec(key), Stop::Halted, "{key}");
        }
    }

    fn valid_mmx_only_keys(rig: &mut Rig) -> Vec<&'static str> {
        SNIPPETS
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| is_mmx_only(key) && !key.contains("; "))
            .filter(|key| rig.run(key) == Stop::Halted)
            .collect()
    }

    #[test]
    fn every_mmx_snippet_is_recognised_and_runs() {
        let mut rig = Rig::new();
        let keys = valid_mmx_only_keys(&mut rig);
        assert!(keys.len() > 300, "{}", keys.len());
    }

    #[test]
    fn pure_mmx_instructions_raise_ud_for_em_and_nm_for_ts_and_ignore_osfxsr() {
        let mut rig = Rig::new();
        for key in valid_mmx_only_keys(&mut rig) {
            for (cr0, cr4_clear, expected) in [
                (CR0_EM, 0, ud()),
                (CR0_TS, 0, nm()),
                (CR0_EM | CR0_TS, 0, ud()),
                (0, CR4_OSFXSR, Stop::Halted),
                (CR0_TS, CR4_OSFXSR, nm()),
            ] {
                rig.reset();
                rig.h.cpu.cr0 |= cr0;
                rig.h.cpu.cr4 &= !cr4_clear;
                rig.h.cpu.fsw = 5 << 11;
                rig.h.cpu.ftw = 0x0f;
                rig.h.cpu.xmm[0] = A;
                let stop = rig.exec(key);
                assert_eq!(
                    stop, expected,
                    "{key} with cr0 {cr0:#x} cr4 clear {cr4_clear:#x}"
                );
                if expected != Stop::Halted {
                    assert_eq!(rig.h.cpu.rip, CODE_BASE, "{key}");
                    assert_eq!(
                        (rig.h.cpu.fsw, rig.h.cpu.ftw),
                        (5 << 11, 0x0f),
                        "{key}: x87 state untouched by a fault"
                    );
                }
            }
        }
    }

    #[test]
    fn every_mmx_instruction_puts_the_x87_stack_in_mmx_state() {
        let mut rig = Rig::new();
        let mut checked = 0;
        for (key, _) in SNIPPETS {
            if !(is_mmx_only(key) || key.contains(" mm")) || key.contains("; ") {
                continue;
            }
            rig.h.cpu.fsw = (5 << 11) | FSW_INVALID_OPERATION;
            rig.h.cpu.ftw = 0x0f;
            rig.reset();
            rig.h.cpu.fsw = (5 << 11) | FSW_INVALID_OPERATION;
            rig.h.cpu.ftw = 0x0f;
            if rig.exec(key) != Stop::Halted {
                continue;
            }
            assert_eq!(
                rig.h.cpu.fsw, FSW_INVALID_OPERATION,
                "{key}: TOP cleared, other FSW bits kept"
            );
            assert_eq!(rig.h.cpu.ftw, ALL_TAGS_VALID, "{key}");
            checked += 1;
        }
        assert!(checked > 300, "{checked}");
    }

    #[test]
    fn emms_is_the_only_mmx_instruction_that_empties_the_tags_and_keeps_top() {
        let mut rig = Rig::new();
        rig.reset();
        rig.h.cpu.fsw = (5 << 11) | FSW_INVALID_OPERATION;
        rig.h.cpu.ftw = ALL_TAGS_VALID;
        assert_eq!(rig.exec("emms"), Stop::Halted);
        assert_eq!(rig.h.cpu.ftw, ALL_TAGS_EMPTY);
        assert_eq!(
            rig.h.cpu.fsw,
            (5 << 11) | FSW_INVALID_OPERATION,
            "emms leaves TOP and the status word"
        );
    }

    #[test]
    fn emms_restores_the_stack_for_x87_use() {
        let mut rig = Rig::new();
        rig.h.cpu.fsw = 0;
        rig.h.cpu.ftw = 0;
        rig.run_ok("paddb mm0, mm1; emms; fld1");
        // after emms TOP is still 0, so fld1 pushes into physical register 7
        assert_eq!(rig.h.cpu.fsw & MMX_TOP_MASK, 7 << 11);
        assert_eq!(rig.h.cpu.ftw, 0x80);
        assert_eq!(
            u64::from_le_bytes(rig.h.cpu.st[7][..8].try_into().unwrap()),
            X87_ONE_MANTISSA
        );
    }

    #[test]
    fn mmx_registers_alias_the_physical_x87_registers_whatever_top_is() {
        let mut rig = Rig::new();
        for top in 0..8u16 {
            rig.h.cpu.fsw = top << 11;
            rig.h.cpu.ftw = 0;
            for k in 0..8 {
                set_mm(&mut rig, k, 0x1000 + k as u64);
            }
            rig.h.cpu.gpr[RAX] = 0;
            rig.run_ok("movq rax, mm3");
            assert_eq!(rig.gpr(RAX), 0x1003, "top {top}");
        }
    }

    #[test]
    fn x87_values_are_visible_to_mmx_through_their_mantissa() {
        let mut rig = Rig::new();
        rig.h.cpu.fsw = 0;
        rig.h.cpu.ftw = 0;
        rig.h.cpu.gpr[RAX] = 0;
        // fld1 pushes into physical register 7 (TOP 0 -> 7)
        rig.run_ok("fld1; movq rax, mm7");
        assert_eq!(rig.gpr(RAX), X87_ONE_MANTISSA);
        expect_mmx_state(&rig);
        assert_eq!(
            mm_exponent(&rig, 7),
            [0xff, 0x3f],
            "a read leaves the 1.0 exponent field (0x3fff)"
        );
    }

    #[test]
    fn mmx_stores_and_reads_keep_the_exponent_but_still_enter_mmx_state() {
        let mut rig = Rig::new();
        set_mm(&mut rig, 0, 0x0102_0304_0506_0708);
        rig.h.cpu.fsw = 3 << 11;
        rig.h.cpu.ftw = 0x01;
        rig.run_ok("movq qword ptr [rsi], mm0");
        assert_eq!(mm_exponent(&rig, 0), MMX_EXPONENT_MARKER);
        expect_mmx_state(&rig);
        rig.h.cpu.ftw = 0;
        rig.h.cpu.gpr[RAX] = 0;
        rig.run_ok("pextrw eax, mm0, 0");
        assert_eq!(mm_exponent(&rig, 0), MMX_EXPONENT_MARKER);
        expect_mmx_state(&rig);
    }

    #[test]
    fn a_faulting_mmx_memory_access_leaves_the_x87_state_alone() {
        let mut rig = Rig::new();
        for key in [
            "movq mm0, qword ptr [rsi]",
            "paddb mm0, qword ptr [rsi]",
            "movd mm0, dword ptr [rsi]",
            "pshufw mm0, qword ptr [rsi], 0x1b",
        ] {
            set_mm(&mut rig, 0, 0x77);
            rig.reset();
            rig.h.cpu.gpr[RSI] = UNMAPPED;
            rig.h.cpu.fsw = 5 << 11;
            rig.h.cpu.ftw = 0x0f;
            assert_eq!(rig.exec(key), fault(PAGE_FAULT_VECTOR, Some(0)), "{key}");
            assert_eq!(mm(&rig, 0), 0x77, "{key}");
            assert_eq!((rig.h.cpu.fsw, rig.h.cpu.ftw), (5 << 11, 0x0f), "{key}");
        }
        for key in [
            "movq qword ptr [rsi], mm0",
            "movntq qword ptr [rsi], mm0",
            "movd dword ptr [rsi], mm0",
        ] {
            rig.reset();
            rig.h.cpu.gpr[RSI] = UNMAPPED;
            rig.h.cpu.fsw = 5 << 11;
            rig.h.cpu.ftw = 0x0f;
            assert_eq!(
                rig.exec(key),
                fault(PAGE_FAULT_VECTOR, Some(PAGE_FAULT_WRITE)),
                "{key}"
            );
            assert_eq!((rig.h.cpu.fsw, rig.h.cpu.ftw), (5 << 11, 0x0f), "{key}");
        }
    }

    #[test]
    fn mmx_memory_operands_need_no_alignment() {
        let mut rig = Rig::new();
        let source: Vec<u8> = (1u8..=24).collect();
        rig.h.write(DATA, &source);
        for offset in 1..9u64 {
            rig.reset();
            rig.h.cpu.gpr[RSI] = DATA + offset;
            assert_eq!(rig.exec("movq mm0, qword ptr [rsi]"), Stop::Halted);
            let expected = u64::from_le_bytes(
                source[offset as usize..offset as usize + 8]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(mm(&rig, 0), expected, "+{offset}");
        }
    }

    #[test]
    fn fxsave_captures_the_mmx_state_as_st_registers() {
        let mut rig = Rig::new();
        rig.h.cpu.fsw = 6 << 11;
        rig.h.cpu.ftw = 0;
        rig.h.write(DATA, &0x0123_4567_89ab_cdefu64.to_le_bytes());
        rig.run_ok("movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]");
        let image = rig.h.read(DATA + 0x100, FXSAVE_AREA_BYTES);
        let fsw = u16::from_le_bytes([image[2], image[3]]);
        assert_eq!(fsw & MMX_TOP_MASK, 0);
        assert_eq!(image[4], 0xff, "abridged tag word: all registers valid");
        // ST(0) is mm0 because TOP is 0; each register occupies 16 bytes from offset 32
        assert_eq!(&image[32..40], 0x0123_4567_89ab_cdefu64.to_le_bytes());
        assert_eq!(&image[40..42], [0xff, 0xff]);
    }

    #[test]
    fn fxrstor_round_trips_mmx_state_across_emms() {
        let mut rig = Rig::new();
        rig.h.cpu.fsw = 0;
        rig.h.cpu.ftw = 0;
        rig.h.write(DATA, &0xfeed_face_cafe_beefu64.to_le_bytes());
        rig.run_ok("movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]; emms; fxrstor [rsi + 0x100]");
        assert_eq!(mm(&rig, 0), 0xfeed_face_cafe_beef);
        assert_eq!(mm_exponent(&rig, 0), [0xff, 0xff]);
        expect_mmx_state(&rig);
    }

    #[test]
    fn mmx_sequence_adds_two_memory_operands() {
        let mut rig = Rig::new();
        rig.h.write(DATA, &0x0101_0101_0101_01ffu64.to_le_bytes());
        rig.h
            .write(DATA + 8, &0x0202_0202_0202_0201u64.to_le_bytes());
        rig.run_ok("movq mm0, qword ptr [rsi]; movq mm1, qword ptr [rsi + 8]; paddb mm0, mm1; movq qword ptr [rsi + 16], mm0; emms");
        assert_eq!(rig.h.read_u64(DATA + 16), 0x0303_0303_0303_0300);
        assert_eq!(rig.h.cpu.ftw, ALL_TAGS_EMPTY);
    }

    #[test]
    fn mmx_form_table() {
        use Prefix::*;
        for opcode in [
            0x60u8, 0x6b, 0x6e, 0x6f, 0x70, 0x71, 0x76, 0x7e, 0x7f, 0xc4, 0xc5, 0xd1, 0xd5, 0xd7,
            0xdf, 0xe0, 0xe5, 0xe7, 0xef, 0xf1, 0xf7, 0xfe,
        ] {
            assert!(mmx_form(opcode, Np), "{opcode:#x}");
            assert!(
                !mmx_form(opcode, P66),
                "{opcode:#x} with 66 is the xmm form"
            );
            assert!(
                !mmx_form(opcode, F3) && !mmx_form(opcode, F2),
                "{opcode:#x}"
            );
        }
        for opcode in [0x6cu8, 0x6d, 0x77, 0xd0, 0xd6, 0xe6, 0xf0, 0x10, 0x58, 0x2b] {
            assert!(!mmx_form(opcode, Np), "{opcode:#x}");
        }
        for opcode in [0x2au8, 0x2c, 0x2d] {
            assert!(mmx_form(opcode, Np) && mmx_form(opcode, P66));
            assert!(!mmx_form(opcode, F3) && !mmx_form(opcode, F2));
        }
        assert!(mmx_form(0xd6, F3) && mmx_form(0xd6, F2) && !mmx_form(0xd6, P66));
    }

    #[test]
    fn mmx_form_validity_of_the_new_encodings() {
        use Prefix::*;
        assert!(form_valid(0xf7, Np, true, 0) && !form_valid(0xf7, Np, false, 0));
        assert!(form_valid(0xd6, F3, true, 0) && !form_valid(0xd6, F3, false, 0));
        assert!(form_valid(0xd6, F2, true, 0) && !form_valid(0xd6, F2, false, 0));
        assert!(
            form_valid(0xd6, P66, false, 0),
            "66 0F D6 is movq xmm/m64, xmm"
        );
        for group in 0..8u8 {
            assert_eq!(
                form_valid(0x73, Np, true, group),
                matches!(group, 2 | 6),
                "np 73 /{group}"
            );
            assert_eq!(
                form_valid(0x73, P66, true, group),
                matches!(group, 2 | 3 | 6 | 7)
            );
            assert_eq!(
                form_valid(0x71, Np, true, group),
                matches!(group, 2 | 4 | 6)
            );
        }
    }

    #[test]
    fn invalid_mmx_encodings_raise_invalid_opcode_even_with_ts_set() {
        let mut rig = Rig::new();
        let mut keys: Vec<String> = [
            ".byte 0x0f, 0xd7, 0x06",
            ".byte 0x0f, 0xf7, 0x06",
            ".byte 0x0f, 0xe7, 0xc1",
            ".byte 0x0f, 0xc5, 0x06, 0x01",
            ".byte 0x0f, 0x71, 0x16, 0x01",
            ".byte 0x0f, 0x73, 0x16, 0x01",
            ".byte 0x0f, 0x6c, 0xc1",
            ".byte 0x0f, 0x6d, 0xc1",
            ".byte 0x0f, 0xd6, 0xc1",
            ".byte 0x0f, 0xe6, 0xc1",
            ".byte 0xf3, 0x0f, 0x60, 0xc1",
            ".byte 0xf2, 0x0f, 0x74, 0xc1",
            ".byte 0xf3, 0x0f, 0xd7, 0xc1",
        ]
        .iter()
        .map(|k| k.to_string())
        .collect();
        for opcode in ["0x71", "0x72"] {
            for group in [0u8, 1, 3, 5, 7] {
                keys.push(format!(
                    ".byte 0x0f, {opcode}, {:#04x}, 0x01",
                    0xc0 + group * 8
                ));
            }
        }
        for group in [0u8, 1, 3, 4, 5, 7] {
            keys.push(format!(".byte 0x0f, 0x73, {:#04x}, 0x01", 0xc0 + group * 8));
        }
        for key in &keys {
            for cr0 in [0, CR0_TS] {
                set_mm(&mut rig, 0, 0x55);
                rig.reset();
                rig.h.cpu.cr0 |= cr0;
                rig.h.cpu.fsw = 5 << 11;
                rig.h.cpu.ftw = 0x0f;
                assert_eq!(rig.exec(key), ud(), "{key} cr0 {cr0:#x}");
                assert_eq!(mm(&rig, 0), 0x55);
                assert_eq!((rig.h.cpu.fsw, rig.h.cpu.ftw), (5 << 11, 0x0f), "{key}");
            }
        }
    }

    #[test]
    fn xmm_forms_with_a_66_prefix_still_work_next_to_the_mmx_ones() {
        let mut rig = Rig::new();
        assert_eq!(
            rig.op("paddb xmm0, xmm1", A, B),
            from_byte_fn(|k| bytes(A)[k].wrapping_add(bytes(B)[k]))
        );
        assert_eq!(
            rig.h.cpu.ftw, 0,
            "xmm instructions do not touch the x87 tags"
        );
        rig.h.cpu.xmm[0] = A;
        rig.run_ok("pshufd xmm0, xmm1, 0x1b");
        rig.run_ok("movd xmm0, eax");
        assert_eq!(rig.h.cpu.fsw & MMX_TOP_MASK, 0);
    }

    #[test]
    fn jit_and_interpreter_agree_on_mmx_and_x87_sequences_for_every_top() {
        for key in [
            "fld1; movq rax, mm7",
            "fld1; fld1; movq mm0, mm1; emms",
            "movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]",
            "paddb mm0, mm1; emms",
            "paddb mm0, mm1; fld1",
            "paddb mm0, mm1; emms; fld1",
            "movq mm0, qword ptr [rsi]; fxsave [rsi + 0x100]; emms; fxrstor [rsi + 0x100]",
            "fldz; fld1; movq mm2, mm0; movq mm3, mm1; emms",
            "movq mm0, qword ptr [rsi]; movq mm1, qword ptr [rsi + 8]; paddb mm0, mm1; movq qword ptr [rsi + 16], mm0; emms",
        ] {
            for top in 0..8u16 {
                for tags in [0u16, 0xff, 0x5a] {
                    assert_jit_matches_interpreter(code(key), |h| {
                        h.cpu.cr4 |= CR4_OSFXSR;
                        h.cpu.gpr[RSI] = DATA;
                        differential_state(h, MXCSR_RESET, 0);
                        h.cpu.fsw = top << 11;
                        h.cpu.ftw = tags;
                    });
                }
            }
        }
    }

    #[test]
    fn jit_and_interpreter_agree_on_mmx_gating_and_misaligned_cvt_operands() {
        for key in [
            "paddb mm0, mm1",
            "cvtpd2pi mm0, xmmword ptr [rsi]",
            "cvtps2pi mm0, qword ptr [rsi]",
            "maskmovq mm0, mm1",
            "movq2dq xmm0, mm1",
        ] {
            for (cr0_bits, cr4_clear, rsi) in [
                (CR0_EM, 0, DATA),
                (CR0_TS, 0, DATA),
                (0, CR4_OSFXSR, DATA),
                (0, 0, DATA + 8),
                (0, 0, UNMAPPED),
            ] {
                assert_jit_matches_interpreter(code(key), |h| {
                    h.cpu.cr4 |= CR4_OSFXSR;
                    differential_state(h, MXCSR_RESET, 0);
                    h.cpu.gpr[RSI] = rsi;
                    h.cpu.gpr[RDI] = rsi;
                    h.cpu.cr0 |= cr0_bits;
                    h.cpu.cr4 &= !cr4_clear;
                });
            }
        }
    }
}
