//! The host-FPU fast path of the soft-float engine (`x-ssefp`, `docs/plans/31-host-fpu.md`).
//!
//! Every function here answers one lane with the host's own `f32`/`f64` arithmetic when the
//! answer is provably the bits and MXCSR flags the soft engine would give, and returns `None`
//! otherwise, which sends the lane to the soft engine. The soft engine stays the oracle:
//! `sse_host_tests.rs` checks the two against each other over edge and random operands in every
//! rounding mode, with DAZ, FTZ and the masks toggled.
//!
//! What makes the answer provable is a narrow domain, checked on the integer bits before any
//! float is made:
//! - Operands are normal numbers or zeros. No NaN, infinity or denormal reaches the host, so the
//!   x86 NaN rules, DAZ and the denormal-operand flag never matter, and no NaN is ever produced
//!   (the hosts differ in NaN propagation and wasm leaves NaN bits unspecified). The invalid and
//!   divide-by-zero cases (0/0, x/0, sqrt of a negative) are refused too.
//! - Results are nonzero normal numbers above the smallest normal one, or zeros that are exact by
//!   construction (a sum that cancels exactly, a product with a zero factor). So the host never
//!   overflows, underflows or flushes, and FTZ, the underflow and overflow flags and the soft
//!   engine's tininess simplification at the smallest normal (see `round_pack`) never come in.
//! - Rounding is round-to-nearest-even, the one mode every host gives Rust code (Rust assumes the
//!   default floating-point environment: a host thread running with another mode or with
//!   flush-to-zero is already undefined behavior for all Rust float code). Operations that cannot
//!   round (min, max, compares, exact conversions) run in any mode.
//!
//! In that domain the only flag left is precision, and it is computed exactly, without host flags
//! (wasm has none, and LLVM does not order float operations against flag reads): an error-free
//! transformation for sums, exact products in a wider type for single precision, and an odd-
//! significand comparison on integers for double precision.

use super::*;
#[cfg(freestanding)]
use crate::math::FloatExt;
use core::cmp::Ordering;

/// One bit above the widest significand: a product of odd significands longer than this many
/// bits did not fit in a double.
const DOUBLE_SIGNIFICAND_BITS: u32 = DOUBLE.fraction_bits + 1;
/// 2^52: adding and subtracting it rounds a double of smaller magnitude to an integer under the
/// current (nearest-even) rounding, because the sum's unit in the last place is 1.
const DOUBLE_INTEGER_THRESHOLD: f64 = (1u64 << DOUBLE.fraction_bits) as f64;

/// A normal number or a zero: an operand the host may see.
#[inline]
fn ordinary(format: Format, bits: u64) -> bool {
    let exponent = (bits >> format.fraction_bits) & format.exponent_mask();
    if exponent == 0 {
        bits & format.fraction_mask() == 0
    } else {
        exponent != format.exponent_mask()
    }
}

#[inline]
fn is_zero(format: Format, bits: u64) -> bool {
    bits & !format.sign_bit() == 0
}

/// A nonzero result the soft engine rounds exactly as the host did and flags only with precision:
/// finite, and larger in magnitude than the smallest normal number. The smallest normal itself is
/// refused because an inexact result that rounded up to it is tiny on real hardware but not in the
/// soft engine; leaving it to the soft engine keeps that difference in one place.
#[inline]
fn safe_result(format: Format, bits: u64) -> bool {
    let magnitude = bits & !format.sign_bit();
    magnitude > 1 << format.fraction_bits
        && magnitude >> format.fraction_bits != format.exponent_mask()
}

/// The significand of a normal number with its trailing zeros removed.
#[inline]
fn odd_significand(format: Format, bits: u64) -> u64 {
    let significand = (bits & format.fraction_mask()) | 1 << format.fraction_bits;
    significand >> significand.trailing_zeros()
}

#[inline]
fn flag_inexact(env: &mut FpEnv, bits: u64, inexact: bool) -> u64 {
    if inexact {
        env.flags |= FLAG_PRECISION;
    }
    bits
}

#[inline]
fn single(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

#[inline]
fn double(bits: u64) -> f64 {
    f64::from_bits(bits)
}

#[inline]
fn single_bits(value: f32) -> u64 {
    value.to_bits() as u64
}

/// The rounding error of `sum = x + y` (Knuth's TwoSum): exact under round-to-nearest when the sum
/// did not overflow, so it is zero exactly when the sum was exact.
macro_rules! sum_error {
    ($x:expr, $y:expr, $sum:expr) => {{
        let (x, y, sum) = ($x, $y, $sum);
        let y_part = sum - x;
        (x - (sum - y_part)) + (y - y_part)
    }};
}

#[inline]
fn add_single(a: u64, b: u64) -> Option<(u64, bool)> {
    let (x, y) = (single(a), single(b));
    let sum = x + y;
    let bits = single_bits(sum);
    // A sum of normals or zeros is zero only when it cancels exactly; the host's signed-zero rule
    // under nearest-even is x86's.
    if sum == 0.0 {
        return Some((bits, false));
    }
    if !safe_result(SINGLE, bits) {
        return None;
    }
    Some((bits, sum_error!(x, y, sum) != 0.0))
}

#[inline]
fn add_double(a: u64, b: u64) -> Option<(u64, bool)> {
    let (x, y) = (double(a), double(b));
    let sum = x + y;
    let bits = sum.to_bits();
    if sum == 0.0 {
        return Some((bits, false));
    }
    if !safe_result(DOUBLE, bits) {
        return None;
    }
    Some((bits, sum_error!(x, y, sum) != 0.0))
}

/// Negating the subtrahend does not round, so subtraction is addition.
#[inline]
fn subtract_single(a: u64, b: u64) -> Option<(u64, bool)> {
    add_single(a, b ^ SINGLE.sign_bit())
}

#[inline]
fn subtract_double(a: u64, b: u64) -> Option<(u64, bool)> {
    add_double(a, b ^ DOUBLE.sign_bit())
}

/// The product of two singles has at most 48 significant bits and fits a double exactly, so the
/// narrowing is the only rounding and comparing it back tells whether it was exact.
#[inline]
fn multiply_single(a: u64, b: u64) -> Option<(u64, bool)> {
    let product = single(a) as f64 * single(b) as f64;
    let rounded = product as f32;
    let bits = single_bits(rounded);
    if is_zero(SINGLE, a) || is_zero(SINGLE, b) {
        return Some((bits, false));
    }
    if !safe_result(SINGLE, bits) {
        return None;
    }
    Some((bits, rounded as f64 != product))
}

/// The exact product's significand is the product of the odd significands times a power of two,
/// so it fits a double exactly when that integer product has at most 53 bits.
#[inline]
fn multiply_double(a: u64, b: u64) -> Option<(u64, bool)> {
    let bits = (double(a) * double(b)).to_bits();
    if is_zero(DOUBLE, a) || is_zero(DOUBLE, b) {
        return Some((bits, false));
    }
    if !safe_result(DOUBLE, bits) {
        return None;
    }
    let product = odd_significand(DOUBLE, a) as u128 * odd_significand(DOUBLE, b) as u128;
    Some((
        bits,
        u128::BITS - product.leading_zeros() > DOUBLE_SIGNIFICAND_BITS,
    ))
}

/// Exact when quotient times divisor gives the dividend back; for singles that product is exact
/// in a double.
#[inline]
fn divide_single(a: u64, b: u64) -> Option<(u64, bool)> {
    if is_zero(SINGLE, b) {
        return None;
    }
    let quotient = single(a) / single(b);
    let bits = single_bits(quotient);
    if is_zero(SINGLE, a) {
        return Some((bits, false));
    }
    if !safe_result(SINGLE, bits) {
        return None;
    }
    let back = quotient as f64 * single(b) as f64;
    Some((bits, back != single(a) as f64))
}

/// Exact when odd(quotient) * odd(divisor) == odd(dividend). The converse needs the powers of two
/// to agree too, which they must: the rounded quotient is within half an ulp of the real one.
#[inline]
fn divide_double(a: u64, b: u64) -> Option<(u64, bool)> {
    if is_zero(DOUBLE, b) {
        return None;
    }
    let bits = (double(a) / double(b)).to_bits();
    if is_zero(DOUBLE, a) {
        return Some((bits, false));
    }
    if !safe_result(DOUBLE, bits) {
        return None;
    }
    let back = odd_significand(DOUBLE, bits) as u128 * odd_significand(DOUBLE, b) as u128;
    Some((bits, back != odd_significand(DOUBLE, a) as u128))
}

/// sqrt(+-0) is the zero itself, as in the soft engine; a negative nonzero operand is invalid and
/// left to it.
#[inline]
fn square_root_single(b: u64) -> Option<(u64, bool)> {
    if is_zero(SINGLE, b) {
        return Some((b, false));
    }
    if b & SINGLE.sign_bit() != 0 {
        return None;
    }
    let root = single(b).sqrt();
    let bits = single_bits(root);
    if !safe_result(SINGLE, bits) {
        return None;
    }
    let wide = root as f64;
    Some((bits, wide * wide != single(b) as f64))
}

#[inline]
fn square_root_double(b: u64) -> Option<(u64, bool)> {
    if is_zero(DOUBLE, b) {
        return Some((b, false));
    }
    if b & DOUBLE.sign_bit() != 0 {
        return None;
    }
    let bits = double(b).sqrt().to_bits();
    if !safe_result(DOUBLE, bits) {
        return None;
    }
    let root = odd_significand(DOUBLE, bits) as u128;
    Some((bits, root * root != odd_significand(DOUBLE, b) as u128))
}

fn binary(format: Format, operation: Arithmetic, a: u64, b: u64) -> Option<(u64, bool)> {
    let is_single = format.fraction_bits == SINGLE.fraction_bits;
    match (operation, is_single) {
        (Arithmetic::Add, true) => add_single(a, b),
        (Arithmetic::Add, false) => add_double(a, b),
        (Arithmetic::Subtract, true) => subtract_single(a, b),
        (Arithmetic::Subtract, false) => subtract_double(a, b),
        (Arithmetic::Multiply, true) => multiply_single(a, b),
        (Arithmetic::Multiply, false) => multiply_double(a, b),
        (Arithmetic::Divide, true) => divide_single(a, b),
        (Arithmetic::Divide, false) => divide_double(a, b),
        (Arithmetic::SquareRoot, true) => square_root_single(b),
        (Arithmetic::SquareRoot, false) => square_root_double(b),
    }
}

/// `arithmetic` on the host: `dest` and `source` as there.
pub(super) fn arithmetic(
    env: &mut FpEnv,
    format: Format,
    operation: Arithmetic,
    dest: u64,
    source: u64,
) -> Option<u64> {
    if env.rounding != ROUND_NEAREST_EVEN
        || !ordinary(format, source)
        || (operation != Arithmetic::SquareRoot && !ordinary(format, dest))
    {
        return None;
    }
    let (bits, inexact) = binary(format, operation, dest, source)?;
    Some(flag_inexact(env, bits, inexact))
}

/// The packed sqrt/add/mul/sub/min/div/max forms (`binary_operation` on every lane): each lane on
/// the host when it can be, on the soft engine when not. Lanes are independent and the flags a
/// sticky OR, so mixing the two is the same as running every lane on the soft engine.
pub(super) fn packed_binary(
    env: &mut FpEnv,
    format: Format,
    opcode: u8,
    dest: u128,
    source: u128,
) -> u128 {
    let mut result = 0;
    let operation = match opcode {
        0x51 => Arithmetic::SquareRoot,
        0x58 => Arithmetic::Add,
        0x59 => Arithmetic::Multiply,
        0x5c => Arithmetic::Subtract,
        0x5e => Arithmetic::Divide,
        _ => {
            let maximum = opcode == 0x5f;
            return if format.fraction_bits == SINGLE.fraction_bits {
                packed_minimum_maximum(env, SINGLE, maximum, dest, source)
            } else {
                packed_minimum_maximum(env, DOUBLE, maximum, dest, source)
            };
        }
    };
    if env.rounding != ROUND_NEAREST_EVEN {
        for k in 0..format.lanes() {
            let (a, b) = (lane_get(dest, format, k), lane_get(source, format, k));
            lane_put(
                &mut result,
                format,
                k,
                soft_arithmetic(env, format, operation, a, b),
            );
        }
        return result;
    }
    // One loop per lane function, so the lane function and the format fold into it.
    let lanes = Lanes {
        operation,
        dest,
        source,
    };
    let is_single = format.fraction_bits == SINGLE.fraction_bits;
    match (operation, is_single) {
        (Arithmetic::Add, true) => lanes.run(env, SINGLE, add_single),
        (Arithmetic::Add, false) => lanes.run(env, DOUBLE, add_double),
        (Arithmetic::Subtract, true) => lanes.run(env, SINGLE, subtract_single),
        (Arithmetic::Subtract, false) => lanes.run(env, DOUBLE, subtract_double),
        (Arithmetic::Multiply, true) => lanes.run(env, SINGLE, multiply_single),
        (Arithmetic::Multiply, false) => lanes.run(env, DOUBLE, multiply_double),
        (Arithmetic::Divide, true) => lanes.run(env, SINGLE, divide_single),
        (Arithmetic::Divide, false) => lanes.run(env, DOUBLE, divide_double),
        (Arithmetic::SquareRoot, true) => lanes.run(env, SINGLE, |_, b| square_root_single(b)),
        (Arithmetic::SquareRoot, false) => lanes.run(env, DOUBLE, |_, b| square_root_double(b)),
    }
}

struct Lanes {
    operation: Arithmetic,
    dest: u128,
    source: u128,
}

impl Lanes {
    #[inline(always)]
    fn run(
        &self,
        env: &mut FpEnv,
        format: Format,
        lane: impl Fn(u64, u64) -> Option<(u64, bool)>,
    ) -> u128 {
        let mut result = 0;
        for k in 0..format.lanes() {
            let (a, b) = (
                lane_get(self.dest, format, k),
                lane_get(self.source, format, k),
            );
            let operands_ordinary = ordinary(format, b)
                && (self.operation == Arithmetic::SquareRoot || ordinary(format, a));
            let bits = match operands_ordinary.then(|| lane(a, b)).flatten() {
                Some((bits, inexact)) => flag_inexact(env, bits, inexact),
                None => soft_arithmetic(env, format, self.operation, a, b),
            };
            lane_put(&mut result, format, k, bits);
        }
        result
    }
}

#[inline(always)]
fn packed_minimum_maximum(
    env: &mut FpEnv,
    format: Format,
    maximum: bool,
    dest: u128,
    source: u128,
) -> u128 {
    let mut result = 0;
    for k in 0..format.lanes() {
        let (a, b) = (lane_get(dest, format, k), lane_get(source, format, k));
        let bits = minimum_maximum(format, maximum, a, b)
            .unwrap_or_else(|| soft_minimum_maximum(env, format, maximum, a, b));
        lane_put(&mut result, format, k, bits);
    }
    result
}

/// Orders normal numbers and zeros by their bits: sign-magnitude, with both zeros equal.
#[inline]
fn ordering_key(format: Format, bits: u64) -> i64 {
    let magnitude = (bits & !format.sign_bit()) as i64;
    if bits & format.sign_bit() != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// `minimum_maximum` without the soft engine: no rounding, so any mode. Ties (including the two
/// zeros) return the source, as there.
#[inline]
pub(super) fn minimum_maximum(
    format: Format,
    maximum: bool,
    dest: u64,
    source: u64,
) -> Option<u64> {
    if !ordinary(format, dest) || !ordinary(format, source) {
        return None;
    }
    let ordering = ordering_key(format, dest).cmp(&ordering_key(format, source));
    let first = if maximum {
        ordering == Ordering::Greater
    } else {
        ordering == Ordering::Less
    };
    Some(if first { dest } else { source })
}

/// `compare` without the soft engine: ordered operands raise nothing, in any mode.
#[inline]
pub(super) fn compare(format: Format, dest: u64, source: u64) -> Option<Ordering> {
    if !ordinary(format, dest) || !ordinary(format, source) {
        return None;
    }
    Some(ordering_key(format, dest).cmp(&ordering_key(format, source)))
}

/// `int_to_float`. A magnitude with no more significant bits than the format's significand is
/// exact in every mode; anything longer rounds, which Rust's `as` does once, to nearest-even.
pub(super) fn int_to_float(env: &mut FpEnv, format: Format, value: i64) -> Option<u64> {
    let magnitude = value.unsigned_abs();
    let span = u64::BITS.saturating_sub(magnitude.leading_zeros() + magnitude.trailing_zeros());
    let exact = span <= format.fraction_bits + 1;
    if !exact && env.rounding != ROUND_NEAREST_EVEN {
        return None;
    }
    let bits = if format.fraction_bits == SINGLE.fraction_bits {
        single_bits(value as f32)
    } else {
        (value as f64).to_bits()
    };
    Some(flag_inexact(env, bits, !exact))
}

/// `float_to_int`. Truncation (cvtt*) runs in any mode; rounding conversions only under
/// nearest-even. The range check is on the rounded value, strict at the top: anything at or past
/// 2^(width-1) and anything below -2^(width-1) is invalid, which the soft engine handles.
pub(super) fn float_to_int(
    env: &mut FpEnv,
    format: Format,
    bits: u64,
    width: u32,
    truncate: bool,
) -> Option<u64> {
    if !ordinary(format, bits) || (!truncate && env.rounding != ROUND_NEAREST_EVEN) {
        return None;
    }
    // Every single is exactly a double.
    let value = if format.fraction_bits == SINGLE.fraction_bits {
        single(bits) as f64
    } else {
        double(bits)
    };
    let limit = (1u64 << (width - 1)) as f64;
    let integral = if value.abs() >= DOUBLE_INTEGER_THRESHOLD {
        value
    } else if truncate {
        // In range below 2^52, so `as` truncates exactly and the integer converts back exactly.
        (value as i64) as f64
    } else {
        let shift = DOUBLE_INTEGER_THRESHOLD.copysign(value);
        (value + shift) - shift
    };
    if integral >= limit || integral < -limit {
        return None;
    }
    let integer = integral as i64 as u64;
    let integer = if width == 32 {
        integer & 0xffff_ffff
    } else {
        integer
    };
    Some(flag_inexact(env, integer, integral != value))
}

/// `convert_float`. Widening a normal number or zero is exact in any mode; narrowing rounds once
/// under nearest-even.
pub(super) fn convert_float(env: &mut FpEnv, from: Format, bits: u64) -> Option<u64> {
    if !ordinary(from, bits) {
        return None;
    }
    if from.fraction_bits == SINGLE.fraction_bits {
        return Some((single(bits) as f64).to_bits());
    }
    if env.rounding != ROUND_NEAREST_EVEN {
        return None;
    }
    let narrowed = double(bits) as f32;
    let result = single_bits(narrowed);
    if is_zero(DOUBLE, bits) {
        return Some(result);
    }
    if !safe_result(SINGLE, result) {
        return None;
    }
    Some(flag_inexact(env, result, narrowed as f64 != double(bits)))
}

/// `fused_lane` on the host, `x * y + z` with the operands already chosen and signed.
pub(super) fn fused(env: &mut FpEnv, format: Format, x: u64, y: u64, z: u64) -> Option<u64> {
    if env.rounding != ROUND_NEAREST_EVEN
        || !ordinary(format, x)
        || !ordinary(format, y)
        || !ordinary(format, z)
    {
        return None;
    }
    let (bits, inexact) = if format.fraction_bits == SINGLE.fraction_bits {
        fused_single(x, y, z)?
    } else {
        fused_double(x, y, z)?
    };
    Some(flag_inexact(env, bits, inexact))
}

/// Single-precision fused multiply-add through doubles, with no host FMA. The product is exact in
/// a double; the sum is rounded to a double *to odd* (TwoSum gives its exact error, and an inexact
/// sum with an even significand is stepped one ulp toward that error), and a value rounded to odd
/// with at least two bits more than the target rounds correctly to the target (53 >= 24 + 2).
fn fused_single(x: u64, y: u64, z: u64) -> Option<(u64, bool)> {
    let product = single(x) as f64 * single(y) as f64;
    let addend = single(z) as f64;
    let sum = product + addend;
    if sum == 0.0 {
        // Exact cancellation, or a zero product plus a zero: the host's signed-zero rule is x86's.
        return Some((single_bits(sum as f32), false));
    }
    let error = sum_error!(product, addend, sum);
    let mut sum_bits = sum.to_bits();
    if error != 0.0 && sum_bits & 1 == 0 {
        if (error > 0.0) == (sum > 0.0) {
            sum_bits += 1;
        } else {
            sum_bits -= 1;
        }
    }
    let odd = f64::from_bits(sum_bits);
    let rounded = odd as f32;
    let bits = single_bits(rounded);
    if !safe_result(SINGLE, bits) {
        return None;
    }
    Some((bits, error != 0.0 || rounded as f64 != odd))
}

/// The exponent band in which `fused_double`'s error terms cannot underflow or overflow: products
/// and results between 2^-900 and 2^1000 leave 106 bits of headroom below and a little above.
const FUSED_DOUBLE_MIN_EXPONENT: i32 = -900;
const FUSED_DOUBLE_MAX_EXPONENT: i32 = 1000;

fn unbiased_exponent(bits: u64) -> i32 {
    ((bits >> DOUBLE.fraction_bits) & DOUBLE.exponent_mask()) as i32 - DOUBLE.bias()
}

fn in_fused_band(exponent: i32) -> bool {
    (FUSED_DOUBLE_MIN_EXPONENT..=FUSED_DOUBLE_MAX_EXPONENT).contains(&exponent)
}

/// Double-precision fused multiply-add on a host instruction that rounds once, where there is
/// one; elsewhere the soft engine does it.
#[cfg(target_arch = "aarch64")]
fn host_fused_double(x: f64, y: f64, z: f64) -> Option<f64> {
    let result: f64;
    // SAFETY: `fmadd` only reads and writes the named registers.
    unsafe {
        core::arch::asm!(
            "fmadd {result:d}, {x:d}, {y:d}, {z:d}",
            result = lateout(vreg) result,
            x = in(vreg) x,
            y = in(vreg) y,
            z = in(vreg) z,
            options(pure, nomem, nostack),
        );
    }
    Some(result)
}

#[cfg(all(target_arch = "x86_64", not(freestanding)))]
fn host_fused_double(x: f64, y: f64, z: f64) -> Option<f64> {
    #[target_feature(enable = "fma")]
    fn fused(x: f64, y: f64, z: f64) -> f64 {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(x), _mm_set_sd(y), _mm_set_sd(z)))
    }
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the host has FMA.
        Some(unsafe { fused(x, y, z) })
    } else {
        None
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", not(freestanding))
)))]
fn host_fused_double(_: f64, _: f64, _: f64) -> Option<f64> {
    None
}

/// Inexact detection is Boldo and Muller's ErrFma ("Exact and Approximated Error of the FMA",
/// IEEE TC 2011) up to its gamma: x*y + z - result is exactly gamma + alpha2 when nothing
/// underflows, which the exponent band guarantees.
fn fused_double(x: u64, y: u64, z: u64) -> Option<(u64, bool)> {
    let (a, b, c) = (double(x), double(y), double(z));
    let product_is_zero = is_zero(DOUBLE, x) || is_zero(DOUBLE, y);
    if !product_is_zero && !in_fused_band(unbiased_exponent(x) + unbiased_exponent(y)) {
        return None;
    }
    if !is_zero(DOUBLE, z) && !in_fused_band(unbiased_exponent(z)) {
        return None;
    }
    let result = host_fused_double(a, b, c)?;
    let bits = result.to_bits();
    if product_is_zero {
        // The sum is the addend, or a zero by the add rule: exact.
        return Some((bits, false));
    }
    if result == 0.0 {
        // Inside the band the exact sum is a multiple of 2^-1005 or so, a normal number when it is
        // not zero, so a zero result is an exact cancellation (+0, as in the soft engine). Outside
        // it, a zero can be an underflow (2^-1126 from a nonzero product and addend): that case
        // never gets here.
        return Some((bits, false));
    }
    if !safe_result(DOUBLE, bits) || !in_fused_band(unbiased_exponent(bits)) {
        return None;
    }
    let product_high = a * b;
    let product_low = host_fused_double(a, b, -product_high)?;
    let alpha_high = c + product_low;
    let alpha_low = sum_error!(c, product_low, alpha_high);
    let beta_high = product_high + alpha_high;
    let beta_low = sum_error!(product_high, alpha_high, beta_high);
    let gamma = (beta_high - result) + beta_low;
    Some((bits, gamma != -alpha_low))
}

impl Cpu {
    /// The 128-bit kernel of packed sqrt/add/mul/sub/min/div/max (`0F 51`, `58`-`5F` with no
    /// prefix or 66) for the EVEX lane loop, without re-entering the legacy decoder through
    /// `run_legacy_registers`. `None` for any other instruction. Flags are committed per call, as
    /// the legacy kernel does, so an unmasked exception in a later lane leaves the earlier lanes'
    /// flags in MXCSR exactly as before.
    pub(in crate::cpu) fn packed_float_lane(
        &mut self,
        i: &Insn,
        destination: u128,
        source: u128,
    ) -> Option<ExecResult<u128>> {
        let op = i.op & !(VEX_OP_FLAG | EVEX_OP_FLAG);
        let opcode = (op & 0xff) as u8;
        if op >> 8 != OPCODE_MAP_0F
            || i.rep != 0
            || !matches!(opcode, 0x51 | 0x58 | 0x59 | 0x5c..=0x5f)
        {
            return None;
        }
        let format = if i.p66 { DOUBLE } else { SINGLE };
        let mut env = FpEnv::new(self.mxcsr);
        let result = packed_binary(&mut env, format, opcode, destination, source);
        if env.flags != 0
            && let Err(exception) = self.fp_commit(&env)
        {
            return Some(Err(exception));
        }
        Some(Ok(result))
    }
}
