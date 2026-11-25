//! The host-FPU fast path (`sse_host.rs`) against the soft engine, its oracle: every operation on
//! edge and random operands under every MXCSR environment that changes arithmetic, comparing the
//! result bits and the flags. The wrapped functions (`arithmetic`, `fused_lane`, ...) take the
//! host path when it applies and the soft one otherwise, so equality with the `soft_*` function
//! on every input is the whole claim.

use super::fma::{FmaForm, FmaSign, fused_lane, soft_fused_lane};
use super::*;

const RC_SHIFTED: [u32; 4] = [
    ROUND_NEAREST_EVEN << MXCSR_RC_SHIFT,
    ROUND_DOWN << MXCSR_RC_SHIFT,
    ROUND_UP << MXCSR_RC_SHIFT,
    ROUND_TOWARD_ZERO << MXCSR_RC_SHIFT,
];
/// MXCSR with every exception masked.
const ALL_MASKED: u32 = MXCSR_EXCEPTION_FLAGS << MXCSR_MASK_SHIFT;

/// Every MXCSR setting the arithmetic reads: the four rounding modes, DAZ, FTZ and the underflow
/// mask (the only mask the soft engine consults; the others only matter in `fp_commit`, which
/// sees the same flags either way).
fn environments() -> impl Iterator<Item = u32> {
    (0..32u32).map(|k| {
        let mut mxcsr = ALL_MASKED | RC_SHIFTED[(k & 3) as usize];
        if k & 4 != 0 {
            mxcsr |= MXCSR_DAZ;
        }
        if k & 8 != 0 {
            mxcsr |= MXCSR_FTZ;
        }
        if k & 16 != 0 {
            mxcsr &= !MXCSR_UNDERFLOW_MASK;
        }
        mxcsr
    })
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Operands of every class the fast path has to refuse or get right: zeros, denormals, the
/// smallest normal and its neighbours, the largest finite and its neighbours, infinities, quiet
/// and signaling NaNs with payloads, the real indefinite, ones, small integers, and normals in a
/// narrow exponent range (so sums cancel and products land near the edges).
fn edge_values(format: Format) -> Vec<u64> {
    let fraction = format.fraction_mask();
    let min_normal = 1u64 << format.fraction_bits;
    let one = (format.bias() as u64) << format.fraction_bits;
    let mut values = vec![
        0,
        1,
        2,
        fraction,
        fraction - 1,
        min_normal,
        min_normal + 1,
        min_normal * 2,
        min_normal * 2 - 1,
        min_normal * 3,
        format.max_finite(false),
        format.max_finite(false) - 1,
        format.max_finite(false) - min_normal,
        format.infinity(false),
        format.infinity(false) | format.quiet_bit(),
        format.infinity(false) | format.quiet_bit() | 0x5a5,
        format.infinity(false) | 1,
        format.infinity(false) | 0x1234,
        format.indefinite() & !format.sign_bit(),
        one,
        one + 1,
        one - 1,
        one + (1 << (format.fraction_bits - 1)),
        one - min_normal,
        one + min_normal,
        // 2^(fraction_bits), the first double/single above which every value is an integer
        one + (format.fraction_bits as u64) * min_normal,
        one + (format.fraction_bits as u64 + 1) * min_normal,
        // 2^31 and 2^63 and their neighbours, for the integer conversions
        one + 31 * min_normal,
        one + 31 * min_normal - 1,
        one + 31 * min_normal + 1,
        one + 30 * min_normal + fraction,
        one + 63 * min_normal,
        one + 63 * min_normal - 1,
        one + 62 * min_normal + fraction,
        // 0.5, 1.5, 2.5
        one - min_normal,
        one + (1 << (format.fraction_bits - 1)),
        one + min_normal + (1 << (format.fraction_bits - 2)),
    ];
    let signed: Vec<u64> = values.iter().map(|v| v | format.sign_bit()).collect();
    values.extend(signed);
    values
}

fn random_value(format: Format, rng: &mut XorShift, edges: &[u64]) -> u64 {
    let width_mask = format.lane_mask() as u64;
    let one_exponent = format.bias() as u64;
    match rng.below(8) {
        0 => edges[rng.below(edges.len() as u64) as usize],
        1 => rng.next() & width_mask,
        // An edge value nudged by a few ulps.
        2 => {
            let base = edges[rng.below(edges.len() as u64) as usize];
            base.wrapping_add(rng.below(5)).wrapping_sub(2) & width_mask
        }
        // Few significant bits: exact results, ties and halfway cases.
        3 => {
            let exponent = one_exponent + rng.below(40) - 20;
            let fraction = (rng.below(16) << (format.fraction_bits - 4)) & format.fraction_mask();
            (rng.below(2) << (format.width() - 1)) | exponent << format.fraction_bits | fraction
        }
        // Exponents near the top and bottom of the range, for overflow, underflow and FTZ.
        4 => {
            let exponent = if rng.below(2) == 0 {
                rng.below(format.fraction_bits as u64 + 4)
            } else {
                format.exponent_mask() - 1 - rng.below(4)
            };
            (rng.below(2) << (format.width() - 1))
                | exponent << format.fraction_bits
                | (rng.next() & format.fraction_mask())
        }
        // Ordinary numbers of moderate size.
        _ => {
            let exponent = one_exponent + rng.below(60) - 30;
            (rng.below(2) << (format.width() - 1))
                | exponent << format.fraction_bits
                | (rng.next() & format.fraction_mask())
        }
    }
}

/// A second operand related to the first: equal, negated, a few ulps away (cancellation), or
/// unrelated.
fn partner(format: Format, rng: &mut XorShift, first: u64, edges: &[u64]) -> u64 {
    let width_mask = format.lane_mask() as u64;
    match rng.below(6) {
        0 => first,
        1 => first ^ format.sign_bit(),
        2 => {
            (first ^ format.sign_bit())
                .wrapping_add(rng.below(7))
                .wrapping_sub(3)
                & width_mask
        }
        3 => first.wrapping_add(rng.below(7)).wrapping_sub(3) & width_mask,
        _ => random_value(format, rng, edges),
    }
}

const CASES: usize = 4000;
const FORMATS: [Format; 2] = [SINGLE, DOUBLE];
const OPERATIONS: [Arithmetic; 5] = [
    Arithmetic::Add,
    Arithmetic::Subtract,
    Arithmetic::Multiply,
    Arithmetic::Divide,
    Arithmetic::SquareRoot,
];

fn format_name(format: Format) -> &'static str {
    if format.fraction_bits == SINGLE.fraction_bits {
        "single"
    } else {
        "double"
    }
}

/// Runs `fast` and `soft` on fresh environments from `mxcsr` and asserts equal bits and flags.
fn check<T: PartialEq + core::fmt::Debug>(
    mxcsr: u32,
    what: impl Fn() -> String,
    fast: impl FnOnce(&mut FpEnv) -> T,
    soft: impl FnOnce(&mut FpEnv) -> T,
) {
    let (mut fast_env, mut soft_env) = (FpEnv::new(mxcsr), FpEnv::new(mxcsr));
    let fast_result = fast(&mut fast_env);
    let soft_result = soft(&mut soft_env);
    assert_eq!(
        (&fast_result, fast_env.flags),
        (&soft_result, soft_env.flags),
        "{} under MXCSR {mxcsr:#x} (left: with the host path, right: soft engine)",
        what()
    );
}

#[test]
fn arithmetic_matches_the_soft_engine() {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    for format in FORMATS {
        let edges = edge_values(format);
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for &a in &edges {
            for &b in &edges {
                pairs.push((a, b));
            }
        }
        for _ in 0..CASES {
            let a = random_value(format, &mut rng, &edges);
            pairs.push((a, partner(format, &mut rng, a, &edges)));
        }
        for mxcsr in environments() {
            for &(a, b) in &pairs {
                for operation in OPERATIONS {
                    check(
                        mxcsr,
                        || format!("{} {operation:?} {a:#x} {b:#x}", format_name(format)),
                        |env| arithmetic(env, format, operation, a, b),
                        |env| soft_arithmetic(env, format, operation, a, b),
                    );
                }
                for maximum in [false, true] {
                    check(
                        mxcsr,
                        || format!("{} max={maximum} {a:#x} {b:#x}", format_name(format)),
                        |env| minimum_maximum(env, format, maximum, a, b),
                        |env| soft_minimum_maximum(env, format, maximum, a, b),
                    );
                }
                for signal_quiet_nan in [false, true] {
                    check(
                        mxcsr,
                        || format!("{} compare {a:#x} {b:#x}", format_name(format)),
                        |env| compare(env, format, a, b, signal_quiet_nan),
                        |env| soft_compare(env, format, a, b, signal_quiet_nan),
                    );
                }
            }
        }
    }
}

/// The packed path picks a lane function once per instruction and mixes host and soft lanes.
#[test]
fn packed_binary_matches_the_soft_engine_lane_by_lane() {
    let mut rng = XorShift(0x0123_4567_89ab_cdef);
    for format in FORMATS {
        let edges = edge_values(format);
        for mxcsr in environments() {
            for _ in 0..CASES / 8 {
                let mut dest = 0u128;
                let mut source = 0u128;
                for k in 0..format.lanes() {
                    let a = random_value(format, &mut rng, &edges);
                    lane_put(&mut dest, format, k, a);
                    lane_put(&mut source, format, k, partner(format, &mut rng, a, &edges));
                }
                for opcode in [0x51, 0x58, 0x59, 0x5c, 0x5d, 0x5e, 0x5f] {
                    check(
                        mxcsr,
                        || format!("{} {opcode:#x} {dest:#x} {source:#x}", format_name(format)),
                        |env| host::packed_binary(env, format, opcode, dest, source),
                        |env| {
                            let mut result = 0;
                            for k in 0..format.lanes() {
                                let (a, b) =
                                    (lane_get(dest, format, k), lane_get(source, format, k));
                                let lane = match opcode {
                                    0x5d | 0x5f => {
                                        soft_minimum_maximum(env, format, opcode == 0x5f, a, b)
                                    }
                                    _ => {
                                        let operation = match opcode {
                                            0x51 => Arithmetic::SquareRoot,
                                            0x58 => Arithmetic::Add,
                                            0x59 => Arithmetic::Multiply,
                                            0x5c => Arithmetic::Subtract,
                                            _ => Arithmetic::Divide,
                                        };
                                        soft_arithmetic(env, format, operation, a, b)
                                    }
                                };
                                lane_put(&mut result, format, k, lane);
                            }
                            result
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn fused_multiply_add_matches_the_soft_engine() {
    let mut rng = XorShift(0x5851_f42d_4c95_7f2d);
    let forms = [FmaForm::Form132, FmaForm::Form213, FmaForm::Form231];
    for format in FORMATS {
        let edges = edge_values(format);
        let mut triples: Vec<[u64; 3]> = Vec::new();
        for _ in 0..CASES * 2 {
            let x = random_value(format, &mut rng, &edges);
            let y = random_value(format, &mut rng, &edges);
            let z = match rng.below(3) {
                // An addend close to minus the product: near-total cancellation.
                0 => {
                    let mut env = FpEnv::new(ALL_MASKED);
                    let product = soft_arithmetic(&mut env, format, Arithmetic::Multiply, x, y);
                    (product ^ format.sign_bit())
                        .wrapping_add(rng.below(5))
                        .wrapping_sub(2)
                        & format.lane_mask() as u64
                }
                _ => random_value(format, &mut rng, &edges),
            };
            triples.push([x, y, z]);
        }
        // x = y = (1 + 2^-52) * 2^-511, z = -(1 + 2^-51) * 2^-1022: the result is 2^-1126, a
        // nonzero product and addend that underflow together.
        if format.fraction_bits == DOUBLE.fraction_bits {
            let x = (1023 - 511) << 52 | 1;
            triples.push([x, x, 1 << 63 | 1 << 52 | 2]);
        }
        for mxcsr in environments() {
            for &sources in &triples {
                for form in forms {
                    for negate_product in [false, true] {
                        for sign in [FmaSign::Add, FmaSign::Subtract] {
                            check(
                                mxcsr,
                                || format!("{} fma {sources:#x?}", format_name(format)),
                                |env| fused_lane(env, format, sources, form, negate_product, sign),
                                |env| {
                                    soft_fused_lane(
                                        env,
                                        format,
                                        sources,
                                        form,
                                        negate_product,
                                        sign,
                                    )
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn conversions_match_the_soft_engine() {
    let mut rng = XorShift(0xd1b5_4a32_d192_ed03);
    let mut integers: Vec<i64> = vec![
        0,
        1,
        -1,
        i64::MIN,
        i64::MAX,
        i64::MIN + 1,
        i32::MIN as i64,
        i32::MAX as i64,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        (1 << 25) + 2,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        (1 << 54) + 2,
        -(1 << 53) - 1,
    ];
    for _ in 0..CASES {
        let bits = rng.below(64) as u32;
        let value = (rng.next() >> (63 - bits)) as i64;
        integers.push(if rng.below(2) == 0 { value } else { -value });
        integers.push(rng.next() as i32 as i64);
    }
    for mxcsr in environments() {
        for format in FORMATS {
            for &value in &integers {
                check(
                    mxcsr,
                    || format!("{} from {value}", format_name(format)),
                    |env| int_to_float(env, format, value),
                    |env| soft_int_to_float(env, format, value),
                );
            }
            let edges = edge_values(format);
            let mut floats = edges.clone();
            for _ in 0..CASES {
                floats.push(random_value(format, &mut rng, &edges));
            }
            for &bits in &floats {
                for width in [32, 64] {
                    for truncate in [false, true] {
                        check(
                            mxcsr,
                            || {
                                format!(
                                    "{} {bits:#x} to i{width} t={truncate}",
                                    format_name(format)
                                )
                            },
                            |env| float_to_int(env, format, bits, width, truncate),
                            |env| soft_float_to_int(env, format, bits, width, truncate),
                        );
                    }
                }
                let to = if format.fraction_bits == SINGLE.fraction_bits {
                    DOUBLE
                } else {
                    SINGLE
                };
                check(
                    mxcsr,
                    || format!("{} {bits:#x} converted", format_name(format)),
                    |env| convert_float(env, format, to, bits),
                    |env| soft_convert_float(env, format, to, bits),
                );
            }
        }
    }
}

/// Doubles at the edges of the integer conversions that the edge table cannot spell as bit
/// patterns: the halves around 2^31 and 2^63 and around zero.
#[test]
fn integer_conversion_edges_match_the_soft_engine() {
    let two_31 = 2147483648.0f64;
    let two_63 = 9223372036854775808.0f64;
    let values = [
        two_31 - 0.5,
        two_31 - 0.25,
        two_31 - 1.5,
        two_31 + 0.5,
        -two_31 - 0.5,
        -two_31 - 0.25,
        -two_31 + 0.5,
        -two_31 - 1.0,
        two_63,
        -two_63,
        two_63 - 1024.0,
        -two_63 - 2048.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.49999999999999994,
        4503599627370495.5,
        4503599627370496.0,
        -4503599627370495.5,
    ];
    for mxcsr in environments() {
        for value in values {
            for width in [32, 64] {
                for truncate in [false, true] {
                    let bits = value.to_bits();
                    check(
                        mxcsr,
                        || format!("{value} to i{width} t={truncate}"),
                        |env| float_to_int(env, DOUBLE, bits, width, truncate),
                        |env| soft_float_to_int(env, DOUBLE, bits, width, truncate),
                    );
                    let narrow = value as f32;
                    if narrow as f64 == value {
                        let bits = narrow.to_bits() as u64;
                        check(
                            mxcsr,
                            || format!("{value}f32 to i{width} t={truncate}"),
                            |env| float_to_int(env, SINGLE, bits, width, truncate),
                            |env| soft_float_to_int(env, SINGLE, bits, width, truncate),
                        );
                    }
                }
            }
        }
    }
}
