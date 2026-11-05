//! Unsigned integers of any size, for RSA: Montgomery exponentiation, division, inverses.
//! Nothing here is constant time.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

const LIMB_BITS: usize = 64;
const LIMB_BYTES: usize = 8;
/// Exponentiation consumes the exponent four bits at a time, so a 1024-bit exponent costs
/// 1024 squarings and 256 multiplications instead of about 512.
const WINDOW_BITS: usize = 4;
const WINDOW_ENTRIES: usize = 1 << WINDOW_BITS;
const WINDOW_MASK: u64 = WINDOW_ENTRIES as u64 - 1;

/// Little-endian 64-bit limbs with no zero limb on top, so zero has no limbs and equal values
/// have equal representations (which `PartialEq` relies on).
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct BigUint {
    limbs: Vec<u64>,
}

impl core::fmt::Debug for BigUint {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("0x")?;
        if self.limbs.is_empty() {
            return formatter.write_str("0");
        }
        for (index, limb) in self.limbs.iter().rev().enumerate() {
            if index == 0 {
                write!(formatter, "{limb:x}")?;
            } else {
                write!(formatter, "{limb:016x}")?;
            }
        }
        Ok(())
    }
}

impl BigUint {
    pub(crate) fn zero() -> Self {
        Self::default()
    }

    pub(crate) fn from_u64(value: u64) -> Self {
        Self::from_limbs(vec![value])
    }

    fn from_limbs(mut limbs: Vec<u64>) -> Self {
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        Self { limbs }
    }

    pub(crate) fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut limbs = vec![0u64; bytes.len().div_ceil(LIMB_BYTES)];
        for (index, &byte) in bytes.iter().rev().enumerate() {
            limbs[index / LIMB_BYTES] |= u64::from(byte) << (8 * (index % LIMB_BYTES));
        }
        Self::from_limbs(limbs)
    }

    /// Big-endian, left-padded to `length` bytes; `None` when the value needs more.
    pub(crate) fn to_be_bytes(&self, length: usize) -> Option<Vec<u8>> {
        if self.bits().div_ceil(8) > length {
            return None;
        }
        Some(
            (0..length)
                .rev()
                .map(|index| {
                    self.limbs
                        .get(index / LIMB_BYTES)
                        .map_or(0, |limb| (limb >> (8 * (index % LIMB_BYTES))) as u8)
                })
                .collect(),
        )
    }

    /// Big-endian with no leading zeros (and no bytes at all for zero).
    pub(crate) fn to_minimal_be_bytes(&self) -> Vec<u8> {
        self.to_be_bytes(self.bits().div_ceil(8))
            .unwrap_or_default()
    }

    /// The limbs zero-extended to `count`, for fixed-width Montgomery arithmetic; the value
    /// must fit.
    fn padded(&self, count: usize) -> Vec<u64> {
        let mut limbs = self.limbs.clone();
        limbs.resize(count, 0);
        limbs
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    pub(crate) fn is_one(&self) -> bool {
        self.limbs == [1]
    }

    pub(crate) fn is_odd(&self) -> bool {
        self.limbs.first().is_some_and(|limb| limb & 1 == 1)
    }

    pub(crate) fn bits(&self) -> usize {
        self.limbs.last().map_or(0, |top| {
            self.limbs.len() * LIMB_BITS - top.leading_zeros() as usize
        })
    }

    pub(crate) fn trailing_zeros(&self) -> usize {
        let zero_limbs = self.limbs.iter().take_while(|&&limb| limb == 0).count();
        self.limbs.get(zero_limbs).map_or(0, |limb| {
            zero_limbs * LIMB_BITS + limb.trailing_zeros() as usize
        })
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        let (long, short) = if self.limbs.len() >= other.limbs.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut limbs = long.limbs.clone();
        limbs.push(0);
        add_in_place(&mut limbs, &short.limbs);
        Self::from_limbs(limbs)
    }

    /// `self - other`; the caller guarantees `self >= other` (otherwise the result wraps).
    pub(crate) fn sub(&self, other: &Self) -> Self {
        debug_assert!(*self >= *other);
        let mut limbs = self.limbs.clone();
        subtract_in_place(&mut limbs, &other.limbs);
        Self::from_limbs(limbs)
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        let mut product = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (index, &mine) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (word, &theirs) in product[index..].iter_mut().zip(&other.limbs) {
                let sum = *word as u128 + mine as u128 * theirs as u128 + carry;
                *word = sum as u64;
                carry = sum >> LIMB_BITS;
            }
            product[index + other.limbs.len()] = carry as u64;
        }
        Self::from_limbs(product)
    }

    pub(crate) fn shift_left(&self, bits: usize) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let (limb_shift, bit_shift) = (bits / LIMB_BITS, bits % LIMB_BITS);
        let mut limbs = vec![0u64; self.limbs.len() + limb_shift + 1];
        for (index, &limb) in self.limbs.iter().enumerate() {
            limbs[index + limb_shift] |= limb << bit_shift;
            if bit_shift != 0 {
                limbs[index + limb_shift + 1] = limb >> (LIMB_BITS - bit_shift);
            }
        }
        Self::from_limbs(limbs)
    }

    pub(crate) fn shift_right(&self, bits: usize) -> Self {
        let (limb_shift, bit_shift) = (bits / LIMB_BITS, bits % LIMB_BITS);
        let Some(kept) = self.limbs.get(limb_shift..) else {
            return Self::zero();
        };
        let limbs = (0..kept.len())
            .map(|index| {
                let high = kept.get(index + 1).copied().unwrap_or(0);
                if bit_shift == 0 {
                    kept[index]
                } else {
                    (kept[index] >> bit_shift) | (high << (LIMB_BITS - bit_shift))
                }
            })
            .collect();
        Self::from_limbs(limbs)
    }

    /// Quotient and remainder; `None` for a zero divisor. Shift and subtract, one quotient
    /// bit per step, so it costs in proportion to the quotient's length: cheap for the
    /// Euclidean algorithm's mostly one-bit quotients and for one-off key arithmetic, too
    /// slow for anything per block.
    pub(crate) fn div_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        if divisor.is_zero() {
            return None;
        }
        if self < divisor {
            return Some((Self::zero(), self.clone()));
        }
        let shift = self.bits() - divisor.bits();
        let mut remainder = self.limbs.clone();
        let mut shifted = divisor.shift_left(shift).padded(remainder.len());
        let mut quotient = vec![0u64; shift / LIMB_BITS + 1];
        for position in (0..=shift).rev() {
            if !less_than(&remainder, &shifted) {
                subtract_in_place(&mut remainder, &shifted);
                quotient[position / LIMB_BITS] |= 1 << (position % LIMB_BITS);
            }
            shift_right_one_in_place(&mut shifted);
        }
        Some((Self::from_limbs(quotient), Self::from_limbs(remainder)))
    }

    pub(crate) fn rem(&self, divisor: &Self) -> Option<Self> {
        self.div_rem(divisor).map(|(_, remainder)| remainder)
    }

    /// `self mod divisor` for a nonzero divisor below 2^32. Works through 32-bit halves of
    /// each limb so every step is a 64-bit division, which is what makes sieving fast.
    pub(crate) fn rem_u32(&self, divisor: u32) -> u32 {
        let divisor = u64::from(divisor.max(1));
        let mut remainder = 0u64;
        for &limb in self.limbs.iter().rev() {
            remainder = ((remainder << 32) | (limb >> 32)) % divisor;
            remainder = ((remainder << 32) | (limb & u64::from(u32::MAX))) % divisor;
        }
        remainder as u32
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normalized, so more limbs is larger.
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

/// Compares equal-length limb slices.
fn less_than(a: &[u64], b: &[u64]) -> bool {
    a.iter().rev().lt(b.iter().rev())
}

/// `target += addend`, with `target` at least as long; returns the carry out of the top.
fn add_in_place(target: &mut [u64], addend: &[u64]) -> bool {
    let mut carry = false;
    for (index, word) in target.iter_mut().enumerate() {
        let theirs = addend.get(index).copied().unwrap_or(0);
        if theirs == 0 && !carry && index >= addend.len() {
            break;
        }
        let (sum, first) = word.overflowing_add(theirs);
        let (sum, second) = sum.overflowing_add(carry as u64);
        *word = sum;
        carry = first || second;
    }
    carry
}

/// `target -= subtrahend`, with `target` at least as long; returns the borrow out of the top.
fn subtract_in_place(target: &mut [u64], subtrahend: &[u64]) -> bool {
    let mut borrow = false;
    for (index, word) in target.iter_mut().enumerate() {
        let theirs = subtrahend.get(index).copied().unwrap_or(0);
        if theirs == 0 && !borrow && index >= subtrahend.len() {
            break;
        }
        let (difference, first) = word.overflowing_sub(theirs);
        let (difference, second) = difference.overflowing_sub(borrow as u64);
        *word = difference;
        borrow = first || second;
    }
    borrow
}

fn shift_right_one_in_place(limbs: &mut [u64]) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut().rev() {
        let next = *limb << (LIMB_BITS - 1);
        *limb = (*limb >> 1) | carry;
        carry = next;
    }
}

pub(crate) fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A Bezout coefficient in the extended Euclidean algorithm, which goes negative.
#[derive(Clone)]
struct Signed {
    magnitude: BigUint,
    negative: bool,
}

impl Signed {
    /// `self - quotient * other`.
    fn minus_product(&self, quotient: &BigUint, other: &Self) -> Self {
        let product = quotient.mul(&other.magnitude);
        if self.negative != other.negative {
            // Opposite signs: the magnitudes add and keep self's sign.
            return Self {
                magnitude: self.magnitude.add(&product),
                negative: self.negative,
            };
        }
        if self.magnitude >= product {
            Self {
                magnitude: self.magnitude.sub(&product),
                negative: self.negative,
            }
        } else {
            Self {
                magnitude: product.sub(&self.magnitude),
                negative: !self.negative,
            }
        }
    }
}

/// `value^-1 mod modulus`, or `None` when they share a factor or the modulus is below 2.
pub(crate) fn mod_inverse(value: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    if modulus.bits() < 2 {
        return None;
    }
    let mut previous_remainder = modulus.clone();
    let mut remainder = value.rem(modulus)?;
    // Only the coefficient of `value` is tracked: previous_remainder = coefficient * value
    // (mod modulus) holds throughout.
    let mut previous = Signed {
        magnitude: BigUint::zero(),
        negative: false,
    };
    let mut current = Signed {
        magnitude: BigUint::from_u64(1),
        negative: false,
    };
    while !remainder.is_zero() {
        let (quotient, next_remainder) = previous_remainder.div_rem(&remainder)?;
        let next = previous.minus_product(&quotient, &current);
        previous_remainder = core::mem::replace(&mut remainder, next_remainder);
        previous = core::mem::replace(&mut current, next);
    }
    if !previous_remainder.is_one() {
        return None;
    }
    let magnitude = previous.magnitude.rem(modulus)?;
    Some(if previous.negative && !magnitude.is_zero() {
        modulus.sub(&magnitude)
    } else {
        magnitude
    })
}

/// Montgomery arithmetic modulo an odd modulus above 1, with R = 2^(64 * limbs). Values in
/// Montgomery form are fixed-width limb vectors, all `modulus.len()` long.
pub(crate) struct Montgomery {
    modulus: Vec<u64>,
    modulus_value: BigUint,
    /// -modulus^-1 mod 2^64.
    inverse: u64,
    /// R^2 mod modulus, which takes a number into Montgomery form.
    r_squared: Vec<u64>,
    /// R mod modulus: one in Montgomery form.
    one: Vec<u64>,
}

impl Montgomery {
    pub(crate) fn new(modulus: &BigUint) -> Option<Self> {
        if !modulus.is_odd() || modulus.is_one() {
            return None;
        }
        let limbs = modulus.limbs.len();
        // Newton's iteration doubles the correct low bits each step; an odd number is its
        // own inverse modulo 8, so five steps reach 64 bits.
        let low = modulus.limbs[0];
        let mut inverse = low;
        for _ in 0..5 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(low.wrapping_mul(inverse)));
        }
        let one = BigUint::from_u64(1).shift_left(LIMB_BITS * limbs);
        let r_squared = one.shift_left(LIMB_BITS * limbs).rem(modulus)?;
        Some(Self {
            modulus: modulus.limbs.clone(),
            modulus_value: modulus.clone(),
            inverse: inverse.wrapping_neg(),
            r_squared: r_squared.padded(limbs),
            one: one.rem(modulus)?.padded(limbs),
        })
    }

    /// `out = a * b / R mod modulus` (coarsely integrated operand scanning). `accumulator`
    /// is scratch of `modulus.len() + 2` limbs.
    fn multiply(&self, a: &[u64], b: &[u64], out: &mut [u64], accumulator: &mut [u64]) {
        let modulus = &self.modulus;
        let limbs = modulus.len();
        accumulator.fill(0);
        for &a_limb in a {
            let mut carry = 0u128;
            for (word, &b_limb) in accumulator.iter_mut().zip(b) {
                let sum = *word as u128 + a_limb as u128 * b_limb as u128 + carry;
                *word = sum as u64;
                carry = sum >> LIMB_BITS;
            }
            let sum = accumulator[limbs] as u128 + carry;
            accumulator[limbs] = sum as u64;
            accumulator[limbs + 1] = (sum >> LIMB_BITS) as u64;

            let factor = accumulator[0].wrapping_mul(self.inverse);
            let mut carry =
                (accumulator[0] as u128 + factor as u128 * modulus[0] as u128) >> LIMB_BITS;
            for index in 1..limbs {
                let sum =
                    accumulator[index] as u128 + factor as u128 * modulus[index] as u128 + carry;
                accumulator[index - 1] = sum as u64;
                carry = sum >> LIMB_BITS;
            }
            let sum = accumulator[limbs] as u128 + carry;
            accumulator[limbs - 1] = sum as u64;
            accumulator[limbs] = accumulator[limbs + 1] + (sum >> LIMB_BITS) as u64;
        }
        let overflow = accumulator[limbs] != 0;
        out.copy_from_slice(&accumulator[..limbs]);
        if overflow || !less_than(out, modulus) {
            subtract_in_place(out, modulus);
        }
    }

    fn reduced(&self, value: &BigUint) -> Vec<u64> {
        if *value < self.modulus_value {
            value.padded(self.modulus.len())
        } else {
            value
                .rem(&self.modulus_value)
                .unwrap_or_default()
                .padded(self.modulus.len())
        }
    }

    /// `base^exponent mod modulus`.
    pub(crate) fn power(&self, base: &BigUint, exponent: &BigUint) -> BigUint {
        let limbs = self.modulus.len();
        let mut accumulator = vec![0u64; limbs + 2];
        let mut table = vec![self.one.clone()];
        let mut entry = vec![0u64; limbs];
        self.multiply(
            &self.reduced(base),
            &self.r_squared,
            &mut entry,
            &mut accumulator,
        );
        table.push(entry.clone());
        for _ in 2..WINDOW_ENTRIES {
            let mut next = vec![0u64; limbs];
            self.multiply(&entry, &table[1], &mut next, &mut accumulator);
            entry.copy_from_slice(&next);
            table.push(next);
        }

        let mut result = self.one.clone();
        let mut scratch = vec![0u64; limbs];
        // LIMB_BITS is a multiple of WINDOW_BITS, so no window straddles two limbs.
        let windows = exponent.bits().div_ceil(WINDOW_BITS);
        for window in (0..windows).rev() {
            if window + 1 != windows {
                for _ in 0..WINDOW_BITS {
                    self.multiply(&result, &result, &mut scratch, &mut accumulator);
                    core::mem::swap(&mut result, &mut scratch);
                }
            }
            let position = window * WINDOW_BITS;
            let value =
                (exponent.limbs[position / LIMB_BITS] >> (position % LIMB_BITS)) & WINDOW_MASK;
            if value != 0 {
                self.multiply(
                    &result,
                    &table[value as usize],
                    &mut scratch,
                    &mut accumulator,
                );
                core::mem::swap(&mut result, &mut scratch);
            }
        }
        self.leave(&result, &mut accumulator)
    }

    /// `a * b mod modulus`.
    pub(crate) fn multiply_modular(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let limbs = self.modulus.len();
        let mut accumulator = vec![0u64; limbs + 2];
        let mut product = vec![0u64; limbs];
        // (a * b / R) * R^2 / R = a * b.
        self.multiply(
            &self.reduced(a),
            &self.reduced(b),
            &mut product,
            &mut accumulator,
        );
        let mut result = vec![0u64; limbs];
        self.multiply(&product, &self.r_squared, &mut result, &mut accumulator);
        BigUint::from_limbs(result)
    }

    /// Out of Montgomery form.
    fn leave(&self, value: &[u64], accumulator: &mut [u64]) -> BigUint {
        let mut unit = vec![0u64; self.modulus.len()];
        unit[0] = 1;
        let mut out = vec![0u64; self.modulus.len()];
        self.multiply(value, &unit, &mut out, accumulator);
        BigUint::from_limbs(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(hex: &str) -> BigUint {
        let digits = if hex.len() % 2 == 1 {
            format!("0{hex}")
        } else {
            hex.into()
        };
        let bytes: Vec<u8> = (0..digits.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&digits[index..index + 2], 16).unwrap())
            .collect();
        BigUint::from_be_bytes(&bytes)
    }

    #[test]
    fn bytes_round_trip_and_normalize() {
        let value = BigUint::from_be_bytes(&[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(value.bits(), 65);
        assert_eq!(value.to_be_bytes(9).unwrap(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(value.to_be_bytes(8), None);
        assert_eq!(
            value.to_be_bytes(11).unwrap(),
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(value.to_minimal_be_bytes(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(BigUint::from_be_bytes(&[0, 0]).is_zero());
        assert_eq!(BigUint::from_be_bytes(&[0, 0]), BigUint::zero());
    }

    #[test]
    fn arithmetic_carries_across_limbs() {
        let all_ones = number("ffffffffffffffffffffffffffffffff");
        let one = BigUint::from_u64(1);
        let sum = all_ones.add(&one);
        assert_eq!(sum, number("100000000000000000000000000000000"));
        assert_eq!(sum.sub(&one), all_ones);
        assert_eq!(
            all_ones.mul(&all_ones),
            number("fffffffffffffffffffffffffffffffe00000000000000000000000000000001")
        );
        assert_eq!(sum.shift_right(128), one);
        assert_eq!(one.shift_left(128), sum);
        assert_eq!(sum.shift_right(129), BigUint::zero());
        assert_eq!(number("1800").trailing_zeros(), 11);
        assert_eq!(sum.trailing_zeros(), 128);
        assert!(number("10000000000000000") > number("ffffffffffffffff"));
    }

    #[test]
    fn division_matches_multiplication() {
        let divisor = number("f123456789abcdef0123456789abcdef1");
        let quotient = number("abcdef0123456789abcdef0123456789abcdef");
        let remainder = number("123456789");
        let dividend = divisor.mul(&quotient).add(&remainder);
        assert_eq!(dividend.div_rem(&divisor), Some((quotient, remainder)));
        assert_eq!(dividend.div_rem(&BigUint::zero()), None);
        assert_eq!(
            divisor.div_rem(&dividend),
            Some((BigUint::zero(), divisor.clone()))
        );
        assert_eq!(dividend.rem_u32(1_000_003), {
            let (_, remainder) = dividend.div_rem(&BigUint::from_u64(1_000_003)).unwrap();
            remainder.limbs.first().copied().unwrap_or(0) as u32
        });
    }

    #[test]
    fn inverses_and_gcd() {
        // 3 * 4 = 12 = 1 (mod 11); 17 * 2753 = 46801 = 1 (mod 3120), the textbook RSA example.
        assert_eq!(
            mod_inverse(&BigUint::from_u64(3), &BigUint::from_u64(11)),
            Some(BigUint::from_u64(4))
        );
        assert_eq!(
            mod_inverse(&BigUint::from_u64(17), &BigUint::from_u64(3120)),
            Some(BigUint::from_u64(2753))
        );
        assert_eq!(
            mod_inverse(&BigUint::from_u64(6), &BigUint::from_u64(9)),
            None
        );
        assert_eq!(
            mod_inverse(&BigUint::from_u64(6), &BigUint::from_u64(1)),
            None
        );
        let modulus = number("fffffffffffffffffffffffffffffffeffffffffffffffff");
        let value = number("123456789abcdef0fedcba987654321");
        let inverse = mod_inverse(&value, &modulus).unwrap();
        assert!(value.mul(&inverse).rem(&modulus).unwrap().is_one());
        assert_eq!(gcd_u64(65537, 65536), 1);
        assert_eq!(gcd_u64(0, 3), 3);
    }

    #[test]
    fn small_numbers_power_correctly() {
        // 4^13 mod 497 = 445 (the textbook example).
        let montgomery = Montgomery::new(&BigUint::from_u64(497)).unwrap();
        let result = montgomery.power(&BigUint::from_u64(4), &BigUint::from_u64(13));
        assert_eq!(result, BigUint::from_u64(445));
        assert_eq!(
            montgomery.power(&BigUint::from_u64(4), &BigUint::zero()),
            BigUint::from_u64(1)
        );
        // Bases at or above the modulus are reduced first.
        assert_eq!(
            montgomery.power(&BigUint::from_u64(4 + 497), &BigUint::from_u64(13)),
            BigUint::from_u64(445)
        );
        assert_eq!(
            montgomery.multiply_modular(&BigUint::from_u64(400), &BigUint::from_u64(300)),
            BigUint::from_u64(400 * 300 % 497)
        );
        assert!(Montgomery::new(&BigUint::from_u64(1)).is_none());
        assert!(Montgomery::new(&BigUint::from_u64(498)).is_none());
    }

    #[test]
    fn multi_limb_numbers_power_correctly() {
        // (2^64 + 1)^3 mod (2^127 - 1) by hand: 2^192 + 3*2^128 + 3*2^64 + 1, and
        // 2^127 = 1 (mod 2^127 - 1), so 2^192 = 2^65 and 2^128 = 2: 5*2^64 + 7.
        let modulus = number("7fffffffffffffffffffffffffffffff");
        let base = number("10000000000000001");
        let result = Montgomery::new(&modulus)
            .unwrap()
            .power(&base, &BigUint::from_u64(3));
        assert_eq!(result, number("50000000000000007"));
    }

    #[test]
    fn fermat_holds_for_a_large_prime() {
        // 2^255 - 19 is prime, so a^(p-1) = 1 for every a it does not divide.
        let prime = number("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");
        let montgomery = Montgomery::new(&prime).unwrap();
        let exponent = prime.sub(&BigUint::from_u64(1));
        for base in [2u64, 3, 0x1234_5678_9abc_def0] {
            assert!(
                montgomery
                    .power(&BigUint::from_u64(base), &exponent)
                    .is_one()
            );
        }
    }
}
