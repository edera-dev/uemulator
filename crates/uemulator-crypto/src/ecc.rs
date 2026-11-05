//! NIST P-256 (FIPS 186-5, SP 800-186): ECDH and ECDSA for the TPM's ECC keys. Field and
//! scalar arithmetic in Montgomery form on four 64-bit limbs, points in Jacobian
//! coordinates, inversion by Fermat. Nothing is constant time.

use crate::Error;

pub const P256_COORDINATE_BYTES: usize = 32;
/// n, the order of the generator (and of the group: the cofactor is 1), big-endian.
pub const P256_ORDER: [u8; P256_COORDINATE_BYTES] =
    hex32("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551");
/// p = 2^256 - 2^224 + 2^192 + 2^96 - 1.
const P256_PRIME: [u8; P256_COORDINATE_BYTES] =
    hex32("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff");
/// b in y^2 = x^3 - 3x + b.
const P256_B: [u8; P256_COORDINATE_BYTES] =
    hex32("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");
const P256_GENERATOR_X: [u8; P256_COORDINATE_BYTES] =
    hex32("6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296");
const P256_GENERATOR_Y: [u8; P256_COORDINATE_BYTES] =
    hex32("4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5");

const LIMBS: usize = 4;
const LIMB_BITS: usize = 64;
const LIMB_BYTES: usize = 8;
/// Scalar multiplication takes the scalar four bits at a time from a table of 0P..15P.
const WINDOW_BITS: usize = 4;
const WINDOW_ENTRIES: usize = 1 << WINDOW_BITS;
const WINDOW_MASK: u64 = WINDOW_ENTRIES as u64 - 1;

type Limbs = [u64; LIMBS];

/// An affine point, coordinates big-endian, as the TPM's TPMS_ECC_POINT carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EccPoint {
    pub x: [u8; P256_COORDINATE_BYTES],
    pub y: [u8; P256_COORDINATE_BYTES],
}

/// 64 hex digits to bytes, at compile time (a malformed constant fails the build).
const fn hex32(text: &str) -> [u8; P256_COORDINATE_BYTES] {
    const fn digit(character: u8) -> u8 {
        match character {
            b'0'..=b'9' => character - b'0',
            b'a'..=b'f' => character - b'a' + 10,
            _ => panic!("not a lowercase hex digit"),
        }
    }
    let text = text.as_bytes();
    assert!(text.len() == 2 * P256_COORDINATE_BYTES);
    let mut out = [0u8; P256_COORDINATE_BYTES];
    let mut index = 0;
    while index < P256_COORDINATE_BYTES {
        out[index] = digit(text[2 * index]) << 4 | digit(text[2 * index + 1]);
        index += 1;
    }
    out
}

const fn limbs_from_be(bytes: &[u8; P256_COORDINATE_BYTES]) -> Limbs {
    let mut limbs = [0u64; LIMBS];
    let mut index = 0;
    while index < P256_COORDINATE_BYTES {
        let byte = bytes[P256_COORDINATE_BYTES - 1 - index] as u64;
        limbs[index / LIMB_BYTES] |= byte << (8 * (index % LIMB_BYTES));
        index += 1;
    }
    limbs
}

fn limbs_to_be(limbs: &Limbs) -> [u8; P256_COORDINATE_BYTES] {
    let mut bytes = [0u8; P256_COORDINATE_BYTES];
    for (index, limb) in limbs.iter().enumerate() {
        let end = P256_COORDINATE_BYTES - index * LIMB_BYTES;
        bytes[end - LIMB_BYTES..end].copy_from_slice(&limb.to_be_bytes());
    }
    bytes
}

const fn less_than(a: &Limbs, b: &Limbs) -> bool {
    let mut index = LIMBS;
    while index > 0 {
        index -= 1;
        if a[index] != b[index] {
            return a[index] < b[index];
        }
    }
    false
}

const fn is_zero(a: &Limbs) -> bool {
    a[0] | a[1] | a[2] | a[3] == 0
}

const fn add_with_carry(a: &Limbs, b: &Limbs) -> (Limbs, bool) {
    let mut sum = [0u64; LIMBS];
    let mut carry = false;
    let mut index = 0;
    while index < LIMBS {
        let (partial, first) = a[index].overflowing_add(b[index]);
        let (partial, second) = partial.overflowing_add(carry as u64);
        sum[index] = partial;
        carry = first || second;
        index += 1;
    }
    (sum, carry)
}

const fn subtract_with_borrow(a: &Limbs, b: &Limbs) -> (Limbs, bool) {
    let mut difference = [0u64; LIMBS];
    let mut borrow = false;
    let mut index = 0;
    while index < LIMBS {
        let (partial, first) = a[index].overflowing_sub(b[index]);
        let (partial, second) = partial.overflowing_sub(borrow as u64);
        difference[index] = partial;
        borrow = first || second;
        index += 1;
    }
    (difference, borrow)
}

/// Arithmetic modulo an odd 256-bit modulus in Montgomery form, R = 2^256.
struct Modulus {
    value: Limbs,
    /// -value^-1 mod 2^64.
    inverse: u64,
    /// R^2 mod value: multiplying by it enters Montgomery form.
    r_squared: Limbs,
    /// R mod value: one in Montgomery form.
    one: Limbs,
}

impl Modulus {
    const fn new(value: Limbs) -> Self {
        // Newton's iteration doubles the correct low bits each step; an odd number is its
        // own inverse modulo 8, so five steps reach 64 bits.
        let mut inverse = value[0];
        let mut step = 0;
        while step < 5 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(value[0].wrapping_mul(inverse)));
            step += 1;
        }
        // 2^256 and 2^512 modulo value by doubling 1.
        let mut power = [1u64, 0, 0, 0];
        let mut one = power;
        let mut doublings = 0;
        while doublings < 2 * LIMBS * LIMB_BITS {
            let (doubled, carry) = add_with_carry(&power, &power);
            power = if carry || !less_than(&doubled, &value) {
                subtract_with_borrow(&doubled, &value).0
            } else {
                doubled
            };
            doublings += 1;
            if doublings == LIMBS * LIMB_BITS {
                one = power;
            }
        }
        Self {
            value,
            inverse: inverse.wrapping_neg(),
            r_squared: power,
            one,
        }
    }

    /// `a * b / R mod value` (coarsely integrated operand scanning).
    fn multiply(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let modulus = &self.value;
        let mut accumulator = [0u64; LIMBS + 2];
        for &a_limb in a {
            let mut carry = 0u128;
            for (word, &b_limb) in accumulator.iter_mut().zip(b) {
                let sum = *word as u128 + a_limb as u128 * b_limb as u128 + carry;
                *word = sum as u64;
                carry = sum >> LIMB_BITS;
            }
            let sum = accumulator[LIMBS] as u128 + carry;
            accumulator[LIMBS] = sum as u64;
            accumulator[LIMBS + 1] = (sum >> LIMB_BITS) as u64;

            let factor = accumulator[0].wrapping_mul(self.inverse);
            let mut carry =
                (accumulator[0] as u128 + factor as u128 * modulus[0] as u128) >> LIMB_BITS;
            for index in 1..LIMBS {
                let sum =
                    accumulator[index] as u128 + factor as u128 * modulus[index] as u128 + carry;
                accumulator[index - 1] = sum as u64;
                carry = sum >> LIMB_BITS;
            }
            let sum = accumulator[LIMBS] as u128 + carry;
            accumulator[LIMBS - 1] = sum as u64;
            accumulator[LIMBS] = accumulator[LIMBS + 1] + (sum >> LIMB_BITS) as u64;
        }
        let result = [
            accumulator[0],
            accumulator[1],
            accumulator[2],
            accumulator[3],
        ];
        if accumulator[LIMBS] != 0 || !less_than(&result, modulus) {
            subtract_with_borrow(&result, modulus).0
        } else {
            result
        }
    }

    fn square(&self, a: &Limbs) -> Limbs {
        self.multiply(a, a)
    }

    fn add(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let (sum, carry) = add_with_carry(a, b);
        if carry || !less_than(&sum, &self.value) {
            subtract_with_borrow(&sum, &self.value).0
        } else {
            sum
        }
    }

    fn subtract(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let (difference, borrow) = subtract_with_borrow(a, b);
        if borrow {
            add_with_carry(&difference, &self.value).0
        } else {
            difference
        }
    }

    /// Takes a value below the modulus into Montgomery form.
    fn enter(&self, a: &Limbs) -> Limbs {
        self.multiply(a, &self.r_squared)
    }

    fn leave(&self, a: &Limbs) -> Limbs {
        self.multiply(a, &[1, 0, 0, 0])
    }

    /// a^-1 as a^(value - 2) (Fermat; both moduli here are prime), in Montgomery form.
    fn invert(&self, a: &Limbs) -> Limbs {
        let exponent = subtract_with_borrow(&self.value, &[2, 0, 0, 0]).0;
        let mut result = self.one;
        for bit in (0..LIMBS * LIMB_BITS).rev() {
            result = self.square(&result);
            if exponent[bit / LIMB_BITS] >> (bit % LIMB_BITS) & 1 == 1 {
                result = self.multiply(&result, a);
            }
        }
        result
    }
}

const FIELD: Modulus = Modulus::new(limbs_from_be(&P256_PRIME));
const SCALARS: Modulus = Modulus::new(limbs_from_be(&P256_ORDER));
const ORDER: Limbs = limbs_from_be(&P256_ORDER);

/// A point in Jacobian coordinates (x = X/Z^2, y = Y/Z^3), each in Montgomery form; Z = 0
/// is the point at infinity (zero is zero in Montgomery form too).
#[derive(Clone, Copy)]
struct Jacobian {
    x: Limbs,
    y: Limbs,
    z: Limbs,
}

const INFINITY: Jacobian = Jacobian {
    x: [0; LIMBS],
    y: [0; LIMBS],
    z: [0; LIMBS],
};

impl Jacobian {
    /// The point if its coordinates are below p and it is on the curve.
    fn from_affine(point: &EccPoint) -> Option<Self> {
        let prime = limbs_from_be(&P256_PRIME);
        let x = limbs_from_be(&point.x);
        let y = limbs_from_be(&point.y);
        if !less_than(&x, &prime) || !less_than(&y, &prime) {
            return None;
        }
        let x = FIELD.enter(&x);
        let y = FIELD.enter(&y);
        // y^2 = x^3 - 3x + b
        let three_x = FIELD.add(&FIELD.add(&x, &x), &x);
        let cube = FIELD.multiply(&FIELD.square(&x), &x);
        let right = FIELD.add(
            &FIELD.subtract(&cube, &three_x),
            &FIELD.enter(&limbs_from_be(&P256_B)),
        );
        (FIELD.square(&y) == right).then_some(Self { x, y, z: FIELD.one })
    }

    fn generator() -> Self {
        Self {
            x: FIELD.enter(&limbs_from_be(&P256_GENERATOR_X)),
            y: FIELD.enter(&limbs_from_be(&P256_GENERATOR_Y)),
            z: FIELD.one,
        }
    }

    fn is_infinity(&self) -> bool {
        is_zero(&self.z)
    }

    fn to_affine(self) -> Option<EccPoint> {
        if self.is_infinity() {
            return None;
        }
        let z_inverse = FIELD.invert(&self.z);
        let z_inverse_squared = FIELD.square(&z_inverse);
        let x = FIELD.multiply(&self.x, &z_inverse_squared);
        let y = FIELD.multiply(&self.y, &FIELD.multiply(&z_inverse_squared, &z_inverse));
        Some(EccPoint {
            x: limbs_to_be(&FIELD.leave(&x)),
            y: limbs_to_be(&FIELD.leave(&y)),
        })
    }

    /// dbl-2001-b from the Explicit-Formulas Database, which uses a = -3. A point with y = 0
    /// doubles to Z = 2YZ = 0, infinity, without a special case.
    fn double(&self) -> Self {
        let delta = FIELD.square(&self.z);
        let gamma = FIELD.square(&self.y);
        let beta = FIELD.multiply(&self.x, &gamma);
        let product = FIELD.multiply(
            &FIELD.subtract(&self.x, &delta),
            &FIELD.add(&self.x, &delta),
        );
        let alpha = FIELD.add(&FIELD.add(&product, &product), &product);
        let four_beta = FIELD.add(&FIELD.add(&beta, &beta), &FIELD.add(&beta, &beta));
        let x = FIELD.subtract(&FIELD.square(&alpha), &FIELD.add(&four_beta, &four_beta));
        let z = FIELD.subtract(
            &FIELD.subtract(&FIELD.square(&FIELD.add(&self.y, &self.z)), &gamma),
            &delta,
        );
        let gamma_squared = FIELD.square(&gamma);
        let four_gamma_squared = FIELD.add(
            &FIELD.add(&gamma_squared, &gamma_squared),
            &FIELD.add(&gamma_squared, &gamma_squared),
        );
        let y = FIELD.subtract(
            &FIELD.multiply(&alpha, &FIELD.subtract(&four_beta, &x)),
            &FIELD.add(&four_gamma_squared, &four_gamma_squared),
        );
        Self { x, y, z }
    }

    /// add-2007-bl, which fails for equal or opposite inputs (H = 0), so those are handled
    /// first: doubling, or infinity.
    fn add(&self, other: &Self) -> Self {
        if self.is_infinity() {
            return *other;
        }
        if other.is_infinity() {
            return *self;
        }
        let z1_squared = FIELD.square(&self.z);
        let z2_squared = FIELD.square(&other.z);
        let u1 = FIELD.multiply(&self.x, &z2_squared);
        let u2 = FIELD.multiply(&other.x, &z1_squared);
        let s1 = FIELD.multiply(&FIELD.multiply(&self.y, &other.z), &z2_squared);
        let s2 = FIELD.multiply(&FIELD.multiply(&other.y, &self.z), &z1_squared);
        let h = FIELD.subtract(&u2, &u1);
        let half_r = FIELD.subtract(&s2, &s1);
        if is_zero(&h) {
            return if is_zero(&half_r) {
                self.double()
            } else {
                INFINITY
            };
        }
        let double_h = FIELD.add(&h, &h);
        let i = FIELD.square(&double_h);
        let j = FIELD.multiply(&h, &i);
        let r = FIELD.add(&half_r, &half_r);
        let v = FIELD.multiply(&u1, &i);
        let x = FIELD.subtract(&FIELD.subtract(&FIELD.square(&r), &j), &FIELD.add(&v, &v));
        let s1_j = FIELD.multiply(&s1, &j);
        let y = FIELD.subtract(
            &FIELD.multiply(&r, &FIELD.subtract(&v, &x)),
            &FIELD.add(&s1_j, &s1_j),
        );
        let z_sum_squared = FIELD.square(&FIELD.add(&self.z, &other.z));
        let z = FIELD.multiply(
            &FIELD.subtract(&FIELD.subtract(&z_sum_squared, &z1_squared), &z2_squared),
            &h,
        );
        Self { x, y, z }
    }

    fn multiply(&self, scalar: &Limbs) -> Self {
        let mut table = [INFINITY; WINDOW_ENTRIES];
        for index in 1..WINDOW_ENTRIES {
            table[index] = table[index - 1].add(self);
        }
        let mut result = INFINITY;
        for window in (0..LIMBS * LIMB_BITS / WINDOW_BITS).rev() {
            for _ in 0..WINDOW_BITS {
                result = result.double();
            }
            let position = window * WINDOW_BITS;
            let value = scalar[position / LIMB_BITS] >> (position % LIMB_BITS) & WINDOW_MASK;
            if value != 0 {
                result = result.add(&table[value as usize]);
            }
        }
        result
    }
}

/// 0 < scalar < n.
fn valid_scalar(scalar: &Limbs) -> bool {
    !is_zero(scalar) && less_than(scalar, &ORDER)
}

/// A field element (below p < 2n) reduced modulo n.
fn reduce_to_scalar(value: &Limbs) -> Limbs {
    if less_than(value, &ORDER) {
        *value
    } else {
        subtract_with_borrow(value, &ORDER).0
    }
}

/// FIPS 186-5 6.4.1 step 3: the leftmost 256 bits of the digest (all of a shorter one) as
/// an integer, then reduced modulo n.
fn digest_scalar(digest: &[u8]) -> Limbs {
    let used = digest.len().min(P256_COORDINATE_BYTES);
    let mut bytes = [0u8; P256_COORDINATE_BYTES];
    bytes[P256_COORDINATE_BYTES - used..].copy_from_slice(&digest[..used]);
    reduce_to_scalar(&limbs_from_be(&bytes))
}

pub fn p256_is_valid_private(private: &[u8; P256_COORDINATE_BYTES]) -> bool {
    valid_scalar(&limbs_from_be(private))
}

/// Whether both coordinates are below p and the point satisfies the curve equation.
pub fn p256_is_on_curve(point: &EccPoint) -> bool {
    Jacobian::from_affine(point).is_some()
}

/// `private * G`. A private key of 0 or at least n is `Unsupported`.
pub fn p256_public_key(private: &[u8; P256_COORDINATE_BYTES]) -> Result<EccPoint, Error> {
    let scalar = limbs_from_be(private);
    if !valid_scalar(&scalar) {
        return Err(Error::Unsupported);
    }
    Jacobian::generator()
        .multiply(&scalar)
        .to_affine()
        .ok_or(Error::Unsupported)
}

/// `scalar * point` for any 256-bit scalar. `Malformed` for a point off the curve or a
/// result at infinity.
pub fn p256_scalar_multiply(
    scalar: &[u8; P256_COORDINATE_BYTES],
    point: &EccPoint,
) -> Result<EccPoint, Error> {
    Jacobian::from_affine(point)
        .ok_or(Error::Malformed)?
        .multiply(&limbs_from_be(scalar))
        .to_affine()
        .ok_or(Error::Malformed)
}

/// The ECDH shared secret: the x coordinate of `private * peer`. `Unsupported` for a private
/// key of 0 or at least n, `Malformed` for a peer point off the curve.
pub fn p256_ecdh(
    private: &[u8; P256_COORDINATE_BYTES],
    peer: &EccPoint,
) -> Result<[u8; P256_COORDINATE_BYTES], Error> {
    if !p256_is_valid_private(private) {
        return Err(Error::Unsupported);
    }
    Ok(p256_scalar_multiply(private, peer)?.x)
}

/// An ECDSA signature (r, s) over `digest` with the caller's per-signature `nonce` k (FIPS
/// 186-5 6.4.1; the TPM makes k from its DRBG). `Unsupported` for a private key or nonce
/// of 0 or at least n, and for the vanishingly rare k that gives r = 0 or s = 0 (draw another).
pub fn p256_ecdsa_sign(
    private: &[u8; P256_COORDINATE_BYTES],
    digest: &[u8],
    nonce: &[u8; P256_COORDINATE_BYTES],
) -> Result<([u8; P256_COORDINATE_BYTES], [u8; P256_COORDINATE_BYTES]), Error> {
    let private = limbs_from_be(private);
    let nonce = limbs_from_be(nonce);
    if !valid_scalar(&private) || !valid_scalar(&nonce) {
        return Err(Error::Unsupported);
    }
    let commitment = Jacobian::generator()
        .multiply(&nonce)
        .to_affine()
        .ok_or(Error::Unsupported)?;
    let r = reduce_to_scalar(&limbs_from_be(&commitment.x));
    if is_zero(&r) {
        return Err(Error::Unsupported);
    }
    // s = k^-1 (e + r d) mod n
    let sum = SCALARS.add(
        &SCALARS.enter(&digest_scalar(digest)),
        &SCALARS.multiply(&SCALARS.enter(&r), &SCALARS.enter(&private)),
    );
    let s = SCALARS.leave(&SCALARS.multiply(&SCALARS.invert(&SCALARS.enter(&nonce)), &sum));
    if is_zero(&s) {
        return Err(Error::Unsupported);
    }
    Ok((limbs_to_be(&r), limbs_to_be(&s)))
}

/// FIPS 186-5 6.4.2. False for anything malformed: r or s outside [1, n - 1], or a public
/// key off the curve.
pub fn p256_ecdsa_verify(
    public: &EccPoint,
    digest: &[u8],
    r: &[u8; P256_COORDINATE_BYTES],
    s: &[u8; P256_COORDINATE_BYTES],
) -> bool {
    let r = limbs_from_be(r);
    let s = limbs_from_be(s);
    let Some(public) = Jacobian::from_affine(public) else {
        return false;
    };
    if !valid_scalar(&r) || !valid_scalar(&s) {
        return false;
    }
    let s_inverse = SCALARS.invert(&SCALARS.enter(&s));
    let u1 = SCALARS.leave(&SCALARS.multiply(&SCALARS.enter(&digest_scalar(digest)), &s_inverse));
    let u2 = SCALARS.leave(&SCALARS.multiply(&SCALARS.enter(&r), &s_inverse));
    let sum = Jacobian::generator()
        .multiply(&u1)
        .add(&public.multiply(&u2));
    sum.to_affine()
        .is_some_and(|point| reduce_to_scalar(&limbs_from_be(&point.x)) == r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha;

    /// `hack/crypto/make-tpm-fixtures.py`: r || s from Python `cryptography` signing
    /// "uemulator" with SHA-256 and `SIGNER_PRIVATE`.
    const PYTHON_SIGNATURE: &[u8] = include_bytes!("testdata/p256-ecdsa-sha256.sig");
    const SIGNER_PRIVATE: [u8; 32] =
        hex32("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721");
    const SIGNER_PUBLIC: EccPoint = EccPoint {
        x: hex32("60fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb6"),
        y: hex32("7903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e9f5177a3c294d4462299"),
    };

    fn scalar(value: u64) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[24..].copy_from_slice(&value.to_be_bytes());
        bytes
    }

    fn generator() -> EccPoint {
        EccPoint {
            x: P256_GENERATOR_X,
            y: P256_GENERATOR_Y,
        }
    }

    fn order_minus(value: u64) -> [u8; 32] {
        let difference = subtract_with_borrow(&ORDER, &[value, 0, 0, 0]).0;
        limbs_to_be(&difference)
    }

    #[test]
    fn generator_multiples_match_python() {
        assert_eq!(p256_public_key(&scalar(1)), Ok(generator()));
        assert_eq!(
            p256_public_key(&scalar(2)),
            Ok(EccPoint {
                x: hex32("7cf27b188d034f7e8a52380304b51ac3c08969e277f21b35a60b48fc47669978"),
                y: hex32("07775510db8ed040293d9ac69f7430dbba7dade63ce982299e04b79d227873d1"),
            })
        );
        assert_eq!(
            p256_public_key(&order_minus(1)),
            Ok(EccPoint {
                x: P256_GENERATOR_X,
                y: hex32("b01cbd1c01e58065711814b583f061e9d431cca994cea1313449bf97c840ae0a"),
            })
        );
        assert_eq!(
            p256_scalar_multiply(&P256_ORDER, &generator()),
            Err(Error::Malformed),
            "n * G is infinity"
        );
        assert_eq!(
            p256_scalar_multiply(&scalar(2), &generator()),
            p256_public_key(&scalar(2))
        );
        assert!(p256_is_on_curve(&generator()));
    }

    #[test]
    fn nist_cavs_ecdh_vector() {
        // ECC CDH Primitive (SP 800-56A section 5.7.1.2), P-256, COUNT = 0.
        let private = hex32("7d7dc5f71eb29ddaf80d6214632eeae03d9058af1fb6d22ed80badb62bc1a534");
        let peer = EccPoint {
            x: hex32("700c48f77f56584c5cc632ca65640db91b6bacce3a4df6b42ce7cc838833d287"),
            y: hex32("db71e509e3fd9b060ddb20ba5c51dcc5948d46fbf640dfe0441782cab85fa4ac"),
        };
        assert_eq!(
            p256_public_key(&private),
            Ok(EccPoint {
                x: hex32("ead218590119e8876b29146ff89ca61770c4edbbf97d38ce385ed281d8a6b230"),
                y: hex32("28af61281fd35e2fa7002523acc85a429cb06ee6648325389f59edfce1405141"),
            })
        );
        assert_eq!(
            p256_ecdh(&private, &peer),
            Ok(hex32(
                "46fc62106420ff012e54a434fbdd2d25ccc5852060561e68040dd7778997bd7b"
            ))
        );
        // Both sides agree with an ephemeral key, as a salted session's ECDH does.
        let ephemeral = scalar(0x1234_5678_9abc_def1);
        let ephemeral_public = p256_public_key(&ephemeral).unwrap();
        let ours = p256_ecdh(&ephemeral, &p256_public_key(&private).unwrap()).unwrap();
        assert_eq!(p256_ecdh(&private, &ephemeral_public), Ok(ours));
    }

    #[test]
    fn bad_points_and_scalars_are_refused() {
        let mut off_curve = generator();
        off_curve.y[31] ^= 1;
        assert!(!p256_is_on_curve(&off_curve));
        assert_eq!(p256_ecdh(&scalar(5), &off_curve), Err(Error::Malformed));
        assert_eq!(
            p256_scalar_multiply(&scalar(5), &off_curve),
            Err(Error::Malformed)
        );
        let mut too_large = generator();
        too_large.x = P256_PRIME;
        assert!(!p256_is_on_curve(&too_large));
        assert_eq!(p256_ecdh(&scalar(5), &too_large), Err(Error::Malformed));
        assert!(!p256_is_on_curve(&EccPoint {
            x: [0; 32],
            y: [0; 32]
        }));

        for private in [[0u8; 32], P256_ORDER, [0xff; 32]] {
            assert!(!p256_is_valid_private(&private));
            assert_eq!(p256_public_key(&private), Err(Error::Unsupported));
            assert_eq!(p256_ecdh(&private, &generator()), Err(Error::Unsupported));
        }
        assert!(p256_is_valid_private(&order_minus(1)));
        assert!(p256_is_valid_private(&scalar(1)));
    }

    #[test]
    fn ecdsa_signs_and_verifies() {
        let digest = sha::sha256(b"uemulator");
        let (r, s) = p256_ecdsa_sign(&SIGNER_PRIVATE, &digest, &scalar(0xdead_beef)).unwrap();
        assert!(p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &r, &s));
        let other = sha::sha256(b"uemulatoR");
        assert!(!p256_ecdsa_verify(&SIGNER_PUBLIC, &other, &r, &s));
        assert!(!p256_ecdsa_verify(&generator(), &digest, &r, &s));
        assert!(!p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &r, &P256_ORDER));
        assert!(!p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &[0; 32], &s));
        // A SHA-384 digest is cut to its leftmost 256 bits.
        let long = sha::sha384(b"uemulator");
        let (r, s) = p256_ecdsa_sign(&SIGNER_PRIVATE, &long, &scalar(7)).unwrap();
        assert!(p256_ecdsa_verify(&SIGNER_PUBLIC, &long, &r, &s));
        assert!(p256_ecdsa_verify(&SIGNER_PUBLIC, &long[..32], &r, &s));
        assert_eq!(
            p256_ecdsa_sign(&SIGNER_PRIVATE, &digest, &[0; 32]),
            Err(Error::Unsupported)
        );
        assert_eq!(
            p256_ecdsa_sign(&P256_ORDER, &digest, &scalar(7)),
            Err(Error::Unsupported)
        );
    }

    #[test]
    fn a_python_ecdsa_signature_verifies() {
        assert_eq!(p256_public_key(&SIGNER_PRIVATE), Ok(SIGNER_PUBLIC));
        let digest = sha::sha256(b"uemulator");
        let r: [u8; 32] = PYTHON_SIGNATURE[..32].try_into().unwrap();
        let s: [u8; 32] = PYTHON_SIGNATURE[32..].try_into().unwrap();
        assert!(p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &r, &s));
        // (r, n - s) is the other valid signature of the same digest.
        let negated = limbs_to_be(&subtract_with_borrow(&ORDER, &limbs_from_be(&s)).0);
        assert!(p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &r, &negated));
        let mut flipped = s;
        flipped[10] ^= 4;
        assert!(!p256_ecdsa_verify(&SIGNER_PUBLIC, &digest, &r, &flipped));
    }

    #[test]
    fn scalar_multiplication_is_fast() {
        let point = p256_public_key(&SIGNER_PRIVATE).unwrap();
        let repeats = 20;
        let started = std::time::Instant::now();
        for index in 0..repeats {
            p256_scalar_multiply(&order_minus(index + 2), &point).unwrap();
        }
        std::println!(
            "P-256 scalar multiplication took {:?}",
            started.elapsed() / repeats as u32
        );
    }
}
