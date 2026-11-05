//! RSA public keys and RSASSA-PKCS1-v1_5 signature verification (RFC 8017 8.2.2).

use alloc::vec;
use alloc::vec::Vec;

use crate::Error;
use crate::der::{self, Reader, TAG_NULL, TAG_SEQUENCE, oid};
use crate::sha::HashAlgorithm;

/// Moduli above this are refused, which bounds the work a hostile certificate can ask for.
pub const MAX_MODULUS_BITS: usize = 8192;
/// RFC 8017 9.2: the 0xFF padding is at least eight bytes.
const MIN_PADDING_BYTES: usize = 8;
const LIMB_BITS: usize = 64;
const LIMB_BYTES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaPublicKey {
    /// Big-endian, no leading zeros.
    pub modulus: Vec<u8>,
    /// Big-endian, no leading zeros.
    pub exponent: Vec<u8>,
}

impl RsaPublicKey {
    pub fn new(modulus: &[u8], exponent: &[u8]) -> Result<Self, Error> {
        let modulus = strip_leading_zeros(modulus);
        let exponent = strip_leading_zeros(exponent);
        // An even modulus is not RSA, and Montgomery reduction needs it odd.
        if modulus.is_empty()
            || modulus.len() * 8 > MAX_MODULUS_BITS
            || modulus[modulus.len() - 1] & 1 == 0
            || exponent.is_empty()
            || exponent.len() > modulus.len()
        {
            return Err(Error::Unsupported);
        }
        Ok(Self {
            modulus: modulus.to_vec(),
            exponent: exponent.to_vec(),
        })
    }

    /// From a SubjectPublicKeyInfo whose algorithm is rsaEncryption (RFC 3279 2.3.1).
    pub fn from_subject_public_key_info(spki: &[u8]) -> Result<Self, Error> {
        let mut outer = Reader::new(spki);
        let mut fields = outer.expect(TAG_SEQUENCE)?.reader();
        let (algorithm, _) = fields.algorithm()?;
        if algorithm != oid::RSA_ENCRYPTION {
            return Err(Error::Unsupported);
        }
        Self::from_pkcs1(fields.bit_string()?)
    }

    /// From an RSAPublicKey (RFC 8017 A.1.1): SEQUENCE { modulus, publicExponent }.
    pub fn from_pkcs1(der: &[u8]) -> Result<Self, Error> {
        let mut fields = Reader::new(der).expect(TAG_SEQUENCE)?.reader();
        let modulus = fields.unsigned_integer()?;
        let exponent = fields.unsigned_integer()?;
        if !fields.is_empty() {
            return Err(Error::Malformed);
        }
        Self::new(modulus, exponent)
    }

    /// The modulus length in bytes, which is also every signature's length.
    pub fn size(&self) -> usize {
        self.modulus.len()
    }

    /// `signature^e mod n`, left-padded to the modulus length.
    pub fn public_operation(&self, signature: &[u8]) -> Result<Vec<u8>, Error> {
        let signature = strip_leading_zeros(signature);
        if signature.len() > self.size() {
            return Err(Error::BadSignature);
        }
        let modulus = Limbs::from_be_bytes(&self.modulus, limb_count(self.size()));
        let base = Limbs::from_be_bytes(signature, modulus.0.len());
        if !base.less_than(&modulus) {
            return Err(Error::BadSignature);
        }
        let result = Montgomery::new(&modulus).power(&base, &self.exponent);
        Ok(result.to_be_bytes(self.size()))
    }

    /// Checks an RSASSA-PKCS1-v1_5 signature over `digest`, a hash made with `algorithm`.
    pub fn verify_pkcs1v15(
        &self,
        algorithm: HashAlgorithm,
        digest: &[u8],
        signature: &[u8],
    ) -> Result<(), Error> {
        if digest.len() != algorithm.digest_len() {
            return Err(Error::BadSignature);
        }
        let encoded = self.public_operation(signature)?;
        // RFC 8017 9.2 note 1 lets a verifier also accept DigestInfo without the NULL
        // parameters, which some old signers emitted.
        for with_null in [true, false] {
            let digest_info = digest_info(algorithm, digest, with_null);
            if self.size() < digest_info.len() + MIN_PADDING_BYTES + 3 {
                return Err(Error::BadSignature);
            }
            let padding = self.size() - digest_info.len() - 3;
            let mut expected = Vec::with_capacity(self.size());
            expected.extend([0x00, 0x01]);
            expected.resize(2 + padding, 0xff);
            expected.push(0x00);
            expected.extend_from_slice(&digest_info);
            if expected == encoded {
                return Ok(());
            }
        }
        Err(Error::BadSignature)
    }
}

/// DigestInfo ::= SEQUENCE { AlgorithmIdentifier, OCTET STRING digest }.
fn digest_info(algorithm: HashAlgorithm, digest: &[u8], with_null: bool) -> Vec<u8> {
    let oid = algorithm.oid();
    let mut identifier = vec![der::TAG_OID, oid.len() as u8];
    identifier.extend_from_slice(oid);
    if with_null {
        identifier.extend([TAG_NULL, 0]);
    }
    let mut body = vec![TAG_SEQUENCE];
    der::encode_length(identifier.len(), &mut body);
    body.extend(identifier);
    body.push(der::TAG_OCTET_STRING);
    der::encode_length(digest.len(), &mut body);
    body.extend_from_slice(digest);
    let mut out = vec![TAG_SEQUENCE];
    der::encode_length(body.len(), &mut out);
    out.extend(body);
    out
}

fn strip_leading_zeros(bytes: &[u8]) -> &[u8] {
    let leading = bytes.iter().take_while(|&&byte| byte == 0).count();
    &bytes[leading..]
}

fn limb_count(bytes: usize) -> usize {
    bytes.div_ceil(LIMB_BYTES)
}

/// A non-negative integer as little-endian 64-bit limbs, all the same length within one
/// computation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Limbs(Vec<u64>);

impl Limbs {
    fn from_be_bytes(bytes: &[u8], limbs: usize) -> Self {
        let mut out = vec![0u64; limbs];
        for (index, byte) in bytes.iter().rev().enumerate() {
            out[index / LIMB_BYTES] |= (*byte as u64) << (8 * (index % LIMB_BYTES));
        }
        Self(out)
    }

    fn to_be_bytes(&self, length: usize) -> Vec<u8> {
        (0..length)
            .rev()
            .map(|index| {
                self.0
                    .get(index / LIMB_BYTES)
                    .map_or(0, |limb| (limb >> (8 * (index % LIMB_BYTES))) as u8)
            })
            .collect()
    }

    fn less_than(&self, other: &Self) -> bool {
        for (mine, theirs) in self.0.iter().rev().zip(other.0.iter().rev()) {
            if mine != theirs {
                return mine < theirs;
            }
        }
        false
    }

    /// `self -= other`, returning the borrow out of the top limb.
    fn subtract(&mut self, other: &[u64]) -> bool {
        let mut borrow = false;
        for (mine, theirs) in self.0.iter_mut().zip(other) {
            let (difference, first) = mine.overflowing_sub(*theirs);
            let (difference, second) = difference.overflowing_sub(borrow as u64);
            *mine = difference;
            borrow = first || second;
        }
        borrow
    }
}

/// Montgomery arithmetic modulo an odd `modulus`, with R = 2^(64 * limbs).
struct Montgomery<'a> {
    modulus: &'a Limbs,
    /// -modulus^-1 mod 2^64.
    inverse: u64,
    /// R^2 mod modulus, which takes a number into Montgomery form.
    r_squared: Limbs,
}

impl<'a> Montgomery<'a> {
    fn new(modulus: &'a Limbs) -> Self {
        // Newton's iteration doubles the correct low bits each step; an odd number is its
        // own inverse modulo 8, so five steps reach 64 bits.
        let low = modulus.0[0];
        let mut inverse = low;
        for _ in 0..5 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(low.wrapping_mul(inverse)));
        }
        let limbs = modulus.0.len();
        // 2^(2 * 64 * limbs) mod modulus by doubling: slow but simple, and done once per
        // signature.
        let mut r_squared = Limbs(vec![0; limbs]);
        r_squared.0[0] = 1;
        for _ in 0..2 * LIMB_BITS * limbs {
            let carry = r_squared.0[limbs - 1] >> 63;
            for index in (1..limbs).rev() {
                r_squared.0[index] = (r_squared.0[index] << 1) | (r_squared.0[index - 1] >> 63);
            }
            r_squared.0[0] <<= 1;
            if carry == 1 || !r_squared.less_than(modulus) {
                r_squared.subtract(&modulus.0);
            }
        }
        Self {
            modulus,
            inverse: inverse.wrapping_neg(),
            r_squared,
        }
    }

    /// `a * b / R mod modulus` (coarsely integrated operand scanning).
    fn multiply(&self, a: &Limbs, b: &Limbs) -> Limbs {
        let modulus = &self.modulus.0;
        let limbs = modulus.len();
        let mut accumulator = vec![0u64; limbs + 2];
        for &a_limb in &a.0 {
            let mut carry = 0u128;
            for (word, &b_limb) in accumulator.iter_mut().zip(&b.0) {
                let sum = *word as u128 + a_limb as u128 * b_limb as u128 + carry;
                *word = sum as u64;
                carry = sum >> 64;
            }
            let sum = accumulator[limbs] as u128 + carry;
            accumulator[limbs] = sum as u64;
            accumulator[limbs + 1] = (sum >> 64) as u64;

            let factor = accumulator[0].wrapping_mul(self.inverse);
            let mut carry = (accumulator[0] as u128 + factor as u128 * modulus[0] as u128) >> 64;
            for index in 1..limbs {
                let sum =
                    accumulator[index] as u128 + factor as u128 * modulus[index] as u128 + carry;
                accumulator[index - 1] = sum as u64;
                carry = sum >> 64;
            }
            let sum = accumulator[limbs] as u128 + carry;
            accumulator[limbs - 1] = sum as u64;
            accumulator[limbs] = accumulator[limbs + 1] + (sum >> 64) as u64;
        }
        let overflow = accumulator[limbs] != 0;
        accumulator.truncate(limbs);
        let mut result = Limbs(accumulator);
        if overflow || !result.less_than(self.modulus) {
            result.subtract(modulus);
        }
        result
    }

    fn power(&self, base: &Limbs, exponent: &[u8]) -> Limbs {
        let base = self.multiply(base, &self.r_squared);
        let mut one = Limbs(vec![0; self.modulus.0.len()]);
        one.0[0] = 1;
        let mut result = self.multiply(&one, &self.r_squared);
        for byte in exponent {
            for bit in (0..8).rev() {
                result = self.multiply(&result, &result);
                if byte >> bit & 1 == 1 {
                    result = self.multiply(&result, &base);
                }
            }
        }
        self.multiply(&result, &one)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha;

    /// A 1024-bit test key and a signature over "uemulator" made by
    /// `openssl dgst -sha256 -sign` (`hack/secure-boot/make-crypto-fixtures.sh`).
    const TEST_PUBLIC_KEY: &[u8] = include_bytes!("testdata/rsa1024-public.der");
    const TEST_SIGNATURE: &[u8] = include_bytes!("testdata/rsa1024-sha256.sig");

    #[test]
    fn small_numbers_power_correctly() {
        // 4^13 mod 497 = 445 (the textbook example).
        let modulus = Limbs::from_be_bytes(&[0x01, 0xf1], 1);
        let base = Limbs::from_be_bytes(&[4], 1);
        let result = Montgomery::new(&modulus).power(&base, &[13]);
        assert_eq!(result.to_be_bytes(2), [0x01, 0xbd]);
    }

    #[test]
    fn multi_limb_numbers_power_correctly() {
        // (2^64 + 1)^3 mod (2^127 - 1) computed by hand: 2^192 + 3*2^128 + 3*2^64 + 1,
        // and 2^127 = 1 (mod 2^127 - 1), so 2^192 = 2^65, 2^128 = 2.
        let modulus_bytes: Vec<u8> = core::iter::once(0x7f)
            .chain(core::iter::repeat_n(0xff, 15))
            .collect();
        let modulus = Limbs::from_be_bytes(&modulus_bytes, 2);
        let mut base_bytes = vec![0u8; 16];
        base_bytes[7] = 1;
        base_bytes[15] = 1;
        let base = Limbs::from_be_bytes(&base_bytes, 2);
        let result = Montgomery::new(&modulus).power(&base, &[3]);
        // 2^65 + 6 + 3*2^64 + 1 = 5*2^64 + 7
        let mut expected = vec![0u8; 16];
        expected[7] = 5;
        expected[15] = 7;
        assert_eq!(result.to_be_bytes(16), expected);
    }

    #[test]
    fn an_openssl_signature_verifies() {
        let key = RsaPublicKey::from_subject_public_key_info(TEST_PUBLIC_KEY).unwrap();
        assert_eq!(key.size(), 128);
        let digest = sha::sha256(b"uemulator");
        key.verify_pkcs1v15(HashAlgorithm::Sha256, &digest, TEST_SIGNATURE)
            .unwrap();
        let other = sha::sha256(b"uemulatoR");
        assert_eq!(
            key.verify_pkcs1v15(HashAlgorithm::Sha256, &other, TEST_SIGNATURE),
            Err(Error::BadSignature)
        );
        let mut flipped = TEST_SIGNATURE.to_vec();
        flipped[40] ^= 1;
        assert_eq!(
            key.verify_pkcs1v15(HashAlgorithm::Sha256, &digest, &flipped),
            Err(Error::BadSignature)
        );
        assert_eq!(
            key.verify_pkcs1v15(HashAlgorithm::Sha384, &digest, TEST_SIGNATURE),
            Err(Error::BadSignature),
            "wrong digest length"
        );
    }

    #[test]
    fn signatures_at_or_above_the_modulus_are_refused() {
        let key = RsaPublicKey::from_subject_public_key_info(TEST_PUBLIC_KEY).unwrap();
        let digest = sha::sha256(b"uemulator");
        assert_eq!(
            key.verify_pkcs1v15(HashAlgorithm::Sha256, &digest, &key.modulus.clone()),
            Err(Error::BadSignature)
        );
        assert_eq!(
            key.verify_pkcs1v15(HashAlgorithm::Sha256, &digest, &[0xff; 129]),
            Err(Error::BadSignature)
        );
    }

    #[test]
    fn keys_must_be_odd_and_bounded() {
        assert_eq!(RsaPublicKey::new(&[0x10], &[3]), Err(Error::Unsupported));
        assert_eq!(RsaPublicKey::new(&[], &[3]), Err(Error::Unsupported));
        assert_eq!(
            RsaPublicKey::new(&vec![0xff; MAX_MODULUS_BITS / 8 + 1], &[3]),
            Err(Error::Unsupported)
        );
        assert!(RsaPublicKey::new(&[0, 0, 0x11], &[0, 3]).is_ok());
    }
}
