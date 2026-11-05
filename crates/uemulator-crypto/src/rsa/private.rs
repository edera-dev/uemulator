//! RSA private keys: generation (FIPS 186-5 A.1.3 style random probable primes), rebuilding
//! from one prime as a TPM's sensitive area stores it, and the CRT private operation behind
//! OAEP decryption and PKCS#1 v1.5 signing.

use alloc::vec;
use alloc::vec::Vec;

use super::{MIN_PADDING_BYTES, RsaPublicKey, digest_info, oaep, strip_leading_zeros};
use crate::Error;
use crate::bignum::{self, BigUint, Montgomery};
use crate::sha::HashAlgorithm;

/// What `generate` is given 0 for.
const DEFAULT_PUBLIC_EXPONENT: u32 = 65537;
const MIN_GENERATED_MODULUS_BITS: usize = 1024;
/// Generation is the one thing here whose cost a guest picks; 4096 is the largest key a TPM
/// offers.
const MAX_GENERATED_MODULUS_BITS: usize = 4096;
/// Moduli are a whole number of 16 bits so each prime is a whole number of bytes, which is
/// what `prime()` and the TPM's sensitive area (half the modulus size) assume.
const GENERATED_MODULUS_GRANULE_BITS: usize = 16;
/// Setting a prime's top two bits makes the product of two of them exactly twice as long:
/// (0.75 * 2^k)^2 > 2^(2k - 1).
const TOP_TWO_BITS: u8 = 0xc0;
/// Odd numbers sieved after each random start. A window this long near 2^1024 holds about
/// six primes, so a fresh start is rarely needed.
const SIEVE_WINDOW: usize = 2048;
/// Trial-division bound for the sieve: leaves about 12% of odd candidates for Miller-Rabin.
const SMALL_PRIME_LIMIT: u32 = 1 << 14;
/// Random starts tried before giving up, which only a broken random stream (one repeating
/// the same bytes) exhausts.
const MAX_SIEVE_WINDOWS: usize = 64;
/// FIPS 186-5 A.1.3 step 5.4: the primes must differ in their top 100 bits.
const MIN_PRIME_DIFFERENCE_BITS: usize = 100;
/// Second primes drawn for one first prime before giving up (as above).
const MAX_SECOND_PRIME_ATTEMPTS: usize = 16;
/// RFC 8017 9.2 note: EMSA-PKCS1-v1_5 is 0x00 0x01 PS 0x00 T with PS at least 8 bytes.
const PKCS1_V15_OVERHEAD_BYTES: usize = 3;
const PKCS1_V15_BLOCK_TYPE: u8 = 0x01;
const PKCS1_V15_PADDING_BYTE: u8 = 0xff;

/// The Miller-Rabin rounds for a random candidate of `bits` bits, from FIPS 186-5 Table B.1
/// (error at most 2^-100 for 512-bit primes, 2^-112 for 1024 and 2^-128 above).
fn miller_rabin_rounds(bits: usize) -> usize {
    match bits {
        1536.. => 4,
        1024.. => 5,
        _ => 7,
    }
}

/// An RSA key pair kept in CRT form. `Debug` shows only the public half.
#[derive(Clone, PartialEq, Eq)]
pub struct RsaPrivateKey {
    pub public: RsaPublicKey,
    /// p, the prime `prime()` returns; the one a TPM keeps.
    first_prime: BigUint,
    /// q = n / p.
    second_prime: BigUint,
    /// d mod (p - 1), computed directly as e^-1 mod (p - 1).
    first_exponent: BigUint,
    /// d mod (q - 1).
    second_exponent: BigUint,
    /// q^-1 mod p.
    coefficient: BigUint,
}

impl core::fmt::Debug for RsaPrivateKey {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RsaPrivateKey")
            .field("public", &self.public)
            .finish_non_exhaustive()
    }
}

impl RsaPrivateKey {
    /// A new key of `bits` bits (1024 to 4096, a multiple of 16) with public exponent
    /// `exponent` (0 for 65537; it must be odd and at least 3). The same `random` byte
    /// stream always gives the same key, which is how a TPM rederives a primary key from its
    /// seed. `Unsupported` for parameters outside those, and for a stream so repetitive that
    /// no prime turns up.
    pub fn generate(
        bits: usize,
        exponent: u32,
        random: &mut dyn FnMut(&mut [u8]),
    ) -> Result<Self, Error> {
        let exponent = if exponent == 0 {
            DEFAULT_PUBLIC_EXPONENT
        } else {
            exponent
        };
        if exponent < 3
            || exponent % 2 == 0
            || !(MIN_GENERATED_MODULUS_BITS..=MAX_GENERATED_MODULUS_BITS).contains(&bits)
            || !bits.is_multiple_of(GENERATED_MODULUS_GRANULE_BITS)
        {
            return Err(Error::Unsupported);
        }
        let prime_bits = bits / 2;
        let small_primes = small_primes();
        let first = generate_prime(prime_bits, exponent, &small_primes, random)?;
        for _ in 0..MAX_SECOND_PRIME_ATTEMPTS {
            let second = generate_prime(prime_bits, exponent, &small_primes, random)?;
            let difference = if first > second {
                first.sub(&second)
            } else {
                second.sub(&first)
            };
            if difference.bits() <= prime_bits - MIN_PRIME_DIFFERENCE_BITS {
                continue;
            }
            let modulus = first.mul(&second);
            let public =
                RsaPublicKey::new(&modulus.to_minimal_be_bytes(), &exponent.to_be_bytes())?;
            return Self::from_primes(public, first, second);
        }
        Err(Error::Unsupported)
    }

    /// Rebuilds a key from its modulus, public exponent and one of its primes (big-endian,
    /// leading zeros allowed). `Malformed` when the prime does not divide the modulus, is
    /// longer than half of it, or leaves no private exponent; `Unsupported` for a modulus or
    /// exponent `RsaPublicKey::new` refuses.
    pub fn from_prime(modulus: &[u8], exponent: &[u8], prime: &[u8]) -> Result<Self, Error> {
        let public = RsaPublicKey::new(modulus, exponent)?;
        let first = BigUint::from_be_bytes(prime);
        let (second, remainder) = BigUint::from_be_bytes(&public.modulus)
            .div_rem(&first)
            .ok_or(Error::Malformed)?;
        if !remainder.is_zero() {
            return Err(Error::Malformed);
        }
        Self::from_primes(public, first, second)
    }

    fn from_primes(public: RsaPublicKey, first: BigUint, second: BigUint) -> Result<Self, Error> {
        let one = BigUint::from_u64(1);
        // Odd primes above 1 are what Montgomery arithmetic needs; the length bound keeps
        // `prime()` exact.
        if !first.is_odd()
            || !second.is_odd()
            || first.is_one()
            || second.is_one()
            || first.bits().div_ceil(8) > public.size() / 2
        {
            return Err(Error::Malformed);
        }
        let exponent = BigUint::from_be_bytes(&public.exponent);
        let first_exponent =
            bignum::mod_inverse(&exponent, &first.sub(&one)).ok_or(Error::Malformed)?;
        let second_exponent =
            bignum::mod_inverse(&exponent, &second.sub(&one)).ok_or(Error::Malformed)?;
        let coefficient = bignum::mod_inverse(&second, &first).ok_or(Error::Malformed)?;
        Ok(Self {
            public,
            first_prime: first,
            second_prime: second,
            first_exponent,
            second_exponent,
            coefficient,
        })
    }

    /// The first prime, big-endian, left-padded to half the modulus length.
    pub fn prime(&self) -> Vec<u8> {
        // `from_primes` checked that it fits.
        self.first_prime
            .to_be_bytes(self.public.size() / 2)
            .unwrap_or_default()
    }

    /// `input^d mod n` by the Chinese remainder theorem, left-padded to the modulus length.
    /// `Malformed` when the input is not below the modulus.
    pub fn private_operation(&self, input: &[u8]) -> Result<Vec<u8>, Error> {
        let input = strip_leading_zeros(input);
        let modulus = BigUint::from_be_bytes(&self.public.modulus);
        let value = BigUint::from_be_bytes(input);
        if input.len() > self.public.size() || value >= modulus {
            return Err(Error::Malformed);
        }
        let first = Montgomery::new(&self.first_prime).ok_or(Error::Malformed)?;
        let second = Montgomery::new(&self.second_prime).ok_or(Error::Malformed)?;
        // Garner: m = m2 + q * (qinv * (m1 - m2) mod p).
        let first_result = first.power(&value, &self.first_exponent);
        let second_result = second.power(&value, &self.second_exponent);
        let second_reduced = second_result
            .rem(&self.first_prime)
            .ok_or(Error::Malformed)?;
        let difference = if first_result >= second_reduced {
            first_result.sub(&second_reduced)
        } else {
            first_result.add(&self.first_prime).sub(&second_reduced)
        };
        let lift = first.multiply_modular(&self.coefficient, &difference);
        second_result
            .add(&lift.mul(&self.second_prime))
            .to_be_bytes(self.public.size())
            .ok_or(Error::Malformed)
    }

    /// RSAES-OAEP decryption (RFC 8017 7.1.2) with MGF1 over `hash`. `label` is used as
    /// given (the TPM's labels carry their NUL). Any failure, including a ciphertext that is
    /// not exactly the modulus length, is `Malformed`.
    pub fn decrypt_oaep(
        &self,
        hash: HashAlgorithm,
        label: &[u8],
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, Error> {
        if ciphertext.len() != self.public.size() {
            return Err(Error::Malformed);
        }
        let encoded = self.private_operation(ciphertext)?;
        oaep::decode(hash, label, &encoded)
    }

    /// An RSASSA-PKCS1-v1_5 signature (RFC 8017 8.2.1) over `digest`, a hash made with
    /// `algorithm`. `Malformed` for a digest of the wrong length, `Unsupported` for a key too
    /// small for the DigestInfo.
    pub fn sign_pkcs1v15(&self, algorithm: HashAlgorithm, digest: &[u8]) -> Result<Vec<u8>, Error> {
        if digest.len() != algorithm.digest_len() {
            return Err(Error::Malformed);
        }
        let digest_info = digest_info(algorithm, digest, true);
        let size = self.public.size();
        if size < digest_info.len() + MIN_PADDING_BYTES + PKCS1_V15_OVERHEAD_BYTES {
            return Err(Error::Unsupported);
        }
        let mut encoded = vec![0x00, PKCS1_V15_BLOCK_TYPE];
        encoded.resize(size - digest_info.len() - 1, PKCS1_V15_PADDING_BYTE);
        encoded.push(0x00);
        encoded.extend(digest_info);
        self.private_operation(&encoded)
    }
}

/// The odd primes below `SMALL_PRIME_LIMIT` (sieve of Eratosthenes).
fn small_primes() -> Vec<u32> {
    let limit = SMALL_PRIME_LIMIT as usize;
    let mut composite = vec![false; limit];
    let mut primes = Vec::new();
    for candidate in 3..limit {
        if candidate % 2 == 0 || composite[candidate] {
            continue;
        }
        primes.push(candidate as u32);
        for multiple in (candidate * candidate..limit).step_by(2 * candidate) {
            composite[multiple] = true;
        }
    }
    primes
}

/// A random probable prime of exactly `bits` bits (a multiple of 8) with its top two bits set
/// and gcd(exponent, prime - 1) = 1: a random odd start, a sieve over the window of odd
/// numbers after it, then Miller-Rabin on the survivors in order.
fn generate_prime(
    bits: usize,
    exponent: u32,
    small_primes: &[u32],
    random: &mut dyn FnMut(&mut [u8]),
) -> Result<BigUint, Error> {
    let rounds = miller_rabin_rounds(bits);
    let mut start_bytes = vec![0u8; bits / 8];
    for _ in 0..MAX_SIEVE_WINDOWS {
        random(&mut start_bytes);
        start_bytes[0] |= TOP_TWO_BITS;
        if let Some(last) = start_bytes.last_mut() {
            *last |= 1;
        }
        let start = BigUint::from_be_bytes(&start_bytes);
        let mut composite = [false; SIEVE_WINDOW];
        for &small in small_primes {
            // start + 2 * offset = 0 (mod small) when offset = -start / 2, and (small + 1) / 2
            // is 2^-1 modulo an odd number.
            let small = u64::from(small);
            let negated = (small - u64::from(start.rem_u32(small as u32))) % small;
            let mut offset = (negated * small.div_ceil(2) % small) as usize;
            while offset < SIEVE_WINDOW {
                composite[offset] = true;
                offset += small as usize;
            }
        }
        for (offset, _) in composite.iter().enumerate().filter(|(_, is)| !**is) {
            let candidate = start.add(&BigUint::from_u64(2 * offset as u64));
            if candidate.bits() != bits {
                // Carried out of the top: the rest of the window is too long as well.
                break;
            }
            // (candidate - 1) mod exponent, then gcd with the exponent.
            let below = (candidate.rem_u32(exponent) + exponent - 1) % exponent;
            if bignum::gcd_u64(u64::from(below), u64::from(exponent)) != 1 {
                continue;
            }
            if is_probable_prime(&candidate, rounds, random) {
                return Ok(candidate);
            }
        }
    }
    Err(Error::Unsupported)
}

/// Miller-Rabin (FIPS 186-5 B.3.1) with bases from `random`, for an odd candidate above 3.
fn is_probable_prime(
    candidate: &BigUint,
    rounds: usize,
    random: &mut dyn FnMut(&mut [u8]),
) -> bool {
    let Some(montgomery) = Montgomery::new(candidate) else {
        return false;
    };
    let one = BigUint::from_u64(1);
    let two = BigUint::from_u64(2);
    let minus_one = candidate.sub(&one);
    let twos = minus_one.trailing_zeros();
    let odd_part = minus_one.shift_right(twos);
    // Bases in [2, candidate - 2].
    let base_range = candidate.sub(&BigUint::from_u64(3));
    let mut base_bytes = vec![0u8; candidate.bits().div_ceil(8)];
    'rounds: for _ in 0..rounds {
        random(&mut base_bytes);
        let Some(base) = BigUint::from_be_bytes(&base_bytes).rem(&base_range) else {
            return false;
        };
        let mut value = montgomery.power(&base.add(&two), &odd_part);
        if value.is_one() || value == minus_one {
            continue;
        }
        for _ in 1..twos {
            value = montgomery.multiply_modular(&value, &value);
            if value == minus_one {
                continue 'rounds;
            }
            if value.is_one() {
                return false;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha;

    /// A 2048-bit key with exponent 65537 made by Python `cryptography`, and a message
    /// "uemulator salt" it OAEP-encrypted with SHA-256 and the label "SECRET\0"
    /// (`hack/crypto/make-tpm-fixtures.py`).
    const FIXTURE_MODULUS: &[u8] = include_bytes!("../testdata/rsa2048-modulus.bin");
    const FIXTURE_PRIME: &[u8] = include_bytes!("../testdata/rsa2048-prime.bin");
    const FIXTURE_CIPHERTEXT: &[u8] = include_bytes!("../testdata/rsa2048-oaep-sha256.bin");
    const FIXTURE_MESSAGE: &[u8] = b"uemulator salt";
    const SECRET_LABEL: &[u8] = b"SECRET\0";
    const F4: [u8; 3] = [0x01, 0x00, 0x01];

    /// A deterministic byte stream (xorshift64*), standing in for the TPM's DRBG.
    fn stream(seed: u64) -> impl FnMut(&mut [u8]) {
        let mut state = seed | 1;
        move |bytes: &mut [u8]| {
            for byte in bytes {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                *byte = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8;
            }
        }
    }

    fn generate(bits: usize, seed: u64) -> RsaPrivateKey {
        RsaPrivateKey::generate(bits, 0, &mut stream(seed)).unwrap()
    }

    #[test]
    fn generated_keys_encrypt_decrypt_sign_and_verify() {
        let key = generate(1024, 1);
        assert_eq!(key.public.size(), 128);
        assert_eq!(key.public.modulus[0] & 0x80, 0x80);
        assert_eq!(key.public.exponent, F4);
        let ciphertext = key
            .public
            .encrypt_oaep(HashAlgorithm::Sha256, SECRET_LABEL, b"salt", &mut stream(9))
            .unwrap();
        assert_eq!(
            key.decrypt_oaep(HashAlgorithm::Sha256, SECRET_LABEL, &ciphertext),
            Ok(b"salt".to_vec())
        );
        assert_eq!(
            key.decrypt_oaep(HashAlgorithm::Sha256, b"SECRET", &ciphertext),
            Err(Error::Malformed),
            "label without its NUL"
        );
        let digest = sha::sha256(b"uemulator");
        let signature = key.sign_pkcs1v15(HashAlgorithm::Sha256, &digest).unwrap();
        key.public
            .verify_pkcs1v15(HashAlgorithm::Sha256, &digest, &signature)
            .unwrap();
        assert_eq!(
            key.sign_pkcs1v15(HashAlgorithm::Sha256, &digest[..20]),
            Err(Error::Malformed)
        );
        // (m^d)^e = m through the CRT path.
        let message = [0x5a; 100];
        let raised = key.private_operation(&message).unwrap();
        assert_eq!(key.public.public_operation(&raised).unwrap()[28..], message);
    }

    #[test]
    fn generation_is_deterministic_and_rebuildable_from_one_prime() {
        let key = generate(1024, 7);
        assert_eq!(key, generate(1024, 7));
        assert_ne!(key.public, generate(1024, 8).public);
        assert_eq!(key.prime().len(), 64);
        let rebuilt =
            RsaPrivateKey::from_prime(&key.public.modulus, &key.public.exponent, &key.prime())
                .unwrap();
        assert_eq!(rebuilt, key);
        // The other prime rebuilds the same key with the primes swapped.
        let second = key.second_prime.to_be_bytes(64).unwrap();
        let swapped =
            RsaPrivateKey::from_prime(&key.public.modulus, &key.public.exponent, &second).unwrap();
        assert_eq!(swapped.prime(), second);
        assert_eq!(
            swapped.private_operation(&[3; 50]),
            key.private_operation(&[3; 50])
        );
    }

    #[test]
    fn a_2048_bit_key_generates_quickly() {
        const KEYS: u32 = 4;
        let started = std::time::Instant::now();
        for seed in 0..KEYS {
            let key = generate(2048, 2048 + u64::from(seed));
            assert_eq!(key.public.size(), 256);
            assert_eq!(key.public.modulus[0] & 0x80, 0x80);
            let digest = sha::sha384(b"uemulator");
            let signature = key.sign_pkcs1v15(HashAlgorithm::Sha384, &digest).unwrap();
            key.public
                .verify_pkcs1v15(HashAlgorithm::Sha384, &digest, &signature)
                .unwrap();
        }
        std::println!(
            "RSA-2048 generation (and one signature) took {:?} on average",
            started.elapsed() / KEYS
        );
    }

    #[test]
    #[ignore = "slow; for timing"]
    fn large_keys_generate() {
        for bits in [3072, 4096] {
            let started = std::time::Instant::now();
            let key = generate(bits, bits as u64);
            assert_eq!(key.public.size(), bits / 8);
            std::println!("RSA-{bits} generation took {:?}", started.elapsed());
        }
    }

    #[test]
    fn other_exponents_and_sizes() {
        let key = RsaPrivateKey::generate(1024, 3, &mut stream(3)).unwrap();
        assert_eq!(key.public.exponent, [3]);
        let raised = key.private_operation(&[7; 64]).unwrap();
        assert_eq!(key.public.public_operation(&raised).unwrap()[64..], [7; 64]);
        for (bits, exponent) in [(1024, 2), (1024, 1), (1024, 65536), (1000, 0), (512, 0)] {
            assert_eq!(
                RsaPrivateKey::generate(bits, exponent, &mut stream(1)),
                Err(Error::Unsupported),
                "{bits} bits, exponent {exponent}"
            );
        }
        assert_eq!(
            RsaPrivateKey::generate(8192, 0, &mut stream(1)),
            Err(Error::Unsupported)
        );
        // A stream that repeats itself gives the same prime every time, so no second one.
        let mut zeros = |bytes: &mut [u8]| bytes.fill(0);
        assert_eq!(
            RsaPrivateKey::generate(1024, 0, &mut zeros),
            Err(Error::Unsupported)
        );
    }

    #[test]
    fn a_python_oaep_ciphertext_decrypts() {
        let key = RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, FIXTURE_PRIME).unwrap();
        assert_eq!(key.public.size(), 256);
        assert_eq!(key.prime(), FIXTURE_PRIME);
        assert_eq!(
            key.decrypt_oaep(HashAlgorithm::Sha256, SECRET_LABEL, FIXTURE_CIPHERTEXT),
            Ok(FIXTURE_MESSAGE.to_vec())
        );
        assert_eq!(
            key.decrypt_oaep(HashAlgorithm::Sha1, SECRET_LABEL, FIXTURE_CIPHERTEXT),
            Err(Error::Malformed)
        );
        assert_eq!(
            key.decrypt_oaep(
                HashAlgorithm::Sha256,
                SECRET_LABEL,
                &FIXTURE_CIPHERTEXT[1..]
            ),
            Err(Error::Malformed)
        );
        assert_eq!(
            key.decrypt_oaep(HashAlgorithm::Sha256, SECRET_LABEL, FIXTURE_MODULUS),
            Err(Error::Malformed),
            "ciphertext equal to the modulus"
        );
    }

    #[test]
    fn broken_padding_is_refused() {
        let key = RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, FIXTURE_PRIME).unwrap();
        let hash = HashAlgorithm::Sha256;
        let encrypt_raw = |encoded: &[u8]| key.public.raw_public(encoded).unwrap();
        // Build the encoding by hand, then break one part of it at a time.
        let build = |leading: u8, separator: u8, label: &[u8]| {
            let mut data_block = hash.digest(label);
            data_block.resize(256 - 1 - 32 - FIXTURE_MESSAGE.len() - 1, 0);
            data_block.push(separator);
            data_block.extend_from_slice(FIXTURE_MESSAGE);
            let mut seed = [0x42u8; 32].to_vec();
            oaep::xor_mask(&mut data_block, hash, &seed);
            oaep::xor_mask(&mut seed, hash, &data_block);
            let mut encoded = vec![leading];
            encoded.extend(seed);
            encoded.extend(data_block);
            encoded
        };
        let good = encrypt_raw(&build(0, 1, SECRET_LABEL));
        assert_eq!(
            key.decrypt_oaep(hash, SECRET_LABEL, &good),
            Ok(FIXTURE_MESSAGE.to_vec())
        );
        for broken in [
            build(0, 2, SECRET_LABEL),
            build(0, 1, b"other"),
            build(1, 1, SECRET_LABEL),
        ] {
            let ciphertext = encrypt_raw(&broken);
            assert_eq!(
                key.decrypt_oaep(hash, SECRET_LABEL, &ciphertext),
                Err(Error::Malformed)
            );
        }
    }

    #[test]
    fn from_prime_refuses_what_is_not_a_factor() {
        let key = RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, FIXTURE_PRIME).unwrap();
        let mut wrong = FIXTURE_PRIME.to_vec();
        wrong[127] ^= 2;
        assert_eq!(
            RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, &wrong),
            Err(Error::Malformed)
        );
        assert_eq!(
            RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, &[]),
            Err(Error::Malformed)
        );
        assert_eq!(
            RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, &[1]),
            Err(Error::Malformed)
        );
        assert_eq!(
            RsaPrivateKey::from_prime(FIXTURE_MODULUS, &F4, FIXTURE_MODULUS),
            Err(Error::Malformed)
        );
        assert_eq!(
            RsaPrivateKey::from_prime(&[0x10], &F4, &[2]),
            Err(Error::Unsupported)
        );
        assert_eq!(key.private_operation(&[0xff; 257]), Err(Error::Malformed));
        assert!(format!("{key:?}").starts_with("RsaPrivateKey { public: "));
    }
}
