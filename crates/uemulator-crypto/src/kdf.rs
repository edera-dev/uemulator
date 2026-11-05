//! The TPM's key derivation functions (TPM 2.0 Library Part 1, 11.4.10): KDFa for session
//! and storage keys, KDFe for secrets agreed by ECDH.

use alloc::vec::Vec;

use crate::hmac::hmac;
use crate::sha::{HashAlgorithm, Hasher};

/// Both KDFs hash the label as a NUL-terminated string; callers pass it without the NUL.
const LABEL_TERMINATOR: &[u8] = &[0];

/// SP800-108 counter mode with HMAC: HMAC(key, i || label || 0 || context_u || context_v ||
/// bits) for i = 1, 2, ..., cut to `bits`.
pub fn kdfa(
    algorithm: HashAlgorithm,
    key: &[u8],
    label: &[u8],
    context_u: &[u8],
    context_v: &[u8],
    bits: u32,
) -> Vec<u8> {
    let bits_field = bits.to_be_bytes();
    derive(algorithm, bits, |counter| {
        hmac(
            algorithm,
            key,
            &[
                &counter.to_be_bytes(),
                label,
                LABEL_TERMINATOR,
                context_u,
                context_v,
                &bits_field,
            ],
        )
    })
}

/// SP800-56A concatenation KDF: Hash(i || z || label || 0 || party_u || party_v) for i = 1,
/// 2, ..., cut to `bits`.
pub fn kdfe(
    algorithm: HashAlgorithm,
    z: &[u8],
    label: &[u8],
    party_u: &[u8],
    party_v: &[u8],
    bits: u32,
) -> Vec<u8> {
    derive(algorithm, bits, |counter| {
        let mut hasher = Hasher::new(algorithm);
        for part in [
            &counter.to_be_bytes()[..],
            z,
            label,
            LABEL_TERMINATOR,
            party_u,
            party_v,
        ] {
            hasher.update(part);
        }
        hasher.finish()
    })
}

/// Concatenates `block(1)`, `block(2)`, ... to ceil(bits / 8) bytes. When `bits` is not a
/// whole number of bytes the extra high bits of the first byte are cleared, as the TPM
/// reference implementation does (the value is a big-endian number of `bits` bits).
fn derive(algorithm: HashAlgorithm, bits: u32, mut block: impl FnMut(u32) -> Vec<u8>) -> Vec<u8> {
    let length = bits.div_ceil(8) as usize;
    let mut out = Vec::with_capacity(length + algorithm.digest_len());
    let mut counter = 1u32;
    while out.len() < length {
        out.extend(block(counter));
        counter = counter.wrapping_add(1);
    }
    out.truncate(length);
    let partial_bits = bits % 8;
    if partial_bits != 0 {
        out[0] &= (1u8 << partial_bits) - 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
            .collect()
    }

    // Expected values from `hack/crypto/make-tpm-fixtures.py`, a direct Python transcription
    // of Part 1 11.4.10 over hmac and hashlib.

    #[test]
    fn kdfa_vectors() {
        assert_eq!(
            kdfa(
                HashAlgorithm::Sha256,
                b"session key",
                b"ATH",
                b"nonce-caller",
                b"nonce-tpm",
                256
            ),
            hex("8ce52a33a20a2a44d7cc500f50ebe5f4a22ee5b6a7064605d0c1f5ea31be5c2f")
        );
        // Three SHA-1 blocks and a 4-bit first byte.
        assert_eq!(
            kdfa(HashAlgorithm::Sha1, b"seed", b"CFB", b"name", b"", 300),
            hex("013f7ed2f07e603fc3366a563a81cb670b5bd60f0187e81a8be814db3190758e31d6661e286a")
        );
        assert_eq!(
            kdfa(HashAlgorithm::Sha384, b"", b"STORAGE", b"", b"", 131),
            hex("07586d23f7710e254cc13b6c9630b2e757")
        );
        assert!(kdfa(HashAlgorithm::Sha256, b"k", b"L", b"", b"", 0).is_empty());
    }

    #[test]
    fn kdfe_vectors() {
        let z: Vec<u8> = (0..32).collect();
        assert_eq!(
            kdfe(
                HashAlgorithm::Sha256,
                &z,
                b"SECRET",
                b"party u",
                b"party v",
                256
            ),
            hex("4ed001dd641b2cbe71823529bf2dcc7b0f4d4da563e445d205b2569d08737449")
        );
        assert_eq!(
            kdfe(HashAlgorithm::Sha512, &z, b"DUPLICATE", b"", b"v", 777),
            hex(
                "012e20e4457f27759171dc54ea3ac23a0a53d5ede2962562c85b0c250e17dd7c93631946e9685919b8391cee62c8d77873fa84b167f5db9f823b82de4fab6ee49faa3978ae36d812e4205664c527656d37a3160fbb7d829896cfeae5550a3a839989"
            )
        );
    }
}
