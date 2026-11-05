//! HMAC (RFC 2104) over the hashes in `sha`.

use alloc::vec::Vec;

use crate::sha::{HashAlgorithm, Hasher, SHA1_BLOCK_BYTES, SHA256_BLOCK_BYTES, SHA512_BLOCK_BYTES};

const INNER_PAD: u8 = 0x36;
const OUTER_PAD: u8 = 0x5c;

fn block_bytes(algorithm: HashAlgorithm) -> usize {
    match algorithm {
        HashAlgorithm::Sha1 => SHA1_BLOCK_BYTES,
        HashAlgorithm::Sha256 => SHA256_BLOCK_BYTES,
        // SHA-384 is SHA-512 with another start and a shorter output: same block.
        HashAlgorithm::Sha384 | HashAlgorithm::Sha512 => SHA512_BLOCK_BYTES,
    }
}

/// HMAC of the concatenation of `parts` (which saves callers copying a TPM's many fields
/// into one buffer). Any key length, including empty.
pub fn hmac(algorithm: HashAlgorithm, key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    let block = block_bytes(algorithm);
    let mut padded_key = if key.len() > block {
        algorithm.digest(key)
    } else {
        key.to_vec()
    };
    padded_key.resize(block, 0);

    let mut inner = Hasher::new(algorithm);
    let inner_key: Vec<u8> = padded_key.iter().map(|byte| byte ^ INNER_PAD).collect();
    inner.update(&inner_key);
    for part in parts {
        inner.update(part);
    }
    let inner_digest = inner.finish();

    let mut outer = Hasher::new(algorithm);
    let outer_key: Vec<u8> = padded_key.iter().map(|byte| byte ^ OUTER_PAD).collect();
    outer.update(&outer_key);
    outer.update(&inner_digest);
    outer.finish()
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

    const CASE_1_KEY: [u8; 20] = [0x0b; 20];
    const CASE_1_DATA: &[u8] = b"Hi There";
    const CASE_2_KEY: &[u8] = b"Jefe";
    const CASE_2_DATA: &[u8] = b"what do ya want for nothing?";
    const LARGE_KEY_DATA: &[u8] = b"Test Using Larger Than Block-Size Key - Hash Key First";
    const LARGE_KEY_AND_DATA: &[u8] = b"This is a test using a larger than block-size key and a \
        larger than block-size data. The key needs to be hashed before being used by the HMAC \
        algorithm.";

    #[test]
    fn rfc_4231_vectors() {
        // Cases 1, 2, 6 and 7 (5 is truncated output, 3 and 4 add nothing over these). The
        // 131-byte key is longer than both block sizes, so it exercises key hashing for all.
        let large_key = [0xaa; 131];
        let cases: [(&[u8], &[u8], [&str; 3]); 4] = [
            (
                &CASE_1_KEY,
                CASE_1_DATA,
                [
                    "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
                    "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59cfaea9ea9076ede7f4af152e8b2fa9cb6",
                    "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854",
                ],
            ),
            (
                CASE_2_KEY,
                CASE_2_DATA,
                [
                    "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
                    "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e8e2240ca5e69e2c78b3239ecfab21649",
                    "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea2505549758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737",
                ],
            ),
            (
                &large_key,
                LARGE_KEY_DATA,
                [
                    "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
                    "4ece084485813e9088d2c63a041bc5b44f9ef1012a2b588f3cd11f05033ac4c60c2ef6ab4030fe8296248df163f44952",
                    "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f3526b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598",
                ],
            ),
            (
                &large_key,
                LARGE_KEY_AND_DATA,
                [
                    "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
                    "6617178e941f020d351e2f254e8fd32c602420feb0b8fb9adccebb82461e99c5a678cc31e799176d3860e6110c46523e",
                    "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58",
                ],
            ),
        ];
        for (key, data, expected) in cases {
            for (algorithm, expected) in [
                HashAlgorithm::Sha256,
                HashAlgorithm::Sha384,
                HashAlgorithm::Sha512,
            ]
            .into_iter()
            .zip(expected)
            {
                assert_eq!(
                    hmac(algorithm, key, &[data]),
                    hex(expected),
                    "{algorithm:?}"
                );
            }
        }
    }

    #[test]
    fn rfc_2202_sha1_vectors() {
        let large_key = [0xaa; 80];
        let cases: [(&[u8], &[u8], &str); 3] = [
            (
                &CASE_1_KEY,
                CASE_1_DATA,
                "b617318655057264e28bc0b6fb378c8ef146be00",
            ),
            (
                CASE_2_KEY,
                CASE_2_DATA,
                "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79",
            ),
            (
                &large_key,
                LARGE_KEY_DATA,
                "aa4ae5e15272d00e95705637ce8a3b55ed402112",
            ),
        ];
        for (key, data, expected) in cases {
            assert_eq!(hmac(HashAlgorithm::Sha1, key, &[data]), hex(expected));
        }
    }

    #[test]
    fn parts_concatenate_and_the_key_may_be_empty() {
        assert_eq!(
            hmac(
                HashAlgorithm::Sha256,
                CASE_2_KEY,
                &[b"what do ya", b"", b" want for nothing?"]
            ),
            hmac(HashAlgorithm::Sha256, CASE_2_KEY, &[CASE_2_DATA])
        );
        assert_eq!(
            hmac(HashAlgorithm::Sha256, b"", &[]),
            hex("b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad")
        );
    }
}
