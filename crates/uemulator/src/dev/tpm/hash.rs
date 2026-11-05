//! The TPM's hash algorithms by their `TPM_ALG_ID`, over `uemulator-crypto`.
#[cfg(freestanding)]
use crate::prelude::*;
use uemulator_crypto::sha::{HashAlgorithm, Hasher};

pub(crate) const ALG_SHA1: u16 = 0x0004;
pub(crate) const ALG_SHA256: u16 = 0x000b;
pub(crate) const ALG_SHA384: u16 = 0x000c;
pub(crate) const ALG_SHA512: u16 = 0x000d;
pub(crate) const ALG_NULL: u16 = 0x0010;

/// Every hash the TPM implements, in `TPM_ALG_ID` order (`TPM2_GetCapability` lists them so, and
/// `TPM2_PCR_Event` returns one digest per entry).
pub(crate) const HASH_ALGORITHMS: [u16; 4] = [ALG_SHA1, ALG_SHA256, ALG_SHA384, ALG_SHA512];

/// The largest digest, SHA-512's: `TPM_PT_MAX_DIGEST` and the cap on `TPM2_GetRandom`.
pub(crate) const MAX_DIGEST_BYTES: usize = 64;

fn algorithm(id: u16) -> Option<HashAlgorithm> {
    match id {
        ALG_SHA1 => Some(HashAlgorithm::Sha1),
        ALG_SHA256 => Some(HashAlgorithm::Sha256),
        ALG_SHA384 => Some(HashAlgorithm::Sha384),
        ALG_SHA512 => Some(HashAlgorithm::Sha512),
        _ => None,
    }
}

pub(crate) fn digest_len(id: u16) -> Option<usize> {
    algorithm(id).map(HashAlgorithm::digest_len)
}

pub(crate) struct Hash {
    pub id: u16,
    hasher: Hasher,
}

impl Hash {
    pub fn new(id: u16) -> Option<Self> {
        Some(Hash {
            id,
            hasher: Hasher::new(algorithm(id)?),
        })
    }

    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    pub fn finish(self) -> Vec<u8> {
        self.hasher.finish()
    }
}

pub(crate) fn digest(id: u16, parts: &[&[u8]]) -> Option<Vec<u8>> {
    let mut hash = Hash::new(id)?;
    for part in parts {
        hash.update(part);
    }
    Some(hash.finish())
}

pub(super) fn hash_algorithm(id: u16) -> Option<HashAlgorithm> {
    algorithm(id)
}

pub(super) fn hmac(id: u16, key: &[u8], parts: &[&[u8]]) -> Option<Vec<u8>> {
    Some(uemulator_crypto::hmac::hmac(algorithm(id)?, key, parts))
}

/// KDFa (Part 1, "KDFa()"): SP800-108 counter mode over HMAC. `label` is given without its
/// terminating NUL.
pub(super) fn kdfa(
    id: u16,
    key: &[u8],
    label: &[u8],
    context_u: &[u8],
    context_v: &[u8],
    bits: u32,
) -> Option<Vec<u8>> {
    Some(uemulator_crypto::kdf::kdfa(
        algorithm(id)?,
        key,
        label,
        context_u,
        context_v,
        bits,
    ))
}

/// KDFe (Part 1, "KDFe()"): the SP800-56A concatenation KDF ECDH salts go through.
pub(super) fn kdfe(
    id: u16,
    z: &[u8],
    label: &[u8],
    party_u: &[u8],
    party_v: &[u8],
    bits: u32,
) -> Option<Vec<u8>> {
    Some(uemulator_crypto::kdf::kdfe(
        algorithm(id)?,
        z,
        label,
        party_u,
        party_v,
        bits,
    ))
}

/// The deterministic bit stream a primary object is made from: KDFa's construction, block
/// after block, without the length field (which a stream does not know up front). The same
/// seed, label and contexts give the same key, which is what lets `TPM2_CreatePrimary` give back
/// the same primary for the same template every time. The reference implementation's DRBG
/// differs, so keys are not those another TPM would derive from the same seed; nothing can tell,
/// since seeds never leave a TPM.
pub(super) struct KdfStream {
    algorithm: HashAlgorithm,
    key: Vec<u8>,
    context: Vec<u8>,
    counter: u32,
    pending: Vec<u8>,
}

impl KdfStream {
    pub fn new(
        id: u16,
        key: &[u8],
        label: &[u8],
        context_u: &[u8],
        context_v: &[u8],
    ) -> Option<Self> {
        let mut context = label.to_vec();
        context.push(0);
        context.extend_from_slice(context_u);
        context.extend_from_slice(context_v);
        Some(KdfStream {
            algorithm: algorithm(id)?,
            key: key.to_vec(),
            context,
            counter: 0,
            pending: Vec::new(),
        })
    }

    pub fn fill(&mut self, bytes: &mut [u8]) {
        for byte in bytes {
            if self.pending.is_empty() {
                self.counter = self.counter.wrapping_add(1);
                let mut block = uemulator_crypto::hmac::hmac(
                    self.algorithm,
                    &self.key,
                    &[&self.counter.to_be_bytes(), &self.context],
                );
                block.reverse();
                self.pending = block;
            }
            *byte = self.pending.pop().unwrap_or(0);
        }
    }
}
