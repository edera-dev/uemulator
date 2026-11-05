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
