//! RSAES-OAEP (RFC 8017 7.1) with MGF1 over the same hash, as the TPM uses it to carry
//! session salts and imported seeds.

use alloc::vec;
use alloc::vec::Vec;

use super::RsaPublicKey;
use crate::Error;
use crate::sha::{HashAlgorithm, Hasher};

/// The byte between the zero padding and the message in the data block.
const MESSAGE_SEPARATOR: u8 = 0x01;
/// The encoded message's first byte is zero so that it is below the modulus.
const LEADING_BYTES: usize = 1;

/// MGF1 (RFC 8017 B.2.1): Hash(seed || counter) for counter = 0, 1, ..., truncated.
fn mgf1(hash: HashAlgorithm, seed: &[u8], length: usize) -> Vec<u8> {
    let mut mask = Vec::with_capacity(length + hash.digest_len());
    let mut counter = 0u32;
    while mask.len() < length {
        let mut hasher = Hasher::new(hash);
        hasher.update(seed);
        hasher.update(&counter.to_be_bytes());
        mask.extend(hasher.finish());
        counter = counter.wrapping_add(1);
    }
    mask.truncate(length);
    mask
}

pub(super) fn xor_mask(target: &mut [u8], hash: HashAlgorithm, seed: &[u8]) {
    let mask = mgf1(hash, seed, target.len());
    for (byte, mask) in target.iter_mut().zip(mask) {
        *byte ^= mask;
    }
}

/// The smallest modulus, in bytes, that has room for OAEP with `hash`.
fn minimum_modulus_bytes(hash: HashAlgorithm) -> usize {
    2 * hash.digest_len() + 2
}

impl RsaPublicKey {
    /// RSAES-OAEP encryption (RFC 8017 7.1.1). `random` supplies the seed. A message longer
    /// than the modulus allows is `Malformed`; a modulus too small for the hash is
    /// `Unsupported`.
    pub fn encrypt_oaep(
        &self,
        hash: HashAlgorithm,
        label: &[u8],
        message: &[u8],
        random: &mut dyn FnMut(&mut [u8]),
    ) -> Result<Vec<u8>, Error> {
        let size = self.size();
        let digest_len = hash.digest_len();
        if size < minimum_modulus_bytes(hash) {
            return Err(Error::Unsupported);
        }
        if message.len() > size - minimum_modulus_bytes(hash) {
            return Err(Error::Malformed);
        }
        // DB = lHash || zeros || 0x01 || M.
        let mut data_block = hash.digest(label);
        data_block.resize(size - LEADING_BYTES - digest_len - message.len() - 1, 0);
        data_block.push(MESSAGE_SEPARATOR);
        data_block.extend_from_slice(message);
        let mut seed = vec![0u8; digest_len];
        random(&mut seed);
        xor_mask(&mut data_block, hash, &seed);
        xor_mask(&mut seed, hash, &data_block);

        let mut encoded = vec![0u8; LEADING_BYTES];
        encoded.extend(seed);
        encoded.extend(data_block);
        self.raw_public(&encoded).ok_or(Error::Unsupported)
    }
}

/// EME-OAEP decoding (RFC 8017 7.1.2 step 3) of the `size`-byte result of the private
/// operation. Every failure is the same `Malformed`, though nothing here is constant time.
pub(super) fn decode(hash: HashAlgorithm, label: &[u8], encoded: &[u8]) -> Result<Vec<u8>, Error> {
    let digest_len = hash.digest_len();
    if encoded.len() < minimum_modulus_bytes(hash) {
        return Err(Error::Malformed);
    }
    let (leading, rest) = encoded.split_at(LEADING_BYTES);
    let (masked_seed, masked_data_block) = rest.split_at(digest_len);
    let mut seed = masked_seed.to_vec();
    xor_mask(&mut seed, hash, masked_data_block);
    let mut data_block = masked_data_block.to_vec();
    xor_mask(&mut data_block, hash, &seed);

    let (label_hash, padded_message) = data_block.split_at(digest_len);
    let padding = padded_message.iter().take_while(|&&byte| byte == 0).count();
    if leading[0] != 0
        || label_hash != hash.digest(label).as_slice()
        || padded_message.get(padding) != Some(&MESSAGE_SEPARATOR)
    {
        return Err(Error::Malformed);
    }
    Ok(padded_message[padding + 1..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha;

    #[test]
    fn mgf1_matches_the_definition() {
        let mask = mgf1(HashAlgorithm::Sha256, b"seed", 40);
        let mut first = b"seed".to_vec();
        first.extend(0u32.to_be_bytes());
        let mut second = b"seed".to_vec();
        second.extend(1u32.to_be_bytes());
        assert_eq!(mask[..32], sha::sha256(&first));
        assert_eq!(mask[32..], sha::sha256(&second)[..8]);
    }

    #[test]
    fn decoding_refuses_short_blocks() {
        assert_eq!(
            decode(HashAlgorithm::Sha256, b"", &[0; 65]),
            Err(Error::Malformed)
        );
    }
}
