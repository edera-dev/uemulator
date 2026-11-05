//! RSA (RFC 8017): public keys and RSASSA-PKCS1-v1_5 verification here, RSAES-OAEP in `oaep`,
//! and private keys (generation, CRT, decryption, signing) in `private`.

use alloc::vec;
use alloc::vec::Vec;

use crate::Error;
use crate::bignum::{BigUint, Montgomery};
use crate::der::{self, Reader, TAG_NULL, TAG_SEQUENCE, oid};
use crate::sha::HashAlgorithm;

mod oaep;
mod private;

pub use private::RsaPrivateKey;

/// Moduli above this are refused, which bounds the work a hostile certificate can ask for.
pub const MAX_MODULUS_BITS: usize = 8192;
/// RFC 8017 9.2: the 0xFF padding is at least eight bytes.
const MIN_PADDING_BYTES: usize = 8;

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
        self.raw_public(signature).ok_or(Error::BadSignature)
    }

    /// `input^e mod n` left-padded to the modulus length, or `None` when the input is not
    /// below the modulus (or the fields were set by hand to something that is not a key).
    fn raw_public(&self, input: &[u8]) -> Option<Vec<u8>> {
        let modulus = BigUint::from_be_bytes(&self.modulus);
        let base = BigUint::from_be_bytes(input);
        if base >= modulus {
            return None;
        }
        Montgomery::new(&modulus)?
            .power(&base, &BigUint::from_be_bytes(&self.exponent))
            .to_be_bytes(self.size())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha;

    /// A 1024-bit test key and a signature over "uemulator" made by
    /// `openssl dgst -sha256 -sign` (`hack/secure-boot/make-crypto-fixtures.sh`).
    const TEST_PUBLIC_KEY: &[u8] = include_bytes!("testdata/rsa1024-public.der");
    const TEST_SIGNATURE: &[u8] = include_bytes!("testdata/rsa1024-sha256.sig");

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
