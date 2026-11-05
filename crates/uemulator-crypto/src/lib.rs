//! The cryptography the TPM device and the firmware's Secure Boot need, written for this
//! project: `no_std` with `alloc`, no dependencies. It holds private-key operations for the
//! emulated TPM, and nothing in it is constant time: the TPM's secrets are as safe as the host
//! process, and a guest that can time the emulator precisely could in principle learn them.
//! Every parser takes untrusted bytes (PE files, variables a guest writes, TPM commands) and
//! returns `Error` rather than panicking.
//!
//! - `sha`: `Sha1`, `Sha256`, `Sha384`, `Sha512` (`new`, `update`, `finish -> [u8; N]`), the
//!   one-shot `sha1`/`sha256`/`sha384`/`sha512`, and `HashAlgorithm` (`digest_len`, `oid`,
//!   `from_oid`, `digest -> Vec<u8>`) with the run-time `Hasher` over it.
//! - `der`: `Reader` over DER (`read`, `expect(tag)`, `optional(tag)`, `unsigned_integer`,
//!   `oid`, `bit_string`, `octet_string`, `algorithm`) yielding `Tlv { tag, contents, raw }`;
//!   the OIDs used here in `der::oid`.
//! - `hmac`: `hmac(HashAlgorithm, key, parts)` over the concatenated parts.
//! - `kdf`: the TPM's `kdfa` (SP800-108 counter mode, HMAC) and `kdfe` (SP800-56A
//!   concatenation), both taking the label without its NUL and cutting to a bit count.
//! - `aes`: `Aes::new(key)` with `encrypt_block`/`decrypt_block`, and the TPM's CFB128
//!   `cfb_encrypt`/`cfb_decrypt(key, iv, data)` in place.
//! - `rsa`: `RsaPublicKey::{from_subject_public_key_info, from_pkcs1, new}`,
//!   `verify_pkcs1v15(HashAlgorithm, digest, signature) -> Result<(), Error>` and
//!   `encrypt_oaep`; `RsaPrivateKey::{generate, from_prime}` (deterministic for a given random
//!   stream; rebuilt from n, e and one prime, as a TPM stores it) with `prime`,
//!   `private_operation`, `decrypt_oaep` and `sign_pkcs1v15`.
//! - `ecc`: NIST P-256 as free functions: `p256_public_key`, `p256_ecdh`,
//!   `p256_scalar_multiply`, `p256_ecdsa_sign`/`p256_ecdsa_verify`, the checks
//!   `p256_is_valid_private`/`p256_is_on_curve`, and `EccPoint`.
//! - `x509`: `Certificate::parse(der)` borrowing the input (`raw`, `tbs`, `serial`, `issuer`,
//!   `subject`, `public_key`, `signature_algorithm`, `signature`), `verifies`/`issued`/
//!   `is_self_signed`, and `tbs_digest(HashAlgorithm)` for dbx `EFI_CERT_X509_SHA*` entries.
//! - `pkcs7`: `SignedData::parse` (ContentInfo-wrapped or bare), `verify(Content, anchors)
//!   -> Verified` (every signer checked and chained to an anchor; `Verified::signers[i]` gives
//!   the anchor index and the chain, signer first), and `verify_signer` for the signature
//!   alone.
//! - `authenticode`: `digest(file, HashAlgorithm)`, `signatures(file)` (the PKCS#7 blobs of the
//!   certificate table), `signed_digest(&SignedData)` and `check_digest(file, &SignedData)`.
//!
//! Secure Boot image check, as the firmware does it: for each blob in `signatures`, parse it,
//! `check_digest`, then `verify(Content::Embedded, db certificates)`; check dbx against
//! `verify(..., dbx certificates)` and every `Verified::certificates()` TBS digest.
#![no_std]

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

pub mod aes;
pub mod authenticode;
mod bignum;
pub mod der;
pub mod ecc;
pub mod hmac;
pub mod kdf;
pub mod pkcs7;
pub mod rsa;
pub mod sha;
pub mod x509;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes are not a well-formed instance of what was expected.
    Malformed,
    /// Well formed, but an algorithm or form this crate does not handle.
    Unsupported,
    /// The signed digest does not match the content or image.
    DigestMismatch,
    /// The RSA signature does not verify with the key.
    BadSignature,
    /// No certificate matches the signer's issuer and serial number.
    SignerNotFound,
    /// The signature is good but no chain from the signer reaches a trust anchor.
    Untrusted,
}
