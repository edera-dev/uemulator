//! The cryptography the TPM device and the firmware's Secure Boot need, written for this
//! project: `no_std` with `alloc`, no dependencies, verification only (no private-key
//! operations, nothing secret to protect, so no constant-time requirements). Every parser takes
//! untrusted bytes (PE files, variables a guest writes) and returns `Error` rather than
//! panicking.
//!
//! - `sha`: `Sha1`, `Sha256`, `Sha384`, `Sha512` (`new`, `update`, `finish -> [u8; N]`), the
//!   one-shot `sha1`/`sha256`/`sha384`/`sha512`, and `HashAlgorithm` (`digest_len`, `oid`,
//!   `from_oid`, `digest -> Vec<u8>`) with the run-time `Hasher` over it.
//! - `der`: `Reader` over DER (`read`, `expect(tag)`, `optional(tag)`, `unsigned_integer`,
//!   `oid`, `bit_string`, `octet_string`, `algorithm`) yielding `Tlv { tag, contents, raw }`;
//!   the OIDs used here in `der::oid`.
//! - `rsa`: `RsaPublicKey::{from_subject_public_key_info, from_pkcs1, new}` and
//!   `verify_pkcs1v15(HashAlgorithm, digest, signature) -> Result<(), Error>`.
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

pub mod authenticode;
pub mod der;
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
