//! Secure Boot policy (UEFI 2.9 chapter 32), apart from where its state lives: signature
//! lists, the keys the firmware enrolls by default, whether db and dbx allow an image, and the
//! checks on an authenticated write to PK, KEK, db, dbx, dbt or dbr. EDK2's
//! `DxeImageVerificationLib` and `AuthVariableLib` are the behavioral reference.
//!
//! The defaults are a uemulator PK whose private key was discarded when it was made (nobody can
//! sign a KEK or PK update; the OS can still update db and dbx with Microsoft's KEK, which is
//! how dbx updates ship), Microsoft's two KEKs, Microsoft's five db certificates and the x64
//! dbx Microsoft published last. `secure-boot/README.md` says where each came from.

use alloc::vec::Vec;

use uemulator_crypto::authenticode;
use uemulator_crypto::pkcs7::{Content, MAX_CHAIN_LENGTH, SignedData};
use uemulator_crypto::sha::HashAlgorithm;
use uemulator_crypto::x509::Certificate;

use crate::uefi::{Guid, guids};
use crate::variable_store::VariableError;

pub mod signature_type {
    use crate::uefi::Guid;

    pub const SHA256: Guid = Guid::new(
        0xc1c41626,
        0x504c,
        0x4092,
        [0xac, 0xa9, 0x41, 0xf9, 0x36, 0x93, 0x43, 0x28],
    );
    pub const SHA384: Guid = Guid::new(
        0xff3e5307,
        0x9fd0,
        0x48c9,
        [0x85, 0xf1, 0x8a, 0xd5, 0x6c, 0x70, 0x1e, 0x01],
    );
    pub const SHA512: Guid = Guid::new(
        0x093e0fae,
        0xa6c4,
        0x4f50,
        [0x9f, 0x1b, 0xd4, 0x1e, 0x2b, 0x89, 0xc1, 0x9a],
    );
    pub const X509: Guid = Guid::new(
        0xa5c059a1,
        0x94e4,
        0x4aa7,
        [0x87, 0xb5, 0xab, 0x15, 0x5c, 0x2b, 0xf0, 0x72],
    );
    /// dbx entries revoking a certificate by the SHA-256 of its TBSCertificate, followed by a
    /// 16-byte revocation time that EDK2 also only checks against a timestamp countersignature.
    pub const X509_SHA256: Guid = Guid::new(
        0x3bd2a492,
        0x96c0,
        0x4079,
        [0xb4, 0x20, 0xfc, 0xf9, 0x8e, 0xf1, 0x03, 0xed],
    );
    pub const X509_SHA384: Guid = Guid::new(
        0x7076876e,
        0x80c2,
        0x4ee6,
        [0xaa, 0xd2, 0x28, 0xb3, 0x49, 0xa6, 0x86, 0x5b],
    );
    pub const X509_SHA512: Guid = Guid::new(
        0x446dbf63,
        0x2502,
        0x4cda,
        [0xbc, 0xfa, 0x24, 0x65, 0xd2, 0xb0, 0xfe, 0x9d],
    );
    /// `EFI_CERT_TYPE_PKCS7_GUID`: the `CertType` of an authenticated write's
    /// `WIN_CERTIFICATE_UEFI_GUID`.
    pub const PKCS7: Guid = Guid::new(
        0x4aafd29d,
        0x68df,
        0x49ee,
        [0x8a, 0xa9, 0x34, 0x7d, 0x37, 0x56, 0x65, 0xa7],
    );
}

/// `SignatureOwner` of Microsoft's entries, as on every PC.
pub const MICROSOFT_OWNER: Guid = Guid::new(
    0x77fa9abd,
    0x0359,
    0x4d32,
    [0xbd, 0x60, 0x28, 0xf4, 0xe7, 0x8f, 0x78, 0x4b],
);
/// `SignatureOwner` of the PK, made up for uemulator.
pub const UEMULATOR_OWNER: Guid = Guid::new(
    0xf4157d1f,
    0xdf64,
    0x482e,
    [0xa2, 0x5d, 0x10, 0xb2, 0x38, 0x5f, 0xb7, 0x35],
);

const PLATFORM_KEY: &[u8] = include_bytes!("../secure-boot/uemulator-pk.der");
const KEY_EXCHANGE_KEYS: [&[u8]; 2] = [
    include_bytes!("../secure-boot/microsoft-kek-ca-2011.der"),
    include_bytes!("../secure-boot/microsoft-kek-2k-ca-2023.der"),
];
const ALLOWED_CERTIFICATES: [&[u8]; 5] = [
    include_bytes!("../secure-boot/microsoft-windows-production-pca-2011.der"),
    include_bytes!("../secure-boot/microsoft-uefi-ca-2011.der"),
    include_bytes!("../secure-boot/windows-uefi-ca-2023.der"),
    include_bytes!("../secure-boot/microsoft-uefi-ca-2023.der"),
    include_bytes!("../secure-boot/microsoft-option-rom-uefi-ca-2023.der"),
];
/// Microsoft's update for dbx as signed with the 2023 KEK; the defaults keep its signature
/// lists and drop the authentication header.
const FORBIDDEN_UPDATE: &[u8] = include_bytes!("../secure-boot/dbx-x64-kek2023.auth2");

/// `EFI_SIGNATURE_LIST`: type GUID, list size, header size, signature size, then the header
/// (always empty in practice) and the signatures, each an owner GUID and the data.
const GUID_BYTES: usize = 16;
const LIST_HEADER_BYTES: usize = GUID_BYTES + 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignatureEntry<'a> {
    pub kind: Guid,
    pub owner: Guid,
    pub data: &'a [u8],
}

impl SignatureEntry<'_> {
    /// `EFI_SIGNATURE_DATA`: the owner, then the data, as the PFP spec's
    /// `EV_EFI_VARIABLE_AUTHORITY` event carries it.
    pub fn signature_data(&self) -> Vec<u8> {
        let mut out = self.owner.to_bytes().to_vec();
        out.extend_from_slice(self.data);
        out
    }
}

fn u32_at(bytes: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize)
}

/// Every signature in a variable's signature lists, or `None` if they are malformed.
pub fn entries(lists: &[u8]) -> Option<Vec<SignatureEntry<'_>>> {
    let mut found = Vec::new();
    let mut rest = lists;
    while !rest.is_empty() {
        let kind = Guid::from_bytes(rest.get(..GUID_BYTES)?);
        let list_size = u32_at(rest, GUID_BYTES)?;
        let header_size = u32_at(rest, GUID_BYTES + 4)?;
        let signature_size = u32_at(rest, GUID_BYTES + 8)?;
        let list = rest.get(..list_size)?;
        let body = list.get(LIST_HEADER_BYTES.checked_add(header_size)?..)?;
        if signature_size <= GUID_BYTES || body.len() % signature_size != 0 {
            return None;
        }
        for signature in body.chunks(signature_size) {
            found.push(SignatureEntry {
                kind,
                owner: Guid::from_bytes(&signature[..GUID_BYTES]),
                data: &signature[GUID_BYTES..],
            });
        }
        rest = &rest[list_size..];
    }
    Some(found)
}

/// One signature list holding `items`, which must all be the same length (a list has one
/// signature size).
pub fn signature_list(kind: Guid, owner: Guid, items: &[&[u8]]) -> Vec<u8> {
    let signature_size = GUID_BYTES + items.first().map_or(0, |item| item.len());
    debug_assert!(
        items
            .iter()
            .all(|item| GUID_BYTES + item.len() == signature_size)
    );
    let list_size = LIST_HEADER_BYTES + signature_size * items.len();
    let mut out = kind.to_bytes().to_vec();
    out.extend_from_slice(&(list_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(signature_size as u32).to_le_bytes());
    for item in items {
        out.extend_from_slice(&owner.to_bytes());
        out.extend_from_slice(item);
    }
    out
}

/// The lists `entries` came from, one list per run of entries with the same type and size.
fn lists_of(entries: &[SignatureEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < entries.len() {
        let first = entries[index];
        let run = entries[index..]
            .iter()
            .take_while(|entry| entry.kind == first.kind && entry.data.len() == first.data.len())
            .count();
        let signature_size = GUID_BYTES + first.data.len();
        out.extend_from_slice(&first.kind.to_bytes());
        out.extend_from_slice(&((LIST_HEADER_BYTES + signature_size * run) as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(signature_size as u32).to_le_bytes());
        for entry in &entries[index..index + run] {
            out.extend_from_slice(&entry.owner.to_bytes());
            out.extend_from_slice(entry.data);
        }
        index += run;
    }
    out
}

/// An `APPEND_WRITE` to a signature database: the new signatures that are not there yet
/// (UEFI 2.9 8.2.1: "the firmware shall not append duplicate signatures").
pub fn append(existing: &[u8], new: &[u8]) -> Option<Vec<u8>> {
    let mut merged = entries(existing)?;
    for entry in entries(new)? {
        if !merged
            .iter()
            .any(|old| old.kind == entry.kind && old.data == entry.data)
        {
            merged.push(entry);
        }
    }
    Some(lists_of(&merged))
}

/// The variables the firmware enrolls when Secure Boot is turned on and there is no PK.
pub struct Defaults {
    pub pk: Vec<u8>,
    pub kek: Vec<u8>,
    pub db: Vec<u8>,
    pub dbx: Vec<u8>,
}

pub fn defaults() -> Defaults {
    let x509_lists = |owner: Guid, certificates: &[&[u8]]| -> Vec<u8> {
        certificates
            .iter()
            .flat_map(|certificate| signature_list(signature_type::X509, owner, &[certificate]))
            .collect()
    };
    let dbx = parse_authentication2(FORBIDDEN_UPDATE)
        .map(|write| write.payload.to_vec())
        .unwrap_or_default();
    Defaults {
        pk: x509_lists(UEMULATOR_OWNER, &[PLATFORM_KEY]),
        kek: x509_lists(MICROSOFT_OWNER, &KEY_EXCHANGE_KEYS),
        db: x509_lists(MICROSOFT_OWNER, &ALLOWED_CERTIFICATES),
        dbx,
    }
}

/// What db and dbx say about an image.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict<'a> {
    /// Signed by a db certificate or its hash is in db; the entry that allowed it, for the
    /// `EV_EFI_VARIABLE_AUTHORITY` measurement.
    Allowed(SignatureEntry<'a>),
    /// dbx names its hash, its signer or a certificate in its chain.
    Forbidden,
    /// Not signed by anything db trusts and its hash is not in db (or the file is not a PE
    /// image that can be hashed).
    NotAllowed,
}

fn certificates_of<'a>(entries: &[SignatureEntry<'a>]) -> Vec<(&'a [u8], SignatureEntry<'a>)> {
    entries
        .iter()
        .filter(|entry| entry.kind == signature_type::X509)
        .map(|entry| (entry.data, *entry))
        .collect()
}

/// dbx's way of revoking a certificate without listing it: the TBS hash, with the revocation
/// time after it.
fn revoked_by_hash(certificate: &Certificate, forbidden: &[SignatureEntry]) -> bool {
    forbidden.iter().any(|entry| {
        let algorithm = match entry.kind {
            kind if kind == signature_type::X509_SHA256 => HashAlgorithm::Sha256,
            kind if kind == signature_type::X509_SHA384 => HashAlgorithm::Sha384,
            kind if kind == signature_type::X509_SHA512 => HashAlgorithm::Sha512,
            _ => return false,
        };
        let digest = certificate.tbs_digest(algorithm);
        entry.data.get(..digest.len()) == Some(digest.as_slice())
    })
}

fn signature_revoked<'a>(
    signed: &SignedData<'a>,
    forbidden_anchors: &[&'a [u8]],
    forbidden: &[SignatureEntry],
) -> bool {
    const MAX_REVOCATION_PATHS: usize = 64;
    const MAX_ISSUER_CHECKS: usize = 256;
    if forbidden_anchors.is_empty()
        && !forbidden.iter().any(|entry| {
            entry.kind == signature_type::X509_SHA256
                || entry.kind == signature_type::X509_SHA384
                || entry.kind == signature_type::X509_SHA512
        })
    {
        return false;
    }
    if signed.verify(Content::Embedded, forbidden_anchors).is_ok() {
        return true;
    }
    let revoked_certificates: Vec<Certificate<'a>> = forbidden_anchors
        .iter()
        .filter_map(|der| Certificate::parse(der).ok())
        .collect();
    let mut issuer_checks = 0;
    for index in 0..signed.signers.len() {
        let Ok(signer) = signed.verify_signer(index, Content::Embedded, &revoked_certificates)
        else {
            continue;
        };
        let mut paths = alloc::vec![alloc::vec![signer]];
        let mut checked = 0;
        while let Some(chain) = paths.pop() {
            checked += 1;
            if checked > MAX_REVOCATION_PATHS {
                // A hostile certificate bag must not make validation unbounded. A chain
                // we did not inspect might be revoked, so fail closed.
                return true;
            }
            let current = &chain[chain.len() - 1];
            if revoked_by_hash(current, forbidden)
                || revoked_certificates.iter().any(|certificate| {
                    certificate.raw == current.raw || certificate.issued(current)
                })
            {
                return true;
            }
            if chain.len() == MAX_CHAIN_LENGTH {
                continue;
            }
            // The bag may carry unrelated and cross-signed certificates. Visit every
            // issuer path of a verified signer, never an unattached certificate.
            for issuer in &signed.certificates {
                issuer_checks += 1;
                if issuer_checks > MAX_ISSUER_CHECKS {
                    return true;
                }
                if !chain.iter().any(|link| link.raw == issuer.raw) && issuer.issued(current) {
                    let mut path = chain.clone();
                    path.push(issuer.clone());
                    if paths.len() == MAX_REVOCATION_PATHS {
                        return true;
                    }
                    paths.push(path);
                }
            }
        }
    }
    false
}

fn image_hash_kind(algorithm: HashAlgorithm) -> Option<Guid> {
    match algorithm {
        HashAlgorithm::Sha256 => Some(signature_type::SHA256),
        HashAlgorithm::Sha384 => Some(signature_type::SHA384),
        HashAlgorithm::Sha512 => Some(signature_type::SHA512),
        HashAlgorithm::Sha1 => None,
    }
}

fn find_hash<'a>(
    file: &[u8],
    entries: &[SignatureEntry<'a>],
    digests: &mut Vec<(HashAlgorithm, Vec<u8>)>,
) -> Option<SignatureEntry<'a>> {
    for algorithm in [
        HashAlgorithm::Sha256,
        HashAlgorithm::Sha384,
        HashAlgorithm::Sha512,
    ] {
        let kind = image_hash_kind(algorithm)?;
        if !entries.iter().any(|entry| entry.kind == kind) {
            continue;
        }
        let digest = match digests.iter().find(|(known, _)| *known == algorithm) {
            Some((_, digest)) => digest.clone(),
            None => {
                let digest = authenticode::digest(file, algorithm).ok()?;
                digests.push((algorithm, digest.clone()));
                digest
            }
        };
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.kind == kind && entry.data == digest.as_slice())
        {
            return Some(*entry);
        }
    }
    None
}

/// Whether db and dbx let `file` run (UEFI 2.9 32.5.3.3, as EDK2's `DxeImageVerificationLib`
/// decides it): anything dbx names is refused; otherwise a signature that chains to a db
/// certificate, or the image's hash in db, allows it. A signature whose digest does not match
/// the image counts as no signature.
pub fn verify_image<'a>(file: &[u8], db: &'a [u8], dbx: &'a [u8]) -> Verdict<'a> {
    let allowed = entries(db).unwrap_or_default();
    let forbidden = entries(dbx).unwrap_or_default();
    let mut digests = Vec::new();
    if find_hash(file, &forbidden, &mut digests).is_some() {
        return Verdict::Forbidden;
    }
    let allowed_certificates = certificates_of(&allowed);
    let allowed_anchors: Vec<&[u8]> = allowed_certificates.iter().map(|(der, _)| *der).collect();
    let forbidden_anchors: Vec<&[u8]> = certificates_of(&forbidden)
        .iter()
        .map(|(der, _)| *der)
        .collect();
    // A malformed later WIN_CERTIFICATE must not erase an earlier revocation and let a
    // db hash authorize the image. Treat an unreadable certificate table as invalid.
    let Ok(signatures) = authenticode::signatures(file) else {
        return Verdict::NotAllowed;
    };
    let mut verdict = None;
    for blob in signatures {
        let Ok(signed) = SignedData::parse(blob) else {
            continue;
        };
        if authenticode::check_digest(file, &signed).is_err() {
            continue;
        }
        if signature_revoked(&signed, &forbidden_anchors, &forbidden) {
            return Verdict::Forbidden;
        }
        if verdict.is_none()
            && let Ok(verified) = signed.verify(Content::Embedded, &allowed_anchors)
        {
            if verified
                .certificates()
                .any(|certificate| revoked_by_hash(certificate, &forbidden))
            {
                return Verdict::Forbidden;
            }
            verdict = Some(allowed_certificates[verified.anchor()].1);
        }
    }
    match verdict.or_else(|| find_hash(file, &allowed, &mut digests)) {
        Some(entry) => Verdict::Allowed(entry),
        None => Verdict::NotAllowed,
    }
}

/// The variables whose writes are authenticated against PK or KEK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecureVariable {
    Pk,
    Kek,
    /// db, dbx, dbt and dbr: signed by a KEK or the PK.
    Database,
}

pub fn classify(name: &str, guid: &Guid) -> Option<SecureVariable> {
    match name {
        "PK" if *guid == guids::GLOBAL_VARIABLE => Some(SecureVariable::Pk),
        "KEK" if *guid == guids::GLOBAL_VARIABLE => Some(SecureVariable::Kek),
        "db" | "dbx" | "dbt" | "dbr" if *guid == guids::IMAGE_SECURITY_DATABASE => {
            Some(SecureVariable::Database)
        }
        _ => None,
    }
}

/// `EFI_TIME`, compared by its calendar fields only.
pub const TIMESTAMP_BYTES: usize = 16;
const TIMESTAMP_PAD1: usize = 7;
const TIMESTAMP_NANOSECOND: usize = 8;
/// TimeZone (2 bytes), Daylight and Pad2.
const TIMESTAMP_ZONE_AND_PADDING: core::ops::Range<usize> = 12..16;
/// Year (u16), month, day, hour, minute, second.
const TIMESTAMP_CALENDAR: core::ops::Range<usize> = 0..7;

pub type Timestamp = [u8; TIMESTAMP_BYTES];

fn calendar(timestamp: &Timestamp) -> [u16; 6] {
    let year = u16::from_le_bytes([timestamp[0], timestamp[1]]);
    let [month, day, hour, minute, second] = [2, 3, 4, 5, 6].map(|at| timestamp[at] as u16);
    debug_assert_eq!(TIMESTAMP_CALENDAR.end, 7);
    [year, month, day, hour, minute, second]
}

pub fn later(a: &Timestamp, b: &Timestamp) -> bool {
    calendar(a) > calendar(b)
}

/// `EFI_VARIABLE_AUTHENTICATION_2` taken apart: the timestamp, the PKCS#7 `SignedData` and the
/// variable data that follows them.
pub struct Authentication2<'a> {
    pub timestamp: Timestamp,
    pub signed_data: &'a [u8],
    pub payload: &'a [u8],
}

/// `WIN_CERTIFICATE_UEFI_GUID`: dwLength, wRevision, wCertificateType, CertType.
const WIN_CERTIFICATE_HEADER_BYTES: usize = 8 + GUID_BYTES;
const WIN_CERT_REVISION: u16 = 0x0200;
const WIN_CERT_TYPE_EFI_GUID: u16 = 0x0ef1;

pub fn parse_authentication2(data: &[u8]) -> Result<Authentication2<'_>, VariableError> {
    let malformed = VariableError::SecurityViolation;
    let timestamp: Timestamp = data
        .get(..TIMESTAMP_BYTES)
        .ok_or(malformed)?
        .try_into()
        .unwrap();
    // UEFI 2.9 8.2.2: everything but the calendar fields must be zero.
    if timestamp[TIMESTAMP_PAD1] != 0
        || timestamp[TIMESTAMP_NANOSECOND..TIMESTAMP_NANOSECOND + 4] != [0; 4]
        || timestamp[TIMESTAMP_ZONE_AND_PADDING] != [0; 4]
    {
        return Err(malformed);
    }
    let certificate = &data[TIMESTAMP_BYTES..];
    let length = u32_at(certificate, 0).ok_or(malformed)?;
    let header = certificate
        .get(..WIN_CERTIFICATE_HEADER_BYTES)
        .ok_or(malformed)?;
    if u16::from_le_bytes([header[4], header[5]]) != WIN_CERT_REVISION
        || u16::from_le_bytes([header[6], header[7]]) != WIN_CERT_TYPE_EFI_GUID
        || Guid::from_bytes(&header[8..]) != signature_type::PKCS7
        || length < WIN_CERTIFICATE_HEADER_BYTES
    {
        return Err(malformed);
    }
    let signed_data = certificate
        .get(WIN_CERTIFICATE_HEADER_BYTES..length)
        .ok_or(malformed)?;
    Ok(Authentication2 {
        timestamp,
        signed_data,
        payload: &certificate[length..],
    })
}

/// What the PKCS#7 signature of a time-based authenticated write covers (UEFI 2.9 8.2.2): the
/// name without its NUL, the vendor GUID, the attributes exactly as `SetVariable` got them
/// (`APPEND_WRITE` included: Microsoft's dbx updates verify only so), the timestamp and the
/// data.
pub fn signed_message(
    name: &[u16],
    guid: &Guid,
    attributes: u32,
    timestamp: &Timestamp,
    payload: &[u8],
) -> Vec<u8> {
    let mut message = Vec::with_capacity(name.len() * 2 + 36 + payload.len());
    for unit in name {
        message.extend_from_slice(&unit.to_le_bytes());
    }
    message.extend_from_slice(&guid.to_bytes());
    message.extend_from_slice(&attributes.to_le_bytes());
    message.extend_from_slice(timestamp);
    message.extend_from_slice(payload);
    message
}

/// Whether `signed_data` is a valid signature over `message` by one of `anchors` (DER
/// certificates) or a certificate they issued.
pub fn signed_by(signed_data: &[u8], message: &[u8], anchors: &[&[u8]]) -> bool {
    SignedData::parse(signed_data)
        .and_then(|signed| {
            signed
                .verify(Content::Detached(message), anchors)
                .map(|_| ())
        })
        .is_ok()
}

/// The X.509 certificates in a PK or KEK value.
pub fn certificates(lists: &[u8]) -> Vec<&[u8]> {
    entries(lists)
        .unwrap_or_default()
        .into_iter()
        .filter(|entry| entry.kind == signature_type::X509)
        .map(|entry| entry.data)
        .collect()
}

/// A PK value is one X.509 certificate (UEFI 2.9 32.3.1); anything else would leave the
/// machine with a PK nothing can verify against.
pub fn valid_platform_key(lists: &[u8]) -> bool {
    matches!(entries(lists).as_deref(), Some([entry]) if entry.kind == signature_type::X509
        && Certificate::parse(entry.data).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uefi::variable_attribute::*;

    const SIGNED_IMAGE: &[u8] = include_bytes!("../../uemulator-crypto/src/testdata/signed.efi");
    const UNSIGNED_IMAGE: &[u8] =
        include_bytes!("../../uemulator-crypto/src/testdata/unsigned.efi");
    const TEST_CA: &[u8] = include_bytes!("../../uemulator-crypto/src/testdata/test-ca.der");
    const TEST_LEAF: &[u8] = include_bytes!("../../uemulator-crypto/src/testdata/test-leaf.der");
    const DBX_KEK2011: &[u8] = include_bytes!("../secure-boot/dbx-x64-kek2011.auth2");
    const DBX_APPEND_ATTRIBUTES: u32 = NON_VOLATILE
        | BOOTSERVICE_ACCESS
        | RUNTIME_ACCESS
        | TIME_BASED_AUTHENTICATED_WRITE_ACCESS
        | APPEND_WRITE;

    fn x509_list(certificate: &[u8]) -> Vec<u8> {
        signature_list(signature_type::X509, MICROSOFT_OWNER, &[certificate])
    }

    fn sha256_list(digest: &[u8]) -> Vec<u8> {
        signature_list(signature_type::SHA256, MICROSOFT_OWNER, &[digest])
    }

    #[test]
    fn db_hash_cannot_authorize_unhashed_section_addresses() {
        // Before header validation, this digest authorized both the original and a file
        // whose section RVAs were swapped outside the hashed SizeOfHeaders.
        const OLD_DIGEST: [u8; 32] = [
            0xcd, 0x96, 0x03, 0x30, 0xa7, 0xa4, 0xfb, 0x47, 0x27, 0x8d, 0x21, 0xa6, 0xe9, 0xc3,
            0xf2, 0x54, 0x5a, 0x61, 0x31, 0xd3, 0x7a, 0x9f, 0x14, 0x1b, 0x87, 0x4d, 0x22, 0x56,
            0xdb, 0xcc, 0x27, 0xc8,
        ];
        let mut file = UNSIGNED_IMAGE.to_vec();
        const PE: usize = 0x80;
        const OPTIONAL: usize = PE + 4 + 20;
        const SECTIONS: usize = OPTIONAL + 240;
        file[OPTIONAL + 60..OPTIONAL + 64].copy_from_slice(&0x180u32.to_le_bytes());
        file[SECTIONS + 12..SECTIONS + 16].copy_from_slice(&0x2000u32.to_le_bytes());
        file[SECTIONS + 40 + 12..SECTIONS + 40 + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        assert_eq!(
            verify_image(&file, &sha256_list(&OLD_DIGEST), &[]),
            Verdict::NotAllowed
        );
    }

    fn image_sha256(file: &[u8]) -> Vec<u8> {
        authenticode::digest(file, HashAlgorithm::Sha256).unwrap()
    }

    #[test]
    fn the_defaults_are_well_formed_and_hold_the_microsoft_keys() {
        let defaults = defaults();
        assert!(valid_platform_key(&defaults.pk));
        assert_eq!(certificates(&defaults.kek).len(), 2);
        assert_eq!(certificates(&defaults.db).len(), 5);
        for der in certificates(&defaults.kek)
            .into_iter()
            .chain(certificates(&defaults.db))
        {
            Certificate::parse(der).unwrap();
        }
        let dbx = entries(&defaults.dbx).unwrap();
        assert!(dbx.len() > 200, "{} dbx entries", dbx.len());
        assert!(
            dbx.iter()
                .all(|entry| entry.kind == signature_type::SHA256 && entry.data.len() == 32)
        );
    }

    #[test]
    fn signature_lists_round_trip_and_reject_bad_sizes() {
        let list = signature_list(
            signature_type::SHA256,
            UEMULATOR_OWNER,
            &[&[1; 32], &[2; 32]],
        );
        let parsed = entries(&list).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].data, [2; 32]);
        assert_eq!(parsed[0].owner, UEMULATOR_OWNER);
        assert_eq!(lists_of(&parsed), list);
        let mut truncated = list.clone();
        truncated.pop();
        assert!(entries(&truncated).is_none());
        let mut bad_size = list;
        bad_size[GUID_BYTES + 8] = 47;
        assert!(entries(&bad_size).is_none());
        assert_eq!(entries(&[]).unwrap(), []);
    }

    #[test]
    fn appending_skips_signatures_already_there() {
        let existing = sha256_list(&[1; 32]);
        let merged = append(
            &existing,
            &[sha256_list(&[1; 32]), sha256_list(&[2; 32])].concat(),
        )
        .unwrap();
        let merged = entries(&merged).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[1].data, [2; 32]);
    }

    #[test]
    fn a_signature_chaining_to_db_allows_the_image() {
        let db = x509_list(TEST_CA);
        match verify_image(SIGNED_IMAGE, &db, &[]) {
            Verdict::Allowed(entry) => assert_eq!(entry.data, TEST_CA),
            other => panic!("{other:?}"),
        }
        let leaf_in_db = x509_list(TEST_LEAF);
        assert!(matches!(
            verify_image(SIGNED_IMAGE, &leaf_in_db, &[]),
            Verdict::Allowed(_)
        ));
    }

    #[test]
    fn the_microsoft_db_does_not_allow_a_test_signed_image() {
        let defaults = defaults();
        assert_eq!(
            verify_image(SIGNED_IMAGE, &defaults.db, &defaults.dbx),
            Verdict::NotAllowed
        );
        assert_eq!(
            verify_image(UNSIGNED_IMAGE, &defaults.db, &defaults.dbx),
            Verdict::NotAllowed
        );
    }

    #[test]
    fn an_image_hash_in_db_allows_it_and_in_dbx_forbids_it() {
        let digest = image_sha256(UNSIGNED_IMAGE);
        let db = sha256_list(&digest);
        assert!(matches!(
            verify_image(UNSIGNED_IMAGE, &db, &[]),
            Verdict::Allowed(entry) if entry.kind == signature_type::SHA256
        ));
        assert_eq!(verify_image(UNSIGNED_IMAGE, &db, &db), Verdict::Forbidden);
        let signed_digest = image_sha256(SIGNED_IMAGE);
        assert_eq!(
            verify_image(
                SIGNED_IMAGE,
                &x509_list(TEST_CA),
                &sha256_list(&signed_digest)
            ),
            Verdict::Forbidden
        );
    }

    #[test]
    fn dbx_forbids_by_certificate_and_by_certificate_hash() {
        let db = x509_list(TEST_CA);
        assert_eq!(
            verify_image(SIGNED_IMAGE, &db, &x509_list(TEST_LEAF)),
            Verdict::Forbidden
        );
        let tbs_hash = Certificate::parse(TEST_LEAF)
            .unwrap()
            .tbs_digest(HashAlgorithm::Sha256);
        let revocation = [tbs_hash.as_slice(), &[0; 16]].concat();
        let dbx = signature_list(signature_type::X509_SHA256, MICROSOFT_OWNER, &[&revocation]);
        assert_eq!(verify_image(SIGNED_IMAGE, &db, &dbx), Verdict::Forbidden);
    }

    #[test]
    fn unrelated_certificate_in_signature_bag_does_not_revoke_signer() {
        let blob = authenticode::signatures(SIGNED_IMAGE).unwrap()[0];
        let mut signed = SignedData::parse(blob).unwrap();
        let unrelated =
            Certificate::parse(include_bytes!("../secure-boot/uemulator-pk.der")).unwrap();
        let mut revocation = unrelated.tbs_digest(HashAlgorithm::Sha256);
        revocation.extend_from_slice(&[0; 16]);
        let dbx = signature_list(signature_type::X509_SHA256, MICROSOFT_OWNER, &[&revocation]);
        let forbidden = entries(&dbx).unwrap();
        signed.certificates.push(unrelated);
        assert!(signed.verify(Content::Embedded, &[TEST_CA]).is_ok());
        assert!(!signature_revoked(&signed, &[], &forbidden));
    }

    #[test]
    fn dbx_checks_all_issuer_paths_in_a_signature_bag() {
        let blob = authenticode::signatures(SIGNED_IMAGE).unwrap()[0];
        let mut signed = SignedData::parse(blob).unwrap();
        let original = Certificate::parse(TEST_CA).unwrap();
        // The test CA's serial starts at byte 15. Changing it keeps its subject and
        // public key, so both certificates can issue the test signer.
        let mut alternate_der = TEST_CA.to_vec();
        alternate_der[15] ^= 1;
        let alternate = Certificate::parse(&alternate_der).unwrap();
        let leaf = Certificate::parse(TEST_LEAF).unwrap();
        assert!(alternate.issued(&leaf));
        assert!(original.issued(&leaf));
        signed.certificates = alloc::vec![leaf, alternate, original.clone()];
        let mut revocation = original.tbs_digest(HashAlgorithm::Sha256);
        revocation.extend_from_slice(&[0; 16]);
        let dbx = signature_list(signature_type::X509_SHA256, MICROSOFT_OWNER, &[&revocation]);
        let forbidden = entries(&dbx).unwrap();
        assert!(signature_revoked(&signed, &[], &forbidden));
    }

    #[test]
    fn excessive_unrelated_certificates_fail_closed() {
        let blob = authenticode::signatures(SIGNED_IMAGE).unwrap()[0];
        let mut signed = SignedData::parse(blob).unwrap();
        let unrelated =
            Certificate::parse(include_bytes!("../secure-boot/uemulator-pk.der")).unwrap();
        signed
            .certificates
            .extend(core::iter::repeat_n(unrelated, 300));
        let other = Certificate::parse(include_bytes!(
            "../secure-boot/microsoft-kek-2k-ca-2023.der"
        ))
        .unwrap();
        let mut revocation = other.tbs_digest(HashAlgorithm::Sha256);
        revocation.extend_from_slice(&[0; 16]);
        let dbx = signature_list(signature_type::X509_SHA256, MICROSOFT_OWNER, &[&revocation]);
        let forbidden = entries(&dbx).unwrap();
        assert!(signature_revoked(&signed, &[], &forbidden));
    }

    #[test]
    fn malformed_later_certificate_cannot_hide_a_dbx_revocation() {
        let digest = image_sha256(SIGNED_IMAGE);
        let db = sha256_list(&digest);
        let dbx = x509_list(TEST_LEAF);
        assert_eq!(verify_image(SIGNED_IMAGE, &db, &dbx), Verdict::Forbidden);

        let mut image = SIGNED_IMAGE.to_vec();
        let pe = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
        let certificate_directory = pe + 24 + 112 + 4 * 8;
        let size = u32::from_le_bytes(
            image[certificate_directory + 4..certificate_directory + 8]
                .try_into()
                .unwrap(),
        );
        image.extend_from_slice(&[0; 8]);
        image[certificate_directory + 4..certificate_directory + 8]
            .copy_from_slice(&(size + 8).to_le_bytes());
        assert_eq!(image_sha256(&image), digest);
        assert!(authenticode::signatures(&image).is_err());
        assert_eq!(verify_image(&image, &db, &dbx), Verdict::NotAllowed);
    }

    #[test]
    fn a_changed_image_is_not_allowed_by_its_old_signature() {
        let mut tampered = SIGNED_IMAGE.to_vec();
        let middle = tampered.len() / 3;
        tampered[middle] ^= 1;
        assert_eq!(
            verify_image(&tampered, &x509_list(TEST_CA), &[]),
            Verdict::NotAllowed
        );
    }

    #[test]
    fn microsofts_dbx_update_verifies_against_its_kek_only() {
        let write = parse_authentication2(DBX_KEK2011).unwrap();
        let name: Vec<u16> = "dbx".encode_utf16().collect();
        let message = signed_message(
            &name,
            &guids::IMAGE_SECURITY_DATABASE,
            DBX_APPEND_ATTRIBUTES,
            &write.timestamp,
            write.payload,
        );
        let defaults = defaults();
        let kek = certificates(&defaults.kek);
        assert!(signed_by(write.signed_data, &message, &kek));
        assert!(!signed_by(
            write.signed_data,
            &message,
            &certificates(&defaults.pk)
        ));
        let without_append = signed_message(
            &name,
            &guids::IMAGE_SECURITY_DATABASE,
            DBX_APPEND_ATTRIBUTES & !APPEND_WRITE,
            &write.timestamp,
            write.payload,
        );
        assert!(!signed_by(write.signed_data, &without_append, &kek));
        assert!(entries(write.payload).is_some());
    }

    #[test]
    fn authentication_headers_are_checked() {
        let mut nonzero_nanoseconds = DBX_KEK2011.to_vec();
        nonzero_nanoseconds[TIMESTAMP_NANOSECOND] = 1;
        assert!(parse_authentication2(&nonzero_nanoseconds).is_err());
        let mut wrong_type = DBX_KEK2011.to_vec();
        wrong_type[TIMESTAMP_BYTES + 6] = 0;
        assert!(parse_authentication2(&wrong_type).is_err());
        for length in [0, 10, TIMESTAMP_BYTES + 4, TIMESTAMP_BYTES + 30] {
            assert!(parse_authentication2(&DBX_KEK2011[..length]).is_err());
        }
    }

    #[test]
    fn timestamps_compare_by_calendar_fields() {
        let at = |year: u16, month: u8, second: u8| {
            let mut timestamp = [0u8; TIMESTAMP_BYTES];
            timestamp[..2].copy_from_slice(&year.to_le_bytes());
            timestamp[2] = month;
            timestamp[6] = second;
            timestamp
        };
        assert!(later(&at(2024, 1, 0), &at(2023, 12, 59)));
        assert!(later(&at(2024, 1, 1), &at(2024, 1, 0)));
        assert!(!later(&at(2024, 1, 0), &at(2024, 1, 0)));
    }

    #[test]
    fn classification_is_by_name_and_vendor() {
        assert_eq!(
            classify("PK", &guids::GLOBAL_VARIABLE),
            Some(SecureVariable::Pk)
        );
        assert_eq!(
            classify("dbx", &guids::IMAGE_SECURITY_DATABASE),
            Some(SecureVariable::Database)
        );
        assert_eq!(classify("db", &guids::GLOBAL_VARIABLE), None);
        assert_eq!(classify("PK", &guids::IMAGE_SECURITY_DATABASE), None);
    }
}
