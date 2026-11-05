//! Authenticode for PE32/PE32+ images ("Windows Authenticode Portable Executable Signature
//! Format" and the PE format's "Attribute Certificate Table"): the image digest, the
//! signatures in the certificate table, and the digest a signature vouches for.
//!
//! The digest follows EDK2's `DxeImageVerificationLib` (`HashPeImage`) byte for byte, which is
//! what every UEFI signer and verifier agrees on, including its treatment of trailing data.

use alloc::vec::Vec;

use crate::Error;
use crate::der::{Reader, TAG_OCTET_STRING, TAG_SEQUENCE, oid};
use crate::pkcs7::SignedData;
use crate::sha::{HashAlgorithm, Hasher};

const DOS_MAGIC: &[u8; 2] = b"MZ";
const DOS_PE_OFFSET_FIELD: usize = 0x3c;
const PE_SIGNATURE: &[u8; 4] = b"PE\0\0";
const COFF_HEADER_BYTES: usize = 20;
const COFF_NUMBER_OF_SECTIONS: usize = 2;
const COFF_SIZE_OF_OPTIONAL_HEADER: usize = 16;
const PE32_MAGIC: u16 = 0x10b;
const PE32_PLUS_MAGIC: u16 = 0x20b;
/// Offsets inside the optional header, the same for PE32 and PE32+.
const OPTIONAL_SIZE_OF_HEADERS: usize = 60;
const OPTIONAL_CHECKSUM: usize = 64;
/// NumberOfRvaAndSizes, then the data directories, differ between the two.
const PE32_NUMBER_OF_RVA_AND_SIZES: usize = 92;
const PE32_PLUS_NUMBER_OF_RVA_AND_SIZES: usize = 108;
const DATA_DIRECTORY_BYTES: usize = 8;
/// IMAGE_DIRECTORY_ENTRY_SECURITY: its "virtual address" is a file offset.
const CERTIFICATE_TABLE_DIRECTORY: usize = 4;
const SECTION_HEADER_BYTES: usize = 40;
const SECTION_SIZE_OF_RAW_DATA: usize = 16;
const SECTION_POINTER_TO_RAW_DATA: usize = 20;

/// WIN_CERTIFICATE: dwLength u32, wRevision u16, wCertificateType u16.
const WIN_CERTIFICATE_HEADER_BYTES: usize = 8;
const WIN_CERT_REVISION_2_0: u16 = 0x0200;
const WIN_CERT_TYPE_PKCS_SIGNED_DATA: u16 = 0x0002;
/// WIN_CERTIFICATE_UEFI_GUID, which EDK2 also accepts in images when its CertType is
/// EFI_CERT_TYPE_PKCS7_GUID.
const WIN_CERT_TYPE_EFI_GUID: u16 = 0x0ef1;
const GUID_BYTES: usize = 16;
/// 4aafd29d-68df-49ee-8aa9-347d375665a7 in the EFI byte order.
pub const EFI_CERT_TYPE_PKCS7_GUID: [u8; GUID_BYTES] = [
    0x9d, 0xd2, 0xaf, 0x4a, 0xdf, 0x68, 0xee, 0x49, 0x8a, 0xa9, 0x34, 0x7d, 0x37, 0x56, 0x65, 0xa7,
];
/// Entries in the certificate table start on 8-byte boundaries.
const CERTIFICATE_ALIGNMENT: usize = 8;

fn u16_at(data: &[u8], offset: usize) -> Result<u16, Error> {
    let bytes = data.get(offset..offset + 2).ok_or(Error::Malformed)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, Error> {
    let bytes = data.get(offset..offset + 4).ok_or(Error::Malformed)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// What hashing and signature lookup need from the headers, all as file offsets.
struct Layout {
    checksum: usize,
    /// The certificate table's data directory entry, when the image has that many entries.
    certificate_directory: Option<usize>,
    size_of_headers: usize,
    /// (offset, size) of the attribute certificate table; size 0 when unsigned.
    certificate_table: (usize, usize),
    /// (PointerToRawData, SizeOfRawData) of every section with raw data, sorted by offset.
    sections: Vec<(usize, usize)>,
}

impl Layout {
    fn parse(file: &[u8]) -> Result<Self, Error> {
        if file.get(..2) != Some(DOS_MAGIC) {
            return Err(Error::Malformed);
        }
        let pe = u32_at(file, DOS_PE_OFFSET_FIELD)? as usize;
        if file.get(pe..pe + 4) != Some(PE_SIGNATURE) {
            return Err(Error::Malformed);
        }
        let coff = pe + 4;
        let section_count = u16_at(file, coff + COFF_NUMBER_OF_SECTIONS)? as usize;
        let optional_size = u16_at(file, coff + COFF_SIZE_OF_OPTIONAL_HEADER)? as usize;
        let optional = coff + COFF_HEADER_BYTES;
        let rva_count_field = match u16_at(file, optional)? {
            PE32_MAGIC => PE32_NUMBER_OF_RVA_AND_SIZES,
            PE32_PLUS_MAGIC => PE32_PLUS_NUMBER_OF_RVA_AND_SIZES,
            _ => return Err(Error::Unsupported),
        };
        let size_of_headers = u32_at(file, optional + OPTIONAL_SIZE_OF_HEADERS)? as usize;
        if size_of_headers > file.len() {
            return Err(Error::Malformed);
        }
        let checksum = optional + OPTIONAL_CHECKSUM;
        let directory_count = u32_at(file, optional + rva_count_field)? as usize;
        let directories = optional + rva_count_field + 4;
        let certificate_directory = (directory_count > CERTIFICATE_TABLE_DIRECTORY)
            .then_some(directories + CERTIFICATE_TABLE_DIRECTORY * DATA_DIRECTORY_BYTES);
        let mut certificate_table = (0, 0);
        if let Some(entry) = certificate_directory {
            // The skipped fields must lie in the hashed header, past what came before them.
            if entry + DATA_DIRECTORY_BYTES > size_of_headers
                || entry + DATA_DIRECTORY_BYTES > optional + optional_size
            {
                return Err(Error::Malformed);
            }
            let offset = u32_at(file, entry)? as usize;
            let size = u32_at(file, entry + 4)? as usize;
            if size > 0 && offset.checked_add(size).is_none_or(|end| end > file.len()) {
                return Err(Error::Malformed);
            }
            certificate_table = (offset, size);
        }
        if checksum + 4 > size_of_headers {
            return Err(Error::Malformed);
        }
        let section_table = optional + optional_size;
        let mut sections = Vec::new();
        for index in 0..section_count {
            let header = section_table + index * SECTION_HEADER_BYTES;
            let size = u32_at(file, header + SECTION_SIZE_OF_RAW_DATA)? as usize;
            let offset = u32_at(file, header + SECTION_POINTER_TO_RAW_DATA)? as usize;
            if size == 0 {
                continue;
            }
            if offset.checked_add(size).is_none_or(|end| end > file.len()) {
                return Err(Error::Malformed);
            }
            sections.push((offset, size));
        }
        sections.sort_unstable_by_key(|&(offset, _)| offset);
        Ok(Self {
            checksum,
            certificate_directory,
            size_of_headers,
            certificate_table,
            sections,
        })
    }
}

/// The Authenticode digest of a PE image file.
pub fn digest(file: &[u8], algorithm: HashAlgorithm) -> Result<Vec<u8>, Error> {
    let layout = Layout::parse(file)?;
    let mut hasher = Hasher::new(algorithm);
    // Headers up to SizeOfHeaders, leaving out CheckSum and the certificate table's entry.
    hasher.update(&file[..layout.checksum]);
    let after_checksum = layout.checksum + 4;
    match layout.certificate_directory {
        Some(entry) => {
            hasher.update(&file[after_checksum..entry]);
            hasher.update(&file[entry + DATA_DIRECTORY_BYTES..layout.size_of_headers]);
        }
        None => hasher.update(&file[after_checksum..layout.size_of_headers]),
    }
    let mut hashed = layout.size_of_headers;
    for &(offset, size) in &layout.sections {
        hasher.update(&file[offset..offset + size]);
        hashed += size;
    }
    // Whatever follows, except the certificate table that is assumed to end the file. Like
    // EDK2 this starts at the byte count hashed so far, not where the last section ends.
    if file.len() > hashed {
        let certificate_size = layout.certificate_table.1;
        let trailing_end = file
            .len()
            .checked_sub(certificate_size)
            .ok_or(Error::Malformed)?;
        if trailing_end < hashed {
            return Err(Error::Malformed);
        }
        hasher.update(&file[hashed..trailing_end]);
    }
    Ok(hasher.finish())
}

/// The PKCS#7 SignedData blobs in the image's attribute certificate table, in order.
pub fn signatures(file: &[u8]) -> Result<Vec<&[u8]>, Error> {
    let layout = Layout::parse(file)?;
    let (start, size) = layout.certificate_table;
    let mut found = Vec::new();
    if size == 0 {
        return Ok(found);
    }
    let table = &file[start..start + size];
    let mut at = 0;
    while at + WIN_CERTIFICATE_HEADER_BYTES <= table.len() {
        let length = u32_at(table, at)? as usize;
        let revision = u16_at(table, at + 4)?;
        let kind = u16_at(table, at + 6)?;
        if length < WIN_CERTIFICATE_HEADER_BYTES || at + length > table.len() {
            return Err(Error::Malformed);
        }
        let body = &table[at + WIN_CERTIFICATE_HEADER_BYTES..at + length];
        match (revision, kind) {
            (WIN_CERT_REVISION_2_0, WIN_CERT_TYPE_PKCS_SIGNED_DATA) => found.push(body),
            (WIN_CERT_REVISION_2_0, WIN_CERT_TYPE_EFI_GUID)
                if body.get(..GUID_BYTES) == Some(&EFI_CERT_TYPE_PKCS7_GUID[..]) =>
            {
                found.push(&body[GUID_BYTES..])
            }
            _ => {}
        }
        at += length.next_multiple_of(CERTIFICATE_ALIGNMENT);
    }
    Ok(found)
}

/// The image digest a signature vouches for, from its embedded SpcIndirectDataContent:
/// SEQUENCE { SpcAttributeTypeAndOptionalValue, DigestInfo }.
pub fn signed_digest<'a>(signed: &SignedData<'a>) -> Result<(HashAlgorithm, &'a [u8]), Error> {
    if signed.content_type != oid::SPC_INDIRECT_DATA {
        return Err(Error::Unsupported);
    }
    let content = signed.content.ok_or(Error::Malformed)?;
    // PKCS#7 embeds the SEQUENCE itself; a CMS signer wraps it in an OCTET STRING.
    let indirect = match content.tag {
        TAG_SEQUENCE => content,
        TAG_OCTET_STRING => Reader::new(content.contents).expect(TAG_SEQUENCE)?,
        _ => return Err(Error::Malformed),
    };
    let mut fields = indirect.reader();
    fields.expect(TAG_SEQUENCE)?;
    let mut digest_info = fields.expect(TAG_SEQUENCE)?.reader();
    let (algorithm, _) = digest_info.algorithm()?;
    let algorithm = HashAlgorithm::from_oid(algorithm).ok_or(Error::Unsupported)?;
    let digest = digest_info.octet_string()?;
    if digest.len() != algorithm.digest_len() {
        return Err(Error::Malformed);
    }
    Ok((algorithm, digest))
}

/// Checks that `signed` vouches for this image file, returning the digest (in the
/// signature's algorithm) on success. Whether the signer is trusted is
/// `SignedData::verify(Content::Embedded, ...)`'s question.
pub fn check_digest(
    file: &[u8],
    signed: &SignedData<'_>,
) -> Result<(HashAlgorithm, Vec<u8>), Error> {
    let (algorithm, expected) = signed_digest(signed)?;
    let actual = digest(file, algorithm)?;
    if actual != expected {
        return Err(Error::DigestMismatch);
    }
    Ok((algorithm, actual))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pkcs7::Content;
    use crate::x509::tests::{MICROSOFT_CERTIFICATES, TEST_CA, TEST_LEAF};

    /// A two-section PE32+ image with trailing data, signed by the test leaf through hand-built
    /// Authenticode (`hack/secure-boot/make-crypto-fixtures.sh`), and the digest that script
    /// computed independently.
    const SIGNED_IMAGE: &[u8] = include_bytes!("testdata/signed.efi");
    const UNSIGNED_IMAGE: &[u8] = include_bytes!("testdata/unsigned.efi");
    const IMAGE_SHA256: &[u8] = include_bytes!("testdata/image.sha256");

    #[test]
    fn the_digest_matches_the_independent_one() {
        assert_eq!(
            digest(SIGNED_IMAGE, HashAlgorithm::Sha256).unwrap(),
            IMAGE_SHA256
        );
        assert_eq!(
            digest(UNSIGNED_IMAGE, HashAlgorithm::Sha256).unwrap(),
            IMAGE_SHA256,
            "signing changes only what the digest leaves out"
        );
        assert!(signatures(UNSIGNED_IMAGE).unwrap().is_empty());
    }

    #[test]
    fn a_signed_image_verifies_against_its_ca_or_its_leaf() {
        let blobs = signatures(SIGNED_IMAGE).unwrap();
        assert_eq!(blobs.len(), 1);
        let signed = SignedData::parse(blobs[0]).unwrap();
        let (algorithm, image_digest) = check_digest(SIGNED_IMAGE, &signed).unwrap();
        assert_eq!(algorithm, HashAlgorithm::Sha256);
        assert_eq!(image_digest, IMAGE_SHA256);
        let microsoft = MICROSOFT_CERTIFICATES[3].1;
        let verified = signed
            .verify(Content::Embedded, &[microsoft, TEST_CA])
            .unwrap();
        assert_eq!(verified.anchor(), 1);
        assert_eq!(verified.signers[0].chain[0].raw, TEST_LEAF);
        assert_eq!(
            signed
                .verify(Content::Embedded, &[TEST_LEAF])
                .unwrap()
                .anchor(),
            0
        );
        assert_eq!(
            signed.verify(Content::Embedded, &[microsoft]).unwrap_err(),
            Error::Untrusted
        );
    }

    #[test]
    fn a_changed_section_byte_breaks_the_digest() {
        let layout = Layout::parse(SIGNED_IMAGE).unwrap();
        let (offset, _) = layout.sections[0];
        let mut tampered = SIGNED_IMAGE.to_vec();
        tampered[offset + 3] ^= 0x01;
        let blobs = signatures(&tampered).unwrap();
        let signed = SignedData::parse(blobs[0]).unwrap();
        assert_eq!(
            check_digest(&tampered, &signed).unwrap_err(),
            Error::DigestMismatch
        );
    }

    #[test]
    fn the_checksum_and_certificate_entry_are_left_out() {
        let layout = Layout::parse(UNSIGNED_IMAGE).unwrap();
        let mut changed = UNSIGNED_IMAGE.to_vec();
        changed[layout.checksum] ^= 0xff;
        assert_eq!(
            digest(&changed, HashAlgorithm::Sha256).unwrap(),
            IMAGE_SHA256
        );
        changed[layout.size_of_headers - 1] ^= 0xff;
        assert_ne!(
            digest(&changed, HashAlgorithm::Sha256).unwrap(),
            IMAGE_SHA256
        );
    }

    #[test]
    fn truncated_and_garbled_images_are_refused_without_panicking() {
        for length in 0..SIGNED_IMAGE.len() {
            let truncated = &SIGNED_IMAGE[..length];
            let _ = digest(truncated, HashAlgorithm::Sha256);
            if let Ok(blobs) = signatures(truncated) {
                for blob in blobs {
                    if let Ok(signed) = SignedData::parse(blob) {
                        let _ = check_digest(truncated, &signed);
                    }
                }
            }
        }
        for position in 0..1024.min(SIGNED_IMAGE.len()) {
            let mut garbled = SIGNED_IMAGE.to_vec();
            garbled[position] = 0xff;
            let _ = digest(&garbled, HashAlgorithm::Sha256);
            let _ = signatures(&garbled);
        }
        assert_eq!(
            digest(b"MZ", HashAlgorithm::Sha256).unwrap_err(),
            Error::Malformed
        );
    }
}
