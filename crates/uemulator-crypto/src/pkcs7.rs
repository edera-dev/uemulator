//! PKCS#7 SignedData (RFC 2315 9; its CMS successor is RFC 5652 5), as Authenticode signatures
//! and UEFI time-based authenticated variables carry it.

use alloc::vec::Vec;

use crate::Error;
use crate::der::{Reader, TAG_INTEGER, TAG_OCTET_STRING, TAG_SEQUENCE, TAG_SET, Tlv, context, oid};
use crate::sha::HashAlgorithm;
use crate::x509::{Certificate, SignatureAlgorithm};

/// Longest signer-to-anchor chain followed: real ones have two or three links.
pub const MAX_CHAIN_LENGTH: usize = 8;

/// What the signature covers.
#[derive(Clone, Copy, Debug)]
pub enum Content<'c> {
    /// The eContent inside the SignedData (Authenticode's SpcIndirectDataContent).
    Embedded,
    /// Bytes signed separately (an authenticated variable's serialized update).
    Detached(&'c [u8]),
}

#[derive(Clone, Debug)]
pub struct SignerInfo<'a> {
    /// IssuerAndSerialNumber: the issuer Name's DER and the serial's INTEGER contents.
    pub issuer: &'a [u8],
    pub serial: &'a [u8],
    pub digest_algorithm: HashAlgorithm,
    /// The `[0] IMPLICIT` authenticated attributes, raw (tag 0xA0 included).
    pub authenticated_attributes: Option<Tlv<'a>>,
    pub signature: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct SignedData<'a> {
    /// The eContentType OID contents (`oid::PKCS7_DATA` when nothing is embedded).
    pub content_type: &'a [u8],
    /// The embedded content value (inside `[0] EXPLICIT`), when present.
    pub content: Option<Tlv<'a>>,
    pub certificates: Vec<Certificate<'a>>,
    pub signers: Vec<SignerInfo<'a>>,
}

/// One signer that checked out: which anchor it reached, and the certificates from the signer
/// (first) up to the one the anchor matched or issued.
#[derive(Clone, Debug)]
pub struct VerifiedSigner<'a> {
    pub anchor: usize,
    pub chain: Vec<Certificate<'a>>,
}

#[derive(Clone, Debug)]
pub struct Verified<'a> {
    /// One entry per SignerInfo, in order; never empty.
    pub signers: Vec<VerifiedSigner<'a>>,
}

impl<'a> Verified<'a> {
    /// The anchor the first signer reached.
    pub fn anchor(&self) -> usize {
        self.signers[0].anchor
    }

    /// Every certificate of every signer's chain.
    pub fn certificates(&self) -> impl Iterator<Item = &Certificate<'a>> {
        self.signers.iter().flat_map(|signer| signer.chain.iter())
    }
}

impl<'a> SignedData<'a> {
    /// Parses a ContentInfo carrying SignedData, or a bare SignedData (UEFI 2.9 8.2.2 lets an
    /// authenticated variable omit the ContentInfo, and EDK2 accepts both). Bytes after the
    /// structure are ignored: WIN_CERTIFICATE entries are padded to 8 bytes.
    pub fn parse(der: &'a [u8]) -> Result<Self, Error> {
        let outer = Reader::new(der).expect(TAG_SEQUENCE)?;
        let mut fields = outer.reader();
        let signed_data = if fields.peek_tag() == Some(crate::der::TAG_OID) {
            if fields.oid()? != oid::PKCS7_SIGNED_DATA {
                return Err(Error::Unsupported);
            }
            let wrapper = fields.expect(context(0))?;
            wrapper.reader().expect(TAG_SEQUENCE)?
        } else {
            outer
        };
        Self::parse_signed_data(signed_data)
    }

    fn parse_signed_data(signed_data: Tlv<'a>) -> Result<Self, Error> {
        let mut fields = signed_data.reader();
        fields.expect(TAG_INTEGER)?;
        fields.expect(TAG_SET)?;
        let mut content_info = fields.expect(TAG_SEQUENCE)?.reader();
        let content_type = content_info.oid()?;
        let content = match content_info.optional(context(0))? {
            Some(wrapper) => Some(wrapper.reader().read()?),
            None => None,
        };
        let mut certificates = Vec::new();
        if let Some(bag) = fields.optional(context(0))? {
            let mut reader = bag.reader();
            while !reader.is_empty() {
                let value = reader.read()?;
                // Attribute certificates and other choices are skipped, not refused.
                if value.tag == TAG_SEQUENCE {
                    certificates.push(Certificate::parse(value.raw)?);
                }
            }
        }
        fields.optional(context(1))?;
        let mut signer_set = fields.expect(TAG_SET)?.reader();
        let mut signers = Vec::new();
        while !signer_set.is_empty() {
            signers.push(SignerInfo::parse(signer_set.expect(TAG_SEQUENCE)?)?);
        }
        if signers.is_empty() {
            return Err(Error::Malformed);
        }
        Ok(Self {
            content_type,
            content,
            certificates,
            signers,
        })
    }

    /// The bytes the content digest covers. For embedded content that is the contents octets
    /// of the content value, without its tag and length (RFC 2315 9.3; Authenticode hashes
    /// SpcIndirectDataContent the same way, and CMS's OCTET STRING eContent reduces to it).
    pub fn content_bytes<'c>(&self, content: Content<'c>) -> Result<&'c [u8], Error>
    where
        'a: 'c,
    {
        match content {
            Content::Embedded => self
                .content
                .map(|value| value.contents)
                .ok_or(Error::Malformed),
            Content::Detached(bytes) => Ok(bytes),
        }
    }

    /// The certificate a signer names, from the bag or else from `extra`.
    fn find_signer(
        &self,
        signer: &SignerInfo<'a>,
        extra: &[Certificate<'a>],
    ) -> Option<Certificate<'a>> {
        self.certificates
            .iter()
            .chain(extra)
            .find(|certificate| {
                certificate.issuer == signer.issuer && certificate.serial == signer.serial
            })
            .cloned()
    }

    /// Checks one signer's signature over `content` (no trust decision) and returns its
    /// certificate. `extra` are certificates to look in when the bag lacks the signer.
    pub fn verify_signer(
        &self,
        index: usize,
        content: Content<'_>,
        extra: &[Certificate<'a>],
    ) -> Result<Certificate<'a>, Error> {
        let signer = self.signers.get(index).ok_or(Error::Malformed)?;
        let algorithm = signer.digest_algorithm;
        let content_digest = algorithm.digest(self.content_bytes(content)?);
        let certificate = self
            .find_signer(signer, extra)
            .ok_or(Error::SignerNotFound)?;
        let key = certificate.public_key.as_ref().ok_or(Error::Unsupported)?;
        match &signer.authenticated_attributes {
            None => key.verify_pkcs1v15(algorithm, &content_digest, signer.signature)?,
            Some(attributes) => {
                let digest =
                    attribute_value(attributes, oid::MESSAGE_DIGEST)?.ok_or(Error::Malformed)?;
                if Reader::new(digest).octet_string()? != content_digest.as_slice() {
                    return Err(Error::DigestMismatch);
                }
                // RFC 2315 9.2: when present, contentType must name the content.
                if let Some(value) = attribute_value(attributes, oid::CONTENT_TYPE)?
                    && Reader::new(value).oid()? != self.content_type
                {
                    return Err(Error::DigestMismatch);
                }
                // The signature covers the attributes encoded as a SET OF, not with the
                // [0] IMPLICIT tag they travel under; the length octets are the same.
                let mut signed = Vec::with_capacity(attributes.raw.len());
                signed.push(TAG_SET);
                signed.extend_from_slice(&attributes.raw[1..]);
                key.verify_pkcs1v15(algorithm, &algorithm.digest(&signed), signer.signature)?;
            }
        }
        Ok(certificate)
    }

    /// Verifies every signer and requires each to chain to one of `anchors` (DER
    /// certificates). A chain is trusted when one of its certificates equals an anchor or was
    /// issued by one (names match and the anchor's key verifies it): any anchor may be an
    /// intermediate, as with OpenSSL's `X509_V_FLAG_PARTIAL_CHAIN` that EDK2 uses. Validity
    /// periods and key usages are not checked, as in EDK2.
    pub fn verify(
        &self,
        content: Content<'_>,
        anchors: &[&'a [u8]],
    ) -> Result<Verified<'a>, Error> {
        // Anchors that do not parse can never match; their positions are kept.
        let parsed: Vec<Option<Certificate<'a>>> = anchors
            .iter()
            .map(|der| Certificate::parse(der).ok())
            .collect();
        let usable: Vec<Certificate<'a>> = parsed.iter().flatten().cloned().collect();
        let mut signers = Vec::new();
        for index in 0..self.signers.len() {
            let signer = self.verify_signer(index, content, &usable)?;
            signers.push(self.chain_to_anchor(signer, &parsed)?);
        }
        Ok(Verified { signers })
    }

    fn chain_to_anchor(
        &self,
        signer: Certificate<'a>,
        anchors: &[Option<Certificate<'a>>],
    ) -> Result<VerifiedSigner<'a>, Error> {
        let mut chain = alloc::vec![signer];
        loop {
            let current = &chain[chain.len() - 1];
            for (index, anchor) in anchors.iter().enumerate() {
                let Some(anchor) = anchor else { continue };
                if anchor.raw == current.raw || anchor.issued(current) {
                    return Ok(VerifiedSigner {
                        anchor: index,
                        chain,
                    });
                }
            }
            if chain.len() >= MAX_CHAIN_LENGTH {
                return Err(Error::Untrusted);
            }
            // A certificate already in the chain is never added again, so loops end.
            let issuer = self
                .certificates
                .iter()
                .find(|candidate| {
                    !chain.iter().any(|link| link.raw == candidate.raw) && candidate.issued(current)
                })
                .cloned()
                .ok_or(Error::Untrusted)?;
            chain.push(issuer);
        }
    }
}

impl<'a> SignerInfo<'a> {
    fn parse(value: Tlv<'a>) -> Result<Self, Error> {
        let mut fields = value.reader();
        fields.expect(TAG_INTEGER)?;
        // CMS's subjectKeyIdentifier choice ([0]) is not supported: Authenticode and UEFI
        // signers use IssuerAndSerialNumber.
        let mut identifier = fields
            .expect(TAG_SEQUENCE)
            .map_err(|_| Error::Unsupported)?
            .reader();
        let issuer = identifier.expect(TAG_SEQUENCE)?.raw;
        let serial = identifier.expect(TAG_INTEGER)?.contents;
        let (digest_oid, _) = fields.algorithm()?;
        let digest_algorithm = HashAlgorithm::from_oid(digest_oid).ok_or(Error::Unsupported)?;
        let authenticated_attributes = fields.optional(context(0))?;
        let (encryption_oid, _) = fields.algorithm()?;
        // rsaEncryption, or (in some signers' output) the combined sha*WithRSAEncryption,
        // which then has to agree with digestAlgorithm.
        if encryption_oid != oid::RSA_ENCRYPTION {
            match SignatureAlgorithm::from_oid(encryption_oid) {
                SignatureAlgorithm::RsaPkcs1(hash) if hash == digest_algorithm => {}
                _ => return Err(Error::Unsupported),
            }
        }
        let signature = fields.expect(TAG_OCTET_STRING)?.contents;
        Ok(Self {
            issuer,
            serial,
            digest_algorithm,
            authenticated_attributes,
            signature,
        })
    }
}

/// The single value of the attribute `wanted` in an attribute set, as raw DER.
fn attribute_value<'a>(attributes: &Tlv<'a>, wanted: &[u8]) -> Result<Option<&'a [u8]>, Error> {
    let mut reader = attributes.reader();
    while !reader.is_empty() {
        let mut attribute = reader.expect(TAG_SEQUENCE)?.reader();
        if attribute.oid()? != wanted {
            continue;
        }
        let mut values = attribute.expect(TAG_SET)?.reader();
        let value = values.read()?;
        // A second value would be ambiguous.
        if !values.is_empty() {
            return Err(Error::Malformed);
        }
        return Ok(Some(value.raw));
    }
    Ok(None)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::x509::tests::{MICROSOFT_CERTIFICATES, TEST_CA, TEST_LEAF, UEMULATOR_PK};

    const DBX_KEK2011: &[u8] =
        include_bytes!("../../uemulator-firmware/secure-boot/dbx-x64-kek2011.auth2");
    const DBX_KEK2023: &[u8] =
        include_bytes!("../../uemulator-firmware/secure-boot/dbx-x64-kek2023.auth2");
    const KEK_2011: &[u8] = MICROSOFT_CERTIFICATES[0].1;
    const KEK_2023: &[u8] = MICROSOFT_CERTIFICATES[1].1;
    /// d719b2cb-3d3a-4596-a3bc-dad00e67656f in the EFI byte order.
    const IMAGE_SECURITY_DATABASE_GUID: [u8; 16] = [
        0xcb, 0xb2, 0x19, 0xd7, 0x3a, 0x3d, 0x96, 0x45, 0xa3, 0xbc, 0xda, 0xd0, 0x0e, 0x67, 0x65,
        0x6f,
    ];
    const EFI_TIME_BYTES: usize = 16;
    /// dwLength, wRevision, wCertificateType and the CertType GUID.
    const WIN_CERTIFICATE_UEFI_GUID_HEADER_BYTES: usize = 24;
    /// NON_VOLATILE | BOOTSERVICE_ACCESS | RUNTIME_ACCESS | TIME_BASED_AUTHENTICATED_WRITE_ACCESS
    /// | APPEND_WRITE: Microsoft's dbx updates are signed as appends, and the attributes in the
    /// signed message are the ones SetVariable gets, APPEND_WRITE included.
    const DBX_ATTRIBUTES: u32 = 0x67;
    const APPEND_WRITE: u32 = 0x40;

    /// Splits an EFI_VARIABLE_AUTHENTICATION_2 payload into (timestamp, PKCS#7, data).
    fn split_auth2(payload: &[u8]) -> (&[u8], &[u8], &[u8]) {
        let timestamp = &payload[..EFI_TIME_BYTES];
        let length = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;
        let signature = &payload
            [EFI_TIME_BYTES + WIN_CERTIFICATE_UEFI_GUID_HEADER_BYTES..EFI_TIME_BYTES + length];
        (timestamp, signature, &payload[EFI_TIME_BYTES + length..])
    }

    /// UEFI 2.9 8.2.6: name (UTF-16, no terminator), vendor GUID, attributes, timestamp, data.
    fn signed_message(attributes: u32, timestamp: &[u8], data: &[u8]) -> Vec<u8> {
        let mut message: Vec<u8> = "dbx".encode_utf16().flat_map(u16::to_le_bytes).collect();
        message.extend(IMAGE_SECURITY_DATABASE_GUID);
        message.extend(attributes.to_le_bytes());
        message.extend(timestamp);
        message.extend(data);
        message
    }

    fn check_dbx(payload: &[u8], kek: &'static [u8], other: &'static [u8]) {
        let (timestamp, signature, data) = split_auth2(payload);
        let signed = SignedData::parse(signature).unwrap();
        assert_eq!(signed.content_type, oid::PKCS7_DATA);
        assert!(signed.content.is_none());
        assert!(signed.signers[0].authenticated_attributes.is_none());
        let message = signed_message(DBX_ATTRIBUTES, timestamp, data);
        let verified = signed
            .verify(Content::Detached(&message), &[other, kek])
            .unwrap();
        assert_eq!(verified.anchor(), 1);
        assert_eq!(
            verified.signers[0].chain.len(),
            1,
            "the signer is issued by the KEK CA"
        );
        assert_eq!(
            signed
                .verify(Content::Detached(&message), &[other])
                .unwrap_err(),
            Error::Untrusted
        );
        let replaced = signed_message(DBX_ATTRIBUTES & !APPEND_WRITE, timestamp, data);
        assert_eq!(
            signed
                .verify(Content::Detached(&replaced), &[kek])
                .unwrap_err(),
            Error::BadSignature,
            "the same update as a replacing write does not verify"
        );
        let mut tampered = message.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            signed
                .verify(Content::Detached(&tampered), &[kek])
                .unwrap_err(),
            Error::BadSignature
        );
    }

    #[test]
    fn the_kek_2011_dbx_update_verifies_against_kek_ca_2011_only() {
        check_dbx(DBX_KEK2011, KEK_2011, KEK_2023);
    }

    #[test]
    fn the_kek_2023_dbx_update_verifies_against_kek_2k_ca_2023_only() {
        check_dbx(DBX_KEK2023, KEK_2023, KEK_2011);
    }

    #[test]
    fn the_signer_certificate_itself_can_be_the_anchor() {
        let (timestamp, signature, data) = split_auth2(DBX_KEK2023);
        let signed = SignedData::parse(signature).unwrap();
        let message = signed_message(DBX_ATTRIBUTES, timestamp, data);
        let leaf = signed.certificates[0].raw;
        let verified = signed.verify(Content::Detached(&message), &[leaf]).unwrap();
        assert_eq!(verified.anchor(), 0);
    }

    /// `openssl cms -sign -binary` over `DETACHED_MESSAGE` with the test leaf, detached, with
    /// and without signed attributes (`hack/secure-boot/make-crypto-fixtures.sh`).
    const DETACHED_MESSAGE: &[u8] = b"uemulator authenticated variable";
    const DETACHED_WITH_ATTRIBUTES: &[u8] = include_bytes!("testdata/detached-attributes.p7");
    const DETACHED_WITHOUT_ATTRIBUTES: &[u8] = include_bytes!("testdata/detached-noattributes.p7");

    #[test]
    fn detached_cms_signatures_verify_with_and_without_signed_attributes() {
        for (name, der) in [
            ("attributes", DETACHED_WITH_ATTRIBUTES),
            ("no attributes", DETACHED_WITHOUT_ATTRIBUTES),
        ] {
            let signed = SignedData::parse(der).unwrap();
            assert_eq!(
                signed.signers[0].authenticated_attributes.is_some(),
                name == "attributes"
            );
            let verified = signed
                .verify(
                    Content::Detached(DETACHED_MESSAGE),
                    &[UEMULATOR_PK, TEST_CA],
                )
                .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            assert_eq!(verified.anchor(), 1, "{name}");
            assert_eq!(verified.signers[0].chain.len(), 1, "{name}");
            assert_eq!(verified.signers[0].chain[0].raw, TEST_LEAF, "{name}");
            let wrong = signed
                .verify(Content::Detached(b"something else"), &[TEST_CA])
                .unwrap_err();
            let expected = if name == "attributes" {
                Error::DigestMismatch
            } else {
                Error::BadSignature
            };
            assert_eq!(wrong, expected, "{name}");
            assert_eq!(
                signed
                    .verify(Content::Detached(DETACHED_MESSAGE), &[UEMULATOR_PK])
                    .unwrap_err(),
                Error::Untrusted,
                "{name}"
            );
            assert_eq!(
                signed.verify(Content::Embedded, &[TEST_CA]).unwrap_err(),
                Error::Malformed,
                "{name}: nothing is embedded"
            );
        }
    }

    #[test]
    fn truncated_and_garbled_signatures_are_refused_without_panicking() {
        let (_, dbx, _) = split_auth2(DBX_KEK2023);
        for der in [dbx, DETACHED_WITH_ATTRIBUTES, DETACHED_WITHOUT_ATTRIBUTES] {
            for length in 0..der.len() {
                if let Ok(signed) = SignedData::parse(&der[..length]) {
                    let _ = signed.verify(Content::Detached(DETACHED_MESSAGE), &[TEST_CA]);
                }
            }
            // Some bytes (the version, unsigned fields) do not matter; this is about never
            // panicking on whatever the bytes become.
            for position in (0..der.len()).step_by(7) {
                let mut garbled = der.to_vec();
                garbled[position] ^= 0x5a;
                if let Ok(signed) = SignedData::parse(&garbled) {
                    let _ =
                        signed.verify(Content::Detached(DETACHED_MESSAGE), &[TEST_CA, KEK_2023]);
                }
            }
        }
    }
}
