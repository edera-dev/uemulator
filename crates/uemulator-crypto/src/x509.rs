//! X.509 certificates (RFC 5280 4.1), parsed in place: what signature checks and Secure Boot's
//! databases need, nothing about validity periods or extensions (UEFI firmware ignores time,
//! having no trusted clock).

use alloc::vec::Vec;

use crate::Error;
use crate::der::{Reader, TAG_INTEGER, TAG_SEQUENCE, context, oid};
use crate::rsa::RsaPublicKey;
use crate::sha::HashAlgorithm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    RsaPkcs1(HashAlgorithm),
    /// Parsed but never verifies (RSASSA-PSS, ECDSA, ...).
    Unknown,
}

impl SignatureAlgorithm {
    pub fn from_oid(contents: &[u8]) -> Self {
        let hash = match contents {
            oid::SHA1_WITH_RSA => HashAlgorithm::Sha1,
            oid::SHA256_WITH_RSA => HashAlgorithm::Sha256,
            oid::SHA384_WITH_RSA => HashAlgorithm::Sha384,
            oid::SHA512_WITH_RSA => HashAlgorithm::Sha512,
            _ => return Self::Unknown,
        };
        Self::RsaPkcs1(hash)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Certificate<'a> {
    /// The whole certificate's DER.
    pub raw: &'a [u8],
    /// The TBSCertificate's DER, tag and length included: what the issuer signed, and what dbx
    /// `EFI_CERT_X509_SHA*` entries hash.
    pub tbs: &'a [u8],
    /// The serial number's INTEGER contents as encoded (sign byte kept), for matching
    /// IssuerAndSerialNumber byte for byte.
    pub serial: &'a [u8],
    /// The issuer and subject Names' DER, tag and length included.
    pub issuer: &'a [u8],
    pub subject: &'a [u8],
    /// The SubjectPublicKeyInfo's DER.
    pub subject_public_key_info: &'a [u8],
    /// `None` when the key is not RSA (or an RSA key this crate refuses).
    pub public_key: Option<RsaPublicKey>,
    pub signature_algorithm: SignatureAlgorithm,
    pub signature: &'a [u8],
}

impl<'a> Certificate<'a> {
    /// Parses the certificate at the start of `der`; bytes after it are ignored.
    pub fn parse(der: &'a [u8]) -> Result<Self, Error> {
        let certificate = Reader::new(der).expect(TAG_SEQUENCE)?;
        let mut fields = certificate.reader();
        let tbs = fields.expect(TAG_SEQUENCE)?;
        let (algorithm, _) = fields.algorithm()?;
        let signature = fields.bit_string()?;
        if !fields.is_empty() {
            return Err(Error::Malformed);
        }

        let mut tbs_fields = tbs.reader();
        tbs_fields.optional(context(0))?;
        let serial = tbs_fields.expect(TAG_INTEGER)?.contents;
        if serial.is_empty() {
            return Err(Error::Malformed);
        }
        let (inner_algorithm, _) = tbs_fields.algorithm()?;
        // RFC 5280 4.1.1.2: the two copies must agree.
        if inner_algorithm != algorithm {
            return Err(Error::Malformed);
        }
        let issuer = tbs_fields.expect(TAG_SEQUENCE)?.raw;
        tbs_fields.expect(TAG_SEQUENCE)?;
        let subject = tbs_fields.expect(TAG_SEQUENCE)?.raw;
        let subject_public_key_info = tbs_fields.expect(TAG_SEQUENCE)?.raw;
        let public_key = RsaPublicKey::from_subject_public_key_info(subject_public_key_info).ok();
        Ok(Self {
            raw: certificate.raw,
            tbs: tbs.raw,
            serial,
            issuer,
            subject,
            subject_public_key_info,
            public_key,
            signature_algorithm: SignatureAlgorithm::from_oid(algorithm),
            signature,
        })
    }

    /// Whether this certificate's key made `subject`'s signature. Names are not compared:
    /// see `issued`.
    pub fn verifies(&self, subject: &Certificate<'_>) -> bool {
        let (Some(key), SignatureAlgorithm::RsaPkcs1(hash)) =
            (&self.public_key, subject.signature_algorithm)
        else {
            return false;
        };
        key.verify_pkcs1v15(hash, &hash.digest(subject.tbs), subject.signature)
            .is_ok()
    }

    /// Whether `subject` names this certificate as its issuer and carries its signature.
    pub fn issued(&self, subject: &Certificate<'_>) -> bool {
        self.subject == subject.issuer && self.verifies(subject)
    }

    pub fn is_self_signed(&self) -> bool {
        self.issued(self)
    }

    pub fn tbs_digest(&self, algorithm: HashAlgorithm) -> Vec<u8> {
        algorithm.digest(self.tbs)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub const MICROSOFT_CERTIFICATES: [(&str, &[u8]); 7] = [
        (
            "Microsoft Corporation KEK CA 2011",
            include_bytes!("../../uemulator-firmware/secure-boot/microsoft-kek-ca-2011.der"),
        ),
        (
            "Microsoft Corporation KEK 2K CA 2023",
            include_bytes!("../../uemulator-firmware/secure-boot/microsoft-kek-2k-ca-2023.der"),
        ),
        (
            "Microsoft Windows Production PCA 2011",
            include_bytes!(
                "../../uemulator-firmware/secure-boot/microsoft-windows-production-pca-2011.der"
            ),
        ),
        (
            "Microsoft Corporation UEFI CA 2011",
            include_bytes!("../../uemulator-firmware/secure-boot/microsoft-uefi-ca-2011.der"),
        ),
        (
            "Windows UEFI CA 2023",
            include_bytes!("../../uemulator-firmware/secure-boot/windows-uefi-ca-2023.der"),
        ),
        (
            "Microsoft UEFI CA 2023",
            include_bytes!("../../uemulator-firmware/secure-boot/microsoft-uefi-ca-2023.der"),
        ),
        (
            "Microsoft Option ROM UEFI CA 2023",
            include_bytes!(
                "../../uemulator-firmware/secure-boot/microsoft-option-rom-uefi-ca-2023.der"
            ),
        ),
    ];
    pub const UEMULATOR_PK: &[u8] =
        include_bytes!("../../uemulator-firmware/secure-boot/uemulator-pk.der");
    pub const TEST_CA: &[u8] = include_bytes!("testdata/test-ca.der");
    pub const TEST_LEAF: &[u8] = include_bytes!("testdata/test-leaf.der");

    fn contains(haystack: &[u8], needle: &str) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle.as_bytes())
    }

    #[test]
    fn every_microsoft_certificate_parses_with_an_rsa_2048_key() {
        for (name, der) in MICROSOFT_CERTIFICATES {
            let certificate = Certificate::parse(der).unwrap_or_else(|error| {
                panic!("{name}: {error:?}");
            });
            assert_eq!(certificate.raw, der, "{name}");
            assert!(contains(certificate.subject, name), "{name}");
            assert_eq!(
                certificate.signature_algorithm,
                SignatureAlgorithm::RsaPkcs1(HashAlgorithm::Sha256),
                "{name}"
            );
            let key = certificate.public_key.as_ref().expect(name);
            assert_eq!(key.size(), 256, "{name}");
            assert_eq!(key.exponent, [1, 0, 1], "{name}");
        }
    }

    #[test]
    fn a_self_signed_certificate_verifies_itself() {
        let pk = Certificate::parse(UEMULATOR_PK).unwrap();
        assert!(pk.is_self_signed());
        let ca = Certificate::parse(TEST_CA).unwrap();
        assert!(ca.is_self_signed());
        assert!(!ca.issued(&pk));
    }

    #[test]
    fn an_issuer_verifies_what_it_issued_and_nothing_else() {
        let ca = Certificate::parse(TEST_CA).unwrap();
        let leaf = Certificate::parse(TEST_LEAF).unwrap();
        assert!(ca.issued(&leaf));
        assert!(!leaf.issued(&ca));
        let kek = Certificate::parse(MICROSOFT_CERTIFICATES[0].1).unwrap();
        assert!(!kek.verifies(&leaf));
        let mut tampered = leaf.clone();
        tampered.tbs = ca.tbs;
        assert!(!ca.verifies(&tampered));
    }

    #[test]
    fn tbs_digests_match_openssl() {
        // `openssl asn1parse -strparse 4 -out tbs` then `openssl dgst -sha256`: the dbx
        // EFI_CERT_X509_SHA256 form.
        let ca = Certificate::parse(TEST_CA).unwrap();
        let expected = include_bytes!("testdata/test-ca-tbs.sha256");
        assert_eq!(ca.tbs_digest(HashAlgorithm::Sha256), expected);
    }

    #[test]
    fn truncated_certificates_are_refused() {
        for (_, der) in MICROSOFT_CERTIFICATES {
            for length in 0..der.len() {
                assert!(Certificate::parse(&der[..length]).is_err(), "{length}");
            }
        }
    }
}
