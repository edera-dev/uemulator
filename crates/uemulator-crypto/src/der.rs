//! A minimal DER reader (ITU-T X.690): enough for X.509, PKCS#7 and Authenticode.
//!
//! Only single-byte tags and definite lengths: every structure parsed here is DER, and the
//! indefinite form (BER) is refused rather than guessed at.

use crate::Error;

pub const TAG_INTEGER: u8 = 0x02;
pub const TAG_BIT_STRING: u8 = 0x03;
pub const TAG_OCTET_STRING: u8 = 0x04;
pub const TAG_NULL: u8 = 0x05;
pub const TAG_OID: u8 = 0x06;
pub const TAG_SEQUENCE: u8 = 0x30;
pub const TAG_SET: u8 = 0x31;
/// `[n]` context-specific, constructed: EXPLICIT tagging and constructed IMPLICIT fields.
pub const fn context(number: u8) -> u8 {
    0xa0 | number
}

/// A tag whose low five bits are all set continues in following bytes (X.690 8.1.2.4).
const HIGH_TAG_NUMBER: u8 = 0x1f;
const LONG_LENGTH_FLAG: u8 = 0x80;
/// Lengths beyond four bytes cannot describe anything that fits in memory here.
const MAX_LENGTH_BYTES: usize = 4;

/// Object identifier contents (without tag and length).
pub mod oid {
    /// 1.3.14.3.2.26
    pub const SHA1: &[u8] = &[0x2b, 0x0e, 0x03, 0x02, 0x1a];
    /// 2.16.840.1.101.3.4.2.1
    pub const SHA256: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
    /// 2.16.840.1.101.3.4.2.2
    pub const SHA384: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02];
    /// 2.16.840.1.101.3.4.2.3
    pub const SHA512: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];
    /// 1.2.840.113549.1.1.1
    pub const RSA_ENCRYPTION: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
    /// 1.2.840.113549.1.1.5
    pub const SHA1_WITH_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05];
    /// 1.2.840.113549.1.1.11
    pub const SHA256_WITH_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
    /// 1.2.840.113549.1.1.12
    pub const SHA384_WITH_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];
    /// 1.2.840.113549.1.1.13
    pub const SHA512_WITH_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];
    /// 1.2.840.113549.1.7.1
    pub const PKCS7_DATA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x07, 0x01];
    /// 1.2.840.113549.1.7.2
    pub const PKCS7_SIGNED_DATA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x07, 0x02];
    /// 1.2.840.113549.1.9.3
    pub const CONTENT_TYPE: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x03];
    /// 1.2.840.113549.1.9.4
    pub const MESSAGE_DIGEST: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x04];
    /// 1.3.6.1.4.1.311.2.1.4, Authenticode's SpcIndirectDataContent.
    pub const SPC_INDIRECT_DATA: &[u8] =
        &[0x2b, 0x06, 0x01, 0x04, 0x01, 0x82, 0x37, 0x02, 0x01, 0x04];
}

/// One encoded value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tlv<'a> {
    pub tag: u8,
    /// The contents octets.
    pub contents: &'a [u8],
    /// Identifier, length and contents together.
    pub raw: &'a [u8],
}

impl<'a> Tlv<'a> {
    /// A reader over the contents of a constructed value.
    pub fn reader(&self) -> Reader<'a> {
        Reader::new(self.contents)
    }
}

/// Reads consecutive values from a byte string.
#[derive(Clone, Copy, Debug)]
pub struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// What has not been read yet.
    pub fn rest(&self) -> &'a [u8] {
        self.data
    }

    pub fn peek_tag(&self) -> Option<u8> {
        self.data.first().copied()
    }

    pub fn read(&mut self) -> Result<Tlv<'a>, Error> {
        let data = self.data;
        let (&tag, after_tag) = data.split_first().ok_or(Error::Malformed)?;
        if tag & HIGH_TAG_NUMBER == HIGH_TAG_NUMBER {
            return Err(Error::Malformed);
        }
        let (&first, mut after_length) = after_tag.split_first().ok_or(Error::Malformed)?;
        let length = if first & LONG_LENGTH_FLAG == 0 {
            first as usize
        } else {
            let count = (first & !LONG_LENGTH_FLAG) as usize;
            // Zero length bytes is the indefinite form.
            if count == 0 || count > MAX_LENGTH_BYTES || after_length.len() < count {
                return Err(Error::Malformed);
            }
            let (bytes, rest) = after_length.split_at(count);
            after_length = rest;
            bytes
                .iter()
                .fold(0usize, |value, &byte| (value << 8) | byte as usize)
        };
        if after_length.len() < length {
            return Err(Error::Malformed);
        }
        let header = data.len() - after_length.len();
        let raw = &data[..header + length];
        self.data = &data[header + length..];
        Ok(Tlv {
            tag,
            contents: &after_length[..length],
            raw,
        })
    }

    pub fn expect(&mut self, tag: u8) -> Result<Tlv<'a>, Error> {
        let value = self.read()?;
        if value.tag != tag {
            return Err(Error::Malformed);
        }
        Ok(value)
    }

    /// Reads the next value only when it carries `tag`.
    pub fn optional(&mut self, tag: u8) -> Result<Option<Tlv<'a>>, Error> {
        if self.peek_tag() == Some(tag) {
            self.expect(tag).map(Some)
        } else {
            Ok(None)
        }
    }

    /// An INTEGER as unsigned big-endian bytes without leading zeros; negative values refused.
    pub fn unsigned_integer(&mut self) -> Result<&'a [u8], Error> {
        unsigned_integer(self.expect(TAG_INTEGER)?.contents)
    }

    pub fn oid(&mut self) -> Result<&'a [u8], Error> {
        Ok(self.expect(TAG_OID)?.contents)
    }

    /// A BIT STRING whose length is a whole number of bytes, as keys and signatures are.
    pub fn bit_string(&mut self) -> Result<&'a [u8], Error> {
        bit_string(self.expect(TAG_BIT_STRING)?.contents)
    }

    pub fn octet_string(&mut self) -> Result<&'a [u8], Error> {
        Ok(self.expect(TAG_OCTET_STRING)?.contents)
    }

    /// AlgorithmIdentifier: the OID and, when present, the raw parameters.
    pub fn algorithm(&mut self) -> Result<(&'a [u8], Option<Tlv<'a>>), Error> {
        let mut fields = self.expect(TAG_SEQUENCE)?.reader();
        let oid = fields.oid()?;
        let parameters = if fields.is_empty() {
            None
        } else {
            Some(fields.read()?)
        };
        if !fields.is_empty() {
            return Err(Error::Malformed);
        }
        Ok((oid, parameters))
    }
}

pub fn unsigned_integer(contents: &[u8]) -> Result<&[u8], Error> {
    match contents.first() {
        None => Err(Error::Malformed),
        Some(&byte) if byte & 0x80 != 0 => Err(Error::Malformed),
        Some(_) => {
            let leading = contents.iter().take_while(|&&byte| byte == 0).count();
            Ok(&contents[leading..])
        }
    }
}

pub fn bit_string(contents: &[u8]) -> Result<&[u8], Error> {
    match contents.split_first() {
        Some((0, bits)) => Ok(bits),
        _ => Err(Error::Malformed),
    }
}

/// The DER length octets for `length`.
pub(crate) fn encode_length(length: usize, out: &mut alloc::vec::Vec<u8>) {
    if length < LONG_LENGTH_FLAG as usize {
        out.push(length as u8);
        return;
    }
    let bytes = (length as u64).to_be_bytes();
    let skip = bytes.iter().take_while(|&&byte| byte == 0).count();
    out.push(LONG_LENGTH_FLAG | (bytes.len() - skip) as u8);
    out.extend_from_slice(&bytes[skip..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn reads_short_and_long_lengths() {
        let mut long = vec![TAG_OCTET_STRING, 0x82, 0x01, 0x00];
        long.extend(core::iter::repeat_n(7u8, 256));
        long.extend([TAG_NULL, 0]);
        let mut reader = Reader::new(&long);
        let value = reader.read().unwrap();
        assert_eq!(value.tag, TAG_OCTET_STRING);
        assert_eq!(value.contents.len(), 256);
        assert_eq!(value.raw.len(), 260);
        assert_eq!(reader.expect(TAG_NULL).unwrap().contents, &[] as &[u8]);
        assert!(reader.is_empty());
    }

    #[test]
    fn refuses_truncation_indefinite_lengths_and_long_tags() {
        for bad in [
            &[][..],
            &[TAG_SEQUENCE],
            &[TAG_SEQUENCE, 2, 0],
            &[TAG_SEQUENCE, 0x80, 0, 0],
            &[TAG_SEQUENCE, 0x85, 0, 0, 0, 0, 1],
            &[0x1f, 0x81, 0x00, 0],
            &[TAG_SEQUENCE, 0x82, 0x01],
        ] {
            assert_eq!(Reader::new(bad).read(), Err(Error::Malformed), "{bad:x?}");
        }
    }

    #[test]
    fn integers_lose_their_sign_byte() {
        assert_eq!(
            unsigned_integer(&[0x00, 0x80, 0x01]).unwrap(),
            &[0x80, 0x01]
        );
        assert_eq!(unsigned_integer(&[0x00]).unwrap(), &[] as &[u8]);
        assert_eq!(unsigned_integer(&[0xff]), Err(Error::Malformed));
        assert_eq!(unsigned_integer(&[]), Err(Error::Malformed));
    }

    #[test]
    fn bit_strings_must_be_whole_bytes() {
        assert_eq!(bit_string(&[0, 1, 2]).unwrap(), &[1, 2]);
        assert_eq!(bit_string(&[3, 1, 2]), Err(Error::Malformed));
    }

    #[test]
    fn lengths_encode_minimally() {
        for (length, expected) in [
            (0usize, &[0u8][..]),
            (127, &[127]),
            (128, &[0x81, 0x80]),
            (256, &[0x82, 1, 0]),
        ] {
            let mut out = Vec::new();
            encode_length(length, &mut out);
            assert_eq!(out, expected);
        }
    }
}
