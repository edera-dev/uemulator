//! TPM 2.0 marshalling (TPM 2.0 Library Part 1, "Marshaling"): every integer is big-endian and a
//! `TPM2B_*` is a u16 byte count followed by that many bytes.
#[cfg(freestanding)]
use crate::prelude::*;

/// Reading past the end of a command. The engine turns it into `TPM_RC_INSUFFICIENT` or
/// `TPM_RC_COMMAND_SIZE`, depending on where it happened.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Truncated;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes }
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len()
    }

    pub fn take(&mut self, count: usize) -> Result<&'a [u8], Truncated> {
        if count > self.bytes.len() {
            return Err(Truncated);
        }
        let (taken, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(taken)
    }

    pub fn u8(&mut self) -> Result<u8, Truncated> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, Truncated> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, Truncated> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn u64(&mut self) -> Result<u64, Truncated> {
        let bytes = self.take(8)?;
        Ok(u64::from_be_bytes(bytes.try_into().unwrap()))
    }

    /// A `TPM2B_*`: the u16 size, then the bytes.
    pub fn sized(&mut self) -> Result<&'a [u8], Truncated> {
        let size = self.u16()? as usize;
        self.take(size)
    }

    /// What is left, all of it.
    pub fn rest(&mut self) -> &'a [u8] {
        let rest = self.bytes;
        self.bytes = &[];
        rest
    }
}

#[derive(Default)]
pub(crate) struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Writer::default()
    }

    pub fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    /// A `TPM2B_*`. Callers keep `value` within the type's maximum, which is always below 64 KiB.
    pub fn sized(&mut self, value: &[u8]) {
        self.u16(value.len() as u16);
        self.bytes.extend_from_slice(value);
    }

    /// A `TPM2B_*` around a structure (`TPM2B_PUBLIC`, `TPM2B_CREATION_DATA`, ...): `write`
    /// marshals it and the size goes in front.
    pub fn sized_with(&mut self, write: impl FnOnce(&mut Writer)) {
        let start = self.bytes.len();
        self.u16(0);
        write(self);
        let size = (self.bytes.len() - start - 2) as u16;
        self.bytes[start..start + 2].copy_from_slice(&size.to_be_bytes());
    }
}

/// Marshals with a fresh writer and hands back the bytes.
pub(crate) fn marshal(write: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut writer = Writer::new();
    write(&mut writer);
    writer.bytes
}
