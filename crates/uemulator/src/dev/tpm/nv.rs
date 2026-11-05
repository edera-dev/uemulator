//! NV indices (Part 1, "NV Memory"; Part 3, "Non-volatile Storage"): ordinary indices only,
//! defined by the owner or the platform, read and written whole or in pieces, kept in the
//! machine's key-value store. Counter, bit-field, extend and PIN indices and the lock commands
//! are not implemented: those types are refused, and the lock attributes (`WRITEDEFINE`,
//! `WRITE_STCLEAR`, `READ_STCLEAR`, `GLOBALLOCK`) are accepted and never acted on, so an index is
//! never locked.
#[cfg(freestanding)]
use crate::prelude::*;

use super::engine::{Response, Tpm, end_of};
use super::hash;
use super::hierarchy::{STORE_VERSION, nv_key};
use super::object::trim_auth;
use super::spec::*;
use super::wire::{Reader, Truncated, Writer, marshal};

pub(super) const NV_PPWRITE: u32 = 1 << 0;
pub(super) const NV_OWNERWRITE: u32 = 1 << 1;
pub(super) const NV_AUTHWRITE: u32 = 1 << 2;
pub(super) const NV_POLICYWRITE: u32 = 1 << 3;
const NV_TYPE_MASK: u32 = 0xf << 4;
const NV_TYPE_ORDINARY: u32 = 0;
const NV_POLICY_DELETE: u32 = 1 << 10;
const NV_WRITELOCKED: u32 = 1 << 11;
const NV_WRITEALL: u32 = 1 << 12;
pub(super) const NV_PPREAD: u32 = 1 << 16;
pub(super) const NV_OWNERREAD: u32 = 1 << 17;
pub(super) const NV_AUTHREAD: u32 = 1 << 18;
pub(super) const NV_POLICYREAD: u32 = 1 << 19;
const NV_CLEAR_STCLEAR: u32 = 1 << 27;
const NV_READLOCKED: u32 = 1 << 28;
const NV_WRITTEN: u32 = 1 << 29;
const NV_PLATFORMCREATE: u32 = 1 << 30;
/// Bits 8, 9 and 20..=24 are reserved.
const NV_RESERVED: u32 = 0b11 << 8 | 0b1_1111 << 20;
const NV_WRITE_AUTHORIZATIONS: u32 = NV_PPWRITE | NV_OWNERWRITE | NV_AUTHWRITE | NV_POLICYWRITE;
const NV_READ_AUTHORIZATIONS: u32 = NV_PPREAD | NV_OWNERREAD | NV_AUTHREAD | NV_POLICYREAD;

/// `MAX_NV_INDEX_SIZE`: the largest index.
pub(super) const MAX_NV_INDEX_BYTES: usize = 2048;
/// `MAX_NV_BUFFER_SIZE`: the most one `TPM2_NV_Read` or `TPM2_NV_Write` moves.
pub(super) const MAX_NV_BUFFER_BYTES: usize = 1024;
/// How many indices may exist at once, and how much they may hold together.
pub(super) const MAX_NV_INDICES: usize = 64;
pub(super) const NV_MEMORY_BYTES: usize = 64 * 1024;

pub(super) struct NvIndex {
    pub handle: u32,
    pub name_alg: u16,
    pub attributes: u32,
    pub auth_policy: Vec<u8>,
    pub auth: Vec<u8>,
    /// `dataSize` bytes; meaningful once `TPMA_NV_WRITTEN` is set.
    pub data: Vec<u8>,
}

impl NvIndex {
    /// `TPMS_NV_PUBLIC`.
    fn write_public(&self, writer: &mut Writer) {
        writer.u32(self.handle);
        writer.u16(self.name_alg);
        writer.u32(self.attributes);
        writer.sized(&self.auth_policy);
        writer.u16(self.data.len() as u16);
    }

    fn public_bytes(&self) -> Vec<u8> {
        marshal(|writer| self.write_public(writer))
    }

    /// `nameAlg || H(TPMS_NV_PUBLIC)`. It changes when the index is first written, since
    /// `TPMA_NV_WRITTEN` is part of what is hashed.
    pub fn name(&self) -> Vec<u8> {
        let mut name = self.name_alg.to_be_bytes().to_vec();
        name.extend(hash::digest(self.name_alg, &[&self.public_bytes()]).unwrap_or_default());
        name
    }

    pub fn has(&self, attribute: u32) -> bool {
        self.attributes & attribute != 0
    }

    pub fn platform_created(&self) -> bool {
        self.has(NV_PLATFORMCREATE)
    }

    pub fn encode(&self) -> Vec<u8> {
        marshal(|writer| {
            writer.u8(STORE_VERSION);
            writer.sized(&self.public_bytes());
            writer.sized(&self.auth);
            writer.sized(&self.data);
        })
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        if reader.u8().ok()? != STORE_VERSION {
            return None;
        }
        let mut public = Reader::new(reader.sized().ok()?);
        let (handle, name_alg, attributes, auth_policy, data_size) =
            read_public(&mut public).ok()?;
        let auth = reader.sized().ok()?.to_vec();
        let data = reader.sized().ok()?.to_vec();
        let consistent = public.remaining() == 0
            && reader.remaining() == 0
            && data.len() == data_size as usize
            && (NV_INDEX_FIRST..=NV_INDEX_LAST).contains(&handle);
        consistent.then_some(NvIndex {
            handle,
            name_alg,
            attributes,
            auth_policy,
            auth,
            data,
        })
    }
}

fn read_public(reader: &mut Reader) -> Result<(u32, u16, u32, Vec<u8>, u16), Truncated> {
    Ok((
        reader.u32()?,
        reader.u16()?,
        reader.u32()?,
        reader.sized()?.to_vec(),
        reader.u16()?,
    ))
}

impl Tpm {
    pub(super) fn nv_index(&self, handle: u32) -> Option<&NvIndex> {
        self.nv.iter().find(|index| index.handle == handle)
    }

    fn save_nv_index(&self, handle: u32) -> Result<(), u32> {
        let index = self.nv_index(handle).ok_or(RC_NV_UNAVAILABLE)?;
        self.store_set(nv_key(handle).as_bytes(), &index.encode())
    }

    pub(super) fn remove_nv_index(&mut self, handle: u32) -> Result<(), u32> {
        self.store_delete(nv_key(handle).as_bytes())?;
        self.nv.retain(|index| index.handle != handle);
        Ok(())
    }

    /// `TPMA_NV_CLEAR_STCLEAR` indices read as unwritten after every TPM Reset.
    pub(super) fn nv_startup(&mut self) {
        for index in &mut self.nv {
            if index.has(NV_CLEAR_STCLEAR) {
                index.attributes &= !NV_WRITTEN;
            }
        }
    }

    pub(super) fn nv_define_space(
        &mut self,
        auth_handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let auth = trim_auth(parameters.sized()?);
        let public_bytes = parameters.sized()?;
        end_of(parameters)?;
        let mut public = Reader::new(public_bytes);
        let (handle, name_alg, attributes, auth_policy, data_size) =
            read_public(&mut public).map_err(|_| parameter_error(RC_SIZE, 2))?;
        if public.remaining() != 0 {
            return Err(parameter_error(RC_SIZE, 2));
        }
        if !matches!(auth_handle, RH_OWNER | RH_PLATFORM) {
            return Err(handle_error(RC_VALUE, 1));
        }
        let digest_bytes = hash::digest_len(name_alg).ok_or(parameter_error(RC_HASH, 2))?;
        if auth.len() > digest_bytes {
            return Err(parameter_error(RC_SIZE, 1));
        }
        if !(NV_INDEX_FIRST..=NV_INDEX_LAST).contains(&handle) {
            return Err(parameter_error(RC_VALUE, 2));
        }
        if !auth_policy.is_empty() && auth_policy.len() != digest_bytes {
            return Err(parameter_error(RC_SIZE, 2));
        }
        let attribute_error = parameter_error(RC_ATTRIBUTES, 2);
        if attributes & NV_TYPE_MASK != NV_TYPE_ORDINARY
            || attributes & (NV_RESERVED | NV_WRITELOCKED | NV_READLOCKED | NV_WRITTEN) != 0
            || attributes & NV_WRITE_AUTHORIZATIONS == 0
            || attributes & NV_READ_AUTHORIZATIONS == 0
            || (attributes & NV_PLATFORMCREATE != 0) != (auth_handle == RH_PLATFORM)
            // Only TPM2_NV_UndefineSpaceSpecial removes these, and it is not implemented.
            || attributes & NV_POLICY_DELETE != 0
        {
            return Err(attribute_error);
        }
        if data_size as usize > MAX_NV_INDEX_BYTES {
            return Err(parameter_error(RC_SIZE, 2));
        }
        if self.nv_index(handle).is_some() {
            return Err(RC_NV_DEFINED);
        }
        let used: usize = self.nv.iter().map(|index| index.data.len()).sum();
        if self.nv.len() == MAX_NV_INDICES || used + data_size as usize > NV_MEMORY_BYTES {
            return Err(RC_NV_SPACE);
        }
        self.nv.push(NvIndex {
            handle,
            name_alg,
            attributes,
            auth_policy,
            auth,
            data: vec![0; data_size as usize],
        });
        if let Err(code) = self.save_nv_index(handle) {
            self.nv.retain(|index| index.handle != handle);
            return Err(code);
        }
        Ok(Response::empty())
    }

    pub(super) fn nv_undefine_space(
        &mut self,
        auth_handle: u32,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let index = self.nv_index(handle).ok_or(handle_error(RC_HANDLE, 2))?;
        if !matches!(auth_handle, RH_OWNER | RH_PLATFORM) {
            return Err(handle_error(RC_VALUE, 1));
        }
        if index.platform_created() && auth_handle != RH_PLATFORM {
            return Err(RC_NV_AUTHORIZATION);
        }
        self.remove_nv_index(handle)?;
        Ok(Response::empty())
    }

    pub(super) fn nv_read_public(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let index = self.nv_index(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        let mut writer = Writer::new();
        writer.sized(&index.public_bytes());
        writer.sized(&index.name());
        Ok(Response::with(writer))
    }

    /// Which authorization an NV command may use (Part 3, "NV_Read" and "NV_Write": the
    /// reference implementation's `NvReadAccessChecks` and `NvWriteAccessChecks`). Called from
    /// authorization, which knows whether the session is a policy session.
    pub(super) fn nv_access_check(
        &self,
        code: u32,
        auth_handle: u32,
        index_handle: u32,
        policy: bool,
    ) -> Result<(), u32> {
        let Some(index) = self.nv_index(index_handle) else {
            return Ok(());
        };
        let (owner, platform, auth, policy_bit) = if code == CC_NV_READ {
            (NV_OWNERREAD, NV_PPREAD, NV_AUTHREAD, NV_POLICYREAD)
        } else {
            (NV_OWNERWRITE, NV_PPWRITE, NV_AUTHWRITE, NV_POLICYWRITE)
        };
        let needed = match auth_handle {
            RH_OWNER => owner,
            RH_PLATFORM => platform,
            handle if handle == index_handle && policy => policy_bit,
            handle if handle == index_handle => auth,
            _ => return Err(RC_NV_AUTHORIZATION),
        };
        if !index.has(needed) {
            return Err(RC_NV_AUTHORIZATION);
        }
        Ok(())
    }

    pub(super) fn nv_read(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let size = parameters.u16()? as usize;
        let offset = parameters.u16()? as usize;
        end_of(parameters)?;
        let index = self.nv_index(handle).ok_or(handle_error(RC_HANDLE, 2))?;
        if !index.has(NV_WRITTEN) {
            return Err(RC_NV_UNINITIALIZED);
        }
        if size > MAX_NV_BUFFER_BYTES {
            return Err(parameter_error(RC_VALUE, 1));
        }
        if offset + size > index.data.len() {
            return Err(RC_NV_RANGE);
        }
        let mut writer = Writer::new();
        writer.sized(&index.data[offset..offset + size]);
        Ok(Response::with(writer))
    }

    pub(super) fn nv_write(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let data = parameters.sized()?.to_vec();
        let offset = parameters.u16()? as usize;
        end_of(parameters)?;
        let index = self
            .nv
            .iter_mut()
            .find(|index| index.handle == handle)
            .ok_or(handle_error(RC_HANDLE, 2))?;
        if data.len() > MAX_NV_BUFFER_BYTES {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let end = offset + data.len();
        if end > index.data.len()
            || (index.has(NV_WRITEALL) && (offset != 0 || data.len() != index.data.len()))
        {
            return Err(RC_NV_RANGE);
        }
        let previous = (index.data.clone(), index.attributes);
        index.data[offset..end].copy_from_slice(&data);
        index.attributes |= NV_WRITTEN;
        if let Err(code) = self.save_nv_index(handle) {
            if let Some(index) = self.nv.iter_mut().find(|index| index.handle == handle) {
                (index.data, index.attributes) = previous;
            }
            return Err(code);
        }
        Ok(Response::empty())
    }
}
