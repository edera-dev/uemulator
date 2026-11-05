//! `TPM2_ContextSave`, `TPM2_ContextLoad` and `TPM2_FlushContext` (Part 1, "Context Management").
//! Linux's resource manager (`/dev/tpmrm0`) saves every transient object and session after each
//! command and loads them again before the next, so this is on the path of everything systemd
//! and the TSS do.
//!
//! An object's context carries the whole object, encrypted and with an HMAC. A session's state
//! stays inside the TPM (the reference implementation does the same with its gap counter): the
//! session keeps its handle in a saved state, and its context only names it and the sequence
//! number it was saved under, so an older context of the same session is refused. Hash
//! sequences cannot be saved: their hash state is not serializable here.
//!
//! The keys come from a secret made at each TPM Reset, so contexts never load after a reset; the
//! HMAC also covers the hierarchy's proof, so an object context stops loading when
//! `TPM2_Clear` changes it.
#[cfg(freestanding)]
use crate::prelude::*;

use super::engine::{Response, Tpm, end_of};
use super::hash::{self, ALG_SHA256};
use super::object::{AES_BLOCK_BYTES, Object, Public, Sensitive};
use super::spec::*;
use super::wire::{Reader, Truncated, Writer, marshal};
use uemulator_crypto::aes;

const LABEL_CONTEXT: &[u8] = b"CONTEXT";
const CONTEXT_KEY_BYTES: usize = 16;
/// `TPM_PT_MAX_OBJECT_CONTEXT`/`_SESSION_CONTEXT`: an RSA-2048 object needs about 800 bytes.
pub(super) const MAX_OBJECT_CONTEXT_BYTES: usize = 2048;
pub(super) const MAX_SESSION_CONTEXT_BYTES: usize = 128;

impl Tpm {
    fn context_keys(&self, sequence: u64, saved_handle: u32) -> (Vec<u8>, [u8; AES_BLOCK_BYTES]) {
        let bits = 8 * (CONTEXT_KEY_BYTES + AES_BLOCK_BYTES) as u32;
        let material = hash::kdfa(
            ALG_SHA256,
            &self.context_secret,
            LABEL_CONTEXT,
            &sequence.to_be_bytes(),
            &saved_handle.to_be_bytes(),
            bits,
        )
        .unwrap_or_default();
        let mut iv = [0u8; AES_BLOCK_BYTES];
        iv.copy_from_slice(&material[CONTEXT_KEY_BYTES..]);
        (material[..CONTEXT_KEY_BYTES].to_vec(), iv)
    }

    fn context_integrity(
        &self,
        sequence: u64,
        saved_handle: u32,
        hierarchy: u32,
        ciphertext: &[u8],
    ) -> Vec<u8> {
        hash::hmac(
            ALG_SHA256,
            &self.context_secret,
            &[
                &sequence.to_be_bytes(),
                &saved_handle.to_be_bytes(),
                &hierarchy.to_be_bytes(),
                self.proof(hierarchy),
                ciphertext,
            ],
        )
        .unwrap_or_default()
    }

    fn write_context(
        &self,
        saved_handle: u32,
        hierarchy: u32,
        sequence: u64,
        mut payload: Vec<u8>,
    ) -> Writer {
        let (key, iv) = self.context_keys(sequence, saved_handle);
        let _ = aes::cfb_encrypt(&key, &iv, &mut payload);
        let integrity = self.context_integrity(sequence, saved_handle, hierarchy, &payload);
        let mut writer = Writer::new();
        writer.u64(sequence);
        writer.u32(saved_handle);
        writer.u32(hierarchy);
        writer.sized_with(|writer| {
            writer.sized(&integrity);
            writer.bytes(&payload);
        });
        writer
    }

    pub(super) fn context_save(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let sequence = self.context_sequence;
        self.context_sequence += 1;
        match handle_type(handle) {
            HT_TRANSIENT => {
                if self
                    .sequences
                    .iter()
                    .any(|sequence| sequence.handle == handle)
                {
                    return Err(handle_error(RC_TYPE, 1));
                }
                let object = self.object(handle).ok_or(RC_REFERENCE_H0)?;
                let payload = marshal(|writer| {
                    writer.sized(&object.public.bytes());
                    writer.sized(&object.sensitive_bytes());
                    writer.sized(&object.qualified_name);
                });
                let hierarchy = object.hierarchy;
                Ok(Response::with(self.write_context(
                    SAVED_OBJECT,
                    hierarchy,
                    sequence,
                    payload,
                )))
            }
            HT_HMAC_SESSION | HT_POLICY_SESSION => {
                let session = self
                    .sessions
                    .iter_mut()
                    .find(|session| session.handle == handle && session.saved.is_none())
                    .ok_or(RC_REFERENCE_H0)?;
                session.saved = Some(sequence);
                let payload = handle.to_be_bytes().to_vec();
                Ok(Response::with(
                    self.write_context(handle, RH_NULL, sequence, payload),
                ))
            }
            _ => Err(handle_error(RC_VALUE, 1)),
        }
    }

    pub(super) fn context_load(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let sequence = parameters.u64()?;
        let saved_handle = parameters.u32()?;
        let hierarchy = parameters.u32()?;
        let blob = parameters.sized()?;
        end_of(parameters)?;
        if !matches!(hierarchy, RH_OWNER | RH_ENDORSEMENT | RH_PLATFORM | RH_NULL) {
            return Err(parameter_error(RC_HIERARCHY, 1));
        }
        let mut reader = Reader::new(blob);
        let integrity = reader.sized().map_err(|_| parameter_error(RC_SIZE, 1))?;
        let mut payload = reader.rest().to_vec();
        if integrity != self.context_integrity(sequence, saved_handle, hierarchy, &payload) {
            return Err(parameter_error(RC_INTEGRITY, 1));
        }
        let (key, iv) = self.context_keys(sequence, saved_handle);
        let _ = aes::cfb_decrypt(&key, &iv, &mut payload);
        match handle_type(saved_handle) {
            HT_TRANSIENT if saved_handle == SAVED_OBJECT => {
                let parse = |reader: &mut Reader| {
                    Ok::<_, Truncated>((
                        reader.sized()?.to_vec(),
                        reader.sized()?.to_vec(),
                        reader.sized()?.to_vec(),
                    ))
                };
                let mut reader = Reader::new(&payload);
                let (public, sensitive, qualified_name) =
                    parse(&mut reader).map_err(|_| RC_BAD_CONTEXT)?;
                let public =
                    Public::read(&mut Reader::new(&public), 1).map_err(|_| RC_BAD_CONTEXT)?;
                let sensitive = Sensitive::read(&mut Reader::new(&sensitive), &public, 1)
                    .map_err(|_| RC_BAD_CONTEXT)?;
                self.check_transient_room()?;
                let handle = self.allocate_transient();
                let name = public.name();
                self.objects.push(Object {
                    handle,
                    hierarchy,
                    public,
                    name,
                    qualified_name,
                    sensitive,
                });
                Ok(Response {
                    handles: vec![handle],
                    parameters: Writer::new(),
                })
            }
            HT_HMAC_SESSION | HT_POLICY_SESSION => {
                if payload != saved_handle.to_be_bytes() {
                    return Err(RC_BAD_CONTEXT);
                }
                if self.loaded_session_count() >= super::session::MAX_LOADED_SESSIONS {
                    return Err(RC_SESSION_MEMORY);
                }
                let session = self
                    .sessions
                    .iter_mut()
                    .find(|session| {
                        session.handle == saved_handle && session.saved == Some(sequence)
                    })
                    .ok_or(parameter_error(RC_HANDLE, 1))?;
                session.saved = None;
                Ok(Response {
                    handles: vec![saved_handle],
                    parameters: Writer::new(),
                })
            }
            _ => Err(parameter_error(RC_VALUE, 1)),
        }
    }

    pub(super) fn flush_context(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let handle = parameters.u32()?;
        end_of(parameters)?;
        let before = (
            self.sequences.len(),
            self.objects.len(),
            self.sessions.len(),
        );
        match handle_type(handle) {
            HT_TRANSIENT => {
                self.sequences.retain(|sequence| sequence.handle != handle);
                self.objects.retain(|object| object.handle != handle);
            }
            HT_HMAC_SESSION | HT_POLICY_SESSION => {
                self.sessions.retain(|session| session.handle != handle);
            }
            _ => return Err(parameter_error(RC_VALUE, 1)),
        }
        if before
            == (
                self.sequences.len(),
                self.objects.len(),
                self.sessions.len(),
            )
        {
            return Err(parameter_error(RC_HANDLE, 1));
        }
        Ok(Response::empty())
    }
}
