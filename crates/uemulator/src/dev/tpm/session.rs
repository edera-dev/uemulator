//! Authorization sessions (Part 1, "Authorizations and Acknowledgments", "Session-based
//! encryption" and "Policy"; Part 3, "Session Commands" and "Enhanced Authorization"): password,
//! HMAC and policy sessions, bound and salted ones, AES-CFB parameter encryption, and the policy
//! commands `PolicyPCR`, `PolicyAuthValue`, `PolicyPassword`, `PolicyGetDigest` and
//! `PolicyRestart`.
//!
//! Every command goes through `authorize` before it runs and `respond` after. The order of the
//! work matters and follows the reference implementation's `ExecCommand`: the command HMAC covers
//! the parameters as they arrived (encrypted), decryption happens after the check, the response
//! parameter is encrypted before the response HMAC is computed over it, and the response HMAC
//! uses the entity's authorization value as it is after the command, which is the new one after
//! `TPM2_HierarchyChangeAuth`.
//!
//! Audit sessions are not implemented; a session that asks for audit is refused. A session
//! present only for parameter encryption has its command HMAC ignored (its response HMAC is
//! computed): no authorization rests on it.
#[cfg(freestanding)]
use crate::prelude::*;

use super::engine::{CommandSpec, Response, Tpm, end_of};
use super::hash;
use super::object::{
    AES_BLOCK_BYTES, SymmetricDefinition, Unique, read_ecc_point_contents, trim_auth,
};
use super::spec::*;
use super::wire::{Reader, Truncated, Writer, marshal};
use uemulator_crypto::{aes, ecc};

const SE_HMAC: u8 = 0x00;
const SE_POLICY: u8 = 0x01;
const SE_TRIAL: u8 = 0x03;
/// The shortest nonce a caller may send (Part 1, "Nonce").
const MIN_NONCE_BYTES: usize = 16;
const LABEL_SESSION_KEY: &[u8] = b"ATH";
const LABEL_CFB: &[u8] = b"CFB";
/// The secret-sharing label, NUL included: RSA OAEP takes it as the label bytes, KDFe as its
/// label (which then adds its own terminator, so the NUL is dropped there).
const LABEL_SECRET: &[u8] = b"SECRET\0";
/// Sessions that may exist at once, loaded or saved, and how many may be loaded.
pub(super) const MAX_ACTIVE_SESSIONS: usize = 64;
pub(super) const MAX_LOADED_SESSIONS: usize = 8;
/// The low 24 bits of a session handle; both session types draw from one space.
const SESSION_INDEX_MASK: u32 = 0x00ff_ffff;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SessionKind {
    Hmac,
    Policy,
    /// A policy session that only computes a digest and never authorizes anything.
    Trial,
}

pub(super) struct AuthSession {
    pub handle: u32,
    pub kind: SessionKind,
    /// `authHash`: every digest, HMAC and KDF of the session uses it.
    pub hash: u16,
    pub session_key: Vec<u8>,
    pub nonce_tpm: Vec<u8>,
    pub symmetric: SymmetricDefinition,
    /// `H(name || authValue)` of the entity the session is bound to (the reference
    /// implementation's `boundEntity`): a bound session authorizes that entity without its
    /// authorization value in the HMAC key, as long as neither has changed.
    pub bound: Option<Vec<u8>>,
    pub policy_digest: Vec<u8>,
    pub auth_value_needed: bool,
    pub password_needed: bool,
    /// The PCR update counter `TPM2_PolicyPCR` saw; a change before the session authorizes
    /// anything fails the authorization.
    pub pcr_counter: Option<u32>,
    /// The context sequence number while the session is saved (`TPM2_ContextSave`); a saved
    /// session keeps its handle but is not loaded.
    pub saved: Option<u64>,
}

impl AuthSession {
    fn reset_policy(&mut self) {
        let digest_bytes = hash::digest_len(self.hash).unwrap_or(0);
        self.policy_digest = vec![0; digest_bytes];
        self.auth_value_needed = false;
        self.password_needed = false;
        self.pcr_counter = None;
    }

    fn extend_policy(&mut self, parts: &[&[u8]]) {
        let mut all: Vec<&[u8]> = vec![&self.policy_digest];
        all.extend_from_slice(parts);
        self.policy_digest = hash::digest(self.hash, &all).unwrap_or_default();
    }
}

/// One entry of a command's authorization area.
pub(super) struct CommandSession {
    pub handle: u32,
    pub nonce: Vec<u8>,
    pub attributes: u8,
    pub hmac: Vec<u8>,
}

impl CommandSession {
    pub fn read(reader: &mut Reader) -> Result<Self, Truncated> {
        Ok(CommandSession {
            handle: reader.u32()?,
            nonce: reader.sized()?.to_vec(),
            attributes: reader.u8()?,
            hmac: reader.sized()?.to_vec(),
        })
    }
}

/// What authorization, decryption and the response need to know about an entity.
pub(super) struct Entity {
    pub name: Vec<u8>,
    pub auth: Vec<u8>,
    pub policy: Vec<u8>,
    /// `TPMA_OBJECT_userWithAuth`: a password or HMAC may stand in for the user role.
    pub user_with_auth: bool,
}

/// What `authorize` found, for running the command and building its response.
pub(super) struct Authorization {
    sessions: Vec<CommandSession>,
    /// The parameter area, decrypted if a session asked for that.
    pub parameters: Vec<u8>,
    /// The authorization value each authorizing session used, in case the entity is gone by the
    /// time the response is built (a flushed sequence).
    auths: Vec<Vec<u8>>,
    encrypt: Option<usize>,
}

fn bind_tag(hash_alg: u16, entity: &Entity) -> Vec<u8> {
    hash::digest(hash_alg, &[&entity.name, &entity.auth]).unwrap_or_default()
}

impl Tpm {
    pub(super) fn session(&self, handle: u32) -> Option<&AuthSession> {
        self.sessions
            .iter()
            .find(|session| session.handle == handle)
    }

    pub(super) fn loaded_session(&self, handle: u32) -> Option<&AuthSession> {
        self.session(handle)
            .filter(|session| session.saved.is_none())
    }

    fn loaded_session_mut(&mut self, handle: u32) -> Option<&mut AuthSession> {
        self.sessions
            .iter_mut()
            .find(|session| session.handle == handle && session.saved.is_none())
    }

    pub(super) fn loaded_session_count(&self) -> usize {
        self.sessions
            .iter()
            .filter(|session| session.saved.is_none())
            .count()
    }

    pub(super) fn entity(&self, handle: u32) -> Option<Entity> {
        let simple = |auth: &[u8]| Entity {
            name: handle.to_be_bytes().to_vec(),
            auth: auth.to_vec(),
            policy: Vec::new(),
            user_with_auth: true,
        };
        match handle_type(handle) {
            HT_PCR => ((handle as usize) < super::engine::PCR_COUNT).then(|| simple(&[])),
            HT_PERMANENT => PERMANENT_HANDLES
                .contains(&handle)
                .then(|| simple(self.hierarchy_auth(handle))),
            HT_TRANSIENT | HT_PERSISTENT => {
                if let Some(sequence) = self.sequences.iter().find(|s| s.handle == handle) {
                    return Some(simple(&sequence.auth));
                }
                let object = self.object(handle)?;
                Some(Entity {
                    name: object.name.clone(),
                    auth: object.sensitive.auth.clone(),
                    policy: object.public.auth_policy.clone(),
                    user_with_auth: object.public.has(OBJECT_USER_WITH_AUTH),
                })
            }
            HT_NV_INDEX => {
                let index = self.nv_index(handle)?;
                Some(Entity {
                    name: index.name(),
                    auth: index.auth.clone(),
                    policy: index.auth_policy.clone(),
                    user_with_auth: true,
                })
            }
            HT_HMAC_SESSION | HT_POLICY_SESSION => self.session(handle).map(|_| simple(&[])),
            _ => None,
        }
    }

    pub(super) fn entity_name(&self, handle: u32) -> Vec<u8> {
        self.entity(handle)
            .map_or_else(|| handle.to_be_bytes().to_vec(), |entity| entity.name)
    }

    /// The HMAC key and CFB key material of a session: `sessionKey`, plus the entity's
    /// authorization value when the session authorizes it, is not bound to it (HMAC sessions)
    /// and, for a policy session, `TPM2_PolicyAuthValue` asked for it.
    fn session_secret(&self, session: &AuthSession, entity: Option<&Entity>) -> Vec<u8> {
        let mut key = session.session_key.clone();
        if let Some(entity) = entity {
            let include = match session.kind {
                SessionKind::Hmac => {
                    session.bound.as_ref() != Some(&bind_tag(session.hash, entity))
                }
                SessionKind::Policy | SessionKind::Trial => session.auth_value_needed,
            };
            if include {
                key.extend_from_slice(&entity.auth);
            }
        }
        key
    }

    /// Checks the authorization area against the command and its handles, and decrypts the
    /// first parameter if a session asks for that. `parameters` is the parameter area as sent.
    pub(super) fn authorize(
        &self,
        command: &CommandSpec,
        handles: &[u32],
        sessions: Vec<CommandSession>,
        parameters: &[u8],
    ) -> Result<Authorization, u32> {
        let authorized = command.authorized as usize;
        let mut decrypt = None;
        let mut encrypt = None;
        for (index, entry) in sessions.iter().enumerate() {
            let number = index as u32 + 1;
            let error = |code| session_error(code, number);
            if entry.handle == RS_PW {
                if !entry.nonce.is_empty() {
                    return Err(error(RC_SIZE));
                }
                if entry.attributes & (SESSION_DECRYPT | SESSION_ENCRYPT | SESSION_AUDIT) != 0 {
                    return Err(error(RC_ATTRIBUTES));
                }
                // A password session with nothing to authorize is not an audit or encryption
                // session; the reference implementation calls that a bad attribute value.
                if index >= authorized {
                    return Err(error(RC_VALUE));
                }
                continue;
            }
            let session = self
                .loaded_session(entry.handle)
                .ok_or(RC_REFERENCE_S0 + index as u32)?;
            let digest_bytes = hash::digest_len(session.hash).unwrap_or(0);
            if entry.nonce.len() < MIN_NONCE_BYTES || entry.nonce.len() > digest_bytes {
                return Err(error(RC_SIZE));
            }
            if entry.attributes & (SESSION_AUDIT | SESSION_AUDIT_EXCLUSIVE | SESSION_AUDIT_RESET)
                != 0
            {
                return Err(error(RC_ATTRIBUTES));
            }
            let crypt = entry.attributes & (SESSION_DECRYPT | SESSION_ENCRYPT);
            if index >= authorized && crypt == 0 {
                return Err(error(RC_ATTRIBUTES));
            }
            if crypt != 0 && session.symmetric.algorithm == hash::ALG_NULL {
                return Err(error(RC_SYMMETRIC));
            }
            if entry.attributes & SESSION_DECRYPT != 0 {
                if decrypt.is_some() || !command.decrypt || parameters.len() < 2 {
                    return Err(error(RC_ATTRIBUTES));
                }
                decrypt = Some(index);
            }
            if entry.attributes & SESSION_ENCRYPT != 0 {
                if encrypt.is_some() || !command.encrypt {
                    return Err(error(RC_ATTRIBUTES));
                }
                encrypt = Some(index);
            }
            if index < authorized && session.kind == SessionKind::Trial {
                return Err(error(RC_ATTRIBUTES));
            }
            if sessions[..index]
                .iter()
                .any(|earlier| earlier.handle == entry.handle)
            {
                return Err(error(RC_HANDLE));
            }
        }
        if sessions.len() < authorized {
            return Err(RC_AUTH_MISSING);
        }
        let names: Vec<Vec<u8>> = handles
            .iter()
            .map(|handle| self.entity_name(*handle))
            .collect();
        let cp_hash = |hash_alg: u16| {
            let mut parts: Vec<&[u8]> = Vec::new();
            let code = command.code.to_be_bytes();
            parts.push(&code);
            parts.extend(names.iter().map(Vec::as_slice));
            parts.push(parameters);
            hash::digest(hash_alg, &parts).unwrap_or_default()
        };
        // The first session's HMAC also covers the nonces of the sessions that decrypt and
        // encrypt, when those are other sessions (Part 1, "HMAC Computation").
        let extra_nonce = |chosen: Option<usize>| {
            chosen
                .filter(|index| *index != 0)
                .and_then(|index| self.loaded_session(sessions[index].handle))
                .map(|session| session.nonce_tpm.clone())
                .unwrap_or_default()
        };
        let (nonce_decrypt, nonce_encrypt) = (extra_nonce(decrypt), extra_nonce(encrypt));
        let mut auths = Vec::new();
        for (index, entry) in sessions.iter().enumerate().take(authorized) {
            let number = index as u32 + 1;
            let handle = handles[index];
            let entity = self.entity(handle).ok_or(handle_error(RC_HANDLE, number))?;
            let session = self.loaded_session(entry.handle);
            let policy = session.is_some_and(|session| session.kind != SessionKind::Hmac);
            if matches!(command.code, CC_NV_READ | CC_NV_WRITE) && index == 0 {
                self.nv_access_check(command.code, handles[0], handles[1], policy)?;
            }
            auths.push(entity.auth.clone());
            let Some(session) = session else {
                if !entity.user_with_auth {
                    return Err(session_error(RC_AUTH_UNAVAILABLE, number));
                }
                if trim_auth(&entry.hmac) != entity.auth {
                    return Err(session_error(RC_AUTH_FAIL, number));
                }
                continue;
            };
            if session.kind == SessionKind::Policy {
                if entity.policy.is_empty() {
                    return Err(session_error(RC_AUTH_UNAVAILABLE, number));
                }
                if session.policy_digest != entity.policy {
                    return Err(session_error(RC_POLICY_FAIL, number));
                }
                if session
                    .pcr_counter
                    .is_some_and(|counter| counter != self.pcr_update_counter)
                {
                    return Err(RC_PCR_CHANGED);
                }
                if session.password_needed {
                    if trim_auth(&entry.hmac) != entity.auth {
                        return Err(session_error(RC_AUTH_FAIL, number));
                    }
                    continue;
                }
            } else if !entity.user_with_auth {
                return Err(session_error(RC_AUTH_UNAVAILABLE, number));
            }
            let key = self.session_secret(session, Some(&entity));
            let expected = hash::hmac(
                session.hash,
                &key,
                &[
                    &cp_hash(session.hash),
                    &entry.nonce,
                    &session.nonce_tpm,
                    if index == 0 { &nonce_decrypt } else { &[] },
                    if index == 0 { &nonce_encrypt } else { &[] },
                    &[entry.attributes],
                ],
            )
            .unwrap_or_default();
            // A policy session with no key at all may leave the HMAC empty.
            let keyless_policy =
                session.kind == SessionKind::Policy && key.is_empty() && entry.hmac.is_empty();
            if entry.hmac != expected && !keyless_policy {
                return Err(session_error(RC_AUTH_FAIL, number));
            }
        }
        let mut parameters = parameters.to_vec();
        if let Some(index) = decrypt {
            let entry = &sessions[index];
            let session = self.loaded_session(entry.handle).ok_or(RC_REFERENCE_S0)?;
            let entity = (index < authorized)
                .then(|| self.entity(handles[index]))
                .flatten();
            let key = self.session_secret(session, entity.as_ref());
            let size = u16::from_be_bytes([parameters[0], parameters[1]]) as usize;
            let data = parameters
                .get_mut(2..2 + size)
                .ok_or(parameter_error(RC_SIZE, 1))?;
            cfb(session, &key, &entry.nonce, &session.nonce_tpm, data, false);
        }
        Ok(Authorization {
            sessions,
            parameters,
            auths,
            encrypt,
        })
    }

    /// Finishes a successful command: new nonces, the response parameter encrypted if asked,
    /// the response HMACs, and the sessions updated or flushed. Returns the whole response.
    pub(super) fn respond(
        &mut self,
        tag: u16,
        command: &CommandSpec,
        handles: &[u32],
        authorization: Authorization,
        response: Response,
    ) -> Vec<u8> {
        let authorized = command.authorized as usize;
        let mut parameters = response.parameters.bytes;
        let mut new_nonces = Vec::new();
        for entry in &authorization.sessions {
            let nonce = match self.loaded_session(entry.handle) {
                Some(session) => {
                    let mut nonce = vec![0; hash::digest_len(session.hash).unwrap_or(0)];
                    self.random.fill(&mut nonce);
                    nonce
                }
                None => Vec::new(),
            };
            new_nonces.push(nonce);
        }
        // Entities as they are now, for the response HMAC and the encryption key.
        let entities: Vec<Option<Entity>> = (0..authorization.sessions.len())
            .map(|index| {
                (index < authorized).then(|| {
                    self.entity(handles[index]).unwrap_or_else(|| Entity {
                        name: Vec::new(),
                        auth: authorization.auths[index].clone(),
                        policy: Vec::new(),
                        user_with_auth: true,
                    })
                })
            })
            .collect();
        if let Some(index) = authorization.encrypt
            && parameters.len() >= 2
            && let Some(session) = self.loaded_session(authorization.sessions[index].handle)
        {
            let key = self.session_secret(session, entities[index].as_ref());
            let size = u16::from_be_bytes([parameters[0], parameters[1]]) as usize;
            if let Some(data) = parameters.get_mut(2..2 + size) {
                let caller_nonce = &authorization.sessions[index].nonce;
                cfb(session, &key, &new_nonces[index], caller_nonce, data, true);
            }
        }
        let mut writer = Writer::new();
        writer.u16(tag);
        writer.u32(0);
        writer.u32(RC_SUCCESS);
        for handle in &response.handles {
            writer.u32(*handle);
        }
        if tag == ST_SESSIONS {
            writer.u32(parameters.len() as u32);
            writer.bytes(&parameters);
            for (index, entry) in authorization.sessions.iter().enumerate() {
                let Some(session) = self.loaded_session(entry.handle) else {
                    // The password session.
                    writer.sized(&[]);
                    writer.u8(entry.attributes & SESSION_CONTINUE);
                    writer.sized(&[]);
                    continue;
                };
                let attributes = entry.attributes & !SESSION_AUDIT_RESET;
                let hmac = if session.password_needed && session.kind == SessionKind::Policy {
                    Vec::new()
                } else {
                    let rp_hash = hash::digest(
                        session.hash,
                        &[
                            &RC_SUCCESS.to_be_bytes(),
                            &command.code.to_be_bytes(),
                            &parameters,
                        ],
                    )
                    .unwrap_or_default();
                    let key = self.session_secret(session, entities[index].as_ref());
                    hash::hmac(
                        session.hash,
                        &key,
                        &[&rp_hash, &new_nonces[index], &entry.nonce, &[attributes]],
                    )
                    .unwrap_or_default()
                };
                writer.sized(&new_nonces[index]);
                writer.u8(attributes);
                writer.sized(&hmac);
            }
        } else {
            writer.bytes(&parameters);
        }
        let length = writer.bytes.len() as u32;
        writer.bytes[2..6].copy_from_slice(&length.to_be_bytes());
        for (index, entry) in authorization.sessions.iter().enumerate() {
            let continued = entry.attributes & SESSION_CONTINUE != 0;
            let Some(session) = self.loaded_session_mut(entry.handle) else {
                continue;
            };
            session.nonce_tpm = core::mem::take(&mut new_nonces[index]);
            // A policy session starts over once it has authorized something (Part 1,
            // "Policy Session Use").
            if index < authorized && session.kind == SessionKind::Policy {
                session.reset_policy();
            }
            if !continued {
                let handle = entry.handle;
                self.sessions.retain(|session| session.handle != handle);
            }
        }
        writer.bytes
    }

    pub(super) fn start_auth_session(
        &mut self,
        tpm_key: u32,
        bind: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let nonce_caller = parameters.sized()?.to_vec();
        let encrypted_salt = parameters.sized()?.to_vec();
        let session_type = parameters.u8()?;
        let symmetric = SymmetricDefinition::read(parameters)?;
        let hash_alg = parameters.u16()?;
        end_of(parameters)?;
        let digest_bytes = hash::digest_len(hash_alg).ok_or(parameter_error(RC_HASH, 5))?;
        if nonce_caller.len() < MIN_NONCE_BYTES || nonce_caller.len() > digest_bytes {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let kind = match session_type {
            SE_HMAC => SessionKind::Hmac,
            SE_POLICY => SessionKind::Policy,
            SE_TRIAL => SessionKind::Trial,
            _ => return Err(parameter_error(RC_VALUE, 3)),
        };
        if symmetric.algorithm != hash::ALG_NULL && !symmetric.is_aes_cfb() {
            return Err(parameter_error(RC_SYMMETRIC, 4));
        }
        let salt = if tpm_key == RH_NULL {
            if !encrypted_salt.is_empty() {
                return Err(parameter_error(RC_VALUE, 2));
            }
            Vec::new()
        } else {
            self.decrypt_salt(tpm_key, &encrypted_salt)?
        };
        let bind_entity = if bind == RH_NULL {
            None
        } else {
            Some(self.entity(bind).ok_or(handle_error(RC_HANDLE, 2))?)
        };
        if self.sessions.len() >= MAX_ACTIVE_SESSIONS {
            return Err(RC_SESSION_HANDLES);
        }
        if self.loaded_session_count() >= MAX_LOADED_SESSIONS {
            return Err(RC_SESSION_MEMORY);
        }
        let mut nonce_tpm = vec![0; digest_bytes];
        self.random.fill(&mut nonce_tpm);
        let bind_auth = bind_entity
            .as_ref()
            .map(|entity| entity.auth.clone())
            .unwrap_or_default();
        // Part 1, "Session Key Creation": an unbound, unsalted session has no key.
        let session_key = if bind_auth.is_empty() && salt.is_empty() {
            Vec::new()
        } else {
            let secret = [bind_auth.as_slice(), &salt].concat();
            hash::kdfa(
                hash_alg,
                &secret,
                LABEL_SESSION_KEY,
                &nonce_tpm,
                &nonce_caller,
                8 * digest_bytes as u32,
            )
            .unwrap_or_default()
        };
        let session_handle_type = match kind {
            SessionKind::Hmac => HT_HMAC_SESSION,
            SessionKind::Policy | SessionKind::Trial => HT_POLICY_SESSION,
        };
        let handle =
            (session_handle_type as u32) << HANDLE_TYPE_SHIFT | self.allocate_session_index();
        let mut session = AuthSession {
            handle,
            kind,
            hash: hash_alg,
            session_key,
            nonce_tpm: nonce_tpm.clone(),
            symmetric,
            bound: bind_entity
                .as_ref()
                .map(|entity| bind_tag(hash_alg, entity)),
            policy_digest: Vec::new(),
            auth_value_needed: false,
            password_needed: false,
            pcr_counter: None,
            saved: None,
        };
        session.reset_policy();
        self.sessions.push(session);
        let mut writer = Writer::new();
        writer.sized(&nonce_tpm);
        Ok(Response {
            handles: vec![handle],
            parameters: writer,
        })
    }

    /// A session index no live session uses, not reused right away (like transient handles).
    fn allocate_session_index(&mut self) -> u32 {
        loop {
            let index = self.next_session & SESSION_INDEX_MASK;
            self.next_session = self.next_session.wrapping_add(1);
            if !self
                .sessions
                .iter()
                .any(|session| session.handle & SESSION_INDEX_MASK == index)
            {
                return index;
            }
        }
    }

    /// The salt `tpmKey` protects (Part 1, "Secret Sharing"): RSA OAEP with label "SECRET", or
    /// ECDH with an ephemeral point and KDFe.
    fn decrypt_salt(&self, tpm_key: u32, encrypted: &[u8]) -> Result<Vec<u8>, u32> {
        let key = self.object(tpm_key).ok_or(handle_error(RC_HANDLE, 1))?;
        if !key.public.has(OBJECT_DECRYPT) {
            return Err(handle_error(RC_ATTRIBUTES, 1));
        }
        let name_alg = key.public.name_alg;
        let bad_salt = parameter_error(RC_VALUE, 2);
        match &key.sensitive.secret {
            super::object::Secret::Rsa(private) => {
                let algorithm = hash::hash_algorithm(name_alg).ok_or(bad_salt)?;
                private
                    .decrypt_oaep(algorithm, LABEL_SECRET, encrypted)
                    .map_err(|_| bad_salt)
            }
            super::object::Secret::Ecc(private) => {
                let mut reader = Reader::new(encrypted);
                // The ephemeral x goes into the KDF exactly as sent.
                let raw_x = Reader::new(encrypted)
                    .sized()
                    .map_err(|_| bad_salt)?
                    .to_vec();
                let point = read_ecc_point_contents(&mut reader, 2)?;
                if reader.remaining() != 0 {
                    return Err(bad_salt);
                }
                let shared = ecc::p256_ecdh(private, &point).map_err(|_| bad_salt)?;
                let Unique::Ecc { x: static_x, .. } = &key.public.unique else {
                    return Err(bad_salt);
                };
                let digest_bits = 8 * hash::digest_len(name_alg).unwrap_or(0) as u32;
                let label = &LABEL_SECRET[..LABEL_SECRET.len() - 1];
                hash::kdfe(name_alg, &shared, label, &raw_x, static_x, digest_bits).ok_or(bad_salt)
            }
            super::object::Secret::Data(_) => Err(handle_error(RC_KEY, 1)),
        }
    }

    fn policy_session(&mut self, handle: u32) -> Result<&mut AuthSession, u32> {
        let session = self
            .loaded_session_mut(handle)
            .ok_or(handle_error(RC_HANDLE, 1))?;
        if session.kind == SessionKind::Hmac {
            return Err(handle_error(RC_TYPE, 1));
        }
        Ok(session)
    }

    pub(super) fn policy_pcr(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let pcr_digest = parameters.sized()?.to_vec();
        let selection = self.read_pcr_selection(parameters, 2)?;
        end_of(parameters)?;
        let counter = self.pcr_update_counter;
        let hash_alg = self.policy_session(handle)?.hash;
        let digest_bytes = hash::digest_len(hash_alg).unwrap_or(0);
        if !pcr_digest.is_empty() && pcr_digest.len() != digest_bytes {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let current = self.pcr_digest(hash_alg, &selection);
        let selection_bytes = marshal(|writer| Self::write_pcr_selection(writer, &selection));
        let session = self.policy_session(handle)?;
        let digest = if session.kind == SessionKind::Trial {
            if pcr_digest.is_empty() {
                current
            } else {
                pcr_digest
            }
        } else {
            if !pcr_digest.is_empty() && pcr_digest != current {
                return Err(parameter_error(RC_VALUE, 1));
            }
            if session
                .pcr_counter
                .is_some_and(|recorded| recorded != counter)
            {
                return Err(RC_PCR_CHANGED);
            }
            session.pcr_counter = Some(counter);
            current
        };
        session.extend_policy(&[&CC_POLICY_PCR.to_be_bytes(), &selection_bytes, &digest]);
        Ok(Response::empty())
    }

    pub(super) fn policy_auth_value(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
        password: bool,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let session = self.policy_session(handle)?;
        // TPM2_PolicyPassword extends with TPM2_PolicyAuthValue's code: the two policies are
        // the same, only how the value is presented differs.
        session.extend_policy(&[&CC_POLICY_AUTH_VALUE.to_be_bytes()]);
        session.auth_value_needed = !password;
        session.password_needed = password;
        Ok(Response::empty())
    }

    pub(super) fn policy_get_digest(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let session = self.policy_session(handle)?;
        let mut writer = Writer::new();
        writer.sized(&session.policy_digest);
        Ok(Response::with(writer))
    }

    pub(super) fn policy_restart(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        self.policy_session(handle)?.reset_policy();
        Ok(Response::empty())
    }
}

/// AES-CFB over a parameter (Part 1, "Session-based encryption"): key and IV from
/// `KDFa(authHash, key, "CFB", nonceNewer, nonceOlder)`.
fn cfb(
    session: &AuthSession,
    key: &[u8],
    nonce_newer: &[u8],
    nonce_older: &[u8],
    data: &mut [u8],
    encrypt: bool,
) {
    let key_bytes = session.symmetric.key_bytes();
    let bits = 8 * (key_bytes + AES_BLOCK_BYTES) as u32;
    let Some(material) = hash::kdfa(session.hash, key, LABEL_CFB, nonce_newer, nonce_older, bits)
    else {
        return;
    };
    let (aes_key, iv) = material.split_at(key_bytes);
    let Ok(iv) = <[u8; AES_BLOCK_BYTES]>::try_from(iv) else {
        return;
    };
    // The session's symmetric definition was checked to be AES-CFB when it started.
    let _ = if encrypt {
        aes::cfb_encrypt(aes_key, &iv, data)
    } else {
        aes::cfb_decrypt(aes_key, &iv, data)
    };
}
