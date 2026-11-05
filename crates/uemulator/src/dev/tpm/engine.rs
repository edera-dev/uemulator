//! The TPM 2.0 itself: commands in, responses out (TPM 2.0 Library, Parts 2 and 3, revision
//! 1.59). Start-up and self tests, capabilities, random numbers, PCRs, hashing and hash
//! sequences and the clock live here, with the dispatch every command goes through; objects
//! (`object.rs`), sessions and policy (`session.rs`), NV indices (`nv.rs`), contexts
//! (`context.rs`) and the hierarchies with what persists of them (`hierarchy.rs`) have their own
//! files. Every command not in `COMMANDS` answers `TPM_RC_COMMAND_CODE` and is left out of
//! `TPM_CAP_COMMANDS`, which is how Linux and the TSS libraries find out what a TPM can do. Only
//! locality 0 exists (the CRB interface offers no other), so the DRTM PCRs 17..=22 can be read
//! but never extended or reset.
//!
//! The volatile state lives as long as the device: a guest reset builds a new one, which is a
//! TPM reset (`_TPM_Init`). Seeds, hierarchy authorizations, persistent objects and NV indices
//! are in the machine's key-value store (`hierarchy.rs`).
#[cfg(freestanding)]
use crate::prelude::*;

use super::hash::{self, ALG_NULL, ALG_SHA256, HASH_ALGORITHMS, Hash, MAX_DIGEST_BYTES};
use super::hierarchy::{Hierarchy, PersistentState};
use super::nv::NvIndex;
use super::object::Object;
use super::session::{AuthSession, CommandSession};
use super::spec::*;
use super::wire::{Reader, Truncated, Writer};
use crate::cpu::xsave::splitmix64;
use crate::dev::RngSource;
use crate::dev::kv::SharedKeyValueStore;

pub(crate) use super::spec::RC_SUCCESS;

const HEADER_BYTES: usize = 10;

/// What `TPMA_CC` says about a command, which Linux's resource manager depends on: it moves
/// `handles` and `response_handles` in and out of its virtual handle space. `decrypt` and
/// `encrypt` say whether the first command or response parameter is a `TPM2B` a session may
/// encrypt.
pub(super) struct CommandSpec {
    pub code: u32,
    /// Handles in the command's handle area.
    pub handles: u8,
    /// How many of those (from the first) need an authorization session.
    pub authorized: u8,
    pub response_handles: u8,
    /// The command flushes the transient object it names (the sequence).
    pub flushes: bool,
    pub decrypt: bool,
    pub encrypt: bool,
}

const fn spec(code: u32, handles: u8, authorized: u8, response_handles: u8) -> CommandSpec {
    CommandSpec {
        code,
        handles,
        authorized,
        response_handles,
        flushes: false,
        decrypt: false,
        encrypt: false,
    }
}

const fn decrypting(command: CommandSpec) -> CommandSpec {
    CommandSpec {
        decrypt: true,
        ..command
    }
}

const fn encrypting(command: CommandSpec) -> CommandSpec {
    CommandSpec {
        encrypt: true,
        ..command
    }
}

const fn flushing(command: CommandSpec) -> CommandSpec {
    CommandSpec {
        flushes: true,
        ..command
    }
}

/// Sorted by code: `TPM_CAP_COMMANDS` lists them in this order.
const COMMANDS: [CommandSpec; 44] = [
    spec(CC_EVICT_CONTROL, 2, 1, 0),
    spec(CC_NV_UNDEFINE_SPACE, 2, 1, 0),
    spec(CC_CLEAR, 1, 1, 0),
    decrypting(spec(CC_HIERARCHY_CHANGE_AUTH, 1, 1, 0)),
    decrypting(spec(CC_NV_DEFINE_SPACE, 1, 1, 0)),
    encrypting(decrypting(spec(CC_CREATE_PRIMARY, 1, 1, 1))),
    decrypting(spec(CC_NV_WRITE, 2, 1, 0)),
    spec(CC_DICTIONARY_ATTACK_LOCK_RESET, 1, 1, 0),
    decrypting(spec(CC_PCR_EVENT, 1, 1, 0)),
    spec(CC_PCR_RESET, 1, 1, 0),
    flushing(encrypting(decrypting(spec(CC_SEQUENCE_COMPLETE, 1, 1, 0)))),
    spec(CC_INCREMENTAL_SELF_TEST, 0, 0, 0),
    spec(CC_SELF_TEST, 0, 0, 0),
    spec(CC_STARTUP, 0, 0, 0),
    spec(CC_SHUTDOWN, 0, 0, 0),
    decrypting(spec(CC_STIR_RANDOM, 0, 0, 0)),
    encrypting(spec(CC_NV_READ, 2, 1, 0)),
    encrypting(decrypting(spec(CC_CREATE, 1, 1, 0))),
    encrypting(decrypting(spec(CC_ECDH_ZGEN, 1, 1, 0))),
    encrypting(decrypting(spec(CC_LOAD, 1, 1, 1))),
    decrypting(spec(CC_SEQUENCE_UPDATE, 1, 1, 0)),
    encrypting(spec(CC_UNSEAL, 1, 1, 0)),
    spec(CC_CONTEXT_LOAD, 0, 0, 1),
    spec(CC_CONTEXT_SAVE, 1, 0, 0),
    encrypting(spec(CC_ECDH_KEYGEN, 1, 0, 0)),
    spec(CC_FLUSH_CONTEXT, 0, 0, 0),
    encrypting(spec(CC_NV_READ_PUBLIC, 1, 0, 0)),
    spec(CC_POLICY_AUTH_VALUE, 1, 0, 0),
    encrypting(spec(CC_READ_PUBLIC, 1, 0, 0)),
    encrypting(decrypting(spec(CC_START_AUTH_SESSION, 2, 0, 1))),
    spec(CC_GET_CAPABILITY, 0, 0, 0),
    encrypting(spec(CC_GET_RANDOM, 0, 0, 0)),
    encrypting(spec(CC_GET_TEST_RESULT, 0, 0, 0)),
    encrypting(decrypting(spec(CC_HASH, 0, 0, 0))),
    spec(CC_PCR_READ, 0, 0, 0),
    decrypting(spec(CC_POLICY_PCR, 1, 0, 0)),
    spec(CC_POLICY_RESTART, 1, 0, 0),
    spec(CC_READ_CLOCK, 0, 0, 0),
    spec(CC_PCR_EXTEND, 1, 1, 0),
    flushing(decrypting(spec(CC_EVENT_SEQUENCE_COMPLETE, 2, 2, 0))),
    decrypting(spec(CC_HASH_SEQUENCE_START, 0, 0, 1)),
    encrypting(spec(CC_POLICY_GET_DIGEST, 1, 0, 0)),
    spec(CC_TEST_PARMS, 0, 0, 0),
    spec(CC_POLICY_PASSWORD, 1, 0, 0),
];

const TPMA_CC_COMMAND_INDEX_MASK: u32 = 0xffff;
const TPMA_CC_FLUSHED: u32 = 1 << 24;
const TPMA_CC_C_HANDLES_SHIFT: u32 = 25;
const TPMA_CC_R_HANDLE: u32 = 1 << 28;

pub(super) fn command_attributes(command: &CommandSpec) -> u32 {
    (command.code & TPMA_CC_COMMAND_INDEX_MASK)
        | if command.flushes { TPMA_CC_FLUSHED } else { 0 }
        | (command.handles as u32) << TPMA_CC_C_HANDLES_SHIFT
        | if command.response_handles > 0 {
            TPMA_CC_R_HANDLE
        } else {
            0
        }
}

pub(super) fn commands() -> &'static [CommandSpec] {
    &COMMANDS
}

const SU_CLEAR: u16 = 0x0000;
const SU_STATE: u16 = 0x0001;

pub(crate) const PCR_COUNT: usize = 24;
/// Bytes of a PCR bitmap: `TPM_PT_PCR_SELECT_MIN`.
pub(super) const PCR_SELECT_BYTES: usize = PCR_COUNT / 8;
/// The PC Client profile's resettable PCRs: 16 (debug) and 23 (application support).
pub(super) const PCR_DEBUG: usize = 16;
pub(super) const PCR_APPLICATION: usize = 23;
/// 17..=22 belong to a dynamic launch: they start as all ones and need locality 2 or above.
pub(super) const PCR_DRTM_FIRST: usize = 17;
pub(super) const PCR_DRTM_LAST: usize = 22;
/// The banks that are allocated. Only SHA-256, as swtpm gives QEMU by default; a bank that
/// is not allocated is not reported by `TPM_CAP_PCRS` at all.
pub(super) const PCR_BANKS: [u16; 1] = [ALG_SHA256];
/// `TPML_DIGEST` holds at most eight digests, so `TPM2_PCR_Read` answers at most eight PCRs.
const PCR_READ_MAX_DIGESTS: usize = 8;

/// Transient objects and hash sequences loaded at once. The spec's floor is 3; more lets a
/// client that skips the resource manager keep a parent, a child and a sequence around.
pub(super) const TRANSIENT_SLOTS: usize = 8;

/// The most sessions one command may carry.
const MAX_SESSIONS: usize = 3;

/// `TPM2B_MAX_BUFFER`/`TPM2B_EVENT` capacity; `TPM_PT_INPUT_BUFFER`.
pub(super) const MAX_BUFFER_BYTES: usize = 1024;
const MAX_EVENT_BYTES: usize = 1024;

const NS_PER_MS: u64 = 1_000_000;
/// `TPMI_YES_NO`: the clock has not gone backwards since the last orderly shutdown.
const CLOCK_SAFE: u8 = 1;
/// The secret behind context protection (`context.rs`), made at each TPM Reset.
const CONTEXT_SECRET_BYTES: usize = 32;

/// The bytes `TPM2_GetRandom` hands out, and the TPM's own randomness: seeds, keys, nonces.
pub(crate) struct Random {
    source: RngSource,
    state: u64,
}

impl Random {
    pub fn new(source: RngSource) -> Self {
        let state = match &source {
            RngSource::Seeded(seed) => *seed,
            RngSource::Host(_) => 0,
        };
        Random { source, state }
    }

    pub(super) fn fill(&mut self, bytes: &mut [u8]) {
        match &self.source {
            RngSource::Host(entropy) => entropy(bytes),
            RngSource::Seeded(_) => {
                for chunk in bytes.chunks_mut(8) {
                    let word = splitmix64(&mut self.state).to_le_bytes();
                    chunk.copy_from_slice(&word[..chunk.len()]);
                }
            }
        }
    }
}

pub(super) enum SequenceKind {
    Hash(Box<Hash>),
    /// `TPM2_HashSequenceStart` with `TPM_ALG_NULL`: one hash per implemented algorithm, for
    /// `TPM2_EventSequenceComplete`.
    Event(Vec<Hash>),
}

impl SequenceKind {
    fn update(&mut self, data: &[u8]) {
        match self {
            SequenceKind::Hash(hash) => hash.update(data),
            SequenceKind::Event(hashes) => hashes.iter_mut().for_each(|hash| hash.update(data)),
        }
    }
}

pub(super) struct Sequence {
    pub handle: u32,
    pub auth: Vec<u8>,
    kind: SequenceKind,
}

/// A command whose header and sessions have been taken apart.
struct Command<'a> {
    handles: Vec<u32>,
    parameters: Reader<'a>,
}

/// A handler's answer: response handles, then the parameter area.
pub(super) struct Response {
    pub handles: Vec<u32>,
    pub parameters: Writer,
}

impl Response {
    pub fn empty() -> Self {
        Response {
            handles: Vec::new(),
            parameters: Writer::new(),
        }
    }

    pub fn with(parameters: Writer) -> Self {
        Response {
            handles: Vec::new(),
            parameters,
        }
    }
}

pub(crate) struct Tpm {
    pub(super) started: bool,
    /// One row per entry of `PCR_BANKS`.
    pub(super) banks: Vec<Vec<Vec<u8>>>,
    pub(super) pcr_update_counter: u32,
    pub(super) sequences: Vec<Sequence>,
    /// Loaded objects: transient ones, and every persistent one (loaded from the store).
    pub(super) objects: Vec<Object>,
    pub(super) sessions: Vec<AuthSession>,
    pub(super) nv: Vec<NvIndex>,
    next_transient: u32,
    pub(super) next_session: u32,
    pub(super) random: Random,
    /// Virtual time at `_TPM_Init`, for `TPM2_ReadClock`.
    init_ns: u64,
    pub(super) store: SharedKeyValueStore,
    pub(super) state: PersistentState,
    /// The null hierarchy's seed and proof, new at every TPM Reset.
    pub(super) null: Hierarchy,
    /// The platform authorization is volatile too: empty after every TPM Reset.
    pub(super) platform_auth: Vec<u8>,
    pub(super) context_secret: Vec<u8>,
    /// `TPMS_CONTEXT.sequence` of the next saved context.
    pub(super) context_sequence: u64,
}

/// `parameters` ran out: which parameter is unknown, so the code names none.
impl From<Truncated> for u32 {
    fn from(_: Truncated) -> u32 {
        RC_INSUFFICIENT
    }
}

pub(super) fn end_of(parameters: &Reader) -> Result<(), u32> {
    if parameters.remaining() == 0 {
        Ok(())
    } else {
        Err(RC_SIZE)
    }
}

fn error_response(code: u32) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.u16(ST_NO_SESSIONS);
    writer.u32(HEADER_BYTES as u32);
    writer.u32(code);
    writer.bytes
}

fn initial_pcr(index: usize, length: usize) -> Vec<u8> {
    let fill = if (PCR_DRTM_FIRST..=PCR_DRTM_LAST).contains(&index) {
        0xff
    } else {
        0
    };
    vec![fill; length]
}

pub(super) fn pcr_bitmap(pcrs: impl Iterator<Item = usize>) -> [u8; PCR_SELECT_BYTES] {
    let mut bitmap = [0u8; PCR_SELECT_BYTES];
    for pcr in pcrs {
        bitmap[pcr / 8] |= 1 << (pcr % 8);
    }
    bitmap
}

fn write_ticket(writer: &mut Writer) {
    writer.u16(ST_HASHCHECK);
    writer.u32(RH_NULL);
    writer.sized(&[]);
}

fn read_hierarchy(parameters: &mut Reader, number: u32) -> Result<(), u32> {
    match parameters.u32()? {
        RH_OWNER | RH_ENDORSEMENT | RH_PLATFORM | RH_NULL => Ok(()),
        _ => Err(parameter_error(RC_VALUE, number)),
    }
}

fn read_max_buffer<'a>(parameters: &mut Reader<'a>, number: u32) -> Result<&'a [u8], u32> {
    let data = parameters.sized()?;
    if data.len() > MAX_BUFFER_BYTES {
        return Err(parameter_error(RC_SIZE, number));
    }
    Ok(data)
}

impl Tpm {
    pub fn new(random: Random, now_ns: u64, store: SharedKeyValueStore) -> Self {
        let empty = Hierarchy {
            seed: Vec::new(),
            proof: Vec::new(),
        };
        let mut tpm = Tpm {
            started: false,
            banks: Vec::new(),
            pcr_update_counter: 0,
            sequences: Vec::new(),
            objects: Vec::new(),
            sessions: Vec::new(),
            nv: Vec::new(),
            next_transient: TRANSIENT_FIRST,
            next_session: 0,
            random,
            init_ns: now_ns,
            store,
            state: PersistentState {
                platform: empty.clone(),
                endorsement: empty.clone(),
                storage: empty.clone(),
                owner_auth: Vec::new(),
                endorsement_auth: Vec::new(),
                lockout_auth: Vec::new(),
            },
            null: empty,
            platform_auth: Vec::new(),
            context_secret: Vec::new(),
            context_sequence: 0,
        };
        tpm.load_persistent_state();
        tpm
    }

    /// Runs one command and returns the whole response. Never fails: every problem with the
    /// command is a response code.
    pub fn execute(&mut self, command: &[u8], now_ns: u64) -> Vec<u8> {
        self.dispatch(command, now_ns)
            .unwrap_or_else(error_response)
    }

    fn dispatch(&mut self, bytes: &[u8], now_ns: u64) -> Result<Vec<u8>, u32> {
        let mut reader = Reader::new(bytes);
        let (tag, size, code) =
            (|| Ok::<_, Truncated>((reader.u16()?, reader.u32()?, reader.u32()?)))()
                .map_err(|_| RC_COMMAND_SIZE)?;
        if tag != ST_NO_SESSIONS && tag != ST_SESSIONS {
            return Err(RC_BAD_TAG);
        }
        if size as usize != bytes.len() {
            return Err(RC_COMMAND_SIZE);
        }
        let command = COMMANDS
            .iter()
            .find(|command| command.code == code)
            .ok_or(RC_COMMAND_CODE)?;
        if self.started == (code == CC_STARTUP) {
            return Err(RC_INITIALIZE);
        }
        let mut handles = Vec::new();
        for _ in 0..command.handles {
            handles.push(reader.u32().map_err(|_| RC_COMMAND_SIZE)?);
        }
        let sessions = if tag == ST_SESSIONS {
            let area_size = reader.u32().map_err(|_| RC_AUTHSIZE)? as usize;
            let area = reader.take(area_size).map_err(|_| RC_AUTHSIZE)?;
            Self::parse_sessions(area)?
        } else {
            Vec::new()
        };
        // Handles are checked before authorization, as the reference implementation does: a
        // flushed sequence is a missing handle, not a failed password.
        for (index, handle) in handles.iter().enumerate() {
            self.check_handle(*handle, index)?;
        }
        let mut authorization = self.authorize(command, &handles, sessions, reader.rest())?;
        let parameters = core::mem::take(&mut authorization.parameters);
        let response = self.run(
            code,
            Command {
                handles: handles.clone(),
                parameters: Reader::new(&parameters),
            },
            now_ns,
        )?;
        Ok(self.respond(tag, command, &handles, authorization, response))
    }

    /// The reference implementation's `EntityGetLoadStatus`: a transient object or session
    /// that is not loaded is `TPM_RC_REFERENCE_H0` plus the handle's position (Linux's resource
    /// manager reads that as "flushed"), anything else that does not exist is `TPM_RC_HANDLE`.
    fn check_handle(&self, handle: u32, index: usize) -> Result<(), u32> {
        let missing = handle_error(RC_HANDLE, index as u32 + 1);
        let exists = match handle_type(handle) {
            HT_PCR => (handle as usize) < PCR_COUNT,
            HT_PERMANENT => PERMANENT_HANDLES.contains(&handle),
            HT_PERSISTENT => self.object(handle).is_some(),
            HT_NV_INDEX => self.nv_index(handle).is_some(),
            HT_TRANSIENT => {
                let loaded = self.object(handle).is_some()
                    || self
                        .sequences
                        .iter()
                        .any(|sequence| sequence.handle == handle);
                if !loaded {
                    return Err(RC_REFERENCE_H0 + index as u32);
                }
                true
            }
            HT_HMAC_SESSION | HT_POLICY_SESSION => {
                if self.loaded_session(handle).is_none() {
                    return Err(RC_REFERENCE_H0 + index as u32);
                }
                true
            }
            _ => false,
        };
        if exists { Ok(()) } else { Err(missing) }
    }

    fn parse_sessions(area: &[u8]) -> Result<Vec<CommandSession>, u32> {
        let mut reader = Reader::new(area);
        let mut sessions = Vec::new();
        while reader.remaining() > 0 {
            if sessions.len() == MAX_SESSIONS {
                return Err(RC_AUTHSIZE);
            }
            sessions.push(CommandSession::read(&mut reader).map_err(|_| RC_AUTHSIZE)?);
        }
        Ok(sessions)
    }

    pub(super) fn object(&self, handle: u32) -> Option<&Object> {
        self.objects.iter().find(|object| object.handle == handle)
    }

    fn transient_count(&self) -> usize {
        self.sequences.len()
            + self
                .objects
                .iter()
                .filter(|object| handle_type(object.handle) == HT_TRANSIENT)
                .count()
    }

    pub(super) fn transient_available(&self) -> usize {
        TRANSIENT_SLOTS.saturating_sub(self.transient_count())
    }

    pub(super) fn check_transient_room(&self) -> Result<(), u32> {
        if self.transient_count() >= TRANSIENT_SLOTS {
            return Err(RC_OBJECT_MEMORY);
        }
        Ok(())
    }

    /// A transient handle nothing loaded uses. Handles are not reused right away, so a stale
    /// handle from a flushed object fails instead of reaching the next one.
    pub(super) fn allocate_transient(&mut self) -> u32 {
        loop {
            let handle = self.next_transient;
            self.next_transient = if handle == u32::MAX {
                TRANSIENT_FIRST
            } else {
                handle + 1
            };
            let used = self
                .sequences
                .iter()
                .any(|sequence| sequence.handle == handle)
                || self.object(handle).is_some();
            if !used {
                return handle;
            }
        }
    }

    fn run(&mut self, code: u32, mut command: Command, now_ns: u64) -> Result<Response, u32> {
        let handles = &command.handles;
        let parameters = &mut command.parameters;
        match code {
            CC_STARTUP => self.startup(parameters),
            CC_SHUTDOWN => {
                let kind = parameters.u16()?;
                end_of(parameters)?;
                if kind != SU_CLEAR && kind != SU_STATE {
                    return Err(parameter_error(RC_VALUE, 1));
                }
                Ok(Response::empty())
            }
            CC_SELF_TEST => {
                parameters.u8()?;
                end_of(parameters)?;
                Ok(Response::empty())
            }
            CC_INCREMENTAL_SELF_TEST => {
                let count = parameters.u32()? as usize;
                parameters.take(count.saturating_mul(2))?;
                end_of(parameters)?;
                // Everything was tested at start-up: nothing is left to do.
                let mut writer = Writer::new();
                writer.u32(0);
                Ok(Response::with(writer))
            }
            CC_GET_TEST_RESULT => {
                end_of(parameters)?;
                let mut writer = Writer::new();
                writer.sized(&[]);
                writer.u32(RC_SUCCESS);
                Ok(Response::with(writer))
            }
            CC_GET_RANDOM => {
                let requested = parameters.u16()? as usize;
                end_of(parameters)?;
                let mut bytes = vec![0u8; requested.min(MAX_DIGEST_BYTES)];
                self.random.fill(&mut bytes);
                let mut writer = Writer::new();
                writer.sized(&bytes);
                Ok(Response::with(writer))
            }
            CC_STIR_RANDOM => {
                let data = parameters.sized()?;
                end_of(parameters)?;
                if data.len() > MAX_BUFFER_BYTES {
                    return Err(parameter_error(RC_SIZE, 1));
                }
                Ok(Response::empty())
            }
            CC_GET_CAPABILITY => self.get_capability(parameters),
            CC_TEST_PARMS => self.test_parms(parameters),
            CC_PCR_READ => self.pcr_read(parameters),
            CC_PCR_EXTEND => self.pcr_extend(handles[0], parameters),
            CC_PCR_EVENT => self.pcr_event(handles[0], parameters),
            CC_PCR_RESET => {
                end_of(parameters)?;
                self.pcr_reset(handles[0])
            }
            CC_HASH => {
                let data = read_max_buffer(parameters, 1)?;
                let algorithm = parameters.u16()?;
                read_hierarchy(parameters, 3)?;
                end_of(parameters)?;
                let digest = hash::digest(algorithm, &[data]).ok_or(parameter_error(RC_HASH, 2))?;
                let mut writer = Writer::new();
                writer.sized(&digest);
                write_ticket(&mut writer);
                Ok(Response::with(writer))
            }
            CC_HASH_SEQUENCE_START => self.hash_sequence_start(parameters),
            CC_SEQUENCE_UPDATE => {
                let data = read_max_buffer(parameters, 1)?;
                end_of(parameters)?;
                self.sequence(handles[0], 1)?.update(data);
                Ok(Response::empty())
            }
            CC_SEQUENCE_COMPLETE => self.sequence_complete(handles[0], parameters),
            CC_EVENT_SEQUENCE_COMPLETE => {
                self.event_sequence_complete(handles[0], handles[1], parameters)
            }
            CC_FLUSH_CONTEXT => self.flush_context(parameters),
            CC_CONTEXT_SAVE => self.context_save(handles[0], parameters),
            CC_CONTEXT_LOAD => self.context_load(parameters),
            CC_READ_CLOCK => {
                end_of(parameters)?;
                let milliseconds = now_ns.saturating_sub(self.init_ns) / NS_PER_MS;
                let mut writer = Writer::new();
                writer.u64(milliseconds);
                writer.u64(milliseconds);
                // resetCount and restartCount: no count of resets is kept, and nothing resumes.
                writer.u32(0);
                writer.u32(0);
                writer.u8(CLOCK_SAFE);
                Ok(Response::with(writer))
            }
            CC_CREATE_PRIMARY => self.create_primary(handles[0], parameters),
            CC_CREATE => self.create(handles[0], parameters),
            CC_LOAD => self.load(handles[0], parameters),
            CC_READ_PUBLIC => self.read_public(handles[0], parameters),
            CC_UNSEAL => self.unseal(handles[0], parameters),
            CC_EVICT_CONTROL => self.evict_control(handles[0], handles[1], parameters),
            CC_ECDH_KEYGEN => self.ecdh_key_gen(handles[0], parameters),
            CC_ECDH_ZGEN => self.ecdh_z_gen(handles[0], parameters),
            CC_START_AUTH_SESSION => self.start_auth_session(handles[0], handles[1], parameters),
            CC_POLICY_PCR => self.policy_pcr(handles[0], parameters),
            CC_POLICY_AUTH_VALUE => self.policy_auth_value(handles[0], parameters, false),
            CC_POLICY_PASSWORD => self.policy_auth_value(handles[0], parameters, true),
            CC_POLICY_GET_DIGEST => self.policy_get_digest(handles[0], parameters),
            CC_POLICY_RESTART => self.policy_restart(handles[0], parameters),
            CC_NV_DEFINE_SPACE => self.nv_define_space(handles[0], parameters),
            CC_NV_UNDEFINE_SPACE => self.nv_undefine_space(handles[0], handles[1], parameters),
            CC_NV_READ_PUBLIC => self.nv_read_public(handles[0], parameters),
            CC_NV_READ => self.nv_read(handles[1], parameters),
            CC_NV_WRITE => self.nv_write(handles[1], parameters),
            CC_CLEAR => self.clear(handles[0], parameters),
            CC_HIERARCHY_CHANGE_AUTH => self.hierarchy_change_auth(handles[0], parameters),
            CC_DICTIONARY_ATTACK_LOCK_RESET => {
                if handles[0] != RH_LOCKOUT {
                    return Err(handle_error(RC_VALUE, 1));
                }
                self.dictionary_attack_lock_reset(parameters)
            }
            _ => Err(RC_COMMAND_CODE),
        }
    }

    fn startup(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let kind = parameters.u16()?;
        end_of(parameters)?;
        // There is never saved state to resume: a TPM2_Shutdown(STATE) does not survive the
        // guest reset that would precede a TPM2_Startup(STATE).
        if kind != SU_CLEAR {
            return Err(parameter_error(RC_VALUE, 1));
        }
        self.banks = PCR_BANKS
            .iter()
            .map(|algorithm| {
                let length = hash::digest_len(*algorithm).unwrap_or(0);
                (0..PCR_COUNT)
                    .map(|index| initial_pcr(index, length))
                    .collect()
            })
            .collect();
        self.pcr_update_counter = 0;
        // A TPM Reset: the null hierarchy, the platform authorization and the context keys start
        // over, and nothing transient survives.
        self.sequences.clear();
        self.sessions.clear();
        self.objects
            .retain(|object| handle_type(object.handle) == HT_PERSISTENT);
        let mut null = PersistentState::manufacture(&mut self.random).storage;
        core::mem::swap(&mut self.null, &mut null);
        self.platform_auth.clear();
        self.context_secret = vec![0; CONTEXT_SECRET_BYTES];
        self.random.fill(&mut self.context_secret);
        self.nv_startup();
        self.started = true;
        Ok(Response::empty())
    }

    /// A `TPML_PCR_SELECTION`. Selections of banks that are not allocated are kept (they select
    /// nothing); a hash this TPM does not implement is `TPM_RC_HASH` on parameter `number`.
    pub(super) fn read_pcr_selection(
        &self,
        parameters: &mut Reader,
        number: u32,
    ) -> Result<Vec<(u16, Vec<u8>)>, u32> {
        let count = parameters.u32()? as usize;
        if count > HASH_ALGORITHMS.len() {
            return Err(parameter_error(RC_SIZE, number));
        }
        let mut selections = Vec::new();
        for _ in 0..count {
            let algorithm = parameters.u16()?;
            let select_bytes = parameters.u8()? as usize;
            let bitmap = parameters.take(select_bytes)?;
            if select_bytes > PCR_SELECT_BYTES {
                return Err(parameter_error(RC_VALUE, number));
            }
            hash::digest_len(algorithm).ok_or(parameter_error(RC_HASH, number))?;
            selections.push((algorithm, bitmap.to_vec()));
        }
        Ok(selections)
    }

    pub(super) fn write_pcr_selection(writer: &mut Writer, selections: &[(u16, Vec<u8>)]) {
        writer.u32(selections.len() as u32);
        for (algorithm, bitmap) in selections {
            writer.u16(*algorithm);
            writer.u8(bitmap.len() as u8);
            writer.bytes(bitmap);
        }
    }

    /// The digest of the selected PCRs' values, in selection order and by index within each
    /// (Part 1, "PCR selection"), under `hash_alg`.
    pub(super) fn pcr_digest(&self, hash_alg: u16, selections: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut values: Vec<&[u8]> = Vec::new();
        for (algorithm, bitmap) in selections {
            let Some(bank) = PCR_BANKS.iter().position(|bank| bank == algorithm) else {
                continue;
            };
            for index in 0..(bitmap.len() * 8).min(PCR_COUNT) {
                if bitmap[index / 8] & (1 << (index % 8)) != 0 {
                    values.push(&self.banks[bank][index]);
                }
            }
        }
        hash::digest(hash_alg, &values).unwrap_or_default()
    }

    /// A PCR handle the guest may change from locality 0, or `None` for `TPM_RH_NULL` (the
    /// command runs and changes nothing).
    fn writable_pcr(handle: u32) -> Result<Option<usize>, u32> {
        if handle == RH_NULL {
            return Ok(None);
        }
        let index = handle as usize;
        if handle_type(handle) != HT_PCR || index >= PCR_COUNT {
            return Err(handle_error(RC_HANDLE, 1));
        }
        if (PCR_DRTM_FIRST..=PCR_DRTM_LAST).contains(&index) {
            return Err(RC_LOCALITY);
        }
        Ok(Some(index))
    }

    fn counts_updates(index: usize) -> bool {
        index != PCR_DEBUG && index != PCR_APPLICATION
    }

    fn extend(&mut self, index: usize, algorithm: u16, digest: &[u8]) {
        let Some(bank) = PCR_BANKS.iter().position(|bank| *bank == algorithm) else {
            return;
        };
        let pcr = &mut self.banks[bank][index];
        if let Some(extended) = hash::digest(algorithm, &[pcr, digest]) {
            *pcr = extended;
        }
    }

    fn note_update(&mut self, index: usize) {
        if Self::counts_updates(index) {
            self.pcr_update_counter = self.pcr_update_counter.wrapping_add(1);
        }
    }

    fn pcr_extend(&mut self, handle: u32, parameters: &mut Reader) -> Result<Response, u32> {
        let pcr = Self::writable_pcr(handle)?;
        let count = parameters.u32()? as usize;
        if count > HASH_ALGORITHMS.len() {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let mut digests = Vec::new();
        for _ in 0..count {
            let algorithm = parameters.u16()?;
            let length = hash::digest_len(algorithm).ok_or(parameter_error(RC_HASH, 1))?;
            digests.push((algorithm, parameters.take(length)?));
        }
        end_of(parameters)?;
        if let Some(index) = pcr {
            for (algorithm, digest) in digests {
                self.extend(index, algorithm, digest);
            }
            self.note_update(index);
        }
        Ok(Response::empty())
    }

    /// Extends one digest per implemented algorithm into `pcr` (each allocated bank takes its
    /// own) and returns the `TPML_DIGEST_VALUES` of all of them.
    fn extend_all(&mut self, pcr: Option<usize>, digests: Vec<(u16, Vec<u8>)>) -> Writer {
        let mut writer = Writer::new();
        writer.u32(digests.len() as u32);
        for (algorithm, digest) in &digests {
            writer.u16(*algorithm);
            writer.bytes(digest);
        }
        if let Some(index) = pcr {
            for (algorithm, digest) in &digests {
                self.extend(index, *algorithm, digest);
            }
            self.note_update(index);
        }
        writer
    }

    fn pcr_event(&mut self, handle: u32, parameters: &mut Reader) -> Result<Response, u32> {
        let pcr = Self::writable_pcr(handle)?;
        let data = parameters.sized()?;
        end_of(parameters)?;
        if data.len() > MAX_EVENT_BYTES {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let digests = HASH_ALGORITHMS
            .iter()
            .filter_map(|algorithm| Some((*algorithm, hash::digest(*algorithm, &[data])?)))
            .collect();
        Ok(Response::with(self.extend_all(pcr, digests)))
    }

    fn pcr_reset(&mut self, handle: u32) -> Result<Response, u32> {
        let Some(index) = Self::writable_pcr(handle)? else {
            return Ok(Response::empty());
        };
        if index != PCR_DEBUG && index != PCR_APPLICATION {
            return Err(RC_LOCALITY);
        }
        for (bank, algorithm) in PCR_BANKS.iter().enumerate() {
            let length = hash::digest_len(*algorithm).unwrap_or(0);
            self.banks[bank][index] = initial_pcr(index, length);
        }
        self.note_update(index);
        Ok(Response::empty())
    }

    fn pcr_read(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let count = parameters.u32()? as usize;
        if count > HASH_ALGORITHMS.len() {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let mut selections = Vec::new();
        for _ in 0..count {
            let algorithm = parameters.u16()?;
            let select_bytes = parameters.u8()? as usize;
            let bitmap = parameters.take(select_bytes)?;
            if select_bytes > PCR_SELECT_BYTES {
                return Err(parameter_error(RC_VALUE, 1));
            }
            hash::digest_len(algorithm).ok_or(parameter_error(RC_HASH, 1))?;
            selections.push((algorithm, bitmap));
        }
        end_of(parameters)?;
        // Selections are answered in order until eight digests are out; the selection that is
        // returned says which ones those were.
        let mut digests = Vec::new();
        let mut answered = Writer::new();
        answered.u32(selections.len() as u32);
        for (algorithm, bitmap) in selections {
            let bank = PCR_BANKS.iter().position(|bank| *bank == algorithm);
            let mut taken = [0u8; PCR_SELECT_BYTES];
            for index in 0..bitmap.len() * 8 {
                let selected = bitmap[index / 8] & (1 << (index % 8)) != 0;
                if !selected || index >= PCR_COUNT || digests.len() == PCR_READ_MAX_DIGESTS {
                    continue;
                }
                if let Some(bank) = bank {
                    digests.push(self.banks[bank][index].clone());
                    taken[index / 8] |= 1 << (index % 8);
                }
            }
            answered.u16(algorithm);
            answered.u8(PCR_SELECT_BYTES as u8);
            answered.bytes(&taken);
        }
        let mut writer = Writer::new();
        writer.u32(self.pcr_update_counter);
        writer.bytes(&answered.bytes);
        writer.u32(digests.len() as u32);
        for digest in &digests {
            writer.sized(digest);
        }
        Ok(Response::with(writer))
    }

    fn hash_sequence_start(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let auth = parameters.sized()?.to_vec();
        let algorithm = parameters.u16()?;
        end_of(parameters)?;
        if auth.len() > MAX_DIGEST_BYTES {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let kind = if algorithm == ALG_NULL {
            SequenceKind::Event(
                HASH_ALGORITHMS
                    .iter()
                    .filter_map(|algorithm| Hash::new(*algorithm))
                    .collect(),
            )
        } else {
            SequenceKind::Hash(Box::new(
                Hash::new(algorithm).ok_or(parameter_error(RC_HASH, 2))?,
            ))
        };
        self.check_transient_room()?;
        let handle = self.allocate_transient();
        self.sequences.push(Sequence { handle, auth, kind });
        Ok(Response {
            handles: vec![handle],
            parameters: Writer::new(),
        })
    }

    fn sequence(&mut self, handle: u32, number: u32) -> Result<&mut SequenceKind, u32> {
        self.sequences
            .iter_mut()
            .find(|sequence| sequence.handle == handle)
            .map(|sequence| &mut sequence.kind)
            .ok_or(handle_error(RC_HANDLE, number))
    }

    fn take_sequence(&mut self, handle: u32, number: u32) -> Result<SequenceKind, u32> {
        let position = self
            .sequences
            .iter()
            .position(|sequence| sequence.handle == handle)
            .ok_or(handle_error(RC_HANDLE, number))?;
        Ok(self.sequences.remove(position).kind)
    }

    fn sequence_complete(&mut self, handle: u32, parameters: &mut Reader) -> Result<Response, u32> {
        let data = read_max_buffer(parameters, 1)?;
        read_hierarchy(parameters, 2)?;
        end_of(parameters)?;
        if !matches!(self.sequence(handle, 1)?, SequenceKind::Hash(_)) {
            return Err(handle_error(RC_SEQUENCE, 1));
        }
        let SequenceKind::Hash(mut hash) = self.take_sequence(handle, 1)? else {
            unreachable!("checked above");
        };
        hash.update(data);
        let mut writer = Writer::new();
        writer.sized(&hash.finish());
        write_ticket(&mut writer);
        Ok(Response::with(writer))
    }

    fn event_sequence_complete(
        &mut self,
        pcr_handle: u32,
        sequence_handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let data = read_max_buffer(parameters, 1)?;
        end_of(parameters)?;
        let pcr = Self::writable_pcr(pcr_handle)?;
        if !matches!(self.sequence(sequence_handle, 2)?, SequenceKind::Event(_)) {
            return Err(handle_error(RC_SEQUENCE, 2));
        }
        let SequenceKind::Event(hashes) = self.take_sequence(sequence_handle, 2)? else {
            unreachable!("checked above");
        };
        let digests = hashes
            .into_iter()
            .map(|mut hash| {
                hash.update(data);
                (hash.id, hash.finish())
            })
            .collect();
        Ok(Response::with(self.extend_all(pcr, digests)))
    }

    #[cfg(test)]
    pub fn pcr(&self, index: usize) -> Option<&[u8]> {
        self.banks.first().map(|bank| bank[index].as_slice())
    }
}
