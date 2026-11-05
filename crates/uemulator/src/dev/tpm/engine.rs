//! The TPM 2.0 itself: commands in, responses out (TPM 2.0 Library, Parts 2 and 3, revision
//! 1.59). This is the measured-boot subset: start-up and self tests, capabilities, random
//! numbers, PCRs, hashing and hash sequences, the clock. Nothing creates or loads keys, so there
//! are no objects, no HMAC or policy sessions and no NV indices; every other command answers
//! `TPM_RC_COMMAND_CODE` and is left out of `TPM_CAP_COMMANDS`, which is how Linux and the
//! TSS libraries find out what a TPM can do. Only locality 0 exists (the CRB interface offers
//! no other), so the DRTM PCRs 17..=22 can be read but never extended or reset.
//!
//! The state lives as long as the device: a guest reset builds a new one, which is a TPM reset
//! (`_TPM_Init`), so nothing is persisted.
#[cfg(freestanding)]
use crate::prelude::*;

use super::hash::{self, ALG_NULL, ALG_SHA256, HASH_ALGORITHMS, Hash, MAX_DIGEST_BYTES};
use super::wire::{Reader, Truncated, Writer};
use crate::cpu::xsave::splitmix64;
use crate::dev::RngSource;

const ST_NO_SESSIONS: u16 = 0x8001;
const ST_SESSIONS: u16 = 0x8002;
const ST_HASHCHECK: u16 = 0x8024;
const HEADER_BYTES: usize = 10;

pub(crate) const RC_SUCCESS: u32 = 0x000;
const RC_BAD_TAG: u32 = 0x01e;
const RC_INITIALIZE: u32 = 0x100;
const RC_SEQUENCE: u32 = 0x103;
const RC_AUTH_MISSING: u32 = 0x125;
const RC_COMMAND_SIZE: u32 = 0x142;
const RC_COMMAND_CODE: u32 = 0x143;
const RC_AUTHSIZE: u32 = 0x144;
/// Format-one codes take which handle, parameter or session was wrong in the bits below.
const RC_HASH: u32 = 0x083;
const RC_VALUE: u32 = 0x084;
const RC_HANDLE: u32 = 0x08b;
const RC_AUTH_FAIL: u32 = 0x08e;
const RC_SIZE: u32 = 0x095;
const RC_INSUFFICIENT: u32 = 0x09a;
const RC_PARAMETER: u32 = 0x040;
const RC_SESSION: u32 = 0x800;
const RC_NUMBER_SHIFT: u32 = 8;
const RC_OBJECT_MEMORY: u32 = 0x902;
const RC_LOCALITY: u32 = 0x907;
/// `TPM_RC_REFERENCE_S0`: the first session handle names a session that is not loaded; the
/// following codes are the later sessions.
const RC_REFERENCE_S0: u32 = 0x910;

fn handle_error(code: u32, number: u32) -> u32 {
    code | number << RC_NUMBER_SHIFT
}

fn parameter_error(code: u32, number: u32) -> u32 {
    code | RC_PARAMETER | number << RC_NUMBER_SHIFT
}

fn session_error(code: u32, number: u32) -> u32 {
    code | RC_SESSION | number << RC_NUMBER_SHIFT
}

const CC_INCREMENTAL_SELF_TEST: u32 = 0x142;
const CC_SELF_TEST: u32 = 0x143;
const CC_STARTUP: u32 = 0x144;
const CC_SHUTDOWN: u32 = 0x145;
const CC_STIR_RANDOM: u32 = 0x146;
const CC_PCR_EVENT: u32 = 0x13c;
const CC_PCR_RESET: u32 = 0x13d;
const CC_SEQUENCE_COMPLETE: u32 = 0x13e;
const CC_SEQUENCE_UPDATE: u32 = 0x15c;
const CC_FLUSH_CONTEXT: u32 = 0x165;
const CC_GET_CAPABILITY: u32 = 0x17a;
const CC_GET_RANDOM: u32 = 0x17b;
const CC_GET_TEST_RESULT: u32 = 0x17c;
const CC_HASH: u32 = 0x17d;
const CC_PCR_READ: u32 = 0x17e;
const CC_READ_CLOCK: u32 = 0x181;
const CC_PCR_EXTEND: u32 = 0x182;
const CC_EVENT_SEQUENCE_COMPLETE: u32 = 0x185;
const CC_HASH_SEQUENCE_START: u32 = 0x186;

/// What `TPMA_CC` says about a command, which Linux's resource manager depends on: it moves
/// `handles` and `response_handles` in and out of its virtual handle space.
struct CommandSpec {
    code: u32,
    /// Handles in the command's handle area.
    handles: u8,
    /// How many of those (from the first) need an authorization session.
    authorized: u8,
    response_handles: u8,
    /// The command flushes the transient object it names (the sequence).
    flushes: bool,
}

const fn spec(code: u32, handles: u8, authorized: u8, response_handles: u8) -> CommandSpec {
    CommandSpec {
        code,
        handles,
        authorized,
        response_handles,
        flushes: false,
    }
}

/// Sorted by code: `TPM_CAP_COMMANDS` lists them in this order.
const COMMANDS: [CommandSpec; 19] = [
    spec(CC_PCR_EVENT, 1, 1, 0),
    spec(CC_PCR_RESET, 1, 1, 0),
    CommandSpec {
        flushes: true,
        ..spec(CC_SEQUENCE_COMPLETE, 1, 1, 0)
    },
    spec(CC_INCREMENTAL_SELF_TEST, 0, 0, 0),
    spec(CC_SELF_TEST, 0, 0, 0),
    spec(CC_STARTUP, 0, 0, 0),
    spec(CC_SHUTDOWN, 0, 0, 0),
    spec(CC_STIR_RANDOM, 0, 0, 0),
    spec(CC_SEQUENCE_UPDATE, 1, 1, 0),
    spec(CC_FLUSH_CONTEXT, 0, 0, 0),
    spec(CC_GET_CAPABILITY, 0, 0, 0),
    spec(CC_GET_RANDOM, 0, 0, 0),
    spec(CC_GET_TEST_RESULT, 0, 0, 0),
    spec(CC_HASH, 0, 0, 0),
    spec(CC_PCR_READ, 0, 0, 0),
    spec(CC_READ_CLOCK, 0, 0, 0),
    spec(CC_PCR_EXTEND, 1, 1, 0),
    CommandSpec {
        flushes: true,
        ..spec(CC_EVENT_SEQUENCE_COMPLETE, 2, 2, 0)
    },
    spec(CC_HASH_SEQUENCE_START, 0, 0, 1),
];

const TPMA_CC_COMMAND_INDEX_MASK: u32 = 0xffff;
const TPMA_CC_FLUSHED: u32 = 1 << 24;
const TPMA_CC_C_HANDLES_SHIFT: u32 = 25;
const TPMA_CC_R_HANDLE: u32 = 1 << 28;

fn command_attributes(command: &CommandSpec) -> u32 {
    (command.code & TPMA_CC_COMMAND_INDEX_MASK)
        | if command.flushes { TPMA_CC_FLUSHED } else { 0 }
        | (command.handles as u32) << TPMA_CC_C_HANDLES_SHIFT
        | if command.response_handles > 0 {
            TPMA_CC_R_HANDLE
        } else {
            0
        }
}

const SU_CLEAR: u16 = 0x0000;
const SU_STATE: u16 = 0x0001;

pub(crate) const PCR_COUNT: usize = 24;
/// Bytes of a PCR bitmap: `TPM_PT_PCR_SELECT_MIN`.
const PCR_SELECT_BYTES: usize = PCR_COUNT / 8;
/// The PC Client profile's resettable PCRs: 16 (debug) and 23 (application support).
const PCR_DEBUG: usize = 16;
const PCR_APPLICATION: usize = 23;
/// 17..=22 belong to a dynamic launch: they start as all ones and need locality 2 or above.
const PCR_DRTM_FIRST: usize = 17;
const PCR_DRTM_LAST: usize = 22;
/// The banks that are allocated. Only SHA-256, as swtpm gives QEMU by default; a bank that
/// is not allocated is not reported by `TPM_CAP_PCRS` at all.
const PCR_BANKS: [u16; 1] = [ALG_SHA256];
/// `TPML_DIGEST` holds at most eight digests, so `TPM2_PCR_Read` answers at most eight PCRs.
const PCR_READ_MAX_DIGESTS: usize = 8;

const HT_PCR: u8 = 0x00;
const HT_NV_INDEX: u8 = 0x01;
const HT_HMAC_SESSION: u8 = 0x02;
const HT_POLICY_SESSION: u8 = 0x03;
const HT_PERMANENT: u8 = 0x40;
const HT_TRANSIENT: u8 = 0x80;
const HT_PERSISTENT: u8 = 0x81;
const HANDLE_TYPE_SHIFT: u32 = 24;

const RH_OWNER: u32 = 0x4000_0001;
/// The password session: authorization by a plaintext value, here always the empty one.
const RS_PW: u32 = 0x4000_0009;
const RH_NULL: u32 = 0x4000_0007;
const RH_LOCKOUT: u32 = 0x4000_000a;
const RH_ENDORSEMENT: u32 = 0x4000_000b;
const RH_PLATFORM: u32 = 0x4000_000c;
const RH_PLATFORM_NV: u32 = 0x4000_000d;
const PERMANENT_HANDLES: [u32; 7] = [
    RH_OWNER,
    RH_NULL,
    RS_PW,
    RH_LOCKOUT,
    RH_ENDORSEMENT,
    RH_PLATFORM,
    RH_PLATFORM_NV,
];

const TRANSIENT_FIRST: u32 = 0x8000_0000;
/// Hash sequences open at once; they are the only transient objects.
const MAX_SEQUENCES: usize = 3;
/// `TPM_PT_HR_TRANSIENT_MIN`/`TPM_PT_HR_LOADED_MIN` promise this many, and the spec's floor is 3.
const TRANSIENT_SLOTS: u32 = MAX_SEQUENCES as u32;

/// The most sessions one command may carry.
const MAX_SESSIONS: usize = 3;
/// `TPMA_SESSION.continueSession`, echoed back for the password session.
const SESSION_CONTINUE: u8 = 1 << 0;

/// `TPM2B_MAX_BUFFER`/`TPM2B_EVENT` capacity; `TPM_PT_INPUT_BUFFER`.
const MAX_BUFFER_BYTES: usize = 1024;
const MAX_EVENT_BYTES: usize = 1024;

const CAP_ALGS: u32 = 0;
const CAP_HANDLES: u32 = 1;
const CAP_COMMANDS: u32 = 2;
const CAP_PP_COMMANDS: u32 = 3;
const CAP_AUDIT_COMMANDS: u32 = 4;
const CAP_PCRS: u32 = 5;
const CAP_TPM_PROPERTIES: u32 = 6;
const CAP_PCR_PROPERTIES: u32 = 7;
const CAP_ECC_CURVES: u32 = 8;
const CAP_AUTH_POLICIES: u32 = 9;
const CAP_ACT: u32 = 10;
/// `MAX_CAP_BUFFER` less the capability and count fields: how much one answer may carry.
const MAX_CAP_DATA_BYTES: usize = 1024 - 8;

/// `TPMA_ALGORITHM.hash`.
const ALGORITHM_HASH: u32 = 1 << 2;

const PT_FAMILY_INDICATOR: u32 = 0x100;
const PT_LEVEL: u32 = 0x101;
const PT_REVISION: u32 = 0x102;
const PT_DAY_OF_YEAR: u32 = 0x103;
const PT_YEAR: u32 = 0x104;
const PT_MANUFACTURER: u32 = 0x105;
const PT_VENDOR_STRING_1: u32 = 0x106;
const PT_VENDOR_STRING_2: u32 = 0x107;
const PT_VENDOR_STRING_3: u32 = 0x108;
const PT_VENDOR_STRING_4: u32 = 0x109;
const PT_VENDOR_TPM_TYPE: u32 = 0x10a;
const PT_FIRMWARE_VERSION_1: u32 = 0x10b;
const PT_FIRMWARE_VERSION_2: u32 = 0x10c;
const PT_INPUT_BUFFER: u32 = 0x10d;
const PT_HR_TRANSIENT_MIN: u32 = 0x10e;
const PT_HR_PERSISTENT_MIN: u32 = 0x10f;
const PT_HR_LOADED_MIN: u32 = 0x110;
const PT_ACTIVE_SESSIONS_MAX: u32 = 0x111;
const PT_PCR_COUNT: u32 = 0x112;
const PT_PCR_SELECT_MIN: u32 = 0x113;
const PT_CONTEXT_GAP_MAX: u32 = 0x114;
const PT_NV_COUNTERS_MAX: u32 = 0x116;
const PT_NV_INDEX_MAX: u32 = 0x117;
const PT_MEMORY: u32 = 0x118;
const PT_CLOCK_UPDATE: u32 = 0x119;
const PT_CONTEXT_HASH: u32 = 0x11a;
const PT_CONTEXT_SYM: u32 = 0x11b;
const PT_CONTEXT_SYM_SIZE: u32 = 0x11c;
const PT_ORDERLY_COUNT: u32 = 0x11d;
const PT_MAX_COMMAND_SIZE: u32 = 0x11e;
const PT_MAX_RESPONSE_SIZE: u32 = 0x11f;
const PT_MAX_DIGEST: u32 = 0x120;
const PT_MAX_OBJECT_CONTEXT: u32 = 0x121;
const PT_MAX_SESSION_CONTEXT: u32 = 0x122;
const PT_PS_FAMILY_INDICATOR: u32 = 0x123;
const PT_PS_LEVEL: u32 = 0x124;
const PT_PS_REVISION: u32 = 0x125;
const PT_PS_DAY_OF_YEAR: u32 = 0x126;
const PT_PS_YEAR: u32 = 0x127;
const PT_SPLIT_MAX: u32 = 0x128;
const PT_TOTAL_COMMANDS: u32 = 0x129;
const PT_LIBRARY_COMMANDS: u32 = 0x12a;
const PT_VENDOR_COMMANDS: u32 = 0x12b;
const PT_NV_BUFFER_MAX: u32 = 0x12c;
const PT_MODES: u32 = 0x12d;
const PT_MAX_CAP_BUFFER: u32 = 0x12e;
const PT_PERMANENT: u32 = 0x200;
const PT_STARTUP_CLEAR: u32 = 0x201;
const PT_HR_NV_INDEX: u32 = 0x202;
const PT_HR_LOADED: u32 = 0x203;
const PT_HR_LOADED_AVAIL: u32 = 0x204;
const PT_HR_ACTIVE: u32 = 0x205;
const PT_HR_ACTIVE_AVAIL: u32 = 0x206;
const PT_HR_TRANSIENT_AVAIL: u32 = 0x207;
const PT_HR_PERSISTENT: u32 = 0x208;
const PT_HR_PERSISTENT_AVAIL: u32 = 0x209;
const PT_NV_COUNTERS: u32 = 0x20a;
const PT_NV_COUNTERS_AVAIL: u32 = 0x20b;
const PT_ALGORITHM_SET: u32 = 0x20c;
const PT_LOADED_CURVES: u32 = 0x20d;
const PT_LOCKOUT_COUNTER: u32 = 0x20e;
const PT_MAX_AUTH_FAIL: u32 = 0x20f;
const PT_LOCKOUT_INTERVAL: u32 = 0x210;
const PT_LOCKOUT_RECOVERY: u32 = 0x211;
const PT_NV_WRITE_RECOVERY: u32 = 0x212;
const PT_AUDIT_COUNTER_0: u32 = 0x213;
const PT_AUDIT_COUNTER_1: u32 = 0x214;

/// "2.0" as `TPM_PT_FAMILY_INDICATOR` spells it.
const FAMILY_2_0: u32 = u32::from_be_bytes(*b"2.0\0");
/// Library revision 1.59 and its publication date, which is what the command set follows.
const SPEC_REVISION: u32 = 159;
const SPEC_DAY_OF_YEAR: u32 = 312;
const SPEC_YEAR: u32 = 2019;
/// Four characters, as the TCG vendor registry spells manufacturers; this one is not in it.
const MANUFACTURER: u32 = u32::from_be_bytes(*b"UEMU");
const VENDOR_STRING: [u8; 16] = *b"uemulator TPM\0\0\0";
/// `TPM_PT_VENDOR_TPM_TYPE` is the vendor's to define.
const VENDOR_TPM_TYPE: u32 = 1;
const FIRMWARE_VERSION_1: u32 = 0x0001_0000;
/// `TPM_PS_PC`: the PC Client platform specification.
const PS_FAMILY_PC_CLIENT: u32 = 1;
/// PTP revision 1.05.
const PS_REVISION: u32 = 0x105;
const CONTEXT_GAP_MAX: u32 = 0xffff;
/// How often the clock is written back to NV, in milliseconds. There is no NV; this only
/// answers the property.
const CLOCK_UPDATE_MS: u32 = 4096;
const ORDERLY_COUNT: u32 = 255;
const MAX_AUTH_FAIL: u32 = 32;
const LOCKOUT_INTERVAL_SECONDS: u32 = 7200;
const LOCKOUT_RECOVERY_SECONDS: u32 = 86400;
/// `TPMA_STARTUP_CLEAR`: the platform, storage and endorsement hierarchies and platform NV
/// are enabled.
const STARTUP_CLEAR_ENABLED: u32 = 0xf;

/// `TPM_PT_PCR_*` tags of `TPM_CAP_PCR_PROPERTIES`.
const PT_PCR_SAVE: u32 = 0x00;
const PT_PCR_EXTEND_L0: u32 = 0x01;
const PT_PCR_RESET_L0: u32 = 0x02;
const PT_PCR_NO_INCREMENT: u32 = 0x11;
const PT_PCR_DRTM_RESET: u32 = 0x12;

const NS_PER_MS: u64 = 1_000_000;
/// `TPMI_YES_NO`: the clock has not gone backwards since the last orderly shutdown.
const CLOCK_SAFE: u8 = 1;

/// The bytes `TPM2_GetRandom` hands out.
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

    fn fill(&mut self, bytes: &mut [u8]) {
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

enum SequenceKind {
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

struct Sequence {
    handle: u32,
    auth: Vec<u8>,
    kind: SequenceKind,
}

/// A command whose header and sessions have been taken apart.
struct Command<'a> {
    handles: Vec<u32>,
    parameters: Reader<'a>,
}

/// A handler's answer: response handles, then the parameter area.
struct Response {
    handles: Vec<u32>,
    parameters: Writer,
}

impl Response {
    fn empty() -> Self {
        Response {
            handles: Vec::new(),
            parameters: Writer::new(),
        }
    }

    fn with(parameters: Writer) -> Self {
        Response {
            handles: Vec::new(),
            parameters,
        }
    }
}

struct Session {
    handle: u32,
    nonce_length: usize,
    attributes: u8,
    hmac: Vec<u8>,
}

pub(crate) struct Tpm {
    started: bool,
    /// One row per entry of `PCR_BANKS`.
    banks: Vec<Vec<Vec<u8>>>,
    pcr_update_counter: u32,
    sequences: Vec<Sequence>,
    next_transient: u32,
    random: Random,
    /// Virtual time at `_TPM_Init`, for `TPM2_ReadClock`.
    init_ns: u64,
}

/// `parameters` ran out: which parameter is unknown, so the code names none.
impl From<Truncated> for u32 {
    fn from(_: Truncated) -> u32 {
        RC_INSUFFICIENT
    }
}

fn end_of(parameters: &Reader) -> Result<(), u32> {
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

fn handle_type(handle: u32) -> u8 {
    (handle >> HANDLE_TYPE_SHIFT) as u8
}

fn initial_pcr(index: usize, length: usize) -> Vec<u8> {
    let fill = if (PCR_DRTM_FIRST..=PCR_DRTM_LAST).contains(&index) {
        0xff
    } else {
        0
    };
    vec![fill; length]
}

fn pcr_bitmap(pcrs: impl Iterator<Item = usize>) -> [u8; PCR_SELECT_BYTES] {
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
    pub fn new(random: Random, now_ns: u64) -> Self {
        Tpm {
            started: false,
            banks: Vec::new(),
            pcr_update_counter: 0,
            sequences: Vec::new(),
            next_transient: TRANSIENT_FIRST,
            random,
            init_ns: now_ns,
        }
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
        // flushed sequence is TPM_RC_HANDLE, not a failed password.
        for (index, handle) in handles.iter().enumerate() {
            if !self.handle_exists(*handle) {
                return Err(handle_error(RC_HANDLE, index as u32 + 1));
            }
        }
        if sessions.len() < command.authorized as usize {
            return Err(RC_AUTH_MISSING);
        }
        self.authorize(command, &handles, &sessions)?;
        let response = self.run(
            code,
            Command {
                handles,
                parameters: reader,
            },
            now_ns,
        )?;
        let mut writer = Writer::new();
        writer.u16(tag);
        writer.u32(0);
        writer.u32(RC_SUCCESS);
        for handle in &response.handles {
            writer.u32(*handle);
        }
        if tag == ST_SESSIONS {
            writer.u32(response.parameters.bytes.len() as u32);
            writer.bytes(&response.parameters.bytes);
            for session in &sessions {
                writer.sized(&[]);
                writer.u8(session.attributes & SESSION_CONTINUE);
                writer.sized(&[]);
            }
        } else {
            writer.bytes(&response.parameters.bytes);
        }
        let length = writer.bytes.len() as u32;
        writer.bytes[2..6].copy_from_slice(&length.to_be_bytes());
        Ok(writer.bytes)
    }

    fn handle_exists(&self, handle: u32) -> bool {
        match handle_type(handle) {
            HT_PCR => (handle as usize) < PCR_COUNT,
            HT_PERMANENT => PERMANENT_HANDLES.contains(&handle),
            HT_TRANSIENT => self
                .sequences
                .iter()
                .any(|sequence| sequence.handle == handle),
            _ => false,
        }
    }

    fn parse_sessions(area: &[u8]) -> Result<Vec<Session>, u32> {
        let mut reader = Reader::new(area);
        let mut sessions = Vec::new();
        while reader.remaining() > 0 {
            if sessions.len() == MAX_SESSIONS {
                return Err(RC_AUTHSIZE);
            }
            let session = (|| {
                Ok::<_, Truncated>(Session {
                    handle: reader.u32()?,
                    nonce_length: reader.sized()?.len(),
                    attributes: reader.u8()?,
                    hmac: reader.sized()?.to_vec(),
                })
            })()
            .map_err(|_| RC_AUTHSIZE)?;
            sessions.push(session);
        }
        Ok(sessions)
    }

    /// Only password sessions exist. The PCRs' authorization value is empty; a sequence's is
    /// what `TPM2_HashSequenceStart` was given.
    fn authorize(
        &self,
        command: &CommandSpec,
        handles: &[u32],
        sessions: &[Session],
    ) -> Result<(), u32> {
        for (index, session) in sessions.iter().enumerate() {
            let number = index as u32 + 1;
            if session.handle != RS_PW {
                return Err(RC_REFERENCE_S0 + index as u32);
            }
            if session.nonce_length != 0 {
                return Err(session_error(RC_SIZE, number));
            }
            let expected: &[u8] = match handles.get(index) {
                Some(handle) if index < command.authorized as usize => self
                    .sequences
                    .iter()
                    .find(|sequence| sequence.handle == *handle)
                    .map_or(&[], |sequence| &sequence.auth),
                // A password session with nothing to authorize is not an audit or encryption
                // session; the reference implementation calls that a bad attribute.
                _ => return Err(session_error(RC_VALUE, number)),
            };
            if session.hmac != expected {
                return Err(session_error(RC_AUTH_FAIL, number));
            }
        }
        Ok(())
    }

    fn run(&mut self, code: u32, mut command: Command, now_ns: u64) -> Result<Response, u32> {
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
            CC_PCR_READ => self.pcr_read(parameters),
            CC_PCR_EXTEND => self.pcr_extend(command.handles[0], parameters),
            CC_PCR_EVENT => self.pcr_event(command.handles[0], parameters),
            CC_PCR_RESET => {
                end_of(parameters)?;
                self.pcr_reset(command.handles[0])
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
                self.sequence(command.handles[0], 1)?.update(data);
                Ok(Response::empty())
            }
            CC_SEQUENCE_COMPLETE => self.sequence_complete(command.handles[0], parameters),
            CC_EVENT_SEQUENCE_COMPLETE => {
                self.event_sequence_complete(command.handles[0], command.handles[1], parameters)
            }
            CC_FLUSH_CONTEXT => {
                let handle = parameters.u32()?;
                end_of(parameters)?;
                let position = self
                    .sequences
                    .iter()
                    .position(|sequence| sequence.handle == handle)
                    .ok_or(parameter_error(RC_HANDLE, 1))?;
                self.sequences.remove(position);
                Ok(Response::empty())
            }
            CC_READ_CLOCK => {
                end_of(parameters)?;
                let milliseconds = now_ns.saturating_sub(self.init_ns) / NS_PER_MS;
                let mut writer = Writer::new();
                writer.u64(milliseconds);
                writer.u64(milliseconds);
                // resetCount and restartCount: no TPM2_Clear and no resume ever happen.
                writer.u32(0);
                writer.u32(0);
                writer.u8(CLOCK_SAFE);
                Ok(Response::with(writer))
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
        self.sequences.clear();
        self.started = true;
        Ok(Response::empty())
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
        if self.sequences.len() == MAX_SEQUENCES {
            return Err(RC_OBJECT_MEMORY);
        }
        let handle = self.allocate_transient();
        self.sequences.push(Sequence { handle, auth, kind });
        Ok(Response {
            handles: vec![handle],
            parameters: Writer::new(),
        })
    }

    /// A transient handle no open sequence uses. Handles are not reused right away, so a stale
    /// handle from a flushed sequence fails instead of reaching the next one.
    fn allocate_transient(&mut self) -> u32 {
        loop {
            let handle = self.next_transient;
            self.next_transient = if handle == u32::MAX {
                TRANSIENT_FIRST
            } else {
                handle + 1
            };
            if !self
                .sequences
                .iter()
                .any(|sequence| sequence.handle == handle)
            {
                return handle;
            }
        }
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

    fn get_capability(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let capability = parameters.u32()?;
        let property = parameters.u32()?;
        let requested = parameters.u32()? as usize;
        end_of(parameters)?;
        let mut writer = Writer::new();
        // moreData, filled in below.
        writer.u8(0);
        writer.u32(capability);
        let more = match capability {
            CAP_ALGS => {
                let algorithms: Vec<(u16, u32)> = HASH_ALGORITHMS
                    .iter()
                    .map(|algorithm| (*algorithm, ALGORITHM_HASH))
                    .chain([(ALG_NULL, 0)])
                    .filter(|(algorithm, _)| *algorithm as u32 >= property)
                    .collect();
                list(
                    &mut writer,
                    &algorithms,
                    requested,
                    6,
                    |writer, (id, bits)| {
                        writer.u16(*id);
                        writer.u32(*bits);
                    },
                )
            }
            CAP_HANDLES => {
                let handles = self.handles_from(property)?;
                list(&mut writer, &handles, requested, 4, |writer, handle| {
                    writer.u32(*handle)
                })
            }
            CAP_COMMANDS => {
                let attributes: Vec<u32> = COMMANDS
                    .iter()
                    .filter(|command| command.code >= property)
                    .map(command_attributes)
                    .collect();
                list(&mut writer, &attributes, requested, 4, |writer, bits| {
                    writer.u32(*bits)
                })
            }
            CAP_PP_COMMANDS | CAP_AUDIT_COMMANDS | CAP_ECC_CURVES | CAP_AUTH_POLICIES | CAP_ACT => {
                writer.u32(0);
                false
            }
            CAP_PCRS => {
                writer.u32(PCR_BANKS.len() as u32);
                for algorithm in PCR_BANKS {
                    writer.u16(algorithm);
                    writer.u8(PCR_SELECT_BYTES as u8);
                    writer.bytes(&pcr_bitmap(0..PCR_COUNT));
                }
                false
            }
            CAP_TPM_PROPERTIES => {
                let properties: Vec<(u32, u32)> = self
                    .properties()
                    .into_iter()
                    .filter(|(tag, _)| *tag >= property)
                    .collect();
                list(
                    &mut writer,
                    &properties,
                    requested,
                    8,
                    |writer, (tag, value)| {
                        writer.u32(*tag);
                        writer.u32(*value);
                    },
                )
            }
            CAP_PCR_PROPERTIES => {
                let not_drtm = (0..PCR_COUNT)
                    .filter(|index| !(PCR_DRTM_FIRST..=PCR_DRTM_LAST).contains(index));
                let properties: Vec<(u32, [u8; PCR_SELECT_BYTES])> = [
                    (PT_PCR_SAVE, pcr_bitmap(0..PCR_DEBUG)),
                    (PT_PCR_EXTEND_L0, pcr_bitmap(not_drtm)),
                    (
                        PT_PCR_RESET_L0,
                        pcr_bitmap([PCR_DEBUG, PCR_APPLICATION].into_iter()),
                    ),
                    (
                        PT_PCR_NO_INCREMENT,
                        pcr_bitmap([PCR_DEBUG, PCR_APPLICATION].into_iter()),
                    ),
                    (
                        PT_PCR_DRTM_RESET,
                        pcr_bitmap(PCR_DRTM_FIRST..=PCR_DRTM_LAST),
                    ),
                ]
                .into_iter()
                .filter(|(tag, _)| *tag >= property)
                .collect();
                list(
                    &mut writer,
                    &properties,
                    requested,
                    4 + 1 + PCR_SELECT_BYTES,
                    |writer, (tag, bitmap)| {
                        writer.u32(*tag);
                        writer.u8(PCR_SELECT_BYTES as u8);
                        writer.bytes(bitmap);
                    },
                )
            }
            _ => return Err(parameter_error(RC_VALUE, 1)),
        };
        writer.bytes[0] = more as u8;
        Ok(Response::with(writer))
    }

    fn handles_from(&self, first: u32) -> Result<Vec<u32>, u32> {
        let mut handles: Vec<u32> = match handle_type(first) {
            HT_PCR => (0..PCR_COUNT as u32).collect(),
            HT_PERMANENT => PERMANENT_HANDLES.to_vec(),
            HT_TRANSIENT => self
                .sequences
                .iter()
                .map(|sequence| sequence.handle)
                .collect(),
            HT_NV_INDEX | HT_HMAC_SESSION | HT_POLICY_SESSION | HT_PERSISTENT => Vec::new(),
            _ => return Err(parameter_error(RC_VALUE, 2)),
        };
        handles.sort_unstable();
        handles.retain(|handle| *handle >= first && handle_type(*handle) == handle_type(first));
        Ok(handles)
    }

    fn properties(&self) -> Vec<(u32, u32)> {
        let vendor_word = |index: usize| {
            u32::from_be_bytes(VENDOR_STRING[index * 4..index * 4 + 4].try_into().unwrap())
        };
        let open = self.sequences.len() as u32;
        let command_count = COMMANDS.len() as u32;
        let buffer = super::DATA_BUFFER_BYTES as u32;
        vec![
            (PT_FAMILY_INDICATOR, FAMILY_2_0),
            (PT_LEVEL, 0),
            (PT_REVISION, SPEC_REVISION),
            (PT_DAY_OF_YEAR, SPEC_DAY_OF_YEAR),
            (PT_YEAR, SPEC_YEAR),
            (PT_MANUFACTURER, MANUFACTURER),
            (PT_VENDOR_STRING_1, vendor_word(0)),
            (PT_VENDOR_STRING_2, vendor_word(1)),
            (PT_VENDOR_STRING_3, vendor_word(2)),
            (PT_VENDOR_STRING_4, vendor_word(3)),
            (PT_VENDOR_TPM_TYPE, VENDOR_TPM_TYPE),
            (PT_FIRMWARE_VERSION_1, FIRMWARE_VERSION_1),
            (PT_FIRMWARE_VERSION_2, 0),
            (PT_INPUT_BUFFER, MAX_BUFFER_BYTES as u32),
            (PT_HR_TRANSIENT_MIN, TRANSIENT_SLOTS),
            (PT_HR_PERSISTENT_MIN, 0),
            (PT_HR_LOADED_MIN, TRANSIENT_SLOTS),
            (PT_ACTIVE_SESSIONS_MAX, 0),
            (PT_PCR_COUNT, PCR_COUNT as u32),
            (PT_PCR_SELECT_MIN, PCR_SELECT_BYTES as u32),
            (PT_CONTEXT_GAP_MAX, CONTEXT_GAP_MAX),
            (PT_NV_COUNTERS_MAX, 0),
            (PT_NV_INDEX_MAX, 0),
            (PT_MEMORY, 0),
            (PT_CLOCK_UPDATE, CLOCK_UPDATE_MS),
            (PT_CONTEXT_HASH, ALG_SHA256 as u32),
            (PT_CONTEXT_SYM, ALG_NULL as u32),
            (PT_CONTEXT_SYM_SIZE, 0),
            (PT_ORDERLY_COUNT, ORDERLY_COUNT),
            (PT_MAX_COMMAND_SIZE, buffer),
            (PT_MAX_RESPONSE_SIZE, buffer),
            (PT_MAX_DIGEST, MAX_DIGEST_BYTES as u32),
            (PT_MAX_OBJECT_CONTEXT, 0),
            (PT_MAX_SESSION_CONTEXT, 0),
            (PT_PS_FAMILY_INDICATOR, PS_FAMILY_PC_CLIENT),
            (PT_PS_LEVEL, 0),
            (PT_PS_REVISION, PS_REVISION),
            (PT_PS_DAY_OF_YEAR, 0),
            (PT_PS_YEAR, 0),
            (PT_SPLIT_MAX, 0),
            (PT_TOTAL_COMMANDS, command_count),
            (PT_LIBRARY_COMMANDS, command_count),
            (PT_VENDOR_COMMANDS, 0),
            (PT_NV_BUFFER_MAX, MAX_BUFFER_BYTES as u32),
            (PT_MODES, 0),
            (PT_MAX_CAP_BUFFER, (MAX_CAP_DATA_BYTES + 8) as u32),
            (PT_PERMANENT, 0),
            (PT_STARTUP_CLEAR, STARTUP_CLEAR_ENABLED),
            (PT_HR_NV_INDEX, 0),
            (PT_HR_LOADED, open),
            (PT_HR_LOADED_AVAIL, TRANSIENT_SLOTS - open),
            (PT_HR_ACTIVE, 0),
            (PT_HR_ACTIVE_AVAIL, 0),
            (PT_HR_TRANSIENT_AVAIL, TRANSIENT_SLOTS - open),
            (PT_HR_PERSISTENT, 0),
            (PT_HR_PERSISTENT_AVAIL, 0),
            (PT_NV_COUNTERS, 0),
            (PT_NV_COUNTERS_AVAIL, 0),
            (PT_ALGORITHM_SET, 0),
            (PT_LOADED_CURVES, 0),
            (PT_LOCKOUT_COUNTER, 0),
            (PT_MAX_AUTH_FAIL, MAX_AUTH_FAIL),
            (PT_LOCKOUT_INTERVAL, LOCKOUT_INTERVAL_SECONDS),
            (PT_LOCKOUT_RECOVERY, LOCKOUT_RECOVERY_SECONDS),
            (PT_NV_WRITE_RECOVERY, 0),
            (PT_AUDIT_COUNTER_0, 0),
            (PT_AUDIT_COUNTER_1, 0),
        ]
    }

    #[cfg(test)]
    pub fn pcr(&self, index: usize) -> Option<&[u8]> {
        self.banks.first().map(|bank| bank[index].as_slice())
    }
}

/// Writes a `TPML_*`: a count, then as many entries as both the caller asked for and one
/// capability answer holds. Returns `moreData`.
fn list<T>(
    writer: &mut Writer,
    entries: &[T],
    requested: usize,
    entry_bytes: usize,
    mut write: impl FnMut(&mut Writer, &T),
) -> bool {
    let count = entries
        .len()
        .min(requested)
        .min(MAX_CAP_DATA_BYTES / entry_bytes);
    writer.u32(count as u32);
    for entry in &entries[..count] {
        write(writer, entry);
    }
    count < entries.len()
}
