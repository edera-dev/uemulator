use super::engine::RC_SUCCESS;
use super::hash::{self, ALG_SHA1, ALG_SHA256, ALG_SHA384, ALG_SHA512};
use super::wire::{Reader, Writer};
use super::*;
use crate::dev::RngSource;
use crate::kv::{KeyValueStore, MemoryKeyValueStore};
use crate::sync::{Arc, Mutex};

const ST_NO_SESSIONS: u16 = 0x8001;
const ST_SESSIONS: u16 = 0x8002;
const RS_PW: u32 = 0x4000_0009;
const RH_NULL: u32 = 0x4000_0007;
const RH_OWNER: u32 = 0x4000_0001;

const CC_STARTUP: u32 = 0x144;
const CC_GET_RANDOM: u32 = 0x17b;
const CC_GET_CAPABILITY: u32 = 0x17a;
const CC_PCR_READ: u32 = 0x17e;
const CC_PCR_EXTEND: u32 = 0x182;
const CC_PCR_EVENT: u32 = 0x13c;
const CC_PCR_RESET: u32 = 0x13d;
const CC_HASH: u32 = 0x17d;
const CC_HASH_SEQUENCE_START: u32 = 0x186;
const CC_SEQUENCE_UPDATE: u32 = 0x15c;
const CC_SEQUENCE_COMPLETE: u32 = 0x13e;
const CC_EVENT_SEQUENCE_COMPLETE: u32 = 0x185;
const CC_FLUSH_CONTEXT: u32 = 0x165;
const CC_READ_CLOCK: u32 = 0x181;
/// TPM2_Sign, which this TPM does not implement.
const CC_SIGN: u32 = 0x15d;

const RC_INITIALIZE: u32 = 0x100;
const RC_COMMAND_CODE: u32 = 0x143;
const RC_COMMAND_SIZE: u32 = 0x142;
const RC_BAD_TAG: u32 = 0x01e;
const RC_AUTH_MISSING: u32 = 0x125;
const RC_LOCALITY: u32 = 0x907;
/// TPM_RC_AUTH_FAIL for session 1.
const RC_AUTH_FAIL_SESSION_1: u32 = 0x08e | 0x800 | 0x100;
const RC_REFERENCE_H0: u32 = 0x910;
/// TPM_RC_HANDLE for parameter 1.
const RC_HANDLE_PARAMETER_1: u32 = 0x08b | 0x040 | 0x100;

const CAP_HANDLES: u32 = 1;
const CAP_COMMANDS: u32 = 2;
const CAP_PCRS: u32 = 5;
const CAP_TPM_PROPERTIES: u32 = 6;
const PT_TOTAL_COMMANDS: u32 = 0x129;
const TRANSIENT_FIRST: u32 = 0x8000_0000;

fn memory_store() -> SharedKeyValueStore {
    let store: Box<dyn KeyValueStore> = Box::new(MemoryKeyValueStore::new());
    Arc::new(Mutex::new(store))
}

fn tpm() -> Tpm {
    Tpm::new(Random::new(RngSource::Seeded(7)), 0, memory_store())
}

fn config(seed: u64) -> TpmConfig {
    TpmConfig {
        random: RngSource::Seeded(seed),
        store: memory_store(),
    }
}

fn started() -> Tpm {
    let mut tpm = tpm();
    let response = tpm.execute(&command(CC_STARTUP, &[], &0u16.to_be_bytes()), 0);
    assert_eq!(code(&response), RC_SUCCESS);
    tpm
}

/// A command with a password session per handle when `handles` is not empty.
fn command(code: u32, handles: &[u32], parameters: &[u8]) -> Vec<u8> {
    command_with_passwords(code, handles, &vec![&[][..]; handles.len()], parameters)
}

fn command_with_passwords(
    code: u32,
    handles: &[u32],
    passwords: &[&[u8]],
    parameters: &[u8],
) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.u16(if passwords.is_empty() {
        ST_NO_SESSIONS
    } else {
        ST_SESSIONS
    });
    writer.u32(0);
    writer.u32(code);
    for handle in handles {
        writer.u32(*handle);
    }
    if !passwords.is_empty() {
        let mut sessions = Writer::new();
        for password in passwords {
            sessions.u32(RS_PW);
            sessions.sized(&[]);
            sessions.u8(1);
            sessions.sized(password);
        }
        writer.u32(sessions.bytes.len() as u32);
        writer.bytes(&sessions.bytes);
    }
    writer.bytes(parameters);
    let length = writer.bytes.len() as u32;
    writer.bytes[2..6].copy_from_slice(&length.to_be_bytes());
    writer.bytes
}

fn code(response: &[u8]) -> u32 {
    u32::from_be_bytes(response[6..10].try_into().unwrap())
}

/// The parameter area of a successful response, past any response handles.
fn parameters(response: &[u8], handles: usize) -> &[u8] {
    assert_eq!(code(response), RC_SUCCESS, "response {response:02x?}");
    let tag = u16::from_be_bytes([response[0], response[1]]);
    let after_handles = &response[10 + 4 * handles..];
    if tag == ST_SESSIONS {
        let size = u32::from_be_bytes(after_handles[..4].try_into().unwrap()) as usize;
        &after_handles[4..4 + size]
    } else {
        after_handles
    }
}

fn sha256(parts: &[&[u8]]) -> Vec<u8> {
    hash::digest(ALG_SHA256, parts).unwrap()
}

fn extend_sha256(digest: &[u8]) -> Vec<u8> {
    let mut parameters = Writer::new();
    parameters.u32(1);
    parameters.u16(ALG_SHA256);
    parameters.bytes(digest);
    parameters.bytes
}

fn read_pcr(tpm: &mut Tpm, index: usize) -> Vec<u8> {
    let mut selection = Writer::new();
    selection.u32(1);
    selection.u16(ALG_SHA256);
    selection.u8(3);
    let mut bitmap = [0u8; 3];
    bitmap[index / 8] = 1 << (index % 8);
    selection.bytes(&bitmap);
    let response = tpm.execute(&command(CC_PCR_READ, &[], &selection.bytes), 0);
    let mut reader = Reader::new(parameters(&response, 0));
    reader.u32().unwrap();
    assert_eq!(reader.u32().unwrap(), 1);
    reader.take(6).unwrap();
    assert_eq!(reader.u32().unwrap(), 1, "one digest");
    reader.sized().unwrap().to_vec()
}

fn capability(tpm: &mut Tpm, capability: u32, property: u32, count: u32) -> Vec<u8> {
    let mut writer = Writer::new();
    writer.u32(capability);
    writer.u32(property);
    writer.u32(count);
    let response = tpm.execute(&command(CC_GET_CAPABILITY, &[], &writer.bytes), 0);
    parameters(&response, 0).to_vec()
}

#[test]
fn nothing_runs_before_startup_and_startup_runs_once() {
    let mut tpm = tpm();
    let random = command(CC_GET_RANDOM, &[], &8u16.to_be_bytes());
    assert_eq!(code(&tpm.execute(&random, 0)), RC_INITIALIZE);
    let startup = command(CC_STARTUP, &[], &0u16.to_be_bytes());
    assert_eq!(code(&tpm.execute(&startup, 0)), RC_SUCCESS);
    assert_eq!(code(&tpm.execute(&startup, 0)), RC_INITIALIZE);
    let response = tpm.execute(&random, 0);
    assert_eq!(parameters(&response, 0).len(), 2 + 8);
}

#[test]
fn startup_state_has_nothing_to_resume() {
    let mut tpm = tpm();
    let response = tpm.execute(&command(CC_STARTUP, &[], &1u16.to_be_bytes()), 0);
    assert_ne!(code(&response), RC_SUCCESS);
}

#[test]
fn malformed_commands_get_the_matching_codes() {
    let mut tpm = started();
    assert_eq!(
        code(&tpm.execute(&command(CC_SIGN, &[], &[]), 0)),
        RC_COMMAND_CODE
    );
    let mut bad_tag = command(CC_GET_RANDOM, &[], &8u16.to_be_bytes());
    bad_tag[1] = 0x99;
    assert_eq!(code(&tpm.execute(&bad_tag, 0)), RC_BAD_TAG);
    let mut short = command(CC_GET_RANDOM, &[], &8u16.to_be_bytes());
    short[5] += 1;
    assert_eq!(code(&tpm.execute(&short, 0)), RC_COMMAND_SIZE);
    assert_eq!(code(&tpm.execute(&[0x80, 0x01, 0, 0], 0)), RC_COMMAND_SIZE);
    let no_session = command_with_passwords(CC_PCR_EXTEND, &[0], &[], &extend_sha256(&[0; 32]));
    assert_eq!(code(&tpm.execute(&no_session, 0)), RC_AUTH_MISSING);
}

#[test]
fn extend_folds_the_digest_into_the_pcr() {
    let mut tpm = started();
    assert_eq!(read_pcr(&mut tpm, 4), [0; 32]);
    let measurement = sha256(&[b"boot loader"]);
    let response = tpm.execute(
        &command(CC_PCR_EXTEND, &[4], &extend_sha256(&measurement)),
        0,
    );
    assert_eq!(code(&response), RC_SUCCESS);
    assert_eq!(read_pcr(&mut tpm, 4), sha256(&[&[0; 32], &measurement]));
    // A digest for a bank that is not allocated is accepted and goes nowhere.
    let mut sha1_only = Writer::new();
    sha1_only.u32(1);
    sha1_only.u16(ALG_SHA1);
    sha1_only.bytes(&[1; 20]);
    let before = read_pcr(&mut tpm, 4);
    let response = tpm.execute(&command(CC_PCR_EXTEND, &[4], &sha1_only.bytes), 0);
    assert_eq!(code(&response), RC_SUCCESS);
    assert_eq!(read_pcr(&mut tpm, 4), before);
}

#[test]
fn the_null_handle_extends_nothing() {
    let mut tpm = started();
    let response = tpm.execute(
        &command(CC_PCR_EXTEND, &[RH_NULL], &extend_sha256(&[9; 32])),
        0,
    );
    assert_eq!(code(&response), RC_SUCCESS);
    assert!(
        (0..24)
            .filter(|pcr| !(17..=22).contains(pcr))
            .all(|pcr| tpm.pcr(pcr) == Some(&[0; 32][..]))
    );
}

#[test]
fn pcr_event_hashes_with_every_algorithm_and_extends_the_sha256_bank() {
    let mut tpm = started();
    let mut event = Writer::new();
    event.sized(b"event data");
    let response = tpm.execute(&command(CC_PCR_EVENT, &[8], &event.bytes), 0);
    let mut reader = Reader::new(parameters(&response, 0));
    assert_eq!(reader.u32().unwrap(), 4);
    for algorithm in [ALG_SHA1, ALG_SHA256, ALG_SHA384, ALG_SHA512] {
        assert_eq!(reader.u16().unwrap(), algorithm);
        let expected = hash::digest(algorithm, &[b"event data"]).unwrap();
        assert_eq!(reader.take(expected.len()).unwrap(), expected);
    }
    assert_eq!(
        read_pcr(&mut tpm, 8),
        sha256(&[&[0; 32], &sha256(&[b"event data"])])
    );
}

#[test]
fn drtm_pcrs_start_as_ones_and_need_a_higher_locality() {
    let mut tpm = started();
    assert_eq!(read_pcr(&mut tpm, 17), [0xff; 32]);
    let response = tpm.execute(&command(CC_PCR_EXTEND, &[17], &extend_sha256(&[1; 32])), 0);
    assert_eq!(code(&response), RC_LOCALITY);
    assert_eq!(
        code(&tpm.execute(&command(CC_PCR_RESET, &[0], &[]), 0)),
        RC_LOCALITY
    );
}

#[test]
fn the_debug_and_application_pcrs_reset() {
    let mut tpm = started();
    for pcr in [16, 23] {
        tpm.execute(&command(CC_PCR_EXTEND, &[pcr], &extend_sha256(&[1; 32])), 0);
        assert_ne!(read_pcr(&mut tpm, pcr as usize), [0; 32]);
        let response = tpm.execute(&command(CC_PCR_RESET, &[pcr], &[]), 0);
        assert_eq!(code(&response), RC_SUCCESS);
        assert_eq!(read_pcr(&mut tpm, pcr as usize), [0; 32]);
    }
}

#[test]
fn pcr_read_answers_at_most_eight_and_says_which() {
    let mut tpm = started();
    let mut selection = Writer::new();
    selection.u32(1);
    selection.u16(ALG_SHA256);
    selection.u8(3);
    selection.bytes(&[0xff, 0xff, 0xff]);
    let response = tpm.execute(&command(CC_PCR_READ, &[], &selection.bytes), 0);
    let mut reader = Reader::new(parameters(&response, 0));
    reader.u32().unwrap();
    assert_eq!(reader.u32().unwrap(), 1);
    assert_eq!(reader.u16().unwrap(), ALG_SHA256);
    assert_eq!(reader.u8().unwrap(), 3);
    assert_eq!(reader.take(3).unwrap(), [0xff, 0, 0]);
    assert_eq!(reader.u32().unwrap(), 8);
}

#[test]
fn hash_returns_the_digest_and_a_null_ticket() {
    let mut tpm = started();
    let mut writer = Writer::new();
    writer.sized(b"abc");
    writer.u16(ALG_SHA256);
    writer.u32(RH_OWNER);
    let response = tpm.execute(&command(CC_HASH, &[], &writer.bytes), 0);
    let mut reader = Reader::new(parameters(&response, 0));
    assert_eq!(reader.sized().unwrap(), sha256(&[b"abc"]));
    assert_eq!(reader.u16().unwrap(), 0x8024);
    assert_eq!(reader.u32().unwrap(), RH_NULL);
    assert_eq!(reader.sized().unwrap(), b"");
}

fn start_sequence(tpm: &mut Tpm, algorithm: u16, password: &[u8]) -> u32 {
    let mut writer = Writer::new();
    writer.sized(password);
    writer.u16(algorithm);
    let response = tpm.execute(&command(CC_HASH_SEQUENCE_START, &[], &writer.bytes), 0);
    assert_eq!(code(&response), RC_SUCCESS);
    u32::from_be_bytes(response[10..14].try_into().unwrap())
}

#[test]
fn a_hash_sequence_checks_its_password_and_hashes_every_piece() {
    let mut tpm = started();
    let handle = start_sequence(&mut tpm, ALG_SHA256, b"secret");
    let mut piece = Writer::new();
    piece.sized(b"hello ");
    let wrong = command_with_passwords(CC_SEQUENCE_UPDATE, &[handle], &[b"guess"], &piece.bytes);
    assert_eq!(code(&tpm.execute(&wrong, 0)), RC_AUTH_FAIL_SESSION_1);
    let right = command_with_passwords(CC_SEQUENCE_UPDATE, &[handle], &[b"secret"], &piece.bytes);
    assert_eq!(code(&tpm.execute(&right, 0)), RC_SUCCESS);
    let mut last = Writer::new();
    last.sized(b"world");
    last.u32(RH_NULL);
    let complete =
        command_with_passwords(CC_SEQUENCE_COMPLETE, &[handle], &[b"secret"], &last.bytes);
    let response = tpm.execute(&complete, 0);
    let mut reader = Reader::new(parameters(&response, 0));
    assert_eq!(reader.sized().unwrap(), sha256(&[b"hello world"]));
    // Completing flushed it: a transient handle that is not loaded.
    assert_eq!(code(&tpm.execute(&right, 0)), RC_REFERENCE_H0);
}

#[test]
fn an_event_sequence_extends_the_pcr_with_every_bank_it_has() {
    let mut tpm = started();
    let handle = start_sequence(&mut tpm, 0x0010, b"");
    let mut data = Writer::new();
    data.sized(b"measured blob");
    let response = tpm.execute(
        &command(CC_EVENT_SEQUENCE_COMPLETE, &[2, handle], &data.bytes),
        0,
    );
    let mut reader = Reader::new(parameters(&response, 0));
    assert_eq!(reader.u32().unwrap(), 4);
    assert_eq!(
        read_pcr(&mut tpm, 2),
        sha256(&[&[0; 32], &sha256(&[b"measured blob"])])
    );
}

#[test]
fn sequences_are_listed_and_flushed_as_transient_handles() {
    let mut tpm = started();
    let first = start_sequence(&mut tpm, ALG_SHA256, b"");
    let second = start_sequence(&mut tpm, ALG_SHA384, b"");
    let listed = capability(&mut tpm, CAP_HANDLES, TRANSIENT_FIRST, 16);
    let mut reader = Reader::new(&listed);
    assert_eq!(reader.u8().unwrap(), 0);
    assert_eq!(reader.u32().unwrap(), CAP_HANDLES);
    assert_eq!(reader.u32().unwrap(), 2);
    assert_eq!(
        [reader.u32().unwrap(), reader.u32().unwrap()],
        [first, second]
    );
    let flush = command(CC_FLUSH_CONTEXT, &[], &first.to_be_bytes());
    assert_eq!(code(&tpm.execute(&flush, 0)), RC_SUCCESS);
    assert_eq!(code(&tpm.execute(&flush, 0)), RC_HANDLE_PARAMETER_1);
}

#[test]
fn capabilities_describe_the_commands_and_the_one_bank() {
    let mut tpm = started();
    let properties = capability(&mut tpm, CAP_TPM_PROPERTIES, PT_TOTAL_COMMANDS, 1);
    let mut reader = Reader::new(&properties);
    assert_eq!(reader.u8().unwrap(), 1, "more properties follow");
    reader.u32().unwrap();
    assert_eq!(reader.u32().unwrap(), 1);
    assert_eq!(reader.u32().unwrap(), PT_TOTAL_COMMANDS);
    let total = reader.u32().unwrap();

    let commands = capability(&mut tpm, CAP_COMMANDS, 0, 256);
    let mut reader = Reader::new(&commands);
    assert_eq!(reader.u8().unwrap(), 0);
    reader.u32().unwrap();
    assert_eq!(reader.u32().unwrap(), total);
    let attributes: Vec<u32> = (0..total).map(|_| reader.u32().unwrap()).collect();
    let extend = attributes
        .iter()
        .find(|bits| *bits & 0xffff == CC_PCR_EXTEND)
        .unwrap();
    assert_eq!(extend >> 25 & 7, 1, "PCR_Extend has one handle");
    let start = attributes
        .iter()
        .find(|bits| *bits & 0xffff == CC_HASH_SEQUENCE_START)
        .unwrap();
    assert_ne!(start & 1 << 28, 0, "HashSequenceStart returns a handle");
    assert!(
        attributes
            .windows(2)
            .all(|pair| pair[0] & 0xffff < pair[1] & 0xffff)
    );

    let pcrs = capability(&mut tpm, CAP_PCRS, 0, 1);
    assert_eq!(&pcrs[5..], [0, 0, 0, 1, 0, 0x0b, 3, 0xff, 0xff, 0xff]);
}

#[test]
fn the_seeded_random_stream_is_repeatable() {
    let random = command(CC_GET_RANDOM, &[], &32u16.to_be_bytes());
    let mut first = started();
    let mut second = started();
    assert_eq!(first.execute(&random, 0), second.execute(&random, 0));
    assert_ne!(
        first.execute(&random, 0),
        second.execute(&command(CC_GET_RANDOM, &[], &16u16.to_be_bytes()), 0)
    );
}

#[test]
fn read_clock_counts_milliseconds_since_init() {
    let mut tpm = Tpm::new(Random::new(RngSource::Seeded(1)), 5_000_000, memory_store());
    tpm.execute(&command(CC_STARTUP, &[], &0u16.to_be_bytes()), 5_000_000);
    let response = tpm.execute(&command(CC_READ_CLOCK, &[], &[]), 1_005_000_000);
    let mut reader = Reader::new(parameters(&response, 0));
    let time = u64::from_be_bytes(reader.take(8).unwrap().try_into().unwrap());
    assert_eq!(time, 1000);
}

fn write_buffer(crb: &mut TpmCrb, bytes: &[u8]) {
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        crb.mmio_write(
            TPM_CRB_BASE + DATA_BUFFER + 4 * index as u64,
            4,
            u32::from_le_bytes(word) as u64,
            0,
        );
    }
}

fn read_buffer(crb: &TpmCrb, length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| crb.mmio_read(TPM_CRB_BASE + DATA_BUFFER + index as u64, 1) as u8)
        .collect()
}

#[test]
fn the_crb_interface_runs_a_command_the_way_linux_drives_it() {
    let mut crb = TpmCrb::new(config(3), 0);
    let interface = crb.mmio_read(TPM_CRB_BASE + INTF_ID, 8);
    assert_eq!(interface & 0xf, 1, "CRB interface type");
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + LOC_STATE, 4) & LOC_STATE_ASSIGNED as u64,
        0
    );
    crb.mmio_write(
        TPM_CRB_BASE + LOC_CTRL,
        4,
        LOC_CTRL_REQUEST_ACCESS as u64,
        0,
    );
    assert_ne!(
        crb.mmio_read(TPM_CRB_BASE + LOC_STATE, 4) & LOC_STATE_ASSIGNED as u64,
        0
    );
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + LOC_STS, 4),
        LOC_STS_GRANTED as u64
    );
    assert_ne!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_STS, 4) & CTRL_STS_IDLE as u64,
        0
    );
    crb.mmio_write(TPM_CRB_BASE + CTRL_REQ, 4, CTRL_REQ_CMD_READY as u64, 0);
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_REQ, 4),
        0,
        "requests complete at once"
    );
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_STS, 4) & CTRL_STS_IDLE as u64,
        0
    );
    // Linux reads the buffer's address and size from the control area.
    let command_address = crb.mmio_read(TPM_CRB_BASE + CTRL_CMD_LADDR, 4)
        | crb.mmio_read(TPM_CRB_BASE + CTRL_CMD_HADDR, 4) << 32;
    assert_eq!(command_address, TPM_CRB_BASE + DATA_BUFFER);
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_RSP_ADDR, 8),
        TPM_CRB_BASE + DATA_BUFFER
    );
    assert_eq!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_CMD_SIZE, 4),
        DATA_BUFFER_BYTES as u64
    );

    write_buffer(&mut crb, &command(CC_STARTUP, &[], &0u16.to_be_bytes()));
    crb.mmio_write(TPM_CRB_BASE + CTRL_START, 4, CTRL_START_INVOKE as u64, 0);
    assert_eq!(crb.mmio_read(TPM_CRB_BASE + CTRL_START, 4), 0);
    assert_eq!(code(&read_buffer(&crb, 10)), RC_SUCCESS);

    write_buffer(&mut crb, &command(CC_GET_RANDOM, &[], &16u16.to_be_bytes()));
    crb.mmio_write(TPM_CRB_BASE + CTRL_START, 4, CTRL_START_INVOKE as u64, 0);
    let header = read_buffer(&crb, 10);
    let length = u32::from_be_bytes(header[2..6].try_into().unwrap()) as usize;
    assert_eq!(length, 10 + 2 + 16);
    assert_eq!(code(&header), RC_SUCCESS);

    crb.mmio_write(TPM_CRB_BASE + CTRL_REQ, 4, CTRL_REQ_GO_IDLE as u64, 0);
    assert_ne!(
        crb.mmio_read(TPM_CRB_BASE + CTRL_STS, 4) & CTRL_STS_IDLE as u64,
        0
    );
    crb.mmio_write(TPM_CRB_BASE + LOC_CTRL, 4, LOC_CTRL_RELINQUISH as u64, 0);
    assert_eq!(crb.mmio_read(TPM_CRB_BASE + LOC_STS, 4), 0);
}

#[test]
fn a_size_past_the_buffer_is_refused_not_overrun() {
    let mut crb = TpmCrb::new(config(3), 0);
    let mut header = command(CC_STARTUP, &[], &0u16.to_be_bytes());
    header[2..6].copy_from_slice(&0xffff_ffffu32.to_be_bytes());
    write_buffer(&mut crb, &header);
    crb.mmio_write(TPM_CRB_BASE + CTRL_START, 4, CTRL_START_INVOKE as u64, 0);
    assert_eq!(code(&read_buffer(&crb, 10)), RC_COMMAND_SIZE);
}
