//! `TPM2_GetCapability`: what the TPM implements and how much room it has. Linux, the firmware
//! and the TSS libraries decide what to use from these answers, so every list must say exactly
//! what is implemented.
#[cfg(freestanding)]
use crate::prelude::*;

use super::context::{MAX_OBJECT_CONTEXT_BYTES, MAX_SESSION_CONTEXT_BYTES};
use super::engine::{
    MAX_BUFFER_BYTES, PCR_APPLICATION, PCR_BANKS, PCR_COUNT, PCR_DEBUG, PCR_DRTM_FIRST,
    PCR_DRTM_LAST, PCR_SELECT_BYTES, Response, TRANSIENT_SLOTS, Tpm, command_attributes, commands,
    end_of, pcr_bitmap,
};
use super::hash::{ALG_NULL, ALG_SHA256, HASH_ALGORITHMS, MAX_DIGEST_BYTES};
use super::hierarchy::MAX_PERSISTENT_OBJECTS;
use super::nv::{MAX_NV_BUFFER_BYTES, MAX_NV_INDEX_BYTES};
use super::session::{MAX_ACTIVE_SESSIONS, MAX_LOADED_SESSIONS};
use super::spec::*;
use super::wire::{Reader, Writer};

/// Contexts are encrypted with AES-128 (`context.rs`).
const CONTEXT_SYM_BITS: u32 = 128;

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
const ALGORITHM_ASYMMETRIC: u32 = 1 << 0;
const ALGORITHM_SYMMETRIC: u32 = 1 << 1;
const ALGORITHM_HASH: u32 = 1 << 2;
const ALGORITHM_OBJECT: u32 = 1 << 3;
const ALGORITHM_SIGNING: u32 = 1 << 8;
const ALGORITHM_ENCRYPTING: u32 = 1 << 9;
const ALGORITHM_METHOD: u32 = 1 << 10;

/// Every algorithm something here implements, by `TPM_ALG_ID`, with its `TPMA_ALGORITHM`. The
/// hashes come from `HASH_ALGORITHMS`.
const OTHER_ALGORITHMS: [(u16, u32); 12] = [
    (ALG_RSA, ALGORITHM_ASYMMETRIC | ALGORITHM_OBJECT),
    (ALG_HMAC, ALGORITHM_HASH | ALGORITHM_SIGNING),
    (ALG_AES, ALGORITHM_SYMMETRIC),
    (ALG_MGF1, ALGORITHM_HASH | ALGORITHM_METHOD),
    (
        ALG_KEYEDHASH,
        ALGORITHM_HASH | ALGORITHM_OBJECT | ALGORITHM_SIGNING | ALGORITHM_ENCRYPTING,
    ),
    (ALG_NULL, 0),
    (ALG_RSAES, ALGORITHM_ASYMMETRIC | ALGORITHM_ENCRYPTING),
    (ALG_OAEP, ALGORITHM_ASYMMETRIC | ALGORITHM_ENCRYPTING),
    (ALG_ECDH, ALGORITHM_ASYMMETRIC | ALGORITHM_METHOD),
    (ALG_KDF1_SP800_56A, ALGORITHM_HASH | ALGORITHM_METHOD),
    (ALG_KDF1_SP800_108, ALGORITHM_HASH | ALGORITHM_METHOD),
    (ALG_ECC, ALGORITHM_ASYMMETRIC | ALGORITHM_OBJECT),
];
/// `TPM_ALG_CFB`, the one symmetric mode.
const MODE_ALGORITHMS: [(u16, u32); 1] = [(ALG_CFB, ALGORITHM_SYMMETRIC | ALGORITHM_ENCRYPTING)];

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

impl Tpm {
    pub(super) fn get_capability(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
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
                let mut algorithms: Vec<(u16, u32)> = HASH_ALGORITHMS
                    .iter()
                    .map(|algorithm| (*algorithm, ALGORITHM_HASH))
                    .chain(OTHER_ALGORITHMS)
                    .chain(MODE_ALGORITHMS)
                    .filter(|(algorithm, _)| *algorithm as u32 >= property)
                    .collect();
                algorithms.sort_unstable();
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
                let attributes: Vec<u32> = commands()
                    .iter()
                    .filter(|command| command.code >= property)
                    .map(command_attributes)
                    .collect();
                list(&mut writer, &attributes, requested, 4, |writer, bits| {
                    writer.u32(*bits)
                })
            }
            CAP_ECC_CURVES => {
                let curves: Vec<u16> = [ECC_NIST_P256]
                    .into_iter()
                    .filter(|curve| *curve as u32 >= property)
                    .collect();
                list(&mut writer, &curves, requested, 2, |writer, curve| {
                    writer.u16(*curve)
                })
            }
            CAP_PP_COMMANDS | CAP_AUDIT_COMMANDS | CAP_AUTH_POLICIES | CAP_ACT => {
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
            HT_TRANSIENT | HT_PERSISTENT => self
                .sequences
                .iter()
                .map(|sequence| sequence.handle)
                .chain(self.objects.iter().map(|object| object.handle))
                .collect(),
            HT_NV_INDEX => self.nv.iter().map(|index| index.handle).collect(),
            // In TPM_CAP_HANDLES the session types mean loaded and saved sessions.
            HT_HMAC_SESSION => self
                .sessions
                .iter()
                .filter(|session| session.saved.is_none())
                .map(|session| session.handle)
                .collect(),
            HT_POLICY_SESSION => self
                .sessions
                .iter()
                .filter(|session| session.saved.is_some())
                .map(|session| session.handle)
                .collect(),
            _ => return Err(parameter_error(RC_VALUE, 2)),
        };
        handles.sort_unstable();
        // A saved or loaded session is listed under its own handle, whose type may be either.
        let session_listing = matches!(handle_type(first), HT_HMAC_SESSION | HT_POLICY_SESSION);
        handles.retain(|handle| {
            *handle >= first && (session_listing || handle_type(*handle) == handle_type(first))
        });
        Ok(handles)
    }

    fn properties(&self) -> Vec<(u32, u32)> {
        let vendor_word = |index: usize| {
            u32::from_be_bytes(VENDOR_STRING[index * 4..index * 4 + 4].try_into().unwrap())
        };
        let command_count = commands().len() as u32;
        let transient_available = self.transient_available() as u32;
        let loaded_sessions = self.loaded_session_count() as u32;
        let active_sessions = self.sessions.len() as u32;
        let persistent = self.persistent_count() as u32;
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
            (PT_HR_TRANSIENT_MIN, TRANSIENT_SLOTS as u32),
            (PT_HR_PERSISTENT_MIN, MAX_PERSISTENT_OBJECTS as u32),
            (PT_HR_LOADED_MIN, MAX_LOADED_SESSIONS as u32),
            (PT_ACTIVE_SESSIONS_MAX, MAX_ACTIVE_SESSIONS as u32),
            (PT_PCR_COUNT, PCR_COUNT as u32),
            (PT_PCR_SELECT_MIN, PCR_SELECT_BYTES as u32),
            (PT_CONTEXT_GAP_MAX, CONTEXT_GAP_MAX),
            (PT_NV_COUNTERS_MAX, 0),
            (PT_NV_INDEX_MAX, MAX_NV_INDEX_BYTES as u32),
            (PT_MEMORY, 0),
            (PT_CLOCK_UPDATE, CLOCK_UPDATE_MS),
            (PT_CONTEXT_HASH, ALG_SHA256 as u32),
            (PT_CONTEXT_SYM, ALG_AES as u32),
            (PT_CONTEXT_SYM_SIZE, CONTEXT_SYM_BITS),
            (PT_ORDERLY_COUNT, ORDERLY_COUNT),
            (PT_MAX_COMMAND_SIZE, buffer),
            (PT_MAX_RESPONSE_SIZE, buffer),
            (PT_MAX_DIGEST, MAX_DIGEST_BYTES as u32),
            (PT_MAX_OBJECT_CONTEXT, MAX_OBJECT_CONTEXT_BYTES as u32),
            (PT_MAX_SESSION_CONTEXT, MAX_SESSION_CONTEXT_BYTES as u32),
            (PT_PS_FAMILY_INDICATOR, PS_FAMILY_PC_CLIENT),
            (PT_PS_LEVEL, 0),
            (PT_PS_REVISION, PS_REVISION),
            (PT_PS_DAY_OF_YEAR, 0),
            (PT_PS_YEAR, 0),
            (PT_SPLIT_MAX, 0),
            (PT_TOTAL_COMMANDS, command_count),
            (PT_LIBRARY_COMMANDS, command_count),
            (PT_VENDOR_COMMANDS, 0),
            (PT_NV_BUFFER_MAX, MAX_NV_BUFFER_BYTES as u32),
            (PT_MODES, 0),
            (PT_MAX_CAP_BUFFER, (MAX_CAP_DATA_BYTES + 8) as u32),
            (PT_PERMANENT, 0),
            (PT_STARTUP_CLEAR, STARTUP_CLEAR_ENABLED),
            (PT_HR_NV_INDEX, self.nv.len() as u32),
            (PT_HR_LOADED, loaded_sessions),
            (
                PT_HR_LOADED_AVAIL,
                MAX_LOADED_SESSIONS as u32 - loaded_sessions.min(MAX_LOADED_SESSIONS as u32),
            ),
            (PT_HR_ACTIVE, active_sessions),
            (
                PT_HR_ACTIVE_AVAIL,
                MAX_ACTIVE_SESSIONS as u32 - active_sessions.min(MAX_ACTIVE_SESSIONS as u32),
            ),
            (PT_HR_TRANSIENT_AVAIL, transient_available),
            (PT_HR_PERSISTENT, persistent),
            (
                PT_HR_PERSISTENT_AVAIL,
                MAX_PERSISTENT_OBJECTS as u32 - persistent.min(MAX_PERSISTENT_OBJECTS as u32),
            ),
            (PT_NV_COUNTERS, 0),
            (PT_NV_COUNTERS_AVAIL, 0),
            (PT_ALGORITHM_SET, 0),
            (PT_LOADED_CURVES, 1),
            (PT_LOCKOUT_COUNTER, 0),
            (PT_MAX_AUTH_FAIL, MAX_AUTH_FAIL),
            (PT_LOCKOUT_INTERVAL, LOCKOUT_INTERVAL_SECONDS),
            (PT_LOCKOUT_RECOVERY, LOCKOUT_RECOVERY_SECONDS),
            (PT_NV_WRITE_RECOVERY, 0),
            (PT_AUDIT_COUNTER_0, 0),
            (PT_AUDIT_COUNTER_1, 0),
        ]
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
