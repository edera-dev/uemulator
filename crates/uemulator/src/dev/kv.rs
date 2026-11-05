//! The guest's window onto the embedder's `KeyValueStore`: four 32-bit registers a firmware
//! drives with a handful of `in`/`out` instructions. Protocol and port numbers are in
//! `docs/plans/19-uefi-firmware.md`; `crates/uemulator-firmware/src/kv.rs` is the guest side.

use crate::dev::tpm::STORE_PREFIX as TPM_STORE_PREFIX;
use crate::kv::{KeyValueStore, MAX_KEY_BYTES, MAX_VALUE_BYTES};
#[cfg(freestanding)]
use crate::prelude::*;
use crate::sync::{Arc, Mutex};

/// What a machine keeps across guest resets: the guest sees the same store on every boot.
pub(crate) type SharedKeyValueStore = Arc<Mutex<Box<dyn KeyValueStore>>>;

pub(crate) const KV_PORT_BASE: u16 = 0x580;
pub(crate) const KV_PORT_COUNT: u16 = 0x10;

const REGISTER_COMMAND_STATUS: u16 = 0x0;
const REGISTER_DATA: u16 = 0x4;
const REGISTER_LENGTH: u16 = 0x8;
const REGISTER_IDENT: u16 = 0xc;
/// "UKV1": lets firmware tell the device from an empty port (reads as all ones).
pub(crate) const IDENT: u32 = 0x3156_4b55;

const COMMAND_SELECT_KEY: u32 = 1;
const COMMAND_SELECT_AUX: u32 = 2;
const COMMAND_GET: u32 = 3;
const COMMAND_SET: u32 = 4;
const COMMAND_DELETE: u32 = 5;
const COMMAND_NEXT: u32 = 6;

pub(crate) const STATUS_OK: u32 = 0;
pub(crate) const STATUS_NOT_FOUND: u32 = 1;
pub(crate) const STATUS_TOO_LARGE: u32 = 2;
pub(crate) const STATUS_IO_ERROR: u32 = 3;
pub(crate) const STATUS_BAD_COMMAND: u32 = 4;

#[derive(Clone, Copy, PartialEq)]
enum Input {
    Key,
    Aux,
}

pub(crate) struct KvDevice {
    store: SharedKeyValueStore,
    key: Vec<u8>,
    aux: Vec<u8>,
    selected: Input,
    /// Bytes the guest announced for the selected input; the stream is cut there, so a final
    /// dword's padding never becomes part of the key or value.
    announced: usize,
    output: Vec<u8>,
    read_position: usize,
    status: u32,
}

impl KvDevice {
    pub fn new(store: SharedKeyValueStore) -> Self {
        Self {
            store,
            key: Vec::new(),
            aux: Vec::new(),
            selected: Input::Key,
            announced: 0,
            output: Vec::new(),
            read_position: 0,
            status: STATUS_OK,
        }
    }

    fn input(&mut self) -> &mut Vec<u8> {
        match self.selected {
            Input::Key => &mut self.key,
            Input::Aux => &mut self.aux,
        }
    }

    pub fn io_in(&mut self, port: u16, size: u8) -> u32 {
        match port - KV_PORT_BASE {
            REGISTER_COMMAND_STATUS => self.status,
            REGISTER_DATA => {
                let mut word = [0u8; 4];
                let rest = self.output.get(self.read_position..).unwrap_or(&[]);
                let take = rest.len().min(size.clamp(1, 4) as usize);
                word[..take].copy_from_slice(&rest[..take]);
                self.read_position += take;
                u32::from_le_bytes(word)
            }
            REGISTER_LENGTH => self.output.len() as u32,
            REGISTER_IDENT => IDENT,
            _ => u32::MAX,
        }
    }

    pub fn io_out(&mut self, port: u16, size: u8, value: u32) {
        match port - KV_PORT_BASE {
            REGISTER_COMMAND_STATUS => self.command(value),
            REGISTER_DATA => {
                let room = self.announced.saturating_sub(self.input().len());
                let take = room.min(size.clamp(1, 4) as usize);
                self.input().extend_from_slice(&value.to_le_bytes()[..take]);
            }
            REGISTER_LENGTH => self.announced = value as usize,
            _ => {}
        }
    }

    fn command(&mut self, command: u32) {
        self.output.clear();
        self.read_position = 0;
        self.status = STATUS_OK;
        // The TPM keeps its seeds and NV in the same store; the guest must not read or forge
        // them, so those keys do not exist as far as the port is concerned.
        let reserved = self.key.starts_with(TPM_STORE_PREFIX);
        if reserved && matches!(command, COMMAND_GET | COMMAND_SET | COMMAND_DELETE) {
            self.status = if command == COMMAND_GET {
                STATUS_NOT_FOUND
            } else {
                STATUS_BAD_COMMAND
            };
            return;
        }
        match command {
            COMMAND_SELECT_KEY | COMMAND_SELECT_AUX => {
                self.selected = if command == COMMAND_SELECT_KEY {
                    Input::Key
                } else {
                    Input::Aux
                };
                self.input().clear();
                self.announced = 0;
            }
            COMMAND_GET => match self.store.lock().unwrap().get(&self.key) {
                Some(value) => self.output = value,
                None => self.status = STATUS_NOT_FOUND,
            },
            COMMAND_SET => {
                if self.key.len() > MAX_KEY_BYTES || self.aux.len() > MAX_VALUE_BYTES {
                    self.status = STATUS_TOO_LARGE;
                } else if self
                    .store
                    .lock()
                    .unwrap()
                    .set(&self.key, &self.aux)
                    .is_err()
                {
                    self.status = STATUS_IO_ERROR;
                }
            }
            COMMAND_DELETE => {
                if self.store.lock().unwrap().delete(&self.key).is_err() {
                    self.status = STATUS_IO_ERROR;
                }
            }
            // The next key after the key buffer among those starting with the aux buffer.
            // Listing everything each time is quadratic over a full walk, which a few hundred
            // variables never notice.
            COMMAND_NEXT => {
                let keys = self.store.lock().unwrap().list(&self.aux);
                match keys.into_iter().find(|candidate| {
                    *candidate > self.key && !candidate.starts_with(TPM_STORE_PREFIX)
                }) {
                    Some(next) => self.output = next,
                    None => self.status = STATUS_NOT_FOUND,
                }
            }
            _ => self.status = STATUS_BAD_COMMAND,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::MemoryKeyValueStore;

    const ANNOUNCE: u16 = KV_PORT_BASE + REGISTER_LENGTH;
    const DATA: u16 = KV_PORT_BASE + REGISTER_DATA;
    const COMMAND: u16 = KV_PORT_BASE + REGISTER_COMMAND_STATUS;

    fn device() -> KvDevice {
        KvDevice::new(Arc::new(Mutex::new(Box::new(MemoryKeyValueStore::new()))))
    }

    /// What the firmware does: select, announce the length, stream dwords.
    fn put(device: &mut KvDevice, select: u32, bytes: &[u8]) {
        device.io_out(COMMAND, 4, select);
        device.io_out(ANNOUNCE, 4, bytes.len() as u32);
        for chunk in bytes.chunks(4) {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            device.io_out(DATA, 4, u32::from_le_bytes(word));
        }
    }

    fn run(device: &mut KvDevice, command: u32) -> u32 {
        device.io_out(COMMAND, 4, command);
        device.io_in(COMMAND, 4)
    }

    fn take_output(device: &mut KvDevice) -> Vec<u8> {
        let length = device.io_in(KV_PORT_BASE + REGISTER_LENGTH, 4) as usize;
        let mut bytes = Vec::new();
        while bytes.len() < length {
            bytes.extend_from_slice(&device.io_in(DATA, 4).to_le_bytes());
        }
        bytes.truncate(length);
        bytes
    }

    fn set(device: &mut KvDevice, key: &[u8], value: &[u8]) -> u32 {
        put(device, COMMAND_SELECT_KEY, key);
        put(device, COMMAND_SELECT_AUX, value);
        run(device, COMMAND_SET)
    }

    fn get(device: &mut KvDevice, key: &[u8]) -> Result<Vec<u8>, u32> {
        put(device, COMMAND_SELECT_KEY, key);
        match run(device, COMMAND_GET) {
            STATUS_OK => Ok(take_output(device)),
            status => Err(status),
        }
    }

    #[test]
    fn the_tpm_keys_are_out_of_the_guests_reach() {
        let store: SharedKeyValueStore = Arc::new(Mutex::new(Box::new(MemoryKeyValueStore::new())));
        store.lock().unwrap().set(b"tpm/state", b"seeds").unwrap();
        store.lock().unwrap().set(b"var", b"value").unwrap();
        let mut device = KvDevice::new(store.clone());
        assert_eq!(get(&mut device, b"tpm/state"), Err(STATUS_NOT_FOUND));
        assert_eq!(
            set(&mut device, b"tpm/state", b"forged"),
            STATUS_BAD_COMMAND
        );
        put(&mut device, COMMAND_SELECT_KEY, b"tpm/state");
        assert_eq!(run(&mut device, COMMAND_DELETE), STATUS_BAD_COMMAND);
        assert_eq!(store.lock().unwrap().get(b"tpm/state").unwrap(), b"seeds");
        // A walk of every key skips them.
        put(&mut device, COMMAND_SELECT_KEY, b"");
        put(&mut device, COMMAND_SELECT_AUX, b"");
        assert_eq!(run(&mut device, COMMAND_NEXT), STATUS_OK);
        assert_eq!(take_output(&mut device), b"var");
        put(&mut device, COMMAND_SELECT_KEY, b"tpm/");
        assert_eq!(run(&mut device, COMMAND_NEXT), STATUS_OK);
        assert_eq!(take_output(&mut device), b"var");
    }

    #[test]
    fn identifies_itself() {
        assert_eq!(device().io_in(KV_PORT_BASE + REGISTER_IDENT, 4), IDENT);
    }

    #[test]
    fn set_then_get_round_trips_values_of_every_alignment() {
        let mut device = device();
        for length in 0..11usize {
            let value: Vec<u8> = (0..length as u8).collect();
            let key = format!("key{length}");
            assert_eq!(set(&mut device, key.as_bytes(), &value), STATUS_OK);
            assert_eq!(get(&mut device, key.as_bytes()), Ok(value));
        }
    }

    #[test]
    fn a_missing_key_is_not_found_and_delete_removes() {
        let mut device = device();
        assert_eq!(get(&mut device, b"nope"), Err(STATUS_NOT_FOUND));
        set(&mut device, b"k", b"v");
        put(&mut device, COMMAND_SELECT_KEY, b"k");
        assert_eq!(run(&mut device, COMMAND_DELETE), STATUS_OK);
        assert_eq!(get(&mut device, b"k"), Err(STATUS_NOT_FOUND));
    }

    #[test]
    fn padding_of_the_last_dword_is_not_part_of_the_value() {
        let mut device = device();
        set(&mut device, b"k", b"abcde");
        assert_eq!(get(&mut device, b"k"), Ok(b"abcde".to_vec()));
    }

    #[test]
    fn next_walks_keys_with_a_prefix_in_order() {
        let mut device = device();
        for key in ["var/b", "var/a", "boot/x", "var/c"] {
            set(&mut device, key.as_bytes(), b"1");
        }
        let mut seen = Vec::new();
        let mut after: Vec<u8> = Vec::new();
        loop {
            put(&mut device, COMMAND_SELECT_KEY, &after);
            put(&mut device, COMMAND_SELECT_AUX, b"var/");
            if run(&mut device, COMMAND_NEXT) != STATUS_OK {
                break;
            }
            after = take_output(&mut device);
            seen.push(String::from_utf8(after.clone()).unwrap());
        }
        assert_eq!(seen, ["var/a", "var/b", "var/c"]);
    }

    #[test]
    fn oversized_values_and_unknown_commands_report_a_status() {
        let mut device = device();
        put(&mut device, COMMAND_SELECT_KEY, b"k");
        device.io_out(COMMAND, 4, COMMAND_SELECT_AUX);
        device.io_out(ANNOUNCE, 4, MAX_VALUE_BYTES as u32 + 1);
        for _ in 0..=MAX_VALUE_BYTES / 4 {
            device.io_out(DATA, 4, 0);
        }
        assert_eq!(run(&mut device, COMMAND_SET), STATUS_TOO_LARGE);
        assert_eq!(run(&mut device, 99), STATUS_BAD_COMMAND);
    }

    #[test]
    fn reads_past_the_output_return_zeroes() {
        let mut device = device();
        set(&mut device, b"k", b"ab");
        assert_eq!(get(&mut device, b"k"), Ok(b"ab".to_vec()));
        assert_eq!(device.io_in(DATA, 4), 0);
    }
}
