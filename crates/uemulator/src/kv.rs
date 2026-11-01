//! The embedder's persistent key-value store (UEFI variables live in it) and its implementations.
//! The guest reaches it through `dev/kv.rs`; the file format is in `docs/plans/19-uefi-firmware.md`.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;
use core::fmt;

/// Longest key the guest device accepts. A UEFI variable key is the vendor GUID text plus the
/// UTF-8 name, which the firmware caps well below this.
pub const MAX_KEY_BYTES: usize = 1024;
/// Longest value the guest device accepts (a variable's attributes plus its data).
pub const MAX_VALUE_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValueError(pub String);

impl fmt::Display for KeyValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "key-value store: {}", self.0)
    }
}

impl core::error::Error for KeyValueError {}

/// Byte-string keys to byte-string values. `Send` because the vCPU runs on its own thread.
/// An implementation that persists must make `set` and `delete` durable before returning: the
/// guest treats a successful `SetVariable` as stored.
pub trait KeyValueStore: Send {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>>;
    fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), KeyValueError>;
    /// Removing a key that is not there is not an error.
    fn delete(&mut self, key: &[u8]) -> Result<(), KeyValueError>;
    /// Every key starting with `prefix` (all keys for an empty prefix), in byte order.
    fn list(&mut self, prefix: &[u8]) -> Vec<Vec<u8>>;
}

#[derive(Default, Clone)]
pub struct MemoryKeyValueStore {
    entries: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl MemoryKeyValueStore {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(not(freestanding))]
    fn live_bytes(&self) -> usize {
        self.entries.iter().map(|(k, v)| k.len() + v.len()).sum()
    }
}

impl KeyValueStore for MemoryKeyValueStore {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.entries.get(key).cloned()
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), KeyValueError> {
        self.entries.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&mut self, key: &[u8]) -> Result<(), KeyValueError> {
        self.entries.remove(key);
        Ok(())
    }

    fn list(&mut self, prefix: &[u8]) -> Vec<Vec<u8>> {
        // Keys sort bytewise, so everything with the prefix is contiguous from `prefix` on.
        self.entries
            .range(prefix.to_vec()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(key, _)| key.clone())
            .collect()
    }
}

#[cfg(not(freestanding))]
pub use file::FileKeyValueStore;

/// The zlib/IEEE 802.3 CRC-32 (reflected), which the store's records and the udisk header use.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    const CRC32_POLYNOMIAL: u32 = 0xedb8_8320;
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (CRC32_POLYNOMIAL & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

#[cfg(not(freestanding))]
mod file {
    use super::*;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 8] = b"UEMUKV01";
    const OP_SET: u8 = 1;
    const OP_DELETE: u8 = 2;
    /// length u32, crc32 u32; then the payload the length counts.
    const RECORD_HEADER_BYTES: usize = 8;
    /// op u8, key length u16; then the key and (for a set) the value.
    const PAYLOAD_HEADER_BYTES: usize = 3;
    /// Rewrite the log once it holds this many bytes beyond what the live entries need, so a
    /// guest that rewrites one variable forever does not grow the file without bound.
    const WASTE_COMPACTION_BYTES: u64 = 1 << 20;

    fn encode_record(op: u8, key: &[u8], value: &[u8]) -> Vec<u8> {
        let payload_len = PAYLOAD_HEADER_BYTES + key.len() + value.len();
        let mut record = Vec::with_capacity(RECORD_HEADER_BYTES + payload_len);
        record.extend_from_slice(&(payload_len as u32).to_le_bytes());
        record.extend_from_slice(&[0; 4]);
        record.push(op);
        record.extend_from_slice(&(key.len() as u16).to_le_bytes());
        record.extend_from_slice(key);
        record.extend_from_slice(value);
        let crc = crc32(&record[RECORD_HEADER_BYTES..]);
        record[4..8].copy_from_slice(&crc.to_le_bytes());
        record
    }

    /// Applies the records of `log` to `entries` and returns how many bytes were good. The log
    /// ends at the first record that is short or fails its checksum: that is a write the host
    /// died in the middle of, and nothing after it can be trusted.
    fn replay(log: &[u8], entries: &mut BTreeMap<Vec<u8>, Vec<u8>>) -> usize {
        let mut at = 0;
        while let Some(header) = log.get(at..at + RECORD_HEADER_BYTES) {
            let payload_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(header[4..8].try_into().unwrap());
            let start = at + RECORD_HEADER_BYTES;
            let Some(payload) = log.get(start..start.saturating_add(payload_len)) else {
                break;
            };
            if payload_len < PAYLOAD_HEADER_BYTES || crc32(payload) != crc {
                break;
            }
            let key_len = u16::from_le_bytes([payload[1], payload[2]]) as usize;
            let Some(key) = payload.get(PAYLOAD_HEADER_BYTES..PAYLOAD_HEADER_BYTES + key_len)
            else {
                break;
            };
            match payload[0] {
                OP_SET => {
                    entries.insert(
                        key.to_vec(),
                        payload[PAYLOAD_HEADER_BYTES + key_len..].to_vec(),
                    );
                }
                OP_DELETE => {
                    entries.remove(key);
                }
                _ => break,
            }
            at = start + payload_len;
        }
        at
    }

    /// A `MemoryKeyValueStore` backed by an append-only log. Every `set` and `delete` appends
    /// one checksummed record and syncs it. Opening replays the log and rewrites it compacted
    /// (one record per live key) through a temporary file and an atomic rename, which also
    /// drops a torn last record. Single writer: two processes on one file will lose updates.
    pub struct FileKeyValueStore {
        path: PathBuf,
        memory: MemoryKeyValueStore,
        log: File,
        log_bytes: u64,
    }

    fn io_error(what: &str, path: &Path, error: std::io::Error) -> KeyValueError {
        KeyValueError(format!("{what} {}: {error}", path.display()))
    }

    impl FileKeyValueStore {
        /// Opens `path`, creating it when missing. A file that does not start with the format's
        /// magic is refused rather than overwritten.
        pub fn open(path: impl Into<PathBuf>) -> Result<Self, KeyValueError> {
            let path = path.into();
            let mut memory = MemoryKeyValueStore::new();
            match File::open(&path) {
                Ok(mut file) => {
                    let mut bytes = Vec::new();
                    file.read_to_end(&mut bytes)
                        .map_err(|e| io_error("reading", &path, e))?;
                    if !bytes.is_empty() {
                        if !bytes.starts_with(MAGIC) {
                            return Err(KeyValueError(format!(
                                "{} is not a uemulator key-value file",
                                path.display()
                            )));
                        }
                        replay(&bytes[MAGIC.len()..], &mut memory.entries);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io_error("opening", &path, e)),
            }
            let (log, log_bytes) = Self::write_compacted(&path, &memory)?;
            Ok(Self {
                path,
                memory,
                log,
                log_bytes,
            })
        }

        /// Writes `memory` to `path.tmp`, syncs it and renames it over `path`; returns the
        /// new file opened for appending. A crash before the rename leaves the old file whole.
        fn write_compacted(
            path: &Path,
            memory: &MemoryKeyValueStore,
        ) -> Result<(File, u64), KeyValueError> {
            let mut temporary = path.as_os_str().to_owned();
            temporary.push(".tmp");
            let temporary = PathBuf::from(temporary);
            let mut contents = MAGIC.to_vec();
            for (key, value) in &memory.entries {
                contents.extend_from_slice(&encode_record(OP_SET, key, value));
            }
            let mut file =
                File::create(&temporary).map_err(|e| io_error("creating", &temporary, e))?;
            file.write_all(&contents)
                .and_then(|()| file.sync_all())
                .map_err(|e| io_error("writing", &temporary, e))?;
            drop(file);
            fs::rename(&temporary, path).map_err(|e| io_error("renaming to", path, e))?;
            // Persist the rename itself. Not possible on every platform, so best effort.
            if let Some(parent) = path.parent() {
                let parent = if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                };
                if let Ok(directory) = File::open(parent) {
                    let _ = directory.sync_all();
                }
            }
            let log = OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(|e| io_error("opening", path, e))?;
            Ok((log, contents.len() as u64))
        }

        fn append(&mut self, record: &[u8]) -> Result<(), KeyValueError> {
            self.log
                .write_all(record)
                .and_then(|()| self.log.sync_data())
                .map_err(|e| io_error("appending to", &self.path, e))?;
            self.log_bytes += record.len() as u64;
            let live = (MAGIC.len() + self.memory.live_bytes()) as u64;
            if self.log_bytes > live + WASTE_COMPACTION_BYTES {
                let (log, log_bytes) = Self::write_compacted(&self.path, &self.memory)?;
                self.log = log;
                self.log_bytes = log_bytes;
            }
            Ok(())
        }
    }

    fn check_sizes(key: &[u8], value: &[u8]) -> Result<(), KeyValueError> {
        if key.len() > MAX_KEY_BYTES || value.len() > MAX_VALUE_BYTES {
            return Err(KeyValueError("key or value too long".into()));
        }
        Ok(())
    }

    impl KeyValueStore for FileKeyValueStore {
        fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
            self.memory.get(key)
        }

        fn set(&mut self, key: &[u8], value: &[u8]) -> Result<(), KeyValueError> {
            check_sizes(key, value)?;
            // Durable before visible: a failed append leaves the in-memory copy unchanged.
            self.append(&encode_record(OP_SET, key, value))?;
            self.memory.set(key, value)
        }

        fn delete(&mut self, key: &[u8]) -> Result<(), KeyValueError> {
            if self.memory.get(key).is_none() {
                return Ok(());
            }
            self.append(&encode_record(OP_DELETE, key, &[]))?;
            self.memory.delete(key)
        }

        fn list(&mut self, prefix: &[u8]) -> Vec<Vec<u8>> {
            self.memory.list(prefix)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn temp_path(name: &str) -> PathBuf {
            let dir =
                std::env::temp_dir().join(format!("uemulator-kv-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            dir.join("nvram.kv")
        }

        #[test]
        fn crc32_matches_the_standard_check_value() {
            assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        }

        #[test]
        fn values_survive_reopening() {
            let path = temp_path("reopen");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            store.set(b"a", b"1").unwrap();
            store.set(b"b", b"2").unwrap();
            store.set(b"a", b"3").unwrap();
            store.delete(b"b").unwrap();
            drop(store);
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert_eq!(store.get(b"a"), Some(b"3".to_vec()));
            assert_eq!(store.get(b"b"), None);
            assert_eq!(store.list(b""), vec![b"a".to_vec()]);
        }

        #[test]
        fn opening_compacts_the_log() {
            let path = temp_path("compact");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            for round in 0..50u8 {
                store.set(b"key", &[round; 100]).unwrap();
            }
            let before = fs::metadata(&path).unwrap().len();
            drop(store);
            let mut store = FileKeyValueStore::open(&path).unwrap();
            let after = fs::metadata(&path).unwrap().len();
            assert!(after < before / 10, "{after} vs {before}");
            assert_eq!(store.get(b"key"), Some(vec![49; 100]));
        }

        #[test]
        fn a_torn_last_record_is_dropped_and_the_rest_kept() {
            let path = temp_path("torn");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            store.set(b"keep", b"yes").unwrap();
            store.set(b"torn", b"partial write").unwrap();
            drop(store);
            let mut bytes = fs::read(&path).unwrap();
            bytes.truncate(bytes.len() - 5);
            fs::write(&path, &bytes).unwrap();
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert_eq!(store.get(b"keep"), Some(b"yes".to_vec()));
            assert_eq!(store.get(b"torn"), None);
            store.set(b"after", b"ok").unwrap();
            drop(store);
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert_eq!(store.list(b""), vec![b"after".to_vec(), b"keep".to_vec()]);
        }

        #[test]
        fn a_flipped_byte_ends_the_log_at_that_record() {
            let path = temp_path("corrupt");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            store.set(b"first", b"1").unwrap();
            store.set(b"second", b"2").unwrap();
            store.set(b"third", b"3").unwrap();
            drop(store);
            let mut bytes = fs::read(&path).unwrap();
            let second = MAGIC.len() + encode_record(OP_SET, b"first", b"1").len();
            bytes[second + RECORD_HEADER_BYTES + PAYLOAD_HEADER_BYTES] ^= 0xff;
            fs::write(&path, &bytes).unwrap();
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert_eq!(store.list(b""), vec![b"first".to_vec()]);
        }

        #[test]
        fn a_stale_temporary_file_from_a_crashed_compaction_is_ignored() {
            let path = temp_path("stale-tmp");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            store.set(b"k", b"v").unwrap();
            drop(store);
            let mut temporary = path.as_os_str().to_owned();
            temporary.push(".tmp");
            fs::write(&temporary, b"garbage half written").unwrap();
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert_eq!(store.get(b"k"), Some(b"v".to_vec()));
        }

        #[test]
        fn a_foreign_file_is_refused_not_overwritten() {
            let path = temp_path("foreign");
            fs::write(&path, b"this is not a key-value file").unwrap();
            assert!(FileKeyValueStore::open(&path).is_err());
            assert_eq!(fs::read(&path).unwrap(), b"this is not a key-value file");
        }

        #[test]
        fn oversized_entries_are_rejected() {
            let path = temp_path("oversize");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            assert!(store.set(&[0; MAX_KEY_BYTES + 1], b"").is_err());
            assert!(store.set(b"k", &vec![0; MAX_VALUE_BYTES + 1]).is_err());
            assert!(store.set(b"k", &vec![0; MAX_VALUE_BYTES]).is_ok());
        }

        #[test]
        fn rewriting_one_key_forever_does_not_grow_the_file_without_bound() {
            let path = temp_path("waste");
            let mut store = FileKeyValueStore::open(&path).unwrap();
            let value = vec![7u8; 64 * 1024];
            for _ in 0..100 {
                store.set(b"k", &value).unwrap();
            }
            let size = fs::metadata(&path).unwrap().len();
            assert!(size < 2 * WASTE_COMPACTION_BYTES + value.len() as u64);
            drop(store);
            assert_eq!(
                FileKeyValueStore::open(&path).unwrap().get(b"k"),
                Some(value)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_lists_by_prefix_in_byte_order() {
        let mut store = MemoryKeyValueStore::new();
        for key in ["var/b", "var/a", "other", "va", "var/"] {
            store.set(key.as_bytes(), b"x").unwrap();
        }
        let keys: Vec<_> = store
            .list(b"var/")
            .into_iter()
            .map(|key| String::from_utf8(key).unwrap())
            .collect();
        assert_eq!(keys, ["var/", "var/a", "var/b"]);
        assert_eq!(store.list(b"").len(), 5);
        assert!(store.list(b"zzz").is_empty());
    }

    #[test]
    fn memory_store_overwrites_and_deletes() {
        let mut store = MemoryKeyValueStore::new();
        store.set(b"k", b"1").unwrap();
        store.set(b"k", b"2").unwrap();
        assert_eq!(store.get(b"k"), Some(b"2".to_vec()));
        store.delete(b"k").unwrap();
        store.delete(b"k").unwrap();
        assert_eq!(store.get(b"k"), None);
    }
}
