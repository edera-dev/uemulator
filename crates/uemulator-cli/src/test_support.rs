//! Helpers shared by the unit tests of this crate: a fixed-seed PRNG, a self-deleting temp
//! directory and a guard that serializes tests touching process environment variables.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

/// xorshift64*; deterministic so a fuzz failure reproduces from the seed alone.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        // A zero state would stay zero forever.
        XorShift(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        const MULTIPLIER: u64 = 0x2545_f491_4f6c_dd1d;
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(MULTIPLIER)
    }

    /// Uniform enough for tests: `0..bound` (`bound` must be non-zero).
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

/// A directory under the system temp dir, removed on drop. The name is unique per process and call.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "uemulator-cli-test-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Sets or clears environment variables for the lifetime of the guard and restores the previous
/// values afterwards. Holding the guard excludes every other test that holds one, which is the only
/// thing that makes mutating the environment sound in a multi-threaded test binary.
pub struct EnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    pub fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut saved = Vec::new();
        for (name, value) in vars {
            saved.push((*name, std::env::var_os(name)));
            // SAFETY: all environment mutation in this test binary goes through this guard.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
        EnvGuard { saved, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..).rev() {
            // SAFETY: still holding `_lock`.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

/// RFC 1071 internet checksum written independently of the production routines (byte at a time,
/// 32-bit accumulator), for checking them.
pub fn reference_checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    for (index, byte) in data.iter().enumerate() {
        sum += if index % 2 == 0 {
            (*byte as u32) << 8
        } else {
            *byte as u32
        };
    }
    while sum > 0xffff {
        sum = (sum >> 16) + (sum & 0xffff);
    }
    !(sum as u16)
}

/// The header of a snapshot as `checkpoints::chain` reads it: format 2, full or a delta of
/// `parent`, ids made of one repeated byte. Nothing past the header: enough for the chain to be
/// found, not to be restored.
pub fn snapshot_header(id: u8, parent: Option<u8>) -> Vec<u8> {
    let mut header = b"UEMUSNAP".to_vec();
    header.extend_from_slice(&2u32.to_le_bytes());
    header.push(u8::from(parent.is_some()));
    header.extend_from_slice(&[id; 16]);
    header.extend_from_slice(&[parent.unwrap_or(0); 16]);
    header
}
