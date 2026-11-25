//! Machine snapshots: the whole state of a stopped machine, written to a host sink and read back
//! into a new machine built from the same configuration (`docs/plans/29-snapshots.md`).
//!
//! Every piece of state implements `Snapshot`, nearly always through `#[derive(Snapshot)]`
//! (`crates/uemulator-snapshot-derive`), and a restore loads *into* the freshly built machine:
//! host handles, callbacks and caches stay as the new machine made them (`#[snapshot(skip)]`),
//! and only what the guest can observe is overwritten. That is why the configuration must match
//! (`Fingerprint`): the snapshot carries no devices, only their state.
//!
//! The encoding is explicit little-endian with fixed widths (`usize` as `u64`), the same on every
//! host, so the native engine and the WebAssembly one read each other's. Each saved struct starts
//! with a tag hashed from its field names; a layout change fails at the first changed type
//! rather than misreading the rest. A CRC-32 of everything written ends the stream.
//!
//! Reading never trusts a length: collections grow as their elements arrive, so a corrupt or
//! hostile file cannot make the library allocate gigabytes or panic.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::string::String;
use core::cell::Cell;
use core::fmt;

pub(crate) use uemulator_snapshot_derive::Snapshot;

pub(crate) mod compress;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotError(pub String);

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "snapshot: {}", self.0)
    }
}

impl core::error::Error for SnapshotError {}

/// Names one snapshot. A delta carries its parent's id, so a host can line a chain up from the
/// files' headers (`read_snapshot_info`) and a restore can refuse a chain that does not fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId(pub [u8; 16]);

impl fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// What a snapshot's header says (`read_snapshot_info`), or what a checkpoint wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotInfo {
    /// The format version: 1 wrote every page as is and has no id (it reads as all zeros); 2
    /// compresses pages and may be a delta.
    pub version: u32,
    pub id: SnapshotId,
    /// For a delta, the snapshot its pages apply to: restore that one (and its own parents)
    /// first. `None` for a full snapshot.
    pub parent: Option<SnapshotId>,
}

/// Where a suspended machine's snapshot goes (`MachineConfig::suspend_to`). Writes arrive in
/// order, in pieces of up to `Writer::BUFFER_BYTES`. `finish` comes after the last one, and the
/// snapshot only counts once it returns Ok: write to a temporary file and rename it there, so a
/// host that dies halfway leaves no snapshot rather than a torn one.
pub trait SnapshotSink: Send {
    fn write(&mut self, bytes: &[u8]) -> Result<(), SnapshotError>;
    fn finish(&mut self) -> Result<(), SnapshotError>;
}

/// Where a machine resumes from (`MachineConfig::resume_from`). `read` fills as much of `buffer`
/// as it can and returns how much; 0 is the end. `finish` comes once the whole snapshot has been
/// read and checked, just before the guest runs again: from then on the guest writes its disks,
/// so the snapshot no longer matches them and the host must not resume from it a second time
/// (delete it there; `docs/plans/29-snapshots.md`, "I/O consistency").
pub trait SnapshotSource: Send {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, SnapshotError>;
    fn finish(&mut self) -> Result<(), SnapshotError>;
    /// Whether the snapshot stays where it is after `finish`. Only then may the machine's next
    /// checkpoint be a delta against it; otherwise that checkpoint is a full snapshot. A
    /// checkpoint is kept (its disks are copied beside it), a suspend's snapshot is not.
    fn retained(&self) -> bool {
        false
    }
}

/// A snapshot in memory, for tests and for hosts that keep it themselves.
#[derive(Default, Clone)]
pub struct MemorySnapshot {
    pub bytes: Vec<u8>,
    position: usize,
}

impl MemorySnapshot {
    pub fn new(bytes: Vec<u8>) -> Self {
        MemorySnapshot { bytes, position: 0 }
    }
}

impl SnapshotSink for MemorySnapshot {
    fn write(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        Ok(())
    }
}

impl SnapshotSource for MemorySnapshot {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, SnapshotError> {
        let remaining = &self.bytes[self.position..];
        let count = remaining.len().min(buffer.len());
        buffer[..count].copy_from_slice(&remaining[..count]);
        self.position += count;
        Ok(count)
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        Ok(())
    }

    fn retained(&self) -> bool {
        true
    }
}

#[cfg(not(freestanding))]
pub use file::{FileSnapshotSink, FileSnapshotSource};

#[cfg(not(freestanding))]
mod file {
    use super::{SnapshotError, SnapshotSink, SnapshotSource};
    use std::fs::{self, File};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    fn io_error(what: &str, path: &Path, error: std::io::Error) -> SnapshotError {
        SnapshotError(format!("{what} {}: {error}", path.display()))
    }

    /// A snapshot written to `path.tmp`, synced and renamed to `path` by `finish`: until then
    /// `path` holds the previous snapshot or nothing, never a torn one. Nothing touches the
    /// filesystem before the first write, so a machine that is never suspended leaves no file.
    pub struct FileSnapshotSink {
        path: PathBuf,
        temporary: PathBuf,
        file: Option<File>,
    }

    impl FileSnapshotSink {
        pub fn new(path: impl Into<PathBuf>) -> Self {
            let path = path.into();
            let mut temporary = path.as_os_str().to_owned();
            temporary.push(".tmp");
            FileSnapshotSink {
                path,
                temporary: PathBuf::from(temporary),
                file: None,
            }
        }

        fn file(&mut self) -> Result<&mut File, SnapshotError> {
            if self.file.is_none() {
                let file = File::create(&self.temporary)
                    .map_err(|e| io_error("creating", &self.temporary, e))?;
                self.file = Some(file);
            }
            Ok(self.file.as_mut().unwrap())
        }
    }

    impl SnapshotSink for FileSnapshotSink {
        fn write(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
            let temporary = self.temporary.clone();
            self.file()?
                .write_all(bytes)
                .map_err(|e| io_error("writing", &temporary, e))
        }

        fn finish(&mut self) -> Result<(), SnapshotError> {
            let temporary = self.temporary.clone();
            self.file()?
                .sync_all()
                .map_err(|e| io_error("syncing", &temporary, e))?;
            fs::rename(&self.temporary, &self.path)
                .map_err(|e| io_error("renaming to", &self.path, e))?;
            // The rename itself, where the platform allows syncing a directory.
            if let Some(parent) = self.path.parent()
                && let Ok(directory) = File::open(if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                })
            {
                let _ = directory.sync_all();
            }
            Ok(())
        }
    }

    /// A snapshot read from a file, which `finish` deletes: the guest is about to write its disks,
    /// and resuming from the file again would hand it a state that no longer matches them.
    pub struct FileSnapshotSource {
        path: PathBuf,
        file: File,
        keep: bool,
    }

    impl FileSnapshotSource {
        pub fn open(path: impl Into<PathBuf>) -> Result<Self, SnapshotError> {
            let path = path.into();
            let file = File::open(&path).map_err(|e| io_error("opening", &path, e))?;
            Ok(FileSnapshotSource {
                path,
                file,
                keep: false,
            })
        }

        /// A checkpoint: `finish` leaves the file, so it can be resumed again (with the disks
        /// as they were when it was taken) and later checkpoints can be deltas against it.
        pub fn open_kept(path: impl Into<PathBuf>) -> Result<Self, SnapshotError> {
            Ok(FileSnapshotSource {
                keep: true,
                ..Self::open(path)?
            })
        }
    }

    impl SnapshotSource for FileSnapshotSource {
        fn read(&mut self, buffer: &mut [u8]) -> Result<usize, SnapshotError> {
            self.file
                .read(buffer)
                .map_err(|e| io_error("reading", &self.path, e))
        }

        fn finish(&mut self) -> Result<(), SnapshotError> {
            if self.keep {
                return Ok(());
            }
            fs::remove_file(&self.path).map_err(|e| io_error("removing", &self.path, e))
        }

        fn retained(&self) -> bool {
            self.keep
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_snapshot_file_appears_only_when_finished_and_goes_once_resumed() {
            let directory = std::env::temp_dir()
                .join(format!("uemulator-snapshot-file-{}", std::process::id()));
            let _ = fs::remove_dir_all(&directory);
            fs::create_dir_all(&directory).unwrap();
            let path = directory.join("machine.snapshot");
            let mut sink = FileSnapshotSink::new(&path);
            sink.write(b"state").unwrap();
            assert!(
                !path.exists(),
                "a snapshot is visible before it is finished"
            );
            sink.finish().unwrap();
            let mut source = FileSnapshotSource::open(&path).unwrap();
            let mut buffer = [0; 16];
            assert_eq!(source.read(&mut buffer).unwrap(), 5);
            assert_eq!(&buffer[..5], b"state");
            source.finish().unwrap();
            assert!(!path.exists(), "a resumed snapshot was kept");
            let _ = fs::remove_dir_all(&directory);
        }
    }
}

/// The table-driven form of `kv::crc32` (same polynomial and result): a snapshot runs gigabytes
/// of RAM through it, which the bitwise loop would take seconds over.
struct Crc32 {
    table: [u32; 256],
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        const CRC32_POLYNOMIAL: u32 = 0xedb8_8320;
        let mut table = [0u32; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            let mut crc = index as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ (CRC32_POLYNOMIAL & (crc & 1).wrapping_neg());
            }
            *entry = crc;
        }
        Crc32 { table, value: !0 }
    }

    fn update(&mut self, bytes: &[u8]) {
        let mut crc = self.value;
        for &byte in bytes {
            crc = self.table[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
        }
        self.value = crc;
    }

    fn finish(&self) -> u32 {
        !self.value
    }
}

pub(crate) struct Writer<'a> {
    sink: &'a mut dyn SnapshotSink,
    buffer: Vec<u8>,
    crc: Crc32,
}

impl<'a> Writer<'a> {
    /// How much is gathered before the sink sees it: big enough that a host which crosses a
    /// process boundary per write (the WebKit engine's POSTs) does few of them.
    pub(crate) const BUFFER_BYTES: usize = 1 << 20;

    pub(crate) fn new(sink: &'a mut dyn SnapshotSink) -> Self {
        Writer {
            sink,
            buffer: Vec::with_capacity(Self::BUFFER_BYTES),
            crc: Crc32::new(),
        }
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.crc.update(bytes);
        let mut rest = bytes;
        while !rest.is_empty() {
            let room = Self::BUFFER_BYTES - self.buffer.len();
            let (now, later) = rest.split_at(room.min(rest.len()));
            self.buffer.extend_from_slice(now);
            rest = later;
            if self.buffer.len() == Self::BUFFER_BYTES {
                self.flush()?;
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), SnapshotError> {
        if !self.buffer.is_empty() {
            self.sink.write(&self.buffer)?;
            self.buffer.clear();
        }
        Ok(())
    }

    pub(crate) fn u8(&mut self, value: u8) -> Result<(), SnapshotError> {
        self.bytes(&[value])
    }

    pub(crate) fn u16(&mut self, value: u16) -> Result<(), SnapshotError> {
        self.bytes(&value.to_le_bytes())
    }

    pub(crate) fn u32(&mut self, value: u32) -> Result<(), SnapshotError> {
        self.bytes(&value.to_le_bytes())
    }

    pub(crate) fn u64(&mut self, value: u64) -> Result<(), SnapshotError> {
        self.bytes(&value.to_le_bytes())
    }

    pub(crate) fn len(&mut self, value: usize) -> Result<(), SnapshotError> {
        self.u64(value as u64)
    }

    pub(crate) fn tag(&mut self, tag: u32) -> Result<(), SnapshotError> {
        self.u32(tag)
    }

    /// Writes the checksum of everything so far, then hands the rest to the sink and finishes it.
    pub(crate) fn finish(mut self) -> Result<(), SnapshotError> {
        let crc = self.crc.finish();
        self.buffer.extend_from_slice(&crc.to_le_bytes());
        self.flush()?;
        self.sink.finish()
    }
}

pub(crate) struct Reader<'a> {
    source: &'a mut dyn SnapshotSource,
    buffer: Vec<u8>,
    start: usize,
    end: usize,
    crc: Crc32,
}

impl<'a> Reader<'a> {
    const BUFFER_BYTES: usize = 1 << 20;
    /// Longest collection a snapshot may describe. Far above anything a machine holds (the
    /// longest are byte buffers of a few MiB), far below what would exhaust memory before the
    /// elements ran out.
    const MAX_LENGTH: u64 = 1 << 32;

    pub(crate) fn new(source: &'a mut dyn SnapshotSource) -> Self {
        Reader {
            source,
            buffer: vec![0; Self::BUFFER_BYTES],
            start: 0,
            end: 0,
            crc: Crc32::new(),
        }
    }

    pub(crate) fn error(&self, message: impl Into<String>) -> SnapshotError {
        SnapshotError(message.into())
    }

    pub(crate) fn bad_variant(&self, name: &str, variant: u32) -> SnapshotError {
        self.error(format!("{name}: no variant {variant}"))
    }

    pub(crate) fn bytes(&mut self, out: &mut [u8]) -> Result<(), SnapshotError> {
        let mut filled = 0;
        while filled < out.len() {
            if self.start == self.end {
                self.start = 0;
                self.end = self.source.read(&mut self.buffer)?;
                if self.end == 0 {
                    return Err(self.error("the snapshot ends early"));
                }
            }
            let count = (self.end - self.start).min(out.len() - filled);
            out[filled..filled + count]
                .copy_from_slice(&self.buffer[self.start..self.start + count]);
            self.start += count;
            filled += count;
        }
        self.crc.update(out);
        Ok(())
    }

    pub(crate) fn u8(&mut self) -> Result<u8, SnapshotError> {
        let mut byte = [0];
        self.bytes(&mut byte)?;
        Ok(byte[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, SnapshotError> {
        let mut bytes = [0; 2];
        self.bytes(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, SnapshotError> {
        let mut bytes = [0; 4];
        self.bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, SnapshotError> {
        let mut bytes = [0; 8];
        self.bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub(crate) fn len(&mut self) -> Result<usize, SnapshotError> {
        let length = self.u64()?;
        if length > Self::MAX_LENGTH {
            return Err(self.error(format!("a length of {length} is not plausible")));
        }
        usize::try_from(length).map_err(|_| self.error("a length does not fit this host"))
    }

    pub(crate) fn tag(&mut self, expected: u32, name: &str) -> Result<(), SnapshotError> {
        let tag = self.u32()?;
        if tag != expected {
            return Err(self.error(format!(
                "{name} is laid out differently in the snapshot: it comes from another build"
            )));
        }
        Ok(())
    }

    /// Checks the checksum `Writer::finish` wrote against what was read.
    pub(crate) fn finish(mut self) -> Result<(), SnapshotError> {
        let computed = self.crc.finish();
        let mut stored = [0; 4];
        self.bytes(&mut stored)?;
        if u32::from_le_bytes(stored) != computed {
            return Err(self.error("the snapshot is damaged (checksum mismatch)"));
        }
        let mut extra = [0];
        if self.start != self.end || self.source.read(&mut extra)? != 0 {
            return Err(self.error("the snapshot has bytes after its end"));
        }
        Ok(())
    }
}

/// A piece of machine state that a snapshot carries. `load` restores into an existing value,
/// keeping whatever the type does not save (host handles); `load_new` builds a value, which only
/// plain data can (`#[snapshot(new)]` on a derived struct).
pub(crate) trait Snapshot {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError>;
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError>;
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError>
    where
        Self: Sized,
    {
        Err(r.error(format!(
            "{} cannot be created from a snapshot: the new machine is configured differently",
            core::any::type_name::<Self>()
        )))
    }
}

macro_rules! snapshot_integers {
    ($($ty:ty),*) => {$(
        impl Snapshot for $ty {
            fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
                w.bytes(&self.to_le_bytes())
            }
            fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
                *self = Self::load_new(r)?;
                Ok(())
            }
            fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
                let mut bytes = [0; core::mem::size_of::<$ty>()];
                r.bytes(&mut bytes)?;
                Ok(<$ty>::from_le_bytes(bytes))
            }
        }
    )*};
}

snapshot_integers!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl Snapshot for usize {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.u64(*self as u64)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let value = r.u64()?;
        usize::try_from(value).map_err(|_| r.error("a value does not fit this host's usize"))
    }
}

impl Snapshot for isize {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        (*self as i64).save(w)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let value = i64::load_new(r)?;
        isize::try_from(value).map_err(|_| r.error("a value does not fit this host's isize"))
    }
}

impl Snapshot for bool {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.u8(*self as u8)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        match r.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(r.error(format!("{other} is not a bool"))),
        }
    }
}

impl Snapshot for f32 {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        self.to_bits().save(w)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(f32::from_bits(u32::load_new(r)?))
    }
}

impl Snapshot for f64 {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        self.to_bits().save(w)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(f64::from_bits(u64::load_new(r)?))
    }
}

impl Snapshot for () {
    fn save(&self, _: &mut Writer<'_>) -> Result<(), SnapshotError> {
        Ok(())
    }
    fn load(&mut self, _: &mut Reader<'_>) -> Result<(), SnapshotError> {
        Ok(())
    }
    fn load_new(_: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(())
    }
}

impl Snapshot for String {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.len(self.len())?;
        w.bytes(self.as_bytes())
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let bytes = Vec::<u8>::load_new(r)?;
        String::from_utf8(bytes).map_err(|_| r.error("a string is not UTF-8"))
    }
}

impl<T: Snapshot + Copy> Snapshot for Cell<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        self.get().save(w)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        self.get_mut().load(r)
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(Cell::new(T::load_new(r)?))
    }
}

/// Sized contents only: a `Box<dyn Trait>` holding device state gets an impl of its own that
/// forwards to the trait (`dev/virtio.rs`), and cannot be built from a snapshot.
impl<T: Snapshot> Snapshot for Box<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        (**self).save(w)
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        (**self).load(r)
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(Box::new(T::load_new(r)?))
    }
}

impl<T: Snapshot> Snapshot for Option<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        match self {
            None => w.u8(0),
            Some(value) => {
                w.u8(1)?;
                value.save(w)
            }
        }
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        match (r.u8()?, self.as_mut()) {
            (0, _) => *self = None,
            (1, Some(value)) => value.load(r)?,
            (1, None) => *self = Some(T::load_new(r)?),
            (other, _) => return Err(r.error(format!("{other} is not an Option"))),
        }
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        match r.u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::load_new(r)?)),
            other => Err(r.error(format!("{other} is not an Option"))),
        }
    }
}

impl<T: Snapshot> Snapshot for Vec<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.len(self.len())?;
        self.iter().try_for_each(|item| item.save(w))
    }
    /// In place when the lengths agree (a machine's devices, its APICs), so elements keep their
    /// host state; otherwise built anew.
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        let length = r.len()?;
        if length == self.len() {
            return self.iter_mut().try_for_each(|item| item.load(r));
        }
        *self = Vec::new();
        for _ in 0..length {
            self.push(T::load_new(r)?);
        }
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let length = r.len()?;
        let mut items = Vec::new();
        for _ in 0..length {
            items.push(T::load_new(r)?);
        }
        Ok(items)
    }
}

impl<T: Snapshot> Snapshot for VecDeque<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.len(self.len())?;
        self.iter().try_for_each(|item| item.save(w))
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(Vec::<T>::load_new(r)?.into())
    }
}

impl<K: Snapshot + Ord, V: Snapshot> Snapshot for BTreeMap<K, V> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.len(self.len())?;
        for (key, value) in self {
            key.save(w)?;
            value.save(w)?;
        }
        Ok(())
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let length = r.len()?;
        let mut map = BTreeMap::new();
        for _ in 0..length {
            let key = K::load_new(r)?;
            let value = V::load_new(r)?;
            map.insert(key, value);
        }
        Ok(map)
    }
}

impl<T: Snapshot + Ord> Snapshot for BTreeSet<T> {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        w.len(self.len())?;
        self.iter().try_for_each(|item| item.save(w))
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        *self = Self::load_new(r)?;
        Ok(())
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        Ok(Vec::<T>::load_new(r)?.into_iter().collect())
    }
}

impl<T: Snapshot, const N: usize> Snapshot for [T; N] {
    fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
        self.iter().try_for_each(|item| item.save(w))
    }
    fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
        self.iter_mut().try_for_each(|item| item.load(r))
    }
    fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
        let mut items = Vec::with_capacity(N);
        for _ in 0..N {
            items.push(T::load_new(r)?);
        }
        items
            .try_into()
            .map_err(|_| r.error("an array came back the wrong length"))
    }
}

macro_rules! snapshot_tuples {
    ($(($($name:ident $index:tt),+))*) => {$(
        impl<$($name: Snapshot),+> Snapshot for ($($name,)+) {
            fn save(&self, w: &mut Writer<'_>) -> Result<(), SnapshotError> {
                $(self.$index.save(w)?;)+
                Ok(())
            }
            fn load(&mut self, r: &mut Reader<'_>) -> Result<(), SnapshotError> {
                $(self.$index.load(r)?;)+
                Ok(())
            }
            fn load_new(r: &mut Reader<'_>) -> Result<Self, SnapshotError> {
                Ok(($($name::load_new(r)?,)+))
            }
        }
    )*};
}

snapshot_tuples! {
    (A 0)
    (A 0, B 1)
    (A 0, B 1, C 2)
    (A 0, B 1, C 2, D 3)
    (A 0, B 1, C 2, D 3, E 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Snapshot, Default, Debug, PartialEq)]
    #[snapshot(new)]
    struct Plain {
        a: u32,
        b: Vec<u16>,
        #[snapshot(skip)]
        host: u64,
        c: Option<(u8, bool)>,
    }

    #[derive(Snapshot, Debug, PartialEq)]
    enum Shape {
        Empty,
        Pair(u8, u64),
        Named { x: i32, y: Vec<u8> },
    }

    fn round_trip<T: Snapshot>(value: &T, into: &mut T) -> Result<(), SnapshotError> {
        let mut sink = MemorySnapshot::default();
        let mut w = Writer::new(&mut sink);
        value.save(&mut w)?;
        w.finish()?;
        let mut source = MemorySnapshot::new(sink.bytes);
        let mut r = Reader::new(&mut source);
        into.load(&mut r)?;
        r.finish()
    }

    #[test]
    fn a_derived_struct_round_trips_and_keeps_skipped_fields() {
        let value = Plain {
            a: 7,
            b: vec![1, 2, 3],
            host: 99,
            c: Some((4, true)),
        };
        let mut into = Plain {
            host: 5,
            ..Default::default()
        };
        round_trip(&value, &mut into).unwrap();
        assert_eq!(into, Plain { host: 5, ..value });
    }

    #[test]
    fn enums_round_trip_every_variant() {
        for value in [
            Shape::Empty,
            Shape::Pair(3, u64::MAX),
            Shape::Named {
                x: -4,
                y: vec![9, 8],
            },
        ] {
            let mut into = Shape::Empty;
            round_trip(&value, &mut into).unwrap();
            assert_eq!(into, value);
        }
    }

    #[test]
    fn a_damaged_or_short_snapshot_is_an_error_not_a_panic() {
        let value = Plain {
            a: 1,
            b: vec![5; 100],
            host: 0,
            c: None,
        };
        let mut sink = MemorySnapshot::default();
        let mut w = Writer::new(&mut sink);
        value.save(&mut w).unwrap();
        w.finish().unwrap();
        let bytes = sink.bytes;
        for cut in 0..bytes.len() {
            let mut source = MemorySnapshot::new(bytes[..cut].to_vec());
            let mut r = Reader::new(&mut source);
            let mut into = Plain::default();
            let result = into.load(&mut r).and_then(|()| r.finish());
            assert!(result.is_err(), "cut at {cut} was accepted");
        }
        for flip in 0..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[flip] ^= 0x40;
            let mut source = MemorySnapshot::new(damaged);
            let mut r = Reader::new(&mut source);
            let mut into = Plain::default();
            let result = into.load(&mut r).and_then(|()| r.finish());
            assert!(result.is_err(), "a flipped byte at {flip} was accepted");
        }
    }

    #[test]
    fn a_huge_length_is_refused_without_allocating_it() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(1u64 << 40).to_le_bytes());
        let mut source = MemorySnapshot::new(bytes);
        let mut r = Reader::new(&mut source);
        assert!(Vec::<u8>::load_new(&mut r).is_err());
    }

    #[test]
    fn the_table_crc_matches_the_bitwise_one() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut crc = Crc32::new();
        crc.update(&data[..400]);
        crc.update(&data[400..]);
        assert_eq!(crc.finish(), crate::kv::crc32(&data));
    }
}
