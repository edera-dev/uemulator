//! Checkpoints of a running machine in a directory (`--checkpoint-dir`,
//! `docs/plans/34-incremental-snapshots.md`).
//!
//! `DIR/NNNN.checkpoint` is the library's snapshot: full for the first, then a delta against the
//! checkpoint before it (or the one the run resumed from). Beside it, `NNNN.disk-K` is a copy of
//! each writable disk and `NNNN.nvram` of the UEFI variables, as they were at that moment: RAM
//! restored over disks that moved on would have the guest write through stale metadata. The
//! copies are made while the machine is paused and its disks flushed, before the snapshot is
//! renamed into place, so a checkpoint file never exists without its disks.
//!
//! Resuming `DIR/NNNN.checkpoint` copies its disks back over the machine's, finds its parents by
//! the ids in the headers, and keeps every file: a checkpoint can be gone back to any number of
//! times. Numbers only grow, so a branch taken after going back never overwrites another.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use uemulator::{
    FileSnapshotSink, FileSnapshotSource, SnapshotError, SnapshotId, SnapshotInfo, SnapshotSink,
    SnapshotSource, StopHandle, read_snapshot_info,
};

pub const EXTENSION: &str = "checkpoint";
const NVRAM_SUFFIX: &str = "nvram";

/// Whether `path` names a checkpoint rather than a suspend's snapshot.
pub fn is_checkpoint(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == EXTENSION)
}

/// The files that belong with a checkpoint besides its snapshot: the writable disks, then the
/// nvram, each with the suffix of its copy.
#[derive(Clone, Default)]
pub struct MachineFiles {
    files: Vec<(PathBuf, String)>,
}

impl MachineFiles {
    pub fn new(writable_disks: &[PathBuf], nvram: Option<&Path>) -> Self {
        let mut files: Vec<(PathBuf, String)> = writable_disks
            .iter()
            .enumerate()
            .map(|(index, path)| (path.clone(), format!("disk-{index}")))
            .collect();
        if let Some(nvram) = nvram {
            files.push((nvram.to_path_buf(), NVRAM_SUFFIX.into()));
        }
        MachineFiles { files }
    }

    /// The copies of `checkpoint` (`DIR/NNNN.checkpoint` -> `DIR/NNNN.disk-0`, ...).
    fn copies(&self, checkpoint: &Path) -> impl Iterator<Item = (&Path, PathBuf)> {
        self.files
            .iter()
            .map(move |(path, suffix)| (path.as_path(), checkpoint.with_extension(suffix.as_str())))
    }

    /// Puts the machine's files back as `checkpoint` had them, before anything opens them. A
    /// file without a copy is an error: the checkpoint was taken with other disks.
    ///
    /// Going back replaces the disks as they are now: that is what a revert is. Every copy is
    /// checked first, so a checkpoint with a copy missing changes nothing.
    pub fn restore_from(&self, checkpoint: &Path) -> Result<(), String> {
        for (path, copy) in self.copies(checkpoint) {
            if !copy.is_file() {
                return Err(format!(
                    "{} has no copy of {} ({} is missing): it was taken with other disks",
                    checkpoint.display(),
                    path.display(),
                    copy.display()
                ));
            }
        }
        for (path, copy) in self.copies(checkpoint) {
            copy_file(&copy, path)?;
        }
        Ok(())
    }
}

/// A copy through a temporary name, so `to` is the old file or the whole new one. On APFS and
/// on Linux file systems with reflinks `fs::copy` clones, which costs no space until either side
/// is written.
fn copy_file(from: &Path, to: &Path) -> Result<(), String> {
    let mut temporary = to.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    let _ = fs::remove_file(&temporary);
    fs::copy(from, &temporary)
        .map_err(|e| format!("copy {} to {}: {e}", from.display(), temporary.display()))?;
    fs::rename(&temporary, to).map_err(|e| format!("rename to {}: {e}", to.display()))
}

/// The snapshots to restore for `checkpoint`, oldest (the full one) first, found by the ids in
/// the headers of the checkpoints beside it.
pub fn chain(checkpoint: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = checkpoint.parent().unwrap_or(Path::new("."));
    let mut by_id: Vec<(SnapshotId, PathBuf)> = Vec::new();
    for entry in fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if is_checkpoint(&path)
            && let Ok(info) = info(&path)
        {
            by_id.push((info.id, path));
        }
    }
    let mut chain = vec![checkpoint.to_path_buf()];
    let mut parent = info(checkpoint)?.parent;
    while let Some(id) = parent {
        let found: Vec<&PathBuf> = by_id
            .iter()
            .filter(|(candidate, _)| *candidate == id)
            .map(|(_, path)| path)
            .collect();
        let [path] = found[..] else {
            return Err(format!(
                "{} is a delta of {id}, and {} checkpoints in {} have that id",
                chain.last().unwrap().display(),
                found.len(),
                directory.display()
            ));
        };
        if chain.contains(path) {
            return Err(format!("{} is its own ancestor", path.display()));
        }
        chain.push(path.clone());
        parent = info(path)?.parent;
    }
    chain.reverse();
    Ok(chain)
}

fn info(path: &Path) -> Result<SnapshotInfo, String> {
    let mut source = FileSnapshotSource::open_kept(path).map_err(|e| e.to_string())?;
    read_snapshot_info(&mut source).map_err(|e| format!("{}: {e}", path.display()))
}

/// The chain of `checkpoint`, opened to resume from; every file stays.
pub fn sources(checkpoint: &Path) -> Result<Vec<Box<dyn SnapshotSource>>, String> {
    chain(checkpoint)?
        .into_iter()
        .map(|path| {
            FileSnapshotSource::open_kept(&path)
                .map(|source| Box::new(source) as Box<dyn SnapshotSource>)
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// The next number this process hands out, once a scan of the directory set it. A checkpoint
/// asked for while another is still being written (the timer and Ctrl-] c) has no file yet for
/// the scan to see, so the numbers come from here, not from the directory each time.
static NEXT_NUMBER: Mutex<Option<u32>> = Mutex::new(None);

fn reserve_number(directory: &Path) -> Result<u32, String> {
    let mut next = NEXT_NUMBER.lock().unwrap();
    let number = match *next {
        Some(number) => number,
        None => next_number(directory)?,
    };
    *next = Some(number + 1);
    Ok(number)
}

/// The next unused number in `directory`.
fn next_number(directory: &Path) -> Result<u32, String> {
    let mut highest = 0;
    for entry in fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if let Some(number) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<u32>().ok())
        {
            highest = highest.max(number);
        }
    }
    Ok(highest + 1)
}

/// The library's sink for one checkpoint: the snapshot goes to `NNNN.checkpoint.tmp`, and
/// `finish`, which runs with the machine paused and its disks flushed, copies the disks and only
/// then renames the snapshot into place.
struct CheckpointSink {
    snapshot: FileSnapshotSink,
    path: PathBuf,
    files: MachineFiles,
}

impl SnapshotSink for CheckpointSink {
    fn write(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.snapshot.write(bytes)
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        let copies: Vec<(PathBuf, PathBuf)> = self
            .files
            .copies(&self.path)
            .map(|(from, to)| (from.to_path_buf(), to))
            .collect();
        for (from, to) in &copies {
            if let Err(why) = copy_file(from, to) {
                for (_, made) in &copies {
                    let _ = fs::remove_file(made);
                }
                return Err(SnapshotError(why));
            }
        }
        self.snapshot.finish()
    }
}

/// Takes the next checkpoint into `directory` and reports on stderr when it is written.
pub fn take(handle: &StopHandle, directory: &Path, files: &MachineFiles) {
    let number = match fs::create_dir_all(directory)
        .map_err(|e| e.to_string())
        .and_then(|()| reserve_number(directory))
    {
        Ok(number) => number,
        Err(why) => {
            eprint!("\r\ncheckpoint failed: {why}\r\n");
            return;
        }
    };
    let path = directory.join(format!("{number:04}.{EXTENSION}"));
    let sink = CheckpointSink {
        snapshot: FileSnapshotSink::new(&path),
        path: path.clone(),
        files: files.clone(),
    };
    let started = Instant::now();
    handle.checkpoint(sink, move |result| match result {
        Ok(info) => {
            let bytes = fs::metadata(&path).map_or(0, |metadata| metadata.len());
            let kind = match info.parent {
                Some(parent) => format!("delta of {}", short(parent)),
                None => "full".into(),
            };
            eprint!(
                "\r\ncheckpoint {} ({kind}, id {}): {:.1} MB in {} ms\r\n",
                path.display(),
                short(info.id),
                bytes as f64 / BYTES_PER_MB,
                started.elapsed().as_millis()
            );
        }
        Err(why) => eprint!("\r\ncheckpoint {} failed: {why}\r\n", path.display()),
    });
}

/// Enough of an id to tell checkpoints apart in a message.
const SHORT_ID_DIGITS: usize = 12;
const BYTES_PER_MB: f64 = 1e6;

fn short(id: SnapshotId) -> String {
    id.to_string()[..SHORT_ID_DIGITS].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;

    #[test]
    fn numbers_only_grow_and_only_checkpoints_count_as_such() {
        let directory = TempDir::new("checkpoint-numbers");
        assert_eq!(next_number(directory.path()).unwrap(), 1);
        fs::write(directory.path().join("0007.checkpoint"), b"").unwrap();
        fs::write(directory.path().join("0009.disk-0"), b"").unwrap();
        fs::write(directory.path().join("notes.txt"), b"").unwrap();
        assert_eq!(next_number(directory.path()).unwrap(), 10);
        assert!(is_checkpoint(Path::new("a/0001.checkpoint")));
        assert!(!is_checkpoint(Path::new("a/machine.snapshot")));
    }

    #[test]
    fn a_checkpoint_carries_copies_of_the_disks_and_puts_them_back() {
        let directory = TempDir::new("checkpoint-files");
        let disk = directory.path().join("disk.img");
        let nvram = directory.path().join("nvram.kv");
        fs::write(&disk, b"before").unwrap();
        fs::write(&nvram, b"variables").unwrap();
        let files = MachineFiles::new(std::slice::from_ref(&disk), Some(&nvram));
        let path = directory.path().join("0001.checkpoint");
        let mut sink = CheckpointSink {
            snapshot: FileSnapshotSink::new(&path),
            path: path.clone(),
            files: files.clone(),
        };
        sink.write(b"snapshot").unwrap();
        sink.finish().unwrap();
        assert_eq!(
            fs::read(directory.path().join("0001.disk-0")).unwrap(),
            b"before"
        );
        assert_eq!(
            fs::read(directory.path().join("0001.nvram")).unwrap(),
            b"variables"
        );
        fs::write(&disk, b"after").unwrap();
        files.restore_from(&path).unwrap();
        assert_eq!(fs::read(&disk).unwrap(), b"before");
        let other = MachineFiles::new(&[disk.clone(), directory.path().join("second.img")], None);
        assert!(
            other
                .restore_from(&path)
                .unwrap_err()
                .contains("other disks")
        );
    }

    #[test]
    fn a_failed_disk_copy_leaves_no_checkpoint() {
        let directory = TempDir::new("checkpoint-failed");
        let files = MachineFiles::new(&[directory.path().join("missing.img")], None);
        let path = directory.path().join("0001.checkpoint");
        let mut sink = CheckpointSink {
            snapshot: FileSnapshotSink::new(&path),
            path: path.clone(),
            files,
        };
        sink.write(b"snapshot").unwrap();
        assert!(sink.finish().is_err());
        assert!(!path.exists());
    }
}
