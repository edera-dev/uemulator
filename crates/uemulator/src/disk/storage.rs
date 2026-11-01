//! What an image file is stored in (`ImageStorage`), raw images over any storage, and a cache for
//! storage that lives across a network.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;

use super::{BlockBackend, io};

/// The bytes of an image file: a host file, a `Vec`, or ranges fetched over HTTP. Unlike a
/// `BlockBackend` it may grow: a write past `len()` extends it, and the gap reads as zeros.
pub trait ImageStorage: Send {
    fn len(&self) -> u64;
    /// Must fill `buf` completely or fail; reading past `len()` fails.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()>;
    /// Makes every completed write durable before returning: the udisk writer orders its map
    /// updates on it.
    fn flush(&mut self) -> io::Result<()>;
    fn read_only(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: ImageStorage + ?Sized> ImageStorage for Box<T> {
    fn len(&self) -> u64 {
        (**self).len()
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        (**self).read_at(offset, buf)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        (**self).write_at(offset, data)
    }
    fn flush(&mut self) -> io::Result<()> {
        (**self).flush()
    }
    fn read_only(&self) -> bool {
        (**self).read_only()
    }
}

/// Volatile storage: an in-memory overlay, or an image the embedder already holds.
impl ImageStorage for Vec<u8> {
    fn len(&self) -> u64 {
        Vec::len(self) as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let range = super::checked_range(ImageStorage::len(self), offset, buf.len())?;
        buf.copy_from_slice(&self[range]);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        let end = offset
            .checked_add(data.len() as u64)
            .and_then(|end| usize::try_from(end).ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "image too large"))?;
        if end > Vec::len(self) {
            self.resize(end, 0);
        }
        self[offset as usize..end].copy_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A raw disk image (the guest's bytes, nothing else) in any storage. Its capacity is the
/// storage's length when it was opened; writes are refused when the storage is read-only.
pub struct RawImage {
    storage: Box<dyn ImageStorage>,
    len: u64,
}

impl RawImage {
    pub fn new(storage: impl ImageStorage + 'static) -> RawImage {
        let len = storage.len();
        RawImage {
            storage: Box::new(storage),
            len,
        }
    }
}

impl BlockBackend for RawImage {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        super::checked_range(self.len, offset, buf.len())?;
        self.storage.read_at(offset, buf)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.storage.read_only() {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        // A write past the capacity would grow the storage, which `BlockBackend` forbids.
        super::checked_range(self.len, offset, data.len())?;
        self.storage.write_at(offset, data)
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.storage.read_only() {
            true => Ok(()),
            false => self.storage.flush(),
        }
    }
    fn read_only(&self) -> bool {
        self.storage.read_only()
    }
}

/// Reads ranges of a remote file into the caller's buffer, which it must fill completely.
pub type FetchFn = dyn FnMut(u64, &mut [u8]) -> io::Result<()> + Send;

/// Remote storage is read in aligned chunks of this size. Big enough that a guest reading 4 KiB
/// at a time does not pay a round trip each, small enough that a random read does not download
/// much it will not use.
pub const REMOTE_CHUNK_BYTES: u64 = 256 * 1024;
/// Chunks a sequential miss fetches in one request (2 MiB): a guest reading a file front to back
/// then pays one round trip per 2 MiB.
const READ_AHEAD_CHUNKS: u64 = 8;
/// Default cache size, in chunks (64 MiB).
const DEFAULT_CACHED_CHUNKS: usize = 256;

/// Read-only storage over a fetch function (an HTTP range request, a `FileReaderSync`), with an
/// LRU cache of aligned chunks. A read that misses the chunk after one it already has is taken as
/// sequential and fetches several chunks at once.
pub struct RemoteImage {
    len: u64,
    fetch: Box<FetchFn>,
    /// Chunk index to (contents, last use). The last chunk is shorter when `len` is not a
    /// multiple of the chunk size.
    chunks: BTreeMap<u64, (Vec<u8>, u64)>,
    capacity: usize,
    clock: u64,
}

impl RemoteImage {
    /// `len` is the remote file's size, which the caller learned when it found the file.
    pub fn new(
        len: u64,
        fetch: impl FnMut(u64, &mut [u8]) -> io::Result<()> + Send + 'static,
    ) -> RemoteImage {
        RemoteImage {
            len,
            fetch: Box::new(fetch),
            chunks: BTreeMap::new(),
            capacity: DEFAULT_CACHED_CHUNKS,
            clock: 0,
        }
    }

    /// At most this many chunks stay cached (at least 1 + the read-ahead).
    pub fn with_cached_chunks(mut self, chunks: usize) -> RemoteImage {
        self.capacity = chunks.max(1 + READ_AHEAD_CHUNKS as usize);
        self
    }

    /// Fetches chunk `index` and, when the read looks sequential, the missing ones after it, in
    /// one request.
    fn fill(&mut self, index: u64) -> io::Result<()> {
        let last_chunk = (self.len - 1) / REMOTE_CHUNK_BYTES;
        let sequential = index > 0 && self.chunks.contains_key(&(index - 1));
        let wanted = if sequential { READ_AHEAD_CHUNKS } else { 1 };
        let count = (index..=last_chunk)
            .take(wanted as usize)
            .take_while(|chunk| *chunk == index || !self.chunks.contains_key(chunk))
            .count() as u64;
        let start = index * REMOTE_CHUNK_BYTES;
        let end = ((index + count) * REMOTE_CHUNK_BYTES).min(self.len);
        let mut bytes = vec![0; (end - start) as usize];
        (self.fetch)(start, &mut bytes)?;
        while self.chunks.len() + count as usize > self.capacity {
            let oldest = self
                .chunks
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(index, _)| *index)
                .expect("a full cache has chunks");
            self.chunks.remove(&oldest);
        }
        for (position, piece) in bytes.chunks(REMOTE_CHUNK_BYTES as usize).enumerate() {
            self.clock += 1;
            self.chunks
                .insert(index + position as u64, (piece.to_vec(), self.clock));
        }
        Ok(())
    }
}

impl ImageStorage for RemoteImage {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        super::checked_range(self.len, offset, buf.len())?;
        let mut done = 0;
        while done < buf.len() {
            let position = offset + done as u64;
            let index = position / REMOTE_CHUNK_BYTES;
            if !self.chunks.contains_key(&index) {
                self.fill(index)?;
            }
            self.clock += 1;
            let clock = self.clock;
            let (chunk, used) = self.chunks.get_mut(&index).expect("just filled");
            *used = clock;
            let within = (position % REMOTE_CHUNK_BYTES) as usize;
            let count = (chunk.len() - within).min(buf.len() - done);
            buf[done..done + count].copy_from_slice(&chunk[within..within + count]);
            done += count;
        }
        Ok(())
    }
    fn write_at(&mut self, _offset: u64, _data: &[u8]) -> io::Result<()> {
        Err(io::ErrorKind::PermissionDenied.into())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn read_only(&self) -> bool {
        true
    }
}
