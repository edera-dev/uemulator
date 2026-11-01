//! Storage behind the virtio-blk devices: the `BlockBackend` trait, raw images (a host file or
//! memory) and the copy-on-write image format in `udisk` (`docs/spec/uemulator-disk-format.md`).
//!
//! Everything but the host-file types builds `no_std`, so the WebAssembly page has disks too:
//! there `io` stands in for `std::io` with the same call shapes.

#[cfg(freestanding)]
use crate::prelude::*;

#[cfg(not(freestanding))]
mod file;
mod storage;
mod udisk;

#[cfg(not(freestanding))]
pub use file::ImageFile;
pub use storage::{FetchFn, ImageStorage, REMOTE_CHUNK_BYTES, RawImage, RemoteImage};
pub use udisk::{
    DEFAULT_CLUSTER_SIZE, MAX_LOCATOR_BYTES, Udisk, UdiskBacking, UdiskInfo, UdiskOptions,
    UdiskUsage,
};

/// `std::io`'s error type where there is one, else a stand-in with the same constructors, so a
/// `BlockBackend` written for a hosted embedder keeps `std::io::Result` in its signatures.
pub mod io {
    #[cfg(not(freestanding))]
    pub use std::io::{Error, ErrorKind, Result};

    #[cfg(freestanding)]
    pub use stand_in::{Error, ErrorKind, Result};

    #[cfg(freestanding)]
    mod stand_in {
        use crate::prelude::*;
        use core::fmt;

        /// The `std::io::ErrorKind`s the disk code and its embedders use.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum ErrorKind {
            InvalidData,
            InvalidInput,
            NotFound,
            PermissionDenied,
            UnexpectedEof,
            Unsupported,
            WriteZero,
            Other,
        }

        #[derive(Debug)]
        pub struct Error {
            kind: ErrorKind,
            message: String,
        }

        pub type Result<T> = core::result::Result<T, Error>;

        impl Error {
            pub fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
                Error {
                    kind,
                    message: message.into(),
                }
            }

            pub fn other(message: impl Into<String>) -> Error {
                Error::new(ErrorKind::Other, message)
            }

            pub fn kind(&self) -> ErrorKind {
                self.kind
            }
        }

        impl From<ErrorKind> for Error {
            fn from(kind: ErrorKind) -> Error {
                Error::new(kind, String::new())
            }
        }

        impl fmt::Display for Error {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self.message.is_empty() {
                    true => write!(f, "{:?}", self.kind),
                    false => f.write_str(&self.message),
                }
            }
        }
    }
}

/// Storage behind the virtio-blk device. Requests are served synchronously on the vCPU thread
/// when the guest kicks the queue, so a slow backend stalls the guest; hence `Send` is the only
/// thread requirement. Offsets and lengths are in bytes; the guest sees `len() / 512` sectors.
pub trait BlockBackend: Send {
    fn len(&self) -> u64;
    /// Must fill `buf` completely or fail.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
    /// Must not extend the disk: the guest sees a fixed capacity.
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    /// Advertised to the guest (virtio-blk RO feature), and the device refuses writes itself.
    fn read_only(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The identity a copy-on-write image records for its base, so an overlay refuses to open
    /// over a base that is not the one it was made from. `None` for storage that has no identity
    /// (a raw image), which an overlay accepts by size alone.
    fn image_id(&self) -> Option<[u8; 16]> {
        None
    }
}

impl<T: BlockBackend + ?Sized> BlockBackend for Box<T> {
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
    fn image_id(&self) -> Option<[u8; 16]> {
        (**self).image_id()
    }
}

pub struct Disk {
    pub(crate) backend: Box<dyn BlockBackend>,
}

struct MemDisk(Vec<u8>);

/// `offset..offset + length` as an index range, or an error if it does not fit in `capacity`.
/// Overflow-proof: the offset comes straight from the guest's request.
fn checked_range(capacity: u64, offset: u64, length: usize) -> io::Result<core::ops::Range<usize>> {
    match offset.checked_add(length as u64) {
        // Both ends are at most `capacity`, which for a `MemDisk` is its `Vec` length.
        Some(end) if end <= capacity => Ok(offset as usize..end as usize),
        _ => Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
    }
}

impl BlockBackend for MemDisk {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let range = checked_range(self.len(), offset, buf.len())?;
        buf.copy_from_slice(&self.0[range]);
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        let range = checked_range(self.len(), offset, data.len())?;
        self.0[range].copy_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Disk {
    /// Volatile in-memory disk; contents are lost when the machine is dropped.
    pub fn memory(data: Vec<u8>) -> Disk {
        Disk {
            backend: Box::new(MemDisk(data)),
        }
    }

    pub fn from_backend(backend: impl BlockBackend + 'static) -> Disk {
        Disk {
            backend: Box::new(backend),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(freestanding))]
    use super::file::read_exact_at;
    use super::*;
    use std::io::{Read, Seek, Write};
    use std::sync::atomic::{AtomicU32, Ordering};

    const SECTOR: usize = 512;

    /// A uniquely named file in the OS temp directory, removed on drop.
    struct TempImage(std::path::PathBuf);

    impl TempImage {
        fn with_contents(contents: &[u8]) -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "uemulator-disk-test-{}-{}.img",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, contents).unwrap();
            TempImage(path)
        }

        fn contents(&self) -> Vec<u8> {
            std::fs::read(&self.0).unwrap()
        }
    }

    impl Drop for TempImage {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 251) as u8).collect()
    }

    fn kind(result: io::Result<()>) -> io::ErrorKind {
        result.expect_err("expected an I/O error").kind()
    }

    mod memory {
        use super::*;

        fn disk(contents: Vec<u8>) -> Box<dyn BlockBackend> {
            Disk::memory(contents).backend
        }

        #[test]
        fn reports_its_length_and_is_writable() {
            let backend = disk(vec![0; 4 * SECTOR]);
            assert_eq!(backend.len(), 4 * SECTOR as u64);
            assert!(!backend.is_empty());
            assert!(!backend.read_only());
        }

        #[test]
        fn empty_disk_is_empty() {
            let backend = disk(Vec::new());
            assert_eq!(backend.len(), 0);
            assert!(backend.is_empty());
        }

        #[test]
        fn reads_what_it_was_created_with() {
            let mut backend = disk(pattern(2 * SECTOR));
            let mut buf = [0u8; 16];
            backend.read_at(5, &mut buf).unwrap();
            assert_eq!(buf, pattern(21)[5..]);
        }

        #[test]
        fn write_then_read_round_trips_and_leaves_neighbours_alone() {
            let mut backend = disk(vec![0xee; 2 * SECTOR]);
            backend.write_at(SECTOR as u64 - 2, &[1, 2, 3, 4]).unwrap();
            let mut buf = [0u8; 8];
            backend.read_at(SECTOR as u64 - 4, &mut buf).unwrap();
            assert_eq!(buf, [0xee, 0xee, 1, 2, 3, 4, 0xee, 0xee]);
        }

        #[test]
        fn access_ending_exactly_at_the_end_is_allowed() {
            let mut backend = disk(vec![7; SECTOR]);
            let mut buf = [0u8; 4];
            backend.read_at(SECTOR as u64 - 4, &mut buf).unwrap();
            assert_eq!(buf, [7; 4]);
            backend.write_at(SECTOR as u64 - 4, &[9; 4]).unwrap();
            backend.read_at(SECTOR as u64 - 4, &mut buf).unwrap();
            assert_eq!(buf, [9; 4]);
        }

        #[test]
        fn access_one_byte_past_the_end_fails_and_changes_nothing() {
            let mut backend = disk(vec![7; SECTOR]);
            let mut buf = [0u8; 4];
            assert_eq!(
                kind(backend.read_at(SECTOR as u64 - 3, &mut buf)),
                io::ErrorKind::UnexpectedEof
            );
            assert_eq!(
                kind(backend.write_at(SECTOR as u64 - 3, &[1; 4])),
                io::ErrorKind::UnexpectedEof
            );
            let mut tail = [0u8; 3];
            backend.read_at(SECTOR as u64 - 3, &mut tail).unwrap();
            assert_eq!(
                tail, [7; 3],
                "a rejected write must not be partially applied"
            );
            assert_eq!(backend.len(), SECTOR as u64, "the disk never grows");
        }

        #[test]
        fn offsets_beyond_the_end_fail() {
            let mut backend = disk(vec![0; SECTOR]);
            let mut buf = [0u8; 1];
            for offset in [
                SECTOR as u64,
                SECTOR as u64 + 1,
                u64::MAX,
                u64::MAX - 1,
                1 << 63,
            ] {
                assert!(backend.read_at(offset, &mut buf).is_err(), "{offset:#x}");
                assert!(backend.write_at(offset, &[0]).is_err(), "{offset:#x}");
            }
        }

        #[test]
        fn offset_plus_length_wrapping_u64_is_an_error_not_a_wild_range() {
            let mut backend = disk(vec![0; SECTOR]);
            let mut buf = [0u8; 16];
            assert!(backend.read_at(u64::MAX - 7, &mut buf).is_err());
            assert!(backend.write_at(u64::MAX - 7, &buf).is_err());
        }

        #[test]
        fn zero_length_access_is_fine_up_to_the_end_and_not_beyond() {
            let mut backend = disk(vec![0; SECTOR]);
            backend.read_at(0, &mut []).unwrap();
            backend.read_at(SECTOR as u64, &mut []).unwrap();
            backend.write_at(SECTOR as u64, &[]).unwrap();
            assert!(backend.read_at(SECTOR as u64 + 1, &mut []).is_err());
        }

        #[test]
        fn flush_succeeds() {
            disk(vec![0; SECTOR]).flush().unwrap();
        }

        #[test]
        fn from_backend_wraps_any_backend() {
            struct Fixed;
            impl BlockBackend for Fixed {
                fn len(&self) -> u64 {
                    1024
                }
                fn read_at(&mut self, _: u64, buf: &mut [u8]) -> io::Result<()> {
                    buf.fill(0xab);
                    Ok(())
                }
                fn write_at(&mut self, _: u64, _: &[u8]) -> io::Result<()> {
                    Err(io::ErrorKind::PermissionDenied.into())
                }
                fn flush(&mut self) -> io::Result<()> {
                    Ok(())
                }
                fn read_only(&self) -> bool {
                    true
                }
            }
            let mut backend = Disk::from_backend(Fixed).backend;
            assert_eq!(backend.len(), 1024);
            assert!(backend.read_only());
            let mut buf = [0u8; 2];
            backend.read_at(0, &mut buf).unwrap();
            assert_eq!(buf, [0xab; 2]);
            assert_eq!(
                kind(backend.write_at(0, &[1])),
                io::ErrorKind::PermissionDenied
            );
        }

        #[test]
        fn the_default_read_only_is_false_and_is_empty_follows_len() {
            struct Minimal(u64);
            impl BlockBackend for Minimal {
                fn len(&self) -> u64 {
                    self.0
                }
                fn read_at(&mut self, _: u64, _: &mut [u8]) -> io::Result<()> {
                    Ok(())
                }
                fn write_at(&mut self, _: u64, _: &[u8]) -> io::Result<()> {
                    Ok(())
                }
                fn flush(&mut self) -> io::Result<()> {
                    Ok(())
                }
            }
            assert!(!Minimal(1).read_only());
            assert!(!Minimal(1).is_empty());
            assert!(Minimal(0).is_empty());
        }
    }

    mod file {
        use super::*;

        fn open(image: &TempImage, read_only: bool) -> Box<dyn BlockBackend> {
            Disk::open(&image.0, read_only).unwrap().backend
        }

        #[test]
        fn opening_a_missing_file_is_an_error() {
            let missing = std::env::temp_dir().join("uemulator-disk-test-does-not-exist.img");
            assert_eq!(
                Disk::open(&missing, false).err().map(|e| e.kind()),
                Some(io::ErrorKind::NotFound)
            );
        }

        #[test]
        fn length_is_the_file_size() {
            let image = TempImage::with_contents(&pattern(3 * SECTOR));
            assert_eq!(open(&image, false).len(), 3 * SECTOR as u64);
            assert_eq!(open(&image, true).len(), 3 * SECTOR as u64);
        }

        #[test]
        fn reads_the_file_contents_at_any_offset() {
            let contents = pattern(2 * SECTOR);
            let image = TempImage::with_contents(&contents);
            let mut backend = open(&image, true);
            let mut buf = vec![0u8; SECTOR];
            backend.read_at(SECTOR as u64 / 2, &mut buf).unwrap();
            assert_eq!(buf, contents[SECTOR / 2..SECTOR / 2 + SECTOR]);
        }

        #[test]
        fn writes_go_straight_to_the_file() {
            let image = TempImage::with_contents(&vec![0u8; 2 * SECTOR]);
            let mut backend = open(&image, false);
            backend.write_at(100, b"hello").unwrap();
            backend.flush().unwrap();
            let on_disk = image.contents();
            assert_eq!(&on_disk[100..105], b"hello");
            assert_eq!(on_disk.len(), 2 * SECTOR);
            assert!(on_disk[..100].iter().all(|&byte| byte == 0));
        }

        #[test]
        fn read_only_refuses_writes_and_leaves_the_file_untouched() {
            let contents = pattern(SECTOR);
            let image = TempImage::with_contents(&contents);
            let mut backend = open(&image, true);
            assert!(backend.read_only());
            assert_eq!(
                kind(backend.write_at(0, &[0xff])),
                io::ErrorKind::PermissionDenied
            );
            assert_eq!(image.contents(), contents);
        }

        #[test]
        fn read_only_flush_is_a_no_op_success() {
            let image = TempImage::with_contents(&[0; SECTOR]);
            open(&image, true).flush().unwrap();
        }

        #[test]
        fn read_write_open_reports_writable() {
            let image = TempImage::with_contents(&[0; SECTOR]);
            assert!(!open(&image, false).read_only());
        }

        #[test]
        fn a_short_read_at_the_end_of_the_file_is_an_error() {
            let image = TempImage::with_contents(&pattern(SECTOR));
            let mut backend = open(&image, true);
            let mut buf = [0u8; 8];
            assert_eq!(
                kind(backend.read_at(SECTOR as u64 - 4, &mut buf)),
                io::ErrorKind::UnexpectedEof
            );
        }

        #[test]
        fn writes_past_the_capacity_fail_and_do_not_grow_the_file() {
            let image = TempImage::with_contents(&vec![0u8; SECTOR]);
            let mut backend = open(&image, false);
            assert!(backend.write_at(SECTOR as u64, &[1]).is_err());
            assert!(backend.write_at(SECTOR as u64 - 1, &[1, 2]).is_err());
            assert!(backend.write_at(u64::MAX - 1, &[1, 2, 3]).is_err());
            assert_eq!(image.contents().len(), SECTOR);
            assert_eq!(backend.len(), SECTOR as u64);
        }

        #[test]
        fn capacity_is_fixed_at_open_even_if_the_file_grows_later() {
            let image = TempImage::with_contents(&vec![0u8; SECTOR]);
            let mut backend = open(&image, false);
            std::fs::OpenOptions::new()
                .append(true)
                .open(&image.0)
                .unwrap()
                .write_all(&[5u8; SECTOR])
                .unwrap();
            assert_eq!(backend.len(), SECTOR as u64);
            let mut buf = [0u8; 1];
            assert!(backend.read_at(SECTOR as u64, &mut buf).is_err());
            assert!(backend.write_at(SECTOR as u64, &[1]).is_err());
        }

        #[test]
        fn access_ending_exactly_at_the_end_is_allowed() {
            let image = TempImage::with_contents(&vec![3u8; SECTOR]);
            let mut backend = open(&image, false);
            backend.write_at(SECTOR as u64 - 2, &[8, 9]).unwrap();
            let mut buf = [0u8; 2];
            backend.read_at(SECTOR as u64 - 2, &mut buf).unwrap();
            assert_eq!(buf, [8, 9]);
        }

        #[test]
        fn two_backends_on_one_file_see_each_others_writes() {
            let image = TempImage::with_contents(&vec![0u8; SECTOR]);
            let mut writer = open(&image, false);
            let mut reader = open(&image, true);
            writer.write_at(7, &[42]).unwrap();
            let mut buf = [0u8; 1];
            reader.read_at(7, &mut buf).unwrap();
            assert_eq!(buf, [42]);
        }

        #[test]
        fn positional_io_does_not_depend_on_a_shared_seek_offset() {
            let contents = pattern(SECTOR);
            let image = TempImage::with_contents(&contents);
            let mut backend = open(&image, true);
            let mut first = [0u8; 4];
            let mut second = [0u8; 4];
            backend.read_at(100, &mut first).unwrap();
            backend.read_at(0, &mut second).unwrap();
            backend.read_at(100, &mut first).unwrap();
            assert_eq!(first, contents[100..104]);
            assert_eq!(second, contents[..4]);
            // The helper itself does not move the file cursor.
            let mut file = std::fs::File::open(&image.0).unwrap();
            read_exact_at(&file, &mut first, 200).unwrap();
            let mut at_start = [0u8; 1];
            file.read_exact(&mut at_start).unwrap();
            assert_eq!(at_start[0], contents[0]);
            assert_eq!(file.stream_position().unwrap(), 1);
        }

        #[test]
        fn an_empty_file_is_an_empty_disk() {
            let image = TempImage::with_contents(&[]);
            let backend = open(&image, true);
            assert!(backend.is_empty());
        }
    }
}
