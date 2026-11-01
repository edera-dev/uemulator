use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use super::{BlockBackend, Disk, ImageStorage, checked_range};

struct FileDisk {
    file: File,
    /// Captured at open; a file that grows or shrinks afterwards does not change the capacity
    /// the guest was told.
    len: u64,
    read_only: bool,
}

// Positional I/O (no shared seek offset). Unix has it natively; Windows `seek_*` can return
// short counts and moves the file pointer, so loop to match `read_exact_at`/`write_all_at`. WASI's
// positional calls are unstable in `std`, and a WASI module has one thread, so seeking is safe.
#[cfg(unix)]
pub(super) fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(unix)]
pub(super) fn write_all_at(file: &File, data: &[u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, data, offset)
}

#[cfg(target_os = "wasi")]
pub(super) fn read_exact_at(mut file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

#[cfg(target_os = "wasi")]
pub(super) fn write_all_at(mut file: &File, data: &[u8], offset: u64) -> io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(data)
}

#[cfg(windows)]
pub(super) fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let count = file.seek_read(buf, offset)?;
        if count == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[count..];
        offset += count as u64;
    }
    Ok(())
}

#[cfg(windows)]
pub(super) fn write_all_at(file: &File, mut data: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        let count = file.seek_write(data, offset)?;
        if count == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        data = &data[count..];
        offset += count as u64;
    }
    Ok(())
}

impl BlockBackend for FileDisk {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        checked_range(self.len, offset, buf.len())?;
        read_exact_at(&self.file, buf, offset)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        // A write past the captured capacity would grow the file, which the trait forbids.
        checked_range(self.len, offset, data.len())?;
        write_all_at(&self.file, data, offset)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.read_only {
            Ok(())
        } else {
            self.file.sync_data()
        }
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

impl Disk {
    /// Raw image file, opened for writing unless `read_only`. Guest writes go straight to the file;
    /// `flush` is `sync_data`.
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> io::Result<Disk> {
        let file = OpenOptions::new().read(true).write(!read_only).open(path)?;
        let len = file.metadata()?.len();
        Ok(Disk {
            backend: Box::new(FileDisk {
                file,
                len,
                read_only,
            }),
        })
    }
}

/// A host file as image storage (the container of a udisk image, or a raw image through
/// `RawImage`). Writes past the end grow the file; `flush` is `sync_data`.
pub struct ImageFile {
    file: File,
    len: u64,
    read_only: bool,
}

impl ImageFile {
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> io::Result<ImageFile> {
        let file = OpenOptions::new().read(true).write(!read_only).open(path)?;
        let len = file.metadata()?.len();
        Ok(ImageFile {
            file,
            len,
            read_only,
        })
    }

    /// A new, empty file; fails if `path` exists, so an image is never created over another.
    pub fn create(path: impl AsRef<Path>) -> io::Result<ImageFile> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        Ok(ImageFile {
            file,
            len: 0,
            read_only: false,
        })
    }
}

impl ImageStorage for ImageFile {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        checked_range(self.len, offset, buf.len())?;
        read_exact_at(&self.file, buf, offset)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.read_only {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or(io::ErrorKind::InvalidInput)?;
        write_all_at(&self.file, data, offset)?;
        self.len = self.len.max(end);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.read_only {
            true => Ok(()),
            false => self.file.sync_data(),
        }
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}
