//! A gzip-compressed ustar archive, the format Alpine's netboot init unpacks an overlay from.
//! No compression: the deflate stream is all "stored" blocks, which every inflater reads, so the
//! crate needs no deflate encoder (the overlay is a few kilobytes).

use crate::{S_IFDIR, S_IFLNK, S_IFMT};

const PERMISSION_BITS: u32 = 0o7777;

const BLOCK_BYTES: usize = 512;
/// A ustar archive ends with two all-zero blocks.
const END_OF_ARCHIVE_BLOCKS: usize = 2;
const OWNER: &str = "root";
/// Every header field is at a fixed offset (POSIX.1-1988 ustar, `pax(1)`'s "ustar Interchange
/// Format"): (offset, length).
const NAME: (usize, usize) = (0, 100);
const MODE: (usize, usize) = (100, 8);
const UID: (usize, usize) = (108, 8);
const GID: (usize, usize) = (116, 8);
const SIZE: (usize, usize) = (124, 12);
const MTIME: (usize, usize) = (136, 12);
const CHECKSUM: (usize, usize) = (148, 8);
const TYPEFLAG: usize = 156;
const LINKNAME: (usize, usize) = (157, 100);
/// "ustar\0" then version "00".
const MAGIC_AND_VERSION: (usize, &[u8]) = (257, b"ustar\x0000");
const UNAME: (usize, usize) = (265, 32);
const GNAME: (usize, usize) = (297, 32);
const REGULAR_FILE: u8 = b'0';
const SYMBOLIC_LINK: u8 = b'2';
const DIRECTORY: u8 = b'5';
/// Fixed so the same answers give the same bytes. Busybox tar applies it to what it unpacks,
/// which only touches the netboot system's RAM root.
const MTIME_SECONDS: u32 = 0;

/// RFC 1952: ID1 ID2, CM = deflate, FLG = none, MTIME = 0 (none), XFL = 0, OS = Unix.
const GZIP_HEADER: [u8; 10] = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
/// RFC 1951 3.2.4: a stored block holds at most 65535 bytes after its LEN and NLEN.
const STORED_BLOCK_MAX: usize = u16::MAX as usize;
/// Block header bits BFINAL = 1 and BTYPE = 00 (stored); the rest of the byte is padding, since
/// every stored block here starts byte aligned.
const FINAL_STORED_BLOCK: u8 = 1;
const STORED_BLOCK: u8 = 0;

/// CRC-32/ISO-HDLC (reflected polynomial 0xEDB88320), the one gzip's trailer carries.
const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

/// One archive entry. `mode` is the whole `st_mode` (`S_IFREG`, `S_IFDIR` or `S_IFLNK` plus
/// permissions); a symlink's `data` is its target. A directory's `path` ends with `/`.
pub(crate) struct TarEntry {
    pub path: String,
    pub mode: u32,
    pub data: Vec<u8>,
}

/// Returns a `.tar.gz` of `entries`, all owned by root. Paths and link targets must fit their
/// 100-byte fields; the overlay's are fixed and short, so a longer one is a bug here.
pub(crate) fn tar_gz(entries: &[TarEntry]) -> Vec<u8> {
    let mut tar = Vec::new();
    for entry in entries {
        write_entry(&mut tar, entry);
    }
    tar.resize(tar.len() + END_OF_ARCHIVE_BLOCKS * BLOCK_BYTES, 0);
    gzip_stored(&tar)
}

fn write_entry(tar: &mut Vec<u8>, entry: &TarEntry) {
    let (typeflag, link_target, contents): (u8, &[u8], &[u8]) = match entry.mode & S_IFMT {
        S_IFDIR => (DIRECTORY, &[], &[]),
        S_IFLNK => (SYMBOLIC_LINK, &entry.data, &[]),
        _ => (REGULAR_FILE, &[], &entry.data),
    };
    let mut header = [0u8; BLOCK_BYTES];
    put_bytes(&mut header, NAME, entry.path.as_bytes());
    put_octal(&mut header, MODE, entry.mode & PERMISSION_BITS);
    put_octal(&mut header, UID, 0);
    put_octal(&mut header, GID, 0);
    put_octal(&mut header, SIZE, contents.len() as u32);
    put_octal(&mut header, MTIME, MTIME_SECONDS);
    header[TYPEFLAG] = typeflag;
    put_bytes(&mut header, LINKNAME, link_target);
    let (magic_offset, magic) = MAGIC_AND_VERSION;
    header[magic_offset..magic_offset + magic.len()].copy_from_slice(magic);
    put_bytes(&mut header, UNAME, OWNER.as_bytes());
    put_bytes(&mut header, GNAME, OWNER.as_bytes());
    // The checksum is summed with its own field read as spaces, then written as six octal
    // digits, a NUL and a space.
    header[CHECKSUM.0..CHECKSUM.0 + CHECKSUM.1].fill(b' ');
    let checksum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
    let digits = format!("{checksum:06o}\0 ");
    header[CHECKSUM.0..CHECKSUM.0 + CHECKSUM.1].copy_from_slice(digits.as_bytes());

    tar.extend_from_slice(&header);
    tar.extend_from_slice(contents);
    tar.resize(tar.len().next_multiple_of(BLOCK_BYTES), 0);
}

fn put_bytes(header: &mut [u8; BLOCK_BYTES], (offset, length): (usize, usize), value: &[u8]) {
    assert!(value.len() <= length, "tar field too long");
    header[offset..offset + value.len()].copy_from_slice(value);
}

/// Zero-padded octal digits followed by a NUL, filling the field.
fn put_octal(header: &mut [u8; BLOCK_BYTES], (offset, length): (usize, usize), value: u32) {
    let digits = format!("{value:0width$o}\0", width = length - 1);
    put_bytes(header, (offset, length), digits.as_bytes());
}

/// Wraps `data` in a gzip member whose deflate stream is stored blocks only.
fn gzip_stored(data: &[u8]) -> Vec<u8> {
    let mut out = GZIP_HEADER.to_vec();
    let mut chunks = data.chunks(STORED_BLOCK_MAX).peekable();
    if chunks.peek().is_none() {
        write_stored_block(&mut out, &[], true);
    }
    while let Some(chunk) = chunks.next() {
        write_stored_block(&mut out, chunk, chunks.peek().is_none());
    }
    out.extend_from_slice(&crc32(data).to_le_bytes());
    // ISIZE is the input length modulo 2^32.
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

fn write_stored_block(out: &mut Vec<u8>, chunk: &[u8], last: bool) {
    out.push(if last {
        FINAL_STORED_BLOCK
    } else {
        STORED_BLOCK
    });
    let length = chunk.len() as u16;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&(!length).to_le_bytes());
    out.extend_from_slice(chunk);
}

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..u8::BITS {
            let low_bit_set = crc & 1 != 0;
            crc >>= 1;
            if low_bit_set {
                crc ^= CRC32_POLYNOMIAL;
            }
        }
    }
    !crc
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{S_IFDIR, S_IFLNK, S_IFREG};

    const BLOCK: usize = 512;

    fn octal(field: &[u8]) -> u32 {
        let text = core::str::from_utf8(field).unwrap();
        u32::from_str_radix(text.trim_matches(['\0', ' ']), 8).unwrap()
    }

    fn c_string(field: &[u8]) -> String {
        let end = field
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(field.len());
        String::from_utf8(field[..end].to_vec()).unwrap()
    }

    /// Undoes [`tar_gz`], checking every checksum on the way: (path, st_mode, data), a symlink's
    /// data being its target. Only stored deflate blocks and the tar types this crate writes.
    pub(crate) fn read_tar_gz(archive: &[u8]) -> Vec<(String, u32, Vec<u8>)> {
        assert_eq!(
            &archive[..4],
            [0x1f, 0x8b, 8, 0],
            "gzip magic, deflate, no flags"
        );
        let mut offset = 10;
        let mut tar = Vec::new();
        loop {
            let header = archive[offset];
            assert_eq!(header & 0b110, 0, "stored block");
            let length = u16::from_le_bytes([archive[offset + 1], archive[offset + 2]]);
            let complement = u16::from_le_bytes([archive[offset + 3], archive[offset + 4]]);
            assert_eq!(length, !complement);
            offset += 5;
            tar.extend_from_slice(&archive[offset..offset + usize::from(length)]);
            offset += usize::from(length);
            if header & 1 == 1 {
                break;
            }
        }
        let crc = u32::from_le_bytes(archive[offset..offset + 4].try_into().unwrap());
        let size = u32::from_le_bytes(archive[offset + 4..offset + 8].try_into().unwrap());
        assert_eq!(offset + 8, archive.len());
        assert_eq!(crc, crc32(&tar));
        assert_eq!(size as usize, tar.len());

        assert_eq!(tar.len() % BLOCK, 0);
        let mut entries = Vec::new();
        let mut offset = 0;
        while tar[offset..offset + BLOCK].iter().any(|byte| *byte != 0) {
            let header = &tar[offset..offset + BLOCK];
            assert_eq!(&header[257..265], b"ustar\x0000");
            let mut summed = header.to_vec();
            summed[148..156].fill(b' ');
            let sum: u32 = summed.iter().map(|byte| u32::from(*byte)).sum();
            assert_eq!(octal(&header[148..156]), sum, "header checksum");
            assert_eq!(c_string(&header[265..297]), "root");
            assert_eq!(octal(&header[108..116]), 0, "uid");
            let path = c_string(&header[..100]);
            let permissions = octal(&header[100..108]);
            let size = octal(&header[124..136]) as usize;
            let (mode, data) = match header[156] {
                b'0' => (S_IFREG, tar[offset + BLOCK..offset + BLOCK + size].to_vec()),
                b'2' => (S_IFLNK, c_string(&header[157..257]).into_bytes()),
                b'5' => (S_IFDIR, Vec::new()),
                other => panic!("type {other}"),
            };
            entries.push((path, mode | permissions, data));
            offset += BLOCK + size.next_multiple_of(BLOCK);
        }
        assert!(tar[offset..].iter().all(|byte| *byte == 0));
        assert_eq!(
            tar.len() - offset,
            2 * BLOCK,
            "two zero blocks end the archive"
        );
        entries
    }

    fn entry(path: &str, mode: u32, data: &[u8]) -> TarEntry {
        TarEntry {
            path: path.into(),
            mode,
            data: data.to_vec(),
        }
    }

    #[test]
    fn crc32_check_value() {
        // The CRC catalogue's check value for CRC-32/ISO-HDLC, the one gzip uses.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn round_trip() {
        let entries = [
            entry("etc/", S_IFDIR | 0o755, b""),
            entry("etc/empty", S_IFREG | 0o644, b""),
            entry("etc/script", S_IFREG | 0o700, b"#!/bin/sh\necho hi\n"),
            entry("etc/link", S_IFLNK | 0o777, b"/etc/init.d/local"),
            entry("etc/exact", S_IFREG | 0o600, &[7; BLOCK]),
        ];
        let read = read_tar_gz(&tar_gz(&entries));
        let expected: Vec<(String, u32, Vec<u8>)> = entries
            .iter()
            .map(|entry| (entry.path.clone(), entry.mode, entry.data.clone()))
            .collect();
        assert_eq!(read, expected);
    }

    #[test]
    fn large_data_spans_several_stored_blocks() {
        let data: Vec<u8> = (0..200_000u32).map(|index| index as u8).collect();
        let archive = tar_gz(&[entry("big", S_IFREG | 0o644, &data)]);
        assert_eq!(read_tar_gz(&archive)[0].2, data);
    }

    #[test]
    fn empty_archive_is_valid() {
        assert!(read_tar_gz(&tar_gz(&[])).is_empty());
    }
}
