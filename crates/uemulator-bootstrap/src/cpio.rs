//! Appends a newc cpio archive to an existing initramfs; the kernel unpacks concatenated archives
//! in order, so later entries overwrite earlier ones.

use std::collections::BTreeSet;

pub const S_IFDIR: u32 = 0o040_000;
pub const S_IFREG: u32 = 0o100_000;
pub const S_IFLNK: u32 = 0o120_000;
const DIRECTORY_PERMISSIONS: u32 = 0o755;

const MAGIC: &str = "070701";
const TRAILER_NAME: &str = "TRAILER!!!";
/// Headers, names and file data all start on 4-byte boundaries.
const ALIGNMENT: usize = 4;
/// Mask of the file type bits in `st_mode`.
const S_IFMT: u32 = 0o170_000;
const FILE_LINKS: u32 = 1;
const DIRECTORY_LINKS: u32 = 2;

/// One archive entry. `mode` is the whole `st_mode`, file type bits (`S_IFREG`, `S_IFLNK`)
/// included; a symlink's `data` is its target. `path` has no leading `/`. Parent directories are
/// added by [`append_to_initramfs`].
pub struct CpioFile {
    pub path: String,
    pub mode: u32,
    pub data: Vec<u8>,
}

/// Returns `base` followed by an archive holding `files`.
///
/// The kernel reads the next archive at the 4-byte boundary after the previous one and gives up
/// with "invalid magic" otherwise. Distro initramfs files are often not a multiple of 4 long (a
/// gzip stream ends wherever it ends), so `base` is zero padded first; the kernel skips zeros
/// between archives.
pub fn append_to_initramfs(base: &[u8], files: &[CpioFile]) -> Vec<u8> {
    let mut out = base.to_vec();
    pad_to_alignment(&mut out);

    let mut next_inode = 1;
    let mut directories = BTreeSet::new();
    for file in files {
        let mut parent = String::new();
        let components: Vec<&str> = file.path.split('/').collect();
        for component in &components[..components.len() - 1] {
            if !parent.is_empty() {
                parent.push('/');
            }
            parent.push_str(component);
            if directories.insert(parent.clone()) {
                write_entry(
                    &mut out,
                    next_inode,
                    &parent,
                    S_IFDIR | DIRECTORY_PERMISSIONS,
                    &[],
                );
                next_inode += 1;
            }
        }
        write_entry(&mut out, next_inode, &file.path, file.mode, &file.data);
        next_inode += 1;
    }
    write_entry(&mut out, 0, TRAILER_NAME, 0, &[]);
    out
}

fn write_entry(out: &mut Vec<u8>, inode: u32, name: &str, mode: u32, data: &[u8]) {
    let links = if mode & S_IFMT == S_IFDIR {
        DIRECTORY_LINKS
    } else {
        FILE_LINKS
    };
    out.extend_from_slice(MAGIC.as_bytes());
    // ino, mode, uid, gid, nlink, mtime, filesize, devmajor, devminor, rdevmajor, rdevminor,
    // namesize (counting the NUL), check.
    let name_size = name.len() as u32 + 1;
    for field in [
        inode,
        mode,
        0,
        0,
        links,
        0,
        data.len() as u32,
        0,
        0,
        0,
        0,
        name_size,
        0,
    ] {
        out.extend_from_slice(format!("{field:08x}").as_bytes());
    }
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    pad_to_alignment(out);
    out.extend_from_slice(data);
    pad_to_alignment(out);
}

fn pad_to_alignment(out: &mut Vec<u8>) {
    out.resize(out.len().next_multiple_of(ALIGNMENT), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thirteen 8-digit hex fields after the magic.
    const HEADER_BYTES: usize = 110;

    fn field(archive: &[u8], offset: usize, index: usize) -> u32 {
        let start = offset + MAGIC.len() + index * 8;
        u32::from_str_radix(
            core::str::from_utf8(&archive[start..start + 8]).unwrap(),
            16,
        )
        .unwrap()
    }

    /// Reads entries from `offset` until the trailer: (path, mode, data).
    fn parse(archive: &[u8], mut offset: usize) -> Vec<(String, u32, Vec<u8>)> {
        let mut entries = Vec::new();
        loop {
            assert_eq!(&archive[offset..offset + MAGIC.len()], MAGIC.as_bytes());
            let mode = field(archive, offset, 1);
            let size = field(archive, offset, 6) as usize;
            let name_size = field(archive, offset, 11) as usize;
            let name_start = offset + HEADER_BYTES;
            let name = String::from_utf8(archive[name_start..name_start + name_size - 1].to_vec())
                .unwrap();
            let data_start = (name_start + name_size).next_multiple_of(ALIGNMENT);
            let data = archive[data_start..data_start + size].to_vec();
            offset = (data_start + size).next_multiple_of(ALIGNMENT);
            if name == TRAILER_NAME {
                assert_eq!(offset, archive.len());
                return entries;
            }
            entries.push((name, mode, data));
        }
    }

    fn file(path: &str, mode: u32, data: &[u8]) -> CpioFile {
        CpioFile {
            path: path.into(),
            mode,
            data: data.to_vec(),
        }
    }

    #[test]
    fn appended_archive_starts_on_four_byte_boundary() {
        for base_len in 0..8 {
            let base = vec![0xAB; base_len];
            let out = append_to_initramfs(&base, &[file("a", S_IFREG | 0o644, b"x")]);
            let start = base_len.next_multiple_of(ALIGNMENT);
            assert_eq!(&out[..base_len], &base[..]);
            assert!(out[base_len..start].iter().all(|byte| *byte == 0));
            assert_eq!(&out[start..start + MAGIC.len()], MAGIC.as_bytes());
        }
    }

    #[test]
    fn newc_round_trip() {
        let files = [
            file("etc/shadow", S_IFREG | 0o600, b"root:x\n"),
            file("etc/init.d/boot", S_IFREG | 0o755, b"#!/bin/sh\n"),
            file("bin/sh", S_IFLNK | 0o777, b"busybox"),
            file("empty", S_IFREG | 0o644, b""),
        ];
        let out = append_to_initramfs(b"", &files);
        let entries = parse(&out, 0);
        let paths: Vec<&str> = entries.iter().map(|(path, _, _)| path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "etc",
                "etc/shadow",
                "etc/init.d",
                "etc/init.d/boot",
                "bin",
                "bin/sh",
                "empty"
            ]
        );
        assert_eq!(entries[0].1, S_IFDIR | 0o755);
        assert_eq!(
            entries[1],
            ("etc/shadow".into(), S_IFREG | 0o600, b"root:x\n".to_vec())
        );
        assert_eq!(
            entries[5],
            ("bin/sh".into(), S_IFLNK | 0o777, b"busybox".to_vec())
        );
        assert_eq!(field(&out, 0, 0), 1, "inodes count from 1");
        assert_eq!(field(&out, 0, 4), DIRECTORY_LINKS);
    }

    #[test]
    fn entries_are_padded_to_four_bytes() {
        for name_len in 1..=8 {
            for data_len in 0..=5 {
                let name = "n".repeat(name_len);
                let out =
                    append_to_initramfs(&[], &[file(&name, S_IFREG | 0o644, &vec![7; data_len])]);
                assert_eq!(out.len() % ALIGNMENT, 0);
                let entries = parse(&out, 0);
                assert_eq!(entries[0].0, name);
                assert_eq!(entries[0].2.len(), data_len);
            }
        }
    }
}
