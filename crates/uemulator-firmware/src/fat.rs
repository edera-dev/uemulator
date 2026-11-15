//! A read-only FAT12/16/32 reader with long file names, over any byte-addressable `Storage`.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FatError {
    /// The storage failed a read.
    Device,
    /// Not a FAT volume, or one whose structures contradict each other.
    Corrupt,
    NotFound,
}

pub trait Storage {
    fn read(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), FatError>;
}

const BOOT_SIGNATURE: [u8; 2] = [0x55, 0xaa];
const SECTOR_BYTES_MAX: u32 = 4096;
const DIRECTORY_ENTRY_BYTES: usize = 32;
const FAT12_CLUSTER_LIMIT: u32 = 4085;
const FAT16_CLUSTER_LIMIT: u32 = 65525;
const FIRST_DATA_CLUSTER: u32 = 2;
/// Bytes of the active FAT cached at a time. A power of two that FAT12's 1.5-byte entries may
/// straddle, so lookups go byte by byte.
const FAT_CACHE_BYTES: u64 = 4096;
const FAT32_ENTRY_MASK: u32 = 0x0fff_ffff;
const FAT32_EXT_FLAGS_OFFSET: usize = 40;
const FAT32_MIRRORING_DISABLED: u16 = 1 << 7;
const FAT32_ACTIVE_FAT_MASK: u16 = 0x0f;

const ATTRIBUTE_VOLUME_LABEL: u8 = 0x08;
const ATTRIBUTE_DIRECTORY: u8 = 0x10;
const ATTRIBUTE_LONG_NAME: u8 = 0x0f;
const DELETED: u8 = 0xe5;
/// A first byte of 0xE5 in a real name is stored as 0x05.
const KANJI_ESCAPE: u8 = 0x05;
const END_OF_DIRECTORY: u8 = 0x00;
const LONG_NAME_LAST: u8 = 0x40;
const LONG_NAME_SEQUENCE_MASK: u8 = 0x1f;
const LONG_NAME_CHARS_PER_ENTRY: usize = 13;
/// Byte offsets of the 13 UTF-16 units inside a long-name entry.
const LONG_NAME_OFFSETS: [usize; LONG_NAME_CHARS_PER_ENTRY] =
    [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
const NAME_LOWERCASE: u8 = 0x08;
const EXTENSION_LOWERCASE: u8 = 0x10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fat12,
    Fat16,
    Fat32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub attributes: u8,
    /// 0 for an empty file; for a directory entry ".." pointing at the root also 0.
    pub first_cluster: u32,
    pub size: u32,
    pub created: (u16, u16),
    pub accessed: u16,
    pub modified: (u16, u16),
}

impl DirEntry {
    pub fn is_directory(&self) -> bool {
        self.attributes & ATTRIBUTE_DIRECTORY != 0
    }
}

/// A position in a file's cluster chain, so sequential reads do not walk it from the start.
#[derive(Debug, Clone, Copy)]
pub struct Cursor {
    pub first_cluster: u32,
    pub size: u32,
    index: u32,
    cluster: u32,
}

impl Cursor {
    pub fn new(entry: &DirEntry) -> Self {
        Self {
            first_cluster: entry.first_cluster,
            size: entry.size,
            index: 0,
            cluster: entry.first_cluster,
        }
    }
}

pub struct Volume<S: Storage> {
    storage: S,
    kind: Kind,
    cluster_bytes: u64,
    fat_start: u64,
    fat_bytes: u64,
    root_start: u64,
    root_bytes: u64,
    root_cluster: u32,
    data_start: u64,
    cluster_count: u32,
    total_bytes: u64,
    pub label: String,
    fat_cache: Vec<u8>,
    fat_cache_at: Option<u64>,
}

fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(data[at..at + 2].try_into().unwrap())
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

impl<S: Storage> Volume<S> {
    pub fn open(mut storage: S) -> Result<Self, FatError> {
        let mut boot = [0u8; 512];
        storage.read(0, &mut boot)?;
        if boot[510..512] != BOOT_SIGNATURE {
            return Err(FatError::Corrupt);
        }
        let sector_bytes = u16_at(&boot, 11) as u32;
        let sectors_per_cluster = boot[13] as u32;
        let reserved = u16_at(&boot, 14) as u64;
        let fat_count = boot[16] as u64;
        let root_entries = u16_at(&boot, 17) as u64;
        let total_sectors = match u16_at(&boot, 19) {
            0 => u32_at(&boot, 32) as u64,
            count => count as u64,
        };
        let fat_sectors = match u16_at(&boot, 22) {
            0 => u32_at(&boot, 36) as u64,
            count => count as u64,
        };
        if !(512..=SECTOR_BYTES_MAX).contains(&sector_bytes)
            || !sector_bytes.is_power_of_two()
            || !sectors_per_cluster.is_power_of_two()
            || reserved == 0
            || fat_count == 0
            || fat_sectors == 0
        {
            return Err(FatError::Corrupt);
        }
        let sector_bytes = sector_bytes as u64;
        let root_sectors = (root_entries * DIRECTORY_ENTRY_BYTES as u64).div_ceil(sector_bytes);
        let data_sector = reserved + fat_count * fat_sectors + root_sectors;
        let data_sectors = total_sectors
            .checked_sub(data_sector)
            .ok_or(FatError::Corrupt)?;
        let cluster_count = (data_sectors / sectors_per_cluster as u64) as u32;
        let kind = if cluster_count < FAT12_CLUSTER_LIMIT {
            Kind::Fat12
        } else if cluster_count < FAT16_CLUSTER_LIMIT {
            Kind::Fat16
        } else {
            Kind::Fat32
        };
        // BPB_ExtFlags only selects a FAT when FAT32 mirroring is disabled; the
        // inactive copies can be stale, so using copy zero can break valid chains.
        let ext_flags = u16_at(&boot, FAT32_EXT_FLAGS_OFFSET);
        let active_fat = if kind == Kind::Fat32 && ext_flags & FAT32_MIRRORING_DISABLED != 0 {
            u64::from(ext_flags & FAT32_ACTIVE_FAT_MASK)
        } else {
            0
        };
        if active_fat >= fat_count {
            return Err(FatError::Corrupt);
        }
        let label_at = if kind == Kind::Fat32 { 71 } else { 43 };
        let label = String::from_utf8_lossy(&boot[label_at..label_at + 11])
            .trim_end()
            .into();
        Ok(Self {
            storage,
            kind,
            cluster_bytes: sector_bytes * sectors_per_cluster as u64,
            fat_start: (reserved + active_fat * fat_sectors) * sector_bytes,
            fat_bytes: fat_sectors * sector_bytes,
            root_start: (reserved + fat_count * fat_sectors) * sector_bytes,
            root_bytes: root_sectors * sector_bytes,
            root_cluster: if kind == Kind::Fat32 {
                u32_at(&boot, 44)
            } else {
                0
            },
            data_start: data_sector * sector_bytes,
            cluster_count,
            total_bytes: total_sectors * sector_bytes,
            label,
            fat_cache: alloc::vec![0; FAT_CACHE_BYTES as usize],
            fat_cache_at: None,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn cluster_bytes(&self) -> u64 {
        self.cluster_bytes
    }

    /// The directory to pass to `list` and `lookup` for the root.
    pub fn root(&self) -> u32 {
        0
    }

    fn fat_byte(&mut self, offset: u64) -> Result<u8, FatError> {
        if offset >= self.fat_bytes {
            return Err(FatError::Corrupt);
        }
        let chunk = offset & !(FAT_CACHE_BYTES - 1);
        if self.fat_cache_at != Some(chunk) {
            // A FAT (or its final cache page) can be smaller than the cache. Do not
            // read the next FAT, directory, or bytes beyond a small volume.
            let count = (self.fat_bytes - chunk).min(FAT_CACHE_BYTES) as usize;
            self.fat_cache_at = None;
            let mut cache = core::mem::take(&mut self.fat_cache);
            let result = self
                .storage
                .read(self.fat_start + chunk, &mut cache[..count]);
            self.fat_cache = cache;
            result?;
            self.fat_cache_at = Some(chunk);
        }
        Ok(self.fat_cache[(offset - chunk) as usize])
    }

    /// The cluster after `cluster`, `None` at the end of the chain.
    fn next_cluster(&mut self, cluster: u32) -> Result<Option<u32>, FatError> {
        if cluster < FIRST_DATA_CLUSTER || cluster >= self.cluster_count + FIRST_DATA_CLUSTER {
            return Err(FatError::Corrupt);
        }
        let (next, end_of_chain) = match self.kind {
            Kind::Fat12 => {
                let at = cluster as u64 * 3 / 2;
                let pair = self.fat_byte(at)? as u32 | (self.fat_byte(at + 1)? as u32) << 8;
                (
                    if cluster & 1 == 0 {
                        pair & 0xfff
                    } else {
                        pair >> 4
                    },
                    0xff8,
                )
            }
            Kind::Fat16 => {
                let at = cluster as u64 * 2;
                (
                    self.fat_byte(at)? as u32 | (self.fat_byte(at + 1)? as u32) << 8,
                    0xfff8,
                )
            }
            Kind::Fat32 => {
                let at = cluster as u64 * 4;
                let mut value = 0u32;
                for byte in 0..4 {
                    value |= (self.fat_byte(at + byte)? as u32) << (8 * byte);
                }
                (value & FAT32_ENTRY_MASK, 0x0fff_fff8)
            }
        };
        if next >= end_of_chain {
            return Ok(None);
        }
        self.cluster_offset(next)?;
        Ok(Some(next))
    }

    fn cluster_offset(&self, cluster: u32) -> Result<u64, FatError> {
        let index = cluster
            .checked_sub(FIRST_DATA_CLUSTER)
            .filter(|index| *index < self.cluster_count)
            .ok_or(FatError::Corrupt)?;
        Ok(self.data_start + u64::from(index) * self.cluster_bytes)
    }

    fn read_directory_bytes(&mut self, directory: u32) -> Result<Vec<u8>, FatError> {
        let directory = if directory == 0 {
            self.root_cluster
        } else {
            directory
        };
        if directory == 0 && self.kind != Kind::Fat32 {
            let mut bytes = alloc::vec![0u8; self.root_bytes as usize];
            self.storage.read(self.root_start, &mut bytes)?;
            return Ok(bytes);
        }
        let mut bytes = Vec::new();
        let mut cluster = Some(directory);
        let mut guard = 0;
        while let Some(current) = cluster {
            let offset = self.cluster_offset(current)?;
            let start = bytes.len();
            bytes.resize(start + self.cluster_bytes as usize, 0);
            self.storage.read(offset, &mut bytes[start..])?;
            guard += 1;
            if guard > self.cluster_count {
                return Err(FatError::Corrupt);
            }
            cluster = self.next_cluster(current)?;
        }
        Ok(bytes)
    }

    /// Every entry of the directory (`root()` or an entry's `first_cluster`), "." and ".."
    /// included where the directory has them.
    pub fn list(&mut self, directory: u32) -> Result<Vec<DirEntry>, FatError> {
        let bytes = self.read_directory_bytes(directory)?;
        let mut entries = Vec::new();
        let mut long_units: Vec<u16> = Vec::new();
        let mut long_checksum = 0u8;
        let mut expected_sequence = 0u8;
        for raw in bytes.as_chunks::<DIRECTORY_ENTRY_BYTES>().0 {
            match raw[0] {
                END_OF_DIRECTORY => break,
                DELETED => {
                    long_units.clear();
                    expected_sequence = 0;
                    continue;
                }
                _ => {}
            }
            let attributes = raw[11];
            if attributes == ATTRIBUTE_LONG_NAME {
                let sequence = raw[0] & LONG_NAME_SEQUENCE_MASK;
                if raw[0] & LONG_NAME_LAST != 0 {
                    long_units = alloc::vec![0u16; sequence as usize * LONG_NAME_CHARS_PER_ENTRY];
                    long_checksum = raw[13];
                    expected_sequence = sequence;
                }
                if sequence == 0 || sequence != expected_sequence || raw[13] != long_checksum {
                    long_units.clear();
                    expected_sequence = 0;
                    continue;
                }
                let base = (sequence as usize - 1) * LONG_NAME_CHARS_PER_ENTRY;
                for (slot, offset) in LONG_NAME_OFFSETS.iter().enumerate() {
                    long_units[base + slot] = u16_at(raw, *offset);
                }
                expected_sequence -= 1;
                continue;
            }
            if attributes & ATTRIBUTE_VOLUME_LABEL != 0 {
                long_units.clear();
                expected_sequence = 0;
                continue;
            }
            let long_name_valid = expected_sequence == 0
                && !long_units.is_empty()
                && short_name_checksum(raw) == long_checksum;
            let name = if long_name_valid {
                let end = long_units
                    .iter()
                    .position(|&unit| unit == 0)
                    .unwrap_or(long_units.len());
                crate::text::to_string(&long_units[..end])
            } else {
                short_name(raw)
            };
            long_units.clear();
            expected_sequence = 0;
            let high_cluster = if self.kind == Kind::Fat32 {
                u16_at(raw, 20) as u32
            } else {
                0
            };
            entries.push(DirEntry {
                name,
                attributes,
                first_cluster: high_cluster << 16 | u16_at(raw, 26) as u32,
                size: u32_at(raw, 28),
                created: (u16_at(raw, 16), u16_at(raw, 14)),
                accessed: u16_at(raw, 18),
                modified: (u16_at(raw, 24), u16_at(raw, 22)),
            });
        }
        Ok(entries)
    }

    pub fn lookup(&mut self, directory: u32, name: &str) -> Result<DirEntry, FatError> {
        self.list(directory)?
            .into_iter()
            .find(|entry| crate::text::eq_ignore_case(&entry.name, name))
            .ok_or(FatError::NotFound)
    }

    /// Reads from `offset` into `buffer`; returns how many bytes were available (0 at the end).
    pub fn read(
        &mut self,
        file: &mut Cursor,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<usize, FatError> {
        if offset >= file.size as u64 || buffer.is_empty() {
            return Ok(0);
        }
        let total = (buffer.len() as u64).min(file.size as u64 - offset) as usize;
        let mut done = 0usize;
        while done < total {
            let position = offset + done as u64;
            let index = (position / self.cluster_bytes) as u32;
            let within = position % self.cluster_bytes;
            // The cursor only moves forward; a read before it restarts from the first cluster.
            if index < file.index {
                file.index = 0;
                file.cluster = file.first_cluster;
            }
            while file.index < index {
                file.cluster = self.next_cluster(file.cluster)?.ok_or(FatError::Corrupt)?;
                file.index += 1;
            }
            let at = self.cluster_offset(file.cluster)? + within;
            // Extend over the clusters that follow in order, so a contiguous file is one request.
            let wanted = (total - done) as u64;
            let mut run = self.cluster_bytes - within;
            let mut last = file.cluster;
            let mut last_index = file.index;
            while run < wanted {
                match self.next_cluster(last)? {
                    Some(next) if next == last + 1 => {
                        last = next;
                        last_index += 1;
                        run += self.cluster_bytes;
                    }
                    _ => break,
                }
            }
            let length = run.min(wanted) as usize;
            self.storage.read(at, &mut buffer[done..done + length])?;
            done += length;
            file.cluster = last;
            file.index = last_index;
        }
        Ok(total)
    }
}

fn short_name_checksum(raw: &[u8]) -> u8 {
    raw[..11]
        .iter()
        .fold(0u8, |sum, &byte| sum.rotate_right(1).wrapping_add(byte))
}

fn short_name(raw: &[u8]) -> String {
    let flags = raw[12];
    let mut name = [0u8; 8];
    name.copy_from_slice(&raw[..8]);
    if name[0] == KANJI_ESCAPE {
        name[0] = DELETED;
    }
    let part = |bytes: &[u8], lowercase: bool| -> String {
        let end = bytes
            .iter()
            .rposition(|&byte| byte != b' ')
            .map_or(0, |last| last + 1);
        bytes[..end]
            .iter()
            .map(|&byte| {
                let character = byte as char;
                if lowercase {
                    character.to_ascii_lowercase()
                } else {
                    character
                }
            })
            .collect()
    };
    let base = part(&name, flags & NAME_LOWERCASE != 0);
    let extension = part(&raw[8..11], flags & EXTENSION_LOWERCASE != 0);
    if extension.is_empty() {
        base
    } else {
        alloc::format!("{base}.{extension}")
    }
}

#[cfg(test)]
pub mod testing {
    //! Builds FAT images for this module's tests and the file system tests that sit on it.

    use super::*;

    pub struct Image(pub Vec<u8>);

    impl Storage for Image {
        fn read(&mut self, offset: u64, buffer: &mut [u8]) -> Result<(), FatError> {
            let range = offset as usize..offset as usize + buffer.len();
            buffer.copy_from_slice(self.0.get(range).ok_or(FatError::Device)?);
            Ok(())
        }
    }

    pub struct Builder {
        pub image: Vec<u8>,
        kind: Kind,
        fat_start: usize,
        root_start: usize,
        data_start: usize,
        next_cluster: u32,
    }

    pub enum Size {
        Fat12,
        Fat16,
        Fat32,
    }

    const SECTOR: usize = 512;
    const ROOT_ENTRIES: usize = 512;

    impl Builder {
        pub fn new(size: Size) -> Self {
            let (kind, clusters, reserved, fat_sectors) = match size {
                Size::Fat12 => (Kind::Fat12, 2000usize, 1usize, 6usize),
                Size::Fat16 => (Kind::Fat16, 5000, 1, 20),
                Size::Fat32 => (Kind::Fat32, 66000, 32, 520),
            };
            let root_sectors = if kind == Kind::Fat32 {
                0
            } else {
                ROOT_ENTRIES * 32 / SECTOR
            };
            let data_sector = reserved + 2 * fat_sectors + root_sectors;
            let total = data_sector + clusters;
            let mut image = alloc::vec![0u8; total * SECTOR];
            image[11..13].copy_from_slice(&(SECTOR as u16).to_le_bytes());
            image[13] = 1;
            image[14..16].copy_from_slice(&(reserved as u16).to_le_bytes());
            image[16] = 2;
            if kind != Kind::Fat32 {
                image[17..19].copy_from_slice(&(ROOT_ENTRIES as u16).to_le_bytes());
                image[22..24].copy_from_slice(&(fat_sectors as u16).to_le_bytes());
            } else {
                image[36..40].copy_from_slice(&(fat_sectors as u32).to_le_bytes());
                image[44..48].copy_from_slice(&2u32.to_le_bytes());
            }
            image[32..36].copy_from_slice(&(total as u32).to_le_bytes());
            let serial_at = if kind == Kind::Fat32 { 67 } else { 39 };
            image[serial_at..serial_at + 4].copy_from_slice(&0xcafe_f00du32.to_le_bytes());
            let label_at = serial_at + 4;
            image[label_at..label_at + 11].copy_from_slice(b"TESTVOLUME ");
            image[510] = 0x55;
            image[511] = 0xaa;
            let mut builder = Builder {
                image,
                kind,
                fat_start: reserved * SECTOR,
                root_start: (reserved + 2 * fat_sectors) * SECTOR,
                data_start: data_sector * SECTOR,
                next_cluster: 2,
            };
            builder.set_fat(0, 0x0fff_fff8);
            builder.set_fat(1, 0x0fff_ffff);
            if kind == Kind::Fat32 {
                let root = builder.allocate_chain(1);
                assert_eq!(root[0], 2);
            }
            builder
        }

        fn set_fat(&mut self, cluster: u32, value: u32) {
            for copy in 0..2 {
                let base = self.fat_start + copy * self.fat_copy_bytes();
                match self.kind {
                    Kind::Fat12 => {
                        let at = base + cluster as usize * 3 / 2;
                        let value = (value & 0xfff) as u16;
                        if cluster & 1 == 0 {
                            self.image[at] = value as u8;
                            self.image[at + 1] = (self.image[at + 1] & 0xf0) | (value >> 8) as u8;
                        } else {
                            self.image[at] = (self.image[at] & 0x0f) | ((value << 4) as u8);
                            self.image[at + 1] = (value >> 4) as u8;
                        }
                    }
                    Kind::Fat16 => {
                        let at = base + cluster as usize * 2;
                        self.image[at..at + 2].copy_from_slice(&(value as u16).to_le_bytes());
                    }
                    Kind::Fat32 => {
                        let at = base + cluster as usize * 4;
                        self.image[at..at + 4].copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }

        fn fat_copy_bytes(&self) -> usize {
            (self.root_start - self.fat_start) / 2
        }

        fn end_of_chain(&self) -> u32 {
            match self.kind {
                Kind::Fat12 => 0xfff,
                Kind::Fat16 => 0xffff,
                Kind::Fat32 => 0x0fff_ffff,
            }
        }

        /// Takes the next `count` clusters in order, chained.
        pub fn allocate_chain(&mut self, count: usize) -> Vec<u32> {
            let clusters: Vec<u32> = (0..count as u32).map(|n| self.next_cluster + n).collect();
            self.next_cluster += count as u32;
            self.link(&clusters);
            clusters
        }

        fn link(&mut self, clusters: &[u32]) {
            for (index, cluster) in clusters.iter().enumerate() {
                let next = clusters
                    .get(index + 1)
                    .copied()
                    .unwrap_or(self.end_of_chain());
                self.set_fat(*cluster, next);
            }
        }

        fn write_cluster(&mut self, cluster: u32, data: &[u8]) {
            let at = self.data_start + (cluster as usize - 2) * SECTOR;
            self.image[at..at + data.len()].copy_from_slice(data);
        }

        /// Writes `data` into `clusters`.
        fn write_chain(&mut self, clusters: &[u32], data: &[u8]) {
            for (cluster, chunk) in clusters.iter().zip(data.chunks(SECTOR)) {
                self.write_cluster(*cluster, chunk);
            }
        }

        fn directory_bytes(entries: &[(String, u8, u32, u32)], dot: Option<(u32, u32)>) -> Vec<u8> {
            let mut bytes = Vec::new();
            let mut short_index = 0;
            let short_entry = |name: &[u8; 11], attributes: u8, cluster: u32, size: u32| {
                let mut raw = [0u8; 32];
                raw[..11].copy_from_slice(name);
                raw[11] = attributes;
                raw[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
                raw[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
                raw[28..32].copy_from_slice(&size.to_le_bytes());
                raw
            };
            if let Some((own, parent)) = dot {
                bytes.extend_from_slice(&short_entry(b".          ", ATTRIBUTE_DIRECTORY, own, 0));
                bytes.extend_from_slice(&short_entry(
                    b"..         ",
                    ATTRIBUTE_DIRECTORY,
                    parent,
                    0,
                ));
            }
            for (name, attributes, cluster, size) in entries {
                short_index += 1;
                let (stem, extension) = match name.rsplit_once('.') {
                    Some((stem, extension)) => (stem, extension),
                    None => (name.as_str(), ""),
                };
                let fits_short = stem.len() <= 8
                    && extension.len() <= 3
                    && name
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '.');
                let mut short = [b' '; 11];
                if fits_short {
                    short[..stem.len()].copy_from_slice(stem.as_bytes());
                    short[8..8 + extension.len()].copy_from_slice(extension.as_bytes());
                } else {
                    let tail = alloc::format!("~{short_index}");
                    let base: Vec<u8> = stem
                        .bytes()
                        .filter(|b| b.is_ascii_alphanumeric())
                        .map(|b| b.to_ascii_uppercase())
                        .take(8 - tail.len())
                        .collect();
                    short[..base.len()].copy_from_slice(&base);
                    short[base.len()..base.len() + tail.len()].copy_from_slice(tail.as_bytes());
                    for (slot, byte) in extension.bytes().take(3).enumerate() {
                        short[8 + slot] = byte.to_ascii_uppercase();
                    }
                    let units: Vec<u16> = name.encode_utf16().collect();
                    let chunks: Vec<&[u16]> = units.chunks(LONG_NAME_CHARS_PER_ENTRY).collect();
                    let checksum = short
                        .iter()
                        .fold(0u8, |sum, &b| sum.rotate_right(1).wrapping_add(b));
                    for (index, chunk) in chunks.iter().enumerate().rev() {
                        let mut raw = [0u8; 32];
                        raw[0] = (index as u8 + 1)
                            | if index == chunks.len() - 1 {
                                LONG_NAME_LAST
                            } else {
                                0
                            };
                        raw[11] = ATTRIBUTE_LONG_NAME;
                        raw[13] = checksum;
                        for (slot, offset) in LONG_NAME_OFFSETS.iter().enumerate() {
                            let unit = match chunk.get(slot) {
                                Some(unit) => *unit,
                                None if slot == chunk.len() => 0,
                                None => 0xffff,
                            };
                            raw[*offset..*offset + 2].copy_from_slice(&unit.to_le_bytes());
                        }
                        bytes.extend_from_slice(&raw);
                    }
                }
                bytes.extend_from_slice(&short_entry(&short, *attributes, *cluster, *size));
            }
            bytes
        }

        /// Adds a file of `data`; `fragmented` interleaves its clusters with gaps so the chain
        /// is not contiguous.
        pub fn file(&mut self, data: &[u8], fragmented: bool) -> (u32, u32) {
            let count = data.len().div_ceil(SECTOR).max(1);
            if data.is_empty() {
                return (0, 0);
            }
            let clusters: Vec<u32> = if fragmented {
                let start = self.next_cluster;
                self.next_cluster += 2 * count as u32;
                (0..count as u32).map(|n| start + 2 * n).collect()
            } else {
                let start = self.next_cluster;
                self.next_cluster += count as u32;
                (0..count as u32).map(|n| start + n).collect()
            };
            self.link(&clusters);
            self.write_chain(&clusters, data);
            (clusters[0], data.len() as u32)
        }

        /// A directory holding `entries` as (name, attributes, first cluster, size); returns its
        /// first cluster. `parent` is the cluster `..` points at (0 for the root).
        pub fn directory(&mut self, entries: &[(String, u8, u32, u32)], parent: u32) -> u32 {
            let own = self.next_cluster;
            let bytes = Self::directory_bytes(entries, Some((own, parent)));
            let clusters = self.allocate_chain(bytes.len().div_ceil(SECTOR).max(1));
            self.write_chain(&clusters, &bytes);
            own
        }

        pub fn set_root(&mut self, entries: &[(String, u8, u32, u32)]) {
            let bytes = Self::directory_bytes(entries, None);
            if self.kind == Kind::Fat32 {
                let clusters: Vec<u32> = vec![2];
                assert!(bytes.len() <= SECTOR);
                self.write_chain(&clusters, &bytes);
            } else {
                self.image[self.root_start..self.root_start + bytes.len()].copy_from_slice(&bytes);
            }
        }

        pub fn finish(self) -> Image {
            Image(self.image)
        }
    }

    pub fn file_entry(name: &str, cluster: u32, size: u32) -> (String, u8, u32, u32) {
        (name.into(), 0x20, cluster, size)
    }

    pub fn directory_entry(name: &str, cluster: u32) -> (String, u8, u32, u32) {
        (name.into(), ATTRIBUTE_DIRECTORY, cluster, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn fat32_reads_the_selected_active_fat() {
        const EXT_FLAGS_OFFSET: usize = 40;
        const MIRRORING_DISABLED: u16 = 0x80;
        const SECTOR_BYTES: usize = 512;
        for active in 0..2u16 {
            let mut builder = Builder::new(Size::Fat32);
            let data = pattern(2 * SECTOR_BYTES);
            let (cluster, size) = builder.file(&data, true);
            builder.set_root(&[file_entry("BOOTX64.EFI", cluster, size)]);
            builder.image[EXT_FLAGS_OFFSET..EXT_FLAGS_OFFSET + 2]
                .copy_from_slice(&(MIRRORING_DISABLED | active).to_le_bytes());
            let reserved = u16_at(&builder.image, 14) as usize;
            let fat_sectors = u32_at(&builder.image, 36) as usize;
            let inactive = 1 - active as usize;
            let entry = (reserved + inactive * fat_sectors) * SECTOR_BYTES + cluster as usize * 4;
            // The inactive FAT is allowed to contain an obsolete end-of-chain.
            builder.image[entry..entry + 4].copy_from_slice(&0x0fff_ffffu32.to_le_bytes());
            let mut volume = Volume::open(builder.finish()).unwrap();
            let file = volume.lookup(volume.root(), "BOOTX64.EFI").unwrap();
            assert_eq!(read_all(&mut volume, &file, SECTOR_BYTES), data);
        }
    }

    #[test]
    fn fat32_rejects_an_active_fat_outside_the_volume() {
        let mut builder = Builder::new(Size::Fat32);
        // Two FATs exist, so selecting index 2 with mirroring disabled is invalid.
        builder.image[40..42].copy_from_slice(&0x82u16.to_le_bytes());
        assert!(matches!(
            Volume::open(builder.finish()),
            Err(FatError::Corrupt)
        ));
    }

    #[test]
    fn mirrored_fat32_ignores_the_active_fat_bits() {
        let mut builder = Builder::new(Size::Fat32);
        // When mirroring is enabled the low nibble is not an active FAT index.
        builder.image[40..42].copy_from_slice(&0x0fu16.to_le_bytes());
        let data = pattern(1024);
        let (cluster, size) = builder.file(&data, true);
        builder.set_root(&[file_entry("BOOTX64.EFI", cluster, size)]);
        let mut volume = Volume::open(builder.finish()).unwrap();
        let file = volume.lookup(volume.root(), "BOOTX64.EFI").unwrap();
        assert_eq!(read_all(&mut volume, &file, 512), data);
    }

    fn tiny_fat() -> Vec<u8> {
        let mut b = vec![0; 5 * 512];
        b[..3].copy_from_slice(&[0xeb, 0x3c, 0x90]);
        b[3..11].copy_from_slice(b"MSDOS5.0");
        b[21] = 0xf8;
        b[24..26].copy_from_slice(&1u16.to_le_bytes());
        b[26..28].copy_from_slice(&1u16.to_le_bytes());
        b[11..13].copy_from_slice(&512u16.to_le_bytes());
        b[13] = 1;
        b[14..16].copy_from_slice(&1u16.to_le_bytes());
        b[16] = 1;
        b[17..19].copy_from_slice(&16u16.to_le_bytes());
        b[19..21].copy_from_slice(&5u16.to_le_bytes());
        b[22..24].copy_from_slice(&1u16.to_le_bytes());
        b[510..512].copy_from_slice(&[0x55, 0xaa]);
        b[512..518].copy_from_slice(&[0xf8, 0xff, 0xff, 0x03, 0xf0, 0xff]);
        b[1024..1035].copy_from_slice(b"BOOTX64 EFI");
        b[1035] = 0x20;
        b[1050..1052].copy_from_slice(&2u16.to_le_bytes());
        b[1052..1056].copy_from_slice(&1024u32.to_le_bytes());
        b[1536..2048].fill(0x11);
        b[2048..2560].fill(0x22);
        b
    }

    #[test]
    fn a_small_fat_volume_reads_a_file_spanning_clusters() {
        let mut volume = Volume::open(Image(tiny_fat())).unwrap();
        let entry = volume.lookup(0, "BOOTX64.EFI").unwrap();
        let mut bytes = [0; 1024];
        assert_eq!(
            volume.read(&mut Cursor::new(&entry), 0, &mut bytes),
            Ok(1024)
        );
        assert_eq!(&bytes[..512], &[0x11; 512]);
        assert_eq!(&bytes[512..], &[0x22; 512]);
    }

    #[test]
    fn invalid_initial_file_clusters_are_corrupt_not_storage_addresses() {
        let mut volume = Volume::open(Image(tiny_fat())).unwrap();
        let entry = volume.lookup(0, "BOOTX64.EFI").unwrap();
        for first_cluster in [0, 1, volume.cluster_count + FIRST_DATA_CLUSTER, u32::MAX] {
            let entry = DirEntry {
                first_cluster,
                ..entry.clone()
            };
            assert_eq!(
                volume.read(&mut Cursor::new(&entry), 0, &mut [0]),
                Err(FatError::Corrupt)
            );
        }
    }

    #[test]
    fn invalid_initial_directory_clusters_are_corrupt_not_storage_addresses() {
        let mut volume = Volume::open(Builder::new(Size::Fat32).finish()).unwrap();
        for cluster in [0, 1, volume.cluster_count + FIRST_DATA_CLUSTER, u32::MAX] {
            volume.root_cluster = cluster;
            assert_eq!(volume.list(volume.root()), Err(FatError::Corrupt));
        }
    }

    #[test]
    fn fat_entries_cannot_be_read_from_the_following_region() {
        let mut volume = Volume::open(Image(tiny_fat())).unwrap();
        const FAT_BYTES: u64 = 512;
        assert_eq!(volume.fat_byte(FAT_BYTES), Err(FatError::Corrupt));
    }

    fn pattern(length: usize) -> Vec<u8> {
        (0..length).map(|index| (index * 7 + 3) as u8).collect()
    }

    fn sample(size: Size) -> (Volume<Image>, Vec<u8>, Vec<u8>) {
        let mut builder = Builder::new(size);
        let kernel = pattern(5000);
        let fragmented = pattern(3000);
        let kernel_at = builder.file(&kernel, false);
        let fragmented_at = builder.file(&fragmented, true);
        let boot = builder.directory(
            &[
                file_entry("BOOTX64.EFI", kernel_at.0, kernel_at.1),
                file_entry("grub.cfg", fragmented_at.0, fragmented_at.1),
                file_entry("A long file name.txt", 0, 0),
            ],
            0,
        );
        let efi = builder.directory(&[directory_entry("BOOT", boot)], 0);
        builder.set_root(&[directory_entry("EFI", efi), file_entry("readme.txt", 0, 0)]);
        (Volume::open(builder.finish()).unwrap(), kernel, fragmented)
    }

    fn read_all(volume: &mut Volume<Image>, entry: &DirEntry, chunk: usize) -> Vec<u8> {
        let mut cursor = Cursor::new(entry);
        let mut data = Vec::new();
        let mut buffer = alloc::vec![0u8; chunk];
        loop {
            let count = volume
                .read(&mut cursor, data.len() as u64, &mut buffer)
                .unwrap();
            if count == 0 {
                return data;
            }
            data.extend_from_slice(&buffer[..count]);
        }
    }

    fn check_volume(size: Size) {
        let (mut volume, kernel, fragmented) = sample(size);
        assert_eq!(volume.label, "TESTVOLUME");
        let root = volume.list(volume.root()).unwrap();
        let names: Vec<&str> = root.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["EFI", "readme.txt"]);
        let efi = volume.lookup(volume.root(), "efi").unwrap();
        assert!(efi.is_directory());
        let boot = volume.lookup(efi.first_cluster, "BOOT").unwrap();
        let listing = volume.list(boot.first_cluster).unwrap();
        let names: Vec<&str> = listing.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(
            names,
            [".", "..", "BOOTX64.EFI", "grub.cfg", "A long file name.txt"]
        );
        let loader = volume.lookup(boot.first_cluster, "bootx64.efi").unwrap();
        assert_eq!(read_all(&mut volume, &loader, 4096), kernel);
        assert_eq!(read_all(&mut volume, &loader, 100), kernel);
        let config = volume.lookup(boot.first_cluster, "GRUB.CFG").unwrap();
        assert_eq!(read_all(&mut volume, &config, 700), fragmented);
        let long = volume
            .lookup(boot.first_cluster, "a LONG file name.TXT")
            .unwrap();
        assert_eq!(long.size, 0);
        assert_eq!(
            volume.lookup(boot.first_cluster, "missing"),
            Err(FatError::NotFound)
        );
    }

    #[test]
    fn fat12_reads_directories_and_files() {
        check_volume(Size::Fat12);
    }

    #[test]
    fn fat16_reads_directories_and_files() {
        check_volume(Size::Fat16);
    }

    #[test]
    fn fat32_reads_directories_and_files() {
        check_volume(Size::Fat32);
    }

    #[test]
    fn a_read_before_the_cursor_restarts_the_chain() {
        let (mut volume, kernel, _) = sample(Size::Fat16);
        let efi = volume.lookup(volume.root(), "EFI").unwrap();
        let boot = volume.lookup(efi.first_cluster, "BOOT").unwrap();
        let loader = volume.lookup(boot.first_cluster, "BOOTX64.EFI").unwrap();
        let mut cursor = Cursor::new(&loader);
        let mut tail = [0u8; 100];
        volume.read(&mut cursor, 4000, &mut tail).unwrap();
        assert_eq!(&tail[..], &kernel[4000..4100]);
        let mut head = [0u8; 100];
        volume.read(&mut cursor, 10, &mut head).unwrap();
        assert_eq!(&head[..], &kernel[10..110]);
        let mut past = [0u8; 10];
        assert_eq!(volume.read(&mut cursor, 5000, &mut past).unwrap(), 0);
        assert_eq!(volume.read(&mut cursor, 4995, &mut past).unwrap(), 5);
    }

    #[test]
    fn a_non_fat_disk_is_rejected() {
        assert_eq!(
            Volume::open(Image(alloc::vec![0; 4096])).err(),
            Some(FatError::Corrupt)
        );
        let mut broken = Builder::new(Size::Fat16).finish();
        broken.0[13] = 3;
        assert_eq!(Volume::open(broken).err(), Some(FatError::Corrupt));
    }

    #[test]
    fn a_looping_chain_is_an_error_not_a_hang() {
        let mut builder = Builder::new(Size::Fat16);
        let looped = builder.allocate_chain(2);
        builder.set_root(&[directory_entry("LOOP", looped[0])]);
        let mut image = builder.finish();
        // Point the second cluster back at the first.
        let fat = 512;
        let at = fat + looped[1] as usize * 2;
        image.0[at..at + 2].copy_from_slice(&(looped[0] as u16).to_le_bytes());
        let mut volume = Volume::open(image).unwrap();
        assert_eq!(volume.list(looped[0]), Err(FatError::Corrupt));
    }
}
