//! The udisk copy-on-write image format. `docs/spec/uemulator-disk-format.md` is the source of
//! truth; field and rule names here follow it, so read the spec before changing the layout.
//!
//! Every value from the file is range-checked before it is used: a corrupt or hostile image (it
//! may come from any web server) is an `InvalidData` error, never a panic or a wild offset.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeSet;

use super::{BlockBackend, ImageStorage, io};

const MAGIC: &[u8; 8] = b"UEMUDISK";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4096;

// Header field offsets (spec, "Header").
const VERSION_OFFSET: usize = 0x008;
const HEADER_SIZE_OFFSET: usize = 0x00c;
const INCOMPATIBLE_OFFSET: usize = 0x010;
const COMPATIBLE_OFFSET: usize = 0x018;
const VIRTUAL_SIZE_OFFSET: usize = 0x020;
const CLUSTER_SHIFT_OFFSET: usize = 0x028;
const MAP_OFFSET_OFFSET: usize = 0x030;
const MAP_ENTRIES_OFFSET: usize = 0x038;
const IMAGE_ID_OFFSET: usize = 0x040;
const BACKING_ID_OFFSET: usize = 0x050;
const BACKING_SIZE_OFFSET: usize = 0x060;
const LOCATOR_LENGTH_OFFSET: usize = 0x068;
const LOCATOR_OFFSET: usize = 0x080;
const CRC_OFFSET: usize = HEADER_BYTES - 4;
pub const MAX_LOCATOR_BYTES: usize = CRC_OFFSET - LOCATOR_OFFSET;

/// Incompatible feature bit: the image is an overlay over a base.
const FEATURE_BACKING: u64 = 1 << 0;
const KNOWN_INCOMPATIBLE_FEATURES: u64 = FEATURE_BACKING;

const MIN_CLUSTER_SHIFT: u32 = 16;
const MAX_CLUSTER_SHIFT: u32 = 24;
/// 64 KiB clusters: 4 KiB sub-blocks, the page size guests write in, and a map of 1 MiB per
/// 8 GiB of disk.
pub const DEFAULT_CLUSTER_SIZE: u64 = 1 << 16;
const SUB_BLOCKS: usize = 16;
/// The map is 4096-aligned so its pages are file-system blocks; one page holds 512 entries.
const MAP_ALIGNMENT: u64 = 4096;
const MAP_PAGE_ENTRIES: u64 = 512;
const ENTRY_BYTES: u64 = 8;
const MAX_MAP_ENTRIES: u64 = 1 << 32;
const SECTOR_BYTES: u64 = 512;
const ID_BYTES: usize = 16;

// Map entry fields (spec, "The map").
const ENTRY_CLUSTER_MASK: u64 = 0xffff_ffff;
const ENTRY_DATA_SHIFT: u32 = 32;
const ENTRY_ZERO_SHIFT: u32 = 48;

/// What a udisk header says, readable without opening the image (to find its base first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdiskInfo {
    pub virtual_size: u64,
    pub cluster_size: u64,
    pub image_id: [u8; 16],
    /// Set for an overlay.
    pub backing: Option<UdiskBacking>,
}

/// An overlay's base, as the overlay recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdiskBacking {
    /// Path (relative to the overlay's directory) or URL; empty when the embedder supplies the
    /// base itself.
    pub locator: String,
    /// The base's `image_id`; `None` when the base was a raw image, which has none.
    pub image_id: Option<[u8; 16]>,
    pub size: u64,
}

/// How a new image is laid out. The base, if any, is passed to `Udisk::create` itself: its size
/// and identity are recorded from it.
#[derive(Clone, Debug)]
pub struct UdiskOptions {
    virtual_size: u64,
    cluster_size: u64,
    image_id: [u8; 16],
    backing_locator: String,
}

impl UdiskOptions {
    /// `image_id` must be random: overlays tell their base apart from any other image by it.
    pub fn new(virtual_size: u64, image_id: [u8; 16]) -> UdiskOptions {
        UdiskOptions {
            virtual_size,
            cluster_size: DEFAULT_CLUSTER_SIZE,
            image_id,
            backing_locator: String::new(),
        }
    }

    /// A power of two from 64 KiB to 16 MiB; 64 KiB by default.
    pub fn cluster_size(mut self, bytes: u64) -> UdiskOptions {
        self.cluster_size = bytes;
        self
    }

    /// Where tools find the base when they open the overlay by itself (`UdiskBacking::locator`).
    pub fn backing_locator(mut self, locator: impl Into<String>) -> UdiskOptions {
        self.backing_locator = locator.into();
        self
    }
}

/// How much of the virtual disk is in each state, in bytes, and how many clusters of file space
/// are allocated (`uemulator disk info`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UdiskUsage {
    pub data_bytes: u64,
    pub zero_bytes: u64,
    pub unallocated_bytes: u64,
    pub allocated_clusters: u64,
}

/// The header fields `open` needs beyond `UdiskInfo`.
struct Layout {
    info: UdiskInfo,
    cluster_shift: u32,
    map_offset: u64,
    map_entries: u64,
}

impl Layout {
    fn map_end(&self) -> u64 {
        self.map_offset + self.map_entries * ENTRY_BYTES
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubBlock {
    Data,
    Zero,
    Unallocated,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Entry {
    data_cluster: u64,
    data_mask: u16,
    zero_mask: u16,
}

impl Entry {
    fn decode(raw: u64) -> Entry {
        Entry {
            data_cluster: raw & ENTRY_CLUSTER_MASK,
            data_mask: (raw >> ENTRY_DATA_SHIFT) as u16,
            zero_mask: (raw >> ENTRY_ZERO_SHIFT) as u16,
        }
    }

    fn encode(self) -> u64 {
        self.data_cluster
            | (self.data_mask as u64) << ENTRY_DATA_SHIFT
            | (self.zero_mask as u64) << ENTRY_ZERO_SHIFT
    }

    fn state(self, sub_block: usize) -> SubBlock {
        let bit = 1 << sub_block;
        if self.data_mask & bit != 0 {
            SubBlock::Data
        } else if self.zero_mask & bit != 0 {
            SubBlock::Zero
        } else {
            SubBlock::Unallocated
        }
    }
}

fn corrupt(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn id_at(bytes: &[u8], offset: usize) -> [u8; 16] {
    bytes[offset..offset + ID_BYTES].try_into().unwrap()
}

fn parse_header(header: &[u8; HEADER_BYTES]) -> io::Result<Layout> {
    if &header[..MAGIC.len()] != MAGIC {
        return Err(corrupt("not a udisk image"));
    }
    let version = le_u32(header, VERSION_OFFSET);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("udisk version {version}; this reader knows version {VERSION}"),
        ));
    }
    if le_u32(header, HEADER_SIZE_OFFSET) != HEADER_BYTES as u32 {
        return Err(corrupt("udisk header size is not 4096"));
    }
    if crate::kv::crc32(&header[..CRC_OFFSET]) != le_u32(header, CRC_OFFSET) {
        return Err(corrupt("udisk header checksum mismatch"));
    }
    let incompatible = le_u64(header, INCOMPATIBLE_OFFSET);
    if incompatible & !KNOWN_INCOMPATIBLE_FEATURES != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "udisk image uses features this reader does not know ({:#x})",
                incompatible & !KNOWN_INCOMPATIBLE_FEATURES
            ),
        ));
    }
    let virtual_size = le_u64(header, VIRTUAL_SIZE_OFFSET);
    if virtual_size == 0 || !virtual_size.is_multiple_of(SECTOR_BYTES) {
        return Err(corrupt(format!(
            "udisk virtual size {virtual_size} is not a nonzero multiple of 512"
        )));
    }
    let cluster_shift = header[CLUSTER_SHIFT_OFFSET] as u32;
    if !(MIN_CLUSTER_SHIFT..=MAX_CLUSTER_SHIFT).contains(&cluster_shift) {
        return Err(corrupt(format!(
            "udisk cluster shift {cluster_shift} out of range"
        )));
    }
    let cluster_size = 1u64 << cluster_shift;
    let map_offset = le_u64(header, MAP_OFFSET_OFFSET);
    let map_entries = le_u64(header, MAP_ENTRIES_OFFSET);
    if map_offset < HEADER_BYTES as u64 || !map_offset.is_multiple_of(MAP_ALIGNMENT) {
        return Err(corrupt(format!(
            "udisk map offset {map_offset:#x} is invalid"
        )));
    }
    if map_entries != virtual_size.div_ceil(cluster_size) || map_entries > MAX_MAP_ENTRIES {
        return Err(corrupt(format!(
            "udisk map has {map_entries} entries for a disk of {virtual_size} bytes"
        )));
    }
    if map_offset.checked_add(map_entries * ENTRY_BYTES).is_none() {
        return Err(corrupt("udisk map offset is invalid"));
    }
    let backing_id = id_at(header, BACKING_ID_OFFSET);
    let backing_size = le_u64(header, BACKING_SIZE_OFFSET);
    let locator_length = u16::from_le_bytes([
        header[LOCATOR_LENGTH_OFFSET],
        header[LOCATOR_LENGTH_OFFSET + 1],
    ]) as usize;
    if locator_length > MAX_LOCATOR_BYTES {
        return Err(corrupt("udisk backing locator is too long"));
    }
    let locator = core::str::from_utf8(&header[LOCATOR_OFFSET..LOCATOR_OFFSET + locator_length])
        .map_err(|_| corrupt("udisk backing locator is not UTF-8"))?;
    let backing = if incompatible & FEATURE_BACKING != 0 {
        if backing_size == 0 {
            return Err(corrupt("udisk overlay records a base of size 0"));
        }
        Some(UdiskBacking {
            locator: locator.into(),
            image_id: (backing_id != [0; ID_BYTES]).then_some(backing_id),
            size: backing_size,
        })
    } else {
        if backing_size != 0 || backing_id != [0; ID_BYTES] || locator_length != 0 {
            return Err(corrupt(
                "udisk image has base fields but no BACKING feature",
            ));
        }
        None
    };
    Ok(Layout {
        info: UdiskInfo {
            virtual_size,
            cluster_size,
            image_id: id_at(header, IMAGE_ID_OFFSET),
            backing,
        },
        cluster_shift,
        map_offset,
        map_entries,
    })
}

fn build_header(layout: &Layout) -> [u8; HEADER_BYTES] {
    let mut header = [0u8; HEADER_BYTES];
    let mut put =
        |offset: usize, bytes: &[u8]| header[offset..offset + bytes.len()].copy_from_slice(bytes);
    put(0, MAGIC);
    put(VERSION_OFFSET, &VERSION.to_le_bytes());
    put(HEADER_SIZE_OFFSET, &(HEADER_BYTES as u32).to_le_bytes());
    let info = &layout.info;
    let features = if info.backing.is_some() {
        FEATURE_BACKING
    } else {
        0
    };
    put(INCOMPATIBLE_OFFSET, &features.to_le_bytes());
    put(COMPATIBLE_OFFSET, &0u64.to_le_bytes());
    put(VIRTUAL_SIZE_OFFSET, &info.virtual_size.to_le_bytes());
    put(CLUSTER_SHIFT_OFFSET, &[layout.cluster_shift as u8]);
    put(MAP_OFFSET_OFFSET, &layout.map_offset.to_le_bytes());
    put(MAP_ENTRIES_OFFSET, &layout.map_entries.to_le_bytes());
    put(IMAGE_ID_OFFSET, &info.image_id);
    if let Some(backing) = &info.backing {
        put(
            BACKING_ID_OFFSET,
            &backing.image_id.unwrap_or([0; ID_BYTES]),
        );
        put(BACKING_SIZE_OFFSET, &backing.size.to_le_bytes());
        put(
            LOCATOR_LENGTH_OFFSET,
            &(backing.locator.len() as u16).to_le_bytes(),
        );
        put(LOCATOR_OFFSET, backing.locator.as_bytes());
    }
    let crc = crate::kv::crc32(&header[..CRC_OFFSET]);
    header[CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
    header
}

/// A udisk image as a disk: reads and writes follow the spec's rules, falling through to the base
/// for unallocated sub-blocks. Map changes are written on `flush` (and, best effort, on drop),
/// after the data they point to is durable.
pub struct Udisk {
    storage: Box<dyn ImageStorage>,
    backing: Option<Box<dyn BlockBackend>>,
    info: UdiskInfo,
    cluster_shift: u32,
    map_offset: u64,
    map_entries: u64,
    first_data_cluster: u64,
    /// Where the next cluster of file space goes; only grows (spec, "Writing", rule 1). A map
    /// entry pointing at or past it is corrupt: the space would be handed out a second time.
    next_cluster: u64,
    /// Map pages, loaded the first time a cluster they cover is touched.
    pages: Vec<Option<Box<[u64]>>>,
    /// Pages with entries changed since the last flush.
    dirty: BTreeSet<u64>,
}

impl Udisk {
    /// The header of `storage` if it is a udisk image, `None` if it is something else (a raw
    /// image). A file that starts with the magic but has a bad header is an error.
    pub fn probe(storage: &mut dyn ImageStorage) -> io::Result<Option<UdiskInfo>> {
        if storage.len() < MAGIC.len() as u64 {
            return Ok(None);
        }
        let mut magic = [0u8; 8];
        storage.read_at(0, &mut magic)?;
        if &magic != MAGIC {
            return Ok(None);
        }
        Self::read_layout(storage).map(|layout| Some(layout.info))
    }

    fn read_layout(storage: &mut dyn ImageStorage) -> io::Result<Layout> {
        if storage.len() < HEADER_BYTES as u64 {
            return Err(corrupt("udisk header is truncated"));
        }
        let mut header = [0u8; HEADER_BYTES];
        storage.read_at(0, &mut header)?;
        let layout = parse_header(&header)?;
        if storage.len() < layout.map_end() {
            return Err(corrupt("udisk map extends past the end of the file"));
        }
        Ok(layout)
    }

    /// Opens an image. `backing` must be given exactly when the image is an overlay, and must
    /// be the base it was made over: same size, and the same image id when it recorded one.
    pub fn open(
        storage: impl ImageStorage + 'static,
        backing: Option<Box<dyn BlockBackend>>,
    ) -> io::Result<Udisk> {
        let mut storage: Box<dyn ImageStorage> = Box::new(storage);
        let layout = Self::read_layout(&mut *storage)?;
        match (&layout.info.backing, &backing) {
            (None, None) => {}
            (Some(_), None) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "udisk image is an overlay and needs its base",
                ));
            }
            (None, Some(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "udisk image has no base, but one was given",
                ));
            }
            (Some(recorded), Some(base)) => {
                if base.len() != recorded.size {
                    return Err(corrupt(format!(
                        "the base is {} bytes, but the overlay was made over {} bytes",
                        base.len(),
                        recorded.size
                    )));
                }
                if let Some(id) = recorded.image_id
                    && base.image_id() != Some(id)
                {
                    return Err(corrupt(
                        "the base is not the image the overlay was made over",
                    ));
                }
            }
        }
        let cluster_size = layout.info.cluster_size;
        let first_data_cluster = layout.map_end().div_ceil(cluster_size);
        let next_cluster = first_data_cluster.max(storage.len().div_ceil(cluster_size));
        let pages = layout.map_entries.div_ceil(MAP_PAGE_ENTRIES);
        Ok(Udisk {
            storage,
            backing,
            info: layout.info,
            cluster_shift: layout.cluster_shift,
            map_offset: layout.map_offset,
            map_entries: layout.map_entries,
            first_data_cluster,
            next_cluster,
            pages: (0..pages).map(|_| None).collect(),
            dirty: BTreeSet::new(),
        })
    }

    /// Writes a new image into empty `storage`: the header, then an all-zero map (every
    /// sub-block unallocated). `backing` makes it an overlay over that base.
    pub fn create(
        mut storage: impl ImageStorage + 'static,
        options: &UdiskOptions,
        backing: Option<Box<dyn BlockBackend>>,
    ) -> io::Result<Udisk> {
        let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidInput, message);
        if !storage.is_empty() {
            return Err(invalid("a new udisk image needs empty storage".into()));
        }
        let cluster_size = options.cluster_size;
        let cluster_shift = cluster_size.trailing_zeros();
        if !cluster_size.is_power_of_two()
            || !(MIN_CLUSTER_SHIFT..=MAX_CLUSTER_SHIFT).contains(&cluster_shift)
        {
            return Err(invalid(format!(
                "cluster size {cluster_size} is not a power of two from 64 KiB to 16 MiB"
            )));
        }
        let virtual_size = options.virtual_size;
        let map_entries = virtual_size.div_ceil(cluster_size);
        if virtual_size == 0
            || !virtual_size.is_multiple_of(SECTOR_BYTES)
            || map_entries > MAX_MAP_ENTRIES
        {
            return Err(invalid(format!(
                "disk size {virtual_size} is not a nonzero multiple of 512 within the format's limit"
            )));
        }
        if options.backing_locator.len() > MAX_LOCATOR_BYTES {
            return Err(invalid("backing locator is too long".into()));
        }
        let backing_record = match &backing {
            Some(base) if base.is_empty() => return Err(invalid("the base is empty".into())),
            Some(base) => Some(UdiskBacking {
                locator: options.backing_locator.clone(),
                image_id: base.image_id(),
                size: base.len(),
            }),
            None if !options.backing_locator.is_empty() => {
                return Err(invalid("a backing locator needs a base".into()));
            }
            None => None,
        };
        let layout = Layout {
            info: UdiskInfo {
                virtual_size,
                cluster_size,
                image_id: options.image_id,
                backing: backing_record,
            },
            cluster_shift,
            map_offset: HEADER_BYTES as u64,
            map_entries,
        };
        storage.write_at(0, &build_header(&layout))?;
        // Extends the file over the map; the map's zeros are holes on most file systems.
        storage.write_at(layout.map_end() - 1, &[0])?;
        storage.flush()?;
        Self::open(storage, backing)
    }

    pub fn info(&self) -> &UdiskInfo {
        &self.info
    }

    fn sub_block_bytes(&self) -> usize {
        (self.info.cluster_size as usize) / SUB_BLOCKS
    }

    fn page(&mut self, index: u64) -> io::Result<&mut [u64]> {
        let slot = index as usize;
        if self.pages[slot].is_none() {
            let count = (self.map_entries - index * MAP_PAGE_ENTRIES).min(MAP_PAGE_ENTRIES);
            let mut bytes = vec![0u8; (count * ENTRY_BYTES) as usize];
            self.storage.read_at(
                self.map_offset + index * MAP_PAGE_ENTRIES * ENTRY_BYTES,
                &mut bytes,
            )?;
            let entries = bytes
                .as_chunks::<{ ENTRY_BYTES as usize }>()
                .0
                .iter()
                .map(|raw| u64::from_le_bytes(*raw))
                .collect();
            self.pages[slot] = Some(entries);
        }
        Ok(self.pages[slot].as_deref_mut().expect("just loaded"))
    }

    /// The map entry of `cluster`, checked against the spec's rules for readers.
    fn entry(&mut self, cluster: u64) -> io::Result<Entry> {
        let raw = self.page(cluster / MAP_PAGE_ENTRIES)?[(cluster % MAP_PAGE_ENTRIES) as usize];
        let entry = Entry::decode(raw);
        if entry.data_mask & entry.zero_mask != 0 {
            return Err(corrupt(format!(
                "udisk map entry {cluster} marks sub-blocks both data and zero"
            )));
        }
        if entry.data_cluster != 0
            && (entry.data_cluster < self.first_data_cluster
                || entry.data_cluster >= self.next_cluster)
        {
            return Err(corrupt(format!(
                "udisk map entry {cluster} points outside the data area"
            )));
        }
        if entry.data_mask != 0 && entry.data_cluster == 0 {
            return Err(corrupt(format!(
                "udisk map entry {cluster} has data but no cluster"
            )));
        }
        let cluster_start = cluster << self.cluster_shift;
        let live = (self.info.virtual_size - cluster_start).div_ceil(self.sub_block_bytes() as u64);
        if live < SUB_BLOCKS as u64 {
            let past_end = !((1u16 << live) - 1);
            if (entry.data_mask | entry.zero_mask) & past_end != 0 {
                return Err(corrupt(format!(
                    "udisk map entry {cluster} has sub-blocks past the end of the disk"
                )));
            }
        }
        Ok(entry)
    }

    fn set_entry(&mut self, cluster: u64, entry: Entry) -> io::Result<()> {
        let page = cluster / MAP_PAGE_ENTRIES;
        self.page(page)?[(cluster % MAP_PAGE_ENTRIES) as usize] = entry.encode();
        self.dirty.insert(page);
        Ok(())
    }

    /// What the base holds at `offset`: zeros past its end, or everywhere without a base.
    fn read_backing(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let available = match &mut self.backing {
            Some(base) => {
                let available = base.len().saturating_sub(offset).min(buf.len() as u64) as usize;
                if available > 0 {
                    base.read_at(offset, &mut buf[..available])?;
                }
                available
            }
            None => 0,
        };
        buf[available..].fill(0);
        Ok(())
    }

    /// Reads `buf.len()` bytes at `within` in `cluster`, one run of same-state sub-blocks at a
    /// time.
    fn read_cluster(&mut self, cluster: u64, within: usize, buf: &mut [u8]) -> io::Result<()> {
        let entry = self.entry(cluster)?;
        let sub_block = self.sub_block_bytes();
        let end = within + buf.len();
        let mut position = within;
        while position < end {
            let state = entry.state(position / sub_block);
            let mut run_end = (position / sub_block + 1) * sub_block;
            while run_end < end && entry.state(run_end / sub_block) == state {
                run_end += sub_block;
            }
            let run_end = run_end.min(end);
            let piece = &mut buf[position - within..run_end - within];
            match state {
                SubBlock::Data => self.storage.read_at(
                    (entry.data_cluster << self.cluster_shift) + position as u64,
                    piece,
                )?,
                SubBlock::Zero => piece.fill(0),
                SubBlock::Unallocated => {
                    self.read_backing((cluster << self.cluster_shift) + position as u64, piece)?
                }
            }
            position = run_end;
        }
        Ok(())
    }

    /// Writes `data` at `within` in `cluster`. Sub-blocks that become data are written whole,
    /// completed from what they read as before; all-zero sub-blocks that are not data yet are
    /// not stored (spec, "Writing").
    fn write_cluster(&mut self, cluster: u64, within: usize, data: &[u8]) -> io::Result<()> {
        let mut entry = self.entry(cluster)?;
        let original = entry;
        let sub_block = self.sub_block_bytes();
        let end = within + data.len();
        let span_start = within / sub_block * sub_block;
        let span_end = end.div_ceil(sub_block) * sub_block;
        let mut buffer = vec![0u8; span_end - span_start];
        buffer[within - span_start..end - span_start].copy_from_slice(data);
        // Byte ranges within the cluster to store, merged when adjacent.
        let mut writes: Vec<(usize, usize)> = Vec::new();
        let mut scratch = Vec::new();
        for index in span_start / sub_block..span_end / sub_block {
            let (start, stop) = (index * sub_block, (index + 1) * sub_block);
            let (from, to) = (within.max(start), end.min(stop));
            let bit = 1u16 << index;
            let range = match entry.state(index) {
                SubBlock::Data => (from, to),
                state => {
                    if state == SubBlock::Unallocated && (from, to) != (start, stop) {
                        scratch.resize(sub_block, 0);
                        self.read_backing(
                            (cluster << self.cluster_shift) + start as u64,
                            &mut scratch,
                        )?;
                        let local = &mut buffer[start - span_start..stop - span_start];
                        local[..from - start].copy_from_slice(&scratch[..from - start]);
                        local[to - start..].copy_from_slice(&scratch[to - start..]);
                    }
                    if buffer[start - span_start..stop - span_start]
                        .iter()
                        .all(|byte| *byte == 0)
                    {
                        // Without a base, unallocated already reads as zeros.
                        if self.backing.is_some() {
                            entry.zero_mask |= bit;
                        }
                        continue;
                    }
                    entry.zero_mask &= !bit;
                    entry.data_mask |= bit;
                    (start, stop)
                }
            };
            match writes.last_mut() {
                Some(last) if last.1 == range.0 => last.1 = range.1,
                _ => writes.push(range),
            }
        }
        if !writes.is_empty() && entry.data_cluster == 0 {
            if self.next_cluster > ENTRY_CLUSTER_MASK {
                return Err(io::Error::other("udisk image is full"));
            }
            entry.data_cluster = self.next_cluster;
            self.next_cluster += 1;
        }
        let base = entry.data_cluster << self.cluster_shift;
        for (from, to) in writes {
            self.storage.write_at(
                base + from as u64,
                &buffer[from - span_start..to - span_start],
            )?;
        }
        if entry != original {
            self.set_entry(cluster, entry)?;
        }
        Ok(())
    }

    /// Calls `body(cluster, offset within it, byte range of the request)` for each cluster the
    /// request `offset..offset + length` touches.
    fn for_each_cluster(
        &mut self,
        offset: u64,
        length: usize,
        mut body: impl FnMut(&mut Self, u64, usize, core::ops::Range<usize>) -> io::Result<()>,
    ) -> io::Result<()> {
        super::checked_range(self.info.virtual_size, offset, length)?;
        let mut done = 0;
        while done < length {
            let position = offset + done as u64;
            let cluster = position >> self.cluster_shift;
            let within = (position & (self.info.cluster_size - 1)) as usize;
            let count = (self.info.cluster_size as usize - within).min(length - done);
            body(self, cluster, within, done..done + count)?;
            done += count;
        }
        Ok(())
    }

    /// Bytes per state over the whole disk; reads the entire map.
    pub fn usage(&mut self) -> io::Result<UdiskUsage> {
        let mut usage = UdiskUsage::default();
        let sub_block = self.sub_block_bytes() as u64;
        for cluster in 0..self.map_entries {
            let entry = self.entry(cluster)?;
            usage.allocated_clusters += (entry.data_cluster != 0) as u64;
            let cluster_start = cluster << self.cluster_shift;
            for index in 0..SUB_BLOCKS {
                let start = cluster_start + index as u64 * sub_block;
                if start >= self.info.virtual_size {
                    break;
                }
                let bytes = sub_block.min(self.info.virtual_size - start);
                match entry.state(index) {
                    SubBlock::Data => usage.data_bytes += bytes,
                    SubBlock::Zero => usage.zero_bytes += bytes,
                    SubBlock::Unallocated => usage.unallocated_bytes += bytes,
                }
            }
        }
        Ok(usage)
    }
}

impl BlockBackend for Udisk {
    fn len(&self) -> u64 {
        self.info.virtual_size
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.for_each_cluster(offset, buf.len(), |disk, cluster, within, range| {
            disk.read_cluster(cluster, within, &mut buf[range])
        })
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.storage.read_only() {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        self.for_each_cluster(offset, data.len(), |disk, cluster, within, range| {
            disk.write_cluster(cluster, within, &data[range])
        })
    }
    /// Data first, then the map (spec, "Writing", rule 3).
    fn flush(&mut self) -> io::Result<()> {
        if self.storage.read_only() {
            return Ok(());
        }
        self.storage.flush()?;
        if self.dirty.is_empty() {
            return Ok(());
        }
        let dirty: Vec<u64> = self.dirty.iter().copied().collect();
        for page in dirty {
            let bytes: Vec<u8> = self.pages[page as usize]
                .as_deref()
                .expect("a dirty page is loaded")
                .iter()
                .flat_map(|entry| entry.to_le_bytes())
                .collect();
            self.storage.write_at(
                self.map_offset + page * MAP_PAGE_ENTRIES * ENTRY_BYTES,
                &bytes,
            )?;
        }
        self.storage.flush()?;
        self.dirty.clear();
        Ok(())
    }
    fn read_only(&self) -> bool {
        self.storage.read_only()
    }
    fn image_id(&self) -> Option<[u8; 16]> {
        Some(self.info.image_id)
    }
}

impl Drop for Udisk {
    /// A machine that is dropped without the guest flushing still keeps what it wrote, as a raw
    /// file would. Errors have nowhere to go here.
    fn drop(&mut self) {
        if !self.dirty.is_empty() {
            let _ = self.flush();
        }
    }
}

#[cfg(test)]
#[path = "udisk_tests.rs"]
mod tests;
