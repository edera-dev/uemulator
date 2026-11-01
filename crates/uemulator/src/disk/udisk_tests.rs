use super::*;
use crate::disk::{MemDisk, REMOTE_CHUNK_BYTES, RemoteImage, checked_range};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const SUB_BLOCK: usize = 4096;
const CLUSTER: usize = 64 * 1024;
const ID: [u8; 16] = *b"udisk-test-id-01";
const BASE_ID: [u8; 16] = *b"udisk-test-base!";

/// What a `Shared` storage was asked to do, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Op {
    Write(u64, usize),
    Flush,
}

/// Storage whose bytes and call log outlive the image, so a test can reopen it or look inside.
#[derive(Clone, Default)]
struct Shared {
    bytes: Arc<Mutex<Vec<u8>>>,
    log: Arc<Mutex<Vec<Op>>>,
    read_only: bool,
}

impl Shared {
    fn with_bytes(bytes: Vec<u8>) -> Shared {
        let shared = Shared::default();
        *shared.bytes.lock().unwrap() = bytes;
        shared
    }
    fn bytes(&self) -> Vec<u8> {
        self.bytes.lock().unwrap().clone()
    }
    fn take_log(&self) -> Vec<Op> {
        core::mem::take(&mut *self.log.lock().unwrap())
    }
    fn as_read_only(&self) -> Shared {
        Shared {
            read_only: true,
            ..self.clone()
        }
    }
}

impl ImageStorage for Shared {
    fn len(&self) -> u64 {
        self.bytes.lock().unwrap().len() as u64
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        ImageStorage::read_at(&mut *self.bytes.lock().unwrap(), offset, buf)
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        if self.read_only {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        self.log.lock().unwrap().push(Op::Write(offset, data.len()));
        ImageStorage::write_at(&mut *self.bytes.lock().unwrap(), offset, data)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.log.lock().unwrap().push(Op::Flush);
        Ok(())
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

/// A read-only base that counts the bytes read from it and can claim an identity.
struct CountingBase {
    data: MemDisk,
    read_bytes: Arc<AtomicU64>,
    id: Option<[u8; 16]>,
}

fn counting_base(data: Vec<u8>, id: Option<[u8; 16]>) -> (Box<dyn BlockBackend>, Arc<AtomicU64>) {
    let read_bytes = Arc::new(AtomicU64::new(0));
    let base = CountingBase {
        data: MemDisk(data),
        read_bytes: read_bytes.clone(),
        id,
    };
    (Box::new(base), read_bytes)
}

impl BlockBackend for CountingBase {
    fn len(&self) -> u64 {
        self.data.len()
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.read_bytes
            .fetch_add(buf.len() as u64, Ordering::Relaxed);
        self.data.read_at(offset, buf)
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
    fn image_id(&self) -> Option<[u8; 16]> {
        self.id
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| (index % 251) as u8 ^ seed.wrapping_add(1))
        .collect()
}

fn image(size: u64) -> Udisk {
    Udisk::create(Vec::new(), &UdiskOptions::new(size, ID), None).unwrap()
}

fn read(disk: &mut dyn BlockBackend, offset: u64, len: usize) -> Vec<u8> {
    let mut buf = vec![0xa5; len];
    disk.read_at(offset, &mut buf).unwrap();
    buf
}

fn kind<T>(result: io::Result<T>) -> io::ErrorKind {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.kind(),
    }
}

/// Deterministic pseudo-random numbers, so a failing case repeats.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn below(&mut self, limit: u64) -> u64 {
        self.next() % limit
    }
}

mod header {
    use super::*;

    #[test]
    fn probe_reads_back_what_create_wrote() {
        let storage = Shared::default();
        drop(image_in(&storage, 3 * MIB));
        let info = Udisk::probe(&mut storage.clone()).unwrap().unwrap();
        assert_eq!(
            info,
            UdiskInfo {
                virtual_size: 3 * MIB,
                cluster_size: DEFAULT_CLUSTER_SIZE,
                image_id: ID,
                backing: None,
            }
        );
    }

    fn image_in(storage: &Shared, size: u64) -> Udisk {
        Udisk::create(storage.clone(), &UdiskOptions::new(size, ID), None).unwrap()
    }

    #[test]
    fn the_header_is_laid_out_as_the_spec_says() {
        let storage = Shared::default();
        drop(image_in(&storage, 8 * MIB));
        let bytes = storage.bytes();
        assert_eq!(&bytes[..8], b"UEMUDISK");
        assert_eq!(le_u32(&bytes, 0x08), 1);
        assert_eq!(le_u32(&bytes, 0x0c), 4096);
        assert_eq!(le_u64(&bytes, 0x20), 8 * MIB);
        assert_eq!(bytes[0x28], 16);
        assert_eq!(le_u64(&bytes, 0x30), 4096);
        assert_eq!(le_u64(&bytes, 0x38), 128);
        assert_eq!(&bytes[0x40..0x50], &ID);
        // zlib's CRC-32 of the first 4092 bytes, which `kv` checks against the standard value.
        assert_eq!(le_u32(&bytes, 0xffc), crate::kv::crc32(&bytes[..0xffc]));
        assert_eq!(
            bytes.len(),
            4096 + 128 * 8,
            "the file covers the map and nothing more"
        );
        assert!(
            bytes[4096..].iter().all(|byte| *byte == 0),
            "a new map is all unallocated"
        );
    }

    #[test]
    fn an_overlay_records_its_base() {
        let (base, _) = counting_base(vec![1; MIB as usize], Some(BASE_ID));
        let options = UdiskOptions::new(2 * MIB, ID).backing_locator("../base.udisk");
        let overlay = Udisk::create(Vec::new(), &options, Some(base)).unwrap();
        assert_eq!(
            overlay.info().backing,
            Some(UdiskBacking {
                locator: "../base.udisk".into(),
                image_id: Some(BASE_ID),
                size: MIB,
            })
        );
    }

    #[test]
    fn a_raw_base_is_recorded_without_an_id() {
        let overlay = Udisk::create(
            Vec::new(),
            &UdiskOptions::new(MIB, ID),
            Some(Box::new(MemDisk(vec![0; 4096]))),
        )
        .unwrap();
        assert_eq!(overlay.info().backing.as_ref().unwrap().image_id, None);
    }

    #[test]
    fn probe_says_none_for_anything_without_the_magic() {
        assert_eq!(Udisk::probe(&mut Vec::new()).unwrap(), None);
        assert_eq!(Udisk::probe(&mut b"UEMU".to_vec()).unwrap(), None);
        assert_eq!(Udisk::probe(&mut vec![0u8; 8192]).unwrap(), None);
    }

    #[test]
    fn create_refuses_bad_options() {
        let create = |options: UdiskOptions| Udisk::create(Vec::new(), &options, None).map(|_| ());
        assert_eq!(
            kind(create(UdiskOptions::new(0, ID))),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            kind(create(UdiskOptions::new(1000, ID))),
            io::ErrorKind::InvalidInput
        );
        for size in [0, 4096, 32 * KIB, 96 * KIB, 32 * MIB] {
            assert_eq!(
                kind(create(UdiskOptions::new(MIB, ID).cluster_size(size))),
                io::ErrorKind::InvalidInput,
                "cluster size {size}"
            );
        }
        assert_eq!(
            kind(create(UdiskOptions::new(MIB, ID).backing_locator("base"))),
            io::ErrorKind::InvalidInput,
            "a locator without a base"
        );
        assert_eq!(
            kind(Udisk::create(vec![0], &UdiskOptions::new(MIB, ID), None)),
            io::ErrorKind::InvalidInput,
            "storage that is not empty"
        );
        let long = "x".repeat(MAX_LOCATOR_BYTES + 1);
        let base: Box<dyn BlockBackend> = Box::new(MemDisk(vec![0; 512]));
        assert_eq!(
            kind(Udisk::create(
                Vec::new(),
                &UdiskOptions::new(MIB, ID).backing_locator(long),
                Some(base)
            )),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn the_longest_locator_round_trips() {
        let locator = "y".repeat(MAX_LOCATOR_BYTES);
        let base: Box<dyn BlockBackend> = Box::new(MemDisk(vec![0; 512]));
        let storage = Shared::default();
        let options = UdiskOptions::new(MIB, ID).backing_locator(locator.clone());
        drop(Udisk::create(storage.clone(), &options, Some(base)).unwrap());
        let info = Udisk::probe(&mut storage.clone()).unwrap().unwrap();
        assert_eq!(info.backing.unwrap().locator, locator);
    }

    #[test]
    fn every_cluster_size_works() {
        for shift in MIN_CLUSTER_SHIFT..=MAX_CLUSTER_SHIFT {
            let cluster = 1u64 << shift;
            let size = 2 * cluster + 4096;
            let mut disk = Udisk::create(
                Vec::new(),
                &UdiskOptions::new(size, ID).cluster_size(cluster),
                None,
            )
            .unwrap();
            let data = pattern(3000, shift as u8);
            disk.write_at(cluster - 1000, &data).unwrap();
            disk.write_at(size - 3000, &data).unwrap();
            assert_eq!(read(&mut disk, cluster - 1000, 3000), data, "shift {shift}");
            assert_eq!(read(&mut disk, size - 3000, 3000), data, "shift {shift}");
        }
    }
}

mod reading_and_writing {
    use super::*;

    #[test]
    fn a_new_image_reads_as_zeros() {
        let mut disk = image(MIB);
        assert_eq!(read(&mut disk, 0, MIB as usize), vec![0; MIB as usize]);
    }

    #[test]
    fn unaligned_writes_across_clusters_read_back() {
        let mut disk = image(MIB);
        let data = pattern(3 * CLUSTER, 7);
        disk.write_at(CLUSTER as u64 - 777, &data).unwrap();
        assert_eq!(read(&mut disk, CLUSTER as u64 - 777, data.len()), data);
        assert_eq!(read(&mut disk, 0, CLUSTER - 777), vec![0; CLUSTER - 777]);
    }

    /// Random reads and writes, checked against a plain `Vec` with the same history.
    fn matches_a_flat_model(base: Option<Vec<u8>>, size: u64, seed: u64) {
        let mut model = base.clone().unwrap_or_else(|| vec![0; size as usize]);
        model.resize(size as usize, 0);
        let backing = base.map(|bytes| Box::new(MemDisk(bytes)) as Box<dyn BlockBackend>);
        let storage = Shared::default();
        let mut disk =
            Udisk::create(storage.clone(), &UdiskOptions::new(size, ID), backing).unwrap();
        let mut random = Lcg(seed);
        for step in 0..400 {
            let offset = random.below(size);
            let len = (random.below(3 * CLUSTER as u64) as usize).min((size - offset) as usize);
            if random.below(3) == 0 {
                assert_eq!(
                    read(&mut disk, offset, len),
                    model[offset as usize..offset as usize + len],
                    "step {step}"
                );
            } else {
                let data: Vec<u8> = match random.below(3) {
                    0 => vec![0; len],
                    _ => pattern(len, step as u8),
                };
                disk.write_at(offset, &data).unwrap();
                model[offset as usize..offset as usize + len].copy_from_slice(&data);
            }
            if random.below(20) == 0 {
                disk.flush().unwrap();
            }
        }
        assert_eq!(read(&mut disk, 0, size as usize), model);
    }

    #[test]
    fn random_io_matches_a_flat_disk() {
        matches_a_flat_model(None, 2 * MIB, 1);
        matches_a_flat_model(None, 2 * MIB + 3 * 512, 2);
    }

    #[test]
    fn random_io_on_an_overlay_matches_a_flat_disk() {
        matches_a_flat_model(Some(pattern(2 * MIB as usize, 9)), 2 * MIB, 3);
        // Bigger than its base: the rest reads as zeros.
        matches_a_flat_model(Some(pattern(MIB as usize + 512, 4)), 2 * MIB, 4);
    }

    #[test]
    fn what_was_flushed_survives_reopening() {
        let storage = Shared::default();
        let data = pattern(100_000, 3);
        {
            let mut disk =
                Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
            disk.write_at(12_345, &data).unwrap();
            disk.flush().unwrap();
        }
        let mut disk = Udisk::open(storage, None).unwrap();
        assert_eq!(read(&mut disk, 12_345, data.len()), data);
    }

    #[test]
    fn dropping_the_image_writes_the_map() {
        let storage = Shared::default();
        {
            let mut disk =
                Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
            disk.write_at(0, &[9; 512]).unwrap();
        }
        let mut disk = Udisk::open(storage, None).unwrap();
        assert_eq!(read(&mut disk, 0, 512), vec![9; 512]);
    }

    #[test]
    fn writes_of_zeros_store_nothing() {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        let before = storage.bytes().len();
        disk.write_at(0, &vec![0; 4 * CLUSTER]).unwrap();
        disk.write_at(100, &[0; 100]).unwrap();
        disk.flush().unwrap();
        assert_eq!(storage.bytes().len(), before);
        assert_eq!(disk.usage().unwrap().data_bytes, 0);
    }

    #[test]
    fn data_clusters_are_appended_in_write_order() {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        let map_end = 4096 + 16 * 8;
        storage.take_log();
        disk.write_at(5 * CLUSTER as u64, &[1; SUB_BLOCK]).unwrap();
        disk.write_at(2 * CLUSTER as u64 + SUB_BLOCK as u64, &[2; SUB_BLOCK])
            .unwrap();
        let first_data = (map_end as u64).div_ceil(CLUSTER as u64) * CLUSTER as u64;
        assert_eq!(
            storage.take_log(),
            vec![
                Op::Write(first_data, SUB_BLOCK),
                Op::Write(first_data + CLUSTER as u64 + SUB_BLOCK as u64, SUB_BLOCK),
            ]
        );
    }

    #[test]
    fn rewriting_stored_data_writes_only_the_bytes_given() {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        disk.write_at(0, &[1; CLUSTER]).unwrap();
        storage.take_log();
        disk.write_at(100, &[2; 10]).unwrap();
        assert_eq!(
            storage.take_log(),
            vec![Op::Write(CLUSTER as u64 + 100, 10)]
        );
    }

    #[test]
    fn flush_makes_data_durable_before_writing_the_map() {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        storage.take_log();
        disk.write_at(0, &[1; 512]).unwrap();
        let map_before_flush = storage.bytes()[4096..4104].to_vec();
        assert_eq!(map_before_flush, [0; 8], "the map waits for a flush");
        disk.flush().unwrap();
        assert_eq!(
            storage.take_log(),
            vec![
                Op::Write(CLUSTER as u64, SUB_BLOCK),
                Op::Flush,
                Op::Write(4096, 16 * 8),
                Op::Flush,
            ]
        );
        assert_ne!(storage.bytes()[4096..4104], [0; 8]);
        disk.flush().unwrap();
        assert_eq!(
            storage.take_log(),
            vec![Op::Flush],
            "nothing dirty: no map write"
        );
    }

    #[test]
    fn out_of_range_access_fails_without_side_effects() {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        storage.take_log();
        let mut buf = [0u8; 2];
        assert_eq!(
            kind(disk.read_at(MIB - 1, &mut buf)),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            kind(disk.write_at(MIB - 1, &[1, 1])),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            kind(disk.write_at(u64::MAX - 1, &[1, 1])),
            io::ErrorKind::UnexpectedEof
        );
        assert!(storage.take_log().is_empty());
        disk.read_at(MIB, &mut []).unwrap();
    }

    #[test]
    fn read_only_storage_refuses_writes() {
        let storage = Shared::default();
        drop(Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap());
        let mut disk = Udisk::open(storage.as_read_only(), None).unwrap();
        assert!(disk.read_only());
        assert_eq!(
            kind(disk.write_at(0, &[1])),
            io::ErrorKind::PermissionDenied
        );
        disk.flush().unwrap();
    }

    #[test]
    fn usage_counts_each_state() {
        let (base, _) = counting_base(vec![7; MIB as usize], None);
        let mut disk = Udisk::create(Vec::new(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap();
        disk.write_at(0, &[1; 2 * SUB_BLOCK]).unwrap();
        disk.write_at(CLUSTER as u64, &[0; 3 * SUB_BLOCK]).unwrap();
        assert_eq!(
            disk.usage().unwrap(),
            UdiskUsage {
                data_bytes: 2 * SUB_BLOCK as u64,
                zero_bytes: 3 * SUB_BLOCK as u64,
                unallocated_bytes: MIB - 5 * SUB_BLOCK as u64,
                allocated_clusters: 1,
            }
        );
    }
}

mod overlays {
    use super::*;

    #[test]
    fn unallocated_reads_come_from_the_base() {
        let base_bytes = pattern(MIB as usize, 5);
        let (base, _) = counting_base(base_bytes.clone(), Some(BASE_ID));
        let mut overlay =
            Udisk::create(Vec::new(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap();
        assert_eq!(read(&mut overlay, 0, MIB as usize), base_bytes);
    }

    #[test]
    fn a_sector_write_copies_one_sub_block_from_the_base() {
        let (base, read_bytes) = counting_base(pattern(MIB as usize, 5), None);
        let mut overlay =
            Udisk::create(Vec::new(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap();
        overlay
            .write_at(3 * SUB_BLOCK as u64 + 512, &[1; 512])
            .unwrap();
        assert_eq!(read_bytes.load(Ordering::Relaxed), SUB_BLOCK as u64);
        let mut expected = pattern(MIB as usize, 5)[..CLUSTER].to_vec();
        expected[3 * SUB_BLOCK + 512..3 * SUB_BLOCK + 1024].fill(1);
        assert_eq!(read(&mut overlay, 0, CLUSTER), expected);
    }

    #[test]
    fn whole_sub_block_writes_read_nothing_from_the_base() {
        let (base, read_bytes) = counting_base(pattern(MIB as usize, 5), None);
        let mut overlay =
            Udisk::create(Vec::new(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap();
        overlay
            .write_at(SUB_BLOCK as u64, &pattern(5 * SUB_BLOCK, 1))
            .unwrap();
        assert_eq!(read_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn zeros_hide_the_base_without_storing_data() {
        let (base, _) = counting_base(vec![0xff; MIB as usize], None);
        let mut overlay =
            Udisk::create(Vec::new(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap();
        overlay.write_at(0, &[0; SUB_BLOCK]).unwrap();
        // A partial zero write over the base is data: the rest of the sub-block is the base's.
        overlay.write_at(SUB_BLOCK as u64, &[0; 512]).unwrap();
        assert_eq!(read(&mut overlay, 0, SUB_BLOCK), vec![0; SUB_BLOCK]);
        let mut second = vec![0xff; SUB_BLOCK];
        second[..512].fill(0);
        assert_eq!(read(&mut overlay, SUB_BLOCK as u64, SUB_BLOCK), second);
        let usage = overlay.usage().unwrap();
        assert_eq!(
            (usage.zero_bytes, usage.data_bytes),
            (SUB_BLOCK as u64, SUB_BLOCK as u64)
        );
    }

    #[test]
    fn overlays_stack() {
        let raw: Box<dyn BlockBackend> = Box::new(MemDisk(vec![1; MIB as usize]));
        let mut middle =
            Udisk::create(Vec::new(), &UdiskOptions::new(MIB, BASE_ID), Some(raw)).unwrap();
        middle.write_at(0, &[2; CLUSTER]).unwrap();
        let mut top = Udisk::create(
            Vec::new(),
            &UdiskOptions::new(MIB, ID),
            Some(Box::new(middle)),
        )
        .unwrap();
        top.write_at(0, &[3; 512]).unwrap();
        assert_eq!(read(&mut top, 0, 512), vec![3; 512]);
        assert_eq!(read(&mut top, 512, 512), vec![2; 512]);
        assert_eq!(read(&mut top, CLUSTER as u64, 512), vec![1; 512]);
    }

    fn overlay_storage(base_len: usize, base_id: Option<[u8; 16]>) -> Shared {
        let (base, _) = counting_base(vec![0; base_len], base_id);
        let storage = Shared::default();
        drop(Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), Some(base)).unwrap());
        storage
    }

    #[test]
    fn opening_needs_the_right_base() {
        let storage = overlay_storage(MIB as usize, Some(BASE_ID));
        let open =
            |base: Option<Box<dyn BlockBackend>>| Udisk::open(storage.clone(), base).map(|_| ());
        assert_eq!(kind(open(None)), io::ErrorKind::InvalidInput);
        let (wrong_size, _) = counting_base(vec![0; MIB as usize + 512], Some(BASE_ID));
        assert_eq!(kind(open(Some(wrong_size))), io::ErrorKind::InvalidData);
        let (wrong_id, _) = counting_base(vec![0; MIB as usize], Some(ID));
        assert_eq!(kind(open(Some(wrong_id))), io::ErrorKind::InvalidData);
        let (no_id, _) = counting_base(vec![0; MIB as usize], None);
        assert_eq!(kind(open(Some(no_id))), io::ErrorKind::InvalidData);
        let (right, _) = counting_base(vec![0; MIB as usize], Some(BASE_ID));
        open(Some(right)).unwrap();
    }

    #[test]
    fn a_raw_base_is_checked_by_size_alone() {
        let storage = overlay_storage(4096, None);
        let (any, _) = counting_base(vec![0; 4096], Some(BASE_ID));
        Udisk::open(storage.clone(), Some(any)).unwrap();
        let (short, _) = counting_base(vec![0; 512], None);
        assert_eq!(
            kind(Udisk::open(storage, Some(short))),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn an_image_without_a_base_refuses_one() {
        let storage = Shared::default();
        drop(Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap());
        let (base, _) = counting_base(vec![0; 4096], None);
        assert_eq!(
            kind(Udisk::open(storage, Some(base))),
            io::ErrorKind::InvalidInput
        );
    }
}

mod corruption {
    use super::*;

    /// A valid 1 MiB image with one data cluster (cluster 1 of the disk, holding `pattern`).
    fn valid_image() -> Vec<u8> {
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(MIB, ID), None).unwrap();
        disk.write_at(CLUSTER as u64, &pattern(CLUSTER, 1)).unwrap();
        disk.flush().unwrap();
        drop(disk);
        storage.bytes()
    }

    fn reseal(bytes: &mut [u8]) {
        let crc = crate::kv::crc32(&bytes[..0xffc]);
        bytes[0xffc..0x1000].copy_from_slice(&crc.to_le_bytes());
    }

    fn open(bytes: Vec<u8>) -> io::Result<Udisk> {
        Udisk::open(bytes, None)
    }

    fn with_header(change: impl FnOnce(&mut [u8])) -> io::ErrorKind {
        let mut bytes = valid_image();
        change(&mut bytes);
        reseal(&mut bytes);
        kind(open(bytes))
    }

    #[test]
    fn the_valid_image_opens() {
        let mut disk = open(valid_image()).unwrap();
        assert_eq!(
            read(&mut disk, CLUSTER as u64, CLUSTER),
            pattern(CLUSTER, 1)
        );
    }

    #[test]
    fn bad_headers_are_refused() {
        use io::ErrorKind::{InvalidData, Unsupported};
        let mut bytes = valid_image();
        bytes[100] ^= 1;
        assert_eq!(kind(open(bytes)), InvalidData, "checksum");
        assert_eq!(with_header(|h| h[0] = b'X'), InvalidData, "magic");
        assert_eq!(with_header(|h| h[0x08] = 2), Unsupported, "version");
        assert_eq!(with_header(|h| h[0x0d] = 0x20), InvalidData, "header size");
        assert_eq!(with_header(|h| h[0x10] = 2), Unsupported, "unknown feature");
        assert_eq!(
            with_header(|h| h[0x20] = 1),
            InvalidData,
            "size not a sector multiple"
        );
        assert_eq!(
            with_header(|h| h[0x20..0x28].fill(0)),
            InvalidData,
            "size zero"
        );
        assert_eq!(
            with_header(|h| h[0x28] = 15),
            InvalidData,
            "cluster shift too small"
        );
        assert_eq!(
            with_header(|h| h[0x28] = 25),
            InvalidData,
            "cluster shift too large"
        );
        assert_eq!(
            with_header(|h| h[0x28] = 17),
            InvalidData,
            "entries no longer match"
        );
        assert_eq!(with_header(|h| h[0x30] = 1), InvalidData, "unaligned map");
        assert_eq!(
            with_header(|h| h[0x31] = 0),
            InvalidData,
            "map inside the header"
        );
        assert_eq!(
            with_header(|h| h[0x38] = 17),
            InvalidData,
            "wrong entry count"
        );
        assert_eq!(
            with_header(|h| h[0x30..0x38].copy_from_slice(&(u64::MAX - 4095).to_le_bytes())),
            InvalidData,
            "map offset overflows"
        );
        assert_eq!(
            with_header(|h| h[0x60] = 1),
            InvalidData,
            "base fields without the feature"
        );
        assert_eq!(
            with_header(|h| h[0x68] = 1),
            InvalidData,
            "a locator without the feature"
        );
        assert_eq!(
            with_header(|h| {
                h[0x10] = 1;
                h[0x68] = 2;
                h[0x80] = 0xff;
                h[0x81] = 0xfe;
            }),
            InvalidData,
            "locator not UTF-8"
        );
        assert_eq!(
            with_header(|h| h[0x10] = 1),
            InvalidData,
            "an overlay of a size-0 base"
        );
        assert_eq!(
            with_header(|h| h[0x68..0x6a].copy_from_slice(&4000u16.to_le_bytes())),
            InvalidData,
            "locator runs into the checksum"
        );
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let bytes = valid_image();
        assert_eq!(
            kind(open(bytes[..4096 + 8].to_vec())),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            kind(open(bytes[..100].to_vec())),
            io::ErrorKind::InvalidData
        );
    }

    /// Replaces map entry `index` and reads cluster `index` of the disk.
    fn read_with_entry(index: usize, entry: u64) -> io::ErrorKind {
        let mut bytes = valid_image();
        bytes[4096 + index * 8..4096 + index * 8 + 8].copy_from_slice(&entry.to_le_bytes());
        let mut disk = open(bytes).unwrap();
        let mut buf = vec![0; CLUSTER];
        kind(disk.read_at(index as u64 * CLUSTER as u64, &mut buf))
    }

    #[test]
    fn bad_map_entries_are_refused_when_read() {
        use io::ErrorKind::InvalidData;
        let data = |cluster: u64, mask: u64| cluster | mask << 32;
        assert_eq!(
            read_with_entry(1, data(1, 1) | 1 << 48),
            InvalidData,
            "data and zero"
        );
        assert_eq!(
            read_with_entry(1, data(0, 1)),
            InvalidData,
            "data without a cluster"
        );
        assert_eq!(
            read_with_entry(0, data(0x9999, 1)),
            InvalidData,
            "a cluster past the file"
        );
        assert_eq!(
            read_with_entry(0, data(0x9999, 0)),
            InvalidData,
            "even with no data in it"
        );
    }

    #[test]
    fn sub_blocks_past_the_end_of_the_disk_are_refused() {
        let storage = Shared::default();
        drop(
            Udisk::create(
                storage.clone(),
                &UdiskOptions::new(CLUSTER as u64 + 8192, ID),
                None,
            )
            .unwrap(),
        );
        let mut bytes = storage.bytes();
        // Cluster 1 has two live sub-blocks; mark the third zero.
        bytes[4096 + 8..4096 + 16].copy_from_slice(&(1u64 << (48 + 2)).to_le_bytes());
        let mut disk = open(bytes).unwrap();
        assert_eq!(
            kind(disk.read_at(CLUSTER as u64, &mut [0; 512])),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn a_data_cluster_inside_the_map_is_refused() {
        // A 16 GiB disk has a 2 MiB map, so clusters 0 to 32 of the file are header and map.
        let storage = Shared::default();
        drop(Udisk::create(storage.clone(), &UdiskOptions::new(16 << 30, ID), None).unwrap());
        let mut bytes = storage.bytes();
        bytes.resize(bytes.len() + CLUSTER, 0);
        bytes[4096..4104].copy_from_slice(&(5 | 1u64 << 32).to_le_bytes());
        let mut disk = open(bytes).unwrap();
        assert_eq!(
            kind(disk.read_at(0, &mut [0; 512])),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn random_damage_never_panics() {
        let original = valid_image();
        let mut random = Lcg(42);
        for _ in 0..300 {
            let mut bytes = original.clone();
            for _ in 0..1 + random.below(4) {
                // Mostly the header and the map, where damage changes how the file is read.
                let limit = if random.below(4) == 0 {
                    bytes.len()
                } else {
                    4096 + 16 * 8
                };
                let at = random.below(limit as u64) as usize;
                bytes[at] ^= 1 << random.below(8);
            }
            if random.below(2) == 0 {
                reseal(&mut bytes);
            }
            if let Ok(mut disk) = open(bytes) {
                let len = disk.len() as usize;
                let mut buf = vec![0; len];
                let _ = disk.read_at(0, &mut buf);
                let _ = disk.write_at(0, &buf[..len.min(3 * CLUSTER)]);
                let _ = disk.flush();
                let _ = disk.usage();
            }
        }
    }
}

mod remote {
    use super::*;

    /// The (offset, length) of every range a remote file was asked for.
    type RequestLog = Arc<Mutex<Vec<(u64, usize)>>>;

    /// A remote file that records every range it was asked for.
    fn remote(bytes: Vec<u8>) -> (RemoteImage, RequestLog) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        let len = bytes.len() as u64;
        let image = RemoteImage::new(len, move |offset, buf: &mut [u8]| {
            log.lock().unwrap().push((offset, buf.len()));
            let range = checked_range(bytes.len() as u64, offset, buf.len())?;
            buf.copy_from_slice(&bytes[range]);
            Ok(())
        });
        (image, requests)
    }

    fn packed_image(size: u64) -> (Vec<u8>, Vec<u8>) {
        let contents = pattern(size as usize, 3);
        let storage = Shared::default();
        let mut disk = Udisk::create(storage.clone(), &UdiskOptions::new(size, ID), None).unwrap();
        disk.write_at(0, &contents).unwrap();
        disk.flush().unwrap();
        drop(disk);
        (storage.bytes(), contents)
    }

    #[test]
    fn opening_and_a_first_read_take_one_request() {
        let (file, contents) = packed_image(4 * MIB);
        let (storage, requests) = remote(file);
        let mut disk = Udisk::open(storage, None).unwrap();
        assert_eq!(read(&mut disk, 0, 4096), contents[..4096]);
        assert_eq!(
            *requests.lock().unwrap(),
            vec![(0, REMOTE_CHUNK_BYTES as usize)]
        );
    }

    #[test]
    fn reads_are_served_from_the_cache() {
        let (file, contents) = packed_image(4 * MIB);
        let (storage, requests) = remote(file);
        let mut disk = Udisk::open(storage, None).unwrap();
        for _ in 0..3 {
            assert_eq!(read(&mut disk, 70_000, 9000), contents[70_000..79_000]);
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn sequential_reads_fetch_ahead() {
        let (file, contents) = packed_image(8 * MIB);
        let file_len = file.len() as u64;
        let (storage, requests) = remote(file);
        let mut disk = Udisk::open(storage, None).unwrap();
        assert_eq!(read(&mut disk, 0, 8 * MIB as usize), contents);
        let requests = requests.lock().unwrap();
        let fetched: u64 = requests.iter().map(|(_, len)| *len as u64).sum();
        assert_eq!(fetched, file_len, "every byte fetched once");
        assert!(
            requests.len() <= 6,
            "{} requests: {requests:?}",
            requests.len()
        );
    }

    #[test]
    fn the_cache_stays_within_its_size() {
        let len = 40 * REMOTE_CHUNK_BYTES + 100;
        let bytes = pattern(len as usize, 1);
        let (storage, requests) = remote(bytes.clone());
        let mut storage = storage.with_cached_chunks(10);
        let mut buf = [0u8; 16];
        // Every other chunk: never sequential, so one chunk per miss.
        for index in (0..40).step_by(2) {
            storage
                .read_at(index * REMOTE_CHUNK_BYTES + 5, &mut buf)
                .unwrap();
        }
        let before = requests.lock().unwrap().len();
        for index in (20..40).step_by(2) {
            storage
                .read_at(index * REMOTE_CHUNK_BYTES + 5, &mut buf)
                .unwrap();
        }
        assert_eq!(
            requests.lock().unwrap().len(),
            before,
            "the ten newest are cached"
        );
        storage.read_at(5, &mut buf).unwrap();
        assert_eq!(
            requests.lock().unwrap().len(),
            before + 1,
            "chunk 0 was evicted"
        );
        storage.read_at(len - 16, &mut buf).unwrap();
        assert_eq!(buf, bytes[len as usize - 16..]);
    }

    #[test]
    fn a_read_spanning_chunks_and_the_short_last_chunk() {
        let len = 2 * REMOTE_CHUNK_BYTES + 7;
        let bytes = pattern(len as usize, 2);
        let (mut storage, _) = remote(bytes.clone());
        let mut buf = vec![0; (REMOTE_CHUNK_BYTES + 50) as usize];
        storage.read_at(REMOTE_CHUNK_BYTES - 43, &mut buf).unwrap();
        assert_eq!(buf, bytes[(REMOTE_CHUNK_BYTES - 43) as usize..]);
        assert!(storage.read_at(len - 1, &mut [0; 2]).is_err());
        assert_eq!(
            kind(storage.write_at(0, &[1])),
            io::ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn a_failed_fetch_is_an_error_and_is_not_cached() {
        let mut fail = true;
        let mut storage = RemoteImage::new(4096, move |_, buf: &mut [u8]| {
            if core::mem::replace(&mut fail, false) {
                return Err(io::Error::other("network down"));
            }
            buf.fill(1);
            Ok(())
        });
        let mut buf = [0u8; 4];
        assert!(storage.read_at(0, &mut buf).is_err());
        storage.read_at(0, &mut buf).unwrap();
        assert_eq!(buf, [1; 4]);
    }
}

mod raw {
    use super::*;
    use crate::disk::RawImage;

    #[test]
    fn a_raw_image_reads_and_writes_its_storage_in_place() {
        let storage = Shared::with_bytes(vec![0; 4096]);
        let mut raw = RawImage::new(storage.clone());
        raw.write_at(10, &[1, 2]).unwrap();
        assert_eq!(&storage.bytes()[10..12], &[1, 2]);
        assert_eq!(
            kind(raw.write_at(4095, &[1, 2])),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(storage.bytes().len(), 4096, "never grows");
    }

    #[test]
    fn a_raw_image_over_read_only_storage_is_read_only() {
        let mut raw = RawImage::new(Shared::with_bytes(vec![3; 512]).as_read_only());
        assert!(raw.read_only());
        assert_eq!(kind(raw.write_at(0, &[1])), io::ErrorKind::PermissionDenied);
        assert_eq!(read(&mut raw, 0, 512), vec![3; 512]);
        raw.flush().unwrap();
    }

    #[test]
    fn vec_storage_grows_on_writes_past_its_end() {
        let mut storage: Vec<u8> = Vec::new();
        ImageStorage::write_at(&mut storage, 10, &[5]).unwrap();
        assert_eq!(storage, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]);
    }
}
