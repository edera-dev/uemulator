use super::guestmem::GuestMem;
use super::virtio::{DEVICE_TYPE_RNG, Queue, VirtioDev};
use crate::cpu::xsave::splitmix64;
#[cfg(freestanding)]
use crate::prelude::*;
use crate::snapshot::Snapshot;
use crate::sync::Arc;

/// The only queue: the guest posts empty buffers, the device fills them.
const REQUESTQ: usize = 0;
const QUEUE_COUNT: usize = 1;

/// Most bytes put into one buffer. The spec lets a device fill less than was offered; a guest
/// with a huge buffer asking for entropy gets it in pieces, and no single kick runs long.
const MAX_BYTES_PER_REQUEST: usize = 1 << 16;

/// Fills a buffer with bytes from the host's entropy source.
pub(crate) type EntropyFn = Arc<dyn Fn(&mut [u8]) + Send + Sync>;

/// Where a virtio-rng device gets its bytes.
#[derive(Clone)]
pub(crate) enum RngSource {
    /// A SplitMix64 stream from this seed, for `--verify`: the lockstep twin builds its own device
    /// from the same seed and must hand the guest the same bytes.
    Seeded(u64),
    /// The host's entropy; each device calls it for every request.
    Host(EntropyFn),
}

/// The host's entropy where the host has one the library can reach without dependencies:
/// `/dev/urandom` on Unix, else the OS-seeded keys of `std`'s `RandomState` run as a PRF over a
/// counter. Freestanding hosts (WebAssembly, UEFI) bring their own through
/// `MachineConfig::entropy`.
pub(crate) fn host_entropy() -> Option<EntropyFn> {
    #[cfg(not(freestanding))]
    {
        use std::hash::{BuildHasher, Hasher};
        use std::io::Read;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicU64, Ordering};

        let urandom = if cfg!(unix) {
            std::fs::File::open("/dev/urandom").ok().map(Mutex::new)
        } else {
            None
        };
        let keys = std::collections::hash_map::RandomState::new();
        let counter = AtomicU64::new(0);
        Some(Arc::new(move |bytes: &mut [u8]| {
            if let Some(file) = &urandom
                && file.lock().unwrap().read_exact(bytes).is_ok()
            {
                return;
            }
            for chunk in bytes.chunks_mut(8) {
                let mut hasher = keys.build_hasher();
                hasher.write_u64(counter.fetch_add(1, Ordering::Relaxed));
                chunk.copy_from_slice(&hasher.finish().to_le_bytes()[..chunk.len()]);
            }
        }))
    }
    #[cfg(freestanding)]
    {
        None
    }
}

#[derive(Snapshot)]
pub(crate) struct VirtioRng {
    #[snapshot(skip)]
    source: RngSource,
    /// State of the `Seeded` stream; unused for `Host`.
    state: u64,
}

impl VirtioRng {
    pub fn new(source: RngSource) -> Self {
        let state = match &source {
            RngSource::Seeded(seed) => *seed,
            RngSource::Host(_) => 0,
        };
        VirtioRng { source, state }
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        match &self.source {
            RngSource::Host(entropy) => entropy(bytes),
            RngSource::Seeded(_) => {
                for chunk in bytes.chunks_mut(8) {
                    let word = splitmix64(&mut self.state).to_le_bytes();
                    chunk.copy_from_slice(&word[..chunk.len()]);
                }
            }
        }
    }
}

impl VirtioDev for VirtioRng {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_RNG
    }

    /// No device-specific features, and no config space.
    fn features(&self) -> u64 {
        0
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, _off: usize) -> u8 {
        0
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if queue != REQUESTQ {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[REQUESTQ].pop(mem) {
            let mut bytes = vec![0u8; chain.writable().min(MAX_BYTES_PER_REQUEST)];
            self.fill(&mut bytes);
            let written = chain.write_at(mem, 0, &bytes);
            queues[REQUESTQ].push_used(mem, chain.head, written as u32);
            any = true;
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::super::virtio::testutil::*;
    use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
    use super::*;
    use crate::dev::guestmem::test_mem;
    use std::sync::Mutex;

    const QUEUE_SIZE: u16 = 8;
    const DESC: u64 = 0x1000;
    const AVAIL: u64 = 0x2000;
    const USED: u64 = 0x3000;
    const BUF: u64 = 0x8000;
    const RAM_BYTES: usize = 0x4_0000;
    const SEED: u64 = 0x1234_5678;

    struct Rig {
        dev: VirtioRng,
        queues: Vec<Queue>,
        ram: Vec<u8>,
    }

    impl Rig {
        fn new(source: RngSource) -> Rig {
            let mut ram = vec![0u8; RAM_BYTES];
            let queues = vec![setup_queue(&mut ram, DESC, AVAIL, USED, QUEUE_SIZE)];
            Rig {
                dev: VirtioRng::new(source),
                queues,
                ram,
            }
        }

        fn kick(&mut self) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.kick(REQUESTQ, &mut self.queues, &mut mem)
        }

        fn post(&mut self, head: u16, addr: u64, len: u32) {
            put_desc(&mut self.ram, DESC, head, addr, len, DESC_F_WRITE, 0);
            post(&mut self.ram, AVAIL, QUEUE_SIZE, head);
        }

        fn used_len(&self, slot: u16) -> u32 {
            let at = (USED + 4 + 8 * slot as u64 + 4) as usize;
            u32::from_le_bytes(self.ram[at..at + 4].try_into().unwrap())
        }
    }

    fn seeded() -> RngSource {
        RngSource::Seeded(SEED)
    }

    #[test]
    fn identity_is_a_virtio_entropy_source_with_one_queue_and_no_features() {
        let rig = Rig::new(seeded());
        assert_eq!(rig.dev.device_type(), 4);
        assert_eq!(rig.dev.features(), 0);
        assert_eq!(rig.dev.num_queues(), 1);
        assert_eq!(rig.dev.config_read(0), 0);
    }

    #[test]
    fn a_request_is_filled_completely_and_reported_used() {
        let mut rig = Rig::new(seeded());
        rig.post(0, BUF, 24);
        assert!(rig.kick());
        assert_eq!(rig.used_len(0), 24);
        assert_ne!(&rig.ram[BUF as usize..BUF as usize + 24], &[0; 24]);
        assert_eq!(rig.ram[BUF as usize + 24], 0, "nothing past the buffer");
    }

    #[test]
    fn a_kick_with_no_buffers_completes_nothing() {
        let mut rig = Rig::new(seeded());
        assert!(!rig.kick());
    }

    #[test]
    fn a_buffer_split_over_descriptors_is_filled_across_them() {
        let mut rig = Rig::new(seeded());
        put_desc(&mut rig.ram, DESC, 0, BUF, 5, DESC_F_WRITE | DESC_F_NEXT, 1);
        put_desc(&mut rig.ram, DESC, 1, BUF + 0x1000, 11, DESC_F_WRITE, 0);
        post(&mut rig.ram, AVAIL, QUEUE_SIZE, 0);
        assert!(rig.kick());
        assert_eq!(rig.used_len(0), 16);
        assert_ne!(&rig.ram[BUF as usize..BUF as usize + 5], &[0; 5]);
        let second = BUF as usize + 0x1000;
        assert_ne!(&rig.ram[second..second + 11], &[0; 11]);
    }

    #[test]
    fn one_request_is_capped_so_a_huge_buffer_gets_a_partial_fill() {
        let mut rig = Rig::new(seeded());
        let offered = (MAX_BYTES_PER_REQUEST * 2) as u32;
        rig.post(0, BUF, offered);
        assert!(rig.kick());
        assert_eq!(rig.used_len(0), MAX_BYTES_PER_REQUEST as u32);
    }

    #[test]
    fn a_read_only_buffer_gets_nothing() {
        let mut rig = Rig::new(seeded());
        put_desc(&mut rig.ram, DESC, 0, BUF, 16, 0, 0);
        post(&mut rig.ram, AVAIL, QUEUE_SIZE, 0);
        assert!(rig.kick());
        assert_eq!(rig.used_len(0), 0);
        assert_eq!(&rig.ram[BUF as usize..BUF as usize + 16], &[0; 16]);
    }

    #[test]
    fn two_seeded_devices_hand_out_the_same_stream() {
        let mut first = Rig::new(seeded());
        let mut second = Rig::new(seeded());
        for rig in [&mut first, &mut second] {
            rig.post(0, BUF, 100);
            rig.kick();
        }
        assert_eq!(
            first.ram[BUF as usize..BUF as usize + 100],
            second.ram[BUF as usize..BUF as usize + 100]
        );
        let mut other = Rig::new(RngSource::Seeded(SEED + 1));
        other.post(0, BUF, 100);
        other.kick();
        assert_ne!(
            first.ram[BUF as usize..BUF as usize + 100],
            other.ram[BUF as usize..BUF as usize + 100]
        );
    }

    #[test]
    fn successive_requests_continue_the_stream_rather_than_repeating_it() {
        let mut rig = Rig::new(seeded());
        rig.post(0, BUF, 16);
        rig.kick();
        let first = rig.ram[BUF as usize..BUF as usize + 16].to_vec();
        rig.post(1, BUF, 16);
        rig.kick();
        assert_ne!(first, rig.ram[BUF as usize..BUF as usize + 16]);
    }

    #[test]
    fn the_host_source_gives_different_bytes_each_time() {
        let entropy = host_entropy().expect("std hosts have entropy");
        let (mut first, mut second) = ([0u8; 32], [0u8; 32]);
        entropy(&mut first);
        entropy(&mut second);
        assert_ne!(first, [0; 32]);
        assert_ne!(first, second);
    }

    #[test]
    fn a_host_source_supplies_every_byte() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let record = calls.clone();
        let source = RngSource::Host(Arc::new(move |bytes: &mut [u8]| {
            record.lock().unwrap().push(bytes.len());
            bytes.fill(0xa5);
        }));
        let mut rig = Rig::new(source);
        rig.post(0, BUF, 33);
        assert!(rig.kick());
        assert_eq!(&rig.ram[BUF as usize..BUF as usize + 33], &[0xa5; 33]);
        assert_eq!(*calls.lock().unwrap(), vec![33]);
    }
}
