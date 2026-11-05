use core::sync::atomic::{AtomicU64, Ordering, fence};

use crate::machine::HostClockFn;
use crate::sync::{Arc, Mutex};

/// The most guest time that passes between two reads of the clock. A longer gap means the host
/// was not running the machine: Chrome froze the tab's workers, the laptop slept, a disk read
/// from a URL blocked the thread. Every wait in the emulator is a millisecond or two (`wait_on_host`,
/// `vcpus::MAX_SLEEP`, the page's idle hook), so a healthy machine reads the clock far more often
/// than this. Passing a stall on whole makes a busy Linux guest report a soft lockup or an RCU
/// stall; QEMU likewise stops the guest clock while a VM is stopped.
pub(crate) const MAX_HOST_GAP_NS: u64 = 100_000_000;
/// How stale the timeline's last host time may get before a read refreshes it. Must stay well
/// under `MAX_HOST_GAP_NS`: that margin is what keeps guest time monotonic across a stall
/// (`Timeline::advance`).
const REFRESH_NS: u64 = 1_000_000;

/// The host clock the guest follows in realtime mode, zeroed when the machine starts so guest
/// time begins at 0 as the virtual clock does, with host stalls longer than `MAX_HOST_GAP_NS`
/// cut down to that. Clones (one per vCPU thread) share one timeline, so every vCPU skips the
/// same stall by the same amount and the TSCs stay in sync.
#[derive(Clone)]
pub(crate) struct WallClock {
    read: HostClockFn,
    origin_ns: u64,
    timeline: Arc<Timeline>,
}

/// A seqlock over the host time last seen and the host time skipped so far: readers take
/// neither a lock nor a shared cache line for writing, except to refresh `last_host_ns` once a
/// millisecond.
struct Timeline {
    /// Odd while `advance` is changing the two below.
    sequence: AtomicU64,
    last_host_ns: AtomicU64,
    skipped_ns: AtomicU64,
    writer: Mutex<()>,
}

impl WallClock {
    pub(crate) fn new(read: HostClockFn) -> Self {
        let origin_ns = read();
        WallClock {
            read,
            origin_ns,
            timeline: Arc::new(Timeline {
                sequence: AtomicU64::new(0),
                last_host_ns: AtomicU64::new(origin_ns),
                skipped_ns: AtomicU64::new(0),
                writer: Mutex::new(()),
            }),
        }
    }

    pub(crate) fn elapsed_ns(&self) -> u64 {
        let timeline = &*self.timeline;
        loop {
            let sequence = timeline.sequence.load(Ordering::Acquire);
            if sequence & 1 != 0 {
                core::hint::spin_loop();
                continue;
            }
            // The host is read inside the sequence window: a sample from before a stall must
            // not meet the skip that stall added, or this read lands after later ones.
            let host = (self.read)();
            let last = timeline.last_host_ns.load(Ordering::Relaxed);
            let skipped = timeline.skipped_ns.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if timeline.sequence.load(Ordering::Relaxed) != sequence {
                continue;
            }
            if host.saturating_sub(last) <= REFRESH_NS {
                return host.saturating_sub(self.origin_ns).saturating_sub(skipped);
            }
            timeline.advance(host);
        }
    }
}

impl Timeline {
    /// Moves `last_host_ns` up to `host`, skipping what a gap has beyond `MAX_HOST_GAP_NS`.
    ///
    /// Guest time stays monotonic: a read that returned without coming here had a host time at
    /// most `REFRESH_NS` past `last`, and after a stall guest time resumes at `last +
    /// MAX_HOST_GAP_NS`, later than all of those.
    fn advance(&self, host: u64) {
        let _writer = self.writer.lock().unwrap();
        let last = self.last_host_ns.load(Ordering::Relaxed);
        if host <= last {
            // Another reader got here first with a later sample.
            return;
        }
        let skip = (host - last).saturating_sub(MAX_HOST_GAP_NS);
        let sequence = self.sequence.load(Ordering::Relaxed);
        self.sequence.store(sequence + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        if skip > 0 {
            let skipped = self.skipped_ns.load(Ordering::Relaxed);
            self.skipped_ns.store(skipped + skip, Ordering::Relaxed);
        }
        self.last_host_ns.store(host, Ordering::Relaxed);
        self.sequence.store(sequence + 2, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};

    const HOST_EPOCH_NS: u64 = 5_000_000_000;
    const MS: u64 = 1_000_000;
    /// What a frozen browser tab or a sleeping laptop takes away from the emulator.
    const STALL_NS: u64 = 30_000_000_000;

    fn manual_clock() -> (Arc<AtomicU64>, WallClock) {
        let now = Arc::new(AtomicU64::new(HOST_EPOCH_NS));
        let reader = now.clone();
        let clock = WallClock::new(Arc::new(move || reader.load(Ordering::SeqCst)));
        (now, clock)
    }

    #[test]
    fn host_time_passes_through_while_the_clock_is_read_often() {
        let (host, clock) = manual_clock();
        let mut expected = 0;
        for step in [MS, 2 * MS, MAX_HOST_GAP_NS, 3 * MS] {
            host.fetch_add(step, Ordering::SeqCst);
            expected += step;
            assert_eq!(clock.elapsed_ns(), expected);
        }
    }

    /// A busy Linux guest whose host stops running it for 30 s and then resumes sees its
    /// watchdog hrtimer fire 30 s after the last time the watchdog thread ran, and reports
    /// "BUG: soft lockup - CPU#0 stuck for 30s!" (Chrome freezes the workers of a background
    /// tab; QEMU stops the guest clock while a VM is stopped).
    #[test]
    fn a_host_stall_moves_guest_time_by_at_most_the_gap_limit() {
        // Steps longer than `REFRESH_NS`, so each read records its host time; a stall right
        // after a shorter step is measured from the read before it and skips that much more.
        const STEP_NS: u64 = 2 * REFRESH_NS;
        let (host, clock) = manual_clock();
        host.fetch_add(STEP_NS, Ordering::SeqCst);
        assert_eq!(clock.elapsed_ns(), STEP_NS);
        host.fetch_add(STALL_NS, Ordering::SeqCst);
        assert_eq!(clock.elapsed_ns(), STEP_NS + MAX_HOST_GAP_NS);
        host.fetch_add(STEP_NS, Ordering::SeqCst);
        assert_eq!(clock.elapsed_ns(), 2 * STEP_NS + MAX_HOST_GAP_NS);
    }

    #[test]
    fn every_vcpu_sees_the_stall_skipped_once() {
        let (host, clock) = manual_clock();
        let other_vcpu = clock.clone();
        host.fetch_add(STALL_NS, Ordering::SeqCst);
        assert_eq!(clock.elapsed_ns(), MAX_HOST_GAP_NS);
        assert_eq!(other_vcpu.elapsed_ns(), MAX_HOST_GAP_NS);
        host.fetch_add(MS, Ordering::SeqCst);
        assert_eq!(other_vcpu.elapsed_ns(), MAX_HOST_GAP_NS + MS);
        assert_eq!(clock.elapsed_ns(), MAX_HOST_GAP_NS + MS);
    }

    /// vCPU threads reading the shared clock while the host stalls now and then: a reader whose
    /// host sample predates a stall must not combine it with the time the stall skipped, or
    /// guest time runs backwards (Linux then finds the TSC unsynchronized). Only a native
    /// aarch64 host catches an ordering bug here (`docs/plans/22-smp-threads.md`).
    #[test]
    fn guest_time_never_runs_backwards_on_any_thread_across_stalls() {
        const READERS: usize = 4;
        const READS: usize = 200_000;
        const STALL_EVERY: u64 = 1_000;
        let host = Arc::new(AtomicU64::new(HOST_EPOCH_NS));
        let reads = Arc::new(AtomicU64::new(0));
        let reader = host.clone();
        let counter = reads.clone();
        // Every read moves the host clock on by a microsecond, and every so often by a stall.
        let clock = WallClock::new(Arc::new(move || {
            let read = counter.fetch_add(1, Ordering::SeqCst);
            let step = if read % STALL_EVERY == STALL_EVERY - 1 {
                STALL_NS
            } else {
                1_000
            };
            reader.fetch_add(step, Ordering::SeqCst) + step
        }));
        let threads: Vec<_> = (0..READERS)
            .map(|_| {
                let clock = clock.clone();
                std::thread::spawn(move || {
                    let mut previous = 0;
                    for _ in 0..READS {
                        let now = clock.elapsed_ns();
                        assert!(now >= previous, "guest time went from {previous} to {now}");
                        previous = now;
                    }
                    previous
                })
            })
            .collect();
        let latest = threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .max()
            .unwrap();
        let stalls = reads.load(Ordering::SeqCst) / STALL_EVERY;
        assert!(
            latest <= (stalls + 1) * MAX_HOST_GAP_NS + reads.load(Ordering::SeqCst) * 1_000,
            "{latest} ns of guest time for {stalls} stalls"
        );
    }
}
