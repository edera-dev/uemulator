//! One host thread per vCPU. The design is in `docs/plans/22-smp-threads.md`; this is the machinery.
//!
//! Every vCPU is a `Cpu` owned by its own thread, with a private soft TLB, decode cache and JIT.
//! What they share is RAM and the code-page flags (`ram.rs`), the devices behind the device lock
//! (`dev/handle.rs`), and a `VcpuLinks`: one atomic word per vCPU that others poke to get its
//! attention, an exclusive section that stops the world for the rare operation that needs it, and
//! a log of pages whose translated code is stale.
//!
//! A vCPU only ever looks at its own `attention` word, at block boundaries (`before_step`), so
//! no thread touches another's `Cpu` except inside an exclusive section, when every other vCPU is
//! parked at such a boundary or halted.
//!
//! Guest time is the host clock (`ClockMode::Realtime`), so the TSC agrees between vCPUs; devices
//! are ticked by one more thread, the device thread, which runs on the thread that started the
//! machine.

use super::*;
use crate::sync::{Arc, Mutex};
use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize};
use core::time::Duration;
#[cfg(not(freestanding))]
use std::thread::Thread;

/// A deliverable interrupt is waiting in the local APIC (or, for vCPU 0, in the PIC). The device
/// model recomputes the three device bits every time the device lock is released.
pub(crate) const ATTENTION_IRQ: u32 = 1 << 0;
pub(crate) const ATTENTION_NMI: u32 = 1 << 1;
/// An INIT or start-up IPI is waiting.
pub(crate) const ATTENTION_START: u32 = 1 << 2;
pub(crate) const ATTENTION_DEVICE_MASK: u32 = ATTENTION_IRQ | ATTENTION_NMI | ATTENTION_START;
/// Another vCPU wants the world stopped (`Cpu::with_exclusive`).
const ATTENTION_EXCLUSIVE: u32 = 1 << 3;
/// Translated code for some pages is stale (`VcpuLinks::publish_invalidation`).
const ATTENTION_INVALIDATE: u32 = 1 << 4;
/// The machine is ending.
const ATTENTION_EXIT: u32 = 1 << 5;

/// How long a halted vCPU or the device thread sleeps at most. It bounds how late the stop flag,
/// host console input and anything that missed a wake-up gets noticed.
const MAX_SLEEP: Duration = Duration::from_millis(1);
/// How long a halted vCPU sleeps before looking again when nothing it waits for needs polling;
/// every real wake-up comes with an explicit one, so this is only a safety net.
const HALTED_SLEEP: Duration = Duration::from_millis(100);
/// Invalidations kept for vCPUs that have not looked yet; one that falls further behind throws
/// all its translations away instead.
const INVALIDATION_LOG_CAPACITY: usize = 4096;
/// Waiting iterations before a thread offers its time slice to others instead of just spinning.
const SPINS_BEFORE_YIELD: u32 = 64;

/// Lets one thread wake another that sleeps with a timeout. Hosted: `park`/`unpark`. A
/// freestanding host has no scheduler, so a wait is a spin and a wake is a no-op.
#[derive(Default)]
pub(crate) struct Waker {
    #[cfg(not(freestanding))]
    thread: Mutex<Option<Thread>>,
}

impl Waker {
    /// Makes the calling thread the one `wake` unparks.
    pub fn register_current(&self) {
        #[cfg(not(freestanding))]
        {
            *self.thread.lock().unwrap() = Some(std::thread::current());
        }
    }

    pub fn wake(&self) {
        #[cfg(not(freestanding))]
        if let Some(thread) = &*self.thread.lock().unwrap() {
            thread.unpark();
        }
    }

    /// Sleeps for up to `timeout`, or until woken. Callers re-check what they wait for.
    pub fn wait(timeout: Duration) {
        #[cfg(not(freestanding))]
        std::thread::park_timeout(timeout);
        #[cfg(freestanding)]
        {
            let _ = timeout;
            core::hint::spin_loop();
        }
    }
}

/// What one vCPU publishes about itself, and how others reach it.
pub(crate) struct VcpuLink {
    /// `ATTENTION_*` bits. Anyone sets them; the owning vCPU clears the non-device ones when it
    /// handles them, and the device model owns the device bits.
    pub attention: AtomicU32,
    /// The vCPU's `Cpu`, for the owner of an exclusive section. Null until its thread starts.
    cpu: AtomicPtr<Cpu>,
    /// Waiting in `wait_for_wake`.
    halted: AtomicBool,
    /// An interrupt could wake it from that wait: it is not waiting for a start-up IPI and has
    /// interrupts enabled. A machine whose vCPUs are all halted and none could wake ends.
    wakeable: AtomicBool,
    waker: Waker,
}

struct InvalidationLog {
    /// Sequence number of `pages[0]`.
    first_sequence: u64,
    pages: VecDeque<u32>,
}

impl InvalidationLog {
    fn next_sequence(&self) -> u64 {
        self.first_sequence + self.pages.len() as u64
    }
}

/// Everything the vCPU threads of one machine share besides RAM and the devices.
pub(crate) struct VcpuLinks {
    links: Vec<VcpuLink>,
    /// A vCPU wants the world stopped; the others park until it is false again.
    exclusive_requested: AtomicBool,
    /// Held by the vCPU running an exclusive section.
    exclusive_owner: AtomicBool,
    /// vCPUs executing guest code, or on their way to or from parking. Halted, parked and
    /// finished vCPUs are not counted, so an exclusive section waits only for those that run.
    running: AtomicUsize,
    invalidations: Mutex<InvalidationLog>,
    exit: Mutex<Option<ExitReason>>,
    exiting: AtomicBool,
    finished: AtomicUsize,
    /// The device thread, for work a vCPU just handed the devices.
    device_waker: Waker,
    device_sleeping: AtomicBool,
}

impl VcpuLinks {
    pub fn new(count: usize) -> Arc<Self> {
        Arc::new(VcpuLinks {
            links: (0..count)
                .map(|_| VcpuLink {
                    attention: AtomicU32::new(0),
                    cpu: AtomicPtr::new(core::ptr::null_mut()),
                    halted: AtomicBool::new(false),
                    wakeable: AtomicBool::new(false),
                    waker: Waker::default(),
                })
                .collect(),
            exclusive_requested: AtomicBool::new(false),
            exclusive_owner: AtomicBool::new(false),
            running: AtomicUsize::new(count),
            invalidations: Mutex::new(InvalidationLog {
                first_sequence: 0,
                pages: VecDeque::new(),
            }),
            exit: Mutex::new(None),
            exiting: AtomicBool::new(false),
            finished: AtomicUsize::new(0),
            device_waker: Waker::default(),
            device_sleeping: AtomicBool::new(false),
        })
    }

    /// Replaces the device bits of vCPU `index` with `bits` and wakes it if one is new. Called
    /// from the device model under the device lock.
    pub fn publish_device_bits(&self, index: usize, bits: u32) {
        let link = &self.links[index];
        let mut current = link.attention.load(Ordering::Relaxed);
        loop {
            let updated = (current & !ATTENTION_DEVICE_MASK) | bits;
            if updated == current {
                return;
            }
            match link.attention.compare_exchange_weak(
                current,
                updated,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(seen) => current = seen,
            }
        }
        if bits & !(current & ATTENTION_DEVICE_MASK) != 0 {
            link.waker.wake();
        }
    }

    /// Wakes the device thread if it is asleep: a vCPU just did something that may need the
    /// devices' attention (a notify, a timer write).
    pub fn poke_device_thread(&self) {
        if self.device_sleeping.load(Ordering::Relaxed)
            && self.device_sleeping.swap(false, Ordering::AcqRel)
        {
            self.device_waker.wake();
        }
    }

    /// Records that the machine ends with `reason` (the first reason wins) and tells everyone.
    pub fn request_exit(&self, reason: ExitReason) {
        {
            let mut slot = self.exit.lock().unwrap();
            if slot.is_none() {
                *slot = Some(reason);
            }
        }
        self.exiting.store(true, Ordering::SeqCst);
        for link in &self.links {
            link.attention.fetch_or(ATTENTION_EXIT, Ordering::SeqCst);
            link.waker.wake();
        }
        self.device_waker.wake();
    }

    pub fn exiting(&self) -> bool {
        self.exiting.load(Ordering::Acquire)
    }

    pub fn exit_reason(&self) -> ExitReason {
        self.exit
            .lock()
            .unwrap()
            .clone()
            .unwrap_or(ExitReason::Stopped)
    }

    /// Tells every vCPU but `except` (which has dropped its own) that `page` held code that is
    /// stale, so they drop their decode caches. It leaves the page's code flag alone: whoever
    /// invalidated the page clears it, and for the JIT that has to happen under the cache's lock
    /// (`Jit::invalidate_page`); clearing it again here, outside it, would let a translation flag,
    /// decode and register a block in between and lose its stores to `code_written` for good.
    /// `except` is `None` for a write by a device.
    pub fn publish_invalidation(&self, except: Option<usize>, page: usize) {
        {
            let mut log = self.invalidations.lock().unwrap();
            if log.pages.len() == INVALIDATION_LOG_CAPACITY {
                log.pages.pop_front();
                log.first_sequence += 1;
            }
            log.pages.push_back(page as u32);
        }
        for (index, link) in self.links.iter().enumerate() {
            if Some(index) != except {
                link.attention
                    .fetch_or(ATTENTION_INVALIDATE, Ordering::SeqCst);
            }
        }
    }

    /// Every vCPU is halted and none could be woken by an interrupt: the guest is dead, which is
    /// how a `cli; hlt` poweroff ends.
    ///
    /// `wakeable` is kept by each vCPU's own thread, so it lags behind what the device model just
    /// published: a start-up IPI for a parked vCPU sets `ATTENTION_START` before that thread has
    /// looked at it, and a bootstrap processor that goes straight to `cli; hlt` after sending it
    /// would otherwise end the machine before the others ever run.
    pub fn all_dead(&self) -> bool {
        self.links.iter().all(|link| {
            link.halted.load(Ordering::SeqCst)
                && !link.wakeable.load(Ordering::SeqCst)
                && link.attention.load(Ordering::SeqCst) & (ATTENTION_NMI | ATTENTION_START) == 0
        })
    }

    pub fn all_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst) == self.links.len()
    }

    fn leave_running(&self) {
        self.running.fetch_sub(1, Ordering::SeqCst);
    }

    /// Becomes running again, waiting first for any exclusive section in progress to end. The
    /// increment comes before the check so the requester, which sets its flag and then reads the
    /// count, cannot miss both.
    fn enter_running(&self) {
        loop {
            self.running.fetch_add(1, Ordering::SeqCst);
            if !self.exclusive_requested.load(Ordering::SeqCst) {
                return;
            }
            self.running.fetch_sub(1, Ordering::SeqCst);
            let mut spins = 0u32;
            while self.exclusive_requested.load(Ordering::SeqCst) {
                spin_or_yield(&mut spins);
            }
        }
    }

    fn wait_while_exclusive(&self) {
        let mut spins = 0u32;
        while self.exclusive_requested.load(Ordering::SeqCst) {
            spin_or_yield(&mut spins);
        }
    }
}

fn spin_or_yield(spins: &mut u32) {
    *spins += 1;
    if *spins < SPINS_BEFORE_YIELD {
        core::hint::spin_loop();
    } else {
        crate::sync::yield_now();
    }
}

/// A vCPU's view of the shared machinery.
pub(crate) struct ThreadRuntime {
    pub links: Arc<VcpuLinks>,
    /// This vCPU's `attention` word, for the check every block boundary makes.
    attention: *const AtomicU32,
}

// `attention` points into `links`, which the runtime keeps alive.
unsafe impl Send for ThreadRuntime {}

impl Cpu {
    pub(crate) fn attach_threads(&mut self, links: Arc<VcpuLinks>) {
        let attention = &links.links[self.cpu_index as usize].attention as *const AtomicU32;
        self.threads = Some(ThreadRuntime { links, attention });
        self.ordered_memory = true;
        if let Some(jit) = &mut self.jit {
            jit.set_ordered_memory(true);
        }
    }

    fn links(&self) -> Arc<VcpuLinks> {
        self.threads
            .as_ref()
            .expect("a threaded vCPU")
            .links
            .clone()
    }

    /// The `ATTENTION_*` bits set for this vCPU right now.
    pub(super) fn attention(&self) -> u32 {
        match &self.threads {
            Some(runtime) => unsafe { (*runtime.attention).load(Ordering::Acquire) },
            None => 0,
        }
    }

    /// Per-vCPU caches a thread owns: the JIT and the decode cache the scheduler would
    /// otherwise lend it.
    pub(crate) fn give_private_caches(&mut self, jit: Option<Box<crate::jit::Jit>>) {
        self.jit = jit;
        if let Some(jit) = &mut self.jit {
            jit.set_ordered_memory(true);
        }
        if self.decode_cache.is_empty() {
            self.decode_cache = vec![
                DecodeCacheEntry {
                    gpa: 0,
                    generation: 0,
                    insn: Insn::default(),
                };
                DECODE_CACHE_ENTRIES
            ];
        }
    }

    /// The vCPU thread's body: runs until the machine ends, then reports.
    pub(crate) fn run_thread(&mut self) -> ExitReason {
        let links = self.links();
        let link = &links.links[self.cpu_index as usize];
        link.waker.register_current();
        link.cpu.store(self as *mut Cpu, Ordering::SeqCst);
        // Everyone is counted as running from the start, so an exclusive request waits for the
        // threads that have not got here yet; being here means being a parked-or-running vCPU.
        let reason = match self.run_slice() {
            Slice::Exit(reason) => reason,
            Slice::Yield | Slice::Halted => {
                ExitReason::Internal("a threaded vCPU asked for a scheduler".into())
            }
        };
        link.cpu.store(core::ptr::null_mut(), Ordering::SeqCst);
        links.leave_running();
        links.request_exit(reason.clone());
        reason
    }

    /// The per-block-boundary work of a threaded vCPU: the machine ending, an exclusive section
    /// to park for, stale translations to drop. `Some` ends this vCPU's run.
    pub(super) fn thread_attention(&mut self) -> Option<ExitReason> {
        let attention = self.attention();
        if attention & !ATTENTION_DEVICE_MASK == 0 {
            return None;
        }
        let links = self.links();
        if attention & ATTENTION_EXIT != 0 {
            return Some(links.exit_reason());
        }
        if attention & ATTENTION_EXCLUSIVE != 0 {
            self.park_for_exclusive(&links);
        }
        if attention & ATTENTION_INVALIDATE != 0 {
            self.drain_invalidations(&links);
        }
        None
    }

    fn clear_attention(&self, bits: u32) {
        if let Some(runtime) = &self.threads {
            unsafe { (*runtime.attention).fetch_and(!bits, Ordering::SeqCst) };
        }
    }

    fn park_for_exclusive(&mut self, links: &VcpuLinks) {
        // Cleared before parking: a request that arrives after this sets it again.
        self.clear_attention(ATTENTION_EXCLUSIVE);
        links.leave_running();
        links.wait_while_exclusive();
        links.enter_running();
    }

    /// Catches this vCPU up with the code other vCPUs and devices wrote since it last looked: its
    /// decode cache goes. The JIT needs nothing here: the cache is shared, so the writer dropped
    /// the blocks of the page for every vCPU at once (`Jit::invalidate_page`), and the
    /// jump-cache entries that named them fail the validity check.
    fn drain_invalidations(&mut self, links: &VcpuLinks) {
        self.clear_attention(ATTENTION_INVALIDATE);
        self.invalidation_seen = links.invalidations.lock().unwrap().next_sequence();
        self.decode_cache_generation = self.decode_cache_generation.wrapping_add(1);
    }

    /// Runs `f` with every other vCPU stopped: parked at a block boundary or halted. For what
    /// cannot be done to memory another vCPU may be using, or to another vCPU's state: flagging a
    /// page as code (which revokes write tags in every TLB), a locked access no host atomic
    /// covers. Never call it with the device lock held: the others could be waiting for it.
    pub(crate) fn with_exclusive<R>(&mut self, f: impl FnOnce(&mut Cpu) -> R) -> R {
        let Some(runtime) = &self.threads else {
            return f(self);
        };
        if self.in_exclusive {
            // Already the only vCPU running: asking again would wait for ourselves.
            return f(self);
        }
        let links = runtime.links.clone();
        // One exclusive section at a time. While waiting for the owner, stay parkable: the owner
        // is waiting for us.
        let mut spins = 0u32;
        while links
            .exclusive_owner
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            if links.exclusive_requested.load(Ordering::SeqCst) {
                self.park_for_exclusive(&links);
            }
            spin_or_yield(&mut spins);
        }
        links.exclusive_requested.store(true, Ordering::SeqCst);
        links.leave_running();
        for (index, link) in links.links.iter().enumerate() {
            if index != self.cpu_index as usize {
                link.attention
                    .fetch_or(ATTENTION_EXCLUSIVE, Ordering::SeqCst);
            }
        }
        let mut spins = 0u32;
        while links.running.load(Ordering::SeqCst) != 0 {
            spin_or_yield(&mut spins);
        }
        self.in_exclusive = true;
        let result = f(self);
        self.in_exclusive = false;
        links.exclusive_requested.store(false, Ordering::SeqCst);
        links.enter_running();
        links.exclusive_owner.store(false, Ordering::Release);
        result
    }

    /// Every vCPU's `Cpu` but this one. Only valid inside `with_exclusive`.
    fn for_each_other_cpu(&mut self, mut f: impl FnMut(&mut Cpu)) {
        let links = self.links();
        for (index, link) in links.links.iter().enumerate() {
            if index == self.cpu_index as usize {
                continue;
            }
            if let Some(other) = unsafe { link.cpu.load(Ordering::SeqCst).as_mut() } {
                f(other);
            }
        }
    }

    /// Flags `page` as holding translated code on behalf of all vCPUs. The flag is set first,
    /// then the write tags go, and the caller reads the guest bytes only afterwards: a store that
    /// ran before the revocation is visible to that read, and one that runs after it takes the
    /// slow path, sees the flag and invalidates.
    pub(super) fn flag_code_page_for_all(&mut self, page: usize) {
        if self.page_has_code.get(page) != 0 {
            return;
        }
        self.page_has_code.set(page, 1);
        fence(Ordering::SeqCst);
        self.revoke_write_tags_for(page);
        self.for_each_other_cpu(|other| other.revoke_write_tags_for(page));
    }

    /// A serializing instruction (`cpuid`) on a threaded vCPU: code another vCPU just wrote must
    /// be what runs next (SDM vol. 3, 8.1.3, cross-modifying code). Instructions already
    /// translated are only dropped at a block boundary, so ask the running block to end there.
    pub(super) fn serialize(&mut self) {
        if self.attention() & ATTENTION_INVALIDATE != 0 {
            self.block_exit_requested = true;
        }
    }

    /// Whether a halted vCPU has something to wake it, from the device bits and local state.
    fn wake_condition(&mut self, attention: u32) -> bool {
        // An exclusive section or stale translations do not wake a halted vCPU: it is not
        // running, and it deals with both before it executes anything (`thread_attention`).
        if attention & ATTENTION_EXIT != 0 {
            return true;
        }
        if attention & ATTENTION_START != 0 {
            return true;
        }
        if self.wait_for_sipi {
            return false;
        }
        if attention & ATTENTION_NMI != 0 {
            return true;
        }
        if attention & ATTENTION_IRQ != 0 && self.eflags & flags::IF != 0 && self.svm.gif {
            return true;
        }
        self.mwait_watch.is_some_and(|(line, contents)| {
            let line = line as usize;
            self.ram
                .get(line..line + MONITOR_WATCH_BYTES)
                .is_some_and(|now| now != contents)
        })
    }

    /// A halted threaded vCPU: sleeps until an interrupt, an NMI, a start-up IPI, a store to the
    /// line `mwait` watches, or the machine ending. Returns `Some` only for the last.
    pub(super) fn wait_for_wake(&mut self) -> Option<ExitReason> {
        let links = self.links();
        let link = &links.links[self.cpu_index as usize];
        link.wakeable.store(
            !self.wait_for_sipi && self.eflags & flags::IF != 0 && self.svm.gif,
            Ordering::SeqCst,
        );
        link.halted.store(true, Ordering::SeqCst);
        links.leave_running();
        let exit = loop {
            if self.stop.load(Ordering::Relaxed) {
                break Some(ExitReason::Stopped);
            }
            let attention = link.attention.load(Ordering::SeqCst);
            if self.wake_condition(attention) {
                break None;
            }
            // Everything that can end the wait comes with a wake-up, except a store to the line
            // `mwait` watches, which has to be polled for.
            Waker::wait(if self.mwait_watch.is_some() {
                MAX_SLEEP
            } else {
                HALTED_SLEEP
            });
        };
        links.enter_running();
        link.halted.store(false, Ordering::SeqCst);
        if exit.is_none() && self.mwait_watch.take().is_some() {
            self.halted = false;
        }
        exit
    }

    /// Acts on INIT and start-up IPIs other vCPUs sent this one (time-sharing runs it when the
    /// scheduler enters the vCPU, threads when the device bit says so).
    pub(super) fn apply_start_signals(&mut self) {
        let index = self.cpu_index as usize;
        let (init, startup) = {
            let mut devices = self.dev.lock();
            let lapic = devices.lapic_of(index);
            if !lapic.start_signal_pending() {
                return;
            }
            let init = lapic.take_init();
            let startup = lapic.take_startup();
            if init {
                lapic.init_reset();
            }
            (init, startup)
        };
        if init {
            self.init();
        }
        // A start-up IPI to a vCPU that is already running is ignored (SDM 10.4.7.3).
        if let Some(vector) = startup
            && self.wait_for_sipi
        {
            self.start_at(vector);
        }
    }

    /// `pause` on a threaded vCPU: tell the host core it is spinning, and offer the slice now and
    /// then, because the vCPU it waits for may be a thread that is not running at all when there
    /// are more vCPUs than host cores.
    pub(super) fn pause_threaded(&mut self) {
        const PAUSES_PER_YIELD: u32 = 128;
        self.pause_count = self.pause_count.wrapping_add(1);
        if self.pause_count.is_multiple_of(PAUSES_PER_YIELD) {
            crate::sync::yield_now();
        } else {
            core::hint::spin_loop();
        }
    }
}

/// How the machine starts a thread. The default (hosted) is `std::thread`; a freestanding embedder
/// supplies its own with `MachineConfig::thread_spawner`.
pub(crate) type ThreadSpawner = Arc<dyn Fn(Box<dyn FnOnce() + Send + 'static>) + Send + Sync>;

#[cfg(not(freestanding))]
pub(crate) fn std_thread_spawner() -> ThreadSpawner {
    Arc::new(|body| {
        std::thread::spawn(body);
    })
}

/// Seeds `rdrand` differently per vCPU: with one seed they would all produce the same stream.
fn vcpu_seed(seed: u64, index: u64) -> u64 {
    const GOLDEN_RATIO: u64 = 0x9e37_79b9_7f4a_7c15;
    seed ^ index.wrapping_mul(GOLDEN_RATIO)
}

/// Marks a vCPU thread finished when it ends, including by a panic (which would otherwise leave
/// the machine waiting for it forever).
struct ThreadGuard {
    links: Arc<VcpuLinks>,
    index: usize,
    done: bool,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        if !self.done {
            let link = &self.links.links[self.index];
            link.cpu.store(core::ptr::null_mut(), Ordering::SeqCst);
            // A vCPU that panicked inside an exclusive section would leave every other vCPU
            // parked for good.
            self.links
                .exclusive_requested
                .store(false, Ordering::SeqCst);
            self.links.exclusive_owner.store(false, Ordering::SeqCst);
            self.links.leave_running();
            self.links.request_exit(ExitReason::Internal(format!(
                "vCPU {} panicked",
                self.index
            )));
            self.links.finished.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// A device (DMA) wrote `page`, which held translated code: the blocks go for every vCPU, the
/// flag is cleared, and the vCPUs are told to drop their decode caches. A vCPU's own write does
/// the same from `Cpu::code_written`; this is the device thread's, which no exclusive section
/// waits for, so it has to go through the cache's lock like any other writer.
fn invalidate_for_device_write(
    links: &VcpuLinks,
    jit: Option<&crate::jit::Jit>,
    flags: &CodePages,
    page: usize,
) {
    match jit {
        Some(jit) => jit.invalidate_page(page as u64, flags),
        None => flags.set(page, 0),
    }
    links.publish_invalidation(None, page);
}

/// Runs the machine with one thread per vCPU and returns why it ended, with the bootstrap
/// processor (for the exit dump) and a line per vCPU.
///
/// The calling thread becomes the device thread: it ticks the devices until every vCPU thread has
/// finished. `make_jit` builds a handle on the shared code cache for each secondary vCPU and for
/// the device thread.
pub(crate) fn run_threaded(
    mut cpus: Vec<Cpu>,
    spawner: &ThreadSpawner,
    make_jit: &dyn Fn() -> Option<Box<crate::jit::Jit>>,
) -> (ExitReason, Cpu, String) {
    let count = cpus.len();
    let links = VcpuLinks::new(count);
    let device_handle = cpus[0].dev.for_cpu(0);
    let stop = cpus[0].stop.clone();
    let poll = cpus[0].poll.take();
    let mut ram = cpus[0].ram.clone();
    let flags = cpus[0].page_has_code.clone();
    let wall = cpus[0].wall_clock.clone();
    device_handle.lock().attach_links(links.clone());
    device_handle.set_locking(true);
    let base_seed = cpus[0].rng_state;
    for (index, cpu) in cpus.iter_mut().enumerate() {
        cpu.rng_state = vcpu_seed(base_seed, index as u64);
        cpu.attach_threads(links.clone());
        if index > 0 {
            cpu.give_private_caches(make_jit());
        }
    }
    // Publish what the device model already has pending (the PIC may be asserting a line).
    drop(device_handle.lock());
    let results: Arc<Mutex<Vec<Option<Box<Cpu>>>>> =
        Arc::new(Mutex::new((0..count).map(|_| None).collect()));
    for (index, cpu) in cpus.drain(..).enumerate() {
        let links = links.clone();
        let results = results.clone();
        let mut cpu = Box::new(cpu);
        spawner(Box::new(move || {
            let mut guard = ThreadGuard {
                links: links.clone(),
                index,
                done: false,
            };
            cpu.run_thread();
            guard.done = true;
            results.lock().unwrap()[index] = Some(cpu);
            links.finished.fetch_add(1, Ordering::SeqCst);
        }));
    }
    links.device_waker.register_current();
    // The device thread is not a vCPU, so nothing else drops the blocks of a page a device wrote.
    let device_jit = make_jit();
    while !links.all_finished() {
        if stop.load(Ordering::Relaxed) {
            links.request_exit(ExitReason::Stopped);
        }
        if let Some(poll) = &poll {
            poll(false);
        }
        let now = wall.as_ref().map_or(0, WallClock::elapsed_ns);
        let written_code_pages = {
            let mut mem = GuestMem::new(&mut ram, flags.as_bytes());
            device_handle.lock().tick(now, &mut mem);
            mem.code_pages_written
        };
        for page in written_code_pages {
            invalidate_for_device_write(&links, device_jit.as_deref(), &flags, page);
        }
        let (reset, shutdown, deadline) = {
            let devices = device_handle.lock();
            (
                devices.reset_requested,
                devices.shutdown_requested(),
                devices.next_deadline(now),
            )
        };
        if reset {
            links.request_exit(ExitReason::Reset);
        } else if shutdown || links.all_dead() {
            links.request_exit(ExitReason::Shutdown);
        }
        let sleep = deadline
            .map(|at| Duration::from_nanos(at.saturating_sub(now)))
            .map_or(MAX_SLEEP, |wait| wait.min(MAX_SLEEP));
        links.device_sleeping.store(true, Ordering::SeqCst);
        // Work that arrived between the tick above and the flag is picked up by the next pass:
        // the sleep is short.
        if !links.exiting() {
            // A freestanding host cannot sleep a thread, but its poll hook can wait for it (the
            // same hook the single-threaded idle loop uses); without one, spin politely.
            #[cfg(freestanding)]
            match &poll {
                Some(poll) => poll(true),
                None => Waker::wait(sleep),
            }
            #[cfg(not(freestanding))]
            Waker::wait(sleep);
        }
        links.device_sleeping.store(false, Ordering::SeqCst);
    }
    let reason = links.exit_reason();
    device_handle.lock().detach_links();
    let mut results = results.lock().unwrap();
    let mut finished: Vec<Cpu> = results.drain(..).flatten().map(|cpu| *cpu).collect();
    let summary = finished
        .iter()
        .map(|cpu| {
            format!(
                "cpu{}: rip={:016x} cpl={} halted={} wait_for_sipi={} if={}\n",
                cpu.cpu_index,
                cpu.rip,
                cpu.cpl,
                cpu.halted,
                cpu.wait_for_sipi,
                cpu.eflags & flags::IF != 0
            )
        })
        .collect();
    if finished.is_empty() {
        // Every thread panicked before handing its `Cpu` back; there is nothing left to inspect.
        return (
            reason,
            Cpu::new(
                Vec::new(),
                Default::default(),
                Default::default(),
                Arc::new(AtomicBool::new(false)),
                None,
                Default::default(),
            ),
            summary,
        );
    }
    let bsp = finished.remove(0);
    (reason, bsp, summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_halted(links: &VcpuLinks) {
        for link in &links.links {
            link.halted.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn halted_vcpus_that_cannot_wake_are_a_dead_guest() {
        let links = VcpuLinks::new(2);
        all_halted(&links);
        assert!(links.all_dead());
    }

    #[test]
    fn a_pending_startup_ipi_is_not_a_dead_guest_even_before_its_vcpu_notices() {
        let links = VcpuLinks::new(2);
        all_halted(&links);
        links.publish_device_bits(1, ATTENTION_START);
        assert!(!links.all_dead(), "the parked vCPU is about to start");
    }

    /// A harness CPU whose JIT has translated the block at `CODE_BASE`, and that block's page.
    fn translated_block() -> Option<(crate::testkit::Harness, Box<crate::jit::Jit>, usize)> {
        use crate::testkit::{CODE_BASE, Harness};
        let mut harness = Harness::new(&[0x90, 0xf4]);
        if !harness.enable_jit() {
            return None;
        }
        let mut jit = harness.cpu.jit.take().unwrap();
        jit.translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        assert_eq!(harness.cpu.page_has_code.get(page), 1);
        Some((harness, jit, page))
    }

    #[test]
    fn a_device_write_to_a_code_page_drops_its_blocks_clears_its_flag_and_tells_the_vcpus() {
        let Some((harness, mut jit, page)) = translated_block() else {
            return;
        };
        let links = VcpuLinks::new(2);
        invalidate_for_device_write(&links, Some(&jit), &harness.cpu.page_has_code, page);
        assert_eq!(harness.cpu.page_has_code.get(page), 0);
        assert_eq!(
            jit.lookup(crate::testkit::CODE_BASE, crate::testkit::CODE_BASE),
            None,
            "the block of a page a device wrote stayed valid"
        );
        for link in &links.links {
            assert_ne!(
                link.attention.load(Ordering::SeqCst) & ATTENTION_INVALIDATE,
                0
            );
        }
    }

    #[test]
    fn a_device_write_without_a_jit_still_clears_the_flag() {
        let Some((harness, _jit, page)) = translated_block() else {
            return;
        };
        let links = VcpuLinks::new(1);
        invalidate_for_device_write(&links, None, &harness.cpu.page_has_code, page);
        assert_eq!(harness.cpu.page_has_code.get(page), 0);
    }

    /// The flag is cleared under the cache's lock by whoever invalidates the page. Telling the
    /// vCPUs about it must not clear it again: a translation that flagged the page in between
    /// would lose it.
    #[test]
    fn publishing_an_invalidation_leaves_the_code_flag_alone() {
        let Some((harness, _jit, page)) = translated_block() else {
            return;
        };
        let links = VcpuLinks::new(2);
        links.publish_invalidation(None, page);
        assert_eq!(harness.cpu.page_has_code.get(page), 1);
    }

    #[test]
    fn a_pending_nmi_is_not_a_dead_guest() {
        let links = VcpuLinks::new(2);
        all_halted(&links);
        links.publish_device_bits(0, ATTENTION_NMI);
        assert!(!links.all_dead());
    }
}
