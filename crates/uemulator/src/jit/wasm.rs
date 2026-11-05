//! The WebAssembly backend's code cache and `Cpu::step_block`. Same job as `native.rs`, much less
//! machinery: there is no executable buffer to manage, no code to patch and no second thread to
//! share it with. A block is a function in a module (`emit_wasm`, `wasm_encode`); the host turns
//! the module's bytes into functions in the emulator's function table (`host_jit_compile`,
//! implemented by `www/jit_host.js`) and a block is then called through its table slot like any
//! function pointer: on wasm32 a pointer to a function is its table index.
//!
//! Compiling costs the engine ~38 µs a block on the thread that does it (V8: making the module,
//! instantiating it, and compiling the function on its first call), so a block is only compiled
//! once its first instruction has run `COMPILE_THRESHOLD` times in the interpreter; until then
//! `step_block` interprets one instruction. Tests build with a threshold of zero.
//!
//! Where the host has a compile worker (V8 only, see `www/jit_host.js`), a hot block's function
//! is generated here and queued (`State::pending`), and the queued blocks go to the worker as one
//! module (`host_jit_submit`) while it has none of ours, so they batch up while it is busy. The
//! block runs in the interpreter until the worker is done (`host_jit_ready`, or `PATIENCE`); then
//! `host_jit_compile` makes the same bytes into a module here, which V8 serves from its
//! process-wide cache already compiled, for about a microsecond a block. Batching and the worker
//! together took the Alpine demo's boot to `Run /init` from 5.4 to 3.7 s under Node
//! (docs/plans/13-web.md). Without a worker, `host_jit_submit` refuses and every block is
//! compiled here as it gets hot, in a module of its own.
//!
//! There is no code to patch, so the native backends' direct links become link cells: memory a
//! block's same-page exits read the successor's table slot from (`emit_wasm::LINK_CELL_BLOCK`),
//! filled by the linking chain stub. A linked transition skips the stub's TLB probe and jump
//! cache, and each exit gets an indirect call site of its own instead of the stub's one shared
//! site (likely why two blocks jumping to each other went from 23 to 12 ns per round trip).
//!
//! The table belongs to one instance, so a second thread (a wasm worker) cannot call the blocks of
//! the first: with this backend the vCPUs always take turns on one thread (`runs_on_threads`).
//! That is also why the jump cache is one table shared by every handle, with no locking: generated
//! code (the chain stub) reads it too.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::{BTreeMap, VecDeque};
use core::cell::UnsafeCell;
use core::mem::{offset_of, size_of};
use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use super::blocks::{
    BlockKey, Decoded, JUMP_CACHE_BITS, NO_SECOND_PAGE, PAGE_MASK, PAGE_SHIFT, PAGE_SIZE,
    SLOT_BLOCK_BITS, SLOT_BLOCK_MASK, STRADDLER_TAG, decode_block, jump_cache_hash,
};
use super::emit_wasm::{self, Addrs, ChainTables, LINK_CELL_SIZE, LINKS_PER_BLOCK};
use super::helpers;
use super::wasm_encode::{self, Code, op};
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;
use crate::ram::CodePages;
use crate::sync::{Arc, Mutex};

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// Instantiates the module at `ptr..ptr + len` against the emulator's memory and function
    /// table and puts its exports "0" to `count - 1` in table slots, whose numbers it writes to
    /// `slots`. Returns 0, or -1 if the engine refused the module.
    fn host_jit_compile(ptr: *const u8, len: usize, slots: *mut u32, count: usize) -> i32;
    /// Gives a slot back; the host empties it so the engine can free the instance.
    fn host_jit_release(slot: i32);
    /// Hands a copy of the module to the host's compile worker and returns a ticket, or -1 when
    /// there is no compile worker or it has no room: the module is then compiled at once.
    fn host_jit_submit(ptr: *const u8, len: usize) -> i32;
    /// Whether the compile worker is done with `ticket` (or the host gave up waiting for it).
    fn host_jit_ready(ticket: i32) -> i32;
}

/// Times the first instruction of a block must run in the interpreter before the block is
/// compiled. The engine needs tens of microseconds per module, so this is what keeps the cold
/// start-up code of a boot (most of it runs once) from being slower than interpreting it.
const COMPILE_THRESHOLD: u32 = 40;
/// `COMPILE_THRESHOLD` once a compile worker takes modules (`State::pump`): compiling then costs
/// this thread a few microseconds a block instead of ~38, and the Alpine boot to `Run /init`
/// took 3.7 s at 10 against 4.0 at 40 (3 and 5 were slower again: more code compiled that runs
/// only a few times).
const COMPILE_THRESHOLD_WITH_WORKER: u32 = 10;
/// A `Heat` count that means the engine refused this block: it is interpreted from now on.
const NEVER_COMPILE: u32 = u32::MAX;
const HEAT_BITS: usize = 16;

/// Most blocks one cache holds before it is flushed. Every block is a live instance in the engine,
/// so this bounds that memory; the validity table is this long.
const MAX_BLOCKS: usize = 1 << 16;
const _: () = assert!(MAX_BLOCKS as u64 <= 1 << SLOT_BLOCK_BITS);
/// The block an empty link cell names (`emit_wasm::LINK_CELL_BLOCK`): one past the last real
/// index, whose validity byte is never set, so a linked exit through an empty cell always takes
/// the stub.
const NO_BLOCK: u32 = MAX_BLOCKS as u32;
const _: () = assert!(LINK_CELL_SIZE as usize == size_of::<AtomicU64>());

/// What the probe module returns.
const PROBE_RESULT: i64 = 0x7e57_ab1e;

/// Most blocks one module holds. A module costs the engine a fixed amount on top of its
/// functions (13 µs a block alone, 1 µs a block at 64 in V8, docs/plans/13-web.md); a batch is
/// rarely full, 128 and 256 measured the same.
const MAX_BATCH_BLOCKS: usize = 64;
/// Modules handed to the compile worker and not installed yet. Blocks that get hot while this
/// many are out wait in `State::queued` and go out together in the next module. One is what
/// batches them best: the Alpine boot to `Run /init` took 4.0 s at one (64 blocks a module),
/// 4.1 s at two, 4.2 s at four (16 a module) and 5.1 s at 64 modules of one block.
const MAX_IN_FLIGHT: usize = 1;
/// Passes through a pending block the interpreter may make before this thread stops waiting for
/// the compile worker and instantiates the block's module itself: `PATIENCE` times the block's
/// instruction count in steps, at its first instruction or inside it, so a loop within the block
/// runs out of patience as soon as passes through all of it would. The worker can be
/// milliseconds behind in a burst of new blocks, and a hot loop runs thousands of times in that.
/// The engine then reuses what the worker already compiled, or compiles just the functions that
/// run. Unbatched, waiting for the worker alone was no faster than compiling here; 4 to 32 measured
/// within noise of each other.
const PATIENCE: u32 = 8;

/// The table slots of `module`'s `count` functions, or `None` if the engine refused it.
fn compile_functions(module: &[u8], count: usize) -> Option<Vec<u32>> {
    let mut slots = vec![0u32; count];
    // SAFETY: the host only reads the bytes and writes `count` slots during the call.
    let status =
        unsafe { host_jit_compile(module.as_ptr(), module.len(), slots.as_mut_ptr(), count) };
    (status == 0).then_some(slots)
}

fn compile_module(module: &[u8]) -> Option<u32> {
    compile_functions(module, 1).map(|slots| slots[0])
}

fn release_slot(slot: u32) {
    // SAFETY: `slot` came from `host_jit_compile` and is released once.
    unsafe { host_jit_release(slot as i32) };
}

/// Whether the host compiles and runs generated modules, by doing it with the smallest one: a
/// function that returns a constant. Cached, so the host sees one probe however many machines are
/// built.
pub(crate) fn probe() -> bool {
    static RESULT: AtomicU8 = AtomicU8::new(PROBE_UNKNOWN);
    const PROBE_UNKNOWN: u8 = 0;
    const PROBE_WORKS: u8 = 1;
    const PROBE_FAILS: u8 = 2;
    match RESULT.load(Ordering::Relaxed) {
        PROBE_WORKS => return true,
        PROBE_FAILS => return false,
        _ => {}
    }
    let mut code = Code::default();
    code.i64_const(PROBE_RESULT).op(op::END);
    let works = compile_module(&wasm_encode::block_module(&code.bytes, 0, 0)).is_some_and(|slot| {
        let block: extern "C" fn(*mut Cpu) -> u64 = unsafe { core::mem::transmute(slot as usize) };
        let result = block(core::ptr::null_mut());
        release_slot(slot);
        result == PROBE_RESULT as u64
    });
    RESULT.store(
        if works { PROBE_WORKS } else { PROBE_FAILS },
        Ordering::Relaxed,
    );
    works
}

/// A function pointer on wasm32 is an index into the function table, which is how generated
/// code names the helpers.
fn addrs(out: *mut u64, chain: u32, chain_link: u32, valid: &[AtomicU8]) -> Addrs {
    Addrs {
        exec: helpers::helper_exec as *const () as usize as u32,
        ld: helpers::helper_load_via_out as *const () as usize as u32,
        st: helpers::helper_store_x64 as *const () as usize as u32,
        ld_stack: helpers::helper_load_stack_via_out as *const () as usize as u32,
        st_stack: helpers::helper_store_stack_x64 as *const () as usize as u32,
        cond: helpers::helper_cond as *const () as usize as u32,
        rotate_flags: helpers::helper_set_rotate_flags as *const () as usize as u32,
        whole_vector: helpers::helper_whole_vector as *const () as usize as u32,
        out: out as usize as u32,
        chain,
        chain_link,
        valid: valid.as_ptr() as usize as u32,
    }
}

/// One slot of the direct-mapped jump cache. `#[repr(C)]` with a power-of-two size because the
/// chain stub (`emit_wasm::compile_chain_stub`) indexes the table and reads these fields at the
/// offsets `ChainTables` is given. `slot == 0` is empty: table slot 0 is the null function and
/// never holds a block.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct JumpEntry {
    rip: u64,
    phys_addr: u64,
    /// Index of the block in `State::blocks`; for a straddler the physical page number of its
    /// second page sits above `SLOT_BLOCK_BITS`, as in `native.rs`. The stub reads only the low
    /// word, which is the index: a straddler's `phys_addr` is tagged, so it never gets that far.
    block: u64,
    slot: u32,
    _pad: u32,
}

const _: () = assert!(size_of::<JumpEntry>().is_power_of_two());

/// The jump cache. Read and written by `Jit` and read by generated code, none of which can run at
/// the same time as another vCPU thread (see the module docs).
struct JumpCache(Box<[UnsafeCell<JumpEntry>]>);

// SAFETY: one thread at a time uses a wasm-backed `Jit`.
unsafe impl Sync for JumpCache {}

impl JumpCache {
    fn new() -> JumpCache {
        JumpCache(
            (0..1 << JUMP_CACHE_BITS)
                .map(|_| UnsafeCell::new(JumpEntry::default()))
                .collect(),
        )
    }

    fn get(&self, rip: u64) -> JumpEntry {
        unsafe { *self.0[jump_cache_hash(rip)].get() }
    }

    fn set(&self, rip: u64, entry: JumpEntry) {
        unsafe { *self.0[jump_cache_hash(rip)].get() = entry };
    }

    fn clear(&self) {
        for entry in self.0.iter() {
            unsafe { *entry.get() = JumpEntry::default() };
        }
    }

    fn base(&self) -> u32 {
        self.0.as_ptr() as usize as u32
    }
}

/// How many times the first instruction of a block at `phys` has run in the interpreter.
#[derive(Clone, Copy, Default)]
struct Heat {
    phys: u64,
    count: u32,
}

/// One translated block. Blocks are only ever added; an invalidated one keeps its slot (cleared in
/// `SharedJit::valid`) until the next flush, so indices held in `page_blocks` and the jump caches
/// stay stable.
struct Block {
    key: BlockKey,
    /// The function table slot the block's function occupies.
    slot: u32,
    /// Generated code calls `helper_exec` with the address of one of these, so they must outlive
    /// the code. Heap storage keeps the addresses stable when `blocks` reallocates.
    _insns: Box<[Insn]>,
    /// The block's link cells (`emit_wasm::LINK_CELL_BLOCK`), written by the linking chain stub
    /// and read by the block's exits; same lifetime rule as `_insns`. Empty for a straddling
    /// block, whose exits are never linked.
    _links: Box<[AtomicU64]>,
}

/// A block whose function is generated but not installed: queued, or in a module the compile
/// worker has. Holds what the code points into (`Block::_insns`, `Block::_links`) from the moment
/// the code is generated.
struct PendingBlock {
    /// Matches `Pending::serial` for as long as the block may still be installed.
    serial: u64,
    phys_addr: u64,
    pages: Vec<u64>,
    insns: Box<[Insn]>,
    links: Box<[AtomicU64]>,
    compiled: emit_wasm::CompiledBlock,
}

/// The `State::pending` entry of a block.
struct Pending {
    /// Physical address just past the block's last instruction (same page as its start).
    end: u64,
    serial: u64,
    /// Guest instructions in it.
    insns: u32,
    /// Its instructions the interpreter ran while it was pending (`PATIENCE`).
    steps: u32,
}

/// One module handed to the compile worker: the blocks' functions in order, exported as "0", "1"...
struct Batch {
    ticket: u32,
    module: Vec<u8>,
    blocks: Vec<PendingBlock>,
}

/// Where a physical address stands against the pending blocks.
enum PendingAt {
    /// A pending block starts there.
    Head,
    /// It is inside the pending block starting at this address, past its first instruction.
    Inside(u64),
    Outside,
}

/// Stands in for the host's compile worker in tests: modules are done only when the test says
/// (`Jit::finish_compiles`), or after `finish_after_polls` asks.
#[cfg(test)]
#[derive(Default)]
struct FakeCompiler {
    submitted: u32,
    /// Tickets below this are done.
    finished: u32,
    finish_after_polls: Option<u32>,
    polls: u32,
}

/// A vCPU's handle on the code cache. The cache itself (`SharedJit`) belongs to the machine and
/// every handle (`sibling`) shares it.
pub(crate) struct Jit {
    shared: Arc<SharedJit>,
}

struct SharedJit {
    /// Per block index: 1 while the block may run, 0 once its page was written (or after a flush).
    /// One longer than `MAX_BLOCKS`: the last byte is `NO_BLOCK`'s, which stays 0.
    valid: Box<[AtomicU8]>,
    jump_cache: JumpCache,
    /// Table slot of the chain stub, or 0 without chaining.
    chain_slot: u32,
    /// Table slot of the linking chain stub, or 0 when exits are not linked.
    chain_link_slot: u32,
    /// Counts `invalidate_page` calls: a translation decodes outside the lock, so it compares this
    /// before and after to see a write to a page it read (see `native.rs`).
    invalidations: AtomicU64,
    /// Where the generated code's slow-path loads leave their value. One slot is enough: the
    /// helper returns before anything else reads it.
    out: Box<AtomicU64>,
    compile_threshold: AtomicU32,
    #[cfg(test)]
    test_block_limit: core::sync::atomic::AtomicUsize,
    #[cfg(test)]
    fake_compiler: Mutex<Option<FakeCompiler>>,
    state: Mutex<State>,
}

struct State {
    blocks: Vec<Block>,
    /// Every live block by `BlockKey`.
    block_map: BTreeMap<BlockKey, u32>,
    /// Physical page number -> blocks that decoded bytes from it.
    page_blocks: BTreeMap<u64, Vec<u32>>,
    heat: Box<[Heat]>,
    /// Blocks generated but not installed, by physical address. A block is installed only while
    /// its entry is here with its serial: a write to its page (`Jit::invalidate_page`) or a flush
    /// removes the entry, and its function, compiled from the old bytes, is then thrown away when
    /// its module comes back. Straddlers are never pending: they are compiled at once.
    pending: BTreeMap<u64, Pending>,
    /// Pending blocks not handed out yet, waiting for room in `in_flight`.
    queued: Vec<PendingBlock>,
    /// Modules the compile worker has, in the order they were handed out (it finishes them in
    /// that order).
    in_flight: VecDeque<Batch>,
    next_serial: u64,
    stats: JitStats,
}

/// Translation counters, read by tests to check that the cache is used and invalidated as
/// intended (`Jit::stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct JitStats {
    /// Blocks compiled, including ones compiled again after an invalidation or a flush.
    pub translated: u64,
    /// Guest instructions lowered inline, i.e. not through `helper_exec`.
    pub inlined: u64,
    /// Guest instructions compiled, inline or not.
    pub total: u64,
    /// Times the whole cache was flushed because it was full.
    pub resets: u64,
    /// Lazy-flag records left out of compiled blocks as dead.
    pub flag_records_skipped: u64,
    /// Decodes thrown away because the page was written before the block was registered.
    pub stale_decodes: u64,
    /// Blocks the host refused to instantiate; they run in the interpreter.
    pub compile_failures: u64,
    /// Guest instructions the interpreter ran because their block was not hot yet (or its
    /// module was not ready).
    pub cold_steps: u64,
    /// Modules instantiated for blocks; fewer than `translated` when modules hold several.
    pub modules: u64,
    /// Modules handed to the compile worker.
    pub submitted: u64,
    /// Modules this thread instantiated before the compile worker was done (`PATIENCE`).
    pub impatient: u64,
    /// Generated blocks thrown away before they were installed: their page was written, or the
    /// cache was flushed, while their module was being compiled.
    pub abandoned: u64,
}

/// What `Jit::install` and `Jit::enqueue` found.
enum Install {
    Ready(u32),
    /// The block waits for its module (`State::pending`).
    Queued,
    /// A page the decode read was written since: decode again.
    Stale,
    /// No room for another block: flush the cache, then decode again.
    Full,
    /// The host refused the module.
    Refused,
}

impl SharedJit {
    fn lock(&self) -> impl core::ops::DerefMut<Target = State> + '_ {
        self.state.lock().unwrap()
    }

    fn block_limit(&self) -> usize {
        #[cfg(test)]
        return self
            .test_block_limit
            .load(Ordering::Relaxed)
            .min(MAX_BLOCKS);
        #[cfg(not(test))]
        MAX_BLOCKS
    }

    /// The ticket of `module` at the compile worker, or `None` if it should be compiled now.
    fn submit(&self, module: &[u8]) -> Option<u32> {
        #[cfg(test)]
        if let Some(fake) = self.fake_compiler.lock().unwrap().as_mut() {
            fake.submitted += 1;
            return Some(fake.submitted - 1);
        }
        // SAFETY: the host copies the bytes during the call.
        let ticket = unsafe { host_jit_submit(module.as_ptr(), module.len()) };
        u32::try_from(ticket).ok()
    }

    fn ready(&self, ticket: u32) -> bool {
        #[cfg(test)]
        if let Some(fake) = self.fake_compiler.lock().unwrap().as_mut() {
            if ticket >= fake.finished
                && let Some(polls) = fake.finish_after_polls
            {
                fake.polls += 1;
                if fake.polls >= polls {
                    fake.finished = fake.submitted;
                    fake.polls = 0;
                }
            }
            return ticket < fake.finished;
        }
        // SAFETY: plain call; the host only looks the ticket up.
        unsafe { host_jit_ready(ticket as i32) != 0 }
    }
}

impl State {
    /// Drops every block and gives their slots back, and forgets the pending ones. No generated
    /// code may be running.
    fn flush(&mut self, shared: &SharedJit) {
        for block in self.blocks.drain(..) {
            release_slot(block.slot);
        }
        for flag in shared.valid.iter() {
            flag.store(0, Ordering::Release);
        }
        // The entries name blocks that are gone, and their indices are about to be reused.
        shared.jump_cache.clear();
        self.block_map.clear();
        self.page_blocks.clear();
        // The worker finishes the modules it has, and nothing will ask for them.
        let dropped =
            self.queued.len() + self.in_flight.iter().map(|b| b.blocks.len()).sum::<usize>();
        self.stats.abandoned += dropped as u64;
        self.queued.clear();
        self.in_flight.clear();
        self.pending.clear();
        self.stats.resets += 1;
    }

    /// Counts one more interpreted step of the pending block at `phys_addr`; its serial once it
    /// has run out of `PATIENCE`.
    fn out_of_patience(&mut self, phys_addr: u64) -> Option<u64> {
        let pending = self.pending.get_mut(&phys_addr)?;
        pending.steps = pending.steps.saturating_add(1);
        (pending.steps >= PATIENCE.saturating_mul(pending.insns)).then_some(pending.serial)
    }

    fn is_pending(&self, block: &PendingBlock) -> bool {
        self.pending
            .get(&block.phys_addr)
            .is_some_and(|pending| pending.serial == block.serial)
    }

    /// Makes block `index` of a translation live: in the block map, valid, and found by its pages.
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        shared: &SharedJit,
        key: BlockKey,
        slot: u32,
        insns: Box<[Insn]>,
        links: Box<[AtomicU64]>,
        pages: Vec<u64>,
        compiled: &emit_wasm::CompiledBlock,
    ) -> u32 {
        self.stats.inlined += compiled.inlined as u64;
        self.stats.flag_records_skipped += compiled.flag_records_skipped as u64;
        self.stats.total += insns.len() as u64;
        let index = self.blocks.len() as u32;
        self.block_map.insert(key, index);
        self.blocks.push(Block {
            key,
            slot,
            _insns: insns,
            _links: links,
        });
        shared.valid[index as usize].store(1, Ordering::Release);
        for page in pages {
            self.page_blocks.entry(page).or_default().push(index);
        }
        self.stats.translated += 1;
        index
    }

    /// Installs the modules the compile worker has finished, oldest first, then hands it queued
    /// blocks while it has room (or compiles them here when there is no worker). `waiting` is
    /// the pending block one of whose instructions is about to be interpreted: once it has run
    /// out of `PATIENCE` its module is installed now, done or not. Returns true when a
    /// module did not fit: the caller flushes the cache, which also drops it.
    fn pump(&mut self, shared: &SharedJit, waiting: Option<u64>) -> bool {
        while let Some(batch) = self.in_flight.front() {
            if !shared.ready(batch.ticket) {
                break;
            }
            let batch = self.in_flight.pop_front().unwrap();
            if !self.install_batch(shared, batch) {
                return true;
            }
        }
        if let Some(serial) = waiting.and_then(|phys_addr| self.out_of_patience(phys_addr)) {
            let holds = |batch: &Batch| batch.blocks.iter().any(|block| block.serial == serial);
            let batch = match self.in_flight.iter().position(holds) {
                Some(position) => self.in_flight.remove(position),
                None => self
                    .queued
                    .iter()
                    .position(|block| block.serial == serial)
                    .map(|position| {
                        let block = self.queued.remove(position);
                        let module = wasm_encode::module_of_functions(&[block.compiled.function()]);
                        Batch {
                            ticket: 0,
                            module,
                            blocks: vec![block],
                        }
                    }),
            };
            if let Some(batch) = batch {
                self.stats.impatient += 1;
                if !self.install_batch(shared, batch) {
                    return true;
                }
            }
        }
        while self.in_flight.len() < MAX_IN_FLIGHT && !self.queued.is_empty() {
            let count = self.queued.len().min(MAX_BATCH_BLOCKS);
            let blocks: Vec<PendingBlock> = self.queued.drain(..count).collect();
            let functions: Vec<_> = blocks.iter().map(|b| b.compiled.function()).collect();
            let module = wasm_encode::module_of_functions(&functions);
            let batch = Batch {
                ticket: 0,
                module,
                blocks,
            };
            match shared.submit(&batch.module) {
                Some(ticket) => {
                    self.stats.submitted += 1;
                    shared
                        .compile_threshold
                        .fetch_min(COMPILE_THRESHOLD_WITH_WORKER, Ordering::Relaxed);
                    self.in_flight.push_back(Batch { ticket, ..batch });
                }
                None => {
                    if !self.install_batch(shared, batch) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Instantiates `batch`'s module and installs the blocks in it that are still pending.
    /// Returns false, installing nothing, if they would not fit in the cache.
    fn install_batch(&mut self, shared: &SharedJit, batch: Batch) -> bool {
        let Batch { module, blocks, .. } = batch;
        let live = blocks.iter().filter(|b| self.is_pending(b)).count();
        if live == 0 {
            self.stats.abandoned += blocks.len() as u64;
            return true;
        }
        if self.blocks.len() + live > shared.block_limit() {
            self.stats.abandoned += blocks.len() as u64;
            return false;
        }
        let Some(slots) = compile_functions(&module, blocks.len()) else {
            // A bug in the generator: these blocks stay interpreted.
            for block in &blocks {
                if !self.is_pending(block) {
                    continue;
                }
                self.pending.remove(&block.phys_addr);
                self.stats.compile_failures += 1;
                self.heat[heat_index(block.phys_addr)] = Heat {
                    phys: block.phys_addr,
                    count: NEVER_COMPILE,
                };
            }
            return true;
        };
        self.stats.modules += 1;
        for (block, slot) in blocks.into_iter().zip(slots) {
            if !self.is_pending(&block) {
                // Its page was written while the module was compiled: the code is stale.
                release_slot(slot);
                self.stats.abandoned += 1;
                continue;
            }
            self.pending.remove(&block.phys_addr);
            let PendingBlock {
                phys_addr,
                pages,
                insns,
                links,
                compiled,
                ..
            } = block;
            self.register(
                shared,
                (phys_addr, NO_SECOND_PAGE),
                slot,
                insns,
                links,
                pages,
                &compiled,
            );
        }
        true
    }
}

impl Drop for SharedJit {
    fn drop(&mut self) {
        // A poisoned lock means a vCPU thread panicked; its blocks still hold slots. The
        // freestanding lock cannot fail, which makes the pattern irrefutable there only.
        #[allow(irrefutable_let_patterns)]
        if let Ok(mut state) = self.state.lock() {
            for block in state.blocks.drain(..) {
                release_slot(block.slot);
            }
        }
        for slot in [self.chain_slot, self.chain_link_slot] {
            if slot != 0 {
                release_slot(slot);
            }
        }
    }
}

fn heat_index(phys_addr: u64) -> usize {
    ((phys_addr ^ (phys_addr >> HEAT_BITS)) as usize) & ((1 << HEAT_BITS) - 1)
}

impl Jit {
    /// `chaining` lets blocks tail-call their successor through the chain stub instead of returning
    /// to the run loop between blocks; the lockstep verifier needs one block per `step_block`, so
    /// it passes false.
    pub fn new(chaining: bool) -> Option<Jit> {
        if !probe() {
            return None;
        }
        let valid: Box<[AtomicU8]> = (0..=MAX_BLOCKS).map(|_| AtomicU8::new(0)).collect();
        let jump_cache = JumpCache::new();
        let (chain_slot, chain_link_slot) = if chaining {
            let tables = ChainTables {
                jump_cache: jump_cache.base(),
                valid: valid.as_ptr() as usize as u32,
                entry_shift: size_of::<JumpEntry>().trailing_zeros(),
                entry_phys: offset_of!(JumpEntry, phys_addr) as u32,
                entry_block: offset_of!(JumpEntry, block) as u32,
                entry_slot: offset_of!(JumpEntry, slot) as u32,
                index_bits: JUMP_CACHE_BITS as u32,
            };
            // An engine without tail calls refuses the stubs; blocks then return to the run loop.
            match compile_module(&emit_wasm::compile_chain_stub(&tables, false)) {
                Some(chain) => (
                    chain,
                    compile_module(&emit_wasm::compile_chain_stub(&tables, true)).unwrap_or(0),
                ),
                None => (0, 0),
            }
        } else {
            (0, 0)
        };
        let shared = Arc::new(SharedJit {
            valid,
            jump_cache,
            chain_slot,
            chain_link_slot,
            invalidations: AtomicU64::new(0),
            out: Box::new(AtomicU64::new(0)),
            compile_threshold: AtomicU32::new(if cfg!(freestanding) {
                COMPILE_THRESHOLD
            } else {
                0
            }),
            #[cfg(test)]
            test_block_limit: core::sync::atomic::AtomicUsize::new(usize::MAX),
            #[cfg(test)]
            fake_compiler: Mutex::new(None),
            state: Mutex::new(State {
                blocks: Vec::new(),
                block_map: BTreeMap::new(),
                page_blocks: BTreeMap::new(),
                heat: vec![Heat::default(); 1 << HEAT_BITS].into_boxed_slice(),
                pending: BTreeMap::new(),
                queued: Vec::new(),
                in_flight: VecDeque::new(),
                next_serial: 0,
                stats: JitStats::default(),
            }),
        });
        Some(Jit { shared })
    }

    /// Another vCPU's handle on the same cache: its blocks, its jump cache and its invalidations.
    pub fn sibling(&self) -> Jit {
        Jit {
            shared: self.shared.clone(),
        }
    }

    /// Nothing to do: the vCPUs share one thread, so guest accesses need no ordering.
    pub fn set_ordered_memory(&mut self, _on: bool) {}

    pub fn stats(&self) -> JitStats {
        self.shared.lock().stats
    }

    /// Compile a block the first time it is reached instead of after `COMPILE_THRESHOLD` runs.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_compile_threshold(&self, runs: u32) {
        self.shared.compile_threshold.store(runs, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn set_block_limit(&self, limit: usize) {
        self.shared.test_block_limit.store(limit, Ordering::Relaxed);
    }

    /// Only the native backend has a window between decoding and registering to widen.
    #[cfg(test)]
    pub fn set_install_pause_micros(&self, _micros: u64) {}

    /// Compiles modules as a compile worker would: they are done when `finish_compiles` says,
    /// or after `finish_after_polls` asks for them.
    #[cfg(test)]
    fn use_fake_compiler(&self, finish_after_polls: Option<u32>) {
        *self.shared.fake_compiler.lock().unwrap() = Some(FakeCompiler {
            finish_after_polls,
            ..FakeCompiler::default()
        });
    }

    /// Every module submitted so far is done.
    #[cfg(test)]
    fn finish_compiles(&self) {
        let mut fake = self.shared.fake_compiler.lock().unwrap();
        let fake = fake.as_mut().unwrap();
        fake.finished = fake.submitted;
    }

    /// Physical addresses of the pending blocks, in order.
    #[cfg(test)]
    fn pending_starts(&self) -> Vec<u64> {
        self.shared.lock().pending.keys().copied().collect()
    }

    #[cfg(test)]
    pub fn block_count(&self) -> usize {
        self.shared.lock().blocks.len()
    }

    #[cfg(test)]
    fn block_is_valid(&self, index: usize) -> bool {
        self.shared.valid[index].load(Ordering::Acquire) != 0
    }

    /// Index of the live block for the code at `phys_addr`.
    #[cfg(test)]
    fn block_index(&self, phys_addr: u64) -> Option<u32> {
        let state = self.shared.lock();
        state.block_map.get(&(phys_addr, NO_SECOND_PAGE)).copied()
    }

    /// What block `index`'s link cells hold: `(block index, table slot)` each.
    #[cfg(test)]
    fn link_cells(&self, index: u32) -> Vec<(u32, u32)> {
        const SLOT_SHIFT: u32 = 8 * emit_wasm::LINK_CELL_SLOT;
        let state = self.shared.lock();
        state.blocks[index as usize]
            ._links
            .iter()
            .map(|cell| {
                let value = cell.load(Ordering::Relaxed);
                (value as u32, (value >> SLOT_SHIFT) as u32)
            })
            .collect()
    }

    #[cfg(test)]
    fn slot_of(&self, index: u32) -> u32 {
        self.shared.lock().blocks[index as usize].slot
    }

    /// Called from `Cpu::code_written` when a guest write hits a page that has blocks: they stop
    /// being usable for every handle at once. They are only marked invalid, not freed: the block
    /// that performed the write may still be running, and `Cpu::block_exit_requested` makes it
    /// return before reaching stale code.
    ///
    /// Also clears the page's code flag under the lock, which is what `install` checks under the
    /// same lock (see `native.rs`).
    pub fn invalidate_page(&self, ppn: u64, flags: &CodePages) {
        let shared = &*self.shared;
        let mut state = shared.lock();
        shared.invalidations.fetch_add(1, Ordering::AcqRel);
        flags.set(ppn as usize, 0);
        // Pending blocks of the page lose their entry, so their modules are thrown away when they
        // come back (`State::install_batch`). They have no index yet, hence before the return.
        let page_start = ppn << PAGE_SHIFT;
        let stale: Vec<u64> = state
            .pending
            .range(page_start..page_start + PAGE_SIZE)
            .map(|(&phys_addr, _)| phys_addr)
            .collect();
        for phys_addr in stale {
            state.pending.remove(&phys_addr);
        }
        let Some(indices) = state.page_blocks.remove(&ppn) else {
            return;
        };
        for index in indices {
            shared.valid[index as usize].store(0, Ordering::Release);
            let key = state.blocks[index as usize].key;
            state.block_map.remove(&key);
        }
    }

    /// The block translated for the code at this physical address, if any; `rip` only picks the
    /// jump cache slot.
    pub fn lookup(&mut self, rip: u64, phys_addr: u64) -> Option<u64> {
        let entry = self.shared.jump_cache.get(rip);
        if entry.slot != 0
            && entry.rip == rip
            && entry.phys_addr == phys_addr
            && self.shared.valid[entry.block as usize].load(Ordering::Relaxed) != 0
        {
            return Some(entry.slot as u64);
        }
        let state = self.shared.lock();
        let index = *state.block_map.get(&(phys_addr, NO_SECOND_PAGE))?;
        let slot = state.blocks[index as usize].slot;
        drop(state);
        self.shared.jump_cache.set(
            rip,
            JumpEntry {
                rip,
                phys_addr,
                block: index as u64,
                slot,
                _pad: 0,
            },
        );
        Some(slot as u64)
    }

    /// The straddling block cached for `rip`, if its slot is still good: the block is valid and the
    /// page after `rip`'s is still the physical page it was decoded from.
    pub fn lookup_straddler(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Option<u64> {
        let entry = self.shared.jump_cache.get(rip);
        if entry.phys_addr != phys_addr | STRADDLER_TAG || entry.rip != rip {
            return None;
        }
        let (slot, block) = (entry.slot, entry.block);
        if self.shared.valid[(block & SLOT_BLOCK_MASK) as usize].load(Ordering::Relaxed) == 0 {
            return None;
        }
        let next_page = cpu
            .translate(
                (rip & !PAGE_MASK).wrapping_add(PAGE_SIZE),
                AccessKind::Exec,
                false,
            )
            .ok()?;
        ((next_page >> PAGE_SHIFT) == block >> SLOT_BLOCK_BITS).then_some(slot as u64)
    }

    /// Whether the code at `phys_addr` is known to have no block and to stay interpreted this
    /// time: its heat entry is its own and still below the threshold after counting this run
    /// (which it does, as a cold step). Lets a cold step skip `lookup`'s block map search, which
    /// was most of the cost of the millions of instructions a boot interprets before their block
    /// is compiled. The heat entry is only made or counted when no block was found (`is_hot`
    /// runs after a failed lookup), so it never hides a live block for longer than the
    /// threshold's worth of runs, and even then the block is only interpreted, never wrong.
    fn known_cold(&self, phys_addr: u64) -> bool {
        let threshold = self.shared.compile_threshold.load(Ordering::Relaxed);
        if threshold == 0 {
            return false;
        }
        let mut state = self.shared.lock();
        let heat = &mut state.heat[heat_index(phys_addr)];
        if heat.phys != phys_addr || heat.count.saturating_add(1) >= threshold {
            return false;
        }
        heat.count += 1;
        state.stats.cold_steps += 1;
        true
    }

    /// Whether the block at `phys_addr` has run often enough in the interpreter to be worth
    /// compiling. Counts this run.
    fn is_hot(&self, phys_addr: u64) -> bool {
        let threshold = self.shared.compile_threshold.load(Ordering::Relaxed);
        if threshold == 0 {
            return true;
        }
        let mut state = self.shared.lock();
        let heat = &mut state.heat[heat_index(phys_addr)];
        if heat.phys != phys_addr {
            *heat = Heat {
                phys: phys_addr,
                count: 0,
            };
        }
        if heat.count == NEVER_COMPILE {
            return false;
        }
        heat.count += 1;
        heat.count >= threshold
    }

    /// Where `phys_addr` stands against the pending blocks.
    fn pending_at(&self, phys_addr: u64) -> PendingAt {
        let state = self.shared.lock();
        match state.pending.range(..=phys_addr).next_back() {
            Some((&start, _)) if start == phys_addr => PendingAt::Head,
            Some((&start, pending)) if phys_addr < pending.end => PendingAt::Inside(start),
            _ => PendingAt::Outside,
        }
    }

    /// `State::pump`, flushing the cache when a finished module does not fit.
    fn pump(&mut self, cpu: &mut Cpu, waiting: Option<u64>) {
        let shared = self.shared.clone();
        if shared.lock().pump(&shared, waiting) {
            self.flush(cpu);
        }
    }

    /// A miss in the block map: the block's slot if it can run compiled now. Otherwise the
    /// caller interprets one instruction, and this counts toward compiling the block.
    ///
    /// Any instruction of a pending block checks whether modules are back. Its instructions after
    /// the first run in the interpreter until then without counting heat: if they did, every
    /// instruction of a hot block would reach the threshold while its module is compiled and get
    /// a block of its own. A branch target inside it waits for the outer block's module too, and
    /// a loop there counts toward the outer block's `PATIENCE`.
    fn compile_on_miss(
        &mut self,
        cpu: &mut Cpu,
        rip: u64,
        phys_addr: u64,
    ) -> Result<Option<u64>, Exception> {
        match self.pending_at(phys_addr) {
            PendingAt::Head => {
                self.pump(cpu, Some(phys_addr));
                return Ok(self.lookup(rip, phys_addr));
            }
            // Once the outer block is installed this address counts heat again, next time.
            PendingAt::Inside(start) => {
                self.pump(cpu, Some(start));
                return Ok(None);
            }
            PendingAt::Outside => {}
        }
        if !self.is_hot(phys_addr) {
            return Ok(None);
        }
        self.try_translate(cpu, rip, phys_addr)
    }

    /// Decodes up to `MAX_INSNS` instructions at `rip` (physical `phys_addr`) and compiles them.
    /// Returns the table slot of the block, or `None` when it waits for its module (with a
    /// compile worker) or the host refused the module (it then stays interpreted). `rip` must be
    /// `cpu.rip`: a page-straddling instruction decodes through `Cpu::fetch_decode`, which reads
    /// it.
    fn try_translate(
        &mut self,
        cpu: &mut Cpu,
        rip: u64,
        phys_addr: u64,
    ) -> Result<Option<u64>, Exception> {
        loop {
            // Code can only execute from RAM; reported as a not-present fetch fault.
            if !cpu.ram.contains(phys_addr, 1) {
                return Err(cpu.page_fault(rip, AccessKind::Exec, cpu.cpl == 3, false, false));
            }
            // Makes guest stores to this page take the slow path so `code_written` sees them,
            // before the bytes are read.
            cpu.mark_code(phys_addr);
            let generation = self.shared.invalidations.load(Ordering::Acquire);
            let decoded = decode_block(cpu, rip, phys_addr)?;
            match self.install(decoded, rip, phys_addr, generation, &cpu.page_has_code) {
                Install::Ready(slot) => return Ok(Some(slot as u64)),
                Install::Queued => {
                    // Without a compile worker this installs it at once.
                    self.pump(cpu, None);
                    return Ok(self.lookup(rip, phys_addr));
                }
                Install::Stale => {}
                Install::Full => self.flush(cpu),
                Install::Refused => {
                    let mut state = self.shared.lock();
                    state.stats.compile_failures += 1;
                    state.heat[heat_index(phys_addr)] = Heat {
                        phys: phys_addr,
                        count: NEVER_COMPILE,
                    };
                    return Ok(None);
                }
            }
        }
    }

    /// `try_translate` for tests and callers that want a block: the host refusing is a failure.
    #[cfg(test)]
    pub fn translate(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Result<u64, Exception> {
        Ok(self
            .try_translate(cpu, rip, phys_addr)?
            .expect("the host compiles every generated module"))
    }

    /// The part of `try_translate` under the lock: the checks that make the decoded bytes still
    /// good, then generate the block's code. A straddler is compiled and registered here; any
    /// other block is queued for its module (`State::pump`).
    fn install(
        &mut self,
        decoded: Decoded,
        rip: u64,
        phys_addr: u64,
        generation: u64,
        flags: &CodePages,
    ) -> Install {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        if shared.invalidations.load(Ordering::Acquire) != generation {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        if decoded
            .pages
            .iter()
            .any(|&page| flags.get(page as usize) == 0)
        {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        let key = (phys_addr, decoded.second_page);
        if let Some(&index) = state.block_map.get(&key) {
            let slot = state.blocks[index as usize].slot;
            self.fill_jump_cache(rip, phys_addr, decoded.second_page, slot, index);
            return Install::Ready(slot);
        }
        if state.blocks.len() >= shared.block_limit() {
            return Install::Full;
        }
        let Decoded {
            insns,
            insn_rips,
            end_rip,
            second_page,
            pages,
            ..
        } = decoded;
        let insns: Box<[Insn]> = insns.into_boxed_slice();
        let links: Box<[AtomicU64]> = if second_page == NO_SECOND_PAGE {
            (0..LINKS_PER_BLOCK)
                .map(|_| AtomicU64::new(NO_BLOCK as u64))
                .collect()
        } else {
            Box::new([])
        };
        let compiled = emit_wasm::compile_block(
            &addrs(
                shared.out.as_ptr(),
                shared.chain_slot,
                shared.chain_link_slot,
                &shared.valid,
            ),
            &insns,
            &insn_rips,
            end_rip,
            if links.is_empty() {
                0
            } else {
                links.as_ptr() as usize as u32
            },
        );
        if second_page == NO_SECOND_PAGE {
            let serial = state.next_serial;
            state.next_serial += 1;
            state.pending.insert(
                phys_addr,
                Pending {
                    // `end_rip` is an offset in the page, like every rip of a block.
                    end: (phys_addr & !PAGE_MASK) + end_rip,
                    serial,
                    insns: insns.len() as u32,
                    steps: 0,
                },
            );
            state.queued.push(PendingBlock {
                serial,
                phys_addr,
                pages,
                insns,
                links,
                compiled,
            });
            return Install::Queued;
        }
        let module = wasm_encode::module_of_functions(&[compiled.function()]);
        let Some(slot) = compile_module(&module) else {
            return Install::Refused;
        };
        state.stats.modules += 1;
        let index = state.register(&shared, key, slot, insns, links, pages, &compiled);
        self.fill_jump_cache(rip, phys_addr, second_page, slot, index);
        Install::Ready(slot)
    }

    /// A straddling block's slot is tagged (`STRADDLER_TAG`) so `lookup` never takes it and
    /// `lookup_straddler` finds it after checking the second page.
    fn fill_jump_cache(
        &mut self,
        rip: u64,
        phys_addr: u64,
        second_page: u64,
        slot: u32,
        index: u32,
    ) {
        let entry = if second_page == NO_SECOND_PAGE {
            JumpEntry {
                rip,
                phys_addr,
                block: index as u64,
                slot,
                _pad: 0,
            }
        } else {
            JumpEntry {
                rip,
                phys_addr: phys_addr | STRADDLER_TAG,
                block: index as u64 | (second_page >> PAGE_SHIFT) << SLOT_BLOCK_BITS,
                slot,
                _pad: 0,
            }
        };
        self.shared.jump_cache.set(rip, entry);
    }

    /// Flushes the whole cache, which is full. Nothing runs generated code at this point (a block
    /// is only entered from `step_block`, which is not on the stack).
    fn flush(&mut self, cpu: &mut Cpu) {
        let shared = self.shared.clone();
        cpu.with_exclusive(|_| shared.lock().flush(&shared));
    }
}

impl Cpu {
    /// Runs one block (or, on a miss, translates then runs it) starting at `self.rip`; a block that
    /// is not hot yet is not compiled, and one instruction runs in the interpreter instead.
    ///
    /// The block is a function `(cpu: *mut Cpu) -> u64` in the function table, returning
    /// `(retired_insns << helpers::EXIT_COUNT_SHIFT) | status` exactly as the native blocks do
    /// (see `native.rs`, whose accounting of `icount` this repeats).
    pub fn step_block(&mut self) -> Result<(), Exception> {
        let rip = self.rip;
        let phys_addr = self.translate(rip, AccessKind::Exec, false)?;
        // `translate` needs `&mut Cpu`, hence the take. The Jit is put back before the block runs
        // because stores inside it reach `Cpu::code_written`, which needs `self.jit` to invalidate.
        let mut jit = self.jit.take().unwrap();
        let cached = match jit.lookup_straddler(self, rip, phys_addr) {
            Some(slot) => Some(slot),
            // Already counted as a cold step.
            None if jit.known_cold(phys_addr) => {
                self.jit = Some(jit);
                self.last_block_insns = 1;
                return self.step();
            }
            None => jit.lookup(rip, phys_addr),
        };
        let slot = match cached {
            Some(slot) => Some(slot),
            None => match jit.compile_on_miss(self, rip, phys_addr) {
                Ok(slot) => slot,
                Err(exception) => {
                    self.jit = Some(jit);
                    return Err(exception);
                }
            },
        };
        if slot.is_none() {
            jit.shared.lock().stats.cold_steps += 1;
        }
        self.jit = Some(jit);
        let Some(slot) = slot else {
            // `Cpu::run` already counted this instruction, as `step` expects.
            self.last_block_insns = 1;
            return self.step();
        };
        // SAFETY: the host put a function of this type in the slot (`CompiledBlock::function`).
        let run_block: extern "C" fn(*mut Cpu) -> u64 =
            unsafe { core::mem::transmute(slot as usize) };
        let exit_code = run_block(self as *mut Cpu);
        let done = exit_code >> helpers::EXIT_COUNT_SHIFT;
        self.last_block_insns = done;
        // Set by `helper` before an instruction at block index k > 0 that may read the clock: it
        // already added k to `icount`, which must not be added twice.
        let prepaid_icount = core::mem::take(&mut self.prepaid_icount);
        self.block_exit_requested = false;
        match exit_code & helpers::EXIT_STATUS_MASK {
            helpers::STATUS_EXCEPTION => {
                self.icount += done - prepaid_icount;
                Err(self.pending.take().expect("pending exception"))
            }
            _ => {
                self.icount += done.saturating_sub(1) - prepaid_icount;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The cache against real generated modules. They need a host that compiles them, which on the
    //! test target is `hack/wasm/run-wasi.mjs`; the differential tests in `diff_*.rs` skip
    //! silently when `Jit::new` is `None`, so these also pin that it is not.

    use super::*;
    use crate::testkit::*;

    // add rax, rcx; hlt
    const ADD_THEN_HLT: [u8; 4] = [0x48, 0x01, 0xc8, 0xf4];
    // mov ecx, 100; l: sub ecx, 1; jnz l; hlt
    const COUNTDOWN_LOOP: [u8; 11] = [
        0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
    ];

    #[test]
    fn the_host_compiles_and_runs_a_generated_module() {
        assert!(probe());
        assert!(Jit::new(true).is_some());
    }

    #[test]
    fn a_harness_with_the_jit_really_runs_translated_blocks() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        harness.cpu.gpr[0] = 40;
        harness.cpu.gpr[1] = 2;
        assert_eq!(harness.run(1000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[0], 42);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 1);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.inlined, 1, "the add is lowered, the hlt is not");
        assert_eq!(stats.cold_steps, 0);
    }

    #[test]
    fn a_loop_is_translated_once_and_run_from_the_cache() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 3, "setup, loop body, hlt");
        assert_eq!(stats.inlined, stats.total - 1);
    }

    #[test]
    fn cold_blocks_are_interpreted_until_they_run_often_enough() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        harness.cpu.jit.as_ref().unwrap().set_compile_threshold(10);
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(stats.cold_steps >= 18, "{stats:?}");
        assert_eq!(stats.translated, 1, "only the loop head got hot: {stats:?}");
    }

    #[test]
    fn invalidating_a_page_makes_its_blocks_untranslated_again() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        let ppn = CODE_BASE >> PAGE_SHIFT;
        let mut jit = harness.cpu.jit.take().unwrap();
        let slot = jit
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(slot));
        assert!(jit.block_is_valid(0));
        jit.invalidate_page(ppn, &harness.cpu.page_has_code);
        assert!(!jit.block_is_valid(0));
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        let again = jit
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(jit.stats().translated, 2);
        assert_ne!(again, 0);
    }

    #[test]
    fn a_full_cache_is_flushed_and_translation_carries_on() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        harness.cpu.jit.as_ref().unwrap().set_block_limit(1);
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(stats.resets >= 1, "{stats:?}");
        assert!(harness.cpu.jit.as_ref().unwrap().block_count() <= 1);
    }

    /// Runs the loop the way `Cpu::run` does and counts the calls to `step_block`.
    fn step_block_calls_for(jit: Jit, event_budget: u64) -> (u64, Cpu) {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        harness.cpu.jit = Some(Box::new(jit));
        harness.cpu.next_event = event_budget;
        let mut calls = 0;
        while !harness.cpu.halted && calls < 1000 {
            harness.cpu.interrupt_shadow = false;
            harness.cpu.icount += 1;
            harness.cpu.step_block().unwrap();
            calls += 1;
            if harness.cpu.icount >= harness.cpu.next_event {
                harness.cpu.next_event = u64::MAX;
            }
        }
        (calls, harness.cpu)
    }

    #[test]
    fn chained_blocks_continue_without_returning_to_the_run_loop() {
        let _ = Harness::new(&[0xf4]);
        let (chained_calls, chained) = step_block_calls_for(Jit::new(true).unwrap(), u64::MAX);
        let (unchained_calls, unchained) = step_block_calls_for(Jit::new(false).unwrap(), u64::MAX);
        assert!(chained.halted && unchained.halted);
        assert_eq!(
            chained.icount, unchained.icount,
            "the accounting is the same"
        );
        assert_eq!(chained.gpr, unchained.gpr);
        assert!(
            unchained_calls > 100,
            "one call per block: {unchained_calls}"
        );
        assert!(
            chained_calls < 5,
            "the loop runs inside one call: {chained_calls}"
        );
    }

    #[test]
    fn a_chain_stops_at_the_event_budget_so_the_run_loop_can_service_events() {
        // 11 instructions of setup, then 2 per iteration: the budget ends the chain mid-loop, the
        // run loop takes over (and here lifts the budget), and the count stays exact.
        let (calls, chained) = step_block_calls_for(Jit::new(true).unwrap(), 50);
        let (_, reference) = step_block_calls_for(Jit::new(false).unwrap(), u64::MAX);
        assert!(calls >= 2, "the budget interrupted the chain: {calls}");
        assert!(chained.halted);
        assert_eq!(chained.icount, reference.icount);
    }

    /// Where the countdown loop's body starts, after the 5-byte `mov ecx, 100`.
    const LOOP_BODY: u64 = CODE_BASE + 5;
    /// mov + 100 * (sub, jnz) + hlt.
    const COUNTDOWN_LOOP_INSNS: u64 = 202;

    /// Runs the countdown loop to its `hlt` and puts the CPU back at its start.
    fn run_countdown_and_rewind(harness: &mut Harness) {
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        harness.cpu.halted = false;
        harness.cpu.rip = CODE_BASE;
    }

    #[test]
    fn a_same_page_exit_links_to_the_block_the_stub_entered() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        assert_eq!(harness.run(10_000), Stop::Halted);
        let jit = harness.cpu.jit.as_ref().unwrap();
        let body = jit.block_index(LOOP_BODY).unwrap();
        // The jnz back to the body went through the stub once and is linked since; its
        // fall-through ran once, before the hlt block existed, so it is still empty.
        assert_eq!(
            jit.link_cells(body),
            [(body, jit.slot_of(body)), (NO_BLOCK, 0)]
        );
        let setup = jit.block_index(CODE_BASE).unwrap();
        assert_eq!(
            jit.link_cells(setup)[0],
            (NO_BLOCK, 0),
            "the body did not exist yet"
        );
    }

    #[test]
    fn a_linked_chain_stops_at_the_event_budget_with_an_exact_count() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        run_countdown_and_rewind(&mut harness);
        let warm = harness.cpu.jit.take().unwrap();
        let body = warm.block_index(LOOP_BODY).unwrap();
        assert_eq!(warm.link_cells(body)[0].0, body, "the loop is linked");
        let (calls, linked) = step_block_calls_for(warm.sibling(), 50);
        let (_, reference) = step_block_calls_for(Jit::new(false).unwrap(), u64::MAX);
        assert!(calls >= 2, "the budget interrupted the chain: {calls}");
        assert!(linked.halted);
        assert_eq!(linked.icount, reference.icount);
        assert_eq!(linked.icount, COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn an_invalidated_destination_is_relinked_to_its_new_translation() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        run_countdown_and_rewind(&mut harness);
        let old_body = harness.cpu.jit.as_ref().unwrap().block_index(LOOP_BODY);
        let flags = harness.cpu.page_has_code.clone();
        harness
            .cpu
            .jit
            .as_ref()
            .unwrap()
            .invalidate_page(CODE_BASE >> PAGE_SHIFT, &flags);
        run_countdown_and_rewind(&mut harness);
        let jit = harness.cpu.jit.as_ref().unwrap();
        let body = jit.block_index(LOOP_BODY).unwrap();
        assert_ne!(Some(body), old_body);
        assert_eq!(jit.link_cells(body)[0], (body, jit.slot_of(body)));
    }

    // mov ecx, 3; xor eax, eax; again: mov dl, cl; mov [rip + tgt + 2], dl; jmp tgt;
    // tgt: add eax, 0x7f; dec ecx; jnz again; hlt
    // Each pass patches the immediate of the `add` to ecx before jumping to it: 3 + 2 + 1.
    const PATCH_THEN_JUMP: [u8; 25] = [
        0xb9, 0x03, 0x00, 0x00, 0x00, 0x31, 0xc0, 0x88, 0xca, 0x88, 0x15, 0x04, 0x00, 0x00, 0x00,
        0xeb, 0x00, 0x83, 0xc0, 0x7f, 0xff, 0xc9, 0x75, 0xef, 0xf4,
    ];

    #[test]
    fn a_store_into_the_linked_destination_never_runs_its_stale_code() {
        // The jmp's exit to the patched block is linked after the first pass.
        let mut harness = Harness::new(&PATCH_THEN_JUMP);
        assert!(harness.enable_jit());
        assert_eq!(harness.run(1000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[0], 6);
    }

    #[test]
    fn a_flush_with_linked_blocks_starts_over_cleanly() {
        let mut harness = Harness::new(&COUNTDOWN_LOOP);
        assert!(harness.enable_jit());
        run_countdown_and_rewind(&mut harness);
        let jit = harness.cpu.jit.as_ref().unwrap();
        // The next translation finds the cache full and flushes every block and its cells.
        jit.set_block_limit(jit.block_count());
        let flags = harness.cpu.page_has_code.clone();
        jit.invalidate_page(CODE_BASE >> PAGE_SHIFT, &flags);
        run_countdown_and_rewind(&mut harness);
        let jit = harness.cpu.jit.as_ref().unwrap();
        assert!(jit.stats().resets >= 1);
        assert_eq!(harness.cpu.icount, 2 * COUNTDOWN_LOOP_INSNS);
    }

    const STRADDLER_VIRTUAL_PAGE: u64 = 0x0000_2000_0000_0000;
    /// Where `l: sub ecx, <imm>` starts: its last byte, the immediate, is the first byte of the
    /// next page, followed by `jnz l; hlt`.
    const STRADDLING_LOOP: u64 = STRADDLER_VIRTUAL_PAGE + PAGE_SIZE - 2;
    /// The `jnz l` after it.
    const STRADDLING_LOOP_JNZ: u64 = STRADDLER_VIRTUAL_PAGE + PAGE_SIZE + 1;

    /// Writes the second-page half of the loop (`<imm>; jnz l; hlt`) at physical `page`.
    fn write_loop_tail(harness: &mut Harness, page: u64, decrement: u8) {
        // jnz rel8 back to the sub: from the end of the jnz (page + 3) to page - 2
        harness.write(page, &[decrement, 0x75, 0xfb, 0xf4]);
    }

    /// A harness whose code is that loop, decrementing by 1, over two virtual pages mapped to
    /// physical `first` and `second`.
    fn harness_with_a_straddling_loop(
        first: u64,
        second: u64,
    ) -> (Harness, crate::cpu::test_support::PageTables) {
        use crate::cpu::test_support::{PTE_PRESENT_WRITABLE_USER, PageTables};
        let mut harness = Harness::new(&[0xf4]);
        assert!(harness.enable_jit());
        let mut tables = PageTables::new(&mut harness.cpu);
        for (page, phys) in [(0, first), (PAGE_SIZE, second)] {
            tables.map_4k(
                &mut harness.cpu,
                STRADDLER_VIRTUAL_PAGE + page,
                phys,
                PTE_PRESENT_WRITABLE_USER,
            );
        }
        // sub ecx, imm8 is 83 e9 ib
        harness.write(first + PAGE_SIZE - 2, &[0x83, 0xe9]);
        write_loop_tail(&mut harness, second, 1);
        harness.cpu.rip = STRADDLING_LOOP;
        (harness, tables)
    }

    /// Calls `step_block` the way `Cpu::run` does until the guest halts or `budget` instructions
    /// ran; returns the number of calls.
    fn step_block_calls_until(harness: &mut Harness, budget: u64) -> u64 {
        let start = harness.cpu.icount;
        harness.cpu.next_event = start + budget;
        let mut calls = 0;
        while !harness.cpu.halted && harness.cpu.icount - start < budget {
            harness.cpu.interrupt_shadow = false;
            harness.cpu.icount += 1;
            harness.cpu.step_block().unwrap();
            calls += 1;
        }
        calls
    }

    #[test]
    fn the_chain_stub_enters_a_straddling_block_whose_second_page_is_unchanged() {
        let (mut harness, _tables) =
            harness_with_a_straddling_loop(SCRATCH, SCRATCH + 2 * PAGE_SIZE);
        harness.cpu.gpr[1] = 100;
        let calls = step_block_calls_until(&mut harness, 10_000);
        assert!(harness.cpu.halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        // sub, jnz per iteration, then hlt
        assert_eq!(harness.cpu.icount, 201);
        // Each block's first visit goes back to the run loop to be translated; after that the
        // jnz's exit to the straddler goes through the stub, not the run loop.
        assert!(calls < 10, "{calls} calls");
    }

    #[test]
    fn the_chain_stub_does_not_enter_a_straddler_whose_second_page_moved() {
        use crate::cpu::test_support::PTE_PRESENT_WRITABLE_USER;
        let (first, second, elsewhere) =
            (SCRATCH, SCRATCH + 2 * PAGE_SIZE, SCRATCH + 3 * PAGE_SIZE);
        let (mut harness, mut tables) = harness_with_a_straddling_loop(first, second);
        harness.cpu.gpr[1] = 1000;
        step_block_calls_until(&mut harness, 50);
        assert!(!harness.cpu.halted);
        // Remap the second page to a copy whose immediate is 2: the straddler cached in the jump
        // cache was decoded with 1, and must not run again.
        write_loop_tail(&mut harness, elsewhere, 2);
        tables.map_4k(
            &mut harness.cpu,
            STRADDLER_VIRTUAL_PAGE + PAGE_SIZE,
            elsewhere,
            PTE_PRESENT_WRITABLE_USER,
        );
        harness.cpu.flush_tlb();
        // Refill the exec TLB for the first page, so the stub gets as far as the straddler check
        // when the jnz (translated afresh from the new page) leaves for the loop head.
        harness
            .cpu
            .translate(STRADDLER_VIRTUAL_PAGE, AccessKind::Exec, false)
            .unwrap();
        harness.cpu.rip = STRADDLING_LOOP_JNZ;
        harness.cpu.gpr[1] = 100;
        let start = harness.cpu.icount;
        step_block_calls_until(&mut harness, 10_000);
        assert!(harness.cpu.halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        // the jnz, 50 iterations of sub 2 and jnz, then hlt; with the stale straddler's sub 1 it
        // would take 100
        assert_eq!(harness.cpu.icount - start, 1 + 2 * 50 + 1);
    }

    /// Instructions of the one block `code` makes that were lowered inline.
    fn inlined_in(code: &[u8]) -> u64 {
        let mut harness = Harness::new(code);
        assert!(harness.enable_jit());
        let mut jit = harness.cpu.jit.take().unwrap();
        jit.translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        jit.stats().inlined
    }

    #[test]
    fn imul_64_is_inline_only_when_its_flags_are_overwritten_and_rotates_always_are() {
        // sbb eax, eax; adc rcx, 7; hlt: adc and sbb are inline whatever their flags
        assert_eq!(inlined_in(&[0x19, 0xc0, 0x48, 0x83, 0xd1, 0x07, 0xf4]), 2);
        // imul rax, rcx; add rdx, 1; hlt
        assert_eq!(
            inlined_in(&[0x48, 0x0f, 0xaf, 0xc1, 0x48, 0x83, 0xc2, 0x01, 0xf4]),
            2
        );
        // imul rax, rcx; hlt: the successor may read the flags
        assert_eq!(inlined_in(&[0x48, 0x0f, 0xaf, 0xc1, 0xf4]), 0);
        // rol rax, 13; ror ecx, cl; sub rdx, rax; hlt
        assert_eq!(
            inlined_in(&[0x48, 0xc1, 0xc0, 0x0d, 0xd3, 0xc9, 0x48, 0x29, 0xc2, 0xf4]),
            3
        );
        // rol rax, 13; adc rdx, rax; hlt: adc reads the CF the rol merged into the flags
        assert_eq!(
            inlined_in(&[0x48, 0xc1, 0xc0, 0x0d, 0x48, 0x11, 0xc2, 0xf4]),
            2
        );
        // rol rbx, 25; jmp next; next: hlt: the successor may read the flags
        assert_eq!(inlined_in(&[0x48, 0xc1, 0xc3, 0x19, 0xeb, 0x00, 0xf4]), 2);
    }

    // add rax, 1; hlt
    const ADD_ONE_THEN_HLT: [u8; 5] = [0x48, 0x83, 0xc0, 0x01, 0xf4];

    fn harness_with_a_compile_worker(code: &[u8], finish_after_polls: Option<u32>) -> Harness {
        let mut harness = Harness::new(code);
        assert!(harness.enable_jit());
        harness
            .cpu
            .jit
            .as_ref()
            .unwrap()
            .use_fake_compiler(finish_after_polls);
        harness
    }

    #[test]
    fn with_a_compile_worker_a_hot_block_is_interpreted_until_its_module_is_ready() {
        let mut harness = harness_with_a_compile_worker(&ADD_ONE_THEN_HLT, None);
        let run_again = |harness: &mut Harness| {
            harness.cpu.halted = false;
            harness.cpu.rip = CODE_BASE;
            assert_eq!(harness.run(100), Stop::Halted);
        };
        assert_eq!(harness.run(100), Stop::Halted);
        run_again(&mut harness);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 0, "{stats:?}");
        assert_eq!(stats.cold_steps, 4, "add and hlt, twice: {stats:?}");
        harness.cpu.jit.as_ref().unwrap().finish_compiles();
        run_again(&mut harness);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 1, "{stats:?}");
        assert_eq!(stats.cold_steps, 4, "the third run is compiled: {stats:?}");
        assert_eq!(harness.cpu.gpr[0], 3);
    }

    #[test]
    fn a_block_whose_page_is_written_while_its_module_is_compiled_is_never_installed() {
        let mut harness = harness_with_a_compile_worker(&ADD_ONE_THEN_HLT, None);
        assert_eq!(harness.run(100), Stop::Halted);
        assert_eq!(harness.cpu.gpr[0], 1);
        // add rax, 5, and what a guest store to the code makes of it
        harness.write(CODE_BASE + 3, &[5]);
        harness.cpu.code_written((CODE_BASE >> PAGE_SHIFT) as usize);
        harness.cpu.jit.as_ref().unwrap().finish_compiles();
        harness.cpu.halted = false;
        harness.cpu.rip = CODE_BASE;
        harness.cpu.gpr[0] = 0;
        assert_eq!(harness.run(100), Stop::Halted);
        assert_eq!(
            harness.cpu.gpr[0], 5,
            "the module of the old bytes did not run"
        );
        assert!(harness.cpu.jit.as_ref().unwrap().stats().abandoned >= 1);
    }

    #[test]
    fn code_the_guest_patches_while_its_module_is_compiled_never_runs_stale() {
        let mut harness = harness_with_a_compile_worker(&PATCH_THEN_JUMP, Some(1));
        assert_eq!(harness.run(1000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[0], 6);
        assert!(harness.cpu.jit.as_ref().unwrap().stats().translated >= 1);
    }

    #[test]
    fn a_flush_forgets_the_modules_being_compiled() {
        let mut harness = harness_with_a_compile_worker(&COUNTDOWN_LOOP, None);
        run_countdown_and_rewind(&mut harness);
        let mut jit = harness.cpu.jit.take().unwrap();
        assert!(!jit.pending_starts().is_empty());
        jit.flush(&mut harness.cpu);
        assert!(jit.pending_starts().is_empty());
        // The modules from before the flush are done now, and must not be installed.
        jit.finish_compiles();
        harness.cpu.jit = Some(jit);
        run_countdown_and_rewind(&mut harness);
        // Nothing submitted since is done, so whatever was installed is what this thread
        // stopped waiting for.
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.modules, stats.impatient, "{stats:?}");
        assert!(stats.abandoned >= 1, "{stats:?}");
        assert_eq!(harness.cpu.icount, 2 * COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn a_pending_block_that_keeps_running_does_not_wait_for_the_worker() {
        // l: sub ecx, 1; jnz l; hlt
        const SELF_LOOP: [u8; 6] = [0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4];
        let mut harness = harness_with_a_compile_worker(&SELF_LOOP, None);
        harness.cpu.gpr[1] = 1000;
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.impatient, 1, "{stats:?}");
        // sub and jnz in the interpreter until the loop's block has waited its patience out
        assert!(stats.cold_steps <= 2 * (PATIENCE as u64 + 1), "{stats:?}");
    }

    #[test]
    fn blocks_that_get_hot_while_a_module_is_out_share_the_next_one() {
        // jmp $+2; jmp $+2; jmp $+2; hlt: four blocks, the first goes out alone
        const JUMPS: [u8; 7] = [0xeb, 0x00, 0xeb, 0x00, 0xeb, 0x00, 0xf4];
        let mut harness = harness_with_a_compile_worker(&JUMPS, None);
        let run_again = |harness: &mut Harness| {
            harness.cpu.jit.as_ref().unwrap().finish_compiles();
            harness.cpu.halted = false;
            harness.cpu.rip = CODE_BASE;
            assert_eq!(harness.run(100), Stop::Halted);
        };
        assert_eq!(harness.run(100), Stop::Halted);
        run_again(&mut harness);
        run_again(&mut harness);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, 4, "{stats:?}");
        assert_eq!(stats.modules, 2, "{stats:?}");
    }

    #[test]
    fn a_compile_worker_lowers_the_compile_threshold() {
        let mut harness = harness_with_a_compile_worker(&COUNTDOWN_LOOP, None);
        let jit = harness.cpu.jit.as_ref().unwrap();
        jit.set_compile_threshold(COMPILE_THRESHOLD);
        // The loop body's 100 runs make it hot, and its module goes to the worker.
        run_countdown_and_rewind(&mut harness);
        let jit = harness.cpu.jit.as_ref().unwrap();
        assert_eq!(jit.stats().submitted, 1);
        assert_eq!(
            jit.shared.compile_threshold.load(Ordering::Relaxed),
            COMPILE_THRESHOLD_WITH_WORKER
        );
    }

    #[test]
    fn a_loop_inside_a_pending_block_installs_it_once_the_worker_is_done() {
        // The loop body lies inside the setup block: running it must still look for the setup
        // block's module, so the loop gets a block of its own and stops being interpreted.
        let mut harness = harness_with_a_compile_worker(&COUNTDOWN_LOOP, Some(1));
        assert_eq!(harness.run(10_000), Stop::Halted);
        assert_eq!(harness.cpu.gpr[1], 0);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(stats.translated >= 2, "{stats:?}");
        assert!(stats.cold_steps < COUNTDOWN_LOOP_INSNS / 2, "{stats:?}");
    }

    #[test]
    fn a_loop_inside_a_pending_block_does_not_wait_for_the_worker_forever() {
        let mut harness = harness_with_a_compile_worker(&COUNTDOWN_LOOP, None);
        assert_eq!(harness.run(10_000), Stop::Halted);
        let stats = harness.cpu.jit.as_ref().unwrap().stats();
        assert!(
            stats.impatient >= 2,
            "the setup block, then the loop's: {stats:?}"
        );
        assert!(stats.cold_steps < COUNTDOWN_LOOP_INSNS / 2, "{stats:?}");
    }

    #[test]
    fn code_inside_a_block_being_compiled_does_not_get_hot_on_its_own() {
        // The loop body (and its jnz) lie inside the setup block, which is waiting for its
        // module: running them in the interpreter must not make blocks of their own.
        let mut harness = harness_with_a_compile_worker(&COUNTDOWN_LOOP, None);
        run_countdown_and_rewind(&mut harness);
        let jit = harness.cpu.jit.as_ref().unwrap();
        let jnz = LOOP_BODY + 3;
        assert!(
            !jit.pending_starts().contains(&jnz),
            "{:x?}",
            jit.pending_starts()
        );
        assert_eq!(jit.block_index(jnz), None);
    }

    #[test]
    fn siblings_share_one_cache() {
        let mut harness = Harness::new(&ADD_THEN_HLT);
        assert!(harness.enable_jit());
        let mut first = harness.cpu.jit.take().unwrap();
        let mut second = first.sibling();
        let slot = first
            .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
            .unwrap();
        assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(slot));
        assert_eq!(
            second
                .translate(&mut harness.cpu, CODE_BASE, CODE_BASE)
                .unwrap(),
            slot
        );
        assert_eq!(first.stats().translated, 1);
    }
}
