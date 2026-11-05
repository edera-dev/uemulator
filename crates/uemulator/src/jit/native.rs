use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use super::arch;
use super::blocks::{
    BlockKey, Decoded, JUMP_CACHE_BITS, NO_SECOND_PAGE, PAGE_MASK, PAGE_SHIFT, PAGE_SIZE,
    SLOT_BLOCK_BITS, SLOT_BLOCK_MASK, STRADDLER_TAG, decode_block, jump_cache_hash,
};
use super::helpers;
use crate::cpu::Cpu;
use crate::cpu::decode::Insn;
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;
use crate::ram::CodePages;
use crate::sys::ExecMem;

/// The code buffer is a few segments, each starting with its own copy of the prologue (entry
/// trampoline, exit/chain stubs, memory thunks). On aarch64 every block reaches those with a
/// direct `b`/`bl` (`Asm::b_abs` asserts on out-of-range), which spans only +-128 MiB, so a
/// segment must stay below that; several of them make a buffer big enough to hold everything a
/// guest boot translates, instead of throwing all code away every 128 MiB and translating the
/// hot parts again (the kernel plus an installer userspace is ~500k blocks, ~140 MiB).
const SEGMENT_SIZE: usize = 96 << 20;
const SEGMENT_COUNT: usize = 5;
const CODE_SIZE: usize = SEGMENT_SIZE * SEGMENT_COUNT;
/// Free space required before translating a block; below it the whole cache is flushed. Assumed to
/// bound the code of one block (`MAX_INSNS` instructions); `ExecMem::write_bytes` asserts if not.
const CODE_HEADROOM: usize = 1 << 20;
const BLOCK_ALIGN: usize = 16;
/// The prologue is padded to a multiple of this before the first block (64 is a common cache
/// line size; the exact reason is not documented).
const HEADER_ALIGN: usize = 64;

/// Most blocks one code buffer can hold, which is what sizes the validity table. The smallest
/// block is a few dozen bytes, so a full buffer (`CODE_SIZE`) holds fewer than this.
const MAX_BLOCKS: usize = 1 << 24;

const _: () = assert!(MAX_BLOCKS as u64 <= 1 << SLOT_BLOCK_BITS);

/// One slot of the direct-mapped jump cache. `#[repr(C)]` with a power-of-two size because the
/// chain stub in `arch_*.rs` indexes the table and reads `rip`, `phys_addr` and `code_addr` at
/// these offsets (`JUMP_ENTRY_*`). An empty slot has `code_addr == 0`, which the stub treats as
/// a miss.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct JumpEntry {
    pub rip: u64,
    pub phys_addr: u64,
    pub code_addr: u64,
    /// Index of the block in `State::blocks`; meaningless while `code_addr` is 0. The chain stub
    /// and `lookup` check it against the validity table, which is how an invalidated block's slots
    /// stop being usable without finding them (the slot is keyed by virtual rip, the block by
    /// physical address).
    block: u64,
}

// Read by the chain stubs in `arch_a64.rs` and `arch_x64.rs`.
pub(crate) const JUMP_ENTRY_SHIFT: u32 = std::mem::size_of::<JumpEntry>().trailing_zeros();
const _: () = assert!(std::mem::size_of::<JumpEntry>().is_power_of_two());
pub(crate) const JUMP_ENTRY_PHYS_OFFSET: u32 = std::mem::offset_of!(JumpEntry, phys_addr) as u32;
pub(crate) const JUMP_ENTRY_CODE_OFFSET: u32 = std::mem::offset_of!(JumpEntry, code_addr) as u32;
pub(crate) const JUMP_ENTRY_BLOCK_OFFSET: u32 = std::mem::offset_of!(JumpEntry, block) as u32;

/// A patchable slot in a compiled block: the exit to the fixed guest address `target_rip` (in the
/// block's own page) is preceded by a nop of `arch::link_bytes` size that becomes a direct jump to
/// the successor's `link_addr` once it exists.
pub(crate) struct LinkSite {
    /// Offset of the slot inside the block's code.
    pub code_offset: u32,
    pub target_rip: u64,
}

/// What `arch::compile_block` produces.
pub(crate) struct CompiledBlock {
    pub code: Vec<u8>,
    /// Guest instructions lowered inline.
    pub inlined: u32,
    /// Offset of the entry the run loop and the chain stub use. Everything before it is the
    /// `link_addr` entry's checks (`icount` against `next_event`, `block_exit_requested`), which
    /// those two callers have already done; a direct link skips the stub and so must do them.
    pub body_offset: u32,
    pub links: Vec<LinkSite>,
    /// Flag records the block leaves out because a later instruction overwrites them.
    pub flag_records_skipped: u32,
}

/// A slot in the code buffer and the block it should jump to: the physical address of the
/// destination, which is its `BlockKey` (links stay within a page, so never a straddling block).
#[derive(Clone)]
struct LinkedSite {
    /// Offset into `ExecMem` of the slot (the instruction or jump `arch::LinkSite` describes).
    exec_offset: usize,
    destination: u64,
}

/// One translated block. Blocks are only ever added; an invalidated block keeps its slot (cleared
/// in `SharedJit::valid`) until the next flush, so indices held in `page_blocks` and the jump
/// caches stay stable.
pub(crate) struct Block {
    key: BlockKey,
    /// The body entry (run loop, chain stub, jump cache).
    code_addr: u64,
    /// The first byte of the block: the entry direct links jump to.
    link_addr: u64,
    /// Generated code calls `helper_exec` with raw pointers into this slice, so it must outlive the
    /// code. Heap storage keeps the addresses stable when `blocks` reallocates.
    _insns: Box<[Insn]>,
}

/// A vCPU's handle on the code cache. The cache itself (`SharedJit`: code buffer, block table,
/// links) belongs to the machine and is shared by every handle (`sibling`), so a page of guest
/// code is translated once however many vCPUs run it. What stays per vCPU is the jump cache,
/// whose three-word entries cannot be read while another thread writes them.
///
/// The buffer is a single read-write-execute mapping (`MAP_JIT` on macOS, never flipped with
/// mprotect; `docs/plans/06-jit.md` describes an RW then RX path for Linux that the code does not
/// follow). All writes must go through `ExecMem`, which toggles per-thread write protection on
/// Apple Silicon and flushes the instruction cache.
pub(crate) struct Jit {
    shared: Arc<SharedJit>,
    /// Direct-mapped cache indexed by `jump_cache_hash(rip)`. Both `lookup` and the chain stub
    /// read it; the stub gets its address from `jump_cache_base` through the trampoline, so it is
    /// allocated once and never resized.
    jump_cache: Box<[JumpEntry]>,
    /// `SharedJit::epoch` when this handle last cleared its jump cache.
    seen_epoch: u64,
}

/// What every handle shares. Blocks and links are added and patched under `state`; the chain
/// stubs and direct links read them without it.
struct SharedJit {
    exec_mem: ExecMem,
    /// Host address of the prologue trampoline at offset 0 of the buffer; see `step_block`.
    enter_trampoline: u64,
    /// The addresses each segment's blocks are compiled against (its own prologue copy).
    segment_addrs: Vec<arch::Addrs>,
    /// Offset of the first block, just past the prologue. A flush rewinds to here.
    blocks_start: usize,
    /// Per block index: 1 while the block may run, 0 once its page was written (or after a
    /// flush). Read by the chain stub through the address baked into the prologue, which is how
    /// an invalidated block's jump cache slots in every vCPU stop being usable without finding
    /// them (a slot is keyed by virtual rip, a block by physical address).
    valid: Box<[AtomicU8]>,
    /// Counts flushes. A handle whose `seen_epoch` is behind clears its jump cache before it
    /// uses it: the blocks its entries name are gone and their code is about to be overwritten.
    epoch: AtomicU64,
    /// Counts `invalidate_page` calls. A translation decodes outside the lock, so it compares
    /// this before and after: a write to the page it read (which may have landed between its read
    /// and its insert, where `invalidate_page` found nothing to drop) shows up as a change.
    invalidations: AtomicU64,
    /// Generate guest loads and stores with x86 ordering (acquire/release), for vCPUs that run
    /// beside others on their own threads. Only blocks compiled after it is set have it.
    ordered_memory: AtomicBool,
    /// Test knob: the number of blocks after which the buffer counts as full, so a test can make
    /// every vCPU hit a flush over and over without translating hundreds of thousands of blocks.
    /// Per cache and not global, so tests running side by side do not flush each other's.
    #[cfg(test)]
    test_block_limit: std::sync::atomic::AtomicUsize,
    /// Test knob: how long `translate` waits between reading the guest's bytes and taking the
    /// lock, so a test can make another vCPU's write land in the window `Jit::install` closes.
    #[cfg(test)]
    test_install_pause_micros: AtomicU64,
    state: Mutex<State>,
}

/// The block table and the bump pointer, behind `SharedJit::state`.
struct State {
    /// Bump pointer into `exec_mem`.
    used_bytes: usize,
    blocks: Vec<Block>,
    /// Every live block by `BlockKey`. The jump caches are direct-mapped and much smaller than the
    /// working set of a boot (hundreds of thousands of blocks), so without this a colliding rip
    /// would evict a block and the next visit would translate it all over again.
    block_map: HashMap<BlockKey, u32>,
    /// Physical page number -> blocks that decoded bytes from it. A block that crosses a page is
    /// listed under both.
    page_blocks: HashMap<u64, Vec<u32>>,
    /// Slots patched to jump into a block (by its index), so invalidating the block can put them
    /// back to falling through to the chain stub.
    incoming: HashMap<u32, Vec<LinkedSite>>,
    /// Unpatched slots by the physical address of the block they want: the destination was not
    /// translated yet, or was invalidated since.
    pending: HashMap<u64, Vec<LinkedSite>>,
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
    /// Times the whole code buffer was flushed because it was full.
    pub resets: u64,
    /// Block-exit slots patched into a direct jump, including relinks after an invalidation.
    pub linked: u64,
    /// Slots put back to the chain stub because their destination was invalidated.
    pub unlinked: u64,
    /// Lazy-flag records left out of compiled blocks as dead.
    pub flag_records_skipped: u64,
    /// Decodes thrown away because the page was written before the block was registered.
    pub stale_decodes: u64,
}

// Test hooks: each runs once, on the calling thread, with the CPU that is translating. The first
// is the window between `translate` reading the guest's bytes and registering the block; the
// second is between `mark_code` (which found the page already flagged and returned) and the
// read of `SharedJit::invalidations`.
#[cfg(test)]
type TranslateHook = std::cell::RefCell<Option<Box<dyn FnOnce(&mut Cpu)>>>;
#[cfg(test)]
thread_local! {
    pub(crate) static BEFORE_INSTALL: TranslateHook = const { std::cell::RefCell::new(None) };
    pub(crate) static AFTER_MARK: TranslateHook = const { std::cell::RefCell::new(None) };
}

pub(crate) const fn jump_cache_mask() -> u64 {
    (1 << JUMP_CACHE_BITS) - 1
}

/// What `Jit::install` found.
enum Install {
    Ready(u64),
    /// A page the decode read was written since: decode again.
    Stale,
    /// No room for another block: flush the buffer, then decode again.
    Full,
}

impl SharedJit {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while the lock was held left a consistent table (every step either completes or
        // is not visible), and the vCPU that panicked is gone: carry on.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl State {
    /// Whether the next block fits, moving on to the next segment when this one is used up.
    fn make_room(&mut self, shared: &SharedJit) -> bool {
        if self.blocks.len() >= MAX_BLOCKS {
            return false;
        }
        #[cfg(test)]
        if self.blocks.len() >= shared.test_block_limit.load(Ordering::Relaxed) {
            return false;
        }
        let segment = self.used_bytes / SEGMENT_SIZE;
        if self.used_bytes + CODE_HEADROOM <= (segment + 1) * SEGMENT_SIZE {
            return true;
        }
        if segment + 1 < SEGMENT_COUNT {
            self.used_bytes = (segment + 1) * SEGMENT_SIZE + shared.blocks_start;
            return true;
        }
        false
    }

    /// Points the slot at `destination`'s block, if the jump reaches. Returns whether it did.
    fn patch_link(&mut self, shared: &SharedJit, site: &LinkedSite, destination: u32) -> bool {
        let slot_addr = shared.exec_mem.base() + site.exec_offset as u64;
        let link_addr = self.blocks[destination as usize].link_addr;
        let Some(word) = arch::link_word(slot_addr, link_addr) else {
            return false;
        };
        shared
            .exec_mem
            .write_word(site.exec_offset + arch::LINK_WORD_OFFSET, word);
        self.incoming
            .entry(destination)
            .or_default()
            .push(site.clone());
        self.stats.linked += 1;
        true
    }

    /// Drops every block and rewinds the bump pointer, keeping the prologue. No generated code may
    /// be running (the caller holds every vCPU outside it, or is the only one).
    fn flush(&mut self, shared: &SharedJit) {
        self.block_map.clear();
        self.page_blocks.clear();
        self.incoming.clear();
        self.pending.clear();
        for flag in &shared.valid[..self.blocks.len()] {
            flag.store(0, Ordering::Release);
        }
        self.blocks.clear();
        self.used_bytes = shared.blocks_start;
        self.stats.resets += 1;
        shared.epoch.fetch_add(1, Ordering::AcqRel);
    }
}

impl Jit {
    /// `chaining` lets blocks jump straight to their successor through the jump cache instead of
    /// returning to the run loop between blocks; the lockstep verifier needs one block per
    /// `step_block`, so it passes false.
    pub fn new(chaining: bool) -> Option<Jit> {
        if !crate::jit::probe() {
            return None;
        }
        let exec_mem = ExecMem::new(CODE_SIZE)?;
        let base = exec_mem.base();
        // Zeroed pages the allocator hands over untouched, so the table costs address space until
        // blocks are made. `AtomicU8` has the layout of `u8`.
        let valid: Box<[AtomicU8]> = unsafe {
            let bytes = vec![0u8; MAX_BLOCKS].into_boxed_slice();
            Box::from_raw(Box::into_raw(bytes) as *mut [AtomicU8])
        };
        let mut segment_addrs = Vec::with_capacity(SEGMENT_COUNT);
        let mut blocks_start = 0;
        for segment in 0..SEGMENT_COUNT {
            let segment_offset = segment * SEGMENT_SIZE;
            let prologue = arch::prologue(base + segment_offset as u64);
            exec_mem.write_bytes(segment_offset, &prologue.code);
            blocks_start = (prologue.code.len() + HEADER_ALIGN - 1) & !(HEADER_ALIGN - 1);
            segment_addrs.push(arch::addrs(&prologue, chaining));
        }
        let shared = Arc::new(SharedJit {
            exec_mem,
            enter_trampoline: base,
            segment_addrs,
            blocks_start,
            valid,
            epoch: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            ordered_memory: AtomicBool::new(false),
            #[cfg(test)]
            test_block_limit: std::sync::atomic::AtomicUsize::new(usize::MAX),
            #[cfg(test)]
            test_install_pause_micros: AtomicU64::new(0),
            state: Mutex::new(State {
                used_bytes: blocks_start,
                blocks: Vec::new(),
                block_map: HashMap::new(),
                page_blocks: HashMap::new(),
                incoming: HashMap::new(),
                pending: HashMap::new(),
                stats: JitStats::default(),
            }),
        });
        Some(Jit::with_shared(shared))
    }

    fn with_shared(shared: Arc<SharedJit>) -> Jit {
        Jit {
            seen_epoch: shared.epoch.load(Ordering::Acquire),
            shared,
            jump_cache: vec![JumpEntry::default(); 1 << JUMP_CACHE_BITS].into_boxed_slice(),
        }
    }

    /// Another vCPU's handle on the same cache: its blocks, its links, its invalidations; a jump
    /// cache of its own.
    pub fn sibling(&self) -> Jit {
        Jit::with_shared(self.shared.clone())
    }

    /// Blocks compiled before the flag flips would lack the ordering, so they are dropped. That
    /// only ever happens before a second vCPU exists, so nothing is running them.
    pub fn set_ordered_memory(&mut self, on: bool) {
        if self.shared.ordered_memory.swap(on, Ordering::AcqRel) != on {
            let mut state = self.shared.lock();
            if !state.blocks.is_empty() {
                state.flush(&self.shared);
            }
            drop(state);
            self.sync();
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stats(&self) -> JitStats {
        self.shared.lock().stats
    }

    pub fn enter_trampoline(&self) -> u64 {
        self.shared.enter_trampoline
    }

    /// What the trampoline is given so the chain stub can find this handle's jump cache.
    pub fn jump_cache_base(&self) -> u64 {
        self.jump_cache.as_ptr() as u64
    }

    /// Likewise the validity table, which is the same for every handle but not a constant of the
    /// shared code: a slot costs one load, an address baked into it several instructions.
    pub fn valid_base(&self) -> u64 {
        self.shared.valid.as_ptr() as u64
    }

    /// Clears the jump cache if the buffer was flushed since it was last used. Called before
    /// anything reads it: it names blocks that are gone.
    pub fn sync(&mut self) {
        // Relaxed: the flush that bumps it ran with this vCPU parked, which already ordered it.
        let epoch = self.shared.epoch.load(Ordering::Relaxed);
        if epoch != self.seen_epoch {
            self.jump_cache.fill(JumpEntry::default());
            self.seen_epoch = epoch;
            // The flush rewrote addresses this core may still have fetched the old code of.
            crate::sys::instruction_barrier();
        }
    }

    #[cfg(test)]
    pub fn set_block_limit(&self, limit: usize) {
        self.shared.test_block_limit.store(limit, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn set_install_pause_micros(&self, micros: u64) {
        self.shared
            .test_install_pause_micros
            .store(micros, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn block_is_valid(&self, index: usize) -> bool {
        self.shared.valid[index].load(Ordering::Acquire) != 0
    }

    #[cfg(test)]
    fn state(&self) -> MutexGuard<'_, State> {
        self.shared.lock()
    }

    #[cfg(test)]
    fn link_state_len(&self) -> (usize, usize) {
        let state = self.shared.lock();
        (
            state.incoming.values().map(Vec::len).sum(),
            state.pending.values().map(Vec::len).sum(),
        )
    }

    /// Called from `Cpu::code_written` when a guest write hits a page that has blocks, by the
    /// vCPU that wrote it (or the one a DMA ran on): the blocks go for every vCPU at once. They
    /// are only marked invalid, not freed: the block that performed the write may still be
    /// running, and `Cpu::block_exit_requested` makes it return before reaching stale code. Other
    /// vCPUs may still run it until they next leave their current block.
    ///
    /// Also clears the page's code flag, under the lock: `install` checks the flag under the same
    /// lock, so a translation that saw the flag set and read the bytes before this write cannot
    /// register its block on a page that no longer makes stores call `code_written` (it would
    /// stay stale for good).
    pub fn invalidate_page(&self, ppn: u64, flags: &CodePages) {
        let shared = &*self.shared;
        let mut state = shared.lock();
        // Bumped even when the page has no blocks yet: a translation of it may be between its
        // read of the bytes and its insert (see `install`).
        shared.invalidations.fetch_add(1, Ordering::AcqRel);
        flags.set(ppn as usize, 0);
        let Some(indices) = state.page_blocks.remove(&ppn) else {
            return;
        };
        for index in indices {
            shared.valid[index as usize].store(0, Ordering::Release);
            let key = state.blocks[index as usize].key;
            state.block_map.remove(&key);
            // A direct jump into the block would skip every lookup, so it goes back to
            // falling through to the chain stub, and waits for the block to be translated
            // again. The invalidated block's own slots stay patched: it never runs again.
            for site in state.incoming.remove(&index).unwrap_or_default() {
                shared.exec_mem.write_word(
                    site.exec_offset + arch::LINK_WORD_OFFSET,
                    arch::unlink_word(),
                );
                state.stats.unlinked += 1;
                state
                    .pending
                    .entry(site.destination)
                    .or_default()
                    .push(site);
            }
        }
    }

    /// Drops every block of the whole cache, for tests that need a known state. Nothing may be
    /// running generated code.
    #[cfg(test)]
    pub fn reset(&mut self) {
        let shared = self.shared.clone();
        shared.lock().flush(&shared);
        self.sync();
    }

    /// The block translated for the code at this physical address, if any; `rip` only picks the
    /// jump cache slot. A jump cache miss falls back to the block map and refills the slot.
    pub fn lookup(&mut self, rip: u64, phys_addr: u64) -> Option<u64> {
        let slot = &self.jump_cache[jump_cache_hash(rip)];
        if slot.code_addr != 0
            && slot.rip == rip
            && slot.phys_addr == phys_addr
            && self.shared.valid[slot.block as usize].load(Ordering::Relaxed) != 0
        {
            return Some(slot.code_addr);
        }
        let state = self.shared.lock();
        let index = *state.block_map.get(&(phys_addr, NO_SECOND_PAGE))?;
        let code_addr = state.blocks[index as usize].code_addr;
        drop(state);
        // Another vCPU may have written this block.
        crate::sys::instruction_barrier();
        self.jump_cache[jump_cache_hash(rip)] = JumpEntry {
            rip,
            phys_addr,
            code_addr,
            block: index as u64,
        };
        Some(code_addr)
    }

    /// The straddling block cached for `rip`, if its slot is still good: the block is valid and the
    /// page after `rip`'s is still the physical page it was decoded from (the first page's
    /// physical address is already known to match). Checked here and not by the chain stub, so
    /// the stub never enters a straddling block.
    pub fn lookup_straddler(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Option<u64> {
        let slot = &self.jump_cache[jump_cache_hash(rip)];
        if slot.phys_addr != phys_addr | STRADDLER_TAG || slot.rip != rip {
            return None;
        }
        let (code_addr, block) = (slot.code_addr, slot.block);
        let index = block & SLOT_BLOCK_MASK;
        if self.shared.valid[index as usize].load(Ordering::Relaxed) == 0 {
            return None;
        }
        let next_page = cpu
            .translate(
                (rip & !PAGE_MASK).wrapping_add(PAGE_SIZE),
                AccessKind::Exec,
                false,
            )
            .ok()?;
        ((next_page >> PAGE_SHIFT) == block >> SLOT_BLOCK_BITS).then_some(code_addr)
    }

    /// Decodes up to `MAX_INSNS` instructions at `rip` (physical `phys_addr`), compiles them and
    /// returns the host address of the block. A block never spans a page boundary, except the
    /// single-instruction block for an instruction that itself straddles one. `rip` must be
    /// `cpu.rip`: that case decodes through `Cpu::fetch_decode`, which reads it. Only the offset
    /// of `rip` in its page matters to the code that comes out (see `BlockKey`).
    ///
    /// Everything that can wait for another vCPU (`mark_code` and `fetch_decode` take the
    /// exclusive section, the page walk reads guest tables) happens before the lock is taken, and
    /// the lock is never held into `Cpu::with_exclusive`: the vCPUs that section waits for may be
    /// waiting for the lock.
    pub fn translate(&mut self, cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Result<u64, Exception> {
        loop {
            // Code can only execute from RAM; reported as a not-present fetch fault.
            if !cpu.ram.contains(phys_addr, 1) {
                return Err(cpu.page_fault(rip, AccessKind::Exec, cpu.cpl == 3, false, false));
            }
            // Makes guest stores to this page take the slow path so `code_written` sees them.
            // Before the bytes are read: with threads, a store by another vCPU is then either
            // already in them or sees the flag (`Cpu::mark_code`).
            cpu.mark_code(phys_addr);
            #[cfg(test)]
            if let Some(hook) = AFTER_MARK.with(|hook| hook.borrow_mut().take()) {
                hook(cpu);
            }
            let generation = self.shared.invalidations.load(Ordering::Acquire);
            let decoded = decode_block(cpu, rip, phys_addr, false)?;
            #[cfg(test)]
            if let Some(hook) = BEFORE_INSTALL.with(|hook| hook.borrow_mut().take()) {
                hook(cpu);
            }
            #[cfg(test)]
            match self
                .shared
                .test_install_pause_micros
                .load(Ordering::Relaxed)
            {
                0 => {}
                micros => std::thread::sleep(std::time::Duration::from_micros(micros)),
            }
            match self.install(decoded, rip, phys_addr, generation, &cpu.page_has_code) {
                Install::Ready(code_addr) => return Ok(code_addr),
                Install::Stale => {}
                Install::Full => self.flush(cpu),
            }
        }
    }

    /// The part of `translate` under the lock: the checks that make the decoded bytes still good,
    /// then compile, write and register the block.
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
        // A write to the page since the bytes were read: its `invalidate_page` may have run
        // before this block existed (so dropped nothing), and the block would outlive it.
        if shared.invalidations.load(Ordering::Acquire) != generation {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        // `invalidate_page` clears a page's flag under this lock, after the write: a flag that is
        // gone means a write the generation check above could not see (it came before this
        // decode started marking, and the flag it cleared was the one `mark_code` found set). The
        // retry flags the page again, which also revokes the writers' TLB tags.
        if decoded
            .pages
            .iter()
            .any(|&page| flags.get(page as usize) == 0)
        {
            state.stats.stale_decodes += 1;
            return Install::Stale;
        }
        self.sync();
        let key = (phys_addr, decoded.second_page);
        // Another vCPU may have translated it since this one missed.
        if let Some(&index) = state.block_map.get(&key) {
            let code_addr = state.blocks[index as usize].code_addr;
            // Translated by another vCPU while this one decoded.
            crate::sys::instruction_barrier();
            self.fill_jump_cache(rip, phys_addr, decoded.second_page, code_addr, index);
            return Install::Ready(code_addr);
        }
        if !state.make_room(&shared) {
            return Install::Full;
        }
        let Decoded {
            insns,
            insn_rips,
            end_rip,
            crosses_page,
            second_page,
            pages,
        } = decoded;
        let insns: Box<[Insn]> = insns.into_boxed_slice();
        let segment = state.used_bytes / SEGMENT_SIZE;
        state.used_bytes = (state.used_bytes + BLOCK_ALIGN - 1) & !(BLOCK_ALIGN - 1);
        let code_addr = shared.exec_mem.base() + state.used_bytes as u64;
        let compiled = arch::compile_block(
            code_addr,
            &shared.segment_addrs[segment],
            &insns,
            &insn_rips,
            end_rip,
            shared.ordered_memory.load(Ordering::Acquire),
        );
        state.stats.inlined += compiled.inlined as u64;
        state.stats.flag_records_skipped += compiled.flag_records_skipped as u64;
        state.stats.total += insns.len() as u64;
        let block_offset = state.used_bytes;
        shared.exec_mem.write_bytes(block_offset, &compiled.code);
        state.used_bytes += compiled.code.len();
        let link_addr = code_addr;
        let code_addr = link_addr + compiled.body_offset as u64;
        let index = state.blocks.len() as u32;
        state.block_map.insert(key, index);
        state.blocks.push(Block {
            key,
            code_addr,
            link_addr,
            _insns: insns,
        });
        shared.valid[index as usize].store(1, Ordering::Release);
        // Only same-page blocks are linked, in either direction. A direct jump skips the chain
        // stub's TLB probe and key check, which is sound when the destination is in the page the
        // jumping block was entered through; a block that straddles a page is keyed by a second
        // page that only the translating address space is known to map there.
        if !crosses_page {
            for site in state.pending.remove(&phys_addr).unwrap_or_default() {
                state.patch_link(&shared, &site, index);
            }
            let page_base = phys_addr & !PAGE_MASK;
            for link in compiled.links {
                let destination = page_base | (link.target_rip & PAGE_MASK);
                let site = LinkedSite {
                    exec_offset: block_offset + link.code_offset as usize,
                    destination,
                };
                match state.block_map.get(&(destination, NO_SECOND_PAGE)).copied() {
                    Some(target) => {
                        state.patch_link(&shared, &site, target);
                    }
                    None => state.pending.entry(destination).or_default().push(site),
                }
            }
        }
        for page in pages {
            state.page_blocks.entry(page).or_default().push(index);
        }
        state.stats.translated += 1;
        self.fill_jump_cache(rip, phys_addr, second_page, code_addr, index);
        Install::Ready(code_addr)
    }

    /// A straddling block's slot is tagged (`STRADDLER_TAG`): the chain stub validates only the
    /// first page, so it must miss, and `lookup_straddler` finds it after checking the second.
    fn fill_jump_cache(
        &mut self,
        rip: u64,
        phys_addr: u64,
        second_page: u64,
        code_addr: u64,
        index: u32,
    ) {
        self.jump_cache[jump_cache_hash(rip)] = if second_page == NO_SECOND_PAGE {
            JumpEntry {
                rip,
                phys_addr,
                code_addr,
                block: index as u64,
            }
        } else {
            JumpEntry {
                rip,
                phys_addr: phys_addr | STRADDLER_TAG,
                code_addr,
                block: index as u64 | (second_page >> PAGE_SHIFT) << SLOT_BLOCK_BITS,
            }
        };
    }

    /// Flushes the whole cache, which is full, for every vCPU. The others are held outside
    /// generated code (`Cpu::with_exclusive`) while the code is thrown away and overwritten; the
    /// lock is not held while waiting for them.
    fn flush(&mut self, cpu: &mut Cpu) {
        let shared = self.shared.clone();
        let observed = shared.epoch.load(Ordering::Acquire);
        cpu.with_exclusive(|_| {
            let mut state = shared.lock();
            // Another vCPU may have flushed while this one waited for its turn.
            if shared.epoch.load(Ordering::Acquire) == observed {
                state.flush(&shared);
            }
        });
        self.sync();
    }
}

impl Cpu {
    /// Runs one block (or, on a miss, translates then runs it) starting at `self.rip`.
    ///
    /// The block is entered through the prologue trampoline as
    /// `extern "C" fn(cpu: *mut Cpu, block: u64) -> u64`. The trampoline pins the `Cpu` and TLB
    /// base registers, saves the callee-saved registers and jumps to `block`. The return value
    /// is built by the exit stubs in `arch::prologue` (`arch_a64.rs`, `arch_x64.rs`), which the
    /// code from `jit/emit.rs` / `jit/emit_x64.rs` jumps to:
    ///
    /// - `r >> helpers::EXIT_COUNT_SHIFT`: instructions fully retired by the block.
    /// - `r & helpers::EXIT_STATUS_MASK`: `STATUS_EXCEPTION` if a helper stored an exception in
    ///   `self.pending`, otherwise 0 or `STATUS_EXIT_REQUESTED` (both mean success).
    ///
    /// `Cpu::run` has already added 1 to `icount` for the first instruction (the interpreter
    /// counts an instruction before executing it, so a faulting one is counted too). The updates
    /// below make the total identical to what the interpreter would produce: `done` on success,
    /// `done + 1` on an exception. `Cpu::run_verify` relies on `last_block_insns` to replay the
    /// same number of instructions in the interpreter.
    pub fn step_block(&mut self) -> Result<(), Exception> {
        let rip = self.rip;
        let phys_addr = self.translate(rip, AccessKind::Exec, false)?;
        // `translate` needs `&mut Cpu`, hence the take. The Jit is put back before the block runs
        // because stores inside it reach `Cpu::code_written`, which needs `self.jit` to invalidate.
        let mut jit = self.jit.take().unwrap();
        // Another vCPU may have flushed the buffer since this one last ran a block.
        jit.sync();
        let cached = match jit.lookup_straddler(self, rip, phys_addr) {
            Some(code) => Some(code),
            None => jit.lookup(rip, phys_addr),
        };
        let code = match cached {
            Some(code) => Ok(code),
            None => jit.translate(self, rip, phys_addr),
        };
        let enter = jit.enter_trampoline();
        let (jump_cache, valid) = (jit.jump_cache_base(), jit.valid_base());
        self.jit = Some(jit);
        let code = code?;
        let run_block: extern "C" fn(*mut Cpu, u64, u64, u64) -> u64 =
            unsafe { std::mem::transmute(enter as usize) };
        let exit_code = run_block(self as *mut Cpu, code, jump_cache, valid);
        let done = exit_code >> helpers::EXIT_COUNT_SHIFT;
        self.last_block_insns = done;
        // Set by the emitted code before rdtsc / port I/O at block index k > 0: it already added
        // k to `icount` so the instruction reads the exact time, and it must not be added twice.
        let prepaid_icount = std::mem::take(&mut self.prepaid_icount);
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
    //! Block cache, jump cache, invalidation, code-buffer management and `Cpu::step_block`
    //! accounting, tested against real translated blocks.

    use super::*;
    use crate::cpu::decode;
    use crate::cpu::exception::Exception;
    use crate::jit::blocks::MAX_INSNS;
    use crate::jit::serialize_jit_tests;
    use crate::testkit::*;

    const PAGE_BYTES_FOR_TESTS: u64 = PAGE_SIZE;
    /// Page offset of the snippet in the harness (`CODE_BASE` is page-aligned plus this).
    const SNIPPET_PAGE_OFFSET: u64 = CODE_BASE & PAGE_MASK;

    // add rax, rcx; hlt
    const ADD_THEN_HLT: [u8; 4] = [0x48, 0x01, 0xc8, 0xf4];
    // add rax, rcx; jmp 1f; 1: hlt
    const ADD_THEN_JMP_NEXT: [u8; 6] = [0x48, 0x01, 0xc8, 0xeb, 0x00, 0xf4];
    // add rax, rcx; add rax, rcx; ud2
    const ADD_THEN_UD2: [u8; 8] = [0x48, 0x01, 0xc8, 0x48, 0x01, 0xc8, 0x0f, 0x0b];
    // mov eax, 0x11223344
    const MOV_EAX_IMM32: [u8; 5] = [0xb8, 0x44, 0x33, 0x22, 0x11];
    // mov rax, [rbx + rcx*8 + 0x12345]
    const MOV_RAX_FROM_INDEXED_MEMORY: [u8; 8] = [0x48, 0x8b, 0x84, 0xcb, 0x45, 0x23, 0x01, 0x00];
    // nop
    const NOP: u8 = 0x90;
    // ud2-free invalid opcode (push es is not valid in 64-bit mode)
    const INVALID_OPCODE: [u8; 1] = [0x06];

    fn cpu_running(code: &[u8]) -> Cpu {
        Harness::new(code).cpu
    }

    fn new_jit() -> Jit {
        Jit::new(true).expect("the host has a JIT")
    }

    /// For tests of one block's accounting: the block returns to the caller instead of chaining on.
    fn new_unchained_jit() -> Jit {
        Jit::new(false).expect("the host has a JIT")
    }

    /// Translates the block at `CODE_BASE` (identity-mapped, so physical == linear).
    fn translate_at_code_base(jit: &mut Jit, cpu: &mut Cpu) -> u64 {
        jit.translate(cpu, CODE_BASE, CODE_BASE)
            .expect("translation succeeds")
    }

    fn nops(count: usize) -> Vec<u8> {
        vec![NOP; count]
    }

    #[test]
    fn a_new_jit_places_its_first_block_after_an_aligned_prologue() {
        let _guard = serialize_jit_tests();
        let jit = new_jit();
        assert_eq!(jit.enter_trampoline(), jit.shared.exec_mem.base());
        assert_eq!(jit.state().used_bytes, jit.shared.blocks_start);
        assert_eq!(jit.shared.blocks_start % HEADER_ALIGN, 0);
        assert!(jit.shared.blocks_start > 0);
        assert!(jit.state().blocks.is_empty());
        assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
        assert_eq!(jit.stats(), JitStats::default());
    }

    // aarch64 `b` reaches +-128 MiB from a block back to the stubs at the start of its segment.
    const _: () = assert!(SEGMENT_SIZE <= 128 << 20);
    const _: () = assert!(CODE_HEADROOM < SEGMENT_SIZE);

    #[test]
    fn lookup_misses_before_translation_and_hits_afterwards() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        let code = translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert_eq!(jit.stats().translated, 1);
    }

    #[test]
    fn translated_blocks_are_aligned_and_do_not_overlap() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
            .unwrap();
        let first = jit.state().blocks[0].link_addr;
        let second = jit.state().blocks[1].link_addr;
        assert_eq!(first % BLOCK_ALIGN as u64, 0);
        assert_eq!(second % BLOCK_ALIGN as u64, 0);
        assert!(first >= jit.shared.exec_mem.base() + jit.shared.blocks_start as u64);
        assert!(second > first);
        assert!(second < jit.shared.exec_mem.base() + jit.state().used_bytes as u64);
    }

    #[test]
    fn lookup_requires_the_same_physical_address() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert!(jit.lookup(CODE_BASE, CODE_BASE).is_some());
        assert_eq!(
            jit.lookup(CODE_BASE, CODE_BASE + PAGE_BYTES_FOR_TESTS),
            None,
            "the same virtual address mapped to another page is another block"
        );
    }

    #[test]
    fn lookup_finds_the_block_at_any_virtual_alias_of_its_physical_address() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let code = translate_at_code_base(&mut jit, &mut cpu);
        // ASLR maps the same physical page at another virtual address in every process: one
        // translation serves all of them.
        let alias = CODE_BASE ^ (1 << 40);
        assert_eq!(jit.lookup(alias, CODE_BASE), Some(code));
        assert_eq!(jit.stats().translated, 1);
        // The alias now has its own jump cache slot, so the first mapping still hits its own.
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
    }

    #[test]
    fn jump_cache_hash_stays_in_range_and_folds_in_the_high_bits() {
        let _guard = serialize_jit_tests();
        for rip in [0, 1, CODE_BASE, u64::MAX, 0xffff_ffff_8000_0000, 1 << 63] {
            assert!(jump_cache_hash(rip) < 1 << JUMP_CACHE_BITS, "{rip:#x}");
        }
        // The bits above the cache index take part, so a rip one 64 KiB away does not alias.
        assert_ne!(
            jump_cache_hash(0x1000),
            jump_cache_hash(0x1000 + (1 << JUMP_CACHE_BITS))
        );
        // A deliberate collision: flipping bit 0 and bit 16 cancels out in `rip ^ (rip >> 16)`.
        assert_eq!(
            jump_cache_hash(CODE_BASE),
            jump_cache_hash(CODE_BASE ^ (1 << 16) ^ 1)
        );
    }

    #[test]
    fn a_colliding_rip_evicts_the_older_block_from_the_jump_cache_but_not_from_the_block_map() {
        let _guard = serialize_jit_tests();
        let colliding_rip = CODE_BASE ^ (1 << 16) ^ 1;
        assert_eq!(jump_cache_hash(colliding_rip), jump_cache_hash(CODE_BASE));
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        let second = jit
            .translate(&mut cpu, colliding_rip, colliding_rip)
            .unwrap();
        assert_ne!(first, second);
        assert_eq!(
            jit.jump_cache[jump_cache_hash(CODE_BASE)].code_addr,
            second,
            "the slot belongs to the newer block"
        );
        // The older block is still found, and found again as the slot's owner, without
        // translating a second copy of it.
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(first));
        assert_eq!(jit.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, first);
        assert_eq!(jit.lookup(colliding_rip, colliding_rip), Some(second));
        assert_eq!(jit.state().blocks.len(), 2);
        assert_eq!(jit.stats().translated, 2);
    }

    #[test]
    fn invalidating_a_page_drops_its_blocks_but_keeps_their_slots() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let page = CODE_BASE >> PAGE_SHIFT;
        assert!(jit.state().page_blocks.contains_key(&page));
        jit.invalidate_page(page, &cpu.page_has_code);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert!(!jit.block_is_valid(0));
        assert_eq!(
            jit.state().blocks.len(),
            1,
            "indices held elsewhere stay valid"
        );
        assert!(!jit.state().page_blocks.contains_key(&page));
    }

    #[test]
    fn invalidating_an_unrelated_page_changes_nothing() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let code = translate_at_code_base(&mut jit, &mut cpu);
        jit.invalidate_page((CODE_BASE >> PAGE_SHIFT) + 1, &cpu.page_has_code);
        jit.invalidate_page(0, &cpu.page_has_code);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert!(jit.block_is_valid(0));
    }

    #[test]
    fn invalidating_a_page_twice_is_harmless() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let page = CODE_BASE >> PAGE_SHIFT;
        jit.invalidate_page(page, &cpu.page_has_code);
        jit.invalidate_page(page, &cpu.page_has_code);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    }

    #[test]
    fn a_block_translated_after_invalidation_is_valid_again() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.invalidate_page(CODE_BASE >> PAGE_SHIFT, &cpu.page_has_code);
        let code = translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert_eq!(jit.stats().translated, 2);
    }

    #[test]
    fn translating_marks_the_page_so_stores_to_it_reach_code_written() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        assert_eq!(cpu.page_has_code[page], 0);
        translate_at_code_base(&mut jit, &mut cpu);
        assert_ne!(cpu.page_has_code[page], 0);
    }

    #[test]
    fn a_block_includes_its_terminator_and_stops_there() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(8)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(
            jit.state().blocks[0]._insns.len(),
            2,
            "add and the jmp that ends the block"
        );
        assert_eq!(jit.stats().total, 2);
    }

    #[test]
    fn a_block_holds_at_most_max_insns_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&nops(MAX_INSNS * 2 + 5));
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.state().blocks[0]._insns.len(), MAX_INSNS);
        let next_rip = CODE_BASE + MAX_INSNS as u64;
        jit.translate(&mut cpu, next_rip, next_rip).unwrap();
        assert_eq!(jit.state().blocks[1]._insns.len(), MAX_INSNS);
        jit.translate(
            &mut cpu,
            next_rip + MAX_INSNS as u64,
            next_rip + MAX_INSNS as u64,
        )
        .unwrap();
        // The five remaining nops are followed by zero bytes, which decode as `add [rax], al`.
        let third = &jit.state().blocks[2]._insns;
        assert!(third[..5].iter().all(|insn| insn.op == NOP as u16));
        assert_eq!(third[5].op, 0x00);
    }

    #[test]
    fn a_block_stops_at_the_end_of_its_page() {
        let _guard = serialize_jit_tests();
        let bytes_to_page_end = (PAGE_SIZE - SNIPPET_PAGE_OFFSET) as usize;
        let mut cpu = cpu_running(&nops(bytes_to_page_end + 10));
        let mut jit = new_jit();
        let page_end_rip = CODE_BASE + bytes_to_page_end as u64 - 3;
        jit.translate(&mut cpu, page_end_rip, page_end_rip).unwrap();
        assert_eq!(
            jit.state().blocks[0]._insns.len(),
            3,
            "three nops fit before the boundary"
        );
        assert_eq!(
            jit.state().page_blocks.len(),
            1,
            "and only the first page is involved"
        );
    }

    #[test]
    fn an_instruction_straddling_a_page_is_a_block_of_its_own_listed_under_both_pages() {
        let _guard = serialize_jit_tests();
        let straddle_start = PAGE_SIZE - 2;
        let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
        let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let rip = CODE_BASE + padding as u64;
        // The straddling case decodes through `Cpu::fetch_decode`, which reads `cpu.rip`.
        cpu.rip = rip;
        jit.translate(&mut cpu, rip, rip).unwrap();
        assert_eq!(jit.state().blocks[0]._insns.len(), 1);
        assert_eq!(
            jit.state().blocks[0]._insns[0].len,
            MOV_EAX_IMM32.len() as u8
        );
        let first_page = rip >> PAGE_SHIFT;
        assert!(jit.state().page_blocks.contains_key(&first_page));
        assert!(jit.state().page_blocks.contains_key(&(first_page + 1)));
        assert!(
            cpu.page_has_code[first_page as usize + 1] != 0,
            "stores to the second page must invalidate it too"
        );
        jit.invalidate_page(first_page + 1, &cpu.page_has_code);
        assert_eq!(
            jit.lookup(rip, rip),
            None,
            "invalidating either page kills the block"
        );
    }

    const STRADDLER_VIRTUAL_PAGE: u64 = 0x0000_2000_0000_0000;

    /// `mov eax, 0x11223344` split over two virtual pages, with the first two bytes at the end of
    /// physical page `first` and the rest at the start of `second`. Returns the CPU, its page
    /// tables and the rip of the instruction.
    fn cpu_with_a_straddling_instruction(
        first: u64,
        second: u64,
    ) -> (Cpu, crate::cpu::test_support::PageTables, u64) {
        use crate::cpu::test_support::{PTE_PRESENT_WRITABLE_USER, PageTables};
        let mut cpu = cpu_running(&[HLT]);
        let mut tables = PageTables::new(&mut cpu);
        tables.map_4k(
            &mut cpu,
            STRADDLER_VIRTUAL_PAGE,
            first,
            PTE_PRESENT_WRITABLE_USER,
        );
        tables.map_4k(
            &mut cpu,
            STRADDLER_VIRTUAL_PAGE + PAGE_SIZE,
            second,
            PTE_PRESENT_WRITABLE_USER,
        );
        let start = first + PAGE_SIZE - 2;
        cpu.ram[start as usize..][..2].copy_from_slice(&MOV_EAX_IMM32[..2]);
        cpu.ram[second as usize..][..3].copy_from_slice(&MOV_EAX_IMM32[2..]);
        let rip = STRADDLER_VIRTUAL_PAGE + PAGE_SIZE - 2;
        cpu.rip = rip;
        (cpu, tables, rip)
    }

    #[test]
    fn a_straddling_block_is_cached_for_the_vcpu_but_the_chain_stub_cannot_take_it() {
        let _guard = serialize_jit_tests();
        let (first, second) = (SCRATCH, SCRATCH + 2 * PAGE_SIZE);
        let (mut cpu, _tables, rip) = cpu_with_a_straddling_instruction(first, second);
        let phys = first + PAGE_SIZE - 2;
        let mut jit = new_jit();
        let code = jit.translate(&mut cpu, rip, phys).unwrap();
        assert_eq!(
            jit.lookup(rip, phys),
            None,
            "its slot is tagged: the stub's exact comparison never matches"
        );
        assert_eq!(jit.lookup_straddler(&mut cpu, rip, phys), Some(code));
        assert_eq!(jit.stats().translated, 1);
    }

    #[test]
    fn a_straddling_block_is_not_reused_when_the_next_page_is_mapped_elsewhere() {
        use crate::cpu::test_support::PTE_PRESENT_WRITABLE_USER;
        let _guard = serialize_jit_tests();
        let (first, second, elsewhere) =
            (SCRATCH, SCRATCH + 2 * PAGE_SIZE, SCRATCH + 3 * PAGE_SIZE);
        let (mut cpu, mut tables, rip) = cpu_with_a_straddling_instruction(first, second);
        let phys = first + PAGE_SIZE - 2;
        let mut jit = new_jit();
        let code = jit.translate(&mut cpu, rip, phys).unwrap();
        // Another address space, or a remap: the same first page, with a different page after it
        // whose bytes make a different instruction.
        let mut remapped = MOV_EAX_IMM32;
        remapped[4] = 0x99;
        cpu.ram[elsewhere as usize..][..3].copy_from_slice(&remapped[2..]);
        tables.map_4k(
            &mut cpu,
            STRADDLER_VIRTUAL_PAGE + PAGE_SIZE,
            elsewhere,
            PTE_PRESENT_WRITABLE_USER,
        );
        cpu.flush_tlb();
        assert_eq!(jit.lookup_straddler(&mut cpu, rip, phys), None);
        let other = jit.translate(&mut cpu, rip, phys).unwrap();
        assert_ne!(other, code);
        assert_eq!(jit.stats().translated, 2);
        assert_eq!(
            jit.state().blocks[1]._insns[0].imm,
            0x9922_3344u32 as i32 as i64,
            "decoded from the page that is mapped now"
        );
    }

    #[test]
    fn a_block_before_a_straddling_instruction_ends_ahead_of_it() {
        let _guard = serialize_jit_tests();
        let straddle_start = PAGE_SIZE - 2;
        let padding = (straddle_start - SNIPPET_PAGE_OFFSET) as usize;
        let code = [&nops(padding)[..], &MOV_EAX_IMM32[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let rip = CODE_BASE + padding as u64 - 3;
        jit.translate(&mut cpu, rip, rip).unwrap();
        assert_eq!(
            jit.state().blocks[0]._insns.len(),
            3,
            "three nops, then the next block starts at the mov"
        );
        assert_eq!(jit.state().page_blocks.len(), 1);
    }

    #[test]
    fn translating_outside_ram_raises_a_page_fault_and_adds_no_block() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let outside = cpu.ram.len() as u64;
        let error = jit.translate(&mut cpu, outside, outside).unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.cr2, outside);
        assert!(jit.state().blocks.is_empty());
        assert_eq!(jit.stats().translated, 0);
    }

    #[test]
    fn undecodable_bytes_become_a_one_byte_marker_block() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&INVALID_OPCODE);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let block = &jit.state().blocks[0];
        assert_eq!(block._insns.len(), 1);
        assert_eq!(block._insns[0].op, helpers::OP_INVALID);
        assert_eq!(block._insns[0].len, 1);
    }

    #[test]
    fn an_undecodable_byte_ends_the_block_after_the_instructions_before_it() {
        let _guard = serialize_jit_tests();
        let code = [&ADD_THEN_HLT[..3], &INVALID_OPCODE[..], &nops(4)[..]].concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.state().blocks[0]._insns.len(), 2);
        assert_eq!(jit.state().blocks[0]._insns[1].op, helpers::OP_INVALID);
    }

    #[test]
    fn the_generated_code_in_the_buffer_is_the_compiler_output_verbatim() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let body_addr = translate_at_code_base(&mut jit, &mut cpu);
        let block = &jit.state().blocks[0];
        let code_addr = block.link_addr;
        // Blocks are compiled against offsets in their page, not guest addresses.
        let start = CODE_BASE & PAGE_MASK;
        let rips = [start, start + 3];
        let expected = arch::compile_block(
            code_addr,
            &jit.shared.segment_addrs[0],
            &block._insns,
            &rips,
            start + 4,
            false,
        );
        let written =
            unsafe { std::slice::from_raw_parts(code_addr as *const u8, expected.code.len()) };
        assert_eq!(written, expected.code.as_slice());
        assert_eq!(body_addr, code_addr + expected.body_offset as u64);
        assert_eq!(
            code_addr,
            jit.shared.exec_mem.base() + jit.shared.blocks_start as u64
        );
    }

    #[test]
    fn the_exec_memory_write_path_leaves_the_prologue_untouched_by_blocks() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        let prologue_len = jit.shared.blocks_start;
        let before = unsafe {
            std::slice::from_raw_parts(jit.shared.exec_mem.base() as *const u8, prologue_len)
        }
        .to_vec();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 3, CODE_BASE + 3)
            .unwrap();
        let after = unsafe {
            std::slice::from_raw_parts(jit.shared.exec_mem.base() as *const u8, prologue_len)
        };
        assert_eq!(after, before.as_slice());
    }

    #[test]
    fn a_worst_case_block_fits_in_the_headroom_with_a_wide_margin() {
        let _guard = serialize_jit_tests();
        // 48 inline loads, each with a slow path emitted out of line: the largest blocks the
        // compiler produces. `CODE_HEADROOM` is the free space required before translating.
        let code = MOV_RAX_FROM_INDEXED_MEMORY.repeat(MAX_INSNS);
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        let before = jit.state().used_bytes;
        translate_at_code_base(&mut jit, &mut cpu);
        let block_bytes = jit.state().used_bytes - before;
        assert_eq!(jit.state().blocks[0]._insns.len(), MAX_INSNS);
        assert!(
            block_bytes * 16 < CODE_HEADROOM,
            "{block_bytes} bytes for the largest block vs {CODE_HEADROOM} bytes of headroom"
        );
    }

    #[test]
    fn reset_discards_every_block_and_rewinds_to_the_first_block_slot() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_JMP_NEXT);
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        jit.translate(&mut cpu, CODE_BASE + 5, CODE_BASE + 5)
            .unwrap();
        jit.reset();
        assert!(jit.state().blocks.is_empty());
        assert!(jit.state().page_blocks.is_empty());
        assert!(jit.jump_cache.iter().all(|slot| slot.code_addr == 0));
        assert_eq!(jit.state().used_bytes, jit.shared.blocks_start);
        assert_eq!(jit.stats().resets, 1);
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert_eq!(
            translate_at_code_base(&mut jit, &mut cpu),
            first,
            "the buffer is reused from the start"
        );
    }

    #[test]
    fn a_nearly_full_code_buffer_is_flushed_before_the_next_translation() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(4)[..]].concat());
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.state().used_bytes = jit.shared.exec_mem.len() - CODE_HEADROOM + 1;
        let second_rip = CODE_BASE + 5;
        let code = jit.translate(&mut cpu, second_rip, second_rip).unwrap();
        assert_eq!(jit.stats().resets, 1);
        assert_eq!(
            jit.state().blocks.len(),
            1,
            "only the block translated after the flush remains"
        );
        assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
        assert_eq!(jit.lookup(second_rip, second_rip), Some(code));
        assert_eq!(
            jit.state().blocks[0].link_addr,
            jit.shared.exec_mem.base() + jit.shared.blocks_start as u64
        );
    }

    #[test]
    fn a_buffer_with_exactly_the_headroom_free_is_not_flushed() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let mut jit = new_jit();
        jit.state().used_bytes = jit.shared.exec_mem.len() - CODE_HEADROOM;
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(jit.stats().resets, 0);
    }

    #[test]
    fn step_block_runs_a_block_and_accounts_for_its_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
        cpu.jit = Some(Box::new(new_unchained_jit()));
        cpu.gpr[0] = 1;
        cpu.gpr[1] = 2;
        // `Cpu::run` counts the first instruction before calling in.
        cpu.icount += 1;
        cpu.step_block().unwrap();
        assert_eq!(cpu.gpr[0], 3);
        assert_eq!(cpu.last_block_insns, 2);
        assert_eq!(cpu.icount, 2, "the block retired two instructions in total");
        assert_eq!(cpu.rip, CODE_BASE + 5);
    }

    #[test]
    fn step_block_reports_an_exception_with_the_faulting_instruction_counted() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_UD2);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.icount += 1;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 6, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.last_block_insns, 2, "two adds retired before the ud2");
        assert_eq!(
            cpu.icount, 3,
            "and the faulting ud2 counts too, as in the interpreter"
        );
        assert!(
            cpu.pending.is_none(),
            "the exception was taken out of `pending`"
        );
    }

    #[test]
    fn step_block_clears_the_exit_request_and_the_prepaid_count() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.block_exit_requested = true;
        cpu.prepaid_icount = 0;
        cpu.icount += 1;
        cpu.step_block().unwrap();
        assert!(!cpu.block_exit_requested);
        assert_eq!(cpu.prepaid_icount, 0);
    }

    #[test]
    fn step_block_keeps_the_jit_when_translation_fails() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.rip = cpu.ram.len() as u64;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert!(cpu.jit.is_some());
    }

    #[test]
    fn step_block_with_an_unmapped_rip_faults_before_touching_the_cache() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.rip = 0x1_0000_0000;
        let error = cpu.step_block().unwrap_err();
        assert!(
            matches!(error, Exception::Fault { vector: 14, .. }),
            "{error:?}"
        );
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 0);
    }

    #[test]
    fn step_block_reuses_the_cached_block_on_the_second_run() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(1)[..]].concat());
        cpu.jit = Some(Box::new(new_jit()));
        for _ in 0..3 {
            cpu.rip = CODE_BASE;
            cpu.icount += 1;
            cpu.step_block().unwrap();
        }
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 1);
    }

    // mov ecx, 100; 1: sub ecx, 1; jnz 1b; hlt
    const COUNTDOWN_LOOP: [u8; 11] = [
        0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
    ];
    const RSP_INDEX: usize = 4;
    /// mov (1) + 100 * (sub, jnz) + hlt (1).
    const COUNTDOWN_LOOP_INSNS: u64 = 202;
    // mov eax, 7; call 1f; add eax, 1; hlt; 1: add eax, eax; ret
    const CALL_AND_RETURN: [u8; 17] = [
        0xb8, 0x07, 0x00, 0x00, 0x00, 0xe8, 0x04, 0x00, 0x00, 0x00, 0x83, 0xc0, 0x01, 0xf4, 0x01,
        0xc0, 0xc3,
    ];

    /// `Cpu::run`'s per-step bookkeeping around one `step_block` call.
    fn step_once(cpu: &mut Cpu) {
        cpu.interrupt_shadow = false;
        cpu.icount += 1;
        cpu.step_block().unwrap();
    }

    /// Calls `step_block` until the guest halts; returns how many calls it took.
    fn steps_to_halt(cpu: &mut Cpu) -> usize {
        for calls in 1..=1000 {
            step_once(cpu);
            if cpu.halted {
                return calls;
            }
        }
        panic!("never halted: {}", cpu.dump_regs());
    }

    /// Runs the countdown loop a few iterations, so its blocks exist and chain into each other,
    /// and stops mid-loop.
    fn warm_up_loop(cpu: &mut Cpu) {
        cpu.next_event = 20;
        while cpu.icount < 20 {
            step_once(cpu);
        }
        assert!(!cpu.halted);
        cpu.next_event = u64::MAX;
    }

    #[test]
    fn chained_blocks_run_a_whole_loop_without_returning_to_the_caller() {
        let _guard = serialize_jit_tests();
        let mut chained = cpu_running(&COUNTDOWN_LOOP);
        chained.jit = Some(Box::new(new_jit()));
        chained.next_event = u64::MAX;
        // Only blocks that do not exist yet send the chain back to the caller (to be translated).
        let chained_calls = steps_to_halt(&mut chained);
        assert!(chained_calls <= 3, "{chained_calls}");
        assert_eq!(chained.gpr[1], 0);
        assert_eq!(chained.icount, COUNTDOWN_LOOP_INSNS);

        let mut unchained = cpu_running(&COUNTDOWN_LOOP);
        unchained.jit = Some(Box::new(new_unchained_jit()));
        unchained.next_event = u64::MAX;
        assert!(steps_to_halt(&mut unchained) > 100);
        assert_eq!(unchained.icount, COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn a_chain_stops_at_the_event_deadline_and_resumes_without_losing_instructions() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        cpu.next_event = cpu.icount + 50;
        let deadline = cpu.next_event;
        step_once(&mut cpu);
        assert!(!cpu.halted);
        assert!(
            cpu.icount >= deadline && cpu.icount < deadline + 4,
            "{} vs {deadline}",
            cpu.icount
        );
        // Every retired instruction is counted once and the next one is not pre-charged: ecx says
        // how many iterations ran.
        let iterations = 100 - cpu.gpr[1];
        assert_eq!(cpu.icount, 1 + 2 * iterations);
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
    }

    #[test]
    fn a_chained_call_and_return_find_their_blocks_through_the_jump_cache() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&CALL_AND_RETURN);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[0], 15);
        // Run it again from the top: now the call, the callee and the return all hit the cache and
        // the whole sequence is a single call.
        cpu.halted = false;
        cpu.rip = CODE_BASE;
        cpu.gpr[RSP_INDEX] = STACK_TOP;
        assert_eq!(steps_to_halt(&mut cpu), 1);
        assert_eq!(cpu.gpr[0], 15);
        assert_eq!(cpu.jit.as_ref().unwrap().stats().translated, 3);
    }

    #[test]
    fn a_chain_does_not_enter_a_block_whose_page_was_invalidated() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let translated = cpu.jit.as_ref().unwrap().stats().translated;
        let flags = cpu.page_has_code.clone();
        cpu.jit
            .as_mut()
            .unwrap()
            .invalidate_page(CODE_BASE >> PAGE_SHIFT, &flags);
        steps_to_halt(&mut cpu);
        assert!(
            cpu.jit.as_ref().unwrap().stats().translated > translated,
            "the stale blocks were retranslated, not chained into"
        );
    }

    #[test]
    fn same_page_exits_are_linked_directly_once_the_successor_exists() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[1], 0);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
        assert!(cpu.jit.as_ref().unwrap().stats().linked >= 1);
    }

    #[test]
    fn an_unchained_jit_never_links() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_unchained_jit()));
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.jit.as_ref().unwrap().stats().linked, 0);
    }

    #[test]
    fn invalidating_the_destination_unlinks_and_a_retranslation_relinks() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let before = cpu.jit.as_ref().unwrap().stats();
        assert!(before.linked >= 1);
        let flags = cpu.page_has_code.clone();
        cpu.jit
            .as_mut()
            .unwrap()
            .invalidate_page(CODE_BASE >> PAGE_SHIFT, &flags);
        let invalidated = cpu.jit.as_ref().unwrap().stats();
        assert_eq!(invalidated.unlinked, before.linked);
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[1], 0);
        assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
        let after = cpu.jit.as_ref().unwrap().stats();
        assert!(after.linked > before.linked, "{after:?} vs {before:?}");
    }

    #[test]
    fn reset_discards_links_and_pending_entries() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&COUNTDOWN_LOOP);
        cpu.jit = Some(Box::new(new_jit()));
        warm_up_loop(&mut cpu);
        let jit = cpu.jit.as_mut().unwrap();
        assert_ne!(jit.link_state_len(), (0, 0));
        jit.reset();
        assert_eq!(jit.link_state_len(), (0, 0));
    }

    /// Instructions of the block at `CODE_BASE` that were lowered inline, not counting the final
    /// `hlt` (which always goes through the interpreter).
    fn inlined_in(code: &[u8]) -> u64 {
        let mut cpu = cpu_running(code);
        let mut jit = new_unchained_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        jit.stats().inlined
    }

    #[test]
    fn narrow_shifts_by_a_count_below_the_width_are_lowered_inline() {
        let _guard = serialize_jit_tests();
        // shr ax, 5 | shl ax, 1 | sar al, 3 | shr al, cl | shl ax, cl | sar ax, cl
        for shift in [
            &[0x66, 0xc1, 0xe8, 0x05][..],
            &[0x66, 0xd1, 0xe0],
            &[0xc0, 0xf8, 0x03],
            &[0xd2, 0xe8],
            &[0x66, 0xd3, 0xe0],
            &[0x66, 0xd3, 0xf8],
        ] {
            assert_eq!(inlined_in(&[shift, &[0xf4]].concat()), 1, "{shift:x?}");
        }
    }

    #[test]
    fn bmi_shifts_bzhi_andn_and_the_blsr_group_are_lowered_inline() {
        let _guard = serialize_jit_tests();
        for code in [
            // shlx rax, rcx, rbx | shrx eax, ecx, ebx | sarx rax, [rsi], rbx
            &[0xc4, 0xe2, 0xe1, 0xf7, 0xc1][..],
            &[0xc4, 0xe2, 0x63, 0xf7, 0xc1],
            &[0xc4, 0xe2, 0xe2, 0xf7, 0x06],
            // bzhi rax, rcx, rbx | bzhi eax, [rsi], ebx
            &[0xc4, 0xe2, 0xe0, 0xf5, 0xc1],
            &[0xc4, 0xe2, 0x60, 0xf5, 0x06],
            // andn rax, rbx, rcx | blsr rax, rcx | blsmsk eax, ecx | blsi rax, [rsi]
            &[0xc4, 0xe2, 0xe0, 0xf2, 0xc1],
            &[0xc4, 0xe2, 0xf8, 0xf3, 0xc9],
            &[0xc4, 0xe2, 0x78, 0xf3, 0xd1],
            &[0xc4, 0xe2, 0xf8, 0xf3, 0x1e],
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), 1, "{code:x?}");
        }
    }

    #[test]
    fn bit_scans_are_lowered_inline_at_32_and_64_bits_only() {
        let _guard = serialize_jit_tests();
        for (code, inlined) in [
            // bsf rax, rcx | bsr eax, [rsi] | tzcnt rax, rcx | lzcnt eax, [rsi]
            (&[0x48, 0x0f, 0xbc, 0xc1][..], 1),
            (&[0x0f, 0xbd, 0x06], 1),
            (&[0xf3, 0x48, 0x0f, 0xbc, 0xc1], 1),
            (&[0xf3, 0x0f, 0xbd, 0x06], 1),
            // bsf ax, cx | tzcnt ax, cx | repne bsf rax, rcx
            (&[0x66, 0x0f, 0xbc, 0xc1], 0),
            (&[0x66, 0xf3, 0x0f, 0xbc, 0xc1], 0),
            (&[0xf2, 0x48, 0x0f, 0xbc, 0xc1], 0),
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), inlined, "{code:x?}");
        }
    }

    #[test]
    fn one_operand_mul_and_imul_are_lowered_inline_at_32_and_64_bits_only() {
        let _guard = serialize_jit_tests();
        for (code, inlined) in [
            // mul rcx | imul dword ptr [rsi] | mul ecx
            (&[0x48, 0xf7, 0xe1][..], 1),
            (&[0xf7, 0x2e], 1),
            (&[0xf7, 0xe1], 1),
            // mul cx | mul cl | div rcx
            (&[0x66, 0xf7, 0xe1], 0),
            (&[0xf6, 0xe1], 0),
            (&[0x48, 0xf7, 0xf1], 0),
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), inlined, "{code:x?}");
        }
    }

    #[test]
    fn double_shifts_by_a_nonzero_immediate_are_lowered_inline_at_32_and_64_bits_only() {
        let _guard = serialize_jit_tests();
        for (code, inlined) in [
            // shld rax, rcx, 5 | shrd dword ptr [rsi], ecx, 31 | shld rax, rcx, 1
            (&[0x48, 0x0f, 0xa4, 0xc8, 0x05][..], 1),
            (&[0x0f, 0xac, 0x0e, 0x1f], 1),
            (&[0x48, 0x0f, 0xa4, 0xc8, 0x01], 1),
            // a count that masks to zero | the cl form | shld ax, cx, 5
            (&[0x48, 0x0f, 0xa4, 0xc8, 0x40], 0),
            (&[0x48, 0x0f, 0xa5, 0xc8], 0),
            (&[0x66, 0x0f, 0xa4, 0xc8, 0x05], 0),
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), inlined, "{code:x?}");
        }
    }

    #[test]
    fn avx512_string_loop_instructions_are_lowered_inline_only_in_their_plain_forms() {
        let _guard = serialize_jit_tests();
        for (code, inlined) in [
            // vmovdqu64 zmm1, [rbx] | vmovdqu64 [rbx+64], zmm2 | vmovdqa32 ymm17, ymm18
            (&[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x0b][..], 1),
            (&[0x62, 0xf1, 0xfe, 0x48, 0x7f, 0x53, 0x01], 1),
            (&[0x62, 0xa1, 0x7d, 0x28, 0x6f, 0xca], 1),
            // vpxord zmm1, zmm2, [rbx] | vpcmpeqb k1, ymm16, [rbx] | vptestnmb k2, zmm3, [rbx]
            (&[0x62, 0xf1, 0x6d, 0x48, 0xef, 0x0b], 1),
            (&[0x62, 0xf1, 0x7d, 0x20, 0x74, 0x0b], 1),
            (&[0x62, 0xf2, 0x66, 0x48, 0x26, 0x13], 1),
            // kmovd eax, k1
            (&[0xc5, 0xfb, 0x93, 0xc1], 1),
            // vpxord zmm1{k1}, zmm2, zmm3 | with zeroing | vpcmpub k1, zmm1, zmm2, 4
            (&[0x62, 0xf1, 0x6d, 0x49, 0xef, 0xcb], 1),
            (&[0x62, 0xf1, 0x6d, 0xc9, 0xef, 0xcb], 1),
            (&[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x04], 1),
            // vmovdqu8 [rbx]{k1}, zmm2: a masked store to memory
            (&[0x62, 0xf1, 0x7f, 0x49, 0x7f, 0x13], 0),
            // broadcast, other element sizes, kmovw
            (&[0x62, 0xf1, 0x6d, 0x58, 0xef, 0x0b], 0),
            (&[0x62, 0xf1, 0x75, 0x48, 0x75, 0xca], 0),
            (&[0xc5, 0xf8, 0x93, 0xc1], 0),
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), inlined, "{code:x?}");
        }
    }

    /// The fast path of an inline instruction with a memory operand only runs once the TLB holds
    /// the page, which the first (interpreted) access fills: the diff corpus cannot tell a fast
    /// path that is never taken from one that is, so count the interpreter's calls.
    #[test]
    fn avx512_memory_forms_run_inline_once_the_tlb_is_warm() {
        use crate::cpu::CR4_OSXSAVE;
        use crate::cpu::xsave::XCR0_SUPPORTED;
        let _guard = serialize_jit_tests();
        // vmovdqu64 zmm1, [rsi] | vpxord zmm2, zmm1, [rsi] | vpcmpeqb k1, zmm1, [rsi]
        for instruction in [
            &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x0e][..],
            &[0x62, 0xf1, 0x75, 0x48, 0xef, 0x16],
            &[0x62, 0xf1, 0x75, 0x48, 0x74, 0x0e],
        ] {
            let code = [instruction, instruction, instruction, &[0xf4]].concat();
            let mut cpu = cpu_running(&code);
            cpu.cr4 |= CR4_OSXSAVE;
            cpu.xcr0 = XCR0_SUPPORTED;
            cpu.gpr[RSI] = SCRATCH;
            cpu.jit = Some(Box::new(new_jit()));
            cpu.helper_histogram = Some(Default::default());
            cpu.next_event = u64::MAX;
            steps_to_halt(&mut cpu);
            let key = decode::decode(instruction).unwrap().histogram_key();
            let interpreted = cpu.helper_histogram.as_ref().unwrap().get(&key).copied();
            assert_eq!(
                interpreted,
                Some(1),
                "only the first, TLB-cold access runs in the interpreter: {instruction:x?}"
            );
        }
    }

    #[test]
    fn adc_and_sbb_are_lowered_inline_at_32_and_64_bits_only() {
        let _guard = serialize_jit_tests();
        for (code, inlined) in [
            // adc rax, rcx | sbb eax, ebx | adc rax, 5 | sbb dword ptr [rbx], 1 | adc rax, [rbx]
            (&[0x48, 0x11, 0xc8][..], 1),
            (&[0x19, 0xd8], 1),
            (&[0x48, 0x83, 0xd0, 0x05], 1),
            // A memory destination: only aarch64 lowers it. On x86_64 the carry waits in the
            // register a read-modify-write store parks its result in.
            (&[0x83, 0x1b, 0x01], cfg!(target_arch = "aarch64") as u64),
            (&[0x48, 0x13, 0x03], 1),
            // adc ax, cx | sbb al, bl | adc byte ptr [rbx], 1
            (&[0x66, 0x11, 0xc8], 0),
            (&[0x18, 0xd8], 0),
            (&[0x80, 0x13, 0x01], 0),
        ] {
            assert_eq!(inlined_in(&[code, &[0xf4]].concat()), inlined, "{code:x?}");
        }
    }

    #[test]
    fn narrow_shifts_by_a_count_at_or_above_the_width_stay_in_the_interpreter() {
        let _guard = serialize_jit_tests();
        // shr ax, 16 | shl al, 8 | shr ax, 0
        for shift in [
            &[0x66, 0xc1, 0xe8, 0x10][..],
            &[0xc0, 0xe0, 0x08],
            &[0x66, 0xc1, 0xe8, 0x00],
        ] {
            assert_eq!(inlined_in(&[shift, &[0xf4]].concat()), 0, "{shift:x?}");
        }
    }

    #[test]
    fn a_flag_write_that_a_later_instruction_overwrites_is_not_recorded() {
        let _guard = serialize_jit_tests();
        let translate = |code: &[u8]| {
            let mut cpu = cpu_running(code);
            let mut jit = new_unchained_jit();
            translate_at_code_base(&mut jit, &mut cpu);
            jit.stats().flag_records_skipped
        };
        // add rax, rcx; add rdx, rsi; hlt
        assert_eq!(translate(&[0x48, 0x01, 0xc8, 0x48, 0x01, 0xf2, 0xf4]), 1);
        // adc in between reads the first add's carry, a load in between can fault.
        assert_eq!(translate(&[0x48, 0x01, 0xc8, 0x48, 0x11, 0xf2, 0xf4]), 0);
        assert_eq!(
            translate(&[0x48, 0x01, 0xc8, 0x48, 0x8b, 0x03, 0x48, 0x01, 0xf2, 0xf4]),
            0
        );
    }

    #[test]
    fn helper_histogram_counts_the_instructions_that_reach_the_interpreter() {
        let _guard = serialize_jit_tests();
        // bsf ax, bx three times (16-bit forms have no inline lowering), then hlt.
        let bsf = [0x66, 0x0f, 0xbc, 0xc3];
        let code = [&bsf[..], &bsf, &bsf, &[0xf4, 0xf4]].concat();
        let mut cpu = cpu_running(&code);
        cpu.jit = Some(Box::new(new_jit()));
        cpu.helper_histogram = Some(Default::default());
        cpu.next_event = u64::MAX;
        steps_to_halt(&mut cpu);
        let histogram = cpu.helper_histogram.as_ref().unwrap();
        let bsf_key = decode::decode(&bsf).unwrap().histogram_key();
        assert_eq!(histogram.get(&bsf_key), Some(&3));
        let add_key = decode::decode(&[0x01, 0xd8]).unwrap().histogram_key();
        assert_eq!(histogram.get(&add_key), None, "add is lowered inline");
    }

    #[test]
    fn the_histogram_key_tells_group_opcodes_apart_but_not_unrelated_registers() {
        // shl eax, 1 and rol eax, 1 are both d1 /n.
        let shl = decode::decode(&[0xd1, 0xe0]).unwrap().histogram_key();
        let rol = decode::decode(&[0xd1, 0xc0]).unwrap().histogram_key();
        assert_ne!(shl, rol);
        // mov eax, ebx and mov ecx, ebx are one instruction as far as the histogram goes.
        let mov_a = decode::decode(&[0x89, 0xd8]).unwrap().histogram_key();
        let mov_c = decode::decode(&[0x89, 0xd9]).unwrap().histogram_key();
        assert_eq!(mov_a, mov_c);
    }

    #[test]
    fn chain_exit_after_an_rdtsc_does_not_charge_the_prepaid_instructions_twice() {
        let _guard = serialize_jit_tests();
        // mov ecx, 3; 1: sub ecx, 1; rdtsc; jnz... rdtsc clobbers eax/edx only, flags survive.
        // mov ecx, 3; 1: rdtsc; sub ecx, 1; jnz 1b; hlt
        let code = [
            0xb9, 0x03, 0x00, 0x00, 0x00, 0x0f, 0x31, 0x83, 0xe9, 0x01, 0x75, 0xf8, 0xf4,
        ];
        assert_jit_matches_interpreter(&code, |_| {});
    }

    #[test]
    fn a_full_segment_moves_on_to_the_next_one_with_its_own_prologue_copy() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&[&ADD_THEN_JMP_NEXT[..], &nops(4)[..]].concat());
        let mut jit = new_jit();
        let first = translate_at_code_base(&mut jit, &mut cpu);
        jit.state().used_bytes = SEGMENT_SIZE - CODE_HEADROOM + 1;
        let second_rip = CODE_BASE + 5;
        let second = jit.translate(&mut cpu, second_rip, second_rip).unwrap();
        assert_eq!(jit.stats().resets, 0, "other segments are still free");
        assert_eq!(
            jit.state().blocks[1].link_addr,
            jit.shared.exec_mem.base() + (SEGMENT_SIZE + jit.shared.blocks_start) as u64
        );
        assert_eq!(jit.lookup(second_rip, second_rip), Some(second));
        assert_eq!(
            jit.lookup(CODE_BASE, CODE_BASE),
            Some(first),
            "nothing was dropped"
        );
        assert_ne!(
            jit.shared.segment_addrs[0].exit_ok,
            jit.shared.segment_addrs[1].exit_ok
        );
        assert_eq!(
            jit.shared.segment_addrs[1].exit_ok - jit.shared.segment_addrs[0].exit_ok,
            SEGMENT_SIZE as u64,
            "identical prologue copies"
        );
    }

    #[test]
    fn blocks_in_later_segments_run() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&ADD_THEN_HLT);
        let jit = new_unchained_jit();
        jit.state().used_bytes = SEGMENT_SIZE * (SEGMENT_COUNT - 1) + jit.shared.blocks_start;
        cpu.jit = Some(Box::new(jit));
        cpu.gpr[0] = 1;
        cpu.gpr[1] = 2;
        steps_to_halt(&mut cpu);
        assert_eq!(cpu.gpr[0], 3);
    }

    /// `mov eax, 1; ret`
    const MOV_EAX_ONE: [u8; 6] = [0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3];
    const REWRITTEN_IMMEDIATE: u8 = 2;

    /// The immediate of the first instruction of the only block in `jit`.
    fn immediate_of_only_block(jit: &Jit) -> i64 {
        let state = jit.state();
        assert_eq!(state.blocks.len(), 1);
        state.blocks[0]._insns[0].imm
    }

    /// Another vCPU (`writer`, a handle on the same cache) rewrites the code after `translate`
    /// read it and before it registers the block, then `remark` says what else happened to the
    /// page's flag in that window. The block that ends up registered must hold the new bytes.
    fn translate_while_another_vcpu_rewrites_the_code(remark: bool) -> (i64, u64) {
        let mut cpu = cpu_running(&MOV_EAX_ONE);
        let mut jit = new_jit();
        let writer = jit.sibling();
        let page = CODE_BASE >> PAGE_SHIFT;
        BEFORE_INSTALL.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |cpu: &mut Cpu| {
                cpu.ram[CODE_BASE as usize + 1] = REWRITTEN_IMMEDIATE;
                writer.invalidate_page(page, &cpu.page_has_code);
                if remark {
                    // A third vCPU translating the page flags it again.
                    cpu.page_has_code.set(page as usize, 1);
                }
            }));
        });
        translate_at_code_base(&mut jit, &mut cpu);
        (immediate_of_only_block(&jit), jit.stats().stale_decodes)
    }

    #[test]
    fn a_block_decoded_before_another_vcpus_write_is_not_registered_after_it() {
        let _guard = serialize_jit_tests();
        let (immediate, redecodes) = translate_while_another_vcpu_rewrites_the_code(false);
        assert_eq!(
            immediate, REWRITTEN_IMMEDIATE as i64,
            "the stale decode was kept"
        );
        assert_eq!(redecodes, 1);
    }

    #[test]
    fn the_write_is_noticed_even_when_another_translation_flagged_the_page_again() {
        let _guard = serialize_jit_tests();
        let (immediate, redecodes) = translate_while_another_vcpu_rewrites_the_code(true);
        assert_eq!(
            immediate, REWRITTEN_IMMEDIATE as i64,
            "the stale decode was kept"
        );
        assert_eq!(redecodes, 1);
    }

    /// The case that let stale code run for good before the flag check existed: `mark_code` finds
    /// the page flagged and returns, another vCPU's write then invalidates the page and clears the
    /// flag, and only afterwards does this translation read the generation and the (new) bytes.
    /// Nothing differs for the generation check, but the block would sit on a page whose stores
    /// no longer call `code_written`.
    #[test]
    fn a_page_unflagged_after_mark_code_returned_is_flagged_again_before_a_block_is_registered() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&MOV_EAX_ONE);
        let mut jit = new_jit();
        let writer = jit.sibling();
        let page = CODE_BASE >> PAGE_SHIFT;
        cpu.mark_code(CODE_BASE);
        AFTER_MARK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move |cpu: &mut Cpu| {
                cpu.ram[CODE_BASE as usize + 1] = REWRITTEN_IMMEDIATE;
                writer.invalidate_page(page, &cpu.page_has_code);
            }));
        });
        translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(
            cpu.page_has_code.get(page as usize),
            1,
            "a block on an unflagged page"
        );
        assert_eq!(immediate_of_only_block(&jit), REWRITTEN_IMMEDIATE as i64);
        assert_eq!(jit.stats().stale_decodes, 1);
    }

    #[test]
    fn invalidating_a_page_clears_its_code_flag_for_every_handle() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&MOV_EAX_ONE);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let page = (CODE_BASE >> PAGE_SHIFT) as usize;
        assert_eq!(cpu.page_has_code.get(page), 1);
        jit.sibling()
            .invalidate_page(page as u64, &cpu.page_has_code);
        assert_eq!(cpu.page_has_code.get(page), 0);
        assert_eq!(
            jit.lookup(CODE_BASE, CODE_BASE),
            None,
            "dropped for the other handle too"
        );
    }

    #[test]
    fn a_sibling_handle_runs_what_another_translated_without_translating_again() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&MOV_EAX_ONE);
        let mut jit = new_jit();
        let mut sibling = jit.sibling();
        let code = translate_at_code_base(&mut jit, &mut cpu);
        assert_eq!(sibling.lookup(CODE_BASE, CODE_BASE), Some(code));
        assert_eq!(sibling.stats().translated, 1);
        assert_eq!(jit.stats().translated, 1);
    }

    #[test]
    fn a_flush_by_one_handle_empties_the_jump_cache_of_the_other() {
        let _guard = serialize_jit_tests();
        let mut cpu = cpu_running(&MOV_EAX_ONE);
        let mut jit = new_jit();
        let mut sibling = jit.sibling();
        translate_at_code_base(&mut jit, &mut cpu);
        assert!(sibling.lookup(CODE_BASE, CODE_BASE).is_some());
        jit.reset();
        sibling.sync();
        assert!(sibling.jump_cache.iter().all(|slot| slot.code_addr == 0));
        assert_eq!(sibling.lookup(CODE_BASE, CODE_BASE), None);
    }
}

#[cfg(test)]
mod shared_tests;
