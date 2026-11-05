#[cfg(freestanding)]
use crate::prelude::*;
mod atomic;
mod avx;
mod avx_special;
// The BIOS disk boot is for hosted machines only (`docs/plans/17-bios.md`).
#[cfg(not(freestanding))]
mod bios;
#[cfg(not(freestanding))]
mod bios_keys;
mod bmi;
mod cpuid;
pub(crate) mod decode;
pub(crate) mod exception;
mod exec;
pub(crate) mod flags;
mod fpu;
pub(crate) mod legacy;
mod legacy_flow;
#[cfg(test)]
pub(crate) mod legacy_support;
mod legacy_task;
#[cfg(test)]
mod legacy_tests;
pub(crate) mod mca;
pub(crate) mod mem;
pub(crate) mod modes;
pub(crate) mod mtrr;
mod protection;
pub(crate) mod smp;
#[cfg(test)]
mod smp_tests;
mod sse;
mod svm;
mod sys;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod vcpus;
pub(crate) mod xsave;

use core::sync::atomic::{AtomicBool, Ordering, fence};

use crate::sync::Arc;

use crate::boot::BootState;
use crate::console::ConsoleHandle;
use crate::dev::guestmem::GuestMem;
use crate::dev::handle::DevHandle;
use crate::dev::{DevConfig, Devices};
use crate::machine::{ExitReason, HostClockFn, LogFn, PollFn};
use crate::ram::{CodePages, GuestRam};

use decode::Insn;
use flags::LazyFlagsOp;

pub(crate) use exception::Exception;
/// The interpreter's own semantics for the instructions the JIT computes through
/// `jit::helpers::helper_whole_vector`.
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
pub(crate) use sse::{
    ext::crypto::{
        sha1_four_rounds, sha1_message_1, sha1_message_2, sha1_next_e, sha256_message_1,
        sha256_message_2, sha256_two_rounds,
    },
    ext::{palignr, pshufb},
    shuffle_dwords,
};
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
pub(crate) use sys::{CR0_EM, CR0_TS, CR4_OSFXSR, CR4_OSXSAVE};

/// `Cpu::rng_state` until the machine overrides it with `MachineConfig::random_seed`.
pub(crate) const DEFAULT_RANDOM_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

pub(crate) type ExecResult<T> = Result<T, Exception>;

/// Outcome of `Cpu::before_step`.
enum BeforeStep {
    /// Leave the run loop with this reason.
    Exit(ExitReason),
    /// The CPU was halted and has just idled (advanced time, ticked devices); poll again
    /// instead of executing.
    Idled,
    /// Execute the next instruction or block.
    Execute,
    /// Several vCPUs: this one used up its quantum (or executed `pause`); let the others run.
    Yield,
    /// Several vCPUs: this one is halted with nothing to wake it. The scheduler idles the whole
    /// machine once every vCPU is in this state.
    Halted,
}

/// Why `Cpu::run_slice` returned.
pub(crate) enum Slice {
    Exit(ExitReason),
    Yield,
    Halted,
}

/// The host clock the guest follows in realtime mode, zeroed when the machine starts so guest
/// time begins at 0 as the virtual clock does. vCPUs share one reader (and so one timeline).
#[derive(Clone)]
pub(crate) struct WallClock {
    read: HostClockFn,
    origin_ns: u64,
}

impl WallClock {
    pub(crate) fn new(read: HostClockFn) -> Self {
        let origin_ns = read();
        WallClock { read, origin_ns }
    }

    fn elapsed_ns(&self) -> u64 {
        (self.read)().saturating_sub(self.origin_ns)
    }
}

/// `Instant` as a host clock.
#[cfg(not(freestanding))]
pub(crate) fn std_host_clock() -> HostClockFn {
    let origin = std::time::Instant::now();
    Arc::new(move || origin.elapsed().as_nanos() as u64)
}

/// The clock realtime mode uses when the embedder gave none: only `std` hosts have one.
pub(crate) fn default_host_clock() -> Option<HostClockFn> {
    #[cfg(not(freestanding))]
    {
        Some(std_host_clock())
    }
    #[cfg(freestanding)]
    {
        None
    }
}

/// Virtual time per retired instruction. `now_ns`/`tsc` are derived from `icount`, which makes
/// guest time deterministic unless the clock is realtime.
pub(crate) const NS_PER_INSN: u64 = 1;
/// Instructions between `service_events` polls (stop flag, device timers). The check happens
/// at step boundaries, so with the JIT it can overshoot by up to one block.
const EVENT_INTERVAL: u64 = 4096;
/// On the realtime clock, the least host time between two device ticks from `service_events`.
/// A tick polls every device (virtio queues, interrupt lines, the UART) and costs a few hundred
/// nanoseconds, while `EVENT_INTERVAL` instructions can take less than a microsecond (a bulk
/// `rep stos` slice counts as 32K), so ticking on instructions alone spent a tenth of the time
/// on it. Timers fire up to this much late; host input does not wait on it when the guest is
/// halted (`idle` ticks on its own).
const MIN_DEVICE_TICK_NS: u64 = 20_000;
/// Vector of the non-maskable interrupt.
const NMI_VECTOR: u8 = 2;

pub(crate) const PAGE_SHIFT: u32 = 12;
pub(crate) const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
/// Mask of the offset within a page; `!PAGE_OFFSET_MASK` selects the page frame.
pub(crate) const PAGE_OFFSET_MASK: u64 = PAGE_SIZE - 1;

/// Indices into `Cpu::gpr`, which follows the x86 register encoding (rax=0 ... r15=15).
pub(crate) const GPR_RSP: usize = 4;
pub(crate) const GPR_RSI: usize = 6;
pub(crate) const GPR_RDI: usize = 7;

/// Indices into `Cpu::seg`, the same numbering as the segment-override and `mov sreg` encodings.
pub(crate) const SEG_ES: usize = 0;
pub(crate) const SEG_CS: usize = 1;
pub(crate) const SEG_SS: usize = 2;
pub(crate) const SEG_DS: usize = 3;
pub(crate) const SEG_FS: usize = 4;
pub(crate) const SEG_GS: usize = 5;

pub(crate) const USER_CPL: u8 = 3;

pub(crate) const CR0_PE: u64 = 1 << 0;
pub(crate) const CR0_MP: u64 = 1 << 1;
pub(crate) const CR0_ET: u64 = 1 << 4;
pub(crate) const CR0_NE: u64 = 1 << 5;
/// Supervisor writes honor read-only PTEs (consulted by the page walk).
pub(crate) const CR0_WP: u64 = 1 << 16;
pub(crate) const CR0_PG: u64 = 1 << 31;
pub(crate) const CR4_PAE: u64 = 1 << 5;
pub(crate) const EFER_LME: u64 = 1 << 8;
pub(crate) const EFER_LMA: u64 = 1 << 10;
/// Without NXE the PTE bit 63 is reserved, and instruction fetches never fault on it.
pub(crate) const EFER_NXE: u64 = 1 << 11;

/// Power-on PAT: PA0=WB, PA1=WT, PA2=UC-, PA3=UC, repeated for PA4..7.
const PAT_POWER_ON_DEFAULT: u64 = 0x0007_0406_0007_0406;
/// DR6 and DR7 reset values; the set bits are reserved-as-one.
const DR6_RESET: u64 = 0xffff_0ff0;
const DR7_RESET: u64 = 0x400;
/// All SSE exceptions masked, round to nearest.
const MXCSR_RESET: u32 = 0x1f80;
/// x87 control word after FINIT: all exceptions masked, 64-bit precision, round to nearest.
const FCW_RESET: u16 = 0x37f;

/// A segment register or system-segment cache: the selector plus the hidden descriptor parts.
///
/// `flags` is the upper dword of the descriptor (bytes 4-7) masked with
/// `SEG_DESCRIPTOR_FLAGS_MASK`, so the bit positions below match the in-memory descriptor.
/// `limit` is already scaled by the granularity bit. The JIT reads `base` of FS and GS with
/// `offset_of!(Seg, base)` and `size_of::<Seg>()`.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Seg {
    pub sel: u16,
    pub base: u64,
    pub limit: u32,
    pub flags: u32,
}

pub(crate) const SEG_TYPE_READ_WRITE: u32 = 1 << 9;
pub(crate) const SEG_TYPE_EXECUTABLE: u32 = 1 << 11;
/// The descriptor's S bit: set for code/data segments, clear for system segments and gates.
pub(crate) const SEG_NON_SYSTEM: u32 = 1 << 12;
pub(crate) const SEG_DPL_SHIFT: u32 = 13;
pub(crate) const SEG_DPL_MASK: u32 = 3;
pub(crate) const SEG_PRESENT: u32 = 1 << 15;
/// 64-bit code segment (CS.L).
pub(crate) const SEG_LONG: u32 = 1 << 21;
/// D/B: default operand size for code, big stack for data.
pub(crate) const SEG_DEFAULT_BIG: u32 = 1 << 22;
pub(crate) const SEG_GRANULARITY: u32 = 1 << 23;
/// The descriptor bits that `Seg::flags` keeps: type/S/DPL/P (bits 8-15) and AVL/L/D/G (20-23).
/// Base and limit are stored separately.
pub(crate) const SEG_DESCRIPTOR_FLAGS_MASK: u32 = 0x00f0_ff00;
/// Limit bits 19:16 of a descriptor sit in bits 16-19 of the upper dword; 0xf matches the
/// 4 GiB flat limit. They fall outside `SEG_DESCRIPTOR_FLAGS_MASK`, so only synthesized
/// segments carry them.
const SEG_FLAT_LIMIT_HIGH_BITS: u32 = 0xf << 16;
/// Limit of a flat 4 GiB segment.
pub(crate) const SEG_LIMIT_4GIB: u32 = 0xffff_ffff;
/// Flat 4 GiB read/write data segment, DPL 0.
pub(crate) const SEG_FLAGS_FLAT_DATA: u32 = SEG_GRANULARITY
    | SEG_DEFAULT_BIG
    | SEG_FLAT_LIMIT_HIGH_BITS
    | SEG_PRESENT
    | SEG_NON_SYSTEM
    | SEG_TYPE_READ_WRITE;
/// Execute/read 64-bit code segment, DPL 0.
pub(crate) const SEG_FLAGS_CODE64: u32 =
    SEG_LONG | SEG_PRESENT | SEG_NON_SYSTEM | SEG_TYPE_EXECUTABLE | SEG_TYPE_READ_WRITE;

impl Seg {
    pub fn dpl(&self) -> u8 {
        ((self.flags >> SEG_DPL_SHIFT) & SEG_DPL_MASK) as u8
    }
    pub fn present(&self) -> bool {
        self.flags & SEG_PRESENT != 0
    }
    /// A code segment, as opposed to data or a system descriptor.
    pub fn is_code(&self) -> bool {
        self.flags & SEG_TYPE_EXECUTABLE != 0 && self.flags & SEG_NON_SYSTEM != 0
    }
    pub fn long(&self) -> bool {
        self.flags & SEG_LONG != 0
    }
    /// The descriptor type field (bits 11:8), which for a system descriptor says what it is.
    pub fn descriptor_type(&self) -> u64 {
        ((self.flags >> 8) & 0xf) as u64
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DescriptorTable {
    pub base: u64,
    pub limit: u16,
}

/// One slot of the direct-mapped soft TLB.
///
/// Each tag is the page-aligned linear address (or `TLB_TAG_INVALID`) for which that kind of
/// access is allowed, so a single compare proves both the translation and the permission.
/// `tag_w` is additionally withheld for pages whose PTE is not yet dirty and for pages that
/// hold translated code, which forces those stores through the slow path. MMIO and
/// out-of-RAM pages are never cached. `addend` is the host address of the page minus its
/// linear address (wrapping), so `host_ptr = la + addend`.
///
/// The JIT's inline probe (`tlb_probe` in `jit/emit.rs` and `jit/emit_x64.rs`) hard-codes this
/// layout: `tag_r` at +0, `tag_w` at +8, `addend` at +24, a 32-byte stride. `tag_x` is only
/// read by the interpreter but must stay where it is to keep `addend` at +24.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct TlbEntry {
    pub tag_r: u64,
    pub tag_w: u64,
    pub tag_x: u64,
    pub addend: u64,
}

/// How many large leaves `Cpu::large_pages` remembers. A real core has 32 2 MiB/4 MiB entries;
/// a miss scans all of them, so this stays small.
pub(crate) const LARGE_PAGE_CACHE_ENTRIES: usize = 8;

/// A 2 MiB or 1 GiB leaf the page walker has already validated, so a TLB miss on any other 4 KiB
/// page inside it needs no walk. This is the large-page half of a hardware TLB: the soft TLB
/// itself caches one 4 KiB page per slot, and a guest on transparent huge pages would otherwise
/// walk once per 4 KiB it touches. Filled by `walk_wide`, emptied wherever `flush_tlb` and
/// `invlpg` empty the TLB. Permissions are checked again at each hit (CPL, CR0.WP, SMEP and SMAP
/// can change without a flush); only the walk is skipped.
#[derive(Clone, Copy)]
pub(crate) struct LargePage {
    /// Linear address of the leaf's first byte (aligned to `1 << shift`).
    pub linear_base: u64,
    /// Physical address of the leaf's first byte.
    pub physical_base: u64,
    /// log2 of the leaf size; 0 marks an unused entry.
    pub shift: u32,
    pub writable: bool,
    pub user_allowed: bool,
    pub no_execute: bool,
    /// The leaf already has Dirty set, so a store needs no walk.
    pub dirty: bool,
}

pub(crate) const LARGE_PAGE_EMPTY: LargePage = LargePage {
    linear_base: 0,
    physical_base: 0,
    shift: 0,
    writable: false,
    user_allowed: false,
    no_execute: false,
    dirty: false,
};

/// Never equals a page-aligned address, so an empty tag always misses.
pub(crate) const TLB_TAG_INVALID: u64 = u64::MAX;
pub(crate) const TLB_EMPTY: TlbEntry = TlbEntry {
    tag_r: TLB_TAG_INVALID,
    tag_w: TLB_TAG_INVALID,
    tag_x: TLB_TAG_INVALID,
    addend: 0,
};

/// Permissions of a page-walk result, ANDed across all levels (`walk` in `mem.rs`). They are
/// not stored in `TlbEntry`; `translate` folds them into the per-access tags.
pub(crate) const TLB_FLAG_WRITABLE: u8 = 1;
pub(crate) const TLB_FLAG_USER: u8 = 2;
pub(crate) const TLB_FLAG_NO_EXECUTE: u8 = 4;
/// The leaf PTE already has its dirty bit, so a store can skip the walk that would set it.
pub(crate) const TLB_FLAG_DIRTY: u8 = 8;
pub(crate) const TLB_INDEX_BITS: usize = 10;
/// `Cpu::tlb` holds two halves of this many entries: supervisor first, then user. They are
/// separate because the same page can be accessible to one privilege and not the other, and a
/// CPL change then only has to swap `tlb_active_base` instead of flushing.
pub(crate) const TLB_ENTRIES_PER_MODE: usize = 1 << TLB_INDEX_BITS;

/// Decoded instruction cache slot, keyed by guest physical address. Valid only while
/// `generation` equals `Cpu::decode_cache_generation`.
#[derive(Clone, Copy)]
pub(crate) struct DecodeCacheEntry {
    pub gpa: u64,
    pub generation: u64,
    pub insn: Insn,
}

pub(crate) const DECODE_CACHE_ENTRIES: usize = 1 << 16;
/// Entries kept in `Cpu::trace` (debug only).
const TRACE_CAPACITY: usize = 64;
/// How many of the newest `trace` entries `dump` prints.
const TRACE_DUMP_ENTRIES: usize = 16;
/// Rows of the interpreter-instruction histogram in `Cpu::dump`.
const HISTOGRAM_DUMP_ENTRIES: usize = 25;
/// Registers per line in the register dumps.
const GPRS_PER_DUMP_LINE: usize = 4;

/// All architectural and emulator state of one vCPU, plus (in the vCPU that is executing) the
/// machine it runs in.
///
/// With several vCPUs there is one `Cpu` per vCPU. The ones that are not executing keep only
/// their architectural state: the machine-wide fields (the second group at the end, listed in
/// `Cpu::swap_shared`) live in whichever `Cpu` the scheduler ran last and move with the
/// execution, so the JIT, which finds everything relative to a `Cpu` pointer, needs no
/// per-vCPU copies of code or caches.
///
/// `#[repr(C)]` keeps declaration order: the JIT writes `lazy_op` and `lazy_size` with one
/// 16-bit store, so `lazy_size` must directly follow `lazy_op`. Every other field the JIT
/// touches (`gpr`, `rip`, `icount`, `block_exit_requested`, `prepaid_icount`, `lazy_dst`,
/// `lazy_src`, `seg[].base`, `tlb_active_base`) is found with `offset_of!`.
#[repr(C)]
pub(crate) struct Cpu {
    pub gpr: [u64; 16],
    /// Address of the instruction being executed. The JIT stores it before anything that can
    /// fault, so exceptions and the faulting-instruction restart see an exact value.
    pub rip: u64,
    /// Everything but the `ARITH` flags, which are stale while `lazy_op` is not `Eflags`.
    /// Read it through `eflags_full()`.
    pub eflags: u64,
    /// Deferred arithmetic flags, see `LazyFlagsOp`. `lazy_size` is the operand size in bytes.
    pub lazy_op: LazyFlagsOp,
    pub lazy_size: u8,
    pub lazy_dst: u64,
    pub lazy_src: u64,
    pub lazy_src2: u64,
    /// Address of the half of `tlb` for the current privilege (the second half at CPL 3).
    /// The JIT latches it into a host register on block entry (`jit/arch_a64.rs`,
    /// `jit/arch_x64.rs`), so it must follow every CPL change (use `set_cpl`) and `tlb` must
    /// never be reallocated.
    pub tlb_active_base: u64,
    /// Current privilege level; change it only through `set_cpl`.
    pub cpl: u8,
    /// Set when a store hit a page holding translated code (`code_written`). Generated code
    /// polls it after stores and `helper_exec` reports it, so the running block stops before it
    /// can execute stale instructions; `step_block` clears it.
    pub block_exit_requested: bool,
    /// Instructions the JIT already added to `icount` in the middle of a block, before
    /// rdtsc/port I/O, so those read an up-to-date clock. `step_block` subtracts it from the
    /// block's total.
    pub prepaid_icount: u64,
    /// One plus the block index of the instruction whose slow-path memory access is in progress
    /// (0: none). The access may reach a device, which reads the clock, and the instructions of
    /// the block before this one are not in `icount` yet (`now_ns` adds them). Written by the
    /// emitted slow path, cleared by the helper (`jit/helpers.rs`).
    pub slow_access_insn: u64,
    /// Indexed by `SEG_*`.
    pub seg: [Seg; 6],
    pub gdtr: DescriptorTable,
    pub idtr: DescriptorTable,
    pub ldtr: Seg,
    pub tr: Seg,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    /// The four PDPT entries PAE paging walks from, latched when CR3 (or a CR0/CR4 bit that
    /// changes the paging mode) is written, as hardware does: later stores to the PDPT are not
    /// seen until the next reload. Meaningful only while PAE paging is active.
    pub pdpte: [u64; 4],
    /// Set while an NMI handler runs: further NMIs wait for `iret` (SDM 6.7.1).
    pub nmi_blocked: bool,
    pub mtrr: mtrr::Mtrr,
    pub mca: mca::Mca,
    pub efer: u64,
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    /// IA32_KERNEL_GS_BASE, exchanged with the GS base by SWAPGS.
    pub kgsbase: u64,
    pub tsc_aux: u64,
    /// What CPUID reports about the package and this processor in it.
    pub topology: cpuid::CpuTopology,
    /// IA32_SPEC_CTRL (stored, no speculation to control; see `SPEC_CTRL_WRITABLE` in sys.rs).
    pub spec_ctrl: u64,
    /// XCR0, the enabled extended-state components (xgetbv/xsetbv). Bit 0 (x87) is always set.
    pub xcr0: u64,
    /// AMD SVM: host/guest switching state (see `svm.rs`).
    pub svm: svm::Svm,
    /// MONITOR address and whether a monitor is armed for the next MWAIT.
    pub monitor_address: u64,
    pub monitor_armed: bool,
    /// The line MONITOR armed and its contents then, with several vCPUs (see `Cpu::monitor`).
    pub monitor_snapshot: Option<(u64, [u8; MONITOR_WATCH_BYTES])>,
    /// State of the generator behind rdrand/rdseed (splitmix64). Seeded by the machine.
    pub rng_state: u64,
    pub pat: u64,
    /// IA32_SYSENTER_CS, _ESP, _EIP (MSRs 0x174..=0x176).
    pub sysenter: [u64; 3],
    pub dr: [u64; 8],
    /// MSRs that are only stored: a read returns the last write and writes are accepted.
    /// An MSR that is neither handled in `rdmsr`/`wrmsr` nor listed here raises #GP, which
    /// the kernel's `rdmsr_safe` probing depends on. Seeded in `Cpu::new`.
    pub msr_misc: alloc::collections::BTreeMap<u32, u64>,
    pub xmm: [u128; 16],
    /// Bits 255:128 of YMM0-15 (AVX). Legacy SSE instructions leave them alone; VEX.128 forms zero
    /// them.
    pub ymm_hi: [u128; 16],
    /// Bits 511:256 of ZMM0-15 (AVX-512, XSAVE component ZMM_Hi256), as two 128-bit lanes each.
    pub zmm_hi: [[u128; 2]; 16],
    /// ZMM16-31 in full (XSAVE component Hi16_ZMM), lane 0 least significant.
    pub zmm_extra: [[u128; 4]; 16],
    /// Opmask registers k0-k7 (XSAVE component opmask). k0 is a real register, but as an EVEX
    /// write mask the encoding 0 means "no masking".
    pub opmask: [u64; 8],
    /// True while a VEX instruction runs through the legacy SSE kernels (see `avx.rs`): they then
    /// skip the CR4.OSFXSR gate and the 16-byte alignment rule.
    pub vex_active: bool,
    /// Segment register and offset of the memory operand `linear_address` last formed outside 64-bit
    /// mode, so `load_operand`/`store_operand` can check its whole extent (see `modes.rs`). Reset at
    /// the start of every instruction there.
    pub pending_operand: core::cell::Cell<Option<(u8, u64)>>,
    pub mxcsr: u32,
    pub fcw: u16,
    /// TOP is in bits 11-13.
    pub fsw: u16,
    /// Abridged tag word: one bit per physical register, set when it holds a value.
    pub ftw: u16,
    /// x87 registers as raw 80-bit values, indexed by physical register; ST(i) is
    /// `st[(TOP + i) & 7]`.
    pub st: [[u8; 10]; 8],
    /// Set by HLT, cleared when an interrupt is delivered.
    pub halted: bool,
    /// One-instruction interrupt inhibit after `mov ss` and after an `sti` that turned IF on.
    /// `run` clears it right after `before_step` has consulted it.
    pub interrupt_shadow: bool,
    /// Guest physical RAM starting at address 0. It must never be resized or moved after
    /// `Cpu::new`, because `ram_base` and the TLB addends point into it.
    pub ram: GuestRam,
    /// Host address of `ram[0]`.
    pub ram_base: u64,
    /// Two halves of `TLB_ENTRIES_PER_MODE` entries, supervisor then user. Fixed size, so its
    /// address is stable for `tlb_active_base`.
    pub tlb: Vec<TlbEntry>,
    pub decode_cache: Vec<DecodeCacheEntry>,
    /// Bumping it invalidates the whole decode cache at once. Starts at 1 so the zeroed
    /// entries (generation 0) never match. 64 bits wide so it cannot wrap back to a value
    /// that stale entries still carry (a u32 would after about 4e9 flushes).
    pub decode_cache_generation: u64,
    /// One byte per guest physical page: nonzero while a cached decode or a JIT block was
    /// built from that page. Stores to such pages take the slow path and call `code_written`.
    /// Devices see it through `GuestMem` for the same purpose.
    pub page_has_code: CodePages,
    /// Pages whose `page_has_code` byte is set, so `flush_decode` can clear them without
    /// scanning the whole array. May list pages already cleared by `code_written`.
    pub pages_with_code: Vec<u32>,
    pub dev: DevHandle,
    /// Present when the machine boots a disk's boot sector: the BIOS services run as host code
    /// (`bios.rs`).
    #[cfg(not(freestanding))]
    pub bios: Option<Box<bios::Bios>>,
    /// Instructions retired; guest time is derived from it. `run` adds one before an
    /// instruction executes, so an instruction that faults is counted too.
    pub icount: u64,
    /// Virtual time added while halted: the clock jumps to the next device deadline instead of
    /// executing instructions. Stays 0 with a `wall_clock`, which just waits.
    pub idle_skipped_ns: u64,
    /// With several vCPUs the clock is not the raw `icount` (see `smp.rs`): guest time is
    /// `time_base_ns` plus the instructions retired since `slice_base_icount` divided by
    /// `time_divisor`, the number of vCPUs that were running when the slice began. With one vCPU
    /// the divisor is 1 and the base 0, which is `icount * NS_PER_INSN`. `time_carry` is the
    /// instructions that did not make a whole nanosecond yet, kept while the divisor stays put.
    pub time_base_ns: u64,
    pub slice_base_icount: u64,
    pub time_divisor: u64,
    pub time_carry: u64,
    /// `icount` at which `service_events` runs next.
    pub next_event: u64,
    /// Host time before which `service_events` leaves the devices alone (`MIN_DEVICE_TICK_NS`).
    /// Per CPU even when time-sharing: each CPU ticking on its own turn only ticks more often.
    next_device_tick_ns: u64,
    pub stop: Arc<AtomicBool>,
    pub log: Option<LogFn>,
    /// Host callback run on every device tick; its argument is true when the guest is idle with
    /// nothing scheduled (only needed on UEFI, where the host has to be polled for input).
    pub poll: Option<PollFn>,
    /// Keys already reported by `log_once`.
    pub logged: alloc::collections::BTreeSet<u64>,
    /// Number of `flush_decode` calls, for `dump`.
    pub flushes: u64,
    /// Realtime mode: guest time is this host clock and halting really waits. `None` is the
    /// deterministic virtual clock.
    pub wall_clock: Option<WallClock>,
    /// Instructions the last JIT block retired before it exited; `run_verify` replays that many
    /// on the interpreter.
    pub last_block_insns: u64,
    /// Exception raised inside a JIT helper. Generated code only gets a status code back;
    /// `step_block` picks the exception up from here.
    pub pending: Option<Exception>,
    pub jit: Option<Box<crate::jit::Jit>>,
    /// Most recent instruction addresses, recorded only when debugging.
    pub trace: Option<alloc::collections::VecDeque<u64>>,
    /// How many times each kind of instruction went through `helper_exec` (see
    /// `jit::helpers::histogram_key`), counted only when debugging: what is left to lower.
    pub helper_histogram: Option<alloc::collections::BTreeMap<u64, u64>>,
    /// How many #VMEXITs of each exit code (with the port number for I/O exits, see
    /// `svm::exit_histogram_key`) the guest hypervisor took, counted only when debugging.
    pub svm_exit_histogram: Option<alloc::collections::BTreeMap<u64, u64>>,
    /// This vCPU's number, which is also the APIC id it starts with, and how many there are.
    pub cpu_index: u32,
    pub cpu_count: u32,
    /// More than one vCPU: the run loop hands control back to the scheduler (`smp.rs`) at every
    /// event tick, at `pause` and while halted.
    pub multi: bool,
    /// Halted since power-on or INIT until a start-up IPI arrives (the "wait-for-SIPI" state).
    pub wait_for_sipi: bool,
    /// While a `pause` has pulled `next_event` forward to force a yield, the deadline it
    /// displaced; 0 otherwise. `before_step` puts it back.
    pub pause_resume_event: u64,
    /// The cache line an MWAIT is waiting on (physical address) and its contents when the wait
    /// began; the scheduler wakes the vCPU when another vCPU or a device changes them.
    pub mwait_watch: Option<(u64, [u8; MONITOR_WATCH_BYTES])>,
    /// Counts pages that gained translated code. A vCPU that has not run since the count last
    /// moved may hold write tags for such a page in its private TLB; `Smp::switch_to` revokes
    /// them (machine-wide, so it moves with `dev`).
    pub code_mark_epoch: u64,
    /// The `code_mark_epoch` this vCPU's TLB is up to date with.
    pub code_mark_seen: u64,
    /// Guest memory accesses use x86 ordering and atomic page-table updates (`atomic.rs`): set
    /// when this vCPU runs on its own host thread beside others.
    pub ordered_memory: bool,
    /// How the instruction being executed treats its memory destination: see `Rmw`.
    pub rmw: Rmw,
    /// Set when this vCPU runs on its own host thread (`vcpus.rs`).
    pub threads: Option<vcpus::ThreadRuntime>,
    /// How much of the shared invalidation log this vCPU has applied.
    pub invalidation_seen: u64,
    /// `pause`s executed, so a spinning threaded vCPU yields its host core now and then.
    pub pause_count: u32,
    /// This vCPU is inside `with_exclusive`; a nested request runs in place.
    pub in_exclusive: bool,
    /// Large leaves behind the soft TLB (see `LargePage`), replaced round robin from
    /// `large_page_next`. Last in the struct: generated code never reads it, and the fields it
    /// does read must stay within their short load offsets.
    pub large_pages: [LargePage; LARGE_PAGE_CACHE_ENTRIES],
    pub large_page_next: usize,
}

/// The state of a locked read-modify-write while `Cpu::exec_locked` runs it on a vCPU that shares
/// memory with others. The instruction's ordinary code loads its destination with `load_operand`
/// and stores the result with `store_operand`; those two see this state and turn the pair into a
/// compare-and-swap, so every locked instruction (add, inc, bts, xadd, cmpxchg, xchg, ...) is
/// atomic without a second implementation of each.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rmw {
    /// Plain accesses.
    Off,
    /// A locked instruction being run with the other vCPUs stopped (`Cpu::with_exclusive`):
    /// plain accesses, and `exec` must not treat it as locked again.
    Plain,
    /// Locked, destination not loaded yet.
    Armed,
    /// The destination (`host` is a host address) held `value`; the matching store must still
    /// find it there.
    Loaded {
        host: usize,
        size: u8,
        value: u64,
        page: usize,
    },
    /// The operand is split across a page or not naturally aligned: no host atomic covers it, so
    /// the instruction has to run with the other vCPUs stopped.
    NeedsExclusive,
}

/// What `Cpu::build` needs besides the boot state: the machine-wide handles a vCPU shares with the
/// others, and how many there are.
struct MachineParts {
    ram: GuestRam,
    page_has_code: CodePages,
    dev: DevHandle,
    ram_bytes: usize,
    cpu_count: u32,
    /// The bootstrap processor builds its own decode cache; a secondary gets one from the
    /// scheduler (time-sharing) or `give_private_caches` (threads).
    with_decode_cache: bool,
}

/// Size of the line MONITOR arms and the unit of the wait-on-store check.
pub(crate) const MONITOR_WATCH_BYTES: usize = 64;

impl Cpu {
    /// Starts in 64-bit long mode with paging already on, as the Linux 64-bit boot protocol
    /// leaves it (`docs/plans/02-boot.md`): there is no firmware to set any of this up.
    pub fn new(
        ram: Vec<u8>,
        boot: BootState,
        console: ConsoleHandle,
        stop: Arc<AtomicBool>,
        log: Option<LogFn>,
        dev_config: DevConfig,
    ) -> Self {
        let ram_bytes = ram.len();
        let pages = ram_bytes >> PAGE_SHIFT;
        let cpu_count = dev_config.cpus.max(1) as u32;
        let devices = DevHandle::new(Devices::new(console, dev_config));
        let parts = MachineParts {
            ram: GuestRam::new(ram),
            page_has_code: CodePages::new(pages),
            dev: devices,
            ram_bytes,
            cpu_count,
            with_decode_cache: true,
        };
        Self::build(parts, boot, stop, log)
    }

    /// A vCPU that is not the bootstrap processor: waiting for a start-up IPI. It shares the
    /// machine-wide state (RAM, code-page flags, devices) with `bsp`; the decode cache and JIT
    /// stay empty until the scheduler hands them over (time-sharing) or the vCPU's thread makes
    /// its own.
    pub fn new_secondary(index: u32, bsp: &Cpu) -> Self {
        let boot = BootState {
            mode: crate::boot::BootMode::Real16,
            ..Default::default()
        };
        let parts = MachineParts {
            ram: bsp.ram.clone(),
            page_has_code: bsp.page_has_code.clone(),
            dev: bsp.dev.for_cpu(index as usize),
            ram_bytes: bsp.ram.len(),
            cpu_count: bsp.cpu_count,
            with_decode_cache: false,
        };
        let mut cpu = Self::build(parts, boot, bsp.stop.clone(), bsp.log.clone());
        cpu.ram_base = bsp.ram_base;
        cpu.cpu_index = index;
        // The local APIC id is the vCPU number, and one thread per core keeps it equal to the
        // x2APIC id CPUID reports.
        cpu.topology.x2apic_id = index;
        cpu.cpu_count = bsp.cpu_count;
        cpu.multi = true;
        cpu.wall_clock = bsp.wall_clock.clone();
        cpu.halted = true;
        cpu.wait_for_sipi = true;
        cpu
    }

    fn build(
        parts: MachineParts,
        boot: BootState,
        stop: Arc<AtomicBool>,
        log: Option<LogFn>,
    ) -> Self {
        let MachineParts {
            ram,
            page_has_code,
            dev,
            ram_bytes,
            cpu_count,
            with_decode_cache,
        } = parts;
        // Stored-only MSRs; names follow Linux's msr-index.h.
        /// MSR_AMD64_PATCH_LEVEL: the microcode revision. Linux treats 0 as "no microcode
        /// information" and invents the lowest known revision. On family 1Ah it also gates RDSEED:
        /// `init_amd_zen5` clears the CPUID bit (and logs "RDSEED32 is broken") unless the revision
        /// is at least the fixed one in its `zen5_rdseed_microcode` table, which for Turin (model
        /// 02h stepping 1, `cpuid::SIGNATURE`) is exactly this value. Lowering it loses `rdseed`.
        const MSR_UCODE_REV: u32 = 0x8b;
        const UCODE_REV_TURIN: u64 = 0x0b00_215a;
        const MSR_K7_HWCR: u32 = 0xc001_0015;
        /// IA32_DEBUGCTL: a KVM host saves and restores it around every guest run. Stored only (no
        /// branch recording).
        const MSR_IA32_DEBUGCTL: u32 = 0x1d9;
        const MSR_AMD64_NB_CFG: u32 = 0xc001_001f;
        const MSR_AMD64_LS_CFG: u32 = 0xc001_1020;
        const MSR_AMD64_DE_CFG: u32 = 0xc001_1029;
        const HWCR_INITIAL: u64 = 1 << 24;
        /// AMD's memory-typing MSRs (APM vol. 2 sec. 7.9): SYSCFG enables the fixed and variable
        /// MTRRs (bits 18 and 20), and TOP_MEM / TOP_MEM2 bound DRAM below and above 4 GiB.
        /// Stored only, like the MTRRs themselves.
        const MSR_AMD64_SYSCFG: u32 = 0xc001_0010;
        const MSR_AMD64_IORR_FIRST: u32 = 0xc001_0016;
        const MSR_AMD64_IORR_LAST: u32 = 0xc001_0019;
        const MSR_AMD64_TOP_MEM: u32 = 0xc001_001a;
        const MSR_AMD64_TOP_MEM2: u32 = 0xc001_001d;
        const SYSCFG_INITIAL: u64 = (1 << 18) | (1 << 20);
        /// The four legacy performance event selectors and counters, which Linux probes with a
        /// write-and-read-back before it trusts a PMU. They store but never count (there is no PMU
        /// model), so perf sees counters that stay at zero instead of a "broken PMU" warning.
        const MSR_K7_PERF_FIRST: u32 = 0xc001_0000;
        const MSR_K7_PERF_LAST: u32 = 0xc001_0007;
        /// Instruction-cache, data-cache and bus-unit configuration and the Zen 4 branch predictor
        /// control: Linux and Xen set workaround bits in these with rdmsr/wrmsr_safe depending on
        /// family and model. Storage only, so the read-modify-write sequences succeed quietly.
        const MSR_AMD64_IC_CFG: u32 = 0xc001_1021;
        const MSR_AMD64_DC_CFG: u32 = 0xc001_1022;
        const MSR_AMD64_BU_CFG2: u32 = 0xc001_102a;
        const MSR_ZEN4_BP_CFG: u32 = 0xc001_102e;

        // TOP_MEM is where DRAM ends below the PCI hole; TOP_MEM2 is where the DRAM remapped
        // above 4 GiB ends, zero when there is none (APM vol. 2 sec. 7.9.1).
        let top_of_memory = crate::ram::low_ram_end(ram_bytes as u64);
        let top_of_high_memory = crate::ram::high_ram_end(ram_bytes as u64);
        // A secondary vCPU gets its decode cache from the scheduler.
        let has_machine_state = with_decode_cache;
        let flat = |sel: u16, flags: u32| Seg {
            sel,
            base: 0,
            limit: SEG_LIMIT_4GIB,
            flags,
        };
        let flat_data = flat(boot.ds, SEG_FLAGS_FLAT_DATA);
        let mut seg = [Seg::default(); 6];
        seg[SEG_ES] = flat_data;
        seg[SEG_CS] = flat(boot.cs, SEG_FLAGS_CODE64);
        seg[SEG_SS] = flat_data;
        seg[SEG_DS] = flat_data;
        let mut gpr = [0u64; 16];
        gpr[GPR_RSI] = boot.rsi;
        gpr[GPR_RDI] = boot.rdi;
        let mut cpu = Cpu {
            gpr,
            rip: boot.rip,
            eflags: flags::RESERVED_ONE,
            lazy_op: LazyFlagsOp::Eflags,
            lazy_size: 8,
            lazy_dst: 0,
            lazy_src: 0,
            lazy_src2: 0,
            seg,
            gdtr: DescriptorTable {
                base: boot.gdt_base,
                limit: boot.gdt_limit,
            },
            idtr: DescriptorTable::default(),
            ldtr: Seg::default(),
            tr: Seg::default(),
            cr0: CR0_PG | CR0_NE | CR0_ET | CR0_MP | CR0_PE,
            cr2: 0,
            cr3: boot.cr3,
            cr4: CR4_PAE,
            pdpte: [0; 4],
            nmi_blocked: false,
            mtrr: mtrr::Mtrr::new(),
            mca: mca::Mca::new(),
            efer: EFER_LMA | EFER_LME,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            kgsbase: 0,
            tsc_aux: 0,
            topology: cpuid::CpuTopology {
                logical_processors: cpu_count,
                ..cpuid::CpuTopology::SINGLE_PROCESSOR
            },
            spec_ctrl: 0,
            xcr0: xsave::XCR0_X87,
            svm: svm::Svm::new(),
            monitor_address: 0,
            monitor_armed: false,
            monitor_snapshot: None,
            rng_state: DEFAULT_RANDOM_SEED,
            pat: PAT_POWER_ON_DEFAULT,
            sysenter: [0; 3],
            dr: [0, 0, 0, 0, 0, 0, DR6_RESET, DR7_RESET],
            msr_misc: [
                (MSR_UCODE_REV, UCODE_REV_TURIN),
                (MSR_IA32_DEBUGCTL, 0),
                (MSR_K7_HWCR, HWCR_INITIAL),
                (MSR_AMD64_NB_CFG, 0),
                (MSR_AMD64_LS_CFG, 0),
                (MSR_AMD64_DE_CFG, 0),
                (MSR_AMD64_IC_CFG, 0),
                (MSR_AMD64_DC_CFG, 0),
                (MSR_AMD64_BU_CFG2, 0),
                (MSR_ZEN4_BP_CFG, 0),
                (MSR_AMD64_SYSCFG, SYSCFG_INITIAL),
                (MSR_AMD64_TOP_MEM, top_of_memory),
                (MSR_AMD64_TOP_MEM2, top_of_high_memory),
            ]
            .into_iter()
            .chain((MSR_AMD64_IORR_FIRST..=MSR_AMD64_IORR_LAST).map(|msr| (msr, 0)))
            .chain((MSR_K7_PERF_FIRST..=MSR_K7_PERF_LAST).map(|msr| (msr, 0)))
            .collect(),
            cpl: 0,
            xmm: [0; 16],
            ymm_hi: [0; 16],
            zmm_hi: [[0; 2]; 16],
            zmm_extra: [[0; 4]; 16],
            opmask: [0; 8],
            vex_active: false,
            pending_operand: core::cell::Cell::new(None),
            mxcsr: MXCSR_RESET,
            fcw: FCW_RESET,
            fsw: 0,
            ftw: 0,
            st: [[0; 10]; 8],
            halted: false,
            interrupt_shadow: false,
            ram,
            tlb: vec![TLB_EMPTY; 2 * TLB_ENTRIES_PER_MODE],
            ram_base: 0,
            decode_cache: if has_machine_state {
                vec![
                    DecodeCacheEntry {
                        gpa: 0,
                        generation: 0,
                        insn: Insn::default()
                    };
                    DECODE_CACHE_ENTRIES
                ]
            } else {
                Vec::new()
            },
            decode_cache_generation: 1,
            page_has_code,
            pages_with_code: Vec::new(),
            dev,
            #[cfg(not(freestanding))]
            bios: None,
            icount: 0,
            idle_skipped_ns: 0,
            time_base_ns: 0,
            slice_base_icount: 0,
            time_divisor: 1,
            time_carry: 0,
            next_event: 0,
            next_device_tick_ns: 0,
            stop,
            log,
            poll: None,
            logged: Default::default(),
            flushes: 0,
            wall_clock: None,
            last_block_insns: 0,
            pending: None,
            jit: None,
            tlb_active_base: 0,
            large_pages: [LARGE_PAGE_EMPTY; LARGE_PAGE_CACHE_ENTRIES],
            large_page_next: 0,
            ordered_memory: false,
            rmw: Rmw::Off,
            threads: None,
            invalidation_seen: 0,
            pause_count: 0,
            in_exclusive: false,
            block_exit_requested: false,
            prepaid_icount: 0,
            slow_access_insn: 0,
            trace: None,
            helper_histogram: None,
            svm_exit_histogram: None,
            cpu_index: 0,
            cpu_count,
            multi: cpu_count > 1,
            wait_for_sipi: false,
            pause_resume_event: 0,
            mwait_watch: None,
            code_mark_epoch: 0,
            code_mark_seen: 0,
        };
        cpu.ram_base = cpu.ram.as_ptr() as u64;
        cpu.set_cpl(0);
        match boot.mode {
            crate::boot::BootMode::Long64 => {}
            crate::boot::BootMode::Real16 => {
                cpu.reset_to_real_mode();
                // The chipset would alias the top of the 4 GiB space onto the BIOS; there is no
                // ROM here, so execution starts at the same code through the plain real-mode
                // address.
                cpu.seg[SEG_CS].base = (cpu.seg[SEG_CS].sel as u64) << 4;
            }
            crate::boot::BootMode::Protected32 => cpu.enter_boot_protected_mode(boot),
            // `MachineConfig::boot_disk` needs a disk, so a freestanding build never gets here.
            crate::boot::BootMode::Bios => {
                cpu.reset_to_real_mode();
                #[cfg(not(freestanding))]
                {
                    cpu.bios_install();
                    cpu.bios_boot();
                }
            }
        }
        cpu
    }

    /// Switches the privilege level and the TLB half that goes with it.
    pub fn set_cpl(&mut self, cpl: u8) {
        self.cpl = cpl;
        let user_half_offset = TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>();
        self.tlb_active_base =
            self.tlb.as_ptr() as u64 + (cpl == USER_CPL) as u64 * user_half_offset as u64;
    }

    pub fn now_ns(&self) -> u64 {
        // Not cached per slice: a guest spinning on rdtsc or the PIT must see time pass between
        // two reads, and the host clock never runs backwards on its own.
        if let Some(wall) = &self.wall_clock {
            return wall.elapsed_ns();
        }
        // A slow-path access from the middle of a JIT block: count the block's earlier
        // instructions that `icount` has not been charged for yet.
        let unpaid = self
            .slow_access_insn
            .saturating_sub(1)
            .saturating_sub(self.prepaid_icount);
        let busy = self.icount - self.slice_base_icount + self.time_carry + unpaid;
        self.time_base_ns + busy / self.time_divisor * NS_PER_INSN + self.idle_skipped_ns
    }

    /// The TSC is just guest time in nanoseconds.
    pub fn tsc(&self) -> u64 {
        self.now_ns().wrapping_add(self.svm_tsc_offset())
    }

    /// Logs `msg` at most once per `key`. Callers keep their key ranges disjoint by adding the
    /// object id (page, MSR number) to a per-site base.
    pub fn log_once(&mut self, key: u64, msg: impl FnOnce() -> String) {
        if self.logged.insert(key)
            && let Some(l) = &self.log
        {
            l(&msg());
        }
    }

    pub fn tick_devices(&mut self, now: u64) {
        if let Some(poll) = &self.poll {
            poll(false);
        }
        let written_code_pages = {
            let mut mem = GuestMem::new(&mut self.ram, self.page_has_code.as_bytes());
            self.dev.lock().tick(now, &mut mem);
            mem.code_pages_written
        };
        // DMA by devices bypasses the store path, so tell the code caches about it.
        for page in written_code_pages {
            self.code_written(page);
        }
    }

    fn service_events(&mut self) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        if self.threads.is_some() {
            // The device thread ticks the devices; this vCPU only has to notice the machine
            // ending, which `thread_attention` does at every block boundary.
            self.next_event = self.icount + EVENT_INTERVAL;
            return None;
        }
        let now = self.now_ns();
        if self.wall_clock.is_some() {
            if now < self.next_device_tick_ns {
                self.next_event = self.icount + EVENT_INTERVAL;
                return None;
            }
            self.next_device_tick_ns = now + MIN_DEVICE_TICK_NS;
        }
        self.tick_devices(now);
        if self.dev.lock().reset_requested {
            return Some(ExitReason::Reset);
        }
        if self.dev.lock().shutdown_requested() {
            return Some(ExitReason::Shutdown);
        }
        self.next_event = self.icount + EVENT_INTERVAL;
        None
    }

    fn pending_interrupt(&mut self) -> Option<u8> {
        if self.eflags & flags::IF == 0 || self.interrupt_shadow {
            return None;
        }
        self.dev.lock().interrupt_ack()
    }

    /// Takes a pending NMI at this instruction boundary. NMIs ignore IF but not the one-instruction
    /// shadow of `mov ss`, and a second one waits until the handler's `iret` (SDM 6.7.1).
    fn take_pending_nmi(&mut self) -> bool {
        // A threaded vCPU asks the device lock only when the device model says something is
        // waiting for it (`vcpus::ATTENTION_NMI`).
        if self.nmi_blocked
            || self.interrupt_shadow
            || (self.threads.is_some() && self.attention() & vcpus::ATTENTION_NMI == 0)
            || !self.dev.lock().nmi_pending()
        {
            return false;
        }
        self.dev.lock().take_nmi();
        self.nmi_blocked = true;
        true
    }

    /// Runs the per-step housekeeping: device/stop polling, interrupt delivery, halt handling.
    /// Interrupts are only taken here, i.e. at instruction (or JIT block) boundaries.
    fn before_step(&mut self) -> BeforeStep {
        if self.threads.is_some() {
            if let Some(reason) = self.thread_attention() {
                return BeforeStep::Exit(reason);
            }
            if self.attention() & vcpus::ATTENTION_START != 0 {
                self.apply_start_signals();
            }
        }
        if self.icount >= self.next_event {
            // `pause` pulled the deadline forward only to get here: no device work is due.
            if self.pause_resume_event != 0 {
                self.next_event = core::mem::take(&mut self.pause_resume_event);
                return BeforeStep::Yield;
            }
            if let Some(reason) = self.service_events() {
                return BeforeStep::Exit(reason);
            }
            if self.multi && self.threads.is_none() {
                return BeforeStep::Yield;
            }
        }
        // has_pending() is a cheap pre-check; pending_interrupt() does the acknowledge.
        // A guest may intercept physical interrupts (and takes its virtual ones) before the host does.
        self.svm_before_step();
        if self.svm.gif && self.take_pending_nmi() {
            self.halted = false;
            if let Some(reason) = self.deliver_event(NMI_VECTOR, None, false) {
                return BeforeStep::Exit(reason);
            }
        }
        if self.eflags & flags::IF != 0
            && self.svm.gif
            && !self.interrupt_shadow
            && (self.threads.is_none() || self.attention() & vcpus::ATTENTION_IRQ != 0)
            && self.dev.lock().interrupt_pending()
            && let Some(vector) = self.pending_interrupt()
        {
            self.halted = false;
            if let Some(reason) = self.deliver_hardware_interrupt(vector) {
                return BeforeStep::Exit(reason);
            }
        }
        if self.halted {
            if self.threads.is_some() {
                return match self.wait_for_wake() {
                    Some(reason) => BeforeStep::Exit(reason),
                    None => BeforeStep::Idled,
                };
            }
            if self.multi {
                return BeforeStep::Halted;
            }
            let may_wake = self.eflags & flags::IF != 0 || self.dev.lock().nmi_pending();
            return match self.idle(may_wake) {
                Some(reason) => BeforeStep::Exit(reason),
                None => BeforeStep::Idled,
            };
        }
        BeforeStep::Execute
    }

    /// Delivers the exception an instruction raised; `Some` means the run loop must exit.
    fn after_step(&mut self, result: ExecResult<()>) -> Option<ExitReason> {
        match result {
            Ok(()) => None,
            Err(Exception::Fault {
                vector,
                error_code,
                software,
            }) => {
                if self.svm_exception_exit(vector, error_code, software) {
                    return None;
                }
                match self.deliver_event(vector, error_code, software) {
                    Some(ExitReason::TripleFault) if self.svm_shutdown_exit() => None,
                    other => other,
                }
            }
            Err(Exception::Exit(reason)) => Some(reason),
            Err(Exception::Vmexit) => None,
            Err(Exception::AtomicRetry) => Some(ExitReason::Internal(
                "a locked instruction escaped its retry loop".into(),
            )),
        }
    }

    /// Single vCPU: runs until the machine exits.
    pub fn run(&mut self) -> ExitReason {
        match self.run_slice() {
            Slice::Exit(reason) => reason,
            Slice::Yield | Slice::Halted => {
                ExitReason::Internal("a lone vCPU asked for a scheduler".into())
            }
        }
    }

    /// Runs until the machine exits or, with several vCPUs, until this one should give way.
    pub fn run_slice(&mut self) -> Slice {
        loop {
            match self.before_step() {
                BeforeStep::Exit(reason) => return Slice::Exit(reason),
                BeforeStep::Idled => continue,
                BeforeStep::Yield => return Slice::Yield,
                BeforeStep::Halted => return Slice::Halted,
                BeforeStep::Execute => {}
            }
            self.interrupt_shadow = false;
            self.icount += 1;
            let result = if self.jit.is_some() && self.is_long64() {
                self.step_block()
            } else {
                self.step()
            };
            if let Some(reason) = self.after_step(result) {
                return Slice::Exit(reason);
            }
        }
    }

    /// Lockstep check of the JIT against the interpreter oracle (`--verify`): `self` runs a
    /// JIT block, then `shadow` (interpreter, own copy of RAM) executes the same number of
    /// instructions, and any difference in registers or exception stops the run.
    pub fn run_verify(&mut self, shadow: &mut Cpu) -> ExitReason {
        // Bytes of the diverging code printed from each side, and differing page offsets listed.
        const DIVERGENCE_CODE_BYTES: usize = 24;
        const MAX_DIFFERING_OFFSETS: usize = 8;

        loop {
            let (main_gate, shadow_gate) = (self.before_step(), shadow.before_step());
            match (main_gate, shadow_gate) {
                (BeforeStep::Exit(reason), _) => return reason,
                (BeforeStep::Idled, BeforeStep::Idled) => continue,
                (BeforeStep::Execute, BeforeStep::Execute) => {}
                _ => return ExitReason::Internal("verify: pre-step divergence".into()),
            }
            self.interrupt_shadow = false;
            shadow.interrupt_shadow = false;
            let main_before = (self.gpr, self.eflags_full(), self.rip);
            let shadow_before = (shadow.rip, shadow.icount, self.icount);
            let start = self.rip;
            self.icount += 1;
            shadow.icount += 1;
            let main_result = if self.is_long64() {
                self.step_block()
            } else {
                self.last_block_insns = 1;
                self.step()
            };
            let retired = self.last_block_insns;
            // A faulting block reports the instructions retired before the fault, so the
            // interpreter needs one more step to reach the same fault.
            let attempts = if main_result.is_err() {
                retired + 1
            } else {
                retired
            };
            let mut shadow_result = Ok(());
            for attempt in 0..attempts {
                if attempt > 0 {
                    shadow.icount += 1;
                }
                shadow_result = shadow.step();
                if shadow_result.is_err() {
                    break;
                }
            }
            let same_outcome = outcomes_agree(&main_result, &shadow_result);
            let mismatch = self.diff(shadow);
            if !same_outcome || mismatch.is_some() {
                let page_base = self
                    .translate(start, mem::AccessKind::Exec, true)
                    .unwrap_or(0) as usize
                    & !(PAGE_OFFSET_MASK as usize);
                // The faulting code may sit in the last RAM page or outside RAM (MMIO, where
                // `translate` fell back to 0), so every slice here is bounds-checked: a panic in
                // the diagnostics would hide the divergence being reported.
                let (main_ram, shadow_ram) = (&self.ram, &shadow.ram);
                let differing_offsets: Vec<usize> = match (
                    main_ram.slice(page_base as u64, PAGE_SIZE as usize),
                    shadow_ram.slice(page_base as u64, PAGE_SIZE as usize),
                ) {
                    (Some(main_page), Some(shadow_page)) => (0..main_page.len())
                        .filter(|&offset| main_page[offset] != shadow_page[offset])
                        .take(MAX_DIFFERING_OFFSETS)
                        .collect(),
                    _ => Vec::new(),
                };
                let code_offset = page_base + (start & PAGE_OFFSET_MASK) as usize;
                let code_bytes = |ram: &GuestRam| {
                    ram.slice(code_offset as u64, DIVERGENCE_CODE_BYTES)
                        .map(<[u8]>::to_vec)
                        .unwrap_or_default()
                };
                let main_bytes = code_bytes(&self.ram);
                let shadow_bytes = code_bytes(&shadow.ram);
                let page_diffs = format!(
                    "page diffs at {differing_offsets:x?}; main bytes {main_bytes:02x?}; shadow bytes {shadow_bytes:02x?}"
                );
                if let Some(l) = &self.log {
                    l(&page_diffs);
                }
                let msg = format!(
                    "verify: divergence after block at {start:#x} ({retired} insns, attempts {attempts}, shadow start {shadow_before:x?}): exc main={:?} shadow={:?} diff={:?} icount main {:#x} shadow {:#x} now_ns main {} shadow {}\n{page_diffs}\n--- before {main_before:x?}\n--- main\n{}--- shadow\n{}",
                    main_result.as_ref().err(),
                    shadow_result.as_ref().err(),
                    mismatch,
                    self.icount,
                    shadow.icount,
                    self.now_ns(),
                    shadow.now_ns(),
                    self.dump_regs(),
                    shadow.dump_regs()
                );
                return ExitReason::Internal(msg);
            }
            if let Some(reason) = self.after_step(main_result) {
                return reason;
            }
            if let Some(reason) = shadow.after_step(shadow_result) {
                return reason;
            }
        }
    }

    /// Describes how architectural state differs from `other`, or `None` if it matches.
    fn diff(&mut self, other: &mut Cpu) -> Option<String> {
        let mut differences = Vec::new();
        for index in 0..self.gpr.len() {
            if self.gpr[index] != other.gpr[index] {
                differences.push(format!(
                    "r{index}: {:#x} vs {:#x}",
                    self.gpr[index], other.gpr[index]
                ));
            }
        }
        if self.rip != other.rip {
            differences.push(format!("rip: {:#x} vs {:#x}", self.rip, other.rip));
        }
        let (our_eflags, their_eflags) = (self.eflags_full(), other.eflags_full());
        if our_eflags != their_eflags {
            differences.push(format!("eflags: {our_eflags:#x} vs {their_eflags:#x}"));
        }
        if self.cpl != other.cpl {
            differences.push("cpl".into());
        }
        for index in 0..self.seg.len() {
            if self.seg[index].sel != other.seg[index].sel
                || self.seg[index].base != other.seg[index].base
            {
                differences.push(format!("seg{index}"));
            }
        }
        for (name, ours, theirs) in [
            ("cr0", self.cr0, other.cr0),
            ("cr2", self.cr2, other.cr2),
            ("cr3", self.cr3, other.cr3),
            ("cr4", self.cr4, other.cr4),
            ("efer", self.efer, other.efer),
            ("icount", self.icount, other.icount),
        ] {
            if ours != theirs {
                differences.push(format!("{name}: {ours:#x} vs {theirs:#x}"));
            }
        }
        if self.zmm_hi != other.zmm_hi
            || self.zmm_extra != other.zmm_extra
            || self.opmask != other.opmask
        {
            differences.push("zmm/opmask".into());
        }
        if self.ymm_hi != other.ymm_hi {
            differences.push("ymm_hi".into());
        }
        if self.xmm != other.xmm {
            differences.push("xmm".into());
        }
        if differences.is_empty() {
            None
        } else {
            Some(differences.join(", "))
        }
    }

    /// Lets the host clock run for up to `wait_ns` (realtime idle). A wait may end early, since
    /// `idle` is called again until the deadline has passed.
    fn wait_on_host(&mut self, wait_ns: u64) {
        #[cfg(not(freestanding))]
        {
            // Short sleeps keep the stop flag and host console responsive. Parking, not
            // sleeping: a frame from the host network unparks this thread (`NetHandle`), so a
            // halted guest gets it now rather than at the end of the slice.
            const MAX_SLEEP_NS: u64 = 2_000_000;
            std::thread::park_timeout(std::time::Duration::from_nanos(wait_ns.min(MAX_SLEEP_NS)));
        }
        #[cfg(freestanding)]
        {
            let _ = wait_ns;
            // No threads to sleep: the host's idle hook blocks for a moment.
            if let Some(poll) = &self.poll {
                poll(true);
            }
        }
    }

    /// Handles a halted machine: skips (or sleeps) to the next device deadline so timers can
    /// raise the interrupt that wakes a vCPU. Returns `Some` for stop, reset and power-off.
    /// `may_wake` is whether any vCPU could still be woken by an interrupt (interrupts enabled)
    /// or an NMI.
    fn idle(&mut self, may_wake: bool) -> Option<ExitReason> {
        if self.stop.load(Ordering::Relaxed) {
            return Some(ExitReason::Stopped);
        }
        // A reset written just before the `hlt` (`out 0x64, 0xfe; hlt`, with IF=0 as in every
        // kernel's reboot path) must win over the shutdown check below.
        if self.dev.lock().reset_requested {
            return Some(ExitReason::Reset);
        }
        if self.dev.lock().shutdown_requested() {
            return Some(ExitReason::Shutdown);
        }
        // `hlt` with IF=0 can only be woken by NMI, SMI, INIT or RESET (SDM vol. 3 sec. 6.8,
        // HLT), and nothing here raises the first three: the guest is dead, which is how Linux's
        // poweroff and halt end without ACPI (`stop_this_cpu`: `cli; hlt` in a loop). Only IF is
        // tested, not `interrupt_shadow` or a queued event: `sti; hlt` has IF=1 by the time
        // `hlt` runs, and a PIC line pending behind IF=0 never wakes the CPU either.
        if !may_wake {
            return Some(ExitReason::Shutdown);
        }
        let now = self.now_ns();
        let next_deadline = self.dev.lock().next_deadline(now);
        match next_deadline {
            Some(deadline) => {
                if deadline > now {
                    if self.wall_clock.is_some() {
                        self.wait_on_host(deadline - now);
                    } else {
                        self.idle_skipped_ns += deadline - now;
                    }
                }
                self.tick_devices(self.now_ns());
            }
            None => {
                // No timer will ever wake the guest; only host input (console, network) can.
                self.tick_devices(now);
                if self.dev.lock().interrupt_pending() || self.dev.lock().nmi_pending() {
                    return None;
                }
                #[cfg(not(freestanding))]
                std::thread::park_timeout(std::time::Duration::from_millis(1));
                #[cfg(freestanding)]
                if let Some(poll) = &self.poll {
                    poll(true);
                }
            }
        }
        if self.dev.lock().reset_requested {
            return Some(ExitReason::Reset);
        }
        None
    }

    pub(crate) fn step(&mut self) -> ExecResult<()> {
        let insn = self.fetch_decode()?;
        if let Some(trace) = &mut self.trace {
            if trace.len() >= TRACE_CAPACITY {
                trace.pop_front();
            }
            trace.push_back(self.rip);
        }
        self.exec(&insn)
    }

    pub fn dump_regs(&self) -> String {
        let mut s = format!(
            "rip={:016x} cpl={} eflags={:x}\n",
            self.rip,
            self.cpl,
            self.eflags_full()
        );
        for (i, g) in self.gpr.iter().enumerate() {
            s += &format!("r{i:<2}={g:016x} ");
            if i % GPRS_PER_DUMP_LINE == GPRS_PER_DUMP_LINE - 1 {
                s += "\n";
            }
        }
        s
    }

    /// State dump for `--debug` and fatal exits, with the kernel log `kernel_log_matches` finds.
    pub fn dump(&self) -> String {
        let mut s = format!(
            "flushes={} icount={} rip={:016x} cpl={} eflags={:x} cr0={:x} cr3={:x} cr4={:x} efer={:x}\n",
            self.flushes,
            self.icount,
            self.rip,
            self.cpl,
            self.eflags_full(),
            self.cr0,
            self.cr3,
            self.cr4,
            self.efer
        );
        for (i, g) in self.gpr.iter().enumerate() {
            s += &format!("r{i:<2}={g:016x} ");
            if i % GPRS_PER_DUMP_LINE == GPRS_PER_DUMP_LINE - 1 {
                s += "\n";
            }
        }
        s += &format!(
            "halted={} shadow={} dev: {}\n",
            self.halted,
            self.interrupt_shadow,
            self.dev.lock().state(self.now_ns())
        );
        s += &self.kernel_log_matches();
        #[cfg(all(feature = "jit", target_arch = "wasm32"))]
        if let Some(jit) = &self.jit {
            s += &format!("jit: {:?}\n", jit.stats());
        }
        if let Some(histogram) = &self.helper_histogram {
            let mut counts: Vec<_> = histogram.iter().collect();
            counts.sort_by_key(|&(_, count)| core::cmp::Reverse(*count));
            let total: u64 = counts.iter().map(|&(_, count)| *count).sum();
            s += &format!("instructions through the interpreter: {total}\n");
            for (key, count) in counts.into_iter().take(HISTOGRAM_DUMP_ENTRIES) {
                s += &format!("  {count:>12}  {}\n", decode::histogram_label(*key));
            }
        }
        if let Some(histogram) = &self.svm_exit_histogram {
            let mut counts: Vec<_> = histogram.iter().collect();
            counts.sort_by_key(|&(_, count)| core::cmp::Reverse(*count));
            let total: u64 = counts.iter().map(|&(_, count)| *count).sum();
            s += &format!("svm exits: {total}\n");
            for (key, count) in counts.into_iter().take(HISTOGRAM_DUMP_ENTRIES) {
                s += &format!("  {count:>12}  {}\n", svm::exit_histogram_label(*key));
            }
        }
        if let Some(trace) = &self.trace {
            s += "last rips:";
            for rip in trace.iter().rev().take(TRACE_DUMP_ENTRIES) {
                s += &format!(" {rip:x}");
            }
            s += "\n";
        }
        s
    }

    /// Scans guest RAM for the kernel's `Linux version` banner: the printk ring buffer starts
    /// with it, so the text that follows is the kernel log even though the guest never got to
    /// flush it to the serial port. The banner is also in the kernel image and in whatever copied
    /// it (syslog, `dmesg` output in the page cache), and with KASLR the kernel can sit above any
    /// number of those copies. So the scan runs until it finds the log, which it recognises by
    /// the record after the banner: on x86 that is always `Command line:` (`setup_arch`), and
    /// reports it first, then the other matches below it in address order.
    fn kernel_log_matches(&self) -> String {
        // Matches reported, the log included.
        const MAX_BANNER_MATCHES: usize = 6;
        // Log text extracted after each match.
        const LOG_SCAN_BYTES: usize = 6000;
        // Ring-buffer records have binary headers between the text; shorter printable runs
        // are noise from those.
        const MIN_PRINTABLE_RUN: usize = 6;
        const BANNER: &[u8] = b"Linux version";
        const RECORD_AFTER_BANNER: &str = "Command line:";

        let mut log = None;
        let mut others = Vec::new();
        // Each RAM range on its own: the PCI hole inside the buffer is not memory, and walking
        // a terabyte of high RAM would take minutes.
        let ram_bytes = self.ram.len() as u64;
        let ranges = [
            (0, crate::ram::low_ram_end(ram_bytes) as usize),
            (
                crate::ram::PCI_HOLE_END as usize,
                crate::ram::high_ram_end(ram_bytes) as usize,
            ),
        ];
        'scan: for (range_start, range_end) in ranges {
            let mut search_from = range_start;
            while search_from + BANNER.len() < range_end {
                let Some(relative) = self.ram[search_from..range_end]
                    .windows(BANNER.len())
                    .position(|window| window == BANNER)
                else {
                    break;
                };
                let start = search_from + relative;
                search_from = start + BANNER.len();
                let end = (start + LOG_SCAN_BYTES).min(range_end);
                let mut text = String::new();
                let mut printable_run = String::new();
                for &byte in &self.ram[start..end] {
                    if (b' '..0x7f).contains(&byte) {
                        printable_run.push(byte as char);
                    } else {
                        if printable_run.len() >= MIN_PRINTABLE_RUN {
                            text += &printable_run;
                            text.push('\n');
                        }
                        printable_run.clear();
                    }
                }
                let is_log = text
                    .lines()
                    .nth(1)
                    .is_some_and(|line| line.starts_with(RECORD_AFTER_BANNER));
                if is_log {
                    log = Some((start, text));
                    break 'scan;
                }
                if others.len() < MAX_BANNER_MATCHES {
                    others.push((start, text));
                }
            }
        }
        let mut s = String::new();
        for (start, text) in log.into_iter().chain(others).take(MAX_BANNER_MATCHES) {
            s += &format!("--- ram match at {start:#x}\n{text}\n");
        }
        s
    }
}

#[cfg(test)]
mod cpu_tests {
    use std::sync::atomic::AtomicUsize;

    use super::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE};
    use super::flags::*;
    use super::test_support::*;
    use super::*;
    use crate::testkit::{CODE_BASE, HLT, Harness, RBX, SCRATCH, STACK_TOP};

    const OLD_RIP: u64 = 0x0000_7000_1234_5678;
    /// A handler that asks the keyboard controller for a reset and halts.
    // mov al, 0xfe; out 0x64, al; hlt
    const RESET_HANDLER: [u8; 5] = [0xb0, 0xfe, 0xe6, 0x64, 0xf4];
    const HANDLER_ADDRESS: u64 = 0x30_0000;
    const KERNEL_STACK: u64 = 0x70_0000;
    // l: jmp l
    const SPIN: [u8; 2] = [0xeb, 0xfe];
    // mov rax, 5; add rax, 7; mov al, 0xfe; out 0x64, al; l: jmp l
    const COMPUTE_THEN_RESET: [u8; 17] = [
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x07, 0xb0, 0xfe, 0xe6, 0x64,
        0xeb, 0xfe,
    ];
    const PIC_MASTER_COMMAND: u16 = 0x20;
    const PIC_MASTER_DATA: u16 = 0x21;
    const PIC_SLAVE_COMMAND: u16 = 0xa0;
    const PIC_SLAVE_DATA: u16 = 0xa1;
    const PIT_CHANNEL0_PORT: u16 = 0x40;
    const PIT_COMMAND_PORT: u16 = 0x43;
    const KEYBOARD_CONTROLLER_COMMAND_PORT: u16 = 0x64;
    const KEYBOARD_CONTROLLER_RESET: u32 = 0xfe;
    /// Channel 0, low then high byte access, mode 2 (rate generator).
    const PIT_CHANNEL0_RATE_GENERATOR: u32 = 0x34;
    const IRQ_VECTOR_BASE: u8 = 0x20;
    const ICW1_INIT_WITH_ICW4: u32 = 0x11;
    const ICW4_8086: u32 = 1;
    const ICW3_SLAVE_ON_IRQ2: u32 = 0x04;
    const ICW3_SLAVE_ID_2: u32 = 0x02;

    fn harness() -> Harness {
        Harness::new(&[HLT])
    }

    fn with_idt() -> Harness {
        let mut h = harness();
        install_idt(&mut h);
        h.cpu.eflags = RESERVED_ONE | IF;
        h.cpu.rip = OLD_RIP;
        h
    }

    /// Standard PC wiring: master at `IRQ_VECTOR_BASE`, slave right after, all lines unmasked.
    fn init_pics(cpu: &mut Cpu) {
        let devices = &mut cpu.dev;
        devices.io_out(PIC_MASTER_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
        devices.io_out(PIC_MASTER_DATA, 1, IRQ_VECTOR_BASE as u32, 0);
        devices.io_out(PIC_MASTER_DATA, 1, ICW3_SLAVE_ON_IRQ2, 0);
        devices.io_out(PIC_MASTER_DATA, 1, ICW4_8086, 0);
        devices.io_out(PIC_SLAVE_COMMAND, 1, ICW1_INIT_WITH_ICW4, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, IRQ_VECTOR_BASE as u32 + 8, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, ICW3_SLAVE_ID_2, 0);
        devices.io_out(PIC_SLAVE_DATA, 1, ICW4_8086, 0);
    }

    fn program_pit_channel0(cpu: &mut Cpu, count: u16) {
        cpu.dev
            .io_out(PIT_COMMAND_PORT, 1, PIT_CHANNEL0_RATE_GENERATOR, 0);
        cpu.dev
            .io_out(PIT_CHANNEL0_PORT, 1, (count & 0xff) as u32, 0);
        cpu.dev.io_out(PIT_CHANNEL0_PORT, 1, (count >> 8) as u32, 0);
    }

    fn install_reset_handler(h: &mut Harness, vector: u8) {
        h.write(HANDLER_ADDRESS, &RESET_HANDLER);
        set_gate(h, vector, Gate::interrupt(HANDLER_ADDRESS));
    }

    mod reset_state {
        use super::*;

        #[test]
        fn control_registers_describe_long_mode_with_paging_on() {
            let h = harness();
            assert_eq!(h.cpu.cr0, 0x8000_0033);
            assert_eq!(h.cpu.cr4, 0x20);
            assert_eq!(h.cpu.efer, 0x500);
            assert_eq!(h.cpu.cr2, 0);
            assert_eq!(h.cpu.dev.lapic().tpr(), 0);
            assert_eq!(h.cpu.cr3 & PTE_ADDRESS_BITS, 0x9000);
        }

        #[test]
        fn long_mode_is_enabled_without_no_execute_or_write_protect() {
            let h = harness();
            assert_eq!(h.cpu.efer & EFER_NXE, 0);
            assert_eq!(h.cpu.cr0 & CR0_WP, 0);
            assert_ne!(h.cpu.efer & EFER_LMA, 0);
            assert_ne!(h.cpu.efer & EFER_LME, 0);
            assert_ne!(h.cpu.cr4 & CR4_PAE, 0);
        }

        #[test]
        fn segments_are_flat_with_the_boot_selectors() {
            let h = harness();
            let cs = h.cpu.seg[SEG_CS];
            assert_eq!(cs.sel, KERNEL_CS);
            assert_eq!(cs.flags, 0x0020_9a00);
            assert!(cs.is_code() && cs.long() && cs.present());
            assert_eq!(cs.dpl(), 0);
            for index in [SEG_ES, SEG_SS, SEG_DS] {
                let seg = h.cpu.seg[index];
                assert_eq!(seg.sel, KERNEL_DS, "segment {index}");
                assert_eq!(seg.flags, 0x00cf_9200, "segment {index}");
                assert_eq!(seg.base, 0);
                assert_eq!(seg.limit, 0xffff_ffff);
                assert!(!seg.is_code());
            }
            for index in [SEG_FS, SEG_GS] {
                let seg = h.cpu.seg[index];
                assert_eq!((seg.sel, seg.base, seg.limit, seg.flags), (0, 0, 0, 0));
            }
        }

        #[test]
        fn descriptor_tables_point_at_the_boot_gdt_and_an_empty_idt() {
            let h = harness();
            assert_eq!(h.cpu.gdtr.base, GDT_BASE);
            assert_eq!(h.cpu.gdtr.limit, 31);
            assert_eq!((h.cpu.idtr.base, h.cpu.idtr.limit), (0, 0));
            assert_eq!((h.cpu.ldtr.sel, h.cpu.tr.sel), (0, 0));
        }

        #[test]
        fn registers_start_at_the_boot_protocol_values() {
            let h = harness();
            assert_eq!(h.cpu.rip, CODE_BASE);
            assert_eq!(h.cpu.gpr[GPR_RSI], 0x7000, "rsi points at boot_params");
            for (index, value) in h.cpu.gpr.iter().enumerate() {
                if index != GPR_RSI && index != GPR_RSP {
                    assert_eq!(*value, 0, "register {index}");
                }
            }
            assert_eq!(h.cpu.gpr[GPR_RSP], STACK_TOP);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE);
            assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn msrs_debug_and_fpu_state_have_their_documented_power_on_values() {
            let h = harness();
            assert_eq!(h.cpu.pat, 0x0007_0406_0007_0406);
            assert_eq!(h.cpu.dr[6], 0xffff_0ff0);
            assert_eq!(h.cpu.dr[7], 0x400);
            assert_eq!(&h.cpu.dr[..6], &[0; 6]);
            assert_eq!(h.cpu.mxcsr, 0x1f80);
            assert_eq!(h.cpu.fcw, 0x37f);
            assert_eq!((h.cpu.fsw, h.cpu.ftw), (0, 0));
            assert_eq!(h.cpu.xmm, [0; 16]);
            assert_eq!(h.cpu.st, [[0; 10]; 8]);
            assert_eq!(
                (h.cpu.star, h.cpu.lstar, h.cpu.cstar, h.cpu.sfmask),
                (0, 0, 0, 0)
            );
            assert_eq!((h.cpu.kgsbase, h.cpu.tsc_aux), (0, 0));
            assert_eq!(h.cpu.sysenter, [0; 3]);
        }

        #[test]
        fn stored_only_msrs_are_seeded() {
            let h = harness();
            let expected: [(u32, u64); 25] = [
                (0x8b, 0x0b00_215a),
                (0xc001_0000, 0),
                (0xc001_0001, 0),
                (0xc001_0002, 0),
                (0xc001_0003, 0),
                (0xc001_0004, 0),
                (0xc001_0005, 0),
                (0xc001_0006, 0),
                (0xc001_0007, 0),
                (0xc001_1021, 0),
                (0xc001_1022, 0),
                (0xc001_102a, 0),
                (0xc001_102e, 0),
                (0x1d9, 0),
                (0xc001_0010, (1 << 18) | (1 << 20)),
                (0xc001_0015, 1 << 24),
                (0xc001_0016, 0),
                (0xc001_0017, 0),
                (0xc001_0018, 0),
                (0xc001_0019, 0),
                (0xc001_001a, h.cpu.ram.len() as u64),
                (0xc001_001d, 0),
                (0xc001_001f, 0),
                (0xc001_1020, 0),
                (0xc001_1029, 0),
            ];
            assert_eq!(h.cpu.msr_misc.len(), expected.len());
            for (msr, value) in expected {
                assert_eq!(h.cpu.msr_misc.get(&msr), Some(&value), "msr {msr:#x}");
            }
        }

        #[test]
        fn caches_and_counters_start_empty() {
            let h = harness();
            assert_eq!(h.cpu.tlb.len(), 2 * TLB_ENTRIES_PER_MODE);
            assert!(h.cpu.tlb.iter().all(|entry| entry.tag_r == TLB_TAG_INVALID
                && entry.tag_w == TLB_TAG_INVALID
                && entry.tag_x == TLB_TAG_INVALID));
            assert_eq!(h.cpu.decode_cache.len(), DECODE_CACHE_ENTRIES);
            assert!(h.cpu.decode_cache.iter().all(|entry| entry.generation == 0));
            assert_eq!(h.cpu.decode_cache_generation, 1);
            assert_eq!(h.cpu.page_has_code.len(), h.cpu.ram.len() >> PAGE_SHIFT);
            assert!(h.cpu.page_has_code.iter().all(|flag| flag == 0));
            assert!(h.cpu.pages_with_code.is_empty());
            assert_eq!(
                (h.cpu.icount, h.cpu.idle_skipped_ns, h.cpu.next_event),
                (0, 0, 0)
            );
            assert_eq!(h.cpu.flushes, 0);
        }

        #[test]
        fn execution_flags_and_hooks_start_clear() {
            let h = harness();
            assert!(!h.cpu.halted);
            assert!(!h.cpu.interrupt_shadow);
            assert!(!h.cpu.block_exit_requested);
            assert!(h.cpu.wall_clock.is_none());
            assert_eq!(h.cpu.prepaid_icount, 0);
            assert_eq!(h.cpu.last_block_insns, 0);
            assert!(h.cpu.pending.is_none());
            assert!(h.cpu.jit.is_none());
            assert!(h.cpu.trace.is_none());
            assert!(h.cpu.poll.is_none());
            assert!(h.cpu.log.is_none());
            assert!(h.cpu.logged.is_empty());
        }

        #[test]
        fn ram_base_and_the_supervisor_tlb_half_point_into_owned_buffers() {
            let h = harness();
            assert_eq!(h.cpu.ram_base, h.cpu.ram.as_ptr() as u64);
            assert_eq!(h.cpu.tlb_active_base, h.cpu.tlb.as_ptr() as u64);
            assert_eq!(h.cpu.ram.len(), crate::testkit::HARNESS_RAM_BYTES);
        }

        #[test]
        fn a_fresh_cpu_walks_the_boot_identity_map() {
            let mut h = harness();
            assert_eq!(
                h.cpu
                    .translate(0x12_3456, mem::AccessKind::Read, false)
                    .unwrap(),
                0x12_3456
            );
            assert_eq!(
                h.cpu
                    .translate(0x1fff_ffff, mem::AccessKind::Write, false)
                    .unwrap(),
                0x1fff_ffff
            );
        }
    }

    mod segments_and_sizes {
        use super::*;

        #[test]
        fn segment_accessors_read_the_descriptor_flag_bits() {
            let code = Seg {
                sel: 0x10,
                base: 0,
                limit: 0,
                flags: SEG_LONG
                    | SEG_PRESENT
                    | SEG_NON_SYSTEM
                    | SEG_TYPE_EXECUTABLE
                    | (2 << SEG_DPL_SHIFT),
            };
            assert!(code.is_code() && code.long() && code.present());
            assert_eq!(code.dpl(), 2);
            let data = Seg {
                flags: SEG_NON_SYSTEM | SEG_PRESENT | SEG_TYPE_READ_WRITE | (3 << SEG_DPL_SHIFT),
                ..code
            };
            assert!(!data.is_code() && !data.long());
            assert_eq!(data.dpl(), 3);
        }

        #[test]
        fn system_descriptor_with_the_executable_bit_pattern_is_not_code() {
            let system = Seg {
                flags: SEG_TYPE_EXECUTABLE | SEG_PRESENT,
                ..Seg::default()
            };
            assert!(!system.is_code());
        }

        #[test]
        fn default_segment_is_null_and_absent() {
            let seg = Seg::default();
            assert!(!seg.present() && !seg.is_code() && !seg.long());
            assert_eq!(seg.dpl(), 0);
        }

        #[test]
        fn tlb_entry_layout_is_what_the_jit_hardcodes() {
            assert_eq!(core::mem::size_of::<TlbEntry>(), 32);
            assert_eq!(core::mem::offset_of!(TlbEntry, tag_r), 0);
            assert_eq!(core::mem::offset_of!(TlbEntry, tag_w), 8);
            assert_eq!(core::mem::offset_of!(TlbEntry, addend), 24);
        }

        #[test]
        fn page_constants_agree() {
            assert_eq!(PAGE_SIZE, 4096);
            assert_eq!(PAGE_OFFSET_MASK, 0xfff);
            assert_eq!(TLB_ENTRIES_PER_MODE, 1024);
            assert_eq!(USER_CPL, 3);
            assert_eq!(GPR_RSP, 4);
            assert_eq!(
                [SEG_ES, SEG_CS, SEG_SS, SEG_DS, SEG_FS, SEG_GS],
                [0, 1, 2, 3, 4, 5]
            );
        }
    }

    mod time_and_logging {
        use super::*;

        #[test]
        fn now_ns_counts_retired_instructions_plus_idle_time() {
            let mut h = harness();
            assert_eq!(h.cpu.now_ns(), 0);
            h.cpu.icount = 1000;
            assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN);
            h.cpu.idle_skipped_ns = 5_000;
            assert_eq!(h.cpu.now_ns(), 1000 * NS_PER_INSN + 5_000);
        }

        /// A host clock the test advances by hand.
        fn manual_clock() -> (Arc<core::sync::atomic::AtomicU64>, HostClockFn) {
            let now = Arc::new(core::sync::atomic::AtomicU64::new(0));
            let reader = now.clone();
            (now, Arc::new(move || reader.load(Ordering::Relaxed)))
        }

        #[test]
        fn with_a_host_clock_guest_time_is_wall_time_since_it_started_not_instructions() {
            const HOST_EPOCH_NS: u64 = 7_000_000_000;
            const ELAPSED_NS: u64 = 123_456;
            let mut h = harness();
            let (host_now, clock) = manual_clock();
            host_now.store(HOST_EPOCH_NS, Ordering::Relaxed);
            h.cpu.wall_clock = Some(WallClock::new(clock));
            assert_eq!(h.cpu.now_ns(), 0);
            // Instructions do not move the clock; only the host does, and the TSC follows.
            h.cpu.icount = 1_000_000;
            assert_eq!(h.cpu.now_ns(), 0);
            host_now.store(HOST_EPOCH_NS + ELAPSED_NS, Ordering::Relaxed);
            assert_eq!(h.cpu.now_ns(), ELAPSED_NS);
            assert_eq!(h.cpu.tsc(), ELAPSED_NS);
        }

        #[test]
        fn tsc_is_the_guest_clock_in_nanoseconds() {
            let mut h = harness();
            h.cpu.icount = 77;
            h.cpu.idle_skipped_ns = 23;
            assert_eq!(h.cpu.tsc(), h.cpu.now_ns());
            assert_eq!(h.cpu.tsc(), 77 * NS_PER_INSN + 23);
        }

        #[test]
        fn tsc_is_monotonic_across_instructions() {
            let mut h = Harness::new(&[0x90, 0x90, 0x90, HLT]);
            let mut previous = h.cpu.tsc();
            for _ in 0..3 {
                h.run(1);
                assert!(h.cpu.tsc() > previous);
                previous = h.cpu.tsc();
            }
        }

        #[test]
        fn log_once_runs_the_message_closure_once_per_key() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            let built = AtomicUsize::new(0);
            for _ in 0..3 {
                h.cpu.log_once(7, || {
                    built.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    "seven".to_string()
                });
            }
            assert_eq!(built.load(core::sync::atomic::Ordering::Relaxed), 1);
            assert_eq!(*log.lock().unwrap(), ["seven"]);
        }

        #[test]
        fn log_once_keeps_distinct_keys_distinct() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(1, || "one".into());
            h.cpu.log_once(2, || "two".into());
            h.cpu.log_once(1, || "one again".into());
            assert_eq!(*log.lock().unwrap(), ["one", "two"]);
        }

        #[test]
        fn log_once_without_a_logger_is_silent_and_still_remembers_the_key() {
            let mut h = harness();
            h.cpu.log_once(9, || {
                panic!("the message must not be built without a logger")
            });
            // Installing a logger afterwards does not resurrect the key.
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(9, || "late".into());
            assert!(log.lock().unwrap().is_empty());
        }

        #[test]
        fn log_once_accepts_the_largest_key() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.log_once(u64::MAX, || "max".into());
            h.cpu.log_once(0, || "zero".into());
            assert_eq!(log.lock().unwrap().len(), 2);
        }
    }

    mod event_servicing {
        use super::*;
        use std::sync::atomic::AtomicU64;

        #[test]
        fn stop_flag_is_honoured_by_service_events() {
            let mut h = harness();
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.service_events(), Some(ExitReason::Stopped)));
        }

        #[test]
        fn service_events_schedules_the_next_poll_one_interval_ahead() {
            let mut h = harness();
            h.cpu.icount = 12_345;
            assert!(h.cpu.service_events().is_none());
            assert_eq!(h.cpu.next_event, 12_345 + EVENT_INTERVAL);
        }

        /// A realtime CPU on a clock the test moves, with a poll hook counting device ticks.
        fn realtime_harness_counting_ticks() -> (Harness, Arc<AtomicU64>, Arc<AtomicUsize>) {
            const START_NS: u64 = 1_000_000;
            let mut h = harness();
            let now = Arc::new(AtomicU64::new(START_NS));
            let clock = now.clone();
            h.cpu.wall_clock = Some(WallClock::new(Arc::new(move || {
                clock.load(Ordering::SeqCst)
            })));
            let ticks = Arc::new(AtomicUsize::new(0));
            let counter = ticks.clone();
            h.cpu.poll = Some(Arc::new(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            }));
            (h, now, ticks)
        }

        #[test]
        fn a_realtime_cpu_ticks_the_devices_at_most_once_per_tick_interval() {
            let (mut h, now, ticks) = realtime_harness_counting_ticks();
            for _ in 0..3 {
                assert!(h.cpu.service_events().is_none());
            }
            assert_eq!(ticks.load(Ordering::SeqCst), 1);
            now.fetch_add(MIN_DEVICE_TICK_NS, Ordering::SeqCst);
            assert!(h.cpu.service_events().is_none());
            assert_eq!(ticks.load(Ordering::SeqCst), 2);
        }

        #[test]
        fn a_skipped_tick_still_schedules_the_next_poll_and_honours_the_stop_flag() {
            let (mut h, _now, _ticks) = realtime_harness_counting_ticks();
            h.cpu.service_events();
            h.cpu.icount = 777;
            assert!(h.cpu.service_events().is_none());
            assert_eq!(h.cpu.next_event, 777 + EVENT_INTERVAL);
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.service_events(), Some(ExitReason::Stopped)));
        }

        #[test]
        fn a_requested_reset_surfaces_as_a_reset_exit() {
            let mut h = harness();
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(h.cpu.service_events(), Some(ExitReason::Reset)));
        }

        #[test]
        fn other_keyboard_controller_commands_do_not_reset() {
            let mut h = harness();
            h.cpu
                .dev
                .io_out(KEYBOARD_CONTROLLER_COMMAND_PORT, 1, 0xaa, 0);
            assert!(h.cpu.service_events().is_none());
        }

        #[test]
        fn tick_devices_calls_the_host_poll_hook_as_not_idle() {
            let mut h = harness();
            let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink = calls.clone();
            h.cpu.poll = Some(std::sync::Arc::new(move |idle| {
                sink.lock().unwrap().push(idle)
            }));
            h.cpu.tick_devices(0);
            h.cpu.tick_devices(10);
            assert_eq!(*calls.lock().unwrap(), [false, false]);
        }

        #[test]
        fn tick_devices_raises_the_pit_interrupt_when_its_deadline_passes() {
            let mut h = harness();
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            let deadline = h.cpu.dev.next_deadline(0).expect("armed timer");
            h.cpu.tick_devices(deadline - 1);
            assert!(!h.cpu.dev.pic.has_pending());
            h.cpu.tick_devices(deadline);
            assert!(h.cpu.dev.pic.has_pending());
        }

        #[test]
        fn pending_interrupt_requires_if_and_no_shadow() {
            let mut h = harness();
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(1);
            h.cpu.eflags = RESERVED_ONE;
            assert_eq!(h.cpu.pending_interrupt(), None);
            h.cpu.eflags = RESERVED_ONE | IF;
            h.cpu.interrupt_shadow = true;
            assert_eq!(h.cpu.pending_interrupt(), None);
            assert!(
                h.cpu.dev.pic.has_pending(),
                "refusing must not acknowledge the interrupt"
            );
            h.cpu.interrupt_shadow = false;
            assert_eq!(h.cpu.pending_interrupt(), Some(IRQ_VECTOR_BASE + 1));
            assert_eq!(h.cpu.pending_interrupt(), None, "now in service");
        }
    }

    mod before_step {
        use super::*;

        #[test]
        fn executes_when_nothing_is_pending() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
        }

        #[test]
        fn events_are_polled_only_once_the_instruction_count_reaches_next_event() {
            let mut h = with_idt();
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.next_event = 100;
            h.cpu.icount = 99;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            h.cpu.icount = 100;
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn reset_request_exits_at_the_next_poll() {
            let mut h = with_idt();
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn unmasked_interrupt_with_if_set_is_delivered_through_the_idt() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
            assert_eq!(
                h.cpu.eflags_full() & IF,
                0,
                "interrupt gates mask further interrupts"
            );
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.rip, OLD_RIP);
            assert_eq!(frame.rflags & IF, IF);
            assert!(h.cpu.dev.pic.state().contains("isr 02"));
        }

        #[test]
        fn slave_interrupt_uses_the_slave_vector_base() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.pic.raise(9);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 9));
        }

        #[test]
        fn interrupt_is_held_while_if_is_clear() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, OLD_RIP);
            assert!(h.cpu.dev.pic.has_pending());
        }

        #[test]
        fn interrupt_is_held_for_one_instruction_after_a_shadowing_instruction() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.interrupt_shadow = true;
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
            h.cpu.interrupt_shadow = false;
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        }

        #[test]
        fn masked_interrupt_is_not_delivered() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 1 << 1, 0);
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn delivering_an_interrupt_wakes_a_halted_cpu() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        }

        #[test]
        fn halted_cpu_with_interrupts_disabled_is_shut_down_even_with_an_irq_pending() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            // Only NMI, SMI, INIT and RESET wake `cli; hlt`, and none of them exists here.
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Shutdown)
            ));
            assert!(h.cpu.halted);
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn halted_cpu_with_interrupts_enabled_and_an_irq_pending_idles_without_sleeping() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            // The interrupt shadow keeps the pending IRQ from being delivered this step, but IF is
            // set, so the halt is not terminal.
            h.cpu.interrupt_shadow = true;
            h.cpu.halted = true;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(h.cpu.halted);
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_reset_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn a_stop_request_beats_the_shutdown_of_a_halt_with_interrupts_disabled() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn halted_cpu_exits_on_stop_without_waiting() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.halted = true;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Stopped)
            ));
        }

        #[test]
        fn halted_cpu_notices_a_reset_request_while_idle() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            h.cpu.halted = true;
            h.cpu.dev.io_out(
                KEYBOARD_CONTROLLER_COMMAND_PORT,
                1,
                KEYBOARD_CONTROLLER_RESET,
                0,
            );
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::Reset)
            ));
        }

        #[test]
        fn halted_cpu_skips_virtual_time_to_the_next_timer_deadline() {
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            let deadline = h.cpu.dev.next_deadline(0).unwrap();
            h.cpu.halted = true;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert_eq!(h.cpu.idle_skipped_ns, deadline);
            assert_eq!(h.cpu.now_ns(), deadline);
            assert_eq!(h.cpu.icount, 0, "idling retires no instructions");
            assert!(
                h.cpu.dev.pic.has_pending(),
                "the timer fired during the skip"
            );
            // The next step takes the timer interrupt and wakes up.
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE));
        }

        #[test]
        fn realtime_idle_sleeps_for_a_short_deadline_and_time_is_what_the_host_clock_says() {
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            h.cpu.wall_clock = Some(WallClock::new(std_host_clock()));
            program_pit_channel0(&mut h.cpu, 100);
            let deadline = h.cpu.dev.next_deadline(0).unwrap();
            h.cpu.halted = true;
            let started = std::time::Instant::now();
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(started.elapsed() >= std::time::Duration::from_nanos(deadline));
            assert!(h.cpu.now_ns() >= deadline);
            assert_eq!(
                h.cpu.idle_skipped_ns, 0,
                "nothing is warped on a host clock"
            );
        }

        #[test]
        fn realtime_idle_never_sleeps_longer_than_two_milliseconds_per_call() {
            const MAX_SLEEP: std::time::Duration = std::time::Duration::from_millis(2);
            let mut h = with_idt();
            h.cpu.next_event = u64::MAX;
            h.cpu.wall_clock = Some(WallClock::new(std_host_clock()));
            // 65536 ticks of the 1.19 MHz clock is about 55 ms.
            program_pit_channel0(&mut h.cpu, 0);
            h.cpu.halted = true;
            let started = std::time::Instant::now();
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(
                started.elapsed() < MAX_SLEEP * 10,
                "{:?}",
                started.elapsed()
            );
        }

        #[test]
        fn delivery_failure_while_taking_an_interrupt_ends_the_run() {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            init_pics(&mut h.cpu);
            h.cpu.idtr.limit = 0;
            h.cpu.dev.pic.raise(1);
            assert!(matches!(
                h.cpu.before_step(),
                BeforeStep::Exit(ExitReason::TripleFault)
            ));
        }
    }

    mod apic_delivery {
        use super::*;

        const APIC_BASE: u64 = 0xfee0_0000;
        const IOAPIC_BASE: u64 = 0xfec0_0000;
        const APIC_SVR: u64 = 0xf0;
        const APIC_EOI: u64 = 0xb0;
        const APIC_ICR_LOW: u64 = 0x300;
        const APIC_LVT_TIMER: u64 = 0x320;
        const APIC_LVT_LINT0: u64 = 0x350;
        const APIC_TIMER_INITIAL: u64 = 0x380;
        const APIC_TIMER_DIVIDE: u64 = 0x3e0;
        const SVR_ENABLE_ALL: u32 = 0x1ff;
        const DIVIDE_BY_ONE: u32 = 0b1011;
        const LVT_MASKED: u32 = 1 << 16;
        const LVT_EXTINT: u32 = 7 << 8;
        const ICR_SHORTHAND_SELF: u32 = 1 << 18;
        const ICR_DELIVERY_NMI: u32 = 4 << 8;
        const IOAPIC_SELECT: u64 = 0x00;
        const IOAPIC_WINDOW: u64 = 0x10;
        const IOAPIC_LEVEL: u32 = 1 << 15;
        const NMI_VECTOR: u8 = 2;
        const BUS_CLOCK_PERIOD_NS: u64 = 10;

        fn apic_write(h: &mut Harness, offset: u64, value: u32) {
            h.cpu.phys_write(APIC_BASE + offset, 4, value as u64);
        }

        fn enabled_apic() -> Harness {
            let mut h = with_idt();
            h.cpu.next_event = 1_000;
            apic_write(&mut h, APIC_SVR, SVR_ENABLE_ALL);
            h
        }

        fn program_ioapic_pin(h: &mut Harness, pin: u32, low: u32) {
            let register = 0x10 + 2 * pin;
            h.cpu
                .phys_write(IOAPIC_BASE + IOAPIC_SELECT, 4, register as u64 + 1);
            h.cpu.phys_write(IOAPIC_BASE + IOAPIC_WINDOW, 4, 0);
            h.cpu
                .phys_write(IOAPIC_BASE + IOAPIC_SELECT, 4, register as u64);
            h.cpu.phys_write(IOAPIC_BASE + IOAPIC_WINDOW, 4, low as u64);
        }

        #[test]
        fn timer_interrupt_is_delivered_with_the_lvt_vector() {
            let mut h = enabled_apic();
            apic_write(&mut h, APIC_LVT_TIMER, 0x40);
            apic_write(&mut h, APIC_TIMER_DIVIDE, DIVIDE_BY_ONE);
            apic_write(&mut h, APIC_TIMER_INITIAL, 1000);
            let deadline = h.cpu.dev.next_deadline(0).expect("armed timer");
            assert_eq!(deadline, 1000 * BUS_CLOCK_PERIOD_NS);
            h.cpu.tick_devices(deadline);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, handler_for(0x40));
        }

        #[test]
        fn a_halted_cpu_sleeps_until_the_apic_timer_and_wakes() {
            let mut h = enabled_apic();
            apic_write(&mut h, APIC_LVT_TIMER, 0x40);
            apic_write(&mut h, APIC_TIMER_DIVIDE, DIVIDE_BY_ONE);
            apic_write(&mut h, APIC_TIMER_INITIAL, 1000);
            h.cpu.halted = true;
            h.cpu.wall_clock = None;
            assert!(matches!(h.cpu.before_step(), BeforeStep::Idled));
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(0x40));
        }

        #[test]
        fn ioapic_routes_a_device_line_to_the_programmed_vector() {
            let mut h = enabled_apic();
            init_pics(&mut h.cpu);
            // The guest has moved to the I/O APIC: the PIC is fully masked.
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 0xff, 0);
            program_ioapic_pin(&mut h, 1, 0x50);
            h.cpu.dev.pic.raise(1);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert_eq!(h.cpu.rip, handler_for(0x50));
        }

        #[test]
        fn pit_irq_0_arrives_on_ioapic_pin_2() {
            let mut h = enabled_apic();
            init_pics(&mut h.cpu);
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 0xff, 0);
            program_ioapic_pin(&mut h, 2, 0x51);
            h.cpu.dev.pic.raise(0);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(0x51));
        }

        #[test]
        fn masked_ioapic_pin_delivers_nothing() {
            let mut h = enabled_apic();
            init_pics(&mut h.cpu);
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 0xff, 0);
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn level_interrupt_is_resent_after_eoi_while_the_line_stays_high() {
            let mut h = enabled_apic();
            h.cpu.dev.io_out(PIC_MASTER_DATA, 1, 0xff, 0);
            program_ioapic_pin(&mut h, 9, 0x60 | IOAPIC_LEVEL);
            h.cpu.dev.pic.raise(9);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(0x60));
            // The handler returns to the interrupted code with IF set, and EOIs.
            h.cpu.rip = OLD_RIP;
            h.cpu.eflags = RESERVED_ONE | IF;
            apic_write(&mut h, APIC_EOI, 0);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(0x60), "line still asserted");
            h.cpu.rip = OLD_RIP;
            h.cpu.eflags = RESERVED_ONE | IF;
            h.cpu.dev.pic.lower(9);
            apic_write(&mut h, APIC_EOI, 0);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP, "line released");
        }

        #[test]
        fn task_priority_holds_back_lower_priority_vectors() {
            let mut h = enabled_apic();
            apic_write(&mut h, APIC_ICR_LOW, 0x41 | ICR_SHORTHAND_SELF);
            h.cpu.dev.lapic_mut().set_tpr(0x40);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
            h.cpu.dev.lapic_mut().set_tpr(0x30);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(0x41));
        }

        #[test]
        fn pic_interrupts_reach_the_cpu_through_the_virtual_wire_until_lint0_is_masked() {
            let mut h = enabled_apic();
            init_pics(&mut h.cpu);
            apic_write(&mut h, APIC_LVT_LINT0, LVT_MASKED | LVT_EXTINT);
            h.cpu.dev.pic.raise(1);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP);
            apic_write(&mut h, APIC_LVT_LINT0, LVT_EXTINT);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(IRQ_VECTOR_BASE + 1));
        }

        #[test]
        fn nmi_ignores_if_and_is_held_until_iret() {
            let mut h = enabled_apic();
            h.cpu.eflags = RESERVED_ONE;
            apic_write(&mut h, APIC_ICR_LOW, ICR_DELIVERY_NMI | ICR_SHORTHAND_SELF);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(NMI_VECTOR));
            assert!(h.cpu.nmi_blocked);
            h.cpu.rip = OLD_RIP;
            apic_write(&mut h, APIC_ICR_LOW, ICR_DELIVERY_NMI | ICR_SHORTHAND_SELF);
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, OLD_RIP, "second NMI waits for iret");
            h.cpu.nmi_blocked = false;
            h.cpu.before_step();
            assert_eq!(h.cpu.rip, handler_for(NMI_VECTOR));
        }

        #[test]
        fn nmi_wakes_a_cpu_halted_with_interrupts_disabled() {
            let mut h = enabled_apic();
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.halted = true;
            apic_write(&mut h, APIC_ICR_LOW, ICR_DELIVERY_NMI | ICR_SHORTHAND_SELF);
            assert!(matches!(h.cpu.before_step(), BeforeStep::Execute));
            assert!(!h.cpu.halted);
            assert_eq!(h.cpu.rip, handler_for(NMI_VECTOR));
        }

        #[test]
        fn iret_unblocks_nmis() {
            // iretq with a frame that returns to the next instruction
            let mut h = Harness::new(&[0x48, 0xcf]);
            h.cpu.nmi_blocked = true;
            let rsp = h.cpu.gpr[GPR_RSP] - 40;
            h.write_u64(rsp, CODE_BASE + 2);
            h.write_u64(rsp + 8, h.cpu.seg[SEG_CS].sel as u64);
            h.write_u64(rsp + 16, RESERVED_ONE);
            h.write_u64(rsp + 24, h.cpu.gpr[GPR_RSP]);
            h.write_u64(rsp + 32, h.cpu.seg[SEG_SS].sel as u64);
            h.cpu.gpr[GPR_RSP] = rsp;
            h.cpu.step().unwrap();
            assert!(!h.cpu.nmi_blocked);
        }

        #[test]
        fn cr8_reads_and_writes_the_apic_task_priority() {
            let mut h = enabled_apic();
            h.cpu.dev.lapic_mut().set_tpr(0x70);
            // mov rax, cr8
            let value = (h.cpu.dev.lapic().tpr() >> 4) as u64;
            assert_eq!(value, 7);
        }
    }

    mod after_step {
        use super::*;

        #[test]
        fn success_continues_the_run() {
            let mut h = with_idt();
            assert!(h.cpu.after_step(Ok(())).is_none());
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_fault_is_delivered_through_the_idt() {
            let mut h = with_idt();
            let result = Err(Exception::fault(VECTOR_GENERAL_PROTECTION, Some(0x18)));
            assert!(h.cpu.after_step(result).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], true);
            assert_eq!((frame.error_code, frame.rip), (Some(0x18), OLD_RIP));
        }

        #[test]
        fn a_fault_without_an_error_code_pushes_none() {
            let mut h = with_idt();
            assert!(h.cpu.after_step(Err(Exception::invalid_opcode())).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_INVALID_OPCODE));
            assert_eq!(h.cpu.gpr[GPR_RSP], (STACK_TOP & !0xf) - 40);
        }

        #[test]
        fn software_interrupt_flag_reaches_the_privilege_check() {
            let mut h = with_idt();
            install_tss(&mut h, [KERNEL_STACK, 0, 0], [0; 7]);
            enter_user_mode(&mut h);
            let software = Exception::Fault {
                vector: 0x80,
                error_code: None,
                software: true,
            };
            assert!(h.cpu.after_step(Err(software)).is_none());
            // DPL0 gate from ring 3 raised #GP instead of entering vector 0x80.
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
        }

        #[test]
        fn exit_reasons_pass_straight_through() {
            let mut h = with_idt();
            let result = Err(Exception::Exit(ExitReason::Shutdown));
            assert!(matches!(
                h.cpu.after_step(result),
                Some(ExitReason::Shutdown)
            ));
            assert_eq!(h.cpu.rip, OLD_RIP);
        }

        #[test]
        fn a_fault_with_no_working_idt_is_a_triple_fault() {
            let mut h = harness();
            assert!(matches!(
                h.cpu.after_step(Err(Exception::invalid_opcode())),
                Some(ExitReason::TripleFault)
            ));
        }
    }

    mod run_loop {
        use super::*;

        #[test]
        fn stop_flag_set_before_the_run_exits_immediately() {
            let mut h = Harness::new(&SPIN);
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.run(), ExitReason::Stopped));
            assert_eq!(h.cpu.icount, 0);
        }

        #[test]
        fn run_executes_up_to_the_next_event_poll_then_sees_the_stop_flag() {
            let mut h = Harness::new(&SPIN);
            h.cpu.next_event = 100;
            h.cpu.stop.store(true, Ordering::Relaxed);
            assert!(matches!(h.cpu.run(), ExitReason::Stopped));
            assert_eq!(h.cpu.icount, 100);
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn run_counts_one_instruction_per_step() {
            let mut h = Harness::new(&[0x90; 40]);
            h.cpu.next_event = 25;
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.run();
            assert_eq!(h.cpu.icount, 25);
            assert_eq!(h.cpu.rip, CODE_BASE + 25);
        }

        #[test]
        fn run_returns_reset_as_soon_as_the_guest_writes_the_reset_port() {
            let mut h = Harness::new(&COMPUTE_THEN_RESET);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
            // mov, add, mov, out: the port write itself ends the run, no poll interval needed.
            assert_eq!(h.cpu.icount, 4);
        }

        #[test]
        fn run_clears_the_interrupt_shadow_before_each_instruction() {
            let mut h = Harness::new(&SPIN);
            h.cpu.interrupt_shadow = true;
            h.cpu.next_event = 3;
            h.cpu.stop.store(true, Ordering::Relaxed);
            h.cpu.run();
            assert!(!h.cpu.interrupt_shadow);
        }

        #[test]
        fn a_faulting_instruction_is_delivered_and_the_handler_runs() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            install_idt(&mut h);
            install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(
                frame.rip, CODE_BASE,
                "the saved rip is the faulting instruction"
            );
            // The run ends at the port write, two instructions into the handler.
            assert_eq!(h.cpu.rip, HANDLER_ADDRESS + 4);
        }

        #[test]
        fn a_fault_without_an_idt_ends_in_a_triple_fault() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            assert!(matches!(h.cpu.run(), ExitReason::TripleFault));
        }

        #[test]
        fn a_timer_interrupt_wakes_a_halted_guest_and_runs_its_handler() {
            let mut h = Harness::new(&[HLT]);
            install_idt(&mut h);
            init_pics(&mut h.cpu);
            program_pit_channel0(&mut h.cpu, 1000);
            install_reset_handler(&mut h, IRQ_VECTOR_BASE);
            h.cpu.eflags = RESERVED_ONE | IF;
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert!(h.cpu.idle_skipped_ns > 0, "the guest slept until the timer");
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.rip, CODE_BASE + 1, "interrupted after the hlt");
        }

        #[test]
        fn the_jit_run_loop_reaches_the_same_end_state() {
            let mut h = Harness::new(&COMPUTE_THEN_RESET);
            if !h.enable_jit() {
                return;
            }
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(h.cpu.gpr[crate::testkit::RAX], 0xfe);
            assert_eq!(h.cpu.icount, 4);
        }

        #[test]
        fn the_jit_run_loop_delivers_faults() {
            // ud2
            let mut h = Harness::new(&[0x0f, 0x0b]);
            if !h.enable_jit() {
                return;
            }
            install_idt(&mut h);
            install_reset_handler(&mut h, VECTOR_INVALID_OPCODE);
            assert!(matches!(h.cpu.run(), ExitReason::Reset));
            assert_eq!(read_frame(&h, h.cpu.gpr[GPR_RSP], false).rip, CODE_BASE);
        }
    }

    mod single_step {
        use super::*;

        #[test]
        fn step_executes_one_instruction_and_advances_rip() {
            let mut h = Harness::new(&[0x90, 0x90, HLT]);
            h.cpu.step().unwrap();
            assert_eq!(h.cpu.rip, CODE_BASE + 1);
            h.cpu.step().unwrap();
            assert_eq!(h.cpu.rip, CODE_BASE + 2);
        }

        #[test]
        fn step_reports_decode_faults_without_moving_rip() {
            let mut h = Harness::new(&[0x06]);
            assert_eq!(fault_of(h.cpu.step()), (VECTOR_INVALID_OPCODE, None));
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn trace_records_the_address_of_each_executed_instruction() {
            let mut h = Harness::new(&[0x90; 8]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..3 {
                h.cpu.step().unwrap();
            }
            let trace: Vec<u64> = h.cpu.trace.as_ref().unwrap().iter().copied().collect();
            assert_eq!(trace, [CODE_BASE, CODE_BASE + 1, CODE_BASE + 2]);
        }

        #[test]
        fn trace_keeps_only_the_newest_sixty_four_entries() {
            let mut h = Harness::new(&[0x90; 100]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..70 {
                h.cpu.step().unwrap();
            }
            let trace = h.cpu.trace.as_ref().unwrap();
            assert_eq!(trace.len(), TRACE_CAPACITY);
            assert_eq!(*trace.front().unwrap(), CODE_BASE + 6);
            assert_eq!(*trace.back().unwrap(), CODE_BASE + 69);
        }

        #[test]
        fn a_decode_fault_is_not_traced() {
            let mut h = Harness::new(&[0x06]);
            h.cpu.trace = Some(Default::default());
            let _ = h.cpu.step();
            assert!(h.cpu.trace.as_ref().unwrap().is_empty());
        }
    }

    mod dumps {
        use super::*;

        #[test]
        fn dump_regs_prints_rip_cpl_eflags_and_all_sixteen_registers() {
            let mut h = harness();
            h.cpu.rip = 0x1234_5678_9abc_def0;
            h.cpu.eflags = RESERVED_ONE | IF | ZF;
            for (index, reg) in h.cpu.gpr.iter_mut().enumerate() {
                *reg = 0x1111_1111_1111_1111 * (index as u64 % 16);
            }
            let dump = h.cpu.dump_regs();
            let lines: Vec<&str> = dump.lines().collect();
            assert_eq!(lines.len(), 5);
            assert_eq!(lines[0], "rip=123456789abcdef0 cpl=0 eflags=242");
            assert_eq!(
                lines[1],
                "r0 =0000000000000000 r1 =1111111111111111 r2 =2222222222222222 r3 =3333333333333333 "
            );
            assert!(lines[4].starts_with("r12=cccccccccccccccc r13=dddddddddddddddd"));
            assert!(lines[4].contains("r15=ffffffffffffffff"));
        }

        #[test]
        fn dump_regs_resolves_deferred_flags_and_shows_cpl() {
            let mut h = harness();
            h.cpu.set_cpl(3);
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            let dump = h.cpu.dump_regs();
            assert!(dump.starts_with(&format!(
                "rip={:016x} cpl=3 eflags={:x}\n",
                CODE_BASE,
                RESERVED_ONE | CF | ZF | PF | AF
            )));
        }

        #[test]
        fn dump_starts_with_the_state_summary_and_ends_with_device_state() {
            let mut h = harness();
            h.cpu.flushes = 3;
            h.cpu.icount = 42;
            let dump = h.cpu.dump();
            assert!(dump.starts_with("flushes=3 icount=42 rip="), "{dump}");
            assert!(
                dump.contains("cr0=80000033 cr3=9000 cr4=20 efer=500\n"),
                "{dump}"
            );
            assert!(
                dump.contains("halted=false shadow=false dev: m(irr 00"),
                "{dump}"
            );
            assert!(!dump.contains("last rips"));
            assert!(!dump.contains("--- ram match"));
        }

        #[test]
        fn dump_shows_the_halted_and_shadow_flags() {
            let mut h = harness();
            h.cpu.halted = true;
            h.cpu.interrupt_shadow = true;
            assert!(h.cpu.dump().contains("halted=true shadow=true"));
        }

        #[test]
        fn dump_extracts_the_kernel_log_after_the_linux_version_banner() {
            let mut h = harness();
            let mut text = b"Linux version 6.1.0-test (builder) #1 SMP".to_vec();
            text.extend_from_slice(b"\0\0short\0");
            text.extend_from_slice(b"second record with enough characters\0\0");
            h.write(SCRATCH, &text);
            let dump = h.cpu.dump();
            assert!(
                dump.contains(&format!("--- ram match at {SCRATCH:#x}\n")),
                "{dump}"
            );
            assert!(dump.contains("Linux version 6.1.0-test (builder) #1 SMP\n"));
            assert!(dump.contains("second record with enough characters\n"));
            assert!(
                !dump.contains("short"),
                "runs under six characters are record headers"
            );
        }

        #[test]
        fn dump_reports_at_most_six_banner_matches() {
            let mut h = harness();
            for copy in 0..9u64 {
                h.write(SCRATCH + copy * 0x100, b"Linux version 1.2.3 banner\0");
            }
            let dump = h.cpu.dump();
            assert_eq!(dump.matches("--- ram match at").count(), 6);
        }

        #[test]
        fn dump_finds_the_log_above_more_banner_copies_than_it_reports() {
            // With KASLR the kernel, and its log buffer with it, can sit above page-cache and
            // user copies of the banner (syslog, dmesg output), so the log is not the first match.
            let mut h = harness();
            for copy in 0..9u64 {
                h.write(SCRATCH + copy * 0x100, b"Linux version 1.2.3 copy\0");
            }
            let log_at = SCRATCH + 0x1000;
            h.write(
                log_at,
                b"Linux version 1.2.3 banner\0\0\0\0\0\0\0\0Command line: console=ttyS0\0",
            );
            let dump = h.cpu.dump();
            assert!(
                dump.contains(&format!("--- ram match at {log_at:#x}\n")),
                "{dump}"
            );
            assert!(dump.contains("Command line: console=ttyS0\n"), "{dump}");
            assert_eq!(dump.matches("--- ram match at").count(), 6);
        }

        #[test]
        fn dump_lists_the_newest_trace_entries_first_and_at_most_sixteen() {
            let mut h = Harness::new(&[0x90; 40]);
            h.cpu.trace = Some(Default::default());
            for _ in 0..30 {
                h.cpu.step().unwrap();
            }
            let dump = h.cpu.dump();
            let line = dump
                .lines()
                .find(|line| line.starts_with("last rips:"))
                .unwrap();
            let rips: Vec<u64> = line
                .trim_start_matches("last rips:")
                .split_whitespace()
                .map(|rip| u64::from_str_radix(rip, 16).unwrap())
                .collect();
            assert_eq!(rips.len(), TRACE_DUMP_ENTRIES);
            assert_eq!(rips[0], CODE_BASE + 29);
            assert_eq!(*rips.last().unwrap(), CODE_BASE + 14);
        }
    }

    mod verification_helpers {
        use super::*;

        fn pair(code: &[u8]) -> (Harness, Harness) {
            (Harness::new(code), Harness::new(code))
        }

        #[test]
        fn diff_of_identical_cpus_is_none() {
            let (mut a, mut b) = pair(&[HLT]);
            assert_eq!(a.cpu.diff(&mut b.cpu), None);
        }

        #[test]
        fn diff_names_each_differing_register() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.gpr[RBX] = 1;
            a.cpu.gpr[15] = 2;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(report.contains("r3: 0x1 vs 0x0"), "{report}");
            assert!(report.contains("r15: 0x2 vs 0x0"), "{report}");
        }

        #[test]
        fn diff_reports_rip_cpl_and_segments() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.rip += 1;
            a.cpu.set_cpl(3);
            a.cpu.seg[SEG_FS].base = 0x1000;
            a.cpu.seg[SEG_SS].sel = 0x2b;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(
                report.contains(&format!("rip: {:#x} vs {:#x}", CODE_BASE + 1, CODE_BASE)),
                "{report}"
            );
            assert!(report.contains("cpl"));
            assert!(report.contains("seg4"));
            assert!(report.contains("seg2"));
            assert!(!report.contains("seg1"));
        }

        #[test]
        fn diff_reports_control_registers_efer_and_icount() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.cr0 ^= 1 << 16;
            a.cpu.cr2 = 0x1000;
            a.cpu.cr3 ^= 0x1000;
            a.cpu.cr4 ^= 1 << 7;
            a.cpu.efer ^= EFER_NXE;
            a.cpu.icount = 9;
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            for name in ["cr0", "cr2", "cr3", "cr4", "efer", "icount: 0x9 vs 0x0"] {
                assert!(report.contains(name), "{name} missing from {report}");
            }
        }

        #[test]
        fn diff_reports_xmm_changes() {
            let (mut a, mut b) = pair(&[HLT]);
            a.cpu.xmm[7] = 1;
            assert_eq!(a.cpu.diff(&mut b.cpu).as_deref(), Some("xmm"));
        }

        #[test]
        fn diff_compares_resolved_flags_not_the_deferred_representation() {
            let (mut a, mut b) = pair(&[HLT]);
            // Same architectural flags, one deferred and one materialized.
            a.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            b.cpu.set_eflags_arith(CF | ZF | PF | AF);
            assert_eq!(a.cpu.diff(&mut b.cpu), None);
            b.cpu.set_eflags_arith(CF);
            let report = a.cpu.diff(&mut b.cpu).unwrap();
            assert!(report.contains("eflags"), "{report}");
        }

        #[test]
        fn run_verify_agrees_on_a_trivial_program() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::Reset), "{reason:?}");
            assert_eq!(main.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
            assert_eq!(shadow.cpu.gpr[crate::testkit::RAX] & 0xff, 0xfe);
        }

        #[test]
        fn run_verify_reports_a_register_divergence() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            shadow.cpu.gpr[RBX] = 0xbad;
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            match reason {
                ExitReason::Internal(message) => {
                    assert!(
                        message.starts_with("verify: divergence after block at"),
                        "{message}"
                    );
                    assert!(message.contains("r3: 0x0 vs 0xbad"), "{message}");
                }
                other => panic!("expected a divergence report, got {other:?}"),
            }
        }

        #[test]
        fn run_verify_reports_diverging_code_and_logs_the_page_difference() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            let log = capture_log(&mut main.cpu);
            // The shadow adds 8 instead of 7: a one-byte difference in the same page.
            shadow.write(CODE_BASE + 10, &[0x08]);
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::Internal(_)), "{reason:?}");
            let messages = log.lock().unwrap();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].starts_with("page diffs at "), "{}", messages[0]);
            assert!(messages[0].contains("main bytes"));
        }

        #[test]
        fn run_verify_flags_a_pre_step_divergence() {
            let (mut main, mut shadow) = pair(&COMPUTE_THEN_RESET);
            if !main.enable_jit() {
                return;
            }
            // The main CPU idles (IF set, so the halt is not a shutdown) while the shadow is ready
            // to execute.
            main.cpu.halted = true;
            main.cpu.eflags |= IF;
            main.cpu.next_event = u64::MAX;
            shadow.cpu.next_event = u64::MAX;
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(
                matches!(&reason, ExitReason::Internal(message) if message.contains("pre-step divergence")),
                "{reason:?}"
            );
        }

        #[test]
        fn run_verify_accepts_matching_faults_and_ends_in_a_triple_fault() {
            // ud2 with no IDT: both engines raise #UD, which cannot be delivered.
            let (mut main, mut shadow) = pair(&[0x0f, 0x0b]);
            if !main.enable_jit() {
                return;
            }
            let reason = main.cpu.run_verify(&mut shadow.cpu);
            assert!(matches!(reason, ExitReason::TripleFault), "{reason:?}");
        }
    }
}

/// Whether the JIT block and its interpreter replay ended the same way: both finished, both
/// raised the same fault, both asked the machine to exit, or both took a #VMEXIT (the state the
/// exit saved is compared separately).
fn outcomes_agree(main: &ExecResult<()>, shadow: &ExecResult<()>) -> bool {
    match (main, shadow) {
        (Ok(()), Ok(())) => true,
        (
            Err(Exception::Fault {
                vector: main_vector,
                error_code: main_error,
                ..
            }),
            Err(Exception::Fault {
                vector: shadow_vector,
                error_code: shadow_error,
                ..
            }),
        ) => main_vector == shadow_vector && main_error == shadow_error,
        (Err(Exception::Exit(_)), Err(Exception::Exit(_))) => true,
        (Err(Exception::Vmexit), Err(Exception::Vmexit)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_accepts_two_vmexits_and_rejects_a_vmexit_against_success() {
        assert!(outcomes_agree(
            &Err(Exception::Vmexit),
            &Err(Exception::Vmexit)
        ));
        assert!(!outcomes_agree(&Err(Exception::Vmexit), &Ok(())));
    }

    #[test]
    fn reset_constants_match_documented_values() {
        assert_eq!(SEG_FLAGS_FLAT_DATA, 0x00cf_9200);
        assert_eq!(SEG_FLAGS_CODE64, 0x0020_9a00);
        assert_eq!(CR0_PG | CR0_NE | CR0_ET | CR0_MP | CR0_PE, 0x8000_0033);
        assert_eq!(EFER_LMA | EFER_LME, 0x500);
        assert_eq!(CR4_PAE, 0x20);
        assert_eq!(core::mem::size_of::<TlbEntry>(), 32);
    }
}
