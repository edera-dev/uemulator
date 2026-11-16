//! Performance monitoring as Zen 5 has it: six core counters (PERFCTR_CORE, MSRs 0xC001_0200..=
//! 0x20B, with the legacy K7 MSRs 0xC001_0000..=0x07 aliasing counters 0-3), the PerfMonV2 global
//! control and status MSRs, rdpmc, the overflow interrupt through the local APIC's LVT performance
//! entry, the 16-entry LBR v2 branch stack, and the data-fabric (northbridge) and L3 counters that
//! `amd_uncore` probes. APM vol. 2 chapter 13.2 and the family 1Ah PPR describe the hardware;
//! Linux's `arch/x86/events/amd/{core,lbr,uncore}.c` is what has to work against it.
//!
//! What counts: cycles not halted (event 0x76), retired instructions and ops (0xC0, 0xC1), retired
//! branches and retired taken branches (0xC2, 0xC4). Every other event, the mispredicted-branch
//! count (0xC3) included, is programmable and reads zero: there is no microarchitecture to
//! observe. The uncore counters likewise store their programming and never count.
//!
//! Counting is lazy so the JIT pays nothing for it. Cycles and instructions derive from `icount`
//! (one instruction per cycle at the 1 GHz the TSC runs at, `CYCLES_PER_ICOUNT`), so a counter is
//! only brought up to date ("synced") when something can observe it or change what it counts: an
//! access to its MSRs or rdpmc, a change of CPL or host/guest mode while a counter filters on it
//! (`Cpu::set_cpl`, VMRUN), and the overflow deadline, which `before_step` reaches through
//! `next_event`. The overflow interrupt therefore lands at the first block boundary at or after
//! the overflow; real hardware has skid too. Branch events and the branch record need every
//! instruction looked at, so while one is enabled the CPU runs the interpreter (`Cpu::uses_jit`).
//!
//! Time-shared vCPUs hand one `icount` around (`Cpu::swap_shared`), so every count here is of
//! this vCPU's own instructions: `icount` less `Pmu::foreign`, the instructions the others retired
//! while this one was parked (`Cpu::pmu_park`, `Cpu::pmu_unpark`). Threaded vCPUs each keep their
//! own `icount` and never park.
//!
//! IRPERF (MSR 0xC000_00E9, CPUID 0x8000_0008 EBX bit 1) is a seventh, fixed counter of retired
//! instructions that runs while HWCR.IRPerfEn is set; it syncs with the others.

use super::decode::Insn;
use super::exception::Exception;
use super::*;
use crate::snapshot::Snapshot;

#[path = "ibs.rs"]
pub(crate) mod ibs;

/// Core counters with PERFCTR_CORE (leaf 0x8000_0022 EBX reports the same number).
pub(crate) const CORE_COUNTERS: usize = 6;
/// The legacy counters at 0xC001_0000, which alias core counters 0-3.
const LEGACY_COUNTERS: u32 = 4;
const MSR_K7_EVNTSEL0: u32 = 0xc001_0000;
const MSR_K7_PERFCTR0: u32 = 0xc001_0004;
const MSR_K7_PERFCTR_LAST: u32 = MSR_K7_PERFCTR0 + LEGACY_COUNTERS - 1;
/// PERF_CTL0, PERF_CTR0, PERF_CTL1, ...: select and counter interleaved.
const MSR_F15H_PERF_CTL0: u32 = 0xc001_0200;
const MSR_F15H_PERF_LAST: u32 = MSR_F15H_PERF_CTL0 + 2 * CORE_COUNTERS as u32 - 1;
/// PerfMonV2 (APM vol. 2 sec. 13.2.2.1). Status is read-only; Clr and Set write-1 to clear or set
/// its bits.
const MSR_PERF_CNTR_GLOBAL_STATUS: u32 = 0xc000_0300;
const MSR_PERF_CNTR_GLOBAL_CTL: u32 = 0xc000_0301;
const MSR_PERF_CNTR_GLOBAL_STATUS_CLR: u32 = 0xc000_0302;
const MSR_PERF_CNTR_GLOBAL_STATUS_SET: u32 = 0xc000_0303;
/// LBR v2: the branch filter, the enable (DBG_EXTN_CFG bit 6), and the stack as from/to pairs.
const MSR_AMD64_LBR_SELECT: u32 = 0xc000_010e;
const MSR_AMD_DBG_EXTN_CFG: u32 = 0xc000_010f;
const MSR_AMD_SAMP_BR_FROM: u32 = 0xc001_0300;
const MSR_AMD_SAMP_BR_LAST: u32 = MSR_AMD_SAMP_BR_FROM + 2 * LBR_ENTRIES as u32 - 1;
/// Data-fabric counters (PERFCTR_NB) and L3 counters (PERFCTR_LLC), select and counter
/// interleaved like the core ones.
const MSR_F15H_NB_PERF_CTL0: u32 = 0xc001_0240;
const MSR_F16H_L2I_PERF_CTL0: u32 = 0xc001_0230;
pub(crate) const NORTHBRIDGE_COUNTERS: usize = 4;
const L3_COUNTERS: usize = 6;
const MSR_NB_PERF_LAST: u32 = MSR_F15H_NB_PERF_CTL0 + 2 * NORTHBRIDGE_COUNTERS as u32 - 1;
const MSR_L3_PERF_LAST: u32 = MSR_F16H_L2I_PERF_CTL0 + 2 * L3_COUNTERS as u32 - 1;
/// rdpmc numbers the data-fabric counters from 6 and the L3 ones from 10 (Linux's
/// `RDPMC_BASE_NB`, `RDPMC_BASE_LLC`).
const RDPMC_BASE_NORTHBRIDGE: u32 = CORE_COUNTERS as u32;
const RDPMC_BASE_L3: u32 = RDPMC_BASE_NORTHBRIDGE + NORTHBRIDGE_COUNTERS as u32;
const RDPMC_END: u32 = RDPMC_BASE_L3 + L3_COUNTERS as u32;
/// Instructions retired, read-only, 48 bits like the other counters (Linux's `msr` PMU reads it
/// as the "irperf" event).
const MSR_IRPERF: u32 = 0xc000_00e9;
/// HWCR bit 30, IRPerfEn: IRPERF counts while set. Linux sets it at boot when CPUID has IRPERF.
const HWCR_IRPERF_ENABLE: u64 = 1 << 30;

/// Entries of the LBR v2 stack (leaf 0x8000_0022 EBX bits 9:4).
pub(crate) const LBR_ENTRIES: usize = 16;

/// Counters are 48 bits wide; the bits above read as zero and writes drop them.
const COUNTER_BITS: u32 = 48;
const COUNTER_MASK: u64 = (1 << COUNTER_BITS) - 1;

// PerfEvtSel fields (APM vol. 2 fig. 13-7).
const EVENT_SELECT_LOW_MASK: u64 = 0xff;
const EVENT_SELECT_HIGH_SHIFT: u32 = 32;
const EVENT_SELECT_HIGH_MASK: u64 = 0xf << EVENT_SELECT_HIGH_SHIFT;
const SELECT_USR: u64 = 1 << 16;
const SELECT_OS: u64 = 1 << 17;
const SELECT_EDGE: u64 = 1 << 18;
const SELECT_INT: u64 = 1 << 20;
const SELECT_EN: u64 = 1 << 22;
const SELECT_INV: u64 = 1 << 23;
const SELECT_CNT_MASK_SHIFT: u32 = 24;
const SELECT_GUEST_ONLY: u64 = 1 << 40;
const SELECT_HOST_ONLY: u64 = 1 << 41;
/// Bits 19 and 21, 39:36 and 63:42 are reserved; writing one is #GP (as KVM's `reserved_bits`).
const SELECT_RESERVED: u64 = (1 << 19) | (1 << 21) | (0xf << 36) | !((1 << 42) - 1);

/// Every core counter enabled: PerfCntrGlobalCtl's reset value, so software that predates
/// PerfMonV2 and only sets PerfEvtSel.En still counts (APM vol. 2 sec. 13.2.2.1; KVM resets it
/// the same way).
const GLOBAL_CTL_ALL: u64 = (1 << CORE_COUNTERS) - 1;

/// DBG_EXTN_CFG bit 6: the LBR v2 stack records.
const DBG_EXTN_CFG_LBRV2EN: u64 = 1 << 6;
/// LBR_SELECT: a set bit suppresses that kind of branch (the same layout as Intel's).
const LBR_SELECT_KERNEL: u64 = 1 << 0;
const LBR_SELECT_USER: u64 = 1 << 1;
const LBR_SELECT_CONDITIONAL: u64 = 1 << 2;
const LBR_SELECT_RELATIVE_CALL: u64 = 1 << 3;
const LBR_SELECT_INDIRECT_CALL: u64 = 1 << 4;
const LBR_SELECT_RETURN: u64 = 1 << 5;
const LBR_SELECT_INDIRECT_JUMP: u64 = 1 << 6;
const LBR_SELECT_RELATIVE_JUMP: u64 = 1 << 7;
const LBR_SELECT_FAR: u64 = 1 << 8;
const LBR_SELECT_WRITABLE: u64 = (1 << 9) - 1;
/// A recorded branch: the from address keeps bits 62:0 (bit 63 would say it was mispredicted,
/// which never happens here), the to address bits 60:0 with bit 63 saying the entry is valid and
/// bit 62 (speculative) clear. Linux sign-extends both from the linear address width.
const LBR_FROM_ADDRESS_MASK: u64 = !(1 << 63);
const LBR_TO_ADDRESS_MASK: u64 = (1 << 61) - 1;
const LBR_TO_VALID: u64 = 1 << 63;

/// Cycles per instruction counted in `icount`: the TSC (which is guest nanoseconds) runs at
/// 1 GHz and every instruction takes `NS_PER_INSN`. The CntMask rules in `scaled_count` rely on
/// no event counting more than once per cycle, which holds while this is at least 1.
const CYCLES_PER_ICOUNT: u64 = NS_PER_INSN;

// Event numbers (PPR for family 1Ah, "Core Performance Monitor Counters").
const EVENT_CYCLES_NOT_HALTED: u64 = 0x076;
const EVENT_RETIRED_INSTRUCTIONS: u64 = 0x0c0;
const EVENT_RETIRED_OPS: u64 = 0x0c1;
const EVENT_RETIRED_BRANCHES: u64 = 0x0c2;
const EVENT_RETIRED_TAKEN_BRANCHES: u64 = 0x0c4;

/// What an event counts, among the things the emulator can count.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Cycles,
    Instructions,
    Branches,
    TakenBranches,
    Nothing,
}

impl Source {
    fn of(select: u64) -> Source {
        let event = (select & EVENT_SELECT_LOW_MASK)
            | ((select & EVENT_SELECT_HIGH_MASK) >> (EVENT_SELECT_HIGH_SHIFT - 8));
        match event {
            EVENT_CYCLES_NOT_HALTED => Source::Cycles,
            // One op per instruction: there is no decoder splitting them.
            EVENT_RETIRED_INSTRUCTIONS | EVENT_RETIRED_OPS => Source::Instructions,
            EVENT_RETIRED_BRANCHES => Source::Branches,
            EVENT_RETIRED_TAKEN_BRANCHES => Source::TakenBranches,
            _ => Source::Nothing,
        }
    }

    fn needs_every_instruction(self) -> bool {
        matches!(self, Source::Branches | Source::TakenBranches)
    }
}

/// Running totals of what the counters count. They only grow (wrapping), and a counter adds the
/// difference between two readings, so nothing has to be reset when counters change.
#[derive(Clone, Copy, Default, Snapshot)]
struct Totals {
    cycles: u64,
    instructions: u64,
    branches: u64,
    taken_branches: u64,
}

impl Totals {
    fn since(self, earlier: Totals) -> Totals {
        Totals {
            cycles: self.cycles.wrapping_sub(earlier.cycles),
            instructions: self.instructions.wrapping_sub(earlier.instructions),
            branches: self.branches.wrapping_sub(earlier.branches),
            taken_branches: self.taken_branches.wrapping_sub(earlier.taken_branches),
        }
    }
}

/// How the LBR filter classifies a branch (the `LBR_SELECT_*` bit that suppresses it).
#[derive(Clone, Copy)]
pub(crate) enum BranchKind {
    Conditional,
    RelativeCall,
    IndirectCall,
    Return,
    IndirectJump,
    RelativeJump,
    Far,
}

impl BranchKind {
    /// Branch instructions by opcode. Conditional ones (jcc, loop, jcxz) may fall through; the
    /// rest always transfer. `int n` and `int3` are absent: they raise their vector and are
    /// counted, like exceptions and interrupts, as they are delivered.
    fn of(insn: &Insn) -> Option<BranchKind> {
        Some(match insn.op {
            0x70..=0x7f | 0xe0..=0xe3 | 0x180..=0x18f => BranchKind::Conditional,
            0xe8 => BranchKind::RelativeCall,
            0xe9 | 0xeb => BranchKind::RelativeJump,
            0xc2 | 0xc3 => BranchKind::Return,
            // far jmp/call ptr16:xx, far ret, iret, syscall, sysret, sysenter, sysexit
            0x9a | 0xea | 0xca | 0xcb | 0xcf | 0x105 | 0x107 | 0x134 | 0x135 => BranchKind::Far,
            0xff => match insn.reg & 7 {
                2 => BranchKind::IndirectCall,
                4 => BranchKind::IndirectJump,
                3 | 5 => BranchKind::Far,
                _ => return None,
            },
            _ => return None,
        })
    }

    fn lbr_select_bit(self) -> u64 {
        match self {
            BranchKind::Conditional => LBR_SELECT_CONDITIONAL,
            BranchKind::RelativeCall => LBR_SELECT_RELATIVE_CALL,
            BranchKind::IndirectCall => LBR_SELECT_INDIRECT_CALL,
            BranchKind::Return => LBR_SELECT_RETURN,
            BranchKind::IndirectJump => LBR_SELECT_INDIRECT_JUMP,
            BranchKind::RelativeJump => LBR_SELECT_RELATIVE_JUMP,
            BranchKind::Far => LBR_SELECT_FAR,
        }
    }
}

/// Per-vCPU counter state. Not machine-wide: each vCPU has its own PMU.
#[derive(Snapshot)]
pub(crate) struct Pmu {
    event_select: [u64; CORE_COUNTERS],
    /// Each counter's value as of the last sync.
    counter: [u64; CORE_COUNTERS],
    global_ctl: u64,
    global_status: u64,
    /// Bit i: counter i counts in the current mode. Only valid between syncs.
    counting: u8,
    /// An enabled counter filters on CPL or host/guest, so a mode change must sync.
    mode_sensitive: bool,
    /// A branch event or the branch record is enabled: run the interpreter.
    pub(crate) precise: bool,
    /// `icount` charges that are not retired instructions: a faulting instruction, and every
    /// iteration of a rep string instruction except the one that finishes it.
    pub(crate) unretired: u64,
    /// Instructions other time-shared vCPUs added to `icount` while this one was parked, and the
    /// `icount` it parked at.
    foreign: u64,
    parked_icount: u64,
    irperf: u64,
    irperf_enabled: bool,
    /// Running totals only `exec_observed` advances (precise mode).
    branches: u64,
    taken_branches: u64,
    /// The totals at the last sync, and the `icount` they were taken at.
    synced: Totals,
    synced_icount: u64,
    /// `icount` at or after which a counting counter may overflow; `u64::MAX` when none counts.
    pub(crate) deadline: u64,
    lbr_select: u64,
    dbg_extn_cfg: u64,
    /// The branch stack, newest first, as (from, to) MSR values.
    lbr: [(u64, u64); LBR_ENTRIES],
    northbridge: [(u64, u64); NORTHBRIDGE_COUNTERS],
    l3: [(u64, u64); L3_COUNTERS],
    /// Instruction-based sampling (`ibs.rs`).
    pub(crate) ibs: ibs::Ibs,
}

impl Pmu {
    pub(crate) fn new() -> Self {
        Pmu {
            event_select: [0; CORE_COUNTERS],
            counter: [0; CORE_COUNTERS],
            global_ctl: GLOBAL_CTL_ALL,
            global_status: 0,
            counting: 0,
            mode_sensitive: false,
            precise: false,
            unretired: 0,
            foreign: 0,
            parked_icount: 0,
            irperf: 0,
            irperf_enabled: false,
            branches: 0,
            taken_branches: 0,
            synced: Totals::default(),
            synced_icount: 0,
            deadline: u64::MAX,
            lbr_select: 0,
            dbg_extn_cfg: 0,
            lbr: [(0, 0); LBR_ENTRIES],
            northbridge: [(0, 0); NORTHBRIDGE_COUNTERS],
            l3: [(0, 0); L3_COUNTERS],
            ibs: ibs::Ibs::new(),
        }
    }

    pub(crate) fn owns(msr: u32) -> bool {
        matches!(
            msr,
            MSR_K7_EVNTSEL0..=MSR_K7_PERFCTR_LAST
                | MSR_F15H_PERF_CTL0..=MSR_F15H_PERF_LAST
                | MSR_PERF_CNTR_GLOBAL_STATUS..=MSR_PERF_CNTR_GLOBAL_STATUS_SET
                | MSR_AMD64_LBR_SELECT
                | MSR_AMD_DBG_EXTN_CFG
                | MSR_AMD_SAMP_BR_FROM..=MSR_AMD_SAMP_BR_LAST
                | MSR_F15H_NB_PERF_CTL0..=MSR_NB_PERF_LAST
                | MSR_F16H_L2I_PERF_CTL0..=MSR_L3_PERF_LAST
                | MSR_IRPERF
        ) || ibs::Ibs::owns(msr)
    }

    fn enabled(&self, index: usize) -> bool {
        self.event_select[index] & SELECT_EN != 0 && self.global_ctl & (1 << index) != 0
    }

    fn lbr_enabled(&self) -> bool {
        self.dbg_extn_cfg & DBG_EXTN_CFG_LBRV2EN != 0
    }

    /// Adds `events` to counter `index`; true when it wrapped past 2^48 (the overflow).
    fn add(&mut self, index: usize, events: u64) -> bool {
        let sum = self.counter[index].wrapping_add(events);
        self.counter[index] = sum & COUNTER_MASK;
        if sum > COUNTER_MASK {
            self.global_status |= 1 << index;
            return true;
        }
        false
    }

    /// Records a taken branch at the top of the stack, unless LBR_SELECT filters it out.
    fn record_branch(&mut self, kind: BranchKind, from: u64, to: u64, source_cpl: u8) {
        let mode_bit = if source_cpl == 0 {
            LBR_SELECT_KERNEL
        } else {
            LBR_SELECT_USER
        };
        if self.lbr_select & (mode_bit | kind.lbr_select_bit()) != 0 {
            return;
        }
        self.lbr.copy_within(0..LBR_ENTRIES - 1, 1);
        self.lbr[0] = (
            from & LBR_FROM_ADDRESS_MASK,
            (to & LBR_TO_ADDRESS_MASK) | LBR_TO_VALID,
        );
    }
}

/// The count for an event whose plain count over the interval is `events`, within `cycles`
/// cycles, after CntMask and Inv: with a nonzero CntMask the counter counts the cycles in which
/// the event occurred at least CntMask times (fewer, with Inv). No event here happens more than
/// once a cycle, so that is all of its occurrences for CntMask 1, none for a larger mask, and
/// with Inv the cycles it did not occur in (mask 1) or every cycle. Edge counts the starts of
/// such stretches, which the totals cannot tell apart, so it counts nothing.
fn scaled_count(select: u64, events: u64, cycles: u64) -> u64 {
    let count_mask = (select >> SELECT_CNT_MASK_SHIFT) & 0xff;
    if select & SELECT_EDGE != 0 {
        return 0;
    }
    // Inv means nothing without a CntMask (APM vol. 2 sec. 13.2.1).
    let inverted = select & SELECT_INV != 0 && count_mask != 0;
    match (count_mask, inverted) {
        (0 | 1, false) => events,
        (_, false) => 0,
        (1, true) => cycles.saturating_sub(events),
        (_, true) => cycles,
    }
}

impl Cpu {
    /// The totals with `icount` taken as `icount`: see `pmu_sync_before_this_instruction`.
    fn pmu_totals(&self, icount: u64) -> Totals {
        let own = icount.wrapping_sub(self.pmu.foreign);
        Totals {
            cycles: own.wrapping_mul(CYCLES_PER_ICOUNT),
            instructions: own.wrapping_sub(self.pmu.unretired),
            branches: self.pmu.branches,
            taken_branches: self.pmu.taken_branches,
        }
    }

    /// Whether counter `index`'s mode filter passes now: OS counts at CPL 0, USR above it
    /// (neither counts nothing), and HostOnly/GuestOnly pick a side of SVM. With both or neither
    /// of those set it counts on both sides; with one set and EFER.SVME clear it counts nothing,
    /// which is why Linux strips HostOnly until KVM enables SVM (`perf_ctr_virt_mask`).
    fn pmu_mode_matches(&self, select: u64) -> bool {
        let level = if self.cpl == 0 { SELECT_OS } else { SELECT_USR };
        if select & level == 0 {
            return false;
        }
        match (
            select & SELECT_HOST_ONLY != 0,
            select & SELECT_GUEST_ONLY != 0,
        ) {
            (true, true) | (false, false) => true,
            // In a guest `efer` is the guest's; the host's SVME is set or VMRUN would have #UD.
            _ if self.svm.guest.is_none() && self.efer & svm::EFER_SVME == 0 => false,
            (_, guest_only) => guest_only == self.svm.guest.is_some(),
        }
    }

    /// Brings every counting counter up to date, raises the overflow interrupt if one wrapped
    /// with PerfEvtSel.Int set, then works out what counts from here on and when the next
    /// overflow could be. An instruction that syncs while it runs is already in `icount` and
    /// counts in the window that ends here: a CPL change counts in the mode it started in, a
    /// wrmsr that enables a counter is not counted by it and one that disables it is.
    pub(crate) fn pmu_sync(&mut self) {
        self.pmu_sync_at(self.icount);
    }

    /// A sync for an instruction that reads a counter (rdpmc, rdmsr): what it reads must not
    /// include itself, as it has not retired yet, so it counts in the next window instead.
    fn pmu_sync_before_this_instruction(&mut self) {
        self.pmu_sync_at(self.icount.saturating_sub(1));
    }

    fn pmu_sync_at(&mut self, icount: u64) {
        // Never back past the last sync (a read right after a sync in the same instruction).
        let icount = icount.max(self.pmu.synced_icount);
        self.pmu.synced_icount = icount;
        let now = self.pmu_totals(icount);
        let elapsed = now.since(self.pmu.synced);
        self.pmu.synced = now;
        let mut interrupt = false;
        if self.pmu.irperf_enabled {
            self.pmu.irperf = self.pmu.irperf.wrapping_add(elapsed.instructions) & COUNTER_MASK;
        }
        for index in 0..CORE_COUNTERS {
            if self.pmu.counting & (1 << index) == 0 {
                continue;
            }
            let select = self.pmu.event_select[index];
            let events = match Source::of(select) {
                Source::Cycles => elapsed.cycles,
                Source::Instructions => elapsed.instructions,
                Source::Branches => elapsed.branches,
                Source::TakenBranches => elapsed.taken_branches,
                Source::Nothing => 0,
            };
            // A faulting far branch is counted before it runs and taken back after; a sync
            // in between (a CPL change before the fault) can leave a window one short.
            let events = if events > elapsed.cycles { 0 } else { events };
            let count = scaled_count(select, events, elapsed.cycles);
            if self.pmu.add(index, count) && select & SELECT_INT != 0 {
                interrupt = true;
            }
        }
        self.pmu_reschedule();
        if interrupt {
            // AMD leaves the LVT entry unmasked after a PMI (Linux's v2 handler relies on it).
            self.dev.lock().lapic_mut().performance_interrupt();
            // The interrupt is taken at the next block boundary that reaches `before_step`.
            self.next_event = self.icount;
        }
    }

    /// Recomputes which counters count in the current mode, whether the interpreter must run,
    /// and the overflow deadline. Call only right after a sync.
    fn pmu_reschedule(&mut self) {
        let mut counting = 0u8;
        let mut mode_sensitive = false;
        let mut precise = self.pmu.lbr_enabled();
        let mut nearest_overflow = u64::MAX;
        for index in 0..CORE_COUNTERS {
            if !self.pmu.enabled(index) {
                continue;
            }
            let select = self.pmu.event_select[index];
            let source = Source::of(select);
            if source == Source::Nothing {
                continue;
            }
            precise |= source.needs_every_instruction();
            mode_sensitive |= select & (SELECT_OS | SELECT_USR) != (SELECT_OS | SELECT_USR)
                || select & (SELECT_HOST_ONLY | SELECT_GUEST_ONLY) != 0;
            if self.pmu_mode_matches(select) {
                counting |= 1 << index;
                let remaining = (COUNTER_MASK + 1) - self.pmu.counter[index];
                nearest_overflow = nearest_overflow.min(remaining.div_ceil(CYCLES_PER_ICOUNT));
            }
        }
        // IBS: interpret around its next tag, and let the JIT run up to the stretch before it.
        let (interpret, runway) = self.ibs_schedule(self.pmu.synced.instructions);
        precise |= interpret;
        nearest_overflow = nearest_overflow.min(runway);
        self.pmu.counting = counting;
        self.pmu.mode_sensitive = mode_sensitive;
        self.pmu.precise = precise;
        self.pmu.deadline = self.pmu.synced_icount.saturating_add(nearest_overflow);
        self.next_event = self.next_event.min(self.pmu.deadline);
    }

    /// CPL or host/guest mode changed: counters that filter on it start or stop. The instruction
    /// that changed it is already in `icount`, so it counts in the mode it started in.
    pub(crate) fn pmu_mode_changed(&mut self) {
        if self.pmu.mode_sensitive {
            self.pmu_sync();
        }
    }

    /// This time-shared vCPU hands `icount` on (`Smp::switch_to`); see the module comment.
    pub(crate) fn pmu_park(&mut self) {
        self.pmu.parked_icount = self.icount;
    }

    /// This vCPU holds `icount` again: what the others retired meanwhile is not its own, so the
    /// counts, the sync point and the overflow deadline all move past it.
    pub(crate) fn pmu_unpark(&mut self) {
        let gap = self.icount.wrapping_sub(self.pmu.parked_icount);
        self.pmu.foreign = self.pmu.foreign.wrapping_add(gap);
        self.pmu.synced_icount = self.pmu.synced_icount.wrapping_add(gap);
        if self.pmu.deadline != u64::MAX {
            self.pmu.deadline = self.pmu.deadline.saturating_add(gap);
            // `next_event` is the machine's; it must still stop at this vCPU's overflow.
            self.next_event = self.next_event.min(self.pmu.deadline);
        }
    }

    /// HWCR is about to take `value` (it is stored in `msr_misc`): IRPERF starts or stops. The
    /// wrmsr itself counts in the window that ends here, like one that enables a core counter.
    pub(crate) fn pmu_hwcr_written(&mut self, value: u64) {
        let enabled = value & HWCR_IRPERF_ENABLE != 0;
        if enabled != self.pmu.irperf_enabled {
            self.pmu_sync();
            self.pmu.irperf_enabled = enabled;
        }
    }

    /// The overflow deadline has passed: sync, which raises any overflow interrupt.
    pub(crate) fn pmu_deadline_reached(&mut self) {
        if self.icount >= self.pmu.deadline {
            self.pmu_sync();
        }
    }

    pub(crate) fn pmu_rdmsr(&mut self, msr: u32) -> ExecResult<u64> {
        Ok(match msr {
            _ if ibs::Ibs::owns(msr) => return self.ibs_rdmsr(msr),
            MSR_K7_EVNTSEL0..MSR_K7_PERFCTR0 => {
                self.pmu.event_select[(msr - MSR_K7_EVNTSEL0) as usize]
            }
            MSR_K7_PERFCTR0..=MSR_K7_PERFCTR_LAST => {
                self.pmu_sync_before_this_instruction();
                self.pmu.counter[(msr - MSR_K7_PERFCTR0) as usize]
            }
            MSR_F15H_PERF_CTL0..=MSR_F15H_PERF_LAST => {
                let index = ((msr - MSR_F15H_PERF_CTL0) / 2) as usize;
                if msr & 1 == 0 {
                    self.pmu.event_select[index]
                } else {
                    self.pmu_sync_before_this_instruction();
                    self.pmu.counter[index]
                }
            }
            MSR_PERF_CNTR_GLOBAL_STATUS => {
                self.pmu_sync_before_this_instruction();
                self.pmu.global_status
            }
            MSR_PERF_CNTR_GLOBAL_CTL => self.pmu.global_ctl,
            MSR_IRPERF => {
                self.pmu_sync_before_this_instruction();
                self.pmu.irperf
            }
            // write-only command registers
            MSR_PERF_CNTR_GLOBAL_STATUS_CLR | MSR_PERF_CNTR_GLOBAL_STATUS_SET => 0,
            MSR_AMD64_LBR_SELECT => self.pmu.lbr_select,
            MSR_AMD_DBG_EXTN_CFG => self.pmu.dbg_extn_cfg,
            MSR_AMD_SAMP_BR_FROM..=MSR_AMD_SAMP_BR_LAST => {
                let (from, to) = self.pmu.lbr[((msr - MSR_AMD_SAMP_BR_FROM) / 2) as usize];
                if msr & 1 == 0 { from } else { to }
            }
            MSR_F15H_NB_PERF_CTL0..=MSR_NB_PERF_LAST => {
                let (select, counter) =
                    self.pmu.northbridge[((msr - MSR_F15H_NB_PERF_CTL0) / 2) as usize];
                if msr & 1 == 0 { select } else { counter }
            }
            MSR_F16H_L2I_PERF_CTL0..=MSR_L3_PERF_LAST => {
                let (select, counter) = self.pmu.l3[((msr - MSR_F16H_L2I_PERF_CTL0) / 2) as usize];
                if msr & 1 == 0 { select } else { counter }
            }
            _ => return Err(Exception::general_protection(0)),
        })
    }

    pub(crate) fn pmu_wrmsr(&mut self, msr: u32, value: u64) -> ExecResult<()> {
        let gp = || Err(Exception::general_protection(0));
        match msr {
            _ if ibs::Ibs::owns(msr) => return self.ibs_wrmsr(msr, value),
            MSR_K7_EVNTSEL0..MSR_K7_PERFCTR0 => {
                return self.pmu_write_select((msr - MSR_K7_EVNTSEL0) as usize, value);
            }
            MSR_K7_PERFCTR0..=MSR_K7_PERFCTR_LAST => {
                self.pmu_write_counter((msr - MSR_K7_PERFCTR0) as usize, value)
            }
            MSR_F15H_PERF_CTL0..=MSR_F15H_PERF_LAST => {
                let index = ((msr - MSR_F15H_PERF_CTL0) / 2) as usize;
                if msr & 1 == 0 {
                    return self.pmu_write_select(index, value);
                }
                self.pmu_write_counter(index, value);
            }
            MSR_PERF_CNTR_GLOBAL_STATUS | MSR_IRPERF => return gp(),
            MSR_PERF_CNTR_GLOBAL_CTL => {
                if value & !GLOBAL_CTL_ALL != 0 {
                    return gp();
                }
                self.pmu_sync();
                self.pmu.global_ctl = value;
                self.pmu_reschedule();
            }
            // Bits without a counter are ignored rather than #GP: Linux's v2 interrupt handler
            // acknowledges with the complement of what it left unhandled, all 64 bits.
            MSR_PERF_CNTR_GLOBAL_STATUS_CLR => {
                self.pmu_sync();
                self.pmu.global_status &= !(value & GLOBAL_CTL_ALL);
            }
            MSR_PERF_CNTR_GLOBAL_STATUS_SET => {
                self.pmu_sync();
                self.pmu.global_status |= value & GLOBAL_CTL_ALL;
            }
            MSR_AMD64_LBR_SELECT => {
                if value & !LBR_SELECT_WRITABLE != 0 {
                    return gp();
                }
                self.pmu.lbr_select = value;
            }
            // Only LBRV2EN is modelled; the other bits are kept but do nothing.
            MSR_AMD_DBG_EXTN_CFG => {
                self.pmu_sync();
                self.pmu.dbg_extn_cfg = value;
                self.pmu_reschedule();
            }
            MSR_AMD_SAMP_BR_FROM..=MSR_AMD_SAMP_BR_LAST => {
                let entry = &mut self.pmu.lbr[((msr - MSR_AMD_SAMP_BR_FROM) / 2) as usize];
                if msr & 1 == 0 {
                    entry.0 = value;
                } else {
                    entry.1 = value;
                }
            }
            MSR_F15H_NB_PERF_CTL0..=MSR_NB_PERF_LAST => {
                let entry = &mut self.pmu.northbridge[((msr - MSR_F15H_NB_PERF_CTL0) / 2) as usize];
                if msr & 1 == 0 {
                    entry.0 = value;
                } else {
                    entry.1 = value & COUNTER_MASK;
                }
            }
            MSR_F16H_L2I_PERF_CTL0..=MSR_L3_PERF_LAST => {
                let entry = &mut self.pmu.l3[((msr - MSR_F16H_L2I_PERF_CTL0) / 2) as usize];
                if msr & 1 == 0 {
                    entry.0 = value;
                } else {
                    entry.1 = value & COUNTER_MASK;
                }
            }
            _ => return gp(),
        }
        Ok(())
    }

    fn pmu_write_select(&mut self, index: usize, value: u64) -> ExecResult<()> {
        if value & SELECT_RESERVED != 0 {
            return Err(Exception::general_protection(0));
        }
        self.pmu_sync();
        self.pmu.event_select[index] = value;
        self.pmu_reschedule();
        Ok(())
    }

    fn pmu_write_counter(&mut self, index: usize, value: u64) {
        self.pmu_sync();
        self.pmu.counter[index] = value & COUNTER_MASK;
        self.pmu_reschedule();
    }

    /// rdpmc's ECX: 0-5 the core counters, 6-9 the data-fabric ones, 10-15 the L3 ones.
    pub(crate) fn pmu_rdpmc(&mut self, index: u32) -> Option<u64> {
        match index {
            0..RDPMC_BASE_NORTHBRIDGE => {
                self.pmu_sync_before_this_instruction();
                Some(self.pmu.counter[index as usize])
            }
            RDPMC_BASE_NORTHBRIDGE..RDPMC_BASE_L3 => {
                Some(self.pmu.northbridge[(index - RDPMC_BASE_NORTHBRIDGE) as usize].1)
            }
            RDPMC_BASE_L3..RDPMC_END => Some(self.pmu.l3[(index - RDPMC_BASE_L3) as usize].1),
            _ => None,
        }
    }

    /// Runs `insn` in precise mode: `exec_counting_branches`, and IBS's sample if this is the
    /// instruction it tagged. A tagged op's memory accesses are watched (`watch_access`) for the
    /// sample's data address.
    pub(crate) fn exec_observed(&mut self, insn: &Insn) -> ExecResult<()> {
        let tagged = self.ibs_before();
        if tagged == (false, false) {
            return self.exec_counting_branches(insn);
        }
        let from = self.rip;
        let watching = self.debug.watch_accesses;
        self.debug.watch_accesses = true;
        let result = self.exec_counting_branches(insn);
        self.debug.watch_accesses = watching;
        self.ibs_after(tagged, insn, from, &result);
        result
    }

    /// Runs `insn`, counting it if it is a branch and recording it in the branch stack if
    /// taken. A branch that always transfers is counted before it runs and taken back if it
    /// faults, so one that changes CPL (syscall, iret) is counted in the mode it started in,
    /// like the instruction itself.
    fn exec_counting_branches(&mut self, insn: &Insn) -> ExecResult<()> {
        let from = self.rip;
        let source_cpl = self.cpl;
        let kind = BranchKind::of(insn);
        let always_taken = matches!(kind, Some(kind) if !matches!(kind, BranchKind::Conditional));
        if always_taken {
            self.pmu.branches = self.pmu.branches.wrapping_add(1);
            self.pmu.taken_branches = self.pmu.taken_branches.wrapping_add(1);
        }
        let result = self.exec(insn);
        let Some(kind) = kind else {
            return result;
        };
        match &result {
            Ok(()) => {
                let taken = always_taken || self.rip != from.wrapping_add(insn.len as u64);
                if !always_taken {
                    self.pmu.branches = self.pmu.branches.wrapping_add(1);
                    if taken {
                        self.pmu.taken_branches = self.pmu.taken_branches.wrapping_add(1);
                    }
                }
                if taken && self.pmu.lbr_enabled() {
                    let to = self.rip;
                    self.pmu.record_branch(kind, from, to, source_cpl);
                }
            }
            Err(Exception::Fault {
                software: false, ..
            }) if always_taken => {
                self.pmu.branches = self.pmu.branches.wrapping_sub(1);
                self.pmu.taken_branches = self.pmu.taken_branches.wrapping_sub(1);
            }
            Err(_) => {}
        }
        result
    }

    /// An exception or interrupt is about to be delivered: count it as a retired taken branch
    /// (the PPR's 0xC2 includes them) in the mode it interrupts. Returns what
    /// `pmu_event_delivered` needs.
    pub(crate) fn pmu_event_delivering(&mut self) -> (u64, u8) {
        self.pmu.branches = self.pmu.branches.wrapping_add(1);
        self.pmu.taken_branches = self.pmu.taken_branches.wrapping_add(1);
        (self.rip, self.cpl)
    }

    /// The event reached its handler: record it as a far branch.
    pub(crate) fn pmu_event_delivered(&mut self, (from, source_cpl): (u64, u8)) {
        if self.pmu.lbr_enabled() {
            let to = self.rip;
            self.pmu
                .record_branch(BranchKind::Far, from, to, source_cpl);
        }
    }
}

#[cfg(test)]
#[path = "pmu_tests.rs"]
mod tests;
