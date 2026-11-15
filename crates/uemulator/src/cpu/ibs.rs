//! Instruction-based sampling (APM vol. 2 sec. 13.3, the family 1Ah PPR's IBS MSRs): fetch
//! sampling tags the fetch of every n-th instruction, op sampling every n-th retired op (or
//! cycle), and each reports what happened to the one it tagged, then raises the interrupt in
//! the extended APIC LVT entry IBS_CTL names. Linux's `arch/x86/events/amd/ibs.c` is the
//! driver that has to work against it: it programs that entry as an NMI and takes the sample
//! from the MSRs in its NMI handler.
//!
//! What is reported is what the emulator knows: the tagged instruction's address, whether it
//! was a branch, a return, taken (never mispredicted, never fused), the target it went to, and
//! the linear and physical address, width and direction of its first memory access, with the
//! valid bits Linux checks. Latencies, cache and TLB miss bits and the data source read as zero:
//! an L1 hit through the L1 TLB, which is what Linux's `perf mem` makes of them. L3MissOnly
//! (IBS_CAPS_ZEN4) therefore never samples. One instruction is one op and one fetch, and a cycle
//! is an instruction (`CYCLES_PER_ICOUNT`).
//!
//! Counting rides on the PMU (`pmu.rs`): counts are of this vCPU's retired instructions
//! (`Totals::instructions`), the tagged instruction's position is fixed when sampling is enabled,
//! and `Cpu::pmu_reschedule` turns it into a deadline. Short of the tag the JIT runs: the deadline
//! stops it `INTERPRET_WINDOW` instructions early (more than a JIT block holds, so no block runs
//! past the tag), then `Pmu::precise` makes the run loop interpret, and `exec_observed` samples
//! the instruction itself. Nothing changes in generated code.

use super::{BranchKind, CYCLES_PER_ICOUNT};
use crate::cpu::decode::Insn;
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;
use crate::cpu::{Cpu, ExecResult};

pub(crate) const MSR_IBS_FETCH_CONTROL: u32 = 0xc001_1030;
const MSR_IBS_FETCH_LINEAR: u32 = 0xc001_1031;
const MSR_IBS_FETCH_PHYSICAL: u32 = 0xc001_1032;
const MSR_IBS_OP_CONTROL: u32 = 0xc001_1033;
const MSR_IBS_OP_RIP: u32 = 0xc001_1034;
const MSR_IBS_OP_DATA: u32 = 0xc001_1035;
const MSR_IBS_OP_DATA2: u32 = 0xc001_1036;
const MSR_IBS_OP_DATA3: u32 = 0xc001_1037;
const MSR_IBS_DC_LINEAR: u32 = 0xc001_1038;
const MSR_IBS_DC_PHYSICAL: u32 = 0xc001_1039;
/// Read-only: which extended LVT entry IBS interrupts through. Firmware sets it on real parts.
const MSR_IBS_CONTROL: u32 = 0xc001_103a;
pub(crate) const MSR_IBS_BRANCH_TARGET: u32 = 0xc001_103b;

/// The extended LVT entry IBS_CTL names. Linux programs entries 1 and 2 for the MCA threshold
/// and deferred-error interrupts (`dev/apic.rs`), so IBS takes 0, as firmware usually gives it.
pub(crate) const IBS_LVT_OFFSET: usize = 0;
const IBS_CONTROL_LVT_OFFSET_VALID: u64 = 1 << 8;

/// CPUID leaf 0x8000_001B EAX, IBS capabilities (Linux's `IBS_CAPS_*`): the leaf is valid,
/// fetch and op sampling, IbsOpCurCnt readable and writable, op counting (IbsOpCntCtl), the
/// branch target MSR, the extended op MaxCnt, the RIP-invalid bit, the fused-branch bit (never
/// set) and the Zen 4 additions (L3MissOnly and the wider data source). Not: the fetch control
/// extension MSR, IbsOpData4, load-latency filtering, the DTLB page size bits.
pub(crate) const LEAF_IBS: u32 = 0x8000_001b;
const CAPS_AVAILABLE: u32 = 1 << 0;
const CAPS_FETCH_SAMPLING: u32 = 1 << 1;
const CAPS_OP_SAMPLING: u32 = 1 << 2;
const CAPS_READ_WRITE_OP_COUNT: u32 = 1 << 3;
const CAPS_OP_COUNT: u32 = 1 << 4;
const CAPS_BRANCH_TARGET: u32 = 1 << 5;
const CAPS_OP_COUNT_EXTENSION: u32 = 1 << 6;
const CAPS_RIP_INVALID_CHECK: u32 = 1 << 7;
const CAPS_OP_BRANCH_FUSE: u32 = 1 << 8;
const CAPS_ZEN4: u32 = 1 << 11;
pub(crate) const IBS_CAPABILITIES: u32 = CAPS_AVAILABLE
    | CAPS_FETCH_SAMPLING
    | CAPS_OP_SAMPLING
    | CAPS_READ_WRITE_OP_COUNT
    | CAPS_OP_COUNT
    | CAPS_BRANCH_TARGET
    | CAPS_OP_COUNT_EXTENSION
    | CAPS_RIP_INVALID_CHECK
    | CAPS_OP_BRANCH_FUSE
    | CAPS_ZEN4;

// IbsFetchCtl. The counts are in units of 16 fetches: MaxCnt holds bits 19:4 of the maximum,
// Cnt bits 19:4 of the count.
const FETCH_MAX_COUNT_MASK: u64 = 0xffff;
const FETCH_COUNT_SHIFT: u32 = 16;
const FETCH_COUNT_MASK: u64 = 0xffff << FETCH_COUNT_SHIFT;
const FETCH_ENABLE: u64 = 1 << 48;
const FETCH_VALID: u64 = 1 << 49;
const FETCH_COMPLETE: u64 = 1 << 50;
const FETCH_PHYSICAL_VALID: u64 = 1 << 52;
const FETCH_L3_MISS_ONLY: u64 = 1 << 59;
/// Latency (47:32), the cache and TLB miss bits, the page size (54:53): no model, so zero.
const FETCH_HARDWARE_FIELDS: u64 = 0xffff << 32
    | FETCH_VALID
    | FETCH_COMPLETE
    | 1 << 51
    | FETCH_PHYSICAL_VALID
    | 0b11 << 53
    | 1 << 55
    | 1 << 56
    | 1 << 58
    | 1 << 60
    | 1 << 61;
const COUNT_UNIT_SHIFT: u32 = 4;

// IbsOpCtl. MaxCnt is bits 19:4 of the maximum in 15:0 plus bits 26:20 in place; CurCnt is the
// full 27-bit count.
const OP_MAX_COUNT_MASK: u64 = 0xffff;
const OP_MAX_COUNT_EXTENSION_MASK: u64 = 0x7f << 20;
const OP_L3_MISS_ONLY: u64 = 1 << 16;
const OP_ENABLE: u64 = 1 << 17;
const OP_VALID: u64 = 1 << 18;
/// Count ops instead of cycles.
const OP_COUNT_CONTROL: u64 = 1 << 19;
const OP_CURRENT_COUNT_SHIFT: u32 = 32;
const OP_CURRENT_COUNT_MASK: u64 = ((1 << 27) - 1) << OP_CURRENT_COUNT_SHIFT;
const OP_WRITABLE: u64 = OP_MAX_COUNT_MASK
    | OP_L3_MISS_ONLY
    | OP_ENABLE
    | OP_VALID
    | OP_COUNT_CONTROL
    | OP_MAX_COUNT_EXTENSION_MASK
    | OP_CURRENT_COUNT_MASK;

// IbsOpData. Bits 31:0 are latencies (completion and tag to retire): zero.
const OP_DATA_RETURN: u64 = 1 << 34;
const OP_DATA_BRANCH_TAKEN: u64 = 1 << 35;
const OP_DATA_BRANCH_RETIRED: u64 = 1 << 37;

// IbsOpData3. The TLB and cache miss bits, latencies and outstanding-request count stay zero.
const OP_DATA3_LOAD: u64 = 1 << 0;
const OP_DATA3_STORE: u64 = 1 << 1;
const OP_DATA3_UNCACHEABLE: u64 = 1 << 14;
const OP_DATA3_LOCKED: u64 = 1 << 15;
const OP_DATA3_LINEAR_VALID: u64 = 1 << 17;
const OP_DATA3_PHYSICAL_VALID: u64 = 1 << 18;
/// OpMemWidth: 1, 2, 4, 8, 16, 32, 64 bytes as 1 to 7.
const OP_DATA3_WIDTH_SHIFT: u32 = 22;

/// How far ahead of the tagged instruction the JIT stops and the interpreter takes over: more
/// than the longest JIT block (`jit::blocks::MAX_INSNS`, 48), which can run past `next_event`
/// before its chain check sees it.
const INTERPRET_WINDOW: u64 = 64;

/// What the tagged instruction's accesses looked like (the first one's address and size).
#[derive(Clone, Copy, Default)]
pub(crate) struct Capture {
    first: Option<(u64, u64, bool)>,
    loaded: bool,
    stored: bool,
}

impl Capture {
    pub(crate) fn note(&mut self, linear: u64, size: u64, write: bool) {
        self.first.get_or_insert((linear, size, write));
        if write {
            self.stored = true;
        } else {
            self.loaded = true;
        }
    }
}

/// The IBS registers and counting state. Lives in `Pmu`, one per vCPU.
pub(crate) struct Ibs {
    fetch_control: u64,
    fetch_linear: u64,
    fetch_physical: u64,
    op_control: u64,
    op_rip: u64,
    op_data: u64,
    op_data2: u64,
    op_data3: u64,
    dc_linear: u64,
    dc_physical: u64,
    branch_target: u64,
    /// Retired instructions (`Totals::instructions`) when fetch / op counting was last
    /// programmed, which is when its count was what the control register then held.
    fetch_origin: u64,
    op_origin: u64,
    /// The retired-instruction number (as `fetch_origin` counts) of the instruction whose fetch
    /// or op is tagged; `u64::MAX` while that sampler is off or holds a sample.
    fetch_target: u64,
    op_target: u64,
    /// The accesses of the instruction being sampled, while it runs (`Cpu::watch_access`).
    pub(crate) capture: Option<Capture>,
}

/// OpMemWidth's encoding of an access of `size` bytes.
fn width_code(size: u64) -> u64 {
    match size {
        1 => 1,
        2 => 2,
        4 => 3,
        8 => 4,
        16 => 5,
        32 => 6,
        64 => 7,
        _ => 0,
    }
}

impl Ibs {
    pub(crate) fn new() -> Self {
        Ibs {
            fetch_control: 0,
            fetch_linear: 0,
            fetch_physical: 0,
            op_control: 0,
            op_rip: 0,
            op_data: 0,
            op_data2: 0,
            op_data3: 0,
            dc_linear: 0,
            dc_physical: 0,
            branch_target: 0,
            fetch_origin: 0,
            op_origin: 0,
            fetch_target: u64::MAX,
            op_target: u64::MAX,
            capture: None,
        }
    }

    pub(crate) fn owns(msr: u32) -> bool {
        (MSR_IBS_FETCH_CONTROL..=MSR_IBS_BRANCH_TARGET).contains(&msr)
    }

    fn op_max_count(&self) -> u64 {
        ((self.op_control & OP_MAX_COUNT_MASK) << COUNT_UNIT_SHIFT)
            | (self.op_control & OP_MAX_COUNT_EXTENSION_MASK)
    }

    /// Op events per retired instruction: an op, or `CYCLES_PER_ICOUNT` cycles.
    fn op_events_per_instruction(&self) -> u64 {
        if self.op_control & OP_COUNT_CONTROL != 0 {
            1
        } else {
            CYCLES_PER_ICOUNT
        }
    }

    fn op_counting(&self) -> bool {
        self.op_control & (OP_ENABLE | OP_VALID) == OP_ENABLE
    }

    fn fetch_counting(&self) -> bool {
        self.fetch_control & (FETCH_ENABLE | FETCH_VALID) == FETCH_ENABLE
    }

    /// Where the samplers' tags fall, from the control registers and origins; the earlier one.
    /// When the count reaches the maximum the next fetch or op is tagged; an L3MissOnly sampler
    /// never reports, since nothing misses.
    pub(crate) fn retarget(&mut self) -> u64 {
        self.fetch_target = if self.fetch_counting() && self.fetch_control & FETCH_L3_MISS_ONLY == 0
        {
            let maximum = (self.fetch_control & FETCH_MAX_COUNT_MASK) << COUNT_UNIT_SHIFT;
            let count =
                ((self.fetch_control & FETCH_COUNT_MASK) >> FETCH_COUNT_SHIFT) << COUNT_UNIT_SHIFT;
            self.fetch_origin + maximum.saturating_sub(count) + 1
        } else {
            u64::MAX
        };
        self.op_target = if self.op_counting() && self.op_control & OP_L3_MISS_ONLY == 0 {
            let count = (self.op_control & OP_CURRENT_COUNT_MASK) >> OP_CURRENT_COUNT_SHIFT;
            let remaining = self.op_max_count().saturating_sub(count);
            self.op_origin + remaining.div_ceil(self.op_events_per_instruction()) + 1
        } else {
            u64::MAX
        };
        self.fetch_target.min(self.op_target)
    }

    /// The control registers as software reads them, with the counts brought up to `retired`.
    fn fetch_control_at(&self, retired: u64) -> u64 {
        if !self.fetch_counting() {
            return self.fetch_control;
        }
        let maximum = (self.fetch_control & FETCH_MAX_COUNT_MASK) << COUNT_UNIT_SHIFT;
        let start =
            ((self.fetch_control & FETCH_COUNT_MASK) >> FETCH_COUNT_SHIFT) << COUNT_UNIT_SHIFT;
        let count = (start + retired.saturating_sub(self.fetch_origin)).min(maximum);
        (self.fetch_control & !FETCH_COUNT_MASK)
            | ((count >> COUNT_UNIT_SHIFT) << FETCH_COUNT_SHIFT) & FETCH_COUNT_MASK
    }

    fn op_control_at(&self, retired: u64) -> u64 {
        if !self.op_counting() {
            return self.op_control;
        }
        let start = (self.op_control & OP_CURRENT_COUNT_MASK) >> OP_CURRENT_COUNT_SHIFT;
        let events = retired.saturating_sub(self.op_origin) * self.op_events_per_instruction();
        let count = (start + events).min(self.op_max_count().max(start));
        (self.op_control & !OP_CURRENT_COUNT_MASK)
            | (count << OP_CURRENT_COUNT_SHIFT) & OP_CURRENT_COUNT_MASK
    }
}

impl Cpu {
    /// This vCPU's retired instructions so far, the one running included.
    fn ibs_retired(&self) -> u64 {
        self.icount
            .wrapping_sub(self.pmu.foreign)
            .wrapping_sub(self.pmu.unretired)
    }

    pub(crate) fn ibs_rdmsr(&mut self, msr: u32) -> ExecResult<u64> {
        // A reading instruction has not retired yet: it does not count itself.
        let retired = self.ibs_retired().wrapping_sub(1);
        let ibs = &self.pmu.ibs;
        Ok(match msr {
            MSR_IBS_FETCH_CONTROL => ibs.fetch_control_at(retired),
            MSR_IBS_FETCH_LINEAR => ibs.fetch_linear,
            MSR_IBS_FETCH_PHYSICAL => ibs.fetch_physical,
            MSR_IBS_OP_CONTROL => ibs.op_control_at(retired),
            MSR_IBS_OP_RIP => ibs.op_rip,
            MSR_IBS_OP_DATA => ibs.op_data,
            MSR_IBS_OP_DATA2 => ibs.op_data2,
            MSR_IBS_OP_DATA3 => ibs.op_data3,
            MSR_IBS_DC_LINEAR => ibs.dc_linear,
            MSR_IBS_DC_PHYSICAL => ibs.dc_physical,
            MSR_IBS_CONTROL => IBS_CONTROL_LVT_OFFSET_VALID | IBS_LVT_OFFSET as u64,
            MSR_IBS_BRANCH_TARGET => ibs.branch_target,
            _ => return Err(Exception::general_protection(0)),
        })
    }

    /// A control write restarts its sampler's count from the value written; the wrmsr itself is
    /// not counted. The data registers take whatever is written (software has no reason to).
    pub(crate) fn ibs_wrmsr(&mut self, msr: u32, value: u64) -> ExecResult<()> {
        let retired = self.ibs_retired();
        let ibs = &mut self.pmu.ibs;
        match msr {
            MSR_IBS_FETCH_CONTROL => {
                ibs.fetch_control = value;
                ibs.fetch_origin = retired;
            }
            MSR_IBS_OP_CONTROL => {
                ibs.op_control = value & OP_WRITABLE;
                ibs.op_origin = retired;
            }
            MSR_IBS_FETCH_LINEAR => ibs.fetch_linear = value,
            MSR_IBS_FETCH_PHYSICAL => ibs.fetch_physical = value,
            MSR_IBS_OP_RIP => ibs.op_rip = value,
            MSR_IBS_OP_DATA => ibs.op_data = value,
            MSR_IBS_OP_DATA2 => ibs.op_data2 = value,
            MSR_IBS_OP_DATA3 => ibs.op_data3 = value,
            MSR_IBS_DC_LINEAR => ibs.dc_linear = value,
            MSR_IBS_DC_PHYSICAL => ibs.dc_physical = value,
            MSR_IBS_BRANCH_TARGET => ibs.branch_target = value,
            // MSR_IBS_CONTROL is read-only
            _ => return Err(Exception::general_protection(0)),
        }
        if matches!(msr, MSR_IBS_FETCH_CONTROL | MSR_IBS_OP_CONTROL) {
            self.pmu_sync();
        }
        Ok(())
    }

    /// Before an interpreted instruction runs: whether its fetch and its op are the tagged ones
    /// (or past them, if the one tagged did not retire), and if the op is, watches its accesses.
    pub(super) fn ibs_before(&mut self) -> (bool, bool) {
        let ibs = &self.pmu.ibs;
        if ibs.fetch_target.min(ibs.op_target) == u64::MAX {
            return (false, false);
        }
        let retired = self.ibs_retired();
        let fetch = retired >= ibs.fetch_target;
        let op = retired >= ibs.op_target;
        if op {
            self.pmu.ibs.capture = Some(Capture::default());
        }
        (fetch, op)
    }

    /// After it ran: takes the samples `ibs_before` tagged and raises the interrupt. A faulting
    /// instruction has not retired: its op is not sampled, and the next one is tagged instead.
    pub(super) fn ibs_after(
        &mut self,
        (fetch, op): (bool, bool),
        insn: &Insn,
        from: u64,
        result: &ExecResult<()>,
    ) {
        let capture = self.pmu.ibs.capture.take();
        let retired = match result {
            Ok(()) => true,
            Err(Exception::Fault { software, .. }) => *software,
            Err(_) => false,
        };
        let mut sampled = false;
        if fetch {
            let physical = self.cached_physical(from, AccessKind::Exec);
            let ibs = &mut self.pmu.ibs;
            ibs.fetch_linear = from;
            ibs.fetch_physical = physical.unwrap_or(0);
            let maximum = ibs.fetch_control & FETCH_MAX_COUNT_MASK;
            ibs.fetch_control = (ibs.fetch_control & !FETCH_HARDWARE_FIELDS & !FETCH_COUNT_MASK)
                | maximum << FETCH_COUNT_SHIFT
                | FETCH_VALID
                | FETCH_COMPLETE
                | if physical.is_some() {
                    FETCH_PHYSICAL_VALID
                } else {
                    0
                };
            sampled = true;
        }
        if op && retired {
            self.ibs_sample_op(insn, from, capture.unwrap_or_default());
            sampled = true;
        }
        if sampled {
            self.dev
                .lock()
                .lapic_mut()
                .extended_interrupt(IBS_LVT_OFFSET);
            // Taken at the next boundary, as the PMU's overflow interrupt is.
            self.next_event = self.icount;
            self.pmu_sync();
        }
    }

    fn ibs_sample_op(&mut self, insn: &Insn, from: u64, capture: Capture) {
        let to = self.rip;
        let kind = BranchKind::of(insn);
        let taken = kind.is_some() && to != from.wrapping_add(insn.len as u64);
        // ret, retf, iret
        let is_return = matches!(insn.op, 0xc2 | 0xc3 | 0xca | 0xcb | 0xcf);
        let mut data = 0;
        if kind.is_some() {
            data |= OP_DATA_BRANCH_RETIRED;
        }
        if taken {
            data |= OP_DATA_BRANCH_TAKEN;
        }
        if is_return {
            data |= OP_DATA_RETURN;
        }
        let mut data3 = 0;
        let (mut linear, mut physical) = (0, 0);
        if let Some((address, size, write)) = capture.first {
            linear = address;
            data3 |= OP_DATA3_LINEAR_VALID | width_code(size) << OP_DATA3_WIDTH_SHIFT;
            let kind = if write {
                AccessKind::Write
            } else {
                AccessKind::Read
            };
            // A store to a page that holds code keeps no write tag; its read tag still says
            // where it went. Neither: not RAM, so uncacheable (device) memory.
            match self
                .cached_physical(address, kind)
                .or_else(|| self.cached_physical(address, AccessKind::Read))
            {
                Some(address) => {
                    physical = address;
                    data3 |= OP_DATA3_PHYSICAL_VALID;
                }
                None => data3 |= OP_DATA3_UNCACHEABLE,
            }
        }
        if capture.loaded {
            data3 |= OP_DATA3_LOAD;
        }
        if capture.stored {
            data3 |= OP_DATA3_STORE;
        }
        if insn.lock && capture.first.is_some() {
            data3 |= OP_DATA3_LOCKED;
        }
        let retired = self.ibs_retired();
        let ibs = &mut self.pmu.ibs;
        ibs.op_control = ibs.op_control_at(retired) | OP_VALID;
        ibs.op_rip = from;
        ibs.op_data = data;
        ibs.op_data2 = 0;
        ibs.op_data3 = data3;
        ibs.dc_linear = linear;
        ibs.dc_physical = physical;
        // For a branch, where execution went: its target when taken, the next instruction
        // otherwise.
        ibs.branch_target = if kind.is_some() { to } else { 0 };
    }

    /// Part of `pmu_reschedule`: where the next tag is, as (whether the interpreter must run
    /// now, how many instructions the JIT may run before it must stop).
    pub(super) fn ibs_schedule(&mut self, retired: u64) -> (bool, u64) {
        let target = self.pmu.ibs.retarget();
        if target == u64::MAX {
            return (false, u64::MAX);
        }
        let remaining = target.saturating_sub(retired);
        if remaining <= INTERPRET_WINDOW {
            (true, u64::MAX)
        } else {
            (false, remaining - INTERPRET_WINDOW)
        }
    }
}

#[cfg(test)]
#[path = "ibs_tests.rs"]
mod tests;
