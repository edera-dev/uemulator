//! The debug registers in action (APM vol. 2 ch. 13, SDM vol. 3 ch. 18): instruction, data and
//! I/O breakpoints, single-step (with DEBUGCTL.BTF), general detect, the task-switch T bit,
//! bus-lock detect (DEBUGCTL.BLD) and the BPEXT address masks (with AdMskExtn).
//!
//! Generated code never checks for any of it. Whenever something could raise a #DB
//! (`Cpu::debug_slow_path`) the run loop executes one instruction at a time through
//! `Cpu::step_debug` instead of a JIT block, and `helper_exec` ends a block as soon as an
//! instruction it runs turns any of it on, so the JIT and the interpreter agree without the JIT
//! knowing about breakpoints. The fast path pays one test per block and one bool per interpreted
//! memory access (`DebugState::watch_accesses`). Bus-lock detect is the exception: it stays on
//! (Linux enables it by default), and `Cpu::exec` catches split locks on any engine.

use super::exception::Exception;
use super::flags::{RF, TF};
use super::*;
use crate::snapshot::Snapshot;

pub(crate) const VECTOR_DEBUG: u8 = 1;

/// DR6 bit 11, bus lock detected. Active low: it reads 1 until a bus lock clears it, and software
/// writes 1 to rearm it. Internally (`DebugState::trap_bits`) it is set to mean "detected".
pub(crate) const DR6_BUS_LOCK: u64 = 1 << 11;
/// DR6 BD: a debug register access while DR7.GD was set.
pub(crate) const DR6_DEBUG_REGISTER_ACCESS: u64 = 1 << 13;
/// DR6 BS: single-step.
pub(crate) const DR6_SINGLE_STEP: u64 = 1 << 14;
/// DR6 BT: a task switch into a TSS with its T bit set.
pub(crate) const DR6_TASK_SWITCH: u64 = 1 << 15;

/// DR7 L0, G0, ... L3, G3: a breakpoint is enabled when either of its two bits is.
const DR7_ENABLES: u64 = 0xff;
/// DR7 GD: the next debug register access raises #DB (BD) instead. Cleared on #DB entry, so the
/// handler can use the registers.
pub(crate) const DR7_GENERAL_DETECT: u64 = 1 << 13;
/// DR7 bits 31:16 hold a 4-bit control field per breakpoint: R/W in the low two bits, LEN above.
const DR7_CONTROL_SHIFT: u32 = 16;
const DR7_CONTROL_BITS: u32 = 4;
const RW_EXECUTE: u64 = 0;
const RW_WRITE: u64 = 1;
/// I/O breakpoint; only with CR4.DE (undefined, here never matching, without it).
const RW_IO: u64 = 2;
const RW_READ_WRITE: u64 = 3;
/// Bytes covered by each LEN encoding: 00 one, 01 two, 10 eight (long mode), 11 four.
const LEN_BYTES: [u64; 4] = [1, 2, 8, 4];
const BREAKPOINTS: usize = 4;

/// IA32_DEBUGCTL (AMD: DebugCtl). Bits 0 (LBR) and 3-5 are stored and do nothing.
pub(crate) const MSR_DEBUGCTL: u32 = 0x1d9;
/// Branch single-step: with TF, trap only after an instruction that branched.
pub(crate) const DEBUGCTL_BTF: u64 = 1 << 1;
/// Bus lock detect (Zen 5 "bus lock trap", CPUID 7 ECX bit 24): #DB trap after an instruction
/// that locked the bus, which here is a locked access split across a cache line.
pub(crate) const DEBUGCTL_BUS_LOCK_DETECT: u64 = 1 << 2;
/// DebugCtl bits that exist: LBR (0), BTF (1) and PB0-PB3 (2-5, the old performance-monitor pin
/// controls, of which bit 2 is now BLD). Writing any other is #GP (APM vol. 2 13.1.6).
const DEBUGCTL_WRITABLE: u64 = 0x3f;

/// BPEXT (CPUID 0x8000_0001 ECX bit 26) address masks, DR0's apart from the other three's (it
/// existed first, on family 16h). A set bit is left out of that breakpoint's address compare.
/// AdMskExtn (bit 30, which every Zen since family 17h sets) widens them from bits 11:0 to 31:0
/// and makes instruction breakpoints honour them as well as data ones; that is the only CPU
/// modelled, so both always hold. The CPUID override MSR hiding the bit changes neither, as on
/// hardware.
pub(crate) const MSR_DR0_ADDRESS_MASK: u32 = 0xc001_1027;
pub(crate) const MSR_DR1_ADDRESS_MASK: u32 = 0xc001_1019;
pub(crate) const MSR_DR3_ADDRESS_MASK: u32 = 0xc001_101b;
/// The masks are 32 bits wide; writing the upper half is #GP.
const ADDRESS_MASK_WRITABLE: u64 = 0xffff_ffff;

/// A locked access that crosses this boundary is a split lock, which takes the bus lock.
const CACHE_LINE_BYTES: u64 = 64;

/// Opcodes whose single-step trap looks at TF after the instruction instead of before: syscall
/// masks TF through SFMASK before the trap would be taken, and a sysret that sets TF traps right
/// after it (QEMU's `gen_eob_syscall` behaves the same way).
const OP_SYSCALL: u16 = 0x105;
const OP_SYSRET: u16 = 0x107;

/// Debug state beyond the debug registers themselves (`Cpu::dr`).
#[derive(Default, Snapshot)]
pub(crate) struct DebugState {
    pub debugctl: u64,
    /// DR0-DR3 address masks (BPEXT).
    pub address_masks: [u64; BREAKPOINTS],
    /// Raised by `step_debug` around the one instruction it runs when an access could hit a data
    /// or I/O breakpoint or take a bus lock; false everywhere else, so accesses made by event
    /// delivery never match (which is what CPUID's NO_NESTED_DATA_BP promises for #DB).
    pub watch_accesses: bool,
    /// DR6 bits the running instruction collected so far.
    pub trap_bits: u64,
    /// The running instruction made an access that crosses a cache line.
    pub split_access: bool,
    /// The running instruction loaded SS through `mov ss` or `pop ss`: its traps wait until the
    /// next instruction has run (SDM 6.8.3). `sti` holds off interrupts but not #DB.
    pub mov_ss_shadow: bool,
    /// Traps a `mov ss`/`pop ss` held back, delivered after the next instruction.
    pub deferred_bits: u64,
    /// The instruction after the `mov ss` delivered an event instead of completing: the held
    /// back #DB is taken before the handler's first instruction, in the handler's context (the
    /// CVE-2018-8897 behavior every x86 CPU has).
    pub deferred_ready: bool,
    /// A task switch went to a TSS with its T bit set: #DB (BT) before the new task's first
    /// instruction.
    pub task_switch_trap: bool,
}

impl DebugState {
    /// Bus-lock detect is not here: Linux turns it on at boot on every CPU that has it, so it
    /// must not cost the JIT. Locked instructions always run in `Cpu::exec`, which checks for it
    /// (`exec_detecting_bus_lock`).
    fn needs_attention(&self) -> bool {
        self.deferred_bits != 0 || self.task_switch_trap
    }
}

/// Breakpoint `index`'s R/W and LEN fields, or `None` while it is disabled.
fn breakpoint_control(dr7: u64, index: usize) -> Option<(u64, u64)> {
    if (dr7 >> (2 * index)) & 3 == 0 {
        return None;
    }
    let control = dr7 >> (DR7_CONTROL_SHIFT + DR7_CONTROL_BITS * index as u32);
    Some((control & 3, LEN_BYTES[((control >> 2) & 3) as usize]))
}

impl Cpu {
    /// Whether the next instruction must go through `step_debug`: something may raise a #DB.
    #[inline]
    pub(crate) fn debug_slow_path(&self) -> bool {
        self.eflags & (TF | RF) != 0
            || self.dr[7] & DR7_ENABLES != 0
            || self.debug.needs_attention()
            || self.gdb.armed
    }

    /// One step of the run loop: a JIT block where one may run (`Cpu::uses_jit`), else one
    /// interpreted instruction, through `step_debug` when it may raise a #DB.
    /// `dispatch_step` for the instruction in the shadow of `sti` or `mov ss`: it runs alone,
    /// never at the head of a JIT block, so the run loop looks for an interrupt right after it.
    /// A block would run on past the end of the shadow, and a loop like iPXE's
    /// `sti; nop; nop; cli` (the only window it gives the timer) would never take one. QEMU ends
    /// the TB after the instruction following `sti` for the same reason.
    pub(crate) fn dispatch_shadowed_step(&mut self) -> ExecResult<()> {
        if self.debug_slow_path() {
            self.step_debug()
        } else {
            self.step()
        }
    }

    pub(crate) fn dispatch_step(&mut self) -> ExecResult<()> {
        if self.debug_slow_path() {
            self.step_debug()
        } else if self.uses_jit() {
            self.step_block()
        } else {
            self.step()
        }
    }

    /// Marks the #DB `bits` describe in DR6 and returns it. DR6 is sticky: the processor sets
    /// status bits and never clears them, except BLD, which is active low.
    pub(crate) fn debug_exception(&mut self, bits: u64) -> Exception {
        let mut dr6 = self.dr[6] | (bits & !DR6_BUS_LOCK);
        if bits & DR6_BUS_LOCK != 0 {
            dr6 &= !DR6_BUS_LOCK;
        }
        self.dr[6] = dr6;
        self.dr[7] &= !DR7_GENERAL_DETECT;
        Exception::fault(VECTOR_DEBUG, None)
    }

    /// MOV to or from a debug register with DR7.GD set: a fault, with RIP on the instruction.
    pub(crate) fn general_detect(&mut self) -> ExecResult<()> {
        if self.dr[7] & DR7_GENERAL_DETECT != 0 {
            return Err(self.debug_exception(DR6_DEBUG_REGISTER_ACCESS));
        }
        Ok(())
    }

    /// The instruction breakpoints the instruction at RIP hits. LEN is ignored for them (it must be
    /// 00); the address mask is not (AdMskExtn).
    fn instruction_breakpoint_hits(&self) -> u64 {
        let Ok(linear) = self.fetch_linear() else {
            return 0;
        };
        let mut hits = 0;
        for index in 0..BREAKPOINTS {
            if let Some((RW_EXECUTE, _)) = breakpoint_control(self.dr[7], index)
                && (self.dr[index] ^ linear) & !self.debug.address_masks[index] == 0
            {
                hits |= 1 << index;
            }
        }
        hits
    }

    /// Whether an access may hit a data or I/O breakpoint, so `watch_accesses` must be on.
    fn access_breakpoints_armed(&self) -> bool {
        (0..BREAKPOINTS).any(|index| match breakpoint_control(self.dr[7], index) {
            Some((RW_WRITE | RW_READ_WRITE, _)) => true,
            Some((RW_IO, _)) => self.cr4 & sys::CR4_DE != 0,
            _ => false,
        })
    }

    /// Called for every data access of `size` bytes at `linear` while `watch_accesses` is on.
    #[cold]
    pub(crate) fn watch_access(&mut self, linear: u64, size: u64, write: bool) {
        // Instruction-based sampling watches the accesses of the instruction it tagged.
        if let Some(capture) = self.pmu.ibs.capture.as_mut() {
            capture.note(linear, size, write);
        }
        if (linear % CACHE_LINE_BYTES) + size > CACHE_LINE_BYTES {
            self.debug.split_access = true;
        }
        for index in 0..BREAKPOINTS {
            let Some((rw, len)) = breakpoint_control(self.dr[7], index) else {
                continue;
            };
            if !(rw == RW_READ_WRITE || (rw == RW_WRITE && write)) {
                continue;
            }
            // LEN aligns the breakpoint down to its size; the BPEXT mask widens it further.
            let ignored = (len - 1) | self.debug.address_masks[index];
            let target = self.dr[index] & !ignored;
            if (0..size).any(|offset| linear.wrapping_add(offset) & !ignored == target) {
                self.debug.trap_bits |= 1 << index;
            }
        }
        // The debugger's watchpoints, apart from the guest's: `cpu/gdb.rs`.
        if !self.gdb.watchpoints.is_empty() {
            self.gdb_watch(linear, size, write);
        }
    }

    /// Called for every port access while `watch_accesses` is on.
    #[cold]
    pub(crate) fn watch_io(&mut self, port: u16, size: u8) {
        if self.cr4 & sys::CR4_DE == 0 {
            return;
        }
        for index in 0..BREAKPOINTS {
            if let Some((RW_IO, len)) = breakpoint_control(self.dr[7], index) {
                let target = self.dr[index] & 0xffff & !(len - 1);
                if (0..size as u64).any(|offset| (port as u64 + offset) & !(len - 1) == target) {
                    self.debug.trap_bits |= 1 << index;
                }
            }
        }
    }

    /// A repeated string instruction stops after the iteration that hit a breakpoint, or after
    /// every iteration while single-stepping, with RIP still on it (SDM 18.3.1.2): the trap is
    /// taken between iterations.
    pub(crate) fn rep_must_pause(&self) -> bool {
        self.debug.watch_accesses
            && (self.debug.trap_bits != 0 || self.eflags & TF != 0 || self.gdb.watch_hit.is_some())
    }

    /// A task switch loaded a TSS with its T bit set.
    pub(crate) fn note_task_switch_trap(&mut self) {
        self.debug.task_switch_trap = true;
    }

    /// One instruction with everything that can raise a #DB around it. Faults (instruction
    /// breakpoints, general detect) come before the instruction runs; traps (data and I/O
    /// breakpoints, single-step, bus lock) after it completes, with RIP at the next instruction.
    pub(crate) fn step_debug(&mut self) -> ExecResult<()> {
        self.gdb_note_step();
        if core::mem::take(&mut self.debug.task_switch_trap) {
            return Err(self.debug_exception(DR6_TASK_SWITCH));
        }
        if self.debug.deferred_ready {
            self.debug.deferred_ready = false;
            let bits = core::mem::take(&mut self.debug.deferred_bits);
            return Err(self.debug_exception(bits));
        }
        let flags_before = self.eflags;
        // RF suppresses instruction breakpoints for one instruction and is cleared when it
        // completes; clearing it now lets an `iret` that loads it set it again.
        if flags_before & RF == 0 {
            let hits = self.instruction_breakpoint_hits();
            if hits != 0 {
                return Err(self.debug_exception(hits));
            }
        }
        self.eflags &= !RF;
        let rip_before = self.rip;
        self.debug.trap_bits = 0;
        self.debug.split_access = false;
        self.debug.mov_ss_shadow = false;
        self.debug.watch_accesses = self.access_breakpoints_armed()
            || self.debug.debugctl & DEBUGCTL_BUS_LOCK_DETECT != 0
            || flags_before & TF != 0
            || !self.gdb.watchpoints.is_empty();
        let result = self.fetch_decode().and_then(|insn| {
            self.trace_step();
            self.exec_interpreted(&insn).map(|()| insn)
        });
        self.debug.watch_accesses = false;
        let insn = match result {
            Ok(insn) => insn,
            Err(error) => {
                self.debug.trap_bits = 0;
                // A faulting instruction did not complete; its accesses are not reported.
                self.gdb.watch_hit = None;
                if self.debug.deferred_bits != 0 {
                    self.debug.deferred_ready = true;
                }
                return Err(error);
            }
        };
        let mut bits = core::mem::take(&mut self.debug.trap_bits);
        if self.debug.debugctl & DEBUGCTL_BUS_LOCK_DETECT != 0
            && self.debug.split_access
            && Self::is_locked(&insn)
        {
            bits |= DR6_BUS_LOCK;
        }
        let stepping = if matches!(insn.op, OP_SYSCALL | OP_SYSRET) {
            self.eflags & TF != 0
        } else {
            flags_before & TF != 0
        };
        let paused_mid_string = self.rip == rip_before && is_rep_string(&insn);
        let branched = self.rip != rip_before.wrapping_add(insn.len as u64) && !paused_mid_string;
        if stepping && (self.debug.debugctl & DEBUGCTL_BTF == 0 || branched) {
            bits |= DR6_SINGLE_STEP;
        }
        if self.debug.mov_ss_shadow {
            self.debug.deferred_bits |= bits;
            return Ok(());
        }
        bits |= core::mem::take(&mut self.debug.deferred_bits);
        if bits == 0 {
            return Ok(());
        }
        // Between iterations of a string instruction RF goes into the pushed image, so the
        // instruction resumes without hitting its own instruction breakpoint again.
        if paused_mid_string {
            self.eflags |= RF;
        }
        self.halted = false;
        Err(self.debug_exception(bits))
    }

    /// A locked instruction while DEBUGCTL.BLD is set, outside `step_debug` (which looks for
    /// split locks itself): runs it watching its accesses, and makes a split one a #DB trap.
    /// Locked instructions never run in generated code (the JIT hands them to `helper_exec`), so
    /// this covers both engines.
    pub(crate) fn exec_detecting_bus_lock(&mut self, insn: &decode::Insn) -> ExecResult<()> {
        self.debug.split_access = false;
        self.debug.trap_bits = 0;
        self.debug.watch_accesses = true;
        let result = self.exec(insn);
        self.debug.watch_accesses = false;
        // Breakpoints are armed only under `step_debug`; any match recorded here is noise.
        self.debug.trap_bits = 0;
        result?;
        if self.debug.split_access {
            return Err(self.debug_exception(DR6_BUS_LOCK));
        }
        Ok(())
    }

    /// rdmsr of the debug MSRs; `None` for any other.
    pub(crate) fn debug_msr_read(&self, msr: u32) -> Option<u64> {
        match msr {
            MSR_DEBUGCTL => Some(self.debug.debugctl),
            MSR_DR0_ADDRESS_MASK => Some(self.debug.address_masks[0]),
            MSR_DR1_ADDRESS_MASK..=MSR_DR3_ADDRESS_MASK => {
                Some(self.debug.address_masks[(msr - MSR_DR1_ADDRESS_MASK) as usize + 1])
            }
            _ => None,
        }
    }

    /// wrmsr of the debug MSRs; `None` for any other.
    pub(crate) fn debug_msr_write(&mut self, msr: u32, value: u64) -> Option<ExecResult<()>> {
        let slot = match msr {
            MSR_DEBUGCTL => {
                if value & !DEBUGCTL_WRITABLE != 0 {
                    return Some(Err(Exception::general_protection(0)));
                }
                self.debug.debugctl = value;
                return Some(Ok(()));
            }
            MSR_DR0_ADDRESS_MASK => 0,
            MSR_DR1_ADDRESS_MASK..=MSR_DR3_ADDRESS_MASK => {
                (msr - MSR_DR1_ADDRESS_MASK) as usize + 1
            }
            _ => return None,
        };
        if value & !ADDRESS_MASK_WRITABLE != 0 {
            return Some(Err(Exception::general_protection(0)));
        }
        self.debug.address_masks[slot] = value;
        Some(Ok(()))
    }
}

/// movs, cmps, stos, lods, scas, ins and outs with a rep prefix.
fn is_rep_string(insn: &decode::Insn) -> bool {
    insn.rep != 0 && matches!(insn.op, 0x6c..=0x6f | 0xa4..=0xa7 | 0xaa..=0xaf)
}
