//! Smaller protection and power features that CPUID advertises together with their CR4 bits:
//! UMIP, SMEP/SMAP (clac/stac; the checks live in the page walk in `mem.rs`), PCID/INVPCID, and
//! MONITOR/MWAIT with AMD's unprivileged MONITORX/MWAITX.

use super::decode::Insn;
use super::exception::Exception;
use super::exec::Loc;
use super::sys::{CR4_UMIP, RAX, RCX};
use super::{Cpu, ExecResult, MONITOR_WATCH_BYTES};

/// INVPCID descriptor: PCID in bits 11:0 (the rest of the first quadword is reserved) followed by
/// a linear address.
const INVPCID_DESCRIPTOR_RESERVED: u64 = !0xfff;
const INVPCID_TYPE_MAX: u64 = 3;
const INVPCID_TYPE_INDIVIDUAL_ADDRESS: u64 = 0;
/// Types 1 (one PCID) and 3 (every PCID) leave global translations alone; 0 and 2 do not.
const INVPCID_TYPE_SINGLE_CONTEXT: u64 = 1;
const INVPCID_TYPE_ALL_CONTEXTS_EXCEPT_GLOBAL: u64 = 3;

/// MWAIT's ECX extensions: bit 0 = treat masked interrupts as break events. Other bits are
/// reserved and #GP.
const MWAIT_ECX_INTERRUPT_BREAK: u64 = 1;
/// MONITOR's ECX/EDX (extensions, hints) must be zero here: no extensions are enumerated.
const MONITOR_NO_EXTENSIONS: u64 = 0;
/// MWAITX's ECX bit 1: EBX holds a timeout in TSC-rate clocks after which the wait ends.
const MWAITX_ECX_TIMER: u64 = 1 << 1;

impl Cpu {
    /// With CR4.UMIP set, the descriptor-table and machine-status reads (sgdt, sidt, sldt, str,
    /// smsw) are #GP(0) outside CPL 0, so user code cannot learn kernel addresses from them.
    pub(super) fn umip_check(&self) -> ExecResult<()> {
        if self.cr4 & CR4_UMIP != 0 && self.cpl != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// 66 0F 38 82 (invpcid r64, m128). The soft TLB is not tagged with PCIDs, so each type
    /// flushes every PCID's entries, which is always a legal over-approximation of the narrower
    /// ones; the types that keep global translations keep them here too. Operand validation is
    /// as specified so a guest probing the instruction sees the faults.
    pub(super) fn invpcid(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.md == 3 || !i.p66 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        let kind = self.gpr[i.reg as usize];
        let Loc::Mem(address) = self.modrm_location(i, next)? else {
            return Err(Exception::invalid_opcode());
        };
        if kind > INVPCID_TYPE_MAX {
            return Err(Exception::general_protection(0));
        }
        let low = self.read(address, 8)?;
        let linear = self.read(address + 8, 8)?;
        if low & INVPCID_DESCRIPTOR_RESERVED != 0 {
            return Err(Exception::general_protection(0));
        }
        if kind == INVPCID_TYPE_INDIVIDUAL_ADDRESS && !self.is_canonical_linear(linear) {
            return Err(Exception::general_protection(0));
        }
        if matches!(
            kind,
            INVPCID_TYPE_SINGLE_CONTEXT | INVPCID_TYPE_ALL_CONTEXTS_EXCEPT_GLOBAL
        ) {
            self.flush_tlb_non_global();
        } else {
            self.flush_tlb();
        }
        self.rip = next;
        Ok(())
    }

    /// 0F 01 C8. CPL 0 only. With one vCPU nothing but device DMA writes guest RAM while it
    /// waits, so MWAIT only wakes on interrupts; with several, the armed line is watched
    /// (`monitor_snapshot`, then `mwait_watch`) and a store to it by another vCPU wakes it.
    pub(super) fn monitor(&mut self) -> ExecResult<()> {
        self.need_cpl0()?;
        if self.gpr[RCX] != MONITOR_NO_EXTENSIONS
            || self.gpr[super::sys::RDX] != MONITOR_NO_EXTENSIONS
        {
            return Err(Exception::general_protection(0));
        }
        self.arm_monitor();
        Ok(())
    }

    /// 0F 01 FA without a prefix (AMD): MONITOR at any CPL. ECX (extensions) must be zero; EDX
    /// holds hints, of which none are defined, so it is ignored (APM vol. 3, MONITORX).
    pub(super) fn monitorx(&mut self) -> ExecResult<()> {
        if self.gpr[RCX] != MONITOR_NO_EXTENSIONS {
            return Err(Exception::general_protection(0));
        }
        self.arm_monitor();
        Ok(())
    }

    fn arm_monitor(&mut self) {
        self.monitor_address = self.gpr[RAX];
        self.monitor_armed = true;
        if self.multi {
            // The line is armed here, not at MWAIT: a store between the two must make MWAIT
            // return at once, or a vCPU that was handed work in that window sleeps through it.
            self.monitor_snapshot = self.snapshot_monitored_line();
        }
    }

    /// The monitored line and its contents, so a store to it by another vCPU (or a device) can
    /// be told from nothing having happened: that is how a remote CPU hands work to an idle one
    /// without an IPI (the kernel's idle loop polls a flag with MONITOR/MWAIT). An address that
    /// does not translate is not watched: only interrupts wake it.
    fn snapshot_monitored_line(&mut self) -> Option<(u64, [u8; MONITOR_WATCH_BYTES])> {
        let physical = self
            .translate(self.monitor_address, super::mem::AccessKind::Read, true)
            .ok()?;
        let line = physical & !(MONITOR_WATCH_BYTES as u64 - 1);
        let contents = self.ram.slice(line, MONITOR_WATCH_BYTES)?;
        Some((line, contents.try_into().unwrap()))
    }

    /// 0F 01 C9. CPL 0 only, #GP on reserved ECX bits. Waits like `hlt` while interrupts are
    /// enabled; with IF=0 and no break-on-masked-interrupt it returns immediately, because a
    /// `hlt`-style stop there would be the machine-shutdown case in the run loop and MWAIT may
    /// always return early.
    pub(super) fn mwait(&mut self) -> ExecResult<()> {
        self.need_cpl0()?;
        if self.gpr[RCX] & !MWAIT_ECX_INTERRUPT_BREAK != 0 {
            return Err(Exception::general_protection(0));
        }
        self.wait_on_monitor();
        Ok(())
    }

    /// 0F 01 FB (AMD): MWAIT at any CPL, with an optional timeout (ECX bit 1, EBX clocks). Linux
    /// spins in `udelay` with it. The timed form returns at once, which MWAITX may always do (the
    /// caller rereads the TSC and waits again): waking the vCPU at a deadline would need a timer
    /// source in all three schedulers, and the busy loop it leaves behind yields like `pause`.
    pub(super) fn mwaitx(&mut self) -> ExecResult<()> {
        if self.gpr[RCX] & !(MWAIT_ECX_INTERRUPT_BREAK | MWAITX_ECX_TIMER) != 0 {
            return Err(Exception::general_protection(0));
        }
        if self.gpr[RCX] & MWAITX_ECX_TIMER != 0 {
            self.monitor_armed = false;
            self.monitor_snapshot = None;
            if self.threads.is_some() {
                self.pause_threaded();
            } else if self.multi {
                self.request_yield();
            }
            return Ok(());
        }
        self.wait_on_monitor();
        Ok(())
    }

    fn wait_on_monitor(&mut self) {
        if !self.monitor_armed {
            // Without a preceding monitor, mwait is a no-op in hardware terms: it just returns.
            return;
        }
        self.monitor_armed = false;
        let snapshot = self.monitor_snapshot.take();
        if self.eflags_full() & super::flags::IF != 0 {
            if self.multi
                && let Some((line, contents)) = snapshot
            {
                let line_now = &self.ram[line as usize..line as usize + MONITOR_WATCH_BYTES];
                if line_now != contents {
                    // Written since MONITOR: MWAIT returns without waiting.
                    return;
                }
                self.mwait_watch = snapshot;
            }
            self.halted = true;
        }
    }
}
