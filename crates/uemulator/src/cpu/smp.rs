//! Several vCPUs on one host thread: a cooperative round-robin scheduler (`Smp`) and the
//! per-vCPU pieces it needs (INIT and start-up, wake predicates, switching the machine-wide state).
//!
//! Each vCPU is a `Cpu`. Only one runs at a time, so the machine-wide state (RAM, devices, JIT,
//! decode cache, the clock) is not duplicated: it lives in the `Cpu` that ran last and
//! `Cpu::swap_shared` moves it to the next one, a dozen pointer-sized swaps. Everything else in
//! `Cpu` is that vCPU's own, including its soft TLB, which is why INVLPG and CR3 writes need no
//! cross-vCPU work here (the guest shoots down TLBs with IPIs, as on hardware).
//!
//! The clock is the machine's instruction count, and `icount` travels with the shared state, so
//! guest time advances by the instructions of whichever vCPU runs and stays monotonic across
//! switches. Guest time therefore runs N times faster per vCPU-instruction when N vCPUs are busy,
//! the price of one host thread; idle (all halted) time is skipped as with one vCPU.
//!
//! The quantum is `EVENT_INTERVAL`: a vCPU gives way at every device event tick, when it halts,
//! and at `pause` (a spin loop is waiting for another vCPU).

use super::*;

pub(crate) struct Smp {
    pub(super) cpus: Vec<Cpu>,
    /// The vCPU whose `Cpu` owns the machine-wide state.
    active: usize,
    /// Consecutive slices that ended halted without retiring an instruction. A whole round of
    /// them means every vCPU claims to be woken by something that does not get delivered; the
    /// machine idles (advances time) instead of spinning.
    stalled_slices: usize,
}

/// The vCPUs of a machine that gives each its own thread: `bootstrap` and the parked secondaries
/// that share its RAM and devices.
pub(crate) fn secondary_cpus(bootstrap: Cpu) -> Vec<Cpu> {
    Smp::new(bootstrap).cpus
}

impl Smp {
    /// `bootstrap` is vCPU 0 and owns the machine-wide state; the others are created here, parked
    /// until a start-up IPI reaches them.
    pub fn new(bootstrap: Cpu) -> Self {
        let count = bootstrap.cpu_count;
        let mut cpus = vec![bootstrap];
        for index in 1..count {
            let secondary = Cpu::new_secondary(index, &cpus[0]);
            cpus.push(secondary);
        }
        Smp {
            cpus,
            active: 0,
            stalled_slices: 0,
        }
    }

    /// The vCPU that owns the machine-wide state: what the machine inspects after the run.
    pub fn into_active(mut self) -> Cpu {
        self.cpus.swap_remove(self.active)
    }

    #[cfg(test)]
    pub fn active(&self) -> &Cpu {
        &self.cpus[self.active]
    }

    #[cfg(test)]
    pub fn active_mut(&mut self) -> &mut Cpu {
        &mut self.cpus[self.active]
    }

    /// A line per vCPU for the exit dump.
    pub fn summary(&self) -> String {
        self.cpus
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
            .collect()
    }

    pub fn run(&mut self) -> ExitReason {
        loop {
            self.cpus[self.active].apply_start_signals();
            let icount_before = self.cpus[self.active].icount;
            let slice = self.cpus[self.active].run_slice();
            let progressed = self.cpus[self.active].icount != icount_before;
            match slice {
                Slice::Exit(reason) => return reason,
                Slice::Halted if !progressed => self.stalled_slices += 1,
                Slice::Halted | Slice::Yield => self.stalled_slices = 0,
            }
            match self.pick_next() {
                Some(next) => self.switch_to(next),
                None => {
                    self.stalled_slices = 0;
                    if let Some(reason) = self.idle_all() {
                        return reason;
                    }
                }
            }
        }
    }

    /// The next vCPU after the active one that can run, the active one last (it may have only
    /// used up its quantum). `None` when all are halted with nothing to wake them.
    fn pick_next(&mut self) -> Option<usize> {
        let count = self.cpus.len();
        if self.stalled_slices >= count {
            return None;
        }
        let active = self.active;
        (1..=count)
            .map(|step| (active + step) % count)
            .find(|&index| self.runnable(index))
    }

    fn runnable(&mut self, index: usize) -> bool {
        let cpu = &self.cpus[index];
        if !cpu.halted {
            return true;
        }
        let interrupts_enabled = cpu.eflags & flags::IF != 0 && cpu.svm.gif;
        let waiting_for_start = cpu.wait_for_sipi;
        let watch_fired = cpu.mwait_watch.is_some_and(|(line, contents)| {
            let line = line as usize;
            self.cpus[self.active]
                .ram
                .get(line..line + MONITOR_WATCH_BYTES)
                .is_some_and(|now| now != contents)
        });
        if watch_fired {
            let cpu = &mut self.cpus[index];
            cpu.halted = false;
            cpu.mwait_watch = None;
            return true;
        }
        let dev = &mut self.cpus[self.active].dev.lock();
        if waiting_for_start {
            // Only a start-up (or another INIT) leaves this state; interrupts and NMIs are held.
            return dev.lapic_of(index).start_signal_pending();
        }
        dev.wake_pending(index, interrupts_enabled)
    }

    /// Everyone is halted: advance time to the next device event, or end the run if nothing can
    /// ever wake a vCPU.
    fn idle_all(&mut self) -> Option<ExitReason> {
        let may_wake = self
            .cpus
            .iter()
            .any(|cpu| !cpu.wait_for_sipi && cpu.eflags & flags::IF != 0)
            || self.cpus[self.active].dev.lock().any_nmi_pending();
        self.cpus[self.active].idle(may_wake)
    }

    fn switch_to(&mut self, target: usize) {
        let from = self.active;
        if from != target {
            let (low, high) = (from.min(target), from.max(target));
            let (head, tail) = self.cpus.split_at_mut(high);
            head[low].swap_shared(&mut tail[0]);
            self.active = target;
        }
        let running = self.cpus.iter().filter(|cpu| !cpu.halted).count() as u64;
        // The vCPU being entered counts even if it is halted now: it wakes up in its slice.
        let divisor = if self.cpus[target].halted {
            running + 1
        } else {
            running
        };
        let cpu = &mut self.cpus[target];
        cpu.start_slice(divisor.max(1));
        cpu.dev.lock().set_current_cpu(target);
        cpu.revoke_stale_write_tags();
    }
}

impl Cpu {
    /// Hands the machine-wide state to `other` (which must be the vCPU that does not hold it
    /// now; the roles swap). Every field of `Cpu` that is not per-vCPU must be listed here: one
    /// that is missed would stay with its old vCPU and the next one would run on a placeholder.
    /// RAM, the code-page flags and the devices are not listed: every vCPU holds a handle on the
    /// same ones (`ram.rs`, `dev/handle.rs`).
    pub(super) fn swap_shared(&mut self, other: &mut Cpu) {
        use core::mem::swap;
        swap(&mut self.pages_with_code, &mut other.pages_with_code);
        swap(&mut self.decode_cache, &mut other.decode_cache);
        swap(
            &mut self.decode_cache_generation,
            &mut other.decode_cache_generation,
        );
        #[cfg(not(freestanding))]
        swap(&mut self.bios, &mut other.bios);
        swap(&mut self.jit, &mut other.jit);
        swap(&mut self.icount, &mut other.icount);
        swap(&mut self.idle_skipped_ns, &mut other.idle_skipped_ns);
        swap(&mut self.time_base_ns, &mut other.time_base_ns);
        swap(&mut self.slice_base_icount, &mut other.slice_base_icount);
        swap(&mut self.time_divisor, &mut other.time_divisor);
        swap(&mut self.time_carry, &mut other.time_carry);
        swap(&mut self.next_event, &mut other.next_event);
        swap(&mut self.rng_state, &mut other.rng_state);
        swap(&mut self.flushes, &mut other.flushes);
        swap(&mut self.poll, &mut other.poll);
        swap(&mut self.log, &mut other.log);
        swap(&mut self.logged, &mut other.logged);
        swap(&mut self.trace, &mut other.trace);
        swap(&mut self.code_mark_epoch, &mut other.code_mark_epoch);
    }

    /// Folds the time the last slice took into the clock and starts the next with `divisor`
    /// vCPUs sharing it. N vCPUs running at once retire N instructions per guest nanosecond
    /// between them (each runs at the 1 instruction per ns of a single vCPU), so a parallel guest
    /// workload finishes sooner in guest time than a serial one, while the clock stays monotonic
    /// and depends only on instruction counts.
    pub(super) fn start_slice(&mut self, divisor: u64) {
        let busy = self.icount - self.slice_base_icount + self.time_carry;
        self.time_base_ns += busy / self.time_divisor * NS_PER_INSN;
        self.time_carry = if divisor == self.time_divisor {
            busy % self.time_divisor
        } else {
            0
        };
        self.slice_base_icount = self.icount;
        self.time_divisor = divisor;
    }

    /// Ends this vCPU's quantum at the end of the current instruction (`pause`). The yield rides
    /// on the event check the run loop already does: the deadline is pulled to 0 and
    /// `before_step` puts it back (`pause_resume_event`), so the hot path gains no test.
    pub(super) fn request_yield(&mut self) {
        if self.pause_resume_event == 0 {
            self.pause_resume_event = self.next_event;
            self.next_event = 0;
            // Chained JIT blocks stop at the next block boundary on `icount >= next_event`;
            // this also stops the one that is running.
            self.block_exit_requested = true;
        }
    }

    /// Drops write tags for pages that gained translated code while this vCPU was not running;
    /// without it a store through a stale tag would skip `code_written` and leave a stale JIT
    /// block (or decoded instruction) behind.
    pub(super) fn revoke_stale_write_tags(&mut self) {
        if self.code_mark_seen == self.code_mark_epoch {
            return;
        }
        self.code_mark_seen = self.code_mark_epoch;
        let ram_base = self.ram_base;
        // The TLB SVM parked comes back without a refill, so it is revoked too.
        let parked = self
            .svm
            .parked_tlb
            .as_deref_mut()
            .map(|parked| &mut parked.entries);
        for entries in core::iter::once(&mut self.tlb).chain(parked) {
            for entry in entries.iter_mut() {
                if entry.tag_w != TLB_TAG_INVALID
                    && self
                        .page_has_code
                        .get_or_zero(super::mem::write_tag_page(entry, ram_base))
                        != 0
                {
                    entry.tag_w = TLB_TAG_INVALID;
                }
            }
        }
    }

    /// INIT (SDM 9.1, table 9-1): the registers go back to their reset values and the vCPU waits
    /// for a start-up IPI. Memory, the TSC, the MTRRs and the other MSRs keep what they had.
    pub(super) fn init(&mut self) {
        self.gpr = [0; 16];
        self.reset_real_mode_registers();
        self.cr2 = 0;
        self.pdpte = [0; 4];
        self.dr = [0, 0, 0, 0, 0, 0, DR6_RESET, DR7_RESET];
        self.xcr0 = xsave::XCR0_X87;
        self.mxcsr = MXCSR_RESET;
        self.fcw = FCW_RESET;
        self.fsw = 0;
        self.ftw = 0;
        self.st = [[0; 10]; 8];
        self.xmm = [0; 16];
        self.ymm_hi = [0; 16];
        self.zmm_hi = [[0; 2]; 16];
        self.zmm_extra = [[0; 4]; 16];
        self.opmask = [0; 8];
        self.svm = svm::Svm::new();
        self.nmi_blocked = false;
        self.interrupt_shadow = false;
        self.monitor_armed = false;
        self.monitor_snapshot = None;
        self.mwait_watch = None;
        self.flush_tlb();
        self.halted = true;
        self.wait_for_sipi = true;
    }

    /// Start-up IPI: real mode at `vector << 12` (CS selector `vector << 8`, IP 0).
    pub(super) fn start_at(&mut self, vector: u8) {
        self.seg[SEG_CS].sel = (vector as u16) << 8;
        self.seg[SEG_CS].base = (vector as u64) << 12;
        self.rip = 0;
        self.halted = false;
        self.wait_for_sipi = false;
    }
}
