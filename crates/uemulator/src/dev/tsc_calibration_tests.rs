//! Linux's two TSC calibrations against PIT channel 2 (`arch/x86/kernel/tsc.c`), run as guest
//! code on the whole path they exercise: port 0x61 (gate, speaker, OUT2), the 8254 in mode 0,
//! MSB-only counter reads and `rdtsc`, all on one emulated clock.
//!
//! Both pass on the virtual clock and on a steady host clock. Both fail, as in Linux, when the
//! host clock jumps while the guest polls: in realtime mode a host thread that loses the CPU for
//! a while is, to the guest, an SMI in the middle of the measurement. That is what a loaded host
//! shows ("Fast TSC calibration failed", "Unable to calibrate against PIT", then the PM timer
//! reference) and it is not a device bug: no emulated clock can hide a stall of the thread that
//! reads it without leaving wall time.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cpu::WallClock;
use crate::testkit::{Harness, Stop};

const R8: usize = 8;
const R9: usize = 9;
const R10: usize = 10;
const R11: usize = 11;
const R12: usize = 12;
const R15: usize = 15;

/// `PIT_TICK_RATE` and the guest TSC rate (the TSC counts nanoseconds).
const PIT_TICK_RATE: u64 = 1_193_182;
const TSC_KHZ: u64 = 1_000_000;
/// `pit_calibrate_tsc(CAL_LATCH, CAL_MS, CAL_PIT_LOOPS)`: 10 ms, and at least 1000 polls.
const CAL_MS: u64 = 10;
const CAL_PIT_LOOPS: u64 = 1000;
/// Its SMI check: the slowest poll may take at most ten times the fastest.
const SMI_RATIO: u64 = 10;
/// `pit_calibrate_tsc` divides by 10 ms but waits `CAL_LATCH` = 11931 ticks, 9.9993 ms, so it
/// reads about 70 ppm low on any machine; one PIT tick more or less is another 84 ppm.
const SLOW_TOLERANCE_KHZ: u64 = TSC_KHZ / 5000;
/// `quick_pit_calibrate` succeeds only within 500 ppm; allow that much.
const QUICK_TOLERANCE_KHZ: u64 = TSC_KHZ / 2000;
const HZ_PER_KHZ: u64 = 1000;
/// The MSB steps every 256 PIT ticks.
const PIT_TICKS_PER_MSB: u64 = 256;
/// Enough for 10 ms of polling at two virtual nanoseconds per instruction, with room to spare.
const INSTRUCTION_BUDGET: u64 = 50_000_000;

/// `pit_calibrate_tsc`: gate high and speaker off through port 0x61, channel 2 in mode 0 with
/// `CAL_LATCH` (11931 = 0x2e9b), then poll port 0x61 bit 5 (OUT2) with an `rdtsc` per poll,
/// tracking the shortest and longest poll. Leaves t1 in r8, t2 in r9, the poll count in r10, the
/// longest poll in r11 and the shortest in r12.
const PIT_CALIBRATE_TSC: &[u8] = &[
    0xe4, 0x61, 0x24, 0xfd, 0x0c, 0x01, 0xe6, 0x61, 0xb0, 0xb0, 0xe6, 0x43, 0xb0, 0x9b, 0xe6, 0x42,
    0xb0, 0x2e, 0xe6, 0x42, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x49, 0x89, 0xc0,
    0x49, 0x89, 0xc1, 0x4d, 0x31, 0xd2, 0x4d, 0x31, 0xdb, 0x49, 0xc7, 0xc4, 0xff, 0xff, 0xff, 0xff,
    0xe4, 0x61, 0xa8, 0x20, 0x75, 0x25, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x48,
    0x89, 0xc1, 0x4c, 0x29, 0xc9, 0x49, 0x89, 0xc1, 0x4c, 0x39, 0xe1, 0x4c, 0x0f, 0x42, 0xe1, 0x4c,
    0x39, 0xd9, 0x4c, 0x0f, 0x47, 0xd9, 0x49, 0xff, 0xc2, 0xeb, 0xd5, 0xf4,
];

/// `quick_pit_calibrate`: channel 2 in mode 0 from 0xffff, then `pit_expect_msb` for 0xff, 0xfe,
/// ... (each polls two counter reads per `rdtsc`, at most 50000 times, and needs more than five
/// matches), stopping when the two edge uncertainties are below 1/2048 of the elapsed TSC and a
/// last `pit_verify_msb` agrees, after at most 233 steps. Leaves the elapsed TSC in r12 and the
/// step count in r15; both are 0 when it gives up.
///
/// ```text
///     in al, 0x61; and al, 0xfd; or al, 1; out 0x61, al
///     mov al, 0xb0; out 0x43, al; mov al, 0xff; out 0x42, al; out 0x42, al
///     in al, 0x42; in al, 0x42                     ; pit_verify_msb(0)
///     mov bl, 0xff; call expect; cmp r10, 5; jbe fail
///     mov r13, r8; mov r14, r9; mov r15, 1         ; tsc, d1, i
/// next: cmp r15, 233; ja fail
///     mov bl, 0xff; sub bl, r15b; call expect; cmp r10, 5; jbe fail
///     mov rax, r8; sub rax, r13; mov r12, rax      ; delta
///     lea rcx, [r14 + r9]                          ; d1 + d2
///     cmp r15, 1; jne 3f
///     imul rdx, rax, 233; shr rdx, 11; cmp rcx, rdx; jae fail
/// 3:  mov rdx, rax; shr rdx, 11; cmp rcx, rdx; jae 4f
///     in al, 0x42; in al, 0x42; mov bl, 0xfe; sub bl, r15b; cmp al, bl; jne fail
///     hlt
/// 4:  inc r15; jmp next
/// fail: xor r12, r12; xor r15, r15; hlt
/// expect: xor r10, r10; xor r8, r8; xor rsi, rsi ; count, tsc, prev_tsc
/// 5:  cmp r10, 50000; jae 6f
///     in al, 0x42; in al, 0x42; cmp al, bl; jne 6f
///     mov rsi, r8; rdtsc; shl rdx, 32; or rax, rdx; mov r8, rax; inc r10; jmp 5b
/// 6:  rdtsc; shl rdx, 32; or rax, rdx; sub rax, rsi; mov r9, rax; ret
/// ```
const QUICK_PIT_CALIBRATE: &[u8] = &[
    0xe4, 0x61, 0x24, 0xfd, 0x0c, 0x01, 0xe6, 0x61, 0xb0, 0xb0, 0xe6, 0x43, 0xb0, 0xff, 0xe6, 0x42,
    0xe6, 0x42, 0xe4, 0x42, 0xe4, 0x42, 0xb3, 0xff, 0xe8, 0x75, 0x00, 0x00, 0x00, 0x49, 0x83, 0xfa,
    0x05, 0x76, 0x68, 0x4d, 0x89, 0xc5, 0x4d, 0x89, 0xce, 0x49, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00,
    0x49, 0x81, 0xff, 0xe9, 0x00, 0x00, 0x00, 0x77, 0x52, 0xb3, 0xff, 0x44, 0x28, 0xfb, 0xe8, 0x4f,
    0x00, 0x00, 0x00, 0x49, 0x83, 0xfa, 0x05, 0x76, 0x42, 0x4c, 0x89, 0xc0, 0x4c, 0x29, 0xe8, 0x49,
    0x89, 0xc4, 0x4b, 0x8d, 0x0c, 0x0e, 0x49, 0x83, 0xff, 0x01, 0x75, 0x10, 0x48, 0x69, 0xd0, 0xe9,
    0x00, 0x00, 0x00, 0x48, 0xc1, 0xea, 0x0b, 0x48, 0x39, 0xd1, 0x73, 0x1f, 0x48, 0x89, 0xc2, 0x48,
    0xc1, 0xea, 0x0b, 0x48, 0x39, 0xd1, 0x73, 0x0e, 0xe4, 0x42, 0xe4, 0x42, 0xb3, 0xfe, 0x44, 0x28,
    0xfb, 0x38, 0xd8, 0x75, 0x06, 0xf4, 0x49, 0xff, 0xc7, 0xeb, 0xa5, 0x4d, 0x31, 0xe4, 0x4d, 0x31,
    0xff, 0xf4, 0x4d, 0x31, 0xd2, 0x4d, 0x31, 0xc0, 0x48, 0x31, 0xf6, 0x49, 0x81, 0xfa, 0x50, 0xc3,
    0x00, 0x00, 0x73, 0x1c, 0xe4, 0x42, 0xe4, 0x42, 0x38, 0xd8, 0x75, 0x14, 0x4c, 0x89, 0xc6, 0x0f,
    0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x49, 0x89, 0xc0, 0x49, 0xff, 0xc2, 0xeb, 0xdb,
    0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x48, 0x29, 0xf0, 0x49, 0x89, 0xc1, 0xc3,
];

/// How the guest's clock runs.
#[derive(Clone, Copy)]
enum Clock {
    /// Instruction count, as the test kit and `--verify` run.
    Virtual,
    /// A host clock that advances `step_ns` per read, as a busy vCPU thread sees it. With a
    /// stall, the first read at or after `at_ns` jumps `by_ns` further: the thread lost the CPU.
    Host {
        step_ns: u64,
        stall: Option<(u64, u64)>,
    },
}

/// Host time per clock read: a poll (two or three reads) then costs a few hundred nanoseconds, about
/// what it costs on the JIT.
const HOST_STEP_NS: u64 = 100;
/// A host time slice lost 1 ms into the measurement: the length of a scheduler tick.
const STALL: Option<(u64, u64)> = Some((1_000_000, 2_000_000));

fn harness(code: &[u8], clock: Clock, jit: bool) -> Harness {
    let mut harness = Harness::new(code);
    if jit {
        harness.enable_jit();
    }
    if let Clock::Host { step_ns, stall } = clock {
        let now = Arc::new(AtomicU64::new(0));
        let stalled = Arc::new(AtomicU64::new(0));
        harness.cpu.wall_clock = Some(WallClock::new(Arc::new(move || {
            let mut time = now.fetch_add(step_ns, Ordering::Relaxed) + step_ns;
            if let Some((at_ns, by_ns)) = stall
                && time >= at_ns
                && stalled.swap(1, Ordering::Relaxed) == 0
            {
                time = now.fetch_add(by_ns, Ordering::Relaxed) + by_ns;
            }
            time
        })));
    }
    harness
}

/// `pit_calibrate_tsc`'s answer in kHz, or `None` where Linux returns ULONG_MAX.
fn pit_calibrate_tsc(clock: Clock, jit: bool) -> Option<u64> {
    let mut h = harness(PIT_CALIBRATE_TSC, clock, jit);
    assert_eq!(h.run(INSTRUCTION_BUDGET), Stop::Halted);
    let [t1, t2, polls, longest, shortest] = [R8, R9, R10, R11, R12].map(|r| h.cpu.gpr[r]);
    (polls >= CAL_PIT_LOOPS && longest <= SMI_RATIO * shortest).then(|| (t2 - t1) / CAL_MS)
}

/// `quick_pit_calibrate`'s answer in kHz, or `None` where Linux prints "Fast TSC calibration
/// failed".
fn quick_pit_calibrate(clock: Clock, jit: bool) -> Option<u64> {
    let mut h = harness(QUICK_PIT_CALIBRATE, clock, jit);
    assert_eq!(h.run(INSTRUCTION_BUDGET), Stop::Halted);
    let (delta, steps) = (h.cpu.gpr[R12], h.cpu.gpr[R15]);
    (steps != 0).then(|| delta * PIT_TICK_RATE / (steps * PIT_TICKS_PER_MSB * HZ_PER_KHZ))
}

fn assert_close(khz: Option<u64>, tolerance: u64, what: &str) {
    let khz = khz.unwrap_or_else(|| panic!("{what} failed"));
    assert!(
        khz.abs_diff(TSC_KHZ) <= tolerance,
        "{what}: {khz} kHz, expected {TSC_KHZ}"
    );
}

#[test]
fn pit_calibrate_tsc_measures_the_tsc_on_the_virtual_clock() {
    for jit in [false, true] {
        assert_close(
            pit_calibrate_tsc(Clock::Virtual, jit),
            SLOW_TOLERANCE_KHZ,
            "pit_calibrate_tsc",
        );
    }
}

#[test]
fn quick_pit_calibrate_measures_the_tsc_on_the_virtual_clock() {
    for jit in [false, true] {
        assert_close(
            quick_pit_calibrate(Clock::Virtual, jit),
            QUICK_TOLERANCE_KHZ,
            "quick_pit_calibrate",
        );
    }
}

#[test]
fn both_calibrations_pass_on_a_steady_host_clock() {
    let clock = Clock::Host {
        step_ns: HOST_STEP_NS,
        stall: None,
    };
    assert_close(
        pit_calibrate_tsc(clock, false),
        SLOW_TOLERANCE_KHZ,
        "pit_calibrate_tsc",
    );
    assert_close(
        quick_pit_calibrate(clock, false),
        QUICK_TOLERANCE_KHZ,
        "quick_pit_calibrate",
    );
}

#[test]
fn a_host_stall_during_the_measurement_fails_both_calibrations_as_an_smi_would() {
    let clock = Clock::Host {
        step_ns: HOST_STEP_NS,
        stall: STALL,
    };
    assert_eq!(pit_calibrate_tsc(clock, false), None);
    assert_eq!(quick_pit_calibrate(clock, false), None);
}
