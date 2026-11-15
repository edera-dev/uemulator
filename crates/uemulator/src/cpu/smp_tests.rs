//! Several vCPUs running hand-assembled code on one machine (`testkit::run_smp`): start-up by
//! INIT and SIPI, IPIs, wake-ups from halt and MWAIT, and what must stay coherent between vCPUs
//! (translated code, soft TLBs). Every test runs on the interpreter and on the JIT.

use super::smp::Smp;
use super::*;
use crate::testkit::{Harness, SmpRun, ThreadRun, run_smp, run_threads, run_threads_with};

/// SIPI vector 8: the start-up code runs in real mode at 0x8000.
const TRAMPOLINE: u64 = 0x8000;
const SIPI_VECTOR: u32 = 8;
/// Where the vCPUs leave notes for each other.
const FLAG: u64 = 0x1000;
const FLAG_DONE: u8 = 0x42;
const COUNTER: u64 = 0x1010;
const HANDLER_NOTE: u64 = 0x1002;
const MWAIT_WOKEN_NOTE: u64 = 0x1008;
const SMC_BEFORE: u64 = 0x1020;
const SMC_AFTER: u64 = 0x1024;
const APIC_ID_NOTES: u64 = 0x1040;
const X2APIC_ID_NOTES: u64 = 0x1050;
const CODE_PAGE: u64 = 0x3000;
const CODE_PAGE_DATA: u64 = 0x3800;

/// ICR low words: assert level, delivery mode INIT (5) or start-up (6); the shorthand sits in
/// bits 19:18 (3 = all excluding self).
const ICR_INIT: u32 = 0x4500;
const ICR_STARTUP: u32 = 0x4600 | SIPI_VECTOR;
const ICR_ALL_EXCLUDING_SELF: u32 = 3 << 18;
const IPI_VECTOR: u32 = 0x40;

// mov ecx, 0x1b; mov eax, 0xfee00d00; xor edx, edx; wrmsr; mov ecx, 0x830
// (IA32_APIC_BASE: x2APIC on; ecx stays the ICR's MSR for the writes after it)
const ENABLE_X2APIC: [u8; 19] = [
    0xb9, 0x1b, 0x00, 0x00, 0x00, 0xb8, 0x00, 0x0d, 0xe0, 0xfe, 0x31, 0xd2, 0x0f, 0x30, 0xb9, 0x30,
    0x08, 0x00, 0x00,
];
// wait: pause; cmp byte [FLAG], 0x42; jne wait
const WAIT_FLAG_PAUSE: [u8; 12] = [
    0xf3, 0x90, 0x80, 0x3c, 0x25, 0x00, 0x10, 0x00, 0x00, 0x42, 0x75, 0xf4,
];
// wait: pause; cmp byte [HANDLER_NOTE], 0x77; jne wait
const WAIT_HANDLER_NOTE: [u8; 12] = [
    0xf3, 0x90, 0x80, 0x3c, 0x25, 0x02, 0x10, 0x00, 0x00, 0x77, 0x75, 0xf4,
];
// wait: pause; cmp word [FLAG], 3; jne wait
const WAIT_THREE_ARRIVALS: [u8; 13] = [
    0xf3, 0x90, 0x66, 0x83, 0x3c, 0x25, 0x00, 0x10, 0x00, 0x00, 0x03, 0x75, 0xf3,
];
// cli; hlt
const CLI_HLT: [u8; 2] = [0xfa, 0xf4];

// mov ecx, 100000; spin: dec ecx; jnz spin
const BSP_BUSY_LOOP: [u8; 9] = [0xb9, 0xa0, 0x86, 0x01, 0x00, 0xff, 0xc9, 0x75, 0xfc];

// inc word [COUNTER]; mov byte [FLAG], 0x42; hlt
const AP_START: [u8; 10] = [0xff, 0x06, 0x10, 0x10, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4];
// lock inc word [FLAG]; hlt
const AP_ARRIVE: [u8; 6] = [0xf0, 0xff, 0x06, 0x00, 0x10, 0xf4];
// mov ecx, 100000; spin: dec ecx; jnz spin; mov byte [FLAG], 0x42; hlt
const AP_BUSY: [u8; 16] = [
    0x66, 0xb9, 0xa0, 0x86, 0x01, 0x00, 0x66, 0x49, 0x75, 0xfc, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4,
];
// Software-enables the x2APIC (MSR 0x80f), points IVT[0x40] at the handler, notes it is ready
// and halts with interrupts on; the handler leaves a note and returns to a `cli; hlt`.
//   jmp start; handler: mov byte [HANDLER_NOTE], 0x77; iret
//   start: xor ax, ax; mov ds, ax; mov word [0x100], 2; mov word [0x102], 0x800
//   (enable x2APIC; mov ecx, 0x80f; mov eax, 0x1ff; xor edx, edx; wrmsr)
//   mov byte [FLAG], 0x42; sti; hlt; cli; hlt
const AP_WAIT_FOR_IPI: [u8; 67] = [
    0xeb, 0x06, 0xc6, 0x06, 0x02, 0x10, 0x77, 0xcf, 0x31, 0xc0, 0x8e, 0xd8, 0xc7, 0x06, 0x00, 0x01,
    0x02, 0x00, 0xc7, 0x06, 0x02, 0x01, 0x00, 0x08, 0x66, 0xb9, 0x1b, 0x00, 0x00, 0x00, 0x66, 0xb8,
    0x00, 0x0d, 0xe0, 0xfe, 0x66, 0x31, 0xd2, 0x0f, 0x30, 0x66, 0xb9, 0x0f, 0x08, 0x00, 0x00, 0x66,
    0xb8, 0xff, 0x01, 0x00, 0x00, 0x66, 0x31, 0xd2, 0x0f, 0x30, 0xc6, 0x06, 0x00, 0x10, 0x42, 0xfb,
    0xf4, 0xfa, 0xf4,
];
// The store lands between MONITOR and MWAIT: arms the monitor, only then tells the AP to go (a
// store before MONITOR is not seen by it, on hardware either), waits until the AP's store is
// visible, and executes MWAIT, which must return at once.
//   lea rax, [0x2000]; xor ecx, ecx; xor edx, edx; monitor; mov byte [0x1004], 1;
//   wait: pause; cmp byte [0x2000], 1; jne wait; sti; mwait; cli; mov byte [0x1008], 0x99; hlt
const BSP_MONITOR_THEN_MWAIT_AFTER_THE_STORE: [u8; 50] = [
    0x48, 0x8d, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00, 0x31, 0xc9, 0x31, 0xd2, 0x0f, 0x01, 0xc8, 0xc6,
    0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0xf3, 0x90, 0x80, 0x3c, 0x25, 0x00, 0x20, 0x00, 0x00,
    0x01, 0x75, 0xf4, 0xfb, 0x0f, 0x01, 0xc9, 0xfa, 0xc6, 0x04, 0x25, 0x08, 0x10, 0x00, 0x00, 0x99,
    0xf4, 0xf4,
];
// lea rax, [0x2000]; xor ecx, ecx; xor edx, edx; monitor; mov byte [0x1004], 1; sti; mwait;
// cli; mov byte [MWAIT_WOKEN_NOTE], 0x99; hlt
// The AP is told to store only after the monitor is armed: a store before MONITOR is not seen.
const BSP_MWAIT: [u8; 37] = [
    0x48, 0x8d, 0x04, 0x25, 0x00, 0x20, 0x00, 0x00, 0x31, 0xc9, 0x31, 0xd2, 0x0f, 0x01, 0xc8, 0xc6,
    0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0xfb, 0x0f, 0x01, 0xc9, 0xfa, 0xc6, 0x04, 0x25, 0x08,
    0x10, 0x00, 0x00, 0x99, 0xf4,
];
// wait: pause; cmp byte [0x1004], 1; jne wait; mov byte [0x2000], 1; hlt
const AP_STORE_TO_MONITORED_LINE: [u8; 15] = [
    0xf3, 0x90, 0x80, 0x3e, 0x04, 0x10, 0x01, 0x75, 0xf7, 0xc6, 0x06, 0x00, 0x20, 0x01, 0xf4,
];
// mov rax, CODE_PAGE; call rax; mov [SMC_BEFORE], eax; mov byte [0x1004], 1;
// wait: pause; cmp byte [0x1008], 1; jne wait;
// mov rax, CODE_PAGE; call rax; mov [SMC_AFTER], eax; cli; hlt
const BSP_CALL_PATCHED_CODE: [u8; 54] = [
    0x48, 0xc7, 0xc0, 0x00, 0x30, 0x00, 0x00, 0xff, 0xd0, 0x89, 0x04, 0x25, 0x20, 0x10, 0x00, 0x00,
    0xc6, 0x04, 0x25, 0x04, 0x10, 0x00, 0x00, 0x01, 0xf3, 0x90, 0x80, 0x3c, 0x25, 0x08, 0x10, 0x00,
    0x00, 0x01, 0x75, 0xf4, 0x48, 0xc7, 0xc0, 0x00, 0x30, 0x00, 0x00, 0xff, 0xd0, 0x89, 0x04, 0x25,
    0x24, 0x10, 0x00, 0x00, 0xfa, 0xf4,
];
// mov eax, 1; ret
const PATCHED_CODE: [u8; 6] = [0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3];
// wait: pause; cmp byte [0x1004], 1; jne wait; mov byte [0x3001], 2 (the immediate of the
// `mov eax, 1` above); mov byte [0x1008], 1; hlt
const AP_PATCH_CODE: [u8; 20] = [
    0xf3, 0x90, 0x80, 0x3e, 0x04, 0x10, 0x01, 0x75, 0xf7, 0xc6, 0x06, 0x01, 0x30, 0x02, 0xc6, 0x06,
    0x08, 0x10, 0x01, 0xf4,
];
// mov byte [CODE_PAGE_DATA], 1 / 2
const STORE_TO_CODE_PAGE_DATA_FIRST: [u8; 8] = [0xc6, 0x04, 0x25, 0x00, 0x38, 0x00, 0x00, 0x01];
const STORE_TO_CODE_PAGE_DATA_AGAIN: [u8; 8] = [0xc6, 0x04, 0x25, 0x00, 0x38, 0x00, 0x00, 0x02];
// ljmp 0:CODE_PAGE
const AP_JUMP_TO_CODE_PAGE: [u8; 5] = [0xea, 0x00, 0x30, 0x00, 0x00];
// mov byte [FLAG], 0x42; hlt (run by the AP from the code page)
const CODE_PAGE_FLAG_AND_HALT: [u8; 6] = [0xc6, 0x06, 0x00, 0x10, 0x42, 0xf4];
// mov eax, 1; cpuid; shr ebx, 24; and ebx, 0xff; mov esi, ebx; mov byte [si+0x1040], 0x7f;
// mov eax, 0xb; mov ecx, 1; cpuid; mov [si+0x1050], dl; hlt
const AP_REPORT_IDS: [u8; 46] = [
    0x66, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0x66, 0xc1, 0xeb, 0x18, 0x66, 0x81, 0xe3, 0xff,
    0x00, 0x00, 0x00, 0x66, 0x89, 0xde, 0xc6, 0x84, 0x40, 0x10, 0x7f, 0x66, 0xb8, 0x0b, 0x00, 0x00,
    0x00, 0x66, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0xa2, 0x88, 0x94, 0x50, 0x10, 0xf4,
];

/// `wrmsr` of the ICR (MSR 0x830, in ecx since `ENABLE_X2APIC`): destination in the high word.
fn send_icr(destination: u32, low: u32) -> Vec<u8> {
    let mut code = vec![0xba];
    code.extend_from_slice(&destination.to_le_bytes());
    code.push(0xb8);
    code.extend_from_slice(&low.to_le_bytes());
    code.extend_from_slice(&[0x0f, 0x30]);
    code
}

fn program(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn place(cpu: &mut Cpu, address: u64, bytes: &[u8]) {
    let start = address as usize;
    cpu.ram[start..start + bytes.len()].copy_from_slice(bytes);
}

fn both_engines(test: impl Fn(bool)) {
    test(false);
    test(true);
}

fn assert_shut_down(run: &SmpRun) {
    assert!(
        matches!(run.exit, ExitReason::Shutdown),
        "{:?}\n{}",
        run.exit,
        run.smp.summary()
    );
}

#[test]
fn a_secondary_vcpu_waits_for_a_startup_ipi_and_shares_the_ram() {
    let smp = Smp::new(
        Harness::with_devices(
            &[0xf4],
            crate::dev::DevConfig {
                cpus: 3,
                ..Default::default()
            },
        )
        .cpu,
    );
    let parked: Vec<_> = smp
        .cpus
        .iter()
        .map(|cpu| (cpu.cpu_index, cpu.halted, cpu.wait_for_sipi))
        .collect();
    assert_eq!(
        parked,
        [(0, false, false), (1, true, true), (2, true, true)]
    );
    assert_eq!(smp.cpus[1].ram.as_ptr(), smp.cpus[0].ram.as_ptr());
    assert_eq!(smp.cpus[1].ram.len(), smp.cpus[0].ram.len());
}

#[test]
fn an_ap_starts_in_real_mode_at_the_sipi_vector_and_a_second_sipi_is_ignored() {
    both_engines(|jit| {
        // The AP is halted by the time the second SIPI comes, but it is not waiting for one.
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &send_icr(1, ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_START));
        assert_shut_down(&run);
        assert_eq!(run.read(FLAG, 1), [FLAG_DONE]);
        assert_eq!(run.read_u16(COUNTER), 1, "the AP ran once");
    });
}

#[test]
fn a_halted_ap_wakes_for_a_fixed_ipi_and_takes_it_through_its_ivt() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &send_icr(1, IPI_VECTOR),
            &WAIT_HANDLER_NOTE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_WAIT_FOR_IPI));
        assert_shut_down(&run);
        assert_eq!(run.read(HANDLER_NOTE, 1), [0x77]);
    });
}

#[test]
fn a_shorthand_ipi_starts_every_other_vcpu() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &WAIT_THREE_ARRIVALS,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_ARRIVE));
        assert_shut_down(&run);
        assert_eq!(run.read_u16(FLAG), 3);
    });
}

#[test]
fn each_vcpu_sees_its_own_apic_id_in_cpuid() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_REPORT_IDS));
        assert_shut_down(&run);
        assert_eq!(run.read(APIC_ID_NOTES, 4), [0, 0x7f, 0x7f, 0x7f]);
        assert_eq!(run.read(X2APIC_ID_NOTES, 4), [0, 1, 2, 3]);
    });
}

#[test]
fn the_machine_keeps_running_while_the_bsp_is_halted_and_an_ap_works() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
        assert_shut_down(&run);
        assert_eq!(run.read(FLAG, 1), [FLAG_DONE], "the AP finished its loop");
        assert!(
            run.smp.active().icount > 2 * EVENT_INTERVAL,
            "the loop spans several quanta"
        );
    });
}

#[test]
fn pause_hands_the_cpu_to_the_vcpu_being_waited_for() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_START));
        assert_shut_down(&run);
        assert!(
            run.smp.active().icount < EVENT_INTERVAL / 2,
            "the spin loop burnt its quantum: {} instructions",
            run.smp.active().icount
        );
    });
}

#[test]
fn a_store_to_the_monitored_line_wakes_an_mwait() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MWAIT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE)
        });
        assert_shut_down(&run);
        assert_eq!(run.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

#[test]
fn code_another_vcpu_overwrites_is_translated_again() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_CALL_PATCHED_CODE,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_PATCH_CODE);
            place(cpu, CODE_PAGE, &PATCHED_CODE);
        });
        assert_shut_down(&run);
        assert_eq!(run.read_u32(SMC_BEFORE), 1);
        assert_eq!(run.read_u32(SMC_AFTER), 2, "a stale block ran");
    });
}

#[test]
fn a_vcpu_cannot_store_past_a_page_another_vcpu_made_code() {
    both_engines(|jit| {
        // The BSP's TLB gets a write tag for the code page's data half, then the AP runs code
        // from that page; the BSP's next store must still notice (`code_written`).
        let bsp = program(&[
            &ENABLE_X2APIC,
            &STORE_TO_CODE_PAGE_DATA_FIRST,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &STORE_TO_CODE_PAGE_DATA_AGAIN,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_JUMP_TO_CODE_PAGE);
            place(cpu, CODE_PAGE, &CODE_PAGE_FLAG_AND_HALT);
        });
        assert_shut_down(&run);
        assert_eq!(run.read(CODE_PAGE_DATA, 1), [2]);
        assert_eq!(
            run.smp.active().page_has_code[(CODE_PAGE >> PAGE_SHIFT) as usize],
            0,
            "the store skipped `code_written`"
        );
    });
}

#[test]
fn guest_time_advances_by_the_instructions_of_one_vcpu_when_two_run_side_by_side() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_BUSY_LOOP,
            &WAIT_FLAG_PAUSE,
            &CLI_HLT,
        ]);
        let run = run_smp(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
        assert_shut_down(&run);
        let (retired, now) = (run.smp.active().icount, run.smp.active().now_ns());
        // Both loops retire 200k instructions; they overlap almost completely.
        assert!(retired > 350_000, "{retired}");
        assert!(now < retired * 3 / 5, "{now} ns for {retired} instructions");
        assert!(now > retired * 2 / 5, "{now} ns for {retired} instructions");
    });
}

#[test]
fn guest_time_never_runs_backwards_across_switches() {
    // `now_ns` read after every slice of a busy two-vCPU run.
    let bsp = program(&[
        &ENABLE_X2APIC,
        &send_icr(1, ICR_INIT),
        &send_icr(1, ICR_STARTUP),
        &BSP_BUSY_LOOP,
        &WAIT_FLAG_PAUSE,
        &CLI_HLT,
    ]);
    let mut run = run_smp(&bsp, 2, false, |cpu| place(cpu, TRAMPOLINE, &AP_BUSY));
    assert_shut_down(&run);
    // Replay the folding the scheduler does with changing divisors.
    let cpu = run.smp.active_mut();
    let mut last = cpu.now_ns();
    for divisor in [2, 2, 1, 3, 3, 1, 2] {
        cpu.icount += 1000;
        let during = cpu.now_ns();
        assert!(during >= last);
        cpu.start_slice(divisor);
        assert!(cpu.now_ns() >= during);
        last = cpu.now_ns();
    }
}

// ---- one host thread per vCPU ------------------------------------------------------------

/// Where the threaded tests keep their counters, clear of the single-thread tests' notes.
const ATOMIC_COUNTER: u64 = 0x1100;
const LOCK_WORD: u64 = 0x1104;
const PROTECTED_COUNTER: u64 = 0x1108;
const DONE_COUNTER: u64 = 0x1110;
const MESSAGE_FLAG: u64 = 0x1240;
const MESSAGE_ERRORS: u64 = 0x1128;

const ATOMIC_INCREMENTS_PER_CPU: u32 = 20_000;
const LOCKED_SECTIONS_PER_CPU: u32 = 3_000;
const MESSAGES: u32 = 5_000_000;
const AP_COUNT: u32 = 3;

// Real mode (DS = 0), run by each AP:
//   mov ecx, 20000; l: lock inc dword [ATOMIC_COUNTER]; dec ecx; jnz l; lock inc dword [DONE]; hlt
const AP_LOCKED_INCREMENTS: [u8; 23] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0xff, 0x06, 0x00, 0x11, 0x66, 0x49, 0x75, 0xf6,
    0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// A test-and-set lock built on `lock cmpxchg`, released with a plain store, guarding a counter
// that is updated with a plain load, add and store; 3000 rounds, then done.
const AP_CMPXCHG_LOCK: [u8; 58] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0x31, 0xc0, 0x66, 0xba, 0x01, 0x00, 0x00,
    0x00, 0x66, 0xf0, 0x0f, 0xb1, 0x16, 0x04, 0x11, 0x75, 0xec, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83,
    0xc0, 0x01, 0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7, 0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66,
    0x49, 0x75, 0xd3, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// The same with `xchg` (implicitly locked, no prefix) as the lock.
const AP_XCHG_LOCK: [u8; 56] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0xba, 0x01, 0x00, 0x00, 0x00, 0x66, 0x87,
    0x16, 0x04, 0x11, 0x66, 0x85, 0xd2, 0x75, 0xee, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83, 0xc0, 0x01,
    0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7, 0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66, 0x49, 0x75,
    0xd5, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// Waits for the consumer, then for i in 1..=5000000 { MESSAGE_DATA = i; MESSAGE_FLAG = i } (two
// plain stores to different cache lines), then is done.
const AP_MESSAGE_PRODUCER: [u8; 46] = [
    0xf3, 0x90, 0x66, 0x83, 0x3e, 0x80, 0x12, 0x01, 0x75, 0xf6, 0x66, 0xb9, 0x01, 0x00, 0x00, 0x00,
    0x66, 0x89, 0x0e, 0x00, 0x12, 0x66, 0x89, 0x0e, 0x40, 0x12, 0x66, 0x83, 0xc1, 0x01, 0x66, 0x81,
    0xf9, 0x41, 0x4b, 0x4c, 0x00, 0x72, 0xe9, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
// 64-bit, run by the BSP: 20000 `lock inc`s, then waits for three APs to be done; cli; hlt
const BSP_LOCKED_INCREMENTS_AND_WAIT: [u8; 31] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x00, 0x11, 0x00, 0x00, 0xff, 0xc9, 0x75,
    0xf4, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// The BSP takes the same lock 3000 times, then waits for three APs.
const BSP_CMPXCHG_LOCK_AND_WAIT: [u8; 71] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x31, 0xc0, 0xba, 0x01, 0x00, 0x00, 0x00, 0xf0, 0x0f,
    0xb1, 0x14, 0x25, 0x04, 0x11, 0x00, 0x00, 0x75, 0xec, 0x8b, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00,
    0x83, 0xc0, 0x01, 0x89, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00, 0xc7, 0x04, 0x25, 0x04, 0x11, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xcc, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11,
    0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// 64-bit, run by the BSP: says it is ready, then reads MESSAGE_FLAG and then MESSAGE_DATA over
// and over, counting the rounds where the data is older than the flag, until the flag says
// 5000000; stores the count and waits for the producer to be done.
const BSP_MESSAGE_CONSUMER_AND_WAIT: [u8; 65] = [
    0xc7, 0x04, 0x25, 0x80, 0x12, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xbe, 0x00, 0x00, 0x00, 0x00,
    0x8b, 0x04, 0x25, 0x40, 0x12, 0x00, 0x00, 0x8b, 0x1c, 0x25, 0x00, 0x12, 0x00, 0x00, 0x39, 0xc3,
    0x73, 0x03, 0x83, 0xc6, 0x01, 0x3d, 0x40, 0x4b, 0x4c, 0x00, 0x72, 0xe4, 0x89, 0x34, 0x25, 0x28,
    0x11, 0x00, 0x00, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x01, 0x75, 0xf4, 0xfa,
    0xf4,
];

/// Starts every other vCPU at `TRAMPOLINE` and then runs `work`.
fn start_the_others_then(work: &[u8]) -> Vec<u8> {
    program(&[
        &ENABLE_X2APIC,
        &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
        &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
        work,
    ])
}

fn assert_thread_shutdown(run: &ThreadRun) {
    assert!(
        matches!(run.exit, ExitReason::Shutdown),
        "{:?}\n{}",
        run.exit,
        run.summary
    );
}

#[test]
fn every_vcpu_on_its_own_thread_reports_in_and_the_machine_shuts_down() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&program(&[&WAIT_THREE_ARRIVALS, &CLI_HLT]));
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_ARRIVE));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(FLAG, 2), [3, 0]);
    });
}

#[test]
fn locked_increments_from_four_threads_lose_no_update() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_INCREMENTS_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_INCREMENTS)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(ATOMIC_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_cmpxchg_lock_released_by_a_plain_store_keeps_a_plain_counter_exact() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_CMPXCHG_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_CMPXCHG_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
        assert_eq!(run.read_u32(LOCK_WORD), 0);
    });
}

#[test]
fn xchg_with_memory_is_locked_without_a_prefix() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_CMPXCHG_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_XCHG_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
    });
}

#[test]
fn data_written_before_a_flag_is_never_read_stale_after_the_flag() {
    // Message passing: the producer's two plain stores must become visible in order, and the
    // consumer's two plain loads must not be reordered. The consumer is the BSP, whose JIT code
    // is what fails on an aarch64 host with plain `ldr`; Rosetta runs x86 with hardware ordering
    // and cannot catch it.
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_MESSAGE_CONSUMER_AND_WAIT);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_MESSAGE_PRODUCER)
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(MESSAGE_FLAG), MESSAGES);
        assert_eq!(run.read_u32(MESSAGE_ERRORS), 0);
    });
}

// A serializing `cpuid` after the flag: what a guest does between writing code on one vCPU and
// running it on another (the rest of the protocol is the IPI that Linux's `text_poke` sends).
// xor eax, eax; cpuid
const SERIALIZE: [u8; 4] = [0x31, 0xc0, 0x0f, 0xa2];

/// `BSP_CALL_PATCHED_CODE` with `SERIALIZE` between the wait and the second call.
fn bsp_call_patched_code_serialized() -> Vec<u8> {
    let mut code = BSP_CALL_PATCHED_CODE.to_vec();
    const AFTER_WAIT: usize = 36;
    code.splice(AFTER_WAIT..AFTER_WAIT, SERIALIZE);
    code
}

#[test]
fn a_halted_ap_on_its_own_thread_takes_an_ipi_through_its_ivt() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &send_icr(1, IPI_VECTOR),
            &WAIT_HANDLER_NOTE,
            &CLI_HLT,
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| place(cpu, TRAMPOLINE, &AP_WAIT_FOR_IPI));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(HANDLER_NOTE, 1), [0x77]);
    });
}

#[test]
fn each_threaded_vcpu_sees_its_own_apic_id_in_cpuid() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_INIT),
            &send_icr(0, ICR_ALL_EXCLUDING_SELF | ICR_STARTUP),
            &CLI_HLT,
        ]);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_REPORT_IDS));
        assert_thread_shutdown(&run);
        assert_eq!(run.read(APIC_ID_NOTES, 4), [0, 0x7f, 0x7f, 0x7f]);
        assert_eq!(run.read(X2APIC_ID_NOTES, 4), [0, 1, 2, 3]);
    });
}

#[test]
fn a_store_to_the_monitored_line_wakes_an_mwait_on_another_thread() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MWAIT,
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE)
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

#[test]
fn code_another_thread_overwrites_is_translated_again() {
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &bsp_call_patched_code_serialized(),
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_PATCH_CODE);
            place(cpu, CODE_PAGE, &PATCHED_CODE);
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(SMC_BEFORE), 1);
        assert_eq!(run.read_u32(SMC_AFTER), 2, "a stale block ran");
    });
}

#[test]
fn a_thread_cannot_store_past_a_page_another_thread_made_code() {
    both_engines(|jit| {
        // The BSP's TLB gets a write tag for the code page's data half, then the AP runs code
        // from that page: flagging the page revokes the BSP's tag from the AP's thread, so the
        // BSP's next store still reaches `code_written`.
        let bsp = program(&[
            &ENABLE_X2APIC,
            &STORE_TO_CODE_PAGE_DATA_FIRST,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &WAIT_FLAG_PAUSE,
            &STORE_TO_CODE_PAGE_DATA_AGAIN,
            &CLI_HLT,
        ]);
        let run = run_threads(&bsp, 2, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_JUMP_TO_CODE_PAGE);
            place(cpu, CODE_PAGE, &CODE_PAGE_FLAG_AND_HALT);
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read(CODE_PAGE_DATA, 1), [2]);
        assert_eq!(
            run.bsp.page_has_code[(CODE_PAGE >> PAGE_SHIFT) as usize],
            0,
            "the store skipped `code_written`"
        );
    });
}

/// Real-mode pages that each jump to the next: the first fetch from a page flags it as code,
/// and four vCPUs do that to the same pages at the same time.
const CHAIN_BASE: u64 = 0x20000;
const CHAIN_PAGES: u64 = 64;
const PAGE_BYTES: u64 = 0x1000;
// wait: pause; cmp dword [DONE], 3; jne wait; cli; hlt
const BSP_WAIT_FOR_THREE: [u8; 17] = [
    0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4, 0xf4, 0xf4,
    0xf4,
];
// lock inc dword [DONE]; hlt
const CHAIN_END: [u8; 7] = [0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4];

fn jump_far(segment: u16) -> [u8; 5] {
    let [low, high] = segment.to_le_bytes();
    [0xea, 0x00, 0x00, low, high]
}

#[test]
fn threads_flagging_the_same_pages_at_once_neither_deadlock_nor_corrupt_each_other() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_WAIT_FOR_THREE);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &jump_far((CHAIN_BASE >> 4) as u16));
            for page in 0..CHAIN_PAGES {
                let at = CHAIN_BASE + page * PAGE_BYTES;
                if page + 1 < CHAIN_PAGES {
                    place(cpu, at, &jump_far(((at + PAGE_BYTES) >> 4) as u16));
                } else {
                    place(cpu, at, &CHAIN_END);
                }
            }
        });
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(DONE_COUNTER), AP_COUNT);
        for page in 0..CHAIN_PAGES {
            let flag = run.bsp.page_has_code[((CHAIN_BASE >> 12) + page) as usize];
            assert_eq!(flag, 1, "page {page} was executed but not flagged");
        }
    });
}

#[test]
fn a_store_between_monitor_and_mwait_makes_mwait_return_at_once() {
    // The line is armed by MONITOR, not by MWAIT: with threads the other vCPU's store can land
    // in between, and a vCPU that snapshotted at MWAIT would sleep through it forever.
    both_engines(|jit| {
        let bsp = program(&[
            &ENABLE_X2APIC,
            &send_icr(1, ICR_INIT),
            &send_icr(1, ICR_STARTUP),
            &BSP_MONITOR_THEN_MWAIT_AFTER_THE_STORE,
        ]);
        let place_ap = |cpu: &mut Cpu| place(cpu, TRAMPOLINE, &AP_STORE_TO_MONITORED_LINE);
        let timeshared = run_smp(&bsp, 2, jit, place_ap);
        assert_shut_down(&timeshared);
        assert_eq!(timeshared.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
        let threaded = run_threads(&bsp, 2, jit, place_ap);
        assert_thread_shutdown(&threaded);
        assert_eq!(threaded.read(MWAIT_WOKEN_NOTE, 1), [0x99]);
    });
}

/// A dword that is misaligned but inside one page, and one that straddles two pages: no host
/// atomic covers either, so the locked instruction runs with the other vCPUs stopped.
const MISALIGNED_COUNTER: u64 = 0x1101;
const STRADDLING_COUNTER: u64 = 0x1ffe;

// lock inc dword [MISALIGNED_COUNTER] 20000 times, then done
const AP_LOCKED_MISALIGNED: [u8; 23] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0xff, 0x06, 0x01, 0x11, 0x66, 0x49, 0x75, 0xf6,
    0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
const BSP_LOCKED_MISALIGNED_AND_WAIT: [u8; 31] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x01, 0x11, 0x00, 0x00, 0xff, 0xc9, 0x75,
    0xf4, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// lock add dword [STRADDLING_COUNTER], 1 20000 times, then done
const AP_LOCKED_STRADDLING: [u8; 24] = [
    0x66, 0xb9, 0x20, 0x4e, 0x00, 0x00, 0x66, 0xf0, 0x83, 0x06, 0xfe, 0x1f, 0x01, 0x66, 0x49, 0x75,
    0xf5, 0x66, 0xf0, 0xff, 0x06, 0x10, 0x11, 0xf4,
];
const BSP_LOCKED_STRADDLING_AND_WAIT: [u8; 32] = [
    0xb9, 0x20, 0x4e, 0x00, 0x00, 0xf0, 0x83, 0x04, 0x25, 0xfe, 0x1f, 0x00, 0x00, 0x01, 0xff, 0xc9,
    0x75, 0xf3, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa, 0xf4,
];
// A spin lock taken with `lock bts [LOCK_WORD], 0` and released by a plain store, guarding a plain
// counter, 3000 rounds each.
const AP_BTS_LOCK: [u8; 50] = [
    0x66, 0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0x66, 0xf0, 0x0f, 0xba, 0x2e, 0x04, 0x11, 0x00,
    0x72, 0xf4, 0x66, 0xa1, 0x08, 0x11, 0x66, 0x83, 0xc0, 0x01, 0x66, 0xa3, 0x08, 0x11, 0x66, 0xc7,
    0x06, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x66, 0x49, 0x75, 0xdb, 0x66, 0xf0, 0xff, 0x06, 0x10,
    0x11, 0xf4,
];
const BSP_BTS_LOCK_AND_WAIT: [u8; 66] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xf3, 0x90, 0xf0, 0x0f, 0xba, 0x2c, 0x25, 0x04, 0x11, 0x00, 0x00,
    0x00, 0x72, 0xf2, 0x8b, 0x04, 0x25, 0x08, 0x11, 0x00, 0x00, 0x83, 0xc0, 0x01, 0x89, 0x04, 0x25,
    0x08, 0x11, 0x00, 0x00, 0xc7, 0x04, 0x25, 0x04, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0xc9, 0x75, 0xd2, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x10, 0x11, 0x00, 0x00, 0x03, 0x75, 0xf4, 0xfa,
    0xf4, 0xf4,
];

#[test]
fn a_locked_increment_of_a_misaligned_operand_loses_no_update() {
    // Not one host atomic: the instruction runs with the other vCPUs stopped. Running that from
    // inside the exclusive section used to wait for itself forever.
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_MISALIGNED_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_MISALIGNED)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(MISALIGNED_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_locked_add_straddling_two_pages_loses_no_update() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_LOCKED_STRADDLING_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| {
            place(cpu, TRAMPOLINE, &AP_LOCKED_STRADDLING)
        });
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(STRADDLING_COUNTER),
            (AP_COUNT + 1) * ATOMIC_INCREMENTS_PER_CPU
        );
    });
}

#[test]
fn a_lock_bts_spin_lock_keeps_a_plain_counter_exact() {
    both_engines(|jit| {
        let bsp = start_the_others_then(&BSP_BTS_LOCK_AND_WAIT);
        let run = run_threads(&bsp, 4, jit, |cpu| place(cpu, TRAMPOLINE, &AP_BTS_LOCK));
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(PROTECTED_COUNTER),
            (AP_COUNT + 1) * LOCKED_SECTIONS_PER_CPU
        );
    });
}

/// Where the stale-translation test keeps its notes and code: RAM the fake kernel leaves alone.
const STALE_DATA: u64 = crate::testkit::SCRATCH;
const STALE_DONE: u64 = STALE_DATA;
const STALE_VERSION: u64 = STALE_DATA + 0x08;
const STALE_FAILURES: u64 = STALE_DATA + 0x10;
const STALE_EXITED: u64 = STALE_DATA + 0x18;
/// The version a worker last read before calling the site. Now and then the writer waits for it
/// before the next rewrite (`STALE_PACED_VERSIONS`), so it rewrites while a worker is translating
/// the site, however the host schedules the threads. Paced by a spin loop instead, the writer's whole run took a few
/// milliseconds of host time, and on a loaded host it was often over before a worker thread got
/// a core: nothing raced, and the test failed for not having tested.
const STALE_CALLING: u64 = STALE_DATA + 0x1c;
// The writer and the workers below have it hand-assembled as an absolute address.
const _: () = assert!(STALE_CALLING == 0x40_001c);
/// How often the writer waits for a worker (`STALE_CALLING`).
const STALE_PACED_VERSIONS: u32 = 32;
/// The page the writer rewrites and the workers run: `mov eax, <version>; ret`.
const STALE_SITE: u64 = STALE_DATA + 0x1000;
const STALE_WORKER_CODE: u64 = STALE_DATA + 0x2000;
const STALE_WORKER_STACKS: u64 = STALE_DATA + 0x10000;
const STALE_WORKER_STACK_BYTES: u64 = 0x1000;
const STALE_WORKERS: usize = 2;
/// How long a translation lingers between reading the bytes and registering the block: the
/// writer, which now and then rewrites as soon as a worker is about to call the site
/// (`STALE_CALLING`), has that long to land in the window.
const STALE_WINDOW_MICROS: u64 = 1000;
// mov eax, 0; ret
const STALE_SITE_CODE: [u8; 6] = [0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3];

// The writer: for version 1 to 1000, patch the immediate of the site (a store to a code page, so
// `code_written` runs) and then publish the version; on every `STALE_PACED_VERSIONS`th, wait for a
// worker to read it and head for the site, whose translation the next rewrite then races; at the
// end raise DONE and wait for the workers. Waiting on every version instead made each one cost a
// translation's exclusive section and pause, and a loaded host ran out the watchdog.
//   xor ebx, ebx
// again: inc ebx; mov [site + 1], ebx; mov [VERSION], ebx; test bl, 31; jnz next
// called: pause; cmp [CALLING], ebx; jb called
// next: cmp ebx, 1000; jb again
//   mov dword [DONE], 1; wait: pause; cmp dword [EXITED], 2; jne wait; cli; hlt
const STALE_WRITER: [u8; 67] = [
    0x31, 0xdb, 0xff, 0xc3, 0x89, 0x1c, 0x25, 0x04, 0x10, 0x40, 0x00, 0x89, 0x1c, 0x25, 0x08, 0x00,
    0x40, 0x00, 0xf6, 0xc3, 0x1f, 0x75, 0x0b, 0xf3, 0x90, 0x39, 0x1c, 0x25, 0x1c, 0x00, 0x40, 0x00,
    0x72, 0xf5, 0x81, 0xfb, 0xe8, 0x03, 0x00, 0x00, 0x72, 0xd8, 0xc7, 0x04, 0x25, 0x00, 0x00, 0x40,
    0x00, 0x01, 0x00, 0x00, 0x00, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x18, 0x00, 0x40, 0x00, 0x02, 0x75,
    0xf4, 0xfa, 0xf4,
];
// The writer's `test bl, 31`.
const _: () = assert!(STALE_WRITER[20] as u32 == STALE_PACED_VERSIONS - 1);

// A worker, the cross-modifying code protocol of SDM vol. 3, 8.1.3: read the version the writer
// published, run a serializing `cpuid`, then run the site, which must be at least that new.
// The store to CALLING tells the writer it may rewrite.
// work: mov r8d, [VERSION]; xor eax, eax; cpuid; mov [CALLING], r8d; mov eax, site; call rax
//   cmp eax, r8d; jae ok; lock inc qword [FAILURES]
// ok: cmp dword [DONE], 0; je work; lock inc dword [EXITED]; cli; hlt
const STALE_WORKER: [u8; 61] = [
    0x44, 0x8b, 0x04, 0x25, 0x08, 0x00, 0x40, 0x00, 0x31, 0xc0, 0x0f, 0xa2, 0x44, 0x89, 0x04, 0x25,
    0x1c, 0x00, 0x40, 0x00, 0xb8, 0x03, 0x10, 0x40, 0x00, 0xff, 0xd0, 0x44, 0x39, 0xc0, 0x73, 0x09,
    0xf0, 0x48, 0xff, 0x04, 0x25, 0x10, 0x00, 0x40, 0x00, 0x83, 0x3c, 0x25, 0x00, 0x00, 0x40, 0x00,
    0x00, 0x74, 0xcd, 0xf0, 0xff, 0x04, 0x25, 0x18, 0x00, 0x40, 0x00, 0xfa, 0xf4,
];

/// Starts a parked secondary in the 64-bit mode the bootstrap processor runs in, at `rip`.
fn start_in_long_mode(cpu: &mut Cpu, bsp: &Cpu, rip: u64, rsp: u64) {
    cpu.cr0 = bsp.cr0;
    cpu.cr3 = bsp.cr3;
    cpu.cr4 = bsp.cr4;
    cpu.efer = bsp.efer;
    cpu.seg = bsp.seg;
    cpu.gdtr = bsp.gdtr;
    cpu.idtr = bsp.idtr;
    cpu.eflags = bsp.eflags;
    cpu.set_cpl(0);
    cpu.flush_tlb();
    cpu.rip = rip;
    cpu.gpr[crate::testkit::RSP] = rsp;
    cpu.halted = false;
    cpu.wait_for_sipi = false;
}

/// One vCPU rewrites code while the others run it. Another vCPU's translation of that page can
/// read the old bytes, lose the race with the writer's `code_written`, and register its block
/// after the writer dropped the page's blocks: stale code that stays valid until the next write.
/// Every translation is held for longer than the writer waits between rewrites, so the race is
/// lost every time; the cache has to notice (`Jit::install` compares `invalidations`) and decode
/// again, and no worker may then run an older immediate than the version it saw published.
#[test]
fn code_rewritten_while_another_vcpu_translates_it_never_runs_stale() {
    both_engines(|jit| {
        let run = run_threads_with(
            &STALE_WRITER,
            1 + STALE_WORKERS,
            jit,
            |cpu| {
                place(cpu, STALE_SITE + 3, &STALE_SITE_CODE);
                place(cpu, STALE_WORKER_CODE, &STALE_WORKER);
                if let Some(jit) = &cpu.jit {
                    jit.set_install_pause_micros(STALE_WINDOW_MICROS);
                }
            },
            |cpus| {
                let (bsp, others) = cpus.split_first_mut().unwrap();
                for (index, worker) in others.iter_mut().enumerate() {
                    let stack = STALE_WORKER_STACKS + (index as u64 + 1) * STALE_WORKER_STACK_BYTES;
                    start_in_long_mode(worker, bsp, STALE_WORKER_CODE, stack);
                }
            },
        );
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(STALE_VERSION), 1000);
        assert_eq!(run.read_u32(STALE_DONE), 1, "the writer finished");
        assert_eq!(
            run.read_u32(STALE_EXITED),
            STALE_WORKERS as u32,
            "every worker left its loop"
        );
        assert_eq!(
            run.read(STALE_FAILURES, 8),
            [0; 8],
            "a worker ran code older than the version it saw published (jit {jit})"
        );
        if let Some(cache) = run.bsp.jit.as_ref() {
            assert!(
                cache.stats().stale_decodes > 0,
                "no translation lost the race: the test did not test"
            );
        }
    });
}

const FLUSH_COUNTER: u64 = STALE_DATA + 0x20;
const FLUSH_ITERATIONS: u32 = 3000;
/// Blocks the cache may hold in `a_full_code_cache_is_flushed_under_vcpus_that_keep_running`:
/// fewer than the loop below needs, so the buffer is flushed every few iterations.
const FLUSH_BLOCK_LIMIT: usize = 3;

// Every vCPU runs this loop (placed at a page of its own): `jmp` chained blocks around a locked
// increment of a shared counter, so a flush lands under blocks that are linked to each other.
//   mov ecx, 3000; top: jmp a; a: lock inc dword [COUNTER]; jmp b; b: dec ecx; jnz top
// then the workers count themselves out and halt:
//   lock inc dword [EXITED]; cli; hlt
const FLUSH_WORKER: [u8; 31] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xeb, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x20, 0x00, 0x40, 0x00, 0xeb,
    0x00, 0xff, 0xc9, 0x75, 0xf0, 0xf0, 0xff, 0x04, 0x25, 0x18, 0x00, 0x40, 0x00, 0xfa, 0xf4,
];
// and the bootstrap processor waits for them instead:
//   wait: pause; cmp dword [EXITED], 2; jne wait; cli; hlt
const FLUSH_BSP: [u8; 35] = [
    0xb9, 0xb8, 0x0b, 0x00, 0x00, 0xeb, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x20, 0x00, 0x40, 0x00, 0xeb,
    0x00, 0xff, 0xc9, 0x75, 0xf0, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x18, 0x00, 0x40, 0x00, 0x02, 0x75,
    0xf4, 0xfa, 0xf4,
];

/// The cache fills up while every vCPU is running blocks that are linked to each other: the
/// flush has to stop them all outside generated code, throw the blocks and links away, make each
/// vCPU forget its jump cache, and leave every vCPU able to carry on. A flush that missed one of
/// those would run freed code, so nothing here may be lost, corrupted or stuck.
#[test]
fn a_full_code_cache_is_flushed_under_vcpus_that_keep_running() {
    both_engines(|jit| {
        let run = run_threads_with(
            &FLUSH_BSP,
            1 + STALE_WORKERS,
            jit,
            |cpu| {
                place(cpu, STALE_WORKER_CODE, &FLUSH_WORKER);
                if let Some(jit) = &cpu.jit {
                    jit.set_block_limit(FLUSH_BLOCK_LIMIT);
                }
            },
            |cpus| {
                let (bsp, others) = cpus.split_first_mut().unwrap();
                for (index, worker) in others.iter_mut().enumerate() {
                    let stack = STALE_WORKER_STACKS + (index as u64 + 1) * STALE_WORKER_STACK_BYTES;
                    start_in_long_mode(worker, bsp, STALE_WORKER_CODE, stack);
                }
            },
        );
        assert_thread_shutdown(&run);
        assert_eq!(
            run.read_u32(FLUSH_COUNTER),
            (1 + STALE_WORKERS as u32) * FLUSH_ITERATIONS,
            "an increment was lost"
        );
        if let Some(cache) = run.bsp.jit.as_ref() {
            assert!(cache.stats().resets > 0, "the cache was never flushed");
        }
    });
}

/// Two virtual addresses of the same physical pages. They differ in the page number's low bits as
/// well as high ones, so they do not evict each other from the soft TLB.
const ALIAS_ONE: u64 = 0x0000_2000_0000_0000;
const ALIAS_TWO: u64 = 0x0000_2100_0000_1000;
const IDENTITY_PAGES: u64 = 0x24;
/// Where `identity_paging` maps what the threaded JIT tests share.
const BSP_CODE_PAGE: u64 = crate::testkit::CODE_BASE & !0xfff;

/// Replaces the bootstrap processor's page tables with 4 KiB identity mappings of what these
/// tests use (its code and stack, the data pages from `STALE_DATA`), plus whatever `aliases`
/// maps: `(virtual page, physical page)`. The others copy the result (`start_in_long_mode`).
fn identity_paging(cpu: &mut Cpu, aliases: &[(u64, u64)]) {
    use super::test_support::{PTE_PRESENT_WRITABLE_USER, PageTables};
    let mut tables = PageTables::new(cpu);
    let flags = PTE_PRESENT_WRITABLE_USER;
    for page in [
        BSP_CODE_PAGE,
        BSP_CODE_PAGE + 0x1000,
        crate::testkit::STACK_TOP - 0x1000,
    ] {
        tables.map_4k(cpu, page, page, flags);
    }
    for index in 0..IDENTITY_PAGES {
        let page = STALE_DATA + index * 0x1000;
        tables.map_4k(cpu, page, page, flags);
    }
    for &(virtual_page, physical) in aliases {
        tables.map_4k(cpu, virtual_page, physical, flags);
    }
}

// The stale-translation worker (`STALE_WORKER`) with the site's address in rdi instead of a
// constant, so each worker can reach it through its own mapping:
// work: mov r8d, [VERSION]; xor eax, eax; cpuid; mov [CALLING], r8d; mov rax, rdi; call rax
//   cmp eax, r8d; jae ok; lock inc qword [FAILURES]
// ok: cmp dword [DONE], 0; je work; lock inc dword [EXITED]; cli; hlt
const ALIASED_WORKER: [u8; 59] = [
    0x44, 0x8b, 0x04, 0x25, 0x08, 0x00, 0x40, 0x00, 0x31, 0xc0, 0x0f, 0xa2, 0x44, 0x89, 0x04, 0x25,
    0x1c, 0x00, 0x40, 0x00, 0x48, 0x89, 0xf8, 0xff, 0xd0, 0x44, 0x39, 0xc0, 0x73, 0x09, 0xf0, 0x48,
    0xff, 0x04, 0x25, 0x10, 0x00, 0x40, 0x00, 0x83, 0x3c, 0x25, 0x00, 0x00, 0x40, 0x00, 0x00, 0x74,
    0xcf, 0xf0, 0xff, 0x04, 0x25, 0x18, 0x00, 0x40, 0x00, 0xfa, 0xf4,
];

// mov ecx, 5000000; l: dec ecx; jnz l. Ahead of a worker it stands in for a host that starts the
// worker threads late: milliseconds, longer than a writer paced by its own spin loop took for
// all of its rewrites, which then failed this test every time.
const STALE_LATE_START: [u8; 9] = [0xb9, 0x40, 0x4b, 0x4c, 0x00, 0xff, 0xc9, 0x75, 0xfc];

/// `STALE_WRITER` and `STALE_WORKER` again, but the workers reach the site through two different
/// virtual addresses while the writer rewrites it through a third (the identity mapping): a
/// rewrite must drop the one translation for every address, and no worker may then run an older
/// immediate than the version it saw published.
#[test]
fn code_rewritten_through_one_address_is_never_stale_through_the_aliases_of_its_page() {
    both_engines(|jit| {
        let aliases = [(ALIAS_ONE, STALE_SITE), (ALIAS_TWO, STALE_SITE)];
        let run = run_threads_with(
            &STALE_WRITER,
            1 + STALE_WORKERS,
            jit,
            |cpu| {
                place(cpu, STALE_SITE + 3, &STALE_SITE_CODE);
                place(
                    cpu,
                    STALE_WORKER_CODE,
                    &program(&[&STALE_LATE_START, &ALIASED_WORKER]),
                );
                identity_paging(cpu, &aliases);
                if let Some(jit) = &cpu.jit {
                    jit.set_install_pause_micros(STALE_WINDOW_MICROS);
                }
            },
            |cpus| {
                let (bsp, others) = cpus.split_first_mut().unwrap();
                for (index, worker) in others.iter_mut().enumerate() {
                    let stack = STALE_WORKER_STACKS + (index as u64 + 1) * STALE_WORKER_STACK_BYTES;
                    start_in_long_mode(worker, bsp, STALE_WORKER_CODE, stack);
                    worker.gpr[crate::testkit::RDI] = aliases[index].0 + 3;
                }
            },
        );
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(STALE_VERSION), 1000);
        assert_eq!(
            run.read(STALE_FAILURES, 8),
            [0; 8],
            "a worker ran code older than the version it saw published (jit {jit})"
        );
        if let Some(cache) = run.bsp.jit.as_ref() {
            assert!(cache.stats().stale_decodes > 0, "the window was never hit");
        }
    });
}

// FLUSH_WORKER/FLUSH_BSP, at the aliases: the same loop, counted into the same variable.
// With the workers on one alias each, and again with both on the same one, the cache must hold
// the same blocks: an alias costs no translation.
fn translated_with_workers_at(first: u64, second: u64) -> (u64, u32) {
    let aliases = [
        (ALIAS_ONE, STALE_WORKER_CODE),
        (ALIAS_TWO, STALE_WORKER_CODE),
    ];
    let run = run_threads_with(
        &FLUSH_BSP,
        3,
        true,
        |cpu| {
            place(cpu, STALE_WORKER_CODE, &FLUSH_WORKER);
            identity_paging(cpu, &aliases);
        },
        |cpus| {
            let (bsp, others) = cpus.split_first_mut().unwrap();
            for (worker, rip) in others.iter_mut().zip([first, second]) {
                let stack = STALE_WORKER_STACKS + 0x1000 * (rip >> 40);
                start_in_long_mode(worker, bsp, rip, stack);
            }
        },
    );
    assert_thread_shutdown(&run);
    let translated = run.bsp.jit.as_ref().map_or(0, |jit| jit.stats().translated);
    (translated, run.read_u32(FLUSH_COUNTER))
}

#[test]
fn vcpus_running_one_physical_page_at_different_addresses_share_its_blocks() {
    let expected = 3 * FLUSH_ITERATIONS;
    let (same, same_count) = translated_with_workers_at(ALIAS_ONE, ALIAS_ONE);
    let (different, different_count) = translated_with_workers_at(ALIAS_ONE, ALIAS_TWO);
    assert_eq!(
        (same_count, different_count),
        (expected, expected),
        "an increment was lost"
    );
    if same != 0 {
        assert_eq!(
            different, same,
            "the second alias translated what the first had"
        );
    }
}

const CHAIN_64: u64 = STALE_DATA + 0x40000;
const CHAIN_64_PAGES: u64 = 64;
const CHAIN_64_COUNTER: u64 = STALE_DATA + 0x40;
const CHAIN_64_DONE: u64 = STALE_DATA + 0x44;
// lock inc dword [CHAIN_64_COUNTER]
const CHAIN_64_STEP: [u8; 8] = [0xf0, 0xff, 0x04, 0x25, 0x40, 0x00, 0x40, 0x00];
// lock inc dword [CHAIN_64_COUNTER]; lock inc dword [CHAIN_64_DONE]; cli; hlt
const CHAIN_64_END: [u8; 18] = [
    0xf0, 0xff, 0x04, 0x25, 0x40, 0x00, 0x40, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x44, 0x00, 0x40, 0x00,
    0xfa, 0xf4,
];
// mov eax, CHAIN_64; jmp rax
const BSP_ENTER_CHAIN_64: [u8; 7] = [0xb8, 0x00, 0x00, 0x44, 0x00, 0xff, 0xe0];

fn run_the_chain(vcpus: usize) -> ThreadRun {
    run_threads_with(
        &BSP_ENTER_CHAIN_64,
        vcpus,
        true,
        |cpu| {
            for page in 0..CHAIN_64_PAGES {
                let at = CHAIN_64 + page * 0x1000;
                if page + 1 < CHAIN_64_PAGES {
                    // lock inc; jmp rel32 to the next page
                    let mut code = CHAIN_64_STEP.to_vec();
                    let rel = 0x1000u32 - (CHAIN_64_STEP.len() as u32 + 5);
                    code.push(0xe9);
                    code.extend_from_slice(&rel.to_le_bytes());
                    place(cpu, at, &code);
                } else {
                    place(cpu, at, &CHAIN_64_END);
                }
            }
        },
        |cpus| {
            let (bsp, others) = cpus.split_first_mut().unwrap();
            for (index, worker) in others.iter_mut().enumerate() {
                let stack = STALE_WORKER_STACKS + (index as u64 + 1) * STALE_WORKER_STACK_BYTES;
                start_in_long_mode(worker, bsp, CHAIN_64, stack);
            }
        },
    )
}

/// Four vCPUs run the same 64 pages of 64-bit code, one block each (and the last page two), all
/// starting at once: every block must be translated exactly once however the translations
/// interleave, which a lone vCPU running the same code shows as the number to expect, and every
/// locked increment must land.
#[test]
fn four_vcpus_running_the_same_pages_at_once_translate_each_block_once() {
    let vcpus = 4usize;
    let alone = run_the_chain(1);
    assert_thread_shutdown(&alone);
    // Without a JIT (a build or host that has none) there is nothing to count.
    let Some(alone_cache) = alone.bsp.jit.as_ref() else {
        return;
    };
    let expected = alone_cache.stats().translated;
    assert!(expected >= CHAIN_64_PAGES, "{expected}");
    for _ in 0..10 {
        let run = run_the_chain(vcpus);
        assert_thread_shutdown(&run);
        assert_eq!(run.read_u32(CHAIN_64_DONE), vcpus as u32);
        assert_eq!(
            run.read_u32(CHAIN_64_COUNTER),
            vcpus as u32 * CHAIN_64_PAGES as u32,
            "a locked increment was lost"
        );
        let stats = run.bsp.jit.as_ref().unwrap().stats();
        assert_eq!(stats.translated, expected, "a block was translated twice");
        assert_eq!(stats.resets, 0);
    }
}

/// The INVLPGB test's notes (the worker is ready at +0x30 and the flush done at +0x34 in the
/// code below): what the worker read before and after the flush, and the two physical pages
/// `SHOOTDOWN_VIRTUAL` maps to before and after.
const SHOOTDOWN_BEFORE: u64 = STALE_DATA + 0x38;
const SHOOTDOWN_AFTER: u64 = STALE_DATA + 0x3c;
const SHOOTDOWN_OLD_PAGE: u64 = STALE_DATA + 0x5000;
const SHOOTDOWN_NEW_PAGE: u64 = STALE_DATA + 0x6000;
const SHOOTDOWN_OLD_VALUE: u32 = 0x11;
const SHOOTDOWN_NEW_VALUE: u32 = 0x22;
/// The page the worker reads. Its page number's low bits differ from those of every page the
/// worker touches (the notes, its code and stack), so the direct-mapped soft TLB keeps its
/// translation until something drops it.
const SHOOTDOWN_VIRTUAL: u64 = 0x0000_2000_0003_5000;
/// Pages at the start of the table pool `identity_paging` allocates from: more than it uses.
const SHOOTDOWN_TABLE_PAGES: u64 = 16;
/// rAX bit 0 of INVLPGB: the address in rAX is valid.
const INVLPGB_VA_VALID: u64 = 1;

// The worker caches the translation of [rbx] (SHOOTDOWN_VIRTUAL), says so, waits for the flush and reads
// it again:
//   mov eax, [rbx]; mov [BEFORE], eax; mov dword [READY], 1
//   wait: pause; cmp dword [FLUSHED], 0; je wait
//   mov eax, [rbx]; mov [AFTER], eax; lock inc dword [EXITED]; cli; hlt
const SHOOTDOWN_WORKER: [u8; 51] = [
    0x8b, 0x03, 0x89, 0x04, 0x25, 0x38, 0x00, 0x40, 0x00, 0xc7, 0x04, 0x25, 0x30, 0x00, 0x40, 0x00,
    0x01, 0x00, 0x00, 0x00, 0xf3, 0x90, 0x83, 0x3c, 0x25, 0x34, 0x00, 0x40, 0x00, 0x00, 0x74, 0xf4,
    0x8b, 0x03, 0x89, 0x04, 0x25, 0x3c, 0x00, 0x40, 0x00, 0xf0, 0xff, 0x04, 0x25, 0x18, 0x00, 0x40,
    0x00, 0xfa, 0xf4,
];
// The bootstrap processor waits for it, repoints the page table entry at rdi to rsi, broadcasts
// the invalidation of rAX and waits for it, then lets the worker go:
//   wait: pause; cmp dword [READY], 0; je wait
//   mov [rdi], rsi; invlpgb; tlbsync; mov dword [FLUSHED], 1
//   wait2: pause; cmp dword [EXITED], 1; jne wait2; cli; hlt
const SHOOTDOWN_BSP: [u8; 46] = [
    0xf3, 0x90, 0x83, 0x3c, 0x25, 0x30, 0x00, 0x40, 0x00, 0x00, 0x74, 0xf4, 0x48, 0x89, 0x37, 0x0f,
    0x01, 0xfe, 0x0f, 0x01, 0xff, 0xc7, 0x04, 0x25, 0x34, 0x00, 0x40, 0x00, 0x01, 0x00, 0x00, 0x00,
    0xf3, 0x90, 0x83, 0x3c, 0x25, 0x18, 0x00, 0x40, 0x00, 0x01, 0x75, 0xf4, 0xfa, 0xf4,
];
/// Where `invlpgb; tlbsync` sit in `SHOOTDOWN_BSP`.
const SHOOTDOWN_BSP_FLUSH: core::ops::Range<usize> = 15..21;

/// The last-level page table entry for `linear` in `cpu`'s 4-level tables.
fn leaf_entry(cpu: &Cpu, linear: u64) -> u64 {
    use super::test_support::{PTE_ADDRESS_BITS, read_pte};
    const TABLE_LEVELS_ABOVE_LEAF: u32 = 3;
    let index = |level: u32| ((linear >> (12 + 9 * level)) & 0x1ff) * 8;
    let mut table = cpu.cr3 & PTE_ADDRESS_BITS;
    for level in (1..=TABLE_LEVELS_ABOVE_LEAF).rev() {
        table = read_pte(cpu, table + index(level)) & PTE_ADDRESS_BITS;
    }
    table + index(0)
}

/// Runs the shootdown on two threaded vCPUs and returns what the worker read after it.
fn worker_reads_after_a_remap(bsp_code: &[u8], jit: bool) -> u32 {
    use super::test_support::PTE_PRESENT_WRITABLE_USER;
    let run = run_threads_with(
        bsp_code,
        2,
        jit,
        |cpu| {
            place(cpu, STALE_WORKER_CODE, &SHOOTDOWN_WORKER);
            place(cpu, SHOOTDOWN_OLD_PAGE, &SHOOTDOWN_OLD_VALUE.to_le_bytes());
            place(cpu, SHOOTDOWN_NEW_PAGE, &SHOOTDOWN_NEW_VALUE.to_le_bytes());
            // The bootstrap processor rewrites a page table entry, so the tables are mapped too.
            let mut mappings = vec![(SHOOTDOWN_VIRTUAL, SHOOTDOWN_OLD_PAGE)];
            mappings.extend((0..SHOOTDOWN_TABLE_PAGES).map(|page| {
                let address = super::test_support::TABLE_POOL + page * 0x1000;
                (address, address)
            }));
            identity_paging(cpu, &mappings);
        },
        |cpus| {
            let (bsp, others) = cpus.split_first_mut().unwrap();
            bsp.gpr[crate::testkit::RDI] = leaf_entry(bsp, SHOOTDOWN_VIRTUAL);
            bsp.gpr[crate::testkit::RSI] = SHOOTDOWN_NEW_PAGE | PTE_PRESENT_WRITABLE_USER;
            bsp.gpr[crate::testkit::RAX] = SHOOTDOWN_VIRTUAL | INVLPGB_VA_VALID;
            bsp.gpr[crate::testkit::RCX] = 0;
            bsp.gpr[crate::testkit::RDX] = 0;
            let worker = &mut others[0];
            start_in_long_mode(worker, bsp, STALE_WORKER_CODE, STALE_WORKER_STACKS + 0x1000);
            worker.gpr[crate::testkit::RBX] = SHOOTDOWN_VIRTUAL;
        },
    );
    assert_thread_shutdown(&run);
    assert_eq!(run.read_u32(SHOOTDOWN_BEFORE), SHOOTDOWN_OLD_VALUE);
    run.read_u32(SHOOTDOWN_AFTER)
}

/// Linux 6.15+ on Zen 3 and later: a TLB shootdown is INVLPGB on the CPU that changed the page
/// tables, then TLBSYNC, with no IPI. The other vCPU, which cached the old translation and is
/// running, must not see it afterwards; without the instruction it does.
#[test]
fn invlpgb_and_tlbsync_drop_another_running_vcpus_stale_translation() {
    both_engines(|jit| {
        let mut without_flush = SHOOTDOWN_BSP.to_vec();
        without_flush.splice(SHOOTDOWN_BSP_FLUSH, [0x90; 6]);
        assert_eq!(
            worker_reads_after_a_remap(&without_flush, jit),
            SHOOTDOWN_OLD_VALUE,
            "jit {jit}: the worker's TLB really holds the old page"
        );
        assert_eq!(
            worker_reads_after_a_remap(&SHOOTDOWN_BSP, jit),
            SHOOTDOWN_NEW_VALUE,
            "jit {jit}"
        );
    });
}
