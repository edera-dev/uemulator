//! Running MSR writes on the application processors, which a PC firmware does for state that is
//! per logical processor and that the OS expects to find the same on all of them (the MTRRs).
//!
//! The APs sit in wait-for-SIPI. One INIT/start-up broadcast runs a real-mode stub on each of
//! them that performs the writes from a table, counts itself done and spins; a second INIT puts
//! them back into wait-for-SIPI for the OS (Linux starts them with INIT/SIPI again). INIT keeps
//! the MTRRs and the other MSRs (SDM vol. 3 table 10-1), which is the whole point.
//!
//! The stub spins with `pause` rather than `cli; hlt`: on this machine HLT with IF clear ends the
//! run (`plans/07-devices.md`), and with time-sharing the spinning AP must give its slice back.

use core::arch::global_asm;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::{self, rdmsr, rdtsc, wrmsr};

/// Physical page the stub runs from: below 1 MiB and 4 KiB aligned as a start-up vector must be,
/// inside the range `memory.rs` keeps reserved (`LOW_PAGES_FIRST..LOW_PAGES_END`, between the
/// handoff page and the boot page tables), so no allocation can hand it out.
const STUB_PAGE: u64 = 0x8000;
const STUB_PAGE_BYTES: usize = 0x1000;
/// Start-up IPI vector: the page number.
const STARTUP_VECTOR: u32 = (STUB_PAGE >> 12) as u32;
/// Layout of the stub page after the code: the done counter, the entry count, the table of
/// (msr u32, low u32, high u32) entries. The stub addresses them relative to its segment.
const DONE_OFFSET: usize = 0x80;
const COUNT_OFFSET: usize = 0x84;
const TABLE_OFFSET: usize = 0x100;
const TABLE_ENTRY_BYTES: usize = 12;
const MAX_ENTRIES: usize = (STUB_PAGE_BYTES - TABLE_OFFSET) / TABLE_ENTRY_BYTES;

const MSR_APIC_BASE: u32 = 0x1b;
const APIC_BASE_X2APIC: u64 = 1 << 10;
const APIC_BASE_ADDRESS_MASK: u64 = 0xf_ffff_f000;
/// xAPIC registers (offsets from the base) and the x2APIC MSR of the ICR.
const APIC_SVR: u64 = 0xf0;
const APIC_LVT_LINT0: u64 = 0x350;
const APIC_LVT_LINT1: u64 = 0x360;
const APIC_ICR_LOW: u64 = 0x300;
const APIC_ICR_HIGH: u64 = 0x310;
const MSR_X2APIC_ICR: u32 = 0x830;
const X2APIC_MSR_BASE: u32 = 0x800;
const SVR_SOFTWARE_ENABLE: u32 = 1 << 8;
/// Virtual-wire mode, what PC firmware leaves in LINT0/LINT1 and what the library's LAPIC comes
/// out of reset with: the PIC's interrupts reach the CPU as ExtINT through LINT0, LINT1 is a
/// masked NMI.
const LVT_DELIVERY_EXTINT: u32 = 7 << 8;
const LVT_DELIVERY_NMI: u32 = 4 << 8;
const LVT_MASKED: u32 = 1 << 16;
/// ICR: delivery mode INIT or start-up, level assert, destination shorthand "all excluding self".
const ICR_DELIVERY_INIT: u32 = 5 << 8;
const ICR_DELIVERY_STARTUP: u32 = 6 << 8;
const ICR_LEVEL_ASSERT: u32 = 1 << 14;
const ICR_ALL_EXCLUDING_SELF: u32 = 3 << 18;
const ICR_DELIVERY_PENDING: u32 = 1 << 12;
/// How long the APs get to check in after each start-up IPI. The Intel MP start-up sequence
/// sends a second SIPI to processors that missed the first.
const CHECK_IN_TIMEOUT_NS: u64 = 100_000_000;
const STARTUP_ATTEMPTS: u32 = 2;
const NS_PER_SECOND: u128 = 1_000_000_000;

global_asm!(
    ".section .text.ap_stub,\"ax\"",
    ".global ap_stub_start",
    ".global ap_stub_end",
    ".code16",
    "ap_stub_start:",
    "cli",
    "mov ax, cs",
    "mov ds, ax",
    "mov si, {table}",
    "mov bx, word ptr [{count}]",
    "2:",
    "test bx, bx",
    "jz 3f",
    "mov ecx, dword ptr [si]",
    "mov eax, dword ptr [si + 4]",
    "mov edx, dword ptr [si + 8]",
    "wrmsr",
    "add si, {entry_bytes}",
    "dec bx",
    "jmp 2b",
    "3:",
    "lock inc dword ptr [{done}]",
    "4:",
    "pause",
    "jmp 4b",
    "ap_stub_end:",
    ".code64",
    table = const TABLE_OFFSET,
    count = const COUNT_OFFSET,
    done = const DONE_OFFSET,
    entry_bytes = const TABLE_ENTRY_BYTES,
);

unsafe extern "C" {
    static ap_stub_start: u8;
    static ap_stub_end: u8;
}

/// The BSP's local APIC, in whichever mode it is.
enum Apic {
    Mmio(u64),
    X2apic,
}

impl Apic {
    fn current() -> Self {
        let base = rdmsr(MSR_APIC_BASE);
        if base & APIC_BASE_X2APIC != 0 {
            Apic::X2apic
        } else {
            Apic::Mmio(base & APIC_BASE_ADDRESS_MASK)
        }
    }

    fn read(&self, register: u64) -> u32 {
        match *self {
            Apic::Mmio(base) => unsafe { ((base + register) as *const u32).read_volatile() },
            Apic::X2apic => rdmsr(x2apic_msr(register)) as u32,
        }
    }

    fn write(&self, register: u64, value: u32) {
        match *self {
            Apic::Mmio(base) => unsafe { ((base + register) as *mut u32).write_volatile(value) },
            Apic::X2apic => wrmsr(x2apic_msr(register), value as u64),
        }
    }

    fn broadcast(&self, command: u32) {
        match self {
            Apic::Mmio(_) => {
                self.write(APIC_ICR_HIGH, 0);
                self.write(APIC_ICR_LOW, command);
                while self.read(APIC_ICR_LOW) & ICR_DELIVERY_PENDING != 0 {
                    core::hint::spin_loop();
                }
            }
            // One 64-bit ICR, no delivery status (SDM 10.12.9).
            Apic::X2apic => wrmsr(MSR_X2APIC_ICR, command as u64),
        }
    }
}

/// x2APIC MSR of an xAPIC register offset (SDM 10.12.1.2).
fn x2apic_msr(register: u64) -> u32 {
    X2APIC_MSR_BASE + (register >> 4) as u32
}

/// Waits until `expected` APs have counted themselves done or the timeout passes.
fn wait_for_check_in(done: &AtomicU32, expected: u32, tsc_hz: u64) -> bool {
    let timeout_ticks = (CHECK_IN_TIMEOUT_NS as u128 * tsc_hz as u128 / NS_PER_SECOND) as u64;
    let start = rdtsc();
    while done.load(Ordering::Acquire) < expected {
        if rdtsc().wrapping_sub(start) > timeout_ticks {
            return false;
        }
        // `pause` also hands the slice to the APs when the vCPUs take turns on one thread.
        core::hint::spin_loop();
    }
    true
}

/// Performs `writes` on every AP, then parks them in wait-for-SIPI again. `tsc_hz` times the
/// wait for them.
pub fn write_msrs_on_application_processors(writes: &[(u32, u64)], tsc_hz: u64) {
    let others = cpu::logical_processors() - 1;
    if others == 0 {
        return;
    }
    assert!(writes.len() <= MAX_ENTRIES);
    let stub = unsafe {
        let start = &raw const ap_stub_start;
        let length = (&raw const ap_stub_end).offset_from(start) as usize;
        core::slice::from_raw_parts(start, length)
    };
    assert!(stub.len() <= DONE_OFFSET);
    let page = unsafe { core::slice::from_raw_parts_mut(STUB_PAGE as *mut u8, STUB_PAGE_BYTES) };
    page.fill(0);
    page[..stub.len()].copy_from_slice(stub);
    page[COUNT_OFFSET..COUNT_OFFSET + 2].copy_from_slice(&(writes.len() as u16).to_le_bytes());
    for (index, &(msr, value)) in writes.iter().enumerate() {
        let at = TABLE_OFFSET + index * TABLE_ENTRY_BYTES;
        page[at..at + 4].copy_from_slice(&msr.to_le_bytes());
        page[at + 4..at + 12].copy_from_slice(&value.to_le_bytes());
    }
    let done = unsafe { AtomicU32::from_ptr((STUB_PAGE as usize + DONE_OFFSET) as *mut u32) };
    // IPIs need a software-enabled APIC, and it stays enabled. Disabling it again would mask every
    // LVT entry (SDM 10.4.7.2), LINT0 among them, which disconnects the PIC: the PIT interrupt
    // that ends `cpu::idle_wait` would never come, and every `Stall`, the boot countdown and
    // GRUB's menu timeout would hang on a machine with more than one CPU. SeaBIOS and OVMF also
    // hand over an enabled APIC in virtual-wire mode.
    let apic = Apic::current();
    let previous_svr = apic.read(APIC_SVR);
    apic.write(APIC_SVR, previous_svr | SVR_SOFTWARE_ENABLE);
    apic.broadcast(ICR_DELIVERY_INIT | ICR_LEVEL_ASSERT | ICR_ALL_EXCLUDING_SELF);
    let mut checked_in = false;
    for _attempt in 0..STARTUP_ATTEMPTS {
        apic.broadcast(ICR_DELIVERY_STARTUP | ICR_ALL_EXCLUDING_SELF | STARTUP_VECTOR);
        checked_in = wait_for_check_in(done, others, tsc_hz);
        if checked_in {
            break;
        }
    }
    if !checked_in {
        crate::serial::println!(
            "firmware: {} of {others} processors ran the start-up stub",
            done.load(Ordering::Acquire)
        );
    }
    apic.broadcast(ICR_DELIVERY_INIT | ICR_LEVEL_ASSERT | ICR_ALL_EXCLUDING_SELF);
    apic.write(APIC_LVT_LINT0, LVT_DELIVERY_EXTINT);
    apic.write(APIC_LVT_LINT1, LVT_DELIVERY_NMI | LVT_MASKED);
}
