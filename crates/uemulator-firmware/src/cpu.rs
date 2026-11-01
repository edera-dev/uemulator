//! The processor as the firmware needs it: port I/O, the time stamp counter, an IDT that turns
//! any exception into a message on the serial port, and the legacy timer interrupt that lets
//! `idle_wait` sleep instead of spinning (the emulator skips virtual time while the CPU halts
//! with a timer pending, so a one-second `Stall` costs no host time).

use core::arch::{asm, global_asm};

use crate::serial;
use crate::sync::Global;

pub fn outb(port: u16, value: u8) {
    unsafe { asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack)) }
}

pub fn outl(port: u16, value: u32) {
    unsafe { asm!("out dx, eax", in("dx") port, in("eax") value, options(nomem, nostack)) }
}

pub fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe { asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack)) }
    value
}

pub fn inl(port: u16) -> u32 {
    let value: u32;
    unsafe { asm!("in eax, dx", out("eax") value, in("dx") port, options(nomem, nostack)) }
    value
}

pub fn rdtsc() -> u64 {
    let (low, high): (u32, u32);
    unsafe { asm!("rdtsc", out("eax") low, out("edx") high, options(nomem, nostack)) }
    (high as u64) << 32 | low as u64
}

pub fn cpuid(leaf: u32, subleaf: u32) -> [u32; 4] {
    let result = core::arch::x86_64::__cpuid_count(leaf, subleaf);
    [result.eax, result.ebx, result.ecx, result.edx]
}

pub fn rdmsr(msr: u32) -> u64 {
    let (low, high): (u32, u32);
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") low, out("edx") high, options(nomem, nostack))
    }
    (high as u64) << 32 | low as u64
}

pub fn wrmsr(msr: u32, value: u64) {
    unsafe {
        asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") value as u32,
            in("edx") (value >> 32) as u32,
            options(nomem, nostack),
        )
    }
}

const CPUID_SIGNATURE: u32 = 1;
const CPUID_BRAND_FIRST: u32 = 0x8000_0002;
const CPUID_BRAND_LEAVES: u32 = 3;
/// ECX bits 7:0: logical processors in the package minus one (AMD APM vol. 3, CPUID Fn8000_0008).
const CPUID_ADDRESS_SIZES: u32 = 0x8000_0008;
const LOGICAL_PROCESSORS_MINUS_ONE_MASK: u32 = 0xff;
/// EBX bits 15:8: threads per core minus one (CPUID Fn8000_001E).
const CPUID_EXTENDED_APIC_ID: u32 = 0x8000_001e;
const THREADS_PER_CORE_SHIFT: u32 = 8;
const THREADS_PER_CORE_MASK: u32 = 0xff;
const HZ_PER_MHZ: u64 = 1_000_000;

/// Logical processors in the package, from CPUID: the library starts every vCPU it has, and
/// they all sit in the one package it reports.
pub fn logical_processors() -> u32 {
    (cpuid(CPUID_ADDRESS_SIZES, 0)[2] & LOGICAL_PROCESSORS_MINUS_ONE_MASK) + 1
}

/// The processor structure's contents, from CPUID and the TSC rate (which is the clock the
/// guest sees: the library's TSC counts at `tsc_hz`).
pub fn smbios_processor(tsc_hz: u64) -> crate::smbios::Processor {
    let [signature, _, _, feature_edx] = cpuid(CPUID_SIGNATURE, 0);
    let mut brand = [0u8; 48];
    for leaf in 0..CPUID_BRAND_LEAVES {
        for (index, word) in cpuid(CPUID_BRAND_FIRST + leaf, 0).into_iter().enumerate() {
            let at = leaf as usize * 16 + index * 4;
            brand[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    let threads = logical_processors();
    let threads_per_core = ((cpuid(CPUID_EXTENDED_APIC_ID, 0)[1] >> THREADS_PER_CORE_SHIFT)
        & THREADS_PER_CORE_MASK)
        + 1;
    crate::smbios::Processor {
        signature,
        feature_edx,
        brand,
        cores: (threads / threads_per_core).max(1) as u16,
        threads: threads as u16,
        mhz: (tsc_hz / HZ_PER_MHZ).min(u16::MAX as u64) as u16,
    }
}

pub fn halt_forever() -> ! {
    loop {
        unsafe { asm!("cli; hlt", options(nomem, nostack)) }
    }
}

const PIC_MASTER_COMMAND: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_COMMAND: u16 = 0xa0;
const PIC_SLAVE_DATA: u16 = 0xa1;
const PIC_ICW1_INIT_WITH_ICW4: u8 = 0x11;
const PIC_ICW4_8086: u8 = 0x01;
const PIC_EOI: u8 = 0x20;
const PIC_SLAVE_ON_IRQ2: u8 = 1 << 2;
const PIC_SLAVE_ID: u8 = 2;
const MASK_ALL: u8 = 0xff;
const MASK_ALL_BUT_TIMER: u8 = 0xfe;
const TIMER_VECTOR: u8 = 0x20;
const SLAVE_VECTOR_BASE: u8 = 0x28;

const PIT_CHANNEL0: u16 = 0x40;
const PIT_COMMAND: u16 = 0x43;
/// Channel 0, access lobyte/hibyte, mode 0 (interrupt on terminal count), binary.
const PIT_ONE_SHOT: u8 = 0x30;
const PIT_HZ: u64 = 1_193_182;
const PIT_MAX_COUNT: u64 = 0xffff;

const IDT_ENTRIES: usize = 256;
const EXCEPTION_VECTORS: usize = 32;
const GATE_PRESENT_INTERRUPT: u8 = 0x8e;

#[repr(C)]
#[derive(Clone, Copy)]
struct Gate {
    offset_low: u16,
    selector: u16,
    ist: u8,
    attributes: u8,
    offset_middle: u16,
    offset_high: u32,
    reserved: u32,
}

const EMPTY_GATE: Gate = Gate {
    offset_low: 0,
    selector: 0,
    ist: 0,
    attributes: 0,
    offset_middle: 0,
    offset_high: 0,
    reserved: 0,
};

static IDT: Global<[Gate; IDT_ENTRIES]> = Global::new([EMPTY_GATE; IDT_ENTRIES]);

#[repr(C, packed)]
struct Idtr {
    limit: u16,
    base: u64,
}

/// What `isr_common` pushes, lowest address first, then the CPU's own frame.
#[repr(C)]
struct ExceptionFrame {
    registers: [u64; 15],
    vector: u64,
    error_code: u64,
    rip: u64,
    cs: u64,
    rflags: u64,
    rsp: u64,
    ss: u64,
}

/// Bytes printed from the faulting instruction's address.
const CODE_DUMP_BYTES: usize = 16;

const REGISTER_NAMES: [&str; 15] = [
    "r15", "r14", "r13", "r12", "r11", "r10", "r9", "r8", "rbp", "rdi", "rsi", "rdx", "rcx", "rbx",
    "rax",
];

// Vectors 8, 10-14, 17, 21, 29 and 30 push an error code; the others get a zero so the frame has
// one shape.
macro_rules! isr_stubs {
    ($($vector:literal $kind:ident),* $(,)?) => {
        global_asm!(
            $(
                concat!("isr", $vector, ":"),
                isr_stubs!(@push $kind $vector),
                "jmp isr_common",
            )*
            ".p2align 3",
            ".global ISR_TABLE",
            "ISR_TABLE:",
            $(concat!(".quad isr", $vector),)*
            "isr_common:",
            "push rax", "push rbx", "push rcx", "push rdx", "push rsi", "push rdi", "push rbp",
            "push r8", "push r9", "push r10", "push r11", "push r12", "push r13", "push r14",
            "push r15",
            "mov rdi, rsp",
            "and rsp, -16",
            "call exception_handler",
            "2: cli", "hlt", "jmp 2b",
            ".global timer_isr",
            "timer_isr:",
            "push rax",
            "mov al, {eoi}",
            "out {master}, al",
            "pop rax",
            "iretq",
            eoi = const PIC_EOI,
            master = const PIC_MASTER_COMMAND,
        );
    };
    (@push none $vector:literal) => { concat!("push 0; push ", $vector) };
    (@push code $vector:literal) => { concat!("push ", $vector) };
}

isr_stubs!(
    0 none, 1 none, 2 none, 3 none, 4 none, 5 none, 6 none, 7 none, 8 code, 9 none, 10 code,
    11 code, 12 code, 13 code, 14 code, 15 none, 16 none, 17 code, 18 none, 19 none, 20 none,
    21 code, 22 none, 23 none, 24 none, 25 none, 26 none, 27 none, 28 none, 29 code, 30 code,
    31 none,
);

unsafe extern "C" {
    static ISR_TABLE: [u64; EXCEPTION_VECTORS];
    fn timer_isr();
}

#[unsafe(no_mangle)]
extern "C" fn exception_handler(frame: *const ExceptionFrame) -> ! {
    let frame = unsafe { &*frame };
    let cr2: u64;
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack)) }
    serial::println!(
        "\nfirmware exception {} error {:#x} at {:#x} (cs {:#x} rflags {:#x} rsp {:#x} cr2 {:#x})",
        frame.vector,
        frame.error_code,
        frame.rip,
        frame.cs,
        frame.rflags,
        frame.rsp,
        cr2
    );
    for (name, value) in REGISTER_NAMES.iter().zip(frame.registers.iter()) {
        serial::print!("{name}={value:#018x} ");
    }
    serial::println!();
    // Identity-mapped RAM; a bad rip that faults here ends in a triple fault, which is no worse.
    let code = unsafe { core::slice::from_raw_parts(frame.rip as *const u8, CODE_DUMP_BYTES) };
    serial::print!("code at rip:");
    for byte in code {
        serial::print!(" {byte:02x}");
    }
    serial::println!();
    halt_forever()
}

fn set_gate(vector: usize, handler: u64) {
    let selector: u16;
    unsafe { asm!("mov {:x}, cs", out(reg) selector, options(nomem, nostack)) }
    IDT.get()[vector] = Gate {
        offset_low: handler as u16,
        selector,
        ist: 0,
        attributes: GATE_PRESENT_INTERRUPT,
        offset_middle: (handler >> 16) as u16,
        offset_high: (handler >> 32) as u32,
        reserved: 0,
    };
}

/// Installs the IDT and remaps the PIC so IRQ0 lands on `TIMER_VECTOR`, clear of the CPU's
/// exception vectors; every line is masked until `idle_wait` needs the timer.
pub fn init() {
    for (vector, handler) in unsafe { ISR_TABLE }.into_iter().enumerate() {
        set_gate(vector, handler);
    }
    set_gate(
        TIMER_VECTOR as usize,
        timer_isr as *const () as usize as u64,
    );
    let idtr = Idtr {
        limit: (IDT_ENTRIES * core::mem::size_of::<Gate>() - 1) as u16,
        base: IDT.get().as_ptr() as u64,
    };
    unsafe { asm!("lidt [{}]", in(reg) &idtr, options(readonly, nostack)) }
    outb(PIC_MASTER_COMMAND, PIC_ICW1_INIT_WITH_ICW4);
    outb(PIC_SLAVE_COMMAND, PIC_ICW1_INIT_WITH_ICW4);
    outb(PIC_MASTER_DATA, TIMER_VECTOR);
    outb(PIC_SLAVE_DATA, SLAVE_VECTOR_BASE);
    outb(PIC_MASTER_DATA, PIC_SLAVE_ON_IRQ2);
    outb(PIC_SLAVE_DATA, PIC_SLAVE_ID);
    outb(PIC_MASTER_DATA, PIC_ICW4_8086);
    outb(PIC_SLAVE_DATA, PIC_ICW4_8086);
    mask_interrupts();
}

pub fn mask_interrupts() {
    outb(PIC_MASTER_DATA, MASK_ALL);
    outb(PIC_SLAVE_DATA, MASK_ALL);
}

/// Halts until the timer fires or `max_nanoseconds` (at most about 54 ms) have passed. May return
/// early; callers loop on their own deadline.
pub fn idle_wait(max_nanoseconds: u64) {
    let count = (max_nanoseconds as u128 * PIT_HZ as u128 / 1_000_000_000)
        .clamp(1, PIT_MAX_COUNT as u128) as u64;
    outb(PIT_COMMAND, PIT_ONE_SHOT);
    outb(PIT_CHANNEL0, count as u8);
    outb(PIT_CHANNEL0, (count >> 8) as u8);
    outb(PIC_MASTER_DATA, MASK_ALL_BUT_TIMER);
    // `sti` takes effect after the next instruction, so the interrupt cannot slip in between it
    // and the `hlt` and leave the CPU asleep with the wakeup already spent.
    unsafe { asm!("sti", "hlt", "cli", options(nomem, nostack)) }
    outb(PIC_MASTER_DATA, MASK_ALL);
}
