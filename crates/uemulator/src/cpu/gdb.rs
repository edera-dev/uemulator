//! The vCPU side of the GDB stub (`docs/plans/30-gdb-stub.md`): noticing a stop, the registers
//! in gdb's layout, memory by guest-virtual address, and the view of every vCPU the protocol
//! engine (`crate::gdb`) works on while the machine is stopped.
//!
//! The debugger's breakpoints, watchpoints and single-step are its own, never the guest's DR0-DR7:
//! the guest can use its debug registers under a debugger and sees nothing of it. Nothing is
//! written into guest memory (no 0xCC). Instead, while anything is armed (`GdbVcpu::armed`) the
//! vCPU runs every instruction through `step_debug`, the interpreter path the guest's own debug
//! registers already use (`debug.rs`), and `debugger_check` compares RIP against the breakpoints
//! before each one. That makes the JIT and the interpreter alike under a debugger, at the price
//! of interpreting the whole machine while a breakpoint is set.

use super::*;
use crate::gdb::{
    DebugConfig, DebugLink, ResumeAction, StopEvent, StopKind, WatchKind, Watchpoint,
};

/// The debugger's state on one vCPU: what it watches for and what it saw. Host-side, never in a
/// snapshot. Changed only while the machine is stopped (`apply_resume`), or by the vCPU itself.
#[derive(Default)]
pub(crate) struct GdbVcpu {
    /// Something may stop this vCPU: breakpoints, watchpoints, a step, or being held. Makes
    /// `debug_slow_path` true, so every instruction goes through `step_debug` and
    /// `debugger_check` runs before each.
    pub armed: bool,
    /// Execute one instruction, then stop.
    pub step: bool,
    /// The instruction `step` asked for has run (or raised an event): stop at the next boundary.
    pub stepped: bool,
    /// Linear addresses, sorted.
    breakpoints: Vec<u64>,
    pub watchpoints: Vec<Watchpoint>,
    /// A watchpoint the last instruction hit: its kind and address. Reported after the
    /// instruction completes, as hardware watchpoints trap.
    pub watch_hit: Option<(WatchKind, u64)>,
}

impl GdbVcpu {
    fn load(&mut self, config: &DebugConfig) {
        self.breakpoints = config.breakpoints.keys().copied().collect();
        self.watchpoints = config.watchpoints.clone();
    }

    fn has_points(&self) -> bool {
        !self.breakpoints.is_empty() || !self.watchpoints.is_empty()
    }
}

/// gdb's numbering of the general registers (rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15) as
/// indices into `Cpu::gpr`, which follows the instruction encoding (rax, rcx, rdx, rbx, ...).
const GDB_GPR_ORDER: [usize; 16] = [0, 3, 1, 2, 6, 7, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15];
/// The segment registers in gdb's order: cs, ss, ds, es, fs, gs.
const GDB_SEGMENT_ORDER: [usize; 6] = [SEG_CS, SEG_SS, SEG_DS, SEG_ES, SEG_FS, SEG_GS];

const REG_RIP: usize = 16;
const REG_EFLAGS: usize = 17;
/// rax through r15: indices into `GDB_GPR_ORDER`.
const REG_GPRS: core::ops::Range<usize> = 0..16;
const REG_SEGMENTS: usize = 18;
const REG_ST0: usize = 24;
const REG_FCTRL: usize = 32;
const REG_FSTAT: usize = 33;
const REG_FTAG: usize = 34;
/// fiseg, fioff, foseg, fooff, fop: the last x87 instruction and operand, which this CPU does not
/// track (they read as zero, as after `fninit`).
const REG_FPU_POINTERS: usize = 35;
const REG_FOP: usize = 39;
pub(crate) const REG_XMM0: usize = 40;
const REG_MXCSR: usize = 56;
const REG_FS_BASE: usize = 57;
const REG_GS_BASE: usize = 58;
const REG_KERNEL_GS_BASE: usize = 59;
const REG_CR0: usize = 60;
const REG_CR2: usize = 61;
const REG_CR3: usize = 62;
const REG_CR4: usize = 63;
const REG_CR8: usize = 64;
const REG_EFER: usize = 65;
pub(crate) const REGISTER_COUNT: usize = 66;

/// CR8 is the local APIC's task priority, bits 7:4 (SDM 10.8.6.1).
const CR8_TPR_SHIFT: u32 = 4;
/// Full x87 tag word: two bits per physical register.
const TAG_VALID: u16 = 0;
const TAG_ZERO: u16 = 1;
const TAG_SPECIAL: u16 = 2;
const TAG_EMPTY: u16 = 3;
const TAG_MASK: u16 = 3;
const TAG_BITS: usize = 2;
/// EFLAGS bits that exist: CF, PF, AF, ZF, SF, TF, IF, DF, OF, IOPL, NT, RF, VM, AC, VIF, VIP,
/// ID. Bit 1 always reads 1; the rest are reserved, and a debugger may not set them.
const EFLAGS_DEFINED: u64 = 0x003f_7fd5;
const X87_EXPONENT_MASK: u16 = 0x7fff;
const X87_INTEGER_BIT: u64 = 1 << 63;
const FSW_TOP_SHIFT: u32 = 11;
const X87_REGISTERS: usize = 8;

/// One register of the target description.
pub(crate) struct Register {
    pub name: &'static str,
    pub bits: u32,
    /// A gdb type: predefined (`int64`, `code_ptr`, `i387_ext`) or `vec128`, which the target
    /// description defines.
    pub kind: &'static str,
    pub group: Option<&'static str>,
}

const fn register(name: &'static str, bits: u32, kind: &'static str) -> Register {
    Register {
        name,
        bits,
        kind,
        group: None,
    }
}

const fn float(name: &'static str) -> Register {
    Register {
        name,
        bits: 32,
        kind: "int",
        group: Some("float"),
    }
}

/// gdb's amd64 register order (the `g` packet), then what QEMU adds for system debugging: the
/// segment bases and the control registers. Clients take the layout from `target.xml`.
pub(crate) const REGISTERS: [Register; REGISTER_COUNT] = [
    register("rax", 64, "int64"),
    register("rbx", 64, "int64"),
    register("rcx", 64, "int64"),
    register("rdx", 64, "int64"),
    register("rsi", 64, "int64"),
    register("rdi", 64, "int64"),
    register("rbp", 64, "data_ptr"),
    register("rsp", 64, "data_ptr"),
    register("r8", 64, "int64"),
    register("r9", 64, "int64"),
    register("r10", 64, "int64"),
    register("r11", 64, "int64"),
    register("r12", 64, "int64"),
    register("r13", 64, "int64"),
    register("r14", 64, "int64"),
    register("r15", 64, "int64"),
    register("rip", 64, "code_ptr"),
    register("eflags", 32, "int32"),
    register("cs", 32, "int32"),
    register("ss", 32, "int32"),
    register("ds", 32, "int32"),
    register("es", 32, "int32"),
    register("fs", 32, "int32"),
    register("gs", 32, "int32"),
    register("st0", 80, "i387_ext"),
    register("st1", 80, "i387_ext"),
    register("st2", 80, "i387_ext"),
    register("st3", 80, "i387_ext"),
    register("st4", 80, "i387_ext"),
    register("st5", 80, "i387_ext"),
    register("st6", 80, "i387_ext"),
    register("st7", 80, "i387_ext"),
    float("fctrl"),
    float("fstat"),
    float("ftag"),
    float("fiseg"),
    float("fioff"),
    float("foseg"),
    float("fooff"),
    float("fop"),
    register("xmm0", 128, "vec128"),
    register("xmm1", 128, "vec128"),
    register("xmm2", 128, "vec128"),
    register("xmm3", 128, "vec128"),
    register("xmm4", 128, "vec128"),
    register("xmm5", 128, "vec128"),
    register("xmm6", 128, "vec128"),
    register("xmm7", 128, "vec128"),
    register("xmm8", 128, "vec128"),
    register("xmm9", 128, "vec128"),
    register("xmm10", 128, "vec128"),
    register("xmm11", 128, "vec128"),
    register("xmm12", 128, "vec128"),
    register("xmm13", 128, "vec128"),
    register("xmm14", 128, "vec128"),
    register("xmm15", 128, "vec128"),
    Register {
        name: "mxcsr",
        bits: 32,
        kind: "int",
        group: Some("vector"),
    },
    register("fs_base", 64, "int64"),
    register("gs_base", 64, "int64"),
    register("k_gs_base", 64, "int64"),
    register("cr0", 64, "int64"),
    register("cr2", 64, "int64"),
    register("cr3", 64, "int64"),
    register("cr4", 64, "int64"),
    register("cr8", 64, "int64"),
    register("efer", 64, "int64"),
];

/// The target description's features and the registers each holds. gdb insists on the first
/// (it names the core registers it requires); the segment bases go where gdb looks for them.
pub(crate) const FEATURES: [(&str, core::ops::Range<usize>); 4] = [
    ("org.gnu.gdb.i386.core", 0..REG_XMM0),
    ("org.gnu.gdb.i386.sse", REG_XMM0..REG_FS_BASE),
    ("org.gnu.gdb.i386.segments", REG_FS_BASE..REG_KERNEL_GS_BASE),
    (
        "org.uemulator.x86_64.system",
        REG_KERNEL_GS_BASE..REGISTER_COUNT,
    ),
];

/// Every vCPU of a stopped machine, as the protocol engine sees them.
pub(crate) trait Vcpus {
    fn count(&self) -> usize;
    /// vCPU `index`; `None` only for a vCPU thread that has not started or already finished.
    fn cpu(&mut self, index: usize) -> Option<&mut Cpu>;
    /// The vCPU that holds the machine-wide state (RAM, devices, JIT): time-sharing moves it to
    /// whichever vCPU ran last (`Cpu::swap_shared`), so another vCPU's own `Cpu` may have none.
    fn machine(&mut self) -> &mut Cpu;
}

/// A machine with one vCPU.
pub(crate) struct OneVcpu<'a>(pub &'a mut Cpu);

impl Vcpus for OneVcpu<'_> {
    fn count(&self) -> usize {
        1
    }
    fn cpu(&mut self, index: usize) -> Option<&mut Cpu> {
        (index == 0).then_some(&mut *self.0)
    }
    fn machine(&mut self) -> &mut Cpu {
        self.0
    }
}

/// Time-shared vCPUs (`smp.rs`).
pub(crate) struct SharedVcpus<'a> {
    pub cpus: &'a mut [Cpu],
    pub active: usize,
}

impl Vcpus for SharedVcpus<'_> {
    fn count(&self) -> usize {
        self.cpus.len()
    }
    fn cpu(&mut self, index: usize) -> Option<&mut Cpu> {
        self.cpus.get_mut(index)
    }
    fn machine(&mut self) -> &mut Cpu {
        &mut self.cpus[self.active]
    }
}

/// vCPUs on threads: the one serving the stop, inside its exclusive section, and the others
/// through the pointers they publish (`vcpus.rs`).
pub(crate) struct ThreadVcpus<'a>(pub &'a mut Cpu);

impl Vcpus for ThreadVcpus<'_> {
    fn count(&self) -> usize {
        self.0.cpu_count as usize
    }
    fn cpu(&mut self, index: usize) -> Option<&mut Cpu> {
        if index == self.0.cpu_index as usize {
            return Some(&mut *self.0);
        }
        self.0.other_cpu(index)
    }
    fn machine(&mut self) -> &mut Cpu {
        self.0
    }
}

impl Cpu {
    /// Connects this vCPU to the machine's debugger (or to none), arming it with what the
    /// debugger already has set: a reboot builds new vCPUs that keep the breakpoints.
    pub(crate) fn attach_debugger(&mut self, link: Option<Arc<DebugLink>>) {
        self.gdb = GdbVcpu::default();
        if let Some(link) = &link {
            self.gdb.load(&link.config());
            self.gdb.armed = self.gdb.has_points() || link.pause_pending();
        }
        self.debugger = link;
    }

    /// At the top of every `before_step`, before interrupts are delivered (so a step or a held
    /// vCPU does not move into a handler first): whether to stop for the debugger, or what a
    /// held vCPU does instead of running.
    pub(super) fn debugger_check(&mut self) -> Option<BeforeStep> {
        let vcpu = self.cpu_index as usize;
        let (pause, held) = {
            let link = self.debugger.as_ref()?;
            // A vCPU the guest has not started yet leaves a pause to one that runs, so the stop
            // names a vCPU worth looking at (the first, at the halt at start).
            let pause = link.pause_pending() && !self.wait_for_sipi;
            if !self.gdb.armed && !pause {
                return None;
            }
            (pause, link.is_held(vcpu))
        };
        let resumes = self.debugger.as_ref().map_or(0, |link| link.resumes());
        let stop = |kind| {
            Some(BeforeStep::Debug(StopEvent {
                vcpu,
                kind,
                resumes,
            }))
        };
        if pause {
            return stop(StopKind::Paused);
        }
        if let Some((kind, address)) = self.gdb.watch_hit.take() {
            return stop(StopKind::Watch(kind, address));
        }
        if core::mem::take(&mut self.gdb.stepped) {
            return stop(StopKind::Step);
        }
        if held {
            if self.threads.is_some() {
                return Some(self.wait_while_held());
            }
            if self.multi {
                return Some(BeforeStep::Yield);
            }
        }
        // A halted vCPU is not about to execute RIP; it hits the breakpoint once it is woken and
        // returns there.
        if !self.halted
            && !self.gdb.breakpoints.is_empty()
            && let Ok(linear) = self.fetch_linear()
            && self.gdb.breakpoints.binary_search(&linear).is_ok()
        {
            return stop(StopKind::Breakpoint(linear));
        }
        None
    }

    /// While single-stepping, interrupts and NMIs wait (QEMU's SSTEP_NOIRQ): a step lands on
    /// the next instruction, not in a timer handler. A halted vCPU still takes them, or the step
    /// would never end.
    pub(super) fn gdb_holds_interrupts(&self) -> bool {
        self.gdb.step && !self.halted
    }

    /// `step_debug` is about to run an instruction (or deliver the #DB it owes).
    pub(super) fn gdb_note_step(&mut self) {
        if self.gdb.step {
            self.gdb.step = false;
            self.gdb.stepped = true;
        }
    }

    /// A data access of `size` bytes at `linear` by the instruction `step_debug` runs.
    pub(super) fn gdb_watch(&mut self, linear: u64, size: u64, write: bool) {
        if let Some(watchpoint) = self
            .gdb
            .watchpoints
            .iter()
            .find(|watchpoint| watchpoint.matches(linear, size, write))
        {
            self.gdb
                .watch_hit
                .get_or_insert((watchpoint.kind, watchpoint.address));
        }
    }

    fn register_bytes(&self, tpr: u8, register: usize) -> Option<Vec<u8>> {
        let width = (REGISTERS.get(register)?.bits / 8) as usize;
        let value: u128 = match register {
            _ if REG_GPRS.contains(&register) => self.gpr[GDB_GPR_ORDER[register]] as u128,
            REG_RIP => self.rip as u128,
            REG_EFLAGS => self.eflags_full() as u128,
            REG_SEGMENTS..REG_ST0 => {
                self.seg[GDB_SEGMENT_ORDER[register - REG_SEGMENTS]].sel as u128
            }
            REG_ST0..REG_FCTRL => {
                let physical = (self.x87_top() + register - REG_ST0) % X87_REGISTERS;
                return Some(self.st[physical].to_vec());
            }
            REG_FCTRL => self.fcw as u128,
            REG_FSTAT => self.fsw as u128,
            REG_FTAG => self.full_tag_word() as u128,
            REG_FPU_POINTERS..=REG_FOP => 0,
            REG_XMM0..REG_MXCSR => self.xmm[register - REG_XMM0],
            REG_MXCSR => self.mxcsr as u128,
            REG_FS_BASE => self.seg[SEG_FS].base as u128,
            REG_GS_BASE => self.seg[SEG_GS].base as u128,
            REG_KERNEL_GS_BASE => self.kgsbase as u128,
            REG_CR0 => self.cr0 as u128,
            REG_CR2 => self.cr2 as u128,
            REG_CR3 => self.cr3 as u128,
            REG_CR4 => self.cr4 as u128,
            REG_CR8 => (tpr >> CR8_TPR_SHIFT) as u128,
            REG_EFER => self.efer as u128,
            _ => return None,
        };
        Some(value.to_le_bytes()[..width].to_vec())
    }

    /// Writes what gdb may change without the CPU having to react: the general registers, RIP,
    /// the flags, x87 and SSE state, the segment bases and CR2. The others (selectors, which
    /// would need a descriptor load, and the control registers, which change paging) are
    /// refused unless the value is the one they hold, so a `G` packet of read-back values works.
    fn write_register_bytes(&mut self, tpr: u8, register: usize, bytes: &[u8]) -> bool {
        let Some(current) = self.register_bytes(tpr, register) else {
            return false;
        };
        if bytes.len() != current.len() {
            return false;
        }
        let mut wide = [0u8; 16];
        wide[..bytes.len()].copy_from_slice(bytes);
        let value = u128::from_le_bytes(wide);
        let value64 = value as u64;
        match register {
            _ if REG_GPRS.contains(&register) => self.gpr[GDB_GPR_ORDER[register]] = value64,
            REG_RIP => self.rip = value64,
            REG_EFLAGS => {
                // VM would switch to virtual-8086 mode without the mode change that goes with it.
                if (value64 ^ self.eflags_full()) & flags::VM != 0
                    || value64 & !(EFLAGS_DEFINED | flags::RESERVED_ONE) != 0
                {
                    return false;
                }
                self.eflags = value64 | flags::RESERVED_ONE;
                self.lazy_op = LazyFlagsOp::Eflags;
            }
            REG_ST0..REG_FCTRL => {
                let physical = (self.x87_top() + register - REG_ST0) % X87_REGISTERS;
                self.st[physical].copy_from_slice(bytes);
            }
            REG_FCTRL => self.fcw = value as u16,
            REG_FSTAT => self.fsw = value as u16,
            REG_FTAG => {
                // The CPU keeps the abridged form: a register is in use unless its tag is empty.
                self.ftw = (0..X87_REGISTERS)
                    .filter(|&index| (value as u16 >> (TAG_BITS * index)) & TAG_MASK != TAG_EMPTY)
                    .fold(0, |tags, index| tags | (1 << index));
            }
            REG_XMM0..REG_MXCSR => self.xmm[register - REG_XMM0] = value,
            // `ldmxcsr` would raise #GP for these; the debugger is refused instead.
            REG_MXCSR if value as u32 & !fpu::MXCSR_VALID_BITS != 0 => return false,
            REG_MXCSR => self.mxcsr = value as u32,
            REG_FS_BASE => self.seg[SEG_FS].base = value64,
            REG_GS_BASE => self.seg[SEG_GS].base = value64,
            REG_KERNEL_GS_BASE => self.kgsbase = value64,
            REG_CR2 => self.cr2 = value64,
            _ => return bytes == current.as_slice(),
        }
        true
    }

    fn x87_top(&self) -> usize {
        ((self.fsw >> FSW_TOP_SHIFT) as usize) % X87_REGISTERS
    }

    /// The full tag word gdb shows, from the abridged one the CPU keeps and the register
    /// contents (the `fxsave` to `fsave` conversion, SDM 10.5.1.1).
    fn full_tag_word(&self) -> u16 {
        (0..X87_REGISTERS).fold(0, |word, physical| {
            let tag = if self.ftw & (1 << physical) == 0 {
                TAG_EMPTY
            } else {
                let bytes = &self.st[physical];
                let mantissa = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                let exponent = u16::from_le_bytes([bytes[8], bytes[9]]) & X87_EXPONENT_MASK;
                match exponent {
                    0 if mantissa == 0 => TAG_ZERO,
                    0 | X87_EXPONENT_MASK => TAG_SPECIAL,
                    _ if mantissa & X87_INTEGER_BIT == 0 => TAG_SPECIAL,
                    _ => TAG_VALID,
                }
            };
            word | (tag << (TAG_BITS * physical))
        })
    }
}

/// The local APIC task priority of vCPU `index`, for CR8.
fn task_priority(vcpus: &mut dyn Vcpus, index: usize) -> u8 {
    vcpus.machine().dev.lock().lapic_of(index).tpr()
}

/// Register `register` of vCPU `index`, little-endian, `None` for no such register or vCPU.
pub(crate) fn read_register(
    vcpus: &mut dyn Vcpus,
    index: usize,
    register: usize,
) -> Option<Vec<u8>> {
    let tpr = if register == REG_CR8 {
        task_priority(vcpus, index)
    } else {
        0
    };
    vcpus.cpu(index)?.register_bytes(tpr, register)
}

/// Every register of vCPU `index` in order: the `g` packet.
pub(crate) fn read_registers(vcpus: &mut dyn Vcpus, index: usize) -> Option<Vec<u8>> {
    let tpr = task_priority(vcpus, index);
    let cpu = vcpus.cpu(index)?;
    let mut bytes = Vec::new();
    for register in 0..REGISTER_COUNT {
        bytes.extend(cpu.register_bytes(tpr, register)?);
    }
    Some(bytes)
}

pub(crate) fn write_register(
    vcpus: &mut dyn Vcpus,
    index: usize,
    register: usize,
    bytes: &[u8],
) -> bool {
    let tpr = if register == REG_CR8 {
        task_priority(vcpus, index)
    } else {
        0
    };
    vcpus
        .cpu(index)
        .is_some_and(|cpu| cpu.write_register_bytes(tpr, register, bytes))
}

/// The `G` packet: every register in order. All or nothing: on a refusal nothing changes.
pub(crate) fn write_registers(vcpus: &mut dyn Vcpus, index: usize, bytes: &[u8]) -> bool {
    let tpr = task_priority(vcpus, index);
    let Some(cpu) = vcpus.cpu(index) else {
        return false;
    };
    let expected: usize = REGISTERS.iter().map(|r| (r.bits / 8) as usize).sum();
    if bytes.len() != expected {
        return false;
    }
    // Refusals are decided against the state before any write, so check every register first.
    let mut offset = 0;
    let mut writes = Vec::new();
    for (register, info) in REGISTERS.iter().enumerate() {
        let width = (info.bits / 8) as usize;
        writes.push((register, &bytes[offset..offset + width]));
        offset += width;
    }
    let flags_before = cpu.eflags_full();
    let ok = writes.iter().all(|&(register, value)| match register {
        REG_EFLAGS => {
            (u32::from_le_bytes(value.try_into().unwrap()) as u64 ^ flags_before) & flags::VM == 0
        }
        _ => {
            register_is_writable(register)
                || cpu.register_bytes(tpr, register).as_deref() == Some(value)
        }
    });
    if !ok {
        return false;
    }
    for (register, value) in writes {
        cpu.write_register_bytes(tpr, register, value);
    }
    true
}

/// What `write_register_bytes` changes to any value (EFLAGS aside, which keeps VM).
fn register_is_writable(register: usize) -> bool {
    matches!(
        register,
        0..=REG_RIP | REG_ST0..=REG_FTAG | REG_XMM0..=REG_KERNEL_GS_BASE | REG_CR2
    )
}

/// Up to `length` bytes of guest memory at linear address `address` as vCPU `index` sees it
/// through its page tables: fewer (maybe none) where the mapping or RAM ends. Only RAM is read,
/// never device registers, whose reads have side effects.
pub(crate) fn read_memory(
    vcpus: &mut dyn Vcpus,
    index: usize,
    address: u64,
    length: usize,
) -> Vec<u8> {
    let Some(paging) = vcpus.cpu(index).map(|cpu| cpu.debug_paging()) else {
        return Vec::new();
    };
    let machine = vcpus.machine();
    let mut bytes = Vec::with_capacity(length);
    while bytes.len() < length {
        let linear = address.wrapping_add(bytes.len() as u64);
        let Some(physical) = machine.debug_translate(&paging, linear) else {
            break;
        };
        let in_page = (PAGE_SIZE - (linear & PAGE_OFFSET_MASK)) as usize;
        let chunk = in_page.min(length - bytes.len());
        let Some(source) = machine.ram.slice(physical, chunk) else {
            break;
        };
        if machine.ordered_memory {
            // A device thread may be writing it: byte-wise atomic reads, as the guest's own.
            bytes.extend(
                source
                    .iter()
                    .map(|byte| unsafe { atomic::load(byte as *const u8, 1) } as u8),
            );
        } else {
            bytes.extend_from_slice(source);
        }
    }
    bytes
}

/// Writes `data` at linear address `address` through vCPU `index`'s page tables, dropping
/// translated code of every page it changes. All or nothing: `false`, and nothing written, if
/// any byte is unmapped or outside RAM.
pub(crate) fn write_memory(vcpus: &mut dyn Vcpus, index: usize, address: u64, data: &[u8]) -> bool {
    let Some(paging) = vcpus.cpu(index).map(|cpu| cpu.debug_paging()) else {
        return false;
    };
    let machine = vcpus.machine();
    let mut pieces = Vec::new();
    let mut done = 0;
    while done < data.len() {
        let linear = address.wrapping_add(done as u64);
        let Some(physical) = machine.debug_translate(&paging, linear) else {
            return false;
        };
        let in_page = (PAGE_SIZE - (linear & PAGE_OFFSET_MASK)) as usize;
        let chunk = in_page.min(data.len() - done);
        if !machine.ram.contains(physical, chunk as u64) {
            return false;
        }
        pieces.push((physical as usize, done, chunk));
        done += chunk;
    }
    for (physical, offset, chunk) in pieces {
        let target = &mut machine.ram[physical..physical + chunk];
        if machine.ordered_memory {
            for (byte, &value) in target.iter_mut().zip(&data[offset..offset + chunk]) {
                unsafe { atomic::store(byte as *mut u8, 1, value as u64) };
            }
            // Store, then look at the flag, as `phys_write` does.
            fence(Ordering::SeqCst);
        } else {
            target.copy_from_slice(&data[offset..offset + chunk]);
        }
        let page = physical >> PAGE_SHIFT;
        if machine.page_has_code.get(page) != 0 {
            machine.code_written(page);
        }
    }
    true
}

/// What `qThreadExtraInfo` says about vCPU `index`.
pub(crate) fn describe(vcpus: &mut dyn Vcpus, index: usize) -> String {
    match vcpus.cpu(index) {
        Some(cpu) if cpu.wait_for_sipi => format!("CPU#{index} [waiting for start-up]"),
        Some(cpu) if cpu.halted => format!("CPU#{index} [halted]"),
        Some(_) => format!("CPU#{index}"),
        None => format!("CPU#{index} [not running]"),
    }
}

pub(crate) fn set_rip(vcpus: &mut dyn Vcpus, index: usize, rip: u64) {
    if let Some(cpu) = vcpus.cpu(index) {
        cpu.rip = rip;
    }
}

/// Lets the machine run again: every vCPU gets the debugger's current breakpoints and
/// watchpoints and its own part of the resume (`actions[index]`).
pub(crate) fn apply_resume(
    vcpus: &mut dyn Vcpus,
    link: &DebugLink,
    config: &DebugConfig,
    actions: &[ResumeAction],
) {
    for (index, &action) in actions.iter().enumerate().take(vcpus.count()) {
        link.set_held(index, action == ResumeAction::Hold);
        let Some(cpu) = vcpus.cpu(index) else {
            continue;
        };
        cpu.gdb.load(config);
        cpu.gdb.step = action == ResumeAction::Step;
        cpu.gdb.stepped = false;
        cpu.gdb.watch_hit = None;
        cpu.gdb.armed = cpu.gdb.has_points() || action != ResumeAction::Continue;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{CODE_BASE, Harness, SCRATCH, Stop};

    /// DR7 L0, and R/W 00 (execute) with LEN 00 in its control field.
    const DR7_L0_EXECUTE: u64 = 1;
    const DR6_B0: u64 = 1;

    #[test]
    fn the_guests_debug_registers_work_beside_the_debuggers_watchpoint() {
        // mov [SCRATCH], rax; nop; hlt
        let code = [0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x40, 0x00, 0x90, 0xf4];
        let mut h = Harness::new(&code);
        h.cpu.gdb.watchpoints = vec![Watchpoint {
            address: SCRATCH,
            length: 8,
            kind: WatchKind::Write,
        }];
        h.cpu.gdb.armed = true;
        // The guest breaks on the nop.
        h.cpu.dr[0] = CODE_BASE + 8;
        h.cpu.dr[7] |= DR7_L0_EXECUTE;
        let dr7 = h.cpu.dr[7];
        assert!(matches!(h.run(10), Stop::Fault { vector: 1, .. }));
        assert_eq!(h.cpu.rip, CODE_BASE + 8);
        assert_eq!(h.cpu.dr[6] & DR6_B0, DR6_B0, "the guest's breakpoint fired");
        assert_eq!(h.cpu.dr[7], dr7, "and its DR7 is its own");
        assert_eq!(
            h.cpu.gdb.watch_hit,
            Some((WatchKind::Write, SCRATCH)),
            "the debugger saw the store"
        );
    }

    const PRESENT_WRITABLE: u64 = 3;
    const LARGE_PAGE: u64 = 1 << 7;

    fn translate(h: &Harness, linear: u64) -> Option<u64> {
        h.cpu.debug_translate(&h.cpu.debug_paging(), linear)
    }

    #[test]
    fn the_debugger_walks_with_paging_off() {
        let mut h = Harness::new(&[0xf4]);
        h.cpu.cr0 &= !CR0_PG;
        h.cpu.efer &= !EFER_LMA;
        assert_eq!(translate(&h, 0x1234_5678_9abc), Some(0x5678_9abc));
    }

    #[test]
    fn the_debugger_walks_32_bit_paging_with_4_mib_pages() {
        let mut h = Harness::new(&[0xf4]);
        let directory = SCRATCH;
        let table = SCRATCH + 0x1000;
        h.write(
            directory + 4,
            &(0x80_0000u32 | (LARGE_PAGE | PRESENT_WRITABLE) as u32).to_le_bytes(),
        );
        h.write(
            directory + 8,
            &((table | PRESENT_WRITABLE) as u32).to_le_bytes(),
        );
        h.write(
            table + 4 * 5,
            &(((SCRATCH + 0x2000) | PRESENT_WRITABLE) as u32).to_le_bytes(),
        );
        h.cpu.efer &= !EFER_LMA;
        h.cpu.cr4 = (h.cpu.cr4 & !(CR4_PAE | sys::CR4_LA57)) | sys::CR4_PSE;
        h.cpu.cr3 = directory;
        assert_eq!(translate(&h, 0x40_1234), Some(0x80_1234));
        assert_eq!(translate(&h, 0x80_5678), Some(SCRATCH + 0x2678));
        assert_eq!(translate(&h, 0x80_6000), None);
        assert_eq!(translate(&h, 0xc0_0000), None);
    }

    #[test]
    fn the_debugger_walks_pae_paging_from_the_latched_pdptes() {
        let mut h = Harness::new(&[0xf4]);
        let directory = SCRATCH + 0x3000;
        h.write_u64(directory + 8 * 3, 0xa0_0000 | LARGE_PAGE | PRESENT_WRITABLE);
        h.cpu.efer &= !EFER_LMA;
        h.cpu.cr4 = (h.cpu.cr4 & !sys::CR4_LA57) | CR4_PAE;
        h.cpu.pdpte = [directory | 1, 0, 0, 0];
        assert_eq!(translate(&h, 0x60_1234), Some(0xa0_1234));
        assert_eq!(translate(&h, 0x4000_0000), None, "PDPTE 1 is not present");
    }

    #[test]
    fn the_debugger_walks_five_levels_with_la57() {
        let mut h = Harness::new(&[0xf4]);
        const LINEAR: u64 = 1 << 48;
        let tables: Vec<u64> = (0..5)
            .map(|level| SCRATCH + 0x4000 + 0x1000 * level)
            .collect();
        // PML5 entry 1 (bit 48), then entry 0 at every level below.
        h.write_u64(tables[0] + 8, tables[1] | PRESENT_WRITABLE);
        for level in 1..4 {
            h.write_u64(tables[level], tables[level + 1] | PRESENT_WRITABLE);
        }
        h.write_u64(tables[4], (SCRATCH + 0x9000) | PRESENT_WRITABLE);
        h.cpu.cr4 |= sys::CR4_LA57;
        h.cpu.cr3 = tables[0];
        assert_eq!(translate(&h, LINEAR | 0x10), Some(SCRATCH + 0x9010));
        assert_eq!(translate(&h, 0x10), None);
    }

    #[test]
    fn a_step_holds_interrupts_only_while_the_vcpu_runs() {
        let mut h = Harness::new(&[0x90, 0xf4]);
        h.cpu.gdb.step = true;
        assert!(h.cpu.gdb_holds_interrupts());
        h.cpu.halted = true;
        assert!(!h.cpu.gdb_holds_interrupts());
        h.cpu.halted = false;
        h.cpu.gdb.armed = true;
        assert_eq!(h.run(1), Stop::Limit);
        assert!(!h.cpu.gdb.step && h.cpu.gdb.stepped);
    }
}
