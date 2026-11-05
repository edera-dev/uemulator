//! Test-only support shared by the unit tests: a builder for tiny fake bzImages, and a harness
//! that runs hand-assembled x86_64 snippets on the interpreter and (where the host has one) the
//! JIT. Nothing here touches the network or any cached kernel; every test builds what it needs.
//!
//! Getting bytes for a snippet: `echo 'mov rax, 5; hlt' | hack/asm.sh`.

// A toolbox: not every helper is used by every build of the tests.
#![allow(dead_code)]

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::boot::{self, BootState, KERNEL_LOAD};
use crate::console::ConsoleHandle;
use crate::cpu::Cpu;
use crate::cpu::exception::Exception;
use crate::dev::DevConfig;
use crate::machine::{ClockMode, EmulatedMachine, ExitReason, JitMode, MachineConfig};

pub(crate) const RAX: usize = 0;
pub(crate) const RCX: usize = 1;
pub(crate) const RDX: usize = 2;
pub(crate) const RBX: usize = 3;
pub(crate) const RSP: usize = 4;
pub(crate) const RBP: usize = 5;
pub(crate) const RSI: usize = 6;
pub(crate) const RDI: usize = 7;
pub(crate) const R8: usize = 8;
pub(crate) const R9: usize = 9;
pub(crate) const R10: usize = 10;
pub(crate) const R11: usize = 11;

pub(crate) const HLT: u8 = 0xf4;

/// Guest RAM of a harness CPU. The fake kernel loads at `KERNEL_LOAD` (16 MiB), so this must be
/// larger; the boot identity map covers all of it.
pub(crate) const HARNESS_RAM_BYTES: usize = 32 << 20;
/// Where `Harness::new` places the snippet: the 64-bit entry point of the fake kernel.
pub(crate) const CODE_BASE: u64 = KERNEL_LOAD + FAKE_ENTRY_OFFSET;
/// Initial `rsp`. Grows down through identity-mapped RAM that nothing else uses.
pub(crate) const STACK_TOP: u64 = 0x80_0000;
/// Where `Harness::real` puts the snippet (0000:7c00) and the top of its stack.
pub(crate) const REAL_CODE: u64 = 0x7c00;
pub(crate) const REAL_STACK: u64 = 0x7000;
/// 1 MiB of RAM (physical == linear) that snippets may use freely for data.
pub(crate) const SCRATCH: u64 = 0x40_0000;
pub(crate) const SCRATCH_BYTES: u64 = 1 << 20;

// The 64-bit boot protocol enters at payload offset 0x200 (`startup_64`).
const FAKE_ENTRY_OFFSET: u64 = 0x200;
const SECTOR: usize = 512;

/// Builds a minimal bzImage that `boot::load` accepts. The 64-bit entry point holds
/// `entry_code`; every other field of the setup header can be overridden to exercise the loader's
/// validation. Defaults describe a relocatable 64-bit kernel speaking protocol 2.15.
#[derive(Clone)]
pub(crate) struct FakeKernel {
    pub entry_code: Vec<u8>,
    pub setup_sects: u8,
    pub version: u16,
    pub loadflags: u8,
    pub xloadflags: u16,
    pub relocatable: bool,
    pub pref_address: u64,
    pub init_size: u32,
    pub cmdline_max: u32,
    pub initrd_addr_max: u32,
    /// Total protected-mode payload length; defaults to just enough for `entry_code`.
    pub payload_len: Option<usize>,
    /// Value of the header's `jump` length byte (how much of boot_params the loader copies).
    pub jump_length: u8,
}

impl FakeKernel {
    pub fn new(entry_code: &[u8]) -> Self {
        FakeKernel {
            entry_code: entry_code.to_vec(),
            setup_sects: 1,
            version: 0x020f,
            loadflags: 1,  // LOADED_HIGH
            xloadflags: 1, // XLF_KERNEL_64
            relocatable: true,
            pref_address: KERNEL_LOAD,
            init_size: 0,
            cmdline_max: 2048,
            initrd_addr_max: 0x7fff_ffff,
            payload_len: None,
            // Header ends at 0x202 + 0x6a = 0x26c, past `handover_offset`.
            jump_length: 0x6a,
        }
    }

    pub fn build(&self) -> Vec<u8> {
        let payload_len = self
            .payload_len
            .unwrap_or(FAKE_ENTRY_OFFSET as usize + self.entry_code.len().max(1));
        let protected_offset = (self.setup_sects as usize + 1) * SECTOR;
        let mut image = vec![0u8; protected_offset + payload_len];
        image[0x1f1] = self.setup_sects;
        image[0x200] = 0xeb; // jmp short
        image[0x201] = self.jump_length;
        image[0x202..0x206].copy_from_slice(b"HdrS");
        image[0x206..0x208].copy_from_slice(&self.version.to_le_bytes());
        image[0x211] = self.loadflags;
        image[0x22c..0x230].copy_from_slice(&self.initrd_addr_max.to_le_bytes());
        image[0x234] = self.relocatable as u8;
        image[0x236..0x238].copy_from_slice(&self.xloadflags.to_le_bytes());
        image[0x238..0x23c].copy_from_slice(&self.cmdline_max.to_le_bytes());
        image[0x258..0x260].copy_from_slice(&self.pref_address.to_le_bytes());
        let init_size = if self.init_size != 0 {
            self.init_size
        } else {
            payload_len as u32
        };
        image[0x260..0x264].copy_from_slice(&init_size.to_le_bytes());
        let entry = protected_offset + FAKE_ENTRY_OFFSET as usize;
        if entry < image.len() {
            let end = (entry + self.entry_code.len()).min(image.len());
            image[entry..end].copy_from_slice(&self.entry_code[..end - entry]);
        }
        image
    }
}

/// Shorthand for a default fake kernel whose entry code is `entry_code`.
pub(crate) fn fake_bzimage(entry_code: &[u8]) -> Vec<u8> {
    FakeKernel::new(entry_code).build()
}

/// Why `Harness::run` returned.
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum Stop {
    /// The snippet executed `hlt`.
    Halted,
    /// An instruction raised an exception. It is not delivered, so `rip` and the register file
    /// are exactly what the fault left behind.
    Fault {
        vector: u8,
        error_code: Option<u32>,
        software: bool,
    },
    /// The emulator asked to stop the run loop (reset, shutdown, ...).
    Exit(String),
    /// `max_insns` instructions retired without halting.
    Limit,
}

/// A CPU in 64-bit long mode with identity-mapped RAM, loaded with a snippet at `CODE_BASE`
/// through the real loader (`boot::load`), so tests also exercise that path.
pub(crate) struct Harness {
    pub cpu: Cpu,
    pub console: ConsoleHandle,
    /// Held from `enable_jit` until the harness is dropped, so on Rosetta (see
    /// `jit::serialize_jit_tests`) a JIT-enabled harness never allocates, runs and frees its code
    /// buffer concurrently with another test's. Declared last: it must outlive `cpu`'s `Jit`.
    _jit_turn: Option<crate::jit::JitTestGuard>,
}

impl Harness {
    pub fn new(code: &[u8]) -> Self {
        Self::with_devices(code, DevConfig::default())
    }

    pub fn with_devices(code: &[u8], devices: DevConfig) -> Self {
        let mut ram = vec![0u8; HARNESS_RAM_BYTES];
        let state: BootState =
            boot::load(&mut ram, &fake_bzimage(code), None, "").expect("fake kernel must load");
        let console = ConsoleHandle::default();
        let mut cpu = Cpu::new(
            ram,
            state,
            console.clone(),
            Arc::new(AtomicBool::new(false)),
            None,
            devices,
        );
        cpu.gpr[RSP] = STACK_TOP;
        Harness {
            cpu,
            console,
            _jit_turn: None,
        }
    }

    /// A real-mode CPU (paging and long mode off) running `code` at 0000:7c00 with the stack at
    /// 0000:7000 and every data segment 0.
    pub fn real(code: &[u8]) -> Self {
        Self::real_with_devices(code, DevConfig::default())
    }

    /// `real`, with devices (a disk, say) attached.
    pub fn real_with_devices(code: &[u8], devices: DevConfig) -> Self {
        let mut harness = Self::with_devices(&[HLT], devices);
        harness.cpu.reset_to_real_mode();
        harness.write(REAL_CODE, code);
        let code_segment = harness.cpu.seg[crate::cpu::SEG_CS];
        harness.cpu.seg[crate::cpu::SEG_CS] = crate::cpu::Seg {
            sel: 0,
            base: 0,
            ..code_segment
        };
        harness.cpu.rip = REAL_CODE;
        harness.cpu.gpr[RSP] = REAL_STACK;
        harness
    }

    /// Switches this CPU to the host JIT. Returns false when the host has none (non-aarch64 and
    /// non-x86_64 hosts, or a build without the `jit` feature); the harness then stays
    /// interpreted.
    pub fn enable_jit(&mut self) -> bool {
        self._jit_turn = Some(crate::jit::serialize_jit_tests());
        self.cpu.jit = crate::jit::Jit::new(true).map(Box::new);
        self.cpu.jit.is_some()
    }

    pub fn write(&mut self, addr: u64, bytes: &[u8]) {
        let start = addr as usize;
        self.cpu.ram[start..start + bytes.len()].copy_from_slice(bytes);
    }

    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.cpu.ram[addr as usize..addr as usize + len].to_vec()
    }

    pub fn write_u64(&mut self, addr: u64, value: u64) {
        self.write(addr, &value.to_le_bytes());
    }

    pub fn read_u64(&self, addr: u64) -> u64 {
        u64::from_le_bytes(self.read(addr, 8).try_into().unwrap())
    }

    /// Runs like `Cpu::run` minus device servicing and idling: one instruction (or JIT block) per
    /// iteration, with the same instruction-count bookkeeping. Exceptions stop the run
    /// undelivered. After a `Fault`, `run` may be called again only after the test repairs state.
    pub fn run(&mut self, max_insns: u64) -> Stop {
        let start = self.cpu.icount;
        // Chained JIT blocks keep running until `icount` reaches this, so the budget bounds
        // them the way the event tick bounds them in `Cpu::run`.
        self.cpu.next_event = start + max_insns;
        while self.cpu.icount - start < max_insns {
            self.cpu.interrupt_shadow = false;
            self.cpu.icount += 1;
            let result = if self.cpu.jit.is_some() && self.cpu.is_long64() {
                self.cpu.step_block()
            } else {
                self.cpu.step()
            };
            match result {
                Ok(()) => {
                    if self.cpu.halted {
                        return Stop::Halted;
                    }
                }
                Err(Exception::Fault {
                    vector,
                    error_code,
                    software,
                }) => {
                    return Stop::Fault {
                        vector,
                        error_code,
                        software,
                    };
                }
                Err(Exception::Exit(reason)) => return Stop::Exit(format!("{reason:?}")),
                // an SVM #VMEXIT switched back to the host; execution simply continues there
                Err(Exception::Vmexit) => {}
                Err(Exception::AtomicRetry) => unreachable!("exec retries locked instructions"),
            }
        }
        Stop::Limit
    }

    /// Like `run`, but exceptions and software interrupts are delivered through the IDT/IVT as the
    /// run loop would, and execution continues.
    pub fn run_delivering(&mut self, max_insns: u64) -> Stop {
        loop {
            match self.run(max_insns) {
                Stop::Fault {
                    vector,
                    error_code,
                    software,
                } => {
                    if let Some(reason) = self.cpu.deliver_event(vector, error_code, software) {
                        return Stop::Exit(format!("{reason:?}"));
                    }
                }
                other => return other,
            }
        }
    }

    /// Runs to `hlt`, panicking with the stop reason and a register dump otherwise.
    pub fn run_to_halt(&mut self) -> &mut Self {
        let stop = self.run(1_000_000);
        assert_eq!(stop, Stop::Halted, "{}", self.cpu.dump_regs());
        self
    }
}

/// Final architectural state compared between two runs of the same snippet.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub eflags: u64,
    pub xmm: [u128; 16],
    /// Bits 255:128 and 511:256 of ZMM0-15, ZMM16-31 and the opmask registers.
    pub ymm_hi: [u128; 16],
    pub zmm_hi: [[u128; 2]; 16],
    pub zmm_extra: [[u128; 4]; 16],
    pub opmask: [u64; 8],
    pub mxcsr: u32,
    pub fcw: u16,
    pub fsw: u16,
    pub ftw: u16,
    pub st: [[u8; 10]; 8],
    pub segment_selectors: [u16; 6],
    pub cpl: u8,
    pub icount: u64,
}

impl Harness {
    pub fn snapshot(&self) -> Snapshot {
        let cpu = &self.cpu;
        Snapshot {
            gpr: cpu.gpr,
            rip: cpu.rip,
            eflags: cpu.eflags_full(),
            xmm: cpu.xmm,
            ymm_hi: cpu.ymm_hi,
            zmm_hi: cpu.zmm_hi,
            zmm_extra: cpu.zmm_extra,
            opmask: cpu.opmask,
            mxcsr: cpu.mxcsr,
            fcw: cpu.fcw,
            fsw: cpu.fsw,
            ftw: cpu.ftw,
            st: cpu.st,
            segment_selectors: std::array::from_fn(|i| cpu.seg[i].sel),
            cpl: cpu.cpl,
            icount: cpu.icount,
        }
    }
}

/// Outcome of running one snippet on both engines.
pub(crate) struct Differential {
    pub interpreter: Harness,
    pub interpreter_stop: Stop,
    /// `None` when the host has no JIT.
    pub jit: Option<(Harness, Stop)>,
}

/// Runs `code` from identical starting states on the interpreter (the oracle) and, if the host
/// has one, the JIT. `setup` prepares registers and memory and is applied to both.
pub(crate) fn run_both(code: &[u8], setup: impl Fn(&mut Harness)) -> Differential {
    let mut interpreter = Harness::new(code);
    setup(&mut interpreter);
    let interpreter_stop = interpreter.run(1_000_000);

    let mut jitted = Harness::new(code);
    setup(&mut jitted);
    let jit = jitted.enable_jit().then(|| {
        let stop = jitted.run(1_000_000);
        (jitted, stop)
    });
    Differential {
        interpreter,
        interpreter_stop,
        jit,
    }
}

/// Asserts the JIT leaves exactly the state the interpreter does (registers, flags, vector and
/// x87 state, instruction count, and all of RAM), and stops for the same reason. Passes trivially
/// on hosts without a JIT.
#[track_caller]
pub(crate) fn assert_jit_matches_interpreter(code: &[u8], setup: impl Fn(&mut Harness)) {
    let result = run_both(code, setup);
    let Some((jit, jit_stop)) = result.jit else {
        return;
    };
    assert_eq!(
        jit_stop, result.interpreter_stop,
        "stop reason differs (jit vs interpreter)"
    );
    assert_eq!(
        jit.snapshot(),
        result.interpreter.snapshot(),
        "state differs (jit vs interpreter)"
    );
    let (jit_ram, interpreter_ram) = (&jit.cpu.ram[0..], &result.interpreter.cpu.ram[0..]);
    if jit_ram != interpreter_ram {
        let offset = jit_ram
            .iter()
            .zip(interpreter_ram)
            .position(|(a, b)| a != b)
            .unwrap();
        panic!("RAM differs first at {offset:#x} (jit vs interpreter)");
    }
}

/// Console output and exit reason of a fake kernel run through the public API.
pub(crate) struct MachineRun {
    pub console_output: Vec<u8>,
    pub exit: ExitReason,
}

impl MachineRun {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.console_output).into_owned()
    }
}

/// Boots `kernel` on a full `EmulatedMachine` and returns when it exits, or after `limit`
/// (a watchdog then stops it, giving `ExitReason::Stopped`). `configure` adjusts the config.
/// Reboot is off unless `configure` turns it on: most guests end with a reset request, which
/// would otherwise restart them forever.
pub(crate) fn run_machine(
    kernel: Vec<u8>,
    jit: JitMode,
    limit: Duration,
    configure: impl FnOnce(MachineConfig) -> MachineConfig,
) -> MachineRun {
    let config = configure(
        MachineConfig::new()
            // Tests are deterministic and never wait out a guest timer, and the device layout is
            // exactly what a test configures; one that wants the host clock or the entropy
            // device asks for it in `configure`.
            .clock(ClockMode::Virtual)
            .rng(false)
            .kernel(kernel)
            .memory_mib(64)
            .jit(jit)
            .reboot(false),
    );
    let mut machine = EmulatedMachine::new(config).expect("config must be valid");
    let output = Arc::new(Mutex::new(Vec::new()));
    let sink = output.clone();
    machine
        .console()
        .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
    let stop = machine.stop_handle();
    let watchdog_done = Arc::new(AtomicBool::new(false));
    let flag = watchdog_done.clone();
    let watchdog = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + limit;
        while std::time::Instant::now() < deadline {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        stop.stop();
    });
    machine.start().expect("machine must start");
    let exit = machine.wait();
    watchdog_done.store(true, std::sync::atomic::Ordering::Relaxed);
    watchdog.join().unwrap();
    let console_output = output.lock().unwrap().clone();
    MachineRun {
        console_output,
        exit,
    }
}

/// A finished multi-vCPU run: the scheduler (inspect RAM and vCPU state through it) and why it
/// ended.
pub(crate) struct SmpRun {
    pub smp: crate::cpu::smp::Smp,
    pub exit: ExitReason,
    /// Declared last, like `Harness::_jit_turn`: it must outlive the JIT.
    _jit_turn: Option<crate::jit::JitTestGuard>,
}

impl SmpRun {
    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.smp.active().ram[addr as usize..addr as usize + len].to_vec()
    }

    pub fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read(addr, 2).try_into().unwrap())
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read(addr, 4).try_into().unwrap())
    }
}

/// How long a multi-vCPU test may run before a watchdog stops it (a hang then shows up as
/// `ExitReason::Stopped`).
const SMP_WATCHDOG: Duration = Duration::from_secs(30);

/// Runs `code` (as the bootstrap processor's 64-bit snippet, like `Harness`) on a machine with
/// `cpus` vCPUs until it exits. `setup` may place more code and data in RAM first, typically a
/// real-mode trampoline for the others. With `jit` the vCPUs run the host JIT where they can
/// (64-bit code); a host without one runs the interpreter either way.
pub(crate) fn run_smp(code: &[u8], cpus: usize, jit: bool, setup: impl FnOnce(&mut Cpu)) -> SmpRun {
    let mut ram = vec![0u8; HARNESS_RAM_BYTES];
    let state: BootState =
        boot::load(&mut ram, &fake_bzimage(code), None, "").expect("fake kernel must load");
    let stop = Arc::new(AtomicBool::new(false));
    let mut cpu = Cpu::new(
        ram,
        state,
        ConsoleHandle::default(),
        stop.clone(),
        None,
        DevConfig {
            cpus,
            ..Default::default()
        },
    );
    cpu.gpr[RSP] = STACK_TOP;
    let jit_turn = jit.then(crate::jit::serialize_jit_tests);
    if jit {
        cpu.jit = crate::jit::Jit::new(true).map(Box::new);
    }
    setup(&mut cpu);
    let mut smp = crate::cpu::smp::Smp::new(cpu);
    let finished = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let finished = finished.clone();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + SMP_WATCHDOG;
            while std::time::Instant::now() < deadline {
                if finished.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let exit = smp.run();
    finished.store(true, std::sync::atomic::Ordering::Relaxed);
    watchdog.join().unwrap();
    SmpRun {
        smp,
        exit,
        _jit_turn: jit_turn,
    }
}

/// A finished run with one host thread per vCPU: the bootstrap processor (inspect RAM through
/// it; the others share it), a line per vCPU, and why the machine ended.
pub(crate) struct ThreadRun {
    pub bsp: Cpu,
    pub summary: String,
    pub exit: ExitReason,
    /// Declared last, like `SmpRun::_jit_turn`.
    _jit_turn: Option<crate::jit::JitTestGuard>,
}

impl ThreadRun {
    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.bsp.ram[addr as usize..addr as usize + len].to_vec()
    }

    pub fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read(addr, 2).try_into().unwrap())
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read(addr, 4).try_into().unwrap())
    }
}

/// `run_smp` with a host thread per vCPU (`vcpus::run_threaded`) on the host clock.
pub(crate) fn run_threads(
    code: &[u8],
    cpus: usize,
    jit: bool,
    setup: impl FnOnce(&mut Cpu),
) -> ThreadRun {
    run_threads_with(code, cpus, jit, setup, |_| {})
}

/// `run_threads`, with `prepare` given the vCPUs (the bootstrap processor first) before their
/// threads start, to put the others somewhere other than waiting for a start-up IPI.
pub(crate) fn run_threads_with(
    code: &[u8],
    cpus: usize,
    jit: bool,
    setup: impl FnOnce(&mut Cpu),
    prepare: impl FnOnce(&mut [Cpu]),
) -> ThreadRun {
    let mut ram = vec![0u8; HARNESS_RAM_BYTES];
    let state: BootState =
        boot::load(&mut ram, &fake_bzimage(code), None, "").expect("fake kernel must load");
    let stop = Arc::new(AtomicBool::new(false));
    let mut cpu = Cpu::new(
        ram,
        state,
        ConsoleHandle::default(),
        stop.clone(),
        None,
        DevConfig {
            cpus,
            ..Default::default()
        },
    );
    cpu.gpr[RSP] = STACK_TOP;
    cpu.wall_clock = Some(crate::cpu::WallClock::new(crate::cpu::std_host_clock()));
    let jit_turn = jit.then(crate::jit::serialize_jit_tests);
    if jit {
        cpu.jit = crate::jit::Jit::new(true).map(Box::new);
    }
    setup(&mut cpu);
    let finished = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let finished = finished.clone();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + SMP_WATCHDOG;
            while std::time::Instant::now() < deadline {
                if finished.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    // As in `EmulatedMachine`: every vCPU gets a handle on the one code cache.
    let jit_source = cpu.jit.as_ref().map(|jit| jit.sibling());
    let make_jit = || jit_source.as_ref().map(|jit| Box::new(jit.sibling()));
    let mut cpus = crate::cpu::smp::secondary_cpus(cpu);
    prepare(&mut cpus);
    let (exit, bsp, summary) =
        crate::cpu::vcpus::run_threaded(cpus, &crate::cpu::vcpus::std_thread_spawner(), &make_jit);
    finished.store(true, std::sync::atomic::Ordering::Relaxed);
    watchdog.join().unwrap();
    ThreadRun {
        bsp,
        summary,
        exit,
        _jit_turn: jit_turn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boot::SetupHeader;

    // mov rax, 5; add rax, 7; hlt
    const ADD_SNIPPET: [u8; 12] = [
        0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x07, 0xf4,
    ];
    // Writes "hi" to the serial port, then asks the keyboard controller to reset the machine.
    // mov dx,0x3f8; mov al,'h'; out dx,al; mov al,'i'; out dx,al; mov al,0xfe; out 0x64,al; hlt
    const HELLO_RESET: [u8; 15] = [
        0x66, 0xba, 0xf8, 0x03, 0xb0, 0x68, 0xee, 0xb0, 0x69, 0xee, 0xb0, 0xfe, 0xe6, 0x64, 0xf4,
    ];

    #[test]
    fn fake_kernel_parses_and_loads() {
        let image = fake_bzimage(&ADD_SNIPPET);
        let header = SetupHeader::parse(&image).unwrap();
        assert!(header.relocatable);
        let mut ram = vec![0u8; HARNESS_RAM_BYTES];
        let state = boot::load(&mut ram, &image, None, "console=ttyS0").unwrap();
        assert_eq!(state.rip, CODE_BASE);
        assert_eq!(
            &ram[CODE_BASE as usize..][..ADD_SNIPPET.len()],
            ADD_SNIPPET.as_slice()
        );
    }

    #[test]
    fn harness_runs_a_snippet_on_the_interpreter() {
        let mut harness = Harness::new(&ADD_SNIPPET);
        harness.run_to_halt();
        assert_eq!(harness.cpu.gpr[RAX], 12);
        assert_eq!(harness.cpu.rip, CODE_BASE + ADD_SNIPPET.len() as u64);
    }

    #[test]
    fn harness_reports_undelivered_faults() {
        // ud2
        let mut harness = Harness::new(&[0x0f, 0x0b]);
        assert_eq!(
            harness.run(10),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
        assert_eq!(harness.cpu.rip, CODE_BASE);
    }

    #[test]
    fn harness_stops_at_the_instruction_limit() {
        // jmp $
        let mut harness = Harness::new(&[0xeb, 0xfe]);
        assert_eq!(harness.run(100), Stop::Limit);
    }

    #[test]
    fn jit_matches_interpreter_on_a_snippet() {
        assert_jit_matches_interpreter(&ADD_SNIPPET, |_| {});
    }

    #[test]
    fn fake_kernel_boots_on_a_full_machine_and_resets() {
        for jit in [JitMode::Off, JitMode::Auto] {
            let run = run_machine(
                fake_bzimage(&HELLO_RESET),
                jit,
                Duration::from_secs(20),
                |config| config,
            );
            assert_eq!(run.text(), "hi");
            assert!(matches!(run.exit, ExitReason::Reset), "{:?}", run.exit);
        }
    }
}
