#[cfg(freestanding)]
use crate::prelude::*;
use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(freestanding))]
use std::thread::JoinHandle;

use crate::acpi::AcpiConfig;
use crate::boot::{self, BootState};
use crate::console::ConsoleHandle;
use crate::cpu::smp::Smp;
use crate::cpu::vcpus::ThreadSpawner;
use crate::cpu::{Cpu, DEFAULT_RANDOM_SEED};
use crate::dev::host_entropy;
use crate::dev::kv::SharedKeyValueStore;
use crate::dev::{DevConfig, EntropyFn, MAX_VIRTIO_FUNCTIONS, RngSource};
use crate::disk::Disk;
use crate::kv::{KeyValueStore, MemoryKeyValueStore};
use crate::netdev::NetHandle;
use crate::platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE};
use crate::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum Error {
    /// The bzImage is not a Linux x86 boot-protocol image this loader can start.
    BadKernel(String),
    /// A setting is out of range or the guest does not fit in the configured RAM.
    BadConfig(String),
    /// `JitMode::On` was requested but the host cannot run generated code.
    JitUnavailable,
    AlreadyStarted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadKernel(m) => write!(f, "bad kernel image: {m}"),
            Error::BadConfig(m) => write!(f, "bad configuration: {m}"),
            Error::JitUnavailable => write!(f, "JIT requested but unavailable on this host"),
            Error::AlreadyStarted => write!(f, "machine already started"),
        }
    }
}

impl core::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JitMode {
    /// Use the JIT when the host allows it, otherwise run the interpreter.
    #[default]
    Auto,
    Off,
    /// Fail `EmulatedMachine::new` with `JitUnavailable` instead of falling back.
    On,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClockMode {
    /// Guest time is the instruction count (one nanosecond each) plus the time skipped while
    /// halted: with a timer pending the clock jumps straight to its deadline, with none it stands
    /// still while the host naps 1 ms per idle poll. Runs are deterministic, which tests and `verify_jit` need.
    Virtual,
    /// Guest time is the host's monotonic clock, as in QEMU without `-icount`: the TSC, the
    /// timers and the RTC advance with the wall clock whatever the emulation speed, and a halted
    /// guest really waits (up to 2 ms per sleep, so the stop flag and host console stay
    /// responsive; a freestanding host's poll hook does the waiting). Not deterministic. Ignored
    /// on UEFI, and on a freestanding host without `MachineConfig::host_clock`, where the clock
    /// stays virtual, and under `verify_jit`.
    #[default]
    Realtime,
}

/// How the vCPUs of a multi-processor guest are run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmpMode {
    /// One host thread per vCPU when the host can do that: the guest then runs in parallel, on
    /// the host clock (`ClockMode::Realtime`), with `MachineConfig::thread_spawner` or, hosted,
    /// `std::thread`. Otherwise the vCPUs take turns on one thread, as `Timeshare` does.
    #[default]
    Auto,
    /// Always take turns on the calling thread: a few thousand instructions each, deterministic
    /// on the virtual clock, and the only mode on a freestanding host that cannot start threads.
    Timeshare,
    /// Fail `EmulatedMachine::new` instead of falling back when threads are not possible: the
    /// clock is not realtime, `verify_jit` is on, or a freestanding host gave no spawner or no
    /// `host_clock`.
    Threads,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// The guest powered the machine off: it wrote the ACPI sleep register (PM1a_CNT) with
    /// SLP_EN and the S5 type, or it executed `hlt` with interrupts disabled (IF=0), which
    /// nothing can wake. Without ACPI tables (`MachineConfig::acpi(false)`, or a guest that
    /// ignores them) the `hlt` is how Linux's poweroff and halt end (`cli; hlt` in a loop). A
    /// reset request made just before the `hlt` wins and is reported as `Reset`.
    Shutdown,
    /// The guest asked for a reset through the keyboard controller (port 0x64), fast A20 (0x92)
    /// or reset control (0xcf9) port; see `dev/mod.rs`. Only produced with
    /// `MachineConfig::reboot(false)` or under `verify_jit`: by default the machine restarts
    /// itself instead.
    Reset,
    TripleFault,
    /// `stop()` / `StopHandle::stop()`, or `wait()` with nothing running.
    Stopped,
    /// An emulator-side failure, not a guest behavior.
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleKind {
    /// 16550 UART at COM1 (`ttyS0`). There is one UART, so this may be configured once.
    Serial,
    /// virtio-console (`hvc0`); the only kind that delivers `ConsoleHandle::resize` to the guest.
    /// The default, and it may be configured several times (`hvc0`, `hvc1`, ...). The UART is
    /// present either way and carries early boot text (`earlyprintk`), but this needs a kernel
    /// whose `virtio_console` driver is built in (an initrd that only loads it as a module has
    /// no console until then); pick `Serial` for such kernels.
    #[default]
    Virtio,
}

pub const MIN_CPUS: usize = 1;
/// The APIC id fits one byte in xAPIC mode, and the I/O APIC and IPI routing keep one bit per
/// vCPU in a `u64`.
pub const MAX_CPUS: usize = 64;
pub const MIN_MEMORY_MIB: usize = 64;
/// The most RAM the guest can address: its physical address space is 40 bits (CPUID
/// MAXPHYADDR), and the 1 GiB PCI hole below 4 GiB is not RAM. RAM past 3 GiB is placed from
/// 4 GiB up (`ram::flat_bytes`), so the last byte lands at 1 TiB when the hole is subtracted.
/// Memory is committed lazily, so the host needs the RAM only as far as the guest touches it.
pub const MAX_MEMORY_MIB: usize = (1 << (40 - BYTES_PER_MIB_SHIFT)) - HOLE_MIB;
/// The PCI hole in MiB: `MAX_MEMORY_MIB` loses it from the 40-bit address space.
const HOLE_MIB: usize =
    ((crate::ram::PCI_HOLE_END - crate::ram::PCI_HOLE_START) >> BYTES_PER_MIB_SHIFT) as usize;
const DEFAULT_MEMORY_MIB: usize = 512;
const BYTES_PER_MIB_SHIFT: usize = 20;
/// The word `DEFAULT_CMDLINE` carries for hosts without ACPI tables.
const ACPI_OFF_WORD: &str = "acpi=off";

pub(crate) type LogFn = Arc<dyn Fn(&str) + Send + Sync>;
/// The poll hook as the CPU sees it: the console is already bound, the flag means "guest is
/// halted with nothing scheduled, the host may wait".
pub(crate) type PollFn = Arc<dyn Fn(bool) + Send + Sync>;
/// Monotonic host nanoseconds from any fixed starting point (`MachineConfig::host_clock`).
pub(crate) type HostClockFn = Arc<dyn Fn() -> u64 + Send + Sync>;
/// The embedder-facing poll hook of `MachineConfig::poll`, which receives the console itself.
type HostPollFn = Arc<dyn Fn(&ConsoleHandle, bool) + Send + Sync>;

pub struct MachineConfig {
    kernel: Vec<u8>,
    firmware: Option<Vec<u8>>,
    uefi_firmware: Option<Vec<u8>>,
    key_value_store: Option<SharedKeyValueStore>,
    boot_disk: bool,
    initrd: Option<Vec<u8>>,
    cmdline: Option<String>,
    cmdline_extra: Vec<String>,
    memory_mib: usize,
    cpus: usize,
    disks: Vec<Disk>,
    net_macs: Vec<[u8; 6]>,
    consoles: Vec<ConsoleKind>,
    iommu: bool,
    acpi: bool,
    jit: JitMode,
    clock: ClockMode,
    smp: SmpMode,
    thread_spawner: Option<ThreadSpawner>,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    host_clock: Option<HostClockFn>,
    rng: bool,
    entropy: Option<EntropyFn>,
    debug: bool,
    verify: bool,
    unix_time: Option<u64>,
    random_seed: Option<u64>,
    reboot: bool,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            kernel: Vec::new(),
            firmware: None,
            uefi_firmware: None,
            key_value_store: None,
            boot_disk: false,
            initrd: None,
            cmdline: None,
            cmdline_extra: Vec::new(),
            memory_mib: DEFAULT_MEMORY_MIB,
            cpus: MIN_CPUS,
            disks: Vec::new(),
            net_macs: Vec::new(),
            consoles: Vec::new(),
            iommu: false,
            acpi: true,
            jit: JitMode::Auto,
            clock: ClockMode::Realtime,
            smp: SmpMode::Auto,
            thread_spawner: None,
            log: None,
            poll: None,
            host_clock: None,
            rng: true,
            entropy: None,
            debug: false,
            verify: false,
            unix_time: None,
            random_seed: None,
            reboot: true,
        }
    }
}

impl MachineConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, bzimage: impl Into<Vec<u8>>) -> Self {
        self.kernel = bzimage.into();
        self
    }

    /// Boots a firmware image (a BIOS, or any real-mode program) instead of a Linux kernel: the
    /// image is placed so that it ends at 1 MiB and the CPU starts in real mode at F000:FFF0.
    /// No BIOS services exist beyond what the image brings, and a guest reset ends the run.
    pub fn firmware(mut self, image: impl Into<Vec<u8>>) -> Self {
        self.firmware = Some(image.into());
        self
    }

    /// Boots a UEFI firmware image (`crates/uemulator-firmware`) instead of a Linux kernel. The
    /// CPU starts at the image's entry in 64-bit mode with the low 4 GiB identity-mapped,
    /// interrupts off and `rdi` pointing at a handoff page that describes the machine
    /// (`plans/19-uefi-firmware.md`); the firmware finds the disks and boots from them. A guest
    /// reset runs the firmware again. Needs a disk or a kernel to be useful and cannot be
    /// combined with `firmware` or `boot_disk`.
    ///
    /// With a `kernel` (and `initrd`, `cmdline`) the firmware boots that kernel first, as an EFI
    /// application with the command line as its load options and the initramfs behind
    /// `LINUX_EFI_INITRD_MEDIA_GUID`, the way OVMF boots QEMU's `-kernel`; it never falls
    /// through to the disks. The kernel must be an EFI-stub bzImage, else `new` fails.
    pub fn uefi_firmware(mut self, image: impl Into<Vec<u8>>) -> Self {
        self.uefi_firmware = Some(image.into());
        self
    }

    /// The persistent store behind the guest's key-value device, which the UEFI firmware keeps
    /// its variables in (`BootOrder`, `Boot0000`, anything a guest writes with `efivar`).
    /// Without one the device is absent and variables last until the guest resets. The store
    /// outlives guest resets.
    pub fn key_value_store(mut self, store: impl KeyValueStore + 'static) -> Self {
        let store: Box<dyn KeyValueStore> = Box::new(store);
        self.key_value_store = Some(Arc::new(Mutex::new(store)));
        self
    }

    /// Boots the first disk the way a PC with a BIOS does: its first sector is loaded at
    /// 0000:7C00 and run in real mode with DL = 0x80, with the BIOS services a boot loader needs
    /// (INT 10h, 11h, 12h, 13h with the extensions, 15h including E820, 16h, 18h, 19h, 1Ah and a
    /// timer tick) provided by the library. The disk's own boot loader, GRUB or syslinux, then
    /// loads the kernel. Needs a disk and replaces the kernel; a guest reset ends the run.
    pub fn boot_disk(mut self, on: bool) -> Self {
        self.boot_disk = on;
        self
    }

    pub fn initrd(mut self, data: impl Into<Vec<u8>>) -> Self {
        self.initrd = Some(data.into());
        self
    }

    pub fn cmdline(mut self, s: impl Into<String>) -> Self {
        self.cmdline = Some(s.into());
        self
    }

    /// Appended to the command line, whether it is the default or set by `cmdline`.
    pub fn cmdline_append(mut self, s: impl Into<String>) -> Self {
        self.cmdline_extra.push(s.into());
        self
    }

    /// Adds a virtio-blk disk. Call it again for more: the guest sees them in call order
    /// (`vda`, `vdb`, ...).
    pub fn disk(mut self, disk: Disk) -> Self {
        self.disks.push(disk);
        self
    }

    /// Adds a virtio-net device with this MAC; frames flow through `EmulatedMachine::nets()`
    /// (`net()` for the first). Call it again for more NICs, each with its own MAC: the n-th
    /// call is the n-th handle.
    pub fn net(mut self, mac: [u8; 6]) -> Self {
        self.net_macs.push(mac);
        self
    }

    /// Adds an AMD IOMMU (AMD-Vi) as PCI function 00:00.2, and translates the DMA of every PCI
    /// function through it once the guest enables it; I/O APIC, HPET and MSI-X interrupts go
    /// through its interrupt remapping, and the virtio functions offer `ACCESS_PLATFORM` so the
    /// guest maps their buffers. Off by default. The guest finds it through the ACPI IVRS table,
    /// which needs `acpi` (on by default); see `plans/16-iommu.md`.
    pub fn iommu(mut self, on: bool) -> Self {
        self.iommu = on;
        self
    }

    /// Publishes ACPI tables (RSDP, XSDT, FADT, DSDT, MADT, MCFG, HPET, and the IVRS with the
    /// IOMMU) to a Linux guest, which then uses the I/O APIC, the HPET, the PM timer and ACPI
    /// power-off. On by default; the default command line then no longer says `acpi=off`. Off
    /// gives the table-less machine of earlier versions (PIC and PIT only; the guest powers off
    /// with `hlt` and IF=0).
    pub fn acpi(mut self, on: bool) -> Self {
        self.acpi = on;
        self
    }

    /// Adds a guest console. The first one configured is the guest's primary console (the
    /// default command line's `console=`). `Serial` may be added once, `Virtio` as often as
    /// wanted; with none added the machine gets one virtio console. Each console has its own
    /// `ConsoleHandle`, in call order (`EmulatedMachine::consoles()`).
    pub fn console(mut self, kind: ConsoleKind) -> Self {
        self.consoles.push(kind);
        self
    }

    /// Guest RAM, `MIN_MEMORY_MIB` to `MAX_MEMORY_MIB` (default 512). RAM beyond 3 GiB is placed
    /// above 4 GiB, as on a PC: the memory map the guest gets has a hole between them.
    pub fn memory_mib(mut self, n: usize) -> Self {
        self.memory_mib = n;
        self
    }

    /// How many vCPUs the guest sees, `MIN_CPUS` to `MAX_CPUS` (default 1). Each gets its own host
    /// thread and the guest runs in parallel when the host allows it (`SmpMode`); otherwise they
    /// run in turns on the one thread (a few thousand instructions each, or until one halts or
    /// spins), which does not make the guest faster in host time, only lets it use its SMP code
    /// paths. Needs ACPI (the MADT is how the guest finds the others) and cannot be combined with
    /// `verify_jit`, a firmware image or `boot_disk`'s BIOS shim.
    pub fn cpus(mut self, n: usize) -> Self {
        self.cpus = n;
        self
    }

    /// Chooses between a host thread per vCPU and taking turns (see `SmpMode`). Only matters with
    /// more than one vCPU.
    pub fn smp(mut self, mode: SmpMode) -> Self {
        self.smp = mode;
        self
    }

    /// How the machine starts the threads of a multi-processor guest, for a host that is not
    /// `std`: `spawn` runs the body it is given on a new thread of the embedder's choosing, and
    /// the body returns when the guest ends. Hosted builds start `std::thread`s without it; a
    /// freestanding build (UEFI, WebAssembly) takes turns on one thread unless it gets a spawner
    /// and a `host_clock`. The spawned threads share the machine's memory and devices, so the host
    /// needs real parallel threads: WebAssembly needs shared memory and atomics.
    pub fn thread_spawner(
        mut self,
        spawn: impl Fn(Box<dyn FnOnce() + Send + 'static>) + Send + Sync + 'static,
    ) -> Self {
        self.thread_spawner = Some(Arc::new(spawn));
        self
    }

    pub fn jit(mut self, mode: JitMode) -> Self {
        self.jit = mode;
        self
    }

    pub fn clock(mut self, mode: ClockMode) -> Self {
        self.clock = mode;
        self
    }

    pub fn debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }

    /// Runs the JIT in lockstep with a second, interpreter-only CPU and stops at the first block
    /// where their state diverges. Very slow; for JIT development. A guest reset ends the run
    /// with `ExitReason::Reset` whatever `reboot` says. The second CPU has the same RTC
    /// boot time and the same virtio functions in the same slots, but not their data: disk
    /// contents and host console or network input are seen by the main CPU only, so a guest that
    /// reads them will report a divergence. Verification always runs on virtual time, whatever
    /// `clock` says. Without a JIT to check (`JitMode::Off`, or a host that has none) the machine
    /// just interprets.
    #[doc(hidden)]
    pub fn verify_jit(mut self, on: bool) -> Self {
        self.verify = on;
        self
    }

    /// Whether a guest reset request (keyboard controller, port 0x92, port 0xCF9) restarts the
    /// machine in place (the default) or ends the run with `ExitReason::Reset`.
    ///
    /// A restart looks like a cold boot to the guest: the kernel, initrd and command line are
    /// loaded again into freshly zeroed RAM, the CPU and every device start from power-on state,
    /// virtual time starts at 0 and the JIT starts with an empty code cache. What survives is
    /// what the embedder holds or what a real machine keeps: the `ConsoleHandle` and `NetHandle`
    /// (with whatever is queued in them), the stop flag, the disk image and its contents, and the
    /// RTC (re-read from the host clock, or re-applied from `unix_time`). A guest that panics
    /// with `panic=-1` therefore reboots forever; end such a run with `stop()`.
    ///
    /// A triple fault is never restarted: it ends the run with `ExitReason::TripleFault` so
    /// diagnostics and exit codes keep working. Under `verify_jit` a reset always ends the run
    /// with `Reset`, because the twin interpreter's devices cannot be rebuilt in step.
    pub fn reboot(mut self, on: bool) -> Self {
        self.reboot = on;
        self
    }

    /// Fixes the RTC's boot time. Guest behavior must not depend on it, so a failure that appears
    /// only at some wall-clock time is an emulator bug.
    pub fn unix_time(mut self, secs: u64) -> Self {
        self.unix_time = Some(secs);
        self
    }

    /// Seeds the generator behind the guest's `rdrand` and `rdseed`. The library has no entropy
    /// source of its own (it is `no_std` on some targets), so without this the sequence is fixed
    /// and every boot of the same image reads the same "random" numbers: fine for tests and
    /// reproducible runs, wrong for a guest that generates keys, and on the virtual clock KASLR
    /// picks the same kernel base every time. The UEFI firmware's EFI_RNG_PROTOCOL, which the
    /// EFI stub randomizes the physical placement with, is seeded from it too. An embedder that
    /// cares passes bytes from the host's CSPRNG. The generator is splitmix64, not a
    /// cryptographic one, and the guest kernel mixes it with its other entropy sources instead
    /// of trusting it alone.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    pub fn log(mut self, f: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.log = Some(Arc::new(f));
        self
    }
}

impl MachineConfig {
    /// Called from the vCPU loop (with threads, the device thread) whenever it ticks the devices:
    /// on a busy realtime machine at most every 20 us, not every few thousand instructions. The
    /// flag is true while the guest is halted with nothing to do, when the host may wait.
    /// Single-threaded hosts use it to pump their console.
    pub fn poll(mut self, f: impl Fn(&ConsoleHandle, bool) + Send + Sync + 'static) -> Self {
        self.poll = Some(Arc::new(f));
        self
    }
}

impl MachineConfig {
    /// The monotonic host clock `ClockMode::Realtime` follows, in nanoseconds from any fixed
    /// starting point. Hosts with `std` need not set it; freestanding ones (WebAssembly) must,
    /// or the clock stays virtual.
    pub fn host_clock(mut self, f: impl Fn() -> u64 + Send + Sync + 'static) -> Self {
        self.host_clock = Some(Arc::new(f));
        self
    }
}

impl MachineConfig {
    /// Whether the machine has a virtio-rng device (default on), which keeps the guest's entropy
    /// pool fed from the host. Needs an entropy source: `std` hosts have one, freestanding ones
    /// (WebAssembly, UEFI) must give `entropy` or the machine goes without the device. Under
    /// `verify_jit` the device serves a seeded stream so that both CPUs see the same bytes.
    pub fn rng(mut self, on: bool) -> Self {
        self.rng = on;
        self
    }

    /// Where virtio-rng gets its bytes instead of the host OS: called with the buffer to fill.
    pub fn entropy(mut self, f: impl Fn(&mut [u8]) + Send + Sync + 'static) -> Self {
        self.entropy = Some(Arc::new(f));
        self
    }
}

/// Cloneable, thread-safe way to end a running machine. The vCPU polls the flag at its event
/// ticks and while idle, so `wait()` returns `Stopped` shortly after, not instantly.
#[derive(Clone)]
pub struct StopHandle(Arc<AtomicBool>);

impl StopHandle {
    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Everything the vCPU thread owns once `start` moves it off the caller's thread.
struct Vcpu {
    ram: Vec<u8>,
    boot: BootState,
    /// What the host poll hook receives: the first configured console.
    primary_console: ConsoleHandle,
    /// The console behind the 16550: the `Serial` console if one is configured, else (so early
    /// boot text reaches the host) the first console, shared with its virtio device.
    uart_console: ConsoleHandle,
    stop: Arc<AtomicBool>,
    log: Option<LogFn>,
    poll: Option<HostPollFn>,
    host_clock: Option<HostClockFn>,
    debug: bool,
    jit: JitMode,
    clock: ClockMode,
    smp: SmpMode,
    /// Hosted builds fall back to `std::thread` when the embedder gave none.
    thread_spawner: Option<ThreadSpawner>,
    verify: bool,
    unix_time: Option<u64>,
    random_seed: Option<u64>,
    /// Pristine copy of RAM for the lockstep interpreter CPU; only set when `verify` is on.
    boot_ram: Option<Vec<u8>>,
    devcfg: DevConfig,
    /// What a restart needs; `None` when a reset should end the run.
    reboot: Option<RebootImage>,
}

/// Everything `boot::load` needs to lay the guest out again, plus the parts of the device set
/// that cannot be rebuilt from the old `Devices` (the disk is recovered from them instead).
struct RebootImage {
    kernel: Vec<u8>,
    initrd: Option<Vec<u8>>,
    /// The final command line, defaults and `cmdline_append` already applied.
    cmdline: String,
    /// The embedder's handle shares its queues with the device, so the new device must be built
    /// around this very handle.
    nets: Vec<([u8; 6], NetHandle)>,
    virtio_consoles: Vec<ConsoleHandle>,
    iommu: bool,
    cpus: usize,
    acpi: Option<AcpiConfig>,
    key_value_store: Option<SharedKeyValueStore>,
    rng: Option<RngSource>,
    /// Set when the machine boots UEFI firmware: that image and handoff replace the kernel.
    uefi: Option<(Vec<u8>, boot::FirmwareParams)>,
}

/// The kernel the UEFI firmware is to boot, if the machine was given one.
fn uefi_payload<'a>(
    kernel: &'a [u8],
    initrd: Option<&'a [u8]>,
    cmdline: &'a str,
) -> Option<boot::UefiPayload<'a>> {
    (!kernel.is_empty()).then_some(boot::UefiPayload {
        kernel,
        initrd,
        cmdline,
    })
}

/// Guest RAM on host huge pages where the host has them (Linux, `jit` feature for `libc`).
fn back_with_huge_pages(ram: &mut [u8]) {
    #[cfg(all(feature = "jit", target_os = "linux"))]
    crate::sys::advise_huge_pages(ram);
    #[cfg(not(all(feature = "jit", target_os = "linux")))]
    let _ = ram;
}

/// Zeroed guest RAM for `memory_mib`, laid out around the PCI hole (`ram::flat_bytes`). Fails
/// where the host cannot reserve that much address space (a 32-bit host, or one that does not
/// overcommit).
fn allocate_ram(memory_mib: usize) -> Result<Vec<u8>, Error> {
    let cannot = || Error::BadConfig(format!("cannot allocate {memory_mib} MiB of guest memory"));
    let flat_bytes = crate::ram::flat_bytes((memory_mib as u64) << BYTES_PER_MIB_SHIFT);
    let mut ram = usize::try_from(flat_bytes)
        .ok()
        .and_then(crate::ram::zeroed_bytes)
        .ok_or_else(cannot)?;
    back_with_huge_pages(&mut ram);
    Ok(ram)
}

/// `std::thread` where there is one.
fn default_thread_spawner() -> Option<ThreadSpawner> {
    #[cfg(not(freestanding))]
    {
        Some(crate::cpu::vcpus::std_thread_spawner())
    }
    #[cfg(freestanding)]
    {
        None
    }
}

/// Wall-clock seconds, or 0 where there is no clock (UEFI keeps its own, see `unix_time`).
fn host_unix_time() -> u64 {
    #[cfg(not(freestanding))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0)
    }
    #[cfg(freestanding)]
    {
        0
    }
}

/// The data the main CPU's disk handed to the guest, in order, for the twin's disks to replay.
/// Both CPUs issue the same requests in the same order (the twin runs each block after the main
/// CPU), so a FIFO is enough; the twin's own writes vanish and a later read of what the guest wrote
/// is answered from what the main CPU read.
type DiskReplay = Arc<Mutex<alloc::collections::VecDeque<(u64, Vec<u8>)>>>;

/// The main CPU's disk under verify: every read is also queued for the twin.
struct RecordingDisk {
    inner: Box<dyn crate::disk::BlockBackend>,
    replay: DiskReplay,
}

impl crate::disk::BlockBackend for RecordingDisk {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> crate::disk::io::Result<()> {
        self.inner.read_at(offset, buf)?;
        self.replay
            .lock()
            .unwrap()
            .push_back((offset, buf.to_vec()));
        Ok(())
    }
    fn write_at(&mut self, offset: u64, data: &[u8]) -> crate::disk::io::Result<()> {
        self.inner.write_at(offset, data)
    }
    fn flush(&mut self) -> crate::disk::io::Result<()> {
        self.inner.flush()
    }
    fn read_only(&self) -> bool {
        self.inner.read_only()
    }
}

/// Stands in for a disk on the verify twin: same capacity and read-only flag as the real one so
/// the guest sees the same device. Reads replay what the main CPU's disk returned; a request the
/// main CPU did not make (or made at another offset) reads zeros and shows up as a divergence.
struct ShadowDisk {
    len: u64,
    read_only: bool,
    replay: DiskReplay,
}

impl crate::disk::BlockBackend for ShadowDisk {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> crate::disk::io::Result<()> {
        match self.replay.lock().unwrap().pop_front() {
            Some((recorded_offset, data))
                if recorded_offset == offset && data.len() == buf.len() =>
            {
                buf.copy_from_slice(&data);
            }
            _ => buf.fill(0),
        }
        Ok(())
    }
    fn write_at(&mut self, _offset: u64, _data: &[u8]) -> crate::disk::io::Result<()> {
        Ok(())
    }
    fn flush(&mut self) -> crate::disk::io::Result<()> {
        Ok(())
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
}

/// Devices for the verify twin that look the same to the guest as `main`: same virtio functions
/// in the same PCI slots. The twin cannot share the real backends (a disk or a host frame
/// queue has one owner): the twin's disks replay what the main CPU's disk returned, but host
/// console input and network frames are not mirrored and diverge legitimately.
///
/// Returns the twin's UART console too: consoles that share a handle with the UART on `main`
/// share one on the twin as well, because that sharing decides whether the UART takes host input.
fn mirrored_devices(main: &mut DevConfig, main_uart: &ConsoleHandle) -> (DevConfig, ConsoleHandle) {
    let twin_uart = ConsoleHandle::default();
    let devices = DevConfig {
        disks: main
            .disks
            .iter_mut()
            .map(|disk| {
                let replay = DiskReplay::default();
                let (len, read_only) = (disk.backend.len(), disk.backend.read_only());
                let placeholder = Disk::from_backend(ShadowDisk {
                    len,
                    read_only,
                    replay: replay.clone(),
                });
                let inner = core::mem::replace(&mut disk.backend, placeholder.backend);
                disk.backend = Box::new(RecordingDisk {
                    inner,
                    replay: replay.clone(),
                });
                Disk::from_backend(ShadowDisk {
                    len,
                    read_only,
                    replay,
                })
            })
            .collect(),
        nets: main
            .nets
            .iter()
            .map(|(mac, _)| (*mac, NetHandle::default()))
            .collect(),
        virtio_consoles: main
            .virtio_consoles
            .iter()
            .map(|console| {
                if console.is_same_console(main_uart) {
                    twin_uart.clone()
                } else {
                    ConsoleHandle::default()
                }
            })
            .collect(),
        iommu: main.iommu,
        // The twin gets an empty store of its own, like its disks: it cannot share the real
        // one, and a guest that reads variables diverges legitimately.
        key_value_store: main.key_value_store.as_ref().map(|_| {
            let store: Box<dyn KeyValueStore> = Box::new(MemoryKeyValueStore::new());
            Arc::new(Mutex::new(store))
        }),
        cpus: main.cpus,
        // Verification always runs on a seeded stream, so cloning gives the twin the same bytes.
        rng: main.rng.clone(),
    };
    (devices, twin_uart)
}

/// The interpreter-only oracle for `Cpu::run_verify`. Anything a guest can observe must be
/// identical to the JIT-enabled CPU, or the first guest read of it shows up as a bogus JIT
/// divergence: the RTC boot time (the twin would otherwise read the wall clock) and the
/// device set. Time is always virtual here (no `wall_clock`) because only instruction
/// counts, which both CPUs share, keep the two clocks equal.
fn interpreter_twin(
    ram: Vec<u8>,
    boot: BootState,
    stop: Arc<AtomicBool>,
    boot_unix: u64,
    devices: DevConfig,
    uart_console: ConsoleHandle,
) -> Cpu {
    let twin = Cpu::new(ram, boot, uart_console, stop, None, devices);
    twin.dev.lock().set_boot_time(boot_unix);
    twin
}

/// What a finished run hands to the next one.
struct Finished {
    ram_bytes: usize,
    disks: Vec<Disk>,
}

impl Vcpu {
    /// Boots, and boots again after every guest reset while `reboot` is set.
    fn run(mut self) -> ExitReason {
        loop {
            let (reason, finished) = self.run_once();
            let (ExitReason::Reset, Some(image)) = (&reason, &self.reboot) else {
                return reason;
            };
            if let Some(log) = &self.log {
                log("guest reset: rebooting");
            }
            // Fresh zeroed RAM, not a `fill(0)` of the old one: `boot::load` relies on zeroes,
            // and filling would commit every page the host had lazily left untouched.
            let Some(mut ram) = crate::ram::zeroed_bytes(finished.ram_bytes) else {
                return ExitReason::Internal("reboot: cannot allocate guest memory".into());
            };
            back_with_huge_pages(&mut ram);
            let loaded = match &image.uefi {
                Some((firmware, params)) => boot::load_uefi_firmware(
                    &mut ram,
                    firmware,
                    params,
                    image.acpi.as_ref(),
                    uefi_payload(&image.kernel, image.initrd.as_deref(), &image.cmdline).as_ref(),
                ),
                None => boot::load_with_acpi(
                    &mut ram,
                    &image.kernel,
                    image.initrd.as_deref(),
                    &image.cmdline,
                    image.acpi.as_ref(),
                ),
            };
            match loaded {
                Ok(boot) => self.boot = boot,
                Err(e) => return ExitReason::Internal(format!("reboot: {e}")),
            }
            self.ram = ram;
            self.devcfg = DevConfig {
                disks: finished.disks,
                nets: image.nets.clone(),
                virtio_consoles: image.virtio_consoles.clone(),
                iommu: image.iommu,
                key_value_store: image.key_value_store.clone(),
                cpus: image.cpus,
                rng: image.rng.clone(),
            };
        }
    }

    /// Whether a multi-processor guest gets a host thread per vCPU: not under `Timeshare`, not
    /// without the host clock the threads' TSC needs, and not where nothing can start a thread.
    fn runs_on_threads(&self, cpu: &Cpu) -> bool {
        // WebAssembly blocks are called through the function table of one instance, which a
        // second thread cannot reach (`jit/wasm.rs`): with the JIT the vCPUs take turns.
        let jit_ties_vcpus_to_one_thread = cfg!(target_arch = "wasm32") && cpu.jit.is_some();
        self.smp != SmpMode::Timeshare
            && !self.verify
            && !jit_ties_vcpus_to_one_thread
            && cpu.wall_clock.is_some()
            && self.thread_spawner.is_some()
    }

    /// One power-on to the next exit. Everything the CPU owns is built here, so a reboot starts
    /// from a CPU with no leftover state and from a JIT with an empty code cache: a reused cache
    /// would never see stores to the new CPU's empty `page_has_code` map and could run
    /// translations of the previous run's RAM.
    fn run_once(&mut self) -> (ExitReason, Finished) {
        let console = self.primary_console.clone();
        // Under verify both CPUs must start from one boot time even when the caller gave none;
        // two separate wall-clock reads could straddle a second boundary.
        let verify_twin = self.verify.then(|| {
            let (devices, uart_console) = mirrored_devices(&mut self.devcfg, &self.uart_console);
            (
                devices,
                uart_console,
                self.unix_time.unwrap_or_else(host_unix_time),
            )
        });
        let ram = core::mem::take(&mut self.ram);
        let ram_bytes = ram.len();
        // This thread runs the devices from here on: the vCPU of a one-processor machine (its
        // idle wait parks) or the device thread of a threaded one (`run_threaded`). A frame from
        // the host unparks it at once instead of waiting out the sleep.
        #[cfg(not(freestanding))]
        let nets: Vec<NetHandle> = self
            .devcfg
            .nets
            .iter()
            .map(|(_, net)| net.clone())
            .collect();
        #[cfg(not(freestanding))]
        for net in &nets {
            let thread = std::thread::current();
            net.set_device_waker(Some(Arc::new(move || thread.unpark())));
        }
        let mut cpu = Cpu::new(
            ram,
            self.boot,
            self.uart_console.clone(),
            self.stop.clone(),
            self.log.clone(),
            core::mem::take(&mut self.devcfg),
        );
        let boot_unix = verify_twin
            .as_ref()
            .map(|(_, _, boot_unix)| *boot_unix)
            .or(self.unix_time);
        if let Some(boot_unix) = boot_unix {
            cpu.dev.lock().set_boot_time(boot_unix);
        }
        if let Some(seed) = self.random_seed {
            cpu.rng_state = seed;
        }
        // Cloned, not taken: a reboot builds a new CPU that needs the hook again.
        cpu.poll = self.poll.clone().map(|host_poll| {
            let poll: PollFn = Arc::new(move |idle| host_poll(&console, idle));
            poll
        });
        if self.jit != JitMode::Off {
            // Lockstep compares state after every block, so blocks must not chain into each other.
            cpu.jit = crate::jit::Jit::new(verify_twin.is_none()).map(Box::new);
        }
        if self.debug {
            cpu.trace = Some(Default::default());
            cpu.helper_histogram = Some(Default::default());
        }
        // Lockstep needs both CPUs to idle identically, and the twin skips to the next timer at
        // once, so verification always runs on virtual time (a realtime main CPU sleeps in 2 ms
        // slices and reports a bogus pre-step divergence).
        let lockstep = verify_twin.is_some() && cpu.jit.is_some();
        // Verification is the one deterministic mode, with or without generated code to compare.
        if self.clock == ClockMode::Realtime && !self.verify {
            cpu.wall_clock = self
                .host_clock
                .clone()
                .or_else(crate::cpu::default_host_clock)
                .map(crate::cpu::WallClock::new);
        }
        // Lockstep needs generated code to compare; without any (`JitMode::Off`, a host without
        // a backend) `step_block` has nothing to run, so just interpret.
        let mut vcpu_summary = String::new();
        let reason = match verify_twin {
            Some((devices, uart_console, boot_unix)) if lockstep => {
                let mut twin = interpreter_twin(
                    self.boot_ram.take().unwrap(),
                    self.boot,
                    cpu.stop.clone(),
                    boot_unix,
                    devices,
                    uart_console,
                );
                twin.rng_state = cpu.rng_state;
                cpu.run_verify(&mut twin)
            }
            _ if cpu.cpu_count > 1 && self.runs_on_threads(&cpu) => {
                let spawner = self
                    .thread_spawner
                    .clone()
                    .expect("checked by runs_on_threads");
                // Every vCPU runs the one code cache the bootstrap processor made, through a
                // handle of its own (`Jit::sibling`); with the JIT off there is none.
                let jit_source = cpu.jit.as_ref().map(|jit| jit.sibling());
                let make_jit = || jit_source.as_ref().map(|jit| Box::new(jit.sibling()));
                let (reason, bootstrap, summary) = crate::cpu::vcpus::run_threaded(
                    crate::cpu::smp::secondary_cpus(cpu),
                    &spawner,
                    &make_jit,
                );
                vcpu_summary = summary;
                cpu = bootstrap;
                reason
            }
            _ if cpu.cpu_count > 1 => {
                let mut smp = Smp::new(cpu);
                let reason = smp.run();
                vcpu_summary = smp.summary();
                cpu = smp.into_active();
                reason
            }
            _ => cpu.run(),
        };
        if let (Some(l), true) = (
            &self.log,
            self.debug || matches!(reason, ExitReason::TripleFault | ExitReason::Internal(_)),
        ) {
            l(&format!("{vcpu_summary}{}", cpu.dump()));
        }
        #[cfg(not(freestanding))]
        for net in &nets {
            net.set_device_waker(None);
        }
        let finished = Finished {
            ram_bytes,
            disks: cpu
                .dev
                .into_devices()
                .map(crate::dev::Devices::into_disks)
                .unwrap_or_default(),
        };
        (reason, finished)
    }
}

pub struct EmulatedMachine {
    /// Taken by `start` (hosted) or `wait` (UEFI); `None` afterwards.
    vcpu: Option<Vcpu>,
    #[cfg(not(freestanding))]
    thread: Option<JoinHandle<ExitReason>>,
    #[cfg(freestanding)]
    started: bool,
    /// One per configured console, in configuration order.
    consoles: Vec<ConsoleHandle>,
    /// Same queues as the virtio-net devices in `Vcpu::devcfg`, in configuration order.
    nets: Vec<NetHandle>,
    stop: Arc<AtomicBool>,
}

impl EmulatedMachine {
    /// Validates the configuration and loads the kernel into guest RAM. Nothing executes until
    /// `start`.
    pub fn new(cfg: MachineConfig) -> Result<Self, Error> {
        if !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&cfg.memory_mib) {
            return Err(Error::BadConfig(format!(
                "memory_mib must be between {MIN_MEMORY_MIB} and {MAX_MEMORY_MIB}"
            )));
        }
        if cfg.jit == JitMode::On && !crate::jit::probe() {
            return Err(Error::JitUnavailable);
        }
        if !(MIN_CPUS..=MAX_CPUS).contains(&cfg.cpus) {
            return Err(Error::BadConfig(format!(
                "cpus must be between {MIN_CPUS} and {MAX_CPUS}"
            )));
        }
        if cfg.cpus > 1 {
            if !cfg.acpi {
                return Err(Error::BadConfig(
                    "more than one cpu needs ACPI: the MADT is how the guest finds them".into(),
                ));
            }
            if cfg.verify || cfg.firmware.is_some() || cfg.boot_disk {
                return Err(Error::BadConfig(
                    "more than one cpu cannot be combined with verify_jit, a firmware image or \
                     a BIOS disk boot"
                        .into(),
                ));
            }
            if cfg.smp == SmpMode::Threads {
                let clock_available =
                    cfg.host_clock.is_some() || crate::cpu::default_host_clock().is_some();
                let spawner_available =
                    cfg.thread_spawner.is_some() || default_thread_spawner().is_some();
                if cfg.clock != ClockMode::Realtime || !clock_available || !spawner_available {
                    return Err(Error::BadConfig(
                        "SmpMode::Threads needs the realtime clock, a host clock and a way to \
                         start threads (`thread_spawner` on a freestanding host)"
                            .into(),
                    ));
                }
            }
        }
        let mut ram = allocate_ram(cfg.memory_mib)?;
        let kinds = if cfg.consoles.is_empty() {
            vec![ConsoleKind::Virtio]
        } else {
            cfg.consoles.clone()
        };
        if kinds
            .iter()
            .filter(|kind| **kind == ConsoleKind::Serial)
            .count()
            > 1
        {
            return Err(Error::BadConfig(
                "there is one UART: ConsoleKind::Serial can be configured only once".into(),
            ));
        }
        let disk_count = cfg.disks.len();
        let virtio_console_count = kinds
            .iter()
            .filter(|kind| **kind == ConsoleKind::Virtio)
            .count();
        // Under verify the stream is seeded, so the twin's device repeats it byte for byte.
        let rng = if !cfg.rng {
            None
        } else if cfg.verify {
            Some(RngSource::Seeded(
                cfg.random_seed.unwrap_or(DEFAULT_RANDOM_SEED),
            ))
        } else {
            cfg.entropy
                .clone()
                .or_else(host_entropy)
                .map(RngSource::Host)
        };
        let functions =
            virtio_console_count + disk_count + cfg.net_macs.len() + rng.is_some() as usize;
        if functions > MAX_VIRTIO_FUNCTIONS {
            return Err(Error::BadConfig(format!(
                "{functions} virtio devices configured (consoles, disks, NICs and the entropy source); \
                 the PCI bus has room for {MAX_VIRTIO_FUNCTIONS}"
            )));
        }
        let mut cmdline = cfg.cmdline.clone().unwrap_or_else(|| {
            // The first console is the guest's primary one.
            let default = if kinds[0] == ConsoleKind::Virtio {
                DEFAULT_CMDLINE
            } else {
                DEFAULT_SERIAL_CMDLINE
            };
            // The defaults are the table-less ones (`Platform` hosts need `acpi=off`).
            if cfg.acpi {
                default.replace(ACPI_OFF_WORD, "").replace("  ", " ")
            } else {
                default.to_string()
            }
        });
        for extra in &cfg.cmdline_extra {
            cmdline.push(' ');
            cmdline.push_str(extra);
        }
        if cfg.uefi_firmware.is_some() {
            if cfg.firmware.is_some() || cfg.boot_disk {
                return Err(Error::BadConfig(
                    "uefi_firmware boots its own firmware: it cannot be combined with a \
                     firmware image or boot_disk"
                        .into(),
                ));
            }
            if !cfg.kernel.is_empty() && !boot::has_efi_header(&cfg.kernel) {
                return Err(Error::BadConfig(
                    "kernel has no EFI stub; boot it with firmware none".into(),
                ));
            }
        }
        if cfg.boot_disk {
            if cfg.firmware.is_some() || !cfg.kernel.is_empty() {
                return Err(Error::BadConfig(
                    "boot_disk boots the disk's own boot loader: it cannot be combined with a \
                     kernel or a firmware image"
                        .into(),
                ));
            }
            if disk_count == 0 {
                return Err(Error::BadConfig("boot_disk needs a disk".into()));
            }
            if cfg.verify {
                return Err(Error::BadConfig(
                    "verify_jit cannot follow a BIOS boot: the JIT only runs 64-bit code".into(),
                ));
            }
        }
        let acpi = cfg.acpi.then_some(AcpiConfig {
            pci_functions: functions,
            iommu: cfg.iommu,
            cpus: cfg.cpus as u32,
        });
        let uefi = cfg.uefi_firmware.as_ref().map(|image| {
            let params = boot::FirmwareParams {
                key_value_store: cfg.key_value_store.is_some(),
                unix_time: cfg.unix_time.unwrap_or_else(host_unix_time),
                random_seed: cfg.random_seed.unwrap_or(DEFAULT_RANDOM_SEED),
                rsdp: 0,
                acpi_region: (0, 0),
            };
            (image.clone(), params)
        });
        let boot = match &cfg.firmware {
            _ if cfg.boot_disk => boot::load_bios(),
            _ if uefi.is_some() => {
                let (image, params) = uefi.as_ref().unwrap();
                boot::load_uefi_firmware(
                    &mut ram,
                    image,
                    params,
                    acpi.as_ref(),
                    uefi_payload(&cfg.kernel, cfg.initrd.as_deref(), &cmdline).as_ref(),
                )?
            }
            Some(image) => boot::load_firmware(&mut ram, image)?,
            None => boot::load_with_acpi(
                &mut ram,
                &cfg.kernel,
                cfg.initrd.as_deref(),
                &cmdline,
                acpi.as_ref(),
            )?,
        };
        let nets: Vec<([u8; 6], NetHandle)> = cfg
            .net_macs
            .iter()
            .map(|mac| (*mac, NetHandle::default()))
            .collect();
        let net_handles = nets.iter().map(|(_, handle)| handle.clone()).collect();
        let consoles: Vec<ConsoleHandle> = kinds.iter().map(|_| ConsoleHandle::default()).collect();
        let uart_console = kinds
            .iter()
            .position(|kind| *kind == ConsoleKind::Serial)
            .map_or_else(|| consoles[0].clone(), |index| consoles[index].clone());
        let virtio_consoles = kinds
            .iter()
            .zip(&consoles)
            .filter(|(kind, _)| **kind == ConsoleKind::Virtio)
            .map(|(_, handle)| handle.clone())
            .collect();
        let devcfg = DevConfig {
            disks: cfg.disks,
            nets,
            virtio_consoles,
            iommu: cfg.iommu,
            key_value_store: cfg.key_value_store.clone(),
            cpus: cfg.cpus,
            rng: rng.clone(),
        };
        let boot_ram = cfg.verify.then(|| ram.clone());
        // Verification cannot restart (the twin's devices cannot be rebuilt in step), so it keeps
        // no image and a reset ends the run.
        let reboot =
            (cfg.reboot && !cfg.verify && cfg.firmware.is_none() && !cfg.boot_disk).then(|| {
                RebootImage {
                    kernel: cfg.kernel,
                    initrd: cfg.initrd,
                    cmdline,
                    nets: devcfg.nets.clone(),
                    virtio_consoles: devcfg.virtio_consoles.clone(),
                    iommu: cfg.iommu,
                    cpus: cfg.cpus,
                    acpi,
                    key_value_store: cfg.key_value_store,
                    rng: devcfg.rng.clone(),
                    uefi,
                }
            });
        let stop = Arc::new(AtomicBool::new(false));
        Ok(Self {
            vcpu: Some(Vcpu {
                ram,
                boot,
                primary_console: consoles[0].clone(),
                uart_console,
                stop: stop.clone(),
                log: cfg.log,
                poll: cfg.poll,
                host_clock: cfg.host_clock,
                debug: cfg.debug,
                jit: cfg.jit,
                clock: cfg.clock,
                smp: cfg.smp,
                thread_spawner: cfg.thread_spawner.or_else(default_thread_spawner),
                verify: cfg.verify,
                unix_time: cfg.unix_time,
                random_seed: cfg.random_seed,
                boot_ram,
                devcfg,
                reboot,
            }),
            #[cfg(not(freestanding))]
            thread: None,
            #[cfg(freestanding)]
            started: false,
            consoles,
            nets: net_handles,
            stop,
        })
    }

    /// The first configured console (the only one unless `MachineConfig::console` was called
    /// more than once).
    pub fn console(&self) -> ConsoleHandle {
        self.consoles[0].clone()
    }

    /// Every console, in `MachineConfig::console` order (one virtio console if none was added).
    /// Each has its own queues and size.
    pub fn consoles(&self) -> Vec<ConsoleHandle> {
        self.consoles.clone()
    }

    /// The first NIC's frame channel, `None` without one.
    pub fn net(&self) -> Option<NetHandle> {
        self.nets.first().cloned()
    }

    /// Every NIC's frame channel, in `MachineConfig::net` order.
    pub fn nets(&self) -> Vec<NetHandle> {
        self.nets.clone()
    }

    pub fn stop_handle(&self) -> StopHandle {
        StopHandle(self.stop.clone())
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// Hosted: spawns the vCPU thread and returns. UEFI has no threads, so this only marks the
    /// machine started and the vCPU runs inside `wait`.
    pub fn start(&mut self) -> Result<(), Error> {
        #[cfg(not(freestanding))]
        {
            let vcpu = self.vcpu.take().ok_or(Error::AlreadyStarted)?;
            self.thread = Some(std::thread::spawn(move || vcpu.run()));
        }
        #[cfg(freestanding)]
        {
            if self.started || self.vcpu.is_none() {
                return Err(Error::AlreadyStarted);
            }
            self.started = true;
        }
        Ok(())
    }

    /// Blocks until the guest exits. `Stopped` if the machine was never started or was already
    /// waited on.
    #[cfg(freestanding)]
    pub fn wait(&mut self) -> ExitReason {
        match self.vcpu.take() {
            Some(v) if self.started => v.run(),
            _ => ExitReason::Stopped,
        }
    }

    /// Blocks until the guest exits; `Stopped` if the machine was never started.
    #[cfg(not(freestanding))]
    pub fn wait(&mut self) -> ExitReason {
        match self.thread.take() {
            Some(t) => t
                .join()
                .unwrap_or_else(|_| ExitReason::Internal("vcpu thread panicked".into())),
            None => ExitReason::Stopped,
        }
    }
}

#[cfg(test)]
mod tests;
