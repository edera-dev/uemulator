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
use crate::dev::tpm::{STORE_PREFIX, TpmConfig};
use crate::dev::{
    DISPLAY_FUNCTIONS, DevConfig, DisplayDevices, EntropyFn, INPUT_FUNCTIONS, MAX_VIRTIO_FUNCTIONS,
    RngSource,
};
use crate::disk::Disk;
use crate::display::{self, DisplayHandle, InputHandle};
use crate::gpu::{self, GpuLink};
use crate::kv::{KeyValueStore, MemoryKeyValueStore};
use crate::netdev::NetHandle;
use crate::platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE};
use crate::ram::GuestRam;
use crate::snapshot::{SnapshotError, SnapshotInfo, SnapshotSink, SnapshotSource};
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
    /// `StopHandle::suspend()`: the machine stopped and its snapshot went to
    /// `MachineConfig::suspend_to`, whose `finish` returned Ok. The disks were flushed first, so
    /// the snapshot and the disks agree until something else writes them.
    Suspended,
    /// `StopHandle::suspend()`, and the snapshot could not be taken or written (a 3D GPU, a full
    /// disk, a failed flush). The machine is stopped as by `stop()`: its state is gone, and the
    /// sink may hold part of a snapshot, which must not be resumed from (its `finish` was not
    /// called).
    SuspendFailed(String),
    /// `MachineConfig::resume_from` could not be restored: another machine's snapshot, another
    /// build's, or a damaged one. Nothing ran.
    ResumeFailed(String),
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

/// Whether the machine has a screen next to its text consoles. Keyboard and pointer are a choice
/// of their own (`MachineConfig::input`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayMode {
    /// No screen: text consoles only (`MachineConfig::console`).
    #[default]
    None,
    /// A virtio-gpu screen, shown by the host through `EmulatedMachine::display()`. Booted
    /// through the UEFI firmware, the firmware draws its console and boot menu on it and hands it
    /// to the OS as a GOP framebuffer. Unless `MachineConfig::console_on_screen(false)`, the
    /// default command line gains `console=tty0`, so the kernel log and `/dev/console` are on
    /// the screen while the text console keeps a copy of the log and early boot text stays on
    /// the 16550. Layout in `docs/plans/24-display-input.md`.
    Framebuffer,
}

pub const MIN_CPUS: usize = 1;
/// The threads of the largest EPYC 9005 package (192 cores, SMT2), so the brand string can always
/// name a part with as many cores (`cpu/profile.rs`), and the most the tests boot. APIC ids are
/// the vCPU numbers, so from 255 up they exist only in x2APIC mode: the MADT lists those
/// processors as x2APIC entries, and a guest needs interrupt remapping (`iommu`) to point device
/// interrupts at them; Linux without it brings up only the processors below 255.
pub const MAX_CPUS: usize = crate::cpu::profile::MAX_LOGICAL_PROCESSORS as usize;
pub const MIN_MEMORY_MIB: usize = 64;
/// The most RAM a machine gets: the first TiB of physical address space (the CPU decodes 52 bits,
/// but the direct-boot identity map in `boot.rs` stops at 1 TiB), less the 1 GiB PCI hole below
/// 4 GiB, which is not RAM. RAM past 3 GiB is placed from 4 GiB up (`ram::flat_bytes`), so the
/// last byte lands at 1 TiB when the hole is subtracted.
/// Memory is committed lazily, so the host needs the RAM only as far as the guest touches it.
pub const MAX_MEMORY_MIB: usize = (1 << (40 - BYTES_PER_MIB_SHIFT)) - HOLE_MIB;
/// The PCI hole in MiB: `MAX_MEMORY_MIB` loses it from the first TiB.
const HOLE_MIB: usize =
    ((crate::ram::PCI_HOLE_END - crate::ram::PCI_HOLE_START) >> BYTES_PER_MIB_SHIFT) as usize;
const DEFAULT_MEMORY_MIB: usize = 512;
const BYTES_PER_MIB_SHIFT: usize = 20;
/// The word `DEFAULT_CMDLINE` carries for hosts without ACPI tables.
const ACPI_OFF_WORD: &str = "acpi=off";
/// Xen's console on the machine's 16550, which is where the host terminal is.
pub const DEFAULT_HYPERVISOR_CMDLINE: &str = "console=com1 com1=115200,8n1";
/// dom0's default under a hypervisor: Xen's console is `hvc0`, and the 16550 belongs to Xen (a
/// PV dom0 that touched it for `earlyprintk=serial` would fault), so early text goes through Xen.
const DEFAULT_DOM0_CMDLINE: &str = "console=hvc0 earlyprintk=xen acpi=off panic=-1";

pub(crate) type LogFn = Arc<dyn Fn(&str) + Send + Sync>;
/// The poll hook as the CPU sees it: the console is already bound, the flag means "guest is
/// halted with nothing scheduled, the host may wait".
pub(crate) type PollFn = Arc<dyn Fn(bool) + Send + Sync>;
/// Monotonic host nanoseconds from any fixed starting point (`MachineConfig::host_clock`).
pub(crate) type HostClockFn = Arc<dyn Fn() -> u64 + Send + Sync>;
/// The embedder-facing poll hook of `MachineConfig::poll`, which receives the console itself.
type HostPollFn = Arc<dyn Fn(&ConsoleHandle, bool) + Send + Sync>;

/// Keeps the TPM's seeded stream apart from virtio-rng's when both come from `random_seed`: a
/// guest that compared them would otherwise see the same bytes twice.
const TPM_RANDOM_SEED_SALT: u64 = 0x5450_4d32_5f52_4e47;

pub struct MachineConfig {
    kernel: Vec<u8>,
    firmware: Option<Vec<u8>>,
    uefi_firmware: Option<Vec<u8>>,
    key_value_store: Option<SharedKeyValueStore>,
    boot_disk: bool,
    initrd: Option<Vec<u8>>,
    cmdline: Option<String>,
    cmdline_extra: Vec<String>,
    hypervisor: Option<Vec<u8>>,
    hypervisor_cmdline: Option<String>,
    memory_mib: usize,
    cpus: usize,
    smt: bool,
    disks: Vec<Disk>,
    net_macs: Vec<[u8; 6]>,
    consoles: Vec<ConsoleKind>,
    display: DisplayMode,
    display_size: (u32, u32),
    /// `None` follows the display: input devices come with a screen unless asked otherwise.
    input: Option<bool>,
    console_on_screen: bool,
    gpu: Option<GpuLink>,
    iommu: bool,
    acpi: bool,
    tpm: bool,
    secure_boot: bool,
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
    suspend_to: Option<Box<dyn SnapshotSink>>,
    resume_from: Vec<Box<dyn SnapshotSource>>,
    check_dirty_tracking: bool,
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
            hypervisor: None,
            hypervisor_cmdline: None,
            memory_mib: DEFAULT_MEMORY_MIB,
            cpus: MIN_CPUS,
            smt: true,
            disks: Vec::new(),
            net_macs: Vec::new(),
            consoles: Vec::new(),
            display: DisplayMode::None,
            display_size: (display::DEFAULT_WIDTH, display::DEFAULT_HEIGHT),
            input: None,
            console_on_screen: true,
            gpu: None,
            iommu: false,
            acpi: true,
            tpm: false,
            secure_boot: false,
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
            suspend_to: None,
            resume_from: Vec::new(),
            check_dirty_tracking: false,
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
    /// (`docs/plans/19-uefi-firmware.md`); the firmware finds the disks and boots from them. A guest
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

    /// Boots a hypervisor (Xen) with the `kernel`, `initrd` and command line as its first
    /// domain, dom0. Takes Xen as it ships: `xen.gz` (or the uncompressed `xen`), an ELF with a
    /// Multiboot2 header, or `xen.efi`. Needs `uefi_firmware` and a kernel; the kernel then needs
    /// no EFI stub. The firmware loads `xen.gz` the way GRUB's `multiboot2` does on UEFI (boot
    /// services kept, Xen's EFI entry point), and starts `xen.efi` with its configuration, the
    /// kernel and the initramfs behind Xen's `LoadFile2` media loaders, as Edera's boot loader
    /// does (`docs/plans/18-xen-dom0.md`). Only Edera's `xen.efi` reads those; an upstream one
    /// looks for `xen.cfg` on a disk, so give such a Xen as `xen.gz`.
    ///
    /// Xen owns the 16550 and gives dom0 its console as `hvc0`, so with a hypervisor the default
    /// console is `ConsoleKind::Serial` and the default command line puts dom0's console on
    /// `hvc0`.
    pub fn hypervisor(mut self, image: impl Into<Vec<u8>>) -> Self {
        self.hypervisor = Some(image.into());
        self
    }

    /// The hypervisor's own command line, `DEFAULT_HYPERVISOR_CMDLINE` (Xen's console on the
    /// 16550) when not set.
    pub fn hypervisor_cmdline(mut self, s: impl Into<String>) -> Self {
        self.hypervisor_cmdline = Some(s.into());
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
    /// which needs `acpi` (on by default); see `docs/plans/16-iommu.md`.
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
    /// Gives the machine a TPM 2.0 (off by default): a CRB interface at 0xFED40000 that the ACPI
    /// tables describe (`TPM2`, `MSFT0101`). The UEFI firmware starts it and measures the boot
    /// into its PCRs, with an event log for the OS. Besides the measured-boot commands it has
    /// RSA 2048 and ECC P-256 keys, sealed data, HMAC and policy sessions, NV indices and
    /// persistent objects; its seeds, NV indices and persistent objects live in the
    /// `key_value_store` (in memory for the machine's lifetime without one), the rest until the
    /// guest resets (`docs/plans/26-tpm-secure-boot.md`).
    pub fn tpm(mut self, on: bool) -> Self {
        self.tpm = on;
        self
    }

    /// Has the UEFI firmware enforce Secure Boot (off by default; needs `uefi_firmware`). On the
    /// first boot of a key-value store without a platform key it enrolls its defaults: a
    /// uemulator PK whose private key was discarded, Microsoft's KEKs, Microsoft's db
    /// certificates (Windows and third-party UEFI CAs, 2011 and 2023) and the current x64 dbx.
    /// Then it refuses any image db does not allow. Off, the enrolled keys stay in the store and
    /// `SecureBoot` reads 0.
    pub fn secure_boot(mut self, on: bool) -> Self {
        self.secure_boot = on;
        self
    }

    pub fn console(mut self, kind: ConsoleKind) -> Self {
        self.consoles.push(kind);
        self
    }

    /// Adds a screen (`DisplayMode::Framebuffer`) or not (`None`, the default).
    pub fn display(mut self, mode: DisplayMode) -> Self {
        self.display = mode;
        self
    }

    /// Gives the screen's virtio-gpu a renderer: the device then offers `VIRGL` and the capsets
    /// in `link`, and forwards the guest's 3D commands to it (`docs/plans/25-gpu-3d.md`). The
    /// embedder keeps a clone and drains it (`GpuLink::take_commands`); a guest whose fences are
    /// never completed waits on them. Needs a display, and cannot be combined with `verify_jit`.
    pub fn gpu(mut self, link: GpuLink) -> Self {
        self.gpu = Some(link);
        self
    }

    /// Adds a virtio keyboard and tablet, fed through `EmulatedMachine::input()`, or leaves
    /// them out. By default a machine has them exactly when it has a screen; without one they
    /// still reach the guest's input layer (`/dev/input`, the firmware's text input), with
    /// pointer positions relative to `display_size`.
    pub fn input(mut self, on: bool) -> Self {
        self.input = Some(on);
        self
    }

    /// Whether the screen is the guest's interactive console when the machine has one (the
    /// default): the default command line then ends in `console=tty0`, so `/dev/console`, and
    /// with it init's shell, is on the screen. With `false` the first text console stays the
    /// guest's console and the screen shows only what the guest runs on its virtual terminals
    /// (a distro's login on tty1). No effect without a display or with an explicit `cmdline`.
    pub fn console_on_screen(mut self, on: bool) -> Self {
        self.console_on_screen = on;
        self
    }

    /// The screen size the display starts with and offers the guest, 320x200 to 2560x1600
    /// (default 1280x800). The firmware's framebuffer keeps it; a guest driver may switch modes
    /// later (`DisplayHandle::request_size`). Without a display it is the area input pointer
    /// positions are relative to.
    pub fn display_size(mut self, width: u32, height: u32) -> Self {
        self.display_size = (width, height);
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

    /// Simultaneous multithreading, on by default as on Turin: an even `cpus` count is presented
    /// as cores of two threads each (vCPUs 0 and 1 share core 0, and so on), an odd one as one
    /// thread per core. Off gives every vCPU a core of its own. Either way the guest sees one
    /// package whose brand string names the smallest EPYC 9005 part with that many cores, and
    /// vCPU `n` has APIC id `n`.
    pub fn smt(mut self, on: bool) -> Self {
        self.smt = on;
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

    /// Where `StopHandle::suspend` writes the machine's snapshot (`docs/plans/29-snapshots.md`).
    pub fn suspend_to(mut self, sink: impl SnapshotSink + 'static) -> Self {
        self.suspend_to = Some(Box::new(sink));
        self
    }

    /// Resume a suspended machine instead of booting: its snapshot is read into the machine this
    /// configuration builds, which must be the same machine (memory, processors, devices, disks
    /// in the same order; the kernel and firmware only matter for a later reboot). `wait()`
    /// returns `ResumeFailed` when it cannot be. The disks must be as the suspend left them: the
    /// guest's file systems are mid-flight in the snapshot.
    pub fn resume_from(mut self, source: impl SnapshotSource + 'static) -> Self {
        self.resume_from = vec![Box::new(source)];
        self
    }

    /// `resume_from` for a delta snapshot: a full snapshot followed by the deltas down to the one
    /// to resume, each the child of the one before (`read_snapshot_info` says whose child a
    /// snapshot is). Each is checked before the next is applied.
    ///
    /// When every one is kept (`SnapshotSource::retained`), the machine's next checkpoint is a
    /// delta against the last; otherwise it is a full snapshot.
    pub fn resume_from_chain(mut self, sources: Vec<Box<dyn SnapshotSource>>) -> Self {
        self.resume_from = sources;
        self
    }

    /// For checking the emulator, not for use: keeps a copy of RAM from each checkpoint and makes
    /// the next one fail, naming the pages, if a page changed that the dirty log did not record
    /// (a store path that skips it, which would make deltas silently wrong). Doubles the memory
    /// used.
    #[doc(hidden)]
    pub fn check_dirty_tracking(mut self, on: bool) -> Self {
        self.check_dirty_tracking = on;
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
pub struct StopHandle {
    stop: Arc<AtomicBool>,
    requests: Arc<HostRequests>,
}

/// Called once a checkpoint is written (or could not be): its header, or why not.
type CheckpointDone = Box<dyn FnOnce(Result<SnapshotInfo, SnapshotError>) + Send>;

struct CheckpointRequest {
    sink: Box<dyn SnapshotSink>,
    done: CheckpointDone,
}

/// What the host asked of the run besides `stop`. The vCPUs only watch `stop`; a checkpoint
/// raises it too, so the run that notices it has to tell a real stop (`ended`) from a pause.
#[derive(Default)]
struct HostRequests {
    /// Raised with `stop` by `stop()` and `suspend()`; never cleared.
    ended: AtomicBool,
    suspend: AtomicBool,
    /// Pending checkpoints. Pushed and drained under the lock together with raising and lowering
    /// `stop`, so a request is never left waiting with the stop lowered.
    checkpoints: Mutex<Vec<CheckpointRequest>>,
    /// No checkpoint can be taken: the machine ended, or it runs `verify_jit` or a BIOS disk
    /// boot. Requests are answered at once.
    closed: AtomicBool,
}

impl HostRequests {
    fn end(&self, stop: &AtomicBool) {
        // Before the stop: the run must see why once it notices the stop.
        self.ended.store(true, Ordering::SeqCst);
        stop.store(true, Ordering::SeqCst);
    }

    /// Answers every pending checkpoint with `why`.
    fn refuse_checkpoints(&self, why: &str) {
        let pending = core::mem::take(&mut *self.checkpoints.lock().unwrap());
        for request in pending {
            (request.done)(Err(SnapshotError(why.into())));
        }
    }

    /// The pending checkpoints, or `None` once there are none, having lowered the stop they
    /// raised (raised again if a real stop came in meanwhile: `ended` is set before it).
    fn take_or_resume(&self, stop: &AtomicBool) -> Option<Vec<CheckpointRequest>> {
        let mut pending = self.checkpoints.lock().unwrap();
        if pending.is_empty() {
            stop.store(false, Ordering::SeqCst);
            if self.ended.load(Ordering::SeqCst) {
                stop.store(true, Ordering::SeqCst);
            }
            return None;
        }
        Some(core::mem::take(&mut *pending))
    }

    /// After a run: a stop raised only for checkpoints must not end the next boot.
    fn lower_stop_unless_ended(&self, stop: &AtomicBool) {
        let _pending = self.checkpoints.lock().unwrap();
        stop.store(false, Ordering::SeqCst);
        if self.ended.load(Ordering::SeqCst) {
            stop.store(true, Ordering::SeqCst);
        }
    }

    /// The machine is gone: later requests are refused, pending ones answered.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.refuse_checkpoints("the machine has ended");
    }
}

impl StopHandle {
    pub fn stop(&self) {
        self.requests.end(&self.stop);
    }

    /// Stops the machine and writes its snapshot to `MachineConfig::suspend_to`; `wait()` then
    /// returns `Suspended` (or `SuspendFailed`). Without a sink it is `stop()`.
    pub fn suspend(&self) {
        self.requests.suspend.store(true, Ordering::SeqCst);
        self.requests.end(&self.stop);
    }

    /// Takes a checkpoint of the running machine (`docs/plans/34-incremental-snapshots.md`): it
    /// pauses at its next event check, flushes the disks, writes its snapshot to `sink`, and runs
    /// on. The snapshot is a delta against the machine's previous checkpoint, or against the
    /// snapshot it resumed from when that is kept; otherwise (the first one, or after a guest
    /// reboot) it is full. `sink.finish` runs while the machine is still paused and the disks are
    /// flushed: the place to copy the disk images that belong with the checkpoint, before the
    /// snapshot itself is made visible.
    ///
    /// `done` gets the snapshot's header, or why there is none; a failed checkpoint changes
    /// nothing, and the next delta is against the last one that succeeded. A machine that stops,
    /// suspends, resets or ends before it gets there answers with an error.
    pub fn checkpoint(
        &self,
        sink: impl SnapshotSink + 'static,
        done: impl FnOnce(Result<SnapshotInfo, SnapshotError>) + Send + 'static,
    ) {
        {
            let mut pending = self.requests.checkpoints.lock().unwrap();
            if !self.requests.closed.load(Ordering::SeqCst) {
                pending.push(CheckpointRequest {
                    sink: Box::new(sink),
                    done: Box::new(done),
                });
                self.stop.store(true, Ordering::SeqCst);
                return;
            }
        }
        done(Err(SnapshotError(
            "this machine takes no checkpoints: it has ended, or runs verify_jit or a BIOS disk \
             boot"
                .into(),
        )));
    }
}

/// Everything the vCPU thread owns once `start` moves it off the caller's thread.
struct Vcpu {
    ram: GuestRam,
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
    boot_ram: Option<GuestRam>,
    devcfg: DevConfig,
    /// What a restart needs; `None` when a reset should end the run.
    reboot: Option<RebootImage>,
    /// Suspend and checkpoint requests, shared with every `StopHandle`.
    requests: Arc<HostRequests>,
    suspend_to: Option<Box<dyn SnapshotSink>>,
    /// Taken by the first run: a reboot after resuming boots afresh.
    resume_from: Vec<Box<dyn SnapshotSource>>,
    check_dirty_tracking: bool,
}

/// The display devices for one boot: the embedder's handle, and the boot framebuffer only
/// when the UEFI firmware publishes one (at the place `framebuffer_region` derives from the RAM
/// size, which is the same on every boot).
fn display_devices(
    display: &Option<DisplayHandle>,
    gpu: &Option<GpuLink>,
    firmware: Option<&boot::FirmwareParams>,
    ram_top: u64,
) -> Option<DisplayDevices> {
    let display = display.as_ref()?;
    let boot_framebuffer = firmware
        .and_then(|params| params.framebuffer)
        .map(|(width, height)| boot::framebuffer_region(ram_top, width, height));
    Some(DisplayDevices {
        display: display.clone(),
        boot_framebuffer,
        gpu: gpu.clone(),
    })
}

/// A renderer needs the screen's GPU to sit behind, and capsets the guest's driver accepts:
/// Linux drops every capset when it finds an id of 0. The twin of `verify_jit` would need a
/// renderer of its own, completing fences in step.
fn validate_gpu(link: &GpuLink, has_display: bool, verify: bool) -> Result<(), Error> {
    if !has_display {
        return Err(Error::BadConfig("gpu needs a display".into()));
    }
    if verify {
        return Err(Error::BadConfig(
            "verify_jit cannot run with a gpu renderer".into(),
        ));
    }
    let capsets = link.capsets();
    if capsets.len() > gpu::MAX_CAPSETS {
        return Err(Error::BadConfig(format!(
            "at most {} gpu capsets",
            gpu::MAX_CAPSETS
        )));
    }
    for (index, capset) in capsets.iter().enumerate() {
        if capset.id == 0 || capsets[..index].iter().any(|other| other.id == capset.id) {
            return Err(Error::BadConfig(format!(
                "gpu capset ids must be nonzero and distinct (capset {index})"
            )));
        }
        if capset.data.len() > gpu::MAX_CAPSET_BYTES {
            return Err(Error::BadConfig(format!(
                "gpu capset {} is larger than {} bytes",
                capset.id,
                gpu::MAX_CAPSET_BYTES
            )));
        }
    }
    Ok(())
}

/// Everything `boot::load` needs to lay the guest out again, plus the parts of the device set
/// that cannot be rebuilt from the old `Devices` (the disk is recovered from them instead).
struct RebootImage {
    kernel: Vec<u8>,
    initrd: Option<Vec<u8>>,
    /// The final command line, defaults and `cmdline_append` already applied.
    cmdline: String,
    /// Decompressed, with its final command line.
    hypervisor: Option<(Vec<u8>, String)>,
    /// The embedder's handle shares its queues with the device, so the new device must be built
    /// around this very handle.
    nets: Vec<([u8; 6], NetHandle)>,
    virtio_consoles: Vec<ConsoleHandle>,
    iommu: bool,
    cpus: usize,
    smt: bool,
    acpi: Option<AcpiConfig>,
    key_value_store: Option<SharedKeyValueStore>,
    rng: Option<RngSource>,
    tpm: Option<TpmConfig>,
    /// The embedder holds these; the new devices must be built around them, like the NICs.
    display: Option<DisplayHandle>,
    input: Option<InputHandle>,
    gpu: Option<GpuLink>,
    /// Set when the machine boots UEFI firmware: that image and handoff replace the kernel.
    uefi: Option<(Vec<u8>, boot::FirmwareParams)>,
}

/// The kernel (and hypervisor) the UEFI firmware is to boot, if the machine was given one.
fn uefi_payload<'a>(
    kernel: &'a [u8],
    initrd: Option<&'a [u8]>,
    cmdline: &'a str,
    hypervisor: Option<&'a (Vec<u8>, String)>,
) -> Option<boot::UefiPayload<'a>> {
    (!kernel.is_empty()).then_some(boot::UefiPayload {
        kernel,
        initrd,
        cmdline,
        hypervisor: hypervisor.map(|(image, text)| (image.as_slice(), text.as_str())),
    })
}

/// Guest RAM on host huge pages where the host has them (Linux, `jit` feature for `libc`).
fn back_with_huge_pages(ram: &mut GuestRam) {
    // A 64-bit host's RAM always fits one slice; only wasm32's can be longer.
    #[cfg(all(feature = "jit", target_os = "linux"))]
    if let Some(bytes) = ram.get_mut(0..ram.len()) {
        crate::sys::advise_huge_pages(bytes);
    }
    #[cfg(not(all(feature = "jit", target_os = "linux")))]
    let _ = ram;
}

/// Zeroed guest RAM for `memory_mib`, laid out around the PCI hole (`ram::flat_bytes`). Fails
/// where the host cannot reserve that much address space (a 32-bit host, or one that does not
/// overcommit).
fn allocate_ram(memory_mib: usize) -> Result<GuestRam, Error> {
    let cannot = || Error::BadConfig(format!("cannot allocate {memory_mib} MiB of guest memory"));
    let flat_bytes = crate::ram::flat_bytes((memory_mib as u64) << BYTES_PER_MIB_SHIFT);
    let mut ram = usize::try_from(flat_bytes)
        .ok()
        .and_then(GuestRam::zeroed)
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
        smt: main.smt,
        // Verification always runs on a seeded stream, so cloning gives the twin the same bytes.
        rng: main.rng.clone(),
        // Seeded under verify, like the rng: the twin's TPM answers GetRandom with the same bytes.
        // It gets a copy of what the TPM keeps in the store, so both start from the same seeds,
        // NV and persistent objects without the twin writing to the real store.
        tpm: main.tpm.as_ref().map(|config| {
            let mut copy = MemoryKeyValueStore::new();
            let mut store = config.store.lock().unwrap();
            for key in store.list(STORE_PREFIX) {
                if let Some(value) = store.get(&key) {
                    let _ = copy.set(&key, &value);
                }
            }
            let copy: Box<dyn KeyValueStore> = Box::new(copy);
            TpmConfig {
                random: config.random.clone(),
                store: Arc::new(Mutex::new(copy)),
            }
        }),
        // Same functions and the same boot framebuffer region; host input reaches the main CPU
        // only, and the twin's screen is never looked at.
        display: main.display.as_ref().map(|devices| {
            let (width, height) = devices.display.size();
            DisplayDevices {
                display: DisplayHandle::with_size(width, height),
                boot_framebuffer: devices.boot_framebuffer,
                // `verify_jit` refuses a renderer.
                gpu: None,
            }
        }),
        input: main.input.as_ref().map(|input| {
            let (width, height) = input.pointer_area();
            InputHandle::without_display(width, height)
        }),
    };
    (devices, twin_uart)
}

/// The interpreter-only oracle for `Cpu::run_verify`. Anything a guest can observe must be
/// identical to the JIT-enabled CPU, or the first guest read of it shows up as a bogus JIT
/// divergence: the RTC boot time (the twin would otherwise read the wall clock) and the
/// device set. Time is always virtual here (no `wall_clock`) because only instruction
/// counts, which both CPUs share, keep the two clocks equal.
fn interpreter_twin(
    ram: GuestRam,
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
    /// The RAM the run ended with, every vCPU's handle to it dropped, so a reboot can reuse
    /// memory that cannot be freed (`GuestRam::zeroed_again`).
    ram: GuestRam,
    disks: Vec<Disk>,
}

impl Vcpu {
    /// Boots, and boots again after every guest reset while `reboot` is set.
    fn run(self) -> ExitReason {
        let requests = self.requests.clone();
        let reason = self.run_boots();
        requests.close();
        reason
    }

    fn run_boots(mut self) -> ExitReason {
        loop {
            let (reason, finished) = self.run_once();
            let (ExitReason::Reset, Some(image)) = (&reason, &self.reboot) else {
                return reason;
            };
            if let Some(log) = &self.log {
                log("guest reset: rebooting");
            }
            // `boot::load` relies on zeroed RAM.
            let Some(mut ram) = finished.ram.zeroed_again() else {
                return ExitReason::Internal("reboot: cannot allocate guest memory".into());
            };
            back_with_huge_pages(&mut ram);
            let loaded = match &image.uefi {
                Some((firmware, params)) => boot::load_uefi_firmware(
                    &mut ram,
                    firmware,
                    params,
                    image.acpi.as_ref(),
                    uefi_payload(
                        &image.kernel,
                        image.initrd.as_deref(),
                        &image.cmdline,
                        image.hypervisor.as_ref(),
                    )
                    .as_ref(),
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
            let display = display_devices(
                &image.display,
                &image.gpu,
                image.uefi.as_ref().map(|(_, params)| params),
                ram.len() as u64,
            );
            self.ram = ram;
            self.devcfg = DevConfig {
                disks: finished.disks,
                nets: image.nets.clone(),
                virtio_consoles: image.virtio_consoles.clone(),
                iommu: image.iommu,
                key_value_store: image.key_value_store.clone(),
                cpus: image.cpus,
                smt: image.smt,
                rng: image.rng.clone(),
                tpm: image.tpm.clone(),
                display,
                input: image.input.clone(),
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

    /// Runs every vCPU of the machine (threaded, time-shared, or the only one) until the run
    /// ends, starting with `active` holding the machine-wide state; hands them all back, and
    /// which one holds it now. Called again after a checkpoint, so whatever a run takes from the
    /// vCPUs it must give back.
    fn run_vcpus(
        &self,
        mut cpus: Vec<Cpu>,
        active: usize,
        threaded: bool,
        summary: &mut String,
    ) -> (ExitReason, Vec<Cpu>, usize) {
        if threaded {
            let spawner = self
                .thread_spawner
                .clone()
                .expect("checked by runs_on_threads");
            // Every vCPU runs the one code cache the bootstrap processor made, through a handle
            // of its own (`Jit::sibling`); with the JIT off there is none.
            let jit_source = cpus[0].jit.as_ref().map(|jit| jit.sibling());
            let make_jit = || jit_source.as_ref().map(|jit| Box::new(jit.sibling()));
            // A vCPU that left the last run on `ATTENTION_EXIT` may not have drained the
            // invalidations published just before; the new run's links start empty.
            for cpu in cpus.iter_mut() {
                cpu.decode_cache_generation = cpu.decode_cache_generation.wrapping_add(1);
            }
            let (reason, cpus, vcpu_summary) =
                crate::cpu::vcpus::run_threaded(cpus, &spawner, &make_jit);
            *summary = vcpu_summary;
            (reason, cpus, 0)
        } else if cpus.len() > 1 {
            let mut smp = Smp::with_cpus(cpus, active);
            let reason = smp.run();
            *summary = smp.summary();
            let (cpus, active) = smp.into_cpus();
            (reason, cpus, active)
        } else {
            let reason = cpus[0].run();
            (reason, cpus, 0)
        }
    }

    /// How many of the pages `check_dirty_tracking` found unmarked a failed checkpoint names.
    const MISSED_PAGES_NAMED: usize = 16;

    /// Writes one checkpoint of the stopped machine to `sink`, and on success starts the dirty
    /// log's next generation against it (`StopHandle::checkpoint`).
    fn checkpoint(
        &self,
        cpus: &mut [Cpu],
        active: usize,
        threaded: bool,
        sink: &mut dyn SnapshotSink,
        armed_copy: &mut Option<GuestRam>,
    ) -> Result<SnapshotInfo, SnapshotError> {
        if cpus
            .get(active)
            .is_none_or(|cpu| cpus.len() != cpu.cpu_count as usize)
        {
            return Err(SnapshotError("a vCPU thread ended abnormally".into()));
        }
        if let Some(copy) = armed_copy.as_ref() {
            let log = cpus[active].page_has_code.dirty_log();
            let missed = suspend::unmarked_changes(&cpus[active].ram, copy, log);
            if !missed.is_empty() {
                let pages: Vec<String> = missed
                    .iter()
                    .take(Self::MISSED_PAGES_NAMED)
                    .map(|page| format!("{:#x}", page << crate::cpu::PAGE_SHIFT))
                    .collect();
                return Err(SnapshotError(format!(
                    "dirty tracking missed {} written pages: {}",
                    missed.len(),
                    pages.join(" ")
                )));
            }
        }
        // The pause is not guest time: the clock resumes where the guest left it.
        let paused_at = cpus[active].now_ns();
        let saved = suspend::save(cpus, active, threaded, sink, SaveKind::Checkpoint);
        suspend::unpark_after_save(cpus, active, threaded);
        if let Some(wall) = &cpus[active].wall_clock {
            wall.skip_pause_since(paused_at);
        }
        let info = saved?;
        suspend::arm_dirty_log(cpus, info.id);
        if self.check_dirty_tracking {
            *armed_copy = cpus[active].ram.duplicate();
        }
        Ok(info)
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
        // The same for keystrokes and pointer moves.
        #[cfg(not(freestanding))]
        let input = self.devcfg.input.clone();
        #[cfg(not(freestanding))]
        if let Some(input) = &input {
            let thread = std::thread::current();
            input.set_device_waker(Some(Arc::new(move || thread.unpark())));
        }
        // And for the renderer's fences, which a guest waiting in its idle loop is halted for.
        #[cfg(not(freestanding))]
        let gpu = self
            .devcfg
            .display
            .as_ref()
            .and_then(|devices| devices.gpu.clone());
        #[cfg(not(freestanding))]
        if let Some(gpu) = &gpu {
            let thread = std::thread::current();
            gpu.set_device_waker(Some(Arc::new(move || thread.unpark())));
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
            cpu.svm_exit_histogram = Some(Default::default());
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
                let reason = cpu.run_verify(&mut twin);
                // `verify_jit` refuses checkpoints (`HostRequests::closed`); one that slipped in
                // before must not end the next boot.
                self.requests
                    .refuse_checkpoints("the machine stopped before the checkpoint");
                self.requests.lower_stop_unless_ended(&self.stop);
                reason
            }
            _ => {
                let threaded = cpu.cpu_count > 1 && self.runs_on_threads(&cpu);
                let cpu_count = cpu.cpu_count as usize;
                // Every vCPU exists before anything runs, so a resume can load each one.
                let mut cpus = if cpu.cpu_count > 1 {
                    crate::cpu::smp::secondary_cpus(cpu)
                } else {
                    vec![cpu]
                };
                let mut sources = core::mem::take(&mut self.resume_from);
                let resumed = if sources.is_empty() {
                    Ok(())
                } else {
                    match suspend::restore(&mut cpus, threaded, &mut sources) {
                        Ok((info, parent_capable)) => {
                            if parent_capable {
                                suspend::arm_dirty_log(&mut cpus, info.id);
                            }
                            Ok(())
                        }
                        Err(error) => Err(ExitReason::ResumeFailed(error.to_string())),
                    }
                };
                drop(sources);
                // `check_dirty_tracking`'s copy of RAM as of the dirty log's parent.
                let mut armed_copy = None;
                if self.check_dirty_tracking && cpus[0].page_has_code.dirty_log().is_armed() {
                    armed_copy = cpus[0].ram.duplicate();
                }
                let mut active = 0;
                let reason = loop {
                    let reason = match &resumed {
                        Err(reason) => reason.clone(),
                        Ok(()) => {
                            let (reason, ran, now_active) =
                                self.run_vcpus(cpus, active, threaded, &mut vcpu_summary);
                            cpus = ran;
                            active = now_active;
                            reason
                        }
                    };
                    if reason != ExitReason::Stopped || self.requests.ended.load(Ordering::SeqCst) {
                        break reason;
                    }
                    // Only `checkpoint` raises the stop without `ended`, and it does so with its
                    // request queued under the same lock: so there is work here, and requests
                    // that arrive while it is written are drained before the stop is lowered.
                    let mut lost_a_vcpu = false;
                    while let Some(batch) = self.requests.take_or_resume(&self.stop) {
                        for request in batch {
                            let mut sink = request.sink;
                            let result = if lost_a_vcpu {
                                Err(SnapshotError("a vCPU thread ended abnormally".into()))
                            } else {
                                self.checkpoint(
                                    &mut cpus,
                                    active,
                                    threaded,
                                    &mut *sink,
                                    &mut armed_copy,
                                )
                            };
                            lost_a_vcpu |= cpus.len() != cpu_count;
                            (request.done)(result);
                        }
                    }
                    if lost_a_vcpu {
                        break ExitReason::Internal("a vCPU thread ended abnormally".into());
                    }
                };
                // Whatever ended the run, the checkpoints asked for meanwhile will not happen,
                // and a stop raised only for them must not end the next boot.
                self.requests
                    .refuse_checkpoints("the machine stopped before the checkpoint");
                self.requests.lower_stop_unless_ended(&self.stop);
                let reason = match (&reason, self.suspend_to.as_deref_mut()) {
                    (ExitReason::Stopped, Some(sink))
                        if self.requests.suspend.load(Ordering::SeqCst) =>
                    {
                        match suspend::save(&mut cpus, active, threaded, sink, SaveKind::Full) {
                            Ok(_) => ExitReason::Suspended,
                            Err(error) => ExitReason::SuspendFailed(error.to_string()),
                        }
                    }
                    _ => reason,
                };
                cpu = if cpus.is_empty() {
                    // Every vCPU thread panicked before handing its `Cpu` back; there is nothing
                    // left to inspect.
                    Cpu::new(
                        Vec::new(),
                        Default::default(),
                        Default::default(),
                        Arc::new(AtomicBool::new(false)),
                        None,
                        Default::default(),
                    )
                } else {
                    cpus.swap_remove(active)
                };
                reason
            }
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
        #[cfg(not(freestanding))]
        if let Some(input) = &input {
            input.set_device_waker(None);
        }
        #[cfg(not(freestanding))]
        if let Some(gpu) = &gpu {
            gpu.set_device_waker(None);
        }
        let finished = Finished {
            ram: core::mem::take(&mut cpu.ram),
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
    /// With `DisplayMode::Framebuffer`: the screen.
    display: Option<DisplayHandle>,
    /// With `MachineConfig::input` (by default together with the screen): keyboard and tablet.
    input: Option<InputHandle>,
    stop: Arc<AtomicBool>,
    requests: Arc<HostRequests>,
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
        let snapshots = cfg.suspend_to.is_some() || !cfg.resume_from.is_empty();
        if snapshots && (cfg.verify || cfg.boot_disk) {
            return Err(Error::BadConfig(
                "suspend and resume cannot be combined with verify_jit (its twin is not saved) or \
                 a BIOS disk boot (the BIOS services keep state on the host)"
                    .into(),
            ));
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
        // Xen keeps the 16550 for itself and hands dom0 its console as `hvc0`; a virtio console
        // would be a second `hvc` that never sees Xen's output.
        let kinds = if cfg.consoles.is_empty() && cfg.hypervisor.is_some() {
            vec![ConsoleKind::Serial]
        } else if cfg.consoles.is_empty() {
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
        // Its GetRandom follows the same rules as virtio-rng; without host entropy (a freestanding
        // embedder that gave none) the TPM still works, on the seeded stream.
        let tpm = cfg.tpm.then(|| {
            let seeded = RngSource::Seeded(
                cfg.random_seed.unwrap_or(DEFAULT_RANDOM_SEED) ^ TPM_RANDOM_SEED_SALT,
            );
            let random = if cfg.verify {
                seeded
            } else {
                cfg.entropy
                    .clone()
                    .or_else(host_entropy)
                    .map_or(seeded, RngSource::Host)
            };
            // Without the embedder's store the TPM still keeps its seeds and NV across guest
            // resets, in a store of its own that lives as long as the machine.
            let store = cfg.key_value_store.clone().unwrap_or_else(|| {
                let store: Box<dyn KeyValueStore> = Box::new(MemoryKeyValueStore::new());
                Arc::new(Mutex::new(store))
            });
            TpmConfig { random, store }
        });
        if cfg.secure_boot && cfg.uefi_firmware.is_none() {
            return Err(Error::BadConfig(
                "secure_boot is enforced by the UEFI firmware: it needs uefi_firmware".into(),
            ));
        }
        let has_display = cfg.display == DisplayMode::Framebuffer;
        let has_input = cfg.input.unwrap_or(has_display);
        let (display_width, display_height) = cfg.display_size;
        if (has_display || has_input)
            && (!(display::MIN_WIDTH..=display::MAX_WIDTH).contains(&display_width)
                || !(display::MIN_HEIGHT..=display::MAX_HEIGHT).contains(&display_height))
        {
            return Err(Error::BadConfig(format!(
                "display_size must be between {}x{} and {}x{}",
                display::MIN_WIDTH,
                display::MIN_HEIGHT,
                display::MAX_WIDTH,
                display::MAX_HEIGHT
            )));
        }
        if let Some(link) = &cfg.gpu {
            validate_gpu(link, has_display, cfg.verify)?;
        }
        let functions = virtio_console_count
            + disk_count
            + cfg.net_macs.len()
            + rng.is_some() as usize
            + if has_display { DISPLAY_FUNCTIONS } else { 0 }
            + if has_input { INPUT_FUNCTIONS } else { 0 };
        if functions > MAX_VIRTIO_FUNCTIONS {
            return Err(Error::BadConfig(format!(
                "{functions} virtio devices configured (consoles, disks, NICs, the display, \
                 input and the entropy source); the PCI bus has room for {MAX_VIRTIO_FUNCTIONS}"
            )));
        }
        let mut cmdline = cfg.cmdline.clone().unwrap_or_else(|| {
            // The first console is the guest's primary one.
            let default = if cfg.hypervisor.is_some() {
                DEFAULT_DOM0_CMDLINE
            } else if kinds[0] == ConsoleKind::Virtio {
                DEFAULT_CMDLINE
            } else {
                DEFAULT_SERIAL_CMDLINE
            };
            // The defaults are the table-less ones (`Platform` hosts need `acpi=off`).
            let mut default = if cfg.acpi {
                default.replace(ACPI_OFF_WORD, "").replace("  ", " ")
            } else {
                default.to_string()
            };
            // The last `console=` is `/dev/console`, where init and its shell go: the screen.
            // The text console stays a kernel log console, and a `cmdline_append` with its own
            // `console=` still comes after this and wins.
            if has_display && cfg.console_on_screen {
                default.push_str(" console=tty0");
            }
            default
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
            if !cfg.kernel.is_empty()
                && cfg.hypervisor.is_none()
                && !boot::has_efi_header(&cfg.kernel)
            {
                return Err(Error::BadConfig(
                    "kernel has no EFI stub; boot it with firmware none".into(),
                ));
            }
        }
        let hypervisor = match &cfg.hypervisor {
            Some(_) if cfg.kernel.is_empty() => {
                return Err(Error::BadConfig(
                    "a hypervisor needs a kernel to boot as its dom0".into(),
                ));
            }
            Some(_) if cfg.uefi_firmware.is_none() => {
                return Err(Error::BadConfig(
                    "a hypervisor is booted by the UEFI firmware: it needs uefi_firmware".into(),
                ));
            }
            Some(image) => {
                let text = cfg
                    .hypervisor_cmdline
                    .clone()
                    .unwrap_or_else(|| DEFAULT_HYPERVISOR_CMDLINE.into());
                Some((boot::hypervisor::prepare(image)?, text))
            }
            None => None,
        };
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
            tpm: cfg.tpm,
        });
        let uefi = cfg.uefi_firmware.as_ref().map(|image| {
            let params = boot::FirmwareParams {
                key_value_store: cfg.key_value_store.is_some(),
                unix_time: cfg.unix_time.unwrap_or_else(host_unix_time),
                random_seed: cfg.random_seed.unwrap_or(DEFAULT_RANDOM_SEED),
                rsdp: 0,
                acpi_region: (0, 0),
                framebuffer: has_display.then_some(cfg.display_size),
                secure_boot: cfg.secure_boot,
                tpm: cfg.tpm,
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
                    uefi_payload(
                        &cfg.kernel,
                        cfg.initrd.as_deref(),
                        &cmdline,
                        hypervisor.as_ref(),
                    )
                    .as_ref(),
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
        let display = has_display.then(|| DisplayHandle::with_size(display_width, display_height));
        let input = has_input.then(|| match &display {
            Some(display) => InputHandle::for_display(display.clone()),
            None => InputHandle::without_display(display_width, display_height),
        });
        let devcfg = DevConfig {
            disks: cfg.disks,
            nets,
            virtio_consoles,
            iommu: cfg.iommu,
            key_value_store: cfg.key_value_store.clone(),
            cpus: cfg.cpus,
            smt: cfg.smt,
            rng: rng.clone(),
            tpm: tpm.clone(),
            display: display_devices(
                &display,
                &cfg.gpu,
                uefi.as_ref().map(|(_, params)| params),
                ram.len() as u64,
            ),
            input: input.clone(),
        };
        // A copy, not a clone: a clone of a `GuestRam` shares the memory.
        let boot_ram = if cfg.verify {
            let copy = ram.duplicate().ok_or_else(|| {
                Error::BadConfig("cannot allocate a second copy of guest memory for verify".into())
            })?;
            Some(copy)
        } else {
            None
        };
        // Verification cannot restart (the twin's devices cannot be rebuilt in step), so it keeps
        // no image and a reset ends the run.
        let reboot =
            (cfg.reboot && !cfg.verify && cfg.firmware.is_none() && !cfg.boot_disk).then(|| {
                RebootImage {
                    kernel: cfg.kernel,
                    initrd: cfg.initrd,
                    cmdline,
                    hypervisor,
                    nets: devcfg.nets.clone(),
                    virtio_consoles: devcfg.virtio_consoles.clone(),
                    iommu: cfg.iommu,
                    cpus: cfg.cpus,
                    smt: cfg.smt,
                    acpi,
                    key_value_store: cfg.key_value_store,
                    rng: devcfg.rng.clone(),
                    tpm,
                    display: display.clone(),
                    input: input.clone(),
                    gpu: cfg.gpu.clone(),
                    uefi,
                }
            });
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(HostRequests::default());
        if cfg.verify || cfg.boot_disk {
            requests.closed.store(true, Ordering::SeqCst);
        }
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
                requests: requests.clone(),
                suspend_to: cfg.suspend_to,
                resume_from: cfg.resume_from,
                check_dirty_tracking: cfg.check_dirty_tracking,
            }),
            #[cfg(not(freestanding))]
            thread: None,
            #[cfg(freestanding)]
            started: false,
            consoles,
            nets: net_handles,
            display,
            input,
            stop,
            requests,
        })
    }

    /// The screen, with `DisplayMode::Framebuffer`.
    pub fn display(&self) -> Option<DisplayHandle> {
        self.display.clone()
    }

    /// The keyboard and tablet, when the machine has them (`MachineConfig::input`; by default
    /// together with a screen).
    pub fn input(&self) -> Option<InputHandle> {
        self.input.clone()
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
        StopHandle {
            stop: self.stop.clone(),
            requests: self.requests.clone(),
        }
    }

    pub fn stop(&self) {
        self.stop_handle().stop();
    }

    /// `StopHandle::suspend`.
    pub fn suspend(&self) {
        self.stop_handle().suspend();
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

mod suspend;
use suspend::SaveKind;
pub use suspend::read_snapshot_info;
#[cfg(test)]
mod tests;
