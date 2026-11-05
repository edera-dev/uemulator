//! The options every way of booting shares. They come in groups so each command takes exactly the
//! ones that mean something to it:
//!
//! - `HardwareArgs`: memory, disks, NICs, consoles, display, JIT. Every command that describes a
//!   machine.
//! - `LinuxArgs`: command line, reboot behaviour and the hypervisor the kernel can run under. Not
//!   on firmware boots, which have none of them.
//! - `RemoveArgs`: `--remove-<device> INDEX`, for commands that start from an existing device list.
//! - `NewDiskArgs`: `--create-disk`, for commands that own a directory to put the image in.
//! - `RuntimeArgs`: debugging and pacing for one run; never persisted.
//!
//! `Settings` is what a named machine persists and what `run` consumes: a command line is a patch
//! on it (`remove_devices`, then `add`), and `advanced boot` patches an empty one.

use std::path::PathBuf;

use clap::error::ErrorKind;
use clap::{ArgAction, ArgMatches, Args, Command};
use serde::{Deserialize, Serialize};

/// clap's flag actions (`SetTrue`, `Count`) record one argument index however often a flag is
/// repeated, which would lose the interleaving of `--network` with its `--network-link-*` flags.
/// So the link flags are appending options that take no value on the command line
/// (`default_missing_value`); every occurrence then has an index of its own.
const REPEATABLE_FLAG: &str = "on";

/// What a boolean option written without a value means.
const BOOL_FLAG_DEFAULT: &str = "true";

/// The NIC models `--network` accepts. Virtio is the only one the library has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetModel {
    Virtio,
}

/// What sits on the host side of a NIC (`--network-link-*`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetLink {
    /// NAT to the host network.
    User,
    /// A built-in gateway that answers ARP and ICMP echo.
    Loop,
    /// `LOCAL,REMOTE`, unparsed.
    Udp(String),
}

/// One NIC. Without a link nothing is attached on the host side and the guest's frames go nowhere.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Nic {
    pub model: NetModel,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<NetLink>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsoleArg {
    /// ttyS0 on the 16550; for kernels without a built-in virtio_console driver
    Serial,
    /// hvc0; the host terminal size is forwarded and follows resizes
    Virtio,
}

/// `--display KIND`: whether the machine has a screen, and which console the guest's shell is on.
/// Keyboard and mouse follow the screen unless `--input` says otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayArg {
    /// Text consoles only: the terminal is the machine's console
    #[default]
    Serial,
    /// A virtio-gpu screen, shown in a browser by the built-in viewer (`--viewer-address`); the
    /// kernel's console moves to the screen (tty0) and the terminal keeps a copy of the kernel log
    Framebuffer,
    /// The screen as with framebuffer, but the terminal stays the machine's console; the screen
    /// shows what the guest runs on its virtual terminals (a distro's login on tty1)
    Both,
}

impl DisplayArg {
    pub fn has_screen(self) -> bool {
        self != DisplayArg::Serial
    }
}

/// Where the web viewer listens unless `--viewer-address` says otherwise: loopback only, on the
/// port VNC's HTTP viewers have used (5800).
pub const DEFAULT_VIEWER_ADDRESS: &str = "127.0.0.1:5800";

/// `--smp MODE`: how the vCPUs of a multi-processor guest run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SmpArg {
    /// One host thread per vCPU, falling back to timesharing where that is not possible
    Auto,
    /// Take turns on one host thread, deterministically on the virtual clock
    Timeshare,
    /// One host thread per vCPU, or fail
    Threads,
}

/// `--disk PATH[,ro]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskSpec {
    pub path: PathBuf,
    #[serde(default)]
    pub read_only: bool,
}

/// Splits `--disk PATH[,ro]` into the path and whether the disk is read-only.
pub fn parse_disk_spec(spec: &str) -> DiskSpec {
    match spec.strip_suffix(",ro") {
        Some(path) => DiskSpec {
            path: path.into(),
            read_only: true,
        },
        None => DiskSpec {
            path: spec.into(),
            read_only: false,
        },
    }
}

/// One entry of the disk list a command line builds, in command-line order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiskStep {
    Attach(DiskSpec),
    /// A new sparse image of this many MiB.
    Create {
        mib: u64,
    },
}

#[derive(Args, Default)]
pub struct HardwareArgs {
    /// Guest memory in MiB, 64 to 1047552 (1 TiB less the PCI hole) [default: 2048, 3072 for a fedora install]
    #[arg(long, short = 'm')]
    pub memory: Option<usize>,
    /// Guest vCPUs, 1 to 64 (each runs on its own host thread unless `--smp timeshare`) [default: 1]
    #[arg(long)]
    pub cpus: Option<usize>,
    /// How the vCPUs run: a host thread each on the host clock (auto, threads) or in turns on one
    /// thread, deterministically (timeshare) [default: auto]
    #[arg(long, value_enum, value_name = "MODE")]
    pub smp: Option<SmpArg>,
    /// Attach a disk image (raw or udisk) as a virtio-blk device; append ",ro" for read-only. A
    /// udisk overlay's base is found from the overlay and opened read-only. Repeat for more
    /// disks: the guest sees them in order (vda, vdb, ...)
    #[arg(long, value_name = "PATH[,ro]")]
    disk: Vec<String>,
    /// Add a NIC. Repeat for more. Give it a host side with the `--network-link-*` option that
    /// follows it
    #[arg(long, value_enum, value_name = "MODEL")]
    network: Vec<NetModel>,
    /// Link the preceding --network to the host network with user-mode NAT (DHCP, DNS, ICMP echo,
    /// UDP and TCP; the guest is 10.0.2.15 and the gateway 10.0.2.2)
    #[arg(long, action = ArgAction::Append, num_args = 0..=1, require_equals = true,
        default_missing_value = REPEATABLE_FLAG, value_parser = [REPEATABLE_FLAG],
        hide_possible_values = true, value_name = "")]
    network_link_user: Vec<String>,
    /// Link the preceding --network to a built-in gateway that only replies to ARP and ICMP echo
    /// for 10.0.2.2: no DHCP, nothing is forwarded
    #[arg(long, action = ArgAction::Append, num_args = 0..=1, require_equals = true,
        default_missing_value = REPEATABLE_FLAG, value_parser = [REPEATABLE_FLAG],
        hide_possible_values = true, value_name = "")]
    network_link_loop: Vec<String>,
    /// Link the preceding --network to a UDP socket pair
    #[arg(long, value_name = "LOCAL,REMOTE")]
    network_link_udp: Vec<String>,
    /// Add a console. Repeat for more: the first is the guest's primary console and the one the
    /// terminal is attached to; the others have no host side (their output is dropped). Serial can
    /// be given once [default: virtio]
    #[arg(long, value_enum, value_name = "KIND")]
    pub console: Vec<ConsoleArg>,
    /// The machine's display: serial (text consoles only), framebuffer (a screen in the browser
    /// viewer that is the guest's console) or both (the screen, with the terminal as the console)
    /// [default: serial]
    #[arg(long, value_enum, value_name = "KIND")]
    pub display: Option<DisplayArg>,
    /// Add a virtio keyboard and tablet, fed by the browser viewer; `--input=false` leaves them
    /// out. They come with a screen by default, and without one the viewer offers only them
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub input: Option<bool>,
    /// Run the interpreter only; `--disable-jit=false` turns it back on
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub disable_jit: Option<bool>,
    /// Leave out the virtio-rng device that feeds the guest's entropy pool from the host;
    /// `--disable-rng=false` turns it back on
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub disable_rng: Option<bool>,
    /// Add an AMD IOMMU (AMD-Vi): DMA of every PCI function and interrupts through the I/O APIC
    /// and MSI go through it once the guest enables it; `--iommu=false` turns it back off
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub iommu: Option<bool>,
    /// Add a TPM 2.0 (CRB interface, ACPI TPM2 table). The UEFI firmware measures the boot into
    /// it and hands the event log to the OS; it does PCRs, hashing and random numbers, not keys
    /// or sealing. `--tpm=false` takes it out again
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub tpm: Option<bool>,
    /// Have the UEFI firmware enforce Secure Boot. The first boot with it enrolls a uemulator
    /// PK and Microsoft's KEKs, db (Windows and third-party UEFI CAs, 2011 and 2023) and dbx into
    /// the machine's nvram: shim-based distributions and Windows boot, unsigned kernels do not.
    /// `--secure-boot=false` stops enforcing and keeps the keys
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub secure_boot: Option<bool>,
    /// The disks in command-line order, `--disk` and `--create-disk` interleaved. `order` fills it.
    #[arg(skip)]
    pub disk_steps: Vec<DiskStep>,
    /// The NICs with their links, in command-line order. `order` fills it.
    #[arg(skip)]
    pub nics: Vec<Nic>,
}

#[derive(Args, Default)]
pub struct LinuxArgs {
    /// Kernel command line; an empty value clears a saved one
    #[arg(long)]
    pub cmdline: Option<String>,
    /// Append to the kernel command line; an empty value clears a saved one
    #[arg(long, value_name = "ARGS")]
    pub append: Option<String>,
    /// End the run with "machine reset" when the guest resets instead of rebooting it in place;
    /// `--no-reboot=false` turns it back off
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = BOOL_FLAG_DEFAULT,
        value_name = "BOOL")]
    pub no_reboot: Option<bool>,
    #[command(flatten)]
    pub hypervisor: HypervisorArgs,
}

/// A hypervisor with the kernel as its dom0. A named machine keeps a copy of the image in its
/// directory (`--remove-hypervisor` drops it) and boots it with the installer's kernel; `quick
/// boot` takes one for that run only.
#[derive(Args, Default)]
pub struct HypervisorArgs {
    /// A hypervisor to boot with the kernel as its dom0: Xen as xen.gz (or the uncompressed
    /// xen), or Edera's xen.efi (an upstream xen.efi cannot take its kernel from the firmware;
    /// give such a Xen as xen.gz). Needs the UEFI firmware. Xen keeps the 16550 and passes dom0's
    /// console (hvc0) through it, so the terminal goes on a serial console first in the list;
    /// --cmdline/--append are dom0's. A named machine runs only installer boots under it: an
    /// installed disk brings its own boot loader
    #[arg(long, value_name = "IMAGE")]
    pub hypervisor: Option<PathBuf>,
    /// The hypervisor's own command line; an empty value clears a saved one [default:
    /// console=com1 com1=115200,8n1]
    #[arg(long, value_name = "ARGS")]
    pub hypervisor_cmdline: Option<String>,
}

/// Indices count from 0 in the order the guest sees the devices, and refer to the list as it is
/// before this command line changes anything.
#[derive(Args, Default)]
pub struct RemoveArgs {
    /// Detach the disk at INDEX. The image file is left where it is
    #[arg(long, value_name = "INDEX")]
    pub remove_disk: Vec<usize>,
    /// Remove the NIC at INDEX
    #[arg(long, value_name = "INDEX")]
    pub remove_network: Vec<usize>,
    /// Remove the console at INDEX
    #[arg(long, value_name = "INDEX")]
    pub remove_console: Vec<usize>,
    /// Remove the hypervisor and its command line; the machine's copy of the image is deleted
    #[arg(long)]
    pub remove_hypervisor: bool,
}

#[derive(Args, Default)]
pub struct NewDiskArgs {
    /// Create a sparse disk image of MIB MiB in the machine's directory and attach it after the
    /// disks already there. Repeat for more
    #[arg(long, value_name = "MIB", value_parser = clap::value_parser!(u64).range(1..))]
    create_disk: Vec<u64>,
}

/// What a named machine boots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BootSource {
    /// The disk once something is installed on it, else the installer: the boot order of real
    /// hardware, which tries its disks before the network. A disk counts as installed when its
    /// first 32 KiB are not all zeros
    #[default]
    Auto,
    /// The distro's netboot kernel and initramfs, which run its installer
    Installer,
    /// The first disk, through the machine's firmware (`--firmware`, UEFI by default): the boot
    /// loader the installer put there loads the system, so kernel arguments come from that boot
    /// loader and `--cmdline` and `--append` do not apply
    Disk,
}

/// The firmware a machine starts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Firmware {
    /// The built-in UEFI firmware: it boots a kernel given to it first (like QEMU's `-kernel`
    /// under OVMF; the kernel needs an EFI stub), else the disk's EFI system partition loader.
    /// Its variables live in the machine's `nvram.kv`
    #[default]
    Uefi,
    /// The library's BIOS: the disk's boot sector runs in real mode. A kernel is not booted
    /// through it: a machine with this firmware runs its installer directly
    Bios,
    /// No firmware: the kernel is entered directly in 64-bit mode (the boot protocol), which is
    /// the way to run a kernel without an EFI stub. A disk cannot be booted this way
    None,
}

impl Firmware {
    pub fn name(self) -> &'static str {
        match self {
            Firmware::Uefi => "uefi",
            Firmware::Bios => "bios",
            Firmware::None => "none",
        }
    }
}

#[derive(Args, Default)]
pub struct FirmwareArgs {
    /// What starts the machine [default: uefi]
    #[arg(long, value_enum, value_name = "FIRMWARE")]
    pub firmware: Option<Firmware>,
}

#[derive(Args, Default)]
pub struct BootSourceArgs {
    /// What to boot [default: auto; `reinstall` sets installer]
    #[arg(long, value_enum, value_name = "SOURCE")]
    pub boot_from: Option<BootSource>,
}

/// Variables written into the nvram before a boot through the UEFI firmware.
#[derive(Args, Default)]
pub struct UefiBootArgs {
    /// Open the UEFI firmware's boot menu on this boot (Esc, F2 or F12 during the countdown
    /// does too)
    #[arg(long)]
    pub boot_menu: bool,
    /// Seconds the UEFI firmware counts down before booting, kept in the nvram (its `Timeout`
    /// variable; 0 boots at once, 65535 waits in the menu)
    #[arg(long, value_name = "SECS")]
    pub boot_timeout: Option<u16>,
}

impl UefiBootArgs {
    pub fn any(&self) -> bool {
        self.boot_menu || self.boot_timeout.is_some()
    }
}

#[derive(Args, Default)]
pub struct RuntimeArgs {
    /// Run the JIT in lockstep with the interpreter and stop at the first divergence
    #[arg(long)]
    pub verify: bool,
    /// Dump CPU state and the guest kernel log on exit
    #[arg(long)]
    pub debug: bool,
    /// Fix the guest RTC to this Unix time
    #[arg(long, value_name = "UNIX")]
    pub time: Option<u64>,
    /// Stop the machine after this many seconds
    #[arg(long, value_name = "SECS")]
    pub timeout: Option<u64>,
    /// Where the viewer of a machine with a screen or input listens, as HOST:PORT: an address or
    /// a host name (the viewer then answers to that name), port 0 picks a free one. Anyone who
    /// can reach it controls the machine's keyboard and mouse [default: 127.0.0.1:5800]
    #[arg(long, value_name = "HOST:PORT", conflicts_with_all = ["viewer_host", "viewer_port"])]
    pub viewer_address: Option<String>,
    /// The viewer's host alone (an address, or a name it then answers to); the port stays
    /// --viewer-port or 5800
    #[arg(long, value_name = "HOST")]
    pub viewer_host: Option<String>,
    /// The viewer's port alone (0 picks a free one); the host stays --viewer-host or 127.0.0.1
    #[arg(long, value_name = "PORT")]
    pub viewer_port: Option<u16>,
    /// Another host name the viewer answers to, for a reverse proxy or a DNS name of a viewer
    /// bound to every address (0.0.0.0). Repeat for more. Names not listed get 403, which is what
    /// keeps other web sites from driving the machine through DNS rebinding
    #[arg(long, value_name = "NAME")]
    pub viewer_allow_host: Vec<String>,
}

/// Everything about a machine that outlives one run, and what `run` needs to build it. `None` and
/// empty lists mean "not chosen": the defaults (2048 MiB, one virtio console, the distro's command
/// line) are applied when the machine starts, so changing a machine's distro changes them too.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpus: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smp: Option<SmpArg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmdline: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub append: Option<String>,
    pub no_reboot: bool,
    pub disable_jit: bool,
    pub disable_rng: bool,
    pub iommu: bool,
    pub tpm: bool,
    pub secure_boot: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<DisplayArg>,
    /// `None` follows the display, so a machine saved before input could be chosen keeps its
    /// keyboard with its screen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<bool>,
    // The arrays come last: TOML wants every plain value before the first table.
    pub consoles: Vec<ConsoleArg>,
    pub disks: Vec<DiskSpec>,
    pub networks: Vec<Nic>,
}

/// `Some("")` is how a command line clears a saved string.
pub fn set_or_clear(slot: &mut Option<String>, value: &Option<String>) {
    match value.as_deref() {
        Some("") => *slot = None,
        Some(text) => *slot = Some(text.to_string()),
        None => {}
    }
}

/// Indices to drop from a list of `len` entries, validated and deduplicated.
fn removal_set(flag: &str, noun: &str, len: usize, indices: &[usize]) -> Result<Vec<bool>, String> {
    let mut drop = vec![false; len];
    for &index in indices {
        if index >= len {
            return Err(match len {
                0 => format!("{flag} {index}: there are no {noun}s"),
                1 => format!("{flag} {index}: there is 1 {noun}, numbered 0"),
                _ => format!(
                    "{flag} {index}: there are {len} {noun}s, numbered 0 to {}",
                    len - 1
                ),
            });
        }
        drop[index] = true;
    }
    Ok(drop)
}

fn retain_unmarked<T>(list: &mut Vec<T>, drop: &[bool]) {
    let mut index = 0;
    list.retain(|_| {
        index += 1;
        !drop[index - 1]
    });
}

impl Settings {
    /// Applies the `--remove-*` options and returns the disks that were detached. Run it before
    /// `add`: indices refer to the list as it was, and nothing changes if any index is wrong.
    pub fn remove_devices(&mut self, remove: &RemoveArgs) -> Result<Vec<DiskSpec>, String> {
        let disks = removal_set(
            "--remove-disk",
            "disk",
            self.disks.len(),
            &remove.remove_disk,
        )?;
        let networks = removal_set(
            "--remove-network",
            "network",
            self.networks.len(),
            &remove.remove_network,
        )?;
        let consoles = removal_set(
            "--remove-console",
            "console",
            self.consoles.len(),
            &remove.remove_console,
        )?;
        let detached = self
            .disks
            .iter()
            .zip(&disks)
            .filter(|(_, dropped)| **dropped)
            .map(|(disk, _)| disk.clone())
            .collect();
        retain_unmarked(&mut self.disks, &disks);
        retain_unmarked(&mut self.networks, &networks);
        retain_unmarked(&mut self.consoles, &consoles);
        Ok(detached)
    }

    /// Applies what the command line chose: scalars replace, devices are appended in command-line
    /// order. `new_disks` is `hardware.disk_steps` with the images made and the paths settled,
    /// which only the caller can do (it knows where a created image goes).
    pub fn add(&mut self, hardware: &HardwareArgs, linux: &LinuxArgs, new_disks: Vec<DiskSpec>) {
        if hardware.memory.is_some() {
            self.memory = hardware.memory;
        }
        if hardware.cpus.is_some() {
            self.cpus = hardware.cpus;
        }
        if hardware.smp.is_some() {
            self.smp = hardware.smp;
        }
        if let Some(disable_jit) = hardware.disable_jit {
            self.disable_jit = disable_jit;
        }
        if let Some(disable_rng) = hardware.disable_rng {
            self.disable_rng = disable_rng;
        }
        if let Some(iommu) = hardware.iommu {
            self.iommu = iommu;
        }
        if let Some(tpm) = hardware.tpm {
            self.tpm = tpm;
        }
        if let Some(secure_boot) = hardware.secure_boot {
            self.secure_boot = secure_boot;
        }
        if hardware.display.is_some() {
            self.display = hardware.display;
        }
        if hardware.input.is_some() {
            self.input = hardware.input;
        }
        set_or_clear(&mut self.cmdline, &linux.cmdline);
        set_or_clear(&mut self.append, &linux.append);
        if let Some(no_reboot) = linux.no_reboot {
            self.no_reboot = no_reboot;
        }
        self.disks.extend(new_disks);
        self.networks.extend(hardware.nics.iter().cloned());
        self.consoles.extend(hardware.console.iter().copied());
    }
}

/// The disks of a command that cannot create images (`advanced boot`, `quick boot`).
pub fn attach_only(steps: &[DiskStep]) -> Result<Vec<DiskSpec>, String> {
    steps
        .iter()
        .map(|step| match step {
            DiskStep::Attach(disk) => Ok(disk.clone()),
            DiskStep::Create { .. } => {
                Err("--create-disk needs a machine to keep the image in".into())
            }
        })
        .collect()
}

/// Whether the leaf command defines `id`: querying one it does not is a debug-build assertion in
/// clap, and the shared groups are not on every command.
fn defines(leaf: &ArgMatches, id: &str) -> bool {
    leaf.try_contains_id(id).unwrap_or(false)
}

/// The disk list in command-line order. `--disk` and `--create-disk` are separate options, so
/// their interleaving is recovered from the argument indices: the n-th disk the guest sees must be
/// the n-th flag the user typed.
fn disk_steps(leaf: &ArgMatches) -> Vec<DiskStep> {
    let mut ordered: Vec<(usize, DiskStep)> = Vec::new();
    if let (Some(indices), Some(specs)) = (leaf.indices_of("disk"), leaf.get_many::<String>("disk"))
    {
        for (index, spec) in indices.zip(specs) {
            ordered.push((index, DiskStep::Attach(parse_disk_spec(spec))));
        }
    }
    if defines(leaf, "create_disk")
        && let (Some(indices), Some(sizes)) = (
            leaf.indices_of("create_disk"),
            leaf.get_many::<u64>("create_disk"),
        )
    {
        for (index, mib) in indices.zip(sizes) {
            ordered.push((index, DiskStep::Create { mib: *mib }));
        }
    }
    ordered.sort_by_key(|(index, _)| *index);
    ordered.into_iter().map(|(_, step)| step).collect()
}

/// The NICs in command-line order: `--network` declares one and every `--network-link-*` binds the
/// one declared most recently, so a link before any `--network` or a second link on one NIC is an
/// error rather than a guess.
fn nics(leaf: &ArgMatches) -> Result<Vec<Nic>, String> {
    enum Event {
        Declare(NetModel),
        Link(&'static str, NetLink),
    }
    let mut events: Vec<(usize, Event)> = Vec::new();
    if let (Some(indices), Some(models)) = (
        leaf.indices_of("network"),
        leaf.get_many::<NetModel>("network"),
    ) {
        for (index, model) in indices.zip(models) {
            events.push((index, Event::Declare(*model)));
        }
    }
    for (id, flag, link) in [
        ("network_link_user", "--network-link-user", NetLink::User),
        ("network_link_loop", "--network-link-loop", NetLink::Loop),
    ] {
        for index in leaf.indices_of(id).into_iter().flatten() {
            events.push((index, Event::Link(flag, link.clone())));
        }
    }
    if let (Some(indices), Some(specs)) = (
        leaf.indices_of("network_link_udp"),
        leaf.get_many::<String>("network_link_udp"),
    ) {
        for (index, spec) in indices.zip(specs) {
            events.push((
                index,
                Event::Link("--network-link-udp", NetLink::Udp(spec.clone())),
            ));
        }
    }
    events.sort_by_key(|(index, _)| *index);
    let mut nics: Vec<Nic> = Vec::new();
    for (_, event) in events {
        match event {
            Event::Declare(model) => nics.push(Nic { model, link: None }),
            Event::Link(flag, link) => {
                let nic = nics
                    .last_mut()
                    .ok_or_else(|| format!("{flag} must follow the --network it links"))?;
                if nic.link.is_some() {
                    return Err(format!("{flag}: that --network already has a link"));
                }
                nic.link = Some(link);
            }
        }
    }
    Ok(nics)
}

impl HardwareArgs {
    /// Fills `disk_steps` and `nics`, the parts clap's derive cannot do: it keeps each option's
    /// occurrences apart and loses how they interleave.
    pub fn order(&mut self, command: &Command, leaf: &ArgMatches) -> Result<(), clap::Error> {
        self.disk_steps = disk_steps(leaf);
        self.nics = nics(leaf)
            .map_err(|message| command.clone().error(ErrorKind::ArgumentConflict, message))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk(path: &str, read_only: bool) -> DiskSpec {
        DiskSpec {
            path: path.into(),
            read_only,
        }
    }

    #[test]
    fn disk_spec_without_suffix_is_read_write() {
        assert_eq!(parse_disk_spec("disk.img"), disk("disk.img", false));
    }

    #[test]
    fn disk_spec_with_ro_suffix_is_read_only() {
        assert_eq!(parse_disk_spec("disk.img,ro"), disk("disk.img", true));
    }

    #[test]
    fn disk_spec_only_strips_one_ro_suffix() {
        assert_eq!(parse_disk_spec("a,ro,ro"), disk("a,ro", true));
    }

    #[test]
    fn disk_spec_other_suffixes_stay_part_of_the_path() {
        assert_eq!(parse_disk_spec("disk.img,rw"), disk("disk.img,rw", false));
        assert_eq!(parse_disk_spec("ro"), disk("ro", false));
        assert_eq!(parse_disk_spec("disk.img,RO"), disk("disk.img,RO", false));
        assert_eq!(parse_disk_spec("a,roo"), disk("a,roo", false));
    }

    #[test]
    fn disk_spec_edge_cases() {
        assert_eq!(parse_disk_spec(",ro"), disk("", true));
        assert_eq!(parse_disk_spec(""), disk("", false));
        assert_eq!(
            parse_disk_spec("/a b/c d.img,ro"),
            disk("/a b/c d.img", true)
        );
    }

    fn settings_with_devices() -> Settings {
        Settings {
            disks: vec![disk("a", false), disk("b", true), disk("c", false)],
            networks: vec![
                Nic {
                    model: NetModel::Virtio,
                    link: Some(NetLink::User),
                },
                Nic {
                    model: NetModel::Virtio,
                    link: None,
                },
            ],
            consoles: vec![ConsoleArg::Virtio, ConsoleArg::Serial],
            ..Settings::default()
        }
    }

    #[test]
    fn removals_refer_to_the_list_before_any_of_them_apply() {
        let mut settings = settings_with_devices();
        let detached = settings
            .remove_devices(&RemoveArgs {
                remove_disk: vec![2, 0],
                remove_network: vec![0],
                remove_console: vec![1],
                remove_hypervisor: false,
            })
            .unwrap();
        assert_eq!(detached, vec![disk("a", false), disk("c", false)]);
        assert_eq!(settings.disks, vec![disk("b", true)]);
        assert_eq!(settings.networks.len(), 1);
        assert_eq!(settings.networks[0].link, None);
        assert_eq!(settings.consoles, vec![ConsoleArg::Virtio]);
    }

    #[test]
    fn removing_the_same_index_twice_removes_it_once() {
        let mut settings = settings_with_devices();
        settings
            .remove_devices(&RemoveArgs {
                remove_disk: vec![1, 1],
                ..RemoveArgs::default()
            })
            .unwrap();
        assert_eq!(settings.disks, vec![disk("a", false), disk("c", false)]);
    }

    #[test]
    fn an_out_of_range_removal_names_the_valid_range_and_changes_nothing() {
        let mut settings = settings_with_devices();
        let before = settings.clone();
        let error = settings
            .remove_devices(&RemoveArgs {
                remove_disk: vec![0],
                remove_console: vec![2],
                ..RemoveArgs::default()
            })
            .unwrap_err();
        assert_eq!(
            error,
            "--remove-console 2: there are 2 consoles, numbered 0 to 1"
        );
        assert_eq!(settings, before);

        let error = Settings::default()
            .remove_devices(&RemoveArgs {
                remove_network: vec![0],
                ..RemoveArgs::default()
            })
            .unwrap_err();
        assert_eq!(error, "--remove-network 0: there are no networks");
    }

    #[test]
    fn add_replaces_scalars_appends_devices_and_leaves_the_rest() {
        let mut settings = settings_with_devices();
        settings.memory = Some(1024);
        settings.cmdline = Some("quiet".into());
        let hardware = HardwareArgs {
            disable_jit: Some(true),
            console: vec![ConsoleArg::Serial],
            nics: vec![Nic {
                model: NetModel::Virtio,
                link: Some(NetLink::Loop),
            }],
            ..HardwareArgs::default()
        };
        settings.add(&hardware, &LinuxArgs::default(), vec![disk("d", false)]);
        assert_eq!(settings.memory, Some(1024));
        assert_eq!(settings.cmdline.as_deref(), Some("quiet"));
        assert!(settings.disable_jit);
        assert!(!settings.disable_rng, "untouched");
        assert_eq!(settings.disks.len(), 4);
        assert_eq!(settings.disks[3], disk("d", false));
        assert_eq!(settings.networks.len(), 3);
        assert_eq!(settings.consoles.len(), 3);
    }

    #[test]
    fn an_empty_string_clears_a_saved_command_line() {
        let mut settings = Settings {
            cmdline: Some("root=/dev/vda".into()),
            append: Some("quiet".into()),
            no_reboot: true,
            ..Settings::default()
        };
        let linux = LinuxArgs {
            cmdline: Some(String::new()),
            append: Some("debug".into()),
            no_reboot: Some(false),
            hypervisor: HypervisorArgs::default(),
        };
        settings.add(&HardwareArgs::default(), &linux, Vec::new());
        assert_eq!(settings.cmdline, None);
        assert_eq!(settings.append.as_deref(), Some("debug"));
        assert!(!settings.no_reboot);
    }

    #[test]
    fn attach_only_rejects_a_created_disk() {
        let steps = [
            DiskStep::Attach(disk("a", false)),
            DiskStep::Create { mib: 1 },
        ];
        assert!(attach_only(&steps).unwrap_err().contains("--create-disk"));
        assert_eq!(attach_only(&steps[..1]).unwrap(), vec![disk("a", false)]);
    }

    #[test]
    fn settings_survive_a_round_trip_through_toml() {
        let settings = Settings {
            memory: Some(4096),
            cpus: Some(4),
            smp: Some(SmpArg::Timeshare),
            cmdline: Some("console=hvc0".into()),
            append: None,
            no_reboot: true,
            disable_jit: false,
            disable_rng: true,
            iommu: true,
            tpm: true,
            secure_boot: true,
            display: Some(DisplayArg::Both),
            input: Some(false),
            consoles: vec![ConsoleArg::Virtio, ConsoleArg::Serial],
            disks: vec![disk("/x/disk-0.img", false), disk("/y.iso", true)],
            networks: vec![
                Nic {
                    model: NetModel::Virtio,
                    link: Some(NetLink::User),
                },
                Nic {
                    model: NetModel::Virtio,
                    link: Some(NetLink::Udp("127.0.0.1:1,127.0.0.1:2".into())),
                },
                Nic {
                    model: NetModel::Virtio,
                    link: None,
                },
            ],
        };
        let text = toml::to_string(&settings).unwrap();
        assert_eq!(toml::from_str::<Settings>(&text).unwrap(), settings);
    }

    #[test]
    fn the_display_is_saved_as_a_plain_word_and_replaced_by_a_later_choice() {
        let mut settings = Settings::default();
        let framebuffer = HardwareArgs {
            display: Some(DisplayArg::Framebuffer),
            ..HardwareArgs::default()
        };
        settings.add(&framebuffer, &LinuxArgs::default(), Vec::new());
        let text = toml::to_string(&settings).unwrap();
        assert!(text.contains("display = \"framebuffer\""), "{text}");
        settings.add(&HardwareArgs::default(), &LinuxArgs::default(), Vec::new());
        assert_eq!(settings.display, Some(DisplayArg::Framebuffer), "kept");
        let serial = HardwareArgs {
            display: Some(DisplayArg::Serial),
            ..HardwareArgs::default()
        };
        settings.add(&serial, &LinuxArgs::default(), Vec::new());
        assert_eq!(settings.display, Some(DisplayArg::Serial));
        assert!(
            !toml::to_string(&Settings::default())
                .unwrap()
                .contains("display")
        );
    }

    #[test]
    fn a_machine_saved_before_input_could_be_chosen_keeps_its_keyboard_with_its_screen() {
        let old: Settings = toml::from_str("display = \"framebuffer\"\n").unwrap();
        assert_eq!(old.display, Some(DisplayArg::Framebuffer));
        assert_eq!(old.input, None, "follows the display");
        let old: Settings = toml::from_str("display = \"serial\"\n").unwrap();
        assert_eq!(old.display, Some(DisplayArg::Serial));
    }

    #[test]
    fn input_is_saved_and_replaced_by_a_later_choice() {
        let mut settings = Settings::default();
        let input = HardwareArgs {
            input: Some(true),
            ..HardwareArgs::default()
        };
        settings.add(&input, &LinuxArgs::default(), Vec::new());
        assert!(toml::to_string(&settings).unwrap().contains("input = true"));
        settings.add(&HardwareArgs::default(), &LinuxArgs::default(), Vec::new());
        assert_eq!(settings.input, Some(true), "kept");
        let off = HardwareArgs {
            input: Some(false),
            ..HardwareArgs::default()
        };
        settings.add(&off, &LinuxArgs::default(), Vec::new());
        assert_eq!(settings.input, Some(false));
    }

    #[test]
    fn an_empty_settings_file_means_every_default() {
        assert_eq!(toml::from_str::<Settings>("").unwrap(), Settings::default());
    }
}
