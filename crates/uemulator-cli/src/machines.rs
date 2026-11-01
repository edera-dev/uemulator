//! Named machines, kept under `<home>/machines/<name>/`:
//!
//! - `machine.toml`: the distro it installs and its `Settings`
//! - `nvram.kv`: the UEFI firmware's variables, for a machine whose firmware is `uefi`; created by
//!   its first boot and removed with the directory
//! - `artifacts/`: the distro's netboot kernel and initramfs, which `reinstall` throws away and
//!   `update-artifacts` replaces
//! - `disk-<n>.udisk`: the images `--create-disk` made (`disk-<n>.img`, raw, in machines made
//!   before the udisk format). Disks attached with `--disk` stay where they are; neither kind is
//!   ever deleted by the CLI.

use std::fs;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::args::{
    BootSource, ConsoleArg, DiskSpec, DiskStep, DisplayArg, Firmware, HardwareArgs, LinuxArgs,
    NetLink, NetModel, Nic, RemoveArgs, Settings, attach_only,
};
use crate::images;
use crate::quick::{Artifacts, Distro, artifacts_present, ensure_in, write_atomically};
use crate::run::DEFAULT_MEMORY_MIB;

const MACHINES_DIR: &str = "machines";
const MACHINE_FILE: &str = "machine.toml";
const ARTIFACTS_DIR: &str = "artifacts";
const NVRAM_FILE: &str = "nvram.kv";
/// Where `update-artifacts` downloads to, so a failed update leaves the old kernel and initramfs
/// together: the Alpine initramfs and its modloop must come from the release of the kernel.
const STAGING_DIR: &str = "artifacts.new";
const MAX_NAME_LEN: usize = 64;
const MIB_SHIFT: u32 = 20;
/// The disk a machine gets when none was asked for.
const DEFAULT_DISK_MIB: u64 = 8 * 1024;
const MAX_DISK_LETTERS: usize = 26;

/// Root of all CLI state: `$UEMULATOR_HOME`, else `~/.uemulator`. `HOME` is unix only, so named
/// machines need `UEMULATOR_HOME` on Windows.
pub fn home() -> Result<PathBuf, String> {
    home_from(|name| std::env::var(name).ok())
}

fn home_from(env: impl Fn(&str) -> Option<String>) -> Result<PathBuf, String> {
    if let Some(dir) = env("UEMULATOR_HOME") {
        return Ok(dir.into());
    }
    let home = env("HOME").ok_or_else(|| "HOME not set".to_string())?;
    Ok(Path::new(&home).join(".uemulator"))
}

/// The name becomes a directory, so it must not be able to name another one: no separators, no
/// `..`, nothing that starts like an option or a hidden file.
pub fn validate_name(name: &str) -> Result<(), String> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    let starts_well = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric());
    if name.len() > MAX_NAME_LEN || !starts_well || !name.chars().all(allowed) {
        return Err(format!(
            "invalid machine name {name:?}: use up to {MAX_NAME_LEN} letters, digits, '.', '_' or '-', starting with a letter or digit"
        ));
    }
    Ok(())
}

/// The on-disk form of `Machine`.
#[derive(Serialize, Deserialize)]
struct MachineFile {
    install: Distro,
    #[serde(default)]
    boot_from: BootSource,
    #[serde(default)]
    firmware: Firmware,
    #[serde(flatten)]
    settings: Settings,
}

pub struct Machine {
    pub name: String,
    pub dir: PathBuf,
    pub install: Distro,
    pub boot_from: BootSource,
    pub firmware: Firmware,
    pub settings: Settings,
}

impl Machine {
    fn config_path(&self) -> PathBuf {
        self.dir.join(MACHINE_FILE)
    }

    pub fn nvram_path(&self) -> PathBuf {
        self.dir.join(NVRAM_FILE)
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.dir.join(ARTIFACTS_DIR)
    }

    fn save(&self) -> Result<(), String> {
        let file = MachineFile {
            install: self.install,
            boot_from: self.boot_from,
            firmware: self.firmware,
            settings: self.settings.clone(),
        };
        let text =
            toml::to_string_pretty(&file).map_err(|e| format!("encode {}: {e}", self.name))?;
        let path = self.config_path();
        write_atomically(Cursor::new(text.into_bytes()), &path)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
}

pub fn load(home: &Path, name: &str) -> Result<Machine, String> {
    validate_name(name)?;
    let dir = home.join(MACHINES_DIR).join(name);
    let path = dir.join(MACHINE_FILE);
    let text = fs::read_to_string(&path).map_err(|e| match e.kind() {
        io::ErrorKind::NotFound => format!(
            "no machine named {name}: create it with `uemulator quick create {name} --install DISTRO`"
        ),
        _ => format!("read {}: {e}", path.display()),
    })?;
    let file: MachineFile =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Machine {
        name: name.to_string(),
        dir,
        install: file.install,
        boot_from: file.boot_from,
        firmware: file.firmware,
        settings: file.settings,
    })
}

/// Creates `path` (and its parent directories) as an empty udisk image of `mib` MiB, which takes
/// next to no space until the guest writes to it.
fn create_disk_image(path: &Path, mib: u64) -> Result<(), String> {
    let len = mib
        .checked_mul(1 << MIB_SHIFT)
        .ok_or_else(|| format!("a disk of {mib} MiB is too large"))?;
    images::create_udisk(path, Some(len), None, None).map(|_| ())
}

/// The images `--create-disk` makes are `disk-<n>.udisk`; machines made before the udisk format
/// have `disk-<n>.img`.
const CREATED_DISK_EXTENSIONS: [&str; 2] = [images::UDISK_EXTENSION, "img"];

/// The first `disk-<n>.udisk` whose number no image in `dir` has: a detached image keeps its
/// name, so a new disk never lands on top of it or reads as its sibling.
fn next_disk_path(dir: &Path) -> PathBuf {
    (0..)
        .find(|n| {
            CREATED_DISK_EXTENSIONS
                .iter()
                .all(|extension| !dir.join(format!("disk-{n}.{extension}")).exists())
        })
        .map(|n| dir.join(format!("disk-{n}.{}", images::UDISK_EXTENSION)))
        .expect("an unused disk name")
}

/// Turns the disk steps of a command line into disks to save: attached images get an absolute
/// path (the machine boots from any directory) and created ones are made in `dir`. Every attached
/// image is checked before any is created, so a typo does not leave an image behind.
fn persist_disks(dir: &Path, steps: &[DiskStep]) -> Result<Vec<DiskSpec>, String> {
    let attached: Vec<Option<PathBuf>> = steps
        .iter()
        .map(|step| match step {
            DiskStep::Attach(disk) => fs::canonicalize(&disk.path)
                .map(Some)
                .map_err(|e| format!("disk {}: {e}", disk.path.display())),
            DiskStep::Create { .. } => Ok(None),
        })
        .collect::<Result<_, _>>()?;
    steps
        .iter()
        .zip(attached)
        .map(|(step, resolved)| match (step, resolved) {
            (DiskStep::Attach(disk), Some(path)) => Ok(DiskSpec {
                path,
                read_only: disk.read_only,
            }),
            (DiskStep::Create { mib }, _) => {
                let path = next_disk_path(dir);
                create_disk_image(&path, *mib)?;
                eprintln!("created {mib} MiB disk {}", path.display());
                Ok(DiskSpec {
                    path,
                    read_only: false,
                })
            }
            (DiskStep::Attach(_), None) => unreachable!("every attached disk was resolved"),
        })
        .collect()
}

/// A new machine. Whatever the command line leaves out gets the `quick` defaults, saved so that
/// `edit` can change them: one 8 GiB disk, one NAT NIC (the installers fetch packages over the
/// network), one virtio console. Nothing is left behind if it fails.
pub fn create(
    home: &Path,
    name: &str,
    install: Distro,
    hardware: &HardwareArgs,
    linux: &LinuxArgs,
    boot_from: Option<BootSource>,
    firmware: Option<Firmware>,
) -> Result<Machine, String> {
    validate_name(name)?;
    let machines = home.join(MACHINES_DIR);
    fs::create_dir_all(&machines).map_err(|e| format!("create {}: {e}", machines.display()))?;
    let dir = machines.join(name);
    fs::create_dir(&dir).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => {
            format!("machine {name} already exists: use `uemulator quick edit {name}`")
        }
        _ => format!("create {}: {e}", dir.display()),
    })?;
    let built = (|| {
        let mut steps = hardware.disk_steps.clone();
        if steps.is_empty() {
            steps.push(DiskStep::Create {
                mib: DEFAULT_DISK_MIB,
            });
        }
        let mut settings = Settings::default();
        settings.add(hardware, linux, persist_disks(&dir, &steps)?);
        if settings.networks.is_empty() {
            settings.networks.push(Nic {
                model: NetModel::Virtio,
                link: Some(NetLink::User),
            });
        }
        if settings.consoles.is_empty() {
            settings.consoles.push(ConsoleArg::Virtio);
        }
        let machine = Machine {
            name: name.to_string(),
            dir: dir.clone(),
            install,
            boot_from: boot_from.unwrap_or_default(),
            firmware: firmware.unwrap_or_default(),
            settings,
        };
        machine.save()?;
        Ok(machine)
    })();
    if built.is_err() {
        let _ = fs::remove_dir_all(&dir);
    }
    built
}

pub struct Edited {
    pub machine: Machine,
    /// The disks `--remove-disk` detached; their images are untouched.
    pub detached: Vec<DiskSpec>,
}

/// Removals first (their indices refer to the saved list), then additions.
pub fn edit(
    home: &Path,
    name: &str,
    hardware: &HardwareArgs,
    linux: &LinuxArgs,
    remove: &RemoveArgs,
    boot_from: Option<BootSource>,
    firmware: Option<Firmware>,
) -> Result<Edited, String> {
    let mut machine = load(home, name)?;
    let detached = machine.settings.remove_devices(remove)?;
    let disks = persist_disks(&machine.dir, &hardware.disk_steps)?;
    machine.settings.add(hardware, linux, disks);
    machine.boot_from = boot_from.unwrap_or(machine.boot_from);
    machine.firmware = firmware.unwrap_or(machine.firmware);
    machine.save()?;
    Ok(Edited { machine, detached })
}

/// Everything `run` needs to boot a named machine.
pub struct Launch {
    /// What was chosen: never `Auto`.
    pub source: BootSource,
    /// Empty for a boot from the disk, whose boot loader brings its own kernel.
    pub kernel: Vec<u8>,
    pub initramfs: Option<Vec<u8>>,
    /// What starts the machine. The installer boots through it only when it is UEFI; the BIOS
    /// cannot boot a kernel, so a `bios` machine runs its installer directly.
    pub firmware: Firmware,
    /// The UEFI variable store of a boot through the UEFI firmware, installer or disk.
    pub nvram: Option<PathBuf>,
    /// The saved settings with this boot's overrides and the distro's defaults applied.
    pub settings: Settings,
}

/// The distro's contribution to an installer boot, applied last so it never reaches the saved
/// file: its kernel arguments come before the user's, and its memory default stands in for an
/// unset one.
pub fn with_distro_defaults(mut settings: Settings, distro: Distro) -> Settings {
    let distro_args = distro.cmdline();
    if !distro_args.is_empty() {
        settings.append = Some(match settings.append.take() {
            Some(user) => format!("{distro_args} {user}"),
            None => distro_args.to_string(),
        });
    }
    settings.memory.get_or_insert(distro.default_memory_mib());
    settings
}

/// A disk is blank, as far as `auto` can tell, when the start of it is all zeros: partition tables,
/// boot sectors and the first filesystem structures all live in the first 32 KiB.
const BLANK_CHECK_BYTES: u64 = 32 * 1024;

/// Read through the image's format: a udisk image's header is never zero, its guest bytes are.
fn disk_is_blank(path: &Path) -> Result<bool, String> {
    let start = images::read_start(path, BLANK_CHECK_BYTES)?;
    Ok(start.iter().all(|byte| *byte == 0))
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// A boot of a saved machine with this command line's overrides, which are not saved. Missing
/// installer artifacts (never fetched, or thrown away by `reinstall`) are fetched first.
pub fn prepare_boot(
    machine: &Machine,
    hardware: &HardwareArgs,
    linux: &LinuxArgs,
    remove: &RemoveArgs,
    boot_from: Option<BootSource>,
    fetch: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Launch, String> {
    let mut settings = machine.settings.clone();
    settings.remove_devices(remove)?;
    settings.add(hardware, linux, attach_only(&hardware.disk_steps)?);
    let installed = match (
        boot_from.unwrap_or(machine.boot_from),
        settings.disks.first(),
    ) {
        (BootSource::Installer, _) => false,
        (BootSource::Auto, first) => match first {
            Some(disk) => !disk_is_blank(&disk.path)?,
            None => false,
        },
        (BootSource::Disk, Some(_)) => true,
        (BootSource::Disk, None) => return Err("the machine has no disk to boot".into()),
    };
    if installed {
        if machine.firmware == Firmware::None {
            return Err(
                "firmware none cannot boot a disk; edit the machine to --firmware bios (a disk installed with none was installed without UEFI) or uefi"
                    .into(),
            );
        }
        // The boot loader on the disk decides the kernel and its arguments, and what memory the
        // installer needed is over.
        settings.memory.get_or_insert(DEFAULT_MEMORY_MIB);
        return Ok(Launch {
            source: BootSource::Disk,
            kernel: Vec::new(),
            initramfs: None,
            firmware: machine.firmware,
            nvram: (machine.firmware == Firmware::Uefi).then(|| machine.nvram_path()),
            settings,
        });
    }
    let artifacts = ensure_in(&machine.artifacts_dir(), machine.install, false, fetch)?;
    Ok(Launch {
        source: BootSource::Installer,
        kernel: read_file(&artifacts.kernel)?,
        initramfs: Some(read_file(&artifacts.initramfs)?),
        firmware: machine.firmware,
        // The installer writes the boot entries the installed system is found through.
        nvram: (machine.firmware == Firmware::Uefi).then(|| machine.nvram_path()),
        settings: with_distro_defaults(settings, machine.install),
    })
}

/// Fetches what is missing, for a machine that is not about to boot.
pub fn fetch_artifacts(
    machine: &Machine,
    fetch: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Artifacts, String> {
    ensure_in(&machine.artifacts_dir(), machine.install, false, fetch)
}

/// Throws the artifacts away (the disk and the settings stay), optionally switches the distro, and
/// fetches the new ones.
pub fn reinstall(
    home: &Path,
    name: &str,
    install: Option<Distro>,
    fetch: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Machine, String> {
    let mut machine = load(home, name)?;
    if let Some(distro) = install {
        machine.install = distro;
    }
    machine.boot_from = BootSource::Installer;
    machine.save()?;
    let artifacts = machine.artifacts_dir();
    match fs::remove_dir_all(&artifacts) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => {
            return Err(format!("remove {}: {e}", artifacts.display()));
        }
        _ => {}
    }
    fetch_artifacts(&machine, fetch)?;
    Ok(machine)
}

/// Downloads the distro's current kernel and initramfs next to the old ones and swaps them in
/// only once both have arrived.
pub fn update_artifacts(
    home: &Path,
    name: &str,
    fetch: impl FnMut(&str, &Path) -> Result<(), String>,
) -> Result<Machine, String> {
    let machine = load(home, name)?;
    let staging = machine.dir.join(STAGING_DIR);
    let _ = fs::remove_dir_all(&staging);
    let fetched = ensure_in(&staging, machine.install, true, fetch);
    if let Err(e) = fetched {
        let _ = fs::remove_dir_all(&staging);
        return Err(e);
    }
    let live = machine.artifacts_dir();
    let _ = fs::remove_dir_all(&live);
    fs::rename(&staging, &live).map_err(|e| format!("replace {}: {e}", live.display()))?;
    Ok(machine)
}

/// A machine's name and either the machine or why it could not be read.
pub type Listing = (String, Result<Machine, String>);

/// The saved machines by name, each either loaded or with the reason it could not be. A directory
/// without a `machine.toml` is not a machine and is skipped.
pub fn list(home: &Path) -> Result<Vec<Listing>, String> {
    let machines = home.join(MACHINES_DIR);
    let entries = match fs::read_dir(&machines) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("read {}: {e}", machines.display())),
    };
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("read {}: {e}", machines.display()))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if validate_name(&name).is_ok() && entry.path().join(MACHINE_FILE).is_file() {
            let machine = load(home, &name);
            found.push((name, machine));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found)
}

/// One line per machine, columns aligned.
pub fn format_list(entries: &[Listing]) -> String {
    let width = entries
        .iter()
        .map(|(name, _)| name.len())
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    for (name, machine) in entries {
        let line = match machine {
            Ok(machine) => {
                let settings = &machine.settings;
                let memory = settings
                    .memory
                    .unwrap_or(machine.install.default_memory_mib());
                format!(
                    "{:<8} {memory} MiB, {} disk(s), {} network(s)",
                    machine.install.name(),
                    settings.disks.len(),
                    settings.networks.len()
                )
            }
            Err(error) => format!("unreadable: {error}"),
        };
        out.push_str(&format!("{name:<width$}  {line}\n"));
    }
    out
}

/// `describe` plus where the machine lives and whether it can boot without downloading.
pub fn show(machine: &Machine) -> String {
    let artifacts = match artifacts_present(&machine.artifacts_dir(), machine.install) {
        true => "fetched",
        false => "missing (the next boot downloads them)",
    };
    let mut out = format!(
        "{}  directory {}\n  artifacts {artifacts}\n",
        describe(machine),
        machine.dir.display()
    );
    if machine.firmware == Firmware::Uefi {
        let nvram = machine.nvram_path();
        let state = match nvram.is_file() {
            true => "present",
            false => "not created yet (the first boot makes it)",
        };
        out.push_str(&format!("  nvram     {} {state}\n", nvram.display()));
    }
    out
}

/// What `delete` removes: the machine's directory, and with it the disk images made in it.
#[derive(Debug)]
pub struct Removal {
    pub dir: PathBuf,
    pub created_disks: Vec<PathBuf>,
}

/// Looks at what deleting `name` would remove. Only a directory holding a `machine.toml` counts,
/// so a name can never point this at some other directory; disks attached with `--disk` live
/// elsewhere and are never touched.
pub fn removal(home: &Path, name: &str) -> Result<Removal, String> {
    validate_name(name)?;
    let dir = home.join(MACHINES_DIR).join(name);
    if !dir.join(MACHINE_FILE).is_file() {
        return Err(format!("no machine named {name}"));
    }
    let mut created_disks: Vec<PathBuf> = fs::read_dir(&dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            file.starts_with("disk-")
                && CREATED_DISK_EXTENSIONS
                    .iter()
                    .any(|extension| file.ends_with(&format!(".{extension}")))
        })
        .collect();
    created_disks.sort();
    Ok(Removal { dir, created_disks })
}

pub fn delete(removal: &Removal) -> Result<(), String> {
    fs::remove_dir_all(&removal.dir).map_err(|e| format!("remove {}: {e}", removal.dir.display()))
}

/// `vda`, `vdb`, ... for the disk at `index`: the order the guest sees them in.
fn guest_disk_name(index: usize) -> String {
    match index < MAX_DISK_LETTERS {
        true => format!("vd{}", char::from(b'a' + index as u8)),
        false => format!("disk {index}"),
    }
}

/// The machine as a numbered device list: the numbers are what `--remove-*` takes.
pub fn describe(machine: &Machine) -> String {
    let settings = &machine.settings;
    let mut out = format!(
        "machine {} (installs {})\n",
        machine.name,
        machine.install.name()
    );
    let memory = settings
        .memory
        .unwrap_or(machine.install.default_memory_mib());
    out.push_str(&format!("  memory    {memory} MiB\n"));
    if let Some(cpus) = settings.cpus {
        out.push_str(&format!("  cpus      {cpus}\n"));
    }
    if let Some(smp) = settings.smp {
        out.push_str(&format!("  smp       {smp:?}\n").to_lowercase());
    }
    let boot = match machine.boot_from {
        BootSource::Auto => "auto (the disk once an install is on it, else the installer)",
        BootSource::Installer => "installer",
        BootSource::Disk => "disk",
    };
    out.push_str(&format!("  boots     {boot}\n"));
    out.push_str(&format!("  firmware  {}\n", machine.firmware.name()));
    let display = settings.display.unwrap_or_default();
    match display {
        DisplayArg::Serial => {}
        DisplayArg::Framebuffer => out.push_str("  display   framebuffer (the console)\n"),
        DisplayArg::Both => out.push_str("  display   framebuffer (the terminal is the console)\n"),
    }
    let input = settings.input.unwrap_or(display.has_screen());
    if input != display.has_screen() {
        let state = if input { "keyboard and tablet" } else { "none" };
        out.push_str(&format!("  input     {state}\n"));
    }
    for (index, disk) in settings.disks.iter().enumerate() {
        let mode = if disk.read_only { "ro" } else { "rw" };
        out.push_str(&format!(
            "  disk {index}    {} {} ({mode})\n",
            guest_disk_name(index),
            disk.path.display()
        ));
    }
    for (index, nic) in settings.networks.iter().enumerate() {
        let link = match &nic.link {
            None => "no link".to_string(),
            Some(NetLink::User) => "link user".to_string(),
            Some(NetLink::Loop) => "link loop".to_string(),
            Some(NetLink::Udp(spec)) => format!("link udp {spec}"),
        };
        out.push_str(&format!("  network {index} virtio, {link}\n"));
    }
    for (index, console) in settings.consoles.iter().enumerate() {
        let kind = match console {
            ConsoleArg::Serial => "serial",
            ConsoleArg::Virtio => "virtio",
        };
        out.push_str(&format!("  console {index} {kind}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::test_support::{EnvGuard, TempDir};
    use crate::{Cmd, QuickCmd, parse_cli};

    const MIB: u64 = 1 << MIB_SHIFT;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    /// The hardware and Linux arguments a `quick create` command line parses to.
    fn create_args(extra: &[&str]) -> (HardwareArgs, LinuxArgs) {
        let mut argv = vec!["uemulator", "quick", "create", "m", "--install", "alpine"];
        argv.extend_from_slice(extra);
        match parse_cli(argv).unwrap().cmd {
            Cmd::Quick {
                cmd: QuickCmd::Create {
                    hardware, linux, ..
                },
            } => (hardware, linux),
            _ => unreachable!(),
        }
    }

    fn edit_args(extra: &[&str]) -> (HardwareArgs, LinuxArgs, RemoveArgs) {
        let mut argv = vec!["uemulator", "quick", "edit", "m"];
        argv.extend_from_slice(extra);
        match parse_cli(argv).unwrap().cmd {
            Cmd::Quick {
                cmd:
                    QuickCmd::Edit {
                        hardware,
                        linux,
                        remove,
                        ..
                    },
            } => (hardware, linux, remove),
            _ => unreachable!(),
        }
    }

    fn boot_args(extra: &[&str]) -> (HardwareArgs, LinuxArgs, RemoveArgs) {
        let mut argv = vec!["uemulator", "quick", "boot", "m"];
        argv.extend_from_slice(extra);
        match parse_cli(argv).unwrap().cmd {
            Cmd::Quick {
                cmd:
                    QuickCmd::Boot {
                        hardware,
                        linux,
                        remove,
                        ..
                    },
            } => (hardware, linux, remove),
            _ => unreachable!(),
        }
    }

    fn create_machine(home: &Path, extra: &[&str]) -> Machine {
        let (hardware, linux) = create_args(extra);
        create(home, "m", Distro::Alpine, &hardware, &linux, None, None).unwrap()
    }

    /// Records every requested URL and writes the URL itself as the file contents.
    struct FakeFetcher {
        urls: RefCell<Vec<String>>,
        fail_on: Option<&'static str>,
    }

    impl FakeFetcher {
        fn new() -> Self {
            FakeFetcher {
                urls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }

        fn fetch(&self, url: &str, dest: &Path) -> Result<(), String> {
            self.urls.borrow_mut().push(url.to_string());
            if self.fail_on.is_some_and(|needle| url.contains(needle)) {
                return Err(format!("download of {url} failed: injected"));
            }
            fs::write(dest, url).map_err(|e| e.to_string())
        }

        fn count(&self) -> usize {
            self.urls.borrow().len()
        }
    }

    #[test]
    fn home_prefers_uemulator_home() {
        let env = [("UEMULATOR_HOME", "/data/emu"), ("HOME", "/home/me")];
        assert_eq!(home_from(env_of(&env)).unwrap(), PathBuf::from("/data/emu"));
    }

    #[test]
    fn home_falls_back_to_dot_uemulator_under_home() {
        let env = [("HOME", "/home/me")];
        assert_eq!(
            home_from(env_of(&env)).unwrap(),
            PathBuf::from("/home/me/.uemulator")
        );
    }

    #[test]
    fn home_without_any_variable_is_an_error() {
        assert_eq!(home_from(env_of(&[])).unwrap_err(), "HOME not set");
    }

    #[test]
    fn uemulator_home_works_without_home() {
        let env = [("UEMULATOR_HOME", "C:\\emu")];
        assert_eq!(home_from(env_of(&env)).unwrap(), PathBuf::from("C:\\emu"));
    }

    #[test]
    fn real_home_reads_the_process_environment() {
        let _env = EnvGuard::set(&[("UEMULATOR_HOME", Some("/tmp/uemulator-env-home"))]);
        assert_eq!(home().unwrap(), PathBuf::from("/tmp/uemulator-env-home"));
    }

    #[test]
    fn machine_names_cannot_name_another_directory() {
        for good in ["test", "a", "alpine-4.0_x", "7up", "A.b"] {
            assert!(validate_name(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            ".",
            "..",
            ".hidden",
            "-rf",
            "a/b",
            "../x",
            "a\\b",
            "sp ace",
            "ünï",
            &"x".repeat(MAX_NAME_LEN + 1),
        ] {
            assert!(validate_name(bad).is_err(), "{bad:?}");
        }
        assert!(validate_name(&"x".repeat(MAX_NAME_LEN)).is_ok());
    }

    #[test]
    fn create_makes_the_directory_the_config_and_the_default_devices() {
        let home = TempDir::new("machines-create");
        let machine = create_machine(home.path(), &[]);
        let dir = home.path().join("machines/m");
        assert_eq!(machine.dir, dir);
        assert!(dir.join(MACHINE_FILE).is_file());
        assert_eq!(disk_size(&dir.join("disk-0.udisk")), DEFAULT_DISK_MIB * MIB);
        let settings = load(home.path(), "m").unwrap().settings;
        assert_eq!(
            settings.disks,
            vec![DiskSpec {
                path: dir.join("disk-0.udisk"),
                read_only: false
            }]
        );
        assert_eq!(
            settings.networks,
            vec![Nic {
                model: NetModel::Virtio,
                link: Some(NetLink::User)
            }]
        );
        assert_eq!(settings.consoles, vec![ConsoleArg::Virtio]);
        assert_eq!(settings.memory, None);
        assert_eq!(load(home.path(), "m").unwrap().install, Distro::Alpine);
    }

    #[test]
    fn create_keeps_what_the_command_line_chose_instead_of_the_defaults() {
        let home = TempDir::new("machines-create-chosen");
        let machine = create_machine(
            home.path(),
            &[
                "--memory",
                "512",
                "--create-disk",
                "16",
                "--network",
                "virtio",
                "--network-link-loop",
                "--console",
                "serial",
                "--append",
                "quiet",
                "--no-reboot",
                "--disable-jit",
            ],
        );
        let settings = load(home.path(), "m").unwrap().settings;
        assert_eq!(settings, machine.settings);
        assert_eq!(settings.memory, Some(512));
        assert_eq!(settings.disks.len(), 1);
        assert_eq!(disk_size(&settings.disks[0].path), 16 * MIB);
        assert_eq!(
            settings.networks,
            vec![Nic {
                model: NetModel::Virtio,
                link: Some(NetLink::Loop)
            }]
        );
        assert_eq!(settings.consoles, vec![ConsoleArg::Serial]);
        assert_eq!(settings.append.as_deref(), Some("quiet"));
        assert!(settings.no_reboot && settings.disable_jit);
    }

    #[test]
    fn create_refuses_an_existing_name_and_leaves_it_alone() {
        let home = TempDir::new("machines-create-twice");
        create_machine(home.path(), &[]);
        let (hardware, linux) = create_args(&["--memory", "99"]);
        let error = create(
            home.path(),
            "m",
            Distro::Debian,
            &hardware,
            &linux,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(error.contains("already exists"), "{error}");
        let kept = load(home.path(), "m").unwrap();
        assert_eq!(kept.install, Distro::Alpine);
        assert_eq!(kept.settings.memory, None);
    }

    #[test]
    fn a_failed_create_leaves_no_directory_behind() {
        let home = TempDir::new("machines-create-fail");
        let missing = home.path().join("missing.img");
        let (hardware, linux) =
            create_args(&["--create-disk", "1", "--disk", &missing.to_string_lossy()]);
        let error = create(
            home.path(),
            "m",
            Distro::Alpine,
            &hardware,
            &linux,
            None,
            None,
        )
        .err()
        .unwrap();
        assert!(error.starts_with("disk "), "{error}");
        assert!(!home.path().join("machines/m").exists());
    }

    #[test]
    fn a_missing_disk_is_reported_before_any_image_is_created() {
        let dir = TempDir::new("machines-disk-order");
        let missing = dir.path().join("missing.img");
        let steps = [
            DiskStep::Create { mib: 1 },
            DiskStep::Attach(DiskSpec {
                path: missing,
                read_only: false,
            }),
        ];
        assert!(persist_disks(dir.path(), &steps).is_err());
        assert!(!dir.path().join("disk-0.udisk").exists());
    }

    #[test]
    fn disks_keep_command_line_order_and_attached_paths_become_absolute() {
        let home = TempDir::new("machines-disk-interleave");
        let existing = home.path().join("existing.img");
        fs::write(&existing, b"x").unwrap();
        let other = home.path().join("other.img");
        fs::write(&other, b"y").unwrap();
        let existing_arg = existing.to_string_lossy().into_owned();
        let other_arg = format!("{},ro", other.to_string_lossy());
        let machine = create_machine(
            home.path(),
            &[
                "--disk",
                &existing_arg,
                "--create-disk",
                "2",
                "--disk",
                &other_arg,
                "--create-disk",
                "3",
            ],
        );
        let disks = &machine.settings.disks;
        let dir = home.path().join("machines/m");
        assert_eq!(disks.len(), 4);
        assert_eq!(disks[0].path, fs::canonicalize(&existing).unwrap());
        assert_eq!(disks[1].path, dir.join("disk-0.udisk"));
        assert_eq!(disks[2].path, fs::canonicalize(&other).unwrap());
        assert!(disks[2].read_only);
        assert_eq!(disks[3].path, dir.join("disk-1.udisk"));
        assert_eq!(disk_size(&disks[1].path), 2 * MIB);
        assert_eq!(disk_size(&disks[3].path), 3 * MIB);
    }

    #[test]
    fn a_created_disk_never_reuses_the_name_of_a_detached_image() {
        let home = TempDir::new("machines-disk-names");
        create_machine(home.path(), &[]);
        let (hardware, linux, remove) = edit_args(&["--remove-disk", "0", "--create-disk", "1"]);
        let edited = edit(home.path(), "m", &hardware, &linux, &remove, None, None).unwrap();
        let dir = home.path().join("machines/m");
        assert_eq!(edited.detached.len(), 1);
        assert_eq!(edited.detached[0].path, dir.join("disk-0.udisk"));
        assert!(
            dir.join("disk-0.udisk").exists(),
            "detaching must not delete the image"
        );
        assert_eq!(edited.machine.settings.disks.len(), 1);
        assert_eq!(
            edited.machine.settings.disks[0].path,
            dir.join("disk-1.udisk")
        );
    }

    #[test]
    fn edit_removes_before_adding_and_saves() {
        let home = TempDir::new("machines-edit");
        create_machine(home.path(), &["--create-disk", "1", "--create-disk", "1"]);
        let (hardware, linux, remove) = edit_args(&[
            "--memory",
            "256",
            "--remove-disk",
            "1",
            "--remove-network",
            "0",
            "--network",
            "virtio",
            "--network-link-udp",
            "a,b",
            "--create-disk",
            "4",
        ]);
        let edited = edit(home.path(), "m", &hardware, &linux, &remove, None, None).unwrap();
        let reloaded = load(home.path(), "m").unwrap().settings;
        assert_eq!(reloaded, edited.machine.settings);
        assert_eq!(reloaded.memory, Some(256));
        let dir = home.path().join("machines/m");
        assert_eq!(
            reloaded
                .disks
                .iter()
                .map(|d| d.path.clone())
                .collect::<Vec<_>>(),
            vec![dir.join("disk-0.udisk"), dir.join("disk-2.udisk")]
        );
        assert_eq!(
            reloaded.networks,
            vec![Nic {
                model: NetModel::Virtio,
                link: Some(NetLink::Udp("a,b".into()))
            }]
        );
    }

    #[test]
    fn edit_with_a_bad_index_changes_nothing_and_makes_nothing() {
        let home = TempDir::new("machines-edit-bad");
        create_machine(home.path(), &[]);
        let before = fs::read_to_string(home.path().join("machines/m/machine.toml")).unwrap();
        let (hardware, linux, remove) =
            edit_args(&["--remove-disk", "5", "--create-disk", "1", "--memory", "99"]);
        let error = edit(home.path(), "m", &hardware, &linux, &remove, None, None)
            .err()
            .unwrap();
        assert_eq!(error, "--remove-disk 5: there is 1 disk, numbered 0");
        assert_eq!(
            fs::read_to_string(home.path().join("machines/m/machine.toml")).unwrap(),
            before
        );
        assert!(!home.path().join("machines/m/disk-1.udisk").exists());
    }

    #[test]
    fn edit_can_turn_a_saved_flag_back_off_and_clear_a_string() {
        let home = TempDir::new("machines-edit-off");
        create_machine(
            home.path(),
            &["--disable-jit", "--no-reboot", "--cmdline", "root=/dev/vda"],
        );
        let (hardware, linux, remove) =
            edit_args(&["--disable-jit=false", "--no-reboot=false", "--cmdline", ""]);
        let settings = edit(home.path(), "m", &hardware, &linux, &remove, None, None)
            .unwrap()
            .machine
            .settings;
        assert!(!settings.disable_jit && !settings.no_reboot);
        assert_eq!(settings.cmdline, None);
    }

    #[test]
    fn edit_of_a_missing_machine_says_how_to_create_it() {
        let home = TempDir::new("machines-edit-missing");
        let (hardware, linux, remove) = edit_args(&[]);
        let error = edit(home.path(), "m", &hardware, &linux, &remove, None, None)
            .err()
            .unwrap();
        assert!(error.contains("no machine named m"), "{error}");
        assert!(error.contains("quick create m --install"), "{error}");
    }

    #[test]
    fn a_corrupt_machine_file_names_the_file() {
        let home = TempDir::new("machines-corrupt");
        create_machine(home.path(), &[]);
        let path = home.path().join("machines/m/machine.toml");
        fs::write(&path, "install = 5").unwrap();
        let error = load(home.path(), "m").err().unwrap();
        assert!(
            error.contains(&path.to_string_lossy().into_owned()),
            "{error}"
        );
    }

    #[test]
    fn boot_overrides_are_applied_to_the_launch_and_not_saved() {
        let home = TempDir::new("machines-boot-override");
        let machine = create_machine(home.path(), &["--memory", "512"]);
        let extra = home.path().join("installer.iso");
        fs::write(&extra, b"iso").unwrap();
        let extra_arg = format!("{},ro", extra.to_string_lossy());
        let (hardware, linux, remove) = boot_args(&[
            "--memory",
            "1024",
            "--disk",
            &extra_arg,
            "--remove-console",
            "0",
            "--console",
            "serial",
        ]);
        let fetcher = FakeFetcher::new();
        let launch = prepare_boot(&machine, &hardware, &linux, &remove, None, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(launch.settings.memory, Some(1024));
        assert_eq!(launch.settings.disks.len(), 2);
        assert!(launch.settings.disks[1].read_only);
        assert_eq!(launch.settings.consoles, vec![ConsoleArg::Serial]);
        let saved = load(home.path(), "m").unwrap().settings;
        assert_eq!(saved.memory, Some(512));
        assert_eq!(saved.disks.len(), 1);
        assert_eq!(saved.consoles, vec![ConsoleArg::Virtio]);
    }

    #[test]
    fn booting_fetches_missing_artifacts_once_and_uses_the_distro_defaults() {
        let home = TempDir::new("machines-boot-fetch");
        let machine = create_machine(home.path(), &["--append", "quiet"]);
        let (hardware, linux, remove) = boot_args(&[]);
        let fetcher = FakeFetcher::new();
        let launch = prepare_boot(&machine, &hardware, &linux, &remove, None, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.count(), 2);
        assert_eq!(launch.source, BootSource::Installer);
        assert_eq!(launch.kernel, Distro::Alpine.source_urls().0.as_bytes());
        assert!(
            home.path()
                .join("machines/m/artifacts/vmlinuz-virt")
                .is_file()
        );
        assert_eq!(
            launch.settings.append,
            Some(format!("{} quiet", Distro::Alpine.cmdline()))
        );
        assert_eq!(
            launch.settings.memory,
            Some(Distro::Alpine.default_memory_mib())
        );
        prepare_boot(&machine, &hardware, &linux, &remove, None, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(fetcher.count(), 2, "cached artifacts are not fetched again");
        assert_eq!(machine.settings.append.as_deref(), Some("quiet"));
    }

    /// Puts something in the machine's first disk, the way an installer does: a partition table
    /// entry in the first sector.
    /// Writes a partition table entry, through the image's format, as an installer would.
    fn finish_install(machine: &Machine) {
        let path = machine.settings.disks[0].path.clone();
        let mut disk = images::open_disk(&images::Location::File(path), false).unwrap();
        disk.write_at(446, &[0x80, 0x01, 0x01, 0x00, 0x83]).unwrap();
        disk.flush().unwrap();
    }

    /// The size the guest sees, whatever the image's format.
    fn disk_size(path: &Path) -> u64 {
        images::open_disk(&images::Location::File(path.to_path_buf()), true)
            .unwrap()
            .len()
    }

    fn launch(machine: &Machine, boot_from: Option<BootSource>) -> Result<Launch, String> {
        let (hardware, linux, remove) = boot_args(&[]);
        let fetcher = FakeFetcher::new();
        prepare_boot(machine, &hardware, &linux, &remove, boot_from, |u, d| {
            fetcher.fetch(u, d)
        })
    }

    #[test]
    fn auto_boots_the_installer_until_something_is_on_the_disk_and_then_the_disk() {
        let home = TempDir::new("machines-auto");
        let machine = create_machine(home.path(), &[]);
        assert_eq!(machine.boot_from, BootSource::Auto);
        let installer = launch(&machine, None).unwrap();
        assert_eq!(installer.source, BootSource::Installer);
        assert!(!installer.kernel.is_empty());

        finish_install(&machine);
        let installed = launch(&machine, None).unwrap();
        assert_eq!(installed.source, BootSource::Disk);
        assert!(installed.kernel.is_empty() && installed.initramfs.is_none());
    }

    #[test]
    fn a_disk_is_blank_when_the_first_32_kib_are_zeros() {
        let dir = TempDir::new("machines-blank");
        let path = dir.path().join("d.img");
        let with_byte_at = |offset: usize| {
            let mut bytes = vec![0u8; 3 * BLANK_CHECK_BYTES as usize];
            bytes[offset] = 1;
            fs::write(&path, bytes).unwrap();
            disk_is_blank(&path).unwrap()
        };
        assert!(!with_byte_at(0));
        assert!(!with_byte_at(BLANK_CHECK_BYTES as usize - 1));
        assert!(
            with_byte_at(BLANK_CHECK_BYTES as usize),
            "past the first 32 KiB is not looked at"
        );
        fs::write(&path, vec![0u8; 100]).unwrap();
        assert!(
            disk_is_blank(&path).unwrap(),
            "a short file of zeros is blank"
        );
        fs::write(&path, b"").unwrap();
        assert!(disk_is_blank(&path).unwrap());
        fs::write(&path, [0, 0, 7]).unwrap();
        assert!(!disk_is_blank(&path).unwrap());
        let error = disk_is_blank(&dir.path().join("missing.img")).unwrap_err();
        assert!(error.starts_with("open "), "{error}");
    }

    #[test]
    fn a_new_udisk_image_is_blank_until_written() {
        let dir = TempDir::new("machines-blank-udisk");
        let path = dir.path().join("d.udisk");
        create_disk_image(&path, 1).unwrap();
        assert!(
            disk_is_blank(&path).unwrap(),
            "the header is not the guest's data"
        );
        let mut disk = images::open_disk(&images::Location::File(path.clone()), false).unwrap();
        disk.write_at(BLANK_CHECK_BYTES - 1, &[1]).unwrap();
        disk.flush().unwrap();
        drop(disk);
        assert!(!disk_is_blank(&path).unwrap());
    }

    #[test]
    fn a_disk_boot_leaves_the_installer_arguments_out_and_uses_the_plain_memory_default() {
        let home = TempDir::new("machines-disk-defaults");
        let machine = create_machine(home.path(), &["--append", "debug"]);
        finish_install(&machine);
        let settings = launch(&machine, None).unwrap().settings;
        assert_eq!(
            settings.append.as_deref(),
            Some("debug"),
            "no modloop, no ip=dhcp"
        );
        assert_eq!(settings.memory, Some(DEFAULT_MEMORY_MIB));

        let fedora = create(
            home.path(),
            "f",
            Distro::Fedora,
            &HardwareArgs::default(),
            &LinuxArgs::default(),
            None,
            None,
        )
        .unwrap();
        finish_install(&fedora);
        let settings = launch(&fedora, None).unwrap().settings;
        assert_eq!(
            settings.memory,
            Some(DEFAULT_MEMORY_MIB),
            "not the installer's 3072"
        );
        let installer = launch(&fedora, Some(BootSource::Installer)).unwrap();
        assert_eq!(
            installer.settings.memory,
            Some(Distro::Fedora.default_memory_mib())
        );
    }

    #[test]
    fn auto_looks_at_the_first_disk_after_this_boots_overrides() {
        let home = TempDir::new("machines-first-disk");
        let machine = create_machine(home.path(), &[]);
        finish_install(&machine);
        let blank = home.path().join("blank.img");
        fs::write(&blank, vec![0u8; 4096]).unwrap();
        let blank_arg = blank.to_string_lossy().into_owned();
        // Detach the installed disk and attach a blank one: nothing is installed any more.
        let (hardware, linux, remove) = boot_args(&["--remove-disk", "0", "--disk", &blank_arg]);
        let fetcher = FakeFetcher::new();
        let launch = prepare_boot(&machine, &hardware, &linux, &remove, None, |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(launch.source, BootSource::Installer);
    }

    #[test]
    fn installer_ignores_an_install_and_disk_needs_a_disk() {
        let home = TempDir::new("machines-force");
        let machine = create_machine(home.path(), &[]);
        assert_eq!(
            launch(&machine, Some(BootSource::Disk)).unwrap().source,
            BootSource::Disk,
            "the BIOS reports an empty disk itself"
        );
        finish_install(&machine);
        assert_eq!(
            launch(&machine, Some(BootSource::Installer))
                .unwrap()
                .source,
            BootSource::Installer
        );

        let (hardware, linux, _) = boot_args(&[]);
        let remove_all = RemoveArgs {
            remove_disk: vec![0],
            ..RemoveArgs::default()
        };
        let fetcher = FakeFetcher::new();
        let error = prepare_boot(
            &machine,
            &hardware,
            &linux,
            &remove_all,
            Some(BootSource::Disk),
            |u, d| fetcher.fetch(u, d),
        )
        .err()
        .unwrap();
        assert_eq!(error, "the machine has no disk to boot");
    }

    #[test]
    fn the_saved_boot_source_is_used_and_an_override_is_not_saved() {
        let home = TempDir::new("machines-saved-source");
        let machine = create_machine(home.path(), &[]);
        finish_install(&machine);
        let (hardware, linux, remove) = edit_args(&[]);
        edit(
            home.path(),
            "m",
            &hardware,
            &linux,
            &remove,
            Some(BootSource::Installer),
            None,
        )
        .unwrap();
        let saved = load(home.path(), "m").unwrap();
        assert_eq!(saved.boot_from, BootSource::Installer);
        assert_eq!(launch(&saved, None).unwrap().source, BootSource::Installer);
        assert_eq!(
            launch(&saved, Some(BootSource::Disk)).unwrap().source,
            BootSource::Disk
        );
        assert_eq!(
            load(home.path(), "m").unwrap().boot_from,
            BootSource::Installer
        );
        edit(home.path(), "m", &hardware, &linux, &remove, None, None).unwrap();
        assert_eq!(
            load(home.path(), "m").unwrap().boot_from,
            BootSource::Installer
        );
    }

    #[test]
    fn a_machine_file_from_before_boot_from_existed_means_auto() {
        let home = TempDir::new("machines-old-file");
        create_machine(home.path(), &[]);
        let path = home.path().join("machines/m/machine.toml");
        let text = fs::read_to_string(&path).unwrap();
        let without: String = text
            .lines()
            .filter(|l| !l.starts_with("boot_from"))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(text.contains("boot_from = \"auto\""), "{text}");
        fs::write(&path, without).unwrap();
        assert_eq!(load(home.path(), "m").unwrap().boot_from, BootSource::Auto);
    }

    #[test]
    fn reinstall_goes_back_to_the_installer_and_says_so() {
        let home = TempDir::new("machines-reinstall-source");
        let machine = create_machine(home.path(), &[]);
        finish_install(&machine);
        let fetcher = FakeFetcher::new();
        let reinstalled = reinstall(home.path(), "m", None, |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(reinstalled.boot_from, BootSource::Installer);
        assert_eq!(
            load(home.path(), "m").unwrap().boot_from,
            BootSource::Installer
        );
        assert_eq!(
            launch(&reinstalled, None).unwrap().source,
            BootSource::Installer
        );
    }

    #[test]
    fn create_can_choose_the_boot_source() {
        let home = TempDir::new("machines-create-source");
        let (hardware, linux) = create_args(&[]);
        let machine = create(
            home.path(),
            "m",
            Distro::Alpine,
            &hardware,
            &linux,
            Some(BootSource::Disk),
            None,
        )
        .unwrap();
        assert_eq!(machine.boot_from, BootSource::Disk);
        assert!(describe(&machine).contains("boots     disk"));
    }

    #[test]
    fn a_boot_override_cannot_create_a_disk() {
        // The parser does not offer `--create-disk` on `quick boot`; this is the library-level
        // guard behind it.
        let steps = [DiskStep::Create { mib: 1 }];
        assert!(attach_only(&steps).is_err());
    }

    #[test]
    fn distro_defaults_put_the_distro_arguments_before_the_users() {
        let none = with_distro_defaults(Settings::default(), Distro::Debian);
        assert_eq!(none.append, None);
        assert_eq!(none.memory, Some(Distro::Debian.default_memory_mib()));
        let user = Settings {
            append: Some("console=ttyS0".into()),
            memory: Some(1024),
            ..Settings::default()
        };
        let fedora = with_distro_defaults(user, Distro::Fedora);
        assert_eq!(
            fedora.append,
            Some(format!("{} console=ttyS0", Distro::Fedora.cmdline()))
        );
        assert_eq!(fedora.memory, Some(1024));
        assert_eq!(
            with_distro_defaults(Settings::default(), Distro::Alpine)
                .append
                .as_deref(),
            Some(Distro::Alpine.cmdline())
        );
    }

    #[test]
    fn reinstall_replaces_the_artifacts_and_keeps_the_disk_and_settings() {
        let home = TempDir::new("machines-reinstall");
        let machine = create_machine(home.path(), &["--memory", "700"]);
        let fetcher = FakeFetcher::new();
        fetch_artifacts(&machine, |u, d| fetcher.fetch(u, d)).unwrap();
        let disk = machine.settings.disks[0].path.clone();
        let mut image = images::open_disk(&images::Location::File(disk.clone()), false).unwrap();
        image.write_at(0, b"guest data").unwrap();
        image.flush().unwrap();
        drop(image);
        let disk_file = fs::read(&disk).unwrap();

        let reinstalled = reinstall(home.path(), "m", Some(Distro::Debian), |u, d| {
            fetcher.fetch(u, d)
        })
        .unwrap();
        assert_eq!(reinstalled.install, Distro::Debian);
        assert_eq!(load(home.path(), "m").unwrap().install, Distro::Debian);
        assert_eq!(load(home.path(), "m").unwrap().settings.memory, Some(700));
        assert_eq!(fs::read(&disk).unwrap(), disk_file);
        let artifacts = home.path().join("machines/m/artifacts");
        assert!(artifacts.join("linux").is_file());
        assert!(artifacts.join("initrd.gz").is_file());
        assert!(
            !artifacts.join("vmlinuz-virt").exists(),
            "the old kernel is jettisoned"
        );
        assert_eq!(fetcher.count(), 4);
    }

    #[test]
    fn reinstall_without_a_distro_refetches_the_same_one() {
        let home = TempDir::new("machines-reinstall-same");
        let machine = create_machine(home.path(), &[]);
        let fetcher = FakeFetcher::new();
        fetch_artifacts(&machine, |u, d| fetcher.fetch(u, d)).unwrap();
        let reinstalled = reinstall(home.path(), "m", None, |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(reinstalled.install, Distro::Alpine);
        assert_eq!(fetcher.count(), 4);
    }

    #[test]
    fn reinstall_works_before_anything_was_fetched() {
        let home = TempDir::new("machines-reinstall-fresh");
        create_machine(home.path(), &[]);
        let fetcher = FakeFetcher::new();
        reinstall(home.path(), "m", None, |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(fetcher.count(), 2);
    }

    #[test]
    fn update_artifacts_refetches_everything_and_swaps_it_in() {
        let home = TempDir::new("machines-update");
        let machine = create_machine(home.path(), &[]);
        let fetcher = FakeFetcher::new();
        fetch_artifacts(&machine, |u, d| fetcher.fetch(u, d)).unwrap();
        let kernel = home.path().join("machines/m/artifacts/vmlinuz-virt");
        fs::write(&kernel, b"stale").unwrap();
        update_artifacts(home.path(), "m", |u, d| fetcher.fetch(u, d)).unwrap();
        assert_eq!(fetcher.count(), 4);
        assert_eq!(
            fs::read_to_string(&kernel).unwrap(),
            Distro::Alpine.source_urls().0
        );
        assert!(!home.path().join("machines/m").join(STAGING_DIR).exists());
    }

    #[test]
    fn a_failed_update_keeps_the_old_pair_together() {
        let home = TempDir::new("machines-update-fail");
        let machine = create_machine(home.path(), &[]);
        let fetcher = FakeFetcher::new();
        fetch_artifacts(&machine, |u, d| fetcher.fetch(u, d)).unwrap();
        let mut failing = FakeFetcher::new();
        failing.fail_on = Some("initramfs-virt");
        let error = update_artifacts(home.path(), "m", |u, d| failing.fetch(u, d))
            .err()
            .unwrap();
        assert!(error.contains("injected"), "{error}");
        let artifacts = home.path().join("machines/m/artifacts");
        let (kernel_url, initramfs_url) = Distro::Alpine.source_urls();
        assert_eq!(
            fs::read_to_string(artifacts.join("vmlinuz-virt")).unwrap(),
            kernel_url
        );
        assert_eq!(
            fs::read_to_string(artifacts.join("initramfs-virt")).unwrap(),
            initramfs_url
        );
        assert!(!home.path().join("machines/m").join(STAGING_DIR).exists());
    }

    #[test]
    fn a_new_disk_has_the_requested_size_without_taking_the_space() {
        let dir = TempDir::new("machines-sparse");
        let path = dir.path().join("nested").join("deeper").join("disk.udisk");
        create_disk_image(&path, 2048).unwrap();
        assert_eq!(disk_size(&path), 2048 * MIB);
        assert!(
            fs::metadata(&path).unwrap().len() < MIB,
            "header and map only"
        );
    }

    #[test]
    fn sparse_disk_creation_reports_the_path_on_failure_and_rejects_overflow() {
        let dir = TempDir::new("machines-sparse-fail");
        let blocker = dir.path().join("file");
        fs::write(&blocker, b"x").unwrap();
        let error = create_disk_image(&blocker.join("disk.udisk"), 1).unwrap_err();
        assert!(error.starts_with("create "), "{error}");
        let error = create_disk_image(&dir.path().join("big.udisk"), u64::MAX).unwrap_err();
        assert!(error.contains("too large"), "{error}");
        assert!(!dir.path().join("big.udisk").exists());
    }

    #[test]
    fn a_framebuffer_display_is_saved_with_the_machine_and_shown() {
        let home = TempDir::new("machines-display");
        let machine = create_machine(home.path(), &["--display", "framebuffer"]);
        let saved = std::fs::read_to_string(home.path().join("machines/m/machine.toml")).unwrap();
        assert!(saved.contains("display = \"framebuffer\""), "{saved}");
        let loaded = load(home.path(), "m").unwrap();
        assert_eq!(loaded.settings.display, Some(DisplayArg::Framebuffer));
        assert!(describe(&machine).contains("display   framebuffer (the console)"));
        assert!(
            !describe(&machine).contains("input"),
            "input follows the screen"
        );
        let plain = create_machine(&home.path().join("other"), &[]);
        assert!(!describe(&plain).contains("display   "));
    }

    #[test]
    fn a_screen_beside_the_terminal_and_input_alone_are_saved_and_shown() {
        let home = TempDir::new("machines-display-both");
        let both = create_machine(home.path(), &["--display", "both", "--input=false"]);
        let text = describe(&both);
        assert!(
            text.contains("display   framebuffer (the terminal is the console)"),
            "{text}"
        );
        assert!(text.contains("input     none"), "{text}");
        let input_only = create_machine(&home.path().join("other"), &["--input"]);
        let text = describe(&input_only);
        assert!(!text.contains("display   "), "{text}");
        assert!(text.contains("input     keyboard and tablet"), "{text}");
    }

    #[test]
    fn describe_numbers_every_device_for_the_remove_options() {
        let home = TempDir::new("machines-describe");
        let machine = create_machine(
            home.path(),
            &[
                "--network",
                "virtio",
                "--network-link-loop",
                "--network",
                "virtio",
                "--console",
                "virtio",
                "--console",
                "serial",
            ],
        );
        let text = describe(&machine);
        assert!(text.contains("machine m (installs alpine)"), "{text}");
        assert!(text.contains("disk 0    vda "), "{text}");
        assert!(text.contains("network 0 virtio, link loop"), "{text}");
        assert!(text.contains("network 1 virtio, no link"), "{text}");
        assert!(text.contains("console 0 virtio"), "{text}");
        assert!(text.contains("console 1 serial"), "{text}");
        assert!(text.contains("2048 MiB"), "{text}");
    }

    #[test]
    fn list_is_empty_without_a_machines_directory() {
        let home = TempDir::new("machines-list-none");
        assert!(list(home.path()).unwrap().is_empty());
        assert_eq!(format_list(&[]), "");
    }

    #[test]
    fn list_sorts_by_name_and_skips_what_is_not_a_machine() {
        let home = TempDir::new("machines-list");
        for name in ["zeta", "alpha"] {
            let (hardware, linux) = create_args(&[]);
            create(
                home.path(),
                name,
                Distro::Debian,
                &hardware,
                &linux,
                None,
                None,
            )
            .unwrap();
        }
        fs::create_dir_all(home.path().join("machines/stray")).unwrap();
        fs::write(home.path().join("machines/file"), b"x").unwrap();
        let names: Vec<String> = list(home.path())
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["alpha", "zeta"]);
    }

    #[test]
    fn a_broken_machine_is_listed_with_its_error_instead_of_hiding_the_rest() {
        let home = TempDir::new("machines-list-broken");
        create_machine(home.path(), &[]);
        let (hardware, linux) = create_args(&[]);
        create(
            home.path(),
            "good",
            Distro::Fedora,
            &hardware,
            &linux,
            None,
            None,
        )
        .unwrap();
        fs::write(home.path().join("machines/m/machine.toml"), "install = 5").unwrap();
        let text = format_list(&list(home.path()).unwrap());
        assert!(text.contains("good  fedora"), "{text}");
        assert!(text.contains("3072 MiB, 1 disk(s), 1 network(s)"), "{text}");
        assert!(
            text.lines().any(|l| l.starts_with("m     unreadable: ")),
            "{text}"
        );
    }

    #[test]
    fn show_says_where_the_machine_lives_and_whether_it_has_artifacts() {
        let home = TempDir::new("machines-show");
        let machine = create_machine(home.path(), &[]);
        let text = show(&machine);
        assert!(text.contains("disk 0    vda "), "{text}");
        assert!(
            text.contains(&format!("directory {}", machine.dir.display())),
            "{text}"
        );
        assert!(text.contains("artifacts missing"), "{text}");
        let fetcher = FakeFetcher::new();
        fetch_artifacts(&machine, |u, d| fetcher.fetch(u, d)).unwrap();
        assert!(show(&machine).contains("artifacts fetched"));
    }

    #[test]
    fn delete_removes_the_directory_and_its_disks_but_not_attached_images() {
        let home = TempDir::new("machines-delete");
        let attached = home.path().join("mine.img");
        fs::write(&attached, b"keep me").unwrap();
        let attached_arg = attached.to_string_lossy().into_owned();
        create_machine(
            home.path(),
            &["--create-disk", "1", "--disk", &attached_arg],
        );
        let plan = removal(home.path(), "m").unwrap();
        assert_eq!(plan.dir, home.path().join("machines/m"));
        assert_eq!(plan.created_disks, vec![plan.dir.join("disk-0.udisk")]);
        delete(&plan).unwrap();
        assert!(!plan.dir.exists());
        assert_eq!(fs::read(&attached).unwrap(), b"keep me");
        assert!(list(home.path()).unwrap().is_empty());
    }

    #[test]
    fn delete_counts_detached_images_still_in_the_directory() {
        let home = TempDir::new("machines-delete-detached");
        create_machine(home.path(), &[]);
        let (hardware, linux, remove) = edit_args(&["--remove-disk", "0", "--create-disk", "1"]);
        edit(home.path(), "m", &hardware, &linux, &remove, None, None).unwrap();
        assert_eq!(removal(home.path(), "m").unwrap().created_disks.len(), 2);
    }

    #[test]
    fn delete_only_ever_targets_a_machine_directory() {
        let home = TempDir::new("machines-delete-guard");
        fs::create_dir_all(home.path().join("machines/stray")).unwrap();
        fs::write(home.path().join("machines/stray/precious"), b"x").unwrap();
        assert!(
            removal(home.path(), "stray")
                .unwrap_err()
                .contains("no machine named stray")
        );
        assert!(removal(home.path(), "missing").is_err());
        for bad in ["..", "../machines", "a/b", ""] {
            assert!(removal(home.path(), bad).is_err(), "{bad:?}");
        }
        assert!(home.path().join("machines/stray/precious").exists());
    }

    #[test]
    fn the_saved_file_is_plain_toml_a_person_can_read() {
        let home = TempDir::new("machines-toml");
        create_machine(home.path(), &["--memory", "300"]);
        let text = fs::read_to_string(home.path().join("machines/m/machine.toml")).unwrap();
        assert!(text.contains("install = \"alpine\""), "{text}");
        assert!(text.contains("memory = 300"), "{text}");
        assert!(text.contains("[[disks]]"), "{text}");
    }

    #[test]
    fn firmware_defaults_to_uefi_and_old_files_without_it_load_as_uefi() {
        let home = TempDir::new("machines-firmware-serde");
        let machine = create_machine(home.path(), &[]);
        assert_eq!(machine.firmware, Firmware::Uefi);
        let path = home.path().join("machines/m/machine.toml");
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("firmware = \"uefi\""), "{text}");
        let without_key: String = text
            .lines()
            .filter(|l| !l.starts_with("firmware"))
            .map(|l| format!("{l}\n"))
            .collect();
        fs::write(&path, &without_key).unwrap();
        assert_eq!(load(home.path(), "m").unwrap().firmware, Firmware::Uefi);
        for (written, want) in [("bios", Firmware::Bios), ("none", Firmware::None)] {
            fs::write(&path, format!("firmware = \"{written}\"\n{without_key}")).unwrap();
            assert_eq!(load(home.path(), "m").unwrap().firmware, want);
        }
    }

    #[test]
    fn firmware_is_saved_by_create_and_changed_by_edit() {
        let home = TempDir::new("machines-firmware-edit");
        let (hardware, linux) = create_args(&["--firmware", "uefi"]);
        create(
            home.path(),
            "m",
            Distro::Alpine,
            &hardware,
            &linux,
            None,
            Some(Firmware::Uefi),
        )
        .unwrap();
        let text = fs::read_to_string(home.path().join("machines/m/machine.toml")).unwrap();
        assert!(text.contains("firmware = \"uefi\""), "{text}");
        assert_eq!(load(home.path(), "m").unwrap().firmware, Firmware::Uefi);
        let (hardware, linux, remove) = edit_args(&[]);
        edit(home.path(), "m", &hardware, &linux, &remove, None, None).unwrap();
        assert_eq!(load(home.path(), "m").unwrap().firmware, Firmware::Uefi);
        let edited = edit(
            home.path(),
            "m",
            &hardware,
            &linux,
            &remove,
            None,
            Some(Firmware::Bios),
        )
        .unwrap();
        assert_eq!(edited.machine.firmware, Firmware::Bios);
    }

    #[test]
    fn a_uefi_machine_boots_its_disk_with_its_nvram() {
        let home = TempDir::new("machines-firmware-launch");
        let machine = create_machine(home.path(), &[]);
        finish_install(&machine);
        for choice in [None, Some(BootSource::Disk)] {
            let uefi = launch(&machine, choice).unwrap();
            assert_eq!(
                (uefi.source, uefi.firmware),
                (BootSource::Disk, Firmware::Uefi)
            );
            assert_eq!(uefi.nvram, Some(home.path().join("machines/m/nvram.kv")));
        }
    }

    #[test]
    fn an_installer_boot_of_a_uefi_machine_keeps_the_nvram() {
        // grub-install and efibootmgr write Boot#### through efivarfs; the installed system
        // finds its loader through them.
        let home = TempDir::new("machines-firmware-installer");
        let machine = create_machine(home.path(), &[]);
        let installer = launch(&machine, None).unwrap();
        assert_eq!(
            (installer.source, installer.firmware),
            (BootSource::Installer, Firmware::Uefi)
        );
        assert!(!installer.kernel.is_empty());
        assert_eq!(installer.nvram, Some(machine.nvram_path()));
    }

    #[test]
    fn a_bios_machine_still_boots_its_installed_disk_through_the_bios() {
        let home = TempDir::new("machines-firmware-bios");
        let mut machine = create_machine(home.path(), &[]);
        machine.firmware = Firmware::Bios;
        let installer = launch(&machine, None).unwrap();
        assert_eq!(installer.source, BootSource::Installer);
        assert_eq!(installer.nvram, None);
        finish_install(&machine);
        let bios = launch(&machine, None).unwrap();
        assert_eq!(
            (bios.source, bios.firmware, bios.nvram),
            (BootSource::Disk, Firmware::Bios, None)
        );
    }

    #[test]
    fn none_cannot_boot_an_installed_disk() {
        let home = TempDir::new("machines-firmware-none");
        let mut machine = create_machine(home.path(), &[]);
        machine.firmware = Firmware::None;
        let installer = launch(&machine, None).unwrap();
        assert_eq!(
            (installer.source, installer.nvram),
            (BootSource::Installer, None)
        );
        finish_install(&machine);
        let error = launch(&machine, None).err().expect("must be refused");
        assert!(
            error.contains("firmware none cannot boot a disk"),
            "{error}"
        );
    }

    #[test]
    fn show_lists_the_firmware_and_for_uefi_the_nvram() {
        let home = TempDir::new("machines-firmware-show");
        let mut machine = create_machine(home.path(), &[]);
        let nvram = machine.nvram_path();
        let text = show(&machine);
        assert!(text.contains("firmware  uefi"), "{text}");
        assert!(
            text.contains(&format!("nvram     {} not created yet", nvram.display())),
            "{text}"
        );
        fs::write(&nvram, b"").unwrap();
        assert!(show(&machine).contains(&format!("{} present", nvram.display())));
        machine.firmware = Firmware::Bios;
        let text = show(&machine);
        assert!(text.contains("firmware  bios"), "{text}");
        assert!(!text.contains("nvram"), "{text}");
    }

    #[test]
    fn delete_removes_the_nvram_with_the_directory() {
        let home = TempDir::new("machines-delete-nvram");
        let machine = create_machine(home.path(), &[]);
        fs::write(machine.nvram_path(), b"variables").unwrap();
        let plan = removal(home.path(), "m").unwrap();
        assert!(plan.created_disks.iter().all(|d| !d.ends_with("nvram.kv")));
        delete(&plan).unwrap();
        assert!(!machine.nvram_path().exists());
        assert!(!plan.dir.exists());
    }
}
