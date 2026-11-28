mod args;
mod bootstrap;
mod checkpoints;
mod firmware;
mod frontend;
mod images;
mod input;
mod machines;
mod netgw;
mod quick;
mod run;
mod term;
#[cfg(test)]
mod test_support;
mod usernet;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand};
use uemulator::ExitReason;

use args::{
    BootSourceArgs, BootstrapArgs, FirmwareArgs, HardwareArgs, LinuxArgs, NewDiskArgs, RemoveArgs,
    RuntimeArgs, Settings, UefiBootArgs, attach_only,
};
use quick::Distro;

/// The machine ran and ended in a way that is not a failure of the emulator or the guest.
const EXIT_SUCCESS: i32 = 0;
/// Setup failed before or while starting the machine: bad arguments, unreadable files, failed
/// downloads, configuration the library rejected.
const EXIT_ERROR: i32 = 1;
/// The machine started and then failed: a guest triple fault or an emulator-internal error
/// (including a JIT divergence under `--verify`).
const EXIT_MACHINE_FAILURE: i32 = 2;

const EXIT_CODES_HELP: &str = "\
Exit codes:
  0  the machine ended by itself (guest power-off, or reset with --no-reboot), or was stopped
     (or suspended, with --suspend-to) with Ctrl-] or --timeout
  1  setup failed (bad arguments, unreadable file, download failure, rejected configuration)
  2  the machine failed: guest triple fault or emulator-internal error (e.g. --verify
     divergence), or a suspend or resume failed";

/// Maps how the machine ended to the process exit status. A power-off is how a guest finishes
/// cleanly; a reset only reaches here with `--no-reboot` and counts as success because Linux
/// guests reboot on panic with `panic=-1` and a clean reboot is how an installer finishes.
/// Stopping the machine (Ctrl-] or `--timeout`) is a deliberate host action, so it is success too;
/// callers that need to tell a timeout from a normal end can run without `--timeout`.
fn exit_code(reason: &ExitReason) -> i32 {
    match reason {
        ExitReason::Shutdown | ExitReason::Reset | ExitReason::Stopped | ExitReason::Suspended => {
            EXIT_SUCCESS
        }
        ExitReason::TripleFault
        | ExitReason::Internal(_)
        | ExitReason::SuspendFailed(_)
        | ExitReason::ResumeFailed(_) => EXIT_MACHINE_FAILURE,
    }
}

#[derive(Parser)]
#[command(
    name = "uemulator",
    about = "Minimal x86_64 system emulator",
    after_help = EXIT_CODES_HELP
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Named machines that install a distro from its netboot images, kept in ~/.uemulator/machines
    Quick {
        #[command(subcommand)]
        cmd: QuickCmd,
    },
    /// Boot a kernel or a firmware image directly, with every option spelled out
    Advanced {
        #[command(subcommand)]
        cmd: AdvancedCmd,
    },
    /// Create, convert and inspect disk images (raw, or udisk: docs/spec/uemulator-disk-format.md)
    Disk {
        #[command(subcommand)]
        cmd: DiskCmd,
    },
}

/// udisk images are sparse and can be copy-on-write overlays over a base image, which may be
/// local or an http(s) URL; `--disk` takes either format and finds an overlay's base itself.
#[derive(Subcommand)]
enum DiskCmd {
    /// Create an empty udisk image, or an overlay that keeps its writes apart from a base image
    Create {
        /// The new image; must not exist
        file: PathBuf,
        /// Size in MiB [default: the base's size]
        #[arg(long, value_name = "MIB", value_parser = clap::value_parser!(u64).range(1..))]
        size: Option<u64>,
        /// Make an overlay over this image (raw or udisk; a path, relative to FILE's directory,
        /// or an http(s) URL served with range requests). The base is opened read-only and must
        /// not change afterwards
        #[arg(long, value_name = "BASE")]
        backing: Option<String>,
        /// Cluster size in KiB, a power of two from 64 to 16384 [default: 64]
        #[arg(long, value_name = "KIB")]
        cluster_size: Option<u64>,
    },
    /// Copy a disk image into a new one, raw or udisk. An overlay comes out flattened with its
    /// bases; a udisk comes out with its clusters in disk order, ready to serve over HTTP
    Convert {
        /// The image to read: a path or an http(s) URL
        source: String,
        /// The image to write; must not exist
        destination: PathBuf,
        /// Format to write [default: raw for a .img or .raw DESTINATION, else udisk]
        #[arg(long, value_enum)]
        format: Option<images::ImageFormat>,
        /// Cluster size of a udisk DESTINATION in KiB [default: 64]
        #[arg(long, value_name = "KIB")]
        cluster_size: Option<u64>,
    },
    /// Show an image's format, size and, for udisk, its base and how much of it is stored
    Info {
        /// A path or an http(s) URL
        image: String,
    },
}

/// Options that change a saved machine apply to it for good (`create`, `edit`) or to one run
/// (`boot`). Lists of devices (`--disk`, `--network`, `--console`) add to the saved ones;
/// `--remove-*` takes saved ones away first.
#[derive(Subcommand)]
enum QuickCmd {
    /// Create a machine, downloading its kernel and initramfs
    ///
    /// Without --disk or --create-disk the machine gets a 64 GiB (sparse) disk, without --network a
    /// NIC linked with user-mode NAT, and without --console a virtio console.
    Create {
        /// Machine name: letters, digits, '.', '_' and '-'
        name: String,
        /// The distro whose netboot installer the machine boots
        #[arg(long, value_name = "DISTRO")]
        install: Distro,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        new_disks: NewDiskArgs,
        #[command(flatten)]
        linux: LinuxArgs,
        #[command(flatten)]
        source: BootSourceArgs,
        #[command(flatten)]
        firmware: FirmwareArgs,
        /// Boot the machine once it is created
        #[arg(long)]
        boot: bool,
    },
    /// Create a machine and install its distro unattended with these answers, then boot the
    /// installed system
    ///
    /// The machine is `quick create`'s, with the answers saved (never a password) for `quick
    /// reinstall` to repeat. --firmware none and --hypervisor are refused: neither can boot the
    /// installed disk.
    #[command(mut_arg("user", |user| user.required(true)))]
    #[command(mut_arg("timeout", |timeout| timeout.help(BOOTSTRAP_TIMEOUT_HELP)))]
    Bootstrap {
        /// Machine name: letters, digits, '.', '_' and '-'
        name: String,
        /// The distro to install
        #[arg(long, value_name = "DISTRO")]
        install: Distro,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        new_disks: NewDiskArgs,
        #[command(flatten)]
        linux: LinuxArgs,
        #[command(flatten)]
        firmware: FirmwareArgs,
        /// Boxed, as `runtime` is: together they would make this variant the size of the enum.
        #[command(flatten)]
        bootstrap: Box<BootstrapArgs>,
        #[command(flatten)]
        uefi: UefiBootArgs,
        #[command(flatten)]
        runtime: Box<RuntimeArgs>,
    },
    /// Change a machine's saved configuration
    ///
    /// Removals apply first and their INDEXes refer to the lists as `create` and `edit` print
    /// them; then the new devices are added after the ones that remain.
    Edit {
        /// The machine to change
        name: String,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        new_disks: NewDiskArgs,
        #[command(flatten)]
        linux: LinuxArgs,
        #[command(flatten)]
        remove: RemoveArgs,
        #[command(flatten)]
        source: BootSourceArgs,
        #[command(flatten)]
        firmware: FirmwareArgs,
    },
    /// Boot a machine, optionally with options that apply to this run only
    Boot {
        /// The machine to boot
        name: String,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        linux: LinuxArgs,
        #[command(flatten)]
        remove: RemoveArgs,
        #[command(flatten)]
        source: BootSourceArgs,
        #[command(flatten)]
        uefi: UefiBootArgs,
        /// Boxed: with the GDB and checkpoint options this variant is far larger than the rest.
        #[command(flatten)]
        runtime: Box<RuntimeArgs>,
    },
    /// Install a machine's distro again, then boot
    ///
    /// The kernel and initramfs are downloaded fresh; the settings are kept. A machine made by
    /// `quick bootstrap` (or any machine, given --user) is installed unattended with its saved
    /// answers, which the options here override and replace, and its first disk is erased after
    /// a question (--yes skips it). The passwords are not saved: they are asked for again, or
    /// given as options. Any other machine boots its distro's interactive installer.
    #[command(mut_arg("timeout", |timeout| timeout.help(BOOTSTRAP_TIMEOUT_HELP)))]
    Reinstall {
        /// The machine to reinstall
        name: String,
        /// Switch the machine to another distro [default: the one it has]
        #[arg(long, value_name = "DISTRO")]
        install: Option<Distro>,
        /// Boxed, as `runtime` is: together they would make this variant the size of the enum.
        #[command(flatten)]
        bootstrap: Box<BootstrapArgs>,
        /// Erase the disk without asking, for an unattended reinstall; required there when
        /// stdin is not a terminal to ask on
        #[arg(long)]
        yes: bool,
        /// Give root no password this time, though the saved answers say it had one
        #[arg(long, conflicts_with = "root_password_source")]
        no_root_password: bool,
        #[command(flatten)]
        uefi: UefiBootArgs,
        #[command(flatten)]
        runtime: Box<RuntimeArgs>,
    },
    /// Download the current kernel and initramfs of the machine's distro
    UpdateArtifacts {
        /// The machine whose kernel and initramfs to refresh
        name: String,
    },
    /// List the machines
    List,
    /// Show a machine's configuration, numbered the way `edit` and `boot` take indices
    Show {
        /// The machine to show
        name: String,
    },
    /// Delete a machine: its directory, including the disk images --create-disk made in it
    ///
    /// Disks attached with --disk are not touched.
    Delete {
        /// The machine to delete
        name: String,
        /// Delete without further ado; required, because the disk images cannot be recovered
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum AdvancedCmd {
    /// Boot a kernel (bzImage) and optional initramfs, optionally under a hypervisor
    ///
    /// The UEFI firmware boots them, which needs a kernel with an EFI stub; `--firmware none`
    /// enters any bzImage directly. With `--hypervisor` the firmware boots Xen and the kernel
    /// becomes its dom0.
    Boot {
        /// Linux bzImage
        #[arg(long)]
        kernel: PathBuf,
        /// Initial ramdisk
        #[arg(long)]
        initramfs: Option<PathBuf>,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        linux: LinuxArgs,
        #[command(flatten)]
        firmware: FirmwareArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
    },
    /// Boot a firmware image (a BIOS, or a real-mode program) at the reset vector in real mode
    BootFirmware {
        /// Image placed so that it ends at 1 MiB; execution starts at F000:FFF0. A guest reset
        /// ends the run
        #[arg(long)]
        firmware: PathBuf,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
    },
    /// Boot the machine's disks through the built-in UEFI firmware
    ///
    /// The firmware runs the boot entries in the nvram file's BootOrder, else
    /// EFI/BOOT/BOOTX64.EFI from the first file system that has it.
    BootUefi {
        /// Persist UEFI variables (BootOrder, Boot####, anything the guest writes) in this file,
        /// created if missing. Without it variables last until the guest resets
        #[arg(long, value_name = "FILE")]
        nvram: Option<PathBuf>,
        /// Make Boot0000 run this path on the first file system that has it, and put it first in
        /// BootOrder. Needs --nvram
        #[arg(long, value_name = "PATH", requires = "nvram")]
        boot_loader: Option<String>,
        /// Load options of --boot-loader: for a Linux EFI stub the kernel command line, with
        /// initrd=PATH naming the initramfs
        #[arg(long, value_name = "OPTIONS", requires = "boot_loader")]
        boot_options: Option<String>,
        #[command(flatten)]
        uefi: UefiBootArgs,
        #[command(flatten)]
        hardware: HardwareArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
    },
}

impl Cmd {
    /// The hardware options of the command, for the parts of them clap cannot order.
    fn hardware_mut(&mut self) -> Option<&mut HardwareArgs> {
        match self {
            Cmd::Quick {
                cmd:
                    QuickCmd::Create { hardware, .. }
                    | QuickCmd::Bootstrap { hardware, .. }
                    | QuickCmd::Edit { hardware, .. }
                    | QuickCmd::Boot { hardware, .. },
            }
            | Cmd::Advanced {
                cmd:
                    AdvancedCmd::Boot { hardware, .. }
                    | AdvancedCmd::BootFirmware { hardware, .. }
                    | AdvancedCmd::BootUefi { hardware, .. },
            } => Some(hardware),
            Cmd::Quick { .. } | Cmd::Disk { .. } => None,
        }
    }
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// The settings of a machine that has no saved ones: nothing but what the command line says.
fn settings_of(hardware: &HardwareArgs, linux: &LinuxArgs) -> Result<Settings, String> {
    let mut settings = Settings::default();
    settings.add(hardware, linux, attach_only(&hardware.disk_steps)?);
    Ok(settings)
}

fn refuse_uefi_args(uefi: &UefiBootArgs, firmware: args::Firmware) -> Result<(), String> {
    if uefi.any() && firmware != args::Firmware::Uefi {
        return Err("--boot-menu and --boot-timeout need a machine with --firmware uefi".into());
    }
    Ok(())
}

/// `--timeout` on the commands that run an unattended install and then boot the installed disk:
/// each of the two runs gets the whole timeout.
const BOOTSTRAP_TIMEOUT_HELP: &str = "Stop the machine after this many seconds: once for the \
    installer, and again for the installed system's boot that follows a finished install";

const INSTALLER_BOOT_MENU: &str =
    "--boot-menu is for disk boots: the installer's kernel is booted directly, without the menu";

fn boot_launch(
    launch: machines::Launch,
    uefi: &UefiBootArgs,
    runtime: &RuntimeArgs,
) -> Result<ExitReason, String> {
    refuse_uefi_args(uefi, launch.firmware)?;
    let nvram_setup = firmware::NvramSetup {
        boot_entry: None,
        boot_menu: uefi.boot_menu,
        boot_timeout: uefi.boot_timeout,
    };
    match launch.source {
        args::BootSource::Disk => {
            eprintln!("booting from the disk");
            match launch.firmware {
                args::Firmware::Bios => run::run_disk(&launch.settings, runtime),
                args::Firmware::Uefi => run::run_uefi(
                    Vec::new(),
                    None,
                    None,
                    &launch.settings,
                    runtime,
                    launch.nvram.as_deref(),
                    &nvram_setup,
                ),
                args::Firmware::None => Err("firmware none cannot boot a disk".into()),
            }
        }
        _ => match launch.firmware {
            args::Firmware::Uefi => {
                if uefi.boot_menu {
                    return Err(INSTALLER_BOOT_MENU.into());
                }
                match launch.hypervisor {
                    Some(_) => eprintln!(
                        "booting the installer through the UEFI firmware, as the hypervisor's dom0"
                    ),
                    None => eprintln!("booting the installer through the UEFI firmware"),
                }
                run::run_uefi(
                    launch.kernel,
                    launch.initramfs,
                    launch.hypervisor,
                    &launch.settings,
                    runtime,
                    launch.nvram.as_deref(),
                    &nvram_setup,
                )
            }
            args::Firmware::Bios | args::Firmware::None => {
                eprintln!("booting the installer");
                run::run(launch.kernel, launch.initramfs, &launch.settings, runtime)
            }
        },
    }
}

#[derive(Debug, PartialEq, Eq)]
enum AfterInstaller {
    /// The install finished and the guest rebooted: boot the disk it wrote.
    BootDisk,
    /// The guest reset before the install said it had finished.
    Failed,
    /// The install finished, but the run ended some other way before the disk boot: the host
    /// stopped or suspended it (Ctrl-], `--timeout`) or the guest powered off. The machine is
    /// switched to its disk, as the app does, so the next `quick boot` does not install again.
    Installed,
    /// The run is over before the install finished: the host stopped or suspended it, or the
    /// guest powered off or failed. The machine stays on its installer.
    Done,
}

/// What `quick bootstrap` does once its installer run has ended. The answer files print the
/// marker and then reboot, and the run has `no_reboot` set, so only a reset is the install's own
/// end and only a reset boots the disk; any other reason is reported as `quick boot` would, but
/// once the marker was seen the install is done whichever way the run ended.
fn after_installer(reason: &ExitReason, marker_seen: bool) -> AfterInstaller {
    match (reason, marker_seen) {
        (ExitReason::Reset, true) => AfterInstaller::BootDisk,
        (ExitReason::Reset, false) => AfterInstaller::Failed,
        (_, true) => AfterInstaller::Installed,
        (_, false) => AfterInstaller::Done,
    }
}

/// Installs `machine` unattended with `answers`, then boots the installed disk
/// (`docs/plans/35-quick-bootstrap.md`, "After the install"). The disk boot is `quick boot`'s:
/// the user's own reboot setting, `uefi`'s boot menu and timeout, and no marker watch.
fn run_bootstrap(
    machine: &mut machines::Machine,
    answers: &uemulator_bootstrap::Answers,
    uefi: &UefiBootArgs,
    runtime: &mut RuntimeArgs,
) -> Result<ExitReason, String> {
    // Before the install, which takes a while, rather than at the disk boot after it.
    refuse_uefi_args(uefi, machine.firmware)?;
    // A new machine is made `auto`, which would boot a disk the installer partitioned and then
    // failed on (seen with Debian on a disk too small for its recipe): until the marker, the
    // machine stays on its installer, as a reinstall already leaves it.
    machine.set_boot_from(args::BootSource::Installer)?;
    let launch = machines::bootstrap_launch(machine, answers, quick::download)?;
    let watch = bootstrap::MarkerWatch::default();
    runtime.marker_watch = Some(watch.clone());
    // `--boot-menu` is refused for an installer boot; it is for the disk boot that follows.
    let installed = boot_launch(launch, &UefiBootArgs::default(), runtime);
    runtime.marker_watch = None;
    let reason = installed?;
    let name = machine.name.clone();
    match after_installer(&reason, watch.seen()) {
        AfterInstaller::Done => Ok(reason),
        AfterInstaller::Installed => {
            machine.set_boot_from(args::BootSource::Auto)?;
            Ok(reason)
        }
        AfterInstaller::Failed => Err(format!(
            "the installer of {name} reset before it finished: run `uemulator quick reinstall \
             {name}` to try again"
        )),
        AfterInstaller::BootDisk => {
            // Saved before the disk boot starts: stopping that boot must not leave the machine
            // to run its installer again on the next `quick boot`.
            machine.set_boot_from(args::BootSource::Auto)?;
            eprintln!("installed; booting {name} from its disk");
            let launch = machines::prepare_boot(
                machine,
                &HardwareArgs::default(),
                &LinuxArgs::default(),
                &RemoveArgs::default(),
                None,
                None,
                quick::download,
            )?;
            boot_launch(launch, uefi, runtime)
        }
    }
}

/// What `quick reinstall` was asked to do, short of running anything.
struct ReinstallRequest<'a> {
    name: &'a str,
    install: Option<Distro>,
    bootstrap: &'a BootstrapArgs,
    yes: bool,
    no_root_password: bool,
    uefi: &'a UefiBootArgs,
}

/// Readies `quick reinstall` (`docs/plans/35-quick-bootstrap.md`, "Reinstall"): the machine with
/// fresh artifacts and the new answers saved, and the answers when the reinstall is unattended
/// (the machine has saved answers, or --user is given). Everything that can refuse runs first,
/// the passwords and the erase question included: until the artifacts are thrown away a refused
/// reinstall leaves the machine as it was, still booting its installed system.
fn prepare_reinstall(
    home: &std::path::Path,
    request: &ReinstallRequest,
    tty: bool,
    read_secret: impl FnMut(&str) -> std::io::Result<String>,
    ask: impl FnMut(&str) -> std::io::Result<String>,
    fetch: impl FnMut(&str, &std::path::Path) -> Result<(), String>,
) -> Result<(machines::Machine, Option<uemulator_bootstrap::Answers>), String> {
    let name = request.name;
    let mut machine = machines::load(home, name)?;
    refuse_uefi_args(request.uefi, machine.firmware)?;
    let unattended = machine.bootstrap.is_some() || request.bootstrap.user.is_some();
    if !unattended {
        if request.bootstrap.any() || request.no_root_password {
            return Err(format!(
                "{name} has no saved answers, so it reinstalls with the interactive installer: \
                 give --user NAME to install it unattended with these answers"
            ));
        }
        // `boot_launch` refuses it too, but only after the machine was switched to its
        // installer and its artifacts thrown away. An unattended reinstall keeps it for the disk
        // boot after the install.
        if request.uefi.boot_menu {
            return Err(INSTALLER_BOOT_MENU.into());
        }
        let machine = machines::reinstall(home, name, request.install, None, fetch)?;
        return Ok((machine, None));
    }
    bootstrap::refuse_unbootable_machine(machine.firmware, machine.hypervisor.is_some(), name)?;
    bootstrap::refuse_framebuffer_machine(machine.settings.display, name)?;
    let mut saved = machine.bootstrap.clone();
    if request.no_root_password
        && let Some(record) = &mut saved
    {
        record.root_password = false;
    }
    let (answers, record) =
        bootstrap::resolve_answers(request.bootstrap, saved.as_ref(), name, tty, read_secret)?;
    // The question names the distro being installed, which --install may have changed.
    machine.install = request.install.unwrap_or(machine.install);
    confirm_erase(&machine, tty, request.yes, ask)?;
    let machine = machines::reinstall(home, name, request.install, Some(record), fetch)?;
    Ok((machine, Some(answers)))
}

/// Asks before an unattended install erases `machine`'s first disk, the one the answer files
/// install to; the installer itself no longer asks. `yes` (--yes) answers for the user, and
/// without a terminal it is the only answer. A machine without a disk has nothing to erase.
fn confirm_erase(
    machine: &machines::Machine,
    tty: bool,
    yes: bool,
    mut ask: impl FnMut(&str) -> std::io::Result<String>,
) -> Result<(), String> {
    let Some(disk) = machine.settings.disks.first() else {
        return Ok(());
    };
    if yes {
        return Ok(());
    }
    // A disk the machine made is named by its file, as `quick show` lists the machine's own;
    // one attached with --disk lives elsewhere and gets its whole path.
    let shown = match (disk.path.parent(), disk.path.file_name()) {
        (Some(parent), Some(file)) if parent == machine.dir => file.to_string_lossy(),
        _ => disk.path.to_string_lossy(),
    };
    let size = images::open_disk(&images::Location::File(disk.path.clone()), true)
        .map_err(|e| format!("open {}: {e}", disk.path.display()))?
        .len();
    let question = format!(
        "Erase {shown} ({}) and install {}? [y/N]",
        size_in_words(size),
        machine.install.title()
    );
    if !tty {
        return Err(format!(
            "the unattended install erases {shown} and stdin is not a terminal to ask on: give \
             --yes to erase it"
        ));
    }
    let answer = ask(&question).map_err(|e| format!("read the answer: {e}"))?;
    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(()),
        _ => Err("not reinstalled".into()),
    }
}

/// A disk size as a person would say it: whole GiB or MiB when it is one, else bytes.
fn size_in_words(bytes: u64) -> String {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if bytes != 0 && bytes.is_multiple_of(GIB) {
        format!("{} GiB", bytes / GIB)
    } else if bytes != 0 && bytes.is_multiple_of(MIB) {
        format!("{} MiB", bytes / MIB)
    } else {
        format!("{bytes} bytes")
    }
}

/// Prints `question` on stderr and reads one line of answer from stdin, echoed as typed.
fn ask_on_terminal(question: &str) -> std::io::Result<String> {
    use std::io::Write;
    let mut stderr = std::io::stderr();
    write!(stderr, "{question} ")?;
    stderr.flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(answer)
}

/// `None` is a command that did not run a machine.
fn quick_command(cmd: QuickCmd) -> Result<Option<ExitReason>, String> {
    let home = machines::home()?;
    match cmd {
        QuickCmd::Create {
            name,
            install,
            hardware,
            new_disks: _,
            linux,
            source,
            firmware,
            boot,
        } => {
            let machine = machines::create(
                &home,
                &name,
                install,
                &hardware,
                &linux,
                source.boot_from,
                firmware.firmware,
                None,
            )?;
            machines::fetch_artifacts(&machine, quick::download).map_err(|e| {
                format!("{e}\nmachine {name} was created; `quick boot {name}` retries the download")
            })?;
            eprint!("{}", machines::describe(&machine));
            if !boot {
                return Ok(None);
            }
            let launch = machines::prepare_boot(
                &machine,
                &HardwareArgs::default(),
                &LinuxArgs::default(),
                &RemoveArgs::default(),
                None,
                None,
                quick::download,
            )?;
            boot_launch(launch, &UefiBootArgs::default(), &RuntimeArgs::default()).map(Some)
        }
        QuickCmd::Bootstrap {
            name,
            install,
            hardware,
            new_disks: _,
            linux,
            firmware,
            bootstrap: bootstrap_args,
            uefi,
            mut runtime,
        } => {
            // Before anything else, the password prompt included: `machines::exists` below
            // would look outside the machines directory for a name such as `../x`.
            machines::validate_name(&name)?;
            bootstrap::refuse_unbootable(firmware.firmware, &linux.hypervisor)?;
            bootstrap::refuse_attached_first_disk(&hardware.disk_steps)?;
            bootstrap::refuse_framebuffer_display(hardware.display)?;
            run::refuse_runtime_conflicts(&runtime)?;
            if runtime.resume_from.is_some() {
                return Err(format!(
                    "--resume-from: bootstrap makes a new machine; resume one with `uemulator \
                     quick boot {name} --resume-from FILE`"
                ));
            }
            // Before the password is asked for, which would be wasted. `machines::create` still
            // refuses a name that turns up meanwhile.
            if machines::exists(&home, &name) {
                return Err(format!(
                    "machine {name} already exists: `uemulator quick reinstall {name} --install \
                     DISTRO --user USER` installs it again unattended, erasing its disk"
                ));
            }
            // Before the machine exists, so a mistyped password or a missing key file leaves
            // nothing behind.
            let (answers, record) = bootstrap::resolve_answers(
                &bootstrap_args,
                None,
                &name,
                std::io::stdin().is_terminal(),
                bootstrap::read_secret_from_terminal,
            )?;
            let mut machine = machines::create(
                &home,
                &name,
                install,
                &hardware,
                &linux,
                None,
                firmware.firmware,
                Some(record),
            )?;
            machines::fetch_artifacts(&machine, quick::download).map_err(|e| {
                format!(
                    "{e}\nmachine {name} was created; `uemulator quick reinstall {name}` retries \
                     the download"
                )
            })?;
            eprint!("{}", machines::describe(&machine));
            run_bootstrap(&mut machine, &answers, &uefi, &mut runtime).map(Some)
        }
        QuickCmd::Edit {
            name,
            hardware,
            new_disks: _,
            linux,
            remove,
            source,
            firmware,
        } => {
            let edited = machines::edit(
                &home,
                &name,
                &hardware,
                &linux,
                &remove,
                source.boot_from,
                firmware.firmware,
            )?;
            for disk in &edited.detached {
                eprintln!(
                    "detached {} (the image is left in place)",
                    disk.path.display()
                );
            }
            eprint!("{}", machines::describe(&edited.machine));
            Ok(None)
        }
        QuickCmd::Boot {
            name,
            hardware,
            linux,
            remove,
            source,
            uefi,
            mut runtime,
        } => {
            let machine = machines::load(&home, &name)?;
            // Before `prepare_boot` copies a checkpoint's disks back.
            run::refuse_runtime_conflicts(&runtime)?;
            let launch = machines::prepare_boot(
                &machine,
                &hardware,
                &linux,
                &remove,
                source.boot_from,
                runtime.resume_from.as_deref(),
                quick::download,
            )?;
            runtime.checkpoint_files_restored = runtime
                .resume_from
                .as_deref()
                .is_some_and(checkpoints::is_checkpoint);
            boot_launch(launch, &uefi, &runtime).map(Some)
        }
        QuickCmd::Reinstall {
            name,
            install,
            bootstrap: bootstrap_args,
            yes,
            no_root_password,
            uefi,
            mut runtime,
        } => {
            run::refuse_runtime_conflicts(&runtime)?;
            if runtime.resume_from.is_some() {
                return Err(format!(
                    "--resume-from: reinstall starts the install over; resume a machine with \
                     `uemulator quick boot {name} --resume-from FILE`"
                ));
            }
            let tty = std::io::stdin().is_terminal();
            let request = ReinstallRequest {
                name: &name,
                install,
                bootstrap: &bootstrap_args,
                yes,
                no_root_password,
                uefi: &uefi,
            };
            let (mut machine, answers) = prepare_reinstall(
                &home,
                &request,
                tty,
                bootstrap::read_secret_from_terminal,
                ask_on_terminal,
                quick::download,
            )?;
            eprint!("{}", machines::describe(&machine));
            match answers {
                Some(answers) => run_bootstrap(&mut machine, &answers, &uefi, &mut runtime),
                None => {
                    // The machine stays on its installer: an interactive install does not say
                    // when it has finished, so nothing switches it to the disk.
                    eprintln!(
                        "after the install, `uemulator quick edit {name} --boot-from auto` boots \
                         the disk"
                    );
                    let launch = machines::prepare_boot(
                        &machine,
                        &HardwareArgs::default(),
                        &LinuxArgs::default(),
                        &RemoveArgs::default(),
                        None,
                        None,
                        quick::download,
                    )?;
                    boot_launch(launch, &uefi, &runtime)
                }
            }
            .map(Some)
        }
        QuickCmd::UpdateArtifacts { name } => {
            machines::update_artifacts(&home, &name, quick::download)?;
            Ok(None)
        }
        QuickCmd::List => {
            let entries = machines::list(&home)?;
            if entries.is_empty() {
                eprintln!(
                    "no machines: create one with `uemulator quick create NAME --install DISTRO`"
                );
            }
            print!("{}", machines::format_list(&entries));
            Ok(None)
        }
        QuickCmd::Show { name } => {
            print!("{}", machines::show(&machines::load(&home, &name)?));
            Ok(None)
        }
        QuickCmd::Delete { name, yes } => {
            let removal = machines::removal(&home, &name)?;
            if !yes {
                return Err(format!(
                    "this deletes {} and the {} disk image(s) in it; run again with --yes",
                    removal.dir.display(),
                    removal.created_disks.len()
                ));
            }
            machines::delete(&removal)?;
            eprintln!("deleted machine {name}");
            Ok(None)
        }
    }
}

fn advanced_command(cmd: AdvancedCmd) -> Result<ExitReason, String> {
    match cmd {
        AdvancedCmd::Boot {
            kernel,
            initramfs,
            hardware,
            linux,
            firmware,
            runtime,
        } => {
            let firmware = firmware.firmware.unwrap_or_default();
            let args::HypervisorArgs {
                hypervisor,
                hypervisor_cmdline,
            } = &linux.hypervisor;
            let hypervisor = match hypervisor {
                None if hypervisor_cmdline.is_some() => {
                    return Err("--hypervisor-cmdline needs --hypervisor".into());
                }
                None => None,
                Some(_) if firmware != args::Firmware::Uefi => {
                    return Err(
                        "--hypervisor is booted by the UEFI firmware; use --firmware uefi".into(),
                    );
                }
                Some(path) => Some(run::Hypervisor {
                    image: read(path)?,
                    cmdline: hypervisor_cmdline.clone(),
                }),
            };
            let kernel_image = read(&kernel)?;
            let initramfs_image = initramfs.as_ref().map(read).transpose()?;
            let settings = settings_of(&hardware, &linux)?;
            match firmware {
                args::Firmware::Uefi => run::run_uefi(
                    kernel_image,
                    initramfs_image,
                    hypervisor,
                    &settings,
                    &runtime,
                    None,
                    &firmware::NvramSetup::default(),
                ),
                args::Firmware::None => {
                    run::run(kernel_image, initramfs_image, &settings, &runtime)
                }
                args::Firmware::Bios => {
                    Err("--firmware bios boots a disk, not a kernel; use uefi or none".into())
                }
            }
        }
        AdvancedCmd::BootFirmware {
            firmware,
            hardware,
            runtime,
        } => {
            let image = read(&firmware)?;
            let settings = settings_of(&hardware, &LinuxArgs::default())?;
            run::run_firmware(image, &settings, &runtime)
        }
        AdvancedCmd::BootUefi {
            nvram,
            boot_loader,
            boot_options,
            uefi,
            hardware,
            runtime,
        } => {
            let settings = settings_of(&hardware, &LinuxArgs::default())?;
            let nvram_setup = firmware::NvramSetup {
                boot_entry: boot_loader
                    .as_deref()
                    .map(|loader| (loader, boot_options.as_deref().unwrap_or(""))),
                boot_menu: uefi.boot_menu,
                boot_timeout: uefi.boot_timeout,
            };
            run::run_uefi(
                Vec::new(),
                None,
                None,
                &settings,
                &runtime,
                nvram.as_deref(),
                &nvram_setup,
            )
        }
    }
}

fn execute(cli: Cli) -> Result<Option<ExitReason>, String> {
    match cli.cmd {
        Cmd::Quick { cmd } => quick_command(cmd),
        Cmd::Advanced { cmd } => advanced_command(cmd).map(Some),
        Cmd::Disk { cmd } => disk_command(cmd).map(|()| None),
    }
}

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;

fn disk_command(cmd: DiskCmd) -> Result<(), String> {
    match cmd {
        DiskCmd::Create {
            file,
            size,
            backing,
            cluster_size,
        } => {
            let size = size
                .map(|mib| mib.checked_mul(MIB).ok_or("--size is too large"))
                .transpose()?;
            let info = images::create_udisk(
                &file,
                size,
                backing.as_deref(),
                cluster_size.map(|kib| kib.saturating_mul(KIB)),
            )?;
            eprintln!(
                "created {} ({} MiB{})",
                file.display(),
                info.virtual_size.div_ceil(MIB),
                match &backing {
                    Some(base) => format!(", over {base}"),
                    None => String::new(),
                }
            );
            Ok(())
        }
        DiskCmd::Convert {
            source,
            destination,
            format,
            cluster_size,
        } => {
            let format = format.unwrap_or_else(|| images::ImageFormat::for_path(&destination));
            let size = images::convert(
                &images::Location::parse(&source),
                &destination,
                format,
                cluster_size.map(|kib| kib.saturating_mul(KIB)),
            )?;
            eprintln!(
                "wrote {} ({} MiB, {format})",
                destination.display(),
                size.div_ceil(MIB)
            );
            Ok(())
        }
        DiskCmd::Info { image } => {
            print!("{}", images::describe(&images::Location::parse(&image))?);
            Ok(())
        }
    }
}

/// clap exits 2 on a usage error, which would collide with `EXIT_MACHINE_FAILURE`; help requests
/// are not errors.
fn usage_exit_code(error: &clap::Error) -> i32 {
    if error.use_stderr() {
        EXIT_ERROR
    } else {
        EXIT_SUCCESS
    }
}

/// The matches of the command that was actually named, however deep the subcommands nest.
fn leaf(matches: &ArgMatches) -> &ArgMatches {
    match matches.subcommand() {
        Some((_, sub)) => leaf(sub),
        None => matches,
    }
}

/// `Cli::try_parse_from` plus the part clap's derive cannot do: ordering the devices.
fn parse_cli<I, T>(argv: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let command = Cli::command();
    let matches = command.clone().try_get_matches_from(argv)?;
    let mut cli = Cli::from_arg_matches(&matches)?;
    if let Some(hardware) = cli.cmd.hardware_mut() {
        hardware.order(&command, leaf(&matches))?;
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = match parse_cli(std::env::args_os()) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(usage_exit_code(&error) as u8);
        }
    };
    match execute(cli) {
        Ok(Some(reason)) => ExitCode::from(exit_code(&reason) as u8),
        Ok(None) => ExitCode::from(EXIT_SUCCESS as u8),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(EXIT_ERROR as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;
    use clap::error::ErrorKind;

    use super::*;
    use crate::args::{ConsoleArg, DiskSpec, DiskStep, NetLink, NetModel, Nic, parse_disk_spec};
    use crate::bootstrap::BootstrapRecord;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["uemulator"];
        argv.extend_from_slice(args);
        parse_cli(argv)
    }

    fn error_kind(args: &[&str]) -> ErrorKind {
        parse(args).err().expect("parse should fail").kind()
    }

    struct Advanced {
        kernel: PathBuf,
        initramfs: Option<PathBuf>,
        hardware: HardwareArgs,
        linux: LinuxArgs,
        runtime: RuntimeArgs,
    }

    fn advanced_boot(args: &[&str]) -> Advanced {
        match parse(args).unwrap().cmd {
            Cmd::Advanced {
                cmd:
                    AdvancedCmd::Boot {
                        kernel,
                        initramfs,
                        hardware,
                        linux,
                        firmware: _,
                        runtime,
                    },
            } => Advanced {
                kernel,
                initramfs,
                hardware,
                linux,
                runtime,
            },
            _ => panic!("expected advanced boot"),
        }
    }

    fn hardware(extra: &[&str]) -> HardwareArgs {
        let mut args = vec!["advanced", "boot", "--kernel", "k"];
        args.extend_from_slice(extra);
        advanced_boot(&args).hardware
    }

    fn help_of(path: &[&str]) -> String {
        let mut command = Cli::command();
        for name in path {
            command = command.find_subcommand(name).unwrap().clone();
        }
        command.render_long_help().to_string()
    }

    fn nic(link: Option<NetLink>) -> Nic {
        Nic {
            model: NetModel::Virtio,
            link,
        }
    }

    #[test]
    fn exit_code_is_zero_when_the_machine_ends_on_its_own() {
        assert_eq!(exit_code(&ExitReason::Reset), 0);
        assert_eq!(exit_code(&ExitReason::Shutdown), 0);
    }

    #[test]
    fn exit_code_is_zero_when_stopped_from_the_host() {
        assert_eq!(exit_code(&ExitReason::Stopped), 0);
    }

    #[test]
    fn exit_code_of_a_triple_fault_is_two() {
        assert_eq!(exit_code(&ExitReason::TripleFault), 2);
    }

    #[test]
    fn exit_code_of_an_internal_error_is_two() {
        assert_eq!(exit_code(&ExitReason::Internal(String::new())), 2);
        assert_eq!(
            exit_code(&ExitReason::Internal("vcpu thread panicked".into())),
            2
        );
        assert_eq!(
            exit_code(&ExitReason::Internal("jit divergence at 0xfff0".into())),
            2
        );
    }

    #[test]
    fn exit_codes_are_distinct_and_setup_errors_are_one() {
        assert_eq!(EXIT_SUCCESS, 0);
        assert_eq!(EXIT_ERROR, 1);
        assert_eq!(EXIT_MACHINE_FAILURE, 2);
        for reason in [
            ExitReason::Shutdown,
            ExitReason::Reset,
            ExitReason::Stopped,
            ExitReason::TripleFault,
            ExitReason::Internal("x".into()),
        ] {
            assert_ne!(exit_code(&reason), EXIT_ERROR, "{reason:?}");
        }
    }

    #[test]
    fn exit_codes_fit_in_the_process_status_byte() {
        for code in [EXIT_SUCCESS, EXIT_ERROR, EXIT_MACHINE_FAILURE] {
            assert!(u8::try_from(code).is_ok());
        }
    }

    #[test]
    fn exit_codes_are_documented_in_the_help() {
        let help = help_of(&[]);
        assert!(help.contains("Exit codes:"), "{help}");
        for code in [EXIT_SUCCESS, EXIT_ERROR, EXIT_MACHINE_FAILURE] {
            assert!(EXIT_CODES_HELP.contains(&format!("  {code}  ")), "{code}");
        }
        assert!(EXIT_CODES_HELP.contains("triple fault"));
        assert!(EXIT_CODES_HELP.contains("--timeout"));
    }

    #[test]
    fn usage_errors_exit_one_not_clap_default_two() {
        for args in [
            &["advanced", "boot"][..],
            &["quick", "create", "m"],
            &["quick", "create", "m", "--install", "ubuntu"],
            &["nope"],
            &["advanced", "boot", "--kernel", "k", "--memory", "x"],
            &[],
        ] {
            let error = parse(args).err().unwrap();
            assert_eq!(usage_exit_code(&error), EXIT_ERROR, "{args:?}");
        }
    }

    #[test]
    fn help_requests_exit_zero() {
        for args in [
            &["--help"][..],
            &["advanced", "boot", "--help"],
            &["quick", "-h"],
            &["quick", "create", "--help"],
        ] {
            let error = parse(args).err().unwrap();
            assert_eq!(usage_exit_code(&error), EXIT_SUCCESS, "{args:?}");
        }
    }

    #[test]
    fn command_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_old_top_level_commands_are_gone() {
        for args in [
            &["boot", "--kernel", "k"][..],
            &["firmware", "bios.bin"],
            &["quick", "debian"],
        ] {
            assert_eq!(error_kind(args), ErrorKind::InvalidSubcommand, "{args:?}");
        }
    }

    #[test]
    fn missing_or_unknown_subcommands_are_errors() {
        assert_eq!(
            error_kind(&[]),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(
            error_kind(&["quick"]),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(
            error_kind(&["advanced"]),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_eq!(error_kind(&["run"]), ErrorKind::InvalidSubcommand);
        assert_eq!(
            error_kind(&["advanced", "run"]),
            ErrorKind::InvalidSubcommand
        );
        assert_eq!(error_kind(&["--kernel", "k"]), ErrorKind::UnknownArgument);
    }

    #[test]
    fn help_is_available_for_every_command() {
        for args in [
            &["--help"][..],
            &["quick", "--help"],
            &["quick", "create", "--help"],
            &["quick", "edit", "--help"],
            &["quick", "boot", "--help"],
            &["quick", "reinstall", "--help"],
            &["quick", "update-artifacts", "--help"],
            &["quick", "list", "--help"],
            &["quick", "show", "--help"],
            &["quick", "delete", "--help"],
            &["advanced", "--help"],
            &["advanced", "boot", "--help"],
            &["advanced", "boot-firmware", "--help"],
        ] {
            assert_eq!(error_kind(args), ErrorKind::DisplayHelp, "{args:?}");
        }
    }

    #[test]
    fn there_is_no_version_flag() {
        assert_eq!(error_kind(&["--version"]), ErrorKind::UnknownArgument);
    }

    #[test]
    fn advanced_boot_requires_a_kernel() {
        assert_eq!(
            error_kind(&["advanced", "boot"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["advanced", "boot", "--initramfs", "i"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn advanced_boot_with_only_a_kernel_chooses_nothing() {
        let boot = advanced_boot(&["advanced", "boot", "--kernel", "bz"]);
        assert_eq!(boot.kernel, PathBuf::from("bz"));
        assert_eq!(boot.initramfs, None);
        let hardware = &boot.hardware;
        assert_eq!(hardware.memory, None);
        assert_eq!(hardware.disable_jit, None);
        assert!(hardware.disk_steps.is_empty());
        assert!(hardware.nics.is_empty());
        assert_eq!(boot.linux.cmdline, None);
        assert_eq!(boot.linux.append, None);
        assert_eq!(boot.linux.no_reboot, None);
        let runtime = &boot.runtime;
        assert_eq!(runtime.time, None);
        assert_eq!(runtime.timeout, None);
        assert!(!runtime.verify && !runtime.debug);
    }

    #[test]
    fn advanced_boot_takes_a_hypervisor_and_its_command_line() {
        let boot = advanced_boot(&[
            "advanced",
            "boot",
            "--kernel",
            "k",
            "--hypervisor",
            "xen.gz",
            "--hypervisor-cmdline",
            "console=com1 dom0_mem=512M",
        ]);
        let hypervisor = &boot.linux.hypervisor;
        assert_eq!(hypervisor.hypervisor, Some(PathBuf::from("xen.gz")));
        assert_eq!(
            hypervisor.hypervisor_cmdline.as_deref(),
            Some("console=com1 dom0_mem=512M")
        );
        assert_eq!(
            advanced_boot(&["advanced", "boot", "--kernel", "k"])
                .linux
                .hypervisor
                .hypervisor,
            None
        );
        // A hypervisor command line without a hypervisor would be silently ignored. It is refused
        // before anything is read, so the kernel need not exist.
        let Cmd::Advanced { cmd } = parse(&[
            "advanced",
            "boot",
            "--kernel",
            "missing",
            "--hypervisor-cmdline",
            "x",
        ])
        .unwrap()
        .cmd
        else {
            unreachable!()
        };
        assert_eq!(
            advanced_command(cmd).err().as_deref(),
            Some("--hypervisor-cmdline needs --hypervisor")
        );
    }

    #[test]
    fn quick_create_edit_and_boot_take_a_hypervisor_and_edit_and_boot_can_remove_it() {
        for command in ["create", "edit", "boot"] {
            let mut argv = vec!["quick", command, "m"];
            if command == "create" {
                argv.extend(["--install", "alpine"]);
            }
            argv.extend([
                "--hypervisor",
                "xen.gz",
                "--hypervisor-cmdline",
                "dom0_mem=1G",
            ]);
            let linux = match parse(&argv).unwrap().cmd {
                Cmd::Quick {
                    cmd:
                        QuickCmd::Create { linux, .. }
                        | QuickCmd::Edit { linux, .. }
                        | QuickCmd::Boot { linux, .. },
                } => linux,
                _ => unreachable!(),
            };
            assert_eq!(linux.hypervisor.hypervisor, Some(PathBuf::from("xen.gz")));
            assert_eq!(
                linux.hypervisor.hypervisor_cmdline.as_deref(),
                Some("dom0_mem=1G")
            );
        }
        for command in ["edit", "boot"] {
            let remove = match parse(&["quick", command, "m", "--remove-hypervisor"])
                .unwrap()
                .cmd
            {
                Cmd::Quick {
                    cmd: QuickCmd::Edit { remove, .. } | QuickCmd::Boot { remove, .. },
                } => remove,
                _ => unreachable!(),
            };
            assert!(remove.remove_hypervisor);
        }
        // There is nothing to remove from a machine that does not exist yet.
        assert_eq!(
            error_kind(&[
                "quick",
                "create",
                "m",
                "--install",
                "alpine",
                "--remove-hypervisor"
            ]),
            ErrorKind::UnknownArgument
        );
    }

    fn bootstrap_parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        let mut args = vec!["quick", "bootstrap", "m", "--install", "debian"];
        args.extend_from_slice(extra);
        parse(&args)
    }

    #[test]
    fn quick_bootstrap_takes_the_answers_and_quick_creates_options() {
        let cli = bootstrap_parse(&[
            "--user",
            "alex",
            "--password-file",
            "pw",
            "--root-password",
            "toor",
            "--hostname",
            "box",
            "--ssh-key",
            "a.pub",
            "--ssh-key",
            "b.pub",
            "--timezone",
            "Europe/Berlin",
            "--locale",
            "de_DE.UTF-8",
            "--keymap",
            "de",
            "--package",
            "vim",
            "--package",
            "git",
            "--memory",
            "4096",
            "--create-disk",
            "65536",
            "--firmware",
            "bios",
            "--timeout",
            "5",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Quick {
                cmd:
                    QuickCmd::Bootstrap {
                        name,
                        install,
                        hardware,
                        firmware,
                        bootstrap,
                        runtime,
                        ..
                    },
            } => {
                assert_eq!(name, "m");
                assert_eq!(install, Distro::Debian);
                assert_eq!(hardware.memory, Some(4096));
                assert_eq!(hardware.disk_steps.len(), 1);
                assert_eq!(firmware.firmware, Some(args::Firmware::Bios));
                assert_eq!(runtime.timeout, Some(5));
                assert_eq!(bootstrap.user.as_deref(), Some("alex"));
                assert_eq!(bootstrap.password_file, Some(PathBuf::from("pw")));
                assert_eq!(bootstrap.root_password.as_deref(), Some("toor"));
                assert_eq!(bootstrap.hostname.as_deref(), Some("box"));
                assert_eq!(
                    bootstrap.ssh_key,
                    vec![PathBuf::from("a.pub"), PathBuf::from("b.pub")]
                );
                assert_eq!(bootstrap.timezone.as_deref(), Some("Europe/Berlin"));
                assert_eq!(bootstrap.locale.as_deref(), Some("de_DE.UTF-8"));
                assert_eq!(bootstrap.keymap.as_deref(), Some("de"));
                assert_eq!(bootstrap.package, vec!["vim", "git"]);
            }
            _ => panic!("expected quick bootstrap"),
        }
    }

    #[test]
    fn quick_bootstrap_needs_a_user_and_a_distro() {
        assert_eq!(
            bootstrap_parse(&["--password", "x"]).err().unwrap().kind(),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["quick", "bootstrap", "m", "--user", "alex"]),
            ErrorKind::MissingRequiredArgument
        );
        assert!(bootstrap_parse(&["--user", "alex", "--ask-password"]).is_ok());
    }

    #[test]
    fn quick_bootstrap_takes_at_most_one_form_of_each_password() {
        for pair in [
            ["--password=x", "--ask-password"],
            ["--password=x", "--password-file=f"],
            ["--ask-password", "--password-file=f"],
            ["--root-password=x", "--ask-root-password"],
            ["--root-password=x", "--root-password-file=f"],
            ["--ask-root-password", "--root-password-file=f"],
        ] {
            let mut args = vec!["--user", "alex"];
            args.extend(pair);
            assert_eq!(
                bootstrap_parse(&args).err().map(|e| e.kind()),
                Some(ErrorKind::ArgumentConflict),
                "{pair:?}"
            );
        }
        assert!(
            bootstrap_parse(&["--user", "alex", "--ask-password", "--ask-root-password"]).is_ok()
        );
    }

    #[test]
    fn after_installer_boots_the_disk_only_after_a_finished_install() {
        assert_eq!(
            after_installer(&ExitReason::Reset, true),
            AfterInstaller::BootDisk
        );
        assert_eq!(
            after_installer(&ExitReason::Reset, false),
            AfterInstaller::Failed
        );
        // Ctrl-], --timeout and --suspend-to end the run as they would any other; only the
        // marker decides whether the machine moves to its disk for the next boot.
        let others = [
            ExitReason::Shutdown,
            ExitReason::Stopped,
            ExitReason::Suspended,
            ExitReason::SuspendFailed("x".into()),
            ExitReason::ResumeFailed("x".into()),
            ExitReason::TripleFault,
            ExitReason::Internal("x".into()),
        ];
        for reason in &others {
            assert_eq!(
                after_installer(reason, false),
                AfterInstaller::Done,
                "{reason:?}"
            );
            assert_eq!(
                after_installer(reason, true),
                AfterInstaller::Installed,
                "{reason:?}"
            );
        }
    }

    #[test]
    fn bootstrap_refuses_firmware_none_hypervisor_and_resume_before_creating() {
        let home = crate::test_support::TempDir::new("main-bootstrap-refused");
        let home_text = home.path().to_str().unwrap().to_string();
        let _env = crate::test_support::EnvGuard::set(&[("UEMULATOR_HOME", Some(&home_text))]);
        let image = home.path().join("precious.img");
        std::fs::write(&image, vec![1u8; 4096]).unwrap();
        let image = image.to_str().unwrap();
        for extra in [
            &["--firmware", "none"][..],
            &["--hypervisor", "xen.gz"][..],
            &["--resume-from", "old.snapshot"][..],
            &["--disk", image][..],
            &["--disk", image, "--create-disk", "1024"][..],
            &["--display", "framebuffer"][..],
        ] {
            let mut args = vec!["--user", "alex", "--password", "pw"];
            args.extend_from_slice(extra);
            let error = match execute(bootstrap_parse(&args).unwrap()) {
                Err(error) => error,
                Ok(_) => panic!("{extra:?} was not refused"),
            };
            assert!(error.starts_with(extra[0]), "{error}");
            assert!(
                !home.path().join("machines/m").exists(),
                "{extra:?} left a machine behind"
            );
        }
    }

    /// The name is checked before the password: a bad one must not cost the user a prompt, and
    /// `machines::exists` must not look outside the machines directory. A missing password
    /// file stands in for the prompt (tests may run on a terminal).
    #[test]
    fn bootstrap_checks_the_name_before_the_password() {
        let home = crate::test_support::TempDir::new("main-bootstrap-bad-name");
        let home_text = home.path().to_str().unwrap().to_string();
        let _env = crate::test_support::EnvGuard::set(&[("UEMULATOR_HOME", Some(&home_text))]);
        for name in ["../x", ".hidden", "a/b"] {
            let cli = parse(&[
                "quick",
                "bootstrap",
                name,
                "--install",
                "debian",
                "--user",
                "alex",
                "--password-file",
                "/nonexistent/password",
            ])
            .unwrap();
            let error = match execute(cli) {
                Err(error) => error,
                Ok(_) => panic!("{name} was bootstrapped"),
            };
            assert!(error.starts_with("invalid machine name"), "{name}: {error}");
        }
    }

    /// The installed disk boots only on a reset, but `--display both` is fine and
    /// `--create-disk` before `--disk` puts the attached image second, where it is not erased.
    #[test]
    fn bootstrap_takes_a_screen_and_an_attached_second_disk() {
        let image = std::path::Path::new("/images/data.img");
        let cli = bootstrap_parse(&[
            "--user",
            "alex",
            "--ask-password",
            "--create-disk",
            "1024",
            "--disk",
            image.to_str().unwrap(),
            "--display",
            "both",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Quick {
                cmd: QuickCmd::Bootstrap { hardware, .. },
            } => {
                assert!(bootstrap::refuse_attached_first_disk(&hardware.disk_steps).is_ok());
                assert!(bootstrap::refuse_framebuffer_display(hardware.display).is_ok());
            }
            _ => unreachable!(),
        }
    }

    /// Spec, "Left to the builder": an existing name suggests `reinstall`, before any password
    /// is asked for.
    #[test]
    fn bootstrap_of_an_existing_machine_suggests_reinstall() {
        let home = crate::test_support::TempDir::new("main-bootstrap-exists");
        let home_text = home.path().to_str().unwrap().to_string();
        let _env = crate::test_support::EnvGuard::set(&[("UEMULATOR_HOME", Some(&home_text))]);
        std::fs::create_dir_all(home.path().join("machines/m")).unwrap();
        let error = match execute(bootstrap_parse(&["--user", "alex", "--ask-password"]).unwrap()) {
            Err(error) => error,
            Ok(_) => panic!("an existing machine was bootstrapped"),
        };
        assert!(error.contains("uemulator quick reinstall m"), "{error}");
    }

    #[test]
    fn quick_bootstrap_refuses_firmware_none_and_a_hypervisor() {
        let refusal = |extra: &[&str]| {
            let mut args = vec!["--user", "alex", "--ask-password"];
            args.extend_from_slice(extra);
            match bootstrap_parse(&args).unwrap().cmd {
                Cmd::Quick {
                    cmd:
                        QuickCmd::Bootstrap {
                            firmware, linux, ..
                        },
                } => bootstrap::refuse_unbootable(firmware.firmware, &linux.hypervisor),
                _ => unreachable!(),
            }
        };
        assert!(refusal(&[]).is_ok());
        assert!(refusal(&["--firmware", "bios"]).is_ok());
        let error = refusal(&["--firmware", "none"]).unwrap_err();
        assert!(error.starts_with("--firmware none"), "{error}");
        let error = refusal(&["--hypervisor", "xen.gz"]).unwrap_err();
        assert!(error.starts_with("--hypervisor"), "{error}");
    }

    #[test]
    fn advanced_boot_takes_an_initramfs() {
        let boot = advanced_boot(&["advanced", "boot", "--kernel", "k", "--initramfs", "rd.gz"]);
        assert_eq!(boot.initramfs, Some(PathBuf::from("rd.gz")));
    }

    #[test]
    fn advanced_boot_accepts_every_option_together() {
        let boot = advanced_boot(&[
            "advanced",
            "boot",
            "--kernel",
            "k",
            "-m",
            "256",
            "--cmdline",
            "console=ttyS0 root=/dev/vda",
            "--disk",
            "d.img,ro",
            "--network",
            "virtio",
            "--network-link-user",
            "--append",
            "quiet",
            "--console",
            "virtio",
            "--disable-jit",
            "--no-reboot",
            "--verify",
            "--debug",
            "--time",
            "1700000000",
            "--timeout",
            "30",
        ]);
        let (hardware, linux, runtime) = (&boot.hardware, &boot.linux, &boot.runtime);
        assert_eq!(hardware.memory, Some(256));
        assert_eq!(
            linux.cmdline.as_deref(),
            Some("console=ttyS0 root=/dev/vda")
        );
        assert_eq!(
            hardware.disk_steps,
            vec![DiskStep::Attach(parse_disk_spec("d.img,ro"))]
        );
        assert_eq!(hardware.nics, vec![nic(Some(NetLink::User))]);
        assert_eq!(linux.append.as_deref(), Some("quiet"));
        assert_eq!(hardware.disable_jit, Some(true));
        assert_eq!(linux.no_reboot, Some(true));
        assert_eq!(runtime.time, Some(1_700_000_000));
        assert_eq!(runtime.timeout, Some(30));
        assert!(runtime.verify && runtime.debug);
    }

    #[test]
    fn boot_firmware_takes_an_image_and_the_hardware_options() {
        match parse(&[
            "advanced",
            "boot-firmware",
            "--firmware",
            "bios.bin",
            "--memory",
            "64",
            "--disable-jit",
            "--timeout",
            "3",
        ])
        .unwrap()
        .cmd
        {
            Cmd::Advanced {
                cmd:
                    AdvancedCmd::BootFirmware {
                        firmware,
                        hardware,
                        runtime,
                    },
            } => {
                assert_eq!(firmware, PathBuf::from("bios.bin"));
                assert_eq!(hardware.memory, Some(64));
                assert_eq!(hardware.disable_jit, Some(true));
                assert_eq!(runtime.timeout, Some(3));
            }
            _ => panic!("expected boot-firmware"),
        }
        assert_eq!(
            error_kind(&["advanced", "boot-firmware"]),
            ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn firmware_boots_have_no_kernel_command_line_options() {
        for flag in ["--cmdline", "--append"] {
            assert_eq!(
                error_kind(&["advanced", "boot-firmware", "--firmware", "f", flag, "x"]),
                ErrorKind::UnknownArgument,
                "{flag}"
            );
        }
        assert_eq!(
            error_kind(&[
                "advanced",
                "boot-firmware",
                "--firmware",
                "f",
                "--no-reboot"
            ]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn memory_has_a_short_and_a_long_flag() {
        assert_eq!(hardware(&["-m", "128"]).memory, Some(128));
        assert_eq!(hardware(&["--memory", "128"]).memory, Some(128));
        assert_eq!(hardware(&["--memory=128"]).memory, Some(128));
    }

    #[test]
    fn memory_must_be_a_non_negative_integer() {
        for bad in ["abc", "-1", "1.5", "", "0x40"] {
            let kind = error_kind(&["advanced", "boot", "--kernel", "k", "--memory", bad]);
            assert!(
                matches!(
                    kind,
                    ErrorKind::ValueValidation | ErrorKind::UnknownArgument
                ),
                "{bad:?}: {kind:?}"
            );
        }
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--memory"]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn memory_range_is_enforced_by_the_library_not_the_parser() {
        assert_eq!(hardware(&["--memory", "0"]).memory, Some(0));
        assert_eq!(hardware(&["--memory", "99999"]).memory, Some(99999));
    }

    #[test]
    fn console_accepts_only_serial_and_virtio_and_repeats() {
        assert_eq!(
            hardware(&["--console", "serial", "--console", "virtio"]).console,
            vec![ConsoleArg::Serial, ConsoleArg::Virtio]
        );
        for bad in ["hvc0", "Serial", "VIRTIO", ""] {
            assert_eq!(
                error_kind(&["advanced", "boot", "--kernel", "k", "--console", bad]),
                ErrorKind::InvalidValue,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn time_and_timeout_must_be_unsigned_integers() {
        let runtime = |extra: &[&str]| {
            let mut args = vec!["advanced", "boot", "--kernel", "k"];
            args.extend_from_slice(extra);
            advanced_boot(&args).runtime
        };
        assert_eq!(runtime(&["--time", "0"]).time, Some(0));
        assert_eq!(runtime(&["--timeout", "0"]).timeout, Some(0));
        assert_eq!(
            runtime(&["--time", "18446744073709551615"]).time,
            Some(u64::MAX)
        );
        for flag in ["--time", "--timeout"] {
            for bad in ["soon", "1.5", "99999999999999999999999"] {
                assert_eq!(
                    error_kind(&["advanced", "boot", "--kernel", "k", flag, bad]),
                    ErrorKind::ValueValidation,
                    "{flag} {bad}"
                );
            }
        }
    }

    #[test]
    fn smp_mode_is_unset_by_default_and_takes_the_three_modes() {
        use crate::args::SmpArg;
        assert_eq!(hardware(&[]).smp, None);
        for (name, mode) in [
            ("auto", SmpArg::Auto),
            ("timeshare", SmpArg::Timeshare),
            ("threads", SmpArg::Threads),
        ] {
            assert_eq!(hardware(&["--smp", name]).smp, Some(mode));
        }
        assert!(matches!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--smp", "fast"]),
            clap::error::ErrorKind::InvalidValue
        ));
    }

    #[test]
    fn cpus_is_a_count_that_is_unset_by_default() {
        assert_eq!(hardware(&[]).cpus, None);
        assert_eq!(hardware(&["--cpus", "4"]).cpus, Some(4));
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--cpus", "x"]),
            ErrorKind::ValueValidation
        );
    }

    #[test]
    fn smt_is_unset_by_default_and_every_machine_command_takes_it() {
        assert_eq!(hardware(&[]).smt, None);
        assert_eq!(hardware(&["--smt"]).smt, Some(true));
        assert_eq!(hardware(&["--smt=false"]).smt, Some(false));
        for command in ["create", "edit", "boot"] {
            let mut argv = vec!["quick", command, "m", "--smt=false"];
            if command == "create" {
                argv.extend(["--install", "alpine"]);
            }
            let hardware = match parse(&argv).unwrap().cmd {
                Cmd::Quick {
                    cmd:
                        QuickCmd::Create { hardware, .. }
                        | QuickCmd::Edit { hardware, .. }
                        | QuickCmd::Boot { hardware, .. },
                } => hardware,
                _ => unreachable!(),
            };
            assert_eq!(hardware.smt, Some(false), "quick {command}");
        }
        let mut settings = Settings::default();
        settings.add(
            &hardware(&["--smt=false"]),
            &LinuxArgs::default(),
            Vec::new(),
        );
        assert_eq!(settings.smt, Some(false));
        settings.add(&hardware(&[]), &LinuxArgs::default(), Vec::new());
        assert_eq!(settings.smt, Some(false), "kept when not given");
    }

    #[test]
    fn boolean_options_take_an_optional_value_only_after_an_equals_sign() {
        assert_eq!(hardware(&[]).disable_jit, None);
        assert_eq!(hardware(&["--disable-jit"]).disable_jit, Some(true));
        assert_eq!(hardware(&["--disable-jit=true"]).disable_jit, Some(true));
        assert_eq!(hardware(&["--disable-jit=false"]).disable_jit, Some(false));
        assert_eq!(hardware(&[]).disable_rng, None);
        assert_eq!(hardware(&["--disable-rng"]).disable_rng, Some(true));
        assert_eq!(hardware(&["--disable-rng=false"]).disable_rng, Some(false));
        assert_eq!(hardware(&[]).iommu, None);
        assert_eq!(hardware(&["--iommu"]).iommu, Some(true));
        assert_eq!(hardware(&["--iommu=false"]).iommu, Some(false));
        assert_eq!(hardware(&[]).tpm, None);
        assert_eq!(hardware(&["--tpm"]).tpm, Some(true));
        assert_eq!(hardware(&["--tpm=false"]).tpm, Some(false));
        assert_eq!(hardware(&[]).secure_boot, None);
        assert_eq!(hardware(&["--secure-boot"]).secure_boot, Some(true));
        assert_eq!(hardware(&["--secure-boot=false"]).secure_boot, Some(false));
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--disable-jit=maybe"]),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            error_kind(&[
                "advanced",
                "boot",
                "--kernel",
                "k",
                "--disable-jit",
                "false"
            ]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn unknown_flags_and_stray_positionals_are_rejected() {
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--nope"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "extra"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn the_old_network_flags_are_gone() {
        for flag in ["--net-user", "--net-loop", "--net-udp"] {
            assert_eq!(
                error_kind(&["advanced", "boot", "--kernel", "k", flag]),
                ErrorKind::UnknownArgument,
                "{flag}"
            );
        }
    }

    #[test]
    fn network_links_bind_the_preceding_network_in_command_line_order() {
        let nics = hardware(&[
            "--network",
            "virtio",
            "--network-link-udp",
            "a,b",
            "--network",
            "virtio",
            "--network",
            "virtio",
            "--network-link-loop",
            "--network",
            "virtio",
            "--network-link-user",
        ])
        .nics;
        assert_eq!(
            nics,
            vec![
                nic(Some(NetLink::Udp("a,b".into()))),
                nic(None),
                nic(Some(NetLink::Loop)),
                nic(Some(NetLink::User)),
            ]
        );
    }

    #[test]
    fn a_link_without_a_network_before_it_is_an_error() {
        for link in [
            &["--network-link-user"][..],
            &["--network-link-loop"],
            &["--network-link-udp", "a,b"],
        ] {
            let mut args = vec!["advanced", "boot", "--kernel", "k"];
            args.extend_from_slice(link);
            args.extend_from_slice(&["--network", "virtio"]);
            let error = parse(&args).err().unwrap();
            assert_eq!(error.kind(), ErrorKind::ArgumentConflict, "{link:?}");
            assert!(
                error.to_string().contains("must follow the --network"),
                "{error}"
            );
            assert_eq!(usage_exit_code(&error), EXIT_ERROR);
        }
    }

    #[test]
    fn a_network_takes_one_link() {
        let error = parse(&[
            "advanced",
            "boot",
            "--kernel",
            "k",
            "--network",
            "virtio",
            "--network-link-user",
            "--network-link-loop",
        ])
        .err()
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
        assert!(error.to_string().contains("already has a link"), "{error}");
    }

    #[test]
    fn network_model_must_be_virtio() {
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--network", "e1000"]),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            error_kind(&["advanced", "boot", "--kernel", "k", "--network"]),
            ErrorKind::InvalidValue
        );
        assert_eq!(
            error_kind(&[
                "advanced",
                "boot",
                "--kernel",
                "k",
                "--network",
                "virtio",
                "--network-link-user=yes"
            ]),
            ErrorKind::InvalidValue
        );
    }

    #[test]
    fn disks_are_listed_in_the_order_they_were_typed() {
        let steps = hardware(&["--disk", "a.img", "--disk", "b.img,ro"]).disk_steps;
        assert_eq!(
            steps,
            vec![
                DiskStep::Attach(DiskSpec {
                    path: "a.img".into(),
                    read_only: false
                }),
                DiskStep::Attach(DiskSpec {
                    path: "b.img".into(),
                    read_only: true
                }),
            ]
        );
    }

    fn create_hardware(extra: &[&str]) -> HardwareArgs {
        let mut args = vec!["quick", "create", "m", "--install", "alpine"];
        args.extend_from_slice(extra);
        match parse(&args).unwrap().cmd {
            Cmd::Quick {
                cmd: QuickCmd::Create { hardware, .. },
            } => hardware,
            _ => panic!("expected quick create"),
        }
    }

    #[test]
    fn disk_and_create_disk_interleave_in_command_line_order() {
        let steps = create_hardware(&[
            "--create-disk",
            "100",
            "--disk",
            "a.img",
            "--create-disk",
            "200",
            "--disk",
            "b.img,ro",
        ])
        .disk_steps;
        assert_eq!(
            steps,
            vec![
                DiskStep::Create { mib: 100 },
                DiskStep::Attach(parse_disk_spec("a.img")),
                DiskStep::Create { mib: 200 },
                DiskStep::Attach(parse_disk_spec("b.img,ro")),
            ]
        );
    }

    #[test]
    fn create_disk_takes_a_positive_number_of_mebibytes() {
        for bad in ["0", "-1", "big", "1.5"] {
            let kind = error_kind(&[
                "quick",
                "create",
                "m",
                "--install",
                "alpine",
                "--create-disk",
                bad,
            ]);
            assert!(
                matches!(
                    kind,
                    ErrorKind::ValueValidation | ErrorKind::UnknownArgument
                ),
                "{bad}: {kind:?}"
            );
        }
    }

    #[test]
    fn quick_create_requires_a_name_and_a_known_distro() {
        assert_eq!(
            error_kind(&["quick", "create", "--install", "alpine"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["quick", "create", "m"]),
            ErrorKind::MissingRequiredArgument
        );
        for bad in ["ubuntu", "Debian", "DEBIAN", "arch", ""] {
            assert_eq!(
                error_kind(&["quick", "create", "m", "--install", bad]),
                ErrorKind::InvalidValue,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn quick_create_accepts_each_distro_and_the_sketched_options() {
        for (name, want) in [
            ("debian", Distro::Debian),
            ("alpine", Distro::Alpine),
            ("fedora", Distro::Fedora),
        ] {
            match parse(&["quick", "create", "m", "--install", name])
                .unwrap()
                .cmd
            {
                Cmd::Quick {
                    cmd:
                        QuickCmd::Create {
                            install,
                            boot,
                            hardware,
                            ..
                        },
                } => {
                    assert_eq!(install, want, "{name}");
                    assert!(!boot);
                    assert_eq!(hardware.memory, None);
                }
                _ => panic!("expected quick create"),
            }
        }
        match parse(&[
            "quick",
            "create",
            "test",
            "--install",
            "alpine",
            "--memory",
            "4096",
            "--create-disk",
            "16384",
            "--console",
            "virtio",
            "--network",
            "virtio",
            "--network-link-user",
            "--boot",
        ])
        .unwrap()
        .cmd
        {
            Cmd::Quick {
                cmd:
                    QuickCmd::Create {
                        name,
                        boot,
                        hardware,
                        ..
                    },
            } => {
                assert_eq!(name, "test");
                assert!(boot);
                assert_eq!(hardware.memory, Some(4096));
                assert_eq!(hardware.disk_steps, vec![DiskStep::Create { mib: 16384 }]);
                assert_eq!(hardware.nics, vec![nic(Some(NetLink::User))]);
            }
            _ => panic!("expected quick create"),
        }
    }

    #[test]
    fn each_quick_command_takes_the_options_that_mean_something_to_it() {
        // (command, option, accepted)
        let cases: &[(&[&str], &str, bool)] = &[
            (
                &["quick", "create", "m", "--install", "alpine"],
                "--create-disk=1",
                true,
            ),
            (
                &["quick", "create", "m", "--install", "alpine"],
                "--boot",
                true,
            ),
            (
                &["quick", "create", "m", "--install", "alpine"],
                "--remove-disk=0",
                false,
            ),
            (
                &["quick", "create", "m", "--install", "alpine"],
                "--timeout=1",
                false,
            ),
            (
                &["quick", "create", "m", "--install", "alpine"],
                "--debug",
                false,
            ),
            (&["quick", "edit", "m"], "--create-disk=1", true),
            (&["quick", "edit", "m"], "--remove-disk=0", true),
            (&["quick", "edit", "m"], "--remove-network=0", true),
            (&["quick", "edit", "m"], "--remove-console=0", true),
            (&["quick", "edit", "m"], "--install=debian", false),
            (&["quick", "edit", "m"], "--boot", false),
            (&["quick", "edit", "m"], "--verify", false),
            (&["quick", "boot", "m"], "--remove-disk=0", true),
            (&["quick", "boot", "m"], "--timeout=1", true),
            (&["quick", "boot", "m"], "--verify", true),
            (&["quick", "boot", "m"], "--memory=8192", true),
            (&["quick", "boot", "m"], "--create-disk=1", false),
            (&["quick", "boot", "m"], "--install=debian", false),
            (&["quick", "reinstall", "m"], "--install=debian", true),
            (&["quick", "reinstall", "m"], "--memory=1", false),
            (
                &["quick", "update-artifacts", "m"],
                "--install=debian",
                false,
            ),
            (
                &["advanced", "boot", "--kernel", "k"],
                "--remove-disk=0",
                false,
            ),
            (
                &["advanced", "boot", "--kernel", "k"],
                "--create-disk=1",
                false,
            ),
            (
                &["advanced", "boot", "--kernel", "k"],
                "--install=alpine",
                false,
            ),
        ];
        for (command, option, accepted) in cases {
            let mut args = command.to_vec();
            args.push(option);
            assert_eq!(parse(&args).is_ok(), *accepted, "{args:?}");
        }
    }

    #[test]
    fn quick_boot_overrides_parse_like_the_saved_options() {
        match parse(&[
            "quick",
            "boot",
            "test",
            "--memory",
            "8192",
            "--timeout",
            "5",
        ])
        .unwrap()
        .cmd
        {
            Cmd::Quick {
                cmd:
                    QuickCmd::Boot {
                        name,
                        hardware,
                        runtime,
                        ..
                    },
            } => {
                assert_eq!(name, "test");
                assert_eq!(hardware.memory, Some(8192));
                assert_eq!(runtime.timeout, Some(5));
            }
            _ => panic!("expected quick boot"),
        }
    }

    #[test]
    fn firmware_is_on_create_and_edit_only() {
        use crate::args::Firmware;
        let from = |args: &[&str]| match parse(args).unwrap().cmd {
            Cmd::Quick {
                cmd: QuickCmd::Create { firmware, .. } | QuickCmd::Edit { firmware, .. },
            } => firmware.firmware,
            _ => panic!("no firmware"),
        };
        assert_eq!(from(&["quick", "edit", "m"]), None);
        for (value, want) in [
            ("bios", Firmware::Bios),
            ("uefi", Firmware::Uefi),
            ("none", Firmware::None),
        ] {
            assert_eq!(
                from(&["quick", "edit", "m", "--firmware", value]),
                Some(want)
            );
            assert_eq!(
                from(&[
                    "quick",
                    "create",
                    "m",
                    "--install",
                    "alpine",
                    "--firmware",
                    value
                ]),
                Some(want)
            );
        }
        assert!(parse(&["quick", "edit", "m", "--firmware", "coreboot"]).is_err());
        assert!(parse(&["quick", "boot", "m", "--firmware", "uefi"]).is_err());
    }

    #[test]
    fn advanced_boot_takes_a_firmware_and_defaults_to_uefi() {
        use crate::args::Firmware;
        let from = |extra: &[&str]| {
            let mut args = vec!["advanced", "boot", "--kernel", "k"];
            args.extend_from_slice(extra);
            match parse(&args).unwrap().cmd {
                Cmd::Advanced {
                    cmd: AdvancedCmd::Boot { firmware, .. },
                } => firmware.firmware,
                _ => panic!("expected advanced boot"),
            }
        };
        assert_eq!(from(&[]), None);
        assert_eq!(from(&["--firmware", "none"]), Some(Firmware::None));
        assert_eq!(from(&["--firmware", "uefi"]), Some(Firmware::Uefi));
        assert!(Firmware::default() == Firmware::Uefi);
    }

    #[test]
    fn boot_from_is_on_create_edit_and_boot_only() {
        use crate::args::BootSource;
        let from = |args: &[&str]| match parse(args).unwrap().cmd {
            Cmd::Quick {
                cmd:
                    QuickCmd::Create { source, .. }
                    | QuickCmd::Edit { source, .. }
                    | QuickCmd::Boot { source, .. },
            } => source.boot_from,
            _ => panic!("no boot source"),
        };
        assert_eq!(from(&["quick", "edit", "m"]), None);
        for (value, want) in [
            ("auto", BootSource::Auto),
            ("installer", BootSource::Installer),
            ("disk", BootSource::Disk),
        ] {
            assert_eq!(
                from(&["quick", "edit", "m", "--boot-from", value]),
                Some(want)
            );
            assert_eq!(
                from(&["quick", "boot", "m", "--boot-from", value]),
                Some(want)
            );
            assert_eq!(
                from(&[
                    "quick",
                    "create",
                    "m",
                    "--install",
                    "alpine",
                    "--boot-from",
                    value
                ]),
                Some(want)
            );
        }
        assert_eq!(
            error_kind(&["quick", "boot", "m", "--boot-from", "network"]),
            ErrorKind::InvalidValue
        );
        for args in [
            &["quick", "reinstall", "m", "--boot-from", "disk"][..],
            &["advanced", "boot", "--kernel", "k", "--boot-from", "disk"],
        ] {
            assert_eq!(error_kind(args), ErrorKind::UnknownArgument, "{args:?}");
        }
    }

    #[test]
    fn boot_from_help_explains_each_source() {
        let help = help_of(&["quick", "boot"]);
        for text in ["auto", "installer", "disk", "boot order of real"] {
            assert!(help.contains(text), "{text}: {help}");
        }
    }

    #[test]
    fn list_show_and_delete_parse() {
        assert!(matches!(
            parse(&["quick", "list"]).unwrap().cmd,
            Cmd::Quick {
                cmd: QuickCmd::List
            }
        ));
        assert!(matches!(
            parse(&["quick", "show", "m"]).unwrap().cmd,
            Cmd::Quick { cmd: QuickCmd::Show { name } } if name == "m"
        ));
        for (args, want) in [
            (&["quick", "delete", "m"][..], false),
            (&["quick", "delete", "m", "--yes"], true),
        ] {
            match parse(args).unwrap().cmd {
                Cmd::Quick {
                    cmd: QuickCmd::Delete { name, yes },
                } => assert_eq!((name.as_str(), yes), ("m", want)),
                _ => panic!("expected quick delete"),
            }
        }
        assert_eq!(
            error_kind(&["quick", "show"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["quick", "delete"]),
            ErrorKind::MissingRequiredArgument
        );
        assert_eq!(
            error_kind(&["quick", "list", "m"]),
            ErrorKind::UnknownArgument
        );
        assert_eq!(
            error_kind(&["quick", "show", "m", "--memory", "1"]),
            ErrorKind::UnknownArgument
        );
    }

    #[test]
    fn delete_without_yes_deletes_nothing_and_says_what_it_would_remove() {
        let home = crate::test_support::TempDir::new("main-delete");
        let _env = crate::test_support::EnvGuard::set(&[(
            "UEMULATOR_HOME",
            Some(&home.path().to_string_lossy()),
        )]);
        let (hardware, linux) = (HardwareArgs::default(), LinuxArgs::default());
        let machine = machines::create(
            home.path(),
            "m",
            Distro::Alpine,
            &hardware,
            &linux,
            None,
            None,
            None,
        )
        .unwrap();
        let error = quick_command(QuickCmd::Delete {
            name: "m".into(),
            yes: false,
        })
        .err()
        .unwrap();
        assert!(
            error.contains("--yes") && error.contains("1 disk image"),
            "{error}"
        );
        assert!(machine.dir.exists());
        quick_command(QuickCmd::Delete {
            name: "m".into(),
            yes: true,
        })
        .unwrap();
        assert!(!machine.dir.exists());
    }

    #[test]
    fn quick_edit_removals_take_zero_based_indices() {
        match parse(&[
            "quick",
            "edit",
            "m",
            "--remove-disk",
            "1",
            "--remove-disk",
            "0",
            "--remove-network",
            "2",
        ])
        .unwrap()
        .cmd
        {
            Cmd::Quick {
                cmd: QuickCmd::Edit { remove, .. },
            } => {
                assert_eq!(remove.remove_disk, vec![1, 0]);
                assert_eq!(remove.remove_network, vec![2]);
                assert!(remove.remove_console.is_empty());
            }
            _ => panic!("expected quick edit"),
        }
        assert!(parse(&["quick", "edit", "m", "--remove-disk", "-1"]).is_err());
        assert!(parse(&["quick", "edit", "m", "--remove-disk", "first"]).is_err());
    }

    #[test]
    fn distro_names_match_the_command_line_names() {
        use clap::ValueEnum;
        for distro in Distro::value_variants() {
            let value = distro.to_possible_value().unwrap();
            assert_eq!(value.get_name(), distro.name());
        }
    }

    #[test]
    fn loopback_link_help_describes_only_what_the_gateway_answers() {
        let help = help_of(&["advanced", "boot"]);
        assert!(help.contains("ARP"), "{help}");
        assert!(help.contains("ICMP echo"), "{help}");
        assert!(help.contains("10.0.2.2"), "{help}");
        assert!(!help.contains("DHCP, ARP"), "{help}");
        assert!(help.contains("no DHCP"), "{help}");
    }

    #[test]
    fn memory_help_states_the_range_the_library_enforces() {
        use crate::run::DEFAULT_MEMORY_MIB;
        let help = help_of(&["advanced", "boot"]);
        assert!(
            help.contains(&format!("64 to {}", uemulator::MAX_MEMORY_MIB)),
            "{help}"
        );
        assert!(
            help.contains(&format!("default: {DEFAULT_MEMORY_MIB}")),
            "{help}"
        );
        assert!(
            help.contains(&format!(
                "{} for a fedora install",
                Distro::Fedora.default_memory_mib()
            )),
            "{help}"
        );
    }

    #[test]
    fn memory_help_range_matches_the_library_error() {
        // The library does not export its limits, so read them from the rejection it produces.
        let rejected =
            uemulator::EmulatedMachine::new(uemulator::MachineConfig::new().memory_mib(0));
        let message = match rejected {
            Ok(_) => panic!("0 MiB must be rejected"),
            Err(e) => e.to_string(),
        };
        let numbers: Vec<usize> = message
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|word| word.parse().ok())
            .collect();
        assert_eq!(numbers.len(), 2, "{message}");
        assert!(
            help_of(&["advanced", "boot"]).contains(&format!("{} to {}", numbers[0], numbers[1])),
            "{message}"
        );
    }

    #[test]
    fn every_option_a_command_takes_is_documented() {
        for path in [
            &["quick", "create"][..],
            &["quick", "bootstrap"],
            &["quick", "edit"],
            &["quick", "boot"],
            &["quick", "reinstall"],
            &["quick", "show"],
            &["quick", "delete"],
            &["advanced", "boot"],
            &["advanced", "boot-firmware"],
        ] {
            let mut command = Cli::command();
            for name in path {
                command = command.find_subcommand(name).unwrap().clone();
            }
            for arg in command.get_arguments() {
                assert!(
                    arg.get_help().is_some() || arg.get_id() == "help",
                    "{path:?}: {} has no help",
                    arg.get_id()
                );
            }
        }
    }

    #[test]
    fn bootstrap_timeout_help_says_it_applies_twice() {
        for path in [&["quick", "bootstrap"][..], &["quick", "reinstall"]] {
            let help = help_of(path);
            assert!(help.contains("once for the installer"), "{path:?}: {help}");
        }
        assert!(!help_of(&["quick", "boot"]).contains("once for the installer"));
    }

    #[test]
    fn edit_help_explains_how_removals_and_additions_combine() {
        let help = help_of(&["quick", "edit"]);
        assert!(help.contains("Removals apply first"), "{help}");
        assert!(help.contains("--remove-disk"), "{help}");
        assert!(help.contains("--create-disk"), "{help}");
        assert!(help.contains("left where it is"), "{help}");
    }

    #[test]
    fn settings_of_applies_only_what_the_command_line_says() {
        let boot = advanced_boot(&[
            "advanced",
            "boot",
            "--kernel",
            "k",
            "--memory",
            "300",
            "--disk",
            "x.img",
            "--network",
            "virtio",
            "--append",
            "quiet",
        ]);
        let settings = settings_of(&boot.hardware, &boot.linux).unwrap();
        assert_eq!(settings.memory, Some(300));
        assert_eq!(settings.disks.len(), 1);
        assert_eq!(settings.networks, vec![nic(None)]);
        assert_eq!(settings.append.as_deref(), Some("quiet"));
        assert!(settings.consoles.is_empty());
    }

    #[test]
    fn read_error_names_the_path() {
        let missing = PathBuf::from("/nonexistent-uemulator-dir/kernel");
        let err = read(&missing).unwrap_err();
        assert!(
            err.starts_with("read /nonexistent-uemulator-dir/kernel: "),
            "{err}"
        );
    }

    #[test]
    fn read_returns_the_file_contents() {
        let dir = crate::test_support::TempDir::new("main-read");
        let path = dir.path().join("blob");
        std::fs::write(&path, [1u8, 2, 3, 0, 255]).unwrap();
        assert_eq!(read(&path).unwrap(), vec![1, 2, 3, 0, 255]);
    }

    fn reinstall_parse(extra: &[&str]) -> Result<Cli, clap::Error> {
        let mut args = vec!["quick", "reinstall", "m"];
        args.extend_from_slice(extra);
        parse(&args)
    }

    #[test]
    fn quick_reinstall_takes_the_answers_and_yes() {
        match reinstall_parse(&["--user", "alex", "--password-file", "f", "--yes"])
            .unwrap()
            .cmd
        {
            Cmd::Quick {
                cmd:
                    QuickCmd::Reinstall {
                        name,
                        install,
                        bootstrap,
                        yes,
                        no_root_password,
                        ..
                    },
            } => {
                assert_eq!(name, "m");
                assert_eq!(install, None);
                assert_eq!(bootstrap.user.as_deref(), Some("alex"));
                assert_eq!(bootstrap.password_file, Some(PathBuf::from("f")));
                assert!(yes);
                assert!(!no_root_password);
            }
            _ => panic!("expected quick reinstall"),
        }
        // A reinstall repeats saved answers, so --user is optional there.
        assert!(reinstall_parse(&[]).is_ok());
    }

    #[test]
    fn yes_is_for_reinstall_only() {
        assert_eq!(
            bootstrap_parse(&["--user", "alex", "--ask-password", "--yes"])
                .err()
                .map(|e| e.kind()),
            Some(ErrorKind::UnknownArgument)
        );
    }

    #[test]
    fn no_root_password_conflicts_with_giving_one() {
        for given in [
            "--root-password=x",
            "--ask-root-password",
            "--root-password-file=f",
        ] {
            assert_eq!(
                reinstall_parse(&["--no-root-password", given])
                    .err()
                    .map(|e| e.kind()),
                Some(ErrorKind::ArgumentConflict),
                "{given}"
            );
        }
        assert_eq!(
            bootstrap_parse(&["--user", "alex", "--ask-password", "--no-root-password"])
                .err()
                .map(|e| e.kind()),
            Some(ErrorKind::UnknownArgument)
        );
    }

    fn reinstall_home(
        label: &str,
        record: Option<BootstrapRecord>,
    ) -> crate::test_support::TempDir {
        let home = crate::test_support::TempDir::new(label);
        let machine = machines::create(
            home.path(),
            "m",
            Distro::Alpine,
            &HardwareArgs::default(),
            &LinuxArgs::default(),
            None,
            None,
            record,
        )
        .unwrap();
        machines::fetch_artifacts(&machine, fake_fetch).unwrap();
        home
    }

    fn fake_fetch(url: &str, destination: &std::path::Path) -> Result<(), String> {
        std::fs::write(destination, url).map_err(|e| e.to_string())
    }

    fn saved_record() -> BootstrapRecord {
        BootstrapRecord {
            user: "alex".into(),
            hostname: "box".into(),
            ssh_keys: Vec::new(),
            timezone: "UTC".into(),
            locale: "en_US.UTF-8".into(),
            keymap: "us".into(),
            packages: vec!["vim".into()],
            root_password: false,
        }
    }

    fn never_asked(prompt: &str) -> std::io::Result<String> {
        panic!("asked {prompt:?}")
    }

    fn request<'a>(bootstrap: &'a BootstrapArgs, uefi: &'a UefiBootArgs) -> ReinstallRequest<'a> {
        ReinstallRequest {
            name: "m",
            install: None,
            bootstrap,
            yes: true,
            no_root_password: false,
            uefi,
        }
    }

    fn password_file(home: &crate::test_support::TempDir) -> PathBuf {
        let path = home.path().join("password");
        std::fs::write(&path, "secret\n").unwrap();
        path
    }

    #[test]
    fn confirm_erase_needs_yes_without_tty() {
        let home = reinstall_home("main-confirm-tty", None);
        let machine = machines::load(home.path(), "m").unwrap();
        let error = confirm_erase(&machine, false, false, never_asked).unwrap_err();
        assert!(error.contains("--yes"), "{error}");
        assert!(confirm_erase(&machine, false, true, never_asked).is_ok());
        assert!(confirm_erase(&machine, true, true, never_asked).is_ok());
    }

    #[test]
    fn confirm_erase_accepts_y_and_yes() {
        let home = reinstall_home("main-confirm-answers", None);
        let machine = machines::load(home.path(), "m").unwrap();
        for answer in ["y", "Y", "yes", "YES", "Yes\n", " y \n"] {
            let reply = || Ok(answer.to_string());
            assert!(
                confirm_erase(&machine, true, false, |_| reply()).is_ok(),
                "{answer:?}"
            );
        }
        for answer in ["", "\n", "n", "no", "yess", "sure"] {
            let reply = || Ok(answer.to_string());
            assert_eq!(
                confirm_erase(&machine, true, false, |_| reply()),
                Err("not reinstalled".into()),
                "{answer:?}"
            );
        }
    }

    #[test]
    fn confirm_erase_names_first_disk_and_size() {
        let home = reinstall_home("main-confirm-prompt", None);
        let mut machine = machines::load(home.path(), "m").unwrap();
        machine.install = Distro::Fedora;
        let mut prompts = Vec::new();
        confirm_erase(&machine, true, false, |prompt| {
            prompts.push(prompt.to_string());
            Ok("y".into())
        })
        .unwrap();
        assert_eq!(
            prompts,
            vec!["Erase disk-0.udisk (64 GiB) and install Fedora? [y/N]"]
        );
    }

    #[test]
    fn confirm_erase_of_a_machine_without_a_disk_asks_nothing() {
        let home = reinstall_home("main-confirm-diskless", None);
        let mut machine = machines::load(home.path(), "m").unwrap();
        machine.settings.disks.clear();
        assert!(confirm_erase(&machine, false, false, never_asked).is_ok());
    }

    #[test]
    fn reinstall_with_record_and_no_password_off_tty_fails_before_touching_artifacts() {
        let home = reinstall_home("main-reinstall-no-password", Some(saved_record()));
        let machine = machines::load(home.path(), "m").unwrap();
        let disk = machine.settings.disks[0].path.clone();
        let disk_before = std::fs::read(&disk).unwrap();
        let artifacts_before: Vec<_> = std::fs::read_dir(machine.artifacts_dir())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let (bootstrap, uefi) = (BootstrapArgs::default(), UefiBootArgs::default());
        let fetches = std::cell::Cell::new(0);
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |url, destination| {
                fetches.set(fetches.get() + 1);
                fake_fetch(url, destination)
            },
        )
        .err()
        .unwrap();
        assert!(error.contains("--password-file"), "{error}");
        assert_eq!(fetches.get(), 0);
        let artifacts_after: Vec<_> = std::fs::read_dir(machine.artifacts_dir())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert!(!artifacts_after.is_empty());
        assert_eq!(artifacts_after, artifacts_before);
        assert_eq!(std::fs::read(&disk).unwrap(), disk_before);
        assert_eq!(
            machines::load(home.path(), "m").unwrap().boot_from,
            machine.boot_from
        );
    }

    #[test]
    fn reinstall_without_yes_off_tty_fails_before_touching_artifacts() {
        let home = reinstall_home("main-reinstall-no-yes", Some(saved_record()));
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let mut unconfirmed = request(&bootstrap, &uefi);
        unconfirmed.yes = false;
        let error = prepare_reinstall(
            home.path(),
            &unconfirmed,
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched before the confirmation"),
        )
        .err()
        .unwrap();
        assert!(error.contains("--yes"), "{error}");
    }

    #[test]
    fn unattended_reinstall_of_a_framebuffer_machine_is_refused_before_touching_artifacts() {
        let home = reinstall_home("main-reinstall-framebuffer", Some(saved_record()));
        machines::edit(
            home.path(),
            "m",
            &hardware(&["--display", "framebuffer"]),
            &LinuxArgs::default(),
            &RemoveArgs::default(),
            None,
            None,
        )
        .unwrap();
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched for a machine that cannot report its install"),
        )
        .err()
        .unwrap();
        assert!(error.contains("--display both"), "{error}");
    }

    #[test]
    fn reinstall_flag_overrides_saved_record_and_is_saved() {
        let home = reinstall_home("main-reinstall-override", Some(saved_record()));
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            hostname: Some("other".into()),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let mut switch = request(&bootstrap, &uefi);
        switch.install = Some(Distro::Debian);
        let (machine, answers) = prepare_reinstall(
            home.path(),
            &switch,
            false,
            never_asked,
            never_asked,
            fake_fetch,
        )
        .unwrap();
        let answers = answers.expect("a machine with a record reinstalls unattended");
        assert_eq!(answers.hostname, "other");
        assert_eq!(answers.user, "alex");
        assert_eq!(answers.packages, vec!["vim"]);
        let saved = machines::load(home.path(), "m").unwrap();
        assert_eq!(saved.install, Distro::Debian);
        assert_eq!(saved.boot_from, args::BootSource::Installer);
        assert_eq!(
            saved.bootstrap,
            Some(BootstrapRecord {
                hostname: "other".into(),
                ..saved_record()
            })
        );
        assert_eq!(machine.bootstrap, saved.bootstrap);
        // The new distro's artifacts were fetched.
        let (kernel_url, _) = Distro::Debian.source_urls();
        let fetched = std::fs::read_dir(machine.artifacts_dir())
            .unwrap()
            .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap())
            .collect::<Vec<_>>();
        assert!(fetched.iter().any(|url| url == kernel_url), "{fetched:?}");
    }

    #[test]
    fn reinstall_with_user_makes_a_bootstrap_of_an_interactive_machine() {
        let home = reinstall_home("main-reinstall-user", None);
        let bootstrap = BootstrapArgs {
            user: Some("sam".into()),
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let (_, answers) = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            fake_fetch,
        )
        .unwrap();
        assert_eq!(answers.unwrap().user, "sam");
        let record = machines::load(home.path(), "m").unwrap().bootstrap.unwrap();
        assert_eq!(record.user, "sam");
    }

    #[test]
    fn reinstall_without_record_or_user_stays_interactive() {
        let home = reinstall_home("main-reinstall-interactive", None);
        let (bootstrap, uefi) = (BootstrapArgs::default(), UefiBootArgs::default());
        let mut interactive = request(&bootstrap, &uefi);
        interactive.yes = false;
        let (machine, answers) = prepare_reinstall(
            home.path(),
            &interactive,
            false,
            never_asked,
            never_asked,
            fake_fetch,
        )
        .unwrap();
        assert!(answers.is_none());
        assert!(machine.bootstrap.is_none());
        assert_eq!(machine.boot_from, args::BootSource::Installer);
    }

    #[test]
    fn reinstall_refuses_answers_without_a_user_on_an_interactive_machine() {
        let home = reinstall_home("main-reinstall-answers-no-user", None);
        let bootstrap = BootstrapArgs {
            hostname: Some("box".into()),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched"),
        )
        .err()
        .unwrap();
        assert!(error.contains("--user"), "{error}");
    }

    #[test]
    fn no_root_password_drops_a_saved_root_password() {
        let home = reinstall_home(
            "main-reinstall-no-root",
            Some(BootstrapRecord {
                root_password: true,
                ..saved_record()
            }),
        );
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        // Without the flag, root's password is asked for again, and there is no terminal.
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched"),
        )
        .err()
        .unwrap();
        assert!(error.contains("--root-password-file"), "{error}");
        let mut dropped = request(&bootstrap, &uefi);
        dropped.no_root_password = true;
        let (_, answers) = prepare_reinstall(
            home.path(),
            &dropped,
            false,
            never_asked,
            never_asked,
            fake_fetch,
        )
        .unwrap();
        assert_eq!(answers.unwrap().root_password_hash, None);
        let record = machines::load(home.path(), "m").unwrap().bootstrap.unwrap();
        assert!(!record.root_password);
    }

    #[test]
    fn bootstrap_reinstall_refuses_firmware_none_and_a_hypervisor() {
        let home = reinstall_home("main-reinstall-unbootable", Some(saved_record()));
        machines::edit(
            home.path(),
            "m",
            &HardwareArgs::default(),
            &LinuxArgs::default(),
            &RemoveArgs::default(),
            None,
            Some(args::Firmware::None),
        )
        .unwrap();
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched"),
        )
        .err()
        .unwrap();
        assert!(error.contains("firmware none"), "{error}");
    }

    fn machine_state(
        home: &crate::test_support::TempDir,
    ) -> (String, Vec<(std::ffi::OsString, Vec<u8>)>) {
        let dir = home.path().join("machines/m");
        let toml = std::fs::read_to_string(dir.join("machine.toml")).unwrap();
        let mut artifacts: Vec<_> = std::fs::read_dir(dir.join("artifacts"))
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), std::fs::read(entry.path()).unwrap())
            })
            .collect();
        artifacts.sort();
        (toml, artifacts)
    }

    #[test]
    fn interactive_reinstall_refuses_boot_menu_before_any_change() {
        let home = reinstall_home("main-reinstall-boot-menu", None);
        let before = machine_state(&home);
        let bootstrap = BootstrapArgs::default();
        let uefi = UefiBootArgs {
            boot_menu: true,
            ..UefiBootArgs::default()
        };
        let error = prepare_reinstall(
            home.path(),
            &request(&bootstrap, &uefi),
            false,
            never_asked,
            never_asked,
            |_, _| panic!("fetched"),
        )
        .err()
        .unwrap();
        assert!(error.starts_with("--boot-menu"), "{error}");
        assert_eq!(machine_state(&home), before);
    }

    #[test]
    fn unattended_reinstall_keeps_boot_menu_for_the_disk_boot() {
        let home = reinstall_home("main-reinstall-boot-menu-unattended", Some(saved_record()));
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs {
            boot_menu: true,
            ..UefiBootArgs::default()
        };
        assert!(
            prepare_reinstall(
                home.path(),
                &request(&bootstrap, &uefi),
                false,
                never_asked,
                never_asked,
                fake_fetch,
            )
            .unwrap()
            .1
            .is_some()
        );
    }

    #[test]
    fn reinstall_question_names_the_new_distro() {
        let home = reinstall_home("main-reinstall-question", Some(saved_record()));
        let bootstrap = BootstrapArgs {
            password_file: Some(password_file(&home)),
            ..BootstrapArgs::default()
        };
        let uefi = UefiBootArgs::default();
        let mut switch = request(&bootstrap, &uefi);
        switch.install = Some(Distro::Fedora);
        switch.yes = false;
        let mut questions = Vec::new();
        prepare_reinstall(
            home.path(),
            &switch,
            true,
            never_asked,
            |question| {
                questions.push(question.to_string());
                Ok("yes\n".into())
            },
            fake_fetch,
        )
        .unwrap();
        assert_eq!(
            questions,
            vec!["Erase disk-0.udisk (64 GiB) and install Fedora? [y/N]"]
        );
        assert_eq!(
            machines::load(home.path(), "m").unwrap().install,
            Distro::Fedora
        );
    }
}
