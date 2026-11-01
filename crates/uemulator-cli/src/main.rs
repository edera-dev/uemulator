mod args;
mod firmware;
mod frontend;
mod images;
mod machines;
mod netgw;
mod quick;
mod run;
mod term;
#[cfg(test)]
mod test_support;
mod usernet;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand};
use uemulator::ExitReason;

use args::{
    BootSourceArgs, FirmwareArgs, HardwareArgs, LinuxArgs, NewDiskArgs, RemoveArgs, RuntimeArgs,
    Settings, UefiBootArgs, attach_only,
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
     with Ctrl-] or --timeout
  1  setup failed (bad arguments, unreadable file, download failure, rejected configuration)
  2  the machine failed: guest triple fault or emulator-internal error (e.g. --verify divergence)";

/// Maps how the machine ended to the process exit status. A power-off is how a guest finishes
/// cleanly; a reset only reaches here with `--no-reboot` and counts as success because Linux
/// guests reboot on panic with `panic=-1` and a clean reboot is how an installer finishes.
/// Stopping the machine (Ctrl-] or `--timeout`) is a deliberate host action, so it is success too;
/// callers that need to tell a timeout from a normal end can run without `--timeout`.
fn exit_code(reason: &ExitReason) -> i32 {
    match reason {
        ExitReason::Shutdown | ExitReason::Reset | ExitReason::Stopped => EXIT_SUCCESS,
        ExitReason::TripleFault | ExitReason::Internal(_) => EXIT_MACHINE_FAILURE,
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
    /// Without --disk or --create-disk the machine gets an 8 GiB disk, without --network a
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
        #[command(flatten)]
        runtime: RuntimeArgs,
    },
    /// Throw away a machine's kernel and initramfs and download fresh ones; the disk and the
    /// configuration are kept
    Reinstall {
        /// The machine to reinstall
        name: String,
        /// Switch the machine to another distro [default: the one it has]
        #[arg(long, value_name = "DISTRO")]
        install: Option<Distro>,
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
    /// Boot a kernel (bzImage) and optional initramfs
    ///
    /// The UEFI firmware boots them, which needs a kernel with an EFI stub; `--firmware none`
    /// enters any bzImage directly.
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

fn boot_launch(
    launch: machines::Launch,
    uefi: &UefiBootArgs,
    runtime: &RuntimeArgs,
) -> Result<ExitReason, String> {
    if uefi.any() && launch.firmware != args::Firmware::Uefi {
        return Err("--boot-menu and --boot-timeout need a machine with --firmware uefi".into());
    }
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
                    return Err("--boot-menu is for disk boots: the installer's kernel is \
                                booted directly, without the menu"
                        .into());
                }
                eprintln!("booting the installer through the UEFI firmware");
                run::run_uefi(
                    launch.kernel,
                    launch.initramfs,
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
                quick::download,
            )?;
            boot_launch(launch, &UefiBootArgs::default(), &RuntimeArgs::default()).map(Some)
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
            runtime,
        } => {
            let machine = machines::load(&home, &name)?;
            let launch = machines::prepare_boot(
                &machine,
                &hardware,
                &linux,
                &remove,
                source.boot_from,
                quick::download,
            )?;
            boot_launch(launch, &uefi, &runtime).map(Some)
        }
        QuickCmd::Reinstall { name, install } => {
            let machine = machines::reinstall(&home, &name, install, quick::download)?;
            eprintln!(
                "{name} will install {}; its disks and settings are unchanged. It boots the \
                 installer until `quick edit {name} --boot-from auto`",
                machine.install.name()
            );
            Ok(None)
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
            let kernel_image = read(&kernel)?;
            let initramfs_image = initramfs.as_ref().map(read).transpose()?;
            let settings = settings_of(&hardware, &linux)?;
            match firmware.firmware.unwrap_or_default() {
                args::Firmware::Uefi => run::run_uefi(
                    kernel_image,
                    initramfs_image,
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
}
