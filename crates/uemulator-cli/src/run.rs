use std::io::{IsTerminal, Read, Write};
use std::time::Duration;

use uemulator::{
    ConsoleKind, Disk, EmulatedMachine, ExitReason, JitMode, MachineConfig, NetHandle, NetOffloads,
    SmpMode,
};

use crate::args::{ConsoleArg, NetLink, RuntimeArgs, Settings, SmpArg};
use crate::netgw;
use crate::term::{stty, term_size};

/// QEMU's conventional default NIC MAC (52:54:00 is its locally administered prefix).
pub(crate) const GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
pub(crate) const DEFAULT_MEMORY_MIB: usize = 2048;
/// Ctrl-] (GS). Raw mode passes Ctrl-C and friends to the guest, so this is the host's escape.
const ESCAPE_BYTE: u8 = 0x1d;
const STDIN_CHUNK_LEN: usize = 256;
/// Polled rather than signal driven (SIGWINCH has no portable path here).
const TERM_SIZE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const UDP_MAX_DATAGRAM: usize = 65536;
/// One thread both receives from the socket and drains guest frames, so a blocking recv must
/// time out quickly to keep guest-to-host frames flowing.
const UDP_RECV_TIMEOUT: Duration = Duration::from_millis(2);
const GATEWAY_RECV_TIMEOUT: Duration = Duration::from_millis(5);

/// Splits `--network-link-udp LOCAL,REMOTE` at the first comma.
fn parse_udp_spec(spec: &str) -> Result<(&str, &str), String> {
    spec.split_once(',')
        .ok_or_else(|| "--network-link-udp takes LOCAL,REMOTE".to_string())
}

fn open_udp_socket(spec: &str) -> Result<std::net::UdpSocket, String> {
    let (local, remote) = parse_udp_spec(spec)?;
    let sock = std::net::UdpSocket::bind(local).map_err(|e| format!("bind {local}: {e}"))?;
    sock.connect(remote)
        .map_err(|e| format!("connect {remote}: {e}"))?;
    sock.set_read_timeout(Some(UDP_RECV_TIMEOUT))
        .map_err(|e| e.to_string())?;
    Ok(sock)
}

/// The consoles to configure: a machine that chose none gets the virtio console, like the library.
fn console_list(settings: &Settings) -> Vec<ConsoleArg> {
    match settings.consoles.is_empty() {
        true => vec![ConsoleArg::Virtio],
        false => settings.consoles.clone(),
    }
}

/// Always set on the config, never skipped for the "default": the library's own default is
/// virtio, so leaving `--console serial` unapplied would silently give hvc0.
fn console_kind(arg: ConsoleArg) -> ConsoleKind {
    match arg {
        ConsoleArg::Serial => ConsoleKind::Serial,
        ConsoleArg::Virtio => ConsoleKind::Virtio,
    }
}

/// The MAC of the `index`-th NIC: `GUEST_MAC` with the last byte counted up, so every NIC is
/// distinct (the library caps a machine at 31 virtio devices, so the byte never wraps).
fn guest_mac(index: usize) -> [u8; 6] {
    let mut mac = GUEST_MAC;
    mac[5] = mac[5].wrapping_add(index as u8);
    mac
}

/// Raw mode passes Ctrl-C and friends to the guest, so a chunk holding the escape byte is the
/// host asking to stop. The whole chunk is dropped, not forwarded.
fn contains_escape(chunk: &[u8]) -> bool {
    chunk.contains(&ESCAPE_BYTE)
}

/// Puts the terminal behind `stty` into raw mode. Returns the settings to hand back to `stty` on
/// exit, or `None` if raw mode was not entered (no `stty`, not a terminal, not unix). Without the
/// saved settings the terminal could not be restored, so raw mode is not attempted then.
fn enter_raw_mode(stty: impl Fn(&[&str]) -> Option<String>) -> Option<String> {
    let saved = stty(&["-g"])?;
    if stty(&["raw", "-echo"]).is_none() {
        stty(&[&saved]);
        return None;
    }
    Some(saved)
}

/// What each link handles in the guest's place, which decides the offloads its NIC offers.
/// usernet terminates TCP and never checks a checksum, so it takes the guest's 64 KiB segments
/// and sends large ones back. A UDP link's far end is another machine expecting real Ethernet,
/// and the loop gateway reads frames itself: both get MTU-sized checksummed frames.
fn offloads_for(link: &NetLink) -> NetOffloads {
    match link {
        NetLink::User => NetOffloads {
            takes_guest_segments: true,
            sends_large_segments: true,
        },
        NetLink::Udp(_) | NetLink::Loop => NetOffloads::default(),
    }
}

fn spawn_udp(net: NetHandle, spec: &str) -> Result<(), String> {
    let sock = open_udp_socket(spec)?;
    std::thread::spawn(move || {
        let mut buf = vec![0u8; UDP_MAX_DATAGRAM];
        loop {
            if let Ok(n) = sock.recv(&mut buf) {
                net.send_frame(&buf[..n]);
            }
            while let Some(frame) = net.try_recv_frame() {
                let _ = sock.send(&frame);
            }
        }
    });
    Ok(())
}

fn spawn_gateway(net: NetHandle) {
    std::thread::spawn(move || {
        loop {
            if let Some(frame) = net.recv_frame(GATEWAY_RECV_TIMEOUT)
                && let Some(reply) = netgw::handle(&frame)
            {
                net.send_frame(&reply);
            }
        }
    });
}

fn build_config(
    kernel: Vec<u8>,
    initramfs: Option<Vec<u8>>,
    settings: &Settings,
    runtime: &RuntimeArgs,
) -> Result<MachineConfig, String> {
    let mut cfg = MachineConfig::new()
        .kernel(kernel)
        .debug(runtime.debug)
        .verify_jit(runtime.verify);
    if runtime.debug {
        cfg = cfg.log(|line| eprintln!("{line}"));
    }
    if let Some(initramfs) = initramfs {
        cfg = cfg.initrd(initramfs);
    }
    cfg = cfg.memory_mib(settings.memory.unwrap_or(DEFAULT_MEMORY_MIB));
    if let Some(cmdline) = &settings.cmdline {
        cfg = cfg.cmdline(cmdline.clone());
    }
    if let Some(extra) = &settings.append {
        cfg = cfg.cmdline_append(extra.clone());
    }
    for disk in &settings.disks {
        let path = disk.path.display();
        let location = crate::images::Location::File(disk.path.clone());
        let backend = crate::images::open_disk(&location, disk.read_only)
            .map_err(|e| format!("open disk {path}: {e}"))?;
        cfg = cfg.disk(Disk::from_backend(backend));
    }
    for index in 0..settings.networks.len() {
        cfg = cfg.net(guest_mac(index));
    }
    for console in console_list(settings) {
        cfg = cfg.console(console_kind(console));
    }
    if let Some(t) = runtime.time {
        cfg = cfg.unix_time(t);
    }
    cfg = cfg.random_seed(host_random_seed());
    cfg = cfg.rng(!settings.disable_rng);
    if settings.disable_jit {
        cfg = cfg.jit(JitMode::Off);
    }
    cfg = cfg.reboot(!settings.no_reboot).iommu(settings.iommu);
    if let Some(cpus) = settings.cpus {
        cfg = cfg.cpus(cpus);
    }
    if let Some(smp) = settings.smp {
        cfg = cfg.smp(match smp {
            SmpArg::Auto => SmpMode::Auto,
            SmpArg::Timeshare => SmpMode::Timeshare,
            SmpArg::Threads => SmpMode::Threads,
        });
    }
    Ok(cfg)
}

/// The line printed when the machine ends: a guest power-off and a guest reset (only seen with
/// `--no-reboot`) read differently from a host stop or a failure.
pub(crate) fn exit_message(reason: &ExitReason) -> String {
    match reason {
        ExitReason::Shutdown => "machine powered off".to_string(),
        ExitReason::Reset => "machine reset".to_string(),
        ExitReason::Stopped => "machine stopped".to_string(),
        ExitReason::TripleFault => "machine failed: guest triple fault".to_string(),
        ExitReason::Internal(message) => format!("machine failed: {message}"),
    }
}

/// Runs the machine to its end and reports why it ended. `Err` is a setup failure; the machine's
/// own failures come back as an `ExitReason` for `main` to map to a status.
pub fn run(
    kernel: Vec<u8>,
    initramfs: Option<Vec<u8>>,
    settings: &Settings,
    runtime: &RuntimeArgs,
) -> Result<ExitReason, String> {
    let tty = std::io::stdin().is_terminal();
    let cfg = build_config(kernel, initramfs, settings, runtime)?;
    run_config(cfg, settings, runtime, tty)
}

/// Like `run`, for the machine's first disk booted through the library's BIOS: the disk's own boot
/// loader loads the system, so there is no kernel, initramfs or command line here.
pub fn run_disk(settings: &Settings, runtime: &RuntimeArgs) -> Result<ExitReason, String> {
    let tty = std::io::stdin().is_terminal();
    let cfg = build_config(Vec::new(), None, settings, runtime)?.boot_disk(true);
    run_config(cfg, settings, runtime, tty)
}

/// Like `run`, for a firmware image started in real mode at the reset vector.
pub fn run_firmware(
    image: Vec<u8>,
    settings: &Settings,
    runtime: &RuntimeArgs,
) -> Result<ExitReason, String> {
    let tty = std::io::stdin().is_terminal();
    let cfg = build_config(Vec::new(), None, settings, runtime)?.firmware(image);
    run_config(cfg, settings, runtime, tty)
}

/// Like `run`, through the built-in UEFI firmware. An empty `kernel` boots the machine's disks;
/// otherwise the firmware boots that kernel (an EFI-stub bzImage) with `initramfs`. `nvram` keeps
/// the firmware's variables in a file; `boot_entry` (loader path, load options) is written into
/// it first so that a boot goes where the caller asked.
pub fn run_uefi(
    kernel: Vec<u8>,
    initramfs: Option<Vec<u8>>,
    settings: &Settings,
    runtime: &RuntimeArgs,
    nvram: Option<&std::path::Path>,
    boot_entry: Option<(&str, &str)>,
) -> Result<ExitReason, String> {
    let tty = std::io::stdin().is_terminal();
    let mut cfg =
        build_config(kernel, initramfs, settings, runtime)?.uefi_firmware(crate::firmware::IMAGE);
    if let Some(path) = nvram {
        let mut store = uemulator::FileKeyValueStore::open(path).map_err(|e| e.to_string())?;
        if let Some((loader, options)) = boot_entry {
            crate::firmware::set_boot_entry(&mut store, loader, options)
                .map_err(|e| e.to_string())?;
        }
        cfg = cfg.key_value_store(store);
    }
    run_config(cfg, settings, runtime, tty)
}

fn run_config(
    cfg: MachineConfig,
    settings: &Settings,
    runtime: &RuntimeArgs,
    tty: bool,
) -> Result<ExitReason, String> {
    let mut machine = EmulatedMachine::new(cfg).map_err(|e| e.to_string())?;
    let console = machine.console();
    if machine.consoles().len() > 1 {
        eprintln!(
            "note: the terminal is attached to the first console; the other {} have no host side",
            machine.consoles().len() - 1
        );
    }
    // `build_config` made one NIC per entry, in this order.
    for (net, nic) in machine.nets().into_iter().zip(&settings.networks) {
        if let Some(link) = &nic.link {
            // Before `start`: the guest's driver reads the features once it probes the NIC.
            net.set_offloads(offloads_for(link));
        }
        match &nic.link {
            Some(NetLink::Udp(spec)) => spawn_udp(net, spec)?,
            Some(NetLink::Loop) => spawn_gateway(net),
            Some(NetLink::User) => crate::usernet::spawn(net),
            None => {}
        }
    }
    if let Some((cols, rows)) = term_size() {
        console.resize(cols, rows);
    }
    let sizer = console.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(TERM_SIZE_POLL_INTERVAL);
            if let Some((cols, rows)) = term_size() {
                sizer.resize(cols, rows);
            }
        }
    });
    console.on_output(|b| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(b);
        let _ = out.flush();
    });
    // `stty -g` output is the exact setting string to hand back to `stty` on exit.
    let saved_tty = if tty { enter_raw_mode(stty) } else { None };
    let input = console.clone();
    let stopper = machine.stop_handle();
    std::thread::spawn(move || {
        let mut buf = [0u8; STDIN_CHUNK_LEN];
        while let Ok(n) = std::io::stdin().read(&mut buf) {
            if n == 0 {
                break;
            }
            if contains_escape(&buf[..n]) {
                stopper.stop();
                break;
            }
            input.write(&buf[..n]);
        }
    });
    if let Some(secs) = runtime.timeout {
        let stop = machine.stop_handle();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            stop.stop();
        });
    }

    let started = machine.start().map_err(|e| e.to_string());
    let reason = started.map(|()| machine.wait());
    // Restore before `reason?` so a failed start does not leave the terminal raw.
    if let Some(saved) = saved_tty {
        stty(&[&saved]);
    }
    let reason = reason?;
    // Clean exits (power-off, reset, stop) stay silent; only failures are reported.
    if matches!(reason, ExitReason::TripleFault | ExitReason::Internal(_)) {
        eprintln!("{}", exit_message(&reason));
    }
    Ok(reason)
}

/// Seed for the guest's rdrand/rdseed from the host's per-process random hasher keys (which the
/// standard library draws from the OS CSPRNG), so two boots do not see the same "hardware" random
/// numbers.
fn host_random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::test_support::TempDir;
    use crate::{AdvancedCmd, Cmd};

    #[test]
    fn only_usernet_gets_offloads() {
        assert_eq!(
            offloads_for(&NetLink::User),
            NetOffloads {
                takes_guest_segments: true,
                sends_large_segments: true,
            }
        );
        assert_eq!(offloads_for(&NetLink::Loop), NetOffloads::default());
        assert_eq!(
            offloads_for(&NetLink::Udp("127.0.0.1:1,127.0.0.1:2".into())),
            NetOffloads::default()
        );
    }

    const SAVED_SETTINGS: &str = "4500:5:bf:8a3b:3:1c:7f:15:4:0:1:0:11:13:1a:0:12:f:17:16:0:0";
    use uemulator::{MAX_MEMORY_MIB, MIN_MEMORY_MIB};

    /// The settings and runtime options `advanced boot --kernel k <extra>` parses to.
    fn boot_args(extra: &[&str]) -> (Settings, RuntimeArgs) {
        let mut argv = vec!["uemulator", "advanced", "boot", "--kernel", "k"];
        argv.extend_from_slice(extra);
        match crate::parse_cli(argv).unwrap().cmd {
            Cmd::Advanced {
                cmd:
                    AdvancedCmd::Boot {
                        hardware,
                        linux,
                        runtime,
                        ..
                    },
            } => (crate::settings_of(&hardware, &linux).unwrap(), runtime),
            _ => unreachable!(),
        }
    }

    fn build(extra: &[&str], initramfs: Option<Vec<u8>>) -> Result<MachineConfig, String> {
        let (settings, runtime) = boot_args(extra);
        build_config(Vec::new(), initramfs, &settings, &runtime)
    }

    /// What the library says about `cfg` before it looks at the kernel. The kernel is empty, so
    /// the only way to get past the configuration checks is to fail on the kernel image.
    fn machine_error(cfg: MachineConfig) -> String {
        match EmulatedMachine::new(cfg) {
            Ok(_) => panic!("an empty kernel must not boot"),
            Err(e) => e.to_string(),
        }
    }

    /// A fake `stty` that records its calls and answers from a script.
    struct FakeStty {
        calls: RefCell<Vec<Vec<String>>>,
        fail_on: Option<&'static str>,
    }

    impl FakeStty {
        fn working() -> Self {
            FakeStty {
                calls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }

        fn failing_on(first_arg: &'static str) -> Self {
            FakeStty {
                calls: RefCell::new(Vec::new()),
                fail_on: Some(first_arg),
            }
        }

        fn call(&self, args: &[&str]) -> Option<String> {
            self.calls
                .borrow_mut()
                .push(args.iter().map(|a| a.to_string()).collect());
            if self.fail_on == Some(args[0]) {
                return None;
            }
            Some(if args[0] == "-g" {
                SAVED_SETTINGS.to_string()
            } else {
                String::new()
            })
        }

        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.borrow().clone()
        }
    }

    #[test]
    fn each_console_argument_maps_to_its_own_kind() {
        assert_eq!(console_kind(ConsoleArg::Serial), ConsoleKind::Serial);
        assert_eq!(console_kind(ConsoleArg::Virtio), ConsoleKind::Virtio);
    }

    #[test]
    fn every_exit_reason_prints_a_distinct_message() {
        assert_eq!(exit_message(&ExitReason::Shutdown), "machine powered off");
        assert_eq!(exit_message(&ExitReason::Reset), "machine reset");
        assert_eq!(exit_message(&ExitReason::Stopped), "machine stopped");
        assert_eq!(
            exit_message(&ExitReason::TripleFault),
            "machine failed: guest triple fault"
        );
        assert_eq!(
            exit_message(&ExitReason::Internal("jit divergence".into())),
            "machine failed: jit divergence"
        );
    }

    #[test]
    fn udp_spec_splits_at_the_first_comma() {
        assert_eq!(
            parse_udp_spec("127.0.0.1:1234,127.0.0.1:5678"),
            Ok(("127.0.0.1:1234", "127.0.0.1:5678"))
        );
        assert_eq!(parse_udp_spec("a,b,c"), Ok(("a", "b,c")));
        assert_eq!(parse_udp_spec(","), Ok(("", "")));
    }

    #[test]
    fn udp_spec_without_comma_is_rejected_with_usage() {
        assert_eq!(
            parse_udp_spec("127.0.0.1:1234"),
            Err("--network-link-udp takes LOCAL,REMOTE".to_string())
        );
        assert!(parse_udp_spec("").is_err());
    }

    #[test]
    fn udp_socket_is_bound_connected_and_has_a_short_read_timeout() {
        let peer = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let sock = open_udp_socket(&format!("127.0.0.1:0,{}", peer.local_addr().unwrap())).unwrap();
        assert_eq!(sock.read_timeout().unwrap(), Some(UDP_RECV_TIMEOUT));
        assert_eq!(sock.peer_addr().unwrap(), peer.local_addr().unwrap());
        sock.send(b"frame").unwrap();
        let mut buf = [0u8; 16];
        let (n, from) = peer.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"frame");
        assert_eq!(from, sock.local_addr().unwrap());
    }

    #[test]
    fn udp_socket_receive_times_out_instead_of_blocking() {
        let peer = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let sock = open_udp_socket(&format!("127.0.0.1:0,{}", peer.local_addr().unwrap())).unwrap();
        let mut buf = [0u8; 16];
        let started = std::time::Instant::now();
        assert!(sock.recv(&mut buf).is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn udp_socket_bind_failure_names_the_local_address() {
        let taken = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let local = taken.local_addr().unwrap();
        let err = open_udp_socket(&format!("{local},127.0.0.1:9")).unwrap_err();
        assert!(err.starts_with(&format!("bind {local}: ")), "{err}");
        let err = open_udp_socket("not-an-address,127.0.0.1:9").unwrap_err();
        assert!(err.starts_with("bind not-an-address: "), "{err}");
    }

    #[test]
    fn udp_socket_connect_failure_names_the_remote_address() {
        let err = open_udp_socket("127.0.0.1:0,not-an-address").unwrap_err();
        assert!(err.starts_with("connect not-an-address: "), "{err}");
    }

    #[test]
    fn udp_socket_without_comma_reports_usage() {
        assert_eq!(
            open_udp_socket("127.0.0.1:0").unwrap_err(),
            "--network-link-udp takes LOCAL,REMOTE"
        );
    }

    #[test]
    fn every_network_adds_a_nic_whether_or_not_it_is_linked() {
        for (flags, count) in [
            (&[][..], 0),
            (&["--network", "virtio"], 1),
            (&["--network", "virtio", "--network-link-loop"], 1),
            (
                &[
                    "--network",
                    "virtio",
                    "--network-link-user",
                    "--network",
                    "virtio",
                ],
                2,
            ),
            (
                &[
                    "--network",
                    "virtio",
                    "--network-link-user",
                    "--network",
                    "virtio",
                    "--network-link-udp",
                    "a,b",
                    "--network",
                    "virtio",
                ],
                3,
            ),
        ] {
            assert_eq!(boot_args(flags).0.networks.len(), count, "{flags:?}");
        }
    }

    #[test]
    fn a_machine_that_chose_no_console_gets_the_virtio_one() {
        assert_eq!(console_list(&Settings::default()), vec![ConsoleArg::Virtio]);
        let serial = Settings {
            consoles: vec![ConsoleArg::Serial, ConsoleArg::Virtio],
            ..Settings::default()
        };
        assert_eq!(
            console_list(&serial),
            vec![ConsoleArg::Serial, ConsoleArg::Virtio]
        );
    }

    #[test]
    fn every_nic_gets_its_own_mac_counted_up_from_the_guest_mac() {
        assert_eq!(guest_mac(0), GUEST_MAC);
        let macs: Vec<[u8; 6]> = (0..31).map(guest_mac).collect();
        for (index, mac) in macs.iter().enumerate() {
            assert_eq!(mac[..5], GUEST_MAC[..5]);
            assert_eq!(mac[5], GUEST_MAC[5] + index as u8);
            assert_eq!(mac[0] & 0x01, 0, "unicast");
        }
        let mut unique = macs.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), macs.len());
    }

    #[test]
    fn repeated_disks_nics_and_consoles_build_one_machine() {
        let dir = TempDir::new("run-multi");
        let (first, second) = (dir.path().join("a.img"), dir.path().join("b.img"));
        std::fs::write(&first, vec![0u8; 4096]).unwrap();
        std::fs::write(&second, vec![0u8; 4096]).unwrap();
        let (first, second) = (first.to_str().unwrap(), second.to_str().unwrap());
        let cfg = build(
            &[
                "--disk",
                first,
                "--disk",
                &format!("{second},ro"),
                "--network",
                "virtio",
                "--network-link-loop",
                "--network",
                "virtio",
                "--network-link-user",
                "--console",
                "serial",
                "--console",
                "virtio",
            ],
            None,
        )
        .unwrap();
        // The empty kernel fails only after the configuration (device count, console rules)
        // was accepted.
        assert!(!machine_error(cfg).contains("virtio devices"));
    }

    #[test]
    fn two_serial_consoles_are_rejected_by_the_library() {
        let cfg = build(&["--console", "serial", "--console", "serial"], None).unwrap();
        assert!(machine_error(cfg).contains("only once"));
    }

    #[test]
    fn escape_byte_is_ctrl_right_bracket() {
        assert_eq!(ESCAPE_BYTE, 0x1d);
        assert_eq!(ESCAPE_BYTE, b']' & 0x1f);
    }

    #[test]
    fn chunk_with_escape_byte_anywhere_stops_the_machine() {
        assert!(contains_escape(&[ESCAPE_BYTE]));
        assert!(contains_escape(&[b'a', b'b', ESCAPE_BYTE]));
        assert!(contains_escape(&[ESCAPE_BYTE, b'a']));
        assert!(contains_escape(b"typed ahead \x1d and more"));
    }

    #[test]
    fn chunk_without_escape_byte_is_forwarded() {
        assert!(!contains_escape(&[]));
        assert!(!contains_escape(b"ls -l\r"));
        assert!(!contains_escape(&[0x03, 0x1c, 0x1e, 0x04]));
        assert!(!contains_escape(&[0xff; STDIN_CHUNK_LEN]));
    }

    #[test]
    fn raw_mode_saves_settings_then_goes_raw() {
        let stty = FakeStty::working();
        let saved = enter_raw_mode(|args| stty.call(args));
        assert_eq!(saved.as_deref(), Some(SAVED_SETTINGS));
        assert_eq!(
            stty.calls(),
            vec![vec!["-g".to_string()], vec!["raw".into(), "-echo".into()]]
        );
    }

    #[test]
    fn raw_mode_is_not_entered_when_stty_is_unavailable() {
        // This is the non-unix case, where `stty` always returns `None`: no raw mode, so the
        // caller must not claim there is one.
        assert_eq!(enter_raw_mode(|_| None), None);
    }

    #[test]
    fn raw_mode_is_not_attempted_without_saved_settings() {
        let stty = FakeStty::failing_on("-g");
        assert_eq!(enter_raw_mode(|args| stty.call(args)), None);
        assert_eq!(stty.calls(), vec![vec!["-g".to_string()]]);
    }

    #[test]
    fn failed_raw_mode_restores_the_saved_settings() {
        let stty = FakeStty::failing_on("raw");
        assert_eq!(enter_raw_mode(|args| stty.call(args)), None);
        assert_eq!(
            stty.calls(),
            vec![
                vec!["-g".to_string()],
                vec!["raw".into(), "-echo".into()],
                vec![SAVED_SETTINGS.to_string()],
            ]
        );
    }

    #[cfg(not(unix))]
    #[test]
    fn real_stty_never_enters_raw_mode_off_unix() {
        assert_eq!(enter_raw_mode(stty), None);
    }

    #[test]
    fn guest_mac_is_a_locally_administered_unicast_address() {
        assert_eq!(GUEST_MAC[0] & 0x01, 0, "unicast");
        assert_eq!(GUEST_MAC[0] & 0x02, 0x02, "locally administered");
        assert_eq!(GUEST_MAC[..3], [0x52, 0x54, 0x00]);
    }

    #[test]
    fn default_memory_fits_the_library_limits() {
        assert!((MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&DEFAULT_MEMORY_MIB));
    }

    #[test]
    fn memory_below_the_library_minimum_is_rejected_by_the_library() {
        let memory = (MIN_MEMORY_MIB - 1).to_string();
        let cfg = build(&["-m", &memory, "--disable-jit"], None).unwrap();
        assert!(machine_error(cfg).contains(&format!("between 64 and {MAX_MEMORY_MIB}")));
    }

    #[test]
    fn memory_above_the_library_maximum_is_rejected_by_the_library() {
        let memory = (MAX_MEMORY_MIB + 1).to_string();
        let cfg = build(&["--memory", &memory, "--disable-jit"], None).unwrap();
        assert!(machine_error(cfg).contains(&format!("between 64 and {MAX_MEMORY_MIB}")));
    }

    #[test]
    fn the_minimum_memory_is_accepted_by_the_library() {
        // Only the minimum is probed: acceptance at the maximum would reserve a terabyte of
        // address space.
        let memory = MIN_MEMORY_MIB.to_string();
        let cfg = build(&["-m", &memory, "--disable-jit"], None).unwrap();
        assert!(machine_error(cfg).contains("kernel"));
    }

    #[test]
    fn missing_disk_image_is_a_setup_error_naming_the_path() {
        let dir = TempDir::new("run-disk");
        let missing = dir.path().join("missing.img");
        let spec = missing.to_string_lossy().into_owned();
        let err = build(&["--disk", &spec], None).err().unwrap();
        assert!(err.starts_with(&format!("open disk {spec}: ")), "{err}");
    }

    #[test]
    fn read_only_suffix_is_not_part_of_the_opened_path() {
        let dir = TempDir::new("run-disk-ro");
        let missing = dir.path().join("missing.img");
        let path = missing.to_string_lossy().into_owned();
        let spec = format!("{path},ro");
        let err = build(&["--disk", &spec], None).err().unwrap();
        assert!(err.starts_with(&format!("open disk {path}: ")), "{err}");
    }

    #[test]
    fn existing_disk_images_open_read_write_and_read_only() {
        let dir = TempDir::new("run-disk-ok");
        let image = dir.path().join("disk.img");
        std::fs::write(&image, vec![0u8; 4096]).unwrap();
        let path = image.to_string_lossy().into_owned();
        for spec in [path.clone(), format!("{path},ro")] {
            assert!(build(&["--disk", &spec], None).is_ok(), "{spec}");
        }
    }

    #[test]
    fn every_backend_and_console_combination_builds_a_config() {
        for flags in [
            &["--network", "virtio", "--network-link-loop"][..],
            &["--network", "virtio", "--network-link-user"],
            &["--network", "virtio", "--network-link-udp", "a,b"],
            &["--network", "virtio"],
            &["--console", "virtio"],
            &["--console", "serial"],
            &["--time", "1700000000"],
            &["--debug", "--verify", "--disable-jit"],
            &["--cmdline", "console=ttyS0", "--append", "quiet"],
        ] {
            assert!(build(flags, Some(vec![1, 2, 3])).is_ok(), "{flags:?}");
        }
    }
}
