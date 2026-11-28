//! Alpine: the netboot init has no answer-file hook, but with `apkovl=` naming a path inside the
//! initramfs it unpacks that overlay into the new root before OpenRC starts
//! (`docs/plans/35-quick-bootstrap.md`, "Delivery per distro"). The overlay turns on the
//! `local` service and gives it a script that runs `setup-alpine` from an answers file, installs
//! to the disk with `setup-disk`, sets up the user on the installed system and reboots.

use std::fmt::Write;

use crate::answers::{PRINT_COMPLETION_MARKER, shell_quote};
use crate::targz::{TarEntry, tar_gz};
use crate::{Answers, CpioFile, Payload, S_IFDIR, S_IFLNK, S_IFREG};

/// No `:` and no URL scheme, so the init's `prepare_apkovl` takes it as a plain path in the
/// initramfs rather than `device:path`.
const CMDLINE: &str = "apkovl=/bootstrap.apkovl.tar.gz";
const OVERLAY_PATH: &str = "bootstrap.apkovl.tar.gz";
/// The script carries the password hashes.
const OVERLAY_MODE: u32 = S_IFREG | 0o600;
const ANSWERS_PATH: &str = "etc/bootstrap/answers";
const SCRIPT_PATH: &str = "etc/local.d/bootstrap.start";
/// OpenRC's `local` service runs every executable `/etc/local.d/*.start`; 0700 is executable
/// for root, which runs it, and keeps the hashes from everyone else.
const SCRIPT_MODE: u32 = S_IFREG | 0o700;
const DIRECTORY_MODE: u32 = S_IFDIR | 0o755;
const PRIVATE_DIRECTORY_MODE: u32 = S_IFDIR | 0o700;
const SYMLINK_MODE: u32 = S_IFLNK | 0o777;
/// The installed system's first disk (virtio-blk).
const DISK: &str = "/dev/vda";
/// The netboot repositories say `latest-stable` (`quick.rs`), which `setup-apkrepos` does not
/// recognize as a release and turns into `edge`: the installed system then upgrades itself to
/// edge from the mixed repositories (seen on a real install). Naming the running release
/// (`v3.24` from `3.24.4`) first makes it pick that one. `alpine-release` is in the netboot world.
const PIN_RELEASE: &str =
    "sed -i \"s|/latest-stable/|/v$(cut -d. -f1,2 /etc/alpine-release)/|\" /etc/apk/repositories";

pub(crate) fn render(answers: &Answers) -> Payload {
    let entry = |path: &str, mode: u32, data: Vec<u8>| TarEntry {
        path: path.into(),
        mode,
        data,
    };
    let overlay = [
        entry("etc/", DIRECTORY_MODE, Vec::new()),
        // An overlay replaces the init's default boot services (devfs, mdev, hwdrivers, modloop
        // and so on) unless this file is present; the init deletes it once it has added them.
        entry("etc/.default_boot_services", S_IFREG | 0o644, Vec::new()),
        entry("etc/bootstrap/", PRIVATE_DIRECTORY_MODE, Vec::new()),
        entry(
            ANSWERS_PATH,
            S_IFREG | 0o600,
            setup_alpine_answers(answers).into_bytes(),
        ),
        entry("etc/local.d/", DIRECTORY_MODE, Vec::new()),
        entry(
            SCRIPT_PATH,
            SCRIPT_MODE,
            install_script(answers).into_bytes(),
        ),
        entry("etc/runlevels/", DIRECTORY_MODE, Vec::new()),
        entry("etc/runlevels/default/", DIRECTORY_MODE, Vec::new()),
        entry(
            "etc/runlevels/default/local",
            SYMLINK_MODE,
            b"/etc/init.d/local".to_vec(),
        ),
    ];
    Payload {
        files: vec![CpioFile {
            path: OVERLAY_PATH.into(),
            mode: OVERLAY_MODE,
            data: tar_gz(&overlay),
        }],
        cmdline: CMDLINE.into(),
    }
}

/// The variables `setup-alpine -f` sources. Validation keeps every value free of quotes and
/// spaces. The disk and the user are `none` because the script sets those up itself on the
/// installed system; SSH is too, so no prompt about root logins comes up.
fn setup_alpine_answers(answers: &Answers) -> String {
    let Answers {
        hostname,
        timezone,
        keymap,
        ..
    } = answers;
    format!(
        "# uemulator quick bootstrap: setup-alpine answers, sourced by setup-alpine -e -f from\n\
         # /etc/local.d/bootstrap.start. The disk and the user are set up by that script instead.\n\
         KEYMAPOPTS=\"{keymap} {keymap}\"\n\
         HOSTNAMEOPTS=\"{hostname}\"\n\
         DEVDOPTS=\"mdev\"\n\
         INTERFACESOPTS=\"auto lo\n\
         iface lo inet loopback\n\
         \n\
         auto eth0\n\
         iface eth0 inet dhcp\n\
         \thostname {hostname}\n\
         \"\n\
         TIMEZONEOPTS=\"{timezone}\"\n\
         PROXYOPTS=\"none\"\n\
         APKREPOSOPTS=\"-1 -c\"\n\
         USEROPTS=\"none\"\n\
         SSHDOPTS=\"none\"\n\
         NTPOPTS=\"busybox\"\n\
         DISKOPTS=\"none\"\n\
         LBUOPTS=\"none\"\n\
         APKCACHEOPTS=\"none\"\n"
    )
}

fn install_script(answers: &Answers) -> String {
    let user = &answers.user;
    let mut out = format!(
        "#!/bin/sh\n\
         # uemulator quick bootstrap: the netboot system's local service runs this once. It installs\n\
         # Alpine Linux to {DISK}, sets up the user on the installed system and reboots into it.\n\
         set -e\n\
         exec > /dev/console 2>&1\n\
         # Any failure (set -e) reboots without the completion marker, which ends the host's run\n\
         # with an error instead of leaving the netboot system at a login prompt.\n\
         trap reboot EXIT\n\
         \n\
         apk add alpine-conf e2fsprogs\n\
         {PIN_RELEASE}\n\
         setup-alpine -e -f /{ANSWERS_PATH}\n"
    );
    // After setup-alpine, whose setup-apkrepos adds the community repository. setup-disk
    // installs the live system's world into the new one, so these land there too.
    let mut packages = String::from("doas");
    if !answers.ssh_keys.is_empty() {
        packages.push_str(" openssh");
    }
    for package in &answers.packages {
        let _ = write!(packages, " {package}");
    }
    let _ = writeln!(out, "apk add {packages}");
    // ERASE_DISKS answers setup-disk's "erase the disk?" question.
    let _ = write!(
        out,
        "ERASE_DISKS={DISK} setup-disk -m sys {DISK}\n\
         \n\
         # setup-disk unmounts the new system when it is done; its root is the partition with\n\
         # /etc/alpine-release (the others are the boot or EFI partition and swap).\n\
         for partition in {DISK}?*; do\n\
         \tmount \"$partition\" /mnt 2>/dev/null || continue\n\
         \t[ -f /mnt/etc/alpine-release ] && break\n\
         \tumount /mnt\n\
         done\n\
         [ -f /mnt/etc/alpine-release ]\n\
         \n"
    );
    // What setup-user -a would do, done on the installed system: setup-disk copies only /etc
    // from the live one, not a home directory.
    let _ = writeln!(out, "chroot /mnt adduser -D {user}");
    let _ = writeln!(out, "chroot /mnt addgroup {user} wheel");
    out.push_str("mkdir -p /mnt/etc/doas.d\n");
    out.push_str("printf '%s\\n' 'permit persist :wheel' > /mnt/etc/doas.d/20-wheel.conf\n");
    let set_password = |out: &mut String, account: &str, hash: &str| {
        let _ = writeln!(
            out,
            "printf '%s\\n' {} | chroot /mnt chpasswd -e",
            shell_quote(&format!("{account}:{hash}"))
        );
    };
    set_password(&mut out, user, &answers.password_hash);
    match &answers.root_password_hash {
        Some(root_hash) => set_password(&mut out, "root", root_hash),
        // setup-alpine -e left root with an empty password, which setup-disk copied.
        None => out.push_str("chroot /mnt passwd -l root\n"),
    }
    if !answers.ssh_keys.is_empty() {
        let ssh_dir = format!("/mnt/home/{user}/.ssh");
        let keys: Vec<String> = answers
            .ssh_keys
            .iter()
            .map(|key| shell_quote(key))
            .collect();
        let _ = writeln!(out, "mkdir -p {ssh_dir}");
        let _ = writeln!(
            out,
            "printf '%s\\n' {} > {ssh_dir}/authorized_keys",
            keys.join(" ")
        );
        let _ = writeln!(out, "chmod 700 {ssh_dir}");
        let _ = writeln!(out, "chmod 600 {ssh_dir}/authorized_keys");
        let _ = writeln!(out, "chroot /mnt chown -R {user}:{user} /home/{user}/.ssh");
        out.push_str("ln -sf /etc/init.d/sshd /mnt/etc/runlevels/default/sshd\n");
    }
    let _ = write!(
        out,
        "\n\
         # setup-disk copied the live /etc, this script and its runlevel link included: without this the\n\
         # installed system would install itself again on every boot.\n\
         rm -rf /mnt/{SCRIPT_PATH} /mnt/etc/runlevels/default/local /mnt/etc/bootstrap\n\
         umount /mnt\n\
         trap - EXIT\n\
         {PRINT_COMPLETION_MARKER}\n\
         reboot\n"
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::fixtures::{full, minimal};
    use crate::targz::tests::read_tar_gz;
    use crate::{S_IFDIR, S_IFLNK, S_IFREG};

    fn overlay(answers: &Answers) -> Vec<(String, u32, Vec<u8>)> {
        let payload = render(answers);
        assert_eq!(payload.files.len(), 1);
        assert_eq!(payload.files[0].path, "bootstrap.apkovl.tar.gz");
        read_tar_gz(&payload.files[0].data)
    }

    fn text<'a>(entries: &'a [(String, u32, Vec<u8>)], path: &str) -> &'a str {
        let (_, _, data) = entries.iter().find(|(name, _, _)| name == path).unwrap();
        core::str::from_utf8(data).unwrap()
    }

    #[test]
    fn full_answers_match_golden() {
        let entries = overlay(&full());
        assert_eq!(
            text(&entries, "etc/local.d/bootstrap.start"),
            include_str!("../testdata/alpine-full.start")
        );
        assert_eq!(
            text(&entries, "etc/bootstrap/answers"),
            include_str!("../testdata/alpine-full.answers")
        );
    }

    #[test]
    fn minimal_answers_match_golden() {
        let entries = overlay(&minimal());
        assert_eq!(
            text(&entries, "etc/local.d/bootstrap.start"),
            include_str!("../testdata/alpine-minimal.start")
        );
        assert_eq!(
            text(&entries, "etc/bootstrap/answers"),
            include_str!("../testdata/alpine-minimal.answers")
        );
    }

    #[test]
    fn the_installed_release_is_the_netboot_one_not_edge() {
        for answers in [full(), minimal()] {
            let entries = overlay(&answers);
            let script = text(&entries, "etc/local.d/bootstrap.start");
            let pin = script
                .find(&format!("\n{PIN_RELEASE}\n"))
                .expect("the repositories name the netboot release");
            assert!(pin < script.find("\nsetup-alpine ").unwrap());
        }
    }

    #[test]
    fn a_failed_install_reboots_without_the_marker() {
        for answers in [full(), minimal()] {
            let entries = overlay(&answers);
            let script = text(&entries, "etc/local.d/bootstrap.start");
            let trap = script
                .find("\ntrap reboot EXIT\n")
                .expect("a failure must reboot");
            assert!(
                trap < script.find("\napk add").unwrap(),
                "trap before the first step"
            );
            let disarmed_then_marker = format!(
                "\ntrap - EXIT\n{}\nreboot\n",
                crate::answers::PRINT_COMPLETION_MARKER
            );
            assert!(
                script.ends_with(&disarmed_then_marker),
                "success disarms the trap right before the marker"
            );
        }
    }

    #[test]
    fn overlay_layout_is_what_the_netboot_init_needs() {
        let entries = overlay(&minimal());
        let layout: Vec<(&str, u32)> = entries
            .iter()
            .map(|(path, mode, _)| (path.as_str(), *mode))
            .collect();
        assert_eq!(
            layout,
            [
                ("etc/", S_IFDIR | 0o755),
                ("etc/.default_boot_services", S_IFREG | 0o644),
                ("etc/bootstrap/", S_IFDIR | 0o700),
                ("etc/bootstrap/answers", S_IFREG | 0o600),
                ("etc/local.d/", S_IFDIR | 0o755),
                ("etc/local.d/bootstrap.start", S_IFREG | 0o700),
                ("etc/runlevels/", S_IFDIR | 0o755),
                ("etc/runlevels/default/", S_IFDIR | 0o755),
                ("etc/runlevels/default/local", S_IFLNK | 0o777),
            ]
        );
        assert_eq!(text(&entries, "etc/.default_boot_services"), "");
        assert_eq!(
            text(&entries, "etc/runlevels/default/local"),
            "/etc/init.d/local"
        );
    }
}
