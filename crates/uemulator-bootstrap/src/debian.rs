//! Debian: a preseed file d-i reads from `/preseed.cfg` at the initramfs root by itself, and a
//! script its `preseed/late_command` runs once the install has succeeded.
//!
//! `priority=critical` means d-i asks only critical questions, and the preseed answers every one
//! it asks on the default path (including the partitioning confirmations, which are critical).
//! A question it misses stays on the console; that is a bug here.

use std::fmt::Write;

use crate::answers::{PRINT_COMPLETION_MARKER, shell_quote};
use crate::{Answers, CpioFile, Payload, S_IFREG};

/// `auto=true` defers the locale and keyboard questions until the preseed is loaded, which for a
/// file in the initramfs is at once.
const CMDLINE: &str = "auto=true priority=critical";
const PRESEED_PATH: &str = "preseed.cfg";
/// d-i runs from its initramfs without switching root, so this is at `/` when the late command
/// runs. The late command runs it rather than holding the commands itself: preseed values go
/// through debconf, and keeping backslashes and quotes out of them avoids its escaping.
const LATE_SCRIPT_PATH: &str = "bootstrap-late.sh";
/// Both carry a password hash.
const FILE_MODE: u32 = S_IFREG | 0o600;
const LATE_SCRIPT_MODE: u32 = S_IFREG | 0o700;
/// The installed system's first disk (virtio-blk).
const DISK: &str = "/dev/vda";
/// d-i's default groups for the first user plus `sudo`. user-setup only adds `sudo` itself when
/// root is locked, and the spec makes the user an admin either way (the group comes from
/// base-passwd, so it exists before pkgsel installs sudo).
const USER_GROUPS: &str = "audio cdrom dip floppy video plugdev netdev sudo";

pub(crate) fn render(answers: &Answers) -> Payload {
    Payload {
        files: vec![
            CpioFile {
                path: PRESEED_PATH.into(),
                mode: FILE_MODE,
                data: preseed(answers).into_bytes(),
            },
            CpioFile {
                path: LATE_SCRIPT_PATH.into(),
                mode: LATE_SCRIPT_MODE,
                data: late_script(answers).into_bytes(),
            },
        ],
        cmdline: CMDLINE.into(),
    }
}

fn preseed(answers: &Answers) -> String {
    let Answers {
        user,
        password_hash,
        hostname,
        timezone,
        locale,
        keymap,
        ..
    } = answers;
    let mut out = String::new();
    let mut line = |text: &str| {
        out.push_str(text);
        out.push('\n');
    };
    line(
        "# uemulator quick bootstrap: the Debian installer reads this from /preseed.cfg in the initramfs,",
    );
    line("# with auto=true priority=critical on the kernel command line.");
    line(&format!("d-i debian-installer/locale string {locale}"));
    line(&format!(
        "d-i keyboard-configuration/xkb-keymap select {keymap}"
    ));
    line("d-i netcfg/choose_interface select auto");
    line(&format!("d-i netcfg/get_hostname string {hostname}"));
    line("d-i netcfg/get_domain string");
    // Wins over a name the DHCP server hands out.
    line(&format!("d-i netcfg/hostname string {hostname}"));
    line("d-i hw-detect/load_firmware boolean true");
    line("d-i mirror/country string manual");
    line("d-i mirror/http/hostname string deb.debian.org");
    line("d-i mirror/http/directory string /debian");
    line("d-i mirror/http/proxy string");
    match &answers.root_password_hash {
        Some(root_hash) => {
            line("d-i passwd/root-login boolean true");
            line(&format!(
                "d-i passwd/root-password-crypted password {root_hash}"
            ));
        }
        None => line("d-i passwd/root-login boolean false"),
    }
    line(&format!("d-i passwd/user-fullname string {user}"));
    line(&format!("d-i passwd/username string {user}"));
    line(&format!(
        "d-i passwd/user-password-crypted password {password_hash}"
    ));
    line(&format!(
        "d-i passwd/user-default-groups string {USER_GROUPS}"
    ));
    line("d-i clock-setup/utc boolean true");
    line(&format!("d-i time/zone string {timezone}"));
    line("d-i clock-setup/ntp boolean true");
    line(&format!("d-i partman-auto/disk string {DISK}"));
    line("d-i partman-auto/method string regular");
    line("d-i partman-auto/choose_recipe select atomic");
    line("d-i partman-lvm/device_remove_lvm boolean true");
    line("d-i partman-md/device_remove_md boolean true");
    line("d-i partman-lvm/confirm boolean true");
    line("d-i partman-lvm/confirm_nooverwrite boolean true");
    line("d-i partman-efi/non_efi_system boolean true");
    line("d-i partman-partitioning/confirm_write_new_label boolean true");
    line("d-i partman/choose_partition select finish");
    line("d-i partman/confirm boolean true");
    line("d-i partman/confirm_nooverwrite boolean true");
    line("d-i apt-setup/non-free-firmware boolean true");
    line("d-i apt-setup/services-select multiselect security, updates");
    line(if answers.ssh_keys.is_empty() {
        "tasksel tasksel/first multiselect standard"
    } else {
        "tasksel tasksel/first multiselect standard, ssh-server"
    });
    let mut include = String::from("sudo");
    for package in &answers.packages {
        let _ = write!(include, " {package}");
    }
    line(&format!("d-i pkgsel/include string {include}"));
    line("popularity-contest popularity-contest/participate boolean false");
    line("d-i grub-installer/only_debian boolean true");
    // grub-installer picks grub-efi or grub-pc from how d-i itself was booted, so one preseed
    // fits both firmwares. The extra removable copy (EFI/BOOT/BOOTX64.EFI) boots even when the
    // firmware has no boot entry for Debian.
    line(&format!("d-i grub-installer/bootdev string {DISK}"));
    line("d-i grub-installer/force-efi-extra-removable boolean true");
    line(&format!(
        "d-i preseed/late_command string sh /{LATE_SCRIPT_PATH}"
    ));
    line("d-i finish-install/reboot_in_progress note");
    out
}

fn late_script(answers: &Answers) -> String {
    let user = &answers.user;
    let mut out = String::from(
        "#!/bin/sh\n\
         # uemulator quick bootstrap: the Debian installer's preseed/late_command runs this once the\n\
         # install has succeeded, with the new system mounted at /target.\n\
         set -e\n",
    );
    if !answers.ssh_keys.is_empty() {
        let ssh_dir = format!("/target/home/{user}/.ssh");
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
        let _ = writeln!(out, "in-target chown -R {user}:{user} /home/{user}/.ssh");
    }
    let _ = writeln!(out, "{PRINT_COMPLETION_MARKER}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::fixtures::{full, minimal};

    fn file<'a>(payload: &'a Payload, path: &str) -> &'a str {
        let file = payload.files.iter().find(|file| file.path == path).unwrap();
        core::str::from_utf8(&file.data).unwrap()
    }

    #[test]
    fn full_answers_match_golden() {
        let payload = render(&full());
        assert_eq!(
            file(&payload, "preseed.cfg"),
            include_str!("../testdata/debian-full.cfg")
        );
        assert_eq!(
            file(&payload, "bootstrap-late.sh"),
            include_str!("../testdata/debian-full.late.sh")
        );
    }

    #[test]
    fn minimal_answers_match_golden() {
        let payload = render(&minimal());
        assert_eq!(
            file(&payload, "preseed.cfg"),
            include_str!("../testdata/debian-minimal.cfg")
        );
        assert_eq!(
            file(&payload, "bootstrap-late.sh"),
            include_str!("../testdata/debian-minimal.late.sh")
        );
    }
}
