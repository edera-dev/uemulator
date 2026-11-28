//! Fedora: a kickstart Anaconda reads from `/ks.cfg` in the initramfs (`inst.ks=file:`).
//! Every command Anaconda would otherwise stop and ask for is here (language, keyboard, time
//! zone, root, user, partitioning), and `text` plus `reboot` keep it off the GUI and away from
//! its final "Reboot" prompt.

use std::fmt::Write;

use crate::answers::PRINT_COMPLETION_MARKER;
use crate::{Answers, CpioFile, Payload, S_IFREG};

const CMDLINE: &str = "inst.ks=file:/ks.cfg";
const KICKSTART_PATH: &str = "ks.cfg";
/// It carries a password hash.
const KICKSTART_MODE: u32 = S_IFREG | 0o600;
/// The same tree the CLI's `quick::Distro::Fedora` names in `inst.repo=`; bump them together,
/// and `EVERYTHING_REPOSITORY` with them.
const REPOSITORY: &str =
    "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/";
/// The release's full package set. The Server tree above holds only what Server installs, so a
/// package asked for beyond it (htop, for one) needs this as well.
const EVERYTHING_REPOSITORY: &str =
    "https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Everything/x86_64/os/";
/// The installed system's first disk (virtio-blk), without `/dev/` as kickstart wants it.
const DISK: &str = "vda";

pub(crate) fn render(answers: &Answers) -> Payload {
    Payload {
        files: vec![CpioFile {
            path: KICKSTART_PATH.into(),
            mode: KICKSTART_MODE,
            data: kickstart(answers).into_bytes(),
        }],
        cmdline: CMDLINE.into(),
    }
}

fn kickstart(answers: &Answers) -> String {
    let Answers {
        user,
        password_hash,
        hostname,
        timezone,
        locale,
        keymap,
        ..
    } = answers;
    let mut out = String::from(
        "# uemulator quick bootstrap: Anaconda reads this from /ks.cfg in the initramfs\n\
         # (inst.ks=file:/ks.cfg on the kernel command line).\n\
         text\n",
    );
    let _ = writeln!(out, "lang {locale}");
    let _ = writeln!(out, "keyboard --vckeymap={keymap}");
    let _ = writeln!(out, "timezone {timezone} --utc");
    let _ = writeln!(out, "network --bootproto=dhcp --hostname={hostname}");
    let _ = writeln!(out, "url --url={REPOSITORY}");
    // Only then: its metadata is a long download in an emulator.
    if !answers.packages.is_empty() {
        let _ = writeln!(
            out,
            "repo --name=everything --baseurl={EVERYTHING_REPOSITORY}"
        );
    }
    // Validation keeps hashes to `./$=` and alphanumerics, which kickstart's shlex-style parser
    // takes literally.
    let _ = writeln!(
        out,
        "user --name={user} --groups=wheel --iscrypted --password={password_hash}"
    );
    match &answers.root_password_hash {
        Some(root_hash) => {
            let _ = writeln!(out, "rootpw --iscrypted {root_hash}");
        }
        None => out.push_str("rootpw --lock\n"),
    }
    for key in &answers.ssh_keys {
        let _ = writeln!(out, "sshkey --username={user} {}", kickstart_quote(key));
    }
    if !answers.ssh_keys.is_empty() {
        out.push_str("services --enabled=sshd\n");
    }
    out.push_str("firstboot --disable\n");
    // autopart makes the EFI system or BIOS boot partition the firmware needs by itself.
    let _ = writeln!(out, "ignoredisk --only-use={DISK}");
    out.push_str("zerombr\n");
    let _ = writeln!(out, "clearpart --all --initlabel --drives={DISK}");
    out.push_str("autopart\n");
    out.push_str("reboot\n");

    out.push_str("\n%packages\n");
    if !answers.ssh_keys.is_empty() {
        out.push_str("openssh-server\n");
    }
    for package in &answers.packages {
        let _ = writeln!(out, "{package}");
    }
    out.push_str("%end\n");

    // The last script Anaconda runs, after the bootloader is in; --nochroot so /dev/console is
    // the installer's console rather than the new system's.
    let _ = write!(out, "\n%post --nochroot\n{PRINT_COMPLETION_MARKER}\n%end\n");
    out
}

/// Pykickstart splits a line with Python's `shlex` (POSIX mode), where a backslash escapes only
/// `\` and `"` inside double quotes, and `'` is literal there.
fn kickstart_quote(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::fixtures::{full, minimal};

    fn kickstart(answers: &Answers) -> String {
        let payload = render(answers);
        String::from_utf8(payload.files[0].data.clone()).unwrap()
    }

    #[test]
    fn full_answers_match_golden() {
        assert_eq!(
            kickstart(&full()),
            include_str!("../testdata/fedora-full.ks")
        );
    }

    #[test]
    fn minimal_answers_match_golden() {
        assert_eq!(
            kickstart(&minimal()),
            include_str!("../testdata/fedora-minimal.ks")
        );
    }

    /// htop is not in the Server tree the installer boots from: Anaconda stopped at "No match
    /// for argument: htop" on a real install.
    #[test]
    fn extra_packages_also_come_from_everything() {
        let repo = format!("\nrepo --name=everything --baseurl={EVERYTHING_REPOSITORY}\n");
        assert!(kickstart(&full()).contains(&repo));
        assert!(!kickstart(&minimal()).contains("repo "));
    }

    #[test]
    fn kickstart_quote_escapes_backslash_and_double_quote() {
        assert_eq!(kickstart_quote(r#"a\b "c" 'd'"#), r#""a\\b \"c\" 'd'""#);
    }
}
