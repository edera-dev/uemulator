//! The answers a person gives `quick bootstrap`, their validation, and the dispatch to the
//! per-distro renderers.

use crate::CpioFile;

/// The line an install prints on the console once it has succeeded, just before it reboots.
/// The rendered files never hold it verbatim (they print it as [`PRINT_COMPLETION_MARKER`]
/// builds it), so the console echoing an answer file or a command line back cannot fake it.
pub const COMPLETION_MARKER: &str = "uemulator-bootstrap: installed";

/// The shell command every renderer ends the install with. Matches [`COMPLETION_MARKER`]; the
/// tests hold the two together.
pub(crate) const PRINT_COMPLETION_MARKER: &str =
    "printf '%s: %s\\n' uemulator-bootstrap installed > /dev/console";

/// The distros `quick bootstrap` can install, the same three as the CLI's `quick::Distro`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distro {
    Debian,
    Alpine,
    Fedora,
}

/// What a person tells `quick bootstrap`. Passwords are already hashed (`$6$`, see
/// [`crate::sha512_crypt`]); the plaintext never gets here.
pub struct Answers {
    /// A regular account in the distro's admin group (`sudo` on Debian, `wheel` elsewhere).
    pub user: String,
    pub password_hash: String,
    /// `None` locks root; the user's sudo or doas is then the way in.
    pub root_password_hash: Option<String>,
    pub hostname: String,
    /// One OpenSSH public key line each. Any key installs and enables the SSH server.
    pub ssh_keys: Vec<String>,
    /// IANA name, such as `Europe/Berlin`.
    pub timezone: String,
    /// Such as `en_US.UTF-8`. Alpine ignores it (musl has no locales to set up).
    pub locale: String,
    pub keymap: String,
    /// Distro package names, added by the installer.
    pub packages: Vec<String>,
}

/// Linux's `LOGIN_NAME_MAX` is 256, but `useradd` and the installers stop at 32.
const USER_NAME_MAX: usize = 32;
/// RFC 1123 label length.
const HOSTNAME_MAX: usize = 63;
const PASSWORD_HASH_PREFIX: &str = "$6$";

impl Answers {
    /// Checks every field against what the answer files can carry without quoting. The values
    /// end up in shell scripts, a kickstart and a preseed, so this is what keeps one answer from
    /// becoming a command; renderers rely on it. Errors name the field but never echo a hash.
    pub fn validate(&self) -> Result<(), String> {
        validate_user(&self.user)?;
        validate_password_hash("password hash", &self.password_hash)?;
        if let Some(hash) = &self.root_password_hash {
            validate_password_hash("root password hash", hash)?;
        }
        validate_hostname(&self.hostname)?;
        for key in &self.ssh_keys {
            // Quoted where it is used; only a line break (or any control character) could
            // escape that quoting or split the key across lines of authorized_keys.
            if key.is_empty() || key.chars().any(char::is_control) {
                return Err(format!(
                    "ssh key {key:?}: must be one non-empty line with no control characters"
                ));
            }
        }
        validate_setting("timezone", &self.timezone)?;
        validate_setting("locale", &self.locale)?;
        validate_setting("keymap", &self.keymap)?;
        for package in &self.packages {
            let allowed = |c: char| c.is_ascii_alphanumeric() || "+._-".contains(c);
            // A leading `-` would be read as an option by apt, dnf or apk.
            if package.is_empty() || package.starts_with('-') || !package.chars().all(allowed) {
                return Err(format!(
                    "package {package:?}: must be a package name of A-Z, a-z, 0-9 and + . _ -"
                ));
            }
        }
        Ok(())
    }
}

fn validate_user(user: &str) -> Result<(), String> {
    let mut characters = user.chars();
    let first_ok = characters
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_');
    let rest_ok =
        characters.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "_-".contains(c));
    if user == "root" {
        return Err("user: must not be root; root's password is a separate answer".into());
    }
    if !first_ok || !rest_ok || user.len() > USER_NAME_MAX {
        return Err(format!(
            "user {user:?}: must be 1 to {USER_NAME_MAX} characters of a-z, 0-9, _ and -, \
             starting with a letter or _"
        ));
    }
    Ok(())
}

/// Crypt hashes use `./0-9A-Za-z` plus `$` separators and an optional `rounds=N$`; anything
/// else is not a hash this crate made and could break out of the files' quoting.
fn validate_password_hash(field: &str, hash: &str) -> Result<(), String> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || "./$=".contains(c);
    if !hash.starts_with(PASSWORD_HASH_PREFIX) || !hash.chars().all(allowed) {
        return Err(format!("{field}: must be a SHA-512 crypt hash ($6$...)"));
    }
    Ok(())
}

fn validate_hostname(hostname: &str) -> Result<(), String> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || c == '-';
    if hostname.is_empty()
        || hostname.len() > HOSTNAME_MAX
        || hostname.starts_with('-')
        || hostname.ends_with('-')
        || !hostname.chars().all(allowed)
    {
        return Err(format!(
            "hostname {hostname:?}: must be 1 to {HOSTNAME_MAX} characters of A-Z, a-z, 0-9 \
             and -, not starting or ending with - (an RFC 1123 label)"
        ));
    }
    Ok(())
}

/// Timezone, locale and keymap names: `Europe/Berlin`, `Etc/GMT+5`, `sr_RS.UTF-8@latin`,
/// `de-latin1`.
fn validate_setting(field: &str, value: &str) -> Result<(), String> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || "_+./@-".contains(c);
    if value.is_empty() || !value.chars().all(allowed) {
        return Err(format!(
            "{field} {value:?}: must be A-Z, a-z, 0-9 and _ + . / @ - only"
        ));
    }
    Ok(())
}

/// The files to append to the installer's initramfs and the kernel arguments to add after the
/// distro's own installer arguments.
pub struct Payload {
    pub files: Vec<CpioFile>,
    pub cmdline: String,
}

/// Validates `answers`, then renders the distro's answer files.
pub fn render(distro: Distro, answers: &Answers) -> Result<Payload, String> {
    answers.validate()?;
    Ok(match distro {
        Distro::Debian => crate::debian::render(answers),
        Distro::Fedora => crate::fedora::render(answers),
        Distro::Alpine => crate::alpine::render(answers),
    })
}

/// Quotes `text` as one POSIX shell word: single quotes, with each `'` closed, escaped and
/// reopened.
pub(crate) fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::Answers;

    pub(crate) fn full() -> Answers {
        Answers {
            user: "alex".into(),
            password_hash: "$6$testsalt$fakeuserhash".into(),
            root_password_hash: Some("$6$testsalt$fakeroothash".into()),
            hostname: "devbox".into(),
            ssh_keys: vec![
                "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyOne alex@laptop".into(),
                // Both quote characters, to exercise shell and kickstart quoting.
                "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABFakeKeyTwo it's \"mine\"".into(),
            ],
            timezone: "Europe/Berlin".into(),
            locale: "de_DE.UTF-8".into(),
            keymap: "de".into(),
            packages: vec!["git".into(), "htop".into()],
        }
    }

    pub(crate) fn minimal() -> Answers {
        Answers {
            user: "alex".into(),
            password_hash: "$6$testsalt$fakeuserhash".into(),
            root_password_hash: None,
            hostname: "box".into(),
            ssh_keys: Vec::new(),
            timezone: "UTC".into(),
            locale: "en_US.UTF-8".into(),
            keymap: "us".into(),
            packages: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{full, minimal};
    use super::*;
    use crate::targz::tests::read_tar_gz;

    const ALL_DISTROS: [Distro; 3] = [Distro::Debian, Distro::Alpine, Distro::Fedora];

    fn rejected(answers: &Answers, field: &str) {
        let error = answers.validate().expect_err("should be rejected");
        assert!(error.contains(field), "{error:?} does not name {field}");
    }

    /// Every file the guest sees, the Alpine overlay unpacked: (path, contents).
    fn rendered_texts(distro: Distro, answers: &Answers) -> Vec<(String, String)> {
        let payload = render(distro, answers).unwrap();
        let mut texts = Vec::new();
        for file in &payload.files {
            if file.path.ends_with(".tar.gz") {
                for (path, _, data) in read_tar_gz(&file.data) {
                    texts.push((path, String::from_utf8(data).unwrap()));
                }
            } else {
                texts.push((
                    file.path.clone(),
                    String::from_utf8(file.data.clone()).unwrap(),
                ));
            }
        }
        texts
    }

    fn all_text(distro: Distro, answers: &Answers) -> String {
        rendered_texts(distro, answers)
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn fixtures_are_valid() {
        full().validate().unwrap();
        minimal().validate().unwrap();
    }

    #[test]
    fn rejects_bad_users() {
        for user in [
            "",
            "root",
            "Alex",
            "1alex",
            "al ex",
            "alex;rm",
            &"a".repeat(33),
        ] {
            rejected(
                &Answers {
                    user: user.into(),
                    ..minimal()
                },
                "user",
            );
        }
        for user in ["_svc", "a-b_c9", &"a".repeat(32)] {
            Answers {
                user: user.into(),
                ..minimal()
            }
            .validate()
            .unwrap();
        }
    }

    #[test]
    fn rejects_bad_hostnames() {
        for hostname in [
            "",
            "-box",
            "box-",
            "my_box",
            "box.local",
            "b x",
            &"a".repeat(64),
        ] {
            rejected(
                &Answers {
                    hostname: hostname.into(),
                    ..minimal()
                },
                "hostname",
            );
        }
        for hostname in ["Box-1", "9", &"a".repeat(63)] {
            Answers {
                hostname: hostname.into(),
                ..minimal()
            }
            .validate()
            .unwrap();
        }
    }

    #[test]
    fn rejects_bad_packages() {
        for package in ["", "vim htop", "a;b", "$(x)", "-f", "a'b"] {
            rejected(
                &Answers {
                    packages: vec![package.into()],
                    ..minimal()
                },
                "package",
            );
        }
        Answers {
            packages: vec!["g++".into(), "python3.12-dev".into(), "lib_x".into()],
            ..minimal()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn rejects_bad_timezone_locale_and_keymap() {
        for bad in ["", "a b", "x\"y", "a;b", "a'b", "$x"] {
            rejected(
                &Answers {
                    timezone: bad.into(),
                    ..minimal()
                },
                "timezone",
            );
            rejected(
                &Answers {
                    locale: bad.into(),
                    ..minimal()
                },
                "locale",
            );
            rejected(
                &Answers {
                    keymap: bad.into(),
                    ..minimal()
                },
                "keymap",
            );
        }
        Answers {
            timezone: "America/Argentina/Buenos_Aires".into(),
            locale: "sr_RS.UTF-8@latin".into(),
            keymap: "de-latin1".into(),
            ..minimal()
        }
        .validate()
        .unwrap();
        Answers {
            timezone: "Etc/GMT+5".into(),
            ..minimal()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn rejects_multi_line_ssh_keys() {
        for key in [
            "ssh-ed25519 AAAA\nrm -rf /",
            "ssh-ed25519 AAAA\r",
            "",
            "ssh-ed25519 \0",
        ] {
            rejected(
                &Answers {
                    ssh_keys: vec![key.into()],
                    ..minimal()
                },
                "ssh key",
            );
        }
    }

    #[test]
    fn rejects_hashes_that_are_not_sha512_crypt() {
        for hash in [
            "",
            "plaintext",
            "$1$salt$md5",
            "$5$salt$sha256",
            "$6$salt$x y",
            "$6$s'alt$x",
        ] {
            rejected(
                &Answers {
                    password_hash: hash.into(),
                    ..minimal()
                },
                "password hash",
            );
            rejected(
                &Answers {
                    root_password_hash: Some(hash.into()),
                    ..minimal()
                },
                "root password hash",
            );
        }
        Answers {
            password_hash: "$6$rounds=10000$salt$abc./XYZ".into(),
            ..minimal()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn render_validates_first() {
        let answers = Answers {
            user: "root".into(),
            ..minimal()
        };
        for distro in ALL_DISTROS {
            assert!(render(distro, &answers).is_err());
        }
    }

    #[test]
    fn no_output_contains_the_marker_verbatim() {
        for distro in ALL_DISTROS {
            for answers in [full(), minimal()] {
                let payload = render(distro, &answers).unwrap();
                for file in &payload.files {
                    assert!(
                        !file
                            .data
                            .windows(COMPLETION_MARKER.len())
                            .any(|window| window == COMPLETION_MARKER.as_bytes())
                    );
                }
                let text = all_text(distro, &answers);
                assert!(!text.contains(COMPLETION_MARKER), "{distro:?}");
                assert!(
                    text.contains(
                        "printf '%s: %s\\n' uemulator-bootstrap installed > /dev/console"
                    ),
                    "{distro:?} never prints the marker"
                );
            }
        }
    }

    #[test]
    fn root_locked_when_no_root_password() {
        let debian = all_text(Distro::Debian, &minimal());
        assert!(debian.contains("d-i passwd/root-login boolean false\n"));
        assert!(!debian.contains("root-password"));

        let fedora = all_text(Distro::Fedora, &minimal());
        assert!(fedora.contains("\nrootpw --lock\n"));

        let alpine = all_text(Distro::Alpine, &minimal());
        assert!(alpine.contains("\nchroot /mnt passwd -l root\n"));
        assert!(!alpine.contains("'root:"));
    }

    #[test]
    fn root_password_hash_is_used_when_given() {
        for distro in ALL_DISTROS {
            assert!(
                all_text(distro, &full()).contains("$6$testsalt$fakeroothash"),
                "{distro:?}"
            );
        }
        assert!(all_text(Distro::Debian, &full()).contains("d-i passwd/root-login boolean true\n"));
        assert!(!all_text(Distro::Fedora, &full()).contains("--lock"));
        assert!(!all_text(Distro::Alpine, &full()).contains("passwd -l root"));
    }

    #[test]
    fn ssh_key_enables_openssh() {
        let with_key = |distro| {
            all_text(
                distro,
                &Answers {
                    ssh_keys: vec!["ssh-ed25519 AAAA k".into()],
                    ..minimal()
                },
            )
        };

        let debian = with_key(Distro::Debian);
        assert!(debian.contains("tasksel tasksel/first multiselect standard, ssh-server\n"));
        assert!(debian.contains("'ssh-ed25519 AAAA k'"));
        assert!(!all_text(Distro::Debian, &minimal()).contains("ssh-server"));

        let fedora = with_key(Distro::Fedora);
        assert!(fedora.contains("\nopenssh-server\n"));
        assert!(fedora.contains("\nservices --enabled=sshd\n"));
        assert!(fedora.contains("sshkey --username=alex \"ssh-ed25519 AAAA k\"\n"));
        assert!(!all_text(Distro::Fedora, &minimal()).contains("sshd"));

        let alpine = with_key(Distro::Alpine);
        assert!(alpine.contains("\napk add doas openssh\n"));
        assert!(alpine.contains("ln -sf /etc/init.d/sshd /mnt/etc/runlevels/default/sshd\n"));
        assert!(alpine.contains("'ssh-ed25519 AAAA k'"));
        assert!(!all_text(Distro::Alpine, &minimal()).contains("ssh"));
    }

    #[test]
    fn cmdline_has_no_stray_whitespace() {
        for distro in ALL_DISTROS {
            for answers in [full(), minimal()] {
                let cmdline = render(distro, &answers).unwrap().cmdline;
                assert!(!cmdline.is_empty());
                assert_eq!(cmdline.trim(), cmdline);
                assert!(!cmdline.contains("  "));
                assert!(!cmdline.contains(['\n', '\t']));
            }
        }
        assert_eq!(
            render(Distro::Debian, &minimal()).unwrap().cmdline,
            "auto=true priority=critical"
        );
        assert_eq!(
            render(Distro::Fedora, &minimal()).unwrap().cmdline,
            "inst.ks=file:/ks.cfg"
        );
        assert_eq!(
            render(Distro::Alpine, &minimal()).unwrap().cmdline,
            "apkovl=/bootstrap.apkovl.tar.gz"
        );
    }

    #[test]
    fn payload_files_are_where_the_cmdline_points() {
        let path_of = |distro| {
            render(distro, &minimal())
                .unwrap()
                .files
                .iter()
                .map(|file| file.path.clone())
                .collect::<Vec<_>>()
        };
        assert!(path_of(Distro::Debian).contains(&"preseed.cfg".to_string()));
        assert_eq!(path_of(Distro::Fedora), ["ks.cfg"]);
        assert_eq!(path_of(Distro::Alpine), ["bootstrap.apkovl.tar.gz"]);
    }

    #[test]
    fn printed_marker_is_the_marker() {
        // printf '%s: %s\n' A B prints "A: B".
        let (name, state) = COMPLETION_MARKER.split_once(": ").unwrap();
        assert_eq!(
            PRINT_COMPLETION_MARKER,
            format!("printf '%s: %s\\n' {name} {state} > /dev/console")
        );
    }

    #[test]
    fn shell_quote_survives_single_quotes() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote("$HOME \"x\""), "'$HOME \"x\"'");
    }
}
