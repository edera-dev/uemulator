//! `quick bootstrap`'s answers on the host side: the host's defaults, reading passwords, and the
//! `[bootstrap]` record a machine keeps so `quick reinstall` can repeat the answers
//! (`docs/plans/35-quick-bootstrap.md`, "Settings" and "Reinstall").

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use uemulator_bootstrap::{Answers, COMPLETION_MARKER, SALT_LEN, salt_from_bytes, sha512_crypt};

use crate::args::{BootstrapArgs, Firmware, HypervisorArgs};
use crate::term;

/// The non-secret answers, saved under `[bootstrap]` in `machine.toml`. It never holds a password
/// or its hash: a reinstall asks for them again.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct BootstrapRecord {
    pub user: String,
    pub hostname: String,
    /// The key lines themselves, not the files they came from, which may be gone by the reinstall.
    #[serde(default)]
    pub ssh_keys: Vec<String>,
    pub timezone: String,
    pub locale: String,
    pub keymap: String,
    #[serde(default)]
    pub packages: Vec<String>,
    /// Whether root got a password, which a reinstall then asks for again.
    #[serde(default)]
    pub root_password: bool,
}

const DEFAULT_TIMEZONE: &str = "UTC";
const DEFAULT_LOCALE: &str = "en_US.UTF-8";
const DEFAULT_KEYMAP: &str = "us";

/// The timezone, locale and keymap of answers that name none, each the host's if it can be read
/// and fits every installer, else its fallback (UTC, en_US.UTF-8, us).
pub fn host_defaults() -> (String, String, String) {
    (
        host_timezone().unwrap_or_else(|| DEFAULT_TIMEZONE.into()),
        host_locale().unwrap_or_else(|| DEFAULT_LOCALE.into()),
        host_keymap().unwrap_or_else(|| DEFAULT_KEYMAP.into()),
    )
}

#[cfg(unix)]
fn host_timezone() -> Option<String> {
    /// A symlink into the zoneinfo database on Linux and macOS alike.
    const LOCALTIME: &str = "/etc/localtime";
    timezone_from_localtime(&std::fs::read_link(LOCALTIME).ok()?)
}

#[cfg(unix)]
fn host_locale() -> Option<String> {
    locale_from_lang(std::env::var("LANG").ok().as_deref())
}

/// macOS keyboard layouts are not console keymaps, so only Linux has one to read.
#[cfg(target_os = "linux")]
fn host_keymap() -> Option<String> {
    const VCONSOLE_CONF: &str = "/etc/vconsole.conf";
    keymap_from_vconsole(&std::fs::read_to_string(VCONSOLE_CONF).ok()?)
}

#[cfg(not(unix))]
fn host_timezone() -> Option<String> {
    None
}

#[cfg(not(unix))]
fn host_locale() -> Option<String> {
    None
}

#[cfg(not(target_os = "linux"))]
fn host_keymap() -> Option<String> {
    None
}

/// The IANA name in the path `/etc/localtime` links to: what follows `zoneinfo/`, without the
/// `posix/` or `right/` subtree some systems link into.
#[cfg(any(unix, test))]
fn timezone_from_localtime(target: &Path) -> Option<String> {
    const ZONEINFO: &str = "zoneinfo/";
    let target = target.to_str()?;
    let name = &target[target.rfind(ZONEINFO)? + ZONEINFO.len()..];
    let name = ["posix/", "right/"]
        .iter()
        .find_map(|subtree| name.strip_prefix(subtree))
        .unwrap_or(name);
    let allowed = |c: char| c.is_ascii_alphanumeric() || "/_+-".contains(c);
    (!name.is_empty() && name.chars().all(allowed)).then(|| name.to_string())
}

/// `LANG` when it names a language and a country (`de_DE.UTF-8`, `sr_RS.UTF-8@latin`). `C` and
/// `POSIX` do not, and the installers' locale lists have no `C.UTF-8`.
#[cfg(any(unix, test))]
fn locale_from_lang(lang: Option<&str>) -> Option<String> {
    let lang = lang?;
    let name_end = lang.find(['.', '@']).unwrap_or(lang.len());
    let (language, country) = lang[..name_end].split_once('_')?;
    let well_formed = (2..=3).contains(&language.len())
        && language.chars().all(|c| c.is_ascii_lowercase())
        && country.len() == 2
        && country.chars().all(|c| c.is_ascii_uppercase())
        && lang[name_end..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || ".@-_".contains(c));
    well_formed.then(|| lang.to_string())
}

/// The plain layout name every installer accepts, as `Answers::validate` checks it: console
/// keymaps such as `de-latin1-nodeadkeys` lose their variant; anything that is not then two or
/// three lowercase letters (`dvorak`, `fr_CH`) is not a layout all three know. Change the two
/// together.
#[cfg(any(target_os = "linux", test))]
fn keymap_layout(keymap: &str) -> Option<String> {
    let layout = keymap.split('-').next()?;
    let plain = (2..=3).contains(&layout.len()) && layout.chars().all(|c| c.is_ascii_lowercase());
    plain.then(|| layout.to_string())
}

/// The layout of `KEYMAP=` in systemd's `/etc/vconsole.conf`, an env-style file.
#[cfg(any(target_os = "linux", test))]
fn keymap_from_vconsole(text: &str) -> Option<String> {
    let value = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("KEYMAP="))?;
    keymap_layout(value.trim().trim_matches(['"', '\'']))
}

/// Which of the two passwords, for messages, prompts and option names.
#[derive(Clone, Copy)]
enum Account {
    User,
    Root,
}

impl Account {
    /// The option stem: `--{stem}`, `--ask-{stem}`, `--{stem}-file`.
    fn option(self) -> &'static str {
        match self {
            Account::User => "password",
            Account::Root => "root-password",
        }
    }

    fn what(self) -> &'static str {
        match self {
            Account::User => "password",
            Account::Root => "root password",
        }
    }
}

enum PasswordSource<'a> {
    Text(&'a str),
    Ask,
    File(&'a Path),
}

impl<'a> PasswordSource<'a> {
    /// The option given, if any; clap lets through at most one.
    fn given(text: &'a Option<String>, ask: bool, file: &'a Option<PathBuf>) -> Option<Self> {
        match (text, ask, file) {
            (Some(text), _, _) => Some(PasswordSource::Text(text)),
            (None, true, _) => Some(PasswordSource::Ask),
            (None, false, Some(path)) => Some(PasswordSource::File(path)),
            (None, false, None) => None,
        }
    }
}

/// The plaintext password, which the caller hashes straight away and drops.
fn read_password(
    source: PasswordSource,
    account: Account,
    prompt_name: &str,
    tty: bool,
    read_secret: &mut impl FnMut(&str) -> io::Result<String>,
) -> Result<String, String> {
    let (option, what) = (account.option(), account.what());
    let password = match source {
        PasswordSource::Text(text) => text.to_string(),
        PasswordSource::File(path) => {
            let mut text = fs::read_to_string(path)
                .map_err(|e| format!("--{option}-file {}: {e}", path.display()))?;
            // `echo secret > file` adds one newline that is not part of the password; anything
            // beyond that one is kept, since a password may end in whitespace.
            if text.ends_with('\n') {
                text.pop();
                if text.ends_with('\r') {
                    text.pop();
                }
            }
            text
        }
        PasswordSource::Ask => {
            if !tty {
                return Err(format!(
                    "--ask-{option} needs a terminal on stdin to ask on: give --{option}-file \
                     FILE instead"
                ));
            }
            let mut ask =
                |prompt: String| read_secret(&prompt).map_err(|e| format!("read the {what}: {e}"));
            let first = ask(format!("Password for {prompt_name}: "))?;
            let again = ask(format!("Password for {prompt_name} again: "))?;
            if first != again {
                return Err(format!("the {what}s did not match"));
            }
            first
        }
    };
    if password.is_empty() {
        return Err(format!("the {what} is empty"));
    }
    Ok(password)
}

/// SHA-512 crypt with a fresh salt from the OS's random source.
fn hash_password(password: &str) -> Result<String, String> {
    let mut random = [0u8; SALT_LEN];
    getrandom::getrandom(&mut random)
        .map_err(|e| format!("random salt for a password hash: {e}"))?;
    Ok(sha512_crypt(password, &salt_from_bytes(&random)))
}

/// The key lines of `--ssh-key` files, comments and blank lines left out. A private key is
/// refused: it would end up in `machine.toml` and the guest.
fn read_ssh_keys(paths: &[PathBuf]) -> Result<Vec<String>, String> {
    const PRIVATE_KEY_MARK: &str = "PRIVATE KEY";
    let mut keys = Vec::new();
    for path in paths {
        let text =
            fs::read_to_string(path).map_err(|e| format!("--ssh-key {}: {e}", path.display()))?;
        if text.contains(PRIVATE_KEY_MARK) {
            return Err(format!(
                "--ssh-key {}: is a private key; give the public one (.pub)",
                path.display()
            ));
        }
        let before = keys.len();
        keys.extend(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(str::to_string),
        );
        if keys.len() == before {
            return Err(format!(
                "--ssh-key {}: has no public key in it",
                path.display()
            ));
        }
    }
    Ok(keys)
}

/// Passes `Answers::validate`'s hash check, so the rest of the answers are checked before anyone
/// is asked for a password; never rendered.
const UNASKED_HASH: &str = "$6$";

/// The answers of one install and the record to save for the next. Each setting is the flag's,
/// else `saved`'s, else the default (the machine name for the host name, `host_defaults` for the
/// rest). Passwords are never saved, so a machine with a record is asked for them again on a
/// terminal (`tty`), through `read_secret`, which gets the prompt; without a terminal they must
/// come from an option.
pub fn resolve_answers(
    args: &BootstrapArgs,
    saved: Option<&BootstrapRecord>,
    machine_name: &str,
    tty: bool,
    mut read_secret: impl FnMut(&str) -> io::Result<String>,
) -> Result<(Answers, BootstrapRecord), String> {
    let user = args
        .user
        .clone()
        .or_else(|| saved.map(|record| record.user.clone()))
        .ok_or("--user is required: the account the install creates")?;
    let hostname_given = args.hostname.is_some() || saved.is_some();
    let hostname = args
        .hostname
        .clone()
        .or_else(|| saved.map(|record| record.hostname.clone()))
        .unwrap_or_else(|| machine_name.to_string());
    let ssh_keys = if args.ssh_key.is_empty() {
        saved
            .map(|record| record.ssh_keys.clone())
            .unwrap_or_default()
    } else {
        read_ssh_keys(&args.ssh_key)?
    };
    let packages = if args.package.is_empty() {
        saved
            .map(|record| record.packages.clone())
            .unwrap_or_default()
    } else {
        args.package.clone()
    };
    let pick = |flag: &Option<String>, from_record: fn(&BootstrapRecord) -> &String| {
        flag.clone()
            .or_else(|| saved.map(|record| from_record(record).clone()))
    };
    let timezone = pick(&args.timezone, |record| &record.timezone);
    let locale = pick(&args.locale, |record| &record.locale);
    let keymap = pick(&args.keymap, |record| &record.keymap);
    let (timezone, locale, keymap) = match (timezone, locale, keymap) {
        (Some(timezone), Some(locale), Some(keymap)) => (timezone, locale, keymap),
        (timezone, locale, keymap) => {
            let (host_timezone, host_locale, host_keymap) = host_defaults();
            (
                timezone.unwrap_or(host_timezone),
                locale.unwrap_or(host_locale),
                keymap.unwrap_or(host_keymap),
            )
        }
    };
    let mut answers = Answers {
        user,
        password_hash: UNASKED_HASH.into(),
        root_password_hash: None,
        hostname,
        ssh_keys,
        timezone,
        locale,
        keymap,
        packages,
    };
    answers.validate().map_err(|error| {
        // `validate` names the field first, so this catches only the host name's own error.
        if !hostname_given && error.starts_with("hostname") {
            format!("{error}; the machine name is the default, give --hostname NAME")
        } else {
            error
        }
    })?;

    let password_source =
        PasswordSource::given(&args.password, args.ask_password, &args.password_file);
    let password_source = match (password_source, saved) {
        (Some(source), _) => source,
        (None, Some(_)) if tty => PasswordSource::Ask,
        (None, Some(_)) => {
            return Err(
                "the password is not saved and stdin is not a terminal to ask for it \
                        on: give --password-file FILE (or --password TEXT)"
                    .into(),
            );
        }
        (None, None) => {
            return Err(
                "a password is required: give --ask-password, --password-file FILE or \
                 --password TEXT"
                    .into(),
            );
        }
    };
    let password = read_password(
        password_source,
        Account::User,
        &answers.user,
        tty,
        &mut read_secret,
    )?;
    answers.password_hash = hash_password(&password)?;
    drop(password);

    let root_source = PasswordSource::given(
        &args.root_password,
        args.ask_root_password,
        &args.root_password_file,
    );
    let root_source = match root_source {
        Some(source) => Some(source),
        None if saved.is_some_and(|record| record.root_password) => {
            if !tty {
                return Err(
                    "root had a password, which is not saved, and stdin is not a \
                            terminal to ask for it on: give --root-password-file FILE (or \
                            --root-password TEXT)"
                        .into(),
                );
            }
            Some(PasswordSource::Ask)
        }
        None => None,
    };
    if let Some(source) = root_source {
        let password = read_password(source, Account::Root, "root", tty, &mut read_secret)?;
        answers.root_password_hash = Some(hash_password(&password)?);
    }
    answers.validate()?;

    let record = BootstrapRecord {
        user: answers.user.clone(),
        hostname: answers.hostname.clone(),
        ssh_keys: answers.ssh_keys.clone(),
        timezone: answers.timezone.clone(),
        locale: answers.locale.clone(),
        keymap: answers.keymap.clone(),
        packages: answers.packages.clone(),
        root_password: answers.root_password_hash.is_some(),
    };
    Ok((answers, record))
}

/// `quick bootstrap` ends by booting the installed disk, which neither of these can:
/// `machines::prepare_boot` refuses both for a disk with an install on it.
pub fn refuse_unbootable(
    firmware: Option<Firmware>,
    hypervisor: &HypervisorArgs,
) -> Result<(), String> {
    if firmware == Some(Firmware::None) {
        return Err(
            "--firmware none cannot boot the disk the install writes: bootstrap with \
                    --firmware uefi (the default) or bios"
                .into(),
        );
    }
    if hypervisor.hypervisor.is_some() || hypervisor.hypervisor_cmdline.is_some() {
        return Err(
            "--hypervisor: the installed disk's boot loader cannot start under the \
                    hypervisor, so bootstrap installs without one"
                .into(),
        );
    }
    Ok(())
}

/// `refuse_unbootable` for `quick reinstall`, whose machine already has its firmware and
/// hypervisor: the way out is `quick edit`, not another option.
pub fn refuse_unbootable_machine(
    firmware: Firmware,
    has_hypervisor: bool,
    name: &str,
) -> Result<(), String> {
    if firmware == Firmware::None {
        return Err(format!(
            "{name} has firmware none, which cannot boot the disk the install writes: `uemulator \
             quick edit {name} --firmware uefi` first"
        ));
    }
    if has_hypervisor {
        return Err(format!(
            "{name} runs its installer under a hypervisor, and the installed disk's boot loader \
             cannot start under it: `uemulator quick edit {name} --remove-hypervisor` first"
        ));
    }
    Ok(())
}

const CTRL_C: u8 = 0x03;
const CTRL_D: u8 = 0x04;
const BACKSPACE: u8 = 0x08;
const DELETE: u8 = 0x7f;
/// The top bits of a UTF-8 continuation byte, `10xxxxxx`.
const UTF8_CONTINUATION_MASK: u8 = 0xc0;
const UTF8_CONTINUATION: u8 = 0x80;

/// One line typed in raw mode, where the terminal neither edits it nor turns Ctrl-C into a
/// signal: that is done here, so the caller always gets control back to restore the terminal.
/// Enter ends it, Backspace takes back a whole character, Ctrl-C is `Interrupted`, and Ctrl-D on
/// an empty line or the end of input is `UnexpectedEof`.
fn read_secret_line(input: &mut impl io::Read) -> io::Result<String> {
    let mut line = Vec::new();
    let mut byte = [0u8];
    loop {
        if input.read(&mut byte)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the input ended before Enter",
            ));
        }
        match byte[0] {
            b'\r' | b'\n' => break,
            CTRL_C => return Err(io::Error::new(io::ErrorKind::Interrupted, "interrupted")),
            CTRL_D if line.is_empty() => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the input ended before Enter",
                ));
            }
            CTRL_D => {}
            BACKSPACE | DELETE => {
                while line
                    .pop()
                    .is_some_and(|b| b & UTF8_CONTINUATION_MASK == UTF8_CONTINUATION)
                {
                }
            }
            other => line.push(other),
        }
    }
    String::from_utf8(line)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "the input is not UTF-8"))
}

/// Prompts on `output` and reads a line from `input` with the terminal behind `stty` in raw mode
/// without echo, putting its settings back on every path out. Refuses to read at all when echo
/// cannot be turned off (no `stty`, not unix), rather than show the password.
fn read_secret_with(
    prompt: &str,
    stty: impl Fn(&[&str]) -> Option<String>,
    input: &mut impl io::Read,
    output: &mut impl io::Write,
) -> io::Result<String> {
    let no_echo_control = || {
        io::Error::other(
            "cannot turn off echo on this terminal: give the password in a file \
             (--password-file, --root-password-file)",
        )
    };
    // Echo goes off before the prompt shows: whatever is typed once it shows must not appear.
    let saved = stty(&["-g"]).ok_or_else(no_echo_control)?;
    // Raw mode covers what is typed (Ctrl-C comes here as a byte); a SIGTERM or SIGHUP from
    // outside during the prompt still leaves the terminal without echo.
    if stty(&["raw", "-echo"]).is_none() {
        restore_terminal(&stty, &saved);
        return Err(no_echo_control());
    }
    let line = write!(output, "{prompt}")
        .and_then(|()| output.flush())
        .and_then(|()| read_secret_line(input));
    restore_terminal(&stty, &saved);
    // The Enter that ended the line was not echoed either.
    writeln!(output)?;
    line
}

/// Puts back the settings `stty -g` saved. A failure is only reported: the password is read
/// by then, and `stty sane` is what the person needs to know about.
fn restore_terminal(stty: &impl Fn(&[&str]) -> Option<String>, saved: &str) {
    if stty(&[saved]).is_none() {
        eprintln!("warning: could not restore the terminal's settings; `stty sane` does");
    }
}

/// `resolve_answers`'s `read_secret` for a person at the terminal: the prompt goes to stderr.
pub fn read_secret_from_terminal(prompt: &str) -> io::Result<String> {
    read_secret_with(
        prompt,
        term::stty,
        &mut io::stdin().lock(),
        &mut io::stderr(),
    )
}

/// Watches console output for `COMPLETION_MARKER`, which an installer prints once it has
/// finished. Clones share one state: the console sink runs on another thread than the run loop
/// that asks `seen`.
#[derive(Clone, Default, Debug)]
pub struct MarkerWatch(Arc<Mutex<MarkerState>>);

#[derive(Default, Debug)]
struct MarkerState {
    /// The last `COMPLETION_MARKER.len() - 1` bytes already fed: enough to complete a marker
    /// split across chunks, too short to hold a whole one, so a marker is never counted twice.
    tail: Vec<u8>,
    seen: bool,
}

impl MarkerWatch {
    pub fn feed(&self, bytes: &[u8]) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.seen {
            return;
        }
        let marker = COMPLETION_MARKER.as_bytes();
        let mut window = std::mem::take(&mut state.tail);
        window.extend_from_slice(bytes);
        if window
            .windows(marker.len())
            .any(|candidate| candidate == marker)
        {
            state.seen = true;
            return;
        }
        let keep = marker.len() - 1;
        let start = window.len().saturating_sub(keep);
        state.tail = window.split_off(start);
    }

    pub fn seen(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .seen
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::path::PathBuf;

    use uemulator_bootstrap::sha512_crypt;

    use super::*;
    use crate::test_support::TempDir;

    fn record() -> BootstrapRecord {
        BootstrapRecord {
            user: "saved".into(),
            hostname: "saved-host".into(),
            ssh_keys: vec!["ssh-ed25519 AAAAsaved saved@host".into()],
            timezone: "Europe/Berlin".into(),
            locale: "de_DE.UTF-8".into(),
            keymap: "de".into(),
            packages: vec!["vim".into()],
            root_password: false,
        }
    }

    fn args_with_password(user: Option<&str>) -> BootstrapArgs {
        BootstrapArgs {
            user: user.map(str::to_string),
            password: Some("hunter2".into()),
            ..BootstrapArgs::default()
        }
    }

    fn no_prompt(prompt: &str) -> io::Result<String> {
        panic!("unexpected prompt {prompt:?}")
    }

    /// Whether `hash` is `password` hashed with the salt it carries.
    fn hash_matches(hash: &str, password: &str) -> bool {
        let salt = hash
            .strip_prefix("$6$")
            .and_then(|rest| rest.split('$').next())
            .unwrap();
        sha512_crypt(password, salt) == hash
    }

    fn resolve(
        args: &BootstrapArgs,
        saved: Option<&BootstrapRecord>,
    ) -> Result<(Answers, BootstrapRecord), String> {
        resolve_answers(args, saved, "machine", false, no_prompt)
    }

    /// `Answers` has no `Debug` (it holds hashes), so no `unwrap_err`.
    fn refused(result: Result<(Answers, BootstrapRecord), String>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(error) => error,
        }
    }

    #[test]
    fn host_defaults_are_never_empty_and_fit_every_installer() {
        let (timezone, locale, keymap) = host_defaults();
        assert!(!timezone.is_empty() && !locale.is_empty());
        assert!(
            keymap_layout(&keymap).as_deref() == Some(keymap.as_str()),
            "{keymap}"
        );
    }

    #[test]
    fn timezone_is_the_localtime_link_after_zoneinfo() {
        for (target, want) in [
            (
                "/var/db/timezone/zoneinfo/Europe/Berlin",
                Some("Europe/Berlin"),
            ),
            (
                "/usr/share/zoneinfo/America/New_York",
                Some("America/New_York"),
            ),
            ("../usr/share/zoneinfo/Etc/UTC", Some("Etc/UTC")),
            ("/usr/share/zoneinfo/", None),
            ("/etc/some-file", None),
        ] {
            assert_eq!(
                timezone_from_localtime(Path::new(target)).as_deref(),
                want,
                "{target}"
            );
        }
    }

    #[test]
    fn locale_comes_from_lang_when_it_names_a_language_and_country() {
        for (lang, want) in [
            (Some("en_GB.UTF-8"), Some("en_GB.UTF-8")),
            (Some("de_DE.UTF-8"), Some("de_DE.UTF-8")),
            (Some("fil_PH.UTF-8"), Some("fil_PH.UTF-8")),
            (Some("C"), None),
            (Some("C.UTF-8"), None),
            (Some("POSIX"), None),
            (Some(""), None),
            (Some("en_US.UTF-8; rm -rf /"), None),
            (None, None),
        ] {
            assert_eq!(locale_from_lang(lang).as_deref(), want, "{lang:?}");
        }
    }

    #[test]
    fn keymaps_become_plain_layout_names() {
        for (keymap, want) in [
            ("us", Some("us")),
            ("de", Some("de")),
            ("de-latin1", Some("de")),
            ("de-latin1-nodeadkeys", Some("de")),
            ("fr_CH", None),
            ("gb", Some("gb")),
            ("dvorak", None),
            ("US", None),
            ("", None),
        ] {
            assert_eq!(keymap_layout(keymap).as_deref(), want, "{keymap}");
        }
    }

    #[test]
    fn vconsole_keymap_is_read_and_mapped() {
        let text = "# written by systemd-localed\nFONT=eurlatgr\nKEYMAP=\"de-latin1\"\n";
        assert_eq!(keymap_from_vconsole(text).as_deref(), Some("de"));
        assert_eq!(keymap_from_vconsole("KEYMAP=fr\n").as_deref(), Some("fr"));
        assert_eq!(keymap_from_vconsole("FONT=x\n"), None);
    }

    #[test]
    fn flag_beats_record_and_record_beats_default() {
        let saved = record();
        let (answers, record) = resolve(&args_with_password(None), Some(&saved)).unwrap();
        assert_eq!(answers.user, "saved");
        assert_eq!(answers.hostname, "saved-host");
        assert_eq!(answers.timezone, "Europe/Berlin");
        assert_eq!(answers.locale, "de_DE.UTF-8");
        assert_eq!(answers.keymap, "de");
        assert_eq!(answers.ssh_keys, saved.ssh_keys);
        assert_eq!(answers.packages, saved.packages);
        assert_eq!(record, saved);

        let home = TempDir::new("bootstrap-keys");
        let key = home.path().join("id.pub");
        fs::write(&key, "ssh-ed25519 AAAAflag flag@host\n").unwrap();
        let args = BootstrapArgs {
            user: Some("flag".into()),
            hostname: Some("flag-host".into()),
            ssh_key: vec![key],
            timezone: Some("Asia/Tokyo".into()),
            locale: Some("ja_JP.UTF-8".into()),
            keymap: Some("jp".into()),
            package: vec!["htop".into(), "git".into()],
            ..args_with_password(None)
        };
        let (answers, record) = resolve(&args, Some(&saved)).unwrap();
        assert_eq!(answers.user, "flag");
        assert_eq!(answers.hostname, "flag-host");
        assert_eq!(answers.timezone, "Asia/Tokyo");
        assert_eq!(answers.locale, "ja_JP.UTF-8");
        assert_eq!(answers.keymap, "jp");
        assert_eq!(answers.ssh_keys, vec!["ssh-ed25519 AAAAflag flag@host"]);
        assert_eq!(answers.packages, vec!["htop", "git"]);
        assert_eq!(record.user, "flag");
        assert_eq!(record.ssh_keys, answers.ssh_keys);
        assert_eq!(record.packages, answers.packages);

        let (answers, _) = resolve(&args_with_password(Some("alex")), None).unwrap();
        let (timezone, locale, keymap) = host_defaults();
        assert_eq!(
            (answers.timezone, answers.locale, answers.keymap),
            (timezone, locale, keymap)
        );
        assert!(answers.ssh_keys.is_empty() && answers.packages.is_empty());
    }

    #[test]
    fn hostname_defaults_to_the_machine_name() {
        let (answers, record) = resolve(&args_with_password(Some("alex")), None).unwrap();
        assert_eq!(answers.hostname, "machine");
        assert_eq!(record.hostname, "machine");
    }

    #[test]
    fn a_user_is_needed_from_the_flag_or_the_record() {
        let error = refused(resolve(&args_with_password(None), None));
        assert!(error.contains("--user"), "{error}");
    }

    #[test]
    fn validation_errors_name_the_field() {
        let error = refused(resolve(&args_with_password(Some("root")), None));
        assert!(error.contains("user"), "{error}");
        let args = BootstrapArgs {
            hostname: Some("bad host".into()),
            ..args_with_password(Some("alex"))
        };
        let error = refused(resolve(&args, None));
        assert!(error.contains("hostname"), "{error}");
        assert!(!error.contains("--hostname"), "{error}");
        // The hint is for a host name that came from the machine name, and only for that.
        let error = refused(resolve_answers(
            &args_with_password(Some("root")),
            None,
            "my_box",
            false,
            no_prompt,
        ));
        assert!(error.starts_with("user"), "{error}");
        assert!(!error.contains("--hostname"), "{error}");
        let error = refused(resolve_answers(
            &args_with_password(Some("alex")),
            None,
            "my_box",
            false,
            no_prompt,
        ));
        assert!(error.starts_with("hostname"), "{error}");
        assert!(error.contains("--hostname"), "{error}");
    }

    #[test]
    fn a_keymap_must_be_a_plain_layout_name() {
        for keymap in ["de-latin1", "dvorak", "US", "fr_CH"] {
            let args = BootstrapArgs {
                keymap: Some(keymap.into()),
                ..args_with_password(Some("alex"))
            };
            let error = refused(resolve(&args, None));
            assert!(error.starts_with("keymap"), "{keymap}: {error}");
            assert!(error.contains("such as us, de or fr"), "{keymap}: {error}");
        }
        let args = BootstrapArgs {
            keymap: Some("de".into()),
            ..args_with_password(Some("alex"))
        };
        let (answers, _) = resolve(&args, None).unwrap();
        assert_eq!(answers.keymap, "de");
    }

    #[test]
    fn the_password_is_hashed_and_root_is_locked_by_default() {
        let (answers, record) = resolve(&args_with_password(Some("alex")), None).unwrap();
        assert!(hash_matches(&answers.password_hash, "hunter2"));
        assert_eq!(answers.root_password_hash, None);
        assert!(!record.root_password);
        let args = BootstrapArgs {
            root_password: Some("toor".into()),
            ..args_with_password(Some("alex"))
        };
        let (answers, record) = resolve(&args, None).unwrap();
        assert!(hash_matches(
            answers.root_password_hash.as_deref().unwrap(),
            "toor"
        ));
        assert!(record.root_password);
    }

    #[test]
    fn each_hash_gets_its_own_salt() {
        let (first, _) = resolve(&args_with_password(Some("alex")), None).unwrap();
        let (second, _) = resolve(&args_with_password(Some("alex")), None).unwrap();
        assert_ne!(first.password_hash, second.password_hash);
    }

    #[test]
    fn an_empty_password_is_refused() {
        let args = BootstrapArgs {
            password: Some(String::new()),
            ..args_with_password(Some("alex"))
        };
        assert!(refused(resolve(&args, None)).contains("empty"));
    }

    #[test]
    fn ask_password_asks_twice_and_the_answers_must_match() {
        let args = BootstrapArgs {
            user: Some("alex".into()),
            ask_password: true,
            ..BootstrapArgs::default()
        };
        let prompts = RefCell::new(Vec::new());
        let (answers, _) = resolve_answers(&args, None, "machine", true, |prompt| {
            prompts.borrow_mut().push(prompt.to_string());
            Ok("secret".into())
        })
        .unwrap();
        assert_eq!(prompts.borrow().len(), 2);
        assert!(hash_matches(&answers.password_hash, "secret"));

        let answers = RefCell::new(vec!["one".to_string(), "two".to_string()]);
        let error = refused(resolve_answers(&args, None, "machine", true, |_| {
            Ok(answers.borrow_mut().remove(0))
        }));
        assert!(error.contains("did not match"), "{error}");
    }

    #[test]
    fn ask_password_needs_a_terminal() {
        let args = BootstrapArgs {
            user: Some("alex".into()),
            ask_password: true,
            ..BootstrapArgs::default()
        };
        let error = refused(resolve(&args, None));
        assert!(error.contains("--password-file"), "{error}");
    }

    #[test]
    fn a_saved_record_without_a_password_option_asks_on_a_terminal_only() {
        let args = BootstrapArgs::default();
        let saved = record();
        let error = refused(resolve(&args, Some(&saved)));
        assert!(error.contains("--password-file"), "{error}");

        let asked = RefCell::new(0);
        let (answers, _) = resolve_answers(&args, Some(&saved), "machine", true, |_| {
            *asked.borrow_mut() += 1;
            Ok("again".into())
        })
        .unwrap();
        assert_eq!(*asked.borrow(), 2);
        assert!(hash_matches(&answers.password_hash, "again"));
    }

    #[test]
    fn a_new_bootstrap_without_a_password_option_is_an_error() {
        let args = BootstrapArgs {
            user: Some("alex".into()),
            ..BootstrapArgs::default()
        };
        let error = refused(resolve_answers(&args, None, "machine", true, no_prompt));
        assert!(error.contains("--password-file"), "{error}");
    }

    #[test]
    fn a_saved_root_password_is_asked_for_again() {
        let saved = BootstrapRecord {
            root_password: true,
            ..record()
        };
        let error = refused(resolve(&args_with_password(None), Some(&saved)));
        assert!(error.contains("--root-password-file"), "{error}");
        let prompts = RefCell::new(Vec::new());
        let (answers, record) = resolve_answers(
            &args_with_password(None),
            Some(&saved),
            "m",
            true,
            |prompt| {
                prompts.borrow_mut().push(prompt.to_string());
                Ok("toor".into())
            },
        )
        .unwrap();
        assert_eq!(prompts.borrow().len(), 2);
        assert!(prompts.borrow().iter().all(|p| p.contains("root")));
        assert!(hash_matches(
            answers.root_password_hash.as_deref().unwrap(),
            "toor"
        ));
        assert!(record.root_password);
    }

    #[test]
    fn password_file_strips_one_trailing_newline() {
        let home = TempDir::new("bootstrap-password-file");
        for (contents, want) in [
            ("pw\n", "pw"),
            ("pw\r\n", "pw"),
            ("pw", "pw"),
            ("pw\n\n", "pw\n"),
            (" pw \n", " pw "),
        ] {
            let path = home.path().join("password");
            fs::write(&path, contents).unwrap();
            let args = BootstrapArgs {
                user: Some("alex".into()),
                password_file: Some(path.clone()),
                root_password_file: Some(path),
                ..BootstrapArgs::default()
            };
            let (answers, _) = resolve(&args, None).unwrap();
            assert!(hash_matches(&answers.password_hash, want), "{contents:?}");
            assert!(
                hash_matches(answers.root_password_hash.as_deref().unwrap(), want),
                "{contents:?}"
            );
        }
    }

    #[test]
    fn a_missing_password_file_is_named() {
        let args = BootstrapArgs {
            user: Some("alex".into()),
            password_file: Some(PathBuf::from("/nonexistent/uemulator-password")),
            ..BootstrapArgs::default()
        };
        let error = refused(resolve(&args, None));
        assert!(error.contains("/nonexistent/uemulator-password"), "{error}");
    }

    #[test]
    fn ssh_key_files_give_one_key_per_line() {
        let home = TempDir::new("bootstrap-ssh-keys");
        let first = home.path().join("a.pub");
        let second = home.path().join("b.pub");
        fs::write(&first, "# laptop\nssh-ed25519 AAAA1 a\n\nssh-rsa AAAA2 b\n").unwrap();
        fs::write(&second, "ssh-ed25519 AAAA3 c").unwrap();
        let args = BootstrapArgs {
            ssh_key: vec![first, second],
            ..args_with_password(Some("alex"))
        };
        let (answers, _) = resolve(&args, None).unwrap();
        assert_eq!(
            answers.ssh_keys,
            vec![
                "ssh-ed25519 AAAA1 a",
                "ssh-rsa AAAA2 b",
                "ssh-ed25519 AAAA3 c"
            ]
        );
        let empty = home.path().join("empty.pub");
        fs::write(&empty, "\n# nothing\n").unwrap();
        let args = BootstrapArgs {
            ssh_key: vec![empty.clone()],
            ..args_with_password(Some("alex"))
        };
        let error = refused(resolve(&args, None));
        assert!(error.contains(&empty.display().to_string()), "{error}");
    }

    #[test]
    fn a_private_ssh_key_is_refused() {
        let home = TempDir::new("bootstrap-private-key");
        let private = home.path().join("id_ed25519");
        fs::write(
            &private,
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\n-----END OPENSSH PRIVATE KEY-----\n",
        )
        .unwrap();
        let args = BootstrapArgs {
            ssh_key: vec![private],
            ..args_with_password(Some("alex"))
        };
        let error = refused(resolve(&args, None));
        assert!(error.contains("private key"), "{error}");
        assert!(!error.contains("b3BlbnNzaC1rZXk"), "{error}");
    }

    #[test]
    fn a_machine_name_that_is_no_hostname_asks_for_one() {
        let error = refused(resolve_answers(
            &args_with_password(Some("alex")),
            None,
            "my_box",
            false,
            no_prompt,
        ));
        assert!(error.contains("--hostname"), "{error}");
    }

    #[test]
    fn firmware_none_and_a_hypervisor_are_refused() {
        let none = HypervisorArgs::default();
        refuse_unbootable(None, &none).unwrap();
        refuse_unbootable(Some(Firmware::Uefi), &none).unwrap();
        refuse_unbootable(Some(Firmware::Bios), &none).unwrap();
        let error = refuse_unbootable(Some(Firmware::None), &none).unwrap_err();
        assert!(error.starts_with("--firmware none"), "{error}");
        let xen = HypervisorArgs {
            hypervisor: Some(PathBuf::from("xen.gz")),
            hypervisor_cmdline: None,
        };
        let error = refuse_unbootable(None, &xen).unwrap_err();
        assert!(error.starts_with("--hypervisor"), "{error}");
        let cmdline = HypervisorArgs {
            hypervisor: None,
            hypervisor_cmdline: Some("dom0_mem=1G".into()),
        };
        assert!(
            refuse_unbootable(None, &cmdline)
                .unwrap_err()
                .starts_with("--hypervisor")
        );
    }

    #[test]
    fn a_secret_line_ends_at_enter_and_takes_backspace() {
        let read = |bytes: &[u8]| read_secret_line(&mut &bytes[..]);
        assert_eq!(read(b"secret\r").unwrap(), "secret");
        assert_eq!(read(b"secret\n").unwrap(), "secret");
        assert_eq!(read(b"secrex\x7ft\r").unwrap(), "secret");
        assert_eq!(read(b"\x08\x08ab\x08c\r").unwrap(), "ac");
        assert_eq!(read("pässwörd\r".as_bytes()).unwrap(), "pässwörd");
        assert_eq!(
            read(b"x\x03").unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert_eq!(
            read(b"\x04").unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(
            read(b"abc").unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    /// A fake `stty` that records its calls; `-g` answers "saved".
    struct FakeStty(RefCell<Vec<String>>);

    impl FakeStty {
        fn call(&self, args: &[&str]) -> Option<String> {
            self.0.borrow_mut().push(args.join(" "));
            Some(if args == ["-g"] { "saved" } else { "" }.to_string())
        }
    }

    #[test]
    fn echo_is_off_while_reading_and_restored_on_every_path() {
        for (input, ok) in [(&b"pw\r"[..], true), (b"p\x03", false), (b"", false)] {
            let stty = FakeStty(RefCell::new(Vec::new()));
            let mut output = Vec::new();
            let result =
                read_secret_with("Password: ", |a| stty.call(a), &mut &input[..], &mut output);
            assert_eq!(result.is_ok(), ok, "{input:?}");
            let calls = stty.0.borrow();
            assert_eq!(calls.first().map(String::as_str), Some("-g"));
            assert!(calls[1].contains("-echo"), "{calls:?}");
            assert_eq!(calls.last().map(String::as_str), Some("saved"), "{input:?}");
            let shown = String::from_utf8(output).unwrap();
            assert!(shown.starts_with("Password: "), "{shown:?}");
            assert!(!shown.contains("pw"), "{shown:?}");
        }
    }

    /// Output that lands in the same log as the `stty` calls, to see their order.
    struct LoggedOutput<'a>(&'a RefCell<Vec<String>>);

    impl io::Write for LoggedOutput<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .borrow_mut()
                .push(format!("output {}", String::from_utf8_lossy(bytes)));
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn the_prompt_shows_only_once_echo_is_off() {
        // Whatever is typed as soon as the prompt shows must not be echoed.
        let log = RefCell::new(Vec::new());
        let stty = |args: &[&str]| {
            log.borrow_mut().push(format!("stty {}", args.join(" ")));
            Some("saved".to_string())
        };
        read_secret_with(
            "Password: ",
            stty,
            &mut &b"pw\r"[..],
            &mut LoggedOutput(&log),
        )
        .unwrap();
        let log = log.into_inner();
        let echo_off = log.iter().position(|e| e.contains("-echo")).unwrap();
        let prompt = log.iter().position(|e| e == "output Password: ").unwrap();
        assert!(echo_off < prompt, "{log:?}");
    }

    #[test]
    fn a_secret_is_not_read_when_echo_cannot_be_turned_off() {
        let mut output = Vec::new();
        let error =
            read_secret_with("Password: ", |_| None, &mut &b"pw\r"[..], &mut output).unwrap_err();
        assert!(error.to_string().contains("--password-file"), "{error}");
    }

    fn marker_line() -> Vec<u8> {
        format!("boot text\r\n{COMPLETION_MARKER}\r\n").into_bytes()
    }

    #[test]
    fn marker_in_one_chunk() {
        let watch = MarkerWatch::default();
        assert!(!watch.seen());
        watch.feed(&marker_line());
        assert!(watch.seen());
    }

    #[test]
    fn marker_split_at_every_byte() {
        let line = marker_line();
        for split in 0..=line.len() {
            let watch = MarkerWatch::default();
            watch.feed(&line[..split]);
            watch.feed(&line[split..]);
            assert!(watch.seen(), "split at {split}");
        }
    }

    #[test]
    fn partial_marker_is_not_seen() {
        let watch = MarkerWatch::default();
        watch.feed(&COMPLETION_MARKER.as_bytes()[..COMPLETION_MARKER.len() - 1]);
        watch.feed(b"\r\nsomething else\r\n");
        assert!(!watch.seen());
    }

    /// A false start that shares the marker's first bytes must not throw away the real marker
    /// that follows it, wherever the chunks split.
    #[test]
    fn marker_after_a_repeated_prefix() {
        let text = format!("uemulator-b{COMPLETION_MARKER}\r\n").into_bytes();
        for split in 0..=text.len() {
            let watch = MarkerWatch::default();
            watch.feed(&text[..split]);
            watch.feed(&text[split..]);
            assert!(watch.seen(), "split at {split}");
        }
        for first in 0..=text.len() {
            for second in first..=text.len() {
                let watch = MarkerWatch::default();
                watch.feed(&text[..first]);
                watch.feed(&text[first..second]);
                watch.feed(&text[second..]);
                assert!(watch.seen(), "split at {first} and {second}");
            }
        }
    }

    #[test]
    fn marker_surrounded_by_ansi_noise() {
        let watch = MarkerWatch::default();
        watch.feed(format!("\x1b[0m\r\n{COMPLETION_MARKER}\r\n\x1b[2J").as_bytes());
        assert!(watch.seen());
    }

    #[test]
    fn refuse_unbootable_machine_names_the_edit_that_fixes_it() {
        assert!(refuse_unbootable_machine(Firmware::Uefi, false, "m").is_ok());
        assert!(refuse_unbootable_machine(Firmware::Bios, false, "m").is_ok());
        let error = refuse_unbootable_machine(Firmware::None, false, "m").unwrap_err();
        assert!(error.contains("quick edit m --firmware uefi"), "{error}");
        let error = refuse_unbootable_machine(Firmware::Uefi, true, "m").unwrap_err();
        assert!(
            error.contains("quick edit m --remove-hypervisor"),
            "{error}"
        );
    }
}
