//! `quick bootstrap` for the app: its answers, as JSON from Swift, turned into the installer's
//! initramfs with the answer files appended and the kernel arguments that go with them, as the
//! CLI's `quick bootstrap` prepares them (`docs/plans/35-quick-bootstrap.md`).

use std::ffi::{CStr, CString, c_char};

use serde::Deserialize;
use uemulator_bootstrap::{
    Answers, Distro, SALT_LEN, append_to_initramfs, render, salt_from_bytes, sha512_crypt,
};

use crate::write_message;

/// Flipped into every byte of the caller's salt to give root's password a salt of its own. One
/// random salt comes across the ABI; a salt only has to differ between hashes, and flipping the
/// low bit changes the character `salt_from_bytes` picks for every byte (`byte % 64`), so the two
/// salts never match and the same password for both accounts never gives the same hash.
const ROOT_SALT_FLIP: u8 = 0x01;

/// The answers JSON, as `ua_bootstrap_initrd`'s header comment spells it. Unknown keys are
/// refused so a misspelled key on the Swift side fails instead of silently using a default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AnswersJson {
    user: String,
    password: String,
    #[serde(default)]
    root_password: Option<String>,
    hostname: String,
    ssh_keys: Vec<String>,
    timezone: String,
    locale: String,
    keymap: String,
    packages: Vec<String>,
}

/// Overwrites a plaintext password before its memory goes back to the allocator. Best effort
/// only: serde_json's scratch buffers for escaped strings, the caller's C string and whatever
/// copies the hash makes internally are not reached.
fn scrub(text: &mut String) {
    let mut bytes = std::mem::take(text).into_bytes();
    bytes.fill(0);
    // Keeps the fill from being optimized away as a dead store before the free.
    std::hint::black_box(&bytes);
}

fn root_salt(salt: &[u8; SALT_LEN]) -> [u8; SALT_LEN] {
    salt.map(|byte| byte ^ ROOT_SALT_FLIP)
}

/// The CLI's machine.toml spelling and the app's `Distro` raw values (apple/App/Presets.swift).
fn parse_distro(name: &str) -> Result<Distro, String> {
    match name {
        "alpine" => Ok(Distro::Alpine),
        "debian" => Ok(Distro::Debian),
        "fedora" => Ok(Distro::Fedora),
        _ => Err(format!(
            "distro {name:?}: must be one of alpine, debian, fedora"
        )),
    }
}

fn hash(what: &str, password: &str, salt: &[u8; SALT_LEN]) -> Result<String, String> {
    // The CLI refuses an empty password too; `root_password: null` is how root gets none.
    if password.is_empty() {
        return Err(format!("the {what} is empty"));
    }
    Ok(sha512_crypt(password, &salt_from_bytes(salt)))
}

/// Hashes the passwords, scrubbing the plaintext whether or not that succeeds.
fn answers_from_json(json: &str, salt: &[u8; SALT_LEN]) -> Result<Answers, String> {
    let mut parsed: AnswersJson =
        serde_json::from_str(json).map_err(|e| format!("the answers: {e}"))?;
    let password_hash = hash("password", &parsed.password, salt);
    let root_password_hash = parsed
        .root_password
        .as_deref()
        .map(|password| hash("root_password", password, &root_salt(salt)))
        .transpose();
    scrub(&mut parsed.password);
    if let Some(password) = parsed.root_password.as_mut() {
        scrub(password);
    }
    Ok(Answers {
        user: parsed.user,
        password_hash: password_hash?,
        root_password_hash: root_password_hash?,
        hostname: parsed.hostname,
        ssh_keys: parsed.ssh_keys,
        timezone: parsed.timezone,
        locale: parsed.locale,
        keymap: parsed.keymap,
        packages: parsed.packages,
    })
}

fn string_argument<'a>(what: &str, text: *const c_char) -> Result<&'a str, String> {
    if text.is_null() {
        return Err(format!("no {what}"));
    }
    unsafe { CStr::from_ptr(text) }
        .to_str()
        .map_err(|_| format!("the {what} is not UTF-8"))
}

/// The extended initramfs and the kernel arguments to append.
fn bootstrap_initrd(
    distro: &str,
    answers_json: &str,
    salt: &[u8],
    initrd: &[u8],
) -> Result<(Vec<u8>, CString), String> {
    let salt: &[u8; SALT_LEN] = salt
        .try_into()
        .map_err(|_| format!("the salt is {} bytes, not {SALT_LEN}", salt.len()))?;
    let distro = parse_distro(distro)?;
    let answers = answers_from_json(answers_json, salt)?;
    let payload = render(distro, &answers)?;
    let cmdline = CString::new(payload.cmdline).map_err(|e| e.to_string())?;
    Ok((append_to_initramfs(initrd, &payload.files), cmdline))
}

/// `quick bootstrap` for the app: `initrd` (the distro's installer initramfs) with the answer
/// files appended, and the kernel arguments that make the installer read them. 0 on success,
/// with `*out`/`*out_len` the initramfs (free with `ua_bytes_free`) and `*cmdline` the arguments
/// (free with `ua_string_free`); -1 with the message in `error` and the outputs untouched.
///
/// # Safety
/// `distro` and `answers_json` are NUL-terminated strings; `salt` and `initrd` have `salt_len`
/// and `initrd_len` readable bytes (or are null with a length of 0); `out`, `out_len` and
/// `cmdline` are writable; `error` has `error_capacity` writable bytes or is null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ua_bootstrap_initrd(
    distro: *const c_char,
    answers_json: *const c_char,
    salt: *const u8,
    salt_len: usize,
    initrd: *const u8,
    initrd_len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
    cmdline: *mut *mut c_char,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    let built = (|| {
        if out.is_null() || out_len.is_null() || cmdline.is_null() {
            return Err("no place for the result".to_string());
        }
        let distro = string_argument("distro", distro)?;
        let answers_json = string_argument("answers", answers_json)?;
        let salt = unsafe { byte_argument("salt", salt, salt_len) }?;
        let initrd = unsafe { byte_argument("initramfs", initrd, initrd_len) }?;
        bootstrap_initrd(distro, answers_json, salt, initrd)
    })();
    match built {
        Ok((initrd, arguments)) => {
            // A boxed slice has capacity == len, so `ua_bytes_free` can rebuild it from the
            // length alone.
            let initrd = initrd.into_boxed_slice();
            unsafe {
                *out_len = initrd.len();
                *out = Box::into_raw(initrd).cast::<u8>();
                *cmdline = arguments.into_raw();
            }
            0
        }
        Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure) };
            -1
        }
    }
}

/// # Safety
/// `bytes` has `len` readable bytes, or is null with `len` 0.
unsafe fn byte_argument<'a>(what: &str, bytes: *const u8, len: usize) -> Result<&'a [u8], String> {
    match (bytes.is_null(), len) {
        (true, 0) => Ok(&[]),
        (true, _) => Err(format!("no {what}")),
        (false, _) => Ok(unsafe { std::slice::from_raw_parts(bytes, len) }),
    }
}

/// Frees bytes `ua_bootstrap_initrd` returned.
///
/// # Safety
/// `bytes` and `len` are what `ua_bootstrap_initrd` wrote, not used again; or `bytes` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_bytes_free(bytes: *mut u8, len: usize) {
    if !bytes.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(bytes, len)) });
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};

    use uemulator_bootstrap::{SALT_LEN, salt_from_bytes, sha512_crypt};

    use super::*;
    use crate::machine_file::ua_string_free;

    const SALT: [u8; SALT_LEN] = *b"0123456789abcdef";
    /// Stands in for the distro's initramfs; only its bytes coming back first matters.
    const FAKE_INITRD: &[u8] = b"070701 a fake initramfs";
    /// Debian's arguments (`uemulator_bootstrap`'s debian.rs).
    const DEBIAN_CMDLINE: &str = "auto=true priority=critical";
    const ERROR_CAPACITY: usize = 512;

    const ANSWERS: &str = r#"{
        "user": "alex",
        "password": "correct horse",
        "root_password": null,
        "hostname": "devbox",
        "ssh_keys": ["ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKey alex@laptop"],
        "timezone": "Europe/Berlin",
        "locale": "de_DE.UTF-8",
        "keymap": "de",
        "packages": ["git"]
    }"#;

    struct Outcome {
        result: i32,
        initrd: Option<Vec<u8>>,
        cmdline: Option<String>,
        error: String,
    }

    /// Calls the C ABI as Swift would and frees what it returns.
    fn call(distro: &str, answers: &str, salt: &[u8]) -> Outcome {
        let distro = CString::new(distro).unwrap();
        let answers = CString::new(answers).unwrap();
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len = 0usize;
        let mut cmdline: *mut c_char = std::ptr::null_mut();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let result = unsafe {
            ua_bootstrap_initrd(
                distro.as_ptr(),
                answers.as_ptr(),
                salt.as_ptr(),
                salt.len(),
                FAKE_INITRD.as_ptr(),
                FAKE_INITRD.len(),
                &mut out,
                &mut out_len,
                &mut cmdline,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let initrd = (!out.is_null()).then(|| {
            let bytes = unsafe { std::slice::from_raw_parts(out, out_len) }.to_vec();
            unsafe { ua_bytes_free(out, out_len) };
            bytes
        });
        let cmdline = (!cmdline.is_null()).then(|| {
            let text = unsafe { CStr::from_ptr(cmdline) }
                .to_str()
                .unwrap()
                .to_owned();
            unsafe { ua_string_free(cmdline) };
            text
        });
        let error = unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Outcome {
            result,
            initrd,
            cmdline,
            error,
        }
    }

    #[test]
    fn bootstrap_initrd_round_trip() {
        let outcome = call("debian", ANSWERS, &SALT);
        assert_eq!(outcome.result, 0, "{}", outcome.error);
        let initrd = outcome.initrd.expect("an initramfs");
        assert!(initrd.starts_with(FAKE_INITRD));
        assert!(initrd.len() > FAKE_INITRD.len());
        // The password arrives hashed, never as typed.
        let text = String::from_utf8_lossy(&initrd);
        assert!(!text.contains("correct horse"));
        assert!(text.contains("$6$"));
        assert_eq!(outcome.cmdline.as_deref(), Some(DEBIAN_CMDLINE));
    }

    #[test]
    fn every_distro_name_is_accepted_and_others_are_not() {
        for distro in ["alpine", "debian", "fedora"] {
            let outcome = call(distro, ANSWERS, &SALT);
            assert_eq!(outcome.result, 0, "{distro}: {}", outcome.error);
        }
        let outcome = call("gentoo", ANSWERS, &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("gentoo"), "{}", outcome.error);
    }

    #[test]
    fn bootstrap_rejects_bad_json_and_invalid_answers_with_field_in_error() {
        let outcome = call("debian", "{not json", &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.initrd.is_none() && outcome.cmdline.is_none());
        assert!(!outcome.error.is_empty());

        // A Swift-side typo in a key fails loudly instead of being dropped.
        let typo = ANSWERS.replace("\"hostname\"", "\"host_name\"");
        let outcome = call("debian", &typo, &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("host_name"), "{}", outcome.error);

        let bad_keymap = ANSWERS.replace("\"de\"", "\"de; reboot\"");
        let outcome = call("debian", &bad_keymap, &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("keymap"), "{}", outcome.error);

        let bad_user = ANSWERS.replace("\"alex\"", "\"root\"");
        let outcome = call("debian", &bad_user, &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("user"), "{}", outcome.error);

        let empty_password = ANSWERS.replace("correct horse", "");
        let outcome = call("debian", &empty_password, &SALT);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("password"), "{}", outcome.error);
    }

    #[test]
    fn salt_must_be_sixteen_bytes() {
        let outcome = call("debian", ANSWERS, &SALT[..SALT_LEN - 1]);
        assert_eq!(outcome.result, -1);
        assert!(outcome.error.contains("salt"), "{}", outcome.error);
        assert!(outcome.initrd.is_none());
    }

    // Swift waits for the header's copy of the marker on the console.
    #[test]
    fn the_header_carries_the_completion_marker() {
        let header = include_str!("../include/uemulator.h");
        let define = format!(
            "#define UA_BOOTSTRAP_COMPLETION_MARKER \"{}\"",
            uemulator_bootstrap::COMPLETION_MARKER
        );
        assert!(header.contains(&define));
    }

    // The same password for both accounts still gives two different hashes.
    #[test]
    fn root_and_user_hashes_use_different_salts() {
        let answers = ANSWERS.replace(
            "\"root_password\": null",
            "\"root_password\": \"correct horse\"",
        );
        let outcome = call("alpine", &answers, &SALT);
        assert_eq!(outcome.result, 0, "{}", outcome.error);
        let text = String::from_utf8_lossy(&outcome.initrd.unwrap()).into_owned();
        let user = salt_from_bytes(&SALT);
        let root = salt_from_bytes(&root_salt(&SALT));
        assert_ne!(user, root);
        assert!(text.contains(&sha512_crypt("correct horse", &user)));
        assert!(text.contains(&sha512_crypt("correct horse", &root)));
    }
}
