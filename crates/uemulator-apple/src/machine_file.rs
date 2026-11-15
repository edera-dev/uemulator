//! A machine's `machine.toml` (the CLI's format, `crates/uemulator-cli/src/machines.rs`) for the
//! app, and new disk images. The file is handled as an untyped TOML table and crosses to Swift as
//! JSON: the app reads the keys it knows and saves only those back, so every other key (a NIC, a
//! hypervisor, whatever the CLI adds later) survives the app.
//!
//! The CLI's serde structs are not reachable from here (the CLI is a binary crate), so the key
//! names are matched by hand; `docs/plans/28-apple.md` lists the ones the app uses.

use std::ffi::{CStr, CString, c_char};
use std::fs;
use std::io::Read;
use std::path::Path;

use uemulator::{ImageFile, Udisk, UdiskOptions};

use crate::write_message;

const BYTES_PER_MIB: u64 = 1 << 20;
/// Bytes in a udisk image id (`UdiskOptions::new`).
const IMAGE_ID_BYTES: usize = 16;
/// The kernel's entropy device; readable inside the iOS and macOS sandboxes.
const RANDOM_DEVICE: &str = "/dev/urandom";

fn read_table(path: &Path) -> Result<toml::Table, String> {
    match fs::read_to_string(path) {
        Ok(text) => text
            .parse::<toml::Table>()
            .map_err(|e| format!("{}: {e}", path.display())),
        // A machine being created has no file yet: an empty table, which save fills in.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(toml::Table::new()),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

/// `changes` is a JSON object: each key replaces the table's, `null` removes it.
fn merge(table: &mut toml::Table, changes: &str) -> Result<(), String> {
    let changes: serde_json::Map<String, serde_json::Value> = serde_json::from_str(changes)
        .map_err(|e| format!("the changes are not a JSON object: {e}"))?;
    for (key, value) in changes {
        if value.is_null() {
            table.remove(&key);
            continue;
        }
        let value = toml::Value::try_from(&value).map_err(|e| format!("{key}: {e}"))?;
        table.insert(key, value);
    }
    Ok(())
}

/// Through a temporary file and a rename, as the CLI saves it: a crash leaves the old file or the
/// new one, never half of either.
fn write_table(path: &Path, table: &toml::Table) -> Result<(), String> {
    let text =
        toml::to_string_pretty(table).map_err(|e| format!("encoding {}: {e}", path.display()))?;
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, text).map_err(|e| format!("{}: {e}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|e| format!("{}: {e}", path.display()))
}

fn path_argument(path: *const c_char) -> Result<String, String> {
    if path.is_null() {
        return Err("no path".into());
    }
    Ok(unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .into_owned())
}

/// `machine.toml` at `path` as a JSON object, NUL-terminated, to be freed with `ua_string_free`;
/// an empty object when the file does not exist. NULL on failure, with the message in `error`.
///
/// # Safety
/// `path` is a NUL-terminated string; `error` has `error_capacity` writable bytes or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_machine_load(
    path: *const c_char,
    error: *mut c_char,
    error_capacity: usize,
) -> *mut c_char {
    let loaded = path_argument(path)
        .and_then(|path| read_table(Path::new(&path)))
        .and_then(|table| serde_json::to_string(&table).map_err(|e| e.to_string()));
    match loaded.map(|json| CString::new(json).map_err(|e| e.to_string())) {
        Ok(Ok(json)) => json.into_raw(),
        Ok(Err(failure)) | Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure) };
            std::ptr::null_mut()
        }
    }
}

/// Merges `changes` (a JSON object; `null` removes a key) into `machine.toml` at `path`, creating
/// it, and writes it back. 0, or -1 with the message in `error`.
///
/// # Safety
/// `path` and `changes` are NUL-terminated strings; `error` as for `ua_machine_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_machine_save(
    path: *const c_char,
    changes: *const c_char,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    let saved = path_argument(path).and_then(|path| {
        let changes = path_argument(changes)?;
        let path = Path::new(&path);
        let mut table = read_table(path)?;
        merge(&mut table, &changes)?;
        write_table(path, &table)
    });
    match saved {
        Ok(()) => 0,
        Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure) };
            -1
        }
    }
}

/// Frees a string `ua_machine_load` returned.
///
/// # Safety
/// `text` came from `ua_machine_load` and is not used again, or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(unsafe { CString::from_raw(text) });
    }
}

fn random_image_id() -> std::io::Result<[u8; IMAGE_ID_BYTES]> {
    let mut id = [0u8; IMAGE_ID_BYTES];
    fs::File::open(RANDOM_DEVICE)?.read_exact(&mut id)?;
    Ok(id)
}

/// A new, empty udisk image of `size_mib` at `path` (sparse: it grows as the guest writes), as
/// the CLI's `disk create` makes. Refuses to replace an existing file. 0, or -1 with the message.
///
/// # Safety
/// As for `ua_machine_load`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_disk_create(
    path: *const c_char,
    size_mib: u64,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    let created = path_argument(path).and_then(|path| {
        let storage = ImageFile::create(&path).map_err(|e| format!("{path}: {e}"))?;
        let id = random_image_id().map_err(|e| format!("{RANDOM_DEVICE}: {e}"))?;
        let size = size_mib
            .checked_mul(BYTES_PER_MIB)
            .ok_or("the size is too large")?;
        Udisk::create(storage, &UdiskOptions::new(size, id), None)
            .map(drop)
            .map_err(|e| format!("{path}: {e}"))
    });
    match created {
        Ok(()) => 0,
        Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure) };
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `uemulator quick create debian` wrote (a real machine.toml, the NIC included).
    const CLI_MACHINE: &str = r#"install = "debian"
boot_from = "auto"
firmware = "uefi"
memory = 4096
no_reboot = false
disable_jit = false
disable_rng = false
iommu = false
consoles = ["virtio"]

[[disks]]
path = "/machines/debian/disk-0.udisk"
read_only = false

[[networks]]
model = "virtio"
link = "user"
"#;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("uemulator-apple-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    // The app changes what it knows and nothing else: the CLI's NIC, its distro and its flags are
    // still there, the disks array is the app's.
    #[test]
    fn saving_keeps_the_keys_the_app_does_not_know() {
        let dir = scratch("merge");
        let path = dir.join("machine.toml");
        fs::write(&path, CLI_MACHINE).unwrap();
        let mut table = read_table(&path).unwrap();
        merge(&mut table, r#"{"memory": 2048, "iommu": null, "disks": [{"path": "/x/disk-0.udisk", "read_only": true}]}"#)
            .unwrap();
        write_table(&path, &table).unwrap();

        let reread = read_table(&path).unwrap();
        assert_eq!(reread["memory"].as_integer(), Some(2048));
        assert!(!reread.contains_key("iommu"));
        assert_eq!(reread["install"].as_str(), Some("debian"));
        assert_eq!(
            reread["networks"].as_array().unwrap()[0]["link"].as_str(),
            Some("user")
        );
        assert_eq!(
            reread["disks"].as_array().unwrap()[0]["read_only"].as_bool(),
            Some(true)
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_file_loads_as_an_empty_machine() {
        let dir = scratch("missing");
        assert!(read_table(&dir.join("machine.toml")).unwrap().is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    // A new disk is a udisk of the asked size that the library opens, and creating it again over
    // the same path is refused rather than wiping it.
    #[test]
    fn disk_create_makes_a_udisk_and_never_overwrites() {
        const SIZE_MIB: u64 = 64;
        let dir = scratch("disk");
        let path = CString::new(dir.join("disk-0.udisk").to_str().unwrap()).unwrap();
        let mut error = [0 as c_char; 256];
        assert_eq!(
            unsafe { ua_disk_create(path.as_ptr(), SIZE_MIB, error.as_mut_ptr(), error.len()) },
            0
        );
        let mut file = ImageFile::open(dir.join("disk-0.udisk"), true).unwrap();
        let info = Udisk::probe(&mut file).unwrap().expect("a udisk image");
        assert_eq!(info.virtual_size, SIZE_MIB * BYTES_PER_MIB);
        assert_eq!(
            unsafe { ua_disk_create(path.as_ptr(), SIZE_MIB, error.as_mut_ptr(), error.len()) },
            -1
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
