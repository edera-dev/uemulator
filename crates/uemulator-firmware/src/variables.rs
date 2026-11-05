//! The variable services: the pointer-and-size conventions of `GetVariable` and friends in front
//! of `variable_store`.

use alloc::vec::Vec;
use core::ffi::c_void;

use crate::kv::KvPort;
use crate::memory;
use crate::secure_boot::{self, Verdict, signature_type};
use crate::sync::Global;
use crate::text;
use crate::uefi::variable_attribute::{BOOTSERVICE_ACCESS, RUNTIME_ACCESS};
use crate::uefi::*;
use crate::variable_store::{VariableError, Variables};

static VARIABLES: Global<Variables<KvPort>> = Global::new(Variables::new());

/// `EFI_OS_INDICATIONS_BOOT_TO_FW_UI`: an OS sets it in `OsIndications` (`systemctl reboot
/// --firmware-setup`) to land in the boot menu on the next start.
pub const OS_INDICATIONS_BOOT_TO_FW_UI: u64 = 1 << 0;
const OS_INDICATIONS_SUPPORTED: u64 = OS_INDICATIONS_BOOT_TO_FW_UI;
/// `EFI_BOOT_OPTION_SUPPORT_APP`: the menu launches application entries. Not `..._KEY`: there
/// are no `Key####` hot keys.
const BOOT_OPTION_SUPPORT: u32 = 0x2;
const PLATFORM_LANGUAGES: &[u8] = b"en-US\0";

pub fn attach_store(port: KvPort) {
    VARIABLES.get().persistent = Some(port);
}

/// Whether `LoadImage` checks images against db and dbx, decided once at start-up.
static ENFORCING: Global<bool> = Global::new(false);

/// What `SignatureSupport` lists: the signature types db and dbx entries may have here.
const SIGNATURE_SUPPORT: [Guid; 6] = [
    signature_type::SHA256,
    signature_type::SHA384,
    signature_type::SHA512,
    signature_type::X509,
    signature_type::X509_SHA256,
    signature_type::X509_SHA384,
];

/// Secure Boot's state for this boot. With `secure_boot` on and no PK the default keys are
/// enrolled first (into the persistent store, so a later boot without Secure Boot keeps them
/// and shows `SetupMode` 0, `SecureBoot` 0). Shim and the OS read these variables to decide
/// whether to verify what they load.
fn init_secure_boot(variables: &mut Variables<KvPort>, secure_boot: bool) {
    let visible = BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
    let global = guids::GLOBAL_VARIABLE;
    let defaults = secure_boot::defaults();
    if secure_boot && variables.platform_key().is_none() {
        // PK last: until it is there the machine is in setup mode.
        let enrolled = variables.enroll("KEK", &global, &defaults.kek)
            && variables.enroll("db", &guids::IMAGE_SECURITY_DATABASE, &defaults.db)
            && variables.enroll("dbx", &guids::IMAGE_SECURITY_DATABASE, &defaults.dbx)
            && variables.enroll("PK", &global, &defaults.pk);
        if !enrolled {
            crate::serial::println!("secure boot: the variable store has no room for the keys");
        }
    }
    let enforcing = secure_boot && variables.platform_key().is_some();
    *ENFORCING.get() = enforcing;
    variables.set_internal("SecureBoot", &global, visible, &[enforcing as u8]);
    variables.refresh_setup_mode();
    for (name, value) in [
        ("PKDefault", &defaults.pk),
        ("KEKDefault", &defaults.kek),
        ("dbDefault", &defaults.db),
        ("dbxDefault", &defaults.dbx),
    ] {
        variables.set_internal(name, &global, visible, value);
    }
    let support: Vec<u8> = SIGNATURE_SUPPORT
        .iter()
        .flat_map(|guid| guid.to_bytes())
        .collect();
    variables.set_internal("SignatureSupport", &global, visible, &support);
}

/// Why Secure Boot refused an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// dbx names the image, its signer or a certificate in its chain.
    Forbidden,
    /// Nothing in db allows it.
    NotAllowed,
}

/// Whether db and dbx let `file` run: `Ok` with the `EFI_SIGNATURE_DATA` of the db entry that
/// allowed it (measured into PCR 7), or `Ok(None)` when Secure Boot is off.
pub fn check_image(file: &[u8]) -> Result<Option<Vec<u8>>, Refusal> {
    if !*ENFORCING.get() {
        return Ok(None);
    }
    let db = read("db", &guids::IMAGE_SECURITY_DATABASE).unwrap_or_default();
    let dbx = read("dbx", &guids::IMAGE_SECURITY_DATABASE).unwrap_or_default();
    match secure_boot::verify_image(file, &db, &dbx) {
        Verdict::Allowed(entry) => Ok(Some(entry.signature_data())),
        Verdict::Forbidden => Err(Refusal::Forbidden),
        Verdict::NotAllowed => Err(Refusal::NotAllowed),
    }
}

pub fn enforcing() -> bool {
    *ENFORCING.get()
}

pub fn init(secure_boot: bool) {
    let variables = VARIABLES.get();
    let visible = BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
    let global = guids::GLOBAL_VARIABLE;
    init_secure_boot(variables, secure_boot);
    variables.set_internal("AuditMode", &global, visible, &[0]);
    variables.set_internal("DeployedMode", &global, visible, &[0]);
    variables.set_internal(
        "OsIndicationsSupported",
        &global,
        visible,
        &OS_INDICATIONS_SUPPORTED.to_le_bytes(),
    );
    variables.set_internal(
        "BootOptionSupport",
        &global,
        visible,
        &BOOT_OPTION_SUPPORT.to_le_bytes(),
    );
    variables.set_internal("PlatformLangCodes", &global, visible, PLATFORM_LANGUAGES);
}

/// For the boot manager and others inside the firmware.
pub fn read(name: &str, guid: &Guid) -> Option<Vec<u8>> {
    let units: Vec<u16> = name.encode_utf16().collect();
    VARIABLES
        .get()
        .get(&units, guid, false)
        .ok()
        .map(|(_, data)| data)
}

pub fn write(name: &str, guid: &Guid, attributes: u32, data: &[u8]) {
    let units: Vec<u16> = name.encode_utf16().collect();
    let _ = VARIABLES.get().set(&units, guid, attributes, data, false);
}

/// The numbers of every `Boot####` variable, parseable or not.
pub fn boot_option_numbers() -> Vec<u16> {
    let variables = VARIABLES.get();
    let mut numbers = Vec::new();
    let (mut name, mut guid) = (Vec::new(), guids::GLOBAL_VARIABLE);
    while let Ok((next_name, next_guid)) = variables.next(&name, &guid, false) {
        if next_guid == guids::GLOBAL_VARIABLE
            && let Some(number) = crate::load_option::number_of(&text::to_string(&next_name))
        {
            numbers.push(number);
        }
        (name, guid) = (next_name, next_guid);
    }
    numbers
}

fn status_of(error: VariableError) -> Status {
    match error {
        VariableError::NotFound => NOT_FOUND,
        VariableError::InvalidParameter => INVALID_PARAMETER,
        VariableError::WriteProtected => WRITE_PROTECTED,
        VariableError::OutOfResources => OUT_OF_RESOURCES,
        VariableError::SecurityViolation => SECURITY_VIOLATION,
        VariableError::Unsupported => UNSUPPORTED,
        VariableError::DeviceError => DEVICE_ERROR,
    }
}

extern "efiapi" fn get_variable(
    name: *const u16,
    guid: *const Guid,
    attributes: *mut u32,
    size: *mut usize,
    data: *mut c_void,
) -> Status {
    memory::rewind_runtime_arena();
    if name.is_null() || guid.is_null() || size.is_null() {
        return INVALID_PARAMETER;
    }
    let name = unsafe { text::to_units(name) };
    let guid = unsafe { guid.read_unaligned() };
    let (stored_attributes, value) =
        match VARIABLES.get().get(&name, &guid, memory::in_runtime_mode()) {
            Ok(found) => found,
            Err(error) => return status_of(error),
        };
    unsafe {
        if !attributes.is_null() {
            attributes.write_unaligned(stored_attributes);
        }
        let capacity = size.read_unaligned();
        size.write_unaligned(value.len());
        if capacity < value.len() {
            return BUFFER_TOO_SMALL;
        }
        if !value.is_empty() {
            if data.is_null() {
                return INVALID_PARAMETER;
            }
            core::ptr::copy_nonoverlapping(value.as_ptr(), data.cast::<u8>(), value.len());
        }
    }
    SUCCESS
}

extern "efiapi" fn set_variable(
    name: *const u16,
    guid: *const Guid,
    attributes: u32,
    size: usize,
    data: *const c_void,
) -> Status {
    memory::rewind_runtime_arena();
    if name.is_null() || guid.is_null() || (size != 0 && data.is_null()) {
        return INVALID_PARAMETER;
    }
    let name = unsafe { text::to_units(name) };
    let guid = unsafe { guid.read_unaligned() };
    let data: &[u8] = if size == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(data.cast(), size) }
    };
    match VARIABLES
        .get()
        .set(&name, &guid, attributes, data, memory::in_runtime_mode())
    {
        Ok(()) => SUCCESS,
        Err(error) => status_of(error),
    }
}

extern "efiapi" fn get_next_variable_name(
    size: *mut usize,
    name: *mut u16,
    guid: *mut Guid,
) -> Status {
    memory::rewind_runtime_arena();
    if size.is_null() || name.is_null() || guid.is_null() {
        return INVALID_PARAMETER;
    }
    let capacity_units = unsafe { size.read_unaligned() } / 2;
    // The name must be NUL-terminated inside the buffer the caller says it has.
    let mut previous = Vec::new();
    loop {
        if previous.len() >= capacity_units {
            return INVALID_PARAMETER;
        }
        let unit = unsafe { name.add(previous.len()).read_unaligned() };
        if unit == 0 {
            break;
        }
        previous.push(unit);
    }
    let previous_guid = unsafe { guid.read_unaligned() };
    let (next, next_guid) =
        match VARIABLES
            .get()
            .next(&previous, &previous_guid, memory::in_runtime_mode())
        {
            Ok(found) => found,
            Err(error) => return status_of(error),
        };
    let needed = (next.len() + 1) * 2;
    unsafe {
        size.write_unaligned(needed);
        if capacity_units * 2 < needed {
            return BUFFER_TOO_SMALL;
        }
        for (index, unit) in next.iter().chain(core::iter::once(&0)).enumerate() {
            name.add(index).write_unaligned(*unit);
        }
        guid.write_unaligned(next_guid);
    }
    SUCCESS
}

extern "efiapi" fn query_variable_info(
    _attributes: u32,
    maximum_storage: *mut u64,
    remaining_storage: *mut u64,
    maximum_variable_size: *mut u64,
) -> Status {
    memory::rewind_runtime_arena();
    if maximum_storage.is_null() || remaining_storage.is_null() || maximum_variable_size.is_null() {
        return INVALID_PARAMETER;
    }
    let (maximum, remaining, largest) = VARIABLES.get().info();
    unsafe {
        maximum_storage.write_unaligned(maximum);
        remaining_storage.write_unaligned(remaining);
        maximum_variable_size.write_unaligned(largest);
    }
    SUCCESS
}

pub fn install(runtime: &mut RuntimeServices) {
    runtime.get_variable = get_variable;
    runtime.set_variable = set_variable;
    runtime.get_next_variable_name = get_next_variable_name;
    runtime.query_variable_info = query_variable_info;
}
