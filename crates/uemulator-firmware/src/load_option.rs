//! `EFI_LOAD_OPTION`: a `Boot####` variable's description, device path and optional data.

use alloc::string::String;
use alloc::vec::Vec;

use crate::devicepath::DevicePath;
use crate::uefi::Guid;

/// `LOAD_OPTION_*` attribute bits (UEFI 2.9 3.1.3).
pub const ACTIVE: u32 = 0x1;
/// Kept out of boot menus; still booted from `BootOrder`.
pub const HIDDEN: u32 = 0x8;
const CATEGORY_MASK: u32 = 0x1f00;
/// Applications (shells, diagnostics) are launched from the menu only, never by `BootOrder`.
const CATEGORY_APP: u32 = 0x100;
/// attributes u32, file path list length u16, then the description.
const FIXED_BYTES: usize = 6;

/// The optional data of the `Boot####` entries the firmware creates for the media it finds,
/// so it can tell them from the ones an OS wrote and delete them when the media goes away.
/// EDK2's BDS uses this GUID (`mBmAutoCreateBootOptionGuid`), so tools that know OVMF's
/// entries recognise ours.
pub const AUTO_CREATED: Guid = Guid::new(
    0x8108_ac4e,
    0x9f11,
    0x4d59,
    [0x85, 0x0e, 0xe2, 0x1a, 0x52, 0x2c, 0x59, 0xb2],
);

pub struct Candidate {
    pub description: String,
    pub path: DevicePath,
    pub options: Vec<u8>,
    /// The `Boot####` number, for `BootCurrent`.
    pub number: Option<u16>,
    pub attributes: u32,
}

impl Candidate {
    pub fn hidden(&self) -> bool {
        self.attributes & HIDDEN != 0
    }

    /// Whether `BootOrder` boots it without being picked from the menu.
    pub fn boots_by_default(&self) -> bool {
        self.attributes & CATEGORY_MASK != CATEGORY_APP
    }

    pub fn auto_created(&self) -> bool {
        self.options == AUTO_CREATED.to_bytes()
    }

    /// The image's load options: the optional data, except the marker on entries the firmware
    /// made, which would reach a Linux EFI stub as a garbage command line.
    pub fn load_options(&self) -> &[u8] {
        if self.auto_created() {
            &[]
        } else {
            &self.options
        }
    }
}

/// The number in a `Boot####` variable name: four upper-case hex digits (UEFI 2.9 3.3), so
/// `BootOrder`, `BootNext` and `Boot00a1` are not options.
pub fn number_of(name: &str) -> Option<u16> {
    let digits = name.strip_prefix("Boot")?;
    let upper_hex = |c: char| c.is_ascii_digit() || ('A'..='F').contains(&c);
    if digits.len() != 4 || !digits.chars().all(upper_hex) {
        return None;
    }
    u16::from_str_radix(digits, 16).ok()
}

/// Serialises an `EFI_LOAD_OPTION`.
pub fn build(attributes: u32, description: &str, path: &DevicePath, data: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&attributes.to_le_bytes());
    bytes.extend_from_slice(&(path.as_bytes().len() as u16).to_le_bytes());
    for unit in crate::text::to_utf16_nul(description) {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    bytes.extend_from_slice(path.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

pub fn u16_list(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// Parses an `EFI_LOAD_OPTION`; `None` for inactive or malformed ones.
pub fn parse(bytes: &[u8], number: u16) -> Option<Candidate> {
    let attributes = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    let path_length = u16::from_le_bytes(bytes.get(4..6)?.try_into().ok()?) as usize;
    if attributes & ACTIVE == 0 {
        return None;
    }
    let rest = bytes.get(FIXED_BYTES..)?;
    let units = u16_list(rest);
    let name_units = units.iter().position(|&unit| unit == 0)?;
    let description = crate::text::to_string(&units[..name_units]);
    let path_start = (name_units + 1) * 2;
    let path_bytes = rest.get(path_start..path_start + path_length)?;
    let path = unsafe { DevicePath::from_raw(path_bytes.as_ptr()) }?;
    Some(Candidate {
        description,
        path,
        options: rest[path_start + path_length..].to_vec(),
        number: Some(number),
        attributes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_option(active: bool, description: &str, path: &DevicePath, data: &[u8]) -> Vec<u8> {
        build(active as u32, description, path, data)
    }

    #[test]
    fn a_load_option_gives_its_path_and_options() {
        let path = DevicePath::new().file_path("\\EFI\\fedora\\shimx64.efi");
        let bytes = load_option(true, "Fedora", &path, b"o\0p\0");
        let candidate = parse(&bytes, 3).unwrap();
        assert_eq!(candidate.description, "Fedora");
        assert_eq!(candidate.path, path);
        assert_eq!(candidate.options, b"o\0p\0");
        assert_eq!(candidate.number, Some(3));
    }

    #[test]
    fn inactive_and_truncated_options_are_skipped() {
        let path = DevicePath::new().file_path("\\x");
        assert!(parse(&load_option(false, "Off", &path, b""), 0).is_none());
        let mut truncated = load_option(true, "Short", &path, b"");
        truncated.truncate(truncated.len() - 6);
        assert!(parse(&truncated, 0).is_none());
        assert!(parse(&[1, 0], 0).is_none());
    }

    #[test]
    fn only_boot_and_four_upper_case_hex_digits_name_an_option() {
        assert_eq!(number_of("Boot0000"), Some(0));
        assert_eq!(number_of("Boot00AF"), Some(0xaf));
        assert_eq!(number_of("BootOrder"), None);
        assert_eq!(number_of("BootNext"), None);
        assert_eq!(number_of("Boot00af"), None);
        assert_eq!(number_of("Boot000"), None);
        assert_eq!(number_of("Driver0000"), None);
    }

    #[test]
    fn attributes_decide_menu_and_default_boot() {
        let path = DevicePath::new().file_path("\\x");
        let hidden = parse(&build(ACTIVE | HIDDEN, "Hidden", &path, b""), 1).unwrap();
        assert!(hidden.hidden());
        assert!(hidden.boots_by_default());
        let shell = parse(&build(ACTIVE | CATEGORY_APP, "Shell", &path, b""), 2).unwrap();
        assert!(!shell.hidden());
        assert!(!shell.boots_by_default());
    }

    #[test]
    fn the_auto_created_marker_is_not_passed_to_the_image() {
        let path = DevicePath::new().file_path("\\EFI\\BOOT\\BOOTX64.EFI");
        let marker = AUTO_CREATED.to_bytes();
        let made = parse(&build(ACTIVE, "UEFI disk", &path, &marker), 1).unwrap();
        assert!(made.auto_created());
        assert!(made.load_options().is_empty());
        let written = parse(&build(ACTIVE, "Linux", &path, b"q\0"), 2).unwrap();
        assert!(!written.auto_created());
        assert_eq!(written.load_options(), b"q\0");
    }
}
