//! Device paths: building them for what we publish, walking and printing the ones applications
//! hand us, and the two protocols (`DevicePathToText`, `DevicePathUtilities`) around them.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::uefi::Guid;

pub const TYPE_HARDWARE: u8 = 1;
pub const TYPE_ACPI: u8 = 2;
pub const TYPE_MESSAGING: u8 = 3;
pub const TYPE_MEDIA: u8 = 4;
pub const TYPE_END: u8 = 0x7f;
pub const SUBTYPE_END_ENTIRE: u8 = 0xff;
pub const SUBTYPE_PCI: u8 = 1;
pub const SUBTYPE_ACPI: u8 = 1;
pub const SUBTYPE_HARD_DRIVE: u8 = 1;
pub const SUBTYPE_VENDOR_MEDIA: u8 = 3;
pub const SUBTYPE_FILE_PATH: u8 = 4;
const SUBTYPE_VENDOR_HARDWARE: u8 = 4;
const SUBTYPE_VENDOR_MESSAGING: u8 = 10;
const NODE_HEADER_BYTES: usize = 4;
/// ACPI `_HID` of a PCI root bridge, EISA-encoded `PNP0A03`.
pub const PCI_ROOT_HID: u32 = 0x0a03_41d0;
const END_NODE: [u8; 4] = [TYPE_END, SUBTYPE_END_ENTIRE, 4, 0];
const MAX_PATH_BYTES: usize = 64 * 1024;

pub const PARTITION_FORMAT_MBR: u8 = 1;
pub const PARTITION_FORMAT_GPT: u8 = 2;
pub const SIGNATURE_MBR: u8 = 1;
pub const SIGNATURE_GUID: u8 = 2;

/// An owned device path in its wire format, end node included.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DevicePath(Vec<u8>);

impl DevicePath {
    pub fn new() -> Self {
        Self(END_NODE.to_vec())
    }

    /// Copies the path at `pointer`, which is untrusted: the walk stops (with `None`) at a node
    /// shorter than its header or after `MAX_PATH_BYTES`.
    ///
    /// # Safety
    /// `pointer` must be null or readable up to the path's end node.
    pub unsafe fn from_raw(pointer: *const u8) -> Option<Self> {
        if pointer.is_null() {
            return None;
        }
        let mut bytes = Vec::new();
        loop {
            let base = bytes.len();
            let header = [0, 1, 2, 3].map(|offset| unsafe { pointer.add(base + offset).read() });
            let length = u16::from_le_bytes([header[2], header[3]]) as usize;
            if length < NODE_HEADER_BYTES || base + length > MAX_PATH_BYTES {
                return None;
            }
            bytes.extend((0..length).map(|offset| unsafe { pointer.add(base + offset).read() }));
            if header[0] == TYPE_END && header[1] == SUBTYPE_END_ENTIRE {
                return Some(Self(bytes));
            }
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// The nodes before the end node, as (type, subtype, payload).
    pub fn nodes(&self) -> impl Iterator<Item = Node<'_>> {
        nodes(&self.0)
    }

    fn push_node(mut self, kind: u8, subtype: u8, payload: &[u8]) -> Self {
        self.0.truncate(self.0.len() - END_NODE.len());
        self.0.push(kind);
        self.0.push(subtype);
        self.0
            .extend_from_slice(&((NODE_HEADER_BYTES + payload.len()) as u16).to_le_bytes());
        self.0.extend_from_slice(payload);
        self.0.extend_from_slice(&END_NODE);
        self
    }

    pub fn acpi_pci_root(self) -> Self {
        let mut payload = [0u8; 8];
        payload[..4].copy_from_slice(&PCI_ROOT_HID.to_le_bytes());
        self.push_node(TYPE_ACPI, SUBTYPE_ACPI, &payload)
    }

    pub fn pci(self, device: u8, function: u8) -> Self {
        self.push_node(TYPE_HARDWARE, SUBTYPE_PCI, &[function, device])
    }

    pub fn hard_drive(
        self,
        partition_number: u32,
        first_block: u64,
        block_count: u64,
        signature: [u8; 16],
        format: u8,
        signature_type: u8,
    ) -> Self {
        let mut payload = Vec::with_capacity(38);
        payload.extend_from_slice(&partition_number.to_le_bytes());
        payload.extend_from_slice(&first_block.to_le_bytes());
        payload.extend_from_slice(&block_count.to_le_bytes());
        payload.extend_from_slice(&signature);
        payload.push(format);
        payload.push(signature_type);
        self.push_node(TYPE_MEDIA, SUBTYPE_HARD_DRIVE, &payload)
    }

    /// A vendor-defined media node: `guid` and no vendor data.
    pub fn vendor_media(self, guid: Guid) -> Self {
        self.push_node(TYPE_MEDIA, SUBTYPE_VENDOR_MEDIA, &guid.to_bytes())
    }

    pub fn file_path(self, path: &str) -> Self {
        let payload: Vec<u8> = path
            .encode_utf16()
            .chain(core::iter::once(0))
            .flat_map(|unit| unit.to_le_bytes())
            .collect();
        self.push_node(TYPE_MEDIA, SUBTYPE_FILE_PATH, &payload)
    }

    /// `self` followed by the nodes of `other`.
    pub fn append(&self, other: &DevicePath) -> Self {
        let mut bytes = self.0[..self.0.len() - END_NODE.len()].to_vec();
        bytes.extend_from_slice(&other.0);
        Self(bytes)
    }

    /// Whether `prefix`'s nodes are the first nodes of `self`.
    pub fn starts_with(&self, prefix: &DevicePath) -> bool {
        let length = prefix.0.len() - END_NODE.len();
        self.0.len() >= length && self.0[..length] == prefix.0[..length]
    }

    #[cfg(test)]
    /// The nodes of `self` after `prefix`, as a path of their own.
    pub fn strip_prefix(&self, prefix: &DevicePath) -> Option<DevicePath> {
        let length = prefix.0.len() - END_NODE.len();
        self.starts_with(prefix)
            .then(|| Self(self.0[length..].to_vec()))
    }

    /// The hard-drive node's partition, if the path has one: (number, first block, blocks).
    pub fn partition(&self) -> Option<(u32, u64, u64)> {
        self.nodes()
            .find(|node| node.kind == TYPE_MEDIA && node.subtype == SUBTYPE_HARD_DRIVE)
            .filter(|node| node.payload.len() >= 20)
            .map(|node| {
                let p = node.payload;
                (
                    u32::from_le_bytes(p[0..4].try_into().unwrap()),
                    u64::from_le_bytes(p[4..12].try_into().unwrap()),
                    u64::from_le_bytes(p[12..20].try_into().unwrap()),
                )
            })
    }

    /// What identifies the partition a hard-drive node names: partition number, then signature,
    /// partition format and signature type (UEFI 2.9 10.3.5.1). The start and size are left
    /// out, as the short-form match in 3.1.2 leaves them out.
    fn partition_identity(payload: &[u8]) -> Option<([u8; 4], [u8; 18])> {
        const NUMBER: core::ops::Range<usize> = 0..4;
        const IDENTITY: core::ops::Range<usize> = 20..38;
        Some((
            payload.get(NUMBER)?.try_into().ok()?,
            payload.get(IDENTITY)?.try_into().ok()?,
        ))
    }

    /// For a short-form path, one that starts at a hard-drive node as efibootmgr writes them
    /// (`HD(1,GPT,...)/\EFI\debian\shimx64.efi`): the full path through whichever of
    /// `volumes` is that partition, or `None` when the path is not of that form or no volume
    /// matches (UEFI 2.9 3.1.2). Without this, nothing but the file name would pick the volume,
    /// and with two installs that boots the wrong one.
    pub fn expand_hard_drive<'a>(
        &self,
        volumes: impl IntoIterator<Item = &'a DevicePath>,
    ) -> Option<DevicePath> {
        let first = self.nodes().next()?;
        if (first.kind, first.subtype) != (TYPE_MEDIA, SUBTYPE_HARD_DRIVE) {
            return None;
        }
        let wanted = Self::partition_identity(first.payload)?;
        let rest = Self(self.0[NODE_HEADER_BYTES + first.payload.len()..].to_vec());
        volumes.into_iter().find_map(|volume| {
            let partition = volume
                .nodes()
                .filter(|node| (node.kind, node.subtype) == (TYPE_MEDIA, SUBTYPE_HARD_DRIVE))
                .last()?;
            (Self::partition_identity(partition.payload)? == wanted).then(|| volume.append(&rest))
        })
    }

    /// The path from its first file-path node on, or the whole path when it has none. This is what
    /// `LoadedImage.FilePath` holds: shim prints it and opens `\\EFI\\BOOT\\grubx64.efi` next to
    /// itself by text, which fails if the device nodes are in it.
    pub fn file_part(&self) -> DevicePath {
        let mut at = 0;
        for node in self.nodes() {
            if node.kind == TYPE_MEDIA && node.subtype == SUBTYPE_FILE_PATH {
                return Self(self.0[at..].to_vec());
            }
            at += NODE_HEADER_BYTES + node.payload.len();
        }
        self.clone()
    }

    /// The text of the final file-path node, if the path ends in one.
    pub fn file_name(&self) -> Option<String> {
        self.nodes()
            .filter(|node| node.kind == TYPE_MEDIA && node.subtype == SUBTYPE_FILE_PATH)
            .last()
            .map(|node| file_node_text(node.payload))
    }

    /// The paths of every instance when the path is a list (`BootOption`s may hold several).
    pub fn text(&self) -> String {
        to_text(&self.0)
    }
}

pub struct Node<'a> {
    pub kind: u8,
    pub subtype: u8,
    pub payload: &'a [u8],
}

/// The nodes in `bytes` up to the first end node of any kind.
pub fn nodes(bytes: &[u8]) -> impl Iterator<Item = Node<'_>> {
    let mut at = 0;
    core::iter::from_fn(move || {
        let header = bytes.get(at..at + NODE_HEADER_BYTES)?;
        let length = u16::from_le_bytes([header[2], header[3]]) as usize;
        if header[0] == TYPE_END || length < NODE_HEADER_BYTES || at + length > bytes.len() {
            return None;
        }
        let node = Node {
            kind: header[0],
            subtype: header[1],
            payload: &bytes[at + NODE_HEADER_BYTES..at + length],
        };
        at += length;
        Some(node)
    })
}

fn file_node_text(payload: &[u8]) -> String {
    let units: Vec<u16> = payload
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .take_while(|&unit| unit != 0)
        .collect();
    crate::text::to_string(&units)
}

/// UEFI 2.9 section 10.6.1.6: the display form, nodes joined by `/` and file names verbatim.
pub fn to_text(path: &[u8]) -> String {
    let mut text = String::new();
    let mut first = true;
    for node in nodes(path) {
        if !first && !text.ends_with('\\') {
            text.push('/');
        }
        first = false;
        let p = node.payload;
        match (node.kind, node.subtype) {
            (TYPE_ACPI, SUBTYPE_ACPI) if p.len() >= 8 => {
                let hid = u32::from_le_bytes(p[..4].try_into().unwrap());
                let uid = u32::from_le_bytes(p[4..8].try_into().unwrap());
                if hid == PCI_ROOT_HID {
                    let _ = write!(text, "PciRoot({uid:#x})");
                } else {
                    let _ = write!(text, "Acpi({hid:#x},{uid:#x})");
                }
            }
            (TYPE_HARDWARE, SUBTYPE_PCI) if p.len() >= 2 => {
                let _ = write!(text, "Pci({:#x},{:#x})", p[1], p[0]);
            }
            (TYPE_MEDIA, SUBTYPE_HARD_DRIVE) if p.len() >= 38 => {
                let number = u32::from_le_bytes(p[0..4].try_into().unwrap());
                let start = u64::from_le_bytes(p[4..12].try_into().unwrap());
                let size = u64::from_le_bytes(p[12..20].try_into().unwrap());
                if p[37] == SIGNATURE_GUID {
                    let guid = Guid::from_bytes(&p[20..36]);
                    let _ = write!(
                        text,
                        "HD({number},GPT,{},{start:#x},{size:#x})",
                        core::str::from_utf8(&guid.text()).unwrap_or("")
                    );
                } else {
                    let signature = u32::from_le_bytes(p[20..24].try_into().unwrap());
                    let _ = write!(
                        text,
                        "HD({number},MBR,{signature:#010x},{start:#x},{size:#x})"
                    );
                }
            }
            (TYPE_MEDIA, SUBTYPE_FILE_PATH) => text.push_str(&file_node_text(p)),
            (TYPE_MEDIA, SUBTYPE_VENDOR_MEDIA)
            | (TYPE_HARDWARE, SUBTYPE_VENDOR_HARDWARE)
            | (TYPE_MESSAGING, SUBTYPE_VENDOR_MESSAGING)
                if p.len() >= 16 =>
            {
                let guid = Guid::from_bytes(&p[..16]);
                let _ = write!(
                    text,
                    "VenMedia({})",
                    core::str::from_utf8(&guid.text()).unwrap_or("")
                );
            }
            _ => {
                let _ = write!(text, "Path({},{}", node.kind, node.subtype);
                for byte in p {
                    let _ = write!(text, ",{byte:02x}");
                }
                text.push(')');
            }
        }
    }
    text
}

#[cfg(target_os = "none")]
pub mod protocol {
    //! `EFI_DEVICE_PATH_TO_TEXT_PROTOCOL` and `EFI_DEVICE_PATH_UTILITIES_PROTOCOL`.

    use core::ffi::c_void;

    use super::*;
    use crate::memory;
    use crate::uefi::{self, DevicePathNode};

    #[repr(C)]
    pub struct ToText {
        convert_device_node: extern "efiapi" fn(*const DevicePathNode, u8, u8) -> *mut u16,
        convert_device_path: extern "efiapi" fn(*const DevicePathNode, u8, u8) -> *mut u16,
    }

    #[repr(C)]
    pub struct Utilities {
        get_device_path_size: extern "efiapi" fn(*const DevicePathNode) -> usize,
        duplicate_device_path: extern "efiapi" fn(*const DevicePathNode) -> *mut DevicePathNode,
        append_device_path:
            extern "efiapi" fn(*const DevicePathNode, *const DevicePathNode) -> *mut DevicePathNode,
        append_device_node:
            extern "efiapi" fn(*const DevicePathNode, *const DevicePathNode) -> *mut DevicePathNode,
        append_device_path_instance:
            extern "efiapi" fn(*const DevicePathNode, *const DevicePathNode) -> *mut DevicePathNode,
        get_next_device_path_instance:
            extern "efiapi" fn(*mut *mut DevicePathNode, *mut usize) -> *mut DevicePathNode,
        is_multi_instance: extern "efiapi" fn(*const DevicePathNode) -> u8,
        create_device_node: extern "efiapi" fn(u8, u8, u16) -> *mut DevicePathNode,
    }

    pub static TO_TEXT: ToText = ToText {
        convert_device_node,
        convert_device_path,
    };

    pub static UTILITIES: Utilities = Utilities {
        get_device_path_size: get_size,
        duplicate_device_path: duplicate,
        append_device_path: append_path,
        append_device_node: append_node,
        append_device_path_instance: append_instance,
        get_next_device_path_instance: next_instance,
        is_multi_instance,
        create_device_node: create_node,
    };

    fn pool_copy(bytes: &[u8]) -> *mut c_void {
        let pointer = memory::pool_allocate(uefi::memory_type::BOOT_SERVICES_DATA, bytes.len());
        if !pointer.is_null() {
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len()) };
        }
        pointer.cast()
    }

    fn pool_utf16(text: &str) -> *mut u16 {
        let units = crate::text::to_utf16_nul(text);
        let bytes: Vec<u8> = units.iter().flat_map(|unit| unit.to_le_bytes()).collect();
        pool_copy(&bytes).cast()
    }

    /// A single node, as the text of a path of one node.
    extern "efiapi" fn convert_device_node(node: *const DevicePathNode, _: u8, _: u8) -> *mut u16 {
        if node.is_null() {
            return core::ptr::null_mut();
        }
        let length = unsafe { (*node).node_length() };
        let mut bytes = unsafe { core::slice::from_raw_parts(node.cast::<u8>(), length) }.to_vec();
        bytes.extend_from_slice(&END_NODE);
        pool_utf16(&to_text(&bytes))
    }

    extern "efiapi" fn convert_device_path(path: *const DevicePathNode, _: u8, _: u8) -> *mut u16 {
        match unsafe { DevicePath::from_raw(path.cast()) } {
            Some(path) => pool_utf16(&path.text()),
            None => core::ptr::null_mut(),
        }
    }

    extern "efiapi" fn get_size(path: *const DevicePathNode) -> usize {
        unsafe { DevicePath::from_raw(path.cast()) }.map_or(0, |path| path.0.len())
    }

    extern "efiapi" fn duplicate(path: *const DevicePathNode) -> *mut DevicePathNode {
        match unsafe { DevicePath::from_raw(path.cast()) } {
            Some(path) => pool_copy(&path.0).cast(),
            None => core::ptr::null_mut(),
        }
    }

    extern "efiapi" fn append_path(
        first: *const DevicePathNode,
        second: *const DevicePathNode,
    ) -> *mut DevicePathNode {
        let first = unsafe { DevicePath::from_raw(first.cast()) }.unwrap_or_default();
        let second = unsafe { DevicePath::from_raw(second.cast()) }.unwrap_or_default();
        pool_copy(&first.append(&second).0).cast()
    }

    extern "efiapi" fn append_node(
        path: *const DevicePathNode,
        node: *const DevicePathNode,
    ) -> *mut DevicePathNode {
        if node.is_null() {
            return duplicate(path);
        }
        let length = unsafe { (*node).node_length() };
        let mut single = unsafe { core::slice::from_raw_parts(node.cast::<u8>(), length) }.to_vec();
        single.extend_from_slice(&END_NODE);
        let first = unsafe { DevicePath::from_raw(path.cast()) }.unwrap_or_default();
        pool_copy(&first.append(&DevicePath(single)).0).cast()
    }

    extern "efiapi" fn append_instance(
        _: *const DevicePathNode,
        _: *const DevicePathNode,
    ) -> *mut DevicePathNode {
        core::ptr::null_mut()
    }

    extern "efiapi" fn next_instance(
        _: *mut *mut DevicePathNode,
        _: *mut usize,
    ) -> *mut DevicePathNode {
        core::ptr::null_mut()
    }

    extern "efiapi" fn is_multi_instance(_: *const DevicePathNode) -> u8 {
        0
    }

    extern "efiapi" fn create_node(kind: u8, subtype: u8, length: u16) -> *mut DevicePathNode {
        if (length as usize) < NODE_HEADER_BYTES {
            return core::ptr::null_mut();
        }
        let mut bytes = alloc::vec![0u8; length as usize];
        bytes[0] = kind;
        bytes[1] = subtype;
        bytes[2..4].copy_from_slice(&length.to_le_bytes());
        pool_copy(&bytes).cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_path() -> DevicePath {
        DevicePath::new().acpi_pci_root().pci(3, 0)
    }

    fn partition_path(disk: u8, number: u32, guid: Guid) -> DevicePath {
        DevicePath::new().acpi_pci_root().pci(disk, 0).hard_drive(
            number,
            0x800,
            0x40000,
            guid.to_bytes(),
            PARTITION_FORMAT_GPT,
            SIGNATURE_GUID,
        )
    }

    #[test]
    fn a_short_form_path_expands_through_the_matching_partition() {
        let first = Guid::new(1, 2, 3, [4; 8]);
        let second = Guid::new(5, 6, 7, [8; 8]);
        let volumes = [partition_path(1, 1, first), partition_path(2, 1, second)];
        // efibootmgr writes the start and size too, but they are not part of the match.
        let short = DevicePath::new()
            .hard_drive(
                1,
                0,
                0,
                second.to_bytes(),
                PARTITION_FORMAT_GPT,
                SIGNATURE_GUID,
            )
            .file_path("\\EFI\\debian\\shimx64.efi");
        assert_eq!(
            short.expand_hard_drive(&volumes),
            Some(volumes[1].clone().file_path("\\EFI\\debian\\shimx64.efi"))
        );
    }

    #[test]
    fn only_short_forms_with_a_known_partition_expand() {
        let guid = Guid::new(1, 2, 3, [4; 8]);
        let volumes = [partition_path(1, 1, guid)];
        let full = volumes[0].clone().file_path("\\x.efi");
        assert_eq!(full.expand_hard_drive(&volumes), None);
        let file_only = DevicePath::new().file_path("\\x.efi");
        assert_eq!(file_only.expand_hard_drive(&volumes), None);
        let other_partition = DevicePath::new()
            .hard_drive(
                2,
                0,
                0,
                guid.to_bytes(),
                PARTITION_FORMAT_GPT,
                SIGNATURE_GUID,
            )
            .file_path("\\x.efi");
        assert_eq!(other_partition.expand_hard_drive(&volumes), None);
    }

    #[test]
    fn builds_and_prints_a_boot_path() {
        let guid = Guid::new(0x11223344, 0x5566, 0x7788, [1, 2, 3, 4, 5, 6, 7, 8]);
        let path = disk_path()
            .hard_drive(
                1,
                0x800,
                0x40000,
                guid.to_bytes(),
                PARTITION_FORMAT_GPT,
                SIGNATURE_GUID,
            )
            .file_path("\\EFI\\BOOT\\BOOTX64.EFI");
        assert_eq!(
            path.text(),
            "PciRoot(0x0)/Pci(0x3,0x0)/HD(1,GPT,11223344-5566-7788-0102-030405060708,0x800,0x40000)/\\EFI\\BOOT\\BOOTX64.EFI"
        );
        assert_eq!(path.partition(), Some((1, 0x800, 0x40000)));
        assert_eq!(
            path.file_name().as_deref(),
            Some("\\EFI\\BOOT\\BOOTX64.EFI")
        );
        assert_eq!(&path.as_bytes()[path.as_bytes().len() - 4..], &END_NODE);
    }

    #[test]
    fn the_file_part_drops_the_device_nodes() {
        let file = DevicePath::new().file_path("\\EFI\\BOOT\\BOOTX64.EFI");
        let full = disk_path().append(&file);
        assert_eq!(full.file_part(), file);
        assert_eq!(file.file_part(), file);
        assert_eq!(disk_path().file_part(), disk_path());
    }

    #[test]
    fn mbr_partitions_print_their_signature() {
        let mut signature = [0u8; 16];
        signature[..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        let path =
            disk_path().hard_drive(2, 100, 200, signature, PARTITION_FORMAT_MBR, SIGNATURE_MBR);
        assert_eq!(
            path.text(),
            "PciRoot(0x0)/Pci(0x3,0x0)/HD(2,MBR,0xdeadbeef,0x64,0xc8)"
        );
    }

    #[test]
    fn prefixes_and_appending() {
        let disk = disk_path();
        let file = DevicePath::new().file_path("\\x");
        let full = disk.append(&file);
        assert!(full.starts_with(&disk));
        assert!(!disk.starts_with(&full));
        assert_eq!(full.strip_prefix(&disk), Some(file));
        assert_eq!(DevicePath::new().text(), "");
    }

    #[test]
    fn copies_from_raw_memory_and_refuses_malformed_paths() {
        let path = disk_path();
        let copy = unsafe { DevicePath::from_raw(path.as_bytes().as_ptr()) };
        assert_eq!(copy, Some(path));
        assert_eq!(unsafe { DevicePath::from_raw(core::ptr::null()) }, None);
        let zero_length_node = [TYPE_HARDWARE, 1, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            unsafe { DevicePath::from_raw(zero_length_node.as_ptr()) },
            None
        );
    }

    #[test]
    fn unknown_nodes_print_as_raw_paths() {
        let path = DevicePath::new().push_node(9, 9, &[0xab]);
        assert_eq!(path.text(), "Path(9,9,ab)");
    }

    #[test]
    fn the_initramfs_path_is_the_vendor_media_node_the_linux_stub_looks_for() {
        // drivers/firmware/efi/libstub/efi-stub-helper.c: efi_initrd_device_path.
        let mut linux = vec![TYPE_MEDIA, SUBTYPE_VENDOR_MEDIA, 20, 0];
        linux.extend_from_slice(&[
            0x27, 0xe4, 0x68, 0x55, 0xfc, 0x68, 0x3d, 0x4f, 0xac, 0x74, 0xca, 0x55, 0x52, 0x31,
            0xcc, 0x68,
        ]);
        linux.extend_from_slice(&[TYPE_END, SUBTYPE_END_ENTIRE, 4, 0]);
        let ours = DevicePath::new().vendor_media(crate::uefi::guids::LINUX_EFI_INITRD_MEDIA);
        assert_eq!(ours.as_bytes(), linux.as_slice());
    }
}
