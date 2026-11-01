#![allow(dead_code)] // the specification's vocabulary, only part of which is used
//! The UEFI 2.9 ABI: status codes, GUIDs, the tables and the structures that cross it. Layouts
//! follow the specification; every function pointer is `extern "efiapi"` (the Microsoft x64
//! convention) and every pointer an application hands us is untrusted.

use core::ffi::c_void;

pub type Status = usize;
pub type Handle = *mut c_void;
pub type Event = *mut c_void;

const ERROR_BIT: usize = 1 << 63;
pub const SUCCESS: Status = 0;
pub const WARN_UNKNOWN_GLYPH: Status = 1;
pub const LOAD_ERROR: Status = ERROR_BIT | 1;
pub const INVALID_PARAMETER: Status = ERROR_BIT | 2;
pub const UNSUPPORTED: Status = ERROR_BIT | 3;
pub const BAD_BUFFER_SIZE: Status = ERROR_BIT | 4;
pub const BUFFER_TOO_SMALL: Status = ERROR_BIT | 5;
pub const NOT_READY: Status = ERROR_BIT | 6;
pub const DEVICE_ERROR: Status = ERROR_BIT | 7;
pub const WRITE_PROTECTED: Status = ERROR_BIT | 8;
pub const OUT_OF_RESOURCES: Status = ERROR_BIT | 9;
pub const VOLUME_CORRUPTED: Status = ERROR_BIT | 10;
pub const NO_MEDIA: Status = ERROR_BIT | 12;
pub const MEDIA_CHANGED: Status = ERROR_BIT | 13;
pub const NOT_FOUND: Status = ERROR_BIT | 14;
pub const ACCESS_DENIED: Status = ERROR_BIT | 15;
pub const TIMEOUT: Status = ERROR_BIT | 18;
pub const NOT_STARTED: Status = ERROR_BIT | 19;
pub const ALREADY_STARTED: Status = ERROR_BIT | 20;
pub const ABORTED: Status = ERROR_BIT | 21;
pub const SECURITY_VIOLATION: Status = ERROR_BIT | 26;
pub const END_OF_FILE: Status = ERROR_BIT | 31;

pub fn is_error(status: Status) -> bool {
    status & ERROR_BIT != 0
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl Guid {
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }

    pub fn to_bytes(self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.data1.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.data2.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.data3.to_le_bytes());
        bytes[8..].copy_from_slice(&self.data4);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data1: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            data2: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            data3: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            data4: bytes[8..16].try_into().unwrap(),
        }
    }

    /// The registry text form, `8-4-4-4-12` lowercase hex. Variable keys use it, so the
    /// nvram file stays readable.
    pub fn text(&self) -> [u8; 36] {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = [b'-'; 36];
        let bytes = self.to_bytes();
        for (digit_pair, byte_index) in TEXT_ORDER.iter().enumerate() {
            let at = text_offset(digit_pair);
            text[at] = HEX[(bytes[*byte_index] >> 4) as usize];
            text[at + 1] = HEX[(bytes[*byte_index] & 15) as usize];
        }
        text
    }

    /// Inverse of `text`; `None` for anything that is not a well-formed GUID string.
    pub fn parse(text: &[u8]) -> Option<Self> {
        if text.len() != 36 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for (digit_pair, byte_index) in TEXT_ORDER.iter().enumerate() {
            let at = text_offset(digit_pair);
            let digit = |c: u8| (c as char).to_digit(16);
            bytes[*byte_index] = (digit(text[at])? * 16 + digit(text[at + 1])?) as u8;
        }
        let dashes_ok = [8, 13, 18, 23].iter().all(|&at| text[at] == b'-');
        dashes_ok.then(|| Self::from_bytes(&bytes))
    }
}

/// Which byte of the in-memory GUID each hex digit pair of the text form is: the first three
/// fields print as numbers, so their bytes come out reversed.
const TEXT_ORDER: [usize; 16] = [3, 2, 1, 0, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15];

/// Offset of digit pair `index` in the text form: a dash follows pairs 4, 6, 8 and 10.
fn text_offset(index: usize) -> usize {
    2 * index + [4, 6, 8, 10].iter().filter(|&&dash| index >= dash).count()
}

pub mod guids {
    use super::Guid;

    pub const LOADED_IMAGE: Guid = Guid::new(
        0x5b1b31a1,
        0x9562,
        0x11d2,
        [0x8e, 0x3f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const LOADED_IMAGE_DEVICE_PATH: Guid = Guid::new(
        0xbc62157e,
        0x3e33,
        0x4fec,
        [0x99, 0x20, 0x2d, 0x3b, 0x36, 0xd7, 0x50, 0xdf],
    );
    pub const DEVICE_PATH: Guid = Guid::new(
        0x09576e91,
        0x6d3f,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const DEVICE_PATH_TO_TEXT: Guid = Guid::new(
        0x8b843e20,
        0x8132,
        0x4852,
        [0x90, 0xcc, 0x55, 0x1a, 0x4e, 0x4a, 0x7f, 0x1c],
    );
    pub const DEVICE_PATH_UTILITIES: Guid = Guid::new(
        0x0379be4e,
        0xd706,
        0x437d,
        [0xb0, 0x37, 0xed, 0xb8, 0x2f, 0xb7, 0x72, 0xa4],
    );
    pub const SIMPLE_TEXT_INPUT: Guid = Guid::new(
        0x387477c1,
        0x69c7,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const SIMPLE_TEXT_INPUT_EX: Guid = Guid::new(
        0xdd9e7534,
        0x7762,
        0x4698,
        [0x8c, 0x14, 0xf5, 0x85, 0x17, 0xa6, 0x25, 0xaa],
    );
    pub const SIMPLE_TEXT_OUTPUT: Guid = Guid::new(
        0x387477c2,
        0x69c7,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const BLOCK_IO: Guid = Guid::new(
        0x964e5b21,
        0x6459,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const DISK_IO: Guid = Guid::new(
        0xce345171,
        0xba0b,
        0x11d2,
        [0x8e, 0x4f, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const SIMPLE_FILE_SYSTEM: Guid = Guid::new(
        0x964e5b22,
        0x6459,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const FILE_INFO: Guid = Guid::new(
        0x09576e92,
        0x6d3f,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const FILE_SYSTEM_INFO: Guid = Guid::new(
        0x09576e93,
        0x6d3f,
        0x11d2,
        [0x8e, 0x39, 0x00, 0xa0, 0xc9, 0x69, 0x72, 0x3b],
    );
    pub const FILE_SYSTEM_VOLUME_LABEL: Guid = Guid::new(
        0xdb47d7d3,
        0xfe81,
        0x11d3,
        [0x9a, 0x35, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d],
    );
    /// `5568e427-68fc-4f3d-ac74-ca555231cc68`: the vendor-media device path node under which the
    /// Linux EFI stub looks for its initramfs.
    pub const LINUX_EFI_INITRD_MEDIA: Guid = Guid::new(
        0x5568e427,
        0x68fc,
        0x4f3d,
        [0xac, 0x74, 0xca, 0x55, 0x52, 0x31, 0xcc, 0x68],
    );
    /// Xen's EFI loader looks for its configuration file, dom0 kernel and ramdisk under these
    /// vendor-media nodes before it looks on disk (Edera's Xen, xen/common/efi/boot.c).
    pub const XEN_EFI_CONFIG_MEDIA: Guid = Guid::new(
        0xbf61f458,
        0xa28e,
        0x46cd,
        [0x93, 0xd7, 0x07, 0xda, 0xc5, 0xe8, 0xcd, 0x66],
    );
    pub const XEN_EFI_KERNEL_MEDIA: Guid = Guid::new(
        0x4010c8bf,
        0x6ced,
        0x40f5,
        [0xa5, 0x3f, 0xe8, 0x20, 0xae, 0xe8, 0xf3, 0x4b],
    );
    pub const XEN_EFI_RAMDISK_MEDIA: Guid = Guid::new(
        0x5db1fd01,
        0xc3cb,
        0x4812,
        [0xb2, 0xba, 0x87, 0x91, 0xe5, 0x2d, 0x4a, 0x89],
    );
    pub const LOAD_FILE2: Guid = Guid::new(
        0x4006c0c1,
        0xfcb3,
        0x403e,
        [0x99, 0x6d, 0x4a, 0x6c, 0x87, 0x24, 0xe0, 0x6d],
    );
    pub const RNG: Guid = Guid::new(
        0x3152bca5,
        0xeade,
        0x433d,
        [0x86, 0x2e, 0xc0, 0x1c, 0xdc, 0x29, 0x1f, 0x44],
    );
    pub const RNG_ALGORITHM_RAW: Guid = Guid::new(
        0xe43176d7,
        0xb6e8,
        0x4827,
        [0xb7, 0x84, 0x7f, 0xfd, 0xc4, 0xb6, 0x85, 0x61],
    );
    pub const PCI_IO: Guid = Guid::new(
        0x4cf5b200,
        0x68b8,
        0x4ca5,
        [0x9e, 0xec, 0xb2, 0x3e, 0x3f, 0x50, 0x02, 0x9a],
    );
    pub const ACPI_20_TABLE: Guid = Guid::new(
        0x8868e871,
        0xe4f1,
        0x11d3,
        [0xbc, 0x22, 0x00, 0x80, 0xc7, 0x3c, 0x88, 0x81],
    );
    pub const ACPI_10_TABLE: Guid = Guid::new(
        0xeb9d2d30,
        0x2d88,
        0x11d3,
        [0x9a, 0x16, 0x00, 0x90, 0x27, 0x3f, 0xc1, 0x4d],
    );
    pub const SMBIOS3_TABLE: Guid = Guid::new(
        0xf2fd1544,
        0x9794,
        0x4a2c,
        [0x99, 0x2e, 0xe5, 0xbb, 0xcf, 0x20, 0xe3, 0x94],
    );
    pub const GLOBAL_VARIABLE: Guid = Guid::new(
        0x8be4df61,
        0x93ca,
        0x11d2,
        [0xaa, 0x0d, 0x00, 0xe0, 0x98, 0x03, 0x2b, 0x8c],
    );
    pub const IMAGE_SECURITY_DATABASE: Guid = Guid::new(
        0xd719b2cb,
        0x3d3a,
        0x4596,
        [0xa3, 0xbc, 0xda, 0xd0, 0x0e, 0x67, 0x65, 0x6f],
    );
    pub const EXIT_BOOT_SERVICES_EVENT: Guid = Guid::new(
        0x27abf055,
        0xb1b8,
        0x4c26,
        [0x80, 0x48, 0x74, 0x8f, 0x37, 0xba, 0xa2, 0xdf],
    );
    pub const VIRTUAL_ADDRESS_CHANGE_EVENT: Guid = Guid::new(
        0x13fa7698,
        0xc831,
        0x49c7,
        [0x87, 0xea, 0x8f, 0x43, 0xfc, 0xc2, 0x51, 0x96],
    );
    pub const MEMORY_ATTRIBUTES_TABLE: Guid = Guid::new(
        0xdcfa911d,
        0x26eb,
        0x469f,
        [0xa2, 0x20, 0x38, 0xb7, 0xdc, 0x46, 0x12, 0x20],
    );
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TableHeader {
    pub signature: u64,
    pub revision: u32,
    pub header_size: u32,
    pub crc32: u32,
    pub reserved: u32,
}

pub const SYSTEM_TABLE_SIGNATURE: u64 = 0x5453_5953_2049_4249;
pub const BOOT_SERVICES_SIGNATURE: u64 = 0x5652_4553_544f_4f42;
pub const RUNTIME_SERVICES_SIGNATURE: u64 = 0x5652_4553_544e_5552;
/// UEFI 2.9, as (major << 16) | minor with the minor in tenths and hundredths.
pub const SPECIFICATION_REVISION: u32 = (2 << 16) | 90;

#[repr(C)]
pub struct SystemTable {
    pub header: TableHeader,
    pub firmware_vendor: *const u16,
    pub firmware_revision: u32,
    pub console_in_handle: Handle,
    pub con_in: *mut SimpleTextInput,
    pub console_out_handle: Handle,
    pub con_out: *mut SimpleTextOutput,
    pub standard_error_handle: Handle,
    pub std_err: *mut SimpleTextOutput,
    pub runtime_services: *mut RuntimeServices,
    pub boot_services: *mut BootServices,
    pub number_of_table_entries: usize,
    pub configuration_table: *mut ConfigurationTable,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConfigurationTable {
    pub guid: Guid,
    pub table: *mut c_void,
}

pub type NotifyFunction = extern "efiapi" fn(Event, *mut c_void);
pub type Unsupported = usize;

#[repr(C)]
pub struct BootServices {
    pub header: TableHeader,
    pub raise_tpl: extern "efiapi" fn(usize) -> usize,
    pub restore_tpl: extern "efiapi" fn(usize),
    pub allocate_pages: extern "efiapi" fn(u32, u32, usize, *mut u64) -> Status,
    pub free_pages: extern "efiapi" fn(u64, usize) -> Status,
    pub get_memory_map: extern "efiapi" fn(
        *mut usize,
        *mut MemoryDescriptor,
        *mut usize,
        *mut usize,
        *mut u32,
    ) -> Status,
    pub allocate_pool: extern "efiapi" fn(u32, usize, *mut *mut c_void) -> Status,
    pub free_pool: extern "efiapi" fn(*mut c_void) -> Status,
    pub create_event:
        extern "efiapi" fn(u32, usize, Option<NotifyFunction>, *mut c_void, *mut Event) -> Status,
    pub set_timer: extern "efiapi" fn(Event, u32, u64) -> Status,
    pub wait_for_event: extern "efiapi" fn(usize, *mut Event, *mut usize) -> Status,
    pub signal_event: extern "efiapi" fn(Event) -> Status,
    pub close_event: extern "efiapi" fn(Event) -> Status,
    pub check_event: extern "efiapi" fn(Event) -> Status,
    pub install_protocol_interface:
        extern "efiapi" fn(*mut Handle, *const Guid, u32, *mut c_void) -> Status,
    pub reinstall_protocol_interface:
        extern "efiapi" fn(Handle, *const Guid, *mut c_void, *mut c_void) -> Status,
    pub uninstall_protocol_interface:
        extern "efiapi" fn(Handle, *const Guid, *mut c_void) -> Status,
    pub handle_protocol: extern "efiapi" fn(Handle, *const Guid, *mut *mut c_void) -> Status,
    pub reserved: usize,
    pub register_protocol_notify:
        extern "efiapi" fn(*const Guid, Event, *mut *mut c_void) -> Status,
    pub locate_handle:
        extern "efiapi" fn(u32, *const Guid, *mut c_void, *mut usize, *mut Handle) -> Status,
    pub locate_device_path:
        extern "efiapi" fn(*const Guid, *mut *mut DevicePathNode, *mut Handle) -> Status,
    pub install_configuration_table: extern "efiapi" fn(*const Guid, *mut c_void) -> Status,
    pub load_image: extern "efiapi" fn(
        u8,
        Handle,
        *mut DevicePathNode,
        *mut c_void,
        usize,
        *mut Handle,
    ) -> Status,
    pub start_image: extern "efiapi" fn(Handle, *mut usize, *mut *mut u16) -> Status,
    pub exit: extern "efiapi" fn(Handle, Status, usize, *mut u16) -> Status,
    pub unload_image: extern "efiapi" fn(Handle) -> Status,
    pub exit_boot_services: extern "efiapi" fn(Handle, usize) -> Status,
    pub get_next_monotonic_count: extern "efiapi" fn(*mut u64) -> Status,
    pub stall: extern "efiapi" fn(usize) -> Status,
    pub set_watchdog_timer: extern "efiapi" fn(usize, u64, usize, *mut u16) -> Status,
    pub connect_controller:
        extern "efiapi" fn(Handle, *mut Handle, *mut DevicePathNode, u8) -> Status,
    pub disconnect_controller: extern "efiapi" fn(Handle, Handle, Handle) -> Status,
    pub open_protocol:
        extern "efiapi" fn(Handle, *const Guid, *mut *mut c_void, Handle, Handle, u32) -> Status,
    pub close_protocol: extern "efiapi" fn(Handle, *const Guid, Handle, Handle) -> Status,
    pub open_protocol_information: extern "efiapi" fn(
        Handle,
        *const Guid,
        *mut *mut OpenProtocolInformation,
        *mut usize,
    ) -> Status,
    pub protocols_per_handle: extern "efiapi" fn(Handle, *mut *mut *mut Guid, *mut usize) -> Status,
    pub locate_handle_buffer:
        extern "efiapi" fn(u32, *const Guid, *mut c_void, *mut usize, *mut *mut Handle) -> Status,
    pub locate_protocol: extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> Status,
    /// Variadic in the specification: see `protocols::install_multiple`.
    pub install_multiple_protocol_interfaces: usize,
    pub uninstall_multiple_protocol_interfaces: usize,
    pub calculate_crc32: extern "efiapi" fn(*const u8, usize, *mut u32) -> Status,
    pub copy_mem: extern "efiapi" fn(*mut u8, *const u8, usize),
    pub set_mem: extern "efiapi" fn(*mut u8, usize, u8),
    pub create_event_ex: extern "efiapi" fn(
        u32,
        usize,
        Option<NotifyFunction>,
        *const c_void,
        *const Guid,
        *mut Event,
    ) -> Status,
}

#[repr(C)]
pub struct OpenProtocolInformation {
    pub agent_handle: Handle,
    pub controller_handle: Handle,
    pub attributes: u32,
    pub open_count: u32,
}

#[repr(C)]
pub struct RuntimeServices {
    pub header: TableHeader,
    pub get_time: extern "efiapi" fn(*mut Time, *mut TimeCapabilities) -> Status,
    pub set_time: extern "efiapi" fn(*const Time) -> Status,
    pub get_wakeup_time: extern "efiapi" fn(*mut u8, *mut u8, *mut Time) -> Status,
    pub set_wakeup_time: extern "efiapi" fn(u8, *const Time) -> Status,
    pub set_virtual_address_map:
        extern "efiapi" fn(usize, usize, u32, *mut MemoryDescriptor) -> Status,
    pub convert_pointer: extern "efiapi" fn(usize, *mut *mut c_void) -> Status,
    pub get_variable:
        extern "efiapi" fn(*const u16, *const Guid, *mut u32, *mut usize, *mut c_void) -> Status,
    pub get_next_variable_name: extern "efiapi" fn(*mut usize, *mut u16, *mut Guid) -> Status,
    pub set_variable:
        extern "efiapi" fn(*const u16, *const Guid, u32, usize, *const c_void) -> Status,
    pub get_next_high_monotonic_count: extern "efiapi" fn(*mut u32) -> Status,
    pub reset_system: extern "efiapi" fn(u32, Status, usize, *const c_void) -> !,
    pub update_capsule: extern "efiapi" fn(*mut *mut c_void, usize, u64) -> Status,
    pub query_capsule_capabilities:
        extern "efiapi" fn(*mut *mut c_void, usize, *mut u64, *mut u32) -> Status,
    pub query_variable_info: extern "efiapi" fn(u32, *mut u64, *mut u64, *mut u64) -> Status,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Time {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub pad1: u8,
    pub nanosecond: u32,
    pub time_zone: i16,
    pub daylight: u8,
    pub pad2: u8,
}

pub const UNSPECIFIED_TIMEZONE: i16 = 0x7ff;

#[repr(C)]
pub struct TimeCapabilities {
    pub resolution: u32,
    pub accuracy: u32,
    pub sets_to_zero: u8,
}

/// Memory types (`EFI_MEMORY_TYPE`).
pub mod memory_type {
    pub const RESERVED: u32 = 0;
    pub const LOADER_CODE: u32 = 1;
    pub const LOADER_DATA: u32 = 2;
    pub const BOOT_SERVICES_CODE: u32 = 3;
    pub const BOOT_SERVICES_DATA: u32 = 4;
    pub const RUNTIME_SERVICES_CODE: u32 = 5;
    pub const RUNTIME_SERVICES_DATA: u32 = 6;
    pub const CONVENTIONAL: u32 = 7;
    pub const UNUSABLE: u32 = 8;
    pub const ACPI_RECLAIM: u32 = 9;
    pub const ACPI_NVS: u32 = 10;
    pub const MEMORY_MAPPED_IO: u32 = 11;
    pub const MEMORY_MAPPED_IO_PORT_SPACE: u32 = 12;
    pub const PAL_CODE: u32 = 13;
    pub const PERSISTENT: u32 = 14;
    /// One past the last type the specification defines; OS-defined types start at 0x8000_0000.
    pub const MAX: u32 = 15;
}

pub mod memory_attribute {
    pub const UNCACHEABLE: u64 = 1;
    pub const WRITE_BACK: u64 = 8;
    pub const RUNTIME: u64 = 1 << 63;
}

pub const ALLOCATE_ANY_PAGES: u32 = 0;
pub const ALLOCATE_MAX_ADDRESS: u32 = 1;
pub const ALLOCATE_ADDRESS: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemoryDescriptor {
    pub kind: u32,
    pub pad: u32,
    pub physical_start: u64,
    pub virtual_start: u64,
    pub number_of_pages: u64,
    pub attribute: u64,
}

/// What `GetMemoryMap` reports as the descriptor stride. Larger than the structure on purpose:
/// callers must step by it, and EDK2 does the same, so code that assumes `size_of` is caught.
pub const MEMORY_DESCRIPTOR_STRIDE: usize = 48;
pub const MEMORY_DESCRIPTOR_VERSION: u32 = 1;

pub const EVT_TIMER: u32 = 0x8000_0000;
pub const EVT_RUNTIME: u32 = 0x4000_0000;
pub const EVT_NOTIFY_WAIT: u32 = 0x0000_0100;
pub const EVT_NOTIFY_SIGNAL: u32 = 0x0000_0200;
pub const EVT_SIGNAL_EXIT_BOOT_SERVICES: u32 = 0x0000_0201;
pub const EVT_SIGNAL_VIRTUAL_ADDRESS_CHANGE: u32 = 0x6000_0202;

pub const TIMER_CANCEL: u32 = 0;
pub const TIMER_PERIODIC: u32 = 1;
pub const TIMER_RELATIVE: u32 = 2;

pub const TPL_APPLICATION: usize = 4;
pub const TPL_CALLBACK: usize = 8;
pub const TPL_NOTIFY: usize = 16;
pub const TPL_HIGH_LEVEL: usize = 31;

pub const NATIVE_INTERFACE: u32 = 0;

pub const ALL_HANDLES: u32 = 0;
pub const BY_REGISTER_NOTIFY: u32 = 1;
pub const BY_PROTOCOL: u32 = 2;

pub const OPEN_PROTOCOL_BY_HANDLE_PROTOCOL: u32 = 0x01;
pub const OPEN_PROTOCOL_GET_PROTOCOL: u32 = 0x02;
pub const OPEN_PROTOCOL_TEST_PROTOCOL: u32 = 0x04;
pub const OPEN_PROTOCOL_BY_CHILD_CONTROLLER: u32 = 0x08;
pub const OPEN_PROTOCOL_BY_DRIVER: u32 = 0x10;
pub const OPEN_PROTOCOL_EXCLUSIVE: u32 = 0x20;

/// Variable attributes.
pub mod variable_attribute {
    pub const NON_VOLATILE: u32 = 0x01;
    pub const BOOTSERVICE_ACCESS: u32 = 0x02;
    pub const RUNTIME_ACCESS: u32 = 0x04;
    pub const HARDWARE_ERROR_RECORD: u32 = 0x08;
    pub const AUTHENTICATED_WRITE_ACCESS: u32 = 0x10;
    pub const TIME_BASED_AUTHENTICATED_WRITE_ACCESS: u32 = 0x20;
    pub const APPEND_WRITE: u32 = 0x40;
}

pub const RESET_COLD: u32 = 0;
pub const RESET_WARM: u32 = 1;
pub const RESET_SHUTDOWN: u32 = 2;

#[repr(C)]
pub struct DevicePathNode {
    pub kind: u8,
    pub sub_kind: u8,
    pub length: [u8; 2],
}

impl DevicePathNode {
    pub fn node_length(&self) -> usize {
        u16::from_le_bytes(self.length) as usize
    }
}

#[repr(C)]
pub struct SimpleTextInput {
    pub reset: extern "efiapi" fn(*mut SimpleTextInput, u8) -> Status,
    pub read_key_stroke: extern "efiapi" fn(*mut SimpleTextInput, *mut InputKey) -> Status,
    pub wait_for_key: Event,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct InputKey {
    pub scan_code: u16,
    pub unicode_char: u16,
}

#[repr(C)]
pub struct SimpleTextOutput {
    pub reset: extern "efiapi" fn(*mut SimpleTextOutput, u8) -> Status,
    pub output_string: extern "efiapi" fn(*mut SimpleTextOutput, *const u16) -> Status,
    pub test_string: extern "efiapi" fn(*mut SimpleTextOutput, *const u16) -> Status,
    pub query_mode:
        extern "efiapi" fn(*mut SimpleTextOutput, usize, *mut usize, *mut usize) -> Status,
    pub set_mode: extern "efiapi" fn(*mut SimpleTextOutput, usize) -> Status,
    pub set_attribute: extern "efiapi" fn(*mut SimpleTextOutput, usize) -> Status,
    pub clear_screen: extern "efiapi" fn(*mut SimpleTextOutput) -> Status,
    pub set_cursor_position: extern "efiapi" fn(*mut SimpleTextOutput, usize, usize) -> Status,
    pub enable_cursor: extern "efiapi" fn(*mut SimpleTextOutput, u8) -> Status,
    pub mode: *mut SimpleTextOutputMode,
}

#[repr(C)]
pub struct SimpleTextOutputMode {
    pub max_mode: i32,
    pub mode: i32,
    pub attribute: i32,
    pub cursor_column: i32,
    pub cursor_row: i32,
    pub cursor_visible: u8,
}

#[repr(C)]
pub struct LoadedImage {
    pub revision: u32,
    pub parent_handle: Handle,
    pub system_table: *mut SystemTable,
    pub device_handle: Handle,
    pub file_path: *mut DevicePathNode,
    pub reserved: *mut c_void,
    pub load_options_size: u32,
    pub load_options: *mut c_void,
    pub image_base: *mut c_void,
    pub image_size: u64,
    pub image_code_type: u32,
    pub image_data_type: u32,
    pub unload: usize,
}

pub const LOADED_IMAGE_REVISION: u32 = 0x1000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guid_text_round_trips() {
        let text = guids::GLOBAL_VARIABLE.text();
        assert_eq!(&text, b"8be4df61-93ca-11d2-aa0d-00e098032b8c");
        assert_eq!(Guid::parse(&text), Some(guids::GLOBAL_VARIABLE));
        assert_eq!(Guid::parse(b"not a guid"), None);
        assert_eq!(Guid::parse(&[b'z'; 36]), None);
    }

    #[test]
    fn guid_bytes_are_the_in_memory_layout() {
        let bytes = guids::LOADED_IMAGE.to_bytes();
        assert_eq!(&bytes[..4], &0x5b1b31a1u32.to_le_bytes());
        assert_eq!(Guid::from_bytes(&bytes), guids::LOADED_IMAGE);
    }

    #[test]
    fn table_layouts_match_the_specification() {
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<TableHeader>(), 24);
        assert_eq!(offset_of!(SystemTable, con_in), 0x30);
        assert_eq!(offset_of!(SystemTable, runtime_services), 0x58);
        assert_eq!(offset_of!(SystemTable, boot_services), 0x60);
        assert_eq!(offset_of!(SystemTable, configuration_table), 0x70);
        assert_eq!(offset_of!(BootServices, allocate_pages), 0x28);
        assert_eq!(offset_of!(BootServices, handle_protocol), 0x98);
        assert_eq!(offset_of!(BootServices, load_image), 0xc8);
        assert_eq!(offset_of!(BootServices, exit_boot_services), 0xe8);
        assert_eq!(offset_of!(BootServices, open_protocol), 0x118);
        assert_eq!(offset_of!(BootServices, locate_protocol), 0x140);
        assert_eq!(offset_of!(BootServices, calculate_crc32), 0x158);
        assert_eq!(offset_of!(BootServices, create_event_ex), 0x170);
        assert_eq!(offset_of!(RuntimeServices, get_variable), 0x48);
        assert_eq!(offset_of!(RuntimeServices, query_variable_info), 0x80);
        assert_eq!(size_of::<Time>(), 16);
        assert_eq!(size_of::<MemoryDescriptor>(), 40);
        assert_eq!(offset_of!(LoadedImage, image_base), 0x40);
    }
}
