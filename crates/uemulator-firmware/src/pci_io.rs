//! `EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL` on a `PciRoot(0)` handle and `EFI_PCI_IO_PROTOCOL` on a
//! handle per bus-0 function (UEFI 2.9 14.2 and 14.4), so a loaded driver can find and drive a
//! device the firmware does not: iPXE (netboot.xyz) reads config space through the root bridge,
//! enables the function and maps DMA buffers through PCI I/O, and drives virtio-net itself. The
//! firmware's own drivers keep talking to the hardware directly (`pci`, `virtio_pci`); a
//! function they drive gets PCI I/O on the handle that already carries its device path, as an
//! EDK2 driver's `BlockIo` sits on the controller's handle.
//!
//! DMA is identity mapped: the virtio functions take 64-bit addresses and there is no IOMMU in
//! the firmware's way, so `Map` never needs a bounce buffer, even for a function whose driver
//! did not ask for dual address cycle.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::handoff::Handoff;
use crate::memory;
use crate::memory_map::PAGE_SIZE;
use crate::pci;
use crate::pci_access::*;
use crate::protocols;
use crate::uefi::*;

type This = *mut c_void;
type RootAccessFn = extern "efiapi" fn(This, u32, u64, usize, *mut c_void) -> Status;
type RootPollFn = extern "efiapi" fn(This, u32, u64, u64, u64, u64, *mut u64) -> Status;
type BarAccessFn = extern "efiapi" fn(This, u32, u8, u64, usize, *mut c_void) -> Status;
type BarPollFn = extern "efiapi" fn(This, u32, u8, u64, u64, u64, u64, *mut u64) -> Status;
type ConfigAccessFn = extern "efiapi" fn(This, u32, u32, usize, *mut c_void) -> Status;
type MapFn =
    extern "efiapi" fn(This, u32, *mut c_void, *mut usize, *mut u64, *mut *mut c_void) -> Status;
type UnmapFn = extern "efiapi" fn(This, *mut c_void) -> Status;
type AllocateBufferFn = extern "efiapi" fn(This, u32, u32, usize, *mut *mut c_void, u64) -> Status;
type FreeBufferFn = extern "efiapi" fn(This, usize, *mut c_void) -> Status;
type FlushFn = extern "efiapi" fn(This) -> Status;

#[repr(C)]
struct Access<F> {
    read: F,
    write: F,
}

#[repr(C)]
struct RootBridgeIo {
    parent_handle: Handle,
    poll_mem: RootPollFn,
    poll_io: RootPollFn,
    mem: Access<RootAccessFn>,
    io: Access<RootAccessFn>,
    pci: Access<RootAccessFn>,
    copy_mem: extern "efiapi" fn(This, u32, u64, u64, usize) -> Status,
    map: MapFn,
    unmap: UnmapFn,
    allocate_buffer: AllocateBufferFn,
    free_buffer: FreeBufferFn,
    flush: FlushFn,
    get_attributes: extern "efiapi" fn(This, *mut u64, *mut u64) -> Status,
    set_attributes: extern "efiapi" fn(This, u64, *mut u64, *mut u64) -> Status,
    configuration: extern "efiapi" fn(This, *mut *mut c_void) -> Status,
    segment_number: u32,
}

#[repr(C)]
struct PciIoProtocol {
    poll_mem: BarPollFn,
    poll_io: BarPollFn,
    mem: Access<BarAccessFn>,
    io: Access<BarAccessFn>,
    pci: Access<ConfigAccessFn>,
    copy_mem: extern "efiapi" fn(This, u32, u8, u64, u8, u64, usize) -> Status,
    map: MapFn,
    unmap: UnmapFn,
    allocate_buffer: AllocateBufferFn,
    free_buffer: FreeBufferFn,
    flush: FlushFn,
    get_location:
        extern "efiapi" fn(This, *mut usize, *mut usize, *mut usize, *mut usize) -> Status,
    attributes: extern "efiapi" fn(This, u32, u64, *mut u64) -> Status,
    get_bar_attributes: extern "efiapi" fn(This, u8, *mut u64, *mut *mut c_void) -> Status,
    set_bar_attributes: extern "efiapi" fn(This, u64, u8, *mut u64, *mut u64) -> Status,
    rom_size: u64,
    rom_image: *mut c_void,
}

/// `this` points at `protocol`, the first field, so a callback finds its function from it.
#[repr(C)]
struct PciIoDevice {
    protocol: PciIoProtocol,
    device: u8,
    function: u8,
    dual_address_cycle: bool,
}

/// The root bridge's `Configuration` answer, built once; callers must not free it.
static RESOURCES: crate::sync::Global<Vec<u8>> = crate::sync::Global::new(Vec::new());

const BAR_COUNT: u8 = 6;
const REGISTER_BAR0: u32 = 0x10;
const BAR_BYTES: u32 = 4;
const BAR_IO_SPACE: u32 = 1 << 0;
const BAR_TYPE_MASK: u32 = 0b110;
const BAR_TYPE_64_BIT: u32 = 0b100;
const BAR_IO_ADDRESS_MASK: u32 = !0b11;
const BAR_MEMORY_ADDRESS_MASK: u32 = !0b1111;
const REGISTER_COMMAND: u32 = 0x04;
const COMMAND_BYTES: usize = 2;

/// `EFI_PCI_IO_PROTOCOL_OPERATION`, the root bridge's included: bus-master read, write and
/// common buffer, then their 64-bit forms.
const MAP_OPERATION_COUNT: u32 = 6;
/// `EFI_ALLOCATE_TYPE`.
const ALLOCATE_ANY_PAGES: u32 = 0;
const ALLOCATE_MAX_ADDRESS: u32 = 1;
/// The attributes `AllocateBuffer` takes (UEFI 2.9 14.4): write combine, cached, dual address
/// cycle.
const ALLOCATE_ATTRIBUTES: u64 = 0x0080 | 0x0800 | ATTRIBUTE_DUAL_ADDRESS_CYCLE;
const BELOW_4_GIB: u64 = 0xffff_ffff;

/// Moves `count` elements between `buffer` and the locations `access` reaches from `start`.
/// `access(location, bytes, Some(value))` writes, `access(location, bytes, None)` reads.
fn transfer(
    width: u32,
    count: usize,
    buffer: *mut c_void,
    write: bool,
    mut access: impl FnMut(usize, usize, Option<u64>) -> Option<u64>,
) -> Status {
    let Some(width) = Width::decode(width) else {
        return INVALID_PARAMETER;
    };
    if buffer.is_null() {
        return INVALID_PARAMETER;
    }
    for index in 0..count {
        let offset = if width.advance_address {
            index * width.bytes
        } else {
            0
        };
        let element = if width.advance_buffer {
            index * width.bytes
        } else {
            0
        };
        let at = unsafe { buffer.cast::<u8>().add(element) };
        let mut bytes = [0u8; 8];
        if write {
            unsafe { ptr::copy_nonoverlapping(at, bytes.as_mut_ptr(), width.bytes) };
            if access(offset, width.bytes, Some(u64::from_le_bytes(bytes))).is_none() {
                return INVALID_PARAMETER;
            }
        } else {
            let Some(value) = access(offset, width.bytes, None) else {
                return INVALID_PARAMETER;
            };
            unsafe { ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), at, width.bytes) };
        }
    }
    SUCCESS
}

fn memory_access(address: u64, bytes: usize, value: Option<u64>) -> Option<u64> {
    unsafe {
        match value {
            None => Some(match bytes {
                1 => (address as *const u8).read_volatile() as u64,
                2 => (address as *const u16).read_volatile() as u64,
                4 => (address as *const u32).read_volatile() as u64,
                _ => (address as *const u64).read_volatile(),
            }),
            Some(value) => {
                match bytes {
                    1 => (address as *mut u8).write_volatile(value as u8),
                    2 => (address as *mut u16).write_volatile(value as u16),
                    4 => (address as *mut u32).write_volatile(value as u32),
                    _ => (address as *mut u64).write_volatile(value),
                }
                Some(0)
            }
        }
    }
}

/// Port I/O has no 64-bit access.
fn io_access(address: u64, bytes: usize, value: Option<u64>) -> Option<u64> {
    let port = u16::try_from(address).ok()?;
    match (bytes, value) {
        (1, None) => Some(crate::cpu::inb(port) as u64),
        (2, None) => Some(crate::cpu::inw(port) as u64),
        (4, None) => Some(crate::cpu::inl(port) as u64),
        (1 | 2 | 4, Some(value)) => {
            match bytes {
                1 => crate::cpu::outb(port, value as u8),
                2 => crate::cpu::outw(port, value as u16),
                _ => crate::cpu::outl(port, value as u32),
            }
            Some(0)
        }
        _ => None,
    }
}

/// Config space of bus 0, the only bus; other buses read as absent functions. A 64-bit access is
/// two dword accesses, low first.
fn config_access(location: ConfigAddress, bytes: usize, value: Option<u64>) -> Option<u64> {
    if !location.register.is_multiple_of(bytes as u32)
        || location.register + bytes as u32 > pci::CONFIG_SPACE_BYTES
    {
        return None;
    }
    if location.bus != 0 {
        return Some(value.map_or(u64::MAX >> (64 - 8 * bytes), |_| 0));
    }
    let (device, function) = (location.device, location.function);
    let dword = |register: u32, value: Option<u32>| match value {
        Some(value) => {
            pci::write(device, function, register, 4, value);
            0
        }
        None => pci::read(device, function, register, 4) as u64,
    };
    match (bytes, value) {
        (8, value) => {
            let low = dword(location.register, value.map(|value| value as u32));
            let high = dword(
                location.register + 4,
                value.map(|value| (value >> 32) as u32),
            );
            Some(low | high << 32)
        }
        (_, Some(value)) => {
            pci::write(device, function, location.register, bytes, value as u32);
            Some(0)
        }
        (_, None) => Some(pci::read(device, function, location.register, bytes) as u64),
    }
}

extern "efiapi" fn root_mem_read(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    transfer(width, count, buffer, false, |offset, bytes, value| {
        memory_access(address + offset as u64, bytes, value)
    })
}

extern "efiapi" fn root_mem_write(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    transfer(width, count, buffer, true, |offset, bytes, value| {
        memory_access(address + offset as u64, bytes, value)
    })
}

extern "efiapi" fn root_io_read(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    transfer(width, count, buffer, false, |offset, bytes, value| {
        io_access(address + offset as u64, bytes, value)
    })
}

extern "efiapi" fn root_io_write(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    transfer(width, count, buffer, true, |offset, bytes, value| {
        io_access(address + offset as u64, bytes, value)
    })
}

extern "efiapi" fn root_pci_read(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    let start = ConfigAddress::decode(address);
    transfer(width, count, buffer, false, |offset, bytes, value| {
        config_access(start.advanced(offset), bytes, value)
    })
}

extern "efiapi" fn root_pci_write(
    _: This,
    width: u32,
    address: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    let start = ConfigAddress::decode(address);
    transfer(width, count, buffer, true, |offset, bytes, value| {
        config_access(start.advanced(offset), bytes, value)
    })
}

extern "efiapi" fn root_poll(
    _: This,
    _: u32,
    _: u64,
    _: u64,
    _: u64,
    _: u64,
    _: *mut u64,
) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn root_copy_mem(_: This, _: u32, _: u64, _: u64, _: usize) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn root_get_attributes(
    _: This,
    supports: *mut u64,
    attributes: *mut u64,
) -> Status {
    if supports.is_null() && attributes.is_null() {
        return INVALID_PARAMETER;
    }
    for out in [supports, attributes] {
        if !out.is_null() {
            unsafe { out.write(0) };
        }
    }
    SUCCESS
}

extern "efiapi" fn root_set_attributes(
    _: This,
    attributes: u64,
    _: *mut u64,
    _: *mut u64,
) -> Status {
    if attributes == 0 {
        SUCCESS
    } else {
        UNSUPPORTED
    }
}

extern "efiapi" fn root_configuration(_: This, resources: *mut *mut c_void) -> Status {
    if resources.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe { resources.write(RESOURCES.get().as_mut_ptr().cast()) };
    SUCCESS
}

extern "efiapi" fn map(
    _: This,
    operation: u32,
    host_address: *mut c_void,
    bytes: *mut usize,
    device_address: *mut u64,
    mapping: *mut *mut c_void,
) -> Status {
    if operation >= MAP_OPERATION_COUNT
        || host_address.is_null()
        || bytes.is_null()
        || device_address.is_null()
        || mapping.is_null()
    {
        return INVALID_PARAMETER;
    }
    unsafe {
        device_address.write(host_address as u64);
        // Any non-null token: `Unmap` has nothing to undo.
        mapping.write(host_address);
    }
    SUCCESS
}

extern "efiapi" fn unmap(_: This, _mapping: *mut c_void) -> Status {
    SUCCESS
}

extern "efiapi" fn allocate_buffer(
    _: This,
    kind: u32,
    memory_type: u32,
    pages: usize,
    host_address: *mut *mut c_void,
    attributes: u64,
) -> Status {
    if host_address.is_null()
        || !matches!(
            memory_type,
            memory_type::BOOT_SERVICES_DATA | memory_type::RUNTIME_SERVICES_DATA
        )
    {
        return INVALID_PARAMETER;
    }
    if attributes & !ALLOCATE_ATTRIBUTES != 0 {
        return UNSUPPORTED;
    }
    let mut limit = if attributes & ATTRIBUTE_DUAL_ADDRESS_CYCLE != 0 {
        u64::MAX
    } else {
        BELOW_4_GIB
    };
    match kind {
        ALLOCATE_ANY_PAGES => {}
        ALLOCATE_MAX_ADDRESS => limit = limit.min(unsafe { host_address.read() } as u64),
        _ => return INVALID_PARAMETER,
    }
    match memory::allocate_highest_in(memory_type, pages as u64, 0, limit, PAGE_SIZE) {
        Some(address) => {
            unsafe { host_address.write(address as *mut c_void) };
            SUCCESS
        }
        None => OUT_OF_RESOURCES,
    }
}

extern "efiapi" fn free_buffer(_: This, pages: usize, host_address: *mut c_void) -> Status {
    if memory::free_pages(host_address as u64, pages as u64) {
        SUCCESS
    } else {
        INVALID_PARAMETER
    }
}

extern "efiapi" fn flush(_: This) -> Status {
    SUCCESS
}

fn device(this: This) -> &'static mut PciIoDevice {
    unsafe { &mut *this.cast::<PciIoDevice>() }
}

/// Where BAR `index` of the function starts, and whether it is in I/O space.
fn bar(this: This, index: u8) -> Option<(u64, bool)> {
    if index >= BAR_COUNT {
        return None;
    }
    let function = device(this);
    let register = REGISTER_BAR0 + index as u32 * BAR_BYTES;
    let low = pci::read(function.device, function.function, register, 4);
    if low & BAR_IO_SPACE != 0 {
        return Some(((low & BAR_IO_ADDRESS_MASK) as u64, true));
    }
    let high = if low & BAR_TYPE_MASK == BAR_TYPE_64_BIT && index + 1 < BAR_COUNT {
        pci::read(function.device, function.function, register + BAR_BYTES, 4)
    } else {
        0
    };
    Some((
        (high as u64) << 32 | (low & BAR_MEMORY_ADDRESS_MASK) as u64,
        false,
    ))
}

/// `Mem` or `Io` (`want_io`) access to a BAR the caller resolved with `bar`, which must be of
/// that space.
fn bar_transfer(
    bar: Option<(u64, bool)>,
    want_io: bool,
    width: u32,
    offset: u64,
    count: usize,
    buffer: *mut c_void,
    write: bool,
) -> Status {
    match bar {
        Some((base, io)) if io == want_io => {
            let access = if io { io_access } else { memory_access };
            transfer(width, count, buffer, write, |at, bytes, value| {
                access(base + offset + at as u64, bytes, value)
            })
        }
        _ => INVALID_PARAMETER,
    }
}

extern "efiapi" fn bar_mem_read(
    this: This,
    width: u32,
    bar: u8,
    offset: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    bar_transfer(
        self::bar(this, bar),
        false,
        width,
        offset,
        count,
        buffer,
        false,
    )
}

extern "efiapi" fn bar_mem_write(
    this: This,
    width: u32,
    bar: u8,
    offset: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    bar_transfer(
        self::bar(this, bar),
        false,
        width,
        offset,
        count,
        buffer,
        true,
    )
}

extern "efiapi" fn bar_io_read(
    this: This,
    width: u32,
    bar: u8,
    offset: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    bar_transfer(
        self::bar(this, bar),
        true,
        width,
        offset,
        count,
        buffer,
        false,
    )
}

extern "efiapi" fn bar_io_write(
    this: This,
    width: u32,
    bar: u8,
    offset: u64,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    bar_transfer(
        self::bar(this, bar),
        true,
        width,
        offset,
        count,
        buffer,
        true,
    )
}

fn own_config(this: This, register: u32) -> ConfigAddress {
    let function = device(this);
    ConfigAddress {
        bus: 0,
        device: function.device,
        function: function.function,
        register,
    }
}

extern "efiapi" fn config_read(
    this: This,
    width: u32,
    register: u32,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    let start = own_config(this, register);
    transfer(width, count, buffer, false, |offset, bytes, value| {
        config_access(start.advanced(offset), bytes, value)
    })
}

extern "efiapi" fn config_write(
    this: This,
    width: u32,
    register: u32,
    count: usize,
    buffer: *mut c_void,
) -> Status {
    let start = own_config(this, register);
    transfer(width, count, buffer, true, |offset, bytes, value| {
        config_access(start.advanced(offset), bytes, value)
    })
}

extern "efiapi" fn bar_poll(
    _: This,
    _: u32,
    _: u8,
    _: u64,
    _: u64,
    _: u64,
    _: u64,
    _: *mut u64,
) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn bar_copy_mem(_: This, _: u32, _: u8, _: u64, _: u8, _: u64, _: usize) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn get_location(
    this: This,
    segment: *mut usize,
    bus: *mut usize,
    device_number: *mut usize,
    function_number: *mut usize,
) -> Status {
    if segment.is_null() || bus.is_null() || device_number.is_null() || function_number.is_null() {
        return INVALID_PARAMETER;
    }
    let function = device(this);
    unsafe {
        segment.write(0);
        bus.write(0);
        device_number.write(function.device as usize);
        function_number.write(function.function as usize);
    }
    SUCCESS
}

extern "efiapi" fn attributes(this: This, operation: u32, given: u64, result: *mut u64) -> Status {
    let function = device(this);
    let (number, function_number) = (function.device, function.function);
    let command = pci::read(number, function_number, REGISTER_COMMAND, COMMAND_BYTES) as u16;
    let mut current = attributes_from_command(command);
    if function.dual_address_cycle {
        current |= ATTRIBUTE_DUAL_ADDRESS_CYCLE;
    }
    match operation {
        ATTRIBUTE_GET | ATTRIBUTE_SUPPORTED => {
            if result.is_null() {
                return INVALID_PARAMETER;
            }
            let answer = if operation == ATTRIBUTE_GET {
                current
            } else {
                SUPPORTED_ATTRIBUTES
            };
            unsafe { result.write(answer) };
            return SUCCESS;
        }
        _ if given & !SUPPORTED_ATTRIBUTES != 0 => return UNSUPPORTED,
        _ => {}
    }
    let Some(wanted) = apply_attributes(operation, current, given) else {
        return INVALID_PARAMETER;
    };
    let command = (command & !COMMAND_OWNED) | command_from_attributes(wanted);
    pci::write(
        number,
        function_number,
        REGISTER_COMMAND,
        COMMAND_BYTES,
        command as u32,
    );
    function.dual_address_cycle = wanted & ATTRIBUTE_DUAL_ADDRESS_CYCLE != 0;
    SUCCESS
}

/// The attributes are the function's; the BAR's own resource descriptor is not offered.
extern "efiapi" fn get_bar_attributes(
    this: This,
    index: u8,
    supports: *mut u64,
    resources: *mut *mut c_void,
) -> Status {
    if (supports.is_null() && resources.is_null()) || bar(this, index).is_none() {
        return INVALID_PARAMETER;
    }
    if !resources.is_null() {
        return UNSUPPORTED;
    }
    unsafe { supports.write(SUPPORTED_ATTRIBUTES) };
    SUCCESS
}

extern "efiapi" fn set_bar_attributes(_: This, _: u64, _: u8, _: *mut u64, _: *mut u64) -> Status {
    UNSUPPORTED
}

fn leak_path(path: &DevicePath) -> *mut c_void {
    Box::leak(path.as_bytes().to_vec().into_boxed_slice())
        .as_mut_ptr()
        .cast()
}

/// The handle whose device path is exactly `path`, if the firmware already made one.
fn handle_with_path(path: &DevicePath) -> Option<Handle> {
    protocols::handles_with(&guids::DEVICE_PATH)
        .into_iter()
        .find(|&handle| {
            protocols::interface_of(handle, &guids::DEVICE_PATH)
                .and_then(|pointer| unsafe { DevicePath::from_raw(pointer.cast()) })
                .is_some_and(|found| found.as_bytes() == path.as_bytes())
        })
}

/// Publishes the root bridge and a PCI I/O instance per function. Runs after the firmware's own
/// drivers made their handles, so a function they drive gets PCI I/O on that handle.
pub fn install(handoff: &Handoff) {
    // The virtio BARs, as the library reports their region.
    let window = handoff
        .mmio_regions
        .first()
        .copied()
        .filter(|&(_, size)| size > 0);
    *RESOURCES.get() = root_bridge_resources(window);
    let root_path = DevicePath::new().acpi_pci_root();
    let Ok(root_handle) =
        protocols::install(ptr::null_mut(), guids::DEVICE_PATH, leak_path(&root_path))
    else {
        return;
    };
    let root = Box::leak(Box::new(RootBridgeIo {
        parent_handle: ptr::null_mut(),
        poll_mem: root_poll,
        poll_io: root_poll,
        mem: Access {
            read: root_mem_read,
            write: root_mem_write,
        },
        io: Access {
            read: root_io_read,
            write: root_io_write,
        },
        pci: Access {
            read: root_pci_read,
            write: root_pci_write,
        },
        copy_mem: root_copy_mem,
        map,
        unmap,
        allocate_buffer,
        free_buffer,
        flush,
        get_attributes: root_get_attributes,
        set_attributes: root_set_attributes,
        configuration: root_configuration,
        segment_number: 0,
    }));
    let _ = protocols::install(
        root_handle,
        guids::PCI_ROOT_BRIDGE_IO,
        (root as *mut RootBridgeIo).cast(),
    );
    for function in pci::scan_all_functions() {
        let path = root_path.clone().pci(function.device, function.function);
        let handle = match handle_with_path(&path) {
            Some(handle) => handle,
            None => match protocols::install(ptr::null_mut(), guids::DEVICE_PATH, leak_path(&path))
            {
                Ok(handle) => handle,
                Err(_) => continue,
            },
        };
        let instance = Box::leak(Box::new(PciIoDevice {
            protocol: PciIoProtocol {
                poll_mem: bar_poll,
                poll_io: bar_poll,
                mem: Access {
                    read: bar_mem_read,
                    write: bar_mem_write,
                },
                io: Access {
                    read: bar_io_read,
                    write: bar_io_write,
                },
                pci: Access {
                    read: config_read,
                    write: config_write,
                },
                copy_mem: bar_copy_mem,
                map,
                unmap,
                allocate_buffer,
                free_buffer,
                flush,
                get_location,
                attributes,
                get_bar_attributes,
                set_bar_attributes,
                rom_size: 0,
                rom_image: ptr::null_mut(),
            },
            device: function.device,
            function: function.function,
            dual_address_cycle: false,
        }));
        let _ = protocols::install(handle, guids::PCI_IO, (instance as *mut PciIoDevice).cast());
    }
}
