//! Loading and running UEFI images, and leaving boot services for the operating system.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::arch::global_asm;
use core::ffi::c_void;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::pe::{self, Image};
use crate::sync::Global;
use crate::tables::{self, SYSTEM_TABLE};
use crate::uefi::*;
use crate::{console, cpu, disks, events, files, memory, protocols};

const PAGE_SIZE: u64 = 4096;

global_asm!(
    // fn call_image(entry, handle, system_table, saved_rsp: *mut usize) -> Status (Microsoft ABI)
    // Saves the callee-saved registers on the stack, records where, and calls the image. `Exit`
    // returns here from any depth by restoring that stack pointer (`exit_image`).
    ".global call_image",
    "call_image:",
    "push rbx",
    "push rbp",
    "push rdi",
    "push rsi",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "mov [r9], rsp",
    // 32 bytes of shadow space plus 8 to keep the stack 16-byte aligned at the call.
    "sub rsp, 40",
    "mov rax, rcx",
    "mov rcx, rdx",
    "mov rdx, r8",
    "call rax",
    "add rsp, 40",
    "call_image_restore:",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rsi",
    "pop rdi",
    "pop rbp",
    "pop rbx",
    "ret",
    // fn exit_image(saved_rsp, status) -> !
    ".global exit_image",
    "exit_image:",
    "mov rsp, rcx",
    "mov rax, rdx",
    "jmp call_image_restore",
);

unsafe extern "win64" {
    fn call_image(
        entry: usize,
        handle: Handle,
        system_table: *mut SystemTable,
        saved_rsp: *mut usize,
    ) -> Status;
    fn exit_image(saved_rsp: usize, status: Status) -> !;
}

struct Record {
    handle: Handle,
    loaded: Box<LoadedImage>,
    base: u64,
    pages: u64,
    entry: usize,
    subsystem: u16,
    started: bool,
    saved_rsp: usize,
    exit_data_size: usize,
    exit_data: *mut u16,
    returned: Option<Status>,
}

// Boxed because `StartImage` hands the saved stack pointer's address to assembly and the
// list may grow while the image runs.
#[allow(clippy::vec_box)]
static IMAGES: Global<Vec<Box<Record>>> = Global::new(Vec::new());

fn record_for(handle: Handle) -> Option<&'static mut Record> {
    IMAGES
        .get()
        .iter_mut()
        .map(|record| &mut **record)
        .find(|record| record.handle == handle)
}

fn leak_path(path: &DevicePath) -> *mut DevicePathNode {
    Box::leak(path.as_bytes().to_vec().into_boxed_slice())
        .as_mut_ptr()
        .cast()
}

/// Reads the file a device path names. The path is a device path to a file system handle followed by
/// file-path nodes; a path of file-path nodes alone ("short form") names a file on whichever
/// volume has it. Returns the bytes, the handle of the volume, and the path.
fn read_file_of(path: &DevicePath) -> Result<(Vec<u8>, Handle), Status> {
    let mut remaining = leak_path(path);
    let mut device: Handle = ptr::null_mut();
    let found = {
        let mut cursor = remaining;
        let status = locate_file_system(&mut cursor, &mut device);
        if status == SUCCESS {
            remaining = cursor;
        }
        status == SUCCESS
    };
    let tail = unsafe { DevicePath::from_raw(remaining.cast()) }.ok_or(INVALID_PARAMETER)?;
    let name = tail.file_name().ok_or(NOT_FOUND)?;
    if found {
        let volume =
            protocols::interface_of(device, &guids::SIMPLE_FILE_SYSTEM).ok_or(NOT_FOUND)?;
        return files::read_file(volume.cast(), &name).map(|bytes| (bytes, device));
    }
    for handle in protocols::handles_with(&guids::SIMPLE_FILE_SYSTEM) {
        let Some(volume) = protocols::interface_of(handle, &guids::SIMPLE_FILE_SYSTEM) else {
            continue;
        };
        if files::exists(volume.cast(), &name) {
            return files::read_file(volume.cast(), &name).map(|bytes| (bytes, handle));
        }
    }
    Err(NOT_FOUND)
}

fn locate_file_system(path: &mut *mut DevicePathNode, device: &mut Handle) -> Status {
    let boot = unsafe { tables::BOOT_SERVICES.get().assume_init_ref() };
    (boot.locate_device_path)(&guids::SIMPLE_FILE_SYSTEM, path, device)
}

/// The handle a loader-supplied path starts with (the partition, for a file path), which is the
/// loaded image's `DeviceHandle` when the caller passed the file's bytes itself, as GRUB's
/// `chainloader` does. xen.efi asks that handle for the file system to find its config and
/// kernel next to itself.
fn device_of(path: &DevicePath) -> Handle {
    let mut bytes = path.as_bytes().to_vec();
    let mut cursor: *mut DevicePathNode = bytes.as_mut_ptr().cast();
    let mut device: Handle = ptr::null_mut();
    let boot = unsafe { tables::BOOT_SERVICES.get().assume_init_ref() };
    match (boot.locate_device_path)(&guids::DEVICE_PATH, &mut cursor, &mut device) {
        SUCCESS => device,
        _ => ptr::null_mut(),
    }
}

/// Loads an image from `file` (the bytes) into memory and publishes its handle.
pub fn load(
    file: &[u8],
    parent: Handle,
    device_handle: Handle,
    path: Option<&DevicePath>,
) -> Result<Handle, Status> {
    let image = Image::parse(file).map_err(|error| match error {
        pe::PeError::Unsupported => UNSUPPORTED,
        _ => LOAD_ERROR,
    })?;
    let code_type = match image.subsystem {
        pe::SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER => memory_type::BOOT_SERVICES_CODE,
        pe::SUBSYSTEM_EFI_RUNTIME_DRIVER => memory_type::RUNTIME_SERVICES_CODE,
        pe::SUBSYSTEM_EFI_APPLICATION => memory_type::LOADER_CODE,
        _ => return Err(UNSUPPORTED),
    };
    let alignment = (image.section_alignment as u64).max(PAGE_SIZE);
    let pages = (image.size_of_image as u64).div_ceil(PAGE_SIZE);
    let base = memory::allocate_aligned(code_type, pages, alignment).ok_or(OUT_OF_RESOURCES)?;
    let memory_slice = unsafe {
        ptr::write_bytes(base as *mut u8, 0, (pages * PAGE_SIZE) as usize);
        core::slice::from_raw_parts_mut(base as *mut u8, (pages * PAGE_SIZE) as usize)
    };
    if image.load(file, memory_slice, base).is_err() {
        memory::free_pages(base, pages);
        return Err(LOAD_ERROR);
    }
    let path_pointer = path.map_or(ptr::null_mut(), |path| leak_path(&path.file_part()));
    let loaded = Box::new(LoadedImage {
        revision: LOADED_IMAGE_REVISION,
        parent_handle: parent,
        system_table: SYSTEM_TABLE.get(),
        device_handle,
        file_path: path_pointer,
        reserved: ptr::null_mut(),
        load_options_size: 0,
        load_options: ptr::null_mut(),
        image_base: base as *mut c_void,
        image_size: pages * PAGE_SIZE,
        image_code_type: code_type,
        image_data_type: memory_type::LOADER_DATA,
        unload: 0,
    });
    let mut record = Box::new(Record {
        handle: ptr::null_mut(),
        loaded,
        base,
        pages,
        entry: base as usize + image.entry_point as usize,
        subsystem: image.subsystem,
        started: false,
        saved_rsp: 0,
        exit_data_size: 0,
        exit_data: ptr::null_mut(),
        returned: None,
    });
    let interface = (&mut *record.loaded as *mut LoadedImage).cast();
    let handle = protocols::install(ptr::null_mut(), guids::LOADED_IMAGE, interface)?;
    if let Some(path) = path {
        let _ = protocols::install(
            handle,
            guids::LOADED_IMAGE_DEVICE_PATH,
            leak_path(path).cast(),
        );
    }
    record.handle = handle;
    IMAGES.get().push(record);
    Ok(handle)
}

/// `LoadOptions` for the next `StartImage`: what the boot manager copies from a load option.
pub fn set_load_options(handle: Handle, options: &[u8]) {
    let Some(record) = record_for(handle) else {
        return;
    };
    let copy = Box::leak(options.to_vec().into_boxed_slice());
    record.loaded.load_options = copy.as_mut_ptr().cast();
    record.loaded.load_options_size = options.len() as u32;
}

extern "efiapi" fn load_image(
    _boot_policy: u8,
    parent: Handle,
    device_path: *mut DevicePathNode,
    source: *mut c_void,
    source_size: usize,
    image: *mut Handle,
) -> Status {
    if image.is_null() || (source.is_null() && device_path.is_null()) {
        return INVALID_PARAMETER;
    }
    let path = unsafe { DevicePath::from_raw(device_path.cast()) };
    let result = if source.is_null() {
        let Some(path) = &path else {
            return INVALID_PARAMETER;
        };
        read_file_of(path).and_then(|(bytes, device)| load(&bytes, parent, device, Some(path)))
    } else {
        let bytes = unsafe { core::slice::from_raw_parts(source.cast::<u8>(), source_size) };
        let device = path.as_ref().map_or(ptr::null_mut(), device_of);
        load(bytes, parent, device, path.as_ref())
    };
    match result {
        Ok(handle) => {
            unsafe { image.write(handle) };
            SUCCESS
        }
        Err(status) => status,
    }
}

fn unload(handle: Handle) -> Status {
    let images = IMAGES.get();
    let Some(position) = images.iter().position(|record| record.handle == handle) else {
        return INVALID_PARAMETER;
    };
    let record = images.remove(position);
    let interface = (&*record.loaded as *const LoadedImage).cast_mut().cast();
    protocols_uninstall(handle, &guids::LOADED_IMAGE, interface);
    memory::free_pages(record.base, record.pages);
    SUCCESS
}

fn protocols_uninstall(handle: Handle, guid: &Guid, interface: *mut c_void) -> Status {
    let boot = unsafe { tables::BOOT_SERVICES.get().assume_init_ref() };
    (boot.uninstall_protocol_interface)(handle, guid, interface)
}

extern "efiapi" fn unload_image(handle: Handle) -> Status {
    match record_for(handle) {
        Some(record) if record.started => UNSUPPORTED,
        Some(_) => unload(handle),
        None => INVALID_PARAMETER,
    }
}

extern "efiapi" fn start_image(
    handle: Handle,
    exit_data_size: *mut usize,
    exit_data: *mut *mut u16,
) -> Status {
    let Some(record) = record_for(handle) else {
        return INVALID_PARAMETER;
    };
    if record.started {
        return INVALID_PARAMETER;
    }
    record.started = true;
    let entry = record.entry;
    let system_table = SYSTEM_TABLE.get();
    let saved_rsp = &mut record.saved_rsp as *mut usize;
    let status = unsafe { call_image(entry, handle, system_table, saved_rsp) };
    // The record may have been moved by the image starting others: look it up again.
    let (size, data) = match record_for(handle) {
        Some(record) => (record.exit_data_size, record.exit_data),
        None => (0, ptr::null_mut()),
    };
    if !exit_data_size.is_null() {
        unsafe { exit_data_size.write(size) };
    }
    if !exit_data.is_null() {
        unsafe { exit_data.write(data) };
    }
    // An application that has returned is finished; a driver stays resident when it succeeded.
    let finished_app =
        record_for(handle).is_some_and(|record| record.subsystem == pe::SUBSYSTEM_EFI_APPLICATION);
    if finished_app || is_error(status) {
        unload(handle);
    }
    status
}

extern "efiapi" fn exit(
    handle: Handle,
    status: Status,
    data_size: usize,
    data: *mut u16,
) -> Status {
    let Some(record) = record_for(handle) else {
        return INVALID_PARAMETER;
    };
    if !record.started {
        return INVALID_PARAMETER;
    }
    record.exit_data_size = data_size;
    record.exit_data = data;
    record.returned = Some(status);
    unsafe { exit_image(record.saved_rsp, status) }
}

extern "efiapi" fn exit_boot_services(_image: Handle, map_key: usize) -> Status {
    if map_key != memory::map_key() {
        return INVALID_PARAMETER;
    }
    events::signal_group(&guids::EXIT_BOOT_SERVICES_EVENT);
    // A notification function may have allocated, which makes the caller's map stale.
    if map_key != memory::map_key() {
        return INVALID_PARAMETER;
    }
    disks::shutdown();
    crate::virtio_input::shutdown();
    console::detach();
    cpu::mask_interrupts();
    SYSTEM_TABLE.get().boot_services = ptr::null_mut();
    memory::enter_runtime_mode();
    tables::seal_all();
    SUCCESS
}

pub fn install(services: &mut BootServices) {
    services.load_image = load_image;
    services.start_image = start_image;
    services.exit = exit;
    services.unload_image = unload_image;
    services.exit_boot_services = exit_boot_services;
}
