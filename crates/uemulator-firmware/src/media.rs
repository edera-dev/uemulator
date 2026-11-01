//! Media loaders: a byte range served by `EFI_LOAD_FILE2_PROTOCOL` on a handle whose device path
//! is one vendor-media node, which the loaded image finds with `LocateDevicePath`. The Linux EFI
//! stub reads its initramfs this way (`LINUX_EFI_INITRD_MEDIA_GUID`), and Edera's Xen reads its
//! configuration, dom0 kernel and ramdisk (`XEN_EFI_*_MEDIA_GUID`). Several can be published at
//! once: the protocol's `this` says which one is asked.

use alloc::boxed::Box;
use core::ffi::c_void;
use core::ptr;

use crate::devicepath::DevicePath;
use crate::protocols;
use crate::uefi::*;

/// `this` points at `protocol`, the first field, so the callback finds its range from it.
#[repr(C)]
struct MediaLoader {
    protocol: LoadFile2,
    base: u64,
    size: u64,
    /// Handed to the protocol database, which keeps the pointer.
    path: DevicePath,
}

#[repr(C)]
struct LoadFile2 {
    load_file:
        extern "efiapi" fn(*mut LoadFile2, *mut c_void, u8, *mut usize, *mut c_void) -> Status,
}

extern "efiapi" fn load_file(
    this: *mut LoadFile2,
    _file_path: *mut c_void,
    boot_policy: u8,
    size: *mut usize,
    buffer: *mut c_void,
) -> Status {
    if this.is_null() || size.is_null() {
        return INVALID_PARAMETER;
    }
    // LoadFile2 has no boot policy (UEFI 2.9 13.2): a loader asking for one is confused.
    if boot_policy != 0 {
        return UNSUPPORTED;
    }
    let loader = unsafe { &*this.cast::<MediaLoader>() };
    let length = loader.size as usize;
    unsafe {
        let capacity = size.read_unaligned();
        size.write_unaligned(length);
        if buffer.is_null() || capacity < length {
            return BUFFER_TOO_SMALL;
        }
        ptr::copy_nonoverlapping(loader.base as *const u8, buffer.cast::<u8>(), length);
    }
    SUCCESS
}

/// Serves `base..base + size` under `guid`. The memory must stay put for as long as the image
/// that reads it runs; the loader itself is never freed.
pub fn publish(guid: Guid, base: u64, size: u64) -> Result<(), Status> {
    let loader = Box::leak(Box::new(MediaLoader {
        protocol: LoadFile2 { load_file },
        base,
        size,
        path: DevicePath::new().vendor_media(guid),
    }));
    let handle = protocols::install(
        ptr::null_mut(),
        guids::DEVICE_PATH,
        loader.path.as_bytes().as_ptr().cast_mut().cast(),
    )?;
    protocols::install(
        handle,
        guids::LOAD_FILE2,
        (&mut loader.protocol as *mut LoadFile2).cast(),
    )?;
    Ok(())
}

/// `publish` for bytes the firmware made itself (a generated configuration file), which then
/// live for the rest of the boot.
pub fn publish_bytes(guid: Guid, bytes: alloc::vec::Vec<u8>) -> Result<(), Status> {
    let bytes = Box::leak(bytes.into_boxed_slice());
    publish(guid, bytes.as_ptr() as u64, bytes.len() as u64)
}
