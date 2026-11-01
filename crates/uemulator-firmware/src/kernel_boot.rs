//! Boots the kernel the library placed in RAM, the way OVMF boots QEMU's `-kernel`: the image is
//! loaded from its buffer, the command line is its load options, and the initramfs is served by
//! the `LoadFile2` protocol the Linux EFI stub looks for under `LINUX_EFI_INITRD_MEDIA_GUID`.

use core::ptr;

use crate::devicepath::DevicePath;
use crate::handoff::Payload;
use crate::tables::BOOT_SERVICES;
use crate::uefi::*;
use crate::{images, media, text};

/// The name the loaded image's device path carries; nothing resolves it.
const KERNEL_FILE_NAME: &str = "kernel";

/// The bytes of a range the library placed. It stays put until the OS reclaims it.
pub fn bytes((base, size): (u64, u64)) -> &'static [u8] {
    unsafe { core::slice::from_raw_parts(base as *const u8, size as usize) }
}

/// UTF-16 load options with a trailing NUL, as an EFI application expects them.
pub fn load_options(text: &str) -> alloc::vec::Vec<u8> {
    text::to_utf16_nul(text)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// Loads an EFI image from a buffer, sets its load options and starts it. `Err` is the status
/// `LoadImage` refused it with; `Ok` the status the image returned with, if it ever returns.
pub fn start_from_buffer(image: &[u8], name: &str, options: &str) -> Result<Status, Status> {
    let path = DevicePath::new().file_path(name);
    let handle = images::load(image, ptr::null_mut(), ptr::null_mut(), Some(&path))?;
    images::set_load_options(handle, &load_options(options));
    let boot = unsafe { BOOT_SERVICES.get().assume_init_ref() };
    Ok((boot.start_image)(handle, ptr::null_mut(), ptr::null_mut()))
}

/// Starts the kernel. Returns only when it could not be loaded or when it returned.
pub fn run(payload: &Payload) -> Status {
    let kernel = bytes(payload.kernel);
    let cmdline = core::str::from_utf8(bytes(payload.cmdline)).unwrap_or("");
    if let Some((base, size)) = payload.initrd
        && let Err(status) = media::publish(guids::LINUX_EFI_INITRD_MEDIA, base, size)
    {
        return status;
    }
    start_from_buffer(kernel, KERNEL_FILE_NAME, cmdline).unwrap_or_else(|status| {
        match status {
            UNSUPPORTED | LOAD_ERROR => crate::serial::println!(
                "the kernel is not an EFI-stub image (status {status:#x}); boot it with --firmware none"
            ),
            _ => crate::serial::println!("could not load the kernel (status {status:#x})"),
        }
        status
    })
}
