//! Boots the hypervisor the library placed in RAM with the supplied kernel as its dom0
//! (`docs/plans/18-xen-dom0.md`). `xen.efi` is started like any EFI application, with its
//! configuration, the kernel and the initramfs behind Xen's media loaders, as Edera's sprout
//! does. Xen's Multiboot2 ELF (`xen.gz`) is loaded the way GRUB's `multiboot2` loads it on UEFI:
//! boot services kept, the image placed where its relocatable tag allows, and entered at its
//! EFI amd64 entry point with the boot information in `rbx`; Xen calls `ExitBootServices`
//! itself.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use core::ptr;

use crate::console::println;
use crate::handoff::{Hypervisor, Payload};
use crate::hypervisor::multiboot2::{self, Information, Module};
use crate::hypervisor::{self, Format, INITRD_IMAGE_NAME, KERNEL_IMAGE_NAME, XEN_IMAGE_NAME};
use crate::kernel_boot::{bytes, load_options, start_from_buffer};
use crate::memory_map::PAGE_SIZE;
use crate::uefi::memory_type::{LOADER_CODE, LOADER_DATA};
use crate::uefi::*;
use crate::{media, memory, protocols, tables};

const EFI_IMAGE_NAME: &str = "xen.efi";
const LOADER_NAME: &str = "uemulator UEFI firmware";
/// Multiboot2 addresses are 32 bits wide: the image, the modules and the boot information all
/// live below 4 GiB.
const MULTIBOOT2_ADDRESS_LIMIT: u64 = 0xffff_ffff;
/// ACPI 2.0's RSDP length; byte 15 is the revision (0 for ACPI 1.0, whose RSDP is 20 bytes).
const RSDP_V2_BYTES: usize = 36;
const RSDP_V1_BYTES: usize = 20;
const RSDP_REVISION_OFFSET: usize = 15;

fn text((base, size): (u64, u64)) -> &'static str {
    core::str::from_utf8(bytes((base, size))).unwrap_or("")
}

/// Starts the hypervisor. Returns only when it could not be started or when it returned.
pub fn run(payload: &Payload, hypervisor: &Hypervisor) -> Status {
    let image = bytes(hypervisor.image);
    let xen_cmdline = text(hypervisor.cmdline);
    let dom0_cmdline = text(payload.cmdline);
    match hypervisor::format(image) {
        Some(Format::Efi) => run_efi(image, payload, xen_cmdline, dom0_cmdline),
        Some(Format::Multiboot2) => match run_multiboot2(image, payload, xen_cmdline, dom0_cmdline)
        {
            Ok(never) => never,
            Err(message) => {
                println!("cannot boot the hypervisor: {message}");
                LOAD_ERROR
            }
        },
        None => {
            println!("the hypervisor is neither an EFI application nor a Multiboot2 ELF");
            LOAD_ERROR
        }
    }
}

fn run_efi(image: &[u8], payload: &Payload, xen_cmdline: &str, dom0_cmdline: &str) -> Status {
    let config = hypervisor::xen_media_config(xen_cmdline, dom0_cmdline);
    let (kernel_base, kernel_size) = payload.kernel;
    let published = media::publish_bytes(guids::XEN_EFI_CONFIG_MEDIA, config.into_bytes())
        .and_then(|()| media::publish(guids::XEN_EFI_KERNEL_MEDIA, kernel_base, kernel_size))
        .and_then(|()| match payload.initrd {
            Some((base, size)) => media::publish(guids::XEN_EFI_RAMDISK_MEDIA, base, size),
            None => Ok(()),
        });
    if let Err(status) = published {
        return status;
    }
    // Xen skips the first word of its load options as the image's own name.
    start_from_buffer(image, EFI_IMAGE_NAME, EFI_IMAGE_NAME).unwrap_or_else(|status| {
        println!("could not load xen.efi (status {status:#x})");
        status
    })
}

/// The RSDP the firmware publishes, as the boot information's ACPI tag carries it.
fn rsdp() -> Option<&'static [u8]> {
    let system = unsafe { &*tables::system_table() };
    let entries = unsafe {
        core::slice::from_raw_parts(system.configuration_table, system.number_of_table_entries)
    };
    let table = [guids::ACPI_20_TABLE, guids::ACPI_10_TABLE]
        .iter()
        .find_map(|guid| entries.iter().find(|entry| entry.guid == *guid))?;
    let pointer = table.table as *const u8;
    let revision = unsafe { pointer.add(RSDP_REVISION_OFFSET).read() };
    let length = if revision >= 2 {
        RSDP_V2_BYTES
    } else {
        RSDP_V1_BYTES
    };
    Some(unsafe { core::slice::from_raw_parts(pointer, length) })
}

/// The firmware's memory map as Multiboot2 reports it, adjacent ranges of the same type merged.
fn memory_map() -> Vec<(u64, u64, u32)> {
    let mut map: Vec<(u64, u64, u32)> = Vec::new();
    for (base, length, efi_type) in memory::snapshot() {
        let kind = multiboot2::memory_type(efi_type);
        match map.last_mut() {
            Some(last) if last.2 == kind && last.0 + last.1 == base => last.1 += length,
            _ => map.push((base, length, kind)),
        }
    }
    map
}

/// A handle with a `LoadedImage` over the loaded hypervisor, which the boot information passes
/// as the image handle (GRUB passes its own): Xen's EFI code calls boot services with it.
fn image_handle(base: u64, pages: u64, cmdline: &str) -> Result<Handle, &'static str> {
    let options = Box::leak(load_options(cmdline).into_boxed_slice());
    let loaded = Box::leak(Box::new(LoadedImage {
        revision: LOADED_IMAGE_REVISION,
        parent_handle: ptr::null_mut(),
        system_table: tables::system_table(),
        device_handle: ptr::null_mut(),
        file_path: ptr::null_mut(),
        reserved: ptr::null_mut(),
        load_options_size: options.len() as u32,
        load_options: options.as_mut_ptr().cast(),
        image_base: base as *mut core::ffi::c_void,
        image_size: pages * PAGE_SIZE,
        image_code_type: LOADER_CODE,
        image_data_type: LOADER_DATA,
        unload: 0,
    }));
    protocols::install(
        ptr::null_mut(),
        guids::LOADED_IMAGE,
        (loaded as *mut LoadedImage).cast(),
    )
    .map_err(|_| "cannot publish the hypervisor's image handle")
}

/// What GRUB's `multiboot2` and `module2` would measure: the files into PCR 9, the command
/// lines into PCR 8.
fn measure_multiboot2(image: &[u8], payload: &Payload, xen_cmdline: &str, dom0_cmdline: &str) {
    use crate::tcg::{PCR_COMMAND_LINES, PCR_LOADED_FILES};
    crate::tpm::measure_loaded(PCR_LOADED_FILES, image, XEN_IMAGE_NAME);
    crate::tpm::measure_loaded(PCR_COMMAND_LINES, xen_cmdline.as_bytes(), xen_cmdline);
    crate::tpm::measure_loaded(PCR_LOADED_FILES, bytes(payload.kernel), KERNEL_IMAGE_NAME);
    crate::tpm::measure_loaded(PCR_COMMAND_LINES, dom0_cmdline.as_bytes(), dom0_cmdline);
    if let Some(initrd) = payload.initrd {
        crate::tpm::measure_loaded(PCR_LOADED_FILES, bytes(initrd), INITRD_IMAGE_NAME);
    }
}

fn run_multiboot2(
    image: &[u8],
    payload: &Payload,
    xen_cmdline: &str,
    dom0_cmdline: &str,
) -> Result<Status, &'static str> {
    // Secure Boot covers PE images only; an ELF has nothing db could vouch for.
    if crate::variables::enforcing() {
        return Err(
            "Secure Boot is on and a Multiboot2 image cannot be verified; \
             boot xen.efi instead, or turn Secure Boot off",
        );
    }
    measure_multiboot2(image, payload, xen_cmdline, dom0_cmdline);
    let header = multiboot2::parse_header(image)?;
    let (Some(efi64_entry), true) = (header.efi64_entry, header.keep_boot_services) else {
        return Err(
            "its Multiboot2 header has no EFI amd64 entry point that keeps boot services; \
             boot it with --firmware none",
        );
    };
    let segments = multiboot2::load_segments(image)?;
    let (link_base, span) = multiboot2::span(&segments).ok_or("its segments overflow")?;
    let pages = span.div_ceil(PAGE_SIZE);
    let load_base = match header.relocatable {
        Some(relocatable) => memory::allocate_highest_in(
            LOADER_CODE,
            pages,
            relocatable.min_address as u64,
            relocatable.max_address as u64,
            (relocatable.alignment as u64).max(PAGE_SIZE),
        ),
        None => memory::allocate_at(LOADER_CODE, link_base, pages).then_some(link_base),
    }
    .ok_or("no free memory where its header lets it be loaded")?;
    let offset = load_base.wrapping_sub(link_base);
    unsafe {
        ptr::write_bytes(load_base as *mut u8, 0, (pages * PAGE_SIZE) as usize);
        for segment in &segments {
            ptr::copy_nonoverlapping(
                image[segment.file_offset..].as_ptr(),
                segment.physical_address.wrapping_add(offset) as *mut u8,
                segment.file_bytes,
            );
        }
    }
    let entry = (efi64_entry as u64).wrapping_add(offset);
    let xen_line = format!("{XEN_IMAGE_NAME} {xen_cmdline}");
    let handle = image_handle(load_base, pages, &xen_line)?;

    // The library placed the kernel and the initramfs on page boundaries below 4 GiB, which is
    // what Xen's module-align tag asks for, so they are handed over where they are.
    let module = |(base, size): (u64, u64), cmdline| Module {
        start: base as u32,
        end: (base + size) as u32,
        cmdline,
    };
    let dom0_line = format!("{KERNEL_IMAGE_NAME} {dom0_cmdline}");
    let mut modules = alloc::vec![module(payload.kernel, dom0_line.as_str())];
    if let Some(initrd) = payload.initrd {
        modules.push(module(initrd, INITRD_IMAGE_NAME));
    }
    if modules
        .iter()
        .any(|module| module.end as u64 > MULTIBOOT2_ADDRESS_LIMIT)
    {
        return Err("the kernel or initramfs lies above 4 GiB");
    }
    let rsdp = rsdp();
    let build = || {
        let map = memory_map();
        multiboot2::boot_information(&Information {
            cmdline: &xen_line,
            loader_name: LOADER_NAME,
            modules: &modules,
            basic_memory: multiboot2::basic_memory(&map),
            memory_map: &map,
            efi_system_table: tables::system_table() as u64,
            efi_image_handle: handle as u64,
            rsdp,
            load_base: load_base as u32,
        })
    };
    // The map describes the allocation that holds it, so it is sized first and built again once
    // that allocation exists; a spare page covers the one or two entries the allocation adds.
    let info_pages = (build().len() as u64).div_ceil(PAGE_SIZE) + 1;
    let info_base = memory::allocate_highest_in(
        LOADER_DATA,
        info_pages,
        0,
        MULTIBOOT2_ADDRESS_LIMIT,
        PAGE_SIZE,
    )
    .ok_or("no memory for the boot information")?;
    let info = build();
    if info.len() as u64 > info_pages * PAGE_SIZE {
        return Err("the boot information outgrew its pages");
    }
    unsafe { ptr::copy_nonoverlapping(info.as_ptr(), info_base as *mut u8, info.len()) };

    println!("Starting the hypervisor at {entry:#x} (loaded at {load_base:#x})");
    // Multiboot2 3.3, EFI amd64 with boot services: 64-bit mode, the magic in eax and the boot
    // information in rbx, the firmware's stack. Xen never returns from here. rbx cannot be an
    // operand (LLVM reserves it), so it is loaded inside the block.
    unsafe {
        core::arch::asm!(
            "mov rbx, {info}",
            "jmp {entry}",
            info = in(reg) info_base,
            entry = in(reg) entry,
            in("eax") multiboot2::BOOTLOADER_MAGIC,
            options(noreturn),
        )
    }
}
