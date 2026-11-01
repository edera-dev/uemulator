//! Assembling the system table from the pieces, and the small boot services that belong to
//! no larger piece.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::devicepath::protocol::{TO_TEXT, UTILITIES};
use crate::handoff::Handoff;
use crate::tables::{self, BOOT_SERVICES, RUNTIME_SERVICES, SYSTEM_TABLE};
use crate::uefi::*;
use crate::{
    console, disks, events, images, kv, memory, pci, protocols, rng, runtime, smbios, time,
    variables,
};

/// UTF-16, NUL-terminated.
static FIRMWARE_VENDOR: [u16; 10] = [
    b'u' as u16,
    b'e' as u16,
    b'm' as u16,
    b'u' as u16,
    b'l' as u16,
    b'a' as u16,
    b't' as u16,
    b'o' as u16,
    b'r' as u16,
    0,
];
const FIRMWARE_REVISION: u32 = 1;

static MONOTONIC_COUNT: AtomicU64 = AtomicU64::new(0);

extern "efiapi" fn stall(microseconds: usize) -> Status {
    time::stall(microseconds as u64);
    SUCCESS
}

/// There is no watchdog to arm; accepting the call is what applications expect.
extern "efiapi" fn set_watchdog_timer(_: usize, _: u64, _: usize, _: *mut u16) -> Status {
    SUCCESS
}

extern "efiapi" fn calculate_crc32(data: *const u8, size: usize, crc: *mut u32) -> Status {
    if data.is_null() || crc.is_null() || size == 0 {
        return INVALID_PARAMETER;
    }
    unsafe { crc.write_unaligned(crate::crc32::crc32(core::slice::from_raw_parts(data, size))) };
    SUCCESS
}

extern "efiapi" fn copy_mem(destination: *mut u8, source: *const u8, length: usize) {
    unsafe { core::ptr::copy(source, destination, length) }
}

extern "efiapi" fn set_mem(buffer: *mut u8, size: usize, value: u8) {
    unsafe { core::ptr::write_bytes(buffer, value, size) }
}

extern "efiapi" fn get_next_monotonic_count(count: *mut u64) -> Status {
    if count.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe { count.write_unaligned(MONOTONIC_COUNT.fetch_add(1, Ordering::Relaxed)) };
    SUCCESS
}

pub fn init(handoff: &Handoff) {
    let boot = unsafe { BOOT_SERVICES.get().assume_init_mut() };
    boot.header.signature = BOOT_SERVICES_SIGNATURE;
    boot.header.revision = SPECIFICATION_REVISION;
    boot.reserved = 0;
    memory::install(boot);
    events::install(boot);
    protocols::install_services(boot);
    images::install(boot);
    boot.install_configuration_table = tables::install_configuration_table;
    boot.stall = stall;
    boot.set_watchdog_timer = set_watchdog_timer;
    boot.calculate_crc32 = calculate_crc32;
    boot.copy_mem = copy_mem;
    boot.set_mem = set_mem;
    boot.get_next_monotonic_count = get_next_monotonic_count;

    let runtime_table = unsafe { RUNTIME_SERVICES.get().assume_init_mut() };
    runtime_table.header.signature = RUNTIME_SERVICES_SIGNATURE;
    runtime_table.header.revision = SPECIFICATION_REVISION;
    runtime::install(runtime_table);

    let system = SYSTEM_TABLE.get();
    system.firmware_vendor = FIRMWARE_VENDOR.as_ptr();
    system.firmware_revision = FIRMWARE_REVISION;
    system.boot_services = boot;
    system.runtime_services = runtime_table;

    kv::init(handoff);
    crate::serial::trace!("kv done");
    variables::init();
    crate::serial::trace!("variables done");
    rng::init(handoff);
    crate::serial::trace!("rng done");
    console::init();
    crate::serial::trace!("console done");
    install_utility_protocols();
    crate::serial::trace!("install_utility_protocols done");
    install_acpi_and_smbios(handoff);
    crate::serial::trace!("install_acpi_and_smbios done");
    pci::init(handoff);
    crate::serial::trace!("pci done");
    disks::init();
    crate::serial::trace!("disks done");
    tables::seal_all();
}

fn install_utility_protocols() {
    let _ = protocols::install(
        core::ptr::null_mut(),
        guids::DEVICE_PATH_TO_TEXT,
        (&raw const TO_TEXT).cast_mut().cast(),
    );
    let _ = protocols::install(
        core::ptr::null_mut(),
        guids::DEVICE_PATH_UTILITIES,
        (&raw const UTILITIES).cast_mut().cast(),
    );
}

fn install_acpi_and_smbios(handoff: &Handoff) {
    if handoff.rsdp != 0 {
        let rsdp = handoff.rsdp as *mut c_void;
        tables::install_configuration(guids::ACPI_20_TABLE, rsdp);
        tables::install_configuration(guids::ACPI_10_TABLE, rsdp);
    }
    smbios::install(handoff);
}
