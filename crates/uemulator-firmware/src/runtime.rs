//! Runtime services: what the operating system keeps calling after `ExitBootServices`.
//! Variables live in `variables.rs`.

use core::ffi::c_void;

use crate::cpu::{halt_forever, outb};
use crate::memory;
use crate::sync::Global;
use crate::tables::{self, SYSTEM_TABLE};
use crate::time;
use crate::uefi::*;
use crate::variables;

const RESET_CONTROL_PORT: u16 = 0xcf9;
/// Reset-CPU plus system-reset bits: the library restarts the machine on the write.
const RESET_CONTROL_FULL_RESET: u8 = 0x06;
const MAX_VIRTUAL_REGIONS: usize = 32;
/// The OS calls `SetVirtualAddressMap` once; the map it passes is kept to convert pointers.
const GET_TIME_RESOLUTION: u32 = 1;

#[derive(Clone, Copy)]
struct Conversion {
    physical: u64,
    bytes: u64,
    virtual_start: u64,
}

struct VirtualMap {
    entries: [Conversion; MAX_VIRTUAL_REGIONS],
    count: usize,
    active: bool,
}

static VIRTUAL_MAP: Global<VirtualMap> = Global::new(VirtualMap {
    entries: [Conversion {
        physical: 0,
        bytes: 0,
        virtual_start: 0,
    }; MAX_VIRTUAL_REGIONS],
    count: 0,
    active: false,
});

extern "efiapi" fn get_time(time_out: *mut Time, capabilities: *mut TimeCapabilities) -> Status {
    memory::rewind_runtime_arena();
    if time_out.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe {
        time_out.write_unaligned(time::get_time());
        if !capabilities.is_null() {
            capabilities.write_unaligned(TimeCapabilities {
                resolution: GET_TIME_RESOLUTION,
                accuracy: 0,
                sets_to_zero: 0,
            });
        }
    }
    SUCCESS
}

/// The emulated RTC is derived from the boot timestamp and cannot be set.
extern "efiapi" fn set_time(_: *const Time) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn get_wakeup_time(_: *mut u8, _: *mut u8, _: *mut Time) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn set_wakeup_time(_: u8, _: *const Time) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn get_next_high_monotonic_count(count: *mut u32) -> Status {
    if count.is_null() {
        return INVALID_PARAMETER;
    }
    // One increment per call is enough: nothing persists the count across boots.
    static HIGH: Global<u32> = Global::new(0);
    let high = HIGH.get();
    *high += 1;
    unsafe { count.write_unaligned(*high) };
    SUCCESS
}

fn convert(address: u64) -> Option<u64> {
    let map = VIRTUAL_MAP.get();
    map.entries[..map.count]
        .iter()
        .find(|entry| entry.physical <= address && address < entry.physical + entry.bytes)
        .map(|entry| address - entry.physical + entry.virtual_start)
}

extern "efiapi" fn convert_pointer(_disposition: usize, pointer: *mut *mut c_void) -> Status {
    if pointer.is_null() {
        return INVALID_PARAMETER;
    }
    let address = unsafe { pointer.read_unaligned() } as u64;
    if address == 0 {
        return SUCCESS;
    }
    if !VIRTUAL_MAP.get().active {
        return UNSUPPORTED;
    }
    match convert(address) {
        Some(converted) => {
            unsafe { pointer.write_unaligned(converted as *mut c_void) };
            SUCCESS
        }
        None => NOT_FOUND,
    }
}

extern "efiapi" fn set_virtual_address_map(
    map_size: usize,
    descriptor_size: usize,
    version: u32,
    map: *mut MemoryDescriptor,
) -> Status {
    memory::rewind_runtime_arena();
    if !memory::in_runtime_mode() {
        return UNSUPPORTED;
    }
    if VIRTUAL_MAP.get().active {
        return UNSUPPORTED;
    }
    if map.is_null()
        || descriptor_size < core::mem::size_of::<MemoryDescriptor>()
        || version < MEMORY_DESCRIPTOR_VERSION
    {
        return INVALID_PARAMETER;
    }
    let virtual_map = VIRTUAL_MAP.get();
    for index in 0..map_size / descriptor_size {
        let descriptor = unsafe {
            map.cast::<u8>()
                .add(index * descriptor_size)
                .cast::<MemoryDescriptor>()
                .read_unaligned()
        };
        if descriptor.attribute & memory_attribute::RUNTIME == 0 {
            continue;
        }
        if virtual_map.count == MAX_VIRTUAL_REGIONS {
            return OUT_OF_RESOURCES;
        }
        virtual_map.entries[virtual_map.count] = Conversion {
            physical: descriptor.physical_start,
            bytes: descriptor.number_of_pages * 4096,
            virtual_start: descriptor.virtual_start,
        };
        virtual_map.count += 1;
    }
    virtual_map.active = true;
    // Everything the OS will follow from here on must point at the new addresses: the system
    // table's own pointers and the entries of the runtime services table (the functions are in
    // the firmware's runtime region).
    let system = SYSTEM_TABLE.get();
    let runtime = unsafe { crate::tables::RUNTIME_SERVICES.get().assume_init_mut() };
    for slot in runtime_function_slots(runtime) {
        if let Some(converted) = convert(*slot as u64) {
            *slot = converted as usize;
        }
    }
    if let Some(converted) = convert(system.configuration_table as u64) {
        system.configuration_table = converted as *mut ConfigurationTable;
    }
    if let Some(converted) = convert(system.runtime_services as u64) {
        system.runtime_services = converted as *mut RuntimeServices;
    }
    tables::seal_all();
    SUCCESS
}

/// The function pointers of the runtime services table as plain words, past the header.
fn runtime_function_slots(table: &mut RuntimeServices) -> &mut [usize] {
    let header = core::mem::size_of::<TableHeader>();
    let words = (core::mem::size_of::<RuntimeServices>() - header) / core::mem::size_of::<usize>();
    unsafe {
        core::slice::from_raw_parts_mut(
            (table as *mut RuntimeServices)
                .cast::<u8>()
                .add(header)
                .cast::<usize>(),
            words,
        )
    }
}

extern "efiapi" fn reset_system(
    kind: u32,
    _status: Status,
    _size: usize,
    _data: *const c_void,
) -> ! {
    if kind == RESET_SHUTDOWN {
        // No power controller exists yet: halting with interrupts off is how the library
        // recognises a power-off.
        halt_forever()
    }
    reset_machine()
}

/// A full reset through the reset control register, as `ResetSystem(EfiResetCold)` does.
pub fn reset_machine() -> ! {
    outb(RESET_CONTROL_PORT, RESET_CONTROL_FULL_RESET);
    halt_forever()
}

extern "efiapi" fn update_capsule(_: *mut *mut c_void, _: usize, _: u64) -> Status {
    UNSUPPORTED
}

extern "efiapi" fn query_capsule_capabilities(
    _: *mut *mut c_void,
    _: usize,
    _: *mut u64,
    _: *mut u32,
) -> Status {
    UNSUPPORTED
}

pub fn install(runtime: &mut RuntimeServices) {
    runtime.get_time = get_time;
    runtime.set_time = set_time;
    runtime.get_wakeup_time = get_wakeup_time;
    runtime.set_wakeup_time = set_wakeup_time;
    runtime.set_virtual_address_map = set_virtual_address_map;
    runtime.convert_pointer = convert_pointer;
    runtime.get_next_high_monotonic_count = get_next_high_monotonic_count;
    runtime.reset_system = reset_system;
    runtime.update_capsule = update_capsule;
    runtime.query_capsule_capabilities = query_capsule_capabilities;
    variables::install(runtime);
}
