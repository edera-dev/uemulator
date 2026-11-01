//! Physical memory: the map the firmware reports, page allocation, `AllocatePool`, and the
//! heap Rust code in the firmware uses (the same pool).
//!
//! After `ExitBootServices` the operating system owns everything the map calls boot services
//! memory, so the heap switches to a small arena inside the firmware image, which stays reserved,
//! and the runtime services entry points rewind it at every call.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::handoff::{Handoff, RAM_USABLE};
use crate::memory_map::{MemoryMap, PAGE_SIZE};
use crate::sync::Global;
use crate::uefi::memory_attribute::{RUNTIME, UNCACHEABLE, WRITE_BACK};
use crate::uefi::memory_type::*;
use crate::uefi::{
    ALLOCATE_ADDRESS, ALLOCATE_ANY_PAGES, ALLOCATE_MAX_ADDRESS, BUFFER_TOO_SMALL,
    INVALID_PARAMETER, MEMORY_DESCRIPTOR_STRIDE, MEMORY_DESCRIPTOR_VERSION, MemoryDescriptor,
    NOT_FOUND, OUT_OF_RESOURCES, SUCCESS, Status,
};

static MAP: Global<MemoryMap> = Global::new(MemoryMap::new());

const POOL_ALIGNMENT: usize = 16;
const HEADER_BYTES: usize = 16;
/// Each pool grows by at least this much at a time.
const ARENA_MIN_BYTES: usize = 256 * 1024;
/// Low nibble clear: the pool kind is OR-ed in there.
const HEADER_MAGIC: usize = 0x5045_4f4c_4d41_4740;
/// Smallest leftover worth keeping as a free block when a block is split.
const MIN_SPLIT_BYTES: usize = 2 * HEADER_BYTES;
const POOL_KINDS: usize = MAX as usize;

/// The legacy structures the library leaves below 1 MiB that must survive until the OS is up:
/// the boot GDT, the handoff page, the page tables.
const LOW_PAGES_FIRST: u64 = 0x6000;
const LOW_PAGES_END: u64 = 0x10000;
const ONE_MIB: u64 = 0x10_0000;

#[repr(C)]
struct FreeBlock {
    size: usize,
    next: *mut FreeBlock,
}

struct Pools {
    free: [*mut FreeBlock; POOL_KINDS],
}

static POOLS: Global<Pools> = Global::new(Pools {
    free: [ptr::null_mut(); POOL_KINDS],
});

static RUNTIME_MODE: AtomicBool = AtomicBool::new(false);
const RUNTIME_ARENA_BYTES: usize = 1 << 20;
#[repr(align(4096))]
struct Arena([u8; RUNTIME_ARENA_BYTES]);
static RUNTIME_ARENA: Global<Arena> = Global::new(Arena([0; RUNTIME_ARENA_BYTES]));
static RUNTIME_USED: AtomicUsize = AtomicUsize::new(0);

/// Heap for the few allocations made before the memory map exists (parsing the handoff page
/// allocates). Bump-only; its pointers are recognised by address and never freed.
const BOOTSTRAP_BYTES: usize = 64 * 1024;
static BOOTSTRAP: Global<Arena64k> = Global::new(Arena64k([0; BOOTSTRAP_BYTES]));
#[repr(align(16))]
struct Arena64k([u8; BOOTSTRAP_BYTES]);
static BOOTSTRAP_USED: AtomicUsize = AtomicUsize::new(0);
static MAP_READY: AtomicBool = AtomicBool::new(false);

fn pages_for(bytes: u64) -> u64 {
    bytes.div_ceil(PAGE_SIZE)
}

pub fn init(handoff: &Handoff) {
    let map = MAP.get();
    for region in &handoff.ram_regions {
        let (kind, attribute) = if region.kind == RAM_USABLE {
            (CONVENTIONAL, WRITE_BACK)
        } else {
            (RESERVED, 0)
        };
        map.set(region.base, region.size / PAGE_SIZE, kind, attribute)
            .expect("memory map has room for the RAM regions");
    }
    let mut mark = |start: u64, bytes: u64, kind: u32, attribute: u64| {
        map.set(start, pages_for(bytes), kind, attribute)
            .expect("memory map has room for the firmware's own regions");
    };
    // Page zero stays unmapped-looking: a null pointer must not look like valid memory.
    mark(0, PAGE_SIZE, RESERVED, 0);
    mark(
        LOW_PAGES_FIRST,
        LOW_PAGES_END - LOW_PAGES_FIRST,
        BOOT_SERVICES_DATA,
        WRITE_BACK,
    );
    // Code and data of the firmware are one runtime region: the OS maps a runtime region as a
    // whole at a new virtual address, so the code must find its data at the same offset.
    mark(
        handoff.firmware_base,
        handoff.firmware_size,
        RUNTIME_SERVICES_CODE,
        WRITE_BACK | RUNTIME,
    );
    if handoff.acpi_size != 0 {
        mark(
            handoff.acpi_base,
            handoff.acpi_size,
            ACPI_RECLAIM,
            WRITE_BACK,
        );
    }
    for &(base, size) in &handoff.mmio_regions {
        mark(base, size, MEMORY_MAPPED_IO, UNCACHEABLE | RUNTIME);
    }
    // The kernel, initramfs and command line the library left in RAM stay allocated until the
    // OS reclaims them as boot services data.
    if let Some(payload) = &handoff.payload {
        let hypervisor = payload.hypervisor.as_ref();
        let ranges = [
            Some(payload.kernel),
            payload.initrd,
            Some(payload.cmdline),
            hypervisor.map(|hypervisor| hypervisor.image),
            hypervisor.map(|hypervisor| hypervisor.cmdline),
        ];
        for (base, size) in ranges.into_iter().flatten().filter(|range| range.1 != 0) {
            mark(base, size, BOOT_SERVICES_DATA, WRITE_BACK);
        }
    }
    debug_assert!(handoff.firmware_base >= ONE_MIB);
    MAP_READY.store(true, Ordering::SeqCst);
}

pub fn map_key() -> usize {
    MAP.get().key()
}

pub fn allocate_any(kind: u32, pages: u64) -> Option<u64> {
    MAP.get()
        .allocate_any(pages, u64::MAX, PAGE_SIZE, kind)
        .ok()
}

pub fn allocate_aligned(kind: u32, pages: u64, alignment: u64) -> Option<u64> {
    MAP.get()
        .allocate_any(pages, u64::MAX, alignment, kind)
        .ok()
}

/// `pages` aligned to `alignment` at the highest address that lies at or above `min_address` and
/// ends at or below `max_address` (inclusive), the way a relocatable Multiboot2 image prefers to
/// be placed.
pub fn allocate_highest_in(
    kind: u32,
    pages: u64,
    min_address: u64,
    max_address: u64,
    alignment: u64,
) -> Option<u64> {
    let map = MAP.get();
    let address = map.allocate_any(pages, max_address, alignment, kind).ok()?;
    if address < min_address {
        map.free(address, pages).ok();
        return None;
    }
    Some(address)
}

/// `pages` at exactly `address`.
pub fn allocate_at(kind: u32, address: u64, pages: u64) -> bool {
    MAP.get().allocate_at(address, pages, kind).is_ok()
}

/// (base, bytes, EFI memory type) of every range in the map, by address.
pub fn snapshot() -> alloc::vec::Vec<(u64, u64, u32)> {
    MAP.get()
        .regions()
        .iter()
        .map(|region| (region.start, region.pages * PAGE_SIZE, region.kind))
        .collect()
}

/// A zeroed, page-aligned block for a device's rings and the like. Freed with `free_pages`.
pub fn allocate_zeroed_pages(kind: u32, pages: u64) -> Option<*mut u8> {
    let address = allocate_any(kind, pages)?;
    unsafe { ptr::write_bytes(address as *mut u8, 0, (pages * PAGE_SIZE) as usize) };
    Some(address as *mut u8)
}

pub fn free_pages(address: u64, pages: u64) -> bool {
    MAP.get().free(address, pages).is_ok()
}

pub fn enter_runtime_mode() {
    RUNTIME_MODE.store(true, Ordering::SeqCst);
    RUNTIME_USED.store(0, Ordering::SeqCst);
}

pub fn in_runtime_mode() -> bool {
    RUNTIME_MODE.load(Ordering::SeqCst)
}

/// Called on entry to every runtime service once boot services are gone: whatever the previous
/// call allocated is dead, because the services are not re-entrant.
pub fn rewind_runtime_arena() {
    if in_runtime_mode() {
        RUNTIME_USED.store(0, Ordering::SeqCst);
    }
}

fn round_up(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

fn grow_pool(kind: u32, at_least: usize) -> bool {
    let bytes = round_up(at_least.max(ARENA_MIN_BYTES), PAGE_SIZE as usize);
    let Some(address) = allocate_any(kind, (bytes as u64) / PAGE_SIZE) else {
        return false;
    };
    insert_free(kind, address as *mut FreeBlock, bytes);
    true
}

/// Puts a block on its pool's address-ordered free list, merging with its neighbours.
fn insert_free(kind: u32, block: *mut FreeBlock, size: usize) {
    unsafe {
        let head = &mut POOLS.get().free[kind as usize];
        let mut previous: *mut FreeBlock = ptr::null_mut();
        let mut next = *head;
        while !next.is_null() && next < block {
            previous = next;
            next = (*next).next;
        }
        (*block).size = size;
        (*block).next = next;
        if !next.is_null() && block.cast::<u8>().add(size) == next.cast() {
            (*block).size += (*next).size;
            (*block).next = (*next).next;
        }
        if previous.is_null() {
            *head = block;
        } else if previous.cast::<u8>().add((*previous).size) == block.cast() {
            (*previous).size += (*block).size;
            (*previous).next = (*block).next;
        } else {
            (*previous).next = block;
        }
    }
}

/// `AllocatePool`: memory of this type, 16-byte aligned, `size` bytes.
pub fn pool_allocate(kind: u32, size: usize) -> *mut u8 {
    if kind as usize >= POOL_KINDS || size > isize::MAX as usize / 2 {
        return ptr::null_mut();
    }
    let needed = round_up(size.max(1), POOL_ALIGNMENT) + HEADER_BYTES;
    for attempt in 0..2 {
        unsafe {
            let head = &mut POOLS.get().free[kind as usize];
            let mut previous: *mut FreeBlock = ptr::null_mut();
            let mut block = *head;
            while !block.is_null() {
                if (*block).size >= needed {
                    let leftover = (*block).size - needed;
                    let taken = if leftover >= MIN_SPLIT_BYTES {
                        let rest = block.cast::<u8>().add(needed).cast::<FreeBlock>();
                        (*rest).size = leftover;
                        (*rest).next = (*block).next;
                        if previous.is_null() {
                            *head = rest
                        } else {
                            (*previous).next = rest
                        }
                        needed
                    } else {
                        if previous.is_null() {
                            *head = (*block).next
                        } else {
                            (*previous).next = (*block).next
                        }
                        (*block).size
                    };
                    let header = block.cast::<usize>();
                    header.write(taken);
                    header.add(1).write(HEADER_MAGIC | kind as usize);
                    return block.cast::<u8>().add(HEADER_BYTES);
                }
                previous = block;
                block = (*block).next;
            }
        }
        if attempt == 0 && !grow_pool(kind, needed) {
            break;
        }
    }
    ptr::null_mut()
}

/// `FreePool`; false for a pointer that is not a live pool allocation.
pub fn pool_free(pointer: *mut u8) -> bool {
    if pointer.is_null() {
        return false;
    }
    unsafe {
        let header = pointer.sub(HEADER_BYTES).cast::<usize>();
        let (size, tag) = (header.read(), header.add(1).read());
        let kind = tag & 0xf;
        if tag & !0xf != HEADER_MAGIC || kind >= POOL_KINDS {
            return false;
        }
        header.add(1).write(0);
        insert_free(kind as u32, header.cast(), size);
    }
    true
}

extern "efiapi" fn allocate_pool(kind: u32, size: usize, buffer: *mut *mut c_void) -> Status {
    if buffer.is_null() || kind as usize >= POOL_KINDS {
        return INVALID_PARAMETER;
    }
    let pointer = pool_allocate(kind, size);
    if pointer.is_null() {
        return OUT_OF_RESOURCES;
    }
    unsafe { buffer.write(pointer.cast()) };
    SUCCESS
}

extern "efiapi" fn free_pool(buffer: *mut c_void) -> Status {
    if pool_free(buffer.cast()) {
        SUCCESS
    } else {
        INVALID_PARAMETER
    }
}

extern "efiapi" fn allocate_pages(mode: u32, kind: u32, pages: usize, address: *mut u64) -> Status {
    if address.is_null() || pages == 0 || (MAX..0x8000_0000).contains(&kind) || kind == CONVENTIONAL
    {
        return INVALID_PARAMETER;
    }
    let map = MAP.get();
    let pages = pages as u64;
    let requested = unsafe { address.read() };
    let result = match mode {
        ALLOCATE_ANY_PAGES => map.allocate_any(pages, u64::MAX, PAGE_SIZE, kind),
        ALLOCATE_MAX_ADDRESS => map.allocate_any(pages, requested, PAGE_SIZE, kind),
        ALLOCATE_ADDRESS if requested % PAGE_SIZE == 0 => {
            map.allocate_at(requested, pages, kind).map(|()| requested)
        }
        _ => return INVALID_PARAMETER,
    };
    match result {
        Ok(start) => {
            unsafe { address.write(start) };
            SUCCESS
        }
        Err(_) => NOT_FOUND,
    }
}

extern "efiapi" fn free_pages_service(address: u64, pages: usize) -> Status {
    if !address.is_multiple_of(PAGE_SIZE) || pages == 0 {
        return INVALID_PARAMETER;
    }
    if free_pages(address, pages as u64) {
        SUCCESS
    } else {
        NOT_FOUND
    }
}

extern "efiapi" fn get_memory_map(
    size: *mut usize,
    descriptors: *mut MemoryDescriptor,
    key: *mut usize,
    descriptor_size: *mut usize,
    descriptor_version: *mut u32,
) -> Status {
    if size.is_null() {
        return INVALID_PARAMETER;
    }
    let map = MAP.get();
    let needed = map.regions().len() * MEMORY_DESCRIPTOR_STRIDE;
    unsafe {
        if !descriptor_size.is_null() {
            descriptor_size.write(MEMORY_DESCRIPTOR_STRIDE);
        }
        if !descriptor_version.is_null() {
            descriptor_version.write(MEMORY_DESCRIPTOR_VERSION);
        }
        let available = size.read();
        size.write(needed);
        if available < needed {
            return BUFFER_TOO_SMALL;
        }
        if descriptors.is_null() || key.is_null() {
            return INVALID_PARAMETER;
        }
        for (index, region) in map.regions().iter().enumerate() {
            let at = descriptors
                .cast::<u8>()
                .add(index * MEMORY_DESCRIPTOR_STRIDE);
            ptr::write_bytes(at, 0, MEMORY_DESCRIPTOR_STRIDE);
            at.cast::<MemoryDescriptor>()
                .write_unaligned(MemoryDescriptor {
                    kind: region.kind,
                    pad: 0,
                    physical_start: region.start,
                    virtual_start: 0,
                    number_of_pages: region.pages,
                    attribute: region.attribute,
                });
        }
        key.write(map.key());
    }
    SUCCESS
}

pub fn install(services: &mut crate::uefi::BootServices) {
    services.allocate_pages = allocate_pages;
    services.free_pages = free_pages_service;
    services.get_memory_map = get_memory_map;
    services.allocate_pool = allocate_pool;
    services.free_pool = free_pool;
}

pub struct Heap;

#[global_allocator]
static HEAP: Heap = Heap;

unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !MAP_READY.load(Ordering::SeqCst) {
            let arena = BOOTSTRAP.get();
            let base = arena.0.as_ptr() as usize;
            let start = round_up(
                base + BOOTSTRAP_USED.load(Ordering::SeqCst),
                layout.align().max(POOL_ALIGNMENT),
            ) - base;
            let end = start + layout.size();
            if end > BOOTSTRAP_BYTES {
                return ptr::null_mut();
            }
            BOOTSTRAP_USED.store(end, Ordering::SeqCst);
            return unsafe { arena.0.as_mut_ptr().add(start) };
        }
        if in_runtime_mode() {
            let arena = RUNTIME_ARENA.get();
            let align = layout.align().max(POOL_ALIGNMENT);
            let start = round_up(
                RUNTIME_USED.load(Ordering::SeqCst) + arena.0.as_ptr() as usize,
                align,
            ) - arena.0.as_ptr() as usize;
            let end = start + layout.size();
            if end > RUNTIME_ARENA_BYTES {
                return ptr::null_mut();
            }
            RUNTIME_USED.store(end, Ordering::SeqCst);
            return unsafe { arena.0.as_mut_ptr().add(start) };
        }
        if layout.align() > POOL_ALIGNMENT {
            return match allocate_zeroed_pages(
                BOOT_SERVICES_DATA,
                pages_for(layout.size().max(1) as u64),
            ) {
                Some(pointer) => pointer,
                None => ptr::null_mut(),
            };
        }
        pool_allocate(BOOT_SERVICES_DATA, layout.size())
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let bootstrap = BOOTSTRAP.get().0.as_ptr_range();
        if in_runtime_mode() || bootstrap.contains(&(pointer as *const u8)) {
            return;
        }
        if layout.align() > POOL_ALIGNMENT {
            free_pages(pointer as u64, pages_for(layout.size().max(1) as u64));
        } else {
            pool_free(pointer);
        }
    }
}
