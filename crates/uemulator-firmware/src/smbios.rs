//! The SMBIOS 3.0 table: BIOS, system, chassis, processor with its three cache levels, the
//! memory array with its one device and mapped ranges, system boot information and the end
//! marker. Xen and Linux read the vendor
//! strings; Linux also counts the memory devices ("DMI: Memory slots populated") and tools like
//! `dmidecode` and `lscpu` read the processor and memory structures. Layouts follow the DMTF
//! SMBIOS 3.0 specification, section 7.

use alloc::vec::Vec;

const ENTRY_POINT_BYTES: usize = 24;
const ANCHOR: &[u8; 5] = b"_SM3_";
const MAJOR: u8 = 3;
const MINOR: u8 = 0;
const ENTRY_POINT_REVISION: u8 = 1;

const TYPE_BIOS: u8 = 0;
const TYPE_SYSTEM: u8 = 1;
const TYPE_CHASSIS: u8 = 3;
const TYPE_PROCESSOR: u8 = 4;
const TYPE_CACHE: u8 = 7;
const TYPE_MEMORY_ARRAY: u8 = 16;
const TYPE_MEMORY_DEVICE: u8 = 17;
const TYPE_MEMORY_ARRAY_MAPPED_ADDRESS: u8 = 19;
const TYPE_SYSTEM_BOOT: u8 = 32;
const TYPE_END: u8 = 127;

/// Formatted-area lengths at the SMBIOS 3.0 revision of each structure.
const BIOS_LENGTH: u8 = 0x18;
const SYSTEM_LENGTH: u8 = 0x1b;
const CHASSIS_LENGTH: u8 = 0x16;
const PROCESSOR_LENGTH: u8 = 0x30;
/// The SMBIOS 3.1 cache structure, with the 32-bit size fields.
const CACHE_LENGTH: u8 = 0x1b;
const MEMORY_ARRAY_LENGTH: u8 = 0x17;
const MEMORY_DEVICE_LENGTH: u8 = 0x28;
const MAPPED_ADDRESS_LENGTH: u8 = 0x1f;
const SYSTEM_BOOT_LENGTH: u8 = 0x0b;
const END_LENGTH: u8 = 4;

/// Handles other structures refer to. Mapped address structures take consecutive handles from
/// `HANDLE_FIRST_MAPPED_ADDRESS`; there are at most as many as the handoff has RAM regions.
const HANDLE_BIOS: u16 = 0x0000;
const HANDLE_SYSTEM: u16 = 0x0100;
const HANDLE_CHASSIS: u16 = 0x0300;
const HANDLE_PROCESSOR: u16 = 0x0400;
/// Cache structures take `HANDLE_FIRST_CACHE + level - 1`.
const HANDLE_FIRST_CACHE: u16 = 0x0700;
const HANDLE_MEMORY_ARRAY: u16 = 0x1000;
const HANDLE_MEMORY_DEVICE: u16 = 0x1100;
const HANDLE_FIRST_MAPPED_ADDRESS: u16 = 0x1300;
const HANDLE_SYSTEM_BOOT: u16 = 0x2000;
const HANDLE_END: u16 = 0x7f00;
/// "No structure": cache handles of a processor without cache structures.
const HANDLE_NONE: u16 = 0xffff;
/// "Not provided": the memory error information handle when no type 18 exists.
const HANDLE_NOT_PROVIDED: u16 = 0xfffe;

/// "BIOS characteristics not supported".
const BIOS_CHARACTERISTICS_UNSUPPORTED: u64 = 1 << 3;
/// BIOS characteristics extension byte 2: UEFI is supported, and this is a virtual machine
/// (what `systemd-detect-virt` and `dmidecode` look for).
const BIOS_EXTENSION_UEFI: u8 = 1 << 3;
const BIOS_EXTENSION_VIRTUAL_MACHINE: u8 = 1 << 4;
/// Embedded controller firmware release 0xff.0xff: there is no embedded controller.
const NO_EMBEDDED_CONTROLLER: u8 = 0xff;
const WAKE_UP_POWER_SWITCH: u8 = 6;

const CHASSIS_TYPE_OTHER: u8 = 0x01;
/// Boot-up, power supply and thermal state "safe"; security status "none".
const CHASSIS_STATE_SAFE: u8 = 0x03;
const CHASSIS_SECURITY_NONE: u8 = 0x03;

const PROCESSOR_TYPE_CENTRAL: u8 = 0x03;
/// Processor family byte 0xfe defers to the Processor Family 2 word; 0x6b is "AMD Zen Processor
/// Family", which covers the family 1Ah (Zen 5) identity the CPUID leaves report.
const PROCESSOR_FAMILY_USE_FAMILY_2: u8 = 0xfe;
const PROCESSOR_FAMILY_ZEN: u16 = 0x6b;
/// Status: socket populated, CPU enabled.
const PROCESSOR_STATUS_POPULATED_ENABLED: u8 = 0x41;
const PROCESSOR_UPGRADE_UNKNOWN: u8 = 0x02;
/// Processor characteristics.
const PROCESSOR_64_BIT: u16 = 1 << 2;
const PROCESSOR_MULTI_CORE: u16 = 1 << 3;
const PROCESSOR_HARDWARE_THREAD: u16 = 1 << 4;
const PROCESSOR_EXECUTE_PROTECTION: u16 = 1 << 5;
const PROCESSOR_ENHANCED_VIRTUALIZATION: u16 = 1 << 6;
/// Power/performance control: the CPPC MSRs and `_CPC`.
const PROCESSOR_POWER_PERFORMANCE_CONTROL: u16 = 1 << 7;
/// The socket designation of a Turin package. The upgrade byte stays "unknown": the socket name
/// carries the information and no SMBIOS enumeration value is assumed for SP5.
const PROCESSOR_SOCKET: &str = "SP5";
/// Core and thread count bytes saturate here; the 3.0 words beside them carry larger counts.
const COUNT_BYTE_OVERFLOW: u16 = 0xff;

/// Levels the processor structure links to (L1, L2, L3).
pub const CACHE_LEVELS: usize = 3;
/// Cache configuration word: level minus one (2:0), internal (6:5 = 0), enabled (7), write-back
/// (9:8 = 01).
const CACHE_ENABLED: u16 = 1 << 7;
const CACHE_WRITE_BACK: u16 = 1 << 8;
/// SRAM type "Unknown" in both the supported and current type words.
const CACHE_SRAM_UNKNOWN: u16 = 1 << 1;
const CACHE_ERROR_CORRECTION_MULTI_BIT_ECC: u8 = 0x06;
/// System cache type "Unified": L1 is reported as one structure, data plus instruction, as AMD
/// firmware does.
const CACHE_TYPE_UNIFIED: u8 = 0x05;
const CACHE_ASSOCIATIVITY_UNKNOWN: u8 = 0x02;
/// Size words: KiB in bits 14:0, or 64 KiB units with bit 15 set; 0xFFFF defers to the 32-bit
/// field, which has the same layout with the granularity in bit 31.
const CACHE_SIZE_64K_GRANULARITY_16: u16 = 1 << 15;
const CACHE_SIZE_64K_GRANULARITY_32: u32 = 1 << 31;
const CACHE_SIZE_FIELD_LIMIT: u64 = 1 << 15;
const CACHE_SIZE_USE_32_BIT: u16 = 0xffff;
const KIB_PER_64K_UNIT: u64 = 64;
/// CPUID 0x8000_001D: type in EAX 4:0 (0 ends the list), level in 7:5, threads sharing minus one
/// in 25:14; ways minus one in EBX 31:22, partitions minus one in 21:12, line size minus one in
/// 11:0; sets minus one in ECX.
const CPUID_CACHE_TYPE_MASK: u32 = 0x1f;
const CPUID_CACHE_LEVEL_SHIFT: u32 = 5;
const CPUID_CACHE_LEVEL_MASK: u32 = 0x7;
const CPUID_CACHE_SHARING_SHIFT: u32 = 14;
const CPUID_CACHE_SHARING_MASK: u32 = 0xfff;
const CPUID_CACHE_WAYS_SHIFT: u32 = 22;
const CPUID_CACHE_PARTITIONS_SHIFT: u32 = 12;
const CPUID_CACHE_FIELD_MASK: u32 = 0x3ff;
const CPUID_CACHE_LINE_MASK: u32 = 0xfff;

const MEMORY_LOCATION_SYSTEM_BOARD: u8 = 0x03;
const MEMORY_USE_SYSTEM: u8 = 0x03;
const MEMORY_ERROR_CORRECTION_NONE: u8 = 0x03;
/// Maximum capacity is a KiB dword; this value says "see the extended (byte) field".
const CAPACITY_USE_EXTENDED: u32 = 0x8000_0000;

const MEMORY_FORM_FACTOR_DIMM: u8 = 0x09;
/// Memory type "RAM": no DDR generation is emulated, so none is claimed.
const MEMORY_TYPE_RAM: u8 = 0x07;
/// Type detail bit 2 is "Unknown" (bit 0 is reserved and bit 1 is "Other").
const MEMORY_TYPE_DETAIL_UNKNOWN: u16 = 1 << 2;
const MEMORY_DATA_WIDTH_BITS: u16 = 64;
const MEMORY_SINGLE_RANK: u8 = 1;
/// Size word: MiB in bits 14:0 (bit 15 clear). 0x7fff says "see the extended size dword", which
/// every size of 32 GiB - 1 MiB and above uses.
const MEMORY_SIZE_USE_EXTENDED: u64 = 0x7fff;

/// Mapped addresses are KiB dwords; 0xffffffff says "see the extended (byte) fields".
const ADDRESS_USE_EXTENDED: u64 = 0xffff_ffff;

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
/// RAM below this is the legacy PC window, where the e820 map reserves the VGA and BIOS areas
/// that a PC's DIMMs still back; ranges separated only by that gap count as one.
const LEGACY_WINDOW_END: u64 = MIB;

/// Upper bound of the table for any machine the handoff can describe (16 RAM regions, a 48-byte
/// brand), which the firmware reserves in its image.
pub const MAX_TABLE_BYTES: usize = 2048;

/// One cache level summed over the socket: every instance of every cache of that level (L1
/// data and instruction together). `bytes == 0` means the level does not exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cache {
    pub bytes: u64,
    /// Ways of the first cache of the level (L1 data for L1).
    pub ways: u32,
}

/// The socket's caches from the subleaves of CPUID 0x8000_001D (ending at the first of type 0)
/// for a package of `threads` logical processors: each cache occurs once per group of threads
/// sharing it.
pub fn socket_caches(subleaves: &[[u32; 4]], threads: u32) -> [Cache; CACHE_LEVELS] {
    let mut caches = [Cache::default(); CACHE_LEVELS];
    for &[eax, ebx, ecx, _] in subleaves {
        if eax & CPUID_CACHE_TYPE_MASK == 0 {
            break;
        }
        let level = (eax >> CPUID_CACHE_LEVEL_SHIFT & CPUID_CACHE_LEVEL_MASK) as usize;
        let Some(cache) = level.checked_sub(1).and_then(|index| caches.get_mut(index)) else {
            continue;
        };
        let sharing = (eax >> CPUID_CACHE_SHARING_SHIFT & CPUID_CACHE_SHARING_MASK) + 1;
        let ways = (ebx >> CPUID_CACHE_WAYS_SHIFT & CPUID_CACHE_FIELD_MASK) + 1;
        let partitions = (ebx >> CPUID_CACHE_PARTITIONS_SHIFT & CPUID_CACHE_FIELD_MASK) + 1;
        let line = (ebx & CPUID_CACHE_LINE_MASK) + 1;
        let sets = ecx as u64 + 1;
        let instances = threads.div_ceil(sharing).max(1) as u64;
        cache.bytes += ways as u64 * partitions as u64 * line as u64 * sets * instances;
        if cache.ways == 0 {
            cache.ways = ways;
        }
    }
    caches
}

/// The processor as CPUID reports it. One socket holds every logical processor.
#[derive(Clone, Copy, Debug)]
pub struct Processor {
    /// CPUID leaf 1 EAX and EDX: the Processor ID field.
    pub signature: u32,
    pub feature_edx: u32,
    /// CPUID leaves 0x8000_0002..4; trailing NULs and spaces are dropped.
    pub brand: [u8; 48],
    pub cores: u16,
    pub threads: u16,
    pub mhz: u16,
    /// L1, L2 and L3 of the whole socket.
    pub caches: [Cache; CACHE_LEVELS],
}

/// What the table describes.
pub struct Machine<'a> {
    /// Usable RAM, as (base, size) ranges from the e820 map.
    pub ram: &'a [(u64, u64)],
    pub processor: Processor,
}

/// One structure: the formatted bytes (header filled in here) and its strings.
fn structure(kind: u8, length: u8, handle: u16, body: &[u8], strings: &[&str]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[kind, length]);
    bytes.extend_from_slice(&handle.to_le_bytes());
    bytes.extend_from_slice(body);
    debug_assert!(bytes.len() <= length as usize, "type {kind} body too long");
    bytes.resize(length as usize, 0);
    if strings.is_empty() {
        bytes.extend_from_slice(&[0, 0]);
    }
    for text in strings {
        bytes.extend_from_slice(text.as_bytes());
        bytes.push(0);
    }
    if !strings.is_empty() {
        bytes.push(0);
    }
    bytes
}

/// The ranges the memory device backs: adjacent RAM ranges merged, and gaps that lie below
/// `LEGACY_WINDOW_END` closed. Sorted by base.
fn memory_ranges(ram: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut sorted: Vec<(u64, u64)> = ram.iter().copied().filter(|&(_, size)| size != 0).collect();
    sorted.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (base, size) in sorted {
        let end = base + size;
        match merged.last_mut() {
            Some((_, last_end)) if base <= *last_end || base <= LEGACY_WINDOW_END => {
                *last_end = (*last_end).max(end);
            }
            _ => merged.push((base, end)),
        }
    }
    merged
}

fn brand_text(brand: &[u8; 48]) -> &str {
    let text = core::str::from_utf8(brand).unwrap_or("");
    text.trim_matches(|c: char| c == '\0' || c == ' ')
}

fn bios() -> Vec<u8> {
    // Vendor, version and release date are strings 1-3; no legacy BIOS segment, no ROM.
    let mut body = Vec::new();
    body.extend_from_slice(&[1, 2]);
    body.extend_from_slice(&0u16.to_le_bytes());
    body.push(3);
    body.push(0);
    body.extend_from_slice(&BIOS_CHARACTERISTICS_UNSUPPORTED.to_le_bytes());
    body.push(0);
    body.push(BIOS_EXTENSION_UEFI | BIOS_EXTENSION_VIRTUAL_MACHINE);
    body.extend_from_slice(&[1, 0, NO_EMBEDDED_CONTROLLER, NO_EMBEDDED_CONTROLLER]);
    structure(
        TYPE_BIOS,
        BIOS_LENGTH,
        HANDLE_BIOS,
        &body,
        &["uemulator", "1", "01/01/2026"],
    )
}

fn system() -> Vec<u8> {
    // Manufacturer, product, version, serial, SKU, family are strings 1-6; the UUID stays zero.
    let mut body = Vec::new();
    body.extend_from_slice(&[1, 2, 3, 4]);
    body.extend_from_slice(&[0; 16]);
    body.extend_from_slice(&[WAKE_UP_POWER_SWITCH, 5, 6]);
    structure(
        TYPE_SYSTEM,
        SYSTEM_LENGTH,
        HANDLE_SYSTEM,
        &body,
        &[
            "uemulator",
            "Virtual Machine",
            "1",
            "0",
            "uemulator",
            "uemulator",
        ],
    )
}

fn chassis() -> Vec<u8> {
    // Manufacturer, version, serial and asset tag are strings 1-4; no contained elements and no
    // SKU string.
    let mut body = Vec::new();
    body.extend_from_slice(&[1, CHASSIS_TYPE_OTHER, 2, 3, 4]);
    body.extend_from_slice(&[
        CHASSIS_STATE_SAFE,
        CHASSIS_STATE_SAFE,
        CHASSIS_STATE_SAFE,
        CHASSIS_SECURITY_NONE,
    ]);
    body.extend_from_slice(&0u32.to_le_bytes()); // OEM-defined
    body.extend_from_slice(&[0, 0, 0, 0, 0]); // height, power cords, element count and length, SKU
    structure(
        TYPE_CHASSIS,
        CHASSIS_LENGTH,
        HANDLE_CHASSIS,
        &body,
        &["uemulator", "1", "0", "0"],
    )
}

fn processor(cpu: &Processor) -> Vec<u8> {
    let mut characteristics = PROCESSOR_64_BIT
        | PROCESSOR_EXECUTE_PROTECTION
        | PROCESSOR_ENHANCED_VIRTUALIZATION
        | PROCESSOR_POWER_PERFORMANCE_CONTROL;
    if cpu.cores > 1 {
        characteristics |= PROCESSOR_MULTI_CORE;
    }
    if cpu.threads > cpu.cores {
        characteristics |= PROCESSOR_HARDWARE_THREAD;
    }
    let count_byte = |count: u16| count.min(COUNT_BYTE_OVERFLOW) as u8;
    // Socket, manufacturer and version are strings 1-3; serial, asset tag and part number none.
    let mut body = Vec::new();
    body.extend_from_slice(&[1, PROCESSOR_TYPE_CENTRAL, PROCESSOR_FAMILY_USE_FAMILY_2, 2]);
    body.extend_from_slice(&cpu.signature.to_le_bytes());
    body.extend_from_slice(&cpu.feature_edx.to_le_bytes());
    body.push(3);
    body.push(0); // voltage: not reported
    body.extend_from_slice(&0u16.to_le_bytes()); // external clock: unknown
    body.extend_from_slice(&cpu.mhz.to_le_bytes()); // maximum speed
    body.extend_from_slice(&cpu.mhz.to_le_bytes()); // current speed
    body.extend_from_slice(&[
        PROCESSOR_STATUS_POPULATED_ENABLED,
        PROCESSOR_UPGRADE_UNKNOWN,
    ]);
    for (index, cache) in cpu.caches.iter().enumerate() {
        let handle = if cache.bytes == 0 {
            HANDLE_NONE
        } else {
            cache_handle(index)
        };
        body.extend_from_slice(&handle.to_le_bytes());
    }
    body.extend_from_slice(&[0, 0, 0]);
    body.extend_from_slice(&[
        count_byte(cpu.cores),
        count_byte(cpu.cores),
        count_byte(cpu.threads),
    ]);
    body.extend_from_slice(&characteristics.to_le_bytes());
    body.extend_from_slice(&PROCESSOR_FAMILY_ZEN.to_le_bytes());
    for count in [cpu.cores, cpu.cores, cpu.threads] {
        body.extend_from_slice(&count.to_le_bytes());
    }
    structure(
        TYPE_PROCESSOR,
        PROCESSOR_LENGTH,
        HANDLE_PROCESSOR,
        &body,
        &[
            PROCESSOR_SOCKET,
            "Advanced Micro Devices, Inc.",
            brand_text(&cpu.brand),
        ],
    )
}

fn cache_handle(index: usize) -> u16 {
    HANDLE_FIRST_CACHE + index as u16
}

/// The 16-bit and 32-bit size fields for `bytes`.
fn cache_size_fields(bytes: u64) -> (u16, u32) {
    let kib = bytes.div_ceil(KIB);
    if kib < CACHE_SIZE_FIELD_LIMIT {
        return (kib as u16, kib as u32);
    }
    let units = kib.div_ceil(KIB_PER_64K_UNIT);
    let wide = CACHE_SIZE_64K_GRANULARITY_32 | units.min(u32::MAX as u64 >> 1) as u32;
    if units < CACHE_SIZE_FIELD_LIMIT {
        (CACHE_SIZE_64K_GRANULARITY_16 | units as u16, wide)
    } else {
        (CACHE_SIZE_USE_32_BIT, wide)
    }
}

/// The associativity byte for a way count; the encoding is not in way order.
fn cache_associativity(ways: u32) -> u8 {
    match ways {
        2 => 0x04,
        4 => 0x05,
        8 => 0x07,
        16 => 0x08,
        12 => 0x09,
        24 => 0x0a,
        32 => 0x0b,
        48 => 0x0c,
        64 => 0x0d,
        20 => 0x0e,
        _ => CACHE_ASSOCIATIVITY_UNKNOWN,
    }
}

/// Type 7 for the cache level `index + 1`.
fn cache(index: usize, cache: &Cache) -> Vec<u8> {
    let (size16, size32) = cache_size_fields(cache.bytes);
    let configuration = index as u16 | CACHE_ENABLED | CACHE_WRITE_BACK;
    let mut body = Vec::new();
    body.push(1); // socket designation: string 1
    body.extend_from_slice(&configuration.to_le_bytes());
    body.extend_from_slice(&size16.to_le_bytes()); // maximum size
    body.extend_from_slice(&size16.to_le_bytes()); // installed size
    body.extend_from_slice(&CACHE_SRAM_UNKNOWN.to_le_bytes()); // supported SRAM type
    body.extend_from_slice(&CACHE_SRAM_UNKNOWN.to_le_bytes()); // current SRAM type
    body.push(0); // speed: unknown
    body.extend_from_slice(&[
        CACHE_ERROR_CORRECTION_MULTI_BIT_ECC,
        CACHE_TYPE_UNIFIED,
        cache_associativity(cache.ways),
    ]);
    body.extend_from_slice(&size32.to_le_bytes()); // maximum size 2
    body.extend_from_slice(&size32.to_le_bytes()); // installed size 2
    let name = ["L1 - Cache", "L2 - Cache", "L3 - Cache"][index];
    structure(
        TYPE_CACHE,
        CACHE_LENGTH,
        cache_handle(index),
        &body,
        &[name],
    )
}

fn memory_array(capacity: u64) -> Vec<u8> {
    let capacity_kib = capacity.div_ceil(KIB);
    let (capacity_field, extended) = match u32::try_from(capacity_kib) {
        Ok(kib) if kib < CAPACITY_USE_EXTENDED => (kib, 0),
        _ => (CAPACITY_USE_EXTENDED, capacity),
    };
    let mut body = Vec::new();
    body.extend_from_slice(&[
        MEMORY_LOCATION_SYSTEM_BOARD,
        MEMORY_USE_SYSTEM,
        MEMORY_ERROR_CORRECTION_NONE,
    ]);
    body.extend_from_slice(&capacity_field.to_le_bytes());
    body.extend_from_slice(&HANDLE_NOT_PROVIDED.to_le_bytes());
    body.extend_from_slice(&1u16.to_le_bytes()); // number of memory devices
    body.extend_from_slice(&extended.to_le_bytes());
    structure(
        TYPE_MEMORY_ARRAY,
        MEMORY_ARRAY_LENGTH,
        HANDLE_MEMORY_ARRAY,
        &body,
        &[],
    )
}

fn memory_device(size: u64) -> Vec<u8> {
    let mib = size / MIB;
    let (size_field, extended) = if mib < MEMORY_SIZE_USE_EXTENDED {
        (mib as u16, 0)
    } else {
        (MEMORY_SIZE_USE_EXTENDED as u16, mib as u32)
    };
    // Device locator and manufacturer are strings 1-2; no bank locator, serial, asset tag or part
    // number. Speeds and voltages are unknown (zero).
    let mut body = Vec::new();
    body.extend_from_slice(&HANDLE_MEMORY_ARRAY.to_le_bytes());
    body.extend_from_slice(&HANDLE_NOT_PROVIDED.to_le_bytes());
    body.extend_from_slice(&MEMORY_DATA_WIDTH_BITS.to_le_bytes()); // total width
    body.extend_from_slice(&MEMORY_DATA_WIDTH_BITS.to_le_bytes()); // data width
    body.extend_from_slice(&size_field.to_le_bytes());
    body.extend_from_slice(&[MEMORY_FORM_FACTOR_DIMM, 0, 1, 0, MEMORY_TYPE_RAM]);
    body.extend_from_slice(&MEMORY_TYPE_DETAIL_UNKNOWN.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes()); // speed
    body.extend_from_slice(&[2, 0, 0, 0, MEMORY_SINGLE_RANK]);
    body.extend_from_slice(&extended.to_le_bytes());
    structure(
        TYPE_MEMORY_DEVICE,
        MEMORY_DEVICE_LENGTH,
        HANDLE_MEMORY_DEVICE,
        &body,
        &["DIMM 0", "uemulator"],
    )
}

fn mapped_address(handle: u16, (base, end): (u64, u64)) -> Vec<u8> {
    // Ending addresses are inclusive. The KiB fields only cover ranges below 4 TiB.
    let last = end - 1;
    let (start_field, end_field, extended_start, extended_end) =
        if last / KIB < ADDRESS_USE_EXTENDED {
            (base / KIB, last / KIB, 0, 0)
        } else {
            (ADDRESS_USE_EXTENDED, ADDRESS_USE_EXTENDED, base, last)
        };
    let mut body = Vec::new();
    body.extend_from_slice(&(start_field as u32).to_le_bytes());
    body.extend_from_slice(&(end_field as u32).to_le_bytes());
    body.extend_from_slice(&HANDLE_MEMORY_ARRAY.to_le_bytes());
    body.push(1); // partition width: the one device
    body.extend_from_slice(&extended_start.to_le_bytes());
    body.extend_from_slice(&extended_end.to_le_bytes());
    structure(
        TYPE_MEMORY_ARRAY_MAPPED_ADDRESS,
        MAPPED_ADDRESS_LENGTH,
        handle,
        &body,
        &[],
    )
}

/// Required by the specification; status 0 is "no errors detected".
fn system_boot() -> Vec<u8> {
    structure(
        TYPE_SYSTEM_BOOT,
        SYSTEM_BOOT_LENGTH,
        HANDLE_SYSTEM_BOOT,
        &[0; 7],
        &[],
    )
}

pub fn build_table(machine: &Machine) -> Vec<u8> {
    let ranges = memory_ranges(machine.ram);
    let installed: u64 = ranges.iter().map(|(base, end)| end - base).sum();
    let mut table = Vec::new();
    table.extend(bios());
    table.extend(system());
    table.extend(chassis());
    table.extend(processor(&machine.processor));
    for (index, level) in machine.processor.caches.iter().enumerate() {
        if level.bytes != 0 {
            table.extend(cache(index, level));
        }
    }
    table.extend(memory_array(installed));
    table.extend(memory_device(installed));
    for (index, range) in ranges.iter().enumerate() {
        table.extend(mapped_address(
            HANDLE_FIRST_MAPPED_ADDRESS + index as u16,
            *range,
        ));
    }
    table.extend(system_boot());
    table.extend(structure(TYPE_END, END_LENGTH, HANDLE_END, &[], &[]));
    table
}

/// The 64-bit entry point for a table of `table_length` bytes at `table_address`.
pub fn entry_point(table_address: u64, table_length: u32) -> [u8; ENTRY_POINT_BYTES] {
    let mut entry = [0u8; ENTRY_POINT_BYTES];
    entry[..5].copy_from_slice(ANCHOR);
    entry[6] = ENTRY_POINT_BYTES as u8;
    entry[7] = MAJOR;
    entry[8] = MINOR;
    entry[10] = ENTRY_POINT_REVISION;
    entry[12..16].copy_from_slice(&table_length.to_le_bytes());
    entry[16..24].copy_from_slice(&table_address.to_le_bytes());
    let sum = entry.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte));
    entry[5] = 0u8.wrapping_sub(sum);
    entry
}

#[cfg(target_os = "none")]
pub fn install(handoff: &crate::handoff::Handoff) {
    use crate::handoff::RAM_USABLE;
    use crate::sync::Global;
    use crate::tables;
    use crate::uefi::guids;

    const SPACE: usize = ENTRY_POINT_BYTES + MAX_TABLE_BYTES;
    /// In the firmware image, which the OS keeps, so the table outlives boot services.
    static STORAGE: Global<[u8; SPACE]> = Global::new([0; SPACE]);
    let ram: Vec<(u64, u64)> = handoff
        .ram_regions
        .iter()
        .filter(|region| region.kind == RAM_USABLE)
        .map(|region| (region.base, region.size))
        .collect();
    let table = build_table(&Machine {
        ram: &ram,
        processor: crate::cpu::smbios_processor(handoff.tsc_hz),
    });
    let storage = STORAGE.get();
    assert!(table.len() + ENTRY_POINT_BYTES <= SPACE);
    storage[ENTRY_POINT_BYTES..ENTRY_POINT_BYTES + table.len()].copy_from_slice(&table);
    let table_address = storage[ENTRY_POINT_BYTES..].as_ptr() as u64;
    storage[..ENTRY_POINT_BYTES].copy_from_slice(&entry_point(table_address, table.len() as u32));
    tables::install_configuration(guids::SMBIOS3_TABLE, storage.as_mut_ptr().cast());
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn brand(text: &str) -> [u8; 48] {
        let mut bytes = [0; 48];
        bytes[..text.len()].copy_from_slice(text.as_bytes());
        bytes
    }

    fn processor_with(cores: u16, threads: u16) -> Processor {
        Processor {
            signature: 0x00b0_0f21,
            feature_edx: 0x178b_fbff,
            brand: brand("AMD EPYC 9135 16-Core Processor                 "),
            cores,
            threads,
            mhz: 1000,
            caches: [
                Cache {
                    bytes: 80 * KIB * cores as u64,
                    ways: 12,
                },
                Cache {
                    bytes: MIB * cores as u64,
                    ways: 16,
                },
                Cache {
                    bytes: 32 * MIB,
                    ways: 16,
                },
            ],
        }
    }

    /// The e820 RAM of a 2 GiB machine: low memory below the VGA window and 1 MiB up.
    const TWO_GIB: [(u64, u64); 2] = [(0, 0x9_fc00), (0x10_0000, 2 * GIB - 0x10_0000)];
    /// 8 GiB with the PCI hole at 3 GiB: 3 GiB below it, 5 GiB above 4 GiB.
    const EIGHT_GIB: [(u64, u64); 3] = [
        (0, 0x9_fc00),
        (0x10_0000, 3 * GIB - 0x10_0000),
        (4 * GIB, 5 * GIB),
    ];

    fn table_for(ram: &[(u64, u64)], cores: u16, threads: u16) -> Vec<u8> {
        build_table(&Machine {
            ram,
            processor: processor_with(cores, threads),
        })
    }

    /// Each structure's formatted area and strings.
    fn structures(table: &[u8]) -> Vec<(&[u8], Vec<&str>)> {
        let mut at = 0;
        let mut found = Vec::new();
        while at < table.len() {
            let length = table[at + 1] as usize;
            let formatted = &table[at..at + length];
            at += length;
            let mut strings = Vec::new();
            if table[at] == 0 {
                at += 2;
            } else {
                while table[at] != 0 {
                    let end = at + table[at..].iter().position(|&byte| byte == 0).unwrap();
                    strings.push(core::str::from_utf8(&table[at..end]).unwrap());
                    at = end + 1;
                }
                at += 1;
            }
            found.push((formatted, strings));
        }
        assert_eq!(at, table.len(), "the last string set runs past the table");
        found
    }

    fn of_type(table: &[u8], kind: u8) -> Vec<(&[u8], Vec<&str>)> {
        structures(table)
            .into_iter()
            .filter(|(formatted, _)| formatted[0] == kind)
            .collect()
    }

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
    }

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn entry_point_checksums_to_zero() {
        let entry = entry_point(0x1234_5000, 200);
        assert_eq!(
            entry.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte)),
            0
        );
        assert_eq!(&entry[..5], b"_SM3_");
        assert_eq!(
            u64::from_le_bytes(entry[16..24].try_into().unwrap()),
            0x1234_5000
        );
    }

    #[test]
    fn the_table_lists_the_required_structures_once_each_and_ends_in_type_127() {
        let table = table_for(&TWO_GIB, 1, 1);
        let kinds: Vec<u8> = structures(&table).iter().map(|(f, _)| f[0]).collect();
        // SMBIOS 3.0 section 6.2 requires types 0, 1, 3, 4, 7, 9, 16, 17, 19 and 32; there are no
        // expansion slots (9) to describe.
        assert_eq!(kinds, [0, 1, 3, 4, 7, 7, 7, 16, 17, 19, 32, 127]);
    }

    #[test]
    fn handles_are_unique() {
        let table = table_for(&EIGHT_GIB, 4, 8);
        let mut handles: Vec<u16> = structures(&table)
            .iter()
            .map(|(f, _)| u16_at(f, 2))
            .collect();
        let count = handles.len();
        handles.sort_unstable();
        handles.dedup();
        assert_eq!(handles.len(), count);
    }

    #[test]
    fn strings_follow_the_formatted_part() {
        let table = table_for(&TWO_GIB, 1, 1);
        assert_eq!(
            &table[BIOS_LENGTH as usize..BIOS_LENGTH as usize + 10],
            b"uemulator\0"
        );
    }

    #[test]
    fn bios_says_uefi_and_virtual_machine() {
        let table = table_for(&TWO_GIB, 1, 1);
        let (bios, _) = &of_type(&table, TYPE_BIOS)[0];
        assert_eq!(bios.len(), 0x18);
        assert_eq!(bios[0x13], (1 << 3) | (1 << 4));
        assert_eq!(&bios[0x16..0x18], [0xff, 0xff]);
    }

    #[test]
    fn chassis_names_the_vendor_and_is_in_a_safe_state() {
        let table = table_for(&TWO_GIB, 1, 1);
        let (chassis, strings) = &of_type(&table, TYPE_CHASSIS)[0];
        assert_eq!(chassis.len(), 0x16);
        assert_eq!(strings[chassis[4] as usize - 1], "uemulator");
        assert_eq!(chassis[5], 1);
        assert_eq!(&chassis[9..12], [3, 3, 3]);
    }

    #[test]
    fn processor_reports_the_cpuid_identity_and_the_core_and_thread_counts() {
        let table = table_for(&TWO_GIB, 4, 8);
        let (cpu, strings) = &of_type(&table, TYPE_PROCESSOR)[0];
        assert_eq!(cpu.len(), 0x30);
        assert_eq!(cpu[5], 3, "central processor");
        assert_eq!((cpu[6], u16_at(cpu, 0x28)), (0xfe, 0x6b), "Zen family");
        assert_eq!(u32_at(cpu, 8), 0x00b0_0f21);
        assert_eq!(u32_at(cpu, 12), 0x178b_fbff);
        assert_eq!(
            strings[cpu[0x10] as usize - 1],
            "AMD EPYC 9135 16-Core Processor"
        );
        assert_eq!((u16_at(cpu, 0x14), u16_at(cpu, 0x16)), (1000, 1000));
        assert_eq!(cpu[0x18], 0x41, "populated and enabled");
        assert_eq!(&cpu[0x23..0x26], [4, 4, 8]);
        assert_eq!(
            (u16_at(cpu, 0x2a), u16_at(cpu, 0x2c), u16_at(cpu, 0x2e)),
            (4, 4, 8)
        );
        let characteristics = u16_at(cpu, 0x26);
        assert_eq!(
            characteristics,
            (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        );
        assert_eq!(strings[cpu[4] as usize - 1], "SP5", "socket");
    }

    #[test]
    fn processor_links_one_cache_structure_per_level() {
        let table = table_for(&TWO_GIB, 16, 32);
        let (cpu, _) = &of_type(&table, TYPE_PROCESSOR)[0];
        let caches = of_type(&table, TYPE_CACHE);
        assert_eq!(caches.len(), 3);
        for (level, (cache, strings)) in caches.iter().enumerate() {
            assert_eq!(cache.len(), 0x1b);
            assert_eq!(
                u16_at(cpu, 0x1a + 2 * level),
                u16_at(cache, 2),
                "L{}",
                level + 1
            );
            assert_eq!(
                u16_at(cache, 5),
                level as u16 | 1 << 7 | 1 << 8,
                "configuration"
            );
            assert_eq!(
                strings[cache[4] as usize - 1],
                format!("L{} - Cache", level + 1)
            );
        }
        let size = |cache: &[u8]| (u16_at(cache, 7), u16_at(cache, 9), u32_at(cache, 0x17));
        // 16 x 80 KiB of L1 and 16 MiB of L2 fit the KiB field; 32 MiB of L3 needs 64 KiB units.
        assert_eq!(size(caches[0].0), (1280, 1280, 1280));
        assert_eq!(size(caches[1].0), (16384, 16384, 16384));
        assert_eq!(
            size(caches[2].0),
            (0x8000 | 512, 0x8000 | 512, 0x8000_0000 | 512)
        );
        assert_eq!(caches[0].0[0x12], 0x09, "12-way");
        assert_eq!(caches[2].0[0x12], 0x08, "16-way");
        assert_eq!(
            u32_at(caches[2].0, 0x13),
            u32_at(caches[2].0, 0x17),
            "max = installed"
        );
    }

    #[test]
    fn caches_too_large_for_the_16_bit_field_use_the_32_bit_one() {
        assert_eq!(cache_size_fields(4 * GIB), (0xffff, 0x8000_0000 | 65536));
    }

    #[test]
    fn socket_caches_count_each_cache_once_per_sharing_group() {
        // Turin's leaves for 32 threads (SMT2): L1d 48 KiB 12-way, L1i 32 KiB 8-way, L2 1 MiB
        // 16-way, all per core; L3 32 MiB 16-way shared by 16 threads.
        let leaf = |kind: u32, level: u32, sharing: u32, ways: u32, bytes: u32| {
            let sets = bytes / (ways * 64);
            [
                kind | level << 5 | 1 << 8 | (sharing - 1) << 14,
                (ways - 1) << 22 | 63,
                sets - 1,
                0,
            ]
        };
        let subleaves = [
            leaf(1, 1, 2, 12, 48 << 10),
            leaf(2, 1, 2, 8, 32 << 10),
            leaf(3, 2, 2, 16, 1 << 20),
            leaf(3, 3, 16, 16, 32 << 20),
            [0; 4],
        ];
        let caches = socket_caches(&subleaves, 32);
        assert_eq!(
            caches,
            [
                Cache {
                    bytes: 16 * 80 * KIB,
                    ways: 12
                },
                Cache {
                    bytes: 16 * MIB,
                    ways: 16
                },
                Cache {
                    bytes: 64 * MIB,
                    ways: 16
                },
            ]
        );
        // fewer threads than share the L3: still one L3
        assert_eq!(socket_caches(&subleaves, 1)[2].bytes, 32 * MIB);
        assert_eq!(socket_caches(&[[0; 4]], 4), [Cache::default(); 3]);
    }

    #[test]
    fn a_single_threaded_single_core_processor_claims_neither_multi_core_nor_threads() {
        let table = table_for(&TWO_GIB, 1, 1);
        let (cpu, _) = &of_type(&table, TYPE_PROCESSOR)[0];
        assert_eq!(u16_at(cpu, 0x26) & ((1 << 3) | (1 << 4)), 0);
        assert_eq!(&cpu[0x23..0x26], [1, 1, 1]);
    }

    #[test]
    fn processor_count_bytes_saturate_and_the_words_carry_the_count() {
        let table = table_for(&TWO_GIB, 300, 600);
        let (cpu, _) = &of_type(&table, TYPE_PROCESSOR)[0];
        assert_eq!(&cpu[0x23..0x26], [0xff, 0xff, 0xff]);
        assert_eq!(u16_at(cpu, 0x2e), 600);
    }

    #[test]
    fn one_dimm_holds_all_the_ram_and_linux_counts_it_as_populated() {
        let table = table_for(&EIGHT_GIB, 1, 1);
        let (array, _) = &of_type(&table, TYPE_MEMORY_ARRAY)[0];
        assert_eq!(array.len(), 0x17);
        assert_eq!(&array[4..7], [3, 3, 3]);
        assert_eq!(
            u32_at(array, 7) as u64,
            8 * GIB / KIB,
            "maximum capacity in KiB"
        );
        assert_eq!(u16_at(array, 0x0d), 1, "one device");
        let devices = of_type(&table, TYPE_MEMORY_DEVICE);
        assert_eq!(devices.len(), 1);
        let (device, strings) = &devices[0];
        assert_eq!(device.len(), 0x28);
        assert_eq!(u16_at(device, 4), HANDLE_MEMORY_ARRAY);
        assert_eq!(u16_at(device, 0x0c), 8192, "size in MiB");
        assert_eq!(device[0x0e], 9, "DIMM");
        assert_eq!(strings[device[0x10] as usize - 1], "DIMM 0");
    }

    #[test]
    fn a_dimm_of_32_gib_or_more_uses_the_extended_size() {
        let table = table_for(&[(0, 0x9_fc00), (0x10_0000, 64 * GIB)], 1, 1);
        let (device, _) = &of_type(&table, TYPE_MEMORY_DEVICE)[0];
        assert_eq!(u16_at(device, 0x0c), 0x7fff);
        assert_eq!(u32_at(device, 0x1c) as u64, (64 * GIB + MIB) / MIB);
    }

    #[test]
    fn an_array_of_2_tib_or_more_uses_the_extended_capacity() {
        let table = table_for(&[(0, 3 * GIB), (4 * GIB, 2048 * GIB)], 1, 1);
        let (array, _) = &of_type(&table, TYPE_MEMORY_ARRAY)[0];
        assert_eq!(u32_at(array, 7), 0x8000_0000);
        assert_eq!(u64_at(array, 0x0f), 2051 * GIB);
    }

    #[test]
    fn mapped_addresses_cover_low_memory_as_one_range_and_the_ram_above_4_gib() {
        let table = table_for(&EIGHT_GIB, 1, 1);
        let mapped = of_type(&table, TYPE_MEMORY_ARRAY_MAPPED_ADDRESS);
        let ranges: Vec<(u32, u32)> = mapped
            .iter()
            .map(|(f, _)| (u32_at(f, 4), u32_at(f, 8)))
            .collect();
        let kib = |bytes: u64| (bytes / KIB) as u32;
        assert_eq!(
            ranges,
            [(0, kib(3 * GIB) - 1), (kib(4 * GIB), kib(9 * GIB) - 1)]
        );
        for (f, _) in &mapped {
            assert_eq!(f.len(), 0x1f);
            assert_eq!(u16_at(f, 0x0c), HANDLE_MEMORY_ARRAY);
            assert_eq!(f[0x0e], 1);
        }
    }

    #[test]
    fn mapped_addresses_past_4_tib_use_the_extended_fields() {
        let table = table_for(&[(0, 3 * GIB), (4 * GIB, 8192 * GIB)], 1, 1);
        let mapped = of_type(&table, TYPE_MEMORY_ARRAY_MAPPED_ADDRESS);
        let (high, _) = &mapped[1];
        assert_eq!((u32_at(high, 4), u32_at(high, 8)), (u32::MAX, u32::MAX));
        assert_eq!(u64_at(high, 0x0f), 4 * GIB);
        assert_eq!(u64_at(high, 0x17), 8196 * GIB - 1);
    }

    #[test]
    fn separate_ram_above_the_legacy_window_stays_separate() {
        let ranges = memory_ranges(&[(4 * GIB, GIB), (0, 0x9_fc00), (0x10_0000, GIB)]);
        assert_eq!(
            ranges,
            [(0, 0x10_0000 + GIB), (4 * GIB, 5 * GIB)],
            "sorted, the legacy gap closed"
        );
        assert_eq!(
            memory_ranges(&[(0, GIB), (GIB, GIB), (3 * GIB, GIB)]),
            [(0, 2 * GIB), (3 * GIB, 4 * GIB)]
        );
    }

    #[test]
    fn system_boot_information_reports_no_errors() {
        let table = table_for(&TWO_GIB, 1, 1);
        let (boot, _) = &of_type(&table, TYPE_SYSTEM_BOOT)[0];
        assert_eq!(boot.len(), 0x0b);
        assert_eq!(boot[0x0a], 0);
    }

    #[test]
    fn the_largest_machine_the_handoff_describes_fits_the_reserved_space() {
        // 16 disjoint RAM regions above the legacy window, 64 processors.
        let ram: Vec<(u64, u64)> = (0..16).map(|i| ((i * 2 + 1) * 4 * GIB, GIB)).collect();
        let table = table_for(&ram, 64, 64);
        assert_eq!(of_type(&table, TYPE_MEMORY_ARRAY_MAPPED_ADDRESS).len(), 16);
        assert!(table.len() <= MAX_TABLE_BYTES, "{} bytes", table.len());
    }
}
