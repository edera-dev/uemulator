//! The TPM as the firmware uses it: the CRB driver, measured boot (what goes into which PCR,
//! TCG PC Client Platform Firmware Profile 1.05, section 3.3.4), the crypto-agile event log and
//! `EFI_TCG2_PROTOCOL` (TCG EFI Protocol Specification 1.0), through which the Linux EFI stub
//! and GRUB measure what they load and the stub hands the log to the kernel.
//!
//! Nothing on the paths `ExitBootServices` takes allocates: by then the OS owns the memory map,
//! so the log, the final events table and every TPM command live in memory set aside here.

use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use uemulator_crypto::authenticode;
use uemulator_crypto::sha::{self, HashAlgorithm};

use crate::handoff::Handoff;
use crate::memory_map::PAGE_SIZE;
use crate::sync::Global;
use crate::tcg::*;
use crate::uefi::memory_type::{ACPI_NVS, BOOT_SERVICES_DATA};
use crate::uefi::*;
use crate::{memory, protocols, tables, variables};

const LOC_CTRL: u64 = 0x08;
const CTRL_REQ: u64 = 0x40;
const CTRL_START: u64 = 0x4c;
const CTRL_CMD_SIZE: u64 = 0x58;
const CTRL_CMD_LADDR: u64 = 0x5c;
const CTRL_CMD_HADDR: u64 = 0x60;
const CTRL_RSP_SIZE: u64 = 0x64;
const CTRL_RSP_ADDR: u64 = 0x68;
const LOC_CTRL_REQUEST_ACCESS: u32 = 1 << 0;
const CTRL_REQ_CMD_READY: u32 = 1 << 0;
const CTRL_START_INVOKE: u32 = 1 << 0;
/// Polls of `CTRL_START` before a command counts as lost. The library's TPM finishes inside
/// the write that starts it, so one poll is all it ever takes.
const START_POLLS: usize = 1_000_000;
const RESPONSE_HEADER_BYTES: usize = 10;
/// Big enough for any command the firmware builds and any response it reads back.
const COMMAND_BYTES: usize = 512;

const LOG_PAGES: u64 = 16;
const FINAL_EVENTS_PAGES: u64 = 4;
/// `EFI_TCG2_FINAL_EVENTS_TABLE`: Version, NumberOfEvents, then the events.
const FINAL_EVENTS_HEADER_BYTES: usize = 16;
const FINAL_EVENTS_VERSION: u64 = 1;
const MAX_BANKS: usize = 4;
const MAX_DIGEST_BYTES: usize = 64;
/// UTF-16 with its NUL, as EDK2 measures its `PcdFirmwareVersionString`.
const FIRMWARE_VERSION: &str = "uemulator UEFI firmware 1";

const TCG2_PROTOCOL: Guid = Guid::new(
    0x607f766c,
    0x7455,
    0x42be,
    [0x93, 0x0b, 0xe4, 0xd7, 0x6d, 0xb2, 0x72, 0x0f],
);
const TCG2_FINAL_EVENTS_TABLE: Guid = Guid::new(
    0x1e2ed096,
    0x30e2,
    0x4254,
    [0xbd, 0x89, 0x86, 0x3b, 0xbe, 0xf8, 0x23, 0x25],
);

const EVENT_LOG_FORMAT_TCG_2: u32 = 0x2;
/// `EFI_TCG2_EXTEND_ONLY`: extend the PCR, log nothing.
const EXTEND_ONLY: u64 = 0x1;
/// `PE_COFF_IMAGE`: the data is a PE file, measured by its Authenticode digest.
const PE_COFF_IMAGE: u64 = 0x10;
/// `EFI_TCG2_EVENT`'s header: HeaderSize, HeaderVersion, PCRIndex, EventType, after the
/// leading Size.
const EVENT_HEADER_BYTES: u32 = 14;
const EVENT_HEADER_VERSION: u16 = 1;
/// `EFI_TCG2_BOOT_SERVICE_CAPABILITY` is packed: 1.1's is 30 bytes, 1.0's stopped after
/// ManufacturerID at 22.
const CAPABILITY_BYTES: u8 = 30;
const CAPABILITY_1_0_BYTES: u8 = 22;
const PROTOCOL_VERSION: [u8; 2] = [1, 1];
/// The CRB data buffer of the library's TPM, which bounds commands and responses alike.
const TPM_BUFFER_BYTES: u16 = 0xf80;
/// "UEMU", what the library's TPM reports as `TPM_PT_MANUFACTURER`.
const MANUFACTURER_ID: u32 = u32::from_be_bytes(*b"UEMU");

#[repr(C)]
struct Tcg2Protocol {
    get_capability: extern "efiapi" fn(*mut Tcg2Protocol, *mut u8) -> Status,
    get_event_log:
        extern "efiapi" fn(*mut Tcg2Protocol, u32, *mut u64, *mut u64, *mut u8) -> Status,
    hash_log_extend_event:
        extern "efiapi" fn(*mut Tcg2Protocol, u64, u64, u64, *const u8) -> Status,
    submit_command: extern "efiapi" fn(*mut Tcg2Protocol, u32, *const u8, u32, *mut u8) -> Status,
    get_active_pcr_banks: extern "efiapi" fn(*mut Tcg2Protocol, *mut u32) -> Status,
    set_active_pcr_banks: extern "efiapi" fn(*mut Tcg2Protocol, u32) -> Status,
    get_result_of_set_active_pcr_banks:
        extern "efiapi" fn(*mut Tcg2Protocol, *mut u32, *mut u32) -> Status,
}

static PROTOCOL: Global<Tcg2Protocol> = Global::new(Tcg2Protocol {
    get_capability,
    get_event_log,
    hash_log_extend_event,
    submit_command,
    get_active_pcr_banks,
    set_active_pcr_banks,
    get_result_of_set_active_pcr_banks,
});

/// One digest per allocated bank, on the stack.
struct Digests {
    banks: [u16; MAX_BANKS],
    values: [[u8; MAX_DIGEST_BYTES]; MAX_BANKS],
    count: usize,
}

impl Digests {
    fn list(&self) -> [(u16, &[u8]); MAX_BANKS] {
        core::array::from_fn(|index| {
            let length = digest_len(self.banks[index]).unwrap_or(0);
            (self.banks[index], &self.values[index][..length])
        })
    }
}

fn algorithm_of(bank: u16) -> Option<HashAlgorithm> {
    match bank {
        ALG_SHA1 => Some(HashAlgorithm::Sha1),
        ALG_SHA256 => Some(HashAlgorithm::Sha256),
        ALG_SHA384 => Some(HashAlgorithm::Sha384),
        ALG_SHA512 => Some(HashAlgorithm::Sha512),
        _ => None,
    }
}

/// Hashes without the heap (the crate's `Hasher` returns a `Vec`).
fn hash_into(bank: u16, data: &[u8], out: &mut [u8; MAX_DIGEST_BYTES]) {
    match bank {
        ALG_SHA1 => out[..20].copy_from_slice(&sha::sha1(data)),
        ALG_SHA256 => out[..32].copy_from_slice(&sha::sha256(data)),
        ALG_SHA384 => out[..48].copy_from_slice(&sha::sha384(data)),
        ALG_SHA512 => out[..64].copy_from_slice(&sha::sha512(data)),
        _ => {}
    }
}

struct Tpm {
    base: u64,
    banks: [u16; MAX_BANKS],
    bank_count: usize,
    log: *mut u8,
    log_capacity: usize,
    log_used: usize,
    /// Offset of the last event, for `GetEventLog`'s `EventLogLastEntry`.
    last_event: usize,
    truncated: bool,
    final_events: *mut u8,
    final_capacity: usize,
    final_used: usize,
    /// Once `GetEventLog` has been called, later events go to the final events table too
    /// (TCG EFI Protocol 1.0, 6.5): the copy the caller took will not have them.
    log_handed_out: bool,
    boot_attempted: bool,
    exit_measured: bool,
    /// The db entries already logged as `EV_EFI_VARIABLE_AUTHORITY`: each is measured once.
    authorities: Vec<Vec<u8>>,
}

static TPM: Global<Option<Tpm>> = Global::new(None);

fn read32(address: u64) -> u32 {
    unsafe { ptr::read_volatile(address as *const u32) }
}

fn write32(address: u64, value: u32) {
    unsafe { ptr::write_volatile(address as *mut u32, value) }
}

impl Tpm {
    /// Runs one command through the CRB and copies the response into `response`. `None` when
    /// the TPM did not answer or the response does not fit.
    fn submit(&self, command: &[u8], response: &mut [u8]) -> Option<usize> {
        let base = self.base;
        write32(base + CTRL_REQ, CTRL_REQ_CMD_READY);
        let command_address =
            read32(base + CTRL_CMD_LADDR) as u64 | (read32(base + CTRL_CMD_HADDR) as u64) << 32;
        let response_address = unsafe { ptr::read_volatile((base + CTRL_RSP_ADDR) as *const u64) };
        let command_capacity = read32(base + CTRL_CMD_SIZE) as usize;
        let response_capacity = read32(base + CTRL_RSP_SIZE) as usize;
        if command.len() > command_capacity {
            return None;
        }
        for (index, byte) in command.iter().enumerate() {
            unsafe { ptr::write_volatile((command_address + index as u64) as *mut u8, *byte) };
        }
        write32(base + CTRL_START, CTRL_START_INVOKE);
        let mut polls = 0;
        while read32(base + CTRL_START) & CTRL_START_INVOKE != 0 {
            polls += 1;
            if polls == START_POLLS {
                return None;
            }
            core::hint::spin_loop();
        }
        let read_byte = |index: usize| unsafe {
            ptr::read_volatile((response_address + index as u64) as *const u8)
        };
        let mut header = [0u8; RESPONSE_HEADER_BYTES];
        for (index, byte) in header.iter_mut().enumerate() {
            *byte = read_byte(index);
        }
        let length = u32::from_be_bytes(header[2..6].try_into().unwrap()) as usize;
        if !(RESPONSE_HEADER_BYTES..=response_capacity).contains(&length) || length > response.len()
        {
            return None;
        }
        for (index, byte) in response[..length].iter_mut().enumerate() {
            *byte = read_byte(index);
        }
        Some(length)
    }

    fn digests(&self, mut hash: impl FnMut(u16, &mut [u8; MAX_DIGEST_BYTES])) -> Digests {
        let mut digests = Digests {
            banks: self.banks,
            values: [[0; MAX_DIGEST_BYTES]; MAX_BANKS],
            count: self.bank_count,
        };
        for index in 0..self.bank_count {
            hash(self.banks[index], &mut digests.values[index]);
        }
        digests
    }

    fn digests_of(&self, data: &[u8]) -> Digests {
        self.digests(|bank, out| hash_into(bank, data, out))
    }

    fn extend(&self, pcr: u32, digests: &Digests) -> bool {
        let mut command = [0u8; COMMAND_BYTES];
        let list = digests.list();
        let Some(command) = pcr_extend_command(&mut command, pcr, &list[..digests.count]) else {
            return false;
        };
        let mut response = [0u8; COMMAND_BYTES];
        self.submit(command, &mut response)
            .and_then(|length| response_code(&response[..length]))
            == Some(RC_SUCCESS)
    }

    fn append(&mut self, pcr: u32, event_type: u32, digests: &Digests, data: &[u8]) {
        let list = digests.list();
        let list = &list[..digests.count];
        let log = unsafe {
            core::slice::from_raw_parts_mut(
                self.log.add(self.log_used),
                self.log_capacity - self.log_used,
            )
        };
        match event2(log, pcr, event_type, list, data) {
            Some(length) => {
                self.last_event = self.log_used;
                self.log_used += length;
            }
            None => self.truncated = true,
        }
        if self.log_handed_out {
            let space = unsafe {
                core::slice::from_raw_parts_mut(
                    self.final_events.add(self.final_used),
                    self.final_capacity - self.final_used,
                )
            };
            if let Some(length) = event2(space, pcr, event_type, list, data) {
                self.final_used += length;
                let count = unsafe { self.final_events.add(8).cast::<u64>() };
                unsafe { count.write_unaligned(count.read_unaligned() + 1) };
            }
        }
    }

    fn measure(&mut self, pcr: u32, event_type: u32, digests: &Digests, data: &[u8], log: bool) {
        if !self.extend(pcr, digests) {
            crate::serial::println!("tpm: extending PCR {pcr} failed");
            return;
        }
        if log {
            self.append(pcr, event_type, digests, data);
        }
    }

    /// An event whose digest is the hash of its own data.
    fn measure_event(&mut self, pcr: u32, event_type: u32, data: &[u8]) {
        let digests = self.digests_of(data);
        self.measure(pcr, event_type, &digests, data, true);
    }

    fn measure_action(&mut self, pcr: u32, action: &str) {
        let digests = self.digests_of(action.as_bytes());
        self.measure(pcr, EV_EFI_ACTION, &digests, action.as_bytes(), true);
    }

    fn active_bits(&self) -> u32 {
        self.banks[..self.bank_count]
            .iter()
            .map(|bank| bank_bit(*bank))
            .fold(0, |bits, bit| bits | bit)
    }
}

fn with_tpm<R>(f: impl FnOnce(&mut Tpm) -> R) -> Option<R> {
    TPM.get().as_mut().map(f)
}

/// Starts the TPM, sets up the log and publishes `EFI_TCG2_PROTOCOL` when the machine has a
/// TPM. Measures the firmware's version into PCR 0, which is the first thing measured.
pub fn init(handoff: &Handoff) {
    let Some(base) = handoff.tpm_base else {
        return;
    };
    write32(base + LOC_CTRL, LOC_CTRL_REQUEST_ACCESS);
    let mut tpm = Tpm {
        base,
        banks: [0; MAX_BANKS],
        bank_count: 0,
        log: ptr::null_mut(),
        log_capacity: 0,
        log_used: 0,
        last_event: 0,
        truncated: false,
        final_events: ptr::null_mut(),
        final_capacity: 0,
        final_used: FINAL_EVENTS_HEADER_BYTES,
        log_handed_out: false,
        boot_attempted: false,
        exit_measured: false,
        authorities: Vec::new(),
    };
    let mut response = [0u8; COMMAND_BYTES];
    let started = tpm
        .submit(&startup_command(), &mut response)
        .and_then(|length| response_code(&response[..length]));
    if !matches!(started, Some(RC_SUCCESS | RC_INITIALIZE)) {
        crate::serial::println!("tpm: TPM2_Startup failed ({started:x?}); measured boot is off");
        return;
    }
    let Some((banks, count)) = tpm
        .submit(&pcr_allocation_command(), &mut response)
        .and_then(|length| allocated_banks(&response[..length]))
        .filter(|(_, count)| *count > 0)
    else {
        crate::serial::println!("tpm: no PCR bank the firmware can hash; measured boot is off");
        return;
    };
    tpm.banks = banks;
    tpm.bank_count = count;
    let (Some(log), Some(final_events)) = (
        memory::allocate_zeroed_pages(BOOT_SERVICES_DATA, LOG_PAGES),
        memory::allocate_zeroed_pages(ACPI_NVS, FINAL_EVENTS_PAGES),
    ) else {
        crate::serial::println!("tpm: no memory for the event log; measured boot is off");
        return;
    };
    tpm.log = log;
    tpm.log_capacity = (LOG_PAGES * PAGE_SIZE) as usize;
    tpm.final_events = final_events;
    tpm.final_capacity = (FINAL_EVENTS_PAGES * PAGE_SIZE) as usize;
    unsafe {
        final_events
            .cast::<u64>()
            .write_unaligned(FINAL_EVENTS_VERSION)
    };
    let header = unsafe { core::slice::from_raw_parts_mut(log, tpm.log_capacity) };
    tpm.log_used = spec_id_event(header, &tpm.banks[..count]).unwrap_or(0);
    let version: Vec<u8> = crate::text::to_utf16_nul(FIRMWARE_VERSION)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    tpm.measure_event(PCR_PLATFORM_CODE, EV_S_CRTM_VERSION, &version);
    *TPM.get() = Some(tpm);
    let _ = protocols::install(
        ptr::null_mut(),
        TCG2_PROTOCOL,
        PROTOCOL.get() as *mut Tcg2Protocol as *mut c_void,
    );
    tables::install_configuration(TCG2_FINAL_EVENTS_TABLE, final_events.cast());
}

/// PCR 7's Secure Boot configuration: `SecureBoot`, PK, KEK, db and dbx, each measured even
/// when empty or absent, then PCR 7's separator so that the authorities that verify images
/// come after it (PFP 3.3.4.8).
pub fn measure_secure_boot_policy() {
    with_tpm(|tpm| {
        for (name, guid) in [
            ("SecureBoot", guids::GLOBAL_VARIABLE),
            ("PK", guids::GLOBAL_VARIABLE),
            ("KEK", guids::GLOBAL_VARIABLE),
            ("db", guids::IMAGE_SECURITY_DATABASE),
            ("dbx", guids::IMAGE_SECURITY_DATABASE),
        ] {
            let data = variables::read(name, &guid).unwrap_or_default();
            let event = variable_data(&guid, name, &data);
            tpm.measure_event(
                PCR_SECURE_BOOT_POLICY,
                EV_EFI_VARIABLE_DRIVER_CONFIG,
                &event,
            );
        }
        tpm.measure_event(PCR_SECURE_BOOT_POLICY, EV_SEPARATOR, &SEPARATOR);
    });
}

/// The db entry (`EFI_SIGNATURE_DATA`) that allowed an image, once per entry.
pub fn measure_authority(signature: &[u8]) {
    with_tpm(|tpm| {
        if tpm.authorities.iter().any(|known| known == signature) {
            return;
        }
        let event = variable_data(&guids::IMAGE_SECURITY_DATABASE, "db", signature);
        tpm.measure_event(PCR_SECURE_BOOT_POLICY, EV_EFI_VARIABLE_AUTHORITY, &event);
        tpm.authorities.push(signature.to_vec());
    });
}

/// A PE image `LoadImage` was given, by its Authenticode digest (the file's plain hash when it
/// cannot be parsed for one), as EDK2's `DxeTpm2MeasureBootLib` measures it: applications into
/// PCR 4, drivers into PCR 2, with the file's address and size as the image's location.
pub fn measure_image(file: &[u8], subsystem: u16, link_address: u64, device_path: &[u8]) {
    with_tpm(|tpm| {
        let (pcr, event_type) = match subsystem {
            crate::pe::SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER => {
                (PCR_DRIVERS, EV_EFI_BOOT_SERVICES_DRIVER)
            }
            crate::pe::SUBSYSTEM_EFI_RUNTIME_DRIVER => {
                (PCR_DRIVERS, EV_EFI_RUNTIME_SERVICES_DRIVER)
            }
            _ => (PCR_BOOT_MANAGER_CODE, EV_EFI_BOOT_SERVICES_APPLICATION),
        };
        let digests = tpm.digests(|bank, out| {
            let digest =
                algorithm_of(bank).and_then(|algorithm| authenticode::digest(file, algorithm).ok());
            match digest {
                Some(digest) => out[..digest.len()].copy_from_slice(&digest),
                None => hash_into(bank, file, out),
            }
        });
        let event = image_load_event(
            file.as_ptr() as u64,
            file.len() as u64,
            link_address,
            device_path,
        );
        tpm.measure(pcr, event_type, &digests, &event, true);
    });
}

/// Something the firmware loads that is not a PE image (Xen's Multiboot2 ELF, the dom0 kernel
/// and initramfs it hands Xen, their command lines), measured as GRUB measures what it loads.
pub fn measure_loaded(pcr: u32, data: &[u8], description: &str) {
    with_tpm(|tpm| {
        let digests = tpm.digests_of(data);
        tpm.measure(pcr, EV_IPL, &digests, description.as_bytes(), true);
    });
}

fn measure_boot_variables(tpm: &mut Tpm) {
    let Some(order) = variables::read("BootOrder", &guids::GLOBAL_VARIABLE) else {
        return;
    };
    let event = variable_data(&guids::GLOBAL_VARIABLE, "BootOrder", &order);
    tpm.measure_event(PCR_PLATFORM_CONFIGURATION, EV_EFI_VARIABLE_BOOT2, &event);
    for number in crate::load_option::u16_list(&order) {
        let name = alloc::format!("Boot{number:04X}");
        if let Some(option) = variables::read(&name, &guids::GLOBAL_VARIABLE) {
            let event = variable_data(&guids::GLOBAL_VARIABLE, &name, &option);
            tpm.measure_event(PCR_PLATFORM_CONFIGURATION, EV_EFI_VARIABLE_BOOT2, &event);
        }
    }
}

/// Before a boot option (or the supplied kernel) starts. The first time: the boot variables
/// into PCR 1, the "Calling" action into PCR 4 and the separators of PCRs 0 to 6, the end of
/// the platform's part of the boot; afterwards only the action.
pub fn before_boot_option() {
    with_tpm(|tpm| {
        let first = !tpm.boot_attempted;
        tpm.boot_attempted = true;
        if first {
            measure_boot_variables(tpm);
        }
        tpm.measure_action(PCR_BOOT_MANAGER_CODE, ACTION_CALLING_BOOT_OPTION);
        if first {
            for pcr in PCR_PLATFORM_CODE..PCR_SECURE_BOOT_POLICY {
                tpm.measure_event(pcr, EV_SEPARATOR, &SEPARATOR);
            }
        }
    });
}

pub fn after_boot_option() {
    with_tpm(|tpm| tpm.measure_action(PCR_BOOT_MANAGER_CODE, ACTION_RETURNING_FROM_BOOT_OPTION));
}

/// `ExitBootServices`, measured once even if the caller retries with a fresh map key.
pub fn exit_boot_services() {
    with_tpm(|tpm| {
        if tpm.exit_measured {
            return;
        }
        tpm.exit_measured = true;
        tpm.measure_action(PCR_BOOT_MANAGER_DATA, ACTION_EXIT_BOOT_SERVICES);
        tpm.measure_action(PCR_BOOT_MANAGER_DATA, ACTION_EXIT_BOOT_SERVICES_RETURNED);
    });
}

extern "efiapi" fn get_capability(_: *mut Tcg2Protocol, capability: *mut u8) -> Status {
    if capability.is_null() {
        return INVALID_PARAMETER;
    }
    let size = unsafe { capability.read() };
    if size < CAPABILITY_1_0_BYTES {
        return BUFFER_TOO_SMALL;
    }
    let (banks, present) =
        with_tpm(|tpm| tpm.active_bits()).map_or((0, false), |bits| (bits, true));
    let mut full = [0u8; CAPABILITY_BYTES as usize];
    let mut cursor = Cursor::new(&mut full);
    let filled = (|| {
        cursor.put(&[CAPABILITY_BYTES])?;
        cursor.put(&PROTOCOL_VERSION)?; // StructureVersion
        cursor.put(&PROTOCOL_VERSION)?; // ProtocolVersion
        cursor.put(&banks.to_le_bytes())?; // HashAlgorithmBitmap
        cursor.put(&EVENT_LOG_FORMAT_TCG_2.to_le_bytes())?; // SupportedEventLogs
        cursor.put(&[present as u8])?; // TPMPresentFlag
        cursor.put(&TPM_BUFFER_BYTES.to_le_bytes())?; // MaxCommandSize
        cursor.put(&TPM_BUFFER_BYTES.to_le_bytes())?; // MaxResponseSize
        cursor.put(&MANUFACTURER_ID.to_le_bytes())?;
        cursor.put(&(banks.count_ones()).to_le_bytes())?; // NumberOfPcrBanks
        cursor.put(&banks.to_le_bytes()) // ActivePcrBanks
    })();
    debug_assert!(filled.is_some());
    let returned = if size >= CAPABILITY_BYTES {
        CAPABILITY_BYTES
    } else {
        CAPABILITY_1_0_BYTES
    };
    full[0] = returned;
    unsafe { ptr::copy_nonoverlapping(full.as_ptr(), capability, returned as usize) };
    SUCCESS
}

extern "efiapi" fn get_event_log(
    _: *mut Tcg2Protocol,
    format: u32,
    location: *mut u64,
    last_entry: *mut u64,
    truncated: *mut u8,
) -> Status {
    if location.is_null() || last_entry.is_null() || truncated.is_null() {
        return INVALID_PARAMETER;
    }
    if format != EVENT_LOG_FORMAT_TCG_2 {
        return INVALID_PARAMETER;
    }
    let found = with_tpm(|tpm| {
        tpm.log_handed_out = true;
        (
            tpm.log as u64,
            tpm.log as u64 + tpm.last_event as u64,
            tpm.truncated,
        )
    });
    let (start, last, was_truncated) = found.unwrap_or((0, 0, false));
    unsafe {
        location.write_unaligned(start);
        last_entry.write_unaligned(last);
        truncated.write(was_truncated as u8);
    }
    SUCCESS
}

extern "efiapi" fn hash_log_extend_event(
    _: *mut Tcg2Protocol,
    flags: u64,
    data: u64,
    data_length: u64,
    event: *const u8,
) -> Status {
    if event.is_null() || (data == 0 && data_length != 0) {
        return INVALID_PARAMETER;
    }
    let read_u32 = |at: usize| unsafe { event.add(at).cast::<u32>().read_unaligned() };
    let size = read_u32(0);
    let header_size = read_u32(4);
    let pcr = read_u32(10);
    let event_type = read_u32(14);
    if header_size != EVENT_HEADER_BYTES
        || unsafe { event.add(8).cast::<u16>().read_unaligned() } != EVENT_HEADER_VERSION
        || size < 4 + header_size
        || pcr > PCR_LAST
    {
        return INVALID_PARAMETER;
    }
    let event_data = unsafe {
        core::slice::from_raw_parts(
            event.add(4 + header_size as usize),
            (size - 4 - header_size) as usize,
        )
    };
    let bytes = if data_length == 0 {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(data as *const u8, data_length as usize) }
    };
    let measured = with_tpm(|tpm| {
        let digests = if flags & PE_COFF_IMAGE != 0 {
            let mut parsed = true;
            let digests = tpm.digests(|bank, out| {
                match algorithm_of(bank)
                    .and_then(|algorithm| authenticode::digest(bytes, algorithm).ok())
                {
                    Some(digest) => out[..digest.len()].copy_from_slice(&digest),
                    None => parsed = false,
                }
            });
            if !parsed {
                return UNSUPPORTED;
            }
            digests
        } else {
            tpm.digests_of(bytes)
        };
        tpm.measure(
            pcr,
            event_type,
            &digests,
            event_data,
            flags & EXTEND_ONLY == 0,
        );
        SUCCESS
    });
    measured.unwrap_or(DEVICE_ERROR)
}

extern "efiapi" fn submit_command(
    _: *mut Tcg2Protocol,
    input_size: u32,
    input: *const u8,
    output_size: u32,
    output: *mut u8,
) -> Status {
    if input.is_null() || output.is_null() || input_size == 0 {
        return INVALID_PARAMETER;
    }
    let command = unsafe { core::slice::from_raw_parts(input, input_size as usize) };
    let response = unsafe { core::slice::from_raw_parts_mut(output, output_size as usize) };
    match with_tpm(|tpm| tpm.submit(command, response)) {
        Some(Some(_)) => SUCCESS,
        Some(None) => BUFFER_TOO_SMALL,
        None => DEVICE_ERROR,
    }
}

extern "efiapi" fn get_active_pcr_banks(_: *mut Tcg2Protocol, banks: *mut u32) -> Status {
    if banks.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe { banks.write_unaligned(with_tpm(|tpm| tpm.active_bits()).unwrap_or(0)) };
    SUCCESS
}

/// The library's TPM has one fixed allocation; asking for it again is all that succeeds.
extern "efiapi" fn set_active_pcr_banks(_: *mut Tcg2Protocol, banks: u32) -> Status {
    match with_tpm(|tpm| tpm.active_bits() == banks) {
        Some(true) => SUCCESS,
        _ => UNSUPPORTED,
    }
}

extern "efiapi" fn get_result_of_set_active_pcr_banks(
    _: *mut Tcg2Protocol,
    operation_present: *mut u32,
    response: *mut u32,
) -> Status {
    if operation_present.is_null() || response.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe {
        operation_present.write_unaligned(0);
        response.write_unaligned(0);
    }
    SUCCESS
}
