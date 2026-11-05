//! What measured boot writes, apart from the hardware: the TPM 2.0 commands the firmware sends
//! (big-endian, TPM 2.0 Library Part 3) and the crypto-agile event log (little-endian, TCG PC
//! Client Platform Firmware Profile 1.05, section 10). Building into caller-provided slices,
//! without the heap, lets `ExitBootServices` measure after the OS took its memory map.

use crate::uefi::Guid;

pub const EV_NO_ACTION: u32 = 0x3;
pub const EV_SEPARATOR: u32 = 0x4;
pub const EV_S_CRTM_VERSION: u32 = 0x8;
pub const EV_IPL: u32 = 0xd;
const EV_EFI_EVENT_BASE: u32 = 0x8000_0000;
pub const EV_EFI_VARIABLE_DRIVER_CONFIG: u32 = EV_EFI_EVENT_BASE + 0x1;
pub const EV_EFI_BOOT_SERVICES_APPLICATION: u32 = EV_EFI_EVENT_BASE + 0x3;
pub const EV_EFI_BOOT_SERVICES_DRIVER: u32 = EV_EFI_EVENT_BASE + 0x4;
pub const EV_EFI_RUNTIME_SERVICES_DRIVER: u32 = EV_EFI_EVENT_BASE + 0x5;
pub const EV_EFI_ACTION: u32 = EV_EFI_EVENT_BASE + 0x7;
/// PFP 1.05 replaced `EV_EFI_VARIABLE_BOOT` (whose digest covered the data only) with this,
/// whose digest covers the whole `UEFI_VARIABLE_DATA` like the other variable events.
pub const EV_EFI_VARIABLE_BOOT2: u32 = EV_EFI_EVENT_BASE + 0xc;
pub const EV_EFI_VARIABLE_AUTHORITY: u32 = EV_EFI_EVENT_BASE + 0xe0;

/// The PCRs the PC Client profile assigns.
pub const PCR_PLATFORM_CODE: u32 = 0;
pub const PCR_PLATFORM_CONFIGURATION: u32 = 1;
pub const PCR_DRIVERS: u32 = 2;
pub const PCR_BOOT_MANAGER_CODE: u32 = 4;
pub const PCR_BOOT_MANAGER_DATA: u32 = 5;
pub const PCR_SECURE_BOOT_POLICY: u32 = 7;
/// GRUB's: command lines into 8, files into 9. The firmware's own Multiboot2 loader uses them
/// the same way.
pub const PCR_COMMAND_LINES: u32 = 8;
pub const PCR_LOADED_FILES: u32 = 9;
pub const PCR_LAST: u32 = 23;

pub const ACTION_CALLING_BOOT_OPTION: &str = "Calling EFI Application from Boot Option";
pub const ACTION_RETURNING_FROM_BOOT_OPTION: &str =
    "Returning from EFI Application from Boot Option";
pub const ACTION_EXIT_BOOT_SERVICES: &str = "Exit Boot Services Invocation";
pub const ACTION_EXIT_BOOT_SERVICES_RETURNED: &str = "Exit Boot Services Returned with Success";
/// `EV_SEPARATOR`'s data: four zero bytes when the boot is going normally.
pub const SEPARATOR: [u8; 4] = [0; 4];

pub const ALG_SHA1: u16 = 0x0004;
pub const ALG_SHA256: u16 = 0x000b;
pub const ALG_SHA384: u16 = 0x000c;
pub const ALG_SHA512: u16 = 0x000d;

/// `EFI_TCG2_BOOT_HASH_ALG_*`: the bits of `HashAlgorithmBitmap` and `ActivePcrBanks`.
pub fn bank_bit(algorithm: u16) -> u32 {
    match algorithm {
        ALG_SHA1 => 1 << 0,
        ALG_SHA256 => 1 << 1,
        ALG_SHA384 => 1 << 2,
        ALG_SHA512 => 1 << 3,
        _ => 0,
    }
}

pub fn digest_len(algorithm: u16) -> Option<usize> {
    match algorithm {
        ALG_SHA1 => Some(20),
        ALG_SHA256 => Some(32),
        ALG_SHA384 => Some(48),
        ALG_SHA512 => Some(64),
        _ => None,
    }
}

/// A slice filled from the front, refusing (not truncating) what does not fit.
pub struct Cursor<'a> {
    buffer: &'a mut [u8],
    pub used: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(buffer: &'a mut [u8]) -> Self {
        Cursor { buffer, used: 0 }
    }

    pub fn put(&mut self, bytes: &[u8]) -> Option<()> {
        let end = self.used.checked_add(bytes.len())?;
        self.buffer.get_mut(self.used..end)?.copy_from_slice(bytes);
        self.used = end;
        Some(())
    }
}

const ST_NO_SESSIONS: u16 = 0x8001;
const ST_SESSIONS: u16 = 0x8002;
const CC_STARTUP: u32 = 0x144;
const CC_PCR_EXTEND: u32 = 0x182;
const CC_GET_CAPABILITY: u32 = 0x17a;
const SU_CLEAR: u16 = 0;
const RS_PW: u32 = 0x4000_0009;
const CAP_PCRS: u32 = 5;
pub const RC_SUCCESS: u32 = 0;
/// The TPM was already started: what Startup answers after a warm reset that did not reset
/// the TPM. The firmware carries on.
pub const RC_INITIALIZE: u32 = 0x100;
const HEADER_BYTES: usize = 10;
/// One password session with an empty password: handle, empty nonce, continueSession,
/// empty password.
const PASSWORD_SESSION: [u8; 9] = [0x40, 0, 0, 0x09, 0, 0, 0x01, 0, 0];

fn finish_header(command: &mut [u8]) {
    let length = command.len() as u32;
    command[2..6].copy_from_slice(&length.to_be_bytes());
}

pub fn startup_command() -> [u8; 12] {
    let mut command = [0u8; 12];
    command[..2].copy_from_slice(&ST_NO_SESSIONS.to_be_bytes());
    command[6..10].copy_from_slice(&CC_STARTUP.to_be_bytes());
    command[10..].copy_from_slice(&SU_CLEAR.to_be_bytes());
    finish_header(&mut command);
    command
}

/// `TPM2_GetCapability(TPM_CAP_PCRS)`.
pub fn pcr_allocation_command() -> [u8; 22] {
    let mut command = [0u8; 22];
    command[..2].copy_from_slice(&ST_NO_SESSIONS.to_be_bytes());
    command[6..10].copy_from_slice(&CC_GET_CAPABILITY.to_be_bytes());
    command[10..14].copy_from_slice(&CAP_PCRS.to_be_bytes());
    command[18..22].copy_from_slice(&1u32.to_be_bytes());
    finish_header(&mut command);
    command
}

pub fn response_code(response: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(response.get(6..10)?.try_into().ok()?))
}

/// The banks with at least one PCR allocated, from the `TPM2_GetCapability(TPM_CAP_PCRS)`
/// response. Algorithms the firmware cannot hash are left out: it could not extend them.
pub fn allocated_banks(response: &[u8]) -> Option<([u16; 4], usize)> {
    if response_code(response)? != RC_SUCCESS {
        return None;
    }
    // moreData, capability, count.
    let mut at = HEADER_BYTES + 1 + 4;
    let count = u32::from_be_bytes(response.get(at..at + 4)?.try_into().ok()?);
    at += 4;
    let mut banks = [0u16; 4];
    let mut found = 0;
    for _ in 0..count {
        let algorithm = u16::from_be_bytes(response.get(at..at + 2)?.try_into().ok()?);
        let select_bytes = *response.get(at + 2)? as usize;
        let bitmap = response.get(at + 3..at + 3 + select_bytes)?;
        at += 3 + select_bytes;
        if bitmap.iter().any(|byte| *byte != 0) && digest_len(algorithm).is_some() && found < 4 {
            banks[found] = algorithm;
            found += 1;
        }
    }
    Some((banks, found))
}

/// `TPM2_PCR_Extend(pcr, digests)` with a password session.
pub fn pcr_extend_command<'a>(
    out: &'a mut [u8],
    pcr: u32,
    digests: &[(u16, &[u8])],
) -> Option<&'a [u8]> {
    let mut cursor = Cursor::new(out);
    cursor.put(&ST_SESSIONS.to_be_bytes())?;
    cursor.put(&[0; 4])?;
    cursor.put(&CC_PCR_EXTEND.to_be_bytes())?;
    cursor.put(&pcr.to_be_bytes())?;
    cursor.put(&(PASSWORD_SESSION.len() as u32).to_be_bytes())?;
    cursor.put(&PASSWORD_SESSION)?;
    debug_assert_eq!(&PASSWORD_SESSION[..4], &RS_PW.to_be_bytes());
    cursor.put(&(digests.len() as u32).to_be_bytes())?;
    for (algorithm, digest) in digests {
        cursor.put(&algorithm.to_be_bytes())?;
        cursor.put(digest)?;
    }
    let Cursor { buffer, used } = cursor;
    let command = &mut buffer[..used];
    finish_header(command);
    Some(command)
}

/// The log's first entry, in the SHA-1 `TCG_PCR_EVENT` format every parser reads first: an
/// `EV_NO_ACTION` whose data is `TCG_EfiSpecIDEventStruct` naming the banks the later
/// `TCG_PCR_EVENT2` entries carry (PFP 10.4.5.1).
pub fn spec_id_event(out: &mut [u8], banks: &[u16]) -> Option<usize> {
    const SIGNATURE: &[u8; 16] = b"Spec ID Event03\0";
    const PLATFORM_CLASS_CLIENT: u32 = 0;
    const SPEC_VERSION_MINOR: u8 = 0;
    const SPEC_VERSION_MAJOR: u8 = 2;
    const SPEC_ERRATA: u8 = 0;
    /// `UINTN` is eight bytes on x86_64; the spec's code is 2.
    const UINTN_SIZE_64: u8 = 2;
    const SHA1_DIGEST_BYTES: usize = 20;
    let event_size = 16 + 4 + 4 + 4 + 4 * banks.len() + 1;
    let mut cursor = Cursor::new(out);
    cursor.put(&0u32.to_le_bytes())?;
    cursor.put(&EV_NO_ACTION.to_le_bytes())?;
    cursor.put(&[0; SHA1_DIGEST_BYTES])?;
    cursor.put(&(event_size as u32).to_le_bytes())?;
    cursor.put(SIGNATURE)?;
    cursor.put(&PLATFORM_CLASS_CLIENT.to_le_bytes())?;
    cursor.put(&[
        SPEC_VERSION_MINOR,
        SPEC_VERSION_MAJOR,
        SPEC_ERRATA,
        UINTN_SIZE_64,
    ])?;
    cursor.put(&(banks.len() as u32).to_le_bytes())?;
    for bank in banks {
        cursor.put(&bank.to_le_bytes())?;
        cursor.put(&(digest_len(*bank)? as u16).to_le_bytes())?;
    }
    // vendorInfoSize
    cursor.put(&[0])?;
    Some(cursor.used)
}

/// A `TCG_PCR_EVENT2`: PCR, type, the digests as `TPML_DIGEST_VALUES` (little-endian here,
/// unlike on the wire to the TPM), the event size and data.
pub fn event2(
    out: &mut [u8],
    pcr: u32,
    event_type: u32,
    digests: &[(u16, &[u8])],
    data: &[u8],
) -> Option<usize> {
    let mut cursor = Cursor::new(out);
    cursor.put(&pcr.to_le_bytes())?;
    cursor.put(&event_type.to_le_bytes())?;
    cursor.put(&(digests.len() as u32).to_le_bytes())?;
    for (algorithm, digest) in digests {
        cursor.put(&algorithm.to_le_bytes())?;
        cursor.put(digest)?;
    }
    cursor.put(&(data.len() as u32).to_le_bytes())?;
    cursor.put(data)?;
    Some(cursor.used)
}

/// `UEFI_VARIABLE_DATA`: the vendor GUID, the name's length in characters and the data's in
/// bytes (both `UINT64`), the name without its NUL, then the data.
pub fn variable_data(guid: &Guid, name: &str, data: &[u8]) -> alloc::vec::Vec<u8> {
    let units: alloc::vec::Vec<u16> = name.encode_utf16().collect();
    let mut out = guid.to_bytes().to_vec();
    out.extend_from_slice(&(units.len() as u64).to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for unit in units {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out.extend_from_slice(data);
    out
}

/// `UEFI_IMAGE_LOAD_EVENT`: where the image was loaded, how big it is in memory, its link-time
/// base and its device path.
pub fn image_load_event(
    base: u64,
    size: u64,
    link_address: u64,
    device_path: &[u8],
) -> alloc::vec::Vec<u8> {
    let mut out = alloc::vec::Vec::with_capacity(32 + device_path.len());
    out.extend_from_slice(&base.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&link_address.to_le_bytes());
    out.extend_from_slice(&(device_path.len() as u64).to_le_bytes());
    out.extend_from_slice(device_path);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uefi::guids;

    #[test]
    fn the_spec_id_event_names_the_banks() {
        let mut buffer = [0u8; 128];
        let length = spec_id_event(&mut buffer, &[ALG_SHA256]).unwrap();
        let event = &buffer[..length];
        assert_eq!(&event[4..8], &EV_NO_ACTION.to_le_bytes());
        let size = u32::from_le_bytes(event[28..32].try_into().unwrap()) as usize;
        assert_eq!(length, 32 + size);
        assert_eq!(&event[32..47], b"Spec ID Event03");
        assert_eq!(&event[56..60], &1u32.to_le_bytes());
        assert_eq!(&event[60..64], &[0x0b, 0, 32, 0]);
    }

    #[test]
    fn an_event_lays_out_its_digests_and_data() {
        let mut buffer = [0u8; 128];
        let digest = [7u8; 32];
        let length = event2(
            &mut buffer,
            4,
            EV_EFI_ACTION,
            &[(ALG_SHA256, &digest)],
            b"x",
        )
        .unwrap();
        assert_eq!(length, 4 + 4 + 4 + 2 + 32 + 4 + 1);
        assert_eq!(&buffer[12..14], &[0x0b, 0]);
        assert_eq!(buffer[length - 1], b'x');
        assert!(
            event2(
                &mut buffer[..20],
                4,
                EV_EFI_ACTION,
                &[(ALG_SHA256, &digest)],
                b""
            )
            .is_none()
        );
    }

    #[test]
    fn extend_and_startup_commands_have_consistent_headers() {
        let startup = startup_command();
        assert_eq!(u32::from_be_bytes(startup[2..6].try_into().unwrap()), 12);
        let mut buffer = [0u8; 128];
        let digest = [1u8; 32];
        let command = pcr_extend_command(&mut buffer, 7, &[(ALG_SHA256, &digest)]).unwrap();
        assert_eq!(
            u32::from_be_bytes(command[2..6].try_into().unwrap()) as usize,
            command.len()
        );
        assert_eq!(&command[10..14], &7u32.to_be_bytes());
        assert_eq!(command.len(), 10 + 4 + 4 + 9 + 4 + 2 + 32);
    }

    #[test]
    fn allocated_banks_skip_empty_and_unknown_ones() {
        let mut response = alloc::vec![
            0x80, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 3
        ];
        for (algorithm, bitmap) in [
            (ALG_SHA1, [0u8; 3]),
            (ALG_SHA256, [0xff; 3]),
            (0x0012, [0xff; 3]),
        ] {
            response.extend_from_slice(&algorithm.to_be_bytes());
            response.push(3);
            response.extend_from_slice(&bitmap);
        }
        let (banks, count) = allocated_banks(&response).unwrap();
        assert_eq!(&banks[..count], &[ALG_SHA256]);
    }

    #[test]
    fn variable_data_counts_characters_and_bytes() {
        let data = variable_data(&guids::GLOBAL_VARIABLE, "PK", &[1, 2, 3]);
        assert_eq!(&data[16..24], &2u64.to_le_bytes());
        assert_eq!(&data[24..32], &3u64.to_le_bytes());
        assert_eq!(&data[32..36], &[b'P', 0, b'K', 0]);
        assert_eq!(&data[36..], &[1, 2, 3]);
    }
}
