//! UEFI variable semantics over two byte-string stores: a fixed RAM arena for volatile
//! variables (it must live in the firmware image, which the OS keeps, so no heap) and the
//! persistent store the library provides. A key is `var/<guid>/<name>` in UTF-8 and a value is
//! the attributes (`u32`, little-endian) followed by the data.

use alloc::string::String;
use alloc::vec::Vec;

use crate::secure_boot::{self, SecureVariable, Timestamp};
use crate::uefi::variable_attribute::*;
use crate::uefi::{Guid, guids};

pub const KEY_PREFIX: &[u8] = b"var/";
/// A secure variable's last `EFI_TIME`, kept beside it in the same store under this prefix
/// (outside `var/`, so enumeration never sees it).
const TIMESTAMP_PREFIX: &[u8] = b"vartime/";
/// Present once anything changed PK, KEK, db or dbx through `SetVariable`: `VendorKeys` then
/// reads 0 (UEFI 2.9 3.3) until the store is wiped.
const KEYS_CHANGED_KEY: &[u8] = b"secure-boot/keys-changed";
/// What PK, KEK, db, dbx, dbt and dbr are always stored with; `APPEND_WRITE` may come on top.
pub const SECURE_ATTRIBUTES: u32 =
    NON_VOLATILE | BOOTSERVICE_ACCESS | RUNTIME_ACCESS | TIME_BASED_AUTHENTICATED_WRITE_ACCESS;
/// What `QueryVariableInfo` reports as the largest variable and the whole store.
pub const MAX_VARIABLE_BYTES: usize = 64 * 1024;
pub const STORAGE_BYTES: u64 = 1 << 20;
const ARENA_BYTES: usize = 192 * 1024;
const SLOTS: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableError {
    NotFound,
    InvalidParameter,
    WriteProtected,
    OutOfResources,
    SecurityViolation,
    Unsupported,
    DeviceError,
}

/// A byte-string store. `next_key` is the smallest key greater than `after` (any key when
/// `after` is empty) that starts with `KEY_PREFIX`.
pub trait Store {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>>;
    fn set(&mut self, key: &[u8], value: &[u8]) -> bool;
    fn delete(&mut self, key: &[u8]) -> bool;
    fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>>;
}

#[derive(Clone, Copy)]
struct Slot {
    used: bool,
    key_offset: u32,
    key_length: u16,
    value_offset: u32,
    value_length: u32,
}

/// Volatile storage in static memory.
pub struct RamStore {
    arena: [u8; ARENA_BYTES],
    used: usize,
    slots: [Slot; SLOTS],
}

impl RamStore {
    pub const fn new() -> Self {
        Self {
            arena: [0; ARENA_BYTES],
            used: 0,
            slots: [Slot {
                used: false,
                key_offset: 0,
                key_length: 0,
                value_offset: 0,
                value_length: 0,
            }; SLOTS],
        }
    }

    fn key_of(&self, slot: &Slot) -> &[u8] {
        &self.arena[slot.key_offset as usize..][..slot.key_length as usize]
    }

    fn find(&self, key: &[u8]) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.used && self.key_of(slot) == key)
    }

    /// Slides every live entry down over the holes deleted ones left.
    fn compact(&mut self) {
        let mut order: Vec<usize> = (0..SLOTS).filter(|&index| self.slots[index].used).collect();
        order.sort_by_key(|&index| self.slots[index].key_offset);
        let mut at = 0usize;
        for index in order {
            let slot = self.slots[index];
            let length = slot.key_length as usize + slot.value_length as usize;
            // Keys and values are stored back to back, key first.
            self.arena.copy_within(
                slot.key_offset as usize..slot.key_offset as usize + length,
                at,
            );
            self.slots[index].key_offset = at as u32;
            self.slots[index].value_offset = (at + slot.key_length as usize) as u32;
            at += length;
        }
        self.used = at;
    }
}

impl Store for RamStore {
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let slot = self.slots[self.find(key)?];
        Some(self.arena[slot.value_offset as usize..][..slot.value_length as usize].to_vec())
    }

    fn set(&mut self, key: &[u8], value: &[u8]) -> bool {
        let existing = self.find(key);
        let needed = key.len() + value.len();
        // Release the old entry first so replacing a big value does not need room for both.
        if let Some(index) = existing {
            self.slots[index].used = false;
        }
        let free_slot = self.slots.iter().position(|slot| !slot.used);
        if self.used + needed > ARENA_BYTES {
            self.compact();
        }
        let Some(index) = free_slot.filter(|_| self.used + needed <= ARENA_BYTES) else {
            // Out of room: the old value stays, as if the write never happened.
            if let Some(index) = existing {
                self.slots[index].used = true;
            }
            return false;
        };
        let at = self.used;
        self.arena[at..at + key.len()].copy_from_slice(key);
        self.arena[at + key.len()..at + needed].copy_from_slice(value);
        self.slots[index] = Slot {
            used: true,
            key_offset: at as u32,
            key_length: key.len() as u16,
            value_offset: (at + key.len()) as u32,
            value_length: value.len() as u32,
        };
        self.used += needed;
        true
    }

    fn delete(&mut self, key: &[u8]) -> bool {
        match self.find(key) {
            Some(index) => {
                self.slots[index].used = false;
                true
            }
            None => false,
        }
    }

    fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>> {
        self.slots
            .iter()
            .filter(|slot| slot.used)
            .map(|slot| self.key_of(slot))
            .filter(|key| *key > after && key.starts_with(KEY_PREFIX))
            .min()
            .map(<[u8]>::to_vec)
    }
}

/// Variables the firmware publishes and applications may not change.
const READ_ONLY: [&str; 13] = [
    "SecureBoot",
    "SetupMode",
    "AuditMode",
    "DeployedMode",
    "OsIndicationsSupported",
    "PlatformLangCodes",
    "BootOptionSupport",
    "SignatureSupport",
    "VendorKeys",
    "PKDefault",
    "KEKDefault",
    "dbDefault",
    "dbxDefault",
];

pub struct Variables<P: Store> {
    pub ram: RamStore,
    pub persistent: Option<P>,
}

pub fn key_for(name: &[u16], guid: &Guid) -> Vec<u8> {
    let mut key = KEY_PREFIX.to_vec();
    key.extend_from_slice(&guid.text());
    key.push(b'/');
    key.extend_from_slice(crate::text::to_string(name).as_bytes());
    key
}

/// Inverse of `key_for`.
pub fn split_key(key: &[u8]) -> Option<(Vec<u16>, Guid)> {
    let rest = key.strip_prefix(KEY_PREFIX)?;
    let guid = Guid::parse(rest.get(..36)?)?;
    let name = core::str::from_utf8(rest.get(37..)?).ok()?;
    Some((name.encode_utf16().collect(), guid))
}

fn timestamp_key(name: &[u16], guid: &Guid) -> Vec<u8> {
    let mut key = TIMESTAMP_PREFIX.to_vec();
    key.extend_from_slice(&key_for(name, guid)[KEY_PREFIX.len()..]);
    key
}

fn split_value(value: &[u8]) -> Option<(u32, &[u8])> {
    Some((
        u32::from_le_bytes(value.get(..4)?.try_into().ok()?),
        &value[4..],
    ))
}

impl<P: Store> Variables<P> {
    pub const fn new() -> Self {
        Self {
            ram: RamStore::new(),
            persistent: None,
        }
    }

    fn load(&mut self, key: &[u8]) -> Option<(bool, Vec<u8>)> {
        if let Some(value) = self.ram.get(key) {
            return Some((false, value));
        }
        self.persistent
            .as_mut()
            .and_then(|store| store.get(key))
            .map(|value| (true, value))
    }

    /// At runtime only variables marked `RUNTIME_ACCESS` exist for the caller.
    fn visible(attributes: u32, runtime: bool) -> bool {
        !runtime || attributes & RUNTIME_ACCESS != 0
    }

    pub fn get(
        &mut self,
        name: &[u16],
        guid: &Guid,
        runtime: bool,
    ) -> Result<(u32, Vec<u8>), VariableError> {
        if name.is_empty() {
            return Err(VariableError::InvalidParameter);
        }
        let (_, value) = self
            .load(&key_for(name, guid))
            .ok_or(VariableError::NotFound)?;
        let (attributes, data) = split_value(&value).ok_or(VariableError::DeviceError)?;
        if !Self::visible(attributes, runtime) {
            return Err(VariableError::NotFound);
        }
        Ok((attributes, data.to_vec()))
    }

    /// Stores a variable the way the firmware itself would, past every check an application
    /// call goes through.
    pub fn set_internal(&mut self, name: &str, guid: &Guid, attributes: u32, data: &[u8]) {
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut value = attributes.to_le_bytes().to_vec();
        value.extend_from_slice(data);
        self.ram.set(&key_for(&units, guid), &value);
    }

    pub fn set(
        &mut self,
        name: &[u16],
        guid: &Guid,
        attributes: u32,
        data: &[u8],
        runtime: bool,
    ) -> Result<(), VariableError> {
        if name.is_empty() || name[0] == 0 {
            return Err(VariableError::InvalidParameter);
        }
        if let Some(kind) = secure_boot::classify(&describe(name), guid) {
            return self.set_secure(kind, name, guid, attributes, data);
        }
        // Time-based authentication of other variables (the signer-bound "private" kind) is
        // not implemented; nothing this firmware boots relies on it.
        if attributes & (AUTHENTICATED_WRITE_ACCESS | TIME_BASED_AUTHENTICATED_WRITE_ACCESS) != 0 {
            return Err(VariableError::SecurityViolation);
        }
        if attributes & HARDWARE_ERROR_RECORD != 0 {
            return Err(VariableError::Unsupported);
        }
        let append = attributes & APPEND_WRITE != 0;
        let attributes = attributes & !APPEND_WRITE;
        if data.len() > MAX_VARIABLE_BYTES {
            return Err(VariableError::OutOfResources);
        }
        let key = key_for(name, guid);
        if *guid == guids::GLOBAL_VARIABLE && READ_ONLY.contains(&describe(name).as_str()) {
            return Err(VariableError::WriteProtected);
        }
        let existing = self.load(&key);
        let existing_parts = existing
            .as_ref()
            .and_then(|(_, value)| split_value(value))
            .map(|(old_attributes, old_data)| (old_attributes, old_data.to_vec()));
        // A variable hidden from runtime callers does not exist for them.
        let existing_parts = existing_parts.filter(|(old, _)| Self::visible(*old, runtime));
        // Zero attributes or no data deletes; appending nothing does nothing.
        let access = attributes & (BOOTSERVICE_ACCESS | RUNTIME_ACCESS);
        if !append && (data.is_empty() || access == 0) {
            return self.remove(&key, existing_parts.is_some());
        }
        if append && data.is_empty() {
            return Ok(());
        }
        if access == 0 {
            return Err(VariableError::InvalidParameter);
        }
        // Runtime access needs boot-service access too (UEFI 8.2).
        if attributes & RUNTIME_ACCESS != 0 && attributes & BOOTSERVICE_ACCESS == 0 {
            return Err(VariableError::InvalidParameter);
        }
        if runtime && attributes & RUNTIME_ACCESS == 0 {
            return Err(VariableError::InvalidParameter);
        }
        let mut combined = Vec::new();
        if let Some((old_attributes, old_data)) = &existing_parts {
            if *old_attributes != attributes {
                return Err(VariableError::InvalidParameter);
            }
            if append {
                combined = old_data.clone();
            }
        }
        combined.extend_from_slice(data);
        if combined.len() > MAX_VARIABLE_BYTES {
            return Err(VariableError::OutOfResources);
        }
        let mut value = attributes.to_le_bytes().to_vec();
        value.extend_from_slice(&combined);
        self.store(&key, &value, attributes & NON_VOLATILE != 0)
            .then_some(())
            .ok_or(VariableError::OutOfResources)
    }

    /// Writes `value` under `key` in the persistent store when it is non-volatile and there
    /// is one, else in RAM, and removes it from the other.
    fn store(&mut self, key: &[u8], value: &[u8], nonvolatile: bool) -> bool {
        if nonvolatile && let Some(store) = self.persistent.as_mut() {
            self.ram.delete(key);
            store.set(key, value)
        } else {
            if let Some(store) = self.persistent.as_mut() {
                store.delete(key);
            }
            self.ram.set(key, value)
        }
    }

    fn data_of(&mut self, name: &str, guid: &Guid) -> Option<Vec<u8>> {
        let units: Vec<u16> = name.encode_utf16().collect();
        self.get(&units, guid, false).ok().map(|(_, data)| data)
    }

    /// The PK's value; absent means setup mode.
    pub fn platform_key(&mut self) -> Option<Vec<u8>> {
        self.data_of("PK", &guids::GLOBAL_VARIABLE)
    }

    /// Enrolls a secure variable as the firmware's own default, past the authentication an
    /// application's write needs, with a zero timestamp so any signed update is newer.
    pub fn enroll(&mut self, name: &str, guid: &Guid, data: &[u8]) -> bool {
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut value = SECURE_ATTRIBUTES.to_le_bytes().to_vec();
        value.extend_from_slice(data);
        self.store(&key_for(&units, guid), &value, true)
            && self.store(
                &timestamp_key(&units, guid),
                &[0; secure_boot::TIMESTAMP_BYTES],
                true,
            )
    }

    /// Whether PK, KEK, db or dbx were changed since the defaults were enrolled.
    pub fn keys_changed(&mut self) -> bool {
        self.load(KEYS_CHANGED_KEY).is_some()
    }

    /// Republishes `SetupMode` (and `VendorKeys`) after the PK came or went. `SecureBoot`
    /// stays what it was at start-up: it describes the boot in progress.
    pub fn refresh_setup_mode(&mut self) {
        let visible = BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
        let has_platform_key = self.platform_key().is_some();
        let vendor_keys = has_platform_key && !self.keys_changed();
        self.set_internal(
            "SetupMode",
            &guids::GLOBAL_VARIABLE,
            visible,
            &[!has_platform_key as u8],
        );
        self.set_internal(
            "VendorKeys",
            &guids::GLOBAL_VARIABLE,
            visible,
            &[vendor_keys as u8],
        );
    }

    /// A write to PK, KEK, db, dbx, dbt or dbr (UEFI 2.9 8.2.2 and 32.3, as EDK2's
    /// `AuthVariableLib` does it): `EFI_VARIABLE_AUTHENTICATION_2` in front of the data, a
    /// timestamp later than the stored one unless appending, and in user mode a signature by
    /// the PK (for PK and KEK) or by a KEK or the PK (for the databases). In setup mode only a
    /// new PK is checked, against its own certificate, as EDK2 does.
    fn set_secure(
        &mut self,
        kind: SecureVariable,
        name: &[u16],
        guid: &Guid,
        attributes: u32,
        data: &[u8],
    ) -> Result<(), VariableError> {
        let append = attributes & APPEND_WRITE != 0;
        if attributes & !APPEND_WRITE != SECURE_ATTRIBUTES {
            return Err(VariableError::InvalidParameter);
        }
        let write = secure_boot::parse_authentication2(data)?;
        let key = key_for(name, guid);
        let time_key = timestamp_key(name, guid);
        let stored_time: Option<Timestamp> = self
            .load(&time_key)
            .and_then(|(_, value)| value.try_into().ok());
        if !append
            && let Some(stored) = &stored_time
            && !secure_boot::later(&write.timestamp, stored)
        {
            return Err(VariableError::SecurityViolation);
        }
        let platform_key = self.platform_key();
        let anchors: Vec<Vec<u8>> = match (&platform_key, kind) {
            (Some(platform_key), SecureVariable::Pk | SecureVariable::Kek) => {
                secure_boot::certificates(platform_key)
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect()
            }
            (Some(platform_key), SecureVariable::Database) => {
                let kek = self
                    .data_of("KEK", &guids::GLOBAL_VARIABLE)
                    .unwrap_or_default();
                secure_boot::certificates(&kek)
                    .into_iter()
                    .chain(secure_boot::certificates(platform_key))
                    .map(<[u8]>::to_vec)
                    .collect()
            }
            (None, SecureVariable::Pk) => secure_boot::certificates(write.payload)
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect(),
            (None, _) => Vec::new(),
        };
        let needs_signature = platform_key.is_some() || kind == SecureVariable::Pk;
        if needs_signature {
            let message = secure_boot::signed_message(
                name,
                guid,
                attributes,
                &write.timestamp,
                write.payload,
            );
            let anchors: Vec<&[u8]> = anchors.iter().map(Vec::as_slice).collect();
            if anchors.is_empty() || !secure_boot::signed_by(write.signed_data, &message, &anchors)
            {
                return Err(VariableError::SecurityViolation);
            }
        }
        if secure_boot::entries(write.payload).is_none()
            || (kind == SecureVariable::Pk
                && !write.payload.is_empty()
                && !secure_boot::valid_platform_key(write.payload))
        {
            return Err(VariableError::InvalidParameter);
        }
        let existing = self
            .load(&key)
            .and_then(|(_, value)| split_value(&value).map(|(_, data)| data.to_vec()));
        if !append && write.payload.is_empty() {
            self.ram.delete(&time_key);
            if let Some(store) = self.persistent.as_mut() {
                store.delete(&time_key);
            }
            if existing.is_some() {
                self.remove(&key, true)?;
            }
        } else {
            let combined = match (&existing, append) {
                (Some(existing), true) => secure_boot::append(existing, write.payload)
                    .ok_or(VariableError::InvalidParameter)?,
                _ => write.payload.to_vec(),
            };
            if combined.len() > MAX_VARIABLE_BYTES {
                return Err(VariableError::OutOfResources);
            }
            let timestamp = match stored_time {
                Some(stored) if append && !secure_boot::later(&write.timestamp, &stored) => stored,
                _ => write.timestamp,
            };
            let mut value = SECURE_ATTRIBUTES.to_le_bytes().to_vec();
            value.extend_from_slice(&combined);
            if !self.store(&key, &value, true) || !self.store(&time_key, &timestamp, true) {
                return Err(VariableError::OutOfResources);
            }
        }
        self.store(KEYS_CHANGED_KEY, &[1], true);
        self.refresh_setup_mode();
        Ok(())
    }

    fn remove(&mut self, key: &[u8], existed: bool) -> Result<(), VariableError> {
        if !existed {
            return Err(VariableError::NotFound);
        }
        self.ram.delete(key);
        if let Some(store) = self.persistent.as_mut() {
            store.delete(key);
        }
        Ok(())
    }

    /// The variable after `(name, guid)` in enumeration order: RAM keys, then persistent ones,
    /// each ascending. An empty name starts the walk; `NotFound` ends it.
    pub fn next(
        &mut self,
        name: &[u16],
        guid: &Guid,
        runtime: bool,
    ) -> Result<(Vec<u16>, Guid), VariableError> {
        let mut after = if name.is_empty() {
            Vec::new()
        } else {
            key_for(name, guid)
        };
        let mut in_ram = true;
        if !name.is_empty() {
            match self.load(&after) {
                Some((persistent, _)) => in_ram = !persistent,
                None => return Err(VariableError::InvalidParameter),
            }
        }
        loop {
            let candidate = if in_ram {
                match self.ram.next_key(&after) {
                    Some(key) => Some(key),
                    None => {
                        in_ram = false;
                        after = Vec::new();
                        continue;
                    }
                }
            } else {
                self.persistent
                    .as_mut()
                    .and_then(|store| store.next_key(&after))
            };
            let key = candidate.ok_or(VariableError::NotFound)?;
            let visible = self
                .load(&key)
                .and_then(|(_, value)| split_value(&value).map(|(attributes, _)| attributes))
                .is_some_and(|attributes| Self::visible(attributes, runtime));
            match split_key(&key) {
                Some((units, guid)) if visible => return Ok((units, guid)),
                _ => after = key,
            }
        }
    }

    /// (maximum storage, remaining storage, maximum variable size) for `QueryVariableInfo`.
    pub fn info(&mut self) -> (u64, u64, u64) {
        let mut used = 0u64;
        for in_ram in [true, false] {
            let mut after = Vec::new();
            loop {
                let next = if in_ram {
                    self.ram.next_key(&after)
                } else {
                    self.persistent
                        .as_mut()
                        .and_then(|store| store.next_key(&after))
                };
                let Some(key) = next else { break };
                used += self
                    .load(&key)
                    .map_or(0, |(_, value)| (key.len() + value.len()) as u64);
                after = key;
            }
        }
        (
            STORAGE_BYTES,
            STORAGE_BYTES.saturating_sub(used),
            MAX_VARIABLE_BYTES as u64,
        )
    }
}

#[cfg(test)]
/// `"Boot0001"`-style names for the boot manager and tests.
pub fn name_units(name: &str) -> Vec<u16> {
    name.encode_utf16().collect()
}

pub fn describe(name: &[u16]) -> String {
    crate::text::to_string(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::collections::BTreeMap;

    #[derive(Default)]
    struct Memory(BTreeMap<Vec<u8>, Vec<u8>>);

    impl Store for Memory {
        fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
            self.0.get(key).cloned()
        }
        fn set(&mut self, key: &[u8], value: &[u8]) -> bool {
            self.0.insert(key.to_vec(), value.to_vec());
            true
        }
        fn delete(&mut self, key: &[u8]) -> bool {
            self.0.remove(key).is_some()
        }
        fn next_key(&mut self, after: &[u8]) -> Option<Vec<u8>> {
            self.0
                .keys()
                .find(|key| key.as_slice() > after && key.starts_with(KEY_PREFIX))
                .cloned()
        }
    }

    const NV_BS_RT: u32 = NON_VOLATILE | BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
    const CUSTOM: Guid = Guid::new(1, 2, 3, [4; 8]);

    fn variables() -> Box<Variables<Memory>> {
        let mut variables = Box::new(Variables::new());
        variables.persistent = Some(Memory::default());
        variables
    }

    fn name(text: &str) -> Vec<u16> {
        name_units(text)
    }

    #[test]
    fn nonvolatile_variables_go_to_the_persistent_store() {
        let mut variables = variables();
        variables
            .set(
                &name("BootOrder"),
                &guids::GLOBAL_VARIABLE,
                NV_BS_RT,
                &[1, 0],
                false,
            )
            .unwrap();
        let (attributes, data) = variables
            .get(&name("BootOrder"), &guids::GLOBAL_VARIABLE, false)
            .unwrap();
        assert_eq!((attributes, data), (NV_BS_RT, vec![1, 0]));
        assert_eq!(variables.persistent.as_ref().unwrap().0.len(), 1);
        assert!(variables.ram.next_key(&[]).is_none());
    }

    #[test]
    fn volatile_variables_stay_in_ram() {
        let mut variables = variables();
        let attributes = BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
        variables
            .set(&name("Temp"), &CUSTOM, attributes, b"x", false)
            .unwrap();
        assert!(variables.persistent.as_ref().unwrap().0.is_empty());
        assert_eq!(variables.get(&name("Temp"), &CUSTOM, true).unwrap().1, b"x");
    }

    #[test]
    fn nonvolatile_without_a_store_falls_back_to_ram() {
        let mut variables: Box<Variables<Memory>> = Box::new(Variables::new());
        variables
            .set(&name("Kept"), &CUSTOM, NV_BS_RT, b"1", false)
            .unwrap();
        assert_eq!(
            variables.get(&name("Kept"), &CUSTOM, false).unwrap().1,
            b"1"
        );
    }

    #[test]
    fn boot_service_only_variables_vanish_at_runtime() {
        let mut variables = variables();
        variables
            .set(&name("Early"), &CUSTOM, BOOTSERVICE_ACCESS, b"1", false)
            .unwrap();
        assert!(variables.get(&name("Early"), &CUSTOM, false).is_ok());
        assert_eq!(
            variables.get(&name("Early"), &CUSTOM, true),
            Err(VariableError::NotFound)
        );
        assert_eq!(
            variables.next(&[], &Guid::default(), true),
            Err(VariableError::NotFound)
        );
    }

    #[test]
    fn writes_check_attributes() {
        let mut variables = variables();
        let set = |variables: &mut Variables<Memory>, attributes, data: &[u8], runtime| {
            variables.set(&name("V"), &CUSTOM, attributes, data, runtime)
        };
        assert_eq!(
            set(&mut variables, RUNTIME_ACCESS, b"x", false),
            Err(VariableError::InvalidParameter)
        );
        assert_eq!(set(&mut variables, NV_BS_RT, b"x", false), Ok(()));
        assert_eq!(
            set(&mut variables, BOOTSERVICE_ACCESS, b"x", false),
            Err(VariableError::InvalidParameter),
            "attributes cannot change"
        );
        assert_eq!(
            set(
                &mut variables,
                NON_VOLATILE | BOOTSERVICE_ACCESS,
                b"x",
                true
            ),
            Err(VariableError::InvalidParameter),
            "no runtime access at runtime"
        );
        assert_eq!(
            set(
                &mut variables,
                NV_BS_RT | TIME_BASED_AUTHENTICATED_WRITE_ACCESS,
                b"x",
                false
            ),
            Err(VariableError::SecurityViolation)
        );
        assert_eq!(
            variables.set(&[], &CUSTOM, NV_BS_RT, b"x", false),
            Err(VariableError::InvalidParameter)
        );
    }

    #[test]
    fn empty_data_deletes() {
        let mut variables = variables();
        variables
            .set(&name("V"), &CUSTOM, NV_BS_RT, b"x", false)
            .unwrap();
        variables
            .set(&name("V"), &CUSTOM, NV_BS_RT, b"", false)
            .unwrap();
        assert_eq!(
            variables.get(&name("V"), &CUSTOM, false),
            Err(VariableError::NotFound)
        );
        assert_eq!(
            variables.set(&name("V"), &CUSTOM, NV_BS_RT, b"", false),
            Err(VariableError::NotFound)
        );
        variables
            .set(&name("W"), &CUSTOM, NV_BS_RT, b"x", false)
            .unwrap();
        variables.set(&name("W"), &CUSTOM, 0, b"x", false).unwrap();
        assert_eq!(
            variables.get(&name("W"), &CUSTOM, false),
            Err(VariableError::NotFound)
        );
    }

    #[test]
    fn append_extends_the_data() {
        let mut variables = variables();
        variables
            .set(&name("V"), &CUSTOM, NV_BS_RT, b"ab", false)
            .unwrap();
        variables
            .set(&name("V"), &CUSTOM, NV_BS_RT | APPEND_WRITE, b"cd", false)
            .unwrap();
        assert_eq!(
            variables.get(&name("V"), &CUSTOM, false).unwrap().1,
            b"abcd"
        );
        variables
            .set(&name("New"), &CUSTOM, NV_BS_RT | APPEND_WRITE, b"x", false)
            .unwrap();
        assert_eq!(variables.get(&name("New"), &CUSTOM, false).unwrap().1, b"x");
    }

    #[test]
    fn the_firmware_owned_variables_are_read_only() {
        let mut variables = variables();
        variables.set_internal(
            "SecureBoot",
            &guids::GLOBAL_VARIABLE,
            BOOTSERVICE_ACCESS | RUNTIME_ACCESS,
            &[0],
        );
        assert_eq!(
            variables
                .get(&name("SecureBoot"), &guids::GLOBAL_VARIABLE, false)
                .unwrap()
                .1,
            [0]
        );
        assert_eq!(
            variables.set(
                &name("SecureBoot"),
                &guids::GLOBAL_VARIABLE,
                BOOTSERVICE_ACCESS | RUNTIME_ACCESS,
                &[1],
                false
            ),
            Err(VariableError::WriteProtected)
        );
        // The same name under another vendor is an ordinary variable.
        assert!(
            variables
                .set(&name("SecureBoot"), &CUSTOM, NV_BS_RT, &[1], false)
                .is_ok()
        );
    }

    #[test]
    fn enumeration_visits_every_variable_once_ram_first() {
        let mut variables = variables();
        variables.set_internal("B", &CUSTOM, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, b"1");
        variables.set_internal("A", &CUSTOM, BOOTSERVICE_ACCESS | RUNTIME_ACCESS, b"1");
        variables
            .set(&name("Z"), &CUSTOM, NV_BS_RT, b"1", false)
            .unwrap();
        variables
            .set(&name("Y"), &CUSTOM, NV_BS_RT, b"1", false)
            .unwrap();
        let mut seen = Vec::new();
        let (mut current, mut guid) = (Vec::new(), Guid::default());
        while let Ok((next, next_guid)) = variables.next(&current, &guid, false) {
            seen.push(describe(&next));
            (current, guid) = (next, next_guid);
        }
        assert_eq!(seen, ["A", "B", "Y", "Z"]);
        assert_eq!(
            variables.next(&name("Nope"), &CUSTOM, false),
            Err(VariableError::InvalidParameter)
        );
    }

    #[test]
    fn info_counts_what_is_stored() {
        let mut variables = variables();
        let (total, remaining_empty, maximum) = variables.info();
        variables
            .set(&name("V"), &CUSTOM, NV_BS_RT, &[0; 1000], false)
            .unwrap();
        let (_, remaining, _) = variables.info();
        assert_eq!(total, STORAGE_BYTES);
        assert!(remaining < remaining_empty - 1000);
        assert_eq!(maximum, MAX_VARIABLE_BYTES as u64);
    }

    #[test]
    fn ram_store_compacts_when_full_and_refuses_when_really_full() {
        let mut store = Box::new(RamStore::new());
        let value = vec![7u8; 60_000];
        for round in 0..20u8 {
            let key = [KEY_PREFIX, &[b'a' + (round % 2)]].concat();
            assert!(store.set(&key, &value), "round {round}");
        }
        assert_eq!(store.get(b"var/a").map(|v| v.len()), Some(60_000));
        assert!(store.set(b"var/c", &value));
        assert!(!store.set(b"var/d", &vec![0u8; ARENA_BYTES]), "cannot fit");
        assert_eq!(
            store.get(b"var/c").map(|v| v.len()),
            Some(60_000),
            "the others survive a refused write"
        );
    }

    const DBX_KEK2011: &[u8] = include_bytes!("../secure-boot/dbx-x64-kek2011.auth2");
    const DBX_KEK2023: &[u8] = include_bytes!("../secure-boot/dbx-x64-kek2023.auth2");

    fn enrolled() -> Box<Variables<Memory>> {
        let mut variables = variables();
        let defaults = secure_boot::defaults();
        assert!(variables.enroll("KEK", &guids::GLOBAL_VARIABLE, &defaults.kek));
        assert!(variables.enroll("db", &guids::IMAGE_SECURITY_DATABASE, &defaults.db));
        assert!(variables.enroll("PK", &guids::GLOBAL_VARIABLE, &defaults.pk));
        variables.refresh_setup_mode();
        variables
    }

    fn mode(variables: &mut Variables<Memory>, name: &str) -> u8 {
        variables
            .get(&name_units(name), &guids::GLOBAL_VARIABLE, false)
            .unwrap()
            .1[0]
    }

    fn dbx_length(variables: &mut Variables<Memory>) -> usize {
        variables
            .get(&name("dbx"), &guids::IMAGE_SECURITY_DATABASE, false)
            .map_or(0, |(_, data)| data.len())
    }

    const APPEND: u32 = SECURE_ATTRIBUTES | APPEND_WRITE;

    #[test]
    fn enrolled_keys_leave_setup_mode_and_count_as_vendor_keys() {
        let mut variables = variables();
        variables.refresh_setup_mode();
        assert_eq!(mode(&mut variables, "SetupMode"), 1);
        assert_eq!(mode(&mut variables, "VendorKeys"), 0);
        let mut variables = enrolled();
        assert_eq!(mode(&mut variables, "SetupMode"), 0);
        assert_eq!(mode(&mut variables, "VendorKeys"), 1);
        let (attributes, _) = variables
            .get(&name("PK"), &guids::GLOBAL_VARIABLE, true)
            .unwrap();
        assert_eq!(attributes, SECURE_ATTRIBUTES);
        assert!(
            variables.persistent.as_ref().unwrap().0.len() >= 6,
            "keys and timestamps are persistent"
        );
    }

    #[test]
    fn a_dbx_update_signed_by_a_microsoft_kek_appends_once() {
        let mut variables = enrolled();
        let dbx = name("dbx");
        let database = guids::IMAGE_SECURITY_DATABASE;
        variables
            .set(&dbx, &database, APPEND, DBX_KEK2011, true)
            .unwrap();
        let first = dbx_length(&mut variables);
        assert!(first > 9000, "{first}");
        // The 2023 update lists the same hashes and a few more; only those are added.
        variables
            .set(&dbx, &database, APPEND, DBX_KEK2023, true)
            .unwrap();
        let second = dbx_length(&mut variables);
        assert!(
            second >= first && second < first + 200,
            "{first} -> {second}"
        );
        variables
            .set(&dbx, &database, APPEND, DBX_KEK2023, true)
            .unwrap();
        assert_eq!(dbx_length(&mut variables), second);
        assert_eq!(mode(&mut variables, "VendorKeys"), 0, "the keys changed");
    }

    #[test]
    fn a_signed_write_checks_its_signature_and_attributes() {
        let mut variables = enrolled();
        let dbx = name("dbx");
        let database = guids::IMAGE_SECURITY_DATABASE;
        // Signed with APPEND_WRITE in the message: replacing instead does not verify.
        assert_eq!(
            variables.set(&dbx, &database, SECURE_ATTRIBUTES, DBX_KEK2011, false),
            Err(VariableError::SecurityViolation)
        );
        let mut tampered = DBX_KEK2011.to_vec();
        *tampered.last_mut().unwrap() ^= 1;
        assert_eq!(
            variables.set(&dbx, &database, APPEND, &tampered, false),
            Err(VariableError::SecurityViolation)
        );
        assert_eq!(
            variables.set(&dbx, &database, NV_BS_RT | APPEND_WRITE, DBX_KEK2011, false),
            Err(VariableError::InvalidParameter)
        );
        // The same payload aimed at db: the name is part of what was signed.
        assert_eq!(
            variables.set(&name("db"), &database, APPEND, DBX_KEK2011, false),
            Err(VariableError::SecurityViolation)
        );
        // A KEK signature is not enough for KEK or PK.
        for target in ["KEK", "PK"] {
            assert_eq!(
                variables.set(
                    &name(target),
                    &guids::GLOBAL_VARIABLE,
                    APPEND,
                    DBX_KEK2011,
                    false
                ),
                Err(VariableError::SecurityViolation),
                "{target}"
            );
        }
        assert_eq!(dbx_length(&mut variables), 0);
    }

    #[test]
    fn setup_mode_takes_database_writes_unsigned_but_checks_a_new_pk() {
        let mut variables = variables();
        variables.refresh_setup_mode();
        let database = guids::IMAGE_SECURITY_DATABASE;
        variables
            .set(&name("db"), &database, APPEND, DBX_KEK2011, false)
            .unwrap();
        assert!(
            variables
                .get(&name("db"), &database, false)
                .is_ok_and(|(_, data)| data.len() > 9000)
        );
        // A PK must be signed by itself even in setup mode, and this payload is no PK.
        assert_eq!(
            variables.set(
                &name("PK"),
                &guids::GLOBAL_VARIABLE,
                APPEND,
                DBX_KEK2011,
                false
            ),
            Err(VariableError::SecurityViolation)
        );
        assert_eq!(mode(&mut variables, "SetupMode"), 1);
    }

    #[test]
    fn a_stale_timestamp_is_refused_unless_appending() {
        let mut variables = variables();
        variables.refresh_setup_mode();
        let database = guids::IMAGE_SECURITY_DATABASE;
        variables
            .set(
                &name("dbx"),
                &database,
                SECURE_ATTRIBUTES,
                DBX_KEK2011,
                false,
            )
            .unwrap();
        assert_eq!(
            variables.set(
                &name("dbx"),
                &database,
                SECURE_ATTRIBUTES,
                DBX_KEK2011,
                false
            ),
            Err(VariableError::SecurityViolation),
            "a replay"
        );
        variables
            .set(&name("dbx"), &database, APPEND, DBX_KEK2023, false)
            .unwrap();
    }

    #[test]
    fn the_default_variables_cannot_be_written() {
        let mut variables = variables();
        for read_only in ["PKDefault", "dbxDefault", "VendorKeys"] {
            assert_eq!(
                variables.set(
                    &name(read_only),
                    &guids::GLOBAL_VARIABLE,
                    NV_BS_RT,
                    b"x",
                    false
                ),
                Err(VariableError::WriteProtected)
            );
        }
    }

    #[test]
    fn keys_split_back_into_name_and_guid() {
        let key = key_for(&name("Boot0001"), &CUSTOM);
        assert_eq!(split_key(&key), Some((name("Boot0001"), CUSTOM)));
        assert_eq!(split_key(b"var/short"), None);
        assert_eq!(split_key(b"other/x"), None);
    }
}
