//! The hierarchies (Part 1, "Hierarchies"): their seeds, proofs and authorization values, what
//! persists of them in the machine's key-value store, `TPM2_Clear`,
//! `TPM2_HierarchyChangeAuth` and `TPM2_DictionaryAttackLockReset`.
//!
//! Everything a real TPM keeps in its NV memory goes to the store under `STORE_PREFIX`: the
//! platform, endorsement and storage seeds and proofs, the owner, endorsement and lockout
//! authorization values (`STATE_KEY`), persistent objects and NV indices. The store holds them
//! in the clear, as swtpm's state file does: the TPM's secrets are as safe as that file. The
//! guest's own window onto the store (`dev/kv.rs`) cannot reach these keys. The null
//! hierarchy's seed, the platform authorization and the context keys are volatile: a TPM Reset
//! (here, every new device, which is every guest reset) makes new ones.
#[cfg(freestanding)]
use crate::prelude::*;

use super::engine::{Response, Tpm, end_of};
use super::hash::MAX_DIGEST_BYTES;
use super::nv::NvIndex;
use super::object::{Object, Public, Sensitive, trim_auth};
use super::spec::*;
use super::wire::{Reader, Truncated, marshal};

/// Keys under this prefix belong to the TPM. `dev/kv.rs` hides them from the guest.
pub(crate) const STORE_PREFIX: &[u8] = b"tpm/";
const STATE_KEY: &[u8] = b"tpm/state";
const PERSISTENT_PREFIX: &str = "tpm/persistent/";
pub(super) const NV_PREFIX: &str = "tpm/nv/";
/// The first byte of every value the TPM stores, so a later format can tell old records apart.
pub(super) const STORE_VERSION: u8 = 1;
/// Seeds and proofs are as long as the largest digest.
const SEED_BYTES: usize = MAX_DIGEST_BYTES;
/// `TPM_PT_CONTEXT_HASH`'s digest size, which bounds a hierarchy's authorization value.
pub(super) const MAX_HIERARCHY_AUTH_BYTES: usize = 32;
/// How many objects may be persistent at once (`TPM_PT_HR_PERSISTENT_MIN` promises as many).
pub(super) const MAX_PERSISTENT_OBJECTS: usize = 16;

#[derive(Clone)]
pub(super) struct Hierarchy {
    pub seed: Vec<u8>,
    /// The HMAC key behind tickets this hierarchy's objects get.
    pub proof: Vec<u8>,
}

impl Hierarchy {
    fn generate(random: &mut super::engine::Random) -> Self {
        let mut seed = vec![0; SEED_BYTES];
        let mut proof = vec![0; SEED_BYTES];
        random.fill(&mut seed);
        random.fill(&mut proof);
        Hierarchy { seed, proof }
    }
}

/// What survives a TPM Reset.
#[derive(Clone)]
pub(super) struct PersistentState {
    pub platform: Hierarchy,
    pub endorsement: Hierarchy,
    pub storage: Hierarchy,
    pub owner_auth: Vec<u8>,
    pub endorsement_auth: Vec<u8>,
    pub lockout_auth: Vec<u8>,
}

impl PersistentState {
    /// A new TPM: fresh seeds and empty authorizations (what manufacture and `TPM2_Clear` leave).
    pub fn manufacture(random: &mut super::engine::Random) -> Self {
        PersistentState {
            platform: Hierarchy::generate(random),
            endorsement: Hierarchy::generate(random),
            storage: Hierarchy::generate(random),
            owner_auth: Vec::new(),
            endorsement_auth: Vec::new(),
            lockout_auth: Vec::new(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        marshal(|writer| {
            writer.u8(STORE_VERSION);
            for hierarchy in [&self.platform, &self.endorsement, &self.storage] {
                writer.sized(&hierarchy.seed);
                writer.sized(&hierarchy.proof);
            }
            writer.sized(&self.owner_auth);
            writer.sized(&self.endorsement_auth);
            writer.sized(&self.lockout_auth);
        })
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let parse = |reader: &mut Reader| -> Result<Option<Self>, Truncated> {
            if reader.u8()? != STORE_VERSION {
                return Ok(None);
            }
            let mut hierarchy = || {
                Ok::<_, Truncated>(Hierarchy {
                    seed: reader.sized()?.to_vec(),
                    proof: reader.sized()?.to_vec(),
                })
            };
            let (platform, endorsement, storage) = (hierarchy()?, hierarchy()?, hierarchy()?);
            Ok(Some(PersistentState {
                platform,
                endorsement,
                storage,
                owner_auth: reader.sized()?.to_vec(),
                endorsement_auth: reader.sized()?.to_vec(),
                lockout_auth: reader.sized()?.to_vec(),
            }))
        };
        let state = parse(&mut reader).ok().flatten()?;
        let seeds_ok = [&state.platform, &state.endorsement, &state.storage]
            .iter()
            .all(|hierarchy| hierarchy.seed.len() == SEED_BYTES && !hierarchy.proof.is_empty());
        (seeds_ok && reader.remaining() == 0).then_some(state)
    }
}

fn persistent_key(handle: u32) -> String {
    format!("{PERSISTENT_PREFIX}{handle:08x}")
}

pub(super) fn nv_key(handle: u32) -> String {
    format!("{NV_PREFIX}{handle:08x}")
}

fn encode_object(object: &Object) -> Vec<u8> {
    marshal(|writer| {
        writer.u8(STORE_VERSION);
        writer.u32(object.handle);
        writer.u32(object.hierarchy);
        writer.sized(&object.public.bytes());
        writer.sized(&object.sensitive_bytes());
        writer.sized(&object.qualified_name);
    })
}

/// A stored object, checked the way `TPM2_Load` checks one: a record that does not hold
/// together is dropped rather than trusted.
fn decode_object(bytes: &[u8]) -> Option<Object> {
    let mut reader = Reader::new(bytes);
    if reader.u8().ok()? != STORE_VERSION {
        return None;
    }
    let handle = reader.u32().ok()?;
    let hierarchy = reader.u32().ok()?;
    let public = Public::read_sized(&mut reader, 1).ok()?;
    let mut sensitive_reader = Reader::new(reader.sized().ok()?);
    let sensitive = Sensitive::read(&mut sensitive_reader, &public, 1).ok()?;
    let qualified_name = reader.sized().ok()?.to_vec();
    if reader.remaining() != 0 || sensitive_reader.remaining() != 0 {
        return None;
    }
    let name = public.name();
    Some(Object {
        handle,
        hierarchy,
        public,
        name,
        qualified_name,
        sensitive,
    })
}

impl Tpm {
    /// Reads the TPM's persistent state from the store, or manufactures it on first use.
    pub(super) fn load_persistent_state(&mut self) {
        let stored = self.store_get(STATE_KEY.to_vec());
        self.state = match stored.as_deref().and_then(PersistentState::decode) {
            Some(state) => state,
            None => {
                let state = PersistentState::manufacture(&mut self.random);
                self.state = state.clone();
                // A store that cannot be written leaves a TPM whose keys change every boot,
                // which is what a TPM without NV would be; nothing better is possible.
                let _ = self.save_state();
                state
            }
        };
        let persistent_keys = self.store_list(PERSISTENT_PREFIX.as_bytes());
        for key in persistent_keys {
            if let Some(object) = self.store_get(key).as_deref().and_then(decode_object)
                && handle_type(object.handle) == HT_PERSISTENT
                && self.object(object.handle).is_none()
            {
                self.objects.push(object);
            }
        }
        let nv_keys = self.store_list(NV_PREFIX.as_bytes());
        for key in nv_keys {
            if let Some(index) = self.store_get(key).as_deref().and_then(NvIndex::decode)
                && self.nv_index(index.handle).is_none()
            {
                self.nv.push(index);
            }
        }
    }

    pub(super) fn save_state(&mut self) -> Result<(), u32> {
        let encoded = self.state.encode();
        self.store_set(STATE_KEY, &encoded)
    }

    fn store_get(&self, key: Vec<u8>) -> Option<Vec<u8>> {
        self.store.lock().unwrap().get(&key)
    }

    fn store_list(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        self.store.lock().unwrap().list(prefix)
    }

    pub(super) fn store_set(&self, key: &[u8], value: &[u8]) -> Result<(), u32> {
        self.store
            .lock()
            .unwrap()
            .set(key, value)
            .map_err(|_| RC_NV_UNAVAILABLE)
    }

    pub(super) fn store_delete(&self, key: &[u8]) -> Result<(), u32> {
        self.store
            .lock()
            .unwrap()
            .delete(key)
            .map_err(|_| RC_NV_UNAVAILABLE)
    }

    pub(super) fn add_persistent(&mut self, object: Object) -> Result<(), u32> {
        let persistent = self
            .objects
            .iter()
            .filter(|object| handle_type(object.handle) == HT_PERSISTENT)
            .count();
        if persistent >= MAX_PERSISTENT_OBJECTS {
            return Err(RC_NV_SPACE);
        }
        self.store_set(
            persistent_key(object.handle).as_bytes(),
            &encode_object(&object),
        )?;
        self.objects.push(object);
        Ok(())
    }

    pub(super) fn remove_persistent(&mut self, handle: u32) -> Result<(), u32> {
        self.store_delete(persistent_key(handle).as_bytes())?;
        self.objects.retain(|object| object.handle != handle);
        Ok(())
    }

    pub(super) fn persistent_count(&self) -> usize {
        self.objects
            .iter()
            .filter(|object| handle_type(object.handle) == HT_PERSISTENT)
            .count()
    }

    pub(super) fn seed(&self, hierarchy: u32) -> &[u8] {
        match hierarchy {
            RH_PLATFORM => &self.state.platform.seed,
            RH_ENDORSEMENT => &self.state.endorsement.seed,
            RH_OWNER => &self.state.storage.seed,
            _ => &self.null.seed,
        }
    }

    pub(super) fn proof(&self, hierarchy: u32) -> &[u8] {
        match hierarchy {
            RH_PLATFORM => &self.state.platform.proof,
            RH_ENDORSEMENT => &self.state.endorsement.proof,
            RH_OWNER => &self.state.storage.proof,
            _ => &self.null.proof,
        }
    }

    /// The authorization value of a hierarchy or of the lockout. `TPM_RH_NULL`'s is empty.
    pub(super) fn hierarchy_auth(&self, handle: u32) -> &[u8] {
        match handle {
            RH_OWNER => &self.state.owner_auth,
            RH_ENDORSEMENT => &self.state.endorsement_auth,
            RH_LOCKOUT => &self.state.lockout_auth,
            RH_PLATFORM | RH_PLATFORM_NV => &self.platform_auth,
            _ => &[],
        }
    }

    pub(super) fn hierarchy_change_auth(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let new_auth = trim_auth(parameters.sized()?);
        end_of(parameters)?;
        if new_auth.len() > MAX_HIERARCHY_AUTH_BYTES {
            return Err(parameter_error(RC_SIZE, 1));
        }
        match handle {
            RH_PLATFORM => {
                self.platform_auth = new_auth;
                return Ok(Response::empty());
            }
            RH_OWNER => self.state.owner_auth = new_auth,
            RH_ENDORSEMENT => self.state.endorsement_auth = new_auth,
            RH_LOCKOUT => self.state.lockout_auth = new_auth,
            _ => return Err(handle_error(RC_VALUE, 1)),
        }
        self.save_state()?;
        Ok(Response::empty())
    }

    /// `TPM2_Clear`: the owner's hierarchy starts over. Its seed and the proofs change, so
    /// nothing made under the old ones loads again; the storage and endorsement hierarchies'
    /// objects go, persistent ones too, and so do the owner's NV indices (Part 3, "TPM2_Clear").
    /// The endorsement seed stays.
    pub(super) fn clear(&mut self, handle: u32, parameters: &mut Reader) -> Result<Response, u32> {
        end_of(parameters)?;
        if !matches!(handle, RH_LOCKOUT | RH_PLATFORM) {
            return Err(handle_error(RC_VALUE, 1));
        }
        let owned: Vec<u32> = self
            .objects
            .iter()
            .filter(|object| matches!(object.hierarchy, RH_OWNER | RH_ENDORSEMENT))
            .map(|object| object.handle)
            .collect();
        for handle in owned {
            if handle_type(handle) == HT_PERSISTENT {
                self.remove_persistent(handle)?;
            } else {
                self.objects.retain(|object| object.handle != handle);
            }
        }
        let owner_indices: Vec<u32> = self
            .nv
            .iter()
            .filter(|index| !index.platform_created())
            .map(|index| index.handle)
            .collect();
        for handle in owner_indices {
            self.remove_nv_index(handle)?;
        }
        let mut fresh = PersistentState::manufacture(&mut self.random);
        fresh.platform = self.state.platform.clone();
        fresh.endorsement.seed = self.state.endorsement.seed.clone();
        self.state = fresh;
        self.save_state()?;
        Ok(Response::empty())
    }

    /// Nothing to reset: this TPM never locks out (`TPM_PT_LOCKOUT_COUNTER` stays 0).
    pub(super) fn dictionary_attack_lock_reset(
        &mut self,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        Ok(Response::empty())
    }
}
