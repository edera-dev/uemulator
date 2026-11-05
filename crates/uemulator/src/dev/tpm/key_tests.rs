//! Keys, sessions, policy, NV, contexts and persistence, driven the way a TSS drives them: the
//! `Client` below computes command HMACs, checks response HMACs and does the parameter
//! encryption itself, from the specification, so a test passes only when both sides agree.
use super::engine::{RC_SUCCESS, Random, Tpm};
use super::hash::{self, ALG_NULL, ALG_SHA256};
use super::object::Public;
use super::spec::*;
use super::wire::{Reader, Writer, marshal};
use crate::dev::RngSource;
use crate::dev::kv::SharedKeyValueStore;
use crate::kv::{KeyValueStore, MemoryKeyValueStore};
use crate::sync::{Arc, Mutex};
use uemulator_crypto::sha::HashAlgorithm;
use uemulator_crypto::{aes, ecc, kdf};

const SE_HMAC: u8 = 0x00;
const SE_POLICY: u8 = 0x01;
const SE_TRIAL: u8 = 0x03;
const AES_128_BITS: u16 = 128;
const DIGEST_BYTES: usize = 32;
const PCR_SECURE_BOOT: u32 = 7;
/// What systemd and Linux ask for in a storage key: fixedTPM, fixedParent, sensitiveDataOrigin,
/// userWithAuth, noDA, restricted, decrypt.
const STORAGE_ATTRIBUTES: u32 = OBJECT_FIXED_TPM
    | OBJECT_FIXED_PARENT
    | OBJECT_SENSITIVE_DATA_ORIGIN
    | OBJECT_USER_WITH_AUTH
    | OBJECT_NO_DA
    | OBJECT_RESTRICTED
    | OBJECT_DECRYPT;
const SEALED_ATTRIBUTES: u32 = OBJECT_FIXED_TPM | OBJECT_FIXED_PARENT | OBJECT_NO_DA;
const NV_INDEX: u32 = 0x0150_0001;
const SRK_HANDLE: u32 = 0x8100_0001;

fn store() -> SharedKeyValueStore {
    let store: Box<dyn KeyValueStore> = Box::new(MemoryKeyValueStore::new());
    Arc::new(Mutex::new(store))
}

fn sha256(parts: &[&[u8]]) -> Vec<u8> {
    hash::digest(ALG_SHA256, parts).unwrap()
}

fn kdfa(key: &[u8], label: &[u8], u: &[u8], v: &[u8], bits: u32) -> Vec<u8> {
    kdf::kdfa(HashAlgorithm::Sha256, key, label, u, v, bits)
}

fn hmac(key: &[u8], parts: &[&[u8]]) -> Vec<u8> {
    uemulator_crypto::hmac::hmac(HashAlgorithm::Sha256, key, parts)
}

/// AES-128-CFB over the first parameter of an area, as a session with decrypt or encrypt
/// does it.
fn crypt_first_parameter(key: &[u8], newer: &[u8], older: &[u8], area: &mut [u8], encrypt: bool) {
    let material = kdfa(key, b"CFB", newer, older, 256);
    let iv: [u8; 16] = material[16..].try_into().unwrap();
    let size = u16::from_be_bytes([area[0], area[1]]) as usize;
    let data = &mut area[2..2 + size];
    if encrypt {
        aes::cfb_encrypt(&material[..16], &iv, data).unwrap();
    } else {
        aes::cfb_decrypt(&material[..16], &iv, data).unwrap();
    }
}

#[derive(Clone)]
struct Session {
    handle: u32,
    key: Vec<u8>,
    nonce_tpm: Vec<u8>,
    attributes: u8,
    /// The authorized entity's value goes into the HMAC key (an HMAC session not bound to it,
    /// or a policy session after `PolicyAuthValue`).
    entity_auth: Vec<u8>,
}

enum Auth<'a> {
    Password(&'a [u8]),
    Session(&'a mut Session),
}

struct Client {
    tpm: Tpm,
    nonce_counter: u8,
}

struct Reply {
    handles: Vec<u32>,
    parameters: Vec<u8>,
}

impl Client {
    fn new(store: SharedKeyValueStore) -> Self {
        let mut client = Client {
            tpm: Tpm::new(Random::new(RngSource::Seeded(11)), 0, store),
            nonce_counter: 0,
        };
        client
            .call(CC_STARTUP, &[], &mut [], &0u16.to_be_bytes(), 0)
            .unwrap();
        client
    }

    fn fresh_nonce(&mut self) -> Vec<u8> {
        self.nonce_counter += 1;
        vec![self.nonce_counter; DIGEST_BYTES]
    }

    fn name(&self, handle: u32) -> Vec<u8> {
        self.tpm.entity_name(handle)
    }

    /// Sends a command and checks the response the way a TSS does. `Err` is the response code.
    fn call(
        &mut self,
        code: u32,
        handles: &[u32],
        auths: &mut [Auth],
        parameters: &[u8],
        response_handles: usize,
    ) -> Result<Reply, u32> {
        let mut parameters = parameters.to_vec();
        let nonces: Vec<Vec<u8>> = auths.iter().map(|_| self.fresh_nonce()).collect();
        for (auth, nonce) in auths.iter().zip(&nonces) {
            if let Auth::Session(session) = auth
                && session.attributes & SESSION_DECRYPT != 0
            {
                let key = [session.key.as_slice(), &session.entity_auth].concat();
                crypt_first_parameter(&key, nonce, &session.nonce_tpm, &mut parameters, true);
            }
        }
        let names: Vec<Vec<u8>> = handles.iter().map(|handle| self.name(*handle)).collect();
        let mut cp_parts: Vec<&[u8]> = Vec::new();
        let code_bytes = code.to_be_bytes();
        cp_parts.push(&code_bytes);
        cp_parts.extend(names.iter().map(Vec::as_slice));
        cp_parts.push(&parameters);
        let cp_hash = sha256(&cp_parts);
        let mut area = Writer::new();
        for (auth, nonce) in auths.iter().zip(&nonces) {
            match auth {
                Auth::Password(password) => {
                    area.u32(RS_PW);
                    area.sized(&[]);
                    area.u8(SESSION_CONTINUE);
                    area.sized(password);
                }
                Auth::Session(session) => {
                    let key = [session.key.as_slice(), &session.entity_auth].concat();
                    let mac = hmac(
                        &key,
                        &[&cp_hash, nonce, &session.nonce_tpm, &[session.attributes]],
                    );
                    area.u32(session.handle);
                    area.sized(nonce);
                    area.u8(session.attributes);
                    area.sized(&mac);
                }
            }
        }
        let mut writer = Writer::new();
        writer.u16(if auths.is_empty() {
            ST_NO_SESSIONS
        } else {
            ST_SESSIONS
        });
        writer.u32(0);
        writer.u32(code);
        for handle in handles {
            writer.u32(*handle);
        }
        if !auths.is_empty() {
            writer.u32(area.bytes.len() as u32);
            writer.bytes(&area.bytes);
        }
        writer.bytes(&parameters);
        let length = writer.bytes.len() as u32;
        writer.bytes[2..6].copy_from_slice(&length.to_be_bytes());
        let response = self.tpm.execute(&writer.bytes, 0);
        let mut reader = Reader::new(&response);
        let tag = reader.u16().unwrap();
        assert_eq!(reader.u32().unwrap() as usize, response.len());
        let rc = reader.u32().unwrap();
        if rc != RC_SUCCESS {
            assert_eq!(response.len(), 10);
            return Err(rc);
        }
        let reply_handles: Vec<u32> = (0..response_handles)
            .map(|_| reader.u32().unwrap())
            .collect();
        let mut reply_parameters = if tag == ST_SESSIONS {
            let size = reader.u32().unwrap() as usize;
            reader.take(size).unwrap().to_vec()
        } else {
            reader.rest().to_vec()
        };
        let rp_hash = sha256(&[&RC_SUCCESS.to_be_bytes(), &code_bytes, &reply_parameters]);
        for (auth, nonce) in auths.iter_mut().zip(&nonces) {
            let nonce_tpm = reader.sized().unwrap().to_vec();
            let attributes = reader.u8().unwrap();
            let mac = reader.sized().unwrap().to_vec();
            if let Auth::Session(session) = auth {
                assert_eq!(attributes, session.attributes);
                let key = [session.key.as_slice(), &session.entity_auth].concat();
                let expected = hmac(&key, &[&rp_hash, &nonce_tpm, nonce, &[attributes]]);
                assert_eq!(mac, expected, "response HMAC");
                session.nonce_tpm = nonce_tpm;
                if session.attributes & SESSION_ENCRYPT != 0 {
                    crypt_first_parameter(
                        &key,
                        &session.nonce_tpm,
                        nonce,
                        &mut reply_parameters,
                        false,
                    );
                }
            }
        }
        assert_eq!(reader.remaining(), 0);
        Ok(Reply {
            handles: reply_handles,
            parameters: reply_parameters,
        })
    }

    fn create_primary(&mut self, hierarchy: u32, template: &[u8]) -> (u32, Public) {
        let reply = self
            .call(
                CC_CREATE_PRIMARY,
                &[hierarchy],
                &mut [Auth::Password(&[])],
                &create_parameters(&[], &[], template),
                1,
            )
            .unwrap();
        let mut reader = Reader::new(&reply.parameters);
        let public = Public::read_sized(&mut reader, 1).unwrap();
        (reply.handles[0], public)
    }

    /// Starts a session salted with `salt_key` (`RH_NULL` for none), the way ESYS and Linux do.
    fn start_session(&mut self, salt_key: u32, session_type: u8, attributes: u8) -> Session {
        let nonce_caller = self.fresh_nonce();
        let (encrypted_salt, salt) = if salt_key == RH_NULL {
            (Vec::new(), Vec::new())
        } else {
            let public = self.tpm.object(salt_key).unwrap().public.clone();
            salt_for(&public)
        };
        let mut parameters = Writer::new();
        parameters.sized(&nonce_caller);
        parameters.sized(&encrypted_salt);
        parameters.u8(session_type);
        parameters.u16(ALG_AES);
        parameters.u16(AES_128_BITS);
        parameters.u16(ALG_CFB);
        parameters.u16(ALG_SHA256);
        let reply = self
            .call(
                CC_START_AUTH_SESSION,
                &[salt_key, RH_NULL],
                &mut [],
                &parameters.bytes,
                1,
            )
            .unwrap();
        let nonce_tpm = Reader::new(&reply.parameters).sized().unwrap().to_vec();
        let key = if salt.is_empty() {
            Vec::new()
        } else {
            kdfa(&salt, b"ATH", &nonce_tpm, &nonce_caller, 256)
        };
        Session {
            handle: reply.handles[0],
            key,
            nonce_tpm,
            attributes,
            entity_auth: Vec::new(),
        }
    }

    fn flush(&mut self, handle: u32) {
        self.call(CC_FLUSH_CONTEXT, &[], &mut [], &handle.to_be_bytes(), 0)
            .unwrap();
    }

    fn policy_pcr(&mut self, session: u32, pcr: u32) -> Result<Reply, u32> {
        let mut parameters = Writer::new();
        parameters.sized(&[]);
        parameters.bytes(&pcr_selection(pcr));
        self.call(CC_POLICY_PCR, &[session], &mut [], &parameters.bytes, 0)
    }

    fn policy_digest(&mut self, session: u32) -> Vec<u8> {
        let reply = self
            .call(CC_POLICY_GET_DIGEST, &[session], &mut [], &[], 0)
            .unwrap();
        Reader::new(&reply.parameters).sized().unwrap().to_vec()
    }

    fn extend(&mut self, pcr: u32, digest: &[u8]) {
        let mut parameters = Writer::new();
        parameters.u32(1);
        parameters.u16(ALG_SHA256);
        parameters.bytes(digest);
        self.call(
            CC_PCR_EXTEND,
            &[pcr],
            &mut [Auth::Password(&[])],
            &parameters.bytes,
            0,
        )
        .unwrap();
    }

    fn read_pcr(&mut self, pcr: u32) -> Vec<u8> {
        let reply = self
            .call(CC_PCR_READ, &[], &mut [], &pcr_selection(pcr), 0)
            .unwrap();
        let mut reader = Reader::new(&reply.parameters);
        reader.u32().unwrap();
        reader.take(4 + 2 + 1 + 3).unwrap();
        assert_eq!(reader.u32().unwrap(), 1);
        reader.sized().unwrap().to_vec()
    }
}

/// An encrypted salt for `public` and the salt itself: OAEP for RSA, an ephemeral ECDH point
/// and KDFe for ECC.
fn salt_for(public: &Public) -> (Vec<u8>, Vec<u8>) {
    if let Some(point) = public.ecc_point() {
        let mut ephemeral = [0u8; 32];
        ephemeral[31] = 0x2a;
        ephemeral[0] = 0x11;
        let ephemeral_public = ecc::p256_public_key(&ephemeral).unwrap();
        let z = ecc::p256_ecdh(&ephemeral, &point).unwrap();
        let salt = kdf::kdfe(
            HashAlgorithm::Sha256,
            &z,
            b"SECRET",
            &ephemeral_public.x,
            &point.x,
            256,
        );
        let encrypted = marshal(|writer| {
            writer.sized(&ephemeral_public.x);
            writer.sized(&ephemeral_public.y);
        });
        (encrypted, salt)
    } else {
        let key = public.rsa_public_key().unwrap();
        let salt = vec![0x5a; DIGEST_BYTES];
        let mut counter = 0u8;
        let encrypted = key
            .encrypt_oaep(HashAlgorithm::Sha256, b"SECRET\0", &salt, &mut |bytes| {
                for byte in bytes {
                    counter = counter.wrapping_add(1);
                    *byte = counter;
                }
            })
            .unwrap();
        (encrypted, salt)
    }
}

fn pcr_selection(pcr: u32) -> Vec<u8> {
    marshal(|writer| {
        writer.u32(1);
        writer.u16(ALG_SHA256);
        writer.u8(3);
        let mut bitmap = [0u8; 3];
        bitmap[pcr as usize / 8] = 1 << (pcr % 8);
        writer.bytes(&bitmap);
    })
}

fn ecc_storage_template() -> Vec<u8> {
    marshal(|writer| {
        writer.u16(ALG_ECC);
        writer.u16(ALG_SHA256);
        writer.u32(STORAGE_ATTRIBUTES);
        writer.sized(&[]);
        writer.u16(ALG_AES);
        writer.u16(AES_128_BITS);
        writer.u16(ALG_CFB);
        writer.u16(ALG_NULL);
        writer.u16(ECC_NIST_P256);
        writer.u16(ALG_NULL);
        writer.sized(&[]);
        writer.sized(&[]);
    })
}

fn rsa_storage_template() -> Vec<u8> {
    marshal(|writer| {
        writer.u16(ALG_RSA);
        writer.u16(ALG_SHA256);
        writer.u32(STORAGE_ATTRIBUTES);
        writer.sized(&[]);
        writer.u16(ALG_AES);
        writer.u16(AES_128_BITS);
        writer.u16(ALG_CFB);
        writer.u16(ALG_NULL);
        writer.u16(2048);
        writer.u32(0);
        writer.sized(&[]);
    })
}

fn sealed_template(attributes: u32, policy: &[u8]) -> Vec<u8> {
    marshal(|writer| {
        writer.u16(ALG_KEYEDHASH);
        writer.u16(ALG_SHA256);
        writer.u32(attributes);
        writer.sized(policy);
        writer.u16(ALG_NULL);
        writer.sized(&[]);
    })
}

/// `inSensitive`, `inPublic`, empty `outsideInfo` and `creationPCR`.
fn create_parameters(auth: &[u8], data: &[u8], template: &[u8]) -> Vec<u8> {
    marshal(|writer| {
        writer.sized_with(|writer| {
            writer.sized(auth);
            writer.sized(data);
        });
        writer.sized(template);
        writer.sized(&[]);
        writer.u32(0);
    })
}

/// Creates a sealed object under `parent` and returns its private and public parts.
fn seal(
    client: &mut Client,
    parent: u32,
    auth: &[u8],
    data: &[u8],
    template: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let reply = client
        .call(
            CC_CREATE,
            &[parent],
            &mut [Auth::Password(&[])],
            &create_parameters(auth, data, template),
            0,
        )
        .unwrap();
    let mut reader = Reader::new(&reply.parameters);
    (
        reader.sized().unwrap().to_vec(),
        reader.sized().unwrap().to_vec(),
    )
}

fn load(client: &mut Client, parent: u32, private: &[u8], public: &[u8]) -> Result<u32, u32> {
    let parameters = marshal(|writer| {
        writer.sized(private);
        writer.sized(public);
    });
    let reply = client.call(
        CC_LOAD,
        &[parent],
        &mut [Auth::Password(&[])],
        &parameters,
        1,
    )?;
    Ok(reply.handles[0])
}

fn unsealed(reply: Result<Reply, u32>) -> Result<Vec<u8>, u32> {
    Ok(Reader::new(&reply?.parameters).sized().unwrap().to_vec())
}

#[test]
fn a_primary_is_the_same_key_every_time_and_its_name_hashes_its_public_area() {
    let mut client = Client::new(store());
    let (first, public) = client.create_primary(RH_OWNER, &ecc_storage_template());
    client.flush(first);
    let (second, again) = client.create_primary(RH_OWNER, &ecc_storage_template());
    assert_eq!(public, again);
    assert!(ecc::p256_is_on_curve(&public.ecc_point().unwrap()));
    let reply = client
        .call(CC_READ_PUBLIC, &[second], &mut [], &[], 0)
        .unwrap();
    let mut reader = Reader::new(&reply.parameters);
    let read_back = Public::read_sized(&mut reader, 1).unwrap();
    let name = reader.sized().unwrap();
    let mut expected = ALG_SHA256.to_be_bytes().to_vec();
    expected.extend(sha256(&[&read_back.bytes()]));
    assert_eq!(name, expected);
    // Another hierarchy has another seed.
    let (endorsement, other) = client.create_primary(RH_ENDORSEMENT, &ecc_storage_template());
    assert_ne!(other, public);
    client.flush(endorsement);
}

#[test]
fn an_rsa_primary_has_a_2048_bit_modulus() {
    let mut client = Client::new(store());
    let (handle, public) = client.create_primary(RH_OWNER, &rsa_storage_template());
    assert_eq!(public.rsa_public_key().unwrap().size(), 256);
    client.flush(handle);
}

#[test]
fn templates_the_tpm_cannot_make_are_refused() {
    let mut client = Client::new(store());
    let mut rsa_1024 = rsa_storage_template();
    let bits_offset = rsa_1024.len() - 2 - 4 - 2;
    rsa_1024[bits_offset..bits_offset + 2].copy_from_slice(&1024u16.to_be_bytes());
    let refused = client.call(
        CC_CREATE_PRIMARY,
        &[RH_OWNER],
        &mut [Auth::Password(&[])],
        &create_parameters(&[], &[], &rsa_1024),
        1,
    );
    assert_eq!(refused.err(), Some(parameter_error(RC_KEY_SIZE, 2)));
    // A storage key needs a symmetric definition.
    let mut no_symmetric = ecc_storage_template();
    no_symmetric.splice(10..16, ALG_NULL.to_be_bytes());
    let refused = client.call(
        CC_CREATE_PRIMARY,
        &[RH_OWNER],
        &mut [Auth::Password(&[])],
        &create_parameters(&[], &[], &no_symmetric),
        1,
    );
    assert_eq!(refused.err(), Some(parameter_error(RC_SYMMETRIC, 2)));
    // TPM2_TestParms says the same before anything is made.
    let parms = |template: &[u8]| {
        let mut parms = template[..2].to_vec();
        parms.extend_from_slice(&template[2 + 2 + 4 + 2..template.len() - 2]);
        parms
    };
    assert!(
        client
            .call(
                CC_TEST_PARMS,
                &[],
                &mut [],
                &parms(&rsa_storage_template()),
                0
            )
            .is_ok()
    );
    assert_eq!(
        client
            .call(CC_TEST_PARMS, &[], &mut [], &parms(&rsa_1024), 0)
            .err(),
        Some(parameter_error(RC_KEY_SIZE, 1))
    );
}

#[test]
fn sealed_data_round_trips_through_create_load_and_unseal() {
    let mut client = Client::new(store());
    let (parent, _) = client.create_primary(RH_OWNER, &ecc_storage_template());
    let template = sealed_template(SEALED_ATTRIBUTES | OBJECT_USER_WITH_AUTH, &[]);
    let (private, public) = seal(&mut client, parent, b"pin", b"the volume key", &template);
    let item = load(&mut client, parent, &private, &public).unwrap();
    let wrong = client.call(CC_UNSEAL, &[item], &mut [Auth::Password(b"nip")], &[], 0);
    assert_eq!(wrong.err(), Some(session_error(RC_AUTH_FAIL, 1)));
    let right = client.call(CC_UNSEAL, &[item], &mut [Auth::Password(b"pin")], &[], 0);
    assert_eq!(unsealed(right).unwrap(), b"the volume key");
    // A private area changed in flight fails its integrity check.
    let mut tampered = private.clone();
    *tampered.last_mut().unwrap() ^= 1;
    assert_eq!(
        load(&mut client, parent, &tampered, &public).err(),
        Some(parameter_error(RC_INTEGRITY, 1))
    );
    // So does one loaded under another parent.
    let (other, _) = client.create_primary(RH_ENDORSEMENT, &ecc_storage_template());
    assert_eq!(
        load(&mut client, other, &private, &public).err(),
        Some(parameter_error(RC_INTEGRITY, 1))
    );
}

#[test]
fn an_hmac_session_authorizes_with_the_value_mixed_into_its_key() {
    let mut client = Client::new(store());
    let (parent, _) = client.create_primary(RH_OWNER, &ecc_storage_template());
    let template = sealed_template(SEALED_ATTRIBUTES | OBJECT_USER_WITH_AUTH, &[]);
    let (private, public) = seal(&mut client, parent, b"pin", b"secret", &template);
    let item = load(&mut client, parent, &private, &public).unwrap();
    let mut session = client.start_session(RH_NULL, SE_HMAC, SESSION_CONTINUE);
    session.entity_auth = b"pin".to_vec();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut session)],
        &[],
        0,
    );
    assert_eq!(unsealed(reply).unwrap(), b"secret");
    // The session continues, with the TPM's new nonce.
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut session)],
        &[],
        0,
    );
    assert_eq!(unsealed(reply).unwrap(), b"secret");
    session.entity_auth = b"wrong".to_vec();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut session)],
        &[],
        0,
    );
    assert_eq!(reply.err(), Some(session_error(RC_AUTH_FAIL, 1)));
}

/// ESYS's pattern: a password authorizes, a second salted session encrypts both ways.
fn unseal_through_salted_session(salt_template: &[u8]) {
    let mut client = Client::new(store());
    let (parent, _) = client.create_primary(RH_OWNER, salt_template);
    let template = sealed_template(SEALED_ATTRIBUTES | OBJECT_USER_WITH_AUTH, &[]);
    let mut encryption = client.start_session(
        parent,
        SE_HMAC,
        SESSION_CONTINUE | SESSION_DECRYPT | SESSION_ENCRYPT,
    );
    assert!(!encryption.key.is_empty());
    // Create with the secret encrypted on the way in.
    let reply = client
        .call(
            CC_CREATE,
            &[parent],
            &mut [Auth::Password(&[]), Auth::Session(&mut encryption)],
            &create_parameters(&[], b"encrypted in transit", &template),
            0,
        )
        .unwrap();
    let mut reader = Reader::new(&reply.parameters);
    let (private, public) = (
        reader.sized().unwrap().to_vec(),
        reader.sized().unwrap().to_vec(),
    );
    let item = load(&mut client, parent, &private, &public).unwrap();
    // Unseal has no parameter to decrypt, and the TPM refuses a decrypt session for it (ESYS
    // drops the attribute itself).
    encryption.attributes = SESSION_CONTINUE | SESSION_ENCRYPT;
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Password(&[]), Auth::Session(&mut encryption)],
        &[],
        0,
    );
    assert_eq!(unsealed(reply).unwrap(), b"encrypted in transit");
}

#[test]
fn an_ecc_salted_session_encrypts_parameters_both_ways() {
    unseal_through_salted_session(&ecc_storage_template());
}

#[test]
fn an_rsa_salted_session_encrypts_parameters_both_ways() {
    unseal_through_salted_session(&rsa_storage_template());
}

#[test]
fn a_salt_the_key_cannot_decrypt_is_refused() {
    let mut client = Client::new(store());
    let (parent, _) = client.create_primary(RH_OWNER, &ecc_storage_template());
    let mut parameters = Writer::new();
    parameters.sized(&[7; DIGEST_BYTES]);
    // A point off the curve.
    parameters.sized_with(|writer| {
        writer.sized(&[1; 32]);
        writer.sized(&[2; 32]);
    });
    parameters.u8(SE_HMAC);
    parameters.u16(ALG_NULL);
    parameters.u16(ALG_SHA256);
    let reply = client.call(
        CC_START_AUTH_SESSION,
        &[parent, RH_NULL],
        &mut [],
        &parameters.bytes,
        1,
    );
    assert!(reply.is_err());
}

/// What systemd-cryptenroll enrolls: a secret sealed to PCR 7 and a PIN.
fn pcr_and_pin_policy(pcr_value: &[u8]) -> Vec<u8> {
    let after_pcr = sha256(&[
        &[0; DIGEST_BYTES],
        &CC_POLICY_PCR.to_be_bytes(),
        &pcr_selection(PCR_SECURE_BOOT),
        &sha256(&[pcr_value]),
    ]);
    sha256(&[&after_pcr, &CC_POLICY_AUTH_VALUE.to_be_bytes()])
}

#[test]
fn a_policy_sealed_secret_unseals_only_while_the_pcr_holds() {
    let mut client = Client::new(store());
    client.extend(PCR_SECURE_BOOT, &[0x77; DIGEST_BYTES]);
    let pcr = client.read_pcr(PCR_SECURE_BOOT);
    let expected = pcr_and_pin_policy(&pcr);
    // A trial session computes the same digest.
    let trial = client.start_session(RH_NULL, SE_TRIAL, SESSION_CONTINUE);
    client.policy_pcr(trial.handle, PCR_SECURE_BOOT).unwrap();
    client
        .call(CC_POLICY_AUTH_VALUE, &[trial.handle], &mut [], &[], 0)
        .unwrap();
    assert_eq!(client.policy_digest(trial.handle), expected);
    client.flush(trial.handle);

    let (parent, _) = client.create_primary(RH_OWNER, &ecc_storage_template());
    let template = sealed_template(SEALED_ATTRIBUTES, &expected);
    let (private, public) = seal(&mut client, parent, b"1234", b"luks key", &template);
    let item = load(&mut client, parent, &private, &public).unwrap();
    // userWithAuth is clear: a password cannot stand in for the policy.
    let refused = client.call(CC_UNSEAL, &[item], &mut [Auth::Password(b"1234")], &[], 0);
    assert_eq!(refused.err(), Some(session_error(RC_AUTH_UNAVAILABLE, 1)));

    let mut policy = client.start_session(parent, SE_POLICY, SESSION_CONTINUE | SESSION_ENCRYPT);
    client.policy_pcr(policy.handle, PCR_SECURE_BOOT).unwrap();
    client
        .call(CC_POLICY_AUTH_VALUE, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    policy.entity_auth = b"1234".to_vec();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut policy)],
        &[],
        0,
    );
    assert_eq!(unsealed(reply).unwrap(), b"luks key");
    // Having authorized, the session started over: its digest is empty again.
    assert_eq!(client.policy_digest(policy.handle), vec![0; DIGEST_BYTES]);

    // A wrong PIN fails.
    client.policy_pcr(policy.handle, PCR_SECURE_BOOT).unwrap();
    client
        .call(CC_POLICY_AUTH_VALUE, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    policy.entity_auth = b"4321".to_vec();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut policy)],
        &[],
        0,
    );
    assert_eq!(reply.err(), Some(session_error(RC_AUTH_FAIL, 1)));

    // A PCR that moved since PolicyPCR fails the authorization.
    client
        .call(CC_POLICY_RESTART, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    client.policy_pcr(policy.handle, PCR_SECURE_BOOT).unwrap();
    client
        .call(CC_POLICY_AUTH_VALUE, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    client.extend(PCR_SECURE_BOOT, &[0x88; DIGEST_BYTES]);
    policy.entity_auth = b"1234".to_vec();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut policy)],
        &[],
        0,
    );
    assert_eq!(reply.err(), Some(RC_PCR_CHANGED));

    // And a PCR that differs from the sealed value gives another digest.
    client
        .call(CC_POLICY_RESTART, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    client.policy_pcr(policy.handle, PCR_SECURE_BOOT).unwrap();
    client
        .call(CC_POLICY_AUTH_VALUE, &[policy.handle], &mut [], &[], 0)
        .unwrap();
    let reply = client.call(
        CC_UNSEAL,
        &[item],
        &mut [Auth::Session(&mut policy)],
        &[],
        0,
    );
    assert_eq!(reply.err(), Some(session_error(RC_POLICY_FAIL, 1)));
}

#[test]
fn policy_password_takes_the_value_in_the_clear() {
    let mut client = Client::new(store());
    let policy_digest = sha256(&[&[0; DIGEST_BYTES], &CC_POLICY_AUTH_VALUE.to_be_bytes()]);
    let (parent, _) = client.create_primary(RH_OWNER, &ecc_storage_template());
    let template = sealed_template(SEALED_ATTRIBUTES, &policy_digest);
    let (private, public) = seal(&mut client, parent, b"pw", b"data", &template);
    let item = load(&mut client, parent, &private, &public).unwrap();
    let session = client.start_session(RH_NULL, SE_POLICY, SESSION_CONTINUE);
    client
        .call(CC_POLICY_PASSWORD, &[session.handle], &mut [], &[], 0)
        .unwrap();
    assert_eq!(client.policy_digest(session.handle), policy_digest);
    let reply = client.call(CC_UNSEAL, &[item], &mut [Auth::Password(b"pw")], &[], 0);
    // userWithAuth is clear, so the password session is refused...
    assert!(reply.is_err());
    // ...and the policy session with the password in its HMAC field works.
    let mut writer = Writer::new();
    writer.u16(ST_SESSIONS);
    writer.u32(0);
    writer.u32(CC_UNSEAL);
    writer.u32(item);
    let area = marshal(|area| {
        area.u32(session.handle);
        area.sized(&[3; DIGEST_BYTES]);
        area.u8(SESSION_CONTINUE);
        area.sized(b"pw");
    });
    writer.u32(area.len() as u32);
    writer.bytes(&area);
    let length = writer.bytes.len() as u32;
    writer.bytes[2..6].copy_from_slice(&length.to_be_bytes());
    let response = client.tpm.execute(&writer.bytes, 0);
    assert_eq!(
        u32::from_be_bytes(response[6..10].try_into().unwrap()),
        RC_SUCCESS
    );
}

#[test]
fn contexts_save_and_load_objects_and_sessions() {
    let mut client = Client::new(store());
    let (primary, public) = client.create_primary(RH_NULL, &ecc_storage_template());
    let reply = client
        .call(CC_CONTEXT_SAVE, &[primary], &mut [], &[], 0)
        .unwrap();
    let context = reply.parameters;
    client.flush(primary);
    assert!(client.tpm.object(primary).is_none());
    let loaded = client
        .call(CC_CONTEXT_LOAD, &[], &mut [], &context, 1)
        .unwrap()
        .handles[0];
    assert_eq!(client.tpm.object(loaded).unwrap().public, public);
    // An object context loads more than once.
    let again = client
        .call(CC_CONTEXT_LOAD, &[], &mut [], &context, 1)
        .unwrap()
        .handles[0];
    assert_ne!(again, loaded);
    let mut tampered = context.clone();
    *tampered.last_mut().unwrap() ^= 1;
    assert_eq!(
        client
            .call(CC_CONTEXT_LOAD, &[], &mut [], &tampered, 1)
            .err(),
        Some(parameter_error(RC_INTEGRITY, 1))
    );

    // A session keeps its handle; while saved it cannot be used, and only its latest context
    // loads.
    let mut session = client.start_session(loaded, SE_HMAC, SESSION_CONTINUE | SESSION_ENCRYPT);
    let first = client
        .call(CC_CONTEXT_SAVE, &[session.handle], &mut [], &[], 0)
        .unwrap()
        .parameters;
    let random = client.call(
        CC_GET_RANDOM,
        &[],
        &mut [Auth::Session(&mut session)],
        &8u16.to_be_bytes(),
        0,
    );
    assert_eq!(random.err(), Some(RC_REFERENCE_S0));
    let handle = client
        .call(CC_CONTEXT_LOAD, &[], &mut [], &first, 1)
        .unwrap()
        .handles[0];
    assert_eq!(handle, session.handle);
    assert!(
        client
            .call(
                CC_GET_RANDOM,
                &[],
                &mut [Auth::Session(&mut session)],
                &8u16.to_be_bytes(),
                0
            )
            .is_ok()
    );
    client
        .call(CC_CONTEXT_SAVE, &[session.handle], &mut [], &[], 0)
        .unwrap();
    assert_eq!(
        client.call(CC_CONTEXT_LOAD, &[], &mut [], &first, 1).err(),
        Some(parameter_error(RC_HANDLE, 1))
    );
    // A saved session can be flushed.
    client.flush(session.handle);
    assert!(client.tpm.session(session.handle).is_none());
}

#[test]
fn the_linux_hmac_session_sequence_works() {
    // drivers/char/tpm/tpm2-sessions.c: the null primary, saved and flushed, loaded for each
    // session start, salted by ECDH; then commands with an HMAC session.
    let mut client = Client::new(store());
    let (null_key, _) = client.create_primary(RH_NULL, &ecc_storage_template());
    let context = client
        .call(CC_CONTEXT_SAVE, &[null_key], &mut [], &[], 0)
        .unwrap()
        .parameters;
    client.flush(null_key);
    let null_key = client
        .call(CC_CONTEXT_LOAD, &[], &mut [], &context, 1)
        .unwrap()
        .handles[0];
    let mut session = client.start_session(null_key, SE_HMAC, SESSION_CONTINUE | SESSION_ENCRYPT);
    client.flush(null_key);
    let reply = client
        .call(
            CC_GET_RANDOM,
            &[],
            &mut [Auth::Session(&mut session)],
            &32u16.to_be_bytes(),
            0,
        )
        .unwrap();
    assert_eq!(Reader::new(&reply.parameters).sized().unwrap().len(), 32);
    // PCR_Extend authorized by the session (the PCR's value is empty).
    session.attributes = SESSION_CONTINUE;
    let mut parameters = Writer::new();
    parameters.u32(1);
    parameters.u16(ALG_SHA256);
    parameters.bytes(&[0x42; DIGEST_BYTES]);
    let before = client.read_pcr(10);
    client
        .call(
            CC_PCR_EXTEND,
            &[10],
            &mut [Auth::Session(&mut session)],
            &parameters.bytes,
            0,
        )
        .unwrap();
    assert_eq!(
        client.read_pcr(10),
        sha256(&[&before, &[0x42; DIGEST_BYTES]])
    );
    // Ending the session: continueSession clear flushes it after the command.
    session.attributes = SESSION_ENCRYPT;
    client
        .call(
            CC_GET_RANDOM,
            &[],
            &mut [Auth::Session(&mut session)],
            &8u16.to_be_bytes(),
            0,
        )
        .unwrap();
    assert!(client.tpm.session(session.handle).is_none());
}

#[test]
fn a_session_that_neither_authorizes_nor_encrypts_is_refused() {
    let mut client = Client::new(store());
    let mut session = client.start_session(RH_NULL, SE_HMAC, SESSION_CONTINUE);
    let reply = client.call(
        CC_GET_RANDOM,
        &[],
        &mut [Auth::Session(&mut session)],
        &8u16.to_be_bytes(),
        0,
    );
    assert_eq!(reply.err(), Some(session_error(RC_ATTRIBUTES, 1)));
}

#[test]
fn persistent_objects_survive_a_new_tpm_on_the_same_store() {
    let store = store();
    let mut client = Client::new(store.clone());
    let (primary, public) = client.create_primary(RH_OWNER, &ecc_storage_template());
    client
        .call(
            CC_EVICT_CONTROL,
            &[RH_OWNER, primary],
            &mut [Auth::Password(&[])],
            &SRK_HANDLE.to_be_bytes(),
            0,
        )
        .unwrap();
    client.flush(primary);
    let template = sealed_template(SEALED_ATTRIBUTES | OBJECT_USER_WITH_AUTH, &[]);
    let (private, sealed_public) = seal(&mut client, SRK_HANDLE, b"", b"kept", &template);

    // A guest reset: a new device on the same store.
    let mut client = Client::new(store);
    let reply = client
        .call(CC_READ_PUBLIC, &[SRK_HANDLE], &mut [], &[], 0)
        .unwrap();
    assert_eq!(
        Public::read_sized(&mut Reader::new(&reply.parameters), 1).unwrap(),
        public
    );
    let item = load(&mut client, SRK_HANDLE, &private, &sealed_public).unwrap();
    let reply = client.call(CC_UNSEAL, &[item], &mut [Auth::Password(&[])], &[], 0);
    assert_eq!(unsealed(reply).unwrap(), b"kept");
    // The same template under the storage seed is still the same key.
    let (again, recreated) = client.create_primary(RH_OWNER, &ecc_storage_template());
    assert_eq!(recreated, public);
    client.flush(again);

    // Evicting removes it, from the store too.
    client
        .call(
            CC_EVICT_CONTROL,
            &[RH_OWNER, SRK_HANDLE],
            &mut [Auth::Password(&[])],
            &SRK_HANDLE.to_be_bytes(),
            0,
        )
        .unwrap();
    assert!(
        client
            .call(CC_READ_PUBLIC, &[SRK_HANDLE], &mut [], &[], 0)
            .is_err()
    );
    let store = client.tpm.store.clone();
    let client = Client::new(store);
    assert!(client.tpm.object(SRK_HANDLE).is_none());
}

#[test]
fn null_hierarchy_objects_cannot_be_persisted() {
    let mut client = Client::new(store());
    let (primary, _) = client.create_primary(RH_NULL, &ecc_storage_template());
    let reply = client.call(
        CC_EVICT_CONTROL,
        &[RH_OWNER, primary],
        &mut [Auth::Password(&[])],
        &SRK_HANDLE.to_be_bytes(),
        0,
    );
    assert_eq!(reply.err(), Some(handle_error(RC_HIERARCHY, 2)));
}

fn define_nv(client: &mut Client, attributes: u32, auth: &[u8], size: u16) -> Result<Reply, u32> {
    let parameters = marshal(|writer| {
        writer.sized(auth);
        writer.sized_with(|writer| {
            writer.u32(NV_INDEX);
            writer.u16(ALG_SHA256);
            writer.u32(attributes);
            writer.sized(&[]);
            writer.u16(size);
        });
    });
    client.call(
        CC_NV_DEFINE_SPACE,
        &[RH_OWNER],
        &mut [Auth::Password(&[])],
        &parameters,
        0,
    )
}

fn nv_write(
    client: &mut Client,
    auth_handle: u32,
    password: &[u8],
    data: &[u8],
    offset: u16,
) -> Result<Reply, u32> {
    let parameters = marshal(|writer| {
        writer.sized(data);
        writer.u16(offset);
    });
    client.call(
        CC_NV_WRITE,
        &[auth_handle, NV_INDEX],
        &mut [Auth::Password(password)],
        &parameters,
        0,
    )
}

fn nv_read(
    client: &mut Client,
    auth_handle: u32,
    password: &[u8],
    size: u16,
    offset: u16,
) -> Result<Vec<u8>, u32> {
    let parameters = marshal(|writer| {
        writer.u16(size);
        writer.u16(offset);
    });
    let reply = client.call(
        CC_NV_READ,
        &[auth_handle, NV_INDEX],
        &mut [Auth::Password(password)],
        &parameters,
        0,
    )?;
    Ok(Reader::new(&reply.parameters).sized().unwrap().to_vec())
}

#[test]
fn an_nv_index_is_defined_written_read_and_kept() {
    use super::nv::{NV_AUTHREAD, NV_AUTHWRITE, NV_OWNERREAD};
    let store = store();
    let mut client = Client::new(store.clone());
    define_nv(&mut client, NV_AUTHWRITE | NV_AUTHREAD, b"nvpass", 16).unwrap();
    assert_eq!(
        define_nv(&mut client, NV_AUTHWRITE | NV_AUTHREAD, b"", 16).err(),
        Some(RC_NV_DEFINED)
    );
    let name_before = client.name(NV_INDEX);
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"nvpass", 4, 0).err(),
        Some(RC_NV_UNINITIALIZED)
    );
    nv_write(&mut client, NV_INDEX, b"nvpass", b"abcd", 4).unwrap();
    // TPMA_NV_WRITTEN is part of the name.
    assert_ne!(client.name(NV_INDEX), name_before);
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"nvpass", 4, 4).unwrap(),
        b"abcd"
    );
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"wrong", 4, 4).err(),
        Some(session_error(RC_AUTH_FAIL, 1))
    );
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"nvpass", 4, 14).err(),
        Some(RC_NV_RANGE)
    );
    // The owner may not read it: OWNERREAD is clear.
    assert_eq!(
        nv_read(&mut client, RH_OWNER, b"", 4, 4).err(),
        Some(RC_NV_AUTHORIZATION)
    );
    let reply = client
        .call(CC_NV_READ_PUBLIC, &[NV_INDEX], &mut [], &[], 0)
        .unwrap();
    let mut reader = Reader::new(&reply.parameters);
    let public = reader.sized().unwrap().to_vec();
    assert_eq!(reader.sized().unwrap(), client.name(NV_INDEX));
    assert_eq!(&public[..4], &NV_INDEX.to_be_bytes());

    // A reset keeps it.
    let mut client = Client::new(store);
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"nvpass", 4, 4).unwrap(),
        b"abcd"
    );
    client
        .call(
            CC_NV_UNDEFINE_SPACE,
            &[RH_OWNER, NV_INDEX],
            &mut [Auth::Password(&[])],
            &[],
            0,
        )
        .unwrap();
    assert_eq!(
        nv_read(&mut client, NV_INDEX, b"nvpass", 4, 4).err(),
        Some(handle_error(RC_HANDLE, 1))
    );
    // An index the owner may read.
    define_nv(&mut client, NV_AUTHWRITE | NV_OWNERREAD, b"", 8).unwrap();
    nv_write(&mut client, NV_INDEX, b"", b"12345678", 0).unwrap();
    assert_eq!(
        nv_read(&mut client, RH_OWNER, b"", 8, 0).unwrap(),
        b"12345678"
    );
}

#[test]
fn hierarchy_auth_changes_persist_and_clear_resets_the_owner() {
    use super::nv::{NV_AUTHREAD, NV_AUTHWRITE};
    let store = store();
    let mut client = Client::new(store.clone());
    let (primary, before) = client.create_primary(RH_OWNER, &ecc_storage_template());
    client
        .call(
            CC_EVICT_CONTROL,
            &[RH_OWNER, primary],
            &mut [Auth::Password(&[])],
            &SRK_HANDLE.to_be_bytes(),
            0,
        )
        .unwrap();
    define_nv(&mut client, NV_AUTHWRITE | NV_AUTHREAD, b"", 8).unwrap();
    let new_auth = marshal(|writer| writer.sized(b"owner"));
    client
        .call(
            CC_HIERARCHY_CHANGE_AUTH,
            &[RH_OWNER],
            &mut [Auth::Password(&[])],
            &new_auth,
            0,
        )
        .unwrap();

    let mut client = Client::new(store.clone());
    let refused = client.call(
        CC_CREATE_PRIMARY,
        &[RH_OWNER],
        &mut [Auth::Password(&[])],
        &create_parameters(&[], &[], &ecc_storage_template()),
        1,
    );
    assert_eq!(refused.err(), Some(session_error(RC_AUTH_FAIL, 1)));
    assert!(
        client
            .call(
                CC_CREATE_PRIMARY,
                &[RH_OWNER],
                &mut [Auth::Password(b"owner")],
                &create_parameters(&[], &[], &ecc_storage_template()),
                1,
            )
            .is_ok()
    );

    client
        .call(CC_CLEAR, &[RH_LOCKOUT], &mut [Auth::Password(&[])], &[], 0)
        .unwrap();
    assert!(client.tpm.object(SRK_HANDLE).is_none());
    assert!(client.tpm.nv_index(NV_INDEX).is_none());
    // The owner's value is empty again and the storage seed is new.
    let (_, after) = client.create_primary(RH_OWNER, &ecc_storage_template());
    assert_ne!(after, before);
    let client = Client::new(store);
    assert!(client.tpm.object(SRK_HANDLE).is_none());
}

#[test]
fn the_response_hmac_after_hierarchy_change_auth_uses_the_new_value() {
    let mut client = Client::new(store());
    let mut session = client.start_session(RH_NULL, SE_HMAC, SESSION_CONTINUE);
    let new_auth = marshal(|writer| writer.sized(b"changed"));
    // The client checks the response HMAC with the value it set, as ESYS does.
    let nonce = client.fresh_nonce();
    let cp_hash = sha256(&[
        &CC_HIERARCHY_CHANGE_AUTH.to_be_bytes(),
        &RH_OWNER.to_be_bytes(),
        &new_auth,
    ]);
    let mac = hmac(
        &[],
        &[&cp_hash, &nonce, &session.nonce_tpm, &[SESSION_CONTINUE]],
    );
    let command = marshal(|writer| {
        writer.u16(ST_SESSIONS);
        writer.u32(0);
        writer.u32(CC_HIERARCHY_CHANGE_AUTH);
        writer.u32(RH_OWNER);
        let area = marshal(|area| {
            area.u32(session.handle);
            area.sized(&nonce);
            area.u8(SESSION_CONTINUE);
            area.sized(&mac);
        });
        writer.u32(area.len() as u32);
        writer.bytes(&area);
        writer.bytes(&new_auth);
    });
    let mut command = command;
    let length = command.len() as u32;
    command[2..6].copy_from_slice(&length.to_be_bytes());
    let response = client.tpm.execute(&command, 0);
    let mut reader = Reader::new(&response);
    reader.take(6).unwrap();
    assert_eq!(reader.u32().unwrap(), RC_SUCCESS);
    let size = reader.u32().unwrap() as usize;
    let parameters = reader.take(size).unwrap().to_vec();
    let nonce_tpm = reader.sized().unwrap().to_vec();
    let attributes = reader.u8().unwrap();
    let response_mac = reader.sized().unwrap();
    let rp_hash = sha256(&[
        &RC_SUCCESS.to_be_bytes(),
        &CC_HIERARCHY_CHANGE_AUTH.to_be_bytes(),
        &parameters,
    ]);
    assert_eq!(
        response_mac,
        hmac(b"changed", &[&rp_hash, &nonce_tpm, &nonce, &[attributes]])
    );
    session.nonce_tpm = nonce_tpm;
}

#[test]
fn ecdh_key_gen_and_z_gen_agree() {
    let mut client = Client::new(store());
    let template = marshal(|writer| {
        writer.u16(ALG_ECC);
        writer.u16(ALG_SHA256);
        writer.u32(
            OBJECT_FIXED_TPM
                | OBJECT_FIXED_PARENT
                | OBJECT_SENSITIVE_DATA_ORIGIN
                | OBJECT_USER_WITH_AUTH
                | OBJECT_DECRYPT,
        );
        writer.sized(&[]);
        writer.u16(ALG_NULL);
        writer.u16(ALG_ECDH);
        writer.u16(ALG_SHA256);
        writer.u16(ECC_NIST_P256);
        writer.u16(ALG_NULL);
        writer.sized(&[]);
        writer.sized(&[]);
    });
    let (key, _) = client.create_primary(RH_OWNER, &template);
    let reply = client
        .call(CC_ECDH_KEYGEN, &[key], &mut [], &[], 0)
        .unwrap();
    let mut reader = Reader::new(&reply.parameters);
    let z_point = reader.sized().unwrap().to_vec();
    let public_point = reader.sized().unwrap().to_vec();
    let reply = client
        .call(
            CC_ECDH_ZGEN,
            &[key],
            &mut [Auth::Password(&[])],
            &marshal(|writer| writer.sized(&public_point)),
            0,
        )
        .unwrap();
    assert_eq!(Reader::new(&reply.parameters).sized().unwrap(), z_point);
}

#[test]
fn capabilities_list_the_curve_and_the_new_algorithms() {
    use super::hash::HASH_ALGORITHMS;
    let mut client = Client::new(store());
    let capability = |client: &mut Client, capability: u32| {
        let parameters = marshal(|writer| {
            writer.u32(capability);
            writer.u32(0);
            writer.u32(256);
        });
        client
            .call(CC_GET_CAPABILITY, &[], &mut [], &parameters, 0)
            .unwrap()
            .parameters
    };
    const CAP_ALGS: u32 = 0;
    const CAP_ECC_CURVES: u32 = 8;
    let algorithms = capability(&mut client, CAP_ALGS);
    let mut reader = Reader::new(&algorithms[5..]);
    let count = reader.u32().unwrap();
    let ids: Vec<u16> = (0..count)
        .map(|_| {
            let id = reader.u16().unwrap();
            reader.u32().unwrap();
            id
        })
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    for id in [
        ALG_RSA,
        ALG_ECC,
        ALG_AES,
        ALG_CFB,
        ALG_KEYEDHASH,
        ALG_OAEP,
        ALG_ECDH,
    ] {
        assert!(ids.contains(&id), "{id:#x}");
    }
    assert!(HASH_ALGORITHMS.iter().all(|hash| ids.contains(hash)));
    let curves = capability(&mut client, CAP_ECC_CURVES);
    assert_eq!(&curves[5..], &[0, 0, 0, 1, 0, ECC_NIST_P256 as u8]);
}
