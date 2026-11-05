//! Objects (Part 1, "Object Structure Elements"; Part 3, "Object Commands"): RSA 2048 and ECC
//! P-256 keys and sealed data, their public and sensitive areas, how a child's sensitive area is
//! protected under its parent, and `TPM2_CreatePrimary`, `TPM2_Create`, `TPM2_Load`,
//! `TPM2_ReadPublic`, `TPM2_Unseal`, `TPM2_EvictControl`, `TPM2_TestParms` and the two ECDH
//! commands.
//!
//! What can be made: storage parents (restricted decrypt RSA or ECC keys with an AES-CFB
//! symmetric definition), unrestricted decrypt keys, and sealed data objects (`KEYEDHASH`
//! without sign or decrypt). Signing keys, HMAC keys, symmetric-cipher objects and derivation
//! parents are refused: no command here could use them.
#[cfg(freestanding)]
use crate::prelude::*;

use super::engine::{Response, Tpm, end_of};
use super::hash::{self, KdfStream};
use super::spec::*;
use super::wire::{Reader, Truncated, Writer, marshal};
use uemulator_crypto::aes;
use uemulator_crypto::ecc::{self, EccPoint};
use uemulator_crypto::rsa::RsaPrivateKey;
#[cfg(test)]
use uemulator_crypto::rsa::RsaPublicKey;

const RC_BINDING: u32 = 0x0a5;

/// The one RSA size: what systemd, Linux and Windows ask for.
pub(super) const RSA_KEY_BITS: u16 = 2048;
pub(super) const RSA_DEFAULT_EXPONENT: u32 = 65537;
const P256_BYTES: usize = ecc::P256_COORDINATE_BYTES;
/// `MAX_SYM_DATA`: the most a sealed object can hold.
pub(super) const MAX_SEALED_BYTES: usize = 128;
/// The AES key sizes a storage parent may name.
const AES_KEY_BITS: [u16; 2] = [128, 256];
pub(super) const AES_BLOCK_BYTES: usize = 16;
/// The protection of a child's sensitive area uses a zero IV: its key is unique to the child's
/// name (Part 1, "Symmetric Encryption" of the storage hierarchy).
const ZERO_IV: [u8; AES_BLOCK_BYTES] = [0; AES_BLOCK_BYTES];
const LABEL_STORAGE: &[u8] = b"STORAGE";
const LABEL_INTEGRITY: &[u8] = b"INTEGRITY";
const LABEL_PRIMARY: &[u8] = b"Primary Object Creation";
/// How many candidates a P-256 private key may take before key generation gives up: each
/// fails with probability about 2^-32, so this is never reached.
const ECC_KEY_ATTEMPTS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct SymmetricDefinition {
    pub algorithm: u16,
    /// Bits of the key, or for `TPM_ALG_XOR` the hash.
    pub key_bits: u16,
    pub mode: u16,
}

impl SymmetricDefinition {
    pub const NULL: SymmetricDefinition = SymmetricDefinition {
        algorithm: hash::ALG_NULL,
        key_bits: 0,
        mode: 0,
    };

    /// A `TPMT_SYM_DEF_OBJECT` or a `TPMT_SYM_DEF` (the session form also allows `XOR`, which has
    /// a hash instead of key bits and no mode).
    pub fn read(reader: &mut Reader) -> Result<Self, Truncated> {
        let algorithm = reader.u16()?;
        Ok(match algorithm {
            hash::ALG_NULL => Self::NULL,
            ALG_XOR => SymmetricDefinition {
                algorithm,
                key_bits: reader.u16()?,
                mode: 0,
            },
            _ => SymmetricDefinition {
                algorithm,
                key_bits: reader.u16()?,
                mode: reader.u16()?,
            },
        })
    }

    pub fn write(&self, writer: &mut Writer) {
        writer.u16(self.algorithm);
        match self.algorithm {
            hash::ALG_NULL => {}
            ALG_XOR => writer.u16(self.key_bits),
            _ => {
                writer.u16(self.key_bits);
                writer.u16(self.mode);
            }
        }
    }

    pub fn is_aes_cfb(&self) -> bool {
        self.algorithm == ALG_AES && self.mode == ALG_CFB && AES_KEY_BITS.contains(&self.key_bits)
    }

    pub fn key_bytes(&self) -> usize {
        self.key_bits as usize / 8
    }
}

/// A `TPMT_*_SCHEME` or `TPMT_KDF_SCHEME`: the scheme and whatever details it carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Scheme {
    pub scheme: u16,
    pub hash: u16,
    /// ECDAA's count, or the KDF of the keyed-hash `XOR` scheme.
    pub extra: u16,
}

impl Scheme {
    const NULL: Scheme = Scheme {
        scheme: hash::ALG_NULL,
        hash: 0,
        extra: 0,
    };

    fn read(reader: &mut Reader) -> Result<Self, Truncated> {
        let scheme = reader.u16()?;
        let (hash, extra) = match scheme {
            hash::ALG_NULL | ALG_RSAES => (0, 0),
            ALG_ECDAA | ALG_XOR => (reader.u16()?, reader.u16()?),
            _ => (reader.u16()?, 0),
        };
        Ok(Scheme {
            scheme,
            hash,
            extra,
        })
    }

    fn write(&self, writer: &mut Writer) {
        writer.u16(self.scheme);
        match self.scheme {
            hash::ALG_NULL | ALG_RSAES => {}
            ALG_ECDAA | ALG_XOR => {
                writer.u16(self.hash);
                writer.u16(self.extra);
            }
            _ => writer.u16(self.hash),
        }
    }

    fn is_null(&self) -> bool {
        self.scheme == hash::ALG_NULL
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Parameters {
    KeyedHash {
        scheme: Scheme,
    },
    SymCipher {
        symmetric: SymmetricDefinition,
    },
    Rsa {
        symmetric: SymmetricDefinition,
        scheme: Scheme,
        key_bits: u16,
        exponent: u32,
    },
    Ecc {
        symmetric: SymmetricDefinition,
        scheme: Scheme,
        curve: u16,
        kdf: Scheme,
    },
}

impl Parameters {
    /// A `TPMU_PUBLIC_PARMS` of `object_type`. `None` is a type this TPM does not know.
    pub fn read(object_type: u16, reader: &mut Reader) -> Result<Option<Self>, Truncated> {
        Ok(Some(match object_type {
            ALG_KEYEDHASH => Parameters::KeyedHash {
                scheme: Scheme::read(reader)?,
            },
            ALG_SYMCIPHER => Parameters::SymCipher {
                symmetric: SymmetricDefinition::read(reader)?,
            },
            ALG_RSA => Parameters::Rsa {
                symmetric: SymmetricDefinition::read(reader)?,
                scheme: Scheme::read(reader)?,
                key_bits: reader.u16()?,
                exponent: reader.u32()?,
            },
            ALG_ECC => Parameters::Ecc {
                symmetric: SymmetricDefinition::read(reader)?,
                scheme: Scheme::read(reader)?,
                curve: reader.u16()?,
                kdf: Scheme::read(reader)?,
            },
            _ => return Ok(None),
        }))
    }

    fn write(&self, writer: &mut Writer) {
        match self {
            Parameters::KeyedHash { scheme } => scheme.write(writer),
            Parameters::SymCipher { symmetric } => symmetric.write(writer),
            Parameters::Rsa {
                symmetric,
                scheme,
                key_bits,
                exponent,
            } => {
                symmetric.write(writer);
                scheme.write(writer);
                writer.u16(*key_bits);
                writer.u32(*exponent);
            }
            Parameters::Ecc {
                symmetric,
                scheme,
                curve,
                kdf,
            } => {
                symmetric.write(writer);
                scheme.write(writer);
                writer.u16(*curve);
                kdf.write(writer);
            }
        }
    }

    pub fn symmetric(&self) -> SymmetricDefinition {
        match self {
            Parameters::SymCipher { symmetric }
            | Parameters::Rsa { symmetric, .. }
            | Parameters::Ecc { symmetric, .. } => *symmetric,
            Parameters::KeyedHash { .. } => SymmetricDefinition::NULL,
        }
    }

    /// The checks `TPM2_TestParms` makes: an algorithm and size this TPM implements. The
    /// combination with object attributes is `check_template`'s.
    pub fn check(&self) -> Result<(), u32> {
        let symmetric = self.symmetric();
        if symmetric.algorithm != hash::ALG_NULL && !symmetric.is_aes_cfb() {
            return Err(RC_SYMMETRIC);
        }
        match self {
            Parameters::KeyedHash { scheme } => {
                if !scheme.is_null() {
                    return Err(RC_SCHEME);
                }
            }
            Parameters::SymCipher { .. } => return Err(RC_TYPE),
            Parameters::Rsa {
                scheme,
                key_bits,
                exponent,
                ..
            } => {
                if *key_bits != RSA_KEY_BITS {
                    return Err(RC_KEY_SIZE);
                }
                if *exponent != 0 && *exponent != RSA_DEFAULT_EXPONENT {
                    return Err(RC_VALUE);
                }
                match scheme.scheme {
                    hash::ALG_NULL | ALG_RSAES => {}
                    ALG_OAEP if hash::digest_len(scheme.hash).is_some() => {}
                    _ => return Err(RC_SCHEME),
                }
            }
            Parameters::Ecc {
                scheme, curve, kdf, ..
            } => {
                if *curve != ECC_NIST_P256 {
                    return Err(RC_CURVE);
                }
                match scheme.scheme {
                    hash::ALG_NULL => {}
                    ALG_ECDH if hash::digest_len(scheme.hash).is_some() => {}
                    _ => return Err(RC_SCHEME),
                }
                if !kdf.is_null() {
                    return Err(RC_KDF);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Unique {
    /// `KEYEDHASH` and `SYMCIPHER`: a digest binding the public area to the secret.
    Digest(Vec<u8>),
    Rsa(Vec<u8>),
    Ecc {
        x: Vec<u8>,
        y: Vec<u8>,
    },
}

/// A `TPMT_PUBLIC`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Public {
    pub object_type: u16,
    pub name_alg: u16,
    pub attributes: u32,
    pub auth_policy: Vec<u8>,
    pub parameters: Parameters,
    pub unique: Unique,
}

impl Public {
    /// A `TPM2B_PUBLIC`, which must hold exactly one `TPMT_PUBLIC`. `number` is the parameter's
    /// position, for the response code.
    pub fn read_sized(reader: &mut Reader, number: u32) -> Result<Self, u32> {
        let bytes = reader.sized()?;
        if bytes.is_empty() {
            return Err(parameter_error(RC_SIZE, number));
        }
        let mut inner = Reader::new(bytes);
        let public = Self::read(&mut inner, number)?;
        if inner.remaining() != 0 {
            return Err(parameter_error(RC_SIZE, number));
        }
        Ok(public)
    }

    pub fn read(reader: &mut Reader, number: u32) -> Result<Self, u32> {
        let truncated = |_| parameter_error(RC_INSUFFICIENT, number);
        let object_type = reader.u16().map_err(truncated)?;
        let name_alg = reader.u16().map_err(truncated)?;
        let attributes = reader.u32().map_err(truncated)?;
        let auth_policy = reader.sized().map_err(truncated)?.to_vec();
        let parameters = Parameters::read(object_type, reader)
            .map_err(truncated)?
            .ok_or(parameter_error(RC_TYPE, number))?;
        let unique = match object_type {
            ALG_RSA => Unique::Rsa(reader.sized().map_err(truncated)?.to_vec()),
            ALG_ECC => Unique::Ecc {
                x: reader.sized().map_err(truncated)?.to_vec(),
                y: reader.sized().map_err(truncated)?.to_vec(),
            },
            _ => Unique::Digest(reader.sized().map_err(truncated)?.to_vec()),
        };
        Ok(Public {
            object_type,
            name_alg,
            attributes,
            auth_policy,
            parameters,
            unique,
        })
    }

    pub fn write(&self, writer: &mut Writer) {
        writer.u16(self.object_type);
        writer.u16(self.name_alg);
        writer.u32(self.attributes);
        writer.sized(&self.auth_policy);
        self.parameters.write(writer);
        match &self.unique {
            Unique::Digest(bytes) | Unique::Rsa(bytes) => writer.sized(bytes),
            Unique::Ecc { x, y } => {
                writer.sized(x);
                writer.sized(y);
            }
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        marshal(|writer| self.write(writer))
    }

    /// `nameAlg || H(TPMT_PUBLIC)` (Part 1, "Names").
    pub fn name(&self) -> Vec<u8> {
        let mut name = self.name_alg.to_be_bytes().to_vec();
        name.extend(hash::digest(self.name_alg, &[&self.bytes()]).unwrap_or_default());
        name
    }

    pub fn has(&self, attribute: u32) -> bool {
        self.attributes & attribute != 0
    }

    /// A parent of other objects: a restricted decryption key with a symmetric definition.
    pub fn is_storage_parent(&self) -> bool {
        self.has(OBJECT_RESTRICTED)
            && self.has(OBJECT_DECRYPT)
            && !self.has(OBJECT_SIGN)
            && matches!(self.object_type, ALG_RSA | ALG_ECC)
    }

    pub fn is_sealed_data(&self) -> bool {
        self.object_type == ALG_KEYEDHASH
            && !self.has(OBJECT_SIGN)
            && !self.has(OBJECT_DECRYPT)
            && !self.has(OBJECT_RESTRICTED)
    }

    pub fn ecc_point(&self) -> Option<EccPoint> {
        match &self.unique {
            Unique::Ecc { x, y } => Some(EccPoint {
                x: x.as_slice().try_into().ok()?,
                y: y.as_slice().try_into().ok()?,
            }),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn rsa_public_key(&self) -> Option<RsaPublicKey> {
        let (Unique::Rsa(modulus), Parameters::Rsa { exponent, .. }) =
            (&self.unique, &self.parameters)
        else {
            return None;
        };
        RsaPublicKey::new(modulus, &rsa_exponent(*exponent).to_be_bytes()).ok()
    }
}

fn rsa_exponent(exponent: u32) -> u32 {
    if exponent == 0 {
        RSA_DEFAULT_EXPONENT
    } else {
        exponent
    }
}

/// Checks a template against Part 1's attribute rules (the reference implementation's
/// `PublicAttributesValidation` and `SchemeChecks`, cut down to the kinds of object this TPM
/// makes). `parent` is `None` for a primary. Errors name parameter `number`.
pub(super) fn check_template(
    public: &Public,
    parent: Option<&Public>,
    number: u32,
) -> Result<(), u32> {
    let error = |code| parameter_error(code, number);
    if public.attributes & !OBJECT_DEFINED_BITS != 0 {
        return Err(error(RC_RESERVED_BITS));
    }
    let digest_bytes = hash::digest_len(public.name_alg).ok_or(error(RC_HASH))?;
    if !public.auth_policy.is_empty() && public.auth_policy.len() != digest_bytes {
        return Err(error(RC_SIZE));
    }
    // fixedTPM needs fixedParent, and a child of a duplicable parent is duplicable too.
    if public.has(OBJECT_FIXED_TPM) && !public.has(OBJECT_FIXED_PARENT) {
        return Err(error(RC_ATTRIBUTES));
    }
    if let Some(parent) = parent
        && public.has(OBJECT_FIXED_TPM)
        && !parent.has(OBJECT_FIXED_TPM)
    {
        return Err(error(RC_ATTRIBUTES));
    }
    if public.has(OBJECT_SIGN) || public.has(OBJECT_X509_SIGN) {
        return Err(error(RC_ATTRIBUTES));
    }
    public.parameters.check().map_err(error)?;
    let symmetric = public.parameters.symmetric();
    match public.object_type {
        ALG_KEYEDHASH => {
            // Sealed data only: the caller supplies the secret, so the TPM did not make it.
            if !public.is_sealed_data() || public.has(OBJECT_SENSITIVE_DATA_ORIGIN) {
                return Err(error(RC_ATTRIBUTES));
            }
        }
        ALG_RSA | ALG_ECC => {
            if !public.has(OBJECT_DECRYPT) || !public.has(OBJECT_SENSITIVE_DATA_ORIGIN) {
                return Err(error(RC_ATTRIBUTES));
            }
            let scheme = match &public.parameters {
                Parameters::Rsa { scheme, .. } | Parameters::Ecc { scheme, .. } => *scheme,
                _ => Scheme::NULL,
            };
            if public.has(OBJECT_RESTRICTED) {
                if !symmetric.is_aes_cfb() {
                    return Err(error(RC_SYMMETRIC));
                }
                if !scheme.is_null() {
                    return Err(error(RC_SCHEME));
                }
            } else if symmetric.algorithm != hash::ALG_NULL {
                return Err(error(RC_SYMMETRIC));
            }
        }
        _ => return Err(error(RC_TYPE)),
    }
    Ok(())
}

/// The secret half of an object.
#[derive(Clone)]
pub(super) enum Secret {
    Rsa(Box<RsaPrivateKey>),
    Ecc([u8; P256_BYTES]),
    Data(Vec<u8>),
}

/// A `TPMT_SENSITIVE`.
#[derive(Clone)]
pub(super) struct Sensitive {
    pub auth: Vec<u8>,
    /// A parent's protection seed, or a sealed object's obfuscation value.
    pub seed: Vec<u8>,
    pub secret: Secret,
}

impl Sensitive {
    pub fn write(&self, object_type: u16, writer: &mut Writer) {
        writer.u16(object_type);
        writer.sized(&self.auth);
        writer.sized(&self.seed);
        match &self.secret {
            Secret::Rsa(key) => writer.sized(&key.prime()),
            Secret::Ecc(private) => writer.sized(private),
            Secret::Data(data) => writer.sized(data),
        }
    }

    /// Reads a `TPMT_SENSITIVE` and checks it belongs to `public`: the type matches, an RSA prime
    /// divides the modulus, an ECC scalar gives the public point, a sealed object's digest
    /// covers its data. Anything else is `TPM_RC_BINDING` on parameter `number`.
    pub fn read(reader: &mut Reader, public: &Public, number: u32) -> Result<Self, u32> {
        let binding = parameter_error(RC_BINDING, number);
        let parse = |reader: &mut Reader| {
            Ok::<_, Truncated>((
                reader.u16()?,
                reader.sized()?.to_vec(),
                reader.sized()?.to_vec(),
                reader.sized()?.to_vec(),
            ))
        };
        let (object_type, auth, seed, secret) =
            parse(reader).map_err(|_| parameter_error(RC_INSUFFICIENT, number))?;
        if object_type != public.object_type {
            return Err(binding);
        }
        let digest_bytes = hash::digest_len(public.name_alg).unwrap_or(0);
        if auth.len() > digest_bytes || (!seed.is_empty() && seed.len() != digest_bytes) {
            return Err(parameter_error(RC_SIZE, number));
        }
        let secret = match (&public.unique, &public.parameters) {
            (Unique::Rsa(modulus), Parameters::Rsa { exponent, .. }) => {
                let key = RsaPrivateKey::from_prime(
                    modulus,
                    &rsa_exponent(*exponent).to_be_bytes(),
                    &secret,
                )
                .map_err(|_| binding)?;
                Secret::Rsa(Box::new(key))
            }
            (Unique::Ecc { .. }, _) => {
                let private: [u8; P256_BYTES] =
                    secret.as_slice().try_into().map_err(|_| binding)?;
                let point = ecc::p256_public_key(&private).map_err(|_| binding)?;
                if Some(point) != public.ecc_point() {
                    return Err(binding);
                }
                Secret::Ecc(private)
            }
            (Unique::Digest(unique), _) => {
                if hash::digest(public.name_alg, &[&seed, &secret]).as_deref() != Some(unique) {
                    return Err(binding);
                }
                Secret::Data(secret)
            }
            _ => return Err(binding),
        };
        Ok(Sensitive { auth, seed, secret })
    }
}

/// A loaded object, transient or persistent.
#[derive(Clone)]
pub(super) struct Object {
    pub handle: u32,
    /// `TPM_RH_OWNER`, `_ENDORSEMENT`, `_PLATFORM` or `_NULL`: whose seed it descends from.
    pub hierarchy: u32,
    pub public: Public,
    pub name: Vec<u8>,
    pub qualified_name: Vec<u8>,
    pub sensitive: Sensitive,
}

impl Object {
    pub fn new(
        handle: u32,
        hierarchy: u32,
        public: Public,
        sensitive: Sensitive,
        parent_qualified_name: &[u8],
    ) -> Self {
        let name = public.name();
        let qualified_name = qualified_name(public.name_alg, parent_qualified_name, &name);
        Object {
            handle,
            hierarchy,
            public,
            name,
            qualified_name,
            sensitive,
        }
    }

    pub fn sensitive_bytes(&self) -> Vec<u8> {
        marshal(|writer| self.sensitive.write(self.public.object_type, writer))
    }

    /// `TPM2B_PRIVATE` of a child of this storage parent (Part 1, "Protected Storage"):
    /// `TPM2B_DIGEST outerHMAC || AES-CFB(TPM2B_SENSITIVE)`. The key comes from this parent's
    /// seed and the child's name, the HMAC binds the ciphertext to the name.
    pub fn wrap(&self, child_name: &[u8], sensitive: &[u8]) -> Vec<u8> {
        let mut plaintext = marshal(|writer| writer.sized(sensitive));
        let (symmetric_key, integrity_key) = self.protection_keys(child_name);
        // A storage parent's symmetric definition is checked to be AES-CFB at creation.
        let _ = aes::cfb_encrypt(&symmetric_key, &ZERO_IV, &mut plaintext);
        let integrity = self.integrity(&integrity_key, &plaintext, child_name);
        marshal(|writer| {
            writer.sized(&integrity);
            writer.bytes(&plaintext);
        })
    }

    /// The inverse of `wrap`: the child's `TPMT_SENSITIVE` bytes, after checking the HMAC.
    pub fn unwrap(&self, child_name: &[u8], private: &[u8], number: u32) -> Result<Vec<u8>, u32> {
        let mut reader = Reader::new(private);
        let integrity = reader
            .sized()
            .map_err(|_| parameter_error(RC_SIZE, number))?;
        let mut ciphertext = reader.rest().to_vec();
        let (symmetric_key, integrity_key) = self.protection_keys(child_name);
        if integrity != self.integrity(&integrity_key, &ciphertext, child_name) {
            return Err(parameter_error(RC_INTEGRITY, number));
        }
        let _ = aes::cfb_decrypt(&symmetric_key, &ZERO_IV, &mut ciphertext);
        let mut reader = Reader::new(&ciphertext);
        let sensitive = reader
            .sized()
            .map_err(|_| parameter_error(RC_SIZE, number))?;
        if reader.remaining() != 0 {
            return Err(parameter_error(RC_SIZE, number));
        }
        Ok(sensitive.to_vec())
    }

    fn protection_keys(&self, child_name: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let name_alg = self.public.name_alg;
        let seed = &self.sensitive.seed;
        let key_bits = self.public.parameters.symmetric().key_bits as u32;
        let digest_bits = 8 * hash::digest_len(name_alg).unwrap_or(0) as u32;
        let symmetric = hash::kdfa(name_alg, seed, LABEL_STORAGE, child_name, &[], key_bits);
        let integrity = hash::kdfa(name_alg, seed, LABEL_INTEGRITY, &[], &[], digest_bits);
        (symmetric.unwrap_or_default(), integrity.unwrap_or_default())
    }

    fn integrity(&self, key: &[u8], ciphertext: &[u8], child_name: &[u8]) -> Vec<u8> {
        hash::hmac(self.public.name_alg, key, &[ciphertext, child_name]).unwrap_or_default()
    }
}

/// `QN(child) = nameAlg || H(QN(parent) || Name(child))`; a hierarchy's qualified name is its
/// handle.
pub(super) fn qualified_name(name_alg: u16, parent_qualified_name: &[u8], name: &[u8]) -> Vec<u8> {
    let mut qualified = name_alg.to_be_bytes().to_vec();
    qualified.extend(hash::digest(name_alg, &[parent_qualified_name, name]).unwrap_or_default());
    qualified
}

/// Makes the secret for `public` and fills in its unique field. `random` is the TPM's RNG for
/// an ordinary object and the seed's deterministic stream for a primary.
fn generate(
    public: &mut Public,
    auth: Vec<u8>,
    data: &[u8],
    random: &mut dyn FnMut(&mut [u8]),
) -> Result<Sensitive, u32> {
    let digest_bytes = hash::digest_len(public.name_alg).unwrap_or(0);
    let mut seed = Vec::new();
    if public.is_storage_parent() || public.object_type == ALG_KEYEDHASH {
        seed = vec![0; digest_bytes];
        random(&mut seed);
    }
    let secret = match &public.parameters {
        Parameters::Rsa { exponent, .. } => {
            let key =
                RsaPrivateKey::generate(RSA_KEY_BITS as usize, rsa_exponent(*exponent), random)
                    .map_err(|_| RC_KEY_SIZE)?;
            public.unique = Unique::Rsa(key.public.modulus.clone());
            Secret::Rsa(Box::new(key))
        }
        Parameters::Ecc { .. } => {
            let mut private = [0u8; P256_BYTES];
            let mut attempts = 0;
            loop {
                random(&mut private);
                if ecc::p256_is_valid_private(&private) {
                    break;
                }
                attempts += 1;
                if attempts == ECC_KEY_ATTEMPTS {
                    return Err(RC_KEY);
                }
            }
            let point = ecc::p256_public_key(&private).map_err(|_| RC_KEY)?;
            public.unique = Unique::Ecc {
                x: point.x.to_vec(),
                y: point.y.to_vec(),
            };
            Secret::Ecc(private)
        }
        Parameters::KeyedHash { .. } | Parameters::SymCipher { .. } => {
            public.unique =
                Unique::Digest(hash::digest(public.name_alg, &[&seed, data]).unwrap_or_default());
            Secret::Data(data.to_vec())
        }
    };
    Ok(Sensitive { auth, seed, secret })
}

/// Drops trailing zero bytes, as the TPM does to every authorization value it stores or
/// compares (Part 1, "Authorization Values").
pub(super) fn trim_auth(auth: &[u8]) -> Vec<u8> {
    let length = auth
        .iter()
        .rposition(|byte| *byte != 0)
        .map_or(0, |last| last + 1);
    auth[..length].to_vec()
}

/// The contents of a `TPM2B_SENSITIVE_CREATE`: the new object's authorization value and, for
/// sealed data, the secret.
fn read_sensitive_create(
    bytes: &[u8],
    name_alg_digest_bytes: Option<usize>,
) -> Result<(Vec<u8>, Vec<u8>), u32> {
    let mut inner = Reader::new(bytes);
    let auth = inner.sized().map_err(|_| parameter_error(RC_SIZE, 1))?;
    let data = inner.sized().map_err(|_| parameter_error(RC_SIZE, 1))?;
    if inner.remaining() != 0 || data.len() > MAX_SEALED_BYTES {
        return Err(parameter_error(RC_SIZE, 1));
    }
    let auth = trim_auth(auth);
    if name_alg_digest_bytes.is_some_and(|limit| auth.len() > limit) {
        return Err(parameter_error(RC_SIZE, 1));
    }
    Ok((auth, data.to_vec()))
}

/// What both creation commands share after the sensitive area: the template, outsideInfo and
/// the PCR selection, and the checks of data against the template.
struct CreateInput {
    auth: Vec<u8>,
    data: Vec<u8>,
    public: Public,
    outside_info: Vec<u8>,
    pcr_selection: Vec<(u16, Vec<u8>)>,
}

impl Tpm {
    fn read_create_input(
        &self,
        parameters: &mut Reader,
        parent: Option<&Public>,
    ) -> Result<CreateInput, u32> {
        let sensitive_create = parameters.sized()?;
        let public = Public::read_sized(parameters, 2)?;
        let outside_info = parameters.sized()?.to_vec();
        let pcr_selection = self.read_pcr_selection(parameters, 4)?;
        end_of(parameters)?;
        check_template(&public, parent, 2)?;
        if outside_info.len() > hash::MAX_DIGEST_BYTES + 2 {
            return Err(parameter_error(RC_SIZE, 3));
        }
        let (auth, data) =
            read_sensitive_create(sensitive_create, hash::digest_len(public.name_alg))?;
        // Keys are made by the TPM; only sealed data comes from the caller.
        if public.object_type != ALG_KEYEDHASH && !data.is_empty() {
            return Err(parameter_error(RC_SIZE, 1));
        }
        Ok(CreateInput {
            auth,
            data,
            public,
            outside_info,
            pcr_selection,
        })
    }

    /// `TPMS_CREATION_DATA`, its digest under the object's name algorithm, and the
    /// `TPMT_TK_CREATION` ticket for it.
    fn creation_data(
        &self,
        object: &Object,
        parent_name_alg: u16,
        parent_name: &[u8],
        parent_qualified_name: &[u8],
        input: &CreateInput,
        writer: &mut Writer,
    ) {
        let name_alg = object.public.name_alg;
        let creation_data = marshal(|writer| {
            Self::write_pcr_selection(writer, &input.pcr_selection);
            writer.sized(&self.pcr_digest(name_alg, &input.pcr_selection));
            writer.u8(LOCALITY_ZERO);
            writer.u16(parent_name_alg);
            writer.sized(parent_name);
            writer.sized(parent_qualified_name);
            writer.sized(&input.outside_info);
        });
        let creation_hash = hash::digest(name_alg, &[&creation_data]).unwrap_or_default();
        writer.sized(&creation_data);
        writer.sized(&creation_hash);
        // The ticket proves to TPM2_CertifyCreation that this TPM made the object; nothing
        // here checks one, but its form is what clients parse.
        writer.u16(ST_CREATION);
        writer.u32(object.hierarchy);
        let ticket = hash::hmac(
            hash::ALG_SHA256,
            self.proof(object.hierarchy),
            &[&ST_CREATION.to_be_bytes(), &object.name, &creation_hash],
        )
        .unwrap_or_default();
        writer.sized(&ticket);
    }

    pub(super) fn create_primary(
        &mut self,
        hierarchy: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        if !matches!(hierarchy, RH_OWNER | RH_ENDORSEMENT | RH_PLATFORM | RH_NULL) {
            return Err(handle_error(RC_VALUE, 1));
        }
        let mut input = self.read_create_input(parameters, None)?;
        self.check_transient_room()?;
        // The template (unique field as given) and the sensitive data pick the key: the same
        // template under the same seed is the same key (Part 1, "Primary Objects").
        let template_name = input.public.name();
        let mut stream = KdfStream::new(
            input.public.name_alg,
            self.seed(hierarchy),
            LABEL_PRIMARY,
            &template_name,
            &input.data,
        )
        .ok_or(parameter_error(RC_HASH, 2))?;
        let sensitive = generate(
            &mut input.public,
            input.auth.clone(),
            &input.data,
            &mut |bytes| stream.fill(bytes),
        )?;
        let handle = self.allocate_transient();
        let hierarchy_name = hierarchy.to_be_bytes();
        let object = Object::new(
            handle,
            hierarchy,
            input.public.clone(),
            sensitive,
            &hierarchy_name,
        );
        let mut writer = Writer::new();
        writer.sized(&object.public.bytes());
        self.creation_data(
            &object,
            hash::ALG_NULL,
            &hierarchy_name,
            &hierarchy_name,
            &input,
            &mut writer,
        );
        writer.sized(&object.name);
        self.objects.push(object);
        Ok(Response {
            handles: vec![handle],
            parameters: writer,
        })
    }

    pub(super) fn create(&mut self, parent: u32, parameters: &mut Reader) -> Result<Response, u32> {
        let parent = self.storage_parent(parent)?;
        let mut input = self.read_create_input(parameters, Some(&parent.public))?;
        let sensitive = {
            let random = &mut self.random;
            generate(
                &mut input.public,
                input.auth.clone(),
                &input.data,
                &mut |bytes| random.fill(bytes),
            )?
        };
        let object = Object::new(
            0,
            parent.hierarchy,
            input.public.clone(),
            sensitive,
            &parent.qualified_name,
        );
        let private = parent.wrap(&object.name, &object.sensitive_bytes());
        let mut writer = Writer::new();
        writer.sized(&private);
        writer.sized(&object.public.bytes());
        self.creation_data(
            &object,
            parent.public.name_alg,
            &parent.name,
            &parent.qualified_name,
            &input,
            &mut writer,
        );
        Ok(Response::with(writer))
    }

    pub(super) fn load(&mut self, parent: u32, parameters: &mut Reader) -> Result<Response, u32> {
        let parent = self.storage_parent(parent)?;
        let private = parameters.sized()?.to_vec();
        let public = Public::read_sized(parameters, 2)?;
        end_of(parameters)?;
        check_template(&public, Some(&parent.public), 2)?;
        self.check_transient_room()?;
        let name = public.name();
        let sensitive_bytes = parent.unwrap(&name, &private, 1)?;
        let mut reader = Reader::new(&sensitive_bytes);
        let sensitive = Sensitive::read(&mut reader, &public, 1)?;
        if reader.remaining() != 0 {
            return Err(parameter_error(RC_SIZE, 1));
        }
        let handle = self.allocate_transient();
        let object = Object::new(
            handle,
            parent.hierarchy,
            public,
            sensitive,
            &parent.qualified_name,
        );
        let mut writer = Writer::new();
        writer.sized(&object.name);
        self.objects.push(object);
        Ok(Response {
            handles: vec![handle],
            parameters: writer,
        })
    }

    /// A loaded storage parent, cloned so the caller can go on to change `self`.
    fn storage_parent(&self, handle: u32) -> Result<Object, u32> {
        let parent = self.object(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        if !parent.public.is_storage_parent() {
            return Err(handle_error(RC_TYPE, 1));
        }
        Ok(parent.clone())
    }

    pub(super) fn read_public(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        if self
            .sequences
            .iter()
            .any(|sequence| sequence.handle == handle)
        {
            return Err(handle_error(RC_SEQUENCE, 1));
        }
        let object = self.object(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        let mut writer = Writer::new();
        writer.sized(&object.public.bytes());
        writer.sized(&object.name);
        writer.sized(&object.qualified_name);
        Ok(Response::with(writer))
    }

    pub(super) fn unseal(&mut self, handle: u32, parameters: &mut Reader) -> Result<Response, u32> {
        end_of(parameters)?;
        let object = self.object(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        if object.public.object_type != ALG_KEYEDHASH {
            return Err(handle_error(RC_TYPE, 1));
        }
        if !object.public.is_sealed_data() {
            return Err(handle_error(RC_ATTRIBUTES, 1));
        }
        let Secret::Data(data) = &object.sensitive.secret else {
            return Err(handle_error(RC_TYPE, 1));
        };
        let mut writer = Writer::new();
        writer.sized(data);
        Ok(Response::with(writer))
    }

    pub(super) fn evict_control(
        &mut self,
        auth: u32,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let persistent = parameters.u32()?;
        end_of(parameters)?;
        if !matches!(auth, RH_OWNER | RH_PLATFORM) {
            return Err(handle_error(RC_HIERARCHY, 1));
        }
        let range = if auth == RH_OWNER {
            PERSISTENT_OWNER_FIRST..PERSISTENT_PLATFORM_FIRST
        } else {
            PERSISTENT_PLATFORM_FIRST..PERSISTENT_LAST + 1
        };
        if !range.contains(&persistent) {
            return Err(parameter_error(RC_RANGE, 1));
        }
        let object = self.object(handle).ok_or(handle_error(RC_HANDLE, 2))?;
        if handle_type(handle) == HT_PERSISTENT {
            if persistent != handle {
                return Err(parameter_error(RC_HANDLE, 1));
            }
            self.remove_persistent(handle)?;
            return Ok(Response::empty());
        }
        if object.public.has(OBJECT_ST_CLEAR) {
            return Err(handle_error(RC_ATTRIBUTES, 2));
        }
        // The owner keeps the storage and endorsement hierarchies' objects, the platform
        // anything but the null hierarchy's, which do not outlive a reset.
        let allowed = match object.hierarchy {
            RH_NULL => false,
            RH_PLATFORM => auth == RH_PLATFORM,
            _ => true,
        };
        if !allowed {
            return Err(handle_error(RC_HIERARCHY, 2));
        }
        if self.object(persistent).is_some() {
            return Err(RC_NV_DEFINED);
        }
        let mut copy = object.clone();
        copy.handle = persistent;
        self.add_persistent(copy)?;
        Ok(Response::empty())
    }

    pub(super) fn test_parms(&mut self, parameters: &mut Reader) -> Result<Response, u32> {
        let object_type = parameters.u16()?;
        let parms =
            Parameters::read(object_type, parameters)?.ok_or(parameter_error(RC_TYPE, 1))?;
        end_of(parameters)?;
        parms.check().map_err(|code| parameter_error(code, 1))?;
        Ok(Response::empty())
    }

    pub(super) fn ecdh_key_gen(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        end_of(parameters)?;
        let object = self.object(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        let peer = object.public.ecc_point().ok_or(handle_error(RC_KEY, 1))?;
        let mut ephemeral = [0u8; P256_BYTES];
        loop {
            self.random.fill(&mut ephemeral);
            if ecc::p256_is_valid_private(&ephemeral) {
                break;
            }
        }
        let shared = ecc::p256_scalar_multiply(&ephemeral, &peer).map_err(|_| RC_KEY)?;
        let public = ecc::p256_public_key(&ephemeral).map_err(|_| RC_KEY)?;
        let mut writer = Writer::new();
        write_ecc_point(&mut writer, &shared);
        write_ecc_point(&mut writer, &public);
        Ok(Response::with(writer))
    }

    pub(super) fn ecdh_z_gen(
        &mut self,
        handle: u32,
        parameters: &mut Reader,
    ) -> Result<Response, u32> {
        let point = read_ecc_point(parameters, 1)?;
        end_of(parameters)?;
        let object = self.object(handle).ok_or(handle_error(RC_HANDLE, 1))?;
        let Secret::Ecc(private) = &object.sensitive.secret else {
            return Err(handle_error(RC_KEY, 1));
        };
        if object.public.has(OBJECT_RESTRICTED) || !object.public.has(OBJECT_DECRYPT) {
            return Err(handle_error(RC_ATTRIBUTES, 1));
        }
        let shared = ecc::p256_scalar_multiply(private, &point)
            .map_err(|_| parameter_error(RC_ECC_POINT, 1))?;
        let mut writer = Writer::new();
        write_ecc_point(&mut writer, &shared);
        Ok(Response::with(writer))
    }
}

/// A `TPM2B_ECC_POINT`.
pub(super) fn write_ecc_point(writer: &mut Writer, point: &EccPoint) {
    writer.sized_with(|writer| {
        writer.sized(&point.x);
        writer.sized(&point.y);
    });
}

/// A `TPMS_ECC_POINT` as P-256 coordinates; anything else is `TPM_RC_ECC_POINT`.
pub(super) fn read_ecc_point_contents(reader: &mut Reader, number: u32) -> Result<EccPoint, u32> {
    let bad = parameter_error(RC_ECC_POINT, number);
    let coordinate = |reader: &mut Reader| -> Result<[u8; P256_BYTES], u32> {
        let bytes = reader.sized().map_err(|_| bad)?;
        // A coordinate may come without its leading zeros.
        if bytes.len() > P256_BYTES {
            return Err(bad);
        }
        let mut padded = [0u8; P256_BYTES];
        padded[P256_BYTES - bytes.len()..].copy_from_slice(bytes);
        Ok(padded)
    };
    let point = EccPoint {
        x: coordinate(reader)?,
        y: coordinate(reader)?,
    };
    if !ecc::p256_is_on_curve(&point) {
        return Err(bad);
    }
    Ok(point)
}

/// A `TPM2B_ECC_POINT`.
fn read_ecc_point(parameters: &mut Reader, number: u32) -> Result<EccPoint, u32> {
    let bytes = parameters.sized()?;
    let mut inner = Reader::new(bytes);
    let point = read_ecc_point_contents(&mut inner, number)?;
    if inner.remaining() != 0 {
        return Err(parameter_error(RC_SIZE, number));
    }
    Ok(point)
}
