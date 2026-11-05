//! Numbers from TPM 2.0 Library Part 2 ("Structures") that more than one part of the TPM uses:
//! response codes, command codes, handles, algorithm identifiers and attribute bits.

pub(super) const ST_NO_SESSIONS: u16 = 0x8001;
pub(super) const ST_SESSIONS: u16 = 0x8002;
pub(super) const ST_CREATION: u16 = 0x8021;
pub(super) const ST_HASHCHECK: u16 = 0x8024;

pub(crate) const RC_SUCCESS: u32 = 0x000;
pub(super) const RC_BAD_TAG: u32 = 0x01e;

// Format-zero codes (`RC_VER1` plus a number).
pub(super) const RC_INITIALIZE: u32 = 0x100;
pub(super) const RC_SEQUENCE: u32 = 0x103;
pub(super) const RC_AUTH_MISSING: u32 = 0x125;
pub(super) const RC_PCR_CHANGED: u32 = 0x128;
pub(super) const RC_AUTH_UNAVAILABLE: u32 = 0x12f;
pub(super) const RC_COMMAND_SIZE: u32 = 0x142;
pub(super) const RC_COMMAND_CODE: u32 = 0x143;
pub(super) const RC_AUTHSIZE: u32 = 0x144;
pub(super) const RC_NV_RANGE: u32 = 0x146;
pub(super) const RC_NV_AUTHORIZATION: u32 = 0x149;
pub(super) const RC_NV_UNINITIALIZED: u32 = 0x14a;
pub(super) const RC_NV_SPACE: u32 = 0x14b;
pub(super) const RC_NV_DEFINED: u32 = 0x14c;
pub(super) const RC_BAD_CONTEXT: u32 = 0x150;

// Format-one codes: which handle, session or parameter was wrong goes in the bits above them
// (`handle_error`, `session_error`, `parameter_error`).
pub(super) const RC_ATTRIBUTES: u32 = 0x082;
pub(super) const RC_HASH: u32 = 0x083;
pub(super) const RC_VALUE: u32 = 0x084;
pub(super) const RC_HIERARCHY: u32 = 0x085;
pub(super) const RC_KEY_SIZE: u32 = 0x087;
pub(super) const RC_TYPE: u32 = 0x08a;
pub(super) const RC_HANDLE: u32 = 0x08b;
pub(super) const RC_KDF: u32 = 0x08c;
pub(super) const RC_RANGE: u32 = 0x08d;
pub(super) const RC_AUTH_FAIL: u32 = 0x08e;
pub(super) const RC_SCHEME: u32 = 0x092;
pub(super) const RC_SIZE: u32 = 0x095;
pub(super) const RC_SYMMETRIC: u32 = 0x096;
pub(super) const RC_INSUFFICIENT: u32 = 0x09a;
pub(super) const RC_KEY: u32 = 0x09c;
pub(super) const RC_POLICY_FAIL: u32 = 0x09d;
pub(super) const RC_INTEGRITY: u32 = 0x09f;
pub(super) const RC_RESERVED_BITS: u32 = 0x0a1;
pub(super) const RC_CURVE: u32 = 0x0a6;
pub(super) const RC_ECC_POINT: u32 = 0x0a7;
const RC_PARAMETER: u32 = 0x040;
const RC_SESSION: u32 = 0x800;
const RC_NUMBER_SHIFT: u32 = 8;

// Warnings (`RC_WARN` plus a number).
pub(super) const RC_OBJECT_MEMORY: u32 = 0x902;
pub(super) const RC_SESSION_MEMORY: u32 = 0x903;
pub(super) const RC_SESSION_HANDLES: u32 = 0x905;
pub(super) const RC_LOCALITY: u32 = 0x907;
/// The first handle names a transient object or session that is not loaded; the next six
/// codes are the later handles.
pub(super) const RC_REFERENCE_H0: u32 = 0x910;
/// Likewise for the first session in the authorization area.
pub(super) const RC_REFERENCE_S0: u32 = 0x918;
pub(super) const RC_NV_UNAVAILABLE: u32 = 0x923;

pub(super) fn handle_error(code: u32, number: u32) -> u32 {
    code | number << RC_NUMBER_SHIFT
}

pub(super) fn parameter_error(code: u32, number: u32) -> u32 {
    code | RC_PARAMETER | number << RC_NUMBER_SHIFT
}

pub(super) fn session_error(code: u32, number: u32) -> u32 {
    code | RC_SESSION | number << RC_NUMBER_SHIFT
}

pub(super) const CC_EVICT_CONTROL: u32 = 0x120;
pub(super) const CC_NV_UNDEFINE_SPACE: u32 = 0x122;
pub(super) const CC_CLEAR: u32 = 0x126;
pub(super) const CC_HIERARCHY_CHANGE_AUTH: u32 = 0x129;
pub(super) const CC_NV_DEFINE_SPACE: u32 = 0x12a;
pub(super) const CC_CREATE_PRIMARY: u32 = 0x131;
pub(super) const CC_NV_WRITE: u32 = 0x137;
pub(super) const CC_DICTIONARY_ATTACK_LOCK_RESET: u32 = 0x139;
pub(super) const CC_PCR_EVENT: u32 = 0x13c;
pub(super) const CC_PCR_RESET: u32 = 0x13d;
pub(super) const CC_SEQUENCE_COMPLETE: u32 = 0x13e;
pub(super) const CC_INCREMENTAL_SELF_TEST: u32 = 0x142;
pub(super) const CC_SELF_TEST: u32 = 0x143;
pub(super) const CC_STARTUP: u32 = 0x144;
pub(super) const CC_SHUTDOWN: u32 = 0x145;
pub(super) const CC_STIR_RANDOM: u32 = 0x146;
pub(super) const CC_NV_READ: u32 = 0x14e;
pub(super) const CC_CREATE: u32 = 0x153;
pub(super) const CC_ECDH_ZGEN: u32 = 0x154;
pub(super) const CC_LOAD: u32 = 0x157;
pub(super) const CC_SEQUENCE_UPDATE: u32 = 0x15c;
pub(super) const CC_UNSEAL: u32 = 0x15e;
pub(super) const CC_CONTEXT_LOAD: u32 = 0x161;
pub(super) const CC_CONTEXT_SAVE: u32 = 0x162;
pub(super) const CC_ECDH_KEYGEN: u32 = 0x163;
pub(super) const CC_FLUSH_CONTEXT: u32 = 0x165;
pub(super) const CC_NV_READ_PUBLIC: u32 = 0x169;
pub(super) const CC_POLICY_AUTH_VALUE: u32 = 0x16b;
pub(super) const CC_READ_PUBLIC: u32 = 0x173;
pub(super) const CC_START_AUTH_SESSION: u32 = 0x176;
pub(super) const CC_GET_CAPABILITY: u32 = 0x17a;
pub(super) const CC_GET_RANDOM: u32 = 0x17b;
pub(super) const CC_GET_TEST_RESULT: u32 = 0x17c;
pub(super) const CC_HASH: u32 = 0x17d;
pub(super) const CC_PCR_READ: u32 = 0x17e;
pub(super) const CC_POLICY_PCR: u32 = 0x17f;
pub(super) const CC_POLICY_RESTART: u32 = 0x180;
pub(super) const CC_READ_CLOCK: u32 = 0x181;
pub(super) const CC_PCR_EXTEND: u32 = 0x182;
pub(super) const CC_EVENT_SEQUENCE_COMPLETE: u32 = 0x185;
pub(super) const CC_HASH_SEQUENCE_START: u32 = 0x186;
pub(super) const CC_POLICY_GET_DIGEST: u32 = 0x189;
pub(super) const CC_TEST_PARMS: u32 = 0x18a;
pub(super) const CC_POLICY_PASSWORD: u32 = 0x18c;

pub(super) const HT_PCR: u8 = 0x00;
pub(super) const HT_NV_INDEX: u8 = 0x01;
/// Also `TPM_HT_LOADED_SESSION` in `TPM_CAP_HANDLES`.
pub(super) const HT_HMAC_SESSION: u8 = 0x02;
/// Also `TPM_HT_SAVED_SESSION` in `TPM_CAP_HANDLES`.
pub(super) const HT_POLICY_SESSION: u8 = 0x03;
pub(super) const HT_PERMANENT: u8 = 0x40;
pub(super) const HT_TRANSIENT: u8 = 0x80;
pub(super) const HT_PERSISTENT: u8 = 0x81;
pub(super) const HANDLE_TYPE_SHIFT: u32 = 24;

pub(super) fn handle_type(handle: u32) -> u8 {
    (handle >> HANDLE_TYPE_SHIFT) as u8
}

pub(super) const RH_OWNER: u32 = 0x4000_0001;
pub(super) const RH_NULL: u32 = 0x4000_0007;
/// The password session: authorization by a plaintext value.
pub(super) const RS_PW: u32 = 0x4000_0009;
pub(super) const RH_LOCKOUT: u32 = 0x4000_000a;
pub(super) const RH_ENDORSEMENT: u32 = 0x4000_000b;
pub(super) const RH_PLATFORM: u32 = 0x4000_000c;
pub(super) const RH_PLATFORM_NV: u32 = 0x4000_000d;
pub(super) const PERMANENT_HANDLES: [u32; 7] = [
    RH_OWNER,
    RH_NULL,
    RS_PW,
    RH_LOCKOUT,
    RH_ENDORSEMENT,
    RH_PLATFORM,
    RH_PLATFORM_NV,
];

pub(super) const TRANSIENT_FIRST: u32 = 0x8000_0000;
/// `TPMS_CONTEXT.savedHandle` for an object (Part 2, "TPMS_CONTEXT").
pub(super) const SAVED_OBJECT: u32 = 0x8000_0000;
/// Persistent handles the owner may use; the platform has the upper half.
pub(super) const PERSISTENT_OWNER_FIRST: u32 = 0x8100_0000;
pub(super) const PERSISTENT_PLATFORM_FIRST: u32 = 0x8180_0000;
pub(super) const PERSISTENT_LAST: u32 = 0x81ff_ffff;
pub(super) const NV_INDEX_FIRST: u32 = 0x0100_0000;
pub(super) const NV_INDEX_LAST: u32 = 0x01ff_ffff;

pub(super) const ALG_RSA: u16 = 0x0001;
pub(super) const ALG_HMAC: u16 = 0x0005;
pub(super) const ALG_AES: u16 = 0x0006;
pub(super) const ALG_MGF1: u16 = 0x0007;
pub(super) const ALG_KEYEDHASH: u16 = 0x0008;
pub(super) const ALG_XOR: u16 = 0x000a;
pub(super) const ALG_RSAES: u16 = 0x0015;
pub(super) const ALG_OAEP: u16 = 0x0017;
pub(super) const ALG_ECDAA: u16 = 0x001a;
pub(super) const ALG_ECDH: u16 = 0x0019;
pub(super) const ALG_KDF1_SP800_56A: u16 = 0x0020;
pub(super) const ALG_KDF1_SP800_108: u16 = 0x0022;
pub(super) const ALG_ECC: u16 = 0x0023;
pub(super) const ALG_SYMCIPHER: u16 = 0x0025;
pub(super) const ALG_CFB: u16 = 0x0043;

pub(super) const ECC_NIST_P256: u16 = 0x0003;

// TPMA_OBJECT.
pub(super) const OBJECT_FIXED_TPM: u32 = 1 << 1;
pub(super) const OBJECT_ST_CLEAR: u32 = 1 << 2;
pub(super) const OBJECT_FIXED_PARENT: u32 = 1 << 4;
pub(super) const OBJECT_SENSITIVE_DATA_ORIGIN: u32 = 1 << 5;
pub(super) const OBJECT_USER_WITH_AUTH: u32 = 1 << 6;
pub(super) const OBJECT_ADMIN_WITH_POLICY: u32 = 1 << 7;
pub(super) const OBJECT_NO_DA: u32 = 1 << 10;
pub(super) const OBJECT_ENCRYPTED_DUPLICATION: u32 = 1 << 11;
pub(super) const OBJECT_RESTRICTED: u32 = 1 << 16;
pub(super) const OBJECT_DECRYPT: u32 = 1 << 17;
pub(super) const OBJECT_SIGN: u32 = 1 << 18;
pub(super) const OBJECT_X509_SIGN: u32 = 1 << 19;
pub(super) const OBJECT_DEFINED_BITS: u32 = OBJECT_FIXED_TPM
    | OBJECT_ST_CLEAR
    | OBJECT_FIXED_PARENT
    | OBJECT_SENSITIVE_DATA_ORIGIN
    | OBJECT_USER_WITH_AUTH
    | OBJECT_ADMIN_WITH_POLICY
    | OBJECT_NO_DA
    | OBJECT_ENCRYPTED_DUPLICATION
    | OBJECT_RESTRICTED
    | OBJECT_DECRYPT
    | OBJECT_SIGN
    | OBJECT_X509_SIGN;

// TPMA_SESSION.
pub(super) const SESSION_CONTINUE: u8 = 1 << 0;
pub(super) const SESSION_AUDIT_EXCLUSIVE: u8 = 1 << 1;
pub(super) const SESSION_AUDIT_RESET: u8 = 1 << 2;
pub(super) const SESSION_DECRYPT: u8 = 1 << 5;
pub(super) const SESSION_ENCRYPT: u8 = 1 << 6;
pub(super) const SESSION_AUDIT: u8 = 1 << 7;

/// `TPMA_LOCALITY` for locality 0, the only one the CRB interface offers.
pub(super) const LOCALITY_ZERO: u8 = 1 << 0;
