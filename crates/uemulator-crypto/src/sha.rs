//! SHA-1, SHA-256, SHA-384 and SHA-512 (FIPS 180-4).
//!
//! The firmware hashes whole kernels and initramfs images with these, as guest code, so the
//! compression functions read full blocks straight out of the caller's slice and only the
//! partial tail goes through the buffer.

use alloc::vec::Vec;

use crate::der::oid;

const SHA1_BLOCK_BYTES: usize = 64;
const SHA256_BLOCK_BYTES: usize = 64;
const SHA512_BLOCK_BYTES: usize = 128;

pub const SHA1_DIGEST_BYTES: usize = 20;
pub const SHA256_DIGEST_BYTES: usize = 32;
pub const SHA384_DIGEST_BYTES: usize = 48;
pub const SHA512_DIGEST_BYTES: usize = 64;

/// The largest digest any algorithm here produces.
pub const MAX_DIGEST_BYTES: usize = SHA512_DIGEST_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha1,
    Sha256,
    Sha384,
    Sha512,
}

impl HashAlgorithm {
    pub const ALL: [HashAlgorithm; 4] = [Self::Sha1, Self::Sha256, Self::Sha384, Self::Sha512];

    pub const fn digest_len(self) -> usize {
        match self {
            Self::Sha1 => SHA1_DIGEST_BYTES,
            Self::Sha256 => SHA256_DIGEST_BYTES,
            Self::Sha384 => SHA384_DIGEST_BYTES,
            Self::Sha512 => SHA512_DIGEST_BYTES,
        }
    }

    /// The algorithm identifier's OID contents (without tag and length), as used in
    /// DigestInfo, SignerInfo.digestAlgorithm and SpcIndirectDataContent.
    pub const fn oid(self) -> &'static [u8] {
        match self {
            Self::Sha1 => oid::SHA1,
            Self::Sha256 => oid::SHA256,
            Self::Sha384 => oid::SHA384,
            Self::Sha512 => oid::SHA512,
        }
    }

    pub fn from_oid(contents: &[u8]) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|algorithm| algorithm.oid() == contents)
    }

    pub fn digest(self, data: &[u8]) -> Vec<u8> {
        let mut hasher = Hasher::new(self);
        hasher.update(data);
        hasher.finish()
    }
}

/// Any of the hashes, picked at run time.
#[derive(Clone)]
pub enum Hasher {
    Sha1(Sha1),
    Sha256(Sha256),
    Sha384(Sha384),
    Sha512(Sha512),
}

impl Hasher {
    pub fn new(algorithm: HashAlgorithm) -> Self {
        match algorithm {
            HashAlgorithm::Sha1 => Self::Sha1(Sha1::new()),
            HashAlgorithm::Sha256 => Self::Sha256(Sha256::new()),
            HashAlgorithm::Sha384 => Self::Sha384(Sha384::new()),
            HashAlgorithm::Sha512 => Self::Sha512(Sha512::new()),
        }
    }

    pub fn algorithm(&self) -> HashAlgorithm {
        match self {
            Self::Sha1(_) => HashAlgorithm::Sha1,
            Self::Sha256(_) => HashAlgorithm::Sha256,
            Self::Sha384(_) => HashAlgorithm::Sha384,
            Self::Sha512(_) => HashAlgorithm::Sha512,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        match self {
            Self::Sha1(hasher) => hasher.update(data),
            Self::Sha256(hasher) => hasher.update(data),
            Self::Sha384(hasher) => hasher.update(data),
            Self::Sha512(hasher) => hasher.update(data),
        }
    }

    pub fn finish(self) -> Vec<u8> {
        match self {
            Self::Sha1(hasher) => hasher.finish().to_vec(),
            Self::Sha256(hasher) => hasher.finish().to_vec(),
            Self::Sha384(hasher) => hasher.finish().to_vec(),
            Self::Sha512(hasher) => hasher.finish().to_vec(),
        }
    }
}

pub fn sha1(data: &[u8]) -> [u8; SHA1_DIGEST_BYTES] {
    let mut hasher = Sha1::new();
    hasher.update(data);
    hasher.finish()
}

pub fn sha256(data: &[u8]) -> [u8; SHA256_DIGEST_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finish()
}

pub fn sha384(data: &[u8]) -> [u8; SHA384_DIGEST_BYTES] {
    let mut hasher = Sha384::new();
    hasher.update(data);
    hasher.finish()
}

pub fn sha512(data: &[u8]) -> [u8; SHA512_DIGEST_BYTES] {
    let mut hasher = Sha512::new();
    hasher.update(data);
    hasher.finish()
}

/// Buffering shared by every hash: whole blocks go straight to `compress`, the remainder waits
/// in `pending` for more input.
#[derive(Clone)]
struct BlockBuffer<const BLOCK: usize> {
    pending: [u8; BLOCK],
    pending_len: usize,
    /// Total message length in bytes. SHA-512's 128-bit length field never needs the high half:
    /// 2^64 bytes is out of reach here.
    total_len: u64,
}

impl<const BLOCK: usize> BlockBuffer<BLOCK> {
    const fn new() -> Self {
        Self {
            pending: [0; BLOCK],
            pending_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8], mut compress: impl FnMut(&[u8])) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.pending_len > 0 {
            let take = (BLOCK - self.pending_len).min(data.len());
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&data[..take]);
            self.pending_len += take;
            data = &data[take..];
            if self.pending_len < BLOCK {
                return;
            }
            compress(&self.pending);
            self.pending_len = 0;
        }
        let whole = data.len() / BLOCK * BLOCK;
        for block in data[..whole].chunks_exact(BLOCK) {
            compress(block);
        }
        let rest = &data[whole..];
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    /// Appends the 0x80 terminator, zero padding and the big-endian bit length in a
    /// `LENGTH_BYTES`-wide field (8 for the 64-byte-block hashes, 16 for SHA-512).
    fn finish<const LENGTH_BYTES: usize>(mut self, mut compress: impl FnMut(&[u8])) {
        let bit_len = (self.total_len as u128).wrapping_mul(8);
        let mut at = self.pending_len;
        self.pending[at] = 0x80;
        at += 1;
        if at > BLOCK - LENGTH_BYTES {
            self.pending[at..].fill(0);
            compress(&self.pending);
            at = 0;
        }
        self.pending[at..BLOCK - LENGTH_BYTES].fill(0);
        let length = bit_len.to_be_bytes();
        self.pending[BLOCK - LENGTH_BYTES..].copy_from_slice(&length[16 - LENGTH_BYTES..]);
        compress(&self.pending);
    }
}

#[derive(Clone)]
pub struct Sha1 {
    state: [u32; 5],
    buffer: BlockBuffer<SHA1_BLOCK_BYTES>,
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha1 {
    const INITIAL: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];

    pub const fn new() -> Self {
        Self {
            state: Self::INITIAL,
            buffer: BlockBuffer::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        let state = &mut self.state;
        self.buffer
            .update(data, |block| sha1_compress(state, block));
    }

    pub fn finish(mut self) -> [u8; SHA1_DIGEST_BYTES] {
        let state = &mut self.state;
        self.buffer.finish::<8>(|block| sha1_compress(state, block));
        let mut out = [0; SHA1_DIGEST_BYTES];
        for (bytes, word) in out.chunks_exact_mut(4).zip(self.state) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn sha1_compress(state: &mut [u32; 5], block: &[u8]) {
    let mut schedule = [0u32; 80];
    for (word, bytes) in schedule.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for index in 16..80 {
        schedule[index] = (schedule[index - 3]
            ^ schedule[index - 8]
            ^ schedule[index - 14]
            ^ schedule[index - 16])
            .rotate_left(1);
    }
    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for (index, word) in schedule.iter().enumerate() {
        let (f, k) = match index {
            0..=19 => ((b & c) | (!b & d), 0x5A827999),
            20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
            40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
            _ => (b ^ c ^ d, 0xCA62C1D6),
        };
        let temp = a
            .rotate_left(5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(*word);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = temp;
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e]) {
        *word = word.wrapping_add(value);
    }
}

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: BlockBuffer<SHA256_BLOCK_BYTES>,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    pub const fn new() -> Self {
        Self {
            state: Self::INITIAL,
            buffer: BlockBuffer::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        let state = &mut self.state;
        self.buffer
            .update(data, |block| sha256_compress(state, block));
    }

    pub fn finish(mut self) -> [u8; SHA256_DIGEST_BYTES] {
        let state = &mut self.state;
        self.buffer
            .finish::<8>(|block| sha256_compress(state, block));
        let mut out = [0; SHA256_DIGEST_BYTES];
        for (bytes, word) in out.chunks_exact_mut(4).zip(self.state) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn sha256_compress(state: &mut [u32; 8], block: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if sha_extensions::available() {
        // Checked by `available`; the firmware hashes every kernel and initramfs it measures,
        // and in the emulated guest the instructions are several times faster than the
        // portable rounds.
        unsafe { sha_extensions::sha256_compress(state, block) };
        return;
    }
    sha256_compress_portable(state, block);
}

fn sha256_compress_portable(state: &mut [u32; 8], block: &[u8]) {
    let mut schedule = [0u32; 64];
    for (word, bytes) in schedule.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (word, constant) in schedule.iter().zip(SHA256_ROUND_CONSTANTS) {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(constant)
            .wrapping_add(*word);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(value);
    }
}

const SHA512_ROUND_CONSTANTS: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// SHA-512's machinery; SHA-384 is the same with other initial values, truncated.
#[derive(Clone)]
struct Sha512Core {
    state: [u64; 8],
    buffer: BlockBuffer<SHA512_BLOCK_BYTES>,
}

impl Sha512Core {
    const fn new(initial: [u64; 8]) -> Self {
        Self {
            state: initial,
            buffer: BlockBuffer::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        let state = &mut self.state;
        self.buffer
            .update(data, |block| sha512_compress(state, block));
    }

    fn finish(mut self) -> [u8; SHA512_DIGEST_BYTES] {
        let state = &mut self.state;
        self.buffer
            .finish::<16>(|block| sha512_compress(state, block));
        let mut out = [0; SHA512_DIGEST_BYTES];
        for (bytes, word) in out.chunks_exact_mut(8).zip(self.state) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn sha512_compress(state: &mut [u64; 8], block: &[u8]) {
    let mut schedule = [0u64; 80];
    for (word, bytes) in schedule.iter_mut().zip(block.chunks_exact(8)) {
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        *word = u64::from_be_bytes(array);
    }
    for index in 16..80 {
        let s0 = schedule[index - 15].rotate_right(1)
            ^ schedule[index - 15].rotate_right(8)
            ^ (schedule[index - 15] >> 7);
        let s1 = schedule[index - 2].rotate_right(19)
            ^ schedule[index - 2].rotate_right(61)
            ^ (schedule[index - 2] >> 6);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (word, constant) in schedule.iter().zip(SHA512_ROUND_CONSTANTS) {
        let sum1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let choose = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(constant)
            .wrapping_add(*word);
        let sum0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (word, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(value);
    }
}

#[derive(Clone)]
pub struct Sha512(Sha512Core);

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha512 {
    const INITIAL: [u64; 8] = [
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    ];

    pub const fn new() -> Self {
        Self(Sha512Core::new(Self::INITIAL))
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub fn finish(self) -> [u8; SHA512_DIGEST_BYTES] {
        self.0.finish()
    }
}

#[derive(Clone)]
pub struct Sha384(Sha512Core);

impl Default for Sha384 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha384 {
    const INITIAL: [u64; 8] = [
        0xcbbb9d5dc1059ed8,
        0x629a292a367cd507,
        0x9159015a3070dd17,
        0x152fecd8f70e5939,
        0x67332667ffc00b31,
        0x8eb44a8768581511,
        0xdb0c2e0d64f98fa7,
        0x47b5481dbefa4fa4,
    ];

    pub const fn new() -> Self {
        Self(Sha512Core::new(Self::INITIAL))
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub fn finish(self) -> [u8; SHA384_DIGEST_BYTES] {
        let full = self.0.finish();
        let mut out = [0; SHA384_DIGEST_BYTES];
        out.copy_from_slice(&full[..SHA384_DIGEST_BYTES]);
        out
    }
}

#[cfg(test)]
pub(crate) fn hex(bytes: &[u8]) -> alloc::string::String {
    use core::fmt::Write;
    let mut text = alloc::string::String::new();
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const FIPS_448_BITS: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const FIPS_896_BITS: &[u8] = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

    fn check(algorithm: HashAlgorithm, input: &[u8], expected: &str) {
        assert_eq!(hex(&algorithm.digest(input)), expected, "{algorithm:?}");
        // The same through odd-sized pieces, which exercises the buffering.
        for piece in [1, 3, 63, 64, 65, 127, 128, 129] {
            let mut hasher = Hasher::new(algorithm);
            for chunk in input.chunks(piece) {
                hasher.update(chunk);
            }
            assert_eq!(hex(&hasher.finish()), expected, "{algorithm:?} in {piece}s");
        }
    }

    #[test]
    fn sha1_matches_fips_180_vectors() {
        let algorithm = HashAlgorithm::Sha1;
        check(algorithm, b"", "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        check(
            algorithm,
            b"abc",
            "a9993e364706816aba3e25717850c26c9cd0d89d",
        );
        check(
            algorithm,
            FIPS_448_BITS,
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
        );
        check(
            algorithm,
            &vec![b'a'; 1_000_000],
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f",
        );
    }

    #[test]
    fn sha256_matches_fips_180_vectors() {
        let algorithm = HashAlgorithm::Sha256;
        check(
            algorithm,
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
        check(
            algorithm,
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        check(
            algorithm,
            FIPS_448_BITS,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        );
        check(
            algorithm,
            &vec![b'a'; 1_000_000],
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        );
    }

    #[test]
    fn sha384_matches_fips_180_vectors() {
        let algorithm = HashAlgorithm::Sha384;
        check(
            algorithm,
            b"",
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
        );
        check(
            algorithm,
            b"abc",
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
        );
        check(
            algorithm,
            FIPS_896_BITS,
            "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712fcc7c71a557e2db966c3e9fa91746039",
        );
        check(
            algorithm,
            &vec![b'a'; 1_000_000],
            "9d0e1809716474cb086e834e310a4a1ced149e9c00f248527972cec5704c2a5b07b8b3dc38ecc4ebae97ddd87f3d8985",
        );
    }

    #[test]
    fn sha512_matches_fips_180_vectors() {
        let algorithm = HashAlgorithm::Sha512;
        check(
            algorithm,
            b"",
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        );
        check(
            algorithm,
            b"abc",
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        );
        check(
            algorithm,
            FIPS_896_BITS,
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
        );
        check(
            algorithm,
            &vec![b'a'; 1_000_000],
            "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b",
        );
    }

    /// Lengths around the padding boundaries, checked against `openssl dgst` for the bytes
    /// 0, 1, 2, ... (mod 256).
    #[test]
    fn padding_boundaries_match_openssl() {
        let data: Vec<u8> = (0..300u32).map(|byte| byte as u8).collect();
        let cases: [(usize, HashAlgorithm, &str); 12] = [
            (
                55,
                HashAlgorithm::Sha1,
                "8ae2d46729cfe68ff927af5eec9c7d1b66d65ac2",
            ),
            (
                56,
                HashAlgorithm::Sha1,
                "636e2ec698dac903498e648bd2f3af641d3c88cb",
            ),
            (
                300,
                HashAlgorithm::Sha1,
                "bf77ecf143ceb21f1676c34b8d89c8bb3c43cc4e",
            ),
            (
                55,
                HashAlgorithm::Sha256,
                "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
            ),
            (
                56,
                HashAlgorithm::Sha256,
                "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
            ),
            (
                64,
                HashAlgorithm::Sha256,
                "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
            ),
            (
                300,
                HashAlgorithm::Sha256,
                "7728ae2f2c36e2aaafbe79ca14c87ae2f89e7c88c4390ecbbf82dce88706958d",
            ),
            (
                111,
                HashAlgorithm::Sha384,
                "f5f9fe110d809d34029de262a01b208356caec6e054c7f926b2591f6c9780579d4b59f5578c6f531a84f158a33660cef",
            ),
            (
                112,
                HashAlgorithm::Sha384,
                "33ba080ec0ccb378e4e95fed3b26c23aa1a280476e007519ee47f60cd9c5c8a65d627259a9aa2fd33ca06d3c14ee5548",
            ),
            (
                111,
                HashAlgorithm::Sha512,
                "a1a111449b198d9b1f538bad7f3fc1022b3a5b1a5e90a0bc860de8512746cbc31599e6c834de3a3235327af0b51ff57bf7acf1974a73014d9c3953812edc7c8d",
            ),
            (
                112,
                HashAlgorithm::Sha512,
                "c5fbd731d19d2ae1180f001be72c2c1aaba1d7b094b3748880e24593b8e117a750e11c1bd867cc2f96dace8c8b74abd2d5c4f236be444e77d30d1916174070b9",
            ),
            (
                300,
                HashAlgorithm::Sha512,
                "f1dca2eb677b303265b0b9baff0e061202818f35c1470a69bbaa9bb66025e948d90e565e69642506c6213aef3cf9e929357a59da263deb34d1236dbdcda279b3",
            ),
        ];
        for (length, algorithm, expected) in cases {
            check(algorithm, &data[..length], expected);
        }
    }

    #[test]
    fn algorithms_round_trip_through_their_oids() {
        for algorithm in HashAlgorithm::ALL {
            assert_eq!(HashAlgorithm::from_oid(algorithm.oid()), Some(algorithm));
            assert_eq!(algorithm.digest(b"x").len(), algorithm.digest_len());
        }
        assert_eq!(HashAlgorithm::from_oid(&[1, 2, 3]), None);
    }
}

/// SHA-256 with the x86 SHA extensions (Intel SDM vol. 1, 5.6; the register choreography is
/// Intel's "Intel SHA Extensions" white paper): two rounds per `sha256rnds2`, the state kept as
/// ABEF and CDGH halves.
///
/// Inline assembly rather than intrinsics: the firmware's target (`x86_64-unknown-none`) is
/// soft-float, where LLVM cannot lower vector values at all. The block saves and restores every
/// XMM register it touches, so it declares no vector clobbers (which that target would reject)
/// and is safe under any ABI, including the Microsoft one the firmware is called through, where
/// XMM6 and up belong to the caller.
#[cfg(target_arch = "x86_64")]
mod sha_extensions {
    use super::SHA256_ROUND_CONSTANTS;
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    use core::sync::atomic::{AtomicU8, Ordering};

    const UNKNOWN: u8 = 0;
    const PRESENT: u8 = 1;
    const ABSENT: u8 = 2;
    static SUPPORT: AtomicU8 = AtomicU8::new(UNKNOWN);

    const LEAF_FEATURES: u32 = 1;
    const LEAF_STRUCTURED_FEATURES: u32 = 7;
    const ECX_SSSE3: u32 = 1 << 9;
    const ECX_SSE4_1: u32 = 1 << 19;
    const EBX_SHA: u32 = 1 << 29;
    /// `pshufb` mask turning each little-endian dword of a block into its big-endian value.
    const BYTE_SWAP: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];
    /// XMM0 to XMM10, saved on entry and restored on exit.
    const SAVED_REGISTERS: usize = 11;

    pub fn available() -> bool {
        match SUPPORT.load(Ordering::Relaxed) {
            PRESENT => true,
            ABSENT => false,
            _ => {
                #[allow(unused_unsafe)]
                let (features, structured) = unsafe {
                    (
                        __cpuid(LEAF_FEATURES),
                        __cpuid_count(LEAF_STRUCTURED_FEATURES, 0),
                    )
                };
                let present = features.ecx & ECX_SSSE3 != 0
                    && features.ecx & ECX_SSE4_1 != 0
                    && structured.ebx & EBX_SHA != 0;
                SUPPORT.store(if present { PRESENT } else { ABSENT }, Ordering::Relaxed);
                present
            }
        }
    }

    #[repr(C, align(16))]
    struct Scratch {
        saved: [[u8; 16]; SAVED_REGISTERS],
        /// The 64-word message schedule, four words per group.
        schedule: [[u32; 4]; 16],
    }

    /// # Safety
    /// The CPU must have SHA, SSSE3 and SSE4.1 (`available`) with SSE enabled in CR4, and
    /// `block` must be 64 bytes.
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    pub unsafe fn sha256_compress(state: &mut [u32; 8], block: &[u8]) {
        debug_assert_eq!(block.len(), 64);
        let mut scratch = Scratch {
            saved: [[0; 16]; SAVED_REGISTERS],
            schedule: [[0; 4]; 16],
        };
        // xmm0: message plus constants (sha256rnds2's implicit operand); xmm1: ABEF; xmm2:
        // CDGH; xmm3-xmm6: schedule work; xmm7: the byte-swap mask; xmm8/xmm9: the state at
        // entry; xmm10: temporary.
        unsafe {
            core::arch::asm!(
                "movdqa [{saved} + 0x00], xmm0",
                "movdqa [{saved} + 0x10], xmm1",
                "movdqa [{saved} + 0x20], xmm2",
                "movdqa [{saved} + 0x30], xmm3",
                "movdqa [{saved} + 0x40], xmm4",
                "movdqa [{saved} + 0x50], xmm5",
                "movdqa [{saved} + 0x60], xmm6",
                "movdqa [{saved} + 0x70], xmm7",
                "movdqa [{saved} + 0x80], xmm8",
                "movdqa [{saved} + 0x90], xmm9",
                "movdqa [{saved} + 0xa0], xmm10",
                "movdqu xmm7, [{mask}]",
                // A..D, E..H to ABEF and CDGH.
                "movdqu xmm10, [{state}]",
                "movdqu xmm2, [{state} + 16]",
                "pshufd xmm10, xmm10, 0xb1",
                "pshufd xmm2, xmm2, 0x1b",
                "movdqa xmm1, xmm10",
                "palignr xmm1, xmm2, 8",
                "pblendw xmm2, xmm10, 0xf0",
                "movdqa xmm8, xmm1",
                "movdqa xmm9, xmm2",
                // Groups 0-3 are the block, byte-swapped.
                "movdqu xmm3, [{block}]",
                "pshufb xmm3, xmm7",
                "movdqa [{schedule}], xmm3",
                "movdqu xmm3, [{block} + 16]",
                "pshufb xmm3, xmm7",
                "movdqa [{schedule} + 16], xmm3",
                "movdqu xmm3, [{block} + 32]",
                "pshufb xmm3, xmm7",
                "movdqa [{schedule} + 32], xmm3",
                "movdqu xmm3, [{block} + 48]",
                "pshufb xmm3, xmm7",
                "movdqa [{schedule} + 48], xmm3",
                // Groups 4-15: W[g] = msg2(msg1(W[g-4], W[g-3]) + alignr(W[g-1], W[g-2], 4),
                // W[g-1]).
                "lea {cursor}, [{schedule} + 64]",
                "mov {count:e}, 12",
                "2:",
                "movdqa xmm3, [{cursor} - 64]",
                "sha256msg1 xmm3, [{cursor} - 48]",
                "movdqa xmm5, [{cursor} - 16]",
                "movdqa xmm10, xmm5",
                "palignr xmm10, [{cursor} - 32], 4",
                "paddd xmm3, xmm10",
                "sha256msg2 xmm3, xmm5",
                "movdqa [{cursor}], xmm3",
                "add {cursor}, 16",
                "dec {count:e}",
                "jnz 2b",
                // 16 groups of four rounds.
                "mov {cursor}, {schedule}",
                "mov {count:e}, 16",
                "3:",
                "movdqa xmm0, [{cursor}]",
                "movdqu xmm10, [{constants}]",
                "paddd xmm0, xmm10",
                "sha256rnds2 xmm2, xmm1, xmm0",
                "pshufd xmm0, xmm0, 0x0e",
                "sha256rnds2 xmm1, xmm2, xmm0",
                "add {cursor}, 16",
                "add {constants}, 16",
                "dec {count:e}",
                "jnz 3b",
                "paddd xmm1, xmm8",
                "paddd xmm2, xmm9",
                // Back to A..D, E..H.
                "pshufd xmm10, xmm1, 0x1b",
                "pshufd xmm2, xmm2, 0xb1",
                "movdqa xmm1, xmm10",
                "pblendw xmm1, xmm2, 0xf0",
                "palignr xmm2, xmm10, 8",
                "movdqu [{state}], xmm1",
                "movdqu [{state} + 16], xmm2",
                "movdqa xmm0, [{saved} + 0x00]",
                "movdqa xmm1, [{saved} + 0x10]",
                "movdqa xmm2, [{saved} + 0x20]",
                "movdqa xmm3, [{saved} + 0x30]",
                "movdqa xmm4, [{saved} + 0x40]",
                "movdqa xmm5, [{saved} + 0x50]",
                "movdqa xmm6, [{saved} + 0x60]",
                "movdqa xmm7, [{saved} + 0x70]",
                "movdqa xmm8, [{saved} + 0x80]",
                "movdqa xmm9, [{saved} + 0x90]",
                "movdqa xmm10, [{saved} + 0xa0]",
                state = in(reg) state.as_mut_ptr(),
                block = in(reg) block.as_ptr(),
                mask = in(reg) BYTE_SWAP.as_ptr(),
                saved = in(reg) scratch.saved.as_mut_ptr(),
                schedule = in(reg) scratch.schedule.as_mut_ptr(),
                constants = inout(reg) SHA256_ROUND_CONSTANTS.as_ptr() => _,
                cursor = out(reg) _,
                count = out(reg) _,
                options(nostack),
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_sha_extensions_agree_with_the_portable_rounds() {
            if !available() {
                return;
            }
            let mut block = [0u8; 64];
            let mut seed = 0x1234_5678u32;
            for round in 0..64 {
                for byte in block.iter_mut() {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    *byte = (seed >> 16) as u8;
                }
                let mut portable = [round as u32; 8];
                let mut extended = portable;
                super::super::sha256_compress_portable(&mut portable, &block);
                unsafe { sha256_compress(&mut extended, &block) };
                assert_eq!(portable, extended, "round {round}");
            }
        }
    }
}
