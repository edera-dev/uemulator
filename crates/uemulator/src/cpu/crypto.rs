//! Pure helpers behind the SSE4.2 CRC32 instruction, AES-NI and PCLMULQDQ.

/// The CRC-32C (Castagnoli) polynomial in reflected bit order, which is what the CRC32
/// instruction implements. Unlike zlib's CRC-32 there is no initial or final inversion: those are
/// the caller's business (the kernel's crc32c does them in software around the instruction).
const CRC32C_POLYNOMIAL_REFLECTED: u32 = 0x82f6_3b78;

const fn crc32c_byte_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut crc = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32C_POLYNOMIAL_REFLECTED
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[byte] = crc;
        byte += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = crc32c_byte_table();

/// Accumulates the low `bytes` bytes of `data` (least significant byte first) into `crc`.
pub(super) fn crc32c(mut crc: u32, data: u64, bytes: u8) -> u32 {
    for index in 0..bytes {
        let byte = (data >> (8 * index)) as u8;
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc as u8) ^ byte) as usize];
    }
    crc
}

/// Multiplication in GF(2^8) with the AES reduction polynomial x^8 + x^4 + x^3 + x + 1.
pub(in crate::cpu::sse) const fn gf_multiply(mut a: u8, mut b: u8) -> u8 {
    const REDUCTION: u8 = 0x1b;
    let mut product = 0;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80 != 0;
        a <<= 1;
        if carry {
            a ^= REDUCTION;
        }
        b >>= 1;
    }
    product
}

/// The AES S-box, derived from its definition (multiplicative inverse in GF(2^8), then the affine
/// transform with constant 0x63) rather than typed in, so a typo cannot hide in a table. The
/// FIPS-197 test vectors in the tests are what pins it.
const fn aes_sbox_table() -> [u8; 256] {
    const AFFINE_CONSTANT: u8 = 0x63;
    let mut table = [0u8; 256];
    let mut value = 0;
    while value < 256 {
        let x = value as u8;
        // x^254 is the inverse of x for nonzero x, and gives 0 for x = 0, which the S-box wants.
        let mut inverse = 1u8;
        let mut base = x;
        let mut exponent = 254;
        while exponent > 0 {
            if exponent & 1 != 0 {
                inverse = gf_multiply(inverse, base);
            }
            base = gf_multiply(base, base);
            exponent >>= 1;
        }
        table[value] = inverse
            ^ inverse.rotate_left(1)
            ^ inverse.rotate_left(2)
            ^ inverse.rotate_left(3)
            ^ inverse.rotate_left(4)
            ^ AFFINE_CONSTANT;
        value += 1;
    }
    table
}

const fn aes_inverse_sbox_table(sbox: &[u8; 256]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut value = 0;
    while value < 256 {
        table[sbox[value] as usize] = value as u8;
        value += 1;
    }
    table
}

static AES_SBOX: [u8; 256] = aes_sbox_table();
static AES_INVERSE_SBOX: [u8; 256] = aes_inverse_sbox_table(&AES_SBOX);

/// AES state as a vector register: byte `row + 4 * column` of the little-endian value is the
/// state matrix entry in that row and column, which is how the instructions lay out the block.
const AES_COLUMNS: usize = 4;
const AES_ROWS: usize = 4;

fn sub_bytes(state: [u8; 16], table: &[u8; 256]) -> [u8; 16] {
    state.map(|byte| table[byte as usize])
}

/// ShiftRows rotates row `r` left by `r` columns; the inverse rotates right.
fn shift_rows(state: [u8; 16], inverse: bool) -> [u8; 16] {
    let mut result = [0u8; 16];
    for column in 0..AES_COLUMNS {
        for row in 0..AES_ROWS {
            let source_column = if inverse {
                (column + AES_COLUMNS - row) % AES_COLUMNS
            } else {
                (column + row) % AES_COLUMNS
            };
            result[row + AES_ROWS * column] = state[row + AES_ROWS * source_column];
        }
    }
    result
}

/// MixColumns, or InvMixColumns: each column is multiplied by the circulant matrix whose first
/// row is (2 3 1 1), or (14 11 13 9) for the inverse.
fn mix_columns(state: [u8; 16], inverse: bool) -> [u8; 16] {
    const FORWARD: [u8; 4] = [2, 3, 1, 1];
    const INVERSE: [u8; 4] = [14, 11, 13, 9];
    let coefficients = if inverse { INVERSE } else { FORWARD };
    let mut result = [0u8; 16];
    for column in 0..AES_COLUMNS {
        for row in 0..AES_ROWS {
            let mut sum = 0;
            for k in 0..AES_ROWS {
                // Entry (row, k) of the circulant matrix is the first row rotated right by `row`.
                let coefficient = coefficients[(k + AES_ROWS - row) % AES_ROWS];
                sum ^= gf_multiply(coefficient, state[k + AES_ROWS * column]);
            }
            result[row + AES_ROWS * column] = sum;
        }
    }
    result
}

fn xor_blocks(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
    core::array::from_fn(|k| a[k] ^ b[k])
}

/// aesenc / aesenclast: one encryption round and the AddRoundKey with `round_key`; the last
/// round has no MixColumns.
pub(super) fn aes_encrypt_round(state: u128, round_key: u128, last: bool) -> u128 {
    let mut block = shift_rows(sub_bytes(state.to_le_bytes(), &AES_SBOX), false);
    if !last {
        block = mix_columns(block, false);
    }
    u128::from_le_bytes(xor_blocks(block, round_key.to_le_bytes()))
}

/// aesdec / aesdeclast: one round of the equivalent inverse cipher (InvShiftRows, InvSubBytes,
/// then InvMixColumns unless last, then AddRoundKey).
pub(super) fn aes_decrypt_round(state: u128, round_key: u128, last: bool) -> u128 {
    let mut block = sub_bytes(shift_rows(state.to_le_bytes(), true), &AES_INVERSE_SBOX);
    if !last {
        block = mix_columns(block, true);
    }
    u128::from_le_bytes(xor_blocks(block, round_key.to_le_bytes()))
}

/// aesimc: InvMixColumns of a round key, to prepare it for the equivalent inverse cipher.
pub(super) fn aes_inverse_mix_columns(state: u128) -> u128 {
    u128::from_le_bytes(mix_columns(state.to_le_bytes(), true))
}

/// aeskeygenassist: SubWord of dwords 1 and 3 of the source, the odd results also rotated by a
/// byte and XORed with the round constant.
pub(super) fn aes_key_generation_assist(source: u128, round_constant: u8) -> u128 {
    let substitute =
        |word: u32| u32::from_le_bytes(word.to_le_bytes().map(|b| AES_SBOX[b as usize]));
    let word = |index: u32| (source >> (32 * index)) as u32;
    let (x1, x3) = (substitute(word(1)), substitute(word(3)));
    let rotated = |value: u32| value.rotate_right(8) ^ round_constant as u32;
    (x1 as u128)
        | ((rotated(x1) as u128) << 32)
        | ((x3 as u128) << 64)
        | ((rotated(x3) as u128) << 96)
}

/// pclmulqdq's multiplication: the 128-bit carry-less (GF(2)[x]) product of two 64-bit values.
/// Guests run CRC32 and GHASH over this one instruction (the kernel's crc32 on a Zen 4 CPU model
/// does it on 512-bit vectors), so it uses the host's own carry-less multiply where there is one.
pub(super) fn carry_less_multiply(a: u64, b: u64) -> u128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // `pmull` is part of the `aes` feature, which Apple Silicon targets enable by default.
        // SAFETY: the feature the intrinsic needs is enabled for the whole build.
        return unsafe { core::arch::aarch64::vmull_p64(a, b) };
    }
    #[cfg(all(target_arch = "x86_64", not(freestanding)))]
    {
        if std::arch::is_x86_feature_detected!("pclmulqdq") {
            // SAFETY: the CPU has the instruction, just checked.
            return unsafe { carry_less_multiply_pclmulqdq(a, b) };
        }
    }
    #[allow(unreachable_code)]
    carry_less_multiply_portable(a, b)
}

#[cfg(all(target_arch = "x86_64", not(freestanding)))]
#[target_feature(enable = "pclmulqdq", enable = "sse2")]
unsafe fn carry_less_multiply_pclmulqdq(a: u64, b: u64) -> u128 {
    use core::arch::x86_64::{_mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_storeu_si128};
    let mut bytes = [0u8; 16];
    // SAFETY: the caller checked for pclmulqdq; `bytes` holds the 16 bytes the store writes.
    unsafe {
        let product =
            _mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(a as i64), _mm_cvtsi64_si128(b as i64));
        _mm_storeu_si128(bytes.as_mut_ptr().cast(), product);
    }
    u128::from_le_bytes(bytes)
}

/// Bits of `b` consumed per step of the portable multiply.
const CLMUL_WINDOW_BITS: u32 = 4;

/// The multiply for hosts without a carry-less multiply instruction: `a * k` for every 4-bit `k`
/// is tabulated once, then each nibble of `b` adds one table entry, shifted into place.
pub(super) fn carry_less_multiply_portable(a: u64, b: u64) -> u128 {
    const WINDOW_VALUES: usize = 1 << CLMUL_WINDOW_BITS;
    let mut table = [0u128; WINDOW_VALUES];
    for window in 1..WINDOW_VALUES {
        // Each entry is the entry for the window without its lowest set bit, plus `a` shifted by
        // that bit's position.
        let lowest = window.trailing_zeros();
        table[window] = table[window & (window - 1)] ^ ((a as u128) << lowest);
    }
    let mut product = 0u128;
    for step in 0..u64::BITS / CLMUL_WINDOW_BITS {
        let window = (b >> (step * CLMUL_WINDOW_BITS)) as usize & (WINDOW_VALUES - 1);
        product ^= table[window] << (step * CLMUL_WINDOW_BITS);
    }
    product
}

/// Multiplicative inverse in GF(2^8) (a^254); 0 has no inverse and maps to 0, as gf2p8affineinvqb
/// defines it.
pub(in crate::cpu::sse) fn gf_inverse(a: u8) -> u8 {
    /// The multiplicative group has order 255, so a^254 = a^-1.
    const INVERSE_EXPONENT: u32 = 254;
    if a == 0 {
        return 0;
    }
    let (mut result, mut power, mut exponent) = (1u8, a, INVERSE_EXPONENT);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = gf_multiply(result, power);
        }
        power = gf_multiply(power, power);
        exponent >>= 1;
    }
    result
}

/// One byte of gf2p8affineqb: bit `n` of the result is the parity of (matrix byte 7-n AND x) XOR
/// bit `n` of `immediate`, where `matrix` is the 8x8 bit matrix held in one quadword.
pub(in crate::cpu::sse) fn gf_affine_byte(matrix: u64, x: u8, immediate: u8) -> u8 {
    let mut result = 0u8;
    for bit in 0..8 {
        let row = (matrix >> (8 * (7 - bit))) as u8;
        let parity = (row & x).count_ones() as u8 & 1;
        result |= (parity ^ (immediate >> bit & 1)) << bit;
    }
    result
}

/// gf2p8mulb on one 128-bit lane: bytewise GF(2^8) products.
pub(super) fn gf_multiply_bytes(a: u128, b: u128) -> u128 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u128::from_le_bytes(core::array::from_fn(|k| gf_multiply(a[k], b[k])))
}

/// gf2p8affineqb/gf2p8affineinvqb on one 128-bit lane: `data` bytes transformed (after inversion
/// for the `inverse` form) by the matrix in the corresponding quadword of `matrix`.
pub(super) fn gf_affine(data: u128, matrix: u128, immediate: u8, inverse: bool) -> u128 {
    let (data, matrices) = (data.to_le_bytes(), matrix.to_le_bytes());
    u128::from_le_bytes(core::array::from_fn(|k| {
        let quadword = u64::from_le_bytes(matrices[k / 8 * 8..k / 8 * 8 + 8].try_into().unwrap());
        let x = if inverse {
            gf_inverse(data[k])
        } else {
            data[k]
        };
        gf_affine_byte(quadword, x, immediate)
    }))
}

/// Dword `index` of a vector counted from the top: the SHA instructions name their operands
/// `SRC[127:96]` first, so index 0 is bits 127:96 and index 3 is bits 31:0.
fn dword_from_top(vector: u128, index: usize) -> u32 {
    (vector >> (96 - 32 * index)) as u32
}

fn vector_from_dwords(dwords: [u32; 4]) -> u128 {
    dwords
        .into_iter()
        .fold(0, |vector, dword| vector << 32 | dword as u128)
}

/// The boolean function and round constant of SHA-1 rounds 0-19, 20-39, 40-59 and 60-79, which
/// sha1rnds4's imm8[1:0] selects (FIPS 180-4 section 4.1.1; groups 1 and 3 share the parity
/// function).
fn sha1_round_function(group: u8, b: u32, c: u32, d: u32) -> (u32, u32) {
    match group & 3 {
        0 => ((b & c) | (!b & d), 0x5a82_7999),
        1 => (b ^ c ^ d, 0x6ed9_eba1),
        2 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
        _ => (b ^ c ^ d, 0xca62_c1d6),
    }
}

/// sha1rnds4: four SHA-1 rounds. `state` holds A, B, C, D from the top. E is not an operand: software
/// folds it into the first schedule dword with sha1nexte, so round 0 runs with E = 0.
pub(crate) fn sha1_four_rounds(state: u128, schedule: u128, group: u8) -> u128 {
    let [mut a, mut b, mut c, mut d] = core::array::from_fn(|k| dword_from_top(state, k));
    let mut e = 0u32;
    for round in 0..4 {
        let (function, constant) = sha1_round_function(group, b, c, d);
        let next = function
            .wrapping_add(a.rotate_left(5))
            .wrapping_add(dword_from_top(schedule, round))
            .wrapping_add(e)
            .wrapping_add(constant);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = next;
    }
    vector_from_dwords([a, b, c, d])
}

/// sha1nexte: the E for the next four rounds is A of the previous group rotated by 30, added to the
/// top schedule dword; the other three pass through.
pub(crate) fn sha1_next_e(state: u128, schedule: u128) -> u128 {
    let top = dword_from_top(schedule, 0).wrapping_add(dword_from_top(state, 0).rotate_left(30));
    (top as u128) << 96 | schedule & (u128::MAX >> 32)
}

/// sha1msg1: `W[t-16] ^ W[t-14]` for four words, the first half of the message expansion.
pub(crate) fn sha1_message_1(older: u128, newer: u128) -> u128 {
    let w = |k| dword_from_top(older, k);
    vector_from_dwords([
        w(2) ^ w(0),
        w(3) ^ w(1),
        dword_from_top(newer, 0) ^ w(2),
        dword_from_top(newer, 1) ^ w(3),
    ])
}

/// sha1msg2: `rol1(partial ^ W[t-3])`. The last word needs the first one computed on the way, so
/// the words are not independent.
pub(crate) fn sha1_message_2(partial: u128, latest: u128) -> u128 {
    let w16 = (dword_from_top(partial, 0) ^ dword_from_top(latest, 1)).rotate_left(1);
    let w17 = (dword_from_top(partial, 1) ^ dword_from_top(latest, 2)).rotate_left(1);
    let w18 = (dword_from_top(partial, 2) ^ dword_from_top(latest, 3)).rotate_left(1);
    let w19 = (dword_from_top(partial, 3) ^ w16).rotate_left(1);
    vector_from_dwords([w16, w17, w18, w19])
}

/// sha256rnds2: two SHA-256 rounds. `cdgh` is the destination (C, D, G, H from the top), `abef`
/// the source (A, B, E, F) and `schedule` the two message-plus-constant dwords in the low half of
/// xmm0. The result is the new A, B, E, F; C, D, G, H are what software swaps in for the next two
/// rounds.
pub(crate) fn sha256_two_rounds(cdgh: u128, abef: u128, schedule: u128) -> u128 {
    let [mut a, mut b, mut e, mut f] = core::array::from_fn(|k| dword_from_top(abef, k));
    let [mut c, mut d, mut g, mut h] = core::array::from_fn(|k| dword_from_top(cdgh, k));
    for round in 0..2 {
        let schedule_word = (schedule >> (32 * round)) as u32;
        let choose = (e & f) ^ (!e & g);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let temporary = choose
            .wrapping_add(sigma1)
            .wrapping_add(schedule_word)
            .wrapping_add(h);
        let next_a = temporary.wrapping_add(majority).wrapping_add(sigma0);
        let next_e = temporary.wrapping_add(d);
        (h, g, f, e) = (g, f, e, next_e);
        (d, c, b, a) = (c, b, a, next_a);
    }
    vector_from_dwords([a, b, e, f])
}

fn sha256_small_sigma0(word: u32) -> u32 {
    word.rotate_right(7) ^ word.rotate_right(18) ^ (word >> 3)
}

fn sha256_small_sigma1(word: u32) -> u32 {
    word.rotate_right(17) ^ word.rotate_right(19) ^ (word >> 10)
}

/// sha256msg1: `W[t-16] + sigma0(W[t-15])` for four words. The first word of `newer` supplies the
/// one sigma0 input past the end of `older`.
pub(crate) fn sha256_message_1(older: u128, newer: u128) -> u128 {
    let w = |k| dword_from_top(older, k);
    vector_from_dwords([
        w(0).wrapping_add(sha256_small_sigma0(dword_from_top(newer, 3))),
        w(1).wrapping_add(sha256_small_sigma0(w(0))),
        w(2).wrapping_add(sha256_small_sigma0(w(1))),
        w(3).wrapping_add(sha256_small_sigma0(w(2))),
    ])
}

/// sha256msg2: adds `sigma1(W[t-2])` to the partial sums. Words 18 and 19 use words 16 and 17
/// computed on the way, so the words are not independent.
pub(crate) fn sha256_message_2(partial: u128, latest: u128) -> u128 {
    let w16 =
        dword_from_top(partial, 3).wrapping_add(sha256_small_sigma1(dword_from_top(latest, 1)));
    let w17 =
        dword_from_top(partial, 2).wrapping_add(sha256_small_sigma1(dword_from_top(latest, 0)));
    let w18 = dword_from_top(partial, 1).wrapping_add(sha256_small_sigma1(w16));
    let w19 = dword_from_top(partial, 0).wrapping_add(sha256_small_sigma1(w17));
    vector_from_dwords([w19, w18, w17, w16])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition: one shifted copy of `a` per set bit of `b`.
    fn reference(a: u64, b: u64) -> u128 {
        (0..u64::BITS)
            .filter(|bit| (b >> bit) & 1 != 0)
            .fold(0, |product, bit| product ^ ((a as u128) << bit))
    }

    /// Edge values and a deterministic spread of the rest.
    fn operands() -> Vec<u64> {
        let mut values = vec![
            0,
            1,
            2,
            3,
            0x1b,
            u64::MAX,
            1 << 63,
            (1 << 63) | 1,
            0x5555_5555_5555_5555,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..40 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push(state);
        }
        values
    }

    #[test]
    fn the_portable_multiply_is_the_definition() {
        for &a in &operands() {
            for &b in &operands() {
                assert_eq!(
                    carry_less_multiply_portable(a, b),
                    reference(a, b),
                    "{a:#x} {b:#x}"
                );
            }
        }
    }

    #[test]
    fn the_multiply_the_cpu_uses_is_the_definition() {
        for &a in &operands() {
            for &b in &operands() {
                assert_eq!(carry_less_multiply(a, b), reference(a, b), "{a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn carry_less_multiplication_known_products() {
        assert_eq!(carry_less_multiply(3, 3), 5, "(x+1)^2 = x^2 + 1");
        assert_eq!(
            carry_less_multiply(u64::MAX, u64::MAX),
            0x5555_5555_5555_5555_5555_5555_5555_5555
        );
        assert_eq!(carry_less_multiply(1 << 63, 1 << 63), 1 << 126);
    }
}
