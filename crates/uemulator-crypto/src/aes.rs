//! AES (FIPS 197) and the CFB mode the TPM encrypts parameters and sensitive areas with.
//! Byte-at-a-time with an S-box built at compile time: the TPM pushes at most a few
//! kilobytes through it per command, so tables of 32-bit words would buy nothing that
//! matters.

use crate::Error;

pub const BLOCK_BYTES: usize = 16;
/// Words (columns) in the state.
const COLUMNS: usize = 4;
const WORD_BYTES: usize = 4;
/// AES-256: 14 rounds, plus the initial key addition.
const MAX_ROUND_KEYS: usize = 15;
/// x^8 + x^4 + x^3 + x + 1 without the x^8 term: what doubling an element folds back in.
const REDUCTION_POLYNOMIAL: u8 = 0x1b;
/// The affine transform's constant (FIPS 197 5.1.1).
const SBOX_AFFINE_CONSTANT: u8 = 0x63;
const MULTIPLICATIVE_INVERSE_EXPONENT: u32 = 254;

/// Multiplication in GF(2^8) modulo the AES polynomial.
const fn gf_multiply(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        a = double(a);
        b >>= 1;
    }
    product
}

const fn double(value: u8) -> u8 {
    (value << 1)
        ^ if value & 0x80 != 0 {
            REDUCTION_POLYNOMIAL
        } else {
            0
        }
}

const fn build_sbox() -> [u8; 256] {
    let mut sbox = [0u8; 256];
    let mut value = 0;
    while value < 256 {
        // The inverse is value^254, the multiplicative group having order 255, and 0^254
        // is the 0 the S-box wants for 0.
        let mut inverse = 1u8;
        let mut square = value as u8;
        let mut exponent = MULTIPLICATIVE_INVERSE_EXPONENT;
        while exponent != 0 {
            if exponent & 1 != 0 {
                inverse = gf_multiply(inverse, square);
            }
            square = gf_multiply(square, square);
            exponent >>= 1;
        }
        sbox[value] = inverse
            ^ inverse.rotate_left(1)
            ^ inverse.rotate_left(2)
            ^ inverse.rotate_left(3)
            ^ inverse.rotate_left(4)
            ^ SBOX_AFFINE_CONSTANT;
        value += 1;
    }
    sbox
}

const fn invert_table(table: &[u8; 256]) -> [u8; 256] {
    let mut inverse = [0u8; 256];
    let mut index = 0;
    while index < 256 {
        inverse[table[index] as usize] = index as u8;
        index += 1;
    }
    inverse
}

const SBOX: [u8; 256] = build_sbox();
const INVERSE_SBOX: [u8; 256] = invert_table(&SBOX);

/// An expanded AES-128, -192 or -256 key.
#[derive(Clone)]
pub struct Aes {
    round_keys: [[u8; BLOCK_BYTES]; MAX_ROUND_KEYS],
    rounds: usize,
}

impl core::fmt::Debug for Aes {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Aes")
            .field("rounds", &self.rounds)
            .finish_non_exhaustive()
    }
}

impl Aes {
    /// A 16-, 24- or 32-byte key; anything else is `Unsupported`.
    pub fn new(key: &[u8]) -> Result<Self, Error> {
        if !matches!(key.len(), 16 | 24 | 32) {
            return Err(Error::Unsupported);
        }
        let key_words = key.len() / WORD_BYTES;
        let rounds = key_words + 6;
        let total_words = COLUMNS * (rounds + 1);
        let mut words = [[0u8; WORD_BYTES]; COLUMNS * MAX_ROUND_KEYS];
        for (word, chunk) in words.iter_mut().zip(key.as_chunks::<WORD_BYTES>().0) {
            *word = *chunk;
        }
        let mut round_constant = 1u8;
        for index in key_words..total_words {
            let mut word = words[index - 1];
            if index % key_words == 0 {
                word.rotate_left(1);
                word = word.map(|byte| SBOX[byte as usize]);
                word[0] ^= round_constant;
                round_constant = double(round_constant);
            } else if key_words > 6 && index % key_words == 4 {
                // AES-256 only (FIPS 197 5.2).
                word = word.map(|byte| SBOX[byte as usize]);
            }
            for (byte, earlier) in word.iter_mut().zip(words[index - key_words]) {
                *byte ^= earlier;
            }
            words[index] = word;
        }
        let mut round_keys = [[0u8; BLOCK_BYTES]; MAX_ROUND_KEYS];
        for (round_key, chunk) in round_keys.iter_mut().zip(words.as_chunks::<COLUMNS>().0) {
            for (column, word) in chunk.iter().enumerate() {
                round_key[column * WORD_BYTES..][..WORD_BYTES].copy_from_slice(word);
            }
        }
        Ok(Self { round_keys, rounds })
    }

    pub fn encrypt_block(&self, block: &mut [u8; BLOCK_BYTES]) {
        add_round_key(block, &self.round_keys[0]);
        for round in 1..=self.rounds {
            for byte in block.iter_mut() {
                *byte = SBOX[*byte as usize];
            }
            shift_rows(block);
            if round != self.rounds {
                mix_columns(block);
            }
            add_round_key(block, &self.round_keys[round]);
        }
    }

    pub fn decrypt_block(&self, block: &mut [u8; BLOCK_BYTES]) {
        add_round_key(block, &self.round_keys[self.rounds]);
        for round in (0..self.rounds).rev() {
            inverse_shift_rows(block);
            for byte in block.iter_mut() {
                *byte = INVERSE_SBOX[*byte as usize];
            }
            add_round_key(block, &self.round_keys[round]);
            if round != 0 {
                inverse_mix_columns(block);
            }
        }
    }
}

fn add_round_key(block: &mut [u8; BLOCK_BYTES], round_key: &[u8; BLOCK_BYTES]) {
    for (byte, key) in block.iter_mut().zip(round_key) {
        *byte ^= key;
    }
}

/// The state is column-major, as the input bytes come: byte `row + 4 * column`. Row r
/// rotates left by r.
fn shift_rows(block: &mut [u8; BLOCK_BYTES]) {
    let old = *block;
    for row in 0..WORD_BYTES {
        for column in 0..COLUMNS {
            block[row + WORD_BYTES * column] = old[row + WORD_BYTES * ((column + row) % COLUMNS)];
        }
    }
}

fn inverse_shift_rows(block: &mut [u8; BLOCK_BYTES]) {
    let old = *block;
    for row in 0..WORD_BYTES {
        for column in 0..COLUMNS {
            block[row + WORD_BYTES * ((column + row) % COLUMNS)] = old[row + WORD_BYTES * column];
        }
    }
}

/// Multiplies each column by the circulant matrix whose first row is `coefficients`.
fn multiply_columns(block: &mut [u8; BLOCK_BYTES], coefficients: [u8; WORD_BYTES]) {
    for column in block.as_chunks_mut::<WORD_BYTES>().0 {
        let old = *column;
        for (row, byte) in column.iter_mut().enumerate() {
            *byte = (0..WORD_BYTES).fold(0, |sum, index| {
                sum ^ gf_multiply(
                    coefficients[(index + WORD_BYTES - row) % WORD_BYTES],
                    old[index],
                )
            });
        }
    }
}

/// FIPS 197 5.1.3.
fn mix_columns(block: &mut [u8; BLOCK_BYTES]) {
    multiply_columns(block, [2, 3, 1, 1]);
}

/// FIPS 197 5.3.3.
fn inverse_mix_columns(block: &mut [u8; BLOCK_BYTES]) {
    multiply_columns(block, [14, 11, 13, 9]);
}

/// CFB with 128-bit feedback (SP800-38A 6.3, s = 128), as TPM 2.0 uses it: the last block
/// may be partial and takes the leading bytes of its keystream block.
pub fn cfb_encrypt(key: &[u8], iv: &[u8; BLOCK_BYTES], data: &mut [u8]) -> Result<(), Error> {
    let aes = Aes::new(key)?;
    let mut feedback = *iv;
    for chunk in data.chunks_mut(BLOCK_BYTES) {
        aes.encrypt_block(&mut feedback);
        for (byte, keystream) in chunk.iter_mut().zip(&mut feedback) {
            *byte ^= *keystream;
            *keystream = *byte;
        }
    }
    Ok(())
}

pub fn cfb_decrypt(key: &[u8], iv: &[u8; BLOCK_BYTES], data: &mut [u8]) -> Result<(), Error> {
    let aes = Aes::new(key)?;
    let mut feedback = *iv;
    for chunk in data.chunks_mut(BLOCK_BYTES) {
        aes.encrypt_block(&mut feedback);
        for (byte, keystream) in chunk.iter_mut().zip(&mut feedback) {
            let ciphertext = *byte;
            *byte ^= *keystream;
            *keystream = ciphertext;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
            .collect()
    }

    fn block(text: &str) -> [u8; BLOCK_BYTES] {
        hex(text).try_into().unwrap()
    }

    #[test]
    fn sbox_matches_fips_197() {
        // Figure 7's corners and the entry worked through in 5.1.1.
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0xff], 0x16);
        assert_eq!(INVERSE_SBOX[0x63], 0x00);
    }

    #[test]
    fn fips_197_appendix_c_blocks() {
        let plaintext = block("00112233445566778899aabbccddeeff");
        for (key_bytes, expected) in [
            (16u8, "69c4e0d86a7b0430d8cdb78070b4c55a"),
            (24, "dda97ca4864cdfe06eaf70a0ec0d7191"),
            (32, "8ea2b7ca516745bfeafc49904b496089"),
        ] {
            let key: Vec<u8> = (0..key_bytes).collect();
            let aes = Aes::new(&key).unwrap();
            let mut data = plaintext;
            aes.encrypt_block(&mut data);
            assert_eq!(data, block(expected), "AES-{}", key_bytes as usize * 8);
            aes.decrypt_block(&mut data);
            assert_eq!(data, plaintext);
        }
    }

    #[test]
    fn key_lengths_are_checked() {
        for length in [0, 15, 17, 20, 31, 33, 64] {
            assert!(matches!(
                Aes::new(&vec![0; length]),
                Err(Error::Unsupported)
            ));
            assert_eq!(
                cfb_encrypt(&vec![0; length], &[0; 16], &mut [0; 4]),
                Err(Error::Unsupported)
            );
        }
    }

    const SP800_38A_KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
    const SP800_38A_PLAINTEXT: &str = "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710";
    const SP800_38A_CIPHERTEXT: &str = "3b3fd92eb72dad20333449f8e83cfb4ac8a64537a0b3a93fcde3cdad9f1ce58b26751f67a3cbb140b1808cf187a4f4dfc04b05357c5d1c0eeac4c66f9ff7f2e6";

    #[test]
    fn sp800_38a_cfb128_aes128() {
        let key = hex(SP800_38A_KEY);
        let iv = block("000102030405060708090a0b0c0d0e0f");
        // F.3.13
        let mut data = hex(SP800_38A_PLAINTEXT);
        cfb_encrypt(&key, &iv, &mut data).unwrap();
        assert_eq!(data, hex(SP800_38A_CIPHERTEXT));
        // F.3.14
        cfb_decrypt(&key, &iv, &mut data).unwrap();
        assert_eq!(data, hex(SP800_38A_PLAINTEXT));
    }

    #[test]
    fn partial_last_blocks_match_python_cryptography() {
        // `hack/crypto/make-tpm-fixtures.py`: 37 bytes (two blocks and five), IV f0..ff.
        let plaintext: Vec<u8> = (0..37u8).map(|index| index.wrapping_mul(7) + 3).collect();
        let iv: [u8; 16] = core::array::from_fn(|index| 0xf0 + index as u8);
        let aes_256_key: Vec<u8> = (0..32).collect();
        for (key, expected) in [
            (
                hex(SP800_38A_KEY),
                "ef86ce6b87465184c9905f25bdc0c488c5a72ea6f0def560f2ab2cac550f7fd52ab27c4e9c",
            ),
            (
                aes_256_key,
                "910adc953cb0adff612baf04176c067841e38461a58b8213e0f6ac834727289cfc28923cab",
            ),
        ] {
            let mut data = plaintext.clone();
            cfb_encrypt(&key, &iv, &mut data).unwrap();
            assert_eq!(data, hex(expected));
            cfb_decrypt(&key, &iv, &mut data).unwrap();
            assert_eq!(data, plaintext);
        }
        let mut empty = [];
        cfb_encrypt(&hex(SP800_38A_KEY), &iv, &mut empty).unwrap();
    }
}
