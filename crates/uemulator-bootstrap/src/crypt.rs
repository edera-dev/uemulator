//! SHA-512 crypt (`$6$`), Ulrich Drepper's "Unix crypt using SHA-256 and SHA-512", at the default
//! 5000 rounds. The output never carries a `rounds=` field, so every verifier assumes the default.

use uemulator_crypto::sha::{SHA512_DIGEST_BYTES, Sha512};

/// Longest salt crypt reads; the rest is dropped, as glibc does.
pub const SALT_LEN: usize = 16;

const SALT_PREFIX: &str = "$6$";
const DEFAULT_ROUNDS: usize = 5000;
/// Round counters taken modulo these decide which inputs a round mixes in (step 21 of the spec).
const SALT_ROUND_PERIOD: usize = 3;
const PASSWORD_ROUND_PERIOD: usize = 7;
/// Crypt's own base 64 alphabet, which is not RFC 4648's.
const ALPHABET: &[u8; 64] = b"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const BITS_PER_CHARACTER: usize = 6;
const CHARACTER_MASK: u32 = (1 << BITS_PER_CHARACTER) - 1;

/// Digest byte indices (high, middle, low byte of each 24-bit group), in the spec's output order.
/// The last digest byte, 63, goes out alone after these.
const OUTPUT_GROUPS: [[usize; 3]; 21] = [
    [0, 21, 42],
    [22, 43, 1],
    [44, 2, 23],
    [3, 24, 45],
    [25, 46, 4],
    [47, 5, 26],
    [6, 27, 48],
    [28, 49, 7],
    [50, 8, 29],
    [9, 30, 51],
    [31, 52, 10],
    [53, 11, 32],
    [12, 33, 54],
    [34, 55, 13],
    [56, 14, 35],
    [15, 36, 57],
    [37, 58, 16],
    [59, 17, 38],
    [18, 39, 60],
    [40, 61, 19],
    [62, 20, 41],
];
const LAST_DIGEST_BYTE: usize = SHA512_DIGEST_BYTES - 1;
/// The final lone byte fills two characters (12 bits), not four.
const LAST_GROUP_CHARACTERS: usize = 2;
const GROUP_CHARACTERS: usize = 4;

/// Maps random bytes onto the salt alphabet. `byte % 64` is uniform because 64 divides 256.
pub fn salt_from_bytes(random: &[u8; SALT_LEN]) -> String {
    random
        .iter()
        .map(|byte| ALPHABET[usize::from(*byte) % ALPHABET.len()] as char)
        .collect()
}

/// Hashes `password` into a `$6$salt$hash` string that `/etc/shadow` accepts. `salt` is cut to
/// [`SALT_LEN`] bytes; callers pass characters from `./0-9A-Za-z`.
pub fn sha512_crypt(password: &str, salt: &str) -> String {
    let password = password.as_bytes();
    let salt = &salt.as_bytes()[..salt.len().min(SALT_LEN)];

    // Digest B (steps 4 to 8).
    let mut alternate = Sha512::new();
    alternate.update(password);
    alternate.update(salt);
    alternate.update(password);
    let alternate = alternate.finish();

    // Digest A (steps 1 to 3, 9 to 12).
    let mut start = Sha512::new();
    start.update(password);
    start.update(salt);
    update_repeated(&mut start, &alternate, password.len());
    let mut remaining = password.len();
    while remaining > 0 {
        if remaining & 1 != 0 {
            start.update(&alternate);
        } else {
            start.update(password);
        }
        remaining >>= 1;
    }
    let start = start.finish();

    // Byte sequences P and S (steps 13 to 20).
    let mut password_digest = Sha512::new();
    for _ in 0..password.len() {
        password_digest.update(password);
    }
    let password_sequence = repeat_to_length(&password_digest.finish(), password.len());
    let mut salt_digest = Sha512::new();
    for _ in 0..(16 + usize::from(start[0])) {
        salt_digest.update(salt);
    }
    let salt_sequence = repeat_to_length(&salt_digest.finish(), salt.len());

    // Step 21.
    let mut digest = start;
    for round in 0..DEFAULT_ROUNDS {
        let mut hasher = Sha512::new();
        if round & 1 != 0 {
            hasher.update(&password_sequence);
        } else {
            hasher.update(&digest);
        }
        if round % SALT_ROUND_PERIOD != 0 {
            hasher.update(&salt_sequence);
        }
        if round % PASSWORD_ROUND_PERIOD != 0 {
            hasher.update(&password_sequence);
        }
        if round & 1 != 0 {
            hasher.update(&digest);
        } else {
            hasher.update(&password_sequence);
        }
        digest = hasher.finish();
    }

    let mut output = String::from(SALT_PREFIX);
    output.push_str(core::str::from_utf8(salt).unwrap_or_default());
    output.push('$');
    for [high, middle, low] in OUTPUT_GROUPS {
        push_group(
            &mut output,
            digest[high],
            digest[middle],
            digest[low],
            GROUP_CHARACTERS,
        );
    }
    push_group(
        &mut output,
        0,
        0,
        digest[LAST_DIGEST_BYTE],
        LAST_GROUP_CHARACTERS,
    );
    output
}

/// Feeds the first `length` bytes of `source` repeated as often as needed.
fn update_repeated(hasher: &mut Sha512, source: &[u8], length: usize) {
    for byte_count in repeat_lengths(source.len(), length) {
        hasher.update(&source[..byte_count]);
    }
}

fn repeat_lengths(chunk: usize, total: usize) -> impl Iterator<Item = usize> {
    let mut remaining = total;
    core::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let take = remaining.min(chunk);
        remaining -= take;
        Some(take)
    })
}

fn repeat_to_length(source: &[u8], length: usize) -> Vec<u8> {
    source.iter().copied().cycle().take(length).collect()
}

/// Low six bits come out first.
fn push_group(output: &mut String, high: u8, middle: u8, low: u8, characters: usize) {
    let mut group = u32::from(high) << 16 | u32::from(middle) << 8 | u32::from(low);
    for _ in 0..characters {
        output.push(ALPHABET[(group & CHARACTER_MASK) as usize] as char);
        group >>= BITS_PER_CHARACTER;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_drepper_vectors() {
        // tests2[] in SHA-crypt.txt. The salt of the second is longer than 16 bytes, and its
        // `rounds=5000` is the default, so the same hash comes out without the field.
        assert_eq!(
            sha512_crypt("Hello world!", "saltstring"),
            concat!(
                "$6$saltstring$svn8UoSVapNtMuq1ukKS4tPQd8iKwSMHWjl/O817G3uBnIFNjnQJu",
                "esI68u4OTLiBFdcbYEdFCoEOfaS35inz1"
            )
        );
        assert_eq!(
            sha512_crypt("This is just a test", "toolongsaltstring"),
            concat!(
                "$6$toolongsaltstrin$lQ8jolhgVRVhY4b5pZKaysCLi0QBxGoNeKQ",
                "zQ3glMhwllF7oGDZxUhx1yxdYcz/e1JSbq3y6JMxxl8audkUEm0"
            )
        );
    }

    #[test]
    fn salt_is_truncated_to_sixteen() {
        let hash = sha512_crypt("pw", "0123456789abcdefOVERFLOW");
        assert!(hash.starts_with("$6$0123456789abcdef$"));
    }

    #[test]
    fn salt_alphabet_only() {
        let all: Vec<u8> = (0..=255).collect();
        for chunk in all.chunks(SALT_LEN) {
            let mut random = [0u8; SALT_LEN];
            random[..chunk.len()].copy_from_slice(chunk);
            let salt = salt_from_bytes(&random);
            assert_eq!(salt.len(), SALT_LEN);
            assert!(salt.bytes().all(|byte| ALPHABET.contains(&byte)));
        }
    }
}
