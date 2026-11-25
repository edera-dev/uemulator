//! The page codec of format-2 snapshots (`docs/plans/34-incremental-snapshots.md`): a byte-wise
//! LZ77 in the shape of an LZ4 block, written here because the library takes no dependencies.
//! Each page is compressed on its own, so a delta's pages need nothing from the pages around
//! them and the window is the page.
//!
//! A compressed page is a run of sequences. Each starts with a token byte: the high nibble is the
//! number of literals, the low one the match length less `MIN_MATCH`; a nibble of 15 continues in
//! the bytes after it, each added in until one is below 255. The literals follow, then (unless
//! the page ends with them) the match: its distance back as a little-endian `u16`, at least 1 and
//! at most what has been produced. A match may overlap what it produces (distance 1 repeats a
//! byte). The last sequence has literals only and ends exactly at the page's end.
//!
//! The format is the same on every host: the native and WebAssembly builds read each other's
//! snapshots. Changing it means a new snapshot format version.

#[cfg(freestanding)]
use crate::prelude::*;

/// Shortest match worth a token and a distance.
const MIN_MATCH: usize = 4;
/// A nibble of 15 means "more length bytes follow".
const NIBBLE_MORE: usize = 15;
const LENGTH_BYTE_MORE: usize = 255;
/// The last bytes of a page are always literals: a match search reads four bytes ahead.
const LAST_LITERALS: usize = 5;
const HASH_BITS: u32 = 12;
/// Fibonacci hashing of the next four bytes.
const HASH_MULTIPLIER: u32 = 2_654_435_761;
/// After this many misses in a row the search steps faster through incompressible data (what
/// LZ4 calls acceleration), so a page of random bytes costs little more than a copy.
const SKIP_SHIFT: u32 = 5;

/// Reusable compressor state: the hash table of recent positions. Entries carry the generation
/// they were written in, so a new page starts with an empty table without clearing it.
pub(crate) struct Compressor {
    table: Vec<u32>,
    generation: u32,
}

impl Compressor {
    pub fn new() -> Self {
        Compressor {
            table: vec![0; 1 << HASH_BITS],
            generation: 0,
        }
    }

    /// Compresses `input` (at most 64 KiB, distances are 16 bits) into `output`, which is cleared
    /// first. Returns false, with `output` holding junk, when the result would not be shorter
    /// than `limit` bytes: the caller then stores the page as it is.
    pub fn compress(&mut self, input: &[u8], output: &mut Vec<u8>, limit: usize) -> bool {
        debug_assert!(input.len() <= u16::MAX as usize + 1);
        output.clear();
        // Positions are stored as generation << 16 | position, so 65535 pages per clear.
        self.generation = self.generation.wrapping_add(1) & 0xffff;
        if self.generation == 0 {
            self.table.fill(0);
            self.generation = 1;
        }
        let tag = self.generation << 16;
        let length = input.len();
        let mut literal_start = 0;
        let mut position = 0;
        let mut misses = 0u32;
        let search_end = length.saturating_sub(LAST_LITERALS);
        while position + MIN_MATCH <= search_end {
            let word = read_u32(input, position);
            let slot = (word.wrapping_mul(HASH_MULTIPLIER) >> (32 - HASH_BITS)) as usize;
            let entry = self.table[slot];
            self.table[slot] = tag | position as u32;
            let candidate = (entry & 0xffff) as usize;
            if entry & !0xffff != tag || candidate >= position || read_u32(input, candidate) != word
            {
                misses += 1;
                position += 1 + (misses >> SKIP_SHIFT) as usize;
                continue;
            }
            misses = 0;
            let mut match_length = MIN_MATCH;
            while position + match_length < search_end
                && input[candidate + match_length] == input[position + match_length]
            {
                match_length += 1;
            }
            emit_sequence(
                output,
                &input[literal_start..position],
                Some((position - candidate, match_length)),
            );
            if output.len() >= limit {
                return false;
            }
            position += match_length;
            literal_start = position;
        }
        emit_sequence(output, &input[literal_start..], None);
        output.len() < limit
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn emit_sequence(output: &mut Vec<u8>, literals: &[u8], matched: Option<(usize, usize)>) {
    let literal_nibble = literals.len().min(NIBBLE_MORE);
    let match_extra = matched.map_or(0, |(_, length)| length - MIN_MATCH);
    let match_nibble = match_extra.min(NIBBLE_MORE);
    output.push((literal_nibble << 4 | match_nibble) as u8);
    if literal_nibble == NIBBLE_MORE {
        push_length(output, literals.len() - NIBBLE_MORE);
    }
    output.extend_from_slice(literals);
    if let Some((distance, _)) = matched {
        output.extend_from_slice(&(distance as u16).to_le_bytes());
        if match_nibble == NIBBLE_MORE {
            push_length(output, match_extra - NIBBLE_MORE);
        }
    }
}

fn push_length(output: &mut Vec<u8>, mut rest: usize) {
    while rest >= LENGTH_BYTE_MORE {
        output.push(LENGTH_BYTE_MORE as u8);
        rest -= LENGTH_BYTE_MORE;
    }
    output.push(rest as u8);
}

/// Why a compressed page did not decode. The input comes from a file the CRC has not been
/// checked against yet (it is checked at the end of the stream), so every length and distance
/// is checked here and nothing can index out of bounds.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Corrupt;

/// Decompresses `input` into `output`, which it must fill exactly.
pub(crate) fn decompress(input: &[u8], output: &mut [u8]) -> Result<(), Corrupt> {
    let mut read = 0usize;
    let mut written = 0usize;
    loop {
        let token = *input.get(read).ok_or(Corrupt)? as usize;
        read += 1;
        let mut literals = token >> 4;
        if literals == NIBBLE_MORE {
            literals += read_length(input, &mut read)?;
        }
        let literal_end = read.checked_add(literals).ok_or(Corrupt)?;
        let out_end = written.checked_add(literals).ok_or(Corrupt)?;
        output
            .get_mut(written..out_end)
            .ok_or(Corrupt)?
            .copy_from_slice(input.get(read..literal_end).ok_or(Corrupt)?);
        read = literal_end;
        written = out_end;
        if read == input.len() {
            // Only literals end a page, and they must end it exactly.
            return if written == output.len() && token & 0xf == 0 {
                Ok(())
            } else {
                Err(Corrupt)
            };
        }
        let distance_bytes = input.get(read..read + 2).ok_or(Corrupt)?;
        let distance = u16::from_le_bytes([distance_bytes[0], distance_bytes[1]]) as usize;
        read += 2;
        let mut length = (token & 0xf) + MIN_MATCH;
        if token & 0xf == NIBBLE_MORE {
            length += read_length(input, &mut read)?;
        }
        if distance == 0 || distance > written {
            return Err(Corrupt);
        }
        let end = written.checked_add(length).ok_or(Corrupt)?;
        if end > output.len() {
            return Err(Corrupt);
        }
        // Byte by byte: the source may overlap what this copy writes.
        for at in written..end {
            output[at] = output[at - distance];
        }
        written = end;
    }
}

fn read_length(input: &[u8], read: &mut usize) -> Result<usize, Corrupt> {
    let mut total = 0usize;
    loop {
        let byte = *input.get(*read).ok_or(Corrupt)? as usize;
        *read += 1;
        // No page is anywhere near this long; a run of 255s is damage, not a length.
        total = total
            .checked_add(byte)
            .filter(|&t| t <= 1 << 20)
            .ok_or(Corrupt)?;
        if byte != LENGTH_BYTE_MORE {
            return Ok(total);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4096;

    fn round_trip(page: &[u8]) -> Option<usize> {
        let mut compressor = Compressor::new();
        let mut packed = Vec::new();
        if !compressor.compress(page, &mut packed, page.len()) {
            return None;
        }
        let mut unpacked = vec![0xcc; page.len()];
        decompress(&packed, &mut unpacked).expect("decodes");
        assert_eq!(unpacked, page);
        Some(packed.len())
    }

    /// xorshift64*: deterministic bytes for the fuzz-style cases.
    struct Bytes(u64);

    impl Bytes {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
    }

    #[test]
    fn repetitive_pages_shrink_and_come_back() {
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .copied()
            .cycle()
            .take(PAGE)
            .collect();
        assert!(round_trip(&text).unwrap() < 200);
        let mut words = vec![0u8; PAGE];
        for (index, chunk) in words.chunks_mut(8).enumerate() {
            chunk.copy_from_slice(&(0xffff_8880_0000_0000u64 + index as u64 * 64).to_le_bytes());
        }
        // One literal and a seven-byte match per word: about half the size.
        assert!(round_trip(&words).unwrap() < PAGE * 3 / 5);
    }

    #[test]
    fn random_pages_are_refused_rather_than_grown() {
        let mut bytes = Bytes(7);
        let page: Vec<u8> = (0..PAGE).map(|_| bytes.next() as u8).collect();
        assert_eq!(round_trip(&page), None);
    }

    #[test]
    fn every_mix_of_runs_and_noise_round_trips() {
        let mut bytes = Bytes(0x1234_5678);
        let mut compressor = Compressor::new();
        let mut packed = Vec::new();
        for case in 0..3000 {
            let length = match case % 4 {
                0 => PAGE,
                1 => (bytes.next() % 64) as usize,
                _ => (bytes.next() % PAGE as u64) as usize + 1,
            };
            let mut page = vec![0u8; length];
            let mut at = 0;
            while at < length {
                let run = ((bytes.next() % 300) as usize + 1).min(length - at);
                match bytes.next() % 4 {
                    0 => page[at..at + run].fill(bytes.next() as u8),
                    1 if at > 0 => {
                        let distance = (bytes.next() as usize % at) + 1;
                        for index in at..at + run {
                            page[index] = page[index - distance];
                        }
                    }
                    _ => page[at..at + run]
                        .iter_mut()
                        .for_each(|byte| *byte = bytes.next() as u8 & 0x0f),
                }
                at += run;
            }
            // No limit: every page is encoded, so every one must decode.
            assert!(compressor.compress(&page, &mut packed, usize::MAX));
            let mut unpacked = vec![0xcc; length];
            decompress(&packed, &mut unpacked).unwrap_or_else(|_| panic!("case {case}"));
            assert_eq!(unpacked, page, "case {case}");
        }
    }

    #[test]
    fn damaged_input_is_refused_without_panicking() {
        let text: Vec<u8> = (0..PAGE).map(|index| (index / 7) as u8).collect();
        let mut packed = Vec::new();
        assert!(Compressor::new().compress(&text, &mut packed, PAGE));
        let mut output = vec![0; PAGE];
        let mut bytes = Bytes(99);
        for cut in 0..packed.len() {
            assert_eq!(decompress(&packed[..cut], &mut output), Err(Corrupt));
        }
        for _ in 0..20_000 {
            let mut damaged = packed.clone();
            let at = bytes.next() as usize % damaged.len();
            damaged[at] ^= 1 << (bytes.next() % 8);
            // Whatever it decodes to, it must not panic or run past the page.
            let _ = decompress(&damaged, &mut output);
        }
        for _ in 0..2000 {
            let length = bytes.next() as usize % 64;
            let junk: Vec<u8> = (0..length).map(|_| bytes.next() as u8).collect();
            let _ = decompress(&junk, &mut output);
        }
        // A match before anything was written, and one reaching past the page.
        assert_eq!(decompress(&[0x00, 1, 0], &mut output), Err(Corrupt));
        assert_eq!(
            decompress(&[0x1f, b'a', 1, 0, 255, 255, 255, 30], &mut output),
            Err(Corrupt)
        );
        // Literals that stop short of the page.
        assert_eq!(decompress(&[0x10, b'a'], &mut output), Err(Corrupt));
    }

    #[test]
    fn many_pages_through_one_compressor_wrap_its_generation() {
        let mut compressor = Compressor::new();
        let page: Vec<u8> = (0..PAGE).map(|index| (index % 251) as u8).collect();
        let mut packed = Vec::new();
        compressor.generation = 0xfffe;
        for _ in 0..4 {
            assert!(compressor.compress(&page, &mut packed, PAGE));
            let mut unpacked = vec![0; PAGE];
            decompress(&packed, &mut unpacked).unwrap();
            assert_eq!(unpacked, page);
        }
    }
}
