//! gzip (RFC 1952) around DEFLATE (RFC 1951), for a hypervisor handed over as it is shipped
//! (`xen.gz`). The library has no dependencies, so this is a small decoder of its own: canonical
//! Huffman codes decoded a bit at a time, which is plenty for a few MiB once per boot. Malformed
//! input is an error, never a panic: the bytes come from whoever configured the machine.

#[cfg(freestanding)]
use crate::prelude::*;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const GZIP_METHOD_DEFLATE: u8 = 8;
/// ID1, ID2, CM, FLG, MTIME u32, XFL, OS.
const GZIP_HEADER_BYTES: usize = 10;
const GZIP_FLAG_HEADER_CRC: u8 = 1 << 1;
const GZIP_FLAG_EXTRA: u8 = 1 << 2;
const GZIP_FLAG_NAME: u8 = 1 << 3;
const GZIP_FLAG_COMMENT: u8 = 1 << 4;
/// CRC-32 of the data, then its length modulo 2^32.
const GZIP_TRAILER_BYTES: usize = 8;

const BLOCK_STORED: u32 = 0;
const BLOCK_FIXED: u32 = 1;
const BLOCK_DYNAMIC: u32 = 2;
const END_OF_BLOCK: u16 = 256;
/// Codes are at most 15 bits long.
const MAX_CODE_BITS: usize = 15;
const LITERAL_LENGTH_SYMBOLS: usize = 288;
const DISTANCE_SYMBOLS: usize = 30;
const CODE_LENGTH_SYMBOLS: usize = 19;
/// RFC 1951 3.2.7: the order the code length code lengths are stored in.
const CODE_LENGTH_ORDER: [usize; CODE_LENGTH_SYMBOLS] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
/// Lengths 257..=285: base and extra bits (RFC 1951 3.2.5).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; DISTANCE_SYMBOLS] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; DISTANCE_SYMBOLS] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

pub(crate) fn is_gzip(data: &[u8]) -> bool {
    data.starts_with(&GZIP_MAGIC)
}

/// The data inside a single-member gzip file, refused when it would grow past `limit` bytes
/// or when the trailer's CRC or length does not match.
pub(crate) fn gunzip(data: &[u8], limit: usize) -> Result<Vec<u8>, &'static str> {
    const TRUNCATED: &str = "truncated gzip header";
    if data.len() < GZIP_HEADER_BYTES || !is_gzip(data) {
        return Err("not a gzip file");
    }
    if data[2] != GZIP_METHOD_DEFLATE {
        return Err("gzip compression method is not deflate");
    }
    let flags = data[3];
    let mut at = GZIP_HEADER_BYTES;
    if flags & GZIP_FLAG_EXTRA != 0 {
        let length = data.get(at..at + 2).ok_or(TRUNCATED)?;
        at += 2 + u16::from_le_bytes([length[0], length[1]]) as usize;
    }
    for flag in [GZIP_FLAG_NAME, GZIP_FLAG_COMMENT] {
        if flags & flag != 0 {
            let rest = data.get(at..).ok_or(TRUNCATED)?;
            at += rest.iter().position(|&byte| byte == 0).ok_or(TRUNCATED)? + 1;
        }
    }
    if flags & GZIP_FLAG_HEADER_CRC != 0 {
        at += 2;
    }
    let body = data.get(at..).ok_or(TRUNCATED)?;
    let (out, used) = inflate(body, limit)?;
    let trailer = body
        .get(used..used + GZIP_TRAILER_BYTES)
        .ok_or("truncated gzip trailer")?;
    let crc = u32::from_le_bytes(trailer[..4].try_into().unwrap());
    let length = u32::from_le_bytes(trailer[4..].try_into().unwrap());
    if length != out.len() as u32 || crc != crate::kv::crc32(&out) {
        return Err("gzip data does not match its checksum");
    }
    Ok(out)
}

struct Bits<'a> {
    data: &'a [u8],
    /// Next byte to load into `buffer`.
    at: usize,
    buffer: u64,
    count: u32,
}

impl Bits<'_> {
    fn take(&mut self, wanted: u32) -> Result<u32, &'static str> {
        while self.count < wanted {
            let byte = *self.data.get(self.at).ok_or("truncated deflate data")?;
            self.buffer |= (byte as u64) << self.count;
            self.at += 1;
            self.count += 8;
        }
        let value = (self.buffer & ((1u64 << wanted) - 1)) as u32;
        self.buffer >>= wanted;
        self.count -= wanted;
        Ok(value)
    }

    /// Drops the rest of the current byte (stored blocks start on a byte boundary).
    fn align(&mut self) {
        let partial = self.count % 8;
        self.buffer >>= partial;
        self.count -= partial;
    }

    /// Bytes of the input consumed so far, whole bytes still in the buffer given back.
    fn consumed(&self) -> usize {
        self.at - (self.count / 8) as usize
    }
}

/// A canonical Huffman code as counts of codes per length and the symbols in code order, which
/// is all decoding needs (RFC 1951 3.2.2).
struct Huffman {
    counts: [u16; MAX_CODE_BITS + 1],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self, &'static str> {
        let mut counts = [0u16; MAX_CODE_BITS + 1];
        for &length in lengths {
            counts[length as usize] += 1;
        }
        counts[0] = 0;
        // An over-subscribed code has more codes of some length than the shorter ones leave
        // room for; an incomplete one is legal (a single distance code is).
        let mut left = 1i32;
        for &count in &counts[1..] {
            left = (left << 1) - count as i32;
            if left < 0 {
                return Err("over-subscribed Huffman code");
            }
        }
        let mut offsets = [0u16; MAX_CODE_BITS + 2];
        for length in 1..=MAX_CODE_BITS {
            offsets[length + 1] = offsets[length] + counts[length];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length != 0 {
                symbols[offsets[length as usize] as usize] = symbol as u16;
                offsets[length as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, &'static str> {
        // Codes are packed most significant bit first, unlike everything else in the stream.
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for &count in &self.counts[1..] {
            code |= bits.take(1)? as i32;
            let count = count as i32;
            if code - first < count {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("invalid Huffman code")
    }
}

fn fixed_codes() -> Result<(Huffman, Huffman), &'static str> {
    let mut lengths = [0u8; LITERAL_LENGTH_SYMBOLS];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Ok((
        Huffman::new(&lengths)?,
        Huffman::new(&[5; DISTANCE_SYMBOLS])?,
    ))
}

fn dynamic_codes(bits: &mut Bits) -> Result<(Huffman, Huffman), &'static str> {
    const MAX_LITERAL_LENGTH_CODES: usize = 286;
    const REPEAT_PREVIOUS: u16 = 16;
    const REPEAT_ZERO_SHORT: u16 = 17;
    const REPEAT_ZERO_LONG: u16 = 18;
    let literal_count = bits.take(5)? as usize + 257;
    let distance_count = bits.take(5)? as usize + 1;
    let code_length_count = bits.take(4)? as usize + 4;
    if literal_count > MAX_LITERAL_LENGTH_CODES || distance_count > DISTANCE_SYMBOLS {
        return Err("too many codes in a dynamic block");
    }
    let mut code_lengths = [0u8; CODE_LENGTH_SYMBOLS];
    for &symbol in &CODE_LENGTH_ORDER[..code_length_count] {
        code_lengths[symbol] = bits.take(3)? as u8;
    }
    let code_length_code = Huffman::new(&code_lengths)?;
    let total = literal_count + distance_count;
    let mut lengths = vec![0u8; total];
    let mut filled = 0;
    while filled < total {
        let symbol = code_length_code.decode(bits)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol as u8, 1),
            REPEAT_PREVIOUS => {
                let previous = *lengths[..filled]
                    .last()
                    .ok_or("length repeat with nothing to repeat")?;
                (previous, 3 + bits.take(2)? as usize)
            }
            REPEAT_ZERO_SHORT => (0, 3 + bits.take(3)? as usize),
            REPEAT_ZERO_LONG => (0, 11 + bits.take(7)? as usize),
            _ => return Err("invalid code length symbol"),
        };
        let run = lengths
            .get_mut(filled..filled + repeat)
            .ok_or("code lengths overflow the table")?;
        run.fill(value);
        filled += repeat;
    }
    if lengths[END_OF_BLOCK as usize] == 0 {
        return Err("dynamic block without an end-of-block code");
    }
    Ok((
        Huffman::new(&lengths[..literal_count])?,
        Huffman::new(&lengths[literal_count..])?,
    ))
}

fn inflate_block(
    bits: &mut Bits,
    out: &mut Vec<u8>,
    limit: usize,
    literals: &Huffman,
    distances: &Huffman,
) -> Result<(), &'static str> {
    loop {
        let symbol = literals.decode(bits)?;
        match symbol {
            0..=255 => out.push(symbol as u8),
            END_OF_BLOCK => return Ok(()),
            _ => {
                let index = (symbol - 257) as usize;
                let (Some(&base), Some(&extra)) = (LENGTH_BASE.get(index), LENGTH_EXTRA.get(index))
                else {
                    return Err("invalid length symbol");
                };
                let length = base as usize + bits.take(extra as u32)? as usize;
                let index = distances.decode(bits)? as usize;
                let (Some(&base), Some(&extra)) =
                    (DISTANCE_BASE.get(index), DISTANCE_EXTRA.get(index))
                else {
                    return Err("invalid distance symbol");
                };
                let distance = base as usize + bits.take(extra as u32)? as usize;
                if distance > out.len() {
                    return Err("distance reaches before the start of the data");
                }
                // Byte by byte: the source may overlap the bytes being written (a run).
                let start = out.len() - distance;
                for offset in 0..length {
                    out.push(out[start + offset]);
                }
            }
        }
        if out.len() > limit {
            return Err("decompressed data is too large");
        }
    }
}

/// Decodes a raw DEFLATE stream. Returns the data and how many input bytes the stream took.
fn inflate(data: &[u8], limit: usize) -> Result<(Vec<u8>, usize), &'static str> {
    let mut bits = Bits {
        data,
        at: 0,
        buffer: 0,
        count: 0,
    };
    let mut out = Vec::new();
    loop {
        let last = bits.take(1)? == 1;
        match bits.take(2)? {
            BLOCK_STORED => {
                bits.align();
                let length = bits.take(16)?;
                let complement = bits.take(16)?;
                if length != !complement & 0xffff {
                    return Err("stored block length does not match its complement");
                }
                // `align` left whole bytes only, and the buffer is empty after the two lengths
                // unless the refill ran ahead, which `consumed` accounts for.
                let start = bits.consumed();
                let block = data
                    .get(start..start + length as usize)
                    .ok_or("truncated stored block")?;
                out.extend_from_slice(block);
                bits = Bits {
                    data,
                    at: start + length as usize,
                    buffer: 0,
                    count: 0,
                };
            }
            BLOCK_FIXED => {
                let (literals, distances) = fixed_codes()?;
                inflate_block(&mut bits, &mut out, limit, &literals, &distances)?;
            }
            BLOCK_DYNAMIC => {
                let (literals, distances) = dynamic_codes(&mut bits)?;
                inflate_block(&mut bits, &mut out, limit, &literals, &distances)?;
            }
            _ => return Err("invalid deflate block type"),
        }
        if out.len() > limit {
            return Err("decompressed data is too large");
        }
        if last {
            return Ok((out, bits.consumed()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Made with Python's zlib (wbits 31 for the gzip wrapper): level 0, level 9 with Z_FIXED,
    // and level 9 on repetitive text, whose first blocks are stored, fixed and dynamic.
    const STORED: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x13, 0x01, 0x16, 0x00, 0xe9, 0xff,
        0x75, 0x65, 0x6d, 0x75, 0x6c, 0x61, 0x74, 0x6f, 0x72, 0x20, 0x73, 0x74, 0x6f, 0x72, 0x65,
        0x64, 0x20, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x44, 0x85, 0x83, 0xd5, 0x16, 0x00, 0x00, 0x00,
    ];
    const FIXED: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x13, 0x4b, 0x4c, 0x4a, 0x4e, 0x44,
        0x45, 0x0a, 0x19, 0xa9, 0x39, 0x39, 0xf9, 0xc8, 0x24, 0x00, 0xae, 0xd8, 0x6c, 0x3b, 0x24,
        0x00, 0x00, 0x00,
    ];
    const DYNAMIC: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x13, 0xed, 0xd1, 0xbb, 0x0d, 0x80,
        0x30, 0x0c, 0x45, 0xd1, 0x9e, 0x29, 0x3c, 0x00, 0x05, 0xff, 0x7d, 0x10, 0x31, 0x22, 0x52,
        0x84, 0x23, 0xc7, 0x0a, 0x62, 0x7b, 0x44, 0x47, 0x6f, 0xca, 0xd7, 0xdc, 0xf2, 0x36, 0x27,
        0xc5, 0x93, 0xa9, 0x23, 0xd9, 0xc9, 0x0e, 0xa6, 0xe3, 0xce, 0xac, 0x35, 0x16, 0x51, 0x32,
        0x2e, 0x46, 0x95, 0x37, 0x13, 0x6d, 0x49, 0x39, 0xf3, 0x6a, 0x1c, 0xe8, 0x12, 0x0d, 0xe5,
        0xdb, 0x26, 0xbd, 0x83, 0xde, 0x3b, 0x18, 0xbc, 0x83, 0xd1, 0x3b, 0x98, 0xbc, 0x83, 0xd9,
        0x3b, 0x58, 0xbc, 0x03, 0x30, 0x82, 0x11, 0x8c, 0x60, 0x04, 0x23, 0x18, 0xc1, 0x08, 0xc6,
        0x3f, 0x19, 0x1f, 0x06, 0x8b, 0x68, 0x6c, 0x28, 0x0a, 0x00, 0x00,
    ];
    const LIMIT: usize = 1 << 20;

    fn dynamic_text() -> Vec<u8> {
        (0..40)
            .flat_map(|line| {
                format!(
                    "line {} of the hypervisor test vector, repeated words words words\n",
                    line % 7
                )
                .into_bytes()
            })
            .collect()
    }

    #[test]
    fn decodes_stored_fixed_and_dynamic_blocks() {
        assert_eq!(gunzip(STORED, LIMIT).unwrap(), b"uemulator stored block");
        assert_eq!(
            gunzip(FIXED, LIMIT).unwrap(),
            b"abcabcabcabcabcabc hello hello hello"
        );
        assert_eq!(gunzip(DYNAMIC, LIMIT).unwrap(), dynamic_text());
    }

    #[test]
    fn skips_the_optional_header_fields() {
        // The stored vector with a file name ("x"), as the gzip tool writes one.
        let mut named = STORED[..GZIP_HEADER_BYTES].to_vec();
        named[3] |= GZIP_FLAG_NAME;
        named.extend_from_slice(b"x\0");
        named.extend_from_slice(&STORED[GZIP_HEADER_BYTES..]);
        assert_eq!(gunzip(&named, LIMIT).unwrap(), b"uemulator stored block");
    }

    #[test]
    fn every_truncation_and_a_flipped_byte_are_errors() {
        for data in [STORED, FIXED, DYNAMIC] {
            for length in 0..data.len() {
                assert!(gunzip(&data[..length], LIMIT).is_err(), "length {length}");
            }
            for at in GZIP_HEADER_BYTES..data.len() {
                let mut corrupt = data.to_vec();
                corrupt[at] ^= 0x55;
                // Either refused by the decoder or by the checksum, and never a panic.
                assert!(gunzip(&corrupt, LIMIT).is_err(), "byte {at}");
            }
        }
    }

    #[test]
    fn output_past_the_limit_is_refused() {
        assert!(gunzip(DYNAMIC, 100).is_err());
        assert!(!is_gzip(b"\x7fELF"));
    }
}
