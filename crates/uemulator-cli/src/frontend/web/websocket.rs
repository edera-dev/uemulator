//! The parts of RFC 6455 the viewer needs: the handshake's accept key, and reading and writing
//! frames on a stream. Browsers send small text messages (input) and the server sends binary
//! ones (screen updates), so there is no extension, no compression and no subprotocol.

use std::io::{self, Read, Write};

/// Appended to the client's key before hashing (RFC 6455 section 1.3).
const HANDSHAKE_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Frame opcodes (section 5.2).
pub const OPCODE_CONTINUATION: u8 = 0x0;
pub const OPCODE_TEXT: u8 = 0x1;
pub const OPCODE_BINARY: u8 = 0x2;
pub const OPCODE_CLOSE: u8 = 0x8;
pub const OPCODE_PING: u8 = 0x9;
pub const OPCODE_PONG: u8 = 0xa;

const FIN: u8 = 0x80;
const OPCODE_MASK: u8 = 0x0f;
const MASKED: u8 = 0x80;
const LENGTH_MASK: u8 = 0x7f;
/// Length codes: up to 125 is the length itself, 126 means a u16 follows, 127 a u64.
const LENGTH_U16: u8 = 126;
const LENGTH_U64: u8 = 127;
const MAX_INLINE_LENGTH: usize = 125;
const MASK_KEY_BYTES: usize = 4;

/// The longest message accepted from a browser. Input messages are a few bytes; anything near
/// this is not the viewer page talking.
pub const MAX_INCOMING_MESSAGE: usize = 64 * 1024;

/// One complete message (fragments joined).
#[derive(Debug, PartialEq, Eq)]
pub struct Message {
    pub opcode: u8,
    pub payload: Vec<u8>,
}

/// `Sec-WebSocket-Accept` for a client's `Sec-WebSocket-Key`.
pub fn accept_key(client_key: &str) -> String {
    let mut input = client_key.trim().as_bytes().to_vec();
    input.extend_from_slice(HANDSHAKE_GUID.as_bytes());
    base64(&sha1(&input))
}

/// Writes one unfragmented, unmasked frame (servers never mask, section 5.1).
pub fn write_frame(stream: &mut impl Write, opcode: u8, payload: &[u8]) -> io::Result<()> {
    let mut header = Vec::with_capacity(10);
    header.push(FIN | opcode);
    match payload.len() {
        length if length <= MAX_INLINE_LENGTH => header.push(length as u8),
        length if length <= u16::MAX as usize => {
            header.push(LENGTH_U16);
            header.extend_from_slice(&(length as u16).to_be_bytes());
        }
        length => {
            header.push(LENGTH_U64);
            header.extend_from_slice(&(length as u64).to_be_bytes());
        }
    }
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    stream.flush()
}

/// Reads messages off one connection. Keeps a fragmented data message across calls, because a
/// control frame may arrive between its fragments (section 5.4) and is returned at once.
#[derive(Default)]
pub struct MessageReader {
    /// Opcode and payload so far of a data message whose last fragment has not come.
    partial: Option<(u8, Vec<u8>)>,
}

impl MessageReader {
    /// The next complete message. Messages over `MAX_INCOMING_MESSAGE` are an error: the
    /// connection is not worth keeping.
    pub fn read(&mut self, stream: &mut impl Read) -> io::Result<Message> {
        read_message(stream, &mut self.partial)
    }
}

fn read_message(stream: &mut impl Read, data: &mut Option<(u8, Vec<u8>)>) -> io::Result<Message> {
    loop {
        let (fin, opcode, payload) = read_frame(stream)?;
        if opcode >= OPCODE_CLOSE {
            return Ok(Message { opcode, payload });
        }
        let (message_opcode, mut joined) = match (data.take(), opcode) {
            (None, OPCODE_CONTINUATION) => return Err(invalid("continuation without a start")),
            (None, opcode) => (opcode, Vec::new()),
            (Some(started), OPCODE_CONTINUATION) => started,
            (Some(_), _) => return Err(invalid("new message inside a fragmented one")),
        };
        if joined.len() + payload.len() > MAX_INCOMING_MESSAGE {
            return Err(invalid("message too long"));
        }
        joined.extend_from_slice(&payload);
        if fin {
            return Ok(Message {
                opcode: message_opcode,
                payload: joined,
            });
        }
        *data = Some((message_opcode, joined));
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// (FIN, opcode, unmasked payload) of one frame.
fn read_frame(stream: &mut impl Read) -> io::Result<(bool, u8, Vec<u8>)> {
    let mut head = [0u8; 2];
    stream.read_exact(&mut head)?;
    let fin = head[0] & FIN != 0;
    let opcode = head[0] & OPCODE_MASK;
    let masked = head[1] & MASKED != 0;
    let length = match head[1] & LENGTH_MASK {
        LENGTH_U16 => {
            let mut bytes = [0u8; 2];
            stream.read_exact(&mut bytes)?;
            u16::from_be_bytes(bytes) as u64
        }
        LENGTH_U64 => {
            let mut bytes = [0u8; 8];
            stream.read_exact(&mut bytes)?;
            u64::from_be_bytes(bytes)
        }
        length => length as u64,
    };
    if length > MAX_INCOMING_MESSAGE as u64 {
        return Err(invalid("frame too long"));
    }
    let mut mask = [0u8; MASK_KEY_BYTES];
    if masked {
        stream.read_exact(&mut mask)?;
    }
    let mut payload = vec![0u8; length as usize];
    stream.read_exact(&mut payload)?;
    if masked {
        for (index, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[index % MASK_KEY_BYTES];
        }
    }
    Ok((fin, opcode, payload))
}

/// SHA-1 (FIPS 180-4), for the handshake only: the accept key is defined with it. Not used for
/// anything that needs a secure hash.
fn sha1(message: &[u8]) -> [u8; 20] {
    const BLOCK_BYTES: usize = 64;
    let mut state: [u32; 5] = [
        0x6745_2301,
        0xefcd_ab89,
        0x98ba_dcfe,
        0x1032_5476,
        0xc3d2_e1f0,
    ];
    let mut padded = message.to_vec();
    padded.push(0x80);
    while padded.len() % BLOCK_BYTES != BLOCK_BYTES - 8 {
        padded.push(0);
    }
    padded.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());
    for block in padded.chunks(BLOCK_BYTES) {
        let mut words = [0u32; 80];
        for (index, word) in block.chunks(4).enumerate() {
            words[index] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (index, word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | (!b & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let temporary = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temporary;
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0u8; 20];
    for (chunk, word) in digest.chunks_mut(4).zip(state) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Standard base64 with padding (RFC 4648 section 4).
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let triple = (chunk[0] as u32) << 16
            | (chunk.get(1).copied().unwrap_or(0) as u32) << 8
            | chunk.get(2).copied().unwrap_or(0) as u32;
        for position in 0..4 {
            if position <= chunk.len() {
                out.push(ALPHABET[(triple >> (18 - 6 * position) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// A frame as a browser sends it: masked with `mask`.
    fn client_frame(fin: bool, opcode: u8, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
        let mut frame = vec![if fin { FIN } else { 0 } | opcode];
        match payload.len() {
            length if length <= MAX_INLINE_LENGTH => frame.push(MASKED | length as u8),
            length => {
                frame.push(MASKED | LENGTH_U16);
                frame.extend_from_slice(&(length as u16).to_be_bytes());
            }
        }
        frame.extend_from_slice(&mask);
        frame.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        frame
    }

    #[test]
    fn sha1_matches_the_fips_test_vectors() {
        assert_eq!(
            hex(&sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn base64_pads_to_whole_quads() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn the_accept_key_is_the_one_in_rfc_6455() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    fn read_one(bytes: Vec<u8>) -> io::Result<Message> {
        MessageReader::default().read(&mut Cursor::new(bytes))
    }

    #[test]
    fn a_masked_text_frame_is_unmasked() {
        let frame = client_frame(true, OPCODE_TEXT, b"k 4 1", [1, 2, 3, 4]);
        let message = read_one(frame).unwrap();
        assert_eq!(
            message,
            Message {
                opcode: OPCODE_TEXT,
                payload: b"k 4 1".to_vec()
            }
        );
    }

    #[test]
    fn fragments_are_joined_and_a_ping_between_them_comes_first() {
        let mut stream = client_frame(false, OPCODE_TEXT, b"hel", [9, 8, 7, 6]);
        stream.extend(client_frame(true, OPCODE_PING, b"p", [1, 1, 1, 1]));
        stream.extend(client_frame(true, OPCODE_CONTINUATION, b"lo", [5, 5, 5, 5]));
        let mut cursor = Cursor::new(stream);
        let mut reader = MessageReader::default();
        assert_eq!(reader.read(&mut cursor).unwrap().opcode, OPCODE_PING);
        let message = reader.read(&mut cursor).unwrap();
        assert_eq!(
            (message.opcode, message.payload),
            (OPCODE_TEXT, b"hello".to_vec())
        );
    }

    #[test]
    fn a_frame_with_a_two_byte_length_is_read_whole() {
        let payload = vec![0x5a; 300];
        let frame = client_frame(true, OPCODE_BINARY, &payload, [0, 1, 2, 3]);
        assert_eq!(read_one(frame).unwrap().payload, payload);
    }

    #[test]
    fn oversized_and_malformed_messages_are_errors() {
        let mut huge = vec![FIN | OPCODE_BINARY, LENGTH_U64];
        huge.extend_from_slice(&(1u64 << 40).to_be_bytes());
        assert!(read_one(huge).is_err());
        let orphan = client_frame(true, OPCODE_CONTINUATION, b"x", [0; 4]);
        assert!(read_one(orphan).is_err());
        let mut interleaved = client_frame(false, OPCODE_TEXT, b"a", [0; 4]);
        interleaved.extend(client_frame(true, OPCODE_TEXT, b"b", [0; 4]));
        assert!(read_one(interleaved).is_err());
        assert!(read_one(vec![FIN | OPCODE_TEXT]).is_err(), "truncated");
    }

    #[test]
    fn server_frames_are_unmasked_with_the_shortest_length_code() {
        for (length, header_bytes) in [(5usize, 2usize), (200, 4), (70_000, 10)] {
            let mut out = Vec::new();
            write_frame(&mut out, OPCODE_BINARY, &vec![7; length]).unwrap();
            assert_eq!(out.len(), header_bytes + length, "{length}");
            assert_eq!(out[0], FIN | OPCODE_BINARY);
            assert_eq!(out[1] & MASKED, 0);
        }
    }
}
