//! The framing of the GDB remote serial protocol: `$data#cc`, where `cc` is the sum of the data
//! bytes modulo 256 in two hex digits, `}` escapes a byte as itself XOR 0x20, and the receiver
//! answers every packet with `+` (or `-` to ask again) until both sides agree to stop
//! (`QStartNoAckMode`). A raw 0x03 between packets is the client's interrupt (Ctrl-C).

#[cfg(freestanding)]
use crate::prelude::*;

const PACKET_START: u8 = b'$';
const CHECKSUM_START: u8 = b'#';
const ESCAPE: u8 = b'}';
/// Run-length marker in a reply; a literal `*` must be escaped or the client expands it.
const RUN_LENGTH: u8 = b'*';
const ESCAPE_XOR: u8 = 0x20;
pub(crate) const ACK: u8 = b'+';
pub(crate) const NACK: u8 = b'-';
pub(crate) const INTERRUPT: u8 = 0x03;
/// The longest packet the client may send (the `PacketSize` in the `qSupported` reply); a longer
/// one is dropped as garbage rather than grown without bound.
pub(crate) const MAX_PACKET_BYTES: usize = 0x4000;

/// What the byte stream from the client holds, in order.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Incoming {
    /// A packet whose checksum matched, unescaped.
    Packet(Vec<u8>),
    /// A packet whose checksum did not match: answer `-` so the client sends it again.
    Corrupt,
    Ack,
    Nack,
    Interrupt,
}

#[derive(Default)]
enum State {
    #[default]
    Idle,
    Data,
    /// The byte after `}`.
    Escaped,
    /// The first checksum digit.
    Checksum1,
    /// The second checksum digit; holds the first.
    Checksum2(u8),
}

/// Splits the client's byte stream into packets, across reads that end anywhere.
#[derive(Default)]
pub(crate) struct Decoder {
    state: State,
    data: Vec<u8>,
    /// The sum of the bytes as sent (escapes included), which is what the checksum covers.
    sum: u8,
    overflow: bool,
}

impl Decoder {
    pub fn feed(&mut self, bytes: &[u8], out: &mut Vec<Incoming>) {
        for &byte in bytes {
            match core::mem::take(&mut self.state) {
                State::Idle => match byte {
                    PACKET_START => {
                        self.data.clear();
                        self.sum = 0;
                        self.overflow = false;
                        self.state = State::Data;
                    }
                    ACK => out.push(Incoming::Ack),
                    NACK => out.push(Incoming::Nack),
                    INTERRUPT => out.push(Incoming::Interrupt),
                    // Line noise between packets.
                    _ => {}
                },
                State::Data => {
                    match byte {
                        CHECKSUM_START => {
                            self.state = State::Checksum1;
                            continue;
                        }
                        ESCAPE => self.state = State::Escaped,
                        _ => {
                            self.push(byte);
                            self.state = State::Data;
                        }
                    }
                    self.sum = self.sum.wrapping_add(byte);
                }
                State::Escaped => {
                    self.sum = self.sum.wrapping_add(byte);
                    self.push(byte ^ ESCAPE_XOR);
                    self.state = State::Data;
                }
                State::Checksum1 => self.state = State::Checksum2(byte),
                State::Checksum2(first) => {
                    let sent = hex_digit(first)
                        .zip(hex_digit(byte))
                        .map(|(high, low)| (high << 4) | low);
                    if sent == Some(self.sum) && !self.overflow {
                        out.push(Incoming::Packet(core::mem::take(&mut self.data)));
                    } else {
                        out.push(Incoming::Corrupt);
                    }
                }
            }
        }
    }

    fn push(&mut self, byte: u8) {
        if self.data.len() < MAX_PACKET_BYTES {
            self.data.push(byte);
        } else {
            self.overflow = true;
        }
    }
}

/// Appends `$data#cc` to `out`, escaping what the client would otherwise read as framing or as
/// run-length encoding.
pub(crate) fn frame(data: &[u8], out: &mut Vec<u8>) {
    out.push(PACKET_START);
    let mut sum = 0u8;
    for &byte in data {
        if matches!(byte, PACKET_START | CHECKSUM_START | ESCAPE | RUN_LENGTH) {
            out.push(ESCAPE);
            out.push(byte ^ ESCAPE_XOR);
            sum = sum.wrapping_add(ESCAPE).wrapping_add(byte ^ ESCAPE_XOR);
        } else {
            out.push(byte);
            sum = sum.wrapping_add(byte);
        }
    }
    out.push(CHECKSUM_START);
    out.extend_from_slice(&hex_byte(sum));
}

pub(crate) fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn hex_byte(byte: u8) -> [u8; 2] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    [DIGITS[(byte >> 4) as usize], DIGITS[(byte & 0xf) as usize]]
}

/// `bytes` as hex, two digits per byte, in order (memory and register contents).
pub(crate) fn to_hex(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&byte| hex_byte(byte)).collect()
}

/// The bytes a hex string spells; `None` for an odd length or a non-hex digit.
pub(crate) fn from_hex(text: &[u8]) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.chunks(2)
        .map(|pair| Some((hex_digit(pair[0])? << 4) | hex_digit(pair[1])?))
        .collect()
}

/// A hex number as the protocol writes addresses and lengths: most significant digit first, no
/// prefix, at most 64 bits.
pub(crate) fn parse_number(text: &[u8]) -> Option<u64> {
    if text.is_empty() || text.len() > 16 {
        return None;
    }
    text.iter().try_fold(0u64, |value, &digit| {
        Some((value << 4) | hex_digit(digit)? as u64)
    })
}

/// `value` in hex without leading zeros, for thread ids and addresses in replies.
pub(crate) fn number(value: u64) -> Vec<u8> {
    format!("{value:x}").into_bytes()
}
