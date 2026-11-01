//! Terminal keystrokes as `int 16h` key words. The BIOS keyboard is the 16550, so arrows and
//! function keys arrive as escape sequences (`ESC [ A`) that a boot loader reading `int 16h`
//! would see as Escape followed by letters. A real BIOS returns the scan code in AH and, for
//! keys without a character, zero in AL: GRUB's console maps AH only when AL is zero.

use alloc::vec::Vec;

/// IBM PC scan codes, the high byte of an `int 16h` key word.
const SCAN_ESCAPE: u16 = 0x01;
const SCAN_BACKSPACE: u16 = 0x0e;
const SCAN_TAB: u16 = 0x0f;
const SCAN_ENTER: u16 = 0x1c;
const SCAN_F1: u16 = 0x3b;
const SCAN_HOME: u16 = 0x47;
const SCAN_UP: u16 = 0x48;
const SCAN_PAGE_UP: u16 = 0x49;
const SCAN_LEFT: u16 = 0x4b;
const SCAN_RIGHT: u16 = 0x4d;
const SCAN_END: u16 = 0x4f;
const SCAN_DOWN: u16 = 0x50;
const SCAN_PAGE_DOWN: u16 = 0x51;
const SCAN_INSERT: u16 = 0x52;
const SCAN_DELETE: u16 = 0x53;
/// F11 and F12 came after the original keyboard's codes ran out.
const SCAN_F11: u16 = 0x85;
const SCAN_F12: u16 = 0x86;

const ESCAPE: u8 = 0x1b;
const BACKSPACE: u8 = 0x08;
const DELETE_BYTE: u8 = 0x7f;
const TAB: u8 = 0x09;
const CARRIAGE_RETURN: u8 = 0x0d;
const LINE_FEED: u8 = 0x0a;
/// No terminal key sends more; anything longer is dropped as garbage.
const LONGEST_SEQUENCE: usize = 16;

/// How long an unfinished sequence may sit with no more input on the line before it counts as
/// the Escape key alone. `bios_collect_keys` only asks once nothing else is queued, so this
/// does not have to cover the 87 us the UART paces each byte at.
pub(crate) const LONE_ESCAPE_TIMEOUT_NS: u64 = 50_000_000;

fn key(scan: u16, character: u8) -> u16 {
    scan << 8 | character as u16
}

fn special(scan: u16) -> u16 {
    key(scan, 0)
}

/// A key that is one byte: the control keys get the scan code a keyboard would send, the rest
/// only their character.
fn single(byte: u8) -> u16 {
    match byte {
        ESCAPE => key(SCAN_ESCAPE, ESCAPE),
        BACKSPACE | DELETE_BYTE => key(SCAN_BACKSPACE, BACKSPACE),
        TAB => key(SCAN_TAB, TAB),
        CARRIAGE_RETURN | LINE_FEED => key(SCAN_ENTER, byte),
        other => other as u16,
    }
}

/// `ESC [ n ~` in xterm's numbering.
fn tilde_key(number: u32) -> Option<u16> {
    Some(special(match number {
        1 | 7 => SCAN_HOME,
        2 => SCAN_INSERT,
        3 => SCAN_DELETE,
        4 | 8 => SCAN_END,
        5 => SCAN_PAGE_UP,
        6 => SCAN_PAGE_DOWN,
        11..=15 => SCAN_F1 + (number - 11) as u16,
        17..=21 => SCAN_F1 + (number - 12) as u16,
        23 => SCAN_F11,
        24 => SCAN_F12,
        _ => return None,
    }))
}

enum Sequence {
    /// Needs more bytes.
    Incomplete,
    /// Ended with this key, or with none for a sequence we do not know.
    Done(Option<u16>),
    /// `ESC` then something that does not start a sequence: Escape, then that byte afresh.
    EscapeAlone,
}

/// Decodes `ESC`, an introducer (`[` or `O`), parameters and a final byte.
fn parse(sequence: &[u8]) -> Sequence {
    let Some(&introducer) = sequence.get(1) else {
        return Sequence::Incomplete;
    };
    if introducer != b'[' && introducer != b'O' {
        return Sequence::EscapeAlone;
    }
    let mut number = 0u32;
    for &byte in &sequence[2..] {
        let found = match byte {
            b'0'..=b'9' => {
                number = number
                    .saturating_mul(10)
                    .saturating_add((byte - b'0') as u32);
                continue;
            }
            b';' => {
                number = 0;
                continue;
            }
            b'A' => Some(special(SCAN_UP)),
            b'B' => Some(special(SCAN_DOWN)),
            b'C' => Some(special(SCAN_RIGHT)),
            b'D' => Some(special(SCAN_LEFT)),
            b'H' => Some(special(SCAN_HOME)),
            b'F' => Some(special(SCAN_END)),
            b'P'..=b'S' => Some(special(SCAN_F1 + (byte - b'P') as u16)),
            b'~' => tilde_key(number),
            _ => None,
        };
        return Sequence::Done(found);
    }
    if sequence.len() >= LONGEST_SEQUENCE {
        Sequence::Done(None)
    } else {
        Sequence::Incomplete
    }
}

#[derive(Default)]
pub(crate) struct KeyDecoder {
    /// An escape sequence that has started.
    pending: Vec<u8>,
    last_byte_ns: u64,
}

impl KeyDecoder {
    /// Takes one byte from the line; returns the keys it completes (two when an `ESC` turns
    /// out not to start a sequence).
    pub(crate) fn push(&mut self, byte: u8, now_ns: u64) -> Vec<u16> {
        self.last_byte_ns = now_ns;
        if self.pending.is_empty() {
            if byte == ESCAPE {
                self.pending.push(byte);
                return Vec::new();
            }
            return alloc::vec![single(byte)];
        }
        self.pending.push(byte);
        match parse(&self.pending) {
            Sequence::Incomplete => Vec::new(),
            Sequence::Done(found) => {
                self.pending.clear();
                found.into_iter().collect()
            }
            Sequence::EscapeAlone => {
                self.pending.clear();
                let mut keys = alloc::vec![single(ESCAPE)];
                keys.extend(self.push(byte, now_ns));
                keys
            }
        }
    }

    /// With nothing more on the line, a sequence that stopped short was the Escape key.
    pub(crate) fn expire(&mut self, now_ns: u64) -> Option<u16> {
        let waited = now_ns.saturating_sub(self.last_byte_ns);
        if self.pending.is_empty() || waited < LONE_ESCAPE_TIMEOUT_NS {
            return None;
        }
        self.pending.clear();
        Some(single(ESCAPE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(bytes: &[u8]) -> Vec<u16> {
        let mut decoder = KeyDecoder::default();
        bytes
            .iter()
            .enumerate()
            .flat_map(|(index, &byte)| decoder.push(byte, index as u64))
            .collect()
    }

    #[test]
    fn arrows_are_a_scan_code_with_no_character() {
        assert_eq!(
            decode(b"\x1b[A\x1b[B\x1b[D\x1b[C\x1bOA"),
            [0x4800, 0x5000, 0x4b00, 0x4d00, 0x4800]
        );
    }

    #[test]
    fn navigation_and_function_keys() {
        assert_eq!(
            decode(b"\x1b[H\x1b[F\x1b[3~\x1b[5~\x1b[6~\x1bOP\x1b[15~\x1b[21~\x1b[24~"),
            [
                0x4700, 0x4f00, 0x5300, 0x4900, 0x5100, 0x3b00, 0x3f00, 0x4400, 0x8600
            ]
        );
    }

    #[test]
    fn modified_arrows_still_move() {
        assert_eq!(decode(b"\x1b[1;5A"), [0x4800]);
    }

    #[test]
    fn control_keys_carry_their_scan_codes() {
        assert_eq!(decode(b"a\r\x7f\t"), [b'a' as u16, 0x1c0d, 0x0e08, 0x0f09]);
    }

    #[test]
    fn an_escape_followed_by_a_letter_is_escape_then_the_letter() {
        assert_eq!(decode(b"\x1bx"), [0x011b, b'x' as u16]);
        assert_eq!(decode(b"\x1b\x1b[A"), [0x011b, 0x4800]);
    }

    #[test]
    fn a_lone_escape_registers_once_the_line_has_been_quiet() {
        let mut decoder = KeyDecoder::default();
        assert_eq!(decoder.push(ESCAPE, 1_000), []);
        assert_eq!(decoder.expire(1_000 + LONE_ESCAPE_TIMEOUT_NS - 1), None);
        assert_eq!(decoder.expire(1_000 + LONE_ESCAPE_TIMEOUT_NS), Some(0x011b));
        assert_eq!(decoder.expire(u64::MAX), None, "only once");
    }

    #[test]
    fn a_sequence_that_waits_between_bytes_is_still_one_key() {
        // Bytes a timer tick apart, as `int 16h` function 0 picks them up between halts.
        let mut decoder = KeyDecoder::default();
        let tick = 55_000_000;
        assert_eq!(decoder.push(ESCAPE, 0), []);
        assert_eq!(decoder.push(b'[', tick), []);
        assert_eq!(decoder.push(b'B', 2 * tick), [0x5000]);
    }

    #[test]
    fn unknown_or_endless_sequences_are_dropped() {
        assert_eq!(decode(b"\x1b[99~x"), [b'x' as u16]);
        let mut endless = alloc::vec![ESCAPE, b'['];
        endless.extend([b'1'; LONGEST_SEQUENCE - 2]);
        endless.push(b'y');
        assert_eq!(decode(&endless), [b'y' as u16]);
    }
}
