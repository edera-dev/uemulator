//! Turns the serial port's byte stream (a terminal's keystrokes) into UEFI key events.

use crate::uefi::InputKey;

pub const SCAN_NONE: u16 = 0;
pub const SCAN_UP: u16 = 0x01;
pub const SCAN_DOWN: u16 = 0x02;
pub const SCAN_RIGHT: u16 = 0x03;
pub const SCAN_LEFT: u16 = 0x04;
pub const SCAN_HOME: u16 = 0x05;
pub const SCAN_END: u16 = 0x06;
pub const SCAN_INSERT: u16 = 0x07;
pub const SCAN_DELETE: u16 = 0x08;
pub const SCAN_PAGE_UP: u16 = 0x09;
pub const SCAN_PAGE_DOWN: u16 = 0x0a;
pub const SCAN_F1: u16 = 0x0b;
pub const SCAN_F2: u16 = 0x0c;
pub const SCAN_F12: u16 = 0x16;
pub const SCAN_ESCAPE: u16 = 0x17;

/// How long each byte after the first of a key may take to arrive. The emulated 16550 delivers
/// host input at one byte per character time (87 us at 115200 baud) like a real line, so an
/// arrow key's `ESC [ A` is never all waiting at once; without the wait it decodes as Escape,
/// `[` and `A`. The UART sees guest time only at the machine's device tick, so this is far
/// longer than a character time; it is also how long a lone Escape key takes to register.
pub const FOLLOWING_BYTE_TIMEOUT_NS: u64 = 50_000_000;

const ESCAPE: u8 = 0x1b;
const DELETE_BYTE: u8 = 0x7f;
const BACKSPACE: u16 = 8;

fn scan(code: u16) -> InputKey {
    InputKey {
        scan_code: code,
        unicode_char: 0,
    }
}

/// Maps the number in `ESC [ n ~` to a key.
fn tilde_key(number: u32) -> Option<InputKey> {
    Some(scan(match number {
        1 | 7 => SCAN_HOME,
        2 => SCAN_INSERT,
        3 => SCAN_DELETE,
        4 | 8 => SCAN_END,
        5 => SCAN_PAGE_UP,
        6 => SCAN_PAGE_DOWN,
        // The function keys skip numbers in xterm's encoding.
        11..=15 => SCAN_F1 + (number - 11) as u16,
        17..=21 => SCAN_F1 + (number - 12) as u16,
        23 | 24 => SCAN_F1 + (number - 13) as u16,
        _ => return None,
    }))
}

/// Waits for the next byte of a key that has started: `read` polls the line without blocking,
/// `now_ns` is the clock. `None` once `FOLLOWING_BYTE_TIMEOUT_NS` passes with nothing.
pub fn following_byte(
    mut read: impl FnMut() -> Option<u8>,
    mut now_ns: impl FnMut() -> u64,
) -> Option<u8> {
    let deadline = now_ns().saturating_add(FOLLOWING_BYTE_TIMEOUT_NS);
    loop {
        if let Some(byte) = read() {
            return Some(byte);
        }
        if now_ns() >= deadline {
            return None;
        }
        core::hint::spin_loop();
    }
}

/// Reads one key. `first` returns a byte if one is waiting, without blocking; `next` returns
/// the following bytes of the same key, waiting a little for each (`following_byte`). A lone
/// `ESC` (the Escape key) is told from the start of a sequence by nothing following it.
pub fn decode(
    mut first: impl FnMut() -> Option<u8>,
    mut next: impl FnMut() -> Option<u8>,
) -> Option<InputKey> {
    let first = first()?;
    let character = |unit: u16| {
        Some(InputKey {
            scan_code: SCAN_NONE,
            unicode_char: unit,
        })
    };
    match first {
        ESCAPE => {}
        DELETE_BYTE => return character(BACKSPACE),
        0x80..=0xff => return decode_utf8(first, &mut next),
        byte => return character(byte as u16),
    }
    let Some(introducer) = next() else {
        return Some(scan(SCAN_ESCAPE));
    };
    match introducer {
        b'[' | b'O' => {}
        _ => return Some(scan(SCAN_ESCAPE)),
    }
    let mut number = 0u32;
    loop {
        let Some(byte) = next() else {
            return Some(scan(SCAN_ESCAPE));
        };
        match byte {
            b'0'..=b'9' => number = number * 10 + (byte - b'0') as u32,
            b';' => number = 0,
            b'A' => return Some(scan(SCAN_UP)),
            b'B' => return Some(scan(SCAN_DOWN)),
            b'C' => return Some(scan(SCAN_RIGHT)),
            b'D' => return Some(scan(SCAN_LEFT)),
            b'H' => return Some(scan(SCAN_HOME)),
            b'F' => return Some(scan(SCAN_END)),
            b'P'..=b'S' => return Some(scan(SCAN_F1 + (byte - b'P') as u16)),
            b'~' => return tilde_key(number),
            _ => return None,
        }
    }
}

fn decode_utf8(first: u8, next: &mut impl FnMut() -> Option<u8>) -> Option<InputKey> {
    let (length, mut unit) = match first {
        0xc0..=0xdf => (1, (first & 0x1f) as u32),
        0xe0..=0xef => (2, (first & 0x0f) as u32),
        _ => return None,
    };
    for _ in 0..length {
        unit = unit << 6 | (next()? & 0x3f) as u32;
    }
    Some(InputKey {
        scan_code: SCAN_NONE,
        unicode_char: u16::try_from(unit).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(bytes: &[u8]) -> alloc::vec::Vec<(u16, u16)> {
        let position = core::cell::Cell::new(0);
        let mut result = alloc::vec::Vec::new();
        let read = || {
            let byte = bytes.get(position.get()).copied();
            position.set(position.get() + 1);
            byte
        };
        while position.get() < bytes.len() {
            let key = decode(read, read);
            if let Some(key) = key {
                result.push((key.scan_code, key.unicode_char));
            }
        }
        result
    }

    #[test]
    fn plain_characters_and_control_keys() {
        assert_eq!(keys(b"a\r\x7f"), [(0, 'a' as u16), (0, 13), (0, 8)]);
    }

    #[test]
    fn arrows_and_navigation() {
        assert_eq!(
            keys(b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~\x1b[5~\x1b[6~"),
            [
                (SCAN_UP, 0),
                (SCAN_DOWN, 0),
                (SCAN_RIGHT, 0),
                (SCAN_LEFT, 0),
                (SCAN_HOME, 0),
                (SCAN_END, 0),
                (SCAN_DELETE, 0),
                (SCAN_PAGE_UP, 0),
                (SCAN_PAGE_DOWN, 0)
            ]
        );
    }

    #[test]
    fn function_keys() {
        assert_eq!(
            keys(b"\x1bOP\x1bOQ\x1b[15~\x1b[17~\x1b[21~\x1b[23~\x1b[24~"),
            [
                (SCAN_F1, 0),
                (SCAN_F2, 0),
                (SCAN_F1 + 4, 0),
                (SCAN_F1 + 5, 0),
                (SCAN_F1 + 9, 0),
                (SCAN_F1 + 10, 0),
                (SCAN_F12, 0)
            ]
        );
    }

    #[test]
    fn a_lone_escape_is_the_escape_key() {
        assert_eq!(keys(b"\x1b"), [(SCAN_ESCAPE, 0)]);
        assert_eq!(keys(b"\x1bx"), [(SCAN_ESCAPE, 0)]);
    }

    #[test]
    fn utf8_becomes_one_character() {
        assert_eq!(keys("é".as_bytes()), [(0, 0xe9)]);
    }

    /// A line that delivers one byte per character time: each byte shows up after `polls_per_byte`
    /// empty polls, and every poll moves the clock on by a microsecond.
    struct PacedLine<'a> {
        bytes: &'a [u8],
        position: core::cell::Cell<usize>,
        empty_polls: core::cell::Cell<u32>,
        polls_per_byte: u32,
        clock_ns: core::cell::Cell<u64>,
    }

    impl<'a> PacedLine<'a> {
        const POLL_NS: u64 = 1_000;

        fn new(bytes: &'a [u8], polls_per_byte: u32) -> Self {
            PacedLine {
                bytes,
                position: core::cell::Cell::new(0),
                empty_polls: core::cell::Cell::new(0),
                polls_per_byte,
                clock_ns: core::cell::Cell::new(0),
            }
        }

        fn read(&self) -> Option<u8> {
            self.clock_ns.set(self.clock_ns.get() + Self::POLL_NS);
            if self.empty_polls.get() < self.polls_per_byte {
                self.empty_polls.set(self.empty_polls.get() + 1);
                return None;
            }
            let byte = self.bytes.get(self.position.get()).copied()?;
            self.position.set(self.position.get() + 1);
            self.empty_polls.set(0);
            Some(byte)
        }

        fn key(&self) -> Option<(u16, u16)> {
            let first = self.bytes.get(self.position.get()).copied()?;
            self.position.set(self.position.get() + 1);
            self.empty_polls.set(0);
            let mut first = Some(first);
            decode(
                || first.take(),
                || following_byte(|| self.read(), || self.clock_ns.get()),
            )
            .map(|key| (key.scan_code, key.unicode_char))
        }
    }

    #[test]
    fn a_sequence_paced_like_a_serial_line_is_still_one_key() {
        // 87 polls of a microsecond each: one byte per character time at 115200 baud.
        let line = PacedLine::new(b"\x1b[B\x1b[1;5A", 87);
        assert_eq!(line.key(), Some((SCAN_DOWN, 0)));
        assert_eq!(line.key(), Some((SCAN_UP, 0)));
        assert_eq!(line.key(), None);
    }

    #[test]
    fn paced_utf8_is_one_character() {
        let line = PacedLine::new("é".as_bytes(), 87);
        assert_eq!(line.key(), Some((0, 0xe9)));
    }

    #[test]
    fn an_escape_with_nothing_after_it_registers_after_the_timeout() {
        let line = PacedLine::new(b"\x1b", 0);
        assert_eq!(line.key(), Some((SCAN_ESCAPE, 0)));
        let waited = line.clock_ns.get();
        assert!(
            waited >= FOLLOWING_BYTE_TIMEOUT_NS,
            "gave up after {waited} ns"
        );
        assert!(waited < FOLLOWING_BYTE_TIMEOUT_NS + 2 * PacedLine::POLL_NS);
    }
}
