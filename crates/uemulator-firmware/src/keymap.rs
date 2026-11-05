//! Linux key codes (what the virtio keyboard reports, `include/uapi/linux/input-event-codes.h`)
//! to UEFI keystrokes, with the shift, control and caps lock state that turns a key into a
//! character. A US layout: the firmware has no way to know another, and boot menus and GRUB's
//! editor need little more than letters, digits and the navigation keys.

use crate::keys::{
    SCAN_DELETE, SCAN_DOWN, SCAN_END, SCAN_ESCAPE, SCAN_F1, SCAN_HOME, SCAN_INSERT, SCAN_LEFT,
    SCAN_NONE, SCAN_PAGE_DOWN, SCAN_PAGE_UP, SCAN_RIGHT, SCAN_UP,
};
use crate::uefi::InputKey;

const KEY_ESC: u16 = 1;
const KEY_1: u16 = 2;
const KEY_0: u16 = 11;
const KEY_MINUS: u16 = 12;
const KEY_EQUAL: u16 = 13;
const KEY_BACKSPACE: u16 = 14;
const KEY_TAB: u16 = 15;
const KEY_LEFTBRACE: u16 = 26;
const KEY_RIGHTBRACE: u16 = 27;
const KEY_ENTER: u16 = 28;
const KEY_LEFTCTRL: u16 = 29;
const KEY_SEMICOLON: u16 = 39;
const KEY_APOSTROPHE: u16 = 40;
const KEY_GRAVE: u16 = 41;
const KEY_LEFTSHIFT: u16 = 42;
const KEY_BACKSLASH: u16 = 43;
const KEY_COMMA: u16 = 51;
const KEY_DOT: u16 = 52;
const KEY_SLASH: u16 = 53;
const KEY_RIGHTSHIFT: u16 = 54;
const KEY_KPASTERISK: u16 = 55;
const KEY_SPACE: u16 = 57;
const KEY_CAPSLOCK: u16 = 58;
const KEY_F1: u16 = 59;
const KEY_F10: u16 = 68;
const KEY_KP7: u16 = 71;
const KEY_KPDOT: u16 = 83;
const KEY_F11: u16 = 87;
const KEY_F12: u16 = 88;
const KEY_KPENTER: u16 = 96;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_KPSLASH: u16 = 98;
const KEY_HOME: u16 = 102;
const KEY_UP: u16 = 103;
const KEY_PAGEUP: u16 = 104;
const KEY_LEFT: u16 = 105;
const KEY_RIGHT: u16 = 106;
const KEY_END: u16 = 107;
const KEY_DOWN: u16 = 108;
const KEY_PAGEDOWN: u16 = 109;
const KEY_INSERT: u16 = 110;
const KEY_DELETE: u16 = 111;

/// UEFI's F11 and F12 scan codes do not follow F10 (`SCAN_F10` is 0x14).
const SCAN_F11: u16 = 0x15;
const SCAN_F12: u16 = 0x16;

const CARRIAGE_RETURN: u16 = 0x0d;
const BACKSPACE: u16 = 0x08;
const TAB: u16 = 0x09;
/// Ctrl with a letter gives its control character (Ctrl-A is 1), as a terminal sends it; GRUB's
/// editor reads Ctrl-X and Ctrl-C that way.
const CONTROL_MASK: u16 = 0x1f;

/// The top row and the punctuation keys: (key, unshifted, shifted).
const SYMBOLS: [(u16, char, char); 11] = [
    (KEY_MINUS, '-', '_'),
    (KEY_EQUAL, '=', '+'),
    (KEY_LEFTBRACE, '[', '{'),
    (KEY_RIGHTBRACE, ']', '}'),
    (KEY_SEMICOLON, ';', ':'),
    (KEY_APOSTROPHE, '\'', '"'),
    (KEY_GRAVE, '`', '~'),
    (KEY_BACKSLASH, '\\', '|'),
    (KEY_COMMA, ',', '<'),
    (KEY_DOT, '.', '>'),
    (KEY_SLASH, '/', '?'),
];
/// Shifted digits 1 to 0.
const SHIFTED_DIGITS: [char; 10] = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'];
/// Letters by key code: the three letter rows of the keyboard.
const LETTER_ROWS: [(u16, &str); 3] = [(16, "qwertyuiop"), (30, "asdfghjkl"), (44, "zxcvbnm")];
/// The keypad keys from 7 (`KEY_KP7`) to `.` in key-code order, as the characters they type
/// with num lock on, which is how the firmware takes it.
const KEYPAD: &[u8; 13] = b"789-456+1230.";

#[derive(Default)]
pub struct Keyboard {
    left_shift: bool,
    right_shift: bool,
    left_control: bool,
    right_control: bool,
    caps_lock: bool,
}

fn character(unicode: char) -> InputKey {
    InputKey {
        scan_code: SCAN_NONE,
        unicode_char: unicode as u16,
    }
}

fn scan(code: u16) -> InputKey {
    InputKey {
        scan_code: code,
        unicode_char: 0,
    }
}

fn letter(code: u16) -> Option<char> {
    LETTER_ROWS.iter().find_map(|(first, letters)| {
        let index = code.checked_sub(*first)? as usize;
        letters.as_bytes().get(index).map(|byte| *byte as char)
    })
}

impl Keyboard {
    fn shift(&self) -> bool {
        self.left_shift || self.right_shift
    }

    fn control(&self) -> bool {
        self.left_control || self.right_control
    }

    /// Takes one key event (`pressed` false for a release) and returns the keystroke it types,
    /// if any. Releases and modifiers type nothing; repeats are presses.
    pub fn event(&mut self, code: u16, pressed: bool) -> Option<InputKey> {
        match code {
            KEY_LEFTSHIFT => self.left_shift = pressed,
            KEY_RIGHTSHIFT => self.right_shift = pressed,
            KEY_LEFTCTRL => self.left_control = pressed,
            KEY_RIGHTCTRL => self.right_control = pressed,
            KEY_CAPSLOCK if pressed => self.caps_lock = !self.caps_lock,
            _ if pressed => return self.translate(code),
            _ => {}
        }
        None
    }

    fn translate(&self, code: u16) -> Option<InputKey> {
        if let Some(lower) = letter(code) {
            if self.control() {
                return Some(InputKey {
                    scan_code: SCAN_NONE,
                    unicode_char: lower as u16 & CONTROL_MASK,
                });
            }
            let upper = self.shift() != self.caps_lock;
            return Some(character(if upper {
                lower.to_ascii_uppercase()
            } else {
                lower
            }));
        }
        if (KEY_1..=KEY_0).contains(&code) {
            let index = (code - KEY_1) as usize;
            return Some(character(if self.shift() {
                SHIFTED_DIGITS[index]
            } else {
                // KEY_1..KEY_9 then KEY_0.
                char::from_digit(((index + 1) % 10) as u32, 10).unwrap()
            }));
        }
        if let Some((_, plain, shifted)) = SYMBOLS.iter().find(|(key, _, _)| *key == code) {
            return Some(character(if self.shift() { *shifted } else { *plain }));
        }
        if (KEY_KP7..=KEY_KPDOT).contains(&code) {
            return Some(character(KEYPAD[(code - KEY_KP7) as usize] as char));
        }
        Some(match code {
            KEY_ESC => scan(SCAN_ESCAPE),
            KEY_ENTER | KEY_KPENTER => InputKey {
                scan_code: SCAN_NONE,
                unicode_char: CARRIAGE_RETURN,
            },
            KEY_BACKSPACE => InputKey {
                scan_code: SCAN_NONE,
                unicode_char: BACKSPACE,
            },
            KEY_TAB => InputKey {
                scan_code: SCAN_NONE,
                unicode_char: TAB,
            },
            KEY_SPACE => character(' '),
            KEY_KPASTERISK => character('*'),
            KEY_KPSLASH => character('/'),
            KEY_F1..=KEY_F10 => scan(SCAN_F1 + (code - KEY_F1)),
            KEY_F11 => scan(SCAN_F11),
            KEY_F12 => scan(SCAN_F12),
            KEY_UP => scan(SCAN_UP),
            KEY_DOWN => scan(SCAN_DOWN),
            KEY_LEFT => scan(SCAN_LEFT),
            KEY_RIGHT => scan(SCAN_RIGHT),
            KEY_HOME => scan(SCAN_HOME),
            KEY_END => scan(SCAN_END),
            KEY_PAGEUP => scan(SCAN_PAGE_UP),
            KEY_PAGEDOWN => scan(SCAN_PAGE_DOWN),
            KEY_INSERT => scan(SCAN_INSERT),
            KEY_DELETE => scan(SCAN_DELETE),
            _ => return None,
        })
    }
}

/// How long a key is held before it repeats, and the time between repeats: EDK2's USB keyboard
/// driver's `USBKBD_REPEAT_DELAY` and `USBKBD_REPEAT_RATE` (`MdeModulePkg/Bus/Usb/UsbKbDxe`).
const REPEAT_DELAY_NS: u64 = 500_000_000;
const REPEAT_PERIOD_NS: u64 = 32_000_000;

struct HeldKey {
    code: u16,
    /// As typed when it went down: shift pressed or let go meanwhile does not change it.
    key: InputKey,
    due_ns: u64,
}

/// Repeats the last key pressed for as long as it is held. The virtio keyboard sends one press
/// and one release and leaves repeating to whoever reads it (it advertises `EV_REP`, so Linux's
/// input core repeats on its own), and the web pages drop the browser's repeats so Linux does
/// not see them twice; without this a held arrow key moves a boot menu or GRUB by one line.
#[derive(Default)]
pub struct Typematic {
    held: Option<HeldKey>,
}

impl Typematic {
    /// Takes a key event and the keystroke `Keyboard::event` made of it. Only a key that types
    /// something starts repeating; a modifier pressed meanwhile leaves the repeat going, as on a
    /// USB keyboard under EDK2.
    pub fn event(&mut self, code: u16, pressed: bool, typed: Option<InputKey>, now_ns: u64) {
        if pressed {
            if let Some(key) = typed {
                self.held = Some(HeldKey {
                    code,
                    key,
                    due_ns: now_ns.saturating_add(REPEAT_DELAY_NS),
                });
            }
        } else if self.held.as_ref().is_some_and(|held| held.code == code) {
            self.held = None;
        }
    }

    /// The repeat due by `now_ns`, if any. The next one is a period from now, not from when this
    /// one was due: a reader that comes back late gets one keystroke, not a burst for every
    /// period it missed.
    pub fn due(&mut self, now_ns: u64) -> Option<InputKey> {
        let held = self.held.as_mut()?;
        if now_ns < held.due_ns {
            return None;
        }
        held.due_ns = now_ns.saturating_add(REPEAT_PERIOD_NS);
        Some(held.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(keyboard: &mut Keyboard, code: u16) -> (u16, u16) {
        let key = keyboard.event(code, true).expect("a keystroke");
        keyboard.event(code, false);
        (key.scan_code, key.unicode_char)
    }

    fn text(keyboard: &mut Keyboard, codes: &[u16]) -> String {
        codes
            .iter()
            .map(|code| char::from_u32(typed(keyboard, *code).1 as u32).unwrap())
            .collect()
    }

    #[test]
    fn letters_digits_and_symbols_type_their_characters() {
        let mut keyboard = Keyboard::default();
        // r o o t, 1 0, -, /, space
        assert_eq!(text(&mut keyboard, &[19, 24, 24, 20]), "root");
        assert_eq!(text(&mut keyboard, &[KEY_1, KEY_0]), "10");
        assert_eq!(
            text(&mut keyboard, &[KEY_MINUS, KEY_SLASH, KEY_SPACE]),
            "-/ "
        );
        assert_eq!(text(&mut keyboard, &[16, 44, 50]), "qzm");
    }

    #[test]
    fn shift_and_caps_lock_choose_the_case_and_the_upper_symbols() {
        let mut keyboard = Keyboard::default();
        keyboard.event(KEY_LEFTSHIFT, true);
        assert_eq!(
            text(&mut keyboard, &[30, KEY_1, KEY_0, KEY_SEMICOLON]),
            "A!):"
        );
        keyboard.event(KEY_LEFTSHIFT, false);
        keyboard.event(KEY_CAPSLOCK, true);
        keyboard.event(KEY_CAPSLOCK, false);
        assert_eq!(
            text(&mut keyboard, &[30, KEY_1]),
            "A1",
            "caps lock is letters only"
        );
        keyboard.event(KEY_RIGHTSHIFT, true);
        assert_eq!(text(&mut keyboard, &[30]), "a", "shift undoes caps lock");
    }

    #[test]
    fn control_with_a_letter_is_its_control_character() {
        let mut keyboard = Keyboard::default();
        keyboard.event(KEY_RIGHTCTRL, true);
        assert_eq!(typed(&mut keyboard, 45), (SCAN_NONE, 0x18), "Ctrl-X");
        assert_eq!(typed(&mut keyboard, 46), (SCAN_NONE, 0x03), "Ctrl-C");
    }

    #[test]
    fn navigation_and_function_keys_are_scan_codes() {
        let mut keyboard = Keyboard::default();
        assert_eq!(typed(&mut keyboard, KEY_UP), (SCAN_UP, 0));
        assert_eq!(typed(&mut keyboard, KEY_DOWN), (SCAN_DOWN, 0));
        assert_eq!(typed(&mut keyboard, KEY_ESC), (SCAN_ESCAPE, 0));
        assert_eq!(typed(&mut keyboard, KEY_F1), (SCAN_F1, 0));
        assert_eq!(typed(&mut keyboard, KEY_F10), (SCAN_F1 + 9, 0));
        assert_eq!(typed(&mut keyboard, KEY_F12), (SCAN_F12, 0));
        assert_eq!(typed(&mut keyboard, KEY_PAGEDOWN), (SCAN_PAGE_DOWN, 0));
        assert_eq!(
            typed(&mut keyboard, KEY_ENTER),
            (SCAN_NONE, CARRIAGE_RETURN)
        );
        assert_eq!(
            typed(&mut keyboard, KEY_KPENTER),
            (SCAN_NONE, CARRIAGE_RETURN)
        );
        assert_eq!(typed(&mut keyboard, KEY_BACKSPACE), (SCAN_NONE, BACKSPACE));
    }

    #[test]
    fn the_keypad_types_digits() {
        let mut keyboard = Keyboard::default();
        assert_eq!(text(&mut keyboard, &[KEY_KP7, 82, KEY_KPDOT]), "70.");
    }

    const MS: u64 = 1_000_000;

    fn press(typematic: &mut Typematic, keyboard: &mut Keyboard, code: u16, now_ns: u64) {
        let typed = keyboard.event(code, true);
        typematic.event(code, true, typed, now_ns);
    }

    fn release(typematic: &mut Typematic, keyboard: &mut Keyboard, code: u16) {
        let typed = keyboard.event(code, false);
        typematic.event(code, false, typed, 0);
    }

    fn unicode(key: Option<InputKey>) -> Option<u16> {
        key.map(|key| key.unicode_char)
    }

    #[test]
    fn a_held_key_repeats_after_the_delay_at_the_period() {
        let (mut keyboard, mut typematic) = (Keyboard::default(), Typematic::default());
        press(&mut typematic, &mut keyboard, 30, 1000 * MS);
        assert!(typematic.due(1000 * MS + REPEAT_DELAY_NS - 1).is_none());
        assert_eq!(
            unicode(typematic.due(1000 * MS + REPEAT_DELAY_NS)),
            Some('a' as u16)
        );
        let first = 1000 * MS + REPEAT_DELAY_NS;
        assert!(typematic.due(first + REPEAT_PERIOD_NS - 1).is_none());
        assert_eq!(
            unicode(typematic.due(first + REPEAT_PERIOD_NS)),
            Some('a' as u16)
        );
        release(&mut typematic, &mut keyboard, 30);
        assert!(typematic.due(first + 100 * REPEAT_PERIOD_NS).is_none());
    }

    #[test]
    fn a_late_reader_gets_one_repeat_not_a_burst() {
        let (mut keyboard, mut typematic) = (Keyboard::default(), Typematic::default());
        press(&mut typematic, &mut keyboard, KEY_DOWN, 0);
        let late = REPEAT_DELAY_NS + 50 * REPEAT_PERIOD_NS;
        assert!(typematic.due(late).is_some());
        assert!(typematic.due(late).is_none());
        assert!(typematic.due(late + REPEAT_PERIOD_NS).is_some());
    }

    #[test]
    fn the_last_key_pressed_repeats_and_modifiers_do_not_stop_it() {
        let (mut keyboard, mut typematic) = (Keyboard::default(), Typematic::default());
        press(&mut typematic, &mut keyboard, 30, 0);
        press(&mut typematic, &mut keyboard, 48, 10 * MS);
        // Letting go of the first key does not stop the second, which is the one repeating.
        release(&mut typematic, &mut keyboard, 30);
        press(&mut typematic, &mut keyboard, KEY_LEFTSHIFT, 20 * MS);
        assert_eq!(
            unicode(typematic.due(10 * MS + REPEAT_DELAY_NS)),
            Some('b' as u16),
            "the keystroke as typed when pressed"
        );
        release(&mut typematic, &mut keyboard, 48);
        assert!(typematic.due(10 * MS + 2 * REPEAT_DELAY_NS).is_none());
    }

    #[test]
    fn releases_modifiers_and_unknown_keys_type_nothing() {
        let mut keyboard = Keyboard::default();
        assert!(keyboard.event(30, false).is_none());
        assert!(keyboard.event(KEY_LEFTSHIFT, true).is_none());
        assert!(keyboard.event(KEY_CAPSLOCK, true).is_none());
        assert!(keyboard.event(240, true).is_none());
    }
}
