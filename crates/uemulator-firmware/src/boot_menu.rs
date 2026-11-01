//! The boot menu and the countdown before it: which key does what, and the text drawn on the
//! serial terminal. The waiting and booting are in `boot.rs`.

use alloc::string::String;
use core::fmt::Write;

use crate::keys::{
    SCAN_DOWN, SCAN_END, SCAN_ESCAPE, SCAN_F2, SCAN_F12, SCAN_HOME, SCAN_PAGE_DOWN, SCAN_PAGE_UP,
    SCAN_UP,
};
use crate::uefi::InputKey;

const CARRIAGE_RETURN: u16 = 0x0d;
const LINE_FEED: u16 = 0x0a;
/// `Timeout` (UEFI 2.9 3.3): seconds before the default boot; this value waits for the user.
pub const WAIT_FOREVER: u16 = 0xffff;

const CLEAR_SCREEN: &str = "\x1b[2J\x1b[H";
const REVERSE_VIDEO: &str = "\x1b[7m";
const NORMAL_VIDEO: &str = "\x1b[0m";

fn is_enter(key: &InputKey) -> bool {
    key.scan_code == 0 && matches!(key.unicode_char, CARRIAGE_RETURN | LINE_FEED)
}

/// What a key pressed during the countdown asks for.
#[derive(Debug, PartialEq, Eq)]
pub enum Hotkey {
    /// Escape, F2 or F12: the keys PC firmware uses for its setup and boot menus.
    Menu,
    BootNow,
    Ignored,
}

pub fn hotkey(key: &InputKey) -> Hotkey {
    match key.scan_code {
        SCAN_ESCAPE | SCAN_F2 | SCAN_F12 => Hotkey::Menu,
        _ if is_enter(key) => Hotkey::BootNow,
        _ => Hotkey::Ignored,
    }
}

/// The line the countdown redraws in place each second.
pub fn countdown_line(seconds_left: u16) -> String {
    alloc::format!(
        "\rPress Esc for the boot menu, Enter to boot now. Booting in {seconds_left} s ... "
    )
}

#[derive(Debug, PartialEq, Eq)]
pub enum Choice {
    Stay,
    Pick(usize),
    /// Escape: carry on with the normal boot order.
    Leave,
}

pub struct Menu {
    pub selected: usize,
    count: usize,
}

impl Menu {
    pub fn new(count: usize) -> Self {
        Menu { selected: 0, count }
    }

    pub fn handle(&mut self, key: &InputKey) -> Choice {
        let last = self.count.saturating_sub(1);
        match key.scan_code {
            SCAN_UP => self.selected = self.selected.checked_sub(1).unwrap_or(last),
            SCAN_DOWN => {
                self.selected = if self.selected >= last {
                    0
                } else {
                    self.selected + 1
                }
            }
            SCAN_HOME | SCAN_PAGE_UP => self.selected = 0,
            SCAN_END | SCAN_PAGE_DOWN => self.selected = last,
            SCAN_ESCAPE => return Choice::Leave,
            _ if is_enter(key) && self.count > 0 => return Choice::Pick(self.selected),
            _ => {}
        }
        Choice::Stay
    }

    /// The whole screen: title, one line per item with the selection in reverse video, help.
    /// Lines end in `\n` alone; the serial writer adds the carriage return.
    pub fn render(&self, items: &[String]) -> String {
        let mut text = String::from(CLEAR_SCREEN);
        text.push_str("uemulator boot menu\n\n");
        for (index, item) in items.iter().enumerate() {
            if index == self.selected {
                let _ = writeln!(text, "  {REVERSE_VIDEO} {item} {NORMAL_VIDEO}");
            } else {
                let _ = writeln!(text, "   {item}");
            }
        }
        text.push_str("\nUp/Down to select, Enter to boot, Esc to boot in the usual order\n");
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(scan_code: u16) -> InputKey {
        InputKey {
            scan_code,
            unicode_char: 0,
        }
    }

    fn character(unit: u16) -> InputKey {
        InputKey {
            scan_code: 0,
            unicode_char: unit,
        }
    }

    #[test]
    fn arrows_move_and_wrap() {
        let mut menu = Menu::new(3);
        assert_eq!(menu.handle(&scan(SCAN_DOWN)), Choice::Stay);
        assert_eq!(menu.selected, 1);
        menu.handle(&scan(SCAN_DOWN));
        menu.handle(&scan(SCAN_DOWN));
        assert_eq!(menu.selected, 0);
        menu.handle(&scan(SCAN_UP));
        assert_eq!(menu.selected, 2);
        menu.handle(&scan(SCAN_HOME));
        assert_eq!(menu.selected, 0);
        menu.handle(&scan(SCAN_END));
        assert_eq!(menu.selected, 2);
    }

    #[test]
    fn enter_picks_and_escape_leaves() {
        let mut menu = Menu::new(2);
        menu.handle(&scan(SCAN_DOWN));
        assert_eq!(menu.handle(&character(CARRIAGE_RETURN)), Choice::Pick(1));
        assert_eq!(menu.handle(&character(LINE_FEED)), Choice::Pick(1));
        assert_eq!(menu.handle(&scan(SCAN_ESCAPE)), Choice::Leave);
        assert_eq!(menu.handle(&character('x' as u16)), Choice::Stay);
    }

    #[test]
    fn an_empty_menu_picks_nothing() {
        let mut menu = Menu::new(0);
        menu.handle(&scan(SCAN_DOWN));
        menu.handle(&scan(SCAN_UP));
        assert_eq!(menu.handle(&character(CARRIAGE_RETURN)), Choice::Stay);
    }

    #[test]
    fn countdown_keys() {
        assert_eq!(hotkey(&scan(SCAN_ESCAPE)), Hotkey::Menu);
        assert_eq!(hotkey(&scan(SCAN_F2)), Hotkey::Menu);
        assert_eq!(hotkey(&scan(SCAN_F12)), Hotkey::Menu);
        assert_eq!(hotkey(&character(CARRIAGE_RETURN)), Hotkey::BootNow);
        assert_eq!(hotkey(&character('a' as u16)), Hotkey::Ignored);
        assert_eq!(hotkey(&scan(SCAN_DOWN)), Hotkey::Ignored);
    }

    #[test]
    fn the_selection_is_drawn_in_reverse_video() {
        let mut menu = Menu::new(2);
        menu.handle(&scan(SCAN_DOWN));
        let items = [String::from("Alpine"), String::from("Debian")];
        let text = menu.render(&items);
        assert!(text.starts_with(CLEAR_SCREEN));
        assert!(text.contains("   Alpine\n"));
        assert!(text.contains(&alloc::format!("{REVERSE_VIDEO} Debian {NORMAL_VIDEO}")));
    }
}
