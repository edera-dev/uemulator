//! The firmware's text console drawn on the framebuffer: an 80x25 grid of characters and
//! attributes, kept in step with what `console.rs` sends to the serial terminal, and painted
//! with an 8x16 font (GNU Unifont glyphs, `font8x16.bin`, built by `hack/uefi/make-font.py`)
//! scaled up to fill as much of the screen as whole pixels allow. Hardware independent: on the
//! target the pixels are the boot framebuffer in RAM, in the tests a `Vec`.

use alloc::vec;
use alloc::vec::Vec;

pub const COLUMNS: usize = 80;
pub const ROWS: usize = 25;
pub const GLYPH_WIDTH: usize = 8;
pub const GLYPH_HEIGHT: usize = 16;

/// `font8x16.bin`: entries of a little-endian u16 code point and 16 rows, sorted.
const FONT: &[u8] = include_bytes!("font8x16.bin");
const FONT_ENTRY_BYTES: usize = 2 + GLYPH_HEIGHT;
/// Drawn for a character the font does not have.
const REPLACEMENT: u16 = 0xfffd;
/// The cursor is the bottom two glyph rows in the foreground colour, as on a VGA text screen.
const CURSOR_ROWS: usize = 2;

/// `EFI_TEXT_ATTR`: foreground in the low nibble, background in the next three bits.
const FOREGROUND_MASK: u8 = 0x0f;
const BACKGROUND_SHIFT: u8 = 4;
const BACKGROUND_MASK: u8 = 0x07;
pub const DEFAULT_ATTRIBUTE: u8 = 0x07;

/// The sixteen EFI text colours (`EFI_BLACK` to `EFI_WHITE`) as 0x00RRGGBB, the VGA palette.
const PALETTE: [u32; 16] = [
    0x000000, 0x0000aa, 0x00aa00, 0x00aaaa, 0xaa0000, 0xaa00aa, 0xaa5500, 0xaaaaaa, 0x555555,
    0x5555ff, 0x55ff55, 0x55ffff, 0xff5555, 0xff55ff, 0xffff55, 0xffffff,
];

/// The 16 rows of `character`'s glyph, or of the replacement character.
pub fn glyph(character: char) -> &'static [u8] {
    let find = |code: u16| -> Option<&'static [u8]> {
        let count = FONT.len() / FONT_ENTRY_BYTES;
        let (mut low, mut high) = (0, count);
        while low < high {
            let middle = (low + high) / 2;
            let at = middle * FONT_ENTRY_BYTES;
            let entry_code = u16::from_le_bytes([FONT[at], FONT[at + 1]]);
            match entry_code.cmp(&code) {
                core::cmp::Ordering::Less => low = middle + 1,
                core::cmp::Ordering::Greater => high = middle,
                core::cmp::Ordering::Equal => return Some(&FONT[at + 2..at + FONT_ENTRY_BYTES]),
            }
        }
        None
    };
    u16::try_from(character as u32)
        .ok()
        .and_then(find)
        .or_else(|| find(REPLACEMENT))
        .expect("the font has the replacement character")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell {
    character: char,
    attribute: u8,
}

const BLANK: Cell = Cell {
    character: ' ',
    attribute: DEFAULT_ATTRIBUTE,
};

/// Where the grid lands on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub pixels_per_line: usize,
    /// Each font pixel is a `scale` x `scale` square.
    pub scale: usize,
    pub left: usize,
    pub top: usize,
}

impl Geometry {
    /// The largest whole scale at which 80x25 cells fit, centred. `None` for a screen too small
    /// for even scale 1.
    pub fn fit(width: usize, height: usize, pixels_per_line: usize) -> Option<Geometry> {
        let scale = (width / (COLUMNS * GLYPH_WIDTH)).min(height / (ROWS * GLYPH_HEIGHT));
        (scale > 0).then(|| Geometry {
            width,
            height,
            pixels_per_line,
            scale,
            left: (width - COLUMNS * GLYPH_WIDTH * scale) / 2,
            top: (height - ROWS * GLYPH_HEIGHT * scale) / 2,
        })
    }
}

pub struct TextScreen {
    geometry: Geometry,
    cells: Vec<Cell>,
    column: usize,
    row: usize,
    attribute: u8,
    cursor_visible: bool,
}

impl TextScreen {
    pub fn new(geometry: Geometry) -> TextScreen {
        TextScreen {
            geometry,
            cells: vec![BLANK; COLUMNS * ROWS],
            column: 0,
            row: 0,
            attribute: DEFAULT_ATTRIBUTE,
            cursor_visible: true,
        }
    }

    #[cfg(test)]
    pub fn cursor(&self) -> (usize, usize) {
        (self.column, self.row)
    }

    /// Paints one cell from the grid, with the cursor if it is there and shown.
    fn paint_cell(&self, pixels: &mut [u32], column: usize, row: usize) {
        let cell = self.cells[row * COLUMNS + column];
        let foreground = PALETTE[(cell.attribute & FOREGROUND_MASK) as usize];
        let background = PALETTE[((cell.attribute >> BACKGROUND_SHIFT) & BACKGROUND_MASK) as usize];
        let with_cursor = self.cursor_visible && (column, row) == (self.column, self.row);
        let geometry = &self.geometry;
        let scale = geometry.scale;
        let rows = glyph(cell.character);
        let left = geometry.left + column * GLYPH_WIDTH * scale;
        let top = geometry.top + row * GLYPH_HEIGHT * scale;
        for (glyph_row, bits) in rows.iter().enumerate() {
            let cursor_row = with_cursor && glyph_row >= GLYPH_HEIGHT - CURSOR_ROWS;
            for repeat in 0..scale {
                let line = (top + glyph_row * scale + repeat) * geometry.pixels_per_line;
                for glyph_column in 0..GLYPH_WIDTH {
                    let lit = cursor_row || bits & (0x80 >> glyph_column) != 0;
                    let colour = if lit { foreground } else { background };
                    let start = line + left + glyph_column * scale;
                    pixels[start..start + scale].fill(colour);
                }
            }
        }
    }

    /// Everything again: the margins in black and every cell.
    pub fn repaint(&self, pixels: &mut [u32]) {
        let geometry = &self.geometry;
        for row in 0..geometry.height {
            let start = row * geometry.pixels_per_line;
            pixels[start..start + geometry.width].fill(PALETTE[0]);
        }
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                self.paint_cell(pixels, column, row);
            }
        }
    }

    /// Moves the grid up a line: the pixels move with it (the framebuffer is RAM, so this is a
    /// copy, not a redraw), and the new bottom line is blank in the current background.
    fn scroll(&mut self, pixels: &mut [u32]) {
        self.cells.copy_within(COLUMNS.., 0);
        let blank = Cell {
            character: ' ',
            attribute: self.attribute,
        };
        self.cells[(ROWS - 1) * COLUMNS..].fill(blank);
        let geometry = self.geometry;
        let line_height = GLYPH_HEIGHT * geometry.scale;
        let span = COLUMNS * GLYPH_WIDTH * geometry.scale;
        for row in geometry.top..geometry.top + (ROWS - 1) * line_height {
            let target = row * geometry.pixels_per_line + geometry.left;
            let source = target + line_height * geometry.pixels_per_line;
            pixels.copy_within(source..source + span, target);
        }
        for column in 0..COLUMNS {
            self.paint_cell(pixels, column, ROWS - 1);
        }
    }

    /// Moves the cursor, repainting the cells it leaves and enters.
    fn move_cursor(&mut self, pixels: &mut [u32], column: usize, row: usize) {
        let (old_column, old_row) = (self.column, self.row);
        self.column = column;
        self.row = row;
        self.paint_cell(pixels, old_column, old_row);
        self.paint_cell(pixels, column, row);
    }

    fn line_feed(&mut self, pixels: &mut [u32]) {
        if self.row + 1 < ROWS {
            self.move_cursor(pixels, self.column, self.row + 1);
        } else {
            self.scroll(pixels);
            self.paint_cell(pixels, self.column, self.row);
        }
    }

    /// Writes a character the way `OutputString` does: CR, LF and backspace move the cursor,
    /// anything else is placed and the cursor advances, wrapping to the next line (and scrolling
    /// at the bottom) after the last column.
    pub fn put(&mut self, pixels: &mut [u32], character: char) {
        match character {
            '\r' => self.move_cursor(pixels, 0, self.row),
            '\n' => self.line_feed(pixels),
            '\x08' => self.move_cursor(pixels, self.column.saturating_sub(1), self.row),
            _ => {
                self.cells[self.row * COLUMNS + self.column] = Cell {
                    character,
                    attribute: self.attribute,
                };
                if self.column + 1 < COLUMNS {
                    self.move_cursor(pixels, self.column + 1, self.row);
                } else {
                    self.move_cursor(pixels, 0, self.row);
                    self.line_feed(pixels);
                }
            }
        }
    }

    pub fn set_attribute(&mut self, attribute: u8) {
        self.attribute = attribute;
    }

    /// Clears to the current background and homes the cursor.
    pub fn clear(&mut self, pixels: &mut [u32]) {
        let blank = Cell {
            character: ' ',
            attribute: self.attribute,
        };
        self.cells.fill(blank);
        self.column = 0;
        self.row = 0;
        self.repaint(pixels);
    }

    /// Positions outside the grid are ignored (the caller has already refused them).
    pub fn set_cursor(&mut self, pixels: &mut [u32], column: usize, row: usize) {
        if column < COLUMNS && row < ROWS {
            self.move_cursor(pixels, column, row);
        }
    }

    pub fn show_cursor(&mut self, pixels: &mut [u32], visible: bool) {
        self.cursor_visible = visible;
        self.paint_cell(pixels, self.column, self.row);
    }
}

/// The control sequences the firmware's own screens (`boot.rs`, `boot_menu.rs`) write to the
/// serial terminal, interpreted for the framebuffer so that both show the same: SGR (`m`: 0
/// resets, 7 is reverse video, 30-37 and 40-47 the colours), erase display (`2J`), cursor
/// position (`H`). Anything else is swallowed.
#[derive(Default)]
pub struct AnsiParser {
    state: AnsiState,
    parameters: Vec<u16>,
}

#[derive(Default, PartialEq, Eq)]
enum AnsiState {
    #[default]
    Text,
    Escape,
    ControlSequence,
}

const ESCAPE: char = '\x1b';
const SGR_RESET: u16 = 0;
const SGR_REVERSE: u16 = 7;
const SGR_FOREGROUND: core::ops::RangeInclusive<u16> = 30..=37;
const SGR_BACKGROUND: core::ops::RangeInclusive<u16> = 40..=47;
/// ANSI numbers red 1, green 2, blue 4; EFI numbers blue 1, green 2, red 4.
const EFI_FROM_ANSI: [u8; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
const ERASE_ALL: u16 = 2;
/// A parameter list longer than any sequence the firmware writes is not one of its own.
const MAX_PARAMETERS: usize = 8;

impl AnsiParser {
    pub fn feed(&mut self, screen: &mut TextScreen, pixels: &mut [u32], character: char) {
        match self.state {
            AnsiState::Text if character == ESCAPE => self.state = AnsiState::Escape,
            AnsiState::Text => screen.put(pixels, character),
            AnsiState::Escape if character == '[' => {
                self.parameters.clear();
                self.parameters.push(0);
                self.state = AnsiState::ControlSequence;
            }
            AnsiState::Escape => self.state = AnsiState::Text,
            AnsiState::ControlSequence => match character {
                '0'..='9' => {
                    if let Some(last) = self.parameters.last_mut() {
                        *last = last
                            .saturating_mul(10)
                            .saturating_add(character as u16 - '0' as u16);
                    }
                }
                ';' if self.parameters.len() < MAX_PARAMETERS => self.parameters.push(0),
                ';' => self.state = AnsiState::Text,
                // Private-mode markers (`?25l`, cursor visibility) carry nothing to draw.
                '?' => {}
                final_byte => {
                    self.finish(screen, pixels, final_byte);
                    self.state = AnsiState::Text;
                }
            },
        }
    }

    fn finish(&mut self, screen: &mut TextScreen, pixels: &mut [u32], final_byte: char) {
        match final_byte {
            'm' => {
                for &parameter in &self.parameters {
                    let attribute = screen.attribute;
                    screen.attribute = match parameter {
                        SGR_RESET => DEFAULT_ATTRIBUTE,
                        SGR_REVERSE => {
                            let foreground = attribute & BACKGROUND_MASK;
                            let background = (attribute >> BACKGROUND_SHIFT) & BACKGROUND_MASK;
                            background | foreground << BACKGROUND_SHIFT
                        }
                        _ if SGR_FOREGROUND.contains(&parameter) => {
                            let ansi = (parameter - SGR_FOREGROUND.start()) as usize;
                            (attribute & !FOREGROUND_MASK) | EFI_FROM_ANSI[ansi]
                        }
                        _ if SGR_BACKGROUND.contains(&parameter) => {
                            let ansi = (parameter - SGR_BACKGROUND.start()) as usize;
                            (attribute & FOREGROUND_MASK) | EFI_FROM_ANSI[ansi] << BACKGROUND_SHIFT
                        }
                        _ => attribute,
                    };
                }
            }
            'J' if self.parameters.first() == Some(&ERASE_ALL) => {
                // Erasing does not move the cursor; `clear` homes it, so put it back.
                let (column, row) = (screen.column, screen.row);
                screen.clear(pixels);
                screen.set_cursor(pixels, column, row);
            }
            'H' => {
                // One-based row;column, either may be left out.
                let row = self.parameters.first().copied().unwrap_or(1).max(1) as usize - 1;
                let column = self.parameters.get(1).copied().unwrap_or(1).max(1) as usize - 1;
                screen.set_cursor(pixels, column.min(COLUMNS - 1), row.min(ROWS - 1));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 1280;
    const HEIGHT: usize = 800;

    fn screen() -> (TextScreen, Vec<u32>) {
        let geometry = Geometry::fit(WIDTH, HEIGHT, WIDTH).unwrap();
        let screen = TextScreen::new(geometry);
        let mut pixels = vec![0x12345678; WIDTH * HEIGHT];
        screen.repaint(&mut pixels);
        (screen, pixels)
    }

    /// The colour of font pixel (x, y) of cell (column, row).
    fn font_pixel(pixels: &[u32], column: usize, row: usize, x: usize, y: usize) -> u32 {
        let geometry = Geometry::fit(WIDTH, HEIGHT, WIDTH).unwrap();
        let scale = geometry.scale;
        let px = geometry.left + (column * GLYPH_WIDTH + x) * scale;
        let py = geometry.top + (row * GLYPH_HEIGHT + y) * scale;
        pixels[py * WIDTH + px]
    }

    /// Whether cell (column, row) shows `character` in the default colours.
    fn shows(pixels: &[u32], column: usize, row: usize, character: char) -> bool {
        let rows = glyph(character);
        (0..GLYPH_HEIGHT - CURSOR_ROWS).all(|y| {
            (0..GLYPH_WIDTH).all(|x| {
                let lit = rows[y] & (0x80 >> x) != 0;
                font_pixel(pixels, column, row, x, y) == if lit { PALETTE[7] } else { PALETTE[0] }
            })
        })
    }

    #[test]
    fn the_grid_is_scaled_to_fill_the_screen_and_centred() {
        assert_eq!(
            Geometry::fit(1280, 800, 1280),
            Some(Geometry {
                width: 1280,
                height: 800,
                pixels_per_line: 1280,
                scale: 2,
                left: 0,
                top: 0
            })
        );
        let wide = Geometry::fit(1920, 1080, 2048).unwrap();
        assert_eq!((wide.scale, wide.left, wide.top), (2, 320, 140));
        assert_eq!(Geometry::fit(1024, 768, 1024).unwrap().scale, 1);
        assert_eq!(Geometry::fit(639, 480, 640), None);
    }

    #[test]
    fn the_font_has_ascii_box_drawing_and_a_fallback() {
        assert_ne!(glyph('A'), glyph('B'));
        assert_ne!(glyph('\u{2554}'), glyph('\u{fffd}'), "double box corner");
        assert_ne!(glyph('\u{2191}'), glyph('\u{fffd}'), "arrow up");
        assert_eq!(glyph('\u{4e00}'), glyph('\u{fffd}'), "not in the font");
        assert_eq!(glyph(' '), &[0u8; GLYPH_HEIGHT][..]);
    }

    #[test]
    fn characters_are_drawn_where_the_cursor_was_and_the_cursor_follows() {
        let (mut screen, mut pixels) = screen();
        for character in "Hi".chars() {
            screen.put(&mut pixels, character);
        }
        assert!(shows(&pixels, 0, 0, 'H'));
        assert!(shows(&pixels, 1, 0, 'i'));
        assert_eq!(screen.cursor(), (2, 0));
        // The cursor is drawn as the bottom rows in the foreground colour.
        assert_eq!(font_pixel(&pixels, 2, 0, 0, GLYPH_HEIGHT - 1), PALETTE[7]);
        assert_eq!(font_pixel(&pixels, 1, 0, 0, GLYPH_HEIGHT - 1), PALETTE[0]);
        screen.put(&mut pixels, '\r');
        screen.put(&mut pixels, '\n');
        assert_eq!(screen.cursor(), (0, 1));
        screen.put(&mut pixels, '\x08');
        assert_eq!(screen.cursor(), (0, 1), "backspace stops at the edge");
    }

    #[test]
    fn writing_past_the_last_column_wraps() {
        let (mut screen, mut pixels) = screen();
        for _ in 0..COLUMNS {
            screen.put(&mut pixels, 'x');
        }
        assert_eq!(screen.cursor(), (0, 1));
        screen.put(&mut pixels, 'y');
        assert!(shows(&pixels, 0, 1, 'y'));
    }

    #[test]
    fn a_line_feed_on_the_last_row_scrolls_everything_up() {
        let (mut screen, mut pixels) = screen();
        screen.put(&mut pixels, 'a');
        for _ in 0..ROWS - 1 {
            screen.put(&mut pixels, '\n');
        }
        screen.put(&mut pixels, 'z');
        assert_eq!(screen.cursor(), (2, ROWS - 1));
        assert!(shows(&pixels, 0, 0, 'a'), "not scrolled yet");
        screen.put(&mut pixels, '\n');
        assert_eq!(screen.cursor(), (2, ROWS - 1));
        assert!(shows(&pixels, 0, 0, ' '), "the first line scrolled away");
        assert!(shows(&pixels, 1, ROWS - 2, 'z'));
        assert!(shows(&pixels, 1, ROWS - 1, ' '));
    }

    #[test]
    fn attributes_colour_the_cells_written_after_them() {
        let (mut screen, mut pixels) = screen();
        const YELLOW_ON_BLUE: u8 = 0x1e;
        screen.set_attribute(YELLOW_ON_BLUE);
        screen.put(&mut pixels, '\u{2588}'); // full block: every pixel foreground
        assert_eq!(font_pixel(&pixels, 0, 0, 3, 3), PALETTE[0xe]);
        screen.put(&mut pixels, ' ');
        assert_eq!(font_pixel(&pixels, 1, 0, 3, 3), PALETTE[1]);
        screen.clear(&mut pixels);
        assert_eq!(
            font_pixel(&pixels, 40, 12, 3, 3),
            PALETTE[1],
            "cleared to the background"
        );
        assert_eq!(screen.cursor(), (0, 0));
    }

    #[test]
    fn the_cursor_can_be_hidden_and_positioned() {
        let (mut screen, mut pixels) = screen();
        screen.set_cursor(&mut pixels, 10, 5);
        assert_eq!(font_pixel(&pixels, 10, 5, 0, GLYPH_HEIGHT - 1), PALETTE[7]);
        screen.show_cursor(&mut pixels, false);
        assert_eq!(font_pixel(&pixels, 10, 5, 0, GLYPH_HEIGHT - 1), PALETTE[0]);
        screen.set_cursor(&mut pixels, COLUMNS, 0);
        assert_eq!(screen.cursor(), (10, 5), "outside the grid is ignored");
    }

    fn feed(parser: &mut AnsiParser, screen: &mut TextScreen, pixels: &mut [u32], text: &str) {
        for character in text.chars() {
            parser.feed(screen, pixels, character);
        }
    }

    #[test]
    fn the_boot_menus_control_sequences_are_drawn_not_printed() {
        let (mut screen, mut pixels) = screen();
        let mut parser = AnsiParser::default();
        let menu = "junk\x1b[2J\x1b[Hmenu\r\n  \x1b[7m item \x1b[0m";
        feed(&mut parser, &mut screen, &mut pixels, menu);
        assert!(
            shows(&pixels, 0, 0, 'm'),
            "cleared, then written from the top"
        );
        assert!(shows(&pixels, 4, 0, ' '), "the old text is gone");
        // Reverse video: the selected item is light grey with black letters.
        assert_eq!(font_pixel(&pixels, 2, 1, 0, 0), PALETTE[7]);
        feed(&mut parser, &mut screen, &mut pixels, "x");
        assert_eq!(
            font_pixel(&pixels, 8, 1, 0, 0),
            PALETTE[0],
            "back to normal"
        );
        assert_eq!(screen.cursor(), (9, 1));
    }

    #[test]
    fn cursor_positions_are_one_based_and_clamped() {
        let (mut screen, mut pixels) = screen();
        let mut parser = AnsiParser::default();
        feed(&mut parser, &mut screen, &mut pixels, "\x1b[3;5H");
        assert_eq!(screen.cursor(), (4, 2));
        feed(&mut parser, &mut screen, &mut pixels, "\x1b[99;999H");
        assert_eq!(screen.cursor(), (COLUMNS - 1, ROWS - 1));
        feed(&mut parser, &mut screen, &mut pixels, "\x1b[H");
        assert_eq!(screen.cursor(), (0, 0));
    }

    #[test]
    fn colours_and_unknown_sequences() {
        let (mut screen, mut pixels) = screen();
        let mut parser = AnsiParser::default();
        // Red on blue in ANSI numbering, then an unknown sequence and a private one.
        let text = "\x1b[31;44m\u{2588}\x1b[5n\x1b[?25l ";
        feed(&mut parser, &mut screen, &mut pixels, text);
        assert_eq!(font_pixel(&pixels, 0, 0, 3, 3), PALETTE[4]);
        assert_eq!(font_pixel(&pixels, 1, 0, 3, 3), PALETTE[1]);
        assert_eq!(
            screen.cursor(),
            (2, 0),
            "only the two characters were drawn"
        );
    }

    #[test]
    fn margins_around_a_centred_grid_are_black() {
        let geometry = Geometry::fit(1920, 1080, 1920).unwrap();
        let screen = TextScreen::new(geometry);
        let mut pixels = vec![0xdeadbeef; 1920 * 1080];
        screen.repaint(&mut pixels);
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[1920 * 1080 - 1], 0);
    }
}
