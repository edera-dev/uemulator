//! The text console on the 16550: `SimpleTextOutput` writes UTF-8 and ANSI escape sequences to
//! a terminal, `SimpleTextInput(Ex)` reads keystrokes from it. A machine with a screen gets the
//! same text drawn on its framebuffer (`text_screen.rs`) and keystrokes from its virtio keyboard
//! too, like a console splitter: both consoles always show the same thing.

use core::ffi::c_void;
use core::ptr;

use crate::events;
use crate::gop;
use crate::handoff::Framebuffer;
use crate::keys;
use crate::protocols;
use crate::serial;
use crate::sync::Global;
use crate::tables::SYSTEM_TABLE;
use crate::text_screen::{AnsiParser, Geometry, TextScreen};
use crate::time;
use crate::uefi::*;
use crate::virtio_input;

const COLUMNS: usize = 80;
const ROWS: usize = 25;
const MODE_80X25: usize = 0;
const MODE_COUNT: i32 = 1;
const ATTRIBUTE_DEFAULT: i32 = 0x07;
const ESCAPE: &str = "\x1b";

/// `EFI_TEXT_ATTR`: foreground in the low nibble, background in the next three bits.
const FOREGROUND_MASK: usize = 0x0f;
const BACKGROUND_SHIFT: usize = 4;
const BACKGROUND_MASK: usize = 0x07;
/// EFI numbers colours blue=1, green=2, red=4; ANSI numbers red=1, green=2, blue=4.
const ANSI_FROM_EFI: [u8; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
const BRIGHT: usize = 8;
/// The attribute bits `EFI_TEXT_ATTR` defines (the top bit is reserved).
const ATTRIBUTE_MASK: usize = 0x7f;

#[repr(C)]
struct KeyState {
    shift_state: u32,
    toggle_state: u8,
}

#[repr(C)]
struct KeyData {
    key: InputKey,
    state: KeyState,
}

#[repr(C)]
struct SimpleTextInputEx {
    reset: extern "efiapi" fn(*mut SimpleTextInputEx, u8) -> Status,
    read_key_stroke_ex: extern "efiapi" fn(*mut SimpleTextInputEx, *mut KeyData) -> Status,
    wait_for_key_ex: Event,
    set_state: extern "efiapi" fn(*mut SimpleTextInputEx, *mut u8) -> Status,
    register_key_notify: extern "efiapi" fn(
        *mut SimpleTextInputEx,
        *mut KeyData,
        *mut c_void,
        *mut *mut c_void,
    ) -> Status,
    unregister_key_notify: extern "efiapi" fn(*mut SimpleTextInputEx, *mut c_void) -> Status,
}

struct Console {
    pending_key: Option<InputKey>,
    column: usize,
    row: usize,
}

static CONSOLE: Global<Console> = Global::new(Console {
    pending_key: None,
    column: 0,
    row: 0,
});

/// The text console on the framebuffer, with the framebuffer it draws on.
static SCREEN: Global<Option<(TextScreen, Framebuffer)>> = Global::new(None);

/// Runs `draw` on the framebuffer console if the machine has one.
fn on_screen(draw: impl FnOnce(&mut TextScreen, &mut [u32])) {
    if let Some((screen, framebuffer)) = SCREEN.get() {
        draw(screen, gop::pixels(framebuffer));
    }
}

/// Reads the control sequences in the firmware's own text (`Writer`) for the screen.
static ANSI: Global<Option<AnsiParser>> = Global::new(None);

/// The firmware's own messages and menus (`boot.rs`): to the serial terminal as they are, and
/// onto the screen with their control sequences interpreted. Lines end in `\n` alone.
pub struct Writer;

impl core::fmt::Write for Writer {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        let _ = serial::Serial.write_str(text);
        on_screen(|screen, pixels| {
            let parser = ANSI.get().get_or_insert_with(AnsiParser::default);
            for character in text.chars() {
                if character == '\n' {
                    parser.feed(screen, pixels, '\r');
                }
                parser.feed(screen, pixels, character);
            }
        });
        Ok(())
    }
}

macro_rules! print {
    ($($argument:tt)*) => {{
        use core::fmt::Write;
        let _ = write!($crate::console::Writer, $($argument)*);
    }};
}

macro_rules! println {
    () => { $crate::console::print!("\n") };
    ($($argument:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::console::Writer, $($argument)*);
    }};
}

pub(crate) use {print, println};

/// Starts drawing the console on `framebuffer` (cleared to black first).
pub fn attach_screen(framebuffer: Option<Framebuffer>) {
    let Some(framebuffer) = framebuffer else {
        return;
    };
    let Some(geometry) = Geometry::fit(
        framebuffer.width as usize,
        framebuffer.height as usize,
        framebuffer.pixels_per_line as usize,
    ) else {
        return;
    };
    let screen = TextScreen::new(geometry);
    screen.repaint(gop::pixels(&framebuffer));
    *SCREEN.get() = Some((screen, framebuffer));
}

static OUTPUT_MODE: Global<SimpleTextOutputMode> = Global::new(SimpleTextOutputMode {
    max_mode: MODE_COUNT,
    mode: MODE_80X25 as i32,
    attribute: ATTRIBUTE_DEFAULT,
    cursor_column: 0,
    cursor_row: 0,
    cursor_visible: 1,
});

static INPUT: Global<SimpleTextInput> = Global::new(SimpleTextInput {
    reset: input_reset,
    read_key_stroke,
    wait_for_key: ptr::null_mut(),
});

static INPUT_EX: Global<SimpleTextInputEx> = Global::new(SimpleTextInputEx {
    reset: input_ex_reset,
    read_key_stroke_ex,
    wait_for_key_ex: ptr::null_mut(),
    set_state,
    register_key_notify,
    unregister_key_notify,
});

static OUTPUT: Global<SimpleTextOutput> = Global::new(SimpleTextOutput {
    reset: output_reset,
    output_string,
    test_string,
    query_mode,
    set_mode,
    set_attribute,
    clear_screen,
    set_cursor_position,
    enable_cursor,
    mode: ptr::null_mut(),
});

fn emit(text: &str) {
    for byte in text.bytes() {
        serial::write_byte(byte);
    }
}

fn track(character: char) {
    let console = CONSOLE.get();
    match character {
        '\r' => console.column = 0,
        '\n' => {
            console.column = 0;
            console.row = (console.row + 1).min(ROWS - 1);
        }
        '\x08' => console.column = console.column.saturating_sub(1),
        _ => {
            console.column += 1;
            if console.column == COLUMNS {
                console.column = 0;
                console.row = (console.row + 1).min(ROWS - 1);
            }
        }
    }
    let mode = OUTPUT_MODE.get();
    mode.cursor_column = console.column as i32;
    mode.cursor_row = console.row as i32;
}

extern "efiapi" fn output_reset(_: *mut SimpleTextOutput, _: u8) -> Status {
    SUCCESS
}

extern "efiapi" fn output_string(_: *mut SimpleTextOutput, string: *const u16) -> Status {
    if string.is_null() {
        return INVALID_PARAMETER;
    }
    let mut at = 0;
    let mut utf8 = [0u8; 4];
    loop {
        let unit = unsafe { string.add(at).read_unaligned() };
        if unit == 0 {
            return SUCCESS;
        }
        at += 1;
        let character = char::from_u32(unit as u32).unwrap_or(char::REPLACEMENT_CHARACTER);
        track(character);
        if character == '\n' {
            serial::write_byte(b'\r');
        }
        // The terminal turns the line feed into CR LF (above), so the screen does too.
        on_screen(|screen, pixels| {
            if character == '\n' {
                screen.put(pixels, '\r');
            }
            screen.put(pixels, character);
        });
        for byte in character.encode_utf8(&mut utf8).bytes() {
            serial::write_byte(byte);
        }
    }
}

/// Everything in the basic plane prints on a UTF-8 terminal, box-drawing characters included.
extern "efiapi" fn test_string(_: *mut SimpleTextOutput, _: *const u16) -> Status {
    SUCCESS
}

extern "efiapi" fn query_mode(
    _: *mut SimpleTextOutput,
    mode: usize,
    columns: *mut usize,
    rows: *mut usize,
) -> Status {
    if mode != MODE_80X25 {
        return UNSUPPORTED;
    }
    if columns.is_null() || rows.is_null() {
        return INVALID_PARAMETER;
    }
    unsafe {
        columns.write(COLUMNS);
        rows.write(ROWS);
    }
    SUCCESS
}

extern "efiapi" fn set_mode(this: *mut SimpleTextOutput, mode: usize) -> Status {
    if mode != MODE_80X25 {
        return UNSUPPORTED;
    }
    clear_screen(this)
}

extern "efiapi" fn set_attribute(_: *mut SimpleTextOutput, attribute: usize) -> Status {
    let foreground = attribute & FOREGROUND_MASK;
    let background = (attribute >> BACKGROUND_SHIFT) & BACKGROUND_MASK;
    let ansi_foreground =
        30 + ANSI_FROM_EFI[foreground & 7] as usize + if foreground & BRIGHT != 0 { 60 } else { 0 };
    let ansi_background = 40 + ANSI_FROM_EFI[background] as usize;
    let mut sequence = alloc::string::String::new();
    use core::fmt::Write;
    let _ = write!(sequence, "{ESCAPE}[0;{ansi_foreground};{ansi_background}m");
    emit(&sequence);
    OUTPUT_MODE.get().attribute = attribute as i32;
    on_screen(|screen, _| screen.set_attribute((attribute & ATTRIBUTE_MASK) as u8));
    SUCCESS
}

extern "efiapi" fn clear_screen(_: *mut SimpleTextOutput) -> Status {
    emit("\x1b[2J\x1b[H");
    on_screen(|screen, pixels| screen.clear(pixels));
    let console = CONSOLE.get();
    console.column = 0;
    console.row = 0;
    track('\r');
    SUCCESS
}

extern "efiapi" fn set_cursor_position(
    _: *mut SimpleTextOutput,
    column: usize,
    row: usize,
) -> Status {
    if column >= COLUMNS || row >= ROWS {
        return UNSUPPORTED;
    }
    let mut sequence = alloc::string::String::new();
    use core::fmt::Write;
    let _ = write!(sequence, "{ESCAPE}[{};{}H", row + 1, column + 1);
    emit(&sequence);
    on_screen(|screen, pixels| screen.set_cursor(pixels, column, row));
    let console = CONSOLE.get();
    console.column = column;
    console.row = row;
    track('\r');
    console.column = column;
    OUTPUT_MODE.get().cursor_column = column as i32;
    SUCCESS
}

extern "efiapi" fn enable_cursor(_: *mut SimpleTextOutput, visible: u8) -> Status {
    emit(if visible != 0 {
        "\x1b[?25h"
    } else {
        "\x1b[?25l"
    });
    OUTPUT_MODE.get().cursor_visible = (visible != 0) as u8;
    on_screen(|screen, pixels| screen.show_cursor(pixels, visible != 0));
    SUCCESS
}

/// Decodes one waiting keystroke into the lookahead slot.
fn fill_pending() {
    let console = CONSOLE.get();
    if console.pending_key.is_none() {
        console.pending_key = virtio_input::read_key();
    }
    if console.pending_key.is_none() {
        console.pending_key = keys::decode(serial::read_byte, || {
            keys::following_byte(serial::read_byte, time::now_ns)
        });
    }
}

extern "efiapi" fn wait_for_key_notify(event: Event, _: *mut c_void) {
    fill_pending();
    if CONSOLE.get().pending_key.is_some() {
        events::signal(event);
    }
}

fn take_key() -> Option<InputKey> {
    fill_pending();
    CONSOLE.get().pending_key.take()
}

/// A waiting keystroke, for the boot manager's own menu.
pub fn read_key() -> Option<InputKey> {
    take_key()
}

extern "efiapi" fn input_reset(_: *mut SimpleTextInput, _: u8) -> Status {
    CONSOLE.get().pending_key = None;
    SUCCESS
}

extern "efiapi" fn read_key_stroke(_: *mut SimpleTextInput, key: *mut InputKey) -> Status {
    if key.is_null() {
        return INVALID_PARAMETER;
    }
    match take_key() {
        Some(pressed) => {
            unsafe { key.write_unaligned(pressed) };
            SUCCESS
        }
        None => NOT_READY,
    }
}

extern "efiapi" fn input_ex_reset(_: *mut SimpleTextInputEx, _: u8) -> Status {
    CONSOLE.get().pending_key = None;
    SUCCESS
}

extern "efiapi" fn read_key_stroke_ex(_: *mut SimpleTextInputEx, data: *mut KeyData) -> Status {
    if data.is_null() {
        return INVALID_PARAMETER;
    }
    match take_key() {
        Some(pressed) => {
            unsafe {
                data.write_unaligned(KeyData {
                    key: pressed,
                    state: KeyState {
                        shift_state: 0,
                        toggle_state: 0,
                    },
                });
            }
            SUCCESS
        }
        None => NOT_READY,
    }
}

extern "efiapi" fn set_state(_: *mut SimpleTextInputEx, _: *mut u8) -> Status {
    SUCCESS
}

/// Key notifications never fire; callers that register one get a token and no callbacks.
extern "efiapi" fn register_key_notify(
    _: *mut SimpleTextInputEx,
    _: *mut KeyData,
    _: *mut c_void,
    handle: *mut *mut c_void,
) -> Status {
    if !handle.is_null() {
        unsafe { handle.write(ptr::dangling_mut()) };
    }
    SUCCESS
}

extern "efiapi" fn unregister_key_notify(_: *mut SimpleTextInputEx, _: *mut c_void) -> Status {
    SUCCESS
}

pub fn init() {
    OUTPUT.get().mode = OUTPUT_MODE.get();
    let wait = events::create(
        EVT_NOTIFY_WAIT,
        TPL_CALLBACK,
        Some(wait_for_key_notify),
        ptr::null_mut(),
        None,
    )
    .expect("console wait event");
    INPUT.get().wait_for_key = wait;
    INPUT_EX.get().wait_for_key_ex = wait;
    let handle = protocols::install(
        ptr::null_mut(),
        guids::SIMPLE_TEXT_INPUT,
        (INPUT.get() as *mut SimpleTextInput).cast(),
    )
    .expect("console handle");
    let _ = protocols::install(
        handle,
        guids::SIMPLE_TEXT_INPUT_EX,
        (INPUT_EX.get() as *mut SimpleTextInputEx).cast(),
    );
    let _ = protocols::install(
        handle,
        guids::SIMPLE_TEXT_OUTPUT,
        (OUTPUT.get() as *mut SimpleTextOutput).cast(),
    );
    let system = SYSTEM_TABLE.get();
    system.console_in_handle = handle;
    system.con_in = INPUT.get();
    system.console_out_handle = handle;
    system.con_out = OUTPUT.get();
    system.standard_error_handle = handle;
    system.std_err = OUTPUT.get();
}

/// After `ExitBootServices` the console tables are gone.
pub fn detach() {
    let system = SYSTEM_TABLE.get();
    system.console_in_handle = ptr::null_mut();
    system.con_in = ptr::null_mut();
    system.console_out_handle = ptr::null_mut();
    system.con_out = ptr::null_mut();
    system.standard_error_handle = ptr::null_mut();
    system.std_err = ptr::null_mut();
}
