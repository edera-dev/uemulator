//! The host side of the graphical console: `DisplayHandle` shows what the guest's screen holds,
//! `InputHandle` feeds it a keyboard and an absolute pointer. Both are frontends' only contact
//! with the machine (the CLI's web viewer, the web page, a native window later), so nothing here
//! knows how the pixels are shown or where the keys come from. A machine may have either without
//! the other (`MachineConfig::display`, `MachineConfig::input`). Layout and protocol choices are
//! in `docs/plans/24-display-input.md`.
//!
//! Locking: the devices publish into these handles while they hold the device lock; a frontend
//! takes only the handle's own lock, never the device lock, so a slow frontend cannot stall the
//! guest beyond the copy it is making.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

use crate::sync::{Arc, Condvar, Mutex};

/// Bytes per pixel of every frame a `DisplayHandle` hands out.
pub const BYTES_PER_PIXEL: usize = 4;

/// The screen size a machine starts with unless `MachineConfig::display_size` says otherwise:
/// 16:10, what OVMF picks on QEMU, and the 8x16 font of the firmware's text console at twice its
/// size fills it with exactly 80x25 cells.
pub(crate) const DEFAULT_WIDTH: u32 = 1280;
pub(crate) const DEFAULT_HEIGHT: u32 = 800;
/// Bounds on `display_size` and on a size a frontend requests. The upper one is what the boot
/// framebuffer and the guest's resources are budgeted for (`dev/vgpu.rs`).
pub(crate) const MIN_WIDTH: u32 = 320;
pub(crate) const MIN_HEIGHT: u32 = 200;
pub(crate) const MAX_WIDTH: u32 = 2560;
pub(crate) const MAX_HEIGHT: u32 = 1600;

/// How many recent damage rectangles are kept for `update_since`. A frontend further behind than
/// this gets the whole screen, which is always correct, just bigger.
const DAMAGE_HISTORY: usize = 64;

/// Range of the tablet's absolute axes as the guest sees them (`ABS_X`, `ABS_Y`), the one QEMU's
/// virtio tablet uses. Fixed, so a mode change needs no new device configuration: pointer
/// positions are scaled from screen pixels into it.
pub(crate) const POINTER_AXIS_MAX: u32 = 0x7fff;

/// Input batches waiting for the guest per device. A guest whose driver is not up yet (or that
/// stopped reading) gets the oldest ones once it reads again; beyond this new input is refused.
const INPUT_QUEUE_BATCHES: usize = 256;

/// Linux input event types and codes (`include/uapi/linux/input-event-codes.h`) that the devices
/// send.
pub(crate) const EV_SYN: u16 = 0x00;
pub(crate) const EV_KEY: u16 = 0x01;
pub(crate) const EV_REL: u16 = 0x02;
pub(crate) const EV_ABS: u16 = 0x03;
pub(crate) const SYN_REPORT: u16 = 0;
pub(crate) const REL_WHEEL: u16 = 0x08;
pub(crate) const ABS_X: u16 = 0x00;
pub(crate) const ABS_Y: u16 = 0x01;
pub(crate) const BTN_LEFT: u16 = 0x110;
pub(crate) const BTN_RIGHT: u16 = 0x111;
pub(crate) const BTN_MIDDLE: u16 = 0x112;
pub(crate) const BTN_SIDE: u16 = 0x113;
pub(crate) const BTN_EXTRA: u16 = 0x114;
/// Key codes the keyboard accepts and advertises: `KEY_ESC` (1) up to 255, which covers every
/// key of a PC keyboard and the multimedia keys; codes above are buttons and exotic keys.
pub(crate) const KEY_CODE_MAX: u16 = 0xff;
const KEY_PRESSED: u32 = 1;
const KEY_RELEASED: u32 = 0;

/// A rectangle of the screen in pixels, origin top left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The smallest rectangle holding both; an empty one does not count.
    pub fn union(&self, other: &Rect) -> Rect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Rect {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }

    /// The part inside a `width` x `height` screen.
    pub(crate) fn clipped(&self, width: u32, height: u32) -> Rect {
        let x = self.x.min(width);
        let y = self.y.min(height);
        Rect {
            x,
            y,
            width: self.width.min(width - x),
            height: self.height.min(height - y),
        }
    }
}

/// A copy of the whole screen. `pixels` holds `height` rows of `width` pixels with no padding;
/// each pixel is four bytes, blue, green, red and one unused byte (little-endian XRGB8888, what
/// UEFI's GOP calls `PixelBlueGreenRedReserved8BitPerColor` and DRM `XRGB8888`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// The screen's change counter when the copy was made (`DisplayHandle::generation`).
    pub generation: u64,
    pub pixels: Vec<u8>,
}

/// What changed since a frontend last looked (`DisplayHandle::update_since`). The pixels of
/// `rect` were written to the caller's buffer, packed: `rect.height` rows of `rect.width` pixels
/// in `Frame`'s format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameUpdate {
    pub generation: u64,
    /// The screen size now; when it changed, `rect` is the whole screen.
    pub width: u32,
    pub height: u32,
    pub rect: Rect,
}

/// Called with the new generation after each change, see `DisplayHandle::on_update`.
type UpdateFn = Arc<dyn Fn(u64) + Send + Sync>;

/// Largest pointer image kept, per side. Linux's virtio_gpu uses 64x64; the bound only stops a
/// guest from making every frontend ship a screen-sized cursor.
pub(crate) const MAX_CURSOR_SIDE: u32 = 256;

/// A pointer image the guest drew (`Cursor::Image`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CursorImage {
    pub width: u32,
    pub height: u32,
    /// The pixel of the image that is the pointer's position.
    pub hot_x: u32,
    pub hot_y: u32,
    /// `height` rows of `width` pixels with no padding, each blue, green, red, alpha (straight
    /// alpha, not premultiplied: DRM's ARGB8888 as Linux hands it over).
    pub pixels: Vec<u8>,
}

/// What the pointer should look like over the screen (`DisplayHandle::cursor`). The guest's
/// pointer is never composited into the frames: the tablet puts it exactly under the host's, so
/// a frontend shows the image as its own pointer and pointer moves cost no screen updates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Cursor {
    /// The guest has no pointer of its own (a text console, the firmware, no GPU driver): show
    /// the host's.
    #[default]
    Host,
    /// The guest hid its pointer (a desktop over a full-screen game, a text field being typed in).
    Hidden,
    Image(CursorImage),
}

struct Screen {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    generation: u64,
    /// (generation, rectangle) of the latest changes, oldest first.
    damage: VecDeque<(u64, Rect)>,
    /// Generation that changed the size; anyone who last looked before it needs everything.
    resized_at: u64,
    /// A size the host asked the guest to switch to, until the GPU device takes it.
    requested_size: Option<(u32, u32)>,
    cursor: Cursor,
    /// Goes up with every change of `cursor`, independent of `generation`.
    cursor_generation: u64,
}

impl Screen {
    fn new(width: u32, height: u32) -> Screen {
        Screen {
            width,
            height,
            pixels: vec![0; width as usize * height as usize * BYTES_PER_PIXEL],
            generation: 0,
            damage: VecDeque::new(),
            resized_at: 0,
            requested_size: None,
            cursor: Cursor::Host,
            cursor_generation: 0,
        }
    }

    fn stride(&self) -> usize {
        self.width as usize * BYTES_PER_PIXEL
    }

    /// What changed after `generation`, or `None` when nothing did.
    fn damage_since(&self, generation: u64) -> Option<Rect> {
        if generation >= self.generation {
            return None;
        }
        let full = Rect {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        // The history must reach back to the generation right after the caller's, or changes
        // in between are unknown.
        let oldest_known = self.damage.front().map_or(u64::MAX, |(oldest, _)| *oldest);
        if generation < self.resized_at || oldest_known > generation + 1 {
            return Some(full);
        }
        Some(
            self.damage
                .iter()
                .filter(|(changed, _)| *changed > generation)
                .fold(Rect::default(), |all, (_, rect)| all.union(rect)),
        )
    }
}

pub(crate) struct DisplayShared {
    screen: Mutex<Screen>,
    changed: Condvar,
    callbacks: Mutex<Vec<UpdateFn>>,
    /// Set while the guest's virtio-gpu driver owns the screen (it has set a scanout and the
    /// device was not reset since); the boot framebuffer is not scanned then. See `dev/bootfb.rs`.
    virtio_owns_screen: AtomicBool,
}

impl Default for DisplayShared {
    fn default() -> Self {
        DisplayShared {
            screen: Mutex::new(Screen::new(DEFAULT_WIDTH, DEFAULT_HEIGHT)),
            changed: Condvar::new(),
            callbacks: Mutex::new(Vec::new()),
            virtio_owns_screen: AtomicBool::new(false),
        }
    }
}

/// The guest's screen as frontends see it, shared by clones. Obtained from
/// `EmulatedMachine::display()`; survives guest reboots.
#[derive(Clone, Default)]
pub struct DisplayHandle {
    pub(crate) shared: Arc<DisplayShared>,
}

impl DisplayHandle {
    pub(crate) fn with_size(width: u32, height: u32) -> DisplayHandle {
        DisplayHandle {
            shared: Arc::new(DisplayShared {
                screen: Mutex::new(Screen::new(width, height)),
                ..DisplayShared::default()
            }),
        }
    }

    /// Current screen size in pixels.
    pub fn size(&self) -> (u32, u32) {
        let screen = self.shared.screen.lock().unwrap();
        (screen.width, screen.height)
    }

    /// A counter that goes up with every change of the screen's contents or size, starting at 0
    /// for the blank screen a machine starts with.
    pub fn generation(&self) -> u64 {
        self.shared.screen.lock().unwrap().generation
    }

    /// A copy of the whole screen.
    pub fn frame(&self) -> Frame {
        let screen = self.shared.screen.lock().unwrap();
        Frame {
            width: screen.width,
            height: screen.height,
            generation: screen.generation,
            pixels: screen.pixels.clone(),
        }
    }

    /// The part of the screen that changed after `generation` (0 to start from the blank screen,
    /// or what a previous `FrameUpdate` returned), its pixels written into `pixels` (replacing
    /// what was there). `None` when nothing changed. Unknown history (a frontend far behind, a
    /// size change) gives the whole screen.
    pub fn update_since(&self, generation: u64, pixels: &mut Vec<u8>) -> Option<FrameUpdate> {
        let screen = self.shared.screen.lock().unwrap();
        let rect = screen.damage_since(generation)?;
        pixels.clear();
        let row_bytes = rect.width as usize * BYTES_PER_PIXEL;
        pixels.reserve(row_bytes * rect.height as usize);
        let stride = screen.stride();
        for row in rect.y..rect.y + rect.height {
            let start = row as usize * stride + rect.x as usize * BYTES_PER_PIXEL;
            pixels.extend_from_slice(&screen.pixels[start..start + row_bytes]);
        }
        Some(FrameUpdate {
            generation: screen.generation,
            width: screen.width,
            height: screen.height,
            rect,
        })
    }

    /// Waits until the generation is past `generation` or `timeout` passes, and returns the
    /// generation then. Waits once, so it may return early; a cursor change also ends the wait
    /// (with the generation unchanged), so a frontend that shows the cursor checks
    /// `cursor` after it. A freestanding host has no threads to wait for and gets the current
    /// value at once; it polls or uses `on_update`.
    pub fn wait_for_update(&self, generation: u64, timeout: Duration) -> u64 {
        let mut screen = self.shared.screen.lock().unwrap();
        if screen.generation <= generation {
            screen = self.shared.changed.wait_timeout(screen, timeout).unwrap().0;
        }
        screen.generation
    }

    /// Calls `notify` with the generation after every change of the screen or the cursor. It
    /// runs on the thread that runs the devices, while the device lock is held: it must return
    /// quickly and must not call into the machine or this handle (signal a frontend thread
    /// instead). Callbacks add up; they live as long as the handle's last clone.
    pub fn on_update(&self, notify: impl Fn(u64) + Send + Sync + 'static) {
        self.shared.callbacks.lock().unwrap().push(Arc::new(notify));
    }

    /// Asks the guest to switch to a `width` x `height` mode, as a monitor hotplug would: the
    /// virtio-gpu device reports it as its preferred mode and signals a display change. The
    /// guest decides whether and when to follow (Linux's console does, a desktop asks its
    /// compositor). Clamped to what the device supports. The firmware's framebuffer keeps the
    /// size the machine was built with.
    pub fn request_size(&self, width: u32, height: u32) {
        let width = width.clamp(MIN_WIDTH, MAX_WIDTH);
        let height = height.clamp(MIN_HEIGHT, MAX_HEIGHT);
        self.shared.screen.lock().unwrap().requested_size = Some((width, height));
    }

    /// The pointer image and its change counter (0 until the guest first sets one).
    pub fn cursor(&self) -> (u64, Cursor) {
        let screen = self.shared.screen.lock().unwrap();
        (screen.cursor_generation, screen.cursor.clone())
    }

    /// Replaces the pointer image; nothing happens when it is the same.
    pub(crate) fn set_cursor(&self, cursor: Cursor) {
        let generation = {
            let mut screen = self.shared.screen.lock().unwrap();
            if screen.cursor == cursor {
                return;
            }
            screen.cursor = cursor;
            screen.cursor_generation += 1;
            self.shared.changed.notify_all();
            screen.generation
        };
        let callbacks = self.shared.callbacks.lock().unwrap().clone();
        for notify in callbacks {
            notify(generation);
        }
    }

    pub(crate) fn take_requested_size(&self) -> Option<(u32, u32)> {
        self.shared.screen.lock().unwrap().requested_size.take()
    }

    pub(crate) fn virtio_owns_screen(&self) -> bool {
        self.shared.virtio_owns_screen.load(Ordering::Acquire)
    }

    pub(crate) fn set_virtio_owns_screen(&self, owns: bool) {
        self.shared
            .virtio_owns_screen
            .store(owns, Ordering::Release);
    }

    /// Changes the screen: `paint` gets the pixels (resized and cleared to black first when the
    /// size changes) and their stride, and returns the rectangle it changed, if any. Bumps the
    /// generation and wakes the frontends only when something changed.
    pub(crate) fn publish(
        &self,
        width: u32,
        height: u32,
        paint: impl FnOnce(&mut [u8], usize) -> Option<Rect>,
    ) {
        let generation = {
            let mut screen = self.shared.screen.lock().unwrap();
            let resized = (screen.width, screen.height) != (width, height);
            if resized {
                screen.width = width;
                screen.height = height;
                screen.pixels.clear();
                screen
                    .pixels
                    .resize(width as usize * height as usize * BYTES_PER_PIXEL, 0);
            }
            let stride = screen.stride();
            let painted = paint(&mut screen.pixels, stride).map(|rect| rect.clipped(width, height));
            let rect = match painted {
                _ if resized => Rect {
                    x: 0,
                    y: 0,
                    width,
                    height,
                },
                Some(rect) if !rect.is_empty() => rect,
                _ => return,
            };
            screen.generation += 1;
            let generation = screen.generation;
            if resized {
                screen.resized_at = generation;
            }
            if screen.damage.len() == DAMAGE_HISTORY {
                screen.damage.pop_front();
            }
            screen.damage.push_back((generation, rect));
            self.shared.changed.notify_all();
            generation
        };
        let callbacks = self.shared.callbacks.lock().unwrap().clone();
        for notify in callbacks {
            notify(generation);
        }
    }
}

/// A pointer button of the tablet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
    /// The "back" thumb button.
    Side,
    /// The "forward" thumb button.
    Extra,
}

impl PointerButton {
    pub(crate) const ALL: [PointerButton; 5] = [
        PointerButton::Left,
        PointerButton::Right,
        PointerButton::Middle,
        PointerButton::Side,
        PointerButton::Extra,
    ];

    pub(crate) fn code(self) -> u16 {
        match self {
            PointerButton::Left => BTN_LEFT,
            PointerButton::Right => BTN_RIGHT,
            PointerButton::Middle => BTN_MIDDLE,
            PointerButton::Side => BTN_SIDE,
            PointerButton::Extra => BTN_EXTRA,
        }
    }
}

/// One Linux input event as virtio-input carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InputEvent {
    pub kind: u16,
    pub code: u16,
    pub value: u32,
}

/// Events that belong together, ending with `SYN_REPORT`: the guest sees all of them or none.
pub(crate) type InputBatch = Vec<InputEvent>;

/// Which virtio-input device a batch is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputDevice {
    Keyboard,
    Tablet,
}

type Notify = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct InputState {
    keyboard: VecDeque<InputBatch>,
    tablet: VecDeque<InputBatch>,
    /// Keys the guest was told are down, one bit per code, for `release_all`.
    keys_down: [u64; 4],
    buttons_down: u8,
    /// Whether the last queued tablet batch is only a move, which the next move may replace.
    tablet_tail_is_move: bool,
    /// Wakes whichever thread runs the devices, as `NetHandle` does for frames: without it a
    /// halted guest sees a keystroke only when its idle sleep ends.
    wake_devices: Option<Notify>,
}

struct InputShared {
    state: Mutex<InputState>,
    /// The screen pointer positions are relative to; none on a machine without a display.
    display: Option<DisplayHandle>,
    /// The area pointer positions are relative to without a display (`display_size`).
    pointer_area: (u32, u32),
}

impl Default for InputShared {
    fn default() -> Self {
        InputShared {
            state: Mutex::new(InputState::default()),
            display: None,
            pointer_area: (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        }
    }
}

/// The guest's keyboard and tablet, shared by clones. Obtained from `EmulatedMachine::input()`;
/// survives guest reboots. A machine may have them without a display: the keyboard works the
/// same, and pointer positions are then relative to `pointer_area`. Every method returns false
/// when the event was not queued (the queue is full because the guest is not reading, or the
/// code is out of range).
#[derive(Clone, Default)]
pub struct InputHandle {
    shared: Arc<InputShared>,
}

fn syn() -> InputEvent {
    InputEvent {
        kind: EV_SYN,
        code: SYN_REPORT,
        value: 0,
    }
}

impl InputHandle {
    /// Pointer positions follow `display`'s current size.
    pub(crate) fn for_display(display: DisplayHandle) -> InputHandle {
        let pointer_area = display.size();
        InputHandle {
            shared: Arc::new(InputShared {
                display: Some(display),
                pointer_area,
                ..InputShared::default()
            }),
        }
    }

    /// No screen: pointer positions are relative to a fixed `width` x `height` area.
    pub(crate) fn without_display(width: u32, height: u32) -> InputHandle {
        InputHandle {
            shared: Arc::new(InputShared {
                pointer_area: (width, height),
                ..InputShared::default()
            }),
        }
    }

    /// The size `pointer_move` positions are relative to: the screen's current size, or on a
    /// machine without a display the configured `display_size`.
    pub fn pointer_area(&self) -> (u32, u32) {
        self.shared
            .display
            .as_ref()
            .map_or(self.shared.pointer_area, DisplayHandle::size)
    }

    /// Presses (`pressed`) or releases a key, by Linux key code (`KEY_A` is 30; codes 1 to 255).
    /// Autorepeat is the guest's business: send one press and one release.
    pub fn key(&self, code: u16, pressed: bool) -> bool {
        if code == 0 || code > KEY_CODE_MAX {
            return false;
        }
        let event = InputEvent {
            kind: EV_KEY,
            code,
            value: if pressed { KEY_PRESSED } else { KEY_RELEASED },
        };
        let wake = {
            let mut state = self.shared.state.lock().unwrap();
            if state.keyboard.len() >= INPUT_QUEUE_BATCHES {
                return false;
            }
            state.keyboard.push_back(vec![event, syn()]);
            let (word, bit) = (code as usize / 64, code % 64);
            if pressed {
                state.keys_down[word] |= 1 << bit;
            } else {
                state.keys_down[word] &= !(1 << bit);
            }
            state.wake_devices.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
        true
    }

    /// `key` by USB HID usage (keyboard page 0x07), the numbering of USB keyboards and of SDL's
    /// scancodes. False for a usage with no Linux key.
    pub fn key_usb(&self, usage: u16, pressed: bool) -> bool {
        match linux_key_for_usb_usage(usage) {
            Some(code) => self.key(code, pressed),
            None => false,
        }
    }

    /// Releases every key and button the guest was told is down, for a frontend that loses the
    /// focus and with it the release events.
    pub fn release_all(&self) {
        let (keys, buttons) = {
            let state = self.shared.state.lock().unwrap();
            (state.keys_down, state.buttons_down)
        };
        for code in 1..=KEY_CODE_MAX {
            if keys[code as usize / 64] & (1 << (code % 64)) != 0 {
                self.key(code, false);
            }
        }
        for (index, button) in PointerButton::ALL.iter().enumerate() {
            if buttons & (1 << index) != 0 {
                self.pointer_button(*button, false);
            }
        }
    }

    /// Moves the pointer to (`x`, `y`) in pixels of `pointer_area` (the screen's current size),
    /// clamped to it. The tablet reports absolute positions, so the guest's pointer lands
    /// exactly where the host's is.
    pub fn pointer_move(&self, x: u32, y: u32) -> bool {
        let (width, height) = self.pointer_area();
        let scale = |position: u32, extent: u32| -> u32 {
            let last = extent.max(2) - 1;
            (position.min(last) as u64 * POINTER_AXIS_MAX as u64 / last as u64) as u32
        };
        let batch = vec![
            InputEvent {
                kind: EV_ABS,
                code: ABS_X,
                value: scale(x, width),
            },
            InputEvent {
                kind: EV_ABS,
                code: ABS_Y,
                value: scale(y, height),
            },
            syn(),
        ];
        self.queue_tablet(batch, true)
    }

    pub fn pointer_button(&self, button: PointerButton, pressed: bool) -> bool {
        let batch = vec![
            InputEvent {
                kind: EV_KEY,
                code: button.code(),
                value: if pressed { KEY_PRESSED } else { KEY_RELEASED },
            },
            syn(),
        ];
        let queued = self.queue_tablet(batch, false);
        if queued {
            let index = PointerButton::ALL
                .iter()
                .position(|b| *b == button)
                .unwrap();
            let mut state = self.shared.state.lock().unwrap();
            if pressed {
                state.buttons_down |= 1 << index;
            } else {
                state.buttons_down &= !(1 << index);
            }
        }
        queued
    }

    /// Turns the wheel by `clicks` notches, positive away from the user (scrolls up).
    pub fn scroll(&self, clicks: i32) -> bool {
        if clicks == 0 {
            return true;
        }
        let batch = vec![
            InputEvent {
                kind: EV_REL,
                code: REL_WHEEL,
                value: clicks as u32,
            },
            syn(),
        ];
        self.queue_tablet(batch, false)
    }

    /// A move replaces a move still waiting at the end of the queue: only the latest position
    /// matters, and a fast mouse would otherwise fill the queue.
    fn queue_tablet(&self, batch: InputBatch, is_move: bool) -> bool {
        let wake = {
            let mut state = self.shared.state.lock().unwrap();
            if is_move && state.tablet_tail_is_move {
                *state.tablet.back_mut().unwrap() = batch;
            } else {
                if state.tablet.len() >= INPUT_QUEUE_BATCHES {
                    return false;
                }
                state.tablet.push_back(batch);
            }
            state.tablet_tail_is_move = is_move;
            state.wake_devices.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
        true
    }

    /// The oldest batch waiting for `device`.
    pub(crate) fn peek(&self, device: InputDevice) -> Option<InputBatch> {
        let state = self.shared.state.lock().unwrap();
        match device {
            InputDevice::Keyboard => state.keyboard.front().cloned(),
            InputDevice::Tablet => state.tablet.front().cloned(),
        }
    }

    /// Drops the batch `peek` returned, once the guest has it.
    pub(crate) fn pop(&self, device: InputDevice) {
        let mut state = self.shared.state.lock().unwrap();
        match device {
            InputDevice::Keyboard => {
                state.keyboard.pop_front();
            }
            InputDevice::Tablet => {
                state.tablet.pop_front();
                if state.tablet.is_empty() {
                    state.tablet_tail_is_move = false;
                }
            }
        }
    }

    /// See `InputState::wake_devices`. A freestanding host has no thread to wake.
    #[cfg(not(freestanding))]
    pub(crate) fn set_device_waker(&self, wake: Option<Notify>) {
        self.shared.state.lock().unwrap().wake_devices = wake;
    }
}

/// The Linux key code (`KEY_*`) of a USB HID keyboard-page usage, as Linux's own `hid-input`
/// maps a USB keyboard (`hid_keyboard[]` in `drivers/hid/hid-input.c`). None for usages a PC
/// keyboard does not have.
pub fn linux_key_for_usb_usage(usage: u16) -> Option<u16> {
    /// Usages 0x04 ('a') to 0x67 (keypad '='), in order.
    const MAIN_BLOCK: [u16; 100] = [
        30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17,
        45, 21, 44, // a..z
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // 1..9, 0
        28, 1, 14, 15, 57, 12, 13, 26, 27, 43, 43, 39, 40, 41, 51, 52, 53,
        58, // enter..capslock
        59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88, // F1..F12
        99, 70, 119, 110, 102, 104, 111, 107, 109, 106, 105, 108, 103, // sysrq..up
        69, 98, 55, 74, 78, 96, 79, 80, 81, 75, 76, 77, 71, 72, 73, 82,
        83, // numlock..keypad .
        86, 127, 116, 117, // 102nd, compose, power, keypad =
    ];
    const FIRST_MAIN_USAGE: u16 = 0x04;
    /// F13 to F24 (usages 0x68 to 0x73) are `KEY_F13` (183) onwards.
    const F13_USAGE: u16 = 0x68;
    const F24_USAGE: u16 = 0x73;
    const KEY_F13: u16 = 183;
    /// The modifiers, usages 0xe0 to 0xe7: left ctrl, shift, alt, meta, then the right ones.
    const MODIFIERS: [u16; 8] = [29, 42, 56, 125, 97, 54, 100, 126];
    const FIRST_MODIFIER_USAGE: u16 = 0xe0;
    const LAST_MODIFIER_USAGE: u16 = 0xe7;
    let extra = |usage: u16| -> Option<u16> {
        Some(match usage {
            0x7f => 113, // mute
            0x80 => 115, // volume up
            0x81 => 114, // volume down
            0x85 => 121, // keypad comma
            0x87 => 89,  // ro
            0x88 => 93,  // katakana/hiragana
            0x89 => 124, // yen
            0x8a => 92,  // henkan
            0x8b => 94,  // muhenkan
            0x8c => 95,  // keypad JP comma
            0x90 => 122, // hangeul
            0x91 => 123, // hanja
            _ => return None,
        })
    };
    match usage {
        FIRST_MAIN_USAGE..F13_USAGE => Some(MAIN_BLOCK[(usage - FIRST_MAIN_USAGE) as usize]),
        F13_USAGE..=F24_USAGE => Some(KEY_F13 + (usage - F13_USAGE)),
        FIRST_MODIFIER_USAGE..=LAST_MODIFIER_USAGE => {
            Some(MODIFIERS[(usage - FIRST_MODIFIER_USAGE) as usize])
        }
        _ => extra(usage),
    }
}

#[cfg(test)]
mod tests;
