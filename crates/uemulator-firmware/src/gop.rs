//! The Graphics Output Protocol (UEFI 2.9 section 12.9) on the machine's boot framebuffer: one
//! mode, the framebuffer's size, `PixelBlueGreenRedReserved8BitPerColor`, the address the
//! library reserved in RAM (`docs/plans/24-display-input.md`). GRUB draws on it directly; the Linux
//! EFI stub hands it to the kernel as `screen_info` (efifb, simpledrm) until virtio-gpu takes the
//! screen over. The text console draws on the same pixels (`console.rs`).

use core::ffi::c_void;
use core::ptr;

use crate::handoff::{FRAMEBUFFER_BYTES_PER_PIXEL, Framebuffer};
use crate::memory;
use crate::protocols;
use crate::sync::Global;
use crate::uefi::memory_type::BOOT_SERVICES_DATA;
use crate::uefi::*;

const GRAPHICS_OUTPUT: Guid = Guid::new(
    0x9042a9de,
    0x23dc,
    0x4a38,
    [0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a],
);

/// `EFI_GRAPHICS_PIXEL_FORMAT`: byte 0 blue, 1 green, 2 red, 3 reserved.
const PIXEL_BLUE_GREEN_RED_RESERVED_8_BIT: u32 = 1;
const ONLY_MODE: u32 = 0;
const MODE_COUNT: u32 = 1;

/// `EFI_GRAPHICS_OUTPUT_BLT_OPERATION`.
const BLT_VIDEO_FILL: u32 = 0;
const BLT_VIDEO_TO_BUFFER: u32 = 1;
const BLT_BUFFER_TO_VIDEO: u32 = 2;
const BLT_VIDEO_TO_VIDEO: u32 = 3;

#[repr(C)]
struct PixelBitmask {
    red: u32,
    green: u32,
    blue: u32,
    reserved: u32,
}

#[repr(C)]
struct ModeInformation {
    version: u32,
    horizontal_resolution: u32,
    vertical_resolution: u32,
    pixel_format: u32,
    pixel_information: PixelBitmask,
    pixels_per_scan_line: u32,
}

#[repr(C)]
struct Mode {
    max_mode: u32,
    mode: u32,
    info: *mut ModeInformation,
    size_of_info: usize,
    frame_buffer_base: u64,
    frame_buffer_size: usize,
}

#[repr(C)]
struct GraphicsOutput {
    query_mode: extern "efiapi" fn(
        *mut GraphicsOutput,
        u32,
        *mut usize,
        *mut *mut ModeInformation,
    ) -> Status,
    set_mode: extern "efiapi" fn(*mut GraphicsOutput, u32) -> Status,
    blt: extern "efiapi" fn(
        *mut GraphicsOutput,
        *mut u32,
        u32,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) -> Status,
    mode: *mut Mode,
}

static INFO: Global<ModeInformation> = Global::new(ModeInformation {
    version: 0,
    horizontal_resolution: 0,
    vertical_resolution: 0,
    pixel_format: PIXEL_BLUE_GREEN_RED_RESERVED_8_BIT,
    pixel_information: PixelBitmask {
        red: 0,
        green: 0,
        blue: 0,
        reserved: 0,
    },
    pixels_per_scan_line: 0,
});

static MODE: Global<Mode> = Global::new(Mode {
    max_mode: MODE_COUNT,
    mode: ONLY_MODE,
    info: ptr::null_mut(),
    size_of_info: size_of::<ModeInformation>(),
    frame_buffer_base: 0,
    frame_buffer_size: 0,
});

static PROTOCOL: Global<GraphicsOutput> = Global::new(GraphicsOutput {
    query_mode,
    set_mode,
    blt,
    mode: ptr::null_mut(),
});

static SCREEN: Global<Option<Framebuffer>> = Global::new(None);

/// The framebuffer's pixels as one slice of `pixels_per_line * height` words.
pub fn pixels(framebuffer: &Framebuffer) -> &'static mut [u32] {
    let count = framebuffer.pixels_per_line as usize * framebuffer.height as usize;
    // The library reserved this range for the framebuffer (the memory map says so), the boot
    // page tables map it, and the handoff checked that `count` words fit its size.
    unsafe { core::slice::from_raw_parts_mut(framebuffer.base as *mut u32, count) }
}

extern "efiapi" fn query_mode(
    _: *mut GraphicsOutput,
    mode: u32,
    size: *mut usize,
    info: *mut *mut ModeInformation,
) -> Status {
    if mode != ONLY_MODE {
        return INVALID_PARAMETER;
    }
    if size.is_null() || info.is_null() {
        return INVALID_PARAMETER;
    }
    // The caller frees the copy with FreePool.
    let copy = memory::pool_allocate(BOOT_SERVICES_DATA, size_of::<ModeInformation>());
    if copy.is_null() {
        return OUT_OF_RESOURCES;
    }
    unsafe {
        ptr::copy_nonoverlapping(INFO.get() as *const ModeInformation, copy.cast(), 1);
        size.write(size_of::<ModeInformation>());
        info.write(copy.cast());
    }
    SUCCESS
}

/// Setting the one mode clears the screen to black, as the spec asks of a mode set.
extern "efiapi" fn set_mode(_: *mut GraphicsOutput, mode: u32) -> Status {
    if mode != ONLY_MODE {
        return UNSUPPORTED;
    }
    if let Some(framebuffer) = SCREEN.get() {
        pixels(framebuffer).fill(0);
    }
    SUCCESS
}

/// Whether a `width` x `height` rectangle at (`x`, `y`) lies on the screen.
fn on_screen(framebuffer: &Framebuffer, x: usize, y: usize, width: usize, height: usize) -> bool {
    x.checked_add(width)
        .is_some_and(|right| right <= framebuffer.width as usize)
        && y.checked_add(height)
            .is_some_and(|bottom| bottom <= framebuffer.height as usize)
}

#[allow(clippy::too_many_arguments)] // the protocol's signature
extern "efiapi" fn blt(
    _: *mut GraphicsOutput,
    buffer: *mut u32,
    operation: u32,
    source_x: usize,
    source_y: usize,
    destination_x: usize,
    destination_y: usize,
    width: usize,
    height: usize,
    delta: usize,
) -> Status {
    let Some(framebuffer) = SCREEN.get().as_ref() else {
        return UNSUPPORTED;
    };
    if width == 0 || height == 0 {
        return INVALID_PARAMETER;
    }
    let screen = pixels(framebuffer);
    let line = framebuffer.pixels_per_line as usize;
    // Bytes between buffer rows; 0 means rows of exactly `width` pixels.
    let buffer_line = if delta == 0 {
        width
    } else {
        delta / FRAMEBUFFER_BYTES_PER_PIXEL as usize
    };
    let buffer_at = |x: usize, y: usize| unsafe { buffer.add(y * buffer_line + x) };
    match operation {
        BLT_VIDEO_FILL => {
            if buffer.is_null()
                || !on_screen(framebuffer, destination_x, destination_y, width, height)
            {
                return INVALID_PARAMETER;
            }
            let colour = unsafe { buffer.read_unaligned() };
            for row in destination_y..destination_y + height {
                let start = row * line + destination_x;
                screen[start..start + width].fill(colour);
            }
        }
        BLT_VIDEO_TO_BUFFER => {
            if buffer.is_null() || !on_screen(framebuffer, source_x, source_y, width, height) {
                return INVALID_PARAMETER;
            }
            for row in 0..height {
                let start = (source_y + row) * line + source_x;
                let target = buffer_at(destination_x, destination_y + row);
                unsafe { ptr::copy(screen[start..].as_ptr(), target, width) };
            }
        }
        BLT_BUFFER_TO_VIDEO => {
            if buffer.is_null()
                || !on_screen(framebuffer, destination_x, destination_y, width, height)
            {
                return INVALID_PARAMETER;
            }
            for row in 0..height {
                let start = (destination_y + row) * line + destination_x;
                let source = buffer_at(source_x, source_y + row);
                unsafe { ptr::copy(source, screen[start..].as_mut_ptr(), width) };
            }
        }
        BLT_VIDEO_TO_VIDEO => {
            if !on_screen(framebuffer, source_x, source_y, width, height)
                || !on_screen(framebuffer, destination_x, destination_y, width, height)
            {
                return INVALID_PARAMETER;
            }
            // Rows in the order that never reads a row already overwritten.
            let copy_row = |screen: &mut [u32], row: usize| {
                let source = (source_y + row) * line + source_x;
                let target = (destination_y + row) * line + destination_x;
                screen.copy_within(source..source + width, target);
            };
            if destination_y <= source_y {
                (0..height).for_each(|row| copy_row(screen, row));
            } else {
                (0..height).rev().for_each(|row| copy_row(screen, row));
            }
        }
        _ => return INVALID_PARAMETER,
    }
    SUCCESS
}

/// Publishes the protocol on a handle of its own when the machine has a screen.
pub fn init(framebuffer: Option<Framebuffer>) {
    let Some(framebuffer) = framebuffer else {
        return;
    };
    *SCREEN.get() = Some(framebuffer);
    let info = INFO.get();
    info.horizontal_resolution = framebuffer.width;
    info.vertical_resolution = framebuffer.height;
    info.pixels_per_scan_line = framebuffer.pixels_per_line;
    let mode = MODE.get();
    mode.info = info;
    mode.frame_buffer_base = framebuffer.base;
    mode.frame_buffer_size = framebuffer.size as usize;
    PROTOCOL.get().mode = mode;
    let interface: *mut c_void = (PROTOCOL.get() as *mut GraphicsOutput).cast();
    if protocols::install(ptr::null_mut(), GRAPHICS_OUTPUT, interface).is_err() {
        crate::serial::println!("firmware: could not publish the graphics output protocol");
    }
}
