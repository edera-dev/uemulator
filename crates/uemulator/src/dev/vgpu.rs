//! virtio-gpu, 2D only (virtio 1.2 section 5.7): the guest creates resources (host-side images),
//! backs them with its own pages, copies pages into them (`TRANSFER_TO_HOST_2D`), points the one
//! scanout at a resource and says which part changed (`RESOURCE_FLUSH`), which is when the
//! `DisplayHandle` gets the pixels. No 3D, no EDID, no blobs: Linux's `virtio_gpu` DRM driver
//! runs its fbdev console and KMS desktops on what is left. Behaviour follows QEMU's
//! `hw/display/virtio-gpu.c`.
//!
//! Everything the guest sends is bounded before the host allocates or copies: resource sizes
//! against `MAX_RESOURCE_BYTES` in total, rectangles against their resource, backing entries
//! against RAM. A bad request gets an error response, never a panic.

use alloc::collections::BTreeMap;

use super::guestmem::GuestMem;
use super::virtio::{Chain, DEVICE_TYPE_GPU, Queue, VirtioDev};
use crate::display::{
    BYTES_PER_PIXEL, Cursor, CursorImage, DisplayHandle, MAX_CURSOR_SIDE, MAX_HEIGHT, MAX_WIDTH,
    Rect,
};
#[cfg(freestanding)]
use crate::prelude::*;

const CONTROL_QUEUE: usize = 0;
const CURSOR_QUEUE: usize = 1;
const QUEUE_COUNT: usize = 2;

/// Command and response types (`enum virtio_gpu_ctrl_type`).
const CMD_GET_DISPLAY_INFO: u32 = 0x0100;
const CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
const CMD_RESOURCE_UNREF: u32 = 0x0102;
const CMD_SET_SCANOUT: u32 = 0x0103;
const CMD_RESOURCE_FLUSH: u32 = 0x0104;
const CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
const CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
const CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
/// Cursor queue commands. Both carry `struct virtio_gpu_update_cursor`; `MOVE_CURSOR` only
/// moves the pointer, which the tablet already put under the host's, so it is ignored.
const CMD_UPDATE_CURSOR: u32 = 0x0300;
/// `struct virtio_gpu_update_cursor` after its header: pos (scanout_id, x, y, padding), then
/// resource_id, hot_x, hot_y, padding.
const CURSOR_RESOURCE_ID: usize = 16;
const CURSOR_HOT_X: usize = 20;
const CURSOR_HOT_Y: usize = 24;
const CURSOR_REQUEST_BYTES: usize = HEADER_BYTES + 32;
const RESP_OK_NODATA: u32 = 0x1100;
const RESP_OK_DISPLAY_INFO: u32 = 0x1101;
const RESP_ERR_UNSPEC: u32 = 0x1200;
const RESP_ERR_OUT_OF_MEMORY: u32 = 0x1201;
const RESP_ERR_INVALID_SCANOUT_ID: u32 = 0x1202;
const RESP_ERR_INVALID_RESOURCE_ID: u32 = 0x1203;
const RESP_ERR_INVALID_PARAMETER: u32 = 0x1205;

/// `struct virtio_gpu_ctrl_hdr`: type u32, flags u32, fence_id u64, ctx_id u32, ring_idx u8,
/// padding. Every request starts with one and every response is one plus a payload.
const HEADER_BYTES: usize = 24;
const HEADER_FLAGS: usize = 4;
const HEADER_FENCE_ID: usize = 8;
const HEADER_CONTEXT_ID: usize = 16;
const HEADER_RING_INDEX: usize = 20;
/// The driver waits for the response of a fenced command before it reuses what the command
/// touched; responses echo the fence (and the ring it was meant for), and since every command
/// completes before its response is written, that is all fencing needs here.
const FLAG_FENCE: u32 = 1 << 0;
const FLAG_INFO_RING_INDEX: u32 = 1 << 1;

/// `struct virtio_gpu_rect` (x, y, width, height), right after the header in the requests that
/// carry one.
const RECT_BYTES: usize = 16;

/// `struct virtio_gpu_resp_display_info`: one (rect, enabled u32, flags u32) per possible scanout.
const MAX_SCANOUTS: usize = 16;
const DISPLAY_ONE_BYTES: usize = RECT_BYTES + 8;
/// One monitor. Linux sizes its connector list by `num_scanouts`.
const SCANOUT_COUNT: u32 = 1;

/// `struct virtio_gpu_mem_entry`: addr u64, length u32, padding u32.
const MEM_ENTRY_BYTES: usize = 16;
/// Most backing entries one resource may have. Linux sends one per contiguous run of pages, so
/// a resource the size of the largest screen needs at most a thousand or so; the cap only bounds
/// what a hostile guest can make the host read and keep.
const MAX_BACKING_ENTRIES: u32 = 1 << 16;

/// Sizes the guest may create. 16384 is what Linux's driver itself advertises as the mode
/// limit; the byte budget is what actually bounds host memory: enough for the largest screen
/// (`MAX_WIDTH` x `MAX_HEIGHT`) several times over (fbdev, double-buffered KMS, cursors).
const MAX_RESOURCE_DIMENSION: u32 = 16384;
const MAX_RESOURCE_BYTES: usize = 256 << 20;

/// Device configuration (`struct virtio_gpu_config`): events_read, events_clear, num_scanouts,
/// num_capsets, each u32.
const CONFIG_EVENTS_READ: usize = 0;
const CONFIG_EVENTS_CLEAR: usize = 4;
const CONFIG_NUM_SCANOUTS: usize = 8;
const CONFIG_NUM_CAPSETS: usize = 12;
const CONFIG_WORD_BYTES: usize = 4;
/// `VIRTIO_GPU_EVENT_DISPLAY`: the display configuration changed; the driver re-reads it.
const EVENT_DISPLAY: u32 = 1 << 0;

/// Resource formats (`enum virtio_gpu_formats`). The names give the byte order in memory, so
/// `B8G8R8X8` is what DRM calls XRGB8888 on a little-endian guest and needs no conversion.
const FORMAT_B8G8R8A8: u32 = 1;
const FORMAT_B8G8R8X8: u32 = 2;
const FORMAT_A8R8G8B8: u32 = 3;
const FORMAT_X8R8G8B8: u32 = 4;
const FORMAT_R8G8B8A8: u32 = 67;
const FORMAT_X8B8G8R8: u32 = 68;
const FORMAT_A8B8G8R8: u32 = 121;
const FORMAT_R8G8B8X8: u32 = 134;

/// Where a pointer image's alpha sits in a pixel of a resource format: the byte its colours leave,
/// in the X formats too. Linux creates every dumb buffer, the cursor's included, as XRGB8888
/// (`B8G8R8X8`) and fills it with ARGB pixels, and QEMU copies cursor words as ARGB whatever the
/// format says. Taking the X byte as opaque turns the transparent pixels around the arrow into a
/// black square. Screens are different: there the fourth byte means nothing (`present`).
fn cursor_alpha_offset(format: u32) -> Option<usize> {
    let colours = channel_offsets(format)?;
    (0..BYTES_PER_PIXEL).find(|offset| !colours.contains(offset))
}

/// Where blue, green and red sit in a pixel of a resource format, as byte indices. The screen
/// is always blue, green, red, unused.
fn channel_offsets(format: u32) -> Option<[usize; 3]> {
    match format {
        FORMAT_B8G8R8A8 | FORMAT_B8G8R8X8 => Some([0, 1, 2]),
        FORMAT_A8R8G8B8 | FORMAT_X8R8G8B8 => Some([3, 2, 1]),
        FORMAT_R8G8B8A8 | FORMAT_R8G8B8X8 => Some([2, 1, 0]),
        FORMAT_A8B8G8R8 | FORMAT_X8B8G8R8 => Some([1, 2, 3]),
        _ => None,
    }
}

/// A guest page range a resource's pixels are copied from.
struct BackingEntry {
    address: u64,
    length: u64,
}

struct Resource {
    format: u32,
    width: u32,
    height: u32,
    /// `width * height` pixels in `format`, rows packed.
    pixels: Vec<u8>,
    /// The guest memory behind it, in order; their concatenation is the image the guest drew.
    backing: Vec<BackingEntry>,
}

impl Resource {
    fn stride(&self) -> usize {
        self.width as usize * BYTES_PER_PIXEL
    }

    fn contains(&self, rect: &Rect) -> bool {
        rect.x
            .checked_add(rect.width)
            .is_some_and(|right| right <= self.width)
            && rect
                .y
                .checked_add(rect.height)
                .is_some_and(|bottom| bottom <= self.height)
    }

    /// Fills `out` from byte `offset` of the backing, as if the entries were one buffer. Bytes
    /// past the backing's end, or in an entry outside RAM, read as zero.
    fn read_backing(&self, mem: &GuestMem, mut offset: u64, out: &mut [u8]) {
        out.fill(0);
        let mut done = 0;
        for entry in &self.backing {
            if done == out.len() {
                break;
            }
            if offset >= entry.length {
                offset -= entry.length;
                continue;
            }
            let count = ((entry.length - offset) as usize).min(out.len() - done);
            mem.read(entry.address + offset, &mut out[done..done + count]);
            done += count;
            offset = 0;
        }
    }
}

/// What scanout 0 shows: part of a resource, the size of the mode.
#[derive(Clone, Copy)]
struct Scanout {
    resource: u32,
    rect: Rect,
}

pub(crate) struct VirtioGpu {
    display: DisplayHandle,
    /// The mode `GET_DISPLAY_INFO` offers; a frontend may change it (`request_size`).
    preferred: (u32, u32),
    resources: BTreeMap<u32, Resource>,
    resource_bytes: usize,
    scanout: Option<Scanout>,
    events_read: u32,
    /// A display event is waiting to be signalled as a configuration change.
    event_pending: bool,
}

/// A request header and the parts of it a response echoes.
#[derive(Clone, Copy)]
struct Header {
    command: u32,
    flags: u32,
    fence_id: u64,
    context_id: u32,
    ring_index: u8,
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}

fn rect_at(bytes: &[u8], at: usize) -> Rect {
    Rect {
        x: u32_at(bytes, at),
        y: u32_at(bytes, at + 4),
        width: u32_at(bytes, at + 8),
        height: u32_at(bytes, at + 12),
    }
}

impl Header {
    fn parse(request: &[u8]) -> Header {
        Header {
            command: u32_at(request, 0),
            flags: u32_at(request, HEADER_FLAGS),
            fence_id: u64_at(request, HEADER_FENCE_ID),
            context_id: u32_at(request, HEADER_CONTEXT_ID),
            ring_index: request.get(HEADER_RING_INDEX).copied().unwrap_or(0),
        }
    }

    /// A response header of `kind` that echoes the request's fence.
    fn response(&self, kind: u32) -> Vec<u8> {
        let mut response = vec![0u8; HEADER_BYTES];
        response[..4].copy_from_slice(&kind.to_le_bytes());
        let echoed = self.flags & (FLAG_FENCE | FLAG_INFO_RING_INDEX);
        response[HEADER_FLAGS..HEADER_FLAGS + 4].copy_from_slice(&echoed.to_le_bytes());
        if self.flags & FLAG_FENCE != 0 {
            response[HEADER_FENCE_ID..HEADER_FENCE_ID + 8]
                .copy_from_slice(&self.fence_id.to_le_bytes());
            response[HEADER_CONTEXT_ID..HEADER_CONTEXT_ID + 4]
                .copy_from_slice(&self.context_id.to_le_bytes());
            response[HEADER_RING_INDEX] = self.ring_index;
        }
        response
    }
}

/// Bytes of a request a command can need: the header, the largest fixed part (a transfer), and
/// for `ATTACH_BACKING` its entries.
fn request_limit(nr_entries: u32) -> usize {
    const LARGEST_FIXED_REQUEST: usize = HEADER_BYTES + 32;
    LARGEST_FIXED_REQUEST + nr_entries.min(MAX_BACKING_ENTRIES) as usize * MEM_ENTRY_BYTES
}

impl VirtioGpu {
    /// Built for every boot around the embedder's handle, which outlives the boot. A guest that
    /// restarts without resetting the device (Linux's emergency restart under `panic=-1` skips
    /// device shutdown) would otherwise leave its pointer image on the firmware's screen and
    /// keep the boot framebuffer from being scanned, freezing the firmware's console.
    pub fn new(display: DisplayHandle) -> Self {
        display.set_virtio_owns_screen(false);
        display.set_cursor(Cursor::Host);
        let preferred = display.size();
        VirtioGpu {
            display,
            preferred,
            resources: BTreeMap::new(),
            resource_bytes: 0,
            scanout: None,
            events_read: 0,
            event_pending: false,
        }
    }

    /// Runs one control command and returns the response bytes.
    fn execute(&mut self, chain: &Chain, mem: &GuestMem) -> Vec<u8> {
        let mut fixed = [0u8; HEADER_BYTES + 32];
        let length = chain.read_from(mem, 0, &mut fixed);
        let header = Header::parse(&fixed[..length]);
        if length < HEADER_BYTES {
            return header.response(RESP_ERR_UNSPEC);
        }
        let body = &fixed[HEADER_BYTES..length];
        let status = match header.command {
            CMD_GET_DISPLAY_INFO => return self.display_info(&header),
            CMD_RESOURCE_CREATE_2D => self.create_2d(body),
            CMD_RESOURCE_UNREF => self.unref(u32_at(body, 0)),
            CMD_SET_SCANOUT => self.set_scanout(body),
            CMD_RESOURCE_FLUSH => self.flush(rect_at(body, 0), u32_at(body, RECT_BYTES)),
            CMD_TRANSFER_TO_HOST_2D => self.transfer(body, mem),
            CMD_RESOURCE_ATTACH_BACKING => {
                let nr_entries = u32_at(body, 4);
                let request = chain.read_all(mem, request_limit(nr_entries));
                self.attach_backing(&request[HEADER_BYTES.min(request.len())..], mem)
            }
            CMD_RESOURCE_DETACH_BACKING => self.detach_backing(u32_at(body, 0)),
            // 3D, capsets, EDID and blobs are not offered; a driver that asks anyway is told.
            _ => RESP_ERR_UNSPEC,
        };
        header.response(status)
    }

    fn display_info(&self, header: &Header) -> Vec<u8> {
        let mut response = header.response(RESP_OK_DISPLAY_INFO);
        let mut modes = vec![0u8; MAX_SCANOUTS * DISPLAY_ONE_BYTES];
        let (width, height) = self.preferred;
        // Scanout 0 at the origin, enabled; the rest stay zero (disabled).
        modes[8..12].copy_from_slice(&width.to_le_bytes());
        modes[12..16].copy_from_slice(&height.to_le_bytes());
        modes[RECT_BYTES..RECT_BYTES + 4].copy_from_slice(&1u32.to_le_bytes());
        response.extend_from_slice(&modes);
        response
    }

    /// `resource_id`, `format`, `width`, `height`.
    fn create_2d(&mut self, body: &[u8]) -> u32 {
        let (id, format, width, height) = (
            u32_at(body, 0),
            u32_at(body, 4),
            u32_at(body, 8),
            u32_at(body, 12),
        );
        if id == 0 || self.resources.contains_key(&id) {
            return RESP_ERR_INVALID_RESOURCE_ID;
        }
        if channel_offsets(format).is_none()
            || width == 0
            || height == 0
            || width > MAX_RESOURCE_DIMENSION
            || height > MAX_RESOURCE_DIMENSION
        {
            return RESP_ERR_INVALID_PARAMETER;
        }
        let bytes = width as usize * height as usize * BYTES_PER_PIXEL;
        if self.resource_bytes + bytes > MAX_RESOURCE_BYTES {
            return RESP_ERR_OUT_OF_MEMORY;
        }
        let mut pixels = Vec::new();
        if pixels.try_reserve_exact(bytes).is_err() {
            return RESP_ERR_OUT_OF_MEMORY;
        }
        pixels.resize(bytes, 0);
        self.resource_bytes += bytes;
        self.resources.insert(
            id,
            Resource {
                format,
                width,
                height,
                pixels,
                backing: Vec::new(),
            },
        );
        RESP_OK_NODATA
    }

    fn unref(&mut self, id: u32) -> u32 {
        let Some(resource) = self.resources.remove(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        self.resource_bytes -= resource.pixels.len();
        // The scanout cannot show a resource that is gone; QEMU blanks it too.
        if self.scanout.is_some_and(|scanout| scanout.resource == id) {
            self.scanout = None;
            self.blank();
        }
        RESP_OK_NODATA
    }

    /// `rect`, `scanout_id`, `resource_id`; resource 0 turns the scanout off.
    fn set_scanout(&mut self, body: &[u8]) -> u32 {
        let rect = rect_at(body, 0);
        let (scanout_id, id) = (u32_at(body, RECT_BYTES), u32_at(body, RECT_BYTES + 4));
        if scanout_id >= SCANOUT_COUNT {
            return RESP_ERR_INVALID_SCANOUT_ID;
        }
        // From the first SET_SCANOUT on, the screen is the driver's, even when it turns the
        // scanout off (a blank screen, not the firmware's stale framebuffer); a device reset
        // gives it back.
        self.display.set_virtio_owns_screen(true);
        if id == 0 {
            self.scanout = None;
            self.blank();
            return RESP_OK_NODATA;
        }
        let Some(resource) = self.resources.get(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        if rect.is_empty()
            || !resource.contains(&rect)
            || rect.width > MAX_WIDTH
            || rect.height > MAX_HEIGHT
        {
            return RESP_ERR_INVALID_PARAMETER;
        }
        self.scanout = Some(Scanout { resource: id, rect });
        self.present(Rect {
            width: resource.width,
            height: resource.height,
            ..Rect::default()
        });
        RESP_OK_NODATA
    }

    fn flush(&mut self, rect: Rect, id: u32) -> u32 {
        let Some(resource) = self.resources.get(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        if !resource.contains(&rect) {
            return RESP_ERR_INVALID_PARAMETER;
        }
        if self.scanout.is_some_and(|scanout| scanout.resource == id) {
            self.present(rect);
        }
        RESP_OK_NODATA
    }

    /// Copies the part of `changed` (in resource coordinates) that the scanout shows to the
    /// screen, converting to the screen's pixel format.
    fn present(&self, changed: Rect) {
        let Some(scanout) = self.scanout else {
            return;
        };
        let Some(resource) = self.resources.get(&scanout.resource) else {
            return;
        };
        let Some([blue, green, red]) = channel_offsets(resource.format) else {
            return;
        };
        let shown = scanout.rect;
        let left = changed.x.max(shown.x);
        let top = changed.y.max(shown.y);
        let right = (changed.x + changed.width).min(shown.x + shown.width);
        let bottom = (changed.y + changed.height).min(shown.y + shown.height);
        if left >= right || top >= bottom {
            return;
        }
        let damage = Rect {
            x: left - shown.x,
            y: top - shown.y,
            width: right - left,
            height: bottom - top,
        };
        let source_stride = resource.stride();
        self.display
            .publish(shown.width, shown.height, |screen, stride| {
                for row in 0..damage.height {
                    let source_row = (top + row) as usize * source_stride;
                    let target_row = (damage.y + row) as usize * stride;
                    for column in 0..damage.width {
                        let source = source_row + (left + column) as usize * BYTES_PER_PIXEL;
                        let target = target_row + (damage.x + column) as usize * BYTES_PER_PIXEL;
                        let pixel = &resource.pixels[source..source + BYTES_PER_PIXEL];
                        screen[target] = pixel[blue];
                        screen[target + 1] = pixel[green];
                        screen[target + 2] = pixel[red];
                        screen[target + 3] = 0;
                    }
                }
                Some(damage)
            });
    }

    /// `UPDATE_CURSOR` hands the frontend the image of the resource it names (resource 0 hides
    /// the pointer). The image is copied now, as QEMU does: the guest may reuse the resource
    /// for the next shape right away. A resource too big to be a pointer, or in a format with
    /// no colour channels known, leaves the pointer as it was. Alpha is the byte the colours
    /// leave, in the X formats too (`cursor_alpha_offset`).
    fn cursor_command(&mut self, request: &[u8]) {
        if request.len() < CURSOR_REQUEST_BYTES || u32_at(request, 0) != CMD_UPDATE_CURSOR {
            return;
        }
        let body = &request[HEADER_BYTES..];
        let id = u32_at(body, CURSOR_RESOURCE_ID);
        if id == 0 {
            self.display.set_cursor(Cursor::Hidden);
            return;
        }
        let Some(resource) = self.resources.get(&id) else {
            return;
        };
        let Some([blue, green, red]) = channel_offsets(resource.format) else {
            return;
        };
        if resource.width > MAX_CURSOR_SIDE || resource.height > MAX_CURSOR_SIDE {
            return;
        }
        let Some(alpha) = cursor_alpha_offset(resource.format) else {
            return;
        };
        let mut pixels = Vec::with_capacity(resource.pixels.len());
        for pixel in resource.pixels.as_chunks::<BYTES_PER_PIXEL>().0 {
            pixels.extend_from_slice(&[pixel[blue], pixel[green], pixel[red], pixel[alpha]]);
        }
        self.display.set_cursor(Cursor::Image(CursorImage {
            width: resource.width,
            height: resource.height,
            hot_x: u32_at(body, CURSOR_HOT_X).min(resource.width.saturating_sub(1)),
            hot_y: u32_at(body, CURSOR_HOT_Y).min(resource.height.saturating_sub(1)),
            pixels,
        }));
    }

    /// A black screen of the current size.
    fn blank(&self) {
        let (width, height) = self.display.size();
        self.display.publish(width, height, |screen, _| {
            screen.fill(0);
            Some(Rect {
                width,
                height,
                ..Rect::default()
            })
        });
    }

    /// `rect`, `offset` u64, `resource_id`. Each row of `rect` is read from the backing at
    /// `offset + row * stride`, where the stride is the resource's: Linux passes the offset of
    /// the rectangle's first pixel in its buffer, which has the resource's layout.
    fn transfer(&mut self, body: &[u8], mem: &GuestMem) -> u32 {
        let rect = rect_at(body, 0);
        let offset = u64_at(body, RECT_BYTES);
        let id = u32_at(body, RECT_BYTES + 8);
        let Some(resource) = self.resources.get_mut(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        if resource.backing.is_empty() {
            return RESP_ERR_UNSPEC;
        }
        if !resource.contains(&rect) {
            return RESP_ERR_INVALID_PARAMETER;
        }
        let stride = resource.stride();
        let row_bytes = rect.width as usize * BYTES_PER_PIXEL;
        let mut row = vec![0u8; row_bytes];
        for line in 0..rect.height as usize {
            let Some(source) = offset.checked_add((line * stride) as u64) else {
                return RESP_ERR_INVALID_PARAMETER;
            };
            resource.read_backing(mem, source, &mut row);
            let target = (rect.y as usize + line) * stride + rect.x as usize * BYTES_PER_PIXEL;
            resource.pixels[target..target + row_bytes].copy_from_slice(&row);
        }
        RESP_OK_NODATA
    }

    /// `resource_id`, `nr_entries`, then that many `virtio_gpu_mem_entry`s.
    fn attach_backing(&mut self, body: &[u8], mem: &GuestMem) -> u32 {
        let (id, count) = (u32_at(body, 0), u32_at(body, 4));
        let Some(resource) = self.resources.get_mut(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        const ENTRIES_START: usize = 8;
        if count == 0
            || count > MAX_BACKING_ENTRIES
            || body.len() < ENTRIES_START + count as usize * MEM_ENTRY_BYTES
        {
            return RESP_ERR_UNSPEC;
        }
        let mut backing = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let at = ENTRIES_START + index * MEM_ENTRY_BYTES;
            let entry = BackingEntry {
                address: u64_at(body, at),
                length: u32_at(body, at + 8) as u64,
            };
            // The pages must be the guest's memory (through the IOMMU when there is one), as
            // QEMU checks when it maps them.
            if !mem.contains(entry.address, entry.length) {
                return RESP_ERR_UNSPEC;
            }
            backing.push(entry);
        }
        resource.backing = backing;
        RESP_OK_NODATA
    }

    fn detach_backing(&mut self, id: u32) -> u32 {
        match self.resources.get_mut(&id) {
            Some(resource) => {
                resource.backing.clear();
                RESP_OK_NODATA
            }
            None => RESP_ERR_INVALID_RESOURCE_ID,
        }
    }
}

impl VirtioDev for VirtioGpu {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_GPU
    }

    /// Neither VIRGL nor EDID nor blobs: the driver then sticks to the 2D commands.
    fn features(&self) -> u64 {
        0
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, offset: usize) -> u8 {
        let word = match offset & !(CONFIG_WORD_BYTES - 1) {
            CONFIG_EVENTS_READ => self.events_read,
            CONFIG_EVENTS_CLEAR => 0,
            CONFIG_NUM_SCANOUTS => SCANOUT_COUNT,
            CONFIG_NUM_CAPSETS => 0,
            _ => 0,
        };
        (word >> (8 * (offset % CONFIG_WORD_BYTES))) as u8
    }

    /// Writing a bit of `events_clear` clears it in `events_read`.
    fn config_write(&mut self, offset: usize, value: u8) {
        if offset & !(CONFIG_WORD_BYTES - 1) == CONFIG_EVENTS_CLEAR {
            let bits = (value as u32) << (8 * (offset % CONFIG_WORD_BYTES));
            self.events_read &= !bits;
        }
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut completed = false;
        while let Some(chain) = queues[queue].pop(mem) {
            let written = if queue == CONTROL_QUEUE {
                let response = self.execute(&chain, mem);
                chain.write_at(mem, 0, &response)
            } else {
                // Cursor commands carry no response.
                debug_assert_eq!(queue, CURSOR_QUEUE);
                let mut request = [0u8; CURSOR_REQUEST_BYTES];
                let length = chain.read_from(mem, 0, &mut request);
                self.cursor_command(&request[..length]);
                0
            };
            queues[queue].push_used(mem, chain.head, written as u32);
            completed = true;
        }
        completed
    }

    /// Picks up a size a frontend asked for and tells the driver.
    fn poll(&mut self, _queues: &mut [Queue], _mem: &mut GuestMem) -> bool {
        if let Some(size) = self.display.take_requested_size()
            && size != self.preferred
        {
            self.preferred = size;
            self.events_read |= EVENT_DISPLAY;
            self.event_pending = true;
        }
        false
    }

    fn config_event(&mut self) -> bool {
        core::mem::take(&mut self.event_pending)
    }

    /// The driver let go: resources are gone, and the boot framebuffer is the screen again
    /// (a guest that reboots through the firmware draws there next).
    fn reset(&mut self) {
        self.resources.clear();
        self.resource_bytes = 0;
        self.scanout = None;
        self.events_read = 0;
        self.event_pending = false;
        self.display.set_virtio_owns_screen(false);
        self.display.set_cursor(Cursor::Host);
    }
}

#[cfg(test)]
mod tests;
