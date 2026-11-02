use super::*;
use crate::dev::guestmem::test_mem;
use crate::dev::virtio::testutil::*;
use crate::dev::virtio::{DESC_F_NEXT, DESC_F_WRITE};

const QUEUE_SIZE: u16 = 8;
const CONTROL_DESC: u64 = 0x1000;
const CONTROL_AVAIL: u64 = 0x2000;
const CONTROL_USED: u64 = 0x3000;
const CURSOR_DESC: u64 = 0x4000;
const CURSOR_AVAIL: u64 = 0x5000;
const CURSOR_USED: u64 = 0x6000;
const REQUEST: u64 = 0x8000;
const ENTRIES: u64 = 0x9000;
const RESPONSE: u64 = 0xa000;
const RESPONSE_BYTES: u32 = 0x400;
/// Guest pages the resources are backed by: two runs, so a transfer crosses an entry.
const BACKING_A: u64 = 0x2_0000;
const BACKING_B: u64 = 0x8_0000;
const RAM_BYTES: usize = 0x10_0000;

const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;
const RESOURCE: u32 = 7;

struct Rig {
    gpu: VirtioGpu,
    display: DisplayHandle,
    queues: Vec<Queue>,
    ram: Vec<u8>,
}

fn header(command: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; HEADER_BYTES];
    bytes[..4].copy_from_slice(&command.to_le_bytes());
    bytes
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn response_type(response: &[u8]) -> u32 {
    u32_at(response, 0)
}

impl Rig {
    fn new() -> Rig {
        let display = DisplayHandle::with_size(WIDTH * 2, HEIGHT * 2);
        let mut ram = vec![0u8; RAM_BYTES];
        let queues = vec![
            setup_queue(
                &mut ram,
                CONTROL_DESC,
                CONTROL_AVAIL,
                CONTROL_USED,
                QUEUE_SIZE,
            ),
            setup_queue(&mut ram, CURSOR_DESC, CURSOR_AVAIL, CURSOR_USED, QUEUE_SIZE),
        ];
        Rig {
            gpu: VirtioGpu::new(display.clone()),
            display,
            queues,
            ram,
        }
    }

    fn kick(&mut self, queue: usize) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.gpu.kick(queue, &mut self.queues, &mut mem)
    }

    /// Sends `request` (split before `split_at` into two descriptors, as Linux sends
    /// `ATTACH_BACKING`) and returns the response the device wrote.
    fn command_split(&mut self, request: &[u8], split_at: usize) -> Vec<u8> {
        let (head, tail) = request.split_at(split_at);
        self.ram[REQUEST as usize..REQUEST as usize + head.len()].copy_from_slice(head);
        self.ram[ENTRIES as usize..ENTRIES as usize + tail.len()].copy_from_slice(tail);
        self.ram[RESPONSE as usize..(RESPONSE + RESPONSE_BYTES as u64) as usize].fill(0xee);
        let mut next = 1;
        put_desc(
            &mut self.ram,
            CONTROL_DESC,
            0,
            REQUEST,
            head.len() as u32,
            DESC_F_NEXT,
            next,
        );
        if !tail.is_empty() {
            put_desc(
                &mut self.ram,
                CONTROL_DESC,
                1,
                ENTRIES,
                tail.len() as u32,
                DESC_F_NEXT,
                2,
            );
            next = 2;
        }
        put_desc(
            &mut self.ram,
            CONTROL_DESC,
            next,
            RESPONSE,
            RESPONSE_BYTES,
            DESC_F_WRITE,
            0,
        );
        post(&mut self.ram, CONTROL_AVAIL, QUEUE_SIZE, 0);
        assert!(self.kick(CONTROL_QUEUE));
        let index = used_idx(&self.ram, CONTROL_USED);
        let (id, length) = used_entry(
            &self.ram,
            CONTROL_USED,
            (index.wrapping_sub(1) % QUEUE_SIZE) as usize,
        );
        assert_eq!(id, 0);
        self.ram[RESPONSE as usize..RESPONSE as usize + length as usize].to_vec()
    }

    fn command(&mut self, request: &[u8]) -> Vec<u8> {
        self.command_split(request, request.len())
    }

    fn status(&mut self, request: &[u8]) -> u32 {
        response_type(&self.command(request))
    }

    fn create(&mut self, id: u32, format: u32, width: u32, height: u32) -> u32 {
        let mut request = header(CMD_RESOURCE_CREATE_2D);
        request.extend(words(&[id, format, width, height]));
        self.status(&request)
    }

    /// Backs `id` with `length` bytes: half at `BACKING_A`, the rest at `BACKING_B`.
    fn attach(&mut self, id: u32, length: u32) -> u32 {
        let first = length / 2;
        let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
        request.extend(words(&[id, 2]));
        let split = request.len();
        for (address, bytes) in [(BACKING_A, first), (BACKING_B, length - first)] {
            request.extend(address.to_le_bytes());
            request.extend(words(&[bytes, 0]));
        }
        response_type(&self.command_split(&request, split))
    }

    fn transfer(&mut self, id: u32, rect: Rect, offset: u64) -> u32 {
        let mut request = header(CMD_TRANSFER_TO_HOST_2D);
        request.extend(words(&[rect.x, rect.y, rect.width, rect.height]));
        request.extend(offset.to_le_bytes());
        request.extend(words(&[id, 0]));
        self.status(&request)
    }

    fn set_scanout(&mut self, scanout: u32, id: u32, rect: Rect) -> u32 {
        let mut request = header(CMD_SET_SCANOUT);
        request.extend(words(&[
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            scanout,
            id,
        ]));
        self.status(&request)
    }

    fn flush(&mut self, id: u32, rect: Rect) -> u32 {
        let mut request = header(CMD_RESOURCE_FLUSH);
        request.extend(words(&[rect.x, rect.y, rect.width, rect.height, id, 0]));
        self.status(&request)
    }

    /// Byte `index` of the guest's image as the two backing runs hold it.
    fn backing_byte_address(&self, index: u32, length: u32) -> usize {
        let first = length / 2;
        if index < first {
            BACKING_A as usize + index as usize
        } else {
            BACKING_B as usize + (index - first) as usize
        }
    }

    /// Paints the guest's copy of pixel (x, y) with these four bytes.
    fn paint(&mut self, x: u32, y: u32, bytes: [u8; 4]) {
        let length = WIDTH * HEIGHT * 4;
        for (offset, byte) in bytes.iter().enumerate() {
            let index = (y * WIDTH + x) * 4 + offset as u32;
            let at = self.backing_byte_address(index, length);
            self.ram[at] = *byte;
        }
    }

    /// A resource of the standard size, backed, transferred, shown and flushed.
    fn show(&mut self, format: u32) {
        assert_eq!(self.create(RESOURCE, format, WIDTH, HEIGHT), RESP_OK_NODATA);
        assert_eq!(self.attach(RESOURCE, WIDTH * HEIGHT * 4), RESP_OK_NODATA);
        assert_eq!(self.transfer(RESOURCE, full(), 0), RESP_OK_NODATA);
        assert_eq!(self.set_scanout(0, RESOURCE, full()), RESP_OK_NODATA);
    }
}

fn full() -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
    }
}

fn screen_pixel(display: &DisplayHandle, x: u32, y: u32) -> [u8; 4] {
    let frame = display.frame();
    let at = ((y * frame.width + x) * 4) as usize;
    frame.pixels[at..at + 4].try_into().unwrap()
}

#[test]
fn identity_is_a_gpu_with_a_control_and_a_cursor_queue_and_one_scanout() {
    let rig = Rig::new();
    assert_eq!(rig.gpu.device_type(), 16);
    assert_eq!(rig.gpu.num_queues(), 2);
    assert_eq!(rig.gpu.features(), 0, "no virgl, no EDID, no blobs");
    let config = |offset: usize| -> u32 {
        (0..4).fold(0, |word, byte| {
            word | (rig.gpu.config_read(offset + byte) as u32) << (8 * byte)
        })
    };
    assert_eq!(config(CONFIG_NUM_SCANOUTS), 1);
    assert_eq!(config(CONFIG_NUM_CAPSETS), 0);
    assert_eq!(config(CONFIG_EVENTS_READ), 0);
}

#[test]
fn display_info_offers_scanout_zero_at_the_display_size() {
    let mut rig = Rig::new();
    let response = rig.command(&header(CMD_GET_DISPLAY_INFO));
    assert_eq!(response_type(&response), RESP_OK_DISPLAY_INFO);
    assert_eq!(
        response.len(),
        HEADER_BYTES + MAX_SCANOUTS * DISPLAY_ONE_BYTES
    );
    let mode = &response[HEADER_BYTES..];
    assert_eq!(
        (
            u32_at(mode, 0),
            u32_at(mode, 4),
            u32_at(mode, 8),
            u32_at(mode, 12)
        ),
        (0, 0, WIDTH * 2, HEIGHT * 2)
    );
    assert_eq!(u32_at(mode, RECT_BYTES), 1, "enabled");
    assert_eq!(
        u32_at(mode, DISPLAY_ONE_BYTES + RECT_BYTES),
        0,
        "scanout 1 is not"
    );
}

#[test]
fn a_fenced_command_echoes_its_fence() {
    let mut rig = Rig::new();
    let mut request = header(CMD_RESOURCE_CREATE_2D);
    request[HEADER_FLAGS..HEADER_FLAGS + 4].copy_from_slice(&FLAG_FENCE.to_le_bytes());
    request[HEADER_FENCE_ID..HEADER_FENCE_ID + 8]
        .copy_from_slice(&0x1234_5678_9abcu64.to_le_bytes());
    request[HEADER_CONTEXT_ID..HEADER_CONTEXT_ID + 4].copy_from_slice(&3u32.to_le_bytes());
    request.extend(words(&[1, FORMAT_B8G8R8X8, 4, 4]));
    let response = rig.command(&request);
    assert_eq!(response.len(), HEADER_BYTES);
    assert_eq!(response_type(&response), RESP_OK_NODATA);
    assert_eq!(u32_at(&response, HEADER_FLAGS), FLAG_FENCE);
    assert_eq!(u64_at(&response, HEADER_FENCE_ID), 0x1234_5678_9abc);
    assert_eq!(u32_at(&response, HEADER_CONTEXT_ID), 3);
}

#[test]
fn an_unfenced_command_leaves_the_fence_fields_zero() {
    let mut rig = Rig::new();
    let mut request = header(CMD_GET_DISPLAY_INFO);
    request[HEADER_FENCE_ID] = 9;
    let response = rig.command(&request);
    assert_eq!(u32_at(&response, HEADER_FLAGS), 0);
    assert_eq!(u64_at(&response, HEADER_FENCE_ID), 0);
}

#[test]
fn transfer_scanout_and_flush_put_the_guest_image_on_the_screen() {
    let mut rig = Rig::new();
    // Pixels in both backing runs: the image crosses from the first entry into the second.
    rig.paint(1, 0, [10, 20, 30, 0]);
    rig.paint(WIDTH - 1, HEIGHT - 1, [40, 50, 60, 0]);
    rig.show(FORMAT_B8G8R8X8);
    assert!(rig.display.virtio_owns_screen());
    assert_eq!(rig.display.size(), (WIDTH, HEIGHT), "the scanout's size");
    assert_eq!(screen_pixel(&rig.display, 1, 0)[..3], [10, 20, 30]);
    assert_eq!(
        screen_pixel(&rig.display, WIDTH - 1, HEIGHT - 1)[..3],
        [40, 50, 60]
    );
    // A later change shows only after transfer and flush, and only the flushed part is damage.
    let generation = rig.display.generation();
    rig.paint(2, 3, [1, 2, 3, 0]);
    let rect = Rect {
        x: 2,
        y: 3,
        width: 1,
        height: 1,
    };
    let offset = ((3 * WIDTH + 2) * 4) as u64;
    assert_eq!(rig.transfer(RESOURCE, rect, offset), RESP_OK_NODATA);
    assert_eq!(
        rig.display.generation(),
        generation,
        "a transfer is not a present"
    );
    assert_eq!(rig.flush(RESOURCE, rect), RESP_OK_NODATA);
    assert_eq!(screen_pixel(&rig.display, 2, 3)[..3], [1, 2, 3]);
    let mut pixels = Vec::new();
    let update = rig.display.update_since(generation, &mut pixels).unwrap();
    assert_eq!(update.rect, rect);
    assert_eq!(pixels[..3], [1, 2, 3]);
}

#[test]
fn the_scanout_rectangle_picks_part_of_a_larger_resource() {
    let mut rig = Rig::new();
    rig.paint(5, 2, [7, 8, 9, 0]);
    assert_eq!(
        rig.create(RESOURCE, FORMAT_B8G8R8X8, WIDTH, HEIGHT),
        RESP_OK_NODATA
    );
    assert_eq!(rig.attach(RESOURCE, WIDTH * HEIGHT * 4), RESP_OK_NODATA);
    assert_eq!(rig.transfer(RESOURCE, full(), 0), RESP_OK_NODATA);
    let window = Rect {
        x: 4,
        y: 2,
        width: 8,
        height: 4,
    };
    assert_eq!(rig.set_scanout(0, RESOURCE, window), RESP_OK_NODATA);
    assert_eq!(rig.display.size(), (8, 4));
    assert_eq!(screen_pixel(&rig.display, 1, 0)[..3], [7, 8, 9]);
}

#[test]
fn every_format_is_converted_to_blue_green_red() {
    // The bytes of one pixel with blue 1, green 2, red 3, in each format's memory order.
    let cases = [
        (FORMAT_B8G8R8A8, [1, 2, 3, 0xff]),
        (FORMAT_B8G8R8X8, [1, 2, 3, 0]),
        (FORMAT_A8R8G8B8, [0xff, 3, 2, 1]),
        (FORMAT_X8R8G8B8, [0, 3, 2, 1]),
        (FORMAT_R8G8B8A8, [3, 2, 1, 0xff]),
        (FORMAT_R8G8B8X8, [3, 2, 1, 0]),
        (FORMAT_A8B8G8R8, [0xff, 1, 2, 3]),
        (FORMAT_X8B8G8R8, [0, 1, 2, 3]),
    ];
    for (format, bytes) in cases {
        let mut rig = Rig::new();
        rig.paint(0, 0, bytes);
        rig.show(format);
        assert_eq!(
            screen_pixel(&rig.display, 0, 0)[..3],
            [1, 2, 3],
            "format {format}"
        );
    }
}

#[test]
fn flushing_a_resource_that_is_not_shown_changes_nothing() {
    let mut rig = Rig::new();
    rig.show(FORMAT_B8G8R8X8);
    assert_eq!(
        rig.create(RESOURCE + 1, FORMAT_B8G8R8X8, 4, 4),
        RESP_OK_NODATA
    );
    let generation = rig.display.generation();
    let small = Rect {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };
    assert_eq!(rig.flush(RESOURCE + 1, small), RESP_OK_NODATA);
    assert_eq!(rig.display.generation(), generation);
}

#[test]
fn creating_a_resource_checks_its_id_format_and_size() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.create(0, FORMAT_B8G8R8X8, 4, 4),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    assert_eq!(rig.create(1, FORMAT_B8G8R8X8, 4, 4), RESP_OK_NODATA);
    assert_eq!(
        rig.create(1, FORMAT_B8G8R8X8, 4, 4),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    assert_eq!(rig.create(2, 99, 4, 4), RESP_ERR_INVALID_PARAMETER);
    assert_eq!(
        rig.create(2, FORMAT_B8G8R8X8, 0, 4),
        RESP_ERR_INVALID_PARAMETER
    );
    assert_eq!(
        rig.create(2, FORMAT_B8G8R8X8, MAX_RESOURCE_DIMENSION + 1, 1),
        RESP_ERR_INVALID_PARAMETER
    );
}

#[test]
fn the_resource_budget_refuses_what_would_exhaust_host_memory() {
    let mut rig = Rig::new();
    // 16384 x 16384 x 4 is a GiB, past the budget: refused, nothing allocated.
    let side = MAX_RESOURCE_DIMENSION;
    assert_eq!(
        rig.create(1, FORMAT_B8G8R8X8, side, side),
        RESP_ERR_OUT_OF_MEMORY
    );
    assert_eq!(rig.gpu.resource_bytes, 0);
    // Resources that fit one by one are refused once the total would not.
    let quarter = side / 4;
    let mut created = 0;
    for id in 1..=64 {
        match rig.create(id, FORMAT_B8G8R8X8, side, quarter) {
            RESP_OK_NODATA => created += 1,
            status => {
                assert_eq!(status, RESP_ERR_OUT_OF_MEMORY);
                break;
            }
        }
    }
    assert_eq!(
        created,
        MAX_RESOURCE_BYTES / (side as usize * quarter as usize * 4)
    );
    // Freeing one makes room again.
    let mut unref = header(CMD_RESOURCE_UNREF);
    unref.extend(words(&[1, 0]));
    assert_eq!(rig.status(&unref), RESP_OK_NODATA);
    assert_eq!(
        rig.create(1000, FORMAT_B8G8R8X8, side, quarter),
        RESP_OK_NODATA
    );
}

#[test]
fn backing_outside_ram_is_refused() {
    let mut rig = Rig::new();
    assert_eq!(rig.create(1, FORMAT_B8G8R8X8, 4, 4), RESP_OK_NODATA);
    let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
    request.extend(words(&[1, 1]));
    request.extend((RAM_BYTES as u64 - 8).to_le_bytes());
    request.extend(words(&[64, 0]));
    assert_eq!(rig.status(&request), RESP_ERR_UNSPEC);
    // An entry count larger than the request carries, or than the cap, is refused too.
    let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
    request.extend(words(&[1, 3]));
    assert_eq!(rig.status(&request), RESP_ERR_UNSPEC);
    let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
    request.extend(words(&[1, MAX_BACKING_ENTRIES + 1]));
    assert_eq!(rig.status(&request), RESP_ERR_UNSPEC);
    let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
    request.extend(words(&[9, 1]));
    request.extend(BACKING_A.to_le_bytes());
    request.extend(words(&[64, 0]));
    assert_eq!(rig.status(&request), RESP_ERR_INVALID_RESOURCE_ID);
}

#[test]
fn transfers_are_checked_against_the_resource() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.create(RESOURCE, FORMAT_B8G8R8X8, WIDTH, HEIGHT),
        RESP_OK_NODATA
    );
    assert_eq!(
        rig.transfer(RESOURCE, full(), 0),
        RESP_ERR_UNSPEC,
        "no backing"
    );
    assert_eq!(rig.attach(RESOURCE, WIDTH * HEIGHT * 4), RESP_OK_NODATA);
    let outside = Rect {
        x: WIDTH - 1,
        y: 0,
        width: 2,
        height: 1,
    };
    assert_eq!(
        rig.transfer(RESOURCE, outside, 0),
        RESP_ERR_INVALID_PARAMETER
    );
    let overflowing = Rect {
        x: u32::MAX,
        y: 0,
        width: 2,
        height: 1,
    };
    assert_eq!(
        rig.transfer(RESOURCE, overflowing, 0),
        RESP_ERR_INVALID_PARAMETER
    );
    assert_eq!(
        rig.transfer(RESOURCE + 1, full(), 0),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    // An offset past the backing reads zeros rather than failing or reading elsewhere.
    assert_eq!(rig.transfer(RESOURCE, full(), u64::MAX / 2), RESP_OK_NODATA);
    // Detached backing cannot be read any more.
    let mut detach = header(CMD_RESOURCE_DETACH_BACKING);
    detach.extend(words(&[RESOURCE, 0]));
    assert_eq!(rig.status(&detach), RESP_OK_NODATA);
    assert_eq!(rig.transfer(RESOURCE, full(), 0), RESP_ERR_UNSPEC);
}

#[test]
fn a_transfer_carries_rows_across_many_small_out_of_order_entries() {
    // Entries of 20 bytes, so pixels straddle them, laid out in RAM in reverse: a row of the
    // rectangle starts part way into one entry and ends in another, as a dumb buffer's rows
    // cross its pages. Each image byte is its own index, so a misplaced byte shows.
    const ENTRY_BYTES: u32 = 20;
    let mut rig = Rig::new();
    let length = WIDTH * HEIGHT * 4;
    let count = length.div_ceil(ENTRY_BYTES);
    let entry_address = |index: u32| BACKING_A + u64::from((count - 1 - index) * 32);
    for byte in 0..length {
        let at = entry_address(byte / ENTRY_BYTES) + u64::from(byte % ENTRY_BYTES);
        rig.ram[at as usize] = byte as u8;
    }
    assert_eq!(
        rig.create(RESOURCE, FORMAT_B8G8R8X8, WIDTH, HEIGHT),
        RESP_OK_NODATA
    );
    let mut request = header(CMD_RESOURCE_ATTACH_BACKING);
    request.extend(words(&[RESOURCE, count]));
    let split = request.len();
    for index in 0..count {
        request.extend(entry_address(index).to_le_bytes());
        request.extend(words(&[ENTRY_BYTES, 0]));
    }
    assert_eq!(
        response_type(&rig.command_split(&request, split)),
        RESP_OK_NODATA
    );
    let rect = Rect {
        x: 3,
        y: 2,
        width: 5,
        height: 4,
    };
    let offset = u64::from((rect.y * WIDTH + rect.x) * 4);
    assert_eq!(rig.transfer(RESOURCE, rect, offset), RESP_OK_NODATA);
    let pixels = &rig.gpu.resources[&RESOURCE].pixels;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = ((y * WIDTH + x) * 4) as usize;
            let inside = (rect.x..rect.x + rect.width).contains(&x)
                && (rect.y..rect.y + rect.height).contains(&y);
            let expected: [u8; 4] = if inside {
                core::array::from_fn(|byte| (at + byte) as u8)
            } else {
                [0; 4]
            };
            assert_eq!(pixels[at..at + 4], expected, "pixel {x},{y}");
        }
    }
}

#[test]
fn set_scanout_checks_the_scanout_the_resource_and_the_rectangle() {
    let mut rig = Rig::new();
    assert_eq!(
        rig.create(RESOURCE, FORMAT_B8G8R8X8, WIDTH, HEIGHT),
        RESP_OK_NODATA
    );
    assert_eq!(
        rig.set_scanout(1, RESOURCE, full()),
        RESP_ERR_INVALID_SCANOUT_ID
    );
    assert_eq!(rig.set_scanout(0, 99, full()), RESP_ERR_INVALID_RESOURCE_ID);
    let too_big = Rect {
        width: WIDTH + 1,
        ..full()
    };
    assert_eq!(
        rig.set_scanout(0, RESOURCE, too_big),
        RESP_ERR_INVALID_PARAMETER
    );
    let empty = Rect { width: 0, ..full() };
    assert_eq!(
        rig.set_scanout(0, RESOURCE, empty),
        RESP_ERR_INVALID_PARAMETER
    );
}

#[test]
fn turning_the_scanout_off_or_freeing_its_resource_blanks_the_screen() {
    let mut rig = Rig::new();
    rig.paint(0, 0, [9, 9, 9, 0]);
    rig.show(FORMAT_B8G8R8X8);
    assert_eq!(rig.set_scanout(0, 0, Rect::default()), RESP_OK_NODATA);
    assert_eq!(screen_pixel(&rig.display, 0, 0), [0; 4]);
    assert!(
        rig.display.virtio_owns_screen(),
        "still the driver's screen"
    );

    let mut rig = Rig::new();
    rig.paint(0, 0, [9, 9, 9, 0]);
    rig.show(FORMAT_B8G8R8X8);
    let mut unref = header(CMD_RESOURCE_UNREF);
    unref.extend(words(&[RESOURCE, 0]));
    assert_eq!(rig.status(&unref), RESP_OK_NODATA);
    assert_eq!(screen_pixel(&rig.display, 0, 0), [0; 4]);
    assert_eq!(rig.status(&unref), RESP_ERR_INVALID_RESOURCE_ID);
}

#[test]
fn unknown_and_short_commands_get_an_error() {
    let mut rig = Rig::new();
    const CMD_CTX_CREATE: u32 = 0x0200;
    assert_eq!(rig.status(&header(CMD_CTX_CREATE)), RESP_ERR_UNSPEC);
    assert_eq!(
        rig.status(&header(CMD_GET_DISPLAY_INFO)[..8]),
        RESP_ERR_UNSPEC
    );
}

#[test]
fn a_response_buffer_too_small_gets_what_fits() {
    let mut rig = Rig::new();
    rig.ram[REQUEST as usize..REQUEST as usize + HEADER_BYTES]
        .copy_from_slice(&header(CMD_GET_DISPLAY_INFO));
    put_desc(
        &mut rig.ram,
        CONTROL_DESC,
        0,
        REQUEST,
        HEADER_BYTES as u32,
        DESC_F_NEXT,
        1,
    );
    put_desc(&mut rig.ram, CONTROL_DESC, 1, RESPONSE, 8, DESC_F_WRITE, 0);
    post(&mut rig.ram, CONTROL_AVAIL, QUEUE_SIZE, 0);
    assert!(rig.kick(CONTROL_QUEUE));
    assert_eq!(used_entry(&rig.ram, CONTROL_USED, 0), (0, 8));
}

impl Rig {
    /// Sends one cursor command (`struct virtio_gpu_update_cursor`) on the cursor queue.
    fn cursor(&mut self, command: u32, id: u32, hot_x: u32, hot_y: u32) {
        let mut request = header(command);
        // pos: scanout 0 at (5, 6); then resource_id, hot_x, hot_y, padding.
        request.extend(words(&[0, 5, 6, 0, id, hot_x, hot_y, 0]));
        assert_eq!(request.len(), CURSOR_REQUEST_BYTES);
        let used_before = used_idx(&self.ram, CURSOR_USED);
        self.ram[REQUEST as usize..REQUEST as usize + request.len()].copy_from_slice(&request);
        put_desc(
            &mut self.ram,
            CURSOR_DESC,
            0,
            REQUEST,
            request.len() as u32,
            0,
            0,
        );
        post(&mut self.ram, CURSOR_AVAIL, QUEUE_SIZE, 0);
        assert!(self.kick(CURSOR_QUEUE));
        let index = used_idx(&self.ram, CURSOR_USED);
        assert_eq!(index, used_before.wrapping_add(1));
        let entry = used_entry(
            &self.ram,
            CURSOR_USED,
            (index.wrapping_sub(1) % QUEUE_SIZE) as usize,
        );
        assert_eq!(entry, (0, 0), "completed with nothing written");
    }
}

const CMD_MOVE_CURSOR: u32 = 0x0301;

#[test]
fn moving_the_cursor_is_completed_and_ignored() {
    let mut rig = Rig::new();
    rig.cursor(CMD_MOVE_CURSOR, RESOURCE, 1, 1);
    assert_eq!(rig.display.cursor(), (0, Cursor::Host));
}

#[test]
fn update_cursor_hands_the_frontend_the_resource_image_with_alpha_and_hotspot() {
    let mut rig = Rig::new();
    // Linux's cursor plane: ARGB8888, which virtio calls B8G8R8A8.
    rig.paint(0, 0, [0x10, 0x20, 0x30, 0x80]);
    rig.paint(WIDTH - 1, HEIGHT - 1, [0xa0, 0xb0, 0xc0, 0xff]);
    rig.show(FORMAT_B8G8R8A8);
    rig.cursor(CMD_UPDATE_CURSOR, RESOURCE, 3, 2);
    let (generation, cursor) = rig.display.cursor();
    assert_eq!(generation, 1);
    let Cursor::Image(image) = cursor else {
        panic!("expected an image, got {cursor:?}");
    };
    assert_eq!((image.width, image.height), (WIDTH, HEIGHT));
    assert_eq!((image.hot_x, image.hot_y), (3, 2));
    assert_eq!(image.pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(image.pixels[..4], [0x10, 0x20, 0x30, 0x80]);
    assert_eq!(
        image.pixels[image.pixels.len() - 4..],
        [0xa0, 0xb0, 0xc0, 0xff]
    );
    // The same image again is no change.
    rig.cursor(CMD_UPDATE_CURSOR, RESOURCE, 3, 2);
    assert_eq!(rig.display.cursor().0, 1);
}

#[test]
fn a_cursor_in_an_x_format_takes_its_alpha_from_the_fourth_byte() {
    // Linux creates every dumb buffer, the cursor's included, as XRGB8888 (B8G8R8X8) and puts
    // ARGB pixels in it; QEMU copies the words as ARGB whatever the format says. Forcing the
    // fourth byte to opaque turned the transparent pixels around the arrow into a black square.
    let mut rig = Rig::new();
    rig.paint(0, 0, [0, 0, 0, 0]);
    rig.paint(1, 0, [0xff, 0xff, 0xff, 0xff]);
    rig.show(FORMAT_B8G8R8X8);
    rig.cursor(CMD_UPDATE_CURSOR, RESOURCE, 0, 0);
    let Cursor::Image(image) = rig.display.cursor().1 else {
        panic!("expected an image");
    };
    assert_eq!(image.pixels[..4], [0, 0, 0, 0], "transparent, not black");
    assert_eq!(image.pixels[4..8], [0xff, 0xff, 0xff, 0xff]);
}

#[test]
fn red_blue_cursor_formats_are_swapped_and_keep_their_alpha_byte() {
    let mut rig = Rig::new();
    rig.paint(0, 0, [0x30, 0x20, 0x10, 0x40]);
    rig.show(FORMAT_R8G8B8X8);
    rig.cursor(CMD_UPDATE_CURSOR, RESOURCE, 0, 0);
    let Cursor::Image(image) = rig.display.cursor().1 else {
        panic!("expected an image");
    };
    assert_eq!(image.pixels[..4], [0x10, 0x20, 0x30, 0x40]);
}

#[test]
fn every_cursor_format_finds_alpha_in_the_byte_its_colours_leave() {
    for (format, alpha) in [
        (FORMAT_B8G8R8A8, 3),
        (FORMAT_B8G8R8X8, 3),
        (FORMAT_A8R8G8B8, 0),
        (FORMAT_X8R8G8B8, 0),
        (FORMAT_R8G8B8A8, 3),
        (FORMAT_R8G8B8X8, 3),
        (FORMAT_A8B8G8R8, 0),
        (FORMAT_X8B8G8R8, 0),
    ] {
        assert_eq!(cursor_alpha_offset(format), Some(alpha), "format {format}");
    }
    assert_eq!(cursor_alpha_offset(0), None);
}

#[test]
fn resource_zero_hides_the_cursor_and_a_reset_gives_it_back_to_the_host() {
    let mut rig = Rig::new();
    rig.cursor(CMD_UPDATE_CURSOR, 0, 0, 0);
    assert_eq!(rig.display.cursor(), (1, Cursor::Hidden));
    rig.gpu.reset();
    assert_eq!(rig.display.cursor(), (2, Cursor::Host));
}

#[test]
fn the_device_of_the_next_boot_takes_the_screen_and_pointer_back_without_a_reset() {
    let mut rig = Rig::new();
    rig.show(FORMAT_B8G8R8X8);
    rig.cursor(CMD_UPDATE_CURSOR, 0, 0, 0);
    assert!(rig.display.virtio_owns_screen());
    assert_eq!(rig.display.cursor().1, Cursor::Hidden);
    // A restart that skipped the driver's shutdown: no reset, just new devices.
    let _next_boot = VirtioGpu::new(rig.display.clone());
    assert!(
        !rig.display.virtio_owns_screen(),
        "the boot framebuffer is scanned again"
    );
    assert_eq!(rig.display.cursor().1, Cursor::Host);
}

#[test]
fn a_cursor_from_a_missing_or_oversized_resource_or_out_of_range_hotspot_is_contained() {
    let mut rig = Rig::new();
    rig.cursor(CMD_UPDATE_CURSOR, 99, 0, 0);
    assert_eq!(rig.display.cursor(), (0, Cursor::Host), "no such resource");
    const BIG: u32 = 99;
    assert_eq!(
        rig.create(BIG, FORMAT_B8G8R8A8, MAX_CURSOR_SIDE + 1, 1),
        RESP_OK_NODATA
    );
    rig.cursor(CMD_UPDATE_CURSOR, BIG, 0, 0);
    assert_eq!(
        rig.display.cursor(),
        (0, Cursor::Host),
        "too big to be a pointer"
    );
    rig.show(FORMAT_B8G8R8A8);
    rig.cursor(CMD_UPDATE_CURSOR, RESOURCE, u32::MAX, u32::MAX);
    let Cursor::Image(image) = rig.display.cursor().1 else {
        panic!("expected an image");
    };
    assert_eq!(
        (image.hot_x, image.hot_y),
        (WIDTH - 1, HEIGHT - 1),
        "clamped"
    );
}

#[test]
fn a_requested_size_becomes_the_preferred_mode_with_a_display_event() {
    let mut rig = Rig::new();
    rig.display.request_size(1024, 768);
    {
        let mut mem = test_mem(&mut rig.ram);
        assert!(!rig.gpu.poll(&mut rig.queues, &mut mem));
    }
    assert!(rig.gpu.config_event());
    assert!(!rig.gpu.config_event(), "signalled once");
    assert_eq!(
        rig.gpu.config_read(CONFIG_EVENTS_READ) as u32,
        EVENT_DISPLAY
    );
    let response = rig.command(&header(CMD_GET_DISPLAY_INFO));
    let mode = &response[HEADER_BYTES..];
    assert_eq!((u32_at(mode, 8), u32_at(mode, 12)), (1024, 768));
    rig.gpu
        .config_write(CONFIG_EVENTS_CLEAR, EVENT_DISPLAY as u8);
    assert_eq!(rig.gpu.config_read(CONFIG_EVENTS_READ), 0);
    // The same size again is no change.
    rig.display.request_size(1024, 768);
    let mut mem = test_mem(&mut rig.ram);
    rig.gpu.poll(&mut rig.queues, &mut mem);
    assert!(!rig.gpu.config_event());
}

#[test]
fn a_device_reset_frees_everything_and_hands_the_screen_back() {
    let mut rig = Rig::new();
    rig.show(FORMAT_B8G8R8X8);
    rig.gpu.reset();
    assert!(!rig.display.virtio_owns_screen());
    assert_eq!(rig.gpu.resource_bytes, 0);
    assert_eq!(rig.flush(RESOURCE, full()), RESP_ERR_INVALID_RESOURCE_ID);
}
