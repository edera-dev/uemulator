use super::*;
use crate::dev::guestmem::test_mem;
use crate::dev::virtio::DESC_F_WRITE;
use crate::dev::virtio::testutil::*;
use crate::display::{DisplayHandle, SYN_REPORT};

const QUEUE_SIZE: u16 = 8;
const EVENT_DESC: u64 = 0x1000;
const EVENT_AVAIL: u64 = 0x2000;
const EVENT_USED: u64 = 0x3000;
const STATUS_DESC: u64 = 0x4000;
const STATUS_AVAIL: u64 = 0x5000;
const STATUS_USED: u64 = 0x6000;
const BUFFERS: u64 = 0x8000;
const RAM_BYTES: usize = 0x1_0000;
const KEY_A: u16 = 30;
const SCREEN_WIDTH: u32 = 640;
const SCREEN_HEIGHT: u32 = 480;

struct Rig {
    device: VirtioInput,
    input: InputHandle,
    queues: Vec<Queue>,
    ram: Vec<u8>,
    posted: u16,
}

impl Rig {
    fn new(kind: InputDevice) -> Rig {
        let display = DisplayHandle::with_size(SCREEN_WIDTH, SCREEN_HEIGHT);
        let input = InputHandle::for_display(Some(display));
        let mut ram = vec![0u8; RAM_BYTES];
        let queues = vec![
            setup_queue(&mut ram, EVENT_DESC, EVENT_AVAIL, EVENT_USED, QUEUE_SIZE),
            setup_queue(&mut ram, STATUS_DESC, STATUS_AVAIL, STATUS_USED, QUEUE_SIZE),
        ];
        Rig {
            device: VirtioInput::new(kind, input.clone()),
            input,
            queues,
            ram,
            posted: 0,
        }
    }

    /// Posts `count` empty event buffers, as the driver does at probe and after each event.
    fn post_buffers(&mut self, count: u16) {
        for _ in 0..count {
            let head = self.posted % QUEUE_SIZE;
            let address = BUFFERS + head as u64 * EVENT_BYTES as u64;
            put_desc(
                &mut self.ram,
                EVENT_DESC,
                head,
                address,
                EVENT_BYTES as u32,
                DESC_F_WRITE,
                0,
            );
            post(&mut self.ram, EVENT_AVAIL, QUEUE_SIZE, head);
            self.posted += 1;
        }
    }

    fn kick(&mut self, queue: usize) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.device.kick(queue, &mut self.queues, &mut mem)
    }

    fn poll(&mut self) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.device.poll(&mut self.queues, &mut mem)
    }

    /// Every event the device completed so far, in order.
    fn delivered(&self) -> Vec<(u16, u16, u32)> {
        let count = used_idx(&self.ram, EVENT_USED);
        (0..count)
            .map(|slot| {
                let (head, length) =
                    used_entry(&self.ram, EVENT_USED, (slot % QUEUE_SIZE) as usize);
                assert_eq!(length, EVENT_BYTES as u32);
                let at = (BUFFERS + head as u64 * EVENT_BYTES as u64) as usize;
                let bytes = &self.ram[at..at + EVENT_BYTES];
                (
                    u16::from_le_bytes([bytes[0], bytes[1]]),
                    u16::from_le_bytes([bytes[2], bytes[3]]),
                    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                )
            })
            .collect()
    }

    /// The configuration union for `select`/`subsel`, as the driver reads it.
    fn query(&mut self, select: u8, subsel: u8) -> Vec<u8> {
        self.device.config_write(CONFIG_SELECT, select);
        self.device.config_write(CONFIG_SUBSEL, subsel);
        let size = self.device.config_read(CONFIG_SIZE) as usize;
        (0..size)
            .map(|index| self.device.config_read(CONFIG_PAYLOAD + index))
            .collect()
    }
}

fn bit(bitmap: &[u8], index: usize) -> bool {
    bitmap
        .get(index / 8)
        .is_some_and(|byte| byte & (1 << (index % 8)) != 0)
}

const SYN: (u16, u16, u32) = (EV_SYN_TYPE, SYN_REPORT, 0);
const EV_SYN_TYPE: u16 = 0;

#[test]
fn identity_is_an_input_device_with_an_event_and_a_status_queue() {
    let rig = Rig::new(InputDevice::Keyboard);
    assert_eq!(rig.device.device_type(), 18);
    assert_eq!(rig.device.num_queues(), 2);
    assert_eq!(rig.device.features(), 0);
}

#[test]
fn the_keyboard_describes_itself_with_keys_repeat_and_leds() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    assert_eq!(rig.query(SELECT_ID_NAME, 0), KEYBOARD_NAME.as_bytes());
    let ids = rig.query(SELECT_ID_DEVIDS, 0);
    assert_eq!(ids.len(), 8);
    assert_eq!(u16::from_le_bytes([ids[0], ids[1]]), BUS_VIRTUAL);
    let keys = rig.query(SELECT_EV_BITS, EV_KEY as u8);
    assert!(!bit(&keys, 0), "code 0 is KEY_RESERVED");
    assert!((1..=KEY_CODE_MAX as usize).all(|code| bit(&keys, code)));
    assert_eq!(keys.len(), 32);
    assert!(!rig.query(SELECT_EV_BITS, EV_REP).is_empty());
    let leds = rig.query(SELECT_EV_BITS, EV_LED);
    assert!(bit(&leds, LED_CAPSL as usize));
    assert!(rig.query(SELECT_EV_BITS, EV_ABS as u8).is_empty());
    assert!(rig.query(SELECT_ABS_INFO, ABS_X as u8).is_empty());
}

#[test]
fn the_tablet_describes_absolute_axes_buttons_and_a_wheel() {
    let mut rig = Rig::new(InputDevice::Tablet);
    assert_eq!(rig.query(SELECT_ID_NAME, 0), TABLET_NAME.as_bytes());
    let axes = rig.query(SELECT_EV_BITS, EV_ABS as u8);
    assert!(bit(&axes, ABS_X as usize) && bit(&axes, ABS_Y as usize));
    let buttons = rig.query(SELECT_EV_BITS, EV_KEY as u8);
    assert!(bit(&buttons, 0x110) && bit(&buttons, 0x111) && bit(&buttons, 0x112));
    assert!(!bit(&buttons, KEY_A as usize), "no keys");
    assert!(bit(
        &rig.query(SELECT_EV_BITS, EV_REL as u8),
        REL_WHEEL as usize
    ));
    for axis in [ABS_X, ABS_Y] {
        let info = rig.query(SELECT_ABS_INFO, axis as u8);
        assert_eq!(info.len(), 20);
        assert_eq!(u32::from_le_bytes(info[0..4].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(info[4..8].try_into().unwrap()),
            POINTER_AXIS_MAX
        );
    }
    assert!(rig.query(SELECT_EV_BITS, EV_REP).is_empty());
}

#[test]
fn an_unknown_query_has_size_zero() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    assert!(rig.query(0x7f, 0).is_empty());
    assert!(rig.query(SELECT_ID_SERIAL, 0).is_empty());
    assert_eq!(rig.device.config_read(CONFIG_SELECT), SELECT_ID_SERIAL);
}

#[test]
fn a_key_press_reaches_the_guest_as_the_key_and_a_report() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    rig.post_buffers(4);
    assert!(rig.input.key(KEY_A, true));
    assert!(rig.poll());
    assert!(rig.input.key(KEY_A, false));
    assert!(rig.poll());
    assert_eq!(
        rig.delivered(),
        vec![(EV_KEY, KEY_A, 1), SYN, (EV_KEY, KEY_A, 0), SYN,]
    );
    assert!(!rig.poll(), "nothing left");
}

#[test]
fn a_batch_waits_until_the_driver_has_room_for_all_of_it() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    rig.input.key(KEY_A, true);
    rig.post_buffers(1);
    assert!(!rig.poll(), "one buffer is not enough for key and report");
    assert!(rig.delivered().is_empty());
    rig.post_buffers(1);
    assert!(rig.kick(EVENT_QUEUE), "the kick that adds buffers delivers");
    assert_eq!(rig.delivered(), vec![(EV_KEY, KEY_A, 1), SYN]);
}

#[test]
fn input_before_the_driver_posts_buffers_is_kept_for_it() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    for _ in 0..3 {
        rig.input.key(KEY_A, true);
        rig.input.key(KEY_A, false);
    }
    rig.post_buffers(QUEUE_SIZE);
    assert!(rig.poll());
    assert_eq!(rig.delivered().len(), 8, "four whole batches fit");
    rig.post_buffers(QUEUE_SIZE);
    assert!(rig.poll());
    assert_eq!(rig.delivered().len(), 12);
}

#[test]
fn a_full_queue_refuses_more_input() {
    let rig = Rig::new(InputDevice::Keyboard);
    let mut accepted = 0;
    while rig.input.key(KEY_A, accepted % 2 == 0) {
        accepted += 1;
        assert!(accepted < 10_000);
    }
    assert!(accepted >= 64);
}

#[test]
fn keys_outside_the_advertised_range_are_refused() {
    let rig = Rig::new(InputDevice::Keyboard);
    assert!(!rig.input.key(0, true));
    assert!(!rig.input.key(KEY_CODE_MAX + 1, true));
    assert!(!rig.input.key_usb(0x00, true));
    assert!(rig.input.key_usb(0x04, true), "USB usage 4 is 'a'");
}

#[test]
fn pointer_positions_are_scaled_from_screen_pixels_to_the_axis_range() {
    let mut rig = Rig::new(InputDevice::Tablet);
    rig.post_buffers(QUEUE_SIZE);
    rig.input.pointer_move(SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1);
    rig.poll();
    rig.input.pointer_move(0, (SCREEN_HEIGHT - 1) / 2);
    rig.poll();
    let events = rig.delivered();
    assert_eq!(
        events[..3],
        [
            (EV_ABS, ABS_X, POINTER_AXIS_MAX),
            (EV_ABS, ABS_Y, POINTER_AXIS_MAX),
            SYN
        ]
    );
    assert_eq!(events[3], (EV_ABS, ABS_X, 0));
    assert_eq!(events[4].2, POINTER_AXIS_MAX * 239 / 479);
    // Past the edge is the edge. Two buffers are left; one more fits a move.
    rig.post_buffers(1);
    rig.input.pointer_move(SCREEN_WIDTH * 4, 0);
    rig.poll();
    assert_eq!(rig.delivered()[6], (EV_ABS, ABS_X, POINTER_AXIS_MAX));
}

#[test]
fn moves_waiting_in_a_row_collapse_into_the_latest() {
    let mut rig = Rig::new(InputDevice::Tablet);
    for x in 0..100 {
        assert!(rig.input.pointer_move(x, 0));
    }
    rig.input.pointer_button(PointerButton::Left, true);
    rig.input.pointer_move(5, 5);
    rig.post_buffers(QUEUE_SIZE);
    rig.poll();
    rig.post_buffers(QUEUE_SIZE);
    rig.poll();
    let events = rig.delivered();
    assert_eq!(events.len(), 3 + 2 + 3, "one move, the click, one move");
    assert_eq!(events[0].2, 99 * POINTER_AXIS_MAX / (SCREEN_WIDTH - 1));
    assert_eq!(events[3], (EV_KEY, 0x110, 1));
}

#[test]
fn the_wheel_sends_relative_clicks() {
    let mut rig = Rig::new(InputDevice::Tablet);
    rig.post_buffers(4);
    rig.input.scroll(-2);
    assert!(rig.input.scroll(0), "no movement is nothing to send");
    rig.poll();
    assert_eq!(
        rig.delivered(),
        vec![(EV_REL, REL_WHEEL, -2i32 as u32), SYN]
    );
}

#[test]
fn release_all_lets_go_of_held_keys_and_buttons() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    rig.input.key(KEY_A, true);
    rig.input.key(42, true);
    rig.input.key(42, false);
    rig.input.pointer_button(PointerButton::Right, true);
    rig.input.release_all();
    rig.post_buffers(QUEUE_SIZE);
    rig.poll();
    rig.post_buffers(QUEUE_SIZE);
    rig.poll();
    let keys: Vec<_> = rig
        .delivered()
        .into_iter()
        .filter(|e| e.0 == EV_KEY)
        .collect();
    assert_eq!(
        keys,
        vec![
            (EV_KEY, KEY_A, 1),
            (EV_KEY, 42, 1),
            (EV_KEY, 42, 0),
            (EV_KEY, KEY_A, 0)
        ]
    );
    let tablet = rig.input.peek(InputDevice::Tablet);
    assert_eq!(tablet.unwrap()[0].code, 0x111, "the press is queued first");
    rig.input.pop(InputDevice::Tablet);
    let release = rig.input.peek(InputDevice::Tablet).unwrap();
    assert_eq!((release[0].code, release[0].value), (0x111, 0));
}

#[test]
fn status_reports_are_taken_and_completed() {
    let mut rig = Rig::new(InputDevice::Keyboard);
    put_desc(
        &mut rig.ram,
        STATUS_DESC,
        0,
        BUFFERS,
        EVENT_BYTES as u32,
        0,
        0,
    );
    post(&mut rig.ram, STATUS_AVAIL, QUEUE_SIZE, 0);
    assert!(rig.kick(STATUS_QUEUE));
    assert_eq!(used_idx(&rig.ram, STATUS_USED), 1);
}

#[test]
fn keyboard_and_tablet_queues_are_separate() {
    let mut keyboard = Rig::new(InputDevice::Keyboard);
    keyboard.input.pointer_move(1, 1);
    keyboard.post_buffers(4);
    assert!(!keyboard.poll(), "a move is not for the keyboard");
    let tablet = VirtioInput::new(InputDevice::Tablet, keyboard.input.clone());
    assert!(tablet.input.peek(InputDevice::Tablet).is_some());
}
