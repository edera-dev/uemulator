use super::*;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;
const SHORT_TIMEOUT: Duration = Duration::from_millis(5);
const NEVER_EXPIRES: Duration = Duration::from_secs(30);

fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
    Rect {
        x,
        y,
        width,
        height,
    }
}

/// Paints `rect` with `value` and reports it as the change.
fn paint(display: &DisplayHandle, area: Rect, value: u8) {
    let (width, height) = display.size();
    display.publish(width, height, |pixels, stride| {
        for row in area.y..area.y + area.height {
            let start = row as usize * stride + area.x as usize * BYTES_PER_PIXEL;
            pixels[start..start + area.width as usize * BYTES_PER_PIXEL].fill(value);
        }
        Some(area)
    });
}

#[test]
fn a_new_display_is_black_at_its_size_with_generation_zero() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    assert_eq!(display.size(), (WIDTH, HEIGHT));
    assert_eq!(display.generation(), 0);
    let frame = display.frame();
    assert_eq!(
        frame.pixels,
        vec![0; (WIDTH * HEIGHT) as usize * BYTES_PER_PIXEL]
    );
    assert_eq!(display.update_since(0, &mut Vec::new()), None);
}

#[test]
fn an_update_carries_the_changed_rectangle_and_its_pixels() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    paint(&display, rect(2, 1, 3, 2), 7);
    let mut pixels = vec![1, 2, 3];
    let update = display.update_since(0, &mut pixels).unwrap();
    assert_eq!(update.generation, 1);
    assert_eq!((update.width, update.height), (WIDTH, HEIGHT));
    assert_eq!(update.rect, rect(2, 1, 3, 2));
    assert_eq!(pixels, vec![7; 3 * 2 * BYTES_PER_PIXEL]);
    assert_eq!(display.update_since(1, &mut pixels), None, "seen already");
}

#[test]
fn several_changes_add_up_to_the_rectangle_holding_them() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    paint(&display, rect(0, 0, 1, 1), 1);
    paint(&display, rect(6, 3, 2, 1), 2);
    paint(&display, rect(3, 2, 1, 1), 3);
    let update = display.update_since(1, &mut Vec::new()).unwrap();
    assert_eq!(
        update.rect,
        rect(3, 2, 5, 2),
        "everything after generation 1"
    );
    assert_eq!(
        display.update_since(0, &mut Vec::new()).unwrap().rect,
        rect(0, 0, 8, 4)
    );
}

#[test]
fn a_frontend_further_behind_than_the_history_gets_the_whole_screen() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    for _ in 0..DAMAGE_HISTORY + 5 {
        paint(&display, rect(0, 0, 1, 1), 1);
    }
    let behind = display.update_since(2, &mut Vec::new()).unwrap();
    assert_eq!(behind.rect, rect(0, 0, WIDTH, HEIGHT));
    let recent = display.generation() - 1;
    assert_eq!(
        display.update_since(recent, &mut Vec::new()).unwrap().rect,
        rect(0, 0, 1, 1)
    );
}

#[test]
fn a_size_change_clears_the_screen_and_reports_all_of_it() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    paint(&display, rect(0, 0, WIDTH, HEIGHT), 9);
    let before = display.generation();
    display.publish(4, 2, |_, _| None);
    assert_eq!(display.size(), (4, 2));
    assert_eq!(display.generation(), before + 1);
    let mut pixels = Vec::new();
    let update = display.update_since(before, &mut pixels).unwrap();
    assert_eq!(update.rect, rect(0, 0, 4, 2));
    assert_eq!(pixels, vec![0; 4 * 2 * BYTES_PER_PIXEL]);
}

#[test]
fn painting_nothing_is_no_change() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    display.publish(WIDTH, HEIGHT, |_, _| None);
    display.publish(WIDTH, HEIGHT, |_, _| Some(Rect::default()));
    assert_eq!(display.generation(), 0);
}

#[test]
fn damage_is_clipped_to_the_screen() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    display.publish(WIDTH, HEIGHT, |_, _| Some(rect(6, 2, 100, 100)));
    let update = display.update_since(0, &mut Vec::new()).unwrap();
    assert_eq!(update.rect, rect(6, 2, 2, 2));
}

#[test]
fn rect_union_ignores_empty_rectangles() {
    let a = rect(1, 1, 2, 2);
    assert_eq!(a.union(&Rect::default()), a);
    assert_eq!(Rect::default().union(&a), a);
    assert_eq!(a.union(&rect(5, 0, 1, 1)), rect(1, 0, 5, 3));
}

#[test]
fn update_callbacks_get_each_new_generation() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    let seen = Arc::new(AtomicU64::new(0));
    let record = seen.clone();
    display.on_update(move |generation| record.store(generation, Ordering::SeqCst));
    paint(&display, rect(0, 0, 1, 1), 1);
    paint(&display, rect(0, 0, 1, 1), 2);
    assert_eq!(seen.load(Ordering::SeqCst), 2);
}

#[test]
fn waiting_returns_at_once_when_there_is_news_and_after_the_timeout_otherwise() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    paint(&display, rect(0, 0, 1, 1), 1);
    assert_eq!(display.wait_for_update(0, NEVER_EXPIRES), 1);
    let started = Instant::now();
    assert_eq!(display.wait_for_update(1, SHORT_TIMEOUT), 1);
    assert!(started.elapsed() >= SHORT_TIMEOUT);
}

#[test]
fn a_waiting_frontend_wakes_when_the_screen_changes() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    let painter = display.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        paint(&painter, rect(0, 0, 1, 1), 1);
    });
    let started = Instant::now();
    let mut generation = 0;
    while generation == 0 && started.elapsed() < NEVER_EXPIRES {
        generation = display.wait_for_update(0, NEVER_EXPIRES);
    }
    assert_eq!(generation, 1);
    assert!(started.elapsed() < NEVER_EXPIRES);
    thread.join().unwrap();
}

#[test]
fn a_requested_size_is_clamped_and_taken_once() {
    let display = DisplayHandle::with_size(WIDTH, HEIGHT);
    display.request_size(10_000, 1);
    assert_eq!(display.take_requested_size(), Some((MAX_WIDTH, MIN_HEIGHT)));
    assert_eq!(display.take_requested_size(), None);
}

#[test]
fn usb_usages_map_to_the_linux_keys_of_a_usb_keyboard() {
    let cases = [
        (0x04, 30),  // a
        (0x14, 16),  // q
        (0x1d, 44),  // z
        (0x1e, 2),   // 1
        (0x27, 11),  // 0
        (0x28, 28),  // enter
        (0x29, 1),   // escape
        (0x2c, 57),  // space
        (0x35, 41),  // grave
        (0x39, 58),  // caps lock
        (0x3a, 59),  // F1
        (0x45, 88),  // F12
        (0x4c, 111), // delete
        (0x4f, 106), // right
        (0x52, 103), // up
        (0x58, 96),  // keypad enter
        (0x62, 82),  // keypad 0
        (0x63, 83),  // keypad .
        (0x64, 86),  // 102nd
        (0x68, 183), // F13
        (0x73, 194), // F24
        (0xe0, 29),  // left ctrl
        (0xe1, 42),  // left shift
        (0xe7, 126), // right meta
        (0x80, 115), // volume up
    ];
    for (usage, code) in cases {
        assert_eq!(
            linux_key_for_usb_usage(usage),
            Some(code),
            "usage {usage:#x}"
        );
    }
    for usage in [0x00, 0x01, 0x03, 0xa5, 0xe8, 0xffff] {
        assert_eq!(linux_key_for_usb_usage(usage), None, "usage {usage:#x}");
    }
}

#[test]
fn every_mapped_usage_lands_on_a_key_the_keyboard_advertises() {
    for usage in 0..=0xffu16 {
        if let Some(code) = linux_key_for_usb_usage(usage) {
            assert!(
                (1..=KEY_CODE_MAX).contains(&code),
                "usage {usage:#x} -> {code}"
            );
        }
    }
}

#[test]
fn an_input_handle_without_a_display_scales_against_its_pointer_area() {
    let input = InputHandle::without_display(640, 480);
    assert_eq!(input.pointer_area(), (640, 480));
    assert!(input.pointer_move(639, 240));
    let batch = input.peek(InputDevice::Tablet).unwrap();
    assert_eq!(batch[0].value, POINTER_AXIS_MAX);
    assert_eq!(batch[1].value, 240 * POINTER_AXIS_MAX / 479);
}

#[test]
fn an_input_handle_with_a_display_follows_the_screen_size() {
    let display = DisplayHandle::with_size(800, 600);
    let input = InputHandle::for_display(display.clone());
    assert_eq!(input.pointer_area(), (800, 600));
    display.publish(1024, 768, |_, _| None);
    assert_eq!(input.pointer_area(), (1024, 768));
}

#[test]
fn a_cursor_change_bumps_its_own_counter_wakes_waiters_and_calls_back() {
    let display = DisplayHandle::with_size(64, 64);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    display.on_update(move |generation| seen.lock().unwrap().push(generation));
    assert_eq!(display.cursor(), (0, Cursor::Host));
    display.set_cursor(Cursor::Hidden);
    assert_eq!(display.cursor(), (1, Cursor::Hidden));
    assert_eq!(display.generation(), 0, "the screen did not change");
    display.set_cursor(Cursor::Hidden);
    assert_eq!(display.cursor().0, 1, "the same cursor is no change");
    assert_eq!(*calls.lock().unwrap(), vec![0]);
}
