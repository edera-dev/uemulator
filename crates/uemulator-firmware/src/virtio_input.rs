//! A polling driver for virtio-input keyboards: it keeps the event queue stocked with buffers
//! and turns `EV_KEY` events into UEFI keystrokes (`keymap.rs`) for the text console, so the
//! boot menu and GRUB answer the graphical console's keyboard as well as the serial line. Tablets
//! (absolute pointers) are left alone. The device model is `crates/uemulator/src/dev/vinput.rs`.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

use crate::keymap::{Keyboard, Typematic};
use crate::memory;
use crate::pci;
use crate::sync::Global;
use crate::time;
use crate::uefi::InputKey;
use crate::uefi::memory_type::BOOT_SERVICES_DATA;
use crate::virtio_pci::{DEVICE_ID_BASE, Transport, VENDOR_VIRTIO, write16};

const DEVICE_ID_INPUT: u16 = DEVICE_ID_BASE + 18;

/// `struct virtio_input_config`: select, subsel, size, reserved, then the 128-byte union.
const CONFIG_SELECT: usize = 0;
const CONFIG_SUBSEL: usize = 1;
const CONFIG_SIZE: usize = 2;
const CONFIG_UNION: usize = 8;
const SELECT_EV_BITS: u8 = 0x11;
const EV_KEY: u16 = 0x01;
/// A device that reports `KEY_A` is a keyboard; a tablet reports only buttons.
const KEY_A: usize = 30;
const KEY_RELEASED: u32 = 0;

const EVENT_QUEUE: u16 = 0;
const QUEUE_SIZE: u16 = 64;
const DESC_BYTES: usize = 16;
const DESC_F_WRITE: u16 = 2;
/// `struct virtio_input_event`: type u16, code u16, value u32.
const EVENT_BYTES: usize = 8;
/// Descriptors, then the available ring; the used ring on the next page; the event buffers on
/// the third.
const RING_PAGES: u64 = 3;
const AVAIL_OFFSET: usize = DESC_BYTES * QUEUE_SIZE as usize;
const USED_OFFSET: usize = 4096;
const EVENTS_OFFSET: usize = 8192;
const RING_HEADER_BYTES: usize = 4;
const USED_ENTRY_BYTES: usize = 8;
/// Keystrokes decoded but not yet read; typing faster than anything reads is dropped beyond it.
const MAX_PENDING_KEYS: usize = 64;

struct VirtioKeyboard {
    transport: Transport,
    notify: *mut u8,
    ring: *mut u8,
    /// Free-running count of used entries taken.
    last_used: u16,
    /// Free-running available index.
    available: u16,
    keyboard: Keyboard,
    typematic: Typematic,
}

static KEYBOARDS: Global<Vec<VirtioKeyboard>> = Global::new(Vec::new());
static PENDING: Global<VecDeque<InputKey>> = Global::new(VecDeque::new());

/// Whether the device's `EV_KEY` bitmap has `KEY_A`.
fn is_keyboard(transport: &Transport) -> bool {
    let config = transport.device_config;
    unsafe {
        write_volatile(config.add(CONFIG_SELECT), SELECT_EV_BITS);
        write_volatile(config.add(CONFIG_SUBSEL), EV_KEY as u8);
        let size = read_volatile(config.add(CONFIG_SIZE)) as usize;
        if size <= KEY_A / 8 {
            return false;
        }
        read_volatile(config.add(CONFIG_UNION + KEY_A / 8)) & (1 << (KEY_A % 8)) != 0
    }
}

impl VirtioKeyboard {
    fn new(function: &pci::Function) -> Option<VirtioKeyboard> {
        let transport = Transport::new(function)?;
        transport.negotiate(0)?;
        if !is_keyboard(&transport) {
            transport.reset();
            return None;
        }
        let ring = memory::allocate_zeroed_pages(BOOT_SERVICES_DATA, RING_PAGES)?;
        let notify =
            transport.setup_queue(EVENT_QUEUE, QUEUE_SIZE, ring, AVAIL_OFFSET, USED_OFFSET)?;
        let mut keyboard = VirtioKeyboard {
            transport,
            notify,
            ring,
            last_used: 0,
            available: 0,
            keyboard: Keyboard::default(),
            typematic: Typematic::default(),
        };
        unsafe {
            for index in 0..QUEUE_SIZE {
                let descriptor = ring.add(index as usize * DESC_BYTES);
                let buffer = ring.add(EVENTS_OFFSET + index as usize * EVENT_BYTES);
                descriptor.cast::<u64>().write(buffer as u64);
                descriptor.add(8).cast::<u32>().write(EVENT_BYTES as u32);
                descriptor.add(12).cast::<u16>().write(DESC_F_WRITE);
                keyboard.post(index);
            }
        }
        keyboard.transport.driver_ok();
        keyboard.kick();
        Some(keyboard)
    }

    /// Hands descriptor `index` (and its buffer) to the device.
    unsafe fn post(&mut self, index: u16) {
        unsafe {
            let avail = self.ring.add(AVAIL_OFFSET);
            let slot = (self.available % QUEUE_SIZE) as usize;
            avail
                .add(RING_HEADER_BYTES + 2 * slot)
                .cast::<u16>()
                .write(index);
            self.available = self.available.wrapping_add(1);
            write_volatile(avail.add(2).cast::<u16>(), self.available);
        }
    }

    fn kick(&self) {
        unsafe { write16(self.notify, 0, EVENT_QUEUE) };
    }

    /// Takes every event the device completed, queues the keystrokes they type and gives the
    /// buffers back.
    fn drain(&mut self, pending: &mut VecDeque<InputKey>) {
        let mut reposted = false;
        unsafe {
            let used = self.ring.add(USED_OFFSET);
            while read_volatile(used.add(2).cast::<u16>()) != self.last_used {
                let slot = (self.last_used % QUEUE_SIZE) as usize;
                let entry = used.add(RING_HEADER_BYTES + slot * USED_ENTRY_BYTES);
                let index = (entry.cast::<u32>().read() as u16) % QUEUE_SIZE;
                let event = self.ring.add(EVENTS_OFFSET + index as usize * EVENT_BYTES);
                let kind = event.cast::<u16>().read();
                let code = event.add(2).cast::<u16>().read();
                let value = event.add(4).cast::<u32>().read();
                self.last_used = self.last_used.wrapping_add(1);
                if kind == EV_KEY {
                    let pressed = value != KEY_RELEASED;
                    let typed = self.keyboard.event(code, pressed);
                    self.typematic.event(code, pressed, typed, time::now_ns());
                    if let Some(key) = typed
                        && pending.len() < MAX_PENDING_KEYS
                    {
                        pending.push_back(key);
                    }
                }
                self.post(index);
                reposted = true;
            }
        }
        if reposted {
            self.kick();
        }
    }
}

/// Finds the virtio keyboards and starts their event queues.
pub fn init() {
    let keyboards = KEYBOARDS.get();
    for function in pci::scan() {
        if function.vendor == VENDOR_VIRTIO
            && function.device_id == DEVICE_ID_INPUT
            && let Some(keyboard) = VirtioKeyboard::new(&function)
        {
            keyboards.push(keyboard);
        }
    }
}

/// The next keystroke typed on any virtio keyboard, or repeated by one held down. Repeats come
/// only once everything typed has been read, so they never get ahead of real keystrokes.
pub fn read_key() -> Option<InputKey> {
    let pending = PENDING.get();
    if pending.is_empty() {
        let keyboards = KEYBOARDS.get();
        for keyboard in keyboards.iter_mut() {
            keyboard.drain(pending);
        }
        if pending.is_empty() {
            let now = time::now_ns();
            pending.extend(
                keyboards
                    .iter_mut()
                    .filter_map(|keyboard| keyboard.typematic.due(now)),
            );
        }
    }
    pending.pop_front()
}

/// Stops the keyboards before the OS takes over: their rings are boot services memory.
pub fn shutdown() {
    for keyboard in KEYBOARDS.get().drain(..) {
        keyboard.transport.reset();
    }
}
