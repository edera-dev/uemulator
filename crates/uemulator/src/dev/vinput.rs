//! virtio-input (virtio 1.2 section 5.8): a keyboard and an absolute-pointer tablet, which is
//! what Linux's `virtio_input` turns into evdev devices (and its VT keyboard handler into
//! console input). The host side is an `InputHandle`; batches of events, each ending in
//! `SYN_REPORT`, go into buffers the driver posted on the event queue, all of a batch or none of
//! it. Configuration space answers the select/subsel queries the way QEMU's
//! `hw/input/virtio-input-hid.c` does for its keyboard and tablet.

use super::guestmem::GuestMem;
use super::virtio::{DEVICE_TYPE_INPUT, Queue, VirtioDev};
use crate::display::{
    ABS_X, ABS_Y, EV_ABS, EV_KEY, EV_REL, InputDevice, InputHandle, KEY_CODE_MAX, POINTER_AXIS_MAX,
    PointerButton, REL_WHEEL,
};
#[cfg(freestanding)]
use crate::prelude::*;

const EVENT_QUEUE: usize = 0;
/// LED states from the guest (caps lock and friends); taken and dropped.
const STATUS_QUEUE: usize = 1;
const QUEUE_COUNT: usize = 2;

/// `struct virtio_input_event`: type u16, code u16, value u32.
const EVENT_BYTES: usize = 8;

/// `struct virtio_input_config`: select u8, subsel u8, size u8, reserved[5], then a 128-byte
/// union. The driver writes select and subsel, then reads size and the union.
const CONFIG_SELECT: usize = 0;
const CONFIG_SUBSEL: usize = 1;
const CONFIG_SIZE: usize = 2;
const CONFIG_PAYLOAD: usize = 8;
const CONFIG_PAYLOAD_BYTES: usize = 128;

/// `enum virtio_input_config_select`.
const SELECT_ID_NAME: u8 = 0x01;
const SELECT_ID_SERIAL: u8 = 0x02;
const SELECT_ID_DEVIDS: u8 = 0x03;
const SELECT_PROP_BITS: u8 = 0x10;
const SELECT_EV_BITS: u8 = 0x11;
const SELECT_ABS_INFO: u8 = 0x12;

/// `EV_REP`: the guest's input core repeats held keys itself; the host sends one press.
const EV_REP: u8 = 0x14;
/// `EV_LED` with num, caps and scroll lock, so the guest tracks lock state (QEMU offers them).
const EV_LED: u8 = 0x11;
const LED_NUML: u8 = 0;
const LED_CAPSL: u8 = 1;
const LED_SCROLLL: u8 = 2;

/// `struct virtio_input_devids`: bustype, vendor, product, version, each u16. `BUS_VIRTUAL`
/// with the vendor QEMU uses for its USB and virtio HID devices, so guest quirks keyed on them
/// treat these the same.
const BUS_VIRTUAL: u16 = 0x06;
const VENDOR_ID: u16 = 0x0627;
const KEYBOARD_PRODUCT: u16 = 0x0001;
const TABLET_PRODUCT: u16 = 0x0003;
const DEVICE_VERSION: u16 = 0x0001;

const KEYBOARD_NAME: &str = "uemulator Virtio Keyboard";
const TABLET_NAME: &str = "uemulator Virtio Tablet";

pub(crate) struct VirtioInput {
    kind: InputDevice,
    input: InputHandle,
    select: u8,
    subsel: u8,
}

/// Sets bit `bit` of a bitmap stored as bytes, least significant first.
fn set_bit(bitmap: &mut [u8], bit: usize) {
    bitmap[bit / 8] |= 1 << (bit % 8);
}

impl VirtioInput {
    pub fn new(kind: InputDevice, input: InputHandle) -> Self {
        VirtioInput {
            kind,
            input,
            select: 0,
            subsel: 0,
        }
    }

    /// The union for the current select/subsel, trimmed to its size; empty for a query the
    /// device has no answer to (the driver reads size 0 and moves on).
    fn config_payload(&self) -> Vec<u8> {
        let mut bitmap = [0u8; CONFIG_PAYLOAD_BYTES];
        let used = |bitmap: &[u8]| -> Vec<u8> {
            let size = bitmap
                .iter()
                .rposition(|byte| *byte != 0)
                .map_or(0, |at| at + 1);
            bitmap[..size].to_vec()
        };
        let keyboard = self.kind == InputDevice::Keyboard;
        match (self.select, self.subsel) {
            (SELECT_ID_NAME, 0) => {
                let name = if keyboard { KEYBOARD_NAME } else { TABLET_NAME };
                name.as_bytes().to_vec()
            }
            (SELECT_ID_DEVIDS, 0) => {
                let product = if keyboard {
                    KEYBOARD_PRODUCT
                } else {
                    TABLET_PRODUCT
                };
                [BUS_VIRTUAL, VENDOR_ID, product, DEVICE_VERSION]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect()
            }
            (SELECT_EV_BITS, subsel) if keyboard => {
                match subsel as u16 {
                    EV_KEY => {
                        (1..=KEY_CODE_MAX).for_each(|code| set_bit(&mut bitmap, code as usize))
                    }
                    _ if subsel == EV_REP => return vec![1],
                    _ if subsel == EV_LED => {
                        for led in [LED_NUML, LED_CAPSL, LED_SCROLLL] {
                            set_bit(&mut bitmap, led as usize);
                        }
                    }
                    _ => {}
                }
                used(&bitmap)
            }
            (SELECT_EV_BITS, subsel) => {
                match subsel as u16 {
                    EV_KEY => PointerButton::ALL
                        .iter()
                        .for_each(|button| set_bit(&mut bitmap, button.code() as usize)),
                    EV_ABS => {
                        set_bit(&mut bitmap, ABS_X as usize);
                        set_bit(&mut bitmap, ABS_Y as usize);
                    }
                    EV_REL => set_bit(&mut bitmap, REL_WHEEL as usize),
                    _ => {}
                }
                used(&bitmap)
            }
            // `struct virtio_input_absinfo`: min, max, fuzz, flat, res.
            (SELECT_ABS_INFO, axis)
                if !keyboard && (axis as u16 == ABS_X || axis as u16 == ABS_Y) =>
            {
                [0, POINTER_AXIS_MAX, 0, 0, 0]
                    .iter()
                    .flat_map(|word: &u32| word.to_le_bytes())
                    .collect()
            }
            (SELECT_ID_SERIAL | SELECT_PROP_BITS, _) => Vec::new(),
            _ => Vec::new(),
        }
    }

    /// Delivers whole batches while the driver has posted room for them.
    fn deliver(&mut self, queue: &mut Queue, mem: &mut GuestMem) -> bool {
        let mut delivered = false;
        while let Some(batch) = self.input.peek(self.kind) {
            let mark = queue.last_avail;
            let mut chains = Vec::with_capacity(batch.len());
            while chains.len() < batch.len() {
                match queue.pop(mem) {
                    Some(chain) => chains.push(chain),
                    None => break,
                }
            }
            if chains.len() < batch.len() {
                // Not enough buffers for the whole batch: give them back and wait for more.
                queue.rewind_to(mark);
                break;
            }
            let mut completions = Vec::with_capacity(batch.len());
            for (event, chain) in batch.iter().zip(&chains) {
                let mut bytes = [0u8; EVENT_BYTES];
                bytes[..2].copy_from_slice(&event.kind.to_le_bytes());
                bytes[2..4].copy_from_slice(&event.code.to_le_bytes());
                bytes[4..].copy_from_slice(&event.value.to_le_bytes());
                let written = chain.write_at(mem, 0, &bytes);
                completions.push((chain.head, written as u32));
            }
            // One used-index update for the batch: the driver hands events to the input core as
            // it finds them, and must not see the start of a batch without its SYN.
            queue.push_used_all(mem, &completions);
            self.input.pop(self.kind);
            delivered = true;
        }
        delivered
    }
}

impl VirtioDev for VirtioInput {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_INPUT
    }

    fn features(&self) -> u64 {
        0
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, offset: usize) -> u8 {
        match offset {
            CONFIG_SELECT => self.select,
            CONFIG_SUBSEL => self.subsel,
            CONFIG_SIZE => self.config_payload().len() as u8,
            CONFIG_PAYLOAD.. => self
                .config_payload()
                .get(offset - CONFIG_PAYLOAD)
                .copied()
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn config_write(&mut self, offset: usize, value: u8) {
        match offset {
            CONFIG_SELECT => self.select = value,
            CONFIG_SUBSEL => self.subsel = value,
            _ => {}
        }
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        match queue {
            // New buffers: whatever was waiting for room goes now.
            EVENT_QUEUE => self.deliver(&mut queues[EVENT_QUEUE], mem),
            STATUS_QUEUE => {
                let mut any = false;
                while let Some(chain) = queues[STATUS_QUEUE].pop(mem) {
                    queues[STATUS_QUEUE].push_used(mem, chain.head, 0);
                    any = true;
                }
                any
            }
            _ => false,
        }
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        self.deliver(&mut queues[EVENT_QUEUE], mem)
    }

    fn reset(&mut self) {
        self.select = 0;
        self.subsel = 0;
    }
}

#[cfg(test)]
mod tests;
