//! The modern virtio-pci transport (virtio 1.2 section 4.1) as the firmware's polling drivers
//! use it: find the capability regions in BAR0, negotiate `VERSION_1`, set up split queues. The
//! device side is `crates/uemulator/src/dev/virtio.rs`, which keeps every region in BAR0.

use core::ptr::{read_volatile, write_volatile};

use crate::pci::{self, Function};

pub const VENDOR_VIRTIO: u16 = 0x1af4;
/// Modern devices are `0x1040` plus the virtio device type.
pub const DEVICE_ID_BASE: u16 = 0x1040;

const CAP_ID_VENDOR: u8 = 0x09;
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;
const CAP_TYPE: u8 = 3;
const CAP_BAR: u8 = 4;
const CAP_OFFSET: u8 = 8;
const CAP_NOTIFY_MULTIPLIER: u8 = 16;
/// The capability list is a handful of entries; this bounds a looping one.
const MAX_CAPABILITIES: usize = 16;
/// BAR0's low bits are flags (memory type, prefetchable), not address.
const BAR_FLAG_BITS: u32 = 0xf;

const COMMON_DEVICE_FEATURE_SELECT: usize = 0x00;
const COMMON_DEVICE_FEATURE: usize = 0x04;
const COMMON_DRIVER_FEATURE_SELECT: usize = 0x08;
const COMMON_DRIVER_FEATURE: usize = 0x0c;
const COMMON_DEVICE_STATUS: usize = 0x14;
const COMMON_QUEUE_SELECT: usize = 0x16;
const COMMON_QUEUE_SIZE: usize = 0x18;
const COMMON_QUEUE_ENABLE: usize = 0x1c;
const COMMON_QUEUE_NOTIFY_OFFSET: usize = 0x1e;
const COMMON_QUEUE_DESC: usize = 0x20;
const COMMON_QUEUE_DRIVER: usize = 0x28;
const COMMON_QUEUE_DEVICE: usize = 0x30;

const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_DRIVER_OK: u8 = 4;
const STATUS_FEATURES_OK: u8 = 8;
/// `VIRTIO_F_VERSION_1` is bit 32: bit 0 of feature word 1.
const FEATURES_WORD_VERSION_1: u32 = 1;

/// The mapped regions of one function.
pub struct Transport {
    common: *mut u8,
    notify: *mut u8,
    notify_multiplier: u32,
    pub device_config: *mut u8,
}

pub unsafe fn read16(base: *mut u8, offset: usize) -> u16 {
    unsafe { read_volatile(base.add(offset).cast::<u16>()) }
}

pub unsafe fn write16(base: *mut u8, offset: usize, value: u16) {
    unsafe { write_volatile(base.add(offset).cast::<u16>(), value) }
}

pub unsafe fn write32(base: *mut u8, offset: usize, value: u32) {
    unsafe { write_volatile(base.add(offset).cast::<u32>(), value) }
}

/// The queue address registers are 64 bits wide but the device decodes them as two 32-bit
/// halves, low first.
unsafe fn write64(base: *mut u8, offset: usize, value: u64) {
    unsafe {
        write32(base, offset, value as u32);
        write32(base, offset + 4, (value >> 32) as u32);
    }
}

impl Transport {
    /// Turns on memory decoding and bus mastering and finds the common, notify and device
    /// configuration regions.
    pub fn new(function: &Function) -> Option<Transport> {
        let (device, number) = (function.device, function.function);
        let command = pci::read32(device, number, pci::REGISTER_COMMAND);
        pci::write32(
            device,
            number,
            pci::REGISTER_COMMAND,
            command | pci::COMMAND_MEMORY_SPACE | pci::COMMAND_BUS_MASTER,
        );
        // The machine has no firmware to assign BARs, so the library preassigned BAR0 and every
        // region the capabilities describe lives in it.
        let bar0 = (pci::read32(device, number, pci::REGISTER_BAR0) & !BAR_FLAG_BITS) as usize;
        if bar0 == 0 {
            return None;
        }
        let (mut common, mut notify, mut device_config, mut multiplier) =
            (0usize, 0usize, 0usize, 0u32);
        let mut capability = pci::read8(device, number, pci::REGISTER_CAPABILITIES) as usize;
        for _ in 0..MAX_CAPABILITIES {
            if capability == 0 {
                break;
            }
            let register = |offset: u8| pci::read32(device, number, capability as u8 + offset);
            if pci::read8(device, number, capability as u8) == CAP_ID_VENDOR {
                let region = bar0 + register(CAP_OFFSET) as usize;
                let kind = pci::read8(device, number, capability as u8 + CAP_TYPE);
                debug_assert_eq!(pci::read8(device, number, capability as u8 + CAP_BAR), 0);
                match kind {
                    CFG_COMMON => common = region,
                    CFG_NOTIFY => {
                        notify = region;
                        multiplier = register(CAP_NOTIFY_MULTIPLIER);
                    }
                    CFG_DEVICE => device_config = region,
                    _ => {}
                }
            }
            capability = pci::read8(device, number, capability as u8 + 1) as usize;
        }
        (common != 0 && notify != 0 && device_config != 0).then_some(Transport {
            common: common as *mut u8,
            notify: notify as *mut u8,
            notify_multiplier: multiplier,
            device_config: device_config as *mut u8,
        })
    }

    /// Resets the device and agrees on `VERSION_1` plus `driver_features` (low word). Returns the
    /// device's low feature word, `None` if it is not a modern device or refuses.
    pub fn negotiate(&self, driver_features: u32) -> Option<u32> {
        let common = self.common;
        unsafe {
            write_volatile(common.add(COMMON_DEVICE_STATUS), 0);
            write_volatile(
                common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER,
            );
            write32(common, COMMON_DEVICE_FEATURE_SELECT, 0);
            let low = read_volatile(common.add(COMMON_DEVICE_FEATURE).cast::<u32>());
            write32(common, COMMON_DEVICE_FEATURE_SELECT, 1);
            let high = read_volatile(common.add(COMMON_DEVICE_FEATURE).cast::<u32>());
            if high & FEATURES_WORD_VERSION_1 == 0 {
                return None;
            }
            write32(common, COMMON_DRIVER_FEATURE_SELECT, 0);
            write32(common, COMMON_DRIVER_FEATURE, driver_features & low);
            write32(common, COMMON_DRIVER_FEATURE_SELECT, 1);
            write32(common, COMMON_DRIVER_FEATURE, FEATURES_WORD_VERSION_1);
            write_volatile(
                common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
            );
            if read_volatile(common.add(COMMON_DEVICE_STATUS)) & STATUS_FEATURES_OK == 0 {
                return None;
            }
            Some(low)
        }
    }

    /// Sets up queue `index` with `size` entries: descriptors at `ring`, the available ring at
    /// `ring + available_offset`, the used ring at `ring + used_offset`. Returns the queue's
    /// notify address.
    pub fn setup_queue(
        &self,
        index: u16,
        size: u16,
        ring: *mut u8,
        available_offset: usize,
        used_offset: usize,
    ) -> Option<*mut u8> {
        let common = self.common;
        unsafe {
            write16(common, COMMON_QUEUE_SELECT, index);
            if read16(common, COMMON_QUEUE_SIZE) < size {
                return None;
            }
            write16(common, COMMON_QUEUE_SIZE, size);
            let base = ring as u64;
            write64(common, COMMON_QUEUE_DESC, base);
            write64(common, COMMON_QUEUE_DRIVER, base + available_offset as u64);
            write64(common, COMMON_QUEUE_DEVICE, base + used_offset as u64);
            let notify_offset = read16(common, COMMON_QUEUE_NOTIFY_OFFSET) as usize;
            write16(common, COMMON_QUEUE_ENABLE, 1);
            Some(
                self.notify
                    .add(notify_offset * self.notify_multiplier as usize),
            )
        }
    }

    pub fn driver_ok(&self) {
        unsafe {
            write_volatile(
                self.common.add(COMMON_DEVICE_STATUS),
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK,
            )
        }
    }

    /// Stops the device, so the OS finds it as it would after power-on.
    pub fn reset(&self) {
        unsafe { write_volatile(self.common.add(COMMON_DEVICE_STATUS), 0) }
    }
}
