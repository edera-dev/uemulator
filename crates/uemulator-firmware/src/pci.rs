//! PCI configuration access through the legacy 0xCF8/0xCFC ports, and enumeration of bus 0.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use crate::cpu::{inb, inl, inw, outb, outl, outw};
use crate::handoff::Handoff;

static CONFIG_ADDRESS_PORT: AtomicU16 = AtomicU16::new(0xcf8);
/// The ECAM window for bus 0, or 0 when the machine has none: the only way to the registers
/// past the first 256 bytes.
static ECAM_BASE: AtomicU64 = AtomicU64::new(0);
const LEGACY_CONFIG_BYTES: u32 = 0x100;
/// ECAM gives each function 4 KiB, at `device << 15 | function << 12` (PCIe 5.0 7.2.2).
pub const CONFIG_SPACE_BYTES: u32 = 0x1000;
const ECAM_DEVICE_SHIFT: u32 = 15;
const ECAM_FUNCTION_SHIFT: u32 = 12;
const FUNCTIONS_PER_DEVICE: u8 = 8;
const REGISTER_HEADER_TYPE: u8 = 0x0e;
const HEADER_MULTI_FUNCTION: u8 = 0x80;
const CONFIG_DATA_OFFSET: u16 = 4;
const ENABLE: u32 = 1 << 31;
const DEVICES_PER_BUS: u8 = 32;
const MISSING_VENDOR: u16 = 0xffff;

pub const REGISTER_VENDOR_DEVICE: u8 = 0x00;
pub const REGISTER_COMMAND: u8 = 0x04;
pub const REGISTER_BAR0: u8 = 0x10;
pub const REGISTER_CAPABILITIES: u8 = 0x34;
pub const COMMAND_MEMORY_SPACE: u32 = 1 << 1;
pub const COMMAND_BUS_MASTER: u32 = 1 << 2;

#[derive(Clone, Copy, Debug)]
pub struct Function {
    pub device: u8,
    pub function: u8,
    pub vendor: u16,
    pub device_id: u16,
}

pub fn init(handoff: &Handoff) {
    CONFIG_ADDRESS_PORT.store(handoff.pci_config_port, Ordering::Relaxed);
    ECAM_BASE.store(handoff.pci_ecam_base, Ordering::Relaxed);
}

fn address(device: u8, function: u8, register: u8) -> u32 {
    ENABLE | (device as u32) << 11 | (function as u32) << 8 | (register & 0xfc) as u32
}

pub fn read32(device: u8, function: u8, register: u8) -> u32 {
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register));
    inl(port + CONFIG_DATA_OFFSET)
}

pub fn write32(device: u8, function: u8, register: u8, value: u32) {
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register));
    outl(port + CONFIG_DATA_OFFSET, value);
}

pub fn read8(device: u8, function: u8, register: u8) -> u8 {
    (read32(device, function, register) >> (8 * (register & 3))) as u8
}

/// A `bytes`-wide (1, 2 or 4) access at a register aligned to it, through the port the access
/// lands on, so it touches only its own bytes: a read-modify-write of the dword would write back
/// write-one-to-clear status bits. Registers past 256 go through ECAM; without it they read as
/// all ones and ignore writes, like an absent function.
pub fn read(device: u8, function: u8, register: u32, bytes: usize) -> u32 {
    if register >= LEGACY_CONFIG_BYTES {
        return match ecam(device, function, register) {
            Some(address) => unsafe {
                match bytes {
                    1 => (address as *const u8).read_volatile() as u32,
                    2 => (address as *const u16).read_volatile() as u32,
                    _ => (address as *const u32).read_volatile(),
                }
            },
            None => u32::MAX >> (32 - 8 * bytes),
        };
    }
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register as u8));
    let data = port + CONFIG_DATA_OFFSET + (register & 3) as u16;
    match bytes {
        1 => inb(data) as u32,
        2 => inw(data) as u32,
        _ => inl(data),
    }
}

pub fn write(device: u8, function: u8, register: u32, bytes: usize, value: u32) {
    if register >= LEGACY_CONFIG_BYTES {
        if let Some(address) = ecam(device, function, register) {
            unsafe {
                match bytes {
                    1 => (address as *mut u8).write_volatile(value as u8),
                    2 => (address as *mut u16).write_volatile(value as u16),
                    _ => (address as *mut u32).write_volatile(value),
                }
            }
        }
        return;
    }
    let port = CONFIG_ADDRESS_PORT.load(Ordering::Relaxed);
    outl(port, address(device, function, register as u8));
    let data = port + CONFIG_DATA_OFFSET + (register & 3) as u16;
    match bytes {
        1 => outb(data, value as u8),
        2 => outw(data, value as u16),
        _ => outl(data, value),
    }
}

fn ecam(device: u8, function: u8, register: u32) -> Option<u64> {
    let base = ECAM_BASE.load(Ordering::Relaxed);
    (base != 0 && register < CONFIG_SPACE_BYTES).then(|| {
        base + ((device as u64) << ECAM_DEVICE_SHIFT)
            + ((function as u64) << ECAM_FUNCTION_SHIFT)
            + register as u64
    })
}

/// Every function on bus 0, the other functions of a multi-function device included.
pub fn scan_all_functions() -> Vec<Function> {
    let mut found = Vec::new();
    for device in 0..DEVICES_PER_BUS {
        let functions = if read8(device, 0, REGISTER_HEADER_TYPE) & HEADER_MULTI_FUNCTION != 0 {
            FUNCTIONS_PER_DEVICE
        } else {
            1
        };
        for function in 0..functions {
            let id = read32(device, function, REGISTER_VENDOR_DEVICE);
            if id as u16 != MISSING_VENDOR {
                found.push(Function {
                    device,
                    function,
                    vendor: id as u16,
                    device_id: (id >> 16) as u16,
                });
            }
        }
    }
    found
}

/// The functions on bus 0. The machine has one bus, and its virtio devices are single-function.
pub fn scan() -> Vec<Function> {
    (0..DEVICES_PER_BUS)
        .filter_map(|device| {
            let id = read32(device, 0, REGISTER_VENDOR_DEVICE);
            let vendor = id as u16;
            (vendor != MISSING_VENDOR).then_some(Function {
                device,
                function: 0,
                vendor,
                device_id: (id >> 16) as u16,
            })
        })
        .collect()
}
