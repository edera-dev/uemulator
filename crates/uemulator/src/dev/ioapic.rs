//! I/O APIC: 24 pins, each with a redirection entry that turns a device line into an interrupt
//! message for the local APIC. Programming model per the 82093AA datasheet (as extended with the
//! EOI register of version 0x20); QEMU's `hw/intc/ioapic.c` is the behavioral reference.

#[cfg(freestanding)]
use crate::prelude::*;
use crate::snapshot::Snapshot;

use super::apic::{DELIVERY_EXTINT, Interrupt};

pub(crate) const IOAPIC_DEFAULT_BASE: u64 = 0xfec0_0000;
pub(crate) const IOAPIC_MMIO_SIZE: u64 = 0x1000;
pub(crate) const IOAPIC_PINS: usize = 24;
/// The id the MADT and IVRS give the I/O APIC (QEMU numbers it after the CPUs; one CPU here).
pub(crate) const IOAPIC_ID: u8 = 1;

const REGISTER_SELECT: u64 = 0x00;
const REGISTER_WINDOW: u64 = 0x10;
const REGISTER_EOI: u64 = 0x40;

const INDIRECT_ID: u32 = 0;
const INDIRECT_VERSION: u32 = 1;
const INDIRECT_ARBITRATION: u32 = 2;
const INDIRECT_REDIRECTION_FIRST: u32 = 0x10;
const INDIRECT_REDIRECTION_LAST: u32 = INDIRECT_REDIRECTION_FIRST + 2 * IOAPIC_PINS as u32 - 1;
/// Version 0x20 (the first with the EOI register); the maximum redirection entry index sits in
/// bits 23:16.
const VERSION: u32 = 0x20 | ((IOAPIC_PINS as u32 - 1) << 16);
const ID_WRITABLE_SHIFT: u32 = 24;
const ID_MASK: u32 = 0x0f;

const RTE_VECTOR_MASK: u64 = 0xff;
const RTE_DELIVERY_SHIFT: u32 = 8;
const RTE_DELIVERY_MASK: u64 = 0b111;
const RTE_LOGICAL: u64 = 1 << 11;
/// Delivery status is read-only and always idle: messages are delivered synchronously.
const RTE_REMOTE_IRR: u64 = 1 << 14;
const RTE_LEVEL: u64 = 1 << 15;
const RTE_MASKED: u64 = 1 << 16;
const RTE_DESTINATION_SHIFT: u32 = 56;
/// What the guest can write: everything except delivery status (12) and remote IRR (14). Bits
/// 55:17 are reserved.
const RTE_WRITABLE: u64 = RTE_VECTOR_MASK
    | (RTE_DELIVERY_MASK << RTE_DELIVERY_SHIFT)
    | RTE_LOGICAL
    | (1 << 13)
    | RTE_LEVEL
    | RTE_MASKED
    | (0xff << RTE_DESTINATION_SHIFT);

#[derive(Snapshot)]
pub(crate) struct IoApic {
    id: u32,
    select: u32,
    redirection: [u64; IOAPIC_PINS],
    /// Device-side line levels, for level-triggered pins.
    line: u32,
    /// Latched edge requests (a pulse while the pin was masked) and unserviced level requests.
    request: u32,
    /// Pins of the messages `service` produced, in order, for the caller that routes them.
    sent_pins: Vec<usize>,
    /// The vector the local APIC actually received for a level-triggered pin. With interrupt
    /// remapping the redirection entry holds an IRTE index instead, but the EOI carries the real
    /// vector (`note_delivered`).
    delivered_vector: [Option<u8>; IOAPIC_PINS],
}

impl IoApic {
    pub fn new() -> Self {
        IoApic {
            id: IOAPIC_ID as u32,
            select: 0,
            // Every pin starts masked, edge triggered, vector 0.
            redirection: [RTE_MASKED; IOAPIC_PINS],
            line: 0,
            request: 0,
            sent_pins: Vec::new(),
            delivered_vector: [None; IOAPIC_PINS],
        }
    }

    /// Pins of the messages produced since the last call, one per entry pushed to `out`.
    pub fn take_sent_pins(&mut self) -> Vec<usize> {
        core::mem::take(&mut self.sent_pins)
    }

    /// Records the vector the LAPIC received for `pin` after the IOMMU remapped it.
    pub fn note_delivered(&mut self, pin: usize, vector: u8) {
        self.delivered_vector[pin] = Some(vector);
    }

    pub fn decodes(&self, phys: u64) -> bool {
        (IOAPIC_DEFAULT_BASE..IOAPIC_DEFAULT_BASE + IOAPIC_MMIO_SIZE).contains(&phys)
    }

    fn interrupt_for(&self, pin: usize) -> Interrupt {
        let entry = self.redirection[pin];
        Interrupt {
            vector: entry as u8,
            delivery_mode: ((entry >> RTE_DELIVERY_SHIFT) & RTE_DELIVERY_MASK) as u32,
            level_triggered: entry & RTE_LEVEL != 0,
            destination: (entry >> RTE_DESTINATION_SHIFT) as u32,
            logical: entry & RTE_LOGICAL != 0,
        }
    }

    /// Sends every request that is both pending and deliverable. A level-triggered pin sets
    /// remote IRR when it sends and does not send again until the EOI clears it.
    fn service(&mut self, out: &mut Vec<Interrupt>) {
        for pin in 0..IOAPIC_PINS {
            let bit = 1u32 << pin;
            if self.request & bit == 0 {
                continue;
            }
            let entry = self.redirection[pin];
            if entry & RTE_MASKED != 0 {
                continue;
            }
            if entry & RTE_LEVEL != 0 {
                if entry & RTE_REMOTE_IRR != 0 {
                    continue;
                }
                self.redirection[pin] |= RTE_REMOTE_IRR;
            }
            // A level request is re-armed by the EOI while the line is still high.
            self.request &= !bit;
            let interrupt = self.interrupt_for(pin);
            // ExtINT is the PIC's job; the I/O APIC only forwards the cycle, which a single
            // CPU with the PIC connected through LINT0 already gets.
            if interrupt.delivery_mode != DELIVERY_EXTINT {
                out.push(interrupt);
                self.sent_pins.push(pin);
            }
        }
    }

    /// A device line changed. An edge-triggered pin treats every rise as one event (a pulse
    /// from a device that only ever "raises" counts too); a level-triggered pin follows the line.
    pub fn set_line(&mut self, pin: usize, level: bool, out: &mut Vec<Interrupt>) {
        if pin >= IOAPIC_PINS {
            return;
        }
        let bit = 1u32 << pin;
        if level {
            self.line |= bit;
            self.request |= bit;
        } else {
            self.line &= !bit;
            // A level request is withdrawn with the line; an edge pulse stays latched.
            if self.redirection[pin] & RTE_LEVEL != 0 {
                self.request &= !bit;
            }
        }
        self.service(out);
    }

    /// The local APIC finished a level-triggered interrupt with this vector: clear remote IRR on
    /// the pins that sent it, and send again for those whose line is still asserted.
    pub fn eoi(&mut self, vector: u8, out: &mut Vec<Interrupt>) {
        for pin in 0..IOAPIC_PINS {
            let entry = self.redirection[pin];
            let sent_vector = self.delivered_vector[pin].unwrap_or(entry as u8);
            if entry & RTE_LEVEL != 0
                && entry & RTE_REMOTE_IRR != 0
                && (entry & RTE_VECTOR_MASK == vector as u64 || sent_vector == vector)
            {
                self.redirection[pin] &= !RTE_REMOTE_IRR;
                if self.line & (1 << pin) != 0 {
                    self.request |= 1 << pin;
                }
            }
        }
        self.service(out);
    }

    pub fn mmio_read(&mut self, phys: u64, _size: u8) -> u64 {
        match phys - IOAPIC_DEFAULT_BASE {
            REGISTER_SELECT => self.select as u64,
            REGISTER_WINDOW => self.read_indirect() as u64,
            _ => 0,
        }
    }

    pub fn mmio_write(&mut self, phys: u64, value: u64, out: &mut Vec<Interrupt>) {
        match phys - IOAPIC_DEFAULT_BASE {
            REGISTER_SELECT => self.select = value as u8 as u32,
            REGISTER_WINDOW => self.write_indirect(value as u32, out),
            // Version 0x20 EOI register: the written vector is the one that finished.
            REGISTER_EOI => self.eoi(value as u8, out),
            _ => {}
        }
    }

    fn read_indirect(&self) -> u32 {
        match self.select {
            INDIRECT_ID => self.id << ID_WRITABLE_SHIFT,
            INDIRECT_VERSION => VERSION,
            INDIRECT_ARBITRATION => self.id << ID_WRITABLE_SHIFT,
            select @ INDIRECT_REDIRECTION_FIRST..=INDIRECT_REDIRECTION_LAST => {
                let pin = ((select - INDIRECT_REDIRECTION_FIRST) / 2) as usize;
                let entry = self.redirection[pin];
                if select & 1 == 0 {
                    entry as u32
                } else {
                    (entry >> 32) as u32
                }
            }
            _ => 0,
        }
    }

    fn write_indirect(&mut self, value: u32, out: &mut Vec<Interrupt>) {
        match self.select {
            INDIRECT_ID => self.id = (value >> ID_WRITABLE_SHIFT) & ID_MASK,
            select @ INDIRECT_REDIRECTION_FIRST..=INDIRECT_REDIRECTION_LAST => {
                let pin = ((select - INDIRECT_REDIRECTION_FIRST) / 2) as usize;
                let entry = self.redirection[pin];
                let merged = if select & 1 == 0 {
                    (entry & !0xffff_ffff) | value as u64
                } else {
                    (entry & 0xffff_ffff) | ((value as u64) << 32)
                };
                // Delivery status and remote IRR survive a write; the rest is what the guest wrote.
                let kept = entry & !RTE_WRITABLE;
                let mut new = (merged & RTE_WRITABLE) | kept;
                // Switching a pin to edge mode clears a stuck remote IRR (datasheet note, and what
                // QEMU's `ioapic_fix_edge_remote_irr` does).
                if new & RTE_LEVEL == 0 {
                    new &= !RTE_REMOTE_IRR;
                }
                self.redirection[pin] = new;
                // Unmasking, or becoming level triggered while the line is high, can release a
                // request that was waiting.
                if new & RTE_LEVEL != 0 {
                    // A level pin's request is its line, not whatever edge was latched earlier.
                    if self.line & (1 << pin) != 0 {
                        self.request |= 1 << pin;
                    } else {
                        self.request &= !(1 << pin);
                    }
                }
                self.service(out);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_register(ioapic: &mut IoApic, select: u32, value: u32, out: &mut Vec<Interrupt>) {
        ioapic.mmio_write(IOAPIC_DEFAULT_BASE + REGISTER_SELECT, select as u64, out);
        ioapic.mmio_write(IOAPIC_DEFAULT_BASE + REGISTER_WINDOW, value as u64, out);
    }

    fn read_register(ioapic: &mut IoApic, select: u32) -> u32 {
        let mut out = Vec::new();
        ioapic.mmio_write(
            IOAPIC_DEFAULT_BASE + REGISTER_SELECT,
            select as u64,
            &mut out,
        );
        ioapic.mmio_read(IOAPIC_DEFAULT_BASE + REGISTER_WINDOW, 4) as u32
    }

    fn program(ioapic: &mut IoApic, pin: usize, low: u32, out: &mut Vec<Interrupt>) {
        let base = INDIRECT_REDIRECTION_FIRST + 2 * pin as u32;
        write_register(ioapic, base + 1, 0, out);
        write_register(ioapic, base, low, out);
    }

    #[test]
    fn version_register_reports_24_pins() {
        let mut ioapic = IoApic::new();
        assert_eq!(read_register(&mut ioapic, INDIRECT_VERSION), 0x0017_0020);
    }

    #[test]
    fn every_pin_starts_masked() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        ioapic.set_line(5, true, &mut out);
        assert!(out.is_empty());
        for pin in 0..IOAPIC_PINS as u32 {
            let low = read_register(&mut ioapic, INDIRECT_REDIRECTION_FIRST + 2 * pin);
            assert_eq!(low as u64 & RTE_MASKED, RTE_MASKED);
        }
    }

    #[test]
    fn edge_pulse_delivers_once_per_rise() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        program(&mut ioapic, 2, 0x30, &mut out);
        ioapic.set_line(2, true, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vector, 0x30);
        assert!(!out[0].level_triggered);
        // A device that only raises (never lowers) still produces one event per raise.
        ioapic.set_line(2, true, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn edge_latched_while_masked_is_delivered_on_unmask() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        ioapic.set_line(3, true, &mut out);
        assert!(out.is_empty());
        program(&mut ioapic, 3, 0x31, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vector, 0x31);
    }

    #[test]
    fn level_pin_sends_once_until_eoi_then_again_if_still_high() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        program(&mut ioapic, 9, 0x40 | RTE_LEVEL as u32, &mut out);
        ioapic.set_line(9, true, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].level_triggered);
        ioapic.set_line(9, true, &mut out);
        assert_eq!(out.len(), 1);
        ioapic.eoi(0x40, &mut out);
        assert_eq!(out.len(), 2);
        ioapic.set_line(9, false, &mut out);
        ioapic.eoi(0x40, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn level_pin_withdrawn_before_unmask_never_sends() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        ioapic.set_line(9, true, &mut out);
        ioapic.set_line(9, false, &mut out);
        program(&mut ioapic, 9, 0x40 | RTE_LEVEL as u32, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn eoi_register_clears_remote_irr_by_vector() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        program(&mut ioapic, 9, 0x40 | RTE_LEVEL as u32, &mut out);
        ioapic.set_line(9, true, &mut out);
        let low = read_register(&mut ioapic, INDIRECT_REDIRECTION_FIRST + 18);
        assert_ne!(low as u64 & RTE_REMOTE_IRR, 0);
        ioapic.set_line(9, false, &mut out);
        ioapic.mmio_write(IOAPIC_DEFAULT_BASE + REGISTER_EOI, 0x40, &mut out);
        let low = read_register(&mut ioapic, INDIRECT_REDIRECTION_FIRST + 18);
        assert_eq!(low as u64 & RTE_REMOTE_IRR, 0);
    }

    #[test]
    fn destination_and_modes_come_from_the_entry() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        let base = INDIRECT_REDIRECTION_FIRST + 2 * 4;
        write_register(&mut ioapic, base + 1, 0x03 << 24, &mut out);
        write_register(&mut ioapic, base, 0x55 | (1 << 8) | (1 << 11), &mut out);
        ioapic.set_line(4, true, &mut out);
        assert_eq!(out[0].vector, 0x55);
        assert_eq!(out[0].delivery_mode, 1);
        assert_eq!(out[0].destination, 3);
        assert!(out[0].logical);
    }

    #[test]
    fn reserved_and_status_bits_are_not_writable() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        program(&mut ioapic, 0, 0xffff_ffff, &mut out);
        let low = read_register(&mut ioapic, INDIRECT_REDIRECTION_FIRST) as u64;
        assert_eq!(low & (1 << 12), 0);
        assert_eq!(low & RTE_REMOTE_IRR, 0);
    }

    #[test]
    fn id_register_keeps_four_bits() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        write_register(&mut ioapic, INDIRECT_ID, 0xff << 24, &mut out);
        assert_eq!(read_register(&mut ioapic, INDIRECT_ID), 0x0f << 24);
    }

    #[test]
    fn extint_entries_are_not_forwarded_as_messages() {
        let mut ioapic = IoApic::new();
        let mut out = Vec::new();
        program(&mut ioapic, 1, 7 << 8, &mut out);
        ioapic.set_line(1, true, &mut out);
        assert!(out.is_empty());
    }
}
