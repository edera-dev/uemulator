//! ACPI fixed hardware: the PM1a event, control and timer blocks the FADT points at (ACPI 6.x
//! section 4.8). Only what a guest without SMM needs: SCI_EN is hardwired (the FADT has no
//! SMI_CMD, so ACPI starts enabled), SLP_EN with the `\_S5` sleep type powers the machine off,
//! and the PM timer runs on virtual time.
use crate::snapshot::Snapshot;

/// PM1a_EVT_BLK: status at +0, enable at +2.
pub(crate) const PM1_EVENT_BASE: u16 = 0x600;
/// PM1a_CNT_BLK, 2 bytes.
pub(crate) const PM1_CONTROL_BASE: u16 = 0x604;
/// PM_TMR_BLK, 4 bytes.
pub(crate) const PM_TIMER_BASE: u16 = 0x608;
pub(crate) const PM_LAST_PORT: u16 = PM_TIMER_BASE + 3;
pub(crate) const PM1_EVENT_LENGTH: u8 = 4;
pub(crate) const PM1_CONTROL_LENGTH: u8 = 2;
pub(crate) const PM_TIMER_LENGTH: u8 = 4;
/// The SCI is ISA IRQ 12 (a PS/2 mouse line, which this machine does not have); the MADT
/// overrides it to level triggered. It is not shared with a PCI function's line: Linux registers
/// the SCI as level/active-high and refuses (EBUSY) a PCI function that then asks for the same
/// legacy GSI as level/active-low, which is what a `_PRT` entry without a link device means.
pub(crate) const SCI_IRQ: u8 = 12;
/// What `\_S5` returns as SLP_TYPa. Any value but 0 (S0) works; the guest writes it back.
pub(crate) const SLEEP_TYPE_S5: u16 = 5;

pub(crate) const PM_TIMER_HZ: u64 = 3_579_545;
const NS_PER_SECOND: u128 = 1_000_000_000;
/// The FADT does not set TMR_VAL_EXT, so the counter is 24 bits and TMR_STS follows bit 23.
const TIMER_BITS: u32 = 24;
const TIMER_MASK: u64 = (1 << TIMER_BITS) - 1;
const TIMER_OVERFLOW_BIT: u32 = TIMER_BITS - 1;

const STATUS_TIMER: u16 = 1 << 0;
const STATUS_GLOBAL: u16 = 1 << 5;
const STATUS_POWER_BUTTON: u16 = 1 << 8;
const STATUS_SLEEP_BUTTON: u16 = 1 << 9;
const STATUS_RTC: u16 = 1 << 10;
const STATUS_WAKE: u16 = 1 << 15;
/// Status bits that exist; the rest read zero and ignore the write-one-to-clear.
const STATUS_IMPLEMENTED: u16 = STATUS_TIMER
    | STATUS_GLOBAL
    | STATUS_POWER_BUTTON
    | STATUS_SLEEP_BUTTON
    | STATUS_RTC
    | STATUS_WAKE;
/// Enable bit n gates status bit n; WAK_STS has no enable and never raises the SCI.
const ENABLE_IMPLEMENTED: u16 =
    STATUS_TIMER | STATUS_GLOBAL | STATUS_POWER_BUTTON | STATUS_SLEEP_BUTTON | STATUS_RTC;
const ENABLE_TIMER: u16 = STATUS_TIMER;

const CONTROL_SCI_ENABLE: u16 = 1 << 0;
const CONTROL_BUS_MASTER_RELOAD: u16 = 1 << 1;
const CONTROL_GLOBAL_RELEASE: u16 = 1 << 2;
const CONTROL_SLEEP_TYPE_SHIFT: u32 = 10;
const CONTROL_SLEEP_TYPE_MASK: u16 = 0b111;
const CONTROL_SLEEP_ENABLE: u16 = 1 << 13;
/// What the register keeps: SLP_EN is write-only and self-clearing, SCI_EN is forced on.
const CONTROL_STORED: u16 = (CONTROL_SLEEP_TYPE_MASK << CONTROL_SLEEP_TYPE_SHIFT)
    | CONTROL_BUS_MASTER_RELOAD
    | CONTROL_GLOBAL_RELEASE;

#[derive(Snapshot)]
pub(crate) struct AcpiPm {
    status: u16,
    enable: u16,
    control: u16,
    /// Level of the timer's overflow bit when TMR_STS was last updated.
    timer_high_bit: bool,
    shutdown_requested: bool,
}

impl AcpiPm {
    pub fn new() -> Self {
        AcpiPm {
            status: 0,
            enable: 0,
            control: CONTROL_SCI_ENABLE,
            timer_high_bit: false,
            shutdown_requested: false,
        }
    }

    pub fn decodes(port: u16) -> bool {
        (PM1_EVENT_BASE..=PM_LAST_PORT).contains(&port)
    }

    /// Set once the guest wrote SLP_EN with the S5 sleep type; stays set.
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    fn timer_ticks(now_ns: u64) -> u64 {
        (now_ns as u128 * PM_TIMER_HZ as u128 / NS_PER_SECOND) as u64
    }

    /// TMR_STS sets when the counter's top bit changes (either way), as on the PIIX4.
    fn update_timer_status(&mut self, now_ns: u64) {
        let high = (Self::timer_ticks(now_ns) >> TIMER_OVERFLOW_BIT) & 1 != 0;
        if high != self.timer_high_bit {
            self.timer_high_bit = high;
            self.status |= STATUS_TIMER;
        }
    }

    /// Whether the SCI line is high: an enabled status bit is set.
    pub fn sci_asserted(&mut self, now_ns: u64) -> bool {
        self.update_timer_status(now_ns);
        self.status & self.enable & ENABLE_IMPLEMENTED != 0
    }

    /// When the timer's top bit next flips, if that would raise the SCI. Without a deadline an
    /// idle CPU would sleep through it.
    pub fn next_deadline(&self, now_ns: u64) -> Option<u64> {
        if self.enable & ENABLE_TIMER == 0 || self.status & STATUS_TIMER != 0 {
            return None;
        }
        let half_period = 1u64 << TIMER_OVERFLOW_BIT;
        let next_tick = (Self::timer_ticks(now_ns) / half_period + 1) * half_period;
        let ns = (next_tick as u128 * NS_PER_SECOND).div_ceil(PM_TIMER_HZ as u128);
        Some(ns as u64)
    }

    pub fn read(&mut self, port: u16, size: u8, now_ns: u64) -> u32 {
        self.update_timer_status(now_ns);
        let (base, value) = match port {
            PM1_EVENT_BASE..PM1_CONTROL_BASE => (
                PM1_EVENT_BASE,
                self.status as u32 | (self.enable as u32) << 16,
            ),
            PM1_CONTROL_BASE..PM_TIMER_BASE => (
                PM1_CONTROL_BASE,
                (self.control & !CONTROL_SLEEP_ENABLE) as u32,
            ),
            _ => (
                PM_TIMER_BASE,
                (Self::timer_ticks(now_ns) & TIMER_MASK) as u32,
            ),
        };
        let shifted = value >> (8 * (port - base));
        match size {
            1 => shifted & 0xff,
            2 => shifted & 0xffff,
            _ => shifted,
        }
    }

    /// Byte lanes are handled independently so any access width works.
    pub fn write(&mut self, port: u16, size: u8, value: u32, now_ns: u64) {
        self.update_timer_status(now_ns);
        for lane in 0..size.min(4) as u16 {
            let at = port + lane;
            let byte = (value >> (8 * lane)) as u8 as u16;
            match at {
                PM1_EVENT_BASE..0x602 => {
                    let clear = byte << (8 * (at - PM1_EVENT_BASE));
                    self.status &= !(clear & STATUS_IMPLEMENTED);
                }
                0x602..PM1_CONTROL_BASE => {
                    let shift = 8 * (at - PM1_EVENT_BASE - 2);
                    self.enable =
                        (self.enable & !(0xff << shift)) | ((byte << shift) & ENABLE_IMPLEMENTED);
                }
                PM1_CONTROL_BASE..0x606 => self.write_control_byte(at - PM1_CONTROL_BASE, byte),
                _ => {}
            }
        }
    }

    fn write_control_byte(&mut self, lane: u16, byte: u16) {
        let shift = 8 * lane;
        self.control = (self.control & !(0xff << shift)) | (byte << shift);
        // SLP_EN is in the high byte, so the sleep type (bits 12:10) is complete when it arrives.
        if self.control & CONTROL_SLEEP_ENABLE != 0 {
            let sleep_type = (self.control >> CONTROL_SLEEP_TYPE_SHIFT) & CONTROL_SLEEP_TYPE_MASK;
            if sleep_type == SLEEP_TYPE_S5 {
                self.shutdown_requested = true;
            }
        }
        self.control = (self.control & CONTROL_STORED) | CONTROL_SCI_ENABLE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_SECOND_NS: u64 = 1_000_000_000;
    const SLEEP_TYPE_S3: u16 = 3;

    fn sleep_command(sleep_type: u16) -> u32 {
        (CONTROL_SLEEP_ENABLE | sleep_type << CONTROL_SLEEP_TYPE_SHIFT | CONTROL_SCI_ENABLE) as u32
    }

    #[test]
    fn acpi_starts_enabled_and_sci_en_cannot_be_cleared() {
        let mut pm = AcpiPm::new();
        assert_eq!(pm.read(PM1_CONTROL_BASE, 2, 0), CONTROL_SCI_ENABLE as u32);
        pm.write(PM1_CONTROL_BASE, 2, 0, 0);
        assert_eq!(pm.read(PM1_CONTROL_BASE, 2, 0), CONTROL_SCI_ENABLE as u32);
    }

    #[test]
    fn s5_sleep_request_powers_off_and_other_types_do_not() {
        let mut pm = AcpiPm::new();
        pm.write(PM1_CONTROL_BASE, 2, sleep_command(SLEEP_TYPE_S3), 0);
        assert!(!pm.shutdown_requested());
        pm.write(PM1_CONTROL_BASE, 2, sleep_command(SLEEP_TYPE_S5), 0);
        assert!(pm.shutdown_requested());
    }

    #[test]
    fn s5_written_as_two_bytes_works_and_sleep_enable_reads_back_clear() {
        let mut pm = AcpiPm::new();
        let command = sleep_command(SLEEP_TYPE_S5);
        pm.write(PM1_CONTROL_BASE, 1, command & 0xff, 0);
        assert!(!pm.shutdown_requested());
        pm.write(PM1_CONTROL_BASE + 1, 1, command >> 8, 0);
        assert!(pm.shutdown_requested());
        assert_eq!(
            pm.read(PM1_CONTROL_BASE, 2, 0) & CONTROL_SLEEP_ENABLE as u32,
            0
        );
    }

    #[test]
    fn timer_counts_at_3_579_545_hz_in_24_bits() {
        let mut pm = AcpiPm::new();
        assert_eq!(pm.read(PM_TIMER_BASE, 4, 0), 0);
        assert_eq!(pm.read(PM_TIMER_BASE, 4, ONE_SECOND_NS), PM_TIMER_HZ as u32);
        let wrap_ns = (1u128 << TIMER_BITS) * 1_000_000_000 / PM_TIMER_HZ as u128;
        let after_wrap = pm.read(PM_TIMER_BASE, 4, wrap_ns as u64 + 1_000);
        assert!(after_wrap < 0x1000, "{after_wrap:#x}");
    }

    #[test]
    fn status_is_write_one_to_clear_and_enable_is_stored() {
        let mut pm = AcpiPm::new();
        pm.write(
            PM1_EVENT_BASE + 2,
            2,
            (STATUS_POWER_BUTTON | STATUS_TIMER) as u32,
            0,
        );
        assert_eq!(
            pm.read(PM1_EVENT_BASE, 4, 0) >> 16,
            (STATUS_POWER_BUTTON | STATUS_TIMER) as u32
        );
        pm.status = STATUS_POWER_BUTTON | STATUS_RTC;
        pm.write(PM1_EVENT_BASE, 2, STATUS_RTC as u32, 0);
        assert_eq!(pm.read(PM1_EVENT_BASE, 2, 0), STATUS_POWER_BUTTON as u32);
    }

    #[test]
    fn sci_follows_enabled_status_bits() {
        let mut pm = AcpiPm::new();
        pm.status = STATUS_POWER_BUTTON;
        assert!(!pm.sci_asserted(0));
        pm.write(PM1_EVENT_BASE + 2, 2, STATUS_POWER_BUTTON as u32, 0);
        assert!(pm.sci_asserted(0));
        pm.write(PM1_EVENT_BASE, 2, STATUS_POWER_BUTTON as u32, 0);
        assert!(!pm.sci_asserted(0));
    }

    #[test]
    fn timer_overflow_raises_the_sci_when_enabled_and_has_a_deadline() {
        let mut pm = AcpiPm::new();
        assert_eq!(pm.next_deadline(0), None);
        pm.write(PM1_EVENT_BASE + 2, 2, ENABLE_TIMER as u32, 0);
        let deadline = pm.next_deadline(0).unwrap();
        assert!(!pm.sci_asserted(deadline - 1_000));
        assert!(pm.sci_asserted(deadline));
        assert_eq!(pm.next_deadline(deadline), None);
        pm.write(PM1_EVENT_BASE, 2, STATUS_TIMER as u32, deadline);
        assert!(!pm.sci_asserted(deadline));
        assert!(pm.next_deadline(deadline).unwrap() > deadline);
    }

    #[test]
    fn reads_of_any_width_and_alignment_see_the_same_registers() {
        let mut pm = AcpiPm::new();
        pm.write(PM1_EVENT_BASE + 2, 2, STATUS_RTC as u32, 0);
        assert_eq!(pm.read(PM1_EVENT_BASE + 3, 1, 0), (STATUS_RTC >> 8) as u32);
        assert_eq!(pm.read(PM1_EVENT_BASE + 2, 2, 0), STATUS_RTC as u32);
    }
}
