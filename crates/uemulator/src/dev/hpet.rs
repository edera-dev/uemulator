//! High Precision Event Timer (IA-PC HPET specification 1.0a): a 64-bit up-counter and three
//! comparators, running on virtual time. Legacy replacement routing is not offered (LEG_RT_CAP
//! clear), so the PIT keeps IRQ 0; timers interrupt through an I/O APIC pin of the guest's
//! choosing or, with FSB delivery, as an MSI message. QEMU's `hw/timer/hpet.c` is the behavioral
//! reference.
#[cfg(freestanding)]
use crate::prelude::*;
use crate::snapshot::Snapshot;

pub(crate) const HPET_BASE: u64 = 0xfed0_0000;
pub(crate) const HPET_MMIO_SIZE: u64 = 0x400;
pub(crate) const TIMER_COUNT: usize = 3;
/// 69.841279 ns per tick (14.31818 MHz, the PIIX's 14.318 MHz oscillator), in femtoseconds as the
/// capabilities register wants it.
pub(crate) const CLOCK_PERIOD_FEMTOSECONDS: u64 = 69_841_279;
const FEMTOSECONDS_PER_NANOSECOND: u128 = 1_000_000;

const REG_CAPABILITIES: u64 = 0x00;
const REG_CONFIG: u64 = 0x10;
const REG_INTERRUPT_STATUS: u64 = 0x20;
const REG_COUNTER: u64 = 0xf0;
const TIMER_REGISTERS_FIRST: u64 = 0x100;
const TIMER_REGISTERS_STRIDE: u64 = 0x20;
const TIMER_CONFIG: u64 = 0x00;
const TIMER_COMPARATOR: u64 = 0x08;
const TIMER_FSB_ROUTE: u64 = 0x10;

const CAP_REVISION: u64 = 1;
const CAP_LAST_TIMER_SHIFT: u32 = 8;
/// COUNT_SIZE_CAP: the main counter is 64 bits wide.
const CAP_COUNTER_64BIT: u64 = 1 << 13;
const CAP_VENDOR_SHIFT: u32 = 16;
const CAP_VENDOR_INTEL: u64 = 0x8086;
const CAP_PERIOD_SHIFT: u32 = 32;
pub(crate) const CAPABILITIES: u64 = CAP_REVISION
    | (TIMER_COUNT as u64 - 1) << CAP_LAST_TIMER_SHIFT
    | CAP_COUNTER_64BIT
    | CAP_VENDOR_INTEL << CAP_VENDOR_SHIFT
    | CLOCK_PERIOD_FEMTOSECONDS << CAP_PERIOD_SHIFT;

/// The low dword is also the HPET ACPI table's event timer block id.
pub(crate) const CAPABILITIES_LOW_DWORD: u32 = CAPABILITIES as u32;

const CONFIG_ENABLE: u64 = 1 << 0;

const TIMER_LEVEL_TRIGGERED: u64 = 1 << 1;
const TIMER_INTERRUPT_ENABLE: u64 = 1 << 2;
const TIMER_PERIODIC: u64 = 1 << 3;
const TIMER_PERIODIC_CAPABLE: u64 = 1 << 4;
const TIMER_64BIT_CAPABLE: u64 = 1 << 5;
const TIMER_SET_VALUE: u64 = 1 << 6;
const TIMER_32BIT_MODE: u64 = 1 << 8;
const TIMER_ROUTE_SHIFT: u32 = 9;
const TIMER_ROUTE_MASK: u64 = 0x1f;
const TIMER_FSB_ENABLE: u64 = 1 << 14;
const TIMER_FSB_CAPABLE: u64 = 1 << 15;
const TIMER_ROUTE_CAPABILITY_SHIFT: u32 = 32;
/// I/O APIC pins 20..=23, the ones QEMU offers; none of them is an ISA line.
const TIMER_ROUTE_CAPABILITY: u64 = 0xf0_0000;
const TIMER_READ_ONLY: u64 = TIMER_PERIODIC_CAPABLE
    | TIMER_64BIT_CAPABLE
    | TIMER_FSB_CAPABLE
    | TIMER_ROUTE_CAPABILITY << TIMER_ROUTE_CAPABILITY_SHIFT;
const TIMER_WRITABLE: u64 = TIMER_LEVEL_TRIGGERED
    | TIMER_INTERRUPT_ENABLE
    | TIMER_PERIODIC
    | TIMER_SET_VALUE
    | TIMER_32BIT_MODE
    | TIMER_ROUTE_MASK << TIMER_ROUTE_SHIFT
    | TIMER_FSB_ENABLE;
const MASK_32BIT: u64 = 0xffff_ffff;
/// A 64-bit comparator more than this far ahead of the counter has already been passed (the
/// counter would have to wrap to reach it).
const COMPARATOR_PASSED: u64 = 1 << 63;

/// What a firing timer asks the platform to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Snapshot)]
pub(crate) enum HpetOutput {
    /// An edge-triggered timer pulsed this I/O APIC pin.
    Pulse { pin: u8 },
    /// A level-triggered timer's line changed.
    Level { pin: u8, high: bool },
    /// FSB interrupt delivery: an MSI write of `data` to `address`, from the HPET's requester id.
    Message { address: u64, data: u32 },
}

#[derive(Clone, Copy, Snapshot)]
struct Timer {
    config: u64,
    comparator: u64,
    period: u64,
    fsb_route: u64,
    /// Virtual time at which the counter next equals the comparator.
    deadline_ns: Option<u64>,
}

#[derive(Snapshot)]
pub(crate) struct Hpet {
    config: u64,
    /// Bit n: timer n's level-triggered interrupt is asserted.
    interrupt_status: u64,
    /// Counter value at `counter_epoch_ns` while running; the frozen value while not.
    counter_base: u64,
    counter_epoch_ns: u64,
    timers: [Timer; TIMER_COUNT],
    outputs: Vec<HpetOutput>,
}

fn ticks_in(nanoseconds: u64) -> u64 {
    (nanoseconds as u128 * FEMTOSECONDS_PER_NANOSECOND / CLOCK_PERIOD_FEMTOSECONDS as u128) as u64
}

fn nanoseconds_for(ticks: u64) -> u64 {
    (ticks as u128 * CLOCK_PERIOD_FEMTOSECONDS as u128).div_ceil(FEMTOSECONDS_PER_NANOSECOND) as u64
}

impl Hpet {
    pub fn new() -> Self {
        let timer = Timer {
            config: TIMER_READ_ONLY,
            comparator: u64::MAX,
            period: 0,
            fsb_route: 0,
            deadline_ns: None,
        };
        Hpet {
            config: 0,
            interrupt_status: 0,
            counter_base: 0,
            counter_epoch_ns: 0,
            timers: [timer; TIMER_COUNT],
            outputs: Vec::new(),
        }
    }

    pub fn decodes(phys: u64) -> bool {
        (HPET_BASE..HPET_BASE + HPET_MMIO_SIZE).contains(&phys)
    }

    fn running(&self) -> bool {
        self.config & CONFIG_ENABLE != 0
    }

    fn counter(&self, now_ns: u64) -> u64 {
        if self.running() {
            self.counter_base
                .wrapping_add(ticks_in(now_ns.saturating_sub(self.counter_epoch_ns)))
        } else {
            self.counter_base
        }
    }

    /// Fixes the counter at `now_ns` so the rate can change underneath it (enable, disable, a
    /// write to the counter itself).
    fn freeze_counter(&mut self, now_ns: u64) {
        self.counter_base = self.counter(now_ns);
        self.counter_epoch_ns = now_ns;
    }

    /// Recomputes when timer `index` fires. The counter's equality with the comparator is what
    /// fires it, so a comparator the counter already passed waits for a wrap that never comes.
    fn arm(&mut self, index: usize, now_ns: u64) {
        let counter = self.counter(now_ns);
        let ticks_since_epoch = counter.wrapping_sub(self.counter_base);
        let epoch_ns = self.counter_epoch_ns;
        let running = self.running();
        let timer = &mut self.timers[index];
        timer.deadline_ns = None;
        if !running {
            return;
        }
        let (mask, comparator) = if timer.config & TIMER_32BIT_MODE != 0 {
            (MASK_32BIT, timer.comparator & MASK_32BIT)
        } else {
            (u64::MAX, timer.comparator)
        };
        let ahead = comparator.wrapping_sub(counter) & mask;
        if mask == u64::MAX && ahead >= COMPARATOR_PASSED {
            return;
        }
        // The exact time the counter reaches the comparator, from the epoch it counts from, so
        // a periodic timer does not drift by the rounding of each period.
        timer.deadline_ns = Some(epoch_ns + nanoseconds_for(ticks_since_epoch + ahead));
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.timers.iter().filter_map(|t| t.deadline_ns).min()
    }

    /// Fires every timer that is due and queues what they raise.
    pub fn tick(&mut self, now_ns: u64) {
        for index in 0..TIMER_COUNT {
            let Some(deadline) = self.timers[index].deadline_ns else {
                continue;
            };
            if deadline > now_ns {
                continue;
            }
            self.fire(index);
            // A periodic timer that was not serviced for several periods fires once per tick
            // call, not once per missed period: the guest only sees "interrupt pending" anyway.
            let timer = &mut self.timers[index];
            if timer.config & TIMER_PERIODIC != 0 && timer.period != 0 {
                timer.comparator = timer.comparator.wrapping_add(timer.period);
                self.arm(index, now_ns);
            } else {
                self.timers[index].deadline_ns = None;
            }
        }
    }

    fn fire(&mut self, index: usize) {
        let config = self.timers[index].config;
        if config & TIMER_INTERRUPT_ENABLE == 0 {
            return;
        }
        if config & TIMER_FSB_ENABLE != 0 {
            let route = self.timers[index].fsb_route;
            self.outputs.push(HpetOutput::Message {
                address: route >> 32,
                data: route as u32,
            });
            return;
        }
        let pin = ((config >> TIMER_ROUTE_SHIFT) & TIMER_ROUTE_MASK) as u8;
        if config & TIMER_LEVEL_TRIGGERED != 0 {
            self.interrupt_status |= 1 << index;
            self.outputs.push(HpetOutput::Level { pin, high: true });
        } else {
            self.outputs.push(HpetOutput::Pulse { pin });
        }
    }

    pub fn take_outputs(&mut self) -> Vec<HpetOutput> {
        core::mem::take(&mut self.outputs)
    }

    pub fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }

    /// `size` is 1, 2, 4 or 8; the register is read whole and shifted, so any aligned or
    /// unaligned part of it works.
    pub fn mmio_read(&self, phys: u64, size: u8, now_ns: u64) -> u64 {
        let offset = phys - HPET_BASE;
        let value = self.read_register(offset & !7, now_ns) >> (8 * (offset & 7));
        match size {
            1 => value & 0xff,
            2 => value & 0xffff,
            4 => value & MASK_32BIT,
            _ => value,
        }
    }

    fn read_register(&self, offset: u64, now_ns: u64) -> u64 {
        match offset {
            REG_CAPABILITIES => CAPABILITIES,
            REG_CONFIG => self.config,
            REG_INTERRUPT_STATUS => self.interrupt_status,
            REG_COUNTER => self.counter(now_ns),
            TIMER_REGISTERS_FIRST.. => {
                let index = ((offset - TIMER_REGISTERS_FIRST) / TIMER_REGISTERS_STRIDE) as usize;
                let Some(timer) = self.timers.get(index) else {
                    return 0;
                };
                match (offset - TIMER_REGISTERS_FIRST) % TIMER_REGISTERS_STRIDE {
                    TIMER_CONFIG => timer.config,
                    TIMER_COMPARATOR => timer.comparator,
                    TIMER_FSB_ROUTE => timer.fsb_route,
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

    pub fn mmio_write(&mut self, phys: u64, size: u8, value: u64, now_ns: u64) {
        let offset = phys - HPET_BASE;
        let shift = 8 * (offset & 7);
        let width_mask = match size {
            1 => 0xff,
            2 => 0xffff,
            4 => MASK_32BIT,
            _ => u64::MAX,
        };
        let mask = width_mask << shift;
        self.write_register(offset & !7, (value << shift) & mask, mask, now_ns);
    }

    /// `value` is already in place within the register; only the bits of `mask` were written.
    fn write_register(&mut self, offset: u64, value: u64, mask: u64, now_ns: u64) {
        let merge = |old: u64| (old & !mask) | value;
        match offset {
            REG_CONFIG => self.write_config(merge(self.config), now_ns),
            REG_INTERRUPT_STATUS => self.clear_interrupt_status(value & mask),
            REG_COUNTER => {
                // Only a halted counter can be written.
                if !self.running() {
                    self.counter_base = merge(self.counter_base);
                }
            }
            TIMER_REGISTERS_FIRST.. => {
                let relative = offset - TIMER_REGISTERS_FIRST;
                let index = (relative / TIMER_REGISTERS_STRIDE) as usize;
                if index >= TIMER_COUNT {
                    return;
                }
                match relative % TIMER_REGISTERS_STRIDE {
                    TIMER_CONFIG => {
                        self.write_timer_config(index, merge(self.timers[index].config), now_ns)
                    }
                    TIMER_COMPARATOR => self.write_comparator(index, value, mask, now_ns),
                    TIMER_FSB_ROUTE => {
                        self.timers[index].fsb_route = merge(self.timers[index].fsb_route)
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn write_config(&mut self, new: u64, now_ns: u64) {
        let was_running = self.running();
        self.freeze_counter(now_ns);
        // LEG_RT_CNF is accepted by real parts only with LEG_RT_CAP, which is clear.
        self.config = new & CONFIG_ENABLE;
        if self.running() != was_running {
            for index in 0..TIMER_COUNT {
                self.arm(index, now_ns);
            }
        }
    }

    fn write_timer_config(&mut self, index: usize, new: u64, now_ns: u64) {
        let timer = &mut self.timers[index];
        timer.config = (timer.config & TIMER_READ_ONLY) | (new & TIMER_WRITABLE);
        // The route field only takes pins the timer is capable of.
        let route = (timer.config >> TIMER_ROUTE_SHIFT) & TIMER_ROUTE_MASK;
        if TIMER_ROUTE_CAPABILITY >> route & 1 == 0 {
            timer.config &= !(TIMER_ROUTE_MASK << TIMER_ROUTE_SHIFT);
        }
        // Switching to edge mode drops an asserted level.
        if timer.config & TIMER_LEVEL_TRIGGERED == 0 {
            self.interrupt_status &= !(1 << index);
        }
        self.arm(index, now_ns);
    }

    /// QEMU's rule: a periodic timer's comparator is only written while SET_VALUE is set, and
    /// every write also sets the period, so the usual "write the first expiry, then the
    /// period" pair works. SET_VALUE clears after one write.
    fn write_comparator(&mut self, index: usize, value: u64, mask: u64, now_ns: u64) {
        let timer = &mut self.timers[index];
        let merge = |old: u64| (old & !mask) | value;
        let periodic = timer.config & TIMER_PERIODIC != 0;
        if !periodic || timer.config & TIMER_SET_VALUE != 0 {
            timer.comparator = merge(timer.comparator);
        }
        if periodic {
            timer.period = merge(timer.period);
        }
        timer.config &= !TIMER_SET_VALUE;
        self.arm(index, now_ns);
    }

    fn clear_interrupt_status(&mut self, clear: u64) {
        for index in 0..TIMER_COUNT {
            if clear & (1 << index) != 0 && self.interrupt_status & (1 << index) != 0 {
                self.interrupt_status &= !(1 << index);
                let pin =
                    ((self.timers[index].config >> TIMER_ROUTE_SHIFT) & TIMER_ROUTE_MASK) as u8;
                self.outputs.push(HpetOutput::Level { pin, high: false });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000_000;
    const PIN: u8 = 22;

    fn read(hpet: &Hpet, offset: u64, size: u8, now: u64) -> u64 {
        hpet.mmio_read(HPET_BASE + offset, size, now)
    }

    fn write(hpet: &mut Hpet, offset: u64, size: u8, value: u64, now: u64) {
        hpet.mmio_write(HPET_BASE + offset, size, value, now);
    }

    fn timer_offset(index: u64, register: u64) -> u64 {
        TIMER_REGISTERS_FIRST + index * TIMER_REGISTERS_STRIDE + register
    }

    fn enable(hpet: &mut Hpet, now: u64) {
        write(hpet, REG_CONFIG, 8, CONFIG_ENABLE, now);
    }

    fn program_oneshot(hpet: &mut Hpet, index: u64, comparator: u64, extra: u64, now: u64) {
        let config = TIMER_INTERRUPT_ENABLE | (PIN as u64) << TIMER_ROUTE_SHIFT | extra;
        write(hpet, timer_offset(index, TIMER_CONFIG), 8, config, now);
        write(
            hpet,
            timer_offset(index, TIMER_COMPARATOR),
            8,
            comparator,
            now,
        );
    }

    #[test]
    fn capabilities_describe_three_64_bit_timers_at_14_3_mhz() {
        let hpet = Hpet::new();
        let caps = read(&hpet, REG_CAPABILITIES, 8, 0);
        assert_eq!(caps >> CAP_PERIOD_SHIFT, CLOCK_PERIOD_FEMTOSECONDS);
        assert_eq!((caps >> CAP_LAST_TIMER_SHIFT) & 0x1f, 2);
        assert_ne!(caps & CAP_COUNTER_64BIT, 0);
        assert_eq!(caps & (1 << 15), 0, "no legacy replacement");
        assert_eq!(
            read(&hpet, REG_CAPABILITIES + 4, 4, 0),
            CLOCK_PERIOD_FEMTOSECONDS
        );
        assert_eq!((caps >> CAP_VENDOR_SHIFT) & 0xffff, CAP_VENDOR_INTEL);
    }

    #[test]
    fn counter_is_frozen_until_enabled_then_follows_virtual_time() {
        let mut hpet = Hpet::new();
        assert_eq!(read(&hpet, REG_COUNTER, 8, NOW), 0);
        enable(&mut hpet, NOW);
        let one_second = read(&hpet, REG_COUNTER, 8, NOW + 1_000_000_000);
        assert!(
            (14_318_179..=14_318_181).contains(&one_second),
            "{one_second}"
        );
        write(&mut hpet, REG_CONFIG, 8, 0, NOW + 1_000_000_000);
        let frozen = read(&hpet, REG_COUNTER, 8, NOW + 5_000_000_000);
        assert_eq!(frozen, one_second);
    }

    #[test]
    fn counter_can_only_be_written_while_halted() {
        let mut hpet = Hpet::new();
        write(&mut hpet, REG_COUNTER, 8, 1000, NOW);
        assert_eq!(read(&hpet, REG_COUNTER, 8, NOW), 1000);
        enable(&mut hpet, NOW);
        write(&mut hpet, REG_COUNTER, 8, 5, NOW);
        assert!(read(&hpet, REG_COUNTER, 8, NOW) >= 1000);
    }

    #[test]
    fn halves_of_the_counter_read_with_32_bit_accesses() {
        let mut hpet = Hpet::new();
        write(&mut hpet, REG_COUNTER, 4, 0x1234_5678, 0);
        write(&mut hpet, REG_COUNTER + 4, 4, 0x9abc, 0);
        assert_eq!(read(&hpet, REG_COUNTER, 8, 0), 0x9abc_1234_5678);
        assert_eq!(read(&hpet, REG_COUNTER + 4, 4, 0), 0x9abc);
    }

    #[test]
    fn edge_timer_pulses_its_pin_at_the_comparator() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        program_oneshot(&mut hpet, 0, 1000, 0, 0);
        let deadline = hpet.next_deadline().unwrap();
        assert_eq!(deadline, nanoseconds_for(1000));
        hpet.tick(deadline - 1);
        assert!(!hpet.has_outputs());
        hpet.tick(deadline);
        assert_eq!(hpet.take_outputs(), vec![HpetOutput::Pulse { pin: PIN }]);
        assert_eq!(hpet.next_deadline(), None);
    }

    #[test]
    fn comparator_in_the_past_never_fires_in_64_bit_mode() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        program_oneshot(&mut hpet, 1, 10, 0, 1_000_000);
        assert_eq!(hpet.next_deadline(), None);
    }

    #[test]
    fn level_timer_holds_its_line_until_the_status_bit_is_cleared() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        program_oneshot(&mut hpet, 2, 100, TIMER_LEVEL_TRIGGERED, 0);
        hpet.tick(nanoseconds_for(100));
        assert_eq!(
            hpet.take_outputs(),
            vec![HpetOutput::Level {
                pin: PIN,
                high: true
            }]
        );
        assert_eq!(read(&hpet, REG_INTERRUPT_STATUS, 8, 0), 1 << 2);
        write(&mut hpet, REG_INTERRUPT_STATUS, 8, 1 << 2, 0);
        assert_eq!(read(&hpet, REG_INTERRUPT_STATUS, 8, 0), 0);
        assert_eq!(
            hpet.take_outputs(),
            vec![HpetOutput::Level {
                pin: PIN,
                high: false
            }]
        );
    }

    #[test]
    fn periodic_timer_reloads_with_the_period() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        let config = TIMER_INTERRUPT_ENABLE
            | TIMER_PERIODIC
            | TIMER_SET_VALUE
            | (PIN as u64) << TIMER_ROUTE_SHIFT;
        write(&mut hpet, timer_offset(0, TIMER_CONFIG), 8, config, 0);
        write(&mut hpet, timer_offset(0, TIMER_COMPARATOR), 8, 500, 0);
        write(&mut hpet, timer_offset(0, TIMER_COMPARATOR), 8, 200, 0);
        assert_eq!(hpet.next_deadline(), Some(nanoseconds_for(500)));
        hpet.tick(nanoseconds_for(500));
        assert_eq!(hpet.take_outputs().len(), 1);
        assert_eq!(hpet.next_deadline(), Some(nanoseconds_for(700)));
    }

    #[test]
    fn fsb_delivery_sends_the_programmed_message() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        write(
            &mut hpet,
            timer_offset(1, TIMER_FSB_ROUTE),
            8,
            0xfee0_1000 << 32 | 0x4041,
            0,
        );
        program_oneshot(&mut hpet, 1, 50, TIMER_FSB_ENABLE, 0);
        hpet.tick(nanoseconds_for(50));
        assert_eq!(
            hpet.take_outputs(),
            vec![HpetOutput::Message {
                address: 0xfee0_1000,
                data: 0x4041
            }]
        );
    }

    #[test]
    fn route_field_only_accepts_pins_in_the_capability_mask() {
        let mut hpet = Hpet::new();
        write(
            &mut hpet,
            timer_offset(0, TIMER_CONFIG),
            8,
            5 << TIMER_ROUTE_SHIFT,
            0,
        );
        let config = read(&hpet, timer_offset(0, TIMER_CONFIG), 8, 0);
        assert_eq!((config >> TIMER_ROUTE_SHIFT) & TIMER_ROUTE_MASK, 0);
        assert_eq!(
            config >> TIMER_ROUTE_CAPABILITY_SHIFT,
            TIMER_ROUTE_CAPABILITY
        );
    }

    #[test]
    fn disabling_the_counter_stops_pending_timers() {
        let mut hpet = Hpet::new();
        enable(&mut hpet, 0);
        program_oneshot(&mut hpet, 0, 1000, 0, 0);
        write(&mut hpet, REG_CONFIG, 8, 0, 10);
        assert_eq!(hpet.next_deadline(), None);
        enable(&mut hpet, 20);
        assert!(hpet.next_deadline().unwrap() > nanoseconds_for(900));
    }

    #[test]
    fn thirty_two_bit_mode_wraps_the_comparison() {
        let mut hpet = Hpet::new();
        write(&mut hpet, REG_COUNTER, 8, 0xffff_ff00, 0);
        enable(&mut hpet, 0);
        program_oneshot(&mut hpet, 0, 0x10, TIMER_32BIT_MODE, 0);
        assert_eq!(hpet.next_deadline(), Some(nanoseconds_for(0x110)));
    }
}
