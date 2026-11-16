#[cfg(freestanding)]
use crate::prelude::*;
use crate::snapshot::Snapshot;

/// Input clock of the 8254 (14.31818 MHz / 12, rounded). The guest calibrates the TSC against
/// this, so it must be derived from the same emulated nanosecond clock as the TSC.
const PIT_HZ: u128 = 1_193_182;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

const CHANNEL0_PORT: u16 = 0x40;
const CHANNEL_COUNT: usize = 3;
/// The port offset selects among four registers; the control word is the fourth.
const PORT_REGISTER_MASK: u16 = 3;
const CONTROL_REGISTER: usize = 3;
/// Channels 0 and 1 have their gate tied high; channel 2's is port 0x61 bit 0.
const CHANNEL_SPEAKER: usize = 2;
/// What a read of the write-only control register returns (nothing drives the bus).
const OPEN_BUS: u8 = 0xff;

/// Control word fields: bits 7:6 channel, 5:4 access, 3:1 mode, 0 BCD.
const CONTROL_CHANNEL_SHIFT: u8 = 6;
const CONTROL_ACCESS_SHIFT: u8 = 4;
const CONTROL_ACCESS_MASK: u8 = 3;
const CONTROL_MODE_SHIFT: u8 = 1;
const CONTROL_MODE_MASK: u8 = 7;
const CONTROL_BCD: u8 = 1;
/// "Channel" value 3 in a control word selects the read-back command instead.
const CONTROL_READ_BACK: usize = 3;
/// Read-back command: bits 3:1 select channels 2..0, and the two inhibit bits are active low:
/// bit 5 clear latches the count, bit 4 clear latches the status.
const READ_BACK_CHANNEL0: u8 = 1 << 1;
const READ_BACK_NO_COUNT: u8 = 1 << 5;
const READ_BACK_NO_STATUS: u8 = 1 << 4;

/// Status byte of the read-back command.
const STATUS_OUT: u8 = 1 << 7;
const STATUS_NULL_COUNT: u8 = 1 << 6;
const STATUS_ACCESS_SHIFT: u8 = 4;
const STATUS_MODE_SHIFT: u8 = 1;

/// Access field 0 is the counter latch command; it never becomes a channel's access mode.
const ACCESS_LATCH: u8 = 0;
const ACCESS_LOW: u8 = 1;
const ACCESS_HIGH: u8 = 2;
const ACCESS_LOW_HIGH: u8 = 3;

const MODE_TERMINAL_COUNT: u8 = 0;
const MODE_ONE_SHOT: u8 = 1;
const MODE_RATE_GENERATOR: u8 = 2;
const MODE_SQUARE_WAVE: u8 = 3;
const MODE_SOFTWARE_STROBE: u8 = 4;
const MODE_HARDWARE_STROBE: u8 = 5;
/// Mode field values 6 and 7 are aliases of 2 and 3.
const MODE_ALIAS_FIRST: u8 = 6;
const MODE_ALIAS_OFFSET: u8 = 4;

/// A programmed count of 0 means 65536 in binary and 10000 in BCD.
const FULL_COUNT_BINARY: u32 = 0x10000;
const FULL_COUNT_BCD: u32 = 10_000;
const COUNT_MASK: u32 = 0xffff;
const BCD_DIGIT_BITS: u32 = 4;
const BCD_DIGIT_MASK: u32 = 0xf;
const BCD_DIGITS: u32 = 4;
const DECIMAL_BASE: u32 = 10;

#[derive(Clone, Copy, Snapshot)]
struct Channel {
    mode: u8,
    access: u8,
    bcd: bool,
    /// Programmed count in clock ticks (1..=65536, or 1..=10000 in BCD); the period of modes 2
    /// and 3 and the count loaded by the next trigger or load.
    reload: u32,
    gate: bool,
    /// Low byte already written in a low/high access sequence; next write is the high byte.
    write_low_byte: Option<u8>,
    /// Next unlatched read returns the high byte.
    read_high_next: bool,
    /// Count captured by a latch command, consumed by the following read(s).
    latched_count: Option<u16>,
    latch_high_next: bool,
    /// Status captured by a read-back command; the next read returns it before any latched count.
    latched_status: Option<u8>,
    /// A count has been written since the last control word.
    loaded: bool,
    /// The count register has been copied into the counting element (status "null count" is the
    /// opposite). Modes 1 and 5 only do that on a trigger.
    counter_loaded: bool,
    running: bool,
    /// Emulated time at which the counter held `start_count`; the current count is derived from
    /// elapsed time rather than ticked.
    start_ns: u64,
    start_count: u32,
    /// Mode 0: terminal count was reached before counting was frozen, so OUT stays high.
    terminal_reached: bool,
    /// Mode 2: a count written mid-period, and the tick (since `start_ns`) at which that period
    /// ends and it takes effect.
    pending_reload: Option<(u32, u64)>,
}

impl Default for Channel {
    /// Power-on is undefined on real hardware; this matches QEMU's reset (mode 3, count 65536)
    /// but idle, and makes the access mode low/high so reads, latches and writes agree before
    /// the first control word.
    fn default() -> Self {
        Channel {
            mode: MODE_SQUARE_WAVE,
            access: ACCESS_LOW_HIGH,
            bcd: false,
            reload: FULL_COUNT_BINARY,
            gate: true,
            write_low_byte: None,
            read_high_next: false,
            latched_count: None,
            latch_high_next: false,
            latched_status: None,
            loaded: false,
            counter_loaded: false,
            running: false,
            start_ns: 0,
            start_count: 0,
            terminal_reached: false,
            pending_reload: None,
        }
    }
}

#[derive(Snapshot)]
pub(crate) struct Pit {
    channels: [Channel; CHANNEL_COUNT],
    /// Emulated time of the next IRQ 0, if channel 0 is armed. Consumed by `tick`.
    next_irq: Option<u64>,
}

impl Default for Pit {
    fn default() -> Self {
        let mut channels = [Channel::default(); CHANNEL_COUNT];
        // The speaker gate is driven from port 0x61 and is low until the guest raises it.
        channels[CHANNEL_SPEAKER].gate = false;
        Pit {
            channels,
            next_irq: None,
        }
    }
}

fn ns_to_ticks(ns: u64) -> u64 {
    (ns as u128 * PIT_HZ / NANOS_PER_SECOND) as u64
}

/// Rounds up so a deadline derived from a tick count is never earlier than that tick.
/// Saturates instead of wrapping for tick counts no real run reaches.
fn ticks_to_ns(ticks: u64) -> u64 {
    let ns = (ticks as u128 * NANOS_PER_SECOND).div_ceil(PIT_HZ);
    u64::try_from(ns).unwrap_or(u64::MAX)
}

/// The four BCD digits of a programmed value as a tick count. Digits above 9 are taken at face
/// value, which is as good as any answer for a value the hardware leaves undefined.
fn bcd_to_ticks(value: u16) -> u32 {
    let mut ticks = 0;
    for digit in (0..BCD_DIGITS).rev() {
        let nibble = (value as u32 >> (digit * BCD_DIGIT_BITS)) & BCD_DIGIT_MASK;
        ticks = ticks * DECIMAL_BASE + nibble;
    }
    ticks
}

fn ticks_to_bcd(ticks: u32) -> u16 {
    let mut value = 0;
    let mut rest = ticks;
    for digit in 0..BCD_DIGITS {
        value |= (rest % DECIMAL_BASE) << (digit * BCD_DIGIT_BITS);
        rest /= DECIMAL_BASE;
    }
    value as u16
}

impl Channel {
    /// Modes 6 and 7 behave as 2 and 3.
    fn effective_mode(&self) -> u8 {
        if self.mode >= MODE_ALIAS_FIRST {
            self.mode - MODE_ALIAS_OFFSET
        } else {
            self.mode
        }
    }

    fn is_periodic(&self) -> bool {
        matches!(
            self.effective_mode(),
            MODE_RATE_GENERATOR | MODE_SQUARE_WAVE
        )
    }

    fn full_count(&self) -> u32 {
        if self.bcd {
            FULL_COUNT_BCD
        } else {
            FULL_COUNT_BINARY
        }
    }

    /// Ticks the counter has run since `start_ns` (only meaningful while running).
    fn ticks_since_start(&self, now: u64) -> u64 {
        ns_to_ticks(now.saturating_sub(self.start_ns))
    }

    /// Applies a mode 2 count whose period boundary has passed: the new period is anchored at
    /// that boundary, not at the moment the guest happens to look.
    fn settled(mut self, now: u64) -> Channel {
        if let Some((count, boundary)) = self.pending_reload
            && self.running
            && self.ticks_since_start(now) >= boundary
        {
            self.start_ns = self.start_ns.saturating_add(ticks_to_ns(boundary));
            self.reload = count;
            self.start_count = count;
            self.pending_reload = None;
        }
        self
    }

    /// Current counter value in ticks. Square wave is counted like a rate generator (r down to 1
    /// once per period) rather than decrementing by two twice per period. The one-shot modes keep
    /// counting down past zero and wrap.
    fn count(&self, now: u64) -> u32 {
        if !self.running {
            return self.start_count;
        }
        let ticks = self.ticks_since_start(now);
        if self.is_periodic() {
            let reload = self.reload as u64;
            (reload - (ticks % reload)) as u32
        } else {
            let modulus = self.full_count() as i64;
            (self.start_count as i64 - ticks as i64).rem_euclid(modulus) as u32
        }
    }

    /// The count as the guest reads it: binary, or four BCD digits.
    fn counter_value(&self, now: u64) -> u16 {
        let count = self.count(now);
        if self.bcd {
            ticks_to_bcd(count % FULL_COUNT_BCD)
        } else {
            (count & COUNT_MASK) as u16
        }
    }

    /// OUT pin level. Counting is derived from time, so this is a pure function of `now`.
    /// Rate generator: low for the one tick at count 1. Software and hardware strobe: low for
    /// the one tick at count 0. Square wave: high for the first half of the period (the larger
    /// half when odd). Mode 0 is low until terminal count; mode 1 is low from trigger to
    /// terminal count. A count of 1 in mode 2 is not a valid program and reads high.
    fn out(&self, now: u64) -> bool {
        let ticks = self.ticks_since_start(now);
        match self.effective_mode() {
            MODE_TERMINAL_COUNT => {
                self.terminal_reached || (self.running && ticks >= self.start_count as u64)
            }
            MODE_ONE_SHOT => !self.running || ticks >= self.start_count as u64,
            MODE_RATE_GENERATOR => {
                let reload = self.reload as u64;
                reload < 2 || !(self.running && ticks % reload == reload - 1)
            }
            MODE_SQUARE_WAVE => {
                let reload = self.reload as u64;
                !self.running || (ticks % reload) < reload.div_ceil(2)
            }
            _ => !(self.running && ticks == self.start_count as u64),
        }
    }

    /// Time of the next rising edge of OUT strictly after `now`, which is what raises IRQ 0 on
    /// the PC. Edges are anchored at `start_ns` and computed from the tick count each time so
    /// rounding does not accumulate over periods.
    fn next_rising_edge(&self, now: u64) -> Option<u64> {
        if !self.running {
            return None;
        }
        let edge_after = |ticks: u64| {
            let edge = self.start_ns.saturating_add(ticks_to_ns(ticks));
            (edge > now).then_some(edge)
        };
        match self.effective_mode() {
            MODE_TERMINAL_COUNT | MODE_ONE_SHOT if !self.terminal_reached => {
                edge_after(self.start_count as u64)
            }
            MODE_SOFTWARE_STROBE | MODE_HARDWARE_STROBE => edge_after(self.start_count as u64 + 1),
            MODE_RATE_GENERATOR | MODE_SQUARE_WAVE => {
                let reload = self.reload as u64;
                let periods_done = self.ticks_since_start(now) / reload;
                edge_after((periods_done + 1).saturating_mul(reload))
            }
            _ => None,
        }
    }

    /// Latches the count unless a previous latch is still unread.
    fn latch_count(&mut self, now: u64) {
        if self.latched_count.is_none() {
            self.latched_count = Some(self.counter_value(now));
            self.latch_high_next = false;
        }
    }

    fn latch_status(&mut self, now: u64) {
        if self.latched_status.is_none() {
            let mut status = (self.access << STATUS_ACCESS_SHIFT)
                | (self.mode << STATUS_MODE_SHIFT)
                | self.bcd as u8;
            if self.out(now) {
                status |= STATUS_OUT;
            }
            if !self.counter_loaded || self.pending_reload.is_some() {
                status |= STATUS_NULL_COUNT;
            }
            self.latched_status = Some(status);
        }
    }

    /// Copies the programmed count into the counter and starts it if the gate allows.
    fn load_counter(&mut self, now: u64) {
        self.start_count = self.reload;
        self.start_ns = now;
        self.running = self.gate;
        self.terminal_reached = false;
        self.counter_loaded = true;
        self.pending_reload = None;
    }

    /// Mode 1 and 5 trigger, on a rising gate or (channel 2 only) never otherwise.
    fn trigger(&mut self, now: u64) {
        self.load_counter(now);
        self.running = true;
    }

    /// The guest finished writing a count (`programmed` is the 16-bit value as written).
    fn count_written(&mut self, programmed: u16, now: u64) {
        let ticks = match (programmed, self.bcd) {
            (0, _) => self.full_count(),
            (value, true) => bcd_to_ticks(value),
            (value, false) => value as u32,
        };
        self.loaded = true;
        match self.effective_mode() {
            MODE_ONE_SHOT | MODE_HARDWARE_STROBE => {
                // The count waits for the next trigger; a run in progress is unaffected.
                self.reload = ticks;
                self.counter_loaded = false;
                if !self.running {
                    self.start_count = ticks;
                }
            }
            MODE_RATE_GENERATOR if self.counter_loaded && self.running => {
                // The current period finishes with the old count.
                let boundary = (self.ticks_since_start(now) / self.reload as u64 + 1)
                    .saturating_mul(self.reload as u64);
                self.pending_reload = Some((ticks, boundary));
            }
            MODE_RATE_GENERATOR if self.counter_loaded => {
                // Counting is frozen by the gate; the new count loads when it rises.
                self.reload = ticks;
                self.pending_reload = None;
            }
            _ => {
                self.reload = ticks;
                self.load_counter(now);
            }
        }
    }
}

impl Pit {
    pub fn new() -> Self {
        Pit::default()
    }

    /// Recomputes the IRQ 0 deadline after channel 0 changed.
    fn reschedule_irq(&mut self, now: u64) {
        self.next_irq = self.channels[0].next_rising_edge(now);
    }

    /// Drives channel 2's gate from port 0x61 bit 0. What a gate edge does depends on the mode:
    /// - 0 and 4: low freezes the count, high resumes it.
    /// - 1 and 5: a rising edge triggers (loads the count and starts), low does nothing.
    /// - 2 and 3: low freezes the count with OUT high, a rising edge reloads and restarts.
    pub fn set_gate2(&mut self, on: bool, now: u64) {
        let channel = &mut self.channels[CHANNEL_SPEAKER];
        *channel = channel.settled(now);
        if channel.gate == on {
            return;
        }
        channel.gate = on;
        if !channel.loaded {
            return;
        }
        match channel.effective_mode() {
            MODE_TERMINAL_COUNT | MODE_SOFTWARE_STROBE => {
                if on {
                    channel.start_ns = now;
                    channel.running = channel.counter_loaded;
                } else if channel.running {
                    if channel.effective_mode() == MODE_TERMINAL_COUNT {
                        channel.terminal_reached |=
                            channel.ticks_since_start(now) >= channel.start_count as u64;
                    }
                    channel.start_count = channel.count(now);
                    channel.running = false;
                }
            }
            MODE_ONE_SHOT | MODE_HARDWARE_STROBE => {
                if on {
                    channel.trigger(now);
                }
            }
            _ => {
                if on {
                    if let Some((count, _)) = channel.pending_reload.take() {
                        channel.reload = count;
                    }
                    channel.load_counter(now);
                } else if channel.running {
                    channel.start_count = channel.count(now);
                    channel.running = false;
                }
            }
        }
    }

    pub fn out2(&self, now: u64) -> bool {
        self.channels[CHANNEL_SPEAKER].settled(now).out(now)
    }

    fn control_word(&mut self, value: u8, now: u64) {
        let selected = (value >> CONTROL_CHANNEL_SHIFT) as usize;
        if selected == CONTROL_READ_BACK {
            for index in 0..CHANNEL_COUNT {
                if value & (READ_BACK_CHANNEL0 << index) == 0 {
                    continue;
                }
                let channel = &mut self.channels[index];
                *channel = channel.settled(now);
                if value & READ_BACK_NO_COUNT == 0 {
                    channel.latch_count(now);
                }
                if value & READ_BACK_NO_STATUS == 0 {
                    channel.latch_status(now);
                }
            }
            return;
        }
        let channel = &mut self.channels[selected];
        *channel = channel.settled(now);
        let access = (value >> CONTROL_ACCESS_SHIFT) & CONTROL_ACCESS_MASK;
        if access == ACCESS_LATCH {
            channel.latch_count(now);
            return;
        }
        channel.access = access;
        channel.mode = (value >> CONTROL_MODE_SHIFT) & CONTROL_MODE_MASK;
        channel.bcd = value & CONTROL_BCD != 0;
        channel.write_low_byte = None;
        channel.read_high_next = false;
        channel.loaded = false;
        channel.counter_loaded = false;
        channel.running = false;
        channel.terminal_reached = false;
        channel.pending_reload = None;
        if selected == 0 {
            self.reschedule_irq(now);
        }
    }

    pub fn write(&mut self, port: u16, value: u8, now: u64) {
        let register = (port.wrapping_sub(CHANNEL0_PORT) & PORT_REGISTER_MASK) as usize;
        if register == CONTROL_REGISTER {
            self.control_word(value, now);
            return;
        }
        let channel = &mut self.channels[register];
        *channel = channel.settled(now);
        let programmed = match channel.access {
            ACCESS_LOW => Some(value as u16),
            ACCESS_HIGH => Some((value as u16) << 8),
            // Includes the power-on state, which is low/high.
            _ => match channel.write_low_byte.take() {
                Some(low) => Some(low as u16 | (value as u16) << 8),
                None => {
                    channel.write_low_byte = Some(value);
                    // The first byte of a pair stops a mode 0 or 4 counter until the second
                    // arrives; in mode 0 OUT drops immediately.
                    let mode = channel.effective_mode();
                    if mode == MODE_TERMINAL_COUNT || mode == MODE_SOFTWARE_STROBE {
                        if channel.running {
                            channel.start_count = channel.count(now);
                        }
                        channel.running = false;
                        channel.terminal_reached = false;
                    }
                    None
                }
            },
        };
        if let Some(programmed) = programmed {
            channel.count_written(programmed, now);
        }
        if register == 0 {
            self.reschedule_irq(now);
        }
    }

    pub fn read(&mut self, port: u16, now: u64) -> u8 {
        let register = (port.wrapping_sub(CHANNEL0_PORT) & PORT_REGISTER_MASK) as usize;
        if register == CONTROL_REGISTER {
            return OPEN_BUS;
        }
        let channel = &mut self.channels[register];
        *channel = channel.settled(now);
        if let Some(status) = channel.latched_status.take() {
            return status;
        }
        if let Some(latched) = channel.latched_count {
            let high = channel.access == ACCESS_HIGH
                || (channel.access == ACCESS_LOW_HIGH && channel.latch_high_next);
            if channel.access == ACCESS_LOW_HIGH && !channel.latch_high_next {
                channel.latch_high_next = true;
                return latched as u8;
            }
            channel.latched_count = None;
            channel.latch_high_next = false;
            return if high {
                (latched >> 8) as u8
            } else {
                latched as u8
            };
        }
        let count = channel.counter_value(now);
        match channel.access {
            ACCESS_LOW => count as u8,
            ACCESS_HIGH => (count >> 8) as u8,
            _ => {
                channel.read_high_next = !channel.read_high_next;
                if channel.read_high_next {
                    count as u8
                } else {
                    (count >> 8) as u8
                }
            }
        }
    }

    pub fn state(&self, now: u64) -> String {
        let channel = &self.channels[0];
        format!(
            "pit0 mode {} reload {} next_irq {:?} now {now}",
            channel.mode, channel.reload, self.next_irq
        )
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.next_irq
    }

    /// Returns true if IRQ 0 should be raised. Periodic modes skip over periods that were
    /// missed while the guest was not being run, so a late tick yields one IRQ rather than a
    /// burst; one-shot modes disarm.
    pub fn tick(&mut self, now: u64) -> bool {
        let Some(deadline) = self.next_irq else {
            return false;
        };
        if now < deadline {
            return false;
        }
        self.channels[0] = self.channels[0].settled(now);
        self.reschedule_irq(now);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHANNEL1_PORT: u16 = 0x41;
    const CHANNEL2_PORT: u16 = 0x42;
    const CONTROL_PORT: u16 = 0x43;

    const fn control_word(channel: u8, access: u8, mode: u8, bcd: u8) -> u8 {
        channel << CONTROL_CHANNEL_SHIFT
            | access << CONTROL_ACCESS_SHIFT
            | mode << CONTROL_MODE_SHIFT
            | bcd
    }

    /// Time at which exactly `ticks` PIT clocks have elapsed since time 0 (`ns_to_ticks(at(t))`
    /// is `t`, and one nanosecond earlier it is `t - 1`).
    fn at(ticks: u64) -> u64 {
        ticks_to_ns(ticks)
    }

    fn write_count(pit: &mut Pit, port: u16, count: u16, now: u64) {
        pit.write(port, count as u8, now);
        pit.write(port, (count >> 8) as u8, now);
    }

    fn read_count(pit: &mut Pit, port: u16, now: u64) -> u16 {
        let low = pit.read(port, now) as u16;
        let high = pit.read(port, now) as u16;
        low | high << 8
    }

    fn program(pit: &mut Pit, channel: u8, mode: u8, count: u16, now: u64) {
        pit.write(
            CONTROL_PORT,
            control_word(channel, ACCESS_LOW_HIGH, mode, 0),
            now,
        );
        write_count(pit, CHANNEL0_PORT + channel as u16, count, now);
    }

    /// Channel 0 programmed at time 0.
    fn channel0(mode: u8, count: u16) -> Pit {
        let mut pit = Pit::new();
        program(&mut pit, 0, mode, count, 0);
        pit
    }

    /// Channel 2 programmed at time 0, with its gate raised first (as Linux does) or left low.
    fn channel2(mode: u8, count: u16, gate: bool) -> Pit {
        let mut pit = Pit::new();
        if gate {
            pit.set_gate2(true, 0);
        }
        program(&mut pit, 2, mode, count, 0);
        pit
    }

    fn read_back(pit: &mut Pit, channel: u8, inhibit: u8, now: u64) {
        pit.write(
            CONTROL_PORT,
            (CONTROL_READ_BACK as u8) << CONTROL_CHANNEL_SHIFT
                | inhibit
                | READ_BACK_CHANNEL0 << channel,
            now,
        );
    }

    fn status_of(pit: &mut Pit, channel: u8, now: u64) -> u8 {
        read_back(pit, channel, READ_BACK_NO_COUNT, now);
        pit.read(CHANNEL0_PORT + channel as u16, now)
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    // Clock conversion

    #[test]
    fn one_second_is_exactly_the_input_clock() {
        assert_eq!(ns_to_ticks(1_000_000_000), 1_193_182);
        assert_eq!(ticks_to_ns(1_193_182), 1_000_000_000);
        assert_eq!(ns_to_ticks(0), 0);
        assert_eq!(ticks_to_ns(0), 0);
    }

    #[test]
    fn ticks_to_ns_rounds_up() {
        assert_eq!(ticks_to_ns(1), 839);
        assert_eq!(ticks_to_ns(2), 1677);
        assert_eq!(ticks_to_ns(100), 83_810);
        assert_eq!(ticks_to_ns(11_932), 10_000_151);
        assert_eq!(ticks_to_ns(65_536), 54_925_402);
    }

    #[test]
    fn ns_to_ticks_rounds_down() {
        assert_eq!(ns_to_ticks(838), 0);
        assert_eq!(ns_to_ticks(839), 1);
        assert_eq!(ns_to_ticks(1676), 1);
        assert_eq!(ns_to_ticks(1677), 2);
    }

    #[test]
    fn tick_conversion_round_trips_exactly() {
        let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..20_000 {
            let ticks = 1 + rng.next() % (1 << 50);
            let ns = ticks_to_ns(ticks);
            assert_eq!(ns_to_ticks(ns), ticks);
            assert_eq!(ns_to_ticks(ns - 1), ticks - 1);
        }
    }

    #[test]
    fn tick_conversion_is_exact_at_large_values() {
        for ticks in [1u64 << 40, 1 << 50, 1 << 53] {
            assert_eq!(ns_to_ticks(ticks_to_ns(ticks)), ticks);
        }
        assert_eq!(ticks_to_ns(1 << 40), 921_495_319_051_076);
        assert_eq!(ticks_to_ns(1 << 50), 943_611_206_708_300_997);
    }

    #[test]
    fn ticks_to_ns_saturates_instead_of_wrapping() {
        assert_eq!(ticks_to_ns(u64::MAX), u64::MAX);
        assert_eq!(ticks_to_ns(1 << 60), u64::MAX);
    }

    #[test]
    fn ns_to_ticks_handles_the_largest_time() {
        assert_eq!(ns_to_ticks(u64::MAX), 22_010_322_987_356_910);
    }

    // BCD conversion

    #[test]
    fn bcd_digits_convert_to_ticks() {
        assert_eq!(bcd_to_ticks(0x0000), 0);
        assert_eq!(bcd_to_ticks(0x0123), 123);
        assert_eq!(bcd_to_ticks(0x9999), 9999);
        assert_eq!(bcd_to_ticks(0x1000), 1000);
        // Invalid digits are taken at face value.
        assert_eq!(bcd_to_ticks(0x00ff), 165);
    }

    #[test]
    fn ticks_convert_to_bcd_digits() {
        assert_eq!(ticks_to_bcd(0), 0x0000);
        assert_eq!(ticks_to_bcd(123), 0x0123);
        assert_eq!(ticks_to_bcd(9999), 0x9999);
        assert_eq!(ticks_to_bcd(1000), 0x1000);
    }

    // Power-on state, control register and ports

    #[test]
    fn nothing_is_scheduled_at_power_on() {
        let mut pit = Pit::new();
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(u64::MAX));
    }

    #[test]
    fn out2_is_high_before_any_programming() {
        assert!(Pit::new().out2(0));
    }

    #[test]
    fn channel_gates_power_on_high_except_the_speaker() {
        let pit = Pit::new();
        assert!(pit.channels[0].gate);
        assert!(pit.channels[1].gate);
        assert!(!pit.channels[CHANNEL_SPEAKER].gate);
    }

    #[test]
    fn control_register_reads_open_bus() {
        let mut pit = Pit::new();
        assert_eq!(pit.read(CONTROL_PORT, 0), OPEN_BUS);
    }

    #[test]
    fn power_on_access_is_low_high_for_reads_writes_and_latches() {
        let mut pit = Pit::new();
        // Latch before any control word: both bytes are returned, like an unlatched read.
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LATCH, 0, 0), 0);
        assert_eq!(pit.read(CHANNEL0_PORT, 0), 0);
        assert!(pit.channels[0].latched_count.is_some());
        assert_eq!(pit.read(CHANNEL0_PORT, 0), 0);
        assert!(pit.channels[0].latched_count.is_none());
        // Writes take two bytes.
        pit.write(CHANNEL0_PORT, 0x10, 0);
        assert_eq!(pit.next_deadline(), None);
        pit.write(CHANNEL0_PORT, 0x00, 0);
        assert_eq!(pit.next_deadline(), Some(at(0x10)));
    }

    #[test]
    fn channels_are_independent() {
        let mut pit = Pit::new();
        program(&mut pit, 0, MODE_RATE_GENERATOR, 100, 0);
        program(&mut pit, 1, MODE_RATE_GENERATOR, 200, 0);
        pit.set_gate2(true, 0);
        program(&mut pit, 2, MODE_TERMINAL_COUNT, 300, 0);
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(10)), 90);
        assert_eq!(read_count(&mut pit, CHANNEL1_PORT, at(10)), 190);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(10)), 290);
        assert_eq!(pit.next_deadline(), Some(at(100)));
    }

    #[test]
    fn mirrored_register_ports_decode_to_the_same_channel() {
        let mut pit = Pit::new();
        pit.write(0x4b, control_word(0, ACCESS_LOW_HIGH, 2, 0), 0);
        pit.write(0x44, 100, 0);
        pit.write(0x44, 0, 0);
        assert_eq!(pit.next_deadline(), Some(at(100)));
    }

    // Access modes

    #[test]
    fn low_byte_only_access_loads_the_low_byte() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 50, 0);
        assert_eq!(pit.next_deadline(), Some(at(50)));
        assert_eq!(pit.read(CHANNEL0_PORT, at(10)), 40);
        assert_eq!(pit.read(CHANNEL0_PORT, at(10)), 40);
    }

    #[test]
    fn high_byte_only_access_loads_the_high_byte() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_HIGH, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0x01, 0);
        assert_eq!(pit.next_deadline(), Some(at(256)));
        // 256 - 10 = 246 = 0x00f6: the high byte is what a read returns.
        assert_eq!(pit.read(CHANNEL0_PORT, at(10)), 0);
        assert_eq!(pit.read(CHANNEL0_PORT, at(100)), 0);
        assert_eq!(pit.read(CHANNEL0_PORT, at(156)), 0);
    }

    #[test]
    fn high_byte_only_read_returns_the_high_byte_of_the_count() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_HIGH, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0x10, 0);
        // Count 4096 - 1 = 0x0fff.
        assert_eq!(pit.read(CHANNEL0_PORT, at(1)), 0x0f);
    }

    #[test]
    fn low_byte_only_zero_means_full_count() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0, 0);
        assert_eq!(pit.next_deadline(), Some(at(65_536)));
    }

    #[test]
    fn low_high_writes_alternate_bytes() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW_HIGH, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0x34, 0);
        assert_eq!(pit.next_deadline(), None);
        pit.write(CHANNEL0_PORT, 0x12, 0);
        assert_eq!(pit.next_deadline(), Some(at(0x1234)));
        // A further pair is a new count.
        pit.write(CHANNEL0_PORT, 0x00, 0);
        pit.write(CHANNEL0_PORT, 0x01, 0);
        assert_eq!(
            pit.channels[0].pending_reload.map(|(count, _)| count),
            Some(256)
        );
    }

    #[test]
    fn control_word_discards_a_half_written_count() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW_HIGH, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0xff, 0);
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW_HIGH, 2, 0), 0);
        pit.write(CHANNEL0_PORT, 0x10, 0);
        pit.write(CHANNEL0_PORT, 0x00, 0);
        assert_eq!(pit.next_deadline(), Some(at(16)));
    }

    #[test]
    fn unlatched_low_high_reads_alternate_and_follow_time() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 0xffff, true);
        // 0xffff - 1000 = 0xfc17.
        assert_eq!(pit.read(CHANNEL2_PORT, at(1000)), 0x17);
        assert_eq!(pit.read(CHANNEL2_PORT, at(1000)), 0xfc);
        assert_eq!(pit.read(CHANNEL2_PORT, at(1000)), 0x17);
        assert_eq!(pit.read(CHANNEL2_PORT, at(1000)), 0xfc);
    }

    #[test]
    fn control_word_restarts_the_read_byte_toggle() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 0x1234, true);
        let low = pit.read(CHANNEL2_PORT, 0);
        assert_eq!(low, 0x34);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LOW_HIGH, 0, 0), 0);
        write_count(&mut pit, CHANNEL2_PORT, 0x5678, 0);
        assert_eq!(pit.read(CHANNEL2_PORT, 0), 0x78);
        assert_eq!(pit.read(CHANNEL2_PORT, 0), 0x56);
    }

    #[test]
    fn linux_quick_calibration_polls_out2_and_reads_the_counter() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, 0xb0, 0);
        pit.write(CHANNEL2_PORT, 0xff, 0);
        pit.write(CHANNEL2_PORT, 0xff, 0);
        assert!(!pit.out2(at(100)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(100)), 0xffff - 100);
        assert!(!pit.out2(at(0xffff) - 1));
        assert!(pit.out2(at(0xffff)));
    }

    #[test]
    fn linux_pit_calibrate_tsc_latch_value() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, 0xb0, 0);
        pit.write(CHANNEL2_PORT, 0x9b, 0);
        pit.write(CHANNEL2_PORT, 0x2e, 0);
        assert!(!pit.out2(at(11_930)));
        assert!(!pit.out2(at(11_931) - 1));
        assert!(pit.out2(at(11_931)));
    }

    // Counter latch

    #[test]
    fn latch_command_freezes_the_count() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(10));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 990);
        // Latch consumed: reads are live again.
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 500);
    }

    #[test]
    fn second_latch_command_is_ignored_until_the_count_is_read() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        let latch = control_word(2, ACCESS_LATCH, 0, 0);
        pit.write(CONTROL_PORT, latch, at(10));
        pit.write(CONTROL_PORT, latch, at(20));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(30)), 990);
    }

    #[test]
    fn latch_is_ignored_while_only_the_low_byte_was_read() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        let latch = control_word(2, ACCESS_LATCH, 0, 0);
        pit.write(CONTROL_PORT, latch, at(10));
        assert_eq!(pit.read(CHANNEL2_PORT, at(15)), 990u16 as u8);
        pit.write(CONTROL_PORT, latch, at(20));
        assert_eq!(pit.read(CHANNEL2_PORT, at(25)), (990u16 >> 8) as u8);
        // Fully read; a new latch is accepted now.
        pit.write(CONTROL_PORT, latch, at(40));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(50)), 960);
    }

    #[test]
    fn latch_after_a_completed_read_captures_the_new_count() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        let latch = control_word(2, ACCESS_LATCH, 0, 0);
        pit.write(CONTROL_PORT, latch, at(10));
        read_count(&mut pit, CHANNEL2_PORT, at(11));
        pit.write(CONTROL_PORT, latch, at(100));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(200)), 900);
    }

    #[test]
    fn latch_on_a_low_only_channel_returns_one_byte() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LOW, 0, 0), 0);
        pit.write(CHANNEL2_PORT, 200, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(50));
        assert_eq!(pit.read(CHANNEL2_PORT, at(100)), 150);
        // Live again: 200 - 100.
        assert_eq!(pit.read(CHANNEL2_PORT, at(100)), 100);
    }

    #[test]
    fn latch_on_a_high_only_channel_returns_the_high_byte() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_HIGH, 0, 0), 0);
        pit.write(CHANNEL2_PORT, 0x20, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(1));
        assert_eq!(pit.read(CHANNEL2_PORT, at(100)), 0x1f);
        assert!(pit.channels[2].latched_count.is_none());
    }

    #[test]
    fn latching_does_not_change_mode_or_stop_counting() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 100, true);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(10));
        assert_eq!(pit.channels[2].mode, MODE_RATE_GENERATOR);
        assert!(pit.channels[2].running);
        read_count(&mut pit, CHANNEL2_PORT, at(11));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(60)), 40);
    }

    // Read-back command

    const READ_BACK_WORD: u8 = (CONTROL_READ_BACK as u8) << CONTROL_CHANNEL_SHIFT;

    #[test]
    fn read_back_latches_count_of_selected_channels_only() {
        let mut pit = Pit::new();
        program(&mut pit, 0, MODE_RATE_GENERATOR, 1000, 0);
        program(&mut pit, 1, MODE_RATE_GENERATOR, 2000, 0);
        // Count only, channels 0 and 2 (channel 2 never programmed).
        pit.write(
            CONTROL_PORT,
            READ_BACK_WORD | READ_BACK_NO_STATUS | READ_BACK_CHANNEL0 | (READ_BACK_CHANNEL0 << 2),
            at(10),
        );
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(500)), 990);
        assert_eq!(read_count(&mut pit, CHANNEL1_PORT, at(500)), 1500);
        assert!(
            pit.channels[2].latched_count.is_none() || pit.channels[2].latched_count == Some(0)
        );
    }

    #[test]
    fn read_back_of_all_three_channels_latches_each() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        for channel in 0..3 {
            program(
                &mut pit,
                channel,
                MODE_TERMINAL_COUNT,
                1000 * (channel as u16 + 1),
                0,
            );
        }
        pit.write(
            CONTROL_PORT,
            READ_BACK_WORD | READ_BACK_NO_STATUS | 0x0e,
            at(10),
        );
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(900)), 990);
        assert_eq!(read_count(&mut pit, CHANNEL1_PORT, at(900)), 1990);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(900)), 2990);
    }

    #[test]
    fn read_back_with_both_inhibit_bits_latches_nothing() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        read_back(
            &mut pit,
            2,
            READ_BACK_NO_COUNT | READ_BACK_NO_STATUS,
            at(10),
        );
        assert!(pit.channels[2].latched_count.is_none());
        assert!(pit.channels[2].latched_status.is_none());
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(20)), 980);
    }

    #[test]
    fn read_back_does_not_overwrite_a_pending_latch() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(10));
        read_back(&mut pit, 2, READ_BACK_NO_STATUS, at(20));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(30)), 990);
    }

    #[test]
    fn plain_latch_does_not_overwrite_a_read_back_latch() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        read_back(&mut pit, 2, READ_BACK_NO_STATUS, at(10));
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(20));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(30)), 990);
    }

    #[test]
    fn read_back_status_comes_before_the_latched_count() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        read_back(&mut pit, 2, 0, at(10));
        let status = pit.read(CHANNEL2_PORT, at(50));
        assert_eq!(
            status,
            control_word(0, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 0)
        );
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(50)), 990);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(50)), 950);
    }

    #[test]
    fn read_back_status_is_returned_once() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 1000, true);
        let status = status_of(&mut pit, 2, at(10));
        assert_eq!(status & !STATUS_OUT & !STATUS_NULL_COUNT, 0x30);
        // The next read is the live count, not the status again.
        assert_eq!(pit.read(CHANNEL2_PORT, at(20)), (1000u16 - 20) as u8);
    }

    #[test]
    fn second_status_latch_is_ignored_until_read() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        // OUT is low at tick 10 and high at tick 200.
        read_back(&mut pit, 2, READ_BACK_NO_COUNT, at(10));
        read_back(&mut pit, 2, READ_BACK_NO_COUNT, at(200));
        assert_eq!(pit.read(CHANNEL2_PORT, at(200)) & STATUS_OUT, 0);
        read_back(&mut pit, 2, READ_BACK_NO_COUNT, at(200));
        assert_eq!(pit.read(CHANNEL2_PORT, at(200)) & STATUS_OUT, STATUS_OUT);
    }

    #[test]
    fn status_reports_access_mode_and_bcd_bits() {
        for (access, mode, bcd) in [
            (ACCESS_LOW, 0, 0),
            (ACCESS_HIGH, 1, 0),
            (ACCESS_LOW_HIGH, 2, 1),
            (ACCESS_LOW_HIGH, 3, 0),
            (ACCESS_LOW, 4, 1),
            (ACCESS_HIGH, 5, 0),
        ] {
            let mut pit = Pit::new();
            pit.write(CONTROL_PORT, control_word(1, access, mode, bcd), 0);
            let status = status_of(&mut pit, 1, 0);
            assert_eq!((status >> STATUS_ACCESS_SHIFT) & 3, access);
            assert_eq!((status >> STATUS_MODE_SHIFT) & 7, mode);
            assert_eq!(status & 1, bcd);
        }
    }

    #[test]
    fn status_reports_the_mode_alias_bits_as_programmed() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW_HIGH, 6, 0), 0);
        assert_eq!((status_of(&mut pit, 0, 0) >> STATUS_MODE_SHIFT) & 7, 6);
        pit.write(CONTROL_PORT, control_word(0, ACCESS_LOW_HIGH, 7, 0), 0);
        assert_eq!((status_of(&mut pit, 0, 0) >> STATUS_MODE_SHIFT) & 7, 7);
    }

    #[test]
    fn status_null_count_is_set_until_the_count_is_loaded() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            0,
        );
        assert_eq!(
            status_of(&mut pit, 2, 0) & STATUS_NULL_COUNT,
            STATUS_NULL_COUNT
        );
        pit.write(CHANNEL2_PORT, 10, 0);
        // Half-written count: still null.
        assert_eq!(
            status_of(&mut pit, 2, 0) & STATUS_NULL_COUNT,
            STATUS_NULL_COUNT
        );
        pit.write(CHANNEL2_PORT, 0, 0);
        assert_eq!(status_of(&mut pit, 2, 0) & STATUS_NULL_COUNT, 0);
    }

    #[test]
    fn status_null_count_is_set_again_by_a_new_control_word() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        assert_eq!(status_of(&mut pit, 2, 0) & STATUS_NULL_COUNT, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            0,
        );
        assert_eq!(
            status_of(&mut pit, 2, 0) & STATUS_NULL_COUNT,
            STATUS_NULL_COUNT
        );
    }

    #[test]
    fn status_out_matches_the_pin_in_each_mode() {
        // Mode 0: low until terminal count.
        let mut pit = channel2(MODE_TERMINAL_COUNT, 10, true);
        assert_eq!(status_of(&mut pit, 2, at(9)) & STATUS_OUT, 0);
        assert_eq!(status_of(&mut pit, 2, at(10)) & STATUS_OUT, STATUS_OUT);
        // Mode 2: high except for the last tick of the period.
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        assert_eq!(status_of(&mut pit, 2, at(8)) & STATUS_OUT, STATUS_OUT);
        assert_eq!(status_of(&mut pit, 2, at(9)) & STATUS_OUT, 0);
    }

    #[test]
    fn status_of_a_mode_0_channel_without_a_count_has_out_low() {
        let mut pit = Pit::new();
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 0),
            0,
        );
        assert_eq!(status_of(&mut pit, 0, 0), STATUS_NULL_COUNT | 0x30);
    }

    #[test]
    fn status_of_a_mode_2_channel_without_a_count_has_out_high() {
        let mut pit = Pit::new();
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            0,
        );
        assert_eq!(
            status_of(&mut pit, 0, 0),
            STATUS_OUT | STATUS_NULL_COUNT | 0x30 | (MODE_RATE_GENERATOR << 1)
        );
    }

    #[test]
    fn status_of_a_running_rate_generator_is_exact() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        assert_eq!(status_of(&mut pit, 0, at(5)), 0x80 | 0x30 | 0x04);
    }

    // Mode 0: interrupt on terminal count

    #[test]
    fn mode0_out_goes_high_exactly_at_terminal_count() {
        let pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        assert!(!pit.out2(0));
        assert!(!pit.out2(at(99)));
        assert!(!pit.out2(at(100) - 1));
        assert!(pit.out2(at(100)));
        assert!(pit.out2(at(1000)));
    }

    #[test]
    fn mode0_counts_down_then_wraps_through_zero() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, 0), 100);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(40)), 60);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(100)), 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(101)), 0xffff);
    }

    #[test]
    fn mode0_count_zero_means_65536() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 0, true);
        assert!(!pit.out2(at(65_535)));
        assert!(pit.out2(at(65_536)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1)), 0xffff);
    }

    #[test]
    fn mode0_waits_for_the_gate() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, false);
        assert!(!pit.out2(at(1000)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1000)), 100);
        pit.set_gate2(true, at(1000));
        assert!(!pit.out2(at(1000) + at(99)));
        assert!(pit.out2(at(1000) + at(100)));
    }

    #[test]
    fn mode0_gate_low_freezes_and_gate_high_resumes() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        pit.set_gate2(false, at(30));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 70);
        assert!(!pit.out2(at(500)));
        pit.set_gate2(true, at(500));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500) + at(20)), 50);
        assert!(!pit.out2(at(500) + at(69)));
        assert!(pit.out2(at(500) + at(70)));
    }

    #[test]
    fn mode0_out_stays_high_when_the_gate_drops_after_terminal_count() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        assert!(pit.out2(at(150)));
        pit.set_gate2(false, at(150));
        assert!(pit.out2(at(400)));
        pit.set_gate2(true, at(400));
        assert!(pit.out2(at(450)));
    }

    #[test]
    fn mode0_writing_a_count_drops_out_immediately_and_restarts() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        assert!(pit.out2(at(150)));
        // First byte of the new count: OUT goes low at once, counting stops.
        pit.write(CHANNEL2_PORT, 50, at(150));
        assert!(!pit.out2(at(150)));
        assert!(!pit.out2(at(10_000)));
        // Second byte: counting restarts from the new count.
        pit.write(CHANNEL2_PORT, 0, at(200));
        assert!(!pit.out2(at(200) + at(49)));
        assert!(pit.out2(at(200) + at(50)));
    }

    #[test]
    fn mode0_first_count_byte_freezes_the_counter() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        pit.write(CHANNEL2_PORT, 0x80, at(30));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 70);
    }

    #[test]
    fn mode0_on_channel0_raises_irq_once_at_terminal_count() {
        let mut pit = channel0(MODE_TERMINAL_COUNT, 100);
        assert_eq!(pit.next_deadline(), Some(at(100)));
        assert!(!pit.tick(at(100) - 1));
        assert!(pit.tick(at(100)));
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(1_000_000)));
        // Writing a new count re-arms it.
        write_count(&mut pit, CHANNEL0_PORT, 20, at(2000));
        assert_eq!(pit.next_deadline(), Some(at(2000) + at(20)));
    }

    #[test]
    fn mode0_new_control_word_cancels_a_pending_irq() {
        let mut pit = channel0(MODE_TERMINAL_COUNT, 100);
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 0),
            at(10),
        );
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(1000)));
    }

    // Mode 1: hardware retriggerable one-shot

    #[test]
    fn mode1_is_high_until_triggered() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        assert!(pit.out2(0));
        assert!(pit.out2(at(1000)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1000)), 10);
    }

    #[test]
    fn mode1_gate_rising_edge_triggers_a_low_pulse_of_count_ticks() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        let trigger = at(100);
        pit.set_gate2(true, trigger);
        assert!(!pit.out2(trigger));
        assert!(!pit.out2(trigger + at(9)));
        assert!(!pit.out2(trigger + at(10) - 1));
        assert!(pit.out2(trigger + at(10)));
        assert!(pit.out2(trigger + at(1000)));
    }

    #[test]
    fn mode1_counts_down_from_the_trigger() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        let trigger = at(100);
        pit.set_gate2(true, trigger);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, trigger + at(4)), 6);
    }

    #[test]
    fn mode1_gate_falling_does_not_stop_the_pulse() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        let trigger = at(100);
        pit.set_gate2(true, trigger);
        pit.set_gate2(false, trigger + at(4));
        assert!(!pit.out2(trigger + at(9)));
        assert!(pit.out2(trigger + at(10)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, trigger + at(8)), 2);
    }

    #[test]
    fn mode1_retrigger_restarts_the_pulse() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        let first = at(100);
        pit.set_gate2(true, first);
        pit.set_gate2(false, first + at(4));
        let second = first + at(5);
        pit.set_gate2(true, second);
        assert!(!pit.out2(first + at(10)));
        assert!(!pit.out2(second + at(9)));
        assert!(pit.out2(second + at(10)));
    }

    #[test]
    fn mode1_retrigger_after_the_pulse_ended_pulls_out_low_again() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        pit.set_gate2(true, at(100));
        pit.set_gate2(false, at(200));
        assert!(pit.out2(at(200)));
        pit.set_gate2(true, at(300));
        assert!(!pit.out2(at(305)));
        assert!(pit.out2(at(310)));
    }

    #[test]
    fn mode1_new_count_waits_for_the_next_trigger() {
        let mut pit = channel2(MODE_ONE_SHOT, 10, false);
        let trigger = at(100);
        pit.set_gate2(true, trigger);
        write_count(&mut pit, CHANNEL2_PORT, 20, trigger + at(3));
        // The pulse in progress still ends after 10 ticks.
        assert!(!pit.out2(trigger + at(9)));
        assert!(pit.out2(trigger + at(10)));
        assert_eq!(
            status_of(&mut pit, 2, trigger + at(10)) & STATUS_NULL_COUNT,
            STATUS_NULL_COUNT
        );
        pit.set_gate2(false, trigger + at(50));
        let second = trigger + at(60);
        pit.set_gate2(true, second);
        assert!(!pit.out2(second + at(19)));
        assert!(pit.out2(second + at(20)));
        assert_eq!(
            status_of(&mut pit, 2, second + at(21)) & STATUS_NULL_COUNT,
            0
        );
    }

    #[test]
    fn mode1_count_written_while_gate_high_still_needs_a_rising_edge() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        program(&mut pit, 2, MODE_ONE_SHOT, 10, 0);
        assert!(pit.out2(at(5)));
        assert!(pit.out2(at(50)));
    }

    #[test]
    fn mode1_on_channel0_never_triggers_because_its_gate_is_tied_high() {
        let mut pit = channel0(MODE_ONE_SHOT, 10);
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(1000)));
    }

    #[test]
    fn mode1_count_zero_means_65536() {
        let mut pit = channel2(MODE_ONE_SHOT, 0, false);
        pit.set_gate2(true, 0);
        assert!(!pit.out2(at(65_535)));
        assert!(pit.out2(at(65_536)));
    }

    // Mode 2: rate generator

    #[test]
    fn mode2_out_is_low_for_the_last_tick_of_each_period() {
        let pit = channel2(MODE_RATE_GENERATOR, 10, true);
        assert!(pit.out2(0));
        assert!(pit.out2(at(8)));
        assert!(!pit.out2(at(9)));
        assert!(!pit.out2(at(10) - 1));
        assert!(pit.out2(at(10)));
        assert!(pit.out2(at(18)));
        assert!(!pit.out2(at(19)));
        assert!(pit.out2(at(20)));
    }

    #[test]
    fn mode2_count_cycles_from_reload_down_to_one() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, 0), 10);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1)), 9);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(9)), 1);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(10)), 10);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(25)), 5);
    }

    #[test]
    fn mode2_count_zero_means_65536() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 0, true);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, 0), 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1)), 0xffff);
        assert!(!pit.out2(at(65_535)));
        assert!(pit.out2(at(65_536)));
    }

    #[test]
    fn mode2_gate_low_forces_out_high_and_freezes_the_count() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        pit.set_gate2(false, at(9));
        assert!(pit.out2(at(9)));
        assert!(pit.out2(at(500)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 1);
    }

    #[test]
    fn mode2_gate_rising_edge_restarts_from_the_reload_value() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        pit.set_gate2(false, at(3));
        pit.set_gate2(true, at(50));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(50)), 10);
        assert!(pit.out2(at(50) + at(8)));
        assert!(!pit.out2(at(50) + at(9)));
        assert!(pit.out2(at(50) + at(10)));
    }

    #[test]
    fn mode2_count_written_while_the_gate_is_low_loads_on_the_rising_edge() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, false);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(5)), 10);
        pit.set_gate2(true, at(100));
        write_count(&mut pit, CHANNEL2_PORT, 20, at(101));
        // Still counting the old period; the new one starts at its end.
        assert!(!pit.out2(at(100) + at(9)));
        assert!(pit.out2(at(100) + at(10)));
        assert!(pit.out2(at(100) + at(28)));
        assert!(!pit.out2(at(100) + at(29)));
        assert!(pit.out2(at(100) + at(30)));
    }

    #[test]
    fn mode2_on_channel0_raises_irq_every_period_from_a_fixed_anchor() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 1193);
        for periods in 1..=5u64 {
            let deadline = at(periods * 1193);
            assert_eq!(pit.next_deadline(), Some(deadline));
            assert!(!pit.tick(deadline - 1));
            assert!(pit.tick(deadline));
            assert!(!pit.tick(deadline));
        }
    }

    #[test]
    fn mode2_deadlines_do_not_accumulate_rounding_error() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 11_932);
        let period_ns = ticks_to_ns(11_932);
        assert_eq!(period_ns, 10_000_151);
        let mut fired = 0u64;
        while fired < 1000 {
            let deadline = pit.next_deadline().unwrap();
            assert!(pit.tick(deadline));
            fired += 1;
            assert_eq!(deadline, ticks_to_ns(fired * 11_932));
        }
        // Summing the rounded period would be 1000 * 10_000_151 ns, which is later.
        assert!(ticks_to_ns(1000 * 11_932) < 1000 * period_ns);
    }

    #[test]
    fn mode2_late_tick_yields_one_irq_and_skips_missed_periods() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        let late = at(550) + 7;
        assert!(pit.tick(late));
        assert_eq!(pit.next_deadline(), Some(at(600)));
        assert!(!pit.tick(late));
    }

    #[test]
    fn mode2_tick_exactly_on_a_later_boundary_moves_past_it() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        assert!(pit.tick(at(500)));
        assert_eq!(pit.next_deadline(), Some(at(600)));
    }

    #[test]
    fn mode2_count_written_mid_period_takes_effect_at_the_period_end() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        write_count(&mut pit, CHANNEL0_PORT, 50, at(30));
        // The running period still ends at tick 100.
        assert_eq!(pit.next_deadline(), Some(at(100)));
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(60)), 40);
        assert!(pit.tick(at(100)));
        assert_eq!(pit.next_deadline(), Some(at(100) + at(50)));
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(100) + at(20)), 30);
    }

    #[test]
    fn mode2_pending_count_applies_on_read_even_without_a_tick() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        write_count(&mut pit, CHANNEL0_PORT, 50, at(30));
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(100) + at(20)), 30);
    }

    #[test]
    fn mode2_second_mid_period_write_replaces_the_pending_count() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        write_count(&mut pit, CHANNEL0_PORT, 50, at(30));
        write_count(&mut pit, CHANNEL0_PORT, 70, at(40));
        assert!(pit.tick(at(100)));
        assert_eq!(pit.next_deadline(), Some(at(100) + at(70)));
    }

    #[test]
    fn mode2_pending_count_shows_as_null_count_until_applied() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        assert_eq!(status_of(&mut pit, 0, at(10)) & STATUS_NULL_COUNT, 0);
        write_count(&mut pit, CHANNEL0_PORT, 50, at(30));
        assert_eq!(
            status_of(&mut pit, 0, at(40)) & STATUS_NULL_COUNT,
            STATUS_NULL_COUNT
        );
        assert_eq!(status_of(&mut pit, 0, at(110)) & STATUS_NULL_COUNT, 0);
    }

    #[test]
    fn mode2_alias_6_behaves_as_mode_2() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x3c, 0);
        write_count(&mut pit, CHANNEL0_PORT, 100, 0);
        assert_eq!(pit.next_deadline(), Some(at(100)));
        assert!(pit.tick(at(100)));
        assert_eq!(pit.next_deadline(), Some(at(200)));
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LOW_HIGH, 6, 0), 0);
        write_count(&mut pit, CHANNEL2_PORT, 10, 0);
        assert!(!pit.out2(at(9)));
        assert!(pit.out2(at(10)));
    }

    #[test]
    fn mode2_count_of_one_does_not_panic_and_reads_high() {
        let pit = channel2(MODE_RATE_GENERATOR, 1, true);
        assert!(pit.out2(at(5)));
        let mut pit = channel0(MODE_RATE_GENERATOR, 1);
        assert_eq!(pit.next_deadline(), Some(at(1)));
        assert!(pit.tick(at(1)));
    }

    // Mode 3: square wave

    #[test]
    fn mode3_even_count_is_symmetric() {
        let pit = channel2(MODE_SQUARE_WAVE, 10, true);
        assert!(pit.out2(0));
        assert!(pit.out2(at(4)));
        assert!(!pit.out2(at(5)));
        assert!(!pit.out2(at(9)));
        assert!(pit.out2(at(10)));
        assert!(pit.out2(at(14)));
        assert!(!pit.out2(at(15)));
    }

    #[test]
    fn mode3_odd_count_is_high_one_tick_longer() {
        let pit = channel2(MODE_SQUARE_WAVE, 9, true);
        assert!(pit.out2(at(4)));
        assert!(!pit.out2(at(5)));
        assert!(!pit.out2(at(8)));
        assert!(pit.out2(at(9)));
        assert!(pit.out2(at(13)));
        assert!(!pit.out2(at(14)));
    }

    #[test]
    fn mode3_count_of_three_has_two_high_ticks_and_one_low() {
        let pit = channel2(MODE_SQUARE_WAVE, 3, true);
        assert!(pit.out2(0));
        assert!(pit.out2(at(1)));
        assert!(!pit.out2(at(2)));
        assert!(pit.out2(at(3)));
    }

    #[test]
    fn mode3_count_zero_means_65536() {
        let pit = channel2(MODE_SQUARE_WAVE, 0, true);
        assert!(pit.out2(at(32_767)));
        assert!(!pit.out2(at(32_768)));
        assert!(!pit.out2(at(65_535)));
        assert!(pit.out2(at(65_536)));
    }

    #[test]
    fn mode3_gate_low_forces_out_high() {
        let mut pit = channel2(MODE_SQUARE_WAVE, 10, true);
        assert!(!pit.out2(at(7)));
        pit.set_gate2(false, at(7));
        assert!(pit.out2(at(7)));
        assert!(pit.out2(at(100)));
    }

    #[test]
    fn mode3_gate_rising_edge_restarts_the_period() {
        let mut pit = channel2(MODE_SQUARE_WAVE, 10, true);
        pit.set_gate2(false, at(7));
        pit.set_gate2(true, at(100));
        assert!(pit.out2(at(100) + at(4)));
        assert!(!pit.out2(at(100) + at(5)));
        assert!(pit.out2(at(100) + at(10)));
    }

    #[test]
    fn mode3_on_channel0_irqs_at_each_period_end() {
        let mut pit = channel0(MODE_SQUARE_WAVE, 2000);
        for periods in 1..=4u64 {
            assert_eq!(pit.next_deadline(), Some(at(periods * 2000)));
            assert!(pit.tick(at(periods * 2000)));
        }
    }

    #[test]
    fn mode3_new_count_restarts_immediately() {
        let mut pit = channel0(MODE_SQUARE_WAVE, 100);
        write_count(&mut pit, CHANNEL0_PORT, 50, at(30));
        assert_eq!(pit.next_deadline(), Some(at(30) + at(50)));
    }

    #[test]
    fn mode3_alias_7_behaves_as_mode_3() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LOW_HIGH, 7, 0), 0);
        write_count(&mut pit, CHANNEL2_PORT, 10, 0);
        assert!(pit.out2(at(4)));
        assert!(!pit.out2(at(5)));
        assert!(pit.out2(at(10)));
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x3e, 0);
        write_count(&mut pit, CHANNEL0_PORT, 40, 0);
        assert_eq!(pit.next_deadline(), Some(at(40)));
    }

    // Mode 4: software triggered strobe

    #[test]
    fn mode4_out_is_low_for_exactly_one_tick_at_terminal_count() {
        let pit = channel2(MODE_SOFTWARE_STROBE, 10, true);
        assert!(pit.out2(0));
        assert!(pit.out2(at(9)));
        assert!(!pit.out2(at(10)));
        assert!(!pit.out2(at(11) - 1));
        assert!(pit.out2(at(11)));
        assert!(pit.out2(at(1000)));
    }

    #[test]
    fn mode4_counter_wraps_after_the_strobe() {
        let mut pit = channel2(MODE_SOFTWARE_STROBE, 10, true);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(4)), 6);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(10)), 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(11)), 0xffff);
    }

    #[test]
    fn mode4_gate_low_freezes_and_gate_high_resumes() {
        let mut pit = channel2(MODE_SOFTWARE_STROBE, 10, true);
        pit.set_gate2(false, at(5));
        assert!(pit.out2(at(500)));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(500)), 5);
        pit.set_gate2(true, at(500));
        assert!(pit.out2(at(500) + at(4)));
        assert!(!pit.out2(at(500) + at(5)));
        assert!(pit.out2(at(500) + at(6)));
    }

    #[test]
    fn mode4_on_channel0_arms_irq_at_the_end_of_the_strobe() {
        // Linux programs its one-shot clock event as control word 0x38 and a count; the guest
        // hangs if IRQ 0 never fires (commit 763dd7e).
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x38, 0);
        pit.write(CHANNEL0_PORT, 0x10, 0);
        pit.write(CHANNEL0_PORT, 0x00, 0);
        assert_eq!(pit.next_deadline(), Some(at(17)));
        assert!(!pit.tick(at(17) - 1));
        assert!(pit.tick(at(17)));
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(100)));
    }

    #[test]
    fn mode4_one_shot_is_rearmed_by_each_new_count() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x38, 0);
        for round in 0..3u64 {
            let now = at(10_000) * round;
            write_count(&mut pit, CHANNEL0_PORT, 100, now);
            assert_eq!(pit.next_deadline(), Some(now + at(101)));
            assert!(pit.tick(now + at(101)));
            assert_eq!(pit.next_deadline(), None);
        }
    }

    #[test]
    fn mode4_first_count_byte_stops_the_counter_until_the_second_arrives() {
        let mut pit = channel0(MODE_SOFTWARE_STROBE, 100);
        assert!(pit.next_deadline().is_some());
        pit.write(CHANNEL0_PORT, 50, at(5));
        assert_eq!(pit.next_deadline(), None);
        pit.write(CHANNEL0_PORT, 0, at(8));
        assert_eq!(pit.next_deadline(), Some(at(8) + at(51)));
    }

    #[test]
    fn mode4_count_written_while_counting_restarts_it() {
        let mut pit = channel0(MODE_SOFTWARE_STROBE, 100);
        write_count(&mut pit, CHANNEL0_PORT, 30, at(20));
        assert_eq!(pit.next_deadline(), Some(at(20) + at(31)));
    }

    #[test]
    fn mode4_count_zero_means_65536() {
        let pit = channel0(MODE_SOFTWARE_STROBE, 0);
        assert_eq!(pit.next_deadline(), Some(at(65_537)));
    }

    // Mode 5: hardware triggered strobe

    #[test]
    fn mode5_is_high_until_triggered_then_strobes_low_for_one_tick() {
        let mut pit = channel2(MODE_HARDWARE_STROBE, 10, false);
        assert!(pit.out2(at(500)));
        let trigger = at(1000);
        pit.set_gate2(true, trigger);
        assert!(pit.out2(trigger));
        assert!(pit.out2(trigger + at(9)));
        assert!(!pit.out2(trigger + at(10)));
        assert!(pit.out2(trigger + at(11)));
    }

    #[test]
    fn mode5_retrigger_restarts_the_count() {
        let mut pit = channel2(MODE_HARDWARE_STROBE, 10, false);
        pit.set_gate2(true, at(100));
        pit.set_gate2(false, at(104));
        pit.set_gate2(true, at(106));
        assert!(pit.out2(at(100) + at(10)));
        assert!(!pit.out2(at(106) + at(10)));
    }

    #[test]
    fn mode5_on_channel0_never_triggers() {
        let mut pit = channel0(MODE_HARDWARE_STROBE, 10);
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(1000)));
    }

    // BCD counting

    #[test]
    fn bcd_count_is_read_back_as_digits() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 1),
            0,
        );
        write_count(&mut pit, CHANNEL2_PORT, 0x0100, 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, 0), 0x0100);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1)), 0x0099);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(30)), 0x0070);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(99)), 0x0001);
    }

    #[test]
    fn bcd_terminal_count_is_the_decimal_value() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 1),
            0,
        );
        write_count(&mut pit, CHANNEL2_PORT, 0x0100, 0);
        assert!(!pit.out2(at(99)));
        assert!(pit.out2(at(100)));
    }

    #[test]
    fn bcd_counter_wraps_to_9999() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 1),
            0,
        );
        write_count(&mut pit, CHANNEL2_PORT, 0x0100, 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(100)), 0x0000);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(101)), 0x9999);
    }

    #[test]
    fn bcd_count_zero_means_10000() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 1),
            0,
        );
        write_count(&mut pit, CHANNEL2_PORT, 0, 0);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1)), 0x9999);
        assert!(!pit.out2(at(9_999)));
        assert!(pit.out2(at(10_000)));
    }

    #[test]
    fn bcd_rate_generator_cycles_through_decimal_counts() {
        let mut pit = Pit::new();
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 1),
            0,
        );
        write_count(&mut pit, CHANNEL0_PORT, 0x0050, 0);
        assert_eq!(pit.next_deadline(), Some(at(50)));
        assert_eq!(read_count(&mut pit, CHANNEL0_PORT, at(10)), 0x0040);
        assert!(pit.tick(at(50)));
        assert_eq!(pit.next_deadline(), Some(at(100)));
    }

    #[test]
    fn bcd_status_bit_is_set() {
        let mut pit = Pit::new();
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 1),
            0,
        );
        write_count(&mut pit, CHANNEL0_PORT, 0x0050, 0);
        assert_eq!(status_of(&mut pit, 0, at(1)) & 1, 1);
    }

    #[test]
    fn bcd_latched_count_is_in_digits() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 0);
        pit.write(
            CONTROL_PORT,
            control_word(2, ACCESS_LOW_HIGH, MODE_TERMINAL_COUNT, 1),
            0,
        );
        write_count(&mut pit, CHANNEL2_PORT, 0x1234, 0);
        pit.write(CONTROL_PORT, control_word(2, ACCESS_LATCH, 0, 0), at(34));
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1000)), 0x1200);
    }

    #[test]
    fn a_binary_control_word_after_a_bcd_one_returns_to_binary() {
        let mut pit = Pit::new();
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 1),
            0,
        );
        write_count(&mut pit, CHANNEL0_PORT, 0x0100, 0);
        assert_eq!(pit.next_deadline(), Some(at(100)));
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            0,
        );
        write_count(&mut pit, CHANNEL0_PORT, 0x0100, 0);
        assert_eq!(pit.next_deadline(), Some(at(256)));
    }

    // Channel 0 scheduling

    #[test]
    fn linux_periodic_100hz_schedule() {
        let mut pit = Pit::new();
        pit.write(CONTROL_PORT, 0x34, 0);
        pit.write(CHANNEL0_PORT, (11_932u16 & 0xff) as u8, 0);
        pit.write(CHANNEL0_PORT, (11_932u16 >> 8) as u8, 0);
        assert_eq!(pit.next_deadline(), Some(10_000_151));
        assert!(pit.tick(10_000_151));
        assert_eq!(pit.next_deadline(), Some(ticks_to_ns(2 * 11_932)));
    }

    #[test]
    fn reprogramming_channel0_cancels_the_deadline_until_a_count_is_written() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        assert!(pit.next_deadline().is_some());
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            at(5),
        );
        assert_eq!(pit.next_deadline(), None);
        write_count(&mut pit, CHANNEL0_PORT, 20, at(6));
        assert_eq!(pit.next_deadline(), Some(at(6) + at(20)));
    }

    #[test]
    fn channel1_and_channel2_programming_never_schedules_irq0() {
        let mut pit = Pit::new();
        program(&mut pit, 1, MODE_RATE_GENERATOR, 100, 0);
        pit.set_gate2(true, 0);
        program(&mut pit, 2, MODE_RATE_GENERATOR, 100, 0);
        assert_eq!(pit.next_deadline(), None);
        assert!(!pit.tick(at(1000)));
    }

    #[test]
    fn programming_channel1_does_not_disturb_channel0_deadline() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 100);
        program(&mut pit, 1, MODE_RATE_GENERATOR, 7, at(3));
        assert_eq!(pit.next_deadline(), Some(at(100)));
    }

    #[test]
    fn state_reports_channel0_programming() {
        let mut pit = channel0(MODE_RATE_GENERATOR, 1193);
        assert_eq!(
            pit.state(5),
            format!(
                "pit0 mode 2 reload 1193 next_irq {:?} now 5",
                Some(at(1193))
            )
        );
        pit.tick(at(1193));
        assert!(pit.state(7).contains("now 7"));
    }

    // Gate

    #[test]
    fn repeated_gate_writes_with_the_same_level_do_nothing() {
        let mut pit = channel2(MODE_RATE_GENERATOR, 10, true);
        pit.set_gate2(true, at(6));
        // A redundant "high" must not restart the period.
        assert!(!pit.out2(at(9)));
        pit.set_gate2(false, at(20));
        pit.set_gate2(false, at(30));
        // Frozen at the period boundary (tick 20), where the count has just reloaded to 10.
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(40)), 10);
    }

    #[test]
    fn gate_changes_before_a_count_is_loaded_only_record_the_level() {
        let mut pit = Pit::new();
        pit.set_gate2(true, 5);
        pit.set_gate2(false, 6);
        pit.set_gate2(true, 7);
        program(&mut pit, 2, MODE_RATE_GENERATOR, 10, at(100));
        assert!(!pit.out2(at(100) + at(9)));
        assert!(pit.out2(at(100) + at(10)));
    }

    #[test]
    fn programming_while_gate_is_low_does_not_count() {
        let mut pit = channel2(MODE_SQUARE_WAVE, 10, false);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, at(1000)), 10);
        assert!(pit.out2(at(1000)));
    }

    #[test]
    fn channels_0_and_1_ignore_the_speaker_gate() {
        let mut pit = Pit::new();
        program(&mut pit, 1, MODE_RATE_GENERATOR, 100, 0);
        pit.set_gate2(false, at(10));
        pit.set_gate2(true, at(20));
        assert_eq!(read_count(&mut pit, CHANNEL1_PORT, at(30)), 70);
    }

    // Overflow behaviour at the end of time

    #[test]
    fn deadline_near_the_end_of_time_saturates() {
        let mut pit = Pit::new();
        let late = u64::MAX - 10;
        pit.write(
            CONTROL_PORT,
            control_word(0, ACCESS_LOW_HIGH, MODE_RATE_GENERATOR, 0),
            late,
        );
        write_count(&mut pit, CHANNEL0_PORT, 100, late);
        assert_eq!(pit.next_deadline(), Some(u64::MAX));
        assert!(!pit.tick(u64::MAX - 1));
        assert!(pit.tick(u64::MAX));
        // No edge can be later than the last representable instant.
        assert_eq!(pit.next_deadline(), None);
    }

    #[test]
    fn reading_the_counter_at_the_end_of_time_does_not_panic() {
        let mut pit = channel2(MODE_TERMINAL_COUNT, 100, true);
        assert_eq!(read_count(&mut pit, CHANNEL2_PORT, u64::MAX), 0x6d76);
        let mut pit = channel2(MODE_RATE_GENERATOR, 7, true);
        let count = read_count(&mut pit, CHANNEL2_PORT, u64::MAX);
        assert!((1..=7).contains(&count));
    }

    #[test]
    fn full_count_programs_stay_in_range_for_every_mode() {
        for mode in 0..8 {
            let mut pit = channel2(mode, 0, true);
            let count = read_count(&mut pit, CHANNEL2_PORT, at(12_345));
            assert_ne!(count, 12_345, "mode {mode}");
            pit.out2(at(12_345));
        }
    }
}
