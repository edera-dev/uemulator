#[cfg(freestanding)]
use crate::prelude::*;
use crate::snapshot::Snapshot;

const MASTER_COMMAND_PORT: u16 = 0x20;
const MASTER_DATA_PORT: u16 = 0x21;
const SLAVE_COMMAND_PORT: u16 = 0xa0;
const SLAVE_DATA_PORT: u16 = 0xa1;
/// Edge/level control registers (one bit per IRQ, 1 = level triggered).
const MASTER_ELCR_PORT: u16 = 0x4d0;
const SLAVE_ELCR_PORT: u16 = 0x4d1;
/// ELCR bits that exist: IRQ 0, 1, 2 (timer, keyboard, cascade) and 8, 13 (RTC, FPU) are wired
/// edge triggered, as on the PIIX/ICH ELCR; the other bits read back as 0.
const MASTER_ELCR_WRITABLE: u8 = 0xf8;
const SLAVE_ELCR_WRITABLE: u8 = 0xde;

/// The slave's INT output is wired to master IR2.
const CASCADE_IRQ: u8 = 2;
const IRQS_PER_PIC: u8 = 8;
const IRQ_COUNT: u8 = 2 * IRQS_PER_PIC;
/// An INTA cycle that finds no request (it vanished between INT and INTA) answers with the
/// lowest-priority IR of the chip that was asked.
const SPURIOUS_IRQ: u8 = 7;
/// What an INTA cycle returns when no chip drives the data bus (a cascade ID nobody answers to).
const OPEN_BUS: u8 = 0xff;

/// Command-port write with this bit set is ICW1 and starts initialization.
const ICW1_INIT: u8 = 0x10;
/// ICW1: an ICW4 follows ICW3.
const ICW1_IC4: u8 = 0x01;
/// ICW1: single chip, no ICW3.
const ICW1_SINGLE: u8 = 0x02;
/// ICW1: every input is level triggered, regardless of ELCR.
const ICW1_LEVEL_TRIGGERED: u8 = 0x08;
/// ICW2 carries the vector base in bits 7:3; the low bits are the IRQ number.
const ICW2_VECTOR_MASK: u8 = 0xf8;
/// ICW3 of a slave: its ID (the master IR it hangs off) is in bits 2:0.
const ICW3_SLAVE_ID_MASK: u8 = 0x07;
/// ICW4: end of interrupt happens automatically at the INTA, so ISR is never set.
const ICW4_AUTO_EOI: u8 = 0x02;
/// ICW4: special fully nested mode, only meaningful on a master.
const ICW4_SPECIAL_FULLY_NESTED: u8 = 0x10;

/// Command-port write with bit 3 set and bit 4 clear is OCW3.
const OCW3_MARKER: u8 = 0x08;
/// OCW3: the next read of either port is a poll (INTA-like) cycle.
const OCW3_POLL: u8 = 0x04;
/// OCW3: bit 0 (RIS) is valid only when this "read register command" bit is set.
const OCW3_READ_REGISTER: u8 = 0x02;
/// OCW3: selects ISR (set) or IRR (clear) for subsequent command-port reads.
const OCW3_READ_ISR: u8 = 0x01;
/// OCW3: bit 5 (SMM) is valid only when this "enable special mask mode" bit is set.
const OCW3_SPECIAL_MASK_ENABLE: u8 = 0x40;
const OCW3_SPECIAL_MASK: u8 = 0x20;
/// A poll read returns this bit with the IRQ number in bits 2:0 if a request was acknowledged.
const POLL_INTERRUPT: u8 = 0x80;

/// OCW2 is a command-port write with bits 4:3 clear; bits 7:5 (R, SL, EOI) select the operation.
const OCW2_OPERATION_SHIFT: u8 = 5;
const OCW2_ROTATE_IN_AUTO_EOI_CLEAR: u8 = 0;
const OCW2_EOI_NONSPECIFIC: u8 = 1;
const OCW2_EOI_SPECIFIC: u8 = 3;
const OCW2_ROTATE_IN_AUTO_EOI_SET: u8 = 4;
const OCW2_EOI_NONSPECIFIC_ROTATE: u8 = 5;
const OCW2_SET_PRIORITY: u8 = 6;
const OCW2_EOI_SPECIFIC_ROTATE: u8 = 7;
/// OCW2 specific forms: bits 2:0 name the IRQ.
const OCW2_IRQ_MASK: u8 = 7;

/// Reset state of the cascade wiring, as if firmware had programmed the PC standard: the master
/// has a slave on IR2 and the slave's ID is 2. ICW3 overrides it.
const DEFAULT_MASTER_ICW3: u8 = 1 << CASCADE_IRQ;
const DEFAULT_SLAVE_ICW3: u8 = CASCADE_IRQ;

#[derive(Clone, Snapshot)]
struct Pic {
    is_master: bool,
    /// Requested lines. Edge inputs latch here on `raise` and clear on acknowledge; level inputs
    /// mirror `line` and only clear when the device lowers the line.
    irr: u8,
    /// In-service lines, cleared by EOI. A set bit blocks every request of equal or lower priority.
    isr: u8,
    imr: u8,
    /// Level of lines that are level triggered (and of the cascade pin), used for edge detection
    /// and to refill IRR when a pin turns level triggered. Edge inputs from devices are pulses
    /// and are not tracked.
    line: u8,
    /// Vector of IRQ 0, from ICW2.
    base: u8,
    /// Which ICW the next data-port write is: 1 = ICW2, 2 = ICW3, 3 = ICW4,
    /// 0 = initialization done and the write is OCW1 (the mask).
    icw_step: u8,
    needs_icw4: bool,
    single: bool,
    /// ICW1 LTIM: all inputs level triggered.
    level_triggered: bool,
    /// ICW3: on a master, the mask of IRs that have a slave; on a slave, its ID in bits 2:0.
    cascade_config: u8,
    auto_eoi: bool,
    rotate_in_auto_eoi: bool,
    special_fully_nested: bool,
    special_mask: bool,
    /// IRQ that currently has the highest priority; the lowest priority is the one before it.
    /// Rotating EOIs move it.
    highest_priority: u8,
    /// OCW3 selection for command-port reads: ISR when set, IRR otherwise.
    read_isr: bool,
    /// OCW3 poll command armed: the next read is a poll cycle.
    poll: bool,
    elcr: u8,
}

impl Pic {
    fn new(is_master: bool) -> Self {
        Pic {
            is_master,
            irr: 0,
            isr: 0,
            imr: 0,
            line: 0,
            base: 0,
            icw_step: 0,
            needs_icw4: false,
            single: false,
            level_triggered: false,
            cascade_config: if is_master {
                DEFAULT_MASTER_ICW3
            } else {
                DEFAULT_SLAVE_ICW3
            },
            auto_eoi: false,
            rotate_in_auto_eoi: false,
            special_fully_nested: false,
            special_mask: false,
            highest_priority: 0,
            read_isr: false,
            poll: false,
            elcr: 0,
        }
    }

    fn elcr_writable(&self) -> u8 {
        if self.is_master {
            MASTER_ELCR_WRITABLE
        } else {
            SLAVE_ELCR_WRITABLE
        }
    }

    fn level_mask(&self) -> u8 {
        if self.level_triggered {
            u8::MAX
        } else {
            self.elcr
        }
    }

    /// Whether INTA on master IR2 goes to a slave rather than being an ordinary line.
    fn has_slave_on_cascade_irq(&self) -> bool {
        !self.single && self.cascade_config & (1 << CASCADE_IRQ) != 0
    }

    /// Pins that just became level triggered take their value from the line; the others keep
    /// their latched requests.
    fn refresh_level_inputs(&mut self) {
        let level = self.level_mask();
        self.irr = (self.irr & !level) | (self.line & level);
    }

    /// Position of the highest-priority IRQ in `mask` counted from `highest_priority`
    /// (0 = highest), or 8 if `mask` is empty.
    fn priority_of(&self, mask: u8) -> u8 {
        if mask == 0 {
            return IRQS_PER_PIC;
        }
        let mut priority = 0;
        while mask & (1 << ((priority + self.highest_priority) % IRQS_PER_PIC)) == 0 {
            priority += 1;
        }
        priority
    }

    /// The IRQ this chip would present on INT now: the highest-priority unmasked request that
    /// beats everything in service.
    fn pick_irq(&self) -> Option<u8> {
        let requested = self.irr & !self.imr;
        let priority = self.priority_of(requested);
        if priority == IRQS_PER_PIC {
            return None;
        }
        let mut in_service = self.isr;
        if self.special_mask {
            in_service &= !self.imr;
        }
        if self.special_fully_nested && self.is_master {
            in_service &= !(1 << CASCADE_IRQ);
        }
        if priority < self.priority_of(in_service) {
            Some((priority + self.highest_priority) % IRQS_PER_PIC)
        } else {
            None
        }
    }

    /// A line changing level. Level pins follow it; edge pins latch only on a rising edge.
    fn drive_line(&mut self, irq: u8, level: bool) {
        let bit = 1 << irq;
        if self.level_mask() & bit != 0 {
            if level {
                self.irr |= bit;
                self.line |= bit;
            } else {
                self.irr &= !bit;
                self.line &= !bit;
            }
        } else {
            if level && self.line & bit == 0 {
                self.irr |= bit;
            }
            if level {
                self.line |= bit;
            } else {
                self.line &= !bit;
            }
        }
    }

    /// Interrupt acknowledge of `irq` on this chip.
    fn intack(&mut self, irq: u8) {
        let bit = 1 << irq;
        if self.auto_eoi {
            if self.rotate_in_auto_eoi {
                self.highest_priority = (irq + 1) % IRQS_PER_PIC;
            }
        } else {
            self.isr |= bit;
        }
        // A level input stays requested until the device drops it.
        if self.level_mask() & bit == 0 {
            self.irr &= !bit;
        }
    }

    /// OCW3 poll: acknowledges the request (if any) and reports it in the data byte.
    fn poll_read(&mut self) -> u8 {
        self.poll = false;
        match self.pick_irq() {
            Some(irq) => {
                self.intack(irq);
                POLL_INTERRUPT | irq
            }
            None => 0,
        }
    }

    fn command(&mut self, value: u8) {
        if value & ICW1_INIT != 0 {
            self.level_triggered = value & ICW1_LEVEL_TRIGGERED != 0;
            self.irr = 0;
            self.isr = 0;
            // Initialization also clears the mask, as on real hardware.
            self.imr = 0;
            self.highest_priority = 0;
            self.read_isr = false;
            self.poll = false;
            self.special_mask = false;
            self.auto_eoi = false;
            self.rotate_in_auto_eoi = false;
            self.special_fully_nested = false;
            self.needs_icw4 = value & ICW1_IC4 != 0;
            self.single = value & ICW1_SINGLE != 0;
            if self.single {
                self.cascade_config = 0;
            }
            self.icw_step = 1;
            self.refresh_level_inputs();
        } else if value & OCW3_MARKER != 0 {
            if value & OCW3_POLL != 0 {
                self.poll = true;
            }
            if value & OCW3_READ_REGISTER != 0 {
                self.read_isr = value & OCW3_READ_ISR != 0;
            }
            if value & OCW3_SPECIAL_MASK_ENABLE != 0 {
                self.special_mask = value & OCW3_SPECIAL_MASK != 0;
            }
        } else {
            let irq = value & OCW2_IRQ_MASK;
            match value >> OCW2_OPERATION_SHIFT {
                OCW2_ROTATE_IN_AUTO_EOI_CLEAR => self.rotate_in_auto_eoi = false,
                OCW2_ROTATE_IN_AUTO_EOI_SET => self.rotate_in_auto_eoi = true,
                operation @ (OCW2_EOI_NONSPECIFIC | OCW2_EOI_NONSPECIFIC_ROTATE) => {
                    let priority = self.priority_of(self.isr);
                    if priority != IRQS_PER_PIC {
                        let served = (priority + self.highest_priority) % IRQS_PER_PIC;
                        self.isr &= !(1 << served);
                        if operation == OCW2_EOI_NONSPECIFIC_ROTATE {
                            self.highest_priority = (served + 1) % IRQS_PER_PIC;
                        }
                    }
                }
                OCW2_EOI_SPECIFIC => self.isr &= !(1 << irq),
                OCW2_EOI_SPECIFIC_ROTATE => {
                    self.isr &= !(1 << irq);
                    self.highest_priority = (irq + 1) % IRQS_PER_PIC;
                }
                // The named IRQ becomes the lowest priority.
                OCW2_SET_PRIORITY => self.highest_priority = (irq + 1) % IRQS_PER_PIC,
                // Operation 2 is a no-op.
                _ => {}
            }
        }
    }

    /// 8086 mode is assumed whatever ICW4 says; BUF/M-S bits are ignored.
    fn data(&mut self, value: u8) {
        match self.icw_step {
            1 => {
                self.base = value & ICW2_VECTOR_MASK;
                self.icw_step = if !self.single {
                    2
                } else if self.needs_icw4 {
                    3
                } else {
                    0
                };
            }
            2 => {
                self.cascade_config = value;
                self.icw_step = if self.needs_icw4 { 3 } else { 0 };
            }
            3 => {
                self.special_fully_nested = value & ICW4_SPECIAL_FULLY_NESTED != 0;
                self.auto_eoi = value & ICW4_AUTO_EOI != 0;
                self.icw_step = 0;
            }
            _ => self.imr = value,
        }
    }

    fn read_command(&self) -> u8 {
        if self.read_isr { self.isr } else { self.irr }
    }

    fn write_elcr(&mut self, value: u8) {
        self.elcr = value & self.elcr_writable();
        self.refresh_level_inputs();
    }
}

#[derive(Clone, Snapshot)]
pub(crate) struct Pics {
    master: Pic,
    slave: Pic,
    /// Device-side level of each of the 16 ISA lines, whatever the PIC's own trigger mode is.
    line_levels: u16,
    /// Changes of `line_levels` (and every raise, which is also a pulse for an edge-triggered
    /// consumer) in order, for the I/O APIC: the same device lines feed both controllers.
    line_events: Vec<(u8, bool)>,
}

impl Default for Pics {
    fn default() -> Self {
        Pics {
            master: Pic::new(true),
            slave: Pic::new(false),
            line_levels: 0,
            line_events: Vec::new(),
        }
    }
}

impl Pics {
    pub fn state(&self) -> String {
        format!(
            "m(irr {:02x} isr {:02x} imr {:02x} base {:02x}) s(irr {:02x} isr {:02x} imr {:02x} base {:02x})",
            self.master.irr,
            self.master.isr,
            self.master.imr,
            self.master.base,
            self.slave.irr,
            self.slave.isr,
            self.slave.imr,
            self.slave.base
        )
    }

    fn chip_for_irq(&mut self, irq: u8) -> (&mut Pic, u8) {
        if irq < IRQS_PER_PIC {
            (&mut self.master, irq)
        } else {
            (&mut self.slave, irq - IRQS_PER_PIC)
        }
    }

    /// Mirrors the slave's INT output onto master IR2. The master latches a rising edge (IR2 is
    /// never level triggered through ELCR), so the request stays at the master even if the slave
    /// loses it again before the acknowledge; that is how a spurious IRQ 15 arises.
    fn sync_cascade(&mut self) {
        let slave_int = self.slave.pick_irq().is_some();
        self.master.drive_line(CASCADE_IRQ, slave_int);
    }

    /// Asserts `irq` (0..15; 8..15 are the slave). An edge input latches a request each call, so
    /// devices that signal events rather than levels can simply call this once per event. A
    /// level-triggered input (ELCR or ICW1 LTIM) stays requested until `lower`. IRQ 2 is the
    /// cascade pin, driven only by the slave.
    pub fn raise(&mut self, irq: u8) {
        if irq >= IRQ_COUNT {
            return;
        }
        // The I/O APIC hears about every raise, even on the PIC's cascade pin or while the line
        // is already high: an edge-triggered pin counts each call as one pulse.
        self.line_levels |= 1 << irq;
        self.line_events.push((irq, true));
        let (chip, line) = self.chip_for_irq(irq);
        if line == CASCADE_IRQ && chip.is_master {
            return;
        }
        if chip.level_mask() & (1 << line) != 0 {
            chip.drive_line(line, true);
        } else {
            chip.irr |= 1 << line;
        }
        self.sync_cascade();
    }

    /// Deasserts `irq`. Only level-triggered inputs react: an edge input's request is already
    /// latched and stays until acknowledged.
    pub fn lower(&mut self, irq: u8) {
        if irq >= IRQ_COUNT {
            return;
        }
        if self.line_levels & (1 << irq) != 0 {
            self.line_levels &= !(1 << irq);
            self.line_events.push((irq, false));
        }
        let (chip, line) = self.chip_for_irq(irq);
        if line == CASCADE_IRQ && chip.is_master {
            return;
        }
        if chip.level_mask() & (1 << line) != 0 {
            chip.drive_line(line, false);
        }
        self.sync_cascade();
    }

    /// Line changes since the last call, for the I/O APIC.
    pub fn take_line_events(&mut self) -> Vec<(u8, bool)> {
        core::mem::take(&mut self.line_events)
    }

    pub fn has_line_events(&self) -> bool {
        !self.line_events.is_empty()
    }

    /// Polled by the CPU between instructions while IF=1, so the first check must stay cheap:
    /// it rejects the common case (nothing requested, or everything requested is masked) without
    /// running `pick_irq`. It is only an early-out and must never reject what an INTA would
    /// accept. The slave's requests are already in the master's IRR (see `sync_cascade`), so the
    /// master alone decides.
    pub fn has_pending(&self) -> bool {
        (self.master.irr & !self.master.imr) != 0 && self.master.pick_irq().is_some()
    }

    /// One INTA cycle: returns the vector the CPU reads and updates IRR/ISR. With nothing to
    /// deliver it answers like the hardware does, with the spurious IR7 vector and no ISR bit:
    /// master spurious is `base + 7` and the guest must not EOI it. If the master routes the
    /// cycle to the slave but the slave's request is gone (masked or lowered meanwhile), the
    /// answer is the slave's IR7 vector with the master's cascade ISR bit set and the slave's
    /// untouched, so the guest EOIs only the master.
    fn inta(&mut self) -> u8 {
        let vector = match self.master.pick_irq() {
            None => self.master.base + SPURIOUS_IRQ,
            Some(CASCADE_IRQ) if self.master.has_slave_on_cascade_irq() => {
                let vector = if self.slave.cascade_config & ICW3_SLAVE_ID_MASK == CASCADE_IRQ {
                    match self.slave.pick_irq() {
                        Some(irq) => {
                            self.slave.intack(irq);
                            self.slave.base + irq
                        }
                        None => self.slave.base + SPURIOUS_IRQ,
                    }
                } else {
                    // The slave is not the one addressed on the CAS lines; nobody answers.
                    OPEN_BUS
                };
                self.master.intack(CASCADE_IRQ);
                vector
            }
            Some(irq) => {
                self.master.intack(irq);
                self.master.base + irq
            }
        };
        self.sync_cascade();
        vector
    }

    /// Interrupt acknowledge cycle: moves the request to in-service and returns the vector, or
    /// `None` if INT is not asserted (the CPU would not run an INTA). A slave interrupt also marks
    /// the cascade IRQ in service on the master, so the guest issues two EOIs (slave, then master).
    pub fn ack(&mut self) -> Option<u8> {
        if self.has_pending() {
            Some(self.inta())
        } else {
            None
        }
    }

    pub fn io_read(&mut self, port: u16) -> u8 {
        let chip = match port {
            MASTER_COMMAND_PORT | MASTER_DATA_PORT => &mut self.master,
            SLAVE_COMMAND_PORT | SLAVE_DATA_PORT => &mut self.slave,
            MASTER_ELCR_PORT => return self.master.elcr,
            SLAVE_ELCR_PORT => return self.slave.elcr,
            _ => return OPEN_BUS,
        };
        if chip.poll {
            let byte = chip.poll_read();
            self.sync_cascade();
            return byte;
        }
        if port == MASTER_COMMAND_PORT || port == SLAVE_COMMAND_PORT {
            chip.read_command()
        } else {
            chip.imr
        }
    }

    pub fn io_write(&mut self, port: u16, value: u8) {
        match port {
            MASTER_COMMAND_PORT => self.master.command(value),
            SLAVE_COMMAND_PORT => self.slave.command(value),
            MASTER_DATA_PORT => self.master.data(value),
            SLAVE_DATA_PORT => self.slave.data(value),
            MASTER_ELCR_PORT => self.master.write_elcr(value),
            SLAVE_ELCR_PORT => self.slave.write_elcr(value),
            _ => return,
        }
        self.sync_cascade();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER_BASE: u8 = 0x20;
    const SLAVE_BASE: u8 = 0x28;
    const ICW1_STANDARD: u8 = ICW1_INIT | ICW1_IC4;
    const ICW4_8086: u8 = 0x01;
    const OCW2_NO_OPERATION: u8 = 2;
    const ALL_MASKED: u8 = 0xff;

    const EOI_NONSPECIFIC: u8 = OCW2_EOI_NONSPECIFIC << OCW2_OPERATION_SHIFT;
    const EOI_NONSPECIFIC_ROTATE: u8 = OCW2_EOI_NONSPECIFIC_ROTATE << OCW2_OPERATION_SHIFT;

    const fn specific_eoi(irq: u8) -> u8 {
        OCW2_EOI_SPECIFIC << OCW2_OPERATION_SHIFT | irq
    }

    const fn specific_eoi_rotate(irq: u8) -> u8 {
        OCW2_EOI_SPECIFIC_ROTATE << OCW2_OPERATION_SHIFT | irq
    }

    const fn set_priority(lowest_irq: u8) -> u8 {
        OCW2_SET_PRIORITY << OCW2_OPERATION_SHIFT | lowest_irq
    }

    const OCW3_READ_IRR: u8 = OCW3_MARKER | OCW3_READ_REGISTER;
    const OCW3_READ_ISR_SELECT: u8 = OCW3_MARKER | OCW3_READ_REGISTER | OCW3_READ_ISR;
    const OCW3_POLL_COMMAND: u8 = OCW3_MARKER | OCW3_POLL;
    const OCW3_SPECIAL_MASK_ON: u8 = OCW3_MARKER | OCW3_SPECIAL_MASK_ENABLE | OCW3_SPECIAL_MASK;
    const OCW3_SPECIAL_MASK_OFF: u8 = OCW3_MARKER | OCW3_SPECIAL_MASK_ENABLE;

    /// Programs both PICs: vectors 0x20 and 0x28, the PC cascade, and the given ICW4 values.
    fn init_with(pics: &mut Pics, master_icw4: u8, slave_icw4: u8) {
        for (command, data, base, icw3, icw4) in [
            (
                MASTER_COMMAND_PORT,
                MASTER_DATA_PORT,
                MASTER_BASE,
                1 << CASCADE_IRQ,
                master_icw4,
            ),
            (
                SLAVE_COMMAND_PORT,
                SLAVE_DATA_PORT,
                SLAVE_BASE,
                CASCADE_IRQ,
                slave_icw4,
            ),
        ] {
            pics.io_write(command, ICW1_STANDARD);
            pics.io_write(data, base);
            pics.io_write(data, icw3);
            pics.io_write(data, icw4);
            pics.io_write(data, 0);
        }
    }

    /// Programs both PICs the way Linux does.
    fn init(pics: &mut Pics) {
        init_with(pics, ICW4_8086, ICW4_8086);
    }

    fn new_pics() -> Pics {
        let mut pics = Pics::default();
        init(&mut pics);
        pics
    }

    fn master_command(pics: &mut Pics, value: u8) {
        pics.io_write(MASTER_COMMAND_PORT, value);
    }

    fn slave_command(pics: &mut Pics, value: u8) {
        pics.io_write(SLAVE_COMMAND_PORT, value);
    }

    fn set_master_mask(pics: &mut Pics, mask: u8) {
        pics.io_write(MASTER_DATA_PORT, mask);
    }

    fn set_slave_mask(pics: &mut Pics, mask: u8) {
        pics.io_write(SLAVE_DATA_PORT, mask);
    }

    /// Ack and EOI at the master, for tests that do not care about the in-service state.
    fn serve_master_irq(pics: &mut Pics, irq: u8) -> u8 {
        pics.raise(irq);
        let vector = pics.ack().unwrap();
        master_command(pics, EOI_NONSPECIFIC);
        vector
    }

    // Initialization

    #[test]
    fn master_irqs_deliver_vector_base_plus_irq() {
        for irq in 0..8u8 {
            if irq == CASCADE_IRQ {
                continue;
            }
            let mut pics = new_pics();
            pics.raise(irq);
            assert!(pics.has_pending(), "irq {irq}");
            assert_eq!(pics.ack(), Some(MASTER_BASE + irq), "irq {irq}");
            assert_eq!(pics.master.isr, 1 << irq);
            assert_eq!(pics.master.irr, 0);
        }
    }

    #[test]
    fn slave_irqs_deliver_slave_vector_base_plus_irq() {
        for irq in 8..16u8 {
            let mut pics = new_pics();
            pics.raise(irq);
            assert!(pics.has_pending(), "irq {irq}");
            assert_eq!(pics.ack(), Some(SLAVE_BASE + irq - 8), "irq {irq}");
            assert_eq!(pics.slave.isr, 1 << (irq - 8));
            assert_eq!(pics.slave.irr, 0);
            assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
        }
    }

    #[test]
    fn icw2_low_bits_are_ignored() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, 0x27);
        pics.io_write(MASTER_DATA_PORT, 0x04);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.raise(1);
        assert_eq!(pics.ack(), Some(0x21));
    }

    #[test]
    fn vector_base_can_be_any_multiple_of_eight() {
        for base in [0x00, 0x08, 0x70, 0xf0, 0xf8] {
            let mut pics = Pics::default();
            pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD);
            pics.io_write(MASTER_DATA_PORT, base);
            pics.io_write(MASTER_DATA_PORT, 0x04);
            pics.io_write(MASTER_DATA_PORT, ICW4_8086);
            pics.raise(7);
            assert_eq!(pics.ack(), Some(base + 7));
        }
    }

    #[test]
    fn icw1_clears_mask_requests_and_in_service() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0xa5);
        pics.raise(1);
        pics.raise(4);
        pics.master.isr = 0x80;
        master_command(&mut pics, ICW1_STANDARD);
        assert_eq!(pics.master.imr, 0);
        assert_eq!(pics.master.irr, 0);
        assert_eq!(pics.master.isr, 0);
    }

    #[test]
    fn icw1_resets_priority_rotation_and_modes() {
        let mut pics = Pics::default();
        init_with(
            &mut pics,
            ICW4_8086 | ICW4_AUTO_EOI | ICW4_SPECIAL_FULLY_NESTED,
            ICW4_8086,
        );
        master_command(&mut pics, set_priority(2));
        master_command(&mut pics, OCW3_SPECIAL_MASK_ON);
        master_command(
            &mut pics,
            OCW2_ROTATE_IN_AUTO_EOI_SET << OCW2_OPERATION_SHIFT,
        );
        assert_eq!(pics.master.highest_priority, 3);
        master_command(&mut pics, ICW1_STANDARD);
        assert_eq!(pics.master.highest_priority, 0);
        assert!(!pics.master.auto_eoi);
        assert!(!pics.master.special_fully_nested);
        assert!(!pics.master.special_mask);
        assert!(!pics.master.rotate_in_auto_eoi);
    }

    #[test]
    fn icw1_resets_the_read_register_selection_to_irr() {
        let mut pics = new_pics();
        master_command(&mut pics, OCW3_READ_ISR_SELECT);
        master_command(&mut pics, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x04);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.raise(5);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x20);
    }

    #[test]
    fn icw_sequence_takes_icw3_and_icw4_before_the_mask() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0xff);
        pics.io_write(MASTER_DATA_PORT, 0xff);
        // Both 0xff writes were ICW3 and ICW4, not masks.
        assert_eq!(pics.master.imr, 0);
        assert_eq!(pics.master.icw_step, 0);
        pics.io_write(MASTER_DATA_PORT, 0xff);
        assert_eq!(pics.master.imr, 0xff);
    }

    #[test]
    fn icw1_without_ic4_skips_icw4() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_INIT);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x04);
        pics.io_write(MASTER_DATA_PORT, 0x5a);
        assert_eq!(pics.master.imr, 0x5a);
    }

    #[test]
    fn single_mode_skips_icw3() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_INIT | ICW1_IC4 | ICW1_SINGLE);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.io_write(MASTER_DATA_PORT, 0x3c);
        assert_eq!(pics.master.imr, 0x3c);
    }

    #[test]
    fn single_mode_without_ic4_goes_straight_to_the_mask() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_INIT | ICW1_SINGLE);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x81);
        assert_eq!(pics.master.imr, 0x81);
    }

    #[test]
    fn data_port_reads_back_the_mask_for_every_value() {
        let mut pics = new_pics();
        for mask in 0..=u8::MAX {
            set_master_mask(&mut pics, mask);
            set_slave_mask(&mut pics, !mask);
            assert_eq!(pics.io_read(MASTER_DATA_PORT), mask);
            assert_eq!(pics.io_read(SLAVE_DATA_PORT), !mask);
        }
    }

    #[test]
    fn power_on_state_has_the_pc_cascade_wired() {
        let mut pics = Pics::default();
        pics.raise(9);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(0x01));
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
        assert_eq!(pics.slave.isr, 1 << 1);
    }

    #[test]
    fn power_on_state_has_everything_unmasked_and_idle() {
        let mut pics = Pics::default();
        assert!(!pics.has_pending());
        assert_eq!(pics.ack(), None);
        assert_eq!(pics.io_read(MASTER_DATA_PORT), 0);
        assert_eq!(pics.io_read(SLAVE_DATA_PORT), 0);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0);
        assert_eq!(pics.io_read(MASTER_ELCR_PORT), 0);
    }

    #[test]
    fn unknown_ports_read_open_bus_and_ignore_writes() {
        let mut pics = new_pics();
        let before = pics.state();
        assert_eq!(pics.io_read(0x22), OPEN_BUS);
        assert_eq!(pics.io_read(0x4d2), OPEN_BUS);
        pics.io_write(0x22, 0x55);
        pics.io_write(0x4d2, 0x55);
        assert_eq!(pics.state(), before);
    }

    #[test]
    fn state_string_shows_both_chips() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0x11);
        set_slave_mask(&mut pics, 0x22);
        pics.raise(1);
        assert_eq!(
            pics.state(),
            "m(irr 02 isr 00 imr 11 base 20) s(irr 00 isr 00 imr 22 base 28)"
        );
    }

    // Registers

    #[test]
    fn command_port_reads_irr_by_default_and_isr_after_ocw3() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0x40);
        pics.raise(6);
        pics.raise(3);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x48);
        pics.ack();
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x40);
        master_command(&mut pics, OCW3_READ_ISR_SELECT);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x08);
        // The selection sticks.
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x08);
        master_command(&mut pics, OCW3_READ_IRR);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x40);
    }

    #[test]
    fn ocw3_without_the_read_register_bit_keeps_the_selection() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.ack();
        master_command(&mut pics, OCW3_READ_ISR_SELECT);
        master_command(&mut pics, OCW3_MARKER);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x08);
        master_command(&mut pics, OCW3_MARKER | OCW3_READ_ISR);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x08);
    }

    #[test]
    fn ocw3_selection_is_per_chip() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        slave_command(&mut pics, OCW3_READ_ISR_SELECT);
        assert_eq!(pics.io_read(SLAVE_COMMAND_PORT), 0x02);
        // The master still reads IRR (empty).
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x00);
        master_command(&mut pics, OCW3_READ_ISR_SELECT);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x04);
    }

    #[test]
    fn slave_irr_is_visible_on_the_slave_command_port() {
        let mut pics = new_pics();
        set_slave_mask(&mut pics, 0x80);
        pics.raise(15);
        assert_eq!(pics.io_read(SLAVE_COMMAND_PORT), 0x80);
        // Masked, so the master never saw it.
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x00);
    }

    // Priority resolution

    #[test]
    fn lowest_irq_number_has_the_highest_priority() {
        let mut pics = new_pics();
        for irq in [6, 3, 5] {
            pics.raise(irq);
        }
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        // IRQ 3 in service blocks the rest.
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 6));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn higher_priority_request_preempts_a_lower_one_in_service() {
        let mut pics = new_pics();
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        pics.raise(1);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 1));
        assert_eq!(pics.master.isr, 0x12);
    }

    #[test]
    fn equal_and_lower_priority_requests_wait_for_eoi() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        pics.raise(4);
        pics.raise(5);
        pics.raise(7);
        assert!(!pics.has_pending());
        assert_eq!(pics.ack(), None);
        // They are held in IRR meanwhile.
        assert_eq!(pics.master.irr, 0xb0);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
    }

    #[test]
    fn masked_in_service_irq_still_blocks_lower_priorities() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.ack();
        set_master_mask(&mut pics, 0x08);
        pics.raise(5);
        assert!(!pics.has_pending());
    }

    #[test]
    fn masked_high_priority_request_does_not_block_a_lower_one() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0x01);
        pics.raise(0);
        pics.raise(6);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 6));
    }

    #[test]
    fn slave_priorities_order_irq8_through_irq15() {
        let mut pics = new_pics();
        pics.raise(15);
        pics.raise(11);
        pics.raise(13);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 3));
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 5));
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
    }

    #[test]
    fn master_irq0_and_irq1_beat_a_slave_interrupt() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.raise(0);
        assert_eq!(pics.ack(), Some(MASTER_BASE));
        // IRQ 0 in service blocks the cascade (IR2 is lower priority).
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
    }

    #[test]
    fn master_irqs_above_the_cascade_priority_preempt_a_slave_interrupt() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        pics.raise(1);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 1));
        pics.raise(0);
        assert_eq!(pics.ack(), Some(MASTER_BASE));
    }

    #[test]
    fn cascade_in_service_blocks_lower_priority_master_irqs() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        pics.raise(3);
        pics.raise(7);
        assert!(!pics.has_pending());
        slave_command(&mut pics, EOI_NONSPECIFIC);
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn slave_irq_in_service_blocks_lower_slave_irqs() {
        let mut pics = new_pics();
        pics.raise(10);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 2));
        master_command(&mut pics, EOI_NONSPECIFIC);
        pics.raise(12);
        assert!(!pics.has_pending());
        // A higher-priority slave line is let through.
        pics.raise(9);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
    }

    #[test]
    fn slave_request_arriving_while_the_master_cascade_is_in_service_waits() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        slave_command(&mut pics, EOI_NONSPECIFIC);
        pics.raise(10);
        // The master has IR2 latched but in service.
        assert_eq!(pics.master.irr & (1 << CASCADE_IRQ), 1 << CASCADE_IRQ);
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 2));
    }

    #[test]
    fn two_slave_requests_are_delivered_in_order_with_two_eois_each() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.raise(10);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 2));
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.slave.isr, 0);
    }

    #[test]
    fn eoi_to_the_master_alone_leaves_the_slave_irq_in_service() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.slave.isr, 0x02);
        pics.raise(11);
        assert!(!pics.has_pending());
    }

    // Masking

    #[test]
    fn masked_request_is_held_in_irr_until_unmasked() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0x10);
        pics.raise(4);
        assert!(!pics.has_pending());
        assert_eq!(pics.ack(), None);
        assert_eq!(pics.master.irr, 0x10);
        set_master_mask(&mut pics, 0);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
    }

    #[test]
    fn all_masked_hides_every_request() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, ALL_MASKED);
        set_slave_mask(&mut pics, ALL_MASKED);
        for irq in 0..16 {
            pics.raise(irq);
        }
        assert!(!pics.has_pending());
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn masking_the_cascade_irq_hides_the_slave_but_not_master_irqs() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 1 << CASCADE_IRQ);
        pics.raise(9);
        assert!(!pics.has_pending());
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        master_command(&mut pics, EOI_NONSPECIFIC);
        set_master_mask(&mut pics, 0);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
    }

    #[test]
    fn masking_a_slave_irq_hides_it_from_the_master_too() {
        let mut pics = new_pics();
        set_slave_mask(&mut pics, 1 << 1);
        pics.raise(9);
        assert!(!pics.has_pending());
        assert_eq!(pics.master.irr, 0);
    }

    #[test]
    fn unmasking_does_not_clear_in_service() {
        let mut pics = new_pics();
        pics.raise(2 + 1);
        pics.ack();
        set_master_mask(&mut pics, ALL_MASKED);
        set_master_mask(&mut pics, 0);
        assert_eq!(pics.master.isr, 0x08);
    }

    // End of interrupt

    #[test]
    fn nonspecific_eoi_clears_the_highest_priority_in_service() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        pics.raise(3);
        pics.ack();
        assert_eq!(pics.master.isr, 0x18);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.isr, 0x10);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.isr, 0x00);
    }

    #[test]
    fn specific_eoi_clears_only_the_named_irq() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        pics.raise(3);
        pics.ack();
        master_command(&mut pics, specific_eoi(4));
        assert_eq!(pics.master.isr, 0x08);
    }

    #[test]
    fn specific_eoi_for_an_idle_irq_changes_nothing() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        master_command(&mut pics, specific_eoi(6));
        assert_eq!(pics.master.isr, 0x10);
    }

    #[test]
    fn eoi_with_nothing_in_service_is_harmless() {
        let mut pics = new_pics();
        master_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, specific_eoi(3));
        slave_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.master.highest_priority, 0);
    }

    #[test]
    fn eoi_does_not_clear_pending_requests() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        pics.raise(4);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.irr, 0x10);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
    }

    #[test]
    fn eoi_forms_ignore_the_low_bits_unless_specific() {
        let mut pics = new_pics();
        pics.raise(2 + 1);
        pics.ack();
        // Non-specific EOI with junk in the IRQ bits still clears the in-service one.
        master_command(&mut pics, EOI_NONSPECIFIC | 0x07);
        assert_eq!(pics.master.isr, 0);
    }

    #[test]
    fn ocw2_no_operation_changes_nothing() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        master_command(&mut pics, OCW2_NO_OPERATION << OCW2_OPERATION_SHIFT | 0x04);
        assert_eq!(pics.master.isr, 0x10);
        assert_eq!(pics.master.highest_priority, 0);
    }

    #[test]
    fn slave_eoi_does_not_touch_the_master() {
        let mut pics = new_pics();
        pics.raise(9);
        pics.ack();
        slave_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
    }

    // Edge and level triggering

    #[test]
    fn each_raise_of_an_edge_input_latches_a_request() {
        let mut pics = new_pics();
        for _ in 0..4 {
            pics.raise(4);
            assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
            master_command(&mut pics, EOI_NONSPECIFIC);
        }
    }

    #[test]
    fn raising_an_edge_input_twice_before_the_ack_delivers_once() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn edge_input_raised_during_service_is_latched_for_later() {
        let mut pics = new_pics();
        pics.raise(4);
        pics.ack();
        pics.raise(4);
        assert_eq!(pics.master.irr, 0x10);
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(pics.has_pending());
    }

    #[test]
    fn lowering_an_edge_input_does_not_cancel_its_latched_request() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.lower(3);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn elcr_stores_only_the_programmable_bits() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 0xff);
        pics.io_write(SLAVE_ELCR_PORT, 0xff);
        assert_eq!(pics.io_read(MASTER_ELCR_PORT), 0xf8);
        assert_eq!(pics.io_read(SLAVE_ELCR_PORT), 0xde);
        for value in 0..=u8::MAX {
            pics.io_write(MASTER_ELCR_PORT, value);
            pics.io_write(SLAVE_ELCR_PORT, value);
            assert_eq!(pics.io_read(MASTER_ELCR_PORT), value & 0xf8);
            assert_eq!(pics.io_read(SLAVE_ELCR_PORT), value & 0xde);
        }
    }

    #[test]
    fn elcr_ports_are_independent_of_the_mask_registers() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 0x80);
        assert_eq!(pics.io_read(MASTER_DATA_PORT), 0);
        assert_eq!(pics.io_read(SLAVE_ELCR_PORT), 0);
        set_master_mask(&mut pics, 0x01);
        assert_eq!(pics.io_read(MASTER_ELCR_PORT), 0x80);
    }

    #[test]
    fn level_input_stays_requested_after_eoi_until_lowered() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 4);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        // The line is still high: IRR keeps it, ISR blocks it.
        assert_eq!(pics.master.irr, 0x10);
        assert!(!pics.has_pending());
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        pics.lower(4);
        assert_eq!(pics.master.irr, 0);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(!pics.has_pending());
    }

    #[test]
    fn lowering_a_level_input_before_the_ack_withdraws_the_request() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 5);
        pics.raise(5);
        assert!(pics.has_pending());
        pics.lower(5);
        assert!(!pics.has_pending());
        assert_eq!(pics.master.irr, 0);
    }

    #[test]
    fn level_and_edge_inputs_coexist() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 5);
        pics.raise(5);
        pics.raise(3);
        pics.lower(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
    }

    #[test]
    fn programming_elcr_makes_a_high_line_request_immediately() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 6);
        pics.raise(6);
        pics.io_write(MASTER_ELCR_PORT, 0);
        // Back to edge: the latched request is delivered once and not re-asserted.
        assert_eq!(pics.ack(), Some(MASTER_BASE + 6));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn switching_a_pin_to_level_adopts_the_line_state() {
        let mut pics = new_pics();
        pics.raise(6);
        // A pulse was latched while edge triggered; the line is not held high.
        pics.io_write(MASTER_ELCR_PORT, 1 << 6);
        assert_eq!(pics.master.irr, 0);
    }

    #[test]
    fn icw1_ltim_makes_every_input_level_triggered() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD | ICW1_LEVEL_TRIGGERED);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x04);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.raise(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        pics.lower(3);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
        assert_eq!(pics.io_read(MASTER_ELCR_PORT), 0);
    }

    #[test]
    fn a_later_icw1_without_ltim_returns_to_edge_triggering() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD | ICW1_LEVEL_TRIGGERED);
        init(&mut pics);
        pics.raise(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn level_triggered_line_survives_reinitialization() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 4);
        pics.raise(4);
        init(&mut pics);
        // A line that is still high requests again after the chip is reset.
        assert!(pics.has_pending());
    }

    #[test]
    fn level_triggered_slave_irq_through_the_cascade() {
        let mut pics = new_pics();
        pics.io_write(SLAVE_ELCR_PORT, 1 << 1);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        assert_eq!(pics.slave.irr, 0x02);
        // The device clears its line while the handler runs.
        pics.lower(9);
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(!pics.has_pending());
    }

    #[test]
    fn level_triggered_slave_line_still_high_at_eoi_interrupts_again() {
        let mut pics = new_pics();
        pics.io_write(SLAVE_ELCR_PORT, 1 << 1);
        pics.raise(9);
        pics.ack();
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
    }

    #[test]
    fn raise_and_lower_ignore_the_cascade_pin_and_out_of_range_lines() {
        let mut pics = new_pics();
        pics.raise(CASCADE_IRQ);
        pics.raise(16);
        pics.raise(255);
        assert!(!pics.has_pending());
        assert_eq!(pics.master.irr, 0);
        assert_eq!(pics.slave.irr, 0);
        pics.lower(CASCADE_IRQ);
        pics.lower(200);
        assert_eq!(pics.state(), new_pics().state());
    }

    // Spurious interrupts

    #[test]
    fn spurious_irq7_when_a_level_request_vanishes_before_the_acknowledge() {
        let mut pics = new_pics();
        pics.io_write(MASTER_ELCR_PORT, 1 << 7);
        pics.raise(7);
        assert!(pics.has_pending());
        pics.lower(7);
        assert!(!pics.has_pending());
        // The CPU already saw INT and runs the INTA cycle anyway.
        assert_eq!(pics.inta(), MASTER_BASE + 7);
        // A spurious IRQ 7 does not set ISR, so the guest must not EOI it.
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.master.irr, 0);
    }

    #[test]
    fn spurious_irq7_when_the_request_is_masked_before_the_acknowledge() {
        let mut pics = new_pics();
        pics.raise(3);
        assert!(pics.has_pending());
        set_master_mask(&mut pics, 0x08);
        assert!(!pics.has_pending());
        assert_eq!(pics.inta(), MASTER_BASE + 7);
        assert_eq!(pics.master.isr, 0);
        // The masked request is still latched and is delivered after unmasking.
        assert_eq!(pics.master.irr, 0x08);
        set_master_mask(&mut pics, 0);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn spurious_cycle_answers_base_plus_7_even_with_nothing_requested() {
        let mut pics = new_pics();
        let before = pics.state();
        assert_eq!(pics.inta(), MASTER_BASE + 7);
        assert_eq!(pics.state(), before);
    }

    #[test]
    fn spurious_irq15_when_the_slave_request_is_masked_before_the_acknowledge() {
        let mut pics = new_pics();
        pics.raise(9);
        assert!(pics.has_pending());
        set_slave_mask(&mut pics, 1 << 1);
        // The master still holds IR2 from the slave's earlier INT edge.
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
        // The master's cascade bit is in service; the slave's ISR is untouched, so the guest
        // EOIs the master only.
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.slave.irr, 0x02);
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert!(!pics.has_pending());
    }

    #[test]
    fn masked_slave_request_is_delivered_normally_after_a_spurious_irq15() {
        let mut pics = new_pics();
        pics.raise(9);
        set_slave_mask(&mut pics, 1 << 1);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
        master_command(&mut pics, EOI_NONSPECIFIC);
        set_slave_mask(&mut pics, 0);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        assert_eq!(pics.slave.isr, 0x02);
    }

    #[test]
    fn spurious_irq15_when_a_level_slave_line_drops_before_the_acknowledge() {
        let mut pics = new_pics();
        pics.io_write(SLAVE_ELCR_PORT, 1 << 4);
        pics.raise(12);
        assert!(pics.has_pending());
        pics.lower(12);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
    }

    #[test]
    fn genuine_irq15_is_distinguished_by_its_isr_bit() {
        let mut pics = new_pics();
        pics.raise(15);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
        assert_eq!(pics.slave.isr, 0x80);
    }

    #[test]
    fn genuine_irq7_sets_its_isr_bit() {
        let mut pics = new_pics();
        pics.raise(7);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 7));
        assert_eq!(pics.master.isr, 0x80);
    }

    // Rotation

    #[test]
    fn rotate_on_nonspecific_eoi_makes_the_served_irq_lowest() {
        let mut pics = new_pics();
        pics.raise(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        master_command(&mut pics, EOI_NONSPECIFIC_ROTATE);
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.master.highest_priority, 4);
        pics.raise(3);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 4));
        master_command(&mut pics, EOI_NONSPECIFIC_ROTATE);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn plain_nonspecific_eoi_does_not_rotate() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.ack();
        master_command(&mut pics, EOI_NONSPECIFIC);
        pics.raise(3);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn rotate_on_specific_eoi_makes_the_named_irq_lowest() {
        let mut pics = new_pics();
        pics.raise(5);
        pics.ack();
        master_command(&mut pics, specific_eoi_rotate(5));
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.master.highest_priority, 6);
        pics.raise(5);
        pics.raise(6);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 6));
    }

    #[test]
    fn rotate_on_specific_eoi_for_an_idle_irq_still_rotates() {
        let mut pics = new_pics();
        master_command(&mut pics, specific_eoi_rotate(7));
        assert_eq!(pics.master.highest_priority, 0);
        master_command(&mut pics, specific_eoi_rotate(1));
        assert_eq!(pics.master.highest_priority, 2);
    }

    #[test]
    fn set_priority_command_names_the_lowest_irq() {
        let mut pics = new_pics();
        master_command(&mut pics, set_priority(3));
        assert_eq!(pics.master.highest_priority, 4);
        pics.raise(2);
        pics.raise(5);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
    }

    #[test]
    fn set_priority_command_does_not_clear_in_service() {
        let mut pics = new_pics();
        pics.raise(1);
        pics.ack();
        master_command(&mut pics, set_priority(7));
        assert_eq!(pics.master.isr, 0x02);
        assert_eq!(pics.master.highest_priority, 0);
    }

    #[test]
    fn set_priority_wraps_for_irq7() {
        let mut pics = new_pics();
        master_command(&mut pics, set_priority(7));
        assert_eq!(pics.master.highest_priority, 0);
        master_command(&mut pics, set_priority(0));
        assert_eq!(pics.master.highest_priority, 1);
    }

    #[test]
    fn rotated_priorities_order_all_eight_lines() {
        let mut pics = new_pics();
        master_command(&mut pics, set_priority(3));
        let lines = [0, 1, 3, 4, 5, 6, 7];
        for irq in lines {
            pics.raise(irq);
        }
        let mut order = Vec::new();
        while let Some(vector) = pics.ack() {
            order.push(vector - MASTER_BASE);
            master_command(&mut pics, EOI_NONSPECIFIC);
        }
        assert_eq!(order, [4, 5, 6, 7, 0, 1, 3]);
    }

    #[test]
    fn default_priorities_order_all_eight_lines() {
        let mut pics = new_pics();
        for irq in [7, 6, 5, 4, 3, 1, 0] {
            pics.raise(irq);
        }
        let mut order = Vec::new();
        while let Some(vector) = pics.ack() {
            order.push(vector - MASTER_BASE);
            master_command(&mut pics, EOI_NONSPECIFIC);
        }
        assert_eq!(order, [0, 1, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn rotation_lets_a_higher_numbered_irq_preempt() {
        let mut pics = new_pics();
        pics.raise(2 + 1);
        pics.ack();
        // Make IRQ 3 the lowest priority; IRQ 0 now outranks it, and so does IRQ 7.
        master_command(&mut pics, set_priority(3));
        pics.raise(7);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 7));
    }

    #[test]
    fn nonspecific_eoi_follows_the_rotated_priority_when_clearing() {
        let mut pics = new_pics();
        pics.raise(2 + 1);
        pics.ack();
        // IRQ 3 is lowest priority now, so IRQ 5 preempts it and both are in service.
        master_command(&mut pics, set_priority(3));
        pics.raise(5);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
        assert_eq!(pics.master.isr, 0x28);
        // The highest priority in service is IRQ 5, not the lowest-numbered bit.
        master_command(&mut pics, EOI_NONSPECIFIC);
        assert_eq!(pics.master.isr, 0x08);
    }

    #[test]
    fn slave_rotates_independently_of_the_master() {
        let mut pics = new_pics();
        slave_command(&mut pics, set_priority(1));
        assert_eq!(pics.slave.highest_priority, 2);
        assert_eq!(pics.master.highest_priority, 0);
        pics.raise(9);
        pics.raise(10);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 2));
    }

    // Auto EOI

    #[test]
    fn auto_eoi_never_sets_in_service() {
        let mut pics = Pics::default();
        init_with(&mut pics, ICW4_8086 | ICW4_AUTO_EOI, ICW4_8086);
        pics.raise(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
        assert_eq!(pics.master.isr, 0);
        // A lower-priority request is deliverable straight away.
        pics.raise(6);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 6));
        pics.raise(3);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 3));
    }

    #[test]
    fn auto_eoi_on_the_slave_leaves_the_master_cascade_in_service() {
        let mut pics = Pics::default();
        init_with(&mut pics, ICW4_8086, ICW4_8086 | ICW4_AUTO_EOI);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
    }

    #[test]
    fn auto_eoi_on_the_master_with_a_slave_keeps_slave_in_service() {
        let mut pics = Pics::default();
        init_with(&mut pics, ICW4_8086 | ICW4_AUTO_EOI, ICW4_8086);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        assert_eq!(pics.master.isr, 0);
        assert_eq!(pics.slave.isr, 0x02);
    }

    #[test]
    fn rotate_in_auto_eoi_mode_rotates_on_each_acknowledge() {
        let mut pics = Pics::default();
        init_with(&mut pics, ICW4_8086 | ICW4_AUTO_EOI, ICW4_8086);
        master_command(
            &mut pics,
            OCW2_ROTATE_IN_AUTO_EOI_SET << OCW2_OPERATION_SHIFT,
        );
        pics.raise(3);
        pics.ack();
        assert_eq!(pics.master.highest_priority, 4);
        pics.raise(0);
        pics.ack();
        assert_eq!(pics.master.highest_priority, 1);
        master_command(
            &mut pics,
            OCW2_ROTATE_IN_AUTO_EOI_CLEAR << OCW2_OPERATION_SHIFT,
        );
        pics.raise(5);
        pics.ack();
        assert_eq!(pics.master.highest_priority, 1);
    }

    #[test]
    fn rotate_in_auto_eoi_flag_is_ignored_without_auto_eoi() {
        let mut pics = new_pics();
        master_command(
            &mut pics,
            OCW2_ROTATE_IN_AUTO_EOI_SET << OCW2_OPERATION_SHIFT,
        );
        pics.raise(3);
        pics.ack();
        assert_eq!(pics.master.highest_priority, 0);
        assert_eq!(pics.master.isr, 0x08);
    }

    // Special modes

    #[test]
    fn special_fully_nested_mode_lets_the_master_take_a_higher_slave_irq() {
        let mut pics = Pics::default();
        init_with(&mut pics, ICW4_8086 | ICW4_SPECIAL_FULLY_NESTED, ICW4_8086);
        pics.raise(10);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 2));
        pics.raise(9);
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
        assert_eq!(pics.slave.isr, 0x06);
    }

    #[test]
    fn without_special_fully_nested_mode_the_master_blocks_the_higher_slave_irq() {
        let mut pics = new_pics();
        pics.raise(10);
        pics.ack();
        pics.raise(9);
        assert!(!pics.has_pending());
    }

    #[test]
    fn special_mask_mode_lets_lower_priorities_through_a_masked_in_service_irq() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.ack();
        set_master_mask(&mut pics, 0x08);
        pics.raise(5);
        assert!(!pics.has_pending());
        master_command(&mut pics, OCW3_SPECIAL_MASK_ON);
        assert!(pics.has_pending());
        master_command(&mut pics, OCW3_SPECIAL_MASK_OFF);
        assert!(!pics.has_pending());
        master_command(&mut pics, OCW3_SPECIAL_MASK_ON);
        assert_eq!(pics.ack(), Some(MASTER_BASE + 5));
    }

    #[test]
    fn special_mask_mode_does_not_let_an_unmasked_in_service_irq_be_bypassed() {
        let mut pics = new_pics();
        pics.raise(3);
        pics.ack();
        master_command(&mut pics, OCW3_SPECIAL_MASK_ON);
        pics.raise(5);
        assert!(!pics.has_pending());
    }

    // Poll command

    #[test]
    fn poll_returns_the_irq_with_bit_7_and_acknowledges_it() {
        let mut pics = new_pics();
        pics.raise(4);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x84);
        assert_eq!(pics.master.isr, 0x10);
        assert_eq!(pics.master.irr, 0);
        // One-shot: the next read is an ordinary IRR read.
        pics.raise(1);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x02);
    }

    #[test]
    fn poll_with_nothing_pending_returns_zero() {
        let mut pics = new_pics();
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0);
        assert!(!pics.master.poll);
    }

    #[test]
    fn poll_ignores_masked_requests() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0x10);
        pics.raise(4);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0);
        assert_eq!(pics.master.irr, 0x10);
    }

    #[test]
    fn poll_respects_priority_and_service_state() {
        let mut pics = new_pics();
        pics.raise(5);
        pics.raise(2 + 1);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x83);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        // IRQ 5 is blocked by IRQ 3 in service.
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0);
    }

    #[test]
    fn poll_can_be_read_from_the_data_port() {
        let mut pics = new_pics();
        pics.raise(6);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_DATA_PORT), 0x86);
        assert_eq!(pics.io_read(MASTER_DATA_PORT), 0);
    }

    #[test]
    fn poll_on_the_slave_acknowledges_the_slave_irq() {
        let mut pics = new_pics();
        pics.raise(11);
        slave_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(SLAVE_COMMAND_PORT), 0x83);
        assert_eq!(pics.slave.isr, 0x08);
        assert_eq!(pics.slave.irr, 0);
    }

    #[test]
    fn poll_on_the_master_reports_the_cascade_irq_for_a_slave_request() {
        let mut pics = new_pics();
        pics.raise(11);
        master_command(&mut pics, OCW3_POLL_COMMAND);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x82);
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
        // The slave still holds the request for its own poll.
        assert_eq!(pics.slave.irr, 0x08);
    }

    #[test]
    fn poll_command_can_be_combined_with_a_register_selection() {
        let mut pics = new_pics();
        pics.raise(1);
        master_command(
            &mut pics,
            OCW3_POLL_COMMAND | OCW3_READ_REGISTER | OCW3_READ_ISR,
        );
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x81);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x02);
    }

    // Cascade identification

    #[test]
    fn master_without_a_slave_on_ir2_treats_it_as_an_ordinary_line() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x00);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.io_write(SLAVE_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(SLAVE_DATA_PORT, SLAVE_BASE);
        pics.io_write(SLAVE_DATA_PORT, 0x02);
        pics.io_write(SLAVE_DATA_PORT, ICW4_8086);
        pics.raise(9);
        assert!(pics.has_pending());
        // The master answers for IR2 itself; the slave is never asked.
        assert_eq!(pics.ack(), Some(MASTER_BASE + CASCADE_IRQ));
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.slave.irr, 0x02);
    }

    #[test]
    fn single_mode_master_has_no_slave() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_INIT | ICW1_IC4 | ICW1_SINGLE);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.io_write(SLAVE_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(SLAVE_DATA_PORT, SLAVE_BASE);
        pics.io_write(SLAVE_DATA_PORT, 0x02);
        pics.io_write(SLAVE_DATA_PORT, ICW4_8086);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(MASTER_BASE + CASCADE_IRQ));
        assert_eq!(pics.slave.isr, 0);
    }

    #[test]
    fn master_icw3_only_needs_bit_2_set() {
        let mut pics = Pics::default();
        pics.io_write(MASTER_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0xff);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(pics.slave.base + 1));
        assert_eq!(pics.slave.isr, 0x02);
    }

    #[test]
    fn slave_with_a_different_id_does_not_answer_the_cascade_cycle() {
        let mut pics = Pics::default();
        init(&mut pics);
        pics.io_write(SLAVE_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(SLAVE_DATA_PORT, SLAVE_BASE);
        pics.io_write(SLAVE_DATA_PORT, 0x01);
        pics.io_write(SLAVE_DATA_PORT, ICW4_8086);
        pics.raise(9);
        assert!(pics.has_pending());
        // Nobody drives the bus; the master still counts the cascade IRQ as serviced.
        assert_eq!(pics.ack(), Some(OPEN_BUS));
        assert_eq!(pics.master.isr, 1 << CASCADE_IRQ);
        assert_eq!(pics.slave.isr, 0);
        assert_eq!(pics.slave.irr, 0x02);
    }

    #[test]
    fn slave_id_uses_only_the_low_three_bits_of_icw3() {
        let mut pics = Pics::default();
        init(&mut pics);
        pics.io_write(SLAVE_COMMAND_PORT, ICW1_STANDARD);
        pics.io_write(SLAVE_DATA_PORT, SLAVE_BASE);
        pics.io_write(SLAVE_DATA_PORT, 0xfa);
        pics.io_write(SLAVE_DATA_PORT, ICW4_8086);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 1));
    }

    #[test]
    fn reinitializing_the_master_discards_a_latched_cascade_request() {
        let mut pics = new_pics();
        pics.raise(9);
        assert!(pics.has_pending());
        master_command(&mut pics, ICW1_STANDARD);
        pics.io_write(MASTER_DATA_PORT, MASTER_BASE);
        pics.io_write(MASTER_DATA_PORT, 0x04);
        pics.io_write(MASTER_DATA_PORT, ICW4_8086);
        assert!(!pics.has_pending());
    }

    #[test]
    fn reinitializing_the_slave_withdraws_its_interrupt() {
        let mut pics = new_pics();
        pics.raise(9);
        slave_command(&mut pics, ICW1_STANDARD);
        pics.io_write(SLAVE_DATA_PORT, SLAVE_BASE);
        pics.io_write(SLAVE_DATA_PORT, 0x02);
        pics.io_write(SLAVE_DATA_PORT, ICW4_8086);
        // The master latched IR2 earlier, so INT stays asserted and the cycle is spurious.
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(SLAVE_BASE + 7));
        assert_eq!(pics.slave.isr, 0);
    }

    // Regression and integration flows

    #[test]
    fn linux_style_irq_flow() {
        let mut pics = new_pics();
        assert_eq!(serve_master_irq(&mut pics, 0), MASTER_BASE);
        pics.raise(4);
        assert_eq!(pics.ack(), Some(0x24));
        pics.raise(3);
        assert_eq!(pics.ack(), Some(0x23));
        master_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        pics.raise(9);
        assert_eq!(pics.ack(), Some(0x29));
        assert_eq!(pics.ack(), None);
        slave_command(&mut pics, EOI_NONSPECIFIC);
        master_command(&mut pics, EOI_NONSPECIFIC);
        set_master_mask(&mut pics, 0x10);
        pics.raise(4);
        assert_eq!(pics.ack(), None);
    }

    #[test]
    fn linux_probes_irqs_by_masking_and_reading_irr() {
        let mut pics = new_pics();
        set_master_mask(&mut pics, 0xfb);
        set_slave_mask(&mut pics, 0xff);
        pics.raise(4);
        master_command(&mut pics, OCW3_READ_IRR);
        assert_eq!(pics.io_read(MASTER_COMMAND_PORT), 0x10);
        assert_eq!(pics.io_read(MASTER_DATA_PORT), 0xfb);
        assert_eq!(pics.io_read(SLAVE_DATA_PORT), 0xff);
    }

    #[test]
    fn the_acknowledge_with_nothing_pending_gives_no_vector() {
        let mut pics = new_pics();
        assert_eq!(pics.ack(), None);
        pics.raise(1);
        pics.ack();
        pics.raise(1);
        assert_eq!(pics.ack(), None);
    }

    // Randomized consistency

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }

        fn below(&mut self, limit: u64) -> u64 {
            self.next() % limit
        }
    }

    /// Straight-line model of when one chip asserts INT: walk the priorities from the highest;
    /// a request wins unless something at or above it is in service.
    fn model_int_asserted(pic: &Pic) -> bool {
        let requested = pic.irr & !pic.imr;
        let mut in_service = pic.isr;
        if pic.special_mask {
            in_service &= !pic.imr;
        }
        if pic.special_fully_nested && pic.is_master {
            in_service &= !(1 << CASCADE_IRQ);
        }
        for rank in 0..IRQS_PER_PIC {
            let irq = (pic.highest_priority + rank) % IRQS_PER_PIC;
            if in_service & (1 << irq) != 0 {
                return false;
            }
            if requested & (1 << irq) != 0 {
                return true;
            }
        }
        false
    }

    fn reinitialize(pics: &mut Pics, rng: &mut Xorshift, master: bool) {
        let (command, data) = if master {
            (MASTER_COMMAND_PORT, MASTER_DATA_PORT)
        } else {
            (SLAVE_COMMAND_PORT, SLAVE_DATA_PORT)
        };
        let ic4 = rng.below(4) != 0;
        let single = rng.below(8) == 0;
        let level = rng.below(6) == 0;
        let mut icw1 = ICW1_INIT;
        if ic4 {
            icw1 |= ICW1_IC4;
        }
        if single {
            icw1 |= ICW1_SINGLE;
        }
        if level {
            icw1 |= ICW1_LEVEL_TRIGGERED;
        }
        pics.io_write(command, icw1);
        let base = if master { MASTER_BASE } else { SLAVE_BASE };
        pics.io_write(data, base | (rng.byte() & 7));
        if !single {
            let icw3 = if master {
                1 << CASCADE_IRQ
            } else {
                CASCADE_IRQ
            };
            pics.io_write(data, if rng.below(8) == 0 { rng.byte() } else { icw3 });
        }
        if ic4 {
            let mut icw4 = ICW4_8086;
            if rng.below(4) == 0 {
                icw4 |= ICW4_AUTO_EOI;
            }
            if rng.below(4) == 0 {
                icw4 |= ICW4_SPECIAL_FULLY_NESTED;
            }
            pics.io_write(data, icw4);
        }
    }

    fn random_operation(pics: &mut Pics, rng: &mut Xorshift) {
        let port_pair = |master: bool| {
            if master {
                (MASTER_COMMAND_PORT, MASTER_DATA_PORT, MASTER_ELCR_PORT)
            } else {
                (SLAVE_COMMAND_PORT, SLAVE_DATA_PORT, SLAVE_ELCR_PORT)
            }
        };
        let master = rng.below(2) == 0;
        let (command, data, elcr) = port_pair(master);
        match rng.below(14) {
            0..=2 => pics.raise(rng.below(IRQ_COUNT as u64) as u8),
            3 => pics.lower(rng.below(IRQ_COUNT as u64) as u8),
            4 => pics.io_write(data, rng.byte()),
            5 | 6 => {
                // OCW2: bits 4:3 clear.
                pics.io_write(command, rng.byte() & !(ICW1_INIT | OCW3_MARKER));
            }
            7 => {
                let mut ocw3 = OCW3_MARKER | (rng.byte() & 0x63);
                ocw3 &= !ICW1_INIT;
                pics.io_write(command, ocw3);
            }
            8 => pics.io_write(elcr, rng.byte()),
            9..=11 => {
                pics.ack();
            }
            12 => {
                if rng.below(6) == 0 {
                    reinitialize(pics, rng, master);
                }
            }
            _ => {
                // Poll cycle.
                pics.io_write(command, OCW3_POLL_COMMAND);
                pics.io_read(command);
            }
        }
    }

    #[test]
    fn has_pending_matches_an_independent_priority_model() {
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        for _ in 0..300 {
            let mut pics = Pics::default();
            init(&mut pics);
            for _ in 0..150 {
                random_operation(&mut pics, &mut rng);
                assert_eq!(
                    pics.has_pending(),
                    model_int_asserted(&pics.master),
                    "{}",
                    pics.state()
                );
                // The cheap early-out never rejects what the full resolution accepts.
                if pics.master.pick_irq().is_some() {
                    assert_ne!(pics.master.irr & !pics.master.imr, 0);
                    assert!(pics.has_pending());
                }
            }
        }
    }

    #[test]
    fn ack_agrees_with_has_pending_and_spurious_cycles_change_nothing() {
        let mut rng = Xorshift(0x1357_9bdf_2468_ace0);
        let mut deliveries = 0u32;
        let mut spurious = 0u32;
        for _ in 0..300 {
            let mut pics = Pics::default();
            init(&mut pics);
            for _ in 0..150 {
                random_operation(&mut pics, &mut rng);
                let pending = pics.has_pending();

                let mut probe = pics.clone();
                let before = probe.state();
                let vector = probe.inta();
                if !pending {
                    assert_eq!(vector, pics.master.base + SPURIOUS_IRQ, "{before}");
                    assert_eq!(probe.state(), before);
                    assert_eq!(probe.master.highest_priority, pics.master.highest_priority);
                } else {
                    let base = vector & ICW2_VECTOR_MASK;
                    assert!(
                        base == pics.master.base || base == pics.slave.base || vector == OPEN_BUS,
                        "vector {vector:#x} {before}"
                    );
                    deliveries += 1;
                    if vector & 7 == SPURIOUS_IRQ {
                        spurious += 1;
                    }
                }

                let acked = pics.ack();
                assert_eq!(acked.is_some(), pending, "{before}");
                if let Some(acked) = acked {
                    assert_eq!(acked, vector);
                    assert_eq!(pics.state(), probe.state());
                }
            }
        }
        assert!(deliveries > 1000, "{deliveries}");
        assert!(spurious > 0, "random runs should include spurious cycles");
    }

    #[test]
    fn requests_are_never_lost_without_a_reset_or_a_withdrawn_level() {
        // Raise a set of edge IRQs, then ack/EOI to completion: each is delivered exactly once.
        let mut rng = Xorshift(0xdead_beef_cafe_f00d);
        for _ in 0..500 {
            let mut pics = new_pics();
            let mut wanted = 0u16;
            for irq in 0..IRQ_COUNT {
                if irq != CASCADE_IRQ && rng.below(2) == 0 {
                    pics.raise(irq);
                    wanted |= 1 << irq;
                }
            }
            let mut seen = 0u16;
            while let Some(vector) = pics.ack() {
                let irq = if vector >= SLAVE_BASE {
                    vector - SLAVE_BASE + IRQS_PER_PIC
                } else {
                    vector - MASTER_BASE
                };
                assert_eq!(seen & (1 << irq), 0, "irq {irq} delivered twice");
                seen |= 1 << irq;
                if irq >= IRQS_PER_PIC {
                    slave_command(&mut pics, EOI_NONSPECIFIC);
                }
                master_command(&mut pics, EOI_NONSPECIFIC);
            }
            assert_eq!(seen, wanted);
        }
    }
}
