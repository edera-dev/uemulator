//! Local APIC: xAPIC MMIO window, x2APIC MSR interface, the timer (one-shot, periodic,
//! TSC-deadline), the interrupt request/service registers and the LVT. One `Lapic` per vCPU;
//! an ICR write only describes the IPI (`Ipi`), `Devices` routes it to the other APICs.
//!
//! SDM vol. 3 chapter 12 (Intel) and AMD APM vol. 2 chapter 16 describe the same register
//! file; QEMU's `hw/intc/apic.c` is the behavioral reference for what guests actually rely on.

pub(crate) const APIC_DEFAULT_BASE: u64 = 0xfee0_0000;
pub(crate) const APIC_MMIO_SIZE: u64 = 0x1000;

/// IA32_APIC_BASE bits.
pub(crate) const APIC_BASE_BSP: u64 = 1 << 8;
pub(crate) const APIC_BASE_X2APIC: u64 = 1 << 10;
pub(crate) const APIC_BASE_ENABLE: u64 = 1 << 11;
/// Bits 12..=51 hold the page-aligned base address.
const APIC_BASE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const APIC_BASE_RESERVED_MASK: u64 =
    !(APIC_BASE_ADDRESS_MASK | APIC_BASE_BSP | APIC_BASE_X2APIC | APIC_BASE_ENABLE);

/// x2APIC register MSRs: 0x800 + (xAPIC MMIO offset >> 4).
pub(crate) const X2APIC_MSR_FIRST: u32 = 0x800;
pub(crate) const X2APIC_MSR_LAST: u32 = 0x8ff;
pub(crate) const MSR_TSC_DEADLINE: u32 = 0x6e0;

const REG_ID: u32 = 0x20;
const REG_VERSION: u32 = 0x30;
const REG_TPR: u32 = 0x80;
const REG_APR: u32 = 0x90;
const REG_PPR: u32 = 0xa0;
const REG_EOI: u32 = 0xb0;
const REG_RRD: u32 = 0xc0;
const REG_LDR: u32 = 0xd0;
const REG_DFR: u32 = 0xe0;
const REG_SVR: u32 = 0xf0;
const REG_ISR: u32 = 0x100;
const REG_TMR: u32 = 0x180;
const REG_IRR: u32 = 0x200;
const REG_ESR: u32 = 0x280;
const REG_LVT_CMCI: u32 = 0x2f0;
const REG_ICR_LOW: u32 = 0x300;
const REG_ICR_HIGH: u32 = 0x310;
const REG_LVT_TIMER: u32 = 0x320;
const REG_LVT_THERMAL: u32 = 0x330;
const REG_LVT_PERFORMANCE: u32 = 0x340;
const REG_LVT_LINT0: u32 = 0x350;
const REG_LVT_LINT1: u32 = 0x360;
const REG_LVT_ERROR: u32 = 0x370;
const REG_TIMER_INITIAL: u32 = 0x380;
const REG_TIMER_CURRENT: u32 = 0x390;
const REG_TIMER_DIVIDE: u32 = 0x3e0;
/// x2APIC only: the self-IPI register.
const REG_SELF_IPI: u32 = 0x3f0;
/// AMD extended APIC space (feature register at 0x400); not implemented, reads as zero.
const REG_EXTENDED_FEATURE: u32 = 0x400;
const REG_EXTENDED_LAST: u32 = 0x530;

/// Version register: version 0x14 (integrated APIC), maximum LVT entry 6 (so LVT entries up to
/// CMCI are valid), and the EOI-broadcast-suppression bit clear.
const VERSION_VALUE: u32 = 0x14 | (6 << 16);

const SVR_SOFTWARE_ENABLE: u32 = 1 << 8;
const SVR_WRITABLE: u32 = 0x1ff | (1 << 12);
const SVR_RESET: u32 = 0xff;

const LVT_DELIVERY_SHIFT: u32 = 8;
const LVT_DELIVERY_MASK: u32 = 0b111 << LVT_DELIVERY_SHIFT;
const LVT_LEVEL: u32 = 1 << 15;
const LVT_MASKED: u32 = 1 << 16;
const LVT_TIMER_MODE_SHIFT: u32 = 17;
const LVT_TIMER_MODE_MASK: u32 = 0b11 << LVT_TIMER_MODE_SHIFT;
const LVT_RESET: u32 = LVT_MASKED;

pub(crate) const DELIVERY_FIXED: u32 = 0;
pub(crate) const DELIVERY_LOWEST_PRIORITY: u32 = 1;
pub(crate) const DELIVERY_NMI: u32 = 4;
pub(crate) const DELIVERY_INIT: u32 = 5;
pub(crate) const DELIVERY_STARTUP: u32 = 6;
pub(crate) const DELIVERY_EXTINT: u32 = 7;

const TIMER_MODE_PERIODIC: u32 = 1;
const TIMER_MODE_TSC_DEADLINE: u32 = 2;

/// The timer counts the bus clock divided by the divide configuration. 100 MHz, as in QEMU's
/// APIC (`APIC_BUS_CYCLE_NS` is 1; real parts run 100 to 133 MHz); a guest calibrates it.
const BUS_CLOCK_PERIOD_NS: u64 = 10;
const DIVIDE_RESET: u32 = 0;
const DIVIDE_WRITABLE: u32 = 0b1011;
/// A periodic timer with a tiny initial count would fire once per bus clock and starve the
/// guest; real hardware is bounded by what the CPU can service. Clamp the period.
const MIN_TIMER_PERIOD_NS: u64 = 1000;
/// `tick` fires at most this many periods per call, so a guest that stalls the clock cannot
/// make one tick loop for seconds of catch-up.
const MAX_TIMER_CATCH_UP: u32 = 4;

const ICR_VECTOR_MASK: u32 = 0xff;
const ICR_DELIVERY_SHIFT: u32 = 8;
const ICR_LOGICAL_DESTINATION: u32 = 1 << 11;
/// Level-assert bit. Only INIT looks at it: INIT with level trigger and this bit clear is the
/// "de-assert" that old IPI sequences send, and does nothing (SDM 12.6.1).
const ICR_LEVEL_ASSERT: u32 = 1 << 14;
const ICR_LEVEL_TRIGGER: u32 = 1 << 15;
const ICR_SHORTHAND_SHIFT: u32 = 18;
const ICR_SHORTHAND_MASK: u32 = 0b11;
pub(crate) const SHORTHAND_NONE: u32 = 0;
pub(crate) const SHORTHAND_SELF: u32 = 1;
pub(crate) const SHORTHAND_ALL_INCLUDING_SELF: u32 = 2;
pub(crate) const SHORTHAND_ALL_EXCLUDING_SELF: u32 = 3;
const ICR_WRITABLE_XAPIC: u32 = 0x000c_cfff;

/// Error status bits.
const ESR_RECEIVE_ILLEGAL_VECTOR: u32 = 1 << 6;
const ESR_SEND_ILLEGAL_VECTOR: u32 = 1 << 5;
const ESR_ILLEGAL_REGISTER: u32 = 1 << 7;
/// Vectors below this are reserved for exceptions; delivering one is an illegal vector.
const FIRST_VALID_VECTOR: u8 = 16;

const BROADCAST_XAPIC: u32 = 0xff;
const BROADCAST_X2APIC: u32 = 0xffff_ffff;

const DFR_FLAT: u32 = 0xffff_ffff;
const DFR_CLUSTER: u32 = 0x0fff_ffff;

/// What an accepted interrupt asks the CPU to do besides setting a request bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Delivered {
    #[default]
    Nothing,
    Nmi,
}

/// An interrupt arriving from the I/O APIC, an MSI, or an IPI.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Interrupt {
    pub vector: u8,
    pub delivery_mode: u32,
    pub level_triggered: bool,
    pub destination: u32,
    pub logical: bool,
}

/// What an ICR write asks `Devices` to send: the message and which processors it goes to when
/// the shorthand is not `SHORTHAND_NONE` (then `interrupt.destination` is the target).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ipi {
    pub interrupt: Interrupt,
    pub shorthand: u32,
}

pub(crate) struct Lapic {
    /// IA32_APIC_BASE.
    base: u64,
    /// The APIC id the processor was created with, which a software reset must keep.
    reset_id: u32,
    id: u32,
    tpr: u8,
    ldr: u32,
    dfr: u32,
    svr: u32,
    esr: u32,
    esr_pending: u32,
    irr: [u32; 8],
    isr: [u32; 8],
    tmr: [u32; 8],
    lvt_cmci: u32,
    lvt_timer: u32,
    lvt_thermal: u32,
    lvt_performance: u32,
    lvt_lint0: u32,
    lvt_lint1: u32,
    lvt_error: u32,
    icr_high: u32,
    timer_initial: u32,
    divide_config: u32,
    /// Virtual ns at which the current count was loaded.
    timer_start_ns: u64,
    /// Next expiry, or `None` while stopped.
    timer_deadline_ns: Option<u64>,
    tsc_deadline: u64,
    /// An NMI the CPU has not taken yet.
    nmi_pending: bool,
    /// INIT and start-up IPIs wait here for the processor itself to act on them
    /// (`Smp::apply_start_signals`), because resetting a vCPU needs the vCPU.
    init_pending: bool,
    startup_vector: Option<u8>,
}

fn bit(regs: &[u32; 8], vector: u8) -> bool {
    regs[(vector >> 5) as usize] & (1 << (vector & 31)) != 0
}

fn set_bit(regs: &mut [u32; 8], vector: u8) {
    regs[(vector >> 5) as usize] |= 1 << (vector & 31);
}

fn clear_bit(regs: &mut [u32; 8], vector: u8) {
    regs[(vector >> 5) as usize] &= !(1 << (vector & 31));
}

/// Highest set vector in a 256-bit register file, if any.
fn highest(regs: &[u32; 8]) -> Option<u8> {
    for word in (0..8).rev() {
        if regs[word] != 0 {
            return Some((word as u8) * 32 + 31 - regs[word].leading_zeros() as u8);
        }
    }
    None
}

impl Lapic {
    /// Power-on state of the APIC of processor `id`; only the bootstrap processor (id 0) has the
    /// BSP bit.
    pub fn new(id: u32) -> Self {
        Lapic {
            base: APIC_DEFAULT_BASE | APIC_BASE_ENABLE | if id == 0 { APIC_BASE_BSP } else { 0 },
            reset_id: id,
            id,
            tpr: 0,
            ldr: 0,
            dfr: DFR_FLAT,
            svr: SVR_RESET,
            esr: 0,
            esr_pending: 0,
            irr: [0; 8],
            isr: [0; 8],
            tmr: [0; 8],
            lvt_cmci: LVT_RESET,
            lvt_timer: LVT_RESET,
            lvt_thermal: LVT_RESET,
            lvt_performance: LVT_RESET,
            // Virtual wire: the BIOS leaves LINT0 as ExtINT and LINT1 as NMI, and Linux relies on
            // that when it boots without ACPI or MP tables (`APIC_VIRTUAL_WIRE_NO_CONFIG`).
            lvt_lint0: DELIVERY_EXTINT << LVT_DELIVERY_SHIFT,
            lvt_lint1: (DELIVERY_NMI << LVT_DELIVERY_SHIFT) | LVT_MASKED,
            lvt_error: LVT_RESET,
            icr_high: 0,
            timer_initial: 0,
            divide_config: DIVIDE_RESET,
            timer_start_ns: 0,
            timer_deadline_ns: None,
            tsc_deadline: 0,
            nmi_pending: false,
            init_pending: false,
            startup_vector: None,
        }
    }

    pub fn base_msr(&self) -> u64 {
        self.base
    }

    /// `false` for an invalid transition or reserved bits (#GP). The enable and x2APIC bits
    /// follow SDM table 12-5: x2APIC without enable is invalid, and once in x2APIC mode only a
    /// full disable (clearing both) is allowed, not a drop back to xAPIC.
    pub fn set_base_msr(&mut self, value: u64) -> bool {
        if value & APIC_BASE_RESERVED_MASK != 0 {
            return false;
        }
        let enable = value & APIC_BASE_ENABLE != 0;
        let x2apic = value & APIC_BASE_X2APIC != 0;
        if x2apic && !enable {
            return false;
        }
        let was_x2apic = self.base & APIC_BASE_X2APIC != 0;
        if was_x2apic && enable && !x2apic {
            return false;
        }
        // The BSP bit is read-only.
        self.base = (value & !APIC_BASE_BSP) | (self.base & APIC_BASE_BSP);
        if !enable {
            self.software_reset();
        }
        if x2apic && !was_x2apic {
            // The xAPIC id becomes the 32-bit x2APIC id; this CPU keeps id 0 and the logical
            // destination is derived from it (SDM 12.12.10.2).
            self.ldr = self.derived_logical_id();
        }
        true
    }

    fn derived_logical_id(&self) -> u32 {
        ((self.id >> 4) << 16) | (1 << (self.id & 0xf))
    }

    /// Guest disabled the APIC: it drops everything it had. Re-enabling starts from reset values.
    fn software_reset(&mut self) {
        let base = self.base;
        let id = self.id;
        *self = Lapic::new(self.reset_id);
        self.base = base;
        self.id = id;
    }

    /// INIT: every register goes back to its reset value except the APIC id and the base MSR
    /// (SDM 9.1.1, table 9-1; the APIC stays enabled in whatever mode it was).
    pub fn init_reset(&mut self) {
        self.software_reset();
    }

    #[cfg(test)]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// An INIT or start-up IPI is waiting for the processor.
    pub fn start_signal_pending(&self) -> bool {
        self.init_pending || self.startup_vector.is_some()
    }

    pub fn init_pending(&self) -> bool {
        self.init_pending
    }

    pub fn take_init(&mut self) -> bool {
        core::mem::take(&mut self.init_pending)
    }

    pub fn take_startup(&mut self) -> Option<u8> {
        self.startup_vector.take()
    }

    pub fn hardware_enabled(&self) -> bool {
        self.base & APIC_BASE_ENABLE != 0
    }

    pub fn x2apic(&self) -> bool {
        self.base & APIC_BASE_X2APIC != 0
    }

    pub fn software_enabled(&self) -> bool {
        self.hardware_enabled() && self.svr & SVR_SOFTWARE_ENABLE != 0
    }

    pub fn mmio_base(&self) -> u64 {
        self.base & APIC_BASE_ADDRESS_MASK
    }

    /// The MMIO window answers only in xAPIC mode.
    pub fn mmio_decodes(&self, phys: u64) -> bool {
        self.hardware_enabled()
            && !self.x2apic()
            && phys >= self.mmio_base()
            && phys < self.mmio_base() + APIC_MMIO_SIZE
    }

    pub fn tpr(&self) -> u8 {
        self.tpr
    }

    /// CR8 holds TPR[7:4].
    pub fn set_tpr(&mut self, tpr: u8) {
        self.tpr = tpr;
    }

    fn processor_priority(&self) -> u8 {
        let in_service = highest(&self.isr).unwrap_or(0) & 0xf0;
        (self.tpr & 0xf0).max(in_service)
    }

    /// What a lowest-priority interrupt compares: the processor priority raised by the highest
    /// request still waiting (the arbitration priority, SDM 12.9.7), smaller wins.
    pub fn arbitration_priority(&self) -> u8 {
        let requested = highest(&self.irr).unwrap_or(0) & 0xf0;
        self.processor_priority().max(requested)
    }

    /// The vector the CPU would take now, if interrupts were enabled. A request is deliverable
    /// when its priority class exceeds the processor priority (SDM 12.8.3.1).
    pub fn pending_vector(&self) -> Option<u8> {
        if !self.software_enabled() {
            return None;
        }
        let vector = highest(&self.irr)?;
        (vector & 0xf0 > self.processor_priority()).then_some(vector)
    }

    /// Interrupt acknowledge: moves the highest request to in-service.
    pub fn ack(&mut self) -> Option<u8> {
        let vector = self.pending_vector()?;
        clear_bit(&mut self.irr, vector);
        set_bit(&mut self.isr, vector);
        Some(vector)
    }

    /// Ends the highest in-service interrupt and returns it when it was level triggered, so the
    /// caller can notify the I/O APIC (EOI broadcast).
    pub fn eoi(&mut self) -> Option<u8> {
        let vector = highest(&self.isr)?;
        clear_bit(&mut self.isr, vector);
        bit(&self.tmr, vector).then_some(vector)
    }

    /// Whether the PIC's INTR output reaches the CPU: with the APIC off it is wired straight in,
    /// with it on only through LINT0 programmed as an unmasked ExtINT (virtual wire mode).
    ///
    /// Like QEMU's `apic_accept_pic_intr` this looks at the mask bit only: LINT0 comes out of
    /// reset as ExtINT (what firmware leaves), so a direct-boot guest has the PIC connected
    /// until it masks LINT0 or software-disables the APIC (which masks every LVT entry).
    pub fn extint_connected(&self) -> bool {
        !self.hardware_enabled() || self.lvt_lint0 & LVT_MASKED == 0
    }

    pub fn take_nmi(&mut self) -> bool {
        core::mem::take(&mut self.nmi_pending)
    }

    pub fn nmi_pending(&self) -> bool {
        self.nmi_pending
    }

    pub fn matches_destination(&self, interrupt: &Interrupt) -> bool {
        let broadcast = if self.x2apic() {
            BROADCAST_X2APIC
        } else {
            BROADCAST_XAPIC
        };
        if !interrupt.logical {
            return interrupt.destination == self.id || interrupt.destination == broadcast;
        }
        if self.x2apic() {
            // Cluster model: the high half selects the cluster, the low half is a bit mask.
            let cluster = interrupt.destination >> 16 == self.ldr >> 16;
            return cluster && interrupt.destination & self.ldr & 0xffff != 0;
        }
        let logical_id = self.ldr >> 24;
        let destination = interrupt.destination & 0xff;
        if self.dfr == DFR_FLAT {
            destination & logical_id != 0
        } else {
            destination >> 4 == logical_id >> 4 && destination & logical_id & 0xf != 0
        }
    }

    /// Accepts an interrupt message addressed to this APIC. Returns what the CPU must do besides
    /// polling `pending_vector`.
    pub fn accept(&mut self, interrupt: &Interrupt) -> Delivered {
        if !self.hardware_enabled() || !self.matches_destination(interrupt) {
            return Delivered::Nothing;
        }
        self.accept_addressed(interrupt)
    }

    /// `accept` for a message that is already known to be for this APIC (a shorthand IPI).
    pub fn accept_addressed(&mut self, interrupt: &Interrupt) -> Delivered {
        if !self.hardware_enabled() {
            return Delivered::Nothing;
        }
        match interrupt.delivery_mode {
            DELIVERY_FIXED | DELIVERY_LOWEST_PRIORITY => {
                if interrupt.vector < FIRST_VALID_VECTOR {
                    self.esr_pending |= ESR_RECEIVE_ILLEGAL_VECTOR;
                    return Delivered::Nothing;
                }
                // A software-disabled APIC accepts only NMI, SMI, INIT, ExtINT and startup.
                if !self.software_enabled() {
                    return Delivered::Nothing;
                }
                set_bit(&mut self.irr, interrupt.vector);
                if interrupt.level_triggered {
                    set_bit(&mut self.tmr, interrupt.vector);
                } else {
                    clear_bit(&mut self.tmr, interrupt.vector);
                }
                Delivered::Nothing
            }
            DELIVERY_NMI => {
                self.nmi_pending = true;
                Delivered::Nmi
            }
            DELIVERY_INIT => {
                self.init_pending = true;
                Delivered::Nothing
            }
            // The vector field is the start page (address = vector << 12).
            DELIVERY_STARTUP => {
                self.startup_vector = Some(interrupt.vector);
                Delivered::Nothing
            }
            // SMI: there is no SMM.
            _ => Delivered::Nothing,
        }
    }

    fn lvt_deliver(&mut self, lvt: u32) -> Delivered {
        if lvt & LVT_MASKED != 0 {
            return Delivered::Nothing;
        }
        let interrupt = Interrupt {
            vector: lvt as u8,
            delivery_mode: (lvt & LVT_DELIVERY_MASK) >> LVT_DELIVERY_SHIFT,
            level_triggered: lvt & LVT_LEVEL != 0,
            destination: self.id,
            logical: false,
        };
        // ExtINT through a local vector is handled by the PIC path (`extint_connected`).
        self.accept(&interrupt)
    }

    fn timer_mode(&self) -> u32 {
        (self.lvt_timer & LVT_TIMER_MODE_MASK) >> LVT_TIMER_MODE_SHIFT
    }

    fn divider(&self) -> u64 {
        // The configuration encodes 1, 2, 4, ..., 128 as bits 0, 1, 3 (bit 2 is reserved):
        // value 0b111 is divide by 1, the rest count up from 2 (SDM figure 12-10).
        let value = (self.divide_config & 0b11) | ((self.divide_config & 0b1000) >> 1);
        if value == 0b111 { 1 } else { 2u64 << value }
    }

    fn timer_period_ns(&self) -> u64 {
        (self.timer_initial as u64 * BUS_CLOCK_PERIOD_NS * self.divider()).max(MIN_TIMER_PERIOD_NS)
    }

    fn arm_timer(&mut self, now: u64) {
        self.timer_start_ns = now;
        self.timer_deadline_ns = match self.timer_mode() {
            TIMER_MODE_TSC_DEADLINE => (self.tsc_deadline != 0).then_some(self.tsc_deadline),
            _ if self.timer_initial == 0 => None,
            _ => Some(now + self.timer_period_ns()),
        };
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.timer_deadline_ns
    }

    /// Fires an expired timer.
    pub fn tick(&mut self, now: u64) {
        for _ in 0..MAX_TIMER_CATCH_UP {
            let Some(deadline) = self.timer_deadline_ns else {
                break;
            };
            if now < deadline {
                break;
            }
            let lvt = self.lvt_timer;
            if self.timer_mode() == TIMER_MODE_PERIODIC {
                self.timer_start_ns = deadline;
                self.timer_deadline_ns = Some(deadline + self.timer_period_ns());
            } else {
                self.timer_deadline_ns = None;
                if self.timer_mode() == TIMER_MODE_TSC_DEADLINE {
                    self.tsc_deadline = 0;
                }
            }
            self.lvt_deliver(lvt);
        }
        // A deadline far in the past (guest idled) collapses to one period ahead.
        if let Some(deadline) = self.timer_deadline_ns
            && deadline < now
            && self.timer_mode() == TIMER_MODE_PERIODIC
        {
            self.timer_deadline_ns = Some(now + self.timer_period_ns());
        }
    }

    fn timer_current_count(&self, now: u64) -> u32 {
        let Some(deadline) = self.timer_deadline_ns else {
            return 0;
        };
        if self.timer_mode() == TIMER_MODE_TSC_DEADLINE {
            return 0;
        }
        let tick_ns = BUS_CLOCK_PERIOD_NS * self.divider();
        let remaining_ns = deadline.saturating_sub(now);
        (remaining_ns / tick_ns).min(self.timer_initial as u64) as u32
    }

    pub fn tsc_deadline(&self) -> u64 {
        self.tsc_deadline
    }

    /// IA32_TSC_DEADLINE. Writing arms the timer when the LVT is in deadline mode; a nonzero
    /// value in the past fires at the next tick. Zero disarms.
    pub fn set_tsc_deadline(&mut self, value: u64, now: u64) {
        self.tsc_deadline = value;
        if self.timer_mode() == TIMER_MODE_TSC_DEADLINE {
            self.timer_deadline_ns = (value != 0).then_some(value.max(now));
        }
    }

    fn read_lvt(&self, offset: u32) -> Option<u32> {
        Some(match offset {
            REG_LVT_CMCI => self.lvt_cmci,
            REG_LVT_TIMER => self.lvt_timer,
            REG_LVT_THERMAL => self.lvt_thermal,
            REG_LVT_PERFORMANCE => self.lvt_performance,
            REG_LVT_LINT0 => self.lvt_lint0,
            REG_LVT_LINT1 => self.lvt_lint1,
            REG_LVT_ERROR => self.lvt_error,
            _ => return None,
        })
    }

    /// Register read by xAPIC MMIO offset or x2APIC MSR; `None` is an access to a reserved
    /// register (#GP in x2APIC mode, ignored in xAPIC mode where it sets the illegal-register bit).
    pub fn read_register(&mut self, offset: u32, now: u64) -> Option<u32> {
        if let Some(value) = self.read_lvt(offset) {
            return Some(value);
        }
        Some(match offset {
            REG_ID => {
                if self.x2apic() {
                    self.id
                } else {
                    self.id << 24
                }
            }
            REG_VERSION => VERSION_VALUE,
            REG_TPR => self.tpr as u32,
            REG_APR => self.processor_priority() as u32,
            REG_PPR => self.processor_priority() as u32,
            REG_RRD => 0,
            REG_LDR => {
                if self.x2apic() {
                    self.ldr
                } else {
                    self.ldr & 0xff00_0000
                }
            }
            REG_DFR => self.dfr,
            REG_SVR => self.svr,
            REG_ISR..=0x170 if offset & 0xf == 0 => self.isr[((offset - REG_ISR) >> 4) as usize],
            REG_TMR..=0x1f0 if offset & 0xf == 0 => self.tmr[((offset - REG_TMR) >> 4) as usize],
            REG_IRR..=0x270 if offset & 0xf == 0 => self.irr[((offset - REG_IRR) >> 4) as usize],
            REG_ESR => self.esr,
            REG_ICR_LOW => self.icr_low_readback(),
            REG_ICR_HIGH => self.icr_high,
            REG_TIMER_INITIAL => self.timer_initial,
            REG_TIMER_CURRENT => self.timer_current_count(now),
            REG_TIMER_DIVIDE => self.divide_config,
            REG_EXTENDED_FEATURE..=REG_EXTENDED_LAST if !self.x2apic() => 0,
            _ => return None,
        })
    }

    fn icr_low_readback(&self) -> u32 {
        0
    }

    /// Register write. Returns `Err(())` for a register that does not exist or is read-only
    /// (callers raise #GP in x2APIC mode). `Ok` carries what to tell the CPU and, for an EOI of a
    /// level-triggered interrupt, the vector the I/O APIC must hear about.
    pub fn write_register(&mut self, offset: u32, value: u32, now: u64) -> Result<WriteEffect, ()> {
        let mut effect = WriteEffect::default();
        match offset {
            REG_ID => {
                if !self.x2apic() {
                    self.id = value >> 24;
                } else {
                    return Err(());
                }
            }
            REG_TPR => self.tpr = value as u8,
            REG_EOI => effect.eoi_vector = self.eoi(),
            REG_LDR => {
                if self.x2apic() {
                    return Err(());
                }
                self.ldr = value & 0xff00_0000;
            }
            REG_DFR => {
                if self.x2apic() {
                    return Err(());
                }
                self.dfr = if value >> 28 == 0xf {
                    DFR_FLAT
                } else {
                    DFR_CLUSTER
                };
            }
            REG_SVR => {
                self.svr = value & SVR_WRITABLE;
                if self.svr & SVR_SOFTWARE_ENABLE == 0 {
                    // Software disable masks every LVT entry (SDM 12.4.3).
                    for lvt in [
                        &mut self.lvt_cmci,
                        &mut self.lvt_timer,
                        &mut self.lvt_thermal,
                        &mut self.lvt_performance,
                        &mut self.lvt_lint0,
                        &mut self.lvt_lint1,
                        &mut self.lvt_error,
                    ] {
                        *lvt |= LVT_MASKED;
                    }
                }
            }
            REG_ESR => {
                // Writing latches the accumulated errors and clears them (SDM 12.5.3).
                self.esr = core::mem::take(&mut self.esr_pending);
            }
            REG_LVT_CMCI => self.lvt_cmci = value & 0x0001_07ff,
            REG_LVT_THERMAL => self.lvt_thermal = value & 0x0001_07ff,
            REG_LVT_PERFORMANCE => self.lvt_performance = value & 0x0001_07ff,
            REG_LVT_LINT0 => self.lvt_lint0 = value & 0x0001_a7ff,
            REG_LVT_LINT1 => self.lvt_lint1 = value & 0x0001_a7ff,
            REG_LVT_ERROR => self.lvt_error = value & 0x0001_00ff,
            REG_LVT_TIMER => {
                let previous_mode = self.timer_mode();
                self.lvt_timer = value & (0x0001_00ff | LVT_TIMER_MODE_MASK);
                if self.timer_mode() == 3 {
                    // Reserved mode: treated as one-shot.
                    self.lvt_timer &= !LVT_TIMER_MODE_MASK;
                }
                if self.timer_mode() != previous_mode {
                    // Changing the mode disarms the timer (SDM 12.5.4.1).
                    self.timer_deadline_ns = None;
                    if self.timer_mode() == TIMER_MODE_TSC_DEADLINE {
                        self.timer_initial = 0;
                    }
                }
            }
            REG_TIMER_INITIAL => {
                self.timer_initial = value;
                if self.timer_mode() != TIMER_MODE_TSC_DEADLINE {
                    self.arm_timer(now);
                }
            }
            REG_TIMER_DIVIDE => self.divide_config = value & DIVIDE_WRITABLE,
            REG_ICR_HIGH => self.icr_high = value,
            REG_ICR_LOW => effect.ipi = self.write_icr(value, self.icr_destination_xapic()),
            REG_SELF_IPI if self.x2apic() => {
                let interrupt = Interrupt {
                    vector: value as u8,
                    delivery_mode: DELIVERY_FIXED,
                    level_triggered: false,
                    destination: self.id,
                    logical: false,
                };
                self.accept(&interrupt);
            }
            REG_EXTENDED_FEATURE..=REG_EXTENDED_LAST if !self.x2apic() => {}
            _ => return Err(()),
        }
        Ok(effect)
    }

    fn icr_destination_xapic(&self) -> u32 {
        self.icr_high >> 24
    }

    /// Describes the IPI an ICR write sends; `Devices::send_ipi` delivers it. `None` when nothing
    /// is sent: an illegal vector (recorded in the ESR) or an INIT de-assert.
    fn write_icr(&mut self, low: u32, destination: u32) -> Option<Ipi> {
        let low = low & ICR_WRITABLE_XAPIC;
        let vector = (low & ICR_VECTOR_MASK) as u8;
        let delivery_mode = (low >> ICR_DELIVERY_SHIFT) & 0b111;
        let shorthand = (low >> ICR_SHORTHAND_SHIFT) & ICR_SHORTHAND_MASK;
        if matches!(delivery_mode, DELIVERY_FIXED | DELIVERY_LOWEST_PRIORITY)
            && vector < FIRST_VALID_VECTOR
        {
            self.esr_pending |= ESR_SEND_ILLEGAL_VECTOR;
            return None;
        }
        let level_triggered = low & ICR_LEVEL_TRIGGER != 0;
        if delivery_mode == DELIVERY_INIT && level_triggered && low & ICR_LEVEL_ASSERT == 0 {
            return None;
        }
        Some(Ipi {
            interrupt: Interrupt {
                vector,
                delivery_mode,
                level_triggered,
                destination,
                logical: low & ICR_LOGICAL_DESTINATION != 0,
            },
            shorthand,
        })
    }

    /// x2APIC writes carry the 32-bit destination in the upper half of the 64-bit ICR value.
    pub fn write_icr_x2apic(&mut self, value: u64) -> Option<Ipi> {
        self.icr_high = (value >> 32) as u32;
        self.write_icr(value as u32, (value >> 32) as u32)
    }

    pub fn read_icr_x2apic(&self) -> u64 {
        (self.icr_high as u64) << 32
    }

    pub fn record_illegal_register(&mut self) {
        self.esr_pending |= ESR_ILLEGAL_REGISTER;
    }
}

/// Side effects of a register write that the device bus must act on.
#[derive(Default)]
pub(crate) struct WriteEffect {
    /// A level-triggered interrupt finished: tell the I/O APIC.
    pub eoi_vector: Option<u8>,
    /// The guest wrote the ICR: send this.
    pub ipi: Option<Ipi>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> Lapic {
        let mut apic = Lapic::new(0);
        apic.write_register(REG_SVR, SVR_SOFTWARE_ENABLE | 0xff, 0)
            .unwrap();
        apic
    }

    fn fixed(vector: u8) -> Interrupt {
        Interrupt {
            vector,
            delivery_mode: DELIVERY_FIXED,
            level_triggered: false,
            destination: 0,
            logical: false,
        }
    }

    #[test]
    fn reset_state_matches_the_sdm() {
        let mut apic = Lapic::new(0);
        assert!(apic.hardware_enabled());
        assert!(!apic.software_enabled());
        assert_eq!(apic.read_register(REG_SVR, 0), Some(0xff));
        assert_eq!(apic.read_register(REG_VERSION, 0), Some(VERSION_VALUE));
        assert_eq!(apic.read_register(REG_LVT_TIMER, 0), Some(LVT_MASKED));
        assert_eq!(apic.read_register(REG_DFR, 0), Some(0xffff_ffff));
    }

    #[test]
    fn pending_interrupt_waits_for_software_enable() {
        let mut apic = Lapic::new(0);
        apic.accept(&fixed(0x41));
        assert_eq!(apic.pending_vector(), None);
    }

    #[test]
    fn highest_vector_is_delivered_first_and_ack_moves_it_to_service() {
        let mut apic = enabled();
        apic.accept(&fixed(0x31));
        apic.accept(&fixed(0x91));
        assert_eq!(apic.ack(), Some(0x91));
        // The 0x31 request is below the in-service priority class.
        assert_eq!(apic.pending_vector(), None);
        assert_eq!(apic.eoi(), None);
        assert_eq!(apic.ack(), Some(0x31));
    }

    #[test]
    fn task_priority_blocks_equal_and_lower_classes() {
        let mut apic = enabled();
        apic.set_tpr(0x40);
        apic.accept(&fixed(0x4f));
        assert_eq!(apic.pending_vector(), None);
        apic.accept(&fixed(0x50));
        assert_eq!(apic.pending_vector(), Some(0x50));
    }

    #[test]
    fn eoi_of_a_level_interrupt_reports_the_vector() {
        let mut apic = enabled();
        let mut interrupt = fixed(0x60);
        interrupt.level_triggered = true;
        apic.accept(&interrupt);
        apic.ack();
        assert_eq!(apic.eoi(), Some(0x60));
    }

    #[test]
    fn illegal_vectors_set_the_receive_error() {
        let mut apic = enabled();
        apic.accept(&fixed(0x0f));
        assert_eq!(apic.pending_vector(), None);
        apic.write_register(REG_ESR, 0, 0).unwrap();
        assert_eq!(
            apic.read_register(REG_ESR, 0),
            Some(ESR_RECEIVE_ILLEGAL_VECTOR)
        );
    }

    #[test]
    fn one_shot_timer_fires_once_after_count_times_divider() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40, 0).unwrap();
        apic.write_register(REG_TIMER_DIVIDE, 0b1011, 0).unwrap(); // divide by 1
        apic.write_register(REG_TIMER_INITIAL, 1000, 0).unwrap();
        assert_eq!(apic.next_deadline(), Some(1000 * BUS_CLOCK_PERIOD_NS));
        apic.tick(9_999);
        assert_eq!(apic.pending_vector(), None);
        apic.tick(10_000);
        assert_eq!(apic.pending_vector(), Some(0x40));
        assert_eq!(apic.next_deadline(), None);
        assert_eq!(apic.read_register(REG_TIMER_CURRENT, 20_000), Some(0));
    }

    #[test]
    fn periodic_timer_reloads() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40 | (1 << LVT_TIMER_MODE_SHIFT), 0)
            .unwrap();
        apic.write_register(REG_TIMER_DIVIDE, 0b1011, 0).unwrap();
        apic.write_register(REG_TIMER_INITIAL, 1000, 0).unwrap();
        apic.tick(10_000);
        assert_eq!(apic.ack(), Some(0x40));
        apic.eoi();
        assert_eq!(apic.next_deadline(), Some(20_000));
        assert_eq!(apic.read_register(REG_TIMER_CURRENT, 15_000), Some(500));
    }

    #[test]
    fn divide_configuration_encoding() {
        let mut apic = Lapic::new(0);
        for (config, divider) in [
            (0b0000, 2),
            (0b0001, 4),
            (0b0010, 8),
            (0b0011, 16),
            (0b1000, 32),
            (0b1001, 64),
            (0b1010, 128),
            (0b1011, 1),
        ] {
            apic.write_register(REG_TIMER_DIVIDE, config, 0).unwrap();
            assert_eq!(apic.divider(), divider, "config {config:#b}");
        }
    }

    #[test]
    fn masked_timer_does_not_deliver() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40 | LVT_MASKED, 0)
            .unwrap();
        apic.write_register(REG_TIMER_DIVIDE, 0b1011, 0).unwrap();
        apic.write_register(REG_TIMER_INITIAL, 10, 0).unwrap();
        apic.tick(1_000_000);
        assert_eq!(apic.pending_vector(), None);
    }

    #[test]
    fn tsc_deadline_fires_at_the_programmed_time_and_disarms() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40 | (2 << LVT_TIMER_MODE_SHIFT), 0)
            .unwrap();
        apic.set_tsc_deadline(5_000, 1_000);
        assert_eq!(apic.next_deadline(), Some(5_000));
        apic.tick(4_999);
        assert_eq!(apic.pending_vector(), None);
        apic.tick(5_000);
        assert_eq!(apic.pending_vector(), Some(0x40));
        assert_eq!(apic.tsc_deadline(), 0);
        assert_eq!(apic.next_deadline(), None);
    }

    #[test]
    fn tsc_deadline_zero_disarms() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40 | (2 << LVT_TIMER_MODE_SHIFT), 0)
            .unwrap();
        apic.set_tsc_deadline(5_000, 0);
        apic.set_tsc_deadline(0, 0);
        assert_eq!(apic.next_deadline(), None);
    }

    #[test]
    fn changing_the_timer_mode_disarms_it() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40, 0).unwrap();
        apic.write_register(REG_TIMER_INITIAL, 1000, 0).unwrap();
        apic.write_register(REG_LVT_TIMER, 0x40 | (1 << LVT_TIMER_MODE_SHIFT), 0)
            .unwrap();
        assert_eq!(apic.next_deadline(), None);
    }

    #[test]
    fn an_icr_write_describes_the_ipi_and_delivers_nothing_itself() {
        let mut apic = enabled();
        apic.write_register(REG_ICR_HIGH, 3 << 24, 0).unwrap();
        let effect = apic.write_register(REG_ICR_LOW, 0x55, 0).unwrap();
        let ipi = effect.ipi.expect("an IPI");
        assert_eq!(ipi.shorthand, SHORTHAND_NONE);
        assert_eq!(ipi.interrupt.vector, 0x55);
        assert_eq!(ipi.interrupt.destination, 3);
        assert_eq!(apic.pending_vector(), None);
    }

    #[test]
    fn the_icr_shorthand_is_reported() {
        let mut apic = enabled();
        let low = 0x55 | (SHORTHAND_ALL_EXCLUDING_SELF << ICR_SHORTHAND_SHIFT);
        let effect = apic.write_register(REG_ICR_LOW, low, 0).unwrap();
        assert_eq!(effect.ipi.unwrap().shorthand, SHORTHAND_ALL_EXCLUDING_SELF);
    }

    #[test]
    fn an_x2apic_icr_write_carries_a_32_bit_destination() {
        let mut apic = enabled();
        let ipi = apic.write_icr_x2apic((0x1_0002u64 << 32) | 0x55).unwrap();
        assert_eq!(ipi.interrupt.destination, 0x1_0002);
    }

    #[test]
    fn an_init_deassert_sends_nothing_but_an_init_assert_does() {
        let mut apic = enabled();
        let init = DELIVERY_INIT << ICR_DELIVERY_SHIFT;
        let deassert = init | ICR_LEVEL_TRIGGER;
        assert!(
            apic.write_register(REG_ICR_LOW, deassert, 0)
                .unwrap()
                .ipi
                .is_none()
        );
        let assert_init = deassert | ICR_LEVEL_ASSERT;
        assert!(
            apic.write_register(REG_ICR_LOW, assert_init, 0)
                .unwrap()
                .ipi
                .is_some()
        );
        assert!(
            apic.write_register(REG_ICR_LOW, init, 0)
                .unwrap()
                .ipi
                .is_some()
        );
    }

    #[test]
    fn an_illegal_ipi_vector_is_an_error_and_sends_nothing() {
        let mut apic = enabled();
        let effect = apic.write_register(REG_ICR_LOW, 0x05, 0).unwrap();
        assert!(effect.ipi.is_none());
        apic.write_register(REG_ESR, 0, 0).unwrap();
        assert_eq!(
            apic.read_register(REG_ESR, 0),
            Some(ESR_SEND_ILLEGAL_VECTOR)
        );
    }

    #[test]
    fn init_and_startup_wait_for_the_processor_even_with_the_apic_software_disabled() {
        let mut apic = Lapic::new(1);
        let init = Interrupt {
            delivery_mode: DELIVERY_INIT,
            destination: 1,
            ..fixed(0)
        };
        let startup = Interrupt {
            delivery_mode: DELIVERY_STARTUP,
            destination: 1,
            ..fixed(0x08)
        };
        assert!(!apic.start_signal_pending());
        apic.accept(&init);
        apic.accept(&startup);
        assert!(apic.take_init());
        assert!(!apic.take_init());
        assert_eq!(apic.take_startup(), Some(0x08));
        assert!(!apic.start_signal_pending());
    }

    #[test]
    fn only_processor_zero_is_the_bsp_and_a_reset_keeps_the_id() {
        assert_ne!(Lapic::new(0).base_msr() & APIC_BASE_BSP, 0);
        let mut other = Lapic::new(2);
        assert_eq!(other.base_msr() & APIC_BASE_BSP, 0);
        assert_eq!(other.read_register(REG_ID, 0), Some(2 << 24));
        other.write_register(REG_TPR, 0x40, 0).unwrap();
        other.init_reset();
        assert_eq!(other.tpr(), 0);
        assert_eq!(other.id(), 2);
    }

    #[test]
    fn the_arbitration_priority_counts_pending_requests() {
        let mut apic = enabled();
        assert_eq!(apic.arbitration_priority(), 0);
        apic.accept(&fixed(0x61));
        assert_eq!(apic.arbitration_priority(), 0x60);
    }

    #[test]
    fn nmi_ignores_the_software_enable_bit() {
        let mut apic = Lapic::new(0);
        let nmi = Interrupt {
            delivery_mode: DELIVERY_NMI,
            ..fixed(0)
        };
        assert_eq!(apic.accept(&nmi), Delivered::Nmi);
        assert!(apic.take_nmi());
        assert!(!apic.take_nmi());
    }

    #[test]
    fn software_disable_masks_the_lvt() {
        let mut apic = enabled();
        apic.write_register(REG_LVT_TIMER, 0x40, 0).unwrap();
        apic.write_register(REG_SVR, 0xff, 0).unwrap();
        assert_eq!(
            apic.read_register(REG_LVT_TIMER, 0).unwrap() & LVT_MASKED,
            LVT_MASKED
        );
    }

    #[test]
    fn extint_is_connected_until_lint0_is_masked() {
        let mut apic = Lapic::new(0);
        assert!(apic.extint_connected());
        apic.write_register(REG_LVT_LINT0, LVT_MASKED | (DELIVERY_EXTINT << 8), 0)
            .unwrap();
        assert!(!apic.extint_connected());
    }

    #[test]
    fn software_disable_disconnects_the_pic() {
        let mut apic = enabled();
        assert!(apic.extint_connected());
        apic.write_register(REG_SVR, 0xff, 0).unwrap();
        assert!(!apic.extint_connected());
    }

    #[test]
    fn hardware_disabled_apic_wires_the_pic_straight_in() {
        let mut apic = Lapic::new(0);
        assert!(apic.set_base_msr(0));
        assert!(apic.extint_connected());
    }

    #[test]
    fn apic_base_transitions() {
        let mut apic = Lapic::new(0);
        let enable = APIC_DEFAULT_BASE | APIC_BASE_ENABLE;
        // x2APIC without enable is invalid.
        assert!(!apic.set_base_msr(APIC_BASE_X2APIC));
        assert!(apic.set_base_msr(enable | APIC_BASE_X2APIC));
        assert!(apic.x2apic());
        // Cannot go back to xAPIC mode directly.
        assert!(!apic.set_base_msr(enable));
        assert!(apic.set_base_msr(0));
        assert!(!apic.hardware_enabled());
        assert!(apic.set_base_msr(enable));
        assert!(!apic.set_base_msr(1 << 63));
    }

    #[test]
    fn bsp_bit_is_read_only() {
        let mut apic = Lapic::new(0);
        assert!(apic.set_base_msr(APIC_DEFAULT_BASE | APIC_BASE_ENABLE));
        assert_ne!(apic.base_msr() & APIC_BASE_BSP, 0);
    }

    #[test]
    fn x2apic_id_is_not_shifted_and_logical_id_is_derived() {
        let mut apic = Lapic::new(0);
        assert!(apic.set_base_msr(APIC_DEFAULT_BASE | APIC_BASE_ENABLE | APIC_BASE_X2APIC));
        assert_eq!(apic.read_register(REG_ID, 0), Some(0));
        assert_eq!(apic.read_register(REG_LDR, 0), Some(1));
    }

    #[test]
    fn x2apic_self_ipi_register() {
        let mut apic = Lapic::new(0);
        apic.set_base_msr(APIC_DEFAULT_BASE | APIC_BASE_ENABLE | APIC_BASE_X2APIC);
        apic.write_register(REG_SVR, SVR_SOFTWARE_ENABLE | 0xff, 0)
            .unwrap();
        apic.write_register(REG_SELF_IPI, 0x77, 0).unwrap();
        assert_eq!(apic.pending_vector(), Some(0x77));
    }

    #[test]
    fn logical_destination_flat_model() {
        let mut apic = enabled();
        apic.write_register(REG_LDR, 0x01 << 24, 0).unwrap();
        let mut interrupt = fixed(0x50);
        interrupt.logical = true;
        interrupt.destination = 0x03;
        apic.accept(&interrupt);
        assert_eq!(apic.pending_vector(), Some(0x50));
        let mut other = fixed(0x60);
        other.logical = true;
        other.destination = 0x02;
        apic.accept(&other);
        assert_eq!(apic.pending_vector(), Some(0x50));
        apic.ack();
        apic.eoi();
        assert_eq!(apic.pending_vector(), None);
    }

    #[test]
    fn disabling_the_apic_drops_its_state() {
        let mut apic = enabled();
        apic.accept(&fixed(0x50));
        apic.set_base_msr(0);
        apic.set_base_msr(APIC_DEFAULT_BASE | APIC_BASE_ENABLE);
        assert_eq!(apic.pending_vector(), None);
        assert!(!apic.software_enabled());
    }
}
