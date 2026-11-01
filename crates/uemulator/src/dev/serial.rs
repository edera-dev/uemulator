use alloc::collections::VecDeque;

use crate::console::ConsoleHandle;

/// Register offsets from the UART base port (COM1 0x3f8). With DLAB set in LCR, offsets 0 and 1
/// address the divisor latch instead of RBR/THR and IER.
const REG_DATA: u16 = 0; // RBR on read, THR on write, DLL with DLAB
const REG_IER: u16 = 1; // DLM with DLAB
const REG_IIR_FCR: u16 = 2; // IIR on read, FCR on write
const REG_LCR: u16 = 3;
const REG_MCR: u16 = 4;
const REG_LSR: u16 = 5;
const REG_MSR: u16 = 6;
const REG_SCRATCH: u16 = 7;

const LCR_DLAB: u8 = 0x80;

/// Only the four defined IER bits are kept; the rest read back as 0.
const IER_MASK: u8 = 0x0f;
const IER_RX_AVAILABLE: u8 = 0x01;
const IER_THR_EMPTY: u8 = 0x02;
const IER_RX_LINE_STATUS: u8 = 0x04;
const IER_MODEM_STATUS: u8 = 0x08;

/// FCR bits that are stored: FIFO enable (0), DMA mode (3) and receive trigger level (7:6).
/// The two self-clearing FIFO reset bits are not kept. The trigger level is stored but does not
/// delay the receive interrupt: there is no character-timeout timer, so holding data back until
/// the trigger level is reached could strand it.
const FCR_STORED_MASK: u8 = 0xc9;
const FCR_FIFO_ENABLE: u8 = 0x01;
const FCR_CLEAR_RX: u8 = 0x02;
const FCR_CLEAR_TX: u8 = 0x04;

/// IIR bit 0 set means no interrupt pending; bits 3:1 give the cause.
const IIR_NO_INTERRUPT: u8 = 0x01;
const IIR_MODEM_STATUS: u8 = 0x00;
const IIR_THR_EMPTY: u8 = 0x02;
const IIR_RX_AVAILABLE: u8 = 0x04;
const IIR_RX_LINE_STATUS: u8 = 0x06;
/// IIR bits 7:6 read 11 when the FIFOs are enabled; this is how the kernel tells a 16550A from
/// an 8250.
const IIR_FIFO_ENABLED: u8 = 0xc0;

/// MCR bits 4:0 are implemented.
const MCR_MASK: u8 = 0x1f;
const MCR_DTR: u8 = 0x01;
const MCR_RTS: u8 = 0x02;
const MCR_OUT1: u8 = 0x04;
const MCR_OUT2: u8 = 0x08;
const MCR_LOOPBACK: u8 = 0x10;

/// Transmitter is always idle: output is instantaneous. Parity, framing and break errors need a
/// line that can corrupt data, which the console never does, so those bits stay clear.
const LSR_DATA_READY: u8 = 0x01;
const LSR_OVERRUN: u8 = 0x02;
const LSR_THR_EMPTY: u8 = 0x20;
const LSR_TRANSMITTER_EMPTY: u8 = 0x40;

const MSR_DELTA_CTS: u8 = 0x01;
const MSR_DELTA_DSR: u8 = 0x02;
const MSR_TRAILING_EDGE_RI: u8 = 0x04;
const MSR_DELTA_DCD: u8 = 0x08;
const MSR_CTS: u8 = 0x10;
const MSR_DSR: u8 = 0x20;
const MSR_RI: u8 = 0x40;
const MSR_DCD: u8 = 0x80;
/// Where a modem input's delta bit sits relative to the input itself, for CTS, DSR and DCD.
const MSR_DELTA_SHIFT: u8 = 4;
const MSR_DELTA_FROM_INPUT_MASK: u8 = MSR_DELTA_CTS | MSR_DELTA_DSR | MSR_DELTA_DCD;

/// Depth of the 16550A receive FIFO; with the FIFOs disabled the chip behaves as a 16450 whose
/// receive buffer holds one byte.
const FIFO_DEPTH: usize = 16;
const NON_FIFO_DEPTH: usize = 1;

pub(crate) struct Uart {
    console: ConsoleHandle,
    rx: VecDeque<u8>,
    /// Divisor latch bytes; stored for read-back only, the baud rate has no effect.
    dll: u8,
    dlm: u8,
    ier: u8,
    fcr: u8,
    lcr: u8,
    mcr: u8,
    scr: u8,
    /// THRE interrupt cause is armed: set by a THR write or by enabling the interrupt in IER,
    /// cleared when IIR reports it.
    thre_pending: bool,
    /// A byte arrived with the receive buffer full; cleared by reading LSR.
    overrun: bool,
    /// Modem status input bits (MSR 7:4) as of the last MCR write, and the delta bits (3:0)
    /// accumulated since MSR was last read.
    modem_inputs: u8,
    modem_deltas: u8,
    /// Interrupt level at the last `irq_edge` call, for edge detection.
    last_level: bool,
    /// Cleared while the virtio console is active so host input goes to hvc0, not the 16550.
    rx_enabled: bool,
    /// Virtual time between two bytes of host input (0: all at once). A real line carries a byte
    /// per 87 us at 115200 baud; Xen's 128-byte console input ring overflowed, dropping the end
    /// of a pasted command line (and its Enter), when the whole line landed in the FIFO faster
    /// than dom0 could drain it.
    byte_interval_ns: u64,
    now_ns: u64,
    next_byte_ns: u64,
}

impl Uart {
    pub fn new(console: ConsoleHandle) -> Self {
        Uart {
            console,
            rx: VecDeque::new(),
            dll: 0,
            dlm: 0,
            ier: 0,
            fcr: 0,
            lcr: 0,
            mcr: 0,
            scr: 0,
            thre_pending: false,
            overrun: false,
            modem_inputs: MSR_CTS | MSR_DSR | MSR_DCD,
            modem_deltas: 0,
            last_level: false,
            rx_enabled: true,
            byte_interval_ns: 0,
            now_ns: 0,
            next_byte_ns: 0,
        }
    }

    fn loopback(&self) -> bool {
        self.mcr & MCR_LOOPBACK != 0
    }

    fn fifo_depth(&self) -> usize {
        if self.fcr & FCR_FIFO_ENABLE != 0 {
            FIFO_DEPTH
        } else {
            NON_FIFO_DEPTH
        }
    }

    /// IIR cause of the highest-priority pending interrupt: line status, then received data,
    /// then THR empty, then modem status. Both the IIR read and the interrupt level come from
    /// here so they cannot disagree.
    fn pending_cause(&self) -> Option<u8> {
        if self.ier & IER_RX_LINE_STATUS != 0 && self.overrun {
            Some(IIR_RX_LINE_STATUS)
        } else if self.ier & IER_RX_AVAILABLE != 0 && !self.rx.is_empty() {
            Some(IIR_RX_AVAILABLE)
        } else if self.ier & IER_THR_EMPTY != 0 && self.thre_pending {
            Some(IIR_THR_EMPTY)
        } else if self.ier & IER_MODEM_STATUS != 0 && self.modem_deltas != 0 {
            Some(IIR_MODEM_STATUS)
        } else {
            None
        }
    }

    fn level(&self) -> bool {
        self.pending_cause().is_some()
    }

    /// Whether the interrupt line is currently asserted (as opposed to `irq_edge`, which fires once
    /// per rising edge).
    pub fn irq_level(&self) -> bool {
        self.level()
    }

    /// True on a rising edge of the interrupt line. The PIC is edge triggered, so IRQ 4 must be
    /// raised once per rising edge and not while the level stays high.
    pub fn irq_edge(&mut self) -> bool {
        let level = self.level();
        let edge = level && !self.last_level;
        self.last_level = level;
        edge
    }

    /// Paces host input at one byte per `interval_ns` of virtual time.
    pub fn set_byte_interval(&mut self, interval_ns: u64) {
        self.byte_interval_ns = interval_ns;
    }

    pub fn set_time(&mut self, now_ns: u64) {
        self.now_ns = now_ns;
    }

    /// When the next waiting host byte may arrive, so an idle guest wakes for it.
    pub fn next_deadline(&self) -> Option<u64> {
        (self.byte_interval_ns > 0
            && self.rx_enabled
            && self.next_byte_ns > self.now_ns
            && self.console.has_input())
        .then_some(self.next_byte_ns)
    }

    /// Whether host input is in the receive buffer or still held back by the pacing.
    #[cfg(not(freestanding))]
    pub fn input_pending(&self) -> bool {
        !self.rx.is_empty() || (self.rx_enabled && self.console.has_input())
    }

    pub fn set_rx_enabled(&mut self, on: bool) {
        self.rx_enabled = on;
    }

    /// A byte reaches the receiver. A full buffer drops the byte (FIFO) or replaces the held one
    /// (no FIFO) and flags an overrun.
    fn receive(&mut self, byte: u8) {
        if self.rx.len() >= self.fifo_depth() {
            self.overrun = true;
            if self.fifo_depth() > NON_FIFO_DEPTH {
                return;
            }
            self.rx.clear();
        }
        self.rx.push_back(byte);
    }

    /// Moves host input into the receive buffer. Returns true if any byte arrived. In loopback
    /// the receiver is disconnected from the line, so host input stays queued in the console.
    pub fn poll(&mut self) -> bool {
        let mut any = false;
        if !self.rx_enabled || self.loopback() {
            return false;
        }
        while self.rx.len() < self.fifo_depth() {
            if self.byte_interval_ns > 0 && self.now_ns < self.next_byte_ns {
                break;
            }
            match self.console.guest_rx() {
                Some(byte) => {
                    self.rx.push_back(byte);
                    self.next_byte_ns = self.now_ns + self.byte_interval_ns;
                    any = true;
                }
                None => break,
            }
        }
        any
    }

    fn lsr(&self) -> u8 {
        let mut lsr = LSR_THR_EMPTY | LSR_TRANSMITTER_EMPTY;
        if !self.rx.is_empty() {
            lsr |= LSR_DATA_READY;
        }
        if self.overrun {
            lsr |= LSR_OVERRUN;
        }
        lsr
    }

    /// What the modem status inputs read right now (MSR 7:4).
    fn current_modem_inputs(&self) -> u8 {
        if self.loopback() {
            // Loopback wires the outputs back to the inputs: RTS->CTS, DTR->DSR, OUT1->RI,
            // OUT2->DCD. The 8250 autoconfig probe checks this.
            let line = |mcr_bit: u8, msr_bit: u8| {
                if self.mcr & mcr_bit != 0 { msr_bit } else { 0 }
            };
            line(MCR_RTS, MSR_CTS)
                | line(MCR_DTR, MSR_DSR)
                | line(MCR_OUT1, MSR_RI)
                | line(MCR_OUT2, MSR_DCD)
        } else {
            // Outside loopback the modem inputs always read as asserted.
            MSR_CTS | MSR_DSR | MSR_DCD
        }
    }

    /// Records the delta bits for any input that changed; call after anything that can move the
    /// inputs (MCR writes). RI only reports its trailing edge.
    fn update_modem_inputs(&mut self) {
        let previous = self.modem_inputs;
        let current = self.current_modem_inputs();
        let changed = previous ^ current;
        self.modem_deltas |= (changed >> MSR_DELTA_SHIFT) & MSR_DELTA_FROM_INPUT_MASK;
        if previous & MSR_RI != 0 && current & MSR_RI == 0 {
            self.modem_deltas |= MSR_TRAILING_EDGE_RI;
        }
        self.modem_inputs = current;
    }

    pub fn read(&mut self, reg: u16) -> u8 {
        let value = self.read_register(reg);
        self.remember_falling_edge();
        value
    }

    pub fn write(&mut self, reg: u16, value: u8) {
        self.write_register(reg, value);
        self.remember_falling_edge();
    }

    /// The line dropped because of a register access: forget that it was high, so the next time
    /// it rises `irq_edge` reports a new edge even if no sample saw it low in between.
    fn remember_falling_edge(&mut self) {
        if !self.level() {
            self.last_level = false;
        }
    }

    fn read_register(&mut self, reg: u16) -> u8 {
        let dlab = self.lcr & LCR_DLAB != 0;
        match reg {
            REG_DATA if dlab => self.dll,
            REG_DATA => {
                let byte = self.rx.pop_front().unwrap_or(0);
                self.poll();
                byte
            }
            REG_IER if dlab => self.dlm,
            REG_IER => self.ier,
            REG_IIR_FCR => {
                let fifo = if self.fcr & FCR_FIFO_ENABLE != 0 {
                    IIR_FIFO_ENABLED
                } else {
                    0
                };
                let cause = self.pending_cause();
                if cause == Some(IIR_THR_EMPTY) {
                    // Reading IIR acknowledges the THRE cause.
                    self.thre_pending = false;
                }
                cause.unwrap_or(IIR_NO_INTERRUPT) | fifo
            }
            REG_LCR => self.lcr,
            REG_MCR => self.mcr,
            REG_LSR => {
                let lsr = self.lsr();
                self.overrun = false;
                lsr
            }
            REG_MSR => {
                let msr = self.modem_inputs | self.modem_deltas;
                self.modem_deltas = 0;
                msr
            }
            _ => self.scr,
        }
    }

    fn write_register(&mut self, reg: u16, value: u8) {
        let dlab = self.lcr & LCR_DLAB != 0;
        match reg {
            REG_DATA if dlab => self.dll = value,
            REG_DATA => {
                self.thre_pending = true;
                if self.loopback() {
                    self.receive(value);
                } else {
                    self.console.guest_tx(value);
                }
            }
            REG_IER if dlab => self.dlm = value,
            REG_IER => {
                let previous = self.ier;
                self.ier = value & IER_MASK;
                // Enabling the THRE interrupt while the transmitter is idle fires it at once.
                if self.ier & IER_THR_EMPTY != 0 && previous & IER_THR_EMPTY == 0 {
                    self.thre_pending = true;
                }
            }
            REG_IIR_FCR => {
                let stored = value & FCR_STORED_MASK;
                // Toggling FIFO enable flushes both FIFOs.
                let flush_both = (self.fcr ^ stored) & FCR_FIFO_ENABLE != 0;
                if flush_both || value & FCR_CLEAR_RX != 0 {
                    self.rx.clear();
                }
                // An emptied transmit FIFO is THR empty again.
                if flush_both || value & FCR_CLEAR_TX != 0 {
                    self.thre_pending = true;
                }
                self.fcr = stored;
            }
            REG_LCR => self.lcr = value,
            REG_MCR => {
                self.mcr = value & MCR_MASK;
                self.update_modem_inputs();
            }
            REG_SCRATCH => self.scr = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::pic::Pics;

    const COM1_IRQ: u8 = 4;
    const LCR_8N1: u8 = 0x03;
    const LSR_IDLE: u8 = LSR_THR_EMPTY | LSR_TRANSMITTER_EMPTY;
    const MSR_ALL_INPUTS: u8 = MSR_CTS | MSR_DSR | MSR_RI | MSR_DCD;
    const MSR_NOT_LOOPBACK: u8 = MSR_CTS | MSR_DSR | MSR_DCD;
    const IER_ALL: u8 = IER_RX_AVAILABLE | IER_THR_EMPTY | IER_RX_LINE_STATUS | IER_MODEM_STATUS;

    fn new_uart() -> (Uart, ConsoleHandle) {
        let console = ConsoleHandle::default();
        (Uart::new(console.clone()), console)
    }

    fn drain_tx(console: &ConsoleHandle) -> Vec<u8> {
        let mut out = Vec::new();
        let mut buf = [0u8; 64];
        loop {
            let count = console.read(&mut buf, None);
            if count == 0 {
                return out;
            }
            out.extend_from_slice(&buf[..count]);
        }
    }

    fn enable_fifo(uart: &mut Uart) {
        uart.write(REG_IIR_FCR, FCR_FIFO_ENABLE);
    }

    fn loopback(uart: &mut Uart) {
        uart.write(REG_MCR, MCR_LOOPBACK);
    }

    /// The IIR cause bits, ignoring the FIFO-enabled flags.
    fn iir_cause(uart: &mut Uart) -> u8 {
        uart.read(REG_IIR_FCR) & !IIR_FIFO_ENABLED
    }

    // Reset state and plain registers

    #[test]
    fn reset_state_reads_idle() {
        let (mut uart, _) = new_uart();
        assert_eq!(uart.read(REG_IER), 0);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
        assert_eq!(uart.read(REG_LCR), 0);
        assert_eq!(uart.read(REG_MCR), 0);
        assert_eq!(uart.read(REG_LSR), LSR_IDLE);
        assert_eq!(uart.read(REG_MSR), MSR_NOT_LOOPBACK);
        assert_eq!(uart.read(REG_SCRATCH), 0);
        assert_eq!(uart.read(REG_DATA), 0);
    }

    #[test]
    fn scratch_register_round_trips_every_value() {
        let (mut uart, _) = new_uart();
        for value in 0..=u8::MAX {
            uart.write(REG_SCRATCH, value);
            assert_eq!(uart.read(REG_SCRATCH), value);
        }
    }

    #[test]
    fn scratch_register_is_independent_of_the_divisor_latch() {
        let (mut uart, _) = new_uart();
        uart.write(REG_SCRATCH, 0xa5);
        uart.write(REG_LCR, LCR_DLAB);
        uart.write(REG_DATA, 0x12);
        assert_eq!(uart.read(REG_SCRATCH), 0xa5);
    }

    #[test]
    fn lcr_stores_all_eight_bits() {
        let (mut uart, _) = new_uart();
        for value in 0..=u8::MAX {
            uart.write(REG_LCR, value);
            assert_eq!(uart.read(REG_LCR), value);
        }
    }

    #[test]
    fn mcr_keeps_only_the_five_defined_bits() {
        let (mut uart, _) = new_uart();
        for value in 0..=u8::MAX {
            uart.write(REG_MCR, value);
            assert_eq!(uart.read(REG_MCR), value & MCR_MASK);
        }
    }

    #[test]
    fn ier_keeps_only_the_four_defined_bits() {
        let (mut uart, _) = new_uart();
        for value in 0..=u8::MAX {
            uart.write(REG_IER, value);
            assert_eq!(uart.read(REG_IER), value & IER_MASK);
        }
    }

    #[test]
    fn lsr_and_msr_are_read_only() {
        let (mut uart, _) = new_uart();
        uart.write(REG_LSR, 0xff);
        uart.write(REG_MSR, 0xff);
        assert_eq!(uart.read(REG_LSR), LSR_IDLE);
        assert_eq!(uart.read(REG_MSR), MSR_NOT_LOOPBACK);
    }

    // Divisor latch

    #[test]
    fn divisor_latch_is_reached_through_dlab() {
        let (mut uart, _) = new_uart();
        uart.write(REG_LCR, LCR_DLAB | LCR_8N1);
        uart.write(REG_DATA, 0x0c);
        uart.write(REG_IER, 0x01);
        assert_eq!(uart.read(REG_DATA), 0x0c);
        assert_eq!(uart.read(REG_IER), 0x01);
        uart.write(REG_LCR, LCR_8N1);
        // DLM did not leak into IER and DLL did not become data.
        assert_eq!(uart.read(REG_IER), 0);
        uart.write(REG_LCR, LCR_DLAB | LCR_8N1);
        assert_eq!(uart.read(REG_DATA), 0x0c);
        assert_eq!(uart.read(REG_IER), 0x01);
    }

    #[test]
    fn divisor_latch_holds_all_sixteen_bits() {
        let (mut uart, _) = new_uart();
        uart.write(REG_LCR, LCR_DLAB);
        for (low, high) in [(0x00, 0x00), (0xff, 0xff), (0x34, 0x12), (0x01, 0x80)] {
            uart.write(REG_DATA, low);
            uart.write(REG_IER, high);
            assert_eq!(uart.read(REG_DATA), low);
            assert_eq!(uart.read(REG_IER), high);
        }
    }

    #[test]
    fn writing_dll_with_dlab_set_does_not_transmit() {
        let (mut uart, console) = new_uart();
        uart.write(REG_LCR, LCR_DLAB);
        uart.write(REG_DATA, b'x');
        assert!(drain_tx(&console).is_empty());
        uart.write(REG_LCR, 0);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }

    #[test]
    fn dlab_does_not_hide_the_other_registers() {
        let (mut uart, _) = new_uart();
        uart.write(REG_LCR, LCR_DLAB);
        uart.write(REG_MCR, MCR_DTR);
        uart.write(REG_SCRATCH, 0x77);
        assert_eq!(uart.read(REG_MCR), MCR_DTR);
        assert_eq!(uart.read(REG_SCRATCH), 0x77);
        assert_eq!(uart.read(REG_LSR), LSR_IDLE);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn receive_data_is_not_readable_while_dlab_is_set() {
        let (mut uart, console) = new_uart();
        console.write(b"q");
        uart.poll();
        uart.write(REG_LCR, LCR_DLAB);
        assert_eq!(uart.read(REG_DATA), 0);
        uart.write(REG_LCR, 0);
        assert_eq!(uart.read(REG_DATA), b'q');
    }

    // Transmit

    #[test]
    fn transmitted_bytes_reach_the_console_in_order() {
        let (mut uart, console) = new_uart();
        for byte in b"hello\r\n" {
            uart.write(REG_DATA, *byte);
        }
        assert_eq!(drain_tx(&console), b"hello\r\n");
    }

    #[test]
    fn every_byte_value_can_be_transmitted() {
        let (mut uart, console) = new_uart();
        let all: Vec<u8> = (0..=u8::MAX).collect();
        for byte in &all {
            uart.write(REG_DATA, *byte);
        }
        assert_eq!(drain_tx(&console), all);
    }

    #[test]
    fn transmitter_is_always_empty_and_ready() {
        let (mut uart, _) = new_uart();
        uart.write(REG_DATA, b'a');
        assert_eq!(uart.read(REG_LSR) & LSR_IDLE, LSR_IDLE);
    }

    // Receive and the FIFO

    #[test]
    fn host_input_is_received_in_order() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        console.write(b"abc");
        assert!(uart.poll());
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, LSR_DATA_READY);
        assert_eq!(uart.read(REG_DATA), b'a');
        assert_eq!(uart.read(REG_DATA), b'b');
        assert_eq!(uart.read(REG_DATA), b'c');
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        assert!(!uart.poll());
    }

    #[test]
    fn reading_an_empty_receive_buffer_returns_zero() {
        let (mut uart, _) = new_uart();
        assert_eq!(uart.read(REG_DATA), 0);
        enable_fifo(&mut uart);
        assert_eq!(uart.read(REG_DATA), 0);
    }

    #[test]
    fn fifo_holds_sixteen_bytes_and_leaves_the_rest_in_the_console() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        let input: Vec<u8> = (0..20).collect();
        console.write(&input);
        assert!(uart.poll());
        assert_eq!(uart.rx.len(), FIFO_DEPTH);
        assert!(console.has_input());
        // Draining the FIFO refills it from the console, so nothing is lost or reordered.
        let received: Vec<u8> = (0..20).map(|_| uart.read(REG_DATA)).collect();
        assert_eq!(received, input);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        assert!(!console.has_input());
    }

    #[test]
    fn without_the_fifo_only_one_byte_is_buffered() {
        let (mut uart, console) = new_uart();
        console.write(b"xyz");
        assert!(uart.poll());
        assert_eq!(uart.rx.len(), 1);
        assert_eq!(uart.read(REG_DATA), b'x');
        assert_eq!(uart.read(REG_DATA), b'y');
        assert_eq!(uart.read(REG_DATA), b'z');
        assert_eq!(uart.read(REG_DATA), 0);
    }

    #[test]
    fn poll_without_input_reports_nothing() {
        let (mut uart, _) = new_uart();
        assert!(!uart.poll());
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }

    #[test]
    fn disabled_receive_leaves_host_input_in_the_console() {
        let (mut uart, console) = new_uart();
        uart.set_rx_enabled(false);
        console.write(b"k");
        assert!(!uart.poll());
        assert!(console.has_input());
        uart.set_rx_enabled(true);
        assert!(uart.poll());
        assert_eq!(uart.read(REG_DATA), b'k');
    }

    #[test]
    fn toggling_fifo_enable_flushes_the_receive_fifo() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        console.write(b"abc");
        uart.poll();
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, LSR_DATA_READY);
        uart.write(REG_IIR_FCR, 0);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        console.write(b"d");
        uart.poll();
        uart.write(REG_IIR_FCR, FCR_FIFO_ENABLE);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }

    #[test]
    fn rewriting_the_same_fifo_enable_does_not_flush() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        console.write(b"ab");
        uart.poll();
        enable_fifo(&mut uart);
        assert_eq!(uart.read(REG_DATA), b'a');
    }

    #[test]
    fn fcr_receive_reset_bit_clears_the_fifo() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        console.write(b"abc");
        uart.poll();
        uart.write(REG_IIR_FCR, FCR_FIFO_ENABLE | FCR_CLEAR_RX);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        assert_eq!(uart.read(REG_DATA), 0);
        // The enable bit is kept; the reset bit is not stored.
        assert_eq!(uart.fcr, FCR_FIFO_ENABLE);
    }

    #[test]
    fn fcr_stores_enable_dma_and_trigger_level_only() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IIR_FCR, 0xff);
        assert_eq!(uart.fcr, FCR_STORED_MASK);
        uart.write(REG_IIR_FCR, 0xc1);
        assert_eq!(uart.fcr, 0xc1);
        uart.write(REG_IIR_FCR, 0x06);
        assert_eq!(uart.fcr, 0);
    }

    #[test]
    fn receive_trigger_level_does_not_delay_the_interrupt() {
        // There is no character-timeout timer, so a level above one byte would strand input.
        let (mut uart, console) = new_uart();
        uart.write(REG_IIR_FCR, FCR_FIFO_ENABLE | 0xc0);
        uart.write(REG_IER, IER_RX_AVAILABLE);
        console.write(b"a");
        uart.poll();
        assert_eq!(iir_cause(&mut uart), IIR_RX_AVAILABLE);
    }

    // IIR

    #[test]
    fn iir_reports_fifo_enable_in_the_top_bits() {
        let (mut uart, _) = new_uart();
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, 0);
        enable_fifo(&mut uart);
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, IIR_FIFO_ENABLED);
        uart.write(REG_IIR_FCR, 0);
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, 0);
    }

    #[test]
    fn iir_with_nothing_pending_has_the_no_interrupt_bit() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_ALL);
        // THRE is pending right after enabling; consume it.
        uart.read(REG_IIR_FCR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
        enable_fifo(&mut uart);
        // Toggling FIFO enable re-arms THRE; consume it.
        uart.read(REG_IIR_FCR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT | IIR_FIFO_ENABLED);
    }

    #[test]
    fn received_data_interrupt_needs_the_ier_bit() {
        let (mut uart, console) = new_uart();
        console.write(b"a");
        uart.poll();
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
        assert!(!uart.irq_edge());
        uart.write(REG_IER, IER_RX_AVAILABLE);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
    }

    #[test]
    fn received_data_interrupt_clears_when_the_buffer_is_read_empty() {
        let (mut uart, console) = new_uart();
        uart.write(REG_IER, IER_RX_AVAILABLE);
        console.write(b"a");
        uart.poll();
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
        // Reading IIR does not clear it, reading RBR does.
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
        uart.read(REG_DATA);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn thre_interrupt_fires_when_ier_enables_it() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        assert!(uart.irq_edge());
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
    }

    #[test]
    fn reading_iir_acknowledges_the_thre_interrupt() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
        assert!(!uart.level());
    }

    #[test]
    fn writing_thr_rearms_the_thre_interrupt() {
        let (mut uart, console) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        uart.read(REG_IIR_FCR);
        assert!(!uart.level());
        uart.write(REG_DATA, b'z');
        assert!(uart.level());
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
        assert_eq!(drain_tx(&console), b"z");
    }

    #[test]
    fn disabling_and_reenabling_thre_in_ier_rearms_it() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        uart.read(REG_IIR_FCR);
        uart.write(REG_IER, 0);
        assert!(!uart.level());
        uart.write(REG_IER, IER_THR_EMPTY);
        assert!(uart.level());
    }

    #[test]
    fn rewriting_ier_with_thre_already_enabled_does_not_rearm() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        uart.read(REG_IIR_FCR);
        uart.write(REG_IER, IER_THR_EMPTY | IER_RX_AVAILABLE);
        assert!(!uart.level());
    }

    #[test]
    fn iir_read_does_not_acknowledge_thre_while_it_is_masked() {
        let (mut uart, _) = new_uart();
        uart.write(REG_DATA, b'a');
        // THRE interrupt is armed but IER does not enable it.
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
        assert!(uart.thre_pending);
    }

    #[test]
    fn thre_bug_probe_sees_an_interrupt_after_enabling_it() {
        // Linux's 8250 startup: enable THRI, read LSR then IIR; no interrupt would flag a
        // buggy UART.
        let (mut uart, _) = new_uart();
        enable_fifo(&mut uart);
        uart.write(REG_IER, IER_THR_EMPTY);
        let lsr = uart.read(REG_LSR);
        let iir = uart.read(REG_IIR_FCR);
        uart.write(REG_IER, 0);
        assert_ne!(lsr & LSR_TRANSMITTER_EMPTY, 0);
        assert_eq!(iir & IIR_NO_INTERRUPT, 0);
    }

    #[test]
    fn fcr_transmit_reset_rearms_thre() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        uart.read(REG_IIR_FCR);
        uart.write(REG_IIR_FCR, FCR_CLEAR_TX);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
    }

    #[test]
    fn toggling_fifo_enable_rearms_thre() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY);
        uart.read(REG_IIR_FCR);
        enable_fifo(&mut uart);
        assert_eq!(iir_cause(&mut uart), IIR_THR_EMPTY);
    }

    #[test]
    fn received_data_has_priority_over_thre() {
        let (mut uart, console) = new_uart();
        uart.write(REG_IER, IER_RX_AVAILABLE | IER_THR_EMPTY);
        console.write(b"a");
        uart.poll();
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
        uart.read(REG_DATA);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn line_status_has_priority_over_received_data() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.write(REG_IER, IER_RX_AVAILABLE | IER_RX_LINE_STATUS);
        uart.write(REG_DATA, b'a');
        uart.write(REG_DATA, b'b');
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_LINE_STATUS);
        // Reading LSR clears the overrun; received data is next.
        assert_eq!(uart.read(REG_LSR) & LSR_OVERRUN, LSR_OVERRUN);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
    }

    #[test]
    fn modem_status_has_the_lowest_priority() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_THR_EMPTY | IER_MODEM_STATUS);
        loopback(&mut uart);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_MODEM_STATUS);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_MODEM_STATUS);
        uart.read(REG_MSR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn all_four_causes_are_reported_in_priority_order() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_ALL);
        loopback(&mut uart);
        // Two loopback bytes: data ready, overrun (no FIFO), THRE armed, modem deltas from
        // entering loopback.
        uart.write(REG_DATA, b'a');
        uart.write(REG_DATA, b'b');
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_LINE_STATUS);
        uart.read(REG_LSR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_AVAILABLE);
        uart.read(REG_DATA);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_MODEM_STATUS);
        uart.read(REG_MSR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn iir_cause_codes_match_the_16550_encoding() {
        assert_eq!(IIR_NO_INTERRUPT, 0b0001);
        assert_eq!(IIR_THR_EMPTY, 0b0010);
        assert_eq!(IIR_RX_AVAILABLE, 0b0100);
        assert_eq!(IIR_RX_LINE_STATUS, 0b0110);
        assert_eq!(IIR_MODEM_STATUS, 0b0000);
    }

    // Interrupt line

    #[test]
    fn irq_edge_fires_once_per_rising_edge() {
        let (mut uart, console) = new_uart();
        uart.write(REG_IER, IER_RX_AVAILABLE);
        assert!(!uart.irq_edge());
        console.write(b"a");
        uart.poll();
        assert!(uart.irq_edge());
        assert!(!uart.irq_edge());
        uart.read(REG_DATA);
        assert!(!uart.irq_edge());
        console.write(b"b");
        uart.poll();
        assert!(uart.irq_edge());
    }

    #[test]
    fn paced_host_input_arrives_one_byte_per_interval() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        uart.set_byte_interval(100);
        console.write(b"abc");
        uart.set_time(1000);
        assert!(uart.poll());
        assert_eq!(uart.rx.len(), 1);
        assert_eq!(uart.next_deadline(), Some(1100));
        uart.set_time(1050);
        assert!(!uart.poll());
        assert_eq!(uart.next_deadline(), Some(1100));
        uart.set_time(1100);
        assert!(uart.poll());
        assert_eq!(uart.rx.len(), 2);
        uart.set_time(1200);
        assert!(uart.poll());
        assert_eq!(uart.next_deadline(), None, "the console is empty");
    }

    #[test]
    fn irq_edge_sees_a_dip_the_guest_caused_between_two_samples() {
        // Xen drains the receive FIFO in its handler and the next byte can arrive before the
        // device tick samples the line low: that is still a new rising edge on an edge-triggered
        // I/O APIC pin, and missing it strands the rest of a long pasted line.
        let (mut uart, console) = new_uart();
        uart.write(REG_IER, IER_RX_AVAILABLE);
        console.write(b"a");
        uart.poll();
        assert!(uart.irq_edge());
        uart.read(REG_DATA);
        console.write(b"b");
        uart.poll();
        assert!(uart.irq_edge());
    }

    #[test]
    fn irq_edge_stays_quiet_while_the_line_is_held_high() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        uart.write(REG_IER, IER_RX_AVAILABLE);
        console.write(b"abc");
        uart.poll();
        assert!(uart.irq_edge());
        uart.read(REG_DATA);
        assert!(!uart.irq_edge());
        uart.read(REG_DATA);
        assert!(!uart.irq_edge());
    }

    #[test]
    fn irq_edge_is_not_raised_with_ier_clear() {
        let (mut uart, console) = new_uart();
        console.write(b"a");
        uart.poll();
        uart.write(REG_DATA, b'x');
        assert!(!uart.irq_edge());
    }

    #[test]
    fn irq_line_follows_ier_masking() {
        let (mut uart, console) = new_uart();
        console.write(b"a");
        uart.poll();
        uart.write(REG_IER, IER_RX_AVAILABLE);
        assert!(uart.level());
        uart.write(REG_IER, 0);
        assert!(!uart.level());
        assert!(!uart.irq_edge());
        uart.write(REG_IER, IER_RX_AVAILABLE);
        assert!(uart.irq_edge());
    }

    #[test]
    fn uart_interrupt_reaches_the_pic_as_irq4() {
        let (mut uart, console) = new_uart();
        let mut pics = Pics::default();
        for (command, data, base, icw3) in [(0x20, 0x21, 0x20, 0x04), (0xa0, 0xa1, 0x28, 0x02)] {
            pics.io_write(command, 0x11);
            pics.io_write(data, base);
            pics.io_write(data, icw3);
            pics.io_write(data, 0x01);
            pics.io_write(data, 0);
        }
        uart.write(REG_IER, IER_RX_AVAILABLE);
        console.write(b"ab");
        uart.poll();
        if uart.irq_edge() {
            pics.raise(COM1_IRQ);
        }
        assert!(pics.has_pending());
        assert_eq!(pics.ack(), Some(0x24));
        // The handler drains the data, then EOIs; no second interrupt follows.
        assert_eq!(uart.read(REG_DATA), b'a');
        assert_eq!(uart.read(REG_DATA), b'b');
        if uart.irq_edge() {
            pics.raise(COM1_IRQ);
        }
        pics.io_write(0x20, 0x20);
        assert!(!pics.has_pending());
        // A later byte raises it again.
        console.write(b"c");
        uart.poll();
        if uart.irq_edge() {
            pics.raise(COM1_IRQ);
        }
        assert_eq!(pics.ack(), Some(0x24));
    }

    #[test]
    fn thre_interrupt_reaches_the_pic_once_per_transmit() {
        let (mut uart, _) = new_uart();
        let mut pics = Pics::default();
        uart.write(REG_IER, IER_THR_EMPTY);
        let mut raised = 0;
        for byte in b"abc" {
            uart.write(REG_DATA, *byte);
            if uart.irq_edge() {
                pics.raise(COM1_IRQ);
                raised += 1;
            }
            // The driver's interrupt handler reads IIR, clearing the cause.
            assert_eq!(uart.read(REG_IIR_FCR), IIR_THR_EMPTY);
            assert!(!uart.irq_edge());
        }
        // The first edge happened when IER was written, which this loop did not sample; each
        // write then arms one more.
        assert_eq!(raised, 3);
    }

    // Loopback

    #[test]
    fn loopback_returns_transmitted_bytes_to_the_receiver_not_the_console() {
        let (mut uart, console) = new_uart();
        loopback(&mut uart);
        uart.write(REG_DATA, 0x5a);
        assert!(drain_tx(&console).is_empty());
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, LSR_DATA_READY);
        assert_eq!(uart.read(REG_DATA), 0x5a);
    }

    #[test]
    fn loopback_fifo_holds_sixteen_bytes() {
        let (mut uart, _) = new_uart();
        enable_fifo(&mut uart);
        loopback(&mut uart);
        for byte in 0..16 {
            uart.write(REG_DATA, byte);
        }
        assert_eq!(uart.read(REG_LSR) & LSR_OVERRUN, 0);
        for byte in 0..16 {
            assert_eq!(uart.read(REG_DATA), byte);
        }
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }

    #[test]
    fn loopback_fifo_overrun_drops_the_new_byte_and_sets_oe() {
        let (mut uart, _) = new_uart();
        enable_fifo(&mut uart);
        loopback(&mut uart);
        for byte in 0..17 {
            uart.write(REG_DATA, byte);
        }
        let lsr = uart.read(REG_LSR);
        assert_eq!(lsr & LSR_OVERRUN, LSR_OVERRUN);
        assert_eq!(lsr & LSR_DATA_READY, LSR_DATA_READY);
        // Reading LSR cleared OE.
        assert_eq!(uart.read(REG_LSR) & LSR_OVERRUN, 0);
        for byte in 0..16 {
            assert_eq!(uart.read(REG_DATA), byte);
        }
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
    }

    #[test]
    fn overrun_without_the_fifo_replaces_the_held_byte() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.write(REG_DATA, 1);
        uart.write(REG_DATA, 2);
        assert_eq!(uart.read(REG_LSR) & LSR_OVERRUN, LSR_OVERRUN);
        assert_eq!(uart.read(REG_DATA), 2);
        assert_eq!(uart.read(REG_DATA), 0);
    }

    #[test]
    fn lsr_read_clears_overrun_only() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.write(REG_DATA, 1);
        uart.write(REG_DATA, 2);
        assert_eq!(uart.read(REG_LSR), LSR_IDLE | LSR_DATA_READY | LSR_OVERRUN);
        assert_eq!(uart.read(REG_LSR), LSR_IDLE | LSR_DATA_READY);
    }

    #[test]
    fn host_input_is_not_received_in_loopback() {
        let (mut uart, console) = new_uart();
        loopback(&mut uart);
        console.write(b"h");
        assert!(!uart.poll());
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        // Reading RBR in loopback must not pull host input in either.
        uart.read(REG_DATA);
        assert_eq!(uart.read(REG_LSR) & LSR_DATA_READY, 0);
        assert!(console.has_input());
        uart.write(REG_MCR, 0);
        assert!(uart.poll());
        assert_eq!(uart.read(REG_DATA), b'h');
    }

    #[test]
    fn console_output_resumes_after_leaving_loopback() {
        let (mut uart, console) = new_uart();
        uart.write(REG_DATA, b'a');
        loopback(&mut uart);
        uart.write(REG_DATA, b'b');
        uart.write(REG_MCR, 0);
        uart.write(REG_DATA, b'c');
        assert_eq!(drain_tx(&console), b"ac");
    }

    // Modem control and status

    #[test]
    fn msr_outside_loopback_always_shows_cts_dsr_dcd() {
        let (mut uart, _) = new_uart();
        for mcr in 0..=MCR_MASK & !MCR_LOOPBACK {
            uart.write(REG_MCR, mcr);
            assert_eq!(
                uart.read(REG_MSR) & MSR_ALL_INPUTS,
                MSR_NOT_LOOPBACK,
                "mcr {mcr:#x}"
            );
        }
    }

    #[test]
    fn loopback_wires_mcr_outputs_to_msr_inputs() {
        let (mut uart, _) = new_uart();
        for outputs in 0..16u8 {
            uart.write(REG_MCR, MCR_LOOPBACK | outputs);
            let mut expected = 0;
            if outputs & MCR_RTS != 0 {
                expected |= MSR_CTS;
            }
            if outputs & MCR_DTR != 0 {
                expected |= MSR_DSR;
            }
            if outputs & MCR_OUT1 != 0 {
                expected |= MSR_RI;
            }
            if outputs & MCR_OUT2 != 0 {
                expected |= MSR_DCD;
            }
            assert_eq!(
                uart.read(REG_MSR) & MSR_ALL_INPUTS,
                expected,
                "outputs {outputs:#x}"
            );
        }
    }

    #[test]
    fn linux_autoconfig_loopback_probe() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, 0);
        assert_eq!(uart.read(REG_IER), 0);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_OUT2 | MCR_RTS);
        assert_eq!(uart.read(REG_MSR) & 0xf0, MSR_CTS | MSR_DCD);
        uart.write(
            REG_MCR,
            MCR_LOOPBACK | MCR_OUT1 | MCR_OUT2 | MCR_RTS | MCR_DTR,
        );
        assert_eq!(uart.read(REG_MSR) & 0xf0, MSR_ALL_INPUTS);
        uart.write(REG_MCR, 0);
        // FIFO check: 16550A reports the FIFO-enabled bits.
        uart.write(
            REG_IIR_FCR,
            FCR_FIFO_ENABLE | FCR_CLEAR_RX | FCR_CLEAR_TX | 0xc0,
        );
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, IIR_FIFO_ENABLED);
        uart.write(REG_IIR_FCR, 0);
        assert_eq!(uart.read(REG_IIR_FCR) & IIR_FIFO_ENABLED, 0);
        // Scratch register distinguishes a 16450 from an 8250.
        uart.write(REG_SCRATCH, 0xa5);
        assert_eq!(uart.read(REG_SCRATCH), 0xa5);
        uart.write(REG_SCRATCH, 0x5a);
        assert_eq!(uart.read(REG_SCRATCH), 0x5a);
    }

    #[test]
    fn entering_loopback_with_idle_outputs_reports_input_deltas() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        // CTS, DSR and DCD fell.
        assert_eq!(
            uart.read(REG_MSR),
            MSR_DELTA_CTS | MSR_DELTA_DSR | MSR_DELTA_DCD
        );
        // Delta bits clear on read.
        assert_eq!(uart.read(REG_MSR), 0);
    }

    #[test]
    fn msr_delta_bits_track_each_input_change() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.read(REG_MSR);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_RTS);
        assert_eq!(uart.read(REG_MSR), MSR_CTS | MSR_DELTA_CTS);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_RTS | MCR_DTR);
        assert_eq!(uart.read(REG_MSR), MSR_CTS | MSR_DSR | MSR_DELTA_DSR);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_RTS | MCR_DTR | MCR_OUT2);
        assert_eq!(
            uart.read(REG_MSR),
            MSR_CTS | MSR_DSR | MSR_DCD | MSR_DELTA_DCD
        );
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_RTS | MCR_DTR | MCR_OUT2);
        assert_eq!(uart.read(REG_MSR), MSR_CTS | MSR_DSR | MSR_DCD);
    }

    #[test]
    fn ri_reports_only_its_trailing_edge() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.read(REG_MSR);
        // Rising RI sets the input bit but no TERI.
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_OUT1);
        assert_eq!(uart.read(REG_MSR), MSR_RI);
        // Falling RI sets TERI.
        uart.write(REG_MCR, MCR_LOOPBACK);
        assert_eq!(uart.read(REG_MSR), MSR_TRAILING_EDGE_RI);
    }

    #[test]
    fn deltas_accumulate_until_msr_is_read() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.read(REG_MSR);
        uart.write(REG_MCR, MCR_LOOPBACK | MCR_RTS);
        uart.write(REG_MCR, MCR_LOOPBACK);
        // CTS toggled twice and is back to 0 but the delta is remembered.
        assert_eq!(uart.read(REG_MSR), MSR_DELTA_CTS);
    }

    #[test]
    fn leaving_loopback_restores_the_idle_modem_inputs() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.read(REG_MSR);
        uart.write(REG_MCR, 0);
        assert_eq!(
            uart.read(REG_MSR),
            MSR_NOT_LOOPBACK | MSR_DELTA_CTS | MSR_DELTA_DSR | MSR_DELTA_DCD
        );
    }

    #[test]
    fn modem_status_interrupt_needs_ier_and_clears_on_msr_read() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        assert!(!uart.level());
        uart.write(REG_IER, IER_MODEM_STATUS);
        assert!(uart.irq_edge());
        assert_eq!(uart.read(REG_IIR_FCR), IIR_MODEM_STATUS);
        uart.read(REG_MSR);
        assert!(!uart.level());
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn mcr_writes_outside_loopback_with_steady_inputs_raise_no_modem_interrupt() {
        let (mut uart, _) = new_uart();
        uart.write(REG_IER, IER_MODEM_STATUS);
        uart.write(REG_MCR, MCR_DTR | MCR_RTS | MCR_OUT2);
        assert!(!uart.level());
        assert_eq!(uart.read(REG_MSR), MSR_NOT_LOOPBACK);
    }

    #[test]
    fn loopback_overrun_flag_is_not_set_by_a_single_byte() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.write(REG_DATA, 7);
        assert_eq!(uart.read(REG_LSR) & LSR_OVERRUN, 0);
    }

    #[test]
    fn line_status_interrupt_needs_its_ier_bit() {
        let (mut uart, _) = new_uart();
        loopback(&mut uart);
        uart.write(REG_DATA, 1);
        uart.write(REG_DATA, 2);
        assert_eq!(iir_cause(&mut uart) & IIR_NO_INTERRUPT, IIR_NO_INTERRUPT);
        uart.write(REG_IER, IER_RX_LINE_STATUS);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_RX_LINE_STATUS);
        uart.read(REG_LSR);
        assert_eq!(uart.read(REG_IIR_FCR), IIR_NO_INTERRUPT);
    }

    #[test]
    fn error_bits_that_need_a_faulty_line_never_set() {
        let (mut uart, console) = new_uart();
        enable_fifo(&mut uart);
        console.write(&[0, 0xff, 0x55]);
        uart.poll();
        assert_eq!(uart.read(REG_LSR) & 0x9c, 0);
    }
}
