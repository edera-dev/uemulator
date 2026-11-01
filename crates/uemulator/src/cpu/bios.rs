//! The BIOS a PC's boot loader expects, written as host code instead of firmware.
//!
//! A disk's boot sector (syslinux, GRUB, the Linux real-mode setup code) was written for the
//! services of a PC BIOS: it calls `int 13h` to read the disk, `int 15h` for the memory map, and
//! so on, and keeps time by the tick counter in the BIOS data area. There is no firmware here, so
//! `bios_install` builds what POST would have left behind and the services run in Rust:
//!
//! - Every interrupt vector points at a small stub in the ROM segment, `[sti;] out TRAP_PORT, al;
//!   iret`; the timer's also runs `int 1Ch` before the `iret`, as the BIOS's does.
//!   The `out` (see `Cpu::port_io`) lands in `bios_trap`, which works out the vector from where
//!   the stub is, reads the caller's registers, and writes the results back. The `iret` then
//!   returns to the caller with the flags the handler left in the stacked FLAGS word, so a
//!   service reports failure with the carry flag exactly as the real ones do. The services start
//!   with `sti`, as the real ones do: a boot loader that polls the tick counter with interrupts
//!   masked (syslinux's menu does) only ever sees it move because each call opens a window for
//!   the timer interrupt.
//! - The BIOS data area, the interrupt vector table, the PIC (IRQ0 at INT 08h) and the PIT (the
//!   18.2 Hz tick that advances the counter in the data area) are set up as POST leaves them.
//!
//! Text goes to the 16550 and keys come from it, so the host console is the screen and keyboard.
//! There is no VGA: a boot loader that writes to B8000 draws into plain RAM.

use alloc::collections::VecDeque;

use super::bios_keys::KeyDecoder;
use super::exception::Exception;
use super::flags::{CF, IF, ZF};
use super::modes::real_mode_segment;
use super::*;
use crate::boot::e820_map;
use crate::machine::ExitReason;

/// Written by the stubs. 0xb2 is the APM control port, which nothing here decodes.
pub(crate) const TRAP_PORT: u16 = 0xb2;

const ROM_SEGMENT: u16 = 0xf000;
const ROM_BASE: u64 = 0xf_0000;
/// Where the stubs start inside the ROM segment, and the size of each: `sti` (1 byte, software
/// interrupts only), `out imm8, al` (2), `iret` (1), padding.
const STUB_OFFSET: u64 = 0x100;
const STUB_STRIDE: u64 = 8;
const VECTOR_COUNT: u64 = 256;
const STI: u8 = 0xfb;
const INT_IMM8: u8 = 0xcd;
const OUT_IMM8_AL: u8 = 0xe6;
const IRET: u8 = 0xcf;
const IVT_ENTRY_BYTES: u64 = 4;
/// Tables the vectors for `int 1Eh` and `int 41h` point at, in the ROM segment.
const FLOPPY_TABLE_OFFSET: u64 = 0x1000;
const FIXED_DISK_TABLE_OFFSET: u64 = 0x1010;
/// The date string and model byte at the top of the BIOS segment, where boot code looks.
const BIOS_DATE_OFFSET: u64 = 0xfff5;
const BIOS_DATE: &[u8; 8] = b"01/01/26";
const BIOS_MODEL_OFFSET: u64 = 0xfffe;
const BIOS_MODEL_AT: u8 = 0xfc;
/// The default floppy parameter table of a 1.44 MB drive (`int 1Eh`).
const FLOPPY_TABLE: [u8; 11] = [
    0xdf, 0x02, 0x25, 0x02, 18, 0x1b, 0xff, 0x54, 0xf6, 0x0f, 0x08,
];

const VECTOR_FLOPPY_PARAMETERS: u64 = 0x1e;
const VECTOR_FIXED_DISK_PARAMETERS: u64 = 0x41;
const VECTOR_TIMER_IRQ: u64 = 0x08;
/// The timer tick hook: the BIOS's INT 08h ends by calling it on every tick, and boot loaders
/// (syslinux keeps its clock there) install their own handler.
const VECTOR_USER_TIMER_TICK: u8 = 0x1c;
const VECTOR_LAST_EXCEPTION: u64 = 0x07;
const VECTOR_MASTER_IRQS: core::ops::RangeInclusive<u64> = 0x09..=0x0f;
const VECTOR_SLAVE_IRQS: core::ops::RangeInclusive<u64> = 0x70..=0x77;
const VECTOR_VIDEO: u64 = 0x10;
const VECTOR_EQUIPMENT: u64 = 0x11;
const VECTOR_MEMORY_SIZE: u64 = 0x12;
const VECTOR_DISK: u64 = 0x13;
const VECTOR_SYSTEM: u64 = 0x15;
const VECTOR_KEYBOARD: u64 = 0x16;
const VECTOR_NO_BOOT: u64 = 0x18;
const VECTOR_BOOTSTRAP: u64 = 0x19;
const VECTOR_TIME: u64 = 0x1a;

const BDA: u64 = 0x400;
const BDA_COM1_BASE: u64 = BDA;
const BDA_EBDA_SEGMENT: u64 = BDA + 0x0e;
const BDA_EQUIPMENT: u64 = BDA + 0x10;
const BDA_MEMORY_KIB: u64 = BDA + 0x13;
const BDA_KEY_BUFFER_HEAD: u64 = BDA + 0x1a;
const BDA_KEY_BUFFER_TAIL: u64 = BDA + 0x1c;
const BDA_VIDEO_MODE: u64 = BDA + 0x49;
const BDA_COLUMNS: u64 = BDA + 0x4a;
const BDA_PAGE_BYTES: u64 = BDA + 0x4c;
const BDA_CURSOR: u64 = BDA + 0x50;
const BDA_CURSOR_SHAPE: u64 = BDA + 0x60;
const BDA_CRTC_PORT: u64 = BDA + 0x63;
const BDA_TICKS: u64 = BDA + 0x6c;
const BDA_MIDNIGHT: u64 = BDA + 0x70;
const BDA_DISK_COUNT: u64 = BDA + 0x75;
const BDA_KEY_BUFFER_START: u64 = BDA + 0x80;
const BDA_KEY_BUFFER_END: u64 = BDA + 0x82;
const BDA_ROWS: u64 = BDA + 0x84;
const BDA_CHARACTER_HEIGHT: u64 = BDA + 0x85;
const KEY_BUFFER_START: u16 = 0x1e;
const KEY_BUFFER_END: u16 = 0x3e;
/// One serial port, a math coprocessor and an 80x25 colour display, no floppy drive.
const EQUIPMENT_WORD: u16 = (1 << 9) | (0b10 << 4) | (1 << 1);
const VIDEO_MODE_TEXT_80X25: u8 = 3;
const TEXT_COLUMNS: u8 = 80;
const TEXT_ROWS: u8 = 25;
const TEXT_PAGE_BYTES: u16 = 0x1000;
const CURSOR_SHAPE: u16 = 0x0607;
const CHARACTER_HEIGHT: u8 = 16;
const CRTC_COLOR_PORT: u16 = 0x3d4;
/// The extended BIOS data area sits just under the 640 KiB the BDA reports.
const EBDA_SEGMENT: u16 = 0x9fc0;
const EBDA_KIB: u64 = 1;
const BASE_MEMORY_KIB: u16 = 639;
/// Timer ticks in a day at 18.2065 Hz: the counter wraps there and sets the midnight flag.
const TICKS_PER_DAY: u32 = 0x18_00b0;

const BOOT_SECTOR_ADDRESS: u64 = 0x7c00;
const SECTOR_BYTES: usize = 512;
const BOOT_SIGNATURE: [u8; 2] = [0x55, 0xaa];
const FIRST_FIXED_DRIVE: u8 = 0x80;

const PIC_MASTER_COMMAND: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_COMMAND: u16 = 0xa0;
const PIC_SLAVE_DATA: u16 = 0xa1;
const PIC_ICW1_INIT_WITH_ICW4: u32 = 0x11;
const PIC_ICW4_8086: u32 = 0x01;
const PIC_EOI: u32 = 0x20;
const PIC_MASTER_VECTORS: u32 = 0x08;
const PIC_SLAVE_VECTORS: u32 = 0x70;
/// The slave hangs off IRQ2 of the master (a bit mask for ICW3), and the master is IRQ2 of the
/// slave (a number).
const PIC_MASTER_ICW3: u32 = 1 << 2;
const PIC_SLAVE_ICW3: u32 = 2;
/// Only the timer and the cascade are unmasked: no device here raises the others for a BIOS.
const PIC_MASTER_MASK: u32 = !((1 << 0) | (1 << 2)) & 0xff;
const PIC_SLAVE_MASK: u32 = 0xff;
const PIT_COMMAND_PORT: u16 = 0x43;
const PIT_CHANNEL0_PORT: u16 = 0x40;
/// Channel 0, low then high byte, mode 3 (square wave), binary; a count of 0 is 65536, which is
/// 18.2 Hz.
const PIT_CHANNEL0_MODE3: u32 = 0x36;
const UART_DATA_PORT: u16 = 0x3f8;
const UART_LINE_STATUS_PORT: u16 = 0x3fd;
const UART_DATA_READY: u32 = 1;
const CMOS_INDEX_PORT: u16 = 0x70;
const CMOS_DATA_PORT: u16 = 0x71;
const CMOS_SECONDS: u32 = 0;
const CMOS_MINUTES: u32 = 2;
const CMOS_HOURS: u32 = 4;
const CMOS_DAY: u32 = 7;
const CMOS_MONTH: u32 = 8;
const CMOS_YEAR: u32 = 9;
const CMOS_STATUS_B: u32 = 0x0b;
const CMOS_CENTURY: u32 = 0x32;
const CMOS_BINARY_MODE: u32 = 1 << 2;
/// Keys read from the console wait here; more than this is dropped.
const KEY_QUEUE_LIMIT: usize = 32;

const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RBX: usize = 3;
const RBP: usize = 5;
const RSI: usize = 6;
const RDI: usize = 7;

/// Offsets of the words `int` pushed, from the stack pointer inside a stub: the stub's `iret`
/// pops them.
const FRAME_IP: u64 = 0;
const FRAME_CS: u64 = 2;
const FRAME_FLAGS: u64 = 4;
const INT_INSTRUCTION_BYTES: u16 = 2;

// Disk status codes in AH.
const STATUS_OK: u8 = 0x00;
const STATUS_BAD_COMMAND: u8 = 0x01;
const STATUS_WRITE_PROTECTED: u8 = 0x03;
const STATUS_SECTOR_NOT_FOUND: u8 = 0x04;
const STATUS_WRITE_FAULT: u8 = 0xcc;
const STATUS_CONTROLLER_FAILURE: u8 = 0x20;
const STATUS_NOT_SUPPORTED: u8 = 0x86;

/// A fixed disk is reported as 255 heads of 63 sectors, which is what every BIOS translates to.
const GEOMETRY_HEADS: u64 = 255;
const GEOMETRY_SECTORS: u64 = 63;
/// The CHS interfaces can address 10 bits of cylinders; the EDD geometry block has 32.
const CHS_MAX_CYLINDERS: u64 = 1024;
const EDD_MAX_CYLINDERS: u64 = 16383;
const EDD_VERSION_3: u8 = 0x30;
const EDD_INSTALL_CHECK: u16 = 0x55aa;
const EDD_INSTALL_CHECK_REPLY: u16 = 0xaa55;
/// Extended read/write/verify/seek (bit 0) and the parameters call (bit 2).
const EDD_SUPPORT: u16 = 0x0005;
const EDD_PARAMETERS_BYTES: usize = 0x1a;
const EDD_GEOMETRY_VALID: u16 = 0x0002;
const EDD_PACKET_BYTES: usize = 16;
const EDD_PACKET_FLAT_BYTES: usize = 24;
const EDD_PACKET_FLAT_OFFSET: u64 = 0x10;
const EDD_FLAT_MARKER: u16 = 0xffff;

const SMAP: u32 = 0x534d_4150;
const E820_ENTRY_BYTES: usize = 20;
const E820_EXTENDED_ENTRY_BYTES: u32 = 24;
const E820_ENTRY_ENABLED: u32 = 1;
const SIXTEEN_MIB: u64 = 1 << 24;
const ONE_MIB: u64 = 1 << 20;
/// What `int 15h` AH=88h reports at most: the ISA hole above 15 MiB.
const EXTENDED_KIB_LIMIT: u64 = 0x3c00;
const COPY_DESCRIPTOR_SOURCE: u64 = 0x10;
const COPY_DESCRIPTOR_DESTINATION: u64 = 0x18;
const A20_GATE_SUPPORT_KEYBOARD_AND_PORT_92: u16 = 3;

const BELL: u8 = 0x07;
const BACKSPACE: u8 = 0x08;
const LINE_FEED: u8 = 0x0a;
const CARRIAGE_RETURN: u8 = 0x0d;
const NO_BOOT_MESSAGE: &[u8] = b"No bootable device.\r\n";
const LOG_KEY_BIOS_EXCEPTION: u64 = 0xb105_0000;
const LOG_KEY_BIOS_UNSUPPORTED: u64 = 0xb105_1000;

/// What the BIOS remembers between calls.
pub(crate) struct Bios {
    /// Keys read from the console that no `int 16h` has taken yet, as key words (scan code in
    /// AH, character in AL).
    keys: VecDeque<u16>,
    key_decoder: KeyDecoder,
    /// Where the terminal's cursor is, so `int 10h` only moves it when the caller asks for a
    /// different place.
    row: u8,
    column: u8,
    drives: usize,
    /// End of the flat RAM buffer, which is past the PCI hole when there is RAM above 4 GiB.
    ram_top: u64,
}

impl Cpu {
    fn reg16(&self, index: usize) -> u16 {
        self.gpr[index] as u16
    }

    fn set_reg16(&mut self, index: usize, value: u16) {
        self.gpr[index] = (self.gpr[index] & !0xffff) | value as u64;
    }

    fn set_reg32(&mut self, index: usize, value: u32) {
        self.gpr[index] = value as u64;
    }

    fn ah(&self) -> u8 {
        (self.gpr[RAX] >> 8) as u8
    }

    fn al(&self) -> u8 {
        self.gpr[RAX] as u8
    }

    fn set_ah(&mut self, value: u8) {
        self.gpr[RAX] = (self.gpr[RAX] & !0xff00) | ((value as u64) << 8);
    }

    fn set_al(&mut self, value: u8) {
        self.gpr[RAX] = (self.gpr[RAX] & !0xff) | value as u64;
    }

    fn dl(&self) -> u8 {
        self.gpr[RDX] as u8
    }

    fn dh(&self) -> u8 {
        (self.gpr[RDX] >> 8) as u8
    }

    /// Linear address of `segment:offset` through a segment register's real-mode base.
    fn real_address(&self, segment: usize, offset: u16) -> u64 {
        self.seg[segment].base.wrapping_add(offset as u64)
    }

    fn peek(&self, linear: u64, len: usize) -> Option<&[u8]> {
        self.ram.slice(linear, len)
    }

    fn peek_u16(&self, linear: u64) -> Option<u16> {
        Some(u16::from_le_bytes(self.peek(linear, 2)?.try_into().ok()?))
    }

    fn peek_u32(&self, linear: u64) -> Option<u32> {
        Some(u32::from_le_bytes(self.peek(linear, 4)?.try_into().ok()?))
    }

    fn peek_u64(&self, linear: u64) -> Option<u64> {
        Some(u64::from_le_bytes(self.peek(linear, 8)?.try_into().ok()?))
    }

    /// Writes guest RAM the way a store does, so translated code on those pages is dropped.
    fn store(&mut self, linear: u64, data: &[u8]) -> bool {
        let Ok(start) = usize::try_from(linear) else {
            return false;
        };
        let Some(end) = start.checked_add(data.len()) else {
            return false;
        };
        if !self.ram.contains(linear, data.len() as u64) {
            return false;
        }
        for page in (start >> PAGE_SHIFT)..=(end.saturating_sub(1) >> PAGE_SHIFT) {
            if end > start && self.page_has_code[page] != 0 {
                self.code_written(page);
            }
        }
        self.ram[start..end].copy_from_slice(data);
        true
    }

    fn store_u16(&mut self, linear: u64, value: u16) {
        self.store(linear, &value.to_le_bytes());
    }

    fn store_u32(&mut self, linear: u64, value: u32) {
        self.store(linear, &value.to_le_bytes());
    }

    /// Where `int` left the return frame: the stub's `iret` pops it.
    fn frame(&self) -> u64 {
        self.seg[SEG_SS].base.wrapping_add(self.stack_pointer())
    }

    /// Sets or clears `bit` of the FLAGS word the caller gets back from `iret`.
    fn set_returned_flag(&mut self, bit: u64, on: bool) {
        let at = self.frame() + FRAME_FLAGS;
        if let Some(flags) = self.peek_u16(at) {
            let flags = if on {
                flags | bit as u16
            } else {
                flags & !(bit as u16)
            };
            self.store_u16(at, flags);
        }
    }

    /// Success: AH = 0 and carry clear.
    fn succeed(&mut self) {
        self.set_ah(STATUS_OK);
        self.set_returned_flag(CF, false);
    }

    /// Failure: `status` in AH and carry set.
    fn fail(&mut self, status: u8) {
        self.set_ah(status);
        self.set_returned_flag(CF, true);
    }

    fn port_write(&mut self, port: u16, value: u32) {
        let now = self.now_ns();
        self.dev.lock().io_out(port, 1, value, now);
    }

    fn port_read(&mut self, port: u16) -> u32 {
        let now = self.now_ns();
        self.dev.lock().io_in(port, 1, now)
    }

    /// Builds the interrupt vector table, the stubs, the BIOS data area and the timer, as a BIOS
    /// leaves them when it hands over to the boot loader. Needs the devices to exist.
    pub(crate) fn bios_install(&mut self) {
        let drives = self.dev.lock().disk_count();
        let ram_top = self.ram.len() as u64;
        self.bios = Some(Box::new(Bios {
            keys: VecDeque::new(),
            key_decoder: KeyDecoder::default(),
            row: 0,
            column: 0,
            drives,
            ram_top,
        }));
        for vector in 0..VECTOR_COUNT {
            let offset = STUB_OFFSET + vector * STUB_STRIDE;
            let mut stub = [0u8; STUB_STRIDE as usize];
            let body: &[u8] = match vector {
                VECTOR_TIMER_IRQ => &[
                    OUT_IMM8_AL,
                    TRAP_PORT as u8,
                    INT_IMM8,
                    VECTOR_USER_TIMER_TICK,
                    IRET,
                ],
                v if Self::bios_vector_is_hardware(v) => &[OUT_IMM8_AL, TRAP_PORT as u8, IRET],
                _ => &[STI, OUT_IMM8_AL, TRAP_PORT as u8, IRET],
            };
            stub[..body.len()].copy_from_slice(body);
            self.store(ROM_BASE + offset, &stub);
            self.bios_set_vector(vector, ROM_SEGMENT, offset as u16);
        }
        self.store(ROM_BASE + FLOPPY_TABLE_OFFSET, &FLOPPY_TABLE);
        self.bios_set_vector(
            VECTOR_FLOPPY_PARAMETERS,
            ROM_SEGMENT,
            FLOPPY_TABLE_OFFSET as u16,
        );
        let (cylinders, heads, sectors) = self.disk_geometry(0).unwrap_or((0, 0, 0));
        let mut table = [0u8; 16];
        table[0..2].copy_from_slice(&(cylinders as u16).to_le_bytes());
        table[2] = heads as u8;
        table[12..14].copy_from_slice(&(cylinders as u16).to_le_bytes());
        table[14] = sectors as u8;
        self.store(ROM_BASE + FIXED_DISK_TABLE_OFFSET, &table);
        self.bios_set_vector(
            VECTOR_FIXED_DISK_PARAMETERS,
            ROM_SEGMENT,
            FIXED_DISK_TABLE_OFFSET as u16,
        );
        self.store(ROM_BASE + BIOS_DATE_OFFSET, BIOS_DATE);
        self.store(ROM_BASE + BIOS_MODEL_OFFSET, &[BIOS_MODEL_AT]);

        self.store_u16(BDA_COM1_BASE, UART_DATA_PORT);
        self.store_u16(BDA_EBDA_SEGMENT, EBDA_SEGMENT);
        self.store(EBDA_SEGMENT as u64 * 16, &[EBDA_KIB as u8]);
        self.store_u16(BDA_EQUIPMENT, EQUIPMENT_WORD);
        self.store_u16(BDA_MEMORY_KIB, BASE_MEMORY_KIB);
        for at in [
            BDA_KEY_BUFFER_HEAD,
            BDA_KEY_BUFFER_TAIL,
            BDA_KEY_BUFFER_START,
        ] {
            self.store_u16(at, KEY_BUFFER_START);
        }
        self.store_u16(BDA_KEY_BUFFER_END, KEY_BUFFER_END);
        self.store(BDA_VIDEO_MODE, &[VIDEO_MODE_TEXT_80X25]);
        self.store_u16(BDA_COLUMNS, TEXT_COLUMNS as u16);
        self.store_u16(BDA_PAGE_BYTES, TEXT_PAGE_BYTES);
        self.store_u16(BDA_CURSOR, 0);
        self.store_u16(BDA_CURSOR_SHAPE, CURSOR_SHAPE);
        self.store_u16(BDA_CRTC_PORT, CRTC_COLOR_PORT);
        self.store(BDA_DISK_COUNT, &[drives as u8]);
        self.store(BDA_ROWS, &[TEXT_ROWS - 1]);
        self.store(BDA_CHARACTER_HEIGHT, &[CHARACTER_HEIGHT]);

        self.port_write(PIC_MASTER_COMMAND, PIC_ICW1_INIT_WITH_ICW4);
        self.port_write(PIC_MASTER_DATA, PIC_MASTER_VECTORS);
        self.port_write(PIC_MASTER_DATA, PIC_MASTER_ICW3);
        self.port_write(PIC_MASTER_DATA, PIC_ICW4_8086);
        self.port_write(PIC_SLAVE_COMMAND, PIC_ICW1_INIT_WITH_ICW4);
        self.port_write(PIC_SLAVE_DATA, PIC_SLAVE_VECTORS);
        self.port_write(PIC_SLAVE_DATA, PIC_SLAVE_ICW3);
        self.port_write(PIC_SLAVE_DATA, PIC_ICW4_8086);
        self.port_write(PIC_MASTER_DATA, PIC_MASTER_MASK);
        self.port_write(PIC_SLAVE_DATA, PIC_SLAVE_MASK);
        self.port_write(PIT_COMMAND_PORT, PIT_CHANNEL0_MODE3);
        self.port_write(PIT_CHANNEL0_PORT, 0);
        self.port_write(PIT_CHANNEL0_PORT, 0);
    }

    /// Exceptions and IRQs enter with interrupts masked and handle them themselves; the software
    /// services are what start with `sti`.
    fn bios_vector_is_hardware(vector: u64) -> bool {
        vector <= VECTOR_LAST_EXCEPTION
            || vector == VECTOR_TIMER_IRQ
            || VECTOR_MASTER_IRQS.contains(&vector)
            || VECTOR_SLAVE_IRQS.contains(&vector)
    }

    fn bios_set_vector(&mut self, vector: u64, segment: u16, offset: u16) {
        self.store_u32(
            vector * IVT_ENTRY_BYTES,
            offset as u32 | ((segment as u32) << 16),
        );
    }

    /// Loads the first sector of the first disk at 0000:7C00 and starts it with DL = 0x80, the
    /// way `int 19h` does. A disk without the boot signature is "no bootable device".
    pub(crate) fn bios_boot(&mut self) {
        let mut sector = [0u8; SECTOR_BYTES];
        let loaded = self
            .dev
            .lock()
            .disk_backend(0)
            .is_some_and(|disk| disk.read_at(0, &mut sector).is_ok());
        if !loaded || sector[SECTOR_BYTES - 2..] != BOOT_SIGNATURE {
            self.bios_no_boot_device();
            return;
        }
        self.store(BOOT_SECTOR_ADDRESS, &sector);
        self.gpr = [0; 16];
        self.gpr[RDX] = FIRST_FIXED_DRIVE as u64;
        self.set_stack_pointer(BOOT_SECTOR_ADDRESS);
        let code = self.seg[SEG_CS];
        self.seg[SEG_CS] = Seg {
            sel: 0,
            base: 0,
            ..code
        };
        for index in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
            self.seg[index] = real_mode_segment(0, 0);
        }
        self.rip = BOOT_SECTOR_ADDRESS;
        self.materialize_lazy_flags();
        self.eflags |= IF;
    }

    /// `int 18h`: nothing to boot. Powers the machine off: a halt with interrupts masked is how
    /// this library reports a guest shutting down.
    fn bios_no_boot_device(&mut self) {
        for byte in NO_BOOT_MESSAGE {
            self.console_put(*byte);
        }
        self.materialize_lazy_flags();
        self.eflags &= !IF;
        self.halted = true;
    }

    /// Called by `port_io` for a write to `TRAP_PORT`: the guest is inside one of the stubs.
    pub(crate) fn bios_trap(&mut self) -> ExecResult<()> {
        if self.bios.is_none() {
            return Ok(());
        }
        let linear = self.seg[SEG_CS].base.wrapping_add(self.rip);
        let vector = linear.wrapping_sub(ROM_BASE + STUB_OFFSET) / STUB_STRIDE;
        if linear < ROM_BASE + STUB_OFFSET || vector >= VECTOR_COUNT {
            return Ok(());
        }
        match vector {
            0..=VECTOR_LAST_EXCEPTION => {
                // There is no handler for a CPU exception in real mode, which a real BIOS does
                // not have either; returning would only re-execute the faulting instruction.
                let (cs, ip) = (
                    self.peek_u16(self.frame() + FRAME_CS),
                    self.peek_u16(self.frame() + FRAME_IP),
                );
                self.log_once(LOG_KEY_BIOS_EXCEPTION + vector, || {
                    format!("real-mode exception {vector} at {cs:x?}:{ip:x?}, nothing handles it")
                });
                return Err(Exception::Exit(ExitReason::TripleFault));
            }
            VECTOR_TIMER_IRQ => self.bios_timer_tick(),
            v if VECTOR_MASTER_IRQS.contains(&v) => self.port_write(PIC_MASTER_COMMAND, PIC_EOI),
            v if VECTOR_SLAVE_IRQS.contains(&v) => {
                self.port_write(PIC_SLAVE_COMMAND, PIC_EOI);
                self.port_write(PIC_MASTER_COMMAND, PIC_EOI);
            }
            VECTOR_VIDEO => self.bios_video(),
            VECTOR_EQUIPMENT => {
                let word = self.peek_u16(BDA_EQUIPMENT).unwrap_or(0);
                self.set_reg16(RAX, word);
            }
            VECTOR_MEMORY_SIZE => {
                let kib = self.peek_u16(BDA_MEMORY_KIB).unwrap_or(0);
                self.set_reg16(RAX, kib);
            }
            VECTOR_DISK => self.bios_disk(),
            VECTOR_SYSTEM => self.bios_system(),
            VECTOR_KEYBOARD => self.bios_keyboard(),
            VECTOR_NO_BOOT => self.bios_no_boot_device(),
            VECTOR_BOOTSTRAP => self.bios_bootstrap(),
            VECTOR_TIME => self.bios_time(),
            // A vector nothing here serves: the stub just returns, like an unused BIOS entry.
            _ => {}
        }
        Ok(())
    }

    fn bios_timer_tick(&mut self) {
        let ticks = self.peek_u32(BDA_TICKS).unwrap_or(0).wrapping_add(1);
        if ticks >= TICKS_PER_DAY {
            self.store_u32(BDA_TICKS, 0);
            self.store(BDA_MIDNIGHT, &[1]);
        } else {
            self.store_u32(BDA_TICKS, ticks);
        }
        self.port_write(PIC_MASTER_COMMAND, PIC_EOI);
    }

    /// `int 19h`: loads the boot sector again and continues there.
    fn bios_bootstrap(&mut self) {
        let mut sector = [0u8; SECTOR_BYTES];
        let loaded = self
            .dev
            .lock()
            .disk_backend(0)
            .is_some_and(|disk| disk.read_at(0, &mut sector).is_ok());
        if !loaded || sector[SECTOR_BYTES - 2..] != BOOT_SIGNATURE {
            self.bios_no_boot_device();
            return;
        }
        self.store(BOOT_SECTOR_ADDRESS, &sector);
        self.set_reg16(RDX, FIRST_FIXED_DRIVE as u16);
        // The stub's `iret` continues at the boot sector.
        let frame = self.frame();
        self.store_u16(frame + FRAME_IP, BOOT_SECTOR_ADDRESS as u16);
        self.store_u16(frame + FRAME_CS, 0);
    }

    // ----- text --------------------------------------------------------------------------

    /// Sends one byte to the console and tracks where that leaves the cursor.
    fn console_put(&mut self, byte: u8) {
        self.port_write(UART_DATA_PORT, byte as u32);
        let Some(bios) = self.bios.as_mut() else {
            return;
        };
        match byte {
            CARRIAGE_RETURN => bios.column = 0,
            LINE_FEED => bios.row = (bios.row + 1).min(TEXT_ROWS - 1),
            BACKSPACE => bios.column = bios.column.saturating_sub(1),
            BELL => {}
            _ => {
                bios.column += 1;
                if bios.column >= TEXT_COLUMNS {
                    bios.column = 0;
                    bios.row = (bios.row + 1).min(TEXT_ROWS - 1);
                }
            }
        }
        let cursor = u16::from_le_bytes([bios.column, bios.row]);
        self.store_u16(BDA_CURSOR, cursor);
    }

    /// Moves the terminal's cursor if `row`, `column` is somewhere else than it already is.
    fn console_move(&mut self, row: u8, column: u8) {
        let Some(bios) = self.bios.as_ref() else {
            return;
        };
        if (bios.row, bios.column) == (row, column) {
            return;
        }
        for byte in format!("\x1b[{};{}H", row as u32 + 1, column as u32 + 1).bytes() {
            self.port_write(UART_DATA_PORT, byte as u32);
        }
        if let Some(bios) = self.bios.as_mut() {
            (bios.row, bios.column) = (row, column);
        }
        self.store_u16(BDA_CURSOR, u16::from_le_bytes([column, row]));
    }

    fn bios_video(&mut self) {
        match self.ah() {
            // Set video mode: the display is a terminal, so there is nothing to switch.
            0x00 => {
                let mode = self.al() & 0x7f;
                self.store(BDA_VIDEO_MODE, &[mode]);
            }
            0x02 => self.console_move((self.gpr[RDX] >> 8) as u8, self.gpr[RDX] as u8),
            0x03 => {
                let (row, column) = self.bios.as_ref().map_or((0, 0), |b| (b.row, b.column));
                self.set_reg16(RAX, 0);
                self.set_reg16(RCX, CURSOR_SHAPE);
                self.set_reg16(RDX, u16::from_le_bytes([column, row]));
            }
            // Scroll up/down: only a clear of the whole window means anything to a terminal.
            0x06 | 0x07 if self.al() == 0 => {
                for byte in b"\x1b[2J" {
                    self.port_write(UART_DATA_PORT, *byte as u32);
                }
                if let Some(bios) = self.bios.as_mut() {
                    (bios.row, bios.column) = (u8::MAX, u8::MAX);
                }
                self.console_move(0, 0);
            }
            // Read the character at the cursor: a blank.
            0x08 => self.set_reg16(RAX, 0x0720),
            0x09 | 0x0a => {
                let count = self.reg16(RCX).max(1);
                let byte = self.al();
                for _ in 0..count.min(TEXT_COLUMNS as u16 * TEXT_ROWS as u16) {
                    self.console_put(byte);
                }
            }
            0x0e => {
                let byte = self.al();
                self.console_put(byte);
            }
            0x0f => {
                let mode = self
                    .peek(BDA_VIDEO_MODE, 1)
                    .map_or(VIDEO_MODE_TEXT_80X25, |b| b[0]);
                self.set_reg16(RAX, u16::from_le_bytes([mode, TEXT_COLUMNS]));
                self.gpr[RBX] &= !0xff00;
            }
            // Write string: ES:BP, CX characters, DH:DL position, BL attribute.
            0x13 => {
                let mode = self.al();
                let (row, column) = (self.dh(), self.dl());
                let start = self.real_address(SEG_ES, self.reg16(RBP));
                let count = self.reg16(RCX) as usize;
                let step = if mode & 2 != 0 { 2 } else { 1 };
                let text: Vec<u8> = (0..count)
                    .filter_map(|index| self.peek(start + (index * step) as u64, 1).map(|b| b[0]))
                    .collect();
                let before = self.bios.as_ref().map_or((0, 0), |b| (b.row, b.column));
                self.console_move(row, column);
                for byte in text {
                    self.console_put(byte);
                }
                if mode & 1 == 0 {
                    self.console_move(before.0, before.1);
                }
            }
            // Everything else (cursor shape, pages, palettes, VESA, the VGA and EGA probes) is
            // left unanswered: the registers stay as they were, which callers read as "absent".
            _ => {}
        }
    }

    // ----- keyboard ----------------------------------------------------------------------

    /// Moves what the console has received into the key queue, turning escape sequences into
    /// scan codes (`bios_keys`). The UART paces host input, so a sequence may take several calls
    /// to arrive; one still pending with the line quiet was the Escape key.
    fn bios_collect_keys(&mut self) {
        for _ in 0..KEY_QUEUE_LIMIT {
            if self.port_read(UART_LINE_STATUS_PORT) & UART_DATA_READY == 0 {
                break;
            }
            let byte = self.port_read(UART_DATA_PORT) as u8;
            let now = self.now_ns();
            if let Some(bios) = self.bios.as_mut() {
                for key in bios.key_decoder.push(byte, now) {
                    if bios.keys.len() < KEY_QUEUE_LIMIT {
                        bios.keys.push_back(key);
                    }
                }
            }
        }
        if self.dev.lock().serial_input_pending() {
            return;
        }
        let now = self.now_ns();
        if let Some(bios) = self.bios.as_mut()
            && let Some(escape) = bios.key_decoder.expire(now)
            && bios.keys.len() < KEY_QUEUE_LIMIT
        {
            bios.keys.push_back(escape);
        }
    }

    fn bios_keyboard(&mut self) {
        self.bios_collect_keys();
        let pending = self.bios.as_ref().and_then(|b| b.keys.front().copied());
        match self.ah() {
            // Read a key; with none, wait: run this call again after the next interrupt.
            0x00 | 0x10 => match pending {
                Some(key) => {
                    if let Some(bios) = self.bios.as_mut() {
                        bios.keys.pop_front();
                    }
                    self.set_reg16(RAX, key);
                }
                None => {
                    let frame = self.frame();
                    if let Some(ip) = self.peek_u16(frame + FRAME_IP) {
                        self.store_u16(frame + FRAME_IP, ip.wrapping_sub(INT_INSTRUCTION_BYTES));
                    }
                    // The stub's `sti` left interrupts on, so the halt ends with the next tick.
                    self.halted = true;
                }
            },
            // Is there a key: ZF clear and the key in AX, or ZF set.
            0x01 | 0x11 => match pending {
                Some(key) => {
                    self.set_reg16(RAX, key);
                    self.set_returned_flag(ZF, false);
                }
                None => self.set_returned_flag(ZF, true),
            },
            // Shift state.
            0x02 | 0x12 => self.set_reg16(RAX, 0),
            _ => {}
        }
    }

    // ----- time --------------------------------------------------------------------------

    fn cmos_read(&mut self, register: u32) -> u8 {
        self.port_write(CMOS_INDEX_PORT, register);
        self.port_read(CMOS_DATA_PORT) as u8
    }

    /// The RTC register as BCD, whichever the RTC keeps.
    fn cmos_bcd(&mut self, register: u32, binary: bool) -> u8 {
        let value = self.cmos_read(register);
        if binary {
            ((value / 10) << 4) | (value % 10)
        } else {
            value
        }
    }

    fn bios_time(&mut self) {
        match self.ah() {
            0x00 => {
                let ticks = self.peek_u32(BDA_TICKS).unwrap_or(0);
                let midnight = self.peek(BDA_MIDNIGHT, 1).map_or(0, |b| b[0]);
                self.set_reg16(RCX, (ticks >> 16) as u16);
                self.set_reg16(RDX, ticks as u16);
                self.set_al(midnight);
                self.store(BDA_MIDNIGHT, &[0]);
                self.set_returned_flag(CF, false);
            }
            0x01 => {
                let ticks = ((self.reg16(RCX) as u32) << 16) | self.reg16(RDX) as u32;
                self.store_u32(BDA_TICKS, ticks);
                self.set_ah(STATUS_OK);
                self.set_returned_flag(CF, false);
            }
            0x02 | 0x04 => {
                let binary = self.cmos_read(CMOS_STATUS_B) as u32 & CMOS_BINARY_MODE != 0;
                if self.ah() == 0x02 {
                    let (hours, minutes, seconds) = (
                        self.cmos_bcd(CMOS_HOURS, binary),
                        self.cmos_bcd(CMOS_MINUTES, binary),
                        self.cmos_bcd(CMOS_SECONDS, binary),
                    );
                    self.set_reg16(RCX, u16::from_le_bytes([minutes, hours]));
                    self.set_reg16(RDX, u16::from_le_bytes([0, seconds]));
                } else {
                    let (century, year, month, day) = (
                        self.cmos_bcd(CMOS_CENTURY, binary),
                        self.cmos_bcd(CMOS_YEAR, binary),
                        self.cmos_bcd(CMOS_MONTH, binary),
                        self.cmos_bcd(CMOS_DAY, binary),
                    );
                    self.set_reg16(RCX, u16::from_le_bytes([year, century]));
                    self.set_reg16(RDX, u16::from_le_bytes([day, month]));
                }
                self.set_returned_flag(CF, false);
            }
            _ => self.set_returned_flag(CF, true),
        }
    }

    // ----- disk --------------------------------------------------------------------------

    /// (cylinders, heads, sectors per track) of disk `index` as the CHS calls see it.
    fn disk_geometry(&mut self, index: usize) -> Option<(u64, u64, u64)> {
        let sectors = self.dev.lock().disk_backend(index)?.len() / SECTOR_BYTES as u64;
        let cylinders = sectors / (GEOMETRY_HEADS * GEOMETRY_SECTORS);
        Some((cylinders, GEOMETRY_HEADS, GEOMETRY_SECTORS))
    }

    /// The disk index of a BIOS drive number, if there is such a disk.
    fn bios_drive(&self, drive: u8) -> Option<usize> {
        let index = drive.checked_sub(FIRST_FIXED_DRIVE)? as usize;
        (index < self.bios.as_ref()?.drives).then_some(index)
    }

    /// Reads or writes `count` sectors from `lba` of disk `index` to or from guest memory at
    /// `linear`. Returns the BIOS status.
    fn disk_transfer(
        &mut self,
        index: usize,
        lba: u64,
        count: u64,
        linear: u64,
        write: bool,
    ) -> u8 {
        let bytes = count as usize * SECTOR_BYTES;
        let range = lba
            .checked_mul(SECTOR_BYTES as u64)
            .and_then(|start| Some((start, start.checked_add(bytes as u64)?)));
        let outgoing = match write {
            true => match self.peek(linear, bytes) {
                Some(data) => Some(data.to_vec()),
                None => return STATUS_BAD_COMMAND,
            },
            false => None,
        };
        let buffer = {
            let mut devices = self.dev.lock();
            let Some(disk) = devices.disk_backend(index) else {
                return STATUS_BAD_COMMAND;
            };
            let Some((start, end)) = range.filter(|(_, end)| *end <= disk.len()) else {
                return STATUS_SECTOR_NOT_FOUND;
            };
            match outgoing {
                Some(data) if disk.read_only() => {
                    let _ = data;
                    return STATUS_WRITE_PROTECTED;
                }
                Some(data) => {
                    return match disk.write_at(start, &data) {
                        Ok(()) => STATUS_OK,
                        Err(_) => STATUS_WRITE_FAULT,
                    };
                }
                None => {
                    let mut buffer = vec![0u8; (end - start) as usize];
                    if disk.read_at(start, &mut buffer).is_err() {
                        return STATUS_CONTROLLER_FAILURE;
                    }
                    buffer
                }
            }
        };
        match self.store(linear, &buffer) {
            true => STATUS_OK,
            false => STATUS_BAD_COMMAND,
        }
    }

    fn bios_disk(&mut self) {
        let function = self.ah();
        if function == 0x00 {
            return self.succeed();
        }
        let Some(index) = self.bios_drive(self.dl()) else {
            return self.fail(STATUS_BAD_COMMAND);
        };
        match function {
            // Status of the last operation.
            0x01 => self.succeed(),
            // CHS read and write: AL sectors, CH and CL the cylinder, CL the sector, DH the
            // head, ES:BX the buffer.
            0x02 | 0x03 => {
                let count = self.al() as u64;
                let (cx, head) = (self.reg16(RCX), self.dh() as u64);
                let sector = (cx & 0x3f) as u64;
                let cylinder = ((cx >> 8) | ((cx & 0xc0) << 2)) as u64;
                if sector == 0 || sector > GEOMETRY_SECTORS || head >= GEOMETRY_HEADS || count == 0
                {
                    return self.fail(STATUS_BAD_COMMAND);
                }
                let lba = (cylinder * GEOMETRY_HEADS + head) * GEOMETRY_SECTORS + (sector - 1);
                let linear = self.real_address(SEG_ES, self.reg16(RBX));
                match self.disk_transfer(index, lba, count, linear, function == 0x03) {
                    STATUS_OK => self.succeed(),
                    status => {
                        self.set_al(0);
                        self.fail(status);
                    }
                }
            }
            // Verify sectors, and seek: nothing to do.
            0x04 | 0x0c | 0x0d | 0x11 | 0x44 | 0x47 => self.succeed(),
            // Drive parameters.
            0x08 => {
                let (cylinders, heads, sectors) = self.disk_geometry(index).unwrap_or((0, 0, 0));
                let last_cylinder = cylinders.clamp(1, CHS_MAX_CYLINDERS) - 1;
                let cx = (((last_cylinder & 0xff) as u16) << 8)
                    | (((last_cylinder >> 2) & 0xc0) as u16)
                    | sectors as u16;
                let drives = self.bios.as_ref().map_or(0, |b| b.drives);
                self.set_reg16(RCX, cx);
                self.set_reg16(RDX, u16::from_le_bytes([drives as u8, (heads - 1) as u8]));
                self.gpr[RBX] &= !0xff;
                self.succeed();
            }
            // Disk type: a fixed disk of CX:DX sectors.
            0x15 => {
                let sectors = self.dev.lock().disk_backend(index).map_or(0, |d| d.len())
                    / SECTOR_BYTES as u64;
                self.set_reg16(RCX, (sectors >> 16) as u16);
                self.set_reg16(RDX, sectors as u16);
                self.succeed();
                self.set_ah(3);
            }
            // Extensions installation check.
            0x41 if self.reg16(RBX) == EDD_INSTALL_CHECK => {
                self.set_reg16(RBX, EDD_INSTALL_CHECK_REPLY);
                self.set_reg16(RCX, EDD_SUPPORT);
                self.succeed();
                self.set_ah(EDD_VERSION_3);
            }
            0x42 | 0x43 => self.bios_extended_transfer(index, function == 0x43),
            0x48 => self.bios_drive_parameters(index),
            _ => self.fail(STATUS_BAD_COMMAND),
        }
    }

    /// `int 13h` AH=42h/43h: the disk address packet at DS:SI says what to move.
    fn bios_extended_transfer(&mut self, index: usize, write: bool) {
        let packet = self.real_address(SEG_DS, self.reg16(RSI));
        let Some(header) = self.peek(packet, EDD_PACKET_BYTES) else {
            return self.fail(STATUS_BAD_COMMAND);
        };
        let size = header[0] as usize;
        if size < EDD_PACKET_BYTES {
            return self.fail(STATUS_BAD_COMMAND);
        }
        let count = u16::from_le_bytes([header[2], header[3]]) as u64;
        let offset = u16::from_le_bytes([header[4], header[5]]);
        let segment = u16::from_le_bytes([header[6], header[7]]);
        let lba = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let linear = if size >= EDD_PACKET_FLAT_BYTES
            && offset == EDD_FLAT_MARKER
            && segment == EDD_FLAT_MARKER
        {
            self.peek_u64(packet + EDD_PACKET_FLAT_OFFSET).unwrap_or(0)
        } else {
            ((segment as u64) << 4) + offset as u64
        };
        match self.disk_transfer(index, lba, count, linear, write) {
            STATUS_OK => {
                self.succeed();
            }
            status => {
                self.store_u16(packet + 2, 0);
                self.fail(status);
            }
        }
    }

    /// `int 13h` AH=48h: the extended drive parameters into the buffer at DS:SI.
    fn bios_drive_parameters(&mut self, index: usize) {
        let buffer = self.real_address(SEG_DS, self.reg16(RSI));
        let size = self.peek_u16(buffer).unwrap_or(0) as usize;
        if size < EDD_PARAMETERS_BYTES {
            return self.fail(STATUS_BAD_COMMAND);
        }
        let sectors =
            self.dev.lock().disk_backend(index).map_or(0, |d| d.len()) / SECTOR_BYTES as u64;
        let cylinders = (sectors / (GEOMETRY_HEADS * GEOMETRY_SECTORS)).min(EDD_MAX_CYLINDERS);
        let mut block = [0u8; EDD_PARAMETERS_BYTES];
        block[0..2].copy_from_slice(&(EDD_PARAMETERS_BYTES as u16).to_le_bytes());
        block[2..4].copy_from_slice(&EDD_GEOMETRY_VALID.to_le_bytes());
        block[4..8].copy_from_slice(&(cylinders as u32).to_le_bytes());
        block[8..12].copy_from_slice(&(GEOMETRY_HEADS as u32).to_le_bytes());
        block[12..16].copy_from_slice(&(GEOMETRY_SECTORS as u32).to_le_bytes());
        block[16..24].copy_from_slice(&sectors.to_le_bytes());
        block[24..26].copy_from_slice(&(SECTOR_BYTES as u16).to_le_bytes());
        self.store(buffer, &block);
        self.succeed();
    }

    // ----- system ------------------------------------------------------------------------

    fn bios_system(&mut self) {
        let Some(ram_top) = self.bios.as_ref().map(|b| b.ram_top) else {
            return;
        };
        // The legacy size calls cannot describe RAM above the PCI hole (16-bit KiB counts, and
        // a real-mode loader could not reach it); only E820 reports it.
        let low_ram_top = crate::ram::low_ram_end(ram_top);
        match (self.ah(), self.al()) {
            // A20 gate: always open, because RAM is flat.
            (0x24, 0x00 | 0x01) => self.succeed(),
            (0x24, 0x02) => {
                self.set_al(1);
                self.succeed();
            }
            (0x24, 0x03) => {
                self.set_reg16(RBX, A20_GATE_SUPPORT_KEYBOARD_AND_PORT_92);
                self.succeed();
            }
            // Copy extended memory: ES:SI is a GDT whose entries 2 and 3 are the source and the
            // destination, CX the number of words. The BIOS does this in protected mode; here it
            // is a copy.
            (0x87, _) => {
                let table = self.real_address(SEG_ES, self.reg16(RSI));
                let base = |bios: &Cpu, at: u64| -> Option<u64> {
                    let descriptor = bios.peek(table + at, 8)?;
                    Some(
                        descriptor[2] as u64
                            | (descriptor[3] as u64) << 8
                            | (descriptor[4] as u64) << 16
                            | (descriptor[7] as u64) << 24,
                    )
                };
                let (Some(source), Some(destination)) = (
                    base(self, COPY_DESCRIPTOR_SOURCE),
                    base(self, COPY_DESCRIPTOR_DESTINATION),
                ) else {
                    return self.fail(STATUS_BAD_COMMAND);
                };
                let length = self.reg16(RCX) as usize * 2;
                match self.peek(source, length).map(<[u8]>::to_vec) {
                    Some(data) if self.store(destination, &data) => self.succeed(),
                    _ => self.fail(STATUS_BAD_COMMAND),
                }
            }
            // Extended memory above 1 MiB, in KiB.
            (0x88, _) => {
                let kib = ((low_ram_top.saturating_sub(ONE_MIB)) / 1024).min(EXTENDED_KIB_LIMIT);
                self.set_reg16(RAX, kib as u16);
                self.set_returned_flag(CF, false);
            }
            // Delay: instant.
            (0x86, _) => self.succeed(),
            (_, _) if self.gpr[RAX] as u16 == 0xe801 => {
                let low = (low_ram_top.min(SIXTEEN_MIB).saturating_sub(ONE_MIB)) / 1024;
                let high = low_ram_top.saturating_sub(SIXTEEN_MIB) / 65536;
                self.set_reg16(RAX, low as u16);
                self.set_reg16(RCX, low as u16);
                self.set_reg16(RBX, high as u16);
                self.set_reg16(RDX, high as u16);
                self.set_returned_flag(CF, false);
            }
            (_, _) if self.gpr[RAX] as u16 == 0xe820 => self.bios_memory_map(ram_top),
            // Not supported: get configuration, APM, the Intel speed-step and target-mode probes,
            // device wait, keyboard intercept.
            (function, _) => {
                self.log_once(LOG_KEY_BIOS_UNSUPPORTED + function as u64, || {
                    format!("BIOS: int 15h function {function:#04x} is not supported")
                });
                self.fail(STATUS_NOT_SUPPORTED);
            }
        }
    }

    /// `int 15h` AX=E820h: one entry of the memory map per call, EBX continuing.
    fn bios_memory_map(&mut self, ram_top: u64) {
        let map = e820_map(ram_top);
        let index = self.gpr[RBX] as u32 as usize;
        if self.gpr[RDX] as u32 != SMAP
            || index >= map.len()
            || (self.gpr[RCX] as u32) < E820_ENTRY_BYTES as u32
        {
            return self.set_returned_flag(CF, true);
        }
        let (base, size, kind) = map[index];
        let mut entry = [0u8; E820_EXTENDED_ENTRY_BYTES as usize];
        entry[0..8].copy_from_slice(&base.to_le_bytes());
        entry[8..16].copy_from_slice(&size.to_le_bytes());
        entry[16..20].copy_from_slice(&kind.to_le_bytes());
        entry[20..24].copy_from_slice(&E820_ENTRY_ENABLED.to_le_bytes());
        let wide = self.gpr[RCX] as u32 >= E820_EXTENDED_ENTRY_BYTES;
        let length = if wide {
            E820_EXTENDED_ENTRY_BYTES as usize
        } else {
            E820_ENTRY_BYTES
        };
        let destination = self.real_address(SEG_ES, self.reg16(RDI));
        if !self.store(destination, &entry[..length]) {
            return self.set_returned_flag(CF, true);
        }
        let next = index + 1;
        self.set_reg32(RAX, SMAP);
        self.set_reg32(RCX, length as u32);
        self.set_reg32(RBX, if next < map.len() { next as u32 } else { 0 });
        self.set_returned_flag(CF, false);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::dev::DevConfig;
    use crate::disk::Disk;
    use crate::testkit::{Harness, Stop};

    const DISK_SECTORS: usize = 64;
    const PACKET: u64 = 0x7e00;
    const BUFFER: u64 = 0x8000;

    // Each snippet ends in `hlt`. Assembled with `BITS=16 hack/asm.sh`.
    /// mov ah,0x42; mov dl,0x80; mov si,0x7e00; int 0x13
    const EXTENDED_READ: [u8; 10] = [0xb4, 0x42, 0xb2, 0x80, 0xbe, 0x00, 0x7e, 0xcd, 0x13, 0xf4];
    /// mov ax,0x0201; mov cx,2; mov dx,0x80; mov bx,0x8000; int 0x13
    const CHS_READ: [u8; 15] = [
        0xb8, 0x01, 0x02, 0xb9, 0x02, 0x00, 0xba, 0x80, 0x00, 0xbb, 0x00, 0x80, 0xcd, 0x13, 0xf4,
    ];
    /// mov ah,8; mov dl,0x80; int 0x13
    const DRIVE_PARAMETERS: [u8; 7] = [0xb4, 0x08, 0xb2, 0x80, 0xcd, 0x13, 0xf4];
    /// mov ah,0x41; mov bx,0x55aa; mov dl,0x80; int 0x13
    const EDD_CHECK: [u8; 10] = [0xb4, 0x41, 0xbb, 0xaa, 0x55, 0xb2, 0x80, 0xcd, 0x13, 0xf4];
    /// mov ah,0x48; mov dl,0x80; mov si,0x7e00; int 0x13
    const EDD_PARAMETERS: [u8; 10] = [0xb4, 0x48, 0xb2, 0x80, 0xbe, 0x00, 0x7e, 0xcd, 0x13, 0xf4];
    /// mov eax,0xe820; xor ebx,ebx; mov ecx,24; mov edx,'SMAP'; mov di,0x8000; int 0x15
    const MEMORY_MAP: [u8; 27] = [
        0x66, 0xb8, 0x20, 0xe8, 0x00, 0x00, 0x66, 0x31, 0xdb, 0x66, 0xb9, 0x18, 0x00, 0x00, 0x00,
        0x66, 0xba, 0x50, 0x41, 0x4d, 0x53, 0xbf, 0x00, 0x80, 0xcd, 0x15, 0xf4,
    ];
    /// mov ax,0xe801; int 0x15
    const MEMORY_E801: [u8; 6] = [0xb8, 0x01, 0xe8, 0xcd, 0x15, 0xf4];
    /// mov ah,0x88; int 0x15
    const MEMORY_88: [u8; 5] = [0xb4, 0x88, 0xcd, 0x15, 0xf4];
    /// mov ah,0x87; mov cx,4; mov si,0x7e00; int 0x15
    const COPY_EXTENDED: [u8; 11] = [
        0xb4, 0x87, 0xb9, 0x04, 0x00, 0xbe, 0x00, 0x7e, 0xcd, 0x15, 0xf4,
    ];
    /// mov ah,0x0e; mov al,'H'; int 0x10; mov al,'i'; int 0x10
    const TELETYPE: [u8; 11] = [
        0xb4, 0x0e, 0xb0, 0x48, 0xcd, 0x10, 0xb0, 0x69, 0xcd, 0x10, 0xf4,
    ];
    /// xor ah,ah; int 0x1a
    const READ_TICKS: [u8; 5] = [0x30, 0xe4, 0xcd, 0x1a, 0xf4];
    /// mov ah,1; int 0x16
    const KEY_STATUS: [u8; 5] = [0xb4, 0x01, 0xcd, 0x16, 0xf4];
    /// The way GRUB reads keys, polling function 1 before taking the key with function 0:
    /// wait: mov ah,1; int 0x16; jz wait; mov ah,0; int 0x16
    const POLL_KEY: [u8; 11] = [
        0xb4, 0x01, 0xcd, 0x16, 0x74, 0xfa, 0xb4, 0x00, 0xcd, 0x16, 0xf4,
    ];
    /// Two keys as `POLL_KEY` reads them, the first left in BX.
    const POLL_TWO_KEYS: [u8; 22] = [
        0xb4, 0x01, 0xcd, 0x16, 0x74, 0xfa, 0xb4, 0x00, 0xcd, 0x16, 0x89, 0xc3, 0xb4, 0x01, 0xcd,
        0x16, 0x74, 0xfa, 0xb4, 0x00, 0xcd, 0x16,
    ];
    /// int 8; int 8
    const TWO_TIMER_TICKS: [u8; 5] = [0xcd, 0x08, 0xcd, 0x08, 0xf4];
    /// int 0x11; mov bx,ax; int 0x12
    const EQUIPMENT_AND_MEMORY: [u8; 7] = [0xcd, 0x11, 0x89, 0xc3, 0xcd, 0x12, 0xf4];
    /// mov ah,0xc0; int 0x15
    const UNSUPPORTED_SYSTEM_CALL: [u8; 5] = [0xb4, 0xc0, 0xcd, 0x15, 0xf4];
    /// mov ax,0x0201; mov cx,1; mov dx,0; mov bx,0x8000; int 0x13: drive 0 is a floppy
    const FLOPPY_READ: [u8; 15] = [
        0xb8, 0x01, 0x02, 0xb9, 0x01, 0x00, 0xba, 0x00, 0x00, 0xbb, 0x00, 0x80, 0xcd, 0x13, 0xf4,
    ];
    /// int 0
    const DIVIDE_ERROR: [u8; 3] = [0xcd, 0x00, 0xf4];
    /// mov ah,0x0f; int 0x10
    const VIDEO_MODE: [u8; 5] = [0xb4, 0x0f, 0xcd, 0x10, 0xf4];
    /// mov ah,2; xor bh,bh; mov dx,0x0305; int 0x10; mov ah,3; int 0x10
    const MOVE_AND_READ_CURSOR: [u8; 14] = [
        0xb4, 0x02, 0x30, 0xff, 0xba, 0x05, 0x03, 0xcd, 0x10, 0xb4, 0x03, 0xcd, 0x10, 0xf4,
    ];
    /// mov ax,0x0e41; int 0x10; cli; hlt: a boot sector that prints "A" and powers off
    const BOOT_SECTOR_PRINTS_A: [u8; 7] = [0xb8, 0x41, 0x0e, 0xcd, 0x10, 0xfa, 0xf4];

    /// Sector `n` is filled with the byte `n + 0x10`.
    fn patterned_disk() -> Vec<u8> {
        (0..DISK_SECTORS)
            .flat_map(|sector| vec![sector as u8 + 0x10; SECTOR_BYTES])
            .collect()
    }

    fn machine_with(disk: Vec<u8>, code: &[u8]) -> Harness {
        let mut harness = Harness::real_with_devices(
            code,
            DevConfig {
                disks: vec![Disk::memory(disk)],
                ..Default::default()
            },
        );
        harness.cpu.bios_install();
        harness
    }

    fn run_to_halt(harness: &mut Harness) {
        assert_eq!(
            harness.run_delivering(100_000),
            Stop::Halted,
            "{}",
            harness.cpu.dump_regs()
        );
    }

    /// Runs a snippet that loops on `int` calls: `run_delivering` restarts its budget at every
    /// interrupt, so it would never stop. Returns how the run ended within `interrupts` calls.
    /// The devices see guest time pass between calls, as the machine's event tick shows it them.
    fn run_polling(harness: &mut Harness, interrupts: usize) -> Stop {
        for _ in 0..interrupts {
            let now = harness.cpu.now_ns();
            harness.cpu.tick_devices(now);
            match harness.run(1_000) {
                Stop::Fault {
                    vector,
                    error_code,
                    software,
                } => {
                    if let Some(reason) = harness.cpu.deliver_event(vector, error_code, software) {
                        return Stop::Exit(format!("{reason:?}"));
                    }
                }
                Stop::Limit => {}
                other => return other,
            }
        }
        Stop::Limit
    }

    fn carry(harness: &Harness) -> bool {
        harness.cpu.eflags_full() & CF != 0
    }

    fn ah(harness: &Harness) -> u8 {
        (harness.cpu.gpr[RAX] >> 8) as u8
    }

    fn console_output(harness: &Harness) -> Vec<u8> {
        let mut buffer = [0u8; 256];
        let length = harness
            .console
            .read(&mut buffer, Some(Duration::from_millis(50)));
        buffer[..length].to_vec()
    }

    fn packet(count: u16, offset: u16, segment: u16, lba: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; EDD_PACKET_BYTES];
        bytes[0] = EDD_PACKET_BYTES as u8;
        bytes[2..4].copy_from_slice(&count.to_le_bytes());
        bytes[4..6].copy_from_slice(&offset.to_le_bytes());
        bytes[6..8].copy_from_slice(&segment.to_le_bytes());
        bytes[8..16].copy_from_slice(&lba.to_le_bytes());
        bytes
    }

    #[test]
    fn every_vector_points_at_a_stub_in_the_rom_segment() {
        let harness = machine_with(patterned_disk(), &[0xf4]);
        for vector in [0u64, 8, 0x10, 0x13, 0x15, 0x1a, 0x70, 0xff] {
            let entry = harness.cpu.peek_u32(vector * 4).unwrap();
            assert_eq!(entry >> 16, ROM_SEGMENT as u32, "vector {vector:#x}");
            let linear = ROM_BASE + (entry & 0xffff) as u64;
            assert_eq!(linear, ROM_BASE + STUB_OFFSET + vector * STUB_STRIDE);
            let expected: &[u8] = match vector {
                0x08 => &[OUT_IMM8_AL, TRAP_PORT as u8, INT_IMM8, 0x1c, IRET],
                v if Cpu::bios_vector_is_hardware(v) => &[OUT_IMM8_AL, TRAP_PORT as u8, IRET],
                _ => &[STI, OUT_IMM8_AL, TRAP_PORT as u8, IRET],
            };
            assert_eq!(harness.cpu.peek(linear, expected.len()).unwrap(), expected);
        }
    }

    #[test]
    fn the_data_area_describes_the_machine() {
        let harness = machine_with(patterned_disk(), &[0xf4]);
        let cpu = &harness.cpu;
        assert_eq!(cpu.peek_u16(BDA_COM1_BASE), Some(0x3f8));
        assert_eq!(cpu.peek_u16(BDA_MEMORY_KIB), Some(639));
        assert_eq!(cpu.peek_u16(BDA_EBDA_SEGMENT), Some(0x9fc0));
        assert_eq!(cpu.peek(BDA_DISK_COUNT, 1).unwrap(), [1]);
        assert_eq!(cpu.peek(BDA_VIDEO_MODE, 1).unwrap(), [3]);
        assert_eq!(cpu.peek_u16(BDA_COLUMNS), Some(80));
        assert_eq!(cpu.peek(BDA_ROWS, 1).unwrap(), [24]);
        assert_eq!(cpu.peek_u32(BDA_TICKS), Some(0));
    }

    #[test]
    fn the_equipment_and_base_memory_calls_read_the_data_area() {
        let mut harness = machine_with(patterned_disk(), &EQUIPMENT_AND_MEMORY);
        run_to_halt(&mut harness);
        assert_eq!(harness.cpu.reg16(RBX), EQUIPMENT_WORD);
        assert_eq!(harness.cpu.reg16(RAX), 639);
    }

    #[test]
    fn extended_read_fills_the_buffer_from_the_lba() {
        let mut harness = machine_with(patterned_disk(), &EXTENDED_READ);
        harness.write(PACKET, &packet(3, BUFFER as u16, 0, 5));
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(ah(&harness), STATUS_OK);
        let data = harness.read(BUFFER, 3 * SECTOR_BYTES);
        for (index, sector) in data.chunks(SECTOR_BYTES).enumerate() {
            assert!(
                sector.iter().all(|b| *b == 5 + index as u8 + 0x10),
                "sector {index}"
            );
        }
    }

    #[test]
    fn extended_read_takes_a_segmented_or_a_flat_buffer() {
        let mut harness = machine_with(patterned_disk(), &EXTENDED_READ);
        harness.write(PACKET, &packet(1, 0x10, 0x900, 2));
        run_to_halt(&mut harness);
        assert!(
            harness
                .read(0x9010, SECTOR_BYTES)
                .iter()
                .all(|b| *b == 0x12)
        );

        let mut harness = machine_with(patterned_disk(), &EXTENDED_READ);
        let mut flat = packet(1, 0xffff, 0xffff, 7);
        flat[0] = EDD_PACKET_FLAT_BYTES as u8;
        flat.extend_from_slice(&0x20_0000u64.to_le_bytes());
        harness.write(PACKET, &flat);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert!(
            harness
                .read(0x20_0000, SECTOR_BYTES)
                .iter()
                .all(|b| *b == 0x17)
        );
    }

    #[test]
    fn extended_read_past_the_end_fails_and_reports_nothing_transferred() {
        let mut harness = machine_with(patterned_disk(), &EXTENDED_READ);
        harness.write(
            PACKET,
            &packet(2, BUFFER as u16, 0, DISK_SECTORS as u64 - 1),
        );
        run_to_halt(&mut harness);
        assert!(carry(&harness));
        assert_eq!(ah(&harness), STATUS_SECTOR_NOT_FOUND);
        assert_eq!(harness.cpu.peek_u16(PACKET + 2), Some(0));
        assert!(
            harness.read(BUFFER, SECTOR_BYTES).iter().all(|b| *b == 0),
            "nothing was copied"
        );
    }

    #[test]
    fn a_packet_that_is_too_small_is_refused() {
        let mut harness = machine_with(patterned_disk(), &EXTENDED_READ);
        let mut bad = packet(1, BUFFER as u16, 0, 0);
        bad[0] = 8;
        harness.write(PACKET, &bad);
        run_to_halt(&mut harness);
        assert!(carry(&harness));
        assert_eq!(ah(&harness), STATUS_BAD_COMMAND);
    }

    #[test]
    fn chs_read_translates_through_255_heads_and_63_sectors() {
        let mut harness = machine_with(patterned_disk(), &CHS_READ);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.cpu.al(), 1, "AL is the sector count read");
        assert!(
            harness
                .read(BUFFER, SECTOR_BYTES)
                .iter()
                .all(|b| *b == 0x11),
            "CHS 0/0/2 is LBA 1"
        );
    }

    #[test]
    fn a_chs_read_of_a_floppy_or_a_missing_drive_fails() {
        let mut harness = machine_with(patterned_disk(), &FLOPPY_READ);
        run_to_halt(&mut harness);
        assert!(carry(&harness));
        assert_eq!(ah(&harness), STATUS_BAD_COMMAND);
        assert!(harness.read(BUFFER, 16).iter().all(|b| *b == 0));
    }

    #[test]
    fn drive_parameters_report_the_geometry_and_the_drive_count() {
        let mut harness = machine_with(patterned_disk(), &DRIVE_PARAMETERS);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        let cpu = &harness.cpu;
        assert_eq!(
            cpu.reg16(RDX),
            u16::from_le_bytes([1, 254]),
            "one drive, 255 heads"
        );
        assert_eq!(cpu.reg16(RCX) & 0x3f, 63, "sectors per track");
    }

    #[test]
    fn the_extensions_are_announced_and_describe_the_disk() {
        let mut harness = machine_with(patterned_disk(), &EDD_CHECK);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.cpu.reg16(RBX), EDD_INSTALL_CHECK_REPLY);
        assert_eq!(ah(&harness), EDD_VERSION_3);
        assert_eq!(harness.cpu.reg16(RCX) & 1, 1, "extended read is supported");

        let mut harness = machine_with(patterned_disk(), &EDD_PARAMETERS);
        harness.write(PACKET, &0x1eu16.to_le_bytes());
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        let block = harness.read(PACKET, EDD_PARAMETERS_BYTES);
        assert_eq!(
            u16::from_le_bytes([block[0], block[1]]),
            EDD_PARAMETERS_BYTES as u16
        );
        assert_eq!(
            u64::from_le_bytes(block[16..24].try_into().unwrap()),
            DISK_SECTORS as u64
        );
        assert_eq!(u16::from_le_bytes([block[24], block[25]]), 512);
        assert_eq!(u32::from_le_bytes(block[12..16].try_into().unwrap()), 63);
    }

    #[test]
    fn the_extensions_check_wants_the_magic_in_bx() {
        let mut harness = machine_with(
            patterned_disk(),
            &[0xb4, 0x41, 0xb2, 0x80, 0xcd, 0x13, 0xf4],
        );
        run_to_halt(&mut harness);
        assert!(carry(&harness));
    }

    #[test]
    fn the_memory_map_is_walked_with_ebx_until_it_returns_zero() {
        let mut harness = machine_with(patterned_disk(), &MEMORY_MAP);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.cpu.gpr[RAX] as u32, SMAP);
        assert_eq!(harness.cpu.gpr[RBX] as u32, 1, "more entries follow");
        assert_eq!(harness.cpu.gpr[RCX] as u32, 24);
        let entry = harness.read(BUFFER, 24);
        assert_eq!(u64::from_le_bytes(entry[0..8].try_into().unwrap()), 0);
        assert_eq!(
            u64::from_le_bytes(entry[8..16].try_into().unwrap()),
            0x9_fc00
        );
        assert_eq!(u32::from_le_bytes(entry[16..20].try_into().unwrap()), 1);

        for (index, (base, kind, next)) in [(0x9_fc00u64, 2u32, 2u32), (0x10_0000, 1, 0)]
            .into_iter()
            .enumerate()
        {
            harness.cpu.gpr[RAX] = 0xe820;
            harness.cpu.gpr[RBX] = (index + 1) as u64;
            harness.cpu.gpr[RCX] = 20;
            harness.cpu.gpr[RDX] = SMAP as u64;
            harness.cpu.gpr[RDI] = BUFFER;
            harness.cpu.halted = false;
            harness.cpu.rip = 0x7c00 + MEMORY_MAP.len() as u64 - 1 - 2;
            run_to_halt(&mut harness);
            let entry = harness.read(BUFFER, 20);
            assert_eq!(u64::from_le_bytes(entry[0..8].try_into().unwrap()), base);
            assert_eq!(u32::from_le_bytes(entry[16..20].try_into().unwrap()), kind);
            assert_eq!(harness.cpu.gpr[RBX] as u32, next);
            assert_eq!(
                harness.cpu.gpr[RCX] as u32, 20,
                "the caller's 20-byte buffer gets 20 bytes"
            );
        }
        let last = harness.read(BUFFER, 20);
        assert_eq!(
            u64::from_le_bytes(last[8..16].try_into().unwrap()),
            harness.cpu.ram.len() as u64 - ONE_MIB
        );
    }

    #[test]
    fn the_memory_map_refuses_a_call_without_smap() {
        let mut harness = machine_with(patterned_disk(), &MEMORY_MAP);
        harness.write(0x7c00 + 17, &[0, 0, 0, 0]);
        run_to_halt(&mut harness);
        assert!(carry(&harness));
    }

    #[test]
    fn the_older_memory_size_calls_report_what_is_above_one_mebibyte() {
        let mut harness = machine_with(patterned_disk(), &MEMORY_E801);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.cpu.reg16(RAX), 0x3c00, "1 MiB to 16 MiB in KiB");
        assert_eq!(harness.cpu.reg16(RCX), 0x3c00);
        let above_16_mib = (harness.cpu.ram.len() as u64 - SIXTEEN_MIB) / 65536;
        assert_eq!(
            harness.cpu.reg16(RBX) as u64,
            above_16_mib,
            "64 KiB blocks above 16 MiB"
        );
        assert_eq!(harness.cpu.reg16(RDX), harness.cpu.reg16(RBX));

        let mut harness = machine_with(patterned_disk(), &MEMORY_88);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.cpu.reg16(RAX), 0x3c00, "capped below the ISA hole");
    }

    #[test]
    fn copy_extended_memory_moves_words_between_the_descriptors_bases() {
        let mut harness = machine_with(patterned_disk(), &COPY_EXTENDED);
        let descriptor = |base: u32| {
            let mut bytes = [0u8; 8];
            bytes[0..2].copy_from_slice(&0xffffu16.to_le_bytes());
            bytes[2..4].copy_from_slice(&(base as u16).to_le_bytes());
            bytes[4] = (base >> 16) as u8;
            bytes[5] = 0x93;
            bytes[7] = (base >> 24) as u8;
            bytes
        };
        let mut table = [0u8; 0x30];
        table[0x10..0x18].copy_from_slice(&descriptor(0x1_0000));
        table[0x18..0x20].copy_from_slice(&descriptor(0x120_0000));
        harness.write(PACKET, &table);
        harness.write(0x1_0000, b"copy me!");
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        assert_eq!(harness.read(0x120_0000, 8), b"copy me!");
    }

    #[test]
    fn unknown_system_calls_fail_the_way_an_old_bios_does() {
        let mut harness = machine_with(patterned_disk(), &UNSUPPORTED_SYSTEM_CALL);
        run_to_halt(&mut harness);
        assert!(carry(&harness));
        assert_eq!(ah(&harness), STATUS_NOT_SUPPORTED);
    }

    #[test]
    fn teletype_output_reaches_the_console() {
        let mut harness = machine_with(patterned_disk(), &TELETYPE);
        run_to_halt(&mut harness);
        assert_eq!(console_output(&harness), b"Hi");
        assert_eq!(
            harness.cpu.peek_u16(BDA_CURSOR),
            Some(u16::from_le_bytes([2, 0]))
        );
    }

    #[test]
    fn the_cursor_moves_the_terminal_only_when_it_changes_place() {
        let mut harness = machine_with(patterned_disk(), &MOVE_AND_READ_CURSOR);
        run_to_halt(&mut harness);
        assert_eq!(console_output(&harness), b"\x1b[4;6H");
        assert_eq!(harness.cpu.reg16(RDX), 0x0305);
    }

    #[test]
    fn the_video_mode_is_80_column_text() {
        let mut harness = machine_with(patterned_disk(), &VIDEO_MODE);
        run_to_halt(&mut harness);
        assert_eq!(harness.cpu.reg16(RAX), u16::from_le_bytes([3, 80]));
    }

    #[test]
    fn the_timer_interrupt_advances_the_tick_counter() {
        let mut harness = machine_with(patterned_disk(), &TWO_TIMER_TICKS);
        run_to_halt(&mut harness);
        assert_eq!(harness.cpu.peek_u32(BDA_TICKS), Some(2));
    }

    #[test]
    fn every_timer_tick_calls_the_user_hook_on_int_1ch() {
        // mov word [0x70],0x7c40; mov word [0x72],0; int 8; int 8; hlt; and at 0x7c40, the hook:
        // inc word [0x500]; iret
        let main: [u8; 17] = [
            0xc7, 0x06, 0x70, 0x00, 0x40, 0x7c, 0xc7, 0x06, 0x72, 0x00, 0x00, 0x00, 0xcd, 0x08,
            0xcd, 0x08, 0xf4,
        ];
        let mut harness = machine_with(patterned_disk(), &main);
        harness.write(0x7c40, &[0xff, 0x06, 0x00, 0x05, 0xcf]);
        run_to_halt(&mut harness);
        assert_eq!(
            harness.cpu.peek_u16(0x500),
            Some(2),
            "the hook ran once per tick"
        );
        assert_eq!(harness.cpu.peek_u32(BDA_TICKS), Some(2));
    }

    #[test]
    fn the_tick_counter_wraps_at_midnight_and_the_read_clears_the_flag() {
        let mut harness = machine_with(
            patterned_disk(),
            &[0xcd, 0x08, 0x30, 0xe4, 0xcd, 0x1a, 0xf4],
        );
        harness.cpu.store_u32(BDA_TICKS, TICKS_PER_DAY - 1);
        run_to_halt(&mut harness);
        assert_eq!(harness.cpu.peek_u32(BDA_TICKS), Some(0));
        assert_eq!(harness.cpu.al(), 1, "midnight passed");
        assert_eq!(
            harness.cpu.peek(BDA_MIDNIGHT, 1).unwrap(),
            [0],
            "reading clears it"
        );
    }

    #[test]
    fn the_time_of_day_call_returns_the_tick_count() {
        let mut harness = machine_with(patterned_disk(), &READ_TICKS);
        harness.cpu.store_u32(BDA_TICKS, 0x0001_0203);
        run_to_halt(&mut harness);
        assert_eq!(harness.cpu.reg16(RCX), 1);
        assert_eq!(harness.cpu.reg16(RDX), 0x0203);
    }

    #[test]
    fn the_keyboard_has_no_key_until_the_console_types_one() {
        let mut harness = machine_with(patterned_disk(), &KEY_STATUS);
        run_to_halt(&mut harness);
        assert!(harness.cpu.eflags_full() & ZF != 0, "ZF set: no key");
    }

    #[test]
    fn an_arrow_key_from_the_terminal_reads_as_its_scan_code() {
        // The UART hands the sequence over a byte per character time (87 us of guest time),
        // so function 1 sees it arrive in pieces.
        let mut code = POLL_TWO_KEYS.to_vec();
        code.push(0xf4);
        let mut harness = machine_with(patterned_disk(), &code);
        harness.console.write(b"\x1b[Ax");
        assert_eq!(run_polling(&mut harness, 1_000_000), Stop::Halted);
        assert_eq!(
            harness.cpu.reg16(RBX),
            0x4800,
            "up arrow: scan 0x48, no character"
        );
        assert_eq!(harness.cpu.reg16(RAX), b'x' as u16);
    }

    #[test]
    fn the_escape_key_alone_still_arrives() {
        let mut harness = machine_with(patterned_disk(), &POLL_KEY);
        harness.console.write(b"\x1b");
        assert_eq!(
            run_polling(&mut harness, 10_000),
            Stop::Limit,
            "not yet: it may be a sequence"
        );
        harness.cpu.idle_skipped_ns += super::super::bios_keys::LONE_ESCAPE_TIMEOUT_NS;
        assert_eq!(run_polling(&mut harness, 100), Stop::Halted);
        assert_eq!(
            harness.cpu.reg16(RAX),
            0x011b,
            "escape: scan 1, character 0x1b"
        );
    }

    #[test]
    fn a_cpu_exception_in_real_mode_ends_the_run_instead_of_looping() {
        let mut harness = machine_with(patterned_disk(), &DIVIDE_ERROR);
        match harness.run_delivering(1000) {
            Stop::Exit(reason) => assert!(reason.contains("TripleFault"), "{reason}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_boot_sector_is_loaded_at_7c00_with_the_boot_drive_in_dl() {
        let mut disk = patterned_disk();
        disk[..BOOT_SECTOR_PRINTS_A.len()].copy_from_slice(&BOOT_SECTOR_PRINTS_A);
        disk[510..512].copy_from_slice(&BOOT_SIGNATURE);
        let mut harness = Harness::with_devices(
            &[0xf4],
            DevConfig {
                disks: vec![Disk::memory(disk)],
                ..Default::default()
            },
        );
        harness.cpu.reset_to_real_mode();
        harness.cpu.bios_install();
        harness.cpu.bios_boot();
        assert_eq!(harness.cpu.rip, 0x7c00);
        assert_eq!(harness.cpu.seg[SEG_CS].sel, 0);
        assert_eq!(harness.cpu.seg[SEG_CS].base, 0);
        assert_eq!(harness.cpu.dl(), 0x80);
        assert_eq!(harness.cpu.stack_pointer(), 0x7c00);
        assert!(harness.cpu.eflags_full() & IF != 0);
        run_to_halt(&mut harness);
        assert_eq!(console_output(&harness), b"A");
    }

    #[test]
    fn a_disk_without_the_boot_signature_is_no_bootable_device() {
        let mut harness = Harness::with_devices(
            &[0xf4],
            DevConfig {
                disks: vec![Disk::memory(patterned_disk())],
                ..Default::default()
            },
        );
        harness.cpu.reset_to_real_mode();
        harness.cpu.bios_install();
        harness.cpu.bios_boot();
        assert!(harness.cpu.halted);
        assert!(
            harness.cpu.eflags_full() & IF == 0,
            "a halt with interrupts off is a power-off"
        );
        assert_eq!(console_output(&harness), NO_BOOT_MESSAGE);
    }

    #[test]
    fn int_19h_loads_the_boot_sector_again() {
        let mut disk = patterned_disk();
        disk[..BOOT_SECTOR_PRINTS_A.len()].copy_from_slice(&BOOT_SECTOR_PRINTS_A);
        disk[510..512].copy_from_slice(&BOOT_SIGNATURE);
        // int 0x19 from somewhere else: ends up in the boot sector, which prints and halts.
        let mut harness = machine_with(disk, &[0xcd, 0x19, 0xf4]);
        run_to_halt(&mut harness);
        assert_eq!(console_output(&harness), b"A");
        assert_eq!(harness.cpu.dl(), 0x80);
    }

    #[test]
    fn writes_reach_the_disk_and_a_read_only_disk_refuses_them() {
        // mov ax,0x0301; mov cx,2; mov dx,0x80; mov bx,0x8000; int 0x13
        let write: [u8; 15] = [
            0xb8, 0x01, 0x03, 0xb9, 0x02, 0x00, 0xba, 0x80, 0x00, 0xbb, 0x00, 0x80, 0xcd, 0x13,
            0xf4,
        ];
        let mut harness = machine_with(patterned_disk(), &write);
        harness.write(BUFFER, &[0xee; SECTOR_BYTES]);
        run_to_halt(&mut harness);
        assert!(!carry(&harness));
        let disk = harness.cpu.dev.disk_backend(0).unwrap();
        let mut sector = [0u8; SECTOR_BYTES];
        disk.read_at(SECTOR_BYTES as u64, &mut sector).unwrap();
        assert!(sector.iter().all(|b| *b == 0xee));
    }

    #[test]
    fn a_machine_without_the_bios_ignores_the_trap_port() {
        let mut harness = Harness::real(&[0xb0, 0x01, 0xe6, 0xb2, 0xf4]);
        run_to_halt(&mut harness);
        assert!(harness.cpu.bios.is_none());
    }

    #[test]
    fn memory_above_one_mebibyte_is_not_wrapped_onto_low_memory() {
        // Boot loaders test the A20 gate by comparing 0000:0200 with FFFF:0210, and enable it
        // through the keyboard controller if they differ; RAM here is flat, so they must.
        // mov ax,0xffff; mov ds,ax; mov bx,0x210; mov al,[bx]
        let mut harness = Harness::real(&[
            0xb8, 0xff, 0xff, 0x8e, 0xd8, 0xbb, 0x10, 0x02, 0x8a, 0x07, 0xf4,
        ]);
        harness.write(0x200, &[0x11]);
        harness.write(0x10_0200, &[0x5a]);
        harness.run_to_halt();
        assert_eq!(harness.cpu.al(), 0x5a);
    }

    #[test]
    fn the_pic_and_the_pit_are_programmed_like_a_bios_leaves_them() {
        let harness = machine_with(patterned_disk(), &[0xf4]);
        let state = harness.cpu.dev.state(0);
        assert!(state.contains("pit"), "{state}");
    }
}
