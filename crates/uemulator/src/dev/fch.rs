//! The FCH functions Linux binds on a Turin (SP5) system: the SMBus controller at 00:14.0
//! (1022:790B) and the LPC/eSPI bridge at 00:14.3 (1022:790E), the power-management (PM)
//! register block both are configured through, and the TCO watchdog behind it.
//!
//! What Linux does with them:
//! - `i2c-piix4` binds to the SMBus function. With its revision at or above 0x51 it reads the
//!   PM block through the AcpiMmio window (0xFED80300) rather than the index/data pair at
//!   0xCD6/0xCD7, which older parts use (both work here): DecodeEn bit 4 says the SMBus decodes,
//!   PM 0x01 is the high byte of its I/O base (0xB00; the auxiliary controller is at 0xB20), and
//!   PM 0x02 bits 4:3 select the port before every transaction ("KERNCZ" port selection). No
//!   device answers on any port: every transaction ends at once with DeviceErr, which the
//!   driver reads as "no response" (-ENXIO), so `i2cdetect` finds an empty bus.
//! - `sp5100_tco` binds to the same function (revision 0x59 and up: the "efch_mmio" layout):
//!   it sets DecodeEn bit 7 (which both decodes the watchdog at 0xFEB00000 and starts its timer
//!   logic), reads ISA control bit 1 for the alternative window at AcpiMmio + 0xB00, and drives
//!   the watchdog's control and count registers.
//! - At boot every Zen kernel reads the S5 reset status at PM 0xC0 and writes it back to clear
//!   it; zero says there is nothing to report.
//!
//! The LPC bridge is config space only: no Linux 6.18 driver binds to 790E, but the bus looks
//! as it does on the hardware (`lspci`). Neither function has an interrupt pin, so the DSDT's
//! `_PRT` has no entry for slot 0x14; their requester ids are in the IVRS like every
//! function's.

use super::data_fabric::{CONFIG_SPACE_SIZE, VENDOR_AMD};
use crate::snapshot::Snapshot;

pub(crate) const FCH_SLOT: u8 = 0x14;
const DEVFN_SLOT_SHIFT: u32 = 3;
pub(crate) const SMBUS_DEVFN: u8 = FCH_SLOT << DEVFN_SLOT_SHIFT;
pub(crate) const LPC_DEVFN: u8 = SMBUS_DEVFN | 3;

/// The ids Turin's FCH reports. The SMBus revision decides how Linux reaches the PM block and
/// the watchdog (MMIO from 0x51 and 0x59 respectively); 0x71 is the Zen 4/5 FCH's.
const DEVICE_SMBUS: u16 = 0x790b;
const SMBUS_REVISION: u8 = 0x71;
const DEVICE_LPC: u16 = 0x790e;
const LPC_REVISION: u8 = 0x51;

const CFG_VENDOR_ID: usize = 0x00;
const CFG_DEVICE_ID: usize = 0x02;
const CFG_REVISION: usize = 0x08;
const CFG_SUBCLASS: usize = 0x0a;
const CFG_CLASS: usize = 0x0b;
const CFG_HEADER_TYPE: usize = 0x0e;
const CFG_SUBSYSTEM_VENDOR_ID: usize = 0x2c;
const HEADER_TYPE_MULTIFUNCTION: u8 = 0x80;
const CLASS_SERIAL_BUS: u8 = 0x0c;
const SUBCLASS_SMBUS: u8 = 0x05;
const CLASS_BRIDGE: u8 = 0x06;
const SUBCLASS_ISA_BRIDGE: u8 = 0x01;

/// The PM block through the index/data pair, and the same registers in the AcpiMmio window.
pub(crate) const PM_INDEX_PORT: u16 = 0xcd6;
pub(crate) const PM_DATA_PORT: u16 = 0xcd7;
const PM_REGISTERS: usize = 256;
pub(crate) const ACPI_MMIO_BASE: u64 = 0xfed8_0000;
const ACPI_MMIO_BYTES: u64 = 0x2000;
const ACPI_MMIO_PM: u64 = 0x300;
/// The watchdog's alternative window in AcpiMmio, decoded while ISA control has MMIOEN.
const ACPI_MMIO_WATCHDOG: u64 = 0xb00;
/// The watchdog's own window (`EFCH_PM_WDT_ADDR`), decoded while DecodeEn has WdtTmrEn.
pub(crate) const WATCHDOG_BASE: u64 = 0xfeb0_0000;
const WATCHDOG_BYTES: u64 = 8;

/// PM 0x00, DecodeEn: bit 4 SmbusAsfIoEn, bit 7 WdtTmrEn. PM 0x01 holds bits 15:8 of the SMBus
/// I/O base. PM 0x02 bits 4:3 select the SMBus port. PM 0x04, ISA control, bit 1 MMIOEN. PM
/// 0xC0, the S5 reset status, write-1-to-clear.
const PM_DECODE_ENABLE: usize = 0x00;
const DECODE_SMBUS: u8 = 1 << 4;
const DECODE_WATCHDOG: u8 = 1 << 7;
const PM_SMBUS_BASE_HIGH: usize = 0x01;
const PM_ISA_CONTROL: usize = 0x04;
const ISA_CONTROL_MMIO_ENABLE: u8 = 1 << 1;
const PM_S5_RESET_STATUS: usize = 0xc0;
const S5_RESET_STATUS_BYTES: usize = 4;
/// Firmware leaves the SMBus at 0xB00 and the AcpiMmio window decoding.
const SMBUS_BASE: u16 = 0xb00;
const PM_RESET: [(usize, u8); 3] = [
    (PM_DECODE_ENABLE, DECODE_SMBUS),
    (PM_SMBUS_BASE_HIGH, (SMBUS_BASE >> 8) as u8),
    (PM_ISA_CONTROL, ISA_CONTROL_MMIO_ENABLE),
];

/// Two host controllers: the main one at the base, the auxiliary one 0x20 above it. Each has
/// the PIIX4 registers, and the main one the I2C bus configuration byte at 0x10 (IRQ routing in
/// bit 0, a revision in 7:4: zero, SMI routing, which nothing here raises).
const SMBUS_CONTROLLERS: usize = 2;
const SMBUS_CONTROLLER_STRIDE: u16 = 0x20;
const SMBUS_PORTS: u16 = SMBUS_CONTROLLER_STRIDE * SMBUS_CONTROLLERS as u16;
const SMBUS_STATUS: u16 = 0x00;
const SMBUS_CONTROL: u16 = 0x02;
const SMBUS_COMMAND: u16 = 0x03;
const SMBUS_ADDRESS: u16 = 0x04;
const SMBUS_DATA0: u16 = 0x05;
const SMBUS_DATA1: u16 = 0x06;
const SMBUS_BLOCK_DATA: u16 = 0x07;
/// SMBSLVCNT: on SB800 and later it carries the host semaphore the SMBus shares with the
/// embedded controller (IMC): writing HostSemaphore (bit 4) takes it and reads back set,
/// ClrHostSemaphore (bit 5) gives it back. i2c-piix4 takes it around every transaction and
/// retries for seconds when it reads back clear. There is no IMC: it is always granted.
const SMBUS_SLAVE_CONTROL: u16 = 0x08;
const HOST_SEMAPHORE: u8 = 1 << 4;
const CLEAR_HOST_SEMAPHORE: u8 = 1 << 5;
/// Status: HostBusy (0), interrupt (1), DeviceErr (2), BusCollision (3), Failed (4); all but
/// HostBusy are write-1-to-clear.
const STATUS_DEVICE_ERROR: u8 = 1 << 2;
const STATUS_FAILED: u8 = 1 << 4;
const STATUS_WRITE_ONE_TO_CLEAR: u8 = 0x1e;
/// Control: Kill (1) aborts, Start (6) runs the transaction; Start reads back clear.
const CONTROL_KILL: u8 = 1 << 1;
const CONTROL_START: u8 = 1 << 6;

/// Watchdog control (offset 0): WatchDogRunStopB (0), WatchDogFired (1, write-1-to-clear),
/// WatchDogAction (2: power off when set, reset when clear), WatchDogDisable (3, always clear
/// here), WatchDogTrigger (7: restarts the count, reads clear). Count (offset 4): seconds.
const WATCHDOG_CONTROL: u64 = 0;
const WATCHDOG_COUNT: u64 = 4;
const WATCHDOG_RUNNING: u32 = 1 << 0;
const WATCHDOG_FIRED: u32 = 1 << 1;
const WATCHDOG_POWER_OFF: u32 = 1 << 2;
const WATCHDOG_TRIGGER: u32 = 1 << 7;
const WATCHDOG_CONTROL_WRITABLE: u32 = WATCHDOG_RUNNING | WATCHDOG_POWER_OFF;
const WATCHDOG_COUNT_MASK: u32 = 0xffff;
const NS_PER_SECOND: u64 = 1_000_000_000;

/// What an expired watchdog does to the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WatchdogAction {
    Reset,
    PowerOff,
}

#[derive(Clone, Copy, Default, Snapshot)]
struct SmbusController {
    status: u8,
    control: u8,
    command: u8,
    address: u8,
    data0: u8,
    data1: u8,
    block_data: u8,
    host_semaphore: bool,
}

impl SmbusController {
    fn read(&self, register: u16) -> u8 {
        match register {
            SMBUS_STATUS => self.status,
            SMBUS_CONTROL => self.control,
            SMBUS_COMMAND => self.command,
            SMBUS_ADDRESS => self.address,
            SMBUS_DATA0 => self.data0,
            SMBUS_DATA1 => self.data1,
            SMBUS_BLOCK_DATA => self.block_data,
            SMBUS_SLAVE_CONTROL if self.host_semaphore => HOST_SEMAPHORE,
            _ => 0,
        }
    }

    /// A started transaction finishes at once: no device acknowledges its address, so it ends
    /// with DeviceErr. Kill ends one with Failed (there is never one in flight to kill).
    fn write(&mut self, register: u16, value: u8) {
        match register {
            SMBUS_STATUS => self.status &= !(value & STATUS_WRITE_ONE_TO_CLEAR),
            SMBUS_CONTROL => {
                self.control = value & !CONTROL_START;
                if value & CONTROL_KILL != 0 {
                    self.status |= STATUS_FAILED;
                } else if value & CONTROL_START != 0 {
                    self.status |= STATUS_DEVICE_ERROR;
                }
            }
            SMBUS_COMMAND => self.command = value,
            SMBUS_ADDRESS => self.address = value,
            SMBUS_DATA0 => self.data0 = value,
            SMBUS_DATA1 => self.data1 = value,
            SMBUS_BLOCK_DATA => self.block_data = value,
            SMBUS_SLAVE_CONTROL if value & CLEAR_HOST_SEMAPHORE != 0 => self.host_semaphore = false,
            SMBUS_SLAVE_CONTROL if value & HOST_SEMAPHORE != 0 => self.host_semaphore = true,
            _ => {}
        }
    }
}

#[derive(Snapshot)]
pub(crate) struct Fch {
    pm: [u8; PM_REGISTERS],
    pm_index: u8,
    smbus: [SmbusController; SMBUS_CONTROLLERS],
    watchdog_control: u32,
    watchdog_count: u32,
    /// When the running watchdog expires, in device time.
    watchdog_deadline: Option<u64>,
}

impl Default for Fch {
    fn default() -> Self {
        Self::new()
    }
}

fn put(config: &mut [u8; CONFIG_SPACE_SIZE], at: usize, bytes: &[u8]) {
    config[at..at + bytes.len()].copy_from_slice(bytes);
}

pub(crate) fn is_fch_function(devfn: u8) -> bool {
    devfn == SMBUS_DEVFN || devfn == LPC_DEVFN
}

/// Config space of 00:14.0 or 00:14.3; read-only (no BARs, no interrupt pin, no capabilities).
pub(crate) fn fch_config(devfn: u8) -> [u8; CONFIG_SPACE_SIZE] {
    let mut config = [0u8; CONFIG_SPACE_SIZE];
    put(&mut config, CFG_VENDOR_ID, &VENDOR_AMD.to_le_bytes());
    put(
        &mut config,
        CFG_SUBSYSTEM_VENDOR_ID,
        &VENDOR_AMD.to_le_bytes(),
    );
    let (device, revision, class, subclass) = if devfn == SMBUS_DEVFN {
        // Function 0 announces the others.
        config[CFG_HEADER_TYPE] = HEADER_TYPE_MULTIFUNCTION;
        (
            DEVICE_SMBUS,
            SMBUS_REVISION,
            CLASS_SERIAL_BUS,
            SUBCLASS_SMBUS,
        )
    } else {
        (DEVICE_LPC, LPC_REVISION, CLASS_BRIDGE, SUBCLASS_ISA_BRIDGE)
    };
    put(&mut config, CFG_DEVICE_ID, &device.to_le_bytes());
    config[CFG_REVISION] = revision;
    config[CFG_CLASS] = class;
    config[CFG_SUBCLASS] = subclass;
    config
}

impl Fch {
    pub fn new() -> Self {
        let mut pm = [0u8; PM_REGISTERS];
        for (register, value) in PM_RESET {
            pm[register] = value;
        }
        Fch {
            pm,
            pm_index: 0,
            smbus: [SmbusController::default(); SMBUS_CONTROLLERS],
            watchdog_control: 0,
            watchdog_count: 0,
            watchdog_deadline: None,
        }
    }

    fn smbus_base(&self) -> Option<u16> {
        (self.pm[PM_DECODE_ENABLE] & DECODE_SMBUS != 0)
            .then(|| (self.pm[PM_SMBUS_BASE_HIGH] as u16) << 8)
    }

    /// The port I/O this block answers: the PM pair, and the SMBus controllers while decoded.
    pub fn decodes_port(&self, port: u16) -> bool {
        matches!(port, PM_INDEX_PORT | PM_DATA_PORT)
            || self
                .smbus_base()
                .is_some_and(|base| (base..base + SMBUS_PORTS).contains(&port))
    }

    pub fn io_read(&self, port: u16) -> u8 {
        match port {
            PM_INDEX_PORT => self.pm_index,
            PM_DATA_PORT => self.pm[self.pm_index as usize],
            _ => {
                let Some(offset) = self.smbus_base().map(|base| port - base) else {
                    return 0xff;
                };
                let controller = (offset / SMBUS_CONTROLLER_STRIDE) as usize;
                self.smbus[controller].read(offset % SMBUS_CONTROLLER_STRIDE)
            }
        }
    }

    pub fn io_write(&mut self, port: u16, value: u8) {
        match port {
            PM_INDEX_PORT => self.pm_index = value,
            PM_DATA_PORT => self.pm_write(self.pm_index as usize, value),
            _ => {
                let Some(offset) = self.smbus_base().map(|base| port - base) else {
                    return;
                };
                let controller = (offset / SMBUS_CONTROLLER_STRIDE) as usize;
                self.smbus[controller].write(offset % SMBUS_CONTROLLER_STRIDE, value);
            }
        }
    }

    /// The S5 reset status bytes clear where written ones; the rest of the block is storage.
    /// Turning WdtTmrEn off stops the watchdog with its decode.
    fn pm_write(&mut self, register: usize, value: u8) {
        if (PM_S5_RESET_STATUS..PM_S5_RESET_STATUS + S5_RESET_STATUS_BYTES).contains(&register) {
            self.pm[register] &= !value;
            return;
        }
        self.pm[register] = value;
        if register == PM_DECODE_ENABLE && value & DECODE_WATCHDOG == 0 {
            self.watchdog_deadline = None;
        }
    }

    fn watchdog_decoded(&self) -> bool {
        self.pm[PM_DECODE_ENABLE] & DECODE_WATCHDOG != 0
    }

    /// The offset of `address` in the watchdog's registers, through either window.
    fn watchdog_offset(&self, address: u64) -> Option<u64> {
        if self.watchdog_decoded()
            && (WATCHDOG_BASE..WATCHDOG_BASE + WATCHDOG_BYTES).contains(&address)
        {
            return Some(address - WATCHDOG_BASE);
        }
        let alternative = ACPI_MMIO_BASE + ACPI_MMIO_WATCHDOG;
        (self.pm[PM_ISA_CONTROL] & ISA_CONTROL_MMIO_ENABLE != 0
            && (alternative..alternative + WATCHDOG_BYTES).contains(&address))
        .then(|| address - alternative)
    }

    pub fn decodes(&self, address: u64) -> bool {
        (ACPI_MMIO_BASE..ACPI_MMIO_BASE + ACPI_MMIO_BYTES).contains(&address)
            || self.watchdog_offset(address).is_some()
    }

    /// AcpiMmio and the watchdog window, `size` bytes little-endian. The AcpiMmio blocks other
    /// than PM and the watchdog (SMI, GPIO, IOMUX, ...) read zero and ignore writes.
    pub fn mmio_read(&self, address: u64, size: u8) -> u64 {
        if let Some(offset) = self.watchdog_offset(address) {
            let register = if offset & !3 == WATCHDOG_CONTROL {
                self.watchdog_control
            } else {
                self.watchdog_count
            };
            return (register as u64 >> (8 * (offset & 3))) & size_mask(size);
        }
        let offset = address - ACPI_MMIO_BASE;
        if (ACPI_MMIO_PM..ACPI_MMIO_PM + PM_REGISTERS as u64).contains(&offset) {
            let start = (offset - ACPI_MMIO_PM) as usize;
            return (0..size as usize)
                .filter(|byte| start + byte < PM_REGISTERS)
                .fold(0, |value, byte| {
                    value | (self.pm[start + byte] as u64) << (8 * byte)
                });
        }
        0
    }

    pub fn mmio_write(&mut self, address: u64, size: u8, value: u64, now: u64) {
        if let Some(offset) = self.watchdog_offset(address) {
            self.watchdog_write(offset, size, value, now);
            return;
        }
        let offset = address - ACPI_MMIO_BASE;
        if (ACPI_MMIO_PM..ACPI_MMIO_PM + PM_REGISTERS as u64).contains(&offset) {
            let start = (offset - ACPI_MMIO_PM) as usize;
            for byte in (0..size as usize).filter(|byte| start + byte < PM_REGISTERS) {
                self.pm_write(start + byte, (value >> (8 * byte)) as u8);
            }
        }
    }

    /// Whole-register writes are what drivers make; a narrower one lands in its bytes.
    fn watchdog_write(&mut self, offset: u64, size: u8, value: u64, now: u64) {
        let shift = 8 * (offset & 3);
        let mask = (size_mask(size) << shift) as u32;
        let written = ((value << shift) as u32) & mask;
        if offset & !3 == WATCHDOG_COUNT {
            self.watchdog_count = ((self.watchdog_count & !mask) | written) & WATCHDOG_COUNT_MASK;
            return;
        }
        if written & WATCHDOG_FIRED != 0 {
            self.watchdog_control &= !WATCHDOG_FIRED;
        }
        let was_running = self.watchdog_control & WATCHDOG_RUNNING != 0;
        self.watchdog_control = (self.watchdog_control & !(mask & WATCHDOG_CONTROL_WRITABLE))
            | (written & WATCHDOG_CONTROL_WRITABLE);
        let running = self.watchdog_control & WATCHDOG_RUNNING != 0;
        if !running {
            self.watchdog_deadline = None;
        } else if !was_running || written & WATCHDOG_TRIGGER != 0 {
            self.watchdog_deadline = Some(now + self.watchdog_count as u64 * NS_PER_SECOND);
        }
    }

    pub fn next_deadline(&self) -> Option<u64> {
        self.watchdog_deadline
    }

    /// An expired watchdog: it stops, says it fired, and the machine resets or powers off.
    pub fn tick(&mut self, now: u64) -> Option<WatchdogAction> {
        let deadline = self.watchdog_deadline?;
        if now < deadline {
            return None;
        }
        self.watchdog_deadline = None;
        self.watchdog_control = (self.watchdog_control & !WATCHDOG_RUNNING) | WATCHDOG_FIRED;
        Some(if self.watchdog_control & WATCHDOG_POWER_OFF != 0 {
            WatchdogAction::PowerOff
        } else {
            WatchdogAction::Reset
        })
    }
}

fn size_mask(size: u8) -> u64 {
    match size {
        0..=7 => (1u64 << (8 * size as u32)) - 1,
        _ => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE_ERROR: u8 = STATUS_DEVICE_ERROR;
    const PM_MMIO: u64 = ACPI_MMIO_BASE + ACPI_MMIO_PM;

    fn dword(config: &[u8; CONFIG_SPACE_SIZE], at: usize) -> u32 {
        u32::from_le_bytes(config[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn smbus_and_lpc_functions_have_turins_ids_classes_and_revisions() {
        assert!(is_fch_function(0xa0) && is_fch_function(0xa3));
        assert!(!is_fch_function(0xa1) && !is_fch_function(0xa2));
        let smbus = fch_config(0xa0);
        assert_eq!(dword(&smbus, 0), 0x790b_1022);
        assert_eq!(dword(&smbus, 8), 0x0c05_0071, "SMBus, revision 0x71");
        assert_eq!(smbus[CFG_HEADER_TYPE], 0x80, "multifunction");
        let lpc = fch_config(0xa3);
        assert_eq!(dword(&lpc, 0), 0x790e_1022);
        assert_eq!(dword(&lpc, 8), 0x0601_0051, "ISA bridge");
        assert_eq!(lpc[0x3d], 0, "no interrupt pin");
    }

    /// What i2c-piix4's `piix4_setup_sb800` reads, by both routes it knows.
    #[test]
    fn the_pm_block_says_where_the_smbus_decodes_through_ports_and_mmio() {
        let mut fch = Fch::new();
        fch.io_write(PM_INDEX_PORT, 0x00);
        assert_ne!(fch.io_read(PM_DATA_PORT) & 0x10, 0, "SmbusAsfIoEn");
        fch.io_write(PM_INDEX_PORT, 0x01);
        assert_eq!(fch.io_read(PM_DATA_PORT), 0x0b);
        assert_eq!(fch.mmio_read(PM_MMIO, 1) & 0x10, 0x10);
        assert_eq!(fch.mmio_read(PM_MMIO + 1, 1), 0x0b);
        assert!(fch.decodes_port(0xb00) && fch.decodes_port(0xb20) && fch.decodes_port(0xb3f));
        assert!(!fch.decodes_port(0xb40));
    }

    /// `piix4_sb800_port_sel` with the KERNCZ register: PM 0x02 bits 4:3.
    #[test]
    fn port_selection_is_stored_in_pm_register_2() {
        let mut fch = Fch::new();
        fch.mmio_write(PM_MMIO + 2, 1, 0x18, 0);
        assert_eq!(fch.mmio_read(PM_MMIO + 2, 1), 0x18);
        fch.io_write(PM_INDEX_PORT, 0x02);
        assert_eq!(fch.io_read(PM_DATA_PORT), 0x18, "the same register");
    }

    /// `piix4_transaction`: idle status, Start, then the status says no device answered, and
    /// writing it back clears it.
    #[test]
    fn a_transaction_on_an_empty_bus_ends_with_device_error() {
        for base in [0xb00u16, 0xb20] {
            let mut fch = Fch::new();
            assert_eq!(fch.io_read(base), 0, "idle");
            fch.io_write(base + SMBUS_ADDRESS, 0x50 << 1 | 1);
            fch.io_write(base + SMBUS_CONTROL, 0x08 | CONTROL_START);
            assert_eq!(
                fch.io_read(base),
                DEVICE_ERROR,
                "{base:#x}: done, nobody acked"
            );
            assert_eq!(fch.io_read(base + SMBUS_CONTROL) & CONTROL_START, 0);
            fch.io_write(base, DEVICE_ERROR);
            assert_eq!(fch.io_read(base), 0);
            fch.io_write(base + SMBUS_CONTROL, CONTROL_KILL);
            assert_eq!(fch.io_read(base), STATUS_FAILED);
        }
    }

    /// `piix4_access_sb800`: take the semaphore, see it taken, release it.
    #[test]
    fn the_host_semaphore_is_granted_and_released() {
        let mut fch = Fch::new();
        let register = 0xb00 + SMBUS_SLAVE_CONTROL;
        assert_eq!(fch.io_read(register), 0);
        fch.io_write(register, HOST_SEMAPHORE);
        assert_eq!(fch.io_read(register) & HOST_SEMAPHORE, HOST_SEMAPHORE);
        fch.io_write(register, HOST_SEMAPHORE | CLEAR_HOST_SEMAPHORE);
        assert_eq!(fch.io_read(register), 0);
    }

    #[test]
    fn the_s5_reset_status_reads_zero_and_clears_where_written() {
        let mut fch = Fch::new();
        assert_eq!(fch.mmio_read(PM_MMIO + 0xc0, 4), 0);
        fch.pm[PM_S5_RESET_STATUS] = 0x11;
        fch.mmio_write(PM_MMIO + 0xc0, 4, 0x01, 0);
        assert_eq!(fch.mmio_read(PM_MMIO + 0xc0, 4), 0x10);
    }

    /// sp5100_tco's efch_mmio setup: WdtTmrEn decodes the watchdog, ISA control MMIOEN the
    /// alternative window, the timer starts, triggers reload it and expiry resets.
    #[test]
    fn the_watchdog_decodes_counts_down_and_fires() {
        let mut fch = Fch::new();
        assert!(!fch.decodes(WATCHDOG_BASE));
        fch.mmio_write(PM_MMIO, 1, 0x10 | DECODE_WATCHDOG as u64, 0);
        assert!(fch.decodes(WATCHDOG_BASE));
        assert_ne!(fch.mmio_read(PM_MMIO + 4, 1) & 2, 0, "MMIOEN");
        assert_eq!(fch.mmio_read(WATCHDOG_BASE, 4) & 8, 0, "not disabled");
        fch.mmio_write(WATCHDOG_BASE + 4, 4, 60, 0);
        fch.mmio_write(WATCHDOG_BASE, 4, WATCHDOG_RUNNING as u64, 0);
        assert_eq!(fch.next_deadline(), Some(60 * NS_PER_SECOND));
        // a ping 30 s in through the alternative window
        let alternative = ACPI_MMIO_BASE + ACPI_MMIO_WATCHDOG;
        fch.mmio_write(
            alternative,
            4,
            (WATCHDOG_RUNNING | WATCHDOG_TRIGGER) as u64,
            30 * NS_PER_SECOND,
        );
        assert_eq!(fch.tick(80 * NS_PER_SECOND), None);
        assert_eq!(fch.tick(90 * NS_PER_SECOND), Some(WatchdogAction::Reset));
        assert_ne!(fch.mmio_read(WATCHDOG_BASE, 4) as u32 & WATCHDOG_FIRED, 0);
        fch.mmio_write(WATCHDOG_BASE, 4, WATCHDOG_FIRED as u64, 0);
        assert_eq!(fch.mmio_read(WATCHDOG_BASE, 4) as u32 & WATCHDOG_FIRED, 0);
        // stopped, nothing fires; with the action bit set it powers off
        fch.mmio_write(WATCHDOG_BASE, 4, WATCHDOG_POWER_OFF as u64, 0);
        assert_eq!(fch.next_deadline(), None);
        fch.mmio_write(
            WATCHDOG_BASE,
            4,
            (WATCHDOG_RUNNING | WATCHDOG_POWER_OFF) as u64,
            100 * NS_PER_SECOND,
        );
        assert_eq!(
            fch.tick(160 * NS_PER_SECOND),
            Some(WatchdogAction::PowerOff)
        );
    }
}
