//! A TPM 2.0 behind the Command Response Buffer interface (TCG PC Client Platform TPM Profile,
//! "CRB Interface"), at the address QEMU's `tpm-crb` uses. The guest finds it through the ACPI
//! `TPM2` table (start method 7, CRB) and the `MSFT0101` device in the DSDT; the UEFI firmware
//! drives it directly and Linux's `tpm_crb` takes over afterwards. QEMU's `hw/tpm/tpm_crb.c` is
//! the behavioral reference.
//!
//! The command runs synchronously inside the write that sets `CTRL_START`, so `START` reads back
//! clear by the time the guest polls it and no interrupt is needed (the TPM2 table says the
//! interface has none). Only locality 0 exists.
#[cfg(freestanding)]
use crate::prelude::*;

mod capability;
mod context;
mod engine;
mod hash;
mod hierarchy;
#[cfg(test)]
mod key_tests;
mod nv;
mod object;
mod session;
mod spec;
#[cfg(test)]
mod tests;
mod wire;

use crate::dev::RngSource;
use crate::dev::kv::SharedKeyValueStore;
pub(crate) use engine::Random;
use engine::Tpm;
pub(crate) use hierarchy::STORE_PREFIX;

/// What a machine's TPM is made from: where its randomness comes from, and the store that holds
/// what it keeps across resets (the machine's key-value store, or one of its own that lives as
/// long as the machine when the embedder gave none).
#[derive(Clone)]
pub(crate) struct TpmConfig {
    pub random: RngSource,
    pub store: SharedKeyValueStore,
}

pub(crate) const TPM_CRB_BASE: u64 = 0xfed4_0000;
/// One 4 KiB page: locality 0's registers, the control area and the data buffer.
pub(crate) const TPM_CRB_SIZE: u64 = 0x1000;

const LOC_STATE: u64 = 0x00;
const LOC_CTRL: u64 = 0x08;
const LOC_STS: u64 = 0x0c;
const INTF_ID: u64 = 0x30;
const INTF_ID_HIGH: u64 = 0x34;
const CTRL_EXT: u64 = 0x38;
/// The control area; `CTRL_REQ` is its first register, and the address the TPM2 table gives.
pub(crate) const CTRL_REQ: u64 = 0x40;
const CTRL_STS: u64 = 0x44;
const CTRL_CANCEL: u64 = 0x48;
const CTRL_START: u64 = 0x4c;
const CTRL_INT_ENABLE: u64 = 0x50;
const CTRL_INT_STS: u64 = 0x54;
const CTRL_CMD_SIZE: u64 = 0x58;
const CTRL_CMD_LADDR: u64 = 0x5c;
const CTRL_CMD_HADDR: u64 = 0x60;
const CTRL_RSP_SIZE: u64 = 0x64;
/// 64 bits, at 0x68..0x70.
const CTRL_RSP_ADDR: u64 = 0x68;
const CTRL_RSP_ADDR_HIGH: u64 = 0x6c;
const DATA_BUFFER: u64 = 0x80;
/// Commands and responses share the rest of the page.
pub(crate) const DATA_BUFFER_BYTES: usize = (TPM_CRB_SIZE - DATA_BUFFER) as usize;

/// `TPM_LOC_STATE_x`: bit 0 reads 1 while no dynamic launch has happened (tpmEstablished is
/// active low), and the registers are always valid.
const LOC_STATE_ESTABLISHED_CLEAR: u32 = 1 << 0;
const LOC_STATE_ASSIGNED: u32 = 1 << 1;
const LOC_STATE_REGISTERS_VALID: u32 = 1 << 7;
const LOC_CTRL_REQUEST_ACCESS: u32 = 1 << 0;
const LOC_CTRL_RELINQUISH: u32 = 1 << 1;
const LOC_STS_GRANTED: u32 = 1 << 0;
const CTRL_REQ_CMD_READY: u32 = 1 << 0;
const CTRL_REQ_GO_IDLE: u32 = 1 << 1;
const CTRL_STS_IDLE: u32 = 1 << 1;
const CTRL_START_INVOKE: u32 = 1 << 0;

/// `TPM_CRB_INTF_ID`: a CRB interface (type 1, version 1), locality 0 only, 64-byte
/// transfers, CRB supported and selected, the selection locked.
const INTF_TYPE_CRB_ACTIVE: u32 = 0b1;
const INTF_VERSION_CRB: u32 = 0b1 << 4;
const INTF_CAP_XFER_SIZE_64: u32 = 0b11 << 11;
const INTF_CAP_CRB: u32 = 1 << 14;
const INTF_SELECTOR_CRB: u32 = 0b1 << 17;
const INTF_SELECTOR_LOCKED: u32 = 1 << 19;
const INTERFACE_ID: u32 = INTF_TYPE_CRB_ACTIVE
    | INTF_VERSION_CRB
    | INTF_CAP_XFER_SIZE_64
    | INTF_CAP_CRB
    | INTF_SELECTOR_CRB
    | INTF_SELECTOR_LOCKED;
/// Vendor and device id in the high dword; QEMU reports Red Hat's PCI vendor and device 1.
/// Nothing in Linux or the firmware reads them.
const INTERFACE_ID_HIGH: u32 = 0x0001_1af4;

pub(crate) struct TpmCrb {
    tpm: Tpm,
    locality_assigned: bool,
    idle: bool,
    /// `CTRL_CANCEL`, `CTRL_INT_ENABLE` and `CTRL_INT_STS` keep what is written; nothing acts on
    /// them, since commands finish before the guest could cancel or wait for an interrupt.
    cancel: u32,
    interrupt_enable: u32,
    data: Vec<u8>,
}

impl TpmCrb {
    pub fn new(config: TpmConfig, now_ns: u64) -> Self {
        TpmCrb {
            tpm: Tpm::new(Random::new(config.random), now_ns, config.store),
            locality_assigned: false,
            idle: true,
            cancel: 0,
            interrupt_enable: 0,
            data: vec![0; DATA_BUFFER_BYTES],
        }
    }

    pub fn decodes(phys_addr: u64) -> bool {
        (TPM_CRB_BASE..TPM_CRB_BASE + TPM_CRB_SIZE).contains(&phys_addr)
    }

    fn register(&self, offset: u64) -> u32 {
        let buffer_address = TPM_CRB_BASE + DATA_BUFFER;
        match offset {
            LOC_STATE => {
                LOC_STATE_ESTABLISHED_CLEAR
                    | LOC_STATE_REGISTERS_VALID
                    | if self.locality_assigned {
                        LOC_STATE_ASSIGNED
                    } else {
                        0
                    }
            }
            LOC_STS => self.locality_assigned as u32 * LOC_STS_GRANTED,
            INTF_ID => INTERFACE_ID,
            INTF_ID_HIGH => INTERFACE_ID_HIGH,
            CTRL_STS => self.idle as u32 * CTRL_STS_IDLE,
            CTRL_CANCEL => self.cancel,
            CTRL_INT_ENABLE => self.interrupt_enable,
            CTRL_CMD_SIZE | CTRL_RSP_SIZE => DATA_BUFFER_BYTES as u32,
            CTRL_CMD_LADDR | CTRL_RSP_ADDR => buffer_address as u32,
            CTRL_CMD_HADDR | CTRL_RSP_ADDR_HIGH => (buffer_address >> 32) as u32,
            // LOC_CTRL, CTRL_EXT, CTRL_REQ (requests complete at once), CTRL_START (likewise)
            // and CTRL_INT_STS read as zero.
            _ => 0,
        }
    }

    pub fn mmio_read(&self, phys_addr: u64, size: u8) -> u64 {
        let offset = phys_addr - TPM_CRB_BASE;
        let size = size.clamp(1, 8) as usize;
        if offset >= DATA_BUFFER {
            let start = (offset - DATA_BUFFER) as usize;
            let mut bytes = [0u8; 8];
            for (index, byte) in bytes.iter_mut().enumerate().take(size) {
                *byte = self.data.get(start + index).copied().unwrap_or(0);
            }
            return u64::from_le_bytes(bytes);
        }
        // Registers are dwords; a wider or unaligned access is put together byte by byte.
        let mut value = 0u64;
        for index in 0..size as u64 {
            let byte_offset = offset + index;
            let byte = self.register(byte_offset & !3) >> (8 * (byte_offset & 3));
            value |= ((byte & 0xff) as u64) << (8 * index);
        }
        value
    }

    pub fn mmio_write(&mut self, phys_addr: u64, size: u8, value: u64, now_ns: u64) {
        let offset = phys_addr - TPM_CRB_BASE;
        let size = size.clamp(1, 8) as usize;
        if offset >= DATA_BUFFER {
            let start = (offset - DATA_BUFFER) as usize;
            for (index, byte) in value.to_le_bytes().iter().enumerate().take(size) {
                if let Some(slot) = self.data.get_mut(start + index) {
                    *slot = *byte;
                }
            }
            return;
        }
        let value = value as u32;
        match offset {
            LOC_CTRL => {
                if value & LOC_CTRL_REQUEST_ACCESS != 0 {
                    self.locality_assigned = true;
                }
                if value & LOC_CTRL_RELINQUISH != 0 {
                    self.locality_assigned = false;
                }
            }
            CTRL_REQ => {
                if value & CTRL_REQ_CMD_READY != 0 {
                    self.idle = false;
                }
                if value & CTRL_REQ_GO_IDLE != 0 {
                    self.idle = true;
                }
            }
            CTRL_CANCEL => self.cancel = value,
            CTRL_INT_ENABLE => self.interrupt_enable = value,
            // Write-one-to-clear, and nothing is ever pending.
            CTRL_INT_STS | CTRL_EXT => {}
            CTRL_START if value & CTRL_START_INVOKE != 0 => self.start(now_ns),
            _ => {}
        }
    }

    /// Runs the command in the data buffer and leaves the response there. QEMU's CRB runs a
    /// command whether or not the guest asked for a locality first or left idle, and so does
    /// this one: Linux skips the locality request on some interfaces.
    fn start(&mut self, now_ns: u64) {
        const HEADER_SIZE_OFFSET: usize = 2;
        let declared = u32::from_be_bytes(
            self.data[HEADER_SIZE_OFFSET..HEADER_SIZE_OFFSET + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        // A size past the buffer is passed through clipped; the engine then finds that the
        // header's size disagrees and answers TPM_RC_COMMAND_SIZE.
        let command = self.data[..declared.min(DATA_BUFFER_BYTES)].to_vec();
        let response = self.tpm.execute(&command, now_ns);
        let length = response.len().min(DATA_BUFFER_BYTES);
        self.data[..length].copy_from_slice(&response[..length]);
        self.idle = false;
    }
}
