//! Interrupt remapping (spec 2.2.5): a device's MSI, or an IOAPIC redirection entry presented
//! as one, goes through the device table entry of its requester id and, if the entry says so,
//! through a table of IRTEs in guest RAM that decides vector, destination and delivery mode.
//!
//! The LAPIC/IOAPIC code calls `Iommu::remap_interrupt` for every message a device sends; the
//! result is what the interrupt controller should deliver. With the IOMMU off nothing changes.
use super::events::FLAG_INTERRUPT;
use super::tables::{DeviceEntryError, DeviceTableEntry, InterruptControl};
use super::{
    CONTROL_GUEST_APIC_ENABLE, INTERRUPT_WINDOW_FIRST, INTERRUPT_WINDOW_LAST, Iommu, PhysRead,
    read_u64,
};

/// MSI address (SDM vol. 3 10.11.1): destination ID in bits 19:12, redirection hint in bit 3,
/// destination mode (1 = logical) in bit 2.
const MSI_ADDRESS_DESTINATION_SHIFT: u32 = 12;
const MSI_ADDRESS_DESTINATION_MASK: u64 = 0xff;
const MSI_ADDRESS_REDIRECTION_HINT: u64 = 1 << 3;
const MSI_ADDRESS_LOGICAL_DESTINATION: u64 = 1 << 2;
/// MSI data: vector in 7:0, delivery mode in 10:8, level assert in 14, trigger mode in 15
/// (1 = level).
const MSI_DATA_VECTOR_MASK: u32 = 0xff;
const MSI_DATA_DELIVERY_SHIFT: u32 = 8;
const MSI_DATA_DELIVERY_MASK: u32 = 0x7;
const MSI_DATA_LEVEL_TRIGGERED: u32 = 1 << 15;
/// The IRTE index is the low 11 bits of the data, which is why a table holds at most 2048
/// entries (IntTabLen 11).
const IRTE_INDEX_MASK: u32 = 0x7ff;
const INTERRUPT_TABLE_MAX_LENGTH: u64 = 11;

const IRTE_LEGACY_BYTES: u64 = 4;
const IRTE_GUEST_APIC_BYTES: u64 = 16;
/// Legacy 32-bit IRTE: RemapEn, SuppIOPF, IntType (4:2), RqEoi, DM, GuestMode, Destination
/// (15:8), Vector (23:16).
const IRTE_REMAP_ENABLE: u64 = 1 << 0;
const IRTE_SUPPRESS_FAULT: u64 = 1 << 1;
const IRTE_INT_TYPE_SHIFT: u32 = 2;
const IRTE_INT_TYPE_MASK: u64 = 0x7;
const IRTE_LOGICAL_DESTINATION: u64 = 1 << 6;
const IRTE_GUEST_MODE: u64 = 1 << 7;
const IRTE_DESTINATION_SHIFT: u32 = 8;
const IRTE_LEGACY_DESTINATION_MASK: u64 = 0xff;
const IRTE_LEGACY_VECTOR_SHIFT: u32 = 16;
/// The 128-bit IRTE (guest APIC mode) has the same flags in its low qword, a 24-bit destination
/// at 31:8 and the vector in the low byte of the high qword.
const IRTE_GUEST_APIC_DESTINATION_MASK: u64 = 0xff_ffff;
/// With XTEn the destination's top byte is in high-qword bits 63:56
/// (`irte_ga_hi` in Linux and `AMDVI_IRTE_GA_HI.DESTINATION_HI` in QEMU).
const IRTE_XT_DESTINATION_HIGH_SHIFT: u32 = 56;
const IRTE_INT_TYPE_FIXED: u64 = 0;
const IRTE_INT_TYPE_ARBITRATED: u64 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DeliveryMode {
    Fixed,
    LowestPriority,
    Smi,
    Nmi,
    Init,
    ExtInt,
}

impl DeliveryMode {
    pub(crate) fn from_msi(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::Fixed),
            1 => Some(Self::LowestPriority),
            2 => Some(Self::Smi),
            4 => Some(Self::Nmi),
            5 => Some(Self::Init),
            7 => Some(Self::ExtInt),
            _ => None,
        }
    }

    pub(crate) fn msi_bits(self) -> u32 {
        match self {
            Self::Fixed => 0,
            Self::LowestPriority => 1,
            Self::Smi => 2,
            Self::Nmi => 4,
            Self::Init => 5,
            Self::ExtInt => 7,
        }
    }
}

/// A device interrupt as it arrives at the IOMMU: the requester that sent it and the MSI
/// address and data. An IOAPIC's redirection entry is turned into one by `ioapic_request`,
/// with the requester id from the IVRS "special device" entry for that IOAPIC.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct InterruptRequest {
    pub requester: u16,
    pub address: u64,
    pub data: u32,
}

/// What the interrupt controller delivers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct RemappedInterrupt {
    pub vector: u8,
    pub delivery_mode: DeliveryMode,
    pub logical_destination: bool,
    pub redirection_hint: bool,
    pub level_triggered: bool,
    pub destination: u32,
}

impl RemappedInterrupt {
    /// The message the local APIC accepts.
    pub(crate) fn into_interrupt(self) -> crate::dev::apic::Interrupt {
        crate::dev::apic::Interrupt {
            vector: self.vector,
            delivery_mode: self.delivery_mode.msi_bits(),
            level_triggered: self.level_triggered,
            destination: self.destination,
            logical: self.logical_destination,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum InterruptFault {
    /// The message address is outside 0xFEExxxxx.
    InvalidAddress,
    /// The device table entry is unreadable or the requester has none.
    DeviceTable,
    /// IntCtl, a pass-through bit or the IRTE's RemapEn says drop it.
    TargetAbort,
    /// The IRTE could not be read, the index is beyond the table, or the table length is
    /// illegal.
    InterruptTable,
    /// Reserved or unsupported (SMI, guest mode) type.
    Unsupported,
}

/// A fault plus whether it is worth an event: SMI, reserved types and guest mode are the
/// platform's limits, not a device misbehaving, and a suppressed IRTE fault must stay quiet.
struct Rejected {
    fault: InterruptFault,
    log: bool,
}

impl From<InterruptFault> for Rejected {
    fn from(fault: InterruptFault) -> Self {
        Rejected {
            log: fault != InterruptFault::Unsupported,
            fault,
        }
    }
}

/// The MSI an IOAPIC redirection entry stands for: its vector and delivery mode become the
/// message data, its destination the address. With remapping on, the entry's vector field holds
/// the IRTE index instead (Linux programs it that way) and the delivery mode bits extend it.
pub(crate) fn ioapic_request(
    requester: u16,
    vector: u8,
    delivery_mode: DeliveryMode,
    logical_destination: bool,
    level_triggered: bool,
    destination: u8,
) -> InterruptRequest {
    ioapic_request_raw(
        requester,
        vector,
        delivery_mode.msi_bits(),
        logical_destination,
        level_triggered,
        destination,
    )
}

/// `ioapic_request` with the delivery mode as the entry's three bits. With remapping on those
/// bits are the top of the IRTE index, so a value that is no valid mode (3, 6) still has to
/// reach the IOMMU.
pub(crate) fn ioapic_request_raw(
    requester: u16,
    vector: u8,
    delivery_mode_bits: u32,
    logical_destination: bool,
    level_triggered: bool,
    destination: u8,
) -> InterruptRequest {
    let mut address =
        INTERRUPT_WINDOW_FIRST | (destination as u64) << MSI_ADDRESS_DESTINATION_SHIFT;
    if logical_destination {
        address |= MSI_ADDRESS_LOGICAL_DESTINATION;
    }
    let mut data = vector as u32 | delivery_mode_bits << MSI_DATA_DELIVERY_SHIFT;
    if level_triggered {
        data |= MSI_DATA_LEVEL_TRIGGERED;
    }
    InterruptRequest {
        requester,
        address,
        data,
    }
}

/// An unremapped message read as the MSI it is.
pub(crate) fn decode_message(
    request: &InterruptRequest,
) -> Result<RemappedInterrupt, InterruptFault> {
    let mode_bits = (request.data >> MSI_DATA_DELIVERY_SHIFT) & MSI_DATA_DELIVERY_MASK;
    let delivery_mode = DeliveryMode::from_msi(mode_bits).ok_or(InterruptFault::Unsupported)?;
    Ok(RemappedInterrupt {
        vector: (request.data & MSI_DATA_VECTOR_MASK) as u8,
        delivery_mode,
        logical_destination: request.address & MSI_ADDRESS_LOGICAL_DESTINATION != 0,
        redirection_hint: request.address & MSI_ADDRESS_REDIRECTION_HINT != 0,
        level_triggered: request.data & MSI_DATA_LEVEL_TRIGGERED != 0,
        destination: ((request.address >> MSI_ADDRESS_DESTINATION_SHIFT)
            & MSI_ADDRESS_DESTINATION_MASK) as u32,
    })
}

impl Iommu {
    /// Decides what a device interrupt becomes. `mem` is read for the device table entry and the
    /// IRTE. Faults queue an IO_PAGE_FAULT event with the interrupt flag and the message address
    /// (unless the IRTE asked for them to be suppressed).
    ///
    /// Pass-through cases (IOMMU off, no device table, V=0, IV=0) return the message as sent. A
    /// device with IV=1 sends fixed and arbitrated interrupts through its IntCtl (abort, forward
    /// unchanged, or remap through the IRTE) and NMI, INIT and ExtInt only when the matching
    /// DTE pass bit is set; LINT0/LINT1 pass also admit ExtInt and NMI, the types those pins
    /// carry. SMI is not supported.
    pub fn remap_interrupt(
        &mut self,
        mem: &dyn PhysRead,
        request: InterruptRequest,
    ) -> Result<RemappedInterrupt, InterruptFault> {
        if !self.enabled() || self.device_table_bytes() == 0 {
            return decode_message(&request);
        }
        let entry = match self.fetch_device_entry(mem, request.requester) {
            Ok(entry) => entry,
            Err(DeviceEntryError::OutsideTable) => {
                self.log_interrupt_fault(&request, 0);
                return Err(InterruptFault::DeviceTable);
            }
            Err(DeviceEntryError::Unreadable(address)) => {
                self.log_device_table_error(request.requester, address, FLAG_INTERRUPT);
                return Err(InterruptFault::DeviceTable);
            }
        };
        // Only IV decides, not V: Linux leaves V clear in the entries of the I/O APIC and HPET
        // special devices (nothing translates their DMA) and still has their interrupts remapped.
        if !entry.interrupt_map_valid() {
            return decode_message(&request);
        }
        match self.remap_with_entry(mem, &request, &entry) {
            Ok(remapped) => Ok(remapped),
            Err(rejected) => {
                if rejected.log {
                    self.log_interrupt_fault(&request, entry.domain_id());
                }
                Err(rejected.fault)
            }
        }
    }

    fn log_interrupt_fault(&mut self, request: &InterruptRequest, domain: u16) {
        self.log_page_fault(request.requester, domain, request.address, FLAG_INTERRUPT);
    }

    fn remap_with_entry(
        &mut self,
        mem: &dyn PhysRead,
        request: &InterruptRequest,
        entry: &DeviceTableEntry,
    ) -> Result<RemappedInterrupt, Rejected> {
        if !(INTERRUPT_WINDOW_FIRST..=INTERRUPT_WINDOW_LAST).contains(&request.address) {
            return Err(InterruptFault::InvalidAddress.into());
        }
        let message = decode_message(request)?;
        let pass = |allowed: bool| {
            if allowed {
                Ok(message)
            } else {
                Err(InterruptFault::TargetAbort.into())
            }
        };
        match message.delivery_mode {
            DeliveryMode::Fixed | DeliveryMode::LowestPriority => match entry.interrupt_control() {
                InterruptControl::Forward => Ok(message),
                InterruptControl::Abort => Err(InterruptFault::TargetAbort.into()),
                InterruptControl::Reserved => Err(InterruptFault::Unsupported.into()),
                InterruptControl::Remap => self.remap_through_table(mem, request, entry, message),
            },
            DeliveryMode::Nmi => pass(entry.nmi_pass() || entry.lint1_pass()),
            DeliveryMode::Init => pass(entry.init_pass()),
            DeliveryMode::ExtInt => pass(entry.extint_pass() || entry.lint0_pass()),
            DeliveryMode::Smi => Err(InterruptFault::Unsupported.into()),
        }
    }

    fn remap_through_table(
        &mut self,
        mem: &dyn PhysRead,
        request: &InterruptRequest,
        entry: &DeviceTableEntry,
        message: RemappedInterrupt,
    ) -> Result<RemappedInterrupt, Rejected> {
        let length = entry.interrupt_table_length();
        if length == 0 || length > INTERRUPT_TABLE_MAX_LENGTH {
            return Err(InterruptFault::InterruptTable.into());
        }
        let index = (request.data & IRTE_INDEX_MASK) as u64;
        if index >= 1 << length {
            return Err(InterruptFault::InterruptTable.into());
        }
        let guest_apic = self.control() & CONTROL_GUEST_APIC_ENABLE != 0;
        let entry_bytes = if guest_apic {
            IRTE_GUEST_APIC_BYTES
        } else {
            IRTE_LEGACY_BYTES
        };
        let address = entry.interrupt_table_root() + index * entry_bytes;
        let (flags, destination, vector) = if guest_apic {
            let low = read_u64(mem, address).ok_or(InterruptFault::InterruptTable)?;
            let high = read_u64(mem, address + 8).ok_or(InterruptFault::InterruptTable)?;
            let mut destination =
                (low >> IRTE_DESTINATION_SHIFT) & IRTE_GUEST_APIC_DESTINATION_MASK;
            if self.xt_enabled() {
                destination |= ((high >> IRTE_XT_DESTINATION_HIGH_SHIFT) & 0xff) << 24;
            }
            (low, destination, high as u8)
        } else {
            let mut bytes = [0u8; 4];
            if !mem.read_phys(address, &mut bytes) {
                return Err(InterruptFault::InterruptTable.into());
            }
            let raw = u32::from_le_bytes(bytes) as u64;
            let destination = (raw >> IRTE_DESTINATION_SHIFT) & IRTE_LEGACY_DESTINATION_MASK;
            (raw, destination, (raw >> IRTE_LEGACY_VECTOR_SHIFT) as u8)
        };
        if flags & IRTE_REMAP_ENABLE == 0 {
            // SuppIOPF: the interrupt is dropped without an event.
            return Err(Rejected {
                fault: InterruptFault::TargetAbort,
                log: flags & IRTE_SUPPRESS_FAULT == 0,
            });
        }
        if flags & IRTE_GUEST_MODE != 0 {
            return Err(InterruptFault::Unsupported.into());
        }
        let delivery_mode = match (flags >> IRTE_INT_TYPE_SHIFT) & IRTE_INT_TYPE_MASK {
            IRTE_INT_TYPE_FIXED => DeliveryMode::Fixed,
            IRTE_INT_TYPE_ARBITRATED => DeliveryMode::LowestPriority,
            _ => return Err(InterruptFault::Unsupported.into()),
        };
        Ok(RemappedInterrupt {
            vector,
            delivery_mode,
            logical_destination: flags & IRTE_LOGICAL_DESTINATION != 0,
            // RqEoi is not a redirection hint (QEMU maps it to one; the lowest-priority intent
            // is already in the IRTE's type).
            redirection_hint: false,
            level_triggered: message.level_triggered,
            destination: destination as u32,
        })
    }
}
