//! AMD IOMMU (AMD-Vi, "AMD I/O Virtualization Technology (IOMMU) Specification" rev 3.x) as seen
//! by Linux and Windows: one PCI function at 00:00.2 with the IOMMU capability and an MSI
//! capability, a 16 KiB MMIO register window, a command buffer and an event log in guest RAM, a
//! device table indexed by requester id, 1 to 6 level I/O page tables and interrupt remapping.
//!
//! The model follows QEMU's `hw/i386/amd_iommu.c` where the spec is silent, and the Linux driver
//! (`drivers/iommu/amd`) as the consumer that has to keep working.
//!
//! Layout:
//! - `mod.rs`: state, PCI function, MMIO registers, the DMA translation entry point.
//! - `commands.rs`: command buffer execution. `events.rs`: the event log.
//! - `tables.rs`: device table entries and the page walk. `irq.rs`: interrupt remapping.
//! - `ivrs.rs`: the ACPI IVRS table that tells an OS this unit exists.
//!
//! Nothing in here touches RAM on its own: every entry point gets the memory it may use. Faults
//! found while a device is doing DMA are queued in `pending_events` and reach the event log in
//! `service`, because `GuestMem::read` is `&self` and cannot write the log.
// Interrupt remapping, the IVRS builder and the MSI accessors are the interface for the LAPIC,
// IOAPIC and ACPI code, none of which exists yet; only tests call them for now. Remove this
// allow when they are wired in.
#![allow(dead_code, unused_imports)]

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;

mod commands;
mod events;
mod irq;
mod ivrs;
mod tables;
#[cfg(test)]
mod tests;

pub(crate) use irq::{
    DeliveryMode, InterruptFault, InterruptRequest, RemappedInterrupt, decode_message,
    ioapic_request, ioapic_request_raw,
};
pub(crate) use ivrs::{IvrsTopology, SpecialDevice};
use tables::{DeviceTableEntry, WalkError};

/// Where the register window sits. Not at QEMU's 0xFED80000: that is the AMD FCH block (PM
/// registers, reset status) Linux ioremaps on every Zen, and Xen refuses a PV dom0 a mapping of
/// the IOMMU's page, which crashed it in `print_s5_reset_status_mmio`. Real Zen server parts put
/// the IOMMU in the 0xFD000000 range.
pub(crate) const MMIO_BASE: u64 = 0xfd20_0000;
pub(crate) const MMIO_WINDOW_BYTES: u64 = 0x4000;
/// 00:00.2, where AMD root complexes put the IOMMU function.
pub(crate) const PCI_DEVFN: u8 = 0x02;
pub(crate) const REQUESTER_ID: u16 = PCI_DEVFN as u16;
const PCI_CONFIG_BYTES: usize = 256;

const PAGE_SHIFT: u32 = 12;
const PAGE_BYTES: u64 = 1 << PAGE_SHIFT;
/// MSI-format interrupt messages from devices target this window (SDM vol. 3 10.11.1); DMA to it
/// is never translated.
pub(crate) const INTERRUPT_WINDOW_FIRST: u64 = 0xfee0_0000;
pub(crate) const INTERRUPT_WINDOW_LAST: u64 = 0xfeef_ffff;

const VENDOR_AMD: u16 = 0x1022;
const DEVICE_IOMMU: u16 = 0x1419;
/// Class 08h (system peripheral), subclass 06h (IOMMU).
const CLASS_SYSTEM_PERIPHERAL: u8 = 0x08;
const SUBCLASS_IOMMU: u8 = 0x06;

const CFG_VENDOR_ID: usize = 0x00;
const CFG_DEVICE_ID: usize = 0x02;
const CFG_COMMAND: usize = 0x04;
const CFG_STATUS: usize = 0x06;
const CFG_SUBCLASS: usize = 0x0a;
const CFG_CLASS: usize = 0x0b;
const CFG_CAPABILITIES_POINTER: usize = 0x34;
const STATUS_CAPABILITIES_LIST: u16 = 1 << 4;
/// Memory space and bus master; the function has no I/O or INTx, but the OS writes them anyway.
const COMMAND_WRITABLE: u8 = 0x06;

/// The IOMMU capability block (capability id 0Fh) and the MSI capability right behind it.
pub(crate) const CAPABILITY_AT: usize = 0x40;
const CAP_HEADER: usize = CAPABILITY_AT;
const CAP_BASE_LOW: usize = CAPABILITY_AT + 0x04;
const CAP_BASE_HIGH: usize = CAPABILITY_AT + 0x08;
const CAP_RANGE: usize = CAPABILITY_AT + 0x0c;
const CAP_MISC: usize = CAPABILITY_AT + 0x10;
const MSI_AT: usize = 0x58;
pub(crate) const MSI_CONTROL: usize = MSI_AT + 0x02;
pub(crate) const MSI_ADDRESS_LOW: usize = MSI_AT + 0x04;
pub(crate) const MSI_ADDRESS_HIGH: usize = MSI_AT + 0x08;
pub(crate) const MSI_DATA: usize = MSI_AT + 0x0c;
const CAPABILITY_ID_IOMMU: u8 = 0x0f;
const CAPABILITY_ID_MSI: u8 = 0x05;
/// CapType (bits 18:16) is 011b for an IOMMU block, CapRev (23:19) is 1.
const CAP_TYPE_IOMMU: u32 = 3 << 16;
const CAP_REVISION: u32 = 1 << 19;
/// EFRSup: the extended feature register at MMIO 0x30 exists. CapExt: the capability block has
/// the extended (misc info) registers.
const CAP_EFR_SUPPORTED: u32 = 1 << 27;
const CAP_EXTENDED: u32 = 1 << 28;
/// Base address enable (bit 0) and the address bits (31:14) of the low dword.
const CAP_BASE_ENABLE: u32 = 1;
const CAP_BASE_ADDRESS_LOW_MASK: u32 = 0xffff_c000;
/// MSI control: enable, 64-bit capable. One message only, so the multiple-message bits stay 0.
pub(crate) const MSI_CONTROL_ENABLE: u16 = 1 << 0;
const MSI_CONTROL_64BIT: u16 = 1 << 7;
const MSI_CONTROL_WRITABLE: u8 = 0x01;
/// The misc info register at cap+0x10 (MsiNum 0, GvaSize 48 bit, PaSize 48, VaSize 64).
const MISC_GVA_SIZE_48: u32 = 2 << 5;
const MISC_PA_SIZE_SHIFT: u32 = 8;
const MISC_VA_SIZE_SHIFT: u32 = 15;
pub(crate) const PHYSICAL_ADDRESS_BITS: u32 = 48;
pub(crate) const VIRTUAL_ADDRESS_BITS: u32 = 64;

/// MMIO registers (64-bit, accessed as dwords or qwords).
const REG_DEVICE_TABLE: u64 = 0x0000;
const REG_COMMAND_BASE: u64 = 0x0008;
const REG_EVENT_BASE: u64 = 0x0010;
const REG_CONTROL: u64 = 0x0018;
const REG_EXCLUSION_BASE: u64 = 0x0020;
const REG_EXCLUSION_LIMIT: u64 = 0x0028;
const REG_EXTENDED_FEATURE: u64 = 0x0030;
const REG_COMMAND_HEAD: u64 = 0x2000;
const REG_COMMAND_TAIL: u64 = 0x2008;
const REG_EVENT_HEAD: u64 = 0x2010;
const REG_EVENT_TAIL: u64 = 0x2018;
const REG_STATUS: u64 = 0x2020;
/// With XTEn the event and completion-wait interrupts are not MSIs: software programs the vector
/// and a 32-bit destination here (IntCapXT, `drivers/iommu/amd/init.c`), and the IOMMU sends
/// them to the local APIC itself. The PPR and GA log ones have no event source here and are only
/// stored.
const REG_XT_GENERAL_INTERRUPT: u64 = 0x0170;
const REG_XT_PPR_INTERRUPT: u64 = 0x0178;
const REG_XT_GA_LOG_INTERRUPT: u64 = 0x0180;
/// Layout of an XT interrupt control register: DM (bit 2), destination bits 23:0 at 31:8, vector
/// at 39:32, destination bits 31:24 at 63:56.
const XT_DESTINATION_MODE_LOGICAL: u64 = 1 << 2;
const XT_DESTINATION_LOW_SHIFT: u32 = 8;
const XT_DESTINATION_LOW_MASK: u64 = 0xff_ffff;
const XT_VECTOR_SHIFT: u32 = 32;
const XT_DESTINATION_HIGH_SHIFT: u32 = 56;

/// Device table base: address bits 51:12, Size (bits 8:0): the table has (Size + 1) * 4 KiB.
const DEVICE_TABLE_WRITABLE: u64 = 0x000f_ffff_ffff_f1ff;
const DEVICE_TABLE_SIZE_MASK: u64 = 0x1ff;
const DEVICE_TABLE_ENTRY_BYTES: u64 = 32;
const DEVICE_TABLE_SIZE_UNIT: u64 = 4096;
/// Command buffer and event log bases: address bits 51:12 and a length (59:56) that is log2 of
/// the entry count. The spec's smallest size is 2^8 entries.
const RING_BASE_WRITABLE: u64 = 0x0f0f_ffff_ffff_f000;
const RING_LENGTH_SHIFT: u32 = 56;
const RING_LENGTH_MASK: u64 = 0xf;
const RING_MIN_LENGTH: u64 = 8;
const RING_ENTRY_BYTES: u64 = 16;
const BASE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
/// Head and tail are byte offsets with the low four bits zero, 19 bits wide.
const RING_POINTER_MASK: u64 = 0x7_fff0;

/// Exclusion range: base bit 0 enables it, bit 1 (ExclusionAllow) applies it to every device
/// rather than only those whose DTE.EX is set. The limit's low 12 bits read as ones: the range
/// is inclusive of its last page.
const EXCLUSION_ENABLE: u64 = 1 << 0;
const EXCLUSION_ALLOW_ALL: u64 = 1 << 1;
const EXCLUSION_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const EXCLUSION_LIMIT_LOW_BITS: u64 = 0xfff;

/// Control register bits (spec 3.3.2 "IOMMU Control Register").
const CONTROL_IOMMU_ENABLE: u64 = 1 << 0;
const CONTROL_EVENT_LOG_ENABLE: u64 = 1 << 2;
const CONTROL_EVENT_INTERRUPT_ENABLE: u64 = 1 << 3;
const CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE: u64 = 1 << 4;
const CONTROL_COMMAND_BUFFER_ENABLE: u64 = 1 << 12;
/// Guest APIC (128-bit IRTE) mode.
const CONTROL_GUEST_APIC_ENABLE: u64 = 1 << 17;
/// XTEn: x2APIC mode, where IRTEs carry a 32-bit destination. IntCapXTEn: the IOMMU's own
/// interrupts use the XT registers instead of the MSI capability.
const CONTROL_XT_ENABLE: u64 = 1 << 50;
const CONTROL_INTCAPXT_ENABLE: u64 = 1 << 51;
/// InvTimeOut (7:5), PassPW, ResPassPW, Coherent, Isoc: stored, no effect.
const CONTROL_STORED_ONLY: u64 = 0xe0 | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11);
const CONTROL_WRITABLE: u64 = CONTROL_IOMMU_ENABLE
    | CONTROL_EVENT_LOG_ENABLE
    | CONTROL_EVENT_INTERRUPT_ENABLE
    | CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE
    | CONTROL_COMMAND_BUFFER_ENABLE
    | CONTROL_GUEST_APIC_ENABLE
    | CONTROL_XT_ENABLE
    | CONTROL_INTCAPXT_ENABLE
    | CONTROL_STORED_ONLY;

/// Status register: overflow, event interrupt and completion-wait interrupt are
/// write-1-to-clear; the run bits mirror the enables.
const STATUS_EVENT_OVERFLOW: u64 = 1 << 0;
const STATUS_EVENT_INTERRUPT: u64 = 1 << 1;
const STATUS_COMPLETION_WAIT_INTERRUPT: u64 = 1 << 2;
const STATUS_EVENT_LOG_RUNNING: u64 = 1 << 3;
const STATUS_COMMAND_BUFFER_RUNNING: u64 = 1 << 4;
const STATUS_WRITE_ONE_TO_CLEAR: u64 =
    STATUS_EVENT_OVERFLOW | STATUS_EVENT_INTERRUPT | STATUS_COMPLETION_WAIT_INTERRUPT;

/// Extended feature register: PreFSup (the prefetch command is accepted), XTSup (x2APIC: 32-bit
/// IRTE destinations and the IntCapXT interrupt registers), IASup (invalidate all), GASup
/// (128-bit IRTEs) and HATS = 10b (host page tables of up to six levels).
const EFR_PREFETCH: u64 = 1 << 0;
const EFR_X2APIC: u64 = 1 << 2;
const EFR_INVALIDATE_ALL: u64 = 1 << 6;
const EFR_GUEST_APIC: u64 = 1 << 7;
const EFR_HOST_ADDRESS_TRANSLATION_SIZE_6_LEVEL: u64 = 2 << 10;
pub(crate) const EXTENDED_FEATURES: u64 = EFR_PREFETCH
    | EFR_X2APIC
    | EFR_INVALIDATE_ALL
    | EFR_GUEST_APIC
    | EFR_HOST_ADDRESS_TRANSLATION_SIZE_6_LEVEL;

/// IOTLB slots, direct-mapped by requester and page (`iotlb_slot`): 16 MiB of 4 KiB pages. It
/// used to be a map dropped wholesale at 1024 pages, which a device streaming through more
/// buffers than that turned into a table walk per page.
const IOTLB_ENTRIES: usize = 4096;
/// Spreads requesters over the IOTLB so two devices using the same addresses (every virtio
/// queue starts low) do not evict each other.
const IOTLB_REQUESTER_SHIFT: u32 = 6;
/// Device table entries cached, direct-mapped by the low bits of the requester id: bus 0, where
/// every function of this machine sits, fits.
const DEVICE_ENTRY_CACHE_ENTRIES: usize = 256;

pub(crate) trait PhysRead {
    fn read_phys(&self, addr: u64, buf: &mut [u8]) -> bool;
}

pub(crate) trait PhysWrite: PhysRead {
    fn write_phys(&mut self, addr: u64, data: &[u8]) -> bool;
}

fn read_u64(mem: &dyn PhysRead, addr: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    mem.read_phys(addr, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Access {
    Read,
    Write,
}

/// Why a DMA access did not happen. Everything but `InterruptWindow` also queued an event.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DmaFault {
    /// Not present, or the page table is malformed (IO_PAGE_FAULT).
    PageFault,
    /// Present but the permissions forbid this access (IO_PAGE_FAULT with PR set).
    Permission,
    /// The device table could not be read or the entry is out of range or illegal.
    DeviceTable,
    /// A page table could not be read (PAGE_TAB_HW_ERROR).
    PageTable,
    /// DMA aimed at the interrupt message window; interrupts do not go through here.
    InterruptWindow,
}

/// `bytes_left` is how many bytes from `phys` stay inside this one mapping's 4 KiB page (the
/// IOTLB works in 4 KiB units), so a caller splits a longer access at that boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Translation {
    pub phys: u64,
    pub bytes_left: u64,
}

/// An interrupt the IOMMU wants delivered (event log or completion-wait).
#[derive(Clone, Copy, Debug)]
pub(crate) enum IommuInterrupt {
    Message(MsiMessage),
    /// Straight to the local APIC, bypassing interrupt remapping (IntCapXT).
    Direct(crate::dev::apic::Interrupt),
}

/// An MSI the IOMMU's capability holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct MsiMessage {
    pub address: u64,
    pub data: u32,
}

#[derive(Clone, Copy)]
struct IotlbEntry {
    requester: u16,
    /// Device address >> `PAGE_SHIFT`; `NO_IOTLB_PAGE` for an empty slot.
    page: u64,
    phys_page: u64,
    read: bool,
    write: bool,
    domain: u16,
}

/// Never a page number: device addresses have at most 64 - 12 bits of page.
const NO_IOTLB_PAGE: u64 = u64::MAX;
const EMPTY_IOTLB_ENTRY: IotlbEntry = IotlbEntry {
    requester: 0,
    page: NO_IOTLB_PAGE,
    phys_page: 0,
    read: false,
    write: false,
    domain: 0,
};

fn iotlb_slot(requester: u16, page: u64) -> usize {
    (page ^ (requester as u64) << IOTLB_REQUESTER_SHIFT) as usize & (IOTLB_ENTRIES - 1)
}

pub(crate) struct Iommu {
    config: [u8; PCI_CONFIG_BYTES],
    config_writable: [u8; PCI_CONFIG_BYTES],

    device_table: u64,
    command_base: u64,
    event_base: u64,
    control: u64,
    exclusion_base: u64,
    exclusion_limit: u64,
    command_head: u64,
    command_tail: u64,
    event_head: u64,
    event_tail: u64,
    status: u64,
    xt_general_interrupt: u64,
    xt_ppr_interrupt: u64,
    xt_ga_log_interrupt: u64,

    /// Set by an illegal command or a failed completion-wait store: nothing runs until software
    /// toggles CmdBufEn.
    command_stalled: bool,
    pending_events: Vec<[u32; 4]>,
    msi_pending: bool,
    /// `IOTLB_ENTRIES` slots, allocated by the first translation. Empty until then because
    /// `GuestMem::with_dma` swaps a default `Iommu` in for every device poll, which must not
    /// allocate.
    iotlb: Vec<IotlbEntry>,
    /// Device table entries as read, by requester (`DEVICE_ENTRY_CACHE_ENTRIES`). Hardware
    /// caches them too and software invalidates one after changing it (INVALIDATE_DEVTAB_ENTRY),
    /// which Linux and Xen do; until then DMA uses the old entry. Interrupt remapping still reads
    /// the table every time.
    /// Allocated by the first fetch, as `iotlb`.
    device_entries: Vec<Option<(u16, DeviceTableEntry)>>,
}

impl Default for Iommu {
    fn default() -> Self {
        Self::new()
    }
}

impl Iommu {
    pub fn new() -> Self {
        let mut iommu = Iommu {
            config: [0; PCI_CONFIG_BYTES],
            config_writable: [0; PCI_CONFIG_BYTES],
            device_table: 0,
            command_base: 0,
            event_base: 0,
            control: 0,
            exclusion_base: 0,
            exclusion_limit: 0,
            command_head: 0,
            command_tail: 0,
            event_head: 0,
            event_tail: 0,
            status: 0,
            xt_general_interrupt: 0,
            xt_ppr_interrupt: 0,
            xt_ga_log_interrupt: 0,
            command_stalled: false,
            pending_events: Vec::new(),
            msi_pending: false,
            iotlb: Vec::new(),
            device_entries: Vec::new(),
        };
        iommu.init_config();
        iommu
    }

    fn put(&mut self, at: usize, bytes: &[u8]) {
        self.config[at..at + bytes.len()].copy_from_slice(bytes);
    }

    fn init_config(&mut self) {
        self.put(CFG_VENDOR_ID, &VENDOR_AMD.to_le_bytes());
        self.put(CFG_DEVICE_ID, &DEVICE_IOMMU.to_le_bytes());
        self.put(CFG_STATUS, &STATUS_CAPABILITIES_LIST.to_le_bytes());
        self.config[CFG_SUBCLASS] = SUBCLASS_IOMMU;
        self.config[CFG_CLASS] = CLASS_SYSTEM_PERIPHERAL;
        self.config[CFG_CAPABILITIES_POINTER] = CAPABILITY_AT as u8;
        self.config_writable[CFG_COMMAND] = COMMAND_WRITABLE;

        let header = CAPABILITY_ID_IOMMU as u32
            | (MSI_AT as u32) << 8
            | CAP_TYPE_IOMMU
            | CAP_REVISION
            | CAP_EFR_SUPPORTED
            | CAP_EXTENDED;
        self.put(CAP_HEADER, &header.to_le_bytes());
        // No firmware programs the base address, so it starts out set and enabled; the IVRS
        // table reports the same address.
        let base_low = (MMIO_BASE as u32 & CAP_BASE_ADDRESS_LOW_MASK) | CAP_BASE_ENABLE;
        self.put(CAP_BASE_LOW, &base_low.to_le_bytes());
        self.put(CAP_BASE_HIGH, &((MMIO_BASE >> 32) as u32).to_le_bytes());
        self.config_writable[CAP_BASE_LOW..CAP_BASE_LOW + 4]
            .copy_from_slice(&(CAP_BASE_ENABLE | CAP_BASE_ADDRESS_LOW_MASK).to_le_bytes());
        self.config_writable[CAP_BASE_HIGH..CAP_BASE_HIGH + 4].fill(0xff);
        // RngValid clear: the unit serves every device; the IVHD block says which ones.
        self.put(CAP_RANGE, &0u32.to_le_bytes());
        let misc = MISC_GVA_SIZE_48
            | PHYSICAL_ADDRESS_BITS << MISC_PA_SIZE_SHIFT
            | VIRTUAL_ADDRESS_BITS << MISC_VA_SIZE_SHIFT;
        self.put(CAP_MISC, &misc.to_le_bytes());

        self.config[MSI_AT] = CAPABILITY_ID_MSI;
        self.config[MSI_AT + 1] = 0;
        self.put(MSI_CONTROL, &MSI_CONTROL_64BIT.to_le_bytes());
        self.config_writable[MSI_CONTROL] = MSI_CONTROL_WRITABLE;
        // Address bits 1:0 are reserved zero in an MSI address.
        self.config_writable[MSI_ADDRESS_LOW..MSI_ADDRESS_LOW + 4]
            .copy_from_slice(&0xffff_fffcu32.to_le_bytes());
        self.config_writable[MSI_ADDRESS_HIGH..MSI_ADDRESS_HIGH + 4].fill(0xff);
        self.config_writable[MSI_DATA..MSI_DATA + 2].fill(0xff);
    }

    /// The 256-byte PCI configuration space, byte-addressed.
    pub fn config(&self) -> &[u8; PCI_CONFIG_BYTES] {
        &self.config
    }

    pub fn config_read(&self, reg: usize, size: usize) -> u32 {
        let mut value = 0u32;
        for i in 0..size.min(4) {
            value |= (*self.config.get(reg + i).unwrap_or(&0xff) as u32) << (8 * i);
        }
        value
    }

    /// Writes only the bits the function implements; everything else (ids, class, capability
    /// headers, the MSI capability's read-only control bits) ignores the write.
    pub fn config_write(&mut self, reg: usize, size: usize, value: u32) {
        for i in 0..size.min(4) {
            let at = reg + i;
            if at >= PCI_CONFIG_BYTES {
                break;
            }
            let mask = self.config_writable[at];
            let byte = (value >> (8 * i)) as u8;
            self.config[at] = (self.config[at] & !mask) | (byte & mask);
        }
    }

    /// The register window, if the capability's base address is enabled.
    pub fn mmio_base(&self) -> Option<u64> {
        let low = self.config_read(CAP_BASE_LOW, 4);
        let high = self.config_read(CAP_BASE_HIGH, 4);
        (low & CAP_BASE_ENABLE != 0)
            .then_some((low & CAP_BASE_ADDRESS_LOW_MASK) as u64 | (high as u64) << 32)
    }

    pub fn mmio_offset(&self, phys_addr: u64) -> Option<u64> {
        let base = self.mmio_base()?;
        let offset = phys_addr.checked_sub(base)?;
        (offset < MMIO_WINDOW_BYTES).then_some(offset)
    }

    fn control(&self) -> u64 {
        self.control
    }

    fn status_set(&mut self, bits: u64) {
        self.status |= bits;
    }

    fn status_clear(&mut self, bits: u64) {
        self.status &= !bits;
    }

    fn status(&self) -> u64 {
        self.status
    }

    pub fn enabled(&self) -> bool {
        self.control & CONTROL_IOMMU_ENABLE != 0
    }

    /// Bytes of device table, 0 until software programs one.
    fn device_table_bytes(&self) -> u64 {
        if self.device_table & BASE_ADDRESS_MASK == 0 {
            return 0;
        }
        ((self.device_table & DEVICE_TABLE_SIZE_MASK) + 1) * DEVICE_TABLE_SIZE_UNIT
    }

    fn device_table_base(&self) -> u64 {
        self.device_table & BASE_ADDRESS_MASK
    }

    /// Size in bytes of the ring a base register describes, 0 if the length is not a legal one.
    fn ring_bytes(base_register: u64) -> u64 {
        let length = (base_register >> RING_LENGTH_SHIFT) & RING_LENGTH_MASK;
        if length < RING_MIN_LENGTH {
            0
        } else {
            RING_ENTRY_BYTES << length
        }
    }

    fn command_buffer(&self) -> (u64, u64) {
        (
            self.command_base & BASE_ADDRESS_MASK,
            Self::ring_bytes(self.command_base),
        )
    }

    fn event_log(&self) -> (u64, u64) {
        (
            self.event_base & BASE_ADDRESS_MASK,
            Self::ring_bytes(self.event_base),
        )
    }

    fn command_head(&self) -> u64 {
        self.command_head
    }

    fn set_command_head(&mut self, head: u64) {
        self.command_head = head;
    }

    fn command_tail(&self) -> u64 {
        self.command_tail
    }

    fn event_pointers(&self) -> (u64, u64) {
        (self.event_head, self.event_tail)
    }

    fn set_event_tail(&mut self, tail: u64) {
        self.event_tail = tail;
    }

    /// An event or completion-wait interrupt is due; `take_interrupt` hands it out once.
    pub(crate) fn raise_msi(&mut self) {
        self.msi_pending = true;
    }

    fn iotlb_remove_device(&mut self, requester: u16) {
        for entry in self.iotlb.iter_mut() {
            if entry.requester == requester {
                *entry = EMPTY_IOTLB_ENTRY;
            }
        }
    }

    fn iotlb_remove_domain(&mut self, domain: u16) {
        for entry in self.iotlb.iter_mut() {
            if entry.page != NO_IOTLB_PAGE && entry.domain == domain {
                *entry = EMPTY_IOTLB_ENTRY;
            }
        }
    }

    fn iotlb_clear(&mut self) {
        self.iotlb.fill(EMPTY_IOTLB_ENTRY);
    }

    fn forget_device_entry(&mut self, requester: u16) {
        if let Some(slot) = self
            .device_entries
            .get_mut(requester as usize % DEVICE_ENTRY_CACHE_ENTRIES)
            && slot.is_some_and(|(cached, _)| cached == requester)
        {
            *slot = None;
        }
    }

    /// Everything cached about the device table: the entries and what was translated with them.
    fn forget_device_table(&mut self) {
        self.iotlb_clear();
        self.device_entries.fill(None);
    }

    /// The message to deliver for a pending event or completion-wait interrupt, once: `None` if
    /// nothing is pending or software has not enabled the MSI capability (the interrupt is then
    /// lost, as it would be on hardware with the capability off).
    pub fn take_msi(&mut self) -> Option<MsiMessage> {
        if !core::mem::take(&mut self.msi_pending) {
            return None;
        }
        let control = self.config_read(MSI_CONTROL, 2) as u16;
        if control & MSI_CONTROL_ENABLE == 0 {
            return None;
        }
        let address = self.config_read(MSI_ADDRESS_LOW, 4) as u64
            | (self.config_read(MSI_ADDRESS_HIGH, 4) as u64) << 32;
        let data = self.config_read(MSI_DATA, 2);
        Some(MsiMessage { address, data })
    }

    /// What the pending event or completion-wait interrupt turns into: the MSI the capability
    /// holds, or with IntCapXTEn the vector and destination of the XT general interrupt register.
    /// Once, like `take_msi`.
    pub fn take_interrupt(&mut self) -> Option<IommuInterrupt> {
        if self.control & CONTROL_INTCAPXT_ENABLE == 0 {
            return self.take_msi().map(IommuInterrupt::Message);
        }
        if !core::mem::take(&mut self.msi_pending) {
            return None;
        }
        let register = self.xt_general_interrupt;
        let low = (register >> XT_DESTINATION_LOW_SHIFT) & XT_DESTINATION_LOW_MASK;
        let high = register >> XT_DESTINATION_HIGH_SHIFT;
        Some(IommuInterrupt::Direct(crate::dev::apic::Interrupt {
            vector: (register >> XT_VECTOR_SHIFT) as u8,
            delivery_mode: crate::dev::apic::DELIVERY_FIXED,
            level_triggered: false,
            destination: (high << 24 | low) as u32,
            logical: register & XT_DESTINATION_MODE_LOGICAL != 0,
        }))
    }

    /// XTEn: IRTEs carry a 32-bit destination (the top 8 bits live in the high qword).
    fn xt_enabled(&self) -> bool {
        self.control & CONTROL_XT_ENABLE != 0
    }

    fn read_register(&self, offset: u64) -> u64 {
        match offset {
            REG_DEVICE_TABLE => self.device_table,
            REG_COMMAND_BASE => self.command_base,
            REG_EVENT_BASE => self.event_base,
            REG_CONTROL => self.control,
            REG_EXCLUSION_BASE => self.exclusion_base,
            REG_EXCLUSION_LIMIT => self.exclusion_limit,
            REG_EXTENDED_FEATURE => EXTENDED_FEATURES,
            REG_COMMAND_HEAD => self.command_head,
            REG_COMMAND_TAIL => self.command_tail,
            REG_EVENT_HEAD => self.event_head,
            REG_EVENT_TAIL => self.event_tail,
            REG_STATUS => self.status,
            REG_XT_GENERAL_INTERRUPT => self.xt_general_interrupt,
            REG_XT_PPR_INTERRUPT => self.xt_ppr_interrupt,
            REG_XT_GA_LOG_INTERRUPT => self.xt_ga_log_interrupt,
            _ => 0,
        }
    }

    /// `offset` is relative to the window. Registers not modeled (PPR, GA log, performance
    /// counters, SMI filters, hardware error registers) read zero.
    pub fn mmio_read(&self, offset: u64, size: u8) -> u64 {
        let lane = offset & 7;
        let value = self.read_register(offset & !7) >> (8 * lane);
        match size {
            1..=7 => value & ((1u64 << (8 * size as u32)) - 1),
            _ => value,
        }
    }

    /// Applies a register write, runs the command buffer if software moved the tail or enabled
    /// it, and writes any queued events to the log.
    pub fn mmio_write(&mut self, offset: u64, size: u8, value: u64, mem: &mut dyn PhysWrite) {
        let lane = offset & 7;
        let bytes = (size as u64).min(8 - lane);
        let lane_mask = match bytes {
            8 => u64::MAX,
            _ => ((1u64 << (8 * bytes)) - 1) << (8 * lane),
        };
        let written = (value << (8 * lane)) & lane_mask;
        let register = offset & !7;
        let merge =
            |old: u64, writable: u64| (old & !(lane_mask & writable)) | (written & writable);
        match register {
            REG_DEVICE_TABLE => {
                self.device_table = merge(self.device_table, DEVICE_TABLE_WRITABLE);
                self.forget_device_table();
            }
            REG_COMMAND_BASE => {
                self.command_base = merge(self.command_base, RING_BASE_WRITABLE);
                self.command_head = 0;
                self.command_tail = 0;
            }
            REG_EVENT_BASE => {
                self.event_base = merge(self.event_base, RING_BASE_WRITABLE);
                self.event_head = 0;
                self.event_tail = 0;
            }
            REG_CONTROL => self.write_control(merge(self.control, CONTROL_WRITABLE)),
            REG_EXCLUSION_BASE => {
                self.exclusion_base = merge(
                    self.exclusion_base,
                    EXCLUSION_ADDRESS_MASK | EXCLUSION_ENABLE | EXCLUSION_ALLOW_ALL,
                );
            }
            REG_EXCLUSION_LIMIT => {
                self.exclusion_limit =
                    merge(self.exclusion_limit, EXCLUSION_ADDRESS_MASK) | EXCLUSION_LIMIT_LOW_BITS;
            }
            REG_COMMAND_HEAD => self.command_head = merge(self.command_head, RING_POINTER_MASK),
            REG_COMMAND_TAIL => self.command_tail = merge(self.command_tail, RING_POINTER_MASK),
            REG_EVENT_HEAD => self.event_head = merge(self.event_head, RING_POINTER_MASK),
            REG_EVENT_TAIL => self.event_tail = merge(self.event_tail, RING_POINTER_MASK),
            REG_STATUS => self.status &= !(written & STATUS_WRITE_ONE_TO_CLEAR),
            REG_XT_GENERAL_INTERRUPT => {
                self.xt_general_interrupt = merge(self.xt_general_interrupt, u64::MAX)
            }
            REG_XT_PPR_INTERRUPT => self.xt_ppr_interrupt = merge(self.xt_ppr_interrupt, u64::MAX),
            REG_XT_GA_LOG_INTERRUPT => {
                self.xt_ga_log_interrupt = merge(self.xt_ga_log_interrupt, u64::MAX)
            }
            _ => {}
        }
        self.service(mem);
    }

    fn write_control(&mut self, new: u64) {
        let old = core::mem::replace(&mut self.control, new);
        let changed = old ^ new;
        if changed & CONTROL_IOMMU_ENABLE != 0 {
            self.forget_device_table();
        }
        if changed & CONTROL_EVENT_LOG_ENABLE != 0 {
            if new & CONTROL_EVENT_LOG_ENABLE != 0 {
                self.status |= STATUS_EVENT_LOG_RUNNING;
            } else {
                self.status &= !STATUS_EVENT_LOG_RUNNING;
            }
        }
        if changed & CONTROL_COMMAND_BUFFER_ENABLE != 0 {
            if new & CONTROL_COMMAND_BUFFER_ENABLE != 0 {
                self.status |= STATUS_COMMAND_BUFFER_RUNNING;
                self.command_stalled = false;
            } else {
                self.status &= !STATUS_COMMAND_BUFFER_RUNNING;
            }
        }
    }

    /// Runs commands the guest queued and writes queued events to the log. Called after every
    /// register write; the owner of the device calls it from its tick too, so events found by
    /// DMA translation reach the log without waiting for the next register access.
    pub fn service(&mut self, mem: &mut dyn PhysWrite) {
        if self.control & CONTROL_COMMAND_BUFFER_ENABLE != 0 {
            self.run_commands(mem);
        }
        self.flush_events(mem);
    }

    /// Translates one device DMA address. With the IOMMU off (or no device table programmed)
    /// every address maps to itself. Faults are queued as events for `service`.
    pub fn translate(
        &mut self,
        mem: &dyn PhysRead,
        requester: u16,
        iova: u64,
        access: Access,
    ) -> Result<Translation, DmaFault> {
        self.translate_inner(mem, requester, iova, access, true)
    }

    /// Like `translate` for a read, but reports no faults and ignores permissions: whether the
    /// address is mapped at all. `GuestMem::contains` validates guest-supplied ranges with it.
    pub fn probe(&mut self, mem: &dyn PhysRead, requester: u16, iova: u64) -> Option<Translation> {
        self.translate_inner(mem, requester, iova, Access::Read, false)
            .ok()
    }

    fn translate_inner(
        &mut self,
        mem: &dyn PhysRead,
        requester: u16,
        iova: u64,
        access: Access,
        report: bool,
    ) -> Result<Translation, DmaFault> {
        let identity = Translation {
            phys: iova,
            bytes_left: u64::MAX,
        };
        if !self.enabled() || self.device_table_bytes() == 0 {
            return Ok(identity);
        }
        if (INTERRUPT_WINDOW_FIRST..=INTERRUPT_WINDOW_LAST).contains(&iova) {
            return Err(DmaFault::InterruptWindow);
        }
        let offset = iova & (PAGE_BYTES - 1);
        let page_translation = |phys_page: u64| Translation {
            phys: phys_page | offset,
            bytes_left: PAGE_BYTES - offset,
        };
        let rw_flag = if access == Access::Write {
            events::FLAG_WRITE
        } else {
            0
        };
        let page = iova >> PAGE_SHIFT;
        if let Some(&entry) = self.iotlb.get(iotlb_slot(requester, page))
            && entry.page == page
            && entry.requester == requester
        {
            let allowed = match access {
                Access::Read => entry.read,
                Access::Write => entry.write,
            };
            if allowed || !report {
                return Ok(page_translation(entry.phys_page));
            }
            self.log_page_fault(
                requester,
                entry.domain,
                iova,
                rw_flag | events::FLAG_PRESENT,
            );
            return Err(DmaFault::Permission);
        }
        let entry = match self.device_entry(mem, requester) {
            Ok(entry) => entry,
            Err(error) => {
                if report {
                    self.report_device_entry_error(requester, iova, rw_flag, error);
                }
                return Err(DmaFault::DeviceTable);
            }
        };
        // V=0: the device is not translated at all (QEMU does the same; Linux blocks a device
        // with V=1, TV=1, mode 0 and no permissions instead).
        if !entry.valid() {
            return Ok(identity);
        }
        if entry.has_reserved_bits() {
            if report {
                self.log_illegal_device_entry(requester, iova, rw_flag);
            }
            return Err(DmaFault::DeviceTable);
        }
        if self.in_exclusion_range(iova) && (self.exclusion_allows_all() || entry.exclusion()) {
            return Ok(identity);
        }
        if !entry.translation_valid() {
            return Ok(identity);
        }
        let domain = entry.domain_id();
        if entry.mode() == 0 {
            let allowed = match access {
                Access::Read => entry.read_allowed(),
                Access::Write => entry.write_allowed(),
            };
            if allowed || !report {
                return Ok(identity);
            }
            self.log_page_fault(requester, domain, iova, rw_flag | events::FLAG_PRESENT);
            return Err(DmaFault::Permission);
        }
        match tables::walk(mem, &entry, iova) {
            Ok(leaf) => {
                let allowed = match access {
                    Access::Read => leaf.read,
                    Access::Write => leaf.write,
                };
                if self.iotlb.is_empty() {
                    self.iotlb = vec![EMPTY_IOTLB_ENTRY; IOTLB_ENTRIES];
                }
                self.iotlb[iotlb_slot(requester, page)] = IotlbEntry {
                    requester,
                    page,
                    phys_page: leaf.phys_page,
                    read: leaf.read,
                    write: leaf.write,
                    domain,
                };
                if allowed || !report {
                    return Ok(page_translation(leaf.phys_page));
                }
                self.log_page_fault(requester, domain, iova, rw_flag | events::FLAG_PRESENT);
                Err(DmaFault::Permission)
            }
            Err(WalkError::PageTableRead(address)) => {
                if report {
                    self.log_page_table_error(requester, domain, address, rw_flag);
                }
                Err(DmaFault::PageTable)
            }
            Err(WalkError::NotPresent | WalkError::Malformed) => {
                if report {
                    self.log_page_fault(requester, domain, iova, rw_flag);
                }
                Err(DmaFault::PageFault)
            }
        }
    }

    fn in_exclusion_range(&self, address: u64) -> bool {
        self.exclusion_base & EXCLUSION_ENABLE != 0
            && address >= self.exclusion_base & EXCLUSION_ADDRESS_MASK
            && address <= self.exclusion_limit
    }

    fn exclusion_allows_all(&self) -> bool {
        self.exclusion_base & EXCLUSION_ALLOW_ALL != 0
    }

    /// The device table entry of `requester`, from `device_entries` or the table (and then
    /// cached; failures are not).
    fn device_entry(
        &mut self,
        mem: &dyn PhysRead,
        requester: u16,
    ) -> Result<DeviceTableEntry, tables::DeviceEntryError> {
        let slot = requester as usize % DEVICE_ENTRY_CACHE_ENTRIES;
        if let Some(&Some((cached, entry))) = self.device_entries.get(slot)
            && cached == requester
        {
            return Ok(entry);
        }
        let entry = self.fetch_device_entry(mem, requester)?;
        if self.device_entries.is_empty() {
            self.device_entries = vec![None; DEVICE_ENTRY_CACHE_ENTRIES];
        }
        self.device_entries[slot] = Some((requester, entry));
        Ok(entry)
    }

    /// Reads the device table entry of `requester`; `Err` if it lies outside the table or RAM.
    fn fetch_device_entry(
        &self,
        mem: &dyn PhysRead,
        requester: u16,
    ) -> Result<DeviceTableEntry, tables::DeviceEntryError> {
        let offset = requester as u64 * DEVICE_TABLE_ENTRY_BYTES;
        if offset + DEVICE_TABLE_ENTRY_BYTES > self.device_table_bytes() {
            return Err(tables::DeviceEntryError::OutsideTable);
        }
        let address = self.device_table_base() + offset;
        let mut quads = [0u64; 4];
        for (i, quad) in quads.iter_mut().enumerate() {
            *quad = read_u64(mem, address + 8 * i as u64)
                .ok_or(tables::DeviceEntryError::Unreadable(address))?;
        }
        Ok(DeviceTableEntry::new(quads))
    }

    fn report_device_entry_error(
        &mut self,
        requester: u16,
        iova: u64,
        flags: u16,
        error: tables::DeviceEntryError,
    ) {
        match error {
            // A device the table does not cover has no entry: report it like a device whose
            // mapping is absent.
            tables::DeviceEntryError::OutsideTable => {
                self.log_page_fault(requester, 0, iova, flags)
            }
            tables::DeviceEntryError::Unreadable(address) => {
                self.log_device_table_error(requester, address, flags)
            }
        }
    }
}
