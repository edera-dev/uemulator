use super::guestmem::GuestMem;
use super::iommu::{self, InterruptRequest, Iommu};
use super::size_mask;
use super::virtio::{
    COMMON_CFG_OFFSET, DEVICE_CFG_OFFSET, DEVICE_TYPE_BLOCK, DEVICE_TYPE_CONSOLE, DEVICE_TYPE_NET,
    ISR_OFFSET, MSIX_CONTROL_ENABLE, MSIX_CONTROL_FUNCTION_MASK, MSIX_PBA_OFFSET,
    MSIX_TABLE_OFFSET, NOTIFY_OFF_MULTIPLIER, NOTIFY_OFFSET, REGION_SIZE, Transport,
};
#[cfg(freestanding)]
use crate::prelude::*;

/// Start of the MMIO window, which is the PCI hole in the RAM layout: guest RAM is never mapped
/// from here up to 4 GiB (`ram::flat_bytes`), and accesses there dispatch to the devices.
pub(crate) const MMIO_BASE: u64 = crate::ram::PCI_HOLE_START;
/// The part of it the ACPI root bridge hands out as BAR space (up to the ECAM window).
pub(crate) const PCI_MMIO_WINDOW_BYTES: u64 = 0x2000_0000;
/// Each virtio function gets one BAR0 of this size, laid out back to back from `MMIO_BASE`.
/// Six `REGION_SIZE` regions (common cfg, ISR, device cfg, notify, MSI-X table, MSI-X PBA),
/// rounded up to a power of two.
pub(crate) const BAR_SIZE: u64 = 8 * REGION_SIZE;

/// PCIe enhanced configuration access (ECAM): 4 KiB of config space per function, laid out as
/// `bus << 20 | devfn << 12 | register` from this base. The MCFG table advertises it for bus 0
/// only. Chosen below the 4 GiB hole's devices and above the BAR window.
pub(crate) const ECAM_BASE: u64 = 0xe000_0000;
const ECAM_DEVFN_SHIFT: u32 = 12;
const ECAM_BUS_SHIFT: u32 = 20;
pub(crate) const ECAM_BYTES_PER_BUS: u64 = 1 << ECAM_BUS_SHIFT;
const ECAM_FUNCTION_BYTES: u64 = 1 << ECAM_DEVFN_SHIFT;
/// Legacy PIC lines handed out to functions in order, wrapping around (functions past the fourth
/// share a line; `line_asserted` ORs them). Without ACPI the kernel takes `int_line` as the IRQ;
/// with it, the DSDT's `_PRT` repeats this assignment as GSIs (`acpi::pci_interrupt_gsi`). Never
/// 0, `mmio_write` uses 0 for "no interrupt". 9-11 are the lines QEMU's PIIX routes PCI
/// interrupts to; 5 is free on a machine without a sound card or LPT.
pub(crate) const LEGACY_IRQS: [u8; 4] = [9, 10, 11, 5];

/// Config mechanism #1 ports: the address dword at 0xCF8..=0xCFB and the data window at
/// 0xCFC..=0xCFF. Either may be accessed as a byte, word or dword; the low two bits of the
/// port are the byte lane. Port 0xCF9 is also the reset control register, which `Devices`
/// claims before it gets here.
pub(crate) const CONFIG_ADDRESS_PORT: u16 = 0xcf8;
pub(crate) const CONFIG_DATA_PORT: u16 = 0xcfc;
const BYTES_PER_DWORD: usize = 4;
const BYTE_LANE_MASK: u16 = 3;
const CONFIG_ADDRESS_ENABLE: u32 = 1 << 31;
const CONFIG_ADDRESS_BUS_SHIFT: u32 = 16;
const CONFIG_ADDRESS_DEVFN_SHIFT: u32 = 8;
/// Dword-aligned register number in the low byte; the low two bits are the byte offset
/// within the dword and come from the data port instead.
const CONFIG_ADDRESS_REGISTER_MASK: u32 = 0xfc;
/// devfn = slot << 3 | function.
const DEVFN_SLOT_SHIFT: u32 = 3;
/// Slot 0 is the host bridge, virtio functions start at slot 1 and fill slots 1..=31.
const FIRST_VIRTIO_SLOT: u8 = 1;
pub(crate) const MAX_VIRTIO_FUNCTIONS: usize = 31;
const CONFIG_SPACE_SIZE: usize = 256;

/// Standard config header offsets.
const CFG_VENDOR_ID: usize = 0x00;
const CFG_DEVICE_ID: usize = 0x02;
const CFG_COMMAND: usize = 0x04;
const CFG_COMMAND_HIGH_BYTE: usize = CFG_COMMAND + 1;
const CFG_STATUS: usize = 0x06;
const CFG_REVISION: usize = 0x08;
const CFG_SUBCLASS: usize = 0x0a;
const CFG_CLASS: usize = 0x0b;
const CFG_HEADER_TYPE: usize = 0x0e;
const HEADER_TYPE_MULTIFUNCTION: u8 = 0x80;
const CFG_BAR0: usize = 0x10;
const CFG_BAR0_LAST_BYTE: usize = CFG_BAR0 + 3;
const CFG_SUBSYSTEM_VENDOR_ID: usize = 0x2c;
const CFG_SUBSYSTEM_ID: usize = 0x2e;
const CFG_CAPABILITIES_POINTER: usize = 0x34;
const CFG_INTERRUPT_LINE: usize = 0x3c;
const CFG_INTERRUPT_PIN: usize = 0x3d;

const COMMAND_MEMORY_SPACE: u16 = 1 << 1;
const COMMAND_INTX_DISABLE: u16 = 1 << 10;
const STATUS_CAPABILITIES_LIST: u16 = 1 << 4;
/// Interrupt pin register value for INTA#.
const INTERRUPT_PIN_INTA: u8 = 1;

const VENDOR_INTEL: u16 = 0x8086;
/// 440FX host bridge (QEMU's default), only needs to exist and have class 0x0600 so the
/// kernel's `pci_sanity_check` accepts config mechanism #1.
const DEVICE_I440FX_HOST: u16 = 0x1237;
const CLASS_BRIDGE: u8 = 0x06;
const SUBCLASS_HOST_BRIDGE: u8 = 0x00;

const VENDOR_VIRTIO: u16 = 0x1af4;
/// Modern (non-transitional) PCI device id is this plus the virtio device type.
const VIRTIO_PCI_DEVICE_ID_BASE: u16 = 0x1040;
/// Non-transitional virtio devices report revision 1 or higher.
const VIRTIO_PCI_REVISION: u8 = 1;
/// Class codes follow QEMU: net is ethernet, block is SCSI storage, everything else (console)
/// is "communication, other".
const CLASS_NETWORK: u8 = 0x02;
const SUBCLASS_ETHERNET: u8 = 0x00;
const CLASS_STORAGE: u8 = 0x01;
const SUBCLASS_SCSI: u8 = 0x00;
const CLASS_COMMUNICATION: u8 = 0x07;
const SUBCLASS_COMMUNICATION_OTHER: u8 = 0x80;

/// Vendor-specific capability id used by every virtio PCI capability.
const CAP_ID_VENDOR_SPECIFIC: u8 = 0x09;
/// `cfg_type` of the virtio capabilities we expose (`VIRTIO_PCI_CAP_*`). The PCI-config-access
/// capability (type 5) is not provided.
const VIRTIO_PCI_CAP_COMMON_CFG: u8 = 1;
const VIRTIO_PCI_CAP_NOTIFY_CFG: u8 = 2;
const VIRTIO_PCI_CAP_ISR_CFG: u8 = 3;
const VIRTIO_PCI_CAP_DEVICE_CFG: u8 = 4;
/// Capability structure: id, next, len, cfg_type, bar (+4), padding, offset u32 (+8), length
/// u32 (+12). The notify capability adds a u32 multiplier at +16, so it is 20 bytes, others 16.
const VIRTIO_CAP_LEN: u8 = 16;
const VIRTIO_NOTIFY_CAP_LEN: u8 = 20;
const VIRTIO_CAP_OFFSET_FIELD: usize = 8;
const VIRTIO_CAP_LENGTH_FIELD: usize = 12;
const VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD: usize = 16;
/// The capability chain is fixed: common -> notify -> ISR -> device, each placed right after the previous.
const CAP_COMMON_AT: usize = 0x40;
const CAP_NOTIFY_AT: usize = 0x50;
const CAP_ISR_AT: usize = 0x64;
const CAP_DEVICE_AT: usize = 0x74;
const CAP_MSIX_AT: usize = 0x84;

/// MSI-X capability: id 0x11, next, message control (table size - 1 in bits 10:0, function mask
/// bit 14, enable bit 15), table offset | BIR, PBA offset | BIR. Both live in BAR0 (BIR 0).
const CAP_ID_MSIX: u8 = 0x11;
const MSIX_CAP_CONTROL_FIELD: usize = 2;
const MSIX_CAP_TABLE_FIELD: usize = 4;
const MSIX_CAP_PBA_FIELD: usize = 8;
const MSIX_CONTROL_TABLE_SIZE_MASK: u16 = 0x7ff;
/// The control register's high byte holds the two writable bits.
const MSIX_CONTROL_HIGH_BYTE: usize = CAP_MSIX_AT + MSIX_CAP_CONTROL_FIELD + 1;
/// `struct virtio_pci_common_cfg` is 56 bytes.
const COMMON_CFG_LENGTH: u32 = 0x38;
const ISR_LENGTH: u32 = 4;
const DEVICE_CFG_LENGTH: u32 = 0x100;

/// One virtio device on bus 0. Config space is synthesized on every read from this state.
struct PciFunction {
    transport: Transport,
    devfn: u8,
    /// Preassigned at construction because there is no firmware to program it; the guest may relocate it.
    bar0: u32,
    command: u16,
    int_line: u8,
}

pub(crate) struct PciHost {
    /// Value of the address dword at 0xCF8.
    config_address: u32,
    funcs: Vec<PciFunction>,
    /// The AMD IOMMU function at 00:00.2 and the translation of every function's DMA.
    iommu: Option<Iommu>,
}

/// Fills in one virtio vendor capability at `at`; `next` is the config-space offset of the
/// following capability (0 ends the list). All regions live in BAR0 (the `bar` byte stays 0).
fn write_virtio_cap(
    config: &mut [u8; CONFIG_SPACE_SIZE],
    at: usize,
    next: usize,
    len: u8,
    cfg_type: u8,
    region_offset: u32,
    region_length: u32,
) {
    config[at] = CAP_ID_VENDOR_SPECIFIC;
    config[at + 1] = next as u8;
    config[at + 2] = len;
    config[at + 3] = cfg_type;
    config[at + 4] = 0;
    config[at + VIRTIO_CAP_OFFSET_FIELD..at + VIRTIO_CAP_OFFSET_FIELD + 4]
        .copy_from_slice(&region_offset.to_le_bytes());
    config[at + VIRTIO_CAP_LENGTH_FIELD..at + VIRTIO_CAP_LENGTH_FIELD + 4]
        .copy_from_slice(&region_length.to_le_bytes());
}

impl PciFunction {
    fn config(&self) -> [u8; CONFIG_SPACE_SIZE] {
        let mut b = [0u8; CONFIG_SPACE_SIZE];
        let device_type = self.transport.dev.device_type();
        let device_id = VIRTIO_PCI_DEVICE_ID_BASE + device_type;
        b[CFG_VENDOR_ID..CFG_VENDOR_ID + 2].copy_from_slice(&VENDOR_VIRTIO.to_le_bytes());
        b[CFG_DEVICE_ID..CFG_DEVICE_ID + 2].copy_from_slice(&device_id.to_le_bytes());
        b[CFG_COMMAND..CFG_COMMAND + 2].copy_from_slice(&self.command.to_le_bytes());
        b[CFG_STATUS..CFG_STATUS + 2].copy_from_slice(&STATUS_CAPABILITIES_LIST.to_le_bytes());
        b[CFG_REVISION] = VIRTIO_PCI_REVISION;
        let (subclass, class) = match device_type {
            DEVICE_TYPE_NET => (SUBCLASS_ETHERNET, CLASS_NETWORK),
            DEVICE_TYPE_BLOCK => (SUBCLASS_SCSI, CLASS_STORAGE),
            _ => (SUBCLASS_COMMUNICATION_OTHER, CLASS_COMMUNICATION),
        };
        b[CFG_SUBCLASS] = subclass;
        b[CFG_CLASS] = class;
        b[CFG_BAR0..CFG_BAR0 + 4].copy_from_slice(&self.bar0.to_le_bytes());
        b[CFG_SUBSYSTEM_VENDOR_ID..CFG_SUBSYSTEM_VENDOR_ID + 2]
            .copy_from_slice(&VENDOR_VIRTIO.to_le_bytes());
        b[CFG_SUBSYSTEM_ID..CFG_SUBSYSTEM_ID + 2].copy_from_slice(&device_type.to_le_bytes());
        b[CFG_CAPABILITIES_POINTER] = CAP_COMMON_AT as u8;
        b[CFG_INTERRUPT_LINE] = self.int_line;
        b[CFG_INTERRUPT_PIN] = INTERRUPT_PIN_INTA;
        write_virtio_cap(
            &mut b,
            CAP_COMMON_AT,
            CAP_NOTIFY_AT,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_COMMON_CFG,
            COMMON_CFG_OFFSET as u32,
            COMMON_CFG_LENGTH,
        );
        write_virtio_cap(
            &mut b,
            CAP_NOTIFY_AT,
            CAP_ISR_AT,
            VIRTIO_NOTIFY_CAP_LEN,
            VIRTIO_PCI_CAP_NOTIFY_CFG,
            NOTIFY_OFFSET as u32,
            REGION_SIZE as u32,
        );
        let multiplier_at = CAP_NOTIFY_AT + VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD;
        b[multiplier_at..multiplier_at + 4].copy_from_slice(&NOTIFY_OFF_MULTIPLIER.to_le_bytes());
        write_virtio_cap(
            &mut b,
            CAP_ISR_AT,
            CAP_DEVICE_AT,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_ISR_CFG,
            ISR_OFFSET as u32,
            ISR_LENGTH,
        );
        write_virtio_cap(
            &mut b,
            CAP_DEVICE_AT,
            CAP_MSIX_AT,
            VIRTIO_CAP_LEN,
            VIRTIO_PCI_CAP_DEVICE_CFG,
            DEVICE_CFG_OFFSET as u32,
            DEVICE_CFG_LENGTH,
        );
        b[CAP_MSIX_AT] = CAP_ID_MSIX;
        let control = (self.transport.msix_vectors() as u16 - 1) & MSIX_CONTROL_TABLE_SIZE_MASK
            | self.transport.msix_control();
        let control_at = CAP_MSIX_AT + MSIX_CAP_CONTROL_FIELD;
        b[control_at..control_at + 2].copy_from_slice(&control.to_le_bytes());
        let table_at = CAP_MSIX_AT + MSIX_CAP_TABLE_FIELD;
        b[table_at..table_at + 4].copy_from_slice(&(MSIX_TABLE_OFFSET as u32).to_le_bytes());
        let pba_at = CAP_MSIX_AT + MSIX_CAP_PBA_FIELD;
        b[pba_at..pba_at + 4].copy_from_slice(&(MSIX_PBA_OFFSET as u32).to_le_bytes());
        b
    }

    /// Byte-wise so sub-dword writes to the data port work. Only the command register, BAR0
    /// and the interrupt line are writable; everything else is read-only.
    fn write_config(&mut self, reg: usize, size: usize, val: u32) {
        for k in 0..size {
            let byte_reg = reg + k;
            let byte = (val >> (8 * k)) as u8;
            match byte_reg {
                CFG_COMMAND => self.command = (self.command & 0xff00) | byte as u16,
                CFG_COMMAND_HIGH_BYTE => {
                    self.command = (self.command & 0x00ff) | ((byte as u16) << 8)
                }
                CFG_BAR0..=CFG_BAR0_LAST_BYTE => {
                    let shift = 8 * (byte_reg - CFG_BAR0);
                    // Low bits are hardwired to zero: this is what makes the all-ones sizing
                    // write read back as !(BAR_SIZE - 1).
                    let size_mask = !((BAR_SIZE - 1) as u32);
                    self.bar0 = (self.bar0 & !(0xff << shift)) | ((byte as u32) << shift);
                    self.bar0 &= size_mask;
                }
                CFG_INTERRUPT_LINE => self.int_line = byte,
                MSIX_CONTROL_HIGH_BYTE => {
                    let control =
                        ((byte as u16) << 8) & (MSIX_CONTROL_ENABLE | MSIX_CONTROL_FUNCTION_MASK);
                    self.transport.set_msix_control(control);
                }
                _ => {}
            }
        }
    }

    fn memory_space_enabled(&self) -> bool {
        self.command & COMMAND_MEMORY_SPACE != 0
    }

    /// Offset into BAR0 if `phys_addr` is decoded by this function, honoring the command
    /// register's memory-space enable and an unassigned (zero) BAR.
    fn bar0_offset(&self, phys_addr: u64) -> Option<u64> {
        let base = self.bar0 as u64 & !(BAR_SIZE - 1);
        (self.memory_space_enabled()
            && base != 0
            && phys_addr >= base
            && phys_addr < base + BAR_SIZE)
            .then(|| phys_addr - base)
    }

    /// False when the guest set the command register's INTx-disable bit.
    fn intx_enabled(&self) -> bool {
        self.command & COMMAND_INTX_DISABLE == 0
    }
}

/// Moves the MSI-X messages a function sent into `messages`, tagged with its requester id.
fn collect_messages(function: &mut PciFunction, messages: &mut Vec<InterruptRequest>) {
    for (address, data) in function.transport.take_messages() {
        messages.push(InterruptRequest {
            requester: function.devfn as u16,
            address,
            data,
        });
    }
}

impl PciHost {
    /// The storage of the `index`-th virtio-blk function, in slot order (the order `into_disks`
    /// returns them in).
    #[cfg(not(freestanding))]
    pub fn disk_backend(&mut self, index: usize) -> Option<&mut dyn crate::disk::BlockBackend> {
        self.funcs
            .iter_mut()
            .filter_map(|function| function.transport.dev.block_backend())
            .nth(index)
    }

    /// Tears the bus down and returns the disks behind its virtio-blk functions, in slot order.
    pub fn into_disks(self) -> Vec<crate::disk::Disk> {
        self.funcs
            .into_iter()
            .filter_map(|function| function.transport.dev.into_disk())
            .collect()
    }

    /// At most `MAX_VIRTIO_FUNCTIONS` devices (the machine configuration rejects more): slot,
    /// BAR and interrupt line all follow the position in `devs`.
    pub fn new(devs: Vec<Transport>) -> Self {
        debug_assert!(devs.len() <= MAX_VIRTIO_FUNCTIONS);
        let funcs = devs
            .into_iter()
            .enumerate()
            .map(|(i, transport)| PciFunction {
                transport,
                devfn: ((i as u8 + FIRST_VIRTIO_SLOT) << DEVFN_SLOT_SHIFT),
                bar0: (MMIO_BASE + i as u64 * BAR_SIZE) as u32,
                command: 0,
                int_line: LEGACY_IRQS[i % LEGACY_IRQS.len()],
            })
            .collect();
        PciHost {
            config_address: 0,
            funcs,
            iommu: None,
        }
    }

    /// Adds the AMD IOMMU (`MachineConfig::iommu`).
    pub fn with_iommu(mut self, enabled: bool) -> Self {
        self.iommu = enabled.then(Iommu::new);
        for function in &mut self.funcs {
            function.transport.set_behind_iommu(enabled);
        }
        self
    }

    /// For the interrupt controllers: `Iommu::remap_interrupt` on every device interrupt.
    #[allow(dead_code)] // no interrupt controller calls it yet
    pub fn iommu(&mut self) -> Option<&mut Iommu> {
        self.iommu.as_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.funcs.is_empty() && self.iommu.is_none()
    }

    /// True once the driver of the virtio console at position `index` is up; when the UART shares
    /// that console's handle it stops taking host input then, so keystrokes go to hvc only.
    /// False for a position that is not a console function.
    pub fn console_active(&self, index: usize) -> bool {
        self.funcs.get(index).is_some_and(|f| {
            f.transport.dev.device_type() == DEVICE_TYPE_CONSOLE && f.transport.driver_ok()
        })
    }

    fn host_bridge(&self) -> [u8; CONFIG_SPACE_SIZE] {
        let mut b = [0u8; CONFIG_SPACE_SIZE];
        // The IOMMU is function 2 of the host bridge's slot, which the OS only scans if
        // function 0 says it is multifunction.
        if self.iommu.is_some() {
            b[CFG_HEADER_TYPE] = HEADER_TYPE_MULTIFUNCTION;
        }
        b[CFG_VENDOR_ID..CFG_VENDOR_ID + 2].copy_from_slice(&VENDOR_INTEL.to_le_bytes());
        b[CFG_DEVICE_ID..CFG_DEVICE_ID + 2].copy_from_slice(&DEVICE_I440FX_HOST.to_le_bytes());
        b[CFG_SUBCLASS] = SUBCLASS_HOST_BRIDGE;
        b[CFG_CLASS] = CLASS_BRIDGE;
        b
    }

    /// `byte` is the byte lane within the dword (low two bits of the data port). Absent
    /// devices read all ones.
    fn config_read(&self, addr: u32, size: usize, byte: usize) -> u32 {
        let (bus, devfn, reg) = (
            (addr >> CONFIG_ADDRESS_BUS_SHIFT) & 0xff,
            ((addr >> CONFIG_ADDRESS_DEVFN_SHIFT) & 0xff) as u8,
            (addr & CONFIG_ADDRESS_REGISTER_MASK) as usize,
        );
        let ones = size_mask(size as u8);
        if bus != 0 {
            return ones;
        }
        let config = if devfn == 0 {
            self.host_bridge()
        } else if let (Some(iommu), iommu::PCI_DEVFN) = (&self.iommu, devfn) {
            *iommu.config()
        } else if let Some(f) = self.funcs.iter().find(|f| f.devfn == devfn) {
            f.config()
        } else {
            return ones;
        };
        let dword = u32::from_le_bytes(config[reg..reg + 4].try_into().unwrap());
        (dword >> (8 * byte)) & ones
    }

    pub fn io_in(&self, port: u16, size: u8) -> u32 {
        let lane = (port.wrapping_sub(CONFIG_ADDRESS_PORT) & BYTE_LANE_MASK) as usize;
        if port < CONFIG_DATA_PORT {
            return (self.config_address >> (8 * lane)) & size_mask(size);
        }
        if self.config_address & CONFIG_ADDRESS_ENABLE == 0 {
            return size_mask(size);
        }
        self.config_read(self.config_address, size as usize, lane)
    }

    pub fn io_out(&mut self, port: u16, size: u8, val: u32) {
        let lane = (port.wrapping_sub(CONFIG_ADDRESS_PORT) & BYTE_LANE_MASK) as usize;
        // An access reaching past the dword is clamped; the rest would belong to the next port.
        let bytes = (size as usize).min(BYTES_PER_DWORD - lane);
        if port < CONFIG_DATA_PORT {
            // A narrow write replaces only its own bytes of the address register.
            let mask = (size_mask(bytes as u8) as u64) << (8 * lane);
            let merged = (self.config_address as u64 & !mask) | ((val as u64) << (8 * lane) & mask);
            self.config_address = merged as u32;
            return;
        }
        let bus = (self.config_address >> CONFIG_ADDRESS_BUS_SHIFT) & 0xff;
        if self.config_address & CONFIG_ADDRESS_ENABLE == 0 || bus != 0 {
            return;
        }
        let devfn = ((self.config_address >> CONFIG_ADDRESS_DEVFN_SHIFT) & 0xff) as u8;
        let reg = (self.config_address & CONFIG_ADDRESS_REGISTER_MASK) as usize + lane;
        self.config_write(devfn, reg, bytes, val);
    }

    fn config_write(&mut self, devfn: u8, reg: usize, bytes: usize, val: u32) {
        if let (Some(iommu), iommu::PCI_DEVFN) = (&mut self.iommu, devfn) {
            iommu.config_write(reg, bytes, val);
        } else if let Some(f) = self.funcs.iter_mut().find(|f| f.devfn == devfn) {
            f.write_config(reg, bytes, val);
        }
    }

    fn find_by_address(&mut self, phys_addr: u64) -> Option<(&mut PciFunction, u64)> {
        self.funcs
            .iter_mut()
            .find_map(|f| f.bar0_offset(phys_addr).map(|offset| (f, offset)))
    }

    /// Whether `phys_addr` is in the ECAM window of bus 0.
    pub fn ecam_decodes(phys_addr: u64) -> bool {
        (ECAM_BASE..ECAM_BASE + ECAM_BYTES_PER_BUS).contains(&phys_addr)
    }

    /// An ECAM read: the function's config space for registers below 256 (extended space reads
    /// zero, there are no extended capabilities) and all ones for an absent function, like the
    /// 0xCF8 mechanism. `size` is 1, 2 or 4 and the access may not cross a dword.
    pub fn ecam_read(&self, phys_addr: u64, size: u8) -> u32 {
        let offset = phys_addr - ECAM_BASE;
        let devfn = (offset / ECAM_FUNCTION_BYTES) as u32;
        let register = (offset % ECAM_FUNCTION_BYTES) as u32;
        let address = CONFIG_ADDRESS_ENABLE
            | devfn << CONFIG_ADDRESS_DEVFN_SHIFT
            | (register & CONFIG_ADDRESS_REGISTER_MASK);
        let lane = (register & 3) as usize;
        if register as usize >= CONFIG_SPACE_SIZE {
            return if self.function_exists(devfn as u8) {
                0
            } else {
                size_mask(size)
            };
        }
        self.config_read(address, size as usize, lane)
    }

    pub fn ecam_write(&mut self, phys_addr: u64, size: u8, value: u32) {
        let offset = phys_addr - ECAM_BASE;
        let devfn = (offset / ECAM_FUNCTION_BYTES) as u8;
        let register = (offset % ECAM_FUNCTION_BYTES) as usize;
        if register < CONFIG_SPACE_SIZE {
            self.config_write(devfn, register, size as usize, value);
        }
    }

    fn function_exists(&self, devfn: u8) -> bool {
        devfn == 0
            || (self.iommu.is_some() && devfn == iommu::PCI_DEVFN)
            || self.funcs.iter().any(|f| f.devfn == devfn)
    }

    /// IRQ numbers the functions can be wired to.
    pub fn interrupt_lines() -> [u8; LEGACY_IRQS.len()] {
        LEGACY_IRQS
    }

    /// Whether any function wired to `irq` is still asserting its interrupt: the ISR has a bit
    /// set (it clears on read) and the guest has not disabled INTx. The PIC needs this to drop a
    /// level-triggered line once the driver has acknowledged the device.
    pub fn line_asserted(&self, irq: u8) -> bool {
        self.funcs
            .iter()
            .any(|f| f.int_line == irq && f.intx_enabled() && f.transport.isr != 0)
    }

    /// None if no BAR decodes `phys_addr`.
    pub fn mmio_read(&mut self, phys_addr: u64, size: u8) -> Option<u32> {
        self.find_by_address(phys_addr)
            .map(|(f, offset)| f.transport.mmio_read(offset, size))
    }

    /// None if no BAR decodes `phys_addr`; otherwise the PIC IRQ to raise, or 0 for none. MSI-X
    /// messages the access caused are appended to `messages`.
    pub fn mmio_write_collecting(
        &mut self,
        phys_addr: u64,
        size: u8,
        val: u32,
        mem: &mut GuestMem,
        messages: &mut Vec<InterruptRequest>,
    ) -> Option<u8> {
        let PciHost { funcs, iommu, .. } = self;
        let (f, offset) = funcs
            .iter_mut()
            .find_map(|f| f.bar0_offset(phys_addr).map(|offset| (f, offset)))?;
        // The device's DMA is translated by the IOMMU as requester 00:slot.0.
        let raise = match iommu {
            Some(iommu) => mem.with_dma(iommu, f.devfn as u16, |mem| {
                f.transport.mmio_write(offset, size, val, mem)
            }),
            None => f.transport.mmio_write(offset, size, val, mem),
        };
        collect_messages(f, messages);
        if let Some(iommu) = iommu {
            iommu.service(mem);
        }
        Some(if raise && f.intx_enabled() {
            f.int_line
        } else {
            0
        })
    }

    /// Appends the IRQ of every function that has a new INTx interrupt to `raise` and the MSI-X
    /// messages of every function to `messages`.
    pub fn poll_collecting(
        &mut self,
        mem: &mut GuestMem,
        raise: &mut Vec<u8>,
        messages: &mut Vec<InterruptRequest>,
    ) {
        let PciHost { funcs, iommu, .. } = self;
        for f in funcs {
            let interrupted = match iommu {
                Some(iommu) => mem.with_dma(iommu, f.devfn as u16, |mem| f.transport.poll(mem)),
                None => f.transport.poll(mem),
            };
            if interrupted && f.intx_enabled() {
                raise.push(f.int_line);
            }
            collect_messages(f, messages);
        }
        if let Some(iommu) = iommu {
            iommu.service(mem);
        }
    }

    /// The IOMMU's register read, if its window decodes `phys_addr`.
    pub fn iommu_mmio_read(&self, phys_addr: u64, size: u8) -> Option<u64> {
        let iommu = self.iommu.as_ref()?;
        Some(iommu.mmio_read(iommu.mmio_offset(phys_addr)?, size))
    }

    /// The IOMMU's register write; false if its window does not decode `phys_addr`.
    pub fn iommu_mmio_write(
        &mut self,
        phys_addr: u64,
        size: u8,
        val: u64,
        mem: &mut GuestMem,
    ) -> bool {
        let Some(iommu) = self.iommu.as_mut() else {
            return false;
        };
        let Some(offset) = iommu.mmio_offset(phys_addr) else {
            return false;
        };
        iommu.mmio_write(offset, size, val, mem);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleHandle;
    use crate::dev::guestmem::test_mem;
    use crate::dev::vblk::VirtioBlk;
    use crate::dev::vcons::VirtioConsole;
    use crate::dev::vnet::VirtioNet;
    use crate::disk::Disk;
    use crate::netdev::NetHandle;

    impl PciHost {
        /// The accessors without the MSI-X outbox, which most tests do not look at.
        fn mmio_write(
            &mut self,
            phys_addr: u64,
            size: u8,
            val: u32,
            mem: &mut GuestMem,
        ) -> Option<u8> {
            self.mmio_write_collecting(phys_addr, size, val, mem, &mut Vec::new())
        }

        fn poll(&mut self, mem: &mut GuestMem, raise: &mut Vec<u8>) {
            self.poll_collecting(mem, raise, &mut Vec::new());
        }
    }

    /// Bytes of the MSI-X capability: id, next, control, table, PBA.
    const MSIX_CAP_LENGTH: u32 = 12;
    const PCI_SLOTS: u32 = 32;
    const PCI_FUNCTIONS: u32 = 8;
    const SLOT_SHIFT: u32 = 11;
    const FUNCTION_SHIFT: u32 = 8;
    const CONSOLE_SLOT: u32 = 1;
    const BLOCK_SLOT: u32 = 2;
    const NET_SLOT: u32 = 3;
    const NO_DEVICE_DWORD: u32 = u32::MAX;
    const BAR_SIZING_VALUE: u32 = !(BAR_SIZE as u32 - 1);
    /// Common config `num_queues`, a register whose value tells which device a BAR belongs to.
    const COMMON_NUM_QUEUES_OFFSET: u64 = 0x12;
    const COMMON_DEVICE_STATUS_OFFSET: u64 = 0x14;
    const STATUS_DRIVER_OK: u32 = 4;
    const CAPABILITY_ID_VENDOR: u8 = 9;
    const HEADER_TYPE_OFFSET: usize = 0x0e;
    const HEADER_TYPE_ENDPOINT: u32 = 0;

    fn console_transport() -> Transport {
        Transport::new(Box::new(VirtioConsole::new(ConsoleHandle::default())))
    }

    fn block_transport() -> Transport {
        Transport::new(Box::new(VirtioBlk::new(Disk::memory(vec![0; 4096]))))
    }

    fn net_transport() -> Transport {
        Transport::new(Box::new(VirtioNet::new(
            [2, 0, 0, 0, 0, 1],
            NetHandle::default(),
        )))
    }

    fn host() -> PciHost {
        PciHost::new(vec![console_transport()])
    }

    /// Console in slot 1, block in slot 2, net in slot 3, the order `Devices::new` uses.
    fn full_host() -> PciHost {
        PciHost::new(vec![
            console_transport(),
            block_transport(),
            net_transport(),
        ])
    }

    fn address(slot: u32, function: u32, reg: u32) -> u32 {
        CONFIG_ADDRESS_ENABLE | slot << SLOT_SHIFT | function << FUNCTION_SHIFT | reg
    }

    fn cfg_read(h: &mut PciHost, slot: u32, reg: u32, size: u8) -> u32 {
        h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, 0, reg & !3));
        h.io_in(CONFIG_DATA_PORT + (reg & 3) as u16, size)
    }

    fn cfg_read32(h: &mut PciHost, slot: u32, reg: u32) -> u32 {
        cfg_read(h, slot, reg, 4)
    }

    fn cfg_write(h: &mut PciHost, slot: u32, reg: u32, size: u8, value: u32) {
        h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, 0, reg & !3));
        h.io_out(CONFIG_DATA_PORT + (reg & 3) as u16, size, value);
    }

    fn enable_memory_space(h: &mut PciHost, slot: u32) {
        cfg_write(h, slot, CFG_COMMAND as u32, 2, COMMAND_MEMORY_SPACE as u32);
    }

    fn bar_base(h: &mut PciHost, slot: u32) -> u64 {
        cfg_read32(h, slot, CFG_BAR0 as u32) as u64
    }

    #[test]
    fn every_slot_and_function_is_enumerated_and_only_the_present_ones_answer() {
        let mut h = full_host();
        let mut present = Vec::new();
        for slot in 0..PCI_SLOTS {
            for function in 0..PCI_FUNCTIONS {
                h.io_out(CONFIG_ADDRESS_PORT, 4, address(slot, function, 0));
                let id = h.io_in(CONFIG_DATA_PORT, 4);
                if id != NO_DEVICE_DWORD {
                    present.push((slot, function, id));
                }
            }
        }
        assert_eq!(
            present,
            vec![
                (
                    0,
                    0,
                    VENDOR_INTEL as u32 | (DEVICE_I440FX_HOST as u32) << 16
                ),
                (1, 0, VENDOR_VIRTIO as u32 | 0x1043 << 16),
                (2, 0, VENDOR_VIRTIO as u32 | 0x1042 << 16),
                (3, 0, VENDOR_VIRTIO as u32 | 0x1041 << 16),
            ]
        );
    }

    #[test]
    fn absent_function_reads_all_ones_at_every_access_size() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 1, 0));
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), 0xffff);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 1), 0xff);
        assert_eq!(h.io_in(CONFIG_DATA_PORT + 3, 1), 0xff);
    }

    #[test]
    fn nonzero_bus_reads_all_ones_even_for_a_present_devfn() {
        let mut h = host();
        h.io_out(
            CONFIG_ADDRESS_PORT,
            4,
            address(CONSOLE_SLOT, 0, 0) | 1 << CONFIG_ADDRESS_BUS_SHIFT,
        );
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
    }

    #[test]
    fn config_data_reads_all_ones_while_enable_bit_is_clear() {
        let mut h = host();
        h.io_out(
            CONFIG_ADDRESS_PORT,
            4,
            address(CONSOLE_SLOT, 0, 0) & !CONFIG_ADDRESS_ENABLE,
        );
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), u32::MAX);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), 0xffff);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 1), 0xff);
    }

    #[test]
    fn config_data_write_is_ignored_while_enable_bit_is_clear() {
        let mut h = host();
        let disabled = address(CONSOLE_SLOT, 0, CFG_COMMAND as u32) & !CONFIG_ADDRESS_ENABLE;
        h.io_out(CONFIG_ADDRESS_PORT, 4, disabled);
        h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
    }

    #[test]
    fn config_write_to_nonzero_bus_is_ignored() {
        let mut h = host();
        h.io_out(
            CONFIG_ADDRESS_PORT,
            4,
            address(CONSOLE_SLOT, 0, CFG_COMMAND as u32) | 1 << CONFIG_ADDRESS_BUS_SHIFT,
        );
        h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
    }

    #[test]
    fn config_write_to_absent_slot_is_ignored() {
        let mut h = host();
        cfg_write(&mut h, NET_SLOT, CFG_COMMAND as u32, 2, 0xffff);
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
    }

    #[test]
    fn host_bridge_identifies_as_intel_440fx_host_bridge() {
        let mut h = host();
        assert_eq!(
            cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
            VENDOR_INTEL as u32
        );
        assert_eq!(
            cfg_read(&mut h, 0, CFG_DEVICE_ID as u32, 2),
            DEVICE_I440FX_HOST as u32
        );
        assert_eq!(
            cfg_read(&mut h, 0, CFG_CLASS as u32, 1),
            CLASS_BRIDGE as u32
        );
        assert_eq!(
            cfg_read(&mut h, 0, CFG_SUBCLASS as u32, 1),
            SUBCLASS_HOST_BRIDGE as u32
        );
        assert_eq!(cfg_read32(&mut h, 0, CFG_REVISION as u32) >> 16, 0x0600);
    }

    #[test]
    fn host_bridge_has_no_capabilities_bars_or_interrupt_pin() {
        let mut h = host();
        assert_eq!(cfg_read32(&mut h, 0, CFG_CAPABILITIES_POINTER as u32), 0);
        assert_eq!(cfg_read32(&mut h, 0, CFG_BAR0 as u32), 0);
        assert_eq!(cfg_read(&mut h, 0, CFG_INTERRUPT_PIN as u32, 1), 0);
    }

    #[test]
    fn host_bridge_config_is_read_only() {
        let mut h = host();
        cfg_write(&mut h, 0, CFG_COMMAND as u32, 2, 0xffff);
        cfg_write(&mut h, 0, CFG_VENDOR_ID as u32, 4, 0);
        assert_eq!(cfg_read(&mut h, 0, CFG_COMMAND as u32, 2), 0);
        assert_eq!(
            cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
            VENDOR_INTEL as u32
        );
    }

    #[test]
    fn host_bridge_is_present_even_with_no_virtio_devices() {
        let mut h = PciHost::new(Vec::new());
        assert!(h.is_empty());
        assert_eq!(
            cfg_read(&mut h, 0, CFG_VENDOR_ID as u32, 2),
            VENDOR_INTEL as u32
        );
        assert_eq!(cfg_read32(&mut h, 1, CFG_VENDOR_ID as u32), u32::MAX);
    }

    #[test]
    fn host_is_not_empty_with_a_device() {
        assert!(!host().is_empty());
    }

    #[test]
    fn virtio_functions_report_vendor_and_modern_device_ids() {
        let mut h = full_host();
        for (slot, device_type) in [
            (CONSOLE_SLOT, DEVICE_TYPE_CONSOLE),
            (BLOCK_SLOT, DEVICE_TYPE_BLOCK),
            (NET_SLOT, DEVICE_TYPE_NET),
        ] {
            assert_eq!(cfg_read(&mut h, slot, CFG_VENDOR_ID as u32, 2), 0x1af4);
            assert_eq!(
                cfg_read(&mut h, slot, CFG_DEVICE_ID as u32, 2),
                0x1040 + device_type as u32
            );
            assert_eq!(
                cfg_read(&mut h, slot, CFG_SUBSYSTEM_VENDOR_ID as u32, 2),
                0x1af4
            );
            assert_eq!(
                cfg_read(&mut h, slot, CFG_SUBSYSTEM_ID as u32, 2),
                device_type as u32
            );
        }
    }

    #[test]
    fn virtio_functions_are_revision_one_endpoints() {
        let mut h = full_host();
        for slot in CONSOLE_SLOT..=NET_SLOT {
            assert_eq!(cfg_read(&mut h, slot, CFG_REVISION as u32, 1), 1);
            assert_eq!(
                cfg_read(&mut h, slot, HEADER_TYPE_OFFSET as u32, 1),
                HEADER_TYPE_ENDPOINT
            );
            assert_eq!(cfg_read(&mut h, slot, 0x09, 1), 0, "programming interface");
        }
    }

    #[test]
    fn class_codes_follow_qemu() {
        let mut h = full_host();
        let class_of = |h: &mut PciHost, slot| cfg_read32(h, slot, CFG_REVISION as u32) >> 16;
        assert_eq!(class_of(&mut h, CONSOLE_SLOT), 0x0780);
        assert_eq!(class_of(&mut h, BLOCK_SLOT), 0x0100);
        assert_eq!(class_of(&mut h, NET_SLOT), 0x0200);
    }

    #[test]
    fn bar0_starts_at_the_mmio_window_one_bar_apart_and_is_memory_type() {
        let mut h = full_host();
        for (index, slot) in [CONSOLE_SLOT, BLOCK_SLOT, NET_SLOT].into_iter().enumerate() {
            let bar = bar_base(&mut h, slot);
            assert_eq!(bar, MMIO_BASE + index as u64 * BAR_SIZE);
            assert_eq!(bar & 0xf, 0, "32-bit non-prefetchable memory BAR");
        }
    }

    #[test]
    fn other_bars_read_zero() {
        let mut h = host();
        for reg in (CFG_BAR0 as u32 + 4..0x28).step_by(4) {
            assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "BAR at {reg:#x}");
        }
    }

    #[test]
    fn bar_sizing_with_dword_all_ones_reports_the_size_mask() {
        let mut h = host();
        let original = bar_base(&mut h, CONSOLE_SLOT) as u32;
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
        let sized = cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32);
        assert_eq!(sized, BAR_SIZING_VALUE);
        assert_eq!(!sized + 1, BAR_SIZE as u32, "size is the lowest set bit");
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, original);
        assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32), original);
    }

    #[test]
    fn bar_sizing_by_bytes_also_works() {
        let mut h = host();
        for lane in 0..4 {
            cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + lane, 1, 0xff);
        }
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            BAR_SIZING_VALUE
        );
    }

    #[test]
    fn bar_sizing_by_words_also_works() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 2, 0xffff);
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 2, 2, 0xffff);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            BAR_SIZING_VALUE
        );
    }

    #[test]
    fn bar_low_bits_below_the_size_are_hardwired_to_zero() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0xd000_7fff);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            0xd000_0000
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0xd000_8001);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            0xd000_8000
        );
    }

    #[test]
    fn bar_sizing_does_not_change_other_devices_bars() {
        let mut h = full_host();
        let before = bar_base(&mut h, BLOCK_SLOT);
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
        assert_eq!(bar_base(&mut h, BLOCK_SLOT), before);
    }

    #[test]
    fn memory_space_disabled_means_bar_is_not_decoded() {
        let mut h = host();
        assert!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
                .is_none()
        );
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        assert_eq!(h.mmio_write(MMIO_BASE, 4, 0, &mut mem), None);
    }

    #[test]
    fn enabling_memory_space_decodes_bar0() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        assert_eq!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2),
            Some(2)
        );
    }

    #[test]
    fn disabling_memory_space_again_stops_decoding() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 0);
        assert!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
                .is_none()
        );
    }

    #[test]
    fn io_space_bit_alone_does_not_enable_mmio_decoding() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 1);
        assert!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
                .is_none()
        );
    }

    #[test]
    fn bar_decode_covers_exactly_bar_size_bytes() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        assert!(h.mmio_read(MMIO_BASE + BAR_SIZE - 1, 1).is_some());
        assert!(h.mmio_read(MMIO_BASE + BAR_SIZE, 1).is_none());
        assert!(h.mmio_read(MMIO_BASE - 1, 1).is_none());
    }

    #[test]
    fn each_device_decodes_only_its_own_bar() {
        let mut h = full_host();
        for slot in [CONSOLE_SLOT, BLOCK_SLOT, NET_SLOT] {
            enable_memory_space(&mut h, slot);
        }
        // num_queues: console 2, block 1, net 2.
        assert_eq!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2),
            Some(2)
        );
        assert_eq!(
            h.mmio_read(MMIO_BASE + BAR_SIZE + COMMON_NUM_QUEUES_OFFSET, 2),
            Some(1)
        );
        assert_eq!(
            h.mmio_read(MMIO_BASE + 2 * BAR_SIZE + COMMON_NUM_QUEUES_OFFSET, 2),
            Some(2)
        );
        assert!(h.mmio_read(MMIO_BASE + 3 * BAR_SIZE, 1).is_none());
    }

    #[test]
    fn relocating_bar0_moves_the_decode_window() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let new_base = 0xd000_0000u32;
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, new_base);
        assert!(
            h.mmio_read(MMIO_BASE + COMMON_NUM_QUEUES_OFFSET, 2)
                .is_none()
        );
        assert_eq!(
            h.mmio_read(new_base as u64 + COMMON_NUM_QUEUES_OFFSET, 2),
            Some(2)
        );
    }

    #[test]
    fn bar_programmed_to_zero_is_not_decoded_even_with_memory_space_on() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, 0);
        assert!(h.mmio_read(COMMON_NUM_QUEUES_OFFSET, 2).is_none());
    }

    #[test]
    fn sizing_a_bar_with_decode_disabled_leaves_the_decode_off() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32, 4, u32::MAX);
        assert!(
            h.mmio_read(BAR_SIZING_VALUE as u64 + COMMON_NUM_QUEUES_OFFSET, 2)
                .is_none()
        );
    }

    #[test]
    fn command_register_stores_written_bits_and_status_is_constant() {
        let mut h = host();
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2), 0);
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, 0x0406);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
            0x0406
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 1, 0x02);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
            0x0402
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32 + 1, 1, 0x00);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2),
            0x0002
        );
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2),
            STATUS_CAPABILITIES_LIST as u32
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2, 0xffff);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2),
            STATUS_CAPABILITIES_LIST as u32
        );
    }

    #[test]
    fn dword_write_at_command_register_updates_command_and_ignores_status() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 4, 0xffff_0002);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32),
            0x0010_0002
        );
    }

    #[test]
    fn id_class_and_subsystem_registers_are_read_only() {
        let mut h = host();
        let before: Vec<u32> = (0..0x40)
            .step_by(4)
            .map(|reg| cfg_read32(&mut h, CONSOLE_SLOT, reg))
            .collect();
        for reg in [0x00, 0x08, 0x0c, 0x2c, 0x34] {
            cfg_write(&mut h, CONSOLE_SLOT, reg, 4, 0xa5a5_a5a5);
        }
        let after: Vec<u32> = (0..0x40)
            .step_by(4)
            .map(|reg| cfg_read32(&mut h, CONSOLE_SLOT, reg))
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn capability_list_is_flagged_in_status_and_pointer_is_dword_aligned() {
        let mut h = host();
        assert_ne!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_STATUS as u32, 2) & STATUS_CAPABILITIES_LIST as u32,
            0
        );
        let pointer = cfg_read(&mut h, CONSOLE_SLOT, CFG_CAPABILITIES_POINTER as u32, 1);
        assert_eq!(pointer, CAP_COMMON_AT as u32);
        assert_eq!(pointer % 4, 0);
    }

    #[derive(Debug, PartialEq)]
    struct Capability {
        at: u32,
        id: u8,
        next: u8,
        len: u8,
        cfg_type: u8,
        bar: u8,
        region_offset: u32,
        region_length: u32,
    }

    /// The vendor-specific (virtio) capabilities, found by following the whole chain (which
    /// also holds the MSI-X capability at its end).
    fn walk_capabilities(h: &mut PciHost, slot: u32) -> Vec<Capability> {
        let mut caps = Vec::new();
        let mut at = cfg_read(h, slot, CFG_CAPABILITIES_POINTER as u32, 1);
        // A well-formed list in 256 bytes of config space cannot have more than 48 entries.
        for _ in 0..48 {
            if at == 0 {
                return caps;
            }
            let header = cfg_read32(h, slot, at);
            if header as u8 == CAPABILITY_ID_VENDOR {
                caps.push(Capability {
                    at,
                    id: header as u8,
                    next: (header >> 8) as u8,
                    len: (header >> 16) as u8,
                    cfg_type: (header >> 24) as u8,
                    bar: cfg_read(h, slot, at + 4, 1) as u8,
                    region_offset: cfg_read32(h, slot, at + VIRTIO_CAP_OFFSET_FIELD as u32),
                    region_length: cfg_read32(h, slot, at + VIRTIO_CAP_LENGTH_FIELD as u32),
                });
            }
            at = (header >> 8) & 0xff;
        }
        panic!("capability list does not terminate");
    }

    #[test]
    fn capability_chain_has_four_vendor_capabilities_and_terminates() {
        let mut h = host();
        let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
        assert_eq!(caps.len(), 4);
        assert_eq!(
            caps.last().unwrap().next as usize,
            CAP_MSIX_AT,
            "the virtio capabilities are followed by MSI-X"
        );
        assert!(caps.iter().all(|cap| cap.id == CAPABILITY_ID_VENDOR));
        assert!(caps.iter().all(|cap| cap.at % 4 == 0 && cap.at >= 0x40));
    }

    #[test]
    fn msix_capability_ends_the_chain_and_points_at_the_table_and_pba_in_bar0() {
        let mut h = host();
        let at = CAP_MSIX_AT as u32;
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, at, 2),
            0x0011,
            "id, next = 0"
        );
        // The console has two queues and one configuration vector: three table entries.
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, at + 2, 2), 2);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, at + 4),
            MSIX_TABLE_OFFSET as u32
        );
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, at + 8),
            MSIX_PBA_OFFSET as u32
        );
        const { assert!(MSIX_PBA_OFFSET + REGION_SIZE <= BAR_SIZE) };
    }

    #[test]
    fn msix_enable_and_function_mask_are_the_only_writable_control_bits() {
        let mut h = host();
        let control = CAP_MSIX_AT as u32 + 2;
        cfg_write(&mut h, CONSOLE_SLOT, control, 2, 0xffff);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, control, 2),
            (MSIX_CONTROL_ENABLE | MSIX_CONTROL_FUNCTION_MASK) as u32 | 2
        );
        cfg_write(&mut h, CONSOLE_SLOT, control, 2, 0);
        assert_eq!(cfg_read(&mut h, CONSOLE_SLOT, control, 2), 2);
    }

    #[test]
    fn capability_types_regions_and_lengths_match_the_bar_layout() {
        let mut h = host();
        let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
        let summary: Vec<(u8, u8, u32, u32)> = caps
            .iter()
            .map(|c| (c.cfg_type, c.bar, c.region_offset, c.region_length))
            .collect();
        assert_eq!(
            summary,
            vec![
                (1, 0, 0x0000, 0x38),
                (2, 0, 0x3000, 0x1000),
                (3, 0, 0x1000, 4),
                (4, 0, 0x2000, 0x100),
            ]
        );
    }

    #[test]
    fn capability_lengths_are_sixteen_bytes_and_twenty_for_notify() {
        let mut h = host();
        let lengths: Vec<(u8, u8)> = walk_capabilities(&mut h, CONSOLE_SLOT)
            .iter()
            .map(|c| (c.cfg_type, c.len))
            .collect();
        assert_eq!(lengths, vec![(1, 16), (2, 20), (3, 16), (4, 16)]);
    }

    #[test]
    fn capabilities_do_not_overlap_each_other() {
        let mut h = host();
        let caps = walk_capabilities(&mut h, CONSOLE_SLOT);
        for pair in caps.windows(2) {
            assert!(pair[0].at + pair[0].len as u32 <= pair[1].at, "{pair:?}");
        }
        let last = caps.last().unwrap();
        assert!(last.at + last.len as u32 <= 256);
    }

    #[test]
    fn notify_capability_carries_the_queue_notify_multiplier() {
        let mut h = host();
        let notify = walk_capabilities(&mut h, CONSOLE_SLOT)
            .into_iter()
            .find(|cap| cap.cfg_type == 2)
            .unwrap();
        let multiplier = cfg_read32(
            &mut h,
            CONSOLE_SLOT,
            notify.at + VIRTIO_CAP_NOTIFY_MULTIPLIER_FIELD as u32,
        );
        assert_eq!(multiplier, NOTIFY_OFF_MULTIPLIER);
        // Every queue's notify address must stay inside the advertised notify region.
        let queue_count = 2;
        assert!(
            notify.region_offset + (queue_count - 1) * multiplier + 2
                <= notify.region_offset + notify.region_length
        );
    }

    #[test]
    fn capability_regions_fit_in_the_bar_and_do_not_overlap() {
        let mut h = host();
        let mut regions: Vec<(u32, u32)> = walk_capabilities(&mut h, CONSOLE_SLOT)
            .iter()
            .map(|cap| (cap.region_offset, cap.region_length))
            .collect();
        regions.sort();
        for pair in regions.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{pair:?}");
        }
        assert!(
            regions
                .iter()
                .all(|&(start, len)| (start + len) as u64 <= BAR_SIZE)
        );
    }

    #[test]
    fn capability_chain_is_identical_for_every_device_type() {
        let mut h = full_host();
        let console = walk_capabilities(&mut h, CONSOLE_SLOT);
        assert_eq!(walk_capabilities(&mut h, BLOCK_SLOT), console);
        assert_eq!(walk_capabilities(&mut h, NET_SLOT), console);
    }

    #[test]
    fn bytes_past_the_last_capability_read_zero() {
        let mut h = host();
        for reg in (CAP_MSIX_AT as u32 + MSIX_CAP_LENGTH..CONFIG_SPACE_SIZE as u32).step_by(4) {
            assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "offset {reg:#x}");
        }
    }

    #[test]
    fn unimplemented_registers_of_a_present_device_read_zero_not_all_ones() {
        let mut h = host();
        for (reg, what) in [
            (0x28, "cardbus cis pointer"),
            (0x30, "expansion rom base"),
            (0x38, "reserved"),
        ] {
            assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, reg), 0, "{what}");
        }
    }

    #[test]
    fn interrupt_line_and_pin_follow_the_legacy_irq_assignment() {
        let mut h = full_host();
        for (slot, irq) in [(CONSOLE_SLOT, 9), (BLOCK_SLOT, 10), (NET_SLOT, 11)] {
            assert_eq!(cfg_read(&mut h, slot, CFG_INTERRUPT_LINE as u32, 1), irq);
            assert_eq!(
                cfg_read(&mut h, slot, CFG_INTERRUPT_PIN as u32, 1),
                INTERRUPT_PIN_INTA as u32
            );
        }
        let dword = cfg_read32(&mut h, NET_SLOT, CFG_INTERRUPT_LINE as u32);
        assert_eq!(
            dword & 0xffff,
            0x0100 | 11,
            "min_gnt/max_lat bytes above are zero"
        );
    }

    #[test]
    fn interrupt_line_is_writable_and_pin_is_read_only() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_LINE as u32, 1, 5);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_LINE as u32, 1),
            5
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_PIN as u32, 1, 3);
        assert_eq!(
            cfg_read(&mut h, CONSOLE_SLOT, CFG_INTERRUPT_PIN as u32, 1),
            1
        );
    }

    #[test]
    fn irq_assignment_wraps_after_four_devices_and_never_uses_zero() {
        let h = PciHost::new(vec![
            console_transport(),
            console_transport(),
            console_transport(),
            console_transport(),
        ]);
        let lines: Vec<u8> = h.funcs.iter().map(|f| f.int_line).collect();
        assert_eq!(lines, vec![9, 10, 11, 5]);
        let fifth = PciHost::new((0..5).map(|_| console_transport()).collect());
        assert_eq!(fifth.funcs[4].int_line, 9);
        assert!(LEGACY_IRQS.iter().all(|&irq| irq != 0));
    }

    #[test]
    fn devfn_is_slot_times_eight_starting_at_slot_one() {
        let h = full_host();
        let devfns: Vec<u8> = h.funcs.iter().map(|f| f.devfn).collect();
        assert_eq!(devfns, vec![8, 16, 24]);
    }

    #[test]
    fn address_register_reads_back_what_was_written() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8000_1234);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_1234);
    }

    #[test]
    fn address_register_byte_and_word_reads_use_the_lane_of_the_port() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8112_3456);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 1), 0x56);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 1, 1), 0x34);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 2, 1), 0x12);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 3, 1), 0x81);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 2), 0x3456);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT + 2, 2), 0x8112);
    }

    #[test]
    fn address_register_byte_writes_merge_into_their_lane() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0x1111_1111);
        h.io_out(CONFIG_ADDRESS_PORT + 3, 1, 0x80);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8011_1111);
        h.io_out(CONFIG_ADDRESS_PORT + 1, 1, 0x22);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8011_2211);
        h.io_out(CONFIG_ADDRESS_PORT + 2, 1, 0x33);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8033_2211);
        h.io_out(CONFIG_ADDRESS_PORT, 1, 0x44);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8033_2244);
    }

    #[test]
    fn address_register_word_writes_merge_into_their_half() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0xaaaa_bbbb);
        h.io_out(CONFIG_ADDRESS_PORT, 2, 0x1234);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xaaaa_1234);
        h.io_out(CONFIG_ADDRESS_PORT + 2, 2, 0x8000);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_1234);
    }

    #[test]
    fn word_write_ignores_the_value_bits_above_the_access_size() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0);
        h.io_out(CONFIG_ADDRESS_PORT, 2, 0xdead_beef);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xbeef);
        h.io_out(CONFIG_ADDRESS_PORT, 1, 0xffff_ff01);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0xbe01);
    }

    #[test]
    fn byte_write_to_address_register_keeps_the_other_bytes() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, 0x8000_0810);
        h.io_out(CONFIG_ADDRESS_PORT, 1, 0x14);
        assert_eq!(h.io_in(CONFIG_ADDRESS_PORT, 4), 0x8000_0814);
    }

    #[test]
    fn config_register_number_ignores_the_low_address_bits() {
        let mut h = host();
        h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 0, 0x03));
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), VENDOR_VIRTIO as u32);
    }

    #[test]
    fn data_window_byte_lanes_select_bytes_of_the_dword() {
        let mut h = host();
        let dword = cfg_read32(&mut h, CONSOLE_SLOT, CFG_VENDOR_ID as u32);
        for lane in 0..4u16 {
            assert_eq!(
                h.io_in(CONFIG_DATA_PORT + lane, 1),
                (dword >> (8 * lane)) & 0xff,
                "lane {lane}"
            );
        }
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), dword & 0xffff);
        assert_eq!(h.io_in(CONFIG_DATA_PORT + 2, 2), dword >> 16);
        assert_eq!(h.io_in(CONFIG_DATA_PORT + 1, 2), (dword >> 8) & 0xffff);
    }

    #[test]
    fn data_window_dword_read_at_lane_one_is_zero_extended_not_wrapped() {
        let mut h = host();
        let dword = cfg_read32(&mut h, CONSOLE_SLOT, CFG_VENDOR_ID as u32);
        assert_eq!(h.io_in(CONFIG_DATA_PORT + 1, 4), dword >> 8);
    }

    #[test]
    fn data_window_byte_write_changes_only_that_byte() {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 3, 1, 0xd0);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            0xd000_0000
        );
        cfg_write(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32 + 2, 1, 0x40);
        assert_eq!(
            cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32),
            0xd040_0000
        );
    }

    #[test]
    fn data_window_access_reaching_past_the_dword_does_not_spill_into_the_next_register() {
        let mut h = host();
        let before = cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32);
        h.io_out(CONFIG_ADDRESS_PORT, 4, address(CONSOLE_SLOT, 0, 0x0c));
        // Lane 3 of register 0x0c is byte 0x0f; bytes 1..3 of the value would land on BAR0.
        h.io_out(CONFIG_DATA_PORT + 3, 4, 0x12c0_0000);
        assert_eq!(cfg_read32(&mut h, CONSOLE_SLOT, CFG_BAR0 as u32), before);
    }

    #[test]
    fn memory_and_config_accesses_do_not_interfere_with_each_other() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        h.io_out(
            CONFIG_ADDRESS_PORT,
            4,
            address(CONSOLE_SLOT, 0, CFG_COMMAND as u32),
        );
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        h.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK,
            &mut mem,
        );
        assert_eq!(
            h.io_in(CONFIG_ADDRESS_PORT, 4),
            address(CONSOLE_SLOT, 0, CFG_COMMAND as u32)
        );
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 2), COMMAND_MEMORY_SPACE as u32);
    }

    #[test]
    fn console_is_active_only_after_the_driver_sets_driver_ok() {
        let mut h = host();
        assert!(!h.console_active(0));
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        h.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK,
            &mut mem,
        );
        assert!(h.console_active(0));
        h.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 0, &mut mem);
        assert!(!h.console_active(0), "a reset deactivates it");
    }

    #[test]
    fn only_a_console_function_counts_for_console_active() {
        let mut h = PciHost::new(vec![block_transport()]);
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        h.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK,
            &mut mem,
        );
        assert!(!h.console_active(0));
    }

    fn host_with_pending_config_event() -> (PciHost, ConsoleHandle) {
        let console = ConsoleHandle::default();
        let mut h = PciHost::new(vec![Transport::new(Box::new(VirtioConsole::new(
            console.clone(),
        )))]);
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        h.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK,
            &mut mem,
        );
        console.resize(100, 40);
        (h, console)
    }

    #[test]
    fn poll_reports_the_function_interrupt_line() {
        let (mut h, _console) = host_with_pending_config_event();
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        let mut raise = Vec::new();
        h.poll(&mut mem, &mut raise);
        assert_eq!(raise, vec![9]);
    }

    #[test]
    fn poll_reports_nothing_when_intx_is_disabled() {
        let (mut h, _console) = host_with_pending_config_event();
        cfg_write(
            &mut h,
            CONSOLE_SLOT,
            CFG_COMMAND as u32,
            2,
            (COMMAND_MEMORY_SPACE | COMMAND_INTX_DISABLE) as u32,
        );
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        let mut raise = Vec::new();
        h.poll(&mut mem, &mut raise);
        assert!(raise.is_empty());
    }

    #[test]
    fn poll_with_nothing_to_report_adds_no_irqs() {
        let mut h = full_host();
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        let mut raise = Vec::new();
        h.poll(&mut mem, &mut raise);
        assert!(raise.is_empty());
    }

    #[test]
    fn mmio_write_that_raises_nothing_returns_zero_not_none() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        let result = h.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK,
            &mut mem,
        );
        assert_eq!(result, Some(0));
    }

    const COMMON_QUEUE_SELECT_OFFSET: u64 = 0x16;
    const COMMON_QUEUE_SIZE_OFFSET: u64 = 0x18;
    const COMMON_QUEUE_ENABLE_OFFSET: u64 = 0x1c;
    const COMMON_QUEUE_DESC_OFFSET: u64 = 0x20;
    const COMMON_QUEUE_DRIVER_OFFSET: u64 = 0x28;
    const COMMON_QUEUE_DEVICE_OFFSET: u64 = 0x30;
    const NOTIFY_REGION_OFFSET: u64 = 0x3000;
    const TRANSMIT_QUEUE: u64 = 1;
    const TX_DESC: u64 = 0x100;
    const TX_AVAIL: u64 = 0x200;
    const TX_USED: u64 = 0x300;
    const TX_BUFFER: u64 = 0x400;

    /// Console with its transmit queue programmed through BAR0 and one buffer ready; the next
    /// kick completes it.
    fn console_with_transmit_ready(command: u16) -> (PciHost, Vec<u8>) {
        let mut h = host();
        cfg_write(&mut h, CONSOLE_SLOT, CFG_COMMAND as u32, 2, command as u32);
        let mut ram = vec![0u8; 0x1000];
        ram[TX_BUFFER as usize] = b'x';
        ram[TX_DESC as usize..TX_DESC as usize + 8].copy_from_slice(&TX_BUFFER.to_le_bytes());
        ram[TX_DESC as usize + 8..TX_DESC as usize + 12].copy_from_slice(&1u32.to_le_bytes());
        ram[TX_AVAIL as usize + 2] = 1;
        let mut mem = test_mem(&mut ram);
        let mut write = |offset: u64, size: u8, value: u32| {
            h.mmio_write(MMIO_BASE + offset, size, value, &mut mem);
        };
        write(COMMON_QUEUE_SELECT_OFFSET, 2, TRANSMIT_QUEUE as u32);
        write(COMMON_QUEUE_SIZE_OFFSET, 2, 8);
        write(COMMON_QUEUE_DESC_OFFSET, 4, TX_DESC as u32);
        write(COMMON_QUEUE_DRIVER_OFFSET, 4, TX_AVAIL as u32);
        write(COMMON_QUEUE_DEVICE_OFFSET, 4, TX_USED as u32);
        write(COMMON_QUEUE_ENABLE_OFFSET, 2, 1);
        write(COMMON_DEVICE_STATUS_OFFSET, 1, STATUS_DRIVER_OK);
        (h, ram)
    }

    fn kick_transmit(h: &mut PciHost, ram: &mut [u8]) -> Option<u8> {
        let mut mem = test_mem(ram);
        h.mmio_write(
            MMIO_BASE + NOTIFY_REGION_OFFSET + TRANSMIT_QUEUE * NOTIFY_OFF_MULTIPLIER as u64,
            2,
            TRANSMIT_QUEUE as u32,
            &mut mem,
        )
    }

    #[test]
    fn mmio_write_that_completes_a_buffer_returns_the_function_irq() {
        let (mut h, mut ram) = console_with_transmit_ready(COMMAND_MEMORY_SPACE);
        assert_eq!(kick_transmit(&mut h, &mut ram), Some(9));
    }

    #[test]
    fn mmio_write_that_completes_a_buffer_returns_zero_when_intx_is_disabled() {
        let (mut h, mut ram) =
            console_with_transmit_ready(COMMAND_MEMORY_SPACE | COMMAND_INTX_DISABLE);
        assert_eq!(kick_transmit(&mut h, &mut ram), Some(0));
    }

    #[test]
    fn mmio_write_to_unmapped_address_returns_none() {
        let mut h = host();
        enable_memory_space(&mut h, CONSOLE_SLOT);
        let mut ram = vec![0u8; 0x1000];
        let mut mem = test_mem(&mut ram);
        assert_eq!(h.mmio_write(MMIO_BASE + BAR_SIZE, 4, 0, &mut mem), None);
    }

    fn config_address(slot: u32, reg: u32) -> u32 {
        CONFIG_ADDRESS_ENABLE | slot << (CONFIG_ADDRESS_DEVFN_SHIFT + DEVFN_SLOT_SHIFT) | reg
    }

    fn rd(h: &mut PciHost, slot: u32, reg: u32) -> u32 {
        h.io_out(CONFIG_ADDRESS_PORT, 4, config_address(slot, reg));
        h.io_in(CONFIG_DATA_PORT, 4)
    }

    #[test]
    fn enumerate_and_size_bar() {
        let mut h = host();
        assert_eq!(
            rd(&mut h, 0, CFG_VENDOR_ID as u32) & 0xffff,
            VENDOR_INTEL as u32
        );
        // Dword at 8 is revision, prog_if, subclass, class: the upper half is the class code.
        assert_eq!(rd(&mut h, 0, CFG_REVISION as u32) >> 16, 0x0600);
        assert_eq!(
            rd(&mut h, 1, CFG_VENDOR_ID as u32),
            VENDOR_VIRTIO as u32 | ((VIRTIO_PCI_DEVICE_ID_BASE + DEVICE_TYPE_CONSOLE) as u32) << 16
        );
        assert_eq!(rd(&mut h, 2, CFG_VENDOR_ID as u32), u32::MAX);
        let bar = rd(&mut h, 1, CFG_BAR0 as u32);
        h.io_out(CONFIG_ADDRESS_PORT, 4, config_address(1, CFG_BAR0 as u32));
        h.io_out(CONFIG_DATA_PORT, 4, u32::MAX);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), !(BAR_SIZE as u32 - 1));
        h.io_out(CONFIG_DATA_PORT, 4, bar);
        assert_eq!(h.io_in(CONFIG_DATA_PORT, 4), bar);
        let mut caps = Vec::new();
        let mut p = rd(&mut h, 1, CFG_CAPABILITIES_POINTER as u32) & 0xff;
        while p != 0 {
            let w = rd(&mut h, 1, p);
            if w as u8 == CAP_ID_VENDOR_SPECIFIC {
                caps.push((w >> 24) as u8);
            }
            p = (w >> 8) & 0xff;
        }
        assert_eq!(
            caps,
            vec![
                VIRTIO_PCI_CAP_COMMON_CFG,
                VIRTIO_PCI_CAP_NOTIFY_CFG,
                VIRTIO_PCI_CAP_ISR_CFG,
                VIRTIO_PCI_CAP_DEVICE_CFG
            ]
        );
    }

    #[test]
    fn bar_decode_follows_command() {
        let mut h = host();
        assert!(h.mmio_read(MMIO_BASE + 0x14, 1).is_none());
        h.io_out(
            CONFIG_ADDRESS_PORT,
            4,
            config_address(1, CFG_COMMAND as u32),
        );
        h.io_out(CONFIG_DATA_PORT, 2, COMMAND_MEMORY_SPACE as u32);
        // BAR0 offset 0x12 is the common config num_queues field; the console has 2 queues.
        assert_eq!(h.mmio_read(MMIO_BASE + 0x12, 2), Some(2));
    }

    #[test]
    fn a_line_is_asserted_only_while_a_function_wired_to_it_has_isr_bits_set() {
        let mut host = host();
        let line = host.funcs[0].int_line;
        assert!(!host.line_asserted(line), "a fresh device asserts nothing");
        host.funcs[0].transport.isr = 1;
        assert!(host.line_asserted(line));
        for other in PciHost::interrupt_lines() {
            if other != line {
                assert!(
                    !host.line_asserted(other),
                    "only the wired line is asserted"
                );
            }
        }
        host.funcs[0].transport.isr = 0;
        assert!(
            !host.line_asserted(line),
            "reading the ISR register releases the line"
        );
    }

    fn many_functions(count: usize) -> PciHost {
        PciHost::new((0..count).map(|_| net_transport()).collect())
    }

    #[test]
    fn functions_get_distinct_bars_and_slots_and_the_lines_wrap_after_the_fourth() {
        let host = many_functions(6);
        let lines: Vec<u8> = host.funcs.iter().map(|f| f.int_line).collect();
        assert_eq!(lines, [9, 10, 11, 5, 9, 10]);
        let bars: Vec<u32> = host.funcs.iter().map(|f| f.bar0).collect();
        for (index, bar) in bars.iter().enumerate() {
            assert_eq!(*bar as u64, MMIO_BASE + index as u64 * BAR_SIZE);
        }
        let devfns: Vec<u8> = host.funcs.iter().map(|f| f.devfn).collect();
        assert_eq!(devfns, [8, 16, 24, 32, 40, 48]);
    }

    #[test]
    fn the_last_slot_holds_the_thirty_first_function() {
        let mut host = many_functions(MAX_VIRTIO_FUNCTIONS);
        let slot = MAX_VIRTIO_FUNCTIONS as u32;
        enable_memory_space(&mut host, slot);
        assert_eq!(
            cfg_read32(&mut host, slot, CFG_VENDOR_ID as u32) & 0xffff,
            VENDOR_VIRTIO as u32
        );
        let bar = bar_base(&mut host, slot);
        assert_eq!(bar, MMIO_BASE + (slot as u64 - 1) * BAR_SIZE);
        assert!(host.mmio_read(bar + 0x12, 2).is_some());
        assert_eq!(
            cfg_read32(&mut host, slot + 1, CFG_VENDOR_ID as u32),
            NO_DEVICE_DWORD
        );
    }

    #[test]
    fn a_shared_line_stays_asserted_until_every_function_on_it_is_quiet() {
        let mut host = many_functions(5);
        assert_eq!(host.funcs[0].int_line, host.funcs[4].int_line);
        let line = host.funcs[0].int_line;
        host.funcs[0].transport.isr = 1;
        host.funcs[4].transport.isr = 1;
        host.funcs[0].transport.isr = 0;
        assert!(
            host.line_asserted(line),
            "the fifth function still asserts it"
        );
        host.funcs[4].transport.isr = 0;
        assert!(!host.line_asserted(line));
    }

    #[test]
    fn intx_disable_releases_the_line_even_with_isr_bits_set() {
        let mut host = host();
        let line = host.funcs[0].int_line;
        host.funcs[0].transport.isr = 1;
        host.funcs[0].command |= COMMAND_INTX_DISABLE;
        assert!(!host.line_asserted(line));
    }

    const IOMMU_DEVICE_TABLE_REGISTER: u64 = 0x00;
    const IOMMU_EVENT_BASE_REGISTER: u64 = 0x10;
    const IOMMU_CONTROL_REGISTER: u64 = 0x18;
    const IOMMU_EVENT_TAIL_REGISTER: u64 = 0x2018;
    const CONTROL_IOMMU_ENABLE: u64 = 1;
    const CONTROL_EVENT_LOG_ENABLE: u64 = 1 << 2;
    const RING_LENGTH_256_ENTRIES: u64 = 8 << 56;
    const EVENT_BYTES: u64 = 16;
    const DEVICE_TABLE_AT: u64 = 0x1_0000;
    const EVENT_LOG_AT: u64 = 0x3_0000;
    const DESCRIPTORS_AT: u64 = 0x5_0000;
    const AVAILABLE_AT: u64 = 0x6_0000;
    const USED_AT: u64 = 0x7_0000;
    const DTE_VALID: u64 = 1;
    const DTE_TRANSLATION_VALID: u64 = 1 << 1;
    const COMMON_DRIVER_FEATURE_SELECT_OFFSET: u64 = 0x08;
    const COMMON_DRIVER_FEATURE_OFFSET: u64 = 0x0c;
    const STATUS_FEATURES_OK: u32 = 8;
    const STATUS_ACKNOWLEDGE_DRIVER: u32 = 3;

    #[test]
    fn with_an_iommu_every_function_offers_access_platform_and_without_none_does() {
        for iommu in [false, true] {
            let mut h = PciHost::new(vec![
                console_transport(),
                block_transport(),
                net_transport(),
            ])
            .with_iommu(iommu);
            for (slot, bar) in [
                (1u32, MMIO_BASE),
                (2, MMIO_BASE + BAR_SIZE),
                (3, MMIO_BASE + 2 * BAR_SIZE),
            ] {
                enable_memory_space(&mut h, slot);
                // Device feature select 1, the word with bits 32..63.
                h.mmio_write(bar, 4, 1, &mut test_mem(&mut [0u8; 16]));
                let features = h.mmio_read(bar + 4, 4).unwrap();
                assert_eq!(features & 0b10 != 0, iommu, "slot {slot}, iommu {iommu}");
                assert_ne!(features & 1, 0, "VERSION_1 either way");
            }
        }
    }

    /// Programs queue `queue` of the function at `bar` the way a driver does and kicks it, with
    /// the rings in RAM at the `*_AT` addresses and one buffer available.
    fn kick_a_queue(h: &mut PciHost, ram: &mut [u8], bar: u64, queue: u64) {
        ram[AVAILABLE_AT as usize + 2] = 1;
        let mut mem = test_mem(ram);
        let mut write = |h: &mut PciHost, offset: u64, size: u8, value: u64| {
            h.mmio_write(bar + offset, size, value as u32, &mut mem);
        };
        // Accept VERSION_1 and, when offered, ACCESS_PLATFORM, so FEATURES_OK sticks.
        write(h, COMMON_DRIVER_FEATURE_SELECT_OFFSET, 4, 1);
        write(h, COMMON_DRIVER_FEATURE_OFFSET, 4, 1);
        write(
            h,
            COMMON_DEVICE_STATUS_OFFSET,
            1,
            (STATUS_ACKNOWLEDGE_DRIVER | STATUS_FEATURES_OK) as u64,
        );
        write(h, COMMON_QUEUE_SELECT_OFFSET, 2, queue);
        for (register, address) in [(0, DESCRIPTORS_AT), (8, AVAILABLE_AT), (16, USED_AT)] {
            write(h, COMMON_QUEUE_DESC_OFFSET + register, 4, address);
            write(h, COMMON_QUEUE_DESC_OFFSET + register + 4, 4, 0);
        }
        write(h, COMMON_QUEUE_ENABLE_OFFSET, 2, 1);
        write(
            h,
            COMMON_DEVICE_STATUS_OFFSET,
            1,
            (STATUS_ACKNOWLEDGE_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK) as u64,
        );
        write(
            h,
            NOTIFY_REGION_OFFSET + queue * NOTIFY_OFF_MULTIPLIER as u64,
            2,
            queue,
        );
    }

    #[test]
    fn the_dma_of_every_virtio_function_goes_through_the_iommu_as_its_own_requester() {
        let mut h = full_host().with_iommu(true);
        let mut ram = vec![0u8; 1 << 20];
        let functions = [(CONSOLE_SLOT, 1u64), (BLOCK_SLOT, 0), (NET_SLOT, 1)];
        // A translating entry with no mapping and no read or write permission: DMA is refused.
        for (slot, _) in functions {
            let devfn = (slot << DEVFN_SLOT_SHIFT) as usize;
            let entry = DEVICE_TABLE_AT as usize + devfn * 32;
            ram[entry..entry + 8]
                .copy_from_slice(&(DTE_VALID | DTE_TRANSLATION_VALID).to_le_bytes());
        }
        {
            let mut mem = test_mem(&mut ram);
            for (register, value) in [
                (IOMMU_DEVICE_TABLE_REGISTER, DEVICE_TABLE_AT),
                (
                    IOMMU_EVENT_BASE_REGISTER,
                    EVENT_LOG_AT | RING_LENGTH_256_ENTRIES,
                ),
                (
                    IOMMU_CONTROL_REGISTER,
                    CONTROL_IOMMU_ENABLE | CONTROL_EVENT_LOG_ENABLE,
                ),
            ] {
                assert!(h.iommu_mmio_write(iommu::MMIO_BASE + register, 8, value, &mut mem));
            }
        }
        for (slot, queue) in functions {
            enable_memory_space(&mut h, slot);
            let bar = bar_base(&mut h, slot);
            kick_a_queue(&mut h, &mut ram, bar, queue);
        }
        let tail = h
            .iommu_mmio_read(iommu::MMIO_BASE + IOMMU_EVENT_TAIL_REGISTER, 8)
            .unwrap();
        let requesters: Vec<u16> = (0..tail / EVENT_BYTES)
            .map(|index| {
                let at = (EVENT_LOG_AT + index * EVENT_BYTES) as usize;
                u16::from_le_bytes([ram[at], ram[at + 1]])
            })
            .collect();
        for (slot, _) in functions {
            let devfn = (slot << DEVFN_SLOT_SHIFT) as u16;
            assert!(
                requesters.contains(&devfn),
                "no fault from {devfn:#x}: {requesters:x?}"
            );
        }
        assert!(
            requesters.iter().all(|r| [8, 16, 24].contains(r)),
            "{requesters:x?}"
        );
    }
}
