//! A fake driver: it programs the registers, device table, command buffer and page tables in a
//! plain RAM vector exactly as the Linux driver does, then checks what the IOMMU does with DMA,
//! commands and interrupts.
use super::*;
use crate::dev::guestmem::GuestMem;

const RAM_BYTES: usize = 8 << 20;
const DEVICE_TABLE_AT: u64 = 0x1_0000;
const COMMAND_BUFFER_AT: u64 = 0x2_0000;
const EVENT_LOG_AT: u64 = 0x3_0000;
const INTERRUPT_TABLE_AT: u64 = 0x5_0000;
const COMPLETION_STORE_AT: u64 = 0x6_0000;
/// Page tables are allocated from here; data pages live above `DATA_AT`.
const TABLES_AT: u64 = 0x10_0000;
const DATA_AT: u64 = 0x40_0000;
/// 256 entries: the smallest length the spec allows, 2^8.
const RING_LENGTH_256: u64 = 8;
const RING_ENTRIES: u64 = 256;
const COMMAND_BYTES: u64 = 16;

/// 00:03.0, where the first virtio function sits.
const DEVICE: u16 = 0x18;
const DOMAIN: u16 = 7;

const CONTROL_ALL: u64 = CONTROL_IOMMU_ENABLE
    | CONTROL_EVENT_LOG_ENABLE
    | CONTROL_EVENT_INTERRUPT_ENABLE
    | CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE
    | CONTROL_COMMAND_BUFFER_ENABLE;

const EVENT_ILLEGAL_DEVICE_TABLE_ENTRY: u32 = 1;
const EVENT_IO_PAGE_FAULT: u32 = 2;
const EVENT_DEVICE_TABLE_HW_ERROR: u32 = 3;
const EVENT_PAGE_TABLE_HW_ERROR: u32 = 4;
const EVENT_ILLEGAL_COMMAND: u32 = 5;
const MSI_ADDRESS: u32 = 0xfee0_1000;
const MSI_DATA_VALUE: u32 = 0x41;

const DTE_V: u64 = 1;
const DTE_TV: u64 = 1 << 1;
const PTE_PRESENT: u64 = 1;
const PTE_IR: u64 = 1 << 61;
const PTE_IW: u64 = 1 << 62;
const ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;

fn pte(next_level: u64, address: u64, read: bool, write: bool) -> u64 {
    PTE_PRESENT
        | next_level << 9
        | address
        | if read { PTE_IR } else { 0 }
        | if write { PTE_IW } else { 0 }
}

fn dte_q0(mode: u64, root: u64, read: bool, write: bool) -> u64 {
    DTE_V
        | DTE_TV
        | mode << 9
        | root
        | if read { PTE_IR } else { 0 }
        | if write { PTE_IW } else { 0 }
}

struct Event {
    requester: u16,
    domain: u16,
    code: u32,
    flags: u16,
    address: u64,
}

struct Fixture {
    ram: Vec<u8>,
    iommu: Iommu,
    next_table: u64,
    command_tail: u64,
}

impl Fixture {
    fn new() -> Self {
        Fixture {
            ram: vec![0; RAM_BYTES],
            iommu: Iommu::new(),
            next_table: TABLES_AT,
            command_tail: 0,
        }
    }

    /// Programs the rings and the device table and turns everything on, with the MSI set up.
    fn enabled() -> Self {
        let mut f = Fixture::new();
        f.iommu.config_write(MSI_ADDRESS_LOW, 4, MSI_ADDRESS);
        f.iommu.config_write(MSI_DATA, 2, MSI_DATA_VALUE);
        f.iommu
            .config_write(MSI_CONTROL, 2, MSI_CONTROL_ENABLE as u32);
        f.mmio_write(REG_DEVICE_TABLE, 8, DEVICE_TABLE_AT);
        f.mmio_write(
            REG_COMMAND_BASE,
            8,
            COMMAND_BUFFER_AT | RING_LENGTH_256 << RING_LENGTH_SHIFT,
        );
        f.mmio_write(
            REG_EVENT_BASE,
            8,
            EVENT_LOG_AT | RING_LENGTH_256 << RING_LENGTH_SHIFT,
        );
        f.mmio_write(REG_CONTROL, 8, CONTROL_ALL);
        f
    }

    fn mmio_write(&mut self, offset: u64, size: u8, value: u64) {
        let mut mem = GuestMem::new(&mut self.ram, &[]);
        self.iommu.mmio_write(offset, size, value, &mut mem);
    }

    fn read(&self, offset: u64) -> u64 {
        self.iommu.mmio_read(offset, 8)
    }

    fn write64(&mut self, address: u64, value: u64) {
        let at = address as usize;
        self.ram[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn read64(&self, address: u64) -> u64 {
        let at = address as usize;
        u64::from_le_bytes(self.ram[at..at + 8].try_into().unwrap())
    }

    fn alloc_page(&mut self) -> u64 {
        let page = self.next_table;
        self.next_table += 0x1000;
        page
    }

    /// Writes a device table entry and, with the command buffer running, invalidates it, as a
    /// driver must: the IOMMU caches entries.
    fn set_dte(&mut self, device: u16, quads: [u64; 4]) {
        self.write_dte_without_invalidating(device, quads);
        if self.iommu.control() & CONTROL_COMMAND_BUFFER_ENABLE != 0 {
            self.invalidate_device(device);
        }
    }

    fn write_dte_without_invalidating(&mut self, device: u16, quads: [u64; 4]) {
        for (i, quad) in quads.iter().enumerate() {
            self.write64(DEVICE_TABLE_AT + device as u64 * 32 + 8 * i as u64, *quad);
        }
    }

    /// Sets `iova`'s entry at `leaf_level` to `leaf`, creating the tables above it.
    fn map(&mut self, root: u64, levels: u64, iova: u64, leaf_level: u64, leaf: u64) {
        let mut table = root;
        for level in (leaf_level + 1..=levels).rev() {
            let slot = table + ((iova >> (12 + 9 * (level - 1))) & 0x1ff) * 8;
            let existing = self.read64(slot);
            table = if existing & PTE_PRESENT == 0 {
                let next = self.alloc_page();
                self.write64(slot, pte(level - 1, next, true, true));
                next
            } else {
                existing & ADDRESS_MASK
            };
        }
        self.write64(
            table + ((iova >> (12 + 9 * (leaf_level - 1))) & 0x1ff) * 8,
            leaf,
        );
    }

    fn map_4k(&mut self, root: u64, levels: u64, iova: u64, phys: u64, write: bool) {
        self.map(root, levels, iova, 1, pte(0, phys, true, write));
    }

    /// A device with a 3-level table and its root; returns the root.
    fn device_with_tables(&mut self, device: u16, levels: u64) -> u64 {
        let root = self.alloc_page();
        self.set_dte(
            device,
            [dte_q0(levels, root, true, true), DOMAIN as u64, 0, 0],
        );
        root
    }

    fn dma_read(&mut self, device: u16, iova: u64, len: usize) -> Option<Vec<u8>> {
        let mut mem = GuestMem::new(&mut self.ram, &[]);
        let data = mem.with_dma(&mut self.iommu, device, |mem| {
            let mut data = vec![0; len];
            mem.read(iova, &mut data).then_some(data)
        });
        self.iommu.service(&mut mem);
        data
    }

    fn dma_write(&mut self, device: u16, iova: u64, data: &[u8]) -> bool {
        let mut mem = GuestMem::new(&mut self.ram, &[]);
        let ok = mem.with_dma(&mut self.iommu, device, |mem| mem.write(iova, data));
        self.iommu.service(&mut mem);
        ok
    }

    fn queue_command(&mut self, low: u64, high: u64) {
        let at = COMMAND_BUFFER_AT + self.command_tail;
        self.write64(at, low);
        self.write64(at + 8, high);
        self.command_tail = (self.command_tail + COMMAND_BYTES) % (RING_ENTRIES * COMMAND_BYTES);
        let tail = self.command_tail;
        self.mmio_write(REG_COMMAND_TAIL, 8, tail);
    }

    fn invalidate_all(&mut self) {
        self.queue_command(8 << 60, 0);
    }

    fn invalidate_domain(&mut self, domain: u16) {
        self.queue_command(3 << 60 | (domain as u64) << 32, 0x7fff_ffff_ffff_f000 | 1);
    }

    fn invalidate_device(&mut self, device: u16) {
        self.queue_command(2 << 60 | device as u64, 0);
    }

    /// Reads and consumes the events between the log's head and tail.
    fn events(&mut self) -> Vec<Event> {
        let head = self.read(REG_EVENT_HEAD);
        let tail = self.read(REG_EVENT_TAIL);
        let mut events = Vec::new();
        let mut at = head;
        while at != tail {
            let base = EVENT_LOG_AT + at;
            let low = self.read64(base);
            let high = self.read64(base + 8);
            let dword1 = (low >> 32) as u32;
            events.push(Event {
                requester: low as u16,
                domain: dword1 as u16,
                code: dword1 >> 28,
                flags: (dword1 >> 16) as u16 & 0xfff,
                address: high,
            });
            at = (at + 16) % (RING_ENTRIES * 16);
        }
        self.mmio_write(REG_EVENT_HEAD, 8, tail);
        events
    }
}

fn expect_one(events: Vec<Event>) -> Event {
    assert_eq!(events.len(), 1, "exactly one event");
    events.into_iter().next().unwrap()
}

#[test]
fn config_space_identifies_an_amd_iommu_with_its_capabilities() {
    let iommu = Iommu::new();
    assert_eq!(iommu.config_read(0, 2), 0x1022);
    assert_eq!(iommu.config_read(2, 2), 0x1419);
    assert_eq!(iommu.config_read(0x0b, 1), 0x08, "class: system peripheral");
    assert_eq!(iommu.config_read(0x0a, 1), 0x06, "subclass: IOMMU");
    assert_eq!(
        iommu.config_read(CFG_STATUS, 2) & 0x10,
        0x10,
        "capability list"
    );
    assert_eq!(iommu.config_read(CFG_CAPABILITIES_POINTER, 1), 0x40);
    let header = iommu.config_read(CAP_HEADER, 4);
    assert_eq!(header & 0xff, 0x0f, "IOMMU capability id");
    assert_eq!(
        (header >> 8) & 0xff,
        MSI_AT as u32,
        "next is the MSI capability"
    );
    assert_eq!((header >> 16) & 7, 3, "CapType: IOMMU");
    assert_ne!(header & CAP_EFR_SUPPORTED, 0);
    assert_eq!(iommu.config_read(MSI_AT, 1), 5, "MSI capability id");
    assert_eq!(iommu.config_read(MSI_AT + 1, 1), 0, "end of the list");
    assert_ne!(iommu.config_read(MSI_CONTROL, 2) & 0x80, 0, "64-bit MSI");
}

#[test]
fn register_window_starts_enabled_at_the_documented_address() {
    let iommu = Iommu::new();
    assert_eq!(iommu.mmio_base(), Some(MMIO_BASE));
    assert_eq!(iommu.mmio_offset(MMIO_BASE + 0x2020), Some(0x2020));
    assert_eq!(iommu.mmio_offset(MMIO_BASE + MMIO_WINDOW_BYTES), None);
    assert_eq!(iommu.mmio_offset(MMIO_BASE - 1), None);
}

#[test]
fn guest_can_move_the_register_window_or_switch_it_off() {
    let mut iommu = Iommu::new();
    iommu.config_write(CAP_BASE_LOW, 4, 0xfed4_0001);
    assert_eq!(iommu.mmio_base(), Some(0xfed4_0000));
    iommu.config_write(CAP_BASE_HIGH, 4, 1);
    assert_eq!(iommu.mmio_base(), Some(0x1_fed4_0000));
    iommu.config_write(CAP_BASE_LOW, 4, 0xfed4_0000);
    assert_eq!(iommu.mmio_base(), None);
}

#[test]
fn identification_and_capability_headers_are_read_only() {
    let mut iommu = Iommu::new();
    iommu.config_write(0, 4, 0xffff_ffff);
    iommu.config_write(CAP_HEADER, 4, 0);
    iommu.config_write(CAP_MISC, 4, 0);
    assert_eq!(iommu.config_read(0, 4), 0x1419_1022);
    assert_eq!(iommu.config_read(CAP_HEADER, 1), 0x0f);
    assert_ne!(iommu.config_read(CAP_MISC, 4), 0);
}

#[test]
fn msi_message_is_delivered_only_when_the_capability_is_enabled() {
    let mut iommu = Iommu::new();
    iommu.raise_msi();
    assert_eq!(
        iommu.take_msi(),
        None,
        "capability off: the interrupt is lost"
    );
    iommu.config_write(MSI_ADDRESS_LOW, 4, 0xfee0_2003);
    iommu.config_write(MSI_ADDRESS_HIGH, 4, 0);
    iommu.config_write(MSI_DATA, 2, 0x4042);
    iommu.config_write(MSI_CONTROL, 2, 1);
    assert_eq!(iommu.take_msi(), None, "nothing pending");
    iommu.raise_msi();
    assert_eq!(
        iommu.take_msi(),
        Some(MsiMessage {
            address: 0xfee0_2000,
            data: 0x4042
        }),
        "address bits 1:0 are reserved"
    );
    assert_eq!(iommu.take_msi(), None, "delivered once");
}

const CONTROL_XT_INTERRUPTS: u64 = CONTROL_XT_ENABLE | CONTROL_INTCAPXT_ENABLE;

/// What `iommu_init_intcapxt` writes: DM, destination in two pieces, vector.
fn intcapxt_register(destination: u32, vector: u8, logical: bool) -> u64 {
    (logical as u64) << 2
        | ((destination as u64) & 0xff_ffff) << 8
        | (vector as u64) << 32
        | ((destination >> 24) as u64) << 56
}

#[test]
fn x2apic_support_is_advertised_and_xten_is_a_control_bit() {
    let mut f = Fixture::new();
    assert_ne!(f.read(REG_EXTENDED_FEATURE) & EFR_X2APIC, 0);
    f.mmio_write(REG_CONTROL, 8, CONTROL_XT_INTERRUPTS);
    assert_eq!(f.read(REG_CONTROL), CONTROL_XT_INTERRUPTS);
}

#[test]
fn the_xt_interrupt_registers_store_what_software_writes() {
    let mut f = Fixture::new();
    for register in [
        REG_XT_GENERAL_INTERRUPT,
        REG_XT_PPR_INTERRUPT,
        REG_XT_GA_LOG_INTERRUPT,
    ] {
        f.mmio_write(register, 8, 0x1234_5678_9abc_def0);
        assert_eq!(f.read(register), 0x1234_5678_9abc_def0, "{register:#x}");
    }
}

#[test]
fn with_intcapxt_the_event_interrupt_goes_straight_to_the_apic_with_a_32_bit_destination() {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL | CONTROL_XT_INTERRUPTS);
    f.mmio_write(
        REG_XT_GENERAL_INTERRUPT,
        8,
        intcapxt_register(0x12ab_cdef, 0x71, false),
    );
    f.iommu.raise_msi();
    let Some(IommuInterrupt::Direct(interrupt)) = f.iommu.take_interrupt() else {
        panic!("expected a direct interrupt");
    };
    assert_eq!(interrupt.vector, 0x71);
    assert_eq!(interrupt.destination, 0x12ab_cdef);
    assert!(!interrupt.logical);
    assert_eq!(interrupt.delivery_mode, crate::dev::apic::DELIVERY_FIXED);
    assert!(f.iommu.take_interrupt().is_none(), "once");
}

#[test]
fn without_intcapxt_the_event_interrupt_is_the_msi_capabilitys_message() {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL | CONTROL_XT_ENABLE);
    f.iommu.raise_msi();
    let Some(IommuInterrupt::Message(message)) = f.iommu.take_interrupt() else {
        panic!("expected an MSI");
    };
    assert_eq!(message.data, MSI_DATA_VALUE);
}

#[test]
fn with_xten_the_irte_destination_gains_the_top_byte_from_the_high_qword() {
    let mut f = interrupt_fixture(9, CONTROL_GUEST_APIC_ENABLE | CONTROL_XT_ENABLE);
    let low: u64 = 1 | 0x00ab_cdef << 8;
    let high: u64 = 0x77 | 0x12 << 16;
    let at = INTERRUPT_TABLE_AT + 16 * 3;
    f.write64(at, low);
    f.write64(at + 8, high);
    let remapped = remap(&mut f, msi(DEVICE, 3, 0)).unwrap();
    assert_eq!(remapped.vector, 0x77);
    assert_eq!(remapped.destination, 0x12ab_cdef);
}

#[test]
fn without_xten_the_high_qwords_destination_bits_are_ignored() {
    let mut f = interrupt_fixture(9, CONTROL_GUEST_APIC_ENABLE);
    let at = INTERRUPT_TABLE_AT;
    f.write64(at, 1 | 0x34 << 8);
    f.write64(at + 8, 0x77 | 0x12 << 16);
    assert_eq!(remap(&mut f, msi(DEVICE, 0, 0)).unwrap().destination, 0x34);
}

#[test]
fn extended_feature_register_advertises_what_is_implemented() {
    let f = Fixture::new();
    assert_eq!(f.read(REG_EXTENDED_FEATURE), EXTENDED_FEATURES);
    assert_ne!(EXTENDED_FEATURES & EFR_INVALIDATE_ALL, 0);
    assert_ne!(EXTENDED_FEATURES & EFR_GUEST_APIC, 0);
}

#[test]
fn base_registers_keep_only_their_implemented_bits() {
    let mut f = Fixture::new();
    f.mmio_write(REG_DEVICE_TABLE, 8, u64::MAX);
    assert_eq!(f.read(REG_DEVICE_TABLE), DEVICE_TABLE_WRITABLE);
    f.mmio_write(REG_COMMAND_BASE, 8, u64::MAX);
    assert_eq!(f.read(REG_COMMAND_BASE), RING_BASE_WRITABLE);
    f.mmio_write(REG_CONTROL, 8, u64::MAX);
    assert_eq!(f.read(REG_CONTROL), CONTROL_WRITABLE);
}

#[test]
fn registers_can_be_accessed_as_dwords() {
    let mut f = Fixture::new();
    f.mmio_write(REG_DEVICE_TABLE, 4, 0x0004_0003);
    f.mmio_write(REG_DEVICE_TABLE + 4, 4, 0x1);
    assert_eq!(f.read(REG_DEVICE_TABLE), 0x1_0004_0003);
    assert_eq!(f.iommu.mmio_read(REG_DEVICE_TABLE + 4, 4), 1);
    assert_eq!(f.iommu.mmio_read(REG_DEVICE_TABLE, 2), 0x0003);
}

#[test]
fn unmodeled_registers_read_zero_and_ignore_writes() {
    let mut f = Fixture::new();
    f.mmio_write(0x0038, 8, 0x1234_5000);
    f.mmio_write(0x2030, 8, 0x10);
    assert_eq!(f.read(0x0038), 0);
    assert_eq!(f.read(0x2030), 0);
    assert_eq!(f.read(0x3ff8), 0);
}

#[test]
fn control_enables_drive_the_run_bits_of_the_status_register() {
    let mut f = Fixture::new();
    assert_eq!(f.read(REG_STATUS), 0);
    f.mmio_write(
        REG_CONTROL,
        8,
        CONTROL_EVENT_LOG_ENABLE | CONTROL_COMMAND_BUFFER_ENABLE,
    );
    assert_eq!(
        f.read(REG_STATUS),
        STATUS_EVENT_LOG_RUNNING | STATUS_COMMAND_BUFFER_RUNNING
    );
    f.mmio_write(REG_CONTROL, 8, 0);
    assert_eq!(f.read(REG_STATUS), 0);
}

#[test]
fn status_interrupt_bits_clear_by_writing_one_and_run_bits_do_not() {
    let mut f = Fixture::enabled();
    f.iommu.status_set(
        STATUS_EVENT_OVERFLOW | STATUS_EVENT_INTERRUPT | STATUS_COMPLETION_WAIT_INTERRUPT,
    );
    f.mmio_write(REG_STATUS, 4, STATUS_EVENT_INTERRUPT);
    assert_eq!(
        f.read(REG_STATUS),
        STATUS_EVENT_OVERFLOW
            | STATUS_COMPLETION_WAIT_INTERRUPT
            | STATUS_EVENT_LOG_RUNNING
            | STATUS_COMMAND_BUFFER_RUNNING
    );
    f.mmio_write(REG_STATUS, 8, u64::MAX);
    assert_eq!(
        f.read(REG_STATUS),
        STATUS_EVENT_LOG_RUNNING | STATUS_COMMAND_BUFFER_RUNNING
    );
}

#[test]
fn writing_a_ring_base_resets_its_pointers() {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_EVENT_TAIL, 8, 0x40);
    f.mmio_write(REG_COMMAND_HEAD, 8, 0x20);
    f.mmio_write(
        REG_EVENT_BASE,
        8,
        EVENT_LOG_AT | RING_LENGTH_256 << RING_LENGTH_SHIFT,
    );
    f.mmio_write(
        REG_COMMAND_BASE,
        8,
        COMMAND_BUFFER_AT | RING_LENGTH_256 << RING_LENGTH_SHIFT,
    );
    assert_eq!(f.read(REG_EVENT_TAIL), 0);
    assert_eq!(f.read(REG_COMMAND_HEAD), 0);
}

#[test]
fn pointers_keep_bits_18_to_4() {
    let mut f = Fixture::new();
    f.mmio_write(REG_EVENT_HEAD, 8, 0xffff_ffff);
    assert_eq!(f.read(REG_EVENT_HEAD), 0x7_fff0);
}

#[test]
fn exclusion_limit_reads_back_with_its_low_bits_set() {
    let mut f = Fixture::new();
    f.mmio_write(REG_EXCLUSION_LIMIT, 8, 0x20_0000);
    assert_eq!(f.read(REG_EXCLUSION_LIMIT), 0x20_0fff);
}

#[test]
fn dma_is_untranslated_while_the_iommu_is_off() {
    let mut f = Fixture::new();
    f.ram[0x1234] = 0x5a;
    assert_eq!(f.dma_read(DEVICE, 0x1234, 1), Some(vec![0x5a]));
    assert!(f.events().is_empty());
}

#[test]
fn dma_is_untranslated_when_enabled_without_a_device_table() {
    let mut f = Fixture::new();
    f.mmio_write(REG_CONTROL, 8, CONTROL_IOMMU_ENABLE);
    f.ram[0x1234] = 0x5a;
    assert_eq!(f.dma_read(DEVICE, 0x1234, 1), Some(vec![0x5a]));
}

#[test]
fn four_kilobyte_mapping_translates_reads_and_writes() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x7000, DATA_AT + 0x3000, true);
    f.ram[(DATA_AT + 0x3010) as usize] = 0xab;
    assert_eq!(f.dma_read(DEVICE, 0x7010, 1), Some(vec![0xab]));
    assert!(f.dma_write(DEVICE, 0x7020, &[1, 2, 3]));
    assert_eq!(&f.ram[(DATA_AT + 0x3020) as usize..][..3], &[1, 2, 3]);
    assert_eq!(f.ram[0x7020], 0, "the device address itself is untouched");
}

#[test]
fn access_spanning_two_pages_follows_each_mapping() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 4);
    f.map_4k(root, 4, 0x10000, DATA_AT + 0x8000, true);
    f.map_4k(root, 4, 0x11000, DATA_AT + 0x2000, true);
    let pattern: Vec<u8> = (0..16).collect();
    assert!(f.dma_write(DEVICE, 0x10ff8, &pattern));
    assert_eq!(&f.ram[(DATA_AT + 0x8ff8) as usize..][..8], &pattern[..8]);
    assert_eq!(&f.ram[(DATA_AT + 0x2000) as usize..][..8], &pattern[8..]);
    assert_eq!(f.dma_read(DEVICE, 0x10ff8, 16), Some(pattern));
}

#[test]
fn every_paging_depth_from_one_to_six_levels_works() {
    for levels in 1..=6u64 {
        let mut f = Fixture::enabled();
        let root = f.device_with_tables(DEVICE, levels);
        // The highest address the depth reaches (capped at what a test can build).
        let top_bits = (12 + 9 * levels).min(48);
        let iova = (1u64 << top_bits) - 0x1000;
        f.map_4k(root, levels, iova, DATA_AT, true);
        f.ram[DATA_AT as usize + 5] = levels as u8;
        assert_eq!(
            f.dma_read(DEVICE, iova + 5, 1),
            Some(vec![levels as u8]),
            "{levels} levels"
        );
    }
}

#[test]
fn address_beyond_the_depth_of_the_table_faults() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 2);
    f.map_4k(root, 2, 0x1000, DATA_AT, true);
    assert!(
        f.dma_read(DEVICE, 1 << 30, 1).is_none(),
        "2 levels cover 1 GiB"
    );
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_IO_PAGE_FAULT);
    assert_eq!(event.address, 1 << 30);
}

#[test]
fn two_megabyte_and_one_gigabyte_leaves_cover_their_whole_range() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map(root, 3, 0x20_0000, 2, pte(0, DATA_AT, true, true));
    f.map(root, 3, 0x4000_0000, 3, pte(0, 0, true, true));
    f.ram[(DATA_AT + 0x1f_f000) as usize] = 0x22;
    assert_eq!(f.dma_read(DEVICE, 0x3f_f000, 1), Some(vec![0x22]));
    f.ram[0x23_4567] = 0x33;
    assert_eq!(
        f.dma_read(DEVICE, 0x4000_0000 + 0x23_4567, 1),
        Some(vec![0x33])
    );
}

#[test]
fn leaf_with_next_level_seven_encodes_its_size_in_the_address() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    // As Linux does, the entry is repeated in every slot the page covers.
    // 8 KiB: bit 12 clear, so the lowest clear bit is 12 and the size is 2^13.
    for slot in 0..2 {
        f.map(
            root,
            3,
            0x10_0000 + slot * 0x1000,
            1,
            pte(7, DATA_AT, true, true),
        );
    }
    f.ram[(DATA_AT + 0x1fff) as usize] = 0x81;
    assert_eq!(f.dma_read(DEVICE, 0x10_1fff, 1), Some(vec![0x81]));
    // 64 KiB: address bits 14:12 set, bit 15 clear.
    let big = DATA_AT + 0x10_0000;
    for slot in 0..16 {
        f.map(
            root,
            3,
            0x20_0000 + slot * 0x1000,
            1,
            pte(7, big | 0x7000, true, true),
        );
    }
    f.ram[(big + 0xffff) as usize] = 0x64;
    assert_eq!(f.dma_read(DEVICE, 0x20_ffff, 1), Some(vec![0x64]));
    f.ram[(big + 0x8000) as usize] = 0x65;
    assert_eq!(f.dma_read(DEVICE, 0x20_8000, 1), Some(vec![0x65]));
}

#[test]
fn a_table_may_skip_levels() {
    let mut f = Fixture::enabled();
    let root = f.alloc_page();
    let low_table = f.alloc_page();
    f.set_dte(DEVICE, [dte_q0(4, root, true, true), DOMAIN as u64, 0, 0]);
    // Index 0 of the level-4 table points straight at a level-1 table.
    f.write64(root, pte(1, low_table, true, true));
    f.write64(low_table + 8, pte(0, DATA_AT, true, true));
    f.ram[DATA_AT as usize + 7] = 0x99;
    assert_eq!(f.dma_read(DEVICE, 0x1007, 1), Some(vec![0x99]));
}

#[test]
fn a_next_level_that_does_not_descend_is_a_fault() {
    let mut f = Fixture::enabled();
    let root = f.alloc_page();
    f.set_dte(DEVICE, [dte_q0(2, root, true, true), DOMAIN as u64, 0, 0]);
    f.write64(root, pte(2, root, true, true));
    assert!(f.dma_read(DEVICE, 0x1000, 1).is_none());
    assert_eq!(expect_one(f.events()).code, EVENT_IO_PAGE_FAULT);
}

#[test]
fn not_present_page_logs_an_io_page_fault_with_the_device_domain_and_address() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    assert!(f.dma_read(DEVICE, 0xdead_b000, 4).is_none());
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_IO_PAGE_FAULT);
    assert_eq!(event.requester, DEVICE);
    assert_eq!(event.domain, DOMAIN);
    assert_eq!(event.address, 0xdead_b000);
    assert_eq!(event.flags & events::FLAG_PRESENT, 0);
    assert_eq!(event.flags & events::FLAG_WRITE, 0);
}

#[test]
fn write_to_a_read_only_mapping_faults_as_a_present_write() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x5000, DATA_AT, false);
    assert!(f.dma_read(DEVICE, 0x5000, 1).is_some());
    assert!(!f.dma_write(DEVICE, 0x5000, &[1]));
    assert_eq!(f.ram[DATA_AT as usize], 0);
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_IO_PAGE_FAULT);
    assert_eq!(event.flags, events::FLAG_PRESENT | events::FLAG_WRITE);
}

#[test]
fn permissions_are_the_intersection_of_the_dte_and_the_entries() {
    let mut f = Fixture::enabled();
    let root = f.alloc_page();
    f.set_dte(DEVICE, [dte_q0(3, root, true, false), DOMAIN as u64, 0, 0]);
    f.map_4k(root, 3, 0x5000, DATA_AT, true);
    assert!(f.dma_read(DEVICE, 0x5000, 1).is_some());
    assert!(!f.dma_write(DEVICE, 0x5000, &[1]), "the DTE forbids writes");
}

#[test]
fn faulting_dma_does_not_touch_ram_or_panic_on_bad_addresses() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    assert!(!f.dma_write(DEVICE, u64::MAX - 2, &[1, 2, 3, 4]));
    assert!(f.dma_read(DEVICE, u64::MAX, 8).is_none());
}

#[test]
fn device_without_a_valid_entry_is_not_translated() {
    let mut f = Fixture::enabled();
    f.ram[0x3000] = 0x11;
    assert_eq!(f.dma_read(DEVICE, 0x3000, 1), Some(vec![0x11]), "V=0");
    f.set_dte(DEVICE, [DTE_V | 3 << 9, 0, 0, 0]);
    assert_eq!(f.dma_read(DEVICE, 0x3000, 1), Some(vec![0x11]), "TV=0");
    assert!(f.events().is_empty());
}

#[test]
fn mode_zero_passes_dma_through_subject_to_the_dte_permissions() {
    let mut f = Fixture::enabled();
    f.ram[0x3000] = 0x11;
    f.set_dte(DEVICE, [dte_q0(0, 0, true, false), DOMAIN as u64, 0, 0]);
    assert_eq!(f.dma_read(DEVICE, 0x3000, 1), Some(vec![0x11]));
    assert!(!f.dma_write(DEVICE, 0x3000, &[1]));
    assert_eq!(
        expect_one(f.events()).flags & events::FLAG_PRESENT,
        events::FLAG_PRESENT
    );
    // The way Linux blocks a device: valid, translation valid, mode 0, no permissions.
    f.set_dte(DEVICE, [dte_q0(0, 0, false, false), DOMAIN as u64, 0, 0]);
    assert!(f.dma_read(DEVICE, 0x3000, 1).is_none());
}

#[test]
fn reserved_dte_bits_log_an_illegal_device_table_entry() {
    let mut f = Fixture::enabled();
    f.set_dte(DEVICE, [dte_q0(3, 0x1000, true, true) | 1 << 3, 0, 0, 0]);
    assert!(f.dma_read(DEVICE, 0x1000, 1).is_none());
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_ILLEGAL_DEVICE_TABLE_ENTRY);
    assert_eq!(event.requester, DEVICE);
    f.set_dte(DEVICE, [dte_q0(7, 0x1000, true, true), 0, 0, 0]);
    assert!(
        f.dma_read(DEVICE, 0x1000, 1).is_none(),
        "mode 7 is reserved"
    );
}

#[test]
fn device_beyond_the_table_has_no_entry() {
    let mut f = Fixture::enabled();
    assert!(
        f.dma_read(200, 0x1000, 1).is_none(),
        "the table holds 128 entries"
    );
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_IO_PAGE_FAULT);
    assert_eq!(event.requester, 200);
}

#[test]
fn unreadable_device_table_is_a_hardware_error() {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_DEVICE_TABLE, 8, 0x7fff_0000_0000);
    assert!(f.dma_read(DEVICE, 0x1000, 1).is_none());
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_DEVICE_TABLE_HW_ERROR);
    assert_eq!(event.address, 0x7fff_0000_0000 + DEVICE as u64 * 32);
}

#[test]
fn unreadable_page_table_is_a_hardware_error_with_the_entry_address() {
    let mut f = Fixture::enabled();
    f.set_dte(
        DEVICE,
        [
            dte_q0(3, 0x7fff_0000_0000 & ADDRESS_MASK, true, true),
            DOMAIN as u64,
            0,
            0,
        ],
    );
    assert!(f.dma_read(DEVICE, 0x1000, 1).is_none());
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_PAGE_TABLE_HW_ERROR);
    assert_eq!(event.domain, DOMAIN);
    assert_eq!(event.address, 0x7fff_0000_0000 & ADDRESS_MASK);
}

#[test]
fn dma_into_the_interrupt_window_is_never_translated_and_logs_nothing() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    assert!(!f.dma_write(DEVICE, 0xfee0_0000, &[1, 2, 3, 4]));
    assert!(f.events().is_empty());
}

#[test]
fn exclusion_range_lets_marked_devices_through_untranslated() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    f.ram[0x30_0000] = 0x77;
    f.mmio_write(REG_EXCLUSION_BASE, 8, 0x30_0000 | EXCLUSION_ENABLE);
    f.mmio_write(REG_EXCLUSION_LIMIT, 8, 0x30_0000);
    assert!(
        f.dma_read(DEVICE, 0x30_0000, 1).is_none(),
        "DTE.EX is clear"
    );
    f.events();
    let root = f.read64(DEVICE_TABLE_AT + DEVICE as u64 * 32) & ADDRESS_MASK;
    f.set_dte(
        DEVICE,
        [dte_q0(3, root, true, true), DOMAIN as u64 | 1 << 39, 0, 0],
    );
    assert_eq!(f.dma_read(DEVICE, 0x30_0000, 1), Some(vec![0x77]));
    assert!(
        f.dma_read(DEVICE, 0x30_1000, 1).is_none(),
        "outside the range"
    );
}

#[test]
fn exclusion_allow_admits_every_device() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    f.ram[0x30_0000] = 0x78;
    f.mmio_write(
        REG_EXCLUSION_BASE,
        8,
        0x30_0000 | EXCLUSION_ENABLE | EXCLUSION_ALLOW_ALL,
    );
    f.mmio_write(REG_EXCLUSION_LIMIT, 8, 0x30_0000);
    assert_eq!(f.dma_read(DEVICE, 0x30_0800, 1).map(|_| ()), Some(()));
    assert_eq!(f.dma_read(DEVICE, 0x30_0000, 1), Some(vec![0x78]));
}

#[test]
fn translations_stay_cached_until_the_driver_invalidates_them() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x5000, DATA_AT, true);
    f.ram[DATA_AT as usize] = 1;
    f.ram[DATA_AT as usize + 0x1000] = 2;
    assert_eq!(f.dma_read(DEVICE, 0x5000, 1), Some(vec![1]));
    f.map_4k(root, 3, 0x5000, DATA_AT + 0x1000, true);
    assert_eq!(
        f.dma_read(DEVICE, 0x5000, 1),
        Some(vec![1]),
        "stale until invalidated"
    );
    f.invalidate_domain(DOMAIN);
    assert_eq!(f.dma_read(DEVICE, 0x5000, 1), Some(vec![2]));
}

#[test]
fn invalidating_a_device_or_everything_drops_its_translations() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x5000, DATA_AT, true);
    f.ram[DATA_AT as usize] = 1;
    f.ram[DATA_AT as usize + 0x1000] = 2;
    f.dma_read(DEVICE, 0x5000, 1);
    f.map_4k(root, 3, 0x5000, DATA_AT + 0x1000, true);
    f.invalidate_device(DEVICE + 8);
    assert_eq!(
        f.dma_read(DEVICE, 0x5000, 1),
        Some(vec![1]),
        "another device's flush"
    );
    f.invalidate_device(DEVICE);
    assert_eq!(f.dma_read(DEVICE, 0x5000, 1), Some(vec![2]));
    f.map_4k(root, 3, 0x5000, DATA_AT, true);
    f.invalidate_all();
    assert_eq!(f.dma_read(DEVICE, 0x5000, 1), Some(vec![1]));
}

/// Hardware caches device table entries too (spec 2.2.2.1); Linux and Xen invalidate one after
/// every change. Passthrough and untranslated devices then need no table read per access.
#[test]
fn device_table_entries_stay_cached_until_the_driver_invalidates_them() {
    let mut f = Fixture::enabled();
    f.ram[0x3000] = 0x11;
    f.set_dte(DEVICE, [dte_q0(0, 0, true, true), DOMAIN as u64, 0, 0]);
    assert_eq!(f.dma_read(DEVICE, 0x3000, 1), Some(vec![0x11]));
    let blocked = [dte_q0(0, 0, false, false), DOMAIN as u64, 0, 0];
    f.write_dte_without_invalidating(DEVICE, blocked);
    assert_eq!(
        f.dma_read(DEVICE, 0x3000, 1),
        Some(vec![0x11]),
        "stale until invalidated"
    );
    f.invalidate_device(DEVICE + 8);
    assert!(
        f.dma_read(DEVICE, 0x3000, 1).is_some(),
        "another device's flush"
    );
    f.invalidate_device(DEVICE);
    assert!(f.dma_read(DEVICE, 0x3000, 1).is_none());
    let passthrough = [dte_q0(0, 0, true, true), DOMAIN as u64, 0, 0];
    f.write_dte_without_invalidating(DEVICE, passthrough);
    f.invalidate_all();
    assert!(f.dma_read(DEVICE, 0x3000, 1).is_some(), "invalidate all");
    f.write_dte_without_invalidating(DEVICE, blocked);
    f.mmio_write(REG_DEVICE_TABLE, 8, DEVICE_TABLE_AT);
    assert!(
        f.dma_read(DEVICE, 0x3000, 1).is_none(),
        "a new device table"
    );
    f.write_dte_without_invalidating(DEVICE, passthrough);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL & !CONTROL_IOMMU_ENABLE);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL);
    assert!(
        f.dma_read(DEVICE, 0x3000, 1).is_some(),
        "the IOMMU switched off and on"
    );
}

/// The IOTLB keeps a working set of thousands of pages (it used to drop everything at 1024, so
/// a device streaming through more buffers than that walked the tables for every page).
#[test]
fn the_iotlb_holds_a_two_thousand_page_working_set() {
    const PAGES: u64 = 2048;
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    for page in 0..PAGES {
        f.map_4k(root, 3, page << 12, DATA_AT + (page % 512) * 0x1000, true);
    }
    for page in 0..PAGES {
        assert!(f.dma_read(DEVICE, page << 12, 1).is_some());
    }
    // Unmapped behind the IOMMU's back: every page must still come from the IOTLB.
    f.write64(root, 0);
    for page in 0..PAGES {
        assert!(f.dma_read(DEVICE, page << 12, 1).is_some(), "page {page}");
    }
}

#[test]
fn invalidating_another_domain_keeps_the_cache() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x5000, DATA_AT, true);
    f.ram[DATA_AT as usize] = 1;
    f.ram[DATA_AT as usize + 0x1000] = 2;
    f.dma_read(DEVICE, 0x5000, 1);
    f.map_4k(root, 3, 0x5000, DATA_AT + 0x1000, true);
    f.invalidate_domain(DOMAIN + 1);
    assert_eq!(f.dma_read(DEVICE, 0x5000, 1), Some(vec![1]));
}

#[test]
fn completion_wait_stores_its_data_and_raises_the_interrupt() {
    let mut f = Fixture::enabled();
    f.queue_command(1 << 60 | 3 | COMPLETION_STORE_AT, 0xfeed_face_cafe_beef);
    assert_eq!(f.read64(COMPLETION_STORE_AT), 0xfeed_face_cafe_beef);
    assert_ne!(f.read(REG_STATUS) & STATUS_COMPLETION_WAIT_INTERRUPT, 0);
    assert_eq!(
        f.iommu.take_msi(),
        Some(MsiMessage {
            address: MSI_ADDRESS as u64,
            data: MSI_DATA_VALUE
        })
    );
    assert_eq!(f.read(REG_COMMAND_HEAD), f.read(REG_COMMAND_TAIL));
}

#[test]
fn completion_wait_without_the_store_or_interrupt_flags_does_neither() {
    let mut f = Fixture::enabled();
    f.queue_command(1 << 60 | COMPLETION_STORE_AT, 0x1234);
    assert_eq!(f.read64(COMPLETION_STORE_AT), 0);
    assert_eq!(f.read(REG_STATUS) & STATUS_COMPLETION_WAIT_INTERRUPT, 0);
    assert_eq!(f.iommu.take_msi(), None);
}

#[test]
fn completion_wait_interrupt_needs_its_control_bit_for_the_message() {
    let mut f = Fixture::enabled();
    f.mmio_write(
        REG_CONTROL,
        8,
        CONTROL_ALL & !CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE,
    );
    f.queue_command(1 << 60 | 2, 0);
    assert_ne!(f.read(REG_STATUS) & STATUS_COMPLETION_WAIT_INTERRUPT, 0);
    assert_eq!(f.iommu.take_msi(), None);
}

#[test]
fn commands_queued_before_the_buffer_is_enabled_run_when_it_is() {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL & !CONTROL_COMMAND_BUFFER_ENABLE);
    f.queue_command(1 << 60 | 1 | COMPLETION_STORE_AT, 42);
    assert_eq!(f.read64(COMPLETION_STORE_AT), 0);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL);
    assert_eq!(f.read64(COMPLETION_STORE_AT), 42);
}

#[test]
fn command_ring_wraps_around() {
    let mut f = Fixture::enabled();
    for round in 1..=RING_ENTRIES + 4 {
        f.queue_command(1 << 60 | 1 | COMPLETION_STORE_AT, round);
        assert_eq!(f.read64(COMPLETION_STORE_AT), round);
    }
    assert_eq!(f.read(REG_COMMAND_HEAD), 4 * COMMAND_BYTES);
}

#[test]
fn illegal_command_logs_an_event_and_stops_until_the_buffer_is_toggled() {
    let mut f = Fixture::enabled();
    f.queue_command(0xf << 60, 0);
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_ILLEGAL_COMMAND);
    assert_eq!(event.address, COMMAND_BUFFER_AT);
    assert_eq!(f.read(REG_STATUS) & STATUS_COMMAND_BUFFER_RUNNING, 0);
    f.queue_command(1 << 60 | 1 | COMPLETION_STORE_AT, 9);
    assert_eq!(
        f.read64(COMPLETION_STORE_AT),
        0,
        "stalled behind the bad command"
    );
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL & !CONTROL_COMMAND_BUFFER_ENABLE);
    f.mmio_write(REG_COMMAND_HEAD, 8, COMMAND_BYTES);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL);
    assert_eq!(f.read64(COMPLETION_STORE_AT), 9);
}

#[test]
fn completion_store_outside_ram_is_a_command_hardware_error() {
    let mut f = Fixture::enabled();
    f.queue_command(1 << 60 | 1 | 0x7fff_0000_0000, 1);
    let event = expect_one(f.events());
    assert_eq!(event.code, 6);
    assert_eq!(f.read(REG_STATUS) & STATUS_COMMAND_BUFFER_RUNNING, 0);
}

#[test]
fn accepted_no_op_commands_do_not_stall_the_buffer() {
    let mut f = Fixture::enabled();
    for opcode in [4u64, 5, 6, 7] {
        f.queue_command(opcode << 60, 0);
    }
    assert_ne!(f.read(REG_STATUS) & STATUS_COMMAND_BUFFER_RUNNING, 0);
    assert!(f.events().is_empty());
}

#[test]
fn events_raise_the_interrupt_status_and_the_msi() {
    let mut f = Fixture::enabled();
    f.dma_read(DEVICE, 0x1000, 1);
    f.device_with_tables(DEVICE, 3);
    f.dma_read(DEVICE, 0x1000, 1);
    assert_ne!(f.read(REG_STATUS) & STATUS_EVENT_INTERRUPT, 0);
    assert!(f.iommu.take_msi().is_some());
    assert_eq!(f.events().len(), 1);
}

#[test]
fn events_are_dropped_while_the_log_is_off() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL & !CONTROL_EVENT_LOG_ENABLE);
    f.dma_read(DEVICE, 0x1000, 1);
    assert_eq!(f.read(REG_EVENT_TAIL), 0);
    assert_eq!(f.read(REG_STATUS) & STATUS_EVENT_INTERRUPT, 0);
}

#[test]
fn full_event_log_overflows_and_restarts_when_software_toggles_it() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    for i in 0..RING_ENTRIES {
        f.dma_read(DEVICE, 0x1000 + i * 0x1000, 1);
    }
    assert_ne!(f.read(REG_STATUS) & STATUS_EVENT_OVERFLOW, 0);
    assert_eq!(f.read(REG_STATUS) & STATUS_EVENT_LOG_RUNNING, 0);
    assert_eq!(
        f.events().len() as u64,
        RING_ENTRIES - 1,
        "one slot stays empty"
    );
    f.mmio_write(REG_STATUS, 8, STATUS_EVENT_OVERFLOW);
    f.dma_read(DEVICE, 0x1000, 1);
    assert!(f.events().is_empty(), "still stopped");
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL & !CONTROL_EVENT_LOG_ENABLE);
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL);
    f.dma_read(DEVICE, 0x1000, 1);
    assert_eq!(f.events().len(), 1);
}

#[test]
fn dma_contains_checks_every_page_of_the_range() {
    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    f.map_4k(root, 3, 0x1000, DATA_AT, false);
    f.map_4k(root, 3, 0x2000, DATA_AT + 0x5000, false);
    f.map_4k(root, 3, 0x4000, 0x7fff_0000_0000 & ADDRESS_MASK, false);
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    mem.with_dma(&mut f.iommu, DEVICE, |mem| {
        assert!(mem.contains(0x1000, 0x2000), "two pages, read-only is fine");
        assert!(!mem.contains(0x1000, 0x3000), "0x3000 is unmapped");
        assert!(!mem.contains(0x4000, 8), "mapped, but not to RAM");
        assert!(!mem.contains(0x1000, u64::MAX));
    });
    assert!(f.events().is_empty(), "probing reports nothing");
}

#[test]
fn guest_mem_without_dma_still_means_physical_addresses() {
    let mut f = Fixture::enabled();
    f.device_with_tables(DEVICE, 3);
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    assert!(mem.write(0x1000, &[5]));
    assert_eq!(mem.read_u16(0x1000), 5);
    assert!(mem.contains(0x1000, 1));
}

#[test]
fn virtio_block_device_dma_goes_through_the_iommu() {
    use crate::dev::vblk::VirtioBlk;
    use crate::dev::virtio::testutil::{post, put_desc, setup_queue, used_idx};
    use crate::dev::virtio::{DESC_F_NEXT, DESC_F_WRITE, VirtioDev};
    use crate::disk::Disk;

    let mut f = Fixture::enabled();
    let root = f.device_with_tables(DEVICE, 3);
    // IOVA page n lives at physical DATA_AT + n * 0x1000 + 0x10_0000, except the data buffer,
    // whose two pages are physically far apart.
    let phys = |iova: u64| DATA_AT + 0x10_0000 + iova;
    for iova in (0x1000..=0x6000).step_by(0x1000) {
        f.map_4k(root, 3, iova, phys(iova), true);
    }
    f.map_4k(root, 3, 0x8000, DATA_AT + 0x30_0000, true);
    f.map_4k(root, 3, 0x9000, DATA_AT + 0x20_0000, true);

    let (desc, avail, used, header, status, data) =
        (0x1000u64, 0x2000, 0x3000, 0x4000, 0x6000, 0x8e00);
    let mut queue = setup_queue(&mut f.ram, desc, avail, used, 8);
    // Descriptors, rings and the request header are what the guest wrote: into physical pages.
    let ram = &mut f.ram;
    let put = |ram: &mut [u8], index: u16, iova: u64, len: u32, flags: u16, next: u16| {
        put_desc(
            &mut ram[phys(desc) as usize..],
            0,
            index,
            iova,
            len,
            flags,
            next,
        );
    };
    put(ram, 0, header, 16, DESC_F_NEXT, 1);
    put(ram, 1, data, 512, DESC_F_WRITE | DESC_F_NEXT, 2);
    put(ram, 2, status, 1, DESC_F_WRITE, 0);
    ram[phys(status) as usize] = 0xff;
    post(&mut ram[phys(avail) as usize..], 0, 8, 0);

    let sector_pattern: Vec<u8> = (0..512usize).map(|i| (i * 7) as u8).collect();
    let mut disk_bytes = vec![0u8; 4096];
    disk_bytes[..512].copy_from_slice(&sector_pattern);
    let mut disk = VirtioBlk::new(Disk::memory(disk_bytes));
    let used_before = used_idx(&f.ram[phys(used) as usize..], 0);
    {
        let mut mem = GuestMem::new(&mut f.ram, &[]);
        let kicked = mem.with_dma(&mut f.iommu, DEVICE, |mem| {
            disk.kick(0, std::slice::from_mut(&mut queue), mem)
        });
        f.iommu.service(&mut mem);
        assert!(kicked);
    }
    assert_eq!(
        used_idx(&f.ram[phys(used) as usize..], 0),
        used_before.wrapping_add(1)
    );
    assert_eq!(f.ram[phys(status) as usize], 0, "VIRTIO_BLK_S_OK");
    // The buffer straddles 0x8e00..0x9000 and 0x9000..: 512 bytes starting 0x200 before the end.
    let first = &f.ram[(DATA_AT + 0x30_0000 + 0xe00) as usize..][..0x200];
    assert_eq!(first, &sector_pattern[..0x200]);
    assert!(f.events().is_empty());
}

/// A block device reading into a buffer at 0xa000 on a chain whose other pages are mapped.
fn block_read_into_0xa000(f: &mut Fixture, data_page: Option<bool>) {
    use crate::dev::vblk::VirtioBlk;
    use crate::dev::virtio::testutil::{post, put_desc, setup_queue};
    use crate::dev::virtio::{DESC_F_NEXT, DESC_F_WRITE, VirtioDev};
    use crate::disk::Disk;

    let root = f.device_with_tables(DEVICE, 3);
    let phys = |iova: u64| DATA_AT + iova;
    for iova in (0x1000..=0x6000).step_by(0x1000) {
        f.map_4k(root, 3, iova, phys(iova), true);
    }
    if let Some(writable) = data_page {
        f.map_4k(root, 3, 0xa000, phys(0xa000), writable);
    }
    let mut queue = setup_queue(&mut f.ram, 0x1000, 0x2000, 0x3000, 8);
    let ram = &mut f.ram;
    put_desc(
        &mut ram[phys(0x1000) as usize..],
        0,
        0,
        0x4000,
        16,
        DESC_F_NEXT,
        1,
    );
    put_desc(
        &mut ram[phys(0x1000) as usize..],
        0,
        1,
        0xa000,
        512,
        DESC_F_WRITE | DESC_F_NEXT,
        2,
    );
    put_desc(
        &mut ram[phys(0x1000) as usize..],
        0,
        2,
        0x6000,
        1,
        DESC_F_WRITE,
        0,
    );
    post(&mut ram[phys(0x2000) as usize..], 0, 8, 0);
    let mut disk = VirtioBlk::new(Disk::memory(vec![0x5a; 4096]));
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    mem.with_dma(&mut f.iommu, DEVICE, |mem| {
        disk.kick(0, std::slice::from_mut(&mut queue), mem);
    });
    f.iommu.service(&mut mem);
}

#[test]
fn virtio_buffer_in_an_unmapped_page_is_refused_and_logged() {
    let mut f = Fixture::enabled();
    block_read_into_0xa000(&mut f, None);
    let faults = f.events();
    let fault = faults
        .iter()
        .find(|event| event.address == 0xa000)
        .expect("a fault at the buffer's device address");
    assert_eq!(fault.code, EVENT_IO_PAGE_FAULT);
    assert_eq!(fault.requester, DEVICE);
    assert_eq!(fault.domain, DOMAIN);
}

#[test]
fn virtio_write_into_a_read_only_page_logs_a_write_fault_and_leaves_it_untouched() {
    let mut f = Fixture::enabled();
    block_read_into_0xa000(&mut f, Some(false));
    assert_eq!(f.ram[(DATA_AT + 0xa000) as usize], 0);
    let faults = f.events();
    let fault = faults
        .iter()
        .find(|event| event.address == 0xa000)
        .expect("a fault at the buffer's device address");
    assert_eq!(fault.flags, events::FLAG_PRESENT | events::FLAG_WRITE);
}

#[test]
fn virtio_write_into_a_writable_page_succeeds_without_events() {
    let mut f = Fixture::enabled();
    block_read_into_0xa000(&mut f, Some(true));
    assert_eq!(f.ram[(DATA_AT + 0xa000) as usize], 0x5a);
    assert!(f.events().is_empty());
}

fn interrupt_fixture(table_length: u64, control: u64) -> Fixture {
    let mut f = Fixture::enabled();
    f.mmio_write(REG_CONTROL, 8, CONTROL_ALL | control);
    f.set_dte(
        DEVICE,
        [
            DTE_V,
            DOMAIN as u64,
            1 | table_length << 1 | INTERRUPT_TABLE_AT | 2 << 60,
            0,
        ],
    );
    f
}

fn msi(requester: u16, index: u32, mode: u32) -> InterruptRequest {
    InterruptRequest {
        requester,
        address: 0xfee0_0000,
        data: index | mode << 8,
    }
}

fn remap(f: &mut Fixture, request: InterruptRequest) -> Result<RemappedInterrupt, InterruptFault> {
    let mem = GuestMem::new(&mut f.ram, &[]);
    f.iommu.remap_interrupt(&mem, request)
}

#[test]
fn legacy_irte_supplies_vector_destination_and_type() {
    let mut f = interrupt_fixture(9, 0);
    // RemapEn, IntType fixed, DM logical, destination 0x12, vector 0x55.
    let irte: u32 = 1 | 1 << 6 | 0x12 << 8 | 0x55 << 16;
    f.ram[INTERRUPT_TABLE_AT as usize + 4 * 5..][..4].copy_from_slice(&irte.to_le_bytes());
    let remapped = remap(&mut f, msi(DEVICE, 5, 0)).unwrap();
    assert_eq!(remapped.vector, 0x55);
    assert_eq!(remapped.destination, 0x12);
    assert!(remapped.logical_destination);
    assert_eq!(remapped.delivery_mode, DeliveryMode::Fixed);
}

#[test]
fn interrupts_are_remapped_by_iv_alone_even_when_the_entry_is_not_valid() {
    // Linux clears V in the I/O APIC's entry and still expects its interrupts remapped.
    let mut f = interrupt_fixture(9, 0);
    f.set_dte(DEVICE, [0, 0, 1 | 9 << 1 | INTERRUPT_TABLE_AT | 2 << 60, 0]);
    let irte: u32 = 1 | 0x66 << 16;
    f.ram[INTERRUPT_TABLE_AT as usize + 4 * 7..][..4].copy_from_slice(&irte.to_le_bytes());
    assert_eq!(remap(&mut f, msi(DEVICE, 7, 0)).unwrap().vector, 0x66);
}

#[test]
fn an_entry_without_iv_passes_the_message_through_whatever_else_it_says() {
    let mut f = interrupt_fixture(9, 0);
    f.set_dte(DEVICE, [DTE_V, 0, INTERRUPT_TABLE_AT | 2 << 60, 0]);
    assert_eq!(remap(&mut f, msi(DEVICE, 0x41, 0)).unwrap().vector, 0x41);
}

#[test]
fn irte_arbitrated_type_is_lowest_priority_delivery() {
    let mut f = interrupt_fixture(9, 0);
    let irte: u32 = 1 | 1 << 2 | 0x80 << 16;
    f.ram[INTERRUPT_TABLE_AT as usize..][..4].copy_from_slice(&irte.to_le_bytes());
    let remapped = remap(&mut f, msi(DEVICE, 0, 0)).unwrap();
    assert_eq!(remapped.delivery_mode, DeliveryMode::LowestPriority);
}

#[test]
fn guest_apic_mode_reads_128_bit_irtes_with_a_24_bit_destination() {
    let mut f = interrupt_fixture(9, CONTROL_GUEST_APIC_ENABLE);
    let low: u64 = 1 | 0x00ab_cdef << 8;
    let high: u64 = 0x77;
    let at = INTERRUPT_TABLE_AT + 16 * 3;
    f.write64(at, low);
    f.write64(at + 8, high);
    let remapped = remap(&mut f, msi(DEVICE, 3, 0)).unwrap();
    assert_eq!(remapped.vector, 0x77);
    assert_eq!(remapped.destination, 0x00ab_cdef);
}

#[test]
fn irte_with_remap_disabled_aborts_and_logs_unless_suppressed() {
    let mut f = interrupt_fixture(9, 0);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 1, 0)),
        Err(InterruptFault::TargetAbort)
    );
    // Any register write services the IOMMU; this one leaves the command ring as it is.
    let tail = f.command_tail;
    f.mmio_write(REG_COMMAND_TAIL, 8, tail);
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    f.iommu.service(&mut mem);
    let event = expect_one(f.events());
    assert_eq!(event.code, EVENT_IO_PAGE_FAULT);
    assert_ne!(event.flags & events::FLAG_INTERRUPT, 0);
    assert_eq!(event.address, 0xfee0_0000);
    // SuppIOPF: the same abort, no event.
    f.ram[INTERRUPT_TABLE_AT as usize + 8..][..4].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(
        remap(&mut f, msi(DEVICE, 2, 0)),
        Err(InterruptFault::TargetAbort)
    );
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    f.iommu.service(&mut mem);
    assert!(f.events().is_empty());
}

#[test]
fn irte_index_beyond_the_table_length_is_rejected() {
    let mut f = interrupt_fixture(1, 0);
    f.ram[INTERRUPT_TABLE_AT as usize + 4..][..4].copy_from_slice(&1u32.to_le_bytes());
    assert!(
        remap(&mut f, msi(DEVICE, 1, 0)).is_ok(),
        "2 entries: index 1 is the last"
    );
    assert_eq!(
        remap(&mut f, msi(DEVICE, 2, 0)),
        Err(InterruptFault::InterruptTable)
    );
}

#[test]
fn illegal_interrupt_table_length_is_rejected() {
    let mut f = interrupt_fixture(0, 0);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 0)),
        Err(InterruptFault::InterruptTable)
    );
    let mut f = interrupt_fixture(12, 0);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 0)),
        Err(InterruptFault::InterruptTable)
    );
}

#[test]
fn interrupt_control_forward_passes_the_message_unchanged() {
    let mut f = interrupt_fixture(9, 0);
    f.set_dte(DEVICE, [DTE_V, 0, 1 | 9 << 1 | 1 << 60, 0]);
    let request = InterruptRequest {
        requester: DEVICE,
        address: 0xfee0_0000 | 0x34 << 12 | 1 << 2,
        data: 0x61 | 1 << 15,
    };
    let remapped = remap(&mut f, request).unwrap();
    assert_eq!(remapped.vector, 0x61);
    assert_eq!(remapped.destination, 0x34);
    assert!(remapped.logical_destination);
    assert!(remapped.level_triggered);
}

#[test]
fn interrupt_control_abort_and_reserved_stop_the_interrupt() {
    let mut f = interrupt_fixture(9, 0);
    f.set_dte(DEVICE, [DTE_V, 0, 1 | 9 << 1, 0]);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 0)),
        Err(InterruptFault::TargetAbort)
    );
    f.set_dte(DEVICE, [DTE_V, 0, 1 | 9 << 1 | 3 << 60, 0]);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 0)),
        Err(InterruptFault::Unsupported)
    );
}

#[test]
fn devices_without_interrupt_remapping_pass_through() {
    let mut f = interrupt_fixture(9, 0);
    f.set_dte(DEVICE, [DTE_V, 0, 0, 0]);
    let remapped = remap(&mut f, msi(DEVICE, 0x30, 0)).unwrap();
    assert_eq!(remapped.vector, 0x30);
    let remapped = remap(&mut f, msi(DEVICE + 8, 0x31, 0)).unwrap();
    assert_eq!(remapped.vector, 0x31, "a device with no entry at all (V=0)");
}

#[test]
fn nmi_init_and_extint_need_their_pass_bits() {
    let mut f = interrupt_fixture(9, 0);
    let base = 1 | 9 << 1 | INTERRUPT_TABLE_AT | 2 << 60;
    for (mode, delivery, bit) in [
        (4, DeliveryMode::Nmi, 58),
        (5, DeliveryMode::Init, 56),
        (7, DeliveryMode::ExtInt, 57),
    ] {
        f.set_dte(DEVICE, [DTE_V, 0, base, 0]);
        assert_eq!(
            remap(&mut f, msi(DEVICE, 0, mode)),
            Err(InterruptFault::TargetAbort),
            "mode {mode} without its pass bit"
        );
        f.set_dte(DEVICE, [DTE_V, 0, base | 1 << bit, 0]);
        assert_eq!(
            remap(&mut f, msi(DEVICE, 0, mode)).unwrap().delivery_mode,
            delivery
        );
    }
    f.set_dte(DEVICE, [DTE_V, 0, base | 1 << 63, 0]);
    assert!(
        remap(&mut f, msi(DEVICE, 0, 4)).is_ok(),
        "LINT1 pass carries NMI"
    );
    f.set_dte(DEVICE, [DTE_V, 0, base | 1 << 62, 0]);
    assert!(
        remap(&mut f, msi(DEVICE, 0, 7)).is_ok(),
        "LINT0 pass carries ExtInt"
    );
}

#[test]
fn smi_and_reserved_delivery_modes_are_unsupported_and_not_logged() {
    let mut f = interrupt_fixture(9, 0);
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 2)),
        Err(InterruptFault::Unsupported)
    );
    assert_eq!(
        remap(&mut f, msi(DEVICE, 0, 3)),
        Err(InterruptFault::Unsupported)
    );
    let mut mem = GuestMem::new(&mut f.ram, &[]);
    f.iommu.service(&mut mem);
    assert!(f.events().is_empty());
}

#[test]
fn message_outside_the_interrupt_window_is_rejected_for_a_remapped_device() {
    let mut f = interrupt_fixture(9, 0);
    let request = InterruptRequest {
        requester: DEVICE,
        address: 0xfed0_0000,
        data: 0,
    };
    assert_eq!(remap(&mut f, request), Err(InterruptFault::InvalidAddress));
}

#[test]
fn unknown_requester_beyond_the_device_table_cannot_interrupt() {
    let mut f = interrupt_fixture(9, 0);
    assert_eq!(
        remap(&mut f, msi(300, 0, 0)),
        Err(InterruptFault::DeviceTable)
    );
}

#[test]
fn interrupts_are_untouched_while_the_iommu_is_off() {
    let mut f = Fixture::new();
    let remapped = remap(&mut f, msi(DEVICE, 0x42, 0)).unwrap();
    assert_eq!(remapped.vector, 0x42);
    assert_eq!(remapped.delivery_mode, DeliveryMode::Fixed);
}

#[test]
fn ioapic_entry_becomes_an_msi_the_remapper_understands() {
    let request = ioapic_request(0xa0, 0x10, DeliveryMode::LowestPriority, true, true, 0x7f);
    assert_eq!(request.requester, 0xa0);
    assert_eq!(request.address, 0xfee0_0000 | 0x7f << 12 | 1 << 2);
    assert_eq!(request.data, 0x10 | 1 << 8 | 1 << 15);
    let mut f = Fixture::new();
    let remapped = remap(&mut f, request).unwrap();
    assert_eq!(remapped.vector, 0x10);
    assert_eq!(remapped.delivery_mode, DeliveryMode::LowestPriority);
    assert_eq!(remapped.destination, 0x7f);
    assert!(remapped.logical_destination && remapped.level_triggered);
}

fn ivrs_blocks(table: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut blocks = Vec::new();
    let mut at = 48;
    while at < table.len() {
        let length = u16::from_le_bytes([table[at + 2], table[at + 3]]) as usize;
        blocks.push((table[at], table[at..at + length].to_vec()));
        at += length;
    }
    blocks
}

#[test]
fn ivrs_table_is_a_valid_acpi_table_describing_the_unit() {
    let iommu = Iommu::new();
    let topology = IvrsTopology {
        pci_devices: vec![0x0000, 0x0002, 0x0018, 0x0020],
        ioapics: vec![SpecialDevice {
            handle: 4,
            requester: 0x00a0,
        }],
        hpets: vec![SpecialDevice {
            handle: 0,
            requester: 0x00a0,
        }],
    };
    let table = iommu.ivrs_table(&topology);
    assert_eq!(&table[..4], b"IVRS");
    assert_eq!(
        u32::from_le_bytes(table[4..8].try_into().unwrap()) as usize,
        table.len()
    );
    assert_eq!(
        table.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)),
        0
    );
    let ivinfo = u32::from_le_bytes(table[36..40].try_into().unwrap());
    assert_eq!(ivinfo & 1, 1, "EFRSup");
    assert_eq!((ivinfo >> 8) & 0x7f, 48, "PASize");

    let blocks = ivrs_blocks(&table);
    let types: Vec<u8> = blocks.iter().map(|(ty, _)| *ty).collect();
    assert_eq!(types, [0x10, 0x11, 0x40]);
    for (ty, block) in &blocks {
        let device_id = u16::from_le_bytes([block[4], block[5]]);
        assert_eq!(device_id, REQUESTER_ID);
        assert_eq!(
            u16::from_le_bytes([block[6], block[7]]),
            CAPABILITY_AT as u16
        );
        assert_eq!(
            u64::from_le_bytes(block[8..16].try_into().unwrap()),
            MMIO_BASE
        );
        let header = if *ty == 0x10 { 24 } else { 40 };
        if *ty != 0x10 {
            assert_eq!(
                u64::from_le_bytes(block[24..32].try_into().unwrap()),
                EXTENDED_FEATURES,
                "EFR image"
            );
        }
        let entries = &block[header..];
        assert_eq!(entries.len(), 4 * 4 + 2 * 8);
        for (i, device) in [0x0000u16, 0x0002, 0x0018, 0x0020].iter().enumerate() {
            assert_eq!(entries[4 * i], 0x02, "select entry");
            assert_eq!(
                u16::from_le_bytes([entries[4 * i + 1], entries[4 * i + 2]]),
                *device
            );
        }
        let ioapic = &entries[16..24];
        assert_eq!(ioapic[0], 0x48, "special entry");
        assert_eq!(ioapic[4], 4, "IOAPIC id");
        assert_eq!(u16::from_le_bytes([ioapic[5], ioapic[6]]), 0x00a0);
        assert_eq!(ioapic[7], 1, "variety: IOAPIC");
        let hpet = &entries[24..32];
        assert_eq!(hpet[7], 2, "variety: HPET");
    }
}

#[test]
fn ivrs_without_special_devices_lists_only_pci_functions() {
    let iommu = Iommu::new();
    let table = iommu.ivrs_table(&IvrsTopology {
        pci_devices: vec![8],
        ..Default::default()
    });
    let (_, block) = &ivrs_blocks(&table)[0];
    assert_eq!(block.len(), 24 + 4);
}

#[test]
fn pci_host_exposes_the_iommu_function_only_when_asked() {
    use crate::dev::pci::{CONFIG_ADDRESS_PORT, CONFIG_DATA_PORT, PciHost};
    let config_address = |devfn: u32, reg: u32| 1 << 31 | devfn << 8 | reg;
    let mut off = PciHost::new(Vec::new());
    off.io_out(CONFIG_ADDRESS_PORT, 4, config_address(2, 0));
    assert_eq!(off.io_in(CONFIG_DATA_PORT, 4), u32::MAX);

    let mut on = PciHost::new(Vec::new()).with_iommu(true);
    on.io_out(CONFIG_ADDRESS_PORT, 4, config_address(2, 0));
    assert_eq!(on.io_in(CONFIG_DATA_PORT, 4), 0x1419_1022);
    on.io_out(CONFIG_ADDRESS_PORT, 4, config_address(0, 0x0c));
    assert_eq!(
        on.io_in(CONFIG_DATA_PORT, 4) >> 16 & 0xff,
        0x80,
        "multifunction bridge"
    );
    on.io_out(
        CONFIG_ADDRESS_PORT,
        4,
        config_address(2, CAP_BASE_LOW as u32),
    );
    on.io_out(CONFIG_DATA_PORT, 4, 0xfed4_0001);
    assert_eq!(on.io_in(CONFIG_DATA_PORT, 4), 0xfed4_0001);

    let mut ram = vec![0u8; 0x1000];
    let mut mem = GuestMem::new(&mut ram, &[]);
    assert_eq!(on.iommu_mmio_read(0xfed4_0030, 8), Some(EXTENDED_FEATURES));
    assert_eq!(
        on.iommu_mmio_read(MMIO_BASE + 0x30, 8),
        None,
        "the window moved"
    );
    assert!(on.iommu_mmio_write(0xfed4_0018, 8, CONTROL_IOMMU_ENABLE, &mut mem));
    assert_eq!(
        on.iommu_mmio_read(0xfed4_0018, 8),
        Some(CONTROL_IOMMU_ENABLE)
    );
    assert!(!on.iommu_mmio_write(MMIO_BASE + 0x18, 8, 0, &mut mem));
}

#[test]
fn devices_route_the_register_window_only_when_the_iommu_is_configured() {
    use crate::console::ConsoleHandle;
    use crate::dev::{DevConfig, Devices};
    let mut ram = vec![0u8; 0x1000];
    let mut off = Devices::new(ConsoleHandle::default(), DevConfig::default());
    assert_eq!(off.mmio_read(MMIO_BASE + 0x30, 8), None);
    let mut on = Devices::new(
        ConsoleHandle::default(),
        DevConfig {
            iommu: true,
            ..Default::default()
        },
    );
    assert_eq!(on.mmio_read(MMIO_BASE + 0x30, 8), Some(EXTENDED_FEATURES));
    let mut mem = GuestMem::new(&mut ram, &[]);
    assert!(on.mmio_write(MMIO_BASE + 0x18, 8, CONTROL_IOMMU_ENABLE, &mut mem));
    assert_eq!(
        on.mmio_read(MMIO_BASE + 0x18, 8),
        Some(CONTROL_IOMMU_ENABLE)
    );
}

#[test]
fn register_window_stays_clear_of_the_fch_block() {
    // Linux reads the FCH reset status at 0xFED80000 + 0x3c0 on every AMD machine.
    const FCH_BLOCK: core::ops::Range<u64> = 0xfed8_0000..0xfed9_0000;
    assert!(!FCH_BLOCK.contains(&MMIO_BASE));
    assert!(!FCH_BLOCK.contains(&(MMIO_BASE + MMIO_WINDOW_BYTES - 1)));
}
