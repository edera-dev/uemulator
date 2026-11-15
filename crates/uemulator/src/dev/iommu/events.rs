//! The event log (spec 2.5): 16-byte entries the IOMMU writes at the tail for software to read
//! at the head. Dword 1 bits 31:28 hold the event code, bits 27:16 flags and bits 15:0 the domain.
use super::{
    CONTROL_EVENT_INTERRUPT_ENABLE, CONTROL_EVENT_LOG_ENABLE, Iommu, PhysWrite, RING_ENTRY_BYTES,
    STATUS_EVENT_INTERRUPT, STATUS_EVENT_LOG_RUNNING, STATUS_EVENT_OVERFLOW,
};

const EVENT_CODE_SHIFT: u32 = 28;
const EVENT_FLAGS_SHIFT: u32 = 16;
const EVENT_ILLEGAL_DEVICE_TABLE_ENTRY: u32 = 0x1;
const EVENT_IO_PAGE_FAULT: u32 = 0x2;
const EVENT_DEVICE_TABLE_HW_ERROR: u32 = 0x3;
const EVENT_PAGE_TABLE_HW_ERROR: u32 = 0x4;
const EVENT_ILLEGAL_COMMAND: u32 = 0x5;
const EVENT_COMMAND_HW_ERROR: u32 = 0x6;

/// IO_PAGE_FAULT flags as the Linux driver names them (`EVENT_FLAG_*`, dword 1 bits 27:16): the
/// access asked for execute permission, the entry was present, the access was a write, a
/// reserved bit was set, it was an interrupt.
pub(super) const FLAG_NO_EXECUTE: u16 = 0x002;
pub(super) const FLAG_INTERRUPT: u16 = 0x008;
pub(super) const FLAG_PRESENT: u16 = 0x010;
pub(super) const FLAG_WRITE: u16 = 0x020;

fn event(requester: u16, domain: u16, code: u32, flags: u16, address: u64) -> [u32; 4] {
    [
        requester as u32,
        domain as u32 | (flags as u32) << EVENT_FLAGS_SHIFT | code << EVENT_CODE_SHIFT,
        address as u32,
        (address >> 32) as u32,
    ]
}

impl Iommu {
    fn queue_event(&mut self, event: [u32; 4]) {
        self.pending_events.push(event);
    }

    pub(super) fn log_page_fault(&mut self, requester: u16, domain: u16, address: u64, flags: u16) {
        self.queue_event(event(
            requester,
            domain,
            EVENT_IO_PAGE_FAULT,
            flags,
            address,
        ));
    }

    pub(super) fn log_illegal_device_entry(&mut self, requester: u16, address: u64, flags: u16) {
        self.queue_event(event(
            requester,
            0,
            EVENT_ILLEGAL_DEVICE_TABLE_ENTRY,
            flags,
            address,
        ));
    }

    /// `address` is the physical address of the device table entry that could not be read.
    pub(super) fn log_device_table_error(&mut self, requester: u16, address: u64, flags: u16) {
        self.queue_event(event(
            requester,
            0,
            EVENT_DEVICE_TABLE_HW_ERROR,
            flags,
            address,
        ));
    }

    /// `address` is the physical address of the page table entry that could not be read.
    pub(super) fn log_page_table_error(
        &mut self,
        requester: u16,
        domain: u16,
        address: u64,
        flags: u16,
    ) {
        self.queue_event(event(
            requester,
            domain,
            EVENT_PAGE_TABLE_HW_ERROR,
            flags,
            address,
        ));
    }

    /// `address` is the physical address of the command, which `dump_command` in Linux reads back.
    pub(super) fn log_illegal_command(&mut self, address: u64) {
        self.queue_event(event(0, 0, EVENT_ILLEGAL_COMMAND, 0, address));
    }

    pub(super) fn log_command_error(&mut self, address: u64) {
        self.queue_event(event(0, 0, EVENT_COMMAND_HW_ERROR, 0, address));
    }

    /// Moves queued events into the guest's log. They are dropped, as on hardware, while the log
    /// is off or has stopped after an overflow (software restarts it by toggling EventLogEn).
    /// Writing the log sets EventLogInt, and raises the MSI if software enabled event interrupts.
    pub(super) fn flush_events(&mut self, mem: &mut dyn PhysWrite) {
        if self.pending_events.is_empty() {
            return;
        }
        let events = core::mem::take(&mut self.pending_events);
        let (base, bytes) = self.event_log();
        if self.control() & CONTROL_EVENT_LOG_ENABLE == 0
            || self.status() & STATUS_EVENT_LOG_RUNNING == 0
            || bytes == 0
        {
            return;
        }
        let mut logged = false;
        for event in events {
            let (head, tail) = self.event_pointers();
            let next = (tail + RING_ENTRY_BYTES) % bytes;
            if next == head {
                self.status_set(STATUS_EVENT_OVERFLOW);
                self.status_clear(STATUS_EVENT_LOG_RUNNING);
                break;
            }
            let mut entry = [0u8; RING_ENTRY_BYTES as usize];
            for (chunk, dword) in entry.as_chunks_mut::<4>().0.iter_mut().zip(event) {
                chunk.copy_from_slice(&dword.to_le_bytes());
            }
            if !mem.write_phys(base + tail, &entry) {
                break;
            }
            self.set_event_tail(next);
            logged = true;
        }
        if logged {
            self.status_set(STATUS_EVENT_INTERRUPT);
            if self.control() & CONTROL_EVENT_INTERRUPT_ENABLE != 0 {
                self.raise_msi();
            }
        }
    }
}
