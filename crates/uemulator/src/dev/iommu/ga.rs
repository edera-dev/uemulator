//! Guest virtual APIC mode (spec 2.2.5, "guest virtual APIC"; EFR GAMSup = 1, control GAMEn = 1):
//! an IRTE in guest mode does not interrupt the host. The IOMMU sets the vector in the IRR of
//! the guest's vAPIC backing page, the page the processor's AVIC (`cpu/svm_avic.rs`) reads, and
//! then either rings the doorbell of the processor the vCPU runs on (IsRun) or, when it is not
//! running and the IRTE asks for it (GALogIntr), writes the IRTE's GATag to the GA log and
//! interrupts the host, which is how KVM (`avic_ga_log_notifier`) learns to wake that vCPU.
//!
//! The doorbell is implied: the AVIC model looks at the backing page's IRR at every guest
//! instruction boundary (JIT block boundary under the JIT), as it does for the doorbell MSR. A
//! vCPU halted in guest mode without the HLT intercept is therefore not woken by it.
//!
//! Remapping only reads guest memory (`remap_interrupt` runs on paths that have nothing else),
//! so a post is queued and `service` performs it with the writes it needs.

use super::{
    CONTROL_GUEST_APIC_ENABLE, CONTROL_INTCAPXT_ENABLE, Iommu, PhysWrite, RING_LENGTH_MASK,
    RING_LENGTH_SHIFT,
};
#[cfg(freestanding)]
use crate::prelude::*;

/// GA log base (address and GALogLen, as the event log's), the address the tail pointer is
/// written to, and the head and tail offsets.
pub(super) const REG_GA_LOG_BASE: u64 = 0x00e0;
pub(super) const REG_GA_LOG_TAIL_ADDRESS: u64 = 0x00e8;
pub(super) const REG_GA_LOG_HEAD: u64 = 0x2040;
pub(super) const REG_GA_LOG_TAIL: u64 = 0x2048;
pub(super) const GA_LOG_BASE_WRITABLE: u64 = 0x0f0f_ffff_ffff_f000;
pub(super) const GA_LOG_TAIL_ADDRESS_WRITABLE: u64 = 0x000f_ffff_ffff_fff8;
/// Head and tail are byte offsets of 8-byte entries.
pub(super) const GA_LOG_POINTER_MASK: u64 = 0x7_fff8;
const GA_LOG_ENTRY_BYTES: u64 = 8;
const GA_LOG_MIN_LENGTH: u64 = 8;
/// A GALogLen of n is 2^n 16-byte units, as for the event log: Linux's 8 is 4 KiB, 512 entries.
const GA_LOG_LENGTH_UNIT_BYTES: u64 = 16;

/// Control: GAMEn (27:25; 1 is virtual APIC mode, 0 legacy), GALogEn, GAIntEn.
pub(super) const CONTROL_GA_MODE_MASK: u64 = 0b111 << 25;
const CONTROL_GA_MODE_VIRTUAL_APIC: u64 = 1 << 25;
pub(super) const CONTROL_GA_LOG_ENABLE: u64 = 1 << 28;
pub(super) const CONTROL_GA_INTERRUPT_ENABLE: u64 = 1 << 29;
/// Status: GALogRun mirrors GALogEn until an overflow stops the log; overflow and the interrupt
/// bit are write-1-to-clear.
pub(super) const STATUS_GA_LOG_RUNNING: u64 = 1 << 8;
pub(super) const STATUS_GA_LOG_OVERFLOW: u64 = 1 << 9;
pub(super) const STATUS_GA_LOG_INTERRUPT: u64 = 1 << 10;

/// GA log entry: the GATag (31:0), the requester (47:32) and the type (63:60), 1 being "guest
/// not running" (Linux's `GA_GUEST_NR`).
const ENTRY_DEVICE_SHIFT: u32 = 32;
const ENTRY_TYPE_GUEST_NOT_RUNNING: u64 = 1 << 60;

/// The 128-bit IRTE in guest mode: GALogIntr (2), IsRun (6), the doorbell destination (31:8 and
/// high 63:56, not needed here), GATag (low 63:32); the vector (high 7:0) and the backing page
/// (high 51:12, `ga_root_ptr` in Linux).
const IRTE_GA_LOG_INTERRUPT: u64 = 1 << 2;
const IRTE_IS_RUNNING: u64 = 1 << 6;
const IRTE_GA_TAG_SHIFT: u32 = 32;
const IRTE_BACKING_PAGE_MASK: u64 = 0x000f_ffff_ffff_f000;

/// The vAPIC backing page's IRR: eight 32-bit words, 16 bytes apart, from offset 0x200.
const BACKING_IRR: u64 = 0x200;
const BACKING_REGISTER_STRIDE: u64 = 0x10;

/// An interrupt for a guest's virtual APIC, to be written by `service`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Post {
    backing_page: u64,
    vector: u8,
    /// GATag and requester, when the vCPU is not running and the IRTE asks for a GA log entry.
    log: Option<(u32, u16)>,
}

impl Iommu {
    /// GAEn and GAMEn = virtual APIC: guest-mode IRTEs post.
    pub(super) fn virtual_apic_mode(&self) -> bool {
        self.control() & CONTROL_GUEST_APIC_ENABLE != 0
            && self.control() & CONTROL_GA_MODE_MASK == CONTROL_GA_MODE_VIRTUAL_APIC
    }

    /// Queues the post a guest-mode IRTE (`low`, `high`) describes for `requester`.
    pub(super) fn queue_post(&mut self, requester: u16, low: u64, high: u64) {
        let log = (low & IRTE_IS_RUNNING == 0 && low & IRTE_GA_LOG_INTERRUPT != 0)
            .then_some(((low >> IRTE_GA_TAG_SHIFT) as u32, requester));
        self.pending_posts.push(Post {
            backing_page: high & IRTE_BACKING_PAGE_MASK,
            vector: high as u8,
            log,
        });
    }

    fn ga_log_bytes(&self) -> u64 {
        let length = (self.ga_log_base >> RING_LENGTH_SHIFT) & RING_LENGTH_MASK;
        if length < GA_LOG_MIN_LENGTH {
            0
        } else {
            GA_LOG_LENGTH_UNIT_BYTES << length
        }
    }

    /// GALogEn written: the log starts or stops running.
    pub(super) fn ga_log_enable_changed(&mut self, enabled: bool) {
        if enabled {
            self.status_set(STATUS_GA_LOG_RUNNING);
        } else {
            self.status_clear(STATUS_GA_LOG_RUNNING);
        }
    }

    /// Performs the queued posts: the IRR bit (atomically: the vCPU clears others as it accepts
    /// them), then any GA log entry, which interrupts the host if GAIntEn is set.
    pub(super) fn flush_posts(&mut self, mem: &mut dyn PhysWrite) {
        let mut logged = false;
        for post in core::mem::take(&mut self.pending_posts) {
            let word = post.backing_page
                + BACKING_IRR
                + (post.vector / 32) as u64 * BACKING_REGISTER_STRIDE;
            mem.fetch_or_u32(word, 1 << (post.vector % 32));
            if let Some((tag, requester)) = post.log {
                logged |= self.write_ga_log(mem, tag, requester);
            }
        }
        if logged {
            self.status_set(STATUS_GA_LOG_INTERRUPT);
            if self.control() & CONTROL_GA_INTERRUPT_ENABLE != 0 {
                if self.control() & CONTROL_INTCAPXT_ENABLE != 0 {
                    self.ga_interrupt_pending = true;
                } else {
                    self.raise_msi();
                }
            }
        }
    }

    /// One GA log entry, unless the log is off or stopped; a full log stops it with
    /// GALogOverflow, as the event log does.
    fn write_ga_log(&mut self, mem: &mut dyn PhysWrite, tag: u32, requester: u16) -> bool {
        let bytes = self.ga_log_bytes();
        if self.control() & CONTROL_GA_LOG_ENABLE == 0
            || self.status() & STATUS_GA_LOG_RUNNING == 0
            || bytes == 0
        {
            return false;
        }
        let next = (self.ga_log_tail + GA_LOG_ENTRY_BYTES) % bytes;
        if next == self.ga_log_head {
            self.status_set(STATUS_GA_LOG_OVERFLOW);
            self.status_clear(STATUS_GA_LOG_RUNNING);
            return false;
        }
        let entry =
            tag as u64 | (requester as u64) << ENTRY_DEVICE_SHIFT | ENTRY_TYPE_GUEST_NOT_RUNNING;
        let base = self.ga_log_base & super::BASE_ADDRESS_MASK;
        if !mem.write_phys(base + self.ga_log_tail, &entry.to_le_bytes()) {
            return false;
        }
        self.ga_log_tail = next;
        // The tail is also stored where software said (Linux gives it 8 bytes and never looks).
        if self.ga_log_tail_address != 0 {
            mem.write_phys(self.ga_log_tail_address, &next.to_le_bytes());
        }
        true
    }
}
