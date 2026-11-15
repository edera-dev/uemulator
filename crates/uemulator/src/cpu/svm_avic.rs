//! AVIC and x2AVIC (APM 15.29): the guest's local APIC lives in a backing page the hypervisor
//! owns (KVM makes it the very page its own APIC model keeps the registers in). Reads come from
//! the page, TPR and edge EOI writes and fixed-mode IPIs complete without an exit, and everything
//! else exits to the hypervisor with the register's offset: after the access (a trap, for
//! registers whose write it has to act on) or before it (a fault, which it emulates).
//!
//! The xAPIC registers are reached through the guest-physical page in the VMCB's APIC BAR field
//! (the nested tables must map it; the access then goes to the backing page instead), x2APIC ones
//! through the MSRs 0x800-0x8FF the MSR permission map lets through, with x2AVIC.
//!
//! Interrupts are evaluated from the backing page at every guest instruction boundary (JIT block
//! boundary under the JIT), which covers what a doorbell would trigger: the doorbell MSR is
//! accepted and does nothing more.

use super::*;

pub(super) const VMCB_AVIC_APIC_BAR: u64 = 0x98;
pub(super) const VMCB_AVIC_BACKING_PAGE: u64 = 0xe0;
pub(super) const VMCB_AVIC_LOGICAL_TABLE: u64 = 0xf0;
pub(super) const VMCB_AVIC_PHYSICAL_TABLE: u64 = 0xf8;
pub(super) const AVIC_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
/// The highest index the physical APIC id table has, in the low bits of its VMCB field (9 bits:
/// x2AVIC allows 512 entries).
pub(super) const AVIC_PHYSICAL_MAX_INDEX_MASK: u64 = 0x1ff;
/// int_ctl (VMCB 0x60).
pub(super) const AVIC_ENABLE: u64 = 1 << 31;
pub(super) const X2AVIC_ENABLE: u64 = 1 << 30;
/// V_INTR_MASKING: the guest's CR8 is V_TPR, not the APIC's TPR.
pub(super) const V_INTR_MASKING: u64 = 1 << 24;

/// Physical APIC id table entries (8 bytes): the backing page of the vCPU with that guest APIC
/// id, whether it is in guest mode now (a doorbell reaches it), and whether the entry is in use.
pub(super) const PHYSICAL_ENTRY_BACKING_PAGE_MASK: u64 = 0x000f_ffff_ffff_f000;
pub(super) const PHYSICAL_ENTRY_IS_RUNNING: u64 = 1 << 62;
pub(super) const PHYSICAL_ENTRY_VALID: u64 = 1 << 63;
pub(super) const PHYSICAL_ENTRY_BYTES: u64 = 8;
/// Logical APIC id table entries (4 bytes): a guest physical APIC id.
pub(super) const LOGICAL_ENTRY_GUEST_PHYSICAL_ID_MASK: u64 = 0xff;
pub(super) const LOGICAL_ENTRY_VALID: u64 = 1 << 31;
pub(super) const LOGICAL_ENTRY_BYTES: u64 = 4;

pub(super) const VMEXIT_AVIC_INCOMPLETE_IPI: u64 = 0x401;
pub(super) const VMEXIT_AVIC_NOACCEL: u64 = 0x402;
/// AVIC_INCOMPLETE_IPI: EXITINFO2 has the cause in bits 63:32 and the table index in 8:0.
pub(super) const IPI_FAILURE_INVALID_INT_TYPE: u64 = 0;
pub(super) const IPI_FAILURE_TARGET_NOT_RUNNING: u64 = 1;
pub(super) const IPI_FAILURE_INVALID_TARGET: u64 = 2;
pub(super) const IPI_FAILURE_INVALID_BACKING_PAGE: u64 = 3;
pub(super) const IPI_FAILURE_INVALID_VECTOR: u64 = 4;
pub(super) const IPI_FAILURE_SHIFT: u32 = 32;
/// AVIC_NOACCEL: EXITINFO1 has the register offset in bits 11:4 and bit 32 set for a write.
pub(super) const NOACCEL_OFFSET_MASK: u64 = 0xff0;
pub(super) const NOACCEL_WRITE: u64 = 1 << 32;

/// Where `nested_translate_with_rights` sends the guest's APIC BAR page: above any physical
/// address, so the soft TLB never caches it (it only caches RAM) and every access reaches
/// `avic_mmio_read`/`avic_mmio_write` through the slow path of `mem_read`/`mem_write`.
pub(crate) const AVIC_MMIO_WINDOW: u64 = 1 << 62;

/// APIC register offsets (in the page; x2APIC MSR 0x800 + offset / 16).
pub(super) const APIC_ID: u64 = 0x20;
pub(super) const APIC_VERSION: u64 = 0x30;
pub(super) const APIC_TPR: u64 = 0x80;
pub(super) const APIC_APR: u64 = 0x90;
pub(super) const APIC_PPR: u64 = 0xa0;
pub(super) const APIC_EOI: u64 = 0xb0;
pub(super) const APIC_LDR: u64 = 0xd0;
pub(super) const APIC_DFR: u64 = 0xe0;
pub(super) const APIC_SVR: u64 = 0xf0;
pub(super) const APIC_ISR: u64 = 0x100;
pub(super) const APIC_TMR: u64 = 0x180;
pub(super) const APIC_IRR: u64 = 0x200;
pub(super) const APIC_ESR: u64 = 0x280;
pub(super) const APIC_ICR_LOW: u64 = 0x300;
pub(super) const APIC_ICR_HIGH: u64 = 0x310;
pub(super) const APIC_LVT_TIMER: u64 = 0x320;
pub(super) const APIC_LVT_ERROR: u64 = 0x370;
pub(super) const APIC_TIMER_INITIAL: u64 = 0x380;
pub(super) const APIC_TIMER_DIVIDE: u64 = 0x3e0;
pub(super) const APIC_SELF_IPI: u64 = 0x3f0;
pub(super) const APIC_REGISTER_STRIDE: u64 = 0x10;
pub(super) const APIC_VECTOR_WORDS: u64 = 8;
pub(super) const PRIORITY_CLASS_MASK: u32 = 0xf0;
pub(super) const X2APIC_MSR_BASE: u32 = 0x800;
pub(super) const X2APIC_OFFSET_SHIFT: u32 = 4;
pub(super) const XAPIC_DESTINATION_SHIFT: u32 = 24;
pub(super) const XAPIC_BROADCAST: u32 = 0xff;
/// DFR model bits 31:28: flat (0xF) or cluster (0x0).
pub(super) const DFR_MODEL_SHIFT: u32 = 28;
pub(super) const DFR_MODEL_FLAT: u32 = 0xf;
pub(super) const XAPIC_CLUSTER_BROADCAST: u32 = 0xf;
pub(super) const LOGICAL_FLAT_IDS: u32 = 8;
pub(super) const LOGICAL_CLUSTER_IDS: u32 = 4;

/// ICR fields.
pub(super) const ICR_VECTOR_MASK: u64 = 0xff;
pub(super) const ICR_DELIVERY_MODE_SHIFT: u32 = 8;
pub(super) const ICR_DELIVERY_MODE_MASK: u64 = 7;
pub(super) const ICR_DELIVERY_FIXED: u64 = 0;
pub(super) const ICR_LOGICAL: u64 = 1 << 11;
pub(super) const ICR_LEVEL_TRIGGERED: u64 = 1 << 15;
pub(super) const ICR_SHORTHAND_SHIFT: u32 = 18;
pub(super) const ICR_SHORTHAND_MASK: u64 = 3;
pub(super) const ICR_SHORTHAND_NONE: u64 = 0;
pub(super) const ICR_SHORTHAND_SELF: u64 = 1;
/// Vectors 0-15 are reserved (illegal in an IPI).
pub(super) const FIRST_LEGAL_VECTOR: u64 = 16;

/// The VMCB's AVIC setup, read at VMRUN.
pub(super) struct Avic {
    /// Guest-physical page of the xAPIC registers (unused in x2APIC mode).
    apic_bar: u64,
    backing_page: u64,
    logical_table: u64,
    physical_table: u64,
    physical_max_index: u64,
    x2apic: bool,
}

/// How a register write is handled (APM table 15-22).
#[derive(PartialEq, Debug)]
enum WriteKind {
    /// Completes in the backing page with no exit.
    Accelerated,
    /// Lands in the backing page, then AVIC_NOACCEL after the instruction.
    Trap,
    /// AVIC_NOACCEL before anything is written; the hypervisor emulates the instruction.
    Fault,
}

/// The register at `offset` reads from the backing page (else the read faults). The timer's
/// current count changes on its own, so only the hypervisor can answer it.
fn readable(offset: u64, x2apic: bool) -> bool {
    match offset {
        APIC_ID | APIC_VERSION | APIC_TPR | APIC_APR | APIC_PPR | APIC_LDR | APIC_SVR
        | APIC_ESR | APIC_ICR_LOW | APIC_TIMER_INITIAL | APIC_TIMER_DIVIDE => true,
        APIC_DFR | APIC_ICR_HIGH => !x2apic,
        APIC_ISR..APIC_ESR => true,
        APIC_LVT_TIMER..=APIC_LVT_ERROR => true,
        _ => false,
    }
}

fn write_kind(offset: u64, x2apic: bool) -> WriteKind {
    match offset {
        APIC_TPR | APIC_EOI | APIC_ICR_LOW => WriteKind::Accelerated,
        APIC_ICR_HIGH if !x2apic => WriteKind::Accelerated,
        APIC_SELF_IPI if x2apic => WriteKind::Accelerated,
        // x2APIC's ID and LDR are read-only and it has no DFR
        APIC_ID | APIC_LDR | APIC_DFR if !x2apic => WriteKind::Trap,
        APIC_SVR | APIC_ESR | APIC_TIMER_INITIAL | APIC_TIMER_DIVIDE => WriteKind::Trap,
        APIC_LVT_TIMER..=APIC_LVT_ERROR => WriteKind::Trap,
        _ => WriteKind::Fault,
    }
}

fn vector_bit(base: u64, vector: u8) -> (u64, u32) {
    (
        base + (vector as u64 / 32) * APIC_REGISTER_STRIDE,
        1 << (vector % 32),
    )
}

impl Avic {
    /// The AVIC fields of the VMCB at `vmcb`, if its int_ctl enables AVIC. Nested paging is
    /// required (KVM uses AVIC only with it, and the APIC BAR is a guest-physical address).
    pub(super) fn from_vmcb(cpu: &mut Cpu, vmcb: u64, int_ctl: u64, nested: bool) -> Option<Self> {
        if int_ctl & AVIC_ENABLE == 0 || !nested {
            return None;
        }
        let physical = cpu.phys_read(vmcb + VMCB_AVIC_PHYSICAL_TABLE, 8);
        Some(Avic {
            apic_bar: cpu.phys_read(vmcb + VMCB_AVIC_APIC_BAR, 8) & AVIC_ADDRESS_MASK,
            backing_page: cpu.phys_read(vmcb + VMCB_AVIC_BACKING_PAGE, 8) & AVIC_ADDRESS_MASK,
            logical_table: cpu.phys_read(vmcb + VMCB_AVIC_LOGICAL_TABLE, 8) & AVIC_ADDRESS_MASK,
            physical_table: physical & AVIC_ADDRESS_MASK,
            physical_max_index: physical & AVIC_PHYSICAL_MAX_INDEX_MASK,
            x2apic: int_ctl & X2AVIC_ENABLE != 0,
        })
    }
}

/// A trap exit an AVIC access owes once its instruction has completed (`svm_before_step`).
pub(super) struct PendingTrap {
    pub(super) code: u64,
    pub(super) info1: u64,
    pub(super) info2: u64,
}

impl Cpu {
    fn avic(&self) -> Option<&Avic> {
        self.svm
            .guest
            .as_ref()
            .and_then(|guest| guest.avic.as_ref())
    }

    /// The guest-physical page whose accesses go to the backing page instead.
    pub(super) fn avic_mmio_page(&self) -> Option<u64> {
        self.avic()
            .filter(|avic| !avic.x2apic)
            .map(|avic| avic.apic_bar)
    }

    fn backing_read(&mut self, offset: u64) -> u32 {
        let page = self.avic().map_or(0, |avic| avic.backing_page);
        self.phys_read(page + offset, 4) as u32
    }

    fn backing_write(&mut self, offset: u64, value: u32) {
        let page = self.avic().map_or(0, |avic| avic.backing_page);
        self.phys_write(page + offset, 4, value as u64);
    }

    /// Sets or clears `bits` in the 32-bit register at `address` (in some vCPU's backing page)
    /// atomically: other vCPUs set IRR bits in it concurrently, as KVM itself does with `lock or`.
    fn update_vector_word(&mut self, address: u64, set: u32, clear: u32) {
        // The registers are 16-byte aligned with 12 reserved bytes after them, so the 8-byte
        // compare-and-swap covers nothing else.
        loop {
            let seen = self.phys_read(address, 8);
            let updated = (seen & !(clear as u64)) | set as u64;
            if !self.ram.contains(address, 8) || self.update_ram_u64(address, seen, updated) {
                return;
            }
        }
    }

    /// The highest vector set in the 256-bit register at `base` of this vCPU's backing page.
    fn highest_vector(&mut self, base: u64) -> Option<u8> {
        (0..APIC_VECTOR_WORDS).rev().find_map(|word| {
            let bits = self.backing_read(base + word * APIC_REGISTER_STRIDE);
            (bits != 0).then(|| (word * 32) as u8 + (31 - bits.leading_zeros()) as u8)
        })
    }

    /// PPR from TPR and the highest in-service vector (SDM 12.8.3.1), kept in the backing page
    /// where the guest reads it.
    fn avic_update_ppr(&mut self) -> u32 {
        let tpr = self.backing_read(APIC_TPR) & 0xff;
        let in_service = self
            .highest_vector(APIC_ISR)
            .map_or(0, |vector| vector as u32 & PRIORITY_CLASS_MASK);
        let ppr = if tpr & PRIORITY_CLASS_MASK >= in_service {
            tpr
        } else {
            in_service
        };
        self.backing_write(APIC_PPR, ppr);
        ppr
    }

    pub(super) fn avic_tpr(&mut self) -> u8 {
        self.backing_read(APIC_TPR) as u8
    }

    pub(super) fn avic_set_tpr(&mut self, tpr: u8) {
        self.backing_write(APIC_TPR, tpr as u32);
        self.avic_update_ppr();
    }

    /// The backing page address the window address `address` stands for (`phys_read` and
    /// `phys_write` use it for the accesses that do not come through `mem_read`/`mem_write`,
    /// such as locked read-modify-writes: those see the raw page, without exits).
    pub(crate) fn avic_window_address(&self, address: u64) -> u64 {
        let page = self.avic().map_or(0, |avic| avic.backing_page);
        page + (address & (PAGE_SIZE - 1))
    }

    /// The interrupt the backing page has for the guest now, if its priority beats PPR; accepting
    /// it moves it from IRR to ISR. The caller has checked IF and the interrupt shadow.
    pub(super) fn avic_accept_interrupt(&mut self) -> Option<u8> {
        self.avic()?;
        let vector = self.highest_vector(APIC_IRR)?;
        let ppr = self.avic_update_ppr();
        if vector as u32 & PRIORITY_CLASS_MASK <= ppr & PRIORITY_CLASS_MASK {
            return None;
        }
        let page = self.avic()?.backing_page;
        let (irr, bit) = vector_bit(page + APIC_IRR, vector);
        self.update_vector_word(irr, 0, bit);
        let (isr, bit) = vector_bit(page + APIC_ISR, vector);
        self.update_vector_word(isr, bit, 0);
        self.avic_update_ppr();
        Some(vector)
    }

    /// Ends the instruction with a trap exit owed: the JIT stops the block after it, and
    /// `svm_before_step` performs the exit with RIP past the instruction.
    fn avic_trap(&mut self, code: u64, info1: u64, info2: u64) {
        if let Some(guest) = self.svm.guest.as_deref_mut() {
            guest.avic_trap = Some(PendingTrap { code, info1, info2 });
        }
        self.block_exit_requested = true;
    }

    fn noaccel_info(offset: u64, write: bool) -> u64 {
        (offset & NOACCEL_OFFSET_MASK) | if write { NOACCEL_WRITE } else { 0 }
    }

    fn avic_fault(&mut self, offset: u64, write: bool) -> Exception {
        self.vmexit(VMEXIT_AVIC_NOACCEL, Self::noaccel_info(offset, write), 0, 0)
    }

    /// A guest load from its APIC BAR page.
    pub(crate) fn avic_mmio_read(&mut self, address: u64, size: u8) -> ExecResult<u64> {
        let offset = address & (PAGE_SIZE - 1);
        if size != 4 || !offset.is_multiple_of(APIC_REGISTER_STRIDE) || !readable(offset, false) {
            return Err(self.avic_fault(offset, false));
        }
        Ok(self.backing_read(offset) as u64)
    }

    /// A guest store to its APIC BAR page.
    pub(crate) fn avic_mmio_write(&mut self, address: u64, size: u8, value: u64) -> ExecResult<()> {
        let offset = address & (PAGE_SIZE - 1);
        if size != 4 || !offset.is_multiple_of(APIC_REGISTER_STRIDE) {
            return Err(self.avic_fault(offset, true));
        }
        self.avic_register_write(offset, value as u32, None)
    }

    /// RDMSR of an x2APIC register in a guest with x2AVIC (`None` otherwise: the MSR is the
    /// processor's own).
    pub(crate) fn avic_x2apic_read(&mut self, msr: u32) -> Option<ExecResult<u64>> {
        if !self.avic().is_some_and(|avic| avic.x2apic) {
            return None;
        }
        let offset = ((msr - X2APIC_MSR_BASE) as u64) << X2APIC_OFFSET_SHIFT;
        if !readable(offset, true) {
            return Some(Err(self.avic_fault(offset, false)));
        }
        let mut value = self.backing_read(offset) as u64;
        if offset == APIC_ICR_LOW {
            value |= (self.backing_read(APIC_ICR_HIGH) as u64) << 32;
        }
        Some(Ok(value))
    }

    /// WRMSR of an x2APIC register in a guest with x2AVIC (`None` otherwise). The ICR is one
    /// 64-bit register with a 32-bit destination; EOI and SELF_IPI take only valid values
    /// without an exit (the hypervisor raises the #GP for the others).
    pub(crate) fn avic_x2apic_write(&mut self, msr: u32, value: u64) -> Option<ExecResult<()>> {
        if !self.avic().is_some_and(|avic| avic.x2apic) {
            return None;
        }
        let offset = ((msr - X2APIC_MSR_BASE) as u64) << X2APIC_OFFSET_SHIFT;
        let reserved = match offset {
            APIC_ICR_LOW => false,
            APIC_EOI => value != 0,
            APIC_SELF_IPI => value & !ICR_VECTOR_MASK != 0,
            _ => value >> 32 != 0,
        };
        if reserved {
            return Some(Err(self.avic_fault(offset, true)));
        }
        let destination = (offset == APIC_ICR_LOW).then_some((value >> 32) as u32);
        Some(self.avic_register_write(offset, value as u32, destination))
    }

    /// A register write, xAPIC or x2APIC. `x2apic_destination` is the high half of an x2APIC ICR
    /// write.
    fn avic_register_write(
        &mut self,
        offset: u64,
        value: u32,
        x2apic_destination: Option<u32>,
    ) -> ExecResult<()> {
        let x2apic = self.avic().is_some_and(|avic| avic.x2apic);
        match write_kind(offset, x2apic) {
            WriteKind::Fault => Err(self.avic_fault(offset, true)),
            WriteKind::Trap => {
                self.backing_write(offset, value);
                self.avic_trap(VMEXIT_AVIC_NOACCEL, Self::noaccel_info(offset, true), 0);
                Ok(())
            }
            WriteKind::Accelerated => {
                match offset {
                    APIC_TPR => self.avic_set_tpr(value as u8),
                    APIC_EOI => self.avic_eoi(),
                    APIC_ICR_HIGH => self.backing_write(APIC_ICR_HIGH, value),
                    APIC_SELF_IPI => self.avic_self_ipi(value as u8),
                    _ => {
                        self.backing_write(APIC_ICR_LOW, value);
                        let destination = match x2apic_destination {
                            Some(destination) => {
                                self.backing_write(APIC_ICR_HIGH, destination);
                                destination
                            }
                            None => self.backing_read(APIC_ICR_HIGH) >> XAPIC_DESTINATION_SHIFT,
                        };
                        self.avic_ipi(value as u64, destination, x2apic);
                    }
                }
                Ok(())
            }
        }
    }

    /// EOI: retires the highest in-service vector. A level-triggered one (its TMR bit set) needs
    /// the hypervisor's I/O APIC to see the EOI, so it traps with ISR untouched (KVM clears it).
    fn avic_eoi(&mut self) {
        let Some(vector) = self.highest_vector(APIC_ISR) else {
            return;
        };
        let (tmr, bit) = vector_bit(APIC_TMR, vector);
        if self.backing_read(tmr) & bit != 0 {
            self.avic_trap(
                VMEXIT_AVIC_NOACCEL,
                Self::noaccel_info(APIC_EOI, true),
                vector as u64,
            );
            return;
        }
        let page = self.avic().map_or(0, |avic| avic.backing_page);
        let (isr, bit) = vector_bit(page + APIC_ISR, vector);
        self.update_vector_word(isr, 0, bit);
        self.avic_update_ppr();
    }

    fn avic_self_ipi(&mut self, vector: u8) {
        let page = self.avic().map_or(0, |avic| avic.backing_page);
        let (irr, bit) = vector_bit(page + APIC_IRR, vector);
        self.update_vector_word(irr, bit, 0);
    }

    /// The physical id table entry for guest APIC id `id`, if valid and its backing page is RAM;
    /// else the failure cause.
    fn avic_physical_entry(&mut self, id: u64) -> Result<u64, u64> {
        let (table, max) = self.avic().map_or((0, 0), |avic| {
            (avic.physical_table, avic.physical_max_index)
        });
        if id > max {
            return Err(IPI_FAILURE_INVALID_TARGET);
        }
        let entry = self.phys_read(table + id * PHYSICAL_ENTRY_BYTES, 8);
        if entry & PHYSICAL_ENTRY_VALID == 0 {
            return Err(IPI_FAILURE_INVALID_TARGET);
        }
        if !self
            .ram
            .contains(entry & PHYSICAL_ENTRY_BACKING_PAGE_MASK, PAGE_SIZE)
        {
            return Err(IPI_FAILURE_INVALID_BACKING_PAGE);
        }
        Ok(entry)
    }

    /// The (physical table index, entry) of every target of a logical-mode xAPIC IPI to `mda`.
    fn avic_logical_targets(&mut self, mda: u32) -> Result<Vec<(u64, u64)>, u64> {
        let table = self.avic().map_or(0, |avic| avic.logical_table);
        let flat = self.backing_read(APIC_DFR) >> DFR_MODEL_SHIFT == DFR_MODEL_FLAT;
        let indices: Vec<u64> = if flat {
            (0..LOGICAL_FLAT_IDS)
                .filter(|bit| mda >> bit & 1 != 0)
                .map(|bit| bit as u64)
                .collect()
        } else {
            let cluster = mda >> LOGICAL_CLUSTER_IDS;
            if cluster == XAPIC_CLUSTER_BROADCAST {
                return Err(IPI_FAILURE_INVALID_TARGET);
            }
            (0..LOGICAL_CLUSTER_IDS)
                .filter(|bit| mda >> bit & 1 != 0)
                .map(|bit| (cluster * LOGICAL_CLUSTER_IDS + bit) as u64)
                .collect()
        };
        let mut targets = Vec::new();
        for index in indices {
            let entry = self.phys_read(table + index * LOGICAL_ENTRY_BYTES, 4);
            if entry & LOGICAL_ENTRY_VALID == 0 {
                return Err(IPI_FAILURE_INVALID_TARGET);
            }
            let id = entry & LOGICAL_ENTRY_GUEST_PHYSICAL_ID_MASK;
            targets.push((id, self.avic_physical_entry(id)?));
        }
        Ok(targets)
    }

    /// IPI virtualization for an ICR write (the ICR is already in the backing page). Fixed,
    /// edge-triggered IPIs to valid targets set the vector in each target's IRR; a target that is
    /// not running gets the hypervisor's attention through AVIC_INCOMPLETE_IPI, a trap, after
    /// all the IRR bits are set. Anything else is that exit with nothing delivered, for the
    /// hypervisor to emulate: other delivery modes, level-triggered, an invalid target anywhere
    /// in the set, and here also the all/all-but-self shorthands and x2APIC logical mode (a
    /// hypervisor emulates whatever it is told is invalid, and these keep the hardware model
    /// small).
    fn avic_ipi(&mut self, icr: u64, destination: u32, x2apic: bool) {
        let high = if x2apic {
            destination
        } else {
            destination << XAPIC_DESTINATION_SHIFT
        };
        let info1 = (high as u64) << 32 | icr;
        let fail = |cpu: &mut Cpu, cause: u64, index: u64| {
            cpu.avic_trap(
                VMEXIT_AVIC_INCOMPLETE_IPI,
                info1,
                cause << IPI_FAILURE_SHIFT | index,
            );
        };
        let vector = icr & ICR_VECTOR_MASK;
        if (icr >> ICR_DELIVERY_MODE_SHIFT) & ICR_DELIVERY_MODE_MASK != ICR_DELIVERY_FIXED
            || icr & ICR_LEVEL_TRIGGERED != 0
        {
            return fail(self, IPI_FAILURE_INVALID_INT_TYPE, 0);
        }
        if vector < FIRST_LEGAL_VECTOR {
            return fail(self, IPI_FAILURE_INVALID_VECTOR, 0);
        }
        let targets = match (icr >> ICR_SHORTHAND_SHIFT) & ICR_SHORTHAND_MASK {
            ICR_SHORTHAND_SELF => return self.avic_self_ipi(vector as u8),
            ICR_SHORTHAND_NONE if icr & ICR_LOGICAL == 0 => {
                if !x2apic && destination == XAPIC_BROADCAST {
                    Err(IPI_FAILURE_INVALID_TARGET)
                } else {
                    let id = destination as u64;
                    self.avic_physical_entry(id).map(|entry| vec![(id, entry)])
                }
            }
            ICR_SHORTHAND_NONE if !x2apic => self.avic_logical_targets(destination),
            _ => Err(IPI_FAILURE_INVALID_TARGET),
        };
        let targets = match targets {
            Ok(targets) => targets,
            Err(cause) => return fail(self, cause, 0),
        };
        for &(_, entry) in &targets {
            let page = entry & PHYSICAL_ENTRY_BACKING_PAGE_MASK;
            let (irr, bit) = vector_bit(page + APIC_IRR, vector as u8);
            self.update_vector_word(irr, bit, 0);
        }
        if let Some(&(index, _)) = targets
            .iter()
            .find(|(_, entry)| entry & PHYSICAL_ENTRY_IS_RUNNING == 0)
        {
            fail(self, IPI_FAILURE_TARGET_NOT_RUNNING, index);
        }
    }
}
