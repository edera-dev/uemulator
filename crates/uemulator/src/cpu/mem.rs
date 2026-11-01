use super::decode::{self, DecodeErr, Insn};
use super::exception::{Exception, VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT};
use super::flags::*;
use super::sys::{CR4_PSE, CR4_SMAP, CR4_SMEP};
use super::*;
use crate::dev::guestmem::GuestMem;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessKind {
    Read,
    Write,
    Exec,
}

/// Which page-table format the MMU walks, from CR0.PG, CR4.PAE and EFER.LMA.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum PagingMode {
    /// CR0.PG clear: linear equals physical (32 bits wide), everything permitted.
    Off,
    /// 32-bit paging: two levels of 4-byte entries, 4 MiB pages with CR4.PSE.
    Legacy,
    /// PAE paging: three levels of 8-byte entries, 2 MiB pages, NX with EFER.NXE.
    Pae,
    /// IA-32e paging (4 levels).
    Long,
}

const PTE_PRESENT: u64 = 1 << 0;
const PTE_WRITABLE: u64 = 1 << 1;
const PTE_USER: u64 = 1 << 2;
const PTE_ACCESSED: u64 = 1 << 5;
const PTE_DIRTY: u64 = 1 << 6;
/// Makes a PDPT entry a 1 GiB leaf or a PD entry a 2 MiB leaf.
const PTE_PAGE_SIZE: u64 = 1 << 7;
const PTE_NO_EXECUTE: u64 = 1 << 63;
/// In a 2 MiB or 1 GiB leaf the PAT bit moves here (bit 7 is page size); the address bits
/// between it and the leaf's alignment are reserved.
const PTE_LARGE_PAGE_PAT: u64 = 1 << 12;
/// Physical address bits 40-51, reserved because CPUID 0x80000008 advertises 40 address bits.
const PTE_RESERVED_MASK: u64 = 0x000f_ff00_0000_0000;
const PTE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const PTE_BYTES: u64 = 8;
const LEVEL_INDEX_BITS: u32 = 9;
const LEVEL_INDEX_MASK: u64 = (1 << LEVEL_INDEX_BITS) - 1;
/// The PML4 is level 3 and the PT is level 0 in long mode. PAE paging starts one level lower, at
/// the PDPT (level 2), whose four entries come from `Cpu::pdpte` instead of memory.
const TOP_LEVEL: u32 = 3;
const PAE_TOP_LEVEL: u32 = 2;
/// CR3 bits 31:5 of a PAE CR3: the 32-byte-aligned PDPT address.
const PAE_PDPT_ADDRESS_MASK: u64 = 0xffff_ffe0;
const PDPT_ENTRIES: usize = 4;
/// PDPTE bits that must be zero when the entry is present: 2:1 and 8:5 (SDM Vol. 3A 4.4.1) and
/// bit 63 and the bits above the physical address width.
const PDPTE_RESERVED_MASK: u64 = 0x1e6 | (1 << 63) | PTE_RESERVED_MASK;
/// 32-bit paging: 4-byte entries, 10 index bits per level, a 4 MiB leaf at the PDE.
const LEGACY_PTE_BYTES: u64 = 4;
const LEGACY_INDEX_BITS: u32 = 10;
const LEGACY_INDEX_MASK: u64 = (1 << LEGACY_INDEX_BITS) - 1;
const LEGACY_TABLE_MASK: u64 = 0xffff_f000;
const LEGACY_LARGE_PAGE_SHIFT: u32 = PAGE_SHIFT + LEGACY_INDEX_BITS;
/// 4 MiB PDE (PSE-36): address bits 31:22 sit in place, bits 39:32 in PDE bits 20:13, and bit 21
/// is reserved. PDE bit 12 is the PAT bit, not an address bit.
const LEGACY_LARGE_ADDRESS_LOW_MASK: u64 = 0xffc0_0000;
const LEGACY_LARGE_ADDRESS_HIGH_SHIFT: u32 = 13;
const LEGACY_LARGE_ADDRESS_HIGH_MASK: u64 = 0xff;
const LEGACY_LARGE_ADDRESS_HIGH_POSITION: u32 = 32;
const LEGACY_LARGE_RESERVED_BIT: u64 = 1 << 21;
/// Linear addresses are 32 bits wide outside long mode.
const LINEAR_32BIT_MASK: u64 = 0xffff_ffff;
/// Bit 47 and everything above must be equal (48-bit linear addresses, no LA57).
const CANONICAL_SIGN_BIT: u32 = 47;
const MAX_INSN_LEN: usize = 15;

/// Bits of the #PF error code.
const PF_PRESENT: u32 = 1 << 0;
const PF_WRITE: u32 = 1 << 1;
const PF_USER: u32 = 1 << 2;
const PF_RESERVED: u32 = 1 << 3;
const PF_INSTRUCTION_FETCH: u32 = 1 << 4;

/// `log_once` key bases for unmapped physical accesses; the page number (at most 40 bits) is
/// added to them. They sit in the top bits so no page number can reach another namespace, and
/// stay clear of the bases other callers use (opcodes in `exec.rs`, MSRs in `sys.rs`), which are
/// all below 2^34.
const LOG_KEY_UNMAPPED_READ: u64 = 1 << 62;
const LOG_KEY_UNMAPPED_WRITE: u64 = 1 << 61;

/// Narrows the TLB flags of a guest walk by what the nested leaf behind its page allows
/// (`svm::NestedRights`): without the store right the entry gets no write tag (clearing Dirty is
/// what withholds it whatever CR0.WP says), without the fetch right no exec tag, so those
/// accesses walk again and meet the nested fault or set the nested Dirty bit.
fn restrict_to_nested(flags: u8, nested: super::svm::NestedRights) -> u8 {
    let mut flags = flags;
    if !nested.store {
        flags &= !TLB_FLAG_DIRTY;
    }
    if !nested.fetch {
        flags |= TLB_FLAG_NO_EXECUTE;
    }
    flags
}

/// The physical page an entry's write tag lets stores reach: tag + addend is the page's host
/// address, minus `ram_base` its physical address.
pub(super) fn write_tag_page(entry: &TlbEntry, ram_base: u64) -> usize {
    (entry
        .tag_w
        .wrapping_add(entry.addend)
        .wrapping_sub(ram_base)
        >> PAGE_SHIFT) as usize
}

/// Slot of `linear` in `Cpu::tlb`: direct-mapped by page number, in the user half if `user`.
/// The JIT's inline probe computes the same index.
#[inline]
fn tlb_slot(linear: u64, user: bool) -> usize {
    ((user as usize) << TLB_INDEX_BITS)
        | ((linear >> PAGE_SHIFT) as usize & (TLB_ENTRIES_PER_MODE - 1))
}

/// See `Cpu::read_stack`.
fn stack_fault_for_non_canonical(exception: Exception) -> Exception {
    match exception {
        Exception::Fault {
            vector: VECTOR_GENERAL_PROTECTION,
            error_code: Some(0),
            ..
        } => Exception::stack_fault(0),
        other => other,
    }
}

impl Cpu {
    /// EFLAGS with the deferred arithmetic flags resolved.
    pub fn eflags_full(&self) -> u64 {
        let arith = flags::compute(
            self.lazy_op,
            self.lazy_size,
            self.lazy_dst,
            self.lazy_src,
            self.lazy_src2,
            self.eflags,
        );
        (self.eflags & !ARITH) | arith | RESERVED_ONE
    }

    /// Resolves the deferred flags into `eflags`. Needed before anything edits or snapshots
    /// non-arithmetic bits, since those writes would otherwise be mixed with stale lazy state.
    pub fn materialize_lazy_flags(&mut self) {
        self.eflags = self.eflags_full();
        self.lazy_op = LazyFlagsOp::Eflags;
    }

    /// Sets the `ARITH` flags eagerly from `arith`, dropping any deferred state.
    pub fn set_eflags_arith(&mut self, arith: u64) {
        self.eflags = (self.eflags & !ARITH) | (arith & ARITH) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
    }

    pub fn set_lazy_flags(&mut self, op: LazyFlagsOp, size: u8, dst: u64, src: u64, src2: u64) {
        self.lazy_op = op;
        self.lazy_size = size;
        self.lazy_dst = dst;
        self.lazy_src = src;
        self.lazy_src2 = src2;
    }

    pub fn cf(&self) -> bool {
        self.eflags_full() & CF != 0
    }

    /// Builds the #PF exception and sets CR2 right here, when the fault is raised, rather than
    /// at delivery.
    pub fn page_fault(
        &mut self,
        linear: u64,
        access: AccessKind,
        user: bool,
        present: bool,
        reserved_bit_violation: bool,
    ) -> Exception {
        self.cr2 = linear;
        let mut error_code = 0u32;
        if present {
            error_code |= PF_PRESENT;
        }
        if access == AccessKind::Write {
            error_code |= PF_WRITE;
        }
        if user {
            error_code |= PF_USER;
        }
        if reserved_bit_violation {
            error_code |= PF_RESERVED;
        }
        // The fetch bit is only reported when NX (which 32-bit paging does not have) or SMEP is
        // enabled.
        let nx_active = self.efer & EFER_NXE != 0 && self.paging_mode() >= PagingMode::Pae;
        if access == AccessKind::Exec && (nx_active || self.cr4 & CR4_SMEP != 0) {
            error_code |= PF_INSTRUCTION_FETCH;
        }
        Exception::fault(VECTOR_PAGE_FAULT, Some(error_code))
    }

    fn read_ram_u64(&self, pa: u64) -> Option<u64> {
        let start = pa as usize;
        let bytes = self.ram.slice(pa, PTE_BYTES as usize)?;
        if self.ordered_memory && start & (PTE_BYTES as usize - 1) == 0 {
            // Another vCPU may be changing the entry; read it whole, as the hardware walker does.
            return Some(unsafe { atomic::load(bytes.as_ptr(), PTE_BYTES as u8) });
        }
        Some(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Sets Accessed/Dirty: replaces the 8-byte entry at `pa` with `updated` if it still holds
    /// `seen`. Hardware does this with a locked update and walks again when the entry changed
    /// under it. A plain read-modify-write here would race Linux's atomic PTE updates (a
    /// copy-on-write `ptep_set_wrprotect`) and write the old writable bit back. `false`: the
    /// entry changed, walk again.
    pub(super) fn update_ram_u64(&mut self, pa: u64, seen: u64, updated: u64) -> bool {
        let start = pa as usize;
        if self.ordered_memory && start & (PTE_BYTES as usize - 1) == 0 {
            let host = self.ram[start..][..PTE_BYTES as usize].as_mut_ptr();
            return unsafe { atomic::update_entry_u64(host, seen, updated) }.is_ok();
        }
        self.ram[start..][..PTE_BYTES as usize].copy_from_slice(&updated.to_le_bytes());
        true
    }

    /// `update_ram_u64` for the 4-byte entries of 32-bit paging.
    fn update_ram_u32(&mut self, pa: u64, seen: u32, updated: u32) -> bool {
        let start = pa as usize;
        if self.ordered_memory && start & (LEGACY_PTE_BYTES as usize - 1) == 0 {
            let host = self.ram[start..][..LEGACY_PTE_BYTES as usize].as_mut_ptr();
            return unsafe { atomic::update_entry_u32(host, seen, updated) }.is_ok();
        }
        self.ram[start..][..LEGACY_PTE_BYTES as usize].copy_from_slice(&updated.to_le_bytes());
        true
    }

    pub(crate) fn paging_mode(&self) -> PagingMode {
        if self.efer & EFER_LMA != 0 {
            PagingMode::Long
        } else if self.cr0 & CR0_PG == 0 {
            PagingMode::Off
        } else if self.cr4 & CR4_PAE != 0 {
            PagingMode::Pae
        } else {
            PagingMode::Legacy
        }
    }

    fn read_ram_u32(&self, pa: u64) -> Option<u64> {
        let start = pa as usize;
        let bytes = self.ram.slice(pa, LEGACY_PTE_BYTES as usize)?;
        if self.ordered_memory && start & (LEGACY_PTE_BYTES as usize - 1) == 0 {
            return Some(unsafe { atomic::load(bytes.as_ptr(), LEGACY_PTE_BYTES as u8) });
        }
        Some(u32::from_le_bytes(bytes.try_into().unwrap()) as u64)
    }

    /// Whether a leaf with these accumulated permissions refuses the access. Supervisor writes to
    /// read-only pages only fault with CR0.WP. SMEP/SMAP: a supervisor-mode access to a page that
    /// is user-accessible at every level; SMAP is waived by EFLAGS.AC for explicit accesses, and
    /// implicit ones (descriptor tables) are indistinguishable here and get the same waiver.
    fn leaf_denied(
        &self,
        access: AccessKind,
        user: bool,
        writable: bool,
        user_allowed: bool,
        no_execute: bool,
    ) -> bool {
        let supervisor_on_user_page = !user && user_allowed;
        let smep_smap_denied = supervisor_on_user_page
            && if access == AccessKind::Exec {
                self.cr4 & CR4_SMEP != 0
            } else {
                self.cr4 & CR4_SMAP != 0 && self.eflags & AC == 0
            };
        let nxe = self.efer & EFER_NXE != 0;
        smep_smap_denied
            || (user && !user_allowed)
            || (access == AccessKind::Write && !writable && (user || self.cr0 & CR0_WP != 0))
            || (access == AccessKind::Exec && no_execute && nxe)
    }

    fn leaf_tlb_flags(&self, writable: bool, user_allowed: bool, no_execute: bool, pte: u64) -> u8 {
        let mut tlb_flags = 0;
        if writable {
            tlb_flags |= TLB_FLAG_WRITABLE;
        }
        if user_allowed {
            tlb_flags |= TLB_FLAG_USER;
        }
        if no_execute && self.efer & EFER_NXE != 0 {
            tlb_flags |= TLB_FLAG_NO_EXECUTE;
        }
        if pte & PTE_DIRTY != 0 {
            tlb_flags |= TLB_FLAG_DIRTY;
        }
        tlb_flags
    }

    /// Page walk in the current paging format. Returns the physical address of the 4 KiB page
    /// containing `linear` (inside a large leaf too) and the `TLB_FLAG_*` permissions, or the
    /// #PF. Page tables are read straight from RAM, never through the TLB; a table outside RAM
    /// counts as not present. Sets Accessed on every level touched, and Dirty on a write, but
    /// only after the permission check passed.
    ///
    /// Long mode and PAE share one loop (same 8-byte entries; PAE just starts at the PDPT
    /// latched in `pdpte`); 32-bit paging has its own because everything about its entries is
    /// narrower.
    fn walk(&mut self, linear: u64, access: AccessKind, user: bool) -> ExecResult<(u64, u8)> {
        if let Some(hit) = self.large_page_lookup(linear, access, user) {
            return Ok(hit);
        }
        match self.paging_mode() {
            // Linear is guest-physical; a nested-paging guest's addresses still go through nCR3.
            PagingMode::Off => {
                let (page_pa, nested) =
                    self.nested_translate_with_rights(linear & !PAGE_OFFSET_MASK, access, false)?;
                Ok((
                    page_pa,
                    restrict_to_nested(TLB_FLAG_WRITABLE | TLB_FLAG_USER | TLB_FLAG_DIRTY, nested),
                ))
            }
            PagingMode::Legacy => self.walk_legacy(linear, access, user),
            mode => self.walk_wide(linear, access, user, mode == PagingMode::Pae),
        }
    }

    /// The cached large leaf holding `linear`, as `walk` would return it, unless the access must
    /// walk anyway: it is not allowed (the walk raises the right #PF), or it is a store to a leaf
    /// that is not dirty yet (the walk sets Dirty).
    fn large_page_lookup(&self, linear: u64, access: AccessKind, user: bool) -> Option<(u64, u8)> {
        let cached = self.large_pages.iter().find(|leaf| {
            leaf.shift != 0 && linear >> leaf.shift == leaf.linear_base >> leaf.shift
        })?;
        if (access == AccessKind::Write && !cached.dirty)
            || self.leaf_denied(
                access,
                user,
                cached.writable,
                cached.user_allowed,
                cached.no_execute,
            )
        {
            return None;
        }
        let offset_mask = (1u64 << cached.shift) - 1;
        let page = cached.physical_base | (linear & offset_mask & !PAGE_OFFSET_MASK);
        let mut flags = 0;
        if cached.writable {
            flags |= TLB_FLAG_WRITABLE;
        }
        if cached.user_allowed {
            flags |= TLB_FLAG_USER;
        }
        if cached.no_execute {
            flags |= TLB_FLAG_NO_EXECUTE;
        }
        if cached.dirty {
            flags |= TLB_FLAG_DIRTY;
        }
        Some((page, flags))
    }

    /// Remembers a large leaf `walk_wide` just resolved. With nested paging the host-physical
    /// page can differ per 4 KiB inside a guest-physical leaf, and a nested fault must exit at
    /// the access that causes it, so nothing is cached then.
    #[allow(clippy::too_many_arguments)]
    fn large_page_insert(
        &mut self,
        linear: u64,
        shift: u32,
        physical_base: u64,
        writable: bool,
        user_allowed: bool,
        no_execute: bool,
        dirty: bool,
    ) {
        if self
            .svm
            .guest
            .as_ref()
            .is_some_and(|g| g.nested_cr3.is_some())
        {
            return;
        }
        let linear_base = linear & !((1u64 << shift) - 1);
        // The same leaf again (a store re-walking a clean one) takes its old slot.
        let slot = self
            .large_pages
            .iter()
            .position(|leaf| leaf.shift == shift && leaf.linear_base == linear_base)
            .unwrap_or_else(|| {
                let slot = self.large_page_next;
                self.large_page_next = (slot + 1) % LARGE_PAGE_CACHE_ENTRIES;
                slot
            });
        self.large_pages[slot] = LargePage {
            linear_base,
            physical_base,
            shift,
            writable,
            user_allowed,
            no_execute: no_execute && self.efer & EFER_NXE != 0,
            dirty,
        };
    }

    fn walk_wide(
        &mut self,
        linear: u64,
        access: AccessKind,
        user: bool,
        pae: bool,
    ) -> ExecResult<(u64, u8)> {
        let write = access == AccessKind::Write;
        let nxe = self.efer & EFER_NXE != 0;
        let mut table = self.cr3 & PTE_ADDRESS_MASK;
        // Effective permissions are the AND (user, write) or OR (no-execute) over all levels.
        let mut writable = true;
        let mut user_allowed = true;
        let mut no_execute = false;
        let mut level = if pae { PAE_TOP_LEVEL } else { TOP_LEVEL };
        loop {
            let index = (linear >> (PAGE_SHIFT + LEVEL_INDEX_BITS * level)) & LEVEL_INDEX_MASK;
            // A PDPTE has no permission or accessed bits and was validated when it was latched.
            if pae && level == PAE_TOP_LEVEL {
                let pdpte = self.pdpte[index as usize & (PDPT_ENTRIES - 1)];
                if pdpte & PTE_PRESENT == 0 {
                    return Err(self.page_fault(linear, access, user, false, false));
                }
                table = pdpte & PTE_ADDRESS_MASK;
                level -= 1;
                continue;
            }
            // With nested paging the guest's page tables are in guest-physical memory too.
            let guest_pte_address = table + index * PTE_BYTES;
            let pte_address = self.nested_translate(guest_pte_address, AccessKind::Read, true)?;
            let Some(mut pte) = self.read_ram_u64(pte_address) else {
                return Err(self.page_fault(linear, access, user, false, false));
            };
            if pte & PTE_PRESENT == 0 {
                return Err(self.page_fault(linear, access, user, false, false));
            }
            let leaf =
                level == 0 || (pte & PTE_PAGE_SIZE != 0 && (level == 1 || (!pae && level == 2)));
            // The page-size bit must be clear in a PML4 entry, and a large leaf must not have
            // address bits below its alignment (PAT aside).
            let below_leaf_alignment = (1u64 << (PAGE_SHIFT + LEVEL_INDEX_BITS * level)) - 1;
            let reserved = pte & PTE_RESERVED_MASK != 0
                || (!nxe && pte & PTE_NO_EXECUTE != 0)
                || (level == TOP_LEVEL && pte & PTE_PAGE_SIZE != 0)
                || (leaf
                    && level > 0
                    && pte & below_leaf_alignment & PTE_ADDRESS_MASK & !PTE_LARGE_PAGE_PAT != 0);
            if reserved {
                return Err(self.page_fault(linear, access, user, true, true));
            }
            writable &= pte & PTE_WRITABLE != 0;
            user_allowed &= pte & PTE_USER != 0;
            no_execute |= pte & PTE_NO_EXECUTE != 0;
            if leaf {
                if self.leaf_denied(access, user, writable, user_allowed, no_execute) {
                    return Err(self.page_fault(linear, access, user, true, false));
                }
                let mut updated = pte | PTE_ACCESSED;
                if write {
                    updated |= PTE_DIRTY;
                }
                if updated != pte {
                    // Updating Accessed/Dirty is a write to the page table.
                    let pte_address =
                        self.nested_translate(guest_pte_address, AccessKind::Write, true)?;
                    if !self.update_ram_u64(pte_address, pte, updated) {
                        return self.walk(linear, access, user);
                    }
                    pte = updated;
                }
                let leaf_shift = PAGE_SHIFT + LEVEL_INDEX_BITS * level;
                let leaf_offset_mask = (1u64 << leaf_shift) - 1;
                let leaf_base = pte & PTE_ADDRESS_MASK & !leaf_offset_mask;
                let guest_page = leaf_base | (linear & leaf_offset_mask & !PAGE_OFFSET_MASK);
                let (page_pa, nested) =
                    self.nested_translate_with_rights(guest_page, access, false)?;
                if level > 0 {
                    self.large_page_insert(
                        linear,
                        leaf_shift,
                        leaf_base,
                        writable,
                        user_allowed,
                        no_execute,
                        pte & PTE_DIRTY != 0,
                    );
                }
                return Ok((
                    page_pa,
                    restrict_to_nested(
                        self.leaf_tlb_flags(writable, user_allowed, no_execute, pte),
                        nested,
                    ),
                ));
            }
            if pte & PTE_ACCESSED == 0 {
                let pte_address =
                    self.nested_translate(guest_pte_address, AccessKind::Write, true)?;
                if !self.update_ram_u64(pte_address, pte, pte | PTE_ACCESSED) {
                    return self.walk(linear, access, user);
                }
                pte |= PTE_ACCESSED;
            }
            table = pte & PTE_ADDRESS_MASK;
            level -= 1;
        }
    }

    /// 32-bit paging (CR4.PAE clear). Entries are 4 bytes, there is no NX and the only reserved
    /// bit is bit 21 of a 4 MiB PDE; CR4.PSE decides whether PDE bit 7 means a 4 MiB page.
    fn walk_legacy(
        &mut self,
        linear: u64,
        access: AccessKind,
        user: bool,
    ) -> ExecResult<(u64, u8)> {
        let write = access == AccessKind::Write;
        let mut table = self.cr3 & LEGACY_TABLE_MASK;
        let mut writable = true;
        let mut user_allowed = true;
        let mut level = 1;
        loop {
            let index = (linear >> (PAGE_SHIFT + LEGACY_INDEX_BITS * level)) & LEGACY_INDEX_MASK;
            let guest_pte_address = table + index * LEGACY_PTE_BYTES;
            let pte_address = self.nested_translate(guest_pte_address, AccessKind::Read, true)?;
            let Some(mut pte) = self.read_ram_u32(pte_address) else {
                return Err(self.page_fault(linear, access, user, false, false));
            };
            if pte & PTE_PRESENT == 0 {
                return Err(self.page_fault(linear, access, user, false, false));
            }
            let large = level == 1 && self.cr4 & CR4_PSE != 0 && pte & PTE_PAGE_SIZE != 0;
            if large && pte & LEGACY_LARGE_RESERVED_BIT != 0 {
                return Err(self.page_fault(linear, access, user, true, true));
            }
            writable &= pte & PTE_WRITABLE != 0;
            user_allowed &= pte & PTE_USER != 0;
            if large || level == 0 {
                if self.leaf_denied(access, user, writable, user_allowed, false) {
                    return Err(self.page_fault(linear, access, user, true, false));
                }
                let mut updated = pte | PTE_ACCESSED;
                if write {
                    updated |= PTE_DIRTY;
                }
                if updated != pte {
                    let pte_address =
                        self.nested_translate(guest_pte_address, AccessKind::Write, true)?;
                    if !self.update_ram_u32(pte_address, pte as u32, updated as u32) {
                        return self.walk_legacy(linear, access, user);
                    }
                    pte = updated;
                }
                let page_offset_in_leaf = if large {
                    linear & ((1u64 << LEGACY_LARGE_PAGE_SHIFT) - 1) & !PAGE_OFFSET_MASK
                } else {
                    0
                };
                let leaf_base = if large {
                    let high =
                        (pte >> LEGACY_LARGE_ADDRESS_HIGH_SHIFT) & LEGACY_LARGE_ADDRESS_HIGH_MASK;
                    (pte & LEGACY_LARGE_ADDRESS_LOW_MASK)
                        | (high << LEGACY_LARGE_ADDRESS_HIGH_POSITION)
                } else {
                    pte & LEGACY_TABLE_MASK
                };
                let (page_pa, nested) = self.nested_translate_with_rights(
                    leaf_base | page_offset_in_leaf,
                    access,
                    false,
                )?;
                return Ok((
                    page_pa,
                    restrict_to_nested(
                        self.leaf_tlb_flags(writable, user_allowed, false, pte),
                        nested,
                    ),
                ));
            }
            if pte & PTE_ACCESSED == 0 {
                let pte_address =
                    self.nested_translate(guest_pte_address, AccessKind::Write, true)?;
                if !self.update_ram_u32(pte_address, pte as u32, (pte | PTE_ACCESSED) as u32) {
                    return self.walk_legacy(linear, access, user);
                }
                pte |= PTE_ACCESSED;
            }
            table = pte & LEGACY_TABLE_MASK;
            level -= 1;
        }
    }

    /// Reads and validates the four PDPT entries PAE paging would use for `cr3`. A present entry
    /// with a reserved bit set is #GP(0), and happens at the CR3/CR0/CR4 write rather than at the
    /// first access. A PDPT outside RAM reads as not present, like page tables do.
    pub(crate) fn read_pdptes(&mut self, cr3: u64) -> ExecResult<[u64; PDPT_ENTRIES]> {
        let base = cr3 & PAE_PDPT_ADDRESS_MASK;
        let mut entries = [0u64; PDPT_ENTRIES];
        for (index, entry) in entries.iter_mut().enumerate() {
            // With nested paging the PDPT is at a guest-physical address.
            let address =
                self.nested_translate(base + index as u64 * PTE_BYTES, AccessKind::Read, true)?;
            let Some(value) = self.read_ram_u64(address) else {
                continue;
            };
            if value & PTE_PRESENT != 0 && value & PDPTE_RESERVED_MASK != 0 {
                return Err(Exception::general_protection(0));
            }
            *entry = value;
        }
        Ok(entries)
    }

    /// Linear to physical address for one access, refilling the TLB on a miss.
    ///
    /// `supervisor` marks implicit accesses by the CPU itself (descriptor tables, TSS,
    /// interrupt delivery): they ignore CPL and use the supervisor TLB half even at CPL 3.
    /// Only the 48-bit canonical check is done, and it raises #GP(0) even for stack accesses
    /// where hardware raises #SS.
    pub fn translate(
        &mut self,
        linear: u64,
        access: AccessKind,
        supervisor: bool,
    ) -> ExecResult<u64> {
        // Outside long mode linear addresses are 32 bits wide: the upper half is dropped rather
        // than checked, so callers may pass an unmasked 64-bit register sum.
        let linear = if self.efer & EFER_LMA == 0 {
            linear & LINEAR_32BIT_MASK
        } else {
            let sign_extension = (linear as i64) >> CANONICAL_SIGN_BIT;
            if sign_extension != 0 && sign_extension != -1 {
                return Err(Exception::general_protection(0));
            }
            linear
        };
        let user = self.cpl == USER_CPL && !supervisor;
        let page = linear & !PAGE_OFFSET_MASK;
        let slot = tlb_slot(linear, user);
        let entry = self.tlb[slot];
        let tag = match access {
            AccessKind::Read => entry.tag_r,
            AccessKind::Write => entry.tag_w,
            AccessKind::Exec => entry.tag_x,
        };
        if tag == page {
            return Ok(linear
                .wrapping_add(entry.addend)
                .wrapping_sub(self.ram_base));
        }
        let (page_pa, walk_flags) = self.walk(linear, access, user)?;
        // A supervisor access to a user page is never cached while SMEP or SMAP is on: whether it
        // is allowed depends on EFLAGS.AC (SMAP), which changes without any TLB flush (stac,
        // clac, popf), and a cached entry would let a later access skip the check.
        let uncacheable = !user
            && walk_flags & TLB_FLAG_USER != 0
            && self.cr4 & (CR4_SMEP | CR4_SMAP) != 0
            && self.paging_mode() != PagingMode::Off;
        // Only RAM-backed pages are cached; MMIO has no host pointer to put in `addend`.
        if self.ram.contains(page_pa, 1) && !uncacheable {
            let write_protect = self.cr0 & CR0_WP != 0;
            // Keep the other tags if the slot already maps this page, so a read refill does not
            // drop the page's exec tag and vice versa.
            let mut refilled = if entry.tag_r == page || entry.tag_w == page || entry.tag_x == page
            {
                entry
            } else {
                TLB_EMPTY
            };
            refilled.addend = (self.ram_base + page_pa).wrapping_sub(page);
            if !user || walk_flags & TLB_FLAG_USER != 0 {
                refilled.tag_r = page;
                if walk_flags & TLB_FLAG_NO_EXECUTE == 0 {
                    refilled.tag_x = page;
                }
            }
            // The write tag is withheld unless the store can skip both the walk (that sets
            // Dirty) and the code-page check, so the first store to such a page goes slow.
            let write_permitted = walk_flags & TLB_FLAG_DIRTY != 0
                && if user {
                    walk_flags & (TLB_FLAG_USER | TLB_FLAG_WRITABLE)
                        == (TLB_FLAG_USER | TLB_FLAG_WRITABLE)
                } else {
                    walk_flags & TLB_FLAG_WRITABLE != 0 || !write_protect
                };
            if write_permitted && self.page_has_code[(page_pa >> PAGE_SHIFT) as usize] == 0 {
                refilled.tag_w = page;
            }
            self.tlb[slot] = refilled;
        }
        Ok(page_pa | (linear & PAGE_OFFSET_MASK))
    }

    pub fn flush_tlb(&mut self) {
        for entry in self.tlb.iter_mut() {
            *entry = TLB_EMPTY;
        }
        self.large_pages = [LARGE_PAGE_EMPTY; LARGE_PAGE_CACHE_ENTRIES];
    }

    /// Empties the page's slot in both halves; whatever other page shares the slot goes too.
    pub fn invlpg(&mut self, linear: u64) {
        for leaf in self.large_pages.iter_mut() {
            if leaf.shift != 0 && linear >> leaf.shift == leaf.linear_base >> leaf.shift {
                *leaf = LARGE_PAGE_EMPTY;
            }
        }
        let slot = tlb_slot(linear, false);
        self.tlb[slot] = TLB_EMPTY;
        self.tlb[TLB_ENTRIES_PER_MODE + slot] = TLB_EMPTY;
    }

    /// Reads `size` bytes (at most 8, all inside one page) at a physical address: RAM, else
    /// MMIO, else all ones for unmapped space.
    pub(super) fn phys_read(&mut self, pa: u64, size: u8) -> u64 {
        let len = size as usize;
        if let Some(src) = self.ram.slice(pa, len) {
            if self.ordered_memory {
                return unsafe { atomic::load(src.as_ptr(), size) };
            }
            let mut bytes = [0u8; 8];
            bytes[..len].copy_from_slice(src);
            u64::from_le_bytes(bytes)
        } else if let Some(value) = {
            // Device registers with timers (the APIC) need the time of this very access.
            let now = self.now_ns();
            self.dev.lock().set_time(now);
            self.dev.lock().mmio_read(pa, size)
        } {
            value
        } else {
            self.log_once(LOG_KEY_UNMAPPED_READ + (pa >> PAGE_SHIFT), || {
                format!("read of unmapped physical address {pa:#x}")
            });
            u64::MAX >> (64 - 8 * size as u32)
        }
    }

    /// Counterpart of `phys_read`; writes to unmapped space are dropped.
    pub(super) fn phys_write(&mut self, pa: u64, size: u8, val: u64) {
        let start = pa as usize;
        let len = size as usize;
        if self.ram.contains(pa, len as u64) {
            if self.ordered_memory {
                // Store first, then look at the flag: a vCPU that flags the page and decodes it
                // after our check must see these bytes, and one that flagged it before is seen
                // here (`Cpu::mark_code` fences the other half).
                let host = self.ram[start..][..len].as_mut_ptr();
                unsafe { atomic::store(host, size, val) };
                fence(Ordering::SeqCst);
                if self.page_has_code.get(start >> PAGE_SHIFT) != 0 {
                    self.code_written(start >> PAGE_SHIFT);
                }
                return;
            }
            if self.page_has_code[start >> PAGE_SHIFT] != 0 {
                self.code_written(start >> PAGE_SHIFT);
            }
            self.ram[start..start + len].copy_from_slice(&val.to_le_bytes()[..len]);
        } else {
            let now = self.now_ns();
            self.dev.lock().set_time(now);
            let written_code_pages = {
                let mut mem = GuestMem::new(&mut self.ram, self.page_has_code.as_bytes());
                if self.dev.lock().mmio_write(pa, size, val, &mut mem) {
                    mem.code_pages_written
                } else {
                    self.log_once(LOG_KEY_UNMAPPED_WRITE + (pa >> PAGE_SHIFT), || {
                        format!("write to unmapped physical address {pa:#x}")
                    });
                    Vec::new()
                }
            };
            for page in written_code_pages {
                self.code_written(page);
            }
        }
    }

    /// A store or device DMA hit a page that translated code came from.
    ///
    /// With the JIT only that page's blocks are dropped; `block_exit_requested` makes the
    /// block that is currently running stop after this store instead of executing bytes it
    /// may just have overwritten. Without it the whole decode cache is flushed.
    pub fn code_written(&mut self, ppn: usize) {
        if let Some(jit) = self.jit.take() {
            // Clears the page's flag too, which has to happen under the cache's lock.
            jit.invalidate_page(ppn as u64, &self.page_has_code);
            self.jit = Some(jit);
            self.decode_cache_generation = self.decode_cache_generation.wrapping_add(1);
            self.block_exit_requested = true;
        } else {
            self.flush_decode();
            self.page_has_code.set(ppn, 0);
        }
        if let Some(runtime) = &self.threads {
            // The other vCPUs' translations of this page are stale too; they drop them at their
            // next block boundary.
            runtime
                .links
                .publish_invalidation(Some(self.cpu_index as usize), ppn);
        }
    }

    pub fn flush_decode(&mut self) {
        self.flushes += 1;
        self.decode_cache_generation = self.decode_cache_generation.wrapping_add(1);
        // Translated blocks survive this (they are keyed by physical address and the JIT only
        // runs 64-bit code), and the page flags are what makes a store to their pages call
        // `code_written`. Clearing them here left stale blocks after every SVM #VMEXIT or VMRUN
        // (both flush): Linux's alternatives patching ran old `jmp` targets under Xen.
        // With threads the flags are shared and other vCPUs' caches depend on them.
        if self.jit.is_some() || self.threads.is_some() {
            return;
        }
        for page in self.pages_with_code.drain(..) {
            self.page_has_code.set(page as usize, 0);
        }
    }

    /// Flags the physical page as holding translated code and revokes write tags pointing at
    /// it, so the next store to it goes through `phys_write` and `code_written`. Callers read
    /// the guest bytes only after this returns (with threads, a store by another vCPU is then
    /// either already in those bytes or sees the flag).
    pub fn mark_code(&mut self, pa: u64) {
        let page = (pa >> PAGE_SHIFT) as usize;
        if page >= self.page_has_code.len() || self.page_has_code.get(page) != 0 {
            return;
        }
        if self.threads.is_some() {
            self.with_exclusive(|cpu| cpu.flag_code_page_for_all(page));
            return;
        }
        self.page_has_code.set(page, 1);
        // `code_written` clears a page's flag without un-listing it, so a page that is
        // rewritten and retranslated over and over would grow the list without bound.
        if self.pages_with_code.len() >= self.page_has_code.len() {
            let flags = &self.page_has_code;
            self.pages_with_code
                .retain(|&listed| flags.get(listed as usize) != 0);
            self.pages_with_code.sort_unstable();
            self.pages_with_code.dedup();
        }
        self.pages_with_code.push(page as u32);
        // The other vCPUs' TLBs are revoked when they next run (`Smp::switch_to`); this one
        // is done below.
        self.code_mark_epoch += 1;
        self.code_mark_seen = self.code_mark_epoch;
        self.revoke_write_tags_for(page);
    }

    /// Drops this vCPU's write tags that point into physical page `page`, in the running TLB and
    /// in the one SVM parked (`svm::ParkedTlb`), which comes back without a refill.
    pub(super) fn revoke_write_tags_for(&mut self, page: usize) {
        let ram_base = self.ram_base;
        let parked = self
            .svm
            .parked_tlb
            .as_deref_mut()
            .map(|parked| &mut parked.entries);
        for entries in core::iter::once(&mut self.tlb).chain(parked) {
            for entry in entries.iter_mut() {
                if entry.tag_w != TLB_TAG_INVALID && write_tag_page(entry, ram_base) == page {
                    entry.tag_w = TLB_TAG_INVALID;
                }
            }
        }
    }

    /// Guest load of `size` bytes (1, 2, 4 or 8). The fast path needs the whole access inside
    /// the page the tag covers: the tag only describes one page, and the next page may map
    /// anywhere, so a straddling access translates and reads each page separately.
    fn mem_read(&mut self, linear: u64, size: u8, supervisor: bool) -> ExecResult<u64> {
        let user = self.cpl == USER_CPL && !supervisor;
        let entry = &self.tlb[tlb_slot(linear, user)];
        if entry.tag_r == linear & !PAGE_OFFSET_MASK
            && (linear & PAGE_OFFSET_MASK) + size as u64 <= PAGE_SIZE
        {
            let host = linear.wrapping_add(entry.addend);
            if self.ordered_memory {
                return Ok(unsafe { atomic::load(host as *const u8, size) });
            }
            unsafe {
                return Ok(match size {
                    1 => *(host as *const u8) as u64,
                    2 => (host as *const u16).read_unaligned() as u64,
                    4 => (host as *const u32).read_unaligned() as u64,
                    _ => (host as *const u64).read_unaligned(),
                });
            }
        }
        let offset = (linear & PAGE_OFFSET_MASK) as usize;
        if offset + size as usize <= PAGE_SIZE as usize {
            let pa = self.translate(linear, AccessKind::Read, supervisor)?;
            return Ok(self.phys_read(pa, size));
        }
        let first_len = PAGE_SIZE as usize - offset;
        let first_pa = self.translate(linear, AccessKind::Read, supervisor)?;
        let second_pa = self.translate(
            (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
            AccessKind::Read,
            supervisor,
        )?;
        let low = self.phys_read(first_pa, first_len as u8);
        let high = self.phys_read(second_pa, size - first_len as u8);
        Ok(low | (high << (8 * first_len)))
    }

    /// Store counterpart of `mem_read`. A straddling store translates both pages before
    /// writing either, so a fault on the second page leaves memory untouched.
    fn mem_write(&mut self, linear: u64, size: u8, val: u64, supervisor: bool) -> ExecResult<()> {
        let user = self.cpl == USER_CPL && !supervisor;
        let entry = &self.tlb[tlb_slot(linear, user)];
        if entry.tag_w == linear & !PAGE_OFFSET_MASK
            && (linear & PAGE_OFFSET_MASK) + size as u64 <= PAGE_SIZE
        {
            let host = linear.wrapping_add(entry.addend);
            if self.ordered_memory {
                unsafe { atomic::store(host as *mut u8, size, val) };
                return Ok(());
            }
            unsafe {
                match size {
                    1 => *(host as *mut u8) = val as u8,
                    2 => (host as *mut u16).write_unaligned(val as u16),
                    4 => (host as *mut u32).write_unaligned(val as u32),
                    _ => (host as *mut u64).write_unaligned(val),
                }
            }
            return Ok(());
        }
        let offset = (linear & PAGE_OFFSET_MASK) as usize;
        if offset + size as usize <= PAGE_SIZE as usize {
            let pa = self.translate(linear, AccessKind::Write, supervisor)?;
            self.phys_write(pa, size, val);
            return Ok(());
        }
        let first_len = PAGE_SIZE as usize - offset;
        let first_pa = self.translate(linear, AccessKind::Write, supervisor)?;
        let second_pa = self.translate(
            (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
            AccessKind::Write,
            supervisor,
        )?;
        self.phys_write(first_pa, first_len as u8, val);
        self.phys_write(second_pa, size - first_len as u8, val >> (8 * first_len));
        Ok(())
    }

    pub fn read(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.mem_read(linear, size, false)
    }

    pub fn write(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.mem_write(linear, size, val, false)
    }

    /// Implicit stack accesses (push, pop, call, ret, ...). The only #GP(0) a data access can
    /// raise is for a non-canonical address, which through SS is reported as #SS(0) instead.
    pub fn read_stack(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.read(linear, size)
            .map_err(stack_fault_for_non_canonical)
    }

    pub fn write_stack(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.write(linear, size, val)
            .map_err(stack_fault_for_non_canonical)
    }

    /// Access by the CPU itself (see `translate`), independent of CPL.
    pub fn read_sys(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.mem_read(linear, size, true)
    }

    pub fn write_sys(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.mem_write(linear, size, val, true)
    }

    /// Implicit supervisor store to the stack, as interrupt delivery does: non-canonical is
    /// #SS(0), like `write_stack`.
    pub fn write_sys_stack(&mut self, linear: u64, size: u8, val: u64) -> ExecResult<()> {
        self.write_sys(linear, size, val)
            .map_err(stack_fault_for_non_canonical)
    }

    pub fn read128(&mut self, linear: u64) -> ExecResult<u128> {
        let low = self.read(linear, 8)?;
        let high = self.read(linear.wrapping_add(8), 8)?;
        Ok(low as u128 | ((high as u128) << 64))
    }

    pub fn write128(&mut self, linear: u64, v: u128) -> ExecResult<()> {
        // Probe both pages first: the stores below must not start if the second one faults.
        // The probe uses the page base, like a straddling `write`, so CR2 agrees.
        self.translate(linear, AccessKind::Write, false)?;
        let last_page = linear.wrapping_add(15) & !PAGE_OFFSET_MASK;
        if last_page != linear & !PAGE_OFFSET_MASK {
            self.translate(last_page, AccessKind::Write, false)?;
        }
        self.write(linear, 8, v as u64)?;
        self.write(linear.wrapping_add(8), 8, (v >> 64) as u64)
    }

    /// Decodes the instruction at `rip`, through the decode cache. Marks the page(s) it was
    /// read from as holding code so later stores invalidate the cache.
    pub fn fetch_decode(&mut self) -> ExecResult<Insn> {
        let rip = self.fetch_linear()?;
        let mode = self.code_mode();
        let pa = self.translate(rip, AccessKind::Exec, false)?;
        let slot = (pa ^ (pa >> 16)) as usize & (DECODE_CACHE_ENTRIES - 1);
        let cached = &self.decode_cache[slot];
        if cached.gpa == pa && cached.generation == self.decode_cache_generation {
            let insn = cached.insn;
            if insn.mode == mode {
                return self.fetched_within_limit(insn);
            }
        }
        let bytes_left_in_page = PAGE_SIZE as usize - (pa & PAGE_OFFSET_MASK) as usize;
        let len = bytes_left_in_page.min(MAX_INSN_LEN);
        let start = pa as usize;
        let mut buf = [0u8; MAX_INSN_LEN];
        // Executing from MMIO or unmapped space is reported as a not-present fetch fault.
        if !self.ram.contains(pa, len as u64) {
            return Err(self.page_fault(rip, AccessKind::Exec, self.cpl == USER_CPL, false, false));
        }
        // Flag the page before reading it: see `mark_code`.
        self.mark_code(pa);
        buf[..len].copy_from_slice(&self.ram[start..start + len]);
        match decode::decode_in_mode(&buf[..len], mode) {
            Ok(insn) => {
                self.decode_cache[slot] = DecodeCacheEntry {
                    gpa: pa,
                    generation: self.decode_cache_generation,
                    insn,
                };
                self.fetched_within_limit(insn)
            }
            Err(DecodeErr::Invalid) => Err(Exception::invalid_opcode()),
            // The 15-byte window ended mid-instruction: it is longer than the architectural
            // limit, which is #GP(0) (an undecodable opcode stays #UD).
            Err(DecodeErr::Truncated) if len == MAX_INSN_LEN => {
                Err(Exception::general_protection(0))
            }
            // The instruction runs into the next page. It is not cached, because its validity
            // depends on two pages; both get marked as code instead.
            Err(DecodeErr::Truncated) => {
                let next_page_pa = self.translate(
                    (rip & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE),
                    AccessKind::Exec,
                    false,
                )?;
                let missing = MAX_INSN_LEN - len;
                let next_start = next_page_pa as usize;
                if !self.ram.contains(next_page_pa, missing as u64) {
                    return Err(Exception::invalid_opcode());
                }
                self.mark_code(next_page_pa);
                buf[len..MAX_INSN_LEN].copy_from_slice(&self.ram[next_start..next_start + missing]);
                match decode::decode_in_mode(&buf[..MAX_INSN_LEN], mode) {
                    Ok(insn) => self.fetched_within_limit(insn),
                    Err(DecodeErr::Truncated) => Err(Exception::general_protection(0)),
                    Err(DecodeErr::Invalid) => Err(Exception::invalid_opcode()),
                }
            }
        }
    }

    /// Outside 64-bit mode the last byte of the instruction must be inside the CS limit as well
    /// as the first one (`fetch_linear`), or it is #GP(0).
    fn fetched_within_limit(&self, insn: Insn) -> ExecResult<Insn> {
        if !self.is_long64() {
            let last = (self.rip & 0xffff_ffff) + insn.len as u64 - 1;
            if last > self.seg[SEG_CS].limit as u64 {
                return Err(Exception::general_protection(0));
            }
        }
        Ok(insn)
    }
}

#[cfg(test)]
#[path = "paging_tests.rs"]
mod paging_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exception::{VECTOR_GENERAL_PROTECTION, VECTOR_PAGE_FAULT, VECTOR_STACK_FAULT};
    use crate::cpu::test_support::*;
    use std::sync::Mutex;

    use crate::dev::DevConfig;
    use crate::testkit::{CODE_BASE, HARNESS_RAM_BYTES, HLT, Harness, SCRATCH};

    /// Canonical linear addresses used for the mappings below; all 4 KiB aligned.
    const VA: u64 = 0x0000_7f12_3456_0000;
    const VA_NEXT_PAGE: u64 = VA + PAGE;
    /// Physical pages inside RAM that are not adjacent, so a straddling access must be stitched
    /// from two separate translations.
    const PA: u64 = SCRATCH;
    const PA_NEXT: u64 = SCRATCH + 5 * PAGE;
    /// First physical address past the harness RAM: unmapped unless a device claims it.
    const FIRST_UNMAPPED_PA: u64 = HARNESS_RAM_BYTES as u64;
    const SUPERVISOR_ONLY: u64 = PTE_P | PTE_W;
    const READ_ONLY_USER: u64 = PTE_P | PTE_U;
    const NOT_CANONICAL: u64 = 0x0000_8000_0000_0000;
    const LAST_LOW_CANONICAL: u64 = 0x0000_7fff_ffff_ffff;
    const FIRST_HIGH_CANONICAL: u64 = 0xffff_8000_0000_0000;
    const LAST_NON_CANONICAL: u64 = 0xffff_7fff_ffff_ffff;

    fn harness() -> Harness {
        Harness::new(&[HLT])
    }

    fn paged() -> (Harness, PageTables) {
        let mut h = harness();
        let tables = PageTables::new(&mut h.cpu);
        (h, tables)
    }

    fn set_nxe(cpu: &mut Cpu, on: bool) {
        cpu.efer = if on {
            cpu.efer | EFER_NXE
        } else {
            cpu.efer & !EFER_NXE
        };
    }

    fn set_wp(cpu: &mut Cpu, on: bool) {
        cpu.cr0 = if on {
            cpu.cr0 | CR0_WP
        } else {
            cpu.cr0 & !CR0_WP
        };
    }

    fn slot_of(cpu: &Cpu, linear: u64, user: bool) -> TlbEntry {
        cpu.tlb[tlb_slot(linear, user)]
    }

    fn logged(log: &Mutex<Vec<String>>) -> Vec<String> {
        log.lock().unwrap().clone()
    }

    fn host_address(cpu: &Cpu, physical: u64) -> u64 {
        cpu.ram_base + physical
    }

    mod page_fault_construction {
        use super::*;

        #[test]
        fn sets_cr2_to_the_faulting_address_and_vector_14() {
            let mut h = harness();
            let exception = h
                .cpu
                .page_fault(0xdead_b000, AccessKind::Read, false, false, false);
            assert_eq!(fault_of::<()>(Err(exception)), (VECTOR_PAGE_FAULT, Some(0)));
            assert_eq!(h.cpu.cr2, 0xdead_b000);
        }

        #[test]
        fn every_error_code_bit_combination_matches_the_architecture() {
            for nxe in [false, true] {
                for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                    for user in [false, true] {
                        for present in [false, true] {
                            for reserved in [false, true] {
                                let mut h = harness();
                                set_nxe(&mut h.cpu, nxe);
                                let exception =
                                    h.cpu.page_fault(0x1234, access, user, present, reserved);
                                let mut expected = 0;
                                if present {
                                    expected |= PF_P;
                                }
                                if access == AccessKind::Write {
                                    expected |= PF_W;
                                }
                                if user {
                                    expected |= PF_U;
                                }
                                if reserved {
                                    expected |= PF_RSVD;
                                }
                                if access == AccessKind::Exec && nxe {
                                    expected |= PF_ID;
                                }
                                assert_eq!(
                                    fault_of::<()>(Err(exception)),
                                    (VECTOR_PAGE_FAULT, Some(expected)),
                                    "nxe={nxe} access={:?} user={user} present={present} reserved={reserved}",
                                    access as u8,
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn instruction_fetch_bit_is_absent_without_nxe() {
            let mut h = harness();
            let exception = h
                .cpu
                .page_fault(0x1000, AccessKind::Exec, false, true, false);
            assert_eq!(fault_of::<()>(Err(exception)).1, Some(PF_P));
        }

        #[test]
        fn a_later_fault_overwrites_cr2() {
            let mut h = harness();
            let _ = h
                .cpu
                .page_fault(0x1000, AccessKind::Read, false, false, false);
            let _ = h
                .cpu
                .page_fault(0x2000, AccessKind::Read, false, false, false);
            assert_eq!(h.cpu.cr2, 0x2000);
        }
    }

    mod page_walk {
        use super::*;

        #[test]
        fn four_kib_mapping_translates_with_the_page_offset() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            for offset in [0, 1, 0x123, 0xabc, 0xfff] {
                assert_eq!(
                    h.cpu
                        .translate(VA + offset, AccessKind::Read, false)
                        .unwrap(),
                    PA + offset
                );
            }
        }

        #[test]
        fn read_sets_accessed_on_every_level_and_never_dirty() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            for level in 0..=3 {
                let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
                assert_ne!(entry & PTE_A, 0, "accessed at level {level}");
                assert_eq!(entry & PTE_D, 0, "dirty at level {level}");
            }
        }

        #[test]
        fn exec_sets_accessed_and_not_dirty() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Exec, false).unwrap();
            assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), PTE_A);
        }

        #[test]
        fn write_sets_dirty_on_the_leaf_only() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            for level in 1..=3 {
                let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
                assert_eq!(entry & PTE_D, 0, "upper level {level} must not get dirty");
                assert_ne!(entry & PTE_A, 0);
            }
            let leaf = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, 0));
            assert_eq!(leaf & (PTE_A | PTE_D), PTE_A | PTE_D);
        }

        #[test]
        fn walk_preserves_the_other_entry_bits() {
            let (mut h, mut tables) = paged();
            let flags = PTE_P | PTE_W | PTE_U | (1 << 3) | (1 << 4) | (1 << 8) | (1 << 9) | PTE_NX;
            set_nxe(&mut h.cpu, true);
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, flags);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            assert_eq!(read_pte(&h.cpu, leaf), PA | flags | PTE_A | PTE_D);
        }

        #[test]
        fn already_set_accessed_and_dirty_bits_are_left_alone() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(
                &mut h.cpu,
                VA,
                PA,
                PTE_PRESENT_WRITABLE_USER | PTE_A | PTE_D,
            );
            let before = read_pte(&h.cpu, leaf);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            assert_eq!(read_pte(&h.cpu, leaf), before);
        }

        #[test]
        fn two_mib_leaf_translates_every_offset_inside_it() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
            for offset in [0, 0x1000, 0x1_2345, 0x10_0fff, TWO_MIB - 1] {
                assert_eq!(
                    h.cpu
                        .translate(va + offset, AccessKind::Read, false)
                        .unwrap(),
                    0x40_0000 + offset,
                    "offset {offset:#x}"
                );
            }
        }

        #[test]
        fn two_mib_leaf_gets_accessed_on_three_levels_and_dirty_on_the_pde() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            let pde = tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
            h.cpu
                .translate(va + 0x5000, AccessKind::Read, false)
                .unwrap();
            assert_eq!(read_pte(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A);
            h.cpu
                .translate(va + 0x6000, AccessKind::Write, false)
                .unwrap();
            assert_eq!(read_pte(&h.cpu, pde) & (PTE_A | PTE_D), PTE_A | PTE_D);
            for level in 2..=3 {
                let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, va, level));
                assert_ne!(entry & PTE_A, 0);
                assert_eq!(entry & PTE_D, 0);
            }
        }

        #[test]
        fn two_mib_leaf_maps_each_small_page_to_its_own_ram_bytes() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
            h.write(0x40_0000 + 3 * PAGE + 8, &[0xa1]);
            h.write(0x40_0000 + 9 * PAGE + 8, &[0xb2]);
            assert_eq!(h.cpu.read(va + 3 * PAGE + 8, 1).unwrap(), 0xa1);
            assert_eq!(h.cpu.read(va + 9 * PAGE + 8, 1).unwrap(), 0xb2);
        }

        #[test]
        fn two_mib_leaf_with_pat_bit_set_is_not_a_reserved_violation() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            tables.map(
                &mut h.cpu,
                va,
                0x40_0000,
                &[
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER | (1 << 12),
                ],
            );
            assert_eq!(
                h.cpu.translate(va, AccessKind::Read, false).unwrap(),
                0x40_0000
            );
        }

        #[test]
        fn one_gib_leaf_translates_inside_ram_and_beyond_it() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_4000_0000_0000;
            let pdpte = tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
            assert_eq!(
                h.cpu
                    .translate(va + 0x40_0123, AccessKind::Read, false)
                    .unwrap(),
                0x40_0123
            );
            assert_eq!(
                h.cpu
                    .translate(va + 0x2345_6789, AccessKind::Read, false)
                    .unwrap(),
                0x2345_6789
            );
            assert_eq!(
                h.cpu
                    .translate(va + ONE_GIB - 1, AccessKind::Read, false)
                    .unwrap(),
                ONE_GIB - 1
            );
            assert_eq!(read_pte(&h.cpu, pdpte) & (PTE_A | PTE_D), PTE_A);
            h.cpu
                .translate(va + 0x1000, AccessKind::Write, false)
                .unwrap();
            assert_eq!(read_pte(&h.cpu, pdpte) & (PTE_A | PTE_D), PTE_A | PTE_D);
        }

        #[test]
        fn one_gib_leaf_with_a_nonzero_physical_base() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_4000_0000_0000;
            tables.map(&mut h.cpu, va, 3 * ONE_GIB, &[PTE_PRESENT_WRITABLE_USER; 2]);
            assert_eq!(
                h.cpu
                    .translate(va + 0x1234_5678, AccessKind::Read, false)
                    .unwrap(),
                3 * ONE_GIB + 0x1234_5678
            );
        }

        #[test]
        fn one_gib_leaf_reads_ram_through_the_mapping() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_4000_0000_0000;
            tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
            h.write_u64(SCRATCH + 0x40, 0x1122_3344_5566_7788);
            assert_eq!(
                h.cpu.read(va + SCRATCH + 0x40, 8).unwrap(),
                0x1122_3344_5566_7788
            );
        }

        #[test]
        fn missing_entry_at_each_level_is_a_not_present_fault() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                let entry_address = tables.entry_address(&h.cpu, VA, level);
                let entry = read_pte(&h.cpu, entry_address);
                write_pte(&mut h.cpu, entry_address, entry & !PTE_P);
                let result = h.cpu.translate(VA + 0x10, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(0)),
                    "level {level}"
                );
                assert_eq!(h.cpu.cr2, VA + 0x10);
            }
        }

        #[test]
        fn not_present_error_codes_carry_write_user_and_fetch_bits() {
            let (mut h, _tables) = paged();
            set_nxe(&mut h.cpu, true);
            let cases = [
                (AccessKind::Read, false, 0),
                (AccessKind::Write, false, PF_W),
                (AccessKind::Exec, false, PF_ID),
                (AccessKind::Read, true, PF_U),
                (AccessKind::Write, true, PF_W | PF_U),
                (AccessKind::Exec, true, PF_U | PF_ID),
            ];
            for (access, user, expected) in cases {
                h.cpu.set_cpl(if user { 3 } else { 0 });
                let result = h.cpu.translate(VA, access, false);
                assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(expected)));
            }
        }

        #[test]
        fn upper_levels_get_accessed_even_when_the_leaf_is_not_present() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            write_pte(&mut h.cpu, leaf, 0);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap_err();
            for level in 1..=3 {
                let entry = read_pte(&h.cpu, tables.entry_address(&h.cpu, VA, level));
                assert_ne!(entry & PTE_A, 0, "level {level}");
            }
        }

        #[test]
        fn denied_access_leaves_the_leaf_accessed_and_dirty_untouched() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, READ_ONLY_USER);
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(VA, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
            assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), 0);
        }

        #[test]
        fn page_table_outside_ram_counts_as_not_present() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            let pml4e = tables.entry_address(&h.cpu, VA, 3);
            write_pte(&mut h.cpu, pml4e, 0x1000_0000 | PTE_PRESENT_WRITABLE_USER);
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
        }

        #[test]
        fn cr3_outside_ram_counts_as_not_present() {
            let mut h = harness();
            h.cpu.cr3 = 0x4000_0000;
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
        }

        #[test]
        fn cr3_low_bits_are_ignored() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.cr3 = tables.pml4 | 0x18;
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        }

        #[test]
        fn higher_half_addresses_use_the_same_walk() {
            let (mut h, mut tables) = paged();
            let va = 0xffff_ffff_8000_0000;
            tables.map_4k(&mut h.cpu, va, PA, SUPERVISOR_ONLY);
            assert_eq!(
                h.cpu.translate(va + 0x88, AccessKind::Read, false).unwrap(),
                PA + 0x88
            );
        }

        #[test]
        fn nonpresent_entry_with_reserved_bits_is_just_not_present() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            write_pte(&mut h.cpu, leaf, PA | (1 << 51) | PTE_NX);
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(0)));
        }
    }

    mod reserved_bits {
        use super::*;

        fn rsvd_code(access: AccessKind, user: bool, nxe: bool) -> u32 {
            let mut code = PF_P | PF_RSVD;
            if access == AccessKind::Write {
                code |= PF_W;
            }
            if user {
                code |= PF_U;
            }
            if access == AccessKind::Exec && nxe {
                code |= PF_ID;
            }
            code
        }

        #[test]
        fn physical_address_bits_40_to_51_are_reserved_at_every_level() {
            for level in 0..=3 {
                for bit in [40, 45, 51] {
                    let (mut h, mut tables) = paged();
                    tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                    let entry_address = tables.entry_address(&h.cpu, VA, level);
                    let entry = read_pte(&h.cpu, entry_address);
                    write_pte(&mut h.cpu, entry_address, entry | (1 << bit));
                    let result = h.cpu.translate(VA, AccessKind::Read, false);
                    assert_eq!(
                        fault_of(result),
                        (
                            VECTOR_PAGE_FAULT,
                            Some(rsvd_code(AccessKind::Read, false, false))
                        ),
                        "level {level} bit {bit}"
                    );
                    assert_eq!(h.cpu.cr2, VA);
                }
            }
        }

        #[test]
        fn bits_52_to_62_are_not_checked() {
            // Software-available and protection-key bits: the emulator does not reserve them.
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            write_pte(
                &mut h.cpu,
                leaf,
                PA | PTE_PRESENT_WRITABLE_USER | (0x7ff << 52),
            );
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        }

        #[test]
        fn no_execute_bit_is_reserved_at_every_level_without_nxe() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                let entry_address = tables.entry_address(&h.cpu, VA, level);
                let entry = read_pte(&h.cpu, entry_address);
                write_pte(&mut h.cpu, entry_address, entry | PTE_NX);
                for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                    let result = h.cpu.translate(VA, access, false);
                    assert_eq!(
                        fault_of(result),
                        (VECTOR_PAGE_FAULT, Some(rsvd_code(access, false, false))),
                        "level {level}"
                    );
                }
            }
        }

        #[test]
        fn no_execute_bit_is_valid_with_nxe_and_only_blocks_fetches() {
            let (mut h, mut tables) = paged();
            set_nxe(&mut h.cpu, true);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
            assert_eq!(h.cpu.translate(VA, AccessKind::Write, false).unwrap(), PA);
            let result = h.cpu.translate(VA, AccessKind::Exec, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID)));
        }

        #[test]
        fn reserved_fault_reports_user_and_write_bits() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            write_pte(&mut h.cpu, leaf, PA | PTE_PRESENT_WRITABLE_USER | (1 << 50));
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(VA, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U | PF_RSVD))
            );
        }

        #[test]
        fn reserved_violation_does_not_set_leaf_accessed_or_dirty() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            write_pte(&mut h.cpu, leaf, PA | PTE_PRESENT_WRITABLE_USER | (1 << 50));
            h.cpu.translate(VA, AccessKind::Write, false).unwrap_err();
            assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), 0);
        }

        #[test]
        fn page_size_bit_in_a_pml4_entry_is_a_reserved_violation() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            let pml4e = tables.entry_address(&h.cpu, VA, 3);
            let entry = read_pte(&h.cpu, pml4e);
            write_pte(&mut h.cpu, pml4e, entry | PTE_PS);
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(
                fault_of(result),
                (
                    VECTOR_PAGE_FAULT,
                    Some(rsvd_code(AccessKind::Read, false, false))
                )
            );
        }

        #[test]
        fn pat_bit_in_a_four_kib_entry_is_not_reserved() {
            let (mut h, mut tables) = paged();
            tables.map(
                &mut h.cpu,
                VA,
                PA,
                &[
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER | PTE_PS,
                ],
            );
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
        }

        #[test]
        fn two_mib_leaf_with_address_bits_20_to_13_set_is_reserved() {
            for bit in [13, 16, 20] {
                let (mut h, mut tables) = paged();
                let va = 0x0000_5000_0020_0000;
                let pde = tables.map(&mut h.cpu, va, 0x40_0000, &[PTE_PRESENT_WRITABLE_USER; 3]);
                let entry = read_pte(&h.cpu, pde);
                write_pte(&mut h.cpu, pde, entry | (1 << bit));
                let result = h.cpu.translate(va, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (
                        VECTOR_PAGE_FAULT,
                        Some(rsvd_code(AccessKind::Read, false, false))
                    ),
                    "bit {bit}"
                );
            }
        }

        #[test]
        fn one_gib_leaf_with_address_bits_29_to_13_set_is_reserved() {
            for bit in [13, 21, 29] {
                let (mut h, mut tables) = paged();
                let va = 0x0000_4000_0000_0000;
                let pdpte = tables.map(&mut h.cpu, va, 0, &[PTE_PRESENT_WRITABLE_USER; 2]);
                let entry = read_pte(&h.cpu, pdpte);
                write_pte(&mut h.cpu, pdpte, entry | (1 << bit));
                let result = h.cpu.translate(va, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (
                        VECTOR_PAGE_FAULT,
                        Some(rsvd_code(AccessKind::Read, false, false))
                    ),
                    "bit {bit}"
                );
            }
        }
    }

    mod permissions {
        use super::*;

        /// What the SDM says should happen, restated independently of `walk`.
        fn expected_error_code(
            access: AccessKind,
            user: bool,
            write_protect: bool,
            nxe: bool,
            pte_user: bool,
            pte_writable: bool,
            pte_no_execute: bool,
        ) -> Option<u32> {
            let mut code = PF_P;
            if access == AccessKind::Write {
                code |= PF_W;
            }
            if user {
                code |= PF_U;
            }
            if access == AccessKind::Exec && nxe {
                code |= PF_ID;
            }
            if pte_no_execute && !nxe {
                return Some(code | PF_RSVD);
            }
            if user && !pte_user {
                return Some(code);
            }
            if access == AccessKind::Write && !pte_writable && (user || write_protect) {
                return Some(code);
            }
            if access == AccessKind::Exec && pte_no_execute && nxe {
                return Some(code);
            }
            None
        }

        #[test]
        fn leaf_permission_matrix() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            for user in [false, true] {
                for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                    for write_protect in [false, true] {
                        for nxe in [false, true] {
                            for pte_user in [false, true] {
                                for pte_writable in [false, true] {
                                    for pte_no_execute in [false, true] {
                                        let mut entry = PA | PTE_P;
                                        if pte_user {
                                            entry |= PTE_U;
                                        }
                                        if pte_writable {
                                            entry |= PTE_W;
                                        }
                                        if pte_no_execute {
                                            entry |= PTE_NX;
                                        }
                                        write_pte(&mut h.cpu, leaf, entry);
                                        h.cpu.flush_tlb();
                                        set_wp(&mut h.cpu, write_protect);
                                        set_nxe(&mut h.cpu, nxe);
                                        h.cpu.set_cpl(if user { 3 } else { 0 });
                                        let context = format!(
                                            "user={user} access={} wp={write_protect} nxe={nxe} pte(u={pte_user} w={pte_writable} nx={pte_no_execute})",
                                            access as u8
                                        );
                                        let expected = expected_error_code(
                                            access,
                                            user,
                                            write_protect,
                                            nxe,
                                            pte_user,
                                            pte_writable,
                                            pte_no_execute,
                                        );
                                        let result = h.cpu.translate(VA + 0x44, access, false);
                                        match expected {
                                            None => {
                                                assert_eq!(result.unwrap(), PA + 0x44, "{context}");
                                                let after = read_pte(&h.cpu, leaf);
                                                assert_ne!(after & PTE_A, 0, "{context}");
                                                assert_eq!(
                                                    after & PTE_D != 0,
                                                    access == AccessKind::Write,
                                                    "{context}"
                                                );
                                            }
                                            Some(code) => {
                                                assert_eq!(
                                                    fault_of(result),
                                                    (VECTOR_PAGE_FAULT, Some(code)),
                                                    "{context}"
                                                );
                                                assert_eq!(h.cpu.cr2, VA + 0x44, "{context}");
                                                assert_eq!(
                                                    read_pte(&h.cpu, leaf) & (PTE_A | PTE_D),
                                                    0,
                                                    "{context}"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        fn weaken(cpu: &mut Cpu, tables: &PageTables, level: u32, clear: u64, set: u64) {
            let address = tables.entry_address(cpu, VA, level);
            let entry = read_pte(cpu, address);
            write_pte(cpu, address, (entry & !clear) | set);
            cpu.flush_tlb();
        }

        #[test]
        fn user_bit_is_the_and_of_every_level() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                weaken(&mut h.cpu, &tables, level, PTE_U, 0);
                assert_eq!(
                    h.cpu.translate(VA, AccessKind::Read, false).unwrap(),
                    PA,
                    "supervisor, level {level}"
                );
                h.cpu.set_cpl(3);
                let result = h.cpu.translate(VA, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)),
                    "user, level {level}"
                );
            }
        }

        #[test]
        fn writable_bit_is_the_and_of_every_level() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                weaken(&mut h.cpu, &tables, level, PTE_W, 0);
                h.cpu.set_cpl(3);
                let result = h.cpu.translate(VA, AccessKind::Write, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U)),
                    "user write, level {level}"
                );
                assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
            }
        }

        #[test]
        fn supervisor_write_to_a_read_only_level_depends_on_cr0_wp() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                weaken(&mut h.cpu, &tables, level, PTE_W, 0);
                set_wp(&mut h.cpu, false);
                assert_eq!(
                    h.cpu.translate(VA, AccessKind::Write, false).unwrap(),
                    PA,
                    "WP clear, level {level}"
                );
                h.cpu.flush_tlb();
                set_wp(&mut h.cpu, true);
                let result = h.cpu.translate(VA, AccessKind::Write, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(PF_P | PF_W)),
                    "WP set, level {level}"
                );
            }
        }

        #[test]
        fn no_execute_is_the_or_of_every_level() {
            for level in 0..=3 {
                let (mut h, mut tables) = paged();
                set_nxe(&mut h.cpu, true);
                tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
                weaken(&mut h.cpu, &tables, level, 0, PTE_NX);
                let result = h.cpu.translate(VA, AccessKind::Exec, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID)),
                    "level {level}"
                );
                assert_eq!(h.cpu.translate(VA, AccessKind::Read, false).unwrap(), PA);
            }
        }

        #[test]
        fn large_page_leaf_applies_its_own_permissions() {
            let (mut h, mut tables) = paged();
            let va = 0x0000_5000_0020_0000;
            tables.map(
                &mut h.cpu,
                va,
                0x40_0000,
                &[
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_PRESENT_WRITABLE_USER,
                    PTE_P | PTE_U,
                ],
            );
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(va + 0x1000, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
            assert_eq!(
                h.cpu
                    .translate(va + 0x1000, AccessKind::Read, false)
                    .unwrap(),
                0x40_0000 + 0x1000
            );
        }

        #[test]
        fn user_access_to_a_supervisor_only_page_faults_but_implicit_access_succeeds() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
            assert_eq!(h.cpu.translate(VA, AccessKind::Read, true).unwrap(), PA);
        }
    }

    mod tlb {
        use super::*;

        #[test]
        fn read_miss_fills_read_and_exec_tags_but_withholds_write_while_clean() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA + 8, AccessKind::Read, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(entry.tag_r, VA);
            assert_eq!(entry.tag_x, VA);
            assert_eq!(entry.tag_w, TLB_TAG_INVALID);
        }

        #[test]
        fn addend_turns_a_linear_address_into_the_host_address() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(
                (VA + 0x123).wrapping_add(entry.addend),
                host_address(&h.cpu, PA + 0x123)
            );
        }

        #[test]
        fn write_miss_walks_sets_dirty_and_grants_the_write_tag() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(entry.tag_w, VA);
            assert_eq!(entry.tag_r, VA);
        }

        #[test]
        fn already_dirty_page_gets_its_write_tag_on_a_read_miss() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
        }

        #[test]
        fn first_store_to_a_clean_page_goes_through_the_walk_that_sets_dirty() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.read(VA, 1).unwrap();
            assert_eq!(read_pte(&h.cpu, leaf) & PTE_D, 0);
            h.cpu.write(VA, 1, 0x5a).unwrap();
            assert_ne!(read_pte(&h.cpu, leaf) & PTE_D, 0);
            assert_eq!(h.read(PA, 1), [0x5a]);
        }

        #[test]
        fn write_tag_is_withheld_for_pages_holding_code() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
            h.cpu.mark_code(PA);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(entry.tag_w, TLB_TAG_INVALID);
            assert_eq!(entry.tag_r, VA);
        }

        #[test]
        fn write_tag_returns_once_the_page_no_longer_holds_code() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
            h.cpu.mark_code(PA);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            h.cpu.flush_decode();
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
        }

        #[test]
        fn no_execute_page_gets_no_exec_tag() {
            let (mut h, mut tables) = paged();
            set_nxe(&mut h.cpu, true);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(entry.tag_r, VA);
            assert_eq!(entry.tag_x, TLB_TAG_INVALID);
        }

        #[test]
        fn read_only_page_gets_a_supervisor_write_tag_only_without_wp() {
            for write_protect in [false, true] {
                let (mut h, mut tables) = paged();
                set_wp(&mut h.cpu, write_protect);
                tables.map_4k(&mut h.cpu, VA, PA, PTE_P | PTE_U | PTE_D);
                h.cpu.translate(VA, AccessKind::Read, false).unwrap();
                let granted = slot_of(&h.cpu, VA, false).tag_w == VA;
                assert_eq!(granted, !write_protect, "WP={write_protect}");
            }
        }

        #[test]
        fn user_never_gets_a_write_tag_on_a_read_only_page() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, READ_ONLY_USER | PTE_D);
            h.cpu.set_cpl(3);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, true).tag_w, TLB_TAG_INVALID);
            assert_eq!(slot_of(&h.cpu, VA, true).tag_r, VA);
        }

        #[test]
        fn user_and_supervisor_accesses_fill_separate_halves() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.set_cpl(3);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, true).tag_r, VA);
            assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
            // Implicit accesses at CPL 3 use the supervisor half.
            h.cpu.translate(VA, AccessKind::Read, true).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_r, VA);
        }

        #[test]
        fn supervisor_half_is_not_visible_to_user_accesses() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu.set_cpl(3);
            // A stale supervisor entry must not let CPL 3 read a supervisor page.
            let result = h.cpu.translate(VA, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
        }

        #[test]
        fn hit_uses_the_cached_translation_without_walking_again() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
                PA + 4
            );
        }

        #[test]
        fn invlpg_drops_the_cached_translation() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
            h.cpu.invlpg(VA + 0x7f0);
            assert_eq!(
                h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
                PA_NEXT + 4
            );
        }

        #[test]
        fn invlpg_clears_the_slot_in_both_halves() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, true).unwrap();
            h.cpu.set_cpl(3);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu.invlpg(VA);
            let slot = tlb_slot(VA, false);
            assert_eq!(h.cpu.tlb[slot].tag_r, TLB_TAG_INVALID);
            assert_eq!(
                h.cpu.tlb[TLB_ENTRIES_PER_MODE + slot].tag_r,
                TLB_TAG_INVALID
            );
        }

        #[test]
        fn invlpg_leaves_other_slots_alone() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu
                .translate(VA_NEXT_PAGE, AccessKind::Read, false)
                .unwrap();
            h.cpu.invlpg(VA);
            assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_r, VA_NEXT_PAGE);
        }

        #[test]
        fn invlpg_also_drops_another_page_that_shares_the_slot() {
            let (mut h, mut tables) = paged();
            let aliasing_va = VA + TLB_ENTRIES_PER_MODE as u64 * PAGE;
            tables.map_4k(&mut h.cpu, aliasing_va, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
            h.cpu
                .translate(aliasing_va, AccessKind::Read, false)
                .unwrap();
            assert_eq!(tlb_slot(VA, false), tlb_slot(aliasing_va, false));
            h.cpu.invlpg(VA);
            assert_eq!(slot_of(&h.cpu, aliasing_va, false).tag_r, TLB_TAG_INVALID);
        }

        #[test]
        fn pages_that_share_a_slot_evict_each_other() {
            let (mut h, mut tables) = paged();
            let aliasing_va = VA + TLB_ENTRIES_PER_MODE as u64 * PAGE;
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(&mut h.cpu, aliasing_va, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            h.cpu
                .translate(aliasing_va, AccessKind::Read, false)
                .unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!(entry.tag_r, aliasing_va);
            assert_eq!(
                entry.tag_w, TLB_TAG_INVALID,
                "the other page's write tag must not leak"
            );
            assert_eq!(entry.tag_x, aliasing_va);
            assert_eq!(
                (aliasing_va + 8).wrapping_add(entry.addend),
                host_address(&h.cpu, PA_NEXT + 8)
            );
        }

        #[test]
        fn refill_for_the_same_page_keeps_the_other_tags() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            let entry = slot_of(&h.cpu, VA, false);
            assert_eq!((entry.tag_r, entry.tag_w, entry.tag_x), (VA, VA, VA));
        }

        #[test]
        fn flush_tlb_empties_every_entry_in_both_halves() {
            let (mut h, mut tables) = paged();
            for i in 0..8 {
                let va = VA + i * PAGE;
                tables.map_4k(&mut h.cpu, va, PA + i * PAGE, PTE_PRESENT_WRITABLE_USER);
                h.cpu.translate(va, AccessKind::Write, true).unwrap();
            }
            h.cpu.set_cpl(3);
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu.flush_tlb();
            assert_eq!(h.cpu.tlb.len(), 2 * TLB_ENTRIES_PER_MODE);
            for entry in &h.cpu.tlb {
                assert_eq!(
                    (entry.tag_r, entry.tag_w, entry.tag_x, entry.addend),
                    (TLB_TAG_INVALID, TLB_TAG_INVALID, TLB_TAG_INVALID, 0)
                );
            }
        }

        #[test]
        fn pages_outside_ram_are_never_cached() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, 0xc000_0000, PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                h.cpu.translate(VA + 4, AccessKind::Read, false).unwrap(),
                0xc000_0004
            );
            assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
        }

        #[test]
        fn last_ram_page_is_cached_and_the_first_page_past_ram_is_not() {
            let (mut h, mut tables) = paged();
            let last_ram_page = FIRST_UNMAPPED_PA - PAGE;
            tables.map_4k(&mut h.cpu, VA, last_ram_page, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(
                &mut h.cpu,
                VA_NEXT_PAGE,
                FIRST_UNMAPPED_PA,
                PTE_PRESENT_WRITABLE_USER,
            );
            h.cpu.translate(VA, AccessKind::Read, false).unwrap();
            h.cpu
                .translate(VA_NEXT_PAGE, AccessKind::Read, false)
                .unwrap();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_r, VA);
            assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_r, TLB_TAG_INVALID);
        }

        #[test]
        fn faulting_translation_leaves_the_tlb_untouched() {
            let (mut h, _tables) = paged();
            h.cpu.translate(VA, AccessKind::Read, false).unwrap_err();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_r, TLB_TAG_INVALID);
        }

        #[test]
        fn mark_code_revokes_write_tags_in_both_halves_but_keeps_read_tags() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
            h.cpu.translate(VA, AccessKind::Write, true).unwrap();
            h.cpu.set_cpl(3);
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            assert_eq!(slot_of(&h.cpu, VA, false).tag_w, VA);
            assert_eq!(slot_of(&h.cpu, VA, true).tag_w, VA);
            h.cpu.mark_code(PA + 0x10);
            for user in [false, true] {
                let entry = slot_of(&h.cpu, VA, user);
                assert_eq!(entry.tag_w, TLB_TAG_INVALID, "user={user}");
                assert_eq!(entry.tag_r, VA, "user={user}");
            }
        }

        #[test]
        fn mark_code_leaves_write_tags_of_other_pages() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_D);
            tables.map_4k(
                &mut h.cpu,
                VA_NEXT_PAGE,
                PA_NEXT,
                PTE_PRESENT_WRITABLE_USER | PTE_D,
            );
            h.cpu.translate(VA, AccessKind::Write, false).unwrap();
            h.cpu
                .translate(VA_NEXT_PAGE, AccessKind::Write, false)
                .unwrap();
            h.cpu.mark_code(PA);
            assert_eq!(slot_of(&h.cpu, VA, false).tag_w, TLB_TAG_INVALID);
            assert_eq!(slot_of(&h.cpu, VA_NEXT_PAGE, false).tag_w, VA_NEXT_PAGE);
        }

        #[test]
        fn set_cpl_switches_the_active_tlb_half() {
            let mut h = harness();
            let base = h.cpu.tlb.as_ptr() as u64;
            let half = (TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>()) as u64;
            assert_eq!(h.cpu.tlb_active_base, base);
            h.cpu.set_cpl(3);
            assert_eq!(h.cpu.cpl, 3);
            assert_eq!(h.cpu.tlb_active_base, base + half);
            for ring in [0, 1, 2] {
                h.cpu.set_cpl(ring);
                assert_eq!(
                    h.cpu.tlb_active_base, base,
                    "ring {ring} uses the supervisor half"
                );
            }
        }
    }

    mod large_pages {
        use super::*;

        const LARGE_VA: u64 = 0x0000_5000_0020_0000;
        const GIGA_VA: u64 = 0x0000_4000_4000_0000;
        const LARGE_PA: u64 = 0x40_0000;
        const OTHER_LARGE_PA: u64 = 0x60_0000;
        const LARGE_2M: u64 = 1 << 21;
        const FULL: u64 = PTE_PRESENT_WRITABLE_USER;

        fn map_2m(h: &mut Harness, tables: &mut PageTables, flags: u64) -> u64 {
            tables.map(&mut h.cpu, LARGE_VA, LARGE_PA, &[FULL, FULL, flags])
        }

        fn translate(h: &mut Harness, linear: u64, access: AccessKind) -> u64 {
            h.cpu.translate(linear, access, false).unwrap()
        }

        #[test]
        fn a_cached_2mib_leaf_serves_the_rest_of_its_pages_without_a_walk() {
            let (mut h, mut tables) = paged();
            let leaf = map_2m(&mut h, &mut tables, FULL | PTE_A | PTE_D);
            assert_eq!(translate(&mut h, LARGE_VA, AccessKind::Read), LARGE_PA);
            // The guest edits its page table without a flush: real hardware keeps using the
            // cached large translation, and so must the walker, or it is not caching anything.
            write_pte(
                &mut h.cpu,
                leaf,
                OTHER_LARGE_PA | FULL | PTE_PS | PTE_A | PTE_D,
            );
            assert_eq!(
                translate(&mut h, LARGE_VA + 0x5000, AccessKind::Read),
                LARGE_PA + 0x5000
            );
            assert_eq!(
                translate(&mut h, LARGE_VA + LARGE_2M - PAGE, AccessKind::Write),
                LARGE_PA + LARGE_2M - PAGE
            );
        }

        #[test]
        fn a_cached_1gib_leaf_serves_pages_far_apart() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map(&mut h.cpu, GIGA_VA, 0, &[FULL, FULL | PTE_A | PTE_D]);
            assert_eq!(
                translate(&mut h, GIGA_VA + 0x1000, AccessKind::Read),
                0x1000
            );
            write_pte(&mut h.cpu, leaf, 0x20_0000 | FULL | PTE_PS | PTE_A | PTE_D);
            let far = 0x30_0000;
            assert_eq!(translate(&mut h, GIGA_VA + far, AccessKind::Read), far);
        }

        #[test]
        fn invlpg_anywhere_in_the_leaf_drops_the_cached_translation() {
            let (mut h, mut tables) = paged();
            let leaf = map_2m(&mut h, &mut tables, FULL | PTE_A | PTE_D);
            translate(&mut h, LARGE_VA, AccessKind::Read);
            write_pte(
                &mut h.cpu,
                leaf,
                OTHER_LARGE_PA | FULL | PTE_PS | PTE_A | PTE_D,
            );
            h.cpu.invlpg(LARGE_VA + 0x1f_f000);
            assert_eq!(
                translate(&mut h, LARGE_VA + 0x3000, AccessKind::Read),
                OTHER_LARGE_PA + 0x3000
            );
        }

        #[test]
        fn a_full_tlb_flush_drops_the_cached_translation() {
            let (mut h, mut tables) = paged();
            let leaf = map_2m(&mut h, &mut tables, FULL | PTE_A | PTE_D);
            translate(&mut h, LARGE_VA, AccessKind::Read);
            write_pte(
                &mut h.cpu,
                leaf,
                OTHER_LARGE_PA | FULL | PTE_PS | PTE_A | PTE_D,
            );
            h.cpu.flush_tlb();
            assert_eq!(
                translate(&mut h, LARGE_VA + 0x3000, AccessKind::Read),
                OTHER_LARGE_PA + 0x3000
            );
        }

        #[test]
        fn the_first_store_through_a_cached_clean_leaf_still_sets_dirty() {
            let (mut h, mut tables) = paged();
            let leaf = map_2m(&mut h, &mut tables, FULL);
            translate(&mut h, LARGE_VA, AccessKind::Read);
            assert_eq!(read_pte(&h.cpu, leaf) & PTE_D, 0);
            translate(&mut h, LARGE_VA + 0x7000, AccessKind::Write);
            assert_ne!(read_pte(&h.cpu, leaf) & PTE_D, 0);
        }

        #[test]
        fn a_cached_supervisor_leaf_still_refuses_user_access() {
            let (mut h, mut tables) = paged();
            tables.map(
                &mut h.cpu,
                LARGE_VA,
                LARGE_PA,
                &[FULL, FULL, SUPERVISOR_ONLY | PTE_A | PTE_D],
            );
            translate(&mut h, LARGE_VA, AccessKind::Read);
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(LARGE_VA + 0x2000, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_PAGE_FAULT, Some(PF_P | PF_U)));
        }

        #[test]
        fn a_cached_read_only_leaf_still_refuses_a_user_store() {
            let (mut h, mut tables) = paged();
            tables.map(
                &mut h.cpu,
                LARGE_VA,
                LARGE_PA,
                &[FULL, FULL, READ_ONLY_USER | PTE_A | PTE_D],
            );
            h.cpu.set_cpl(3);
            translate(&mut h, LARGE_VA, AccessKind::Read);
            let result = h.cpu.translate(LARGE_VA + 0x2000, AccessKind::Write, false);
            assert_eq!(
                fault_of(result),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
        }

        #[test]
        fn a_neighbouring_leaf_is_not_served_from_the_cache() {
            let (mut h, mut tables) = paged();
            map_2m(&mut h, &mut tables, FULL | PTE_A | PTE_D);
            translate(&mut h, LARGE_VA, AccessKind::Read);
            tables.map(
                &mut h.cpu,
                LARGE_VA + LARGE_2M,
                OTHER_LARGE_PA,
                &[FULL, FULL, FULL | PTE_A | PTE_D],
            );
            assert_eq!(
                translate(&mut h, LARGE_VA + LARGE_2M + 0x1000, AccessKind::Read),
                OTHER_LARGE_PA + 0x1000
            );
        }

        #[test]
        fn many_leaves_beyond_the_cache_capacity_all_translate_correctly() {
            let (mut h, mut tables) = paged();
            let leaves = LARGE_PAGE_CACHE_ENTRIES as u64 * 3;
            for i in 0..leaves {
                let leaf_flags = FULL | PTE_A | PTE_D;
                tables.map(
                    &mut h.cpu,
                    LARGE_VA + i * LARGE_2M,
                    LARGE_PA,
                    &[FULL, FULL, leaf_flags],
                );
            }
            for round in 0..2 {
                for i in 0..leaves {
                    let linear = LARGE_VA + i * LARGE_2M + 0x2000 * (round + 1);
                    assert_eq!(
                        translate(&mut h, linear, AccessKind::Read),
                        LARGE_PA + 0x2000 * (round + 1)
                    );
                }
            }
        }
    }

    mod canonical_addresses {
        use super::*;

        #[test]
        fn first_non_canonical_addresses_raise_gp0() {
            let mut h = harness();
            for address in [
                NOT_CANONICAL,
                LAST_NON_CANONICAL,
                0x1234_5678_0000_0000,
                u64::MAX >> 1,
            ] {
                let result = h.cpu.translate(address, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_GENERAL_PROTECTION, Some(0)),
                    "{address:#x}"
                );
            }
        }

        #[test]
        fn last_low_and_first_high_canonical_addresses_reach_the_walk() {
            let mut h = harness();
            // The boot identity map only covers low memory, so these fault as not present: a
            // page fault, not the #GP of a non-canonical address.
            for address in [LAST_LOW_CANONICAL, FIRST_HIGH_CANONICAL, u64::MAX] {
                let result = h.cpu.translate(address, AccessKind::Read, false);
                assert_eq!(
                    fault_of(result),
                    (VECTOR_PAGE_FAULT, Some(0)),
                    "{address:#x}"
                );
            }
        }

        #[test]
        fn non_canonical_check_applies_to_every_access_kind_and_privilege() {
            let mut h = harness();
            for access in [AccessKind::Read, AccessKind::Write, AccessKind::Exec] {
                for supervisor in [false, true] {
                    let result = h.cpu.translate(NOT_CANONICAL, access, supervisor);
                    assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
                }
            }
            h.cpu.set_cpl(3);
            let result = h.cpu.translate(NOT_CANONICAL, AccessKind::Read, false);
            assert_eq!(fault_of(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
        }

        #[test]
        fn non_canonical_fault_does_not_touch_cr2() {
            let mut h = harness();
            h.cpu.cr2 = 0x1111;
            h.cpu
                .translate(NOT_CANONICAL, AccessKind::Read, false)
                .unwrap_err();
            assert_eq!(h.cpu.cr2, 0x1111);
        }

        #[test]
        fn data_accesses_raise_gp0_and_stack_accesses_ss0() {
            let mut h = harness();
            assert_eq!(
                fault_of(h.cpu.read(NOT_CANONICAL, 8)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.write(NOT_CANONICAL, 8, 0)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.read_stack(NOT_CANONICAL, 8)),
                (VECTOR_STACK_FAULT, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.write_stack(NOT_CANONICAL, 8, 0)),
                (VECTOR_STACK_FAULT, Some(0))
            );
        }

        #[test]
        fn implicit_system_accesses_to_non_canonical_addresses_raise_gp0() {
            let mut h = harness();
            assert_eq!(
                fault_of(h.cpu.read_sys(NOT_CANONICAL, 8)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.write_sys(NOT_CANONICAL, 8, 0)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
        }

        #[test]
        fn stack_accesses_keep_page_faults_as_page_faults() {
            let (mut h, _tables) = paged();
            assert_eq!(
                fault_of(h.cpu.read_stack(VA, 8)),
                (VECTOR_PAGE_FAULT, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.write_stack(VA, 8, 1)),
                (VECTOR_PAGE_FAULT, Some(PF_W))
            );
        }

        #[test]
        fn stack_accesses_work_like_data_accesses_on_valid_memory() {
            let mut h = harness();
            h.cpu
                .write_stack(SCRATCH + 0x10, 8, 0xfeed_face_cafe_beef)
                .unwrap();
            assert_eq!(h.read_u64(SCRATCH + 0x10), 0xfeed_face_cafe_beef);
            assert_eq!(
                h.cpu.read_stack(SCRATCH + 0x10, 8).unwrap(),
                0xfeed_face_cafe_beef
            );
        }

        #[test]
        fn access_crossing_into_the_non_canonical_hole_faults() {
            let (mut h, mut tables) = paged();
            let last_page = LAST_LOW_CANONICAL & !PAGE_OFFSET_MASK;
            tables.map_4k(&mut h.cpu, last_page, PA, PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                fault_of(h.cpu.read(LAST_LOW_CANONICAL - 3, 8)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.read_stack(LAST_LOW_CANONICAL - 3, 8)),
                (VECTOR_STACK_FAULT, Some(0))
            );
            assert_eq!(
                fault_of(h.cpu.write(LAST_LOW_CANONICAL - 3, 8, 0)),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
            // Entirely inside the last canonical page is fine.
            h.cpu.read(LAST_LOW_CANONICAL - 7, 8).unwrap();
        }
    }

    mod data_accesses {
        use super::*;

        fn two_pages() -> Harness {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
            h
        }

        /// Where byte `index` of an access starting `offset` bytes into the first page lives.
        fn physical_of(offset: u64, index: u64) -> u64 {
            let position = offset + index;
            if position < PAGE {
                PA + position
            } else {
                PA_NEXT + (position - PAGE)
            }
        }

        const PATTERN: u64 = 0x8877_6655_4433_2211;

        #[test]
        fn write_then_read_roundtrips_at_every_offset_for_every_size() {
            for size in [1u8, 2, 4, 8] {
                for offset in 0..PAGE {
                    // All straddling offsets plus a few interior ones.
                    let interesting =
                        offset + size as u64 > PAGE || offset < 3 || offset == PAGE / 2;
                    if !interesting {
                        continue;
                    }
                    let mut h = two_pages();
                    let value = PATTERN & mask(size);
                    h.cpu.write(VA + offset, size, value).unwrap();
                    for index in 0..size as u64 {
                        assert_eq!(
                            h.read(physical_of(offset, index), 1)[0],
                            (value >> (8 * index)) as u8,
                            "size {size} offset {offset:#x} byte {index}"
                        );
                    }
                    assert_eq!(
                        h.cpu.read(VA + offset, size).unwrap(),
                        value,
                        "size {size} offset {offset:#x} (warm TLB)"
                    );
                    h.cpu.flush_tlb();
                    assert_eq!(
                        h.cpu.read(VA + offset, size).unwrap(),
                        value,
                        "size {size} offset {offset:#x} (cold TLB)"
                    );
                }
            }
        }

        #[test]
        fn straddling_write_does_not_touch_bytes_outside_the_access() {
            let mut h = two_pages();
            h.write(PA + PAGE - 8, &[0xee; 8]);
            h.write(PA_NEXT, &[0xdd; 8]);
            h.cpu.write(VA + PAGE - 3, 4, 0x0403_0201).unwrap();
            assert_eq!(
                h.read(PA + PAGE - 8, 8),
                [0xee, 0xee, 0xee, 0xee, 0xee, 1, 2, 3]
            );
            assert_eq!(
                h.read(PA_NEXT, 8),
                [4, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd, 0xdd]
            );
        }

        #[test]
        fn ordered_memory_accesses_touch_exactly_their_own_bytes_at_every_offset() {
            // A vCPU on its own thread reads and writes through `atomic::load`/`store`. A
            // straddling access reaches them in pieces of 1 to 7 bytes, and widening one to a
            // host atomic width would clobber (or return) its neighbours.
            for size in [1u8, 2, 4, 8] {
                for offset in (PAGE - 8)..PAGE {
                    let mut h = two_pages();
                    h.cpu.ordered_memory = true;
                    h.write(PA, &[0xee; 16]);
                    h.write(PA + PAGE - 16, &[0xee; 16]);
                    h.write(PA_NEXT, &[0xdd; 16]);
                    let value = PATTERN & mask(size);
                    h.cpu.write(VA + offset, size, value).unwrap();
                    for index in 0..size as u64 {
                        assert_eq!(
                            h.read(physical_of(offset, index), 1)[0],
                            (value >> (8 * index)) as u8,
                            "size {size} offset {offset:#x} byte {index}"
                        );
                    }
                    let spill = (offset + size as u64).saturating_sub(PAGE);
                    assert!(
                        h.read(PA_NEXT + spill, 8).iter().all(|&byte| byte == 0xdd),
                        "size {size} offset {offset:#x} wrote past its end"
                    );
                    assert_eq!(h.cpu.read(VA + offset, size).unwrap(), value);
                    h.cpu.flush_tlb();
                    assert_eq!(h.cpu.read(VA + offset, size).unwrap(), value);
                }
            }
        }

        #[test]
        fn read_of_sizes_one_to_eight_returns_zero_extended_values() {
            let mut h = two_pages();
            h.write(PA, &[0xff; 16]);
            for size in [1u8, 2, 4, 8] {
                assert_eq!(h.cpu.read(VA, size).unwrap(), mask(size), "size {size}");
            }
        }

        #[test]
        fn narrow_write_leaves_neighbouring_bytes() {
            let mut h = two_pages();
            h.write(PA, &[0xaa; 8]);
            h.cpu.write(VA + 2, 2, 0x1234).unwrap();
            assert_eq!(
                h.read(PA, 8),
                [0xaa, 0xaa, 0x34, 0x12, 0xaa, 0xaa, 0xaa, 0xaa]
            );
        }

        #[test]
        fn value_wider_than_the_access_is_truncated_on_write() {
            let mut h = two_pages();
            h.cpu.write(VA, 2, 0xffff_ffff_ffff_1234).unwrap();
            assert_eq!(h.read(PA, 4), [0x34, 0x12, 0, 0]);
        }

        #[test]
        fn sixteen_byte_accesses_roundtrip_at_every_straddling_offset() {
            for offset in (PAGE - 16)..PAGE {
                let mut h = two_pages();
                let value: u128 = 0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1110;
                h.cpu.write128(VA + offset, value).unwrap();
                for index in 0..16u64 {
                    assert_eq!(
                        h.read(physical_of(offset, index), 1)[0],
                        0x10 + index as u8,
                        "offset {offset:#x} byte {index}"
                    );
                }
                assert_eq!(
                    h.cpu.read128(VA + offset).unwrap(),
                    value,
                    "offset {offset:#x}"
                );
                h.cpu.flush_tlb();
                assert_eq!(
                    h.cpu.read128(VA + offset).unwrap(),
                    value,
                    "cold, offset {offset:#x}"
                );
            }
        }

        #[test]
        fn straddling_read_with_unmapped_second_page_reports_the_second_page_in_cr2() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                fault_of(h.cpu.read(VA + PAGE - 2, 4)),
                (VECTOR_PAGE_FAULT, Some(0))
            );
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        }

        #[test]
        fn straddling_write_with_unmapped_second_page_writes_nothing() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.write(PA + PAGE - 8, &[0x77; 8]);
            assert_eq!(
                fault_of(h.cpu.write(VA + PAGE - 4, 8, u64::MAX)),
                (VECTOR_PAGE_FAULT, Some(PF_W))
            );
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
            assert_eq!(h.read(PA + PAGE - 8, 8), [0x77; 8]);
        }

        #[test]
        fn straddling_write_into_a_read_only_second_page_for_user_writes_nothing() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, READ_ONLY_USER);
            h.cpu.set_cpl(3);
            assert_eq!(
                fault_of(h.cpu.write(VA + PAGE - 2, 4, 0xaabb_ccdd)),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
            assert_eq!(h.read(PA + PAGE - 2, 2), [0, 0]);
            assert_eq!(h.read(PA_NEXT, 2), [0, 0]);
        }

        #[test]
        fn write128_probes_both_ends_before_storing() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                fault_of(h.cpu.write128(VA + PAGE - 8, u128::MAX)),
                (VECTOR_PAGE_FAULT, Some(PF_W))
            );
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
            assert_eq!(h.read(PA + PAGE - 8, 8), [0; 8]);
        }

        #[test]
        fn read128_with_unmapped_second_page_faults() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            assert_eq!(
                fault_of(h.cpu.read128(VA + PAGE - 8 + 1)),
                (VECTOR_PAGE_FAULT, Some(0))
            );
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        }

        #[test]
        fn sys_access_reads_supervisor_pages_from_ring_three_but_data_access_does_not() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
            h.write_u64(PA, 0x0123_4567_89ab_cdef);
            h.cpu.set_cpl(3);
            assert_eq!(
                fault_of(h.cpu.read(VA, 8)),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_U))
            );
            assert_eq!(h.cpu.read_sys(VA, 8).unwrap(), 0x0123_4567_89ab_cdef);
            h.cpu.write_sys(VA + 8, 8, 0x42).unwrap();
            assert_eq!(h.read_u64(PA + 8), 0x42);
            assert_eq!(
                fault_of(h.cpu.write(VA + 8, 8, 1)),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
        }

        #[test]
        fn user_read_of_a_user_page_succeeds_through_the_user_tlb_half() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.write_u64(PA, 0x55);
            h.cpu.set_cpl(3);
            assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x55);
            assert_eq!(h.cpu.read(VA, 8).unwrap(), 0x55);
        }

        #[test]
        fn stores_set_the_dirty_bit_even_when_the_page_was_read_first() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.cpu.read(VA, 8).unwrap();
            h.cpu.write(VA + 8, 8, 1).unwrap();
            assert_eq!(read_pte(&h.cpu, leaf) & (PTE_A | PTE_D), PTE_A | PTE_D);
        }

        #[test]
        fn read_after_remap_and_invlpg_sees_the_new_page() {
            let (mut h, mut tables) = paged();
            let leaf = tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.write_u64(PA, 1);
            h.write_u64(PA_NEXT, 2);
            assert_eq!(h.cpu.read(VA, 8).unwrap(), 1);
            write_pte(&mut h.cpu, leaf, PA_NEXT | PTE_PRESENT_WRITABLE_USER);
            h.cpu.invlpg(VA);
            assert_eq!(h.cpu.read(VA, 8).unwrap(), 2);
        }
    }

    mod physical_memory {
        use super::*;

        /// The boot identity map covers 4 GiB, so linear == physical for these.
        const UNMAPPED_A: u64 = 0x4000_0000;
        const UNMAPPED_B: u64 = 0x4000_1000;

        #[test]
        fn unmapped_physical_reads_return_all_ones_for_each_size() {
            let mut h = harness();
            for size in [1u8, 2, 4, 8] {
                assert_eq!(
                    h.cpu.read(UNMAPPED_A, size).unwrap(),
                    mask(size),
                    "size {size}"
                );
            }
        }

        #[test]
        fn unmapped_physical_read_is_logged_once_per_page() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.read(UNMAPPED_A, 4).unwrap();
            h.cpu.read(UNMAPPED_A + 8, 4).unwrap();
            assert_eq!(
                logged(&log),
                ["read of unmapped physical address 0x40000000"]
            );
            h.cpu.read(UNMAPPED_B, 4).unwrap();
            assert_eq!(logged(&log).len(), 2);
        }

        #[test]
        fn unmapped_physical_write_is_dropped_and_logged_once() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.write(UNMAPPED_A, 8, 0x1234).unwrap();
            h.cpu.write(UNMAPPED_A + 8, 8, 0x5678).unwrap();
            assert_eq!(
                logged(&log),
                ["write to unmapped physical address 0x40000000"]
            );
            // Nothing was stored: the page still reads as unmapped.
            assert_eq!(h.cpu.read(UNMAPPED_A, 8).unwrap(), u64::MAX);
        }

        #[test]
        fn unmapped_read_and_write_of_the_same_page_are_logged_separately() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.read(UNMAPPED_A, 1).unwrap();
            h.cpu.write(UNMAPPED_A, 1, 0).unwrap();
            assert_eq!(logged(&log).len(), 2);
        }

        #[test]
        fn log_keys_for_unmapped_reads_and_writes_never_collide() {
            // Page 0x3000 read and page 0x2000 written used to map to the same key
            // (0x1000 + 0x3000 == 0x2000 + 0x2000).
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.cpu.read(0x3000 << PAGE_SHIFT, 1).unwrap();
            h.cpu.write(0x2000 << PAGE_SHIFT, 1, 0).unwrap();
            assert_eq!(logged(&log).len(), 2, "{:?}", logged(&log));
        }

        #[test]
        fn log_keys_for_unmapped_reads_do_not_collide_across_pages() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            for page in [0x2000u64, 0x2001, 0x3000, 0x3001, 0x4000] {
                h.cpu.read(page << PAGE_SHIFT, 1).unwrap();
            }
            assert_eq!(logged(&log).len(), 5);
        }

        #[test]
        fn no_log_callback_is_fine() {
            let mut h = harness();
            assert!(h.cpu.log.is_none());
            h.cpu.read(UNMAPPED_A, 8).unwrap();
            h.cpu.write(UNMAPPED_A, 8, 1).unwrap();
        }

        #[test]
        fn last_byte_of_ram_is_ram_and_the_next_byte_is_not() {
            let mut h = harness();
            let log = capture_log(&mut h.cpu);
            h.write(FIRST_UNMAPPED_PA - 1, &[0x5c]);
            assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA - 1, 1).unwrap(), 0x5c);
            assert!(logged(&log).is_empty());
            assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA, 1).unwrap(), 0xff);
            assert_eq!(logged(&log).len(), 1);
        }

        #[test]
        fn straddle_from_ram_into_unmapped_space_combines_both_parts() {
            let mut h = harness();
            h.write(FIRST_UNMAPPED_PA - 2, &[0x11, 0x22]);
            assert_eq!(h.cpu.read(FIRST_UNMAPPED_PA - 2, 4).unwrap(), 0xffff_2211);
        }

        #[test]
        fn straddling_write_from_ram_into_unmapped_space_stores_only_the_ram_part() {
            let mut h = harness();
            h.cpu.write(FIRST_UNMAPPED_PA - 2, 4, 0xaabb_ccdd).unwrap();
            assert_eq!(h.read(FIRST_UNMAPPED_PA - 2, 2), [0xdd, 0xcc]);
        }

        fn enable_virtio_console_bar(h: &mut Harness) {
            // Slot 1 function 0, PCI command register: set memory space enable.
            const CONFIG_ADDRESS: u32 = 0x8000_0000 | (1 << 11) | 0x04;
            h.cpu.dev.io_out(0xcf8, 4, CONFIG_ADDRESS, 0);
            h.cpu.dev.io_out(0xcfc, 2, 2, 0);
        }

        #[test]
        fn mmio_region_is_routed_to_the_device_not_to_ram() {
            let mut h = Harness::with_devices(
                &[HLT],
                DevConfig {
                    virtio_consoles: vec![crate::console::ConsoleHandle::default()],
                    ..Default::default()
                },
            );
            let log = capture_log(&mut h.cpu);
            enable_virtio_console_bar(&mut h);
            // device_feature_select is a plain read/write register at offset 0 of the BAR.
            h.cpu.write(0xc000_0000, 4, 1).unwrap();
            assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 1);
            h.cpu.write(0xc000_0000, 4, 0).unwrap();
            assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 0);
            assert!(logged(&log).is_empty(), "{:?}", logged(&log));
        }

        #[test]
        fn mmio_is_unmapped_while_the_device_decoding_is_disabled() {
            let mut h = Harness::with_devices(
                &[HLT],
                DevConfig {
                    virtio_consoles: vec![crate::console::ConsoleHandle::default()],
                    ..Default::default()
                },
            );
            let log = capture_log(&mut h.cpu);
            assert_eq!(h.cpu.read(0xc000_0000, 4).unwrap(), 0xffff_ffff);
            assert_eq!(logged(&log).len(), 1);
        }

        #[test]
        fn mmio_accesses_are_never_cached_in_the_tlb() {
            let mut h = Harness::with_devices(
                &[HLT],
                DevConfig {
                    virtio_consoles: vec![crate::console::ConsoleHandle::default()],
                    ..Default::default()
                },
            );
            enable_virtio_console_bar(&mut h);
            h.cpu.read(0xc000_0000, 4).unwrap();
            assert_eq!(slot_of(&h.cpu, 0xc000_0000, false).tag_r, TLB_TAG_INVALID);
        }
    }

    mod code_pages {
        use super::*;

        #[test]
        fn mark_code_flags_the_page_and_records_it_once() {
            let mut h = harness();
            let page = (CODE_BASE >> PAGE_SHIFT) as usize;
            assert_eq!(h.cpu.page_has_code[page], 0);
            h.cpu.mark_code(CODE_BASE + 0x10);
            h.cpu.mark_code(CODE_BASE + 0x20);
            assert_eq!(h.cpu.page_has_code[page], 1);
            assert_eq!(h.cpu.pages_with_code, [page as u32]);
        }

        #[test]
        fn repeatedly_retranslated_code_page_does_not_grow_the_page_list_without_bound() {
            let mut h = harness();
            let pages = h.cpu.page_has_code.len();
            let page = 0x300usize;
            // What `code_written` does under the JIT: clear the flag, keep the list entry.
            for _ in 0..3 * pages {
                h.cpu.mark_code((page as u64) << PAGE_SHIFT);
                h.cpu.page_has_code[page] = 0;
            }
            assert!(
                h.cpu.pages_with_code.len() <= pages,
                "{}",
                h.cpu.pages_with_code.len()
            );
            // Still correct: flushing clears exactly the pages that are marked.
            h.cpu.mark_code((page as u64) << PAGE_SHIFT);
            h.cpu.mark_code(0x400 << PAGE_SHIFT);
            h.cpu.flush_decode();
            assert_eq!(h.cpu.page_has_code[page], 0);
            assert_eq!(h.cpu.page_has_code[0x400], 0);
        }

        #[test]
        fn page_list_compaction_keeps_every_page_that_is_still_marked() {
            let mut h = harness();
            let pages = h.cpu.page_has_code.len();
            for page in 0..(pages as u64 - 1) {
                h.cpu.mark_code(page << PAGE_SHIFT);
            }
            // List is now one short of full; un-flag half of it, then mark more pages to compact.
            for page in (0..pages).step_by(2) {
                h.cpu.page_has_code[page] = 0;
            }
            h.cpu.mark_code(0);
            h.cpu.mark_code(2 << PAGE_SHIFT);
            h.cpu.mark_code((pages as u64 - 1) << PAGE_SHIFT);
            h.cpu.flush_decode();
            assert!(h.cpu.page_has_code.iter().all(|flag| flag == 0));
        }

        #[test]
        fn mark_code_ignores_addresses_outside_ram() {
            let mut h = harness();
            h.cpu.mark_code(FIRST_UNMAPPED_PA);
            h.cpu.mark_code(u64::MAX);
            assert!(h.cpu.pages_with_code.is_empty());
        }

        #[test]
        fn flush_decode_clears_every_marked_page_and_bumps_the_generation() {
            let mut h = harness();
            h.cpu.mark_code(0x10_0000);
            h.cpu.mark_code(0x20_0000);
            let generation = h.cpu.decode_cache_generation;
            h.cpu.flush_decode();
            assert_eq!(h.cpu.decode_cache_generation, generation + 1);
            assert_eq!(h.cpu.flushes, 1);
            assert!(h.cpu.pages_with_code.is_empty());
            assert_eq!(h.cpu.page_has_code[0x100], 0);
            assert_eq!(h.cpu.page_has_code[0x200], 0);
        }

        #[test]
        fn store_to_a_code_page_flushes_the_decode_cache_without_a_jit() {
            let mut h = harness();
            h.cpu.fetch_decode().unwrap();
            let generation = h.cpu.decode_cache_generation;
            h.cpu.write(CODE_BASE + 0x100, 1, 0x90).unwrap();
            assert_eq!(h.cpu.decode_cache_generation, generation + 1);
            assert_eq!(h.cpu.flushes, 1);
            assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 0);
            assert_eq!(h.read(CODE_BASE + 0x100, 1), [0x90]);
        }

        #[test]
        fn store_to_a_code_page_through_a_warm_tlb_entry_is_still_caught() {
            let mut h = harness();
            // Warm a write tag first, then translate code from the same page.
            h.cpu.write(CODE_BASE + 0x200, 1, 1).unwrap();
            let code_page = CODE_BASE & !PAGE_OFFSET_MASK;
            assert_eq!(slot_of(&h.cpu, CODE_BASE, false).tag_w, code_page);
            h.cpu.fetch_decode().unwrap();
            assert_eq!(slot_of(&h.cpu, CODE_BASE, false).tag_w, TLB_TAG_INVALID);
            h.cpu.write(CODE_BASE + 0x200, 1, 2).unwrap();
            assert_eq!(h.cpu.flushes, 1);
        }

        #[test]
        fn store_to_a_non_code_page_does_not_flush() {
            let mut h = harness();
            h.cpu.fetch_decode().unwrap();
            h.cpu.write(SCRATCH, 8, 1).unwrap();
            assert_eq!(h.cpu.flushes, 0);
        }

        #[test]
        fn code_written_with_a_jit_invalidates_only_that_page_and_requests_a_block_exit() {
            let mut h = harness();
            if !h.enable_jit() {
                return;
            }
            let page = (CODE_BASE >> PAGE_SHIFT) as usize;
            h.cpu.mark_code(CODE_BASE);
            h.cpu.mark_code(SCRATCH);
            let generation = h.cpu.decode_cache_generation;
            h.cpu.write(CODE_BASE + 0x100, 1, 0x90).unwrap();
            assert_eq!(h.cpu.page_has_code[page], 0);
            assert_eq!(h.cpu.page_has_code[(SCRATCH >> PAGE_SHIFT) as usize], 1);
            assert_eq!(h.cpu.decode_cache_generation, generation + 1);
            assert!(h.cpu.block_exit_requested);
            assert_eq!(h.cpu.flushes, 0);
            assert!(h.cpu.jit.is_some(), "the JIT must be put back");
        }

        #[test]
        fn device_writes_to_code_pages_use_the_same_invalidation() {
            let mut h = harness();
            h.cpu.mark_code(CODE_BASE);
            h.cpu.code_written((CODE_BASE >> PAGE_SHIFT) as usize);
            assert_eq!(h.cpu.flushes, 1);
            assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 0);
        }
    }

    fn code_of_length(len: usize) -> Vec<u8> {
        let mut code = vec![0x66; len - 1];
        code.push(0x90);
        code
    }

    mod instruction_fetch {
        use super::*;

        // mov rax, 5; hlt
        const MOV_RAX_5: [u8; 8] = [0x48, 0xc7, 0xc0, 0x05, 0x00, 0x00, 0x00, 0xf4];
        // nop
        const NOP: u8 = 0x90;
        const ONE_BYTE_OP_NOP: u16 = 0x90;
        const OP_MOV_RM_IMM: u16 = 0xc7;
        // 14 operand-size prefixes then nop: exactly 15 bytes.
        const MAX_LENGTH_INSTRUCTION_LEN: usize = 15;

        #[test]
        fn decodes_the_instruction_at_rip() {
            let mut h = harness();
            h.write(CODE_BASE, &MOV_RAX_5);
            let insn = h.cpu.fetch_decode().unwrap();
            assert_eq!(insn.op, OP_MOV_RM_IMM);
            assert_eq!(insn.len, 7);
            assert_eq!(insn.imm, 5);
        }

        #[test]
        fn decode_follows_rip() {
            let mut h = harness();
            h.write(CODE_BASE, &MOV_RAX_5);
            h.cpu.rip = CODE_BASE + 7;
            let insn = h.cpu.fetch_decode().unwrap();
            assert_eq!(insn.op, 0xf4);
            assert_eq!(insn.len, 1);
        }

        #[test]
        fn marks_the_code_page_and_caches_the_decode() {
            let mut h = harness();
            h.write(CODE_BASE, &MOV_RAX_5);
            h.cpu.fetch_decode().unwrap();
            assert_eq!(h.cpu.page_has_code[(CODE_BASE >> PAGE_SHIFT) as usize], 1);
            // Changing RAM behind the CPU's back: the cached decode must still be returned.
            h.write(CODE_BASE, &[NOP]);
            assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
            h.cpu.flush_decode();
            assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
        }

        #[test]
        fn cache_entries_that_share_a_slot_do_not_return_each_others_instruction() {
            let mut h = harness();
            // Two physical addresses with the same cache slot but different contents.
            let first_pa = 0x1000u64;
            let second_pa = 0x1_1001u64;
            let slot = |pa: u64| (pa ^ (pa >> 16)) as usize & (DECODE_CACHE_ENTRIES - 1);
            assert_eq!(slot(first_pa), slot(second_pa));
            h.write(first_pa, &MOV_RAX_5);
            h.write(second_pa, &[NOP]);
            h.cpu.rip = first_pa;
            assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
            h.cpu.rip = second_pa;
            assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
            h.cpu.rip = first_pa;
            assert_eq!(h.cpu.fetch_decode().unwrap().op, OP_MOV_RM_IMM);
        }

        #[test]
        fn generation_bump_invalidates_even_beyond_the_old_32_bit_range() {
            let mut h = harness();
            h.write(CODE_BASE, &MOV_RAX_5);
            h.cpu.decode_cache_generation = u32::MAX as u64;
            h.cpu.fetch_decode().unwrap();
            h.write(CODE_BASE, &[NOP]);
            h.cpu.flush_decode();
            // A u32 counter would have wrapped to 0 here and matched zeroed entries.
            assert_eq!(h.cpu.decode_cache_generation, u32::MAX as u64 + 1);
            assert_eq!(h.cpu.fetch_decode().unwrap().op, ONE_BYTE_OP_NOP);
        }

        #[test]
        fn fifteen_byte_instruction_is_accepted() {
            let mut h = harness();
            h.write(CODE_BASE, &code_of_length(MAX_LENGTH_INSTRUCTION_LEN));
            assert_eq!(
                h.cpu.fetch_decode().unwrap().len as usize,
                MAX_LENGTH_INSTRUCTION_LEN
            );
        }

        #[test]
        fn sixteen_byte_instruction_raises_gp0() {
            let mut h = harness();
            h.write(CODE_BASE, &code_of_length(MAX_LENGTH_INSTRUCTION_LEN + 1));
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
        }

        #[test]
        fn long_instruction_that_decodes_by_itself_is_still_gp0_beyond_fifteen() {
            let mut h = harness();
            // 11 prefixes + mov rax, imm64 (10 bytes) = 21 bytes.
            let mut code = vec![0x66; 11];
            code.extend_from_slice(&[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8]);
            h.write(CODE_BASE, &code);
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
        }

        #[test]
        fn undecodable_opcode_is_ud() {
            let mut h = harness();
            // push es is invalid in 64-bit mode.
            h.write(CODE_BASE, &[0x06]);
            assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
        }

        #[test]
        fn invalid_opcode_after_many_prefixes_is_ud_not_gp() {
            let mut h = harness();
            let mut code = vec![0x66; 10];
            code.push(0x06);
            h.write(CODE_BASE, &code);
            assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
        }

        #[test]
        fn non_canonical_rip_raises_gp0() {
            let mut h = harness();
            h.cpu.rip = NOT_CANONICAL;
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
        }

        #[test]
        fn fetch_from_a_page_that_is_not_present_faults_with_cr2_at_rip() {
            let (mut h, _tables) = paged();
            h.cpu.rip = VA + 0x123;
            assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
            assert_eq!(h.cpu.cr2, VA + 0x123);
        }

        #[test]
        fn fetch_fault_reports_the_instruction_fetch_bit_only_with_nxe() {
            for nxe in [false, true] {
                let (mut h, _tables) = paged();
                set_nxe(&mut h.cpu, nxe);
                h.cpu.rip = VA;
                let expected = if nxe { PF_ID } else { 0 };
                assert_eq!(
                    fault_of(h.cpu.fetch_decode()),
                    (VECTOR_PAGE_FAULT, Some(expected)),
                    "nxe={nxe}"
                );
            }
        }

        #[test]
        fn fetch_from_a_no_execute_page_faults_with_the_fetch_bit() {
            let (mut h, mut tables) = paged();
            set_nxe(&mut h.cpu, true);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER | PTE_NX);
            h.write(PA, &[NOP]);
            h.cpu.rip = VA;
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID))
            );
        }

        #[test]
        fn user_fetch_from_a_supervisor_page_faults() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, SUPERVISOR_ONLY);
            h.write(PA, &[NOP]);
            h.cpu.rip = VA;
            h.cpu.set_cpl(3);
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_U))
            );
        }

        #[test]
        fn fetch_from_unmapped_physical_space_is_a_not_present_fault() {
            let mut h = harness();
            h.cpu.rip = 0x4000_0000;
            assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
            assert_eq!(h.cpu.cr2, 0x4000_0000);
        }

        fn code_split_across_pages(
            h: &mut Harness,
            tables: &mut PageTables,
            code: &[u8],
            first_len: usize,
        ) {
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(&mut h.cpu, VA_NEXT_PAGE, PA_NEXT, PTE_PRESENT_WRITABLE_USER);
            h.write(PA + PAGE - first_len as u64, &code[..first_len]);
            h.write(PA_NEXT, &code[first_len..]);
            h.cpu.rip = VA + PAGE - first_len as u64;
        }

        #[test]
        fn instruction_straddling_two_pages_decodes_from_both() {
            for first_len in 1..7 {
                let (mut h, mut tables) = paged();
                code_split_across_pages(&mut h, &mut tables, &MOV_RAX_5[..7], first_len);
                let insn = h.cpu.fetch_decode().unwrap();
                assert_eq!(insn.len, 7, "first_len {first_len}");
                assert_eq!(insn.imm, 5, "first_len {first_len}");
            }
        }

        #[test]
        fn straddling_instruction_marks_both_pages_as_code_but_is_not_cached() {
            let (mut h, mut tables) = paged();
            code_split_across_pages(&mut h, &mut tables, &MOV_RAX_5[..7], 3);
            h.cpu.fetch_decode().unwrap();
            assert_eq!(h.cpu.page_has_code[(PA >> PAGE_SHIFT) as usize], 1);
            assert_eq!(h.cpu.page_has_code[(PA_NEXT >> PAGE_SHIFT) as usize], 1);
            // Patch the second page behind the CPU's back: no stale cache may be used.
            h.write(PA_NEXT, &[9]);
            assert_eq!(h.cpu.fetch_decode().unwrap().imm, 9);
        }

        #[test]
        fn straddling_instruction_with_unmapped_second_page_faults_on_that_page() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.write(PA + PAGE - 3, &MOV_RAX_5[..3]);
            h.cpu.rip = VA + PAGE - 3;
            assert_eq!(fault_of(h.cpu.fetch_decode()), (VECTOR_PAGE_FAULT, Some(0)));
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        }

        #[test]
        fn straddling_instruction_whose_second_page_is_no_execute_faults() {
            let (mut h, mut tables) = paged();
            set_nxe(&mut h.cpu, true);
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            tables.map_4k(
                &mut h.cpu,
                VA_NEXT_PAGE,
                PA_NEXT,
                PTE_PRESENT_WRITABLE_USER | PTE_NX,
            );
            h.write(PA + PAGE - 3, &MOV_RAX_5[..3]);
            h.cpu.rip = VA + PAGE - 3;
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_ID))
            );
            assert_eq!(h.cpu.cr2, VA_NEXT_PAGE);
        }

        #[test]
        fn complete_instruction_at_the_end_of_a_page_does_not_touch_the_next_page() {
            let (mut h, mut tables) = paged();
            tables.map_4k(&mut h.cpu, VA, PA, PTE_PRESENT_WRITABLE_USER);
            h.write(PA + PAGE - 1, &[NOP]);
            h.cpu.rip = VA + PAGE - 1;
            assert_eq!(h.cpu.fetch_decode().unwrap().len, 1);
            assert_eq!(h.cpu.page_has_code[(PA_NEXT >> PAGE_SHIFT) as usize], 0);
        }

        #[test]
        fn sixteen_byte_instruction_across_a_page_boundary_raises_gp0() {
            let (mut h, mut tables) = paged();
            let code = code_of_length(MAX_LENGTH_INSTRUCTION_LEN + 1);
            code_split_across_pages(&mut h, &mut tables, &code, 8);
            assert_eq!(
                fault_of(h.cpu.fetch_decode()),
                (VECTOR_GENERAL_PROTECTION, Some(0))
            );
        }

        #[test]
        fn fifteen_byte_instruction_across_a_page_boundary_decodes() {
            let (mut h, mut tables) = paged();
            let code = code_of_length(MAX_LENGTH_INSTRUCTION_LEN);
            code_split_across_pages(&mut h, &mut tables, &code, 8);
            assert_eq!(
                h.cpu.fetch_decode().unwrap().len as usize,
                MAX_LENGTH_INSTRUCTION_LEN
            );
        }

        #[test]
        fn invalid_opcode_across_a_page_boundary_is_ud() {
            let (mut h, mut tables) = paged();
            // 66 66 66 then push es, with the invalid byte on the second page.
            code_split_across_pages(&mut h, &mut tables, &[0x66, 0x66, 0x66, 0x06], 3);
            assert_eq!(fault_of(h.cpu.fetch_decode()), (6, None));
        }
    }

    mod jit_agreement {
        use super::*;
        use crate::testkit::{Stop, assert_jit_matches_interpreter, run_both};

        const NOP: u8 = 0x90;

        fn with_halt(mut code: Vec<u8>) -> Vec<u8> {
            code.push(HLT);
            code
        }

        #[test]
        fn sixteen_byte_instruction_is_gp0_on_both_engines() {
            let code = with_halt(code_of_length(16));
            let outcome = run_both(&code, |_| {});
            assert_eq!(
                outcome.interpreter_stop,
                Stop::Fault {
                    vector: VECTOR_GENERAL_PROTECTION,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(outcome.interpreter.cpu.rip, CODE_BASE);
            assert_jit_matches_interpreter(&code, |_| {});
        }

        #[test]
        fn fifteen_byte_instruction_runs_on_both_engines() {
            let code = with_halt(code_of_length(15));
            let outcome = run_both(&code, |_| {});
            assert_eq!(outcome.interpreter_stop, Stop::Halted);
            assert_jit_matches_interpreter(&code, |_| {});
        }

        #[test]
        fn too_long_instruction_after_other_instructions_faults_only_when_reached() {
            let mut code = vec![NOP, NOP];
            code.extend(code_of_length(16));
            let outcome = run_both(&code, |_| {});
            assert_eq!(
                outcome.interpreter_stop,
                Stop::Fault {
                    vector: VECTOR_GENERAL_PROTECTION,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(outcome.interpreter.cpu.rip, CODE_BASE + 2);
            assert_jit_matches_interpreter(&code, |_| {});
        }

        #[test]
        fn undecodable_opcode_behind_prefixes_is_ud_on_both_engines() {
            // 66 66 66 push es
            let code = [0x66, 0x66, 0x66, 0x06];
            let outcome = run_both(&code, |_| {});
            assert_eq!(
                outcome.interpreter_stop,
                Stop::Fault {
                    vector: 6,
                    error_code: None,
                    software: false
                }
            );
            assert_jit_matches_interpreter(&code, |_| {});
        }
    }

    mod large_pages_on_both_engines {
        use super::*;
        use crate::testkit::{SCRATCH, Stop, assert_jit_matches_interpreter, run_both};

        const ALIAS: u64 = 0x0000_5000_0000_0000;
        const FULL: u64 = PTE_PRESENT_WRITABLE_USER;

        /// Identity-maps RAM with one 1 GiB leaf and puts a 2 MiB leaf over the scratch page at
        /// `ALIAS`; neither leaf starts out dirty.
        fn map_with_large_pages(h: &mut Harness) {
            let mut tables = PageTables::new(&mut h.cpu);
            tables.map(&mut h.cpu, 0, 0, &[FULL, FULL]);
            tables.map(
                &mut h.cpu,
                ALIAS,
                SCRATCH & !(TWO_MIB - 1),
                &[FULL, FULL, FULL],
            );
        }

        #[test]
        fn stores_and_loads_across_a_2mib_leaf_agree() {
            // mov rbx, ALIAS + (SCRATCH offset in its 2 MiB page); mov rax, 0x1234
            // mov [rbx], rax; mov [rbx+0x1000], rax; mov rcx, [rbx+0x100000]; mov rdx, [rbx+0x1000]
            let offset = SCRATCH & (TWO_MIB - 1);
            let mut code = vec![0x48, 0xbb];
            code.extend((ALIAS + offset).to_le_bytes());
            code.extend([0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00]);
            code.extend([0x48, 0x89, 0x03]);
            code.extend([0x48, 0x89, 0x83, 0x00, 0x10, 0x00, 0x00]);
            code.extend([0x48, 0x8b, 0x8b, 0x00, 0x00, 0x10, 0x00]);
            code.extend([0x48, 0x8b, 0x93, 0x00, 0x10, 0x00, 0x00]);
            code.push(HLT);
            let outcome = run_both(&code, map_with_large_pages);
            assert_eq!(outcome.interpreter_stop, Stop::Halted);
            // gpr[2] is rdx.
            assert_eq!(outcome.interpreter.cpu.gpr[2], 0x1234);
            assert_jit_matches_interpreter(&code, map_with_large_pages);
        }
    }

    mod eflags_helpers {
        use super::*;

        #[test]
        fn eflags_full_always_has_the_reserved_bit_set() {
            let mut h = harness();
            h.cpu.eflags = 0;
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE);
        }

        #[test]
        fn eflags_full_resolves_the_deferred_flags_over_stale_eflags() {
            let mut h = harness();
            h.cpu.eflags = RESERVED_ONE | CF | ZF | OF | IF;
            // 0x7f + 1 in 8 bits: result 0x80.
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0x80, 1, 0);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | IF | SF | OF | AF);
        }

        #[test]
        fn materialize_keeps_the_value_and_drops_the_deferred_state() {
            let mut h = harness();
            h.cpu.eflags = RESERVED_ONE | IF | DF;
            // 0 - 5 in 32 bits: the result is 0xfffffffb.
            h.cpu.set_lazy_flags(LazyFlagsOp::Sub, 4, 0xffff_fffb, 5, 0);
            let before = h.cpu.eflags_full();
            h.cpu.materialize_lazy_flags();
            assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
            assert_eq!(h.cpu.eflags, before);
            assert_eq!(h.cpu.eflags_full(), before);
            assert_eq!(before & (IF | DF), IF | DF);
            assert_eq!(before & ARITH, CF | AF | SF);
        }

        #[test]
        fn set_eflags_arith_replaces_only_the_arithmetic_flags() {
            let mut h = harness();
            h.cpu.eflags = RESERVED_ONE | IF | TF | CF | ZF;
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 8, 1, 1, 0);
            h.cpu.set_eflags_arith(SF | IF | DF | AC);
            assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | IF | TF | SF);
        }

        #[test]
        fn cf_reads_through_deferred_flags() {
            let mut h = harness();
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            assert!(h.cpu.cf());
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 2, 1, 0);
            assert!(!h.cpu.cf());
            h.cpu.eflags = RESERVED_ONE | CF;
            h.cpu.lazy_op = LazyFlagsOp::Eflags;
            assert!(h.cpu.cf());
        }
    }
}
