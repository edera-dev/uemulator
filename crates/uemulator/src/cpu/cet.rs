//! CET shadow stacks, the half of CET that Zen 3 and later implement (AMD APM Vol. 2 chapter
//! "Shadow Stacks", Vol. 3 for the instructions; the Intel SDM's CET chapter describes the same
//! architecture and its pseudocode is what this follows where the APM is terse). Indirect branch
//! tracking (ENDBRANCH, the IBT bits of U_CET/S_CET) is Intel only: CPUID.7.0:EDX[20] stays clear
//! and those MSR bits are reserved here.
//!
//! A shadow stack is in use at privilege level `cpl` when CR4.CET and the SH_STK_EN bit of U_CET
//! (CPL 3) or S_CET (CPL 0-2) are set, in protected mode outside virtual-8086 mode. Calls push the
//! return address on it as well, returns compare (#CP on a mismatch), and far transfers and event
//! delivery save CS, the linear return address and the old SSP there. Supervisor shadow stacks
//! are claimed through a token at their top whose bit 0 says "busy".
//!
//! The JIT does not know about any of this: `Cpu::uses_jit` keeps the run loop on the
//! interpreter while a shadow stack is in use at the current CPL. What can switch one on (a CPL
//! change, mov to CR4, wrmsr, VMRUN and #VMEXIT) always ends a block and goes back to the run
//! loop, so a chain of blocks never runs into one. xrstors can load U_CET, but it runs at CPL 0
//! and U_CET only matters at CPL 3, which takes a CPL change to reach.

use super::decode::Insn;
use super::exception::Exception;
use super::flags::{CF, VM};
use super::sys::is_canonical;
use super::*;

pub(crate) const MSR_U_CET: u32 = 0x6a0;
pub(crate) const MSR_S_CET: u32 = 0x6a2;
/// IA32_PL0_SSP..IA32_PL3_SSP: the SSP each privilege level starts with when entered from a less
/// privileged one (PL3_SSP also keeps the user SSP while the kernel runs).
pub(crate) const MSR_PL0_SSP: u32 = 0x6a4;
pub(crate) const MSR_PL3_SSP: u32 = 0x6a7;
/// IA32_INTERRUPT_SSP_TABLE_ADDR: eight SSPs, the shadow-stack counterpart of the TSS IST slots
/// (entry 0 unused).
pub(crate) const MSR_ISST_ADDR: u32 = 0x6a8;

/// CR4 bit 23. Needs CR0.WP, which cannot be cleared while it is set: shadow-stack pages are
/// read-only, and only WP keeps supervisor stores off them.
pub(crate) const CR4_CET: u64 = 1 << 23;

/// U_CET/S_CET bit 0: shadow stacks in use at that privilege.
pub(crate) const CET_SH_STK_EN: u64 = 1 << 0;
/// U_CET/S_CET bit 1: wrss is allowed at that privilege.
pub(crate) const CET_WR_SHSTK_EN: u64 = 1 << 1;
/// Everything above bit 1 belongs to IBT (ENDBR_EN, LEG_IW_EN, NO_TRACK_EN, SUPPRESS_DIS,
/// SUPPRESS, TRACKER, the legacy bitmap base), which this CPU does not have.
pub(crate) const CET_CONTROL_WRITABLE: u64 = CET_SH_STK_EN | CET_WR_SHSTK_EN;
/// The PLn_SSP MSRs hold 4-byte-aligned addresses (KVM checks the same).
const SSP_MSR_ALIGNMENT_MASK: u64 = 3;

/// #CP, the control-protection exception. It has an error code and is contributory.
pub(crate) const VECTOR_CONTROL_PROTECTION: u8 = 21;
/// #CP error codes: a near ret whose return address differs from the shadow stack's.
pub(crate) const CP_NEAR_RET: u32 = 1;
/// A far ret or iret whose CS, return address or saved SSP does not match the shadow stack.
pub(crate) const CP_FAR_RET_IRET: u32 = 2;
/// rstorssp found no valid restore token.
pub(crate) const CP_RSTORSSP: u32 = 4;
/// setssbsy found the PL0_SSP token busy or invalid.
pub(crate) const CP_SETSSBSY: u32 = 5;

/// Bit 0 of a supervisor shadow-stack token: the stack is in use. The rest of the token is its
/// own address.
const TOKEN_BUSY: u64 = 1;
/// A restore token (what saveprevssp leaves and rstorssp consumes) holds the SSP it restores to
/// plus this bit when that SSP belongs to 64-bit code; bit 1 marks a "previous SSP" token.
const TOKEN_MODE_64: u64 = 1;
const TOKEN_PREVIOUS_SSP: u64 = 2;
const TOKEN_FLAG_MASK: u64 = TOKEN_MODE_64 | TOKEN_PREVIOUS_SSP;
const SHADOW_STACK_ALIGNMENT_MASK: u64 = 7;
const SHADOW_STACK_HALF_ALIGNMENT: u64 = 4;
const LOW_32_BITS: u64 = 0xffff_ffff;
/// Bytes of the frame far calls and event delivery push: CS, the linear return address and the
/// previous SSP, 8 bytes each.
const FRAME_CS_OFFSET: u64 = 16;
const FRAME_LIP_OFFSET: u64 = 8;
const FRAME_BYTES: u64 = 24;
/// Bytes per interrupt shadow-stack table entry.
const ISST_ENTRY_BYTES: u64 = 8;
/// incssp uses only the low byte of its count register.
const INCSSP_COUNT_MASK: u64 = 0xff;

/// SSP and the CET MSRs of one vCPU.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Cet {
    pub ssp: u64,
    pub u_cet: u64,
    pub s_cet: u64,
    /// IA32_PL0_SSP..IA32_PL3_SSP.
    pub pl_ssp: [u64; 4],
    pub isst_addr: u64,
}

/// What the PLn_SSP MSRs (and the XSAVE images of them) accept: a canonical, 4-byte-aligned
/// address.
pub(super) fn ssp_msr_value_is_valid(value: u64) -> bool {
    is_canonical(value) && value & SSP_MSR_ALIGNMENT_MASK == 0
}

fn control_protection(code: u32) -> Exception {
    Exception::fault(VECTOR_CONTROL_PROTECTION, Some(code))
}

impl Cpu {
    /// Whether the shadow stack is enabled for privilege level `cpl` by CR4.CET and U_CET/S_CET.
    /// Callers that ask about the running code also need protected mode outside virtual-8086
    /// mode (`shadow_stack_active`); a transfer into a handler asks about the level it enters.
    pub(crate) fn shadow_stack_enabled_at(&self, cpl: u8) -> bool {
        let control = if cpl == USER_CPL {
            self.cet.u_cet
        } else {
            self.cet.s_cet
        };
        self.cr4 & CR4_CET != 0 && self.cr0 & CR0_PE != 0 && control & CET_SH_STK_EN != 0
    }

    /// Whether the running code has a shadow stack.
    pub(crate) fn shadow_stack_active(&self) -> bool {
        self.eflags & VM == 0 && self.shadow_stack_enabled_at(self.cpl)
    }

    /// The CET MSRs, or `None` for any other MSR.
    pub(super) fn cet_msr(&self, msr: u32) -> Option<u64> {
        Some(match msr {
            MSR_U_CET => self.cet.u_cet,
            MSR_S_CET => self.cet.s_cet,
            MSR_PL0_SSP..=MSR_PL3_SSP => self.cet.pl_ssp[(msr - MSR_PL0_SSP) as usize],
            MSR_ISST_ADDR => self.cet.isst_addr,
            _ => return None,
        })
    }

    /// Writes a CET MSR; `Ok(false)` for any other MSR. Reserved bits, a non-canonical address
    /// and a misaligned SSP are #GP(0).
    pub(super) fn write_cet_msr(&mut self, msr: u32, value: u64) -> ExecResult<bool> {
        match msr {
            MSR_U_CET | MSR_S_CET => {
                if value & !CET_CONTROL_WRITABLE != 0 {
                    return Err(Exception::general_protection(0));
                }
                if msr == MSR_U_CET {
                    self.cet.u_cet = value;
                } else {
                    self.cet.s_cet = value;
                }
            }
            MSR_PL0_SSP..=MSR_PL3_SSP => {
                if !ssp_msr_value_is_valid(value) {
                    return Err(Exception::general_protection(0));
                }
                self.cet.pl_ssp[(msr - MSR_PL0_SSP) as usize] = value;
            }
            MSR_ISST_ADDR => {
                if !is_canonical(value) {
                    return Err(Exception::general_protection(0));
                }
                self.cet.isst_addr = value;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Bytes per shadow-stack entry of a near call: 8 in 64-bit mode, 4 everywhere else.
    fn shadow_entry_size(&self) -> u8 {
        if self.is_long64() { 8 } else { 4 }
    }

    pub(super) fn shadow_read(&mut self, linear: u64, size: u8, user: bool) -> ExecResult<u64> {
        let offset = linear & PAGE_OFFSET_MASK;
        let first = self.shadow_stack_translate(linear, user, false)?;
        if offset + size as u64 <= PAGE_SIZE {
            return Ok(self.phys_read(first, size));
        }
        // Only an entry at an SSP misaligned by incsspd or rstorssp can straddle pages.
        let first_len = (PAGE_SIZE - offset) as u8;
        let second_page = (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE);
        let second = self.shadow_stack_translate(second_page, user, false)?;
        let low = self.phys_read(first, first_len);
        let high = self.phys_read(second, size - first_len);
        Ok(low | (high << (8 * first_len)))
    }

    /// A shadow-stack store; both pages of a straddling one are checked before either is written.
    pub(super) fn shadow_write(
        &mut self,
        linear: u64,
        size: u8,
        value: u64,
        user: bool,
    ) -> ExecResult<()> {
        let offset = linear & PAGE_OFFSET_MASK;
        let first = self.shadow_stack_translate(linear, user, true)?;
        if offset + size as u64 <= PAGE_SIZE {
            self.phys_write(first, size, value);
            return Ok(());
        }
        let first_len = (PAGE_SIZE - offset) as u8;
        let second_page = (linear & !PAGE_OFFSET_MASK).wrapping_add(PAGE_SIZE);
        let second = self.shadow_stack_translate(second_page, user, true)?;
        self.phys_write(first, first_len, value);
        self.phys_write(second, size - first_len, value >> (8 * first_len));
        Ok(())
    }

    /// Replaces the 8-byte token at `linear` (8-byte aligned, a supervisor shadow-stack page) with
    /// `new` if it holds `expected`, atomically against other vCPUs as the locked hardware access
    /// is. Returns what the token held.
    fn shadow_token_exchange(&mut self, linear: u64, expected: u64, new: u64) -> ExecResult<u64> {
        let pa = self.shadow_stack_translate(linear, false, true)?;
        if !self.ram.contains(pa, 8) {
            let seen = self.phys_read(pa, 8);
            if seen == expected {
                self.phys_write(pa, 8, new);
            }
            return Ok(seen);
        }
        loop {
            let seen = self.phys_read(pa, 8);
            if seen != expected {
                return Ok(seen);
            }
            if self.update_ram_u64(pa, seen, new) {
                let page = (pa >> PAGE_SHIFT) as usize;
                if self.page_has_code.get(page) != 0 {
                    self.code_written(page);
                }
                return Ok(seen);
            }
        }
    }

    fn shadow_push(&mut self, ssp: &mut u64, size: u8, value: u64, user: bool) -> ExecResult<()> {
        let at = ssp.wrapping_sub(size as u64);
        self.shadow_write(at, size, value, user)?;
        *ssp = at;
        Ok(())
    }

    fn shadow_pop(&mut self, ssp: &mut u64, size: u8, user: bool) -> ExecResult<u64> {
        let value = self.shadow_read(*ssp, size, user)?;
        *ssp = ssp.wrapping_add(size as u64);
        Ok(value)
    }

    /// The shadow-stack push of a near call: pushes `return_address` and returns the SSP to commit
    /// once the call can no longer fault (the current SSP when no shadow stack is in use).
    pub(super) fn shadow_stack_call(&mut self, return_address: u64) -> ExecResult<u64> {
        let mut ssp = self.cet.ssp;
        if self.shadow_stack_active() {
            let size = self.shadow_entry_size();
            let value = return_address & mask_for(size);
            self.shadow_push(&mut ssp, size, value, self.cpl == USER_CPL)?;
        }
        Ok(ssp)
    }

    /// The shadow-stack pop of a near return to `target`: #CP(near ret) unless the shadow stack
    /// holds the same address. Returns the SSP to commit.
    pub(super) fn shadow_stack_near_return(&mut self, target: u64) -> ExecResult<u64> {
        let mut ssp = self.cet.ssp;
        if self.shadow_stack_active() {
            let size = self.shadow_entry_size();
            let saved = self.shadow_pop(&mut ssp, size, self.cpl == USER_CPL)?;
            if saved != target & mask_for(size) {
                return Err(control_protection(CP_NEAR_RET));
            }
        }
        Ok(ssp)
    }

    /// Pushes a near call's return address on the stack and, with a shadow stack, on that too.
    /// `shadow`: false for `call` to the next instruction (displacement 0), the get-the-address
    /// idiom, which the architecture exempts so it does not unbalance the shadow stack. A fault
    /// leaves RSP and SSP as they were.
    pub(super) fn push_return_address(
        &mut self,
        size: u8,
        value: u64,
        shadow: bool,
    ) -> ExecResult<()> {
        let saved_rsp = self.gpr[GPR_RSP];
        self.push(size, value)?;
        if shadow {
            match self.shadow_stack_call(value) {
                Ok(ssp) => self.cet.ssp = ssp,
                Err(error) => {
                    self.gpr[GPR_RSP] = saved_rsp;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Pushes the CS, linear return address and SSP frame of a far call or event delivery onto
    /// the shadow stack at `ssp`, first padding it to 8 bytes with a zero dword.
    fn shadow_push_frame(
        &mut self,
        ssp: &mut u64,
        cs: u16,
        return_lip: u64,
        previous_ssp: u64,
        user: bool,
    ) -> ExecResult<()> {
        if *ssp & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            self.shadow_push(ssp, 4, 0, user)?;
        }
        self.shadow_push(ssp, 8, cs as u64, user)?;
        self.shadow_push(ssp, 8, return_lip, user)?;
        self.shadow_push(ssp, 8, previous_ssp, user)
    }

    /// The linear address of the current instruction pointer, as the shadow-stack frames record
    /// it: CS.base + rIP, 32 bits wide outside 64-bit mode.
    fn current_lip(&self) -> u64 {
        let lip = self.seg[SEG_CS].base.wrapping_add(self.rip);
        if self.is_long64() {
            lip
        } else {
            lip & LOW_32_BITS
        }
    }

    /// The shadow-stack side of a far call to code at the current privilege: pushes CS, the
    /// linear return address `return_rip` and SSP. `target_long`: the callee is 64-bit code
    /// (otherwise SSP must be below 4 GiB). Returns the SSP to commit with the call.
    pub(super) fn shadow_stack_far_call(
        &mut self,
        return_rip: u64,
        target_long: bool,
    ) -> ExecResult<u64> {
        let mut ssp = self.cet.ssp;
        if !self.shadow_stack_active() {
            return Ok(ssp);
        }
        if !(target_long && self.efer & EFER_LMA != 0) && ssp >> 32 != 0 {
            return Err(Exception::general_protection(0));
        }
        let cs = self.seg[SEG_CS].sel;
        let mut lip = self.seg[SEG_CS].base.wrapping_add(return_rip);
        if !self.is_long64() {
            lip &= LOW_32_BITS;
        }
        let previous_ssp = ssp;
        self.shadow_push_frame(&mut ssp, cs, lip, previous_ssp, self.cpl == USER_CPL)?;
        Ok(ssp)
    }

    /// Takes the supervisor shadow stack whose token is at `ssp`: the token must be its own
    /// address with the busy bit clear, and becomes busy. Anything else is #GP(0), as for every
    /// transfer that switches stacks this way (setssbsy has its own #CP).
    fn claim_supervisor_token(&mut self, ssp: u64) -> ExecResult<()> {
        let seen = self.shadow_token_exchange(ssp, ssp, ssp | TOKEN_BUSY)?;
        if seen != ssp {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// Clears the busy bit of the token at `ssp` if it is the busy token of that very address;
    /// a mismatch is silently left alone.
    fn release_supervisor_token(&mut self, ssp: u64) -> ExecResult<()> {
        self.shadow_token_exchange(ssp, ssp | TOKEN_BUSY, ssp)?;
        Ok(())
    }

    /// The shadow-stack side of event delivery and of a call through a call gate, worked out
    /// before the transfer commits (`ShadowStackSwitch::commit`). `new_cpl` is the handler's
    /// privilege, `ist` the 64-bit gate's IST index (0: none) and `handler_long` whether the
    /// handler is 64-bit code. `from_v86`: the interrupted code was virtual-8086 code, which never
    /// has a shadow stack.
    ///
    /// Entering a more privileged level, or using an IST slot, switches to that level's shadow
    /// stack (PLn_SSP or the interrupt SSP table) and claims its token; from CPL 3 the user SSP is
    /// parked in PL3_SSP. Unless the transfer comes from user mode, the old CS, linear return
    /// address and SSP are pushed on the new shadow stack for iret or far ret to check.
    pub(super) fn shadow_stack_enter(
        &mut self,
        new_cpl: u8,
        ist: u8,
        handler_long: bool,
        from_v86: bool,
    ) -> ExecResult<ShadowStackSwitch> {
        let old_cpl = self.cpl;
        let old_ssp = self.cet.ssp;
        let mut switch = ShadowStackSwitch {
            ssp: old_ssp,
            pl3_ssp: None,
        };
        if old_cpl == USER_CPL
            && new_cpl != USER_CPL
            && !from_v86
            && self.shadow_stack_enabled_at(USER_CPL)
        {
            switch.pl3_ssp = Some(old_ssp);
        }
        if !self.shadow_stack_enabled_at(new_cpl) {
            return Ok(switch);
        }
        let user = new_cpl == USER_CPL;
        let use_ist = ist != 0 && self.efer & EFER_LMA != 0 && !user;
        let mut ssp = old_ssp;
        if new_cpl < old_cpl || use_ist {
            let candidate = if use_ist {
                let entry = self
                    .cet
                    .isst_addr
                    .wrapping_add(ist as u64 * ISST_ENTRY_BYTES);
                self.read_sys(entry, 8)?
            } else {
                self.cet.pl_ssp[new_cpl as usize]
            };
            if candidate & SHADOW_STACK_ALIGNMENT_MASK != 0
                || (!handler_long && candidate >> 32 != 0)
            {
                return Err(Exception::general_protection(0));
            }
            self.claim_supervisor_token(candidate)?;
            ssp = candidate;
        }
        let from_user = old_cpl == USER_CPL && !user;
        if !from_user && !from_v86 {
            let cs = self.seg[SEG_CS].sel;
            let lip = self.current_lip();
            self.shadow_push_frame(&mut ssp, cs, lip, old_ssp, user)?;
        }
        switch.ssp = ssp;
        Ok(switch)
    }

    /// The shadow-stack side of iret (`iret`) and far ret to `new_cs:new_rip` at `new_cpl`,
    /// checked before the return commits; returns the SSP to commit with it.
    ///
    /// With a shadow stack at the current level, a return to the same or to a supervisor level
    /// pops the CS / return address / SSP frame that the call or event pushed and compares it
    /// (#CP(far ret/iret) on a difference). A return to a less privileged level then releases the
    /// token of the stack it leaves; so does an iret in IA-32e mode that goes back to another
    /// shadow stack at the same level (the event used an IST slot). Returning to CPL 3 restores
    /// the user SSP from PL3_SSP.
    pub(super) fn shadow_stack_return(
        &mut self,
        new_cpl: u8,
        new_cs: &Seg,
        new_rip: u64,
        iret: bool,
    ) -> ExecResult<u64> {
        let old_cpl = self.cpl;
        let long_mode = self.efer & EFER_LMA != 0;
        let new_long = long_mode && new_cs.long();
        let mut ssp = self.cet.ssp;
        let mut new_ssp = ssp;
        let mut release_token = false;
        if self.shadow_stack_active() {
            let user = old_cpl == USER_CPL;
            if ssp & SHADOW_STACK_ALIGNMENT_MASK != 0 {
                return Err(control_protection(CP_FAR_RET_IRET));
            }
            if new_cpl == old_cpl || new_cpl != USER_CPL {
                let previous = self.shadow_read(ssp, 8, user)?;
                let lip = self.shadow_read(ssp.wrapping_add(FRAME_LIP_OFFSET), 8, user)?;
                let cs = self.shadow_read(ssp.wrapping_add(FRAME_CS_OFFSET), 8, user)?;
                ssp = ssp.wrapping_add(FRAME_BYTES);
                let mut expected_lip = new_cs.base.wrapping_add(new_rip);
                if !new_long {
                    expected_lip &= LOW_32_BITS;
                }
                if cs != new_cs.sel as u64 || lip != expected_lip || previous & TOKEN_FLAG_MASK != 0
                {
                    return Err(control_protection(CP_FAR_RET_IRET));
                }
                if !new_long && previous >> 32 != 0 {
                    return Err(Exception::general_protection(0));
                }
                new_ssp = previous;
            }
            release_token = new_cpl > old_cpl || (iret && long_mode && ssp != new_ssp);
        }
        if new_cpl == USER_CPL && old_cpl != USER_CPL && self.shadow_stack_enabled_at(USER_CPL) {
            new_ssp = self.cet.pl_ssp[USER_CPL as usize];
            if !new_long && new_ssp >> 32 != 0 {
                return Err(Exception::general_protection(0));
            }
        }
        if release_token {
            self.release_supervisor_token(ssp)?;
        }
        Ok(new_ssp)
    }

    /// iret from ring 0 into virtual-8086 mode, which has no shadow stack: like any return to an
    /// outer level it gives up the supervisor shadow stack, releasing its token.
    pub(super) fn shadow_stack_leave_for_virtual_8086(&mut self) -> ExecResult<()> {
        if !self.shadow_stack_active() {
            return Ok(());
        }
        let ssp = self.cet.ssp;
        if ssp & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            return Err(control_protection(CP_FAR_RET_IRET));
        }
        self.release_supervisor_token(ssp)
    }

    /// SYSCALL and SYSENTER, before CPL becomes 0: parks the user SSP in PL3_SSP. Afterwards
    /// `shadow_stack_after_fast_entry` clears SSP when the kernel has a shadow stack; the kernel
    /// must claim one itself (setssbsy), as nothing about these entries reads a token.
    pub(super) fn shadow_stack_before_fast_entry(&mut self) {
        if self.cpl == USER_CPL && self.shadow_stack_active() {
            self.cet.pl_ssp[USER_CPL as usize] = self.cet.ssp;
        }
    }

    pub(super) fn shadow_stack_after_fast_entry(&mut self) {
        if self.shadow_stack_active() {
            self.cet.ssp = 0;
        }
    }

    /// SYSRET and SYSEXIT: the SSP CPL 3 gets back, checked before anything changes (`None`:
    /// leave SSP alone). `to_long`: the return is to 64-bit code; compatibility-mode and legacy
    /// code need an SSP below 4 GiB, #GP(0) otherwise.
    pub(super) fn shadow_stack_fast_return(&self, to_long: bool) -> ExecResult<Option<u64>> {
        if !self.shadow_stack_enabled_at(USER_CPL) {
            return Ok(None);
        }
        let ssp = self.cet.pl_ssp[USER_CPL as usize];
        if !to_long && ssp >> 32 != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(Some(ssp))
    }

    fn need_shadow_stack(&self) -> ExecResult<()> {
        if self.shadow_stack_active() {
            Ok(())
        } else {
            Err(Exception::invalid_opcode())
        }
    }

    /// F3 0F 1E /1 (register form): rdsspd/rdsspq. Without a shadow stack it is the hint nop it
    /// encodes on older CPUs, so code can probe for SSP with it.
    pub(super) fn rdssp(&mut self, register: u8, size: u8) {
        if self.shadow_stack_active() {
            let ssp = self.cet.ssp;
            self.write_gpr(register, size, true, ssp);
        }
    }

    /// F3 0F AE /5 (register form): incsspd/incsspq. Pops `count` (the register's low byte)
    /// entries of `size` bytes without looking at them, except that the first and last are read so
    /// a count running off the shadow stack faults.
    pub(super) fn incssp(&mut self, register: u8, size: u8) -> ExecResult<()> {
        self.need_shadow_stack()?;
        let count = self.gpr[register as usize & 0xf] & INCSSP_COUNT_MASK;
        let user = self.cpl == USER_CPL;
        let ssp = self.cet.ssp;
        self.shadow_read(ssp, size, user)?;
        let last = ssp.wrapping_add(count.max(1).wrapping_sub(1) * size as u64);
        self.shadow_read(last, size, user)?;
        self.cet.ssp = ssp.wrapping_add(count * size as u64);
        Ok(())
    }

    /// 0F 38 F6 (memory form): wrssd/wrssq, a store to the current privilege's shadow stack, if
    /// WR_SHSTK_EN allows it.
    pub(super) fn wrss(&mut self, address: u64, size: u8, value: u64) -> ExecResult<()> {
        self.need_shadow_stack()?;
        let control = if self.cpl == USER_CPL {
            self.cet.u_cet
        } else {
            self.cet.s_cet
        };
        if control & CET_WR_SHSTK_EN == 0 {
            return Err(Exception::invalid_opcode());
        }
        if address & (size as u64 - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        self.shadow_write(address, size, value, self.cpl == USER_CPL)
    }

    /// 66 0F 38 F5 (memory form): wrussd/wrussq, the kernel's store to a user shadow stack. Needs
    /// CR4.CET but no SH_STK_EN, which is how Linux writes user shadow stacks (signal frames,
    /// restore tokens) without one of its own.
    pub(super) fn wruss(&mut self, address: u64, size: u8, value: u64) -> ExecResult<()> {
        if self.cr4 & CR4_CET == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        if address & (size as u64 - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        self.shadow_write(address, size, value, true)
    }

    /// wrss (`user` false) and wruss with their memory operand; the register form is #UD. REX.W
    /// selects the 8-byte form (a 66 prefix is part of wruss's encoding, not an operand size).
    pub(super) fn write_shadow_stack_operand(
        &mut self,
        i: &Insn,
        next: u64,
        user: bool,
    ) -> ExecResult<()> {
        if i.md == 3 {
            return Err(Exception::invalid_opcode());
        }
        let size = if i.opsize == 8 { 8 } else { 4 };
        let address = self.linear_address(i, self.effective_address(i, next))?;
        let value = self.read_gpr(i.reg, size, true);
        if user {
            self.wruss(address, size, value)
        } else {
            self.wrss(address, size, value)
        }
    }

    /// The checks setssbsy and clrssbsy share: CR4.CET and S_CET.SH_STK_EN (#UD), CPL 0 (#GP).
    fn supervisor_token_instruction(&self) -> ExecResult<()> {
        if self.cr4 & CR4_CET == 0 || self.cet.s_cet & CET_SH_STK_EN == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()
    }

    /// F3 0F 01 E8: setssbsy. Claims the token at PL0_SSP and switches to that shadow stack; a
    /// busy or foreign token is #CP(setssbsy). Xen does this on every entry through SYSCALL.
    pub(super) fn setssbsy(&mut self) -> ExecResult<()> {
        self.supervisor_token_instruction()?;
        let ssp = self.cet.pl_ssp[0];
        if ssp & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            return Err(Exception::general_protection(0));
        }
        if self.shadow_token_exchange(ssp, ssp, ssp | TOKEN_BUSY)? != ssp {
            return Err(control_protection(CP_SETSSBSY));
        }
        self.cet.ssp = ssp;
        Ok(())
    }

    /// F3 0F AE /6 (memory form): clrssbsy. Releases the busy token at `address` and leaves SSP
    /// 0; CF says whether the token was not the busy token of that address (nothing changes then).
    pub(super) fn clrssbsy(&mut self, address: u64) -> ExecResult<()> {
        self.supervisor_token_instruction()?;
        if address & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            return Err(Exception::general_protection(0));
        }
        let busy = address | TOKEN_BUSY;
        let invalid = self.shadow_token_exchange(address, busy, address)? != busy;
        self.set_eflags_arith(if invalid { CF } else { 0 });
        self.cet.ssp = 0;
        Ok(())
    }

    /// F3 0F 01 /5 (memory form): rstorssp. Switches to the shadow stack whose restore token is
    /// at `address` (8-byte aligned): the token must name the address right above itself and the
    /// current mode, else #CP(rstorssp). The token is replaced by a "previous SSP" token holding
    /// the SSP being left, for saveprevssp to move to the old stack. CF says the restored stack
    /// was only 4-byte aligned.
    pub(super) fn rstorssp(&mut self, address: u64) -> ExecResult<()> {
        self.need_shadow_stack()?;
        if address & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            return Err(Exception::general_protection(0));
        }
        let mode = self.is_long64() as u64;
        let user = self.cpl == USER_CPL;
        let pa = self.shadow_stack_translate(address, user, true)?;
        let previous_token = self.cet.ssp | mode | TOKEN_PREVIOUS_SSP;
        let token = self.phys_read(pa, 8);
        let points_here =
            (token & !TOKEN_MODE_64).wrapping_sub(8) & !SHADOW_STACK_ALIGNMENT_MASK == address;
        let valid =
            token & TOKEN_FLAG_MASK == mode && (mode != 0 || token >> 32 == 0) && points_here;
        if !valid {
            return Err(control_protection(CP_RSTORSSP));
        }
        if !self.ram.contains(pa, 8) || !self.update_ram_u64(pa, token, previous_token) {
            // Another vCPU changed the token under us: what hardware's locked access rules out.
            return Err(control_protection(CP_RSTORSSP));
        }
        let page = (pa >> PAGE_SHIFT) as usize;
        if self.page_has_code.get(page) != 0 {
            self.code_written(page);
        }
        self.cet.ssp = address;
        let misaligned_restore = token & SHADOW_STACK_HALF_ALIGNMENT != 0;
        self.set_eflags_arith(if misaligned_restore { CF } else { 0 });
        Ok(())
    }

    /// F3 0F 01 EA: saveprevssp. Pops the "previous SSP" token rstorssp left (with CF set, after
    /// the zero alignment dword below it) and writes a restore token for that SSP onto the old
    /// shadow stack, so the old stack can be switched back to later.
    pub(super) fn saveprevssp(&mut self) -> ExecResult<()> {
        self.need_shadow_stack()?;
        let mut ssp = self.cet.ssp;
        if ssp & SHADOW_STACK_ALIGNMENT_MASK != 0 {
            return Err(Exception::general_protection(0));
        }
        let long = self.is_long64();
        let user = self.cpl == USER_CPL;
        let previous = self.shadow_pop(&mut ssp, 8, user)?;
        if self.eflags_full() & CF != 0 {
            if long {
                return Err(Exception::general_protection(0));
            }
            if self.shadow_pop(&mut ssp, 4, user)? != 0 {
                return Err(Exception::general_protection(0));
            }
        }
        if previous & TOKEN_PREVIOUS_SSP == 0 || (!long && previous >> 32 != 0) {
            return Err(Exception::general_protection(0));
        }
        let old_ssp = previous & !TOKEN_FLAG_MASK;
        let restore_token = old_ssp | long as u64;
        // The restore token goes to the next 8-byte boundary below the old SSP; a 4-byte-aligned
        // one gets a zero dword above it.
        self.shadow_write(
            old_ssp.wrapping_sub(SHADOW_STACK_HALF_ALIGNMENT),
            4,
            0,
            user,
        )?;
        let token_address = (old_ssp & !SHADOW_STACK_ALIGNMENT_MASK).wrapping_sub(8);
        self.shadow_write(token_address, 8, restore_token, user)?;
        self.cet.ssp = ssp;
        Ok(())
    }
}

/// See `Cpu::shadow_stack_enter`.
pub(super) struct ShadowStackSwitch {
    ssp: u64,
    /// The user SSP to park in PL3_SSP.
    pl3_ssp: Option<u64>,
}

impl ShadowStackSwitch {
    pub(super) fn commit(self, cpu: &mut Cpu) {
        if let Some(user_ssp) = self.pl3_ssp {
            cpu.cet.pl_ssp[USER_CPL as usize] = user_ssp;
        }
        cpu.cet.ssp = self.ssp;
    }
}

fn mask_for(size: u8) -> u64 {
    if size == 8 { u64::MAX } else { LOW_32_BITS }
}
