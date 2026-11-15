//! Virtual-8086 mode extensions (CR4.VME) and protected-mode virtual interrupts (CR4.PVI), AMD
//! APM Vol. 2 "Virtual-8086 Mode Extensions" and SDM Vol. 3 chapter 20.3. With IOPL below 3 the
//! IOPL-sensitive instructions of a virtual-8086 task work on EFLAGS.VIF instead of faulting, so a
//! monitor sees a #GP only when it has something to do: sti or popf/iret setting the virtual IF
//! while VIP says an interrupt is pending, or an `int n` the TSS bitmap sends to it. PVI gives
//! CPL 3 protected-mode code the same cli/sti.
//!
//! Hardware interrupts are not affected: IF still gates them, VIF is only bookkeeping for the
//! monitor (which sets VIP when it holds one back).

use super::exception::Exception;
use super::flags::*;
use super::modes::real_mode_segment;
use super::*;

/// CR4 bit 0: virtual-8086 mode extensions.
pub(crate) const CR4_VME: u64 = 1 << 0;
/// CR4 bit 1: protected-mode virtual interrupts.
pub(crate) const CR4_PVI: u64 = 1 << 1;

/// The software-interrupt redirection bitmap: 256 bits right below the I/O permission bitmap of
/// the 32-bit TSS, whose base is the word at this offset. A clear bit sends `int n` through the
/// task's own real-mode vector table.
const TSS32_IO_MAP_BASE_OFFSET: u64 = 0x66;
const REDIRECTION_BITMAP_BYTES: u64 = 32;
const IVT_ENTRY_BYTES: u64 = 4;
const WORD_MASK: u64 = 0xffff;

impl Cpu {
    /// Whether an IOPL-sensitive cli/sti at CPL > IOPL works on VIF: virtual-8086 mode with VME,
    /// or CPL 3 protected-mode code with PVI.
    fn virtual_interrupt_flag_in_use(&self) -> bool {
        if self.eflags & VM != 0 {
            self.cr4 & CR4_VME != 0
        } else {
            self.cr0 & CR0_PE != 0 && self.cpl == USER_CPL && self.cr4 & CR4_PVI != 0
        }
    }

    /// cli (`set` false) and sti at CPL > IOPL: clear or set VIF where VME/PVI allow it, else
    /// #GP(0). sti with VIP set is #GP(0) too: the monitor then delivers the pending interrupt.
    pub(super) fn virtual_cli_sti(&mut self, set: bool) -> ExecResult<()> {
        if !self.virtual_interrupt_flag_in_use() || (set && self.eflags & VIP != 0) {
            return Err(Exception::general_protection(0));
        }
        if set {
            self.eflags |= VIF;
        } else {
            self.eflags &= !VIF;
        }
        Ok(())
    }

    /// Whether VME applies to an IOPL-sensitive instruction of `size` bytes here: only the
    /// 16-bit forms of pushf/popf/iret get the virtual treatment, the 32-bit ones still fault.
    fn vme_applies(&self, size: u8) -> bool {
        self.eflags & VM != 0 && self.iopl() < USER_CPL && self.cr4 & CR4_VME != 0 && size == 2
    }

    /// pushf in virtual-8086 mode below IOPL 3: with VME the 16-bit form pushes FLAGS with VIF in
    /// the IF position and IOPL reading 3, as an 8086 program running at IOPL 3 would see it.
    pub(super) fn pushf_virtual_8086(&mut self, size: u8) -> ExecResult<()> {
        if !self.vme_applies(size) {
            return Err(Exception::general_protection(0));
        }
        let flags = self.eflags_full();
        let image = (flags & !(IF | IOPL)) | IOPL | if flags & VIF != 0 { IF } else { 0 };
        self.push(2, image & WORD_MASK)
    }

    /// The virtual-IF part of a 16-bit popf or iret under VME: #GP(0) for an image that sets TF,
    /// or IF while VIP is pending; otherwise the image's low word lands in FLAGS except IOPL and
    /// IF, and its IF goes to VIF. Nothing is committed on a fault.
    fn load_virtual_flags(&mut self, image: u64) -> ExecResult<()> {
        if image & TF != 0 || (image & IF != 0 && self.eflags & VIP != 0) {
            return Err(Exception::general_protection(0));
        }
        let loadable = WORD_MASK & !(IOPL | IF);
        let current = self.eflags_full();
        let mut flags = (current & !loadable) | (image & loadable) | RESERVED_ONE;
        flags = if image & IF != 0 {
            flags | VIF
        } else {
            flags & !VIF
        };
        self.eflags = flags;
        self.lazy_op = LazyFlagsOp::Eflags;
        Ok(())
    }

    /// popf in virtual-8086 mode below IOPL 3 (see `load_virtual_flags`).
    pub(super) fn popf_virtual_8086(&mut self, size: u8) -> ExecResult<()> {
        if !self.vme_applies(size) {
            return Err(Exception::general_protection(0));
        }
        let sp = self.stack_pointer();
        let image = self.read_stack(self.stack_linear(sp)?, 2)?;
        self.load_virtual_flags(image)?;
        self.set_stack_pointer(sp.wrapping_add(2));
        Ok(())
    }

    /// iret in virtual-8086 mode below IOPL 3: the 16-bit form under VME pops IP, CS and FLAGS
    /// like an 8086 iret, with FLAGS loaded as popf loads it.
    pub(super) fn iret_virtual_8086(&mut self, size: u8) -> ExecResult<()> {
        if !self.vme_applies(size) {
            return Err(Exception::general_protection(0));
        }
        let sp = self.stack_pointer();
        let mask = self.stack_mask();
        let pop = |cpu: &mut Cpu, index: u64| -> ExecResult<u64> {
            let linear = cpu.stack_linear(sp.wrapping_add(2 * index) & mask)?;
            cpu.read_stack(linear, 2)
        };
        let ip = pop(self, 0)?;
        let cs = pop(self, 1)? as u16;
        let image = pop(self, 2)?;
        self.load_virtual_flags(image)?;
        self.seg[SEG_CS] = real_mode_segment(cs, USER_CPL);
        self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
        self.rip = ip;
        self.set_stack_pointer(sp.wrapping_add(6));
        Ok(())
    }

    /// `int n` in virtual-8086 mode with VME: when the TSS redirection bitmap has bit `vector`
    /// clear, the interrupt goes through the task's real-mode vector table at linear 0, like an
    /// 8086 `int`, and `Ok(true)` says it is done (with rIP at the handler). Below IOPL 3 the
    /// pushed FLAGS carry VIF as IF and IOPL 3, and VIF instead of IF is cleared. `Ok(false)`: the
    /// bit is set, and the interrupt takes the protected-mode path (a #GP below IOPL 3).
    /// `next` is the return address.
    pub(super) fn redirect_software_interrupt(
        &mut self,
        vector: u8,
        next: u64,
    ) -> ExecResult<bool> {
        if self.eflags & VM == 0 || self.cr4 & CR4_VME == 0 {
            return Ok(false);
        }
        let limit = self.tr.limit as u64;
        if limit < TSS32_IO_MAP_BASE_OFFSET + 1 {
            return Err(Exception::general_protection(0));
        }
        let map_base = self.read_sys(self.tr.base.wrapping_add(TSS32_IO_MAP_BASE_OFFSET), 2)?;
        let byte_offset = map_base
            .wrapping_sub(REDIRECTION_BITMAP_BYTES)
            .wrapping_add(vector as u64 / 8);
        if map_base < REDIRECTION_BITMAP_BYTES || byte_offset > limit {
            return Err(Exception::general_protection(0));
        }
        let byte = self.read_sys(self.tr.base.wrapping_add(byte_offset), 1)?;
        if byte & (1 << (vector % 8)) != 0 {
            return Ok(false);
        }
        let entry = self.read(vector as u64 * IVT_ENTRY_BYTES, 4)?;
        let flags = self.eflags_full();
        let virtual_if = self.iopl() < USER_CPL;
        let image = if virtual_if {
            (flags & !(IF | IOPL)) | IOPL | if flags & VIF != 0 { IF } else { 0 }
        } else {
            flags
        };
        let saved_sp = self.gpr[GPR_RSP];
        let old_cs = self.seg[SEG_CS].sel as u64;
        let pushed = self
            .push(2, image & WORD_MASK)
            .and_then(|()| self.push(2, old_cs))
            .and_then(|()| self.push(2, next & WORD_MASK));
        if let Err(error) = pushed {
            self.gpr[GPR_RSP] = saved_sp;
            return Err(error);
        }
        self.materialize_lazy_flags();
        self.eflags &= !(TF | if virtual_if { VIF } else { IF });
        self.seg[SEG_CS] = real_mode_segment((entry >> 16) as u16, USER_CPL);
        self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
        self.rip = entry & WORD_MASK;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exception::VECTOR_GENERAL_PROTECTION;
    use crate::cpu::legacy_support::{install_legacy_tss, protected32};
    use crate::testkit::{Harness, Stop};

    /// The task's code and stack segments, and where its vector table sends `int 0x21`.
    const V86_CODE_SEGMENT: u16 = 0x2000;
    const V86_STACK_SEGMENT: u16 = 0x3000;
    const V86_STACK_POINTER: u64 = 0x1000;
    const HANDLER_SEGMENT: u16 = 0x4000;
    const HANDLER_OFFSET: u16 = 0x10;
    const LEGACY_TSS_BASE: u64 = crate::cpu::legacy_support::LEGACY_TSS_BASE;
    /// A TSS long enough for the redirection bitmap and an I/O map base right after it.
    const TSS_WITH_BITMAP_LIMIT: u32 = 0x67 + 32 + 2;
    const IO_MAP_BASE: u16 = 0x68 + 32;

    fn gp() -> Stop {
        Stop::Fault {
            vector: VECTOR_GENERAL_PROTECTION,
            error_code: Some(0),
            software: false,
        }
    }

    fn stack_linear(sp: u64) -> u64 {
        ((V86_STACK_SEGMENT as u64) << 4) + sp
    }

    /// A virtual-8086 task at IOPL 0 running `code` at 2000:0000, with VME on.
    fn v86(code: &[u8]) -> Harness {
        let mut h = protected32(&[HLT_BYTE]);
        h.write((V86_CODE_SEGMENT as u64) << 4, code);
        h.cpu.cr4 |= CR4_VME;
        h.cpu.eflags = RESERVED_ONE | VM | IF;
        h.cpu.seg[SEG_CS] = real_mode_segment(V86_CODE_SEGMENT, USER_CPL);
        h.cpu.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
        for index in [SEG_SS, SEG_DS, SEG_ES, SEG_FS, SEG_GS] {
            h.cpu.seg[index] = real_mode_segment(V86_STACK_SEGMENT, USER_CPL);
        }
        h.cpu.set_cpl(USER_CPL);
        h.cpu.rip = 0;
        h.cpu.gpr[GPR_RSP] = V86_STACK_POINTER;
        h
    }

    const HLT_BYTE: u8 = 0xf4;

    /// Runs until the instruction before the task's final hlt is done: v86 hlt is #GP(0).
    fn run_to_hlt(h: &mut Harness) {
        assert_eq!(h.run(100), gp());
        let byte = h.read(h.cpu.seg[SEG_CS].base + h.cpu.rip, 1)[0];
        assert_eq!(byte, HLT_BYTE, "stopped at {:#x}", h.cpu.rip);
    }

    #[test]
    fn cli_and_sti_work_on_vif_below_iopl_3() {
        // cli; hlt / sti; hlt
        let mut h = v86(&[0xfa, HLT_BYTE]);
        h.cpu.eflags |= VIF;
        run_to_hlt(&mut h);
        assert_eq!(h.cpu.eflags & (VIF | IF), IF, "VIF cleared, IF untouched");
        let mut h = v86(&[0xfb, HLT_BYTE]);
        run_to_hlt(&mut h);
        assert_ne!(h.cpu.eflags & VIF, 0);
    }

    #[test]
    fn sti_with_a_pending_virtual_interrupt_faults() {
        let mut h = v86(&[0xfb, HLT_BYTE]);
        h.cpu.eflags |= VIP;
        assert_eq!(h.run(10), gp());
        assert_eq!(h.cpu.rip, 0);
        assert_eq!(h.cpu.eflags & VIF, 0);
    }

    #[test]
    fn without_vme_cli_faults() {
        let mut h = v86(&[0xfa, HLT_BYTE]);
        h.cpu.cr4 &= !CR4_VME;
        assert_eq!(h.run(10), gp());
        assert_eq!(h.cpu.rip, 0);
    }

    #[test]
    fn pushf_shows_vif_as_if_and_iopl_3_and_popf_loads_vif() {
        // pushf; pop ax; push bx; popf; hlt
        let mut h = v86(&[0x9c, 0x58, 0x53, 0x9d, HLT_BYTE]);
        h.cpu.eflags = (h.cpu.eflags & !IF) | VIF;
        h.cpu.gpr[3] = IF | CF;
        run_to_hlt(&mut h);
        let pushed = h.cpu.gpr[0] & WORD_MASK;
        assert_eq!(pushed & (IF | IOPL), IF | IOPL);
        assert_eq!(
            h.cpu.eflags & (VIF | IF | CF),
            VIF | CF,
            "IF itself unchanged"
        );
        assert_eq!(h.cpu.iopl(), 0);
    }

    #[test]
    fn popf_of_tf_or_of_if_with_vip_faults_and_32_bit_forms_still_fault() {
        // push bx; popf; hlt
        for (image, vip) in [(TF, false), (IF, true)] {
            let mut h = v86(&[0x53, 0x9d, HLT_BYTE]);
            if vip {
                h.cpu.eflags |= VIP;
            }
            h.cpu.gpr[3] = image;
            assert_eq!(h.run(10), gp(), "{image:#x}");
            assert_eq!(
                h.cpu.gpr[GPR_RSP],
                V86_STACK_POINTER - 2,
                "the popf is undone"
            );
        }
        // pushfd
        let mut h = v86(&[0x66, 0x9c, HLT_BYTE]);
        assert_eq!(h.run(10), gp());
        assert_eq!(h.cpu.rip, 0);
    }

    #[test]
    fn iret_pops_ip_cs_and_virtual_flags() {
        // iret
        let mut h = v86(&[0xcf]);
        let frame: [u16; 3] = [0x0002, V86_CODE_SEGMENT, (IF | CF) as u16];
        for (k, word) in frame.iter().enumerate() {
            h.write(
                stack_linear(V86_STACK_POINTER) + 2 * k as u64,
                &word.to_le_bytes(),
            );
        }
        h.write(((V86_CODE_SEGMENT as u64) << 4) + 2, &[HLT_BYTE]);
        run_to_hlt(&mut h);
        assert_eq!(h.cpu.rip, 2);
        assert_eq!(h.cpu.gpr[GPR_RSP], V86_STACK_POINTER + 6);
        assert_eq!(h.cpu.eflags & (VIF | CF), VIF | CF);
    }

    fn with_redirection_bitmap(h: &mut Harness, redirected_to_monitor: &[u8]) {
        install_legacy_tss(h, 0x8000, crate::cpu::legacy_support::LEGACY_DS);
        h.cpu.tr.limit = TSS_WITH_BITMAP_LIMIT;
        h.write(
            LEGACY_TSS_BASE + TSS32_IO_MAP_BASE_OFFSET,
            &IO_MAP_BASE.to_le_bytes(),
        );
        let mut bitmap = [0u8; 32];
        for &vector in redirected_to_monitor {
            bitmap[vector as usize / 8] |= 1 << (vector % 8);
        }
        h.write(LEGACY_TSS_BASE + IO_MAP_BASE as u64 - 32, &bitmap);
    }

    #[test]
    fn int_with_a_clear_bitmap_bit_goes_through_the_tasks_vector_table() {
        // int 0x21 at 2000:0000; the handler at 4000:0010 is a hlt
        let mut h = v86(&[0xcd, 0x21]);
        with_redirection_bitmap(&mut h, &[0x13]);
        h.write(
            0x21 * 4,
            &((HANDLER_SEGMENT as u32) << 16 | HANDLER_OFFSET as u32).to_le_bytes(),
        );
        h.write(
            ((HANDLER_SEGMENT as u64) << 4) + HANDLER_OFFSET as u64,
            &[HLT_BYTE],
        );
        h.cpu.eflags |= VIF;
        run_to_hlt(&mut h);
        assert_eq!(h.cpu.seg[SEG_CS].sel, HANDLER_SEGMENT);
        assert_eq!(h.cpu.rip, HANDLER_OFFSET as u64);
        let sp = V86_STACK_POINTER - 6;
        assert_eq!(h.cpu.gpr[GPR_RSP], sp);
        let word =
            |k: u64| u16::from_le_bytes(h.read(stack_linear(sp) + 2 * k, 2).try_into().unwrap());
        assert_eq!(word(0), 2, "return IP");
        assert_eq!(word(1), V86_CODE_SEGMENT);
        assert_eq!(word(2) as u64 & (IF | IOPL), IF | IOPL, "VIF as IF, IOPL 3");
        assert_eq!(h.cpu.eflags & (VIF | IF), IF, "VIF cleared, IF kept");
    }

    #[test]
    fn int_with_a_set_bitmap_bit_below_iopl_3_faults() {
        let mut h = v86(&[0xcd, 0x13]);
        with_redirection_bitmap(&mut h, &[0x13]);
        // Not redirected: the protected-mode path, whose delivery refuses it below IOPL 3.
        assert_eq!(
            h.run(10),
            Stop::Fault {
                vector: 0x13,
                error_code: None,
                software: true
            }
        );
        assert_eq!(
            crate::cpu::test_support::fault_of(h.cpu.deliver_legacy(0x13, None, true)),
            (VECTOR_GENERAL_PROTECTION, Some(0))
        );
    }

    #[test]
    fn pvi_gives_ring_3_protected_mode_a_virtual_if() {
        // cli; hlt
        let mut h = protected32(&[0xfa, HLT_BYTE]);
        crate::cpu::legacy_support::enter_ring3(&mut h, 0x9000);
        h.cpu.eflags |= IF | VIF;
        assert_eq!(h.run(10), gp(), "without PVI");
        assert_eq!(h.cpu.eflags & VIF, VIF);
        h.cpu.cr4 |= CR4_PVI;
        assert_eq!(h.run(10), gp(), "the hlt");
        assert_eq!(h.cpu.eflags & (VIF | IF), IF);
    }
}
