//! Control transfers outside 64-bit mode: interrupt and exception delivery (real-mode vector
//! table, 16/32-bit gates, virtual-8086 entry and exit), far jumps, calls and returns through
//! code segments, call gates and task gates, iret, and 32-bit task switches.
//!
//! Delivery here is used whenever EFER.LMA is clear. With LMA set (even compatibility mode)
//! interrupts still go through the 16-byte gates of `exception.rs`.

use super::decode::Insn;
use super::exception::{
    Exception, VECTOR_INVALID_TSS, VECTOR_SEGMENT_NOT_PRESENT, VECTOR_STACK_FAULT,
};
use super::exec::Loc;
use super::flags::*;
use super::legacy::selector_is_null;
use super::modes::real_mode_segment;
use super::*;

const SELECTOR_RPL_MASK: u16 = 3;
const SELECTOR_ERROR_CODE_MASK: u16 = 0xfffc;
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
const SELECTOR_TI_LDT: u16 = 1 << 2;
const WORD_MASK: u64 = 0xffff;
const DWORD_MASK: u64 = 0xffff_ffff;

/// Real-mode interrupt vector table entries are 4 bytes: offset then segment.
const IVT_ENTRY_BYTES: u64 = 4;
/// Protected-mode IDT entries are 8 bytes.
const IDT_ENTRY_BYTES: u64 = 8;
const IDT_ERROR_CODE_FLAG: u32 = 2;

/// Descriptor fields, as bit positions in the descriptor's low qword.
const DESC_TYPE_SHIFT: u32 = 40;
const DESC_TYPE_MASK: u64 = 0xf;
const DESC_S_BIT: u64 = 1 << 44;
const DESC_DPL_SHIFT: u32 = 45;
const DESC_PRESENT_BIT: u64 = 1 << 47;
const DESC_ACCESSED_BIT: u64 = 1 << 40;
/// Gate descriptors: target selector in bits 16-31, parameter count in 32-36.
const GATE_SELECTOR_SHIFT: u32 = 16;
const GATE_PARAM_COUNT_MASK: u64 = 0x1f;

const TYPE_TSS16_AVAILABLE: u64 = 1;
const TYPE_CALL_GATE16: u64 = 4;
const TYPE_TASK_GATE: u64 = 5;
const TYPE_INTERRUPT_GATE16: u64 = 6;
const TYPE_TRAP_GATE16: u64 = 7;
const TYPE_TSS32_AVAILABLE: u64 = 9;
const TYPE_CALL_GATE32: u64 = 0xc;
const TYPE_INTERRUPT_GATE32: u64 = 0xe;
const TYPE_TRAP_GATE32: u64 = 0xf;

const TSS32_STACK_SLOT_BYTES: u64 = 8;
const TSS32_ESP0: u64 = 4;
/// TSS16 stack slots (SP0 at 2, SS0 at 4, 4 bytes per privilege level).
const TSS16_SP0: u64 = 2;
const TSS16_STACK_SLOT_BYTES: u64 = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Transfer {
    Jump,
    Call,
}

fn descriptor_type(lo: u64) -> u64 {
    (lo >> DESC_TYPE_SHIFT) & DESC_TYPE_MASK
}

fn descriptor_dpl(lo: u64) -> u8 {
    ((lo >> DESC_DPL_SHIFT) & 3) as u8
}

fn is_system(lo: u64) -> bool {
    lo & DESC_S_BIT == 0
}

fn gate_offset(lo: u64, wide: bool) -> u64 {
    let low = lo & WORD_MASK;
    if wide {
        low | ((lo >> 32) & 0xffff_0000)
    } else {
        low
    }
}

fn selector_fault(vector: u8, selector: u16) -> Exception {
    Exception::fault(vector, Some((selector & SELECTOR_ERROR_CODE_MASK) as u32))
}

impl Cpu {
    /// Pushes on the current stack as an implicit supervisor access (interrupt delivery).
    fn push_implicit(&mut self, sp: &mut u64, size: u8, value: u64) -> ExecResult<()> {
        let mask = self.stack_mask();
        *sp = sp.wrapping_sub(size as u64) & mask;
        let linear = self.stack_linear(*sp)?;
        self.write_sys(linear, size, value)
    }

    /// Pushes onto a stack segment other than the current one (a privilege-changing transfer).
    fn push_on(&mut self, stack: &Seg, sp: &mut u64, size: u8, value: u64) -> ExecResult<()> {
        let mask = if stack.flags & SEG_DEFAULT_BIG != 0 {
            DWORD_MASK
        } else {
            WORD_MASK
        };
        *sp = sp.wrapping_sub(size as u64) & mask;
        if *sp + size as u64 - 1 > stack.limit as u64 {
            return Err(selector_fault(VECTOR_STACK_FAULT, stack.sel));
        }
        let linear = stack.base.wrapping_add(*sp) & DWORD_MASK;
        self.write_sys(linear, size, value)
    }

    /// Reads the stack pointer and segment for privilege level `cpl` from the TSS.
    fn tss_stack(&mut self, cpl: u8) -> ExecResult<(u64, u16, bool)> {
        let wide = self.tr.descriptor_type() >= TYPE_TSS32_AVAILABLE;
        let (slot, size, offset) = if wide {
            (TSS32_STACK_SLOT_BYTES, 4u8, TSS32_ESP0)
        } else {
            (TSS16_STACK_SLOT_BYTES, 2u8, TSS16_SP0)
        };
        let base = offset + cpl as u64 * slot;
        if base + size as u64 + 1 > self.tr.limit as u64 {
            return Err(selector_fault(VECTOR_INVALID_TSS, self.tr.sel));
        }
        let sp = self.read_sys(self.tr.base.wrapping_add(base), size)?;
        let ss = self.read_sys(self.tr.base.wrapping_add(base + size as u64), 2)? as u16;
        Ok((sp, ss, wide))
    }

    /// Validates `selector` as a stack segment for privilege level `cpl` (inner stack of a
    /// transfer): #TS on any violation.
    fn checked_stack_segment(&mut self, selector: u16, cpl: u8) -> ExecResult<Seg> {
        if selector_is_null(selector) {
            return Err(selector_fault(VECTOR_INVALID_TSS, selector));
        }
        let invalid = selector_fault(VECTOR_INVALID_TSS, selector);
        let (lo, _) = self
            .read_desc(selector)
            .map_err(|_| selector_fault(VECTOR_INVALID_TSS, selector))?;
        let seg = self.load_desc_seg(selector, false)?;
        if (selector & SELECTOR_RPL_MASK) as u8 != cpl
            || is_system(lo)
            || seg.is_code()
            || seg.flags & SEG_TYPE_READ_WRITE == 0
            || descriptor_dpl(lo) != cpl
        {
            return Err(invalid);
        }
        if !seg.present() {
            return Err(selector_fault(VECTOR_STACK_FAULT, selector));
        }
        Ok(seg)
    }

    /// Sets the accessed bit of a code or data descriptor, as loading it into a segment register does.
    fn mark_descriptor_accessed(&mut self, selector: u16) -> ExecResult<()> {
        let (table_base, _) = if selector & SELECTOR_TI_LDT != 0 {
            (self.ldtr.base, self.ldtr.limit)
        } else {
            (self.gdtr.base, self.gdtr.limit as u32)
        };
        let address = table_base.wrapping_add((selector & SELECTOR_OFFSET_MASK) as u64);
        let (lo, _) = self.read_desc(selector)?;
        if lo & DESC_ACCESSED_BIT == 0 {
            self.write_sys(address, 8, lo | DESC_ACCESSED_BIT)?;
        }
        Ok(())
    }

    /// Delivers `vector` through the real-mode vector table or the protected-mode IDT.
    pub(super) fn deliver_legacy(
        &mut self,
        vector: u8,
        error_code: Option<u32>,
        software: bool,
    ) -> ExecResult<()> {
        if self.cr0 & CR0_PE == 0 {
            return self.deliver_real(vector);
        }
        let entry_offset = vector as u64 * IDT_ENTRY_BYTES;
        let idt_error = vector as u32 * 8 + IDT_ERROR_CODE_FLAG;
        if entry_offset + (IDT_ENTRY_BYTES - 1) > self.idtr.limit as u64 {
            return Err(Exception::general_protection(idt_error));
        }
        let lo = self.read_sys(self.idtr.base.wrapping_add(entry_offset), 8)?;
        let gate_type = descriptor_type(lo);
        let dpl = descriptor_dpl(lo);
        // INT n in virtual-8086 mode needs IOPL 3 (VME redirection is not implemented).
        if software && self.eflags & VM != 0 && self.iopl() < USER_CPL {
            return Err(Exception::general_protection(0));
        }
        if !matches!(
            gate_type,
            TYPE_TASK_GATE
                | TYPE_INTERRUPT_GATE16
                | TYPE_TRAP_GATE16
                | TYPE_INTERRUPT_GATE32
                | TYPE_TRAP_GATE32
        ) || !is_system(lo)
        {
            return Err(Exception::general_protection(idt_error));
        }
        if software && dpl < self.cpl {
            return Err(Exception::general_protection(idt_error));
        }
        if lo & DESC_PRESENT_BIT == 0 {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(idt_error),
            ));
        }
        let selector = (lo >> GATE_SELECTOR_SHIFT) as u16;
        if gate_type == TYPE_TASK_GATE {
            let return_eip = self.rip;
            return self.task_switch(selector, TaskSwitchSource::Gate, return_eip, error_code);
        }
        let wide = gate_type >= TYPE_INTERRUPT_GATE32;
        let size: u8 = if wide { 4 } else { 2 };
        let offset = gate_offset(lo, wide);
        if selector_is_null(selector) {
            return Err(Exception::general_protection(0));
        }
        let new_cs = self.load_desc_seg(selector, false)?;
        let (target_lo, _) = self.read_desc(selector)?;
        let selector_error = (selector & SELECTOR_ERROR_CODE_MASK) as u32;
        if !new_cs.is_code() || is_system(target_lo) || new_cs.dpl() > self.cpl {
            return Err(Exception::general_protection(selector_error));
        }
        if !new_cs.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error),
            ));
        }
        let conforming = new_cs.flags & super::exec::SEG_TYPE_CONFORMING != 0;
        let from_v86 = self.eflags & VM != 0;
        let new_cpl = if conforming { self.cpl } else { new_cs.dpl() };
        if from_v86 && (conforming || new_cs.dpl() != 0) {
            return Err(Exception::general_protection(selector_error));
        }
        if offset > new_cs.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        let mut new_cs = new_cs;
        new_cs.sel = (selector & !SELECTOR_RPL_MASK) | new_cpl as u16;
        let old_flags = self.eflags_full();
        let old_cs = self.seg[SEG_CS].sel as u64;
        let old_rip = self.rip;
        let saved_sp = self.gpr[GPR_RSP];
        let saved_ss = self.seg[SEG_SS];
        let saved_cpl = self.cpl;
        let result = (|| -> ExecResult<()> {
            if new_cpl < self.cpl {
                let (stack_pointer, stack_selector, tss_wide) = self.tss_stack(new_cpl)?;
                let _ = tss_wide;
                let new_ss = self.checked_stack_segment(stack_selector, new_cpl)?;
                let old_ss = self.seg[SEG_SS].sel as u64;
                let old_sp = self.gpr[GPR_RSP];
                let mut sp = stack_pointer;
                if from_v86 {
                    for index in [SEG_GS, SEG_FS, SEG_DS, SEG_ES] {
                        let selector = self.seg[index].sel as u64;
                        self.push_on(&new_ss, &mut sp, size, selector)?;
                    }
                }
                self.push_on(&new_ss, &mut sp, size, old_ss)?;
                self.push_on(&new_ss, &mut sp, size, old_sp)?;
                self.push_on(&new_ss, &mut sp, size, old_flags)?;
                self.push_on(&new_ss, &mut sp, size, old_cs)?;
                self.push_on(&new_ss, &mut sp, size, old_rip)?;
                if let Some(code) = error_code {
                    self.push_on(&new_ss, &mut sp, size, code as u64)?;
                }
                self.seg[SEG_SS] = new_ss;
                self.gpr[GPR_RSP] = if new_ss.flags & SEG_DEFAULT_BIG != 0 {
                    sp
                } else {
                    (self.gpr[GPR_RSP] & !WORD_MASK) | sp
                };
                if from_v86 {
                    for index in [SEG_ES, SEG_DS, SEG_FS, SEG_GS] {
                        self.seg[index] = Seg::default();
                    }
                }
            } else {
                let mut sp = self.stack_pointer();
                self.push_implicit(&mut sp, size, old_flags)?;
                self.push_implicit(&mut sp, size, old_cs)?;
                self.push_implicit(&mut sp, size, old_rip)?;
                if let Some(code) = error_code {
                    self.push_implicit(&mut sp, size, code as u64)?;
                }
                self.set_stack_pointer(sp);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.gpr[GPR_RSP] = saved_sp;
            self.seg[SEG_SS] = saved_ss;
            self.set_cpl(saved_cpl);
            return Err(error);
        }
        self.seg[SEG_CS] = new_cs;
        self.set_cpl(new_cpl);
        self.rip = offset;
        self.materialize_lazy_flags();
        self.eflags &= !(TF | VM | RF | NT);
        if matches!(gate_type, TYPE_INTERRUPT_GATE16 | TYPE_INTERRUPT_GATE32) {
            self.eflags &= !IF;
        }
        Ok(())
    }

    fn deliver_real(&mut self, vector: u8) -> ExecResult<()> {
        let entry_offset = vector as u64 * IVT_ENTRY_BYTES;
        if entry_offset + (IVT_ENTRY_BYTES - 1) > self.idtr.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        let entry = self.read_sys(self.idtr.base.wrapping_add(entry_offset), 4)?;
        let offset = entry & WORD_MASK;
        let selector = (entry >> 16) as u16;
        let flags = self.eflags_full();
        let old_cs = self.seg[SEG_CS].sel as u64;
        let old_ip = self.rip & WORD_MASK;
        let mut sp = self.stack_pointer();
        self.push_implicit(&mut sp, 2, flags)?;
        self.push_implicit(&mut sp, 2, old_cs)?;
        self.push_implicit(&mut sp, 2, old_ip)?;
        self.set_stack_pointer(sp);
        let old = self.seg[SEG_CS];
        self.seg[SEG_CS] = Seg {
            sel: selector,
            base: (selector as u64) << 4,
            ..old
        };
        self.rip = offset;
        self.materialize_lazy_flags();
        self.eflags &= !(IF | TF | AC | RF);
        Ok(())
    }

    /// Far jump or call to `selector:offset` (both already decoded), pushing the return address
    /// for a call. `return_eip` is the address of the next instruction.
    fn far_transfer(
        &mut self,
        selector: u16,
        offset: u64,
        transfer: Transfer,
        operand_size: u8,
        return_eip: u64,
    ) -> ExecResult<()> {
        let size = operand_size.min(4);
        let offset = offset & mask(size);
        if !self.segments_are_protected() {
            if transfer == Transfer::Call {
                let old_cs = self.seg[SEG_CS].sel as u64;
                self.push(size, old_cs)?;
                self.push(size, return_eip & mask(size))?;
            }
            let old = self.seg[SEG_CS];
            self.seg[SEG_CS] = Seg {
                sel: selector,
                base: (selector as u64) << 4,
                ..old
            };
            if offset > self.seg[SEG_CS].limit as u64 {
                return Err(Exception::general_protection(0));
            }
            self.rip = offset;
            return Ok(());
        }
        if selector_is_null(selector) {
            return Err(Exception::general_protection(0));
        }
        let (lo, _) = self.read_desc(selector)?;
        let selector_error = (selector & SELECTOR_ERROR_CODE_MASK) as u32;
        if !is_system(lo) {
            let target = self.load_desc_seg(selector, false)?;
            self.check_code_target(&target, selector, transfer)?;
            return self.commit_far_code(target, selector, offset, transfer, size, return_eip);
        }
        let rpl = (selector & SELECTOR_RPL_MASK) as u8;
        match descriptor_type(lo) {
            TYPE_CALL_GATE16 | TYPE_CALL_GATE32 if self.efer & EFER_LMA == 0 => {
                if descriptor_dpl(lo) < self.cpl.max(rpl) {
                    return Err(Exception::general_protection(selector_error));
                }
                if lo & DESC_PRESENT_BIT == 0 {
                    return Err(Exception::fault(
                        VECTOR_SEGMENT_NOT_PRESENT,
                        Some(selector_error),
                    ));
                }
                let wide = descriptor_type(lo) == TYPE_CALL_GATE32;
                let target_selector = (lo >> GATE_SELECTOR_SHIFT) as u16;
                let gate_target = gate_offset(lo, wide);
                let parameters = ((lo >> 32) & GATE_PARAM_COUNT_MASK) as u32;
                self.through_call_gate(
                    target_selector,
                    gate_target,
                    parameters,
                    wide,
                    transfer,
                    return_eip,
                )
            }
            TYPE_TASK_GATE => {
                if descriptor_dpl(lo) < self.cpl.max(rpl) {
                    return Err(Exception::general_protection(selector_error));
                }
                if lo & DESC_PRESENT_BIT == 0 {
                    return Err(Exception::fault(
                        VECTOR_SEGMENT_NOT_PRESENT,
                        Some(selector_error),
                    ));
                }
                let tss_selector = (lo >> GATE_SELECTOR_SHIFT) as u16;
                let source = if transfer == Transfer::Call {
                    TaskSwitchSource::Call
                } else {
                    TaskSwitchSource::Jump
                };
                self.task_switch(tss_selector, source, return_eip, None)
            }
            TYPE_TSS16_AVAILABLE | TYPE_TSS32_AVAILABLE => {
                if descriptor_dpl(lo) < self.cpl.max(rpl) {
                    return Err(Exception::general_protection(selector_error));
                }
                let source = if transfer == Transfer::Call {
                    TaskSwitchSource::Call
                } else {
                    TaskSwitchSource::Jump
                };
                self.task_switch(selector, source, return_eip, None)
            }
            _ => Err(Exception::general_protection(selector_error)),
        }
    }

    /// Privilege checks for a far jump or call straight to a code segment.
    fn check_code_target(
        &self,
        target: &Seg,
        selector: u16,
        _transfer: Transfer,
    ) -> ExecResult<()> {
        let selector_error = (selector & SELECTOR_ERROR_CODE_MASK) as u32;
        if !target.is_code() {
            return Err(Exception::general_protection(selector_error));
        }
        let rpl = (selector & SELECTOR_RPL_MASK) as u8;
        let conforming = target.flags & super::exec::SEG_TYPE_CONFORMING != 0;
        let privileged_ok = if conforming {
            target.dpl() <= self.cpl
        } else {
            rpl <= self.cpl && target.dpl() == self.cpl
        };
        if !privileged_ok {
            return Err(Exception::general_protection(selector_error));
        }
        if !target.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error),
            ));
        }
        Ok(())
    }

    fn commit_far_code(
        &mut self,
        target: Seg,
        selector: u16,
        offset: u64,
        transfer: Transfer,
        size: u8,
        return_eip: u64,
    ) -> ExecResult<()> {
        // In IA-32e mode a far transfer may enter 64-bit mode; both L and D set is reserved.
        if target.flags & SEG_LONG != 0
            && (self.efer & EFER_LMA == 0 || target.flags & SEG_DEFAULT_BIG != 0)
        {
            return Err(Exception::general_protection(
                (selector & SELECTOR_ERROR_CODE_MASK) as u32,
            ));
        }
        if target.flags & SEG_LONG == 0 && offset > target.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        if transfer == Transfer::Call {
            let old_cs = self.seg[SEG_CS].sel as u64;
            let saved = self.gpr[GPR_RSP];
            let pushed = self
                .push(size, old_cs)
                .and_then(|()| self.push(size, return_eip & mask(size)));
            if let Err(error) = pushed {
                self.gpr[GPR_RSP] = saved;
                return Err(error);
            }
        }
        self.mark_descriptor_accessed(selector)?;
        let mut loaded = target;
        loaded.sel = (selector & !SELECTOR_RPL_MASK) | self.cpl as u16;
        self.seg[SEG_CS] = loaded;
        self.rip = offset;
        Ok(())
    }

    fn through_call_gate(
        &mut self,
        target_selector: u16,
        offset: u64,
        parameters: u32,
        wide: bool,
        transfer: Transfer,
        return_eip: u64,
    ) -> ExecResult<()> {
        let size: u8 = if wide { 4 } else { 2 };
        if selector_is_null(target_selector) {
            return Err(Exception::general_protection(0));
        }
        let target_error = (target_selector & SELECTOR_ERROR_CODE_MASK) as u32;
        let (target_lo, _) = self.read_desc(target_selector)?;
        let target = self.load_desc_seg(target_selector, false)?;
        if !target.is_code() || is_system(target_lo) || target.dpl() > self.cpl {
            return Err(Exception::general_protection(target_error));
        }
        if !target.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(target_error),
            ));
        }
        let conforming = target.flags & super::exec::SEG_TYPE_CONFORMING != 0;
        let new_cpl = if conforming { self.cpl } else { target.dpl() };
        if offset > target.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        let mut new_cs = target;
        new_cs.sel = (target_selector & !SELECTOR_RPL_MASK) | new_cpl as u16;
        let old_cs = self.seg[SEG_CS].sel as u64;
        if transfer == Transfer::Jump {
            if !conforming && target.dpl() != self.cpl {
                return Err(Exception::general_protection(target_error));
            }
            self.seg[SEG_CS] = new_cs;
            self.rip = offset;
            return Ok(());
        }
        if new_cpl < self.cpl {
            let (stack_pointer, stack_selector, _) = self.tss_stack(new_cpl)?;
            let new_ss = self.checked_stack_segment(stack_selector, new_cpl)?;
            let old_ss = self.seg[SEG_SS].sel as u64;
            let old_sp = self.stack_pointer();
            let old_ss_seg = self.seg[SEG_SS];
            let mut sp = stack_pointer;
            self.push_on(&new_ss, &mut sp, size, old_ss)?;
            self.push_on(&new_ss, &mut sp, size, old_sp)?;
            // Parameters are copied from the old stack, last one first.
            for index in (0..parameters as u64).rev() {
                let source = old_sp.wrapping_add(index * size as u64) & self.stack_mask();
                let linear = old_ss_seg.base.wrapping_add(source) & DWORD_MASK;
                let value = self.read_sys(linear, size)?;
                self.push_on(&new_ss, &mut sp, size, value)?;
            }
            self.push_on(&new_ss, &mut sp, size, old_cs)?;
            self.push_on(&new_ss, &mut sp, size, return_eip & mask(size))?;
            self.seg[SEG_SS] = new_ss;
            self.gpr[GPR_RSP] = sp;
            self.set_cpl(new_cpl);
        } else {
            let saved = self.gpr[GPR_RSP];
            let pushed = self
                .push(size, old_cs)
                .and_then(|()| self.push(size, return_eip & mask(size)));
            if let Err(error) = pushed {
                self.gpr[GPR_RSP] = saved;
                return Err(error);
            }
        }
        self.seg[SEG_CS] = new_cs;
        self.rip = offset;
        Ok(())
    }

    /// `call ptr16:16/32` and `jmp ptr16:16/32` (opcodes 9a and ea).
    pub(super) fn far_branch_pointer(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let transfer = if i.op == 0x9a {
            Transfer::Call
        } else {
            Transfer::Jump
        };
        self.far_transfer(i.imm2, i.imm as u64, transfer, i.opsize, next)
    }

    /// `call m16:16/32` and `jmp m16:16/32` (ff /3 and ff /5).
    pub(super) fn far_branch_indirect(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        if i.md == 3 {
            return Err(Exception::invalid_opcode());
        }
        let Loc::Mem(address) = self.modrm_location(i, next)? else {
            return Err(Exception::invalid_opcode());
        };
        let offset = self.read(address, i.opsize)?;
        let selector = self.read(address.wrapping_add(i.opsize as u64), 2)? as u16;
        let transfer = if i.reg & 7 == 3 {
            Transfer::Call
        } else {
            Transfer::Jump
        };
        self.far_transfer(selector, offset, transfer, i.opsize, next)
    }

    /// Far return (`retf`, `retf imm16`), including a return to an outer privilege level.
    pub(super) fn far_return_legacy(&mut self, i: &Insn, extra: u64) -> ExecResult<()> {
        let size = i.opsize.min(4);
        let sp = self.stack_pointer();
        let stack_mask = self.stack_mask();
        let read = |cpu: &mut Cpu, at: u64, size: u8| -> ExecResult<u64> {
            let linear = cpu.stack_linear(at & stack_mask)?;
            cpu.read_stack(linear, size)
        };
        let offset = read(self, sp, size)?;
        let selector = read(self, sp.wrapping_add(size as u64), size)? as u16;
        let released = sp.wrapping_add(2 * size as u64).wrapping_add(extra) & stack_mask;
        if !self.segments_are_protected() {
            let old = self.seg[SEG_CS];
            self.seg[SEG_CS] = Seg {
                sel: selector,
                base: (selector as u64) << 4,
                ..old
            };
            self.rip = offset;
            self.set_stack_pointer(released);
            return Ok(());
        }
        let new_cs = self.checked_return_code_segment(selector)?;
        let new_cpl = (selector & SELECTOR_RPL_MASK) as u8;
        if offset > new_cs.limit as u64 && new_cs.flags & SEG_LONG == 0 {
            return Err(Exception::general_protection(0));
        }
        if new_cpl == self.cpl {
            self.mark_descriptor_accessed(selector)?;
            self.seg[SEG_CS] = new_cs;
            self.rip = offset;
            self.set_stack_pointer(released);
            return Ok(());
        }
        let new_sp = read(self, released, size)?;
        let ss = read(self, released.wrapping_add(size as u64), size)? as u16;
        let new_ss = self.checked_data_seg_for_return(ss, new_cpl)?;
        self.seg[SEG_CS] = new_cs;
        self.seg[SEG_SS] = new_ss;
        self.set_cpl(new_cpl);
        self.rip = offset;
        let stack_mask = self.stack_mask();
        self.gpr[GPR_RSP] = (new_sp.wrapping_add(extra)) & stack_mask;
        self.null_inaccessible_after_outer_return(new_cpl);
        Ok(())
    }

    /// Validates the code selector popped by a far return or iret (protected mode, 16/32-bit).
    fn checked_return_code_segment(&mut self, selector: u16) -> ExecResult<Seg> {
        if selector_is_null(selector) {
            return Err(Exception::general_protection(0));
        }
        let selector_error = (selector & SELECTOR_ERROR_CODE_MASK) as u32;
        let (lo, _) = self.read_desc(selector)?;
        let seg = self.load_desc_seg(selector, false)?;
        let rpl = (selector & SELECTOR_RPL_MASK) as u8;
        let conforming = seg.flags & super::exec::SEG_TYPE_CONFORMING != 0;
        let dpl_ok = if conforming {
            seg.dpl() <= rpl
        } else {
            seg.dpl() == rpl
        };
        if is_system(lo) || !seg.is_code() || rpl < self.cpl || !dpl_ok {
            return Err(Exception::general_protection(selector_error));
        }
        if !seg.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error),
            ));
        }
        let mut loaded = seg;
        loaded.sel = selector;
        Ok(loaded)
    }

    fn checked_data_seg_for_return(&mut self, selector: u16, cpl: u8) -> ExecResult<Seg> {
        if selector_is_null(selector) {
            return Err(Exception::general_protection(0));
        }
        self.checked_data_seg(SEG_SS, selector, cpl)
    }

    fn null_inaccessible_after_outer_return(&mut self, new_cpl: u8) {
        for register in [SEG_ES, SEG_DS, SEG_FS, SEG_GS] {
            let seg = self.seg[register];
            let conforming_code =
                seg.is_code() && seg.flags & super::exec::SEG_TYPE_CONFORMING != 0;
            if !selector_is_null(seg.sel) && seg.dpl() < new_cpl && !conforming_code {
                self.seg[register] = Seg::default();
            }
        }
    }

    /// iret / iretd / 16-bit iret in real mode, virtual-8086 mode and protected mode.
    /// `next` is the address after the `iret`, which a task switch saves in the outgoing TSS.
    pub(super) fn iret_legacy(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let size = i.opsize.min(4);
        let stack_mask = self.stack_mask();
        let sp = self.stack_pointer();
        let read = |cpu: &mut Cpu, at: u64, size: u8| -> ExecResult<u64> {
            let linear = cpu.stack_linear(at & stack_mask)?;
            cpu.read_stack(linear, size)
        };
        let step = size as u64;
        if self.cr0 & CR0_PE == 0 {
            let ip = read(self, sp, size)?;
            let cs = read(self, sp.wrapping_add(step), size)? as u16;
            let flags = read(self, sp.wrapping_add(2 * step), size)?;
            let old = self.seg[SEG_CS];
            if ip > old.limit as u64 {
                return Err(Exception::general_protection(0));
            }
            self.seg[SEG_CS] = Seg {
                sel: cs,
                base: (cs as u64) << 4,
                ..old
            };
            self.rip = ip;
            self.set_stack_pointer(sp.wrapping_add(3 * step));
            let mut loadable = self.loadable_eflags(self.cpl);
            if size == 2 {
                loadable &= WORD_MASK;
            }
            let current = self.eflags_full();
            self.eflags = ((current & !loadable) | (flags & loadable)) | RESERVED_ONE;
            self.lazy_op = LazyFlagsOp::Eflags;
            return Ok(());
        }
        if self.eflags & VM != 0 {
            if self.iopl() < USER_CPL {
                return Err(Exception::general_protection(0));
            }
            let ip = read(self, sp, size)?;
            let cs = read(self, sp.wrapping_add(step), size)? as u16;
            let flags = read(self, sp.wrapping_add(2 * step), size)?;
            if ip > WORD_MASK {
                return Err(Exception::general_protection(0));
            }
            self.seg[SEG_CS] = real_mode_segment(cs, USER_CPL);
            self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
            self.rip = ip;
            self.set_stack_pointer(sp.wrapping_add(3 * step));
            // IOPL and VM are not changed by an iret in virtual-8086 mode.
            let mut loadable = self.loadable_eflags(self.cpl) & !IOPL;
            if size == 2 {
                loadable &= WORD_MASK;
            }
            let current = self.eflags_full();
            self.eflags = ((current & !loadable) | (flags & loadable)) | RESERVED_ONE;
            self.lazy_op = LazyFlagsOp::Eflags;
            return Ok(());
        }
        if self.eflags & NT != 0 {
            return self.task_return(next);
        }
        let eip = read(self, sp, size)?;
        let selector = read(self, sp.wrapping_add(step), size)? as u16;
        let flags = read(self, sp.wrapping_add(2 * step), size)?;
        let old_cpl = self.cpl;
        if size == 4 && flags & VM != 0 && old_cpl == 0 {
            return self.return_to_virtual_8086(sp, eip, selector, flags);
        }
        let new_cs = self.checked_return_code_segment(selector)?;
        let new_cpl = (selector & SELECTOR_RPL_MASK) as u8;
        if new_cs.flags & SEG_LONG == 0 && eip > new_cs.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        let loadable = {
            let mut loadable = self.loadable_eflags(old_cpl);
            if size == 2 {
                loadable &= WORD_MASK;
            }
            loadable
        };
        if new_cpl == old_cpl {
            self.mark_descriptor_accessed(selector)?;
            self.seg[SEG_CS] = new_cs;
            self.rip = eip;
            self.set_stack_pointer(sp.wrapping_add(3 * step));
        } else {
            let new_sp = read(self, sp.wrapping_add(3 * step), size)?;
            let ss = read(self, sp.wrapping_add(4 * step), size)? as u16;
            let new_ss = self.checked_data_seg_for_return(ss, new_cpl)?;
            self.seg[SEG_CS] = new_cs;
            self.seg[SEG_SS] = new_ss;
            self.set_cpl(new_cpl);
            self.rip = eip;
            let stack_mask = self.stack_mask();
            self.gpr[GPR_RSP] = if size == 2 {
                (self.gpr[GPR_RSP] & !WORD_MASK) | (new_sp & WORD_MASK)
            } else {
                new_sp & stack_mask
            };
            self.null_inaccessible_after_outer_return(new_cpl);
        }
        let current = self.eflags_full();
        self.eflags = ((current & !loadable) | (flags & loadable)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        Ok(())
    }

    /// iretd from ring 0 with EFLAGS.VM set in the popped image: enter virtual-8086 mode.
    fn return_to_virtual_8086(
        &mut self,
        sp: u64,
        eip: u64,
        selector: u16,
        flags: u64,
    ) -> ExecResult<()> {
        let stack_mask = self.stack_mask();
        let read = |cpu: &mut Cpu, at: u64| -> ExecResult<u64> {
            let linear = cpu.stack_linear(at & stack_mask)?;
            cpu.read_stack(linear, 4)
        };
        let new_esp = read(self, sp.wrapping_add(12))?;
        let ss = read(self, sp.wrapping_add(16))? as u16;
        let es = read(self, sp.wrapping_add(20))? as u16;
        let ds = read(self, sp.wrapping_add(24))? as u16;
        let fs = read(self, sp.wrapping_add(28))? as u16;
        let gs = read(self, sp.wrapping_add(32))? as u16;
        let loadable = self.loadable_eflags(0) | VM;
        let current = self.eflags_full();
        self.eflags = ((current & !loadable) | (flags & loadable)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.seg[SEG_CS] = real_mode_segment(selector, USER_CPL);
        self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
        for (index, value) in [
            (SEG_SS, ss),
            (SEG_ES, es),
            (SEG_DS, ds),
            (SEG_FS, fs),
            (SEG_GS, gs),
        ] {
            self.seg[index] = real_mode_segment(value, USER_CPL);
        }
        self.set_cpl(USER_CPL);
        self.rip = eip & WORD_MASK;
        self.gpr[GPR_RSP] = new_esp;
        Ok(())
    }
}

/// How a task switch was started; it decides the busy bits, the back link and NT.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TaskSwitchSource {
    Jump,
    Call,
    Gate,
    Return,
}
