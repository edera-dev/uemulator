//! Hardware task switching with 32-bit TSSes: `jmp`/`call` to a TSS or task gate, an interrupt
//! through a task gate, and `iret` with EFLAGS.NT set. 16-bit TSSes are not switched to.

use super::exception::{Exception, VECTOR_INVALID_TSS, VECTOR_SEGMENT_NOT_PRESENT};
use super::flags::*;
use super::legacy::selector_is_null;
use super::legacy_flow::TaskSwitchSource;
use super::modes::real_mode_segment;
use super::*;

const SELECTOR_RPL_MASK: u16 = 3;
const SELECTOR_ERROR_CODE_MASK: u16 = 0xfffc;
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
const SELECTOR_TI_LDT: u16 = 1 << 2;
const WORD_MASK: u64 = 0xffff;
const DWORD_MASK: u64 = 0xffff_ffff;
const TSS_TYPE_SHIFT: u32 = 8;
const TSS_TYPE_MASK: u32 = 0xf;
const TSS_TYPE_32_AVAILABLE: u32 = 9;
const TSS_TYPE_32_BUSY: u32 = 0xb;
const TSS_TYPE_LDT: u32 = 2;
/// Offset of the upper dword of a descriptor, which holds the type.
const DESCRIPTOR_HIGH_DWORD: u64 = 4;
const TSS_BUSY_BIT: u64 = 2 << TSS_TYPE_SHIFT;
const TSS_LINK: u64 = 0;
const TSS_CR3: u64 = 28;
const TSS_EIP: u64 = 32;
const TSS_EFLAGS: u64 = 36;
const TSS_GPRS: u64 = 40;
const TSS_SEGMENTS: u64 = 72;
const TSS_LDT: u64 = 96;
const TSS_MIN_LIMIT: u64 = 0x67;
/// TSS segment slots in memory order.
const TSS_SEGMENT_ORDER: [usize; 6] = [SEG_ES, SEG_CS, SEG_SS, SEG_DS, SEG_FS, SEG_GS];

fn type_of(seg: &Seg) -> u32 {
    (seg.flags >> TSS_TYPE_SHIFT) & TSS_TYPE_MASK
}

impl Cpu {
    fn tss_selector_fault(vector: u8, selector: u16) -> Exception {
        Exception::fault(vector, Some((selector & SELECTOR_ERROR_CODE_MASK) as u32))
    }

    /// Reads the descriptor of the TSS `selector` names (it must be in the GDT).
    fn tss_descriptor(&mut self, selector: u16) -> ExecResult<Seg> {
        if selector & SELECTOR_TI_LDT != 0 {
            return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, selector));
        }
        let (lo, _) = self
            .read_desc(selector)
            .map_err(|_| Self::tss_selector_fault(VECTOR_INVALID_TSS, selector))?;
        let dword0 = lo as u32;
        let dword1 = (lo >> 32) as u32;
        let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64;
        let mut limit = (dword0 & 0xffff) | (dword1 & 0xf_0000);
        if dword1 & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        Ok(Seg {
            sel: selector,
            base,
            limit,
            flags: dword1 & SEG_DESCRIPTOR_FLAGS_MASK,
        })
    }

    fn set_tss_busy(&mut self, selector: u16, busy: bool) -> ExecResult<()> {
        let address = self
            .gdtr
            .base
            .wrapping_add((selector & SELECTOR_OFFSET_MASK) as u64 + DESCRIPTOR_HIGH_DWORD);
        let high = self.read_sys(address, 4)?;
        let updated = if busy {
            high | TSS_BUSY_BIT
        } else {
            high & !TSS_BUSY_BIT
        };
        self.write_sys(address, 4, updated)
    }

    /// `iret` with NT set: switch back to the task the current TSS links to. The outgoing task
    /// resumes at `next`, after the `iret`.
    pub(super) fn task_return(&mut self, next: u64) -> ExecResult<()> {
        let link = self.read_sys(self.tr.base.wrapping_add(TSS_LINK), 2)? as u16;
        self.task_switch(link, TaskSwitchSource::Return, next, None)
    }

    pub(super) fn task_switch(
        &mut self,
        selector: u16,
        source: TaskSwitchSource,
        return_eip: u64,
        error_code: Option<u32>,
    ) -> ExecResult<()> {
        // A guest that intercepts task switches exits before anything is checked or written; the
        // hypervisor emulates the whole switch.
        self.svm_task_switch_intercept(selector, source, return_eip, error_code)?;
        let new_tss = self.tss_descriptor(selector)?;
        let required = if source == TaskSwitchSource::Return {
            TSS_TYPE_32_BUSY
        } else {
            TSS_TYPE_32_AVAILABLE
        };
        if new_tss.flags & SEG_NON_SYSTEM != 0 || type_of(&new_tss) != required {
            return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, selector));
        }
        if !new_tss.present() {
            return Err(Self::tss_selector_fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                selector,
            ));
        }
        if (new_tss.limit as u64) < TSS_MIN_LIMIT {
            return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, selector));
        }
        let old_selector = self.tr.sel;
        let old_base = self.tr.base;
        if (self.tr.limit as u64) < TSS_MIN_LIMIT {
            return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, old_selector));
        }
        let leaves_busy = matches!(source, TaskSwitchSource::Call | TaskSwitchSource::Gate);
        let outgoing_flags = self.eflags_full() & !if leaves_busy { 0 } else { NT };
        // Save the outgoing task.
        self.write_sys(old_base.wrapping_add(TSS_EIP), 4, return_eip)?;
        self.write_sys(old_base.wrapping_add(TSS_EFLAGS), 4, outgoing_flags)?;
        let saved_registers: [u64; 8] = core::array::from_fn(|index| self.gpr[index]);
        for (index, value) in saved_registers.into_iter().enumerate() {
            self.write_sys(old_base.wrapping_add(TSS_GPRS + 4 * index as u64), 4, value)?;
        }
        for (slot, register) in TSS_SEGMENT_ORDER.into_iter().enumerate() {
            let value = self.seg[register].sel as u64;
            self.write_sys(
                old_base.wrapping_add(TSS_SEGMENTS + 4 * slot as u64),
                4,
                value,
            )?;
        }
        if matches!(source, TaskSwitchSource::Jump | TaskSwitchSource::Return) {
            self.set_tss_busy(old_selector, false)?;
        }
        // Read the incoming task before anything is replaced.
        let base = new_tss.base;
        let incoming_cr3 = self.read_sys(base.wrapping_add(TSS_CR3), 4)?;
        let incoming_eip = self.read_sys(base.wrapping_add(TSS_EIP), 4)?;
        let mut incoming_flags = self.read_sys(base.wrapping_add(TSS_EFLAGS), 4)?;
        let mut registers = [0u64; 8];
        for (index, register) in registers.iter_mut().enumerate() {
            *register = self.read_sys(base.wrapping_add(TSS_GPRS + 4 * index as u64), 4)?;
        }
        let mut selectors = [0u16; 6];
        for (slot, value) in selectors.iter_mut().enumerate() {
            *value = self.read_sys(base.wrapping_add(TSS_SEGMENTS + 4 * slot as u64), 2)? as u16;
        }
        let incoming_ldt = self.read_sys(base.wrapping_add(TSS_LDT), 2)? as u16;
        if leaves_busy {
            self.write_sys(base.wrapping_add(TSS_LINK), 2, old_selector as u64)?;
            incoming_flags |= NT;
        }
        if source != TaskSwitchSource::Return {
            self.set_tss_busy(selector, true)?;
        }
        let mut loaded_tss = new_tss;
        loaded_tss.flags |= TSS_BUSY_BIT as u32;
        self.tr = loaded_tss;
        self.cr0 |= super::sys::CR0_TS;
        if self.cr0 & CR0_PG != 0 && self.cr3 != incoming_cr3 {
            self.cr3 = incoming_cr3;
            self.flush_tlb();
            self.flush_decode();
        }
        self.gpr[..8].copy_from_slice(&registers);
        self.rip = incoming_eip;
        self.eflags = (incoming_flags & !(RF)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.load_task_segments(selectors, incoming_ldt, incoming_flags & VM != 0)?;
        if self.seg[SEG_CS].flags & SEG_LONG == 0 && incoming_eip > self.seg[SEG_CS].limit as u64 {
            return Err(Exception::general_protection(0));
        }
        if let Some(code) = error_code {
            let mut sp = self.stack_pointer();
            let mask = self.stack_mask();
            sp = sp.wrapping_sub(4) & mask;
            let linear = self.stack_linear(sp)?;
            self.write_sys(linear, 4, code as u64)?;
            self.set_stack_pointer(sp);
        }
        Ok(())
    }

    /// Loads LDTR and the segment registers of the incoming task; any bad one is #TS.
    fn load_task_segments(&mut self, selectors: [u16; 6], ldt: u16, v86: bool) -> ExecResult<()> {
        if v86 {
            for (slot, register) in TSS_SEGMENT_ORDER.into_iter().enumerate() {
                self.seg[register] = real_mode_segment(selectors[slot], USER_CPL);
            }
            self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
            self.set_cpl(USER_CPL);
            return Ok(());
        }
        if !selector_is_null(ldt) {
            if ldt & SELECTOR_TI_LDT != 0 {
                return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, ldt));
            }
            let seg = self.tss_descriptor(ldt)?;
            if seg.flags & SEG_NON_SYSTEM != 0 || type_of(&seg) != TSS_TYPE_LDT {
                return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, ldt));
            }
            self.ldtr = seg;
        } else {
            self.ldtr = Seg::default();
        }
        let code_selector = selectors[1];
        let new_cpl = (code_selector & SELECTOR_RPL_MASK) as u8;
        let code = self
            .load_desc_seg(code_selector, false)
            .map_err(|_| Self::tss_selector_fault(VECTOR_INVALID_TSS, code_selector))?;
        if selector_is_null(code_selector) || !code.is_code() || !code.present() {
            return Err(Self::tss_selector_fault(VECTOR_INVALID_TSS, code_selector));
        }
        self.seg[SEG_CS] = code;
        self.set_cpl(new_cpl);
        for register in [SEG_SS, SEG_ES, SEG_DS, SEG_FS, SEG_GS] {
            let slot = TSS_SEGMENT_ORDER
                .iter()
                .position(|&r| r == register)
                .unwrap();
            let selector = selectors[slot];
            let loaded = self
                .checked_data_seg(register, selector, new_cpl)
                .map_err(|error| match error {
                    Exception::Fault {
                        vector,
                        error_code,
                        software,
                    } if vector != 14 => {
                        let _ = (error_code, software);
                        Self::tss_selector_fault(VECTOR_INVALID_TSS, selector)
                    }
                    other => other,
                })?;
            self.seg[register] = loaded;
        }
        let _ = (WORD_MASK, DWORD_MASK);
        Ok(())
    }
}
