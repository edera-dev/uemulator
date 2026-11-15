//! What differs outside 64-bit mode: how CS selects the decoding mode, segmentation (bases,
//! limits, real-mode selectors) and the instruction and stack pointer widths.
//!
//! 64-bit mode ignores the ES/CS/SS/DS bases and every limit, and the JIT relies on that, so the
//! JIT only compiles code in `CodeMode::Long64` (`Cpu::run`) and everything here is the
//! interpreter's business.

use super::decode::CodeMode;
use super::exception::Exception;
use super::flags::VM;
use super::*;
use crate::boot::BootState;

/// Real mode and virtual-8086 mode form a linear address as `selector << 4` plus the offset.
const REAL_MODE_SEGMENT_SHIFT: u32 = 4;
const REAL_MODE_LIMIT: u32 = 0xffff;
/// Descriptor type bit 2 of a data segment: the valid offsets are above the limit.
const SEG_TYPE_EXPAND_DOWN: u32 = 1 << 10;
/// Linear addresses wrap at 4 GiB outside 64-bit mode.
const LINEAR_MASK_32: u64 = 0xffff_ffff;
const STACK_MASK_16: u64 = 0xffff;
const STACK_MASK_32: u64 = 0xffff_ffff;
/// Largest offset of a segment whose D/B bit is clear or set, for the expand-down limit check.
const OFFSET_MAX_16: u64 = 0xffff;
const OFFSET_MAX_32: u64 = 0xffff_ffff;

/// A data segment as real mode and virtual-8086 mode load it: present, writable, 64 KiB, with the
/// descriptor-privilege-level the caller passes (3 in virtual-8086 mode).
pub(crate) fn real_mode_segment(selector: u16, dpl: u8) -> Seg {
    Seg {
        sel: selector,
        base: (selector as u64) << REAL_MODE_SEGMENT_SHIFT,
        limit: REAL_MODE_LIMIT,
        flags: SEG_PRESENT | SEG_NON_SYSTEM | SEG_TYPE_READ_WRITE | ((dpl as u32) << SEG_DPL_SHIFT),
    }
}

impl Cpu {
    /// Whether CS selects 64-bit mode: CS.L with EFER.LMA. Compatibility mode (LMA with CS.L
    /// clear) is a 16- or 32-bit mode as far as decoding and addressing go.
    pub(crate) fn is_long64(&self) -> bool {
        self.seg[SEG_CS].flags & SEG_LONG != 0 && self.efer & EFER_LMA != 0
    }

    pub(crate) fn code_mode(&self) -> CodeMode {
        if self.is_long64() {
            CodeMode::Long64
        } else if self.seg[SEG_CS].flags & SEG_DEFAULT_BIG != 0 {
            CodeMode::Bits32
        } else {
            CodeMode::Bits16
        }
    }

    /// Protected mode, virtual-8086 mode excluded: the segment checks of the SDM apply only here.
    pub(crate) fn segments_are_protected(&self) -> bool {
        self.cr0 & CR0_PE != 0 && self.eflags & VM == 0
    }

    /// Linear address of `offset` in segment register `index` for a one-byte read; see
    /// `segment_access`.
    pub(crate) fn segment_linear(&self, index: usize, offset: u64) -> ExecResult<u64> {
        self.segment_access(index, offset, 1, false)
    }

    /// Linear address of `offset` in segment register `index`, checked for an access of `size`
    /// bytes against the segment limit (expand-down segments included) and, in protected mode, that
    /// the segment is present, readable (or, for a write, a writable data segment). A violation is
    /// #SS(0) through SS and #GP(0) otherwise. 64-bit mode only applies the FS and GS bases.
    pub(crate) fn segment_access(
        &self,
        index: usize,
        offset: u64,
        size: u8,
        write: bool,
    ) -> ExecResult<u64> {
        let seg = &self.seg[index];
        if self.is_long64() {
            return Ok(match index {
                SEG_FS | SEG_GS => offset.wrapping_add(seg.base),
                _ => offset,
            });
        }
        let violation = || {
            if index == SEG_SS {
                Exception::stack_fault(0)
            } else {
                Exception::general_protection(0)
            }
        };
        if self.segments_are_protected() {
            // A null or not-present selector cannot be used; code is never writable and may be
            // execute-only; data needs its write bit to be stored to.
            let unusable = !seg.present()
                || if write {
                    seg.is_code() || seg.flags & SEG_TYPE_READ_WRITE == 0
                } else {
                    seg.is_code() && seg.flags & SEG_TYPE_READ_WRITE == 0
                };
            if unusable {
                return Err(violation());
            }
        }
        let last = offset.wrapping_add(size.max(1) as u64 - 1);
        let outside = if seg.flags & SEG_TYPE_EXPAND_DOWN != 0 && !seg.is_code() {
            let top = if seg.flags & SEG_DEFAULT_BIG != 0 {
                OFFSET_MAX_32
            } else {
                OFFSET_MAX_16
            };
            offset <= seg.limit as u64 || last > top || last < offset
        } else {
            last > seg.limit as u64 || last < offset
        };
        if outside {
            return Err(violation());
        }
        Ok(seg.base.wrapping_add(offset) & LINEAR_MASK_32)
    }

    /// Checks the whole extent of the operand `linear_address` last formed (outside 64-bit
    /// mode): `load_operand` and `store_operand` know the size and direction, `linear_address`
    /// does not.
    pub(crate) fn check_operand_extent(&self, size: u8, write: bool) -> ExecResult<()> {
        if let Some((index, offset)) = self.pending_operand.get() {
            self.segment_access(index as usize, offset, size, write)?;
        }
        Ok(())
    }

    /// Linear address of the instruction at `rip`, faulting (#GP(0)) if it lies beyond the CS limit.
    pub(crate) fn fetch_linear(&self) -> ExecResult<u64> {
        if self.is_long64() {
            return Ok(self.rip);
        }
        let cs = &self.seg[SEG_CS];
        let offset = self.rip & LINEAR_MASK_32;
        if offset > cs.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        Ok(cs.base.wrapping_add(offset) & LINEAR_MASK_32)
    }

    /// The part of RSP the stack uses: all of it in 64-bit mode, ESP or SP by SS.B otherwise.
    pub(crate) fn stack_mask(&self) -> u64 {
        if self.is_long64() {
            u64::MAX
        } else if self.seg[SEG_SS].flags & SEG_DEFAULT_BIG != 0 {
            STACK_MASK_32
        } else {
            STACK_MASK_16
        }
    }

    pub(crate) fn stack_pointer(&self) -> u64 {
        self.gpr[GPR_RSP] & self.stack_mask()
    }

    /// Writes the stack pointer, leaving the bits above the stack width alone as hardware does
    /// for a 16-bit SP.
    pub(crate) fn set_stack_pointer(&mut self, sp: u64) {
        let mask = self.stack_mask();
        self.gpr[GPR_RSP] = if mask == STACK_MASK_32 {
            // Writing ESP clears the upper half, like any 32-bit register write.
            sp & mask
        } else {
            (self.gpr[GPR_RSP] & !mask) | (sp & mask)
        };
    }

    /// The architectural reset state: real mode, CS:IP = F000:FFF0 with the base at 0xFFFF0000,
    /// paging and long mode off, every data segment 0 with a 64 KiB limit, the interrupt vector
    /// table at 0 with its 256 four-byte entries.
    pub(crate) fn reset_to_real_mode(&mut self) {
        self.reset_real_mode_registers();
        self.flush_tlb();
        self.flush_decode();
    }

    /// `reset_to_real_mode` without the TLB and decode-cache flushes, for an INIT that arrives
    /// while the machine runs: `flush_decode` also forgets which pages hold translated code,
    /// which the shared JIT cache cannot afford.
    pub(crate) fn reset_real_mode_registers(&mut self) {
        const RESET_CODE_SELECTOR: u16 = 0xf000;
        const RESET_CODE_BASE: u64 = 0xffff_0000;
        const RESET_IP: u64 = 0xfff0;
        const IVT_LIMIT: u16 = 0x3ff;
        const CR0_RESET: u64 = CR0_ET;
        for index in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
            self.seg[index] = real_mode_segment(0, 0);
        }
        self.seg[SEG_CS] = Seg {
            sel: RESET_CODE_SELECTOR,
            base: RESET_CODE_BASE,
            ..real_mode_segment(0, 0)
        };
        self.seg[SEG_CS].flags |= SEG_TYPE_EXECUTABLE;
        self.cr0 = CR0_RESET;
        self.cr3 = 0;
        self.cr4 = 0;
        self.efer = 0;
        self.eflags = flags::RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.idtr = DescriptorTable {
            base: 0,
            limit: IVT_LIMIT,
        };
        self.gdtr = DescriptorTable::default();
        self.ldtr = Seg::default();
        self.tr = Seg::default();
        self.rip = RESET_IP;
        self.halted = false;
        self.cet.ssp = 0;
        self.set_cpl(0);
    }

    /// The 32-bit Linux boot protocol's CPU state (Documentation/arch/x86/boot.rst): protected
    /// mode with flat 4 GiB code and data segments, paging off, interrupts off, ESI pointing at
    /// `boot_params` and EBP, EDI and EBX zero.
    pub(crate) fn enter_boot_protected_mode(&mut self, boot: BootState) {
        let flat = |selector: u16, flags: u32| Seg {
            sel: selector,
            base: 0,
            limit: SEG_LIMIT_4GIB,
            flags,
        };
        let code_flags = SEG_GRANULARITY
            | SEG_DEFAULT_BIG
            | SEG_PRESENT
            | SEG_NON_SYSTEM
            | SEG_TYPE_EXECUTABLE
            | SEG_TYPE_READ_WRITE;
        self.seg[SEG_CS] = flat(boot.cs, code_flags);
        for index in [SEG_ES, SEG_SS, SEG_DS, SEG_FS, SEG_GS] {
            self.seg[index] = flat(boot.ds, SEG_FLAGS_FLAT_DATA);
        }
        self.cr0 = CR0_PE | CR0_ET;
        self.cr3 = 0;
        self.cr4 = 0;
        self.efer = 0;
        self.eflags = flags::RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.set_cpl(0);
        self.flush_tlb();
        self.flush_decode();
    }

    /// Linear address of a stack access at stack offset `sp`.
    pub(crate) fn stack_linear(&self, sp: u64) -> ExecResult<u64> {
        self.segment_linear(SEG_SS, sp)
    }
}
