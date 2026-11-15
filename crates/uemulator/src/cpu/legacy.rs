//! Instructions whose behaviour depends on the code mode, for 16- and 32-bit code (real mode,
//! virtual-8086 mode, protected mode and compatibility mode). `Cpu::exec` tries `exec_legacy`
//! first outside 64-bit mode; an opcode it does not claim runs the shared implementation, which
//! only depends on the operand size the decoder already resolved.
//!
//! What lives here: the stack and branch instructions (their operand and stack widths come from
//! the mode and SS.B), the string instructions (address size, ES base), the opcodes that exist
//! only outside 64-bit mode (inc/dec r, pusha, BCD, bound, arpl, push/pop segment, far pointers,
//! les/lds), and the segment loads.

use super::decode::{CodeMode, Insn, NO_SEG};
use super::exception::Exception;
use super::exec::Loc;
use super::flags::*;
use super::modes::real_mode_segment;
use super::*;

const GPR_RAX: usize = 0;
const GPR_RCX: usize = 1;
const GPR_RDX: usize = 2;
const GPR_RBX: usize = 3;
const GPR_RBP: usize = 5;
const GPR_RDI: usize = 7;

const BYTE_MASK: u64 = 0xff;
const WORD_MASK: u64 = 0xffff;
const DWORD_MASK: u64 = 0xffff_ffff;
const ENTER_LEVEL_MASK: u16 = 31;
const VECTOR_BOUND_RANGE: u8 = 5;
const VECTOR_OVERFLOW: u8 = 4;
/// Registers pusha stores, in push order; popa reads them back in reverse and skips SP.
const PUSHA_ORDER: [usize; 8] = [
    GPR_RAX, GPR_RCX, GPR_RDX, GPR_RBX, GPR_RSP, GPR_RBP, 6, GPR_RDI,
];
const REG_COUNT: usize = 8;
const REP_REPE: u8 = 1;
const REP_REPNE: u8 = 2;
/// A rep string instruction yields to the run loop after this many iterations so events are
/// serviced; the interpreter's 64-bit path counts bytes (`REP_BYTES_PER_SLICE`).
const REP_ITERATIONS_PER_SLICE: u64 = 4096;
const BCD_ADJUST_LOW: u64 = 0x06;
const BCD_ADJUST_HIGH: u64 = 0x60;
const BCD_NIBBLE_LIMIT: u64 = 9;
const BCD_BYTE_LIMIT: u64 = 0x99;
const SELECTOR_RPL_MASK: u16 = 3;

/// A selector with no table index, which is "null" for every purpose here.
pub(crate) fn selector_is_null(selector: u16) -> bool {
    selector & 0xfffc == 0
}

fn stack_size(i: &Insn) -> u8 {
    match (i.mode, i.p66) {
        (CodeMode::Long64, false) => 8,
        (CodeMode::Long64, true) => 2,
        (CodeMode::Bits32, false) | (CodeMode::Bits16, true) => 4,
        (CodeMode::Bits32, true) | (CodeMode::Bits16, false) => 2,
    }
}

fn address_mask(i: &Insn) -> u64 {
    if i.addr16 {
        WORD_MASK
    } else if i.addr32 {
        DWORD_MASK
    } else {
        u64::MAX
    }
}

impl Cpu {
    /// Writes `value` to the register an address-size-wide operand names (SI, DI, CX, ...): 16-bit
    /// writes keep the upper bits, 32-bit writes clear them.
    fn write_address_register(&mut self, register: usize, value: u64, i: &Insn) {
        if i.addr16 {
            self.gpr[register] = (self.gpr[register] & !WORD_MASK) | (value & WORD_MASK);
        } else if i.addr32 {
            self.gpr[register] = value & DWORD_MASK;
        } else {
            self.gpr[register] = value;
        }
    }

    /// Takes a near branch to `target` (an offset in CS): truncated to the operand size, and
    /// #GP(0) when it lies beyond the CS limit.
    pub(super) fn branch_near(&mut self, i: &Insn, target: u64) -> ExecResult<()> {
        let target = target & mask(i.opsize.min(4));
        if !self.is_long64() && target > self.seg[SEG_CS].limit as u64 {
            return Err(Exception::general_protection(0));
        }
        self.rip = target;
        Ok(())
    }

    pub(super) fn push_legacy(&mut self, size: u8, value: u64) -> ExecResult<()> {
        let sp = self.stack_pointer().wrapping_sub(size as u64) & self.stack_mask();
        let linear = self.stack_linear(sp)?;
        self.write_stack(linear, size, value)?;
        self.set_stack_pointer(sp);
        Ok(())
    }

    pub(super) fn pop_legacy(&mut self, size: u8) -> ExecResult<u64> {
        let sp = self.stack_pointer();
        let linear = self.stack_linear(sp)?;
        let value = self.read_stack(linear, size)?;
        self.set_stack_pointer(sp.wrapping_add(size as u64) & self.stack_mask());
        Ok(value)
    }

    /// Loads a segment register from a selector. Real mode and virtual-8086 mode build the
    /// segment from the selector (real mode keeps the old limit and attributes, which is how
    /// "unreal mode" works); protected mode validates the descriptor.
    pub fn load_segment_register(&mut self, index: usize, selector: u16) -> ExecResult<()> {
        if self.cr0 & CR0_PE == 0 {
            let old = self.seg[index];
            self.seg[index] = if old.flags == 0 {
                real_mode_segment(selector, 0)
            } else {
                Seg {
                    sel: selector,
                    base: (selector as u64) << 4,
                    ..old
                }
            };
            return Ok(());
        }
        if self.eflags & VM != 0 {
            self.seg[index] = real_mode_segment(selector, USER_CPL);
            return Ok(());
        }
        self.load_data_seg(index, selector)
    }

    /// Returns whether `i` was an opcode this module handles (RIP is then already updated).
    pub(super) fn exec_legacy(&mut self, i: &Insn, next: u64) -> ExecResult<bool> {
        let op = i.op;
        let osz = i.opsize;
        let sz = stack_size(i);
        match op {
            // inc / dec r16/r32 (REX in 64-bit mode)
            0x40..=0x4f => {
                let r = (op & 7) as u8;
                let a = self.read_gpr(r, osz, true);
                let cf = self.cf() as u64;
                let (result, lazy_op) = if op < 0x48 {
                    (a.wrapping_add(1) & mask(osz), LazyFlagsOp::Inc)
                } else {
                    (a.wrapping_sub(1) & mask(osz), LazyFlagsOp::Dec)
                };
                self.set_lazy_flags(lazy_op, osz, result, cf, 0);
                self.write_gpr(r, osz, true, result);
            }
            0x50..=0x57 => {
                let value = self.read_gpr((op & 7) as u8, sz, true);
                self.push(sz, value)?;
            }
            0x58..=0x5f => {
                let value = self.pop(sz)?;
                self.write_gpr((op & 7) as u8, sz, true, value);
            }
            0x68 | 0x6a => self.push(sz, i.imm as u64)?,
            0x60 => {
                let original_sp = self.read_gpr(GPR_RSP as u8, sz, true);
                let mut sp = self.stack_pointer();
                for register in PUSHA_ORDER {
                    let value = if register == GPR_RSP {
                        original_sp
                    } else {
                        self.read_gpr(register as u8, sz, true)
                    };
                    sp = sp.wrapping_sub(sz as u64) & self.stack_mask();
                    let linear = self.stack_linear(sp)?;
                    self.write_stack(linear, sz, value)?;
                }
                self.set_stack_pointer(sp);
            }
            0x61 => {
                let mut sp = self.stack_pointer();
                let mut values = [0u64; REG_COUNT];
                for register in PUSHA_ORDER.iter().rev() {
                    let linear = self.stack_linear(sp)?;
                    let value = self.read_stack(linear, sz)?;
                    sp = sp.wrapping_add(sz as u64) & self.stack_mask();
                    values[*register] = value;
                }
                for register in PUSHA_ORDER {
                    if register != GPR_RSP {
                        self.write_gpr(register as u8, sz, true, values[register]);
                    }
                }
                self.set_stack_pointer(sp);
            }
            // push es/cs/ss/ds
            0x06 | 0x0e | 0x16 | 0x1e => {
                let index = match op {
                    0x06 => SEG_ES,
                    0x0e => SEG_CS,
                    0x16 => SEG_SS,
                    _ => SEG_DS,
                };
                self.push(sz, self.seg[index].sel as u64)?;
            }
            // pop es/ss/ds: the stack pointer is restored if the load faults
            0x07 | 0x17 | 0x1f => {
                let index = match op {
                    0x07 => SEG_ES,
                    0x17 => SEG_SS,
                    _ => SEG_DS,
                };
                self.pop_segment(index, sz)?;
            }
            0x1a0 | 0x1a8 => {
                let index = if op == 0x1a0 { SEG_FS } else { SEG_GS };
                self.push(sz, self.seg[index].sel as u64)?;
            }
            0x1a1 | 0x1a9 => {
                let index = if op == 0x1a1 { SEG_FS } else { SEG_GS };
                self.pop_segment(index, sz)?;
            }
            0x82 => {
                let mut alias = *i;
                alias.op = 0x80;
                self.exec(&alias)?;
                return Ok(true);
            }
            // mov sreg, r/m
            0x8e => {
                let index = (i.reg & 7) as usize;
                if index == SEG_CS || index > SEG_GS {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let selector = self.load_operand(loc, 2, i)? as u16;
                self.load_segment_register(index, selector)?;
                if index == SEG_SS {
                    self.interrupt_shadow = true;
                }
            }
            0x8f => {
                if i.reg & 7 != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let old_sp = self.gpr[GPR_RSP];
                let value = self.pop(sz)?;
                // A memory destination addressed through the stack pointer uses the value after
                // the pop.
                let result = self
                    .modrm_location(i, next)
                    .and_then(|loc| self.store_operand(loc, sz, i, value));
                if let Err(error) = result {
                    self.gpr[GPR_RSP] = old_sp;
                    return Err(error);
                }
            }
            0x9c => {
                if self.eflags & VM != 0 && self.iopl() < USER_CPL {
                    return Err(Exception::general_protection(0));
                }
                let flags = self.eflags_full() & !(RF | VM);
                self.push(sz, flags)?;
            }
            0x9d => {
                if self.eflags & VM != 0 && self.iopl() < USER_CPL {
                    return Err(Exception::general_protection(0));
                }
                let value = self.pop(sz)?;
                let mut loadable = self.loadable_eflags(self.cpl);
                if sz == 2 {
                    loadable &= WORD_MASK;
                }
                let current = self.eflags_full();
                self.eflags = ((current & !loadable) | (value & loadable)) | RESERVED_ONE;
                self.lazy_op = LazyFlagsOp::Eflags;
            }
            // BCD adjustments
            0x27 | 0x2f => self.decimal_adjust(op == 0x2f),
            0x37 | 0x3f => self.ascii_adjust(op == 0x3f),
            0xd4 | 0xd5 => {
                let al = self.gpr[GPR_RAX] & BYTE_MASK;
                let ah = (self.gpr[GPR_RAX] >> 8) & BYTE_MASK;
                let base = i.imm as u64 & BYTE_MASK;
                let (new_ah, new_al) = if op == 0xd4 {
                    if base == 0 {
                        return Err(Exception::divide_error());
                    }
                    (al / base, al % base)
                } else {
                    (0, (al + ah * base) & BYTE_MASK)
                };
                self.write_gpr(0, 2, true, (new_ah << 8) | new_al);
                self.set_lazy_flags(LazyFlagsOp::Logic, 1, new_al, 0, 0);
            }
            // salc
            0xd6 => {
                let value = if self.cf() { BYTE_MASK } else { 0 };
                self.write_gpr(0, 1, true, value);
            }
            0xce => {
                if self.eflags_full() & OF != 0 {
                    self.rip = next;
                    return Err(Exception::Fault {
                        vector: VECTOR_OVERFLOW,
                        error_code: None,
                        software: true,
                    });
                }
            }
            0x62 => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let Loc::Mem(address) = loc else {
                    return Err(Exception::invalid_opcode());
                };
                let width = osz as u64;
                let lower = self.read(address, osz)?;
                let upper = self.read(address.wrapping_add(width), osz)?;
                let index = self.read_gpr(i.reg, osz, true);
                let signed = |value: u64| match osz {
                    2 => value as u16 as i16 as i64,
                    _ => value as u32 as i32 as i64,
                };
                if signed(index) < signed(lower) || signed(index) > signed(upper) {
                    return Err(Exception::fault(VECTOR_BOUND_RANGE, None));
                }
            }
            // arpl r/m16, r16
            0x63 => {
                if !self.segments_are_protected() {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let destination = self.load_operand(loc, 2, i)?;
                let source = self.read_gpr(i.reg, 2, true);
                let flags = self.eflags_full();
                if (destination & SELECTOR_RPL_MASK as u64) < (source & SELECTOR_RPL_MASK as u64) {
                    let adjusted = (destination & !(SELECTOR_RPL_MASK as u64))
                        | (source & SELECTOR_RPL_MASK as u64);
                    self.store_operand(loc, 2, i, adjusted)?;
                    self.set_eflags_arith(flags | ZF);
                } else {
                    self.set_eflags_arith(flags & !ZF);
                }
            }
            // call / jmp ptr16:16/32
            0x9a | 0xea => return self.far_branch_pointer(i, next).map(|()| true),
            // les / lds, lss / lfs / lgs
            0xc4 | 0xc5 | 0x1b2 | 0x1b4 | 0x1b5 => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let index = match op {
                    0xc4 => SEG_ES,
                    0xc5 => SEG_DS,
                    0x1b2 => SEG_SS,
                    0x1b4 => SEG_FS,
                    _ => SEG_GS,
                };
                let loc = self.modrm_location(i, next)?;
                let Loc::Mem(address) = loc else {
                    return Err(Exception::invalid_opcode());
                };
                let offset = self.read(address, osz)?;
                let selector = self.read(address.wrapping_add(osz as u64), 2)? as u16;
                self.load_segment_register(index, selector)?;
                if index == SEG_SS {
                    self.interrupt_shadow = true;
                }
                self.write_gpr(i.reg, osz, true, offset);
            }
            0xc2 | 0xc3 => {
                let extra = if op == 0xc2 {
                    i.imm as u64 & WORD_MASK
                } else {
                    0
                };
                let sp = self.stack_pointer();
                let linear = self.stack_linear(sp)?;
                let target = self.read_stack(linear, osz)?;
                self.branch_near(i, target)?;
                let new_sp = sp.wrapping_add(osz as u64).wrapping_add(extra) & self.stack_mask();
                self.set_stack_pointer(new_sp);
                return Ok(true);
            }
            0xc8 => self.enter_legacy(i, sz)?,
            0xc9 => {
                let frame = self.gpr[GPR_RBP] & self.stack_mask();
                let linear = self.stack_linear(frame)?;
                let value = self.read_stack(linear, sz)?;
                self.set_stack_pointer(frame.wrapping_add(sz as u64) & self.stack_mask());
                self.write_gpr(GPR_RBP as u8, sz, true, value);
            }
            0xca | 0xcb => {
                let extra = if op == 0xca {
                    i.imm as u64 & WORD_MASK
                } else {
                    0
                };
                return self.far_return_legacy(i, extra).map(|()| true);
            }
            0xcf => {
                let result = self.iret_legacy(i, next);
                if result.is_ok() {
                    self.nmi_blocked = false;
                }
                return result.map(|()| true);
            }
            0xd7 => {
                let offset =
                    self.gpr[GPR_RBX].wrapping_add(self.gpr[GPR_RAX] & BYTE_MASK) & address_mask(i);
                let linear = self.linear_address(i, offset)?;
                let value = self.read(linear, 1)?;
                self.write_gpr(0, 1, true, value);
            }
            // loopne, loope, loop, jcxz / jecxz
            0xe0..=0xe3 => {
                let amask = address_mask(i);
                let take = if op == 0xe3 {
                    self.gpr[GPR_RCX] & amask == 0
                } else {
                    let count = self.gpr[GPR_RCX].wrapping_sub(1) & amask;
                    self.write_address_register(GPR_RCX, count, i);
                    let zf = self.eflags_full() & ZF != 0;
                    count != 0
                        && match op {
                            0xe0 => !zf,
                            0xe1 => zf,
                            _ => true,
                        }
                };
                if take {
                    return self
                        .branch_near(i, next.wrapping_add(i.imm as u64))
                        .map(|()| true);
                }
            }
            0x70..=0x7f | 0x180..=0x18f => {
                if self.condition_holds((op & 15) as u8) {
                    return self
                        .branch_near(i, next.wrapping_add(i.imm as u64))
                        .map(|()| true);
                }
            }
            0xe9 | 0xeb => {
                return self
                    .branch_near(i, next.wrapping_add(i.imm as u64))
                    .map(|()| true);
            }
            0xe8 => {
                self.push(osz, next & mask(osz))?;
                return self
                    .branch_near(i, next.wrapping_add(i.imm as u64))
                    .map(|()| true);
            }
            0xff => match i.reg & 7 {
                2 => {
                    let loc = self.modrm_location(i, next)?;
                    let target = self.load_operand(loc, osz, i)?;
                    self.push(osz, next & mask(osz))?;
                    return self.branch_near(i, target).map(|()| true);
                }
                4 => {
                    let loc = self.modrm_location(i, next)?;
                    let target = self.load_operand(loc, osz, i)?;
                    return self.branch_near(i, target).map(|()| true);
                }
                3 | 5 => return self.far_branch_indirect(i, next).map(|()| true),
                6 => {
                    let loc = self.modrm_location(i, next)?;
                    let value = self.load_operand(loc, sz, i)?;
                    self.push(sz, value)?;
                }
                _ => return Ok(false),
            },
            0xa4..=0xa7 | 0xaa..=0xaf => return self.string_op_legacy(i, next).map(|()| true),
            0x6c..=0x6f => return self.string_io(i, next).map(|()| true),
            _ => return Ok(false),
        }
        self.rip = next;
        Ok(true)
    }

    /// pop into a segment register, restoring the stack pointer if the load faults.
    fn pop_segment(&mut self, index: usize, size: u8) -> ExecResult<()> {
        let old_sp = self.gpr[GPR_RSP];
        let selector = self.pop(size)? as u16;
        if let Err(error) = self.load_segment_register(index, selector) {
            self.gpr[GPR_RSP] = old_sp;
            return Err(error);
        }
        if index == SEG_SS {
            self.interrupt_shadow = true;
        }
        Ok(())
    }

    fn enter_legacy(&mut self, i: &Insn, size: u8) -> ExecResult<()> {
        let width = size as u64;
        let stack_mask = self.stack_mask();
        let level = i.imm2 & ENTER_LEVEL_MASK;
        let frame_base = self.read_gpr(GPR_RBP as u8, size, true);
        let mut sp = self.stack_pointer().wrapping_sub(width) & stack_mask;
        let linear = self.stack_linear(sp)?;
        self.write_stack(linear, size, frame_base)?;
        let frame_pointer = sp;
        if level > 0 {
            let mut outer = self.gpr[GPR_RBP] & stack_mask;
            for _ in 1..level {
                outer = outer.wrapping_sub(width) & stack_mask;
                let linear = self.stack_linear(outer)?;
                let enclosing = self.read_stack(linear, size)?;
                sp = sp.wrapping_sub(width) & stack_mask;
                let linear = self.stack_linear(sp)?;
                self.write_stack(linear, size, enclosing)?;
            }
            sp = sp.wrapping_sub(width) & stack_mask;
            let linear = self.stack_linear(sp)?;
            self.write_stack(linear, size, frame_pointer)?;
        }
        self.write_gpr(GPR_RBP as u8, size, true, frame_pointer);
        self.set_stack_pointer(sp.wrapping_sub(i.imm as u64 & WORD_MASK) & stack_mask);
        Ok(())
    }

    /// daa / das (`subtract` selects das). SDM: the final CF depends only on the original AL and CF.
    fn decimal_adjust(&mut self, subtract: bool) {
        let flags = self.eflags_full();
        let original = self.gpr[GPR_RAX] & BYTE_MASK;
        let old_carry = flags & CF != 0;
        let mut al = original;
        let aux = original & 0xf > BCD_NIBBLE_LIMIT || flags & AF != 0;
        if aux {
            al = if subtract {
                al.wrapping_sub(BCD_ADJUST_LOW)
            } else {
                al.wrapping_add(BCD_ADJUST_LOW)
            } & BYTE_MASK;
        }
        let carry = original > BCD_BYTE_LIMIT || old_carry;
        if carry {
            al = if subtract {
                al.wrapping_sub(BCD_ADJUST_HIGH)
            } else {
                al.wrapping_add(BCD_ADJUST_HIGH)
            } & BYTE_MASK;
        }
        self.write_gpr(0, 1, true, al);
        let mut result = szp_flags(al, 1);
        if carry {
            result |= CF;
        }
        if aux {
            result |= AF;
        }
        self.set_eflags_arith(result);
    }

    /// aaa / aas (`subtract` selects aas).
    fn ascii_adjust(&mut self, subtract: bool) {
        const NIBBLE_MASK: u64 = 0xf;
        const AH_STEP: u64 = 0x100;
        const AL_NIBBLE_KEPT: u64 = 0xff0f;
        let flags = self.eflags_full();
        let ax = self.gpr[GPR_RAX] & WORD_MASK;
        let adjust = ax & NIBBLE_MASK > BCD_NIBBLE_LIMIT || flags & AF != 0;
        let adjusted = if !adjust {
            ax
        } else if subtract {
            ax.wrapping_sub(BCD_ADJUST_LOW).wrapping_sub(AH_STEP)
        } else {
            ax.wrapping_add(BCD_ADJUST_LOW).wrapping_add(AH_STEP)
        };
        self.write_gpr(0, 2, true, adjusted & AL_NIBBLE_KEPT);
        let mut result = flags & !(CF | AF);
        if adjust {
            result |= CF | AF;
        }
        self.set_eflags_arith(result);
    }

    /// movs/cmps/stos/lods/scas outside 64-bit mode: SI/DI/CX by the address size, the source
    /// through DS (or the override) and the destination always through ES.
    fn string_op_legacy(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op as u8;
        let size = if op & 1 == 0 { 1 } else { i.opsize };
        let amask = address_mask(i);
        let step: u64 = if self.eflags & DF != 0 {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        let rep = i.rep != 0;
        let compares = matches!(op, 0xa6 | 0xa7 | 0xae | 0xaf);
        let source_segment = if i.seg != NO_SEG {
            i.seg as usize
        } else {
            SEG_DS
        };
        let mut iterations = 0;
        loop {
            if rep && self.gpr[GPR_RCX] & amask == 0 {
                break;
            }
            let source = self.gpr[GPR_RSI] & amask;
            let destination = self.gpr[GPR_RDI] & amask;
            match op {
                0xa4 | 0xa5 => {
                    let from = self.segment_access(source_segment, source, size, false)?;
                    let to = self.segment_access(SEG_ES, destination, size, true)?;
                    let value = self.read(from, size)?;
                    self.write(to, size, value)?;
                    self.write_address_register(GPR_RSI, source.wrapping_add(step), i);
                    self.write_address_register(GPR_RDI, destination.wrapping_add(step), i);
                }
                0xaa | 0xab => {
                    let to = self.segment_access(SEG_ES, destination, size, true)?;
                    let value = self.read_gpr(0, size, true);
                    self.write(to, size, value)?;
                    self.write_address_register(GPR_RDI, destination.wrapping_add(step), i);
                }
                0xac | 0xad => {
                    let from = self.segment_access(source_segment, source, size, false)?;
                    let value = self.read(from, size)?;
                    self.write_gpr(0, size, true, value);
                    self.write_address_register(GPR_RSI, source.wrapping_add(step), i);
                }
                0xa6 | 0xa7 => {
                    let from = self.segment_access(source_segment, source, size, false)?;
                    let to = self.segment_access(SEG_ES, destination, size, false)?;
                    let a = self.read(from, size)?;
                    let b = self.read(to, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.write_address_register(GPR_RSI, source.wrapping_add(step), i);
                    self.write_address_register(GPR_RDI, destination.wrapping_add(step), i);
                }
                _ => {
                    let to = self.segment_access(SEG_ES, destination, size, false)?;
                    let a = self.read_gpr(0, size, true);
                    let b = self.read(to, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.write_address_register(GPR_RDI, destination.wrapping_add(step), i);
                }
            }
            if !rep {
                break;
            }
            let count = self.gpr[GPR_RCX].wrapping_sub(1) & amask;
            self.write_address_register(GPR_RCX, count, i);
            iterations += 1;
            if compares {
                let zf = self.eflags_full() & ZF != 0;
                if (i.rep == REP_REPE && !zf) || (i.rep == REP_REPNE && zf) {
                    break;
                }
            }
            if iterations >= REP_ITERATIONS_PER_SLICE && count != 0 {
                self.icount += iterations;
                self.pmu.unretired += iterations + 1;
                return Ok(());
            }
        }
        self.icount += iterations;
        self.pmu.unretired += iterations;
        self.rip = next;
        Ok(())
    }

    /// ins / outs, in every mode: the port is DX, the memory side is ES:DI (ins) or DS:SI (outs).
    pub(super) fn string_io(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op as u8;
        let input = op < 0x6e;
        let size = if op & 1 == 0 {
            1
        } else if i.opsize == 2 {
            2
        } else {
            4
        };
        let amask = address_mask(i);
        let step: u64 = if self.eflags & DF != 0 {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        let rep = i.rep != 0;
        let port = self.gpr[GPR_RDX] as u16;
        let source_segment = if i.seg != NO_SEG {
            i.seg as usize
        } else {
            SEG_DS
        };
        let mut iterations = 0;
        loop {
            if rep && self.gpr[GPR_RCX] & amask == 0 {
                break;
            }
            self.check_io_permission(port, size)?;
            let now = self.now_ns();
            if input {
                let destination = self.gpr[GPR_RDI] & amask;
                let to = self.segment_access(SEG_ES, destination, size, true)?;
                let value = self.dev.lock().io_in(port, size, now);
                self.write(to, size, value as u64)?;
                self.write_address_register(GPR_RDI, destination.wrapping_add(step), i);
            } else {
                let source = self.gpr[GPR_RSI] & amask;
                let from = self.segment_access(source_segment, source, size, false)?;
                let value = self.read(from, size)? as u32;
                self.dev.lock().io_out(port, size, value, now);
                self.write_address_register(GPR_RSI, source.wrapping_add(step), i);
                if self.dev.lock().reset_requested {
                    self.rip = next;
                    return Err(Exception::Exit(ExitReason::Reset));
                }
                if self.dev.lock().shutdown_requested() {
                    self.rip = next;
                    return Err(Exception::Exit(ExitReason::Shutdown));
                }
            }
            if !rep {
                break;
            }
            let count = self.gpr[GPR_RCX].wrapping_sub(1) & amask;
            self.write_address_register(GPR_RCX, count, i);
            iterations += 1;
            if iterations >= REP_ITERATIONS_PER_SLICE && count != 0 {
                self.icount += iterations;
                self.pmu.unretired += iterations + 1;
                return Ok(());
            }
        }
        self.icount += iterations;
        self.pmu.unretired += iterations;
        self.rip = next;
        Ok(())
    }
}
