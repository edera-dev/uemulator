use super::decode::*;
use super::exception::{Exception, VECTOR_SEGMENT_NOT_PRESENT, VECTOR_STACK_FAULT};
use super::flags::*;
use super::mem::AccessKind;
use super::sys::{CR0_TS, CR4_TSD};
use super::*;

// Indices into `Cpu::gpr` (architectural register numbers); GPR_RSP and GPR_RSI come from mod.rs.
const GPR_RAX: usize = 0;
const GPR_RCX: usize = 1;
const GPR_RDX: usize = 2;
const GPR_RBX: usize = 3;
const GPR_RBP: usize = 5;
const GPR_RDI: usize = 7;

/// Register numbers are 4 bits wide once REX extends the ModRM/opcode field.
const GPR_INDEX_MASK: usize = 15;

/// Without a REX prefix, byte-register numbers 4..=7 mean AH, CH, DH, BH (bits 15:8 of
/// registers 0..=3) instead of SPL, BPL, SIL, DIL.
const BYTE_REG_AH: u8 = 4;
const BYTE_REG_BH: u8 = 7;

const BYTE_MASK: u64 = 0xff;
const WORD_MASK: u64 = 0xffff;
const DWORD_MASK: u64 = 0xffff_ffff;

const REX_B: u8 = 1;

/// `Insn::rep` values: F3 is rep/repe, F2 is repne.
const REP_REPE: u8 = 1;
const REP_REPNE: u8 = 2;

/// A rep string instruction returns to the run loop after moving this many bytes (RIP unchanged,
/// so it resumes where it left off) so pending interrupts and device events are still serviced.
/// Measured in bytes, not iterations: with ERMS/FSRM the kernel's memcpy and clear_page are
/// `rep movsb`/`rep stosb`, and counting elements gave them an eighth of the bytes per slice of
/// the `rep movsq` they replace, which is eight times the run-loop round trips.
const REP_BYTES_PER_SLICE: u64 = 32768;

const VECTOR_BREAKPOINT: u8 = 3;

/// The low two bits of a selector are the RPL; the rest (index and TI) identify the descriptor.
const SELECTOR_RPL_MASK: u16 = 3;

const EFLAGS_ID: u64 = 1 << 21;
const IOPL_SHIFT: u32 = 12;

/// EFLAGS bits popf/iretq load at any CPL. IF is added when CPL <= IOPL and IOPL when CPL is 0
/// (see `Cpu::loadable_eflags`). RF and VM are never loaded by either.
const EFLAGS_ALWAYS_LOADABLE: u64 = CF | PF | AF | ZF | SF | TF | DF | OF | NT | AC | EFLAGS_ID;

/// sahf/lahf transfer these flags (SF, ZF, AF, PF, CF) to and from AH.
const LAHF_SAHF_FLAGS: u64 = SF | ZF | AF | PF | CF;

/// Shift/rotate counts are masked to 5 bits, or 6 bits for 64-bit operands.
const SHIFT_COUNT_MASK: u64 = 0x1f;
const SHIFT_COUNT_MASK_64BIT: u64 = 0x3f;

/// Offset of the 16-bit I/O permission bitmap base in a 64-bit TSS.
const TSS64_IO_MAP_BASE_OFFSET: u64 = 102;

/// ENTER takes its nesting level modulo 32.
const ENTER_LEVEL_MASK: u16 = 31;

/// `log_once` key space for unimplemented opcodes; other callers (mem.rs, sys.rs) use other bases.
const UNIMPLEMENTED_OPCODE_LOG_BASE: u64 = 0x4000_0000;

// ALU operation numbers: bits 5:3 of opcodes 0x00..=0x3f and ModRM.reg of group 1 (0x80/81/83).
// 6 is xor and is the fallthrough arm of `alu`.
const ALU_ADD: u8 = 0;
const ALU_OR: u8 = 1;
const ALU_ADC: u8 = 2;
const ALU_SBB: u8 = 3;
const ALU_AND: u8 = 4;
const ALU_SUB: u8 = 5;
const ALU_CMP: u8 = 7;

// ModRM.reg of the shift/rotate group (0xc0/c1/d0..d3). /6 is an undocumented alias of SHL and
// /7 (sar) is the fallthrough arm of `shift_rotate`.
const SHIFT_ROL: u8 = 0;
const SHIFT_ROR: u8 = 1;
const SHIFT_RCL: u8 = 2;
const SHIFT_RCR: u8 = 3;
const SHIFT_SHL: u8 = 4;
const SHIFT_SHR: u8 = 5;
const SHIFT_SAL_ALIAS: u8 = 6;

// ModRM.reg of group 3 (0xf6/f7); /1 is an alias of /0 (test), /7 is idiv.
const GROUP3_TEST: u8 = 0;
const GROUP3_TEST_ALIAS: u8 = 1;
const GROUP3_NOT: u8 = 2;
const GROUP3_NEG: u8 = 3;
const GROUP3_MUL: u8 = 4;
const GROUP3_IMUL: u8 = 5;
const GROUP3_DIV: u8 = 6;

// ModRM.reg of groups 4/5 (0xfe/0xff).
const GROUP5_INC: u8 = 0;
const GROUP5_DEC: u8 = 1;
const GROUP5_CALL_NEAR: u8 = 2;
const GROUP5_JMP_NEAR: u8 = 4;
const GROUP5_PUSH: u8 = 6;

// Bit-test family: bt, bts, btr, btc.
const BIT_TEST: u8 = 0;
const BIT_SET: u8 = 1;
const BIT_RESET: u8 = 2;
const BIT_COMPLEMENT: u8 = 3;
/// Group 8 (0f ba): ModRM.reg 4..=7 are bt/bts/btr/btc with an imm8 bit offset.
const GROUP8_FIRST_BIT_OP: u8 = 4;

/// Operand location after ModRM decoding. Memory addresses are linear (segment base applied).
#[derive(Clone, Copy)]
pub(crate) enum Loc {
    Reg(u8),
    Mem(u64),
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// `log_once` key base of `check_locked_instruction_was_atomic`; the opcode is added.
const LOG_KEY_LOCKED_NOT_ATOMIC: u64 = 1 << 58;

/// The flag state `Cpu::save_flags` captures: `eflags` plus the deferred-arithmetic fields.
#[derive(Clone, Copy)]
struct FlagsSnapshot {
    eflags: u64,
    lazy_op: LazyFlagsOp,
    lazy_size: u8,
    lazy_dst: u64,
    lazy_src: u64,
    lazy_src2: u64,
}

/// A null selector has index 0 and TI 0 whatever its RPL. A selector with TI set and index 0
/// refers to LDT entry 0 and is not null, so TI stays part of the comparison.
fn selector_is_null(sel: u16) -> bool {
    sel & !SELECTOR_RPL_MASK == 0
}

/// Descriptor type bit 10 of a code segment: conforming (callable from less privileged code and
/// exempt from the DPL checks).
pub(super) const SEG_TYPE_CONFORMING: u32 = 1 << 10;

fn is_conforming_code(seg: &Seg) -> bool {
    seg.is_code() && seg.flags & SEG_TYPE_CONFORMING != 0
}

/// Error code for a selector-related fault: the selector with RPL cleared (EXT bit 0).
fn selector_error_code(sel: u16) -> u32 {
    (sel & !SELECTOR_RPL_MASK) as u32
}

/// Operand size of push/pop/pushf/popf in 64-bit mode: 8 bytes, or 2 with a 66 prefix. REX.W
/// is irrelevant and 4 is not encodable, so `Insn::opsize` must not be used for these.
fn stack_operand_size(i: &Insn) -> u8 {
    match (i.mode, i.p66) {
        (CodeMode::Long64, false) => 8,
        (CodeMode::Long64, true) => 2,
        (CodeMode::Bits32, false) | (CodeMode::Bits16, true) => 4,
        (CodeMode::Bits32, true) | (CodeMode::Bits16, false) => 2,
    }
}

/// Register number embedded in the low opcode bits (push/pop/mov imm/xchg/bswap), extended by REX.B.
fn opcode_register(i: &Insn) -> u8 {
    (i.op as u8 & 7) | ((i.rex & REX_B) << 3)
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

fn shift_count_mask(size: u8) -> u64 {
    if size == 8 {
        SHIFT_COUNT_MASK_64BIT
    } else {
        SHIFT_COUNT_MASK
    }
}

/// Result and carry of shld (`left`) or shrd for a nonzero `count` already masked to the
/// operand's range, with `a` the destination and `b` the source.
///
/// A 16-bit operand with a count of 17..=31 is undefined in the SDM. Intel hardware shifts the
/// 48-bit value A:B:A, which is what this does (QEMU's `gen_shiftd_rm_T1` implements the same); it
/// also gives the ordinary result for counts up to 16, and keeps `bits - count` from underflowing.
fn double_shift(left: bool, a: u64, b: u64, count: u32, size: u8) -> (u64, u64) {
    let m = mask(size);
    if size == 2 {
        // Bits of the shifted value above bit 63 are lost, but only bits 47:32 (left) or 15:0
        // (right) of the result are used.
        let wide = (a << 32) | (b << 16) | a;
        return if left {
            (
                ((wide << count) >> 32) & m,
                ((wide << (count - 1)) >> 47) & 1,
            )
        } else {
            ((wide >> count) & m, (wide >> (count - 1)) & 1)
        };
    }
    let bits = size as u32 * 8;
    if left {
        (
            ((a << count) | (b >> (bits - count))) & m,
            (a >> (bits - count)) & 1,
        )
    } else {
        (
            ((a >> count) | (b << (bits - count))) & m,
            (a >> (count - 1)) & 1,
        )
    }
}

/// Joins EDX:EAX style halves (wrmsr, cmpxchg8b); the upper 32 bits of both inputs are ignored.
fn combine_dwords(low: u64, high: u64) -> u64 {
    (low & DWORD_MASK) | (high << 32)
}

impl Cpu {
    /// Reads a general register at the given operand size. With `rex` false, byte registers 4..=7
    /// are AH/CH/DH/BH.
    pub fn read_gpr(&self, r: u8, size: u8, rex: bool) -> u64 {
        let g = self.gpr[r as usize & GPR_INDEX_MASK];
        match size {
            1 => {
                if !rex && (BYTE_REG_AH..=BYTE_REG_BH).contains(&r) {
                    (self.gpr[(r - BYTE_REG_AH) as usize] >> 8) & BYTE_MASK
                } else {
                    g & BYTE_MASK
                }
            }
            2 => g & WORD_MASK,
            4 => g & DWORD_MASK,
            _ => g,
        }
    }

    /// Writes a general register. 8- and 16-bit writes merge into the old value; a 32-bit write
    /// zero-extends into the full register (64-bit mode rule). `rex` as in `read_gpr`.
    pub fn write_gpr(&mut self, r: u8, size: u8, rex: bool, v: u64) {
        match size {
            1 => {
                if !rex && (BYTE_REG_AH..=BYTE_REG_BH).contains(&r) {
                    let g = &mut self.gpr[(r - BYTE_REG_AH) as usize];
                    *g = (*g & !(BYTE_MASK << 8)) | ((v & BYTE_MASK) << 8);
                } else {
                    let g = &mut self.gpr[r as usize];
                    *g = (*g & !BYTE_MASK) | (v & BYTE_MASK);
                }
            }
            2 => {
                let g = &mut self.gpr[r as usize];
                *g = (*g & !WORD_MASK) | (v & WORD_MASK);
            }
            4 => self.gpr[r as usize] = v & DWORD_MASK,
            _ => self.gpr[r as usize] = v,
        }
    }

    pub(super) fn set_edx_eax(&mut self, v: u64) {
        self.gpr[GPR_RAX] = v & DWORD_MASK;
        self.gpr[GPR_RDX] = v >> 32;
    }

    /// cmpxchg16b: compares RDX:RAX with the 16 bytes at `address` and either stores RCX:RBX (ZF
    /// set) or loads the memory into RDX:RAX (ZF clear). The operand must be 16-byte aligned
    /// (#GP(0), checked before any access). Hardware always performs the write cycle, so the
    /// failing case writes the old value back to take a write fault on a read-only page.
    fn cmpxchg16b(&mut self, address: u64, next: u64) -> ExecResult<()> {
        const OPERAND_ALIGNMENT: u64 = 16;
        if address & (OPERAND_ALIGNMENT - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        if self.rmw == Rmw::Armed {
            return self.locked_cmpxchg16b(address, next);
        }
        let current = self.read128(address)?;
        let expected = ((self.gpr[GPR_RDX] as u128) << 64) | self.gpr[GPR_RAX] as u128;
        let f = self.eflags_full();
        if current == expected {
            let replacement = ((self.gpr[GPR_RCX] as u128) << 64) | self.gpr[GPR_RBX] as u128;
            self.write128(address, replacement)?;
            self.set_eflags_arith(f | ZF);
        } else {
            self.write128(address, current)?;
            self.gpr[GPR_RAX] = current as u64;
            self.gpr[GPR_RDX] = (current >> 64) as u64;
            self.set_eflags_arith(f & !ZF);
        }
        self.rip = next;
        Ok(())
    }

    /// `cmpxchg16b` with a LOCK prefix on a vCPU that shares memory: one 16-byte host atomic.
    fn locked_cmpxchg16b(&mut self, address: u64, next: u64) -> ExecResult<()> {
        const OPERAND_BYTES: u64 = 16;
        let physical = self.translate(address, mem::AccessKind::Write, false)?;
        let host = unsafe { self.ram.base().add(physical as usize) };
        if !self.ram.contains(physical, OPERAND_BYTES) || host as usize & 15 != 0 {
            self.rmw = Rmw::NeedsExclusive;
            return Err(Exception::AtomicRetry);
        }
        let expected = ((self.gpr[GPR_RDX] as u128) << 64) | self.gpr[GPR_RAX] as u128;
        let replacement = ((self.gpr[GPR_RCX] as u128) << 64) | self.gpr[GPR_RBX] as u128;
        let Some(outcome) = (unsafe { atomic::compare_exchange_u128(host, expected, replacement) })
        else {
            self.rmw = Rmw::NeedsExclusive;
            return Err(Exception::AtomicRetry);
        };
        let f = self.eflags_full();
        match outcome {
            Ok(()) => {
                let page = (physical >> PAGE_SHIFT) as usize;
                if self.page_has_code.get(page) != 0 {
                    self.code_written(page);
                }
                self.set_eflags_arith(f | ZF);
            }
            Err(found) => {
                self.gpr[GPR_RAX] = found as u64;
                self.gpr[GPR_RDX] = (found >> 64) as u64;
                self.set_eflags_arith(f & !ZF);
            }
        }
        self.rip = next;
        Ok(())
    }

    pub(super) fn iopl(&self) -> u8 {
        ((self.eflags & IOPL) >> IOPL_SHIFT) as u8
    }

    /// EFLAGS bits that popf/iretq may change when executed at `cpl`.
    pub(super) fn loadable_eflags(&self, cpl: u8) -> u64 {
        let mut loadable = EFLAGS_ALWAYS_LOADABLE;
        if cpl <= self.iopl() {
            loadable |= IF;
        }
        if cpl == 0 {
            loadable |= IOPL;
        }
        loadable
    }

    /// Effective address (offset within the segment), truncated for the 67 prefix. RIP-relative
    /// displacements are relative to the end of the whole instruction, immediate included, which
    /// is what `next` is.
    pub fn effective_address(&self, i: &Insn, next: u64) -> u64 {
        let mut a = i.disp as u64;
        if i.rip_rel {
            a = a.wrapping_add(next);
        } else {
            if i.base != NO_REG {
                a = a.wrapping_add(self.gpr[i.base as usize]);
            }
            if i.index != NO_REG {
                a = a.wrapping_add(self.gpr[i.index as usize] << i.scale);
            }
        }
        if i.addr32 {
            a &= DWORD_MASK;
        } else if i.addr16 {
            a &= WORD_MASK;
        }
        a
    }

    /// Segment-relative offset `a` to a linear address: the base and limit of the segment the
    /// instruction names (override prefix, else SS for `bp`/`esp`-based operands, else DS). In
    /// 64-bit mode only FS and GS have a base; ES/CS/SS/DS overrides are ignored. The limit is
    /// checked for the first byte only, so an access that starts inside the limit and ends past
    /// it is not caught.
    pub fn linear_address(&self, i: &Insn, a: u64) -> ExecResult<u64> {
        let seg = if i.seg != NO_SEG {
            i.seg as usize
        } else if i.ss_default {
            SEG_SS
        } else {
            SEG_DS
        };
        if !self.is_long64() {
            self.pending_operand.set(Some((seg as u8, a)));
        }
        self.segment_linear(seg, a)
    }

    pub fn modrm_location(&self, i: &Insn, next: u64) -> ExecResult<Loc> {
        if i.md == 3 {
            Ok(Loc::Reg(i.rm))
        } else {
            Ok(Loc::Mem(
                self.linear_address(i, self.effective_address(i, next))?,
            ))
        }
    }

    pub fn load_operand(&mut self, loc: Loc, size: u8, i: &Insn) -> ExecResult<u64> {
        match loc {
            Loc::Reg(r) => Ok(self.read_gpr(r, size, i.rex_present)),
            Loc::Mem(a) => {
                if self.rmw == Rmw::Armed {
                    return self.locked_load(a, size);
                }
                self.check_operand_extent(size, false)?;
                self.read(a, size)
            }
        }
    }

    pub fn store_operand(&mut self, loc: Loc, size: u8, i: &Insn, v: u64) -> ExecResult<()> {
        match loc {
            Loc::Reg(r) => {
                self.write_gpr(r, size, i.rex_present, v);
                Ok(())
            }
            Loc::Mem(a) => {
                if let Rmw::Loaded {
                    host,
                    size: loaded_size,
                    value,
                    page,
                } = self.rmw
                    && loaded_size == size
                {
                    return self.locked_store(host, size, value, page, v);
                }
                self.check_operand_extent(size, true)?;
                self.write(a, size, v)
            }
        }
    }

    /// Whether `exec` has to run this instruction as an atomic read-modify-write: it carries a
    /// LOCK prefix, or it is `xchg` with a memory operand, which is locked implicitly (SDM
    /// vol. 2, XCHG).
    fn is_locked(i: &Insn) -> bool {
        i.lock || (matches!(i.op, 0x86 | 0x87) && i.md != 3)
    }

    /// The load half of a locked instruction: translates for writing (a locked access faults as a
    /// write even when it would not change anything), then reads the operand atomically and
    /// remembers where and what, for `locked_store`.
    fn locked_load(&mut self, linear: u64, size: u8) -> ExecResult<u64> {
        self.check_operand_extent(size, true)?;
        if (linear & PAGE_OFFSET_MASK) + size as u64 > PAGE_SIZE {
            self.rmw = Rmw::NeedsExclusive;
            return Err(Exception::AtomicRetry);
        }
        let physical = self.translate(linear, mem::AccessKind::Write, false)?;
        if !self.ram.contains(physical, size as u64) {
            // Device memory: there is nothing to make atomic, and the device lock serializes it.
            self.rmw = Rmw::Off;
            return self.read(linear, size);
        }
        let host = unsafe { self.ram.base().add(physical as usize) };
        if !atomic::is_atomic_operand(host, size) {
            self.rmw = Rmw::NeedsExclusive;
            return Err(Exception::AtomicRetry);
        }
        let value = unsafe { atomic::load(host, size) };
        self.rmw = Rmw::Loaded {
            host: host as usize,
            size,
            value,
            page: (physical >> PAGE_SHIFT) as usize,
        };
        Ok(value)
    }

    /// The store half: replaces the operand only if it still holds what `locked_load` saw. When
    /// another vCPU got there first the instruction is run again from its start.
    fn locked_store(
        &mut self,
        host: usize,
        size: u8,
        expected: u64,
        page: usize,
        new: u64,
    ) -> ExecResult<()> {
        self.rmw = Rmw::Off;
        if unsafe { atomic::compare_exchange(host as *mut u8, size, expected, new) }.is_err() {
            return Err(Exception::AtomicRetry);
        }
        // The compare-and-swap bypasses the store path that notices writes to code pages.
        if self.page_has_code.get(page) != 0 {
            self.code_written(page);
        }
        Ok(())
    }

    /// Runs a locked instruction to completion however many times other vCPUs make it start
    /// over. Registers and flags are restored before each attempt because the instruction
    /// updates some of them (xadd's source register, the arithmetic flags) before its store.
    fn exec_locked(&mut self, i: &Insn) -> ExecResult<()> {
        let registers = self.gpr;
        let flags = self.save_flags();
        loop {
            self.rmw = Rmw::Armed;
            let result = self.exec(i);
            let state = core::mem::replace(&mut self.rmw, Rmw::Off);
            if !matches!(result, Err(Exception::AtomicRetry)) {
                if result.is_ok() {
                    self.check_locked_instruction_was_atomic(i, state);
                }
                return result;
            }
            self.gpr = registers;
            self.restore_flags(&flags);
            if state == Rmw::NeedsExclusive {
                // Split locks and misaligned operands: no host atomic exists, so stop everyone
                // else and do it with plain accesses. `Plain` keeps `exec` from taking this
                // path again from inside the section, which would wait for itself.
                return self.with_exclusive(|cpu| {
                    cpu.rmw = Rmw::Plain;
                    let result = cpu.exec(i);
                    cpu.rmw = Rmw::Off;
                    result
                });
            }
        }
    }

    /// A locked instruction with a memory operand that finished with its read-modify-write still
    /// pending bypassed `load_operand`/`store_operand`, so it was not atomic: some form that
    /// reads or writes its destination another way. Reported rather than silently racy.
    fn check_locked_instruction_was_atomic(&mut self, i: &Insn, state: Rmw) {
        let bypassed = match state {
            Rmw::Armed => i.md != 3,
            Rmw::Loaded { .. } => true,
            _ => false,
        };
        if !bypassed {
            return;
        }
        debug_assert!(
            false,
            "locked op {:#x} did not go through the atomic path",
            i.op
        );
        self.log_once(LOG_KEY_LOCKED_NOT_ATOMIC + i.op as u64, || {
            format!("locked instruction {:#x} is not atomic here", i.op)
        });
    }

    fn save_flags(&self) -> FlagsSnapshot {
        FlagsSnapshot {
            eflags: self.eflags,
            lazy_op: self.lazy_op,
            lazy_size: self.lazy_size,
            lazy_dst: self.lazy_dst,
            lazy_src: self.lazy_src,
            lazy_src2: self.lazy_src2,
        }
    }

    /// Stores the result of a read-modify-write instruction that has already updated the flags.
    /// If the store faults the flags are put back: the instruction is restarted after the fault
    /// and adc/sbb/rcl/rcr would otherwise read the carry the failed attempt left behind.
    fn store_result(
        &mut self,
        loc: Loc,
        size: u8,
        i: &Insn,
        v: u64,
        saved: FlagsSnapshot,
    ) -> ExecResult<()> {
        let result = self.store_operand(loc, size, i, v);
        if result.is_err() {
            self.restore_flags(&saved);
        }
        result
    }

    fn restore_flags(&mut self, saved: &FlagsSnapshot) {
        self.eflags = saved.eflags;
        self.lazy_op = saved.lazy_op;
        self.lazy_size = saved.lazy_size;
        self.lazy_dst = saved.lazy_dst;
        self.lazy_src = saved.lazy_src;
        self.lazy_src2 = saved.lazy_src2;
    }

    /// GPR_RSP is only updated after the store succeeds so a fault leaves it unchanged.
    pub fn push(&mut self, size: u8, v: u64) -> ExecResult<()> {
        if !self.is_long64() {
            return self.push_legacy(size, v);
        }
        let sp = self.gpr[GPR_RSP].wrapping_sub(size as u64);
        self.write_stack(sp, size, v)?;
        self.gpr[GPR_RSP] = sp;
        Ok(())
    }

    pub fn pop(&mut self, size: u8) -> ExecResult<u64> {
        if !self.is_long64() {
            return self.pop_legacy(size);
        }
        let v = self.read_stack(self.gpr[GPR_RSP], size)?;
        self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_add(size as u64);
        Ok(v)
    }

    /// Evaluates a 4-bit condition code (jcc/setcc/cmovcc low opcode nibble). Bits 3:1 pick the
    /// condition and bit 0 negates it.
    pub fn condition_holds(&self, cc: u8) -> bool {
        let f = self.eflags_full();
        let sf = f & SF != 0;
        let of = f & OF != 0;
        let r = match cc >> 1 {
            0 => of,                      // o
            1 => f & CF != 0,             // b
            2 => f & ZF != 0,             // e
            3 => f & (CF | ZF) != 0,      // be
            4 => sf,                      // s
            5 => f & PF != 0,             // p
            6 => sf != of,                // l
            _ => f & ZF != 0 || sf != of, // le
        };
        r ^ (cc & 1 == 1)
    }

    /// Two-operand ALU op. CMP computes flags like SUB; the caller decides not to store it.
    fn alu(&mut self, operation: u8, a: u64, b: u64, size: u8) -> u64 {
        let m = mask(size);
        let (a, b) = (a & m, b & m);
        match operation {
            ALU_ADD => {
                let r = a.wrapping_add(b) & m;
                self.set_lazy_flags(LazyFlagsOp::Add, size, r, b, 0);
                r
            }
            ALU_OR => {
                let r = a | b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
            ALU_ADC => {
                let carry = self.cf() as u64;
                let r = a.wrapping_add(b).wrapping_add(carry) & m;
                self.set_lazy_flags(LazyFlagsOp::Adc, size, r, b, carry);
                r
            }
            ALU_SBB => {
                let carry = self.cf() as u64;
                let r = a.wrapping_sub(b).wrapping_sub(carry) & m;
                self.set_lazy_flags(LazyFlagsOp::Sbb, size, r, b, carry);
                r
            }
            ALU_AND => {
                let r = a & b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
            ALU_SUB | ALU_CMP => {
                let r = a.wrapping_sub(b) & m;
                self.set_lazy_flags(LazyFlagsOp::Sub, size, r, b, 0);
                r
            }
            _ => {
                let r = a ^ b;
                self.set_lazy_flags(LazyFlagsOp::Logic, size, r, 0, 0);
                r
            }
        }
    }

    /// Shifts and rotates. A count that is zero after masking changes neither the value nor any
    /// flag. OF is computed for every count, not only count 1 where the SDM defines it.
    fn shift_rotate(&mut self, operation: u8, value: u64, count: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let m = mask(size);
        let count = (count & shift_count_mask(size)) as u32;
        if count == 0 {
            return value & m;
        }
        let a = value & m;
        let fl = self.eflags_full();
        let msb = |v: u64| (v >> (bits - 1)) & 1;
        let second_msb = |v: u64| (v >> (bits - 2)) & 1;
        // (result, CF, OF, whether SF/ZF/PF come from the result)
        let (r, cf, of, szp) = match operation {
            SHIFT_ROL => {
                // Counts above the width wrap, but CF/OF are still updated when count % bits == 0.
                let rotate = count % bits;
                let r = if rotate == 0 {
                    a
                } else {
                    ((a << rotate) | (a >> (bits - rotate))) & m
                };
                let cf = r & 1;
                (r, cf, msb(r) ^ cf, false)
            }
            SHIFT_ROR => {
                let rotate = count % bits;
                let r = if rotate == 0 {
                    a
                } else {
                    ((a >> rotate) | (a << (bits - rotate))) & m
                };
                (r, msb(r), msb(r) ^ second_msb(r), false)
            }
            SHIFT_RCL => {
                // Rotating through CF is a cycle of bits + 1 positions.
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..count % (bits + 1) {
                    let out = msb(r);
                    r = ((r << 1) | cf) & m;
                    cf = out;
                }
                (r, cf, msb(r) ^ cf, false)
            }
            SHIFT_RCR => {
                let mut r = a;
                let mut cf = fl & CF;
                for _ in 0..count % (bits + 1) {
                    let out = r & 1;
                    r = (r >> 1) | (cf << (bits - 1));
                    cf = out;
                }
                (r, cf, msb(r) ^ second_msb(r), false)
            }
            SHIFT_SHL | SHIFT_SAL_ALIAS => {
                let r = if count >= bits { 0 } else { (a << count) & m };
                let cf = if count <= bits {
                    (a >> (bits - count)) & 1
                } else {
                    0
                };
                (r, cf, msb(r) ^ cf, true)
            }
            SHIFT_SHR => {
                let r = if count >= bits { 0 } else { a >> count };
                let cf = if count <= bits {
                    (a >> (count - 1)) & 1
                } else {
                    0
                };
                (r, cf, msb(a), true)
            }
            _ => {
                // SAR: shift the sign-extended value so counts above the width fill with the sign.
                let signed = sign_extend(a, bits);
                let r = (signed >> count.min(63)) as u64 & m;
                let cf = ((signed >> (count - 1).min(63)) & 1) as u64;
                (r, cf, 0, true)
            }
        };
        let mut f = if szp {
            szp_flags(r, size)
        } else {
            fl & (PF | AF | ZF | SF)
        };
        if cf != 0 {
            f |= CF;
        }
        if of != 0 {
            f |= OF;
        }
        self.set_eflags_arith(f);
        r
    }

    /// Group 3 mul/imul/div/idiv (one-operand forms). Byte forms use AX as the wide register;
    /// the others use rDX:rAX. Division raises #DE for a zero divisor and for a quotient that does
    /// not fit the destination.
    fn mul_div(&mut self, i: &Insn, next: u64, size: u8) -> ExecResult<()> {
        let loc = self.modrm_location(i, next)?;
        let kind = i.reg & 7;
        let src = self.load_operand(loc, size, i)?;
        let bits = size as u32 * 8;
        let m = mask(size);
        let rax = self.gpr[GPR_RAX] & m;
        let rdx = self.gpr[GPR_RDX] & m;
        match kind {
            GROUP3_MUL | GROUP3_IMUL => {
                let (lo, hi, overflow) = if kind == GROUP3_MUL {
                    let p = rax as u128 * src as u128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    (lo, hi, hi != 0)
                } else {
                    let p = sign_extend(rax, bits) as i128 * sign_extend(src, bits) as i128;
                    let lo = p as u64 & m;
                    let hi = (p >> bits) as u64 & m;
                    // The high half is just the sign of the low half iff the product fits.
                    (lo, hi, p != sign_extend(lo, bits) as i128)
                };
                if size == 1 {
                    self.write_gpr(0, 2, true, lo | (hi << 8));
                } else {
                    self.write_gpr(0, size, true, lo);
                    self.write_gpr(2, size, true, hi);
                }
                let mut f = szp_flags(lo, size);
                if overflow {
                    f |= CF | OF;
                }
                self.set_eflags_arith(f);
            }
            _ => {
                if src == 0 {
                    return Err(Exception::divide_error());
                }
                let (quotient, remainder);
                if kind == GROUP3_DIV {
                    let dividend = if size == 1 {
                        self.gpr[GPR_RAX] as u128 & WORD_MASK as u128
                    } else {
                        (rdx as u128) << bits | rax as u128
                    };
                    let (q, r) = (dividend / src as u128, dividend % src as u128);
                    if q > m as u128 {
                        return Err(Exception::divide_error());
                    }
                    quotient = q as u64;
                    remainder = r as u64;
                } else {
                    let dividend: i128 = if size == 1 {
                        (self.gpr[GPR_RAX] as u16 as i16) as i128
                    } else {
                        // Sign-extend the 2 * bits wide rDX:rAX value to 128 bits.
                        (((rdx as u128) << bits | rax as u128) << (128 - 2 * bits)) as i128
                            >> (128 - 2 * bits)
                    };
                    let divisor = sign_extend(src, bits) as i128;
                    // i128::MIN / -1 overflows the host division itself (a panic, not a wrap). It
                    // is the one quotient that cannot fit any operand size, so it is #DE too.
                    let (Some(q), Some(r)) =
                        (dividend.checked_div(divisor), dividend.checked_rem(divisor))
                    else {
                        return Err(Exception::divide_error());
                    };
                    let limit = 1i128 << (bits - 1);
                    if q >= limit || q < -limit {
                        return Err(Exception::divide_error());
                    }
                    quotient = q as u64 & m;
                    remainder = r as u64 & m;
                }
                if size == 1 {
                    self.write_gpr(0, 2, true, quotient | (remainder << 8));
                } else {
                    self.write_gpr(0, size, true, quotient);
                    self.write_gpr(2, size, true, remainder);
                }
            }
        }
        Ok(())
    }

    /// Signed multiply truncated to `size` (imul r, r/m and the three-operand forms). CF and OF
    /// are set when the full product does not fit.
    fn imul_truncated(&mut self, a: u64, b: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let p = sign_extend(a, bits) as i128 * sign_extend(b, bits) as i128;
        let r = p as u64 & mask(size);
        let mut f = szp_flags(r, size);
        if p != sign_extend(r, bits) as i128 {
            f |= CF | OF;
        }
        self.set_eflags_arith(f);
        r
    }

    /// bt/bts/btr/btc. With a register bit offset and a memory operand the offset is signed and
    /// may reach outside the addressed operand: the address moves by whole operands and the bit
    /// index is the remainder. An immediate offset is taken modulo the operand width instead.
    fn bit_op(&mut self, i: &Insn, next: u64, operation: u8, imm_offset: bool) -> ExecResult<()> {
        let size = i.opsize;
        let bits = size as i64 * 8;
        let offset = if imm_offset {
            i.imm as u64
        } else {
            self.read_gpr(i.reg, size, true)
        };
        let loc = if i.md == 3 {
            Loc::Reg(i.rm)
        } else {
            let mut address = self.effective_address(i, next);
            if !imm_offset {
                let signed_offset = sign_extend(offset, bits as u32);
                address =
                    address.wrapping_add((signed_offset.div_euclid(bits) * size as i64) as u64);
            }
            Loc::Mem(self.linear_address(i, address)?)
        };
        let bit = if imm_offset || i.md == 3 {
            (offset as i64).rem_euclid(bits) as u32
        } else {
            sign_extend(offset, bits as u32).rem_euclid(bits) as u32
        };
        let v = self.load_operand(loc, size, i)?;
        let old = (v >> bit) & 1;
        let new_value = match operation {
            BIT_SET => v | (1 << bit),
            BIT_RESET => v & !(1 << bit),
            BIT_COMPLEMENT => v ^ (1 << bit),
            _ => v,
        };
        if operation != BIT_TEST {
            self.store_operand(loc, size, i, new_value)?;
        }
        let f = self.eflags_full();
        self.set_eflags_arith((f & !CF) | old);
        Ok(())
    }

    /// Bulk path of a forward `rep movs`/`rep stos` with 64-bit addressing: moves as many whole
    /// elements as fit in the current source and destination pages (and `budget`) with one page
    /// translation and one host copy or fill, then returns the count. Returns 0, having touched
    /// nothing, when the plain per-element loop must do it: an element straddling a page, a
    /// destination or source outside RAM (MMIO), a destination page holding translated code
    /// (`phys_write` invalidates it), or a forward overlap where the copy has to re-read what it
    /// just wrote (`rep movsb` as a memset idiom). A fault is raised exactly as the first
    /// element would raise it: the source is translated before the destination, and a fault on a
    /// later page comes from the next call with the registers advanced by what was done.
    fn rep_bulk(&mut self, i: &Insn, op: u8, size: u8, budget: u64) -> ExecResult<u64> {
        let element = size as u64;
        let copies = matches!(op, 0xa4 | 0xa5);
        let dst_linear = self.gpr[GPR_RDI];
        let src_linear = self.linear_address(i, self.gpr[GPR_RSI])?;
        let room = |linear: u64| (PAGE_SIZE - (linear & PAGE_OFFSET_MASK)) / element;
        let mut elements = self.gpr[GPR_RCX].min(budget).min(room(dst_linear));
        if copies {
            elements = elements.min(room(src_linear));
        }
        if elements == 0 {
            return Ok(0);
        }
        let bytes = elements * element;
        let src_pa = if copies {
            Some(self.translate(src_linear, AccessKind::Read, false)?)
        } else {
            None
        };
        let dst_pa = self.translate(dst_linear, AccessKind::Write, false)?;
        if !self.ram.contains(dst_pa, bytes)
            || self.page_has_code[(dst_pa >> PAGE_SHIFT) as usize] != 0
        {
            return Ok(0);
        }
        let ram = self.ram.base();
        if let Some(src_pa) = src_pa {
            if !self.ram.contains(src_pa, bytes) || (dst_pa > src_pa && dst_pa < src_pa + bytes) {
                return Ok(0);
            }
            // SAFETY: both ranges lie inside `ram` (checked above); a copy that moves memory
            // down (dst below src) may overlap, which `copy` allows.
            unsafe {
                core::ptr::copy(
                    ram.add(src_pa as usize),
                    ram.add(dst_pa as usize),
                    bytes as usize,
                );
            }
            self.gpr[GPR_RSI] = self.gpr[GPR_RSI].wrapping_add(bytes);
        } else {
            let pattern = self.read_gpr(0, size, true).to_le_bytes();
            let element = &pattern[..size as usize];
            let bytes = bytes as usize;
            // SAFETY: the destination range lies inside `ram`. Each doubling copies a prefix that
            // already holds whole elements onto the bytes right after it, so the two never overlap.
            unsafe {
                let dst = ram.add(dst_pa as usize);
                if element.iter().all(|&byte| byte == element[0]) {
                    // musl's and the kernel's memset widen the fill byte, so this is the usual case.
                    core::ptr::write_bytes(dst, element[0], bytes);
                } else {
                    core::ptr::copy_nonoverlapping(element.as_ptr(), dst, element.len());
                    let mut filled = element.len();
                    while filled < bytes {
                        let chunk = filled.min(bytes - filled);
                        core::ptr::copy_nonoverlapping(dst, dst.add(filled), chunk);
                        filled += chunk;
                    }
                }
            }
        }
        self.gpr[GPR_RDI] = self.gpr[GPR_RDI].wrapping_add(bytes);
        self.gpr[GPR_RCX] -= elements;
        Ok(elements)
    }

    /// movs/stos/lods/cmps/scas with optional rep. State is committed per iteration, so a fault
    /// mid-rep leaves GPR_RSI/GPR_RDI/GPR_RCX consistent with the iterations already done.
    fn string_op(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op as u8;
        let size = if op & 1 == 0 { 1 } else { i.opsize };
        let amask = address_mask(i);
        let step: u64 = if self.eflags & DF != 0 {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        // Only the index registers an instruction uses are written back; with the 67 prefix that
        // is a 32-bit write, which zeroes the upper half. An unused one keeps its upper half.
        let advance = |index: u64| {
            let moved = index.wrapping_add(step);
            if i.addr32 { moved & DWORD_MASK } else { moved }
        };
        let rep = i.rep != 0;
        let compares = matches!(op, 0xa6 | 0xa7 | 0xae | 0xaf);
        let slice = REP_BYTES_PER_SLICE / size as u64;
        let mut iterations = 0;
        loop {
            if rep && self.gpr[GPR_RCX] & amask == 0 {
                break;
            }
            if rep && matches!(op, 0xa4 | 0xa5 | 0xaa | 0xab) && !i.addr32 && self.eflags & DF == 0
            {
                let done = self.rep_bulk(i, op, size, slice - iterations)?;
                if done > 0 {
                    iterations += done;
                    if self.gpr[GPR_RCX] == 0 {
                        break;
                    }
                    if iterations >= slice {
                        self.icount += iterations;
                        return Ok(());
                    }
                    continue;
                }
            }
            // Only the source honours a segment override; the destination is always ES, which has
            // no base in 64-bit mode.
            let src = self.linear_address(i, self.gpr[GPR_RSI] & amask)?;
            let dst = self.gpr[GPR_RDI] & amask;
            match op {
                // movs
                0xa4 | 0xa5 => {
                    let v = self.read(src, size)?;
                    self.write(dst, size, v)?;
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // stos
                0xaa | 0xab => {
                    let v = self.read_gpr(0, size, true);
                    self.write(dst, size, v)?;
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // lods
                0xac | 0xad => {
                    let v = self.read(src, size)?;
                    self.write_gpr(0, size, true, v);
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                }
                // cmps: flags of [rsi] - [rdi]
                0xa6 | 0xa7 => {
                    let a = self.read(src, size)?;
                    let b = self.read(dst, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.gpr[GPR_RSI] = advance(self.gpr[GPR_RSI]);
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
                // scas: flags of rAX - [rdi]
                _ => {
                    let a = self.read_gpr(0, size, true);
                    let b = self.read(dst, size)?;
                    self.set_lazy_flags(
                        LazyFlagsOp::Sub,
                        size,
                        a.wrapping_sub(b) & mask(size),
                        b,
                        0,
                    );
                    self.gpr[GPR_RDI] = advance(self.gpr[GPR_RDI]);
                }
            }
            if !rep {
                break;
            }
            let count = self.gpr[GPR_RCX].wrapping_sub(1);
            self.gpr[GPR_RCX] = if i.addr32 { count & DWORD_MASK } else { count };
            iterations += 1;
            if compares {
                let zf = self.eflags_full() & ZF != 0;
                if (i.rep == REP_REPE && !zf) || (i.rep == REP_REPNE && zf) {
                    break;
                }
            }
            if iterations >= slice && self.gpr[GPR_RCX] & amask != 0 {
                // RIP stays on the instruction; the run loop re-executes it after servicing events.
                self.icount += iterations;
                return Ok(());
            }
        }
        self.icount += iterations;
        self.rip = next;
        Ok(())
    }

    /// Loads DS/ES/FS/GS/SS (`idx` is a `SEG_*` index) from a selector in 64-bit mode, with the
    /// privilege checks of `mov sreg` made against the current CPL. Nothing is loaded on a fault.
    pub fn load_data_seg(&mut self, idx: usize, sel: u16) -> ExecResult<()> {
        // A null SS is only usable when it names the current privilege level.
        if idx == SEG_SS && selector_is_null(sel) && (sel & SELECTOR_RPL_MASK) as u8 != self.cpl {
            return Err(Exception::general_protection(0));
        }
        let seg = self.checked_data_seg(idx, sel, self.cpl)?;
        self.seg[idx] = seg;
        Ok(())
    }

    /// Validates `sel` for DS/ES/FS/GS/SS against privilege level `cpl` and returns the segment
    /// to load, leaving the CPU untouched (iretq and far return validate before they commit).
    /// `cpl` is the current CPL for `mov sreg` and the one being returned to for iretq/retf.
    ///
    /// A null selector is allowed everywhere except SS at CPL 3. Otherwise (SDM 3A 5.4):
    /// system descriptors and execute-only code are #GP for a data register; data and
    /// non-conforming code need DPL >= max(CPL, RPL), conforming code is exempt; SS must be
    /// writable data with RPL = DPL = CPL; a missing segment is #SS for SS and #NP otherwise.
    pub(super) fn checked_data_seg(&mut self, idx: usize, sel: u16, cpl: u8) -> ExecResult<Seg> {
        if selector_is_null(sel) {
            if idx == SEG_SS && cpl == USER_CPL {
                return Err(Exception::general_protection(0));
            }
            return Ok(Seg {
                sel,
                ..Default::default()
            });
        }
        let s = self.load_desc_seg(sel, false)?;
        let rpl = (sel & SELECTOR_RPL_MASK) as u8;
        let privilege_violation = Exception::general_protection(selector_error_code(sel));
        if s.flags & SEG_NON_SYSTEM == 0 {
            return Err(privilege_violation);
        }
        if idx == SEG_SS {
            if s.is_code() || s.flags & SEG_TYPE_READ_WRITE == 0 || rpl != cpl || s.dpl() != cpl {
                return Err(privilege_violation);
            }
        } else {
            if s.is_code() && s.flags & SEG_TYPE_READ_WRITE == 0 {
                return Err(privilege_violation);
            }
            if !is_conforming_code(&s) && (s.dpl() < cpl || s.dpl() < rpl) {
                return Err(privilege_violation);
            }
        }
        if !s.present() {
            return Err(Exception::fault(
                if idx == SEG_SS {
                    VECTOR_STACK_FAULT
                } else {
                    VECTOR_SEGMENT_NOT_PRESENT
                },
                Some(selector_error_code(sel)),
            ));
        }
        Ok(s)
    }

    /// Validates the code segment selector popped by iretq or a far return (SDM 3A IRET/RET):
    /// present 64-bit code, RPL not below the current CPL, and DPL equal to RPL for
    /// non-conforming code or at most RPL for conforming code. Nothing is loaded.
    fn checked_return_code_seg(&mut self, cs: u16) -> ExecResult<Seg> {
        if selector_is_null(cs) {
            return Err(Exception::general_protection(0));
        }
        let s = self.load_desc_seg(cs, false)?;
        let rpl = (cs & SELECTOR_RPL_MASK) as u8;
        let privilege_violation = Exception::general_protection(selector_error_code(cs));
        let dpl_matches = if is_conforming_code(&s) {
            s.dpl() <= rpl
        } else {
            s.dpl() == rpl
        };
        if !s.is_code() || rpl < self.cpl || !dpl_matches {
            return Err(privilege_violation);
        }
        if !s.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error_code(cs)),
            ));
        }
        // CS.L clear returns to compatibility mode (SDM 6.14.3: iretq and far return may target
        // it; GRUB's multiboot2 handoff to Xen does). CS.L with CS.D is reserved.
        if s.long() && s.flags & SEG_DEFAULT_BIG != 0 {
            return Err(privilege_violation);
        }
        Ok(s)
    }

    /// A return into compatibility mode is limit-checked like any 32-bit target.
    fn check_compatibility_target(code: &Seg, rip: u64) -> ExecResult<()> {
        if !code.long() && rip > code.limit as u64 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// After a return to an outer ring, DS/ES/FS/GS that hold a data or non-conforming code
    /// segment with DPL below the new CPL are replaced by null selectors, so the less privileged
    /// code cannot keep using them. A register that is already null is left as it is, which keeps
    /// the FS/GS bases.
    fn null_inaccessible_data_segments(&mut self, new_cpl: u8) {
        for register in [SEG_ES, SEG_DS, SEG_FS, SEG_GS] {
            let s = self.seg[register];
            if !selector_is_null(s.sel) && s.dpl() < new_cpl && !is_conforming_code(&s) {
                self.seg[register] = Seg::default();
            }
        }
    }

    /// 64-bit iretq: pops RIP, CS, RFLAGS, RSP, SS (always all five in 64-bit mode). Only the
    /// 8-byte operand size is implemented. Every check is made before any state changes, so a
    /// fault leaves CS, SS, CPL, RSP, RIP and the flags as they were. The SS checks are made
    /// against the CPL being returned to.
    fn iretq(&mut self, i: &Insn) -> ExecResult<()> {
        const FRAME_CS: u64 = 8;
        const FRAME_RFLAGS: u64 = 16;
        const FRAME_RSP: u64 = 24;
        const FRAME_SS: u64 = 32;
        if i.opsize != 8 {
            return Err(Exception::invalid_opcode());
        }
        let sp = self.gpr[GPR_RSP];
        let rip = self.read_stack(sp, 8)?;
        let cs = self.read_stack(sp.wrapping_add(FRAME_CS), 8)? as u16;
        let new_rflags = self.read_stack(sp.wrapping_add(FRAME_RFLAGS), 8)?;
        let new_cs = self.checked_return_code_seg(cs)?;
        Self::check_compatibility_target(&new_cs, rip)?;
        let new_rsp = self.read_stack(sp.wrapping_add(FRAME_RSP), 8)?;
        let ss = self.read_stack(sp.wrapping_add(FRAME_SS), 8)? as u16;
        let new_cpl = (cs & SELECTOR_RPL_MASK) as u8;
        let new_ss = self.checked_data_seg(SEG_SS, ss, new_cpl)?;
        let old_cpl = self.cpl;
        self.seg[SEG_CS] = new_cs;
        self.seg[SEG_SS] = new_ss;
        // set_cpl also switches to the user or supervisor half of the soft TLB.
        self.set_cpl(new_cpl);
        // IF and IOPL are gated by the CPL iretq started at, not the one it returns to.
        let loadable = self.loadable_eflags(old_cpl);
        let current = self.eflags_full();
        self.eflags = ((current & !loadable) | (new_rflags & loadable)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.gpr[GPR_RSP] = new_rsp;
        self.rip = rip;
        if new_cpl > old_cpl {
            self.null_inaccessible_data_segments(new_cpl);
        }
        Ok(())
    }

    /// Far return (0xca/0xcb) with `imm` extra bytes released after CS. Only the 8-byte operand
    /// size (REX.W) is implemented. For a return to an outer ring RSP:SS sit above the released
    /// bytes, and the bytes are released from the new stack as well. Validation comes before any
    /// state change, as for `iretq`.
    fn far_return(&mut self, i: &Insn, imm: u64) -> ExecResult<()> {
        // Frame: rip, cs, `imm` released bytes, then (outer-ring returns only) rsp and ss.
        // FRAME_RSP and FRAME_SS are offsets from RSP before `imm` is added.
        const FRAME_CS: u64 = 8;
        const FRAME_RSP: u64 = 16;
        const FRAME_SS: u64 = 24;
        if i.opsize != 8 {
            return Err(Exception::invalid_opcode());
        }
        let sp = self.gpr[GPR_RSP];
        let rip = self.read_stack(sp, 8)?;
        let cs = self.read_stack(sp.wrapping_add(FRAME_CS), 8)? as u16;
        let new_cs = self.checked_return_code_seg(cs)?;
        Self::check_compatibility_target(&new_cs, rip)?;
        let new_cpl = (cs & SELECTOR_RPL_MASK) as u8;
        if new_cpl == self.cpl {
            self.seg[SEG_CS] = new_cs;
            self.gpr[GPR_RSP] = sp.wrapping_add(FRAME_RSP).wrapping_add(imm);
        } else {
            let new_rsp = self.read_stack(sp.wrapping_add(FRAME_RSP).wrapping_add(imm), 8)?;
            let ss = self.read_stack(sp.wrapping_add(FRAME_SS).wrapping_add(imm), 8)? as u16;
            let new_ss = self.checked_data_seg(SEG_SS, ss, new_cpl)?;
            self.seg[SEG_CS] = new_cs;
            self.seg[SEG_SS] = new_ss;
            self.set_cpl(new_cpl);
            self.gpr[GPR_RSP] = new_rsp.wrapping_add(imm);
            self.null_inaccessible_data_segments(new_cpl);
        }
        self.rip = rip;
        Ok(())
    }

    /// Executes one decoded instruction; also the slow path the JIT calls for instructions it does
    /// not compile (`jit/helpers.rs`), which makes it the oracle for `--verify`.
    ///
    /// On `Ok` RIP is already advanced (next instruction or branch target). On `Err` RIP is still
    /// the faulting instruction, except int/int3 which set it to the next instruction first so the
    /// pushed return address is right. Opcodes are `Insn::op`: one-byte opcodes as is, `0f xx` as
    /// `0x1xx`.
    pub fn exec(&mut self, i: &Insn) -> ExecResult<()> {
        if self.ordered_memory && self.rmw == Rmw::Off && Self::is_locked(i) {
            return self.exec_locked(i);
        }
        let next = self.rip.wrapping_add(i.len as u64);
        let osz = i.opsize;
        let op = i.op;
        let rex = i.rex_present;
        if self.svm.guest.is_some() {
            self.svm_intercept(i, next)?;
        }
        if i.mode != CodeMode::Long64 {
            self.pending_operand.set(None);
            if self.exec_legacy(i, next)? {
                return Ok(());
            }
        }
        if i.vex {
            return self.exec_vex(i, next);
        }
        if i.evex {
            return self.exec_evex(i, next);
        }
        match op {
            // The eight ALU ops (add/or/adc/sbb/and/sub/xor/cmp), forms 0..=5: r/m,r (byte, full),
            // r,r/m (byte, full), al,imm and eax,imm.
            0x00..=0x3f if op & 7 < 6 => {
                let operation = ((op >> 3) & 7) as u8;
                let form = op & 7;
                let size = if form & 1 == 0 { 1 } else { osz };
                match form {
                    0 | 1 => {
                        let loc = self.modrm_location(i, next)?;
                        let a = self.load_operand(loc, size, i)?;
                        let b = self.read_gpr(i.reg, size, rex);
                        let saved = self.save_flags();
                        let r = self.alu(operation, a, b, size);
                        if operation != ALU_CMP {
                            self.store_result(loc, size, i, r, saved)?;
                        }
                    }
                    2 | 3 => {
                        let loc = self.modrm_location(i, next)?;
                        let b = self.load_operand(loc, size, i)?;
                        let a = self.read_gpr(i.reg, size, rex);
                        let r = self.alu(operation, a, b, size);
                        if operation != ALU_CMP {
                            self.write_gpr(i.reg, size, rex, r);
                        }
                    }
                    _ => {
                        let a = self.read_gpr(0, size, true);
                        let r = self.alu(operation, a, i.imm as u64, size);
                        if operation != ALU_CMP {
                            self.write_gpr(0, size, true, r);
                        }
                    }
                }
            }
            // push r64
            0x50..=0x57 => {
                let r = opcode_register(i);
                let sz = stack_operand_size(i);
                let v = self.gpr[r as usize];
                self.push(sz, v)?;
            }
            // pop r64
            0x58..=0x5f => {
                let r = opcode_register(i);
                let sz = stack_operand_size(i);
                let v = self.pop(sz)?;
                self.write_gpr(r, sz, true, v);
            }
            // movsxd
            0x63 => {
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, 4, i)?;
                if osz == 8 {
                    self.gpr[i.reg as usize] = v as u32 as i32 as i64 as u64;
                } else {
                    self.write_gpr(i.reg, osz, true, v);
                }
            }
            // push imm32 / imm8
            0x68 | 0x6a => {
                let sz = stack_operand_size(i);
                self.push(sz, i.imm as u64)?;
            }
            // imul r, r/m, imm32 / imm8
            0x69 | 0x6b => {
                let loc = self.modrm_location(i, next)?;
                let a = self.load_operand(loc, osz, i)?;
                let r = self.imul_truncated(a, i.imm as u64, osz);
                self.write_gpr(i.reg, osz, true, r);
            }
            // jcc rel8
            0x70..=0x7f => {
                if self.condition_holds((op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // group 1: ALU r/m, imm
            0x80 | 0x81 | 0x83 => {
                let size = if op == 0x80 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let a = self.load_operand(loc, size, i)?;
                let operation = i.reg & 7;
                let saved = self.save_flags();
                let r = self.alu(operation, a, i.imm as u64, size);
                if operation != ALU_CMP {
                    self.store_result(loc, size, i, r, saved)?;
                }
            }
            // test r/m, r
            0x84 | 0x85 => {
                let size = if op == 0x84 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let a = self.load_operand(loc, size, i)?;
                let b = self.read_gpr(i.reg, size, rex);
                self.set_lazy_flags(LazyFlagsOp::Logic, size, a & b, 0, 0);
            }
            // xchg r/m, r
            0x86 | 0x87 => {
                let size = if op == 0x86 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let a = self.load_operand(loc, size, i)?;
                let b = self.read_gpr(i.reg, size, rex);
                self.store_operand(loc, size, i, b)?;
                self.write_gpr(i.reg, size, rex, a);
            }
            // mov r/m, r
            0x88 | 0x89 => {
                let size = if op == 0x88 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let v = self.read_gpr(i.reg, size, rex);
                self.store_operand(loc, size, i, v)?;
            }
            // mov r, r/m
            0x8a | 0x8b => {
                let size = if op == 0x8a { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, size, i)?;
                self.write_gpr(i.reg, size, rex, v);
            }
            // mov r/m, sreg: a memory destination is always 16 bits, a register one takes the
            // operand size (upper bits zeroed).
            0x8c => {
                // REX.R extends ModRM.reg, but the segment register field is only 3 bits.
                let idx = (i.reg & 7) as usize;
                if idx > SEG_GS {
                    return Err(Exception::invalid_opcode());
                }
                let v = self.seg[idx].sel as u64;
                let loc = self.modrm_location(i, next)?;
                let size = if i.md == 3 { osz } else { 2 };
                self.store_operand(loc, size, i, v)?;
            }
            // lea
            0x8d => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let a = self.effective_address(i, next);
                self.write_gpr(i.reg, osz, true, a);
            }
            // mov sreg, r/m. CS cannot be loaded this way.
            0x8e => {
                let idx = (i.reg & 7) as usize;
                if idx == SEG_CS || idx > SEG_GS {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let sel = self.load_operand(loc, 2, i)? as u16;
                self.load_data_seg(idx, sel)?;
                // Interrupts are held off until after the instruction following mov ss, so that
                // a mov ss / mov rsp pair is atomic.
                if idx == SEG_SS {
                    self.interrupt_shadow = true;
                }
            }
            // pop r/m (only /0; the other encodings are XOP)
            0x8f => {
                if i.reg & 7 != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let sz = stack_operand_size(i);
                let loc = self.modrm_location(i, next)?;
                let sp = self.gpr[GPR_RSP];
                let v = self.read_stack(sp, sz)?;
                self.gpr[GPR_RSP] = sp.wrapping_add(sz as u64);
                // A memory destination addressed through GPR_RSP uses the value after the pop.
                let loc = match loc {
                    Loc::Mem(a) if i.base == GPR_RSP as u8 => Loc::Mem(a.wrapping_add(sz as u64)),
                    l => l,
                };
                if let Err(e) = self.store_operand(loc, sz, i, v) {
                    self.gpr[GPR_RSP] = sp;
                    return Err(e);
                }
            }
            // nop, or xchg r8, rax with REX.B. pause (f3 90) is a nop to this vCPU, but a spin
            // loop is waiting for another one, so it gives up the rest of its quantum.
            0x90 => {
                if i.rex & REX_B != 0 {
                    let r = 8;
                    let a = self.read_gpr(0, osz, true);
                    let b = self.read_gpr(r, osz, true);
                    self.write_gpr(0, osz, true, b);
                    self.write_gpr(r, osz, true, a);
                } else if i.rep != 0 && self.threads.is_some() {
                    self.pause_threaded();
                } else if i.rep != 0 && self.multi {
                    self.request_yield();
                }
            }
            // xchg r, rax
            0x91..=0x97 => {
                let r = opcode_register(i);
                let a = self.read_gpr(0, osz, true);
                let b = self.read_gpr(r, osz, true);
                self.write_gpr(0, osz, true, b);
                self.write_gpr(r, osz, true, a);
            }
            // cdqe / cwde / cbw
            0x98 => match osz {
                8 => self.gpr[GPR_RAX] = self.gpr[GPR_RAX] as u32 as i32 as i64 as u64,
                4 => self.gpr[GPR_RAX] = self.gpr[GPR_RAX] as u16 as i16 as i32 as u32 as u64,
                _ => {
                    let v = self.gpr[GPR_RAX] as u8 as i8 as i16 as u16 as u64;
                    self.write_gpr(0, 2, true, v)
                }
            },
            // cqo / cdq / cwd: fill rDX with the sign of rAX
            0x99 => match osz {
                8 => self.gpr[GPR_RDX] = ((self.gpr[GPR_RAX] as i64) >> 63) as u64,
                4 => self.gpr[GPR_RDX] = (((self.gpr[GPR_RAX] as u32 as i32) >> 31) as u32) as u64,
                _ => {
                    let v = (((self.gpr[GPR_RAX] as u16 as i16) >> 15) as u16) as u64;
                    self.write_gpr(2, 2, true, v)
                }
            },
            // fwait: #NM when both CR0.MP and CR0.TS are set
            0x9b => {
                if self.cr0 & (CR0_MP | CR0_TS) == (CR0_MP | CR0_TS) {
                    return Err(Exception::device_not_available());
                }
            }
            // pushf: the pushed image has RF and VM clear
            0x9c => {
                let sz = stack_operand_size(i);
                let f = self.eflags_full() & !(RF | VM);
                self.push(sz, f)?;
            }
            // popf
            0x9d => {
                let sz = stack_operand_size(i);
                let v = self.pop(sz)?;
                let mut loadable = self.loadable_eflags(self.cpl);
                if sz == 2 {
                    loadable &= WORD_MASK;
                }
                let current = self.eflags_full();
                self.eflags = ((current & !loadable) | (v & loadable)) | RESERVED_ONE;
                self.lazy_op = LazyFlagsOp::Eflags;
            }
            // sahf
            0x9e => {
                let ah = (self.gpr[GPR_RAX] >> 8) & LAHF_SAHF_FLAGS;
                let f = self.eflags_full();
                self.set_eflags_arith((f & !LAHF_SAHF_FLAGS) | ah);
            }
            // lahf: bit 1 of AH always reads 1
            0x9f => {
                let f = (self.eflags_full() & LAHF_SAHF_FLAGS) | RESERVED_ONE;
                self.write_gpr(BYTE_REG_AH, 1, false, f);
            }
            // mov al/eax, moffs and mov moffs, al/eax
            0xa0..=0xa3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let a = self.linear_address(i, i.imm as u64 & address_mask(i))?;
                self.check_operand_extent(size, op >= 0xa2)?;
                if op < 0xa2 {
                    let v = self.read(a, size)?;
                    self.write_gpr(0, size, true, v);
                } else {
                    let v = self.read_gpr(0, size, true);
                    self.write(a, size, v)?;
                }
            }
            // movs, cmps, stos, lods, scas
            0xa4..=0xa7 | 0xaa..=0xaf => return self.string_op(i, next),
            // test al/eax, imm
            0xa8 | 0xa9 => {
                let size = if op == 0xa8 { 1 } else { osz };
                let a = self.read_gpr(0, size, true);
                self.set_lazy_flags(
                    LazyFlagsOp::Logic,
                    size,
                    a & i.imm as u64 & mask(size),
                    0,
                    0,
                );
            }
            // mov r8, imm8
            0xb0..=0xb7 => {
                let r = opcode_register(i);
                self.write_gpr(r, 1, rex, i.imm as u64);
            }
            // mov r, imm (imm64 with REX.W)
            0xb8..=0xbf => {
                let r = opcode_register(i);
                self.write_gpr(r, osz, true, i.imm as u64);
            }
            // shift/rotate group: imm8, 1 or CL count
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let size = if op & 1 == 0 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let count = match op {
                    0xc0 | 0xc1 => i.imm as u64,
                    0xd0 | 0xd1 => 1,
                    _ => self.gpr[GPR_RCX] & BYTE_MASK,
                };
                let a = self.load_operand(loc, size, i)?;
                let saved = self.save_flags();
                let r = self.shift_rotate(i.reg & 7, a, count, size);
                self.store_result(loc, size, i, r, saved)?;
            }
            // ret / ret imm16
            0xc2 | 0xc3 => {
                let target = self.read_stack(self.gpr[GPR_RSP], 8)?;
                let extra = if op == 0xc2 { i.imm as u64 } else { 0 };
                self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_add(8 + extra);
                self.rip = target;
                return Ok(());
            }
            // mov r/m, imm. Only /0 is defined (c6 /7 and c7 /7 are xabort/xbegin, which need RTM).
            0xc6 | 0xc7 => {
                if i.reg & 7 != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let size = if op == 0xc6 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                self.store_operand(loc, size, i, i.imm as u64)?;
            }
            // enter: the nesting level is taken modulo 32. Registers are committed only after the
            // last push, so a fault leaves RSP and RBP as they were (the pushes already made
            // stay in memory, as on hardware).
            0xc8 => {
                let size = stack_operand_size(i);
                let width = size as u64;
                let level = i.imm2 & ENTER_LEVEL_MASK;
                let mut sp = self.gpr[GPR_RSP].wrapping_sub(width);
                self.write_stack(sp, size, self.gpr[GPR_RBP])?;
                let frame_pointer = sp;
                if level > 0 {
                    // Copy the enclosing frames' pointers down the chain, then push our own.
                    let mut outer = self.gpr[GPR_RBP];
                    for _ in 1..level {
                        outer = outer.wrapping_sub(width);
                        let enclosing = self.read_stack(outer, size)?;
                        sp = sp.wrapping_sub(width);
                        self.write_stack(sp, size, enclosing)?;
                    }
                    sp = sp.wrapping_sub(width);
                    self.write_stack(sp, size, frame_pointer)?;
                }
                self.gpr[GPR_RBP] = frame_pointer;
                self.gpr[GPR_RSP] = sp.wrapping_sub(i.imm as u64);
            }
            // leave: GPR_RSP is only committed after the load of GPR_RBP succeeded. With a 66
            // prefix it pops a word into BP and leaves the rest of RBP alone.
            0xc9 => {
                let size = stack_operand_size(i);
                let bp = self.gpr[GPR_RBP];
                let v = self.read_stack(bp, size)?;
                self.gpr[GPR_RSP] = bp.wrapping_add(size as u64);
                self.write_gpr(GPR_RBP as u8, size, true, v);
            }
            // far ret imm16 / far ret
            0xca | 0xcb => {
                let imm = if op == 0xca { i.imm as u64 } else { 0 };
                return self.far_return(i, imm);
            }
            // int3
            0xcc => {
                self.rip = next;
                return Err(Exception::Fault {
                    vector: VECTOR_BREAKPOINT,
                    error_code: None,
                    software: true,
                });
            }
            // int imm8
            0xcd => {
                self.rip = next;
                return Err(Exception::Fault {
                    vector: i.imm as u8,
                    error_code: None,
                    software: true,
                });
            }
            0xcf => {
                let result = self.iretq(i);
                // Only a completed iret re-enables NMIs; one that faulted left them blocked.
                if result.is_ok() {
                    self.nmi_blocked = false;
                }
                return result;
            }
            // xlat: al = [rbx + al]; with the 67 prefix the sum is formed and wrapped at 32 bits
            0xd7 => {
                let a =
                    self.gpr[GPR_RBX].wrapping_add(self.gpr[GPR_RAX] & BYTE_MASK) & address_mask(i);
                let v = self.read(self.linear_address(i, a)?, 1)?;
                self.write_gpr(0, 1, true, v);
            }
            0xd8..=0xdf => return self.x87(i, next),
            // loopne, loope, loop, jrcxz (jecxz with the 67 prefix)
            0xe0..=0xe3 => {
                let amask = address_mask(i);
                let take = if op == 0xe3 {
                    self.gpr[GPR_RCX] & amask == 0
                } else {
                    let count = self.gpr[GPR_RCX].wrapping_sub(1) & amask;
                    self.gpr[GPR_RCX] = count;
                    let zf = self.eflags_full() & ZF != 0;
                    count != 0
                        && match op {
                            0xe0 => !zf,
                            0xe1 => zf,
                            _ => true,
                        }
                };
                if take {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // in/out with imm8 port (e4..e7) or DX (ec..ef)
            0xe4..=0xe7 | 0xec..=0xef => return self.port_io(i, next),
            // insb/insw/insd, outsb/outsw/outsd
            0x6c..=0x6f => return self.string_io(i, next),
            // call rel32
            0xe8 => {
                self.push(8, next)?;
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            // jmp rel32 / rel8
            0xe9 | 0xeb => {
                self.rip = next.wrapping_add(i.imm as u64);
                return Ok(());
            }
            // hlt
            0xf4 => {
                if self.cpl != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.halted = true;
                self.mwait_watch = None;
            }
            // cmc
            0xf5 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f ^ CF);
            }
            // group 3: test/not/neg/mul/imul/div/idiv
            0xf6 | 0xf7 => {
                let size = if op == 0xf6 { 1 } else { osz };
                match i.reg & 7 {
                    GROUP3_TEST | GROUP3_TEST_ALIAS => {
                        let loc = self.modrm_location(i, next)?;
                        let a = self.load_operand(loc, size, i)?;
                        self.set_lazy_flags(
                            LazyFlagsOp::Logic,
                            size,
                            a & i.imm as u64 & mask(size),
                            0,
                            0,
                        );
                    }
                    GROUP3_NOT => {
                        let loc = self.modrm_location(i, next)?;
                        let a = self.load_operand(loc, size, i)?;
                        self.store_operand(loc, size, i, !a)?;
                    }
                    GROUP3_NEG => {
                        let loc = self.modrm_location(i, next)?;
                        let a = self.load_operand(loc, size, i)?;
                        let r = 0u64.wrapping_sub(a) & mask(size);
                        let saved = self.save_flags();
                        self.set_lazy_flags(LazyFlagsOp::Sub, size, r, a & mask(size), 0);
                        self.store_result(loc, size, i, r, saved)?;
                    }
                    _ => self.mul_div(i, next, size)?,
                }
            }
            // clc
            0xf8 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f & !CF);
            }
            // stc
            0xf9 => {
                let f = self.eflags_full();
                self.set_eflags_arith(f | CF);
            }
            // cli / sti: need CPL <= IOPL (VME/PVI are not emulated)
            0xfa | 0xfb => {
                if self.cpl > self.iopl() {
                    return Err(Exception::general_protection(0));
                }
                if op == 0xfa {
                    self.eflags &= !IF;
                } else {
                    // Only the transition from IF=0 opens the one-instruction interrupt shadow.
                    if self.eflags & IF == 0 {
                        self.interrupt_shadow = true;
                    }
                    self.eflags |= IF;
                }
            }
            // cld / std
            0xfc => self.eflags &= !DF,
            0xfd => self.eflags |= DF,
            // group 4/5: inc, dec, call, jmp, push (far forms are not implemented)
            0xfe | 0xff => {
                let size = if op == 0xfe { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                match i.reg & 7 {
                    GROUP5_INC | GROUP5_DEC => {
                        let a = self.load_operand(loc, size, i)?;
                        // inc/dec leave CF alone: the lazy op takes the current CF as its source.
                        let cf = self.cf() as u64;
                        let (r, lazy_op) = if i.reg & 7 == GROUP5_INC {
                            (a.wrapping_add(1) & mask(size), LazyFlagsOp::Inc)
                        } else {
                            (a.wrapping_sub(1) & mask(size), LazyFlagsOp::Dec)
                        };
                        let saved = self.save_flags();
                        self.set_lazy_flags(lazy_op, size, r, cf, 0);
                        self.store_result(loc, size, i, r, saved)?;
                    }
                    GROUP5_CALL_NEAR if op == 0xff => {
                        // The target is read before the push so `call [rsp+x]` sees the old GPR_RSP.
                        let target = self.load_operand(loc, 8, i)?;
                        self.push(8, next)?;
                        self.rip = target;
                        return Ok(());
                    }
                    GROUP5_JMP_NEAR if op == 0xff => {
                        self.rip = self.load_operand(loc, 8, i)?;
                        return Ok(());
                    }
                    GROUP5_PUSH if op == 0xff => {
                        let sz = stack_operand_size(i);
                        let v = self.load_operand(loc, sz, i)?;
                        self.push(sz, v)?;
                    }
                    _ => return Err(self.unimplemented_opcode(i)),
                }
            }
            _ => return self.exec_two_byte(i, next),
        }
        self.rip = next;
        Ok(())
    }

    /// #GP(0) unless an I/O access of `size` bytes at `port` is permitted. At CPL <= IOPL it
    /// always is; otherwise every bit the access covers in the TSS I/O permission bitmap must be
    /// clear, and the bitmap must lie inside the TSS limit (SDM 3A 20.5; QEMU `check_io`).
    pub(super) fn check_io_permission(&mut self, port: u16, size: u8) -> ExecResult<()> {
        if self.cpl <= self.iopl() {
            return Ok(());
        }
        let tss_limit = self.tr.limit as u64;
        // The 16-bit map base is the last field of the fixed part of the TSS.
        if tss_limit < TSS64_IO_MAP_BASE_OFFSET + 1 {
            return Err(Exception::general_protection(0));
        }
        let map_base = self.read_sys(self.tr.base.wrapping_add(TSS64_IO_MAP_BASE_OFFSET), 2)?;
        let byte_offset = map_base + (port as u64 >> 3);
        // The bits are read as a 16-bit word because an access can straddle a bitmap byte.
        if byte_offset + 1 > tss_limit {
            return Err(Exception::general_protection(0));
        }
        let word = self.read_sys(self.tr.base.wrapping_add(byte_offset), 2)?;
        if (word >> (port & 7)) & ((1 << size) - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(())
    }

    /// in/out with an imm8 or DX port.
    fn port_io(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op;
        let size = if op & 1 == 0 {
            1
        } else if i.p66 {
            2
        } else {
            4
        };
        // e4..e7 take an imm8 port, ec..ef use DX.
        let port = if op < 0xe8 {
            i.imm as u8 as u16
        } else {
            self.gpr[GPR_RDX] as u16
        };
        self.check_io_permission(port, size)?;
        #[cfg(not(freestanding))]
        if op & 2 != 0 && port == super::bios::TRAP_PORT && self.bios.is_some() {
            self.bios_trap()?;
            self.rip = next;
            return Ok(());
        }
        let now = self.now_ns();
        if op & 2 == 0 {
            let v = self.dev.lock().io_in(port, size, now);
            self.write_gpr(0, size, true, v as u64);
        } else {
            let v = self.read_gpr(0, size, true) as u32;
            self.dev.lock().io_out(port, size, v, now);
            // A device asked for a machine reset; RIP is advanced first so the exit state is
            // after the out.
            if self.dev.lock().reset_requested {
                self.rip = next;
                return Err(Exception::Exit(ExitReason::Reset));
            }
            // The ACPI sleep register: S5 is the only state the DSDT offers and nothing wakes it.
            if self.dev.lock().shutdown_requested() {
                self.rip = next;
                return Err(Exception::Exit(ExitReason::Shutdown));
            }
        }
        self.rip = next;
        Ok(())
    }

    /// Raises #UD for an opcode that is valid but not implemented, logging each distinct
    /// (opcode, ModRM.reg) once.
    pub fn unimplemented_opcode(&mut self, i: &Insn) -> Exception {
        // ModRM.reg has 8 values, so this keeps each opcode's /0../7 variants on distinct keys.
        const REG_VALUES_PER_OPCODE: u64 = 8;
        let rip = self.rip;
        self.log_once(
            UNIMPLEMENTED_OPCODE_LOG_BASE
                + i.op as u64 * REG_VALUES_PER_OPCODE
                + (i.reg as u64 & 7),
            || {
                format!(
                    "unimplemented opcode {:#05x} /{} at rip {:#x} (modrm {:#04x})",
                    i.op,
                    i.reg & 7,
                    rip,
                    i.modrm
                )
            },
        );
        Exception::invalid_opcode()
    }

    /// `0f xx` opcodes (`Insn::op` is `0x100 | xx`); anything not handled here goes to SSE.
    fn exec_two_byte(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let op = i.op;
        let osz = i.opsize;
        let rex = i.rex_present;
        match op {
            // group 6 / group 7 (sldt, ltr, sgdt, lidt, ...)
            0x100 | 0x101 => return self.sys_group(i, next),
            0x102 | 0x103 => return self.load_access_rights_or_limit(i, next),
            0x105 => return self.do_syscall(next),
            0x134 => return self.do_sysenter(),
            0x135 => return self.do_sysexit(),
            0x107 => return self.do_sysret(i),
            // clts
            0x106 => {
                self.need_cpl0()?;
                self.cr0 &= !CR0_TS;
            }
            // invd / wbinvd: privileged no-ops
            0x108 | 0x109 => self.need_cpl0()?,
            // ud2
            0x10b => return Err(Exception::invalid_opcode()),
            // prefetch and the reserved hint-nop space (includes multi-byte nop, endbr64)
            0x10d | 0x118..=0x11f => {}
            // mov to/from cr and dr
            0x120..=0x123 => self.mov_cr_dr(i)?,
            // wrmsr: MSR in ECX, value in EDX:EAX
            0x130 => {
                self.need_cpl0()?;
                let v = combine_dwords(self.gpr[GPR_RAX], self.gpr[GPR_RDX]);
                self.wrmsr(self.gpr[GPR_RCX] as u32, v)?;
            }
            // invpcid
            0x282 if i.p66 => return self.invpcid(i, next),
            // rdtsc
            0x131 => {
                if self.cr4 & CR4_TSD != 0 && self.cpl != 0 {
                    return Err(Exception::general_protection(0));
                }
                let t = self.tsc();
                self.set_edx_eax(t);
            }
            // rdmsr
            0x132 => {
                self.need_cpl0()?;
                let v = self.rdmsr(self.gpr[GPR_RCX] as u32)?;
                self.set_edx_eax(v);
            }
            // rdpmc: the stored value of one of the four legacy AMD counters
            0x133 => self.rdpmc()?,
            // cmovcc: the source is always read (it can fault); a 32-bit cmov that does not move
            // still zero-extends the destination.
            0x140..=0x14f => {
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, osz, i)?;
                if self.condition_holds((i.op & 15) as u8) {
                    self.write_gpr(i.reg, osz, true, v);
                } else if osz == 4 {
                    self.gpr[i.reg as usize] &= DWORD_MASK;
                }
            }
            // jcc rel32
            0x180..=0x18f => {
                if self.condition_holds((i.op & 15) as u8) {
                    self.rip = next.wrapping_add(i.imm as u64);
                    return Ok(());
                }
            }
            // setcc
            0x190..=0x19f => {
                let loc = self.modrm_location(i, next)?;
                let v = self.condition_holds((i.op & 15) as u8) as u64;
                self.store_operand(loc, 1, i, v)?;
            }
            // push fs / push gs
            0x1a0 | 0x1a8 => {
                let reg = if op == 0x1a0 { SEG_FS } else { SEG_GS };
                let sel = self.seg[reg].sel as u64;
                let sz = stack_operand_size(i);
                self.push(sz, sel)?;
            }
            // pop fs / pop gs: GPR_RSP is rolled back if the segment load faults
            0x1a1 | 0x1a9 => {
                let sz = stack_operand_size(i);
                let v = self.pop(sz)? as u16;
                let reg = if op == 0x1a1 { SEG_FS } else { SEG_GS };
                if let Err(e) = self.load_data_seg(reg, v) {
                    self.gpr[GPR_RSP] = self.gpr[GPR_RSP].wrapping_sub(sz as u64);
                    return Err(e);
                }
            }
            0x1a2 => {
                self.cpuid();
                self.serialize();
            }
            // bt / bts / btr / btc r/m, r
            0x1a3 => self.bit_op(i, next, BIT_TEST, false)?,
            0x1ab => self.bit_op(i, next, BIT_SET, false)?,
            0x1b3 => self.bit_op(i, next, BIT_RESET, false)?,
            0x1bb => self.bit_op(i, next, BIT_COMPLEMENT, false)?,
            // group 8: bt/bts/btr/btc r/m, imm8
            0x1ba => {
                let group_op = i.reg & 7;
                if group_op < GROUP8_FIRST_BIT_OP {
                    return Err(Exception::invalid_opcode());
                }
                self.bit_op(i, next, group_op - GROUP8_FIRST_BIT_OP, true)?
            }
            // shld imm8 / cl, shrd imm8 / cl
            0x1a4 | 0x1a5 | 0x1ac | 0x1ad => {
                let loc = self.modrm_location(i, next)?;
                let a = self.load_operand(loc, osz, i)?;
                let b = self.read_gpr(i.reg, osz, rex);
                let count = if op & 1 == 0 {
                    i.imm as u64
                } else {
                    self.gpr[GPR_RCX] & BYTE_MASK
                };
                let bits = osz as u32 * 8;
                let c = (count & shift_count_mask(osz)) as u32;
                if c != 0 {
                    let (r, cf) = double_shift(op < 0x1a8, a, b, c, osz);
                    // OF is set when the sign bit changed.
                    let of = ((r ^ a) >> (bits - 1)) & 1;
                    let mut f = szp_flags(r, osz);
                    if cf != 0 {
                        f |= CF;
                    }
                    if of != 0 {
                        f |= OF;
                    }
                    let saved = self.save_flags();
                    self.set_eflags_arith(f);
                    self.store_result(loc, osz, i, r, saved)?;
                }
            }
            // group 15 (fences, fxsave, ldmxcsr, ...)
            0x1ae => return self.group_ae(i, next),
            // imul r, r/m
            0x1af => {
                let loc = self.modrm_location(i, next)?;
                let b = self.load_operand(loc, osz, i)?;
                let a = self.read_gpr(i.reg, osz, true);
                let r = self.imul_truncated(a, b, osz);
                self.write_gpr(i.reg, osz, true, r);
            }
            // cmpxchg r/m, r. The destination is stored even when the compare fails, as on
            // hardware, so a write fault is taken either way.
            0x1b0 | 0x1b1 => {
                let size = if op == 0x1b0 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let dest = self.load_operand(loc, size, i)?;
                let acc = self.read_gpr(0, size, true);
                let m = mask(size);
                let saved = self.save_flags();
                self.set_lazy_flags(LazyFlagsOp::Sub, size, acc.wrapping_sub(dest) & m, dest, 0);
                if acc == dest {
                    let src = self.read_gpr(i.reg, size, rex);
                    self.store_result(loc, size, i, src, saved)?;
                } else {
                    self.store_result(loc, size, i, dest, saved)?;
                    self.write_gpr(0, size, true, dest);
                }
            }
            // xadd r/m, r
            0x1c0 | 0x1c1 => {
                let size = if op == 0x1c0 { 1 } else { osz };
                let loc = self.modrm_location(i, next)?;
                let dest = self.load_operand(loc, size, i)?;
                let src = self.read_gpr(i.reg, size, rex);
                let saved = self.save_flags();
                let sum = self.alu(ALU_ADD, dest, src, size);
                self.store_result(loc, size, i, sum, saved)?;
                self.write_gpr(i.reg, size, rex, dest);
            }
            // movnti
            0x1c3 => {
                if i.md == 3 {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let v = self.read_gpr(i.reg, osz, true);
                self.store_operand(loc, osz, i, v)?;
            }
            // cmpxchg8b m64 / cmpxchg16b m128 (group 9 /1 without / with REX.W)
            0x1c7 => {
                // rdrand /6 and rdseed /7 take a register
                if i.md == 3 && matches!(i.reg & 7, 6 | 7) && i.rep == 0 {
                    self.rdrand(i.rm, osz);
                    self.rip = next;
                    return Ok(());
                }
                // rdpid (F3 0F C7 /7): IA32_TSC_AUX, 64 bits wide in 64-bit mode
                if i.md == 3 && i.reg & 7 == 7 && i.rep == REP_REPE {
                    let size = if self.seg[SEG_CS].long() { 8 } else { 4 };
                    self.write_gpr(i.rm, size, true, self.tsc_aux & 0xffff_ffff);
                    self.rip = next;
                    return Ok(());
                }
                // xrstors /3, xsavec /4, xsaves /5
                if i.md != 3 && matches!(i.reg & 7, 3..=5) && i.rep == 0 {
                    let a = self.linear_address(i, self.effective_address(i, next))?;
                    match i.reg & 7 {
                        3 => self.xrstors(a)?,
                        4 => self.xsavec(a)?,
                        _ => self.xsaves(a)?,
                    }
                    self.rip = next;
                    return Ok(());
                }
                if i.md == 3 || i.reg & 7 != 1 {
                    return Err(self.unimplemented_opcode(i));
                }
                let a = self.linear_address(i, self.effective_address(i, next))?;
                if osz == 8 {
                    return self.cmpxchg16b(a, next);
                }
                let current = self.load_operand(Loc::Mem(a), 8, i)?;
                let expected = combine_dwords(self.gpr[GPR_RAX], self.gpr[GPR_RDX]);
                let f = self.eflags_full();
                if current == expected {
                    let replacement = combine_dwords(self.gpr[GPR_RBX], self.gpr[GPR_RCX]);
                    self.store_operand(Loc::Mem(a), 8, i, replacement)?;
                    self.set_eflags_arith(f | ZF);
                } else {
                    self.set_edx_eax(current);
                    self.set_eflags_arith(f & !ZF);
                }
            }
            // bswap: a 16-bit bswap is undefined; it is treated as the 32-bit form
            0x1c8..=0x1cf => {
                let r = opcode_register(i) as usize;
                if osz == 8 {
                    self.gpr[r] = self.gpr[r].swap_bytes();
                } else {
                    self.gpr[r] = (self.gpr[r] as u32).swap_bytes() as u64;
                }
            }
            // movzx / movsx from byte or word
            0x1b6 | 0x1b7 | 0x1be | 0x1bf => {
                let source_size = if op & 1 == 0 { 1 } else { 2 };
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, source_size, i)?;
                let v = if op < 0x1b8 {
                    v
                } else if source_size == 1 {
                    v as u8 as i8 as i64 as u64
                } else {
                    v as u16 as i16 as i64 as u64
                };
                self.write_gpr(i.reg, osz, true, v);
            }
            // popcnt (F3 0F B8). Without F3 this opcode is JMPE, which does not exist here.
            0x1b8 => {
                if i.rep != REP_REPE {
                    return Err(self.unimplemented_opcode(i));
                }
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, osz, i)? & mask(osz);
                self.write_gpr(i.reg, osz, true, v.count_ones() as u64);
                // CF, OF, SF, AF and PF are cleared; ZF reports a zero source.
                self.set_eflags_arith(if v == 0 { ZF } else { 0 });
            }
            // lzcnt (F3 0F BD): counts leading zeros within the operand size, so a zero source
            // gives the operand width and CF. On a CPU without ABM this opcode is plain bsr, which
            // is why CPUID must advertise ABM (0x8000_0001 ECX) together with this arm.
            0x1bd if i.rep == REP_REPE => {
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, osz, i)? & mask(osz);
                let operand_bits = osz as u32 * 8;
                let count = v.leading_zeros() - (u64::BITS - operand_bits);
                self.write_gpr(i.reg, osz, true, count as u64);
                // SF/PF are architecturally undefined; they follow the count, as the JIT's
                // `LazyFlagsOp::SzpCfOf` record has them.
                let mut f = szp_flags(count as u64, osz);
                if v == 0 {
                    f |= CF;
                }
                self.set_eflags_arith(f);
            }
            // tzcnt (F3 0F BC): trailing zeros within the operand size, so a zero source gives the
            // operand width and CF. Plain bsf without the prefix, hence BMI1 and this arm go
            // together.
            0x1bc if i.rep == REP_REPE => {
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, osz, i)? & mask(osz);
                let count = if v == 0 {
                    osz as u32 * 8
                } else {
                    v.trailing_zeros()
                };
                self.write_gpr(i.reg, osz, true, count as u64);
                let mut f = szp_flags(count as u64, osz);
                if v == 0 {
                    f |= CF;
                }
                self.set_eflags_arith(f);
            }
            // bsf / bsr: a zero source sets ZF and leaves the destination unchanged
            0x1bc | 0x1bd => {
                let loc = self.modrm_location(i, next)?;
                let v = self.load_operand(loc, osz, i)? & mask(osz);
                // Everything but ZF is undefined: SF/PF follow the source and CF/OF/AF clear, as
                // in QEMU and as the JIT's `LazyFlagsOp::SzpCfOf` record has them.
                if v != 0 {
                    let bit_index = if op == 0x1bc {
                        v.trailing_zeros()
                    } else {
                        63 - v.leading_zeros()
                    };
                    self.write_gpr(i.reg, osz, true, bit_index as u64);
                }
                self.set_eflags_arith(szp_flags(v, osz));
            }
            _ => {
                if self.sse(i, next)? {
                    self.rip = next;
                    return Ok(());
                }
                return Err(self.unimplemented_opcode(i));
            }
        }
        self.rip = next;
        Ok(())
    }
}

#[cfg(test)]
mod exec_tests {
    //! Unit tests for the interpreter (`exec.rs`). Snippets are hand-assembled with `hack/asm.sh`
    //! (the assembly is the comment above each byte array); registers and memory are prepared from
    //! Rust, so most snippets are a single instruction plus `hlt`. Every case also runs on the host JIT
    //! when there is one and must leave identical state (`Pair`), which makes this suite the JIT's
    //! corpus for the instructions it compiles inline.
    //!
    //! Flags that the SDM leaves undefined are masked out of the expectations; the JIT comparison
    //! still covers them because it compares the whole of EFLAGS.

    use super::*;
    use crate::cpu::exception::{
        VECTOR_DEVICE_NOT_AVAILABLE, VECTOR_DIVIDE_ERROR, VECTOR_GENERAL_PROTECTION,
        VECTOR_INVALID_OPCODE, VECTOR_PAGE_FAULT,
    };
    use crate::testkit::*;

    const ALL_ARITH: u64 = ARITH;
    const NO_FLAGS: u64 = 0;
    /// Bits 63:48 of a canonical address are copies of bit 47, so this is the first non-canonical one.
    const NON_CANONICAL: u64 = 0x0000_8000_0000_0000;
    /// Linear addresses at or above 4 GiB are unmapped in the harness identity map.
    const UNMAPPED: u64 = 0x1_0000_0000;
    /// Scratch layout used by the tests, all inside the harness's free RAM window.
    const DATA: u64 = SCRATCH + 0x4_0000;
    const GDT_BASE: u64 = SCRATCH + 0x1_0000;
    const TSS_BASE: u64 = SCRATCH + 0x2_0000;
    const TOP_TABLES: u64 = SCRATCH + 0x8_0000;
    const TOP_FRAME: u64 = SCRATCH + 0x9_0000;

    const PAGE_TABLE_PML4: u64 = 0x9000;
    const PAGE_TABLE_PDPT: u64 = 0xa000;
    const PAGE_TABLE_FIRST_PD: u64 = 0xb000;
    const PTE_PRESENT_WRITABLE: u64 = 3;
    const PTE_USER: u64 = 4;
    const PD_ENTRIES: u64 = 512;
    const PD_COUNT: u64 = 4;
    const PML4_LAST_SLOT: u64 = 511;
    const PAGE_SIZE_BYTES: u64 = 4096;
    const HUGE_PAGE_BYTES: u64 = 2 << 20;
    /// Highest 4 KiB page of the address space, mapped on demand by `map_top_page`.
    const TOP_PAGE_LINEAR: u64 = 0xffff_ffff_ffff_f000;

    // Test GDT (installed by `install_gdt`). Selectors are index * 8; the RPL is added at the use site.
    const SEL_NULL: u16 = 0;
    const SEL_KCODE: u16 = 0x08;
    const SEL_KDATA: u16 = 0x10;
    const SEL_UCODE: u16 = 0x18;
    const SEL_UDATA: u16 = 0x20;
    const SEL_CONFORMING_DPL0: u16 = 0x28;
    const SEL_CONFORMING_DPL3: u16 = 0x30;
    const SEL_CODE_NOT_PRESENT: u16 = 0x38;
    const SEL_DATA_NOT_PRESENT: u16 = 0x40;
    const SEL_CODE32: u16 = 0x48;
    const SEL_CODE_EXEC_ONLY: u16 = 0x50;
    const SEL_SYSTEM: u16 = 0x58;
    const SEL_RODATA: u16 = 0x68;
    const SEL_DATA_DPL1: u16 = 0x70;
    const SEL_CODE_DPL1: u16 = 0x78;
    const SEL_DATA_DPL2: u16 = 0x80;
    const SEL_CODE_DPL2: u16 = 0x88;
    const SEL_UDATA_NOT_PRESENT: u16 = 0x90;
    const SEL_BEYOND_GDT: u16 = 0x1f8;
    const GDT_ENTRIES: usize = 0x98 / 8;

    const ACCESS_PRESENT: u64 = 0x80;
    const ACCESS_CODE_READ: u64 = 0x9a;
    const ACCESS_CODE_CONFORMING: u64 = 0x9e;
    const ACCESS_CODE_EXEC_ONLY: u64 = 0x98;
    const ACCESS_DATA_RW: u64 = 0x92;
    const ACCESS_DATA_RO: u64 = 0x90;
    const ACCESS_TSS_AVAILABLE: u64 = 0x89;
    const ACCESS_DPL_SHIFT: u32 = 5;
    const DESC_FLAGS_CODE64: u64 = 0xa;
    const DESC_FLAGS_32BIT: u64 = 0xc;
    const DESC_LIMIT_AND_BASE_LOW: u64 = 0x000f_0000_0000_ffff;

    fn descriptor(access: u64, flags: u64, dpl: u64) -> u64 {
        DESC_LIMIT_AND_BASE_LOW | ((access | dpl << ACCESS_DPL_SHIFT) << 40) | (flags << 52)
    }

    fn gdt_entries() -> [u64; GDT_ENTRIES] {
        let mut gdt = [0u64; GDT_ENTRIES];
        let index = |sel: u16| sel as usize / 8;
        gdt[index(SEL_KCODE)] = descriptor(ACCESS_CODE_READ, DESC_FLAGS_CODE64, 0);
        gdt[index(SEL_KDATA)] = descriptor(ACCESS_DATA_RW, DESC_FLAGS_32BIT, 0);
        gdt[index(SEL_UCODE)] = descriptor(ACCESS_CODE_READ, DESC_FLAGS_CODE64, 3);
        gdt[index(SEL_UDATA)] = descriptor(ACCESS_DATA_RW, DESC_FLAGS_32BIT, 3);
        gdt[index(SEL_CONFORMING_DPL0)] = descriptor(ACCESS_CODE_CONFORMING, DESC_FLAGS_CODE64, 0);
        gdt[index(SEL_CONFORMING_DPL3)] = descriptor(ACCESS_CODE_CONFORMING, DESC_FLAGS_CODE64, 3);
        gdt[index(SEL_CODE_NOT_PRESENT)] =
            descriptor(ACCESS_CODE_READ & !ACCESS_PRESENT, DESC_FLAGS_CODE64, 0);
        gdt[index(SEL_DATA_NOT_PRESENT)] =
            descriptor(ACCESS_DATA_RW & !ACCESS_PRESENT, DESC_FLAGS_32BIT, 0);
        gdt[index(SEL_CODE32)] = descriptor(ACCESS_CODE_READ, DESC_FLAGS_32BIT, 0);
        gdt[index(SEL_CODE_EXEC_ONLY)] = descriptor(ACCESS_CODE_EXEC_ONLY, DESC_FLAGS_CODE64, 0);
        gdt[index(SEL_SYSTEM)] = descriptor(ACCESS_TSS_AVAILABLE, 0, 0);
        gdt[index(SEL_RODATA)] = descriptor(ACCESS_DATA_RO, DESC_FLAGS_32BIT, 0);
        gdt[index(SEL_DATA_DPL1)] = descriptor(ACCESS_DATA_RW, DESC_FLAGS_32BIT, 1);
        gdt[index(SEL_CODE_DPL1)] = descriptor(ACCESS_CODE_READ, DESC_FLAGS_CODE64, 1);
        gdt[index(SEL_DATA_DPL2)] = descriptor(ACCESS_DATA_RW, DESC_FLAGS_32BIT, 2);
        gdt[index(SEL_UDATA_NOT_PRESENT)] =
            descriptor(ACCESS_DATA_RW & !ACCESS_PRESENT, DESC_FLAGS_32BIT, 3);
        gdt[index(SEL_CODE_DPL2)] = descriptor(ACCESS_CODE_READ, DESC_FLAGS_CODE64, 2);
        gdt
    }

    /// What a faulting instruction reports: `Stop::Fault` for a hardware exception.
    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    fn gp(error_code: u32) -> Stop {
        fault(VECTOR_GENERAL_PROTECTION, Some(error_code))
    }

    fn ss_fault(error_code: u32) -> Stop {
        fault(VECTOR_STACK_FAULT, Some(error_code))
    }

    fn np(error_code: u32) -> Stop {
        fault(VECTOR_SEGMENT_NOT_PRESENT, Some(error_code))
    }

    fn de() -> Stop {
        fault(VECTOR_DIVIDE_ERROR, None)
    }

    fn ud() -> Stop {
        fault(VECTOR_INVALID_OPCODE, None)
    }

    fn selector_code(selector: u16) -> u32 {
        (selector & !SELECTOR_RPL_MASK) as u32
    }

    /// A snippet on two CPUs (interpreter, and the JIT when the host has one) that are reset and run
    /// side by side; any difference in the outcome fails the test. The interpreter's state is what
    /// the accessors and the assertions look at.
    struct Pair {
        interp: Harness,
        jit: Option<Harness>,
    }

    fn reset_registers(h: &mut Harness) {
        h.cpu.gpr = [0; 16];
        h.cpu.gpr[RSP] = STACK_TOP;
        h.cpu.rip = CODE_BASE;
        h.cpu.eflags = RESERVED_ONE;
        h.cpu.lazy_op = LazyFlagsOp::Eflags;
        h.cpu.halted = false;
        h.cpu.interrupt_shadow = false;
        h.write(DATA, &[0u8; 0x1000]);
        h.write(STACK_TOP - 0x1000, &[0u8; 0x1000]);
    }

    /// Memory the two engines must agree on after a run: the harness scratch area, the stack and low
    /// memory.
    const COMPARED_RAM: [(u64, usize); 3] = [
        (0, 0x1000),
        (SCRATCH, SCRATCH_BYTES as usize),
        (STACK_TOP - 0x2000, 0x2000),
    ];

    impl Pair {
        fn new(code: &[u8]) -> Pair {
            let interp = Harness::new(code);
            let mut jit = Harness::new(code);
            let jit = jit.enable_jit().then_some(jit);
            Pair { interp, jit }
        }

        /// Interpreter only, for behavior the JIT is not expected to reproduce (it never runs the
        /// snippet) or that a test needs to set up by hand on one CPU.
        fn interpreter_only(code: &[u8]) -> Pair {
            Pair {
                interp: Harness::new(code),
                jit: None,
            }
        }

        /// Resets registers, applies `setup` to each CPU and runs at most `limit` instructions.
        fn run_for(&mut self, limit: u64, setup: impl Fn(&mut Harness)) -> Stop {
            reset_registers(&mut self.interp);
            setup(&mut self.interp);
            let stop = self.interp.run(limit);
            if let Some(jit) = &mut self.jit {
                reset_registers(jit);
                setup(jit);
                // Rosetta runs generated code unreliably when several threads do it at once.
                let _generated_code_turn = crate::jit::serialize_jit_tests();
                let jit_stop = jit.run(limit);
                assert_eq!(jit_stop, stop, "stop reason differs (jit vs interpreter)");
                assert_eq!(
                    jit.snapshot(),
                    self.interp.snapshot(),
                    "state differs (jit vs interpreter)"
                );
                for (start, len) in COMPARED_RAM {
                    assert!(
                        jit.read(start, len) == self.interp.read(start, len),
                        "RAM at {start:#x} differs (jit vs interpreter)"
                    );
                }
            }
            stop
        }

        fn run(&mut self, setup: impl Fn(&mut Harness)) -> Stop {
            self.run_for(1_000_000, setup)
        }

        /// Runs to `hlt` and returns the pair for inspection.
        fn halted(&mut self, setup: impl Fn(&mut Harness)) -> &mut Pair {
            let stop = self.run(setup);
            assert_eq!(stop, Stop::Halted, "{}", self.interp.cpu.dump_regs());
            self
        }

        fn reg(&self, index: usize) -> u64 {
            self.interp.cpu.gpr[index]
        }

        fn flags(&self) -> u64 {
            self.interp.cpu.eflags_full()
        }

        fn cpu(&self) -> &Cpu {
            &self.interp.cpu
        }
    }

    /// Runs a snippet once to `hlt` from the given register values (and no other setup).
    fn run_regs(code: &[u8], regs: &[(usize, u64)]) -> Pair {
        let mut pair = Pair::new(code);
        pair.halted(|h| {
            for &(index, value) in regs {
                h.cpu.gpr[index] = value;
            }
        });
        pair
    }

    /// Runs a snippet once from the given register values and returns the stop reason.
    fn run_regs_stop(code: &[u8], regs: &[(usize, u64)]) -> (Pair, Stop) {
        let mut pair = Pair::new(code);
        let stop = pair.run(|h| {
            for &(index, value) in regs {
                h.cpu.gpr[index] = value;
            }
        });
        (pair, stop)
    }

    /// Like `run_regs_stop` without the JIT comparison, for behavior the JIT does not reproduce yet.
    fn interpreter_regs_stop(code: &[u8], regs: &[(usize, u64)]) -> (Pair, Stop) {
        let mut pair = Pair::interpreter_only(code);
        let stop = pair.run(|h| {
            for &(index, value) in regs {
                h.cpu.gpr[index] = value;
            }
        });
        (pair, stop)
    }

    #[track_caller]
    fn assert_flags(actual: u64, mask: u64, expected: u64, context: &str) {
        assert_eq!(
            actual & mask,
            expected & mask,
            "flags {context}: got {} want {}",
            flag_names(actual & mask),
            flag_names(expected & mask)
        );
    }

    fn flag_names(flags: u64) -> String {
        let names = [
            (CF, "CF"),
            (PF, "PF"),
            (AF, "AF"),
            (ZF, "ZF"),
            (SF, "SF"),
            (OF, "OF"),
            (DF, "DF"),
            (IF, "IF"),
        ];
        let set: Vec<&str> = names
            .iter()
            .filter(|(bit, _)| flags & bit != 0)
            .map(|(_, name)| *name)
            .collect();
        format!("[{}]", set.join(" "))
    }

    fn parity_flag(value: u64) -> u64 {
        if (value as u8).count_ones().is_multiple_of(2) {
            PF
        } else {
            0
        }
    }

    fn zero_sign_parity(value: u64, size: u8) -> u64 {
        let m = mask(size);
        let bits = size as u32 * 8;
        let mut f = parity_flag(value & m);
        if value & m == 0 {
            f |= ZF;
        }
        if (value >> (bits - 1)) & 1 != 0 {
            f |= SF;
        }
        f
    }

    fn install_gdt(h: &mut Harness) {
        for (index, entry) in gdt_entries().iter().enumerate() {
            h.write_u64(GDT_BASE + index as u64 * 8, *entry);
        }
        h.cpu.gdtr = DescriptorTable {
            base: GDT_BASE,
            limit: (GDT_ENTRIES * 8 - 1) as u16,
        };
    }

    /// Segment cache for a selector of the test GDT, as a `mov` or `iretq` to it would load.
    fn gdt_segment(h: &mut Harness, selector: u16) -> Seg {
        h.cpu
            .load_desc_seg(selector, false)
            .expect("test GDT selector")
    }

    /// Puts the CPU at `cpl` running in the test GDT's code segment of that privilege (selectors
    /// with RPL = `cpl`, data segments DPL 0 except at CPL 3). CPLs 1 and 2 still use supervisor
    /// paging, so the harness identity map works unchanged for them.
    fn enter_cpl(h: &mut Harness, cpl: u8) {
        install_gdt(h);
        let (code, data) = match cpl {
            0 => (SEL_KCODE, SEL_KDATA),
            1 => (SEL_CODE_DPL1 | 1, SEL_DATA_DPL1 | 1),
            2 => (SEL_CODE_DPL2 | 2, SEL_DATA_DPL2 | 2),
            _ => (SEL_UCODE | 3, SEL_UDATA | 3),
        };
        let mut code_seg = gdt_segment(h, code & !SELECTOR_RPL_MASK);
        code_seg.sel = code;
        h.cpu.seg[SEG_CS] = code_seg;
        let mut data_seg = gdt_segment(h, data & !SELECTOR_RPL_MASK);
        data_seg.sel = data;
        h.cpu.seg[SEG_SS] = data_seg;
        h.cpu.set_cpl(cpl);
        if cpl == USER_CPL {
            make_pages_user_accessible(h);
        }
    }

    /// Sets the U/S bit throughout the harness's identity map so CPL 3 may fetch and access it.
    fn make_pages_user_accessible(h: &mut Harness) {
        h.write_u64(PAGE_TABLE_PML4, h.read_u64(PAGE_TABLE_PML4) | PTE_USER);
        for page_directory in 0..PD_COUNT {
            let pdpt_entry = PAGE_TABLE_PDPT + page_directory * 8;
            h.write_u64(pdpt_entry, h.read_u64(pdpt_entry) | PTE_USER);
            for slot in 0..PD_ENTRIES {
                let entry = PAGE_TABLE_FIRST_PD + page_directory * PAGE_SIZE_BYTES + slot * 8;
                h.write_u64(entry, h.read_u64(entry) | PTE_USER);
            }
        }
        h.cpu.flush_tlb();
    }

    /// Maps the last 4 KiB page of the 64-bit address space (`TOP_PAGE_LINEAR`) onto `TOP_FRAME`, so
    /// accesses can wrap around the end of the address space.
    fn map_top_page(h: &mut Harness) {
        let pdpt = TOP_TABLES;
        let pd = TOP_TABLES + PAGE_SIZE_BYTES;
        let pt = TOP_TABLES + 2 * PAGE_SIZE_BYTES;
        h.write(TOP_TABLES, &vec![0u8; 3 * PAGE_SIZE_BYTES as usize]);
        h.write_u64(
            PAGE_TABLE_PML4 + PML4_LAST_SLOT * 8,
            pdpt | PTE_PRESENT_WRITABLE,
        );
        h.write_u64(pdpt + PML4_LAST_SLOT * 8, pd | PTE_PRESENT_WRITABLE);
        h.write_u64(pd + PML4_LAST_SLOT * 8, pt | PTE_PRESENT_WRITABLE);
        h.write_u64(pt + PML4_LAST_SLOT * 8, TOP_FRAME | PTE_PRESENT_WRITABLE);
        h.cpu.flush_tlb();
    }

    /// Physical address backing a linear address inside the page mapped by `map_top_page`.
    fn top_page_physical(linear: u64) -> u64 {
        TOP_FRAME + (linear - TOP_PAGE_LINEAR)
    }

    /// Makes the 2 MiB page containing `linear` (below 1 GiB) not present.
    fn unmap_huge_page(h: &mut Harness, linear: u64) {
        let entry = PAGE_TABLE_FIRST_PD + (linear / HUGE_PAGE_BYTES) * 8;
        h.write_u64(entry, 0);
        h.cpu.flush_tlb();
    }

    /// Makes the 2 MiB page containing `linear` (below 1 GiB) read-only and turns on CR0.WP, so
    /// stores from any privilege fault.
    fn write_protect_huge_page(h: &mut Harness, linear: u64) {
        let entry = PAGE_TABLE_FIRST_PD + (linear / HUGE_PAGE_BYTES) * 8;
        h.write_u64(entry, h.read_u64(entry) & !2);
        h.cpu.cr0 |= CR0_WP;
        h.cpu.flush_tlb();
    }

    /// A 2 MiB page well away from code, stack, GDT and data; `unmap_huge_page` on it gives a #PF target
    /// that is inside the identity map's RAM range.
    const HOLE: u64 = 0x140_0000;

    /// #PF error code of a write to a present page that forbids it (P | W), seen from supervisor mode.
    const PF_PROTECTION_WRITE: u32 = 3;
    const MSR_LSTAR: u64 = 0xc000_0082;
    /// STAR for the syscall/sysret tests: kernel CS 0x08 in bits 47:32, user base selector 0x13 in 63:48.
    const STAR_FOR_TESTS: u64 = 0x0013_0008_0000_0000;
    const EFER_SCE: u64 = 1;
    const ALU_SNIPPETS: &[AluSnippet] = &[
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADD_8_RR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADD_8_RM,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADD_8_MR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::ADD_8_RIA_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::ADD_8_RI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::ADD_8_MI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::ADD_8_RIA_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::ADD_8_RI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::ADD_8_MI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADD_8_RIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADD_8_MIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADD_4_RR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADD_4_RM,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADD_4_MR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::ADD_4_RIA_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::ADD_4_RI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::ADD_4_MI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::ADD_4_RIA_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::ADD_4_RI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::ADD_4_MI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADD_4_RIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADD_4_MIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADD_2_RR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADD_2_RM,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADD_2_MR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::ADD_2_RIA_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::ADD_2_RI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::ADD_2_MI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::ADD_2_RIA_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::ADD_2_RI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::ADD_2_MI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADD_2_RIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADD_2_MIS,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADD_1_RR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADD_1_RM,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADD_1_MR,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::ADD_1_RIA_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::ADD_1_RI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::ADD_1_MI_P,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::ADD_1_RIA_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::ADD_1_RI_N,
        },
        AluSnippet {
            op: ALU_ADD,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::ADD_1_MI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::OR_8_RR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::OR_8_RM,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::OR_8_MR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::OR_8_RIA_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::OR_8_RI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::OR_8_MI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::OR_8_RIA_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::OR_8_RI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::OR_8_MI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::OR_8_RIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::OR_8_MIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::OR_4_RR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::OR_4_RM,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::OR_4_MR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::OR_4_RIA_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::OR_4_RI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::OR_4_MI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::OR_4_RIA_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::OR_4_RI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::OR_4_MI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::OR_4_RIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::OR_4_MIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::OR_2_RR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::OR_2_RM,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::OR_2_MR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::OR_2_RIA_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::OR_2_RI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::OR_2_MI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::OR_2_RIA_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::OR_2_RI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::OR_2_MI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::OR_2_RIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::OR_2_MIS,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::OR_1_RR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::OR_1_RM,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::OR_1_MR,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::OR_1_RIA_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::OR_1_RI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::OR_1_MI_P,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::OR_1_RIA_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::OR_1_RI_N,
        },
        AluSnippet {
            op: ALU_OR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::OR_1_MI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADC_8_RR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADC_8_RM,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADC_8_MR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::ADC_8_RIA_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::ADC_8_RI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::ADC_8_MI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::ADC_8_RIA_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::ADC_8_RI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::ADC_8_MI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADC_8_RIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADC_8_MIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADC_4_RR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADC_4_RM,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADC_4_MR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::ADC_4_RIA_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::ADC_4_RI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::ADC_4_MI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::ADC_4_RIA_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::ADC_4_RI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::ADC_4_MI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADC_4_RIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADC_4_MIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADC_2_RR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADC_2_RM,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADC_2_MR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::ADC_2_RIA_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::ADC_2_RI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::ADC_2_MI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::ADC_2_RIA_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::ADC_2_RI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::ADC_2_MI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::ADC_2_RIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::ADC_2_MIS,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::ADC_1_RR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::ADC_1_RM,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::ADC_1_MR,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::ADC_1_RIA_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::ADC_1_RI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::ADC_1_MI_P,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::ADC_1_RIA_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::ADC_1_RI_N,
        },
        AluSnippet {
            op: ALU_ADC,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::ADC_1_MI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SBB_8_RR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SBB_8_RM,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SBB_8_MR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::SBB_8_RIA_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::SBB_8_RI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::SBB_8_MI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::SBB_8_RIA_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::SBB_8_RI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::SBB_8_MI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SBB_8_RIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SBB_8_MIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SBB_4_RR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SBB_4_RM,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SBB_4_MR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::SBB_4_RIA_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::SBB_4_RI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::SBB_4_MI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::SBB_4_RIA_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::SBB_4_RI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::SBB_4_MI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SBB_4_RIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SBB_4_MIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SBB_2_RR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SBB_2_RM,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SBB_2_MR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::SBB_2_RIA_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::SBB_2_RI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::SBB_2_MI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::SBB_2_RIA_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::SBB_2_RI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::SBB_2_MI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SBB_2_RIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SBB_2_MIS,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SBB_1_RR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SBB_1_RM,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SBB_1_MR,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::SBB_1_RIA_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::SBB_1_RI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::SBB_1_MI_P,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::SBB_1_RIA_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::SBB_1_RI_N,
        },
        AluSnippet {
            op: ALU_SBB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::SBB_1_MI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::AND_8_RR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::AND_8_RM,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::AND_8_MR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::AND_8_RIA_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::AND_8_RI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::AND_8_MI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::AND_8_RIA_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::AND_8_RI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::AND_8_MI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::AND_8_RIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::AND_8_MIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::AND_4_RR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::AND_4_RM,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::AND_4_MR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::AND_4_RIA_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::AND_4_RI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::AND_4_MI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::AND_4_RIA_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::AND_4_RI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::AND_4_MI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::AND_4_RIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::AND_4_MIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::AND_2_RR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::AND_2_RM,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::AND_2_MR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::AND_2_RIA_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::AND_2_RI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::AND_2_MI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::AND_2_RIA_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::AND_2_RI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::AND_2_MI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::AND_2_RIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::AND_2_MIS,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::AND_1_RR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::AND_1_RM,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::AND_1_MR,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::AND_1_RIA_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::AND_1_RI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::AND_1_MI_P,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::AND_1_RIA_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::AND_1_RI_N,
        },
        AluSnippet {
            op: ALU_AND,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::AND_1_MI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SUB_8_RR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SUB_8_RM,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SUB_8_MR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::SUB_8_RIA_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::SUB_8_RI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::SUB_8_MI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::SUB_8_RIA_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::SUB_8_RI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::SUB_8_MI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SUB_8_RIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SUB_8_MIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SUB_4_RR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SUB_4_RM,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SUB_4_MR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::SUB_4_RIA_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::SUB_4_RI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::SUB_4_MI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::SUB_4_RIA_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::SUB_4_RI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::SUB_4_MI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SUB_4_RIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SUB_4_MIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SUB_2_RR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SUB_2_RM,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SUB_2_MR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::SUB_2_RIA_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::SUB_2_RI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::SUB_2_MI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::SUB_2_RIA_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::SUB_2_RI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::SUB_2_MI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::SUB_2_RIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::SUB_2_MIS,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::SUB_1_RR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::SUB_1_RM,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::SUB_1_MR,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::SUB_1_RIA_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::SUB_1_RI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::SUB_1_MI_P,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::SUB_1_RIA_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::SUB_1_RI_N,
        },
        AluSnippet {
            op: ALU_SUB,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::SUB_1_MI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::XOR_8_RR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::XOR_8_RM,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::XOR_8_MR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::XOR_8_RIA_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::XOR_8_RI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::XOR_8_MI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::XOR_8_RIA_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::XOR_8_RI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::XOR_8_MI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::XOR_8_RIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::XOR_8_MIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::XOR_4_RR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::XOR_4_RM,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::XOR_4_MR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::XOR_4_RIA_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::XOR_4_RI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::XOR_4_MI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::XOR_4_RIA_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::XOR_4_RI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::XOR_4_MI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::XOR_4_RIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::XOR_4_MIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::XOR_2_RR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::XOR_2_RM,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::XOR_2_MR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::XOR_2_RIA_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::XOR_2_RI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::XOR_2_MI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::XOR_2_RIA_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::XOR_2_RI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::XOR_2_MI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::XOR_2_RIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::XOR_2_MIS,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::XOR_1_RR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::XOR_1_RM,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::XOR_1_MR,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::XOR_1_RIA_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::XOR_1_RI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::XOR_1_MI_P,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::XOR_1_RIA_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::XOR_1_RI_N,
        },
        AluSnippet {
            op: ALU_XOR,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::XOR_1_MI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::CMP_8_RR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::CMP_8_RM,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::CMP_8_MR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::CMP_8_RIA_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::CMP_8_RI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::CMP_8_MI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::CMP_8_RIA_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::CMP_8_RI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::CMP_8_MI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::CMP_8_RIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::CMP_8_MIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::CMP_4_RR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::CMP_4_RM,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::CMP_4_MR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::CMP_4_RIA_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::CMP_4_RI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::CMP_4_MI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::CMP_4_RIA_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::CMP_4_RI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::CMP_4_MI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::CMP_4_RIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::CMP_4_MIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::CMP_2_RR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::CMP_2_RM,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::CMP_2_MR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::CMP_2_RIA_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::CMP_2_RI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::CMP_2_MI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::CMP_2_RIA_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::CMP_2_RI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::CMP_2_MI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-3),
            code: code::CMP_2_RIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-3),
            code: code::CMP_2_MIS,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::CMP_1_RR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Mem,
            code: code::CMP_1_RM,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::CMP_1_MR,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::CMP_1_RIA_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::CMP_1_RI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::CMP_1_MI_P,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::CMP_1_RIA_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::CMP_1_RI_N,
        },
        AluSnippet {
            op: ALU_CMP,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::CMP_1_MI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::TEST_8_RR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::TEST_8_MR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::TEST_8_RIA_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::TEST_8_RI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::TEST_8_MI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::TEST_8_RIA_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::TEST_8_RI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 8,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::TEST_8_MI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::TEST_4_RR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::TEST_4_MR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(305419896),
            code: code::TEST_4_RIA_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(305419896),
            code: code::TEST_4_RI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(305419896),
            code: code::TEST_4_MI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-305419896),
            code: code::TEST_4_RIA_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-305419896),
            code: code::TEST_4_RI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 4,
            dest: Dest::Mem,
            src: Src::Imm(-305419896),
            code: code::TEST_4_MI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::TEST_2_RR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::TEST_2_MR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(4660),
            code: code::TEST_2_RIA_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(4660),
            code: code::TEST_2_RI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(4660),
            code: code::TEST_2_MI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-4660),
            code: code::TEST_2_RIA_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-4660),
            code: code::TEST_2_RI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 2,
            dest: Dest::Mem,
            src: Src::Imm(-4660),
            code: code::TEST_2_MI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Reg(RBX),
            code: code::TEST_1_RR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Mem,
            src: Src::Reg(RBX),
            code: code::TEST_1_MR,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(53),
            code: code::TEST_1_RIA_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(53),
            code: code::TEST_1_RI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(53),
            code: code::TEST_1_MI_P,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Reg(RAX),
            src: Src::Imm(-42),
            code: code::TEST_1_RIA_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Reg(RBX),
            src: Src::Imm(-42),
            code: code::TEST_1_RI_N,
        },
        AluSnippet {
            op: ALU_TEST,
            size: 1,
            dest: Dest::Mem,
            src: Src::Imm(-42),
            code: code::TEST_1_MI_N,
        },
    ];

    const ALU_XOR: u8 = 6;
    /// Not an ALU group number: test shares and's flags and never stores.
    const ALU_TEST: u8 = 8;

    #[derive(Clone, Copy, Debug)]
    enum Dest {
        Reg(usize),
        Mem,
    }

    #[derive(Clone, Copy, Debug)]
    enum Src {
        Reg(usize),
        Mem,
        Imm(i64),
    }

    #[derive(Clone, Copy, Debug)]
    struct AluSnippet {
        op: u8,
        size: u8,
        dest: Dest,
        src: Src,
        code: &'static [u8],
    }

    /// Operand values around the sign, carry and nibble boundaries of the operand size.
    fn interesting_values(size: u8) -> Vec<u64> {
        let m = mask(size);
        let sign_bit = 1u64 << (size as u32 * 8 - 1);
        let mut values = vec![
            0,
            1,
            2,
            0xf,
            0x10,
            0x7f & m,
            0x80 & m,
            0xff & m,
            sign_bit - 1,
            sign_bit,
            sign_bit + 1,
            m,
            m - 1,
            0x5555_5555_5555_5555 & m,
            0xaaaa_aaaa_aaaa_aaaa & m,
            0x0123_4567_89ab_cdef & m,
        ];
        values.sort_unstable();
        values.dedup();
        values
    }

    struct Reference {
        result: u64,
        flags: u64,
        /// Flags the SDM defines for the operation (logic ops leave AF undefined).
        defined: u64,
    }

    /// ADD/ADC/SUB/SBB/CMP/AND/OR/XOR computed from the definitions in the SDM with wide integers,
    /// independently of the lazy-flag machinery under test.
    fn alu_reference(op: u8, size: u8, a: u64, b: u64, carry_in: u64) -> Reference {
        let m = mask(size);
        let sign_bit = 1u64 << (size as u32 * 8 - 1);
        let (result, carry, overflow, defined) = match op {
            ALU_ADD | ALU_ADC => {
                let carry_in = if op == ALU_ADC { carry_in } else { 0 };
                let wide = a as u128 + b as u128 + carry_in as u128;
                let r = wide as u64 & m;
                let overflow = (a & sign_bit == b & sign_bit) && (r & sign_bit != a & sign_bit);
                (r, wide > m as u128, overflow, ALL_ARITH)
            }
            ALU_SUB | ALU_SBB | ALU_CMP => {
                let carry_in = if op == ALU_SBB { carry_in } else { 0 };
                let r = a.wrapping_sub(b).wrapping_sub(carry_in) & m;
                let overflow = (a & sign_bit != b & sign_bit) && (r & sign_bit != a & sign_bit);
                (
                    r,
                    (a as u128) < b as u128 + carry_in as u128,
                    overflow,
                    ALL_ARITH,
                )
            }
            ALU_AND | ALU_TEST => (a & b, false, false, ALL_ARITH & !AF),
            ALU_OR => (a | b, false, false, ALL_ARITH & !AF),
            _ => (a ^ b, false, false, ALL_ARITH & !AF),
        };
        let mut flags = zero_sign_parity(result, size);
        if carry {
            flags |= CF;
        }
        if overflow {
            flags |= OF;
        }
        if (a ^ b ^ result) & 0x10 != 0 {
            flags |= AF;
        }
        Reference {
            result,
            flags,
            defined,
        }
    }

    const MEMORY_MARKER: u8 = 0xa5;
    const OPERAND_JUNK: u64 = 0x9e37_79b9_7f4a_7c15;

    fn memory_with_marker(h: &mut Harness, value: u64, size: u8) {
        h.write(DATA, &[MEMORY_MARKER; 16]);
        h.write(DATA, &value.to_le_bytes()[..size as usize]);
        h.cpu.gpr[RDX] = DATA;
    }

    /// Checks one ALU snippet against the reference for every operand pair (and both carry-ins for
    /// the carrying operations).
    fn check_alu_snippet(snippet: &AluSnippet) {
        let AluSnippet {
            op,
            size,
            dest,
            src,
            code,
        } = *snippet;
        let m = mask(size);
        let values = interesting_values(size);
        let sources: Vec<u64> = match src {
            Src::Imm(imm) => vec![imm as u64 & m],
            _ => values.clone(),
        };
        let carries: &[u64] = if op == ALU_ADC || op == ALU_SBB {
            &[0, 1]
        } else {
            &[0]
        };
        let mut pair = Pair::new(code);
        for &a in &values {
            for &b in &sources {
                for &carry in carries {
                    let context = format!(
                        "{op} size {size} dest {dest:?} src {src:?}: {a:#x}, {b:#x}, carry {carry}"
                    );
                    let junk = OPERAND_JUNK & !m;
                    pair.halted(|h| {
                        match dest {
                            Dest::Reg(r) => h.cpu.gpr[r] = junk | a,
                            Dest::Mem => memory_with_marker(h, a, size),
                        }
                        match src {
                            Src::Reg(r) => h.cpu.gpr[r] = junk | b,
                            Src::Mem => memory_with_marker(h, b, size),
                            Src::Imm(_) => {}
                        }
                        if matches!(dest, Dest::Mem) {
                            h.cpu.gpr[RDX] = DATA;
                        }
                        h.cpu.set_eflags_arith(if carry != 0 { CF } else { 0 });
                    });
                    let expected = alu_reference(op, size, a, b, carry);
                    let stores = op != ALU_CMP && op != ALU_TEST;
                    match dest {
                        Dest::Reg(r) => {
                            let old = junk | a;
                            let written = expected.result;
                            let want = if !stores {
                                old
                            } else {
                                match size {
                                    1 | 2 => (old & !m) | written,
                                    _ => written,
                                }
                            };
                            assert_eq!(pair.reg(r), want, "{context}: destination register");
                        }
                        Dest::Mem => {
                            let stored = if stores { expected.result } else { a };
                            let mut want = [MEMORY_MARKER; 16];
                            want[..size as usize]
                                .copy_from_slice(&stored.to_le_bytes()[..size as usize]);
                            assert_eq!(pair.interp.read(DATA, 16), want, "{context}: memory");
                        }
                    }
                    if let Src::Reg(r) = src {
                        assert_eq!(pair.reg(r), junk | b, "{context}: source untouched");
                    }
                    assert_flags(pair.flags(), expected.defined, expected.flags, &context);
                    assert_eq!(pair.cpu().rip, CODE_BASE + code.len() as u64, "{context}");
                }
            }
        }
    }

    fn check_alu_op_and_size(op: u8, size: u8) {
        let mut checked = 0;
        for snippet in ALU_SNIPPETS.iter().filter(|s| s.op == op && s.size == size) {
            check_alu_snippet(snippet);
            checked += 1;
        }
        assert!(checked >= 6, "op {op} size {size}: only {checked} snippets");
    }

    macro_rules! alu_tests {
    ($($name:ident: $op:expr, $size:expr;)*) => {
        $(#[test] fn $name() { check_alu_op_and_size($op, $size); })*
    };
}

    alu_tests! {
        add_64bit_every_form: ALU_ADD, 8;
        add_32bit_every_form: ALU_ADD, 4;
        add_16bit_every_form: ALU_ADD, 2;
        add_8bit_every_form: ALU_ADD, 1;
        or_64bit_every_form: ALU_OR, 8;
        or_32bit_every_form: ALU_OR, 4;
        or_16bit_every_form: ALU_OR, 2;
        or_8bit_every_form: ALU_OR, 1;
        adc_64bit_every_form: ALU_ADC, 8;
        adc_32bit_every_form: ALU_ADC, 4;
        adc_16bit_every_form: ALU_ADC, 2;
        adc_8bit_every_form: ALU_ADC, 1;
        sbb_64bit_every_form: ALU_SBB, 8;
        sbb_32bit_every_form: ALU_SBB, 4;
        sbb_16bit_every_form: ALU_SBB, 2;
        sbb_8bit_every_form: ALU_SBB, 1;
        and_64bit_every_form: ALU_AND, 8;
        and_32bit_every_form: ALU_AND, 4;
        and_16bit_every_form: ALU_AND, 2;
        and_8bit_every_form: ALU_AND, 1;
        sub_64bit_every_form: ALU_SUB, 8;
        sub_32bit_every_form: ALU_SUB, 4;
        sub_16bit_every_form: ALU_SUB, 2;
        sub_8bit_every_form: ALU_SUB, 1;
        xor_64bit_every_form: ALU_XOR, 8;
        xor_32bit_every_form: ALU_XOR, 4;
        xor_16bit_every_form: ALU_XOR, 2;
        xor_8bit_every_form: ALU_XOR, 1;
        cmp_64bit_every_form: ALU_CMP, 8;
        cmp_32bit_every_form: ALU_CMP, 4;
        cmp_16bit_every_form: ALU_CMP, 2;
        cmp_8bit_every_form: ALU_CMP, 1;
        test_64bit_every_form: ALU_TEST, 8;
        test_32bit_every_form: ALU_TEST, 4;
        test_16bit_every_form: ALU_TEST, 2;
        test_8bit_every_form: ALU_TEST, 1;
    }

    #[test]
    fn alu_flag_examples_from_the_manual() {
        // 0x7f + 1 in 8 bits: signed overflow, auxiliary carry, sign.
        let pair = run_regs(code::ADD_1_RR, &[(RAX, 0x7f), (RBX, 1)]);
        assert_eq!(pair.reg(RAX) & 0xff, 0x80);
        assert_flags(pair.flags(), ALL_ARITH, OF | AF | SF, "0x7f + 1");
        // 0xff + 1 in 8 bits: carry out and a zero result.
        let pair = run_regs(code::ADD_1_RR, &[(RAX, 0xff), (RBX, 1)]);
        assert_flags(pair.flags(), ALL_ARITH, CF | AF | ZF | PF, "0xff + 1");
        // 0 - 1 in 64 bits: borrow, sign, no signed overflow.
        let pair = run_regs(code::SUB_8_RR, &[(RAX, 0), (RBX, 1)]);
        assert_eq!(pair.reg(RAX), u64::MAX);
        assert_flags(pair.flags(), ALL_ARITH, CF | AF | SF | PF, "0 - 1");
        // INT_MIN - 1: signed overflow without borrow.
        let pair = run_regs(code::SUB_4_RR, &[(RAX, 0x8000_0000), (RBX, 1)]);
        assert_eq!(pair.reg(RAX), 0x7fff_ffff);
        assert_flags(pair.flags(), ALL_ARITH, OF | AF | PF, "INT_MIN - 1");
    }

    #[test]
    fn adc_and_sbb_use_the_carry_flag_as_the_third_operand() {
        let mut pair = Pair::new(code::ADC_4_RR);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xffff_ffff;
            h.cpu.gpr[RBX] = 0;
            h.cpu.set_eflags_arith(CF);
        });
        assert_eq!(pair.reg(RAX), 0);
        assert_flags(
            pair.flags(),
            ALL_ARITH,
            CF | ZF | AF | PF,
            "0xffffffff + 0 + CF",
        );
        let mut pair = Pair::new(code::SBB_4_RR);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0;
            h.cpu.gpr[RBX] = 0xffff_ffff;
            h.cpu.set_eflags_arith(CF);
        });
        assert_eq!(pair.reg(RAX), 0);
        assert_flags(
            pair.flags(),
            ALL_ARITH,
            CF | ZF | AF | PF,
            "0 - 0xffffffff - CF",
        );
    }

    #[test]
    fn thirty_two_bit_alu_writes_zero_the_upper_half_but_sixteen_and_eight_bit_keep_it() {
        let regs = [
            (RAX, 0xdead_beef_8000_0001u64),
            (RBX, 0x1234_5678_0000_0001),
        ];
        let pair = run_regs(code::ADD_4_RR, &regs);
        assert_eq!(pair.reg(RAX), 0x8000_0002);
        let pair = run_regs(code::ADD_2_RR, &regs);
        assert_eq!(pair.reg(RAX), 0xdead_beef_8000_0002);
        let pair = run_regs(code::ADD_1_RR, &regs);
        assert_eq!(pair.reg(RAX), 0xdead_beef_8000_0002);
        let pair = run_regs(code::ADD_8_RR, &regs);
        assert_eq!(pair.reg(RAX), 0xf0e2_1567_8000_0002);
    }

    #[test]
    fn cmp_writes_nothing_even_for_thirty_two_bit_operands() {
        let pair = run_regs(code::CMP_4_RR, &[(RAX, 0xdead_beef_0000_0005), (RBX, 5)]);
        assert_eq!(
            pair.reg(RAX),
            0xdead_beef_0000_0005,
            "no zero-extension for cmp"
        );
        assert_flags(pair.flags(), ALL_ARITH, ZF | PF, "5 - 5");
    }
    const BIT_SNIPPETS: &[BitSnippet] = &[
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::RegReg,
            code: code::BT_8_RR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::MemReg,
            code: code::BT_8_MR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::RegImm(5),
            code: code::BT_8_RI5,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::RegImm(0xc5),
            code: code::BT_8_RIBIG,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::MemImm(37),
            code: code::BT_8_MI,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 8,
            form: BitForm::MemImm(0xfd),
            code: code::BT_8_MIBIG,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::RegReg,
            code: code::BT_4_RR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::MemReg,
            code: code::BT_4_MR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::RegImm(5),
            code: code::BT_4_RI5,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::RegImm(0xc5),
            code: code::BT_4_RIBIG,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::MemImm(37),
            code: code::BT_4_MI,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 4,
            form: BitForm::MemImm(0xfd),
            code: code::BT_4_MIBIG,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::RegReg,
            code: code::BT_2_RR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::MemReg,
            code: code::BT_2_MR,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::RegImm(5),
            code: code::BT_2_RI5,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::RegImm(0xc5),
            code: code::BT_2_RIBIG,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::MemImm(37),
            code: code::BT_2_MI,
        },
        BitSnippet {
            op: BIT_TEST,
            size: 2,
            form: BitForm::MemImm(0xfd),
            code: code::BT_2_MIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::RegReg,
            code: code::BTS_8_RR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::MemReg,
            code: code::BTS_8_MR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::RegImm(5),
            code: code::BTS_8_RI5,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::RegImm(0xc5),
            code: code::BTS_8_RIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::MemImm(37),
            code: code::BTS_8_MI,
        },
        BitSnippet {
            op: BIT_SET,
            size: 8,
            form: BitForm::MemImm(0xfd),
            code: code::BTS_8_MIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::RegReg,
            code: code::BTS_4_RR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::MemReg,
            code: code::BTS_4_MR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::RegImm(5),
            code: code::BTS_4_RI5,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::RegImm(0xc5),
            code: code::BTS_4_RIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::MemImm(37),
            code: code::BTS_4_MI,
        },
        BitSnippet {
            op: BIT_SET,
            size: 4,
            form: BitForm::MemImm(0xfd),
            code: code::BTS_4_MIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::RegReg,
            code: code::BTS_2_RR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::MemReg,
            code: code::BTS_2_MR,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::RegImm(5),
            code: code::BTS_2_RI5,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::RegImm(0xc5),
            code: code::BTS_2_RIBIG,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::MemImm(37),
            code: code::BTS_2_MI,
        },
        BitSnippet {
            op: BIT_SET,
            size: 2,
            form: BitForm::MemImm(0xfd),
            code: code::BTS_2_MIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::RegReg,
            code: code::BTR_8_RR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::MemReg,
            code: code::BTR_8_MR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::RegImm(5),
            code: code::BTR_8_RI5,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::RegImm(0xc5),
            code: code::BTR_8_RIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::MemImm(37),
            code: code::BTR_8_MI,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 8,
            form: BitForm::MemImm(0xfd),
            code: code::BTR_8_MIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::RegReg,
            code: code::BTR_4_RR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::MemReg,
            code: code::BTR_4_MR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::RegImm(5),
            code: code::BTR_4_RI5,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::RegImm(0xc5),
            code: code::BTR_4_RIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::MemImm(37),
            code: code::BTR_4_MI,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 4,
            form: BitForm::MemImm(0xfd),
            code: code::BTR_4_MIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::RegReg,
            code: code::BTR_2_RR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::MemReg,
            code: code::BTR_2_MR,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::RegImm(5),
            code: code::BTR_2_RI5,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::RegImm(0xc5),
            code: code::BTR_2_RIBIG,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::MemImm(37),
            code: code::BTR_2_MI,
        },
        BitSnippet {
            op: BIT_RESET,
            size: 2,
            form: BitForm::MemImm(0xfd),
            code: code::BTR_2_MIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::RegReg,
            code: code::BTC_8_RR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::MemReg,
            code: code::BTC_8_MR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::RegImm(5),
            code: code::BTC_8_RI5,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::RegImm(0xc5),
            code: code::BTC_8_RIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::MemImm(37),
            code: code::BTC_8_MI,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 8,
            form: BitForm::MemImm(0xfd),
            code: code::BTC_8_MIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::RegReg,
            code: code::BTC_4_RR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::MemReg,
            code: code::BTC_4_MR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::RegImm(5),
            code: code::BTC_4_RI5,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::RegImm(0xc5),
            code: code::BTC_4_RIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::MemImm(37),
            code: code::BTC_4_MI,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 4,
            form: BitForm::MemImm(0xfd),
            code: code::BTC_4_MIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::RegReg,
            code: code::BTC_2_RR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::MemReg,
            code: code::BTC_2_MR,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::RegImm(5),
            code: code::BTC_2_RI5,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::RegImm(0xc5),
            code: code::BTC_2_RIBIG,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::MemImm(37),
            code: code::BTC_2_MI,
        },
        BitSnippet {
            op: BIT_COMPLEMENT,
            size: 2,
            form: BitForm::MemImm(0xfd),
            code: code::BTC_2_MIBIG,
        },
    ];

    const IMUL3_SNIPPETS: &[(u8, i64, &[u8], &[u8])] = &[
        (8, 7, code::IMUL3_8_S, code::IMUL3_8_S_MEM),
        (8, 74565, code::IMUL3_8_B, code::IMUL3_8_B_MEM),
        (8, -3, code::IMUL3_8_N, code::IMUL3_8_N_MEM),
        (4, 7, code::IMUL3_4_S, code::IMUL3_4_S_MEM),
        (4, 74565, code::IMUL3_4_B, code::IMUL3_4_B_MEM),
        (4, -3, code::IMUL3_4_N, code::IMUL3_4_N_MEM),
        (2, 7, code::IMUL3_2_S, code::IMUL3_2_S_MEM),
        (2, 300, code::IMUL3_2_B, code::IMUL3_2_B_MEM),
        (2, -3, code::IMUL3_2_N, code::IMUL3_2_N_MEM),
    ];

    #[derive(Clone, Copy, Debug)]
    enum BitForm {
        RegReg,
        MemReg,
        RegImm(u8),
        MemImm(u8),
    }

    #[derive(Clone, Copy, Debug)]
    struct BitSnippet {
        op: u8,
        size: u8,
        form: BitForm,
        code: &'static [u8],
    }

    const BIT_PATTERN_START: u64 = DATA - 0x80;
    const BIT_PATTERN_LEN: usize = 0x100;

    fn bit_pattern() -> Vec<u8> {
        (0..BIT_PATTERN_LEN)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect()
    }

    /// Applies bt/bts/btr/btc to one bit of a flat little-endian bit string; returns the old bit.
    fn apply_bit_op(bytes: &mut [u8], index: usize, bit: u32, op: u8) -> bool {
        let old = bytes[index] >> bit & 1 != 0;
        match op {
            BIT_SET => bytes[index] |= 1 << bit,
            BIT_RESET => bytes[index] &= !(1 << bit),
            BIT_COMPLEMENT => bytes[index] ^= 1 << bit,
            _ => {}
        }
        old
    }

    fn check_bit_snippet(snippet: &BitSnippet) {
        let BitSnippet {
            op,
            size,
            form,
            code,
        } = *snippet;
        let bits = size as u32 * 8;
        let m = mask(size);
        let mut pair = Pair::new(code);
        // Offsets that cross operand and byte boundaries; signed ones matter for memory operands.
        let register_offsets: Vec<i64> = vec![0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 255];
        let memory_offsets: Vec<i64> = {
            let mut v = vec![0, 1, 5, 8, 15, 16, 31, 32, 63, 64, 65, 100, 255, 500];
            let negatives: Vec<i64> = v.iter().skip(1).map(|x| -x).collect();
            v.extend(negatives);
            if size == 2 {
                v.retain(|x| (-0x8000..0x8000).contains(x));
            }
            v
        };
        match form {
            BitForm::RegReg | BitForm::RegImm(_) => {
                let offsets: Vec<i64> = match form {
                    BitForm::RegImm(imm) => vec![imm as i64],
                    _ => register_offsets
                        .iter()
                        .copied()
                        .chain([0x1_0000_0000 + 3, -1, -64 - 5])
                        .collect(),
                };
                for a in interesting_values(size) {
                    for &offset in &offsets {
                        let junk = OPERAND_JUNK & !m;
                        for flags_in in [0, ZF, ALL_ARITH] {
                            pair.halted(|h| {
                                h.cpu.gpr[RAX] = junk | a;
                                h.cpu.gpr[RBX] = (OPERAND_JUNK << 1 & !m) | (offset as u64 & m);
                                h.cpu.set_eflags_arith(flags_in);
                            });
                            let bit = (offset as u64 & m) % bits as u64;
                            let old = a >> bit & 1;
                            let new = match op {
                                BIT_SET => a | 1 << bit,
                                BIT_RESET => a & !(1 << bit),
                                BIT_COMPLEMENT => a ^ 1 << bit,
                                _ => a,
                            };
                            let context =
                                format!("op {op} size {size} {form:?}: {a:#x} offset {offset}");
                            let want_rax = if op == BIT_TEST {
                                junk | a
                            } else if size >= 4 {
                                new
                            } else {
                                junk | new
                            };
                            assert_eq!(pair.reg(RAX), want_rax, "{context}");
                            let want_flags = (flags_in & ZF) | old;
                            assert_flags(pair.flags(), CF | ZF, want_flags, &context);
                        }
                    }
                }
            }
            BitForm::MemReg | BitForm::MemImm(_) => {
                let offsets: Vec<i64> = match form {
                    BitForm::MemImm(imm) => vec![imm as i64],
                    _ => memory_offsets,
                };
                for &offset in &offsets {
                    let register_form = matches!(form, BitForm::MemReg);
                    pair.halted(|h| {
                        h.write(BIT_PATTERN_START, &bit_pattern());
                        h.cpu.gpr[RDX] = DATA;
                        h.cpu.gpr[RBX] = (OPERAND_JUNK & !m) | (offset as u64 & m);
                        h.cpu.set_eflags_arith(ZF);
                    });
                    let mut expected = bit_pattern();
                    let (flat_bit, context) = if register_form {
                        (offset, format!("op {op} size {size} offset {offset}"))
                    } else {
                        // An immediate offset is taken modulo the operand width and never moves
                        // the address.
                        (
                            offset.rem_euclid(bits as i64),
                            format!("op {op} size {size} imm {offset}"),
                        )
                    };
                    let byte = (DATA - BIT_PATTERN_START) as i64 + flat_bit.div_euclid(8);
                    let old = apply_bit_op(
                        &mut expected,
                        byte as usize,
                        flat_bit.rem_euclid(8) as u32,
                        op,
                    );
                    assert_eq!(
                        pair.interp.read(BIT_PATTERN_START, BIT_PATTERN_LEN),
                        expected,
                        "{context}: memory"
                    );
                    assert_flags(pair.flags(), CF | ZF, ZF | old as u64, &context);
                    assert_eq!(pair.reg(RDX), DATA);
                }
            }
        }
    }

    #[test]
    fn bt_bts_btr_btc_every_size_form_and_offset() {
        for snippet in BIT_SNIPPETS {
            check_bit_snippet(snippet);
        }
        assert_eq!(BIT_SNIPPETS.len(), 4 * 3 * 6);
    }

    #[test]
    fn bt_with_a_register_offset_ignores_the_rest_of_the_offset_register() {
        // bt eax, ebx uses ebx modulo 32, whatever the upper half holds.
        let pair = run_regs(
            code::BT_4_RR,
            &[(RAX, 1 << 3), (RBX, 0xffff_ffff_0000_0003)],
        );
        assert_flags(pair.flags(), CF, CF, "bit 3 of eax");
        let pair = run_regs(
            code::BT_4_RR,
            &[(RAX, 1 << 3), (RBX, 0xffff_ffff_0000_0023)],
        );
        assert_flags(pair.flags(), CF, CF, "35 mod 32 = 3");
    }

    #[test]
    fn bt_with_a_memory_operand_can_reach_bytes_far_from_the_base() {
        // Offset 1000 bits is byte 125 of the string, bit 0.
        let mut pair = Pair::new(code::BTS_4_MR);
        pair.halted(|h| {
            h.write(DATA, &[0u8; 256]);
            h.cpu.gpr[RDX] = DATA;
            h.cpu.gpr[RBX] = 1000;
        });
        assert_eq!(pair.interp.read(DATA + 125, 1), [1]);
        assert_flags(pair.flags(), CF, 0, "the bit was clear");
        // A negative offset reaches below the base.
        let mut pair = Pair::new(code::BTS_8_MR);
        pair.halted(|h| {
            h.write(DATA - 16, &[0u8; 16]);
            h.cpu.gpr[RDX] = DATA;
            h.cpu.gpr[RBX] = (-1i64) as u64;
        });
        assert_eq!(
            pair.interp.read(DATA - 1, 1),
            [0x80],
            "bit -1 is bit 7 of the previous byte"
        );
    }

    #[test]
    fn group_8_encodings_below_bt_are_undefined() {
        // 0f ba /0 is undefined.
        let (_, stop) = run_regs_stop(code::BT_GROUP8_REG0, &[]);
        assert_eq!(stop, ud());
        let (_, stop) = run_regs_stop(code::BT_GROUP8_MEM3, &[(RDX, DATA)]);
        assert_eq!(stop, ud(), "0f ba /3 is undefined");
        let (_, stop) = run_regs_stop(code::BT_GROUP8_MEM7, &[(RDX, DATA)]);
        assert_eq!(stop, Stop::Halted, "0f ba /7 is btc");
    }

    fn bit_scan_reference(forward: bool, value: u64) -> Option<u64> {
        if value == 0 {
            None
        } else if forward {
            Some(value.trailing_zeros() as u64)
        } else {
            Some(63 - value.leading_zeros() as u64)
        }
    }

    #[test]
    fn bsf_and_bsr_find_the_lowest_and_highest_set_bit() {
        let snippets: [(bool, u8, &[u8], &[u8]); 6] = [
            (true, 8, code::BSF_8, code::BSF_MEM_8),
            (true, 4, code::BSF_4, code::BSF_MEM_4),
            (true, 2, code::BSF_2, code::BSF_MEM_2),
            (false, 8, code::BSR_8, code::BSR_MEM_8),
            (false, 4, code::BSR_4, code::BSR_MEM_4),
            (false, 2, code::BSR_2, code::BSR_MEM_2),
        ];
        for (forward, size, register_code, memory_code) in snippets {
            let m = mask(size);
            let bits = size as u32 * 8;
            let mut values: Vec<u64> = (0..bits).map(|b| 1u64 << b).collect();
            values.extend([
                0,
                m,
                m >> 1,
                0x5a5a_5a5a_5a5a_5a5a & m,
                0x8000_0000_0000_0001 & m,
            ]);
            for memory in [false, true] {
                let mut pair = Pair::new(if memory { memory_code } else { register_code });
                for &value in &values {
                    let old_rax = 0xfedc_ba98_7654_3210u64;
                    pair.halted(|h| {
                        h.cpu.gpr[RAX] = old_rax;
                        if memory {
                            memory_with_marker(h, value, size);
                        } else {
                            h.cpu.gpr[RBX] = (OPERAND_JUNK & !m) | value;
                        }
                        h.cpu.set_eflags_arith(CF | SF | OF);
                    });
                    let context =
                        format!("forward {forward} size {size} memory {memory} value {value:#x}");
                    match bit_scan_reference(forward, value) {
                        None => {
                            assert_flags(pair.flags(), ZF, ZF, &context);
                            let unchanged = if size == 8 { old_rax } else { old_rax & m };
                            assert_eq!(
                                pair.reg(RAX) & m,
                                unchanged & m,
                                "{context}: destination kept"
                            );
                        }
                        Some(index) => {
                            assert_flags(pair.flags(), ZF, 0, &context);
                            let want = match size {
                                8 | 4 => index,
                                _ => (old_rax & !m) | index,
                            };
                            assert_eq!(pair.reg(RAX), want, "{context}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tzcnt_and_lzcnt_count_zeros_around_the_set_bit() {
        // (More tzcnt cases are in bmi.rs; LZCNT and POPCNT are tested in sse_ext.rs too.)
        let pair = run_regs(code::TZCNT_64, &[(RBX, 0x100)]);
        assert_eq!(pair.reg(RAX), 8);
        let pair = run_regs(code::LZCNT_64, &[(RBX, 0x100)]);
        assert_eq!(pair.reg(RAX), 55, "lzcnt counts leading zeros");
    }

    #[test]
    fn popcnt_counts_set_bits() {
        let pair = run_regs(code::POPCNT_64, &[(RBX, 0xff)]);
        assert_eq!(pair.reg(RAX), 8);
    }

    /// Signed product truncated to `size`, with CF = OF set when the full product does not fit.
    fn imul_reference(size: u8, a: u64, b: u64) -> (u64, bool) {
        let a = sign_extend_test(a & mask(size), size);
        let b = sign_extend_test(b & mask(size), size);
        let product = a * b;
        let bits = size as u32 * 8;
        let truncated = product as u128 as u64 & mask(size);
        let fits = sign_extend_test(truncated, size) == product;
        let _ = bits;
        (truncated, !fits)
    }

    #[test]
    fn imul_two_operand_forms_truncate_and_flag_overflow() {
        let snippets: [(u8, &[u8], &[u8]); 3] = [
            (8, code::IMUL2_8_RR, code::IMUL2_8_RM),
            (4, code::IMUL2_4_RR, code::IMUL2_4_RM),
            (2, code::IMUL2_2_RR, code::IMUL2_2_RM),
        ];
        for (size, register_code, memory_code) in snippets {
            let m = mask(size);
            let junk = OPERAND_JUNK & !m;
            for memory in [false, true] {
                let mut pair = Pair::new(if memory { memory_code } else { register_code });
                for a in interesting_values(size) {
                    for b in interesting_values(size) {
                        pair.halted(|h| {
                            h.cpu.gpr[RAX] = junk | a;
                            if memory {
                                memory_with_marker(h, b, size);
                            } else {
                                h.cpu.gpr[RBX] = junk | b;
                            }
                        });
                        let (result, overflow) = imul_reference(size, a, b);
                        let context = format!("size {size} {a:#x} * {b:#x}");
                        let want = if size >= 4 { result } else { junk | result };
                        assert_eq!(pair.reg(RAX), want, "{context}");
                        assert_flags(
                            pair.flags(),
                            CF | OF,
                            if overflow { CF | OF } else { 0 },
                            &context,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn imul_three_operand_forms_multiply_by_the_immediate() {
        for &(size, imm, register_code, memory_code) in IMUL3_SNIPPETS {
            let m = mask(size);
            let junk = OPERAND_JUNK & !m;
            for memory in [false, true] {
                let mut pair = Pair::new(if memory { memory_code } else { register_code });
                for a in interesting_values(size) {
                    pair.halted(|h| {
                        h.cpu.gpr[RAX] = u64::MAX;
                        if memory {
                            memory_with_marker(h, a, size);
                        } else {
                            h.cpu.gpr[RBX] = junk | a;
                        }
                    });
                    let (result, overflow) = imul_reference(size, a, imm as u64);
                    let context = format!("size {size} {a:#x} * {imm}");
                    let want = if size >= 4 { result } else { !m | result };
                    assert_eq!(pair.reg(RAX), want, "{context}");
                    assert_flags(
                        pair.flags(),
                        CF | OF,
                        if overflow { CF | OF } else { 0 },
                        &context,
                    );
                }
            }
        }
    }

    #[test]
    fn imul_examples_by_hand() {
        // 0x4000_0000 * 2 overflows a signed 32-bit result.
        let pair = run_regs(code::IMUL2_4_RR, &[(RAX, 0x4000_0000), (RBX, 2)]);
        assert_eq!(pair.reg(RAX), 0x8000_0000);
        assert_flags(pair.flags(), CF | OF, CF | OF, "0x40000000 * 2");
        // -1 * -1 = 1 fits.
        let pair = run_regs(code::IMUL2_8_RR, &[(RAX, u64::MAX), (RBX, u64::MAX)]);
        assert_eq!(pair.reg(RAX), 1);
        assert_flags(pair.flags(), CF | OF, 0, "-1 * -1");
        // INT64_MIN * -1 overflows.
        let pair = run_regs(code::IMUL2_8_RR, &[(RAX, 1 << 63), (RBX, u64::MAX)]);
        assert_eq!(pair.reg(RAX), 1 << 63);
        assert_flags(pair.flags(), CF | OF, CF | OF, "INT64_MIN * -1");
    }

    fn put_words(h: &mut Harness, address: u64, words: &[u64]) {
        for (index, word) in words.iter().enumerate() {
            h.write_u64(address.wrapping_add(index as u64 * 8), *word);
        }
    }

    /// Everything an instruction must leave alone when it faults.
    #[derive(Debug, PartialEq, Eq)]
    struct Architectural {
        gpr: [u64; 16],
        rip: u64,
        eflags: u64,
        cpl: u8,
        segments: [(u16, u64, u32, u32); 6],
    }

    fn capture(cpu: &Cpu) -> Architectural {
        Architectural {
            gpr: cpu.gpr,
            rip: cpu.rip,
            eflags: cpu.eflags_full(),
            cpl: cpu.cpl,
            segments: std::array::from_fn(|i| {
                let s = cpu.seg[i];
                (s.sel, s.base, s.limit, s.flags)
            }),
        }
    }

    /// Runs the snippet from state prepared by `setup`, expects `expected` to be raised, and checks
    /// that the faulting instruction changed no register, segment, CPL or flag.
    #[track_caller]
    fn assert_fault_changes_nothing(code: &[u8], setup: impl Fn(&mut Harness), expected: Stop) {
        let mut pair = Pair::new(code);
        let before = std::cell::RefCell::new(None);
        let stop = pair.run(|h| {
            setup(h);
            *before.borrow_mut() = Some(capture(&h.cpu));
        });
        assert_eq!(stop, expected, "{}", pair.cpu().dump_regs());
        assert_eq!(
            capture(pair.cpu()),
            before.into_inner().unwrap(),
            "a faulting instruction must leave the architectural state untouched"
        );
    }

    /// (rdx, rax) holding `dividend` as a double-width value; byte divides use AX alone.
    fn split_dividend(size: u8, dividend: i128) -> (u64, u64) {
        let bits = size as u32 * 8;
        let value = dividend as u128;
        match size {
            1 => (0, value as u64 & 0xffff),
            _ => (
                (value >> bits) as u64 & mask(size),
                value as u64 & mask(size),
            ),
        }
    }

    const JUNK_HIGH: u64 = 0xdead_beef_cafe_0000;

    /// Runs one divide and checks it against the architecture: truncating division, remainder with
    /// the dividend's sign, #DE for a zero divisor or a quotient outside the destination.
    #[track_caller]
    fn check_divide(code: &[u8], size: u8, signed: bool, dividend: i128, divisor: i128) {
        let bits = size as u32 * 8;
        let (rdx_in, rax_in) = split_dividend(size, dividend);
        // Bits above the operand are ignored on input, and preserved by 8/16-bit results.
        let rax_junk = if size < 4 { JUNK_HIGH } else { 0 };
        let rdx_junk = if size == 2 { JUNK_HIGH } else { 0 };
        let (pair, stop) = run_regs_stop(
            code,
            &[
                (RAX, rax_in | rax_junk),
                (RDX, rdx_in | rdx_junk),
                (RCX, divisor as u64 & mask(size)),
            ],
        );
        let quotient_and_remainder = if divisor == 0 {
            None
        } else if signed {
            let half = 1i128 << (bits - 1);
            dividend
                .checked_div(divisor)
                .filter(|q| (-half..half).contains(q))
                .map(|q| (q, dividend - q * divisor))
        } else {
            let (n, d) = (dividend as u128, divisor as u128);
            let q = n / d;
            (q <= mask(size) as u128).then(|| (q as i128, (n % d) as i128))
        };
        let context = format!("size {size} signed {signed} {dividend} / {divisor}");
        match quotient_and_remainder {
            None => {
                assert_eq!(stop, de(), "{context}");
                assert_eq!(pair.reg(RAX), rax_in | rax_junk, "{context}: rax kept");
                assert_eq!(pair.reg(RDX), rdx_in | rdx_junk, "{context}: rdx kept");
                assert_eq!(pair.cpu().rip, CODE_BASE, "{context}: rip at the divide");
            }
            Some((q, r)) => {
                assert_eq!(stop, Stop::Halted, "{context}");
                let (q, r) = (q as u64 & mask(size), r as u64 & mask(size));
                match size {
                    1 => {
                        assert_eq!(pair.reg(RAX), (rax_in | rax_junk) & !0xffff | r << 8 | q);
                        assert_eq!(pair.reg(RDX), 0, "{context}");
                    }
                    2 => {
                        assert_eq!(
                            pair.reg(RAX),
                            (rax_in | rax_junk) & !0xffff | q,
                            "{context}"
                        );
                        assert_eq!(
                            pair.reg(RDX),
                            (rdx_in | rdx_junk) & !0xffff | r,
                            "{context}"
                        );
                    }
                    _ => {
                        assert_eq!(pair.reg(RAX), q, "{context}");
                        assert_eq!(pair.reg(RDX), r, "{context}");
                    }
                }
            }
        }
    }

    const IDIV_BY_SIZE: [(u8, &[u8]); 4] = [
        (1, code::IDIV_CL),
        (2, code::IDIV_CX),
        (4, code::IDIV_ECX),
        (8, code::IDIV_RCX),
    ];
    const DIV_BY_SIZE: [(u8, &[u8]); 4] = [
        (1, code::DIV_CL),
        (2, code::DIV_CX),
        (4, code::DIV_ECX),
        (8, code::DIV_RCX),
    ];

    #[test]
    fn idiv64_minimum_dividend_by_minus_one_raises_divide_error() {
        // rdx:rax = -2^127, divisor -1: the quotient 2^127 does not fit, and the host division of
        // i128::MIN by -1 would itself panic.
        let (pair, stop) = run_regs_stop(
            code::IDIV_RCX,
            &[(RDX, 0x8000_0000_0000_0000), (RAX, 0), (RCX, u64::MAX)],
        );
        assert_eq!(stop, de());
        assert_eq!(pair.reg(RDX), 0x8000_0000_0000_0000);
        assert_eq!(pair.reg(RAX), 0);
        assert_eq!(pair.cpu().rip, CODE_BASE);
    }

    #[test]
    fn idiv_signed_cases_every_size() {
        for (size, code) in IDIV_BY_SIZE {
            let half = 1i128 << (size as u32 * 8 - 1);
            // For size 8 this is 2^127 - 1: the shift lands on i128::MIN and the subtraction must wrap.
            let wide_max = (half << (size as u32 * 8)).wrapping_sub(1);
            let mut cases = vec![
                (7, 2),
                (-7, 2),
                (7, -2),
                (-7, -2),
                (0, 5),
                (0, -5),
                (half - 1, 1),
                (half, 1),
                (-half, 1),
                (-half - 1, 1),
                (-half, -1),
                (half - 1, -1),
                (-half + 1, -1),
                ((half - 1) * 7 + 3, 7),
                (half * 7, 7),
                (-(half - 1) * 7 - 3, 7),
                (-half * 7, 7),
                (-half * 7 - 1, 7),
                (wide_max, 1),
                (-wide_max - 1, 1),
                (wide_max, -1),
                (-wide_max - 1, -1),
                (-wide_max - 1, 2),
                (wide_max, half - 1),
                (5, 0),
                (-5, 0),
                (0, 0),
            ];
            // Largest and smallest 2-wide dividends whose quotient still fits.
            cases.push(((half - 1) * (half - 1), half - 1));
            cases.push((-half * (half - 1), half - 1));
            for (dividend, divisor) in cases {
                check_divide(code, size, true, dividend, divisor);
            }
        }
    }

    #[test]
    fn div_unsigned_cases_every_size() {
        for (size, code) in DIV_BY_SIZE {
            let bits = size as u32 * 8;
            let max = mask(size) as i128;
            let wide_max = (1i128 << (bits - 1)) - 1; // the model works in i128, so stay below 2^127
            let wide_max = if size == 8 {
                i128::MAX
            } else {
                (wide_max << 1) | 1
            };
            let cases = [
                (0, 1),
                (1, 1),
                (max, 1),
                (max + 1, 1),
                (wide_max, 1),
                (max * 7 + 6, 7),
                (max * 7 + 7, 7),
                (wide_max, max),
                // At size 8 these dividends pass 2^127; check_divide reads the i128 bits as u128.
                (max.wrapping_mul(max), max),
                (max.wrapping_mul(max).wrapping_add(max - 1), max),
                (max.wrapping_mul(max).wrapping_add(max), max),
                (5, 0),
                (0, 0),
                (wide_max, 0),
                (100, 7),
                (100, max),
            ];
            for (dividend, divisor) in cases {
                check_divide(code, size, false, dividend, divisor);
            }
        }
    }

    #[test]
    fn div64_with_a_dividend_above_two_to_the_127() {
        // The i128 model cannot hold these, so check by hand: (2^128 - 1) / 2^63 = 2^65 - 1 overflows;
        // 2^127 / 2^63 = 2^64 overflows; (2^127 - 1) / 2^63 = 2^64 - 1 - 1 + ... fits.
        let (pair, stop) = run_regs_stop(
            code::DIV_RCX,
            &[(RDX, u64::MAX), (RAX, u64::MAX), (RCX, 1 << 63)],
        );
        assert_eq!(stop, de());
        assert_eq!(pair.reg(RDX), u64::MAX);
        let pair = run_regs(
            code::DIV_RCX,
            &[(RDX, (1 << 63) - 1), (RAX, u64::MAX), (RCX, 1 << 63)],
        );
        // (2^127 - 1) = (2^64 - 1) * 2^63 + (2^63 - 1)
        assert_eq!(pair.reg(RAX), u64::MAX);
        assert_eq!(pair.reg(RDX), (1 << 63) - 1);
    }

    const MUL_BY_SIZE: [(u8, &[u8]); 4] = [
        (1, code::MUL_CL),
        (2, code::MUL_CX),
        (4, code::MUL_ECX),
        (8, code::MUL_RCX),
    ];
    const IMUL1_BY_SIZE: [(u8, &[u8]); 4] = [
        (1, code::IMUL1_CL),
        (2, code::IMUL1_CX),
        (4, code::IMUL1_ECX),
        (8, code::IMUL1_RCX),
    ];

    fn sign_extend_test(value: u64, size: u8) -> i128 {
        let bits = size as u32 * 8;
        if bits == 64 {
            value as i64 as i128
        } else if value >> (bits - 1) & 1 == 1 {
            value as i128 - (1i128 << bits)
        } else {
            value as i128
        }
    }

    #[track_caller]
    fn check_multiply(code: &[u8], size: u8, signed: bool, a: u64, b: u64) {
        let bits = size as u32 * 8;
        let (a, b) = (a & mask(size), b & mask(size));
        let rax_junk = if size < 4 { JUNK_HIGH } else { 0 };
        let rdx_junk = if size == 2 { JUNK_HIGH } else { 0 };
        let rdx_in = if size == 1 { 0 } else { 0x1234 & mask(size) } | rdx_junk;
        let pair = run_regs(code, &[(RAX, a | rax_junk), (RCX, b), (RDX, rdx_in)]);
        let product: i128 = if signed {
            sign_extend_test(a, size) * sign_extend_test(b, size)
        } else {
            (a as u128 * b as u128) as i128
        };
        let bits_mask = (1u128 << bits) - 1;
        let low = product as u128 & bits_mask;
        let high = if bits == 64 {
            (product as u128 >> 64) as u64 as u128
        } else {
            (product as u128 >> bits) & bits_mask
        };
        let fits = if signed {
            let half = 1i128 << (bits - 1);
            (-half..half).contains(&product)
        } else {
            high == 0
        };
        let context = format!("size {size} signed {signed} {a:#x} * {b:#x}");
        match size {
            1 => assert_eq!(
                pair.reg(RAX),
                (a | rax_junk) & !0xffff | (high as u64) << 8 | low as u64,
                "{context}"
            ),
            2 => {
                assert_eq!(
                    pair.reg(RAX),
                    (a | rax_junk) & !0xffff | low as u64,
                    "{context}"
                );
                assert_eq!(pair.reg(RDX), rdx_in & !0xffff | high as u64, "{context}");
            }
            _ => {
                assert_eq!(pair.reg(RAX), low as u64, "{context}");
                assert_eq!(pair.reg(RDX), high as u64, "{context}");
            }
        }
        let expected = if fits { 0 } else { CF | OF };
        assert_flags(pair.flags(), CF | OF, expected, &context);
    }

    #[test]
    fn mul_and_imul_every_size_set_carry_and_overflow_when_the_high_half_matters() {
        for size in [1u8, 2, 4, 8] {
            let max = mask(size);
            let half = 1u64 << (size as u32 * 8 - 1);
            let values = [
                0,
                1,
                2,
                3,
                half - 1,
                half,
                half + 1,
                max,
                max - 1,
                0x1234_5678_9abc_def0 & max,
            ];
            let (_, mul_code) = MUL_BY_SIZE.iter().find(|(s, _)| *s == size).unwrap();
            let (_, imul_code) = IMUL1_BY_SIZE.iter().find(|(s, _)| *s == size).unwrap();
            for a in values {
                for b in values {
                    check_multiply(mul_code, size, false, a, b);
                    check_multiply(imul_code, size, true, a, b);
                }
            }
        }
    }

    /// Intel's result for a 16-bit double shift: the operand is the 48-bit value A:B:A, so counts of
    /// 17..=31 keep shifting in B and then A again (QEMU `gen_shiftd_rm_T1` documents the same).
    fn double_shift16_model(left: bool, a: u64, b: u64, count: u32) -> (u64, bool) {
        let wide = (a << 32) | (b << 16) | a;
        if left {
            (
                ((wide << count) >> 32) & 0xffff,
                (wide << (count - 1)) >> 47 & 1 == 1,
            )
        } else {
            ((wide >> count) & 0xffff, (wide >> (count - 1)) & 1 == 1)
        }
    }

    #[test]
    fn shld16_count_17_takes_bits_from_the_source_operand() {
        // shld ax, bx, 17: the sources are B then A; the result is B << 1 | A >> 15.
        let pair = run_regs(
            code::SHLD_AX_BX_CL,
            &[(RAX, 0x1234), (RBX, 0xabcd), (RCX, 17)],
        );
        assert_eq!(pair.reg(RAX), 0x579a);
        assert_flags(pair.flags(), CF | ZF | SF | PF, CF | PF, "shld ax,bx,17");
    }

    #[test]
    fn shrd16_count_17_takes_bits_from_the_source_operand() {
        // shrd ax, bx, 17: the result is B >> 1 | A << 15.
        let pair = run_regs(
            code::SHRD_AX_BX_CL,
            &[(RAX, 0x1234), (RBX, 0xabcd), (RCX, 17)],
        );
        assert_eq!(pair.reg(RAX), 0x55e6);
        assert_flags(pair.flags(), CF | ZF | SF | PF, CF, "shrd ax,bx,17");
    }

    #[test]
    fn double_shift16_every_count_matches_the_a_b_a_model() {
        for left in [true, false] {
            let code = if left {
                code::SHLD_AX_BX_CL
            } else {
                code::SHRD_AX_BX_CL
            };
            let mut pair = Pair::new(code);
            for (a, b) in [
                (0x1234u64, 0xabcdu64),
                (0xffff, 0),
                (0, 0xffff),
                (0x8001, 0x8001),
            ] {
                for count in 1..=31u32 {
                    pair.halted(|h| {
                        h.cpu.gpr[RAX] = 0xffff_0000 | a;
                        h.cpu.gpr[RBX] = b;
                        h.cpu.gpr[RCX] = count as u64;
                    });
                    let (result, carry) = double_shift16_model(left, a, b, count);
                    let context = format!("left {left} a {a:#x} b {b:#x} count {count}");
                    assert_eq!(pair.reg(RAX), 0xffff_0000 | result, "{context}");
                    let mut expected = zero_sign_parity(result, 2);
                    if carry {
                        expected |= CF;
                    }
                    // OF is only defined for count 1; AF never.
                    assert_flags(pair.flags(), CF | ZF | SF | PF, expected, &context);
                }
            }
        }
    }

    #[test]
    fn shld16_with_an_immediate_count_above_16_does_not_underflow() {
        let pair = run_regs(code::SHLD_AX_BX_20, &[(RAX, 0x1234), (RBX, 0xabcd)]);
        let (expected, _) = double_shift16_model(true, 0x1234, 0xabcd, 20);
        assert_eq!(pair.reg(RAX), expected);
        let pair = run_regs(code::SHRD_AX_BX_20, &[(RAX, 0x1234), (RBX, 0xabcd)]);
        let (expected, _) = double_shift16_model(false, 0x1234, 0xabcd, 20);
        assert_eq!(pair.reg(RAX), expected);
    }

    #[test]
    fn shld16_count_16_moves_the_source_into_the_destination() {
        let pair = run_regs(
            code::SHLD_AX_BX_CL,
            &[(RAX, 0x1235), (RBX, 0xabcd), (RCX, 16)],
        );
        assert_eq!(pair.reg(RAX), 0xabcd);
        // The last bit shifted out is bit 0 of the destination.
        assert_flags(pair.flags(), CF, CF, "shld ax,bx,16");
        let pair = run_regs(
            code::SHRD_AX_BX_CL,
            &[(RAX, 0x1235), (RBX, 0xabcd), (RCX, 16)],
        );
        assert_eq!(pair.reg(RAX), 0xabcd);
        // Bit 15 of the destination is the last one out.
        assert_flags(pair.flags(), CF, NO_FLAGS, "shrd ax,bx,16");
    }

    #[test]
    fn double_shift_count_is_masked_and_zero_changes_nothing() {
        for (code, size) in [
            (code::SHLD_EAX_EBX_CL, 4u8),
            (code::SHRD_EAX_EBX_CL, 4),
            (code::SHLD_RAX_RBX_CL, 8),
            (code::SHRD_RAX_RBX_CL, 8),
            (code::SHLD_AX_BX_CL, 2),
            (code::SHRD_AX_BX_CL, 2),
        ] {
            let modulus = if size == 8 { 64 } else { 32 };
            let flags_before = |h: &mut Harness| {
                h.cpu.set_eflags_arith(CF | ZF | SF | PF | AF | OF);
            };
            let mut pair = Pair::new(code);
            pair.halted(|h| {
                h.cpu.gpr[RAX] = 0x1111_2222_3333_4444;
                h.cpu.gpr[RBX] = 0x5555_6666_7777_8888;
                h.cpu.gpr[RCX] = modulus; // masks to zero
                flags_before(h);
            });
            assert_eq!(pair.reg(RAX), 0x1111_2222_3333_4444, "size {size} count 0");
            assert_flags(
                pair.flags(),
                ALL_ARITH,
                CF | ZF | SF | PF | AF | OF,
                "count 0",
            );
        }
    }

    #[test]
    fn shld32_and_shrd64_follow_the_two_operand_concatenation() {
        let pair = run_regs(
            code::SHLD_EAX_EBX_CL,
            &[(RAX, 0x1000_0001), (RBX, 0xf000_0000), (RCX, 4)],
        );
        assert_eq!(pair.reg(RAX), 0x0000_001f);
        assert_flags(pair.flags(), CF | SF | ZF, CF, "shld eax,ebx,4");
        let pair = run_regs(
            code::SHRD_RAX_RBX_CL,
            &[(RAX, 0x0000_0000_0000_0019), (RBX, 0xf), (RCX, 4)],
        );
        assert_eq!(pair.reg(RAX), 0xf000_0000_0000_0001);
        assert_flags(pair.flags(), CF | SF | ZF, CF | SF, "shrd rax,rbx,4");
    }

    #[test]
    fn mov_from_sreg_ignores_rex_r() {
        let ds = 0x1234u64;
        let pair = {
            let mut pair = Pair::new(code::MOV_EAX_DS_REX_R);
            pair.halted(|h| {
                h.cpu.seg[SEG_DS].sel = ds as u16;
                h.cpu.gpr[RAX] = u64::MAX;
            });
            pair
        };
        assert_eq!(pair.reg(RAX), ds);
    }

    #[test]
    fn mov_from_sreg_reads_each_of_the_six_registers() {
        for (index, code) in [
            (SEG_ES, code::MOV_EAX_ES),
            (SEG_CS, code::MOV_EAX_CS),
            (SEG_SS, code::MOV_EAX_SS),
            (SEG_DS, code::MOV_EAX_DS),
            (SEG_FS, code::MOV_EAX_FS),
            (SEG_GS, code::MOV_EAX_GS),
        ] {
            let mut pair = Pair::new(code);
            pair.halted(|h| {
                h.cpu.seg[index].sel = 0xabc0 + index as u16;
                h.cpu.gpr[RAX] = u64::MAX;
            });
            assert_eq!(pair.reg(RAX), 0xabc0 + index as u64, "sreg {index}");
        }
    }

    #[test]
    fn mov_from_sreg_to_register_takes_the_operand_size_and_memory_is_16_bits() {
        let setup = |h: &mut Harness| {
            h.cpu.seg[SEG_DS].sel = 0x2b;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RBX] = DATA;
            h.write_u64(DATA, u64::MAX);
        };
        // 32-bit destination zero-extends, 64-bit zero-extends, 16-bit (66 prefix) merges.
        let mut pair = Pair::new(code::MOV_EAX_DS);
        pair.halted(setup);
        assert_eq!(pair.reg(RAX), 0x2b);
        let mut pair = Pair::new(code::MOV_RAX_DS);
        pair.halted(setup);
        assert_eq!(pair.reg(RAX), 0x2b);
        let mut pair = Pair::new(code::MOV_AX_DS);
        pair.halted(setup);
        assert_eq!(pair.reg(RAX), 0xffff_ffff_ffff_002b);
        let mut pair = Pair::new(code::MOV_MEM_DS);
        pair.halted(setup);
        assert_eq!(pair.interp.read_u64(DATA), 0xffff_ffff_ffff_002b);
    }

    #[test]
    fn mov_from_sreg_6_and_7_are_undefined() {
        for code in [
            code::MOV_EAX_SREG6,
            code::MOV_EAX_SREG7,
            code::MOV_EAX_SREG6_REX_R,
        ] {
            let (_, stop) = run_regs_stop(code, &[]);
            assert_eq!(stop, ud(), "{code:02x?}");
        }
    }

    #[test]
    fn mov_to_sreg_6_and_7_and_to_cs_are_undefined() {
        for code in [code::MOV_SREG6_AX, code::MOV_CS_AX] {
            let (pair, stop) = run_regs_stop(code, &[]);
            assert_eq!(stop, ud(), "{code:02x?}");
            assert_eq!(pair.cpu().rip, CODE_BASE);
        }
    }

    #[test]
    fn mov_to_sreg_ignores_rex_r_like_mov_from_sreg() {
        let mut pair = Pair::new(code::MOV_DS_AX_REX_R);
        pair.halted(|h| {
            install_gdt(h);
            h.cpu.gpr[RAX] = SEL_KDATA as u64;
        });
        assert_eq!(pair.cpu().seg[SEG_DS].sel, SEL_KDATA);
    }

    #[test]
    fn xlat_reads_the_byte_at_rbx_plus_al() {
        let mut pair = Pair::new(code::XLAT);
        pair.halted(|h| {
            h.write(DATA, &[10, 11, 12, 13, 14]);
            h.cpu.gpr[RBX] = DATA;
            h.cpu.gpr[RAX] = 0xffff_ff03;
        });
        assert_eq!(pair.reg(RAX), 0xffff_ff0d);
    }

    #[test]
    fn xlat_with_an_address_size_prefix_wraps_the_sum_at_32_bits() {
        let mut pair = Pair::new(code::XLAT_ADDR32);
        pair.halted(|h| {
            h.write(0x0, &[0xaa, 0xbb, 0xcc]);
            h.cpu.gpr[RBX] = 0xffff_ffff;
            h.cpu.gpr[RAX] = 0x1_0000_0002;
        });
        // ebx + al = 0xffff_ffff + 2 wraps to 1, not 0x1_0000_0001.
        assert_eq!(pair.reg(RAX) & 0xff, 0xbb);
    }

    #[test]
    fn xlat_with_an_address_size_prefix_ignores_the_upper_half_of_rbx() {
        let mut pair = Pair::new(code::XLAT_ADDR32);
        pair.halted(|h| {
            h.write(DATA + 5, &[0x77]);
            h.cpu.gpr[RBX] = (7 << 32) | DATA;
            h.cpu.gpr[RAX] = 5;
        });
        assert_eq!(pair.reg(RAX), 0x77);
    }

    #[test]
    fn xlat_without_the_prefix_uses_the_full_64_bit_sum() {
        let (_, stop) = run_regs_stop(code::XLAT, &[(RBX, UNMAPPED), (RAX, 0)]);
        assert_eq!(
            stop,
            fault(VECTOR_PAGE_FAULT, Some(0)),
            "rbx = 4 GiB is outside the identity map"
        );
    }

    #[test]
    fn xlat_applies_the_fs_segment_base() {
        let mut pair = Pair::new(code::XLAT_FS);
        pair.halted(|h| {
            h.write(DATA + 0x20 + 3, &[0x42]);
            h.cpu.seg[SEG_FS].base = DATA + 0x20;
            h.cpu.gpr[RBX] = 0;
            h.cpu.gpr[RAX] = 3;
        });
        assert_eq!(pair.reg(RAX), 0x42);
    }

    const FRAME_TOP: u64 = DATA + 0x800;

    fn enter_setup(h: &mut Harness) {
        h.cpu.gpr[RSP] = FRAME_TOP;
        h.cpu.gpr[RBP] = FRAME_TOP - 0x100;
        // The caller's frame chain: [rbp - 8] = outer frame pointer 1, [rbp - 16] = pointer 2, ...
        for slot in 1..=8u64 {
            h.write_u64(FRAME_TOP - 0x100 - 8 * slot, 0x1000 + slot);
        }
    }

    #[test]
    fn enter_level_0_pushes_rbp_and_reserves_the_frame() {
        let mut pair = Pair::new(code::ENTER_16_0);
        pair.halted(enter_setup);
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 8), FRAME_TOP - 0x100);
        assert_eq!(pair.reg(RBP), FRAME_TOP - 8);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 8 - 16);
    }

    #[test]
    fn enter_level_0_with_zero_size() {
        let mut pair = Pair::new(code::ENTER_0_0);
        pair.halted(enter_setup);
        assert_eq!(pair.reg(RBP), FRAME_TOP - 8);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 8);
    }

    #[test]
    fn enter_level_1_pushes_rbp_then_the_frame_pointer() {
        let mut pair = Pair::new(code::ENTER_16_1);
        pair.halted(enter_setup);
        let frame = FRAME_TOP - 8;
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 8), FRAME_TOP - 0x100);
        // Level 1: no outer frame pointers are copied, only the new frame pointer is pushed.
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 16), frame);
        assert_eq!(pair.reg(RBP), frame);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 16 - 16);
    }

    #[test]
    fn enter_level_2_copies_one_outer_frame_pointer() {
        let mut pair = Pair::new(code::ENTER_8_2);
        pair.halted(enter_setup);
        let frame = FRAME_TOP - 8;
        assert_eq!(
            pair.interp.read_u64(FRAME_TOP - 8),
            FRAME_TOP - 0x100,
            "saved rbp"
        );
        assert_eq!(
            pair.interp.read_u64(FRAME_TOP - 16),
            0x1001,
            "[rbp - 8] copied"
        );
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 24), frame, "frame pointer");
        assert_eq!(pair.reg(RBP), frame);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 24 - 8);
    }

    #[test]
    fn enter_level_3_copies_two_outer_frame_pointers_in_order() {
        let mut pair = Pair::new(code::ENTER_0_3);
        pair.halted(enter_setup);
        let frame = FRAME_TOP - 8;
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 16), 0x1001);
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 24), 0x1002);
        assert_eq!(pair.interp.read_u64(FRAME_TOP - 32), frame);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 32);
        assert_eq!(pair.reg(RBP), frame);
    }

    #[test]
    fn enter_level_is_taken_modulo_32() {
        // Level 33 behaves as level 1, level 32 as level 0.
        let mut level_33 = Pair::new(code::ENTER_0_33);
        level_33.halted(enter_setup);
        let mut level_1 = Pair::new(code::ENTER_0_1);
        level_1.halted(enter_setup);
        assert_eq!(level_33.reg(RSP), level_1.reg(RSP));
        assert_eq!(level_33.reg(RBP), level_1.reg(RBP));
        let mut level_32 = Pair::new(code::ENTER_0_32);
        level_32.halted(enter_setup);
        assert_eq!(level_32.reg(RSP), FRAME_TOP - 8 - 0x10);
        assert_eq!(level_32.reg(RBP), FRAME_TOP - 8);
    }

    #[test]
    fn enter_with_operand_size_prefix_pushes_words() {
        let mut pair = Pair::new(code::ENTER_16_0_P66);
        pair.halted(|h| {
            enter_setup(h);
            h.write_u64(FRAME_TOP - 8, u64::MAX);
        });
        // Only the low word of rbp is pushed, two bytes below the old stack pointer.
        assert_eq!(
            pair.interp.read(FRAME_TOP - 2, 2),
            ((FRAME_TOP - 0x100) as u16).to_le_bytes()
        );
        assert_eq!(pair.interp.read(FRAME_TOP - 8, 6), vec![0xff; 6]);
        assert_eq!(pair.reg(RBP), FRAME_TOP - 2);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 2 - 16);
    }

    #[test]
    fn enter_level_2_with_operand_size_prefix_copies_one_word() {
        let mut pair = Pair::new(code::ENTER_16_2_P66);
        pair.halted(|h| {
            enter_setup(h);
            h.write(FRAME_TOP - 0x100 - 2, &0xbeefu16.to_le_bytes());
        });
        let frame = FRAME_TOP - 2;
        assert_eq!(pair.interp.read(FRAME_TOP - 4, 2), 0xbeefu16.to_le_bytes());
        assert_eq!(
            pair.interp.read(FRAME_TOP - 6, 2),
            (frame as u16).to_le_bytes()
        );
        assert_eq!(pair.reg(RBP), frame);
        assert_eq!(pair.reg(RSP), FRAME_TOP - 6 - 16);
    }

    #[test]
    fn enter_fault_leaves_rsp_and_rbp_unchanged() {
        let setup = |h: &mut Harness| {
            enter_setup(h);
            // The first push lands just above the missing page, the second one inside it.
            unmap_huge_page(h, HOLE);
            h.cpu.gpr[RSP] = HOLE + HUGE_PAGE_BYTES + 8;
        };
        assert_fault_changes_nothing(
            code::ENTER_0_1,
            setup,
            fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)),
        );
    }

    #[test]
    fn enter_with_a_non_canonical_stack_is_a_stack_fault() {
        let (pair, stop) = interpreter_regs_stop(code::ENTER_16_0, &[(RSP, NON_CANONICAL + 8)]);
        assert_eq!(stop, ss_fault(0));
        assert_eq!(pair.reg(RSP), NON_CANONICAL + 8);
    }

    /// #PF error code for a write to a not-present page from supervisor mode.
    const PF_WRITE_NOT_PRESENT: u32 = 2;
    /// #PF error code for a read of a not-present page from supervisor mode.
    const PF_READ_NOT_PRESENT: u32 = 0;

    #[test]
    fn leave_restores_rsp_from_rbp_and_pops_rbp() {
        let mut pair = Pair::new(code::LEAVE);
        pair.halted(|h| {
            h.cpu.gpr[RBP] = DATA + 0x40;
            h.write_u64(DATA + 0x40, 0xfeed);
        });
        assert_eq!(pair.reg(RSP), DATA + 0x48);
        assert_eq!(pair.reg(RBP), 0xfeed);
    }

    #[test]
    fn leave_with_an_operand_size_prefix_pops_a_word_into_bp() {
        let frame = DATA + 0x40;
        let mut pair = Pair::new(code::LEAVE_P66);
        pair.halted(|h| {
            h.cpu.gpr[RBP] = frame;
            h.write_u64(frame, 0xaaaa_bbbb_cccc_dddd);
        });
        assert_eq!(pair.reg(RSP), frame + 2, "a word is released");
        assert_eq!(
            pair.reg(RBP),
            (frame & !0xffff) | 0xdddd,
            "only bp is replaced"
        );
    }

    #[test]
    fn push_to_a_non_canonical_stack_raises_stack_fault() {
        // rsp - 8 is the first non-canonical address above the low half.
        let (pair, stop) = interpreter_regs_stop(code::PUSH_RAX, &[(RSP, NON_CANONICAL + 8)]);
        assert_eq!(stop, ss_fault(0));
        assert_eq!(pair.reg(RSP), NON_CANONICAL + 8);
        assert_eq!(pair.cpu().rip, CODE_BASE);
    }

    #[test]
    fn pop_from_a_non_canonical_stack_raises_stack_fault() {
        let (pair, stop) = interpreter_regs_stop(code::POP_RAX, &[(RSP, NON_CANONICAL)]);
        assert_eq!(stop, ss_fault(0));
        assert_eq!(pair.reg(RSP), NON_CANONICAL);
    }

    #[test]
    fn ret_call_leave_pushf_popf_on_a_non_canonical_stack_raise_stack_fault() {
        for (code, rsp, rbp) in [
            (code::RET, NON_CANONICAL, 0),
            (code::RET_IMM, NON_CANONICAL, 0),
            (code::CALL_REL, NON_CANONICAL + 8, 0),
            (code::LEAVE, STACK_TOP, NON_CANONICAL),
            (code::PUSHFQ, NON_CANONICAL + 8, 0),
            (code::POPFQ, NON_CANONICAL, 0),
            (code::IRETQ, NON_CANONICAL, 0),
            (code::RETFQ, NON_CANONICAL, 0),
        ] {
            let (pair, stop) = interpreter_regs_stop(code, &[(RSP, rsp), (RBP, rbp)]);
            assert_eq!(stop, ss_fault(0), "{code:02x?}");
            assert_eq!(pair.reg(RSP), rsp, "{code:02x?}: rsp unchanged");
            assert_eq!(pair.cpu().rip, CODE_BASE, "{code:02x?}");
        }
    }

    #[test]
    fn explicit_non_canonical_data_access_is_still_a_general_protection_fault() {
        // mov rax, [rbx] with a non-canonical rbx goes through DS, not SS.
        let (_, stop) = run_regs_stop(code::MOV_RAX_MEM_RBX, &[(RBX, NON_CANONICAL)]);
        assert_eq!(stop, gp(0));
    }

    #[test]
    fn push_and_pop_to_an_unmapped_page_are_page_faults_that_keep_rsp() {
        let setup = |h: &mut Harness| {
            unmap_huge_page(h, HOLE);
            h.cpu.gpr[RSP] = HOLE + 0x100;
        };
        let mut pair = Pair::new(code::PUSH_RAX);
        let stop = pair.run(setup);
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)));
        assert_eq!(pair.reg(RSP), HOLE + 0x100);
        assert_eq!(pair.cpu().cr2, HOLE + 0xf8);
        let mut pair = Pair::new(code::POP_RAX);
        let stop = pair.run(setup);
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(pair.reg(RSP), HOLE + 0x100);
        assert_eq!(pair.cpu().cr2, HOLE + 0x100);
    }
    const CONDITION_SNIPPETS: [ConditionSnippets; 16] = [
        ConditionSnippets {
            jcc8: code::JCC8_O,
            jcc32: code::JCC32_O,
            setcc: code::SETCC_O,
            setcc_mem: code::SETCC_MEM_O,
            cmov64: code::CMOV64_O,
            cmov32: code::CMOV32_O,
            cmov16: code::CMOV16_O,
            cmov_mem: code::CMOV_MEM_O,
        },
        ConditionSnippets {
            jcc8: code::JCC8_NO,
            jcc32: code::JCC32_NO,
            setcc: code::SETCC_NO,
            setcc_mem: code::SETCC_MEM_NO,
            cmov64: code::CMOV64_NO,
            cmov32: code::CMOV32_NO,
            cmov16: code::CMOV16_NO,
            cmov_mem: code::CMOV_MEM_NO,
        },
        ConditionSnippets {
            jcc8: code::JCC8_B,
            jcc32: code::JCC32_B,
            setcc: code::SETCC_B,
            setcc_mem: code::SETCC_MEM_B,
            cmov64: code::CMOV64_B,
            cmov32: code::CMOV32_B,
            cmov16: code::CMOV16_B,
            cmov_mem: code::CMOV_MEM_B,
        },
        ConditionSnippets {
            jcc8: code::JCC8_AE,
            jcc32: code::JCC32_AE,
            setcc: code::SETCC_AE,
            setcc_mem: code::SETCC_MEM_AE,
            cmov64: code::CMOV64_AE,
            cmov32: code::CMOV32_AE,
            cmov16: code::CMOV16_AE,
            cmov_mem: code::CMOV_MEM_AE,
        },
        ConditionSnippets {
            jcc8: code::JCC8_E,
            jcc32: code::JCC32_E,
            setcc: code::SETCC_E,
            setcc_mem: code::SETCC_MEM_E,
            cmov64: code::CMOV64_E,
            cmov32: code::CMOV32_E,
            cmov16: code::CMOV16_E,
            cmov_mem: code::CMOV_MEM_E,
        },
        ConditionSnippets {
            jcc8: code::JCC8_NE,
            jcc32: code::JCC32_NE,
            setcc: code::SETCC_NE,
            setcc_mem: code::SETCC_MEM_NE,
            cmov64: code::CMOV64_NE,
            cmov32: code::CMOV32_NE,
            cmov16: code::CMOV16_NE,
            cmov_mem: code::CMOV_MEM_NE,
        },
        ConditionSnippets {
            jcc8: code::JCC8_BE,
            jcc32: code::JCC32_BE,
            setcc: code::SETCC_BE,
            setcc_mem: code::SETCC_MEM_BE,
            cmov64: code::CMOV64_BE,
            cmov32: code::CMOV32_BE,
            cmov16: code::CMOV16_BE,
            cmov_mem: code::CMOV_MEM_BE,
        },
        ConditionSnippets {
            jcc8: code::JCC8_A,
            jcc32: code::JCC32_A,
            setcc: code::SETCC_A,
            setcc_mem: code::SETCC_MEM_A,
            cmov64: code::CMOV64_A,
            cmov32: code::CMOV32_A,
            cmov16: code::CMOV16_A,
            cmov_mem: code::CMOV_MEM_A,
        },
        ConditionSnippets {
            jcc8: code::JCC8_S,
            jcc32: code::JCC32_S,
            setcc: code::SETCC_S,
            setcc_mem: code::SETCC_MEM_S,
            cmov64: code::CMOV64_S,
            cmov32: code::CMOV32_S,
            cmov16: code::CMOV16_S,
            cmov_mem: code::CMOV_MEM_S,
        },
        ConditionSnippets {
            jcc8: code::JCC8_NS,
            jcc32: code::JCC32_NS,
            setcc: code::SETCC_NS,
            setcc_mem: code::SETCC_MEM_NS,
            cmov64: code::CMOV64_NS,
            cmov32: code::CMOV32_NS,
            cmov16: code::CMOV16_NS,
            cmov_mem: code::CMOV_MEM_NS,
        },
        ConditionSnippets {
            jcc8: code::JCC8_P,
            jcc32: code::JCC32_P,
            setcc: code::SETCC_P,
            setcc_mem: code::SETCC_MEM_P,
            cmov64: code::CMOV64_P,
            cmov32: code::CMOV32_P,
            cmov16: code::CMOV16_P,
            cmov_mem: code::CMOV_MEM_P,
        },
        ConditionSnippets {
            jcc8: code::JCC8_NP,
            jcc32: code::JCC32_NP,
            setcc: code::SETCC_NP,
            setcc_mem: code::SETCC_MEM_NP,
            cmov64: code::CMOV64_NP,
            cmov32: code::CMOV32_NP,
            cmov16: code::CMOV16_NP,
            cmov_mem: code::CMOV_MEM_NP,
        },
        ConditionSnippets {
            jcc8: code::JCC8_L,
            jcc32: code::JCC32_L,
            setcc: code::SETCC_L,
            setcc_mem: code::SETCC_MEM_L,
            cmov64: code::CMOV64_L,
            cmov32: code::CMOV32_L,
            cmov16: code::CMOV16_L,
            cmov_mem: code::CMOV_MEM_L,
        },
        ConditionSnippets {
            jcc8: code::JCC8_GE,
            jcc32: code::JCC32_GE,
            setcc: code::SETCC_GE,
            setcc_mem: code::SETCC_MEM_GE,
            cmov64: code::CMOV64_GE,
            cmov32: code::CMOV32_GE,
            cmov16: code::CMOV16_GE,
            cmov_mem: code::CMOV_MEM_GE,
        },
        ConditionSnippets {
            jcc8: code::JCC8_LE,
            jcc32: code::JCC32_LE,
            setcc: code::SETCC_LE,
            setcc_mem: code::SETCC_MEM_LE,
            cmov64: code::CMOV64_LE,
            cmov32: code::CMOV32_LE,
            cmov16: code::CMOV16_LE,
            cmov_mem: code::CMOV_MEM_LE,
        },
        ConditionSnippets {
            jcc8: code::JCC8_G,
            jcc32: code::JCC32_G,
            setcc: code::SETCC_G,
            setcc_mem: code::SETCC_MEM_G,
            cmov64: code::CMOV64_G,
            cmov32: code::CMOV32_G,
            cmov16: code::CMOV16_G,
            cmov_mem: code::CMOV_MEM_G,
        },
    ];

    struct ConditionSnippets {
        jcc8: &'static [u8],
        jcc32: &'static [u8],
        setcc: &'static [u8],
        setcc_mem: &'static [u8],
        cmov64: &'static [u8],
        cmov32: &'static [u8],
        cmov16: &'static [u8],
        cmov_mem: &'static [u8],
    }

    const STATUS_FLAGS_FOR_CONDITIONS: [u64; 5] = [CF, PF, ZF, SF, OF];

    /// All 32 combinations of the five flags the condition codes read.
    fn flag_combinations() -> Vec<u64> {
        (0..32u32)
            .map(|bits| {
                STATUS_FLAGS_FOR_CONDITIONS
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| bits >> index & 1 != 0)
                    .fold(0, |acc, (_, flag)| acc | flag)
            })
            .collect()
    }

    /// The SDM's table of condition mnemonics, by condition number (the low opcode nibble):
    /// o no b ae e ne be a s ns p np l ge le g.
    fn condition_expected(cc: usize, flags: u64) -> bool {
        let (cf, zf, sf, of, pf) = (
            flags & CF != 0,
            flags & ZF != 0,
            flags & SF != 0,
            flags & OF != 0,
            flags & PF != 0,
        );
        match cc {
            0 => of,
            1 => !of,
            2 => cf,
            3 => !cf,
            4 => zf,
            5 => !zf,
            6 => cf || zf,
            7 => !cf && !zf,
            8 => sf,
            9 => !sf,
            10 => pf,
            11 => !pf,
            12 => sf != of,
            13 => sf == of,
            14 => zf || sf != of,
            _ => !zf && sf == of,
        }
    }

    #[test]
    fn the_condition_table_used_by_these_tests_is_self_consistent() {
        for flags in flag_combinations() {
            for pair in (0..16).step_by(2) {
                assert_ne!(
                    condition_expected(pair, flags),
                    condition_expected(pair + 1, flags),
                    "odd conditions negate the even ones"
                );
            }
        }
        assert_eq!(flag_combinations().len(), 32);
    }

    #[test]
    fn raw_condition_encodings_match_the_assembler_spellings() {
        assert_eq!(code::JE_REAL, code::JCC8_E);
        assert_eq!(code::JNE_REAL, code::JCC8_NE);
        assert_eq!(code::JG_REAL, code::JCC8_G);
        assert_eq!(code::JB_REAL, code::JCC8_B);
    }

    #[test]
    fn jcc_rel8_and_rel32_follow_every_condition_for_every_flag_combination() {
        for (cc, snippets) in CONDITION_SNIPPETS.iter().enumerate() {
            for snippet in [snippets.jcc8, snippets.jcc32] {
                let mut pair = Pair::new(snippet);
                for flags in flag_combinations() {
                    pair.halted(|h| h.cpu.set_eflags_arith(flags | AF));
                    let taken = condition_expected(cc, flags);
                    let context = format!("cc {cc} flags {}", flag_names(flags));
                    assert_eq!(pair.reg(RAX), if taken { 2 } else { 1 }, "{context}");
                    let end = CODE_BASE + snippet.len() as u64;
                    let not_taken_end = end - 6;
                    assert_eq!(
                        pair.cpu().rip,
                        if taken { end } else { not_taken_end },
                        "{context}"
                    );
                    assert_flags(
                        pair.flags(),
                        ALL_ARITH,
                        flags | AF,
                        &format!("{context}: jcc leaves flags"),
                    );
                }
            }
        }
    }

    #[test]
    fn jcc_taken_and_not_taken_branches_back_and_forth() {
        // Backward conditional branch: loop counter in rcx.
        let mut pair = Pair::new(code::LOOP_COUNTING_DOWN);
        pair.halted(|h| h.cpu.gpr[RCX] = 5);
        assert_eq!(pair.reg(RAX), 15, "5 + 4 + 3 + 2 + 1");
        assert_eq!(pair.reg(RCX), 0);
    }

    #[test]
    fn setcc_writes_exactly_one_byte_for_every_condition() {
        for (cc, snippets) in CONDITION_SNIPPETS.iter().enumerate() {
            let mut register = Pair::new(snippets.setcc);
            let mut memory = Pair::new(snippets.setcc_mem);
            for flags in flag_combinations() {
                let want = condition_expected(cc, flags) as u8;
                register.halted(|h| {
                    h.cpu.gpr[RAX] = u64::MAX;
                    h.cpu.set_eflags_arith(flags);
                });
                assert_eq!(
                    register.reg(RAX),
                    !0xff | want as u64,
                    "cc {cc} {}",
                    flag_names(flags)
                );
                memory.halted(|h| {
                    memory_with_marker(h, 0xee, 1);
                    h.cpu.set_eflags_arith(flags);
                });
                assert_eq!(
                    memory.interp.read(DATA, 2),
                    [want, MEMORY_MARKER],
                    "cc {cc} {}",
                    flag_names(flags)
                );
                assert_flags(memory.flags(), ALL_ARITH, flags, "setcc leaves flags");
            }
        }
    }

    #[test]
    fn cmovcc_moves_only_when_the_condition_holds_and_keeps_the_flags() {
        let (old, new) = (0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888u64);
        for (cc, snippets) in CONDITION_SNIPPETS.iter().enumerate() {
            let mut cmov64 = Pair::new(snippets.cmov64);
            let mut cmov32 = Pair::new(snippets.cmov32);
            let mut cmov16 = Pair::new(snippets.cmov16);
            let mut cmov_mem = Pair::new(snippets.cmov_mem);
            for flags in flag_combinations() {
                let taken = condition_expected(cc, flags);
                let context = format!("cc {cc} {}", flag_names(flags));
                let setup = |h: &mut Harness| {
                    h.cpu.gpr[RAX] = old;
                    h.cpu.gpr[RBX] = new;
                    h.cpu.set_eflags_arith(flags);
                };
                cmov64.halted(setup);
                assert_eq!(
                    cmov64.reg(RAX),
                    if taken { new } else { old },
                    "{context}: 64-bit"
                );
                assert_flags(cmov64.flags(), ALL_ARITH, flags, &context);
                cmov32.halted(setup);
                // A 32-bit cmov zero-extends its destination even when the move is not taken.
                assert_eq!(
                    cmov32.reg(RAX),
                    if taken {
                        new & 0xffff_ffff
                    } else {
                        old & 0xffff_ffff
                    },
                    "{context}: 32-bit"
                );
                cmov16.halted(setup);
                assert_eq!(
                    cmov16.reg(RAX),
                    if taken {
                        old & !0xffff | new & 0xffff
                    } else {
                        old
                    },
                    "{context}: 16-bit"
                );
                cmov_mem.halted(|h| {
                    setup(h);
                    memory_with_marker(h, new, 8);
                });
                assert_eq!(
                    cmov_mem.reg(RAX),
                    if taken { new } else { old },
                    "{context}: memory source"
                );
            }
        }
    }

    #[test]
    fn cmovcc_reads_its_memory_source_even_when_the_condition_is_false() {
        // cmovo with OF clear: the load still happens and faults on an unmapped address.
        let (pair, stop) = {
            let mut pair = Pair::new(CONDITION_SNIPPETS[0].cmov_mem);
            let stop = pair.run(|h| {
                h.cpu.gpr[RDX] = UNMAPPED;
                h.cpu.set_eflags_arith(0);
            });
            (pair, stop)
        };
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(pair.cpu().rip, CODE_BASE);
    }

    /// Address of the second `mov r, 2; hlt` pair that the indirect-branch snippets end with.
    fn second_target(snippet: &[u8]) -> u64 {
        CODE_BASE + snippet.len() as u64 - 6
    }

    #[test]
    fn direct_jumps_forward_backward_and_rel32() {
        assert_eq!(run_regs(code::JMP_FWD, &[]).reg(RAX), 2);
        assert_eq!(run_regs(code::JMP_REL32, &[]).reg(RAX), 2);
        assert_eq!(run_regs(code::JMP_BACK, &[]).reg(RAX), 3);
    }

    #[test]
    fn indirect_jumps_through_a_register_and_through_memory() {
        let target = second_target(code::JMP_REG);
        assert_eq!(run_regs(code::JMP_REG, &[(RAX, target)]).reg(RCX), 2);
        assert_eq!(
            run_regs(code::JMP_R9, &[(R9, second_target(code::JMP_R9))]).reg(RCX),
            2
        );
        let mut pair = Pair::new(code::JMP_MEM);
        pair.halted(|h| {
            h.write_u64(DATA, second_target(code::JMP_MEM));
            h.cpu.gpr[RDX] = DATA;
        });
        assert_eq!(pair.reg(RCX), 2);
        assert_eq!(pair.reg(RSP), STACK_TOP, "jmp pushes nothing");
    }

    #[test]
    fn indirect_jump_to_a_non_canonical_target_raises_general_protection() {
        let (_, stop) = run_regs_stop(code::JMP_REG, &[(RAX, NON_CANONICAL)]);
        assert_eq!(stop, gp(0));
    }

    #[test]
    fn call_pushes_the_return_address_and_transfers_control() {
        let mut pair = Pair::new(code::CALL_REG);
        pair.halted(|h| h.cpu.gpr[RAX] = second_target(code::CALL_REG));
        assert_eq!(pair.reg(RCX), 2);
        assert_eq!(pair.reg(RSP), STACK_TOP - 8);
        assert_eq!(
            pair.interp.read_u64(STACK_TOP - 8),
            CODE_BASE + 2,
            "address of the next instruction"
        );

        let mut pair = Pair::new(code::CALL_MEM);
        pair.halted(|h| {
            h.write_u64(DATA, second_target(code::CALL_MEM));
            h.cpu.gpr[RDX] = DATA;
        });
        assert_eq!(pair.reg(RCX), 2);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), CODE_BASE + 2);

        let (pair, _) = run_regs_stop(code::CALL_REL, &[]);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), CODE_BASE + 5);
    }

    #[test]
    fn call_through_a_memory_operand_at_rsp_reads_the_target_before_pushing() {
        let mut pair = Pair::new(code::CALL_RSP_MEM);
        pair.halted(|h| {
            h.cpu.gpr[RSP] = STACK_TOP - 0x100;
            h.write_u64(STACK_TOP - 0x100, second_target(code::CALL_RSP_MEM));
        });
        assert_eq!(pair.reg(RCX), 2);
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x108);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 0x108), CODE_BASE + 3);
    }

    #[test]
    fn call_and_ret_round_trip() {
        let pair = run_regs(code::CALL_RET, &[]);
        assert_eq!((pair.reg(RAX), pair.reg(RBX)), (5, 6));
        assert_eq!(pair.reg(RSP), STACK_TOP);
        // ret imm16 also releases the caller's argument bytes.
        let pair = run_regs(code::CALL_RET_IMM, &[]);
        assert_eq!(pair.reg(RSP), STACK_TOP + 16);
    }

    #[test]
    fn ret_pops_the_target_and_ret_imm_releases_extra_bytes() {
        let mut pair = Pair::new(code::RET);
        pair.halted(|h| {
            h.cpu.gpr[RSP] = STACK_TOP - 0x40;
            h.write_u64(STACK_TOP - 0x40, CODE_BASE + 1);
        });
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x38);
        assert_eq!(
            pair.cpu().rip,
            CODE_BASE + 2,
            "landed on the hlt after the ret"
        );
        let mut pair = Pair::new(code::RET_IMM);
        pair.halted(|h| {
            h.cpu.gpr[RSP] = STACK_TOP - 0x40;
            h.write_u64(STACK_TOP - 0x40, CODE_BASE + 3);
        });
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x40 + 8 + 16);
        let mut pair = Pair::new(code::REP_RET);
        pair.halted(|h| {
            h.cpu.gpr[RSP] = STACK_TOP - 0x40;
            h.write_u64(STACK_TOP - 0x40, CODE_BASE + 2);
        });
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x38, "rep ret is a plain ret");
    }

    #[test]
    fn far_indirect_jump_and_call_and_odd_group_5_and_4_forms_are_undefined() {
        for snippet in [
            code::JMP_FAR_IND,
            code::CALL_FAR_IND,
            code::GROUP5_REG7,
            code::GROUP4_CALL,
            code::GROUP4_PUSH,
            code::GROUP4_JMP,
        ] {
            let (pair, stop) = run_regs_stop(snippet, &[(RDX, DATA)]);
            assert_eq!(stop, ud(), "{snippet:02x?}");
            assert_eq!(pair.cpu().rip, CODE_BASE);
            assert_eq!(pair.reg(RSP), STACK_TOP);
        }
    }

    fn loop_outcome(snippet: &[u8], rcx: u64, flags: u64) -> (bool, u64, u64) {
        let mut pair = Pair::new(snippet);
        pair.halted(|h| {
            h.cpu.gpr[RCX] = rcx;
            h.cpu.set_eflags_arith(flags);
        });
        (pair.reg(RAX) == 2, pair.reg(RCX), pair.flags())
    }

    #[test]
    fn loop_decrements_rcx_and_branches_while_it_is_nonzero() {
        for (rcx_in, taken, rcx_out) in [
            (3u64, true, 2u64),
            (2, true, 1),
            (1, false, 0),
            (0, true, u64::MAX),
            (0x1_0000_0001, true, 0x1_0000_0000),
        ] {
            let (was_taken, rcx, flags) = loop_outcome(code::LOOP_, rcx_in, ZF | CF);
            assert_eq!((was_taken, rcx), (taken, rcx_out), "rcx {rcx_in:#x}");
            assert_flags(flags, ALL_ARITH, ZF | CF, "loop leaves flags");
        }
    }

    #[test]
    fn loope_and_loopne_also_test_zf() {
        for (snippet, name, take_when_zf) in [
            (code::LOOPE, "loope", true),
            (code::LOOPNE, "loopne", false),
        ] {
            for zf in [0, ZF] {
                for rcx_in in [1u64, 2, 0] {
                    let (taken, rcx, _) = loop_outcome(snippet, rcx_in, zf);
                    let new_rcx = rcx_in.wrapping_sub(1);
                    assert_eq!(rcx, new_rcx, "{name}");
                    let want = new_rcx != 0 && ((zf != 0) == take_when_zf);
                    assert_eq!(taken, want, "{name} rcx {rcx_in} zf {zf:#x}");
                }
            }
        }
    }

    #[test]
    fn jrcxz_tests_rcx_without_changing_it_or_the_flags() {
        for rcx_in in [0u64, 1, 0x1_0000_0000, u64::MAX] {
            let (taken, rcx, flags) = loop_outcome(code::JRCXZ, rcx_in, ALL_ARITH);
            assert_eq!(taken, rcx_in == 0, "rcx {rcx_in:#x}");
            assert_eq!(rcx, rcx_in);
            assert_flags(flags, ALL_ARITH, ALL_ARITH, "jrcxz leaves flags");
        }
    }

    #[test]
    fn address_size_prefix_makes_loop_and_jrcxz_use_ecx() {
        // jecxz: only the low 32 bits matter.
        for (rcx_in, taken) in [
            (0u64, true),
            (0x1_0000_0000, true),
            (1, false),
            (0x1_0000_0001, false),
        ] {
            let (was_taken, rcx, _) = loop_outcome(code::JECXZ, rcx_in, 0);
            assert_eq!((was_taken, rcx), (taken, rcx_in), "jecxz rcx {rcx_in:#x}");
        }
        // loop with ecx: decrements the 32-bit register, zero-extending it.
        let (taken, rcx, _) = loop_outcome(code::LOOP_ADDR32, 0x1_0000_0001, 0);
        assert!(!taken);
        assert_eq!(rcx, 0);
        let (taken, rcx, _) = loop_outcome(code::LOOP_ADDR32, 0x1_0000_0000, 0);
        assert!(taken, "ecx wraps from 0 to 0xffffffff, which is nonzero");
        assert_eq!(rcx, 0xffff_ffff);
        let (taken, _, _) = loop_outcome(code::LOOPE_ADDR32, 5, ZF);
        assert!(taken);
        let (taken, _, _) = loop_outcome(code::LOOPE_ADDR32, 5, 0);
        assert!(!taken);
        let (taken, _, _) = loop_outcome(code::LOOPNE_ADDR32, 5, 0);
        assert!(taken);
    }

    fn stack_bytes(pair: &Pair, below_top: u64, len: usize) -> Vec<u8> {
        pair.interp.read(STACK_TOP - below_top, len)
    }

    #[test]
    fn push_register_pushes_eight_bytes_and_moves_rsp_down() {
        let value = 0x1122_3344_5566_7788u64;
        let pair = run_regs(code::PUSH_RAX_, &[(RAX, value)]);
        assert_eq!(pair.reg(RSP), STACK_TOP - 8);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), value);
        let pair = run_regs(code::PUSH_R8, &[(R8, value)]);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), value);
        let pair = run_regs(code::PUSH_RSP_, &[]);
        assert_eq!(
            pair.interp.read_u64(STACK_TOP - 8),
            STACK_TOP,
            "push rsp pushes the value before the decrement"
        );
        assert_eq!(pair.reg(RSP), STACK_TOP - 8);
    }

    #[test]
    fn push_immediate_is_sign_extended_to_the_pushed_width() {
        assert_eq!(
            run_regs(code::PUSH_IMM8, &[])
                .interp
                .read_u64(STACK_TOP - 8),
            0xffff_ffff_ffff_fffe
        );
        assert_eq!(
            run_regs(code::PUSH_IMM32, &[])
                .interp
                .read_u64(STACK_TOP - 8),
            0x1234_5678
        );
        assert_eq!(
            run_regs(code::PUSH_IMM32_NEG, &[])
                .interp
                .read_u64(STACK_TOP - 8),
            0xffff_ffff_edcb_a988
        );
    }

    #[test]
    fn push_with_an_operand_size_prefix_pushes_two_bytes() {
        let pair = run_regs(code::PUSH_AX, &[(RAX, 0x1122_3344_5566_7788)]);
        assert_eq!(pair.reg(RSP), STACK_TOP - 2);
        assert_eq!(stack_bytes(&pair, 2, 4), [0x88, 0x77, 0, 0]);
        let pair = run_regs(code::PUSH_IMM16, &[]);
        assert_eq!(pair.reg(RSP), STACK_TOP - 2);
        assert_eq!(stack_bytes(&pair, 2, 2), [0x34, 0x12]);
        let pair = run_regs(code::PUSH_IMM8_WORD, &[]);
        assert_eq!(stack_bytes(&pair, 2, 2), [0xfe, 0xff]);
    }

    #[test]
    fn push_from_memory_reads_the_operand_before_moving_rsp() {
        let mut pair = Pair::new(code::PUSH_MEM);
        pair.halted(|h| {
            h.write_u64(DATA, 0xaabb_ccdd_eeff_0011);
            h.cpu.gpr[RDX] = DATA;
        });
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), 0xaabb_ccdd_eeff_0011);
        let mut pair = Pair::new(code::PUSH_MEM16);
        pair.halted(|h| {
            h.write_u64(DATA, 0xaabb_ccdd_eeff_0011);
            h.cpu.gpr[RDX] = DATA;
        });
        assert_eq!(pair.reg(RSP), STACK_TOP - 2);
        assert_eq!(stack_bytes(&pair, 2, 2), [0x11, 0x00]);
    }

    fn stack_with(h: &mut Harness, offset_below_top: u64, value: u64) {
        h.cpu.gpr[RSP] = STACK_TOP - offset_below_top;
        h.write_u64(STACK_TOP - offset_below_top, value);
    }

    #[test]
    fn pop_loads_the_register_and_moves_rsp_up() {
        let value = 0x1122_3344_5566_7788u64;
        let mut pair = Pair::new(code::POP_RAX_);
        pair.halted(|h| stack_with(h, 0x40, value));
        assert_eq!(pair.reg(RAX), value);
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x38);
        let mut pair = Pair::new(code::POP_R9);
        pair.halted(|h| stack_with(h, 0x40, value));
        assert_eq!(pair.reg(R9), value);
        let mut pair = Pair::new(code::POP_AX);
        pair.halted(|h| {
            stack_with(h, 0x40, value);
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(
            pair.reg(RAX),
            0xffff_ffff_ffff_7788,
            "16-bit pop merges into ax"
        );
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x40 + 2);
    }

    #[test]
    fn pop_rsp_takes_the_popped_value_not_the_incremented_pointer() {
        let mut pair = Pair::new(code::POP_RSP_);
        pair.halted(|h| stack_with(h, 0x40, 0x1234_5000));
        assert_eq!(pair.reg(RSP), 0x1234_5000);
    }

    #[test]
    fn pop_to_memory_addressed_through_rsp_uses_the_incremented_value() {
        // The destination is computed after rsp has been incremented.
        let mut pair = Pair::new(code::POP_MEM_RSP);
        pair.halted(|h| {
            stack_with(h, 0x100, 0xfeed_face_cafe_beef);
            h.write_u64(STACK_TOP - 0xf8, 0);
        });
        assert_eq!(
            pair.interp.read_u64(STACK_TOP - 0xf8),
            0xfeed_face_cafe_beef
        );
        assert_eq!(pair.reg(RSP), STACK_TOP - 0xf8);
        let mut pair = Pair::new(code::POP_MEM_RSP_DISP);
        pair.halted(|h| stack_with(h, 0x100, 0x1122_3344_5566_7788));
        assert_eq!(
            pair.interp.read_u64(STACK_TOP - 0xf8 + 8),
            0x1122_3344_5566_7788
        );
        let mut pair = Pair::new(code::POP_MEM);
        pair.halted(|h| {
            stack_with(h, 0x100, 0x1122_3344_5566_7788);
            h.cpu.gpr[RDX] = DATA;
        });
        assert_eq!(pair.interp.read_u64(DATA), 0x1122_3344_5566_7788);
        let mut pair = Pair::new(code::POP_MEM16);
        pair.halted(|h| {
            stack_with(h, 0x100, 0x1122_3344_5566_7788);
            memory_with_marker(h, 0, 0);
        });
        assert_eq!(
            pair.interp.read(DATA, 4),
            [0x88, 0x77, MEMORY_MARKER, MEMORY_MARKER]
        );
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x100 + 2);
    }

    #[test]
    fn pop_to_a_faulting_memory_destination_keeps_rsp() {
        assert_fault_changes_nothing(
            code::POP_MEM,
            |h| {
                write_protect_huge_page(h, HOLE);
                stack_with(h, 0x100, 5);
                h.cpu.gpr[RDX] = HOLE + 0x40;
            },
            fault(VECTOR_PAGE_FAULT, Some(PF_PROTECTION_WRITE)),
        );
    }

    #[test]
    fn push_and_pop_wrap_around_the_ends_of_the_address_space() {
        // push at rsp = 0 stores at the very top of the address space.
        let mut pair = Pair::new(code::PUSH_RAX_);
        pair.halted(|h| {
            map_top_page(h);
            h.cpu.gpr[RSP] = 0;
            h.cpu.gpr[RAX] = 0x4242;
        });
        assert_eq!(pair.reg(RSP), 0xffff_ffff_ffff_fff8);
        assert_eq!(
            pair.interp
                .read_u64(top_page_physical(0xffff_ffff_ffff_fff8)),
            0x4242
        );
        let mut pair = Pair::new(code::POP_RAX_);
        pair.halted(|h| {
            map_top_page(h);
            h.cpu.gpr[RSP] = 0xffff_ffff_ffff_fff8;
            h.write_u64(top_page_physical(0xffff_ffff_ffff_fff8), 0x4343);
        });
        assert_eq!(pair.reg(RAX), 0x4343);
        assert_eq!(pair.reg(RSP), 0, "rsp wraps to zero");
    }

    #[test]
    fn push_and_pop_of_fs_and_gs_use_the_selector() {
        let mut pair = Pair::new(code::PUSH_FS);
        pair.halted(|h| h.cpu.seg[SEG_FS].sel = 0x1234);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), 0x1234);
        let mut pair = Pair::new(code::PUSH_GS);
        pair.halted(|h| h.cpu.seg[SEG_GS].sel = 0x2345);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), 0x2345);
        // With a 66 prefix only two bytes are pushed.
        let mut pair = Pair::new(code::PUSH_FS_16);
        pair.halted(|h| h.cpu.seg[SEG_FS].sel = 0x1234);
        assert_eq!(pair.reg(RSP), STACK_TOP - 2);
        assert_eq!(stack_bytes(&pair, 2, 2), [0x34, 0x12]);
        for (snippet, register) in [(code::POP_FS, SEG_FS), (code::POP_GS, SEG_GS)] {
            let mut pair = Pair::new(snippet);
            let stop = pair.run_for(1, |h| {
                enter_cpl(h, 0);
                stack_with(h, 0x40, SEL_KDATA as u64 | 0xffff_0000);
            });
            assert_eq!(stop, Stop::Limit);
            assert_eq!(
                pair.cpu().seg[register].sel,
                SEL_KDATA,
                "only the low 16 bits are the selector"
            );
            assert_eq!(pair.reg(RSP), STACK_TOP - 0x38);
        }
    }

    #[test]
    fn pop_fs_with_a_bad_selector_faults_and_restores_rsp() {
        assert_fault_changes_nothing(
            code::POP_FS,
            |h| {
                enter_cpl(h, 0);
                stack_with(h, 0x40, SEL_SYSTEM as u64);
            },
            gp(selector_code(SEL_SYSTEM)),
        );
        assert_fault_changes_nothing(
            code::POP_GS,
            |h| {
                enter_cpl(h, 0);
                stack_with(h, 0x40, SEL_DATA_NOT_PRESENT as u64);
            },
            np(selector_code(SEL_DATA_NOT_PRESENT)),
        );
    }

    #[test]
    fn pop_fs_with_an_operand_size_prefix_pops_two_bytes() {
        let mut pair = Pair::new(code::POP_FS_16);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            stack_with(h, 0x40, SEL_KDATA as u64 | 0xaaaa_0000);
        });
        assert_eq!(stop, Stop::Limit);
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x40 + 2);
        assert_eq!(pair.cpu().seg[SEG_FS].sel, SEL_KDATA);
    }

    const WRITE_PROTECTED_ADDRESS: u64 = HOLE + 0x40;

    /// Starting state for the restart tests: carry set (adc/sbb/rcl/rcr read it), the other status
    /// flags in a recognizable pattern, and a destination on a read-only page.
    fn write_fault_setup(h: &mut Harness) {
        write_protect_huge_page(h, HOLE);
        h.write_u64(WRITE_PROTECTED_ADDRESS, 0x1234_5678_9abc_def0);
        h.cpu.gpr[RDX] = WRITE_PROTECTED_ADDRESS;
        h.cpu.gpr[RAX] = 0x0123_4567_89ab_cdef;
        h.cpu.gpr[RBX] = 0x0000_0000_0000_0009;
        h.cpu.gpr[RCX] = 3;
        h.cpu.set_eflags_arith(CF | ZF | PF | OF);
    }

    #[test]
    fn memory_destination_write_faults_leave_flags_and_registers_untouched() {
        let snippets: &[(&str, &[u8])] = &[
            ("adc [mem], reg", code::ADC_8_MR),
            ("sbb [mem], reg", code::SBB_4_MR),
            ("add [mem], reg", code::ADD_2_MR),
            ("and [mem], imm", code::AND_8_MI_P),
            ("xor [mem], reg", code::XOR_1_MR),
            ("or [mem], imm8", code::OR_8_MIS),
            ("inc [mem]", code::INC_MEM_4),
            ("dec [mem]", code::DEC_MEM_8),
            ("neg [mem]", code::NEG_MEM_8),
            ("not [mem]", code::NOT_MEM_2),
            ("rcl [mem], cl", code::RCL_4_MEM_CL),
            ("rcr [mem], cl", code::RCR_8_MEM_CL),
            ("rol [mem], 1", code::ROL_1_MEM_ONE),
            ("shl [mem], cl", code::SHL_8_MEM_CL),
            ("sar [mem], 1", code::SAR_2_MEM_ONE),
            ("shld [mem], reg, cl", code::SHLD_MEM_CL),
            ("shrd [mem], reg, imm", code::SHRD_MEM_IMM),
            ("xadd [mem], reg", code::XADD_MEM_8),
            ("cmpxchg [mem], reg", code::CMPXCHG_MEM_8),
            ("xchg [mem], reg", code::XCHG_MEM_RBX),
            ("bts [mem], reg", code::BTS_MEM_REG),
            ("btc [mem], imm", code::BTC_MEM_IMM),
            ("btr [mem], reg", code::BTR_MEM_REG),
            ("sete [mem]", code::SETE_MEM),
            ("mov [mem], reg", code::MOV_MEM_RAX2),
        ];
        for (name, snippet) in snippets {
            let mut pair = Pair::new(snippet);
            let before = std::cell::RefCell::new(None);
            let stop = pair.run(|h| {
                write_fault_setup(h);
                *before.borrow_mut() = Some(capture(&h.cpu));
            });
            assert_eq!(
                stop,
                fault(VECTOR_PAGE_FAULT, Some(PF_PROTECTION_WRITE)),
                "{name}"
            );
            assert_eq!(pair.cpu().cr2, WRITE_PROTECTED_ADDRESS, "{name}");
            assert_eq!(
                capture(pair.cpu()),
                before.into_inner().unwrap(),
                "{name}: state after the faulting write"
            );
            assert_eq!(
                pair.interp.read_u64(WRITE_PROTECTED_ADDRESS),
                0x1234_5678_9abc_def0,
                "{name}: memory"
            );
        }
    }

    #[test]
    fn bt_only_reads_so_a_write_protected_page_is_fine() {
        let mut pair = Pair::new(code::BT_8_MR);
        pair.halted(|h| {
            write_protect_huge_page(h, HOLE);
            h.write_u64(WRITE_PROTECTED_ADDRESS, 0b100);
            h.cpu.gpr[RDX] = WRITE_PROTECTED_ADDRESS;
            h.cpu.gpr[RBX] = 2;
        });
        assert_flags(pair.flags(), CF, CF, "bit 2 is set");
    }
    const INVALID_IN_LONG_MODE: &[(&str, &[u8])] = &[
        ("push es", code::INVALID_06),
        ("pop es", code::INVALID_07),
        ("push cs", code::INVALID_0E),
        ("push ss", code::INVALID_16),
        ("pop ss", code::INVALID_17),
        ("push ds", code::INVALID_1E),
        ("pop ds", code::INVALID_1F),
        ("daa", code::INVALID_27),
        ("das", code::INVALID_2F),
        ("aaa", code::INVALID_37),
        ("aas", code::INVALID_3F),
        ("pusha", code::INVALID_60),
        ("popa", code::INVALID_61),
        ("into", code::INVALID_CE),
        ("salc", code::INVALID_D6),
        ("bound/evex", code::INVALID_62_00),
        ("group1 alias", code::INVALID_82),
        ("far call", code::INVALID_9A),
        ("far jmp", code::INVALID_EA),
        ("aam", code::INVALID_D4),
        ("aad", code::INVALID_D5),
        ("les/vex", code::INVALID_C4),
        ("lds/vex", code::INVALID_C5),
    ];
    const UNDEFINED_TWO_BYTE: &[(&str, &[u8])] = &[
        ("0f04", code::UNDEF_0F04),
        ("0f0a", code::UNDEF_0F0A),
        ("0f0c", code::UNDEF_0F0C),
        ("0f24", code::UNDEF_0F24),
        ("0f26", code::UNDEF_0F26),
        ("0fb9", code::UNDEF_0FB9),
        ("0fff", code::UNDEF_0FFF),
        ("0f36", code::UNDEF_0F36),
        ("0fa6", code::UNDEF_0FA6),
        ("0fa7", code::UNDEF_0FA7),
        ("0f78", code::UNDEF_0F78),
        ("0f79", code::UNDEF_0F79),
    ];

    const TSS_IO_MAP_BASE_FIELD: u64 = 102;
    /// A free, unmapped port: `in` reads all ones from it and `out` is dropped.
    const SPARE_PORT: u16 = 0x1234;
    const MAP_BASE: u16 = 0x68;
    /// Limit that covers the bitmap for every 16-bit port plus its trailing byte.
    const FULL_TSS_LIMIT: u32 = MAP_BASE as u32 + 0x2000;
    const TSS_TYPE_AVAILABLE: u32 = 9;
    const TSS_TYPE_SHIFT: u32 = 8;
    const TSS_SELECTOR: u16 = 0x28;
    const TSS_AVAILABLE_FLAGS: u32 = SEG_PRESENT | (TSS_TYPE_AVAILABLE << TSS_TYPE_SHIFT);

    /// Installs a TSS whose I/O bitmap starts at `map_base` and denies the listed ports.
    fn install_io_bitmap(h: &mut Harness, map_base: u16, limit: u32, denied: &[u16]) {
        h.write(TSS_BASE, &vec![0u8; 0x3000]);
        h.write(TSS_BASE + TSS_IO_MAP_BASE_FIELD, &map_base.to_le_bytes());
        for port in denied {
            let byte = TSS_BASE + map_base as u64 + (*port as u64 >> 3);
            let current = h.read(byte, 1)[0];
            h.write(byte, &[current | 1 << (port & 7)]);
        }
        h.cpu.tr = Seg {
            sel: TSS_SELECTOR,
            base: TSS_BASE,
            limit,
            flags: TSS_AVAILABLE_FLAGS,
        };
    }

    /// Runs one port I/O instruction at `cpl` with the given IOPL and expects `expected`.
    fn run_io(
        snippet: &[u8],
        cpl: u8,
        iopl: u64,
        port: u16,
        tss: impl Fn(&mut Harness),
    ) -> (Pair, Stop) {
        let mut pair = Pair::new(snippet);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, cpl);
            h.cpu.eflags = RESERVED_ONE | iopl << IOPL_SHIFT;
            h.cpu.gpr[RDX] = port as u64;
            h.cpu.gpr[RAX] = 0x1111_2222_3333_4444;
            tss(h);
        });
        (pair, stop)
    }

    const IN_BY_SIZE: [(u8, &[u8]); 3] = [
        (1, code::IN_AL_DX),
        (2, code::IN_AX_DX),
        (4, code::IN_EAX_DX),
    ];
    const OUT_BY_SIZE: [(u8, &[u8]); 3] = [
        (1, code::OUT_DX_AL),
        (2, code::OUT_DX_AX),
        (4, code::OUT_DX_EAX),
    ];

    #[test]
    fn io_at_cpl0_never_consults_the_bitmap() {
        // No TSS at all, and a TSS that would deny the port: both fine at CPL 0.
        for (size, snippet) in IN_BY_SIZE {
            let (pair, stop) = run_io(snippet, 0, 0, SPARE_PORT, |_| {});
            assert_eq!(stop, Stop::Limit, "in size {size}");
            let ones = mask(size);
            assert_eq!(
                pair.reg(RAX) & ones,
                ones,
                "unmapped ports read as all ones"
            );
            let (_, stop) = run_io(snippet, 0, 0, SPARE_PORT, |h| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[SPARE_PORT]);
            });
            assert_eq!(stop, Stop::Limit);
        }
    }

    #[test]
    fn io_is_allowed_outright_when_iopl_covers_the_cpl() {
        for cpl in [1u8, 2] {
            for iopl in cpl as u64..=3 {
                let (_, stop) = run_io(code::IN_AL_DX, cpl, iopl, SPARE_PORT, |_| {});
                assert_eq!(stop, Stop::Limit, "cpl {cpl} iopl {iopl}: no TSS needed");
            }
        }
    }

    #[test]
    fn io_above_iopl_without_a_usable_tss_raises_general_protection() {
        for (size, snippet) in IN_BY_SIZE.iter().chain(OUT_BY_SIZE.iter()) {
            let (pair, stop) = run_io(snippet, 1, 0, SPARE_PORT, |_| {});
            assert_eq!(stop, gp(0), "size {size}: no TSS loaded");
            assert_eq!(pair.cpu().rip, CODE_BASE);
            assert_eq!(pair.reg(RAX), 0x1111_2222_3333_4444, "register untouched");
            // Limit too small to hold the map base field.
            let (_, stop) = run_io(snippet, 1, 0, SPARE_PORT, |h| {
                install_io_bitmap(h, MAP_BASE, TSS_IO_MAP_BASE_FIELD as u32, &[]);
            });
            assert_eq!(stop, gp(0), "size {size}: limit below the map base field");
        }
    }

    #[test]
    fn io_above_iopl_is_allowed_when_every_covered_bit_is_clear() {
        for (size, snippet) in IN_BY_SIZE {
            let (pair, stop) = run_io(snippet, 1, 0, SPARE_PORT, |h| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[]);
            });
            assert_eq!(stop, Stop::Limit, "in size {size}");
            assert_eq!(pair.reg(RAX) & mask(size), mask(size));
            assert_eq!(pair.cpu().rip, CODE_BASE + snippet.len() as u64 - 1);
        }
        for (size, snippet) in OUT_BY_SIZE {
            let (_, stop) = run_io(snippet, 1, 0, SPARE_PORT, |h| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[]);
            });
            assert_eq!(stop, Stop::Limit, "out size {size}");
        }
    }

    #[test]
    fn io_above_iopl_is_denied_when_the_ports_bit_is_set() {
        for (size, snippet) in IN_BY_SIZE.iter().chain(OUT_BY_SIZE.iter()) {
            let (pair, stop) = run_io(snippet, 1, 0, SPARE_PORT, |h| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[SPARE_PORT]);
            });
            assert_eq!(stop, gp(0), "size {size}");
            assert_eq!(pair.cpu().rip, CODE_BASE);
            assert_eq!(pair.reg(RAX), 0x1111_2222_3333_4444);
        }
    }

    #[test]
    fn io_bitmap_is_found_through_the_map_base_field() {
        // A bitmap starting at 0x100 instead of 0x68: bits at the default place must not matter.
        let denied_elsewhere = |h: &mut Harness| {
            install_io_bitmap(h, 0x100, 0x100 + 0x2000, &[]);
            let wrong_place = TSS_BASE + 0x68 + (SPARE_PORT as u64 >> 3);
            h.write(wrong_place, &[0xff]);
        };
        let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, denied_elsewhere);
        assert_eq!(stop, Stop::Limit);
        let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, 0x100, 0x100 + 0x2000, &[SPARE_PORT]);
        });
        assert_eq!(stop, gp(0));
    }

    #[test]
    fn io_wide_accesses_check_one_bit_per_byte_of_the_access() {
        // A dword access at SPARE_PORT covers ports +0..=+3.
        for offset in 0..4u16 {
            let setup = |h: &mut Harness| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[SPARE_PORT + offset]);
            };
            let (_, stop) = run_io(code::IN_EAX_DX, 1, 0, SPARE_PORT, setup);
            assert_eq!(stop, gp(0), "dword in with port +{offset} denied");
            // A word covers +0..=+1, a byte only +0.
            let (_, stop) = run_io(code::IN_AX_DX, 1, 0, SPARE_PORT, setup);
            assert_eq!(stop == gp(0), offset < 2, "word in, port +{offset}");
            let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, setup);
            assert_eq!(stop == gp(0), offset == 0, "byte in, port +{offset}");
        }
        // The port just past the access does not matter.
        let (_, stop) = run_io(code::OUT_DX_EAX, 1, 0, SPARE_PORT, |h| {
            install_io_bitmap(
                h,
                MAP_BASE,
                FULL_TSS_LIMIT,
                &[SPARE_PORT + 4, SPARE_PORT - 1],
            );
        });
        assert_eq!(stop, Stop::Limit);
    }

    #[test]
    fn io_access_straddling_a_bitmap_byte_checks_the_next_byte() {
        // Ports 0x1236..=0x1239 span bits 6,7 of one bitmap byte and bits 0,1 of the next.
        let port = 0x1236u16;
        for denied in port..port + 4 {
            let (_, stop) = run_io(code::IN_EAX_DX, 1, 0, port, |h| {
                install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[denied]);
            });
            assert_eq!(stop, gp(0), "port {denied:#x} denied");
        }
        let (_, stop) = run_io(code::IN_EAX_DX, 1, 0, port, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[port - 1, port + 4]);
        });
        assert_eq!(stop, Stop::Limit);
    }

    #[test]
    fn io_bitmap_must_lie_inside_the_tss_limit() {
        let byte_offset = MAP_BASE as u32 + (SPARE_PORT as u32 >> 3);
        // The check reads two bytes at byte_offset, so the limit must reach byte_offset + 1.
        let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, MAP_BASE, byte_offset + 1, &[]);
        });
        assert_eq!(stop, Stop::Limit, "limit covers both bytes");
        let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, MAP_BASE, byte_offset, &[]);
        });
        assert_eq!(stop, gp(0), "limit one byte short");
        // The map base itself may point past the limit.
        let (_, stop) = run_io(code::IN_AL_DX, 1, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, 0xffff, FULL_TSS_LIMIT, &[]);
        });
        assert_eq!(stop, gp(0), "map base beyond the limit");
    }

    #[test]
    fn io_with_an_immediate_port_checks_the_bitmap_too() {
        let (_, stop) = run_io(code::IN_AL_IMM, 1, 0, 0, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[0x80]);
        });
        assert_eq!(stop, gp(0));
        let (_, stop) = run_io(code::OUT_IMM_AL, 1, 0, 0, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[0x80]);
        });
        assert_eq!(stop, gp(0));
        let (_, stop) = run_io(code::IN_EAX_IMM, 1, 0, 0, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[0x83]);
        });
        assert_eq!(stop, gp(0), "dword in at 0x80 covers port 0x83");
        let (_, stop) = run_io(code::OUT_IMM_EAX, 1, 0, 0, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[0x84]);
        });
        assert_eq!(stop, Stop::Limit);
    }

    #[test]
    fn io_at_cpl3_follows_the_same_rule() {
        let (_, stop) = run_io(code::IN_AL_DX, 3, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[SPARE_PORT]);
        });
        assert_eq!(stop, gp(0));
        let (_, stop) = run_io(code::IN_AL_DX, 3, 0, SPARE_PORT, |h| {
            install_io_bitmap(h, MAP_BASE, FULL_TSS_LIMIT, &[]);
        });
        assert_eq!(stop, Stop::Limit);
        let (_, stop) = run_io(code::IN_AL_DX, 3, 3, SPARE_PORT, |_| {});
        assert_eq!(stop, Stop::Limit, "IOPL 3 allows CPL 3 without a bitmap");
    }

    #[test]
    fn out_reaches_the_device_and_in_reads_it_back() {
        // COM1's scratch register (port 0x3ff) stores a byte.
        let mut pair = Pair::new(code::OUT_IN_SCRATCH);
        pair.halted(|h| {
            h.cpu.gpr[RDX] = 0x3ff;
            h.cpu.gpr[RAX] = 0x5a;
        });
        assert_eq!(pair.reg(RAX) & 0xff, 0x5a);
    }

    #[test]
    fn in_and_out_widths_write_the_right_part_of_eax() {
        // `in ax, dx` merges into ax; `in eax, dx` zero-extends.
        let (pair, _) = run_io(code::IN_AX_DX, 0, 0, SPARE_PORT, |_| {});
        assert_eq!(pair.reg(RAX), 0x1111_2222_3333_ffff);
        let (pair, _) = run_io(code::IN_EAX_DX, 0, 0, SPARE_PORT, |_| {});
        assert_eq!(pair.reg(RAX), 0xffff_ffff);
        let (pair, _) = run_io(code::IN_AL_DX, 0, 0, SPARE_PORT, |_| {});
        assert_eq!(pair.reg(RAX), 0x1111_2222_3333_44ff);
    }

    #[test]
    fn ins_and_outs_move_bytes_between_a_port_and_memory() {
        // insb; outsb; hlt: the spare port reads as all ones.
        let mut h = Harness::new(&[0x6c, 0x6e, HLT]);
        h.cpu.gpr[RDX] = SPARE_PORT as u64;
        h.cpu.gpr[RDI] = SCRATCH;
        h.cpu.gpr[RSI] = SCRATCH + 0x10;
        h.write(SCRATCH + 0x10, &[0x5a]);
        h.run_to_halt();
        assert_eq!(h.read(SCRATCH, 1), [0xff]);
        assert_eq!(h.cpu.gpr[RDI], SCRATCH + 1);
        assert_eq!(h.cpu.gpr[RSI], SCRATCH + 0x11);
    }

    #[test]
    fn fs_and_gs_overrides_add_the_segment_base() {
        let mut pair = Pair::new(code::FS_LOAD);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = DATA + 0x100;
            h.cpu.gpr[RBX] = 0x20;
            h.write_u64(DATA + 0x120, 0x1111_2222_3333_4444);
        });
        assert_eq!(pair.reg(RAX), 0x1111_2222_3333_4444);
        let mut pair = Pair::new(code::GS_LOAD_DISP);
        pair.halted(|h| {
            h.cpu.seg[SEG_GS].base = DATA + 0x200;
            h.write_u64(DATA + 0x210, 0x5555_6666_7777_8888);
        });
        assert_eq!(pair.reg(RAX), 0x5555_6666_7777_8888);
        let mut pair = Pair::new(code::FS_LOAD_SIB);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = DATA;
            h.cpu.gpr[RBX] = 0x100;
            h.cpu.gpr[RCX] = 2;
            h.write_u64(DATA + 0x100 + 16 + 0x20, 0x9999);
        });
        assert_eq!(pair.reg(RAX), 0x9999);
    }

    #[test]
    fn fs_override_with_rip_relative_addressing_adds_the_base_to_the_rip_relative_address() {
        // The linear address is base + rip-relative address; choose the base so it lands on low
        // memory, where the test can place the data.
        let mut pair = Pair::new(code::GS_LOAD_RIP);
        pair.halted(|h| {
            h.cpu.seg[SEG_GS].base = 0u64.wrapping_sub(CODE_BASE);
            h.write_u64(8 + 0x100, 0xabcd);
        });
        assert_eq!(pair.reg(RAX), 0xabcd, "[gs.base + rip + 8 + 0x100]");
    }

    #[test]
    fn fs_override_with_a_32_bit_address_wraps_before_the_base_is_added() {
        let mut pair = Pair::new(code::FS_LOAD_ADDR32);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = DATA;
            h.cpu.gpr[RBX] = 0x1_0000_0010;
            h.write(DATA + 0x10, &[0x11, 0x22, 0x33, 0x44]);
        });
        assert_eq!(
            pair.reg(RAX),
            0x4433_2211,
            "ebx = 0x10, the upper half is dropped"
        );
    }

    #[test]
    fn fs_override_store_and_es_cs_ss_ds_overrides_without_a_base() {
        let mut pair = Pair::new(code::FS_STORE);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = DATA;
            h.cpu.gpr[RBX] = 8;
            h.cpu.gpr[RAX] = 0xdead_beef;
        });
        assert_eq!(pair.interp.read_u64(DATA + 8), 0xdead_beef);
        // In 64-bit mode only FS and GS have a base: the others are ignored even if one is cached.
        for (snippet, register) in [
            (code::ES_LOAD, SEG_ES),
            (code::CS_LOAD, SEG_CS),
            (code::SS_LOAD, SEG_SS),
            (code::DS_LOAD, SEG_DS),
        ] {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                h.cpu.seg[register].base = 0x4_0000;
                h.cpu.gpr[RBX] = DATA;
                h.write_u64(DATA, 0x1234_5678);
            });
            assert_eq!(pair.reg(RAX), 0x1234_5678, "segment {register}");
        }
    }

    #[test]
    fn segment_base_plus_offset_beyond_the_canonical_range_is_a_general_protection_fault() {
        let mut pair = Pair::new(code::FS_LOAD);
        let stop = pair.run(|h| {
            h.cpu.seg[SEG_FS].base = NON_CANONICAL;
            h.cpu.gpr[RBX] = 0x10;
        });
        assert_eq!(stop, gp(0));
    }

    #[test]
    fn non_canonical_data_accesses_raise_general_protection_not_page_fault() {
        for snippet in [
            code::LOAD_RBX,
            code::STORE_RBX,
            code::ADD_MEM_RBX,
            code::LOAD64_RBX_DISP,
            code::STORE16_RBX,
        ] {
            let (pair, stop) = run_regs_stop(snippet, &[(RBX, NON_CANONICAL), (RAX, 0x55)]);
            assert_eq!(stop, gp(0), "{snippet:02x?}");
            assert_eq!(pair.reg(RAX), 0x55);
            assert_eq!(pair.cpu().rip, CODE_BASE);
        }
        // The last canonical addresses of each half are fine to compute with: lea never faults.
        for address in [
            NON_CANONICAL,
            u64::MAX,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
        ] {
            let (_, stop) = run_regs_stop(code::LEA_RBX, &[(RBX, address)]);
            assert_eq!(stop, Stop::Halted, "lea of {address:#x}");
        }
        // A push of a memory operand at a non-canonical address is a data access through DS.
        let (_, stop) = run_regs_stop(code::PUSH_MEM, &[(RDX, NON_CANONICAL)]);
        assert_eq!(stop, gp(0));
    }

    #[test]
    fn page_faults_report_the_address_and_whether_it_was_a_read_or_a_write() {
        let unmapped = |h: &mut Harness| {
            unmap_huge_page(h, HOLE);
            h.cpu.gpr[RBX] = HOLE + 0x123;
            h.cpu.gpr[RAX] = 0x4242;
        };
        let mut pair = Pair::new(code::LOAD_RBX);
        let stop = pair.run(unmapped);
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(pair.cpu().cr2, HOLE + 0x123);
        assert_eq!(
            pair.reg(RAX),
            0x4242,
            "the destination register is untouched"
        );
        assert_eq!(pair.cpu().rip, CODE_BASE);
        let mut pair = Pair::new(code::STORE_RBX);
        let stop = pair.run(unmapped);
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)));
        assert_eq!(pair.cpu().cr2, HOLE + 0x123);
        // A read-modify-write faults at its load, which reports a read (hardware may report a
        // write here; left unasserted).
        let mut pair = Pair::new(code::ADD_MEM_RBX);
        let stop = pair.run(unmapped);
        assert!(matches!(
            stop,
            Stop::Fault {
                vector: VECTOR_PAGE_FAULT,
                ..
            }
        ));
        assert_eq!(pair.cpu().cr2, HOLE + 0x123);
    }

    #[test]
    fn page_faults_from_user_mode_set_the_user_bit() {
        const PF_PROTECTION: u32 = 1;
        const PF_WRITE: u32 = 2;
        const PF_USER: u32 = 4;
        let user_setup = |not_present: bool| {
            move |h: &mut Harness| {
                enter_cpl(h, 3);
                let entry = PAGE_TABLE_FIRST_PD + (HOLE / HUGE_PAGE_BYTES) * 8;
                let value = h.read_u64(entry);
                // Present but supervisor-only, or not present at all.
                h.write_u64(entry, if not_present { 0 } else { value & !PTE_USER });
                h.cpu.flush_tlb();
                h.cpu.gpr[RBX] = HOLE + 0x40;
            }
        };
        let mut pair = Pair::new(code::LOAD_RBX);
        assert_eq!(
            pair.run(user_setup(true)),
            fault(VECTOR_PAGE_FAULT, Some(PF_USER))
        );
        let mut pair = Pair::new(code::STORE_RBX);
        assert_eq!(
            pair.run(user_setup(true)),
            fault(VECTOR_PAGE_FAULT, Some(PF_USER | PF_WRITE))
        );
        let mut pair = Pair::new(code::LOAD_RBX);
        assert_eq!(
            pair.run(user_setup(false)),
            fault(VECTOR_PAGE_FAULT, Some(PF_USER | PF_PROTECTION))
        );
        let mut pair = Pair::new(code::STORE_RBX);
        assert_eq!(
            pair.run(user_setup(false)),
            fault(VECTOR_PAGE_FAULT, Some(PF_USER | PF_PROTECTION | PF_WRITE))
        );
    }

    #[test]
    fn a_store_crossing_into_a_missing_page_writes_nothing() {
        // The first four bytes are on a mapped page, the last four on a missing one.
        let mut pair = Pair::new(code::MOV_MEM_RAX);
        let stop = pair.run(|h| {
            unmap_huge_page(h, HOLE);
            h.write(HOLE - 8, &[0xee; 8]);
            h.cpu.gpr[RDX] = HOLE - 4;
            h.cpu.gpr[RAX] = 0x1111_2222_3333_4444;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)));
        assert_eq!(
            pair.cpu().cr2,
            HOLE,
            "cr2 is the first byte on the faulting page"
        );
        assert_eq!(pair.interp.read(HOLE - 8, 8), [0xee; 8], "no partial store");
    }

    #[test]
    fn a_load_crossing_into_a_missing_page_faults_with_the_second_pages_address() {
        let mut pair = Pair::new(code::MOV_RAX_MEM);
        let stop = pair.run(|h| {
            unmap_huge_page(h, HOLE);
            h.cpu.gpr[RDX] = HOLE - 4;
            h.cpu.gpr[RAX] = 7;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(pair.cpu().cr2, HOLE);
        assert_eq!(pair.reg(RAX), 7);
    }

    #[test]
    fn unaligned_accesses_work_across_page_boundaries() {
        let mut pair = Pair::new(code::MOV_RAX_MEM);
        pair.halted(|h| {
            h.write(SCRATCH + 0xffc, &[1, 2, 3, 4, 5, 6, 7, 8]);
            h.cpu.gpr[RDX] = SCRATCH + 0xffc;
        });
        assert_eq!(pair.reg(RAX), 0x0807_0605_0403_0201);
        let mut pair = Pair::new(code::MOV_MEM_RAX);
        pair.halted(|h| {
            h.write(SCRATCH + 0xffc, &[0; 8]);
            h.cpu.gpr[RDX] = SCRATCH + 0xffc;
            h.cpu.gpr[RAX] = 0x0807_0605_0403_0201;
        });
        assert_eq!(
            pair.interp.read(SCRATCH + 0xffc, 8),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn string_instruction_faults_leave_the_indexes_at_the_faulting_element() {
        // movsb: the load succeeds, the store faults; nothing is committed.
        let mut pair = Pair::new(code::MOVSB_PLAIN);
        let stop = pair.run(|h| {
            unmap_huge_page(h, HOLE);
            h.cpu.gpr[RSI] = DATA;
            h.cpu.gpr[RDI] = HOLE + 8;
            h.cpu.gpr[RCX] = 3;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)));
        assert_eq!(
            (pair.reg(RSI), pair.reg(RDI), pair.reg(RCX)),
            (DATA, HOLE + 8, 3)
        );
        // rep stosb across the edge of a page: the elements before the fault are stored and counted.
        let mut pair = Pair::new(code::REP_STOSB_PLAIN);
        let stop = pair.run(|h| {
            unmap_huge_page(h, HOLE);
            h.write(HOLE - 8, &[0; 8]);
            h.cpu.gpr[RDI] = HOLE - 3;
            h.cpu.gpr[RCX] = 10;
            h.cpu.gpr[RAX] = 0x5a;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_WRITE_NOT_PRESENT)));
        assert_eq!(pair.reg(RDI), HOLE, "three elements were stored");
        assert_eq!(pair.reg(RCX), 7);
        assert_eq!(pair.cpu().cr2, HOLE);
        assert_eq!(
            pair.cpu().rip,
            CODE_BASE,
            "restartable at the same instruction"
        );
        assert_eq!(
            pair.interp.read(HOLE - 8, 8),
            [0, 0, 0, 0, 0, 0x5a, 0x5a, 0x5a]
        );
        // cmpsb, lodsb and scasb fault on their loads without touching registers or flags.
        for (snippet, rsi, rdi) in [
            (code::CMPSB_PLAIN, DATA, HOLE + 0x10),
            (code::CMPSB_PLAIN, HOLE + 0x10, DATA),
            (code::LODSB_PLAIN, HOLE + 0x10, DATA),
            (code::SCASB_PLAIN, DATA, HOLE + 0x10),
            (code::REPE_CMPSB_PLAIN, DATA, HOLE + 0x10),
        ] {
            assert_fault_changes_nothing(
                snippet,
                |h| {
                    unmap_huge_page(h, HOLE);
                    h.cpu.gpr[RSI] = rsi;
                    h.cpu.gpr[RDI] = rdi;
                    h.cpu.gpr[RCX] = 4;
                    h.cpu.gpr[RAX] = 0x11;
                    h.cpu.set_eflags_arith(CF | ZF);
                },
                fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)),
            );
        }
        // rep movsb reading from a missing page after two elements.
        let mut pair = Pair::new(code::REP_MOVSB_PLAIN);
        let stop = pair.run(|h| {
            unmap_huge_page(h, HOLE + HUGE_PAGE_BYTES);
            h.write(HOLE + HUGE_PAGE_BYTES - 2, &[0x11, 0x22]);
            h.cpu.gpr[RSI] = HOLE + HUGE_PAGE_BYTES - 2;
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RCX] = 5;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(
            (pair.reg(RSI), pair.reg(RDI), pair.reg(RCX)),
            (HOLE + HUGE_PAGE_BYTES, DATA + 2, 3)
        );
    }

    #[test]
    fn overlapping_forward_rep_movsb_replicates_the_first_byte() {
        // The classic fill idiom: destination one byte above the source.
        let mut pair = Pair::new(code::REP_MOVSB_PLAIN);
        pair.halted(|h| {
            h.write(DATA, &[0xaa, 1, 2, 3, 4, 5, 6, 7]);
            h.cpu.gpr[RSI] = DATA;
            h.cpu.gpr[RDI] = DATA + 1;
            h.cpu.gpr[RCX] = 6;
        });
        assert_eq!(
            pair.interp.read(DATA, 8),
            [0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 7]
        );
        // Backwards (DF = 1) the same overlap copies instead of replicating.
        let mut pair = Pair::new(code::REP_MOVSB_PLAIN);
        pair.halted(|h| {
            h.write(DATA, &[1, 2, 3, 4, 5, 6, 7, 8]);
            h.cpu.gpr[RSI] = DATA + 5;
            h.cpu.gpr[RDI] = DATA + 6;
            h.cpu.gpr[RCX] = 6;
            h.cpu.eflags |= DF;
        });
        assert_eq!(pair.interp.read(DATA, 8), [1, 1, 2, 3, 4, 5, 6, 8]);
    }

    /// The bulk path of `rep stos`/`rep movs` works a page at a time, so the interesting starts are
    /// unaligned ones that cross several pages and elements that straddle a page boundary.
    #[test]
    fn rep_stos_of_every_size_fills_across_page_boundaries_and_straddling_elements() {
        // A fill value of distinct bytes, and one that repeats a byte (the memset case).
        for fill in [0x1122_3344_5566_7788u64, 0x5a5a_5a5a_5a5a_5a5a] {
            rep_stos_fills_with(fill);
        }
    }

    fn rep_stos_fills_with(fill: u64) {
        for (snippet, size) in [
            (code::REP_STOS_1, 1usize),
            (code::REP_STOS_2, 2),
            (code::REP_STOS_4, 4),
            (code::REP_STOS_8, 8),
        ] {
            // Starts 3 bytes before a page end, so for sizes above 1 the first element straddles.
            let start = DATA + 0x1000 - 3;
            let elements = 0x1800 / size as u64;
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                h.write(DATA, &vec![0u8; 0x4000]);
                h.cpu.gpr[RDI] = start;
                h.cpu.gpr[RCX] = elements;
                h.cpu.gpr[RAX] = fill;
            });
            let bytes = elements as usize * size;
            let written = pair.interp.read(start, bytes + 4);
            let pattern = fill.to_le_bytes();
            for (k, byte) in written[..bytes].iter().enumerate() {
                assert_eq!(*byte, pattern[k % size], "size {size} byte {k}");
            }
            assert!(written[bytes..].iter().all(|&b| b == 0), "size {size}");
            assert_eq!(pair.reg(RDI), start + bytes as u64);
            assert_eq!(pair.reg(RCX), 0);
        }
    }

    #[test]
    fn rep_movs_copies_forward_across_pages_without_touching_the_source() {
        let source: Vec<u8> = (0..0x2800u32).map(|k| (k * 7 + 1) as u8).collect();
        let mut pair = Pair::new(code::REP_MOVS_8);
        pair.halted(|h| {
            h.write(DATA, &vec![0u8; 0x6000]);
            h.write(DATA + 0x100, &source);
            h.cpu.gpr[RSI] = DATA + 0x100;
            h.cpu.gpr[RDI] = DATA + 0x3ff8;
            h.cpu.gpr[RCX] = 0x2800 / 8;
        });
        assert_eq!(pair.interp.read(DATA + 0x3ff8, 0x2800), source);
        assert_eq!(pair.interp.read(DATA + 0x100, 0x2800), source);
        assert_eq!(pair.reg(RSI), DATA + 0x100 + 0x2800);
        assert_eq!(pair.reg(RDI), DATA + 0x3ff8 + 0x2800);
        assert_eq!(pair.reg(RCX), 0);
    }

    #[test]
    fn rep_instructions_resume_correctly_across_the_per_slice_iteration_limit() {
        // More iterations than one slice (4096): the run loop re-enters the same instruction.
        let mut pair = Pair::new(code::REP_STOSB_PLAIN);
        pair.halted(|h| {
            h.write(DATA, &vec![0u8; 0x5000]);
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RCX] = 0x4800;
            h.cpu.gpr[RAX] = 0x6b;
        });
        assert_eq!(pair.reg(RCX), 0);
        assert_eq!(pair.reg(RDI), DATA + 0x4800);
        let written = pair.interp.read(DATA, 0x5000);
        assert!(written[..0x4800].iter().all(|&b| b == 0x6b));
        assert!(written[0x4800..].iter().all(|&b| b == 0));
    }

    #[test]
    fn plain_string_instructions_ignore_rcx_and_do_one_element() {
        let mut pair = Pair::new(code::STOSB_PLAIN);
        pair.halted(|h| {
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RCX] = 100;
            h.cpu.gpr[RAX] = 0x77;
        });
        assert_eq!(pair.reg(RCX), 100);
        assert_eq!(pair.reg(RDI), DATA + 1);
    }

    #[test]
    fn test_group_3_slash_1_is_an_alias_of_test() {
        let pair = run_regs(code::TEST_ALIAS_F6_1, &[(RAX, 0xf0)]);
        assert_flags(pair.flags(), ZF | CF | OF, ZF, "test al, 0x0f via f6 /1");
        let pair = run_regs(code::TEST_ALIAS_F7_1, &[(RAX, 0x0f)]);
        assert_flags(pair.flags(), ZF | CF | OF, 0, "test eax, 0x0f via f7 /1");
    }

    fn decoded(bytes: &[u8]) -> Insn {
        super::super::decode::decode(bytes).expect("snippet decodes")
    }

    #[test]
    fn sign_extend_replicates_the_top_bit_of_the_given_width() {
        assert_eq!(sign_extend(0x80, 8), -128);
        assert_eq!(sign_extend(0x7f, 8), 127);
        assert_eq!(sign_extend(0xffff, 16), -1);
        assert_eq!(sign_extend(0x8000, 16), -32768);
        assert_eq!(sign_extend(0x8000_0000, 32), -(1 << 31));
        assert_eq!(sign_extend(u64::MAX, 64), -1);
        assert_eq!(
            sign_extend(0x1_0000_0001, 32),
            1,
            "bits above the width are ignored"
        );
    }

    #[test]
    fn selector_helpers_ignore_the_rpl_but_not_the_table_indicator() {
        for sel in [0u16, 1, 2, 3] {
            assert!(selector_is_null(sel), "{sel:#x}");
        }
        for sel in [4u16, 5, 8, 0x1b, 0xffff] {
            assert!(!selector_is_null(sel), "{sel:#x}: TI or index set");
        }
        assert_eq!(selector_error_code(0x1b), 0x18);
        assert_eq!(selector_error_code(0xffff), 0xfffc);
        assert_eq!(selector_error_code(0x2), 0);
    }

    #[test]
    fn operand_size_helpers_follow_the_prefixes() {
        assert_eq!(stack_operand_size(&decoded(&[0x50])), 8);
        assert_eq!(stack_operand_size(&decoded(&[0x66, 0x50])), 2);
        assert_eq!(
            stack_operand_size(&decoded(&[0x48, 0x50])),
            8,
            "REX.W is irrelevant to push"
        );
        assert_eq!(address_mask(&decoded(&[0x90])), u64::MAX);
        assert_eq!(address_mask(&decoded(&[0x67, 0x90])), 0xffff_ffff);
        assert_eq!(shift_count_mask(8), 63);
        for size in [1u8, 2, 4] {
            assert_eq!(shift_count_mask(size), 31);
        }
        assert_eq!(
            combine_dwords(0xffff_ffff_1234_5678, 0xaabb_ccdd),
            0xaabb_ccdd_1234_5678
        );
        assert_eq!(
            combine_dwords(0, 0xffff_ffff_aabb_ccdd),
            0xaabb_ccdd_0000_0000,
            "upper bits of both halves are dropped"
        );
    }

    #[test]
    fn opcode_register_combines_the_low_opcode_bits_with_rex_b() {
        assert_eq!(opcode_register(&decoded(&[0x0f, 0xc8])), 0);
        assert_eq!(opcode_register(&decoded(&[0x0f, 0xcf])), 7);
        assert_eq!(opcode_register(&decoded(&[0x41, 0x0f, 0xc9])), 9);
        assert_eq!(opcode_register(&decoded(&[0x49, 0x57])), 15);
        assert_eq!(opcode_register(&decoded(&[0x50])), 0);
    }

    #[test]
    fn conforming_code_detection_needs_the_code_type() {
        assert!(!is_conforming_code(&Seg::default()));
        let conforming = Seg {
            flags: SEG_FLAGS_CODE64 | SEG_TYPE_CONFORMING,
            ..Seg::default()
        };
        assert!(is_conforming_code(&conforming));
        let nonconforming = Seg {
            flags: SEG_FLAGS_CODE64,
            ..Seg::default()
        };
        assert!(!is_conforming_code(&nonconforming));
        // Bit 10 of a data segment is "expand down", not "conforming".
        let expand_down = Seg {
            flags: SEG_FLAGS_FLAT_DATA | SEG_TYPE_CONFORMING,
            ..Seg::default()
        };
        assert!(!is_conforming_code(&expand_down));
    }

    #[test]
    fn loadable_eflags_depends_on_cpl_and_iopl() {
        let mut pair = Pair::interpreter_only(code::NOP);
        let cpu = &mut pair.interp.cpu;
        let always = EFLAGS_ALWAYS_LOADABLE;
        assert_eq!(always & (IF | IOPL | RF | VM), 0);
        cpu.eflags = RESERVED_ONE;
        assert_eq!(cpu.loadable_eflags(0), always | IF | IOPL);
        assert_eq!(
            cpu.loadable_eflags(1),
            always,
            "IOPL 0: CPL 1 may not touch IF"
        );
        cpu.eflags = RESERVED_ONE | IOPL_ONE;
        assert_eq!(cpu.iopl(), 1);
        assert_eq!(cpu.loadable_eflags(1), always | IF);
        assert_eq!(cpu.loadable_eflags(2), always);
        cpu.eflags = RESERVED_ONE | IOPL_THREE;
        assert_eq!(cpu.iopl(), 3);
        assert_eq!(cpu.loadable_eflags(3), always | IF);
        assert_eq!(cpu.loadable_eflags(0), always | IF | IOPL);
    }

    #[test]
    fn double_shift_helper_matches_the_two_operand_concatenation_for_wide_sizes() {
        // shld: (a:b) << count, high half; shrd: (b:a) >> count, low half.
        assert_eq!(
            double_shift(true, 0x1234_5678, 0x9abc_def0, 8, 4),
            (0x3456_789a, 0)
        );
        assert_eq!(
            double_shift(false, 0x1234_5678, 0x9abc_def0, 8, 4),
            (0xf012_3456, 0)
        );
        assert_eq!(double_shift(true, 0x8000_0000_0000_0000, 0, 1, 8), (0, 1));
        assert_eq!(double_shift(false, 1, 0, 1, 8), (0, 1));
        assert_eq!(
            double_shift(true, 0x80, 0x80, 1, 1),
            (0x01, 1),
            "byte sizes follow the same formula"
        );
    }

    #[test]
    fn general_registers_read_and_write_with_the_architectural_merge_rules() {
        let mut pair = Pair::interpreter_only(code::NOP);
        let cpu = &mut pair.interp.cpu;
        cpu.gpr[RAX] = 0x1122_3344_5566_7788;
        cpu.gpr[RBX] = 0xaabb_ccdd_eeff_0011;
        cpu.gpr[RSI] = 0x0102_0304_0506_0708;
        cpu.gpr[8] = 0x1111_2222_3333_4444;
        assert_eq!(cpu.read_gpr(RAX as u8, 8, false), 0x1122_3344_5566_7788);
        assert_eq!(cpu.read_gpr(RAX as u8, 4, false), 0x5566_7788);
        assert_eq!(cpu.read_gpr(RAX as u8, 2, false), 0x7788);
        assert_eq!(cpu.read_gpr(RAX as u8, 1, false), 0x88);
        // Register numbers 4..7 are AH, CH, DH, BH without REX and SPL, BPL, SIL, DIL with it.
        assert_eq!(cpu.read_gpr(4, 1, false), 0x77, "ah");
        assert_eq!(cpu.read_gpr(7, 1, false), 0x00, "bh");
        assert_eq!(cpu.read_gpr(7, 1, true), cpu.gpr[7] & 0xff, "dil");
        assert_eq!(cpu.read_gpr(6, 1, true), 0x08, "sil");
        assert_eq!(
            cpu.read_gpr(8, 1, false),
            0x44,
            "r8b needs no REX bit to be addressed as r8"
        );
        // Writes.
        cpu.write_gpr(RAX as u8, 1, false, 0xff);
        assert_eq!(cpu.gpr[RAX], 0x1122_3344_5566_77ff);
        cpu.write_gpr(4, 1, false, 0x99);
        assert_eq!(cpu.gpr[RAX], 0x1122_3344_5566_99ff, "ah");
        cpu.write_gpr(RAX as u8, 2, false, 0xabcd);
        assert_eq!(cpu.gpr[RAX], 0x1122_3344_5566_abcd);
        cpu.write_gpr(RAX as u8, 4, false, 0xdead_beef_1234_5678);
        assert_eq!(cpu.gpr[RAX], 0x1234_5678, "32-bit writes zero-extend");
        cpu.write_gpr(RAX as u8, 8, false, 0x2345_6789_abcd_ef01);
        assert_eq!(cpu.gpr[RAX], 0x2345_6789_abcd_ef01);
        cpu.write_gpr(6, 1, true, 0x55);
        assert_eq!(cpu.gpr[RSI], 0x0102_0304_0506_0755, "sil with REX");
        cpu.write_gpr(7, 1, false, 0x66);
        assert_eq!(cpu.gpr[RBX], 0xaabb_ccdd_eeff_6611, "bh without REX");
        cpu.write_gpr(8, 1, false, 0x77);
        assert_eq!(cpu.gpr[8], 0x1111_2222_3333_4477);
    }

    #[test]
    fn effective_address_covers_the_addressing_modes() {
        let mut pair = Pair::interpreter_only(code::NOP);
        let cpu = &mut pair.interp.cpu;
        cpu.gpr[RBX] = 0x1000;
        cpu.gpr[RCX] = 3;
        cpu.gpr[RSP] = 0x2000;
        cpu.gpr[12] = 0x3000;
        cpu.gpr[13] = 5;
        cpu.seg[SEG_FS].base = 0x1_0000;
        let lea = |bytes: &[u8]| decoded(bytes);
        // lea rax, [rbx+rcx*4+8]
        let i = lea(&[0x48, 0x8d, 0x44, 0x8b, 0x08]);
        assert_eq!(cpu.effective_address(&i, 0), 0x1014);
        // lea rax, [rbx-1]
        let i = lea(&[0x48, 0x8d, 0x43, 0xff]);
        assert_eq!(cpu.effective_address(&i, 0), 0xfff);
        // lea rax, [rsp+8] (SIB with no index)
        let i = lea(&[0x48, 0x8d, 0x44, 0x24, 0x08]);
        assert_eq!(cpu.effective_address(&i, 0), 0x2008);
        // lea rax, [r12+r13*2] (REX.X and REX.B extend index and base)
        let i = lea(&[0x4b, 0x8d, 0x04, 0x6c]);
        assert_eq!(cpu.effective_address(&i, 0), 0x300a);
        // lea rax, [rip+0x10]: relative to the next instruction, passed in
        let i = lea(&[0x48, 0x8d, 0x05, 0x10, 0x00, 0x00, 0x00]);
        assert_eq!(cpu.effective_address(&i, 0x5000), 0x5010);
        // lea eax, [ebx+ecx*4+8] with the 67 prefix wraps at 32 bits
        let i = lea(&[0x67, 0x8d, 0x44, 0x8b, 0x08]);
        cpu.gpr[RBX] = 0xffff_fff0;
        assert_eq!(
            cpu.effective_address(&i, 0),
            0x4,
            "0xfffffff0 + 12 + 8 wraps past 2^32"
        );
        // mov rax, fs:[rbx]: the effective address excludes the base, the linear address adds it
        cpu.gpr[RBX] = 0x40;
        let i = lea(&[0x64, 0x48, 0x8b, 0x03]);
        assert_eq!(cpu.effective_address(&i, 0), 0x40);
        assert_eq!(cpu.linear_address(&i, 0x40).unwrap(), 0x1_0040);
        // Without a segment prefix, or with CS/DS/ES/SS, nothing is added.
        let i = lea(&[0x48, 0x8b, 0x03]);
        assert_eq!(cpu.linear_address(&i, 0x40).unwrap(), 0x40);
        cpu.seg[SEG_DS].base = 0x9999;
        let i = lea(&[0x3e, 0x48, 0x8b, 0x03]);
        assert_eq!(cpu.linear_address(&i, 0x40).unwrap(), 0x40);
    }

    #[test]
    fn modrm_location_distinguishes_registers_from_memory() {
        let mut pair = Pair::interpreter_only(code::NOP);
        let cpu = &mut pair.interp.cpu;
        cpu.gpr[RBX] = 0x1000;
        let i = decoded(&[0x48, 0x89, 0xd8]); // mov rax, rbx: reg-reg
        assert!(matches!(cpu.modrm_location(&i, 0).unwrap(), Loc::Reg(0)));
        let i = decoded(&[0x48, 0x89, 0x03]); // mov [rbx], rax
        assert!(matches!(
            cpu.modrm_location(&i, 0).unwrap(),
            Loc::Mem(0x1000)
        ));
        let i = decoded(&[0x4c, 0x89, 0xc8]); // mov rax, r9: REX.R is the reg field, rm stays rax
        assert!(matches!(cpu.modrm_location(&i, 0).unwrap(), Loc::Reg(0)));
        let i = decoded(&[0x49, 0x89, 0xc1]); // mov r9, rax: REX.B extends rm
        assert!(matches!(cpu.modrm_location(&i, 0).unwrap(), Loc::Reg(9)));
    }

    #[test]
    fn push_and_pop_helpers_move_rsp_by_the_operand_size() {
        let mut pair = Pair::interpreter_only(code::NOP);
        let cpu = &mut pair.interp.cpu;
        cpu.gpr[RSP] = STACK_TOP;
        cpu.push(8, 0x1122_3344_5566_7788).unwrap();
        cpu.push(2, 0xaabb).unwrap();
        assert_eq!(cpu.gpr[RSP], STACK_TOP - 10);
        assert_eq!(cpu.pop(2).unwrap(), 0xaabb);
        assert_eq!(cpu.pop(8).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(cpu.gpr[RSP], STACK_TOP);
        // A failed push leaves rsp alone.
        cpu.gpr[RSP] = NON_CANONICAL + 8;
        assert!(cpu.push(8, 1).is_err());
        assert_eq!(cpu.gpr[RSP], NON_CANONICAL + 8);
        // A failed pop leaves it too.
        cpu.gpr[RSP] = NON_CANONICAL;
        assert!(cpu.pop(8).is_err());
        assert_eq!(cpu.gpr[RSP], NON_CANONICAL);
    }

    #[test]
    fn unimplemented_opcode_is_logged_once_per_opcode_and_reg_and_raises_ud() {
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink = messages.clone();
        let mut pair = Pair::interpreter_only(code::GROUP9_REG0);
        pair.interp.cpu.log = Some(std::sync::Arc::new(move |message: &str| {
            sink.lock().unwrap().push(message.to_string());
        }));
        for _ in 0..3 {
            let stop = pair.run(|h| h.cpu.gpr[RDX] = DATA);
            assert_eq!(stop, ud());
        }
        let logged = messages.lock().unwrap().clone();
        assert_eq!(
            logged.len(),
            1,
            "the same opcode and reg field is reported once: {logged:?}"
        );
        assert!(
            logged[0].contains("0x1c7") && logged[0].contains("/0"),
            "{}",
            logged[0]
        );
        assert!(
            logged[0].contains(&format!("{CODE_BASE:#x}")),
            "the message names the rip: {}",
            logged[0]
        );
        // A different /reg of the same opcode is a different key.
        let mut other = Pair::interpreter_only(code::CMPXCHG8B_REG);
        other.interp.cpu.log = pair.interp.cpu.log.clone();
        other.interp.cpu.logged = pair.interp.cpu.logged.clone();
        let stop = other.run(|h| h.cpu.gpr[RDX] = DATA);
        assert_eq!(stop, ud());
        assert_eq!(messages.lock().unwrap().len(), 2);
        assert!(
            other
                .interp
                .cpu
                .logged
                .iter()
                .all(|key| *key >= UNIMPLEMENTED_OPCODE_LOG_BASE)
        );
        assert_eq!(other.interp.cpu.logged.len(), 2);
    }

    #[test]
    fn out_to_the_reset_port_exits_after_advancing_rip() {
        let (pair, stop) = run_regs_stop(code::OUT_KBC_RESET, &[]);
        assert_eq!(stop, Stop::Exit("Reset".to_string()));
        assert_eq!(
            pair.cpu().rip,
            CODE_BASE + 4,
            "the exit state is after the out instruction"
        );
    }

    #[test]
    fn rep_string_iterations_count_as_retired_instructions() {
        for (count, expected_icount) in [(0u64, 2u64), (10, 12)] {
            let mut pair = Pair::new(code::REP_STOSB_COUNTED);
            pair.halted(|h| {
                h.cpu.gpr[RDI] = DATA;
                h.cpu.gpr[RCX] = count;
            });
            // One step for the rep instruction, one tick per iteration, one for the hlt.
            assert_eq!(pair.cpu().icount, expected_icount, "count {count}");
        }
    }

    #[test]
    fn syscall_needs_efer_sce_and_loads_rcx_r11_and_the_kernel_segments() {
        let (_, stop) = run_regs_stop(code::SYSCALL_, &[]);
        assert_eq!(stop, ud(), "EFER.SCE is clear by default");
        let mut pair = Pair::new(code::SYSCALL_);
        let stop = pair.run_for(1, |h| {
            h.cpu.efer |= EFER_SCE;
            h.cpu.star = STAR_FOR_TESTS;
            h.cpu.lstar = CODE_BASE + 0x100;
            h.cpu.sfmask = IF;
            h.cpu.eflags = RESERVED_ONE | IF | DF;
        });
        assert_eq!(stop, Stop::Limit);
        let cpu = pair.cpu();
        assert_eq!(cpu.rip, CODE_BASE + 0x100);
        assert_eq!(pair.reg(RCX), CODE_BASE + 2);
        assert_eq!(pair.reg(11) & (IF | DF), IF | DF, "r11 keeps the old flags");
        assert_eq!(cpu.seg[SEG_CS].sel, 0x08);
        assert_eq!(cpu.seg[SEG_SS].sel, 0x10);
        assert_eq!(cpu.cpl, 0);
        assert_eq!(cpu.eflags_full() & IF, 0, "sfmask cleared IF");
        assert_eq!(cpu.eflags_full() & DF, DF);
    }

    #[test]
    fn sysretq_returns_to_ring_3_at_rcx_with_flags_from_r11() {
        let mut pair = Pair::new(code::SYSRETQ);
        let stop = pair.run_for(1, |h| {
            h.cpu.efer |= EFER_SCE;
            h.cpu.star = STAR_FOR_TESTS;
            h.cpu.gpr[RCX] = 0x40_1000;
            h.cpu.gpr[11] = RESERVED_ONE | IF | CF | IOPL_THREE | RF;
        });
        assert_eq!(stop, Stop::Limit);
        let cpu = pair.cpu();
        assert_eq!(cpu.cpl, 3);
        assert_eq!(cpu.rip, 0x40_1000);
        assert_eq!(
            cpu.seg[SEG_CS].sel,
            (0x13 + 16) | 3,
            "star[63:48] + 16, RPL 3"
        );
        assert_eq!(cpu.seg[SEG_SS].sel, (0x13 + 8) | 3);
        assert_eq!(cpu.eflags_full() & (IF | CF), IF | CF);
        assert_eq!(
            cpu.eflags_full() & (IOPL | RF),
            0,
            "sysret restores neither IOPL nor RF"
        );
        // Needs CPL 0 and the 64-bit form.
        let mut pair = Pair::new(code::SYSRETQ);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 1);
            h.cpu.efer |= EFER_SCE;
        });
        assert_eq!(stop, gp(0));
        let (_, stop) = run_regs_stop(code::SYSRET_32, &[]);
        assert_eq!(
            stop,
            ud(),
            "without SCE, and the 32-bit form is not implemented"
        );
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Unary {
        Inc,
        Dec,
        Neg,
        Not,
    }

    const UNARY_SNIPPETS: [(Unary, u8, &[u8], &[u8]); 16] = [
        (Unary::Inc, 8, code::INC_8, code::INC_MEM_8),
        (Unary::Inc, 4, code::INC_4, code::INC_MEM_4),
        (Unary::Inc, 2, code::INC_2, code::INC_MEM_2),
        (Unary::Inc, 1, code::INC_1, code::INC_MEM_1),
        (Unary::Dec, 8, code::DEC_8, code::DEC_MEM_8),
        (Unary::Dec, 4, code::DEC_4, code::DEC_MEM_4),
        (Unary::Dec, 2, code::DEC_2, code::DEC_MEM_2),
        (Unary::Dec, 1, code::DEC_1, code::DEC_MEM_1),
        (Unary::Neg, 8, code::NEG_8, code::NEG_MEM_8),
        (Unary::Neg, 4, code::NEG_4, code::NEG_MEM_4),
        (Unary::Neg, 2, code::NEG_2, code::NEG_MEM_2),
        (Unary::Neg, 1, code::NEG_1, code::NEG_MEM_1),
        (Unary::Not, 8, code::NOT_8, code::NOT_MEM_8),
        (Unary::Not, 4, code::NOT_4, code::NOT_MEM_4),
        (Unary::Not, 2, code::NOT_2, code::NOT_MEM_2),
        (Unary::Not, 1, code::NOT_1, code::NOT_MEM_1),
    ];

    /// (result, flags, mask of defined flags) of a one-operand ALU instruction, by reduction to the
    /// two-operand reference: INC/DEC keep CF, NEG is `0 - a`, NOT touches no flag.
    fn unary_reference(kind: Unary, size: u8, a: u64, flags_in: u64) -> (u64, u64, u64) {
        match kind {
            Unary::Inc | Unary::Dec => {
                let op = if kind == Unary::Inc { ALU_ADD } else { ALU_SUB };
                let r = alu_reference(op, size, a, 1, 0);
                (r.result, (r.flags & !CF) | (flags_in & CF), ALL_ARITH)
            }
            Unary::Neg => {
                let r = alu_reference(ALU_SUB, size, 0, a, 0);
                (r.result, r.flags, ALL_ARITH)
            }
            Unary::Not => (!a & mask(size), flags_in, ALL_ARITH),
        }
    }

    #[test]
    fn inc_dec_neg_not_every_size_register_and_memory() {
        for (kind, size, register_code, memory_code) in UNARY_SNIPPETS {
            let m = mask(size);
            let junk = OPERAND_JUNK & !m;
            for memory in [false, true] {
                let mut pair = Pair::new(if memory { memory_code } else { register_code });
                for a in interesting_values(size) {
                    for flags_in in [0, CF, ALL_ARITH & !CF, ALL_ARITH] {
                        pair.halted(|h| {
                            if memory {
                                memory_with_marker(h, a, size);
                            } else {
                                h.cpu.gpr[RAX] = junk | a;
                            }
                            h.cpu.set_eflags_arith(flags_in);
                        });
                        let (result, flags, defined) = unary_reference(kind, size, a, flags_in);
                        let context = format!("{kind:?} size {size} memory {memory}: {a:#x}");
                        if memory {
                            let mut want = [MEMORY_MARKER; 16];
                            want[..size as usize]
                                .copy_from_slice(&result.to_le_bytes()[..size as usize]);
                            assert_eq!(pair.interp.read(DATA, 16), want, "{context}");
                        } else {
                            let want = if size < 4 { junk | result } else { result };
                            assert_eq!(pair.reg(RAX), want, "{context}");
                        }
                        assert_flags(pair.flags(), defined, flags, &context);
                    }
                }
            }
        }
    }

    #[test]
    fn inc_and_dec_boundary_flags_by_hand() {
        // inc: 0x7f -> 0x80 sets OF and AF but CF stays as it was.
        let mut pair = Pair::new(code::INC_1);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0x7f;
            h.cpu.set_eflags_arith(CF);
        });
        assert_flags(pair.flags(), ALL_ARITH, CF | OF | AF | SF, "inc 0x7f");
        // inc: 0xff -> 0 sets ZF, AF and PF, not CF.
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xff;
            h.cpu.set_eflags_arith(0);
        });
        assert_flags(pair.flags(), ALL_ARITH, ZF | AF | PF, "inc 0xff");
        // dec: 0x80 -> 0x7f sets OF and AF, keeps CF.
        let mut pair = Pair::new(code::DEC_1);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0x80;
            h.cpu.set_eflags_arith(CF);
        });
        assert_flags(pair.flags(), ALL_ARITH, CF | OF | AF, "dec 0x80");
        // neg: 0 -> 0 clears CF; any other value sets it; the minimum value overflows.
        let mut pair = Pair::new(code::NEG_4);
        pair.halted(|h| h.cpu.gpr[RAX] = 0);
        assert_flags(pair.flags(), ALL_ARITH, ZF | PF, "neg 0");
        pair.halted(|h| h.cpu.gpr[RAX] = 0x8000_0000);
        assert_flags(pair.flags(), ALL_ARITH, CF | OF | SF | PF, "neg INT_MIN");
        assert_eq!(pair.reg(RAX), 0x8000_0000);
    }

    const WIDE_A: u64 = 0x1122_3344_5566_7788;
    const WIDE_B: u64 = 0xaabb_ccdd_eeff_9900;

    #[test]
    fn legacy_high_byte_registers_address_bits_15_to_8() {
        let regs = [(RAX, WIDE_A), (RBX, WIDE_B)];
        assert_eq!(
            run_regs(code::MOV_AH_BH, &regs).reg(RAX),
            0x1122_3344_5566_9988
        );
        assert_eq!(
            run_regs(code::MOV_AL_BH, &regs).reg(RAX),
            0x1122_3344_5566_7799
        );
        assert_eq!(
            run_regs(code::MOV_BH_AL, &regs).reg(RBX),
            0xaabb_ccdd_eeff_8800
        );
        assert_eq!(
            run_regs(code::MOV_BL_AH, &regs).reg(RBX),
            0xaabb_ccdd_eeff_9977
        );
        assert_eq!(
            run_regs(code::MOV_DH_IMM, &[(RDX, WIDE_A)]).reg(RDX),
            0x1122_3344_5566_5a88
        );
    }

    #[test]
    fn rex_prefixed_byte_registers_address_spl_bpl_sil_dil_and_r8b_to_r15b() {
        // Without REX, register 6 in a byte instruction is DH; with any REX it is SIL.
        assert_eq!(
            run_regs(code::MOV_SIL_IMM, &[(RSI, WIDE_A)]).reg(RSI),
            0x1122_3344_5566_775a
        );
        assert_eq!(
            run_regs(code::MOV_SPL_IMM, &[(RSP, WIDE_A)]).reg(RSP),
            0x1122_3344_5566_7711
        );
        let pair = run_regs(code::MOV_R8B_IMM, &[(R8, WIDE_A)]);
        assert_eq!(pair.reg(R8), 0x1122_3344_5566_777f);
        let pair = run_regs(code::MOV_R15B_BL, &[(15, WIDE_A), (RBX, WIDE_B)]);
        assert_eq!(pair.reg(15), 0x1122_3344_5566_7700);
    }

    #[test]
    fn alu_on_a_high_byte_register_leaves_the_rest_of_rax_alone() {
        let pair = run_regs(
            code::ADD_AH_BH,
            &[(RAX, 0x1122_3344_5566_ff88), (RBX, 0x0000_0000_0000_0100)],
        );
        assert_eq!(
            pair.reg(RAX),
            0x1122_3344_5566_0088,
            "ah wraps, al and the upper bits stay"
        );
        assert_flags(pair.flags(), ALL_ARITH, CF | ZF | AF | PF, "add ah, bh");
        let pair = run_regs(code::INC_AH, &[(RAX, 0x0000_0000_0000_7f00)]);
        assert_eq!(pair.reg(RAX), 0x8000);
        assert_flags(pair.flags(), OF | SF, OF | SF, "inc ah");
        let pair = run_regs(code::NEG_BH, &[(RBX, 0x0000_0000_0000_0100)]);
        assert_eq!(pair.reg(RBX), 0xff00);
        let pair = run_regs(code::NOT_CH, &[(RCX, 0xffff_ffff_ffff_0f55)]);
        assert_eq!(pair.reg(RCX), 0xffff_ffff_ffff_f055);
        let pair = run_regs(code::SHL_AH_1, &[(RAX, 0x0000_0000_0000_81ff)]);
        assert_eq!(
            pair.reg(RAX),
            0x02ff,
            "bit 7 of ah is shifted out, the carry is set"
        );
        assert_flags(pair.flags(), CF, CF, "shl ah, 1");
        let pair = run_regs(code::TEST_AH_BH, &[(RAX, 0xf0ff), (RBX, 0x0f00)]);
        assert_eq!(pair.reg(RAX), 0xf0ff);
        assert_flags(pair.flags(), ZF | CF | OF, ZF, "test ah, bh");
        let pair = run_regs(code::CMP_AH_IMM, &[(RAX, 0x0500)]);
        assert_flags(pair.flags(), ZF | CF, ZF, "cmp ah, 5");
    }

    #[test]
    fn xchg_of_high_and_low_byte_registers() {
        assert_eq!(
            run_regs(code::XCHG_AH_AL, &[(RAX, 0xdead_beef_1234_5678)]).reg(RAX),
            0xdead_beef_1234_7856
        );
        let pair = run_regs(code::XCHG_AH_BH, &[(RAX, 0x1100), (RBX, 0x2200)]);
        assert_eq!((pair.reg(RAX), pair.reg(RBX)), (0x2200, 0x1100));
    }

    #[test]
    fn movzx_and_movsx_from_high_and_rex_byte_registers() {
        assert_eq!(
            run_regs(code::MOVZX_EAX_BH, &[(RAX, u64::MAX), (RBX, 0x8100)]).reg(RAX),
            0x81
        );
        assert_eq!(
            run_regs(code::MOVSX_EAX_DH, &[(RAX, u64::MAX), (RDX, 0x8100)]).reg(RAX),
            0xffff_ff81
        );
        // dil needs REX; without it register 7 would be bh.
        assert_eq!(
            run_regs(code::MOVZX_EAX_DIL, &[(RDI, 0x1234_56ff)]).reg(RAX),
            0xff
        );
        assert_eq!(
            run_regs(code::MOVSX_RAX_SIL, &[(RSI, 0x80)]).reg(RAX),
            0xffff_ffff_ffff_ff80
        );
    }

    #[test]
    fn register_writes_merge_below_32_bits_and_zero_extend_at_32_bits() {
        let regs = [(RAX, WIDE_A), (RBX, WIDE_B)];
        assert_eq!(run_regs(code::MOV_RAX_RBX, &regs).reg(RAX), WIDE_B);
        assert_eq!(run_regs(code::MOV_EAX_EBX, &regs).reg(RAX), 0xeeff_9900);
        assert_eq!(
            run_regs(code::MOV_AX_BX, &regs).reg(RAX),
            0x1122_3344_5566_9900
        );
        assert_eq!(
            run_regs(code::MOV_AL_BL, &regs).reg(RAX),
            0x1122_3344_5566_7700
        );
        let regs = [(R9, WIDE_A), (R10, WIDE_B)];
        assert_eq!(run_regs(code::MOV_R9D_R10D, &regs).reg(R9), 0xeeff_9900);
        assert_eq!(
            run_regs(code::MOV_R9W_R10W, &regs).reg(R9),
            0x1122_3344_5566_9900
        );
    }

    #[test]
    fn mov_immediate_to_register_by_size() {
        let regs = [(RAX, u64::MAX), (R9, u64::MAX), (RBX, u64::MAX), (RCX, 0)];
        assert_eq!(run_regs(code::MOV_EAX_IMM, &regs).reg(RAX), 0x1234_5678);
        assert_eq!(
            run_regs(code::MOV_RAX_IMM64, &regs).reg(RAX),
            0x1122_3344_5566_7788
        );
        assert_eq!(
            run_regs(code::MOV_RAX_IMM_NEG, &regs).reg(RAX),
            0xffff_ffff_ffff_fffe
        );
        assert_eq!(
            run_regs(code::MOV_AX_IMM, &regs).reg(RAX),
            0xffff_ffff_ffff_1234
        );
        assert_eq!(
            run_regs(code::MOV_AL_IMM, &regs).reg(RAX),
            0xffff_ffff_ffff_ff80
        );
        assert_eq!(run_regs(code::MOV_R9D_IMM, &regs).reg(R9), 7);
        assert_eq!(
            run_regs(code::MOV_R12_IMM64, &regs).reg(12),
            0xfedc_ba98_7654_3210
        );
        assert_eq!(
            run_regs(code::MOV_RBX_IMM32_REG, &regs).reg(RBX),
            0x7fff_ffff
        );
        assert_eq!(
            run_regs(code::MOV_RM64_IMM_REG, &regs).reg(RCX),
            u64::MAX,
            "imm32 is sign-extended"
        );
    }

    #[test]
    fn mov_immediate_to_memory_stores_only_the_operand_width() {
        let cases: [(&[u8], u8, u64); 4] = [
            (code::MOV_MEM8_IMM, 1, 0x9c),
            (code::MOV_MEM16_IMM, 2, 0x9abc),
            (code::MOV_MEM32_IMM, 4, 0x89ab_cdef),
            (code::MOV_MEM64_IMM, 8, 0xffff_ffff_ffff_fffd),
        ];
        for (snippet, size, value) in cases {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| memory_with_marker(h, 0, 0));
            let mut want = [MEMORY_MARKER; 16];
            want[..size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
            assert_eq!(pair.interp.read(DATA, 16), want, "size {size}");
        }
    }

    #[test]
    fn mov_register_to_and_from_memory_by_size() {
        let value = 0x8877_6655_4433_2211u64;
        let stores: [(&[u8], u8); 4] = [
            (code::MOV_MEM_RAX, 8),
            (code::MOV_MEM_EAX, 4),
            (code::MOV_MEM_AX, 2),
            (code::MOV_MEM_AL, 1),
        ];
        for (snippet, size) in stores {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                memory_with_marker(h, 0, 0);
                h.cpu.gpr[RAX] = value;
            });
            let mut want = [MEMORY_MARKER; 16];
            want[..size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
            assert_eq!(pair.interp.read(DATA, 16), want, "store size {size}");
        }
        let mut pair = Pair::new(code::MOV_MEM_AH);
        pair.halted(|h| {
            memory_with_marker(h, 0, 0);
            h.cpu.gpr[RAX] = value;
        });
        assert_eq!(
            pair.interp.read(DATA, 2),
            [0x22, MEMORY_MARKER],
            "ah is the byte stored"
        );
        let loads: [(&[u8], u64); 4] = [
            (code::MOV_RAX_MEM, value),
            (code::MOV_EAX_MEM, 0x4433_2211),
            (code::MOV_AX_MEM, 0xffff_ffff_ffff_2211),
            (code::MOV_AL_MEM, 0xffff_ffff_ffff_ff11),
        ];
        for (snippet, want) in loads {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                memory_with_marker(h, value, 8);
                h.cpu.gpr[RAX] = u64::MAX;
            });
            assert_eq!(pair.reg(RAX), want, "{snippet:02x?}");
        }
        let mut pair = Pair::new(code::MOV_AH_MEM);
        pair.halted(|h| {
            memory_with_marker(h, 0x5a, 1);
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(pair.reg(RAX), 0xffff_ffff_ffff_5aff);
        let mut pair = Pair::new(code::MOV_BYTE_R8B_MEM);
        pair.halted(|h| {
            memory_with_marker(h, 0x5a, 1);
            h.cpu.gpr[R8] = u64::MAX;
        });
        assert_eq!(pair.reg(R8), 0xffff_ffff_ffff_ff5a);
    }

    #[test]
    fn mov_to_and_from_an_absolute_address_uses_the_accumulator() {
        let value = 0x1122_3344_5566_7788u64;
        let mut pair = Pair::new(code::MOVABS_RAX_MOFFS);
        pair.halted(|h| h.write_u64(SCRATCH, value));
        assert_eq!(pair.reg(RAX), value);
        let loads: [(&[u8], u64); 3] = [
            (code::MOVABS_EAX_MOFFS, 0x5566_7788),
            (code::MOVABS_AX_MOFFS, 0xffff_ffff_ffff_7788),
            (code::MOVABS_AL_MOFFS, 0xffff_ffff_ffff_ff88),
        ];
        for (snippet, want) in loads {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                h.write_u64(SCRATCH, value);
                h.cpu.gpr[RAX] = u64::MAX;
            });
            assert_eq!(pair.reg(RAX), want, "{snippet:02x?}");
        }
        let stores: [(&[u8], u8); 4] = [
            (code::MOVABS_MOFFS_RAX, 8),
            (code::MOVABS_MOFFS_EAX, 4),
            (code::MOVABS_MOFFS_AX, 2),
            (code::MOVABS_MOFFS_AL, 1),
        ];
        for (snippet, size) in stores {
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                h.write(SCRATCH, &[MEMORY_MARKER; 16]);
                h.cpu.gpr[RAX] = value;
            });
            let mut want = [MEMORY_MARKER; 16];
            want[..size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
            assert_eq!(pair.interp.read(SCRATCH, 16), want, "store size {size}");
        }
    }

    #[test]
    fn mov_absolute_address_with_the_address_size_prefix_takes_a_four_byte_address() {
        let mut pair = Pair::new(code::MOVABS_EAX_MOFFS_ADDR32);
        pair.halted(|h| h.write_u64(SCRATCH, 0xaabb_ccdd));
        assert_eq!(pair.reg(RAX), 0xaabb_ccdd);
        let mut pair = Pair::new(code::MOVABS_MOFFS_EAX_ADDR32);
        pair.halted(|h| h.cpu.gpr[RAX] = 0x1234_5678);
        assert_eq!(pair.interp.read(SCRATCH, 4), 0x1234_5678u32.to_le_bytes());
    }

    #[test]
    fn mov_absolute_address_applies_the_fs_base_and_faults_outside_the_map() {
        let mut pair = Pair::new(code::MOVABS_AL_MOFFS_FS);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = SCRATCH;
            h.write(SCRATCH + 0x10, &[0x6e]);
        });
        assert_eq!(pair.reg(RAX) & 0xff, 0x6e);
        let (pair, stop) = run_regs_stop(code::MOVABS_AL_MOFFS_HIGH, &[]);
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)));
        assert_eq!(pair.cpu().cr2, UNMAPPED);
    }

    #[test]
    fn movzx_and_movsx_extend_from_byte_and_word_sources() {
        let regs = [(RAX, u64::MAX), (RBX, 0x1234_5678_9abc_de80)];
        let cases: [(&[u8], u64); 10] = [
            (code::MOVZX_EAX_BL, 0x80),
            (code::MOVZX_RAX_BL, 0x80),
            (code::MOVZX_AX_BL, 0xffff_ffff_ffff_0080),
            (code::MOVZX_EAX_BX, 0xde80),
            (code::MOVZX_RAX_BX, 0xde80),
            (code::MOVSX_EAX_BL, 0xffff_ff80),
            (code::MOVSX_RAX_BL, 0xffff_ffff_ffff_ff80),
            (code::MOVSX_AX_BL, 0xffff_ffff_ffff_ff80),
            (code::MOVSX_EAX_BX, 0xffff_de80),
            (code::MOVSX_RAX_BX, 0xffff_ffff_ffff_de80),
        ];
        for (snippet, want) in cases {
            assert_eq!(run_regs(snippet, &regs).reg(RAX), want, "{snippet:02x?}");
        }
        // A positive source stays positive.
        let regs = [(RAX, u64::MAX), (RBX, 0x7f)];
        assert_eq!(run_regs(code::MOVSX_RAX_BL, &regs).reg(RAX), 0x7f);
        let regs = [(RAX, u64::MAX), (RBX, 0x7fff)];
        assert_eq!(run_regs(code::MOVSX_RAX_BX, &regs).reg(RAX), 0x7fff);
    }

    #[test]
    fn movzx_and_movsx_from_memory_read_only_the_source_width() {
        let setup = |h: &mut Harness| {
            h.write(DATA, &[0x81, 0x92, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
            h.cpu.gpr[RDX] = DATA;
            h.cpu.gpr[RAX] = u64::MAX;
        };
        let cases: [(&[u8], u64); 4] = [
            (code::MOVZX_EAX_MEM8, 0x81),
            (code::MOVZX_RAX_MEM16, 0x9281),
            (code::MOVSX_RAX_MEM8, 0xffff_ffff_ffff_ff81),
            (code::MOVSX_EAX_MEM16, 0xffff_9281),
        ];
        for (snippet, want) in cases {
            let mut pair = Pair::new(snippet);
            pair.halted(setup);
            assert_eq!(pair.reg(RAX), want, "{snippet:02x?}");
        }
    }

    #[test]
    fn movsxd_sign_extends_only_with_rex_w() {
        let regs = [(RAX, u64::MAX), (RBX, 0x1234_5678_8000_0001)];
        assert_eq!(
            run_regs(code::MOVSXD_RAX_EBX, &regs).reg(RAX),
            0xffff_ffff_8000_0001
        );
        let regs = [(RAX, u64::MAX), (RBX, 0x7fff_ffff)];
        assert_eq!(run_regs(code::MOVSXD_RAX_EBX, &regs).reg(RAX), 0x7fff_ffff);
        // Without REX.W it is a plain 32-bit move, which zero-extends.
        let regs = [(RAX, u64::MAX), (RBX, 0x8000_0001)];
        assert_eq!(run_regs(code::MOVSXD_32_EBX, &regs).reg(RAX), 0x8000_0001);
        let mut pair = Pair::new(code::MOVSXD_RAX_MEM);
        pair.halted(|h| {
            memory_with_marker(h, 0xffff_fffe, 4);
            h.cpu.gpr[RAX] = 0;
        });
        assert_eq!(pair.reg(RAX), 0xffff_ffff_ffff_fffe);
    }

    #[test]
    fn lea_computes_the_effective_address_without_touching_memory() {
        let regs = [(RBX, 0x1000), (RCX, 3), (RAX, u64::MAX)];
        assert_eq!(run_regs(code::LEA_RAX_SIB, &regs).reg(RAX), 0x1014);
        assert_eq!(run_regs(code::LEA_RAX_BASE_NEG, &regs).reg(RAX), 0xff0);
        assert_eq!(run_regs(code::LEA_RAX_SCALE8, &regs).reg(RAX), 0x118);
        assert_eq!(run_regs(code::LEA_RAX_DISP32, &regs).reg(RAX), 0x1234);
        // An address that points at unmapped or non-canonical memory is just a number.
        let regs = [(RBX, NON_CANONICAL), (RCX, 1), (RAX, 0)];
        assert_eq!(
            run_regs(code::LEA_RAX_SIB, &regs).reg(RAX),
            NON_CANONICAL + 12
        );
    }

    #[test]
    fn lea_truncates_or_merges_to_the_operand_size() {
        let regs = [(RBX, 0x1_0000_1000), (RCX, 3), (RAX, u64::MAX)];
        assert_eq!(
            run_regs(code::LEA_EAX_SIB, &regs).reg(RAX),
            0x1014,
            "32-bit result zero-extends"
        );
        assert_eq!(
            run_regs(code::LEA_AX_SIB, &regs).reg(RAX),
            0xffff_ffff_ffff_1014,
            "16-bit result merges"
        );
    }

    #[test]
    fn lea_with_extended_registers_rip_and_rbp_r13_bases() {
        let regs = [
            (12, 0x100),
            (13, 0x10),
            (RBP, 0x2000),
            (RSP, 0x3000),
            (RAX, 0),
        ];
        assert_eq!(run_regs(code::LEA_RAX_R12_R13, &regs).reg(RAX), 0x19f);
        assert_eq!(run_regs(code::LEA_RAX_RBP, &regs).reg(RAX), 0x2000);
        assert_eq!(run_regs(code::LEA_RAX_R13, &regs).reg(RAX), 0x10);
        assert_eq!(run_regs(code::LEA_RAX_RSP, &regs).reg(RAX), 0x3008);
        // RIP-relative is relative to the end of the instruction (7 bytes long).
        assert_eq!(
            run_regs(code::LEA_RAX_RIP, &regs).reg(RAX),
            CODE_BASE + 7 + 0x10
        );
    }

    #[test]
    fn lea_with_an_address_size_prefix_wraps_at_32_bits() {
        let regs = [(RBX, 0xffff_ffff), (RCX, 1), (RAX, u64::MAX)];
        assert_eq!(run_regs(code::LEA_EAX_ADDR32, &regs).reg(RAX), 1);
        assert_eq!(run_regs(code::LEA_RAX_ADDR32, &regs).reg(RAX), 1);
        // The upper halves of the base and index registers play no part.
        let regs = [
            (RBX, 0xabcd_0000_0000_0010),
            (RCX, 0x1234_0000_0000_0002),
            (RAX, 0),
        ];
        assert_eq!(run_regs(code::LEA_EAX_ADDR32, &regs).reg(RAX), 0x14);
    }

    #[test]
    fn lea_ignores_segment_overrides_and_rejects_a_register_operand() {
        let mut pair = Pair::new(code::LEA_RAX_FS);
        pair.halted(|h| {
            h.cpu.seg[SEG_FS].base = 0x5000;
            h.cpu.gpr[RBX] = 0x10;
        });
        assert_eq!(pair.reg(RAX), 0x18);
        let (_, stop) = run_regs_stop(code::LEA_REG_FORM, &[]);
        assert_eq!(stop, ud());
    }

    #[test]
    fn xchg_register_forms_by_operand_size() {
        let regs = [(RAX, 0x1111_1111_2222_2222), (RBX, 0x3333_3333_4444_4444)];
        let pair = run_regs(code::XCHG_RAX_RBX, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(RBX)),
            (0x3333_3333_4444_4444, 0x1111_1111_2222_2222)
        );
        let pair = run_regs(code::XCHG_RAX_RBX_MODRM, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(RBX)),
            (0x3333_3333_4444_4444, 0x1111_1111_2222_2222)
        );
        let pair = run_regs(code::XCHG_EAX_EBX, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(RBX)),
            (0x4444_4444, 0x2222_2222),
            "32-bit forms zero-extend both"
        );
        let pair = run_regs(code::XCHG_AX_BX, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(RBX)),
            (0x1111_1111_2222_4444, 0x3333_3333_4444_2222)
        );
        let pair = run_regs(code::XCHG_AL_BL, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(RBX)),
            (0x1111_1111_2222_2244, 0x3333_3333_4444_4422)
        );
        let regs = [(RAX, 0x1111_1111_2222_2222), (R8, 0x3333_3333_4444_4444)];
        let pair = run_regs(code::XCHG_R8_RAX, &regs);
        assert_eq!(
            (pair.reg(RAX), pair.reg(R8)),
            (0x3333_3333_4444_4444, 0x1111_1111_2222_2222)
        );
        let pair = run_regs(code::XCHG_R8D_EAX, &regs);
        assert_eq!((pair.reg(RAX), pair.reg(R8)), (0x4444_4444, 0x2222_2222));
    }

    #[test]
    fn xchg_with_memory_swaps_exactly_the_operand_width() {
        let cases: [(&[u8], u8); 4] = [
            (code::XCHG_MEM_RBX, 8),
            (code::XCHG_MEM_EBX, 4),
            (code::XCHG_MEM_BX, 2),
            (code::XCHG_MEM_BL, 1),
        ];
        for (snippet, size) in cases {
            let m = mask(size);
            let mut pair = Pair::new(snippet);
            pair.halted(|h| {
                memory_with_marker(h, 0x1122_3344_5566_7788, size);
                h.cpu.gpr[RBX] = 0xaabb_ccdd_eeff_0011;
            });
            let mut want = [MEMORY_MARKER; 16];
            want[..size as usize]
                .copy_from_slice(&0xaabb_ccdd_eeff_0011u64.to_le_bytes()[..size as usize]);
            assert_eq!(pair.interp.read(DATA, 16), want, "size {size}");
            let old = 0x1122_3344_5566_7788u64 & m;
            let want_rbx = match size {
                8 => old,
                4 => old,
                _ => (0xaabb_ccdd_eeff_0011 & !m) | old,
            };
            assert_eq!(pair.reg(RBX), want_rbx, "size {size}");
        }
    }

    #[test]
    fn nop_changes_nothing_while_the_long_form_of_xchg_eax_eax_zero_extends() {
        let regs = [(RAX, 0x1_0000_0005), (RBX, 7), (RSP, STACK_TOP)];
        let pair = run_regs(code::NOP, &regs);
        assert_eq!(pair.reg(RAX), 0x1_0000_0005);
        let pair = run_regs(code::XCHG_EAX_EAX_LONG, &regs);
        assert_eq!(pair.reg(RAX), 5);
    }

    #[test]
    fn nop_pause_endbr64_and_prefetch_do_not_touch_state_or_memory() {
        for snippet in [code::NOP, code::PAUSE, code::ENDBR64] {
            let regs: Vec<(usize, u64)> =
                (0..16).map(|r| (r, 0x1111_0000 * (r as u64 + 1))).collect();
            let pair = run_regs(snippet, &regs);
            for (r, v) in &regs {
                if *r != RSP {
                    assert_eq!(pair.reg(*r), *v, "{snippet:02x?} register {r}");
                }
            }
        }
        // The memory operand of a hint nop or a prefetch is never accessed, so an unmapped or
        // non-canonical address does not fault.
        for snippet in [code::NOP_MEM_UNMAPPED, code::NOP_WORD_MEM, code::PREFETCH] {
            for address in [UNMAPPED, NON_CANONICAL] {
                let (_, stop) = run_regs_stop(snippet, &[(RAX, address)]);
                assert_eq!(stop, Stop::Halted, "{snippet:02x?} at {address:#x}");
            }
        }
    }

    /// PREFETCHI (CPUID 0x8000_0021 EAX bit 20): prefetchit0/prefetchit1 are `0F 18 /7` and `/6`
    /// with a RIP-relative operand, inside the hint space, so they change nothing and never fault,
    /// even when the line they name is unmapped, in the interpreter and the JIT alike.
    #[test]
    fn prefetchit0_and_prefetchit1_are_hints_that_never_fault() {
        // prefetchit0 [rip+0x1000]; prefetchit1 [rip-0x100000]
        for snippet in [
            &[0x0f, 0x18, 0x3d, 0x00, 0x10, 0x00, 0x00, 0xf4][..],
            &[0x0f, 0x18, 0x35, 0x00, 0x00, 0xf0, 0xff, 0xf4][..],
        ] {
            let regs: Vec<(usize, u64)> =
                (0..16).map(|r| (r, 0x2222_0000 * (r as u64 + 1))).collect();
            let pair = run_regs(snippet, &regs);
            for (r, v) in &regs {
                if *r != RSP {
                    assert_eq!(pair.reg(*r), *v, "{snippet:02x?} register {r}");
                }
            }
            assert_eq!(pair.cpu().rip, CODE_BASE + snippet.len() as u64);
        }
    }

    /// (operand size, memory form, register form, locked memory form) of a read-modify-write family.
    type SnippetFamily = (u8, &'static [u8], &'static [u8], &'static [u8]);

    const CMPXCHG_SNIPPETS: [SnippetFamily; 4] = [
        (
            8,
            code::CMPXCHG_MEM_8,
            code::CMPXCHG_REG_8,
            code::LOCK_CMPXCHG_MEM_8,
        ),
        (
            4,
            code::CMPXCHG_MEM_4,
            code::CMPXCHG_REG_4,
            code::LOCK_CMPXCHG_MEM_4,
        ),
        (
            2,
            code::CMPXCHG_MEM_2,
            code::CMPXCHG_REG_2,
            code::LOCK_CMPXCHG_MEM_2,
        ),
        (
            1,
            code::CMPXCHG_MEM_1,
            code::CMPXCHG_REG_1,
            code::LOCK_CMPXCHG_MEM_1,
        ),
    ];

    #[test]
    fn cmpxchg_succeeds_when_the_accumulator_equals_the_destination() {
        for (size, memory_code, register_code, lock_code) in CMPXCHG_SNIPPETS {
            let m = mask(size);
            let value = 0x1122_3344_5566_7788u64 & m;
            let replacement = 0x99aa_bbcc_ddee_ff00u64 & m;
            for snippet in [memory_code, lock_code] {
                let mut pair = Pair::new(snippet);
                pair.halted(|h| {
                    memory_with_marker(h, value, size);
                    h.cpu.gpr[RAX] = OPERAND_JUNK & !m | value;
                    h.cpu.gpr[RBX] = replacement;
                });
                let mut want = [MEMORY_MARKER; 16];
                want[..size as usize].copy_from_slice(&replacement.to_le_bytes()[..size as usize]);
                assert_eq!(pair.interp.read(DATA, 16), want, "size {size}");
                assert_eq!(
                    pair.reg(RAX),
                    OPERAND_JUNK & !m | value,
                    "accumulator untouched"
                );
                assert_flags(
                    pair.flags(),
                    ALL_ARITH,
                    alu_reference(ALU_SUB, size, value, value, 0).flags,
                    "flags",
                );
            }
            let mut pair = Pair::new(register_code);
            pair.halted(|h| {
                h.cpu.gpr[RAX] = value;
                h.cpu.gpr[RCX] = value;
                h.cpu.gpr[RBX] = replacement;
            });
            assert_eq!(
                pair.reg(RCX) & m,
                replacement,
                "size {size} register destination"
            );
            assert_flags(pair.flags(), ZF, ZF, "ZF set on success");
        }
    }

    #[test]
    fn cmpxchg_fails_when_they_differ_and_loads_the_accumulator() {
        for (size, memory_code, register_code, lock_code) in CMPXCHG_SNIPPETS {
            let m = mask(size);
            let value = 0x1122_3344_5566_7788u64 & m;
            let accumulator = 0x0102_0304_0506_0708u64 & m;
            let replacement = 0x99aa_bbcc_ddee_ff00u64 & m;
            for snippet in [memory_code, lock_code] {
                let mut pair = Pair::new(snippet);
                pair.halted(|h| {
                    memory_with_marker(h, value, size);
                    h.cpu.gpr[RAX] = OPERAND_JUNK & !m | accumulator;
                    h.cpu.gpr[RBX] = replacement;
                });
                let mut want = [MEMORY_MARKER; 16];
                want[..size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
                assert_eq!(
                    pair.interp.read(DATA, 16),
                    want,
                    "size {size}: destination unchanged"
                );
                let want_rax = match size {
                    8 | 4 => value,
                    _ => (OPERAND_JUNK & !m) | value,
                };
                assert_eq!(pair.reg(RAX), want_rax, "size {size}: accumulator loaded");
                let expected = alu_reference(ALU_SUB, size, accumulator, value, 0);
                assert_flags(
                    pair.flags(),
                    ALL_ARITH,
                    expected.flags,
                    &format!("size {size}"),
                );
            }
            let mut pair = Pair::new(register_code);
            pair.halted(|h| {
                h.cpu.gpr[RAX] = accumulator;
                h.cpu.gpr[RCX] = value;
                h.cpu.gpr[RBX] = replacement;
            });
            assert_eq!(
                pair.reg(RCX) & m,
                value,
                "size {size}: register destination unchanged"
            );
            assert_eq!(pair.reg(RAX) & m, value);
            assert_flags(pair.flags(), ZF, 0, "ZF clear on failure");
        }
    }

    #[test]
    fn cmpxchg_writes_the_destination_even_when_the_compare_fails() {
        // The failing form still performs the store, so a write-protected destination faults.
        let mut pair = Pair::new(code::CMPXCHG_MEM_8);
        let stop = pair.run(|h| {
            write_protect_huge_page(h, HOLE);
            h.write_u64(HOLE + 0x40, 5);
            h.cpu.gpr[RDX] = HOLE + 0x40;
            h.cpu.gpr[RAX] = 6;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_PROTECTION_WRITE)));
        assert_eq!(pair.cpu().rip, CODE_BASE);
        assert_eq!(pair.reg(RAX), 6, "rax is not loaded when the store faults");
    }

    #[test]
    fn xadd_stores_the_sum_and_returns_the_old_destination() {
        let xadd: [SnippetFamily; 4] = [
            (8, code::XADD_MEM_8, code::XADD_REG_8, code::LOCK_XADD_MEM_8),
            (4, code::XADD_MEM_4, code::XADD_REG_4, code::LOCK_XADD_MEM_4),
            (2, code::XADD_MEM_2, code::XADD_REG_2, code::LOCK_XADD_MEM_2),
            (1, code::XADD_MEM_1, code::XADD_REG_1, code::LOCK_XADD_MEM_1),
        ];
        for (size, memory_code, register_code, lock_code) in xadd {
            let m = mask(size);
            for (a, b) in [
                (0x7fu64, 1u64),
                (0xff, 1),
                (0x1234_5678_9abc_def0, 0x0fed_cba9_8765_4321),
                (m, m),
            ] {
                let (a, b) = (a & m, b & m);
                let expected = alu_reference(ALU_ADD, size, a, b, 0);
                for snippet in [memory_code, lock_code] {
                    let mut pair = Pair::new(snippet);
                    pair.halted(|h| {
                        memory_with_marker(h, a, size);
                        h.cpu.gpr[RBX] = (OPERAND_JUNK & !m) | b;
                    });
                    let mut want = [MEMORY_MARKER; 16];
                    want[..size as usize]
                        .copy_from_slice(&expected.result.to_le_bytes()[..size as usize]);
                    assert_eq!(pair.interp.read(DATA, 16), want, "size {size}");
                    let want_rbx = if size >= 4 {
                        a
                    } else {
                        (OPERAND_JUNK & !m) | a
                    };
                    assert_eq!(
                        pair.reg(RBX),
                        want_rbx,
                        "size {size}: source gets the old value"
                    );
                    assert_flags(pair.flags(), ALL_ARITH, expected.flags, "xadd flags");
                }
                let mut pair = Pair::new(register_code);
                pair.halted(|h| {
                    h.cpu.gpr[RCX] = (OPERAND_JUNK & !m) | a;
                    h.cpu.gpr[RBX] = (OPERAND_JUNK & !m) | b;
                });
                let want_rcx = if size >= 4 {
                    expected.result
                } else {
                    (OPERAND_JUNK & !m) | expected.result
                };
                assert_eq!(pair.reg(RCX), want_rcx, "size {size}");
                let want_rbx = if size >= 4 {
                    a
                } else {
                    (OPERAND_JUNK & !m) | a
                };
                assert_eq!(pair.reg(RBX), want_rbx, "size {size}");
            }
        }
    }

    #[test]
    fn cmpxchg8b_match_and_mismatch() {
        // The snippet addresses memory through rdi so edx:eax stay free.
        let mut pair = Pair::new(code::CMPXCHG8B_RDI);
        pair.halted(|h| {
            h.write_u64(DATA, 0x1111_1111_2222_2222);
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RAX] = 0xffff_ffff_2222_2222;
            h.cpu.gpr[RDX] = 0xffff_ffff_1111_1111;
            h.cpu.gpr[RBX] = 0x4444_4444;
            h.cpu.gpr[RCX] = 0x3333_3333;
            h.cpu.set_eflags_arith(CF | OF | SF | AF | PF);
        });
        assert_eq!(pair.interp.read_u64(DATA), 0x3333_3333_4444_4444);
        assert_flags(
            pair.flags(),
            ALL_ARITH,
            ZF | CF | OF | SF | AF | PF,
            "only ZF changes",
        );
        assert_eq!(
            pair.reg(RAX),
            0xffff_ffff_2222_2222,
            "registers kept on a match"
        );
        pair.halted(|h| {
            h.write_u64(DATA, 0x1111_1111_2222_2222);
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RAX] = 0xffff_ffff_0000_0001;
            h.cpu.gpr[RDX] = 0xffff_ffff_0000_0002;
            h.cpu.set_eflags_arith(ZF | CF);
        });
        assert_eq!(
            pair.interp.read_u64(DATA),
            0x1111_1111_2222_2222,
            "memory kept on a mismatch"
        );
        assert_eq!(pair.reg(RAX), 0x2222_2222, "eax loaded, upper half cleared");
        assert_eq!(pair.reg(RDX), 0x1111_1111);
        assert_flags(pair.flags(), ALL_ARITH, CF, "ZF cleared, others kept");
    }

    #[test]
    fn locked_cmpxchg8b_behaves_like_the_unlocked_form() {
        let mut pair = Pair::new(code::LOCK_CMPXCHG8B_RDI);
        pair.halted(|h| {
            h.write_u64(DATA, 0x1111_1111_2222_2222);
            h.cpu.gpr[RDI] = DATA;
            h.cpu.gpr[RAX] = 0x2222_2222;
            h.cpu.gpr[RDX] = 0x1111_1111;
            h.cpu.gpr[RBX] = 0x4444_4444;
            h.cpu.gpr[RCX] = 0x3333_3333;
        });
        assert_eq!(pair.interp.read_u64(DATA), 0x3333_3333_4444_4444);
        assert_flags(pair.flags(), ZF, ZF, "match");
    }

    #[test]
    fn cmpxchg8b_register_form_and_group9_other_forms_are_not_implemented() {
        for snippet in [code::CMPXCHG8B_REG, code::GROUP9_REG0] {
            let (_, stop) = run_regs_stop(snippet, &[(RDX, DATA)]);
            assert_eq!(stop, ud(), "{snippet:02x?}");
        }
    }

    #[test]
    fn lock_prefix_on_memory_read_modify_write_forms_executes_normally() {
        let mut pair = Pair::new(code::LOCK_ADD_MEM);
        pair.halted(|h| {
            memory_with_marker(h, 40, 8);
            h.cpu.gpr[RBX] = 2;
        });
        assert_eq!(pair.interp.read_u64(DATA), 42);
        let mut pair = Pair::new(code::LOCK_INC_MEM);
        pair.halted(|h| memory_with_marker(h, 0xffff_ffff, 4));
        assert_eq!(
            pair.interp.read(DATA, 8),
            [
                0,
                0,
                0,
                0,
                MEMORY_MARKER,
                MEMORY_MARKER,
                MEMORY_MARKER,
                MEMORY_MARKER
            ]
        );
        assert_flags(pair.flags(), ZF | CF, ZF, "inc leaves CF");
        let mut pair = Pair::new(code::LOCK_XCHG_MEM);
        pair.halted(|h| {
            memory_with_marker(h, 7, 8);
            h.cpu.gpr[RBX] = 9;
        });
        assert_eq!(pair.interp.read_u64(DATA), 9);
        assert_eq!(pair.reg(RBX), 7);
    }

    #[test]
    fn sign_extension_of_the_accumulator() {
        let neg = 0xffee_ddcc_bbaa_9988u64;
        let pos = 0x7766_5544_3322_1100u64 | 0x7f;
        assert_eq!(
            run_regs(code::CBW, &[(RAX, neg)]).reg(RAX),
            0xffee_ddcc_bbaa_ff88,
            "al -> ax, upper kept"
        );
        assert_eq!(
            run_regs(code::CBW, &[(RAX, pos)]).reg(RAX),
            0x7766_5544_3322_007f
        );
        assert_eq!(
            run_regs(code::CWDE, &[(RAX, neg)]).reg(RAX),
            0xffff_9988,
            "ax -> eax, zero-extended to rax"
        );
        assert_eq!(
            run_regs(code::CWDE, &[(RAX, 0x1234_5678_0000_7fff)]).reg(RAX),
            0x7fff
        );
        assert_eq!(
            run_regs(code::CDQE, &[(RAX, neg)]).reg(RAX),
            0xffff_ffff_bbaa_9988
        );
        assert_eq!(
            run_regs(code::CDQE, &[(RAX, 0x1_7fff_ffff)]).reg(RAX),
            0x7fff_ffff
        );
    }

    #[test]
    fn sign_fill_of_rdx() {
        let neg = 0xffee_ddcc_bbaa_9988u64;
        let rdx = 0x1122_3344_5566_7788u64;
        assert_eq!(
            run_regs(code::CWD, &[(RAX, neg), (RDX, rdx)]).reg(RDX),
            0x1122_3344_5566_ffff
        );
        assert_eq!(
            run_regs(code::CWD, &[(RAX, 0x7fff), (RDX, rdx)]).reg(RDX),
            0x1122_3344_5566_0000
        );
        assert_eq!(
            run_regs(code::CDQ, &[(RAX, neg), (RDX, rdx)]).reg(RDX),
            0xffff_ffff
        );
        assert_eq!(
            run_regs(code::CDQ, &[(RAX, 0x7fff_ffff), (RDX, rdx)]).reg(RDX),
            0
        );
        assert_eq!(
            run_regs(code::CQO, &[(RAX, neg), (RDX, rdx)]).reg(RDX),
            u64::MAX
        );
        assert_eq!(
            run_regs(code::CQO, &[(RAX, 0x7fff_ffff_ffff_ffff), (RDX, rdx)]).reg(RDX),
            0
        );
        let pair = run_regs(code::CQO, &[(RAX, neg), (RDX, rdx)]);
        assert_eq!(pair.reg(RAX), neg, "rax is not modified");
    }

    #[test]
    fn bswap_reverses_bytes_and_the_32_bit_form_zero_extends() {
        let wide = 0x1122_3344_5566_7788u64;
        assert_eq!(
            run_regs(code::BSWAP_EAX, &[(RAX, wide)]).reg(RAX),
            0x8877_6655
        );
        assert_eq!(
            run_regs(code::BSWAP_RAX, &[(RAX, wide)]).reg(RAX),
            0x8877_6655_4433_2211
        );
        assert_eq!(
            run_regs(code::BSWAP_R9D, &[(R9, wide)]).reg(R9),
            0x8877_6655
        );
        assert_eq!(
            run_regs(code::BSWAP_R10, &[(R10, wide)]).reg(R10),
            0x8877_6655_4433_2211
        );
    }

    #[test]
    fn lahf_packs_the_five_status_flags_and_the_fixed_bit() {
        let mut pair = Pair::new(code::LAHF);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xffff_ffff_ffff_ffff;
            h.cpu.set_eflags_arith(SF | ZF | AF | PF | CF | OF);
        });
        // SF ZF 0 AF 0 PF 1 CF; OF is not part of AH.
        assert_eq!(pair.reg(RAX), 0xffff_ffff_ffff_d7ff);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0;
            h.cpu.set_eflags_arith(0);
        });
        assert_eq!(pair.reg(RAX), 0x0200, "bit 1 of ah always reads as 1");
    }

    #[test]
    fn sahf_loads_the_five_status_flags_and_leaves_of_and_the_rest() {
        let mut pair = Pair::new(code::SAHF);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xd500; // SF ZF AF PF CF
            h.cpu.set_eflags_arith(OF);
            h.cpu.eflags |= DF;
        });
        assert_flags(
            pair.flags(),
            ALL_ARITH | DF,
            OF | SF | ZF | AF | PF | CF | DF,
            "sahf sets",
        );
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0x2a00; // none of the five flag bits (bit 1 set, bits 3 and 5 junk)
            h.cpu.set_eflags_arith(ALL_ARITH);
        });
        assert_flags(
            pair.flags(),
            ALL_ARITH,
            OF,
            "sahf clears the five, OF stays",
        );
    }

    #[test]
    fn carry_flag_and_direction_flag_instructions() {
        let flags_in = |h: &mut Harness| h.cpu.set_eflags_arith(ZF | OF);
        let mut pair = Pair::new(code::STC);
        pair.halted(flags_in);
        assert_flags(pair.flags(), ALL_ARITH, ZF | OF | CF, "stc");
        let mut pair = Pair::new(code::CLC);
        pair.halted(|h| h.cpu.set_eflags_arith(ALL_ARITH));
        assert_flags(pair.flags(), ALL_ARITH, ALL_ARITH & !CF, "clc");
        let mut pair = Pair::new(code::CMC);
        pair.halted(flags_in);
        assert_flags(pair.flags(), ALL_ARITH, ZF | OF | CF, "cmc sets");
        pair.halted(|h| h.cpu.set_eflags_arith(ZF | OF | CF));
        assert_flags(pair.flags(), ALL_ARITH, ZF | OF, "cmc clears");
        let mut pair = Pair::new(code::STD);
        pair.halted(flags_in);
        assert_flags(pair.flags(), DF | ALL_ARITH, DF | ZF | OF, "std");
        let mut pair = Pair::new(code::CLD);
        pair.halted(|h| h.cpu.eflags |= DF);
        assert_flags(pair.flags(), DF, 0, "cld");
    }

    #[test]
    fn movnti_stores_and_rejects_a_register_destination() {
        let mut pair = Pair::new(code::MOVNTI_MEM32);
        pair.halted(|h| {
            memory_with_marker(h, 0, 0);
            h.cpu.gpr[RAX] = 0x1122_3344_5566_7788;
        });
        assert_eq!(
            pair.interp.read(DATA, 8),
            [
                0x88,
                0x77,
                0x66,
                0x55,
                MEMORY_MARKER,
                MEMORY_MARKER,
                MEMORY_MARKER,
                MEMORY_MARKER
            ]
        );
        let mut pair = Pair::new(code::MOVNTI_MEM64);
        pair.halted(|h| {
            memory_with_marker(h, 0, 0);
            h.cpu.gpr[RAX] = 0x1122_3344_5566_7788;
        });
        assert_eq!(pair.interp.read_u64(DATA), 0x1122_3344_5566_7788);
        let (_, stop) = run_regs_stop(code::MOVNTI_REG, &[]);
        assert_eq!(stop, ud());
    }

    /// Where a return frame sits on the stack, the stack the return installs, and the instruction
    /// after the 2-byte `iretq`/`retfq` (a `hlt` in every snippet).
    const FRAME: u64 = STACK_TOP - 0x200;
    const NEW_RSP: u64 = STACK_TOP - 0x800;
    const LANDING: u64 = CODE_BASE + 2;
    const RFLAGS_IMAGE: u64 = 0x246;

    #[derive(Clone, Copy, Debug)]
    enum ReturnKind {
        Iretq,
        FarReturn,
    }

    impl ReturnKind {
        fn code(self) -> &'static [u8] {
            match self {
                ReturnKind::Iretq => code::IRETQ,
                ReturnKind::FarReturn => code::RETFQ,
            }
        }

        /// Stack frame in the layout the instruction reads. The far-return layout is also correct for
        /// same-privilege returns, which just never read the last two words.
        fn write_frame(self, h: &mut Harness, cs: u16, ss: u16) {
            h.cpu.gpr[RSP] = FRAME;
            match self {
                ReturnKind::Iretq => put_words(
                    h,
                    FRAME,
                    &[LANDING, cs as u64, RFLAGS_IMAGE, NEW_RSP, ss as u64],
                ),
                ReturnKind::FarReturn => {
                    put_words(h, FRAME, &[LANDING, cs as u64, NEW_RSP, ss as u64])
                }
            }
        }

        /// Stack pointer after a same-privilege return.
        fn same_privilege_rsp(self) -> u64 {
            match self {
                ReturnKind::Iretq => NEW_RSP,
                ReturnKind::FarReturn => FRAME + 16,
            }
        }
    }

    const BOTH_RETURNS: [ReturnKind; 2] = [ReturnKind::Iretq, ReturnKind::FarReturn];

    #[track_caller]
    fn check_return_fault(kind: ReturnKind, start_cpl: u8, cs: u16, ss: u16, expected: Stop) {
        assert_fault_changes_nothing(
            kind.code(),
            |h| {
                enter_cpl(h, start_cpl);
                kind.write_frame(h, cs, ss);
            },
            expected,
        );
    }

    /// Runs the return one instruction and returns the CPU state afterwards.
    #[track_caller]
    fn run_return(
        kind: ReturnKind,
        start_cpl: u8,
        cs: u16,
        ss: u16,
        extra: impl Fn(&mut Harness),
    ) -> Pair {
        let mut pair = Pair::new(kind.code());
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, start_cpl);
            extra(h);
            kind.write_frame(h, cs, ss);
        });
        assert_eq!(stop, Stop::Limit, "{kind:?}: {}", pair.cpu().dump_regs());
        pair
    }

    #[test]
    fn return_to_the_same_privilege_loads_cs_rip_and_rsp() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_CONFORMING_DPL0, SEL_KDATA, |_| {});
            let cpu = pair.cpu();
            assert_eq!(cpu.rip, LANDING, "{kind:?}");
            assert_eq!(cpu.seg[SEG_CS].sel, SEL_CONFORMING_DPL0, "{kind:?}");
            assert!(
                cpu.seg[SEG_CS].is_code() && cpu.seg[SEG_CS].long(),
                "{kind:?}"
            );
            assert_eq!(cpu.cpl, 0, "{kind:?}");
            assert_eq!(cpu.gpr[RSP], kind.same_privilege_rsp(), "{kind:?}");
        }
    }

    #[test]
    fn return_to_a_32_bit_code_segment_enters_compatibility_mode() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_CODE32, SEL_KDATA, |_| {});
            let cpu = pair.cpu();
            assert_eq!(cpu.rip, LANDING, "{kind:?}");
            assert_eq!(cpu.seg[SEG_CS].sel, SEL_CODE32, "{kind:?}");
            assert_ne!(cpu.efer & EFER_LMA, 0, "{kind:?}");
            assert_eq!(cpu.code_mode(), CodeMode::Bits32, "{kind:?}");
        }
    }

    #[test]
    fn iretq_in_64_bit_mode_always_reloads_ss() {
        // Even without a privilege change the five-word frame is popped and SS is loaded from it.
        let pair = run_return(ReturnKind::Iretq, 0, SEL_KCODE, SEL_KDATA, |h| {
            h.cpu.seg[SEG_SS] = Seg::default();
        });
        assert_eq!(pair.cpu().seg[SEG_SS].sel, SEL_KDATA);
        assert!(pair.cpu().seg[SEG_SS].present());
    }

    #[test]
    fn far_return_in_the_same_privilege_does_not_touch_ss() {
        let pair = run_return(ReturnKind::FarReturn, 0, SEL_KCODE, SEL_UDATA | 3, |_| {});
        assert_eq!(pair.cpu().seg[SEG_SS].sel, SEL_KDATA);
    }

    #[test]
    fn far_return_releases_the_immediate_after_the_return_address() {
        let mut pair = Pair::new(code::RETFQ_IMM);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RSP] = FRAME;
            put_words(h, FRAME, &[LANDING + 2, SEL_KCODE as u64]);
        });
        assert_eq!(stop, Stop::Limit);
        assert_eq!(pair.cpu().rip, LANDING + 2);
        assert_eq!(pair.reg(RSP), FRAME + 16 + 16);
    }

    #[test]
    fn iretq_to_user_mode_switches_privilege_stack_and_nulls_inaccessible_segments() {
        let pair = run_return(ReturnKind::Iretq, 0, SEL_UCODE | 3, SEL_UDATA | 3, |h| {
            let kernel_data = gdt_segment(h, SEL_KDATA);
            let user_data = gdt_segment(h, SEL_UDATA);
            h.cpu.seg[SEG_DS] = kernel_data;
            h.cpu.seg[SEG_ES] = kernel_data;
            h.cpu.seg[SEG_FS] = kernel_data;
            h.cpu.seg[SEG_FS].base = 0x1234;
            h.cpu.seg[SEG_GS] = user_data;
            h.cpu.seg[SEG_GS].sel = SEL_UDATA | 3;
            h.cpu.seg[SEG_GS].base = 0x5678;
        });
        let cpu = pair.cpu();
        assert_eq!(cpu.cpl, 3);
        assert_eq!(cpu.rip, LANDING);
        assert_eq!(cpu.seg[SEG_CS].sel, SEL_UCODE | 3);
        assert_eq!(cpu.seg[SEG_SS].sel, SEL_UDATA | 3);
        assert_eq!(cpu.gpr[RSP], NEW_RSP);
        for register in [SEG_DS, SEG_ES, SEG_FS] {
            assert_eq!(
                cpu.seg[register].sel, 0,
                "register {register} had DPL 0 and must be nulled"
            );
            assert!(!cpu.seg[register].present(), "register {register}");
        }
        assert_eq!(
            cpu.seg[SEG_GS].sel,
            SEL_UDATA | 3,
            "DPL 3 data stays usable"
        );
        assert_eq!(cpu.seg[SEG_GS].base, 0x5678);
    }

    #[test]
    fn far_return_to_user_mode_switches_privilege_stack_and_nulls_inaccessible_segments() {
        let pair = run_return(
            ReturnKind::FarReturn,
            0,
            SEL_UCODE | 3,
            SEL_UDATA | 3,
            |h| {
                let kernel_data = gdt_segment(h, SEL_KDATA);
                let user_data = gdt_segment(h, SEL_UDATA);
                h.cpu.seg[SEG_DS] = kernel_data;
                h.cpu.seg[SEG_ES] = user_data;
                h.cpu.seg[SEG_ES].sel = SEL_UDATA | 3;
                h.cpu.seg[SEG_FS] = kernel_data;
                h.cpu.seg[SEG_GS] = kernel_data;
            },
        );
        let cpu = pair.cpu();
        assert_eq!(cpu.cpl, 3);
        assert_eq!(cpu.rip, LANDING);
        assert_eq!(cpu.seg[SEG_CS].sel, SEL_UCODE | 3);
        assert_eq!(cpu.seg[SEG_SS].sel, SEL_UDATA | 3);
        assert_eq!(cpu.gpr[RSP], NEW_RSP);
        for register in [SEG_DS, SEG_FS, SEG_GS] {
            assert_eq!(cpu.seg[register].sel, 0, "register {register}");
        }
        assert_eq!(cpu.seg[SEG_ES].sel, SEL_UDATA | 3);
    }

    #[test]
    fn far_return_to_an_outer_ring_also_releases_the_immediate_from_the_new_stack() {
        // SDM: the parameters are released from both the called and the calling procedure's stack.
        let mut pair = Pair::new(code::RETFQ_IMM);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RSP] = FRAME;
            // rip, cs, 16 bytes of parameters, then rsp and ss.
            put_words(
                h,
                FRAME,
                &[
                    LANDING + 2,
                    (SEL_UCODE | 3) as u64,
                    0,
                    0,
                    NEW_RSP,
                    (SEL_UDATA | 3) as u64,
                ],
            );
        });
        assert_eq!(stop, Stop::Limit);
        assert_eq!(pair.cpu().cpl, 3);
        assert_eq!(pair.reg(RSP), NEW_RSP + 16);
    }

    #[test]
    fn data_segment_nulling_covers_non_conforming_code_but_not_conforming_code() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_UCODE | 3, SEL_UDATA | 3, |h| {
                h.cpu.seg[SEG_DS] = gdt_segment(h, SEL_KCODE); // readable non-conforming, DPL 0
                h.cpu.seg[SEG_ES] = gdt_segment(h, SEL_CONFORMING_DPL0);
                h.cpu.seg[SEG_FS] = Seg::default(); // already null stays null
                h.cpu.seg[SEG_FS].base = 0x99;
            });
            let cpu = pair.cpu();
            assert_eq!(
                cpu.seg[SEG_DS].sel, 0,
                "{kind:?}: non-conforming code with DPL < new CPL"
            );
            assert_eq!(
                cpu.seg[SEG_ES].sel, SEL_CONFORMING_DPL0,
                "{kind:?}: conforming code survives"
            );
            assert_eq!(
                cpu.seg[SEG_FS].base, 0x99,
                "{kind:?}: a null FS keeps its base"
            );
        }
    }

    #[test]
    fn return_to_ring_1_nulls_only_dpl_0_segments() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_CODE_DPL1 | 1, SEL_DATA_DPL1 | 1, |h| {
                h.cpu.seg[SEG_DS] = gdt_segment(h, SEL_KDATA);
                h.cpu.seg[SEG_ES] = gdt_segment(h, SEL_DATA_DPL1);
                h.cpu.seg[SEG_ES].sel = SEL_DATA_DPL1;
            });
            let cpu = pair.cpu();
            assert_eq!(cpu.cpl, 1, "{kind:?}");
            assert_eq!(cpu.seg[SEG_DS].sel, 0, "{kind:?}");
            assert_eq!(
                cpu.seg[SEG_ES].sel, SEL_DATA_DPL1,
                "{kind:?}: DPL 1 is accessible at CPL 1"
            );
        }
    }

    #[test]
    fn return_to_ring_3_accepts_a_conforming_segment_whose_dpl_is_not_above_the_rpl() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_CONFORMING_DPL3 | 3, SEL_UDATA | 3, |_| {});
            assert_eq!(pair.cpu().cpl, 3, "{kind:?}");
            assert_eq!(pair.cpu().seg[SEG_CS].sel, SEL_CONFORMING_DPL3 | 3);
            let pair = run_return(kind, 0, SEL_CONFORMING_DPL0 | 3, SEL_UDATA | 3, |_| {});
            assert_eq!(pair.cpu().cpl, 3, "{kind:?}: conforming DPL 0 <= RPL 3");
        }
    }

    #[test]
    fn return_code_segment_checks_raise_the_architectural_fault_and_change_nothing() {
        for kind in BOTH_RETURNS {
            let ss = SEL_KDATA;
            // (start CPL, return CS, expected fault)
            let cases = [
                (0, SEL_NULL, gp(0)),
                (0, SEL_NULL | 3, gp(0)),
                (0, SEL_BEYOND_GDT, gp(selector_code(SEL_BEYOND_GDT))),
                (0, SEL_KDATA, gp(selector_code(SEL_KDATA))),
                (0, SEL_SYSTEM, gp(selector_code(SEL_SYSTEM))),
                (0, SEL_UCODE, gp(selector_code(SEL_UCODE))),
                (0, SEL_KCODE | 3, gp(selector_code(SEL_KCODE))),
                (
                    0,
                    SEL_CONFORMING_DPL3,
                    gp(selector_code(SEL_CONFORMING_DPL3)),
                ),
                (
                    0,
                    SEL_CODE_NOT_PRESENT,
                    np(selector_code(SEL_CODE_NOT_PRESENT)),
                ),
                (1, SEL_KCODE, gp(selector_code(SEL_KCODE))),
                (1, SEL_CODE_DPL1, gp(selector_code(SEL_CODE_DPL1))),
                (2, SEL_CODE_DPL1 | 1, gp(selector_code(SEL_CODE_DPL1))),
                (3, SEL_CODE_DPL2 | 2, gp(selector_code(SEL_CODE_DPL2))),
            ];
            for (cpl, cs, expected) in cases {
                check_return_fault(kind, cpl, cs, ss, expected);
            }
        }
    }

    #[test]
    fn return_stack_segment_checks_raise_the_architectural_fault_and_change_nothing() {
        for kind in BOTH_RETURNS {
            // Returning to ring 3: SS must be a present writable data segment with RPL = DPL = 3.
            let to_user = SEL_UCODE | 3;
            let cases = [
                (SEL_NULL, gp(0)),
                (SEL_NULL | 3, gp(0)),
                (SEL_UDATA, gp(selector_code(SEL_UDATA))),
                (SEL_KDATA | 3, gp(selector_code(SEL_KDATA))),
                (SEL_UCODE | 3, gp(selector_code(SEL_UCODE))),
                (SEL_BEYOND_GDT | 3, gp(selector_code(SEL_BEYOND_GDT))),
                (SEL_SYSTEM | 3, gp(selector_code(SEL_SYSTEM))),
            ];
            for (ss, expected) in cases {
                check_return_fault(kind, 0, to_user, ss, expected);
            }
            check_return_fault(
                kind,
                0,
                to_user,
                SEL_UDATA_NOT_PRESENT | 3,
                ss_fault(selector_code(SEL_UDATA_NOT_PRESENT)),
            );
        }
        // iretq reloads SS even without a privilege change, so it also checks it there.
        let to_kernel = SEL_KCODE;
        let kind = ReturnKind::Iretq;
        check_return_fault(
            kind,
            0,
            to_kernel,
            SEL_RODATA,
            gp(selector_code(SEL_RODATA)),
        );
        check_return_fault(kind, 0, to_kernel, SEL_KCODE, gp(selector_code(SEL_KCODE)));
        check_return_fault(
            kind,
            0,
            to_kernel,
            SEL_DATA_NOT_PRESENT,
            ss_fault(selector_code(SEL_DATA_NOT_PRESENT)),
        );
    }

    #[test]
    fn return_does_not_check_ss_without_a_stack_switch_for_far_return() {
        // A far return within the same privilege never looks at the words above CS.
        let pair = run_return(ReturnKind::FarReturn, 0, SEL_KCODE, SEL_NULL, |_| {});
        assert_eq!(pair.cpu().rip, LANDING);
    }

    #[test]
    fn return_with_a_missing_stack_segment_leaves_cs_and_cpl_alone() {
        // The CS load would succeed, so a naive implementation changes CS and CPL before failing.
        for kind in BOTH_RETURNS {
            let mut pair = Pair::new(kind.code());
            let stop = pair.run_for(1, |h| {
                enter_cpl(h, 0);
                kind.write_frame(h, SEL_UCODE | 3, SEL_UDATA_NOT_PRESENT | 3);
            });
            assert_eq!(
                stop,
                ss_fault(selector_code(SEL_UDATA_NOT_PRESENT)),
                "{kind:?}"
            );
            let cpu = pair.cpu();
            assert_eq!(cpu.cpl, 0, "{kind:?}");
            assert_eq!(cpu.seg[SEG_CS].sel, SEL_KCODE, "{kind:?}");
            assert_eq!(cpu.rip, CODE_BASE, "{kind:?}");
            assert_eq!(cpu.gpr[RSP], FRAME, "{kind:?}");
        }
    }

    #[test]
    fn null_stack_segment_is_allowed_when_returning_to_a_ring_below_3() {
        for kind in BOTH_RETURNS {
            let pair = run_return(kind, 0, SEL_CODE_DPL1 | 1, SEL_NULL | 1, |_| {});
            assert_eq!(pair.cpu().cpl, 1, "{kind:?}");
            assert_eq!(pair.cpu().seg[SEG_SS].sel, 1, "{kind:?}");
            assert_eq!(pair.cpu().gpr[RSP], NEW_RSP, "{kind:?}");
        }
    }

    #[test]
    fn iretq_flags_follow_the_privilege_it_started_at() {
        let image_with = |extra: u64| RFLAGS_IMAGE | extra;
        let run_flags = |start_cpl: u8, iopl: u64, image: u64| {
            let (cs, ss) = match start_cpl {
                0 => (SEL_KCODE, SEL_KDATA),
                _ => (SEL_CODE_DPL1 | 1, SEL_DATA_DPL1 | 1),
            };
            let mut pair = Pair::new(code::IRETQ);
            let stop = pair.run_for(1, |h| {
                enter_cpl(h, start_cpl);
                h.cpu.eflags = RESERVED_ONE | iopl << IOPL_SHIFT;
                ReturnKind::Iretq.write_frame(h, cs, ss);
                h.write_u64(FRAME + 16, image);
            });
            assert_eq!(stop, Stop::Limit);
            pair.cpu().eflags_full()
        };
        // CPL 0: IF and IOPL are both loaded.
        let f = run_flags(0, 0, image_with(IF | (3 << IOPL_SHIFT)));
        assert_eq!(f & IF, IF);
        assert_eq!(f & IOPL, IOPL);
        // CPL 1 with IOPL 0: neither IF nor IOPL changes.
        let f = run_flags(1, 0, image_with(IF | (3 << IOPL_SHIFT)));
        assert_eq!(f & IF, 0);
        assert_eq!(f & IOPL, 0);
        // CPL 1 with IOPL 1: IF is loadable, IOPL still is not.
        let f = run_flags(1, 1, image_with(IF | (3 << IOPL_SHIFT)));
        assert_eq!(f & IF, IF);
        assert_eq!(f & IOPL, 1 << IOPL_SHIFT);
        // RF and VM are never loaded; the arithmetic flags and DF/TF/NT/AC/ID always are.
        let f = run_flags(
            0,
            0,
            image_with(RF | VM | DF | TF | NT | AC | EFLAGS_ID | CF | OF),
        );
        assert_eq!(f & (RF | VM), 0);
        assert_eq!(
            f & (DF | TF | NT | AC | EFLAGS_ID | CF | OF),
            DF | TF | NT | AC | EFLAGS_ID | CF | OF
        );
        assert_eq!(f & RESERVED_ONE, RESERVED_ONE);
    }

    #[test]
    fn iretq_and_far_return_near_the_top_of_the_address_space_wrap_instead_of_overflowing() {
        // The frame straddles 2^64: the last word is read from linear address 0.
        let top = TOP_PAGE_LINEAR + 0xfe0;
        let mut pair = Pair::new(code::IRETQ);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            map_top_page(h);
            h.cpu.gpr[RSP] = top;
            for (index, word) in [LANDING, SEL_KCODE as u64, RFLAGS_IMAGE, NEW_RSP]
                .iter()
                .enumerate()
            {
                h.write_u64(top_page_physical(top) + index as u64 * 8, *word);
            }
            h.write_u64(0, SEL_KDATA as u64);
        });
        assert_eq!(stop, Stop::Limit, "{}", pair.cpu().dump_regs());
        assert_eq!(pair.reg(RSP), NEW_RSP);
        assert_eq!(pair.cpu().rip, LANDING);

        // retfq: rip and cs at the last two words; the new rsp wraps to 0.
        let top = TOP_PAGE_LINEAR + 0xff0;
        let mut pair = Pair::new(code::RETFQ);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            map_top_page(h);
            h.cpu.gpr[RSP] = top;
            h.write_u64(top_page_physical(top), LANDING);
            h.write_u64(top_page_physical(top) + 8, SEL_KCODE as u64);
        });
        assert_eq!(stop, Stop::Limit);
        assert_eq!(
            pair.reg(RSP),
            0,
            "rsp + 16 wraps around the end of the address space"
        );

        // retfq to an outer ring: the rsp and ss words are read past the wrap.
        let top = TOP_PAGE_LINEAR + 0xfe8;
        let mut pair = Pair::new(code::RETFQ_IMM);
        let stop = pair.run_for(1, |h| {
            map_top_page(h);
            enter_cpl(h, 0);
            h.cpu.gpr[RSP] = top;
            h.write_u64(top_page_physical(top), LANDING + 2);
            h.write_u64(top_page_physical(top) + 8, (SEL_UCODE | 3) as u64);
            // imm = 16: rsp at top + 16 + 16 wraps to 0x8, ss to 0x10.
            h.write_u64(8, NEW_RSP);
            h.write_u64(16, (SEL_UDATA | 3) as u64);
        });
        assert_eq!(stop, Stop::Limit, "{}", pair.cpu().dump_regs());
        assert_eq!(pair.cpu().cpl, 3);
        assert_eq!(pair.reg(RSP), NEW_RSP + 16);
    }

    #[test]
    fn iret_and_far_return_without_rex_w_are_not_implemented() {
        for snippet in [code::IRETD, code::RETF_32] {
            let (_, stop) = run_regs_stop(snippet, &[]);
            assert_eq!(stop, ud(), "{snippet:02x?}");
        }
    }

    #[test]
    fn iretq_page_fault_reading_the_frame_changes_nothing() {
        assert_fault_changes_nothing(
            code::IRETQ,
            |h| {
                unmap_huge_page(h, HOLE);
                h.cpu.gpr[RSP] = HOLE + 0x100;
            },
            fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)),
        );
        // The first three words (with a valid CS) are readable; the last two are on the missing page.
        assert_fault_changes_nothing(
            code::IRETQ,
            |h| {
                enter_cpl(h, 0);
                let frame = HOLE + HUGE_PAGE_BYTES - 0x18;
                put_words(h, frame, &[LANDING, SEL_KCODE as u64, RFLAGS_IMAGE]);
                unmap_huge_page(h, HOLE + HUGE_PAGE_BYTES);
                h.cpu.gpr[RSP] = frame;
            },
            fault(VECTOR_PAGE_FAULT, Some(PF_READ_NOT_PRESENT)),
        );
    }

    fn mov_to_sreg(snippet: &[u8], cpl: u8, selector: u16) -> (Pair, Stop) {
        let mut pair = Pair::new(snippet);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, cpl);
            h.cpu.gpr[RAX] = selector as u64 | 0xffff_ffff_0000_0000;
        });
        (pair, stop)
    }

    #[test]
    fn mov_to_data_segment_register_privilege_and_type_checks() {
        // (cpl, selector, expected result: Ok or the fault)
        let cases: [(u8, u16, Result<(), Stop>); 18] = [
            (0, SEL_NULL, Ok(())),
            (0, SEL_NULL | 3, Ok(())),
            (3, SEL_NULL, Ok(())),
            (0, SEL_KDATA, Ok(())),
            (0, SEL_UDATA, Ok(())),
            (0, SEL_KDATA | 3, Err(gp(selector_code(SEL_KDATA)))),
            (1, SEL_KDATA | 1, Err(gp(selector_code(SEL_KDATA)))),
            (3, SEL_KDATA | 3, Err(gp(selector_code(SEL_KDATA)))),
            (3, SEL_UDATA | 3, Ok(())),
            (3, SEL_UDATA, Ok(())),
            (3, SEL_CONFORMING_DPL0 | 3, Ok(())),
            (0, SEL_KCODE, Ok(())),
            (
                0,
                SEL_CODE_EXEC_ONLY,
                Err(gp(selector_code(SEL_CODE_EXEC_ONLY))),
            ),
            (0, SEL_SYSTEM, Err(gp(selector_code(SEL_SYSTEM)))),
            (0, SEL_BEYOND_GDT, Err(gp(selector_code(SEL_BEYOND_GDT)))),
            (
                0,
                SEL_DATA_NOT_PRESENT,
                Err(np(selector_code(SEL_DATA_NOT_PRESENT))),
            ),
            (
                0,
                SEL_CODE_NOT_PRESENT,
                Err(np(selector_code(SEL_CODE_NOT_PRESENT))),
            ),
            (2, SEL_DATA_DPL1 | 1, Err(gp(selector_code(SEL_DATA_DPL1)))),
        ];
        for (snippet, register) in [
            (code::MOV_DS_AX, SEG_DS),
            (code::MOV_ES_AX, SEG_ES),
            (code::MOV_FS_AX, SEG_FS),
            (code::MOV_GS_AX, SEG_GS),
        ] {
            for (cpl, selector, expected) in &cases {
                let (pair, stop) = mov_to_sreg(snippet, *cpl, *selector);
                let context = format!("register {register} cpl {cpl} selector {selector:#x}");
                match expected {
                    Ok(()) => {
                        assert_eq!(stop, Stop::Limit, "{context}");
                        assert_eq!(pair.cpu().seg[register].sel, *selector, "{context}");
                    }
                    Err(fault_stop) => {
                        assert_eq!(&stop, fault_stop, "{context}");
                        assert_ne!(
                            pair.cpu().seg[register].sel,
                            *selector,
                            "{context}: not loaded"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mov_to_ss_checks_rpl_dpl_type_and_presence() {
        // (cpl, selector, expected)
        let cases: [(u8, u16, Result<(), Stop>); 17] = [
            (0, SEL_KDATA, Ok(())),
            (1, SEL_DATA_DPL1 | 1, Ok(())),
            (2, SEL_DATA_DPL2 | 2, Ok(())),
            (3, SEL_UDATA | 3, Ok(())),
            (1, SEL_KDATA | 1, Err(gp(selector_code(SEL_KDATA)))),
            (1, SEL_DATA_DPL1, Err(gp(selector_code(SEL_DATA_DPL1)))),
            (0, SEL_DATA_DPL1 | 1, Err(gp(selector_code(SEL_DATA_DPL1)))),
            (0, SEL_KDATA | 1, Err(gp(selector_code(SEL_KDATA)))),
            (0, SEL_KCODE, Err(gp(selector_code(SEL_KCODE)))),
            (0, SEL_RODATA, Err(gp(selector_code(SEL_RODATA)))),
            (0, SEL_SYSTEM, Err(gp(selector_code(SEL_SYSTEM)))),
            (0, SEL_BEYOND_GDT, Err(gp(selector_code(SEL_BEYOND_GDT)))),
            (
                0,
                SEL_DATA_NOT_PRESENT,
                Err(ss_fault(selector_code(SEL_DATA_NOT_PRESENT))),
            ),
            (0, SEL_NULL, Ok(())),
            (1, SEL_NULL | 1, Ok(())),
            (1, SEL_NULL, Err(gp(0))),
            (3, SEL_NULL | 3, Err(gp(0))),
        ];
        for (cpl, selector, expected) in cases {
            let (pair, stop) = mov_to_sreg(code::MOV_SS_AX, cpl, selector);
            let context = format!("cpl {cpl} selector {selector:#x}");
            match expected {
                Ok(()) => {
                    assert_eq!(stop, Stop::Limit, "{context}");
                    assert_eq!(pair.cpu().seg[SEG_SS].sel, selector, "{context}");
                }
                Err(fault_stop) => {
                    assert_eq!(stop, fault_stop, "{context}");
                    assert_ne!(
                        pair.cpu().seg[SEG_SS].sel,
                        selector,
                        "{context}: not loaded"
                    );
                }
            }
        }
    }

    #[test]
    fn mov_to_ss_blocks_interrupts_for_one_instruction() {
        let mut pair = Pair::new(code::MOV_SS_AX);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RAX] = SEL_KDATA as u64;
        });
        assert_eq!(stop, Stop::Limit);
        assert!(pair.cpu().interrupt_shadow);
    }

    #[test]
    fn mov_to_a_data_segment_does_not_open_an_interrupt_shadow() {
        let mut pair = Pair::new(code::MOV_DS_AX);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RAX] = SEL_KDATA as u64;
        });
        assert_eq!(stop, Stop::Limit);
        assert!(!pair.cpu().interrupt_shadow);
    }

    #[test]
    fn failed_mov_to_ss_opens_no_interrupt_shadow_and_keeps_ss() {
        let mut pair = Pair::new(code::MOV_SS_AX);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RAX] = SEL_UDATA as u64;
        });
        assert_eq!(stop, gp(selector_code(SEL_UDATA)));
        assert!(!pair.cpu().interrupt_shadow);
        assert_eq!(pair.cpu().seg[SEG_SS].sel, SEL_KDATA);
    }

    #[test]
    fn mov_to_a_data_segment_from_a_null_selector_clears_the_cache() {
        let mut pair = Pair::new(code::MOV_FS_AX);
        pair.halted(|h| {
            enter_cpl(h, 0);
            h.cpu.seg[SEG_FS] = gdt_segment(h, SEL_KDATA);
            h.cpu.seg[SEG_FS].base = 0x1000;
            h.cpu.gpr[RAX] = 0;
        });
        assert_eq!(pair.cpu().seg[SEG_FS].sel, 0);
        assert!(!pair.cpu().seg[SEG_FS].present());
    }

    /// CPUID 0x8000_0021 EAX bit 6 (NullSelectorClearsBase) promises this: in 64-bit mode a null
    /// selector loaded by `mov` or `pop` leaves base 0, including the FS and GS bases a wrmsr or
    /// wrfsbase set, at any CPL and with any RPL. Linux skips its X86_BUG_NULL_SEG workaround on
    /// the strength of it.
    #[test]
    fn a_null_selector_load_clears_the_segment_base() {
        for (snippet, register) in [
            (code::MOV_DS_AX, SEG_DS),
            (code::MOV_ES_AX, SEG_ES),
            (code::MOV_FS_AX, SEG_FS),
            (code::MOV_GS_AX, SEG_GS),
        ] {
            for (cpl, selector) in [(0, SEL_NULL), (3, SEL_NULL | 3), (0, SEL_NULL | 1)] {
                let mut pair = Pair::new(snippet);
                let stop = pair.run_for(1, |h| {
                    enter_cpl(h, cpl);
                    h.cpu.seg[register] = gdt_segment(h, SEL_UDATA);
                    h.cpu.seg[register].base = 0x7fff_1234_5000;
                    h.cpu.gpr[RAX] = selector as u64;
                });
                assert_eq!(stop, Stop::Limit);
                let segment = pair.cpu().seg[register];
                assert_eq!(
                    (segment.sel, segment.base),
                    (selector, 0),
                    "{register} cpl {cpl}"
                );
            }
        }
        for (snippet, register) in [(code::POP_FS, SEG_FS), (code::POP_GS, SEG_GS)] {
            let mut pair = Pair::new(snippet);
            let stop = pair.run_for(1, |h| {
                enter_cpl(h, 0);
                h.cpu.seg[register].base = 0xffff_8000_0000_1000;
                stack_with(h, 0x40, SEL_NULL as u64);
            });
            assert_eq!(stop, Stop::Limit);
            assert_eq!(pair.cpu().seg[register].base, 0, "pop into {register}");
        }
    }

    #[test]
    fn mov_to_a_data_segment_loads_base_and_limit_from_the_descriptor() {
        let mut pair = Pair::new(code::MOV_ES_AX);
        pair.halted(|h| {
            enter_cpl(h, 0);
            h.cpu.gpr[RAX] = SEL_KDATA as u64;
        });
        let es = pair.cpu().seg[SEG_ES];
        assert_eq!(es.sel, SEL_KDATA);
        assert_eq!(es.limit, 0xffff_ffff);
        assert!(es.present());
        assert_eq!(es.dpl(), 0);
    }
    const SHIFT_SNIPPETS: &[ShiftSnippet] = &[
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::ROL_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::ROL_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::ROL_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::ROL_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::ROL_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::ROL_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::ROL_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::ROL_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::ROL_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::ROL_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::ROL_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::ROL_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::ROL_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::ROL_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::ROL_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::ROL_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::ROL_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::ROL_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::ROL_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::ROL_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::ROL_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::ROL_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::ROL_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::ROL_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::ROL_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::ROL_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::ROL_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROL,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::ROL_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::ROR_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::ROR_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::ROR_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::ROR_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::ROR_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::ROR_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::ROR_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::ROR_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::ROR_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::ROR_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::ROR_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::ROR_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::ROR_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::ROR_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::ROR_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::ROR_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::ROR_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::ROR_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::ROR_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::ROR_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::ROR_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::ROR_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::ROR_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::ROR_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::ROR_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::ROR_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::ROR_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_ROR,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::ROR_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::RCL_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::RCL_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::RCL_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::RCL_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::RCL_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::RCL_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::RCL_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::RCL_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::RCL_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::RCL_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::RCL_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::RCL_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::RCL_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::RCL_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::RCL_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::RCL_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::RCL_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::RCL_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::RCL_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::RCL_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::RCL_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::RCL_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::RCL_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::RCL_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::RCL_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::RCL_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::RCL_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCL,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::RCL_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::RCR_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::RCR_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::RCR_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::RCR_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::RCR_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::RCR_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::RCR_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::RCR_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::RCR_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::RCR_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::RCR_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::RCR_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::RCR_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::RCR_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::RCR_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::RCR_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::RCR_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::RCR_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::RCR_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::RCR_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::RCR_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::RCR_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::RCR_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::RCR_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::RCR_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::RCR_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::RCR_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_RCR,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::RCR_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::SHL_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::SHL_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::SHL_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::SHL_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::SHL_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::SHL_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::SHL_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::SHL_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::SHL_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::SHL_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::SHL_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::SHL_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::SHL_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::SHL_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::SHL_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::SHL_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::SHL_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::SHL_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::SHL_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::SHL_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::SHL_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::SHL_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::SHL_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::SHL_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::SHL_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::SHL_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::SHL_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHL,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::SHL_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::SHR_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::SHR_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::SHR_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::SHR_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::SHR_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::SHR_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::SHR_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::SHR_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::SHR_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::SHR_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::SHR_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::SHR_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::SHR_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::SHR_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::SHR_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::SHR_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::SHR_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::SHR_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::SHR_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::SHR_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::SHR_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::SHR_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::SHR_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::SHR_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::SHR_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::SHR_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::SHR_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SHR,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::SHR_1_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Cl,
            memory: false,
            code: code::SAR_8_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Imm(1),
            memory: false,
            code: code::SAR_8_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Imm(3),
            memory: false,
            code: code::SAR_8_I3,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Imm(17),
            memory: false,
            code: code::SAR_8_I17,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Imm(33),
            memory: false,
            code: code::SAR_8_I33,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Cl,
            memory: true,
            code: code::SAR_8_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 8,
            count: Count::Imm(1),
            memory: true,
            code: code::SAR_8_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Cl,
            memory: false,
            code: code::SAR_4_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Imm(1),
            memory: false,
            code: code::SAR_4_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Imm(3),
            memory: false,
            code: code::SAR_4_I3,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Imm(17),
            memory: false,
            code: code::SAR_4_I17,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Imm(33),
            memory: false,
            code: code::SAR_4_I33,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Cl,
            memory: true,
            code: code::SAR_4_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 4,
            count: Count::Imm(1),
            memory: true,
            code: code::SAR_4_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Cl,
            memory: false,
            code: code::SAR_2_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Imm(1),
            memory: false,
            code: code::SAR_2_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Imm(3),
            memory: false,
            code: code::SAR_2_I3,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Imm(17),
            memory: false,
            code: code::SAR_2_I17,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Imm(33),
            memory: false,
            code: code::SAR_2_I33,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Cl,
            memory: true,
            code: code::SAR_2_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 2,
            count: Count::Imm(1),
            memory: true,
            code: code::SAR_2_MEM_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Cl,
            memory: false,
            code: code::SAR_1_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Imm(1),
            memory: false,
            code: code::SAR_1_ONE,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Imm(3),
            memory: false,
            code: code::SAR_1_I3,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Imm(17),
            memory: false,
            code: code::SAR_1_I17,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Imm(33),
            memory: false,
            code: code::SAR_1_I33,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Cl,
            memory: true,
            code: code::SAR_1_MEM_CL,
        },
        ShiftSnippet {
            op: SHIFT_SAR,
            size: 1,
            count: Count::Imm(1),
            memory: true,
            code: code::SAR_1_MEM_ONE,
        },
    ];

    const SHIFT_SAR: u8 = 7;

    #[derive(Clone, Copy, Debug)]
    enum Count {
        Cl,
        Imm(u64),
    }

    #[derive(Clone, Copy, Debug)]
    struct ShiftSnippet {
        op: u8,
        size: u8,
        count: Count,
        memory: bool,
        code: &'static [u8],
    }

    const SHIFT_COUNTS_FOR_CL: [u64; 22] = [
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        35,
        63,
        64,
        65,
        71,
        100,
        255,
        256 + 3,
    ];

    struct ShiftReference {
        result: u64,
        flags: u64,
        defined: u64,
    }

    /// One bit position at a time, straight from the SDM descriptions of each operation.
    fn shift_reference(op: u8, size: u8, a: u64, raw_count: u64, flags_in: u64) -> ShiftReference {
        let bits = size as u32 * 8;
        let m = mask(size);
        let count = raw_count & if size == 8 { 0x3f } else { 0x1f };
        let msb = |v: u64| (v >> (bits - 1)) & 1;
        let second_msb = |v: u64| (v >> (bits - 2)) & 1;
        let preserved = flags_in & (SF | ZF | AF | PF);
        if count == 0 {
            return ShiftReference {
                result: a,
                flags: flags_in,
                defined: ALL_ARITH,
            };
        }
        let mut r = a;
        let mut cf = flags_in & CF;
        let rotate = matches!(op, SHIFT_ROL | SHIFT_ROR | SHIFT_RCL | SHIFT_RCR);
        for _ in 0..count {
            match op {
                SHIFT_ROL => {
                    let out = msb(r);
                    r = ((r << 1) | out) & m;
                    cf = out;
                }
                SHIFT_ROR => {
                    let out = r & 1;
                    r = (r >> 1) | (out << (bits - 1));
                    cf = out;
                }
                SHIFT_RCL => {
                    let out = msb(r);
                    r = ((r << 1) | cf) & m;
                    cf = out;
                }
                SHIFT_RCR => {
                    let out = r & 1;
                    r = (r >> 1) | (cf << (bits - 1));
                    cf = out;
                }
                SHIFT_SHL => {
                    cf = msb(r);
                    r = (r << 1) & m;
                }
                SHIFT_SHR => {
                    cf = r & 1;
                    r >>= 1;
                }
                _ => {
                    cf = r & 1;
                    r = (r >> 1) | (r & (1 << (bits - 1)));
                }
            }
        }
        let of = match op {
            SHIFT_ROL | SHIFT_RCL | SHIFT_SHL => msb(r) ^ cf,
            SHIFT_ROR | SHIFT_RCR => msb(r) ^ second_msb(r),
            SHIFT_SHR => msb(a),
            _ => 0,
        };
        let mut flags = if rotate {
            preserved
        } else {
            zero_sign_parity(r, size)
        };
        if cf != 0 {
            flags |= CF;
        }
        if of != 0 {
            flags |= OF;
        }
        let mut defined = if rotate {
            CF | SF | ZF | AF | PF
        } else {
            CF | SF | ZF | PF
        };
        if count == 1 {
            defined |= OF;
        }
        // CF of SHL/SHR is undefined once the count reaches the operand width.
        if matches!(op, SHIFT_SHL | SHIFT_SHR) && count >= bits as u64 {
            defined &= !CF;
        }
        ShiftReference {
            result: r,
            flags,
            defined,
        }
    }

    fn check_shift_snippet(snippet: &ShiftSnippet) {
        let ShiftSnippet {
            op,
            size,
            count,
            memory,
            code,
        } = *snippet;
        let m = mask(size);
        let counts: Vec<u64> = match count {
            Count::Cl => SHIFT_COUNTS_FOR_CL.to_vec(),
            Count::Imm(n) => vec![n],
        };
        let mut pair = Pair::new(code);
        for a in interesting_values(size) {
            for &raw_count in &counts {
                for flags_in in [0, CF, SF | ZF | AF | PF | OF, ALL_ARITH] {
                    let context = format!(
                        "op {op} size {size} {count:?} memory {memory}: value {a:#x} count {raw_count} flags {}",
                        flag_names(flags_in)
                    );
                    let junk = OPERAND_JUNK & !m;
                    pair.halted(|h| {
                        if memory {
                            memory_with_marker(h, a, size);
                        } else {
                            h.cpu.gpr[RAX] = junk | a;
                        }
                        // Only CL counts; the rest of RCX is junk.
                        h.cpu.gpr[RCX] = OPERAND_JUNK & !0xff | raw_count & 0xff;
                        h.cpu.set_eflags_arith(flags_in);
                    });
                    let expected = shift_reference(op, size, a, raw_count & 0xff, flags_in);
                    if memory {
                        let mut want = [MEMORY_MARKER; 16];
                        want[..size as usize]
                            .copy_from_slice(&expected.result.to_le_bytes()[..size as usize]);
                        assert_eq!(pair.interp.read(DATA, 16), want, "{context}: memory");
                    } else {
                        let want = match size {
                            1 | 2 => (junk | a) & !m | expected.result,
                            // A 32-bit destination is zero-extended even when the count is 0 mod 32.
                            4 => expected.result,
                            _ => expected.result,
                        };
                        assert_eq!(pair.reg(RAX), want, "{context}: register");
                    }
                    assert_flags(pair.flags(), expected.defined, expected.flags, &context);
                }
            }
        }
    }

    fn check_shift_op(op: u8) {
        let mut checked = 0;
        for snippet in SHIFT_SNIPPETS.iter().filter(|s| s.op == op) {
            check_shift_snippet(snippet);
            checked += 1;
        }
        assert_eq!(checked, 28);
    }

    macro_rules! shift_tests {
    ($($name:ident: $op:expr;)*) => {
        $(#[test] fn $name() { check_shift_op($op); })*
    };
}

    shift_tests! {
        rol_every_size_count_and_form: SHIFT_ROL;
        ror_every_size_count_and_form: SHIFT_ROR;
        rcl_every_size_count_and_form: SHIFT_RCL;
        rcr_every_size_count_and_form: SHIFT_RCR;
        shl_every_size_count_and_form: SHIFT_SHL;
        shr_every_size_count_and_form: SHIFT_SHR;
        sar_every_size_count_and_form: SHIFT_SAR;
    }

    #[test]
    fn shift_examples_from_the_manual() {
        // shl al, 1 on 0x80: carry out, zero result, OF = MSB(result) ^ CF = 1.
        let pair = run_regs(code::SHL_1_ONE, &[(RAX, 0x80)]);
        assert_eq!(pair.reg(RAX), 0);
        assert_flags(
            pair.flags(),
            CF | OF | ZF | SF | PF,
            CF | OF | ZF | PF,
            "shl al,1",
        );
        // shr al, 1 on 0x81: OF is the original sign bit.
        let pair = run_regs(code::SHR_1_ONE, &[(RAX, 0x81)]);
        assert_eq!(pair.reg(RAX), 0x40);
        assert_flags(pair.flags(), CF | OF | ZF | SF, CF | OF, "shr al,1");
        // sar always clears OF for count 1.
        let pair = run_regs(code::SAR_1_ONE, &[(RAX, 0x81)]);
        assert_eq!(pair.reg(RAX), 0xc0);
        assert_flags(pair.flags(), CF | OF | SF | ZF, CF | SF, "sar al,1");
        // rol/ror: only CF (and OF at count 1) change.
        let mut pair = Pair::new(code::ROL_4_ONE);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0x8000_0001;
            h.cpu.set_eflags_arith(ZF | SF | PF | AF);
        });
        assert_eq!(pair.reg(RAX), 3);
        assert_flags(
            pair.flags(),
            ALL_ARITH,
            ZF | SF | PF | AF | CF | OF,
            "rol eax,1",
        );
    }

    #[test]
    fn shift_count_zero_leaves_flags_alone_and_zero_extends_32_bit_destinations() {
        // A zero count changes no flag. The destination is still written back (QEMU does the same),
        // so a 32-bit register is zero-extended even though its value is unchanged.
        for (code, size) in [(code::SHL_4_CL, 4u8), (code::SHL_8_CL, 8)] {
            let mut pair = Pair::new(code);
            pair.halted(|h| {
                h.cpu.gpr[RAX] = 0xdead_beef_8000_0001;
                h.cpu.gpr[RCX] = 0;
                h.cpu.set_eflags_arith(CF | OF | ZF);
            });
            let want = if size == 4 {
                0x8000_0001
            } else {
                0xdead_beef_8000_0001
            };
            assert_eq!(pair.reg(RAX), want, "size {size}");
            assert_flags(
                pair.flags(),
                ALL_ARITH,
                CF | OF | ZF,
                "flags unchanged by a zero count",
            );
        }
    }

    #[test]
    fn memory_shift_with_a_zero_count_still_stores_so_a_write_fault_is_taken() {
        // On hardware a read-modify-write with a zero count still performs the write; pinned here with
        // a read-only page and CR0.WP.
        let mut pair = Pair::new(code::SHL_4_MEM_CL);
        let stop = pair.run(|h| {
            write_protect_huge_page(h, HOLE);
            h.cpu.gpr[RDX] = HOLE + 0x40;
            h.cpu.gpr[RCX] = 0;
        });
        assert_eq!(stop, fault(VECTOR_PAGE_FAULT, Some(PF_PROTECTION_WRITE)));
        assert_eq!(pair.cpu().cr2, HOLE + 0x40);
    }

    #[test]
    fn shift_group_alias_6_behaves_as_shl() {
        let mut pair = Pair::new(code::SAL_ALIAS_EAX_4);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0x0800_0001;
        });
        assert_eq!(pair.reg(RAX), 0x8000_0010);
        assert_flags(pair.flags(), CF | SF | ZF, SF, "shl alias");
    }

    #[test]
    fn shift_by_cl_uses_only_the_low_byte_of_rcx() {
        let pair = run_regs(code::SHL_8_CL, &[(RAX, 1), (RCX, 0x1_0000_0100 | 4)]);
        assert_eq!(pair.reg(RAX), 0x10);
    }

    #[test]
    fn rcl_and_rcr_rotate_through_carry_with_period_width_plus_one() {
        // rcl al by 9 is a full turn: value and carry return to where they started.
        let mut pair = Pair::new(code::RCL_1_CL);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xa5;
            h.cpu.gpr[RCX] = 9;
            h.cpu.set_eflags_arith(CF);
        });
        assert_eq!(pair.reg(RAX) & 0xff, 0xa5);
        assert_flags(pair.flags(), CF, CF, "rcl al, 9");
        let mut pair = Pair::new(code::RCR_2_CL);
        pair.halted(|h| {
            h.cpu.gpr[RAX] = 0xa5a5;
            h.cpu.gpr[RCX] = 17;
            h.cpu.set_eflags_arith(0);
        });
        assert_eq!(pair.reg(RAX) & 0xffff, 0xa5a5);
        assert_flags(pair.flags(), CF, 0, "rcr ax, 17");
    }
    const STRING_SNIPPETS: &[StringSnippet] = &[
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::MOVS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_MOVS_1,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::STOS_1,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_STOS_1,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::LODS_1,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_LODS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::SCAS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_SCAS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_SCAS_1,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::CMPS_1,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 1,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_CMPS_1,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 1,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_CMPS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 2,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::MOVS_2,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 2,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_MOVS_2,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 2,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::STOS_2,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 2,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_STOS_2,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 2,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::LODS_2,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 2,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_LODS_2,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 2,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::SCAS_2,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 2,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_SCAS_2,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 2,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_SCAS_2,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 2,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::CMPS_2,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 2,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_CMPS_2,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 2,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_CMPS_2,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 4,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::MOVS_4,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 4,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_MOVS_4,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 4,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::STOS_4,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 4,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_STOS_4,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 4,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::LODS_4,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 4,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_LODS_4,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 4,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::SCAS_4,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 4,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_SCAS_4,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 4,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_SCAS_4,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 4,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::CMPS_4,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 4,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_CMPS_4,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 4,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_CMPS_4,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::MOVS_8,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_MOVS_8,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::STOS_8,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_STOS_8,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::LODS_8,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: None,
            code: code::REP_LODS_8,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::SCAS_8,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 8,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_SCAS_8,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 8,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_SCAS_8,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: None,
            code: code::CMPS_8,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 8,
            rep: Rep::Repe,
            addr32: false,
            segment: None,
            code: code::REPE_CMPS_8,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 8,
            rep: Rep::Repne,
            addr32: false,
            segment: None,
            code: code::REPNE_CMPS_8,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::None,
            addr32: true,
            segment: None,
            code: code::A32_MOVS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_MOVS_1,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 1,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_STOS_1,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 1,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_LODS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::Repe,
            addr32: true,
            segment: None,
            code: code::A32_REPE_SCAS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::Repne,
            addr32: true,
            segment: None,
            code: code::A32_REPNE_SCAS_1,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 1,
            rep: Rep::Repe,
            addr32: true,
            segment: None,
            code: code::A32_REPE_CMPS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 4,
            rep: Rep::None,
            addr32: true,
            segment: None,
            code: code::A32_MOVS_4,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 4,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_MOVS_4,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 4,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_STOS_4,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 4,
            rep: Rep::Repeat,
            addr32: true,
            segment: None,
            code: code::A32_REP_LODS_4,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 4,
            rep: Rep::Repe,
            addr32: true,
            segment: None,
            code: code::A32_REPE_SCAS_4,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 4,
            rep: Rep::Repne,
            addr32: true,
            segment: None,
            code: code::A32_REPNE_SCAS_4,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 4,
            rep: Rep::Repe,
            addr32: true,
            segment: None,
            code: code::A32_REPE_CMPS_4,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::None,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_MOVS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REP_MOVS_1,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_GS),
            code: code::GS_REP_LODS_1,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 1,
            rep: Rep::Repe,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REPE_CMPS_1,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 1,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REP_STOS_1,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 1,
            rep: Rep::Repne,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REPNE_SCAS_1,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 8,
            rep: Rep::None,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_MOVS_8,
        },
        StringSnippet {
            op: StringOp::Movs,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REP_MOVS_8,
        },
        StringSnippet {
            op: StringOp::Lods,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_GS),
            code: code::GS_REP_LODS_8,
        },
        StringSnippet {
            op: StringOp::Cmps,
            size: 8,
            rep: Rep::Repe,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REPE_CMPS_8,
        },
        StringSnippet {
            op: StringOp::Stos,
            size: 8,
            rep: Rep::Repeat,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REP_STOS_8,
        },
        StringSnippet {
            op: StringOp::Scas,
            size: 8,
            rep: Rep::Repne,
            addr32: false,
            segment: Some(SEG_FS),
            code: code::FS_REPNE_SCAS_8,
        },
    ];

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum StringOp {
        Movs,
        Stos,
        Lods,
        Scas,
        Cmps,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Rep {
        None,
        Repeat,
        Repe,
        Repne,
    }

    #[derive(Clone, Copy, Debug)]
    struct StringSnippet {
        op: StringOp,
        size: u8,
        rep: Rep,
        addr32: bool,
        segment: Option<usize>,
        code: &'static [u8],
    }

    const STRING_WINDOW: u64 = DATA;
    const STRING_WINDOW_LEN: usize = 0x1_8000;
    const STRING_SRC: u64 = DATA + 0x2000;
    const STRING_DST: u64 = DATA + 0xc000;
    const STRING_SEGMENT_BASE: u64 = 0x1000;
    const ADDRESS_JUNK: u64 = 0x1234_5678_0000_0000;
    const STRING_ACCUMULATOR: u64 = 0x1122_3344_5566_7788;
    const STRING_FLAGS_IN: u64 = CF | AF | SF | OF;

    /// The state a string instruction reads and writes, for the reference model.
    struct StringMachine {
        rsi: u64,
        rdi: u64,
        rcx: u64,
        rax: u64,
        flags: u64,
        window: Vec<u8>,
    }

    impl StringMachine {
        fn read(&self, address: u64, size: u8) -> u64 {
            let offset = address
                .checked_sub(STRING_WINDOW)
                .expect("below the window") as usize;
            let mut bytes = [0u8; 8];
            bytes[..size as usize].copy_from_slice(&self.window[offset..offset + size as usize]);
            u64::from_le_bytes(bytes)
        }

        fn write(&mut self, address: u64, size: u8, value: u64) {
            let offset = address
                .checked_sub(STRING_WINDOW)
                .expect("below the window") as usize;
            self.window[offset..offset + size as usize]
                .copy_from_slice(&value.to_le_bytes()[..size as usize]);
        }
    }

    /// The SDM's operation pseudo-code for each string instruction, iteration by iteration.
    fn string_reference(s: &StringSnippet, df: bool, m: &mut StringMachine) {
        let size = s.size;
        let m_size = mask(size);
        let step = if df {
            (size as u64).wrapping_neg()
        } else {
            size as u64
        };
        let amask = if s.addr32 { 0xffff_ffff } else { u64::MAX };
        let source_base = match s.segment {
            Some(SEG_FS | SEG_GS)
                if matches!(s.op, StringOp::Movs | StringOp::Lods | StringOp::Cmps) =>
            {
                STRING_SEGMENT_BASE
            }
            _ => 0,
        };
        let advance = |register: u64| {
            (register & amask).wrapping_add(step) & amask
                | (register & !amask & if s.addr32 { 0 } else { u64::MAX })
        };
        let repeating = s.rep != Rep::None;
        let compares = matches!(s.op, StringOp::Scas | StringOp::Cmps);
        loop {
            if repeating && m.rcx & amask == 0 {
                break;
            }
            let source = source_base.wrapping_add(m.rsi & amask);
            let destination = m.rdi & amask;
            match s.op {
                StringOp::Movs => {
                    let value = m.read(source, size);
                    m.write(destination, size, value);
                    m.rsi = advance(m.rsi);
                    m.rdi = advance(m.rdi);
                }
                StringOp::Stos => {
                    m.write(destination, size, m.rax & m_size);
                    m.rdi = advance(m.rdi);
                }
                StringOp::Lods => {
                    let value = m.read(source, size);
                    m.rax = match size {
                        1 | 2 => (m.rax & !m_size) | value,
                        _ => value,
                    };
                    m.rsi = advance(m.rsi);
                }
                StringOp::Scas => {
                    let memory = m.read(destination, size);
                    m.flags = alu_reference(ALU_SUB, size, m.rax & m_size, memory, 0).flags;
                    m.rdi = advance(m.rdi);
                }
                StringOp::Cmps => {
                    let left = m.read(source, size);
                    let right = m.read(destination, size);
                    m.flags = alu_reference(ALU_SUB, size, left, right, 0).flags;
                    m.rsi = advance(m.rsi);
                    m.rdi = advance(m.rdi);
                }
            }
            if !repeating {
                break;
            }
            m.rcx = (m.rcx & amask).wrapping_sub(1) & amask
                | (m.rcx & !amask & if s.addr32 { 0 } else { u64::MAX });
            if compares {
                let zero = m.flags & ZF != 0;
                if (s.rep == Rep::Repe && !zero) || (s.rep == Rep::Repne && zero) {
                    break;
                }
            }
        }
    }

    fn random_window(seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..STRING_WINDOW_LEN)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u8
            })
            .collect()
    }

    fn check_string_snippet(s: &StringSnippet) {
        let compares = matches!(s.op, StringOp::Scas | StringOp::Cmps);
        let repeating = s.rep != Rep::None;
        let counts: Vec<u64> = match (repeating, compares) {
            (false, _) => vec![7],
            (true, true) => vec![0, 1, 2, 5, 17, 300],
            (true, false) => vec![0, 1, 2, 5, 17, 5000],
        };
        let mismatches: Vec<usize> = if compares {
            vec![0, 1, 3, usize::MAX]
        } else {
            vec![usize::MAX]
        };
        let size = s.size;
        let m_size = mask(size);
        let mut pair = Pair::new(s.code);
        for df in [false, true] {
            for &count in &counts {
                for &mismatch in &mismatches {
                    let elements = if repeating { count.max(1) as usize } else { 1 };
                    let start = |base: u64| {
                        if df {
                            base + (elements as u64 - 1) * size as u64
                        } else {
                            base
                        }
                    };
                    let visit = |base: u64, j: usize| {
                        if df {
                            start(base) - j as u64 * size as u64
                        } else {
                            base + j as u64 * size as u64
                        }
                    };
                    let mut window =
                        random_window(count * 31 + mismatch as u64 % 97 + df as u64 + size as u64);
                    let matches_at = |j: usize| match s.rep {
                        Rep::Repne => j == mismatch,
                        Rep::None => mismatch != 0,
                        _ => j < mismatch,
                    };
                    let accumulator = STRING_ACCUMULATOR;
                    if compares {
                        for j in 0..elements {
                            let equal = matches_at(j);
                            let (offset, value) = if s.op == StringOp::Cmps {
                                let src_offset = (visit(STRING_SRC, j) - STRING_WINDOW) as usize;
                                let mut src = [0u8; 8];
                                src[..size as usize].copy_from_slice(
                                    &window[src_offset..src_offset + size as usize],
                                );
                                let value = u64::from_le_bytes(src);
                                (
                                    (visit(STRING_DST, j) - STRING_WINDOW) as usize,
                                    value ^ if equal { 0 } else { 1 },
                                )
                            } else {
                                (
                                    (visit(STRING_DST, j) - STRING_WINDOW) as usize,
                                    (accumulator & m_size) ^ if equal { 0 } else { 1 },
                                )
                            };
                            window[offset..offset + size as usize]
                                .copy_from_slice(&value.to_le_bytes()[..size as usize]);
                        }
                    }
                    let source_register = start(STRING_SRC)
                        - if s.segment.is_some() && s.op != StringOp::Stos && s.op != StringOp::Scas
                        {
                            STRING_SEGMENT_BASE
                        } else {
                            0
                        };
                    let junk = if s.addr32 { ADDRESS_JUNK } else { 0 };
                    let initial = StringMachine {
                        rsi: source_register | junk,
                        rdi: start(STRING_DST) | junk,
                        rcx: count | junk,
                        rax: if s.op == StringOp::Lods {
                            u64::MAX
                        } else {
                            accumulator
                        },
                        flags: STRING_FLAGS_IN,
                        window: window.clone(),
                    };
                    let mut expected = StringMachine {
                        window,
                        ..StringMachine {
                            rsi: initial.rsi,
                            rdi: initial.rdi,
                            rcx: initial.rcx,
                            rax: initial.rax,
                            flags: initial.flags,
                            window: Vec::new(),
                        }
                    };
                    string_reference(s, df, &mut expected);
                    pair.halted(|h| {
                        h.write(STRING_WINDOW, &initial.window);
                        h.cpu.gpr[RSI] = initial.rsi;
                        h.cpu.gpr[RDI] = initial.rdi;
                        h.cpu.gpr[RCX] = initial.rcx;
                        h.cpu.gpr[RAX] = initial.rax;
                        h.cpu.set_eflags_arith(initial.flags);
                        if df {
                            h.cpu.eflags |= DF;
                        }
                        if let Some(register) = s.segment {
                            h.cpu.seg[register].base = STRING_SEGMENT_BASE;
                        }
                    });
                    let context = format!("{s:?} df {df} count {count} mismatch {mismatch}");
                    assert_eq!(pair.reg(RSI), expected.rsi, "{context}: rsi");
                    assert_eq!(pair.reg(RDI), expected.rdi, "{context}: rdi");
                    assert_eq!(pair.reg(RCX), expected.rcx, "{context}: rcx");
                    assert_eq!(pair.reg(RAX), expected.rax, "{context}: rax");
                    assert_flags(pair.flags(), ALL_ARITH, expected.flags, &context);
                    assert!(
                        pair.interp.read(STRING_WINDOW, STRING_WINDOW_LEN) == expected.window,
                        "{context}: memory"
                    );
                }
            }
        }
    }

    #[test]
    fn string_instructions_match_the_reference_model_in_every_variant() {
        for snippet in STRING_SNIPPETS {
            check_string_snippet(snippet);
        }
    }

    const IOPL_ONE: u64 = 1 << IOPL_SHIFT;
    const IOPL_THREE: u64 = 3 << IOPL_SHIFT;
    const ALL_POPF_BITS: u64 =
        CF | PF | AF | ZF | SF | TF | IF | DF | OF | IOPL_THREE | NT | RF | VM | AC | EFLAGS_ID;

    /// Runs one instruction (the snippet's first) at `cpl` with the given initial flags and the value
    /// on top of the stack.
    fn run_flag_instruction(snippet: &[u8], cpl: u8, flags: u64, stack_value: u64) -> Pair {
        let mut pair = Pair::new(snippet);
        let stop = pair.run_for(1, |h| {
            enter_cpl(h, cpl);
            h.cpu.eflags = RESERVED_ONE | flags;
            h.cpu.lazy_op = LazyFlagsOp::Eflags;
            stack_with(h, 0x40, stack_value);
        });
        assert_eq!(stop, Stop::Limit, "{}", pair.cpu().dump_regs());
        pair
    }

    #[test]
    fn pushfq_pushes_the_flags_image_without_rf_and_vm() {
        let mut pair = Pair::new(code::PUSHFQ);
        pair.halted(|h| {
            h.cpu.eflags = RESERVED_ONE | IF | DF | TF | NT | AC | EFLAGS_ID | RF | VM | IOPL_ONE;
            h.cpu.set_eflags_arith(CF | ZF | OF);
        });
        let image = pair.interp.read_u64(STACK_TOP - 8);
        assert_eq!(pair.reg(RSP), STACK_TOP - 8);
        assert_eq!(
            image,
            RESERVED_ONE | IF | DF | TF | NT | AC | EFLAGS_ID | IOPL_ONE | CF | ZF | OF,
            "RF and VM read as zero in the pushed image"
        );
        // Bit 1 reads as one even when everything else is clear.
        let pair = run_regs(code::PUSHFQ, &[]);
        assert_eq!(pair.interp.read_u64(STACK_TOP - 8), RESERVED_ONE);
    }

    #[test]
    fn pushfw_pushes_the_low_word() {
        let mut pair = Pair::new(code::PUSHFW);
        pair.halted(|h| {
            h.cpu.eflags = RESERVED_ONE | IF | AC;
            h.cpu.set_eflags_arith(SF);
        });
        assert_eq!(pair.reg(RSP), STACK_TOP - 2);
        assert_eq!(
            pair.interp.read(STACK_TOP - 2, 2),
            ((RESERVED_ONE | IF | SF) as u16).to_le_bytes()
        );
    }

    #[test]
    fn popfq_at_cpl0_loads_every_modifiable_flag_but_not_rf_and_vm() {
        let pair = run_flag_instruction(code::POPFQ, 0, 0, ALL_POPF_BITS);
        let flags = pair.cpu().eflags_full();
        let loaded = CF | PF | AF | ZF | SF | TF | IF | DF | OF | IOPL_THREE | NT | AC | EFLAGS_ID;
        assert_eq!(flags & !RESERVED_ONE, loaded, "RF and VM stay clear");
        assert_eq!(flags & RESERVED_ONE, RESERVED_ONE);
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x38);
        // Clearing works too, and bit 1 cannot be cleared.
        let pair = run_flag_instruction(code::POPFQ, 0, ALL_POPF_BITS & !RF & !VM, 0);
        assert_eq!(pair.cpu().eflags_full(), RESERVED_ONE);
    }

    #[test]
    fn popfq_above_iopl_keeps_if_and_iopl_but_loads_the_rest() {
        let image = ALL_POPF_BITS;
        // CPL 1, IOPL 0, IF initially set: neither IF nor IOPL can change.
        let pair = run_flag_instruction(code::POPFQ, 1, IF, image & !IF);
        let flags = pair.cpu().eflags_full();
        assert_eq!(flags & IF, IF, "IF kept");
        assert_eq!(flags & IOPL, 0, "IOPL kept");
        assert_eq!(
            flags & (CF | ZF | TF | DF | NT | AC | EFLAGS_ID),
            CF | ZF | TF | DF | NT | AC | EFLAGS_ID
        );
        // CPL 1, IOPL 0, IF initially clear: a set IF in the image is ignored.
        let pair = run_flag_instruction(code::POPFQ, 1, 0, image);
        assert_eq!(pair.cpu().eflags_full() & (IF | IOPL), 0);
        // At CPL 3 the same rules apply.
        let pair = run_flag_instruction(code::POPFQ, 3, 0, image);
        assert_eq!(pair.cpu().eflags_full() & (IF | IOPL), 0);
        assert_eq!(pair.cpu().eflags_full() & CF, CF);
    }

    #[test]
    fn popfq_within_iopl_can_change_if_but_never_iopl_above_cpl0() {
        // CPL 1 with IOPL 1: IF is loadable.
        let pair = run_flag_instruction(code::POPFQ, 1, IOPL_ONE, IF | IOPL_THREE);
        let flags = pair.cpu().eflags_full();
        assert_eq!(flags & IF, IF);
        assert_eq!(flags & IOPL, IOPL_ONE, "only CPL 0 may change IOPL");
        let pair = run_flag_instruction(code::POPFQ, 1, IOPL_ONE | IF, 0);
        assert_eq!(pair.cpu().eflags_full() & IF, 0, "and may clear it");
        // CPL 3 with IOPL 3.
        let pair = run_flag_instruction(code::POPFQ, 3, IOPL_THREE, IF | IOPL_ONE);
        let flags = pair.cpu().eflags_full();
        assert_eq!(flags & IF, IF);
        assert_eq!(flags & IOPL, IOPL_THREE);
    }

    #[test]
    fn popfw_loads_only_the_low_sixteen_bits() {
        let initial = AC | EFLAGS_ID | NT;
        let pair = run_flag_instruction(
            code::POPFW,
            0,
            initial | IF,
            0xffff_ffff_ffff_0000 | (CF | ZF | DF),
        );
        let flags = pair.cpu().eflags_full();
        assert_eq!(flags & (CF | ZF | DF), CF | ZF | DF);
        assert_eq!(
            flags & IF,
            0,
            "IF is in the low word and was cleared by the zero bit"
        );
        assert_eq!(
            flags & (AC | EFLAGS_ID),
            AC | EFLAGS_ID,
            "bits above 15 are untouched"
        );
        assert_eq!(flags & NT, 0, "NT is bit 14, in the low word");
        assert_eq!(pair.reg(RSP), STACK_TOP - 0x40 + 2);
    }

    #[test]
    fn popf_resolves_deferred_flags_so_conditions_see_the_loaded_values() {
        // An arithmetic instruction leaves lazy flags; popf then replaces them wholesale.
        let mut pair = Pair::new(code::POPF_THEN_JE);
        pair.halted(|h| {
            stack_with(h, 0x40, ZF | RESERVED_ONE);
            h.cpu.gpr[RAX] = 5;
            h.cpu.gpr[RBX] = 7;
        });
        assert_eq!(
            pair.reg(RCX),
            2,
            "ZF from the popped image decides the branch"
        );
    }

    #[test]
    fn popf_fault_keeps_rsp_and_flags() {
        assert_fault_changes_nothing(
            code::POPFQ,
            |h| {
                h.cpu.gpr[RSP] = NON_CANONICAL;
                h.cpu.set_eflags_arith(CF);
            },
            ss_fault(0),
        );
    }

    /// `cli` does not end a JIT block, so a JIT run of these snippets cannot stop after one
    /// instruction; `jit::diff_misc` compares the JIT's cli (and its faults) instead.
    fn one_instruction_pair(snippet: &[u8]) -> Pair {
        if snippet == code::CLI {
            Pair::interpreter_only(snippet)
        } else {
            Pair::new(snippet)
        }
    }

    #[test]
    fn cli_and_sti_toggle_if_and_sti_opens_a_one_instruction_shadow() {
        let run = |snippet: &[u8], flags: u64| {
            let mut pair = one_instruction_pair(snippet);
            let stop = pair.run_for(1, |h| {
                h.cpu.eflags = RESERVED_ONE | flags;
            });
            assert_eq!(stop, Stop::Limit);
            pair
        };
        let pair = run(code::STI, 0);
        assert_eq!(pair.cpu().eflags_full() & IF, IF);
        assert!(
            pair.cpu().interrupt_shadow,
            "sti from IF=0 delays interrupts by one instruction"
        );
        let pair = run(code::STI, IF);
        assert!(
            !pair.cpu().interrupt_shadow,
            "sti with IF already set opens no shadow"
        );
        let pair = run(code::CLI, IF);
        assert_eq!(pair.cpu().eflags_full() & IF, 0);
        assert!(!pair.cpu().interrupt_shadow);
    }

    #[test]
    fn cli_and_sti_need_cpl_not_above_iopl() {
        for snippet in [code::CLI, code::STI] {
            for (cpl, iopl, allowed) in [
                (0u8, 0u64, true),
                (1, 0, false),
                (1, 1, true),
                (2, 1, false),
                (3, 2, false),
                (3, 3, true),
            ] {
                let mut pair = one_instruction_pair(snippet);
                let stop = pair.run_for(1, |h| {
                    enter_cpl(h, cpl);
                    h.cpu.eflags = RESERVED_ONE | iopl << IOPL_SHIFT;
                });
                let want = if allowed { Stop::Limit } else { gp(0) };
                assert_eq!(stop, want, "{snippet:02x?} cpl {cpl} iopl {iopl}");
                if !allowed {
                    assert_eq!(pair.cpu().rip, CODE_BASE);
                    assert!(!pair.cpu().interrupt_shadow);
                }
            }
        }
    }

    #[test]
    fn hlt_halts_the_cpu_at_cpl0_only() {
        let mut pair = Pair::new(code::HLT_ONLY);
        let stop = pair.run(|_| {});
        assert_eq!(stop, Stop::Halted);
        assert!(pair.cpu().halted);
        assert_eq!(
            pair.cpu().rip,
            CODE_BASE + 1,
            "rip is past the hlt so an interrupt resumes after it"
        );
        for cpl in [1u8, 2, 3] {
            let mut pair = Pair::new(code::HLT_ONLY);
            let stop = pair.run(|h| enter_cpl(h, cpl));
            assert_eq!(stop, gp(0), "cpl {cpl}");
            assert!(!pair.cpu().halted);
            assert_eq!(pair.cpu().rip, CODE_BASE);
        }
    }

    fn software_interrupt(vector: u8) -> Stop {
        Stop::Fault {
            vector,
            error_code: None,
            software: true,
        }
    }

    #[test]
    fn int3_and_int_n_raise_software_exceptions_with_rip_past_the_instruction() {
        let (pair, stop) = run_regs_stop(code::INT3, &[]);
        assert_eq!(stop, software_interrupt(3));
        assert_eq!(
            pair.cpu().rip,
            CODE_BASE + 1,
            "the pushed return address is the next instruction"
        );
        let (pair, stop) = run_regs_stop(code::INT_80, &[]);
        assert_eq!(stop, software_interrupt(0x80));
        assert_eq!(pair.cpu().rip, CODE_BASE + 2);
        let (pair, stop) = run_regs_stop(code::INT_3_LONG, &[]);
        assert_eq!(
            stop,
            software_interrupt(3),
            "cd 03 is the two-byte form of int 3"
        );
        assert_eq!(pair.cpu().rip, CODE_BASE + 2);
        assert_eq!(
            pair.reg(RSP),
            STACK_TOP,
            "nothing is pushed by the instruction itself"
        );
    }

    #[test]
    fn into_and_the_one_byte_int1_are_not_available_in_64_bit_mode() {
        let (_, stop) = run_regs_stop(code::INTO, &[]);
        assert_eq!(stop, ud());
        let (_, stop) = run_regs_stop(code::INT1, &[]);
        assert_eq!(stop, ud(), "icebp (f1) is not implemented");
    }

    #[test]
    fn ud2_raises_invalid_opcode_at_the_instruction() {
        let (pair, stop) = run_regs_stop(code::UD2_, &[]);
        assert_eq!(stop, ud());
        assert_eq!(pair.cpu().rip, CODE_BASE);
    }

    #[test]
    fn fwait_faults_only_with_cr0_mp_and_ts_both_set() {
        for (cr0_bits, faults) in [
            (0, false),
            (CR0_MP, false),
            (CR0_TS, false),
            (CR0_MP | CR0_TS, true),
        ] {
            let mut pair = Pair::new(code::FWAIT);
            let stop = pair.run(|h| {
                h.cpu.cr0 = (h.cpu.cr0 & !(CR0_MP | CR0_TS)) | cr0_bits;
            });
            let want = if faults {
                fault(VECTOR_DEVICE_NOT_AVAILABLE, None)
            } else {
                Stop::Halted
            };
            assert_eq!(stop, want, "cr0 bits {cr0_bits:#x}");
        }
    }

    #[test]
    fn opcodes_removed_in_64_bit_mode_raise_invalid_opcode() {
        for (name, snippet) in INVALID_IN_LONG_MODE {
            let (pair, stop) = run_regs_stop(snippet, &[]);
            assert_eq!(stop, ud(), "{name} {snippet:02x?}");
            assert_eq!(pair.cpu().rip, CODE_BASE, "{name}");
            assert_eq!(pair.reg(RSP), STACK_TOP, "{name}");
        }
    }

    #[test]
    fn undefined_two_byte_opcodes_raise_invalid_opcode() {
        for (name, snippet) in UNDEFINED_TWO_BYTE {
            let (_, stop) = run_regs_stop(snippet, &[]);
            assert_eq!(stop, ud(), "0f {name} {snippet:02x?}");
        }
        // sysenter/sysexit are not implemented (SEP is not advertised).
        for snippet in [code::SYSENTER, code::SYSEXIT] {
            let (_, stop) = run_regs_stop(snippet, &[]);
            assert_eq!(stop, ud(), "{snippet:02x?}");
        }
    }

    fn cpuid(leaf: u32, subleaf: u32) -> [u64; 4] {
        let pair = run_regs(
            code::CPUID_,
            &[
                (RAX, ADDRESS_JUNK | leaf as u64),
                (RCX, ADDRESS_JUNK | subleaf as u64),
                (RBX, u64::MAX),
                (RDX, u64::MAX),
            ],
        );
        [pair.reg(RAX), pair.reg(RBX), pair.reg(RCX), pair.reg(RDX)]
    }

    #[test]
    fn cpuid_leaf_0_returns_the_vendor_string_in_ebx_edx_ecx() {
        let [eax, ebx, ecx, edx] = cpuid(0, 0);
        assert!(eax >= 1, "at least leaf 1 exists");
        let mut vendor = Vec::new();
        vendor.extend_from_slice(&(ebx as u32).to_le_bytes());
        vendor.extend_from_slice(&(edx as u32).to_le_bytes());
        vendor.extend_from_slice(&(ecx as u32).to_le_bytes());
        assert_eq!(vendor, b"AuthenticAMD");
        for value in [eax, ebx, ecx, edx] {
            assert_eq!(
                value >> 32,
                0,
                "cpuid results are written as 32-bit registers"
            );
        }
    }

    #[test]
    fn cpuid_feature_bits_cover_the_instructions_tested_here() {
        const TSC: u64 = 1 << 4;
        const MSR: u64 = 1 << 5;
        const CX8: u64 = 1 << 8;
        const APIC: u64 = 1 << 9;
        const SEP: u64 = 1 << 11;
        const CMOV: u64 = 1 << 15;
        const SYSCALL: u64 = 1 << 11;
        const NX: u64 = 1 << 20;
        const LONG_MODE: u64 = 1 << 29;
        const LAHF_SAHF: u64 = 1;
        const LZCNT_ABM: u64 = 1 << 5;
        const SVM: u64 = 1 << 2;
        let [_, _, ecx, edx] = cpuid(1, 0);
        for (bit, name) in [
            (TSC, "tsc"),
            (MSR, "msr"),
            (CX8, "cmpxchg8b"),
            (CMOV, "cmov"),
            (SEP, "sysenter"),
        ] {
            assert_ne!(edx & bit, 0, "{name} is implemented and advertised");
        }
        assert_ne!(
            edx & APIC,
            0,
            "the local APIC is implemented and advertised"
        );
        // The exact set of leaf 1 ECX bits is pinned in the tests in sys.rs; these must stay clear because
        // their instructions (or the OS support behind them) do not exist.
        for (bit, name) in [(1 << 5, "vmx"), (1 << 31, "hypervisor")] {
            assert_eq!(ecx & bit, 0, "{name} must not be advertised");
        }
        let [_, _, ecx, edx] = cpuid(0x8000_0001, 0);
        assert_ne!(edx & SYSCALL, 0);
        assert_ne!(edx & NX, 0);
        assert_ne!(edx & LONG_MODE, 0);
        assert_ne!(ecx & LAHF_SAHF, 0, "lahf/sahf work in 64-bit mode");
        // prefetch/prefetchw are executed as hints in sys.rs's `prefetchw_is_a_hint_that_never_faults`
        const PREFETCHW: u64 = 1 << 8;
        const TOPOEXT: u64 = 1 << 22;
        assert_eq!(
            ecx & !(LAHF_SAHF | LZCNT_ABM | SVM | PREFETCHW | TOPOEXT),
            0,
            "nothing beyond lahf, svm, abm, prefetchw and the topology leaves"
        );
    }

    #[test]
    fn cpuid_ignores_the_subleaf_and_unknown_leaves_read_as_zero() {
        assert_eq!(cpuid(1, 0), cpuid(1, 5));
        assert_eq!(cpuid(0x1234_5678, 0), [0, 0, 0, 0]);
        assert_eq!(cpuid(0x4000_0000, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn rdtsc_returns_guest_time_split_across_edx_eax() {
        let pair = run_regs(code::RDTSC_, &[(RAX, u64::MAX), (RDX, u64::MAX)]);
        assert_eq!(pair.reg(RAX) >> 32, 0);
        assert_eq!(pair.reg(RDX) >> 32, 0);
        let pair = run_regs(code::RDTSC_TWICE, &[]);
        let first = pair.reg(R9) << 32 | pair.reg(R8);
        let second = pair.reg(RDX) << 32 | pair.reg(RAX);
        assert!(
            second > first,
            "the time stamp counter advances between instructions"
        );
        assert!(
            second - first < 10,
            "and by about one tick per instruction here"
        );
    }

    #[test]
    fn rdtsc_at_cpl_above_0_obeys_cr4_tsd() {
        for (tsd, cpl, allowed) in [
            (false, 3u8, true),
            (true, 3, false),
            (true, 1, false),
            (true, 0, true),
            (false, 1, true),
        ] {
            let mut pair = Pair::new(code::RDTSC_UD2);
            let stop = pair.run(|h| {
                enter_cpl(h, cpl);
                if tsd {
                    h.cpu.cr4 |= CR4_TSD;
                }
            });
            // An allowed rdtsc runs on to the ud2 behind it; a denied one faults itself.
            let want = if allowed { ud() } else { gp(0) };
            assert_eq!(stop, want, "tsd {tsd} cpl {cpl}");
        }
    }

    #[test]
    fn wrmsr_then_rdmsr_round_trips_through_edx_eax() {
        let mut pair = Pair::new(code::WRMSR_);
        pair.halted(|h| {
            h.cpu.gpr[RCX] = MSR_LSTAR;
            h.cpu.gpr[RAX] = 0xffff_ffff_5566_7788;
            h.cpu.gpr[RDX] = 0xffff_ffff_0000_1234;
        });
        assert_eq!(
            pair.cpu().lstar,
            0x0000_1234_5566_7788,
            "upper halves of rax and rdx are ignored"
        );
        let mut pair = Pair::new(code::RDMSR_);
        pair.halted(|h| {
            h.cpu.lstar = 0x0000_1234_5566_7788;
            h.cpu.gpr[RCX] = MSR_LSTAR;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
        });
        assert_eq!((pair.reg(RDX), pair.reg(RAX)), (0x1234, 0x5566_7788));
    }

    #[test]
    fn rdmsr_and_wrmsr_of_an_unknown_msr_raise_general_protection() {
        for snippet in [code::RDMSR_, code::WRMSR_] {
            let (pair, stop) = run_regs_stop(snippet, &[(RCX, 0xdead_beef)]);
            assert_eq!(stop, gp(0), "{snippet:02x?}");
            assert_eq!(pair.cpu().rip, CODE_BASE);
        }
    }

    #[test]
    fn rdmsr_and_wrmsr_and_privileged_no_ops_need_cpl0() {
        for snippet in [
            code::RDMSR_,
            code::WRMSR_,
            code::CLTS,
            code::WBINVD,
            code::INVD,
        ] {
            let mut pair = Pair::new(snippet);
            let stop = pair.run(|h| {
                enter_cpl(h, 1);
                h.cpu.gpr[RCX] = MSR_LSTAR;
            });
            assert_eq!(stop, gp(0), "{snippet:02x?} at CPL 1");
            assert_eq!(pair.cpu().rip, CODE_BASE);
            let mut pair = Pair::new(snippet);
            let stop = pair.run(|h| h.cpu.gpr[RCX] = MSR_LSTAR);
            assert_eq!(stop, Stop::Halted, "{snippet:02x?} at CPL 0");
        }
    }

    #[test]
    fn clts_clears_cr0_ts() {
        let mut pair = Pair::new(code::CLTS);
        pair.halted(|h| h.cpu.cr0 |= CR0_TS);
        assert_eq!(pair.cpu().cr0 & CR0_TS, 0);
    }

    #[test]
    fn group_11_and_the_pop_group_are_defined_only_for_reg_zero() {
        for snippet in [
            code::POP_GROUP_REG1,
            code::MOV_GROUP11_REG1_8,
            code::MOV_GROUP11_REG1_32,
            code::MOV_GROUP11_REG7_8,
        ] {
            let (pair, stop) = run_regs_stop(snippet, &[(RAX, 0x77), (RCX, 0x99)]);
            assert_eq!(stop, ud(), "{snippet:02x?}");
            assert_eq!(
                (pair.reg(RAX), pair.reg(RSP)),
                (0x77, STACK_TOP),
                "{snippet:02x?}: nothing executed"
            );
        }
    }

    /// Hand-assembled snippets; the assembly is in the comment above each array.
#[rustfmt::skip]
mod code {
    // add rax, rbx
    // hlt
    pub(super) const ADD_8_RR: &[u8] = &[0x48, 0x01, 0xd8, 0xf4];
    // add rax, qword ptr [rdx]
    // hlt
    pub(super) const ADD_8_RM: &[u8] = &[0x48, 0x03, 0x02, 0xf4];
    // add qword ptr [rdx], rbx
    // hlt
    pub(super) const ADD_8_MR: &[u8] = &[0x48, 0x01, 0x1a, 0xf4];
    // add rax, 305419896
    // hlt
    pub(super) const ADD_8_RIA_P: &[u8] = &[0x48, 0x05, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add rbx, 305419896
    // hlt
    pub(super) const ADD_8_RI_P: &[u8] = &[0x48, 0x81, 0xc3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add qword ptr [rdx], 305419896
    // hlt
    pub(super) const ADD_8_MI_P: &[u8] = &[0x48, 0x81, 0x02, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add rax, -305419896
    // hlt
    pub(super) const ADD_8_RIA_N: &[u8] = &[0x48, 0x05, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add rbx, -305419896
    // hlt
    pub(super) const ADD_8_RI_N: &[u8] = &[0x48, 0x81, 0xc3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add qword ptr [rdx], -305419896
    // hlt
    pub(super) const ADD_8_MI_N: &[u8] = &[0x48, 0x81, 0x02, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add rbx, -3
    // hlt
    pub(super) const ADD_8_RIS: &[u8] = &[0x48, 0x83, 0xc3, 0xfd, 0xf4];
    // add qword ptr [rdx], -3
    // hlt
    pub(super) const ADD_8_MIS: &[u8] = &[0x48, 0x83, 0x02, 0xfd, 0xf4];
    // add eax, ebx
    // hlt
    pub(super) const ADD_4_RR: &[u8] = &[0x01, 0xd8, 0xf4];
    // add eax, dword ptr [rdx]
    // hlt
    pub(super) const ADD_4_RM: &[u8] = &[0x03, 0x02, 0xf4];
    // add dword ptr [rdx], ebx
    // hlt
    pub(super) const ADD_4_MR: &[u8] = &[0x01, 0x1a, 0xf4];
    // add eax, 305419896
    // hlt
    pub(super) const ADD_4_RIA_P: &[u8] = &[0x05, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add ebx, 305419896
    // hlt
    pub(super) const ADD_4_RI_P: &[u8] = &[0x81, 0xc3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add dword ptr [rdx], 305419896
    // hlt
    pub(super) const ADD_4_MI_P: &[u8] = &[0x81, 0x02, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // add eax, -305419896
    // hlt
    pub(super) const ADD_4_RIA_N: &[u8] = &[0x05, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add ebx, -305419896
    // hlt
    pub(super) const ADD_4_RI_N: &[u8] = &[0x81, 0xc3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add dword ptr [rdx], -305419896
    // hlt
    pub(super) const ADD_4_MI_N: &[u8] = &[0x81, 0x02, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // add ebx, -3
    // hlt
    pub(super) const ADD_4_RIS: &[u8] = &[0x83, 0xc3, 0xfd, 0xf4];
    // add dword ptr [rdx], -3
    // hlt
    pub(super) const ADD_4_MIS: &[u8] = &[0x83, 0x02, 0xfd, 0xf4];
    // add ax, bx
    // hlt
    pub(super) const ADD_2_RR: &[u8] = &[0x66, 0x01, 0xd8, 0xf4];
    // add ax, word ptr [rdx]
    // hlt
    pub(super) const ADD_2_RM: &[u8] = &[0x66, 0x03, 0x02, 0xf4];
    // add word ptr [rdx], bx
    // hlt
    pub(super) const ADD_2_MR: &[u8] = &[0x66, 0x01, 0x1a, 0xf4];
    // add ax, 4660
    // hlt
    pub(super) const ADD_2_RIA_P: &[u8] = &[0x66, 0x05, 0x34, 0x12, 0xf4];
    // add bx, 4660
    // hlt
    pub(super) const ADD_2_RI_P: &[u8] = &[0x66, 0x81, 0xc3, 0x34, 0x12, 0xf4];
    // add word ptr [rdx], 4660
    // hlt
    pub(super) const ADD_2_MI_P: &[u8] = &[0x66, 0x81, 0x02, 0x34, 0x12, 0xf4];
    // add ax, -4660
    // hlt
    pub(super) const ADD_2_RIA_N: &[u8] = &[0x66, 0x05, 0xcc, 0xed, 0xf4];
    // add bx, -4660
    // hlt
    pub(super) const ADD_2_RI_N: &[u8] = &[0x66, 0x81, 0xc3, 0xcc, 0xed, 0xf4];
    // add word ptr [rdx], -4660
    // hlt
    pub(super) const ADD_2_MI_N: &[u8] = &[0x66, 0x81, 0x02, 0xcc, 0xed, 0xf4];
    // add bx, -3
    // hlt
    pub(super) const ADD_2_RIS: &[u8] = &[0x66, 0x83, 0xc3, 0xfd, 0xf4];
    // add word ptr [rdx], -3
    // hlt
    pub(super) const ADD_2_MIS: &[u8] = &[0x66, 0x83, 0x02, 0xfd, 0xf4];
    // add al, bl
    // hlt
    pub(super) const ADD_1_RR: &[u8] = &[0x00, 0xd8, 0xf4];
    // add al, byte ptr [rdx]
    // hlt
    pub(super) const ADD_1_RM: &[u8] = &[0x02, 0x02, 0xf4];
    // add byte ptr [rdx], bl
    // hlt
    pub(super) const ADD_1_MR: &[u8] = &[0x00, 0x1a, 0xf4];
    // add al, 53
    // hlt
    pub(super) const ADD_1_RIA_P: &[u8] = &[0x04, 0x35, 0xf4];
    // add bl, 53
    // hlt
    pub(super) const ADD_1_RI_P: &[u8] = &[0x80, 0xc3, 0x35, 0xf4];
    // add byte ptr [rdx], 53
    // hlt
    pub(super) const ADD_1_MI_P: &[u8] = &[0x80, 0x02, 0x35, 0xf4];
    // add al, -42
    // hlt
    pub(super) const ADD_1_RIA_N: &[u8] = &[0x04, 0xd6, 0xf4];
    // add bl, -42
    // hlt
    pub(super) const ADD_1_RI_N: &[u8] = &[0x80, 0xc3, 0xd6, 0xf4];
    // add byte ptr [rdx], -42
    // hlt
    pub(super) const ADD_1_MI_N: &[u8] = &[0x80, 0x02, 0xd6, 0xf4];
    // or rax, rbx
    // hlt
    pub(super) const OR_8_RR: &[u8] = &[0x48, 0x09, 0xd8, 0xf4];
    // or rax, qword ptr [rdx]
    // hlt
    pub(super) const OR_8_RM: &[u8] = &[0x48, 0x0b, 0x02, 0xf4];
    // or qword ptr [rdx], rbx
    // hlt
    pub(super) const OR_8_MR: &[u8] = &[0x48, 0x09, 0x1a, 0xf4];
    // or rax, 305419896
    // hlt
    pub(super) const OR_8_RIA_P: &[u8] = &[0x48, 0x0d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or rbx, 305419896
    // hlt
    pub(super) const OR_8_RI_P: &[u8] = &[0x48, 0x81, 0xcb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or qword ptr [rdx], 305419896
    // hlt
    pub(super) const OR_8_MI_P: &[u8] = &[0x48, 0x81, 0x0a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or rax, -305419896
    // hlt
    pub(super) const OR_8_RIA_N: &[u8] = &[0x48, 0x0d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or rbx, -305419896
    // hlt
    pub(super) const OR_8_RI_N: &[u8] = &[0x48, 0x81, 0xcb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or qword ptr [rdx], -305419896
    // hlt
    pub(super) const OR_8_MI_N: &[u8] = &[0x48, 0x81, 0x0a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or rbx, -3
    // hlt
    pub(super) const OR_8_RIS: &[u8] = &[0x48, 0x83, 0xcb, 0xfd, 0xf4];
    // or qword ptr [rdx], -3
    // hlt
    pub(super) const OR_8_MIS: &[u8] = &[0x48, 0x83, 0x0a, 0xfd, 0xf4];
    // or eax, ebx
    // hlt
    pub(super) const OR_4_RR: &[u8] = &[0x09, 0xd8, 0xf4];
    // or eax, dword ptr [rdx]
    // hlt
    pub(super) const OR_4_RM: &[u8] = &[0x0b, 0x02, 0xf4];
    // or dword ptr [rdx], ebx
    // hlt
    pub(super) const OR_4_MR: &[u8] = &[0x09, 0x1a, 0xf4];
    // or eax, 305419896
    // hlt
    pub(super) const OR_4_RIA_P: &[u8] = &[0x0d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or ebx, 305419896
    // hlt
    pub(super) const OR_4_RI_P: &[u8] = &[0x81, 0xcb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or dword ptr [rdx], 305419896
    // hlt
    pub(super) const OR_4_MI_P: &[u8] = &[0x81, 0x0a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // or eax, -305419896
    // hlt
    pub(super) const OR_4_RIA_N: &[u8] = &[0x0d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or ebx, -305419896
    // hlt
    pub(super) const OR_4_RI_N: &[u8] = &[0x81, 0xcb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or dword ptr [rdx], -305419896
    // hlt
    pub(super) const OR_4_MI_N: &[u8] = &[0x81, 0x0a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // or ebx, -3
    // hlt
    pub(super) const OR_4_RIS: &[u8] = &[0x83, 0xcb, 0xfd, 0xf4];
    // or dword ptr [rdx], -3
    // hlt
    pub(super) const OR_4_MIS: &[u8] = &[0x83, 0x0a, 0xfd, 0xf4];
    // or ax, bx
    // hlt
    pub(super) const OR_2_RR: &[u8] = &[0x66, 0x09, 0xd8, 0xf4];
    // or ax, word ptr [rdx]
    // hlt
    pub(super) const OR_2_RM: &[u8] = &[0x66, 0x0b, 0x02, 0xf4];
    // or word ptr [rdx], bx
    // hlt
    pub(super) const OR_2_MR: &[u8] = &[0x66, 0x09, 0x1a, 0xf4];
    // or ax, 4660
    // hlt
    pub(super) const OR_2_RIA_P: &[u8] = &[0x66, 0x0d, 0x34, 0x12, 0xf4];
    // or bx, 4660
    // hlt
    pub(super) const OR_2_RI_P: &[u8] = &[0x66, 0x81, 0xcb, 0x34, 0x12, 0xf4];
    // or word ptr [rdx], 4660
    // hlt
    pub(super) const OR_2_MI_P: &[u8] = &[0x66, 0x81, 0x0a, 0x34, 0x12, 0xf4];
    // or ax, -4660
    // hlt
    pub(super) const OR_2_RIA_N: &[u8] = &[0x66, 0x0d, 0xcc, 0xed, 0xf4];
    // or bx, -4660
    // hlt
    pub(super) const OR_2_RI_N: &[u8] = &[0x66, 0x81, 0xcb, 0xcc, 0xed, 0xf4];
    // or word ptr [rdx], -4660
    // hlt
    pub(super) const OR_2_MI_N: &[u8] = &[0x66, 0x81, 0x0a, 0xcc, 0xed, 0xf4];
    // or bx, -3
    // hlt
    pub(super) const OR_2_RIS: &[u8] = &[0x66, 0x83, 0xcb, 0xfd, 0xf4];
    // or word ptr [rdx], -3
    // hlt
    pub(super) const OR_2_MIS: &[u8] = &[0x66, 0x83, 0x0a, 0xfd, 0xf4];
    // or al, bl
    // hlt
    pub(super) const OR_1_RR: &[u8] = &[0x08, 0xd8, 0xf4];
    // or al, byte ptr [rdx]
    // hlt
    pub(super) const OR_1_RM: &[u8] = &[0x0a, 0x02, 0xf4];
    // or byte ptr [rdx], bl
    // hlt
    pub(super) const OR_1_MR: &[u8] = &[0x08, 0x1a, 0xf4];
    // or al, 53
    // hlt
    pub(super) const OR_1_RIA_P: &[u8] = &[0x0c, 0x35, 0xf4];
    // or bl, 53
    // hlt
    pub(super) const OR_1_RI_P: &[u8] = &[0x80, 0xcb, 0x35, 0xf4];
    // or byte ptr [rdx], 53
    // hlt
    pub(super) const OR_1_MI_P: &[u8] = &[0x80, 0x0a, 0x35, 0xf4];
    // or al, -42
    // hlt
    pub(super) const OR_1_RIA_N: &[u8] = &[0x0c, 0xd6, 0xf4];
    // or bl, -42
    // hlt
    pub(super) const OR_1_RI_N: &[u8] = &[0x80, 0xcb, 0xd6, 0xf4];
    // or byte ptr [rdx], -42
    // hlt
    pub(super) const OR_1_MI_N: &[u8] = &[0x80, 0x0a, 0xd6, 0xf4];
    // adc rax, rbx
    // hlt
    pub(super) const ADC_8_RR: &[u8] = &[0x48, 0x11, 0xd8, 0xf4];
    // adc rax, qword ptr [rdx]
    // hlt
    pub(super) const ADC_8_RM: &[u8] = &[0x48, 0x13, 0x02, 0xf4];
    // adc qword ptr [rdx], rbx
    // hlt
    pub(super) const ADC_8_MR: &[u8] = &[0x48, 0x11, 0x1a, 0xf4];
    // adc rax, 305419896
    // hlt
    pub(super) const ADC_8_RIA_P: &[u8] = &[0x48, 0x15, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc rbx, 305419896
    // hlt
    pub(super) const ADC_8_RI_P: &[u8] = &[0x48, 0x81, 0xd3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc qword ptr [rdx], 305419896
    // hlt
    pub(super) const ADC_8_MI_P: &[u8] = &[0x48, 0x81, 0x12, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc rax, -305419896
    // hlt
    pub(super) const ADC_8_RIA_N: &[u8] = &[0x48, 0x15, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc rbx, -305419896
    // hlt
    pub(super) const ADC_8_RI_N: &[u8] = &[0x48, 0x81, 0xd3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc qword ptr [rdx], -305419896
    // hlt
    pub(super) const ADC_8_MI_N: &[u8] = &[0x48, 0x81, 0x12, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc rbx, -3
    // hlt
    pub(super) const ADC_8_RIS: &[u8] = &[0x48, 0x83, 0xd3, 0xfd, 0xf4];
    // adc qword ptr [rdx], -3
    // hlt
    pub(super) const ADC_8_MIS: &[u8] = &[0x48, 0x83, 0x12, 0xfd, 0xf4];
    // adc eax, ebx
    // hlt
    pub(super) const ADC_4_RR: &[u8] = &[0x11, 0xd8, 0xf4];
    // adc eax, dword ptr [rdx]
    // hlt
    pub(super) const ADC_4_RM: &[u8] = &[0x13, 0x02, 0xf4];
    // adc dword ptr [rdx], ebx
    // hlt
    pub(super) const ADC_4_MR: &[u8] = &[0x11, 0x1a, 0xf4];
    // adc eax, 305419896
    // hlt
    pub(super) const ADC_4_RIA_P: &[u8] = &[0x15, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc ebx, 305419896
    // hlt
    pub(super) const ADC_4_RI_P: &[u8] = &[0x81, 0xd3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc dword ptr [rdx], 305419896
    // hlt
    pub(super) const ADC_4_MI_P: &[u8] = &[0x81, 0x12, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // adc eax, -305419896
    // hlt
    pub(super) const ADC_4_RIA_N: &[u8] = &[0x15, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc ebx, -305419896
    // hlt
    pub(super) const ADC_4_RI_N: &[u8] = &[0x81, 0xd3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc dword ptr [rdx], -305419896
    // hlt
    pub(super) const ADC_4_MI_N: &[u8] = &[0x81, 0x12, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // adc ebx, -3
    // hlt
    pub(super) const ADC_4_RIS: &[u8] = &[0x83, 0xd3, 0xfd, 0xf4];
    // adc dword ptr [rdx], -3
    // hlt
    pub(super) const ADC_4_MIS: &[u8] = &[0x83, 0x12, 0xfd, 0xf4];
    // adc ax, bx
    // hlt
    pub(super) const ADC_2_RR: &[u8] = &[0x66, 0x11, 0xd8, 0xf4];
    // adc ax, word ptr [rdx]
    // hlt
    pub(super) const ADC_2_RM: &[u8] = &[0x66, 0x13, 0x02, 0xf4];
    // adc word ptr [rdx], bx
    // hlt
    pub(super) const ADC_2_MR: &[u8] = &[0x66, 0x11, 0x1a, 0xf4];
    // adc ax, 4660
    // hlt
    pub(super) const ADC_2_RIA_P: &[u8] = &[0x66, 0x15, 0x34, 0x12, 0xf4];
    // adc bx, 4660
    // hlt
    pub(super) const ADC_2_RI_P: &[u8] = &[0x66, 0x81, 0xd3, 0x34, 0x12, 0xf4];
    // adc word ptr [rdx], 4660
    // hlt
    pub(super) const ADC_2_MI_P: &[u8] = &[0x66, 0x81, 0x12, 0x34, 0x12, 0xf4];
    // adc ax, -4660
    // hlt
    pub(super) const ADC_2_RIA_N: &[u8] = &[0x66, 0x15, 0xcc, 0xed, 0xf4];
    // adc bx, -4660
    // hlt
    pub(super) const ADC_2_RI_N: &[u8] = &[0x66, 0x81, 0xd3, 0xcc, 0xed, 0xf4];
    // adc word ptr [rdx], -4660
    // hlt
    pub(super) const ADC_2_MI_N: &[u8] = &[0x66, 0x81, 0x12, 0xcc, 0xed, 0xf4];
    // adc bx, -3
    // hlt
    pub(super) const ADC_2_RIS: &[u8] = &[0x66, 0x83, 0xd3, 0xfd, 0xf4];
    // adc word ptr [rdx], -3
    // hlt
    pub(super) const ADC_2_MIS: &[u8] = &[0x66, 0x83, 0x12, 0xfd, 0xf4];
    // adc al, bl
    // hlt
    pub(super) const ADC_1_RR: &[u8] = &[0x10, 0xd8, 0xf4];
    // adc al, byte ptr [rdx]
    // hlt
    pub(super) const ADC_1_RM: &[u8] = &[0x12, 0x02, 0xf4];
    // adc byte ptr [rdx], bl
    // hlt
    pub(super) const ADC_1_MR: &[u8] = &[0x10, 0x1a, 0xf4];
    // adc al, 53
    // hlt
    pub(super) const ADC_1_RIA_P: &[u8] = &[0x14, 0x35, 0xf4];
    // adc bl, 53
    // hlt
    pub(super) const ADC_1_RI_P: &[u8] = &[0x80, 0xd3, 0x35, 0xf4];
    // adc byte ptr [rdx], 53
    // hlt
    pub(super) const ADC_1_MI_P: &[u8] = &[0x80, 0x12, 0x35, 0xf4];
    // adc al, -42
    // hlt
    pub(super) const ADC_1_RIA_N: &[u8] = &[0x14, 0xd6, 0xf4];
    // adc bl, -42
    // hlt
    pub(super) const ADC_1_RI_N: &[u8] = &[0x80, 0xd3, 0xd6, 0xf4];
    // adc byte ptr [rdx], -42
    // hlt
    pub(super) const ADC_1_MI_N: &[u8] = &[0x80, 0x12, 0xd6, 0xf4];
    // sbb rax, rbx
    // hlt
    pub(super) const SBB_8_RR: &[u8] = &[0x48, 0x19, 0xd8, 0xf4];
    // sbb rax, qword ptr [rdx]
    // hlt
    pub(super) const SBB_8_RM: &[u8] = &[0x48, 0x1b, 0x02, 0xf4];
    // sbb qword ptr [rdx], rbx
    // hlt
    pub(super) const SBB_8_MR: &[u8] = &[0x48, 0x19, 0x1a, 0xf4];
    // sbb rax, 305419896
    // hlt
    pub(super) const SBB_8_RIA_P: &[u8] = &[0x48, 0x1d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb rbx, 305419896
    // hlt
    pub(super) const SBB_8_RI_P: &[u8] = &[0x48, 0x81, 0xdb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb qword ptr [rdx], 305419896
    // hlt
    pub(super) const SBB_8_MI_P: &[u8] = &[0x48, 0x81, 0x1a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb rax, -305419896
    // hlt
    pub(super) const SBB_8_RIA_N: &[u8] = &[0x48, 0x1d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb rbx, -305419896
    // hlt
    pub(super) const SBB_8_RI_N: &[u8] = &[0x48, 0x81, 0xdb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb qword ptr [rdx], -305419896
    // hlt
    pub(super) const SBB_8_MI_N: &[u8] = &[0x48, 0x81, 0x1a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb rbx, -3
    // hlt
    pub(super) const SBB_8_RIS: &[u8] = &[0x48, 0x83, 0xdb, 0xfd, 0xf4];
    // sbb qword ptr [rdx], -3
    // hlt
    pub(super) const SBB_8_MIS: &[u8] = &[0x48, 0x83, 0x1a, 0xfd, 0xf4];
    // sbb eax, ebx
    // hlt
    pub(super) const SBB_4_RR: &[u8] = &[0x19, 0xd8, 0xf4];
    // sbb eax, dword ptr [rdx]
    // hlt
    pub(super) const SBB_4_RM: &[u8] = &[0x1b, 0x02, 0xf4];
    // sbb dword ptr [rdx], ebx
    // hlt
    pub(super) const SBB_4_MR: &[u8] = &[0x19, 0x1a, 0xf4];
    // sbb eax, 305419896
    // hlt
    pub(super) const SBB_4_RIA_P: &[u8] = &[0x1d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb ebx, 305419896
    // hlt
    pub(super) const SBB_4_RI_P: &[u8] = &[0x81, 0xdb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb dword ptr [rdx], 305419896
    // hlt
    pub(super) const SBB_4_MI_P: &[u8] = &[0x81, 0x1a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sbb eax, -305419896
    // hlt
    pub(super) const SBB_4_RIA_N: &[u8] = &[0x1d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb ebx, -305419896
    // hlt
    pub(super) const SBB_4_RI_N: &[u8] = &[0x81, 0xdb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb dword ptr [rdx], -305419896
    // hlt
    pub(super) const SBB_4_MI_N: &[u8] = &[0x81, 0x1a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sbb ebx, -3
    // hlt
    pub(super) const SBB_4_RIS: &[u8] = &[0x83, 0xdb, 0xfd, 0xf4];
    // sbb dword ptr [rdx], -3
    // hlt
    pub(super) const SBB_4_MIS: &[u8] = &[0x83, 0x1a, 0xfd, 0xf4];
    // sbb ax, bx
    // hlt
    pub(super) const SBB_2_RR: &[u8] = &[0x66, 0x19, 0xd8, 0xf4];
    // sbb ax, word ptr [rdx]
    // hlt
    pub(super) const SBB_2_RM: &[u8] = &[0x66, 0x1b, 0x02, 0xf4];
    // sbb word ptr [rdx], bx
    // hlt
    pub(super) const SBB_2_MR: &[u8] = &[0x66, 0x19, 0x1a, 0xf4];
    // sbb ax, 4660
    // hlt
    pub(super) const SBB_2_RIA_P: &[u8] = &[0x66, 0x1d, 0x34, 0x12, 0xf4];
    // sbb bx, 4660
    // hlt
    pub(super) const SBB_2_RI_P: &[u8] = &[0x66, 0x81, 0xdb, 0x34, 0x12, 0xf4];
    // sbb word ptr [rdx], 4660
    // hlt
    pub(super) const SBB_2_MI_P: &[u8] = &[0x66, 0x81, 0x1a, 0x34, 0x12, 0xf4];
    // sbb ax, -4660
    // hlt
    pub(super) const SBB_2_RIA_N: &[u8] = &[0x66, 0x1d, 0xcc, 0xed, 0xf4];
    // sbb bx, -4660
    // hlt
    pub(super) const SBB_2_RI_N: &[u8] = &[0x66, 0x81, 0xdb, 0xcc, 0xed, 0xf4];
    // sbb word ptr [rdx], -4660
    // hlt
    pub(super) const SBB_2_MI_N: &[u8] = &[0x66, 0x81, 0x1a, 0xcc, 0xed, 0xf4];
    // sbb bx, -3
    // hlt
    pub(super) const SBB_2_RIS: &[u8] = &[0x66, 0x83, 0xdb, 0xfd, 0xf4];
    // sbb word ptr [rdx], -3
    // hlt
    pub(super) const SBB_2_MIS: &[u8] = &[0x66, 0x83, 0x1a, 0xfd, 0xf4];
    // sbb al, bl
    // hlt
    pub(super) const SBB_1_RR: &[u8] = &[0x18, 0xd8, 0xf4];
    // sbb al, byte ptr [rdx]
    // hlt
    pub(super) const SBB_1_RM: &[u8] = &[0x1a, 0x02, 0xf4];
    // sbb byte ptr [rdx], bl
    // hlt
    pub(super) const SBB_1_MR: &[u8] = &[0x18, 0x1a, 0xf4];
    // sbb al, 53
    // hlt
    pub(super) const SBB_1_RIA_P: &[u8] = &[0x1c, 0x35, 0xf4];
    // sbb bl, 53
    // hlt
    pub(super) const SBB_1_RI_P: &[u8] = &[0x80, 0xdb, 0x35, 0xf4];
    // sbb byte ptr [rdx], 53
    // hlt
    pub(super) const SBB_1_MI_P: &[u8] = &[0x80, 0x1a, 0x35, 0xf4];
    // sbb al, -42
    // hlt
    pub(super) const SBB_1_RIA_N: &[u8] = &[0x1c, 0xd6, 0xf4];
    // sbb bl, -42
    // hlt
    pub(super) const SBB_1_RI_N: &[u8] = &[0x80, 0xdb, 0xd6, 0xf4];
    // sbb byte ptr [rdx], -42
    // hlt
    pub(super) const SBB_1_MI_N: &[u8] = &[0x80, 0x1a, 0xd6, 0xf4];
    // and rax, rbx
    // hlt
    pub(super) const AND_8_RR: &[u8] = &[0x48, 0x21, 0xd8, 0xf4];
    // and rax, qword ptr [rdx]
    // hlt
    pub(super) const AND_8_RM: &[u8] = &[0x48, 0x23, 0x02, 0xf4];
    // and qword ptr [rdx], rbx
    // hlt
    pub(super) const AND_8_MR: &[u8] = &[0x48, 0x21, 0x1a, 0xf4];
    // and rax, 305419896
    // hlt
    pub(super) const AND_8_RIA_P: &[u8] = &[0x48, 0x25, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and rbx, 305419896
    // hlt
    pub(super) const AND_8_RI_P: &[u8] = &[0x48, 0x81, 0xe3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and qword ptr [rdx], 305419896
    // hlt
    pub(super) const AND_8_MI_P: &[u8] = &[0x48, 0x81, 0x22, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and rax, -305419896
    // hlt
    pub(super) const AND_8_RIA_N: &[u8] = &[0x48, 0x25, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and rbx, -305419896
    // hlt
    pub(super) const AND_8_RI_N: &[u8] = &[0x48, 0x81, 0xe3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and qword ptr [rdx], -305419896
    // hlt
    pub(super) const AND_8_MI_N: &[u8] = &[0x48, 0x81, 0x22, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and rbx, -3
    // hlt
    pub(super) const AND_8_RIS: &[u8] = &[0x48, 0x83, 0xe3, 0xfd, 0xf4];
    // and qword ptr [rdx], -3
    // hlt
    pub(super) const AND_8_MIS: &[u8] = &[0x48, 0x83, 0x22, 0xfd, 0xf4];
    // and eax, ebx
    // hlt
    pub(super) const AND_4_RR: &[u8] = &[0x21, 0xd8, 0xf4];
    // and eax, dword ptr [rdx]
    // hlt
    pub(super) const AND_4_RM: &[u8] = &[0x23, 0x02, 0xf4];
    // and dword ptr [rdx], ebx
    // hlt
    pub(super) const AND_4_MR: &[u8] = &[0x21, 0x1a, 0xf4];
    // and eax, 305419896
    // hlt
    pub(super) const AND_4_RIA_P: &[u8] = &[0x25, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and ebx, 305419896
    // hlt
    pub(super) const AND_4_RI_P: &[u8] = &[0x81, 0xe3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and dword ptr [rdx], 305419896
    // hlt
    pub(super) const AND_4_MI_P: &[u8] = &[0x81, 0x22, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // and eax, -305419896
    // hlt
    pub(super) const AND_4_RIA_N: &[u8] = &[0x25, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and ebx, -305419896
    // hlt
    pub(super) const AND_4_RI_N: &[u8] = &[0x81, 0xe3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and dword ptr [rdx], -305419896
    // hlt
    pub(super) const AND_4_MI_N: &[u8] = &[0x81, 0x22, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // and ebx, -3
    // hlt
    pub(super) const AND_4_RIS: &[u8] = &[0x83, 0xe3, 0xfd, 0xf4];
    // and dword ptr [rdx], -3
    // hlt
    pub(super) const AND_4_MIS: &[u8] = &[0x83, 0x22, 0xfd, 0xf4];
    // and ax, bx
    // hlt
    pub(super) const AND_2_RR: &[u8] = &[0x66, 0x21, 0xd8, 0xf4];
    // and ax, word ptr [rdx]
    // hlt
    pub(super) const AND_2_RM: &[u8] = &[0x66, 0x23, 0x02, 0xf4];
    // and word ptr [rdx], bx
    // hlt
    pub(super) const AND_2_MR: &[u8] = &[0x66, 0x21, 0x1a, 0xf4];
    // and ax, 4660
    // hlt
    pub(super) const AND_2_RIA_P: &[u8] = &[0x66, 0x25, 0x34, 0x12, 0xf4];
    // and bx, 4660
    // hlt
    pub(super) const AND_2_RI_P: &[u8] = &[0x66, 0x81, 0xe3, 0x34, 0x12, 0xf4];
    // and word ptr [rdx], 4660
    // hlt
    pub(super) const AND_2_MI_P: &[u8] = &[0x66, 0x81, 0x22, 0x34, 0x12, 0xf4];
    // and ax, -4660
    // hlt
    pub(super) const AND_2_RIA_N: &[u8] = &[0x66, 0x25, 0xcc, 0xed, 0xf4];
    // and bx, -4660
    // hlt
    pub(super) const AND_2_RI_N: &[u8] = &[0x66, 0x81, 0xe3, 0xcc, 0xed, 0xf4];
    // and word ptr [rdx], -4660
    // hlt
    pub(super) const AND_2_MI_N: &[u8] = &[0x66, 0x81, 0x22, 0xcc, 0xed, 0xf4];
    // and bx, -3
    // hlt
    pub(super) const AND_2_RIS: &[u8] = &[0x66, 0x83, 0xe3, 0xfd, 0xf4];
    // and word ptr [rdx], -3
    // hlt
    pub(super) const AND_2_MIS: &[u8] = &[0x66, 0x83, 0x22, 0xfd, 0xf4];
    // and al, bl
    // hlt
    pub(super) const AND_1_RR: &[u8] = &[0x20, 0xd8, 0xf4];
    // and al, byte ptr [rdx]
    // hlt
    pub(super) const AND_1_RM: &[u8] = &[0x22, 0x02, 0xf4];
    // and byte ptr [rdx], bl
    // hlt
    pub(super) const AND_1_MR: &[u8] = &[0x20, 0x1a, 0xf4];
    // and al, 53
    // hlt
    pub(super) const AND_1_RIA_P: &[u8] = &[0x24, 0x35, 0xf4];
    // and bl, 53
    // hlt
    pub(super) const AND_1_RI_P: &[u8] = &[0x80, 0xe3, 0x35, 0xf4];
    // and byte ptr [rdx], 53
    // hlt
    pub(super) const AND_1_MI_P: &[u8] = &[0x80, 0x22, 0x35, 0xf4];
    // and al, -42
    // hlt
    pub(super) const AND_1_RIA_N: &[u8] = &[0x24, 0xd6, 0xf4];
    // and bl, -42
    // hlt
    pub(super) const AND_1_RI_N: &[u8] = &[0x80, 0xe3, 0xd6, 0xf4];
    // and byte ptr [rdx], -42
    // hlt
    pub(super) const AND_1_MI_N: &[u8] = &[0x80, 0x22, 0xd6, 0xf4];
    // sub rax, rbx
    // hlt
    pub(super) const SUB_8_RR: &[u8] = &[0x48, 0x29, 0xd8, 0xf4];
    // sub rax, qword ptr [rdx]
    // hlt
    pub(super) const SUB_8_RM: &[u8] = &[0x48, 0x2b, 0x02, 0xf4];
    // sub qword ptr [rdx], rbx
    // hlt
    pub(super) const SUB_8_MR: &[u8] = &[0x48, 0x29, 0x1a, 0xf4];
    // sub rax, 305419896
    // hlt
    pub(super) const SUB_8_RIA_P: &[u8] = &[0x48, 0x2d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub rbx, 305419896
    // hlt
    pub(super) const SUB_8_RI_P: &[u8] = &[0x48, 0x81, 0xeb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub qword ptr [rdx], 305419896
    // hlt
    pub(super) const SUB_8_MI_P: &[u8] = &[0x48, 0x81, 0x2a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub rax, -305419896
    // hlt
    pub(super) const SUB_8_RIA_N: &[u8] = &[0x48, 0x2d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub rbx, -305419896
    // hlt
    pub(super) const SUB_8_RI_N: &[u8] = &[0x48, 0x81, 0xeb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub qword ptr [rdx], -305419896
    // hlt
    pub(super) const SUB_8_MI_N: &[u8] = &[0x48, 0x81, 0x2a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub rbx, -3
    // hlt
    pub(super) const SUB_8_RIS: &[u8] = &[0x48, 0x83, 0xeb, 0xfd, 0xf4];
    // sub qword ptr [rdx], -3
    // hlt
    pub(super) const SUB_8_MIS: &[u8] = &[0x48, 0x83, 0x2a, 0xfd, 0xf4];
    // sub eax, ebx
    // hlt
    pub(super) const SUB_4_RR: &[u8] = &[0x29, 0xd8, 0xf4];
    // sub eax, dword ptr [rdx]
    // hlt
    pub(super) const SUB_4_RM: &[u8] = &[0x2b, 0x02, 0xf4];
    // sub dword ptr [rdx], ebx
    // hlt
    pub(super) const SUB_4_MR: &[u8] = &[0x29, 0x1a, 0xf4];
    // sub eax, 305419896
    // hlt
    pub(super) const SUB_4_RIA_P: &[u8] = &[0x2d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub ebx, 305419896
    // hlt
    pub(super) const SUB_4_RI_P: &[u8] = &[0x81, 0xeb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub dword ptr [rdx], 305419896
    // hlt
    pub(super) const SUB_4_MI_P: &[u8] = &[0x81, 0x2a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // sub eax, -305419896
    // hlt
    pub(super) const SUB_4_RIA_N: &[u8] = &[0x2d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub ebx, -305419896
    // hlt
    pub(super) const SUB_4_RI_N: &[u8] = &[0x81, 0xeb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub dword ptr [rdx], -305419896
    // hlt
    pub(super) const SUB_4_MI_N: &[u8] = &[0x81, 0x2a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // sub ebx, -3
    // hlt
    pub(super) const SUB_4_RIS: &[u8] = &[0x83, 0xeb, 0xfd, 0xf4];
    // sub dword ptr [rdx], -3
    // hlt
    pub(super) const SUB_4_MIS: &[u8] = &[0x83, 0x2a, 0xfd, 0xf4];
    // sub ax, bx
    // hlt
    pub(super) const SUB_2_RR: &[u8] = &[0x66, 0x29, 0xd8, 0xf4];
    // sub ax, word ptr [rdx]
    // hlt
    pub(super) const SUB_2_RM: &[u8] = &[0x66, 0x2b, 0x02, 0xf4];
    // sub word ptr [rdx], bx
    // hlt
    pub(super) const SUB_2_MR: &[u8] = &[0x66, 0x29, 0x1a, 0xf4];
    // sub ax, 4660
    // hlt
    pub(super) const SUB_2_RIA_P: &[u8] = &[0x66, 0x2d, 0x34, 0x12, 0xf4];
    // sub bx, 4660
    // hlt
    pub(super) const SUB_2_RI_P: &[u8] = &[0x66, 0x81, 0xeb, 0x34, 0x12, 0xf4];
    // sub word ptr [rdx], 4660
    // hlt
    pub(super) const SUB_2_MI_P: &[u8] = &[0x66, 0x81, 0x2a, 0x34, 0x12, 0xf4];
    // sub ax, -4660
    // hlt
    pub(super) const SUB_2_RIA_N: &[u8] = &[0x66, 0x2d, 0xcc, 0xed, 0xf4];
    // sub bx, -4660
    // hlt
    pub(super) const SUB_2_RI_N: &[u8] = &[0x66, 0x81, 0xeb, 0xcc, 0xed, 0xf4];
    // sub word ptr [rdx], -4660
    // hlt
    pub(super) const SUB_2_MI_N: &[u8] = &[0x66, 0x81, 0x2a, 0xcc, 0xed, 0xf4];
    // sub bx, -3
    // hlt
    pub(super) const SUB_2_RIS: &[u8] = &[0x66, 0x83, 0xeb, 0xfd, 0xf4];
    // sub word ptr [rdx], -3
    // hlt
    pub(super) const SUB_2_MIS: &[u8] = &[0x66, 0x83, 0x2a, 0xfd, 0xf4];
    // sub al, bl
    // hlt
    pub(super) const SUB_1_RR: &[u8] = &[0x28, 0xd8, 0xf4];
    // sub al, byte ptr [rdx]
    // hlt
    pub(super) const SUB_1_RM: &[u8] = &[0x2a, 0x02, 0xf4];
    // sub byte ptr [rdx], bl
    // hlt
    pub(super) const SUB_1_MR: &[u8] = &[0x28, 0x1a, 0xf4];
    // sub al, 53
    // hlt
    pub(super) const SUB_1_RIA_P: &[u8] = &[0x2c, 0x35, 0xf4];
    // sub bl, 53
    // hlt
    pub(super) const SUB_1_RI_P: &[u8] = &[0x80, 0xeb, 0x35, 0xf4];
    // sub byte ptr [rdx], 53
    // hlt
    pub(super) const SUB_1_MI_P: &[u8] = &[0x80, 0x2a, 0x35, 0xf4];
    // sub al, -42
    // hlt
    pub(super) const SUB_1_RIA_N: &[u8] = &[0x2c, 0xd6, 0xf4];
    // sub bl, -42
    // hlt
    pub(super) const SUB_1_RI_N: &[u8] = &[0x80, 0xeb, 0xd6, 0xf4];
    // sub byte ptr [rdx], -42
    // hlt
    pub(super) const SUB_1_MI_N: &[u8] = &[0x80, 0x2a, 0xd6, 0xf4];
    // xor rax, rbx
    // hlt
    pub(super) const XOR_8_RR: &[u8] = &[0x48, 0x31, 0xd8, 0xf4];
    // xor rax, qword ptr [rdx]
    // hlt
    pub(super) const XOR_8_RM: &[u8] = &[0x48, 0x33, 0x02, 0xf4];
    // xor qword ptr [rdx], rbx
    // hlt
    pub(super) const XOR_8_MR: &[u8] = &[0x48, 0x31, 0x1a, 0xf4];
    // xor rax, 305419896
    // hlt
    pub(super) const XOR_8_RIA_P: &[u8] = &[0x48, 0x35, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor rbx, 305419896
    // hlt
    pub(super) const XOR_8_RI_P: &[u8] = &[0x48, 0x81, 0xf3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor qword ptr [rdx], 305419896
    // hlt
    pub(super) const XOR_8_MI_P: &[u8] = &[0x48, 0x81, 0x32, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor rax, -305419896
    // hlt
    pub(super) const XOR_8_RIA_N: &[u8] = &[0x48, 0x35, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor rbx, -305419896
    // hlt
    pub(super) const XOR_8_RI_N: &[u8] = &[0x48, 0x81, 0xf3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor qword ptr [rdx], -305419896
    // hlt
    pub(super) const XOR_8_MI_N: &[u8] = &[0x48, 0x81, 0x32, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor rbx, -3
    // hlt
    pub(super) const XOR_8_RIS: &[u8] = &[0x48, 0x83, 0xf3, 0xfd, 0xf4];
    // xor qword ptr [rdx], -3
    // hlt
    pub(super) const XOR_8_MIS: &[u8] = &[0x48, 0x83, 0x32, 0xfd, 0xf4];
    // xor eax, ebx
    // hlt
    pub(super) const XOR_4_RR: &[u8] = &[0x31, 0xd8, 0xf4];
    // xor eax, dword ptr [rdx]
    // hlt
    pub(super) const XOR_4_RM: &[u8] = &[0x33, 0x02, 0xf4];
    // xor dword ptr [rdx], ebx
    // hlt
    pub(super) const XOR_4_MR: &[u8] = &[0x31, 0x1a, 0xf4];
    // xor eax, 305419896
    // hlt
    pub(super) const XOR_4_RIA_P: &[u8] = &[0x35, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor ebx, 305419896
    // hlt
    pub(super) const XOR_4_RI_P: &[u8] = &[0x81, 0xf3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor dword ptr [rdx], 305419896
    // hlt
    pub(super) const XOR_4_MI_P: &[u8] = &[0x81, 0x32, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // xor eax, -305419896
    // hlt
    pub(super) const XOR_4_RIA_N: &[u8] = &[0x35, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor ebx, -305419896
    // hlt
    pub(super) const XOR_4_RI_N: &[u8] = &[0x81, 0xf3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor dword ptr [rdx], -305419896
    // hlt
    pub(super) const XOR_4_MI_N: &[u8] = &[0x81, 0x32, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // xor ebx, -3
    // hlt
    pub(super) const XOR_4_RIS: &[u8] = &[0x83, 0xf3, 0xfd, 0xf4];
    // xor dword ptr [rdx], -3
    // hlt
    pub(super) const XOR_4_MIS: &[u8] = &[0x83, 0x32, 0xfd, 0xf4];
    // xor ax, bx
    // hlt
    pub(super) const XOR_2_RR: &[u8] = &[0x66, 0x31, 0xd8, 0xf4];
    // xor ax, word ptr [rdx]
    // hlt
    pub(super) const XOR_2_RM: &[u8] = &[0x66, 0x33, 0x02, 0xf4];
    // xor word ptr [rdx], bx
    // hlt
    pub(super) const XOR_2_MR: &[u8] = &[0x66, 0x31, 0x1a, 0xf4];
    // xor ax, 4660
    // hlt
    pub(super) const XOR_2_RIA_P: &[u8] = &[0x66, 0x35, 0x34, 0x12, 0xf4];
    // xor bx, 4660
    // hlt
    pub(super) const XOR_2_RI_P: &[u8] = &[0x66, 0x81, 0xf3, 0x34, 0x12, 0xf4];
    // xor word ptr [rdx], 4660
    // hlt
    pub(super) const XOR_2_MI_P: &[u8] = &[0x66, 0x81, 0x32, 0x34, 0x12, 0xf4];
    // xor ax, -4660
    // hlt
    pub(super) const XOR_2_RIA_N: &[u8] = &[0x66, 0x35, 0xcc, 0xed, 0xf4];
    // xor bx, -4660
    // hlt
    pub(super) const XOR_2_RI_N: &[u8] = &[0x66, 0x81, 0xf3, 0xcc, 0xed, 0xf4];
    // xor word ptr [rdx], -4660
    // hlt
    pub(super) const XOR_2_MI_N: &[u8] = &[0x66, 0x81, 0x32, 0xcc, 0xed, 0xf4];
    // xor bx, -3
    // hlt
    pub(super) const XOR_2_RIS: &[u8] = &[0x66, 0x83, 0xf3, 0xfd, 0xf4];
    // xor word ptr [rdx], -3
    // hlt
    pub(super) const XOR_2_MIS: &[u8] = &[0x66, 0x83, 0x32, 0xfd, 0xf4];
    // xor al, bl
    // hlt
    pub(super) const XOR_1_RR: &[u8] = &[0x30, 0xd8, 0xf4];
    // xor al, byte ptr [rdx]
    // hlt
    pub(super) const XOR_1_RM: &[u8] = &[0x32, 0x02, 0xf4];
    // xor byte ptr [rdx], bl
    // hlt
    pub(super) const XOR_1_MR: &[u8] = &[0x30, 0x1a, 0xf4];
    // xor al, 53
    // hlt
    pub(super) const XOR_1_RIA_P: &[u8] = &[0x34, 0x35, 0xf4];
    // xor bl, 53
    // hlt
    pub(super) const XOR_1_RI_P: &[u8] = &[0x80, 0xf3, 0x35, 0xf4];
    // xor byte ptr [rdx], 53
    // hlt
    pub(super) const XOR_1_MI_P: &[u8] = &[0x80, 0x32, 0x35, 0xf4];
    // xor al, -42
    // hlt
    pub(super) const XOR_1_RIA_N: &[u8] = &[0x34, 0xd6, 0xf4];
    // xor bl, -42
    // hlt
    pub(super) const XOR_1_RI_N: &[u8] = &[0x80, 0xf3, 0xd6, 0xf4];
    // xor byte ptr [rdx], -42
    // hlt
    pub(super) const XOR_1_MI_N: &[u8] = &[0x80, 0x32, 0xd6, 0xf4];
    // cmp rax, rbx
    // hlt
    pub(super) const CMP_8_RR: &[u8] = &[0x48, 0x39, 0xd8, 0xf4];
    // cmp rax, qword ptr [rdx]
    // hlt
    pub(super) const CMP_8_RM: &[u8] = &[0x48, 0x3b, 0x02, 0xf4];
    // cmp qword ptr [rdx], rbx
    // hlt
    pub(super) const CMP_8_MR: &[u8] = &[0x48, 0x39, 0x1a, 0xf4];
    // cmp rax, 305419896
    // hlt
    pub(super) const CMP_8_RIA_P: &[u8] = &[0x48, 0x3d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp rbx, 305419896
    // hlt
    pub(super) const CMP_8_RI_P: &[u8] = &[0x48, 0x81, 0xfb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp qword ptr [rdx], 305419896
    // hlt
    pub(super) const CMP_8_MI_P: &[u8] = &[0x48, 0x81, 0x3a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp rax, -305419896
    // hlt
    pub(super) const CMP_8_RIA_N: &[u8] = &[0x48, 0x3d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp rbx, -305419896
    // hlt
    pub(super) const CMP_8_RI_N: &[u8] = &[0x48, 0x81, 0xfb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp qword ptr [rdx], -305419896
    // hlt
    pub(super) const CMP_8_MI_N: &[u8] = &[0x48, 0x81, 0x3a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp rbx, -3
    // hlt
    pub(super) const CMP_8_RIS: &[u8] = &[0x48, 0x83, 0xfb, 0xfd, 0xf4];
    // cmp qword ptr [rdx], -3
    // hlt
    pub(super) const CMP_8_MIS: &[u8] = &[0x48, 0x83, 0x3a, 0xfd, 0xf4];
    // cmp eax, ebx
    // hlt
    pub(super) const CMP_4_RR: &[u8] = &[0x39, 0xd8, 0xf4];
    // cmp eax, dword ptr [rdx]
    // hlt
    pub(super) const CMP_4_RM: &[u8] = &[0x3b, 0x02, 0xf4];
    // cmp dword ptr [rdx], ebx
    // hlt
    pub(super) const CMP_4_MR: &[u8] = &[0x39, 0x1a, 0xf4];
    // cmp eax, 305419896
    // hlt
    pub(super) const CMP_4_RIA_P: &[u8] = &[0x3d, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp ebx, 305419896
    // hlt
    pub(super) const CMP_4_RI_P: &[u8] = &[0x81, 0xfb, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp dword ptr [rdx], 305419896
    // hlt
    pub(super) const CMP_4_MI_P: &[u8] = &[0x81, 0x3a, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // cmp eax, -305419896
    // hlt
    pub(super) const CMP_4_RIA_N: &[u8] = &[0x3d, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp ebx, -305419896
    // hlt
    pub(super) const CMP_4_RI_N: &[u8] = &[0x81, 0xfb, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp dword ptr [rdx], -305419896
    // hlt
    pub(super) const CMP_4_MI_N: &[u8] = &[0x81, 0x3a, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // cmp ebx, -3
    // hlt
    pub(super) const CMP_4_RIS: &[u8] = &[0x83, 0xfb, 0xfd, 0xf4];
    // cmp dword ptr [rdx], -3
    // hlt
    pub(super) const CMP_4_MIS: &[u8] = &[0x83, 0x3a, 0xfd, 0xf4];
    // cmp ax, bx
    // hlt
    pub(super) const CMP_2_RR: &[u8] = &[0x66, 0x39, 0xd8, 0xf4];
    // cmp ax, word ptr [rdx]
    // hlt
    pub(super) const CMP_2_RM: &[u8] = &[0x66, 0x3b, 0x02, 0xf4];
    // cmp word ptr [rdx], bx
    // hlt
    pub(super) const CMP_2_MR: &[u8] = &[0x66, 0x39, 0x1a, 0xf4];
    // cmp ax, 4660
    // hlt
    pub(super) const CMP_2_RIA_P: &[u8] = &[0x66, 0x3d, 0x34, 0x12, 0xf4];
    // cmp bx, 4660
    // hlt
    pub(super) const CMP_2_RI_P: &[u8] = &[0x66, 0x81, 0xfb, 0x34, 0x12, 0xf4];
    // cmp word ptr [rdx], 4660
    // hlt
    pub(super) const CMP_2_MI_P: &[u8] = &[0x66, 0x81, 0x3a, 0x34, 0x12, 0xf4];
    // cmp ax, -4660
    // hlt
    pub(super) const CMP_2_RIA_N: &[u8] = &[0x66, 0x3d, 0xcc, 0xed, 0xf4];
    // cmp bx, -4660
    // hlt
    pub(super) const CMP_2_RI_N: &[u8] = &[0x66, 0x81, 0xfb, 0xcc, 0xed, 0xf4];
    // cmp word ptr [rdx], -4660
    // hlt
    pub(super) const CMP_2_MI_N: &[u8] = &[0x66, 0x81, 0x3a, 0xcc, 0xed, 0xf4];
    // cmp bx, -3
    // hlt
    pub(super) const CMP_2_RIS: &[u8] = &[0x66, 0x83, 0xfb, 0xfd, 0xf4];
    // cmp word ptr [rdx], -3
    // hlt
    pub(super) const CMP_2_MIS: &[u8] = &[0x66, 0x83, 0x3a, 0xfd, 0xf4];
    // cmp al, bl
    // hlt
    pub(super) const CMP_1_RR: &[u8] = &[0x38, 0xd8, 0xf4];
    // cmp al, byte ptr [rdx]
    // hlt
    pub(super) const CMP_1_RM: &[u8] = &[0x3a, 0x02, 0xf4];
    // cmp byte ptr [rdx], bl
    // hlt
    pub(super) const CMP_1_MR: &[u8] = &[0x38, 0x1a, 0xf4];
    // cmp al, 53
    // hlt
    pub(super) const CMP_1_RIA_P: &[u8] = &[0x3c, 0x35, 0xf4];
    // cmp bl, 53
    // hlt
    pub(super) const CMP_1_RI_P: &[u8] = &[0x80, 0xfb, 0x35, 0xf4];
    // cmp byte ptr [rdx], 53
    // hlt
    pub(super) const CMP_1_MI_P: &[u8] = &[0x80, 0x3a, 0x35, 0xf4];
    // cmp al, -42
    // hlt
    pub(super) const CMP_1_RIA_N: &[u8] = &[0x3c, 0xd6, 0xf4];
    // cmp bl, -42
    // hlt
    pub(super) const CMP_1_RI_N: &[u8] = &[0x80, 0xfb, 0xd6, 0xf4];
    // cmp byte ptr [rdx], -42
    // hlt
    pub(super) const CMP_1_MI_N: &[u8] = &[0x80, 0x3a, 0xd6, 0xf4];
    // test rax, rbx
    // hlt
    pub(super) const TEST_8_RR: &[u8] = &[0x48, 0x85, 0xd8, 0xf4];
    // test qword ptr [rdx], rbx
    // hlt
    pub(super) const TEST_8_MR: &[u8] = &[0x48, 0x85, 0x1a, 0xf4];
    // test rax, 305419896
    // hlt
    pub(super) const TEST_8_RIA_P: &[u8] = &[0x48, 0xa9, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test rbx, 305419896
    // hlt
    pub(super) const TEST_8_RI_P: &[u8] = &[0x48, 0xf7, 0xc3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test qword ptr [rdx], 305419896
    // hlt
    pub(super) const TEST_8_MI_P: &[u8] = &[0x48, 0xf7, 0x02, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test rax, -305419896
    // hlt
    pub(super) const TEST_8_RIA_N: &[u8] = &[0x48, 0xa9, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test rbx, -305419896
    // hlt
    pub(super) const TEST_8_RI_N: &[u8] = &[0x48, 0xf7, 0xc3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test qword ptr [rdx], -305419896
    // hlt
    pub(super) const TEST_8_MI_N: &[u8] = &[0x48, 0xf7, 0x02, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test eax, ebx
    // hlt
    pub(super) const TEST_4_RR: &[u8] = &[0x85, 0xd8, 0xf4];
    // test dword ptr [rdx], ebx
    // hlt
    pub(super) const TEST_4_MR: &[u8] = &[0x85, 0x1a, 0xf4];
    // test eax, 305419896
    // hlt
    pub(super) const TEST_4_RIA_P: &[u8] = &[0xa9, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test ebx, 305419896
    // hlt
    pub(super) const TEST_4_RI_P: &[u8] = &[0xf7, 0xc3, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test dword ptr [rdx], 305419896
    // hlt
    pub(super) const TEST_4_MI_P: &[u8] = &[0xf7, 0x02, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // test eax, -305419896
    // hlt
    pub(super) const TEST_4_RIA_N: &[u8] = &[0xa9, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test ebx, -305419896
    // hlt
    pub(super) const TEST_4_RI_N: &[u8] = &[0xf7, 0xc3, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test dword ptr [rdx], -305419896
    // hlt
    pub(super) const TEST_4_MI_N: &[u8] = &[0xf7, 0x02, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // test ax, bx
    // hlt
    pub(super) const TEST_2_RR: &[u8] = &[0x66, 0x85, 0xd8, 0xf4];
    // test word ptr [rdx], bx
    // hlt
    pub(super) const TEST_2_MR: &[u8] = &[0x66, 0x85, 0x1a, 0xf4];
    // test ax, 4660
    // hlt
    pub(super) const TEST_2_RIA_P: &[u8] = &[0x66, 0xa9, 0x34, 0x12, 0xf4];
    // test bx, 4660
    // hlt
    pub(super) const TEST_2_RI_P: &[u8] = &[0x66, 0xf7, 0xc3, 0x34, 0x12, 0xf4];
    // test word ptr [rdx], 4660
    // hlt
    pub(super) const TEST_2_MI_P: &[u8] = &[0x66, 0xf7, 0x02, 0x34, 0x12, 0xf4];
    // test ax, -4660
    // hlt
    pub(super) const TEST_2_RIA_N: &[u8] = &[0x66, 0xa9, 0xcc, 0xed, 0xf4];
    // test bx, -4660
    // hlt
    pub(super) const TEST_2_RI_N: &[u8] = &[0x66, 0xf7, 0xc3, 0xcc, 0xed, 0xf4];
    // test word ptr [rdx], -4660
    // hlt
    pub(super) const TEST_2_MI_N: &[u8] = &[0x66, 0xf7, 0x02, 0xcc, 0xed, 0xf4];
    // test al, bl
    // hlt
    pub(super) const TEST_1_RR: &[u8] = &[0x84, 0xd8, 0xf4];
    // test byte ptr [rdx], bl
    // hlt
    pub(super) const TEST_1_MR: &[u8] = &[0x84, 0x1a, 0xf4];
    // test al, 53
    // hlt
    pub(super) const TEST_1_RIA_P: &[u8] = &[0xa8, 0x35, 0xf4];
    // test bl, 53
    // hlt
    pub(super) const TEST_1_RI_P: &[u8] = &[0xf6, 0xc3, 0x35, 0xf4];
    // test byte ptr [rdx], 53
    // hlt
    pub(super) const TEST_1_MI_P: &[u8] = &[0xf6, 0x02, 0x35, 0xf4];
    // test al, -42
    // hlt
    pub(super) const TEST_1_RIA_N: &[u8] = &[0xa8, 0xd6, 0xf4];
    // test bl, -42
    // hlt
    pub(super) const TEST_1_RI_N: &[u8] = &[0xf6, 0xc3, 0xd6, 0xf4];
    // test byte ptr [rdx], -42
    // hlt
    pub(super) const TEST_1_MI_N: &[u8] = &[0xf6, 0x02, 0xd6, 0xf4];
    // bt rax, rbx
    // hlt
    pub(super) const BT_8_RR: &[u8] = &[0x48, 0x0f, 0xa3, 0xd8, 0xf4];
    // bt qword ptr [rdx], rbx
    // hlt
    pub(super) const BT_8_MR: &[u8] = &[0x48, 0x0f, 0xa3, 0x1a, 0xf4];
    // bt rax, 5
    // hlt
    pub(super) const BT_8_RI5: &[u8] = &[0x48, 0x0f, 0xba, 0xe0, 0x05, 0xf4];
    // bt rax, 0xc5
    // hlt
    pub(super) const BT_8_RIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0xe0, 0xc5, 0xf4];
    // bt qword ptr [rdx], 37
    // hlt
    pub(super) const BT_8_MI: &[u8] = &[0x48, 0x0f, 0xba, 0x22, 0x25, 0xf4];
    // bt qword ptr [rdx], 0xfd
    // hlt
    pub(super) const BT_8_MIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0x22, 0xfd, 0xf4];
    // bt eax, ebx
    // hlt
    pub(super) const BT_4_RR: &[u8] = &[0x0f, 0xa3, 0xd8, 0xf4];
    // bt dword ptr [rdx], ebx
    // hlt
    pub(super) const BT_4_MR: &[u8] = &[0x0f, 0xa3, 0x1a, 0xf4];
    // bt eax, 5
    // hlt
    pub(super) const BT_4_RI5: &[u8] = &[0x0f, 0xba, 0xe0, 0x05, 0xf4];
    // bt eax, 0xc5
    // hlt
    pub(super) const BT_4_RIBIG: &[u8] = &[0x0f, 0xba, 0xe0, 0xc5, 0xf4];
    // bt dword ptr [rdx], 37
    // hlt
    pub(super) const BT_4_MI: &[u8] = &[0x0f, 0xba, 0x22, 0x25, 0xf4];
    // bt dword ptr [rdx], 0xfd
    // hlt
    pub(super) const BT_4_MIBIG: &[u8] = &[0x0f, 0xba, 0x22, 0xfd, 0xf4];
    // bt ax, bx
    // hlt
    pub(super) const BT_2_RR: &[u8] = &[0x66, 0x0f, 0xa3, 0xd8, 0xf4];
    // bt word ptr [rdx], bx
    // hlt
    pub(super) const BT_2_MR: &[u8] = &[0x66, 0x0f, 0xa3, 0x1a, 0xf4];
    // bt ax, 5
    // hlt
    pub(super) const BT_2_RI5: &[u8] = &[0x66, 0x0f, 0xba, 0xe0, 0x05, 0xf4];
    // bt ax, 0xc5
    // hlt
    pub(super) const BT_2_RIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0xe0, 0xc5, 0xf4];
    // bt word ptr [rdx], 37
    // hlt
    pub(super) const BT_2_MI: &[u8] = &[0x66, 0x0f, 0xba, 0x22, 0x25, 0xf4];
    // bt word ptr [rdx], 0xfd
    // hlt
    pub(super) const BT_2_MIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0x22, 0xfd, 0xf4];
    // bts rax, rbx
    // hlt
    pub(super) const BTS_8_RR: &[u8] = &[0x48, 0x0f, 0xab, 0xd8, 0xf4];
    // bts qword ptr [rdx], rbx
    // hlt
    pub(super) const BTS_8_MR: &[u8] = &[0x48, 0x0f, 0xab, 0x1a, 0xf4];
    // bts rax, 5
    // hlt
    pub(super) const BTS_8_RI5: &[u8] = &[0x48, 0x0f, 0xba, 0xe8, 0x05, 0xf4];
    // bts rax, 0xc5
    // hlt
    pub(super) const BTS_8_RIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0xe8, 0xc5, 0xf4];
    // bts qword ptr [rdx], 37
    // hlt
    pub(super) const BTS_8_MI: &[u8] = &[0x48, 0x0f, 0xba, 0x2a, 0x25, 0xf4];
    // bts qword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTS_8_MIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0x2a, 0xfd, 0xf4];
    // bts eax, ebx
    // hlt
    pub(super) const BTS_4_RR: &[u8] = &[0x0f, 0xab, 0xd8, 0xf4];
    // bts dword ptr [rdx], ebx
    // hlt
    pub(super) const BTS_4_MR: &[u8] = &[0x0f, 0xab, 0x1a, 0xf4];
    // bts eax, 5
    // hlt
    pub(super) const BTS_4_RI5: &[u8] = &[0x0f, 0xba, 0xe8, 0x05, 0xf4];
    // bts eax, 0xc5
    // hlt
    pub(super) const BTS_4_RIBIG: &[u8] = &[0x0f, 0xba, 0xe8, 0xc5, 0xf4];
    // bts dword ptr [rdx], 37
    // hlt
    pub(super) const BTS_4_MI: &[u8] = &[0x0f, 0xba, 0x2a, 0x25, 0xf4];
    // bts dword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTS_4_MIBIG: &[u8] = &[0x0f, 0xba, 0x2a, 0xfd, 0xf4];
    // bts ax, bx
    // hlt
    pub(super) const BTS_2_RR: &[u8] = &[0x66, 0x0f, 0xab, 0xd8, 0xf4];
    // bts word ptr [rdx], bx
    // hlt
    pub(super) const BTS_2_MR: &[u8] = &[0x66, 0x0f, 0xab, 0x1a, 0xf4];
    // bts ax, 5
    // hlt
    pub(super) const BTS_2_RI5: &[u8] = &[0x66, 0x0f, 0xba, 0xe8, 0x05, 0xf4];
    // bts ax, 0xc5
    // hlt
    pub(super) const BTS_2_RIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0xe8, 0xc5, 0xf4];
    // bts word ptr [rdx], 37
    // hlt
    pub(super) const BTS_2_MI: &[u8] = &[0x66, 0x0f, 0xba, 0x2a, 0x25, 0xf4];
    // bts word ptr [rdx], 0xfd
    // hlt
    pub(super) const BTS_2_MIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0x2a, 0xfd, 0xf4];
    // btr rax, rbx
    // hlt
    pub(super) const BTR_8_RR: &[u8] = &[0x48, 0x0f, 0xb3, 0xd8, 0xf4];
    // btr qword ptr [rdx], rbx
    // hlt
    pub(super) const BTR_8_MR: &[u8] = &[0x48, 0x0f, 0xb3, 0x1a, 0xf4];
    // btr rax, 5
    // hlt
    pub(super) const BTR_8_RI5: &[u8] = &[0x48, 0x0f, 0xba, 0xf0, 0x05, 0xf4];
    // btr rax, 0xc5
    // hlt
    pub(super) const BTR_8_RIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0xf0, 0xc5, 0xf4];
    // btr qword ptr [rdx], 37
    // hlt
    pub(super) const BTR_8_MI: &[u8] = &[0x48, 0x0f, 0xba, 0x32, 0x25, 0xf4];
    // btr qword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTR_8_MIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0x32, 0xfd, 0xf4];
    // btr eax, ebx
    // hlt
    pub(super) const BTR_4_RR: &[u8] = &[0x0f, 0xb3, 0xd8, 0xf4];
    // btr dword ptr [rdx], ebx
    // hlt
    pub(super) const BTR_4_MR: &[u8] = &[0x0f, 0xb3, 0x1a, 0xf4];
    // btr eax, 5
    // hlt
    pub(super) const BTR_4_RI5: &[u8] = &[0x0f, 0xba, 0xf0, 0x05, 0xf4];
    // btr eax, 0xc5
    // hlt
    pub(super) const BTR_4_RIBIG: &[u8] = &[0x0f, 0xba, 0xf0, 0xc5, 0xf4];
    // btr dword ptr [rdx], 37
    // hlt
    pub(super) const BTR_4_MI: &[u8] = &[0x0f, 0xba, 0x32, 0x25, 0xf4];
    // btr dword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTR_4_MIBIG: &[u8] = &[0x0f, 0xba, 0x32, 0xfd, 0xf4];
    // btr ax, bx
    // hlt
    pub(super) const BTR_2_RR: &[u8] = &[0x66, 0x0f, 0xb3, 0xd8, 0xf4];
    // btr word ptr [rdx], bx
    // hlt
    pub(super) const BTR_2_MR: &[u8] = &[0x66, 0x0f, 0xb3, 0x1a, 0xf4];
    // btr ax, 5
    // hlt
    pub(super) const BTR_2_RI5: &[u8] = &[0x66, 0x0f, 0xba, 0xf0, 0x05, 0xf4];
    // btr ax, 0xc5
    // hlt
    pub(super) const BTR_2_RIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0xf0, 0xc5, 0xf4];
    // btr word ptr [rdx], 37
    // hlt
    pub(super) const BTR_2_MI: &[u8] = &[0x66, 0x0f, 0xba, 0x32, 0x25, 0xf4];
    // btr word ptr [rdx], 0xfd
    // hlt
    pub(super) const BTR_2_MIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0x32, 0xfd, 0xf4];
    // btc rax, rbx
    // hlt
    pub(super) const BTC_8_RR: &[u8] = &[0x48, 0x0f, 0xbb, 0xd8, 0xf4];
    // btc qword ptr [rdx], rbx
    // hlt
    pub(super) const BTC_8_MR: &[u8] = &[0x48, 0x0f, 0xbb, 0x1a, 0xf4];
    // btc rax, 5
    // hlt
    pub(super) const BTC_8_RI5: &[u8] = &[0x48, 0x0f, 0xba, 0xf8, 0x05, 0xf4];
    // btc rax, 0xc5
    // hlt
    pub(super) const BTC_8_RIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0xf8, 0xc5, 0xf4];
    // btc qword ptr [rdx], 37
    // hlt
    pub(super) const BTC_8_MI: &[u8] = &[0x48, 0x0f, 0xba, 0x3a, 0x25, 0xf4];
    // btc qword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTC_8_MIBIG: &[u8] = &[0x48, 0x0f, 0xba, 0x3a, 0xfd, 0xf4];
    // btc eax, ebx
    // hlt
    pub(super) const BTC_4_RR: &[u8] = &[0x0f, 0xbb, 0xd8, 0xf4];
    // btc dword ptr [rdx], ebx
    // hlt
    pub(super) const BTC_4_MR: &[u8] = &[0x0f, 0xbb, 0x1a, 0xf4];
    // btc eax, 5
    // hlt
    pub(super) const BTC_4_RI5: &[u8] = &[0x0f, 0xba, 0xf8, 0x05, 0xf4];
    // btc eax, 0xc5
    // hlt
    pub(super) const BTC_4_RIBIG: &[u8] = &[0x0f, 0xba, 0xf8, 0xc5, 0xf4];
    // btc dword ptr [rdx], 37
    // hlt
    pub(super) const BTC_4_MI: &[u8] = &[0x0f, 0xba, 0x3a, 0x25, 0xf4];
    // btc dword ptr [rdx], 0xfd
    // hlt
    pub(super) const BTC_4_MIBIG: &[u8] = &[0x0f, 0xba, 0x3a, 0xfd, 0xf4];
    // btc ax, bx
    // hlt
    pub(super) const BTC_2_RR: &[u8] = &[0x66, 0x0f, 0xbb, 0xd8, 0xf4];
    // btc word ptr [rdx], bx
    // hlt
    pub(super) const BTC_2_MR: &[u8] = &[0x66, 0x0f, 0xbb, 0x1a, 0xf4];
    // btc ax, 5
    // hlt
    pub(super) const BTC_2_RI5: &[u8] = &[0x66, 0x0f, 0xba, 0xf8, 0x05, 0xf4];
    // btc ax, 0xc5
    // hlt
    pub(super) const BTC_2_RIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0xf8, 0xc5, 0xf4];
    // btc word ptr [rdx], 37
    // hlt
    pub(super) const BTC_2_MI: &[u8] = &[0x66, 0x0f, 0xba, 0x3a, 0x25, 0xf4];
    // btc word ptr [rdx], 0xfd
    // hlt
    pub(super) const BTC_2_MIBIG: &[u8] = &[0x66, 0x0f, 0xba, 0x3a, 0xfd, 0xf4];
    // bsf rax, rbx
    // hlt
    pub(super) const BSF_8: &[u8] = &[0x48, 0x0f, 0xbc, 0xc3, 0xf4];
    // bsf rax, qword ptr [rdx]
    // hlt
    pub(super) const BSF_MEM_8: &[u8] = &[0x48, 0x0f, 0xbc, 0x02, 0xf4];
    // bsf eax, ebx
    // hlt
    pub(super) const BSF_4: &[u8] = &[0x0f, 0xbc, 0xc3, 0xf4];
    // bsf eax, dword ptr [rdx]
    // hlt
    pub(super) const BSF_MEM_4: &[u8] = &[0x0f, 0xbc, 0x02, 0xf4];
    // bsf ax, bx
    // hlt
    pub(super) const BSF_2: &[u8] = &[0x66, 0x0f, 0xbc, 0xc3, 0xf4];
    // bsf ax, word ptr [rdx]
    // hlt
    pub(super) const BSF_MEM_2: &[u8] = &[0x66, 0x0f, 0xbc, 0x02, 0xf4];
    // bsr rax, rbx
    // hlt
    pub(super) const BSR_8: &[u8] = &[0x48, 0x0f, 0xbd, 0xc3, 0xf4];
    // bsr rax, qword ptr [rdx]
    // hlt
    pub(super) const BSR_MEM_8: &[u8] = &[0x48, 0x0f, 0xbd, 0x02, 0xf4];
    // bsr eax, ebx
    // hlt
    pub(super) const BSR_4: &[u8] = &[0x0f, 0xbd, 0xc3, 0xf4];
    // bsr eax, dword ptr [rdx]
    // hlt
    pub(super) const BSR_MEM_4: &[u8] = &[0x0f, 0xbd, 0x02, 0xf4];
    // bsr ax, bx
    // hlt
    pub(super) const BSR_2: &[u8] = &[0x66, 0x0f, 0xbd, 0xc3, 0xf4];
    // bsr ax, word ptr [rdx]
    // hlt
    pub(super) const BSR_MEM_2: &[u8] = &[0x66, 0x0f, 0xbd, 0x02, 0xf4];
    // tzcnt rax, rbx
    // hlt
    pub(super) const TZCNT_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xbc, 0xc3, 0xf4];
    // lzcnt rax, rbx
    // hlt
    pub(super) const LZCNT_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xbd, 0xc3, 0xf4];
    // popcnt rax, rbx
    // hlt
    pub(super) const POPCNT_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xb8, 0xc3, 0xf4];
    // .byte 0x0f, 0xba, 0xc0, 0x05
    // hlt
    pub(super) const BT_GROUP8_REG0: &[u8] = &[0x0f, 0xba, 0xc0, 0x05, 0xf4];
    // .byte 0x0f, 0xba, 0x1a, 0x05
    // hlt
    pub(super) const BT_GROUP8_MEM3: &[u8] = &[0x0f, 0xba, 0x1a, 0x05, 0xf4];
    // .byte 0x0f, 0xba, 0x3a, 0x05
    // hlt
    pub(super) const BT_GROUP8_MEM7: &[u8] = &[0x0f, 0xba, 0x3a, 0x05, 0xf4];
    // imul rax, rbx
    // hlt
    pub(super) const IMUL2_8_RR: &[u8] = &[0x48, 0x0f, 0xaf, 0xc3, 0xf4];
    // imul rax, qword ptr [rdx]
    // hlt
    pub(super) const IMUL2_8_RM: &[u8] = &[0x48, 0x0f, 0xaf, 0x02, 0xf4];
    // imul rax, rbx, 7
    // hlt
    pub(super) const IMUL3_8_S: &[u8] = &[0x48, 0x6b, 0xc3, 0x07, 0xf4];
    // imul rax, qword ptr [rdx], 7
    // hlt
    pub(super) const IMUL3_8_S_MEM: &[u8] = &[0x48, 0x6b, 0x02, 0x07, 0xf4];
    // imul rax, rbx, 74565
    // hlt
    pub(super) const IMUL3_8_B: &[u8] = &[0x48, 0x69, 0xc3, 0x45, 0x23, 0x01, 0x00, 0xf4];
    // imul rax, qword ptr [rdx], 74565
    // hlt
    pub(super) const IMUL3_8_B_MEM: &[u8] = &[0x48, 0x69, 0x02, 0x45, 0x23, 0x01, 0x00, 0xf4];
    // imul rax, rbx, -3
    // hlt
    pub(super) const IMUL3_8_N: &[u8] = &[0x48, 0x6b, 0xc3, 0xfd, 0xf4];
    // imul rax, qword ptr [rdx], -3
    // hlt
    pub(super) const IMUL3_8_N_MEM: &[u8] = &[0x48, 0x6b, 0x02, 0xfd, 0xf4];
    // imul eax, ebx
    // hlt
    pub(super) const IMUL2_4_RR: &[u8] = &[0x0f, 0xaf, 0xc3, 0xf4];
    // imul eax, dword ptr [rdx]
    // hlt
    pub(super) const IMUL2_4_RM: &[u8] = &[0x0f, 0xaf, 0x02, 0xf4];
    // imul eax, ebx, 7
    // hlt
    pub(super) const IMUL3_4_S: &[u8] = &[0x6b, 0xc3, 0x07, 0xf4];
    // imul eax, dword ptr [rdx], 7
    // hlt
    pub(super) const IMUL3_4_S_MEM: &[u8] = &[0x6b, 0x02, 0x07, 0xf4];
    // imul eax, ebx, 74565
    // hlt
    pub(super) const IMUL3_4_B: &[u8] = &[0x69, 0xc3, 0x45, 0x23, 0x01, 0x00, 0xf4];
    // imul eax, dword ptr [rdx], 74565
    // hlt
    pub(super) const IMUL3_4_B_MEM: &[u8] = &[0x69, 0x02, 0x45, 0x23, 0x01, 0x00, 0xf4];
    // imul eax, ebx, -3
    // hlt
    pub(super) const IMUL3_4_N: &[u8] = &[0x6b, 0xc3, 0xfd, 0xf4];
    // imul eax, dword ptr [rdx], -3
    // hlt
    pub(super) const IMUL3_4_N_MEM: &[u8] = &[0x6b, 0x02, 0xfd, 0xf4];
    // imul ax, bx
    // hlt
    pub(super) const IMUL2_2_RR: &[u8] = &[0x66, 0x0f, 0xaf, 0xc3, 0xf4];
    // imul ax, word ptr [rdx]
    // hlt
    pub(super) const IMUL2_2_RM: &[u8] = &[0x66, 0x0f, 0xaf, 0x02, 0xf4];
    // imul ax, bx, 7
    // hlt
    pub(super) const IMUL3_2_S: &[u8] = &[0x66, 0x6b, 0xc3, 0x07, 0xf4];
    // imul ax, word ptr [rdx], 7
    // hlt
    pub(super) const IMUL3_2_S_MEM: &[u8] = &[0x66, 0x6b, 0x02, 0x07, 0xf4];
    // imul ax, bx, 300
    // hlt
    pub(super) const IMUL3_2_B: &[u8] = &[0x66, 0x69, 0xc3, 0x2c, 0x01, 0xf4];
    // imul ax, word ptr [rdx], 300
    // hlt
    pub(super) const IMUL3_2_B_MEM: &[u8] = &[0x66, 0x69, 0x02, 0x2c, 0x01, 0xf4];
    // imul ax, bx, -3
    // hlt
    pub(super) const IMUL3_2_N: &[u8] = &[0x66, 0x6b, 0xc3, 0xfd, 0xf4];
    // imul ax, word ptr [rdx], -3
    // hlt
    pub(super) const IMUL3_2_N_MEM: &[u8] = &[0x66, 0x6b, 0x02, 0xfd, 0xf4];
    // .byte 0x70, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_O: &[u8] = &[0x70, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x80, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_O: &[u8] = &[0x0f, 0x80, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // seto al
    // hlt
    pub(super) const SETCC_O: &[u8] = &[0x0f, 0x90, 0xc0, 0xf4];
    // seto byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_O: &[u8] = &[0x0f, 0x90, 0x02, 0xf4];
    // cmovo rax, rbx
    // hlt
    pub(super) const CMOV64_O: &[u8] = &[0x48, 0x0f, 0x40, 0xc3, 0xf4];
    // cmovo eax, ebx
    // hlt
    pub(super) const CMOV32_O: &[u8] = &[0x0f, 0x40, 0xc3, 0xf4];
    // cmovo ax, bx
    // hlt
    pub(super) const CMOV16_O: &[u8] = &[0x66, 0x0f, 0x40, 0xc3, 0xf4];
    // cmovo rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_O: &[u8] = &[0x48, 0x0f, 0x40, 0x02, 0xf4];
    // .byte 0x71, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_NO: &[u8] = &[0x71, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x81, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_NO: &[u8] = &[0x0f, 0x81, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setno al
    // hlt
    pub(super) const SETCC_NO: &[u8] = &[0x0f, 0x91, 0xc0, 0xf4];
    // setno byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_NO: &[u8] = &[0x0f, 0x91, 0x02, 0xf4];
    // cmovno rax, rbx
    // hlt
    pub(super) const CMOV64_NO: &[u8] = &[0x48, 0x0f, 0x41, 0xc3, 0xf4];
    // cmovno eax, ebx
    // hlt
    pub(super) const CMOV32_NO: &[u8] = &[0x0f, 0x41, 0xc3, 0xf4];
    // cmovno ax, bx
    // hlt
    pub(super) const CMOV16_NO: &[u8] = &[0x66, 0x0f, 0x41, 0xc3, 0xf4];
    // cmovno rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_NO: &[u8] = &[0x48, 0x0f, 0x41, 0x02, 0xf4];
    // .byte 0x72, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_B: &[u8] = &[0x72, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x82, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_B: &[u8] = &[0x0f, 0x82, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setb al
    // hlt
    pub(super) const SETCC_B: &[u8] = &[0x0f, 0x92, 0xc0, 0xf4];
    // setb byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_B: &[u8] = &[0x0f, 0x92, 0x02, 0xf4];
    // cmovb rax, rbx
    // hlt
    pub(super) const CMOV64_B: &[u8] = &[0x48, 0x0f, 0x42, 0xc3, 0xf4];
    // cmovb eax, ebx
    // hlt
    pub(super) const CMOV32_B: &[u8] = &[0x0f, 0x42, 0xc3, 0xf4];
    // cmovb ax, bx
    // hlt
    pub(super) const CMOV16_B: &[u8] = &[0x66, 0x0f, 0x42, 0xc3, 0xf4];
    // cmovb rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_B: &[u8] = &[0x48, 0x0f, 0x42, 0x02, 0xf4];
    // .byte 0x73, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_AE: &[u8] = &[0x73, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x83, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_AE: &[u8] = &[0x0f, 0x83, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setae al
    // hlt
    pub(super) const SETCC_AE: &[u8] = &[0x0f, 0x93, 0xc0, 0xf4];
    // setae byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_AE: &[u8] = &[0x0f, 0x93, 0x02, 0xf4];
    // cmovae rax, rbx
    // hlt
    pub(super) const CMOV64_AE: &[u8] = &[0x48, 0x0f, 0x43, 0xc3, 0xf4];
    // cmovae eax, ebx
    // hlt
    pub(super) const CMOV32_AE: &[u8] = &[0x0f, 0x43, 0xc3, 0xf4];
    // cmovae ax, bx
    // hlt
    pub(super) const CMOV16_AE: &[u8] = &[0x66, 0x0f, 0x43, 0xc3, 0xf4];
    // cmovae rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_AE: &[u8] = &[0x48, 0x0f, 0x43, 0x02, 0xf4];
    // .byte 0x74, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_E: &[u8] = &[0x74, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x84, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_E: &[u8] = &[0x0f, 0x84, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // sete al
    // hlt
    pub(super) const SETCC_E: &[u8] = &[0x0f, 0x94, 0xc0, 0xf4];
    // sete byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_E: &[u8] = &[0x0f, 0x94, 0x02, 0xf4];
    // cmove rax, rbx
    // hlt
    pub(super) const CMOV64_E: &[u8] = &[0x48, 0x0f, 0x44, 0xc3, 0xf4];
    // cmove eax, ebx
    // hlt
    pub(super) const CMOV32_E: &[u8] = &[0x0f, 0x44, 0xc3, 0xf4];
    // cmove ax, bx
    // hlt
    pub(super) const CMOV16_E: &[u8] = &[0x66, 0x0f, 0x44, 0xc3, 0xf4];
    // cmove rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_E: &[u8] = &[0x48, 0x0f, 0x44, 0x02, 0xf4];
    // .byte 0x75, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_NE: &[u8] = &[0x75, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x85, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_NE: &[u8] = &[0x0f, 0x85, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setne al
    // hlt
    pub(super) const SETCC_NE: &[u8] = &[0x0f, 0x95, 0xc0, 0xf4];
    // setne byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_NE: &[u8] = &[0x0f, 0x95, 0x02, 0xf4];
    // cmovne rax, rbx
    // hlt
    pub(super) const CMOV64_NE: &[u8] = &[0x48, 0x0f, 0x45, 0xc3, 0xf4];
    // cmovne eax, ebx
    // hlt
    pub(super) const CMOV32_NE: &[u8] = &[0x0f, 0x45, 0xc3, 0xf4];
    // cmovne ax, bx
    // hlt
    pub(super) const CMOV16_NE: &[u8] = &[0x66, 0x0f, 0x45, 0xc3, 0xf4];
    // cmovne rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_NE: &[u8] = &[0x48, 0x0f, 0x45, 0x02, 0xf4];
    // .byte 0x76, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_BE: &[u8] = &[0x76, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x86, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_BE: &[u8] = &[0x0f, 0x86, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setbe al
    // hlt
    pub(super) const SETCC_BE: &[u8] = &[0x0f, 0x96, 0xc0, 0xf4];
    // setbe byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_BE: &[u8] = &[0x0f, 0x96, 0x02, 0xf4];
    // cmovbe rax, rbx
    // hlt
    pub(super) const CMOV64_BE: &[u8] = &[0x48, 0x0f, 0x46, 0xc3, 0xf4];
    // cmovbe eax, ebx
    // hlt
    pub(super) const CMOV32_BE: &[u8] = &[0x0f, 0x46, 0xc3, 0xf4];
    // cmovbe ax, bx
    // hlt
    pub(super) const CMOV16_BE: &[u8] = &[0x66, 0x0f, 0x46, 0xc3, 0xf4];
    // cmovbe rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_BE: &[u8] = &[0x48, 0x0f, 0x46, 0x02, 0xf4];
    // .byte 0x77, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_A: &[u8] = &[0x77, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x87, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_A: &[u8] = &[0x0f, 0x87, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // seta al
    // hlt
    pub(super) const SETCC_A: &[u8] = &[0x0f, 0x97, 0xc0, 0xf4];
    // seta byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_A: &[u8] = &[0x0f, 0x97, 0x02, 0xf4];
    // cmova rax, rbx
    // hlt
    pub(super) const CMOV64_A: &[u8] = &[0x48, 0x0f, 0x47, 0xc3, 0xf4];
    // cmova eax, ebx
    // hlt
    pub(super) const CMOV32_A: &[u8] = &[0x0f, 0x47, 0xc3, 0xf4];
    // cmova ax, bx
    // hlt
    pub(super) const CMOV16_A: &[u8] = &[0x66, 0x0f, 0x47, 0xc3, 0xf4];
    // cmova rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_A: &[u8] = &[0x48, 0x0f, 0x47, 0x02, 0xf4];
    // .byte 0x78, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_S: &[u8] = &[0x78, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x88, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_S: &[u8] = &[0x0f, 0x88, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // sets al
    // hlt
    pub(super) const SETCC_S: &[u8] = &[0x0f, 0x98, 0xc0, 0xf4];
    // sets byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_S: &[u8] = &[0x0f, 0x98, 0x02, 0xf4];
    // cmovs rax, rbx
    // hlt
    pub(super) const CMOV64_S: &[u8] = &[0x48, 0x0f, 0x48, 0xc3, 0xf4];
    // cmovs eax, ebx
    // hlt
    pub(super) const CMOV32_S: &[u8] = &[0x0f, 0x48, 0xc3, 0xf4];
    // cmovs ax, bx
    // hlt
    pub(super) const CMOV16_S: &[u8] = &[0x66, 0x0f, 0x48, 0xc3, 0xf4];
    // cmovs rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_S: &[u8] = &[0x48, 0x0f, 0x48, 0x02, 0xf4];
    // .byte 0x79, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_NS: &[u8] = &[0x79, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x89, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_NS: &[u8] = &[0x0f, 0x89, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setns al
    // hlt
    pub(super) const SETCC_NS: &[u8] = &[0x0f, 0x99, 0xc0, 0xf4];
    // setns byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_NS: &[u8] = &[0x0f, 0x99, 0x02, 0xf4];
    // cmovns rax, rbx
    // hlt
    pub(super) const CMOV64_NS: &[u8] = &[0x48, 0x0f, 0x49, 0xc3, 0xf4];
    // cmovns eax, ebx
    // hlt
    pub(super) const CMOV32_NS: &[u8] = &[0x0f, 0x49, 0xc3, 0xf4];
    // cmovns ax, bx
    // hlt
    pub(super) const CMOV16_NS: &[u8] = &[0x66, 0x0f, 0x49, 0xc3, 0xf4];
    // cmovns rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_NS: &[u8] = &[0x48, 0x0f, 0x49, 0x02, 0xf4];
    // .byte 0x7a, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_P: &[u8] = &[0x7a, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8a, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_P: &[u8] = &[0x0f, 0x8a, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setp al
    // hlt
    pub(super) const SETCC_P: &[u8] = &[0x0f, 0x9a, 0xc0, 0xf4];
    // setp byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_P: &[u8] = &[0x0f, 0x9a, 0x02, 0xf4];
    // cmovp rax, rbx
    // hlt
    pub(super) const CMOV64_P: &[u8] = &[0x48, 0x0f, 0x4a, 0xc3, 0xf4];
    // cmovp eax, ebx
    // hlt
    pub(super) const CMOV32_P: &[u8] = &[0x0f, 0x4a, 0xc3, 0xf4];
    // cmovp ax, bx
    // hlt
    pub(super) const CMOV16_P: &[u8] = &[0x66, 0x0f, 0x4a, 0xc3, 0xf4];
    // cmovp rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_P: &[u8] = &[0x48, 0x0f, 0x4a, 0x02, 0xf4];
    // .byte 0x7b, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_NP: &[u8] = &[0x7b, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8b, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_NP: &[u8] = &[0x0f, 0x8b, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setnp al
    // hlt
    pub(super) const SETCC_NP: &[u8] = &[0x0f, 0x9b, 0xc0, 0xf4];
    // setnp byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_NP: &[u8] = &[0x0f, 0x9b, 0x02, 0xf4];
    // cmovnp rax, rbx
    // hlt
    pub(super) const CMOV64_NP: &[u8] = &[0x48, 0x0f, 0x4b, 0xc3, 0xf4];
    // cmovnp eax, ebx
    // hlt
    pub(super) const CMOV32_NP: &[u8] = &[0x0f, 0x4b, 0xc3, 0xf4];
    // cmovnp ax, bx
    // hlt
    pub(super) const CMOV16_NP: &[u8] = &[0x66, 0x0f, 0x4b, 0xc3, 0xf4];
    // cmovnp rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_NP: &[u8] = &[0x48, 0x0f, 0x4b, 0x02, 0xf4];
    // .byte 0x7c, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_L: &[u8] = &[0x7c, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8c, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_L: &[u8] = &[0x0f, 0x8c, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setl al
    // hlt
    pub(super) const SETCC_L: &[u8] = &[0x0f, 0x9c, 0xc0, 0xf4];
    // setl byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_L: &[u8] = &[0x0f, 0x9c, 0x02, 0xf4];
    // cmovl rax, rbx
    // hlt
    pub(super) const CMOV64_L: &[u8] = &[0x48, 0x0f, 0x4c, 0xc3, 0xf4];
    // cmovl eax, ebx
    // hlt
    pub(super) const CMOV32_L: &[u8] = &[0x0f, 0x4c, 0xc3, 0xf4];
    // cmovl ax, bx
    // hlt
    pub(super) const CMOV16_L: &[u8] = &[0x66, 0x0f, 0x4c, 0xc3, 0xf4];
    // cmovl rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_L: &[u8] = &[0x48, 0x0f, 0x4c, 0x02, 0xf4];
    // .byte 0x7d, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_GE: &[u8] = &[0x7d, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8d, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_GE: &[u8] = &[0x0f, 0x8d, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setge al
    // hlt
    pub(super) const SETCC_GE: &[u8] = &[0x0f, 0x9d, 0xc0, 0xf4];
    // setge byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_GE: &[u8] = &[0x0f, 0x9d, 0x02, 0xf4];
    // cmovge rax, rbx
    // hlt
    pub(super) const CMOV64_GE: &[u8] = &[0x48, 0x0f, 0x4d, 0xc3, 0xf4];
    // cmovge eax, ebx
    // hlt
    pub(super) const CMOV32_GE: &[u8] = &[0x0f, 0x4d, 0xc3, 0xf4];
    // cmovge ax, bx
    // hlt
    pub(super) const CMOV16_GE: &[u8] = &[0x66, 0x0f, 0x4d, 0xc3, 0xf4];
    // cmovge rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_GE: &[u8] = &[0x48, 0x0f, 0x4d, 0x02, 0xf4];
    // .byte 0x7e, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_LE: &[u8] = &[0x7e, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8e, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_LE: &[u8] = &[0x0f, 0x8e, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setle al
    // hlt
    pub(super) const SETCC_LE: &[u8] = &[0x0f, 0x9e, 0xc0, 0xf4];
    // setle byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_LE: &[u8] = &[0x0f, 0x9e, 0x02, 0xf4];
    // cmovle rax, rbx
    // hlt
    pub(super) const CMOV64_LE: &[u8] = &[0x48, 0x0f, 0x4e, 0xc3, 0xf4];
    // cmovle eax, ebx
    // hlt
    pub(super) const CMOV32_LE: &[u8] = &[0x0f, 0x4e, 0xc3, 0xf4];
    // cmovle ax, bx
    // hlt
    pub(super) const CMOV16_LE: &[u8] = &[0x66, 0x0f, 0x4e, 0xc3, 0xf4];
    // cmovle rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_LE: &[u8] = &[0x48, 0x0f, 0x4e, 0x02, 0xf4];
    // .byte 0x7f, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC8_G: &[u8] = &[0x7f, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x0f, 0x8f, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JCC32_G: &[u8] = &[0x0f, 0x8f, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // setg al
    // hlt
    pub(super) const SETCC_G: &[u8] = &[0x0f, 0x9f, 0xc0, 0xf4];
    // setg byte ptr [rdx]
    // hlt
    pub(super) const SETCC_MEM_G: &[u8] = &[0x0f, 0x9f, 0x02, 0xf4];
    // cmovg rax, rbx
    // hlt
    pub(super) const CMOV64_G: &[u8] = &[0x48, 0x0f, 0x4f, 0xc3, 0xf4];
    // cmovg eax, ebx
    // hlt
    pub(super) const CMOV32_G: &[u8] = &[0x0f, 0x4f, 0xc3, 0xf4];
    // cmovg ax, bx
    // hlt
    pub(super) const CMOV16_G: &[u8] = &[0x66, 0x0f, 0x4f, 0xc3, 0xf4];
    // cmovg rax, qword ptr [rdx]
    // hlt
    pub(super) const CMOV_MEM_G: &[u8] = &[0x48, 0x0f, 0x4f, 0x02, 0xf4];
    // je target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const JE_REAL: &[u8] = &[0x74, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jne target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const JNE_REAL: &[u8] = &[0x75, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jg target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const JG_REAL: &[u8] = &[0x7f, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jb target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const JB_REAL: &[u8] = &[0x72, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // rol rax, cl
    // hlt
    pub(super) const ROL_8_CL: &[u8] = &[0x48, 0xd3, 0xc0, 0xf4];
    // rol rax, 1
    // hlt
    pub(super) const ROL_8_ONE: &[u8] = &[0x48, 0xd1, 0xc0, 0xf4];
    // rol rax, 3
    // hlt
    pub(super) const ROL_8_I3: &[u8] = &[0x48, 0xc1, 0xc0, 0x03, 0xf4];
    // rol rax, 17
    // hlt
    pub(super) const ROL_8_I17: &[u8] = &[0x48, 0xc1, 0xc0, 0x11, 0xf4];
    // rol rax, 33
    // hlt
    pub(super) const ROL_8_I33: &[u8] = &[0x48, 0xc1, 0xc0, 0x21, 0xf4];
    // rol qword ptr [rdx], cl
    // hlt
    pub(super) const ROL_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x02, 0xf4];
    // rol qword ptr [rdx], 1
    // hlt
    pub(super) const ROL_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x02, 0xf4];
    // rol eax, cl
    // hlt
    pub(super) const ROL_4_CL: &[u8] = &[0xd3, 0xc0, 0xf4];
    // rol eax, 1
    // hlt
    pub(super) const ROL_4_ONE: &[u8] = &[0xd1, 0xc0, 0xf4];
    // rol eax, 3
    // hlt
    pub(super) const ROL_4_I3: &[u8] = &[0xc1, 0xc0, 0x03, 0xf4];
    // rol eax, 17
    // hlt
    pub(super) const ROL_4_I17: &[u8] = &[0xc1, 0xc0, 0x11, 0xf4];
    // rol eax, 33
    // hlt
    pub(super) const ROL_4_I33: &[u8] = &[0xc1, 0xc0, 0x21, 0xf4];
    // rol dword ptr [rdx], cl
    // hlt
    pub(super) const ROL_4_MEM_CL: &[u8] = &[0xd3, 0x02, 0xf4];
    // rol dword ptr [rdx], 1
    // hlt
    pub(super) const ROL_4_MEM_ONE: &[u8] = &[0xd1, 0x02, 0xf4];
    // rol ax, cl
    // hlt
    pub(super) const ROL_2_CL: &[u8] = &[0x66, 0xd3, 0xc0, 0xf4];
    // rol ax, 1
    // hlt
    pub(super) const ROL_2_ONE: &[u8] = &[0x66, 0xd1, 0xc0, 0xf4];
    // rol ax, 3
    // hlt
    pub(super) const ROL_2_I3: &[u8] = &[0x66, 0xc1, 0xc0, 0x03, 0xf4];
    // rol ax, 17
    // hlt
    pub(super) const ROL_2_I17: &[u8] = &[0x66, 0xc1, 0xc0, 0x11, 0xf4];
    // rol ax, 33
    // hlt
    pub(super) const ROL_2_I33: &[u8] = &[0x66, 0xc1, 0xc0, 0x21, 0xf4];
    // rol word ptr [rdx], cl
    // hlt
    pub(super) const ROL_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x02, 0xf4];
    // rol word ptr [rdx], 1
    // hlt
    pub(super) const ROL_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x02, 0xf4];
    // rol al, cl
    // hlt
    pub(super) const ROL_1_CL: &[u8] = &[0xd2, 0xc0, 0xf4];
    // rol al, 1
    // hlt
    pub(super) const ROL_1_ONE: &[u8] = &[0xd0, 0xc0, 0xf4];
    // rol al, 3
    // hlt
    pub(super) const ROL_1_I3: &[u8] = &[0xc0, 0xc0, 0x03, 0xf4];
    // rol al, 17
    // hlt
    pub(super) const ROL_1_I17: &[u8] = &[0xc0, 0xc0, 0x11, 0xf4];
    // rol al, 33
    // hlt
    pub(super) const ROL_1_I33: &[u8] = &[0xc0, 0xc0, 0x21, 0xf4];
    // rol byte ptr [rdx], cl
    // hlt
    pub(super) const ROL_1_MEM_CL: &[u8] = &[0xd2, 0x02, 0xf4];
    // rol byte ptr [rdx], 1
    // hlt
    pub(super) const ROL_1_MEM_ONE: &[u8] = &[0xd0, 0x02, 0xf4];
    // ror rax, cl
    // hlt
    pub(super) const ROR_8_CL: &[u8] = &[0x48, 0xd3, 0xc8, 0xf4];
    // ror rax, 1
    // hlt
    pub(super) const ROR_8_ONE: &[u8] = &[0x48, 0xd1, 0xc8, 0xf4];
    // ror rax, 3
    // hlt
    pub(super) const ROR_8_I3: &[u8] = &[0x48, 0xc1, 0xc8, 0x03, 0xf4];
    // ror rax, 17
    // hlt
    pub(super) const ROR_8_I17: &[u8] = &[0x48, 0xc1, 0xc8, 0x11, 0xf4];
    // ror rax, 33
    // hlt
    pub(super) const ROR_8_I33: &[u8] = &[0x48, 0xc1, 0xc8, 0x21, 0xf4];
    // ror qword ptr [rdx], cl
    // hlt
    pub(super) const ROR_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x0a, 0xf4];
    // ror qword ptr [rdx], 1
    // hlt
    pub(super) const ROR_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x0a, 0xf4];
    // ror eax, cl
    // hlt
    pub(super) const ROR_4_CL: &[u8] = &[0xd3, 0xc8, 0xf4];
    // ror eax, 1
    // hlt
    pub(super) const ROR_4_ONE: &[u8] = &[0xd1, 0xc8, 0xf4];
    // ror eax, 3
    // hlt
    pub(super) const ROR_4_I3: &[u8] = &[0xc1, 0xc8, 0x03, 0xf4];
    // ror eax, 17
    // hlt
    pub(super) const ROR_4_I17: &[u8] = &[0xc1, 0xc8, 0x11, 0xf4];
    // ror eax, 33
    // hlt
    pub(super) const ROR_4_I33: &[u8] = &[0xc1, 0xc8, 0x21, 0xf4];
    // ror dword ptr [rdx], cl
    // hlt
    pub(super) const ROR_4_MEM_CL: &[u8] = &[0xd3, 0x0a, 0xf4];
    // ror dword ptr [rdx], 1
    // hlt
    pub(super) const ROR_4_MEM_ONE: &[u8] = &[0xd1, 0x0a, 0xf4];
    // ror ax, cl
    // hlt
    pub(super) const ROR_2_CL: &[u8] = &[0x66, 0xd3, 0xc8, 0xf4];
    // ror ax, 1
    // hlt
    pub(super) const ROR_2_ONE: &[u8] = &[0x66, 0xd1, 0xc8, 0xf4];
    // ror ax, 3
    // hlt
    pub(super) const ROR_2_I3: &[u8] = &[0x66, 0xc1, 0xc8, 0x03, 0xf4];
    // ror ax, 17
    // hlt
    pub(super) const ROR_2_I17: &[u8] = &[0x66, 0xc1, 0xc8, 0x11, 0xf4];
    // ror ax, 33
    // hlt
    pub(super) const ROR_2_I33: &[u8] = &[0x66, 0xc1, 0xc8, 0x21, 0xf4];
    // ror word ptr [rdx], cl
    // hlt
    pub(super) const ROR_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x0a, 0xf4];
    // ror word ptr [rdx], 1
    // hlt
    pub(super) const ROR_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x0a, 0xf4];
    // ror al, cl
    // hlt
    pub(super) const ROR_1_CL: &[u8] = &[0xd2, 0xc8, 0xf4];
    // ror al, 1
    // hlt
    pub(super) const ROR_1_ONE: &[u8] = &[0xd0, 0xc8, 0xf4];
    // ror al, 3
    // hlt
    pub(super) const ROR_1_I3: &[u8] = &[0xc0, 0xc8, 0x03, 0xf4];
    // ror al, 17
    // hlt
    pub(super) const ROR_1_I17: &[u8] = &[0xc0, 0xc8, 0x11, 0xf4];
    // ror al, 33
    // hlt
    pub(super) const ROR_1_I33: &[u8] = &[0xc0, 0xc8, 0x21, 0xf4];
    // ror byte ptr [rdx], cl
    // hlt
    pub(super) const ROR_1_MEM_CL: &[u8] = &[0xd2, 0x0a, 0xf4];
    // ror byte ptr [rdx], 1
    // hlt
    pub(super) const ROR_1_MEM_ONE: &[u8] = &[0xd0, 0x0a, 0xf4];
    // rcl rax, cl
    // hlt
    pub(super) const RCL_8_CL: &[u8] = &[0x48, 0xd3, 0xd0, 0xf4];
    // rcl rax, 1
    // hlt
    pub(super) const RCL_8_ONE: &[u8] = &[0x48, 0xd1, 0xd0, 0xf4];
    // rcl rax, 3
    // hlt
    pub(super) const RCL_8_I3: &[u8] = &[0x48, 0xc1, 0xd0, 0x03, 0xf4];
    // rcl rax, 17
    // hlt
    pub(super) const RCL_8_I17: &[u8] = &[0x48, 0xc1, 0xd0, 0x11, 0xf4];
    // rcl rax, 33
    // hlt
    pub(super) const RCL_8_I33: &[u8] = &[0x48, 0xc1, 0xd0, 0x21, 0xf4];
    // rcl qword ptr [rdx], cl
    // hlt
    pub(super) const RCL_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x12, 0xf4];
    // rcl qword ptr [rdx], 1
    // hlt
    pub(super) const RCL_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x12, 0xf4];
    // rcl eax, cl
    // hlt
    pub(super) const RCL_4_CL: &[u8] = &[0xd3, 0xd0, 0xf4];
    // rcl eax, 1
    // hlt
    pub(super) const RCL_4_ONE: &[u8] = &[0xd1, 0xd0, 0xf4];
    // rcl eax, 3
    // hlt
    pub(super) const RCL_4_I3: &[u8] = &[0xc1, 0xd0, 0x03, 0xf4];
    // rcl eax, 17
    // hlt
    pub(super) const RCL_4_I17: &[u8] = &[0xc1, 0xd0, 0x11, 0xf4];
    // rcl eax, 33
    // hlt
    pub(super) const RCL_4_I33: &[u8] = &[0xc1, 0xd0, 0x21, 0xf4];
    // rcl dword ptr [rdx], cl
    // hlt
    pub(super) const RCL_4_MEM_CL: &[u8] = &[0xd3, 0x12, 0xf4];
    // rcl dword ptr [rdx], 1
    // hlt
    pub(super) const RCL_4_MEM_ONE: &[u8] = &[0xd1, 0x12, 0xf4];
    // rcl ax, cl
    // hlt
    pub(super) const RCL_2_CL: &[u8] = &[0x66, 0xd3, 0xd0, 0xf4];
    // rcl ax, 1
    // hlt
    pub(super) const RCL_2_ONE: &[u8] = &[0x66, 0xd1, 0xd0, 0xf4];
    // rcl ax, 3
    // hlt
    pub(super) const RCL_2_I3: &[u8] = &[0x66, 0xc1, 0xd0, 0x03, 0xf4];
    // rcl ax, 17
    // hlt
    pub(super) const RCL_2_I17: &[u8] = &[0x66, 0xc1, 0xd0, 0x11, 0xf4];
    // rcl ax, 33
    // hlt
    pub(super) const RCL_2_I33: &[u8] = &[0x66, 0xc1, 0xd0, 0x21, 0xf4];
    // rcl word ptr [rdx], cl
    // hlt
    pub(super) const RCL_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x12, 0xf4];
    // rcl word ptr [rdx], 1
    // hlt
    pub(super) const RCL_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x12, 0xf4];
    // rcl al, cl
    // hlt
    pub(super) const RCL_1_CL: &[u8] = &[0xd2, 0xd0, 0xf4];
    // rcl al, 1
    // hlt
    pub(super) const RCL_1_ONE: &[u8] = &[0xd0, 0xd0, 0xf4];
    // rcl al, 3
    // hlt
    pub(super) const RCL_1_I3: &[u8] = &[0xc0, 0xd0, 0x03, 0xf4];
    // rcl al, 17
    // hlt
    pub(super) const RCL_1_I17: &[u8] = &[0xc0, 0xd0, 0x11, 0xf4];
    // rcl al, 33
    // hlt
    pub(super) const RCL_1_I33: &[u8] = &[0xc0, 0xd0, 0x21, 0xf4];
    // rcl byte ptr [rdx], cl
    // hlt
    pub(super) const RCL_1_MEM_CL: &[u8] = &[0xd2, 0x12, 0xf4];
    // rcl byte ptr [rdx], 1
    // hlt
    pub(super) const RCL_1_MEM_ONE: &[u8] = &[0xd0, 0x12, 0xf4];
    // rcr rax, cl
    // hlt
    pub(super) const RCR_8_CL: &[u8] = &[0x48, 0xd3, 0xd8, 0xf4];
    // rcr rax, 1
    // hlt
    pub(super) const RCR_8_ONE: &[u8] = &[0x48, 0xd1, 0xd8, 0xf4];
    // rcr rax, 3
    // hlt
    pub(super) const RCR_8_I3: &[u8] = &[0x48, 0xc1, 0xd8, 0x03, 0xf4];
    // rcr rax, 17
    // hlt
    pub(super) const RCR_8_I17: &[u8] = &[0x48, 0xc1, 0xd8, 0x11, 0xf4];
    // rcr rax, 33
    // hlt
    pub(super) const RCR_8_I33: &[u8] = &[0x48, 0xc1, 0xd8, 0x21, 0xf4];
    // rcr qword ptr [rdx], cl
    // hlt
    pub(super) const RCR_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x1a, 0xf4];
    // rcr qword ptr [rdx], 1
    // hlt
    pub(super) const RCR_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x1a, 0xf4];
    // rcr eax, cl
    // hlt
    pub(super) const RCR_4_CL: &[u8] = &[0xd3, 0xd8, 0xf4];
    // rcr eax, 1
    // hlt
    pub(super) const RCR_4_ONE: &[u8] = &[0xd1, 0xd8, 0xf4];
    // rcr eax, 3
    // hlt
    pub(super) const RCR_4_I3: &[u8] = &[0xc1, 0xd8, 0x03, 0xf4];
    // rcr eax, 17
    // hlt
    pub(super) const RCR_4_I17: &[u8] = &[0xc1, 0xd8, 0x11, 0xf4];
    // rcr eax, 33
    // hlt
    pub(super) const RCR_4_I33: &[u8] = &[0xc1, 0xd8, 0x21, 0xf4];
    // rcr dword ptr [rdx], cl
    // hlt
    pub(super) const RCR_4_MEM_CL: &[u8] = &[0xd3, 0x1a, 0xf4];
    // rcr dword ptr [rdx], 1
    // hlt
    pub(super) const RCR_4_MEM_ONE: &[u8] = &[0xd1, 0x1a, 0xf4];
    // rcr ax, cl
    // hlt
    pub(super) const RCR_2_CL: &[u8] = &[0x66, 0xd3, 0xd8, 0xf4];
    // rcr ax, 1
    // hlt
    pub(super) const RCR_2_ONE: &[u8] = &[0x66, 0xd1, 0xd8, 0xf4];
    // rcr ax, 3
    // hlt
    pub(super) const RCR_2_I3: &[u8] = &[0x66, 0xc1, 0xd8, 0x03, 0xf4];
    // rcr ax, 17
    // hlt
    pub(super) const RCR_2_I17: &[u8] = &[0x66, 0xc1, 0xd8, 0x11, 0xf4];
    // rcr ax, 33
    // hlt
    pub(super) const RCR_2_I33: &[u8] = &[0x66, 0xc1, 0xd8, 0x21, 0xf4];
    // rcr word ptr [rdx], cl
    // hlt
    pub(super) const RCR_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x1a, 0xf4];
    // rcr word ptr [rdx], 1
    // hlt
    pub(super) const RCR_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x1a, 0xf4];
    // rcr al, cl
    // hlt
    pub(super) const RCR_1_CL: &[u8] = &[0xd2, 0xd8, 0xf4];
    // rcr al, 1
    // hlt
    pub(super) const RCR_1_ONE: &[u8] = &[0xd0, 0xd8, 0xf4];
    // rcr al, 3
    // hlt
    pub(super) const RCR_1_I3: &[u8] = &[0xc0, 0xd8, 0x03, 0xf4];
    // rcr al, 17
    // hlt
    pub(super) const RCR_1_I17: &[u8] = &[0xc0, 0xd8, 0x11, 0xf4];
    // rcr al, 33
    // hlt
    pub(super) const RCR_1_I33: &[u8] = &[0xc0, 0xd8, 0x21, 0xf4];
    // rcr byte ptr [rdx], cl
    // hlt
    pub(super) const RCR_1_MEM_CL: &[u8] = &[0xd2, 0x1a, 0xf4];
    // rcr byte ptr [rdx], 1
    // hlt
    pub(super) const RCR_1_MEM_ONE: &[u8] = &[0xd0, 0x1a, 0xf4];
    // shl rax, cl
    // hlt
    pub(super) const SHL_8_CL: &[u8] = &[0x48, 0xd3, 0xe0, 0xf4];
    // shl rax, 1
    // hlt
    pub(super) const SHL_8_ONE: &[u8] = &[0x48, 0xd1, 0xe0, 0xf4];
    // shl rax, 3
    // hlt
    pub(super) const SHL_8_I3: &[u8] = &[0x48, 0xc1, 0xe0, 0x03, 0xf4];
    // shl rax, 17
    // hlt
    pub(super) const SHL_8_I17: &[u8] = &[0x48, 0xc1, 0xe0, 0x11, 0xf4];
    // shl rax, 33
    // hlt
    pub(super) const SHL_8_I33: &[u8] = &[0x48, 0xc1, 0xe0, 0x21, 0xf4];
    // shl qword ptr [rdx], cl
    // hlt
    pub(super) const SHL_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x22, 0xf4];
    // shl qword ptr [rdx], 1
    // hlt
    pub(super) const SHL_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x22, 0xf4];
    // shl eax, cl
    // hlt
    pub(super) const SHL_4_CL: &[u8] = &[0xd3, 0xe0, 0xf4];
    // shl eax, 1
    // hlt
    pub(super) const SHL_4_ONE: &[u8] = &[0xd1, 0xe0, 0xf4];
    // shl eax, 3
    // hlt
    pub(super) const SHL_4_I3: &[u8] = &[0xc1, 0xe0, 0x03, 0xf4];
    // shl eax, 17
    // hlt
    pub(super) const SHL_4_I17: &[u8] = &[0xc1, 0xe0, 0x11, 0xf4];
    // shl eax, 33
    // hlt
    pub(super) const SHL_4_I33: &[u8] = &[0xc1, 0xe0, 0x21, 0xf4];
    // shl dword ptr [rdx], cl
    // hlt
    pub(super) const SHL_4_MEM_CL: &[u8] = &[0xd3, 0x22, 0xf4];
    // shl dword ptr [rdx], 1
    // hlt
    pub(super) const SHL_4_MEM_ONE: &[u8] = &[0xd1, 0x22, 0xf4];
    // shl ax, cl
    // hlt
    pub(super) const SHL_2_CL: &[u8] = &[0x66, 0xd3, 0xe0, 0xf4];
    // shl ax, 1
    // hlt
    pub(super) const SHL_2_ONE: &[u8] = &[0x66, 0xd1, 0xe0, 0xf4];
    // shl ax, 3
    // hlt
    pub(super) const SHL_2_I3: &[u8] = &[0x66, 0xc1, 0xe0, 0x03, 0xf4];
    // shl ax, 17
    // hlt
    pub(super) const SHL_2_I17: &[u8] = &[0x66, 0xc1, 0xe0, 0x11, 0xf4];
    // shl ax, 33
    // hlt
    pub(super) const SHL_2_I33: &[u8] = &[0x66, 0xc1, 0xe0, 0x21, 0xf4];
    // shl word ptr [rdx], cl
    // hlt
    pub(super) const SHL_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x22, 0xf4];
    // shl word ptr [rdx], 1
    // hlt
    pub(super) const SHL_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x22, 0xf4];
    // shl al, cl
    // hlt
    pub(super) const SHL_1_CL: &[u8] = &[0xd2, 0xe0, 0xf4];
    // shl al, 1
    // hlt
    pub(super) const SHL_1_ONE: &[u8] = &[0xd0, 0xe0, 0xf4];
    // shl al, 3
    // hlt
    pub(super) const SHL_1_I3: &[u8] = &[0xc0, 0xe0, 0x03, 0xf4];
    // shl al, 17
    // hlt
    pub(super) const SHL_1_I17: &[u8] = &[0xc0, 0xe0, 0x11, 0xf4];
    // shl al, 33
    // hlt
    pub(super) const SHL_1_I33: &[u8] = &[0xc0, 0xe0, 0x21, 0xf4];
    // shl byte ptr [rdx], cl
    // hlt
    pub(super) const SHL_1_MEM_CL: &[u8] = &[0xd2, 0x22, 0xf4];
    // shl byte ptr [rdx], 1
    // hlt
    pub(super) const SHL_1_MEM_ONE: &[u8] = &[0xd0, 0x22, 0xf4];
    // shr rax, cl
    // hlt
    pub(super) const SHR_8_CL: &[u8] = &[0x48, 0xd3, 0xe8, 0xf4];
    // shr rax, 1
    // hlt
    pub(super) const SHR_8_ONE: &[u8] = &[0x48, 0xd1, 0xe8, 0xf4];
    // shr rax, 3
    // hlt
    pub(super) const SHR_8_I3: &[u8] = &[0x48, 0xc1, 0xe8, 0x03, 0xf4];
    // shr rax, 17
    // hlt
    pub(super) const SHR_8_I17: &[u8] = &[0x48, 0xc1, 0xe8, 0x11, 0xf4];
    // shr rax, 33
    // hlt
    pub(super) const SHR_8_I33: &[u8] = &[0x48, 0xc1, 0xe8, 0x21, 0xf4];
    // shr qword ptr [rdx], cl
    // hlt
    pub(super) const SHR_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x2a, 0xf4];
    // shr qword ptr [rdx], 1
    // hlt
    pub(super) const SHR_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x2a, 0xf4];
    // shr eax, cl
    // hlt
    pub(super) const SHR_4_CL: &[u8] = &[0xd3, 0xe8, 0xf4];
    // shr eax, 1
    // hlt
    pub(super) const SHR_4_ONE: &[u8] = &[0xd1, 0xe8, 0xf4];
    // shr eax, 3
    // hlt
    pub(super) const SHR_4_I3: &[u8] = &[0xc1, 0xe8, 0x03, 0xf4];
    // shr eax, 17
    // hlt
    pub(super) const SHR_4_I17: &[u8] = &[0xc1, 0xe8, 0x11, 0xf4];
    // shr eax, 33
    // hlt
    pub(super) const SHR_4_I33: &[u8] = &[0xc1, 0xe8, 0x21, 0xf4];
    // shr dword ptr [rdx], cl
    // hlt
    pub(super) const SHR_4_MEM_CL: &[u8] = &[0xd3, 0x2a, 0xf4];
    // shr dword ptr [rdx], 1
    // hlt
    pub(super) const SHR_4_MEM_ONE: &[u8] = &[0xd1, 0x2a, 0xf4];
    // shr ax, cl
    // hlt
    pub(super) const SHR_2_CL: &[u8] = &[0x66, 0xd3, 0xe8, 0xf4];
    // shr ax, 1
    // hlt
    pub(super) const SHR_2_ONE: &[u8] = &[0x66, 0xd1, 0xe8, 0xf4];
    // shr ax, 3
    // hlt
    pub(super) const SHR_2_I3: &[u8] = &[0x66, 0xc1, 0xe8, 0x03, 0xf4];
    // shr ax, 17
    // hlt
    pub(super) const SHR_2_I17: &[u8] = &[0x66, 0xc1, 0xe8, 0x11, 0xf4];
    // shr ax, 33
    // hlt
    pub(super) const SHR_2_I33: &[u8] = &[0x66, 0xc1, 0xe8, 0x21, 0xf4];
    // shr word ptr [rdx], cl
    // hlt
    pub(super) const SHR_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x2a, 0xf4];
    // shr word ptr [rdx], 1
    // hlt
    pub(super) const SHR_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x2a, 0xf4];
    // shr al, cl
    // hlt
    pub(super) const SHR_1_CL: &[u8] = &[0xd2, 0xe8, 0xf4];
    // shr al, 1
    // hlt
    pub(super) const SHR_1_ONE: &[u8] = &[0xd0, 0xe8, 0xf4];
    // shr al, 3
    // hlt
    pub(super) const SHR_1_I3: &[u8] = &[0xc0, 0xe8, 0x03, 0xf4];
    // shr al, 17
    // hlt
    pub(super) const SHR_1_I17: &[u8] = &[0xc0, 0xe8, 0x11, 0xf4];
    // shr al, 33
    // hlt
    pub(super) const SHR_1_I33: &[u8] = &[0xc0, 0xe8, 0x21, 0xf4];
    // shr byte ptr [rdx], cl
    // hlt
    pub(super) const SHR_1_MEM_CL: &[u8] = &[0xd2, 0x2a, 0xf4];
    // shr byte ptr [rdx], 1
    // hlt
    pub(super) const SHR_1_MEM_ONE: &[u8] = &[0xd0, 0x2a, 0xf4];
    // sar rax, cl
    // hlt
    pub(super) const SAR_8_CL: &[u8] = &[0x48, 0xd3, 0xf8, 0xf4];
    // sar rax, 1
    // hlt
    pub(super) const SAR_8_ONE: &[u8] = &[0x48, 0xd1, 0xf8, 0xf4];
    // sar rax, 3
    // hlt
    pub(super) const SAR_8_I3: &[u8] = &[0x48, 0xc1, 0xf8, 0x03, 0xf4];
    // sar rax, 17
    // hlt
    pub(super) const SAR_8_I17: &[u8] = &[0x48, 0xc1, 0xf8, 0x11, 0xf4];
    // sar rax, 33
    // hlt
    pub(super) const SAR_8_I33: &[u8] = &[0x48, 0xc1, 0xf8, 0x21, 0xf4];
    // sar qword ptr [rdx], cl
    // hlt
    pub(super) const SAR_8_MEM_CL: &[u8] = &[0x48, 0xd3, 0x3a, 0xf4];
    // sar qword ptr [rdx], 1
    // hlt
    pub(super) const SAR_8_MEM_ONE: &[u8] = &[0x48, 0xd1, 0x3a, 0xf4];
    // sar eax, cl
    // hlt
    pub(super) const SAR_4_CL: &[u8] = &[0xd3, 0xf8, 0xf4];
    // sar eax, 1
    // hlt
    pub(super) const SAR_4_ONE: &[u8] = &[0xd1, 0xf8, 0xf4];
    // sar eax, 3
    // hlt
    pub(super) const SAR_4_I3: &[u8] = &[0xc1, 0xf8, 0x03, 0xf4];
    // sar eax, 17
    // hlt
    pub(super) const SAR_4_I17: &[u8] = &[0xc1, 0xf8, 0x11, 0xf4];
    // sar eax, 33
    // hlt
    pub(super) const SAR_4_I33: &[u8] = &[0xc1, 0xf8, 0x21, 0xf4];
    // sar dword ptr [rdx], cl
    // hlt
    pub(super) const SAR_4_MEM_CL: &[u8] = &[0xd3, 0x3a, 0xf4];
    // sar dword ptr [rdx], 1
    // hlt
    pub(super) const SAR_4_MEM_ONE: &[u8] = &[0xd1, 0x3a, 0xf4];
    // sar ax, cl
    // hlt
    pub(super) const SAR_2_CL: &[u8] = &[0x66, 0xd3, 0xf8, 0xf4];
    // sar ax, 1
    // hlt
    pub(super) const SAR_2_ONE: &[u8] = &[0x66, 0xd1, 0xf8, 0xf4];
    // sar ax, 3
    // hlt
    pub(super) const SAR_2_I3: &[u8] = &[0x66, 0xc1, 0xf8, 0x03, 0xf4];
    // sar ax, 17
    // hlt
    pub(super) const SAR_2_I17: &[u8] = &[0x66, 0xc1, 0xf8, 0x11, 0xf4];
    // sar ax, 33
    // hlt
    pub(super) const SAR_2_I33: &[u8] = &[0x66, 0xc1, 0xf8, 0x21, 0xf4];
    // sar word ptr [rdx], cl
    // hlt
    pub(super) const SAR_2_MEM_CL: &[u8] = &[0x66, 0xd3, 0x3a, 0xf4];
    // sar word ptr [rdx], 1
    // hlt
    pub(super) const SAR_2_MEM_ONE: &[u8] = &[0x66, 0xd1, 0x3a, 0xf4];
    // sar al, cl
    // hlt
    pub(super) const SAR_1_CL: &[u8] = &[0xd2, 0xf8, 0xf4];
    // sar al, 1
    // hlt
    pub(super) const SAR_1_ONE: &[u8] = &[0xd0, 0xf8, 0xf4];
    // sar al, 3
    // hlt
    pub(super) const SAR_1_I3: &[u8] = &[0xc0, 0xf8, 0x03, 0xf4];
    // sar al, 17
    // hlt
    pub(super) const SAR_1_I17: &[u8] = &[0xc0, 0xf8, 0x11, 0xf4];
    // sar al, 33
    // hlt
    pub(super) const SAR_1_I33: &[u8] = &[0xc0, 0xf8, 0x21, 0xf4];
    // sar byte ptr [rdx], cl
    // hlt
    pub(super) const SAR_1_MEM_CL: &[u8] = &[0xd2, 0x3a, 0xf4];
    // sar byte ptr [rdx], 1
    // hlt
    pub(super) const SAR_1_MEM_ONE: &[u8] = &[0xd0, 0x3a, 0xf4];
    // movsb
    // hlt
    pub(super) const MOVS_1: &[u8] = &[0xa4, 0xf4];
    // rep movsb
    // hlt
    pub(super) const REP_MOVS_1: &[u8] = &[0xf3, 0xa4, 0xf4];
    // stosb
    // hlt
    pub(super) const STOS_1: &[u8] = &[0xaa, 0xf4];
    // rep stosb
    // hlt
    pub(super) const REP_STOS_1: &[u8] = &[0xf3, 0xaa, 0xf4];
    // lodsb
    // hlt
    pub(super) const LODS_1: &[u8] = &[0xac, 0xf4];
    // rep lodsb
    // hlt
    pub(super) const REP_LODS_1: &[u8] = &[0xf3, 0xac, 0xf4];
    // scasb
    // hlt
    pub(super) const SCAS_1: &[u8] = &[0xae, 0xf4];
    // repe scasb
    // hlt
    pub(super) const REPE_SCAS_1: &[u8] = &[0xf3, 0xae, 0xf4];
    // repne scasb
    // hlt
    pub(super) const REPNE_SCAS_1: &[u8] = &[0xf2, 0xae, 0xf4];
    // cmpsb
    // hlt
    pub(super) const CMPS_1: &[u8] = &[0xa6, 0xf4];
    // repe cmpsb
    // hlt
    pub(super) const REPE_CMPS_1: &[u8] = &[0xf3, 0xa6, 0xf4];
    // repne cmpsb
    // hlt
    pub(super) const REPNE_CMPS_1: &[u8] = &[0xf2, 0xa6, 0xf4];
    // movsw
    // hlt
    pub(super) const MOVS_2: &[u8] = &[0x66, 0xa5, 0xf4];
    // rep movsw
    // hlt
    pub(super) const REP_MOVS_2: &[u8] = &[0xf3, 0x66, 0xa5, 0xf4];
    // stosw
    // hlt
    pub(super) const STOS_2: &[u8] = &[0x66, 0xab, 0xf4];
    // rep stosw
    // hlt
    pub(super) const REP_STOS_2: &[u8] = &[0xf3, 0x66, 0xab, 0xf4];
    // lodsw
    // hlt
    pub(super) const LODS_2: &[u8] = &[0x66, 0xad, 0xf4];
    // rep lodsw
    // hlt
    pub(super) const REP_LODS_2: &[u8] = &[0xf3, 0x66, 0xad, 0xf4];
    // scasw
    // hlt
    pub(super) const SCAS_2: &[u8] = &[0x66, 0xaf, 0xf4];
    // repe scasw
    // hlt
    pub(super) const REPE_SCAS_2: &[u8] = &[0xf3, 0x66, 0xaf, 0xf4];
    // repne scasw
    // hlt
    pub(super) const REPNE_SCAS_2: &[u8] = &[0xf2, 0x66, 0xaf, 0xf4];
    // cmpsw
    // hlt
    pub(super) const CMPS_2: &[u8] = &[0x66, 0xa7, 0xf4];
    // repe cmpsw
    // hlt
    pub(super) const REPE_CMPS_2: &[u8] = &[0xf3, 0x66, 0xa7, 0xf4];
    // repne cmpsw
    // hlt
    pub(super) const REPNE_CMPS_2: &[u8] = &[0xf2, 0x66, 0xa7, 0xf4];
    // movsd
    // hlt
    pub(super) const MOVS_4: &[u8] = &[0xa5, 0xf4];
    // rep movsd
    // hlt
    pub(super) const REP_MOVS_4: &[u8] = &[0xf3, 0xa5, 0xf4];
    // stosd
    // hlt
    pub(super) const STOS_4: &[u8] = &[0xab, 0xf4];
    // rep stosd
    // hlt
    pub(super) const REP_STOS_4: &[u8] = &[0xf3, 0xab, 0xf4];
    // lodsd
    // hlt
    pub(super) const LODS_4: &[u8] = &[0xad, 0xf4];
    // rep lodsd
    // hlt
    pub(super) const REP_LODS_4: &[u8] = &[0xf3, 0xad, 0xf4];
    // scasd
    // hlt
    pub(super) const SCAS_4: &[u8] = &[0xaf, 0xf4];
    // repe scasd
    // hlt
    pub(super) const REPE_SCAS_4: &[u8] = &[0xf3, 0xaf, 0xf4];
    // repne scasd
    // hlt
    pub(super) const REPNE_SCAS_4: &[u8] = &[0xf2, 0xaf, 0xf4];
    // cmpsd
    // hlt
    pub(super) const CMPS_4: &[u8] = &[0xa7, 0xf4];
    // repe cmpsd
    // hlt
    pub(super) const REPE_CMPS_4: &[u8] = &[0xf3, 0xa7, 0xf4];
    // repne cmpsd
    // hlt
    pub(super) const REPNE_CMPS_4: &[u8] = &[0xf2, 0xa7, 0xf4];
    // movsq
    // hlt
    pub(super) const MOVS_8: &[u8] = &[0x48, 0xa5, 0xf4];
    // rep movsq
    // hlt
    pub(super) const REP_MOVS_8: &[u8] = &[0xf3, 0x48, 0xa5, 0xf4];
    // stosq
    // hlt
    pub(super) const STOS_8: &[u8] = &[0x48, 0xab, 0xf4];
    // rep stosq
    // hlt
    pub(super) const REP_STOS_8: &[u8] = &[0xf3, 0x48, 0xab, 0xf4];
    // lodsq
    // hlt
    pub(super) const LODS_8: &[u8] = &[0x48, 0xad, 0xf4];
    // rep lodsq
    // hlt
    pub(super) const REP_LODS_8: &[u8] = &[0xf3, 0x48, 0xad, 0xf4];
    // scasq
    // hlt
    pub(super) const SCAS_8: &[u8] = &[0x48, 0xaf, 0xf4];
    // repe scasq
    // hlt
    pub(super) const REPE_SCAS_8: &[u8] = &[0xf3, 0x48, 0xaf, 0xf4];
    // repne scasq
    // hlt
    pub(super) const REPNE_SCAS_8: &[u8] = &[0xf2, 0x48, 0xaf, 0xf4];
    // cmpsq
    // hlt
    pub(super) const CMPS_8: &[u8] = &[0x48, 0xa7, 0xf4];
    // repe cmpsq
    // hlt
    pub(super) const REPE_CMPS_8: &[u8] = &[0xf3, 0x48, 0xa7, 0xf4];
    // repne cmpsq
    // hlt
    pub(super) const REPNE_CMPS_8: &[u8] = &[0xf2, 0x48, 0xa7, 0xf4];
    // .byte 0x67
    // movsb
    // hlt
    pub(super) const A32_MOVS_1: &[u8] = &[0x67, 0xa4, 0xf4];
    // .byte 0x67
    // rep movsb
    // hlt
    pub(super) const A32_REP_MOVS_1: &[u8] = &[0x67, 0xf3, 0xa4, 0xf4];
    // .byte 0x67
    // rep stosb
    // hlt
    pub(super) const A32_REP_STOS_1: &[u8] = &[0x67, 0xf3, 0xaa, 0xf4];
    // .byte 0x67
    // rep lodsb
    // hlt
    pub(super) const A32_REP_LODS_1: &[u8] = &[0x67, 0xf3, 0xac, 0xf4];
    // .byte 0x67
    // repe scasb
    // hlt
    pub(super) const A32_REPE_SCAS_1: &[u8] = &[0x67, 0xf3, 0xae, 0xf4];
    // .byte 0x67
    // repne scasb
    // hlt
    pub(super) const A32_REPNE_SCAS_1: &[u8] = &[0x67, 0xf2, 0xae, 0xf4];
    // .byte 0x67
    // repe cmpsb
    // hlt
    pub(super) const A32_REPE_CMPS_1: &[u8] = &[0x67, 0xf3, 0xa6, 0xf4];
    // .byte 0x67
    // movsd
    // hlt
    pub(super) const A32_MOVS_4: &[u8] = &[0x67, 0xa5, 0xf4];
    // .byte 0x67
    // rep movsd
    // hlt
    pub(super) const A32_REP_MOVS_4: &[u8] = &[0x67, 0xf3, 0xa5, 0xf4];
    // .byte 0x67
    // rep stosd
    // hlt
    pub(super) const A32_REP_STOS_4: &[u8] = &[0x67, 0xf3, 0xab, 0xf4];
    // .byte 0x67
    // rep lodsd
    // hlt
    pub(super) const A32_REP_LODS_4: &[u8] = &[0x67, 0xf3, 0xad, 0xf4];
    // .byte 0x67
    // repe scasd
    // hlt
    pub(super) const A32_REPE_SCAS_4: &[u8] = &[0x67, 0xf3, 0xaf, 0xf4];
    // .byte 0x67
    // repne scasd
    // hlt
    pub(super) const A32_REPNE_SCAS_4: &[u8] = &[0x67, 0xf2, 0xaf, 0xf4];
    // .byte 0x67
    // repe cmpsd
    // hlt
    pub(super) const A32_REPE_CMPS_4: &[u8] = &[0x67, 0xf3, 0xa7, 0xf4];
    // .byte 0x64
    // movsb
    // hlt
    pub(super) const FS_MOVS_1: &[u8] = &[0x64, 0xa4, 0xf4];
    // .byte 0x64
    // rep movsb
    // hlt
    pub(super) const FS_REP_MOVS_1: &[u8] = &[0x64, 0xf3, 0xa4, 0xf4];
    // .byte 0x65
    // rep lodsb
    // hlt
    pub(super) const GS_REP_LODS_1: &[u8] = &[0x65, 0xf3, 0xac, 0xf4];
    // .byte 0x64
    // repe cmpsb
    // hlt
    pub(super) const FS_REPE_CMPS_1: &[u8] = &[0x64, 0xf3, 0xa6, 0xf4];
    // .byte 0x64
    // rep stosb
    // hlt
    pub(super) const FS_REP_STOS_1: &[u8] = &[0x64, 0xf3, 0xaa, 0xf4];
    // .byte 0x64
    // repne scasb
    // hlt
    pub(super) const FS_REPNE_SCAS_1: &[u8] = &[0x64, 0xf2, 0xae, 0xf4];
    // .byte 0x64
    // movsq
    // hlt
    pub(super) const FS_MOVS_8: &[u8] = &[0x64, 0x48, 0xa5, 0xf4];
    // .byte 0x64
    // rep movsq
    // hlt
    pub(super) const FS_REP_MOVS_8: &[u8] = &[0x64, 0xf3, 0x48, 0xa5, 0xf4];
    // .byte 0x65
    // rep lodsq
    // hlt
    pub(super) const GS_REP_LODS_8: &[u8] = &[0x65, 0xf3, 0x48, 0xad, 0xf4];
    // .byte 0x64
    // repe cmpsq
    // hlt
    pub(super) const FS_REPE_CMPS_8: &[u8] = &[0x64, 0xf3, 0x48, 0xa7, 0xf4];
    // .byte 0x64
    // rep stosq
    // hlt
    pub(super) const FS_REP_STOS_8: &[u8] = &[0x64, 0xf3, 0x48, 0xab, 0xf4];
    // .byte 0x64
    // repne scasq
    // hlt
    pub(super) const FS_REPNE_SCAS_8: &[u8] = &[0x64, 0xf2, 0x48, 0xaf, 0xf4];
    // idiv rcx
    // hlt
    pub(super) const IDIV_RCX: &[u8] = &[0x48, 0xf7, 0xf9, 0xf4];
    // idiv ecx
    // hlt
    pub(super) const IDIV_ECX: &[u8] = &[0xf7, 0xf9, 0xf4];
    // idiv cx
    // hlt
    pub(super) const IDIV_CX: &[u8] = &[0x66, 0xf7, 0xf9, 0xf4];
    // idiv cl
    // hlt
    pub(super) const IDIV_CL: &[u8] = &[0xf6, 0xf9, 0xf4];
    // div rcx
    // hlt
    pub(super) const DIV_RCX: &[u8] = &[0x48, 0xf7, 0xf1, 0xf4];
    // div ecx
    // hlt
    pub(super) const DIV_ECX: &[u8] = &[0xf7, 0xf1, 0xf4];
    // div cx
    // hlt
    pub(super) const DIV_CX: &[u8] = &[0x66, 0xf7, 0xf1, 0xf4];
    // div cl
    // hlt
    pub(super) const DIV_CL: &[u8] = &[0xf6, 0xf1, 0xf4];
    // mul rcx
    // hlt
    pub(super) const MUL_RCX: &[u8] = &[0x48, 0xf7, 0xe1, 0xf4];
    // mul ecx
    // hlt
    pub(super) const MUL_ECX: &[u8] = &[0xf7, 0xe1, 0xf4];
    // mul cx
    // hlt
    pub(super) const MUL_CX: &[u8] = &[0x66, 0xf7, 0xe1, 0xf4];
    // mul cl
    // hlt
    pub(super) const MUL_CL: &[u8] = &[0xf6, 0xe1, 0xf4];
    // imul rcx
    // hlt
    pub(super) const IMUL1_RCX: &[u8] = &[0x48, 0xf7, 0xe9, 0xf4];
    // imul ecx
    // hlt
    pub(super) const IMUL1_ECX: &[u8] = &[0xf7, 0xe9, 0xf4];
    // imul cx
    // hlt
    pub(super) const IMUL1_CX: &[u8] = &[0x66, 0xf7, 0xe9, 0xf4];
    // imul cl
    // hlt
    pub(super) const IMUL1_CL: &[u8] = &[0xf6, 0xe9, 0xf4];
    // shld ax, bx, cl
    // hlt
    pub(super) const SHLD_AX_BX_CL: &[u8] = &[0x66, 0x0f, 0xa5, 0xd8, 0xf4];
    // shrd ax, bx, cl
    // hlt
    pub(super) const SHRD_AX_BX_CL: &[u8] = &[0x66, 0x0f, 0xad, 0xd8, 0xf4];
    // shld ax, bx, 20
    // hlt
    pub(super) const SHLD_AX_BX_20: &[u8] = &[0x66, 0x0f, 0xa4, 0xd8, 0x14, 0xf4];
    // shrd ax, bx, 20
    // hlt
    pub(super) const SHRD_AX_BX_20: &[u8] = &[0x66, 0x0f, 0xac, 0xd8, 0x14, 0xf4];
    // shld eax, ebx, cl
    // hlt
    pub(super) const SHLD_EAX_EBX_CL: &[u8] = &[0x0f, 0xa5, 0xd8, 0xf4];
    // shrd eax, ebx, cl
    // hlt
    pub(super) const SHRD_EAX_EBX_CL: &[u8] = &[0x0f, 0xad, 0xd8, 0xf4];
    // shld rax, rbx, cl
    // hlt
    pub(super) const SHLD_RAX_RBX_CL: &[u8] = &[0x48, 0x0f, 0xa5, 0xd8, 0xf4];
    // shrd rax, rbx, cl
    // hlt
    pub(super) const SHRD_RAX_RBX_CL: &[u8] = &[0x48, 0x0f, 0xad, 0xd8, 0xf4];
    // .byte 0x44
    // mov eax, ds
    // hlt
    pub(super) const MOV_EAX_DS_REX_R: &[u8] = &[0x44, 0x8c, 0xd8, 0xf4];
    // mov eax, ds
    // hlt
    pub(super) const MOV_EAX_DS: &[u8] = &[0x8c, 0xd8, 0xf4];
    // mov eax, es
    // hlt
    pub(super) const MOV_EAX_ES: &[u8] = &[0x8c, 0xc0, 0xf4];
    // mov eax, cs
    // hlt
    pub(super) const MOV_EAX_CS: &[u8] = &[0x8c, 0xc8, 0xf4];
    // mov eax, ss
    // hlt
    pub(super) const MOV_EAX_SS: &[u8] = &[0x8c, 0xd0, 0xf4];
    // mov eax, fs
    // hlt
    pub(super) const MOV_EAX_FS: &[u8] = &[0x8c, 0xe0, 0xf4];
    // mov eax, gs
    // hlt
    pub(super) const MOV_EAX_GS: &[u8] = &[0x8c, 0xe8, 0xf4];
    // mov rax, ds
    // hlt
    pub(super) const MOV_RAX_DS: &[u8] = &[0x48, 0x8c, 0xd8, 0xf4];
    // .byte 0x66
    // mov eax, ds
    // hlt
    pub(super) const MOV_AX_DS: &[u8] = &[0x66, 0x8c, 0xd8, 0xf4];
    // mov word ptr [rbx], ds
    // hlt
    pub(super) const MOV_MEM_DS: &[u8] = &[0x8c, 0x1b, 0xf4];
    // .byte 0x8c, 0xf0
    // hlt
    pub(super) const MOV_EAX_SREG6: &[u8] = &[0x8c, 0xf0, 0xf4];
    // .byte 0x8c, 0xf8
    // hlt
    pub(super) const MOV_EAX_SREG7: &[u8] = &[0x8c, 0xf8, 0xf4];
    // .byte 0x44, 0x8c, 0xf0
    // hlt
    pub(super) const MOV_EAX_SREG6_REX_R: &[u8] = &[0x44, 0x8c, 0xf0, 0xf4];
    // mov ds, ax
    // hlt
    pub(super) const MOV_DS_AX: &[u8] = &[0x66, 0x8e, 0xd8, 0xf4];
    // mov es, ax
    // hlt
    pub(super) const MOV_ES_AX: &[u8] = &[0x66, 0x8e, 0xc0, 0xf4];
    // mov ss, ax
    // hlt
    pub(super) const MOV_SS_AX: &[u8] = &[0x66, 0x8e, 0xd0, 0xf4];
    // mov fs, ax
    // hlt
    pub(super) const MOV_FS_AX: &[u8] = &[0x66, 0x8e, 0xe0, 0xf4];
    // mov gs, ax
    // hlt
    pub(super) const MOV_GS_AX: &[u8] = &[0x66, 0x8e, 0xe8, 0xf4];
    // .byte 0x8e, 0xc8
    // hlt
    pub(super) const MOV_CS_AX: &[u8] = &[0x8e, 0xc8, 0xf4];
    // .byte 0x8e, 0xf0
    // hlt
    pub(super) const MOV_SREG6_AX: &[u8] = &[0x8e, 0xf0, 0xf4];
    // .byte 0x66, 0x44, 0x8e, 0xd8
    // hlt
    pub(super) const MOV_DS_AX_REX_R: &[u8] = &[0x66, 0x44, 0x8e, 0xd8, 0xf4];
    // xlatb
    // hlt
    pub(super) const XLAT: &[u8] = &[0xd7, 0xf4];
    // .byte 0x67
    // xlatb
    // hlt
    pub(super) const XLAT_ADDR32: &[u8] = &[0x67, 0xd7, 0xf4];
    // .byte 0x64
    // xlatb
    // hlt
    pub(super) const XLAT_FS: &[u8] = &[0x64, 0xd7, 0xf4];
    // enter 16, 0
    // hlt
    pub(super) const ENTER_16_0: &[u8] = &[0xc8, 0x10, 0x00, 0x00, 0xf4];
    // enter 0, 0
    // hlt
    pub(super) const ENTER_0_0: &[u8] = &[0xc8, 0x00, 0x00, 0x00, 0xf4];
    // enter 0, 1
    // hlt
    pub(super) const ENTER_0_1: &[u8] = &[0xc8, 0x00, 0x00, 0x01, 0xf4];
    // enter 16, 1
    // hlt
    pub(super) const ENTER_16_1: &[u8] = &[0xc8, 0x10, 0x00, 0x01, 0xf4];
    // enter 8, 2
    // hlt
    pub(super) const ENTER_8_2: &[u8] = &[0xc8, 0x08, 0x00, 0x02, 0xf4];
    // enter 0, 3
    // hlt
    pub(super) const ENTER_0_3: &[u8] = &[0xc8, 0x00, 0x00, 0x03, 0xf4];
    // enter 0, 33
    // hlt
    pub(super) const ENTER_0_33: &[u8] = &[0xc8, 0x00, 0x00, 0x21, 0xf4];
    // enter 0x10, 32
    // hlt
    pub(super) const ENTER_0_32: &[u8] = &[0xc8, 0x10, 0x00, 0x20, 0xf4];
    // .byte 0x66
    // enter 16, 0
    // hlt
    pub(super) const ENTER_16_0_P66: &[u8] = &[0x66, 0xc8, 0x10, 0x00, 0x00, 0xf4];
    // .byte 0x66
    // enter 16, 2
    // hlt
    pub(super) const ENTER_16_2_P66: &[u8] = &[0x66, 0xc8, 0x10, 0x00, 0x02, 0xf4];
    // leave
    // hlt
    pub(super) const LEAVE: &[u8] = &[0xc9, 0xf4];
    // .byte 0x66
    // leave
    // hlt
    pub(super) const LEAVE_P66: &[u8] = &[0x66, 0xc9, 0xf4];
    // out dx, al
    // hlt
    pub(super) const OUT_DX_AL: &[u8] = &[0xee, 0xf4];
    // out dx, ax
    // hlt
    pub(super) const OUT_DX_AX: &[u8] = &[0x66, 0xef, 0xf4];
    // out dx, eax
    // hlt
    pub(super) const OUT_DX_EAX: &[u8] = &[0xef, 0xf4];
    // in al, dx
    // hlt
    pub(super) const IN_AL_DX: &[u8] = &[0xec, 0xf4];
    // in ax, dx
    // hlt
    pub(super) const IN_AX_DX: &[u8] = &[0x66, 0xed, 0xf4];
    // in eax, dx
    // hlt
    pub(super) const IN_EAX_DX: &[u8] = &[0xed, 0xf4];
    // in al, 0x80
    // hlt
    pub(super) const IN_AL_IMM: &[u8] = &[0xe4, 0x80, 0xf4];
    // out 0x80, al
    // hlt
    pub(super) const OUT_IMM_AL: &[u8] = &[0xe6, 0x80, 0xf4];
    // in eax, 0x80
    // hlt
    pub(super) const IN_EAX_IMM: &[u8] = &[0xe5, 0x80, 0xf4];
    // out 0x80, eax
    // hlt
    pub(super) const OUT_IMM_EAX: &[u8] = &[0xe7, 0x80, 0xf4];
    // push rax
    // hlt
    pub(super) const PUSH_RAX: &[u8] = &[0x50, 0xf4];
    // pop rax
    // hlt
    pub(super) const POP_RAX: &[u8] = &[0x58, 0xf4];
    // call target
    // hlt
    // target: hlt
    pub(super) const CALL_REL: &[u8] = &[0xe8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xf4];
    // ret
    // hlt
    pub(super) const RET: &[u8] = &[0xc3, 0xf4];
    // ret 16
    // hlt
    pub(super) const RET_IMM: &[u8] = &[0xc2, 0x10, 0x00, 0xf4];
    // pushfq
    // hlt
    pub(super) const PUSHFQ: &[u8] = &[0x9c, 0xf4];
    // popfq
    // hlt
    pub(super) const POPFQ: &[u8] = &[0x9d, 0xf4];
    // iretq
    // hlt
    pub(super) const IRETQ: &[u8] = &[0x48, 0xcf, 0xf4];
    // retfq
    // hlt
    pub(super) const RETFQ: &[u8] = &[0x48, 0xcb, 0xf4];
    // rex64 retf 16
    // hlt
    pub(super) const RETFQ_IMM: &[u8] = &[0x48, 0xca, 0x10, 0x00, 0xf4];
    // iretd
    // hlt
    pub(super) const IRETD: &[u8] = &[0xcf, 0xf4];
    // retf
    // hlt
    pub(super) const RETF_32: &[u8] = &[0xcb, 0xf4];
    // mov rax, [rbx]
    // hlt
    pub(super) const MOV_RAX_MEM_RBX: &[u8] = &[0x48, 0x8b, 0x03, 0xf4];
    // out dx, al
    // xor eax, eax
    // in al, dx
    // hlt
    pub(super) const OUT_IN_SCRATCH: &[u8] = &[0xee, 0x31, 0xc0, 0xec, 0xf4];
    // .byte 0xc1, 0xf0, 0x04
    // hlt
    pub(super) const SAL_ALIAS_EAX_4: &[u8] = &[0xc1, 0xf0, 0x04, 0xf4];
    // inc rax
    // hlt
    pub(super) const INC_8: &[u8] = &[0x48, 0xff, 0xc0, 0xf4];
    // inc qword ptr [rdx]
    // hlt
    pub(super) const INC_MEM_8: &[u8] = &[0x48, 0xff, 0x02, 0xf4];
    // inc eax
    // hlt
    pub(super) const INC_4: &[u8] = &[0xff, 0xc0, 0xf4];
    // inc dword ptr [rdx]
    // hlt
    pub(super) const INC_MEM_4: &[u8] = &[0xff, 0x02, 0xf4];
    // inc ax
    // hlt
    pub(super) const INC_2: &[u8] = &[0x66, 0xff, 0xc0, 0xf4];
    // inc word ptr [rdx]
    // hlt
    pub(super) const INC_MEM_2: &[u8] = &[0x66, 0xff, 0x02, 0xf4];
    // inc al
    // hlt
    pub(super) const INC_1: &[u8] = &[0xfe, 0xc0, 0xf4];
    // inc byte ptr [rdx]
    // hlt
    pub(super) const INC_MEM_1: &[u8] = &[0xfe, 0x02, 0xf4];
    // dec rax
    // hlt
    pub(super) const DEC_8: &[u8] = &[0x48, 0xff, 0xc8, 0xf4];
    // dec qword ptr [rdx]
    // hlt
    pub(super) const DEC_MEM_8: &[u8] = &[0x48, 0xff, 0x0a, 0xf4];
    // dec eax
    // hlt
    pub(super) const DEC_4: &[u8] = &[0xff, 0xc8, 0xf4];
    // dec dword ptr [rdx]
    // hlt
    pub(super) const DEC_MEM_4: &[u8] = &[0xff, 0x0a, 0xf4];
    // dec ax
    // hlt
    pub(super) const DEC_2: &[u8] = &[0x66, 0xff, 0xc8, 0xf4];
    // dec word ptr [rdx]
    // hlt
    pub(super) const DEC_MEM_2: &[u8] = &[0x66, 0xff, 0x0a, 0xf4];
    // dec al
    // hlt
    pub(super) const DEC_1: &[u8] = &[0xfe, 0xc8, 0xf4];
    // dec byte ptr [rdx]
    // hlt
    pub(super) const DEC_MEM_1: &[u8] = &[0xfe, 0x0a, 0xf4];
    // neg rax
    // hlt
    pub(super) const NEG_8: &[u8] = &[0x48, 0xf7, 0xd8, 0xf4];
    // neg qword ptr [rdx]
    // hlt
    pub(super) const NEG_MEM_8: &[u8] = &[0x48, 0xf7, 0x1a, 0xf4];
    // neg eax
    // hlt
    pub(super) const NEG_4: &[u8] = &[0xf7, 0xd8, 0xf4];
    // neg dword ptr [rdx]
    // hlt
    pub(super) const NEG_MEM_4: &[u8] = &[0xf7, 0x1a, 0xf4];
    // neg ax
    // hlt
    pub(super) const NEG_2: &[u8] = &[0x66, 0xf7, 0xd8, 0xf4];
    // neg word ptr [rdx]
    // hlt
    pub(super) const NEG_MEM_2: &[u8] = &[0x66, 0xf7, 0x1a, 0xf4];
    // neg al
    // hlt
    pub(super) const NEG_1: &[u8] = &[0xf6, 0xd8, 0xf4];
    // neg byte ptr [rdx]
    // hlt
    pub(super) const NEG_MEM_1: &[u8] = &[0xf6, 0x1a, 0xf4];
    // not rax
    // hlt
    pub(super) const NOT_8: &[u8] = &[0x48, 0xf7, 0xd0, 0xf4];
    // not qword ptr [rdx]
    // hlt
    pub(super) const NOT_MEM_8: &[u8] = &[0x48, 0xf7, 0x12, 0xf4];
    // not eax
    // hlt
    pub(super) const NOT_4: &[u8] = &[0xf7, 0xd0, 0xf4];
    // not dword ptr [rdx]
    // hlt
    pub(super) const NOT_MEM_4: &[u8] = &[0xf7, 0x12, 0xf4];
    // not ax
    // hlt
    pub(super) const NOT_2: &[u8] = &[0x66, 0xf7, 0xd0, 0xf4];
    // not word ptr [rdx]
    // hlt
    pub(super) const NOT_MEM_2: &[u8] = &[0x66, 0xf7, 0x12, 0xf4];
    // not al
    // hlt
    pub(super) const NOT_1: &[u8] = &[0xf6, 0xd0, 0xf4];
    // not byte ptr [rdx]
    // hlt
    pub(super) const NOT_MEM_1: &[u8] = &[0xf6, 0x12, 0xf4];
    // mov ah, bh
    // hlt
    pub(super) const MOV_AH_BH: &[u8] = &[0x88, 0xfc, 0xf4];
    // mov al, bh
    // hlt
    pub(super) const MOV_AL_BH: &[u8] = &[0x88, 0xf8, 0xf4];
    // mov bh, al
    // hlt
    pub(super) const MOV_BH_AL: &[u8] = &[0x88, 0xc7, 0xf4];
    // mov sil, 0x5a
    // hlt
    pub(super) const MOV_SIL_IMM: &[u8] = &[0x40, 0xb6, 0x5a, 0xf4];
    // mov dh, 0x5a
    // hlt
    pub(super) const MOV_DH_IMM: &[u8] = &[0xb6, 0x5a, 0xf4];
    // mov spl, 0x11
    // hlt
    pub(super) const MOV_SPL_IMM: &[u8] = &[0x40, 0xb4, 0x11, 0xf4];
    // mov r8b, 0x7f
    // hlt
    pub(super) const MOV_R8B_IMM: &[u8] = &[0x41, 0xb0, 0x7f, 0xf4];
    // mov r15b, bl
    // hlt
    pub(super) const MOV_R15B_BL: &[u8] = &[0x41, 0x88, 0xdf, 0xf4];
    // mov bl, ah
    // hlt
    pub(super) const MOV_BL_AH: &[u8] = &[0x88, 0xe3, 0xf4];
    // add ah, bh
    // hlt
    pub(super) const ADD_AH_BH: &[u8] = &[0x00, 0xfc, 0xf4];
    // inc ah
    // hlt
    pub(super) const INC_AH: &[u8] = &[0xfe, 0xc4, 0xf4];
    // neg bh
    // hlt
    pub(super) const NEG_BH: &[u8] = &[0xf6, 0xdf, 0xf4];
    // not ch
    // hlt
    pub(super) const NOT_CH: &[u8] = &[0xf6, 0xd5, 0xf4];
    // shl ah, 1
    // hlt
    pub(super) const SHL_AH_1: &[u8] = &[0xd0, 0xe4, 0xf4];
    // test ah, bh
    // hlt
    pub(super) const TEST_AH_BH: &[u8] = &[0x84, 0xfc, 0xf4];
    // cmp ah, 5
    // hlt
    pub(super) const CMP_AH_IMM: &[u8] = &[0x80, 0xfc, 0x05, 0xf4];
    // xchg ah, al
    // hlt
    pub(super) const XCHG_AH_AL: &[u8] = &[0x86, 0xe0, 0xf4];
    // xchg ah, bh
    // hlt
    pub(super) const XCHG_AH_BH: &[u8] = &[0x86, 0xe7, 0xf4];
    // movzx eax, bh
    // hlt
    pub(super) const MOVZX_EAX_BH: &[u8] = &[0x0f, 0xb6, 0xc7, 0xf4];
    // movsx eax, dh
    // hlt
    pub(super) const MOVSX_EAX_DH: &[u8] = &[0x0f, 0xbe, 0xc6, 0xf4];
    // movzx eax, dil
    // hlt
    pub(super) const MOVZX_EAX_DIL: &[u8] = &[0x40, 0x0f, 0xb6, 0xc7, 0xf4];
    // movsx rax, sil
    // hlt
    pub(super) const MOVSX_RAX_SIL: &[u8] = &[0x48, 0x0f, 0xbe, 0xc6, 0xf4];
    // mov eax, ebx
    // hlt
    pub(super) const MOV_EAX_EBX: &[u8] = &[0x89, 0xd8, 0xf4];
    // mov ax, bx
    // hlt
    pub(super) const MOV_AX_BX: &[u8] = &[0x66, 0x89, 0xd8, 0xf4];
    // mov al, bl
    // hlt
    pub(super) const MOV_AL_BL: &[u8] = &[0x88, 0xd8, 0xf4];
    // mov rax, rbx
    // hlt
    pub(super) const MOV_RAX_RBX: &[u8] = &[0x48, 0x89, 0xd8, 0xf4];
    // mov r9d, r10d
    // hlt
    pub(super) const MOV_R9D_R10D: &[u8] = &[0x45, 0x89, 0xd1, 0xf4];
    // mov r9w, r10w
    // hlt
    pub(super) const MOV_R9W_R10W: &[u8] = &[0x66, 0x45, 0x89, 0xd1, 0xf4];
    // mov eax, 0x12345678
    // hlt
    pub(super) const MOV_EAX_IMM: &[u8] = &[0xb8, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // movabs rax, 0x1122334455667788
    // hlt
    pub(super) const MOV_RAX_IMM64: &[u8] = &[0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0xf4];
    // mov rax, -2
    // hlt
    pub(super) const MOV_RAX_IMM_NEG: &[u8] = &[0x48, 0xc7, 0xc0, 0xfe, 0xff, 0xff, 0xff, 0xf4];
    // mov ax, 0x1234
    // hlt
    pub(super) const MOV_AX_IMM: &[u8] = &[0x66, 0xb8, 0x34, 0x12, 0xf4];
    // mov al, 0x80
    // hlt
    pub(super) const MOV_AL_IMM: &[u8] = &[0xb0, 0x80, 0xf4];
    // mov r9d, 7
    // hlt
    pub(super) const MOV_R9D_IMM: &[u8] = &[0x41, 0xb9, 0x07, 0x00, 0x00, 0x00, 0xf4];
    // movabs r12, 0xfedcba9876543210
    // hlt
    pub(super) const MOV_R12_IMM64: &[u8] = &[0x49, 0xbc, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0xf4];
    // mov byte ptr [rdx], 0x9c
    // hlt
    pub(super) const MOV_MEM8_IMM: &[u8] = &[0xc6, 0x02, 0x9c, 0xf4];
    // mov word ptr [rdx], 0x9abc
    // hlt
    pub(super) const MOV_MEM16_IMM: &[u8] = &[0x66, 0xc7, 0x02, 0xbc, 0x9a, 0xf4];
    // mov dword ptr [rdx], 0x89abcdef
    // hlt
    pub(super) const MOV_MEM32_IMM: &[u8] = &[0xc7, 0x02, 0xef, 0xcd, 0xab, 0x89, 0xf4];
    // mov qword ptr [rdx], -3
    // hlt
    pub(super) const MOV_MEM64_IMM: &[u8] = &[0x48, 0xc7, 0x02, 0xfd, 0xff, 0xff, 0xff, 0xf4];
    // mov rbx, 0x7fffffff
    // hlt
    pub(super) const MOV_RBX_IMM32_REG: &[u8] = &[0x48, 0xc7, 0xc3, 0xff, 0xff, 0xff, 0x7f, 0xf4];
    // mov rcx, -1
    // hlt
    pub(super) const MOV_RM64_IMM_REG: &[u8] = &[0x48, 0xc7, 0xc1, 0xff, 0xff, 0xff, 0xff, 0xf4];
    // mov qword ptr [rdx], rax
    // hlt
    pub(super) const MOV_MEM_RAX: &[u8] = &[0x48, 0x89, 0x02, 0xf4];
    // mov dword ptr [rdx], eax
    // hlt
    pub(super) const MOV_MEM_EAX: &[u8] = &[0x89, 0x02, 0xf4];
    // mov word ptr [rdx], ax
    // hlt
    pub(super) const MOV_MEM_AX: &[u8] = &[0x66, 0x89, 0x02, 0xf4];
    // mov byte ptr [rdx], al
    // hlt
    pub(super) const MOV_MEM_AL: &[u8] = &[0x88, 0x02, 0xf4];
    // mov byte ptr [rdx], ah
    // hlt
    pub(super) const MOV_MEM_AH: &[u8] = &[0x88, 0x22, 0xf4];
    // mov rax, qword ptr [rdx]
    // hlt
    pub(super) const MOV_RAX_MEM: &[u8] = &[0x48, 0x8b, 0x02, 0xf4];
    // mov eax, dword ptr [rdx]
    // hlt
    pub(super) const MOV_EAX_MEM: &[u8] = &[0x8b, 0x02, 0xf4];
    // mov ax, word ptr [rdx]
    // hlt
    pub(super) const MOV_AX_MEM: &[u8] = &[0x66, 0x8b, 0x02, 0xf4];
    // mov al, byte ptr [rdx]
    // hlt
    pub(super) const MOV_AL_MEM: &[u8] = &[0x8a, 0x02, 0xf4];
    // mov ah, byte ptr [rdx]
    // hlt
    pub(super) const MOV_AH_MEM: &[u8] = &[0x8a, 0x22, 0xf4];
    // mov r8b, byte ptr [rdx]
    // hlt
    pub(super) const MOV_BYTE_R8B_MEM: &[u8] = &[0x44, 0x8a, 0x02, 0xf4];
    // movabs al, [0x400000]
    // hlt
    pub(super) const MOVABS_AL_MOFFS: &[u8] = &[0xa0, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs ax, [0x400000]
    // hlt
    pub(super) const MOVABS_AX_MOFFS: &[u8] = &[0x66, 0xa1, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs eax, [0x400000]
    // hlt
    pub(super) const MOVABS_EAX_MOFFS: &[u8] = &[0xa1, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs rax, [0x400000]
    // hlt
    pub(super) const MOVABS_RAX_MOFFS: &[u8] = &[0x48, 0xa1, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs [0x400000], al
    // hlt
    pub(super) const MOVABS_MOFFS_AL: &[u8] = &[0xa2, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs [0x400000], ax
    // hlt
    pub(super) const MOVABS_MOFFS_AX: &[u8] = &[0x66, 0xa3, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs [0x400000], eax
    // hlt
    pub(super) const MOVABS_MOFFS_EAX: &[u8] = &[0xa3, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs [0x400000], rax
    // hlt
    pub(super) const MOVABS_MOFFS_RAX: &[u8] = &[0x48, 0xa3, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x67, 0xa1, 0x00, 0x00, 0x40, 0x00
    // hlt
    pub(super) const MOVABS_EAX_MOFFS_ADDR32: &[u8] = &[0x67, 0xa1, 0x00, 0x00, 0x40, 0x00, 0xf4];
    // .byte 0x67, 0xa3, 0x00, 0x00, 0x40, 0x00
    // hlt
    pub(super) const MOVABS_MOFFS_EAX_ADDR32: &[u8] = &[0x67, 0xa3, 0x00, 0x00, 0x40, 0x00, 0xf4];
    // .byte 0x64
    // movabs al, [0x10]
    // hlt
    pub(super) const MOVABS_AL_MOFFS_FS: &[u8] = &[0x64, 0xa0, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // movabs al, [0x100000000]
    // hlt
    pub(super) const MOVABS_AL_MOFFS_HIGH: &[u8] = &[0xa0, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xf4];
    // movzx eax, bl
    // hlt
    pub(super) const MOVZX_EAX_BL: &[u8] = &[0x0f, 0xb6, 0xc3, 0xf4];
    // movzx rax, bl
    // hlt
    pub(super) const MOVZX_RAX_BL: &[u8] = &[0x48, 0x0f, 0xb6, 0xc3, 0xf4];
    // movzx ax, bl
    // hlt
    pub(super) const MOVZX_AX_BL: &[u8] = &[0x66, 0x0f, 0xb6, 0xc3, 0xf4];
    // movzx eax, bx
    // hlt
    pub(super) const MOVZX_EAX_BX: &[u8] = &[0x0f, 0xb7, 0xc3, 0xf4];
    // movzx rax, bx
    // hlt
    pub(super) const MOVZX_RAX_BX: &[u8] = &[0x48, 0x0f, 0xb7, 0xc3, 0xf4];
    // movsx eax, bl
    // hlt
    pub(super) const MOVSX_EAX_BL: &[u8] = &[0x0f, 0xbe, 0xc3, 0xf4];
    // movsx rax, bl
    // hlt
    pub(super) const MOVSX_RAX_BL: &[u8] = &[0x48, 0x0f, 0xbe, 0xc3, 0xf4];
    // movsx ax, bl
    // hlt
    pub(super) const MOVSX_AX_BL: &[u8] = &[0x66, 0x0f, 0xbe, 0xc3, 0xf4];
    // movsx eax, bx
    // hlt
    pub(super) const MOVSX_EAX_BX: &[u8] = &[0x0f, 0xbf, 0xc3, 0xf4];
    // movsx rax, bx
    // hlt
    pub(super) const MOVSX_RAX_BX: &[u8] = &[0x48, 0x0f, 0xbf, 0xc3, 0xf4];
    // movzx eax, byte ptr [rdx]
    // hlt
    pub(super) const MOVZX_EAX_MEM8: &[u8] = &[0x0f, 0xb6, 0x02, 0xf4];
    // movzx rax, word ptr [rdx]
    // hlt
    pub(super) const MOVZX_RAX_MEM16: &[u8] = &[0x48, 0x0f, 0xb7, 0x02, 0xf4];
    // movsx rax, byte ptr [rdx]
    // hlt
    pub(super) const MOVSX_RAX_MEM8: &[u8] = &[0x48, 0x0f, 0xbe, 0x02, 0xf4];
    // movsx eax, word ptr [rdx]
    // hlt
    pub(super) const MOVSX_EAX_MEM16: &[u8] = &[0x0f, 0xbf, 0x02, 0xf4];
    // movsxd rax, ebx
    // hlt
    pub(super) const MOVSXD_RAX_EBX: &[u8] = &[0x48, 0x63, 0xc3, 0xf4];
    // movsxd rax, dword ptr [rdx]
    // hlt
    pub(super) const MOVSXD_RAX_MEM: &[u8] = &[0x48, 0x63, 0x02, 0xf4];
    // .byte 0x63, 0xc3
    // hlt
    pub(super) const MOVSXD_32_EBX: &[u8] = &[0x63, 0xc3, 0xf4];
    // lea rax, [rbx+rcx*4+8]
    // hlt
    pub(super) const LEA_RAX_SIB: &[u8] = &[0x48, 0x8d, 0x44, 0x8b, 0x08, 0xf4];
    // lea eax, [rbx+rcx*4+8]
    // hlt
    pub(super) const LEA_EAX_SIB: &[u8] = &[0x8d, 0x44, 0x8b, 0x08, 0xf4];
    // lea ax, [rbx+rcx*4+8]
    // hlt
    pub(super) const LEA_AX_SIB: &[u8] = &[0x66, 0x8d, 0x44, 0x8b, 0x08, 0xf4];
    // lea rax, [rip+0x10]
    // hlt
    pub(super) const LEA_RAX_RIP: &[u8] = &[0x48, 0x8d, 0x05, 0x10, 0x00, 0x00, 0x00, 0xf4];
    // lea rax, [0x1234]
    // hlt
    pub(super) const LEA_RAX_DISP32: &[u8] = &[0x48, 0x8d, 0x04, 0x25, 0x34, 0x12, 0x00, 0x00, 0xf4];
    // lea rax, [rbx-0x10]
    // hlt
    pub(super) const LEA_RAX_BASE_NEG: &[u8] = &[0x48, 0x8d, 0x43, 0xf0, 0xf4];
    // lea rax, [r12+r13*2+0x7f]
    // hlt
    pub(super) const LEA_RAX_R12_R13: &[u8] = &[0x4b, 0x8d, 0x44, 0x6c, 0x7f, 0xf4];
    // lea rax, [rbp]
    // hlt
    pub(super) const LEA_RAX_RBP: &[u8] = &[0x48, 0x8d, 0x45, 0x00, 0xf4];
    // lea rax, [r13]
    // hlt
    pub(super) const LEA_RAX_R13: &[u8] = &[0x49, 0x8d, 0x45, 0x00, 0xf4];
    // lea rax, [rsp+8]
    // hlt
    pub(super) const LEA_RAX_RSP: &[u8] = &[0x48, 0x8d, 0x44, 0x24, 0x08, 0xf4];
    // lea rax, [rcx*8+0x100]
    // hlt
    pub(super) const LEA_RAX_SCALE8: &[u8] = &[0x48, 0x8d, 0x04, 0xcd, 0x00, 0x01, 0x00, 0x00, 0xf4];
    // lea eax, [ebx+ecx*2]
    // hlt
    pub(super) const LEA_EAX_ADDR32: &[u8] = &[0x67, 0x8d, 0x04, 0x4b, 0xf4];
    // lea rax, [ebx+ecx*2]
    // hlt
    pub(super) const LEA_RAX_ADDR32: &[u8] = &[0x67, 0x48, 0x8d, 0x04, 0x4b, 0xf4];
    // .byte 0x64
    // lea rax, [rbx+8]
    // hlt
    pub(super) const LEA_RAX_FS: &[u8] = &[0x64, 0x48, 0x8d, 0x43, 0x08, 0xf4];
    // .byte 0x48, 0x8d, 0xc3
    // hlt
    pub(super) const LEA_REG_FORM: &[u8] = &[0x48, 0x8d, 0xc3, 0xf4];
    // xchg rax, rbx
    // hlt
    pub(super) const XCHG_RAX_RBX: &[u8] = &[0x48, 0x93, 0xf4];
    // xchg eax, ebx
    // hlt
    pub(super) const XCHG_EAX_EBX: &[u8] = &[0x93, 0xf4];
    // xchg ax, bx
    // hlt
    pub(super) const XCHG_AX_BX: &[u8] = &[0x66, 0x93, 0xf4];
    // xchg al, bl
    // hlt
    pub(super) const XCHG_AL_BL: &[u8] = &[0x86, 0xc3, 0xf4];
    // xchg qword ptr [rdx], rbx
    // hlt
    pub(super) const XCHG_MEM_RBX: &[u8] = &[0x48, 0x87, 0x1a, 0xf4];
    // xchg dword ptr [rdx], ebx
    // hlt
    pub(super) const XCHG_MEM_EBX: &[u8] = &[0x87, 0x1a, 0xf4];
    // xchg word ptr [rdx], bx
    // hlt
    pub(super) const XCHG_MEM_BX: &[u8] = &[0x66, 0x87, 0x1a, 0xf4];
    // xchg byte ptr [rdx], bl
    // hlt
    pub(super) const XCHG_MEM_BL: &[u8] = &[0x86, 0x1a, 0xf4];
    // xchg r8, rax
    // hlt
    pub(super) const XCHG_R8_RAX: &[u8] = &[0x49, 0x90, 0xf4];
    // xchg r8d, eax
    // hlt
    pub(super) const XCHG_R8D_EAX: &[u8] = &[0x41, 0x90, 0xf4];
    // .byte 0x87, 0xc0
    // hlt
    pub(super) const XCHG_EAX_EAX_LONG: &[u8] = &[0x87, 0xc0, 0xf4];
    // xchg rbx, rax
    // hlt
    pub(super) const XCHG_RAX_RBX_MODRM: &[u8] = &[0x48, 0x93, 0xf4];
    // nop
    // hlt
    pub(super) const NOP: &[u8] = &[0x90, 0xf4];
    // pause
    // hlt
    pub(super) const PAUSE: &[u8] = &[0xf3, 0x90, 0xf4];
    // endbr64
    // hlt
    pub(super) const ENDBR64: &[u8] = &[0xf3, 0x0f, 0x1e, 0xfa, 0xf4];
    // nop dword ptr [rax]
    // hlt
    pub(super) const NOP_MEM_UNMAPPED: &[u8] = &[0x0f, 0x1f, 0x00, 0xf4];
    // nop word ptr [rax+rax*1+0]
    // hlt
    pub(super) const NOP_WORD_MEM: &[u8] = &[0x66, 0x0f, 0x1f, 0x04, 0x00, 0xf4];
    // prefetchnta [rax]
    // hlt
    pub(super) const PREFETCH: &[u8] = &[0x0f, 0x18, 0x00, 0xf4];
    // cbw
    // hlt
    pub(super) const CBW: &[u8] = &[0x66, 0x98, 0xf4];
    // cwde
    // hlt
    pub(super) const CWDE: &[u8] = &[0x98, 0xf4];
    // cdqe
    // hlt
    pub(super) const CDQE: &[u8] = &[0x48, 0x98, 0xf4];
    // cwd
    // hlt
    pub(super) const CWD: &[u8] = &[0x66, 0x99, 0xf4];
    // cdq
    // hlt
    pub(super) const CDQ: &[u8] = &[0x99, 0xf4];
    // cqo
    // hlt
    pub(super) const CQO: &[u8] = &[0x48, 0x99, 0xf4];
    // bswap eax
    // hlt
    pub(super) const BSWAP_EAX: &[u8] = &[0x0f, 0xc8, 0xf4];
    // bswap rax
    // hlt
    pub(super) const BSWAP_RAX: &[u8] = &[0x48, 0x0f, 0xc8, 0xf4];
    // bswap r9d
    // hlt
    pub(super) const BSWAP_R9D: &[u8] = &[0x41, 0x0f, 0xc9, 0xf4];
    // bswap r10
    // hlt
    pub(super) const BSWAP_R10: &[u8] = &[0x49, 0x0f, 0xca, 0xf4];
    // lahf
    // hlt
    pub(super) const LAHF: &[u8] = &[0x9f, 0xf4];
    // sahf
    // hlt
    pub(super) const SAHF: &[u8] = &[0x9e, 0xf4];
    // cmc
    // hlt
    pub(super) const CMC: &[u8] = &[0xf5, 0xf4];
    // clc
    // hlt
    pub(super) const CLC: &[u8] = &[0xf8, 0xf4];
    // stc
    // hlt
    pub(super) const STC: &[u8] = &[0xf9, 0xf4];
    // cld
    // hlt
    pub(super) const CLD: &[u8] = &[0xfc, 0xf4];
    // std
    // hlt
    pub(super) const STD: &[u8] = &[0xfd, 0xf4];
    // movnti dword ptr [rdx], eax
    // hlt
    pub(super) const MOVNTI_MEM32: &[u8] = &[0x0f, 0xc3, 0x02, 0xf4];
    // movnti qword ptr [rdx], rax
    // hlt
    pub(super) const MOVNTI_MEM64: &[u8] = &[0x48, 0x0f, 0xc3, 0x02, 0xf4];
    // .byte 0x0f, 0xc3, 0xc2
    // hlt
    pub(super) const MOVNTI_REG: &[u8] = &[0x0f, 0xc3, 0xc2, 0xf4];
    // cmpxchg qword ptr [rdx], rbx
    // hlt
    pub(super) const CMPXCHG_MEM_8: &[u8] = &[0x48, 0x0f, 0xb1, 0x1a, 0xf4];
    // cmpxchg rcx, rbx
    // hlt
    pub(super) const CMPXCHG_REG_8: &[u8] = &[0x48, 0x0f, 0xb1, 0xd9, 0xf4];
    // lock cmpxchg qword ptr [rdx], rbx
    // hlt
    pub(super) const LOCK_CMPXCHG_MEM_8: &[u8] = &[0xf0, 0x48, 0x0f, 0xb1, 0x1a, 0xf4];
    // xadd qword ptr [rdx], rbx
    // hlt
    pub(super) const XADD_MEM_8: &[u8] = &[0x48, 0x0f, 0xc1, 0x1a, 0xf4];
    // xadd rcx, rbx
    // hlt
    pub(super) const XADD_REG_8: &[u8] = &[0x48, 0x0f, 0xc1, 0xd9, 0xf4];
    // lock xadd qword ptr [rdx], rbx
    // hlt
    pub(super) const LOCK_XADD_MEM_8: &[u8] = &[0xf0, 0x48, 0x0f, 0xc1, 0x1a, 0xf4];
    // cmpxchg dword ptr [rdx], ebx
    // hlt
    pub(super) const CMPXCHG_MEM_4: &[u8] = &[0x0f, 0xb1, 0x1a, 0xf4];
    // cmpxchg ecx, ebx
    // hlt
    pub(super) const CMPXCHG_REG_4: &[u8] = &[0x0f, 0xb1, 0xd9, 0xf4];
    // lock cmpxchg dword ptr [rdx], ebx
    // hlt
    pub(super) const LOCK_CMPXCHG_MEM_4: &[u8] = &[0xf0, 0x0f, 0xb1, 0x1a, 0xf4];
    // xadd dword ptr [rdx], ebx
    // hlt
    pub(super) const XADD_MEM_4: &[u8] = &[0x0f, 0xc1, 0x1a, 0xf4];
    // xadd ecx, ebx
    // hlt
    pub(super) const XADD_REG_4: &[u8] = &[0x0f, 0xc1, 0xd9, 0xf4];
    // lock xadd dword ptr [rdx], ebx
    // hlt
    pub(super) const LOCK_XADD_MEM_4: &[u8] = &[0xf0, 0x0f, 0xc1, 0x1a, 0xf4];
    // cmpxchg word ptr [rdx], bx
    // hlt
    pub(super) const CMPXCHG_MEM_2: &[u8] = &[0x66, 0x0f, 0xb1, 0x1a, 0xf4];
    // cmpxchg cx, bx
    // hlt
    pub(super) const CMPXCHG_REG_2: &[u8] = &[0x66, 0x0f, 0xb1, 0xd9, 0xf4];
    // lock cmpxchg word ptr [rdx], bx
    // hlt
    pub(super) const LOCK_CMPXCHG_MEM_2: &[u8] = &[0x66, 0xf0, 0x0f, 0xb1, 0x1a, 0xf4];
    // xadd word ptr [rdx], bx
    // hlt
    pub(super) const XADD_MEM_2: &[u8] = &[0x66, 0x0f, 0xc1, 0x1a, 0xf4];
    // xadd cx, bx
    // hlt
    pub(super) const XADD_REG_2: &[u8] = &[0x66, 0x0f, 0xc1, 0xd9, 0xf4];
    // lock xadd word ptr [rdx], bx
    // hlt
    pub(super) const LOCK_XADD_MEM_2: &[u8] = &[0x66, 0xf0, 0x0f, 0xc1, 0x1a, 0xf4];
    // cmpxchg byte ptr [rdx], bl
    // hlt
    pub(super) const CMPXCHG_MEM_1: &[u8] = &[0x0f, 0xb0, 0x1a, 0xf4];
    // cmpxchg cl, bl
    // hlt
    pub(super) const CMPXCHG_REG_1: &[u8] = &[0x0f, 0xb0, 0xd9, 0xf4];
    // lock cmpxchg byte ptr [rdx], bl
    // hlt
    pub(super) const LOCK_CMPXCHG_MEM_1: &[u8] = &[0xf0, 0x0f, 0xb0, 0x1a, 0xf4];
    // xadd byte ptr [rdx], bl
    // hlt
    pub(super) const XADD_MEM_1: &[u8] = &[0x0f, 0xc0, 0x1a, 0xf4];
    // xadd cl, bl
    // hlt
    pub(super) const XADD_REG_1: &[u8] = &[0x0f, 0xc0, 0xd9, 0xf4];
    // lock xadd byte ptr [rdx], bl
    // hlt
    pub(super) const LOCK_XADD_MEM_1: &[u8] = &[0xf0, 0x0f, 0xc0, 0x1a, 0xf4];
    // .byte 0x0f, 0xc7, 0xca
    // hlt
    pub(super) const CMPXCHG8B_REG: &[u8] = &[0x0f, 0xc7, 0xca, 0xf4];
    // .byte 0x0f, 0xc7, 0x02
    // hlt
    pub(super) const GROUP9_REG0: &[u8] = &[0x0f, 0xc7, 0x02, 0xf4];
    // lock add qword ptr [rdx], rbx
    // hlt
    pub(super) const LOCK_ADD_MEM: &[u8] = &[0xf0, 0x48, 0x01, 0x1a, 0xf4];
    // lock inc dword ptr [rdx]
    // hlt
    pub(super) const LOCK_INC_MEM: &[u8] = &[0xf0, 0xff, 0x02, 0xf4];
    // lock xchg qword ptr [rdx], rbx
    // hlt
    pub(super) const LOCK_XCHG_MEM: &[u8] = &[0xf0, 0x48, 0x87, 0x1a, 0xf4];
    // cmpxchg8b [rdi]
    // hlt
    pub(super) const CMPXCHG8B_RDI: &[u8] = &[0x0f, 0xc7, 0x0f, 0xf4];
    // lock cmpxchg8b [rdi]
    // hlt
    pub(super) const LOCK_CMPXCHG8B_RDI: &[u8] = &[0xf0, 0x0f, 0xc7, 0x0f, 0xf4];
    // shld dword ptr [rdx], ebx, cl
    // hlt
    pub(super) const SHLD_MEM_CL: &[u8] = &[0x0f, 0xa5, 0x1a, 0xf4];
    // shrd qword ptr [rdx], rbx, 3
    // hlt
    pub(super) const SHRD_MEM_IMM: &[u8] = &[0x48, 0x0f, 0xac, 0x1a, 0x03, 0xf4];
    // bts qword ptr [rdx], rbx
    // hlt
    pub(super) const BTS_MEM_REG: &[u8] = &[0x48, 0x0f, 0xab, 0x1a, 0xf4];
    // btc dword ptr [rdx], 5
    // hlt
    pub(super) const BTC_MEM_IMM: &[u8] = &[0x0f, 0xba, 0x3a, 0x05, 0xf4];
    // btr dword ptr [rdx], ebx
    // hlt
    pub(super) const BTR_MEM_REG: &[u8] = &[0x0f, 0xb3, 0x1a, 0xf4];
    // sete byte ptr [rdx]
    // hlt
    pub(super) const SETE_MEM: &[u8] = &[0x0f, 0x94, 0x02, 0xf4];
    // mov qword ptr [rdx], rax
    // hlt
    pub(super) const MOV_MEM_RAX2: &[u8] = &[0x48, 0x89, 0x02, 0xf4];
    // jmp over
    // mov eax, 1
    // hlt
    // over: mov eax, 2
    // hlt
    pub(super) const JMP_FWD: &[u8] = &[0xeb, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0xe9, 0x06, 0x00, 0x00, 0x00
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JMP_REL32: &[u8] = &[0xe9, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jmp fwd
    // back: mov eax, 3
    // hlt
    // fwd: jmp back
    pub(super) const JMP_BACK: &[u8] = &[0xeb, 0x06, 0xb8, 0x03, 0x00, 0x00, 0x00, 0xf4, 0xeb, 0xf8];
    // jmp rax
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const JMP_REG: &[u8] = &[0xff, 0xe0, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jmp qword ptr [rdx]
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const JMP_MEM: &[u8] = &[0xff, 0x22, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jmp r9
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const JMP_R9: &[u8] = &[0x41, 0xff, 0xe1, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // call rax
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const CALL_REG: &[u8] = &[0xff, 0xd0, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // call qword ptr [rdx]
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const CALL_MEM: &[u8] = &[0xff, 0x12, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // call qword ptr [rsp]
    // mov ecx, 1
    // hlt
    // mov ecx, 2
    // hlt
    pub(super) const CALL_RSP_MEM: &[u8] = &[0xff, 0x14, 0x24, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // call func
    // mov eax, 5
    // hlt
    // func: mov ebx, 6
    // ret
    pub(super) const CALL_RET: &[u8] = &[0xe8, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xf4, 0xbb, 0x06, 0x00, 0x00, 0x00, 0xc3];
    // call func
    // mov eax, 5
    // hlt
    // func: mov ebx, 6
    // ret 16
    pub(super) const CALL_RET_IMM: &[u8] = &[0xe8, 0x06, 0x00, 0x00, 0x00, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xf4, 0xbb, 0x06, 0x00, 0x00, 0x00, 0xc2, 0x10, 0x00];
    // .byte 0xf3, 0xc3
    // hlt
    pub(super) const REP_RET: &[u8] = &[0xf3, 0xc3, 0xf4];
    // .byte 0xff, 0x2a
    // hlt
    pub(super) const JMP_FAR_IND: &[u8] = &[0xff, 0x2a, 0xf4];
    // .byte 0xff, 0x1a
    // hlt
    pub(super) const CALL_FAR_IND: &[u8] = &[0xff, 0x1a, 0xf4];
    // .byte 0xff, 0xfa
    // hlt
    pub(super) const GROUP5_REG7: &[u8] = &[0xff, 0xfa, 0xf4];
    // .byte 0xfe, 0xd0
    // hlt
    pub(super) const GROUP4_CALL: &[u8] = &[0xfe, 0xd0, 0xf4];
    // .byte 0xfe, 0xf0
    // hlt
    pub(super) const GROUP4_PUSH: &[u8] = &[0xfe, 0xf0, 0xf4];
    // .byte 0xfe, 0xe0
    // hlt
    pub(super) const GROUP4_JMP: &[u8] = &[0xfe, 0xe0, 0xf4];
    // loop target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const LOOP_: &[u8] = &[0xe2, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // loope target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const LOOPE: &[u8] = &[0xe1, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // loopne target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const LOOPNE: &[u8] = &[0xe0, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // jrcxz target
    // mov eax, 1
    // hlt
    // target: mov eax, 2
    // hlt
    pub(super) const JRCXZ: &[u8] = &[0xe3, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x67, 0xe2, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const LOOP_ADDR32: &[u8] = &[0x67, 0xe2, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x67, 0xe1, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const LOOPE_ADDR32: &[u8] = &[0x67, 0xe1, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x67, 0xe0, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const LOOPNE_ADDR32: &[u8] = &[0x67, 0xe0, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x67, 0xe3, 0x06
    // mov eax, 1
    // hlt
    // mov eax, 2
    // hlt
    pub(super) const JECXZ: &[u8] = &[0x67, 0xe3, 0x06, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb8, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // push rax
    // hlt
    pub(super) const PUSH_RAX_: &[u8] = &[0x50, 0xf4];
    // push r8
    // hlt
    pub(super) const PUSH_R8: &[u8] = &[0x41, 0x50, 0xf4];
    // push rsp
    // hlt
    pub(super) const PUSH_RSP_: &[u8] = &[0x54, 0xf4];
    // push -2
    // hlt
    pub(super) const PUSH_IMM8: &[u8] = &[0x6a, 0xfe, 0xf4];
    // push 0x12345678
    // hlt
    pub(super) const PUSH_IMM32: &[u8] = &[0x68, 0x78, 0x56, 0x34, 0x12, 0xf4];
    // push -0x12345678
    // hlt
    pub(super) const PUSH_IMM32_NEG: &[u8] = &[0x68, 0x88, 0xa9, 0xcb, 0xed, 0xf4];
    // push qword ptr [rdx]
    // hlt
    pub(super) const PUSH_MEM: &[u8] = &[0xff, 0x32, 0xf4];
    // push ax
    // hlt
    pub(super) const PUSH_AX: &[u8] = &[0x66, 0x50, 0xf4];
    // .byte 0x66, 0x68, 0x34, 0x12
    // hlt
    pub(super) const PUSH_IMM16: &[u8] = &[0x66, 0x68, 0x34, 0x12, 0xf4];
    // .byte 0x66, 0x6a, 0xfe
    // hlt
    pub(super) const PUSH_IMM8_WORD: &[u8] = &[0x66, 0x6a, 0xfe, 0xf4];
    // push word ptr [rdx]
    // hlt
    pub(super) const PUSH_MEM16: &[u8] = &[0x66, 0xff, 0x32, 0xf4];
    // .byte 0x0f, 0xa0
    // hlt
    pub(super) const PUSH_FS: &[u8] = &[0x0f, 0xa0, 0xf4];
    // push fs
    // hlt
    pub(super) const PUSH_FS_16: &[u8] = &[0x66, 0x0f, 0xa0, 0xf4];
    // .byte 0x0f, 0xa8
    // hlt
    pub(super) const PUSH_GS: &[u8] = &[0x0f, 0xa8, 0xf4];
    // pop rax
    // hlt
    pub(super) const POP_RAX_: &[u8] = &[0x58, 0xf4];
    // pop r9
    // hlt
    pub(super) const POP_R9: &[u8] = &[0x41, 0x59, 0xf4];
    // pop rsp
    // hlt
    pub(super) const POP_RSP_: &[u8] = &[0x5c, 0xf4];
    // pop qword ptr [rdx]
    // hlt
    pub(super) const POP_MEM: &[u8] = &[0x8f, 0x02, 0xf4];
    // pop qword ptr [rsp]
    // hlt
    pub(super) const POP_MEM_RSP: &[u8] = &[0x8f, 0x04, 0x24, 0xf4];
    // pop qword ptr [rsp+8]
    // hlt
    pub(super) const POP_MEM_RSP_DISP: &[u8] = &[0x8f, 0x44, 0x24, 0x08, 0xf4];
    // pop ax
    // hlt
    pub(super) const POP_AX: &[u8] = &[0x66, 0x58, 0xf4];
    // pop word ptr [rdx]
    // hlt
    pub(super) const POP_MEM16: &[u8] = &[0x66, 0x8f, 0x02, 0xf4];
    // .byte 0x0f, 0xa1
    // hlt
    pub(super) const POP_FS: &[u8] = &[0x0f, 0xa1, 0xf4];
    // .byte 0x0f, 0xa9
    // hlt
    pub(super) const POP_GS: &[u8] = &[0x0f, 0xa9, 0xf4];
    // .byte 0x66, 0x0f, 0xa1
    // hlt
    pub(super) const POP_FS_16: &[u8] = &[0x66, 0x0f, 0xa1, 0xf4];
    // pushfw
    // hlt
    pub(super) const PUSHFW: &[u8] = &[0x66, 0x9c, 0xf4];
    // popfw
    // hlt
    pub(super) const POPFW: &[u8] = &[0x66, 0x9d, 0xf4];
    // cli
    // hlt
    pub(super) const CLI: &[u8] = &[0xfa, 0xf4];
    // sti
    // hlt
    pub(super) const STI: &[u8] = &[0xfb, 0xf4];
    // int3
    // hlt
    pub(super) const INT3: &[u8] = &[0xcc, 0xf4];
    // int 0x80
    // hlt
    pub(super) const INT_80: &[u8] = &[0xcd, 0x80, 0xf4];
    // .byte 0xcd, 0x03
    // hlt
    pub(super) const INT_3_LONG: &[u8] = &[0xcd, 0x03, 0xf4];
    // .byte 0xf1
    // hlt
    pub(super) const INT1: &[u8] = &[0xf1, 0xf4];
    // .byte 0xce
    // hlt
    pub(super) const INTO: &[u8] = &[0xce, 0xf4];
    // rdtsc
    // hlt
    pub(super) const RDTSC_: &[u8] = &[0x0f, 0x31, 0xf4];
    // rdtsc
    // mov r8, rax
    // mov r9, rdx
    // rdtsc
    // hlt
    pub(super) const RDTSC_TWICE: &[u8] = &[0x0f, 0x31, 0x49, 0x89, 0xc0, 0x49, 0x89, 0xd1, 0x0f, 0x31, 0xf4];
    // rdmsr
    // hlt
    pub(super) const RDMSR_: &[u8] = &[0x0f, 0x32, 0xf4];
    // wrmsr
    // hlt
    pub(super) const WRMSR_: &[u8] = &[0x0f, 0x30, 0xf4];
    // cpuid
    // hlt
    pub(super) const CPUID_: &[u8] = &[0x0f, 0xa2, 0xf4];
    // clts
    // hlt
    pub(super) const CLTS: &[u8] = &[0x0f, 0x06, 0xf4];
    // wbinvd
    // hlt
    pub(super) const WBINVD: &[u8] = &[0x0f, 0x09, 0xf4];
    // invd
    // hlt
    pub(super) const INVD: &[u8] = &[0x0f, 0x08, 0xf4];
    // ud2
    // hlt
    pub(super) const UD2_: &[u8] = &[0x0f, 0x0b, 0xf4];
    // xor eax, eax
    // again: add rax, rcx
    // dec rcx
    // jne again
    // hlt
    pub(super) const LOOP_COUNTING_DOWN: &[u8] = &[0x31, 0xc0, 0x48, 0x01, 0xc8, 0x48, 0xff, 0xc9, 0x75, 0xf8, 0xf4];
    // .byte 0x06
    // hlt
    pub(super) const INVALID_06: &[u8] = &[0x06, 0xf4];
    // .byte 0x07
    // hlt
    pub(super) const INVALID_07: &[u8] = &[0x07, 0xf4];
    // .byte 0x0e
    // hlt
    pub(super) const INVALID_0E: &[u8] = &[0x0e, 0xf4];
    // .byte 0x16
    // hlt
    pub(super) const INVALID_16: &[u8] = &[0x16, 0xf4];
    // .byte 0x17
    // hlt
    pub(super) const INVALID_17: &[u8] = &[0x17, 0xf4];
    // .byte 0x1e
    // hlt
    pub(super) const INVALID_1E: &[u8] = &[0x1e, 0xf4];
    // .byte 0x1f
    // hlt
    pub(super) const INVALID_1F: &[u8] = &[0x1f, 0xf4];
    // .byte 0x27
    // hlt
    pub(super) const INVALID_27: &[u8] = &[0x27, 0xf4];
    // .byte 0x2f
    // hlt
    pub(super) const INVALID_2F: &[u8] = &[0x2f, 0xf4];
    // .byte 0x37
    // hlt
    pub(super) const INVALID_37: &[u8] = &[0x37, 0xf4];
    // .byte 0x3f
    // hlt
    pub(super) const INVALID_3F: &[u8] = &[0x3f, 0xf4];
    // .byte 0x60
    // hlt
    pub(super) const INVALID_60: &[u8] = &[0x60, 0xf4];
    // .byte 0x61
    // hlt
    pub(super) const INVALID_61: &[u8] = &[0x61, 0xf4];
    // .byte 0xce
    // hlt
    pub(super) const INVALID_CE: &[u8] = &[0xce, 0xf4];
    // .byte 0xd6
    // hlt
    pub(super) const INVALID_D6: &[u8] = &[0xd6, 0xf4];
    // .byte 0x62, 0x00
    // hlt
    pub(super) const INVALID_62_00: &[u8] = &[0x62, 0x00, 0xf4];
    // .byte 0x82, 0xc0, 0x01
    // hlt
    pub(super) const INVALID_82: &[u8] = &[0x82, 0xc0, 0x01, 0xf4];
    // .byte 0x9a, 0, 0, 0, 0, 0, 0
    // hlt
    pub(super) const INVALID_9A: &[u8] = &[0x9a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0xea, 0, 0, 0, 0, 0, 0
    // hlt
    pub(super) const INVALID_EA: &[u8] = &[0xea, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0xd4, 0x0a
    // hlt
    pub(super) const INVALID_D4: &[u8] = &[0xd4, 0x0a, 0xf4];
    // .byte 0xd5, 0x0a
    // hlt
    pub(super) const INVALID_D5: &[u8] = &[0xd5, 0x0a, 0xf4];
    // .byte 0xc4, 0x00
    // hlt
    pub(super) const INVALID_C4: &[u8] = &[0xc4, 0x00, 0xf4];
    // .byte 0xc5, 0x00
    // hlt
    pub(super) const INVALID_C5: &[u8] = &[0xc5, 0x00, 0xf4];
    // .byte 0x0f, 0x04, 0xc0
    // hlt
    pub(super) const UNDEF_0F04: &[u8] = &[0x0f, 0x04, 0xc0, 0xf4];
    // .byte 0x0f, 0x0a, 0xc0
    // hlt
    pub(super) const UNDEF_0F0A: &[u8] = &[0x0f, 0x0a, 0xc0, 0xf4];
    // .byte 0x0f, 0x0c, 0xc0
    // hlt
    pub(super) const UNDEF_0F0C: &[u8] = &[0x0f, 0x0c, 0xc0, 0xf4];
    // .byte 0x0f, 0x24, 0xc0
    // hlt
    pub(super) const UNDEF_0F24: &[u8] = &[0x0f, 0x24, 0xc0, 0xf4];
    // .byte 0x0f, 0x26, 0xc0
    // hlt
    pub(super) const UNDEF_0F26: &[u8] = &[0x0f, 0x26, 0xc0, 0xf4];
    // .byte 0x0f, 0xb9, 0xc0
    // hlt
    pub(super) const UNDEF_0FB9: &[u8] = &[0x0f, 0xb9, 0xc0, 0xf4];
    // .byte 0x0f, 0xff, 0xc0
    // hlt
    pub(super) const UNDEF_0FFF: &[u8] = &[0x0f, 0xff, 0xc0, 0xf4];
    // .byte 0x0f, 0x36, 0xc0
    // hlt
    pub(super) const UNDEF_0F36: &[u8] = &[0x0f, 0x36, 0xc0, 0xf4];
    // .byte 0x0f, 0xa6, 0xc0
    // hlt
    pub(super) const UNDEF_0FA6: &[u8] = &[0x0f, 0xa6, 0xc0, 0xf4];
    // .byte 0x0f, 0xa7, 0xc0
    // hlt
    pub(super) const UNDEF_0FA7: &[u8] = &[0x0f, 0xa7, 0xc0, 0xf4];
    // .byte 0x0f, 0x78, 0xc0
    // hlt
    pub(super) const UNDEF_0F78: &[u8] = &[0x0f, 0x78, 0xc0, 0xf4];
    // .byte 0x0f, 0x79, 0xc0
    // hlt
    pub(super) const UNDEF_0F79: &[u8] = &[0x0f, 0x79, 0xc0, 0xf4];
    // mov rax, fs:[rbx]
    // hlt
    pub(super) const FS_LOAD: &[u8] = &[0x64, 0x48, 0x8b, 0x03, 0xf4];
    // mov rax, gs:[0x10]
    // hlt
    pub(super) const GS_LOAD_DISP: &[u8] = &[0x65, 0x48, 0x8b, 0x04, 0x25, 0x10, 0x00, 0x00, 0x00, 0xf4];
    // mov rax, es:[rbx]
    // hlt
    pub(super) const ES_LOAD: &[u8] = &[0x26, 0x48, 0x8b, 0x03, 0xf4];
    // mov rax, cs:[rbx]
    // hlt
    pub(super) const CS_LOAD: &[u8] = &[0x2e, 0x48, 0x8b, 0x03, 0xf4];
    // mov rax, ss:[rbx]
    // hlt
    pub(super) const SS_LOAD: &[u8] = &[0x36, 0x48, 0x8b, 0x03, 0xf4];
    // mov rax, ds:[rbx]
    // hlt
    pub(super) const DS_LOAD: &[u8] = &[0x3e, 0x48, 0x8b, 0x03, 0xf4];
    // mov eax, fs:[ebx]
    // hlt
    pub(super) const FS_LOAD_ADDR32: &[u8] = &[0x64, 0x67, 0x8b, 0x03, 0xf4];
    // mov fs:[rbx], rax
    // hlt
    pub(super) const FS_STORE: &[u8] = &[0x64, 0x48, 0x89, 0x03, 0xf4];
    // mov rax, gs:[rip+0x100]
    // hlt
    pub(super) const GS_LOAD_RIP: &[u8] = &[0x65, 0x48, 0x8b, 0x05, 0x00, 0x01, 0x00, 0x00, 0xf4];
    // mov rax, fs:[rbx+rcx*8+0x20]
    // hlt
    pub(super) const FS_LOAD_SIB: &[u8] = &[0x64, 0x48, 0x8b, 0x44, 0xcb, 0x20, 0xf4];
    // mov rax, qword ptr [rbx]
    // hlt
    pub(super) const LOAD_RBX: &[u8] = &[0x48, 0x8b, 0x03, 0xf4];
    // fwait
    // hlt
    pub(super) const FWAIT: &[u8] = &[0x9b, 0xf4];
    // sysenter
    // hlt
    pub(super) const SYSENTER: &[u8] = &[0x0f, 0x34, 0xf4];
    // sysexit
    // hlt
    pub(super) const SYSEXIT: &[u8] = &[0x0f, 0x35, 0xf4];
    // mov qword ptr [rbx], rax
    // hlt
    pub(super) const STORE_RBX: &[u8] = &[0x48, 0x89, 0x03, 0xf4];
    // add qword ptr [rbx], rax
    // hlt
    pub(super) const ADD_MEM_RBX: &[u8] = &[0x48, 0x01, 0x03, 0xf4];
    // lea rax, [rbx]
    // hlt
    pub(super) const LEA_RBX: &[u8] = &[0x48, 0x8d, 0x03, 0xf4];
    // hlt
    pub(super) const HLT_ONLY: &[u8] = &[0xf4];
    // rdtsc
    // ud2
    pub(super) const RDTSC_UD2: &[u8] = &[0x0f, 0x31, 0x0f, 0x0b];
    // add rax, rbx
    // popfq
    // je target
    // mov ecx, 1
    // hlt
    // target: mov ecx, 2
    // hlt
    pub(super) const POPF_THEN_JE: &[u8] = &[0x48, 0x01, 0xd8, 0x9d, 0x74, 0x06, 0xb9, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xb9, 0x02, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0x8f, 0xc8
    // hlt
    pub(super) const POP_GROUP_REG1: &[u8] = &[0x8f, 0xc8, 0xf4];
    // .byte 0xc6, 0xc8, 0x05
    // hlt
    pub(super) const MOV_GROUP11_REG1_8: &[u8] = &[0xc6, 0xc8, 0x05, 0xf4];
    // .byte 0xc7, 0xc8, 0x05, 0x00, 0x00, 0x00
    // hlt
    pub(super) const MOV_GROUP11_REG1_32: &[u8] = &[0xc7, 0xc8, 0x05, 0x00, 0x00, 0x00, 0xf4];
    // .byte 0xc6, 0xf8, 0x05
    // hlt
    pub(super) const MOV_GROUP11_REG7_8: &[u8] = &[0xc6, 0xf8, 0x05, 0xf4];
    // rep stosb
    // hlt
    pub(super) const REP_STOSB_PLAIN: &[u8] = &[0xf3, 0xaa, 0xf4];
    // rep movsb
    // hlt
    pub(super) const REP_MOVSB_PLAIN: &[u8] = &[0xf3, 0xa4, 0xf4];
    // movsb
    // hlt
    pub(super) const MOVSB_PLAIN: &[u8] = &[0xa4, 0xf4];
    // cmpsb
    // hlt
    pub(super) const CMPSB_PLAIN: &[u8] = &[0xa6, 0xf4];
    // repe cmpsb
    // hlt
    pub(super) const REPE_CMPSB_PLAIN: &[u8] = &[0xf3, 0xa6, 0xf4];
    // lodsb
    // hlt
    pub(super) const LODSB_PLAIN: &[u8] = &[0xac, 0xf4];
    // stosb
    // hlt
    pub(super) const STOSB_PLAIN: &[u8] = &[0xaa, 0xf4];
    // scasb
    // hlt
    pub(super) const SCASB_PLAIN: &[u8] = &[0xae, 0xf4];
    // mov rax, qword ptr [rbx+4]
    // hlt
    pub(super) const LOAD64_RBX_DISP: &[u8] = &[0x48, 0x8b, 0x43, 0x04, 0xf4];
    // mov word ptr [rbx], ax
    // hlt
    pub(super) const STORE16_RBX: &[u8] = &[0x66, 0x89, 0x03, 0xf4];
    // .byte 0xf6, 0xc8, 0x0f
    // hlt
    pub(super) const TEST_ALIAS_F6_1: &[u8] = &[0xf6, 0xc8, 0x0f, 0xf4];
    // .byte 0xf7, 0xc8, 0x0f, 0x00, 0x00, 0x00
    // hlt
    pub(super) const TEST_ALIAS_F7_1: &[u8] = &[0xf7, 0xc8, 0x0f, 0x00, 0x00, 0x00, 0xf4];
    // mov al, 0xfe
    // out 0x64, al
    // hlt
    pub(super) const OUT_KBC_RESET: &[u8] = &[0xb0, 0xfe, 0xe6, 0x64, 0xf4];
    // syscall
    // hlt
    pub(super) const SYSCALL_: &[u8] = &[0x0f, 0x05, 0xf4];
    // sysretq
    // hlt
    pub(super) const SYSRETQ: &[u8] = &[0x48, 0x0f, 0x07, 0xf4];
    // sysret
    // hlt
    pub(super) const SYSRET_32: &[u8] = &[0x0f, 0x07, 0xf4];
    // rep stosb
    // hlt
    pub(super) const REP_STOSB_COUNTED: &[u8] = &[0xf3, 0xaa, 0xf4];
}
}

#[cfg(test)]
mod rep_slice_tests {
    //! Large `rep movsb`/`rep stosb` (the ERMS/FSRM idiom) cross several `REP_BYTES_PER_SLICE`
    //! slices, each ending with RIP still on the instruction.
    use super::*;
    use crate::testkit::{CODE_BASE, Harness, RAX, RCX, RDI, RSI, SCRATCH, Stop};

    const SOURCE: u64 = 0x20_0000;
    const DESTINATION: u64 = 0x40_0000;
    // rep movsb; hlt / rep stosb; hlt
    const REP_MOVSB: &[u8] = &[0xf3, 0xa4, 0xf4];
    const REP_STOSB: &[u8] = &[0xf3, 0xaa, 0xf4];

    fn pattern(length: usize) -> Vec<u8> {
        (0..length).map(|k| (k * 7 + (k >> 8)) as u8).collect()
    }

    #[test]
    fn rep_movsb_copies_many_slices_and_pages_exactly() {
        let _ = SCRATCH;
        for length in [1usize, 4095, 4096, 4097, 32768, 32769, 100_003, 262_144] {
            let mut h = Harness::new(REP_MOVSB);
            let data = pattern(length);
            h.write(SOURCE, &data);
            h.write(DESTINATION - 8, &[0xee; 8]);
            h.write(DESTINATION + length as u64, &[0xee; 8]);
            h.cpu.gpr[RSI] = SOURCE;
            h.cpu.gpr[RDI] = DESTINATION;
            h.cpu.gpr[RCX] = length as u64;
            assert_eq!(h.run(10_000_000), Stop::Halted, "length {length}");
            assert_eq!(h.read(DESTINATION, length), data, "length {length}");
            assert_eq!(h.read(DESTINATION - 8, 8), vec![0xee; 8], "no write below");
            assert_eq!(
                h.read(DESTINATION + length as u64, 8),
                vec![0xee; 8],
                "no write above"
            );
            assert_eq!(h.cpu.gpr[RCX], 0);
            assert_eq!(h.cpu.gpr[RSI], SOURCE + length as u64);
            assert_eq!(h.cpu.gpr[RDI], DESTINATION + length as u64);
            assert_eq!(h.cpu.rip, CODE_BASE + 3, "stopped after the hlt");
        }
    }

    #[test]
    fn rep_stosb_fills_many_slices_and_stops_at_the_slice_boundary_with_rip_on_the_instruction() {
        let length = 3 * REP_BYTES_PER_SLICE as usize + 17;
        let mut h = Harness::new(REP_STOSB);
        h.cpu.gpr[RAX] = 0x5a;
        h.cpu.gpr[RDI] = DESTINATION;
        h.cpu.gpr[RCX] = length as u64;
        // one instruction step: returns after a slice with rip unchanged and rcx partly consumed
        h.cpu.icount += 1;
        h.cpu.step().unwrap();
        assert_eq!(
            h.cpu.rip, CODE_BASE,
            "the instruction resumes where it left off"
        );
        assert_eq!(h.cpu.gpr[RCX], length as u64 - REP_BYTES_PER_SLICE);
        // each slice charges its bytes to the instruction counter, hence the large budget
        assert_eq!(h.run(10_000_000), Stop::Halted);
        assert_eq!(h.read(DESTINATION, length), vec![0x5a; length]);
        assert_eq!(h.read(DESTINATION + length as u64, 1), vec![0]);
    }
}
