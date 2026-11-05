//! `extern "C"` entry points that generated code calls back into, plus the exit-code encoding
//! shared with the emitters (`emit.rs`, `emit_x64.rs`) and the prologues (`arch_a64.rs`,
//! `arch_x64.rs`). The addresses are handed to the emitters through `Addrs` in `arch::addrs`.
//! Argument order and count are hard-coded in the call sequences of `emit.rs` (`helper`,
//! `emit_slows`) and `emit_x64.rs`: changing a signature here means updating those.

#[cfg(freestanding)]
use crate::prelude::*;

use crate::cpu::Cpu;
use crate::cpu::decode::Insn;

/// Helper return value / low byte of the block exit code: nothing special happened.
pub(crate) const STATUS_OK: u64 = 0;
/// The helper stored a guest exception in `Cpu::pending`; the faulting instruction did not
/// retire. `Cpu::step_block` takes the exception out of `pending`.
pub(crate) const STATUS_EXCEPTION: u64 = 1;
/// The instruction retired but wrote to a page that held translated code (`Cpu::code_written`
/// set `block_exit_requested`), so the rest of the block may be stale and must not run.
/// Returned by `helper_exec` only; the `exit_fail` stub counts such an instruction as retired.
pub(crate) const STATUS_EXIT_REQUESTED: u64 = 2;

/// Block exit code layout, produced by the exit stubs in the prologue and decoded by
/// `Cpu::step_block`: `(retired_insns << EXIT_COUNT_SHIFT) | status`. `emit.rs`/`emit_x64.rs`
/// build the success form with `mov x0/rax, n << 8` and jump to `exit_ok`; the `exit_fail` stub
/// takes the helper status in x0/rax and the failing instruction's block index k in x1/rcx and
/// retires k instructions, or k + 1 for `STATUS_EXIT_REQUESTED`. The stubs hard-code 2 and 8.
pub(crate) const EXIT_COUNT_SHIFT: u32 = 8;
pub(crate) const EXIT_STATUS_MASK: u64 = (1 << EXIT_COUNT_SHIFT) - 1;

/// `Insn::op` for bytes that failed to decode, produced by `Jit::translate`. Executes as #UD.
pub(crate) const OP_INVALID: u16 = 0xffff;

/// aarch64 load result. Two `u64`s are returned in x0/x1 (AAPCS64), so `emit.rs` reads the value
/// from x0 and the status from x1.
#[cfg(target_arch = "aarch64")]
#[repr(C)]
pub(crate) struct LoadResult {
    pub val: u64,
    pub status: u64,
}

/// Runs one instruction through the interpreter. `insn` points into the owning `Block`.
/// `rip` is set first because `Cpu::exec` computes the next rip from `cpu.rip`; on success exec
/// leaves `cpu.rip` at the next instruction (or the branch target), so the caller stores nothing.
pub(crate) extern "C" fn helper_exec(cpu: &mut Cpu, insn: &Insn, rip: u64) -> u64 {
    cpu.rip = rip;
    if let Some(histogram) = cpu.helper_histogram.as_mut() {
        *histogram.entry(insn.histogram_key()).or_default() += 1;
    }
    match cpu.exec(insn) {
        Ok(()) => {
            if cpu.block_exit_requested {
                STATUS_EXIT_REQUESTED
            } else {
                STATUS_OK
            }
        }
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// `helper_exec` as the aarch64 blocks call it: `pair` is the `[*const Insn, page offset]`
/// constant pair the block keeps next to its code, so the call site is one `adr` instead of two
/// 64-bit immediates. The block is position independent, so the instruction's `rip` is the page
/// offset on top of `page_base` (`emit::PAGE_BASE`).
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_exec_pair(cpu: &mut Cpu, pair: &[u64; 2], page_base: u64) -> u64 {
    // SAFETY: the block that made the pair keeps its `Insn` slice alive (`Block::_insns`).
    let insn = unsafe { &*(pair[0] as *const Insn) };
    helper_exec(cpu, insn, page_base.wrapping_add(pair[1]))
}

/// Finishes an inlined `rol`/`ror`: they replace CF and OF and leave SF/ZF/PF/AF as the
/// instructions before them made them (`Cpu::shift_rotate`), which only the interpreter can
/// resolve from the lazy flags. `packed` has CF in bit 0 and OF in bit 1.
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_set_rotate_flags(cpu: &mut Cpu, packed: u64) {
    merge_rotate_flags(cpu, packed);
}

/// `helper_set_rotate_flags` for the WebAssembly blocks, in the `(cpu, u64) -> u64` shape of
/// `helper_cond` so it is called through a function type every block module already declares
/// (`wasm_encode::COND_TYPE`). Always returns 0.
#[cfg(target_arch = "wasm32")]
pub(crate) extern "C" fn helper_set_rotate_flags(cpu: &mut Cpu, packed: u64) -> u64 {
    merge_rotate_flags(cpu, packed);
    0
}

#[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
fn merge_rotate_flags(cpu: &mut Cpu, packed: u64) {
    use crate::cpu::flags::{AF, CF, OF, PF, SF, ZF};
    let mut flags = cpu.eflags_full() & (PF | AF | ZF | SF);
    if packed & 1 != 0 {
        flags |= CF;
    }
    if packed & 2 != 0 {
        flags |= OF;
    }
    cpu.set_eflags_arith(flags);
}

/// Runs a slow-path memory access with `Cpu::slow_access_insn` set by the thunk, and clears it
/// afterwards so no later clock read sees it (`Cpu::now_ns`).
fn slow_access<T>(cpu: &mut Cpu, access: impl FnOnce(&mut Cpu) -> T) -> T {
    let result = access(cpu);
    cpu.slow_access_insn = 0;
    result
}

/// Slow-path load, called when the inline TLB probe misses. Generated code does not keep
/// `cpu.rip` current, so the faulting instruction's `rip` is passed in and stored here.
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_load(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    rip: u64,
) -> LoadResult {
    cpu.rip = rip;
    match slow_access(cpu, |cpu| cpu.read(linear_addr, size as u8)) {
        Ok(value) => LoadResult {
            val: value,
            status: STATUS_OK,
        },
        Err(exception) => {
            cpu.pending = Some(exception);
            LoadResult {
                val: 0,
                status: STATUS_EXCEPTION,
            }
        }
    }
}

/// `helper_load` for the implicit stack accesses of push/pop/call/ret: a non-canonical address is
/// #SS(0), not #GP(0) (`Cpu::read_stack`).
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_load_stack(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    rip: u64,
) -> LoadResult {
    cpu.rip = rip;
    match slow_access(cpu, |cpu| cpu.read_stack(linear_addr, size as u8)) {
        Ok(value) => LoadResult {
            val: value,
            status: STATUS_OK,
        },
        Err(exception) => {
            cpu.pending = Some(exception);
            LoadResult {
                val: 0,
                status: STATUS_EXCEPTION,
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_store_stack(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
    rip: u64,
) -> u64 {
    cpu.rip = rip;
    match slow_access(cpu, |cpu| cpu.write_stack(linear_addr, size as u8, value)) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// Slow-path store; returns `STATUS_OK` or `STATUS_EXCEPTION`, never `STATUS_EXIT_REQUESTED`: the
/// emitters poll `block_exit_requested` themselves (`smc_check`) after an inlined instruction
/// that stores, including when the store went through this helper.
#[cfg(target_arch = "aarch64")]
pub(crate) extern "C" fn helper_store(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
    rip: u64,
) -> u64 {
    cpu.rip = rip;
    match slow_access(cpu, |cpu| cpu.write(linear_addr, size as u8, value)) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

// x86-64 variants: at most four arguments (Win64 passes more on the stack) and no 16-byte
// struct return (Win64 returns it through a hidden pointer). The JIT stores `cpu.rip` itself.
// The WebAssembly backend uses them too: a wasm function returns one value, so the load result
// comes back through memory there as well.
/// `out` points at the 8-byte result slot in the block's stack frame (`OUT` in `emit_x64.rs`), or
/// at the slot the wasm JIT keeps for it (`wasm::SharedJit::out`).
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(crate) extern "C" fn helper_load_via_out(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    out: *mut u64,
) -> u64 {
    match slow_access(cpu, |cpu| cpu.read(linear_addr, size as u8)) {
        Ok(value) => {
            unsafe { *out = value };
            STATUS_OK
        }
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// Stack flavor of `helper_load_via_out`; see `helper_load_stack`.
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(crate) extern "C" fn helper_load_stack_via_out(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    out: *mut u64,
) -> u64 {
    match slow_access(cpu, |cpu| cpu.read_stack(linear_addr, size as u8)) {
        Ok(value) => {
            unsafe { *out = value };
            STATUS_OK
        }
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(crate) extern "C" fn helper_store_stack_x64(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
) -> u64 {
    match slow_access(cpu, |cpu| cpu.write_stack(linear_addr, size as u8, value)) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(crate) extern "C" fn helper_store_x64(
    cpu: &mut Cpu,
    linear_addr: u64,
    size: u64,
    value: u64,
) -> u64 {
    match slow_access(cpu, |cpu| cpu.write(linear_addr, size as u8, value)) {
        Ok(()) => STATUS_OK,
        Err(exception) => {
            cpu.pending = Some(exception);
            STATUS_EXCEPTION
        }
    }
}

/// Evaluates condition code `cc` (low four bits of the Jcc/SETcc/CMOVcc opcode) against the
/// guest flags. Used when the emitter cannot test host flags directly.
pub(crate) extern "C" fn helper_cond(cpu: &mut Cpu, cc: u64) -> u64 {
    cpu.condition_holds(cc as u8) as u64
}

/// The instructions an emitter may lower with its own exit sequence (`try_inline` returns true
/// and the block stops): jcc rel8/rel32, jmp rel8/rel32, call rel32, ret, and near indirect
/// call/jmp (`ff /2`, `ff /4`; a form the emitter does not lower goes to the interpreter helper,
/// which also ends the block). All of them are also `ends_block`. Keep in step with the terminal
/// arms of `try_inline`.
/// jcc rel8 and rel32: the one kind of block end the WebAssembly backend can decode past
/// (`blocks::decode_block`), making the taken branch a side exit.
pub(crate) fn is_conditional_branch(insn: &Insn) -> bool {
    matches!(insn.op, 0x70..=0x7f | 0x180..=0x18f)
}

pub(crate) fn is_inline_terminator(insn: &Insn) -> bool {
    matches!(
        insn.op,
        0x70..=0x7f | 0x180..=0x18f | 0xe9 | 0xeb | 0xe8 | 0xc3
    ) || (insn.op == 0xff && matches!(insn.reg & 7, 2 | 4))
}

/// Register forms of `0f 01` (group 7) that the block compilers continue past.
pub(crate) const MODRM_SWAPGS: u8 = 0xf8;
pub(crate) const MODRM_RDTSCP: u8 = 0xf9;
pub(crate) const MODRM_CLAC: u8 = 0xca;
pub(crate) const MODRM_STAC: u8 = 0xcb;

/// True if the block must end after `insn`. The set is an allow-list of what the block compilers
/// cannot continue past: control transfers; string and I/O instructions (they loop on `rep` or
/// have device side effects); and instructions that change interrupt delivery, segmentation,
/// privilege, paging or other control state that the run loop must observe before the next
/// instruction. Opcodes at or above 0x100 are `0f xx` (`decode` puts the opcode map in bits 8+).
pub(crate) fn ends_block(insn: &Insn) -> bool {
    match insn.op {
        // jcc rel8, loopnz/loopz/loop/jcxz, call rel32, jmp rel32/rel8, ret near/far (with and
        // without imm16), int3, int imm8, iret, hlt
        0x70..=0x7f
        | 0xe0..=0xe3
        | 0xe8
        | 0xe9
        | 0xeb
        | 0xc2
        | 0xc3
        | 0xca
        | 0xcb
        | 0xcc
        | 0xcd
        | 0xcf
        | 0xf4 => true,
        // ins/outs, movs/cmps, stos/lods/scas, in/out imm8, in/out dx, sti, popf, mov sreg.
        // `cli` is not here: interrupts are only taken between blocks, so the rest of the block
        // runs exactly as it would with IF already clear. `sti` must stay: its shadow covers one
        // instruction, after which a pending interrupt has to be taken.
        0x6c..=0x6f
        | 0xa4..=0xa7
        | 0xaa..=0xaf
        | 0xe4..=0xe7
        | 0xec..=0xef
        | 0xfb
        | 0x9d
        | 0x8e => true,
        // swapgs, rdtscp, clac and stac change nothing a block caches: generated code reads the
        // GS base from `Cpu` on every access, and the soft TLB never caches a supervisor access
        // to a user page while SMAP is on (`Cpu::translate`), so AC is looked at afresh. They
        // sit on the syscall entry and exit paths, where ending a block at each one costs a
        // trip through the run loop.
        0x101
            if matches!(
                insn.modrm,
                MODRM_SWAPGS | MODRM_RDTSCP | MODRM_CLAC | MODRM_STAC
            ) =>
        {
            false
        }
        // group 5: call/jmp indirect, near and far
        0xff => matches!(insn.reg & 7, 2..=5),
        // syscall, clts, sysret, ud2, jcc rel32, pop fs, pop gs, mov to/from cr/dr, wrmsr,
        // groups 6 and 7 (lldt/ltr/lgdt/lidt/lmsw/invlpg/...)
        0x105..=0x107
        | 0x10b
        | 0x180..=0x18f
        | 0x1a1
        | 0x1a9
        | 0x120..=0x123
        | 0x130
        | 0x100
        | 0x101
        | OP_INVALID => true,
        _ => false,
    }
}

/// For each instruction of a block, whether the lazy flags it records must really be stored.
/// A record is dead when a later instruction overwrites all the arithmetic flags before anything
/// can look at them, so only runs of register-only ALU instructions are ever skipped. What can
/// look: a flag reader (jcc, setcc, cmov, adc, inc, a helper-executed instruction, ...), anything
/// that can fault (an exception pushes the flags), and the end of the block (the successor may
/// read them). Only the register-only flag writers listed in `writes_all_flags` get `false`;
/// the answer for every other instruction is `true`, as is the answer for a writer whose
/// flags are still wanted.
///
/// The native generators ask this; the WebAssembly one defers every record it can instead
/// (`emit_wasm::Gen::pending_record`).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn flag_record_needed(insns: &[Insn]) -> Vec<bool> {
    let mut needed = vec![true; insns.len()];
    let mut live = true;
    for (index, insn) in insns.iter().enumerate().rev() {
        if writes_all_flags(insn) {
            needed[index] = live;
            live = false;
        } else if !leaves_flags_alone(insn) {
            live = true;
        }
    }
    needed
}

/// Whether the flags instruction `index` would leave are overwritten before anything can read
/// them: what follows it in the block, up to a register-only writer of every flag, neither reads
/// flags nor can fault (the rule of `flag_record_needed`, asked about one instruction). The
/// WebAssembly generator then lowers instructions whose flags it cannot cheaply compute (a 64-bit
/// `imul`, rotates) without computing them.
#[cfg(target_arch = "wasm32")]
pub(crate) fn flags_dead_after(insns: &[Insn], index: usize) -> bool {
    for insn in &insns[index + 1..] {
        if writes_all_flags(insn) {
            return true;
        }
        if !leaves_flags_alone(insn) {
            return false;
        }
    }
    false
}

const ALU_ADC: u16 = 2;
const ALU_SBB: u16 = 3;

/// Register-only instructions that set every arithmetic flag from their result alone, reading
/// none: add/or/and/sub/xor/cmp and test with a register or immediate source and a register
/// destination. adc and sbb read CF; every memory form can fault.
fn writes_all_flags(insn: &Insn) -> bool {
    if insn.lock {
        return false;
    }
    let register_form = !insn.has_modrm || insn.md == 3;
    match insn.op {
        // add/or/adc/sbb/and/sub/xor/cmp in the forms r/m,r (0, 1), r,r/m (2, 3), al,imm8 (4)
        // and eax,imm32 (5)
        0x00..=0x3f if insn.op & 7 < 6 => {
            register_form && !matches!((insn.op >> 3) & 7, ALU_ADC | ALU_SBB)
        }
        // group 1 with an immediate: /2 and /3 are adc and sbb
        0x80 | 0x81 | 0x83 => register_form && !matches!(insn.reg & 7, 2 | 3),
        // test r/m,r and test al/eax,imm
        0x84 | 0x85 | 0xa8 | 0xa9 => register_form,
        // group 3: /0 and /1 are test r/m,imm
        0xf6 | 0xf7 => register_form && insn.reg & 7 <= 1,
        _ => false,
    }
}

/// Register-only instructions that neither read nor write flags and cannot fault.
fn leaves_flags_alone(insn: &Insn) -> bool {
    if insn.lock || insn.rep != 0 {
        return false;
    }
    match insn.op {
        // mov r/m,r and r,r/m (register forms), mov r,imm, mov r/m,imm (register form)
        0x88..=0x8b => insn.md == 3,
        0xb0..=0xbf => true,
        0xc6 | 0xc7 => insn.md == 3 && insn.reg & 7 == 0,
        // lea only computes an address (its register form is #UD)
        0x8d => insn.md != 3,
        // movsxd, movzx and movsx from a register
        0x63 | 0x1b6 | 0x1b7 | 0x1be | 0x1bf => insn.md == 3,
        // cwde/cdqe and cdq/cqo
        0x98 | 0x99 => true,
        _ => false,
    }
}

/// The 128-bit whole-register moves the generators lower inline: movups/movupd, movaps/movapd and
/// movdqu/movdqa, legacy-encoded, register or memory form.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VectorMove {
    /// The destination is the xmm register `reg`; otherwise it is `rm` (register) or memory.
    pub load: bool,
    /// A memory operand must be 16-byte aligned (#GP(0) otherwise): movaps, movapd, movdqa.
    pub aligned: bool,
}

/// `Some` for the forms `VectorMove` describes. The scalar forms (`f3`/`f2` movss/movsd, which
/// merge into the low lane), movlps/movhps, movntdq, MMX `movq` and every VEX/EVEX form are not
/// included: `Insn::op` carries a flag for the VEX and EVEX ones, so they never match.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
pub(crate) fn vector_move(insn: &Insn) -> Option<VectorMove> {
    const OPCODE_MAP_0F: u16 = 1 << 8;
    let opcode = insn
        .op
        .checked_sub(OPCODE_MAP_0F)
        .filter(|&byte| byte <= 0xff)?;
    // Same precedence as `Cpu::sse`: an f3/f2 prefix wins over 66.
    let packed = insn.rep == 0;
    let (load, aligned) = match opcode {
        // movups/movupd (no prefix or 66)
        0x10 | 0x11 if packed => (opcode == 0x10, false),
        // movaps/movapd
        0x28 | 0x29 if packed => (opcode == 0x28, true),
        // movdqa (66) and movdqu (f3); without a prefix 0f 6f/7f is MMX movq
        0x6f | 0x7f if insn.rep == 1 || (packed && insn.p66) => (opcode == 0x6f, insn.rep == 0),
        _ => return None,
    };
    Some(VectorMove { load, aligned })
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum VectorOp {
    And,
    /// `dst = !dst & src`.
    AndNot,
    Or,
    Xor,
    /// Lanes become all ones where equal, zero elsewhere.
    CompareEqual,
    MinUnsigned8,
    MaxUnsigned8,
    Add,
    Subtract,
    /// pmovmskb: the sign bit of each byte of an xmm register into the low 16 bits of a general
    /// register. Register source only; it is the one form whose destination is not an xmm register.
    MoveMask,
}

/// A legacy-encoded `66 0f xx` integer vector instruction that works lane by lane on a full xmm
/// register (or `MoveMask`). The `Cpu::sse` semantics are the definition; the same instruction
/// without the 66 prefix is MMX and with f3/f2 is a different instruction, so neither matches.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VectorAlu {
    pub op: VectorOp,
    /// Lane width as log2 of the byte count: 0 = bytes, 1 = words, 2 = dwords, 3 = qwords.
    pub lane_log2: u32,
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
pub(crate) fn vector_alu(insn: &Insn) -> Option<VectorAlu> {
    const OPCODE_MAP_0F: u16 = 1 << 8;
    if insn.rep != 0 || !insn.p66 {
        return None;
    }
    let opcode = insn
        .op
        .checked_sub(OPCODE_MAP_0F)
        .filter(|&byte| byte <= 0xff)?;
    let (op, lane_log2) = match opcode {
        0xdb => (VectorOp::And, 0),
        0xdf => (VectorOp::AndNot, 0),
        0xeb => (VectorOp::Or, 0),
        0xef => (VectorOp::Xor, 0),
        // pcmpeqb, pcmpeqw, pcmpeqd
        0x74..=0x76 => (VectorOp::CompareEqual, (opcode - 0x74) as u32),
        0xda => (VectorOp::MinUnsigned8, 0),
        0xde => (VectorOp::MaxUnsigned8, 0),
        // paddb, paddw, paddd; paddq
        0xfc..=0xfe => (VectorOp::Add, (opcode - 0xfc) as u32),
        0xd4 => (VectorOp::Add, 3),
        // psubb, psubw, psubd; psubq
        0xf8..=0xfa => (VectorOp::Subtract, (opcode - 0xf8) as u32),
        0xfb => (VectorOp::Subtract, 3),
        0xd7 => (VectorOp::MoveMask, 0),
        _ => return None,
    };
    Some(VectorAlu { op, lane_log2 })
}

/// A legacy-encoded instruction that computes a whole xmm register from the destination, a
/// register or 16-byte aligned memory source and, for `Sha256Rounds2`, the implicit xmm0: the SHA
/// extensions plus pshufd, palignr and pshufb, which the SHA-256 loops of firmware and kernels
/// interleave with them. The generators load the source and call `helper_whole_vector` (or emit
/// a host instruction where one computes exactly the same thing), so these never pay for the
/// interpreter's decode and dispatch. `Cpu::sse_ext` and `Cpu::sse` are the definition.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub(crate) enum WholeVectorOp {
    /// sha1rnds4; the immediate's low two bits pick the round function.
    Sha1Rounds4,
    Sha1NextE,
    Sha1Message1,
    Sha1Message2,
    /// sha256rnds2: also reads xmm0.
    Sha256Rounds2,
    Sha256Message1,
    Sha256Message2,
    /// pshufd (66 prefix): each result dword picked by two bits of the immediate.
    ShuffleDwords,
    /// palignr (66 prefix): `dst:src` shifted right by the immediate in bytes.
    AlignRight,
    /// pshufb (66 prefix): each result byte is the destination byte the source byte selects, or
    /// zero where the source byte has its top bit set.
    ShuffleBytes,
}

impl WholeVectorOp {
    /// Indexed by the discriminant, which is how `helper_whole_vector` gets the operation back.
    const ALL: [WholeVectorOp; 10] = [
        WholeVectorOp::Sha1Rounds4,
        WholeVectorOp::Sha1NextE,
        WholeVectorOp::Sha1Message1,
        WholeVectorOp::Sha1Message2,
        WholeVectorOp::Sha256Rounds2,
        WholeVectorOp::Sha256Message1,
        WholeVectorOp::Sha256Message2,
        WholeVectorOp::ShuffleDwords,
        WholeVectorOp::AlignRight,
        WholeVectorOp::ShuffleBytes,
    ];
}

/// An instruction `whole_vector` accepts: the operation and its immediate (0 where it has none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WholeVector {
    pub op: WholeVectorOp,
    pub imm: u8,
}

/// Bit positions of `helper_whole_vector`'s `packed` argument: the operation in bits 7:0, the
/// immediate in 15:8, the destination xmm register in 19:16.
const PACKED_OPERATION_MASK: u64 = 0xff;
const PACKED_IMM_SHIFT: u32 = 8;
const PACKED_DESTINATION_SHIFT: u32 = 16;
const XMM_INDEX_MASK: u64 = 0xf;

impl WholeVector {
    /// `helper_whole_vector`'s `packed` argument for this instruction writing xmm `destination`.
    pub fn packed(self, destination: u8) -> u64 {
        self.op as u64
            | (self.imm as u64) << PACKED_IMM_SHIFT
            | (destination as u64 & XMM_INDEX_MASK) << PACKED_DESTINATION_SHIFT
    }

    /// The new destination, as the interpreter computes it.
    pub fn apply(self, destination: u128, source: u128, xmm0: u128) -> u128 {
        use crate::cpu::{
            palignr, pshufb, sha1_four_rounds, sha1_message_1, sha1_message_2, sha1_next_e,
            sha256_message_1, sha256_message_2, sha256_two_rounds, shuffle_dwords,
        };
        match self.op {
            WholeVectorOp::Sha1Rounds4 => sha1_four_rounds(destination, source, self.imm),
            WholeVectorOp::Sha1NextE => sha1_next_e(destination, source),
            WholeVectorOp::Sha1Message1 => sha1_message_1(destination, source),
            WholeVectorOp::Sha1Message2 => sha1_message_2(destination, source),
            WholeVectorOp::Sha256Rounds2 => sha256_two_rounds(destination, source, xmm0),
            WholeVectorOp::Sha256Message1 => sha256_message_1(destination, source),
            WholeVectorOp::Sha256Message2 => sha256_message_2(destination, source),
            WholeVectorOp::ShuffleDwords => shuffle_dwords(source, self.imm),
            WholeVectorOp::AlignRight => palignr(destination, source, self.imm),
            WholeVectorOp::ShuffleBytes => pshufb(destination, source),
        }
    }
}

/// `Some` for the instructions `WholeVectorOp` lists, legacy-encoded with exactly the prefix the
/// interpreter decodes them with: none for the SHA extensions (with 66 they are #UD), 66 for
/// pshufd, palignr and pshufb (unprefixed they are the MMX pshufw, palignr and pshufb, with f3/f2
/// pshufhw, pshuflw or #UD). VEX and EVEX forms carry a flag in `Insn::op`, so they never match.
pub(crate) fn whole_vector(insn: &Insn) -> Option<WholeVector> {
    const OPCODE_MAP_0F: u16 = 1 << 8;
    const OPCODE_MAP_0F38: u16 = 2 << 8;
    const OPCODE_MAP_0F3A: u16 = 3 << 8;
    const OPCODE_MASK: u16 = 0xff;
    const SHA1_ROUND_FUNCTION_MASK: i64 = 3;
    if insn.rep != 0 {
        return None;
    }
    let imm = insn.imm as u8;
    // The map is the high byte; a VEX or EVEX flag there makes it match none of these.
    let (map, opcode) = (insn.op & !OPCODE_MASK, (insn.op & OPCODE_MASK) as u8);
    let (op, imm) = match (map, opcode, insn.p66) {
        (OPCODE_MAP_0F3A, 0xcc, false) => (
            WholeVectorOp::Sha1Rounds4,
            (insn.imm & SHA1_ROUND_FUNCTION_MASK) as u8,
        ),
        (OPCODE_MAP_0F38, 0xc8, false) => (WholeVectorOp::Sha1NextE, 0),
        (OPCODE_MAP_0F38, 0xc9, false) => (WholeVectorOp::Sha1Message1, 0),
        (OPCODE_MAP_0F38, 0xca, false) => (WholeVectorOp::Sha1Message2, 0),
        (OPCODE_MAP_0F38, 0xcb, false) => (WholeVectorOp::Sha256Rounds2, 0),
        (OPCODE_MAP_0F38, 0xcc, false) => (WholeVectorOp::Sha256Message1, 0),
        (OPCODE_MAP_0F38, 0xcd, false) => (WholeVectorOp::Sha256Message2, 0),
        (OPCODE_MAP_0F, 0x70, true) => (WholeVectorOp::ShuffleDwords, imm),
        (OPCODE_MAP_0F3A, 0x0f, true) => (WholeVectorOp::AlignRight, imm),
        (OPCODE_MAP_0F38, 0x00, true) => (WholeVectorOp::ShuffleBytes, 0),
        _ => return None,
    };
    Some(WholeVector { op, imm })
}

/// The byte permutation pshufd with immediate `order` applies, as a byte-shuffle index vector
/// (`tbl` on aarch64): result byte `4k + j` is source byte `4 * order[2k+1:2k] + j`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn shuffle_dwords_byte_indices(order: u8) -> u128 {
    const DWORD_BYTES: u32 = 4;
    const SELECTOR_MASK: u8 = 3;
    let mut indices = 0u128;
    for byte in 0..16u32 {
        let dword = byte / DWORD_BYTES;
        let selected = ((order >> (2 * dword)) & SELECTOR_MASK) as u32;
        let index = selected * DWORD_BYTES + byte % DWORD_BYTES;
        indices |= (index as u128) << (8 * byte);
    }
    indices
}

/// Runs a `WholeVector` instruction for generated code that has already passed the SSE gate and
/// loaded the source (`source_low`, `source_high`) from its register or from guest memory, so
/// nothing here can fault. `packed` is `WholeVector::packed`. Returns 0, so the WebAssembly blocks
/// can call it through the store helpers' function type (`wasm_encode::STORE_TYPE`).
pub(crate) extern "C" fn helper_whole_vector(
    cpu: &mut Cpu,
    packed: u64,
    source_low: u64,
    source_high: u64,
) -> u64 {
    // The generators only pack valid operations; an unknown one leaves the register alone rather
    // than panicking.
    let Some(&op) = WholeVectorOp::ALL.get((packed & PACKED_OPERATION_MASK) as usize) else {
        return STATUS_OK;
    };
    let instruction = WholeVector {
        op,
        imm: (packed >> PACKED_IMM_SHIFT) as u8,
    };
    let destination = ((packed >> PACKED_DESTINATION_SHIFT) & XMM_INDEX_MASK) as usize;
    let source = (source_high as u128) << 64 | source_low as u128;
    cpu.xmm[destination] = instruction.apply(cpu.xmm[destination], source, cpu.xmm[0]);
    STATUS_OK
}

/// What an inline AVX-512 instruction does. Whole-vector data movement and integer logic and
/// arithmetic (the shapes of glibc's EVEX string and copy loops, with merge or zero masking
/// except for stores to memory), byte compares into an opmask register, and `kmov` of an opmask
/// register to a general register are lowered; broadcast, rounding and the rest stay in the
/// interpreter, which also raises every #UD, so a form that could be one is not accepted here.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EvexKind {
    /// vmovdqu8/16/32/64, vmovdqa32/64, vmovups/vmovupd, vmovaps/vmovapd between a register and
    /// a register or memory. The aligned forms need an operand aligned to the vector length.
    Move { load: bool, aligned: bool },
    /// `reg = vvvv op rm`, lane by lane (`VectorOp` as for the legacy forms, without `MoveMask`).
    Alu { op: VectorOp, lane_log2: u32 },
    /// `k[reg] = (vvvv == rm)` per byte, ANDed with the write mask when there is one.
    CompareEqualBytes,
    /// `k[reg] = ((vvvv & rm) != 0)` per byte (vptestmb), ANDed with the write mask.
    TestBytes,
    /// `k[reg] = ((vvvv & rm) == 0)` per byte (vptestnmb), ANDed with the write mask.
    TestNotBytes,
    /// vpcmpub/vpcmpb `k[reg] = predicate(vvvv, rm)` per byte, ANDed with the write mask. The
    /// predicate is the low three bits of the immediate (`PREDICATE_*` order of the SDM table).
    CompareBytes { signed: bool, predicate: u8 },
    /// VEX `kmovd`/`kmovq r, k`: the opmask register `rm` into general register `reg`.
    MaskToGeneral { wide: bool },
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EvexForm {
    pub kind: EvexKind,
    /// Vector length in bytes: 16, 32 or 64.
    pub bytes: u32,
    /// log2 of the element size the write mask applies to (0 = bytes .. 3 = quadwords).
    pub element_log2: u32,
}

/// `Some` for the instructions `EvexKind` lists, with the encodings the interpreter accepts as
/// valid (see `Cpu::exec_evex`): no broadcast or rounding (EVEX.b), no reserved length, and no
/// EVEX.z without a write mask.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
pub(crate) fn evex_form(insn: &Insn) -> Option<EvexForm> {
    use crate::cpu::decode::{EVEX_OP_FLAG, VEX_OP_FLAG};
    const MAP_0F: u16 = 1;
    const MAP_0F38: u16 = 2;
    const MAP_0F3A: u16 = 3;
    const OPCODE_KMOV_TO_GENERAL: u16 = 0x93;
    const VECTOR_LENGTH_RESERVED: u8 = 3;
    const OPMASK_REGISTERS: u8 = 8;
    const PREDICATE_MASK: i64 = 7;
    if insn.vex && insn.op == VEX_OP_FLAG | (MAP_0F << 8) | OPCODE_KMOV_TO_GENERAL {
        // F2 0F 93: kmovd (W0) / kmovq (W1) r32/r64, k. Register source only, L0, no vvvv.
        return (insn.rep == 2 && insn.md == 3 && !insn.vex_l && insn.vex_v == 0).then_some(
            EvexForm {
                kind: EvexKind::MaskToGeneral {
                    wide: insn.opsize == 8,
                },
                bytes: 0,
                element_log2: 0,
            },
        );
    }
    if !insn.evex || insn.evex_b || insn.evex_ll == VECTOR_LENGTH_RESERVED {
        return None;
    }
    let bytes = 16u32 << insn.evex_ll;
    let op = insn.op & !EVEX_OP_FLAG;
    let (map, opcode) = (op >> 8, (op & 0xff) as u8);
    let masked = insn.evex_mask != 0;
    // EVEX.z with no write mask: the interpreter ignores it; not worth a lowering.
    let zeroing_ok = masked || !insn.evex_zeroing;
    let wide = insn.opsize == 8;
    // The implied prefix, as `Cpu::exec_evex` reads it: F3 and F2 win over 66.
    let (no_prefix, p66, f3, f2) = (
        insn.rep == 0 && !insn.p66,
        insn.rep == 0 && insn.p66,
        insn.rep == 1,
        insn.rep == 2,
    );
    let alu = |op, lane_log2| Some((EvexKind::Alu { op, lane_log2 }, lane_log2));
    let logic_element = if wide { 3 } else { 2 };
    let (kind, element_log2) = match (map, opcode) {
        // A masked store to memory writes only the selected elements, which the inline path
        // could only do as a read-modify-write of the whole vector: left to the interpreter.
        (MAP_0F, 0x6f | 0x7f)
            if zeroing_ok
                && insn.vex_v == 0
                && (p66 || f3 || f2)
                && !(masked && opcode == 0x7f && insn.md != 3) =>
        {
            let element_log2 = match (f2, wide) {
                (true, false) => 0,
                (true, true) => 1,
                (false, false) => 2,
                (false, true) => 3,
            };
            (
                EvexKind::Move {
                    load: opcode == 0x6f,
                    aligned: p66,
                },
                element_log2,
            )
        }
        (MAP_0F, 0x10 | 0x11 | 0x28 | 0x29)
            if zeroing_ok
                && insn.vex_v == 0
                && (no_prefix || p66)
                && !(masked && matches!(opcode, 0x11 | 0x29) && insn.md != 3) =>
        {
            (
                EvexKind::Move {
                    load: matches!(opcode, 0x10 | 0x28),
                    aligned: opcode >= 0x28,
                },
                if p66 { 3 } else { 2 },
            )
        }
        (MAP_0F, 0xdb | 0xdf | 0xeb | 0xef | 0xda | 0xde | 0xfc..=0xfe | 0xd4 | 0xf8..=0xfb)
            if zeroing_ok && p66 =>
        {
            match opcode {
                0xdb => alu(VectorOp::And, 0).map(|(kind, _)| (kind, logic_element)),
                0xdf => alu(VectorOp::AndNot, 0).map(|(kind, _)| (kind, logic_element)),
                0xeb => alu(VectorOp::Or, 0).map(|(kind, _)| (kind, logic_element)),
                0xef => alu(VectorOp::Xor, 0).map(|(kind, _)| (kind, logic_element)),
                0xda => alu(VectorOp::MinUnsigned8, 0),
                0xde => alu(VectorOp::MaxUnsigned8, 0),
                0xfc..=0xfe => alu(VectorOp::Add, (opcode - 0xfc) as u32),
                0xd4 => alu(VectorOp::Add, 3),
                0xf8..=0xfa => alu(VectorOp::Subtract, (opcode - 0xf8) as u32),
                _ => alu(VectorOp::Subtract, 3),
            }?
        }
        // The destination is an opmask register, so the register field must name one, and EVEX.z
        // is reserved.
        (MAP_0F, 0x74) if p66 && !insn.evex_zeroing && insn.reg < OPMASK_REGISTERS => {
            (EvexKind::CompareEqualBytes, 0)
        }
        (MAP_0F38, 0x26)
            if (p66 || f3) && !wide && !insn.evex_zeroing && insn.reg < OPMASK_REGISTERS =>
        {
            (
                if p66 {
                    EvexKind::TestBytes
                } else {
                    EvexKind::TestNotBytes
                },
                0,
            )
        }
        // vpcmpub (3e) and vpcmpb (3f) with byte elements (W0)
        (MAP_0F3A, 0x3e | 0x3f)
            if p66 && !wide && !insn.evex_zeroing && insn.reg < OPMASK_REGISTERS =>
        {
            (
                EvexKind::CompareBytes {
                    signed: opcode == 0x3f,
                    predicate: (insn.imm & PREDICATE_MASK) as u8,
                },
                0,
            )
        }
        _ => return None,
    };
    Some(EvexForm {
        kind,
        bytes,
        element_log2,
    })
}

/// `OPMASK_EXPANSION[log2][bits]`: eight bytes of a vector register whose `8 >> log2` elements of
/// `1 << log2` bytes are all ones where the matching bit of `bits` is set, zero elsewhere.
/// The generators turn the write mask of an element size into a vector mask half by half.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) static OPMASK_EXPANSION: [[u64; OPMASK_EXPANSION_ENTRIES]; 4] = {
    let mut table = [[0u64; OPMASK_EXPANSION_ENTRIES]; 4];
    let mut log2 = 0;
    while log2 < 4 {
        let element_bytes = 1usize << log2;
        let elements = 8 / element_bytes;
        let element_ones = if element_bytes == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * element_bytes)) - 1
        };
        let mut bits = 0;
        while bits < 1usize << elements {
            let mut value = 0u64;
            let mut element = 0;
            while element < elements {
                if bits >> element & 1 == 1 {
                    value |= element_ones << (8 * element_bytes * element);
                }
                element += 1;
            }
            table[log2][bits] = value;
            bits += 1;
        }
        log2 += 1;
    }
    table
};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) const OPMASK_EXPANSION_ENTRIES: usize = 256;

/// Offset in `Cpu` of 128-bit lane `lane` (0 least significant) of vector register `register`
/// (0..=31). zmm0-15 are split across `xmm`, `ymm_hi` and `zmm_hi`, zmm16-31 sit in `zmm_extra`
/// (`cpu/evex.rs`).
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
))]
pub(crate) fn vector_lane_offset(register: u8, lane: u32) -> usize {
    use core::mem::offset_of;
    const LANE_BYTES: usize = size_of::<u128>();
    const LEGACY_REGISTERS: usize = 16;
    const LANES_PER_REGISTER: usize = 4;
    let (register, lane) = (register as usize, lane as usize);
    if register >= LEGACY_REGISTERS {
        return offset_of!(Cpu, zmm_extra)
            + (register - LEGACY_REGISTERS) * LANES_PER_REGISTER * LANE_BYTES
            + lane * LANE_BYTES;
    }
    match lane {
        0 => offset_of!(Cpu, xmm) + register * LANE_BYTES,
        1 => offset_of!(Cpu, ymm_hi) + register * LANE_BYTES,
        _ => offset_of!(Cpu, zmm_hi) + register * 2 * LANE_BYTES + (lane - 2) * LANE_BYTES,
    }
}

#[cfg(test)]
mod tests {
    //! The block-ending set, the clock-reading set, the exit-code layout and the helper ABI that
    //! generated code relies on.

    use super::*;
    use crate::cpu::decode;
    use crate::cpu::exception::Exception;
    use crate::cpu::flags::{AF, CF, OF, PF, SF, ZF};
    use crate::testkit::*;

    fn insn_with_op(op: u16) -> Insn {
        Insn {
            op,
            ..Default::default()
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn sse_insn(opcode: u8, rep: u8, p66: bool) -> Insn {
        Insn {
            op: 0x100 | opcode as u16,
            rep,
            p66,
            ..Default::default()
        }
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn vector_alu_covers_the_lane_wise_integer_instructions() {
        let alu = |opcode: u8, rep: u8, p66: bool| {
            vector_alu(&sse_insn(opcode, rep, p66)).map(|a| (a.op, a.lane_log2))
        };
        use VectorOp::*;
        for (opcode, expected) in [
            (0xdb, (And, 0)),
            (0xdf, (AndNot, 0)),
            (0xeb, (Or, 0)),
            (0xef, (Xor, 0)),
            (0x74, (CompareEqual, 0)),
            (0x75, (CompareEqual, 1)),
            (0x76, (CompareEqual, 2)),
            (0xda, (MinUnsigned8, 0)),
            (0xde, (MaxUnsigned8, 0)),
            (0xfc, (Add, 0)),
            (0xfd, (Add, 1)),
            (0xfe, (Add, 2)),
            (0xd4, (Add, 3)),
            (0xf8, (Subtract, 0)),
            (0xf9, (Subtract, 1)),
            (0xfa, (Subtract, 2)),
            (0xfb, (Subtract, 3)),
            (0xd7, (MoveMask, 0)),
        ] {
            assert_eq!(alu(opcode, 0, true), Some(expected), "{opcode:#x}");
            // without the 66 prefix these are MMX, with f3/f2 something else
            assert_eq!(alu(opcode, 0, false), None, "{opcode:#x} mmx");
            assert_eq!(alu(opcode, 1, true), None, "{opcode:#x} f3");
            assert_eq!(alu(opcode, 2, true), None, "{opcode:#x} f2");
        }
        // pcmpgtb, punpcklbw, pshufd: not lowered
        for opcode in [0x64, 0x60, 0x70] {
            assert_eq!(alu(opcode, 0, true), None, "{opcode:#x}");
        }
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn vector_move_covers_exactly_the_whole_register_moves() {
        let form = |opcode, rep, p66| vector_move(&sse_insn(opcode, rep, p66));
        let (load, store) = (true, false);
        for (opcode, rep, p66, expected) in [
            (0x10, 0, false, Some((load, false))),
            (0x10, 0, true, Some((load, false))),
            (0x11, 0, true, Some((store, false))),
            (0x28, 0, false, Some((load, true))),
            (0x29, 0, true, Some((store, true))),
            (0x6f, 0, true, Some((load, true))),
            (0x7f, 0, true, Some((store, true))),
            (0x6f, 1, false, Some((load, false))),
            (0x7f, 1, true, Some((store, false))),
            // movss/movsd merge into the low lane; 0f 6f/7f without a prefix is MMX movq
            (0x10, 1, false, None),
            (0x11, 2, false, None),
            (0x6f, 0, false, None),
            (0x7f, 2, false, None),
            // movlps
            (0x12, 0, false, None),
        ] {
            assert_eq!(
                form(opcode, rep, p66).map(|m| (m.load, m.aligned)),
                expected,
                "{opcode:#x} rep {rep} 66 {p66}"
            );
        }
        // a VEX-encoded op number never lands in the 0f map
        assert_eq!(
            vector_move(&insn_with_op(crate::cpu::decode::VEX_OP_FLAG | 0x110)),
            None
        );
        assert_eq!(vector_move(&insn_with_op(0x6f)), None);
    }

    fn group_five(reg: u8) -> Insn {
        Insn {
            op: 0xff,
            reg,
            ..Default::default()
        }
    }

    fn ends(op: u16) -> bool {
        ends_block(&insn_with_op(op))
    }

    #[test]
    fn control_transfers_end_a_block() {
        for op in (0x70..=0x7f).chain([
            0xe0, 0xe1, 0xe2, 0xe3, 0xe8, 0xe9, 0xeb, 0xc2, 0xc3, 0xca, 0xcb, 0xcc, 0xcd, 0xcf,
            0xf4,
        ]) {
            assert!(ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn conditional_near_jumps_end_a_block() {
        for op in 0x180..=0x18f {
            assert!(ends(op), "opcode {op:#x}");
        }
        assert!(!ends(0x17f));
        assert!(!ends(0x190), "setcc is not a jump");
    }

    #[test]
    fn string_and_port_instructions_end_a_block() {
        for op in (0x6c..=0x6f)
            .chain(0xa4..=0xa7)
            .chain(0xaa..=0xaf)
            .chain(0xe4..=0xe7)
            .chain(0xec..=0xef)
        {
            assert!(ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn instructions_that_change_interrupt_or_segment_state_end_a_block() {
        // sti, popf, mov sreg, pop fs, pop gs
        for op in [0xfb, 0x9d, 0x8e, 0x1a1, 0x1a9] {
            assert!(ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn cli_does_not_end_a_block() {
        assert!(!ends(0xfa));
    }

    fn group_seven(modrm: u8) -> Insn {
        Insn {
            op: 0x101,
            modrm,
            md: modrm >> 6,
            reg: (modrm >> 3) & 7,
            rm: modrm & 7,
            ..Default::default()
        }
    }

    #[test]
    fn swapgs_rdtscp_clac_and_stac_do_not_end_a_block() {
        for modrm in [0xf8, 0xf9, 0xca, 0xcb] {
            assert!(!ends_block(&group_seven(modrm)), "0f 01 {modrm:02x}");
        }
    }

    #[test]
    fn the_rest_of_group_seven_ends_a_block() {
        // sgdt [rax], lgdt [rax], invlpg [rax], vmrun, stgi, clgi, monitor, mwait, xsetbv, smsw
        // eax, lmsw ax
        for modrm in [
            0x00, 0x10, 0x38, 0xd8, 0xdc, 0xdd, 0xc8, 0xc9, 0xd1, 0xe0, 0xf0,
        ] {
            assert!(ends_block(&group_seven(modrm)), "0f 01 {modrm:02x}");
        }
    }

    #[test]
    fn system_instructions_end_a_block() {
        // groups 6 and 7, syscall, clts, sysret, ud2, mov to and from cr/dr, wrmsr
        for op in [
            0x100, 0x101, 0x105, 0x106, 0x107, 0x10b, 0x120, 0x121, 0x122, 0x123, 0x130,
        ] {
            assert!(ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn the_invalid_marker_ends_a_block() {
        assert!(ends(OP_INVALID));
    }

    #[test]
    fn indirect_calls_and_jumps_end_a_block_but_inc_dec_and_push_do_not() {
        for reg in 2..=5 {
            assert!(ends_block(&group_five(reg)), "ff /{reg}");
        }
        for reg in [0, 1, 6, 7] {
            assert!(!ends_block(&group_five(reg)), "ff /{reg}");
        }
        // The ModRM.reg field is masked to three bits (REX.R is not part of it).
        assert!(ends_block(&group_five(8 + 2)));
    }

    #[test]
    fn ordinary_instructions_do_not_end_a_block() {
        // add/mov/lea/test/imul, nop, leave, enter, cdq, cmpxchg, rdtsc, rdmsr, cpuid, setcc, cmovcc,
        // movzx, bswap, xchg, push/pop register, lahf/sahf, pushf, hlt-adjacent cmc/clc/stc
        for op in [
            0x00, 0x01, 0x05, 0x31, 0x39, 0x3d, 0x50, 0x58, 0x63, 0x69, 0x6b, 0x80, 0x81, 0x83,
            0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8d, 0x90, 0x98, 0x99, 0x9c, 0x9e,
            0x9f, 0xa8, 0xa9, 0xb0, 0xb8, 0xc0, 0xc1, 0xc6, 0xc7, 0xc8, 0xc9, 0xd0, 0xd1, 0xd2,
            0xd3, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfc, 0xfd, 0xfe, 0x10d, 0x118, 0x131, 0x132,
            0x140, 0x1a2, 0x190, 0x1af, 0x1b6, 0x1be, 0x1c8, 0x1ba,
        ] {
            assert!(!ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn opcodes_just_outside_the_block_ending_ranges_do_not_end_one() {
        // below and above jcc rel8, below the string and port ranges, between the call/jmp opcodes
        // and the prefixes at 0xf0..=0xf3
        for op in [
            0x6b, 0x80, 0xa3, 0xa8, 0xa9, 0xb0, 0xdf, 0xea, 0xf0, 0xf3, 0xf5,
        ] {
            assert!(!ends(op), "opcode {op:#x}");
        }
    }

    #[test]
    fn every_inline_terminator_also_ends_a_block() {
        for op in 0..=0x1ffu16 {
            let insn = insn_with_op(op);
            if is_inline_terminator(&insn) {
                assert!(ends_block(&insn), "opcode {op:#x}");
            }
        }
    }

    #[test]
    fn the_inline_terminators_are_jcc_jmp_call_and_ret() {
        for op in (0x70..=0x7f)
            .chain(0x180..=0x18f)
            .chain([0xe9, 0xeb, 0xe8, 0xc3])
        {
            assert!(is_inline_terminator(&insn_with_op(op)), "opcode {op:#x}");
        }
        for op in [0xe0, 0xe3, 0xc2, 0xcb, 0xf4, 0xff, 0xcc, 0xcd, OP_INVALID] {
            assert!(!is_inline_terminator(&insn_with_op(op)), "opcode {op:#x}");
        }
    }

    #[test]
    fn status_values_are_distinct_and_fit_in_the_status_byte() {
        let statuses = [STATUS_OK, STATUS_EXCEPTION, STATUS_EXIT_REQUESTED];
        assert_eq!(
            statuses[0], 0,
            "success must be zero: generated code tests it with cbz/test"
        );
        for (index, status) in statuses.iter().enumerate() {
            assert!(*status <= EXIT_STATUS_MASK);
            for other in &statuses[index + 1..] {
                assert_ne!(status, other);
            }
        }
        assert_eq!(EXIT_STATUS_MASK, 0xff);
        assert_eq!(
            EXIT_COUNT_SHIFT, 8,
            "the prologue stubs hard-code this shift"
        );
    }

    #[test]
    fn exit_codes_pack_and_unpack_the_retired_count_and_status() {
        for retired in [0u64, 1, 2, 47, 48, 255, 1 << 20] {
            for status in [STATUS_OK, STATUS_EXCEPTION, STATUS_EXIT_REQUESTED] {
                let exit_code = (retired << EXIT_COUNT_SHIFT) | status;
                assert_eq!(exit_code >> EXIT_COUNT_SHIFT, retired);
                assert_eq!(exit_code & EXIT_STATUS_MASK, status);
            }
        }
    }

    // MAX_INSNS is 48; a block's count is stored as `retired << EXIT_COUNT_SHIFT` through a `mov`
    // immediate and must not collide with the status.
    const _: () = assert!((48u64 << EXIT_COUNT_SHIFT) & EXIT_STATUS_MASK == 0);

    fn decoded(bytes: &[u8]) -> Insn {
        decode::decode(bytes).expect("snippet decodes")
    }

    // add rax, rcx
    const ADD_RAX_RCX: [u8; 3] = [0x48, 0x01, 0xc8];
    // ud2
    const UD2: [u8; 2] = [0x0f, 0x0b];
    // mov [rbx], rax
    const MOV_TO_RBX_MEMORY: [u8; 3] = [0x48, 0x89, 0x03];

    #[test]
    fn helper_exec_runs_the_instruction_and_leaves_rip_at_the_next_one() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        cpu.gpr[0] = 40;
        cpu.gpr[1] = 2;
        let insn = decoded(&ADD_RAX_RCX);
        let status = helper_exec(&mut cpu, &insn, 0x1234);
        assert_eq!(status, STATUS_OK);
        assert_eq!(cpu.gpr[0], 42);
        assert_eq!(
            cpu.rip,
            0x1234 + ADD_RAX_RCX.len() as u64,
            "exec derives the next rip from the rip passed in"
        );
    }

    #[test]
    fn helper_exec_stores_an_exception_in_pending_and_reports_the_status() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        let insn = decoded(&UD2);
        let status = helper_exec(&mut cpu, &insn, 0x4000);
        assert_eq!(status, STATUS_EXCEPTION);
        assert!(matches!(
            cpu.pending,
            Some(Exception::Fault { vector: 6, .. })
        ));
        assert_eq!(cpu.rip, 0x4000, "rip names the faulting instruction");
    }

    #[test]
    fn helper_exec_reports_a_requested_block_exit_after_a_successful_instruction() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        cpu.block_exit_requested = true;
        let insn = decoded(&ADD_RAX_RCX);
        assert_eq!(helper_exec(&mut cpu, &insn, 0x1000), STATUS_EXIT_REQUESTED);
    }

    #[test]
    fn helper_exec_prefers_the_exception_over_a_pending_exit_request() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        cpu.block_exit_requested = true;
        let insn = decoded(&UD2);
        assert_eq!(helper_exec(&mut cpu, &insn, 0x1000), STATUS_EXCEPTION);
    }

    #[test]
    fn a_store_through_helper_exec_to_a_code_page_requests_a_block_exit() {
        let _guard = crate::jit::serialize_jit_tests();
        let mut harness = Harness::new(&[0xf4]);
        assert!(harness.enable_jit());
        let cpu = &mut harness.cpu;
        // Mark the page holding the snippet as translated code, as `translate` does.
        cpu.mark_code(CODE_BASE);
        cpu.gpr[3] = CODE_BASE + 0x100;
        let insn = decoded(&MOV_TO_RBX_MEMORY);
        assert_eq!(helper_exec(cpu, &insn, CODE_BASE), STATUS_EXIT_REQUESTED);
    }

    #[test]
    fn helper_cond_evaluates_every_condition_against_the_guest_flags() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        // (flags, conditions that hold): O NO B AE E NE BE A S NS P NP L GE LE G
        let cases: [(u64, [bool; 16]); 4] = [
            (
                0,
                [
                    false, true, false, true, false, true, false, true, false, true, false, true,
                    false, true, false, true,
                ],
            ),
            (
                ZF | PF,
                [
                    false, true, false, true, true, false, true, false, false, true, true, false,
                    false, true, true, false,
                ],
            ),
            (
                CF | SF | AF,
                [
                    false, true, true, false, false, true, true, false, true, false, false, true,
                    true, false, true, false,
                ],
            ),
            (
                OF | SF,
                [
                    true, false, false, true, false, true, false, true, true, false, false, true,
                    false, true, false, true,
                ],
            ),
        ];
        for (flags, expected) in cases {
            cpu.set_eflags_arith(flags);
            for (condition, holds) in expected.iter().enumerate() {
                assert_eq!(
                    helper_cond(&mut cpu, condition as u64),
                    *holds as u64,
                    "flags {flags:#x} condition {condition}"
                );
            }
        }
    }

    #[test]
    fn helper_cond_masks_the_condition_to_the_low_four_bits() {
        let mut cpu = Harness::new(&[0xf4]).cpu;
        cpu.set_eflags_arith(ZF);
        assert_eq!(helper_cond(&mut cpu, 4), 1);
        assert_eq!(
            helper_cond(&mut cpu, 0x74),
            1,
            "the low nibble of the jcc opcode selects the condition"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
    mod x86_64_memory_helpers {
        use super::*;

        #[test]
        fn helper_load_via_out_writes_the_value_through_the_pointer() {
            let mut harness = Harness::new(&[0xf4]);
            harness.write_u64(SCRATCH, 0x1122_3344_5566_7788);
            let mut slot = 0u64;
            let status = helper_load_via_out(&mut harness.cpu, SCRATCH, 8, &mut slot);
            assert_eq!(status, STATUS_OK);
            assert_eq!(slot, 0x1122_3344_5566_7788);
            let status = helper_load_via_out(&mut harness.cpu, SCRATCH + 2, 2, &mut slot);
            assert_eq!(status, STATUS_OK);
            assert_eq!(slot, 0x5566);
        }

        #[test]
        fn helper_load_via_out_reports_a_fault_and_leaves_the_slot_alone() {
            let mut cpu = Harness::new(&[0xf4]).cpu;
            let mut slot = 0xdead;
            let status = helper_load_via_out(&mut cpu, 0x1_0000_0000, 8, &mut slot);
            assert_eq!(status, STATUS_EXCEPTION);
            assert_eq!(slot, 0xdead);
            assert!(matches!(
                cpu.pending,
                Some(Exception::Fault { vector: 14, .. })
            ));
        }

        #[test]
        fn helper_store_x64_writes_memory_and_reports_faults() {
            let mut harness = Harness::new(&[0xf4]);
            assert_eq!(
                helper_store_x64(&mut harness.cpu, SCRATCH, 4, 0xaabb_ccdd),
                STATUS_OK
            );
            assert_eq!(harness.read(SCRATCH, 4), vec![0xdd, 0xcc, 0xbb, 0xaa]);
            assert_eq!(
                helper_store_x64(&mut harness.cpu, 0x1_0000_0000, 8, 1),
                STATUS_EXCEPTION
            );
            assert!(matches!(
                harness.cpu.pending,
                Some(Exception::Fault { vector: 14, .. })
            ));
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod aarch64_memory_helpers {
        use super::*;

        #[test]
        fn helper_load_returns_the_value_and_ok_status() {
            let mut harness = Harness::new(&[0xf4]);
            harness.write_u64(SCRATCH, 0x1122_3344_5566_7788);
            let result = helper_load(&mut harness.cpu, SCRATCH, 8, 0x5000);
            assert_eq!(result.status, STATUS_OK);
            assert_eq!(result.val, 0x1122_3344_5566_7788);
            let result = helper_load(&mut harness.cpu, SCRATCH + 1, 1, 0x5000);
            assert_eq!(result.val, 0x77);
        }

        #[test]
        fn helper_load_reports_the_fault_with_the_instruction_rip() {
            let mut cpu = Harness::new(&[0xf4]).cpu;
            let result = helper_load(&mut cpu, 0x1_0000_0000, 8, 0x5000);
            assert_eq!(result.status, STATUS_EXCEPTION);
            assert_eq!(result.val, 0);
            assert_eq!(
                cpu.rip, 0x5000,
                "generated code does not keep rip current, the helper stores it"
            );
            assert!(matches!(
                cpu.pending,
                Some(Exception::Fault { vector: 14, .. })
            ));
        }

        #[test]
        fn helper_store_writes_memory_and_reports_faults_with_the_rip() {
            let mut harness = Harness::new(&[0xf4]);
            assert_eq!(
                helper_store(&mut harness.cpu, SCRATCH, 2, 0xbeef, 0x5000),
                STATUS_OK
            );
            assert_eq!(harness.read(SCRATCH, 2), vec![0xef, 0xbe]);
            assert_eq!(
                helper_store(&mut harness.cpu, 0x1_0000_0000, 8, 1, 0x6000),
                STATUS_EXCEPTION
            );
            assert_eq!(harness.cpu.rip, 0x6000);
        }

        #[test]
        fn load_result_is_returned_in_two_registers() {
            // x0 = value, x1 = status (AAPCS64 returns a 16-byte struct of two u64 in x0/x1); the
            // emitters read them that way.
            assert_eq!(std::mem::size_of::<LoadResult>(), 16);
            assert_eq!(std::mem::offset_of!(LoadResult, val), 0);
            assert_eq!(std::mem::offset_of!(LoadResult, status), 8);
        }
    }

    mod flag_records {
        use super::*;

        /// Decodes a block from concatenated instruction encodings.
        fn decode_all(code: &[u8]) -> Vec<Insn> {
            let mut insns = Vec::new();
            let mut offset = 0;
            while offset < code.len() {
                let insn = decode::decode(&code[offset..]).expect("valid encoding");
                offset += insn.len as usize;
                insns.push(insn);
            }
            insns
        }

        const ADD_RAX_RCX: [u8; 3] = [0x48, 0x01, 0xc8];
        const ADD_RDX_RSI: [u8; 3] = [0x48, 0x01, 0xf2];
        const CMP_RAX_RCX: [u8; 3] = [0x48, 0x39, 0xc8];
        const MOV_RAX_RCX: [u8; 3] = [0x48, 0x89, 0xc8];
        const LEA_RAX_RBX_8: [u8; 4] = [0x48, 0x8d, 0x43, 0x08];
        const MOVZX_EAX_CL: [u8; 3] = [0x0f, 0xb6, 0xc1];
        const MOV_RAX_FROM_RBX: [u8; 3] = [0x48, 0x8b, 0x03];
        const ADD_TO_RBX_MEMORY: [u8; 3] = [0x48, 0x01, 0x03];
        const ADC_RDX_RSI: [u8; 3] = [0x48, 0x11, 0xf2];
        const INC_RAX: [u8; 3] = [0x48, 0xff, 0xc0];
        const PUSH_RAX: [u8; 1] = [0x50];
        const JNZ_SELF: [u8; 2] = [0x75, 0xfe];
        const HLT: [u8; 1] = [0xf4];

        fn needed(parts: &[&[u8]]) -> Vec<bool> {
            flag_record_needed(&decode_all(&parts.concat()))
        }

        #[test]
        fn a_flag_write_overwritten_before_anything_can_look_is_not_needed() {
            assert_eq!(
                needed(&[&ADD_RAX_RCX, &ADD_RDX_RSI, &HLT]),
                [false, true, true]
            );
            assert_eq!(
                needed(&[&ADD_RAX_RCX, &CMP_RAX_RCX, &JNZ_SELF]),
                [false, true, true]
            );
            // Instructions that neither touch flags nor can fault do not end the window.
            assert_eq!(
                needed(&[
                    &ADD_RAX_RCX,
                    &MOV_RAX_RCX,
                    &LEA_RAX_RBX_8,
                    &MOVZX_EAX_CL,
                    &ADD_RDX_RSI,
                    &HLT
                ]),
                [false, true, true, true, true, true]
            );
        }

        #[test]
        fn the_last_flag_write_of_a_block_is_needed_by_the_next_block() {
            assert_eq!(needed(&[&ADD_RAX_RCX]), [true]);
            assert_eq!(needed(&[&ADD_RAX_RCX, &ADD_RDX_RSI]), [false, true]);
        }

        #[test]
        fn a_flag_write_stays_needed_when_something_can_look_at_it() {
            // A reader.
            assert!(needed(&[&ADD_RAX_RCX, &ADC_RDX_RSI, &HLT])[0]);
            assert!(needed(&[&ADD_RAX_RCX, &INC_RAX, &ADD_RDX_RSI, &HLT])[0]);
            assert!(needed(&[&ADD_RAX_RCX, &JNZ_SELF])[0]);
            // A memory access may fault, and the exception frame holds the flags.
            assert!(needed(&[&ADD_RAX_RCX, &MOV_RAX_FROM_RBX, &ADD_RDX_RSI, &HLT])[0]);
            assert!(needed(&[&ADD_RAX_RCX, &PUSH_RAX, &ADD_RDX_RSI, &HLT])[0]);
            assert!(needed(&[&ADD_RAX_RCX, &ADD_TO_RBX_MEMORY, &HLT])[0]);
        }
    }
}
