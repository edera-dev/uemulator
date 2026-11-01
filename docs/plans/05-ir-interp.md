# IR, TB cache, interpreter

## Status / deviation (M1-M2 built this way)

The interpreter currently executes decoded `Insn` records directly (`src/cpu/decode.rs`, `exec.rs`), with a decode cache keyed by guest physical address (`mem.rs::fetch_decode`) and generation-based invalidation when a store hits a page that holds cached code. There is no uop layer yet; it is introduced with the JIT (M4), lowering from `Insn` (or via uops) while the Insn interpreter stays the oracle. Only 64-bit long mode (CS.L=1) is executed; other modes are not implemented. Measured: ~35-60 MIPS release on an Apple M-series host.

The text below is the target design for the JIT phase.

## uops (`ir.rs`)

Register-based, non-SSA, per-TB temporaries `t0..tN` (u64; `u128` handled by paired temps or helpers). Compact enum, ≤ 16 bytes each, stored in a `Vec<Uop>` per TB.

Groups:
- moves/consts: `Movi`, `Mov`, `LdReg(gpr)`, `StReg(gpr)` with width semantics (8/16/32/64, high-byte regs, 32-bit writes zero-extend)
- ALU: `Add Sub And Or Xor Shl Shr Sar Rol Ror Mul MulHi Div...` on temps, with `Ext8/16/32 (s/u)`
- flags: `SetCc(op, dst, src, src2)` (stores lazy state), `ComputeAll`, `ComputeC`, `CondEval(cc)` → temp bool
- memory: `Ld{8,16,32,64}(dst, addr, mmu_idx)`, `St..(src, addr, mmu_idx)`; `LdSeg`/base add folded into address computation by the translator
- control: `InsnStart(rip, len)`, `Label`, `Br`, `BrCond(cond, a, b)`, `GotoTb(slot)` (chainable), `ExitTb(next_rip kind)`, `ExitIndirect(t)` (lookup via jump cache)
- helpers: `Call(helper_id, args.., dst)` for anything complex (exceptions, descriptor loads, cpuid, msr, x87, SSE FP, rep-string iteration, in/out)
- vector: `Vld/Vst` (128-bit), `Vop(kind, dst_xmm, src_xmm)` as helper-backed in the interpreter and inlined to NEON in the JIT later

Each TB records: guest start pc, gpa(s), hflags, uops, instruction count (for icount), jump slots, `next_tb` cache pointers.

## Translation unit

Basic block ending at: control transfer, instruction that changes hflags (mov cr, wrmsr to EFER, lgdt, segment loads that alter code size, sti/iret/syscall), page boundary, or 64 insns. Entering an instruction fetch uses the exec TLB (so a fetch fault yields `#PF` with correct RIP and the TB is truncated before the faulting insn, QEMU `translator` style).

## TB cache (`tbcache.rs`)

- Hash: `(gpa_of_pc, hflags, cs_base/flags subset)` → TB id; arena of TBs, free list.
- Per-vCPU jump cache: 4096 entries indexed by `(pc ^ pc>>12) & mask`, stores `(pc, hflags, tb id)`; validated by exact compare, falls back to hash.
- Code buffer budget (JIT) or arena budget (interp): when full, flush everything (QEMU's `tb_flush`). Simple and correct.
- Invalidate-by-physical-page list per 04.

Fork notes (QEMU uplink `x-tblookup`, `x-tblayout`): jump-cache lookup from generated code and cold-code separation are JIT-phase optimizations; keep the TB struct layout compatible with that.

## Run loop

```
loop {
    if exit_request or irq pending-and-enabled → handle (deliver interrupt, or exit)
    if halted → wait on clock (warp to next deadline in Virtual mode) or console input
    tb = lookup(pc, hflags) or translate
    execute(tb)        // interp or jit, returns next action: continue | exception(vec,err) | halt | exit
    clock.advance(tb.icount)   // only counted instructions actually retired (fix up on mid-TB exception)
    poll devices/timers when clock >= next_deadline
}
```

With several vCPUs the loop body is `Cpu::run_slice`, which also returns when the vCPU should give way (quantum end at the event tick, `pause`, halted with nothing to wake it) and `cpu/smp.rs` picks the next vCPU; the TB/decode caches are shared between them (`docs/plans/03-cpu.md`, "SMP").

Chained TBs bypass the loop; therefore every `GotoTb` slot decrements/checks `exit_request` (one load+branch) — loops that never leave a chain must still be interruptible by timers (virtual clock deadline compares at the same point; use an icount budget counter in Env decremented per TB: reaching zero exits to the loop).

## Interpreter

Direct-threaded loop over `&[Uop]` with a `[u64; N]` temp file on the Rust stack. Memory ops call `tlb::load/store` (inline fast path). Helpers are plain Rust fns taking `&mut Env`. `unsafe` limited to Env-pointer and RAM-pointer access. Goal ≥50 MIPS in release, achieved by: lazy flags, TB caching, chaining by direct TB id in the loop (skip hash lookup when `next_tb` valid), no allocation in the hot path.

The interpreter stays in the tree permanently: fallback when JIT is unavailable, oracle for differential tests, and exec path for MMIO-heavy slow cases.
