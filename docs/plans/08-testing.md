# Testing

**Status:** partially implemented. Unit, instruction and differential tests exist as described; the cached-kernel boot tests (`tests/boot.rs`) do not, and tests build tiny guests with `testkit::FakeKernel` instead ([CLAUDE.md](../../CLAUDE.md), "Tests").

## Unit tests (`cargo test --release`)

- `boot/tests.rs`: parse bzImage headers built by `testkit::FakeKernel` (no cached kernel) and assert the numbers in 02.
- Decoder: byte sequences → expected `Insn` (length, operand shape) incl. prefixes, REX, RIP-relative, 0F 18–1F nops, ENDBR64.
- Flags (`cpu/flags.rs`): the lazy evaluation vs reference formulas for every `LazyFlagsOp` × width × edge values (0, 1, max, min, signed overflow).
- Devices: 16550 register protocol (probe sequence the 8250 driver uses), PIT counting against the clock, PIC priority/EOI.
- aarch64 encoder: instruction bytes vs known encodings.

## Instruction tests

Hand-assembled byte snippets (kept as `const &[u8]` with the assembly in a comment), run on a tiny machine (identity paging, flat 64-bit) that halts via `hlt`; assert registers/flags/memory. Cover every implemented opcode with edge cases (divide errors, shift counts 0/≥width, partial-register writes, cmov with false condition zero-extending, rep with RCX=0, string direction flag, misaligned and page-crossing accesses, faults mid-instruction restoring state).

Optional oracle: build `../ext/qemu-uplink` out of tree (never write inside it) and compare final state on the same snippets via `qemu-system-x86_64 -accel tcg -s`/semihost-less trace. Not required.

## Differential harness (JIT vs interpreter)

`testkit::run_both`/`assert_jit_matches_interpreter`, which pass trivially on a host without a JIT. For each snippet (`jit/diff_*.rs`) and a fixed-seed random corpus (`jit/diff_random.rs`: straight-line programs from instruction templates over seeded registers and memory): run it in the interpreter and JIT from identical starting states and compare GPRs, rflags (materialized), vector and x87 state, instruction count, all of RAM, and the stop reason (exception vector/error code/CR2). The same corpus runs on the WebAssembly backend under Node (CLAUDE.md). Also run chained-TB loops and SMC (write to code page between runs) in both modes.

## Boot tests (`tests/boot.rs`, `#[ignore]` unless cached assets exist or `UEMULATOR_BOOT=1`)

Use the same download cache as the example. Collect console output, stop at a match or timeout:

1. M1: output contains `Linux version 6.12`.
2. M2: no initrd → `Kernel panic - not syncing: VFS: Unable to mount root fs` then `ExitReason::Reset` (via `panic=-1`).
3. M3: with initrd → installer init/userspace output (e.g. `Debian GNU/Linux installer` / `Run /init as init process`).
4. M4: same as M2 with JIT on; differential suite green; wall-clock budget recorded.

Both `JitMode::Off` and `On` run in CI-equivalent local runs. Record MIPS (icount / wall) in test output to track perf.

## Debug aids (behind `log` hook, off by default)

Opcode coverage report (#UD sites), last-N-instruction trace ring dumped on triple fault, `cc_op` histogram, TB stats (translated, flushed, chained).

Done differently: `--debug` (`MachineConfig::debug`) dumps on exit the CPU state, the last instruction addresses (`Cpu::trace`), a histogram of the instructions that went through the interpreter from JIT blocks (`Cpu::helper_histogram`, what is left to lower), one of a guest hypervisor's #VMEXITs, and the guest's kernel log scanned from RAM. Not done: a #UD-site report, a `cc_op` histogram and reporting of the JIT's block counts (`JitStats`: translated, flushes, links), which only tests read.
