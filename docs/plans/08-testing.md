# Testing

## Unit tests (`cargo test --release`)

- `boot.rs`: parse the real bzImage header from the cache when present (skip if absent) and assert the numbers in 02.
- Decoder: byte sequences → expected `Insn` (length, operand shape) incl. prefixes, REX, RIP-relative, 0F 18–1F nops, ENDBR64.
- Flags: `compute_all` vs reference formulas for every `cc_op` × width × edge values (0, 1, max, min, signed overflow).
- Devices: 16550 register protocol (probe sequence the 8250 driver uses), PIT counting against the clock, PIC priority/EOI.
- aarch64 encoder: instruction bytes vs known encodings.

## Instruction tests

Hand-assembled byte snippets (kept as `const &[u8]` with the assembly in a comment), run on a tiny machine (identity paging, flat 64-bit) that halts via `hlt`; assert registers/flags/memory. Cover every implemented opcode with edge cases (divide errors, shift counts 0/≥width, partial-register writes, cmov with false condition zero-extending, rep with RCX=0, string direction flag, misaligned and page-crossing accesses, faults mid-instruction restoring state).

Optional oracle: build `../ext/qemu-uplink` out of tree (never write inside it) and compare final state on the same snippets via `qemu-system-x86_64 -accel tcg -s`/semihost-less trace. Not required.

## Differential harness (JIT vs interpreter)

Gated on `jit_available()`. For each snippet and a random-state fuzz corpus (random GPRs/flags/memory, random valid instruction forms from the decoder tables): run the TB in the interpreter and JIT from identical snapshots and compare GPRs, rflags (materialized), XMM, memory dirty pages, exit reason and exception vector/error code/CR2. Also run chained-TB loops and SMC (write to code page between runs) in both modes.

## Boot tests (`tests/boot.rs`, `#[ignore]` unless cached assets exist or `UEMULATOR_BOOT=1`)

Use the same download cache as the example. Collect console output, stop at a match or timeout:

1. M1: output contains `Linux version 6.12`.
2. M2: no initrd → `Kernel panic - not syncing: VFS: Unable to mount root fs` then `ExitReason::Reset` (via `panic=-1`).
3. M3: with initrd → installer init/userspace output (e.g. `Debian GNU/Linux installer` / `Run /init as init process`).
4. M4: same as M2 with JIT on; differential suite green; wall-clock budget recorded.

Both `JitMode::Off` and `On` run in CI-equivalent local runs. Record MIPS (icount / wall) in test output to track perf.

## Debug aids (behind `log` hook, off by default)

Opcode coverage report (#UD sites), last-N-instruction trace ring dumped on triple fault, `cc_op` histogram, TB stats (translated, flushed, chained).
