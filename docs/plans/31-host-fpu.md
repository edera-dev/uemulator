# 31 — SSE/AVX/AVX-512 floating point on the host FPU (`x-ssefp`)

**Status:** implemented in the interpreter's lane functions, which every backend and every vector width reaches (legacy SSE, VEX and EVEX all run the same lane code; the JITs call it through `helper_exec`), with two dispatch shortcuts for the commonest forms. No inline JIT lowering: see "JIT lowering" for why not yet.

## Why

Every scalar and packed float instruction ran on the soft-float engine in `cpu/sse.rs` (`unpack`, `round_pack`, 128-bit significands). It is exact and the same on every host, and slow: llvmpipe's AVX-512 shaders and Plasma's startup spent most of their time there (KDE's wallpaper took 20 minutes, `24-display-input.md`).

## Design

The soft engine stays, as the oracle and as the fallback. In front of its lane functions (`arithmetic`, `minimum_maximum`, `compare`, `int_to_float`, `float_to_int`, `convert_float`, `fma::fused_lane`) sits `cpu/sse_host.rs`, which answers a lane with Rust's own `f32`/`f64` arithmetic when that answer is provably the soft engine's, bits and MXCSR flags both, and returns `None` otherwise. The old bodies are `soft_*` and run on `None`. The packed sqrt/add/mul/sub/min/div/max arm picks the lane function once per instruction (`host::packed_binary`) and sends only the lanes it cannot prove to the soft engine; flags are a sticky OR and lanes independent, so a mix of host and soft lanes is the all-soft answer.

Once the lane math was cheap, the interpreter's dispatch was most of what was left (`helper_exec`, `Cpu::exec`, `exec_two_byte`, `Cpu::sse`'s prologue). Two shortcuts take the commonest forms past it, each checked against the path it skips with exceptions unmasked:

- `Cpu::exec` gives the register forms of legacy `0F 51` and `58`-`5F` (every prefix) to `Cpu::exec_sse_float_register` before its opcode match: the SSE gate, the lanes, `fp_commit`, the register write, `rip`. Memory forms still go through `Cpu::sse`.
- The EVEX lane loop (`legacy_lane`) calls `Cpu::packed_float_lane` for the packed forms instead of staging scratch registers and re-entering `Cpu::sse` through `run_legacy_registers`. Flags are still committed per 128-bit lane, as before. VEX still runs `Cpu::sse` per lane.

The proof rests on a narrow domain, checked on the integer bits before any float is made:

- **Operands are normal numbers or zeros.** No NaN, infinity or denormal reaches the host. That removes everything the hosts disagree on: x86's first-operand NaN propagation and its negative "real indefinite" default NaN (aarch64 propagates and defaults differently, and wasm leaves NaN bits unspecified, which would also make snapshots host-dependent), DAZ, and the denormal-operand flag (aarch64 has none). Invalid and divide-by-zero cases (0/0, x/0, sqrt of a negative) are refused, so the host never makes a NaN.
- **Results are nonzero normals above the smallest normal, or zeros exact by construction** (a sum that cancels exactly, a product with a zero factor, sqrt of a zero, a zero divided by a normal). The host therefore never overflows, underflows or flushes, and FTZ, UE, OE and the soft engine's tininess simplification never come in. The smallest normal itself is refused: an inexact result that rounded up to it is tiny on hardware (UE, and a flushed zero under FTZ) but not in the soft engine, and refusing it keeps that difference in the soft engine alone. A double fused multiply-add only runs on the host inside an exponent band, where a zero result can only be an exact cancellation; outside it, unlike a plain sum, a nonzero product and addend can underflow together to zero.
- **Rounding is nearest-even.** Rust assumes the default floating-point environment on every host (running Rust float code under another mode or with flush-to-zero is already undefined behavior), so the host needs no FPCR/MXCSR control and there is no per-target asm for it; nothing in the library changes the host's mode. A host thread that ran with flush-to-zero anyway (audio threads sometimes do) would not change any result, since every result is normal, but could flush a subnormal TwoSum error term and so drop a precision flag near the bottom of the range. Operations that cannot round run in any MXCSR.RC: min, max, compares, truncating conversions (cvtt*, the C cast), widening single to double, and integer to float when the integer fits the significand. Directed modes and EVEX embedded rounding otherwise go to the soft engine.

In that domain precision (PE) is the only flag left, and it is computed exactly, without host flags (wasm has none, and LLVM does not order float operations against flag reads):

| Operation | Single | Double |
|---|---|---|
| add, sub | TwoSum error term is nonzero | same |
| mul | the product is exact in a double; inexact iff narrowing changed it | the product of the odd significands (u128) has more than 53 bits |
| div | quotient times divisor, exact in a double, differs from the dividend | odd(q) * odd(divisor) != odd(dividend) |
| sqrt | root squared, exact in a double, differs from the operand | odd(r)^2 != odd(operand) |
| fma | through doubles with no host FMA: exact product, TwoSum, round to odd, narrow (53 >= 24 + 2 makes round-to-odd then nearest correct) | host FMA instruction (aarch64 `fmadd`; x86_64 `vfmadd` when the host has FMA, detected at run time on hosted builds) and Boldo–Muller ErrFma up to its gamma, inside an exponent band that keeps every error term from underflowing; soft elsewhere (wasm, freestanding x86_64) |
| int to float | the magnitude's significant-bit span exceeds 24 | exceeds 53 |
| float to int | the rounded integral value differs from the operand; rounding by `(x + 2^52) - 2^52` under nearest-even, the range check on the rounded value | same, singles widened first |
| double to single | narrowing changed the value | — |

The odd-significand checks need the converse too: if the odd parts match, quotient times divisor is the dividend times a power of two, and since the rounded quotient is within half an ulp of the real one that power is 1 (likewise for the root).

## Exactness is shown by tests, not by the argument above

`cpu/sse_host_tests.rs` runs each wrapped function against its `soft_*` oracle and compares result bits and `FpEnv::flags`, under all 32 combinations of the four rounding modes, DAZ, FTZ and the underflow mask (the only mask the arithmetic reads; the others act in `fp_commit`, which sees identical flags). Operands are every pair of an edge table (zeros, the denormal extremes, the smallest normal and neighbours, the largest finite and neighbours, infinities, quiet and signaling NaNs with payloads, the indefinite, ones, 2^23/2^24/2^31/2^52/2^63 and neighbours, 0.5/1.5/2.5) plus random operands biased to edge neighbourhoods, few-bit significands (ties, exact results), extreme exponents, and partners that equal, negate or nearly cancel the first operand. FMA adds addends within a few ulps of minus the product and the 2^-1126 underflow case; the integer conversions add the halves around ±2^31, ±2^63 and zero, and integers at 2^24 ± 1 and 2^53 ± 1. The packed path is checked lane by lane against a soft-only loop. Two more tests compare the dispatch shortcuts with the code they skip, at the `Cpu` level: result, MXCSR, `rip` and the exception, with PE, IE and ZE unmasked, CR4.OSXMMEXCPT on and off, and CR0.TS set now and then. One mutation survives, as it should: refusing a double FMA that cancels to zero inside the band only makes the fast path more cautious.

The tests were checked by breaking the fast path on purpose (drop the inexact flag of add, compare gamma with 0 instead of -alpha2, accept the smallest normal, a non-strict range limit, round to even instead of odd, a 54-bit product limit, no div/sqrt precision, the double FMA band widened to -1022, no SSE gate or the wrong prefix priority or a commit after the write in the register shortcut, no commit or the scalar prefixes in the EVEX shortcut, a span limit one too wide, directed rounding allowed for inexact integer conversions, a reversed ordering key): each mutation fails a test.

They run natively on aarch64, on x86_64 under Rosetta, and on wasm32-wasip1 under Node.

## Numbers

`jit::bench` (`cargo test --release -p uemulator --lib jit::bench -- --ignored --nocapture --test-threads=1`) has five float loops, each with operands that make every product and sum inexact: `packed-fp` (movaps, mulps, addps, movapd, mulpd, addpd), `min-max` (minps, maxps, minpd, maxpd: almost no lane work, so the dispatch floor), `sqrt-div` (sqrtps, divps, sqrtpd, divpd and two moves), `convert` (cvtdq2ps, cvtps2dq, cvtsi2sd, cvttsd2si, cvtss2sd, cvtsd2ss) and `fma` (vfmadd231ps, vmulps, vaddps on zmm, vfmadd231sd). Each loop also counts its `dec`/`jnz` and moves, so the cost per float instruction is higher than the per-instruction figure.

Nanoseconds per guest instruction on an Apple Silicon host, the base commit against this branch, the two binaries alternated round by round, median of three rounds. Other agents were building on the same machine throughout; the load average is given for each set, and a round in a load spike is visible as an outlier that the median drops.

| Loop | aarch64 before | lane fast path | + dispatch shortcuts | x86_64 (Rosetta) before → after | wasm (Node) before → after |
|---|---|---|---|---|---|
| packed-fp | 25.8 | 12.7 | 7.7 | 41.7 → 13.5 | 39.5 → 11.9 |
| min-max | 22.0 | 15.9 | 9.5 | 34.9 → 14.5 | 33.6 → 13.2 |
| sqrt-div | 97.9 | 12.8 | 7.7 | 244 → 13.5 | 215 → 13.4 |
| convert | 22.1 | 18.8 | 20.2 | 40.6 → 34.9 | 28.9 → 24.8 |
| fma | 139 | 46.7 | 48.3 | 298 → 132 | 274 → 88 |
| load average | 28-32 | 28-32 | 13-20 | 11-13 | 15-22 |

The aarch64 "lane fast path" column is the host lanes, the monomorphic packed loop and the EVEX lane shortcut; the last column adds the register-form shortcut in `Cpu::exec`, measured against the previous column in its own rounds (12.6, 15.8, 13.1, 21.7, 48.2 there). The Rosetta and wasm columns compare the base commit with the whole branch. `convert` barely moves: its instructions are mostly dispatch, and the register shortcut does not cover conversions. On wasm the double FMA stays on the soft engine (no host FMA), which is why `fma` gains less there; wasip1 is hosted, so its sqrt is the `f64.sqrt` instruction (see "Limits" for the web build). A last aarch64 run at load 18-32 gave the same picture with every figure two to three times higher for the base.

No guest-level measurement was made: the machines under `~/.uemulator` have no float workload set up, and the host was too loaded for boot timings to mean anything.

## JIT lowering

Not done. After the dispatch shortcuts the lane math is about 1-3 ns of a 7-10 ns instruction; the rest is the block's call through `helper_exec` and `Cpu::exec`. Inline NEON or wasm `f32x4` code would need, besides the vector operation, a vectorised "every lane ordinary, every result safe" check and a vectorised exactness check for PE, or a guard that PE is already set and masked in MXCSR (the steady state of a float-heavy guest, where the precision flag cannot change anything) with a bail-out to the helper. Ranked by expected yield for the effort:

1. A direct JIT helper for the register-form packed and scalar float instructions (`helper_packed_float(cpu, opcode and format, destination, source)`), skipping `helper_exec` and `Cpu::exec` entirely. It is one call site per backend and no new semantics, so the existing corpus covers it. It touches the three emitters, which other work was changing at the same time, so it is left for after those merge.
2. A sticky-precision shortcut in the lanes: with PE already set and masked, skip the exactness work (TwoSum, the u128 products, ErrFma). Worth most for double FMA.
3. Inline lowering under that same guard, only if a profile still shows the helper call dominating.

AVX (VEX) float instructions still go through `Cpu::sse` once per 128-bit lane with register staging (`run_vex_passes`); they get the host lanes but not a dispatch shortcut.

## Limits

- Directed rounding modes and EVEX embedded rounding run on the soft engine for every operation that rounds. The cheap exact way to add them is to step the nearest-even result one ulp by the sign of the error term the inexact checks already compute (with care for the zero sign under round-down); nothing hot seems to use them.
- Denormal operands or results, NaNs and infinities always take the soft engine.
- Double-precision FMA is on the host only on aarch64 and on x86_64 hosts with FMA (hosted builds); wasm and freestanding x86_64 use the soft engine for it. Rosetta reports no FMA, so the x86_64 FMA branch runs only on a real x86_64 host (CI's Linux runners); its logic is the aarch64 branch's with `_mm_fmadd_sd` for `fmadd`.
- In the web build (`wasm32-unknown-unknown`, freestanding) sqrt comes from `libm`, which is correctly rounded but its generic software routine, not the `f64.sqrt` instruction: `core` has no stable sqrt (`core::arch::wasm32::f64_sqrt` is unstable, and `f64x2_sqrt` would put SIMD into a module that runs without it today). The differential tests run on wasip1, which uses std's sqrt, so the libm path is checked only by libm being correctly rounded.
- Partially masked EVEX lanes: the legacy kernels compute every element of a 128-bit lane that has any element enabled, so masked elements can still raise flags (a masked divide by the zero a masked memory element loads as). This predates the host path, which keeps it unchanged; passing the element mask into `packed_binary` would fix it and skip the work.
- The other float instructions (dpps, rounding, getexp/getmant, reductions, bf16, F16C) reach the fast path only where they call the lane functions above.
- x87 is unchanged.
