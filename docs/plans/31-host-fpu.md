# 31 — SSE/AVX/AVX-512 floating point on the host FPU (`x-ssefp`)

**Status:** implemented in the interpreter's lane functions, which every backend and every vector width reaches (legacy SSE, VEX and EVEX all run the same lane code; the JITs call it through `helper_exec`). No inline JIT lowering: see "JIT lowering" for why not yet.

## Why

Every scalar and packed float instruction ran on the soft-float engine in `cpu/sse.rs` (`unpack`, `round_pack`, 128-bit significands). It is exact and the same on every host, and slow: llvmpipe's AVX-512 shaders and Plasma's startup spent most of their time there (KDE's wallpaper took 20 minutes, `24-display-input.md`).

## Design

The soft engine stays, as the oracle and as the fallback. In front of its lane functions (`arithmetic`, `minimum_maximum`, `compare`, `int_to_float`, `float_to_int`, `convert_float`, `fma::fused_lane`) sits `cpu/sse_host.rs`, which answers a lane with Rust's own `f32`/`f64` arithmetic when that answer is provably the soft engine's, bits and MXCSR flags both, and returns `None` otherwise. The old bodies are `soft_*` and run on `None`. The packed sqrt/add/mul/sub/min/div/max arm picks the lane function once per instruction (`host::packed_binary`) and sends only the lanes it cannot prove to the soft engine; flags are a sticky OR and lanes independent, so a mix of host and soft lanes is the all-soft answer.

The proof rests on a narrow domain, checked on the integer bits before any float is made:

- **Operands are normal numbers or zeros.** No NaN, infinity or denormal reaches the host. That removes everything the hosts disagree on: x86's first-operand NaN propagation and its negative "real indefinite" default NaN (aarch64 propagates and defaults differently, and wasm leaves NaN bits unspecified, which would also make snapshots host-dependent), DAZ, and the denormal-operand flag (aarch64 has none). Invalid and divide-by-zero cases (0/0, x/0, sqrt of a negative) are refused, so the host never makes a NaN.
- **Results are nonzero normals above the smallest normal, or zeros exact by construction** (a sum that cancels exactly, a product with a zero factor, sqrt of a zero, a zero divided by a normal). The host therefore never overflows, underflows or flushes, and FTZ, UE, OE and the soft engine's tininess simplification never come in. The smallest normal itself is refused: an inexact result that rounded up to it is tiny on hardware (UE, and a flushed zero under FTZ) but not in the soft engine, and refusing it keeps that difference in the soft engine alone. A fused multiply-add whose nonzero product and addend cancel to zero is refused too: unlike a plain sum, that zero can be an underflow.
- **Rounding is nearest-even.** Rust assumes the default floating-point environment on every host (running Rust float code under another mode or with flush-to-zero is already undefined behavior), so the host needs no FPCR/MXCSR control and there is no per-target asm for it. Operations that cannot round run in any MXCSR.RC: min, max, compares, truncating conversions (cvtt*, the C cast), widening single to double, and integer to float when the integer fits the significand. Directed modes and EVEX embedded rounding otherwise go to the soft engine.

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

`cpu/sse_host_tests.rs` runs each wrapped function against its `soft_*` oracle and compares result bits and `FpEnv::flags`, under all 32 combinations of the four rounding modes, DAZ, FTZ and the underflow mask (the only mask the arithmetic reads; the others act in `fp_commit`, which sees identical flags). Operands are every pair of an edge table (zeros, the denormal extremes, the smallest normal and neighbours, the largest finite and neighbours, infinities, quiet and signaling NaNs with payloads, the indefinite, ones, 2^23/2^24/2^31/2^52/2^63 and neighbours, 0.5/1.5/2.5) plus random operands biased to edge neighbourhoods, few-bit significands (ties, exact results), extreme exponents, and partners that equal, negate or nearly cancel the first operand. FMA adds addends within a few ulps of minus the product and the 2^-1126 underflow case; the integer conversions add the halves around ±2^31, ±2^63 and zero, and integers at 2^24 ± 1 and 2^53 ± 1. The packed path is checked lane by lane against a soft-only loop.

The tests were checked by breaking the fast path on purpose (drop the inexact flag of add, compare gamma with 0 instead of -alpha2, accept the smallest normal, a non-strict range limit, round to even instead of odd, a 54-bit product limit, no div/sqrt precision, accept FMA zero results, a span limit one too wide, directed rounding allowed for inexact integer conversions, a reversed ordering key): each mutation fails a test.

They run natively on aarch64, on x86_64 under Rosetta, and on wasm32-wasip1 under Node.

## Numbers

NUMBERS

## JIT lowering

LOWERING

## Limits

- Directed rounding modes and EVEX embedded rounding run on the soft engine for every operation that rounds. The cheap exact way to add them is to step the nearest-even result one ulp by the sign of the error term the inexact checks already compute (with care for the zero sign under round-down); nothing hot seems to use them.
- Denormal operands or results, NaNs and infinities always take the soft engine.
- Double-precision FMA is on the host only on aarch64 and on x86_64 hosts with FMA (hosted builds); wasm and freestanding x86_64 use the soft engine for it.
- In the web build (`wasm32-unknown-unknown`, freestanding) sqrt comes from `libm`, which is correctly rounded but not the wasm `f64.sqrt` instruction; `core` has no stable sqrt.
- The other float instructions (dpps, rounding, getexp/getmant, reductions, bf16, F16C) reach the fast path only where they call the lane functions above.
- x87 is unchanged.
