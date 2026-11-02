# JIT performance: direct chaining, more inline integer ops, native SIMD

**Status:** implemented as far as the Status paragraph and the Results section below say; the task lists are the plan as written.

> **For agentic workers:** execute task by task; steps use checkbox syntax. The interpreter is the oracle for every JIT change (`testkit::assert_jit_matches_interpreter`).

**Goal:** make the initramfs unpack (and everything else a daily-driven guest does) several times faster than today's ~21 s host time for the 252 MB Fedora netboot initramfs on an Apple Silicon host.

**Architecture:** three independent projects, each measured before and after with the same workload. (1) Direct block chaining in the style of QEMU's `goto_tb`. (2) Inline lowering for the integer instructions that still go through `helper_exec`. (3) Fast paths for the SIMD instructions the kernel's CRC32 and copy loops use on the Zen 4 CPU model.

**Tech Stack:** Rust, the in-tree aarch64 and x86_64 assemblers (`jit/a64.rs`, `jit/x64.rs`), `jit/native.rs` (block cache, shared by both backends).

## Baseline (2026-10-28, aarch64 host, `sample` 8 s window at t=6 s)

Boot: `uemulator advanced boot --kernel ~/.uemulator/machines/test/artifacts/vmlinuz --initramfs .../initrd.img --console serial`, wall time to the kernel's `Freeing initrd memory` line: 21.1 s; `Run /init` at 22.2 s. Busy-thread split: ~45% generated code, ~10% `pclmulqdq`, ~8% EVEX/AVX-512 helpers, ~13% interpreter core (`shift_rotate`, `flags::compute`, `load_operand`, `write_gpr`), ~1.5% MMU. A tight ~1000-sample cluster in the prologue region is the chain stub (inferred from its address and size, not disassembled).

`hack/perf/unpack.sh` (Task 0) reproduces the wall-time measurement; record each project's result in the table at the end of this file.

**Status (2026-10-28):** project 1 done on both backends; project 2 done for the instructions the histogram ranked highest, the tail is left; project 3 done as S1 plus two cheap fixes. The step lists below are the plan as written; the Results section at the end says what was built, in what order and why, and what is left.

## Global Constraints

- The library never touches stdio, never panics on guest behavior, holds no global mutable state.
- Interpreter first: a new uop or fast path gets the interpreter-oracle test before the lowering.
- Advertise a CPUID bit only together with the tests for the instructions behind it (nothing here changes CPUID).
- No magic constants: name offsets, encodings, sizes as `const`s.
- Before every commit: `./hack/autofix.sh`, then `cargo clippy --release --all-targets` prints zero warnings. Never commit with warnings.
- Tests: `cargo test --release --workspace`; x86_64 backend under Rosetta: `cargo test --release -p uemulator --target x86_64-apple-darwin`.
- After touching `jit/`: boot to the shell prompt with `--verify` and stop (do not type into it).
- `--verify` runs with chaining off (`Jit::new(false)`), so it does not exercise direct links; the differential tests and a normal boot are what cover them.

## Review Focus

- A linked block whose destination page is invalidated by the very store that runs in the source block: must not enter the stale destination (flag check at the linked entry).
- Destination invalidated, then the same `(rip, phys)` translated again: the old source site must be relinked to the new block, not left pointing at stale code or dropped forever.
- `Jit::reset` (buffer full) with live links: every site, pending entry and incoming list is discarded together.
- A block whose last instruction straddles a page, and a branch to the first byte of the next page: never directly linked.
- `next_event` reached exactly on a linked transition: the chain must stop, with no instruction lost or counted twice (`prepaid_icount`).

---

## Project 1: direct chaining

Design: a block's exit to a static same-page target has a patchable 4-byte (aarch64) or 5-byte (x86_64) slot placed immediately before the existing `mov rip; b chain` sequence. It starts as a nop, so execution falls into the existing stub path. Once the destination exists the slot becomes `b dest.link_entry`. Each block has two entries: `link_entry` (start of the block: the `icount >= next_event` and `block_exit_requested` checks the chain stub does, then falls into the body) and `code_addr` (the body, for the run loop and the stub, which have already done those checks). A taken `jcc` and the fall-through each get their own slot. Cross-page and indirect exits keep the stub. Same-page is sufficient for skipping the TLB probe: the source was entered through a validated mapping and the destination shares its page.

### Task 0: reproducible measurement

**Files:**
- Create: `hack/perf/unpack.sh`

**Interfaces:**
- Produces: `hack/perf/unpack.sh` prints `unpack <seconds>` (wall seconds from process start to the `Freeing initrd memory` line) using the `test` machine's artifacts, or the paths in `$KERNEL` and `$INITRAMFS`.

- [ ] **Step 1: Write the script** (boot with `advanced boot --console serial`, poll the output file every 0.25 s, kill the emulator at the line, print elapsed).
- [ ] **Step 2: Run it three times and write the median into the results table below as `baseline`.**
- [ ] **Step 3: Commit** `Perf: hack/perf/unpack.sh times the initramfs unpack`.

### Task 1: link bookkeeping in `native.rs` (arch-independent)

**Files:**
- Modify: `crates/uemulator/src/jit/native.rs`
- Test: same file, `mod tests`

**Interfaces:**
- Consumes: `arch::compile_block(...)` which after Task 2/3 returns `CompiledBlock`.
- Produces:
  - `pub(crate) struct LinkSite { pub code_offset: u32, pub target_rip: u64 }` (offset of the patch slot inside the compiled block's bytes; the block's own page decides `target_phys`).
  - `pub(crate) struct CompiledBlock { pub code: Vec<u8>, pub inlined: u32, pub body_offset: u32, pub links: Vec<LinkSite> }`.
  - `Block { link_addr: u64, .. }` (`code_addr` stays the body).
  - `Jit::link_pending(&mut self, key: (u64, u64))`, `Jit::unlink_into_pending(&mut self, index: u32)`, `JitStats::linked: u64` (sites patched, cumulative) and `JitStats::unlinked: u64`.
  - Arch hooks in `arch_a64.rs` and `arch_x64.rs`: `pub(crate) fn link_bytes(site_addr: u64, dest_addr: u64) -> Option<Vec<u8>>` (None when out of branch range) and `pub(crate) fn unlink_bytes() -> Vec<u8>` (the nop).

- [ ] **Step 1: Failing tests** (in `native.rs` tests, using `new_jit`/`translate_at_code_base` helpers): `a_loop_block_links_to_itself_once_translated` asserts `stats().linked == 1` after translating a `dec ecx; jnz loop` block; `invalidating_the_destination_unlinks_and_a_retranslation_relinks` asserts `linked`/`unlinked` counts across invalidate then retranslate; `reset_discards_links_and_pending_entries` asserts all three tables are empty afterwards (add `#[cfg(test)] fn link_state_len(&self) -> (usize, usize)` returning `(incoming sites, pending sites)`).
- [ ] **Step 2: Run, expect compile failure / FAIL.**
- [ ] **Step 3: Implement.** `Jit` gains `incoming: HashMap<u32, Vec<LinkedSite>>` (`LinkedSite { block: u32, code_offset: usize, key: (u64,u64) }`, offset is into `exec_mem`) and `pending: HashMap<(u64, u64), Vec<LinkedSite>>`. In `translate`, after `write_bytes`: for each `LinkSite`, only if `target_rip >> 12 == rip >> 12` and the block does not cross a page, compute `key = (target_rip, (phys_addr & !PAGE_MASK) | (target_rip & PAGE_MASK))`; if `block_map` has it, patch and record in `incoming`, else push to `pending`. After inserting the new block, take `pending.remove(&(rip, phys_addr))` and patch each to the new block's `link_addr`. In `invalidate_page`, for each dropped block take `incoming.remove(&index)`, write `unlink_bytes()` to each site and push the site into `pending` under its key. `reset` clears both maps.
- [ ] **Step 4: Run the new tests, expect PASS (with Task 2 or 3 in place for the compiler half; until then the arch `compile_block` returns an empty `links`, which keeps the tests of this task at "linked == 0" and the bookkeeping tests use a hand-built `CompiledBlock` through a `#[cfg(test)]` hook).**
- [ ] **Step 5: Commit** `JIT: block link bookkeeping (pending sites, incoming sites, unlink on invalidation)`.

### Task 2: aarch64 emission

**Files:**
- Modify: `crates/uemulator/src/jit/emit.rs` (`Gen::chain_exit`, `Gen::chain_to`, `Gen::jcc`, block prologue), `crates/uemulator/src/jit/arch_a64.rs` (`compile_block`, `link_bytes`, `unlink_bytes`)

**Interfaces:**
- Consumes: Task 1's `CompiledBlock`, `LinkSite`.
- Produces: blocks that begin with the link-entry checks, then the body; `Gen::links: Vec<LinkSite>`; `Gen::link_slot(target_rip: u64)` emitting the nop and recording the site.

- [ ] **Step 1: Failing tests** in `native.rs`: `a_linked_loop_does_not_go_through_the_chain_stub` (run a `dec ecx; jnz` loop of 1000 iterations twice; the second run has `linked >= 1`; assert the guest result matches and that `Cpu::icount` equals the interpreter's); `a_linked_chain_stops_at_the_event_deadline` (copy of `a_chain_stops_at_the_event_deadline_and_resumes_without_losing_instructions` after warm-up so links are patched); `a_store_into_the_linked_destination_page_exits_before_the_stale_block` (variant of `a_chain_does_not_enter_a_block_whose_page_was_invalidated` where source and destination share a page and the source's own store rewrites the destination).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Block layout: `link_entry:` ldr icount, ldr next_event, cmp, `b.cs miss`; ldrb block_exit_requested, `cbnz miss`; `body:`. Out of line at the end of the block: `miss:` (sub icount,1; str icount; store the block's own rip to `Cpu::rip`; `mov x0, 1<<EXIT_COUNT_SHIFT; b exit_ok`; the same sequence as the stub's miss path). `link_slot` emits `nop` (4 bytes, 4-aligned). In `jcc` the pay and the condition branch stay first (host flags are live), then each outcome emits `link_slot(target); mov x1, rip; b chain`. `link_bytes` encodes `b` with the imm26 range check; `unlink_bytes` is `nop`. `CompiledBlock::body_offset` is the offset of `body:`; `Jit::translate` sets `code_addr = base + body_offset`, `link_addr = base`.
- [ ] **Step 4: Run the Task 2 tests plus the whole `jit::` suite, expect PASS.**
- [ ] **Step 5: Run `cargo test --release --workspace`, then boot to the shell with `--verify` and stop.**
- [ ] **Step 6: Commit** `JIT: aarch64 direct chaining for same-page static exits`.

### Task 3: x86_64 emission

**Files:**
- Modify: `crates/uemulator/src/jit/emit_x64.rs` (`chain_exit`, `chain_to`, jcc, block start), `crates/uemulator/src/jit/arch_x64.rs`

**Interfaces:**
- Consumes/Produces: as Task 2. `link_slot` is the 5-byte nop `0F 1F 44 00 00`; `link_bytes` is `E9 rel32`.

- [ ] **Step 1: Port the three Task 2 tests (they are arch-independent; they live in `native.rs` and run on both) and run them under `--target x86_64-apple-darwin`, expect FAIL.**
- [ ] **Step 2: Implement like Task 2 using `x64::Asm` primitives.**
- [ ] **Step 3: Run `cargo test --release -p uemulator --target x86_64-apple-darwin`, expect PASS.**
- [ ] **Step 4: Commit** `JIT: x86_64 direct chaining for same-page static exits`.

### Task 4: measure and document

- [ ] **Step 1: `hack/perf/unpack.sh` three times; sample again; record in the results table.**
- [ ] **Step 2: Update `docs/plans/06-jit.md` (chaining paragraph) and `docs/plans/10-milestones.md`; commit** `Plans: direct chaining`.

---

## Project 2: inline the remaining hot integer instructions

Candidates, from the profile (`shift_rotate`, `flags::compute`, `load_operand`, `write_gpr`, `exec`): rotates (`rol`/`ror`), `adc`/`sbb`, `bt`/`bts`/`btr`/`btc` with register or immediate, `xchg r,r`, `bswap`, `movbe`, `tzcnt`/`lzcnt`/`popcnt`/`bsf`/`bsr`, `cmpxchg` (non-lock), 8/16-bit forms of the existing ALU set. Order is decided by data, not guesswork.

### Task 5: count which instructions reach `helper_exec`

**Files:**
- Modify: `crates/uemulator/src/jit/helpers.rs` (`helper_exec_pair`), `crates/uemulator/src/jit/native.rs` (`JitStats`)
- Test: `native.rs` tests

**Interfaces:**
- Produces: `#[cfg(feature = "jit_profile")]`-free, always-on cost kept near zero: `Jit::helper_histogram(&self) -> Vec<(u16 /*op*/, u64 /*count*/)>` backed by a `[u64; 512]` in `Cpu` incremented in `helper_exec_pair` only when `Cpu::count_helpers` is set; CLI `--jit-stats` prints the top 25 on exit.

- [ ] **Step 1: Failing test** `helper_histogram_counts_non_inlined_ops`: run `adc eax, ebx` three times under the JIT with counting on, expect the `adc` opcode to have count 3.
- [ ] **Step 2: Implement, run, expect PASS.**
- [x] **Step 3: Boot the unpack with `--debug --timeout 19` (the histogram is part of the `--debug` exit dump, not a new flag or API); the top rows are in the table below.**

#### Measured (after direct chaining; first 19 s of the Fedora netboot boot, `--debug --timeout 19`): 329M instructions through the interpreter

| count | instruction |
|------:|-------------|
| 218M | `66 C1 /5 ib` shr r/m16, imm8 (LZMA probability update) |
| 22.7M | `F2 0F38 F7` shrx; 18.8M `0F38 F5` bzhi; 12.1M `66 0F38 F7` shlx; 1.3M shlx r32 |
| 18.1M | `19 /r` sbb r/m32, r32; 2.7M `83 /3` sbb r/m64, imm8 |
| 10.9M | EVEX `66 0F3A 44` vpclmulqdq; 5.6M EVEX `66 0F3A 25` vpternlogq |
| 1.6M / 1.1M | `F3 0F BC` tzcnt / `F3 0F BD` lzcnt |
| ~1M each | pushf (9C), leave (C9), cmpxchg (0F B1), cli (FA), sti (FB), bt/bts/btr (0F BA /4../6, 0F A3/AB/B3), bzhi/bextr variants |

Order of lowering follows from this: **16-bit (and 8-bit) shifts** first (two thirds of all helper calls), then **BMI1/2 shifts and bzhi/bextr/andn/blsr** (vex GPR ops), then **adc/sbb**, then tzcnt/lzcnt/bsf/bsr, then the rest; the SIMD ones belong to Project 3.

### Task 6..N: one lowering per instruction family, in descending count

For each family chosen from Task 5, in this order of steps:

- [ ] **Step 1: Add diff cases in the matching `jit/diff_*.rs` (`diff_alu.rs` for adc/sbb/rotates/bswap, `diff_flags.rs` for flag-producing forms) covering every operand size, register and memory forms, and flag-in/flag-out combinations, using `assert_jit_matches_interpreter`.**
- [ ] **Step 2: Run, expect PASS (still through the helper) so the oracle is known good; then add the lowering in `emit.rs` (`try_inline`) and `emit_x64.rs`, with the flag model of the existing ops (`LazyFlagsOp`, `Gen::live`).**
- [ ] **Step 3: Run the diff tests on both hosts (`--target x86_64-apple-darwin`), `--verify` boot to the prompt, then `hack/perf/unpack.sh`; keep the lowering only if the unpack time or the helper count improves.**
- [ ] **Step 4: Commit** `JIT: inline <family> on aarch64 and x86_64`.

---

## Project 3: SIMD on the hot path

Guests on the Zen 4 model run `crc32` with `pclmulqdq`/`vpclmulqdq` and copy with EVEX loads/stores, so these go through `Cpu::exec` -> `sse_ext`/`evex` generic decode every instruction. Stages, cheapest first, each gated on the unpack time:

### Task S1: dedicated helper for `pclmulqdq`

**Files:**
- Modify: `crates/uemulator/src/jit/helpers.rs`, `crates/uemulator/src/jit/emit.rs`, `crates/uemulator/src/jit/emit_x64.rs`, `crates/uemulator/src/cpu/sse_ext.rs` (extract the multiply into `fn clmul64(a: u64, b: u64) -> u128`, hardware `PMULL` on aarch64 behind `target_feature`, a portable shift-xor loop otherwise)
- Test: `crates/uemulator/src/cpu/sse_ext.rs` tests, `jit/diff_misc.rs`

**Interfaces:**
- Produces: `fn clmul64(a: u64, b: u64) -> u128`; `extern "C" fn helper_pclmulqdq(cpu: *mut Cpu, dst_xmm: u64, src: u64, imm_and_form: u64) -> u64` (status like other helpers; `src` is an xmm index or a guest address when the memory form flag in `imm_and_form` is set); `Addrs::pclmulqdq`.

- [ ] **Step 1: Failing tests:** `clmul64_matches_the_shift_xor_definition` (random pairs, edge cases 0, 1, u64::MAX, bit 63 times bit 63); a diff test per `imm` selector (0x00, 0x01, 0x10, 0x11), register and memory source, plus a faulting memory source (page fault delivered with the right rip).
- [ ] **Step 2: Implement, run, expect PASS; then measure with `hack/perf/unpack.sh`.**
- [ ] **Step 3: Commit** `JIT: dedicated pclmulqdq helper`.

### Task S2: VEX/EVEX 128-512 bit moves, logic and ternary logic without the generic engine

The kernel's `crc32_pclmul` and `memcpy`-style loops are made of `vmovdqu(8|16|32|64)`, `vpxor(d|q)`, `vpternlogd/q`, `vpclmulqdq` on ymm/zmm. Add `helper_vector_fast(cpu, insn_ptr, rip)` handling exactly these opcodes on the register file layout `Cpu` uses (`cpu/xsave.rs` documents it), selected at translate time so other EVEX instructions keep the generic path.

- [ ] **Step 1: Failing diff tests per instruction in `jit/diff_misc.rs` (all widths, mask register and zeroing variants for the EVEX forms, memory operands crossing a page, `{k}` merge masking).**
- [ ] **Step 2: Implement, PASS, measure.**
- [ ] **Step 3: Commit** `JIT: fast path for common AVX-512 moves, logic and vpclmulqdq`.

### Task S3: inline NEON lowering (aarch64 host) for the same set

Only if S1+S2 leave SIMD above ~10% of the busy samples. Inline `ldp q`/`stp q` of the xmm/ymm/zmm lanes in `Cpu`, `eor`, `bsl`/`bif` for ternary logic, `pmull`/`pmull2`. x86_64 host keeps the helper. The tasks mirror S2 with the diff tests unchanged.

- [ ] **Step 1: Re-sample; write the SIMD share in the results table; proceed only if above threshold.**
- [ ] **Step 2: Lower each instruction with its diff test already green; PASS; measure; commit per instruction family.**

### Task S4: decide on hiding features

Only if the unpack is still well over 2x slower than the target after S1-S3: write the options and numbers into this file and ask before changing CPUID (it conflicts with the Zen 4 model contract in `docs/plans/03-cpu.md`).

---

## Results

`hack/perf/unpack.sh`, median of three on an aarch64 host with another emulator job running alongside (so absolute numbers are noisy; the runs within one row agree to 0.1 s).

| step | unpack seconds | commit |
|------|---------------:|--------|
| baseline | 22.1 | |
| direct chaining (aarch64) | 18.8 | 5c669ca |
| shl/shr/sar of 8- and 16-bit operands | 16.8 | 0c4fac1 |
| pclmulqdq on the host's carry-less multiply | 15.7 | 78025cb |
| dead lazy-flag records | 15.5 | b3b972a |
| shlx/shrx/sarx, bzhi, andn, blsr group | 15.05 | 6d3c054 |
| adc/sbb | 14.87 | b4febc2 |
| unmasked EVEX commit by lane | 14.73 | 032a018 |

Total: 22.1 s -> 14.7 s, about 1.5x. The x86_64 backend has every lowering except adc/sbb into memory (checked under Rosetta; not timed).

## What changed from the plan, and what is left

- Project 2 was reordered by the `--debug` histogram, not by the plan's guesses: 16-bit shifts (two thirds of all interpreter calls at the start), then BMI, then adc/sbb. Left in the interpreter: mul (`f7 /4`, 9M calls in 19 s), leave (4M), tzcnt/lzcnt/bsf/bsr, bt*, cmpxchg, pushf, cli/sti, shld, pshufd/pxor/movdqa and the other legacy SSE integer ops, and `vpclmulqdq`/`vpternlogq` (17M calls).
- Project 3 stopped at S1 plus two cheap fixes. The SIMD helpers were 25% of the emulator thread's time because `pclmulqdq` was a 64-iteration bit loop; with the hardware multiply and the lane copy they are a few percent, which does not pay for a dedicated vector fast path (S2) or NEON lowering (S3). S4 (hiding CPUID bits) was not needed and was not done.
- Measured but not built: dropping the link entry's two checks would save about 3% (they could stay only on exits that follow a store); keeping `icount` in a host register across blocks removes a store-to-load chain per block.
- What bounds the rest: guest registers live in `Cpu` memory, so every instruction loads and stores them, and every block pays an `icount` read-modify-write, a link-entry check and a TLB probe per access. Roughly two thirds of the host instructions in a hot block were that bookkeeping. Getting to the next factor means pinning hot guest registers in host registers across blocks (with spills at helper calls and exits) and `icount` in a register: a redesign of the emitters, not another lowering.
- Seen on the way: the CLI's terminal-size polling thread wakes about every millisecond (an `ioctl` each time); it is off the emulator thread, so it does not slow the guest, but it burns a core fraction. A workspace `cargo build --release` fails in `uemulator-web` on a macOS host (the crate is wasm32-only; not looked into, `-p uemulator-cli` builds).

## Round 2: the installer's userspace (Debian netboot)

The Fedora initramfs of round 1 is gone from `~/.uemulator`, and its unpack was already nearly all JIT. Round 2 measures the Debian netboot kernel and initrd (`linux`, `initrd.gz` from `quick create --install debian`, copied somewhere stable) from process start to the installer's first curses box (the `lqqqqqqq` of its frame on the serial console), CPU seconds and wall seconds agreeing to 0.01 s because the emulator thread never idles. `--debug --timeout 8` gave 121M instructions through the interpreter at the start, almost all of them glibc's AVX-512 string loops (`kmovd`, `vpcmpeqb` into a mask, `bsf`, `vmovdqu64`, `vptestmb`, `mul`).

| step | boot to installer, s | interpreter calls in 8 s |
|------|---------------------:|-------------------------:|
| round 1 | 11.3 | 121M |
| bsf/bsr/tzcnt/lzcnt, one-operand mul/imul, shld/shrd by immediate | (not timed alone) | |
| AVX-512 moves, ALU, byte compares into opmasks, kmov | 6.6 | 41M |
| write masks and `vpcmpub`/`vpcmpb` predicates | 5.1 | 27M |
| carry-in and condition at a block start from the previous record | 5.0 | |

What the lowerings rest on:

- Flag records. `bsf`/`bsr` record `SzpCfOf` of the source with CF clear (the undefined flags are what QEMU leaves, and the interpreter now leaves the same so it stays the oracle); `tzcnt`/`lzcnt` record the count with CF for a zero source. `mul`/`imul` record CF = OF from the high half.
- AVX-512. Vectors are processed a 128-bit lane at a time through `v0`/`v1` (`xmm0`/`xmm1`), loaded from and stored to `Cpu` per lane; `helpers::vector_lane_offset` maps zmm0-31 to the split register file. One gate per block (CR4.OSXSAVE, XCR0, CR0.TS) sends the first instruction to the interpreter to raise #UD/#NM. A write mask becomes a vector mask through `OPMASK_EXPANSION` and is applied with `bsl` / `pand`-`pandn`-`por`; a masked load reads the whole vector inside the TLB-vouched page (it cannot fault). Masked stores to memory, broadcasts and wider compares stay in the interpreter.
- Diff tests of anything with a memory operand must warm the TLB first (`diff_evex::WARM_TLB`): the first access of a block always runs in the interpreter, and an idempotent instruction run twice hides a fast path that is never taken.
- `rebuild_flags_from_sub` (both backends) and `carry_from_lazy_record` (aarch64): `helper_cond` plus `flags::compute` were a fifth of the emulator thread's samples; a conditional branch that starts a block is the second of two on one compare. The operands are shifted to the top of 64 bits, so SF/ZF/CF/OF come out right and the parity condition keeps the helper on x86-64.

What is left, from a `sample` of the emulator thread at the installer (a third of it outside generated code: page walks in `translate`, `Jit::lookup`, the interpreter for `rep stos`, `cmpxchg`, `pushf`, `cli`, `div`, `bt*`, VEX `vmovdqu`, `kortest`, `vpbroadcastb`): more than half of the samples inside generated code are the shared chain stub (`arch_a64::chain_stub`), the lookup every `ret`, indirect branch and cross-page jump pays: a budget check, a TLB probe of the target page, a jump-cache probe and a validity byte, four dependent loads in different cache lines. The jump cache's size does not matter (14, 16 and 18 bits all give 5.0 s). Two ways to shrink it, neither tried: link cross-page direct jumps and calls with an inline TLB check instead of the stub (same-page ones are linked already), and an inline cache of the last target at each `ret`. Pinning guest registers across blocks (round 1's conclusion) is still the structural item; it was not attempted because every emitter assumes the register file lives in `Cpu`.

## Round 3: Xen with a PV and a PVH dom0 (2026-11-01)

Workload: the web demo's Xen machine (Edera's Xen 4.22.1-pre `xen.gz` pinned in `hack/web/demo-xen.ref`, Alpine 3.24.3's virt kernel as dom0, the demo initramfs, 512 MiB, 1 vCPU, `dom0_mem=384M,max:384M`), from process start to the dom0 shell's banner, natively (`advanced boot --hypervisor`) and under Node on the WebAssembly build (`hack/wasm/boot.mjs`): `hack/perf/xen.sh pv|pv-iommu|pvh-iommu [native|wasm]`. `pvh-iommu` is `dom0=pvh-auto` with the IOMMU; there is no PVH configuration without it (Xen falls back to PV). `pv`/`pv-iommu` force `dom0=pv`. `xen.efi` was not measured: the pinned image only ships `xen.gz`. Guest time is the host's wall clock, so time-bound phases (calibration loops, the RTC) cost wall time whatever the emulator's speed. The host (M3 Max) had two other emulators at 100% and a load average of 6 to 9 throughout, so only interleaved medians mean anything.

| configuration | before | after |
|---|---:|---:|
| PV, native | 5.64 | 5.65 (5.25 -> 5.12 in quieter interleaved runs) |
| PV + IOMMU, native | 5.77 | 5.59 |
| PVH + IOMMU, native | 4.31 | 3.47 (4.04 -> 3.29 in a quieter run) |
| PV, Node | 23.66 | 20.57 |
| PV + IOMMU, Node | 23.70 | 20.41 |
| PVH + IOMMU, Node | 16.68 | 14.12 |

Seconds, median of three interleaved runs. With and without the IOMMU the boots take the same time: the demo's dom0 has no DMA-heavy device, and interrupt remapping is a handful of table reads per interrupt.

Where the time was (`sample` natively, `node --cpu-prof` on the web build, `--debug`'s interpreter histogram and its new #VMEXIT histogram):

- **A PVH dom0 takes 100k #VMEXITs, nearly all one loop.** Linux's `quick_pit_calibrate` reads PIT channel 2, which Xen does not emulate for a PVH dom0, so every read returns 0xff and `pit_expect_msb` runs its whole 50000 iterations of two port reads, each an IOIO exit. Natively that was 0.52 s at 5.3 us an exit, under Node 1.2 s. Everything else is a few hundred exits (cpuid, a dozen `vmmcall`s, MSRs): nested SVM is not exit-bound here.
- **VMRUN and #VMEXIT flushed the whole soft TLB.** Every exit then refilled Xen's working set, and the guest redid a two-dimensional walk (up to 24 table reads) for every page it touched again. They now swap TLBs instead, keyed by the guest's ASID, nested CR3 and paging registers and honouring TLB_CONTROL and INVLPGA (`docs/plans/03-cpu.md`, SVM): the calibration loop's exit costs 1.7 us natively (0.17 s for the loop) and the loop 0.54 s under Node.
- **Nested walks.** `translate` plus `nested_translate` were 18% of the native dom0 kernel phase. `NestedLeafCache` keeps resolved nested leaves (Xen maps a dom0 with 1 GiB and 2 MiB pages) for as long as the guest's TLB lives; about 4% of the dom0 kernel phase natively, more under Node.
- **Xen waited a second for the RTC.** It reads the CMOS clock by waiting up to a second for update-in-progress to rise, and the RTC never raised it. It now does for the last 244 us of each second (`docs/plans/07-devices.md`); the wait is now until the next second of the machine's clock (0.7 s natively, about 0.4 s under Node, where Xen gets there later).
- **134M `rol r64, imm8` through the interpreter under Node in a PV boot.** Linux's PV start-up runs a hash loop that ends its block with `rol rbx, 25; jmp`; the WebAssembly backend only lowered rotates whose flags a later instruction of the block overwrote. Rotates are now always inline, with CF/OF merged by `helper_set_rotate_flags` when they may be read: the PV dom0's start-up phase went from 10.4 s to 7.5 s under Node, and interpreter calls in the boot from 165M to 25M.

`--verify` (JIT in lockstep with the interpreter) boots both the PVH dom0 behind the IOMMU and the PV dom0 to the shell with no divergence (133 s and 237 s), and `hack/kvm/run.sh` passes (KVM inside the guest on the emulator's SVM, which relies on TLB_CONTROL and ASIDs).

Found on the way: the soft TLB of a nested guest took its write and exec tags from the guest's page tables alone, so after a read cached a page, a store went through without the nested page fault a read-only nested mapping (Xen's log-dirty mode, read-only p2m types) must raise and without setting the nested Dirty bit, and a fetch ignored nested NX. Fixed (`NestedRights`) with tests.

What is left, in order of size:

- **Xen inflating the dom0 kernel: 1.0 s natively, 2.0 s under Node** (the "Building a P[V|VH] Dom0" phase; Xen's gunzip in the JIT). Handing Xen the ELF inside the bzImage instead (`--kernel vmlinux`, which Xen accepts as is) takes the phase to 0.08 s: PVH 3.25 -> 2.35 s and PV 5.0 -> 4.2 s natively, PVH 13.9 -> 12.5 s and PV 21.4 -> 17.7 s under Node. Doing that in the library (inflate a gzip payload with `boot/inflate.rs` when booting a hypervisor) changes what the firmware hands Xen compared with GRUB, needs ~26 MiB more RAM for the boot module, and its cost under WebAssembly is unmeasured, so it is not done; if it is, make it an option the demo turns on.
- **The dom0 kernel itself under Node: 8.9 s of a PVH boot** against 6.2 s for the same kernel without Xen. General JIT speed (round 1 and 2's register pinning conclusion) and the WebAssembly backend's gaps: `mul r64`, `shrd`, 64-bit `imul` with live flags and `cmpxchg` still go through the interpreter (about 10M calls in a boot), and `Cpu::step_block` itself is 16% of the samples under Node (cold steps and lookups).
- **The RTC wait** (0.4 to 0.7 s): it is up to a second on hardware too; only the phase of the machine's clock decides it.
- **A PV dom0's start-up: 1.8 s natively, 7.5 s under Node** (from Xen's hand-over to dom0's first console line), and natively 45% of it is the rotate flag merge: `helper_set_rotate_flags` and the `flags::compute` it calls for SF/ZF/PF/AF, once per iteration of the `rol rbx, 25; jmp` hash loop (134M times). The same helper serves the WebAssembly backend now. Making the merge cheap (computing the kept flags inline when the previous flag writer is in the same block, or a lazy record that keeps the previous one and overrides CF/OF) is the next step for PV. The x86-64 backend does not lower rotates at all, so an x86-64 host still sends those 134M through the interpreter (not measured here: no x86-64 host; Rosetta would not be representative).
- Native PV start-up measured about 0.05 to 0.1 s (3 to 5%) slower after the nested-leaf-cache commit, in interleaved runs on the loaded host (1.78 s before, 1.80 to 1.91 s after; the nested-rights commit alone is 1.77 to 1.80 s). A PV dom0 never enters SVM, so the cache's code does not run; inlining the walker's no-nested-paging path did not change it, and what is left (the `Cpu` layout shifting with the new `Svm` field) was not pursued. The PV boot is still 0.1 s faster overall natively (the RTC).
- **The wasm block cache flushes once in a PV boot** (76k blocks against `MAX_BLOCKS` = 65536); raising it costs engine memory per instance (the boot peaks at 1.8 GB RSS under Node already).
- The IOMMU: per-access cost is now within 2 ns of no IOMMU (`docs/plans/16-iommu.md`); the device lock around every DMA was left alone (nothing here measures it, and the threading rules make it the riskiest part).
