# Overview

**Status:** implemented. The goal and non-goals below are the original v1 scope, which the project has since grown past (firmware, SMP, many more devices); [10-milestones.md](10-milestones.md) has what exists now.

## Goal

A small, embeddable x86_64 system emulator in Rust that boots an unmodified Linux kernel (Debian netboot `linux` bzImage, 6.12.x) with a serial console, with no BIOS and no firmware. Design follows QEMU TCG (translate guest blocks to micro-ops, cache them, run via interpreter or host JIT), minus everything not needed to boot Linux.

```rust
let mut machine = EmulatedMachine::new(MachineConfig::new().kernel(bzimage).cmdline("console=ttyS0"))?;
machine.start()?;
```

## Non-goals (v1)

SMP, ACPI/MP tables, PCI devices, virtio, disks, network, BIOS/real-mode boot, AVX/XSAVE, x87 precision beyond what Linux/glibc/busybox exercise, debugger stub, snapshots.

## Architecture

```
MachineConfig ──► EmulatedMachine ──► vCPU thread
                                       │
   boot (bzImage → RAM, boot_params)   ▼
                         ┌──────────── run loop ─────────────┐
                         │ irq/exit check → TB lookup → exec │
                         └─────┬─────────────────────┬───────┘
                          decode│                    │exec
                    x86 bytes → uops  ──►  interpreter | aarch64 JIT
                                       │
                 Env (regs, lazy flags, TLB) ── Bus ── RAM / ioports / devices
                                       │
                                  Clock (single virtual time source)
```

Both executors consume the same uop lists, so the interpreter is the correctness oracle for the JIT (see 08).

As built there is no uop layer: the interpreter and the JITs (aarch64, x86_64 and WebAssembly hosts) both work from decoded `Insn` records (05, 06), the CPU state is `Cpu` (`cpu/mod.rs`), and the devices are `Devices` (`dev/mod.rs`), not a `Bus`.

## Crate layout (single crate, zero deps)

The v1 plan, kept for the record. It is now a Cargo workspace (`uemulator` library, CLI, UEFI firmware and application, crypto, virgl, snapshot derive, web and Apple bindings; CLAUDE.md "Rules" has the list), and the library depends on `uemulator-crypto` and `uemulator-snapshot-derive` (both first-party). Of the files below, `clock.rs`, `ir.rs`, `exec/`, `mem/`, `cpu/{regs,tables,translate,seg,exc,msr}.rs`, `jit/{probe,buf}.rs` and `dev/{cmos,misc}.rs` were never created: guest RAM is `ram.rs`, the soft TLB and page walk are `cpu/mem.rs`, the decode cache and run loop are in `cpu/`, the JIT is `jit/` (`native.rs`, `emit.rs`/`a64.rs`, `emit_x64.rs`/`x64.rs`, `emit_wasm.rs`/`wasm.rs`).

```
(paths relative to crates/uemulator/)
src/lib.rs            re-exports only
src/machine.rs        EmulatedMachine, MachineConfig, ExitReason, thread control
src/console.rs        ConsoleHandle byte stream
src/boot.rs           bzImage parsing, loader, boot_params, e820, initial page tables/GDT
src/clock.rs          virtual clock, timer wheel
src/sys.rs            executable memory via libc / windows-sys (`jit` feature only)
src/cpu/{mod,regs,flags,decode,tables,translate,seg,exc,fpu,sse,cpuid,msr}.rs
src/mem/{ram,bus,tlb,mmu}.rs
src/ir.rs             uop definitions
src/exec/{tbcache,interp,helpers}.rs
src/jit/{mod,probe,buf,a64}.rs
src/dev/{mod,serial,pic,pit,cmos,misc}.rs
```

## Locked decisions

1. Interpreter first, JIT second; one decoder, one IR.
2. JIT backend for aarch64 hosts only in v1 (since then also x86_64 and WebAssembly hosts, 06). Everything else, and any host where the runtime probe fails, runs the interpreter. `JitMode::{Auto, Off, On}` in config.
3. Faulting state is exact: guest RIP and architectural state are consistent at every instruction that can fault (memory access, helper). Instructions commit register writes only after all faulting accesses.
4. Self-modifying code is handled by guest-physical page tracking (Linux alternatives/text_poke write through aliases).
5. IRQ and exit-request checks happen at TB boundaries even when TBs are chained.
6. One virtual clock feeds TSC, PIT, RTC. Default is instruction-count driven (deterministic); HLT warps time to the next deadline.
7. CPUID is a contract: advertise only implemented features.
8. Decoder is parameterized by mode (16/32/64-bit code size) even though only 64-bit is tested in v1; decompressor stubs and far returns may touch compat mode. (Real, 16/32-bit protected, virtual-8086 and compatibility mode now all run, on the interpreter only: 15.)
9. 0F 18–1F hint space is NOP, including ENDBR64 (F3 0F 1E FA).
10. The library never touches stdio, never calls `process::exit`, holds no global mutable state, and never panics on guest behavior (guest faults are guest exceptions; unimplemented opcodes raise #UD and are logged through an optional hook).
11. The guest has one vCPU by default, or up to 384 (pairs of SMT threads by default when the count is even). They take turns on one host thread (`docs/plans/03-cpu.md`, "SMP"; deterministic on the virtual clock, and the only mode on a freestanding host with no threads) or, by default where the host allows it, run on one host thread each on the host clock (`docs/plans/22-smp-threads.md`). With threads, x86 memory ordering and atomics are the emulator's job (x-tso in the fork is no longer moot): acquire/release guest accesses, locked instructions as compare-exchanges, a lock around the devices.

## Risks

- zstd decompression of the kernel payload (11 MB → ~58 MB) is ~1e9 guest instructions before any kernel code runs. Interpreter speed target: ≥50 MIPS in release; JIT target: ≥300 MIPS.
- Decode coverage: the stock kernel and glibc use a wide integer/SSE2 subset plus x87 in a few places. Mitigate with an opcode coverage log (unimplemented opcode → #UD + report).
- macOS W^X: JIT writes need per-thread toggling; chaining patches code, so toggling must wrap patch sites.
