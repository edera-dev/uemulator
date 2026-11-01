# uemulator

An x86_64 system emulator in Rust that boots unmodified distro kernels. It runs Alpine, the Debian installer, Fedora and a Xen 4.21 dom0 on an emulated AMD Zen 4 PC, with its own UEFI firmware, ACPI tables, APICs, IOMMU and virtio devices. The design follows QEMU TCG: decode to micro-ops, cache the blocks, run them on an interpreter or a host JIT.

About 190k lines of Rust in five crates, tests included. The library is `no_std` + `alloc` when built freestanding, and with `--no-default-features` it has zero dependencies and is interpreter-only. The same core runs as a CLI on a desktop, as a Web Worker in a browser, and as a UEFI application.

```
$ cargo run --release -- quick create test --install alpine
$ cargo run --release -- quick boot test --cpus 4
```

- [What it boots](#what-it-boots)
- [Execution engine](#execution-engine)
- [CPU model](#cpu-model)
- [SMP](#smp)
- [Platform and devices](#platform-and-devices)
- [Firmware and boot paths](#firmware-and-boot-paths)
- [Hosts](#hosts)
- [Quick start](#quick-start)
- [Embedding](#embedding)
- [CLI reference](#cli-reference)
- [Architecture](#architecture)
- [Testing and debugging](#testing-and-debugging)
- [Status and limits](#status-and-limits)
- [Design docs](#design-docs)

## What it boots

Everything below was run on the emulator, not just decoded. "Verified" means the result is recorded in `docs/plans/10-milestones.md` or the plan file named in the last column.

| Guest | What works | Details |
|---|---|---|
| Alpine 6.18 netboot kernel and initramfs | Installer to a login prompt on 1, 4 and 64 vCPUs, by direct 64-bit entry and through UEFI. Installing to disk and booting the installed system is verified under `--firmware bios` (syslinux), see the last row | 10, 17, 19 |
| Debian trixie 6.12 installer | Runs to the installer dialogs. With `--iommu` the guest enables x2APIC cluster mode and interrupt remapping and drives virtio-net through the IOMMU | 16 |
| Fedora 44 | Cloud image to a root login through shim and GRUB on 4 vCPUs, `dnf` installing Xen over the NAT. Netboot kernel with 8 GiB of RAM through UEFI up to dracut | 18, 04 |
| Xen 4.21 | GRUB multiboot2 or `xen.efi` from the UEFI firmware, a PVH Alpine dom0 to its login prompt in 26 s of host time, `xl info`, `xl list`, a PVH domU running its initramfs | 18 |
| Linux KVM inside the guest | Alpine loads `kvm_amd` and runs vCPUs in real, protected, long, PAE and 32-bit-paging mode on the emulator's SVM and nested paging (`hack/kvm/run.sh`) | 03, 15 |
| Alpine i686 | A 32-bit kernel to a shell | 15 |
| Alpine in the browser | busybox shell in Chrome, keyboard included | 13 |
| Alpine disk boot through the BIOS | A disk installed by `setup-alpine`: syslinux 6.04 loads the kernel from the boot sector and boots it to a login | 17 |

Getting there found real bugs. Seven emulator bugs surfaced during the Xen bring-up alone (compatibility-mode returns, `EFER.LMA` at `#VMEXIT`, an IOMMU/FCH address overlap, stale JIT blocks after SVM exits, MMIO clock skew in the JIT, MADT and `_PRT` polarity, 16550 input pacing). The milestones file records them.

## Execution engine

Guest code is decoded once into a micro-op list (`ir.rs`). Both executors consume the same lists, so the interpreter is the oracle for every JIT: a new uop gets an interpreter implementation and a test before any backend sees it.

| Piece | How it works |
|---|---|
| Decoder | Table driven, parameterized by CPU mode (16/32/64-bit code size), legacy, VEX and EVEX prefixes |
| Flags | Lazy: operands and result are kept, flags are computed when something reads them. The JIT drops lazy-flag records that nothing reads |
| Memory | Soft TLB with inline fast paths in generated code, a page walker for 4-level, PAE and 32-bit paging, and A/D bit updates as compare-exchanges |
| Self-modifying code | Tracked per guest-physical page, so Linux's alternatives and `text_poke` through aliases invalidate the right blocks |
| Exactness | Guest `rip` and architectural state are exact at every instruction that can fault; register writes commit after the last faulting access |
| Interpreter | Runs everywhere, always available, and is the fallback when the JIT probe fails |

### JIT backends

| Backend | Host | Notes |
|---|---|---|
| aarch64 | Apple Silicon macOS, Linux | `MAP_JIT` and per-thread write protection on macOS, inline TLB paths, `ldar`/`stlr` for x86 ordering under threads |
| x86_64 | macOS, Linux, Windows | Same inlined subset as aarch64, lockstep-verified (see [limits](#status-and-limits) for what has been run where) |
| WebAssembly | Browser, Node | Each block becomes a WebAssembly module that the host instantiates; blocks chain with tail calls through a stub module |

All three backends: blocks are position independent. The cache is keyed by physical address, and every guest `rip` inside a block is the page base plus a translation-time offset, never an absolute virtual address. One code cache serves all vCPUs; an Alpine boot builds about 193k blocks in 58 MiB, where before the cache was shared and position independent it was 1.64M blocks and a flush. Anything the backend does not lower runs through a helper call into the interpreter, and `--debug` prints a histogram of what still gets there, which is how the next inlining target is picked.

Native backends (aarch64, x86_64) only:

- Direct chaining of same-page static exits, patched under the cache lock through `ExecMem::write_word`.
- Wide inline lowering: ALU, shifts including `shlx`/`shrx`/`sarx`, `bzhi`, `andn`, `blsr`, `adc`/`sbb`, `push`/`pop`, `lea`, `call`/`ret`, `jcc` on lazy flags.
- Hardware carry-less multiply for `pclmulqdq`.

WebAssembly differs: there is nothing to patch, so every block transition goes through the chain stub. It keeps no host flags (a `cmp; jcc` is computed from locals), compiles a block only after its first instruction has run enough times in the interpreter, and leaves `adc`/`sbb`, rotates, narrow shifts, 64-bit `imul` and BMI to `helper_exec`.

Separately from the JITs, the interpreter runs `rep movsb` in byte-sized slices for ERMS/FSRM guests, about 12x faster on copy-heavy workloads.

Measured numbers from the plans, each on the host named:

| Measurement | Result | Host |
|---|---|---|
| Alpine boot, no initrd, CPU time | 4.8 s to 1.3 s after chaining and wide lowering | aarch64 JIT |
| Boot to the Alpine installer with initrd | 13.9 s to 4.7 s; the interpreter takes about 60 s | aarch64 JIT |
| 252 MB Fedora netboot initramfs unpack | 22.1 s to 14.7 s (`hack/perf/unpack.sh`) | Apple Silicon |
| Hot code under the WebAssembly JIT | about 9x the interpreter | browser |
| Debian netboot kernel to `Machine keyring initialized` | about 6 s with the JIT, about 55 s interpreted | Node, `hack/wasm/boot.mjs` |

### Lockstep verification

`--verify` (`MachineConfig::verify_jit`) runs the JIT and the interpreter side by side, compares architectural state after every block, and stops at the first divergence. The installer boot is divergence-free (milestone M4). It does not take typed input: host keystrokes reach only the JIT side's console, so the first one shows up as a bogus divergence.

## CPU model

The identity is an AMD EPYC 9004 "Genoa" (family 19h, model 11h, signature `0x00A10F11`), cross-checked against QEMU's `EPYC-Genoa`. Linux reports `AMD EPYC-Genoa Processor` and picks the Zen 4 erratum and mitigation paths. CPUID is a contract: a bit is advertised in the same commit as the instructions and tests behind it, and a bit with no implementation is absent.

| Area | Implemented and advertised |
|---|---|
| Base | x86-64, x87, MMX, SSE through SSE4.2 (CRC32, `pcmpestri` family), POPCNT, LZCNT, CMPXCHG16B, MOVBE, FSGSBASE, RDTSCP, SYSCALL/SYSENTER |
| Vector | AVX, AVX2, FMA, F16C, gathers and masked moves, YMM state in XSAVE |
| AVX-512 | F, VL, BW, DQ, CD, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, BF16, with opmasks, broadcast, embedded rounding and fault suppression; 32 ZMM registers |
| Crypto and bit tricks | AES-NI, PCLMULQDQ, SHA-NI, GFNI, VAES, VPCLMULQDQ, BMI1, BMI2, ADX, CLZERO, CLWB, CLFLUSHOPT, WBNOINVD |
| State management | XSAVE, XSAVEOPT, XSAVEC, XSAVES, XGETBV1 with the compacted format, RDRAND, RDSEED, RDPID, ERMS, FSRM |
| Protection | SMEP, SMAP, UMIP, PCID, INVPCID, NX, 1 GiB pages, MTRRs, PAT, MONITOR/MWAIT |
| Virtualization | AMD SVM with nested paging: `vmrun`, `vmload`, `vmsave`, `stgi`, `clgi`, `invlpga`, intercepts, event injection, TSC offset, VMCB clean bits, flush-by-ASID, next-RIP save |
| Legacy modes | Real mode, 16/32-bit protected mode, compatibility mode, virtual-8086, hardware task switching, call gates, 32-bit and PAE paging with latched PDPTEs |
| Topology | Leaves 0xB, 0x8000001D and 0x8000001E consistent with the configured vCPU count and x2APIC ids; Zen 4 cache and TLB geometry |

Speculation: the guest sees an in-order CPU that is immune to the speculative-execution bugs, so Linux skips retpolines, return thunks, KPTI and `VERW`. That is true of the model, since it never speculates.

Deliberate departures, kept short: TSC-deadline is advertised even though AMD has none (calibration is much faster), the speculation-control interface uses its Intel spelling, and the address widths are 40 physical and 48 virtual bits. There is no PMU, no AVX512_FP16, no LA57, no PKU, no CET and no VMX. `docs/plans/21-ryzen-7000-parity.md` is the plan for the next ones.

One note on trust. An earlier milestone entry claimed AVX-512 coverage that unit tests could not back: `vpsubsw` (EVEX) and `vdbpsadbw` raised `#UD` while AVX512BW was advertised, and only a real guest running one of every intrinsic found it. `docs/plans/03-cpu.md` ("Real-guest check") records how AVX was then verified in a guest against QEMU TCG, under signals and context switches, and JIT against interpreter.

## SMP

`--cpus N`, 1 to 64. There are two scheduling modes (`--smp`).

| Mode | How it runs | Use it for |
|---|---|---|
| `threads` (what `auto` picks when the host has threads and a clock) | One host thread per vCPU on the host clock, a device thread, shared RAM | Speed. Four parallel hashes finish in the time of one |
| `timeshare` | All vCPUs take turns on one thread, deterministic on the virtual clock | Reproducibility, and hosts with no threads (UEFI, wasm) |

With threads, x86 memory ordering is the emulator's job:

- Guest memory accesses are acquire/release atomics (`ldar`/`stlr` in the aarch64 JIT), never plain slice reads when another vCPU could be writing.
- Locked instructions and `xchg` are compare-exchange loops, and page-table accessed/dirty updates are compare-exchanges.
- Devices sit behind a reentrant lock. An exclusive section stops the world to flag a page as code (revoking every vCPU's write tags) and to run locked accesses no host atomic covers.
- Stale translations are invalidated through a log and per-vCPU attention words; the JIT cache is shared and patched only under its lock.
- MONITOR arms its line at MONITOR time. Arming at MWAIT loses a store that lands in between, which only threads can provoke.

Each vCPU has its own local APIC with IPIs of every addressing mode, INIT/SIPI start-up into real mode, MSI and I/O APIC and IOMMU destinations routed to the right APIC, and its own CPUID ids. Alpine 6.18 runs with 4 and with 64 vCPUs on threads: CPU hot-plug through sysfs, a fork storm, a pthread mmap/mprotect/munmap stress with spin locks, and four parallel hashes in the time of one.

The threaded tests only catch ordering bugs on a native aarch64 host; under Rosetta, x86 runs with hardware TSO and hides them. The project rule is to remove the thing under test and watch the test fail before trusting it.

## Platform and devices

| Device | Notes |
|---|---|
| ACPI | RSDP, XSDT, FADT, DSDT (with an AML builder), MADT, MCFG, HPET, IVRS; PM1 block and PM timer; S5 power-off ends the run with `ExitReason::Shutdown` |
| Interrupts | 8259 PIC pair, xAPIC and x2APIC local APICs (timer, TSC-deadline, IPIs, NMI with `iret` unblocking, CR8/TPR), 24-pin I/O APIC |
| Timers | PIT, HPET (three timers), APIC timer, RTC/CMOS, invariant TSC; one clock feeds them all |
| PCI | PCIe ECAM, MSI and MSI-X |
| AMD IOMMU (`--iommu`) | MMIO registers, command buffer, event log, device table, 1 to 6 level I/O page tables with large and arbitrary-size pages, IOTLB, DMA translation for every virtio function, interrupt remapping with 32- and 128-bit IRTEs, x2APIC IRTEs |
| Serial | 16550, always present; carries early boot text |
| virtio-console | `hvc0`, host terminal size forwarded and follows resizes |
| virtio-blk | Raw images, optional read-only, any number (`vda`, `vdb`, ...); embedders implement `BlockBackend` for their own storage |
| virtio-net | Several NICs, each with its own MAC. Links: user-mode NAT, a tiny loopback gateway, or frames over UDP |
| virtio-rng | On by default, feeds the guest's entropy pool from the host |

The user-mode NAT is a from-scratch network stack: DHCP, DNS, ICMP echo, UDP and TCP, the guest at 10.0.2.15 and the gateway at 10.0.2.2, which also reaches the host's loopback. What is verified through it: Fedora's `dnf` installing Xen, and DHCP for the Alpine netboot kernel through UEFI. The Alpine installer's `apk` could not fetch its index in the environment the UEFI checks ran in.

Up to 31 virtio devices per machine. RAM runs from 64 MiB to 1 TiB less the PCI hole; memory past 3 GiB maps from 4 GiB up, and pages are committed only when the guest touches them.

## Firmware and boot paths

| `--firmware` | What starts the machine |
|---|---|
| `uefi` (default) | The built-in UEFI firmware (`crates/uemulator-firmware`). It boots the disk's EFI system partition loader, or a kernel passed with `--kernel`, QEMU `-kernel` under OVMF style: the kernel needs an EFI stub, else `new` fails and tells you to use `none`. The initramfs is served over `LoadFile2` on `LINUX_EFI_INITRD_MEDIA_GUID`. Variables persist in the machine's `nvram.kv` through a key-value store the embedder can replace |
| `bios` | The library's BIOS services as host code (INT 10h, 13h with EDD, 15h with E820, 16h, 1Ah, the timer tick). The disk's boot sector runs in real mode at 0000:7C00 |
| `none` | The kernel is entered directly in 64-bit mode through the Linux boot protocol. What `--verify` and the library's fake-kernel tests use |

`uemulator advanced boot-firmware --firmware IMAGE` loads a raw image ending at 1 MiB and starts the CPU in real mode at F000:FFF0. A bzImage with no 64-bit entry boots in 32-bit protected mode.

Verified through UEFI: the Linux EFI stub, Debian's GRUB 2.12 from a GPT ESP, a variable written by Linux surviving a guest reset, Fedora's shim and GRUB, `xen.efi`.

## Hosts

| Host | Engine | Disk | Network | Threads |
|---|---|---|---|---|
| CLI on macOS | Interpreter + host JIT | yes | yes | yes |
| CLI on Linux and Windows | Same code, builds; not run by the author | untested | untested | untested |
| Browser (`crates/uemulator-web`) | Interpreter + WebAssembly JIT | no | no | one thread (the JIT's function table belongs to one instance) |
| UEFI application (`crates/uemulator-uefi`) | Interpreter, or the CPU itself through SVM/VMX | no | no | one thread |

**Browser.** `./hack/web.sh serve` builds to `wasm32-unknown-unknown` and serves on :8080. There is no wasm-bindgen; the module exports a small C ABI and imports `host_output`, `host_input`, `host_idle`, `host_jit_compile` and a few more. `uw_run` blocks, so it lives in a Web Worker and talks to the page through a `SharedArrayBuffer` ring, which is why the page needs COOP/COEP headers (`hack/web/serve.py` sends them). Guest time follows `performance.now()`. The page boots Alpine with `rdinit=/bin/sh` into a busybox shell.

**UEFI application.** An x86_64 or aarch64 `no_std` EFI binary (`./hack/uefi/boot.sh [x86_64|aarch64]`) that loads a kernel from its filesystem and runs it on the emulator. On x86_64 it can instead run the guest on the CPU itself: a minimal hypervisor where AMD NPT (or Intel EPT) is the MMU and there is no soft TLB and no translation. Exit cost (about 1 to 2 us each, dominated by port I/O to the UART) is expected to be the main overhead, but no speed measurement exists yet. AMD SVM boots Alpine to userspace under QEMU TCG in about 13 s including firmware; that run says nothing about speed, since TCG emulates the nested walk too.

**Freestanding embedding.** `cfg(freestanding)` (UEFI and wasm32) means no threads, no clock and no `std::fs`. The vCPU runs on the caller's thread inside `EmulatedMachine::wait`, and the host is reached through `MachineConfig::poll` and `host_clock`. A freestanding embedder gets threads only through `MachineConfig::thread_spawner`.

## Quick start

Requires a Rust toolchain (edition 2024) and, for `quick create`, network access to the distro's mirror.

```
# a named machine: machine.toml, kernel and initramfs, sparse disks
cargo run --release -- quick create test --install debian
cargo run --release -- quick boot test

# change the distro and keep the disk
cargo run --release -- quick reinstall test --install alpine

# more of everything
cargo run --release -- quick create big --install fedora --memory 8192 --cpus 8 --iommu

# a kernel you already have, with every option spelled out
cargo run --release -- advanced boot --kernel vmlinuz --initramfs initrd --memory 2048 --firmware none
```

Machines live in `~/.uemulator/machines/<name>/` (`UEMULATOR_HOME` moves `~/.uemulator`). `Ctrl-]` stops the machine; the CLI puts the terminal in raw mode.

New machines default to `--firmware uefi`, but the Alpine install-then-disk-boot cycle was checked under `--firmware bios`; a full install and disk boot under UEFI is not yet verified (see [limits](#status-and-limits)). Use `--firmware bios` on `quick create` for the checked path.

The first boot of a machine runs the distro's netboot installer: Debian's installer, Alpine to a login where you run `setup-alpine` and then `poweroff`, Fedora's Anaconda in text mode. After that, the same `quick boot` boots the disk through the machine's firmware.

## Embedding

`lib.rs` is the entire public API.

```rust
use uemulator::{EmulatedMachine, MachineConfig};

let mut machine = EmulatedMachine::new(
    MachineConfig::new()
        .kernel(bzimage)
        .initrd(initramfs)
        .memory_mib(1024)
        .cpus(4),
)?;
machine.console().on_output(|bytes| { /* guest console output */ });
let stop = machine.stop_handle();       // Send + Sync, callable from any thread
machine.start()?;
let reason = machine.wait();            // Shutdown, Reset, TripleFault, Stopped or Internal(_)
```

| Builder call | What it does |
|---|---|
| `kernel`, `initrd`, `cmdline`, `cmdline_append` | Boot a bzImage |
| `uefi_firmware`, `key_value_store` | Boot through the built-in firmware; keep its variables where you like |
| `firmware`, `boot_disk` | Real-mode firmware image, or a disk's boot sector through the BIOS |
| `memory_mib`, `cpus`, `smp`, `thread_spawner` | RAM, vCPUs, scheduling mode, host threads for freestanding builds |
| `disk(Disk)`, `net(mac)`, `console(ConsoleKind)`, `rng`, `iommu`, `acpi` | Devices. `Disk::from_backend` takes your own `BlockBackend` |
| `jit(JitMode)`, `clock(ClockMode)`, `verify_jit`, `debug` | Engine and time source |
| `reboot`, `unix_time`, `random_seed`, `entropy` | Reset policy, fixed RTC, deterministic seeds |
| `log`, `poll`, `host_clock` | Hooks: diagnostics, a poll callback for single-threaded hosts, a monotonic clock |

The library never touches stdio, never calls `process::exit`, holds no global mutable state and never panics on guest behavior. Guest faults are guest exceptions; an unimplemented opcode raises `#UD` and goes to the log hook. Handles (`ConsoleHandle`, `NetHandle`, `StopHandle`) are `Send + Sync + Clone`. On macOS, hardened-runtime binaries need the `com.apple.security.cs.allow-jit` entitlement for the JIT.

## CLI reference

Subcommands: `quick create|edit|boot|reinstall|update-artifacts|list|show|delete` and `advanced boot|boot-firmware|boot-uefi`.

### Machine options

Shared by `quick create|edit|boot` and `advanced boot` unless noted.

| Option | Meaning |
|---|---|
| `-m, --memory MIB` | 64 to 1047552. Default 2048; a Fedora machine defaults to 3072 because its installer image is downloaded into RAM |
| `--cpus N`, `--smp auto\|threads\|timeshare` | 1 to 64 vCPUs and how they run |
| `--disk PATH[,ro]` | Raw image as virtio-blk. Repeatable; guest order is command-line order |
| `--network virtio` | Adds a NIC. Repeatable. The link option after it picks the host side |
| `--network-link-user` | NAT to the host network: DHCP, DNS, ICMP echo, UDP, TCP |
| `--network-link-loop` | A 10.0.2.2 gateway that answers ARP and ping and forwards nothing |
| `--network-link-udp LOCAL,REMOTE` | Frames over a UDP socket pair. A NIC with no link has no host side |
| `--console virtio\|serial` | Default virtio (`hvc0`). `serial` is `ttyS0`, for kernels without a built-in `virtio_console`. Repeatable, serial at most once; the first is primary and gets the terminal |
| `--iommu` | Adds the AMD IOMMU |
| `--disable-jit`, `--disable-rng`, `--no-reboot` | Each takes `=false` to turn a saved setting back off |
| `--cmdline`, `--append` | Replace or extend the kernel command line; an empty value clears a saved one. They apply to the installer, not to a disk boot |
| `--firmware uefi\|bios\|none` | Default `uefi` (`quick create`, `quick edit`, `advanced boot`) |
| `--kernel`, `--initramfs` | `advanced boot` only |

### Run options

`quick boot` and the `advanced` commands: `--time UNIX` (fix the RTC; guest behavior must not depend on it), `--timeout SECS`, `--debug` (dump CPU state and scan guest RAM for the kernel log ring buffer on exit, plus the JIT's interpreter-fallback histogram), `--verify` (lockstep JIT check, single vCPU only).

### What a machine boots (`--boot-from`)

| Value | Behavior |
|---|---|
| `auto` (default) | The installer while the first 32 KiB of the first disk are all zeros, the disk afterwards, like hardware that tries its disks before the network |
| `installer` | Run the distro's netboot kernel and initramfs. `quick reinstall` sets this; `quick edit NAME --boot-from auto` goes back. For `uefi` machines it keeps the machine's `nvram.kv` |
| `disk` | The first disk through the machine's firmware. Kernel arguments come from the boot loader on the disk |

A machine file with no `firmware` key loads as `uefi`, so a machine installed under the old BIOS default needs `firmware = "bios"`.

### Managing devices on a saved machine

`quick create` falls back to an 8 GiB disk, one NAT NIC and a virtio console when given none. `quick edit` and `quick boot` append to what the machine has.

| Option | Meaning |
|---|---|
| `--create-disk MIB` (create, edit) | Sparse image in the machine's directory, attached after the existing disks |
| `--remove-disk\|--remove-network\|--remove-console INDEX` (edit, boot) | Counted from 0 in the order `create`, `edit` and `show` print them, against the list as it was before the command. A removed disk is detached, its image file stays |
| `quick delete NAME --yes` | Removes the machine's directory, including every `disk-<n>.img` in it, detached ones too. Disks attached with `--disk` are never touched |
| `quick update-artifacts`, `quick reinstall` | Refetch the distro's current kernel and initramfs; `reinstall` can switch distro and keeps the disk |

Replacing a disk is `quick edit test --remove-disk 0 --create-disk 32768`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | The machine ended by itself (power-off, or reset with `--no-reboot`), or was stopped with `Ctrl-]` or `--timeout` |
| 1 | Setup failed: bad arguments, unreadable file, download failure, rejected configuration |
| 2 | The machine failed: guest triple fault, or an emulator-internal error such as a `--verify` divergence |

A guest reset reboots the machine in place by default. `hlt` with interrupts off is a power-off.

## Architecture

```
MachineConfig ──► EmulatedMachine ──► one vCPU thread each (or one shared thread)
                                       │
   boot: bzImage / UEFI firmware /     ▼
         BIOS / raw image      ┌───────── run loop ────────────┐
                               │ irq + exit check → block → run │
                               └──────┬──────────────────┬─────┘
                               decode │                  │ execute
                  x86 bytes → uops ──►  interpreter | aarch64 | x86_64 | wasm JIT
                                       │                       (shared, position-independent cache)
        Cpu (regs, lazy flags, soft TLB, APIC, SVM state)
                                       │
          GuestRam ── device bus ── PIC PIT APIC IOAPIC HPET ACPI PCI IOMMU virtio
                                       │
                                  Clock (one time source)
```

| Crate | Role |
|---|---|
| `crates/uemulator` | The library. `no_std` + `alloc` when freestanding. Default feature `jit` pulls in `libc` on unix and `windows-sys` on Windows; `--no-default-features` has no dependencies |
| `crates/uemulator-cli` | `uemulator` binary (clap, ureq, toml; may use crates freely) |
| `crates/uemulator-firmware` | The UEFI firmware the library loads into a guest |
| `crates/uemulator-uefi` | `no_std` UEFI application, plus the SVM/VMX hypervisor path |
| `crates/uemulator-web` | wasm32 C ABI and the `www/` page |

## Testing and debugging

Over 5,000 `#[test]` functions live next to the code. They need no network, no cached kernels and nothing under `~/.uemulator`: a test that needs a guest assembles a tiny one (`testkit::FakeKernel` makes a valid bzImage around hand-assembled 64-bit code; `hack/asm.sh` prints the bytes).

```
cargo test --release --workspace
cargo test --release -p uemulator --no-default-features
cargo test --release --target wasm32-wasip1 -p uemulator --lib -- --test-threads=1 jit::   # wasm JIT under Node
cargo test --release -p uemulator --target x86_64-apple-darwin                              # x86_64 JIT under Rosetta
```

- **Differential corpus.** Anything the JIT compiles has a case through `assert_jit_matches_interpreter`. The same corpus (`jit/diff_*.rs`) runs on the aarch64, x86_64 and WebAssembly backends.
- **Bug fixes start with a failing test.**
- **Real-guest checks, not in the suite:** `--verify` over a boot, `hack/kvm/run.sh` (KVM inside the guest on the emulator's SVM), `hack/perf/unpack.sh`, `hack/wasm/boot.mjs`, `hack/uefi/boot.sh`.
- **Clippy is a gate.** `cargo clippy --release --all-targets` must print zero warnings, and so must the wasm32 and UEFI targets. Run `./hack/autofix.sh` before every commit.

## Status and limits

What is not done, or not checked, as of this writing:

- **x86_64 JIT hosts.** It has run on macOS x86_64 under Rosetta. Linux and Windows x86_64 builds compile and are untested. The aarch64 JIT has been developed on Apple Silicon.
- **Intel VMX** in the UEFI application is written to the SDM and reviewed, never run. AMD SVM is the verified path, and only under QEMU TCG.
- **UEFI application:** interpreter only, no disk, no network, console only. The SVM/VMX path has no MMIO, APIC, SMP, AVX or string port I/O.
- **Browser build:** no disk, no network, one thread, RAM 128 to 1024 MiB.
- **More than one vCPU** is refused with `--verify`, with raw firmware images (`boot-firmware`), with the BIOS disk boot (`--firmware bios` and `boot_disk`) and with `acpi(false)`. The built-in UEFI firmware does take several vCPUs (Fedora 44 boots on 4).
- **CPU:** no LA57, PKU, CET, PMU, IBS or VMX; address widths are 40/48 bits; the model has no speculation and makes no timing claims. Instruction latencies, cache behavior and performance counters are out of scope.
- **Guests not yet booted:** a Fedora Xen dom0 (the Xen entry installs but has not been booted), `xen.efi` through GRUB's chainloader, a complete Alpine install followed by a disk boot under UEFI (the installer's `apk` could not fetch its index there), Debian's installer under UEFI, and guests larger than the host's RAM.
- **Guest ISA coverage** is whatever Linux, glibc, busybox, the installers and Xen exercise, plus the unit corpus. A program that uses an instruction no test reaches can find a hole; unimplemented opcodes raise `#UD` and are logged.

## Design docs

`docs/plans/README.md` indexes them, and `docs/plans/10-milestones.md` has the status log, corrections included.

| File | Topic |
|---|---|
| `03-cpu.md` | CPU state, decoder, lazy flags, CPUID/MSR contract, SMP, AVX-512, SVM |
| `04-memory.md` | Physical layout, soft TLB, page walk, SMC tracking |
| `06-jit.md`, `20-jit-perf.md` | JIT design, chaining, performance work |
| `07-devices.md`, `11-storage-net-console.md` | Platform devices, virtio, NAT |
| `15-legacy-modes.md`, `17-bios.md` | Real and protected modes, the BIOS |
| `16-iommu.md` | AMD IOMMU |
| `18-xen-dom0.md` | The Xen bring-up, step by step |
| `19-uefi-firmware.md`, `12-uefi.md` | The firmware, and the UEFI application |
| `13-web.md` | Browser port and the WebAssembly JIT |
| `21-ryzen-7000-parity.md`, `22-smp-threads.md` | Planned CPU work, threaded SMP |

`../ext/qemu-uplink` is read-only reference for behavior (`docs/plans/09-qemu-map.md`). The Rust is written from scratch, not transliterated.

## License

GPL-2.0. See `LICENSE`. Agent notes for contributors are in `CLAUDE.md`.
