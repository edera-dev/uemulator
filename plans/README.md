# uemulator plans

Index. Read in order; each file ends with open questions where the answer needs checking against the QEMU tree or the kernel docs.

| file | topic |
|------|-------|
| 00-overview.md | goals, non-goals, architecture, crate layout, key decisions |
| 01-api.md | public embedding API, console, exit reasons, threading |
| 02-boot.md | the target kernel, direct 64-bit boot, boot_params, e820 |
| 03-cpu.md | x86_64 state, decoder, lazy flags, exceptions, CPUID/MSR contract |
| 04-memory.md | physical memory, bus, soft TLB, page walk, SMC tracking |
| 05-ir-interp.md | micro-op IR, TB cache, interpreter |
| 06-jit.md | W^X handling, aarch64 backend, probe/fallback, chaining |
| 07-devices.md | serial, PIC, PIT, CMOS, clock, reset, gotchas |
| 08-testing.md | unit, differential, boot tests |
| 09-qemu-map.md | which QEMU files to read for which subsystem; fork notes |
| 10-milestones.md | ordered milestones with observable exit criteria |
| 11-storage-net-console.md | configurable RAM, virtio-pci disk/net/console (resize) |
| 12-uefi.md | UEFI application, `no_std` port, hardware-MMU analysis |
| 13-web.md | WebAssembly port |
| 14-avx-and-virtualization.md | designs for AVX/AVX2/FMA, SVM and VMX (not implemented) |
| 15-legacy-modes.md | real mode, protected mode, compatibility mode, virtual-8086, task switching, firmware boot |
| 16-iommu.md | AMD IOMMU: registers, command buffer, event log, device table, I/O page tables, interrupt remapping, IVRS |
| 17-bios.md | BIOS services as host code, booting a disk's boot sector |
| 18-xen-dom0.md | Fedora Xen dom0 through UEFI: ACPI, IOMMU wiring, SMP and what follows; status per step |
| 18-xen-dom0.md | plan for a Xen dom0 on a modern AMD machine |
| 19-uefi-firmware.md | the UEFI firmware, key-value store, handoff page |
| 20-jit-perf.md | JIT performance: direct chaining, inline integer ops, SIMD fast paths, measured against the initramfs unpack |
| 22-smp-threads.md | one host thread per vCPU: shared RAM and devices, the exclusive section, code invalidation across threads, atomics and x86 ordering, the thread-spawn provider |
| 23-guest-io-perf.md | guest-visible performance (syscalls, memcpy/memset, disk, network): the in-guest benchmarks in `hack/perf/guest/`, what bounded each and what changed |
| 24-display-input.md | graphical console: virtio-gpu (2D), virtio-input keyboard and tablet, the boot framebuffer and the firmware's GOP and screen console, the display/input handles, `--display`, the CLI's web viewer |
| 21-ryzen-7000-parity.md | feature parity with a Ryzen 5 7000-class (Zen 4) part from public documentation: instruction sets, LA57, PKU, CET, SVM extras, PerfMonV2, CPU profile |

Rules for implementers: `../ext/qemu-uplink` is read-only reference. Behaviour and algorithms are studied there; code is written fresh in Rust, not transliterated. Zero crate dependencies.
