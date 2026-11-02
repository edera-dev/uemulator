# uemulator documentation

The design plans in [`plans/`](plans/README.md) are the source of truth for how uemulator works and what it does. Each one starts with a status line. [`plans/10-milestones.md`](plans/10-milestones.md) is the log of what is done and how each piece was verified. Plans are numbered in the order they were written. This page groups them by topic.

## Architecture overview

- [00-overview.md](plans/00-overview.md): goals, architecture, crate layout and the decisions that shaped the design.
- [01-api.md](plans/01-api.md): the public embedding API (`MachineConfig`, `EmulatedMachine`, console, exit reasons, threading) and the CLI.
- [10-milestones.md](plans/10-milestones.md): status log and the original milestones with their exit criteria.
- [09-qemu-map.md](plans/09-qemu-map.md): which QEMU sources were used as reference for which subsystem.

## CPU and JIT

- [03-cpu.md](plans/03-cpu.md): CPU state, decoder, lazy flags, exactness rules, the CPUID/MSR contract, instruction extensions, AVX-512, SVM, time-shared SMP.
- [05-ir-interp.md](plans/05-ir-interp.md): the interpreter and run loop; the micro-op IR as originally designed.
- [06-jit.md](plans/06-jit.md): JIT design, covering the aarch64, x86_64 and WebAssembly backends, W^X, chaining and the shared code cache.
- [14-avx-and-virtualization.md](plans/14-avx-and-virtualization.md): AVX and SVM pointers, and the VMX design that has not been implemented.
- [15-legacy-modes.md](plans/15-legacy-modes.md): real mode, 16/32-bit protected mode, compatibility mode, virtual-8086 and task switching.
- [22-smp-threads.md](plans/22-smp-threads.md): one host thread per vCPU, covering shared state, the exclusive section, atomics and x86 memory ordering.
- [21-ryzen-7000-parity.md](plans/21-ryzen-7000-parity.md): proposal for feature parity with a Zen 4 Ryzen 7000-class part.

## Memory

- [04-memory.md](plans/04-memory.md): physical layout, soft TLB, page walk, self-modifying-code tracking, atomicity.

## Devices

- [07-devices.md](plans/07-devices.md): clock, legacy PC devices, LAPIC and IOAPIC, ACPI, HPET, PCIe ECAM, MTRRs.
- [16-iommu.md](plans/16-iommu.md): the AMD IOMMU, covering translation, interrupt remapping and IVRS.
- [11-storage-net-console.md](plans/11-storage-net-console.md): configurable RAM and the virtio-pci disk, network and console.
- [24-display-input.md](plans/24-display-input.md): the graphical console: virtio-gpu, virtio-input, the firmware's GOP framebuffer, the display and input handles and the CLI's web viewer.
- [25-gpu-3d.md](plans/25-gpu-3d.md): guest 3D through virtio-gpu: the device side and `GpuLink` (done), virgl on WebGL2 (recommended) against Venus on WebGPU, the render worker and milestones.

## Firmware and boot

- [02-boot.md](plans/02-boot.md): direct 64-bit Linux boot, `boot_params` and e820.
- [19-uefi-firmware.md](plans/19-uefi-firmware.md): the built-in UEFI firmware, its handoff page and key-value store.
- [17-bios.md](plans/17-bios.md): BIOS services as host code, and booting a disk's boot sector.
- [18-xen-dom0.md](plans/18-xen-dom0.md): bringing up a Xen dom0 on the emulated AMD machine, step by step.

## Storage and network

- [11-storage-net-console.md](plans/11-storage-net-console.md): virtio-blk, virtio-net and user-mode NAT.
- [spec/uemulator-disk-format.md](spec/uemulator-disk-format.md): the udisk image format (sparse, copy-on-write, servable over HTTP).

## Other hosts

- [13-web.md](plans/13-web.md): the WebAssembly port and the browser page.
- [12-uefi.md](plans/12-uefi.md): uemulator as a UEFI application, including the hardware-virtualization path.

## Performance

- [20-jit-perf.md](plans/20-jit-perf.md): JIT performance work, covering direct chaining, inline lowering and SIMD fast paths, with measurements.
- [23-guest-io-perf.md](plans/23-guest-io-perf.md): performance as the guest sees it (syscalls, memory, disk and network), with in-guest benchmarks.

## Testing

- [08-testing.md](plans/08-testing.md): unit, instruction and differential (JIT against interpreter) tests.
- [`CLAUDE.md`](../CLAUDE.md): the project's test rules and the real-guest checks to run after touching particular subsystems.

## Specifications

- [spec/uemulator-disk-format.md](spec/uemulator-disk-format.md): udisk, version 1.

## Implementation plans

Step-by-step task plans for single changes, kept for the decisions they record.

- [superpowers/plans/2026-10-28-uefi-default-kernel-boot.md](superpowers/plans/2026-10-28-uefi-default-kernel-boot.md): making UEFI the default firmware, and booting `--kernel` through it.
