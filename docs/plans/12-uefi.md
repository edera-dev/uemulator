# UEFI application

`crates/uemulator-uefi` is a `no_std` UEFI application (`x86_64-unknown-uefi`, `aarch64-unknown-uefi`) that embeds the library, loads a bzImage and an initramfs from the boot filesystem and runs them on the emulated machine. The guest is always x86_64. On an aarch64 host build that is cross-architecture emulation; on an x86_64 build it is x86_64 on x86_64.

## Status

- Boots the Alpine netboot kernel and initramfs to userspace on both targets (the same stop as the CLI without a disk: `Mounting boot media: failed`). Verified: aarch64 under `qemu-system-aarch64 -accel hvf`, about 39 s with the interpreter. x86_64 under TCG on an arm64 host reaches the kernel banner (about 8 minutes, an interpreter inside an emulator) but was not run to userspace.
- Emulator path: interpreter only. The UEFI build has no `jit` feature: the executable-memory code in `sys.rs` is hosted-OS specific. To enable it: back `ExecMem` with `AllocatePages(EfiLoaderCode)` and flush the icache on aarch64, then run `--verify` equivalents. Firmware with NX memory protection may refuse to execute pool memory, so use pages typed as code.
- No disk or network device in the UEFI build (`Disk` uses `std::fs`). Console only.

## Library changes for `no_std`

The switch is `target_os = "uefi"`, not a feature, so `--no-default-features` on a hosted target is unchanged and still has no dependencies.

- `lib.rs` is `no_std` on UEFI and links `alloc`. `prelude.rs` re-exports the `alloc` names that `std` provides implicitly.
- `sync.rs` re-exports `std::sync` on hosted targets; on UEFI `Mutex` is a spinlock and `Condvar` does nothing (one thread).
- `math.rs` provides the float methods `core` lacks (`sin`, `sqrt`, `floor`, ...) through `libm`, under their `std` names.
- No threads: `EmulatedMachine::start` only marks the machine started and `wait` runs the vCPU on the calling thread.
- `MachineConfig::poll(|console, idle| ..)` is called every `EVENT_INTERVAL` instructions and, with `idle = true`, whenever the guest is halted with nothing scheduled. A single-threaded host uses it to move console bytes and to wait.
- The RTC starts at `unix_time` (the app reads the firmware clock); there is no `SystemTime`. `ClockMode::Realtime` is ignored, the clock stays virtual.
- `HashMap` and `HashSet` in the CPU became `BTreeMap` and `BTreeSet`.

## Application

Reads `\uemulator.cfg` from the filesystem it was loaded from, then the load options. Both use whitespace-separated words: `kernel=\vmlinuz`, `initrd=\initramfs` (or `initrd=none`), `mem=1024` (MiB), and everything after `--` is appended to the kernel command line. Lines starting with `#` are comments. Console output goes to the firmware text console, which QEMU `-nographic` attaches to the serial port, and input comes from the firmware text input. The emulator path uses the library's default virtio console and restarts the guest in place on a reset; guest power-off (`hlt` with interrupts off) shuts the machine down. The hardware-virtualization path keeps the serial console (it forwards only port I/O to `Platform`) and has no in-process reboot: a guest reset or power-off shuts the machine down.

`hack/uefi/boot.sh [x86_64|aarch64] [-- kernel words]` builds, assembles an ESP directory (`EFI/BOOT/BOOT*.EFI`, `vmlinuz`, `initramfs`, `uemulator.cfg`), and starts QEMU with KVM or HVF when the host architecture matches the target, TCG otherwise. See the script header for its environment variables.

## Hardware virtualization

On x86_64 the application can run the guest on the CPU itself instead of emulating it (`accel=auto|hv|emu`, `auto` by default: hardware when the CPU offers it). It is a minimal hypervisor, and the nested page tables (Intel EPT, AMD NPT) are the hardware MMU: the CPU walks the guest's own page tables and the nested table maps guest physical memory onto one contiguous host allocation with 2 MiB pages. There is no soft TLB, no translation and no interpreter on this path.

Layout, in `crates/uemulator-uefi/src/hv/`:

- `mod.rs`: guest RAM, the nested table, the TSC clock, and the event loop shared by both vendors. Per iteration it ticks the devices, injects a pending PIC interrupt (or asks for an interrupt-window exit), enters the guest, and handles the exit.
- `svm.rs` (AMD, VMCB, MSR and I/O permission maps, `VMLOAD`/`VMSAVE` for the syscall MSRs) and `vmx.rs` (Intel, VMCS, unrestricted guest, MSR bitmap, MSR load/store lists for the syscall MSRs). Both are behind the `Vcpu` trait.
- `arch.rs`: the two world-switch stubs in `global_asm!` (Windows x64 calling convention), CPU register helpers, and the FXSAVE save and restore around exits, because firmware code that runs between exits clobbers XMM.
- `cpuid.rs`, `msr.rs`: what the guest sees. The host identity with no APIC, MTRR, XSAVE, machine check or virtualization bits, as none of those are emulated; the invariant-TSC bit is set. Unknown MSRs raise `#GP`.

Devices are the library's `Platform` (UART, PIC, PIT, CMOS, reset ports). Port I/O exits go to it; its interrupts are injected as external interrupts. Time is the host TSC, calibrated once against the firmware's `Stall`; the guest is told its TSC frequency with `tsc_early_khz=` and `tsc=reliable`, because calibrating against the emulated PIT does not work when every port read is a VM exit.

Physical interrupts are intercepted so the firmware keeps working: on that exit the host briefly enables interrupts and the firmware's handler runs. The firmware's tick is set to 1 ms through the timer architectural protocol, which bounds how late the guest's timer interrupts can be. A halted guest waits in the host until a device interrupt is pending.

### Status

- AMD SVM with NPT: boots Alpine to userspace under `qemu-system-x86_64 -accel tcg -cpu max,svm=on` (`UEFI_ACCEL=hv hack/uefi/boot.sh x86_64`), about 13 s wall clock including firmware, against more than 8 minutes to reach the banner with the emulator inside the same TCG. That comparison measures nothing about the MMU, since TCG emulates the nested walk too.
- Intel VMX with EPT: written to the SDM and reviewed, but never run: this was developed on an arm64 host, where QEMU cannot provide VMX. Treat it as untested until it has booted on real hardware or under KVM with nested virtualization (`QEMU_ACCEL=kvm hack/uefi/boot.sh x86_64`, `-cpu host`).
- No speed measurement exists yet. Expect near-native execution once it runs on real VMX/SVM hardware, with exit cost (about 1-2 us each, dominated by port I/O to the UART) as the main overhead; the emulator path is the fallback and the reference.

### Not supported

- String port I/O (`INS`/`OUTS`), MMIO and the virtio devices (a guest access outside RAM ends the run with a message), APIC and SMP, XSAVE and AVX in the guest, TSC offsetting, and running under a hypervisor that does not expose SVM/VMX. The library's disk and network devices are not in the UEFI build either way.
- A host with `ExitBootServices` already called: the app keeps boot services, so the firmware keeps handling its own interrupts.
