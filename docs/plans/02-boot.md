# Boot

**Status:** implemented: the direct 64-bit boot (`--firmware none`). Booting through firmware is in [19-uefi-firmware.md](19-uefi-firmware.md) and [17-bios.md](17-bios.md).

## The target kernel (inspected, cached at ~/.cache/uemulator/linux)

- `Linux 6.12.107+deb13-amd64`, bzImage, 12,142,528 bytes
- setup header magic `HdrS`, protocol **2.15**, `setup_sects` 39
- `loadflags` 0x01 (LOADED_HIGH); `xloadflags` 0x7f → XLF_KERNEL_64 set (64-bit entry at +0x200 usable), can be loaded above 4G, EFI handover bits (ignored)
- `relocatable_kernel` 1, `kernel_alignment` 0x200000, `min_alignment` 21, `pref_address` 0x1000000
- `init_size` 0x37ef000 (~58 MB decompressed footprint), `syssize` 0xb8f200
- compressed payload is **zstd** (magic 28 b5 2f fd)
- `cmdline_size` max 2047
- initrd.gz: ~39 MB (Debian installer)

Re-run the header parse in `boot.rs` tests so the numbers in this file stay verified.

## Direct 64-bit entry (no firmware)

Follow the kernel's 64-bit boot protocol (Documentation/arch/x86/boot.rst, "64-bit Boot Protocol") and QEMU's `x86_load_linux` for the boot_params layout:

1. Copy setup header (0x1F1..) into a zeroed `boot_params` (4 KiB) and patch fields: `type_of_loader=0xFF`, `loadflags |= LOADED_HIGH|KEEP_SEGMENTS?` (check), `cmd_line_ptr`, `ramdisk_image/size`, `hdr.heap_end_ptr` unused for 64-bit.
2. Copy protected-mode kernel (file offset `(setup_sects+1)*512`) to `0x1000000`.
3. cmdline NUL-terminated at e.g. `0x20000`; boot_params at `0x7000`; initrd at top of low RAM (below 4 GiB, page aligned, below `initrd_addr_max` header field) — 512 MiB default RAM keeps it simple.
4. e820 table in `boot_params.e820_table`, `e820_entries`:
   - `0x0 – 0x9FC00` usable (kernel needs low memory for trampoline), reserved `0x9FC00 – 0x100000`, usable `0x100000 – top`.
5. Initial CPU state: long mode active (CR0.PE|PG, CR4.PAE, EFER.LME|LMA), identity-mapped page tables covering the first 4 GiB (2 MiB pages) placed at e.g. `0x9000`; GDT with `__BOOT_CS` at selector 0x10 (64-bit code) and `__BOOT_DS` at 0x18 per protocol; `RIP = 0x1000200`; `RSI = &boot_params`; interrupts masked; `CLD`; other GPRs zero.
6. No ACPI, no MP table → `acpi=off` (superseded for `EmulatedMachine`: `boot::load_with_acpi` publishes tables, see 07; `Platform` hosts still say `acpi=off`). No LAPIC/IOAPIC in CPUID → kernel uses 8259 PIC + PIT.

## Things to verify during implementation

- Exact `__BOOT_CS/DS` selectors for 6.12 (`arch/x86/include/asm/segment.h`) — decompressor reloads its own GDT early, but the first `lretq`/segment loads must not fault.
- Whether the 6.12 decompressor still far-returns into compat mode for 5-level paging probing (`paging_prepare` / `trampoline`). If so the CPU must support switching CS.L=0 and back with correct decoding. Hence the mode-parameterized decoder.
- The decompressor's early page-fault/IDT handling and `rdrand`/`rdtsc` KASLR entropy: verified, see KASLR below.
- `initrd_addr_max` and the `ext_ramdisk_image` fields when the initrd sits above 4 GiB (not needed at 512 MiB).

## KASLR

The default command lines leave KASLR on. Checked on the Alpine 6.18 virt kernel with 2 and 8 GiB, 1 and 4 vCPUs, the JIT and `--verify`, reading `Kernel code` in `/proc/iomem` and `_text` in `/proc/kallsyms`: physical and virtual base differ on every boot, physical slots above 4 GiB included.

- Direct entry: the decompressor picks the physical slot from the zero page's e820 table, stepping around the initrd, command line and setup_data, and maps a slot above our 4 GiB identity map itself through its page-fault handler. Nothing in the zero page opts in: `KASLR_FLAG` in `loadflags` is an output the decompressor sets for the kernel.
- UEFI with a kernel, or GRUB from the disk: the EFI stub places the kernel with `efi_random_alloc`, which needs `EFI_RNG_PROTOCOL` (`uemulator-firmware/src/rng.rs`) and AllocatePages at an address it chose.
- The UEFI application: both its emulator path and its hardware path (`hack/uefi/boot.sh`, SVM under QEMU TCG) boot with a different base each time; the hardware path's entropy is the host CPU's own.
- BIOS disk boot: not booted with KASLR (no BIOS-installed disk was at hand). Its `int 15h` E820 is the same `e820_map` as direct entry and the BIOS keeps nothing above 1 MiB, so the decompressor sees the same usable RAM; what differs is the 32-bit entry, whose page tables are the decompressor's own.
- Entropy is `rdrand`/`rdseed` (seeded by `MachineConfig::random_seed`), the firmware's RNG (seeded from the same value through the handoff page) and the TSC. The CLI and the web page draw the seed from the host's CSPRNG on every boot; with a fixed seed on the virtual clock a boot is reproducible, which is what `--verify` relies on.
- Nothing in the emulator depends on where the kernel is: JIT blocks are keyed by physical address and position independent (06-jit.md), and the `--debug` log scanner looks for the printk buffer by content (`Cpu::kernel_log_matches`).
- Cost: none measurable. Host time from launch to the banner (firmware plus decompressor) and to `Run /init`, Alpine, 2 GiB, JIT, median of 5: 0.43 s and 1.59 s with KASLR against 0.44 s and 1.65 s without on direct entry; 0.43 s and 1.62 s against 0.43 s and 1.63 s through UEFI.
- Embedders other than the CLI and the web page must pass `random_seed` themselves; the UEFI application seeds from the platform's EFI_RNG_PROTOCOL. Without a seed the virtual clock makes every boot choose the same base (on the realtime clock only TSC jitter varies it).

## Reset

Port 0x64←0xFE, port 0xCF9 reset bits, and triple fault → machine exit (`ExitReason::Reset`/`TripleFault`). Kernel `reboot=` defaults then terminate automated runs; `panic=-1` makes panics reboot immediately.
