# Boot

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
- The decompressor's early page-fault/IDT handling and `rdrand`/`rdtsc` KASLR entropy (`nokaslr` skips most).
- `initrd_addr_max` and the `ext_ramdisk_image` fields when the initrd sits above 4 GiB (not needed at 512 MiB).

## Reset

Port 0x64←0xFE, port 0xCF9 reset bits, and triple fault → machine exit (`ExitReason::Reset`/`TripleFault`). Kernel `reboot=` defaults then terminate automated runs; `panic=-1` makes panics reboot immediately.
