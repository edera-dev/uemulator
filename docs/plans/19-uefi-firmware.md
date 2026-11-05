# UEFI firmware

**Status:** implemented.

`crates/uemulator-firmware` is a small `no_std` UEFI 2.9 firmware (not EDK2) for `x86_64-unknown-none`. The library starts it with `MachineConfig::uefi_firmware(image)`; the CLI embeds it (`crates/uemulator-cli/build.rs` builds it into its own target dir, so `rustup target add x86_64-unknown-none` is needed) and runs it with `advanced boot-uefi` or a `quick` machine with `firmware = "uefi"`.

## Image and start state

The image is a flat binary linked at 1 MiB (`link.ld`). Header (little endian): `"UEMUFW64"`, version u32 = 1, entry offset u32, footprint u64 (image plus zero-initialised data). The library copies the image to `FIRMWARE_BASE = 0x100000`, builds the same GDT and 4 GiB identity map as the direct 64-bit Linux boot, and starts the CPU at the entry in long mode, interrupts off, `rdi` = the handoff page. A guest reset loads everything again (the key-value store survives).

The firmware's first instructions enable SSE (CR4.OSFXSR; loaded applications use it), install an IDT that prints exceptions to the 16550, and remap the PIC so a PIT one-shot can wake `hlt` (idle waits cost no host time).

Then it programs the MTRRs as a PC firmware leaves them (`mtrr.rs`): default type write-back with the fixed ranges on, 0xA0000..0xBFFFF uncacheable through the fixed ranges, and variable ranges making the MMIO hole uncacheable: naturally aligned blocks covering `[S, 4 GiB)` for the `S` between the top of the RAM below 4 GiB and the lowest MMIO (handoff MMIO regions and ECAM) that needs the fewest blocks, which on this machine is always one (3 GiB with 3 GiB or more of RAM, else 2 GiB or 3 GiB). If the hole needed more blocks than MTRRcap has, they stay disabled. Linux then prints an `MTRR map` instead of `MTRRs disabled by BIOS`. MTRRs are per logical processor and INIT keeps them, so with several vCPUs `processors.rs` broadcasts INIT and start-up to a real-mode stub at 0x8000 (inside the reserved low pages) that performs the same MSR writes from a table, counts itself done and spins on `pause` (not `cli; hlt`, which ends the machine), then broadcasts INIT again so the APs wait for the OS's SIPI, and leaves the boot CPU's APIC software-enabled in virtual-wire mode (LINT0 ExtINT, LINT1 masked NMI) as SeaBIOS and OVMF do: putting the SVR back to disabled masks LINT0, which cut off the PIT interrupt that ends `idle_wait`, and every timed wait (the boot countdown, GRUB's timeout) hung with more than one CPU; without it Linux warns "your CPUs had inconsistent ... MTRR settings". The vCPU count comes from CPUID 0x8000_0008 ECX. The direct boot (`--firmware none`) has no firmware and leaves the MTRRs disabled, as before.

## Handoff page (0x7000, 0x388 bytes, version 4)

| offset | field |
|---|---|
| 0x00 | magic `"UEMUHAND"`; 0x08 version u32; 0x0c size u32 |
| 0x10 / 0x18 | firmware base / footprint u64 |
| 0x20 | RAM top u64 (end of the flat RAM buffer; with RAM above 4 GiB that is past the PCI hole, which the region list leaves out) |
| 0x28 | ACPI RSDP address (0: none); 0x30 / 0x38 ACPI table region base / size (reported as ACPI reclaim) |
| 0x40 | boot unix time u64 |
| 0x48 / 0x4a / 0x4c / 0x4e | serial port, key-value base port (0: no store), its port count, PCI config address port (0xcf8) u16 |
| 0x50 | PCI ECAM base (0: none, config is via 0xcf8) |
| 0x58 / 0x5c | RAM region count, MMIO region count u32 |
| 0x60 | 32 random bytes (splitmix64 of `random_seed`) |
| 0x80 | up to 16 RAM regions: base u64, size u64, e820 type u32, pad (24 bytes) |
| 0x200 | up to 16 MMIO regions: base u64, size u64 (virtio BARs, IOAPIC, LAPIC) |
| 0x300 | TSC frequency in Hz (the guest TSC counts nanoseconds) |
| 0x308 / 0x310 | kernel base / size u64 (0 size: no kernel was given) |
| 0x318 / 0x320 | initramfs base / size u64 (0 size: none) |
| 0x328 / 0x330 | kernel command line base / size u64 (UTF-8, no NUL) |
| 0x338 / 0x340 | boot framebuffer base / size u64 (0 size: no display; reserved in the RAM regions) |
| 0x348 / 0x34c / 0x350 | framebuffer width, height, pixels per scan line u32 (4 bytes per pixel, blue, green, red, unused; `docs/plans/24-display-input.md`) |
| 0x358 / 0x360 | hypervisor base / size u64 (0 size: none; Xen's Multiboot2 ELF, already decompressed, or `xen.efi`) |
| 0x368 / 0x370 | hypervisor command line base / size u64 (UTF-8, no NUL) |
| 0x378 | flags u32: bit 0 enforce Secure Boot (`docs/plans/26-tpm-secure-boot.md`) |
| 0x380 | TPM CRB base u64 (0: no TPM) |

`crates/uemulator/src/boot/uefi.rs` writes it, `crates/uemulator-firmware/src/handoff.rs` reads it; both have tests. Other agents fill `rsdp`/ACPI region through `FirmwareParams` when ACPI tables exist.

## Key-value store

`trait KeyValueStore: Send { get, set, delete, list(prefix) }` (byte strings), `MemoryKeyValueStore`, and `FileKeyValueStore` (hosted only). File format: magic `"UEMUKV01"` then records `length u32 | crc32 u32 | payload` (payload: op u8 (1 set, 2 delete), key length u16, key, value). Every `set`/`delete` appends one record and syncs. Opening replays the log (a short or bad-checksum record ends it: a torn write), then rewrites it compacted through `path.tmp`, fsync and atomic rename; the log is also rewritten when it holds 1 MiB of dead bytes. A file without the magic is refused, not overwritten. Limits: keys 1 KiB, values 256 KiB.

Guest device, port I/O at 0x580..0x58f, 32-bit registers: +0 command (w) / status (r: 0 ok, 1 not found, 2 too large, 3 I/O error, 4 bad command), +4 data (w appends up to 4 bytes to the selected input, r pops 4 output bytes), +8 length (w: bytes the selected input will carry; r: output length), +0xc ident `"UKV1"`. Commands: 1 select key buffer, 2 select aux buffer (both clear it), 3 get (key -> output), 4 set (key, aux = value), 5 delete, 6 next (output = the smallest key greater than the key buffer that starts with the aux buffer). Absent device: the ports read all ones.

Variables: key `var/<guid>/<name utf8>`, value = attributes u32 + data. Non-volatile variables go to the store (RAM if there is none); the rest live in a 192 KiB arena in the firmware image, which stays reserved.

## What the firmware implements

- Boot services: memory pages/pool with a real map and map key (top-down allocation), events/timers (polled; notifications run at TPL transitions), the full protocol database (install/reinstall/uninstall incl. the variadic multiple forms, Handle/Open/Close/LocateHandle(Buffer)/LocateProtocol/LocateDevicePath/ProtocolsPerHandle/RegisterProtocolNotify), LoadImage (file by device path, short-form paths search every volume, or a source buffer) with a PE32+ loader and relocations, StartImage/Exit (stack-switching setjmp), UnloadImage, ExitBootServices (map key check, event group, virtio reset, PIC mask), Stall, watchdog, CRC32, CopyMem/SetMem, monotonic count, configuration tables.
- Runtime services: GetTime (CMOS RTC), SetTime/wakeup/capsules unsupported, variables with NV/BS/RT attributes and append, SecureBoot / SetupMode / VendorKeys / AuditMode / DeployedMode / the `*Default` keys / SignatureSupport (26) / PlatformLangCodes / OsIndicationsSupported / BootOptionSupport read-only, QueryVariableInfo, ResetSystem (0xCF9; shutdown halts with IF clear), SetVirtualAddressMap with ConvertPointer (table pointers are rewritten; the whole firmware is one runtime-code region so code finds its data), GetNextHighMonotonicCount. After ExitBootServices allocations come from a 1 MiB arena in the image that each runtime call rewinds.
- Protocols: LoadedImage, LoadedImageDevicePath, DevicePath, DevicePathToText, DevicePathUtilities (instances unsupported), SimpleTextInput/InputEx/Output on the 16550 (ANSI cursor/clear/colour, key decoding for arrows, Home/End, F-keys, one 80x25 mode) and, on a machine with a display, also drawn on the boot framebuffer and fed by the virtio keyboard, Graphics Output on the boot framebuffer (`gop.rs`, `text_screen.rs`, `virtio_input.rs`; `docs/plans/24-display-input.md`), BlockIO and DiskIO for every virtio-blk disk, GPT/MBR partition and El Torito EFI boot image (`eltorito.rs`, hosted tests: an ISO 9660 image with no partition table gets its catalog's first bootable entry for platform 0xEF as a volume with a `CDROM(entry,rba,size)` node, sized by the entry's 512-byte sector count, else the image's FAT boot sector, else the rest of the disk; GPT and MBR are tried first, as EDK2 does, so a hybrid ISO's ESP is not found twice), SimpleFileSystem (read-only FAT12/16/32 with long names; writes return `WRITE_PROTECTED`), RNG.
- Tables: ACPI 2.0/1.0 (when the handoff has an RSDP), SMBIOS 3.0 (`smbios.rs`, hosted tests): BIOS (UEFI and virtual-machine extension bits), system, chassis, processor (signature, features and brand from CPUID, Zen family, one socket with the cores and threads CPUID 0x8000_0008/0x8000_001E report, speed from the TSC rate), the physical memory array with one DIMM holding all the RAM (Linux: `Memory slots populated: 1/1`), one mapped-address structure per contiguous RAM range (the legacy window below 1 MiB counts as covered), system boot information and the end marker. The space reserved for it in the image fits 16 RAM regions. Memory types: runtime code (the image), BS data, ACPI reclaim, MMIO for BARs/IOAPIC/LAPIC.
- Kernel payload (`kernel_boot.rs`): when the library was given a kernel (`MachineConfig::kernel` with `uefi_firmware`, CLI `--firmware uefi`, the default), it copies kernel, initramfs and command line to the top of the RAM below the PCI hole (initramfs highest, each page aligned) and fills the handoff fields above; `memory::init` keeps those ranges allocated as boot services data. The boot manager tries that kernel before anything else, like OVMF's QemuKernelLoader: `LoadImage` from the buffer (it must be an EFI-stub bzImage; the library refuses others up front with a hint to use `firmware none`), the command line as UTF-16 load options, the initramfs behind a `LoadFile2` protocol on a handle with the `LINUX_EFI_INITRD_MEDIA_GUID` vendor-media device path, which the Linux stub reads. If it fails or returns the firmware halts: a requested kernel never falls through to a disk. A guest reset places the payload again. With a hypervisor in the payload (`MachineConfig::hypervisor`) the firmware boots that instead, with the kernel, initramfs and command line as its dom0: `xen.efi` with its configuration, kernel and ramdisk behind Edera's `XEN_EFI_*_MEDIA` `LoadFile2` loaders (`media.rs`, `hypervisor_boot.rs`), Xen's Multiboot2 ELF as GRUB's `multiboot2` loads it on UEFI (relocated, entered at its EFI amd64 entry with boot services kept); details in `docs/plans/18-xen-dom0.md`, "Xen as the machine's hypervisor".
- Boot manager (no kernel given; `boot.rs`, with the testable parts in `boot_options.rs`, `boot_menu.rs`, `load_option.rs`):
  - Discovery, as EDK2's BDS: every volume with `\EFI\BOOT\BOOTX64.EFI` gets a `Boot####` ("UEFI VirtIO disk N, partition P", or "UEFI VirtIO disk N, CD boot image", lowest free number, appended to `BootOrder`) whose optional data is EDK2's auto-created GUID `8108ac4e-...`; marked entries whose volume is gone are deleted; entries an OS wrote are never touched, and a loader one of them already boots gets no second entry. The marker is not passed to the image as load options.
  - `OsIndications` bit 0 (`BOOT_TO_FW_UI`, advertised in `OsIndicationsSupported`) opens the menu once and is cleared. `BootOptionSupport` is `APP` only (no `Key####`).
  - `Timeout` (u16 seconds, absent = 0, 0xFFFF = straight to the menu) is a countdown on the console; Esc, F2 or F12 opens the menu, Enter boots now. With 0, a key typed ahead still opens it.
  - Then `BootNext` (deleted before it is tried), then the `BootOrder` entries that are not `CATEGORY_APP`, then removable-media loaders the order does not cover; `BootCurrent` is set. A path starting at an `HD()` node (efibootmgr's short form) is expanded through the partition with that number and signature (UEFI 2.9 3.1.2) before `LoadImage`; file-only short forms are searched on every volume.
  - The menu lists the non-hidden `BootOrder` entries and "Reset the machine"; arrows, Home/End, Enter, Esc (boot in the usual order). When nothing in the order starts the menu opens instead of halting; with no entries at all the firmware still halts with "No bootable device."
  - The CLI writes `OsIndications`/`Timeout` into a machine's nvram with `--boot-menu` / `--boot-timeout SECS` on `quick boot` and `advanced boot-uefi`.
- Console input: the 16550 delivers host input one byte per character time (the library paces it for Xen), so the key decoder waits up to `FOLLOWING_BYTE_TIMEOUT_NS` (50 ms) for each byte after the first of a key; without that, an arrow key read as Escape, `[`, `A`.

Secure Boot, the authenticated writes to PK, KEK and the databases, and `EFI_TCG2_PROTOCOL` with measured boot are in `docs/plans/26-tpm-secure-boot.md`.

Not implemented: USB/network/PCI I/O protocols, file writes, private time-based authenticated variables, EFI_MEMORY_ATTRIBUTES_TABLE, SetTime, multi-instance device paths, key-notify callbacks. Verified: the Linux EFI stub, Debian GRUB 2.12, Alpine GRUB 2.14 booting Xen 4.21 through `multiboot2` (see 18) and Xen's `xen.efi` started by the boot manager (it reads its `xen.cfg`, kernel and initramfs from the ESP), and Edera's Xen 4.22 booted as the machine's hypervisor both ways (`xen.gz` by the firmware's own Multiboot2 loader, `xen.efi` through the media loaders). Known failing: `xen.efi` started by GRUB's `chainloader` (`Couldn't obtain the File System Protocol Interface`, the `DeviceHandle` GRUB sets). Fedora's shim 16.1 and GRUB 2.12 boot through `\EFI\BOOT\BOOTX64.EFI` (shim reads `LoadedImage.FilePath` as text and expects the file nodes only, without the device nodes; the firmware strips them, `LoadedImageDevicePath` keeps the full path) and the `fbx64`/`BOOTX64.CSV` fallback creates the boot entry. With `--secure-boot` Debian 13's shim, GRUB and kernel boot (26); MOK enrolment is not exercised. Fedora 44's Server netinst ISO (GPT hybrid) attached read-only boots its GRUB 2.12 menu and the installer to Anaconda (2047 MiB or more; 1 GiB is too little for its initramfs). netboot.xyz's ISO (El Torito only, no partition table) boots its iPXE from the catalog's EFI image.

## Tools

- `hack/uefi/make-disk.py OUT --size MiB DEST=SRC...` builds a GPT disk with a FAT32 ESP (pure python).
- `UEMULATOR_FIRMWARE_TRACE=1 cargo build -p uemulator-cli` compiles in start-up trace messages; `UEMULATOR_FIRMWARE_ELF=1 cargo build -p uemulator-firmware --target x86_64-unknown-none --target-dir X` keeps an ELF for `objdump`.
- `advanced boot-uefi --disk D --nvram F [--boot-loader PATH --boot-options TEXT]` writes `Boot0000`/`BootOrder` into F first.

## Verification (not tests; needs downloaded artifacts)

- Alpine `vmlinuz-virt` as `\EFI\BOOT\BOOTX64.EFI` with `initrd=\initramfs-virt console=ttyS0` in `Boot0000`: kernel prints `efi: EFI v2.9 by uemulator`, SMBIOS 3.0, RNG, reaches userspace, `/sys/firmware/efi` exists. With the matching `efivarfs.ko` appended to the initramfs, a variable written from Linux survives a guest reset (read back from the nvram file after the second boot).
- Debian `grub-efi-amd64-signed` 2.12 (`grubx64.efi.signed` as BOOTX64.EFI, `grub.cfg` in `\EFI\debian`): menu, countdown, `linux`/`initrd` from the ESP, Alpine to userspace.
