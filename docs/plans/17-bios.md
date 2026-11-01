# BIOS boot of a disk

`MachineConfig::boot_disk(true)` boots the first disk the way a PC with a BIOS does, so the disk's own boot loader (syslinux, GRUB) loads the kernel. There is no firmware image: the BIOS is host code in `cpu/bios.rs`. CLI: `quick boot` with `--boot-from auto|disk` (`docs/plans/01-api.md`).

## Shape

- `Cpu::bios_install` builds what POST leaves behind: an IVT whose 256 vectors point at 8-byte stubs in the ROM segment (`F000:0100 + 8 * vector`), the BIOS data area, an EBDA stub, the BIOS date and model bytes, the 1.44 MB floppy table (`int 1Eh`) and a fixed-disk table (`int 41h`), the PIC (IRQ0 at INT 08h, slave at 70h, only IRQ0 and the cascade unmasked) and the PIT (channel 0, mode 3, count 0: 18.2 Hz).
- A stub is `[sti;] out 0xB2, al; iret`. `Cpu::port_io` sends the `out` to `bios_trap`, which derives the vector from the stub's address, reads the caller's registers, does the service in Rust and writes results back. Flags go into the stacked FLAGS word, so the stub's `iret` returns the carry flag the way a real `retf 2` BIOS would.
- `bios_boot` reads sector 0 of disk 0 to 0000:7C00 and starts it with DL = 0x80, interrupts on, every other register zero. A missing `55 AA` signature (or no disk) prints "No bootable device." and halts with IF clear, which this library reports as `ExitReason::Shutdown`.
- Real-mode interrupt delivery is the CPU's own (`deliver_real`); nothing here changes the CPU.

## Services

| vector | what |
|---|---|
| 08h | timer: BDA tick counter (wraps at 0x1800B0 with the midnight flag), EOI, then `int 1Ch` |
| 10h | text only, to the 16550: teletype, write char/string, cursor (an ANSI `ESC[r;cH` only when the caller moves it), clear, mode/cursor queries; no VGA, EGA or VESA (calls leave registers alone, which callers read as "absent") |
| 11h, 12h | equipment word and base memory from the BDA |
| 13h | drives 80h+ in disk order: reset, status, CHS read/write (255 heads x 63 sectors), parameters (08h), type (15h), EDD install check (41h), extended read/write (42h/43h, segmented or flat buffer), extended parameters (48h) |
| 15h | E820 (the same three-entry map as the direct boot), E801, 88h, 87h block copy, A20 gate (always open, RAM is flat), delay; everything else fails with AH = 86h |
| 16h | keys come from the UART's receive side; with none, 00h/10h halt until the next interrupt and retry. Terminal escape sequences become key words (`cpu/bios_keys.rs`): arrows, Home/End, PgUp/PgDn, Ins/Del and F1-F12 return AL=0 and the scan code in AH (GRUB maps AH only when AL is 0); Esc, Enter, Backspace (and DEL) and Tab carry their scan codes; other bytes are AL alone. The UART paces input, so a sequence can take several calls to arrive; an unfinished one counts as Escape once the line has held nothing for 50 ms |
| 18h, 19h | no boot device; bootstrap again |
| 1Ah | tick count and the RTC time and date from CMOS |

Not provided: VGA text memory (B8000 is plain RAM), ACPI/SMBIOS/MP tables (the kernel runs without them), PCI BIOS, floppies, a keyboard controller (port 64h reads as all ones: a loader that falls back to enabling A20 through it hangs, which is why the A20 test must see flat memory), a second boot device, and in-place reboot (a guest reset ends the run, as for firmware images). The JIT only runs 64-bit code, so the real-mode and 32-bit parts are interpreted, and `verify_jit` is refused.

## Why the stubs look like this

- **`sti` first.** Real BIOS services start with `sti`. syslinux's menu polls `int 16h` with interrupts masked and reads its clock from the timer; without the window each call opens, the timer interrupt is never taken and the menu's countdown never ends.
- **`int 1Ch` from INT 08h.** syslinux keeps its own clock by hooking `int 1Ch`, the BIOS's per-tick user hook. A timer handler that only bumps the BDA counter leaves that clock at zero.
- **Exceptions end the run.** CPU exceptions 0 to 7 have no handler in a real-mode BIOS either; an `iret` stub would re-execute the faulting instruction forever, so the vector is reported as a triple fault.

## Verification

`cpu/bios.rs` has unit tests for every service (including the failure statuses) and `machine/tests.rs` boots tiny boot sectors on a full machine, one of them waiting for the PIT's IRQ0 to reach the BIOS tick. Not a test, needs a disk you supply: an Alpine `setup-alpine` install (syslinux 6.04 MBR, `ldlinux.sys`, `menu.c32`) boots through the MBR, the menu countdown, the kernel and initramfs load (INT 13h, INT 15h), the kernel's real-mode setup and the 32-bit entry to `alpine login:` on hvc0 and ttyS0 (`quick boot` on a machine whose disk the installer wrote). The clock follows the host (`--verify` alone makes it virtual), so the menu's 1-second countdown takes one second.

## Open questions

- GRUB (Debian) and the Fedora installs are untested: their loaders may want INT 10h or INT 15h calls that are stubbed.
- A graphical or colour menu (vesamenu) needs VBE and a framebuffer and will not work.
