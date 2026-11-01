# Real mode, protected mode, compatibility mode and virtual-8086

The emulator started as a long-mode-only CPU (the Linux 64-bit boot protocol enters in long mode and nothing leaves it). It now runs 16- and 32-bit code too, so a BIOS, a DOS-style real-mode program, a 32-bit protected-mode kernel, or a compatibility-mode (IA-32e with CS.L clear) user process all execute. Paging formats are in 04 ("Non-long-mode paging") and 03 ("Control-register rules outside long mode").

## The mode is a property of CS

`Cpu::code_mode()` (`cpu/modes.rs`) derives `CodeMode` from CS and EFER.LMA, and nothing else:

| `CodeMode` | CS | default operand / address size |
|---|---|---|
| `Long64` | CS.L set and EFER.LMA | 32 / 64 (REX, `rip`-relative addressing, 64-bit opcode map) |
| `Bits32` | CS.D set (protected or compatibility mode, or real mode after an unreal-mode load) | 32 / 32 |
| `Bits16` | CS.D clear: real mode, virtual-8086 mode, 16-bit protected mode | 16 / 16 |

Real mode and virtual-8086 mode are `Bits16` with CR0.PE clear or EFLAGS.VM set. Because the mode comes from CS's cached attributes, "unreal mode" (a real-mode CS or data segment that kept a 4 GiB limit or D bit from a protected-mode load) works without special cases.

`decode_in_mode(bytes, mode)` resolves prefixes for the mode: `Insn::opsize` is already 2, 4 or 8, and `addr16`/`addr32` say what the effective address size is. `40..4f` are `inc`/`dec` outside 64-bit mode; `c4`/`c5` are VEX only when ModRM.mod would be 3 (else `les`/`lds`); the 64-bit invalid opcode set applies only in `Long64`; `mod=0, rm=5` is an absolute `disp32` (RIP-relative only in `Long64`); 16-bit ModRM uses the `[bx+si]` table (`ADDRESS16_REGISTERS`) and sets `ss_default` for `bp`-based forms (also `ebp`/`esp` in 32-bit code). `Insn::mode` records it, and the decode cache hit requires `insn.mode == code_mode()` so the same bytes decode once per mode.

The JIT only compiles `Long64`: `Cpu::run` and `run_verify` call `step_block` only while `is_long64()`, and every instruction that can change CS, CR0, EFER or the page tables already ends a block (`ends_block`). The interpreter is the only implementation of everything in this document.

## Segmentation

`Cpu::segment_linear(index, offset)` returns the linear address: in `Long64` only FS/GS bases apply (as before); elsewhere `(base + offset) & 0xffff_ffff` after checking the limit (expand-down segments included) and, in protected mode, that the segment is present and readable. A violation is #SS(0) through SS and #GP(0) otherwise. `linear_address(i, a)` picks the override, else SS for `ss_default`, else DS, and is fallible; callers use `?`.

`segment_access(index, offset, size, write)` is the full check: limit for the whole extent of the access, the present and readable/writable rules in protected mode. `modrm_location` cannot know the size, so `linear_address` checks the first byte and remembers the operand (`Cpu::pending_operand`); `load_operand`, `store_operand` and the `moffs` moves then check the whole extent and the direction (`check_operand_extent`). String instructions check each iteration with its size.

Limits of this model: x87 and SSE/AVX memory operands, `bound`/`les` pairs and descriptor-table accesses take the first-byte check only, and A20 is always enabled.

Real mode builds a segment as `selector << 4` with a 64 KiB limit (`real_mode_segment`); a real-mode load keeps the old limit and attributes, a V86 load installs DPL 3 segments. `Cpu::load_segment_register` does this and falls to the descriptor-validating `load_data_seg` in protected mode. `fetch_decode` adds the CS base and checks the CS limit for the whole instruction. The stack is `stack_pointer()`/`set_stack_pointer()`: SP, ESP or RSP by SS.B (or `Long64`); `push`/`pop` take the mode-aware path (`push_legacy`) whenever `!is_long64()`.

## Instructions (`cpu/legacy.rs`)

`Cpu::exec` calls `exec_legacy` first outside `Long64`. It claims the opcodes whose behaviour depends on the mode and returns false for the rest, which then run the shared code that only depends on the resolved operand size. In `legacy.rs`: inc/dec r, push/pop (all forms, operand size from the mode), pusha/popa, enter/leave, near call/jmp/jcc/loop/jcxz/ret with the target truncated to the operand size and checked against the CS limit, push/pop of segment registers, `mov sreg`, `les/lds/lss/lfs/lgs`, daa/das/aaa/aas/aam/aad/salc, bound, arpl, into, xlat, the string instructions with 16/32-bit address registers and the ES base, ins/outs (also reachable in `Long64`: they were missing), far `call`/`jmp ptr16:16/32`, group 5 far forms, retf and iret.

Other places that changed: `mov cr/dr` is 32 bits wide outside IA-32e mode; `lgdt/lidt/sgdt/sidt` use a 24-bit base with a 16-bit operand size, 32-bit base with a 32-bit one; `lldt/ltr` use 8-byte descriptors outside IA-32e (`read_desc` no longer reads a second quadword there) and accept a 32-bit TSS; `fnstenv/fldenv/fnsave/frstor` have the 14/94-byte 16-bit formats; `sysenter`/`sysexit` exist in protected mode only; `syscall` from compatibility mode enters at CSTAR and `sysretl` returns to compatibility mode; `lar`/`lsl` exist (the CPUID identity is AMD, where they are #UD under IA-32e, so SEP is advertised and the instructions are #UD with EFER.LMA).

## Interrupts, exceptions and control transfers (`cpu/legacy_flow.rs`, `legacy_task.rs`)

`Cpu::deliver` dispatches on EFER.LMA: with it set (including compatibility mode) the 16-byte gates of `exception.rs` apply; otherwise `deliver_legacy`:

- Real mode: the vector table at IDTR.base (4-byte entries), FLAGS/CS/IP pushed as words, IF/TF/AC/RF cleared.
- Protected mode: 8-byte IDT gates (interrupt, trap, task; 16- or 32-bit), DPL check for software interrupts, #NP for a missing gate, target code segment checks, same-level and inner-level (TSS ring stack: SS0/ESP0.. of a 32-bit TSS, SP0/SS0.. of a 16-bit one, validated: #TS), the V86 frame (GS, FS, DS, ES, SS, ESP, EFLAGS, CS, EIP, error code) with the segment registers nulled afterwards, EFLAGS.VM cleared. A fault while pushing restores SS, ESP and CPL.
- Task gates switch tasks.
- `iret` (all of real mode, V86 with the IOPL check, protected same-level and outer-level, the return to virtual-8086 mode, and NT = task return), `retf`, far `jmp`/`call` through code segments, call gates (stack switch and parameter copy) and TSS/task gates.
- Hardware task switch (`task_switch`): 32-bit TSS only; saves the registers, links back (`call`, task gate), sets and clears busy bits and NT, loads CR3, LDT and the segment registers with #TS on failure, sets CR0.TS. A 16-bit TSS cannot be switched to.
- In IA-32e mode a far `jmp`/`call` to a code descriptor with L=1 and D=0 enters 64-bit mode, and to one with L=0 enters compatibility mode (the way a boot loader leaves 32-bit code).

## Boot without Linux

`MachineConfig::firmware(image)` and `uemulator advanced boot-firmware --firmware IMAGE` place the image so that it ends at 1 MiB and start the CPU at the reset state (`Cpu::reset_to_real_mode`: real mode, paging and long mode off, F000:FFF0). There is no ROM or chipset: the image sits in RAM and the 4 GiB-top alias of the reset vector is not mapped (CS's base is set so that execution starts at F000:FFF0 = 0xFFFF0, which every BIOS image continues from with a far jump). The reboot hook is off for firmware images: a guest reset ends the run with `ExitReason::Reset`.

## Booting a 32-bit Linux kernel

A bzImage without `XLF_KERNEL_64` is no longer rejected: `boot::load` places it exactly like a 64-bit one but returns `BootMode::Protected32` with `rip` at the first byte of the protected-mode image (`startup_32`), a boot GDT whose `__BOOT_CS` (0x10) is a 32-bit code descriptor, and `Cpu::enter_boot_protected_mode` (CR0.PE only, flat segments, paging and long mode off, ESI = `boot_params`). Verified end to end with Alpine's x86 `vmlinuz-lts` (6.18, non-PAE i686): the kernel enables 32-bit paging itself, takes timer and device interrupts through 32-bit gates, runs busybox in ring 3 via `int 0x80`/`sysenter`, and reaches a shell on the serial console (`uemulator advanced boot --firmware none --kernel vmlinuz-lts --initramfs initramfs-lts --console serial -m 512 --cmdline "console=ttyS0 rdinit=/bin/sh"`). Everything runs on the interpreter (the JIT is 64-bit-only), so expect it to be several times slower than a 64-bit guest.

## Validation by KVM

`hack/kvm/run.sh` (see 03, "SVM") runs Linux KVM inside the emulated guest and starts vCPUs in each mode this document covers, over SVM and nested paging; all pass. That is the check that the segment, paging and interrupt rules agree with what a hypervisor sets up from `KVM_SET_SREGS`.

## Tests

`cpu/legacy_tests.rs` holds 49 tests, generated from assembly snippets (`BITS=16 hack/asm.sh`, `BITS=32 hack/asm.sh`) and run on real-mode and flat 32-bit harnesses (`Harness::real`, `legacy_support::protected32`): 16-bit addressing and wrap, segment bases and limits, stack widths, real-mode interrupts, far jumps/calls, string ops, BCD, 32-bit gates, error codes, ring 3 to ring 0 and back through the TSS, V86 entry and exit, sysenter/sysexit, call gates with parameters, task switch and nested-task return, LDT, 16-bit code in protected mode, real mode to protected mode to long mode with the boot code a real loader runs, compatibility mode and the far jump back to 64-bit mode. `machine/tests.rs` boots a real-mode firmware image on the full machine under both JIT modes.

SVM guests may be in any of these modes (see 03, "SVM"); the VMRUN state swap also flushes the decode cache, since the same bytes decode differently per mode.

## Not implemented

- Everything 32-bit runs on the interpreter only; a 32-bit JIT is not planned.
- Virtual-8086 extensions (CR4.VME/PVI, the interrupt redirection bitmap, VIF/VIP), SMM, and the whole-extent segment check for x87/SSE operands.
- 16-bit TSS switching, legacy-mode (non-IA-32e) `syscall`, 64-bit call gates and task switches in IA-32e mode (not architectural), the debug registers' breakpoints.
