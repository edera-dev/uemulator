# QEMU reference map

`../ext/qemu-uplink` (branch `xtcg/next`, QEMU 11.1.50 + uplink fork, ~16.5k commits past the merge-base with the latest upstream tag). Read-only. Study behavior and algorithms; write Rust from scratch (code is GPLv2 like the reference, but not copied; QEMU's own AGENTS.md restricts AI output to local experiments, so nothing here is intended for upstream).

| subsystem | read |
|-----------|------|
| machine model | `hw/i386/microvm.c` (minimal devices, direct boot) — not `pc.c` |
| bzImage boot, boot_params, e820 | `hw/i386/x86-common.c` (`x86_load_linux`), `include/standard-headers/asm-x86/bootparam.h` |
| decode tables | `target/i386/tcg/decode-new.c.inc`, `decode-new.h` |
| uop emission | `target/i386/tcg/emit.c.inc`, `translate.c` |
| lazy flags | `target/i386/tcg/cc_helper.c`, `cc_helper_template.h.inc` |
| segments, exceptions, interrupts | `seg_helper.c`, `system/seg_helper.c`, `excp_helper.c`, `system/excp_helper.c` (page walk) |
| MSRs, CPUID | `system/misc_helper.c`, `target/i386/cpu.c` (`x86_cpu_...` cpuid), fork identity commit "The fork's CPU identity" |
| x87 / SSE | `fpu_helper.c`, `ops_sse.h`, `fpu/` softfloat |
| soft TLB | `accel/tcg/cputlb.c` |
| TB lifecycle | `accel/tcg/translate-all.c`, `tb-maint.c`, `cpu-exec.c`, `translator.c` |
| IR + codegen | `tcg/tcg.c`, `tcg/optimize.c`, `tcg/aarch64/tcg-target.c.inc`, `docs/devel/tcg-ops.rst` |
| 16550 | `hw/char/serial.c` |
| PIC, PIT, RTC | `hw/intc/i8259.c`, `hw/timer/i8254.c`, `hw/rtc/mc146818rtc.c` |
| port 0x61 | `hw/audio/pcspk.c` / `hw/i386/pc.c` (port92, kbd controller `hw/input/pckbd.c`) |
| LAPIC (later) | `hw/intc/apic.c`, `apic_common.c` |

## Fork triage (`git log $(git merge-base HEAD <latest upstream tag>)..HEAD`)

- **Correctness fixes to adopt by design** (check each against our implementation): "fix immediate stack adjustment for RET and RETF", "Fix FXCH to unconditionally clear C1", "Mark MOVNTI as not valid with prefixes 66/F2/F3", "tcg/optimize: Fix expansion/simplification of deposit", "Fix fold_multiply2", "tcg/aarch64: Correct register MIN/MAX instructions", "fix EFLAGS reads in TCG plugin callbacks".
- **SMP with threads** (`plans/22-smp-threads.md`): MTTCG's ideas, written fresh. `start_exclusive`/`end_exclusive` and `cpu_exec_step_atomic` (`Cpu::with_exclusive`, the retry of locked instructions), `tlb_protect_code` (revoking write tags for a page that gains code, here inside the exclusive section), the big lock around devices, and `TCG_GUEST_DEFAULT_MO` (x86 is TSO, so the aarch64 backend needs acquire loads and release stores; QEMU refuses MTTCG for that pairing by default, here it is on).
- **Perf features, JIT phase notes only**: x-tblookup, x-tblayout, x-iofast, x-edgesys, x-slowthunk, x-ssefp, lazy INC/DEC CC_OP, x-vexzero/regcache/superblock.
- **Not applicable**: x-hyperv, x-user, x-versal/zynqmp/zcu/aio/blockdev, `[DEV]` stats commits.
- **CPU identity**: AMD Zen 5 (family 0x1a model 0x02 stepping 1, EPYC-Turin; see 03 "CPUID/MSR contract"; QEMU's `EPYC-Turin` says model 0 stepping 0), no hypervisor bit. Read its leaves, then subtract (03).

## Open questions to answer from the tree

- Exact `lcall`/`lret` compat-mode transitions the decompressor needs (`seg_helper.c`).
- How `microvm` presents `boot_params.e820` and whether it writes `hardware_subarch`.
- i8254 mode-3 square wave edge timing needed by `pit_calibrate_tsc`.
