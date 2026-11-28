# Ryzen 5 7000-class feature parity

**Status:** partially implemented. Sections A to E (all but the host-visible `VIRT_SPEC_CTRL` of C) and step 10 are done, on the Zen 5 Turin model that replaced Genoa ("Zen 5 Turin model" below); the selectable CPU profile and the Ryzen 7000 identity of section F (steps 3 and 9) are not: `CpuProfile` (`cpu/profile.rs`) carries only the processor count, threads per core and brand, and there is no `MachineConfig::cpu_model` or `--cpu-model`.

Goal: the emulated CPU **has every feature a Ryzen 5 7000-series part (Zen 4: Raphael 7600/7600X, Phoenix 7640U/7640HS, family 19h) advertises**, so a guest, a hypervisor or a test suite that looks for a feature finds it and it works. This is a feature list, not a fidelity project:

- Not the exact core. Instruction latency, speculation, cache and TLB replacement, performance-counter values, undefined-flag quirks beyond what the manuals specify: out of scope.
- Not the exact platform. ACPI, PCI topology and fabric registers stay the generic virtual PC (`docs/plans/07-devices.md`). (Written for the Ryzen goal; with the Turin model the platform did take Turin's identity, "Platform identity" below.)
- No access to the real part. Everything comes from public documents; nothing here needs a capture from silicon. Where a public source is silent or two disagree, we follow the manual and record the choice.

The identity was an EPYC 9004 Genoa (family 19h model 11h) when this plan was written and is now an EPYC 9005 Turin (Zen 5, family 1Ah model 02h; "Zen 5 Turin model" below, `docs/plans/03-cpu.md`, "CPUID/MSR contract"). Either shares the Zen 4 ISA baseline (AVX2, AVX-512 with VNNI/BF16/IFMA/VBMI/BITALG/VPOPCNTDQ, SHA, GFNI, VAES, VPCLMULQDQ, SMEP/SMAP/UMIP, FSGSBASE, PCID/INVPCID, RDPID, RDSEED, XSAVEC, SVM with nested paging). The work is the delta.

## Sources, in order of authority

1. AMD64 Architecture Programmer's Manual vols. 1-3 (CPUID function and feature-bit appendix, SVM, paging, system instructions) and vol. 2's per-feature chapters.
2. The Processor Programming Reference for family 19h models 60h-6Fh (Raphael) and 70h-7Fh (Phoenix): MSR lists, CPUID values, supported features.
3. The consumers' own tables, which say what a guest does with each bit: Linux `arch/x86/include/asm/cpufeatures.h` and `arch/x86/kernel/cpu/amd.c`, Xen `cpufeatureset.h`, QEMU `target/i386/cpu.c` (`EPYC-Genoa`, `EPYC-Turin` as a feature cross-check, not a source to copy: write Rust fresh, `docs/plans/09-qemu-map.md`).
4. Published CPUID dumps of Ryzen 7000 parts, as a cross-check of the manuals, never as the only evidence.

## Feature delta

Status is relative to `cpu/cpuid.rs` today. "Believed present" means I expect Zen 4 client to have it from the sources above and each item is confirmed against sources 1-2 before work starts; an item that turns out absent is dropped here, not implemented.

Rules from `CLAUDE.md` hold for all of it: a CPUID bit goes on in the same commit as the instructions or MSRs behind it and their tests; any new uop gets an interpreter implementation and a test first, then the JIT with `assert_jit_matches_interpreter`.

### A. Small instruction sets (each independent, days not weeks)

All done, on the Turin model ("Small Zen extensions" below).

| feature | CPUID | what it needs |
|---|---|---|
| SSE4A | 8000_0001 ECX 6 | `extrq`, `insertq`, `movntss`, `movntsd` (decoder, interpreter, JIT) |
| MMXEXT (AMD extensions to MMX/SSE) | 8000_0001 EDX 22 | the integer MMX extensions on MMX registers (`pshufw`, `pavgb`, `pmaxsw`, `maskmovq`, `movntq`, `pextrw`...), most of which the SSE2 forms already implement on XMM |
| FFXSR | 8000_0001 EDX 25 | `EFER.FFXSR` (bit 14): `fxsave`/`fxrstor` skip XMM in 64-bit mode at CPL 0; EFER writes currently reject the bit |
| MWAITX / MONITORX | 8000_0001 ECX 29 | the two instructions with a TSC-based timeout; builds on the existing MONITOR/MWAIT arm |
| RDPRU | 8000_0008 EBX 4 | `rdpru` reading MPERF/APERF (fixed ratio 1; the MSRs `0xE7`/`0xE8` come with it) |
| CR8_LEGACY | 8000_0001 ECX 4 | `lock mov cr8` in 32-bit code, a 3-line decoder rule |
| OSVW | 8000_0001 ECX 9 | `OSVW_ID_LENGTH` / `OSVW_STATUS` MSRs (`0xC0010140/1`): storage only |
| INVLPGB / TLBSYNC | 8000_0008 EBX 3, EDX[15:0] = 7 | done (`cpu/invlpgb.rs`): threaded vCPUs are stopped and invalidated in place, time-shared ones flush their whole TLB before they next run; TLBSYNC has nothing left to wait for |

### B. Architectural state that touches paging and XSAVE (the real work)

1. **LA57, 5-level paging.** Done for the Turin model (03, "5-level paging and protection keys"): `PagingMode` kept one `Long` variant and `walk_wide` takes its top level from CR4.LA57; the JIT needed no change, since its probe only matches tags `translate` validated. The original plan: `CR4.LA57`, a fifth page-walk level (PML5), 57-bit canonical checks (the `translate` canonical test, branch targets, `lea`-based address generation, `invlpg`, `invpcid`), the 57-bit virtual-address field in 0x8000_0008, LA57 in the SVM nested-paging guest state, and the JIT's inline canonical check. `PagingMode` gains a `Long5` variant; `walk_wide` already loops over levels, so the level count is the main parameter. Large-page cache (`docs/plans/04-memory.md`) needs the shift for the new level only if 1 GiB+ leaves at the top are allowed (they are not at PML5).
2. **PKU/OSPKE, protection keys for user pages.** Done for the Turin model (03, same section), with the second option below: restricted tags stay out of the soft TLB and a PKRU change flushes it. The original plan: `CR4.PKE`, `PKRU`, `rdpkru`/`wrpkru`, PTE bits 62:59, page-fault error-code bit 5 (PK), XSAVE component 9, leaf 7 ECX bits for PKU/OSPKE. The walker evaluates the key against `PKRU` per access (user pages only; supervisor keys are Intel-only), so cached TLB entries must carry or recheck the key: either fold the key into the permission tags at fill time (and flush on `wrpkru`, which is architecturally not a TLB flush, so this must be re-evaluated per access) or keep PK pages out of the write/read tags like the SMAP case in `translate`.
3. **CET shadow stacks (user and supervisor).** `CR4.CET`, `U_CET`/`S_CET`/`PL0_SSP..PL3_SSP`/`ISST_ADDR` MSRs, `#CP` (vector 21), shadow-stack pages (`W=0, D=1` encoding), `incssp`, `rdssp`, `saveprevssp`, `rstorssp`, `wrss`, `setssbsy`, `clrssbsy`, shadow-stack pushes on `call` and checks on `ret`/`iret`/far transfers, the supervisor shadow-stack token protocol, XSS components 11 and 12. Zen 4 has shadow stacks but not indirect-branch tracking (the `endbr64` NOP encoding already decodes). This is the largest item in the plan and last in this section; Linux uses it for user shadow stacks (6.6+) and the SVM guest state carries `S_CET` and `SSP` fields. **Done** (`cpu/cet.rs`, see "Zen 5 Turin model"); the JIT compiles near call and ret with the shadow-stack push and compare inline and leaves the rest to the interpreter.

### C. Virtualisation (SVM) completeness

`svm::SVM_FEATURES` today: nested paging, next-RIP save, VMCB clean bits, flush-by-ASID (all in their "reload and flush everything" form), LBR virtualization, SVM lock, the TSC ratio MSR, the pause filter and its threshold, decode assists, virtual VMLOAD/VMSAVE, vGIF, vNMI, SPEC_CTRL virtualization, SVME_ADDR_CHK, AVIC and x2AVIC; 32768 ASIDs; the NMI, INIT, SMI (never raised) and task-switch intercepts (`docs/plans/03-cpu.md`, "SVM"). Zen 4 reports more in 0x8000_000A EDX; audit the list against the APM and add what a hypervisor will use, each with a guest test in `hack/kvm/run.sh`'s style or a fake kernel:
- The host-visible `VIRT_SPEC_CTRL` remains. Done: GMET, ROGPT, HOST_MCE_OVERRIDE and TlbiCtl (the guest INVLPGB enable; `docs/plans/03-cpu.md`, "SVM"), TSC rate MSR, pause filter (count and threshold), SVM lock and `SVM_KEY`, LBR virtualization, vGIF, vNMI, `V_VMSAVE_VMLOAD`, decode assists, SPEC_CTRL virtualization, SVME_ADDR_CHK, VM_CR.R_INIT.
- AVIC/x2AVIC are done (`cpu/svm_avic.rs`; `docs/plans/03-cpu.md`, "SVM"), with the IOMMU's guest virtual APIC mode posting into the backing page (`docs/plans/16-iommu.md`) and the broadcast shorthands and x2APIC logical IPIs left to the hypervisor.
- SEV, SEV-ES, SEV-SNP: not on client Ryzen; not planned.

### D. Counters and monitoring a guest can program

All done ("Performance monitoring" and IBS below; MPERF/APERF and RAPL with the small Zen extensions). The plan as written:

- **PerfMonV2** (`PERFCTR_CORE` `8000_0001 ECX 23`, `PERFMON_V2` `8000_0022 EAX 0`): six core counters at `0xC0010200`, global control/status MSRs (`0xC0000300..`), counting a small set of events the emulator can count exactly: retired instructions, retired branches, retired taken branches, cycles (= `icount` at the model's rate, `NS_PER_INSN`). Keep the legacy four-counter set that is already there. Overflow raises the LVT performance-counter interrupt, which makes the NMI watchdog and `perf stat` work.
- `PERFCTR_NB`/`PERFCTR_L2`: counters that read zero (present, never counting); only if a guest probes them and fails otherwise.
- **LBR v2** (`8000_0022 EAX 1`), **IBS** (`8000_0001 ECX 10`, `8000_001B`) and the extended APIC space (`EXTAPIC`, `8000_0001 ECX 3`) it hangs its LVT on: moderate; all three are done (below), IBS with a sampling model on the retired-instruction count.
- MPERF/APERF (`0xE7/0xE8`) at ratio 1, RAPL energy MSRs reading a monotonic fake counter: cheap, and they stop `amd_pstate`/`turbostat`/`powertop` probes failing.

### E. Security and system bits that are cheap and honest

Done: NULL_SEL_CLR_BASE, NO_NESTED_DATA_BP and AUTOIBRS are advertised (the speculation-control table below). The departures the last bullet kept are gone too: the speculation bits are in the AMD spelling only and `TSA_*_NO` are Turin's own bits; there are still no hypervisor leaves. The plan as written:

Add only where the behaviour is real in this implementation and testable:
- `NullSelectorClearsBase` (`8000_0021 EAX 6`): loading a null selector into a data segment register clears the base in 64-bit mode; confirm the current behaviour and set the bit.
- `NoNestedDataBp` (`EAX 0`) and `AutoIBRS` (`EAX 8`): `EFER.AIBRSE` accepted and inert (no speculation to restrict), exactly the stance already taken for `IBRS`.
- Keep the declared departures from `docs/plans/03-cpu.md`: the Intel spelling of the speculation-control leaf, `TSA_*_NO`, no hypervisor leaves. (TSC-deadline and `SRSO_NO` were on this list and are gone with the Turin model.) Decide on removing the Intel spellings separately; the right test is what Linux and Xen do with the AMD-only spelling.

### F. Identity (last, small)

- `CpuProfile { signature, brand, logical processors, cache sizes, feature set }` in `cpu/profile.rs`, Genoa expressed as one profile with no behaviour change, then a Ryzen 5 7000-class profile (model per the PPR for the chosen SKU, brand `AMD Ryzen 5 7600X 6-Core Processor` style, 6 cores 12 threads, 32 KiB L1, 1 MiB L2 per core, 32 MiB L3) selected by `MachineConfig::cpu_model`; CLI `--cpu-model`. The feature bits come from sections A-E, so the profile only chooses which are on; a profile cannot enable a bit the emulator lacks (checked by a test that walks every profile and asserts each advertised bit has a registered implementation).
- Done so far: `CpuProfile` (`cpu/profile.rs`) with the logical processor count, threads per core and brand, built from `MachineConfig::cpus` and `MachineConfig::smt`. Signature, caches and feature set are still the one Turin model's.
- Flip the default only after the guest check below passes: family and model drive Linux and Xen erratum, mitigation and perf-driver selection, so the identity change is the riskiest one-line change in the list.

## Zen 5 Turin model (the current identity)

The emulated CPU presents a 2024 AMD server core, an EPYC 9005 "Turin" (Zen 5), so that Linux sees hardware that needs no software mitigation and the default command line can go without `mitigations=off`. `CpuProfile` holds only the processor count, threads per core and brand (section F's selectable profile is still a plan), so Turin replaced Genoa outright: the Genoa leaves are in the history of `cpu/cpuid.rs`, not selectable.

Identity and sources:
- Signature `0x00B00F21`: family 1Ah, model 02h, stepping 1, the Turin (classic, Zen 5) die. Linux's `bsp_init_amd` names family 1Ah models 00h-2Fh, 40h-4Fh and 60h-7Fh Zen 5, and `init_amd_zen5` keeps RDSEED only when model, stepping and microcode revision match `zen5_rdseed_microcode`, whose entry for 1Ah/02h/1 is revision `0x0b00215a`; PATCH_LEVEL (MSR 0x8b) reports exactly that. Model 11h stepping 0 in the same table is Turin Dense (Zen 5c). QEMU's `EPYC-Turin` uses model 0 stepping 0, which is in no kernel table and would cost RDSEED, so only its feature list and cache sizes were used as a cross-check (`../ext/qemu-uplink/target/i386/cpu.c`).
- Brand: the smallest EPYC 9005 part with at least as many cores as the machine has (`cpu/profile.rs`: 9015, 9115, 9255, 9335, 9365, 9455, 9535, 9565, 9655, 9755, then the Turin Dense 9825, 9845, 9965 above 128 cores, whose model 11h the signature does not follow), so the guest never sees more cores than the name says, as in a VM on that part. It was QEMU's "AMD EPYC-Turin Processor" while the core count was fixed nowhere.
- Caches per core complex die: L1D 48 KiB 12-way, L1I 32 KiB 8-way, L2 1 MiB 16-way per core, L3 32 MiB 16-way shared by 16 threads (QEMU's `epyc_turin_cache_info` agrees). TLBs from public Zen 5 microarchitecture descriptions, not a CPUID dump: L1 DTLB 96, L1 ITLB 64, L2 ITLB 2048; the L2 DTLB keeps 3072 because the 4096 Zen 5 is said to have does not fit the 12-bit field of leaf 0x8000_0006. The field widths are asserted at compile time.
- Topology: leaf 0x8000_0026 (core, complex, die, socket) is new; Linux prefers it to 0xB. The complex level is the threads of the 8 cores behind one L3, the same number leaf 0x8000_001D reports, so Linux's tile and LLC domains agree; the max extended leaf is 0x8000_0026 with 0x22-0x25 zero. SMT2 is on by default (`MachineConfig::smt`): an even vCPU count is cores of two threads. Tests walk every processor of 1-384 vCPUs with SMT on and off.
- CET shadow stacks (leaf 7 ECX 7, `cpu/cet.rs`, `cpu/cet_tests.rs`); no IBT (leaf 7 EDX 20 clear, the IBT bits of U_CET/S_CET #GP), as on every Zen. CR4.CET (needs CR0.WP), U_CET/S_CET/PL0-3_SSP/ISST_ADDR, #CP (vector 21, contributory), shadow-stack pages (read-only and dirty leaf under writable levels, #PF bit 6), near and far call/ret, iret and event delivery in every protected mode (frames of CS, linear return address and SSP; supervisor tokens claimed from PLn_SSP or the interrupt SSP table and released on the way out), syscall/sysret/sysenter/sysexit (PL3_SSP), incssp, rdssp (a nop without a shadow stack), saveprevssp, rstorssp, wrss, wruss, setssbsy, clrssbsy, XSAVES components 11 and 12 behind IA32_XSS, and S_CET/SSP/ISST_ADDR in the VMCB. Not modelled: task switches and SMM with shadow stacks, and nested-page write checks for shadow-stack stores (`Cpu::shadow_stack_translate`). Always offered, as on Turin. The JIT compiles near call and ret with the shadow-stack push and compare inline and leaves the rest to the interpreter (`docs/plans/06-jit.md`, "CET shadow stacks").
- VME and PVI (leaf 1 EDX 1, `cpu/vme.rs`): VIF/VIP, cli/sti/pushf/popf/iret on VIF in virtual-8086 mode below IOPL 3, the TSS interrupt redirection bitmap for `int n`, and cli/sti on VIF at CPL 3 with PVI.

Speculation-control bits (all in the AMD spelling; the Intel leaf 7 EDX bits and `IA32_ARCH_CAPABILITIES`/`IA32_FLUSH_CMD` are gone, as on real AMD parts):

| bit | leaf | behaviour behind it | what Linux does with it |
|---|---|---|---|
| IBPB, IBRS, STIBP, SSBD | 0x8000_0008 EBX 12/14/15/24 | PRED_CMD bit 0, SPEC_CTRL bits 0-2 stored | conditional IBPB, SSBD on prctl |
| STIBP_ALWAYS_ON | EBX 17 | (preference only) | STIBP set once when SMT is on |
| IBRS_SAME_MODE | EBX 19 | IBRS restricts nothing, so every mode is covered | KVM passes it through |
| EFER_LMSLE_MBZ | EBX 20 | EFER bit 13 is #GP | KVM refuses LMSLE to guests |
| PSFD | EBX 28 | SPEC_CTRL bit 7 stored | (opt-in) |
| BTC_NO | EBX 29 | no branch type confusion | Retbleed not affected |
| IBPB_RET | EBX 30 | (IBPB flushes nothing) | no `ibpb_no_ret` bug |
| NO_NESTED_DATA_BP | 0x8000_0021 EAX 0 | only the accesses of the instruction `step_debug` runs are watched, never event delivery's | KVM |
| WRMSR_XX_BASE_NS | EAX 1 | serialization only ever errs the safe way | |
| LFENCE_ALWAYS_SERIALIZING | EAX 2 | in-order | lfence as the rdtsc barrier |
| NULL_SEL_CLR_BASE | EAX 6 | null selector loads leave base 0 (test pins it) | no `null_seg` bug, no FS/GS reload per switch |
| AUTOIBRS | EAX 8 | EFER.AIBRSE (bit 21) stored, also in VMCB guest EFER | Spectre v2 "Enhanced / Automatic IBRS", one EFER write per CPU |
| SBPB, IBPB_BRTYPE | EAX 27/28 | PRED_CMD bit 7 accepted | SBPB instead of IBPB (an unchecked wrmsr: the MSR bit must stay) |
| SRSO_USER_KERNEL_NO | EAX 30 | never speculates | what real Turin says; SRSO_NO (bit 29) is clear, as on Turin |
| SRSO_BP_SPEC_REDUCE | EAX 31 | BP_CFG (0xC001102E) bit 4 stored | KVM sets the bit while it has guests instead of an IBPB on every #VMEXIT |
| TSA_SQ_NO, TSA_L1_NO | ECX 1/2 | never speculates | TSA not affected (Linux would set them itself for family 1Ah) |

Vulnerability files of the Alpine 6.18 guest (`grep -r . /sys/devices/system/cpu/vulnerabilities/`):

| file | Genoa, `mitigations=off` | Genoa, mitigations on | Turin, default (no `mitigations=off`) |
|---|---|---|---|
| spectre_v2 | Vulnerable | Enhanced / Automatic IBRS (via Intel IBRS_ALL) | Mitigation: Enhanced / Automatic IBRS; IBPB: conditional |
| spectre_v1 | Vulnerable: __user pointer sanitization ... | usercopy/swapgs barriers | Mitigation: usercopy/swapgs barriers and __user pointer sanitization |
| spec_store_bypass | Not affected (Intel SSB_NO) | Not affected | Mitigation: Speculative Store Bypass disabled via prctl |
| vmscape | Vulnerable | IBPB before exit to userspace | Mitigation: IBPB before exit to userspace |
| every other file | Not affected | Not affected | Not affected |

`bugs` went from `sysret_ss_attrs null_seg spectre_v1 spectre_v2 spectre_v2_user vmscape` to `sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass spectre_v2_user vmscape`. Boot time (host wall seconds to init and power-off, 5 alternating runs on a loaded host, medians): 1.61 with `mitigations=off`, 1.60 without; with a 200000-syscall `dd` and 300 forks in init, 2.59 and 2.64, `dd` 0.250 s and 0.258 s: within the noise. `hack/kvm/run.sh` passes on the new model.

Zen 5 instructions advertised, each with reference tests and an interpreter/JIT differential case (they reach the JIT through the interpreter helper, no `jit/` change): AVX-VNNI (VEX vpdpbusd(s)/vpdpwssd(s), one kernel with the EVEX forms), MOVDIRI, MOVDIR64B, AVX512_VP2INTERSECT, PREFETCHI (`0F 18` hints). MOVDIR64B's 64-byte store is eight 8-byte stores (no host has a 64-byte atomic store for ordinary memory: aarch64's ST64B is optional and for device memory, wasm32 stops at 8 bytes), made under `Cpu::with_exclusive` so that with one host thread per vCPU no other vCPU sees it half done, as the SDM promises (`another_vcpu_never_sees_a_movdir64b_store_half_done`). Device DMA and MMIO still see eight writes; no emulated device takes 64-byte enqueue stores.

Machine-check architecture and platform QoS are Turin's (`docs/plans/03-cpu.md`, "Machine check and platform QoS"): ScalableMca, SUCCOR and McaOverflowRecov in 0x8000_0007 EBX with 24 SMCA banks named by Zen 5 IPIDs, and PQM/PQE with leaves 0xF, 0x10 and 0x8000_0020 (L3 monitoring, allocation and bandwidth enforcement, no CDP, BMEC, ABMC or slow-memory enforcement) behind MSRs that store and counters that read zero.

Performance monitoring (`cpu/pmu.rs`, tests in `cpu/pmu_tests.rs`): PERFCTR_CORE (six counters at `0xC0010200`, the K7 MSRs `0xC0010000-7` aliasing counters 0-3), PerfMonV2 (leaf `8000_0022` EAX 0; global control, status, status clear and status set at `0xC0000300-3`, control resetting to all six enabled), LBR v2 (EAX 1, 16 entries at `0xC0010300`, LBR_SELECT, DBG_EXTN_CFG.LBRV2EN; newest entry first, never mispredicted), PERFCTR_NB and PERFCTR_LLC (four data-fabric counters at `0xC0010240`, six L3 counters at `0xC0010230`, rdpmc 6-9 and 10-15: programmable, never counting, so `amd_uncore` probes) and EXTAPIC (feature register 0x400 with four extended LVT entries at 0x500, in xAPIC and x2APIC mode). Exact counts: cycles not halted (0x76, one per `icount` at the 1 GHz TSC, so rep iterations take cycles), retired instructions and ops (0xC0, 0xC1: `icount` less faulting instructions and rep iterations), retired branches and taken branches (0xC2, 0xC4, delivered exceptions and interrupts included); other events, mispredictions (0xC3) among them, count zero. OS/USR, HostOnly/GuestOnly (nothing counts with one of them and EFER.SVME clear), CntMask and Inv are honoured within the model of at most one event per cycle; Edge counts nothing. Counting is lazy: counters are brought up to date at their MSR and rdpmc accesses, at CPL and host/guest changes while a counter filters on them, and at the overflow deadline, which pulls `next_event` in so the PMI (LVT performance entry, fixed or NMI, left unmasked as on AMD) lands at the next block boundary. A reading instruction does not count itself. IRPERF (`0xC00000E9`, leaf `8000_0008` EBX 1) counts retired instructions like event 0xC0 while HWCR bit 30 is set; it is read-only. Time-shared vCPUs share one `icount`, so each PMU subtracts what the others retired while it was parked (`Cpu::pmu_park`/`pmu_unpark` around `Smp::switch_to`). Branch events and the LBR need every instruction: while one is enabled the CPU runs the interpreter (`Cpu::uses_jit`). Nothing changes in generated code. Alpine's `virt` kernel (6.18) reports `Fam17h+ 16-deep LBR, core perfctr, AMD PMU driver`, version 2 with six counters, `NMI watchdog: Enabled`, and `amd_uncore: 4 amd_df counters`, `6 amd_l3 counters`; its boot to `Mounting boot media` (host seconds, 12 alternating runs on a host at load 60) took a best of 1.69 with the PMU against 1.66 without, medians 2.35 and 2.50: noise.

IBS (`cpu/ibs.rs`, tests in `cpu/ibs_tests.rs`; `8000_0001 ECX 10`, leaf `8000_001B` EAX 0x9FF: fetch and op sampling, readable and writable IbsOpCurCnt, op counting, IbsBrTarget, the extended MaxCnt, the RIP-invalid and fused-branch bits, the Zen 4 L3MissOnly): the MSRs 0xC001_1030-103B, IBS_CTL naming extended LVT entry 0 (Linux programs 1 and 2 for the MCA interrupts), and the interrupt through that entry, which Linux makes an NMI. A sampler tags the instruction after its count reaches MaxCnt (one fetch and one op per retired instruction, a cycle per instruction); the tagged one reports its address, branch, return and taken bits (never mispredicted or fused), where it went (IbsBrTarget), and its first memory access's linear and physical address, width and direction with the valid bits; latencies, cache and TLB miss bits and the data source are zero (an L1 hit, which is how `perf mem` reads them), so L3MissOnly never samples. A fetch tag reports the fetch's linear and physical address with IbsFetchComp. The tag's position is fixed when the control register is written; `pmu_reschedule` lets the JIT run until 64 instructions before it (a JIT block holds at most 48) and then interprets, so `exec_observed` samples the tagged instruction itself; nothing changes in generated code. Alpine's 6.18 kernel reports `perf: AMD IBS detected (0x000009ff)`, lists `ibs_op` and `ibs_fetch`, and a static program sampling a spin loop through `perf_event_open` on each (period 0x10000) got 2159 op samples, 2149 inside the loop's 64 bytes and 10 in the kernel, and 2162 fetch samples, 2158 in the loop.

Platform identity (`docs/plans/07-devices.md`): the PCI root complex is Turin's (1022:153A) with the SMN pair, node 0's data fabric sits at 00:18.0-7 (1022:12C0-12C7; amd_nb and k10temp bind), the IOMMU is 1022:153B with NXSup, the DSDT has an `ACPI0007` device per CPU with `_CPC` over the CPPC MSRs (0xC001_02B0-B4, leaf 0x8000_0008 EBX bit 27; amd_pstate loads), SRAT/SLIT describe one NUMA node (leaf 0x8000_001E keeps node id 0, one node per processor), SPCR names COM1, and SMBIOS type 4 says socket SP5 with type 7 structures for L1/L2/L3 summed over the socket from leaf 0x8000_001D. The FCH's SMBus (00:14.0, i2c-piix4 binds) and LPC bridge (00:14.3) are there with the PM block (0xCD6/0xCD7 and AcpiMmio 0xFED80300) and the TCO watchdog (07, "FCH"). Not modelled: HSMP, the UMC registers amd64_edac wants, APEI.

Small Zen extensions, advertised with tests (`cpu/model_msr_tests.rs`, `cpu/sse_amd_tests.rs`, JIT differential cases through the interpreter helper): SSE4A, MMXEXT (the SSE integer forms on MMX registers, already implemented), the MMX forms of SSSE3, MISALIGNSSE, MWAITX/MONITORX, RDPRU, MCOMMIT, FFXSR, CR8_LEGACY, TCE, OSVW, NO_SMM_CTL_MSR, FSRS/FSRC, PREFETCH_CTL_MSR, CPUID_USER_DIS, SRSO_BP_SPEC_REDUCE, INT_WBINVD, PPIN, APERF/MPERF (leaf 6) with EffFreqRO, HwPstate, CPB, RAPL, and leaf 0x8000001A (MOVU, FP256). TSC-deadline and SRSO_NO, the former departures, are gone. The CPUID override MSRs, MMIO_CONF_BASE, SMM, TW_CFG, FP_CFG and SEV_STATUS MSRs exist.

Debug and TLB features (`cpu/debug.rs`, `cpu/invlpgb.rs`, tests in `cpu/debug_tests.rs`, `cpu/invlpgb.rs`, `cpu/smp_tests.rs` and `cpu/svm.rs`):
- The debug registers fire: instruction breakpoints (faults; RF suppresses them for one instruction, fault-class exceptions push RF set, `iretq` loads it), data write and read/write watchpoints (traps, LEN 1/2/4/8 aligned down, repeated string instructions stop between iterations), I/O breakpoints with CR4.DE, general detect, single-step with DEBUGCTL.BTF, the `mov ss`/`pop ss` delay (including the CVE-2018-8897 case of an event in between), the TSS T bit, and DR6 status bits. Whenever anything could raise a #DB the run loop executes one interpreted instruction at a time (`Cpu::debug_slow_path`) and `helper_exec` ends a JIT block that turns it on; generated code never checks. A guest's DR6/DR7 come and go with the VMCB. `--verify` runs through armed breakpoints and single-stepping: the main CPU takes the same `step_debug` path the oracle does whenever `uses_jit` says so, and `Cpu::diff` compares DR0-DR7, DEBUGCTL and the address masks, so a #DB taken on one side only is reported at the block that took it (`run_verify_agrees_while_single_stepping_and_with_breakpoints_armed`).
- BPEXT (0x8000_0001 ECX 26): DR0_ADDR_MASK (0xC001_1027) and DR1-DR3 (0xC001_1019-101B). AdMskExtn (bit 30, which Genoa/Turin set) is advertised: the masks are 32 bits wide (bits 11:0 without it) and apply to instruction breakpoints as well as data ones (`an_address_mask_widens_an_instruction_breakpoint`). That reading follows the bit's APM name ("address mask extension support for instruction breakpoint"); Linux only uses data range breakpoints (`perf` `mem:ADDR/LEN`), which BPEXT alone covers.
- BUS_LOCK_DETECT (leaf 7 ECX 24, Zen 5's "bus lock trap"): DEBUGCTL bit 2 makes a locked access split across a 64-byte line a #DB trap with DR6 bit 11 (active low, now writable) cleared. Linux turns it on at boot (`split_lock_detect=warn` by default), so it is checked in `Cpu::exec`, which every locked instruction goes through on either engine, and does not slow the JIT. Locked accesses to uncached (device) memory are not counted as bus locks.
- INVLPGB/TLBSYNC (0x8000_0008 EBX 3, EDX[15:0] = 7): the PCID, global and final-only filters widen to "all" (the soft TLB keeps none of them apart); the ASID picks the running or parked TLB. #UD in real and virtual-8086 mode and in a guest whose VMCB leaves 0x90 bit 7 clear, #GP at CPL > 0 or with more than eight pages; the SVM intercepts (vector 5 bits 0, 1, 4; exits 0xA0, 0xA1, 0xA4) are honored first. In a guest that may (TlbiCtl, 0x8000_000A EDX 24) INVLPGB reaches only the guest's own ASID whatever EDX and rAX[2] say, on every vCPU running or parking that ASID, and only its linear translations: never the host's nor the cached nested-table leaves, which the hypervisor invalidates; the host's INVLPGB with a guest ASID now reaches only that ASID's parked TLB (`invlpgb_in_a_guest_drops_its_own_translation_and_leaves_the_hosts`, with and without nested paging), and so is the MCOMMIT one (bit 3, exit 0xA3), ahead of its EFER.MCOMMIT #UD. EFER.TCE is not involved.

Address widths, 5-level paging and protection keys are Turin's (03, "5-level paging and protection keys"): 52 physical and 57 linear bits, LA57, PKU/OSPKE with PKRU as XSAVE component 9.

Present on Turin, not advertised (not implemented): ERAPS, UpperAddressIgnore, WDT, the thermal bits of leaf 0x80000007, SSSCheck and IBS virtualization in 0x8000_000A EDX, SEV/SEV-ES/SEV-SNP. Left clear for want of a source: FP512 (leaf 0x8000001A bit 3), IBRS_PREFERRED, AMD_ERMSB and the leaf 0x80000021 EBX microcode patch size. AMD_SSB_NO and IBRS_ALWAYS_ON stay clear because Turin does not report them.

## Order

| step | content | depends on |
|---|---|---|
| 1 | Section A, in table order, one commit per feature | none |
| 2 | Section E bits, the MPERF/APERF/RAPL MSRs from D | none |
| 3 | `CpuProfile` with Genoa as the only profile | none |
| 4 | LA57 | none |
| 5 | PKU | none (shares the walker with 4) |
| 6 | PerfMonV2 | 2 (cycles/instructions source) |
| 7 | SVM audit and additions from C | none |
| 8 | CET shadow stacks | 4, 5 (walker permission model) |
| 9 | Ryzen profile; boot the installers and the Xen dom0 on it and compare `lscpu` flags, `/sys/devices/system/cpu/vulnerabilities/` and `dmesg` CPU lines with the Genoa profile (expected differences only: brand, model, topology); flip the default | 1-8 |
| 10 | LBR v2, IBS, EXTAPIC, AVIC if wanted | 6, 7 |

Done: steps 1, 2, 4 to 6, 8 and 10, on the Turin model, and step 7 but for the host-visible `VIRT_SPEC_CTRL`. Step 3 only in part: `CpuProfile` exists for the count, threads and brand, but signature, caches and features are fixed in `cpu/cpuid.rs` and there is no Genoa profile. Step 9 is not started.

## Checking it

- Per feature: unit tests beside the code, interpreter first, JIT through `assert_jit_matches_interpreter`; CPUID tests spell the expected words from the APM rather than the constants in `cpuid.rs`, as `cpuid_tests.rs` does.
- Whole-CPU: `hack/kvm/run.sh` (SVM in the emulated guest), a Linux boot printing `/proc/cpuinfo` flags, and `--verify` over a boot after any JIT-visible change. `kvm-unit-tests` is the best external check for B and C and can run unmodified in a guest.
- A parity test in `cpuid_tests.rs`: the Ryzen profile's feature bits equal an explicit list checked in from the sources above, so adding a feature means touching the list and its test together.

## Open questions

- Which SKU names the profile (Raphael 7600X vs Phoenix 7640U changes the PPR consulted and the brand string, not the feature set)?
- Keep TSC-deadline as a departure (fast calibration) or remove it for fidelity? Decided: removed, as on the hardware; Linux calibrates the APIC timer against the PIT.
- Is a guest hypervisor with AVIC a requirement, or is the legacy SVM path enough? Settled by doing it: AVIC and x2AVIC are implemented (section C).
- LA57 changes the JIT's inline fast paths (canonical checks): confirm with `docs/plans/20-jit-perf.md` how much it costs before turning it on by default in the profile. Moot: the Turin model advertises LA57 and the JIT needed no change (section B). CET is always on, and the JIT compiles shadow-stacked near `call`/`ret` inline (a run-time byte test, `Cpu::shadow_stack_on`, rather than a block key; `docs/plans/06-jit.md`), so Xen, which enables supervisor shadow stacks on Zen 5 by default, runs its own code in the JIT.
