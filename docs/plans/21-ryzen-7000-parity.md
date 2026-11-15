# Ryzen 5 7000-class feature parity

**Status:** proposal.

Goal: the emulated CPU **has every feature a Ryzen 5 7000-series part (Zen 4: Raphael 7600/7600X, Phoenix 7640U/7640HS, family 19h) advertises**, so a guest, a hypervisor or a test suite that looks for a feature finds it and it works. This is a feature list, not a fidelity project:

- Not the exact core. Instruction latency, speculation, cache and TLB replacement, performance-counter values, undefined-flag quirks beyond what the manuals specify: out of scope.
- Not the exact platform. ACPI, PCI topology and fabric registers stay the generic virtual PC (`docs/plans/07-devices.md`).
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

| feature | CPUID | what it needs |
|---|---|---|
| SSE4A | 8000_0001 ECX 6 | `extrq`, `insertq`, `movntss`, `movntsd` (decoder, interpreter, JIT) |
| MMXEXT (AMD extensions to MMX/SSE) | 8000_0001 EDX 22 | the integer MMX extensions on MMX registers (`pshufw`, `pavgb`, `pmaxsw`, `maskmovq`, `movntq`, `pextrw`...), most of which the SSE2 forms already implement on XMM |
| FFXSR | 8000_0001 EDX 25 | `EFER.FFXSR` (bit 14): `fxsave`/`fxrstor` skip XMM in 64-bit mode at CPL 0; EFER writes currently reject the bit |
| MWAITX / MONITORX | 8000_0001 ECX 29 | the two instructions with a TSC-based timeout; builds on the existing MONITOR/MWAIT arm |
| RDPRU | 8000_0008 EBX 4 | `rdpru` reading MPERF/APERF (fixed ratio 1; the MSRs `0xE7`/`0xE8` come with it) |
| CR8_LEGACY | 8000_0001 ECX 4 | `lock mov cr8` in 32-bit code, a 3-line decoder rule |
| OSVW | 8000_0001 ECX 9 | `OSVW_ID_LENGTH` / `OSVW_STATUS` MSRs (`0xC0010140/1`): storage only |
| INVLPGB / TLBSYNC | 8000_0008 EBX 4... (confirm bit in APM) | broadcast TLB invalidation; with one thread running all vCPUs, "broadcast" is a flush of every vCPU's TLB in `Smp` |

### B. Architectural state that touches paging and XSAVE (the real work)

1. **LA57, 5-level paging.** `CR4.LA57`, a fifth page-walk level (PML5), 57-bit canonical checks (the `translate` canonical test, branch targets, `lea`-based address generation, `invlpg`, `invpcid`), the 57-bit virtual-address field in 0x8000_0008, LA57 in the SVM nested-paging guest state, and the JIT's inline canonical check. `PagingMode` gains a `Long5` variant; `walk_wide` already loops over levels, so the level count is the main parameter. Large-page cache (`docs/plans/04-memory.md`) needs the shift for the new level only if 1 GiB+ leaves at the top are allowed (they are not at PML5).
2. **PKU/OSPKE, protection keys for user pages.** `CR4.PKE`, `PKRU`, `rdpkru`/`wrpkru`, PTE bits 62:59, page-fault error-code bit 5 (PK), XSAVE component 9, leaf 7 ECX bits for PKU/OSPKE. The walker evaluates the key against `PKRU` per access (user pages only; supervisor keys are Intel-only), so cached TLB entries must carry or recheck the key: either fold the key into the permission tags at fill time (and flush on `wrpkru`, which is architecturally not a TLB flush, so this must be re-evaluated per access) or keep PK pages out of the write/read tags like the SMAP case in `translate`.
3. **CET shadow stacks (user and supervisor).** `CR4.CET`, `U_CET`/`S_CET`/`PL0_SSP..PL3_SSP`/`ISST_ADDR` MSRs, `#CP` (vector 21), shadow-stack pages (`W=0, D=1` encoding), `incssp`, `rdssp`, `saveprevssp`, `rstorssp`, `wrss`, `setssbsy`, `clrssbsy`, shadow-stack pushes on `call` and checks on `ret`/`iret`/far transfers, the supervisor shadow-stack token protocol, XSS components 11 and 12 (currently `IA32_XSS` accepts only 0). Zen 4 has shadow stacks but not indirect-branch tracking (the `endbr64` NOP encoding already decodes). This is the largest item in the plan and last in this section; Linux uses it for user shadow stacks (6.6+) and the SVM guest state carries `S_CET` and `SSP` fields.

### C. Virtualisation (SVM) completeness

`svm::SVM_FEATURES` today: nested paging, next-RIP save, VMCB clean bits, flush-by-ASID (all in their "reload and flush everything" form), LBR virtualization, SVM lock, the TSC ratio MSR, the pause filter and its threshold; 32768 ASIDs; the NMI, INIT, SMI (never raised) and task-switch intercepts (`docs/plans/03-cpu.md`, "SVM"). Zen 4 reports more in 0x8000_000A EDX; audit the list against the APM and add what a hypervisor will use, each with a guest test in `hack/kvm/run.sh`'s style or a fake kernel:
- `vGIF` (virtual GIF, `STGI`/`CLGI` without exit), `V_VMSAVE_VMLOAD`, decode assists (`DecodeAssists` fills the VMCB instruction bytes on #PF/INVLPG exits), `GMET`, and the host-visible `VIRT_SPEC_CTRL`. Done: TSC rate MSR, pause filter (count and threshold), SVM lock and `SVM_KEY`, LBR virtualization.
- AVIC/x2AVIC (hardware-virtualised local APIC) is the large one; skip unless a guest hypervisor needs it, and then design it with the multi-APIC model in `docs/plans/03-cpu.md`.
- SEV, SEV-ES, SEV-SNP: not on client Ryzen; not planned.

### D. Counters and monitoring a guest can program

- **PerfMonV2** (`PERFCTR_CORE` `8000_0001 ECX 23`, `PERFMON_V2` `8000_0022 EAX 0`): six core counters at `0xC0010200`, global control/status MSRs (`0xC0000300..`), counting a small set of events the emulator can count exactly: retired instructions, retired branches, retired taken branches, cycles (= `icount` at the model's rate, `NS_PER_INSN`). Keep the legacy four-counter set that is already there. Overflow raises the LVT performance-counter interrupt, which makes the NMI watchdog and `perf stat` work.
- `PERFCTR_NB`/`PERFCTR_L2`: counters that read zero (present, never counting); only if a guest probes them and fails otherwise.
- **LBR v2** (`8000_0022 EAX 1`), **IBS** (`8000_0001 ECX 10`, `8000_001B`) and the extended APIC space (`EXTAPIC`, `8000_0001 ECX 3`) it hangs its LVT on: moderate; IBS needs a sampling model (take a sample every N retired ops from `icount`), so it is a later, optional item.
- MPERF/APERF (`0xE7/0xE8`) at ratio 1, RAPL energy MSRs reading a monotonic fake counter: cheap, and they stop `amd_pstate`/`turbostat`/`powertop` probes failing.

### E. Security and system bits that are cheap and honest

Add only where the behaviour is real in this implementation and testable:
- `NullSelectorClearsBase` (`8000_0021 EAX 6`): loading a null selector into a data segment register clears the base in 64-bit mode; confirm the current behaviour and set the bit.
- `NoNestedDataBp` (`EAX 0`) and `AutoIBRS` (`EAX 8`): `EFER.AIBRSE` accepted and inert (no speculation to restrict), exactly the stance already taken for `IBRS`.
- Keep the declared departures from `docs/plans/03-cpu.md`: TSC-deadline advertised though AMD has none, the Intel spelling of the speculation-control leaf, `SRSO_NO`/`TSA_*_NO`, no hypervisor leaves. Decide on removing the Intel spellings separately; the right test is what Linux and Xen do with the AMD-only spelling.

### F. Identity (last, small)

- `CpuProfile { signature, brand, logical processors, cache sizes, feature set }` in `cpu/profile.rs`, Genoa expressed as one profile with no behaviour change, then a Ryzen 5 7000-class profile (model per the PPR for the chosen SKU, brand `AMD Ryzen 5 7600X 6-Core Processor` style, 6 cores 12 threads, 32 KiB L1, 1 MiB L2 per core, 32 MiB L3) selected by `MachineConfig::cpu_model`; CLI `--cpu-model`. The feature bits come from sections A-E, so the profile only chooses which are on; a profile cannot enable a bit the emulator lacks (checked by a test that walks every profile and asserts each advertised bit has a registered implementation).
- Done so far: `CpuProfile` (`cpu/profile.rs`) with the logical processor count, threads per core and brand, built from `MachineConfig::cpus` and `MachineConfig::smt`. Signature, caches and feature set are still the one Turin model's.
- Flip the default only after the guest check below passes: family and model drive Linux and Xen erratum, mitigation and perf-driver selection, so the identity change is the riskiest one-line change in the list.

## Zen 5 Turin model (the current identity)

The emulated CPU presents a 2024 AMD server core, an EPYC 9005 "Turin" (Zen 5), so that Linux sees hardware that needs no software mitigation and the default command line can go without `mitigations=off`. There is no `CpuProfile` (section F is still a plan), so Turin replaced Genoa outright: the Genoa leaves are in the history of `cpu/cpuid.rs`, not selectable.

Identity and sources:
- Signature `0x00B00F21`: family 1Ah, model 02h, stepping 1, the Turin (classic, Zen 5) die. Linux's `bsp_init_amd` names family 1Ah models 00h-2Fh, 40h-4Fh and 60h-7Fh Zen 5, and `init_amd_zen5` keeps RDSEED only when model, stepping and microcode revision match `zen5_rdseed_microcode`, whose entry for 1Ah/02h/1 is revision `0x0b00215a`; PATCH_LEVEL (MSR 0x8b) reports exactly that. Model 11h stepping 0 in the same table is Turin Dense (Zen 5c). QEMU's `EPYC-Turin` uses model 0 stepping 0, which is in no kernel table and would cost RDSEED, so only its feature list and cache sizes were used as a cross-check (`../ext/qemu-uplink/target/i386/cpu.c`).
- Brand: the smallest EPYC 9005 part with at least as many cores as the machine has (`cpu/profile.rs`: 9015, 9115, 9255, 9335, 9365, 9455, 9535, 9565, 9655, 9755, then the Turin Dense 9825, 9845, 9965 above 128 cores, whose model 11h the signature does not follow), so the guest never sees more cores than the name says, as in a VM on that part. It was QEMU's "AMD EPYC-Turin Processor" while the core count was fixed nowhere.
- Caches per core complex die: L1D 48 KiB 12-way, L1I 32 KiB 8-way, L2 1 MiB 16-way per core, L3 32 MiB 16-way shared by 16 threads (QEMU's `epyc_turin_cache_info` agrees). TLBs from public Zen 5 microarchitecture descriptions, not a CPUID dump: L1 DTLB 96, L1 ITLB 64, L2 ITLB 2048; the L2 DTLB keeps 3072 because the 4096 Zen 5 is said to have does not fit the 12-bit field of leaf 0x8000_0006. The field widths are asserted at compile time.
- Topology: leaf 0x8000_0026 (core, complex, die, socket) is new; Linux prefers it to 0xB. The complex level is the threads of the 8 cores behind one L3, the same number leaf 0x8000_001D reports, so Linux's tile and LLC domains agree; the max extended leaf is 0x8000_0026 with 0x22-0x25 zero. SMT2 is on by default (`MachineConfig::smt`): an even vCPU count is cores of two threads. Tests walk every processor of 1-384 vCPUs with SMT on and off.

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
| NO_NESTED_DATA_BP | 0x8000_0021 EAX 0 | no hardware breakpoint ever fires | KVM |
| WRMSR_XX_BASE_NS | EAX 1 | serialization only ever errs the safe way | |
| LFENCE_ALWAYS_SERIALIZING | EAX 2 | in-order | lfence as the rdtsc barrier |
| NULL_SEL_CLR_BASE | EAX 6 | null selector loads leave base 0 (test pins it) | no `null_seg` bug, no FS/GS reload per switch |
| AUTOIBRS | EAX 8 | EFER.AIBRSE (bit 21) stored, also in VMCB guest EFER | Spectre v2 "Enhanced / Automatic IBRS", one EFER write per CPU |
| SBPB, IBPB_BRTYPE | EAX 27/28 | PRED_CMD bit 7 accepted | SBPB instead of IBPB (an unchecked wrmsr: the MSR bit must stay) |
| SRSO_NO | EAX 29 | never speculates | SRSO not affected (departure: real Turin lacks it and gets "IBPB on VMEXIT only") |
| SRSO_USER_KERNEL_NO | EAX 30 | implied by SRSO_NO | what real Turin says |
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

Zen 5 instructions advertised, each with reference tests and an interpreter/JIT differential case (they reach the JIT through the interpreter helper, no `jit/` change): AVX-VNNI (VEX vpdpbusd(s)/vpdpwssd(s), one kernel with the EVEX forms), MOVDIRI, MOVDIR64B, AVX512_VP2INTERSECT, PREFETCHI (`0F 18` hints). MOVDIR64B's 64-byte store is eight 8-byte stores, so with one host thread per vCPU another vCPU can see it half done; the SDM promises a single unit, which no guest seen so far depends on.

Machine-check architecture and platform QoS are Turin's (`docs/plans/03-cpu.md`, "Machine check and platform QoS"): ScalableMca, SUCCOR and McaOverflowRecov in 0x8000_0007 EBX with 24 SMCA banks named by Zen 5 IPIDs, and PQM/PQE with leaves 0xF, 0x10 and 0x8000_0020 (L3 monitoring, allocation and bandwidth enforcement, no CDP, BMEC, ABMC or slow-memory enforcement) behind MSRs that store and counters that read zero.

Performance monitoring (`cpu/pmu.rs`, tests in `cpu/pmu_tests.rs`): PERFCTR_CORE (six counters at `0xC0010200`, the K7 MSRs `0xC0010000-7` aliasing counters 0-3), PerfMonV2 (leaf `8000_0022` EAX 0; global control, status, status clear and status set at `0xC0000300-3`, control resetting to all six enabled), LBR v2 (EAX 1, 16 entries at `0xC0010300`, LBR_SELECT, DBG_EXTN_CFG.LBRV2EN; newest entry first, never mispredicted), PERFCTR_NB and PERFCTR_LLC (four data-fabric counters at `0xC0010240`, six L3 counters at `0xC0010230`, rdpmc 6-9 and 10-15: programmable, never counting, so `amd_uncore` probes) and EXTAPIC (feature register 0x400 with four extended LVT entries at 0x500, in xAPIC and x2APIC mode). Exact counts: cycles not halted (0x76, one per `icount` at the 1 GHz TSC, so rep iterations take cycles), retired instructions and ops (0xC0, 0xC1: `icount` less faulting instructions and rep iterations), retired branches and taken branches (0xC2, 0xC4, delivered exceptions and interrupts included); other events, mispredictions (0xC3) among them, count zero. OS/USR, HostOnly/GuestOnly (nothing counts with one of them and EFER.SVME clear), CntMask and Inv are honoured within the model of at most one event per cycle; Edge counts nothing. Counting is lazy: counters are brought up to date at their MSR and rdpmc accesses, at CPL and host/guest changes while a counter filters on them, and at the overflow deadline, which pulls `next_event` in so the PMI (LVT performance entry, fixed or NMI, left unmasked as on AMD) lands at the next block boundary. A reading instruction does not count itself. Branch events and the LBR need every instruction: while one is enabled the CPU runs the interpreter (`Cpu::uses_jit`). Nothing changes in generated code. Alpine's `virt` kernel (6.18) reports `Fam17h+ 16-deep LBR, core perfctr, AMD PMU driver`, version 2 with six counters, `NMI watchdog: Enabled`, and `amd_uncore: 4 amd_df counters`, `6 amd_l3 counters`; its boot to `Mounting boot media` (host seconds, 12 alternating runs on a host at load 60) took a best of 1.69 with the PMU against 1.66 without, medians 2.35 and 2.50: noise. IBS (`8000_0001 ECX 10`, `8000_001B`) stays unadvertised: fetch and op sampling need per-op latencies, cache-miss and data-address fields the model does not have, and a sampler that only filled in RIP and the branch bits would mislead `perf mem`/`perf c2c`.

Platform identity (`docs/plans/07-devices.md`): the PCI root complex is Turin's (1022:153A) with the SMN pair, node 0's data fabric sits at 00:18.0-7 (1022:12C0-12C7; amd_nb and k10temp bind), the IOMMU is 1022:153B with NXSup, the DSDT has an `ACPI0007` device per CPU with `_CPC` over the CPPC MSRs (0xC001_02B0-B4, leaf 0x8000_0008 EBX bit 27; amd_pstate loads), SRAT/SLIT describe one NUMA node (leaf 0x8000_001E keeps node id 0, one node per processor), SPCR names COM1, and SMBIOS type 4 says socket SP5 with type 7 structures for L1/L2/L3 summed over the socket from leaf 0x8000_001D. Not modelled: the FCH functions, HSMP, the UMC registers amd64_edac wants, APEI.

Present on Turin, not advertised (not implemented): LA57, PKU/OSPKE, CET shadow stacks, SSE4A, MMXEXT, FFXSR, MWAITX/MONITORX, RDPRU, INVLPGB/TLBSYNC, OSVW, CR8_LEGACY, IBS, SRSO_BP_SPEC_REDUCE (needs the BP_CFG bit KVM toggles), ERAPS, NO_SMM_CTL_MSR, the extended SVM features (vGIF, vNMI, V_VMSAVE_VMLOAD, decode assists, AVIC/x2AVIC; TSC scaling, the pause filter, LBR virtualization and SVM lock are advertised), SEV/SEV-ES/SEV-SNP, RAPL, 52/57 address bits. AMD_SSB_NO and IBRS_ALWAYS_ON stay clear because Turin does not report them.

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

## Checking it

- Per feature: unit tests beside the code, interpreter first, JIT through `assert_jit_matches_interpreter`; CPUID tests spell the expected words from the APM rather than the constants in `cpuid.rs`, as `cpuid_tests.rs` does.
- Whole-CPU: `hack/kvm/run.sh` (SVM in the emulated guest), a Linux boot printing `/proc/cpuinfo` flags, and `--verify` over a boot after any JIT-visible change. `kvm-unit-tests` is the best external check for B and C and can run unmodified in a guest.
- A parity test in `cpuid_tests.rs`: the Ryzen profile's feature bits equal an explicit list checked in from the sources above, so adding a feature means touching the list and its test together.

## Open questions

- Which SKU names the profile (Raphael 7600X vs Phoenix 7640U changes the PPR consulted and the brand string, not the feature set)?
- Keep TSC-deadline as a departure (fast calibration) or remove it for fidelity? Measure boot cost before deciding.
- Is a guest hypervisor with AVIC a requirement, or is the legacy SVM path enough?
- CET and LA57 both change the JIT's inline fast paths (shadow-stack traffic on `call`/`ret`, canonical checks): confirm with `docs/plans/20-jit-perf.md` how much they cost before turning them on by default in the profile.
