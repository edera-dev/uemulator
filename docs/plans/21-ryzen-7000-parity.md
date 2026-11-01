# Ryzen 5 7000-class feature parity

**Status:** proposal.

Goal: the emulated CPU **has every feature a Ryzen 5 7000-series part (Zen 4: Raphael 7600/7600X, Phoenix 7640U/7640HS, family 19h) advertises**, so a guest, a hypervisor or a test suite that looks for a feature finds it and it works. This is a feature list, not a fidelity project:

- Not the exact core. Instruction latency, speculation, cache and TLB replacement, performance-counter values, undefined-flag quirks beyond what the manuals specify: out of scope.
- Not the exact platform. ACPI, PCI topology and fabric registers stay the generic virtual PC (`docs/plans/07-devices.md`).
- No access to the real part. Everything comes from public documents; nothing here needs a capture from silicon. Where a public source is silent or two disagree, we follow the manual and record the choice.

Today the identity is an EPYC 9004 Genoa (family 19h model 11h, `docs/plans/03-cpu.md`, "CPUID/MSR contract"). It already shares the Zen 4 ISA baseline (AVX2, AVX-512 with VNNI/BF16/IFMA/VBMI/BITALG/VPOPCNTDQ, SHA, GFNI, VAES, VPCLMULQDQ, SMEP/SMAP/UMIP, FSGSBASE, PCID/INVPCID, RDPID, RDSEED, XSAVEC, SVM with nested paging). The work is the delta.

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

`svm::SVM_FEATURES` today: nested paging, next-RIP save, VMCB clean bits, flush-by-ASID, all in their "reload and flush everything" form. Zen 4 reports more in 0x8000_000A EDX; audit the list against the APM and add what a hypervisor will use, each with a guest test in `hack/kvm/run.sh`'s style or a fake kernel:
- `vGIF` (virtual GIF, `STGI`/`CLGI` without exit), `V_VMSAVE_VMLOAD`, TSC rate MSR (`TscRatioMsr`), pause filter (count and threshold), decode assists (`DecodeAssists` fills the VMCB instruction bytes on #PF/INVLPG exits), SVM lock/`SvmLock` and `SVME_DISABLE` plumbing, `GMET`, and the host-visible `VIRT_SPEC_CTRL`.
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
- Flip the default only after the guest check below passes: family and model drive Linux and Xen erratum, mitigation and perf-driver selection, so the identity change is the riskiest one-line change in the list.

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
