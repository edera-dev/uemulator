//! CPUID: an AMD Zen 4 (family 19h, EPYC 9004 "Genoa") with only the features that have tests
//! behind them. Read `docs/plans/03-cpu.md` ("CPUID/MSR contract") before adding a bit: a set bit is a
//! promise that the instruction, MSR or table it describes works.
//!
//! The identity matters to the software that matters. Linux picks erratum workarounds, mitigation
//! tables and perf drivers by family and model; Xen additionally builds its CPU policy from the
//! raw leaves and refuses to run guests that depend on leaves it cannot make sense of. So the
//! leaves a Zen 4 has are all present and consistent with one another (cache geometry and
//! topology agree with each other and with the configured CPU count), and the ones it does not
//! (Intel-only leaf 4, the hypervisor range, IBS, SEV) read as zero.
//!
//! Deliberate departures from the hardware, each so a guest does not run code the emulator would
//! only slow down:
//! - TSC deadline is advertised (AMD has none) because the local APIC model implements it and
//!   guests calibrate the timer far faster with it.
//! - Leaf 7 EDX and `IA32_ARCH_CAPABILITIES` keep the Intel spelling of the speculation-control
//!   interface (see `sys.rs`), and leaf 0x8000_0021 claims SRSO_NO: the CPU never speculates, so
//!   no mitigation (retpolines, IBPB on entry, safe-ret thunks) is wanted.
//! - 40 physical and 48 virtual address bits, not 52 and 57.

use super::sys::{CR4_OSXSAVE, PHYSICAL_ADDRESS_BITS, RAX, RBX, RCX, RDX, VIRTUAL_ADDRESS_BITS};
use super::{Cpu, xsave};

/// Where this logical processor sits. The defaults describe a machine with one CPU. SMP is not
/// implemented yet; the shape is accepted now so that the leaves already agree with whatever
/// count the embedder configures, and the local APIC id has to be set to the same `x2apic_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuTopology {
    /// Logical processors in the package, 1 to 255.
    pub logical_processors: u32,
    /// Hardware threads per core: 1 or 2 (Zen has SMT2). Must divide `logical_processors`.
    pub threads_per_core: u32,
    /// This processor's x2APIC id: `core << smt_shift | thread` (AMD numbers threads of a core
    /// consecutively, which is what leaf 0x8000_001E reports).
    pub x2apic_id: u32,
}

impl CpuTopology {
    pub(crate) const SINGLE_PROCESSOR: CpuTopology = CpuTopology {
        logical_processors: 1,
        threads_per_core: 1,
        x2apic_id: 0,
    };

    /// Bits of the x2APIC id that select the thread within a core.
    fn smt_shift(&self) -> u32 {
        ceil_log2(self.threads_per_core)
    }

    /// Bits of the x2APIC id below the package: leaf 0xB level 1 must exist even for one CPU,
    /// where the field is a single bit wide.
    fn package_shift(&self) -> u32 {
        ceil_log2(self.logical_processors).max(1)
    }

    fn core_id(&self) -> u32 {
        self.x2apic_id >> self.smt_shift()
    }
}

fn ceil_log2(value: u32) -> u32 {
    value.next_power_of_two().trailing_zeros()
}

const LEAF_VENDOR: u32 = 0;
const LEAF_FEATURES: u32 = 1;
const LEAF_MONITOR: u32 = 5;
const LEAF_THERMAL_POWER: u32 = 6;
const LEAF_STRUCTURED_EXTENDED_FEATURES: u32 = 7;
const LEAF_EXTENDED_TOPOLOGY: u32 = 0xb;
const LEAF_XSTATE: u32 = 0xd;
const LEAF_EXT_MAX: u32 = 0x8000_0000;
const LEAF_EXT_FEATURES: u32 = 0x8000_0001;
const LEAF_BRAND_FIRST: u32 = 0x8000_0002;
const LEAF_BRAND_LAST: u32 = 0x8000_0004;
const LEAF_L1_CACHE_TLB: u32 = 0x8000_0005;
const LEAF_L2_L3_CACHE_TLB: u32 = 0x8000_0006;
const LEAF_ADVANCED_POWER: u32 = 0x8000_0007;
const LEAF_ADDRESS_SIZES: u32 = 0x8000_0008;
const LEAF_SVM: u32 = 0x8000_000a;
const LEAF_TLB_1G: u32 = 0x8000_0019;
const LEAF_CACHE_PROPERTIES: u32 = 0x8000_001d;
const LEAF_EXTENDED_APIC_ID: u32 = 0x8000_001e;
const LEAF_EXT_FEATURES_2: u32 = 0x8000_0021;

/// Highest standard leaf. Zen 4 stops at 0x10 (the quality-of-service leaves 0xF and 0x10 read
/// as zero here); 2-4, 8-0xA, 0xC, 0xE are within range and zero, as on AMD.
const MAX_STANDARD_LEAF: u32 = 0x10;
const MAX_EXTENDED_LEAF: u32 = LEAF_EXT_FEATURES_2;

/// Vendor string "AuthenticAMD" is returned in EBX, EDX, ECX order.
const VENDOR_EBX: u32 = u32::from_le_bytes(*b"Auth");
const VENDOR_EDX: u32 = u32::from_le_bytes(*b"enti");
const VENDOR_ECX: u32 = u32::from_le_bytes(*b"cAMD");

/// Leaf 1 and leaf 0x8000_0001 EAX: stepping 1 (bits 3:0), base model 1 (7:4), base family 0xf
/// (11:8), extended model 1 (19:16), extended family 0xa (27:20): family 0x19, model 0x11. That
/// is Genoa (EPYC 9004, B1 stepping), the part Xen and Linux name Zen 4.
pub(super) const SIGNATURE: u32 = 0x00a1_0f11;
/// Leaf 1 EBX: CLFLUSH line size in 8-byte units (bits 15:8).
const CLFLUSH_LINE_UNITS: u32 = 8;
const LEAF1_EBX_CLFLUSH_SHIFT: u32 = 8;
const LEAF1_EBX_LOGICAL_PROCESSORS_SHIFT: u32 = 16;
const LEAF1_EBX_APIC_ID_SHIFT: u32 = 24;

const FEATURE_FPU: u32 = 1 << 0;
const FEATURE_DE: u32 = 1 << 2;
const FEATURE_PSE: u32 = 1 << 3;
const FEATURE_TSC: u32 = 1 << 4;
const FEATURE_MSR: u32 = 1 << 5;
const FEATURE_PAE: u32 = 1 << 6;
const FEATURE_MCE: u32 = 1 << 7;
const FEATURE_CX8: u32 = 1 << 8;
/// On-chip local APIC (`dev/apic.rs`); its MMIO window is enabled by IA32_APIC_BASE.
const FEATURE_APIC: u32 = 1 << 9;
/// SEP is set for protected-mode sysenter/sysexit (`do_sysenter`).
const FEATURE_SEP: u32 = 1 << 11;
const FEATURE_MTRR: u32 = 1 << 12;
const FEATURE_PGE: u32 = 1 << 13;
const FEATURE_MCA: u32 = 1 << 14;
const FEATURE_CMOV: u32 = 1 << 15;
const FEATURE_PAT: u32 = 1 << 16;
/// PSE-36 comes with the 4 MiB PDE address bits 39:32 (`mem.rs`), which match the 40 physical
/// address bits of leaf 0x8000_0008.
const FEATURE_PSE36: u32 = 1 << 17;
const FEATURE_CLFSH: u32 = 1 << 19;
/// MMX is advertised because glibc's x86-64 baseline ISA check (CMOV, CX8, FPU, FXSR, MMX, SSE,
/// SSE2) needs the bit; without it every Debian binary refuses to start.
const FEATURE_MMX: u32 = 1 << 23;
const FEATURE_FXSR: u32 = 1 << 24;
const FEATURE_SSE: u32 = 1 << 25;
const FEATURE_SSE2: u32 = 1 << 26;
/// Hyper-threading: more than one logical processor in the package. Not advertised for one.
const FEATURE_HTT: u32 = 1 << 28;
const LEAF1_EDX: u32 = FEATURE_FPU
    | FEATURE_DE
    | FEATURE_PSE
    | FEATURE_TSC
    | FEATURE_MSR
    | FEATURE_PAE
    | FEATURE_MCE
    | FEATURE_CX8
    | FEATURE_APIC
    | FEATURE_SEP
    | FEATURE_MTRR
    | FEATURE_PGE
    | FEATURE_MCA
    | FEATURE_CMOV
    | FEATURE_PAT
    | FEATURE_PSE36
    | FEATURE_CLFSH
    | FEATURE_MMX
    | FEATURE_FXSR
    | FEATURE_SSE
    | FEATURE_SSE2;

/// Leaf 0x8000_0001 EDX repeats this subset of the leaf 1 bits (the AMD aliases); DE and PSE
/// are not mirrored.
const EXT_ALIASED_EDX: u32 = FEATURE_FPU
    | FEATURE_TSC
    | FEATURE_MSR
    | FEATURE_PAE
    | FEATURE_CX8
    | FEATURE_PGE
    | FEATURE_CMOV
    | FEATURE_PAT
    | FEATURE_MMX
    | FEATURE_FXSR;

// Leaf 1 ECX. Each bit is advertised together with the tests of the instructions behind it.
/// SSE3: movddup, movshdup, movsldup, lddqu, addsub*, hadd*, hsub*, and x87 fisttp.
const FEATURE_SSE3: u32 = 1 << 0;
const FEATURE_PCLMULQDQ: u32 = 1 << 1;
/// MONITOR/MWAIT (leaf 5 describes them).
const FEATURE_MONITOR: u32 = 1 << 3;
/// SSSE3, XMM forms only (the MMX forms are #UD, see `sse_ext.rs`).
const FEATURE_SSSE3: u32 = 1 << 9;
/// FMA3: the vfmadd/vfmsub/vfnmadd/vfnmsub families.
const FEATURE_FMA: u32 = 1 << 12;
const FEATURE_CX16: u32 = 1 << 13;
/// CR4.PCIDE and the PCID field of CR3.
const FEATURE_PCID: u32 = 1 << 17;
const FEATURE_SSE4_1: u32 = 1 << 19;
/// SSE4.2: pcmpgtq, CRC32 and the pcmp[ei]str[im] string instructions.
const FEATURE_SSE4_2: u32 = 1 << 20;
/// x2APIC mode of the local APIC: register access by MSR (0x800..=0x8ff), 32-bit APIC ids.
const FEATURE_X2APIC: u32 = 1 << 21;
const FEATURE_MOVBE: u32 = 1 << 22;
const FEATURE_POPCNT: u32 = 1 << 23;
/// IA32_TSC_DEADLINE (0x6e0) and the LVT timer's deadline mode.
const FEATURE_TSC_DEADLINE: u32 = 1 << 24;
/// AES-NI: aesenc, aesenclast, aesdec, aesdeclast, aesimc, aeskeygenassist.
const FEATURE_AES: u32 = 1 << 25;
/// xsave, xrstor, xsaveopt, xgetbv, xsetbv; the state components and layouts are in `xsave.rs`.
const FEATURE_XSAVE: u32 = 1 << 26;
/// Mirrors CR4.OSXSAVE; computed in `cpuid`, never part of the static mask.
const FEATURE_OSXSAVE: u32 = 1 << 27;
/// AVX (VEX-encoded 128/256-bit floating point and the AVX forms of the SSE integer set).
const FEATURE_AVX: u32 = 1 << 28;
/// F16C: vcvtph2ps and vcvtps2ph.
const FEATURE_F16C: u32 = 1 << 29;
/// rdrand (see `MachineConfig::random_seed`).
const FEATURE_RDRAND: u32 = 1 << 30;
const LEAF1_ECX: u32 = FEATURE_SSE3
    | FEATURE_PCLMULQDQ
    | FEATURE_MONITOR
    | FEATURE_SSSE3
    | FEATURE_FMA
    | FEATURE_CX16
    | FEATURE_PCID
    | FEATURE_SSE4_1
    | FEATURE_SSE4_2
    | FEATURE_X2APIC
    | FEATURE_MOVBE
    | FEATURE_POPCNT
    | FEATURE_TSC_DEADLINE
    | FEATURE_AES
    | FEATURE_XSAVE
    | FEATURE_AVX
    | FEATURE_F16C
    | FEATURE_RDRAND;

/// Leaf 5 smallest/largest monitor-line size, and ECX bits: extensions enumerated, interrupts
/// break MWAIT even when masked.
const MONITOR_LINE_BYTES: u32 = 64;
const MONITOR_ECX_EXTENSIONS_AND_BREAK: u32 = 0b11;

/// Leaf 6 EAX bit 2: the APIC timer keeps running in every power state (Linux "arat"), true
/// because the machine has no power states.
const THERMAL_POWER_ARAT: u32 = 1 << 2;

// Leaf 7 subleaf 0 EBX.
/// rdfsbase, rdgsbase, wrfsbase, wrgsbase.
const STRUCTURED_FEATURE_FSGSBASE: u32 = 1 << 0;
/// BMI1: andn, bextr, blsi, blsmsk, blsr (VEX) and tzcnt.
const STRUCTURED_FEATURE_BMI1: u32 = 1 << 3;
/// AVX2: the 256-bit integer set, vperm*, vpgather*, vpmaskmov*, variable shifts, broadcasts.
const STRUCTURED_FEATURE_AVX2: u32 = 1 << 5;
const STRUCTURED_FEATURE_SMEP: u32 = 1 << 7;
/// BMI2: bzhi, mulx, pdep, pext, rorx, sarx, shlx, shrx (VEX).
const STRUCTURED_FEATURE_BMI2: u32 = 1 << 8;
/// ERMS (here and FSRM in EDX) make the kernel's memcpy, clear_page and copy_user use `rep
/// movsb`/`rep stosb` for everything, which is only a win because `Cpu::rep_bulk` turns each
/// page of such an instruction into one host copy and `REP_BYTES_PER_SLICE` lets one run-loop
/// round trip cover 32 KiB. Measured: a `dd` of 2 GB zero -> null (all copy_to_user) went from
/// 3.6 s to 0.3 s.
const STRUCTURED_FEATURE_ERMS: u32 = 1 << 9;
const STRUCTURED_FEATURE_INVPCID: u32 = 1 << 10;
/// AVX-512 foundation and its sub-extensions (EBX): F, DQ, IFMA, CD, BW, VL. PF and ER (Xeon Phi
/// only) are absent. The state they need is XCR0 bits 7:5 (leaf 0xd).
const STRUCTURED_FEATURE_AVX512F: u32 = 1 << 16;
const STRUCTURED_FEATURE_AVX512DQ: u32 = 1 << 17;
const STRUCTURED_FEATURE_RDSEED: u32 = 1 << 18;
/// ADX: adcx and adox.
const STRUCTURED_FEATURE_ADX: u32 = 1 << 19;
const STRUCTURED_FEATURE_SMAP: u32 = 1 << 20;
const STRUCTURED_FEATURE_AVX512IFMA: u32 = 1 << 21;
/// clflushopt and clwb execute as the access check clflush performs; there is no cache to flush.
const STRUCTURED_FEATURE_CLFLUSHOPT: u32 = 1 << 23;
const STRUCTURED_FEATURE_CLWB: u32 = 1 << 24;
const STRUCTURED_FEATURE_AVX512CD: u32 = 1 << 28;
/// SHA-1 and SHA-256 instructions (`crypto.rs`).
const STRUCTURED_FEATURE_SHA: u32 = 1 << 29;
const STRUCTURED_FEATURE_AVX512BW: u32 = 1 << 30;
const STRUCTURED_FEATURE_AVX512VL: u32 = 1 << 31;
const STRUCTURED_EBX: u32 = STRUCTURED_FEATURE_FSGSBASE
    | STRUCTURED_FEATURE_BMI1
    | STRUCTURED_FEATURE_AVX2
    | STRUCTURED_FEATURE_SMEP
    | STRUCTURED_FEATURE_BMI2
    | STRUCTURED_FEATURE_ERMS
    | STRUCTURED_FEATURE_INVPCID
    | STRUCTURED_FEATURE_AVX512F
    | STRUCTURED_FEATURE_AVX512DQ
    | STRUCTURED_FEATURE_RDSEED
    | STRUCTURED_FEATURE_ADX
    | STRUCTURED_FEATURE_SMAP
    | STRUCTURED_FEATURE_AVX512IFMA
    | STRUCTURED_FEATURE_CLFLUSHOPT
    | STRUCTURED_FEATURE_CLWB
    | STRUCTURED_FEATURE_AVX512CD
    | STRUCTURED_FEATURE_SHA
    | STRUCTURED_FEATURE_AVX512BW
    | STRUCTURED_FEATURE_AVX512VL;

// Leaf 7 subleaf 0 ECX. GFNI, VAES and VPCLMULQDQ also cover the legacy and VEX encodings.
const STRUCTURED_ECX_AVX512_VBMI: u32 = 1 << 1;
const STRUCTURED_ECX_UMIP: u32 = 1 << 2;
const STRUCTURED_ECX_AVX512_VBMI2: u32 = 1 << 6;
const STRUCTURED_ECX_GFNI: u32 = 1 << 8;
const STRUCTURED_ECX_VAES: u32 = 1 << 9;
const STRUCTURED_ECX_VPCLMULQDQ: u32 = 1 << 10;
const STRUCTURED_ECX_AVX512_VNNI: u32 = 1 << 11;
const STRUCTURED_ECX_AVX512_BITALG: u32 = 1 << 12;
const STRUCTURED_ECX_AVX512_VPOPCNTDQ: u32 = 1 << 14;
/// rdpid: IA32_TSC_AUX into a register.
const STRUCTURED_ECX_RDPID: u32 = 1 << 22;
const STRUCTURED_ECX: u32 = STRUCTURED_ECX_AVX512_VBMI
    | STRUCTURED_ECX_UMIP
    | STRUCTURED_ECX_AVX512_VBMI2
    | STRUCTURED_ECX_GFNI
    | STRUCTURED_ECX_VAES
    | STRUCTURED_ECX_VPCLMULQDQ
    | STRUCTURED_ECX_AVX512_VNNI
    | STRUCTURED_ECX_AVX512_BITALG
    | STRUCTURED_ECX_AVX512_VPOPCNTDQ
    | STRUCTURED_ECX_RDPID;

// Leaf 7 subleaf 0 EDX: the speculation-control interface. The MSRs behind these bits are
// implemented as inert storage (see `ARCH_CAPABILITIES` in sys.rs).
/// FSRM: fast short `rep movsb`.
const STRUCTURED_FEATURE_FSRM: u32 = 1 << 4;
/// IA32_SPEC_CTRL and IA32_PRED_CMD (IBRS and IBPB).
const STRUCTURED_FEATURE_SPEC_CTRL: u32 = 1 << 26;
/// STIBP bit in IA32_SPEC_CTRL.
const STRUCTURED_FEATURE_STIBP: u32 = 1 << 27;
/// IA32_FLUSH_CMD.
const STRUCTURED_FEATURE_L1D_FLUSH: u32 = 1 << 28;
/// IA32_ARCH_CAPABILITIES.
const STRUCTURED_FEATURE_ARCH_CAPABILITIES: u32 = 1 << 29;
/// SSBD bit in IA32_SPEC_CTRL.
const STRUCTURED_FEATURE_SSBD: u32 = 1 << 31;
const STRUCTURED_EDX: u32 = STRUCTURED_FEATURE_FSRM
    | STRUCTURED_FEATURE_SPEC_CTRL
    | STRUCTURED_FEATURE_STIBP
    | STRUCTURED_FEATURE_L1D_FLUSH
    | STRUCTURED_FEATURE_ARCH_CAPABILITIES
    | STRUCTURED_FEATURE_SSBD;

/// Leaf 7 subleaf 1 EAX bit 5: AVX512_BF16 (vcvtne2ps2bf16, vcvtneps2bf16, vdpbf16ps).
const STRUCTURED_1_EAX_AVX512_BF16: u32 = 1 << 5;
/// Highest subleaf of leaf 7.
const STRUCTURED_MAX_SUBLEAF: u32 = 1;

// Leaf 0xB (and AMD's 0x8000_001E) describe the topology; ECX of leaf 0xB: level type in bits
// 15:8 (1 = SMT, 2 = core), level number in 7:0, which echoes the subleaf for the invalid level
// that ends the list.
const TOPOLOGY_LEVEL_TYPE_SHIFT: u32 = 8;
const TOPOLOGY_LEVEL_TYPE_SMT: u32 = 1;
const TOPOLOGY_LEVEL_TYPE_CORE: u32 = 2;
const TOPOLOGY_SUBLEAF_SMT: u32 = 0;
const TOPOLOGY_SUBLEAF_CORE: u32 = 1;

// Leaf 0xD subleaf 1 EAX: which of the optional XSAVE instructions exist.
/// Leaf 0xd subleaf 1 EAX bit 0.
const XSTATE_FEATURE_XSAVEOPT: u32 = 1 << 0;
/// Leaf 0xd subleaf 1 EAX bit 1: xsavec and the compacted format in xrstor.
const XSTATE_FEATURE_XSAVEC: u32 = 1 << 1;
/// Leaf 0xd subleaf 1 EAX bit 2: xgetbv with ECX=1 (XINUSE).
const XSTATE_FEATURE_XGETBV1: u32 = 1 << 2;
/// Leaf 0xd subleaf 1 EAX bit 3: xsaves, xrstors and IA32_XSS.
const XSTATE_FEATURE_XSAVES: u32 = 1 << 3;

// Leaf 0x8000_0001 ECX.
/// LAHF/SAHF in 64-bit mode.
const EXT_FEATURE_LAHF_LM: u32 = 1 << 0;
/// Core multi-processing legacy mode: set when there is more than one logical processor, which
/// tells legacy topology code that leaf 1 EBX counts cores.
const EXT_FEATURE_CMP_LEGACY: u32 = 1 << 1;
/// AMD SVM.
const EXT_FEATURE_SVM: u32 = 1 << 2;
/// Advanced bit manipulation, i.e. LZCNT (POPCNT has its own bit in leaf 1).
const EXT_FEATURE_ABM: u32 = 1 << 5;
/// PREFETCH and PREFETCHW (`0F 0D`, hints that execute as no-ops). Windows 8.1 and later refuse
/// to start without it.
const EXT_FEATURE_PREFETCHW: u32 = 1 << 8;
/// Leaves 0x8000_001D and 0x8000_001E (cache properties, extended APIC id and core topology).
const EXT_FEATURE_TOPOEXT: u32 = 1 << 22;
// Leaf 0x8000_0001 EDX, besides the aliases of leaf 1.
const EXT_FEATURE_SYSCALL: u32 = 1 << 11;
const EXT_FEATURE_NX: u32 = 1 << 20;
/// 1 GiB pages (PDPT entries with PS set; tested in `mem.rs`).
const EXT_FEATURE_PDPE1GB: u32 = 1 << 26;
/// rdtscp: rdtsc plus IA32_TSC_AUX in ECX.
const EXT_FEATURE_RDTSCP: u32 = 1 << 27;
const EXT_FEATURE_LONG_MODE: u32 = 1 << 29;

/// 48 bytes returned over leaves 0x8000_0002..=0x8000_0004, NUL padded. The name QEMU gives the
/// same model; the SKU strings carry core counts that would be wrong for a configurable machine.
const BRAND_STRING: &[u8; 48] =
    b"AMD EPYC-Genoa Processor\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

// Cache and TLB geometry of one Zen 4 core complex die, reported by leaves 0x8000_0005,
// 0x8000_0006 and 0x8000_001D from the same numbers so they cannot disagree.
const CACHE_LINE_BYTES: u32 = 64;
const L1_DATA_BYTES: u32 = 32 * 1024;
const L1_DATA_WAYS: u32 = 8;
const L1_INSTRUCTION_BYTES: u32 = 32 * 1024;
const L1_INSTRUCTION_WAYS: u32 = 8;
const L2_BYTES: u32 = 1024 * 1024;
const L2_WAYS: u32 = 8;
const L3_BYTES: u32 = 32 * 1024 * 1024;
const L3_WAYS: u32 = 16;
/// Logical processors that share one L3: the 8 cores (16 threads) of a die.
const L3_SHARING_THREADS: u32 = 16;
const L1_DATA_TLB_ENTRIES: u32 = 72;
const L1_INSTRUCTION_TLB_ENTRIES: u32 = 64;
const L2_DATA_TLB_ENTRIES: u32 = 3072;
const L2_INSTRUCTION_TLB_ENTRIES: u32 = 512;
const L2_TLB_WAYS: u32 = 8;

/// Leaf 0x8000_0005 TLB associativity field meaning fully associative.
const TLB_FULLY_ASSOCIATIVE: u32 = 0xff;
/// Leaf 0x8000_0006 associativity field meaning fully associative.
const L2_FULLY_ASSOCIATIVE: u32 = 0xf;
/// L1 cache lines per tag, in leaves 0x8000_0005 and 0x8000_0006 (one on every AMD part since K8).
const LINES_PER_TAG: u32 = 1;
const L3_SIZE_UNIT_BYTES: u32 = 512 * 1024;

/// The 4-bit associativity code of leaf 0x8000_0006 (L2 and L3 and the L2 TLBs): not the way
/// count itself above 4.
fn l2_associativity_code(ways: u32) -> u32 {
    match ways {
        0 => 0,
        1 => 1,
        2 => 2,
        4 => 4,
        8 => 6,
        16 => 8,
        32 => 0xa,
        48 => 0xb,
        64 => 0xc,
        96 => 0xd,
        128 => 0xe,
        _ => L2_FULLY_ASSOCIATIVE,
    }
}

/// Leaf 0x8000_0019 EAX: the L1 TLBs for 1 GiB pages, data (31:16) then instruction (15:0), each
/// as associativity code (15:12) and entries (11:0). Zen's L1 TLBs are fully associative and hold
/// any page size, so these are the same entry counts as leaf 0x8000_0005. EBX would describe the
/// L2 TLBs for 1 GiB pages; Zen's L2 TLBs do not hold them (a 1 GiB page is cached in L1 only),
/// so it reads zero.
fn l1_tlb_1g_descriptor() -> u32 {
    (L2_FULLY_ASSOCIATIVE << 12 | L1_DATA_TLB_ENTRIES) << 16
        | (L2_FULLY_ASSOCIATIVE << 12 | L1_INSTRUCTION_TLB_ENTRIES)
}

/// Leaf 0x8000_0005 ECX or EDX: an L1 cache as size in KiB (31:24), ways (23:16), lines per tag
/// (15:8) and line size (7:0).
fn l1_cache_descriptor(bytes: u32, ways: u32) -> u32 {
    (bytes / 1024) << 24 | ways << 16 | LINES_PER_TAG << 8 | CACHE_LINE_BYTES
}

/// A 2M/4M or 4K TLB pair of leaf 0x8000_0005: data (31:16) then instruction (15:0), each as
/// associativity (high byte) and entries (low byte).
fn l1_tlb_descriptor() -> u32 {
    (TLB_FULLY_ASSOCIATIVE << 8 | L1_DATA_TLB_ENTRIES) << 16
        | (TLB_FULLY_ASSOCIATIVE << 8 | L1_INSTRUCTION_TLB_ENTRIES)
}

/// A TLB pair of leaf 0x8000_0006: data (31:16) then instruction (15:0), each as associativity
/// code (15:12) and entries (11:0).
fn l2_tlb_descriptor() -> u32 {
    let code = l2_associativity_code(L2_TLB_WAYS);
    (code << 12 | L2_DATA_TLB_ENTRIES) << 16 | (code << 12 | L2_INSTRUCTION_TLB_ENTRIES)
}

/// Leaf 0x8000_0006 ECX: the L2 cache as size in KiB (31:16), associativity code (15:12), lines
/// per tag (11:8) and line size (7:0).
fn l2_cache_descriptor() -> u32 {
    (L2_BYTES / 1024) << 16
        | l2_associativity_code(L2_WAYS) << 12
        | LINES_PER_TAG << 8
        | CACHE_LINE_BYTES
}

/// Leaf 0x8000_0006 EDX: the L3 as size in 512 KiB units (31:18), the same fields below.
fn l3_cache_descriptor() -> u32 {
    (L3_BYTES / L3_SIZE_UNIT_BYTES) << 18
        | l2_associativity_code(L3_WAYS) << 12
        | LINES_PER_TAG << 8
        | CACHE_LINE_BYTES
}

/// Leaf 0x8000_001D: one cache per subleaf, ended by a subleaf of type 0.
#[derive(Clone, Copy)]
struct CacheLevel {
    /// 1 data, 2 instruction, 3 unified.
    kind: u32,
    level: u32,
    bytes: u32,
    ways: u32,
    /// Whether the cache is shared by all the threads of a die (L3) or private to a core.
    per_die: bool,
}

const CACHE_KIND_DATA: u32 = 1;
const CACHE_KIND_INSTRUCTION: u32 = 2;
const CACHE_KIND_UNIFIED: u32 = 3;
const CACHES: [CacheLevel; 4] = [
    CacheLevel {
        kind: CACHE_KIND_DATA,
        level: 1,
        bytes: L1_DATA_BYTES,
        ways: L1_DATA_WAYS,
        per_die: false,
    },
    CacheLevel {
        kind: CACHE_KIND_INSTRUCTION,
        level: 1,
        bytes: L1_INSTRUCTION_BYTES,
        ways: L1_INSTRUCTION_WAYS,
        per_die: false,
    },
    CacheLevel {
        kind: CACHE_KIND_UNIFIED,
        level: 2,
        bytes: L2_BYTES,
        ways: L2_WAYS,
        per_die: false,
    },
    CacheLevel {
        kind: CACHE_KIND_UNIFIED,
        level: 3,
        bytes: L3_BYTES,
        ways: L3_WAYS,
        per_die: true,
    },
];
/// EAX of leaf 0x8000_001D: type (4:0), level (7:5), self-initializing (8), threads sharing minus
/// one (25:14); fully associative (9) stays clear.
const CACHE_LEVEL_SHIFT: u32 = 5;
const CACHE_SELF_INITIALIZING: u32 = 1 << 8;
const CACHE_SHARING_SHIFT: u32 = 14;
/// EBX: ways minus one (31:22), physical line partitions minus one (21:12), line size minus one
/// (11:0).
const CACHE_WAYS_SHIFT: u32 = 22;

/// Leaf 0x8000_0008 EAX: physical address bits (7:0) and linear address bits (15:8).
const ADDRESS_SIZES: u32 = (VIRTUAL_ADDRESS_BITS << 8) | PHYSICAL_ADDRESS_BITS;
// Leaf 0x8000_0008 EBX.
/// clzero (0F 01 FC).
const EXT_ADDRESS_CLZERO: u32 = 1 << 0;
/// XSAVE/XRSTOR and FXSAVE/FXRSTOR save and restore the x87 error pointers whether or not an
/// exception is pending. The kernel otherwise does a fnclex/emms/fildl dance on every context
/// switch to avoid a leak this model has no way to have.
const EXT_ADDRESS_XSAVEERPTR: u32 = 1 << 2;
/// wbnoinvd (F3 0F 09).
const EXT_ADDRESS_WBNOINVD: u32 = 1 << 9;
// The AMD spelling of the speculation-control interface (Linux consults these on AMD vendor
// CPUs).
const EXT_ADDRESS_IBPB: u32 = 1 << 12;
const EXT_ADDRESS_IBRS: u32 = 1 << 14;
const EXT_ADDRESS_STIBP: u32 = 1 << 15;
const EXT_ADDRESS_SSBD: u32 = 1 << 24;
/// IBPB also flushes the return predictor; without it Linux marks the CPU "ibpb_no_ret".
const EXT_ADDRESS_IBPB_RET: u32 = 1 << 30;
const EXT_ADDRESS_EBX: u32 = EXT_ADDRESS_CLZERO
    | EXT_ADDRESS_XSAVEERPTR
    | EXT_ADDRESS_WBNOINVD
    | EXT_ADDRESS_IBPB
    | EXT_ADDRESS_IBRS
    | EXT_ADDRESS_STIBP
    | EXT_ADDRESS_SSBD
    | EXT_ADDRESS_IBPB_RET;
/// ECX: number of logical processors minus one (7:0), APIC id width in bits (15:12).
const EXT_ADDRESS_APIC_ID_SIZE_SHIFT: u32 = 12;

/// Leaf 0x8000_0007 EDX bit 8: invariant TSC, so the kernel trusts it as a clocksource.
const APM_INVARIANT_TSC: u32 = 1 << 8;

/// Leaf 0x8000_0021 EAX: LFENCE is always dispatch-serializing (nothing to configure through
/// DE_CFG; the CPU executes in order), and SRSO_NO (see the module comment). ECX: the "not
/// affected" bits for Transient Scheduler Attacks, which Linux keys on family 0x19 (store-queue
/// and L1 variants).
const EXT_FEATURES_2_LFENCE_ALWAYS_SERIALIZING: u32 = 1 << 2;
const EXT_FEATURES_2_SRSO_NO: u32 = 1 << 29;
const EXT_FEATURES_2_TSA_SQ_NO: u32 = 1 << 1;
const EXT_FEATURES_2_TSA_L1_NO: u32 = 1 << 2;

impl Cpu {
    /// The subleaf (ECX) matters for leaves 7, 0xB, 0xD and 0x8000_001D; unknown leaves return
    /// zeros.
    pub fn cpuid(&mut self) {
        let leaf = self.gpr[RAX] as u32;
        let subleaf = self.gpr[RCX] as u32;
        let topology = self.topology;
        let threads = topology.logical_processors > 1;
        let (eax, ebx, ecx, edx): (u32, u32, u32, u32) = match leaf {
            LEAF_VENDOR => (MAX_STANDARD_LEAF, VENDOR_EBX, VENDOR_ECX, VENDOR_EDX),
            LEAF_FEATURES => {
                let osxsave = if self.cr4 & CR4_OSXSAVE != 0 {
                    FEATURE_OSXSAVE
                } else {
                    0
                };
                (
                    SIGNATURE,
                    (topology.x2apic_id & 0xff) << LEAF1_EBX_APIC_ID_SHIFT
                        | topology.logical_processors.min(0xff)
                            << LEAF1_EBX_LOGICAL_PROCESSORS_SHIFT
                        | CLFLUSH_LINE_UNITS << LEAF1_EBX_CLFLUSH_SHIFT,
                    LEAF1_ECX | osxsave,
                    LEAF1_EDX | if threads { FEATURE_HTT } else { 0 },
                )
            }
            LEAF_MONITOR => (
                MONITOR_LINE_BYTES,
                MONITOR_LINE_BYTES,
                MONITOR_ECX_EXTENSIONS_AND_BREAK,
                0,
            ),
            LEAF_THERMAL_POWER => (THERMAL_POWER_ARAT, 0, 0, 0),
            LEAF_STRUCTURED_EXTENDED_FEATURES => match subleaf {
                0 => (
                    STRUCTURED_MAX_SUBLEAF,
                    STRUCTURED_EBX,
                    STRUCTURED_ECX,
                    STRUCTURED_EDX,
                ),
                1 => (STRUCTURED_1_EAX_AVX512_BF16, 0, 0, 0),
                _ => (0, 0, 0, 0),
            },
            LEAF_EXTENDED_TOPOLOGY => {
                let (shift, count, level_type) = match subleaf {
                    TOPOLOGY_SUBLEAF_SMT => (
                        topology.smt_shift(),
                        topology.threads_per_core,
                        TOPOLOGY_LEVEL_TYPE_SMT,
                    ),
                    TOPOLOGY_SUBLEAF_CORE => (
                        topology.package_shift(),
                        topology.logical_processors,
                        TOPOLOGY_LEVEL_TYPE_CORE,
                    ),
                    // the invalid level that terminates the list; ECX echoes the subleaf
                    _ => (0, 0, 0),
                };
                (
                    shift,
                    count,
                    level_type << TOPOLOGY_LEVEL_TYPE_SHIFT | subleaf & 0xff,
                    topology.x2apic_id,
                )
            }
            // Subleaf 0: supported XCR0 bits, size of the area for the currently enabled bits and
            // for all supported ones. Subleaf 1: which optional instructions exist, and the size
            // of the compacted format for the enabled bits (ECX:EDX are the IA32_XSS bits the OS
            // may enable).
            LEAF_XSTATE => match subleaf {
                0 => (
                    xsave::XCR0_SUPPORTED as u32,
                    xsave::enabled_area_size(self.xcr0),
                    xsave::enabled_area_size(xsave::XCR0_SUPPORTED),
                    0,
                ),
                1 => (
                    XSTATE_FEATURE_XSAVEOPT
                        | XSTATE_FEATURE_XSAVEC
                        | XSTATE_FEATURE_XGETBV1
                        | XSTATE_FEATURE_XSAVES,
                    xsave::compacted_area_size(self.xcr0),
                    xsave::XSS_SUPPORTED as u32,
                    (xsave::XSS_SUPPORTED >> 32) as u32,
                ),
                // Components 2 (AVX) and 5-7 (AVX-512): size and offset in the standard format
                _ => match xsave::COMPONENTS.iter().find(|part| part.index == subleaf) {
                    Some(part) => (part.size as u32, part.offset as u32, 0, 0),
                    None => (0, 0, 0, 0),
                },
            },
            LEAF_EXT_MAX => (MAX_EXTENDED_LEAF, VENDOR_EBX, VENDOR_ECX, VENDOR_EDX),
            LEAF_EXT_FEATURES => (
                SIGNATURE,
                0,
                EXT_FEATURE_LAHF_LM
                    | if threads { EXT_FEATURE_CMP_LEGACY } else { 0 }
                    | EXT_FEATURE_SVM
                    | EXT_FEATURE_ABM
                    | EXT_FEATURE_PREFETCHW
                    | EXT_FEATURE_TOPOEXT,
                EXT_FEATURE_SYSCALL
                    | EXT_FEATURE_NX
                    | EXT_FEATURE_PDPE1GB
                    | EXT_FEATURE_RDTSCP
                    | EXT_FEATURE_LONG_MODE
                    | (LEAF1_EDX & EXT_ALIASED_EDX),
            ),
            LEAF_BRAND_FIRST..=LEAF_BRAND_LAST => {
                let offset = (leaf - LEAF_BRAND_FIRST) as usize * 16;
                let word = |k: usize| {
                    u32::from_le_bytes(
                        BRAND_STRING[offset + k * 4..offset + k * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                };
                (word(0), word(1), word(2), word(3))
            }
            LEAF_L1_CACHE_TLB => (
                l1_tlb_descriptor(),
                l1_tlb_descriptor(),
                l1_cache_descriptor(L1_DATA_BYTES, L1_DATA_WAYS),
                l1_cache_descriptor(L1_INSTRUCTION_BYTES, L1_INSTRUCTION_WAYS),
            ),
            LEAF_L2_L3_CACHE_TLB => (
                l2_tlb_descriptor(),
                l2_tlb_descriptor(),
                l2_cache_descriptor(),
                l3_cache_descriptor(),
            ),
            LEAF_ADVANCED_POWER => (0, 0, 0, APM_INVARIANT_TSC),
            LEAF_ADDRESS_SIZES => (
                ADDRESS_SIZES,
                EXT_ADDRESS_EBX,
                (topology.logical_processors - 1) & 0xff
                    | ceil_log2(topology.logical_processors) << EXT_ADDRESS_APIC_ID_SIZE_SHIFT,
                0,
            ),
            // revision 1, number of ASIDs, features
            LEAF_SVM => (1, super::svm::SVM_ASIDS, 0, super::svm::SVM_FEATURES),
            LEAF_TLB_1G => (l1_tlb_1g_descriptor(), 0, 0, 0),
            LEAF_CACHE_PROPERTIES => match CACHES.get(subleaf as usize) {
                Some(cache) => self.cache_properties(cache),
                None => (0, 0, 0, 0),
            },
            // EAX: the extended APIC id. EBX: threads per core minus one (15:8), core id (7:0).
            // ECX: node id (7:0) and nodes per processor minus one (10:8): one node.
            LEAF_EXTENDED_APIC_ID => (
                topology.x2apic_id,
                (topology.threads_per_core - 1) << 8 | topology.core_id() & 0xff,
                0,
                0,
            ),
            LEAF_EXT_FEATURES_2 => (
                EXT_FEATURES_2_LFENCE_ALWAYS_SERIALIZING | EXT_FEATURES_2_SRSO_NO,
                0,
                EXT_FEATURES_2_TSA_SQ_NO | EXT_FEATURES_2_TSA_L1_NO,
                0,
            ),
            _ => (0, 0, 0, 0),
        };
        self.gpr[RAX] = eax as u64;
        self.gpr[RBX] = ebx as u64;
        self.gpr[RCX] = ecx as u64;
        self.gpr[RDX] = edx as u64;
    }

    /// One subleaf of leaf 0x8000_001D.
    fn cache_properties(&self, cache: &CacheLevel) -> (u32, u32, u32, u32) {
        let topology = self.topology;
        let sharing = if cache.per_die {
            topology.logical_processors.min(L3_SHARING_THREADS)
        } else {
            topology.threads_per_core
        };
        let sets = cache.bytes / (cache.ways * CACHE_LINE_BYTES);
        (
            cache.kind
                | cache.level << CACHE_LEVEL_SHIFT
                | CACHE_SELF_INITIALIZING
                | (sharing - 1) << CACHE_SHARING_SHIFT,
            (cache.ways - 1) << CACHE_WAYS_SHIFT | (CACHE_LINE_BYTES - 1),
            sets - 1,
            0,
        )
    }
}

#[cfg(test)]
#[path = "cpuid_tests.rs"]
mod tests;
