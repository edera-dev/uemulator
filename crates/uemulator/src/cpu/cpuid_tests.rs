//! CPUID leaf contents. Expected values are spelled out from the AMD APM and the Zen 5 (family 1Ah
//! model 02h) leaves, independent of the constants in `cpuid.rs`.

use super::*;
use crate::testkit::Harness;

fn blank() -> Harness {
    Harness::new(&[0xf4])
}

fn cpuid(harness: &mut Harness, leaf: u32, subleaf: u32) -> [u32; 4] {
    // garbage in the upper halves: only EAX and ECX select, as the SDM says
    harness.cpu.gpr[RAX] = 0xdead_beef_0000_0000 | leaf as u64;
    harness.cpu.gpr[RCX] = 0xdead_beef_0000_0000 | subleaf as u64;
    harness.cpu.cpuid();
    [RAX, RBX, RCX, RDX].map(|register| harness.cpu.gpr[register] as u32)
}

fn bits(positions: &[u32]) -> u32 {
    positions.iter().fold(0, |value, bit| value | 1 << bit)
}

/// A machine with `logical` processors of `threads` threads per core, seen from `x2apic_id`.
fn with_topology(logical: u32, threads: u32, x2apic_id: u32) -> Harness {
    let mut harness = blank();
    harness.cpu.topology = CpuTopology {
        logical_processors: logical,
        threads_per_core: threads,
        x2apic_id,
        brand: harness.cpu.topology.brand,
    };
    harness
}

#[test]
fn leaf_0_reports_the_max_standard_leaf_and_vendor_string() {
    let mut h = blank();
    let [eax, ebx, ecx, edx] = cpuid(&mut h, 0, 0);
    assert_eq!(eax, 0x10);
    // "AuthenticAMD" comes back in EBX, EDX, ECX order.
    assert_eq!(
        [ebx, edx, ecx].map(u32::to_le_bytes).concat(),
        b"AuthenticAMD"
    );
}

#[test]
fn the_signature_is_family_1ah_model_02h_stepping_1() {
    let mut h = blank();
    for leaf in [1, 0x8000_0001] {
        let [eax, ..] = cpuid(&mut h, leaf, 0);
        let (stepping, model, base_family) = (eax & 0xf, (eax >> 4) & 0xf, (eax >> 8) & 0xf);
        let (extended_model, extended_family) = ((eax >> 16) & 0xf, (eax >> 20) & 0xff);
        assert_eq!(
            (
                stepping,
                model,
                base_family,
                extended_model,
                extended_family
            ),
            (1, 2, 0xf, 0, 0xb)
        );
        assert_eq!(base_family + extended_family, 0x1a);
        assert_eq!(model | extended_model << 4, 0x02);
        assert_eq!(eax, 0x00b0_0f21);
    }
}

#[test]
fn leaf_1_ebx_has_a_64_byte_clflush_line_and_the_processor_count_and_apic_id() {
    let mut h = blank();
    let [_, ebx, ..] = cpuid(&mut h, 1, 0);
    assert_eq!((ebx >> 8) & 0xff, 8, "CLFLUSH line size in 8-byte units");
    assert_eq!((ebx >> 16) & 0xff, 1, "logical processor count");
    assert_eq!(ebx >> 24, 0, "initial APIC id");
    assert_eq!(ebx & 0xff, 0, "brand index");
    let mut h = with_topology(12, 2, 0x105);
    let [_, ebx, ..] = cpuid(&mut h, 1, 0);
    assert_eq!((ebx >> 16) & 0xff, 12);
    assert_eq!(ebx >> 24, 0x05, "only the low 8 bits of the x2APIC id fit");
}

#[test]
fn leaf_1_edx_advertises_exactly_the_implemented_features() {
    let mut h = blank();
    let [_, _, _, edx] = cpuid(&mut h, 1, 0);
    // FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE-36 CLFSH MMX FXSR SSE
    // SSE2
    let expected = bits(&[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 23, 24, 25, 26,
    ]);
    assert_eq!(edx, expected, "{edx:#034b}");
    assert_eq!(edx, 0x078b_fbff);
    for (bit, name) in [
        (18, "PSN"),
        (21, "DS"),
        (22, "ACPI"),
        (27, "SS"),
        (28, "HTT"),
    ] {
        assert_eq!(edx & (1 << bit), 0, "{name} must not be advertised");
    }
}

#[test]
fn hyper_threading_follows_the_processor_count() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 1, 0)[3] & (1 << 28), 0);
    let mut h = with_topology(2, 2, 0);
    assert_ne!(cpuid(&mut h, 1, 0)[3] & (1 << 28), 0);
    let mut h = with_topology(3, 1, 0);
    assert_ne!(
        cpuid(&mut h, 1, 0)[3] & (1 << 28),
        0,
        "several cores count too"
    );
}

#[test]
fn leaf_1_ecx_advertises_only_what_is_implemented() {
    let mut h = blank();
    let [_, _, ecx, _] = cpuid(&mut h, 1, 0);
    // SSE3 PCLMULQDQ MONITOR SSSE3 FMA CX16 PCID SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES XSAVE AVX
    // F16C RDRAND
    let expected = bits(&[
        0, 1, 3, 9, 12, 13, 17, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30,
    ]);
    assert_eq!(
        ecx, expected,
        "no VMX, SMX, hypervisor or PDCM bit, and OSXSAVE only follows CR4"
    );
    assert_eq!(
        ecx & 1 << 24,
        0,
        "no TSC-deadline timer, as on every AMD part"
    );
    h.cpu.cr4 |= CR4_OSXSAVE;
    assert_ne!(cpuid(&mut h, 1, 0)[2] & (1 << 27), 0);
    h.cpu.cr4 &= !CR4_OSXSAVE;
    assert_eq!(cpuid(&mut h, 1, 0)[2] & (1 << 27), 0);
}

#[test]
fn leaf_1_ignores_the_subleaf() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 1, 0), cpuid(&mut h, 1, 7));
}

#[test]
fn leaf_5_describes_monitor_mwait() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 5, 0), [64, 64, 0b11, 0]);
}

#[test]
fn leaf_6_has_the_always_running_apic_timer_and_aperf_mperf() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 6, 0), [1 << 2, 0, 1, 0]);
}

#[test]
fn leaf_7_subleaf_0_features() {
    let mut h = blank();
    // FSGSBASE BMI1 AVX2 SMEP BMI2 ERMS INVPCID PQM PQE AVX512F AVX512DQ RDSEED ADX SMAP
    // AVX512IFMA CLFLUSHOPT CLWB AVX512CD SHA AVX512BW AVX512VL
    let ebx = bits(&[
        0, 3, 5, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24, 28, 29, 30, 31,
    ]);
    // AVX512_VBMI UMIP PKU AVX512_VBMI2 CET_SS GFNI VAES VPCLMULQDQ AVX512_VNNI AVX512_BITALG
    // AVX512_VPOPCNTDQ LA57 RDPID BUS_LOCK_DETECT MOVDIRI MOVDIR64B; OSPKE only follows CR4
    let ecx = bits(&[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 14, 16, 22, 24, 27, 28]);
    // FSRM and AVX512_VP2INTERSECT; the speculation controls are in the AMD leaves
    let edx = bits(&[4, 8]);
    // EAX is the highest subleaf
    assert_eq!(cpuid(&mut h, 7, 0), [1, ebx, ecx, edx]);
}

#[test]
fn leaf_7_ospke_mirrors_cr4_pke() {
    const OSPKE: u32 = 1 << 4;
    const CR4_PKE: u64 = 1 << 22;
    let mut h = blank();
    h.cpu.cr4 |= CR4_PKE;
    assert_ne!(cpuid(&mut h, 7, 0)[2] & OSPKE, 0);
    h.cpu.cr4 &= !CR4_PKE;
    assert_eq!(cpuid(&mut h, 7, 0)[2] & OSPKE, 0);
}

#[test]
fn leaf_7_subleaf_1_has_avx_vnni_and_avx512_bf16_and_nothing_else() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 7, 1), [bits(&[4, 5]), 0, 0, 0]);
    for subleaf in [2, 5, 0x8000_0000] {
        assert_eq!(cpuid(&mut h, 7, subleaf), [0, 0, 0, 0], "subleaf {subleaf}");
    }
}

#[test]
fn other_standard_leaves_are_zero() {
    let mut h = blank();
    for leaf in [2, 3, 4, 8, 0xa, 0xc, 0xe, 0x11, 0x14, 0x15, 0x1f] {
        assert_eq!(cpuid(&mut h, leaf, 0), [0, 0, 0, 0], "leaf {leaf:#x}");
    }
}

#[test]
fn leaf_0xb_describes_one_logical_processor_with_x2apic_id_0() {
    let mut h = blank();
    // SMT level: 0 shift bits, 1 thread, level type 1; core level: 1 bit (a one-bit field has to
    // exist to be well formed), 1 processor, type 2; then the invalid level.
    assert_eq!(cpuid(&mut h, 0xb, 0), [0, 1, 0x100, 0]);
    assert_eq!(cpuid(&mut h, 0xb, 1), [1, 1, 0x201, 0]);
    assert_eq!(cpuid(&mut h, 0xb, 2), [0, 0, 2, 0]);
}

#[test]
fn leaf_0xb_follows_the_configured_topology() {
    // 4 cores x 2 threads, the second thread of core 2: x2APIC id 0b101
    let mut h = with_topology(8, 2, 5);
    assert_eq!(cpuid(&mut h, 0xb, 0), [1, 2, 0x100, 5]);
    assert_eq!(cpuid(&mut h, 0xb, 1), [3, 8, 0x201, 5]);
    assert_eq!(cpuid(&mut h, 0xb, 2), [0, 0, 2, 5]);
    // 6 single-threaded cores: the package field needs 3 bits
    let mut h = with_topology(6, 1, 4);
    assert_eq!(cpuid(&mut h, 0xb, 0), [0, 1, 0x100, 4]);
    assert_eq!(cpuid(&mut h, 0xb, 1), [3, 6, 0x201, 4]);
}

#[test]
fn leaf_0xd_describes_the_xsave_area() {
    let mut h = blank();
    // supported XCR0 bits (x87, SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM, PKRU), size for the
    // enabled set, size for all supported
    assert_eq!(cpuid(&mut h, 0xd, 0), [0x2e7, 576, 2696, 0]);
    h.cpu.xcr0 = 7;
    assert_eq!(cpuid(&mut h, 0xd, 0), [0x2e7, 832, 2696, 0]);
    h.cpu.xcr0 = 0xe7;
    assert_eq!(cpuid(&mut h, 0xd, 0), [0x2e7, 2688, 2696, 0]);
    h.cpu.xcr0 = 0x2e7;
    assert_eq!(cpuid(&mut h, 0xd, 0), [0x2e7, 2696, 2696, 0]);
    // xsaveopt, xsavec, xgetbv1, xsaves; EBX is the compacted size for the enabled set; ECX the
    // IA32_XSS components (CET_U, CET_S)
    assert_eq!(cpuid(&mut h, 0xd, 1), [0xf, 2440, 0x1800, 0]);
    h.cpu.xcr0 = 0xe7;
    assert_eq!(cpuid(&mut h, 0xd, 1), [0xf, 2432, 0x1800, 0]);
    h.cpu.xcr0 = 7;
    assert_eq!(cpuid(&mut h, 0xd, 1), [0xf, 832, 0x1800, 0]);
    h.cpu.xcr0 = 0xe7;
    // component 2 (AVX): 256 bytes at offset 576
    assert_eq!(cpuid(&mut h, 0xd, 2), [256, 576, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 3), [0, 0, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 4), [0, 0, 0, 0]);
    // opmask: 64 bytes at 1088; ZMM_Hi256: 512 at 1152; Hi16_ZMM: 1024 at 1664
    assert_eq!(cpuid(&mut h, 0xd, 5), [64, 1088, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 6), [512, 1152, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 7), [1024, 1664, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 8), [0, 0, 0, 0]);
    // PKRU: 8 bytes at 2688, a user component without alignment needs
    assert_eq!(cpuid(&mut h, 0xd, 9), [8, 2688, 0, 0]);
    assert_eq!(cpuid(&mut h, 0xd, 10), [0, 0, 0, 0]);
}

#[test]
fn hypervisor_leaves_are_absent() {
    let mut h = blank();
    for leaf in [0x4000_0000, 0x4000_0001, 0x4000_0100] {
        assert_eq!(cpuid(&mut h, leaf, 0), [0, 0, 0, 0], "leaf {leaf:#x}");
    }
}

#[test]
fn extended_leaf_0_reports_the_max_extended_leaf_and_vendor() {
    let mut h = blank();
    let [eax, ebx, ecx, edx] = cpuid(&mut h, 0x8000_0000, 0);
    assert_eq!(eax, 0x8000_0026);
    assert_eq!(
        [ebx, edx, ecx].map(u32::to_le_bytes).concat(),
        b"AuthenticAMD"
    );
}

#[test]
fn extended_leaf_1_ecx_has_the_zen_5_features_that_are_implemented() {
    let mut h = blank();
    let [_, ebx, ecx, _] = cpuid(&mut h, 0x8000_0001, 0);
    assert_eq!(ebx, 0);
    // LAHF_LM SVM EXTAPIC CR8_LEGACY ABM SSE4A MISALIGNSSE 3DNOWPREFETCH OSVW TCE TOPOEXT
    // PERFCTR_CORE PERFCTR_NB BPEXT PERFCTR_LLC MWAITX
    let expected = bits(&[0, 2, 3, 4, 5, 6, 7, 8, 9, 17, 22, 23, 24, 26, 28, 29]);
    assert_eq!(ecx, expected, "{ecx:#034b}");
    for (bit, name) in [
        (10, "IBS"),
        (11, "XOP"),
        (12, "SKINIT"),
        (13, "WDT"),
        (15, "LWP"),
        (16, "FMA4"),
        (30, "ADDR_MASK_EXT"),
    ] {
        assert_eq!(ecx & (1 << bit), 0, "{name} is not implemented");
    }
    let mut h = with_topology(2, 2, 0);
    assert_eq!(
        cpuid(&mut h, 0x8000_0001, 0)[2],
        expected | 1 << 1,
        "CMP_LEGACY"
    );
}

#[test]
fn extended_leaf_1_edx_has_syscall_nx_page1gb_rdtscp_long_mode_and_the_amd_aliases() {
    let mut h = blank();
    let [_, _, _, edx] = cpuid(&mut h, 0x8000_0001, 0);
    // SYSCALL NX MMXEXT FFXSR PDPE1GB RDTSCP LM, then the aliased FPU VME TSC MSR PAE CX8 PGE CMOV
    // PAT MMX FXSR
    let expected = bits(&[
        11, 20, 22, 25, 26, 27, 29, 0, 1, 4, 5, 6, 8, 13, 15, 16, 23, 24,
    ]);
    assert_eq!(edx, expected, "{edx:#034b}");
    for (bit, name) in [(30, "3DNOWEXT"), (31, "3DNOW")] {
        assert_eq!(edx & (1 << bit), 0, "{name}");
    }
}

#[test]
fn brand_string_spans_three_leaves() {
    let mut h = blank();
    let mut brand = Vec::new();
    for leaf in 0x8000_0002..=0x8000_0004 {
        for register in cpuid(&mut h, leaf, 0) {
            brand.extend_from_slice(&register.to_le_bytes());
        }
    }
    assert_eq!(brand.len(), 48);
    assert!(brand.starts_with(b"AMD EPYC 9015 8-Core Processor\0"));
    assert!(
        brand[30..].iter().all(|&byte| byte == 0),
        "the rest is NUL padding"
    );
}

#[test]
fn l1_cache_leaf_describes_48k_12way_data_and_32k_8way_instruction_caches() {
    let mut h = blank();
    let [l1_dtlb_2m, l1_dtlb_4k, l1d, l1i] = cpuid(&mut h, 0x8000_0005, 0);
    assert_eq!(
        l1d, 0x300c_0140,
        "48 KiB, 12 ways, 1 line per tag, 64-byte lines"
    );
    assert_eq!(l1i, 0x2008_0140, "32 KiB, 8 ways");
    // fully associative (0xff) 96-entry data and 64-entry instruction TLBs
    assert_eq!(l1_dtlb_2m, 0xff60_ff40);
    assert_eq!(l1_dtlb_4k, 0xff60_ff40);
}

#[test]
fn l2_and_l3_leaf_report_1m_16way_and_32m_16way() {
    let mut h = blank();
    let [l2_itlb, l2_dtlb, l2, l3] = cpuid(&mut h, 0x8000_0006, 0);
    assert_eq!(
        l2, 0x0400_8140,
        "1024 KiB, associativity code 8 (16 ways), 64-byte lines"
    );
    assert_eq!(
        l3, 0x0100_8140,
        "64 x 512 KiB, associativity code 8 (16 ways)"
    );
    // data TLB (31:16) 3072 entries, instruction TLB (15:0) 2048 entries, both code 6 (8 ways)
    assert_eq!((l2_itlb, l2_dtlb), (0x6c00_6800, 0x6c00_6800));
}

#[test]
fn advanced_power_leaf_has_scalable_mca_pstates_the_invariant_tsc_boost_and_rapl() {
    let mut h = blank();
    // EBX: McaOverflowRecov SUCCOR ScalableMca; EDX: HwPstate, invariant TSC, CPB, EffFreqRO,
    // RAPL; no TS, TTP, TM, 100MHzSteps, ConnectedStandby
    assert_eq!(
        cpuid(&mut h, 0x8000_0007, 0),
        [0, bits(&[0, 1, 3]), 0, bits(&[7, 8, 9, 10, 14])]
    );
}

#[test]
fn the_machine_check_bank_count_matches_the_scalable_mca_banks() {
    let mut h = blank();
    let banks = h.cpu.rdmsr(0x179).unwrap() & 0xff;
    assert_eq!(banks, 24);
    // every bank's SMCA block exists, and the next one does not
    assert!(h.cpu.rdmsr(0xc000_2005 + 0x10 * (banks as u32 - 1)).is_ok());
    assert!(h.cpu.rdmsr(0xc000_2000 + 0x10 * banks as u32).is_err());
}

#[test]
fn quality_of_service_leaves_describe_l3_monitoring_allocation_and_bandwidth() {
    let mut h = blank();
    // leaf 0xF: highest RMID 255, L3 monitored; subleaf 1: 64-byte occupancy units, 256 RMIDs,
    // occupancy and total and local bandwidth events
    assert_eq!(cpuid(&mut h, 0xf, 0), [0, 255, 0, 1 << 1]);
    assert_eq!(cpuid(&mut h, 0xf, 1), [0, 64, 255, 0b111]);
    // leaf 0x10: L3 allocation with a 16-bit mask, no CDP, 16 classes of service
    assert_eq!(cpuid(&mut h, 0x10, 0), [0, 1 << 1, 0, 0]);
    assert_eq!(cpuid(&mut h, 0x10, 1), [15, 0, 0, 15]);
    // leaf 0x8000_0020: L3 bandwidth enforcement, 11-bit limits, 16 classes
    assert_eq!(cpuid(&mut h, 0x8000_0020, 0), [0, 1 << 1, 0, 0]);
    assert_eq!(cpuid(&mut h, 0x8000_0020, 1), [11, 0, 0, 15]);
    // the registers behind them
    assert_eq!(h.cpu.rdmsr(0xc90 + 15).ok(), Some(0xffff));
    assert_eq!(h.cpu.rdmsr(0xc000_0200 + 15).ok(), Some(1 << 11));
    assert!(h.cpu.rdmsr(0xc90 + 16).is_err());
}

/// Leaf 0x8000_0008 EBX bit 27 promises the CPPC MSRs (`cppc.rs`); amd_pstate reads CAPABILITY_1
/// and REQUEST and writes ENABLE through them.
#[test]
fn cppc_bit_comes_with_its_msrs() {
    let mut h = blank();
    assert_ne!(cpuid(&mut h, 0x8000_0008, 0)[1] & 1 << 27, 0);
    let capability = h.cpu.rdmsr(0xc001_02b0).unwrap();
    assert_ne!(capability >> 24 & 0xff, 0, "highest performance");
    h.cpu.wrmsr(0xc001_02b1, 1).unwrap();
    assert_eq!(h.cpu.rdmsr(0xc001_02b1).unwrap(), 1);
    h.cpu.wrmsr(0xc001_02b3, 0x8064_0064).unwrap();
    assert_eq!(h.cpu.rdmsr(0xc001_02b3).unwrap(), 0x8064_0064);
    assert!(h.cpu.wrmsr(0xc001_02b0, 0).is_err());
    assert_eq!(h.cpu.rdmsr(0xc001_02b4).unwrap(), 0);
}

#[test]
fn leaf_0x8000001a_says_256_bit_operations_and_unaligned_loads_run_at_full_speed() {
    let mut h = blank();
    // MOVU and FP256; FP128 (the Zen 1 split) clear
    assert_eq!(cpuid(&mut h, 0x8000_001a, 0), [0b110, 0, 0, 0]);
}

#[test]
fn address_size_leaf_reports_52_physical_and_57_virtual_bits_and_the_extensions() {
    let mut h = blank();
    let [eax, ebx, ecx, edx] = cpuid(&mut h, 0x8000_0008, 0);
    assert_eq!(eax & 0xff, 52);
    assert_eq!((eax >> 8) & 0xff, 57);
    assert_eq!(eax >> 16, 0);
    // CLZERO IRPERF RDPRU INVLPGB XSAVEERPTR MCOMMIT MBE WBNOINVD IBPB INT_WBINVD IBRS STIBP
    // STIBP_ALWAYS_ON IBRS_SAME_MODE EFER_LMSLE_MBZ PPIN SSBD CPPC PSFD BTC_NO IBPB_RET; no
    // AMD_SSB_NO, VIRT_SSBD, IBRS_ALWAYS_ON
    assert_eq!(
        ebx,
        bits(&[
            0, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 15, 17, 19, 20, 23, 24, 27, 28, 29, 30
        ])
    );
    // EDX: rdpru takes ECX 0 and 1 (23:16); INVLPGB takes up to eight pages (15:0 = 7)
    assert_eq!((ecx, edx), (0, 1 << 16 | 7));
    // NC is the processor count minus one, ApicIdSize the bits the ids need
    let mut h = with_topology(8, 2, 0);
    assert_eq!(cpuid(&mut h, 0x8000_0008, 0)[2], 7 | 3 << 12);
    let mut h = with_topology(6, 1, 0);
    assert_eq!(cpuid(&mut h, 0x8000_0008, 0)[2], 5 | 3 << 12);
}

#[test]
fn leaf_0x8000000a_describes_svm() {
    let mut h = blank();
    let [revision, asids, ecx, features] = cpuid(&mut h, 0x8000_000a, 0);
    assert_eq!((revision, asids, ecx), (1, 32768, 0));
    // nested paging, LBR virtualization, SVM lock, next-RIP save, the TSC ratio MSR, VMCB clean
    // bits, flush by ASID, decode assists, the pause filter and its threshold, AVIC, virtual
    // VMLOAD/VMSAVE, vGIF, x2AVIC, SPEC_CTRL virtualization, vNMI and the SVM instruction address
    // check order (tested in svm.rs); GMET, ROGPT, host MCE override and the rest are not
    // implemented
    assert_eq!(
        features,
        bits(&[0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 15, 16, 18, 20, 25, 28])
    );
}

/// One subleaf of leaf 0x8000_001D as type, level, sharing threads, ways, sets, line size.
fn cache(harness: &mut Harness, subleaf: u32) -> (u32, u32, u32, u32, u32, u32) {
    let [eax, ebx, ecx, edx] = cpuid(harness, 0x8000_001d, subleaf);
    assert_eq!(edx, 0, "no WBINVD or inclusiveness flags");
    (
        eax & 0x1f,
        (eax >> 5) & 7,
        ((eax >> 14) & 0xfff) + 1,
        (ebx >> 22) + 1,
        ecx + 1,
        (ebx & 0xfff) + 1,
    )
}

#[test]
fn leaf_0x8000001d_enumerates_the_caches_with_their_sharing() {
    // 1 thread per core, one CPU: every cache is shared by that one thread
    let mut h = blank();
    assert_eq!(cache(&mut h, 0), (1, 1, 1, 12, 64, 64), "L1D 48 KiB");
    assert_eq!(cache(&mut h, 1), (2, 1, 1, 8, 64, 64), "L1I 32 KiB");
    assert_eq!(cache(&mut h, 2), (3, 2, 1, 16, 1024, 64), "L2 1 MiB");
    assert_eq!(cache(&mut h, 3), (3, 3, 1, 16, 32768, 64), "L3 32 MiB");
    assert_eq!(
        cpuid(&mut h, 0x8000_001d, 4),
        [0, 0, 0, 0],
        "type 0 ends the list"
    );
    assert_eq!(
        cpuid(&mut h, 0x8000_001d, 0)[0],
        0x121,
        "self-initializing, not fully associative"
    );
    // 32 threads, SMT2: L1/L2 are per core, the L3 is shared by the 16 threads of a die
    let mut h = with_topology(32, 2, 0);
    assert_eq!(cache(&mut h, 0).2, 2);
    assert_eq!(cache(&mut h, 2).2, 2);
    assert_eq!(cache(&mut h, 3).2, 16);
    let mut h = with_topology(4, 2, 0);
    assert_eq!(
        cache(&mut h, 3).2,
        4,
        "a small machine shares the L3 among all of it"
    );
}

#[test]
fn the_cache_leaves_agree_with_each_other() {
    let mut h = blank();
    let size_kib =
        |(_, _, _, ways, sets, line): (u32, u32, u32, u32, u32, u32)| ways * sets * line / 1024;
    let [_, _, l1d, l1i] = cpuid(&mut h, 0x8000_0005, 0);
    assert_eq!(l1d >> 24, size_kib(cache(&mut h, 0)));
    assert_eq!(l1i >> 24, size_kib(cache(&mut h, 1)));
    let [_, _, l2, l3] = cpuid(&mut h, 0x8000_0006, 0);
    assert_eq!(l2 >> 16, size_kib(cache(&mut h, 2)));
    assert_eq!(l3 >> 18, size_kib(cache(&mut h, 3)) / 512);
}

#[test]
fn leaf_0x8000001e_gives_the_extended_apic_id_and_core_numbering() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 0x8000_001e, 0), [0, 0, 0, 0]);
    // second thread of core 2 in a 4-core SMT2 machine
    let mut h = with_topology(8, 2, 5);
    assert_eq!(cpuid(&mut h, 0x8000_001e, 0), [5, 1 << 8 | 2, 0, 0]);
    let mut h = with_topology(8, 2, 4);
    assert_eq!(
        cpuid(&mut h, 0x8000_001e, 0),
        [4, 1 << 8 | 2, 0, 0],
        "threads share a core id"
    );
}

#[test]
fn every_topology_leaf_describes_the_same_machine() {
    for (logical, threads) in [
        (1, 1),
        (2, 1),
        (2, 2),
        (4, 2),
        (8, 2),
        (6, 1),
        (16, 2),
        (64, 2),
    ] {
        for x2apic_id in 0..logical {
            let mut h = with_topology(logical, threads, x2apic_id);
            let [_, ebx, ..] = cpuid(&mut h, 1, 0);
            assert_eq!((ebx >> 16) & 0xff, logical);
            let [_, smt_count, smt_type, smt_id] = cpuid(&mut h, 0xb, 0);
            let [package_shift, package_count, package_type, package_id] = cpuid(&mut h, 0xb, 1);
            assert_eq!(smt_count, threads);
            assert_eq!(package_count, logical);
            assert_eq!((smt_type >> 8, package_type >> 8), (1, 2));
            assert_eq!((smt_id, package_id), (x2apic_id, x2apic_id));
            assert!(
                1 << package_shift >= logical,
                "{logical}: {package_shift} bits"
            );
            let [apic_id, topology, ..] = cpuid(&mut h, 0x8000_001e, 0);
            assert_eq!(apic_id, x2apic_id);
            assert_eq!(((topology >> 8) & 0xff) + 1, threads);
            assert_eq!(topology & 0xff, x2apic_id / threads, "core id");
            assert_eq!(cpuid(&mut h, 0x8000_0008, 0)[2] & 0xff, logical - 1);
        }
    }
}

#[test]
fn leaf_0x80000021_has_turins_bits_and_no_srso_no() {
    let mut h = blank();
    // NO_NESTED_DATA_BP WRMSR_XX_BASE_NS LFENCE_ALWAYS_SERIALIZING NULL_SEL_CLR_BASE AUTOIBRS
    // NO_SMM_CTL_MSR FSRS FSRC PREFETCH_CTL_MSR GP_ON_USER_CPUID PREFETCHI SBPB IBPB_BRTYPE
    // SRSO_USER_KERNEL_NO SRSO_BP_SPEC_REDUCE; no SRSO_NO (bit 29), UPPER_ADDRESS_IGNORE, ERAPS,
    // VERW_CLEAR; ECX: TSA_SQ_NO, TSA_L1_NO
    assert_eq!(
        cpuid(&mut h, 0x8000_0021, 0),
        [
            bits(&[0, 1, 2, 6, 8, 9, 10, 11, 13, 17, 20, 27, 28, 30, 31]),
            0,
            bits(&[1, 2]),
            0
        ]
    );
}

#[test]
fn leaves_past_the_last_extended_leaf_and_the_unused_ones_are_zero() {
    let mut h = blank();
    for leaf in [
        0x8000_0009,
        0x8000_000b,
        0x8000_001b,
        0x8000_001c,
        0x8000_001f,
        0x8000_0023,
        0x8000_0024,
        0x8000_0025,
        0x8000_0027,
        0xffff_ffff,
    ] {
        assert_eq!(cpuid(&mut h, leaf, 0), [0, 0, 0, 0], "leaf {leaf:#x}");
    }
}

#[test]
fn the_leaf_is_taken_from_eax_only() {
    let mut h = blank();
    assert_eq!(cpuid(&mut h, 0, 0)[0], 0x10);
}

#[test]
fn advertised_instruction_sets_have_the_state_they_need() {
    // AVX-512 needs the three XSAVE components, XSAVEC/XSAVES the compacted-size field
    let mut h = blank();
    let [xcr0_supported, ..] = cpuid(&mut h, 0xd, 0);
    assert_eq!(xcr0_supported & 0xe0, 0xe0);
    assert_ne!(cpuid(&mut h, 7, 0)[1] & (1 << 16), 0, "AVX512F");
    assert_ne!(cpuid(&mut h, 0xd, 1)[0] & 0b1010, 0, "XSAVEC and XSAVES");
}

#[test]
fn leaf_0x8000_0019_describes_fully_associative_l1_tlbs_for_1gib_pages_and_no_l2() {
    let mut h = blank();
    // Data TLB: associativity 0xf (fully), 96 entries; instruction TLB: 0xf, 64 entries.
    assert_eq!(
        cpuid(&mut h, 0x8000_0019, 0),
        [0xf060_f040, 0, 0, 0],
        "EAX is the L1 pair, EBX the (absent) L2 pair"
    );
}

#[test]
fn leaf_0x80000026_describes_one_processor_as_amd_levels() {
    let mut h = blank();
    // core level: no thread bits, 1 thread, type 1; complex: 1 bit (never narrower than the
    // package field of leaf 0xB), 1 processor, type 2; die and socket: the package, types 3 and
    // 4; then the invalid level echoing the subleaf.
    assert_eq!(cpuid(&mut h, 0x8000_0026, 0), [0, 1, 0x100, 0]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 1), [1, 1, 0x201, 0]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 2), [1, 1, 0x302, 0]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 3), [1, 1, 0x403, 0]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 4), [0, 0, 4, 0]);
}

#[test]
fn leaf_0x80000026_splits_a_large_package_into_16_thread_complexes() {
    // 32 threads, SMT2: thread bit, then 3 core bits per complex, then one complex bit
    let mut h = with_topology(32, 2, 0x13);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 0), [1, 2, 0x100, 0x13]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 1), [4, 16, 0x201, 0x13]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 2), [5, 32, 0x302, 0x13]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 3), [5, 32, 0x403, 0x13]);
    // 6 single-threaded cores fit in one complex
    let mut h = with_topology(6, 1, 5);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 0), [0, 1, 0x100, 5]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 1), [3, 6, 0x201, 5]);
    assert_eq!(cpuid(&mut h, 0x8000_0026, 2), [3, 6, 0x302, 5]);
}

/// What Linux derives from leaf 0x8000_0026 must match leaves 0xB and 0x8000_001D: the SMT width
/// and count of 0xB, an L3 (LLC) id equal to the complex id, and a package that is the whole
/// machine. Covers the processor counts on both sides of a complex and SMT on and off.
#[test]
fn leaf_0x80000026_agrees_with_the_other_topology_leaves() {
    for (logical, threads) in [
        (1, 1),
        (2, 1),
        (2, 2),
        (16, 1),
        (16, 2),
        (17, 1),
        (34, 2),
        (64, 1),
        (64, 2),
    ] {
        for x2apic_id in 0..logical {
            let mut h = with_topology(logical, threads, x2apic_id);
            let levels: Vec<[u32; 4]> = (0..4).map(|n| cpuid(&mut h, 0x8000_0026, n)).collect();
            assert_eq!(cpuid(&mut h, 0x8000_0026, 4)[1], 0, "the list ends");
            let mut previous_shift = 0;
            for (n, [shift, count, ecx, edx]) in levels.iter().copied().enumerate() {
                assert_eq!(ecx, (n as u32 + 1) << 8 | n as u32, "type and number");
                assert_eq!(edx, x2apic_id, "the same x2APIC id on every subleaf");
                assert!(count > 0 && shift >= previous_shift, "{logical}/{threads}");
                assert!(
                    1 << shift >= count,
                    "{logical}/{threads}: {shift} bits for {count}"
                );
                previous_shift = shift;
            }
            let [smt_shift, smt_count, ..] = cpuid(&mut h, 0xb, 0);
            let [package_shift, package_count, ..] = cpuid(&mut h, 0xb, 1);
            assert_eq!((levels[0][0], levels[0][1]), (smt_shift, smt_count));
            assert_eq!((levels[3][0], levels[3][1]), (package_shift, package_count));
            let l3_sharing = ((cpuid(&mut h, 0x8000_001d, 3)[0] >> 14) & 0xfff) + 1;
            assert_eq!(levels[1][1], l3_sharing, "{logical}/{threads}");
            // Linux: llc_id = apicid >> get_count_order(sharing)
            let llc_id = x2apic_id >> l3_sharing.next_power_of_two().trailing_zeros();
            assert_eq!(x2apic_id >> levels[1][0], llc_id, "{logical}/{threads}");
        }
    }
}

/// The machine's shapes as `profile.rs` builds them, up to the largest package: every topology
/// leaf, read on every processor, describes the same cores, threads and complexes, and the full
/// x2APIC id (past 255) comes back wherever a leaf has room for it.
#[test]
fn every_processor_of_every_machine_shape_reads_one_consistent_topology() {
    use crate::cpu::profile::{CpuProfile, MAX_LOGICAL_PROCESSORS};
    let mut h = blank();
    for logical in 1..=MAX_LOGICAL_PROCESSORS {
        for smt in [false, true] {
            let profile = CpuProfile::new(logical, smt);
            let threads = profile.threads_per_core;
            let paired = smt && logical % 2 == 0;
            assert_eq!(threads, if paired { 2 } else { 1 }, "{logical} smt {smt}");
            let mut cores_seen = alloc::collections::BTreeMap::<u32, u32>::new();
            for x2apic_id in 0..logical {
                h.cpu.topology = profile.topology(x2apic_id);
                let shape = format!("{logical} smt {smt} id {x2apic_id}");

                let [_, ebx, _, edx] = cpuid(&mut h, 1, 0);
                assert_eq!((ebx >> 16) & 0xff, logical.min(0xff), "{shape}");
                assert_eq!(ebx >> 24, x2apic_id & 0xff, "{shape}");
                assert_eq!(edx >> 28 & 1, (logical > 1) as u32, "HTT {shape}");

                let [smt_shift, smt_count, _, smt_id] = cpuid(&mut h, 0xb, 0);
                let [package_shift, package_count, _, package_id] = cpuid(&mut h, 0xb, 1);
                assert_eq!((smt_count, package_count), (threads, logical), "{shape}");
                assert_eq!((smt_id, package_id), (x2apic_id, x2apic_id), "{shape}");
                assert_eq!(1 << smt_shift, threads, "{shape}");
                assert!(1u32 << package_shift >= logical, "{shape}");
                assert_eq!(x2apic_id >> package_shift, 0, "one package {shape}");

                let [_, _, ecx, _] = cpuid(&mut h, 0x8000_0008, 0);
                assert_eq!(ecx & 0xff, (logical - 1).min(0xff), "{shape}");
                assert!(1u32 << (ecx >> 12 & 0xf) >= logical, "ApicIdSize {shape}");

                let [apic_id, core, node, _] = cpuid(&mut h, 0x8000_001e, 0);
                let core_id = x2apic_id >> smt_shift;
                assert_eq!(apic_id, x2apic_id, "{shape}");
                assert_eq!((core >> 8 & 0xff) + 1, threads, "{shape}");
                assert_eq!(core & 0xff, core_id & 0xff, "{shape}");
                assert_eq!(node, 0, "one node {shape}");
                *cores_seen.entry(core_id).or_default() += 1;

                let levels: Vec<[u32; 4]> = (0..4).map(|n| cpuid(&mut h, 0x8000_0026, n)).collect();
                for level in &levels {
                    assert_eq!(level[3], x2apic_id, "{shape}");
                }
                assert_eq!(
                    (levels[0][0], levels[0][1]),
                    (smt_shift, threads),
                    "{shape}"
                );
                assert_eq!(levels[3][1], logical, "{shape}");
                let l1_sharing = (cpuid(&mut h, 0x8000_001d, 0)[0] >> 14 & 0xfff) + 1;
                let l3_sharing = (cpuid(&mut h, 0x8000_001d, 3)[0] >> 14 & 0xfff) + 1;
                assert_eq!(l1_sharing, threads, "{shape}");
                assert_eq!(levels[1][1], l3_sharing, "{shape}");
                assert_eq!(
                    l3_sharing,
                    (8 * threads).min(logical),
                    "8 cores per L3 {shape}"
                );
                assert_eq!(
                    x2apic_id >> levels[1][0],
                    x2apic_id >> l3_sharing.next_power_of_two().trailing_zeros(),
                    "complex and LLC ids agree {shape}"
                );
            }
            assert_eq!(
                cores_seen.len() as u32,
                logical / threads,
                "{logical} smt {smt}"
            );
            assert!(cores_seen.values().all(|&count| count == threads));
        }
    }
}

#[test]
fn the_brand_leaves_spell_out_the_profiles_part() {
    let mut h = blank();
    h.cpu.topology = crate::cpu::profile::CpuProfile::new(256, true).topology(255);
    let brand: Vec<u8> = (0x8000_0002..=0x8000_0004)
        .flat_map(|leaf| cpuid(&mut h, leaf, 0))
        .flat_map(u32::to_le_bytes)
        .collect();
    let text = b"AMD EPYC 9755 128-Core Processor";
    assert_eq!(&brand[..text.len()], text);
    assert!(brand[text.len()..].iter().all(|&byte| byte == 0));
}
