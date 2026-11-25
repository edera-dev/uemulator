//! AVX and AVX2 (VEX-encoded) instructions the JIT lowers inline (`helpers::evex_form`, which
//! hands VEX to `vex_form`): whole-vector moves, integer logic, arithmetic and compares, and
//! vpmovmskb; plus the EVEX broadcasts from a general register and kmov of every width, which
//! share the machinery. The register fillings, the TLB warm-up and the comparison are
//! `diff_evex`'s. The assembly is the string in each table entry (`hack/asm.sh`).

use super::diff_evex::{CR4_OSXSAVE, run_table, setup, warmed};
use super::diff_tests::*;

#[test]
fn vex_whole_vector_moves_match_the_interpreter() {
    run_table(MOVES);
}

#[test]
fn vex_integer_logic_arithmetic_and_compares_match_the_interpreter() {
    run_table(ALU);
}

#[test]
fn vpmovmskb_matches_the_interpreter() {
    run_table(MOVE_MASKS);
}

#[test]
fn broadcasts_from_general_registers_match_the_interpreter() {
    run_table(BROADCASTS);
}

#[test]
fn kmov_of_every_width_matches_the_interpreter() {
    run_table(KMOVS);
}

const MOVES: &[(&str, &[u8])] = &[
    ("vmovdqu ymm1, [rbx]", &[0xc5, 0xfe, 0x6f, 0x0b, 0xf4]),
    (
        "vmovdqu [rbx+32], ymm2",
        &[0xc5, 0xfe, 0x7f, 0x53, 0x20, 0xf4],
    ),
    (
        "vmovdqu xmm3, [rbx+1]",
        &[0xc5, 0xfa, 0x6f, 0x5b, 0x01, 0xf4],
    ),
    (
        "vmovdqu [rbx+7], xmm4",
        &[0xc5, 0xfa, 0x7f, 0x63, 0x07, 0xf4],
    ),
    ("vmovdqa ymm4, [rbx]", &[0xc5, 0xfd, 0x6f, 0x23, 0xf4]),
    (
        "vmovdqa [rbx+32], ymm5",
        &[0xc5, 0xfd, 0x7f, 0x6b, 0x20, 0xf4],
    ),
    (
        "vmovdqa xmm6, [rbx+16]",
        &[0xc5, 0xf9, 0x6f, 0x73, 0x10, 0xf4],
    ),
    (
        "vmovups ymm7, [rbx+3]",
        &[0xc5, 0xfc, 0x10, 0x7b, 0x03, 0xf4],
    ),
    (
        "vmovaps ymm8, [rbx+64]",
        &[0xc5, 0x7c, 0x28, 0x43, 0x40, 0xf4],
    ),
    (
        "vmovupd [rbx+5], xmm9",
        &[0xc5, 0x79, 0x11, 0x4b, 0x05, 0xf4],
    ),
    (
        "vmovaps [rbx+96], ymm10",
        &[0xc5, 0x7c, 0x29, 0x53, 0x60, 0xf4],
    ),
    (
        "vmovapd xmm10, xmm11",
        &[0xc4, 0x41, 0x79, 0x28, 0xd3, 0xf4],
    ),
    (
        "vmovdqu ymm12, ymm13",
        &[0xc4, 0x41, 0x7e, 0x6f, 0xe5, 0xf4],
    ),
    (
        "vmovdqa ymm14, ymm15",
        &[0xc4, 0x41, 0x7d, 0x6f, 0xf7, 0xf4],
    ),
    // the store forms between registers: the destination is r/m
    ("vmovdqu ymm2, ymm3 (7f)", &[0xc5, 0xfe, 0x7f, 0xda, 0xf4]),
    ("vmovups xmm1, xmm2 (11)", &[0xc5, 0xf8, 0x11, 0xd1, 0xf4]),
    (
        "vmovdqu ymm0, [rbx+rcx*4+0x40]",
        &[0xc5, 0xfe, 0x6f, 0x44, 0x8b, 0x40, 0xf4],
    ),
    (
        "vmovdqu ymm1, [rbx+0x7f]",
        &[0xc5, 0xfe, 0x6f, 0x4b, 0x7f, 0xf4],
    ),
    // vmovss/vmovsd (F3/F2 0f 10) are not whole-vector moves
    ("vmovss xmm1, [rbx]", &[0xc5, 0xfa, 0x10, 0x0b, 0xf4]),
];

const ALU: &[(&str, &[u8])] = &[
    ("vpxor ymm1, ymm2, ymm3", &[0xc5, 0xed, 0xef, 0xcb, 0xf4]),
    ("vpxor ymm0, ymm0, ymm0", &[0xc5, 0xfd, 0xef, 0xc0, 0xf4]),
    (
        "vpxor xmm9, xmm10, xmm11",
        &[0xc4, 0x41, 0x29, 0xef, 0xcb, 0xf4],
    ),
    ("vpand xmm1, xmm2, [rbx]", &[0xc5, 0xe9, 0xdb, 0x0b, 0xf4]),
    (
        "vpandn ymm4, ymm5, [rbx+33]",
        &[0xc5, 0xd5, 0xdf, 0x63, 0x21, 0xf4],
    ),
    ("vpandn ymm4, ymm4, ymm5", &[0xc5, 0xdd, 0xdf, 0xe5, 0xf4]),
    ("vpor ymm6, ymm7, ymm8", &[0xc5, 0xbd, 0xeb, 0xf7, 0xf4]),
    (
        "vpor ymm15, ymm14, [rbx+64]",
        &[0xc5, 0x0d, 0xeb, 0x7b, 0x40, 0xf4],
    ),
    (
        "vpcmpeqb ymm1, ymm2, [rbx]",
        &[0xc5, 0xed, 0x74, 0x0b, 0xf4],
    ),
    ("vpcmpeqb ymm1, ymm1, ymm2", &[0xc5, 0xf5, 0x74, 0xca, 0xf4]),
    ("vpcmpeqb ymm2, ymm1, ymm2", &[0xc5, 0xf5, 0x74, 0xd2, 0xf4]),
    ("vpcmpeqb xmm5, xmm5, xmm5", &[0xc5, 0xd1, 0x74, 0xed, 0xf4]),
    ("vpcmpeqw xmm3, xmm4, xmm5", &[0xc5, 0xd9, 0x75, 0xdd, 0xf4]),
    (
        "vpcmpeqw ymm3, ymm4, [rbx+2]",
        &[0xc5, 0xdd, 0x75, 0x5b, 0x02, 0xf4],
    ),
    (
        "vpcmpeqd ymm9, ymm10, [rbx+7]",
        &[0xc5, 0x2d, 0x76, 0x4b, 0x07, 0xf4],
    ),
    (
        "vpcmpeqd xmm9, xmm10, xmm11",
        &[0xc4, 0x41, 0x29, 0x76, 0xcb, 0xf4],
    ),
    ("vpminub ymm1, ymm2, ymm3", &[0xc5, 0xed, 0xda, 0xcb, 0xf4]),
    (
        "vpminub xmm1, xmm1, [rbx+9]",
        &[0xc5, 0xf1, 0xda, 0x4b, 0x09, 0xf4],
    ),
    ("vpmaxub ymm4, ymm5, [rbx]", &[0xc5, 0xd5, 0xde, 0x23, 0xf4]),
    ("vpaddb ymm1, ymm2, ymm3", &[0xc5, 0xed, 0xfc, 0xcb, 0xf4]),
    ("vpaddw xmm1, xmm2, [rbx]", &[0xc5, 0xe9, 0xfd, 0x0b, 0xf4]),
    ("vpaddd ymm1, ymm2, ymm3", &[0xc5, 0xed, 0xfe, 0xcb, 0xf4]),
    (
        "vpaddq ymm1, ymm2, [rbx+32]",
        &[0xc5, 0xed, 0xd4, 0x4b, 0x20, 0xf4],
    ),
    ("vpsubb ymm4, ymm5, ymm6", &[0xc5, 0xd5, 0xf8, 0xe6, 0xf4]),
    ("vpsubw xmm4, xmm5, xmm6", &[0xc5, 0xd1, 0xf9, 0xe6, 0xf4]),
    ("vpsubd ymm4, ymm5, [rbx]", &[0xc5, 0xd5, 0xfa, 0x23, 0xf4]),
    ("vpsubq ymm4, ymm5, ymm6", &[0xc5, 0xd5, 0xfb, 0xe6, 0xf4]),
];

const MOVE_MASKS: &[(&str, &[u8])] = &[
    ("vpmovmskb eax, ymm1", &[0xc5, 0xfd, 0xd7, 0xc1, 0xf4]),
    ("vpmovmskb r9d, xmm2", &[0xc5, 0x79, 0xd7, 0xca, 0xf4]),
    (
        "vpmovmskb ecx, ymm15",
        &[0xc4, 0xc1, 0x7d, 0xd7, 0xcf, 0xf4],
    ),
    ("vpmovmskb r15d, ymm0", &[0xc5, 0x7d, 0xd7, 0xf8, 0xf4]),
];

const BROADCASTS: &[(&str, &[u8])] = &[
    (
        "vpbroadcastb zmm1, eax",
        &[0x62, 0xf2, 0x7d, 0x48, 0x7a, 0xc8, 0xf4],
    ),
    // EVEX.X set: r/m 16 names eax all the same
    (
        "vpbroadcastb zmm1, eax (EVEX.X)",
        &[0x62, 0xb2, 0x7d, 0x48, 0x7a, 0xc8, 0xf4],
    ),
    (
        "vpbroadcastb ymm17 {k1}, ecx",
        &[0x62, 0xe2, 0x7d, 0x29, 0x7a, 0xc9, 0xf4],
    ),
    (
        "vpbroadcastb xmm2 {k2} {z}, r9d",
        &[0x62, 0xd2, 0x7d, 0x8a, 0x7a, 0xd1, 0xf4],
    ),
    // r/m 4 is esp's low byte, never ah
    (
        "vpbroadcastb zmm6, esp",
        &[0x62, 0xf2, 0x7d, 0x48, 0x7a, 0xf4, 0xf4],
    ),
    (
        "vpbroadcastb zmm30 {k7}, r15d",
        &[0x62, 0x42, 0x7d, 0x4f, 0x7a, 0xf7, 0xf4],
    ),
    (
        "vpbroadcastw zmm3 {k3}, esi",
        &[0x62, 0xf2, 0x7d, 0x4b, 0x7b, 0xde, 0xf4],
    ),
    (
        "vpbroadcastw ymm3, edi",
        &[0x62, 0xf2, 0x7d, 0x28, 0x7b, 0xdf, 0xf4],
    ),
    (
        "vpbroadcastd ymm4, edx",
        &[0x62, 0xf2, 0x7d, 0x28, 0x7c, 0xe2, 0xf4],
    ),
    (
        "vpbroadcastd zmm4 {k4} {z}, eax",
        &[0x62, 0xf2, 0x7d, 0xcc, 0x7c, 0xe0, 0xf4],
    ),
    (
        "vpbroadcastq zmm5 {k1} {z}, r10",
        &[0x62, 0xd2, 0xfd, 0xc9, 0x7c, 0xea, 0xf4],
    ),
    (
        "vpbroadcastq xmm21 {k5}, rbx",
        &[0x62, 0xe2, 0xfd, 0x0d, 0x7c, 0xeb, 0xf4],
    ),
    (
        "vpbroadcastq zmm5, rax",
        &[0x62, 0xf2, 0xfd, 0x48, 0x7c, 0xe8, 0xf4],
    ),
];

const KMOVS: &[(&str, &[u8])] = &[
    ("kmovb eax, k1", &[0xc5, 0xf9, 0x93, 0xc1, 0xf4]),
    ("kmovb k2, ecx", &[0xc5, 0xf9, 0x92, 0xd1, 0xf4]),
    ("kmovw r9d, k3", &[0xc5, 0x78, 0x93, 0xcb, 0xf4]),
    ("kmovw k4, r10d", &[0xc4, 0xc1, 0x78, 0x92, 0xe2, 0xf4]),
    ("kmovb k5, r15d", &[0xc4, 0xc1, 0x79, 0x92, 0xef, 0xf4]),
    ("kmovd k6, esp", &[0xc5, 0xfb, 0x92, 0xf4, 0xf4]),
];

/// Encodings the classifier must refuse, which the interpreter raises #UD for: vvvv other than
/// 1111 on a move and on vpmovmskb, vpbroadcastb with W1, kmov with F3.
const UNDEFINED: &[(&str, &[u8])] = &[
    (
        "vmovdqu ymm1, [rbx] (vvvv=1)",
        &[0xc5, 0xf6, 0x6f, 0x0b, 0xf4],
    ),
    (
        "vpmovmskb eax, ymm1 (vvvv=1)",
        &[0xc5, 0xf5, 0xd7, 0xc1, 0xf4],
    ),
    (
        "vpbroadcastb zmm1, eax (W1)",
        &[0x62, 0xf2, 0xfd, 0x48, 0x7a, 0xc8, 0xf4],
    ),
    ("kmov k1, eax (F3)", &[0xc5, 0xfa, 0x92, 0xc8, 0xf4]),
];

#[test]
fn undefined_encodings_raise_ud_alike() {
    const UD: u8 = 6;
    for (name, code) in UNDEFINED {
        eprintln!("case `{name}`");
        check_stop_with(&warmed(code), |harness| setup(harness, 1), fault(UD, None));
    }
}

/// vmovdqu ymm1, [rbx+0x3d]
const LOAD_32: &[u8] = &[0xc5, 0xfe, 0x6f, 0x4b, 0x3d, 0xf4];
/// vmovdqu [rbx+0x3d], ymm2
const STORE_32: &[u8] = &[0xc5, 0xfe, 0x7f, 0x53, 0x3d, 0xf4];
/// vpcmpeqb ymm1, ymm2, [rbx+0x3d]
const COMPARE_32: &[u8] = &[0xc5, 0xed, 0x74, 0x4b, 0x3d, 0xf4];
/// vmovdqu xmm1, [rbx+0x3d]
const LOAD_16: &[u8] = &[0xc5, 0xfa, 0x6f, 0x4b, 0x3d, 0xf4];

/// Memory operands the inline path hands to the interpreter: a page crossing at every offset
/// around the end of a page.
#[test]
fn vex_memory_operands_at_the_end_of_a_page_match() {
    const PAGE: u64 = 4096;
    for code in [LOAD_32, STORE_32, COMPARE_32, LOAD_16] {
        let code = &warmed(code);
        for back in [0u64, 1, 15, 16, 17, 31, 32, 33, 0x3d + 31] {
            check_with(code, |harness| {
                setup(harness, 2);
                // rbx + 0x3d is the operand: put it `back` bytes before the end of a page
                let end = (harness.cpu.gpr[3] & !(PAGE - 1)) + 2 * PAGE;
                harness.cpu.gpr[3] = end - back - 0x3d;
            });
        }
    }
}

#[test]
fn unmapped_vex_memory_operands_fault_alike() {
    const UNMAPPED: u64 = 0x1_0000_0000;
    for (code, error_code) in [(LOAD_32, 0), (STORE_32, 2), (COMPARE_32, 0), (LOAD_16, 0)] {
        check_stop_with(
            code,
            |harness| {
                setup(harness, 0);
                harness.cpu.gpr[3] = UNMAPPED - 0x3d;
            },
            fault(14, Some(error_code)),
        );
    }
}

/// vmovdqa needs the operand aligned to the vector length: 32 bytes for ymm, so a 16-byte
/// aligned operand is #GP(0) too.
#[test]
fn misaligned_vmovdqa_faults_alike() {
    const GP: u8 = 13;
    // vmovdqa ymm4, [rbx+16] | vmovdqa [rbx+16], ymm4 | vmovdqa xmm6, [rbx+8]
    for code in [
        &[0xc5, 0xfd, 0x6f, 0x63, 0x10, 0xf4][..],
        &[0xc5, 0xfd, 0x7f, 0x63, 0x10, 0xf4],
        &[0xc5, 0xf9, 0x6f, 0x73, 0x08, 0xf4],
    ] {
        check_stop_with(
            &warmed(code),
            |harness| setup(harness, 0),
            fault(GP, Some(0)),
        );
    }
}

const UD: u8 = 6;
const NM: u8 = 7;
const CR0_EM_BIT: u64 = 1 << 2;
const CR0_TS_BIT: u64 = 1 << 3;

/// The VEX gate is `Cpu::vex_gate`: #UD without CR4.OSXSAVE or XCR0's AVX state, #NM under CR0.TS,
/// and neither the AVX-512 state nor CR0.EM matters.
#[test]
fn the_avx_state_gate_raises_ud_and_nm() {
    for code in [MOVES[0].1, ALU[8].1, MOVE_MASKS[0].1] {
        check_stop_with(
            code,
            |harness| {
                setup(harness, 0);
                harness.cpu.cr4 &= !CR4_OSXSAVE;
            },
            fault(UD, None),
        );
        check_stop_with(
            code,
            |harness| {
                setup(harness, 0);
                harness.cpu.xcr0 &= !crate::cpu::xsave::XCR0_AVX;
            },
            fault(UD, None),
        );
        check_stop_with(
            code,
            |harness| {
                setup(harness, 0);
                harness.cpu.cr0 |= CR0_TS_BIT;
            },
            fault(NM, None),
        );
        check_with(code, |harness| {
            setup(harness, 0);
            harness.cpu.xcr0 &= !crate::cpu::xsave::XCR0_AVX512;
            harness.cpu.cr0 |= CR0_EM_BIT;
        });
    }
}

/// pxor xmm1, xmm2 (legacy SSE)
const LEGACY: &[u8] = &[0x66, 0x0f, 0xef, 0xca];
/// vpxor ymm1, ymm2, ymm3 (VEX)
const VEX: &[u8] = &[0xc5, 0xed, 0xef, 0xcb];
/// vpxord zmm1, zmm2, zmm3 (EVEX)
const EVEX: &[u8] = &[0x62, 0xf1, 0x6d, 0x48, 0xef, 0xcb];
/// kmovd eax, k1 (VEX, behind the AVX-512 gate)
const KMOV: &[u8] = &[0xc5, 0xfb, 0x93, 0xc1];

/// Each gate is checked once per block, at the first instruction that needs it: one that passed
/// must not stand in for another that checks other bits. Every order of a legacy SSE, a VEX, an
/// EVEX and a VEX opmask instruction in one block, under each state that fails some of them.
#[test]
fn the_sse_avx_and_avx512_gates_of_one_block_stay_apart() {
    let instructions = [LEGACY, VEX, EVEX, KMOV];
    let states: [&dyn Fn(&mut crate::testkit::Harness); 3] = [
        &|harness| harness.cpu.cr0 |= CR0_EM_BIT,
        &|harness| harness.cpu.xcr0 &= !crate::cpu::xsave::XCR0_AVX512,
        &|harness| harness.cpu.cr4 &= !CR4_OSXSAVE,
    ];
    for first in 0..instructions.len() {
        for second in 0..instructions.len() {
            if first == second {
                continue;
            }
            let code = [instructions[first], instructions[second], &[0xf4]].concat();
            for state in states {
                let result = run_checked(&code, |harness| {
                    setup(harness, 1);
                    state(harness);
                });
                assert!(
                    matches!(
                        result.interpreter_stop,
                        crate::testkit::Stop::Halted | crate::testkit::Stop::Fault { .. }
                    ),
                    "{:?}",
                    result.interpreter_stop
                );
            }
        }
    }
}

/// The diff corpus cannot tell a fast path that is never taken from one that is (the fallback
/// agrees with the interpreter by construction), so count what reaches the interpreter: after
/// `WARM_TLB` none of these may, on any backend, masked forms included.
#[test]
fn the_lowered_forms_run_without_the_interpreter() {
    use crate::testkit::{Harness, Stop};
    const REPEATS: usize = 3;
    let instructions: &[&[u8]] = &[
        // vmovdqu ymm1, [rbx] | vmovdqu [rbx+32], ymm2 | vpcmpeqb ymm1, ymm2, [rbx]
        &[0xc5, 0xfe, 0x6f, 0x0b],
        &[0xc5, 0xfe, 0x7f, 0x53, 0x20],
        &[0xc5, 0xed, 0x74, 0x0b],
        // vpxor ymm1, ymm2, ymm3 | vpmovmskb eax, ymm1 | vmovdqa xmm6, [rbx+16]
        &[0xc5, 0xed, 0xef, 0xcb],
        &[0xc5, 0xfd, 0xd7, 0xc1],
        &[0xc5, 0xf9, 0x6f, 0x73, 0x10],
        // vpbroadcastb ymm17 {k1}, ecx | vpbroadcastq zmm5 {k1} {z}, r10
        &[0x62, 0xe2, 0x7d, 0x29, 0x7a, 0xc9],
        &[0x62, 0xd2, 0xfd, 0xc9, 0x7c, 0xea],
        // kmovd k6, esp | kmovb eax, k1
        &[0xc5, 0xfb, 0x92, 0xf4],
        &[0xc5, 0xf9, 0x93, 0xc1],
        // vmovdqu8 zmm1{k1}, [rbx] | vmovdqu16 ymm2{k2}, [rbx] | vpaddb ymm4{k1}{z}, ymm5, [rbx+32]
        &[0x62, 0xf1, 0x7f, 0x49, 0x6f, 0x0b],
        &[0x62, 0xf1, 0xff, 0x2a, 0x6f, 0x13],
        &[0x62, 0xf1, 0x55, 0xa9, 0xfc, 0x63, 0x01],
    ];
    for instruction in instructions {
        let code = warmed(&[instruction.repeat(REPEATS), vec![0xf4]].concat());
        let mut harness = Harness::new(&code);
        if !harness.enable_jit() {
            return;
        }
        setup(&mut harness, 1);
        harness.cpu.helper_histogram = Some(Default::default());
        assert_eq!(harness.run(1000), Stop::Halted);
        let key = crate::cpu::decode::decode(instruction)
            .unwrap()
            .histogram_key();
        let interpreted = harness.cpu.helper_histogram.as_ref().unwrap().get(&key);
        assert_eq!(interpreted, None, "{instruction:x?}");
    }
}
