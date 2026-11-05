//! AVX-512 instructions the JIT lowers inline (`helpers::evex_form`): unmasked whole-vector
//! moves, integer logic and arithmetic, byte compares into an opmask register, and `kmov` to a
//! general register. Each case runs over several register fillings, including ones where the
//! compares find equal and zero bytes, and the whole vector register file and the opmasks are
//! compared. The assembly is the string in each table entry (`hack/asm.sh`).

use super::diff_tests::*;
use crate::cpu::xsave::XCR0_SUPPORTED;
use crate::testkit::Harness;

/// The 64-byte pattern repeated over the memory the cases read and write, so every
/// vector-aligned offset of a case sees the same bytes and a register filled from it compares equal.
/// CR4.OSXSAVE; `cpu::CR4_OSXSAVE` is only exported to the generators' own hosts.
const CR4_OSXSAVE: u64 = 1 << 18;
const PATTERN_BYTES: usize = 64;
const PATTERN_REPEATS: usize = 6;
const VECTOR_REGISTERS: usize = 32;
const OPMASK_REGISTERS: usize = 8;
const FILLINGS: u64 = 6;

/// Enables the AVX-512 state and fills the vector registers and opmasks. `filling` picks what the
/// registers hold relative to the memory pattern: 0 and 1 random, 2 the pattern with a few bytes
/// changed, 3 all zero, 4 all ones, 5 the pattern exactly.
fn setup(harness: &mut Harness, filling: u64) {
    seed_registers_and_memory(harness);
    harness.cpu.cr4 |= CR4_OSXSAVE;
    harness.cpu.xcr0 = XCR0_SUPPORTED;
    let mut rng = Rng(0x1234_5678_9abc_def0 ^ filling.wrapping_mul(0x9e37_79b9));
    let mut pattern = [0u8; PATTERN_BYTES];
    for byte in &mut pattern {
        *byte = rng.next() as u8;
    }
    let address = harness.cpu.gpr[3];
    for repeat in 0..PATTERN_REPEATS {
        harness.write(address + (repeat * PATTERN_BYTES) as u64, &pattern);
    }
    for register in 0..VECTOR_REGISTERS {
        let mut bytes = [0u8; PATTERN_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = match filling {
                2 => {
                    if rng.below(8) == 0 {
                        rng.next() as u8
                    } else {
                        pattern[index]
                    }
                }
                3 => 0,
                4 => 0xff,
                5 => pattern[index],
                _ => rng.next() as u8,
            };
        }
        let lane = |index: usize| {
            u128::from_le_bytes(bytes[16 * index..16 * index + 16].try_into().unwrap())
        };
        if register < 16 {
            harness.cpu.xmm[register] = lane(0);
            harness.cpu.ymm_hi[register] = lane(1);
            harness.cpu.zmm_hi[register] = [lane(2), lane(3)];
        } else {
            harness.cpu.zmm_extra[register - 16] = [lane(0), lane(1), lane(2), lane(3)];
        }
    }
    for mask in &mut harness.cpu.opmask[..OPMASK_REGISTERS] {
        *mask = match filling {
            3 => 0,
            4 => u64::MAX,
            _ => rng.next(),
        };
    }
}

/// A load and a store that fill the TLB for the page `rbx` is in, ahead of the instruction: the
/// first memory access of a block always misses and runs in the interpreter, and the instruction
/// under test must take its inline fast path, not just agree with itself a second time.
/// (mov rax, [rbx]; mov [rbx + 0x300], rax)
const WARM_TLB: &[u8] = &[0x48, 0x8b, 0x03, 0x48, 0x89, 0x83, 0x00, 0x03, 0x00, 0x00];

fn warmed(code: &[u8]) -> Vec<u8> {
    [WARM_TLB, code].concat()
}

fn run_table(table: &[(&str, &[u8])]) {
    for (name, code) in table {
        let code = warmed(code);
        for filling in 0..FILLINGS {
            let result =
                std::panic::catch_unwind(|| check_with(&code, |harness| setup(harness, filling)));
            if let Err(panic) = result {
                eprintln!("case `{name}`, filling {filling}");
                std::panic::resume_unwind(panic);
            }
        }
    }
}

#[test]
fn whole_vector_moves_match_the_interpreter() {
    run_table(MOVES);
}

#[test]
fn integer_logic_and_arithmetic_match_the_interpreter() {
    run_table(ALU);
}

#[test]
fn byte_compares_into_opmasks_match_the_interpreter() {
    run_table(COMPARES);
}

#[test]
fn kmov_between_opmask_and_general_registers_matches_the_interpreter() {
    run_table(KMOVS);
}

const MOVES: &[(&str, &[u8])] = &[
    (
        "vmovdqu64 zmm1, [rbx]",
        &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x0b, 0xf4],
    ),
    (
        "vmovdqu64 [rbx+64], zmm2",
        &[0x62, 0xf1, 0xfe, 0x48, 0x7f, 0x53, 0x01, 0xf4],
    ),
    (
        "vmovdqu64 zmm3, [rbx+192]",
        &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x5b, 0x03, 0xf4],
    ),
    (
        "vmovdqu32 ymm17, [rbx+3]",
        &[
            0x62, 0xe1, 0x7e, 0x28, 0x6f, 0x8b, 0x03, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqu8 [rbx+32], ymm18",
        &[0x62, 0xe1, 0x7f, 0x28, 0x7f, 0x53, 0x01, 0xf4],
    ),
    (
        "vmovdqu16 xmm3, [rbx+1]",
        &[
            0x62, 0xf1, 0xff, 0x08, 0x6f, 0x9b, 0x01, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqu8 zmm30, [rbx+7]",
        &[
            0x62, 0x61, 0x7f, 0x48, 0x6f, 0xb3, 0x07, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqa64 zmm4, [rbx]",
        &[0x62, 0xf1, 0xfd, 0x48, 0x6f, 0x23, 0xf4],
    ),
    (
        "vmovdqa32 [rbx+64], ymm19",
        &[0x62, 0xe1, 0x7d, 0x28, 0x7f, 0x5b, 0x02, 0xf4],
    ),
    (
        "vmovdqa64 ymm20, [rbx+32]",
        &[0x62, 0xe1, 0xfd, 0x28, 0x6f, 0x63, 0x01, 0xf4],
    ),
    (
        "vmovdqu64 zmm5, zmm6",
        &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0xee, 0xf4],
    ),
    (
        "vmovdqu8 ymm20, ymm21",
        &[0x62, 0xa1, 0x7f, 0x28, 0x6f, 0xe5, 0xf4],
    ),
    (
        "vmovdqa32 xmm22, xmm9",
        &[0x62, 0xc1, 0x7d, 0x08, 0x6f, 0xf1, 0xf4],
    ),
    (
        "vmovdqu64 zmm31, zmm0",
        &[0x62, 0x61, 0xfe, 0x48, 0x6f, 0xf8, 0xf4],
    ),
    (
        "vmovups zmm7, [rbx]",
        &[0x62, 0xf1, 0x7c, 0x48, 0x10, 0x3b, 0xf4],
    ),
    (
        "vmovaps [rbx], zmm8",
        &[0x62, 0x71, 0x7c, 0x48, 0x29, 0x03, 0xf4],
    ),
    (
        "vmovapd ymm9, [rbx+32]",
        &[0xc5, 0x7d, 0x28, 0x4b, 0x20, 0xf4],
    ),
    (
        "vmovupd [rbx+5], xmm10",
        &[0xc5, 0x79, 0x11, 0x53, 0x05, 0xf4],
    ),
    ("vmovups ymm1, ymm2", &[0xc5, 0xfc, 0x10, 0xca, 0xf4]),
    (
        "vmovdqu64 ymm0, [rbx+rcx*4+0x40]",
        &[0x62, 0xf1, 0xfe, 0x28, 0x6f, 0x44, 0x8b, 0x02, 0xf4],
    ),
    (
        "vmovdqu64 zmm2, [rbx+rcx*8]",
        &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x14, 0xcb, 0xf4],
    ),
];

const ALU: &[(&str, &[u8])] = &[
    (
        "vpxord zmm1, zmm2, [rbx]",
        &[0x62, 0xf1, 0x6d, 0x48, 0xef, 0x0b, 0xf4],
    ),
    (
        "vpxorq ymm17, ymm18, ymm19",
        &[0x62, 0xa1, 0xed, 0x20, 0xef, 0xcb, 0xf4],
    ),
    (
        "vpxord xmm1, xmm2, xmm3",
        &[0x62, 0xf1, 0x6d, 0x08, 0xef, 0xcb, 0xf4],
    ),
    (
        "vpandd zmm4, zmm5, zmm6",
        &[0x62, 0xf1, 0x55, 0x48, 0xdb, 0xe6, 0xf4],
    ),
    (
        "vpandnq ymm7, ymm8, [rbx+32]",
        &[0x62, 0xf1, 0xbd, 0x28, 0xdf, 0x7b, 0x01, 0xf4],
    ),
    (
        "vporq zmm9, zmm10, [rbx+64]",
        &[0x62, 0x71, 0xad, 0x48, 0xeb, 0x4b, 0x01, 0xf4],
    ),
    (
        "vpaddb zmm1, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x48, 0xfc, 0xcb, 0xf4],
    ),
    (
        "vpaddw ymm17, ymm18, [rbx]",
        &[0x62, 0xe1, 0x6d, 0x20, 0xfd, 0x0b, 0xf4],
    ),
    ("vpaddd xmm1, xmm2, xmm3", &[0xc5, 0xe9, 0xfe, 0xcb, 0xf4]),
    (
        "vpaddq zmm1, zmm2, [rbx]",
        &[0x62, 0xf1, 0xed, 0x48, 0xd4, 0x0b, 0xf4],
    ),
    ("vpsubb ymm4, ymm5, ymm6", &[0xc5, 0xd5, 0xf8, 0xe6, 0xf4]),
    (
        "vpsubw zmm4, zmm5, zmm6",
        &[0x62, 0xf1, 0x55, 0x48, 0xf9, 0xe6, 0xf4],
    ),
    (
        "vpsubd zmm20, zmm21, zmm22",
        &[0x62, 0xa1, 0x55, 0x40, 0xfa, 0xe6, 0xf4],
    ),
    ("vpsubq ymm4, ymm5, [rbx]", &[0xc5, 0xd5, 0xfb, 0x23, 0xf4]),
    (
        "vpminub zmm1, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x48, 0xda, 0xcb, 0xf4],
    ),
    (
        "vpmaxub ymm17, ymm18, [rbx]",
        &[0x62, 0xe1, 0x6d, 0x20, 0xde, 0x0b, 0xf4],
    ),
    ("vpminub xmm1, xmm2, xmm3", &[0xc5, 0xe9, 0xda, 0xcb, 0xf4]),
    (
        "vpxord zmm1, zmm1, zmm1",
        &[0x62, 0xf1, 0x75, 0x48, 0xef, 0xc9, 0xf4],
    ),
    (
        "vpandnd zmm1, zmm1, zmm2",
        &[0x62, 0xf1, 0x75, 0x48, 0xdf, 0xca, 0xf4],
    ),
    (
        "vpaddb zmm2, zmm1, zmm2",
        &[0x62, 0xf1, 0x75, 0x48, 0xfc, 0xd2, 0xf4],
    ),
];

const COMPARES: &[(&str, &[u8])] = &[
    (
        "vpcmpeqb k1, ymm16, [rbx]",
        &[0x62, 0xf1, 0x7d, 0x20, 0x74, 0x0b, 0xf4],
    ),
    (
        "vpcmpeqb k2, zmm1, zmm2",
        &[0x62, 0xf1, 0x75, 0x48, 0x74, 0xd2, 0xf4],
    ),
    (
        "vpcmpeqb k3 {k1}, ymm1, ymm2",
        &[0x62, 0xf1, 0x75, 0x29, 0x74, 0xda, 0xf4],
    ),
    (
        "vpcmpeqb k4, xmm1, xmm2",
        &[0x62, 0xf1, 0x75, 0x08, 0x74, 0xe2, 0xf4],
    ),
    (
        "vpcmpeqb k5 {k2}, zmm17, [rbx+64]",
        &[0x62, 0xf1, 0x75, 0x42, 0x74, 0x6b, 0x01, 0xf4],
    ),
    (
        "vpcmpeqb k0, ymm20, ymm21",
        &[0x62, 0xb1, 0x5d, 0x20, 0x74, 0xc5, 0xf4],
    ),
    (
        "vptestmb k1, ymm1, ymm2",
        &[0x62, 0xf2, 0x75, 0x28, 0x26, 0xca, 0xf4],
    ),
    (
        "vptestnmb k2, zmm3, [rbx]",
        &[0x62, 0xf2, 0x66, 0x48, 0x26, 0x13, 0xf4],
    ),
    (
        "vptestmb k1 {k2}, xmm4, xmm5",
        &[0x62, 0xf2, 0x5d, 0x0a, 0x26, 0xcd, 0xf4],
    ),
    (
        "vptestnmb k7 {k3}, ymm20, [rbx+32]",
        &[0x62, 0xf2, 0x5e, 0x23, 0x26, 0x7b, 0x01, 0xf4],
    ),
    (
        "vptestmb k6, zmm1, zmm1",
        &[0x62, 0xf2, 0x75, 0x48, 0x26, 0xf1, 0xf4],
    ),
    (
        "vptestnmb k1, ymm3, ymm3",
        &[0x62, 0xf2, 0x66, 0x28, 0x26, 0xcb, 0xf4],
    ),
];

const KMOVS: &[(&str, &[u8])] = &[
    ("kmovd eax, k1", &[0xc5, 0xfb, 0x93, 0xc1, 0xf4]),
    ("kmovq rcx, k2", &[0xc4, 0xe1, 0xfb, 0x93, 0xca, 0xf4]),
    ("kmovd r9d, k3", &[0xc5, 0x7b, 0x93, 0xcb, 0xf4]),
    ("kmovq r10, k7", &[0xc4, 0x61, 0xfb, 0x93, 0xd7, 0xf4]),
    ("kmovd edx, k0", &[0xc5, 0xfb, 0x93, 0xd0, 0xf4]),
    ("kmovw eax, k1", &[0xc5, 0xf8, 0x93, 0xc1, 0xf4]),
    ("kmovd k1, eax", &[0xc5, 0xfb, 0x92, 0xc8, 0xf4]),
    ("kmovq k2, rcx", &[0xc4, 0xe1, 0xfb, 0x92, 0xd1, 0xf4]),
    ("kmovd k3, r9d", &[0xc4, 0xc1, 0x7b, 0x92, 0xd9, 0xf4]),
    ("kmovq k7, r10", &[0xc4, 0xc1, 0xfb, 0x92, 0xfa, 0xf4]),
    ("kmovd k0, edx", &[0xc5, 0xfb, 0x92, 0xc2, 0xf4]),
    ("kmovw k1, eax", &[0xc5, 0xf8, 0x92, 0xc8, 0xf4]),
];

const FALLBACKS: &[(&str, &[u8])] = &[
    (
        "vpxord zmm1, zmm2, [rbx]{1to16}",
        &[0x62, 0xf1, 0x6d, 0x58, 0xef, 0x0b, 0xf4],
    ),
    (
        "vpcmpeqw k1, zmm1, zmm2",
        &[0x62, 0xf1, 0x75, 0x48, 0x75, 0xca, 0xf4],
    ),
    (
        "vpcmpeqd k1 {k2}, ymm1, [rbx]",
        &[0x62, 0xf1, 0x75, 0x2a, 0x76, 0x0b, 0xf4],
    ),
    (
        "vpcmpuw k1, zmm1, zmm2, 4",
        &[0x62, 0xf3, 0xf5, 0x48, 0x3e, 0xca, 0x04, 0xf4],
    ),
    (
        "vptestmw k1, zmm1, zmm2",
        &[0x62, 0xf2, 0xf5, 0x48, 0x26, 0xca, 0xf4],
    ),
    (
        "vptestnmd k1, zmm1, zmm2",
        &[0x62, 0xf2, 0x76, 0x48, 0x27, 0xca, 0xf4],
    ),
    (
        "vpternlogd zmm1, zmm2, zmm3, 0x96",
        &[0x62, 0xf3, 0x6d, 0x48, 0x25, 0xcb, 0x96, 0xf4],
    ),
    ("kmovw eax, k1", &[0xc5, 0xf8, 0x93, 0xc1, 0xf4]),
    ("kmovb eax, k1", &[0xc5, 0xf9, 0x93, 0xc1, 0xf4]),
    ("kmovd k1, eax", &[0xc5, 0xfb, 0x92, 0xc8, 0xf4]),
    (
        "vmovdqu8 [rbx]{k1}, zmm2",
        &[0x62, 0xf1, 0x7f, 0x49, 0x7f, 0x13, 0xf4],
    ),
    (
        "vmovdqu64 [rbx]{k2}, ymm3",
        &[0x62, 0xf1, 0xfe, 0x2a, 0x7f, 0x1b, 0xf4],
    ),
    (
        "vpmullw zmm1, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x48, 0xd5, 0xcb, 0xf4],
    ),
    (
        "vpbroadcastb zmm1, eax",
        &[0x62, 0xf2, 0x7d, 0x48, 0x7a, 0xc8, 0xf4],
    ),
];

const ALIGNED_MOVE: &[u8] = &[
    0x62, 0xf1, 0xfd, 0x48, 0x6f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0xf4,
];
const ALIGNED_STORE: &[u8] = &[
    0x62, 0xf1, 0xfd, 0x48, 0x7f, 0x8b, 0x20, 0x00, 0x00, 0x00, 0xf4,
];
const LOAD_64: &[u8] = &[
    0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x8b, 0x3d, 0x00, 0x00, 0x00, 0xf4,
];
const STORE_32: &[u8] = &[
    0x62, 0xf1, 0xfe, 0x28, 0x7f, 0x8b, 0x3d, 0x00, 0x00, 0x00, 0xf4,
];
const COMPARE_MEM: &[u8] = &[
    0x62, 0xf1, 0x75, 0x48, 0x74, 0x8b, 0x3d, 0x00, 0x00, 0x00, 0xf4,
];
const ALU_MEM: &[u8] = &[
    0x62, 0xf1, 0x6d, 0x48, 0xef, 0x8b, 0x3d, 0x00, 0x00, 0x00, 0xf4,
];

#[test]
fn masked_and_zeroing_moves_and_arithmetic_match_the_interpreter() {
    run_table(MASKED);
}

#[test]
fn byte_compares_by_every_predicate_match_the_interpreter() {
    run_table(PREDICATES);
}

#[test]
fn broadcast_masked_stores_and_other_forms_stay_in_the_interpreter_and_agree() {
    run_table(FALLBACKS);
}

/// Memory operands the inline path hands to the interpreter: a page crossing at every offset
/// around the end of a page, for loads, stores, compares and arithmetic.
#[test]
fn vector_memory_operands_at_the_end_of_a_page_match() {
    const PAGE: u64 = 4096;
    for code in [LOAD_64, STORE_32, COMPARE_MEM, ALU_MEM] {
        let code = &warmed(code);
        for back in [0u64, 1, 31, 32, 33, 63, 64, 65, 0x3d + 63] {
            check_with(code, |harness| {
                setup(harness, 2);
                // rbx + 0x3d is the operand: put it `back` bytes before the end of a page
                let end = (harness.cpu.gpr[3] & !(PAGE - 1)) + 2 * PAGE;
                harness.cpu.gpr[3] = end - back - 0x3d;
            });
        }
    }
}

/// The aligned moves fault on a misaligned operand: #GP(0), the same on both engines.
#[test]
fn misaligned_aligned_moves_fault_alike() {
    for code in [ALIGNED_MOVE, ALIGNED_STORE] {
        check_stop_with(code, |harness| setup(harness, 0), fault(13, Some(0)));
    }
}

/// An unmapped operand is a page fault whose address and error code both engines agree on.
#[test]
fn unmapped_vector_memory_operands_fault_alike() {
    const UNMAPPED: u64 = 0x1_0000_0000;
    for (code, error_code) in [(LOAD_64, 0), (STORE_32, 2), (COMPARE_MEM, 0), (ALU_MEM, 0)] {
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

/// Without the OS having enabled the state, or with CR0.TS set, the instruction raises #UD or
/// #NM from the first inline use in a block, whatever follows.
#[test]
fn the_avx512_state_gate_raises_ud_and_nm() {
    const UD: u8 = 6;
    const NM: u8 = 7;
    const CR0_TS_BIT: u64 = 1 << 3;
    let code = &COMPARES[1].1;
    let vector = &MOVES[0].1;
    for code in [code, vector, &KMOVS[0].1] {
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
                harness.cpu.xcr0 &= !crate::cpu::xsave::XCR0_AVX512;
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
    }
}

const MASKED: &[(&str, &[u8])] = &[
    (
        "vpxord zmm1{k1}, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x49, 0xef, 0xcb, 0xf4],
    ),
    (
        "vpxorq ymm17{k2}{z}, ymm18, [rbx]",
        &[0x62, 0xe1, 0xed, 0xa2, 0xef, 0x0b, 0xf4],
    ),
    (
        "vpaddb zmm1{k3}, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x4b, 0xfc, 0xcb, 0xf4],
    ),
    (
        "vpaddb ymm4{k1}{z}, ymm5, [rbx+32]",
        &[0x62, 0xf1, 0x55, 0xa9, 0xfc, 0x63, 0x01, 0xf4],
    ),
    (
        "vpaddw xmm1{k2}, xmm2, xmm3",
        &[0x62, 0xf1, 0x6d, 0x0a, 0xfd, 0xcb, 0xf4],
    ),
    (
        "vpaddw zmm1{k2}{z}, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0xca, 0xfd, 0xcb, 0xf4],
    ),
    (
        "vpaddd zmm1{k1}, zmm2, [rbx]",
        &[0x62, 0xf1, 0x6d, 0x49, 0xfe, 0x0b, 0xf4],
    ),
    (
        "vpaddq zmm1{k1}, zmm2, zmm3",
        &[0x62, 0xf1, 0xed, 0x49, 0xd4, 0xcb, 0xf4],
    ),
    (
        "vpsubq zmm1{k4}{z}, zmm2, zmm3",
        &[0x62, 0xf1, 0xed, 0xcc, 0xfb, 0xcb, 0xf4],
    ),
    (
        "vpsubb zmm20{k7}, zmm21, zmm22",
        &[0x62, 0xa1, 0x55, 0x47, 0xf8, 0xe6, 0xf4],
    ),
    (
        "vpminub zmm1{k1}, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x49, 0xda, 0xcb, 0xf4],
    ),
    (
        "vpmaxub ymm1{k1}{z}, ymm2, [rbx]",
        &[0x62, 0xf1, 0x6d, 0xa9, 0xde, 0x0b, 0xf4],
    ),
    (
        "vpandnd zmm1{k5}, zmm2, zmm3",
        &[0x62, 0xf1, 0x6d, 0x4d, 0xdf, 0xcb, 0xf4],
    ),
    (
        "vpandq zmm9{k6}, zmm10, zmm11",
        &[0x62, 0x51, 0xad, 0x4e, 0xdb, 0xcb, 0xf4],
    ),
    (
        "vporq xmm1{k1}, xmm2, xmm3",
        &[0x62, 0xf1, 0xed, 0x09, 0xeb, 0xcb, 0xf4],
    ),
    (
        "vpaddb zmm1{k1}, zmm1, zmm1",
        &[0x62, 0xf1, 0x75, 0x49, 0xfc, 0xc9, 0xf4],
    ),
    (
        "vmovdqu8 zmm1{k1}, [rbx]",
        &[0x62, 0xf1, 0x7f, 0x49, 0x6f, 0x0b, 0xf4],
    ),
    (
        "vmovdqu8 zmm1{k1}{z}, [rbx+3]",
        &[
            0x62, 0xf1, 0x7f, 0xc9, 0x6f, 0x8b, 0x03, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqu16 ymm2{k2}, [rbx]",
        &[0x62, 0xf1, 0xff, 0x2a, 0x6f, 0x13, 0xf4],
    ),
    (
        "vmovdqu32 zmm3{k3}{z}, [rbx]",
        &[0x62, 0xf1, 0x7e, 0xcb, 0x6f, 0x1b, 0xf4],
    ),
    (
        "vmovdqu64 ymm20{k4}, [rbx+7]",
        &[
            0x62, 0xe1, 0xfe, 0x2c, 0x6f, 0xa3, 0x07, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqu8 zmm1{k1}, zmm2",
        &[0x62, 0xf1, 0x7f, 0x49, 0x6f, 0xca, 0xf4],
    ),
    (
        "vmovdqu64 zmm1{k1}{z}, zmm2",
        &[0x62, 0xf1, 0xfe, 0xc9, 0x6f, 0xca, 0xf4],
    ),
    (
        "vmovups zmm1{k1}, [rbx]",
        &[0x62, 0xf1, 0x7c, 0x49, 0x10, 0x0b, 0xf4],
    ),
    (
        "vmovdqu8 xmm4{k1}{z}, [rbx+1]",
        &[
            0x62, 0xf1, 0x7f, 0x89, 0x6f, 0xa3, 0x01, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqa32 zmm5{k2}, [rbx]",
        &[0x62, 0xf1, 0x7d, 0x4a, 0x6f, 0x2b, 0xf4],
    ),
    (
        "vmovupd ymm6{k3}, [rbx+8]",
        &[
            0x62, 0xf1, 0xfd, 0x2b, 0x10, 0xb3, 0x08, 0x00, 0x00, 0x00, 0xf4,
        ],
    ),
    (
        "vmovdqu16 zmm7{k7}{z}, zmm8",
        &[0x62, 0xd1, 0xff, 0xcf, 0x6f, 0xf8, 0xf4],
    ),
];

const PREDICATES: &[(&str, &[u8])] = &[
    (
        "vpcmpub k1, zmm1, zmm2, 0",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x00, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 0",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x00, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 1",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x01, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 1",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x01, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 2",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x02, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 2",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x02, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 3",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x03, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 3",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x03, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 4",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x04, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 4",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x04, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 5",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x05, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 5",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x05, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 6",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x06, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 6",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x06, 0xf4],
    ),
    (
        "vpcmpub k1, zmm1, zmm2, 7",
        &[0x62, 0xf3, 0x75, 0x48, 0x3e, 0xca, 0x07, 0xf4],
    ),
    (
        "vpcmpb k2, ymm17, [rbx], 7",
        &[0x62, 0xf3, 0x75, 0x20, 0x3f, 0x13, 0x07, 0xf4],
    ),
    (
        "vpcmpub k3 {k1}, ymm1, ymm2, 4",
        &[0x62, 0xf3, 0x75, 0x29, 0x3e, 0xda, 0x04, 0xf4],
    ),
    (
        "vpcmpub k4 {k2}, zmm3, [rbx+64], 6",
        &[0x62, 0xf3, 0x65, 0x4a, 0x3e, 0x63, 0x01, 0x06, 0xf4],
    ),
    (
        "vpcmpb k5 {k3}, xmm4, xmm5, 1",
        &[0x62, 0xf3, 0x5d, 0x0b, 0x3f, 0xed, 0x01, 0xf4],
    ),
    (
        "vpcmpub k6, xmm1, xmm1, 2",
        &[0x62, 0xf3, 0x75, 0x08, 0x3e, 0xf1, 0x02, 0xf4],
    ),
    (
        "vpcmpub k0, ymm20, ymm21, 5",
        &[0x62, 0xb3, 0x5d, 0x20, 0x3e, 0xc5, 0x05, 0xf4],
    ),
];
