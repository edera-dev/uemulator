//! The instructions the JIT computes as whole registers (`helpers::WholeVectorOp`): the SHA
//! extensions, pshufd, palignr and pshufb, in register and memory forms, every immediate, the faults the
//! interpreter fallback raises, and the SHA-1 and SHA-256 block loops firmware hashes with.

use super::diff_tests::*;
use crate::testkit::*;

const CR0_TS: u64 = 1 << 3;
const CR4_OSFXSR: u64 = 1 << 9;
const CR4_OSXMMEXCPT: u64 = 1 << 10;
const HLT: u8 = 0xf4;
const RSI: usize = 6;

/// The default seeds plus SSE enabled and sixteen distinct, irregular xmm registers.
fn sse_setup(harness: &mut Harness) {
    seed_registers_and_memory(harness);
    harness.cpu.cr4 |= CR4_OSFXSR | CR4_OSXMMEXCPT;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for xmm in harness.cpu.xmm.iter_mut() {
        let mut half = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            state ^ state >> 29
        };
        *xmm = (half() as u128) << 64 | half() as u128;
    }
}

/// Opcode map escape bytes after `0f`.
const MAP_0F38: u8 = 0x38;
const MAP_0F3A: u8 = 0x3a;

/// A register-form instruction `[66] [REX] 0f [map] opcode modrm [imm]` with xmm `reg` as the
/// destination and xmm `rm` as the source.
fn register_form(
    p66: bool,
    map: Option<u8>,
    opcode: u8,
    reg: u8,
    rm: u8,
    imm: Option<u8>,
) -> Vec<u8> {
    const REX: u8 = 0x40;
    const REX_R: u8 = 0x04;
    const REX_B: u8 = 0x01;
    const MODRM_REGISTER: u8 = 0xc0;
    let mut bytes = Vec::new();
    if p66 {
        bytes.push(0x66);
    }
    let rex = (if reg >= 8 { REX_R } else { 0 }) | (if rm >= 8 { REX_B } else { 0 });
    if rex != 0 {
        bytes.push(REX | rex);
    }
    bytes.push(0x0f);
    bytes.extend(map);
    bytes.push(opcode);
    bytes.push(MODRM_REGISTER | (reg & 7) << 3 | (rm & 7));
    bytes.extend(imm);
    bytes
}

/// Every whole-vector instruction as `(66 prefix, map, opcode, takes an immediate)`.
const FORMS: [(bool, Option<u8>, u8, bool); 10] = [
    (false, Some(MAP_0F3A), 0xcc, true),  // sha1rnds4
    (false, Some(MAP_0F38), 0xc8, false), // sha1nexte
    (false, Some(MAP_0F38), 0xc9, false), // sha1msg1
    (false, Some(MAP_0F38), 0xca, false), // sha1msg2
    (false, Some(MAP_0F38), 0xcb, false), // sha256rnds2
    (false, Some(MAP_0F38), 0xcc, false), // sha256msg1
    (false, Some(MAP_0F38), 0xcd, false), // sha256msg2
    (true, None, 0x70, true),             // pshufd
    (true, Some(MAP_0F3A), 0x0f, true),   // palignr
    (true, Some(MAP_0F38), 0x00, false),  // pshufb
];

fn with_hlt(mut code: Vec<u8>) -> Vec<u8> {
    code.push(HLT);
    code
}

#[test]
fn every_form_between_registers_matches_the_interpreter() {
    // Each form on low, high and mixed registers, including the destination as its own source
    // and xmm0 (sha256rnds2's implicit operand) as the destination.
    for (p66, map, opcode, has_imm) in FORMS {
        let mut code = Vec::new();
        for (index, (reg, rm)) in [(1, 2), (0, 3), (9, 4), (5, 14), (11, 11), (15, 8), (0, 0)]
            .into_iter()
            .enumerate()
        {
            let imm = has_imm.then_some(0x1b + 0x35 * index as u8);
            code.extend(register_form(p66, map, opcode, reg, rm, imm));
        }
        check_with(&with_hlt(code), sse_setup);
    }
}

#[test]
fn every_immediate_of_pshufd_palignr_and_sha1rnds4() {
    // One snippet per immediate: chained, the palignr counts past 31 would zero every register
    // and hide what the earlier ones did.
    for (p66, map, opcode) in [
        (true, None, 0x70),
        (true, Some(MAP_0F3A), 0x0f),
        (false, Some(MAP_0F3A), 0xcc),
    ] {
        for imm in 0..=255u8 {
            let code = [
                register_form(p66, map, opcode, 1, 2, Some(imm)),
                register_form(p66, map, opcode, 13, 9, Some(imm)),
            ]
            .concat();
            check_with(&with_hlt(code), sse_setup);
        }
    }
}

/// sha1rnds4 xmm0, [rbx], 1; sha1nexte xmm1, [rbx+16]; sha1msg1 xmm2, [rbx+32]
/// sha1msg2 xmm3, [rbx+48]; sha256rnds2 xmm4, [rbx+64]; sha256msg1 xmm9, [rbx+80]
/// sha256msg2 xmm10, [rbx+96]; pshufd xmm11, [rbx+112], 0x39; palignr xmm12, [rbx+128], 4
/// palignr xmm13, [rsi], 20; pshufd xmm14, [rsi+16], 0x1b; hlt
const MEMORY_FORMS: [u8; 70] = [
    0x0f, 0x3a, 0xcc, 0x03, 0x01, 0x0f, 0x38, 0xc8, 0x4b, 0x10, 0x0f, 0x38, 0xc9, 0x53, 0x20, 0x0f,
    0x38, 0xca, 0x5b, 0x30, 0x0f, 0x38, 0xcb, 0x63, 0x40, 0x44, 0x0f, 0x38, 0xcc, 0x4b, 0x50, 0x44,
    0x0f, 0x38, 0xcd, 0x53, 0x60, 0x66, 0x44, 0x0f, 0x70, 0x5b, 0x70, 0x39, 0x66, 0x44, 0x0f, 0x3a,
    0x0f, 0xa3, 0x80, 0x00, 0x00, 0x00, 0x04, 0x66, 0x44, 0x0f, 0x3a, 0x0f, 0x2e, 0x14, 0x66, 0x44,
    0x0f, 0x70, 0x76, 0x10, 0x1b, HLT,
];

#[test]
fn memory_operands_match_the_interpreter() {
    check_with(&MEMORY_FORMS, sse_setup);
    // pshufb xmm7, [rbx+144]; pshufb xmm15, [rsi+32]; hlt
    check_with(
        &[
            0x66, 0x0f, 0x38, 0x00, 0xbb, 0x90, 0x00, 0x00, 0x00, 0x66, 0x44, 0x0f, 0x38, 0x00,
            0x7e, 0x20, HLT,
        ],
        sse_setup,
    );
}

#[test]
fn misaligned_operands_and_the_sse_gate_fault_like_the_interpreter() {
    // sha256msg1 xmm3, [rbx+8]; hlt
    let misaligned_sha = [0x0f, 0x38, 0xcc, 0x5b, 0x08, HLT];
    check_stop_with(&misaligned_sha, sse_setup, fault(13, Some(0)));
    // pshufd xmm0, [rbx+4], 0; hlt
    check_stop_with(
        &[0x66, 0x0f, 0x70, 0x43, 0x04, 0x00, HLT],
        sse_setup,
        fault(13, Some(0)),
    );
    // sha256rnds2 xmm2, xmm1; hlt
    let rounds = [0x0f, 0x38, 0xcb, 0xd1, HLT];
    check_stop_with(
        &rounds,
        |harness| {
            sse_setup(harness);
            harness.cpu.cr0 |= CR0_TS;
        },
        fault(7, None),
    );
    // palignr xmm1, xmm2, 3; hlt
    check_stop_with(
        &[0x66, 0x0f, 0x3a, 0x0f, 0xca, 0x03, HLT],
        |harness| {
            sse_setup(harness);
            harness.cpu.cr4 &= !CR4_OSFXSR;
        },
        fault(6, None),
    );
}

#[test]
fn forms_the_lowering_must_not_claim() {
    // 66 sha256rnds2 is #UD; unprefixed 0f 3a 0f is the MMX palignr the interpreter refuses.
    check_stop_with(
        &[0x66, 0x0f, 0x38, 0xcb, 0xd1, HLT],
        sse_setup,
        fault(6, None),
    );
    check_stop_with(
        &[0x0f, 0x3a, 0x0f, 0xca, 0x03, HLT],
        sse_setup,
        fault(6, None),
    );
    // f2/f3 0f 70 are pshuflw/pshufhw, which only touch one half.
    check_with(
        &[
            0xf2, 0x0f, 0x70, 0xca, 0x1b, 0xf3, 0x0f, 0x70, 0xd3, 0x1b, HLT,
        ],
        sse_setup,
    );
    // A page-crossing operand cannot be aligned, so the last 16 bytes of a page are the edge.
    check_with(&[0x0f, 0x38, 0xcc, 0x1e, HLT], |harness| {
        // sha256msg1 xmm3, [rsi]
        sse_setup(harness);
        harness.cpu.gpr[RSI] = SCRATCH + 0x1000 - 16;
    });
}

/// What the JIT lowered of the snippet `result` ran: (instructions translated, instructions that
/// went through `helper_exec`).
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
fn not_inlined(result: &Differential) -> u64 {
    let (jit, _) = result.jit.as_ref().expect("this host has a JIT");
    let stats = jit.cpu.jit.as_ref().unwrap().stats();
    stats.total - stats.inlined
}

#[test]
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
fn every_form_is_lowered_without_the_interpreter() {
    let register_forms: Vec<u8> = FORMS
        .into_iter()
        .flat_map(|(p66, map, opcode, has_imm)| {
            register_form(p66, map, opcode, 3, 12, has_imm.then_some(0x93))
        })
        .collect();
    assert_eq!(
        not_inlined(&check_with(&with_hlt(register_forms), sse_setup)),
        1,
        "only hlt"
    );
    assert_eq!(
        not_inlined(&check_with(&MEMORY_FORMS, sse_setup)),
        1,
        "only hlt"
    );
}

#[test]
fn sha256_block_loop_matches_the_interpreter() {
    // The SHA-256 compression loop of `uemulator-crypto` (Intel's SHA extensions white paper
    // shape): the schedule from sha256msg1 on memory, palignr and sha256msg2, then sixteen
    // groups of two sha256rnds2 around a pshufd.
    //     lea rcx, [rbx + 64]
    //     mov edx, 12
    // 2:  movdqa xmm3, [rcx - 64]
    //     sha256msg1 xmm3, [rcx - 48]
    //     movdqa xmm5, [rcx - 16]
    //     movdqa xmm10, xmm5
    //     palignr xmm10, [rcx - 32], 4
    //     paddd xmm3, xmm10
    //     sha256msg2 xmm3, xmm5
    //     movdqa [rcx], xmm3
    //     add rcx, 16
    //     dec edx
    //     jnz 2b
    //     mov rcx, rbx
    //     mov edx, 16
    // 3:  movdqa xmm0, [rcx]
    //     paddd xmm0, [rdi]
    //     sha256rnds2 xmm2, xmm1, xmm0
    //     pshufd xmm0, xmm0, 0x0e
    //     sha256rnds2 xmm1, xmm2, xmm0
    //     add rcx, 16
    //     add rdi, 16
    //     dec edx
    //     jnz 3b
    //     hlt
    let code = [
        0x48, 0x8d, 0x4b, 0x40, 0xba, 0x0c, 0x00, 0x00, 0x00, 0x66, 0x0f, 0x6f, 0x59, 0xc0, 0x0f,
        0x38, 0xcc, 0x59, 0xd0, 0x66, 0x0f, 0x6f, 0x69, 0xf0, 0x66, 0x44, 0x0f, 0x6f, 0xd5, 0x66,
        0x44, 0x0f, 0x3a, 0x0f, 0x51, 0xe0, 0x04, 0x66, 0x41, 0x0f, 0xfe, 0xda, 0x0f, 0x38, 0xcd,
        0xdd, 0x66, 0x0f, 0x7f, 0x19, 0x48, 0x83, 0xc1, 0x10, 0xff, 0xca, 0x75, 0xcf, 0x48, 0x89,
        0xd9, 0xba, 0x10, 0x00, 0x00, 0x00, 0x66, 0x0f, 0x6f, 0x01, 0x66, 0x0f, 0xfe, 0x07, 0x0f,
        0x38, 0xcb, 0xd1, 0x66, 0x0f, 0x70, 0xc0, 0x0e, 0x0f, 0x38, 0xcb, 0xca, 0x48, 0x83, 0xc1,
        0x10, 0x48, 0x83, 0xc7, 0x10, 0xff, 0xca, 0x75, 0xdf, HLT,
    ];
    let result = check_with(&code, sse_setup);
    assert_eq!(
        result.interpreter.cpu.gpr[2], 0,
        "both loops ran to the end"
    );
}

#[test]
fn sha1_steps_match_the_interpreter() {
    // SHA-1's four-round step as software writes it: sha1nexte folds E into the schedule, then
    // sha1rnds4 runs four rounds; sha1msg1/sha1msg2 extend the schedule in between.
    //     mov edx, 20
    // 2:  movdqa xmm1, xmm0
    //     sha1nexte xmm4, xmm6
    //     sha1rnds4 xmm0, xmm4, 2
    //     sha1msg1 xmm5, xmm6
    //     sha1msg2 xmm6, xmm7
    //     pxor xmm5, xmm7
    //     sha1nexte xmm1, [rbx]
    //     movdqa xmm4, xmm1
    //     sha1rnds4 xmm0, xmm1, 3
    //     add rbx, 16
    //     dec edx
    //     jnz 2b
    //     hlt
    let code = [
        0xba, 0x14, 0x00, 0x00, 0x00, 0x66, 0x0f, 0x6f, 0xc8, 0x0f, 0x38, 0xc8, 0xe6, 0x0f, 0x3a,
        0xcc, 0xc4, 0x02, 0x0f, 0x38, 0xc9, 0xee, 0x0f, 0x38, 0xca, 0xf7, 0x66, 0x0f, 0xef, 0xef,
        0x0f, 0x38, 0xc8, 0x0b, 0x66, 0x0f, 0x6f, 0xe1, 0x0f, 0x3a, 0xcc, 0xc1, 0x03, 0x48, 0x83,
        0xc3, 0x10, 0xff, 0xca, 0x75, 0xd2, HLT,
    ];
    let result = check_with(&code, sse_setup);
    assert_eq!(result.interpreter.cpu.gpr[2], 0, "the loop ran to the end");
}
