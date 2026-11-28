//! Instructions a real boot sent to the interpreter (`helper_exec`) until they were lowered on
//! every backend: each case runs against the interpreter, and `lowered` checks that the inline
//! path is the one taken, because a fallback agrees with the interpreter by construction.

use super::diff_evex::{FILLINGS, setup, warmed};
use super::diff_tests::*;
use crate::testkit::*;

/// Runs each snippet (a TLB warm-up, then the code, then `hlt`) against the interpreter over the
/// AVX-512 register fillings, so vector, opmask and general registers all hold varied values.
fn run_table(table: &[(&str, &[u8])]) {
    for (name, code) in table {
        let code = warmed(&[code, &[0xf4][..]].concat());
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

/// Fails if any of `instructions`, each run a few times after the TLB warm-up, reaches the
/// interpreter on this backend.
#[cfg(feature = "jit")]
fn assert_lowered(instructions: &[&[u8]]) {
    const REPEATS: usize = 3;
    let mut missed = Vec::new();
    for instruction in instructions {
        let code = warmed(&[instruction.repeat(REPEATS), vec![0xf4]].concat());
        let mut harness = Harness::new(&code);
        if !harness.enable_jit() {
            return;
        }
        setup(&mut harness, 1);
        harness.cpu.helper_histogram = Some(Default::default());
        assert_eq!(harness.run(1000), Stop::Halted, "{instruction:x?}");
        let key = crate::cpu::decode::decode(instruction)
            .unwrap()
            .histogram_key();
        if harness
            .cpu
            .helper_histogram
            .as_ref()
            .unwrap()
            .get(&key)
            .is_some()
        {
            missed.push(*instruction);
        }
    }
    assert!(missed.is_empty(), "through the interpreter: {missed:x?}");
}

/// movzx/movsx from ah, ch, dh and bh (no REX), which the native generators used to refuse.
const HIGH_BYTE_EXTENDS: &[(&str, &[u8])] = &[
    ("movzx eax, ah", &[0x0f, 0xb6, 0xc4]),
    ("movzx ebp, bh", &[0x0f, 0xb6, 0xef]),
    ("movsx edx, ch", &[0x0f, 0xbe, 0xd5]),
    ("movzx eax, al", &[0x0f, 0xb6, 0xc0]),
    ("movzx esp, dh", &[0x0f, 0xb6, 0xe6]),
];

/// adcx and adox: alone, after flag writers of every lazy kind, interleaved the way bignum code
/// runs them, and followed by readers of the flags they keep and replace.
const CARRY_CHAINS: &[(&str, &[u8])] = &[
    // adcx rax, rbx | adcx eax, ebx
    ("adcx64", &[0x66, 0x48, 0x0f, 0x38, 0xf6, 0xc3]),
    ("adcx32", &[0x66, 0x0f, 0x38, 0xf6, 0xc3]),
    // adox r8, [rbx] | adox eax, [rbx+8]
    ("adox64 mem", &[0xf3, 0x4c, 0x0f, 0x38, 0xf6, 0x03]),
    ("adox32 mem", &[0xf3, 0x0f, 0x38, 0xf6, 0x43, 0x08]),
    // add rax, rbx; adcx rcx, rdx; adox rsi, rdi; adcx r8, r9; adox r10, r11
    (
        "after add",
        &[
            0x48, 0x01, 0xd8, 0x66, 0x48, 0x0f, 0x38, 0xf6, 0xca, 0xf3, 0x48, 0x0f, 0x38, 0xf6,
            0xf7, 0x66, 0x4d, 0x0f, 0x38, 0xf6, 0xc1, 0xf3, 0x4d, 0x0f, 0x38, 0xf6, 0xd3,
        ],
    ),
    // xor eax, eax; adcx rcx, [rbx]; adox rdx, [rbx+8]; adcx rcx, [rbx+16]; adox rdx, [rbx+24]
    (
        "after xor",
        &[
            0x31, 0xc0, 0x66, 0x48, 0x0f, 0x38, 0xf6, 0x0b, 0xf3, 0x48, 0x0f, 0x38, 0xf6, 0x53,
            0x08, 0x66, 0x48, 0x0f, 0x38, 0xf6, 0x4b, 0x10, 0xf3, 0x48, 0x0f, 0x38, 0xf6, 0x53,
            0x18,
        ],
    ),
    // sub rax, rbx; dec rcx; adox rdx, rsi; setc al; seto ah; sets bl; setz bh; setp cl
    (
        "after dec, read back",
        &[
            0x48, 0x29, 0xd8, 0x48, 0xff, 0xc9, 0xf3, 0x48, 0x0f, 0x38, 0xf6, 0xd6, 0x0f, 0x92,
            0xc0, 0x0f, 0x90, 0xc4, 0x0f, 0x98, 0xc3, 0x0f, 0x94, 0xc7, 0x0f, 0x9a, 0xc1,
        ],
    ),
    // stc; adcx rax, r10 (r10 = all ones: carry through every bit); jc +2; inc ebx; adcx eax, eax
    (
        "carry through",
        &[
            0xf9, 0x66, 0x49, 0x0f, 0x38, 0xf6, 0xc2, 0x72, 0x02, 0xff, 0xc3, 0x66, 0x0f, 0x38,
            0xf6, 0xc0,
        ],
    ),
    // shl rax, 1 (SzpCfOf record); adox rbx, rcx; cmovo edx, esi; adcx rbx, rcx; cmovc edi, esp
    (
        "after shift",
        &[
            0x48, 0xd1, 0xe0, 0xf3, 0x48, 0x0f, 0x38, 0xf6, 0xd9, 0x0f, 0x40, 0xd6, 0x66, 0x48,
            0x0f, 0x38, 0xf6, 0xd9, 0x0f, 0x42, 0xfc,
        ],
    ),
];

/// mulx: both widths, a memory source, the destinations aliasing each other and rdx.
const MULX: &[(&str, &[u8])] = &[
    ("mulx rax, rbx, rcx", &[0xc4, 0xe2, 0xe3, 0xf6, 0xc1]),
    ("mulx edx, ebx, [rbx]", &[0xc4, 0xe2, 0x63, 0xf6, 0x13]),
    ("mulx r9, r9, r10", &[0xc4, 0x42, 0xb3, 0xf6, 0xca]),
    ("mulx rcx, rdx, rdx", &[0xc4, 0xe2, 0xeb, 0xf6, 0xca]),
    // cmp rax, rbx; mulx rsi, rdi, r8; jb +2; inc ecx (the flags of the cmp survive)
    (
        "flags survive",
        &[
            0x48, 0x39, 0xd8, 0xc4, 0xc2, 0xc3, 0xf6, 0xf0, 0x72, 0x02, 0xff, 0xc1,
        ],
    ),
];

/// movd/movq between xmm and general registers or memory.
const MOVD: &[(&str, &[u8])] = &[
    ("movq xmm1, rax", &[0x66, 0x48, 0x0f, 0x6e, 0xc8]),
    ("movd xmm2, ecx", &[0x66, 0x0f, 0x6e, 0xd1]),
    ("movq rax, xmm3", &[0x66, 0x48, 0x0f, 0x7e, 0xd8]),
    ("movd ecx, xmm4", &[0x66, 0x0f, 0x7e, 0xe1]),
    ("movq xmm5, [rbx]", &[0x66, 0x48, 0x0f, 0x6e, 0x2b]),
    ("movd xmm5, [rbx+4]", &[0x66, 0x0f, 0x6e, 0x6b, 0x04]),
    ("movd [rbx+4], xmm6", &[0x66, 0x0f, 0x7e, 0x73, 0x04]),
    ("movq [rbx], xmm7", &[0x66, 0x48, 0x0f, 0x7e, 0x3b]),
    ("movd xmm9, r10d", &[0x66, 0x45, 0x0f, 0x6e, 0xca]),
    ("movq r15, xmm14", &[0x66, 0x4d, 0x0f, 0x7e, 0xf7]),
];

#[test]
fn high_byte_extends_match_the_interpreter() {
    run_table(HIGH_BYTE_EXTENDS);
}

#[test]
fn carry_chains_match_the_interpreter() {
    run_table(CARRY_CHAINS);
}

#[test]
fn mulx_matches_the_interpreter() {
    run_table(MULX);
}

#[test]
fn movd_matches_the_interpreter() {
    run_table(MOVD);
}

/// Without the SSE gate open (CR4.OSFXSR clear) movd raises #UD in the interpreter, and the
/// inline path must leave it to it.
#[test]
fn movd_without_osfxsr_faults_like_the_interpreter() {
    for (name, code) in MOVD {
        let code = warmed(&[code, &[0xf4][..]].concat());
        let result = run_checked(&code, |harness| {
            setup(harness, 1);
            harness.cpu.cr4 &= !(1 << 9);
        });
        assert!(
            matches!(result.interpreter_stop, Stop::Fault { .. }),
            "{name}: {:?}",
            result.interpreter_stop
        );
    }
}

/// Every form above reaches its inline path (the diff tests cannot tell).
#[test]
#[cfg(feature = "jit")]
fn the_new_forms_run_without_the_interpreter() {
    let single: Vec<&[u8]> = [HIGH_BYTE_EXTENDS, &MULX[..4], MOVD, &CARRY_CHAINS[..4]]
        .iter()
        .flat_map(|table| table.iter().map(|(_, code)| *code))
        .collect();
    assert_lowered(&single);
}

/// Unaligned vector loads and stores, which glibc's string functions are full of.
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
const UNALIGNED_VECTORS: &[&[u8]] = &[
    // vmovdqu64 zmm1, [rbx+3] | vmovdqu64 [rbx+69], zmm2
    &[0x62, 0xf1, 0xfe, 0x48, 0x6f, 0x8b, 0x03, 0x00, 0x00, 0x00],
    &[0x62, 0xf1, 0xfe, 0x48, 0x7f, 0x93, 0x45, 0x00, 0x00, 0x00],
    // movdqu xmm1, [rbx+1] | movdqu [rbx+9], xmm2
    &[0xf3, 0x0f, 0x6f, 0x4b, 0x01],
    &[0xf3, 0x0f, 0x7f, 0x53, 0x09],
    // vmovdqu ymm3, [rbx+7] | vmovdqu [rbx+13], ymm4 | vpcmpeqb k1, zmm5, [rbx+11]
    &[0xc5, 0xfe, 0x6f, 0x5b, 0x07],
    &[0xc5, 0xfe, 0x7f, 0x63, 0x0d],
    &[0x62, 0xf1, 0x55, 0x48, 0x74, 0x8b, 0x0b, 0x00, 0x00, 0x00],
    // movlpd xmm1, [rbx+3] | movsd [rbx+5], xmm2
    &[0x66, 0x0f, 0x12, 0x4b, 0x03],
    &[0xf2, 0x0f, 0x11, 0x53, 0x05],
];

/// With threads the native generators order every access (`ordered_memory`) and send misaligned
/// ones of up to 8 bytes to the interpreter (`ldar`/`stlr` need natural alignment). Wider ones
/// are plain vector or pair accesses with a barrier and must stay inline at any alignment, and
/// still agree with the interpreter.
#[test]
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn unaligned_vectors_stay_inline_with_ordered_memory() {
    const REPEATS: usize = 3;
    for instruction in UNALIGNED_VECTORS {
        let code = warmed(&[instruction.repeat(REPEATS), vec![0xf4]].concat());
        let mut interpreter = Harness::new(&code);
        setup(&mut interpreter, 1);
        interpreter.cpu.ordered_memory = true;
        assert_eq!(interpreter.run(1000), Stop::Halted, "{instruction:x?}");
        let mut jitted = Harness::new(&code);
        if !jitted.enable_jit() {
            return;
        }
        setup(&mut jitted, 1);
        jitted.cpu.ordered_memory = true;
        jitted.cpu.jit.as_mut().unwrap().set_ordered_memory(true);
        jitted.cpu.helper_histogram = Some(Default::default());
        assert_eq!(jitted.run(1000), Stop::Halted, "{instruction:x?}");
        assert_eq!(
            jitted.snapshot(),
            interpreter.snapshot(),
            "{instruction:x?}"
        );
        assert_eq!(
            jitted.read(SCRATCH, 0x1000),
            interpreter.read(SCRATCH, 0x1000),
            "{instruction:x?}"
        );
        let key = crate::cpu::decode::decode(instruction)
            .unwrap()
            .histogram_key();
        let interpreted = jitted.cpu.helper_histogram.as_ref().unwrap().get(&key);
        assert_eq!(interpreted, None, "{instruction:x?}");
    }
}

/// Like `run_table` for snippets put together from pieces; with `may_fault` a snippet may also
/// end in an exception (a #DE), as long as both engines raise the same one at the same place.
fn run_cases(cases: &[(&str, Vec<u8>)], may_fault: bool) {
    for (name, code) in cases {
        let code = warmed(&[code, &[0xf4][..]].concat());
        for filling in 0..FILLINGS {
            let result = std::panic::catch_unwind(|| {
                let result = run_checked(&code, |harness| setup(harness, filling));
                assert!(
                    matches!(result.interpreter_stop, Stop::Halted)
                        || may_fault && matches!(result.interpreter_stop, Stop::Fault { .. }),
                    "{:?}",
                    result.interpreter_stop
                );
            });
            if let Err(panic) = result {
                eprintln!("case `{name}`, filling {filling}");
                std::panic::resume_unwind(panic);
            }
        }
    }
}

const XOR_EDX: &[u8] = &[0x31, 0xd2];
const CQO: &[u8] = &[0x48, 0x99];
const CDQ: &[u8] = &[0x99];
/// cmp rax, rbx: a lazy flag record for what follows to keep or replace.
const CMP_RAX_RBX: &[u8] = &[0x48, 0x39, 0xd8];
/// setc al; setz ah; sets bl; seto bh; setp cl: every arithmetic flag but AF into registers.
const READ_FLAGS: &[u8] = &[
    0x0f, 0x92, 0xc0, 0x0f, 0x94, 0xc4, 0x0f, 0x98, 0xc3, 0x0f, 0x90, 0xc7, 0x0f, 0x9a, 0xc1,
];

fn cat(pieces: &[&[u8]]) -> Vec<u8> {
    pieces.concat()
}

fn divisions() -> Vec<(&'static str, Vec<u8>)> {
    let div_rcx: &[u8] = &[0x48, 0xf7, 0xf1];
    let div_ecx: &[u8] = &[0xf7, 0xf1];
    let idiv_rcx: &[u8] = &[0x48, 0xf7, 0xf9];
    let idiv_ecx: &[u8] = &[0xf7, 0xf9];
    let mov_rcx_minus_one: &[u8] = &[0x48, 0xc7, 0xc1, 0xff, 0xff, 0xff, 0xff];
    let mov_ecx_minus_one: &[u8] = &[0xb9, 0xff, 0xff, 0xff, 0xff];
    vec![
        ("div rcx", cat(&[XOR_EDX, div_rcx])),
        ("div ecx", cat(&[XOR_EDX, div_ecx])),
        // div qword [rbx] | div dword [rbx+8] (the seeded memory may be zero: #DE)
        ("div mem64", cat(&[XOR_EDX, &[0x48, 0xf7, 0x33]])),
        ("div mem32", cat(&[XOR_EDX, &[0xf7, 0x73, 0x08]])),
        ("idiv rcx", cat(&[CQO, idiv_rcx])),
        ("idiv ecx", cat(&[CDQ, idiv_ecx])),
        // idiv qword [rbx] | idiv dword [rbx+4]
        ("idiv mem64", cat(&[CQO, &[0x48, 0xf7, 0x3b]])),
        ("idiv mem32", cat(&[CDQ, &[0xf7, 0x7b, 0x04]])),
        // rdx as seeded is neither zero nor the sign of rax: 128-bit dividends, and #DE
        ("div rcx, 128-bit dividend", div_rcx.to_vec()),
        ("idiv rcx, 128-bit dividend", idiv_rcx.to_vec()),
        // mov rdx, 1; div r10 (r10 = all ones): a 128-bit dividend whose quotient fits
        (
            "div r10, wide but fits",
            cat(&[
                &[0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00],
                &[0x49, 0xf7, 0xf2],
            ]),
        ),
        // mov edx, 3; div ecx (4): fits; mov edx, 1; mov ecx, 1; div ecx: does not
        ("div ecx, edx set", cat(&[&[0xba, 0x03, 0, 0, 0], div_ecx])),
        (
            "div ecx, overflow",
            cat(&[&[0xba, 0x01, 0, 0, 0], &[0xb9, 0x01, 0, 0, 0], div_ecx]),
        ),
        // mov rax, i64::MIN; cqo; mov rcx, -1; idiv rcx
        (
            "idiv rcx, MIN / -1",
            cat(&[
                &[0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0x80],
                CQO,
                mov_rcx_minus_one,
                idiv_rcx,
            ]),
        ),
        // mov eax, 0x80000000; cdq; mov ecx, -1; idiv ecx
        (
            "idiv ecx, MIN / -1",
            cat(&[&[0xb8, 0, 0, 0, 0x80], CDQ, mov_ecx_minus_one, idiv_ecx]),
        ),
        // mov rax, -7; cqo; mov rcx, 2; idiv rcx (rounds toward zero)
        (
            "idiv rcx, negative",
            cat(&[
                &[0x48, 0xc7, 0xc0, 0xf9, 0xff, 0xff, 0xff],
                CQO,
                &[0x48, 0xc7, 0xc1, 0x02, 0, 0, 0],
                idiv_rcx,
            ]),
        ),
        // div rcx by zero: xor edx, edx; xor ecx, ecx; div rcx
        ("div by zero", cat(&[XOR_EDX, &[0x31, 0xc9], div_rcx])),
        // the flags survive: cmp rax, rbx; xor edx, edx is a writer, so read them after a div
        (
            "flags after div",
            cat(&[XOR_EDX, CMP_RAX_RBX, div_rcx, READ_FLAGS]),
        ),
    ]
}

/// bt/bts/btr/btc in every form, between a flag writer and readers of all the flags.
const BIT_TESTS: &[(&str, &[u8])] = &[
    ("bt rax, rcx", &[0x48, 0x0f, 0xa3, 0xc8]),
    ("bt eax, 33", &[0x0f, 0xba, 0xe0, 0x21]),
    ("bts rbx, 63", &[0x48, 0x0f, 0xba, 0xeb, 0x3f]),
    ("btr rdx, rsi", &[0x48, 0x0f, 0xb3, 0xf2]),
    ("btc ecx, edi", &[0x0f, 0xbb, 0xf9]),
    ("bt [rbx], rcx", &[0x48, 0x0f, 0xa3, 0x0b]),
    ("bts [rbx+16], esi", &[0x0f, 0xab, 0x73, 0x10]),
    ("btr [rbx+64], 5", &[0x48, 0x0f, 0xba, 0x73, 0x40, 0x05]),
    ("btc [rbx], 31", &[0x0f, 0xba, 0x3b, 0x1f]),
    (
        "bt [rbx+256], eax",
        &[0x0f, 0xa3, 0x83, 0x00, 0x01, 0x00, 0x00],
    ),
];

fn bit_tests() -> Vec<(&'static str, Vec<u8>)> {
    // mov rcx, -100 and mov esi, -3: negative offsets reach below the address
    let negative_offsets: &[u8] = &[
        0x48, 0xc7, 0xc1, 0x9c, 0xff, 0xff, 0xff, 0xbe, 0xfd, 0xff, 0xff, 0xff,
    ];
    BIT_TESTS
        .iter()
        .flat_map(|(name, code)| {
            [
                (*name, cat(&[CMP_RAX_RBX, code, READ_FLAGS])),
                (*name, cat(&[negative_offsets, code, READ_FLAGS])),
                (*name, cat(&[code, code])),
            ]
        })
        .collect()
}

/// leave, bswap and xchg between registers.
const MISC_INTEGER: &[(&str, &[u8])] = &[
    ("leave", &[0xc9]),
    ("bswap eax", &[0x0f, 0xc8]),
    ("bswap r9", &[0x49, 0x0f, 0xc9]),
    ("xchg rax, rcx", &[0x48, 0x91]),
    ("xchg r8d, eax", &[0x41, 0x90]),
    ("xchg rsi, rdi", &[0x48, 0x87, 0xf7]),
    ("xchg ecx, ecx", &[0x87, 0xc9]),
    ("xchg eax, eax", &[0x87, 0xc0]),
    ("xchg rax, r8", &[0x49, 0x90]),
];

#[test]
fn divisions_match_the_interpreter() {
    run_cases(&divisions(), true);
}

#[test]
fn bit_tests_match_the_interpreter() {
    run_cases(&bit_tests(), true);
}

#[test]
fn leave_bswap_and_xchg_match_the_interpreter() {
    run_table(MISC_INTEGER);
}

/// A leave whose load faults (rbp unmapped) changes neither rsp nor rbp.
#[test]
fn a_faulting_leave_matches_the_interpreter() {
    let result = run_checked(&warmed(&[0xc9, 0xf4]), |harness| {
        setup(harness, 1);
        harness.cpu.gpr[5] = 0x7fff_ffff_f000;
    });
    assert!(matches!(result.interpreter_stop, Stop::Fault { .. }));
}

#[test]
#[cfg(feature = "jit")]
fn batch_two_forms_run_without_the_interpreter() {
    // leave runs once: a second one would pop from wherever the first left rbp.
    let mut forms: Vec<&[u8]> = MISC_INTEGER[1..].iter().map(|(_, code)| *code).collect();
    forms.extend(BIT_TESTS[..9].iter().map(|(_, code)| *code));
    assert_lowered_after(&[], &[MISC_INTEGER[0].1, BIT_TESTS[9].1]);
    // div rcx with rdx = 0 and idiv rcx with rdx the sign of rax run inline; the repeated form
    // keeps them valid: cqo; idiv rcx and xor edx, edx; div rcx would each be two forms.
    assert_lowered(&forms);
    assert_lowered_after(XOR_EDX, &[&[0x48, 0xf7, 0xf1], &[0xf7, 0xf1]]);
}

/// `assert_lowered` with `prefix` (run once, not counted) ahead of each instruction.
#[cfg(feature = "jit")]
fn assert_lowered_after(prefix: &[u8], instructions: &[&[u8]]) {
    let mut missed = Vec::new();
    for instruction in instructions {
        let code = warmed(&[prefix, instruction, &[0xf4]].concat());
        let mut harness = Harness::new(&code);
        if !harness.enable_jit() {
            return;
        }
        setup(&mut harness, 1);
        harness.cpu.helper_histogram = Some(Default::default());
        assert_eq!(harness.run(1000), Stop::Halted, "{instruction:x?}");
        let key = crate::cpu::decode::decode(instruction)
            .unwrap()
            .histogram_key();
        if harness
            .cpu
            .helper_histogram
            .as_ref()
            .unwrap()
            .get(&key)
            .is_some()
        {
            missed.push(*instruction);
        }
    }
    assert!(missed.is_empty(), "through the interpreter: {missed:x?}");
}

/// Locked read-modify-writes, xchg with memory, and cmpxchg/xadd with memory without LOCK.
const MEMORY_RMW: &[(&str, &[u8])] = &[
    ("lock add [rbx], rcx", &[0xf0, 0x48, 0x01, 0x0b]),
    ("lock or [rbx+4], esi", &[0xf0, 0x09, 0x73, 0x04]),
    (
        "lock and [rbx+8], -16",
        &[0xf0, 0x48, 0x83, 0x63, 0x08, 0xf0],
    ),
    ("lock sub [rbx+12], 7", &[0xf0, 0x83, 0x6b, 0x0c, 0x07]),
    ("lock xor [rbx+16], rdx", &[0xf0, 0x48, 0x31, 0x53, 0x10]),
    ("lock inc [rbx+24]", &[0xf0, 0x48, 0xff, 0x43, 0x18]),
    ("lock dec [rbx+32]", &[0xf0, 0xff, 0x4b, 0x20]),
    (
        "lock xadd [rbx+40], rcx",
        &[0xf0, 0x48, 0x0f, 0xc1, 0x4b, 0x28],
    ),
    ("lock xadd [rbx+48], esi", &[0xf0, 0x0f, 0xc1, 0x73, 0x30]),
    (
        "lock cmpxchg [rbx+56], rcx",
        &[0xf0, 0x48, 0x0f, 0xb1, 0x4b, 0x38],
    ),
    (
        "lock cmpxchg [rbx+64], esi",
        &[0xf0, 0x0f, 0xb1, 0x73, 0x40],
    ),
    ("xchg [rbx+72], rdx", &[0x48, 0x87, 0x53, 0x48]),
    ("xchg [rbx+80], eax", &[0x87, 0x43, 0x50]),
    ("cmpxchg [rbx], rsi", &[0x48, 0x0f, 0xb1, 0x33]),
    ("cmpxchg [rbx+4], edi", &[0x0f, 0xb1, 0x7b, 0x04]),
    ("xadd [rbx+8], rdx", &[0x48, 0x0f, 0xc1, 0x53, 0x08]),
    ("lock bts [rbx], 5", &[0xf0, 0x48, 0x0f, 0xba, 0x2b, 0x05]),
    (
        "lock btr [rbx+4], 31",
        &[0xf0, 0x0f, 0xba, 0x73, 0x04, 0x1f],
    ),
    (
        "lock btc [rbx+8], 63",
        &[0xf0, 0x48, 0x0f, 0xba, 0x7b, 0x08, 0x3f],
    ),
    // misaligned: the interpreter's path
    ("lock add [rbx+3], rcx", &[0xf0, 0x48, 0x01, 0x4b, 0x03]),
    (
        "lock cmpxchg [rbx+5], rcx",
        &[0xf0, 0x48, 0x0f, 0xb1, 0x4b, 0x05],
    ),
    // another page, not in the TLB: the interpreter's path
    (
        "lock inc [rbx+5000]",
        &[0xf0, 0x48, 0xff, 0x83, 0x88, 0x13, 0x00, 0x00],
    ),
];

fn memory_rmw_cases() -> Vec<(&'static str, Vec<u8>)> {
    // mov rax, [rbx+56]; mov eax, [rbx+4] and mov rax, [rbx]: cmpxchg that succeed
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "lock cmpxchg [rbx+56], rcx, equal",
            cat(&[&[0x48, 0x8b, 0x43, 0x38], MEMORY_RMW[9].1, READ_FLAGS]),
        ),
        (
            "cmpxchg [rbx+4], edi, equal",
            cat(&[&[0x8b, 0x43, 0x04], MEMORY_RMW[14].1, READ_FLAGS]),
        ),
        (
            "cmpxchg [rbx], rsi, equal",
            cat(&[&[0x48, 0x8b, 0x03], MEMORY_RMW[13].1, READ_FLAGS]),
        ),
    ];
    for (name, code) in MEMORY_RMW {
        cases.push((name, cat(&[CMP_RAX_RBX, code, READ_FLAGS])));
        cases.push((name, cat(&[code, code])));
    }
    cases
}

#[test]
fn memory_read_modify_writes_match_the_interpreter() {
    run_cases(&memory_rmw_cases(), false);
}

/// The same with threads' ordering (`ordered_memory`), where the locked forms are exclusive
/// load/store loops on aarch64.
#[test]
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn memory_read_modify_writes_match_the_interpreter_with_ordered_memory() {
    for (name, code) in memory_rmw_cases() {
        let code = warmed(&[&code[..], &[0xf4]].concat());
        let mut interpreter = Harness::new(&code);
        setup(&mut interpreter, 1);
        interpreter.cpu.ordered_memory = true;
        assert_eq!(interpreter.run(1000), Stop::Halted, "{name}");
        let mut jitted = Harness::new(&code);
        if !jitted.enable_jit() {
            return;
        }
        setup(&mut jitted, 1);
        jitted.cpu.ordered_memory = true;
        jitted.cpu.jit.as_mut().unwrap().set_ordered_memory(true);
        jitted.cpu.helper_histogram = Some(Default::default());
        assert_eq!(jitted.run(1000), Stop::Halted, "{name}");
        assert_eq!(jitted.snapshot(), interpreter.snapshot(), "{name}");
        assert_eq!(
            jitted.read(SCRATCH, 0x2000),
            interpreter.read(SCRATCH, 0x2000),
            "{name}"
        );
        // Everything in these snippets but the misaligned and far operands, and the final hlt,
        // is inline.
        let hlt = crate::cpu::decode::decode(&[0xf4]).unwrap().histogram_key();
        let mut interpreted = jitted.cpu.helper_histogram.take().unwrap();
        interpreted.remove(&hlt);
        let inline = !["+3]", "+5]", "+5000]"]
            .iter()
            .any(|far| name.contains(far));
        assert!(
            !inline || interpreted.is_empty(),
            "{name}: {:?}",
            interpreted
                .keys()
                .map(|key| crate::cpu::decode::histogram_label(*key))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
#[cfg(feature = "jit")]
fn memory_read_modify_writes_run_without_the_interpreter() {
    let aligned: Vec<&[u8]> = MEMORY_RMW[..19].iter().map(|(_, code)| *code).collect();
    assert_lowered(&aligned);
}

/// SSE instructions the shared `whole_vector` classifier sends through `helper_whole_vector`
/// (or a native form) on every backend.
const SSE_SHUFFLES: &[(&str, &[u8])] = &[
    ("punpcklqdq xmm1, xmm2", &[0x66, 0x0f, 0x6c, 0xca]),
    ("punpckhqdq xmm3, [rbx]", &[0x66, 0x0f, 0x6d, 0x1b]),
    ("punpckldq xmm4, xmm9", &[0x66, 0x41, 0x0f, 0x62, 0xe1]),
    ("punpcklbw xmm5, [rbx+16]", &[0x66, 0x0f, 0x60, 0x6b, 0x10]),
    ("packuswb xmm6, xmm7", &[0x66, 0x0f, 0x67, 0xf7]),
    ("pcmpgtb xmm1, xmm2", &[0x66, 0x0f, 0x64, 0xca]),
    ("psubusb xmm2, xmm3", &[0x66, 0x0f, 0xd8, 0xd3]),
    ("pmullw xmm3, [rbx]", &[0x66, 0x0f, 0xd5, 0x1b]),
    ("psrlw xmm4, xmm5", &[0x66, 0x0f, 0xd1, 0xe5]),
    ("psraw xmm6, xmm1", &[0x66, 0x0f, 0xe1, 0xf1]),
    ("pavgb xmm7, xmm8", &[0x66, 0x41, 0x0f, 0xe0, 0xf8]),
    ("pmaddwd xmm1, xmm2", &[0x66, 0x0f, 0xf5, 0xca]),
    ("pmulhuw xmm1, xmm2", &[0x66, 0x0f, 0xe4, 0xca]),
    ("psadbw xmm1, xmm2", &[0x66, 0x0f, 0xf6, 0xca]),
    ("pshuflw xmm1, xmm2, 0x1b", &[0xf2, 0x0f, 0x70, 0xca, 0x1b]),
    ("pshufhw xmm3, [rbx], 0xe4", &[0xf3, 0x0f, 0x70, 0x1b, 0xe4]),
    ("pshuflw xmm4, xmm4, 0", &[0xf2, 0x0f, 0x70, 0xe4, 0x00]),
    (
        "pblendw xmm1, xmm2, 0xa5",
        &[0x66, 0x0f, 0x3a, 0x0e, 0xca, 0xa5],
    ),
    (
        "pblendw xmm3, [rbx], 0x0f",
        &[0x66, 0x0f, 0x3a, 0x0e, 0x1b, 0x0f],
    ),
    ("movhlps xmm1, xmm2", &[0x0f, 0x12, 0xca]),
    ("movlhps xmm3, xmm4", &[0x0f, 0x16, 0xdc]),
];

#[test]
fn sse_shuffles_match_the_interpreter() {
    run_table(SSE_SHUFFLES);
}

/// A misaligned 16-byte operand raises #GP in the interpreter, which the inline path leaves to it.
#[test]
fn a_misaligned_sse_operand_faults_like_the_interpreter() {
    // punpckhqdq xmm3, [rbx+8]
    let result = run_checked(&warmed(&[0x66, 0x0f, 0x6d, 0x5b, 0x08, 0xf4]), |harness| {
        setup(harness, 1)
    });
    assert!(matches!(result.interpreter_stop, Stop::Fault { .. }));
}

#[test]
#[cfg(feature = "jit")]
fn sse_shuffles_run_without_the_interpreter() {
    let forms: Vec<&[u8]> = SSE_SHUFFLES.iter().map(|(_, code)| *code).collect();
    assert_lowered(&forms);
}

/// Moves of part of an xmm register (`helpers::partial_move`).
const PARTIAL_MOVES: &[(&str, &[u8])] = &[
    ("movsd xmm1, [rbx]", &[0xf2, 0x0f, 0x10, 0x0b]),
    ("movss xmm2, [rbx+4]", &[0xf3, 0x0f, 0x10, 0x53, 0x04]),
    ("movsd xmm3, xmm4", &[0xf2, 0x0f, 0x10, 0xdc]),
    ("movss xmm5, xmm6", &[0xf3, 0x0f, 0x10, 0xee]),
    ("movsd [rbx+8], xmm7", &[0xf2, 0x0f, 0x11, 0x7b, 0x08]),
    (
        "movss [rbx+12], xmm8",
        &[0xf3, 0x44, 0x0f, 0x11, 0x43, 0x0c],
    ),
    (
        "movsd xmm4 (rm), xmm3 (store form)",
        &[0xf2, 0x0f, 0x11, 0xdc],
    ),
    ("movlpd xmm1, [rbx]", &[0x66, 0x0f, 0x12, 0x0b]),
    ("movlps xmm2, [rbx+3]", &[0x0f, 0x12, 0x53, 0x03]),
    ("movhpd xmm3, [rbx+16]", &[0x66, 0x0f, 0x16, 0x5b, 0x10]),
    ("movhps xmm4, [rbx+24]", &[0x0f, 0x16, 0x63, 0x18]),
    ("movlpd [rbx+32], xmm5", &[0x66, 0x0f, 0x13, 0x6b, 0x20]),
    ("movhps [rbx+40], xmm6", &[0x0f, 0x17, 0x73, 0x28]),
    ("movq xmm7, [rbx]", &[0xf3, 0x0f, 0x7e, 0x3b]),
    ("movq xmm8, xmm9", &[0xf3, 0x45, 0x0f, 0x7e, 0xc1]),
    ("movsd xmm10, xmm10", &[0xf2, 0x45, 0x0f, 0x10, 0xd2]),
];

#[test]
fn partial_moves_match_the_interpreter() {
    run_table(PARTIAL_MOVES);
}

#[test]
#[cfg(feature = "jit")]
fn partial_moves_run_without_the_interpreter() {
    let forms: Vec<&[u8]> = PARTIAL_MOVES.iter().map(|(_, code)| *code).collect();
    assert_lowered(&forms);
}

const PAUSE: &[u8] = &[0xf3, 0x90];
const RDTSC: &[u8] = &[0x0f, 0x31];
const PUSHF: &[u8] = &[0x9c];
const STI: &[u8] = &[0xfb];
const CLI: &[u8] = &[0xfa];
/// pop rax
const POP_RAX: &[u8] = &[0x58];

/// The opmask tests at every width, followed by readers of every flag.
const OPMASK_TESTS: &[(&str, &[u8])] = &[
    ("kortestw k1, k2", &[0xc5, 0xf8, 0x98, 0xca]),
    ("kortestb k3, k4", &[0xc5, 0xf9, 0x98, 0xdc]),
    ("kortestd k5, k6", &[0xc4, 0xe1, 0xf9, 0x98, 0xee]),
    ("kortestq k7, k1", &[0xc4, 0xe1, 0xf8, 0x98, 0xf9]),
    ("ktestw k1, k2", &[0xc5, 0xf8, 0x99, 0xca]),
    ("ktestq k3, k4", &[0xc4, 0xe1, 0xf8, 0x99, 0xdc]),
];

fn system_cases() -> Vec<(&'static str, Vec<u8>)> {
    let mut cases = vec![
        ("pause", cat(&[PAUSE, PAUSE])),
        // the TSC counts the instructions before it in the block
        (
            "rdtsc",
            cat(&[CMP_RAX_RBX, CMP_RAX_RBX, RDTSC, PAUSE, RDTSC]),
        ),
        ("pushf after cmp", cat(&[CMP_RAX_RBX, PUSHF, POP_RAX])),
        ("pushf twice", cat(&[PUSHF, PUSHF, POP_RAX])),
        ("cli; sti", cat(&[CLI, STI, PAUSE])),
        ("sti; sti", cat(&[STI, STI])),
    ];
    for (name, code) in OPMASK_TESTS {
        cases.push((name, cat(&[CMP_RAX_RBX, code, READ_FLAGS])));
    }
    cases
}

#[test]
fn system_instructions_match_the_interpreter() {
    run_cases(&system_cases(), false);
}

/// rdtsc above CPL 0 with CR4.TSD raises #GP; with TSD at CPL 0 it runs.
#[test]
fn rdtsc_with_cr4_tsd_matches_the_interpreter() {
    const CR4_TSD: u64 = 1 << 2;
    let result = run_checked(&warmed(&[RDTSC, &[0xf4]].concat()), |harness| {
        setup(harness, 1);
        harness.cpu.cr4 |= CR4_TSD;
    });
    assert_eq!(result.interpreter_stop, Stop::Halted);
}

/// With more than one vCPU taking turns, pause asks for a hand-over; the block stops there and
/// the run goes on as in the interpreter.
#[test]
fn pause_with_several_vcpus_matches_the_interpreter() {
    let code = warmed(&[PAUSE, CMP_RAX_RBX, PAUSE, &[0xf4]].concat());
    let result = run_checked(&code, |harness| {
        setup(harness, 1);
        harness.cpu.multi = true;
    });
    assert_eq!(result.interpreter_stop, Stop::Halted);
}

#[test]
#[cfg(feature = "jit")]
fn system_instructions_run_without_the_interpreter() {
    let mut forms: Vec<&[u8]> = vec![PAUSE, RDTSC, PUSHF, CLI];
    forms.extend(OPMASK_TESTS.iter().map(|(_, code)| *code));
    assert_lowered(&forms);
    // sti ends its block: once is enough
    assert_lowered_after(CLI, &[STI]);
}

/// sti opens the interrupt shadow (not part of `Snapshot`) exactly when IF was clear.
#[test]
fn sti_leaves_the_interrupt_shadow_as_the_interpreter() {
    for code in [cat(&[CLI, STI]), cat(&[STI, STI])] {
        let result = run_checked(&warmed(&[&code[..], &[0xf4]].concat()), |harness| {
            setup(harness, 1)
        });
        if let Some((jit, _)) = &result.jit {
            assert_eq!(
                jit.cpu.interrupt_shadow, result.interpreter.cpu.interrupt_shadow,
                "{code:x?}"
            );
        }
    }
}

/// mov r/m, sreg and cld/std.
const SEGMENT_AND_DIRECTION: &[(&str, &[u8])] = &[
    ("mov eax, ds", &[0x8c, 0xd8]),
    ("mov rcx, fs", &[0x48, 0x8c, 0xe1]),
    ("mov dx, ss", &[0x66, 0x8c, 0xd2]),
    ("mov [rbx], gs", &[0x8c, 0x2b]),
    ("cld", &[0xfc]),
    ("std", &[0xfd]),
];

#[test]
fn segment_reads_and_direction_match_the_interpreter() {
    run_table(SEGMENT_AND_DIRECTION);
}

#[test]
#[cfg(feature = "jit")]
fn segment_reads_and_direction_run_without_the_interpreter() {
    let forms: Vec<&[u8]> = SEGMENT_AND_DIRECTION
        .iter()
        .map(|(_, code)| *code)
        .collect();
    assert_lowered(&forms);
}

const RDTSCP: &[u8] = &[0x0f, 0x01, 0xf9];
const RDPKRU: &[u8] = &[0x0f, 0x01, 0xee];
/// xor ecx, ecx
const XOR_ECX: &[u8] = &[0x31, 0xc9];
const CR4_PKE: u64 = 1 << 22;

/// rdtscp, rdpkru (with and without CR4.PKE, with ecx zero and not), and register forms with a
/// 67 prefix.
#[test]
fn rdtscp_rdpkru_and_addr32_registers_match_the_interpreter() {
    let cases = [
        ("rdtscp", cat(&[CMP_RAX_RBX, RDTSCP, RDTSCP])),
        ("rdpkru", cat(&[XOR_ECX, RDPKRU])),
        ("rdpkru, ecx not zero", RDPKRU.to_vec()),
        // addr32 inc rax | addr32 add eax, ebx
        ("addr32 inc rax", vec![0x67, 0x48, 0xff, 0xc0]),
        ("addr32 add eax, ebx", vec![0x67, 0x01, 0xd8]),
    ];
    for pke in [0, CR4_PKE] {
        for (name, code) in &cases {
            let result = run_checked(&warmed(&[&code[..], &[0xf4]].concat()), |harness| {
                setup(harness, 1);
                harness.cpu.cr4 |= pke;
                harness.cpu.pkru = 0x5555_0004;
                harness.cpu.tsc_aux = 0xdead_beef_0000_0007;
            });
            assert!(
                matches!(result.interpreter_stop, Stop::Halted | Stop::Fault { .. }),
                "{name}"
            );
        }
    }
}

#[test]
#[cfg(feature = "jit")]
fn rdtscp_rdpkru_and_addr32_registers_run_without_the_interpreter() {
    assert_lowered(&[RDTSCP, &[0x67, 0x48, 0xff, 0xc0], &[0x67, 0x01, 0xd8]]);
    let harness_pke = |instruction: &[u8]| {
        let code = warmed(&[XOR_ECX, instruction, instruction, &[0xf4]].concat());
        let mut harness = Harness::new(&code);
        if !harness.enable_jit() {
            return;
        }
        setup(&mut harness, 1);
        harness.cpu.cr4 |= CR4_PKE;
        harness.cpu.helper_histogram = Some(Default::default());
        assert_eq!(harness.run(1000), Stop::Halted);
        let key = crate::cpu::decode::decode(instruction)
            .unwrap()
            .histogram_key();
        assert_eq!(
            harness.cpu.helper_histogram.as_ref().unwrap().get(&key),
            None
        );
    };
    harness_pke(RDPKRU);
}

const CMPXCHG16B: &[u8] = &[0x48, 0x0f, 0xc7, 0x0b];
const LOCK_CMPXCHG16B: &[u8] = &[0xf0, 0x48, 0x0f, 0xc7, 0x4b, 0x10];
/// mov rax, [rbx]; mov rdx, [rbx+8]: rdx:rax equal to the operand of `CMPXCHG16B`
const LOAD_RDX_RAX: &[u8] = &[0x48, 0x8b, 0x03, 0x48, 0x8b, 0x53, 0x08];

fn cmpxchg16b_cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        (
            "cmpxchg16b, differ",
            cat(&[CMP_RAX_RBX, CMPXCHG16B, READ_FLAGS]),
        ),
        (
            "cmpxchg16b, equal",
            cat(&[LOAD_RDX_RAX, CMP_RAX_RBX, CMPXCHG16B, READ_FLAGS]),
        ),
        ("cmpxchg16b twice", cat(&[CMPXCHG16B, CMPXCHG16B])),
        ("lock cmpxchg16b", cat(&[LOCK_CMPXCHG16B, READ_FLAGS])),
        // cmpxchg16b [rbx+8]: misaligned, #GP
        ("cmpxchg16b, misaligned", vec![0x48, 0x0f, 0xc7, 0x4b, 0x08]),
    ]
}

#[test]
fn cmpxchg16b_matches_the_interpreter() {
    run_cases(&cmpxchg16b_cases(), true);
}

#[test]
#[cfg(all(feature = "jit", any(target_arch = "aarch64", target_arch = "x86_64")))]
fn cmpxchg16b_matches_the_interpreter_with_ordered_memory() {
    for (name, code) in cmpxchg16b_cases() {
        let code = warmed(&[&code[..], &[0xf4]].concat());
        let mut interpreter = Harness::new(&code);
        setup(&mut interpreter, 1);
        interpreter.cpu.ordered_memory = true;
        let stop = interpreter.run(1000);
        let mut jitted = Harness::new(&code);
        if !jitted.enable_jit() {
            return;
        }
        setup(&mut jitted, 1);
        jitted.cpu.ordered_memory = true;
        jitted.cpu.jit.as_mut().unwrap().set_ordered_memory(true);
        assert_eq!(jitted.run(1000), stop, "{name}");
        assert_eq!(jitted.snapshot(), interpreter.snapshot(), "{name}");
        assert_eq!(
            jitted.read(SCRATCH, 0x1000),
            interpreter.read(SCRATCH, 0x1000),
            "{name}"
        );
    }
}

#[test]
#[cfg(feature = "jit")]
fn cmpxchg16b_runs_without_the_interpreter() {
    assert_lowered(&[CMPXCHG16B, LOCK_CMPXCHG16B]);
}

const VZEROUPPER: &[u8] = &[0xc5, 0xf8, 0x77];
const VZEROALL: &[u8] = &[0xc5, 0xfc, 0x77];

#[test]
fn vzeroupper_and_vzeroall_match_the_interpreter() {
    run_table(&[("vzeroupper", VZEROUPPER), ("vzeroall", VZEROALL)]);
}

/// Without XCR0's AVX state they raise #UD in the interpreter.
#[test]
fn vzeroupper_without_avx_state_faults_like_the_interpreter() {
    let result = run_checked(&warmed(&[VZEROUPPER, &[0xf4]].concat()), |harness| {
        setup(harness, 1);
        harness.cpu.xcr0 = 1;
    });
    assert!(matches!(result.interpreter_stop, Stop::Fault { .. }));
}

#[test]
#[cfg(feature = "jit")]
fn vzeroupper_and_vzeroall_run_without_the_interpreter() {
    assert_lowered(&[VZEROUPPER, VZEROALL]);
}

/// swapgs, clac and stac at CPL 0 (the kernel's syscall entry and exit).
const RING0_SYSTEM: &[(&str, &[u8])] = &[
    ("swapgs", &[0x0f, 0x01, 0xf8]),
    ("clac", &[0x0f, 0x01, 0xca]),
    ("stac", &[0x0f, 0x01, 0xcb]),
];

#[test]
fn ring0_system_instructions_match_the_interpreter() {
    run_table(RING0_SYSTEM);
}

#[test]
#[cfg(feature = "jit")]
fn ring0_system_instructions_run_without_the_interpreter() {
    let forms: Vec<&[u8]> = RING0_SYSTEM.iter().map(|(_, code)| *code).collect();
    assert_lowered(&forms);
}
