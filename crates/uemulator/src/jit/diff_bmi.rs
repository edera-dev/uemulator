//! BMI1/BMI2 instructions the JIT lowers: shlx, shrx, sarx, bzhi, andn and the blsr/blsmsk/blsi
//! group, at both operand sizes, from registers and from memory, over the count and index values
//! around the operand width. Flags are read back with `pushfq` after each one.

use super::diff_tests::{check_with, seed_registers_and_memory};
use crate::testkit::*;

const PUSHFQ_POP_R8: [u8; 3] = [0x9c, 0x41, 0x58];
const HLT: u8 = 0xf4;

/// Source operand values: zero, all ones, the sign bit, a lone low bit and an arbitrary pattern.
const SOURCES: [u64; 6] = [0, u64::MAX, 1 << 63, 1, 0x8000_0000, 0x1234_5678_9abc_def0];
/// Shift counts and bzhi indices: below, at and above both widths, and with garbage above the
/// low byte (bzhi reads only bits 7:0, shifts mask the count to the width).
const COUNTS: [u64; 13] = [
    0,
    1,
    7,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    200,
    0x1_0000_0003,
    0xffff_ffff_ffff_ff1f,
];

struct Encodings {
    /// `op rax, rcx, rbx` (or `op rax, rbx, rcx` for andn): register source.
    register: &'static [u8],
    /// The same with the r/m operand read from `[rsi]`.
    memory: &'static [u8],
}

fn run(encodings: &Encodings, source: u64, other: u64) {
    for form in [encodings.register, encodings.memory] {
        let code = [form, &PUSHFQ_POP_R8, &[HLT]].concat();
        check_with(&code, |harness| {
            seed_registers_and_memory(harness);
            harness.cpu.gpr[RCX] = source;
            harness.cpu.gpr[RBX] = other;
            harness.cpu.gpr[RAX] = 0xdead_beef_dead_beef;
            // [rsi] holds the same source for the memory form.
            let address = harness.cpu.gpr[RSI];
            harness.write_u64(address, source);
        });
    }
}

#[test]
fn shlx_shrx_sarx_shift_by_vvvv_modulo_the_width() {
    const FORMS: [Encodings; 6] = [
        // shlx rax, rcx, rbx / shlx rax, [rsi], rbx; 64-bit then 32-bit
        Encodings {
            register: &[0xc4, 0xe2, 0xe1, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0xe1, 0xf7, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0xe3, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0xe3, 0xf7, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0xe2, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0xe2, 0xf7, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x61, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0x61, 0xf7, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x63, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0x63, 0xf7, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x62, 0xf7, 0xc1],
            memory: &[0xc4, 0xe2, 0x62, 0xf7, 0x06],
        },
    ];
    for encodings in &FORMS {
        for source in SOURCES {
            for count in COUNTS {
                run(encodings, source, count);
            }
        }
    }
}

#[test]
fn bzhi_clears_from_the_index_and_reports_a_too_large_index_in_cf() {
    const FORMS: [Encodings; 2] = [
        Encodings {
            register: &[0xc4, 0xe2, 0xe0, 0xf5, 0xc1],
            memory: &[0xc4, 0xe2, 0xe0, 0xf5, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x60, 0xf5, 0xc1],
            memory: &[0xc4, 0xe2, 0x60, 0xf5, 0x06],
        },
    ];
    for encodings in &FORMS {
        for source in SOURCES {
            for index in COUNTS {
                run(encodings, source, index);
            }
        }
    }
}

#[test]
fn andn_and_the_blsr_group_set_flags_from_the_result() {
    const FORMS: [Encodings; 8] = [
        // andn rax, rbx, rcx (rbx is vvvv): 64-bit and 32-bit
        Encodings {
            register: &[0xc4, 0xe2, 0xe0, 0xf2, 0xc1],
            memory: &[0xc4, 0xe2, 0xe0, 0xf2, 0x06],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x60, 0xf2, 0xc1],
            memory: &[0xc4, 0xe2, 0x60, 0xf2, 0x06],
        },
        // blsr / blsmsk / blsi rax(vvvv), rcx, 64-bit
        Encodings {
            register: &[0xc4, 0xe2, 0xf8, 0xf3, 0xc9],
            memory: &[0xc4, 0xe2, 0xf8, 0xf3, 0x0e],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0xf8, 0xf3, 0xd1],
            memory: &[0xc4, 0xe2, 0xf8, 0xf3, 0x16],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0xf8, 0xf3, 0xd9],
            memory: &[0xc4, 0xe2, 0xf8, 0xf3, 0x1e],
        },
        // the same, 32-bit
        Encodings {
            register: &[0xc4, 0xe2, 0x78, 0xf3, 0xc9],
            memory: &[0xc4, 0xe2, 0x78, 0xf3, 0x0e],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x78, 0xf3, 0xd1],
            memory: &[0xc4, 0xe2, 0x78, 0xf3, 0x16],
        },
        Encodings {
            register: &[0xc4, 0xe2, 0x78, 0xf3, 0xd9],
            memory: &[0xc4, 0xe2, 0x78, 0xf3, 0x1e],
        },
    ];
    for encodings in &FORMS {
        for source in SOURCES {
            for other in [0, u64::MAX, 0xf0f0_f0f0_f0f0_f0f0, 0x8000_0000_0000_0001] {
                run(encodings, source, other);
            }
        }
    }
}

/// mulx: rdx times r/m into two registers, high half last, leaving the flags alone. A cmp ahead
/// of it sets CF, which a setc after it must still see (the WebAssembly generator keeps the cmp's
/// operands in locals mulx must not touch), and pushfq takes the rest.
#[test]
fn mulx_leaves_the_flags_of_the_instruction_before() {
    const CMP_RAX_RBX: [u8; 3] = [0x48, 0x39, 0xd8];
    // setc r10b; pushfq; pop r11; hlt
    const SETC_PUSHFQ_HLT: [u8; 8] = [0x41, 0x0f, 0x92, 0xc2, 0x9c, 0x41, 0x5b, 0xf4];
    const FORMS: [&[u8]; 5] = [
        &[0xc4, 0x62, 0xb3, 0xf6, 0xc1], // mulx r8, r9, rcx
        &[0xc4, 0x62, 0x33, 0xf6, 0xc1], // mulx r8d, r9d, ecx
        &[0xc4, 0x62, 0xb3, 0xf6, 0x06], // mulx r8, r9, qword ptr [rsi]
        &[0xc4, 0x62, 0x33, 0xf6, 0x06], // mulx r8d, r9d, dword ptr [rsi]
        &[0xc4, 0xe2, 0xf3, 0xf6, 0xcb], // mulx rcx, rcx, rbx: the high half wins
    ];
    for form in FORMS {
        let code = [&CMP_RAX_RBX[..], form, &SETC_PUSHFQ_HLT].concat();
        for source in SOURCES {
            for multiplier in SOURCES {
                check_with(&code, |harness| {
                    seed_registers_and_memory(harness);
                    harness.cpu.gpr[RAX] = 1;
                    harness.cpu.gpr[RBX] = source | 2;
                    harness.cpu.gpr[RCX] = source;
                    harness.cpu.gpr[RDX] = multiplier;
                    let address = harness.cpu.gpr[RSI];
                    harness.write_u64(address, source);
                });
            }
        }
    }
}
