//! AMD's additions to the SSE family: SSE4A, misaligned SSE mode (MXCSR.MM), and the MMX-register
//! forms of SSSE3 that AMD parts implement alongside the XMM ones. Expected values come from
//! scalar models written out here, not from the XMM kernels the implementation reuses.

use crate::cpu::exception::{
    VECTOR_DEVICE_NOT_AVAILABLE, VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE,
};
use crate::testkit::{CODE_BASE, HLT, Harness, RSI, SCRATCH, Stop, assert_jit_matches_interpreter};

const DATA: u64 = SCRATCH + 0x1000;
const MXCSR_MM: u32 = 1 << 17;
const MXCSR_DEFAULT: u32 = 0x1f80;
const CR4_OSFXSR_BIT: u64 = 1 << 9;
const CR0_EM_BIT: u64 = 1 << 2;
const CR0_TS_BIT: u64 = 1 << 3;

fn fault(vector: u8, error_code: Option<u32>) -> Stop {
    Stop::Fault {
        vector,
        error_code,
        software: false,
    }
}

fn gp0() -> Stop {
    fault(VECTOR_GENERAL_PROTECTION, Some(0))
}

fn ud() -> Stop {
    fault(VECTOR_INVALID_OPCODE, None)
}

fn sse_on(h: &mut Harness) {
    h.cpu.cr4 |= CR4_OSFXSR_BIT;
    h.cpu.gpr[RSI] = DATA;
}

fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
    let mut h = Harness::new(code);
    sse_on(&mut h);
    setup(&mut h);
    let stop = h.run(100);
    (h, stop)
}

/// `code` followed by `hlt`.
fn snippet(code: &[u8]) -> Vec<u8> {
    let mut bytes = code.to_vec();
    bytes.push(HLT);
    bytes
}

/// A deterministic spread of 64-bit values with the lane boundaries every op cares about.
fn samples() -> Vec<u64> {
    let mut values = vec![
        0,
        u64::MAX,
        0x8000_8000_8000_8000,
        0x7fff_7fff_7fff_7fff,
        0x8080_8080_8080_8080,
        0x7f7f_7f7f_7f7f_7f7f,
        0x0001_ffff_8000_7fff,
        0x0102_0304_0506_0708,
        0x8000_0000_7fff_ffff,
        0xff00_ff00_00ff_00ff,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..12 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(state);
    }
    values
}

mod sse4a {
    use super::*;

    /// The field of `length` bits (0 means 64) at `index`, both 6-bit, bit by bit.
    fn field_mask(length: u64, index: u64) -> u64 {
        let length = if length & 63 == 0 { 64 } else { length & 63 };
        (0..length)
            .map(|k| (index & 63) + k)
            .filter(|&bit| bit < 64)
            .fold(0, |mask, bit| mask | 1 << bit)
    }

    fn extract(value: u64, length: u64, index: u64) -> u64 {
        (value & field_mask(length, index)) >> (index & 63)
    }

    fn insert(destination: u64, source: u64, length: u64, index: u64) -> u64 {
        let mask = field_mask(length, index);
        (destination & !mask) | ((source << (index & 63)) & mask)
    }

    const UPPER: u128 = 0xaaaa_bbbb_cccc_dddd << 64;

    #[test]
    fn extrq_with_immediates_moves_the_field_down_and_keeps_the_upper_quadword() {
        for (length, index) in [(8, 4), (1, 63), (0, 0), (16, 48), (63, 1), (32, 32)] {
            for &value in &samples() {
                // extrq xmm3, length, index (66 0F 78 /0 with xmm3 in ModRM.rm)
                let code = snippet(&[0x66, 0x0f, 0x78, 0xc3, length, index]);
                let (h, stop) = run(&code, |h| h.cpu.xmm[3] = UPPER | value as u128);
                assert_eq!(stop, Stop::Halted);
                assert_eq!(
                    h.cpu.xmm[3],
                    UPPER | extract(value, length as u64, index as u64) as u128,
                    "extrq {length},{index} of {value:#x}"
                );
            }
        }
    }

    #[test]
    fn extrq_from_a_register_takes_the_field_from_bits_5_0_and_13_8() {
        // extrq xmm1, xmm2
        let code = snippet(&[0x66, 0x0f, 0x79, 0xca]);
        for (length, index) in [(12u64, 20u64), (0, 0), (64 + 3, 7), (5, 64 + 2)] {
            let value = 0x0123_4567_89ab_cdefu64;
            let (h, stop) = run(&code, |h| {
                h.cpu.xmm[1] = UPPER | value as u128;
                // garbage above the fields is ignored
                h.cpu.xmm[2] = (u128::MAX << 16) | (index << 8 | length) as u128 & 0xffff;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.xmm[1], UPPER | extract(value, length, index) as u128);
        }
    }

    #[test]
    fn insertq_with_immediates_puts_the_low_bits_of_the_source_in_the_field() {
        for (length, index) in [(8, 4), (1, 63), (0, 0), (16, 48), (40, 8)] {
            for &source in &samples()[..8] {
                // insertq xmm1, xmm2, length, index
                let code = snippet(&[0xf2, 0x0f, 0x78, 0xca, length, index]);
                let destination = 0x5555_5555_5555_5555u64;
                let (h, stop) = run(&code, |h| {
                    h.cpu.xmm[1] = UPPER | destination as u128;
                    h.cpu.xmm[2] = u128::MAX << 64 | source as u128;
                });
                assert_eq!(stop, Stop::Halted);
                assert_eq!(
                    h.cpu.xmm[1],
                    UPPER | insert(destination, source, length as u64, index as u64) as u128,
                    "insertq {length},{index} of {source:#x}"
                );
            }
        }
    }

    #[test]
    fn insertq_from_a_register_takes_the_field_from_the_upper_quadword() {
        // insertq xmm1, xmm2
        let code = snippet(&[0xf2, 0x0f, 0x79, 0xca]);
        let (h, stop) = run(&code, |h| {
            h.cpu.xmm[1] = UPPER;
            // length 12 at bit 72-64 = 8 of the upper quadword, index 20
            h.cpu.xmm[2] = (20u128 << 8 | 12) << 64 | 0xfff_fabc;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.xmm[1], UPPER | (0xabc << 20));
    }

    #[test]
    fn memory_forms_and_the_other_prefixes_are_ud() {
        for code in [
            &[0x66, 0x0f, 0x78, 0x06, 8, 4][..], // extrq with a memory operand
            &[0x66, 0x0f, 0x79, 0x06],           // extrq xmm0, [rsi]
            &[0xf2, 0x0f, 0x78, 0x06, 8, 4],     // insertq with a memory operand
            &[0xf2, 0x0f, 0x79, 0x06],
            &[0x66, 0x0f, 0x78, 0xcb, 8, 4], // extrq immediate needs /0
            &[0x0f, 0x78, 0xc1],             // vmread (Intel): not on AMD
            &[0x0f, 0x79, 0xc1],             // vmwrite
            &[0xf3, 0x0f, 0x78, 0xc1],
            &[0xf3, 0x0f, 0x2b, 0xc1], // movntss to a register
            &[0xf2, 0x0f, 0x2b, 0xc1],
        ] {
            let (_, stop) = run(&snippet(code), |_| {});
            assert_eq!(stop, ud(), "{code:02x?}");
        }
    }

    #[test]
    fn the_extrq_immediates_are_two_bytes_long() {
        let insn = crate::cpu::decode::decode(&[0x66, 0x0f, 0x78, 0xc3, 0x08, 0x04]).unwrap();
        assert_eq!((insn.len, insn.imm, insn.imm2), (6, 8, 4));
        let insn = crate::cpu::decode::decode(&[0xf2, 0x0f, 0x78, 0xca, 0x10, 0x20]).unwrap();
        assert_eq!((insn.len, insn.imm, insn.imm2), (6, 0x10, 0x20));
        // without the prefix there is no immediate (vmread r/m, r)
        assert_eq!(
            crate::cpu::decode::decode(&[0x0f, 0x78, 0xc3]).unwrap().len,
            3
        );
    }

    #[test]
    fn movntss_and_movntsd_store_the_low_lane_without_an_alignment_rule() {
        let value = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00u128;
        // movntss [rsi], xmm1
        let (h, stop) = run(&snippet(&[0xf3, 0x0f, 0x2b, 0x0e]), |h| {
            h.cpu.xmm[1] = value;
            h.cpu.gpr[RSI] = DATA + 3;
            h.write(DATA, &[0xcc; 16]);
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(
            h.read(DATA, 12),
            [
                0xcc, 0xcc, 0xcc, 0x00, 0xff, 0xee, 0xdd, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc
            ]
        );
        // movntsd [rsi], xmm1
        let (h, stop) = run(&snippet(&[0xf2, 0x0f, 0x2b, 0x0e]), |h| {
            h.cpu.xmm[1] = value;
            h.cpu.gpr[RSI] = DATA + 5;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.read(DATA + 5, 8), 0x99aa_bbcc_ddee_ff00u64.to_le_bytes());
    }

    #[test]
    fn they_follow_the_sse_enable_rules() {
        for code in [&[0xf2, 0x0f, 0x79, 0xca][..], &[0xf3, 0x0f, 0x2b, 0x0e]] {
            let (_, stop) = run(&snippet(code), |h| h.cpu.cr4 &= !CR4_OSFXSR_BIT);
            assert_eq!(stop, ud());
            let (_, stop) = run(&snippet(code), |h| h.cpu.cr0 |= CR0_TS_BIT);
            assert_eq!(stop, fault(VECTOR_DEVICE_NOT_AVAILABLE, None));
        }
    }

    #[test]
    fn the_jit_matches_the_interpreter() {
        // extrq xmm3, 8, 4; insertq xmm1, xmm2, 16, 40; extrq xmm1, xmm2; insertq xmm4, xmm2;
        // movntss [rsi], xmm1; movntsd [rsi+8], xmm3; hlt
        let code = [
            0x66, 0x0f, 0x78, 0xc3, 0x08, 0x04, 0xf2, 0x0f, 0x78, 0xca, 0x10, 0x28, 0x66, 0x0f,
            0x79, 0xca, 0xf2, 0x0f, 0x79, 0xe2, 0xf3, 0x0f, 0x2b, 0x0e, 0xf2, 0x0f, 0x2b, 0x5e,
            0x08, HLT,
        ];
        assert_jit_matches_interpreter(&code, |h| {
            sse_on(h);
            for (k, value) in samples().into_iter().enumerate().take(8) {
                h.cpu.xmm[k] = (value as u128) << 64 | value.rotate_left(17) as u128;
            }
        });
    }
}

mod misaligned_sse {
    use super::*;

    // paddb xmm0, [rsi]; movaps xmm0, [rsi]; movdqa xmm0, [rsi]; movntps [rsi], xmm0;
    // pshufb xmm0, [rsi]; movntdqa xmm0, [rsi]; ldmxcsr [rsi]
    const PADDB: &[u8] = &[0x66, 0x0f, 0xfc, 0x06];
    const MOVAPS: &[u8] = &[0x0f, 0x28, 0x06];
    const MOVDQA: &[u8] = &[0x66, 0x0f, 0x6f, 0x06];
    const MOVNTPS: &[u8] = &[0x0f, 0x2b, 0x06];
    const MOVNTDQ: &[u8] = &[0x66, 0x0f, 0xe7, 0x06];
    const PSHUFB: &[u8] = &[0x66, 0x0f, 0x38, 0x00, 0x06];
    const MOVNTDQA: &[u8] = &[0x66, 0x0f, 0x38, 0x2a, 0x06];
    const LDMXCSR: &[u8] = &[0x0f, 0xae, 0x16];

    fn misaligned(code: &[u8], mxcsr: u32) -> Stop {
        run(&snippet(code), |h| {
            h.cpu.mxcsr = mxcsr;
            h.cpu.gpr[RSI] = DATA + 1;
        })
        .1
    }

    #[test]
    fn mxcsr_mm_lifts_the_alignment_rule_of_ordinary_operands() {
        for code in [PADDB, PSHUFB] {
            assert_eq!(misaligned(code, MXCSR_DEFAULT), gp0(), "{code:02x?}");
            assert_eq!(
                misaligned(code, MXCSR_DEFAULT | MXCSR_MM),
                Stop::Halted,
                "{code:02x?}"
            );
        }
    }

    #[test]
    fn the_explicitly_aligned_moves_still_fault() {
        for code in [MOVAPS, MOVDQA, MOVNTPS, MOVNTDQ, MOVNTDQA] {
            assert_eq!(
                misaligned(code, MXCSR_DEFAULT | MXCSR_MM),
                gp0(),
                "{code:02x?}"
            );
        }
    }

    #[test]
    fn the_misaligned_operand_is_read_whole() {
        let (h, stop) = run(&snippet(PADDB), |h| {
            h.cpu.mxcsr = MXCSR_DEFAULT | MXCSR_MM;
            h.cpu.gpr[RSI] = DATA + 7;
            h.write(DATA + 7, &(1..=16).collect::<Vec<u8>>());
            h.cpu.xmm[0] = u128::from_le_bytes([0x10; 16]);
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(
            h.cpu.xmm[0].to_le_bytes(),
            core::array::from_fn::<u8, 16, _>(|k| 0x11 + k as u8)
        );
    }

    #[test]
    fn ldmxcsr_and_fxsave_know_the_mm_bit() {
        let (h, stop) = run(&snippet(LDMXCSR), |h| {
            h.write(DATA, &(MXCSR_DEFAULT | MXCSR_MM).to_le_bytes())
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.mxcsr, MXCSR_DEFAULT | MXCSR_MM);
        // bit 16 and 18 are still reserved
        for bit in [16, 18] {
            let (_, stop) = run(&snippet(LDMXCSR), |h| {
                h.write(DATA, &(MXCSR_DEFAULT | 1 << bit).to_le_bytes())
            });
            assert_eq!(stop, gp0(), "bit {bit}");
        }
        // fxsave [rsi]: MXCSR_MASK tells the OS the bit exists
        let (h, stop) = run(&snippet(&[0x0f, 0xae, 0x06]), |_| {});
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.read(DATA + 28, 4), 0x2_ffffu32.to_le_bytes());
    }

    #[test]
    fn the_jit_leaves_a_misaligned_operand_to_the_interpreter() {
        // paddb xmm0, [rsi]; movdqu xmm1, xmm0; pcmpeqb xmm1, [rsi+16]; movaps xmm2, [rsi]; hlt
        let code = [
            0x66, 0x0f, 0xfc, 0x06, 0xf3, 0x0f, 0x6f, 0xc8, 0x66, 0x0f, 0x74, 0x4e, 0x10, 0x0f,
            0x28, 0x16, HLT,
        ];
        for mxcsr in [MXCSR_DEFAULT, MXCSR_DEFAULT | MXCSR_MM] {
            assert_jit_matches_interpreter(&code, |h| {
                sse_on(h);
                h.cpu.mxcsr = mxcsr;
                h.cpu.gpr[RSI] = DATA + 1;
                h.write(DATA, &(0..64).collect::<Vec<u8>>());
            });
        }
    }
}

mod ssse3_on_mmx {
    use super::*;

    fn bytes(value: u64) -> [u8; 8] {
        value.to_le_bytes()
    }

    fn words(value: u64) -> [i16; 4] {
        core::array::from_fn(|k| (value >> (16 * k)) as i16)
    }

    fn dwords(value: u64) -> [i32; 2] {
        [value as i32, (value >> 32) as i32]
    }

    fn from_bytes(lanes: [u8; 8]) -> u64 {
        u64::from_le_bytes(lanes)
    }

    fn from_words(lanes: [i16; 4]) -> u64 {
        lanes
            .iter()
            .enumerate()
            .fold(0, |r, (k, &w)| r | (w as u16 as u64) << (16 * k))
    }

    fn from_dwords(lanes: [i32; 2]) -> u64 {
        lanes[0] as u32 as u64 | (lanes[1] as u32 as u64) << 32
    }

    fn sign<T: Copy + Default + PartialOrd + core::ops::Neg<Output = T>>(a: T, b: T) -> T {
        if b < T::default() {
            -a
        } else if b == T::default() {
            T::default()
        } else {
            a
        }
    }

    type Model = fn(u64, u64) -> u64;

    /// (0F 38 opcode, name, model of `dst = f(dst, src)`).
    fn models() -> Vec<(u8, &'static str, Model)> {
        vec![
            (0x00, "pshufb", |a, b| {
                let (a, b) = (bytes(a), bytes(b));
                from_bytes(core::array::from_fn(|k| {
                    if b[k] & 0x80 != 0 {
                        0
                    } else {
                        a[(b[k] & 7) as usize]
                    }
                }))
            }),
            (0x01, "phaddw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words([
                    a[0].wrapping_add(a[1]),
                    a[2].wrapping_add(a[3]),
                    b[0].wrapping_add(b[1]),
                    b[2].wrapping_add(b[3]),
                ])
            }),
            (0x02, "phaddd", |a, b| {
                let (a, b) = (dwords(a), dwords(b));
                from_dwords([a[0].wrapping_add(a[1]), b[0].wrapping_add(b[1])])
            }),
            (0x03, "phaddsw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words([
                    a[0].saturating_add(a[1]),
                    a[2].saturating_add(a[3]),
                    b[0].saturating_add(b[1]),
                    b[2].saturating_add(b[3]),
                ])
            }),
            (0x04, "pmaddubsw", |a, b| {
                let (a, b) = (bytes(a), bytes(b));
                from_words(core::array::from_fn(|k| {
                    let sum = a[2 * k] as i32 * b[2 * k] as i8 as i32
                        + a[2 * k + 1] as i32 * b[2 * k + 1] as i8 as i32;
                    sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16
                }))
            }),
            (0x05, "phsubw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words([
                    a[0].wrapping_sub(a[1]),
                    a[2].wrapping_sub(a[3]),
                    b[0].wrapping_sub(b[1]),
                    b[2].wrapping_sub(b[3]),
                ])
            }),
            (0x06, "phsubd", |a, b| {
                let (a, b) = (dwords(a), dwords(b));
                from_dwords([a[0].wrapping_sub(a[1]), b[0].wrapping_sub(b[1])])
            }),
            (0x07, "phsubsw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words([
                    a[0].saturating_sub(a[1]),
                    a[2].saturating_sub(a[3]),
                    b[0].saturating_sub(b[1]),
                    b[2].saturating_sub(b[3]),
                ])
            }),
            (0x08, "psignb", |a, b| {
                let (a, b) = (bytes(a), bytes(b));
                from_bytes(core::array::from_fn(|k| {
                    sign(a[k] as i8 as i16, b[k] as i8 as i16) as u8
                }))
            }),
            (0x09, "psignw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words(core::array::from_fn(|k| {
                    sign(a[k] as i32, b[k] as i32) as i16
                }))
            }),
            (0x0a, "psignd", |a, b| {
                let (a, b) = (dwords(a), dwords(b));
                from_dwords(core::array::from_fn(|k| {
                    sign(a[k] as i64, b[k] as i64) as i32
                }))
            }),
            (0x0b, "pmulhrsw", |a, b| {
                let (a, b) = (words(a), words(b));
                from_words(core::array::from_fn(|k| {
                    ((((a[k] as i32 * b[k] as i32) >> 14) + 1) >> 1) as i16
                }))
            }),
            (0x1c, "pabsb", |_, b| {
                from_bytes(bytes(b).map(|x| (x as i8).unsigned_abs()))
            }),
            (0x1d, "pabsw", |_, b| {
                from_words(words(b).map(|x| x.unsigned_abs() as i16))
            }),
            (0x1e, "pabsd", |_, b| {
                from_dwords(dwords(b).map(|x| x.unsigned_abs() as i32))
            }),
        ]
    }

    fn mm(h: &Harness, index: usize) -> u64 {
        u64::from_le_bytes(h.cpu.st[index][..8].try_into().unwrap())
    }

    fn set_mm(h: &mut Harness, index: usize, value: u64) {
        h.cpu.st[index][..8].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn every_operation_matches_its_model_in_register_and_memory_form() {
        let values = samples();
        for (opcode, name, model) in models() {
            // op mm0, mm1 and op mm2, [rsi]
            let register = snippet(&[0x0f, 0x38, opcode, 0xc1]);
            let memory = snippet(&[0x0f, 0x38, opcode, 0x16]);
            for (k, &a) in values.iter().enumerate() {
                let b = values[(k * 7 + 3) % values.len()];
                let want = model(a, b);
                let (h, stop) = run(&register, |h| {
                    set_mm(h, 0, a);
                    set_mm(h, 1, b);
                });
                assert_eq!(stop, Stop::Halted, "{name}");
                assert_eq!(mm(&h, 0), want, "{name} {a:#x} {b:#x}");
                assert_eq!(mm(&h, 1), b, "{name}: the source is untouched");
                let (h, stop) = run(&memory, |h| {
                    set_mm(h, 2, a);
                    // no alignment rule for an m64
                    h.cpu.gpr[RSI] = DATA + 3;
                    h.write(DATA + 3, &b.to_le_bytes());
                });
                assert_eq!(stop, Stop::Halted, "{name} memory");
                assert_eq!(mm(&h, 2), want, "{name} memory {a:#x} {b:#x}");
            }
        }
    }

    #[test]
    fn palignr_shifts_the_16_byte_concatenation_right_by_bytes() {
        for count in [0u8, 1, 3, 7, 8, 9, 15, 16, 200] {
            // palignr mm0, mm1, count
            let code = snippet(&[0x0f, 0x3a, 0x0f, 0xc1, count]);
            let (dst, src) = (0x1716_1514_1312_1110u64, 0x0706_0504_0302_0100u64);
            let (h, stop) = run(&code, |h| {
                set_mm(h, 0, dst);
                set_mm(h, 1, src);
            });
            assert_eq!(stop, Stop::Halted);
            // src is bytes 0-7, dst bytes 8-15 of the joined value; beyond that zeros
            let mut joined = [0u8; 16];
            joined[..8].copy_from_slice(&src.to_le_bytes());
            joined[8..].copy_from_slice(&dst.to_le_bytes());
            let want = from_bytes(core::array::from_fn(|k| {
                joined.get(k + count as usize).copied().unwrap_or(0)
            }));
            assert_eq!(mm(&h, 0), want, "palignr {count}");
        }
    }

    #[test]
    fn the_mmx_rules_apply_not_the_sse_ones() {
        let code = snippet(&[0x0f, 0x38, 0x00, 0xc1]);
        // CR4.OSFXSR does not matter
        let (h, stop) = run(&code, |h| {
            h.cpu.cr4 &= !CR4_OSFXSR_BIT;
            h.cpu.fsw = 5 << 11;
            h.cpu.ftw = 0;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.fsw & 7 << 11, 0, "TOP is 0");
        assert_eq!(h.cpu.ftw, 0xff, "every tag valid");
        assert_eq!(
            h.cpu.st[0][8..],
            [0xff, 0xff],
            "the exponent of mm0 is all ones"
        );
        let (_, stop) = run(&code, |h| h.cpu.cr0 |= CR0_EM_BIT);
        assert_eq!(stop, ud());
        let (_, stop) = run(&code, |h| h.cpu.cr0 |= CR0_TS_BIT);
        assert_eq!(stop, fault(VECTOR_DEVICE_NOT_AVAILABLE, None));
        // F2/F3 forms are still nothing
        let (_, stop) = run(&snippet(&[0xf3, 0x0f, 0x38, 0x00, 0xc1]), |_| {});
        assert_eq!(stop, ud());
    }

    #[test]
    fn a_memory_fault_changes_nothing() {
        let code = snippet(&[0x0f, 0x38, 0x01, 0x06]);
        let (h, stop) = run(&code, |h| {
            h.cpu.gpr[RSI] = 0x4000_0000_0000;
            h.cpu.fsw = 3 << 11;
        });
        assert!(matches!(stop, Stop::Fault { .. }), "{stop:?}");
        assert_eq!(h.cpu.fsw, 3 << 11);
        assert_eq!(h.cpu.rip, CODE_BASE);
    }

    #[test]
    fn the_jit_matches_the_interpreter() {
        let mut code = Vec::new();
        for (opcode, _, _) in models() {
            // op mm0, mm1; op mm2, [rsi]
            code.extend([0x0f, 0x38, opcode, 0xc1, 0x0f, 0x38, opcode, 0x16]);
        }
        // palignr mm3, mm0, 5; hlt
        code.extend([0x0f, 0x3a, 0x0f, 0xd8, 0x05, HLT]);
        assert_jit_matches_interpreter(&code, |h| {
            sse_on(h);
            for (k, value) in samples().into_iter().enumerate().take(8) {
                h.cpu.st[k][..8].copy_from_slice(&value.to_le_bytes());
            }
            h.write(DATA, &0x8081_7f00_0102_fffeu64.to_le_bytes());
        });
    }
}
