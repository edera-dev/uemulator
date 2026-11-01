//! Reference-model tests for the instructions in `avx_special.rs`. Encodings come from the
//! assembler (`avx_special_encodings.rs`); each expectation is a plain loop over bytes written
//! from the SDM description, independent of the implementation.

// The predicate table spells out the SDM's negated comparisons (`!(a < b)` is NLT, not GE: they
// differ for NaN); the half-precision samples are deliberately written out beyond f32 precision;
// the gather reference takes every parameter the instruction has.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::excessive_precision,
    clippy::too_many_arguments
)]

use crate::cpu::sys::CR4_OSXSAVE;
use crate::cpu::xsave::{XCR0_AVX, XCR0_SSE, XCR0_X87};
use crate::testkit::{Harness, RDI, RSI, SCRATCH, Stop};

// not every generated encoding is used by a test
#[allow(dead_code)]
#[path = "avx_special_encodings.rs"]
mod encodings;
use encodings::*;

const DATA: u64 = SCRATCH + 0x1000;
const OUT: u64 = SCRATCH + 0x2000;
const ZF: u64 = 1 << 6;
const CF: u64 = 1;

type Ymm = [u8; 32];
/// (ymm form, xmm form, memory form, kind, element bytes)
type ShiftCase = (
    &'static [u8],
    &'static [u8],
    &'static [u8],
    &'static str,
    usize,
);
/// (ymm form, xmm form, kind, element bytes, count)
type ImmediateShiftCase = (&'static [u8], &'static [u8], &'static str, usize, u64);
/// (ymm form, xmm form, memory form, source bytes, destination bytes, signed)
type WidenCase = (
    &'static [u8],
    &'static [u8],
    &'static [u8],
    usize,
    usize,
    bool,
);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn ymm(&mut self) -> Ymm {
        core::array::from_fn(|_| self.next() as u8)
    }
}

fn ymm_of(low: u128, high: u128) -> Ymm {
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&low.to_le_bytes());
    bytes[16..].copy_from_slice(&high.to_le_bytes());
    bytes
}

fn harness(code: &[u8]) -> Harness {
    let mut h = Harness::new(code);
    h.cpu.cr4 = (h.cpu.cr4 & !(1 << 9)) | CR4_OSXSAVE;
    h.cpu.xcr0 = XCR0_X87 | XCR0_SSE | XCR0_AVX;
    h.cpu.gpr[RSI] = DATA;
    h.cpu.gpr[RDI] = OUT;
    h
}

fn set(h: &mut Harness, register: usize, value: Ymm) {
    h.cpu.xmm[register] = u128::from_le_bytes(value[..16].try_into().unwrap());
    h.cpu.ymm_hi[register] = u128::from_le_bytes(value[16..].try_into().unwrap());
}

fn get(h: &Harness, register: usize) -> Ymm {
    ymm_of(h.cpu.xmm[register], h.cpu.ymm_hi[register])
}

fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
    let mut h = harness(code);
    setup(&mut h);
    let stop = h.run(100);
    assert_eq!(stop, Stop::Halted, "{code:02x?}: {}", h.cpu.dump_regs());
    h
}

fn dword(v: &Ymm, k: usize) -> u32 {
    u32::from_le_bytes(v[4 * k..4 * k + 4].try_into().unwrap())
}

fn qword(v: &Ymm, k: usize) -> u64 {
    u64::from_le_bytes(v[8 * k..8 * k + 8].try_into().unwrap())
}

fn from_dwords(d: &[u32]) -> Ymm {
    let mut out = [0u8; 32];
    for (k, v) in d.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn from_qwords(d: &[u64]) -> Ymm {
    let mut out = [0u8; 32];
    for (k, v) in d.iter().enumerate() {
        out[8 * k..8 * k + 8].copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn zero_upper(mut v: Ymm) -> Ymm {
    v[16..].fill(0);
    v
}

fn inputs() -> Vec<(Ymm, Ymm)> {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    (0..16).map(|_| (rng.ymm(), rng.ymm())).collect()
}

#[test]
fn movmsk_collects_sign_bits() {
    for (a, _) in inputs() {
        let bits = |width: usize, count: usize| -> u64 {
            (0..count).fold(0, |m, k| m | ((a[k * width + width - 1] >> 7) as u64) << k)
        };
        for (code, width, count) in [
            (MOVMSKPS_Y, 4, 8),
            (MOVMSKPS_X, 4, 4),
            (MOVMSKPD_Y, 8, 4),
            (MOVMSKPD_X, 8, 2),
            (PMOVMSKB_Y, 1, 32),
            (PMOVMSKB_X, 1, 16),
        ] {
            let h = run(code, |h| {
                set(h, 1, a);
                h.cpu.gpr[0] = u64::MAX;
            });
            assert_eq!(h.cpu.gpr[0], bits(width, count), "{code:02x?}");
        }
    }
}

#[test]
fn widening_conversions_cover_the_whole_destination() {
    let singles: [f32; 4] = [1.5, -2.25, 1e10, 0.125];
    let source = from_dwords(&singles.map(f32::to_bits));
    let h = run(CVTPS2PD_Y, |h| set(h, 1, source));
    let doubles: Vec<u64> = singles.iter().map(|&s| (s as f64).to_bits()).collect();
    assert_eq!(get(&h, 0), from_qwords(&doubles));
    // the 128-bit form converts two and zeroes the upper half
    let h = run(CVTPS2PD_X, |h| {
        set(h, 1, source);
        set(h, 0, [0xaa; 32]);
    });
    assert_eq!(get(&h, 0), from_qwords(&doubles[..2]));
    // memory source: 16 bytes for the 256-bit form, 8 for the 128-bit one
    let h = run(CVTPS2PD_YM, |h| h.write(DATA, &source[..16]));
    assert_eq!(get(&h, 0), from_qwords(&doubles));
    let h = run(CVTPS2PD_XM, |h| h.write(DATA, &source[..8]));
    assert_eq!(get(&h, 0), from_qwords(&doubles[..2]));
    let ints = [-7i32, 3, i32::MAX, i32::MIN];
    let h = run(CVTDQ2PD_Y, |h| {
        set(h, 1, from_dwords(&ints.map(|v| v as u32)))
    });
    assert_eq!(get(&h, 0), from_qwords(&ints.map(|v| (v as f64).to_bits())));
}

#[test]
fn narrowing_conversions_fill_an_xmm_and_zero_the_rest() {
    let doubles = [1.5f64, -2.75, 123456789.125, 1e30];
    let source = from_qwords(&doubles.map(f64::to_bits));
    let h = run(CVTPD2PS_Y, |h| {
        set(h, 1, source);
        set(h, 0, [0xaa; 32]);
    });
    let expected = from_dwords(&doubles.map(|d| (d as f32).to_bits()));
    assert_eq!(
        get(&h, 0),
        expected,
        "ymm source -> 4 singles, upper half cleared"
    );
    let h = run(CVTPD2PS_X, |h| set(h, 1, source));
    assert_eq!(
        get(&h, 0),
        from_dwords(&[(1.5f32).to_bits(), (-2.75f32).to_bits()])
    );
    let values = [2.5f64, -3.5, 100.75, -0.25];
    let source = from_qwords(&values.map(f64::to_bits));
    let h = run(CVTTPD2DQ_Y, |h| set(h, 1, source));
    assert_eq!(
        get(&h, 0),
        from_dwords(&values.map(|d| d.trunc() as i32 as u32))
    );
    let h = run(CVTPD2DQ_Y, |h| set(h, 1, source));
    assert_eq!(
        get(&h, 0),
        from_dwords(&values.map(|d| d.round_ties_even() as i32 as u32))
    );
}

fn shift_model(kind: &str, width: usize, value: u64, count: u64) -> u64 {
    let bits = width * 8;
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let value = value & mask;
    match kind {
        "sll" => {
            if count >= bits as u64 {
                0
            } else {
                (value << count) & mask
            }
        }
        "srl" => {
            if count >= bits as u64 {
                0
            } else {
                value >> count
            }
        }
        _ => {
            // arithmetic: sign-fill, count saturates at width - 1
            let signed = ((value << (64 - bits)) as i64) >> (64 - bits);
            ((signed >> count.min(bits as u64 - 1)) as u64) & mask
        }
    }
}

fn apply_shift(v: &Ymm, kind: &str, width: usize, count: u64, lanes: usize) -> Ymm {
    let mut out = [0u8; 32];
    for k in 0..lanes * 16 / width {
        let mut element = 0u64;
        for b in 0..width {
            element |= (v[k * width + b] as u64) << (8 * b);
        }
        let r = shift_model(kind, width, element, count);
        for b in 0..width {
            out[k * width + b] = (r >> (8 * b)) as u8;
        }
    }
    out
}

#[test]
fn shifts_by_register_count_use_the_low_quadword_for_both_lanes() {
    let table: [ShiftCase; 8] = [
        (SLLW_Y, SLLW_X, SLLW_YM, "sll", 2),
        (SRLW_Y, SRLW_X, SRLW_YM, "srl", 2),
        (SRAW_Y, SRAW_X, SRAW_YM, "sra", 2),
        (SLLD_Y, SLLD_X, SLLD_YM, "sll", 4),
        (SRLD_Y, SRLD_X, SRLD_YM, "srl", 4),
        (SRAD_Y, SRAD_X, SRAD_YM, "sra", 4),
        (SLLQ_Y, SLLQ_X, SLLQ_YM, "sll", 8),
        (SRLQ_Y, SRLQ_X, SRLQ_YM, "srl", 8),
    ];
    for (wide, narrow, memory, kind, width) in table {
        for (value, count_source) in inputs() {
            for count in [0u64, 1, 3, 15, 16, 31, 32, 63, 64, 200, u64::MAX] {
                let mut count_register = count_source;
                count_register[..8].copy_from_slice(&count.to_le_bytes());
                let h = run(wide, |h| {
                    set(h, 1, value);
                    set(h, 2, count_register);
                });
                assert_eq!(
                    get(&h, 0),
                    apply_shift(&value, kind, width, count, 2),
                    "{kind}{width} {count}"
                );
                let h = run(narrow, |h| {
                    set(h, 1, value);
                    set(h, 2, count_register);
                });
                assert_eq!(
                    get(&h, 0),
                    zero_upper(apply_shift(&value, kind, width, count, 1)),
                    "{kind}{width} xmm {count}"
                );
                // a memory count is 16 bytes at rsi, not 32
                let h = run(memory, |h| {
                    set(h, 1, value);
                    h.write(DATA, &count_register[..16]);
                    h.write(DATA + 16, &[0xff; 16]);
                });
                assert_eq!(get(&h, 0), apply_shift(&value, kind, width, count, 2));
            }
        }
    }
}

#[test]
fn shifts_by_immediate_write_vvvv_from_rm() {
    let table: [ImmediateShiftCase; 5] = [
        (SLLD_I_Y, SLLD_I_X, "sll", 4, 3),
        (SRLQ_I_Y, SRLQ_I_X, "srl", 8, 7),
        (SRAW_I_Y, SRAW_I_X, "sra", 2, 20),
        (SLLDQ_I_Y, SLLDQ_I_X, "bytes-left", 16, 4),
        (SRLDQ_I_Y, SRLDQ_I_X, "bytes-right", 16, 3),
    ];
    for (wide, narrow, kind, width, count) in table {
        for (value, _) in inputs() {
            let expect = |lanes: usize| -> Ymm {
                match kind {
                    "bytes-left" | "bytes-right" => {
                        let mut out = [0u8; 32];
                        for lane in 0..lanes {
                            for b in 0..16usize {
                                let source = if kind == "bytes-left" {
                                    b.checked_sub(count as usize)
                                } else {
                                    Some(b + count as usize).filter(|s| *s < 16)
                                };
                                if let Some(s) = source {
                                    out[lane * 16 + b] = value[lane * 16 + s];
                                }
                            }
                        }
                        out
                    }
                    _ => apply_shift(&value, kind, width, count, lanes),
                }
            };
            // vpslld ymm0, ymm1, imm: destination is vvvv (= ymm0), source is rm (= ymm1)
            let h = run(wide, |h| set(h, 1, value));
            assert_eq!(get(&h, 0), expect(2), "{kind} ymm");
            let h = run(narrow, |h| set(h, 1, value));
            assert_eq!(get(&h, 0), zero_upper(expect(1)), "{kind} xmm");
        }
    }
}

#[test]
fn ptest_and_vtest_combine_both_lanes() {
    for (a, b) in inputs().into_iter().chain([
        ([0; 32], [0; 32]),
        ([0xff; 32], [0xff; 32]),
        ([0xff; 32], [0; 32]),
    ]) {
        let flags = |h: &Harness| h.cpu.eflags_full() & (ZF | CF);
        let zf = |x: &[u8], y: &[u8], mask: fn(&[u8]) -> Vec<u8>| {
            mask(x).iter().zip(mask(y)).all(|(p, q)| p & q == 0)
        };
        let identity = |v: &[u8]| v.to_vec();
        let nots = |v: &[u8]| v.iter().map(|b| !b).collect::<Vec<u8>>();
        // ptest dst=ymm0, src=ymm1: ZF = (dst & src) == 0, CF = (!dst & src) == 0
        let h = run(PTEST_Y, |h| {
            set(h, 0, a);
            set(h, 1, b);
        });
        let expected = (if zf(&a, &b, identity) { ZF } else { 0 })
            | (if b.iter().zip(&a).all(|(s, d)| !d & s == 0) {
                CF
            } else {
                0
            });
        let _ = nots;
        assert_eq!(flags(&h), expected, "vptest ymm {a:02x?} {b:02x?}");
        let h = run(PTEST_X, |h| {
            set(h, 0, a);
            set(h, 1, b);
        });
        let expected = (if zf(&a[..16], &b[..16], identity) {
            ZF
        } else {
            0
        }) | (if b[..16].iter().zip(&a[..16]).all(|(s, d)| !d & s == 0) {
            CF
        } else {
            0
        });
        assert_eq!(flags(&h), expected, "vptest xmm");
        // vtestps looks at the dword sign bits only
        let sign_zero = |x: &Ymm, y: &Ymm, negate: bool, lanes: usize, width: usize| {
            (0..lanes * 16 / width).all(|k| {
                let (xs, ys) = (x[k * width + width - 1] >> 7, y[k * width + width - 1] >> 7);
                (if negate { (1 - xs) & ys } else { xs & ys }) == 0
            })
        };
        for (code, width, lanes) in [(TESTPS_Y, 4, 2), (TESTPD_Y, 8, 2), (TESTPD_X, 8, 1)] {
            let h = run(code, |h| {
                set(h, 0, a);
                set(h, 1, b);
            });
            let expected = (if sign_zero(&a, &b, false, lanes, width) {
                ZF
            } else {
                0
            }) | (if sign_zero(&a, &b, true, lanes, width) {
                CF
            } else {
                0
            });
            assert_eq!(flags(&h), expected, "vtest width {width} lanes {lanes}");
        }
    }
}

#[test]
fn in_lane_permutes_follow_their_selectors() {
    for (v, selector) in inputs() {
        // vpermilps ymm0, ymm1, ymm2: dword k takes dword (sel[k] & 3) of its own lane
        let h = run(PERMILPS_VAR_Y, |h| {
            set(h, 1, v);
            set(h, 2, selector);
        });
        let expected: Vec<u32> = (0..8)
            .map(|k| dword(&v, k / 4 * 4 + (dword(&selector, k) & 3) as usize))
            .collect();
        assert_eq!(get(&h, 0), from_dwords(&expected));
        let h = run(PERMILPS_VAR_X, |h| {
            set(h, 1, v);
            set(h, 2, selector);
        });
        assert_eq!(get(&h, 0), zero_upper(from_dwords(&expected[..4])));
        // vpermilpd: selector bit 1 of each qword
        let h = run(PERMILPD_VAR_Y, |h| {
            set(h, 1, v);
            set(h, 2, selector);
        });
        let expected: Vec<u64> = (0..4)
            .map(|k| qword(&v, k / 2 * 2 + ((qword(&selector, k) >> 1) & 1) as usize))
            .collect();
        assert_eq!(get(&h, 0), from_qwords(&expected));
        // immediates: 0x1b reverses the dwords of each lane; pd imm 0b0101 swaps within each lane
        let h = run(PERMILPS_IMM_Y, |h| set(h, 1, v));
        let expected: Vec<u32> = (0..8).map(|k| dword(&v, k / 4 * 4 + (3 - k % 4))).collect();
        assert_eq!(get(&h, 0), from_dwords(&expected));
        let h = run(PERMILPD_IMM_Y, |h| set(h, 1, v));
        let expected: Vec<u64> = (0..4)
            .map(|k| {
                let bit = (0x5 >> k) & 1;
                qword(&v, k / 2 * 2 + bit)
            })
            .collect();
        assert_eq!(get(&h, 0), from_qwords(&expected));
        let h = run(PERMILPD_IMM_X, |h| set(h, 1, v));
        assert_eq!(
            get(&h, 0),
            zero_upper(from_qwords(
                &[qword(&v, 1), qword(&v, 0)][..1]
                    .to_vec()
                    .into_iter()
                    .chain([qword(&v, 0)])
                    .collect::<Vec<_>>()
            ))
        );
    }
}

#[test]
fn cross_lane_permutes() {
    for (v, selector) in inputs() {
        // vpermq imm 0x1b reverses the four qwords; vpermpd imm 0xc6
        let h = run(PERMQ, |h| set(h, 1, v));
        assert_eq!(
            get(&h, 0),
            from_qwords(&[qword(&v, 3), qword(&v, 2), qword(&v, 1), qword(&v, 0)])
        );
        let h = run(PERMPD, |h| set(h, 1, v));
        let pick = |k: usize| qword(&v, ((0xc6 >> (2 * k)) & 3) as usize);
        assert_eq!(
            get(&h, 0),
            from_qwords(&[pick(0), pick(1), pick(2), pick(3)])
        );
        // vpermd/vpermps: indices are ymm1 (vvvv), data is ymm2 (r/m)
        for code in [PERMPS, PERMD] {
            let h = run(code, |h| {
                set(h, 1, selector);
                set(h, 2, v);
            });
            let expected: Vec<u32> = (0..8)
                .map(|k| dword(&v, (dword(&selector, k) & 7) as usize))
                .collect();
            assert_eq!(get(&h, 0), from_dwords(&expected));
        }
        // 128-bit lane shuffles: imm 0x31 -> low = src1.hi, high = src2.hi; 0x83 -> low = src2.hi, high = 0
        for (code, imm) in [(PERM2F128, 0x31u8), (PERM2I128, 0x83)] {
            let h = run(code, |h| {
                set(h, 1, v);
                set(h, 2, selector);
            });
            let halves = [&v[..16], &v[16..], &selector[..16], &selector[16..]];
            let pick = |c: u8| -> Vec<u8> {
                if c & 8 != 0 {
                    vec![0; 16]
                } else {
                    halves[(c & 3) as usize].to_vec()
                }
            };
            let expected = [pick(imm & 0xf), pick(imm >> 4)].concat();
            assert_eq!(get(&h, 0).to_vec(), expected, "imm {imm:#x}");
        }
    }
}

#[test]
fn insert_and_extract_128() {
    for (v, w) in inputs() {
        let h = run(INSERTF128_1, |h| {
            set(h, 1, v);
            set(h, 2, w);
        });
        assert_eq!(get(&h, 0).to_vec(), [&v[..16], &w[..16]].concat());
        let h = run(INSERTI128_0, |h| {
            set(h, 1, v);
            set(h, 2, w);
        });
        assert_eq!(get(&h, 0).to_vec(), [&w[..16], &v[16..]].concat());
        let h = run(INSERTF128_M, |h| {
            set(h, 1, v);
            h.write(DATA, &w[..16]);
        });
        assert_eq!(get(&h, 0).to_vec(), [&v[..16], &w[..16]].concat());
        let h = run(EXTRACTF128_1, |h| {
            set(h, 1, v);
            set(h, 0, [0xaa; 32]);
        });
        assert_eq!(
            get(&h, 0),
            zero_upper({
                let mut x = [0u8; 32];
                x[..16].copy_from_slice(&v[16..]);
                x
            })
        );
        let h = run(EXTRACTI128_0, |h| set(h, 1, v));
        assert_eq!(&get(&h, 0)[..16], &v[..16]);
        let h = run(EXTRACTF128_MEM, |h| {
            set(h, 1, v);
            h.write(OUT, &[0xcc; 32]);
        });
        assert_eq!(h.read(OUT, 16), v[16..].to_vec());
        assert_eq!(
            h.read(OUT + 16, 16),
            vec![0xcc; 16],
            "only 16 bytes are stored"
        );
    }
}

#[test]
fn broadcasts() {
    for (v, _) in inputs() {
        let lanes = |element: &[u8], count: usize| -> Ymm {
            let mut out = [0u8; 32];
            for k in 0..count {
                out[k * element.len()..(k + 1) * element.len()].copy_from_slice(element);
            }
            out
        };
        let h = run(BROADCASTSS_Y, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), lanes(&v[..4], 8));
        let h = run(BROADCASTSS_YM, |h| h.write(DATA, &v[..4]));
        assert_eq!(get(&h, 0), lanes(&v[..4], 8));
        let h = run(BROADCASTSS_X, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), zero_upper(lanes(&v[..4], 8)));
        let h = run(BROADCASTSD_Y, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), lanes(&v[..8], 4));
        let h = run(PBROADCASTQ_Y, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), lanes(&v[..8], 4));
        let h = run(PBROADCASTD_Y, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), lanes(&v[..4], 8));
        let h = run(PBROADCASTB_Y, |h| set(h, 1, v));
        assert_eq!(get(&h, 0), lanes(&v[..1], 32));
        let h = run(PBROADCASTB_XM, |h| h.write(DATA, &v[..1]));
        assert_eq!(get(&h, 0), zero_upper(lanes(&v[..1], 32)));
        let h = run(PBROADCASTW_XM, |h| h.write(DATA, &v[..2]));
        assert_eq!(get(&h, 0), zero_upper(lanes(&v[..2], 16)));
        for code in [BROADCASTF128, BROADCASTI128] {
            let h = run(code, |h| h.write(DATA, &v[..16]));
            assert_eq!(get(&h, 0).to_vec(), [&v[..16], &v[..16]].concat());
        }
    }
}

/// Widening integer conversion of the low `elements` elements of `source`.
fn widen_model(source: &[u8], from: usize, to: usize, signed: bool, elements: usize) -> Vec<u8> {
    let mut out = vec![0u8; elements * to];
    for k in 0..elements {
        let mut value = 0u64;
        for b in 0..from {
            value |= (source[k * from + b] as u64) << (8 * b);
        }
        let extended = if signed && (value >> (8 * from - 1)) & 1 != 0 {
            value | (u64::MAX << (8 * from))
        } else {
            value
        };
        out[k * to..(k + 1) * to].copy_from_slice(&extended.to_le_bytes()[..to]);
    }
    out
}

#[test]
fn pmovsx_pmovzx_widen_twice_as_many_elements_with_vex_256() {
    let table: [WidenCase; 10] = [
        (PMOVSXBW_Y, PMOVSXBW_X, PMOVSXBW_YM, 1, 2, true),
        (PMOVZXBW_Y, PMOVZXBW_X, PMOVZXBW_YM, 1, 2, false),
        (PMOVSXBD_Y, PMOVSXBD_X, PMOVSXBD_YM, 1, 4, true),
        (PMOVZXBD_Y, PMOVZXBD_X, PMOVZXBD_YM, 1, 4, false),
        (PMOVSXBQ_Y, PMOVSXBQ_X, PMOVSXBQ_YM, 1, 8, true),
        (PMOVZXWD_Y, PMOVZXWD_X, PMOVZXWD_YM, 2, 4, false),
        (PMOVSXWQ_Y, PMOVSXWQ_X, PMOVSXWQ_YM, 2, 8, true),
        (PMOVZXDQ_Y, PMOVZXDQ_X, PMOVZXDQ_YM, 4, 8, false),
        (PMOVSXDQ_Y, PMOVSXDQ_X, PMOVSXDQ_YM, 4, 8, true),
        (PMOVSXWD_Y, PMOVSXWD_X, PMOVSXWD_YM, 2, 4, true),
    ];
    for (wide, narrow, memory, from, to, signed) in table {
        for (v, _) in inputs() {
            let wide_elements = 32 / to;
            let expected_wide = widen_model(&v, from, to, signed, wide_elements);
            let h = run(wide, |h| set(h, 1, v));
            assert_eq!(
                get(&h, 0).to_vec(),
                expected_wide,
                "{from}->{to} signed={signed} ymm"
            );
            let h = run(narrow, |h| {
                set(h, 1, v);
                set(h, 0, [0xaa; 32]);
            });
            let mut expected = widen_model(&v, from, to, signed, 16 / to);
            expected.resize(32, 0);
            assert_eq!(get(&h, 0).to_vec(), expected, "{from}->{to} xmm");
            // the memory form reads exactly the bytes it widens (the rest of the page is junk)
            let needed = wide_elements * from;
            let h = run(memory, |h| {
                h.write(DATA, &[0xee; 64]);
                h.write(DATA, &v[..needed]);
            });
            assert_eq!(get(&h, 0).to_vec(), expected_wide, "{from}->{to} memory");
        }
    }
}

#[test]
fn variable_blends_use_the_register_named_in_the_immediate() {
    for (a, b) in inputs() {
        let mut rng = Rng(a[0] as u64 + 1);
        let mask = rng.ymm();
        let (blendvps, blendvpd, pblendvb_y, pblendvb_x) =
            (BLENDVPS_Y, BLENDVPD_Y, PBLENDVB_Y, PBLENDVB_X);
        for (code, width, lanes) in [
            (blendvps, 4, 2),
            (blendvpd, 8, 2),
            (pblendvb_y, 1, 2),
            (pblendvb_x, 1, 1),
        ] {
            // vblendvps ymm0, ymm1, ymm2, ymm3: dst = mask(ymm3) ? ymm2 : ymm1
            let h = run(code, |h| {
                set(h, 1, a);
                set(h, 2, b);
                set(h, 3, mask);
            });
            let mut expected = [0u8; 32];
            for k in 0..lanes * 16 / width {
                let taken = mask[k * width + width - 1] >> 7 != 0;
                let source = if taken { &b } else { &a };
                expected[k * width..(k + 1) * width]
                    .copy_from_slice(&source[k * width..(k + 1) * width]);
            }
            assert_eq!(get(&h, 0), expected, "{code:02x?}");
        }
    }
}

#[test]
fn vpblendd_selects_dwords_by_immediate_bits() {
    for (a, b) in inputs() {
        let h = run(PBLENDD_Y, |h| {
            set(h, 1, a);
            set(h, 2, b);
        });
        let expected: Vec<u32> = (0..8)
            .map(|k| {
                if 0xa5 >> k & 1 != 0 {
                    dword(&b, k)
                } else {
                    dword(&a, k)
                }
            })
            .collect();
        assert_eq!(get(&h, 0), from_dwords(&expected));
        let h = run(PBLENDD_X, |h| {
            set(h, 1, a);
            set(h, 2, b);
        });
        let expected: Vec<u32> = (0..4)
            .map(|k| {
                if 0x6 >> k & 1 != 0 {
                    dword(&b, k)
                } else {
                    dword(&a, k)
                }
            })
            .collect();
        assert_eq!(get(&h, 0), zero_upper(from_dwords(&expected)));
    }
}

#[test]
fn per_element_variable_shifts() {
    for (value, counts) in inputs() {
        let mut small = counts;
        for byte in small.iter_mut().step_by(4) {
            *byte %= 70;
        }
        for counts in [counts, small] {
            let h = run(PSRLVD_Y, |h| {
                set(h, 1, value);
                set(h, 2, counts);
            });
            let expected: Vec<u32> = (0..8)
                .map(|k| {
                    let c = dword(&counts, k);
                    if c >= 32 { 0 } else { dword(&value, k) >> c }
                })
                .collect();
            assert_eq!(get(&h, 0), from_dwords(&expected));
            let h = run(PSRAVD_Y, |h| {
                set(h, 1, value);
                set(h, 2, counts);
            });
            let expected: Vec<u32> = (0..8)
                .map(|k| ((dword(&value, k) as i32) >> dword(&counts, k).min(31)) as u32)
                .collect();
            assert_eq!(get(&h, 0), from_dwords(&expected));
            let h = run(PSLLVQ_Y, |h| {
                set(h, 1, value);
                set(h, 2, counts);
            });
            let expected: Vec<u64> = (0..4)
                .map(|k| {
                    let c = qword(&counts, k);
                    if c >= 64 { 0 } else { qword(&value, k) << c }
                })
                .collect();
            assert_eq!(get(&h, 0), from_qwords(&expected));
            let h = run(PSRLVQ_X, |h| {
                set(h, 1, value);
                set(h, 2, counts);
            });
            let expected: Vec<u64> = (0..2)
                .map(|k| {
                    let c = qword(&counts, k);
                    if c >= 64 { 0 } else { qword(&value, k) >> c }
                })
                .collect();
            assert_eq!(get(&h, 0), zero_upper(from_qwords(&expected)));
            let h = run(PSLLVD_X, |h| {
                set(h, 1, value);
                set(h, 2, counts);
            });
            let expected: Vec<u32> = (0..4)
                .map(|k| {
                    let c = dword(&counts, k);
                    if c >= 32 { 0 } else { dword(&value, k) << c }
                })
                .collect();
            assert_eq!(get(&h, 0), zero_upper(from_dwords(&expected)));
        }
    }
}

#[test]
fn the_256_bit_only_instructions_fault_with_vex_l_clear() {
    // vpermq with L=0: flip the L bit of the vpermq encoding (byte 2 of the c4 form, bit 2)
    let mut code = PERMQ.to_vec();
    code[2] &= !0x04;
    let mut h = harness(&code);
    assert!(matches!(h.run(10), Stop::Fault { vector: 6, .. }));
    let mut code = BROADCASTSD_Y.to_vec();
    code[2] &= !0x04;
    let mut h = harness(&code);
    assert!(matches!(h.run(10), Stop::Fault { vector: 6, .. }));
}

// ---- FMA, F16C and the VEX compare predicates ----

const MXCSR_INVALID: u32 = 1;
const MXCSR_PRECISION: u32 = 1 << 5;
const MXCSR_UNDERFLOW: u32 = 1 << 4;
const MXCSR_OVERFLOW: u32 = 1 << 3;

fn singles(v: &Ymm) -> [f32; 8] {
    core::array::from_fn(|k| f32::from_bits(dword(v, k)))
}

fn doubles(v: &Ymm) -> [f64; 4] {
    core::array::from_fn(|k| f64::from_bits(qword(v, k)))
}

/// Finite floats over a wide range of magnitudes, with subnormals and exact cancellations.
fn float_samples(rng: &mut Rng, count: usize) -> Vec<f64> {
    let mut out = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        3.0,
        1e-310,
        -1e-310,
        1e300,
        -1e300,
        5e-324,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    for _ in 0..count {
        let mantissa = (rng.next() >> 12) | (1023u64 << 52);
        let exponent = (rng.next() % 200) as i32 - 100;
        let value = f64::from_bits(mantissa) * 2f64.powi(exponent);
        out.push(if rng.next() & 1 == 0 { value } else { -value });
    }
    out
}

fn fma_ref<T: Copy + core::ops::Neg<Output = T>>(
    mul_add: impl Fn(T, T, T) -> T,
    form: &str,
    kind: &str,
    ops: [T; 3],
    lane: usize,
) -> T {
    let [op1, op2, op3] = ops;
    let (x, y, z) = match form {
        "132" => (op1, op3, op2),
        "213" => (op2, op1, op3),
        _ => (op2, op3, op1),
    };
    match kind {
        "MADD" => mul_add(x, y, z),
        "MSUB" => mul_add(x, y, -z),
        "NMADD" => mul_add(-x, y, z),
        "NMSUB" => mul_add(-x, y, -z),
        // vfmaddsub: even lanes subtract
        "MADDSUB" => {
            if lane.is_multiple_of(2) {
                mul_add(x, y, -z)
            } else {
                mul_add(x, y, z)
            }
        }
        _ => {
            if lane.is_multiple_of(2) {
                mul_add(x, y, z)
            } else {
                mul_add(x, y, -z)
            }
        }
    }
}

fn same_float32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_float64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn encoding(form: &str, kind: &str, suffix: &str) -> &'static [u8] {
    match (form, kind, suffix) {
        ("132", "MADD", "PS_Y") => FMA_132_MADD_PS_Y,
        ("132", "MADD", "PS_X") => FMA_132_MADD_PS_X,
        ("132", "MADD", "PD_Y") => FMA_132_MADD_PD_Y,
        ("132", "MADD", "PD_X") => FMA_132_MADD_PD_X,
        ("132", "MADD", "SS") => FMA_132_MADD_SS,
        ("132", "MADD", "SD") => FMA_132_MADD_SD,
        ("132", "MSUB", "PS_Y") => FMA_132_MSUB_PS_Y,
        ("132", "MSUB", "PS_X") => FMA_132_MSUB_PS_X,
        ("132", "MSUB", "PD_Y") => FMA_132_MSUB_PD_Y,
        ("132", "MSUB", "PD_X") => FMA_132_MSUB_PD_X,
        ("132", "MSUB", "SS") => FMA_132_MSUB_SS,
        ("132", "MSUB", "SD") => FMA_132_MSUB_SD,
        ("132", "NMADD", "PS_Y") => FMA_132_NMADD_PS_Y,
        ("132", "NMADD", "PS_X") => FMA_132_NMADD_PS_X,
        ("132", "NMADD", "PD_Y") => FMA_132_NMADD_PD_Y,
        ("132", "NMADD", "PD_X") => FMA_132_NMADD_PD_X,
        ("132", "NMADD", "SS") => FMA_132_NMADD_SS,
        ("132", "NMADD", "SD") => FMA_132_NMADD_SD,
        ("132", "NMSUB", "PS_Y") => FMA_132_NMSUB_PS_Y,
        ("132", "NMSUB", "PS_X") => FMA_132_NMSUB_PS_X,
        ("132", "NMSUB", "PD_Y") => FMA_132_NMSUB_PD_Y,
        ("132", "NMSUB", "PD_X") => FMA_132_NMSUB_PD_X,
        ("132", "NMSUB", "SS") => FMA_132_NMSUB_SS,
        ("132", "NMSUB", "SD") => FMA_132_NMSUB_SD,
        ("132", "MADDSUB", "PS_Y") => FMA_132_MADDSUB_PS_Y,
        ("132", "MADDSUB", "PD_X") => FMA_132_MADDSUB_PD_X,
        ("132", "MSUBADD", "PS_Y") => FMA_132_MSUBADD_PS_Y,
        ("132", "MSUBADD", "PD_Y") => FMA_132_MSUBADD_PD_Y,
        ("213", "MADD", "PS_Y") => FMA_213_MADD_PS_Y,
        ("213", "MADD", "PS_X") => FMA_213_MADD_PS_X,
        ("213", "MADD", "PD_Y") => FMA_213_MADD_PD_Y,
        ("213", "MADD", "PD_X") => FMA_213_MADD_PD_X,
        ("213", "MADD", "SS") => FMA_213_MADD_SS,
        ("213", "MADD", "SD") => FMA_213_MADD_SD,
        ("213", "MSUB", "PS_Y") => FMA_213_MSUB_PS_Y,
        ("213", "MSUB", "PS_X") => FMA_213_MSUB_PS_X,
        ("213", "MSUB", "PD_Y") => FMA_213_MSUB_PD_Y,
        ("213", "MSUB", "PD_X") => FMA_213_MSUB_PD_X,
        ("213", "MSUB", "SS") => FMA_213_MSUB_SS,
        ("213", "MSUB", "SD") => FMA_213_MSUB_SD,
        ("213", "NMADD", "PS_Y") => FMA_213_NMADD_PS_Y,
        ("213", "NMADD", "PS_X") => FMA_213_NMADD_PS_X,
        ("213", "NMADD", "PD_Y") => FMA_213_NMADD_PD_Y,
        ("213", "NMADD", "PD_X") => FMA_213_NMADD_PD_X,
        ("213", "NMADD", "SS") => FMA_213_NMADD_SS,
        ("213", "NMADD", "SD") => FMA_213_NMADD_SD,
        ("213", "NMSUB", "PS_Y") => FMA_213_NMSUB_PS_Y,
        ("213", "NMSUB", "PS_X") => FMA_213_NMSUB_PS_X,
        ("213", "NMSUB", "PD_Y") => FMA_213_NMSUB_PD_Y,
        ("213", "NMSUB", "PD_X") => FMA_213_NMSUB_PD_X,
        ("213", "NMSUB", "SS") => FMA_213_NMSUB_SS,
        ("213", "NMSUB", "SD") => FMA_213_NMSUB_SD,
        ("213", "MADDSUB", "PS_Y") => FMA_213_MADDSUB_PS_Y,
        ("213", "MADDSUB", "PD_X") => FMA_213_MADDSUB_PD_X,
        ("213", "MSUBADD", "PS_Y") => FMA_213_MSUBADD_PS_Y,
        ("213", "MSUBADD", "PD_Y") => FMA_213_MSUBADD_PD_Y,
        ("231", "MADD", "PS_Y") => FMA_231_MADD_PS_Y,
        ("231", "MADD", "PS_X") => FMA_231_MADD_PS_X,
        ("231", "MADD", "PD_Y") => FMA_231_MADD_PD_Y,
        ("231", "MADD", "PD_X") => FMA_231_MADD_PD_X,
        ("231", "MADD", "SS") => FMA_231_MADD_SS,
        ("231", "MADD", "SD") => FMA_231_MADD_SD,
        ("231", "MSUB", "PS_Y") => FMA_231_MSUB_PS_Y,
        ("231", "MSUB", "PS_X") => FMA_231_MSUB_PS_X,
        ("231", "MSUB", "PD_Y") => FMA_231_MSUB_PD_Y,
        ("231", "MSUB", "PD_X") => FMA_231_MSUB_PD_X,
        ("231", "MSUB", "SS") => FMA_231_MSUB_SS,
        ("231", "MSUB", "SD") => FMA_231_MSUB_SD,
        ("231", "NMADD", "PS_Y") => FMA_231_NMADD_PS_Y,
        ("231", "NMADD", "PS_X") => FMA_231_NMADD_PS_X,
        ("231", "NMADD", "PD_Y") => FMA_231_NMADD_PD_Y,
        ("231", "NMADD", "PD_X") => FMA_231_NMADD_PD_X,
        ("231", "NMADD", "SS") => FMA_231_NMADD_SS,
        ("231", "NMADD", "SD") => FMA_231_NMADD_SD,
        ("231", "NMSUB", "PS_Y") => FMA_231_NMSUB_PS_Y,
        ("231", "NMSUB", "PS_X") => FMA_231_NMSUB_PS_X,
        ("231", "NMSUB", "PD_Y") => FMA_231_NMSUB_PD_Y,
        ("231", "NMSUB", "PD_X") => FMA_231_NMSUB_PD_X,
        ("231", "NMSUB", "SS") => FMA_231_NMSUB_SS,
        ("231", "NMSUB", "SD") => FMA_231_NMSUB_SD,
        ("231", "MADDSUB", "PS_Y") => FMA_231_MADDSUB_PS_Y,
        ("231", "MADDSUB", "PD_X") => FMA_231_MADDSUB_PD_X,
        ("231", "MSUBADD", "PS_Y") => FMA_231_MSUBADD_PS_Y,
        ("231", "MSUBADD", "PD_Y") => FMA_231_MSUBADD_PD_Y,
        _ => unreachable!(),
    }
}
const FMA_VARIANTS: &[(&str, &str, &str)] = &[
    ("132", "MADD", "PS_Y"),
    ("132", "MADD", "PS_X"),
    ("132", "MADD", "PD_Y"),
    ("132", "MADD", "PD_X"),
    ("132", "MADD", "SS"),
    ("132", "MADD", "SD"),
    ("132", "MSUB", "PS_Y"),
    ("132", "MSUB", "PS_X"),
    ("132", "MSUB", "PD_Y"),
    ("132", "MSUB", "PD_X"),
    ("132", "MSUB", "SS"),
    ("132", "MSUB", "SD"),
    ("132", "NMADD", "PS_Y"),
    ("132", "NMADD", "PS_X"),
    ("132", "NMADD", "PD_Y"),
    ("132", "NMADD", "PD_X"),
    ("132", "NMADD", "SS"),
    ("132", "NMADD", "SD"),
    ("132", "NMSUB", "PS_Y"),
    ("132", "NMSUB", "PS_X"),
    ("132", "NMSUB", "PD_Y"),
    ("132", "NMSUB", "PD_X"),
    ("132", "NMSUB", "SS"),
    ("132", "NMSUB", "SD"),
    ("132", "MADDSUB", "PS_Y"),
    ("132", "MADDSUB", "PD_X"),
    ("132", "MSUBADD", "PS_Y"),
    ("132", "MSUBADD", "PD_Y"),
    ("213", "MADD", "PS_Y"),
    ("213", "MADD", "PS_X"),
    ("213", "MADD", "PD_Y"),
    ("213", "MADD", "PD_X"),
    ("213", "MADD", "SS"),
    ("213", "MADD", "SD"),
    ("213", "MSUB", "PS_Y"),
    ("213", "MSUB", "PS_X"),
    ("213", "MSUB", "PD_Y"),
    ("213", "MSUB", "PD_X"),
    ("213", "MSUB", "SS"),
    ("213", "MSUB", "SD"),
    ("213", "NMADD", "PS_Y"),
    ("213", "NMADD", "PS_X"),
    ("213", "NMADD", "PD_Y"),
    ("213", "NMADD", "PD_X"),
    ("213", "NMADD", "SS"),
    ("213", "NMADD", "SD"),
    ("213", "NMSUB", "PS_Y"),
    ("213", "NMSUB", "PS_X"),
    ("213", "NMSUB", "PD_Y"),
    ("213", "NMSUB", "PD_X"),
    ("213", "NMSUB", "SS"),
    ("213", "NMSUB", "SD"),
    ("213", "MADDSUB", "PS_Y"),
    ("213", "MADDSUB", "PD_X"),
    ("213", "MSUBADD", "PS_Y"),
    ("213", "MSUBADD", "PD_Y"),
    ("231", "MADD", "PS_Y"),
    ("231", "MADD", "PS_X"),
    ("231", "MADD", "PD_Y"),
    ("231", "MADD", "PD_X"),
    ("231", "MADD", "SS"),
    ("231", "MADD", "SD"),
    ("231", "MSUB", "PS_Y"),
    ("231", "MSUB", "PS_X"),
    ("231", "MSUB", "PD_Y"),
    ("231", "MSUB", "PD_X"),
    ("231", "MSUB", "SS"),
    ("231", "MSUB", "SD"),
    ("231", "NMADD", "PS_Y"),
    ("231", "NMADD", "PS_X"),
    ("231", "NMADD", "PD_Y"),
    ("231", "NMADD", "PD_X"),
    ("231", "NMADD", "SS"),
    ("231", "NMADD", "SD"),
    ("231", "NMSUB", "PS_Y"),
    ("231", "NMSUB", "PS_X"),
    ("231", "NMSUB", "PD_Y"),
    ("231", "NMSUB", "PD_X"),
    ("231", "NMSUB", "SS"),
    ("231", "NMSUB", "SD"),
    ("231", "MADDSUB", "PS_Y"),
    ("231", "MADDSUB", "PD_X"),
    ("231", "MSUBADD", "PS_Y"),
    ("231", "MSUBADD", "PD_Y"),
];

#[test]
fn fma_matches_the_host_fused_multiply_add_for_every_form() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let samples = float_samples(&mut rng, 24);
    for &(form, kind, suffix) in FMA_VARIANTS {
        let code = encoding(form, kind, suffix);
        for round in 0..24 {
            let pick = |rng: &mut Rng| samples[(rng.next() % samples.len() as u64) as usize];
            let mut inputs = [[0u8; 32]; 3];
            let lane_values: Vec<[f64; 4]> = (0..3)
                .map(|_| core::array::from_fn(|_| pick(&mut rng)))
                .collect();
            // every few rounds, force an exact cancellation: z = -(x*y) rounded, so fusing matters
            let cancel = round % 4 == 0;
            let mut values = lane_values.clone();
            if cancel {
                let products: Vec<f64> = values[0]
                    .iter()
                    .zip(&values[1])
                    .map(|(x, y)| -(x * y))
                    .collect();
                values[2].copy_from_slice(&products);
            }
            let double = suffix.contains("PD") || suffix == "SD";
            for (op, v) in inputs.iter_mut().zip(&values) {
                *op = if double {
                    from_qwords(&v.map(f64::to_bits))
                } else {
                    // single precision: reuse the same samples narrowed (cancellation recomputed in f32)
                    from_dwords(
                        &[
                            v[0] as f32,
                            v[1] as f32,
                            v[2] as f32,
                            v[3] as f32,
                            v[3] as f32,
                            v[2] as f32,
                            v[1] as f32,
                            v[0] as f32,
                        ]
                        .map(f32::to_bits),
                    )
                };
            }
            if cancel && !double {
                let s: Vec<[f32; 8]> = inputs.iter().map(singles).collect();
                let z: [f32; 8] = core::array::from_fn(|k| -(s[0][k] * s[1][k]));
                inputs[2] = from_dwords(&z.map(f32::to_bits));
            }
            // operand 1 is the destination register (xmm0/ymm0), 2 is vvvv (xmm1), 3 is rm (xmm2)
            let h = run(code, |h| {
                set(h, 0, inputs[0]);
                set(h, 1, inputs[1]);
                set(h, 2, inputs[2]);
            });
            let result = get(&h, 0);
            let wide = suffix.ends_with("_Y");
            let scalar = suffix == "SS" || suffix == "SD";
            let lanes = if double {
                if wide {
                    4
                } else if scalar {
                    1
                } else {
                    2
                }
            } else if wide {
                8
            } else if scalar {
                1
            } else {
                4
            };
            for lane in 0..lanes {
                if double {
                    let ops = [
                        doubles(&inputs[0])[lane],
                        doubles(&inputs[1])[lane],
                        doubles(&inputs[2])[lane],
                    ];
                    let expected = fma_ref(f64::mul_add, form, kind, ops, lane);
                    assert!(
                        same_float64(doubles(&result)[lane], expected),
                        "{form} {kind} {suffix} lane {lane}: {ops:?} -> {:e} expected {expected:e}",
                        doubles(&result)[lane]
                    );
                } else {
                    let ops = [
                        singles(&inputs[0])[lane],
                        singles(&inputs[1])[lane],
                        singles(&inputs[2])[lane],
                    ];
                    let expected = fma_ref(f32::mul_add, form, kind, ops, lane);
                    assert!(
                        same_float32(singles(&result)[lane], expected),
                        "{form} {kind} {suffix} lane {lane}: {ops:?} -> {:e} expected {expected:e}",
                        singles(&result)[lane]
                    );
                }
            }
            // everything the instruction does not compute is preserved or zeroed as specified
            let element = if double { 8 } else { 4 };
            let computed = lanes * element;
            if scalar {
                assert_eq!(
                    &result[computed..16],
                    &inputs[0][computed..16],
                    "scalar keeps operand 1's upper bits"
                );
            }
            if !wide {
                assert_eq!(&result[16..], &[0u8; 16], "VEX.128 zeroes the upper half");
            }
        }
    }
}

#[test]
fn fma_nan_and_special_operand_rules() {
    const QNAN: u32 = 0x7fc0_1234;
    const SNAN: u32 = 0x7f80_0001;
    const DEFAULT_NAN: u32 = 0xffc0_0000;
    let code = encoding("231", "MADD", "SS");
    // vfmadd231ss xmm0(op1), xmm1(op2), xmm2(op3): op2*op3 + op1
    let scalar = |op1: u32, op2: u32, op3: u32| {
        let h = run(code, |h| {
            set(h, 0, from_dwords(&[op1]));
            set(h, 1, from_dwords(&[op2]));
            set(h, 2, from_dwords(&[op3]));
        });
        (dword(&get(&h, 0), 0), h.cpu.mxcsr & 0x3f)
    };
    let one = 1.0f32.to_bits();
    // the first NaN in operand order wins, quieted
    assert_eq!(scalar(QNAN, one, one).0, QNAN);
    assert_eq!(scalar(one, QNAN | 1, QNAN).0, QNAN | 1);
    assert_eq!(scalar(one, one, SNAN).0, SNAN | 0x0040_0000);
    assert_eq!(scalar(one, one, SNAN).1 & MXCSR_INVALID, MXCSR_INVALID);
    assert_eq!(
        scalar(QNAN, one, one).1 & MXCSR_INVALID,
        0,
        "a quiet NaN raises nothing"
    );
    // 0 * inf is invalid
    let inf = f32::INFINITY.to_bits();
    let (value, flags) = scalar(one, 0, inf);
    assert_eq!((value, flags & MXCSR_INVALID), (DEFAULT_NAN, MXCSR_INVALID));
    // inf - inf is invalid, inf + inf is inf
    assert_eq!(scalar(f32::NEG_INFINITY.to_bits(), one, inf).0, DEFAULT_NAN);
    assert_eq!(scalar(inf, one, inf).0, inf);
    // a zero product keeps the sign rules of an exact zero sum
    assert_eq!(
        scalar((-0.0f32).to_bits(), 0, one).0,
        0,
        "(+0*1) + (-0) is +0"
    );
    assert_eq!(
        scalar((-0.0f32).to_bits(), (-0.0f32).to_bits(), one).0,
        (-0.0f32).to_bits()
    );
}

#[test]
fn fma_rounds_once_where_multiply_then_add_would_not() {
    // (1 + 2^-23) * (1 - 2^-23) = 1 - 2^-46: the unfused product rounds to 1.0, so x*y + (-1) is 0
    // unfused but exactly -2^-46 fused.
    let x = f32::from_bits(0x3f80_0001);
    let y = f32::from_bits(0x3f7f_fffe);
    let code = encoding("231", "MADD", "SS");
    let h = run(code, |h| {
        set(h, 0, from_dwords(&[(-1.0f32).to_bits()]));
        set(h, 1, from_dwords(&[x.to_bits()]));
        set(h, 2, from_dwords(&[y.to_bits()]));
    });
    let fused = f32::from_bits(dword(&get(&h, 0), 0));
    assert_ne!(fused, 0.0);
    assert_eq!(fused, x.mul_add(y, -1.0));
    assert_eq!(
        h.cpu.mxcsr & MXCSR_PRECISION,
        0,
        "the exact result needs no rounding"
    );
}

#[test]
fn fma_respects_mxcsr_rounding_and_flags() {
    let code = encoding("231", "MADD", "SD");
    // 1e308 * 10 overflows; under round-toward-zero the result is the largest finite double
    let h = run(code, |h| {
        h.cpu.mxcsr = 0x1f80 | (3 << 13);
        set(h, 0, from_qwords(&[0.0f64.to_bits()]));
        set(h, 1, from_qwords(&[1e308f64.to_bits()]));
        set(h, 2, from_qwords(&[10.0f64.to_bits()]));
    });
    assert_eq!(qword(&get(&h, 0), 0), f64::MAX.to_bits());
    assert_eq!(
        h.cpu.mxcsr & (MXCSR_OVERFLOW | MXCSR_PRECISION),
        MXCSR_OVERFLOW | MXCSR_PRECISION
    );
    // unmasked overflow is #XM and leaves the destination alone
    let mut h = harness(code);
    h.cpu.cr4 |= 1 << 10;
    h.cpu.mxcsr = 0x1f80 & !(1 << 10);
    set(&mut h, 0, from_qwords(&[7.0f64.to_bits()]));
    set(&mut h, 1, from_qwords(&[1e308f64.to_bits()]));
    set(&mut h, 2, from_qwords(&[10.0f64.to_bits()]));
    assert!(matches!(h.run(10), Stop::Fault { vector: 19, .. }));
    assert_eq!(qword(&get(&h, 0), 0), 7.0f64.to_bits());
}

#[test]
fn fma_memory_forms_read_exactly_their_operand() {
    let h = run(FMA_231_MADD_PS_YM, |h| {
        set(h, 0, from_dwords(&[1.0f32.to_bits(); 8]));
        set(h, 1, from_dwords(&[2.0f32.to_bits(); 8]));
        h.write(DATA, &from_dwords(&[3.0f32.to_bits(); 8]));
    });
    assert_eq!(get(&h, 0), from_dwords(&[7.0f32.to_bits(); 8]));
    let h = run(FMA_231_MADD_SD_M, |h| {
        set(h, 0, from_qwords(&[1.0f64.to_bits(), 0xaaaa]));
        set(h, 1, from_qwords(&[2.0f64.to_bits()]));
        h.write(DATA, &3.0f64.to_bits().to_le_bytes());
        h.write(DATA + 8, &[0xff; 8]);
    });
    assert_eq!(
        get(&h, 0)[..16],
        from_qwords(&[7.0f64.to_bits(), 0xaaaa])[..16]
    );
}

/// Exact value of a half-precision bit pattern, in f64.
fn half_value(bits: u16) -> f64 {
    let sign = if bits >> 15 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f64;
    if exponent == 0 {
        sign * fraction * 2f64.powi(-24)
    } else {
        sign * (1024.0 + fraction) * 2f64.powi(exponent - 25)
    }
}

#[test]
fn vcvtph2ps_converts_every_half_exactly() {
    for chunk in (0..=u16::MAX as u32).collect::<Vec<_>>().chunks(8) {
        let mut source = [0u8; 32];
        for (k, &half) in chunk.iter().enumerate() {
            source[2 * k..2 * k + 2].copy_from_slice(&(half as u16).to_le_bytes());
        }
        let h = run(CVTPH2PS_Y, |h| set(h, 1, source));
        let result = singles(&get(&h, 0));
        for (k, &half) in chunk.iter().enumerate() {
            let half = half as u16;
            let is_nan = (half & 0x7c00) == 0x7c00 && (half & 0x3ff) != 0;
            if is_nan {
                assert!(result[k].is_nan(), "{half:#06x}");
                assert_ne!(
                    dword(&get(&h, 0), k) & 0x0040_0000,
                    0,
                    "NaNs come out quiet"
                );
                assert_eq!(dword(&get(&h, 0), k) >> 31, (half >> 15) as u32);
                let signaling = half & 0x200 == 0;
                assert!(h.cpu.mxcsr & MXCSR_INVALID != 0 || !signaling);
            } else if (half & 0x7c00) == 0x7c00 {
                assert_eq!(
                    result[k],
                    if half >> 15 != 0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                );
            } else {
                assert_eq!(result[k] as f64, half_value(half), "{half:#06x}");
                assert_eq!(result[k].is_sign_negative(), half >> 15 != 0);
            }
        }
    }
    // 128-bit form converts four and zeroes the rest; memory form reads 8 or 16 bytes
    let h = run(CVTPH2PS_X, |h| {
        set(
            h,
            1,
            from_qwords(&[0x4000_3c00_c000_bc00, 0x3c00_3c00_3c00_3c00]),
        );
        set(h, 0, [0xaa; 32]);
    });
    assert_eq!(
        get(&h, 0),
        from_dwords(&[
            (-1.0f32).to_bits(),
            (-2.0f32).to_bits(),
            1.0f32.to_bits(),
            2.0f32.to_bits()
        ])
    );
    let h = run(CVTPH2PS_XM, |h| {
        h.write(DATA, &0x4000_3c00_c000_bc00u64.to_le_bytes());
        h.write(DATA + 8, &[0xff; 8]);
    });
    assert_eq!(singles(&get(&h, 0))[3], 2.0);
    let _ = CVTPH2PS_YM;
    // a signaling NaN input raises invalid
    let h = run(CVTPH2PS_X, |h| set(h, 1, from_qwords(&[0x7c01])));
    assert_eq!(h.cpu.mxcsr & MXCSR_INVALID, MXCSR_INVALID);
}

/// Independent reference for single -> half: search the table of all half values.
fn reference_single_to_half(x: f32, rounding: u32) -> (u16, u32) {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if x.is_nan() {
        let payload = ((x.to_bits() >> 13) & 0x3ff) as u16;
        let flags = if x.to_bits() & 0x0040_0000 == 0 {
            MXCSR_INVALID
        } else {
            0
        };
        return (sign | 0x7e00 | payload, flags);
    }
    if x.is_infinite() {
        return (sign | 0x7c00, 0);
    }
    let magnitude = (x as f64).abs();
    let negative = x.is_sign_negative();
    // finite non-negative half magnitudes in increasing order
    let table: Vec<(u16, f64)> = (0..0x7c00u16).map(|b| (b, half_value(b))).collect();
    let max = table.last().unwrap();
    if magnitude > max.1 {
        // beyond the largest finite half: infinity or max depending on the mode and the midpoint
        let midpoint = max.1 + 16.0; // 65520, the round-to-nearest threshold
        let to_infinity = match rounding {
            0 => magnitude >= midpoint,
            1 => negative,
            2 => !negative,
            _ => false,
        };
        let bits = if to_infinity { 0x7c00 } else { max.0 };
        return (sign | bits, MXCSR_OVERFLOW | MXCSR_PRECISION);
    }
    let upper = table.partition_point(|&(_, v)| v < magnitude);
    let (lo, hi) = if table[upper].1 == magnitude {
        (upper, upper)
    } else {
        (upper - 1, upper)
    };
    if lo == hi {
        return (sign | table[lo].0, 0);
    }
    let (below, above) = (table[lo], table[hi]);
    let choose_above = match rounding {
        0 => {
            let (d_below, d_above) = (magnitude - below.1, above.1 - magnitude);
            if d_below == d_above {
                above.0 & 1 == 0
            } else {
                d_above < d_below
            }
        }
        1 => negative, // toward -inf: magnitude grows for negatives
        2 => !negative,
        _ => false,
    };
    let result = if choose_above { above.0 } else { below.0 };
    let tiny = magnitude < 2f64.powi(-14);
    let mut flags = MXCSR_PRECISION;
    if tiny {
        flags |= MXCSR_UNDERFLOW;
    }
    (sign | result, flags)
}

#[test]
fn vcvtps2ph_rounds_like_the_reference_in_every_mode() {
    let mut rng = Rng(0xdead_beef_1234_5678);
    let mut values: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        65504.0,
        65519.9,
        65520.0,
        65536.0,
        -65520.0,
        1e10,
        6.1035156e-5,
        5.9604645e-8,
        2.9802322e-8,
        2.9802323e-8,
        1e-8,
        1.00048828125,
        1.0004883,
        0.333333333,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-40,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_4000),
    ];
    for _ in 0..400 {
        let bits = rng.next() as u32;
        let exponent = (bits >> 23) & 0xff;
        // keep exponents near the half range so most samples are interesting
        let adjusted = (bits & 0x807f_ffff) | ((100 + exponent % 50) << 23);
        values.push(f32::from_bits(adjusted));
    }
    for rounding in 0u32..4 {
        for chunk in values.chunks(8) {
            let mut source = [0u8; 32];
            for (k, v) in chunk.iter().enumerate() {
                source[4 * k..4 * k + 4].copy_from_slice(&v.to_bits().to_le_bytes());
            }
            let code = match rounding {
                0 => CVTPS2PH_Y_0,
                1 => CVTPS2PH_Y_1,
                2 => CVTPS2PH_Y_2,
                _ => CVTPS2PH_Y_3,
            };
            let h = run(code, |h| {
                set(h, 1, source);
                set(h, 0, [0xaa; 32]);
            });
            let result = get(&h, 0);
            let mut flags = 0;
            for (k, &v) in chunk.iter().enumerate() {
                let (expected, raised) = reference_single_to_half(v, rounding);
                let got = u16::from_le_bytes([result[2 * k], result[2 * k + 1]]);
                assert_eq!(
                    got,
                    expected,
                    "{v:e} ({:#010x}) mode {rounding}",
                    v.to_bits()
                );
                flags |= raised;
            }
            assert_eq!(
                h.cpu.mxcsr & 0x3f,
                flags,
                "flags for {chunk:?} mode {rounding}"
            );
            if chunk.len() == 8 {
                assert_eq!(
                    &result[16..],
                    &[0u8; 16],
                    "the upper half of the destination is zeroed"
                );
            }
        }
    }
    // imm8[2] = 1 takes the rounding mode from MXCSR.RC
    let source = from_dwords(&[1.0004883f32.to_bits()]);
    for rc in 0u32..4 {
        let h = run(CVTPS2PH_X_4, |h| {
            h.cpu.mxcsr = 0x1f80 | (rc << 13);
            set(h, 1, source);
        });
        let (expected, _) = reference_single_to_half(1.0004883, rc);
        assert_eq!(
            u16::from_le_bytes([get(&h, 0)[0], get(&h, 0)[1]]),
            expected,
            "MXCSR.RC {rc}"
        );
    }
    // memory destination: 8 bytes for the 128-bit form, 16 for the 256-bit form
    let h = run(CVTPS2PH_Y_MEM, |h| {
        set(h, 1, from_dwords(&[1.0f32.to_bits(); 8]));
        h.write(OUT, &[0xcc; 32]);
    });
    assert_eq!(h.read(OUT, 16), [0x00, 0x3c].repeat(8));
    assert_eq!(h.read(OUT + 16, 4), vec![0xcc; 4]);
    let h = run(CVTPS2PH_X_MEM, |h| {
        set(h, 1, from_dwords(&[1.0f32.to_bits(); 4]));
        h.write(OUT, &[0xcc; 32]);
    });
    assert_eq!(h.read(OUT, 8), [0x00, 0x3c].repeat(4));
    assert_eq!(h.read(OUT + 8, 4), vec![0xcc; 4]);
}

const COMPARE_ENCODINGS_PS: [&[u8]; 32] = [
    CMPPS_0_Y, CMPPS_1_Y, CMPPS_2_Y, CMPPS_3_Y, CMPPS_4_Y, CMPPS_5_Y, CMPPS_6_Y, CMPPS_7_Y,
    CMPPS_8_Y, CMPPS_9_Y, CMPPS_10_Y, CMPPS_11_Y, CMPPS_12_Y, CMPPS_13_Y, CMPPS_14_Y, CMPPS_15_Y,
    CMPPS_16_Y, CMPPS_17_Y, CMPPS_18_Y, CMPPS_19_Y, CMPPS_20_Y, CMPPS_21_Y, CMPPS_22_Y, CMPPS_23_Y,
    CMPPS_24_Y, CMPPS_25_Y, CMPPS_26_Y, CMPPS_27_Y, CMPPS_28_Y, CMPPS_29_Y, CMPPS_30_Y, CMPPS_31_Y,
];
const COMPARE_ENCODINGS_SD: [&[u8]; 32] = [
    CMPSD_0, CMPSD_1, CMPSD_2, CMPSD_3, CMPSD_4, CMPSD_5, CMPSD_6, CMPSD_7, CMPSD_8, CMPSD_9,
    CMPSD_10, CMPSD_11, CMPSD_12, CMPSD_13, CMPSD_14, CMPSD_15, CMPSD_16, CMPSD_17, CMPSD_18,
    CMPSD_19, CMPSD_20, CMPSD_21, CMPSD_22, CMPSD_23, CMPSD_24, CMPSD_25, CMPSD_26, CMPSD_27,
    CMPSD_28, CMPSD_29, CMPSD_30, CMPSD_31,
];

/// The SDM's name for each predicate, as the host-float expression it means.
fn predicate_holds(p: u8, a: f64, b: f64) -> bool {
    let unordered = a.is_nan() || b.is_nan();
    match p {
        0 | 16 => a == b,
        8 | 24 => a == b || unordered,
        1 | 17 => a < b,
        2 | 18 => a <= b,
        3 | 19 => unordered,
        4 | 20 => a != b,
        12 | 28 => !unordered && a != b,
        5 | 21 => !(a < b),
        6 | 22 => !(a <= b),
        7 | 23 => !unordered,
        9 | 25 => !(a >= b),
        10 | 26 => !(a > b),
        11 | 27 => false,
        13 | 29 => a >= b,
        14 | 30 => a > b,
        _ => true,
    }
}

/// `_S` predicates signal on quiet NaNs.
fn predicate_signals(p: u8) -> bool {
    [1, 2, 5, 6, 9, 10, 13, 14, 16, 19, 20, 23, 24, 27, 28, 31].contains(&p)
}

#[test]
fn all_32_compare_predicates_for_packed_and_scalar() {
    let values = [1.0f64, 2.0, -1.0, 0.0, -0.0, f64::INFINITY, f64::NAN];
    for p in 0u8..32 {
        for &a in &values {
            for &b in &values {
                let expected = predicate_holds(p, a, b);
                let h = run(COMPARE_ENCODINGS_SD[p as usize], |h| {
                    set(h, 1, from_qwords(&[a.to_bits(), 0x1111]));
                    set(h, 2, from_qwords(&[b.to_bits()]));
                });
                let result = get(&h, 0);
                assert_eq!(
                    qword(&result, 0),
                    if expected { u64::MAX } else { 0 },
                    "p{p} {a} {b}"
                );
                assert_eq!(
                    qword(&result, 1),
                    0x1111,
                    "scalar compare merges the upper quadword from vvvv"
                );
                let invalid = (a.is_nan() || b.is_nan()) && predicate_signals(p);
                assert_eq!(
                    h.cpu.mxcsr & MXCSR_INVALID != 0,
                    invalid,
                    "p{p} {a} {b} invalid flag"
                );
                // packed single: every lane the same comparison, in both halves
                let (fa, fb) = (a as f32, b as f32);
                let h = run(COMPARE_ENCODINGS_PS[p as usize], |h| {
                    set(h, 1, from_dwords(&[fa.to_bits(); 8]));
                    set(h, 2, from_dwords(&[fb.to_bits(); 8]));
                });
                let expected = predicate_holds(p, fa as f64, fb as f64);
                assert_eq!(
                    get(&h, 0),
                    from_dwords(&[if expected { u32::MAX } else { 0 }; 8]),
                    "ps p{p}"
                );
            }
        }
    }
}

#[test]
fn a_signaling_nan_raises_invalid_even_for_the_quiet_predicates() {
    let snan = f64::from_bits(0x7ff0_0000_0000_0001);
    let h = run(COMPARE_ENCODINGS_SD[0], |h| {
        set(h, 1, from_qwords(&[snan.to_bits()]));
        set(h, 2, from_qwords(&[1.0f64.to_bits()]));
    });
    assert_eq!(h.cpu.mxcsr & MXCSR_INVALID, MXCSR_INVALID);
}

// ---- masked moves, gathers, alignment ----

const PAGE_FAULT: u8 = 14;
/// The last page of the 4 GiB identity map ends here; anything above is not present.
const MAPPED_LIMIT: u64 = 0x1_0000_0000;

fn is_fault(stop: &Stop, vector: u8) -> bool {
    matches!(stop, Stop::Fault { vector: v, .. } if *v == vector)
}

fn sign_set(v: &Ymm, element_bytes: usize, k: usize) -> bool {
    v[k * element_bytes + element_bytes - 1] >> 7 != 0
}

#[test]
fn masked_loads_and_stores_follow_the_mask_per_element() {
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    // (load code, store code, element bytes, vector bytes)
    let loads: [(&[u8], usize, usize); 6] = [
        (MASKMOVPS_LOAD_Y, 4, 32),
        (MASKMOVPS_LOAD_X, 4, 16),
        (MASKMOVPD_LOAD_Y, 8, 32),
        (PMASKMOVD_LOAD_Y, 4, 32),
        (PMASKMOVQ_LOAD_Y, 8, 32),
        (MASKMOVPS_LOAD_Y, 4, 32),
    ];
    for (code, element, bytes) in loads {
        for _ in 0..16 {
            let (mask, data) = (rng.ymm(), rng.ymm());
            let h = run(code, |h| {
                set(h, 0, [0xaa; 32]);
                set(h, 1, mask);
                h.write(DATA, &data);
            });
            let mut expected = [0u8; 32];
            for k in 0..bytes / element {
                if sign_set(&mask, element, k) {
                    expected[k * element..(k + 1) * element]
                        .copy_from_slice(&data[k * element..(k + 1) * element]);
                }
            }
            assert_eq!(get(&h, 0), expected, "{code:02x?}");
        }
    }
    let stores: [(&[u8], usize, usize); 4] = [
        (MASKMOVPS_STORE_Y, 4, 32),
        (MASKMOVPD_STORE_X, 8, 16),
        (PMASKMOVD_STORE_X, 4, 16),
        (PMASKMOVQ_STORE_Y, 8, 32),
    ];
    for (code, element, bytes) in stores {
        for _ in 0..16 {
            let (mask, value) = (rng.ymm(), rng.ymm());
            let h = run(code, |h| {
                set(h, 0, value);
                set(h, 1, mask);
                h.write(OUT, &[0xcc; 64]);
            });
            let mut expected = [0xccu8; 64];
            for k in 0..bytes / element {
                if sign_set(&mask, element, k) {
                    expected[k * element..(k + 1) * element]
                        .copy_from_slice(&value[k * element..(k + 1) * element]);
                }
            }
            assert_eq!(h.read(OUT, 64), expected.to_vec(), "{code:02x?}");
        }
    }
}

#[test]
fn masked_out_elements_never_fault() {
    // a 32-byte operand whose upper half lies beyond the mapped address space
    let address = MAPPED_LIMIT - 16;
    let mut low_only = [0u8; 32];
    low_only[3] = 0x80;
    let mut upper_active = low_only;
    upper_active[19] = 0x80;
    let (mut ok, mut faulting) = (harness(MASKMOVPS_LOAD_Y), harness(MASKMOVPS_LOAD_Y));
    for (h, mask) in [(&mut ok, low_only), (&mut faulting, upper_active)] {
        h.cpu.gpr[RSI] = address;
        set(h, 1, mask);
    }
    assert_eq!(ok.run(10), Stop::Halted);
    let stop = faulting.run(10);
    assert!(is_fault(&stop, PAGE_FAULT), "{stop:?}");
    assert_eq!(
        get(&faulting, 0),
        [0u8; 32],
        "the destination is untouched by the fault"
    );
    // stores the same way
    let mut ok = harness(MASKMOVPS_STORE_Y);
    ok.cpu.gpr[RDI] = address;
    set(&mut ok, 1, low_only);
    assert_eq!(ok.run(10), Stop::Halted);
}

fn gather_reference(
    memory: &[u8],
    base: u64,
    dest: &Ymm,
    mask: &Ymm,
    indices: &Ymm,
    element: usize,
    index_bytes: usize,
    scale: u64,
    displacement: u64,
    count: usize,
) -> Ymm {
    let mut out = *dest;
    for k in 0..count {
        if !sign_set(mask, element, k) {
            continue;
        }
        let raw: i64 = match index_bytes {
            4 => dword(indices, k) as i32 as i64,
            _ => qword(indices, k) as i64,
        };
        let address = base
            .wrapping_add((raw as u64).wrapping_mul(scale))
            .wrapping_add(displacement);
        let offset = (address - DATA) as usize;
        out[k * element..(k + 1) * element].copy_from_slice(&memory[offset..offset + element]);
    }
    // everything above the gathered elements is zeroed
    for byte in &mut out[count * element..] {
        *byte = 0;
    }
    out
}

#[test]
fn gathers_load_unmasked_elements_and_clear_the_mask() {
    // (code, element bytes, index bytes, scale, displacement, vector bytes, mask reg, index reg)
    type Case = (&'static [u8], usize, usize, u64, u64, usize, usize);
    let cases: [Case; 9] = [
        (GATHER_DD_Y, 4, 4, 4, 0, 32, 2),
        (GATHER_DD_X, 4, 4, 4, 0, 16, 2),
        (GATHER_DQ_Y, 8, 4, 8, 0, 32, 2),
        (GATHER_QD_Y, 4, 8, 4, 0, 32, 2),
        (GATHER_QD_X, 4, 8, 4, 0, 16, 2),
        (GATHER_QQ_Y, 8, 8, 8, 0, 32, 2),
        (GATHER_DPS_X4, 4, 4, 4, 0, 16, 4),
        (GATHER_QPD_Y, 8, 8, 8, 0, 32, 2),
        (GATHER_DPS_Y_HIGH, 4, 4, 2, 8, 32, 12),
    ];
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    for (code, element, index_bytes, scale, displacement, vector, index_register) in cases {
        for _ in 0..24 {
            let mut memory = vec![0u8; 1024];
            for byte in &mut memory {
                *byte = rng.next() as u8;
            }
            // small indices (some negative-free), so every address stays inside the table
            let limit = 400 / scale;
            let mut indices = [0u8; 32];
            for k in 0..32 / index_bytes {
                let value = rng.next() % limit;
                indices[k * index_bytes..(k + 1) * index_bytes]
                    .copy_from_slice(&value.to_le_bytes()[..index_bytes]);
            }
            let (dest, mask) = (rng.ymm(), rng.ymm());
            let count = vector / element.max(index_bytes);
            let h = run(code, |h| {
                h.write(DATA, &memory);
                set(h, 0, dest);
                set(h, 1, mask);
                set(h, index_register, indices);
            });
            let expected = gather_reference(
                &memory,
                DATA,
                &dest,
                &mask,
                &indices,
                element,
                index_bytes,
                scale,
                displacement,
                count,
            );
            assert_eq!(get(&h, 0), expected, "{code:02x?} destination");
            assert_eq!(
                get(&h, 1),
                [0u8; 32],
                "{code:02x?}: the mask register is cleared"
            );
            assert_eq!(
                get(&h, index_register),
                indices,
                "{code:02x?}: the index register is untouched"
            );
        }
    }
}

#[test]
fn a_gather_with_overlapping_registers_is_undefined() {
    let mut h = harness(GATHER_DD_SAME);
    assert!(matches!(h.run(10), Stop::Fault { vector: 6, .. }));
}

#[test]
fn a_faulting_gather_leaves_completed_elements_and_clears_their_mask_bits() {
    let mut h = harness(GATHER_DD_Y);
    let mut memory = [0u8; 64];
    for (k, byte) in memory.iter_mut().enumerate() {
        *byte = k as u8;
    }
    h.write(DATA, &memory);
    // element 3 points far beyond the mapped space
    let mut indices = from_dwords(&[0, 1, 2, 0x4000_0000, 4, 5, 6, 7]);
    indices[15] = 0x40;
    set(&mut h, 2, indices);
    set(&mut h, 1, from_dwords(&[0x8000_0000; 8]));
    set(&mut h, 0, [0xee; 32]);
    let stop = h.run(10);
    assert!(is_fault(&stop, PAGE_FAULT), "{stop:?}");
    let dest = get(&h, 0);
    assert_eq!(&dword(&dest, 0).to_le_bytes(), &memory[0..4]);
    assert_eq!(&dword(&dest, 2).to_le_bytes(), &memory[8..12]);
    assert_eq!(
        dword(&dest, 3),
        0xeeee_eeee,
        "the faulting element is not loaded"
    );
    let mask = get(&h, 1);
    assert_eq!((dword(&mask, 0), dword(&mask, 2)), (0, 0));
    assert_eq!(
        dword(&mask, 3),
        u32::MAX,
        "its mask bit stays set so the instruction can be restarted"
    );
}

#[test]
fn explicitly_aligned_256_bit_moves_need_32_byte_alignment() {
    for code in [MOVAPS_Y_LOAD, MOVDQA_Y_LOAD, MOVNTDQA_Y] {
        for offset in [0u64, 16, 8, 1] {
            let mut h = harness(code);
            h.cpu.gpr[RSI] = DATA + offset;
            let stop = h.run(10);
            if offset == 0 {
                assert_eq!(stop, Stop::Halted, "{code:02x?}");
            } else {
                assert!(is_fault(&stop, 13), "{code:02x?} offset {offset}: {stop:?}");
            }
        }
    }
    for code in [MOVAPS_Y_STORE, MOVNTDQ_Y_STORE] {
        let mut h = harness(code);
        h.cpu.gpr[RDI] = OUT + 16;
        assert!(is_fault(&h.run(10), 13));
    }
    // unaligned forms and 128-bit aligned moves are unaffected
    for code in [MOVUPS_Y_LOAD, MOVDQU_Y_LOAD] {
        let mut h = harness(code);
        h.cpu.gpr[RSI] = DATA + 1;
        assert_eq!(h.run(10), Stop::Halted);
    }
    let mut h = harness(MOVAPS_X_LOAD);
    h.cpu.gpr[RSI] = DATA + 16;
    assert_eq!(h.run(10), Stop::Halted);
}

#[test]
fn vpcmpistrm_zeroes_the_upper_half_of_xmm0() {
    let mut h = harness(PCMPISTRM_X);
    h.cpu.ymm_hi[0] = u128::MAX;
    h.cpu.xmm[1] = u128::from_le_bytes(*b"abcdefghabcdefgh");
    h.cpu.xmm[2] = u128::from_le_bytes(*b"abcdefghabcdefgh");
    assert_eq!(h.run(10), Stop::Halted);
    assert_eq!(h.cpu.ymm_hi[0], 0);
}

/// AVX-VNNI: the VEX encodings of the VNNI dot products. Encodings from `hack/asm.sh` (`{vex}`
/// selects VEX over EVEX); the reference is the SDM's per-dword sum.
mod avx_vnni {
    use super::*;

    /// {vex} vpdpbusd / vpdpbusds / vpdpwssd / vpdpwssds ymm0, ymm1, ymm2
    const VPDPBUSD_Y: &[u8] = &[0xc4, 0xe2, 0x75, 0x50, 0xc2, 0xf4];
    const VPDPBUSDS_Y: &[u8] = &[0xc4, 0xe2, 0x75, 0x51, 0xc2, 0xf4];
    const VPDPWSSD_Y: &[u8] = &[0xc4, 0xe2, 0x75, 0x52, 0xc2, 0xf4];
    const VPDPWSSDS_Y: &[u8] = &[0xc4, 0xe2, 0x75, 0x53, 0xc2, 0xf4];
    /// {vex} vpdpbusd xmm0, xmm1, xmm2
    const VPDPBUSD_X: &[u8] = &[0xc4, 0xe2, 0x71, 0x50, 0xc2, 0xf4];
    /// {vex} vpdpwssds xmm0, xmm1, xmmword ptr [rsi]
    const VPDPWSSDS_XM: &[u8] = &[0xc4, 0xe2, 0x71, 0x53, 0x06, 0xf4];
    /// {vex} vpdpbusd ymm0, ymm1, ymmword ptr [rsi]
    const VPDPBUSD_YM: &[u8] = &[0xc4, 0xe2, 0x75, 0x50, 0x06, 0xf4];
    /// {vex} vpdpbusd ymm1, ymm1, ymm1: every operand the same register
    const VPDPBUSD_Y_SAME: &[u8] = &[0xc4, 0xe2, 0x75, 0x50, 0xc9, 0xf4];
    /// VPDPBUSD_Y with VEX.W = 1, which AVX-VNNI does not define
    const VPDPBUSD_Y_W1: &[u8] = &[0xc4, 0xe2, 0xf5, 0x50, 0xc2, 0xf4];

    /// The SDM's VPDPBUSD/VPDPWSSD: per dword, the accumulator plus four products of unsigned
    /// bytes of `first` with signed bytes of `second` (or two products of signed words).
    fn reference(opcode: u8, accumulator: &Ymm, first: &Ymm, second: &Ymm, dwords: usize) -> Ymm {
        let mut out = [0u32; 8];
        for k in 0..dwords {
            let mut sum = dword(accumulator, k) as i32 as i64;
            if opcode >= 0x52 {
                for j in 0..2 {
                    let at = 4 * k + 2 * j;
                    let a = i16::from_le_bytes([first[at], first[at + 1]]) as i64;
                    let b = i16::from_le_bytes([second[at], second[at + 1]]) as i64;
                    sum += a * b;
                }
            } else {
                for j in 0..4 {
                    sum += first[4 * k + j] as i64 * second[4 * k + j] as i8 as i64;
                }
            }
            out[k] = if opcode & 1 == 1 {
                sum.clamp(i32::MIN as i64, i32::MAX as i64) as u32
            } else {
                sum as u32
            };
        }
        from_dwords(&out[..dwords])
    }

    fn inputs_with_extremes() -> Vec<(Ymm, Ymm, Ymm)> {
        let mut rng = Rng(0x5eed_0fa7_7e11);
        let mut cases: Vec<(Ymm, Ymm, Ymm)> =
            (0..8).map(|_| (rng.ymm(), rng.ymm(), rng.ymm())).collect();
        // accumulators at the i32 limits with the largest products of either sign, so the
        // saturating forms clamp and the others wrap
        let near_max = from_dwords(&[0x7fff_fff0; 8]);
        let near_min = from_dwords(&[0x8000_0010; 8]);
        cases.push((near_max, [0xff; 32], [0x7f; 32]));
        cases.push((near_min, [0xff; 32], [0x80; 32]));
        cases.push((near_max, [0x80; 32], [0x80; 32]));
        cases
    }

    #[test]
    fn vex_dot_products_accumulate_into_the_destination() {
        for (accumulator, first, second) in inputs_with_extremes() {
            for (code, opcode) in [
                (VPDPBUSD_Y, 0x50),
                (VPDPBUSDS_Y, 0x51),
                (VPDPWSSD_Y, 0x52),
                (VPDPWSSDS_Y, 0x53),
            ] {
                let h = run(code, |h| {
                    set(h, 0, accumulator);
                    set(h, 1, first);
                    set(h, 2, second);
                });
                assert_eq!(
                    get(&h, 0),
                    reference(opcode, &accumulator, &first, &second, 8),
                    "{code:02x?}"
                );
            }
        }
    }

    #[test]
    fn the_128_bit_form_zeroes_the_upper_half_and_memory_sources_work() {
        for (accumulator, first, second) in inputs_with_extremes() {
            let h = run(VPDPBUSD_X, |h| {
                set(h, 0, accumulator);
                set(h, 1, first);
                set(h, 2, second);
            });
            assert_eq!(
                get(&h, 0),
                zero_upper(reference(0x50, &accumulator, &first, &second, 4))
            );
            let h = run(VPDPWSSDS_XM, |h| {
                set(h, 0, accumulator);
                set(h, 1, first);
                h.write(DATA, &second[..16]);
            });
            assert_eq!(
                get(&h, 0),
                zero_upper(reference(0x53, &accumulator, &first, &second, 4))
            );
            let h = run(VPDPBUSD_YM, |h| {
                set(h, 0, accumulator);
                set(h, 1, first);
                h.write(DATA, &second);
            });
            assert_eq!(
                get(&h, 0),
                reference(0x50, &accumulator, &first, &second, 8)
            );
        }
    }

    #[test]
    fn one_register_as_every_operand_reads_it_before_writing_it() {
        let value = Rng(0xa11a5).ymm();
        let h = run(VPDPBUSD_Y_SAME, |h| set(h, 1, value));
        assert_eq!(get(&h, 1), reference(0x50, &value, &value, &value, 8));
    }

    #[test]
    fn vex_w1_is_invalid() {
        let mut h = harness(VPDPBUSD_Y_W1);
        assert_eq!(
            h.run(10),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
    }

    #[test]
    fn avx_vnni_runs_identically_under_the_jit() {
        let (accumulator, first, second) = inputs_with_extremes()[0];
        for code in [
            VPDPBUSD_Y,
            VPDPWSSDS_Y,
            VPDPBUSD_X,
            VPDPBUSD_YM,
            VPDPBUSD_Y_W1,
        ] {
            crate::testkit::assert_jit_matches_interpreter(code, |h| {
                h.cpu.cr4 = (h.cpu.cr4 & !(1 << 9)) | CR4_OSXSAVE;
                h.cpu.xcr0 = XCR0_X87 | XCR0_SSE | XCR0_AVX;
                h.cpu.gpr[RSI] = DATA;
                set(h, 0, accumulator);
                set(h, 1, first);
                set(h, 2, second);
                h.write(DATA, &second);
            });
        }
    }
}
