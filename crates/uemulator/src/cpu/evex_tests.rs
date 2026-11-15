//! Tests for the AVX-512 state, the EVEX decoder, write masking and the instructions in `evex.rs`,
//! `avx512_data.rs` and `avx512_mask.rs`. Encodings come from the assembler
//! (`hack/gen_avx512.py` -> `avx512_encodings.rs`); the expectations are plain loops over elements
//! written from the SDM, independent of the implementation.

// The reference models index several parallel arrays by element number, spell out the SDM's
// range checks and write float constants with every digit the reference defines.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::excessive_precision
)]

use super::*;
use crate::cpu::decode::decode;
use crate::cpu::test_support::unmap_boot_identity_2m;
use crate::cpu::xsave::{XCR0_SUPPORTED, XCR0_X87};
use crate::testkit::{Harness, RDI, RSI, SCRATCH, Stop};

// One shape per line is the point of the generated table: keep rustfmt out of it.
#[rustfmt::skip]
#[path = "avx512_encodings.rs"]
pub(in crate::cpu) mod encodings;
pub(in crate::cpu) use encodings::*;

pub(in crate::cpu) const DATA: u64 = SCRATCH + 0x1000;
pub(in crate::cpu) const OUT: u64 = SCRATCH + 0x2000;
/// A 2 MiB region the page-fault tests unmap, and an address 32 bytes before it: a 64-byte access
/// there has its low half mapped and its high half faulting.
const UNMAPPED: u64 = 0x180_0000;
const EDGE: u64 = UNMAPPED - 32;
pub(in crate::cpu) const LENGTHS: [usize; 3] = [16, 32, 64];

pub(in crate::cpu) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn vector(&mut self) -> Vector {
        core::array::from_fn(|_| (self.next() as u128) << 64 | self.next() as u128)
    }
}

pub(in crate::cpu) fn bytes_of(v: &Vector) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (k, lane) in v.iter().enumerate() {
        out[16 * k..16 * k + 16].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

pub(in crate::cpu) fn vector_of(bytes: &[u8]) -> Vector {
    let mut padded = [0u8; 64];
    padded[..bytes.len()].copy_from_slice(bytes);
    core::array::from_fn(|k| u128::from_le_bytes(padded[16 * k..16 * k + 16].try_into().unwrap()))
}

pub(in crate::cpu) fn rig(code: &[u8]) -> Harness {
    let mut h = Harness::new(code);
    h.cpu.cr4 = (h.cpu.cr4 & !(1 << 9)) | CR4_OSXSAVE;
    h.cpu.xcr0 = XCR0_SUPPORTED;
    h.cpu.gpr[RSI] = DATA;
    h.cpu.gpr[RDI] = OUT;
    h
}

pub(in crate::cpu) fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
    let mut h = rig(code);
    setup(&mut h);
    let stop = h.run(100);
    assert_eq!(stop, Stop::Halted, "{code:02x?}: {}", h.cpu.dump_regs());
    h
}

pub(in crate::cpu) fn stop_of(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Stop {
    let mut h = rig(code);
    setup(&mut h);
    h.run(100)
}

/// The expected result of an element-wise instruction: for each element of the vector length,
/// `f(first, second)` under the write mask, merged into `old` or zeroed; zero above the length.
pub(in crate::cpu) fn expected(
    length: usize,
    element: usize,
    mask: Option<(u64, bool)>,
    old: &Vector,
    mut f: impl FnMut(usize) -> u64,
) -> Vector {
    let mut out = [0u128; 4];
    for k in 0..length / element {
        let value = match mask {
            Some((bits, _)) if bits >> k & 1 == 0 => {
                if mask.unwrap().1 {
                    0
                } else {
                    super::element(old, element, k)
                }
            }
            _ => f(k),
        };
        set_element(&mut out, element, k, value);
    }
    out
}

/// What an instruction form is checked against.
pub(in crate::cpu) struct Spec {
    /// Mask granularity in bytes.
    pub element: usize,
    /// Broadcast element size in bytes, 0 when the form has none.
    pub broadcast: usize,
    pub unary: bool,
}

/// Runs every shape of `variants` on random data and compares with `f(a, b)` per element.
pub(in crate::cpu) fn check_elementwise(
    name: &str,
    variants: &Variants,
    spec: Spec,
    f: impl Fn(u64, u64) -> u64,
) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ name.len() as u64);
    for trial in 0..8 {
        let (a, b, old) = (rng.vector(), rng.vector(), rng.vector());
        let k1 = rng.next();
        for (index, length) in LENGTHS.into_iter().enumerate() {
            for (shape, codes) in [
                ("plain", &variants.p),
                ("merge", &variants.k),
                ("zero", &variants.kz),
                ("memory", &variants.m),
                ("memory merge", &variants.mk),
                ("broadcast", &variants.b),
                ("broadcast merge", &variants.bk),
            ] {
                let code = codes[index];
                if code.is_empty() {
                    continue;
                }
                let masked = matches!(shape, "merge" | "zero" | "memory merge" | "broadcast merge");
                let zeroing = shape == "zero";
                let from_memory = shape.starts_with("memory") || shape.starts_with("broadcast");
                let broadcast = shape.starts_with("broadcast");
                let h = run(code, |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(1, a);
                    h.cpu.set_zmm(2, b);
                    h.cpu.opmask[1] = k1;
                    h.write(DATA, &bytes_of(&b));
                });
                let size = spec.element;
                let source = if broadcast {
                    let e = super::element(&b, spec.broadcast, 0);
                    let mut v = [0u128; 4];
                    for k in 0..64 / spec.broadcast {
                        set_element(&mut v, spec.broadcast, k, e);
                    }
                    v
                } else {
                    b
                };
                let _ = from_memory;
                let want = expected(length, size, masked.then_some((k1, zeroing)), &old, |k| {
                    f(
                        super::element(&a, size, k),
                        super::element(&source, size, k),
                    )
                });
                let got = h.cpu.zmm(0);
                assert_eq!(
                    bytes_of(&got),
                    bytes_of(&want),
                    "{name} {shape} {length} bytes, trial {trial}"
                );
                if spec.unary {
                    assert_eq!(h.cpu.zmm(1), a, "{name}: first source unchanged");
                }
            }
        }
        // ZMM16-31 operands
        let h = run(variants.high, |h| {
            h.cpu.set_zmm(20, old);
            h.cpu.set_zmm(21, a);
            h.cpu.set_zmm(22, b);
        });
        let want = expected(64, spec.element, None, &old, |k| {
            f(
                super::element(&a, spec.element, k),
                super::element(&b, spec.element, k),
            )
        });
        assert_eq!(
            bytes_of(&h.cpu.zmm(20)),
            bytes_of(&want),
            "{name} high registers"
        );
        assert_eq!(
            h.cpu.zmm(0),
            [0; 4],
            "{name}: zmm0 untouched by the high form"
        );
    }
}

pub(in crate::cpu) fn spec(element: usize, broadcast: usize) -> Spec {
    Spec {
        element,
        broadcast,
        unary: false,
    }
}

pub(in crate::cpu) fn unary_spec(element: usize, broadcast: usize) -> Spec {
    Spec {
        element,
        broadcast,
        unary: true,
    }
}

#[path = "evex_fp_tests.rs"]
mod floating;
#[path = "evex_int_tests.rs"]
mod integer;
#[path = "evex_lane_tests.rs"]
mod lanes;
#[path = "evex_perm_tests.rs"]
mod permutation;

mod decoding {
    use super::*;

    fn decoded(code: &[u8]) -> crate::cpu::decode::Insn {
        decode(code).expect("decodes")
    }

    #[test]
    fn registers_above_15_come_from_r_prime_v_prime_and_x() {
        let i = decoded(VPADDD.high);
        assert!(i.evex && !i.vex);
        assert_eq!((i.reg, i.vex_v, i.rm), (20, 21, 22));
        assert_eq!(i.evex_ll, 2);
        assert_eq!(i.op, EVEX_OP_FLAG | 0x1fe);
    }

    #[test]
    fn mask_zeroing_length_and_broadcast_fields() {
        let i = decoded(VPADDD.kz[2]);
        assert_eq!(
            (i.evex_mask, i.evex_zeroing, i.evex_b, i.evex_ll),
            (1, true, false, 2)
        );
        let i = decoded(VPADDD.bk[1]);
        assert_eq!(
            (i.evex_mask, i.evex_zeroing, i.evex_b, i.evex_ll),
            (1, false, true, 1)
        );
        let i = decoded(VPADDD.p[0]);
        assert_eq!((i.evex_mask, i.evex_b, i.evex_ll), (0, false, 0));
    }

    #[test]
    fn compressed_displacement_is_left_raw_for_the_executor() {
        let i = decoded(VPADDD_DISP8);
        assert!(i.evex_disp8);
        assert_eq!(i.disp, 2, "128 / 64");
        let i = decoded(VPADDD_DISP8_NEGATIVE);
        assert_eq!(i.disp, -1);
        let i = decoded(VPADDD.m[2]);
        assert!(!i.evex_disp8, "no displacement, no scaling");
    }

    #[test]
    fn rounding_control_rides_in_the_length_field() {
        let i = decoded(VADDPS_RD_SAE);
        assert!(i.evex_b && i.md == 3);
        assert_eq!(i.evex_ll, 1, "round down");
    }

    #[test]
    fn instruction_length_includes_the_four_byte_prefix() {
        for code in [VPADDD.p[2], VPADDD.m[2], VPADDD_DISP8] {
            let i = decoded(code);
            assert_eq!(i.len as usize, code.len() - 1, "all but the trailing hlt");
        }
    }

    #[test]
    fn reserved_prefix_bits_are_invalid() {
        let mut code = VPADDD.p[2].to_vec();
        code[1] |= 0x08;
        assert!(decode(&code).is_err(), "P0 bit 3 must be zero");
        let mut code = VPADDD.p[2].to_vec();
        code[2] &= !0x04;
        assert!(decode(&code).is_err(), "P1 bit 2 must be one");
        let mut code = VPADDD.p[2].to_vec();
        code[1] = (code[1] & !7) | 5;
        assert!(decode(&code).is_err(), "map 5 (FP16) is not supported");
    }

    #[test]
    fn legacy_prefixes_before_evex_are_invalid() {
        for prefix in [0x66u8, 0xf2, 0xf3, 0xf0, 0x48] {
            let mut code = vec![prefix];
            code.extend_from_slice(VPADDD.p[2]);
            assert!(decode(&code).is_err(), "{prefix:#x} before EVEX");
        }
    }

    #[test]
    fn vsib_index_takes_v_prime_for_registers_above_15() {
        let i = decoded(VPGATHERDD_HIGH_INDEX);
        assert_eq!(i.index, 20);
        assert_eq!(i.base, 6);
        assert_eq!(i.scale, 2);
    }
}

mod state {
    use super::*;

    #[test]
    fn registers_round_trip_across_the_split_storage() {
        let mut h = rig(&[0xf4]);
        let mut rng = Rng(1);
        for n in 0..32 {
            h.cpu.set_zmm(n, rng.vector());
        }
        let mut rng = Rng(1);
        for n in 0..32 {
            assert_eq!(h.cpu.zmm(n), rng.vector(), "zmm{n}");
        }
    }

    #[test]
    fn element_helpers_address_every_size() {
        let mut v = [0u128; 4];
        for size in [1usize, 2, 4, 8] {
            for k in 0..64 / size {
                set_element(&mut v, size, k, (k as u64 + 1) * 0x0101_0101_0101_0101);
            }
            for k in 0..64 / size {
                assert_eq!(
                    super::element(&v, size, k),
                    ((k as u64 + 1) * 0x0101_0101_0101_0101) & size_mask(size)
                );
            }
        }
    }

    #[test]
    fn xcr0_accepts_the_avx512_components_only_together_and_with_avx() {
        let write = |value: u64| {
            let mut h = rig(&[0x0f, 0x01, 0xd1, 0xf4]);
            h.cpu.gpr[1] = 0;
            h.cpu.gpr[0] = value & 0xffff_ffff;
            h.cpu.gpr[2] = value >> 32;
            (h.run(10), h.cpu.xcr0)
        };
        let base = XCR0_X87 | XCR0_SSE | XCR0_AVX;
        assert_eq!(
            write(base | XCR0_AVX512),
            (Stop::Halted, base | XCR0_AVX512)
        );
        for partial in [1 << 5, 1 << 6, 1 << 7, (1 << 5) | (1 << 6)] {
            assert!(matches!(
                write(base | partial).0,
                Stop::Fault { vector: 13, .. }
            ));
        }
        let no_avx = XCR0_X87 | XCR0_SSE | XCR0_AVX512;
        assert!(matches!(write(no_avx).0, Stop::Fault { vector: 13, .. }));
    }

    #[test]
    fn evex_needs_all_five_state_bits_and_osxsave() {
        let code = VPADDD.p[2];
        let ud = Stop::Fault {
            vector: 6,
            error_code: None,
            software: false,
        };
        for missing in [XCR0_SSE, XCR0_AVX, 1 << 5, 1 << 6, 1 << 7] {
            assert_eq!(
                stop_of(code, |h| h.cpu.xcr0 = XCR0_SUPPORTED & !missing),
                ud,
                "XCR0 without {missing:#x}"
            );
        }
        assert_eq!(stop_of(code, |h| h.cpu.cr4 &= !CR4_OSXSAVE), ud);
        let nm = Stop::Fault {
            vector: 7,
            error_code: None,
            software: false,
        };
        assert_eq!(stop_of(code, |h| h.cpu.cr0 |= CR0_TS), nm);
    }

    #[test]
    fn vex_writes_zero_bits_511_to_256_and_vzeroupper_spares_zmm16_and_up() {
        // vpaddd xmm0, xmm1, xmm2 (VEX.128)
        let h = run(&[0xc5, 0xf1, 0xfe, 0xc2, 0xf4], |h| {
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        let v = h.cpu.zmm(0);
        assert_eq!(
            &v[1..],
            &[0, 0, 0],
            "VEX.128 clears everything above bit 127"
        );
        // vpaddd ymm0, ymm1, ymm2 (VEX.256)
        let h = run(&[0xc5, 0xf5, 0xfe, 0xc2, 0xf4], |h| {
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        assert_eq!(&h.cpu.zmm(0)[2..], &[0, 0], "VEX.256 clears bits 511:256");
        // vzeroupper
        let h = run(&[0xc5, 0xf8, 0x77, 0xf4], |h| {
            for n in [0, 15, 16, 31] {
                h.cpu.set_zmm(n, [u128::MAX; 4]);
            }
        });
        for n in [0, 15] {
            assert_eq!(h.cpu.zmm(n), [u128::MAX, 0, 0, 0], "zmm{n}");
        }
        for n in [16, 31] {
            assert_eq!(h.cpu.zmm(n), [u128::MAX; 4], "zmm{n} is not affected");
        }
        // vzeroall (VEX.256): all of zmm0-15 become zero, the high bank stays
        let h = run(&[0xc5, 0xfc, 0x77, 0xf4], |h| {
            for n in [0, 15, 16] {
                h.cpu.set_zmm(n, [u128::MAX; 4]);
            }
        });
        assert_eq!(h.cpu.zmm(0), [0; 4]);
        assert_eq!(h.cpu.zmm(15), [0; 4]);
        assert_eq!(h.cpu.zmm(16), [u128::MAX; 4]);
    }

    #[test]
    fn vex_instructions_with_a_general_register_source_do_not_treat_it_as_a_vector_register() {
        // rax is general register 0 and the destination is xmm0: the staging code must not take
        // the r/m operand for a colliding vector register and redirect it to a scratch copy
        let a = [0x1111_2222_3333_4444_5555_6666_7777_8888u128, 0, 0, 0];
        let h = run(VEX_PINSRD_EAX, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.gpr[0] = 0xcafe_f00d;
        });
        assert_eq!(h.cpu.zmm(0)[0], 0x1111_2222_3333_4444_cafe_f00d_7777_8888);
        let h = run(VEX_PINSRQ_RAX, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.gpr[0] = 0x0123_4567_89ab_cdef;
        });
        assert_eq!(h.cpu.zmm(0)[0], 0x0123_4567_89ab_cdef_5555_6666_7777_8888);
        let h = run(VEX_PINSRW_EAX, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.gpr[0] = 0xbeef;
        });
        assert_eq!(h.cpu.zmm(0)[0], 0x1111_2222_3333_4444_5555_beef_7777_8888);
        let h = run(VEX_PINSRB_EAX, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.gpr[0] = 0xab;
        });
        assert_eq!(h.cpu.zmm(0)[0], 0x1111_2222_3333_4444_5555_6666_ab77_8888);
        let h = run(VEX_CVTSI2SD_RAX, |h| {
            h.cpu.set_zmm(1, [7 << 64, 0, 0, 0]);
            h.cpu.gpr[0] = (-3i64) as u64;
        });
        assert_eq!(h.cpu.zmm(0)[0], 7 << 64 | (-3.0f64).to_bits() as u128);
    }

    #[test]
    fn legacy_sse_leaves_the_upper_bits_alone() {
        // paddd xmm0, xmm1
        let h = run(&[0x66, 0x0f, 0xfe, 0xc1, 0xf4], |h| {
            h.cpu.set_zmm(0, [1, 2, 3, 4]);
            h.cpu.set_zmm(1, [1, 0, 0, 0]);
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
        });
        assert_eq!(h.cpu.zmm(0), [2, 2, 3, 4]);
    }
}

mod xsave_state {
    use super::*;

    const AREA: u64 = SCRATCH + 0x4000;
    const XSAVE_RDI: &[u8] = &[0x0f, 0xae, 0x27, 0xf4];
    const XRSTOR_RDI: &[u8] = &[0x0f, 0xae, 0x2f, 0xf4];
    const XSAVEOPT_RDI: &[u8] = &[0x0f, 0xae, 0x37, 0xf4];
    const ALL: u64 = XCR0_SUPPORTED;

    fn full_state(h: &mut Harness, seed: u64) {
        let mut rng = Rng(seed);
        for n in 0..32 {
            h.cpu.set_zmm(n, rng.vector());
        }
        for k in 0..8 {
            h.cpu.opmask[k] = rng.next();
        }
    }

    fn save(code: &[u8], seed: u64) -> Harness {
        run(code, |h| {
            h.cpu.gpr[RDI] = AREA;
            h.cpu.gpr[0] = ALL;
            h.cpu.gpr[2] = 0;
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            full_state(h, seed);
        })
    }

    #[test]
    fn xsave_writes_components_at_the_cpuid_offsets() {
        let h = save(XSAVE_RDI, 7);
        let mut want = Harness::new(&[0xf4]);
        full_state(&mut want, 7);
        let read = |at: u64, len: usize| h.read(AREA + at, len);
        // AVX: YMM_Hi128 at 576, 16 bytes each
        for n in 0..16 {
            assert_eq!(
                read(576 + 16 * n as u64, 16),
                want.cpu.ymm_hi[n].to_le_bytes()
            );
        }
        // opmask at 1088, eight bytes each
        for k in 0..8 {
            assert_eq!(
                read(1088 + 8 * k as u64, 8),
                want.cpu.opmask[k].to_le_bytes()
            );
        }
        // ZMM_Hi256 at 1152, 32 bytes per register
        for n in 0..16 {
            let mut expect = want.cpu.zmm_hi[n][0].to_le_bytes().to_vec();
            expect.extend(want.cpu.zmm_hi[n][1].to_le_bytes());
            assert_eq!(read(1152 + 32 * n as u64, 32), expect, "zmm{n} high");
        }
        // Hi16_ZMM at 1664, 64 bytes per register
        for n in 0..16 {
            let mut expect = Vec::new();
            for lane in want.cpu.zmm_extra[n] {
                expect.extend(lane.to_le_bytes());
            }
            assert_eq!(read(1664 + 64 * n as u64, 64), expect, "zmm{}", 16 + n);
        }
        assert_eq!(
            h.read_u64(AREA + 512) & ALL,
            ALL,
            "XSTATE_BV marks everything in use"
        );
    }

    #[test]
    fn xrstor_restores_what_xsave_saved() {
        let mut saved = save(XSAVE_RDI, 11);
        let area = saved.read(AREA, 2688);
        let h = run(XRSTOR_RDI, |h| {
            h.cpu.gpr[RDI] = AREA;
            h.cpu.gpr[0] = ALL;
            h.cpu.gpr[2] = 0;
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            h.write(AREA, &area);
            full_state(h, 99);
        });
        let mut want = Harness::new(&[0xf4]);
        full_state(&mut want, 11);
        for n in 0..32 {
            assert_eq!(h.cpu.zmm(n), want.cpu.zmm(n), "zmm{n}");
        }
        assert_eq!(h.cpu.opmask, want.cpu.opmask);
        let _ = &mut saved;
    }

    #[test]
    fn xrstor_with_a_clear_state_bit_resets_the_component_to_its_initial_state() {
        let h = run(XRSTOR_RDI, |h| {
            h.cpu.gpr[RDI] = AREA;
            h.cpu.gpr[0] = ALL;
            h.cpu.gpr[2] = 0;
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            h.write(AREA, &[0u8; 2688]);
            full_state(h, 5);
        });
        for n in 0..32 {
            assert_eq!(h.cpu.zmm(n), [0; 4], "zmm{n} initialised");
        }
        assert_eq!(h.cpu.opmask, [0; 8]);
    }

    #[test]
    fn requested_feature_bitmap_selects_components() {
        // only the opmask component (bit 5) requested
        let h = run(XSAVE_RDI, |h| {
            h.cpu.gpr[RDI] = AREA;
            h.cpu.gpr[0] = 1 << 5;
            h.cpu.gpr[2] = 0;
            h.write(AREA, &[0xaa; 2688]);
            full_state(h, 3);
        });
        assert_eq!(
            h.read(AREA + 576, 16),
            vec![0xaa; 16],
            "AVX component untouched"
        );
        assert_eq!(
            h.read(AREA + 1152, 32),
            vec![0xaa; 32],
            "ZMM_Hi256 untouched"
        );
        assert_ne!(h.read_u64(AREA + 1088), 0xaaaa_aaaa_aaaa_aaaa);
        assert_eq!(
            h.read_u64(AREA + 512),
            0xaaaa_aaaa_aaaa_aaaa | 1 << 5,
            "other XSTATE_BV bits are kept"
        );
    }

    #[test]
    fn xsaveopt_saves_the_same_state() {
        let a = save(XSAVE_RDI, 21);
        let b = save(XSAVEOPT_RDI, 21);
        assert_eq!(a.read(AREA, 2688), b.read(AREA, 2688));
    }

    #[test]
    fn area_size_follows_the_enabled_components() {
        use crate::cpu::xsave::enabled_area_size;
        assert_eq!(enabled_area_size(XCR0_X87 | XCR0_SSE), 576);
        assert_eq!(enabled_area_size(XCR0_X87 | XCR0_SSE | XCR0_AVX), 832);
        // PKRU's 8 bytes end the area
        assert_eq!(enabled_area_size(XCR0_SUPPORTED), 2696);
    }

    #[test]
    fn xrstor_checks_the_whole_area_before_changing_anything() {
        // XSTATE_BV names a component XCR0 does not enable: #GP, registers untouched
        let stop = stop_of(XRSTOR_RDI, |h| {
            h.cpu.gpr[RDI] = AREA;
            h.cpu.gpr[0] = ALL;
            h.cpu.gpr[2] = 0;
            h.cpu.xcr0 = XCR0_X87 | XCR0_SSE | XCR0_AVX;
            h.write_u64(AREA + 512, 1 << 5);
        });
        assert!(matches!(stop, Stop::Fault { vector: 13, .. }));
    }
}

mod lane_wise {
    use super::*;

    macro_rules! integer {
        ($name:ident, $variants:expr, $element:expr, $f:expr) => {
            #[test]
            fn $name() {
                let mask = size_mask($element);
                check_elementwise(
                    stringify!($name),
                    &$variants,
                    spec($element, $element),
                    |a, b| {
                        let f: fn(u64, u64) -> u64 = $f;
                        f(a, b) & mask
                    },
                );
            }
        };
    }

    integer!(vpaddd, VPADDD, 4, |a, b| (a as u32).wrapping_add(b as u32)
        as u64);
    integer!(vpaddq, VPADDQ, 8, |a, b| a.wrapping_add(b));
    integer!(vpsubd, VPSUBD, 4, |a, b| (a as u32).wrapping_sub(b as u32)
        as u64);
    integer!(vpsubq, VPSUBQ, 8, |a, b| a.wrapping_sub(b));
    integer!(vpandd, VPANDD, 4, |a, b| a & b);
    integer!(vpandq, VPANDQ, 8, |a, b| a & b);
    integer!(vpord, VPORD, 4, |a, b| a | b);
    integer!(vpxorq, VPXORQ, 8, |a, b| a ^ b);
    integer!(vpandnq, VPANDNQ, 8, |a, b| !a & b);
    integer!(
        vpmulld,
        VPMULLD,
        4,
        |a, b| (a as u32).wrapping_mul(b as u32) as u64
    );
    integer!(vpmaxsd, VPMAXSD, 4, |a, b| (a as i32).max(b as i32) as u32
        as u64);
    integer!(vpminud, VPMINUD, 4, |a, b| (a as u32).min(b as u32) as u64);

    #[test]
    fn byte_and_word_adds_mask_per_element() {
        check_elementwise("vpaddb", &VPADDB, spec(1, 0), |a, b| {
            a.wrapping_add(b) & 0xff
        });
        check_elementwise("vpaddw", &VPADDW, spec(2, 0), |a, b| {
            a.wrapping_add(b) & 0xffff
        });
    }

    #[test]
    fn pshufb_selects_within_each_128_bit_lane() {
        let mut rng = Rng(0x55);
        let (table, selectors) = (rng.vector(), rng.vector());
        let h = run(VPSHUFB.p[2], |h| {
            h.cpu.set_zmm(1, table);
            h.cpu.set_zmm(2, selectors);
        });
        let (t, s) = (bytes_of(&table), bytes_of(&selectors));
        let mut want = [0u8; 64];
        for k in 0..64 {
            let lane = k / 16 * 16;
            want[k] = if s[k] & 0x80 != 0 {
                0
            } else {
                t[lane + (s[k] & 15) as usize]
            };
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
    }

    #[test]
    fn punpck_and_unpck_interleave_within_lanes() {
        let mut rng = Rng(0x77);
        let (a, b) = (rng.vector(), rng.vector());
        let (ab, bb) = (bytes_of(&a), bytes_of(&b));
        let h = run(VPUNPCKLDQ.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let mut want = [0u8; 64];
        for lane in 0..4 {
            for (slot, source) in [(0, &ab), (1, &bb), (2, &ab), (3, &bb)] {
                let from = lane * 16 + (slot / 2) * 4;
                want[lane * 16 + slot * 4..lane * 16 + slot * 4 + 4]
                    .copy_from_slice(&source[from..from + 4]);
            }
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
    }

    #[test]
    fn pshufd_and_shufps_apply_the_immediate_per_lane() {
        let mut rng = Rng(0x99);
        let (a, b) = (rng.vector(), rng.vector());
        let h = run(VPSHUFD.p[2], |h| h.cpu.set_zmm(2, a));
        let d = |v: &Vector, k: usize| element(v, 4, k);
        let mut want = [0u128; 4];
        for lane in 0..4 {
            for slot in 0..4 {
                let from = (0x1b >> (2 * slot)) & 3;
                set_element(&mut want, 4, lane * 4 + slot, d(&a, lane * 4 + from));
            }
        }
        assert_eq!(h.cpu.zmm(0), want);
        let h = run(VSHUFPS.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let mut want = [0u128; 4];
        for lane in 0..4 {
            for slot in 0..4 {
                let from = (0x1b >> (2 * slot)) & 3;
                let source = if slot < 2 { &a } else { &b };
                set_element(&mut want, 4, lane * 4 + slot, d(source, lane * 4 + from));
            }
        }
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn palignr_works_per_lane() {
        let mut rng = Rng(0x1234);
        let (a, b) = (rng.vector(), rng.vector());
        let h = run(VPALIGNR.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let (ab, bb) = (bytes_of(&a), bytes_of(&b));
        let mut want = [0u8; 64];
        for lane in 0..4 {
            let mut joined = [0u8; 32];
            joined[..16].copy_from_slice(&bb[lane * 16..lane * 16 + 16]);
            joined[16..].copy_from_slice(&ab[lane * 16..lane * 16 + 16]);
            want[lane * 16..lane * 16 + 16].copy_from_slice(&joined[5..21]);
        }
        assert_eq!(bytes_of(&h.cpu.zmm(0)), want);
    }

    /// vdbpsadbw: the Intrinsics Guide pseudo-code, written byte by byte (immediate 0x93 selects
    /// dwords 3, 0, 1, 2 of every 128-bit lane of the second source).
    fn double_block_sad(a: &[u8; 64], b: &[u8; 64], length: usize) -> Vector {
        const DWORDS_SELECTED: [usize; 4] = [3, 0, 1, 2];
        let mut shuffled = [0u8; 64];
        for lane in 0..length / 16 {
            for (slot, &from) in DWORDS_SELECTED.iter().enumerate() {
                let source = lane * 16 + from * 4;
                shuffled[lane * 16 + slot * 4..][..4].copy_from_slice(&b[source..source + 4]);
            }
        }
        let sad = |first: usize, second: usize| -> u64 {
            (0..4)
                .map(|k| (a[first + k] as i64 - shuffled[second + k] as i64).unsigned_abs())
                .sum()
        };
        let mut out = [0u128; 4];
        for qword in 0..length / 8 {
            let base = qword * 8;
            let words = [
                sad(base, base),
                sad(base, base + 1),
                sad(base + 4, base + 2),
                sad(base + 4, base + 3),
            ];
            for (k, word) in words.into_iter().enumerate() {
                set_element(&mut out, 2, qword * 4 + k, word);
            }
        }
        out
    }

    #[test]
    fn dbpsadbw_sums_absolute_differences_of_shuffled_quadruplets() {
        let mut rng = Rng(0xdb5a_d001);
        for trial in 0..4 {
            let (a, b, old) = (rng.vector(), rng.vector(), rng.vector());
            let k1 = rng.next() & 0xffff_ffff;
            for (index, length) in LENGTHS.into_iter().enumerate() {
                for (shape, codes) in [
                    ("plain", &VDBPSADBW.p),
                    ("merge", &VDBPSADBW.k),
                    ("zero", &VDBPSADBW.kz),
                    ("memory", &VDBPSADBW.m),
                    ("memory merge", &VDBPSADBW.mk),
                ] {
                    let masked = shape.contains("merge") || shape == "zero";
                    let h = run(codes[index], |h| {
                        h.cpu.set_zmm(0, old);
                        h.cpu.set_zmm(1, a);
                        h.cpu.set_zmm(2, b);
                        h.cpu.opmask[1] = k1;
                        h.write(DATA, &bytes_of(&b));
                    });
                    let sads = double_block_sad(&bytes_of(&a), &bytes_of(&b), length);
                    let want = expected(
                        length,
                        2,
                        masked.then_some((k1, shape == "zero")),
                        &old,
                        |k| super::element(&sads, 2, k),
                    );
                    assert_eq!(
                        bytes_of(&h.cpu.zmm(0)),
                        bytes_of(&want),
                        "{shape} {length} bytes, trial {trial}"
                    );
                }
            }
        }
    }

    #[test]
    fn pabsd_and_sqrt_are_unary() {
        check_elementwise("vpabsd", &VPABSD, unary_spec(4, 4), |_, b| {
            (b as u32 as i32).unsigned_abs() as u64
        });
        check_elementwise("vsqrtps", &VSQRTPS, unary_spec(4, 4), |_, b| {
            // inputs are random bits: NaNs and negatives give the default NaN; compare only
            // through the host for finite non-negative values
            let x = f32::from_bits(b as u32);
            if x.is_nan() || x < 0.0 {
                return sqrt_special(b as u32) as u64;
            }
            x.sqrt().to_bits() as u64
        });
    }

    /// NaN and negative inputs: a NaN input comes back quieted, anything else negative is the
    /// default NaN (0xffc00000), and -0 stays -0.
    fn sqrt_special(bits: u32) -> u32 {
        let x = f32::from_bits(bits);
        if x.is_nan() {
            bits | 0x0040_0000
        } else if x == 0.0 {
            bits
        } else {
            0xffc0_0000
        }
    }

    #[test]
    fn packed_float_arithmetic_matches_the_host_for_ordinary_values() {
        // avoid NaN payload and denormal cases here; the soft-float engine has its own tests
        let mut rng = Rng(0xabc);
        let ordinary = |rng: &mut Rng| {
            let mut v = [0u128; 4];
            for k in 0..16 {
                let x = ((rng.next() % 2000) as f32 - 1000.0) / 8.0;
                set_element(&mut v, 4, k, x.to_bits() as u64);
            }
            v
        };
        for (variants, f) in [
            (&VADDPS, (|a: f32, b: f32| a + b) as fn(f32, f32) -> f32),
            (&VMULPS, |a, b| a * b),
            (&VDIVPS, |a, b| {
                if b == 0.0 {
                    f32::INFINITY * a.signum()
                } else {
                    a / b
                }
            }),
        ] {
            let (a, b) = (ordinary(&mut rng), ordinary(&mut rng));
            let h = run(variants.p[2], |h| {
                h.cpu.set_zmm(1, a);
                h.cpu.set_zmm(2, b);
            });
            for k in 0..16 {
                let want = f(
                    f32::from_bits(element(&a, 4, k) as u32),
                    f32::from_bits(element(&b, 4, k) as u32),
                );
                let got = f32::from_bits(element(&h.cpu.zmm(0), 4, k) as u32);
                assert!(
                    got == want || (got.is_nan() && want.is_nan()),
                    "{got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn masked_float_ops_keep_or_zero_masked_elements() {
        let mut rng = Rng(0xdef);
        let (a, b, old) = (rng.vector(), rng.vector(), rng.vector());
        let k1 = 0xa5a5u64;
        let plain = run(VADDPS.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        })
        .cpu
        .zmm(0);
        for (code, zeroing) in [(VADDPS.k[2], false), (VADDPS.kz[2], true)] {
            let h = run(code, |h| {
                h.cpu.set_zmm(0, old);
                h.cpu.set_zmm(1, a);
                h.cpu.set_zmm(2, b);
                h.cpu.opmask[1] = k1;
            });
            for k in 0..16 {
                let want = if k1 >> k & 1 == 1 {
                    element(&plain, 4, k)
                } else if zeroing {
                    0
                } else {
                    element(&old, 4, k)
                };
                assert_eq!(element(&h.cpu.zmm(0), 4, k), want, "element {k}");
            }
        }
    }

    #[test]
    fn double_arithmetic_and_min_max() {
        check_elementwise("vaddpd", &VADDPD, spec(8, 8), |a, b| {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            let r = x + y;
            if r.is_nan() {
                return if x.is_nan() {
                    a | (1 << 51)
                } else if y.is_nan() {
                    b | (1 << 51)
                } else {
                    0xfff8_0000_0000_0000
                };
            }
            r.to_bits()
        });
    }

    #[test]
    fn sqrtpd_cvtdq2ps_and_movsldup_cover_other_element_sizes() {
        check_elementwise("vcvtdq2ps", &VCVTDQ2PS, unary_spec(4, 4), |_, b| {
            (b as u32 as i32 as f32).to_bits() as u64
        });
        let mut rng = Rng(0x31);
        let a = rng.vector();
        let h = run(VMOVSLDUP.p[2], |h| h.cpu.set_zmm(2, a));
        let mut want = [0u128; 4];
        for k in 0..8 {
            let e = element(&a, 4, 2 * k);
            set_element(&mut want, 4, 2 * k, e);
            set_element(&mut want, 4, 2 * k + 1, e);
        }
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn compressed_displacements_scale_by_the_tuple_size() {
        let mut rng = Rng(0x42);
        let (a, memory) = (rng.vector(), rng.vector());
        let data = |h: &mut Harness| {
            h.write(DATA, &[0u8; 64]);
            // fill the 256 bytes the scaled addresses reach
            for k in 0..32 {
                h.write(DATA - 64 + 16 * k, &memory[(k % 4) as usize].to_le_bytes());
            }
        };
        let sum = |data: &Vector, lanes: usize| {
            let mut want = [0u128; 4];
            for k in 0..lanes * 4 {
                let v = element(&a, 4, k).wrapping_add(element(data, 4, k)) as u32 as u64;
                set_element(&mut want, 4, k, v);
            }
            want
        };
        let at = |h: &Harness, offset: i64, bytes: usize| {
            let address = (DATA as i64 + offset) as u64;
            vector_of(&h.read(address, bytes))
        };
        // [rsi+128] with a 512-bit operand is disp8 2 scaled by 64
        let h = run(VPADDD_DISP8, |h| {
            data(h);
            h.cpu.set_zmm(1, a);
        });
        assert_eq!(h.cpu.zmm(0), sum(&at(&h, 128, 64), 4));
        let h = run(VPADDD_DISP8_NEGATIVE, |h| {
            data(h);
            h.cpu.set_zmm(1, a);
        });
        assert_eq!(h.cpu.zmm(0), sum(&at(&h, -64, 64), 4));
        // a broadcast scales by the element size: [rsi+8]{1to16}
        let h = run(VPADDD_BCAST_DISP8, |h| {
            data(h);
            h.cpu.set_zmm(1, a);
        });
        let element8 = element(&at(&h, 8, 4), 4, 0);
        let mut want = [0u128; 4];
        for k in 0..16 {
            set_element(
                &mut want,
                4,
                k,
                element(&a, 4, k).wrapping_add(element8) & 0xffff_ffff,
            );
        }
        assert_eq!(h.cpu.zmm(0), want);
        // 256- and 128-bit operands scale by 32 and 16
        for (code, offset, lanes) in [(VPADDD_Y_DISP8, 64, 2usize), (VPADDD_X_DISP8, 16, 1)] {
            let h = run(code, |h| {
                data(h);
                h.cpu.set_zmm(1, a);
            });
            let got = h.cpu.zmm(0);
            let mut want = sum(&at(&h, offset, 16 * lanes), lanes);
            want[lanes..].fill(0);
            assert_eq!(got, want, "{} bytes", 16 * lanes);
        }
    }

    #[test]
    fn evex_b_on_a_register_operand_without_rounding_support_is_invalid() {
        // vpaddd with EVEX.b set on a register form: patch the assembled vpaddd zmm0,zmm1,zmm2
        let mut code = VPADDD.p[2].to_vec();
        code[3] |= 0x10;
        assert_eq!(
            stop_of(&code, |_| {}),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
    }

    #[test]
    fn reserved_vector_length_is_invalid() {
        let mut code = VPADDD.p[2].to_vec();
        code[3] |= 0x60;
        assert_eq!(
            stop_of(&code, |_| {}),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
    }

    #[test]
    fn vvvv_must_be_unused_by_unary_instructions() {
        // vsqrtps zmm0, zmm2 with vvvv = 1 instead of 1111
        let mut code = VSQRTPS.p[2].to_vec();
        code[2] = (code[2] & !0x78) | (0xe << 3);
        assert_eq!(
            stop_of(&code, |_| {}),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
    }

    #[test]
    fn shifts_by_a_count_use_the_low_128_bits_for_every_lane() {
        let mut rng = Rng(0x600d);
        let (value, count) = (rng.vector(), rng.vector());
        let low = count[0] as u64;
        let h = run(VPSLLD_COUNT_Z, |h| {
            h.cpu.set_zmm(1, value);
            h.cpu
                .set_zmm(2, [low as u128, u128::MAX, u128::MAX, u128::MAX]);
        });
        let mut want = [0u128; 4];
        for k in 0..16 {
            let v = if low >= 32 {
                0
            } else {
                (element(&value, 4, k) as u32) << low
            };
            set_element(&mut want, 4, k, v as u64);
        }
        assert_eq!(h.cpu.zmm(0), want);
        // memory count: 16 bytes at rsi, no scaling by the vector length
        let h = run(VPSLLD_COUNT_Z_M, |h| {
            h.cpu.set_zmm(1, value);
            h.write(DATA, &(low as u128).to_le_bytes());
            h.write(DATA + 16, &[0xff; 48]);
        });
        assert_eq!(h.cpu.zmm(0), want);
        // masked, 256-bit
        let h = run(VPSLLQ_COUNT_Y_K, |h| {
            h.cpu.set_zmm(0, [7 << 64 | 7; 4]);
            h.cpu.set_zmm(1, value);
            h.cpu.set_zmm(2, [3, 0, 0, 0]);
            h.cpu.opmask[1] = 0b0101;
        });
        let v = h.cpu.zmm(0);
        assert_eq!(element(&v, 8, 0), element(&value, 8, 0) << 3);
        assert_eq!(element(&v, 8, 1), 7);
        assert_eq!(element(&v, 8, 2), element(&value, 8, 2) << 3);
        assert_eq!(element(&v, 8, 3), 7);
        assert_eq!(&v[2..], &[0, 0], "nothing above 256 bits");
    }
}

mod rounding {
    use super::*;

    fn lane_of(single: f32) -> Vector {
        let mut v = [0u128; 4];
        for k in 0..16 {
            set_element(&mut v, 4, k, single.to_bits() as u64);
        }
        v
    }

    #[test]
    fn embedded_rounding_overrides_mxcsr_for_one_instruction() {
        // 1 + 2^-24 is a tie between 1 and the next float
        let one = lane_of(1.0);
        let tiny = lane_of(f32::from_bits(0x3380_0000)); // 2^-24
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        for (code, want) in [
            (VADDPS_RN_SAE, 1.0f32),
            (VADDPS_RD_SAE, 1.0),
            (VADDPS_RU_SAE, next),
            (VADDPS_RZ_SAE, 1.0),
        ] {
            let h = run(code, |h| {
                h.cpu.set_zmm(1, one);
                h.cpu.set_zmm(2, tiny);
            });
            assert_eq!(h.cpu.zmm(0), lane_of(want), "{code:02x?}");
            assert_eq!(
                h.cpu.mxcsr, 0x1f80,
                "MXCSR rounding and flags are untouched"
            );
        }
    }

    #[test]
    fn embedded_rounding_suppresses_exception_flags_and_faults() {
        // 1/3-ish inexact add with {rd-sae}: no precision flag, no #XM even when unmasked
        let one = lane_of(1.0);
        let tiny = lane_of(f32::from_bits(0x3380_0000));
        let h = run(VADDPS_RD_SAE, |h| {
            h.cpu.mxcsr = 0x1f80 & !(1 << 12); // unmask precision
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSXMMEXCPT;
            h.cpu.set_zmm(1, one);
            h.cpu.set_zmm(2, tiny);
        });
        assert_eq!(h.cpu.mxcsr & 0x3f, 0);
        // without {sae} the same inexact add sets the flag and faults
        let stop = stop_of(VADDPS.p[2], |h| {
            h.cpu.mxcsr = 0x1f80 & !(1 << 12);
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSXMMEXCPT;
            h.cpu.set_zmm(1, one);
            h.cpu.set_zmm(2, tiny);
        });
        assert!(matches!(stop, Stop::Fault { vector: 19, .. }));
    }

    #[test]
    fn sae_only_instructions_keep_the_mxcsr_rounding() {
        let h = run(VMAXPS_SAE, |h| {
            h.cpu.set_zmm(1, lane_of(1.0));
            h.cpu.set_zmm(2, lane_of(2.0));
        });
        assert_eq!(h.cpu.zmm(0), lane_of(2.0));
    }

    #[test]
    fn masked_embedded_rounding_zeroes_masked_elements() {
        let one = lane_of(1.0);
        let tiny = lane_of(f32::from_bits(0x3380_0000));
        let h = run(VADDPS_K1_RD_SAE, |h| {
            h.cpu.set_zmm(0, [u128::MAX; 4]);
            h.cpu.set_zmm(1, one);
            h.cpu.set_zmm(2, tiny);
            h.cpu.opmask[1] = 0b11;
        });
        let v = h.cpu.zmm(0);
        assert_eq!(element(&v, 4, 0), 1.0f32.to_bits() as u64);
        assert_eq!(element(&v, 4, 1), 1.0f32.to_bits() as u64);
        assert_eq!(element(&v, 4, 2), 0);
    }

    #[test]
    fn scalar_ops_merge_the_upper_elements_from_vvvv_and_mask_element_zero() {
        let mut rng = Rng(0x5ca1a2);
        let (a, b, old) = (rng.vector(), rng.vector(), rng.vector());
        let sum = |x: u64, y: u64| (f32::from_bits(x as u32) + f32::from_bits(y as u32)).to_bits();
        let (a0, b0) = (element(&a, 4, 0), element(&b, 4, 0));
        // keep the sum ordinary so the host reference applies
        let mut a = a;
        let mut b = b;
        set_element(&mut a, 4, 0, 1.5f32.to_bits() as u64);
        set_element(&mut b, 4, 0, 2.25f32.to_bits() as u64);
        let _ = (a0, b0);
        let want_sum = sum(1.5f32.to_bits() as u64, 2.25f32.to_bits() as u64) as u128;
        let h = run(VADDSS, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let v = h.cpu.zmm(0);
        assert_eq!(
            v[0],
            a[0] & !0xffff_ffff | want_sum,
            "low lane: sum then vvvv's upper bits"
        );
        assert_eq!(
            &v[1..],
            &[0, 0, 0],
            "everything above the low lane is zeroed"
        );
        // merge masking with the bit clear keeps the old element 0
        let h = run(VADDSS_K1, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(h.cpu.zmm(0)[0], a[0] & !0xffff_ffff | old[0] & 0xffff_ffff);
        // zeroing
        let h = run(VADDSS_K1Z, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(h.cpu.zmm(0)[0], a[0] & !0xffff_ffff);
        // bit set: the sum
        let h = run(VADDSS_K1Z, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 1;
        });
        assert_eq!(h.cpu.zmm(0)[0], a[0] & !0xffff_ffff | want_sum);
    }

    #[test]
    fn scalar_double_with_a_memory_operand_reads_eight_bytes() {
        let mut rng = Rng(0x77);
        let a = rng.vector();
        let mut a = a;
        set_element(&mut a, 8, 0, 1.25f64.to_bits());
        let h = run(VADDSD_M, |h| {
            h.cpu.set_zmm(1, a);
            h.write(DATA, &2.5f64.to_bits().to_le_bytes());
            h.write(DATA + 8, &[0xff; 8]);
        });
        let v = h.cpu.zmm(0);
        assert_eq!(v[0] as u64, 3.75f64.to_bits());
        assert_eq!((v[0] >> 64) as u64, (a[0] >> 64) as u64);
        // disp8 scales by the element size (8): [rsi+16] is disp8 2
        let h = run(VADDSD_M_DISP8, |h| {
            h.cpu.set_zmm(1, a);
            h.write(DATA + 16, &2.5f64.to_bits().to_le_bytes());
        });
        assert_eq!(h.cpu.zmm(0)[0] as u64, 3.75f64.to_bits());
    }
}

mod moves {
    use super::*;

    #[test]
    fn full_moves_honour_masks_for_every_element_size() {
        for (name, variants, element) in [
            ("vmovdqu8", &VMOVDQU8, 1usize),
            ("vmovdqu16", &VMOVDQU16, 2),
            ("vmovdqu32", &VMOVDQU32, 4),
            ("vmovdqu64", &VMOVDQU64, 8),
            ("vmovups", &VMOVUPS, 4),
        ] {
            check_elementwise(name, variants, unary_spec(element, 0), |_, b| b);
        }
    }

    #[test]
    fn aligned_moves_fault_on_a_misaligned_address_but_unaligned_ones_do_not() {
        let gp = Stop::Fault {
            vector: 13,
            error_code: Some(0),
            software: false,
        };
        assert_eq!(stop_of(VMOVDQA64_Z_M, |h| h.cpu.gpr[RSI] = DATA + 8), gp);
        assert_eq!(stop_of(VMOVDQA64_M_Z, |h| h.cpu.gpr[RDI] = OUT + 32), gp);
        assert_eq!(stop_of(VMOVNTDQ_M_Z, |h| h.cpu.gpr[RDI] = OUT + 16), gp);
        assert_eq!(stop_of(VMOVAPS_M_Z_K, |h| h.cpu.gpr[RDI] = OUT + 4), gp);
        assert_eq!(
            stop_of(VMOVUPS_M_Z_K, |h| h.cpu.gpr[RDI] = OUT + 4),
            Stop::Halted
        );
        assert_eq!(stop_of(VMOVDQA64_Z_M, |_| {}), Stop::Halted);
    }

    #[test]
    fn masked_stores_write_only_the_selected_elements() {
        let mut rng = Rng(0x57);
        let value = rng.vector();
        let k1 = 0x0123_4567_89ab_cdefu64;
        let h = run(VMOVDQU8_M_Z_K, |h| {
            h.cpu.set_zmm(1, value);
            h.cpu.opmask[1] = k1;
            h.write(OUT, &[0xee; 64]);
        });
        let (bytes, stored) = (bytes_of(&value), h.read(OUT, 64));
        for k in 0..64 {
            let want = if k1 >> k & 1 == 1 { bytes[k] } else { 0xee };
            assert_eq!(stored[k], want, "byte {k}");
        }
        let h = run(VMOVUPS_M_Z_K, |h| {
            h.cpu.set_zmm(1, value);
            h.cpu.opmask[1] = 0x8001;
            h.write(OUT, &[0xee; 64]);
        });
        let stored = h.read(OUT, 64);
        assert_eq!(&stored[..4], &bytes[..4]);
        assert_eq!(&stored[4..60], &[0xee; 56][..]);
        assert_eq!(&stored[60..], &bytes[60..]);
    }

    #[test]
    fn masked_loads_do_not_fault_on_masked_out_elements() {
        // The 64-byte load starts 32 bytes before the end of RAM; the upper 32 bytes are beyond
        // it. With only the low elements selected there is no fault; selecting one of the others
        // faults and changes nothing.
        let mut rng = Rng(0x10ad);
        let old = rng.vector();
        let h = run(VMOVUPS_Z_M_K, |h| {
            h.write(EDGE, &[0x5a; 32]);
            unmap_boot_identity_2m(h, UNMAPPED);
            h.cpu.gpr[RSI] = EDGE;
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = 0x00ff;
        });
        let loaded = h.cpu.zmm(0);
        assert_eq!(
            &loaded[..2],
            &[0x5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a_5a5a; 2]
        );
        assert_eq!(
            &loaded[2..],
            &old[2..],
            "masked-out elements keep their old value"
        );
        let mut h = rig(VMOVUPS_Z_M_K);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RSI] = EDGE;
        h.cpu.set_zmm(0, old);
        h.cpu.opmask[1] = 0x0100;
        assert!(matches!(h.run(10), Stop::Fault { vector: 14, .. }));
        assert_eq!(
            h.cpu.zmm(0),
            old,
            "a faulting load leaves the destination alone"
        );
        // zeroing form too
        let mut h = rig(VMOVUPS_Z_M_KZ);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RSI] = EDGE;
        h.cpu.opmask[1] = 0x00ff;
        assert_eq!(h.run(10), Stop::Halted);
    }

    #[test]
    fn faulting_masked_stores_write_nothing() {
        let mut h = rig(VMOVUPS_M_Z_K);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RDI] = EDGE;
        h.cpu.set_zmm(1, [u128::MAX; 4]);
        h.cpu.opmask[1] = 0x0101; // element 0 is fine, element 8 is in the unmapped region
        h.write(EDGE, &[0u8; 32]);
        assert!(matches!(h.run(10), Stop::Fault { vector: 14, .. }));
        assert_eq!(
            h.read(EDGE, 32),
            vec![0u8; 32],
            "the valid element must not be written"
        );
        // only valid elements selected: no fault
        let mut h = rig(VMOVUPS_M_Z_K);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RDI] = EDGE;
        h.cpu.set_zmm(1, [u128::MAX; 4]);
        h.cpu.opmask[1] = 0x00ff;
        assert_eq!(h.run(10), Stop::Halted);
    }

    #[test]
    fn registers_above_15_move() {
        let mut rng = Rng(0x3131);
        let v = rng.vector();
        let h = run(VMOVDQU64_HIGH, |h| h.cpu.set_zmm(17, v));
        assert_eq!(h.cpu.zmm(30), v);
        let h = run(VMOVDQU64_HIGH_M, |h| h.write(DATA, &bytes_of(&v)));
        assert_eq!(h.cpu.zmm(31), v);
        let h = run(VMOVDQU64_M_HIGH, |h| h.cpu.set_zmm(29, v));
        assert_eq!(h.read(OUT, 64), bytes_of(&v).to_vec());
    }

    #[test]
    fn non_temporal_store_writes_the_whole_vector() {
        let mut rng = Rng(0x7);
        let v = rng.vector();
        let h = run(VMOVNTDQ_M_Z, |h| h.cpu.set_zmm(1, v));
        assert_eq!(h.read(OUT, 64), bytes_of(&v).to_vec());
    }

    #[test]
    fn movd_and_movq_transfer_with_zero_extension() {
        let h = run(VMOVD_X_EAX, |h| {
            h.cpu.gpr[0] = 0xdead_beef_1234_5678;
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        assert_eq!(h.cpu.zmm(0), [0x1234_5678, 0, 0, 0]);
        let h = run(VMOVQ_X_RAX, |h| {
            h.cpu.gpr[0] = 0xdead_beef_1234_5678;
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        assert_eq!(h.cpu.zmm(0), [0xdead_beef_1234_5678, 0, 0, 0]);
        let h = run(VMOVD_EAX_X, |h| {
            h.cpu.gpr[0] = u64::MAX;
            h.cpu
                .set_zmm(1, [0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00, 0, 0, 0]);
        });
        assert_eq!(h.cpu.gpr[0], 0xddee_ff00);
        let h = run(VMOVQ_RAX_X, |h| {
            h.cpu
                .set_zmm(1, [0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00, 0, 0, 0]);
        });
        assert_eq!(h.cpu.gpr[0], 0x99aa_bbcc_ddee_ff00);
        let h = run(VMOVD_X_M, |h| {
            h.write(DATA, &[1, 2, 3, 4, 5, 6, 7, 8]);
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        assert_eq!(h.cpu.zmm(0), [0x0403_0201, 0, 0, 0]);
        let h = run(VMOVQ_X_X, |h| {
            h.cpu.set_zmm(1, [u128::MAX, 5, 5, 5]);
            h.cpu.set_zmm(0, [9; 4]);
        });
        assert_eq!(h.cpu.zmm(0), [u64::MAX as u128, 0, 0, 0]);
        let h = run(VMOVQ_X_M, |h| h.write(DATA, &[0xff; 16]));
        assert_eq!(h.cpu.zmm(0), [u64::MAX as u128, 0, 0, 0]);
        let h = run(VMOVQ_M_X, |h| {
            h.cpu.set_zmm(1, [0x0102_0304_0506_0708, 0, 0, 0])
        });
        assert_eq!(h.read_u64(OUT), 0x0102_0304_0506_0708);
    }

    #[test]
    fn vmovss_register_form_merges_and_memory_form_zeroes() {
        let mut rng = Rng(0x55aa);
        let (a, b) = (rng.vector(), rng.vector());
        let h = run(VMOVSS_X_X_X, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        assert_eq!(
            h.cpu.zmm(0),
            [a[0] & !0xffff_ffff | b[0] & 0xffff_ffff, 0, 0, 0]
        );
        let h = run(VMOVSS_X_M, |h| {
            h.write(DATA, &0x1234_5678u32.to_le_bytes());
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        assert_eq!(h.cpu.zmm(0), [0x1234_5678, 0, 0, 0]);
        // masked load with the bit clear does not touch memory (so cannot fault) and merges or
        // zeroes
        let mut h = rig(VMOVSS_X_M_K);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RSI] = UNMAPPED;
        h.cpu.opmask[1] = 0;
        h.cpu.set_zmm(0, [7, 8, 9, 10]);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(
            h.cpu.zmm(0),
            [7, 0, 0, 0],
            "upper lanes of a scalar write are cleared"
        );
        let mut h = rig(VMOVSS_X_M_KZ);
        unmap_boot_identity_2m(&mut h, UNMAPPED);
        h.cpu.gpr[RSI] = UNMAPPED;
        h.cpu.opmask[1] = 0;
        h.cpu.set_zmm(0, [7, 8, 9, 10]);
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.zmm(0), [0, 0, 0, 0]);
        // masked store with the bit clear writes nothing
        let h = run(VMOVSS_M_X_K, |h| {
            h.cpu.set_zmm(1, [0x5555_5555, 0, 0, 0]);
            h.write(OUT, &[0xee; 8]);
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(h.read(OUT, 8), vec![0xee; 8]);
        let h = run(VMOVSS_M_X_K, |h| {
            h.cpu.set_zmm(1, [0x5555_5555, 0, 0, 0]);
            h.write(OUT, &[0xee; 8]);
            h.cpu.opmask[1] = 1;
        });
        assert_eq!(
            h.read(OUT, 8),
            [0x55, 0x55, 0x55, 0x55, 0xee, 0xee, 0xee, 0xee]
        );
        // vmovsd with a mask: merge element 0 from vvvv's source register
        let h = run(VMOVSD_X_X_X_K, |h| {
            h.cpu.set_zmm(0, [0x11, 0, 0, 0]);
            h.cpu.set_zmm(1, [u128::MAX, 0, 0, 0]);
            h.cpu.set_zmm(2, [0x22, 0, 0, 0]);
            h.cpu.opmask[1] = 0;
        });
        assert_eq!(h.cpu.zmm(0), [!(u64::MAX as u128) | 0x11, 0, 0, 0]);
        let h = run(VMOVSD_X_M, |h| h.write(DATA, &[0xff; 16]));
        assert_eq!(h.cpu.zmm(0), [u64::MAX as u128, 0, 0, 0]);
    }
}

mod opmask {
    use super::*;

    const ZF: u64 = 1 << 6;
    const CF: u64 = 1;

    fn width_mask(name: char) -> u64 {
        match name {
            'b' => 0xff,
            'w' => 0xffff,
            'd' => 0xffff_ffff,
            _ => u64::MAX,
        }
    }

    fn pairs() -> Vec<(u64, u64)> {
        let mut rng = Rng(0x6b);
        let mut out = vec![(0, 0), (u64::MAX, u64::MAX), (u64::MAX, 0), (0, u64::MAX)];
        out.extend((0..12).map(|_| (rng.next(), rng.next())));
        out
    }

    fn run_k(code: &[u8], k2: u64, k3: u64) -> Harness {
        run(code, |h| {
            h.cpu.opmask[1] = 0xdead_beef_dead_beef;
            h.cpu.opmask[2] = k2;
            h.cpu.opmask[3] = k3;
        })
    }

    #[test]
    fn binary_operations_truncate_to_the_operand_width_and_zero_the_rest() {
        type Operation = fn(u64, u64) -> u64;
        let ops: [(&str, [&[u8]; 4], Operation); 6] = [
            ("kand", [KANDB, KANDW, KANDD, KANDQ], |a, b| a & b),
            ("kandn", [KANDNB, KANDNW, KANDND, KANDNQ], |a, b| !a & b),
            ("kor", [KORB, KORW, KORD, KORQ], |a, b| a | b),
            ("kxnor", [KXNORB, KXNORW, KXNORD, KXNORQ], |a, b| !(a ^ b)),
            ("kxor", [KXORB, KXORW, KXORD, KXORQ], |a, b| a ^ b),
            ("kadd", [KADDB, KADDW, KADDD, KADDQ], |a, b| {
                a.wrapping_add(b)
            }),
        ];
        for (name, codes, f) in ops {
            for (width, code) in "bwdq".chars().zip(codes) {
                for (a, b) in pairs() {
                    let h = run_k(code, a, b);
                    let m = width_mask(width);
                    assert_eq!(
                        h.cpu.opmask[1],
                        f(a & m, b & m) & m,
                        "{name}{width} {a:#x} {b:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn knot_inverts_within_the_width() {
        for (width, code) in "bwdq".chars().zip([KNOTB, KNOTW, KNOTD, KNOTQ]) {
            for (a, _) in pairs() {
                let h = run_k(code, a, 0);
                assert_eq!(h.cpu.opmask[1], !a & width_mask(width));
            }
        }
    }

    #[test]
    fn kortest_and_ktest_set_zf_and_cf() {
        for (width, code) in "bwdq".chars().zip([KORTESTB, KORTESTW, KORTESTD, KORTESTQ]) {
            let m = width_mask(width);
            for (a, b) in pairs() {
                // kortest k1, k2: k1 is the register operand, k2 the r/m
                let h = run(code, |h| {
                    h.cpu.opmask[1] = a;
                    h.cpu.opmask[2] = b;
                });
                let or = (a | b) & m;
                let want = if or == 0 { ZF } else { 0 } | if or == m { CF } else { 0 };
                assert_eq!(h.cpu.eflags_full() & (ZF | CF), want, "kortest{width}");
            }
        }
        for (width, code) in "bwdq".chars().zip([KTESTB, KTESTW, KTESTD, KTESTQ]) {
            let m = width_mask(width);
            for (a, b) in pairs() {
                let h = run(code, |h| {
                    h.cpu.opmask[1] = a;
                    h.cpu.opmask[2] = b;
                });
                let (a, b) = (a & m, b & m);
                let want = if a & b == 0 { ZF } else { 0 } | if !a & b & m == 0 { CF } else { 0 };
                assert_eq!(h.cpu.eflags_full() & (ZF | CF), want, "ktest{width}");
            }
        }
    }

    #[test]
    fn shifts_move_bits_and_saturate_to_zero_at_the_width() {
        for (width, bits, codes) in [
            ('b', 8u32, [KSHIFTLB_3, KSHIFTRB_3, KSHIFTLB_9, KSHIFTRB_9]),
            ('w', 16, [KSHIFTLW_3, KSHIFTRW_3, KSHIFTLW_9, KSHIFTRW_9]),
            ('d', 32, [KSHIFTLD_3, KSHIFTRD_3, KSHIFTLD_9, KSHIFTRD_9]),
            ('q', 64, [KSHIFTLQ_3, KSHIFTRQ_3, KSHIFTLQ_9, KSHIFTRQ_9]),
        ] {
            let m = width_mask(width);
            for (a, _) in pairs() {
                for (code, left, count) in [
                    (codes[0], true, 3u32),
                    (codes[1], false, 3),
                    (codes[2], true, 9),
                    (codes[3], false, 9),
                ] {
                    let h = run_k(code, a, 0);
                    let source = a & m;
                    let want = if count >= bits {
                        0
                    } else if left {
                        (source << count) & m
                    } else {
                        source >> count
                    };
                    assert_eq!(h.cpu.opmask[1], want, "kshift{width} {count} left={left}");
                }
            }
        }
        for code in [KSHIFTLB_70, KSHIFTLW_70, KSHIFTLD_70, KSHIFTLQ_70] {
            assert_eq!(
                run_k(code, u64::MAX, 0).cpu.opmask[1],
                0,
                "count beyond the width"
            );
        }
    }

    #[test]
    fn kunpck_concatenates_the_low_halves() {
        for (code, half) in [(KUNPCKBW, 8u32), (KUNPCKWD, 16), (KUNPCKDQ, 32)] {
            for (a, b) in pairs() {
                let h = run_k(code, a, b);
                let m = (1u64 << half) - 1;
                // kunpck k1, k2, k3: k2 (vvvv) is the high half, k3 (r/m) the low half
                assert_eq!(h.cpu.opmask[1], (a & m) << half | b & m);
            }
        }
    }

    #[test]
    fn kmov_between_masks_memory_and_general_registers() {
        for (width, kk, km, mk, kr, rk) in [
            ('b', KMOVB_KK, KMOVB_KM, KMOVB_MK, KMOVB_KR, KMOVB_RK),
            ('w', KMOVW_KK, KMOVW_KM, KMOVW_MK, KMOVW_KR, KMOVW_RK),
            ('d', KMOVD_KK, KMOVD_KM, KMOVD_MK, KMOVD_KR, KMOVD_RK),
            ('q', KMOVQ_KK, KMOVQ_KM, KMOVQ_MK, KMOVQ_KR, KMOVQ_RK),
        ] {
            let m = width_mask(width);
            let bytes = (m.count_ones() / 8) as usize;
            let value = 0x1122_3344_5566_7788u64;
            // k <- k zero-extends
            let h = run_k(kk, value, 0);
            assert_eq!(h.cpu.opmask[1], value & m);
            // k <- memory reads exactly its width
            let h = run(km, |h| {
                h.write(DATA, &value.to_le_bytes());
                h.write(DATA + bytes as u64, &[0xff; 8]);
            });
            assert_eq!(h.cpu.opmask[1], value & m);
            // memory <- k writes exactly its width
            let h = run(mk, |h| {
                h.cpu.opmask[1] = value;
                h.write(OUT, &[0xee; 16]);
            });
            let stored = h.read(OUT, 16);
            assert_eq!(&stored[..bytes], &value.to_le_bytes()[..bytes]);
            assert!(
                stored[bytes..].iter().all(|&b| b == 0xee),
                "kmov{width} wrote too much"
            );
            // k <- general register: 32-bit source for b/w/d, 64-bit for q
            let h = run(kr, |h| h.cpu.gpr[0] = 0xffff_eeee_dddd_cccc);
            assert_eq!(h.cpu.opmask[1], 0xffff_eeee_dddd_cccc & m);
            // general register <- k zero-extends
            let h = run(rk, |h| {
                h.cpu.opmask[1] = u64::MAX;
                h.cpu.gpr[0] = 0;
            });
            assert_eq!(h.cpu.gpr[0], m);
        }
    }

    #[test]
    fn opmask_instructions_need_the_avx512_state() {
        let ud = Stop::Fault {
            vector: 6,
            error_code: None,
            software: false,
        };
        assert_eq!(
            stop_of(KANDW, |h| h.cpu.xcr0 = XCR0_X87 | XCR0_SSE | XCR0_AVX),
            ud
        );
        assert_eq!(stop_of(KANDW, |h| h.cpu.cr4 &= !CR4_OSXSAVE), ud);
    }
}

mod jit_matches_interpreter {
    use super::*;
    use crate::testkit::assert_jit_matches_interpreter;

    fn setup(h: &mut Harness) {
        h.cpu.cr4 = (h.cpu.cr4 & !(1 << 9)) | CR4_OSXSAVE;
        h.cpu.xcr0 = XCR0_SUPPORTED;
        h.cpu.gpr[RSI] = DATA;
        h.cpu.gpr[RDI] = OUT;
        let mut rng = Rng(0x7a7a);
        for n in 0..32 {
            h.cpu.set_zmm(n, rng.vector());
        }
        for k in 0..8 {
            h.cpu.opmask[k] = rng.next();
        }
        h.write(DATA, &bytes_of(&rng.vector()));
    }

    #[test]
    fn evex_instructions_run_identically_under_the_jit() {
        let codes: Vec<&[u8]> = vec![
            VPADDD.kz[2],
            VPADDD.mk[2],
            VPADDD.bk[2],
            VPADDD.high,
            VADDPS.k[2],
            VPSHUFB.p[2],
            VPXORQ.k[1],
            VMOVDQU8_M_Z_K,
            VMOVUPS_Z_M_KZ,
            VMOVDQU64_HIGH,
            VMOVD_X_EAX,
            VMOVSS_X_M_KZ,
            VADDSS_K1,
            VADDPS_RD_SAE,
            KANDW,
            KORTESTW,
            KSHIFTLQ_9,
            KUNPCKBW,
            KMOVQ_RK,
            VPSLLD_COUNT_Z,
            VPERMD.p[2],
            VPERMI2Q.k[2],
            VPTERNLOGD.kz[2],
            VPMOVDB.mk[2],
            VPMOVZXBW.k[1],
            VCVTPS2UDQ.k[2],
            VCVTPD2PS.p[2],
            VFMADD231PS.kz[2],
            VRNDSCALEPS.k[2],
            VGETEXPPS.p[2],
            VSCALEFPD.p[2],
            VCMPPS_1.k[2],
            VPCMPD_4.m[2],
            VPCOMPRESSD_M,
            VPEXPANDD_MZ,
            VPMOVM2D,
            VBROADCASTF32X4,
            VINSERTI64X2_ZM,
            VEXTRACTI64X4_ZMEM,
            VPEXTRQ_RAX,
            VPGATHERDD_Z,
            VPSCATTERDQ_Z,
            VCVTSI2SS_Q,
            VCVTTSD2SI_Q,
            VFIXUPIMMSS,
            VGF2P8AFFINEQB,
            VPDPBUSD,
            VPSHUFBITQMB,
        ];
        for code in codes {
            assert_jit_matches_interpreter(code, setup);
        }
    }

    #[test]
    fn a_block_of_evex_instructions_between_scalar_code() {
        // add rax, 1 ; vpaddd zmm0{k1}{z}, zmm1, zmm2 ; kmovw eax, k1 ; add rax, 2 ; hlt
        let mut code = vec![0x48, 0x83, 0xc0, 0x01];
        code.extend_from_slice(&VPADDD.kz[2][..VPADDD.kz[2].len() - 1]);
        code.extend_from_slice(&KMOVW_RK[..KMOVW_RK.len() - 1]);
        code.extend_from_slice(&[0x48, 0x83, 0xc0, 0x02, 0xf4]);
        assert_jit_matches_interpreter(&code, setup);
    }
}

#[test]
fn committing_without_a_mask_matches_committing_under_an_all_ones_mask() {
    let mut cpu = Harness::new(&[0xf4]).cpu;
    let result: Vector = [
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        0x1111_2222_3333_4444_5555_6666_7777_8888,
        0x9999_aaaa_bbbb_cccc_dddd_eeee_ffff_0000,
        0x0f1e_2d3c_4b5a_6978_8796_a5b4_c3d2_e1f0,
    ];
    let unmasked = Insn::default();
    let all_ones = Insn {
        evex_mask: 1,
        ..Default::default()
    };
    for bytes in [16, 32, 64] {
        for element_bytes in [1, 2, 4, 8] {
            // Stale content everywhere, so what the commit leaves above `bytes` shows.
            cpu.set_zmm(3, [u128::MAX; 4]);
            cpu.commit_masked(&unmasked, 3, element_bytes, bytes, &result);
            let without_mask = cpu.zmm(3);
            cpu.set_zmm(3, [u128::MAX; 4]);
            cpu.opmask[1] = u64::MAX;
            cpu.commit_masked(&all_ones, 3, element_bytes, bytes, &result);
            assert_eq!(without_mask, cpu.zmm(3), "{bytes} bytes of {element_bytes}");
            let mut expected = [0u128; 4];
            expected[..bytes / 16].copy_from_slice(&result[..bytes / 16]);
            assert_eq!(without_mask, expected, "{bytes} bytes of {element_bytes}");
        }
    }
}
