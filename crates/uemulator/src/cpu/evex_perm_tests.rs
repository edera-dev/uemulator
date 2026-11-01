//! Tests for permutes, blends, ternary logic, widening/narrowing, compress/expand and the
//! integer extensions (IFMA, VNNI, BITALG, VBMI, VBMI2, GFNI).

use super::*;

fn s(v: u64, size: usize) -> i64 {
    signed(v, size)
}

/// Runs `code` on three random vectors in zmm0 (old destination), zmm1 and zmm2 and returns them
/// with the final state.
fn run3(code: &[u8], seed: u64) -> (Vector, Vector, Vector, Harness) {
    let mut rng = Rng(seed);
    let (old, a, b) = (rng.vector(), rng.vector(), rng.vector());
    let h = run(code, |h| {
        h.cpu.set_zmm(0, old);
        h.cpu.set_zmm(1, a);
        h.cpu.set_zmm(2, b);
        h.cpu.opmask[1] = 0x5a5a_a5a5_0f0f_f0f0;
        h.write(DATA, &bytes_of(&b));
    });
    (old, a, b, h)
}

/// Index operands must stay in range to be interesting: keep the low bits of each element.
fn small_indices(seed: u64, element: usize, modulus: u64) -> Vector {
    let mut rng = Rng(seed);
    build(64, element, |_| rng.next() % modulus)
}

mod permutes {
    use super::*;

    #[test]
    fn vpermd_and_friends_index_the_whole_register() {
        for (name, variants, element) in [
            ("vpermd", &VPERMD, 4usize),
            ("vpermq", &VPERMQ_V, 8),
            ("vpermps", &VPERMPS, 4),
            ("vpermpd", &VPERMPD_V, 8),
        ] {
            let mut rng = Rng(0x71);
            let (index, table, old) = (rng.vector(), rng.vector(), rng.vector());
            for (position, length) in [(1usize, 32usize), (2, 64)] {
                let h = run(variants.p[position], |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(1, index);
                    h.cpu.set_zmm(2, table);
                });
                let count = length / element;
                let want = build(length, element, |k| {
                    element_of(
                        &table,
                        element,
                        element_of(&index, element, k) as usize & (count - 1),
                    )
                });
                assert_eq!(h.cpu.zmm(0), want, "{name} {length}");
            }
            // merge masking keeps the old elements
            let h = run(variants.k[2], |h| {
                h.cpu.set_zmm(0, old);
                h.cpu.set_zmm(1, index);
                h.cpu.set_zmm(2, table);
                h.cpu.opmask[1] = 0xaaaa;
            });
            for k in 0..64 / element {
                let want = if 0xaaaau64 >> k & 1 == 1 {
                    element_of(
                        &table,
                        element,
                        element_of(&index, element, k) as usize & (64 / element - 1),
                    )
                } else {
                    element_of(&old, element, k)
                };
                assert_eq!(
                    element_of(&h.cpu.zmm(0), element, k),
                    want,
                    "{name} masked {k}"
                );
            }
        }
    }

    fn element_of(v: &Vector, size: usize, k: usize) -> u64 {
        element(v, size, k)
    }

    #[test]
    fn vpermq_by_immediate_permutes_each_256_bit_half() {
        for (variants, imm) in [(&VPERMQ_I, 0x1bu64), (&VPERMPD_I, 0xc6)] {
            let mut rng = Rng(0x72);
            let source = rng.vector();
            let h = run(variants.p[2], |h| h.cpu.set_zmm(2, source));
            let want = build(64, 8, |k| {
                element(&source, 8, k / 4 * 4 + (imm >> (2 * (k % 4)) & 3) as usize)
            });
            assert_eq!(h.cpu.zmm(0), want);
        }
    }

    #[test]
    fn vpermi2_overwrites_the_index_register_and_vpermt2_the_first_table() {
        for (name, i2, t2, size) in [
            ("d", &VPERMI2D, &VPERMT2D, 4usize),
            ("q", &VPERMI2Q, &VPERMT2Q, 8),
        ] {
            let mut rng = Rng(0x73);
            let (first, second) = (rng.vector(), rng.vector());
            let index = small_indices(0x74, size, 64 / size as u64 * 2);
            let count = 64 / size;
            let select = |index_element: u64, low: &Vector, high: &Vector| {
                let table = if index_element as usize & count == 0 {
                    low
                } else {
                    high
                };
                element(table, size, index_element as usize & (count - 1))
            };
            // vpermi2: dest = indices, zmm1 and zmm2 the tables
            let h = run(i2.p[2], |h| {
                h.cpu.set_zmm(0, index);
                h.cpu.set_zmm(1, first);
                h.cpu.set_zmm(2, second);
            });
            let want = build(64, size, |k| {
                select(element(&index, size, k), &first, &second)
            });
            assert_eq!(h.cpu.zmm(0), want, "vpermi2{name}");
            // vpermt2: dest = first table, zmm1 = indices, zmm2 = second table
            let h = run(t2.p[2], |h| {
                h.cpu.set_zmm(0, first);
                h.cpu.set_zmm(1, index);
                h.cpu.set_zmm(2, second);
            });
            let want = build(64, size, |k| {
                select(element(&index, size, k), &first, &second)
            });
            assert_eq!(h.cpu.zmm(0), want, "vpermt2{name}");
        }
        // a 256-bit table has 8 dword elements: index bit 3 selects the second table
        let mut rng = Rng(0x75);
        let (first, second) = (rng.vector(), rng.vector());
        let index = small_indices(0x76, 4, 16);
        let h = run(VPERMT2D.p[1], |h| {
            h.cpu.set_zmm(0, first);
            h.cpu.set_zmm(1, index);
            h.cpu.set_zmm(2, second);
        });
        let want = build(32, 4, |k| {
            let selector = element(&index, 4, k) as usize;
            let table = if selector & 8 == 0 { &first } else { &second };
            element(table, 4, selector & 7)
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn byte_and_word_permutes() {
        let mut rng = Rng(0x77);
        let (index, table, second) = (rng.vector(), rng.vector(), rng.vector());
        let h = run(VPERMB.p[2], |h| {
            h.cpu.set_zmm(1, index);
            h.cpu.set_zmm(2, table);
        });
        let want = build(64, 1, |k| {
            element(&table, 1, element(&index, 1, k) as usize & 63)
        });
        assert_eq!(h.cpu.zmm(0), want);
        let h = run(VPERMW.p[2], |h| {
            h.cpu.set_zmm(1, index);
            h.cpu.set_zmm(2, table);
        });
        let want = build(64, 2, |k| {
            element(&table, 2, element(&index, 2, k) as usize & 31)
        });
        assert_eq!(h.cpu.zmm(0), want);
        let h = run(VPERMI2W.p[2], |h| {
            h.cpu.set_zmm(0, index);
            h.cpu.set_zmm(1, table);
            h.cpu.set_zmm(2, second);
        });
        let want = build(64, 2, |k| {
            let selector = element(&index, 2, k) as usize;
            element(
                if selector & 32 == 0 { &table } else { &second },
                2,
                selector & 31,
            )
        });
        assert_eq!(h.cpu.zmm(0), want);
        let h = run(VPERMI2B.p[2], |h| {
            h.cpu.set_zmm(0, index);
            h.cpu.set_zmm(1, table);
            h.cpu.set_zmm(2, second);
        });
        let want = build(64, 1, |k| {
            let selector = element(&index, 1, k) as usize;
            element(
                if selector & 64 == 0 { &table } else { &second },
                1,
                selector & 63,
            )
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn vpermil_variable_and_immediate_stay_inside_each_lane() {
        let mut rng = Rng(0x78);
        let (control, source) = (rng.vector(), rng.vector());
        let h = run(VPERMILPS_V.p[2], |h| {
            h.cpu.set_zmm(1, source);
            h.cpu.set_zmm(2, control);
        });
        let want = build(64, 4, |k| {
            element(
                &source,
                4,
                k / 4 * 4 + (element(&control, 4, k) & 3) as usize,
            )
        });
        assert_eq!(h.cpu.zmm(0), want, "vpermilps variable");
        let h = run(VPERMILPD_V.p[2], |h| {
            h.cpu.set_zmm(1, source);
            h.cpu.set_zmm(2, control);
        });
        // vpermilpd selects with bit 1 of each control quadword
        let want = build(64, 8, |k| {
            element(
                &source,
                8,
                k / 2 * 2 + (element(&control, 8, k) >> 1 & 1) as usize,
            )
        });
        assert_eq!(h.cpu.zmm(0), want, "vpermilpd variable");
        let h = run(VPERMILPS_I.p[2], |h| h.cpu.set_zmm(2, source));
        let want = build(64, 4, |k| {
            element(&source, 4, k / 4 * 4 + (0x1b >> (2 * (k % 4)) & 3))
        });
        assert_eq!(h.cpu.zmm(0), want, "vpermilps immediate");
        let h = run(VPERMILPD_I.p[2], |h| h.cpu.set_zmm(2, source));
        let want = build(64, 8, |k| element(&source, 8, k / 2 * 2 + (0x5 >> k & 1)));
        assert_eq!(h.cpu.zmm(0), want, "vpermilpd immediate");
    }

    #[test]
    fn valign_shifts_the_concatenation_right_by_elements() {
        let mut rng = Rng(0x79);
        let (high, low) = (rng.vector(), rng.vector());
        for (variants, element_size, imm) in [(&VALIGND, 4usize, 3usize), (&VALIGNQ, 8, 5)] {
            let h = run(variants.p[2], |h| {
                h.cpu.set_zmm(1, high);
                h.cpu.set_zmm(2, low);
            });
            let count = 64 / element_size;
            let want = build(64, element_size, |k| {
                let at = k + imm;
                if at < count {
                    element(&low, element_size, at)
                } else {
                    element(&high, element_size, at - count)
                }
            });
            assert_eq!(h.cpu.zmm(0), want, "valign{element_size}");
        }
    }

    #[test]
    fn shuffles_of_128_bit_lanes() {
        let mut rng = Rng(0x7a);
        let (a, b) = (rng.vector(), rng.vector());
        let h = run(VSHUFF32X4.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        // imm 0xb1: lanes 1, 0 from a; 3, 2 from b
        assert_eq!(h.cpu.zmm(0), [a[1], a[0], b[3], b[2]]);
        let h = run(VSHUFI64X2.p[2], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        // imm 0x1b: lanes 3, 2 from a; 1, 0 from b
        assert_eq!(h.cpu.zmm(0), [a[3], a[2], b[1], b[0]]);
        let h = run(VSHUFI64X2.p[1], |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        // 256-bit form: imm bit 0 picks a's lane, bit 1 picks b's lane
        assert_eq!(h.cpu.zmm(0), [a[1], b[1], 0, 0]);
    }
}

mod logic_and_blends {
    use super::*;

    #[test]
    fn vpternlog_evaluates_the_truth_table_per_bit() {
        for (name, variants, imm, element) in [
            ("d", &VPTERNLOGD, 0xcau8, 4usize),
            ("q", &VPTERNLOGQ, 0x96, 8),
        ] {
            let mut rng = Rng(0x7b);
            let (old, a, b) = (rng.vector(), rng.vector(), rng.vector());
            let h = run(variants.p[2], |h| {
                h.cpu.set_zmm(0, old);
                h.cpu.set_zmm(1, a);
                h.cpu.set_zmm(2, b);
            });
            let (o, x, y) = (bytes_of(&old), bytes_of(&a), bytes_of(&b));
            let mut want = [0u8; 64];
            for k in 0..64 {
                for bit in 0..8 {
                    let index = (o[k] >> bit & 1) << 2 | (x[k] >> bit & 1) << 1 | (y[k] >> bit & 1);
                    want[k] |= (imm >> index & 1) << bit;
                }
            }
            assert_eq!(bytes_of(&h.cpu.zmm(0)), want, "vpternlog{name}");
            // masked: element granularity follows W
            let h = run(variants.kz[2], |h| {
                h.cpu.set_zmm(0, old);
                h.cpu.set_zmm(1, a);
                h.cpu.set_zmm(2, b);
                h.cpu.opmask[1] = 0x0f0f;
            });
            let got = bytes_of(&h.cpu.zmm(0));
            for k in 0..64 / element {
                let selected = 0x0f0fu64 >> k & 1 == 1;
                for byte in 0..element {
                    let at = k * element + byte;
                    assert_eq!(
                        got[at],
                        if selected { want[at] } else { 0 },
                        "{name} element {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn blendm_takes_masked_elements_from_the_source_and_others_from_vvvv() {
        for (name, variants, element_size) in [
            ("d", &VPBLENDMD, 4usize),
            ("q", &VPBLENDMQ, 8),
            ("b", &VPBLENDMB, 1),
            ("w", &VPBLENDMW, 2),
            ("ps", &VBLENDMPS, 4),
        ] {
            let mut rng = Rng(0x7c);
            let (old, a, b) = (rng.vector(), rng.vector(), rng.vector());
            let k1 = 0x5a5a_a5a5_0f0f_f0f0u64;
            for (zeroing, code) in [(false, variants.k[2]), (true, variants.kz[2])] {
                let h = run(code, |h| {
                    h.cpu.set_zmm(0, old);
                    h.cpu.set_zmm(1, a);
                    h.cpu.set_zmm(2, b);
                    h.cpu.opmask[1] = k1;
                });
                let want = build(64, element_size, |k| {
                    if k1 >> k & 1 == 1 {
                        element(&b, element_size, k)
                    } else if zeroing {
                        0
                    } else {
                        element(&a, element_size, k)
                    }
                });
                assert_eq!(h.cpu.zmm(0), want, "vpblendm{name} zeroing={zeroing}");
            }
            // without a mask register the whole result is the source
            let h = run(variants.p[2], |h| {
                h.cpu.set_zmm(1, a);
                h.cpu.set_zmm(2, b);
            });
            assert_eq!(h.cpu.zmm(0), b);
        }
    }
}

mod conversions_of_width {
    use super::*;

    #[test]
    fn pmovzx_and_pmovsx_widen_every_element() {
        for (name, variants, from, to, sign) in [
            ("bw", &VPMOVZXBW, 1usize, 2usize, false),
            ("bd", &VPMOVSXBD, 1, 4, true),
            ("bq", &VPMOVZXBQ, 1, 8, false),
            ("wd", &VPMOVSXWD, 2, 4, true),
            ("wq", &VPMOVZXWQ, 2, 8, false),
            ("dq s", &VPMOVSXDQ, 4, 8, true),
            ("dq z", &VPMOVZXDQ, 4, 8, false),
            ("bw s", &VPMOVSXBW, 1, 2, true),
        ] {
            let mut rng = Rng(0x7d);
            let (old, source) = (rng.vector(), rng.vector());
            let k1 = rng.next();
            for (index, length) in LENGTHS.into_iter().enumerate() {
                for (shape, code) in [("plain", variants.p[index]), ("zero", variants.kz[index])] {
                    if code.is_empty() {
                        continue;
                    }
                    let h = run(code, |h| {
                        h.cpu.set_zmm(0, old);
                        h.cpu.set_zmm(2, source);
                        h.cpu.opmask[1] = k1;
                    });
                    let want = expected(
                        length,
                        to,
                        (shape == "zero").then_some((k1, true)),
                        &old,
                        |k| {
                            let v = element(&source, from, k);
                            if sign {
                                s(v, from) as u64 & size_mask(to)
                            } else {
                                v
                            }
                        },
                    );
                    assert_eq!(h.cpu.zmm(0), want, "vpmov{name} {length} {shape}");
                }
            }
        }
    }

    #[test]
    fn masked_pmovzx_from_memory_reads_only_the_selected_elements() {
        let mut rng = Rng(0x7e);
        let source = rng.vector();
        let h = run(VPMOVZXBW_M_Z, |h| {
            h.write(DATA, &bytes_of(&source));
            h.cpu.opmask[1] = 0x0000_ffff;
        });
        let want = build(64, 2, |k| if k < 16 { element(&source, 1, k) } else { 0 });
        assert_eq!(h.cpu.zmm(0), want);
        // the upper 16 source bytes are in an unmapped page: masked out they are not read
        let edge = 0x180_0000u64 - 16;
        let mut h = rig(VPMOVZXBW_M_Z);
        h.write(edge, &[7; 16]);
        crate::cpu::test_support::unmap_boot_identity_2m(&mut h, 0x180_0000);
        h.cpu.gpr[RSI] = edge;
        h.cpu.opmask[1] = 0xffff;
        assert_eq!(h.run(10), Stop::Halted);
        let mut h = rig(VPMOVZXBW_M_Z);
        crate::cpu::test_support::unmap_boot_identity_2m(&mut h, 0x180_0000);
        h.cpu.gpr[RSI] = edge;
        h.cpu.opmask[1] = 0xffff_ffff;
        assert!(matches!(h.run(10), Stop::Fault { vector: 14, .. }));
    }

    #[test]
    fn down_converts_truncate_or_saturate_and_zero_the_rest_of_the_register() {
        for (name, variants, from, to, kind) in [
            ("db", &VPMOVDB, 4usize, 1usize, 't'),
            ("qd", &VPMOVQD, 8, 4, 't'),
            ("wb", &VPMOVWB, 2, 1, 't'),
            ("qb", &VPMOVQB, 8, 1, 't'),
            ("qw", &VPMOVQW, 8, 2, 't'),
            ("dw", &VPMOVDW, 4, 2, 't'),
            ("sdb", &VPMOVSDB, 4, 1, 's'),
            ("usdb", &VPMOVUSDB, 4, 1, 'u'),
            ("sqd", &VPMOVSQD, 8, 4, 's'),
            ("usqw", &VPMOVUSQW, 8, 2, 'u'),
            ("swb", &VPMOVSWB, 2, 1, 's'),
            ("uswb", &VPMOVUSWB, 2, 1, 'u'),
            ("usqd", &VPMOVUSQD, 8, 4, 'u'),
            ("sqb", &VPMOVSQB, 8, 1, 's'),
        ] {
            let mut rng = Rng(0x7f ^ name.len() as u64);
            let mut source = rng.vector();
            // make sure some elements are small and some negative so saturation shows both ways
            for k in 0..64 / from {
                if k % 3 == 0 {
                    set_element(&mut source, from, k, k as u64 * 5);
                }
            }
            let old = rng.vector();
            let k1 = rng.next();
            let convert = |v: u64| -> u64 {
                let bits = to * 8;
                match kind {
                    't' => v & size_mask(to),
                    's' => {
                        (s(v, from).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)) as u64
                            & size_mask(to)
                    }
                    _ => v.min(size_mask(to)),
                }
            };
            for (index, length) in LENGTHS.into_iter().enumerate() {
                let count = length / from;
                let result_bytes = count * to;
                for (shape, code, masked, zeroing) in [
                    ("plain", variants.p[index], false, false),
                    ("merge", variants.k[index], true, false),
                    ("zero", variants.kz[index], true, true),
                ] {
                    if code.is_empty() {
                        continue;
                    }
                    let h = run(code, |h| {
                        h.cpu.set_zmm(0, old);
                        h.cpu.set_zmm(1, source);
                        h.cpu.opmask[1] = k1;
                    });
                    let want = expected(
                        result_bytes,
                        to,
                        masked.then_some((k1, zeroing)),
                        &old,
                        |k| convert(element(&source, from, k)),
                    );
                    assert_eq!(h.cpu.zmm(0), want, "vpmov{name} {length} {shape}");
                }
                // memory destination: only the selected elements are written, nothing else
                for (code, masked) in [(variants.m[index], false), (variants.mk[index], true)] {
                    if code.is_empty() {
                        continue;
                    }
                    let h = run(code, |h| {
                        h.cpu.set_zmm(1, source);
                        h.cpu.opmask[1] = k1;
                        h.write(OUT, &[0xee; 64]);
                    });
                    let stored = h.read(OUT, 64);
                    for k in 0..count {
                        let want = if !masked || k1 >> k & 1 == 1 {
                            convert(element(&source, from, k))
                        } else {
                            size_mask(to) & 0xeeee_eeee_eeee_eeee
                        };
                        let mut got = 0u64;
                        for byte in 0..to {
                            got |= (stored[k * to + byte] as u64) << (8 * byte);
                        }
                        assert_eq!(got, want, "vpmov{name} store {length} element {k}");
                    }
                    assert!(
                        stored[result_bytes..].iter().all(|&b| b == 0xee),
                        "wrote past the result"
                    );
                }
            }
        }
    }
}

mod compress_expand_and_masks {
    use super::*;

    const K1: u64 = 0b1011_0110_1100_0101;

    #[test]
    fn compress_packs_selected_elements_to_the_low_end() {
        let mut rng = Rng(0x81);
        let (old, source) = (rng.vector(), rng.vector());
        let selected: Vec<u64> = (0..16)
            .filter(|k| K1 >> k & 1 == 1)
            .map(|k| element(&source, 4, k))
            .collect();
        // register destination: the rest merges from the old destination ...
        let h = run(VPCOMPRESSD_R, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = K1;
        });
        let want = build(64, 4, |k| {
            selected.get(k).copied().unwrap_or(element(&old, 4, k))
        });
        assert_eq!(h.cpu.zmm(0), want);
        // ... or is zeroed
        let h = run(VPCOMPRESSD_RZ, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = K1;
        });
        let want = build(64, 4, |k| selected.get(k).copied().unwrap_or(0));
        assert_eq!(h.cpu.zmm(0), want);
        // memory destination: exactly the packed elements, nothing past them
        let h = run(VPCOMPRESSD_M, |h| {
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = K1;
            h.write(OUT, &[0xee; 64]);
        });
        let stored = h.read(OUT, 64);
        for (k, value) in selected.iter().enumerate() {
            assert_eq!(&stored[4 * k..4 * k + 4], &(*value as u32).to_le_bytes());
        }
        assert!(stored[4 * selected.len()..].iter().all(|&b| b == 0xee));
        let h = run(VCOMPRESSPD_M, |h| {
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = 0b1001_0010;
            h.write(OUT, &[0xee; 64]);
        });
        let stored = h.read(OUT, 64);
        for (n, k) in [1usize, 4, 7].into_iter().enumerate() {
            assert_eq!(
                &stored[8 * n..8 * n + 8],
                &element(&source, 8, k).to_le_bytes()
            );
        }
        assert!(stored[24..].iter().all(|&b| b == 0xee));
        // 256-bit quadwords, byte and word elements
        let h = run(VPCOMPRESSQ_Y_R, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = 0b1010;
        });
        let got = h.cpu.zmm(0);
        assert_eq!(element(&got, 8, 0), element(&source, 8, 1));
        assert_eq!(element(&got, 8, 1), element(&source, 8, 3));
        assert_eq!(element(&got, 8, 2), element(&old, 8, 2));
        assert_eq!(&got[2..], &[0, 0]);
        let h = run(VPCOMPRESSB_M, |h| {
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = 0xf0f0_0000_0000_00ff;
            h.write(OUT, &[0xee; 64]);
        });
        let stored = h.read(OUT, 64);
        let picks: Vec<usize> = (0..64)
            .filter(|k| 0xf0f0_0000_0000_00ffu64 >> k & 1 == 1)
            .collect();
        for (n, &k) in picks.iter().enumerate() {
            assert_eq!(stored[n] as u64, element(&source, 1, k));
        }
        assert!(stored[picks.len()..].iter().all(|&b| b == 0xee));
        let h = run(VPCOMPRESSW_R, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = 0x8000_0001;
        });
        assert_eq!(element(&h.cpu.zmm(0), 2, 0), element(&source, 2, 0));
        assert_eq!(element(&h.cpu.zmm(0), 2, 1), element(&source, 2, 31));
        assert_eq!(element(&h.cpu.zmm(0), 2, 2), element(&old, 2, 2));
    }

    #[test]
    fn expand_spreads_consecutive_elements_to_the_selected_positions() {
        let mut rng = Rng(0x82);
        let (old, source) = (rng.vector(), rng.vector());
        let count = K1.count_ones() as usize;
        let place = |fill: &dyn Fn(usize) -> u64, size: usize, mask: u64, elements: usize| {
            let mut taken = 0;
            build(elements * size, size, |k| {
                if mask >> k & 1 == 1 {
                    taken += 1;
                    element(&source, size, taken - 1)
                } else {
                    fill(k)
                }
            })
        };
        let h = run(VPEXPANDD_R, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = K1;
        });
        assert_eq!(h.cpu.zmm(0), place(&|k| element(&old, 4, k), 4, K1, 16));
        let h = run(VPEXPANDD_RZ, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = K1;
        });
        assert_eq!(h.cpu.zmm(0), place(&|_| 0, 4, K1, 16));
        // memory source: exactly popcount(mask) elements are read
        let h = run(VPEXPANDD_M, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = K1;
            h.write(DATA, &bytes_of(&source));
        });
        assert_eq!(h.cpu.zmm(0), place(&|k| element(&old, 4, k), 4, K1, 16));
        let mut guard = rig(VPEXPANDD_MZ);
        let edge = 0x180_0000u64 - 4 * count as u64;
        guard.write(edge, &bytes_of(&source)[..4 * count]);
        crate::cpu::test_support::unmap_boot_identity_2m(&mut guard, 0x180_0000);
        guard.cpu.gpr[RSI] = edge;
        guard.cpu.opmask[1] = K1;
        assert_eq!(
            guard.run(10),
            Stop::Halted,
            "elements beyond the count are not read"
        );
        assert_eq!(guard.cpu.zmm(0), place(&|_| 0, 4, K1, 16));
        let h = run(VEXPANDPS_M, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = 0xffff;
            h.write(DATA, &bytes_of(&source));
        });
        assert_eq!(h.cpu.zmm(0), source, "a full mask is a plain load");
        let h = run(VPEXPANDQ_Y_M, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = 0b0101;
            h.write(DATA, &bytes_of(&source));
        });
        let got = h.cpu.zmm(0);
        assert_eq!(element(&got, 8, 0), element(&source, 8, 0));
        assert_eq!(element(&got, 8, 1), element(&old, 8, 1));
        assert_eq!(element(&got, 8, 2), element(&source, 8, 1));
        assert_eq!(&got[2..], &[0, 0]);
        let h = run(VPEXPANDB_M, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.opmask[1] = 0xaaaa_aaaa_aaaa_aaaa;
            h.write(DATA, &bytes_of(&source));
        });
        assert_eq!(
            h.cpu.zmm(0),
            place(&|k| element(&old, 1, k), 1, 0xaaaa_aaaa_aaaa_aaaa, 64)
        );
        let h = run(VPEXPANDW_R, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, source);
            h.cpu.opmask[1] = 0xf000_000f;
        });
        assert_eq!(
            h.cpu.zmm(0),
            place(&|k| element(&old, 2, k), 2, 0xf000_000f, 32)
        );
    }

    #[test]
    fn vector_to_mask_and_back() {
        let mut rng = Rng(0x83);
        let v = rng.vector();
        for (code, size, length) in [
            (VPMOVB2M, 1usize, 64usize),
            (VPMOVW2M, 2, 32),
            (VPMOVD2M, 4, 64),
            (VPMOVQ2M, 8, 16),
        ] {
            let h = run(code, |h| {
                h.cpu.set_zmm(1, v);
                h.cpu.opmask[2] = u64::MAX;
            });
            let mut want = 0u64;
            for k in 0..length / size {
                want |= (element(&v, size, k) >> (size * 8 - 1) & 1) << k;
            }
            assert_eq!(h.cpu.opmask[2], want, "vpmov{size}2m");
        }
        for (code, size, length) in [
            (VPMOVM2B, 1usize, 64usize),
            (VPMOVM2W, 2, 32),
            (VPMOVM2D, 4, 64),
            (VPMOVM2Q, 8, 16),
        ] {
            let h = run(code, |h| {
                h.cpu.set_zmm(0, [u128::MAX; 4]);
                h.cpu.opmask[1] = 0x9e37_79b9_7f4a_7c15;
            });
            let want = build(length, size, |k| {
                if 0x9e37_79b9_7f4a_7c15u64 >> k & 1 == 1 {
                    size_mask(size)
                } else {
                    0
                }
            });
            assert_eq!(h.cpu.zmm(0), want, "vpmovm2 {size}");
        }
    }
}

mod extensions {
    use super::*;

    #[test]
    fn conflict_detection_lists_earlier_equal_elements() {
        let mut rng = Rng(0x91);
        let mut source = rng.vector();
        for k in 0..16 {
            set_element(&mut source, 4, k, rng.next() % 4);
        }
        let h = run(VPCONFLICTD.p[2], |h| h.cpu.set_zmm(2, source));
        let want = build(64, 4, |k| {
            (0..k)
                .filter(|&j| element(&source, 4, j) == element(&source, 4, k))
                .fold(0, |m, j| m | 1 << j)
        });
        assert_eq!(h.cpu.zmm(0), want);
        let mut quad = rng.vector();
        for k in 0..8 {
            set_element(&mut quad, 8, k, rng.next() % 3);
        }
        let h = run(VPCONFLICTQ.p[2], |h| h.cpu.set_zmm(2, quad));
        let want = build(64, 8, |k| {
            (0..k)
                .filter(|&j| element(&quad, 8, j) == element(&quad, 8, k))
                .fold(0, |m, j| m | 1 << j)
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn mask_registers_broadcast_into_vectors() {
        let k = 0x1234_5678_9abc_def0u64;
        let h = run(VPBROADCASTMB2Q_Z, |h| h.cpu.opmask[1] = k);
        assert_eq!(
            h.cpu.zmm(0),
            [(k & 0xff) as u128 * 0x1_0000_0000_0000_0001; 4]
        );
        let h = run(VPBROADCASTMW2D_Y, |h| {
            h.cpu.opmask[1] = k;
            h.cpu.set_zmm(0, [u128::MAX; 4]);
        });
        let want = build(32, 4, |_| k & 0xffff);
        assert_eq!(h.cpu.zmm(0), [want[0], want[1], 0, 0]);
        let h = run(VPBROADCASTMW2D_HIGH, |h| h.cpu.opmask[1] = k);
        assert_eq!(h.cpu.zmm(20), build(64, 4, |_| k & 0xffff));
    }

    #[test]
    fn ifma_accumulates_the_low_or_high_half_of_the_52_bit_product() {
        let (old, a, b, h) = run3(VPMADD52LUQ, 0x92);
        let mask52 = (1u64 << 52) - 1;
        let want = build(64, 8, |k| {
            let product =
                (element(&a, 8, k) & mask52) as u128 * (element(&b, 8, k) & mask52) as u128;
            element(&old, 8, k).wrapping_add(product as u64 & mask52)
        });
        assert_eq!(h.cpu.zmm(0), want);
        let (old, a, b, h) = run3(VPMADD52HUQ, 0x93);
        let k1 = 0x5a5a_a5a5_0f0f_f0f0u64;
        let want = build(64, 8, |k| {
            if k1 >> k & 1 == 0 {
                return element(&old, 8, k);
            }
            let product =
                (element(&a, 8, k) & mask52) as u128 * (element(&b, 8, k) & mask52) as u128;
            element(&old, 8, k).wrapping_add((product >> 52) as u64 & mask52)
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn vnni_dot_products() {
        let (old, a, b, h) = run3(VPDPBUSD, 0x94);
        let want = build(64, 4, |k| {
            let sum: i64 = (0..4)
                .map(|j| element(&a, 1, 4 * k + j) as i64 * s(element(&b, 1, 4 * k + j), 1))
                .sum();
            (s(element(&old, 4, k), 4) + sum) as u64 & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpdpbusd");
        let (old, a, b, h) = run3(VPDPBUSDS, 0x95);
        let want = build(64, 4, |k| {
            let sum: i64 = (0..4)
                .map(|j| element(&a, 1, 4 * k + j) as i64 * s(element(&b, 1, 4 * k + j), 1))
                .sum();
            (s(element(&old, 4, k), 4) + sum).clamp(i32::MIN as i64, i32::MAX as i64) as u64
                & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpdpbusds");
        let (old, a, b, h) = run3(VPDPWSSD, 0x96);
        let want = build(64, 4, |k| {
            let sum: i64 = (0..2)
                .map(|j| s(element(&a, 2, 2 * k + j), 2) * s(element(&b, 2, 2 * k + j), 2))
                .sum();
            (s(element(&old, 4, k), 4) + sum) as u64 & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpdpwssd");
        // saturating word form under a merge mask, with accumulators near the limit
        let mut rng = Rng(0x97);
        let (mut old, a, b) = (rng.vector(), rng.vector(), rng.vector());
        for k in (0..16).step_by(2) {
            set_element(&mut old, 4, k, 0x7fff_fff0);
        }
        let h = run(VPDPWSSDS, |h| {
            h.cpu.set_zmm(0, old);
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
            h.cpu.opmask[1] = 0xff00;
        });
        let want = build(64, 4, |k| {
            if 0xff00u64 >> k & 1 == 0 {
                return element(&old, 4, k);
            }
            let sum: i64 = (0..2)
                .map(|j| s(element(&a, 2, 2 * k + j), 2) * s(element(&b, 2, 2 * k + j), 2))
                .sum();
            (s(element(&old, 4, k), 4) + sum).clamp(i32::MIN as i64, i32::MAX as i64) as u64
                & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpdpwssds");
    }

    #[test]
    fn bit_shuffles_and_multishift() {
        let mut rng = Rng(0x98);
        let (data, index) = (rng.vector(), rng.vector());
        let h = run(VPSHUFBITQMB, |h| {
            h.cpu.set_zmm(1, data);
            h.cpu.set_zmm(2, index);
        });
        let mut want = 0u64;
        for k in 0..64 {
            want |= (element(&data, 8, k / 8) >> (element(&index, 1, k) & 63) & 1) << k;
        }
        assert_eq!(h.cpu.opmask[2], want);
        let h = run(VPMULTISHIFTQB, |h| {
            h.cpu.set_zmm(1, index);
            h.cpu.set_zmm(2, data);
        });
        let want = build(64, 1, |k| {
            element(&data, 8, k / 8).rotate_right(element(&index, 1, k) as u32 & 63) & 0xff
        });
        assert_eq!(h.cpu.zmm(0), want);
    }

    #[test]
    fn concatenate_and_shift_by_immediate_and_by_element() {
        let (old, a, b, h) = run3(VPSHLDD, 0x99);
        let _ = old;
        let want = build(64, 4, |k| {
            let (x, y) = (element(&a, 4, k), element(&b, 4, k));
            (((x << 32 | y) << 9) >> 32) & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshldd");
        let (_, a, b, h) = run3(VPSHLDW, 0x9a);
        let want = build(64, 2, |k| {
            let (x, y) = (element(&a, 2, k), element(&b, 2, k));
            (((x << 16 | y) << 5) >> 16) & 0xffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshldw");
        let (_, a, b, h) = run3(VPSHRDQ, 0x9b);
        let want = build(64, 8, |k| {
            let (x, y) = (element(&a, 8, k) as u128, element(&b, 8, k) as u128);
            ((y << 64 | x) >> 13) as u64
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshrdq");
        let (_, a, b, h) = run3(VPSHRDW, 0x9c);
        let want = build(64, 2, |k| {
            let (x, y) = (element(&a, 2, k), element(&b, 2, k));
            ((y << 16 | x) >> 3) & 0xffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshrdw");
        // variable forms: vvvv is the count, the destination the first operand
        let (old, counts, b, h) = run3(VPSHLDVD, 0x9d);
        let want = build(64, 4, |k| {
            let n = element(&counts, 4, k) & 31;
            let wide = element(&old, 4, k) << 32 | element(&b, 4, k);
            ((wide << n) >> 32) & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshldvd");
        let (old, counts, b, h) = run3(VPSHRDVQ, 0x9e);
        let want = build(64, 8, |k| {
            let n = element(&counts, 8, k) & 63;
            let wide = (element(&b, 8, k) as u128) << 64 | element(&old, 8, k) as u128;
            (wide >> n) as u64
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshrdvq");
        let (old, counts, b, h) = run3(VPSHLDVW, 0x9f);
        let want = build(64, 2, |k| {
            let n = element(&counts, 2, k) & 15;
            let wide = element(&old, 2, k) << 16 | element(&b, 2, k);
            ((wide << n) >> 16) & 0xffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshldvw");
        let (old, counts, b, h) = run3(VPSHRDVD, 0xa0);
        let want = build(64, 4, |k| {
            let n = element(&counts, 4, k) & 31;
            let wide = element(&b, 4, k) << 32 | element(&old, 4, k);
            (wide >> n) & 0xffff_ffff
        });
        assert_eq!(h.cpu.zmm(0), want, "vpshrdvd");
    }

    /// GF(2^8) multiply by the schoolbook definition with the AES polynomial.
    fn gf_product(a: u8, b: u8) -> u8 {
        let mut result = 0u16;
        for bit in 0..8 {
            if b >> bit & 1 == 1 {
                result ^= (a as u16) << bit;
            }
        }
        for bit in (8..16).rev() {
            if result >> bit & 1 == 1 {
                result ^= 0x11b << (bit - 8);
            }
        }
        result as u8
    }

    #[test]
    fn gfni_vaes_and_vpclmulqdq_exist_in_the_legacy_and_vex_encodings_too() {
        let mut rng = Rng(0xd1);
        let (a, b) = (rng.vector(), rng.vector());
        // legacy SSE: data in the destination, matrix in the source
        let h = run(SSE_GF2P8MULB, |h| {
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            h.cpu.set_zmm(0, a);
            h.cpu.set_zmm(1, b);
        });
        let want = build(16, 1, |k| {
            gf_product(element(&a, 1, k) as u8, element(&b, 1, k) as u8) as u64
        });
        assert_eq!(h.cpu.zmm(0)[0], want[0]);
        assert_eq!(
            h.cpu.zmm(0)[1..],
            a[1..],
            "legacy SSE leaves the upper bits alone"
        );
        let matrix = |v: &Vector, k: usize| element(v, 8, k / 8);
        let affine = |matrix: u64, x: u8| {
            let mut out = 0u8;
            for bit in 0..8 {
                let row = (matrix >> (8 * (7 - bit))) as u8;
                out |= (((row & x).count_ones() as u8 & 1) ^ (0x63 >> bit & 1)) << bit;
            }
            out
        };
        let h = run(SSE_GF2P8AFFINEQB, |h| {
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            h.cpu.set_zmm(0, a);
            h.cpu.set_zmm(1, b);
        });
        let want = build(16, 1, |k| {
            affine(matrix(&b, k), element(&a, 1, k) as u8) as u64
        });
        assert_eq!(h.cpu.zmm(0)[0], want[0]);
        let h = run(SSE_GF2P8AFFINEINVQB, |h| {
            h.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
            h.cpu.set_zmm(0, a);
            h.cpu.set_zmm(1, b);
        });
        let inverse = |x: u8| (0..=255u8).find(|&y| gf_product(x, y) == 1).unwrap_or(0);
        let want = build(16, 1, |k| {
            affine(matrix(&b, k), inverse(element(&a, 1, k) as u8)) as u64
        });
        assert_eq!(h.cpu.zmm(0)[0], want[0]);
        // VEX.256 forms run each 128-bit lane
        let h = run(VEX_GF2P8MULB_Y, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = build(32, 1, |k| {
            gf_product(element(&a, 1, k) as u8, element(&b, 1, k) as u8) as u64
        });
        assert_eq!(h.cpu.zmm(0), [want[0], want[1], 0, 0]);
        let h = run(VEX_GF2P8AFFINEQB_Y, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = build(32, 1, |k| {
            affine(matrix(&b, k), element(&a, 1, k) as u8) as u64
        });
        assert_eq!(h.cpu.zmm(0), [want[0], want[1], 0, 0]);
        let h = run(VEX_GF2P8AFFINEINVQB_X, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        let want = build(16, 1, |k| {
            affine(matrix(&b, k), inverse(element(&a, 1, k) as u8)) as u64
        });
        assert_eq!(h.cpu.zmm(0), [want[0], 0, 0, 0]);
        // VAES: a 256-bit AES round is two independent 128-bit rounds
        let h = run(VEX_VAESENC_Y, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        for lane in 0..2 {
            let legacy = run(&[0x66, 0x0f, 0x38, 0xdc, 0xc1, 0xf4], |l| {
                l.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
                l.cpu.set_zmm(0, [a[lane], 0, 0, 0]);
                l.cpu.set_zmm(1, [b[lane], 0, 0, 0]);
            });
            assert_eq!(h.cpu.zmm(0)[lane], legacy.cpu.zmm(0)[0], "lane {lane}");
        }
        assert_eq!(h.cpu.zmm(0)[2..], [0, 0]);
        let h = run(VEX_VAESDECLAST_YM, |h| {
            h.cpu.set_zmm(1, a);
            h.write(DATA, &bytes_of(&b));
        });
        for lane in 0..2 {
            let legacy = run(&[0x66, 0x0f, 0x38, 0xdf, 0xc1, 0xf4], |l| {
                l.cpu.cr4 |= crate::cpu::sys::CR4_OSFXSR;
                l.cpu.set_zmm(0, [a[lane], 0, 0, 0]);
                l.cpu.set_zmm(1, [b[lane], 0, 0, 0]);
            });
            assert_eq!(
                h.cpu.zmm(0)[lane],
                legacy.cpu.zmm(0)[0],
                "memory lane {lane}"
            );
        }
        // VPCLMULQDQ with imm 0x11: high quadwords of each lane
        let h = run(VEX_VPCLMULQDQ_Y, |h| {
            h.cpu.set_zmm(1, a);
            h.cpu.set_zmm(2, b);
        });
        for lane in 0..2 {
            let mut product = 0u128;
            for bit in 0..64 {
                if (b[lane] >> 64) as u64 >> bit & 1 == 1 {
                    product ^= ((a[lane] >> 64) as u64 as u128) << bit;
                }
            }
            assert_eq!(h.cpu.zmm(0)[lane], product, "lane {lane}");
        }
        // aesimc stays 128-bit only; its VEX.256 form is undefined
        let mut code = VEX_AESIMC_X.to_vec();
        code[2] |= 0x04;
        assert!(matches!(
            stop_of(&code, |_| {}),
            Stop::Fault { vector: 6, .. }
        ));
    }

    #[test]
    fn galois_field_instructions() {
        let (_, a, b, h) = run3(VGF2P8MULB, 0xa1);
        let want = build(64, 1, |k| {
            gf_product(element(&a, 1, k) as u8, element(&b, 1, k) as u8) as u64
        });
        assert_eq!(h.cpu.zmm(0), want, "vgf2p8mulb");
        // the multiplicative inverse by search
        let inverse = |x: u8| (0..=255u8).find(|&y| gf_product(x, y) == 1).unwrap_or(0);
        let affine = |matrix: u64, x: u8, imm: u8| {
            let mut out = 0u8;
            for bit in 0..8 {
                let row = (matrix >> (8 * (7 - bit))) as u8;
                out |= (((row & x).count_ones() as u8 & 1) ^ (imm >> bit & 1)) << bit;
            }
            out
        };
        let (_, a, b, h) = run3(VGF2P8AFFINEQB, 0xa2);
        let want = build(64, 1, |k| {
            affine(element(&b, 8, k / 8), element(&a, 1, k) as u8, 0x63) as u64
        });
        assert_eq!(h.cpu.zmm(0), want, "vgf2p8affineqb");
        let (_, a, b, h) = run3(VGF2P8AFFINEINVQB, 0xa3);
        let want = build(64, 1, |k| {
            affine(
                element(&b, 8, k / 8),
                inverse(element(&a, 1, k) as u8),
                0x63,
            ) as u64
        });
        assert_eq!(h.cpu.zmm(0), want, "vgf2p8affineinvqb");
        // the AES S-box is the affine transform of the inverse with the identity-ish matrix:
        // sbox(0) = 0x63, sbox(1) = 0x7c
        let aes_matrix = 0xf1e3_c78f_1f3e_7cf8u64;
        let (_, _, _, h) = {
            let mut h = rig(VGF2P8AFFINEINVQB);
            let mut data = [0u8; 64];
            data[0] = 0;
            data[1] = 1;
            h.cpu.set_zmm(1, vector_of(&data));
            h.cpu
                .set_zmm(2, [(aes_matrix as u128) << 64 | aes_matrix as u128; 4]);
            assert_eq!(h.run(10), Stop::Halted);
            (0, 0, 0, h)
        };
        let out = bytes_of(&h.cpu.zmm(0));
        assert_eq!((out[0], out[1]), (0x63, 0x7c), "AES S-box of 0 and 1");
    }
}

/// AVX512_VP2INTERSECT. Encodings from the assembler (`hack/asm.sh`); the reference is the SDM's
/// double loop over element pairs.
mod pair_intersection {
    use super::*;

    /// vp2intersectd k2, zmm1, zmm2
    const VP2INTERSECTD_Z: &[u8] = &[0x62, 0xf2, 0x77, 0x48, 0x68, 0xd2, 0xf4];
    /// vp2intersectq k2, zmm1, zmm2
    const VP2INTERSECTQ_Z: &[u8] = &[0x62, 0xf2, 0xf7, 0x48, 0x68, 0xd2, 0xf4];
    /// vp2intersectq k6, xmm1, xmm2
    const VP2INTERSECTQ_X: &[u8] = &[0x62, 0xf2, 0xf7, 0x08, 0x68, 0xf2, 0xf4];
    /// vp2intersectd k0, zmm1, zmmword ptr [rsi]
    const VP2INTERSECTD_Z_MEMORY: &[u8] = &[0x62, 0xf2, 0x77, 0x48, 0x68, 0x06, 0xf4];
    /// vp2intersectd k4, ymm1, dword ptr [rsi]{1to8}
    const VP2INTERSECTD_Y_BROADCAST: &[u8] = &[0x62, 0xf2, 0x77, 0x38, 0x68, 0x26, 0xf4];
    /// VP2INTERSECTD_Z with ModRM.reg = 3: the pair is still k2, k3.
    const VP2INTERSECTD_Z_ODD: &[u8] = &[0x62, 0xf2, 0x77, 0x48, 0x68, 0xda, 0xf4];
    /// VP2INTERSECTD_Z with EVEX.aaa = 1, and with EVEX.z set.
    const VP2INTERSECTD_Z_MASKED: &[u8] = &[0x62, 0xf2, 0x77, 0x49, 0x68, 0xd2, 0xf4];
    const VP2INTERSECTD_Z_ZEROING: &[u8] = &[0x62, 0xf2, 0x77, 0xc8, 0x68, 0xd2, 0xf4];

    const OPMASK_JUNK: u64 = 0xdead_beef_f00d_cafe;

    /// Elements drawn from a few values so that both sources share some and miss others.
    fn sparse(seed: u64, size: usize) -> Vector {
        let mut rng = Rng(seed);
        build(64, size, |_| rng.next() % 24)
    }

    fn reference(first: &Vector, second: &Vector, size: usize, count: usize) -> (u64, u64) {
        let (mut first_bits, mut second_bits) = (0, 0);
        for j in 0..count {
            for k in 0..count {
                if element(first, size, j) == element(second, size, k) {
                    first_bits |= 1 << j;
                    second_bits |= 1 << k;
                }
            }
        }
        (first_bits, second_bits)
    }

    fn setup(h: &mut Harness, first: Vector, second: Vector) {
        h.cpu.set_zmm(1, first);
        h.cpu.set_zmm(2, second);
        h.write(DATA, &bytes_of(&second));
        h.cpu.opmask = [OPMASK_JUNK; 8];
    }

    #[test]
    fn each_mask_of_the_pair_marks_the_elements_found_in_the_other_source() {
        for (code, size, bytes, even) in [
            (VP2INTERSECTD_Z, 4usize, 64usize, 2usize),
            (VP2INTERSECTQ_Z, 8, 64, 2),
            (VP2INTERSECTQ_X, 8, 16, 6),
            (VP2INTERSECTD_Z_MEMORY, 4, 64, 0),
            (VP2INTERSECTD_Z_ODD, 4, 64, 2),
        ] {
            for seed in [0xb1, 0xb2, 0xb3] {
                let (first, second) = (sparse(seed, size), sparse(seed ^ 0xff, size));
                let h = run(code, |h| setup(h, first, second));
                let want = reference(&first, &second, size, bytes / size);
                assert_eq!(
                    (h.cpu.opmask[even], h.cpu.opmask[even + 1]),
                    want,
                    "{code:02x?} seed {seed:#x}"
                );
                for (k, value) in h.cpu.opmask.iter().enumerate() {
                    if k != even && k != even + 1 {
                        assert_eq!(*value, OPMASK_JUNK, "{code:02x?}: k{k} is not in the pair");
                    }
                }
            }
        }
    }

    #[test]
    fn a_broadcast_second_source_compares_every_element_with_one_value() {
        let first = sparse(0xc1, 4);
        let mut second = [0u128; 4];
        set_element(&mut second, 4, 0, element(&first, 4, 5));
        let h = run(VP2INTERSECTD_Y_BROADCAST, |h| setup(h, first, second));
        let broadcast = build(32, 4, |_| element(&first, 4, 5));
        let (first_bits, second_bits) = reference(&first, &broadcast, 4, 8);
        assert_eq!(
            (h.cpu.opmask[4], h.cpu.opmask[5]),
            (first_bits, second_bits)
        );
        assert_eq!(
            second_bits, 0xff,
            "every copy of the broadcast value matches"
        );
    }

    #[test]
    fn write_masking_and_zeroing_are_invalid() {
        for code in [VP2INTERSECTD_Z_MASKED, VP2INTERSECTD_Z_ZEROING] {
            let stop = stop_of(code, |h| setup(h, sparse(1, 4), sparse(2, 4)));
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: 6,
                    error_code: None,
                    software: false
                },
                "{code:02x?}"
            );
        }
    }

    #[test]
    fn vp2intersect_runs_identically_under_the_jit() {
        for code in [
            VP2INTERSECTD_Z,
            VP2INTERSECTQ_X,
            VP2INTERSECTD_Y_BROADCAST,
            VP2INTERSECTD_Z_MASKED,
        ] {
            crate::testkit::assert_jit_matches_interpreter(code, |h| {
                h.cpu.cr4 = (h.cpu.cr4 & !(1 << 9)) | CR4_OSXSAVE;
                h.cpu.xcr0 = XCR0_SUPPORTED;
                h.cpu.gpr[RSI] = DATA;
                setup(h, sparse(3, 4), sparse(4, 4));
            });
        }
    }
}
