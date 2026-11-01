//! The AVX-512 extensions built on integer vectors: conflict detection (CD), IFMA, VNNI, BITALG,
//! VBMI, VBMI2 and GFNI. The floating-point instructions are in `avx512_fp.rs`.

use super::*;
use crate::cpu::sse::ext::crypto;

const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;
/// IFMA operates on the low 52 bits of each quadword.
const IFMA_BITS: u32 = 52;
const IFMA_MASK: u64 = (1 << IFMA_BITS) - 1;
/// vdbpsadbw adds up the differences of this many bytes per 16-bit result.
const DBPSADBW_WINDOW: usize = 4;
/// `0F 38 50..=53`: vpdpbusd, vpdpbusds, vpdpwssd, vpdpwssds, in both the EVEX (AVX512_VNNI) and
/// VEX (AVX-VNNI, `avx_special.rs`) encodings. Bit 0 of the opcode selects saturation, bit 1
/// word instead of byte products.
pub(in crate::cpu) const VNNI_FIRST_OPCODE: u8 = 0x50;
pub(in crate::cpu) const VNNI_LAST_OPCODE: u8 = 0x53;
const VNNI_SATURATING: u8 = 1;
const VNNI_WORDS: u8 = 2;
/// `EVEX.F2.0F38 68`: vp2intersectd (W0) and vp2intersectq (W1).
const OPCODE_VP2INTERSECT: u8 = 0x68;
/// vp2intersect writes the opmask pair k(reg & 6), k(reg & 6 | 1).
const MASK_PAIR_EVEN: u8 = 0b110;

/// One dword of a VNNI dot product: `accumulator` plus the four products of `first`'s unsigned
/// bytes with `second`'s signed bytes, or the two products of their signed words. The saturating
/// forms clamp the sum to i32; the others wrap. `first` is the VEX.vvvv/EVEX.vvvv source and
/// `second` the r/m one; swapping them changes which bytes are unsigned.
pub(in crate::cpu) fn vnni_dword(opcode: u8, accumulator: u32, first: u32, second: u32) -> u32 {
    let products: i64 = if opcode & VNNI_WORDS != 0 {
        (0..2)
            .map(|j| (first >> (16 * j)) as i16 as i64 * (second >> (16 * j)) as i16 as i64)
            .sum()
    } else {
        (0..4)
            .map(|j| (first >> (8 * j)) as u8 as i64 * (second >> (8 * j)) as i8 as i64)
            .sum()
    };
    let total = accumulator as i32 as i64 + products;
    if opcode & VNNI_SATURATING != 0 {
        total.clamp(i32::MIN as i64, i32::MAX as i64) as u32
    } else {
        total as u32
    }
}

impl Cpu {
    pub(super) fn exec_evex_misc(
        &mut self,
        i: &Insn,
        next: u64,
        map: u16,
        opcode: u8,
        prefix: Prefix,
        wide: bool,
    ) -> ExecResult<bool> {
        use Prefix::*;
        if (map, opcode, prefix) == (OPCODE_MAP_0F38, OPCODE_VP2INTERSECT, F2) {
            self.intersect_into_mask_pair(i, next, wide)?;
            return Ok(true);
        }
        if prefix != P66 {
            return self.exec_evex_fp(i, next, map, opcode, prefix, wide);
        }
        let size = if wide { 8 } else { 4 };
        match (map, opcode) {
            (OPCODE_MAP_0F38, 0xc4) => {
                // vpconflictd/q: for each element, the earlier elements equal to it as a bit set
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(size).unary().crossing(),
                    |_, _, _, source, bytes| {
                        Ok(build(bytes, size, |k| {
                            let value = element(source, size, k);
                            (0..k)
                                .filter(|&j| element(source, size, j) == value)
                                .fold(0, |bits, j| bits | 1 << j)
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F38, 0xb4 | 0xb5) if wide => {
                let high = opcode == 0xb5;
                self.exec_vector_op(i, next, OpShape::elementwise(8), |_, old, a, b, bytes| {
                    Ok(build(bytes, 8, |k| {
                        let product = (element(a, 8, k) & IFMA_MASK) as u128
                            * (element(b, 8, k) & IFMA_MASK) as u128;
                        let part =
                            if high { product >> IFMA_BITS } else { product } as u64 & IFMA_MASK;
                        element(old, 8, k).wrapping_add(part)
                    }))
                })?;
            }
            // vpdpbusd(s) and vpdpwssd(s): dot products accumulated into the destination
            (OPCODE_MAP_0F38, VNNI_FIRST_OPCODE..=VNNI_LAST_OPCODE) if !wide => {
                self.exec_vector_op(i, next, OpShape::elementwise(4), |_, old, a, b, bytes| {
                    Ok(build(bytes, 4, |k| {
                        vnni_dword(
                            opcode,
                            element(old, 4, k) as u32,
                            element(a, 4, k) as u32,
                            element(b, 4, k) as u32,
                        ) as u64
                    }))
                })?;
            }
            // vdbpsadbw: the second source is first shuffled by dwords within each 128-bit lane
            // (imm8 holds four 2-bit selectors), then every quadword yields four 16-bit sums of
            // absolute differences of 4-byte windows that slide by one byte (SDM VDBPSADBW)
            (OPCODE_MAP_0F3A, 0x42) if !wide => {
                let selectors = i.imm as u8;
                let mut shape = OpShape::elementwise(2).crossing();
                shape.broadcast = false;
                self.exec_vector_op(i, next, shape, |_, _, first, second, bytes| {
                    let shuffled = build(bytes, 4, |k| {
                        let from = selectors >> (2 * (k % 4)) & 3;
                        element(second, 4, k / 4 * 4 + from as usize)
                    });
                    let sad = |a_start: usize, b_start: usize| -> u64 {
                        (0..DBPSADBW_WINDOW)
                            .map(|j| {
                                element(first, 1, a_start + j).abs_diff(element(
                                    &shuffled,
                                    1,
                                    b_start + j,
                                ))
                            })
                            .sum()
                    };
                    Ok(build(bytes, 2, |k| {
                        let base = k / 4 * 8;
                        match k % 4 {
                            0 => sad(base, base),
                            1 => sad(base, base + 1),
                            2 => sad(base + DBPSADBW_WINDOW, base + 2),
                            _ => sad(base + DBPSADBW_WINDOW, base + 3),
                        }
                    }))
                })?;
            }
            (OPCODE_MAP_0F38, 0x8f) if !wide => self.shuffle_bits_to_mask(i, next)?,
            (OPCODE_MAP_0F38, 0x83) if wide => {
                let mut shape = OpShape::elementwise(1).crossing();
                shape.rm_element = Some(8);
                self.exec_vector_op(i, next, shape, |_, _, control, data, bytes| {
                    Ok(build(bytes, 1, |k| {
                        let quadword = element(data, 8, k / 8);
                        let start = element(control, 1, k) as u32 & 63;
                        quadword.rotate_right(start) & 0xff
                    }))
                })?;
            }
            // vpshldv/vpshrdv: concatenate-and-shift by per-element counts (VBMI2)
            (OPCODE_MAP_0F38, 0x70..=0x73) => {
                let element_bytes = if opcode & 1 == 0 { 2 } else { size };
                if opcode & 1 == 0 && !wide {
                    return Err(Exception::invalid_opcode());
                }
                let left = opcode < 0x72;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element_bytes),
                    |_, old, counts, other, bytes| {
                        Ok(build(bytes, element_bytes, |k| {
                            let bits = element_bytes as u32 * 8;
                            let n = element(counts, element_bytes, k) as u32 % bits;
                            funnel(
                                left,
                                element(old, element_bytes, k),
                                element(other, element_bytes, k),
                                n,
                                bits,
                            )
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F3A, 0x70..=0x73) => {
                let element_bytes = if opcode & 1 == 0 { 2 } else { size };
                if opcode & 1 == 0 && !wide {
                    return Err(Exception::invalid_opcode());
                }
                let left = opcode < 0x72;
                let imm = i.imm as u32;
                self.exec_vector_op(
                    i,
                    next,
                    OpShape::elementwise(element_bytes),
                    |_, _, a, b, bytes| {
                        Ok(build(bytes, element_bytes, |k| {
                            let bits = element_bytes as u32 * 8;
                            funnel(
                                left,
                                element(a, element_bytes, k),
                                element(b, element_bytes, k),
                                imm % bits,
                                bits,
                            )
                        }))
                    },
                )?;
            }
            (OPCODE_MAP_0F38, 0xcf) if !wide => {
                let mut shape = OpShape::elementwise(1);
                shape.broadcast = false;
                self.exec_vector_op(i, next, shape, |_, _, a, b, bytes| {
                    Ok(build(bytes, 1, |k| {
                        crypto::gf_multiply(element(a, 1, k) as u8, element(b, 1, k) as u8) as u64
                    }))
                })?;
            }
            (OPCODE_MAP_0F3A, 0xce | 0xcf) if wide => {
                let inverse = opcode == 0xcf;
                let immediate = i.imm as u8;
                let mut shape = OpShape::elementwise(1);
                shape.rm_element = Some(8);
                shape.suppress = false;
                self.exec_vector_op(i, next, shape, |_, _, data, matrix, bytes| {
                    Ok(build(bytes, 1, |k| {
                        let x = element(data, 1, k) as u8;
                        let x = if inverse { crypto::gf_inverse(x) } else { x };
                        crypto::gf_affine_byte(element(matrix, 8, k / 8), x, immediate) as u64
                    }))
                })?;
            }
            _ => return self.exec_evex_fp(i, next, map, opcode, prefix, wide),
        }
        Ok(true)
    }

    /// vpshufbitqmb: bit `k` of the result is bit `index[k] & 63` of the quadword of vvvv that byte
    /// `k` belongs to.
    fn shuffle_bits_to_mask(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let mask = self.write_mask(i);
        let shape = RmShape {
            suppress: (i.evex_mask != 0).then_some(mask),
            broadcast: false,
            ..RmShape::full(bytes, 1)
        };
        let indices = self.fetch_rm(i, next, shape)?;
        let data = self.zmm(i.vex_v as usize);
        let mut bits = 0u64;
        for k in 0..bytes {
            let bit = element(&data, 8, k / 8) >> (element(&indices, 1, k) & 63) & 1;
            if mask >> k & 1 == 1 {
                bits |= bit << k;
            }
        }
        self.opmask[(i.reg & 7) as usize] = bits;
        Ok(())
    }

    /// vp2intersectd/q (AVX512_VP2INTERSECT): compares every element of the first source (vvvv)
    /// with every element of the second (r/m, broadcast allowed). Bit j of the even mask register
    /// of the pair ModRM.reg names (bit 0 of the field is ignored) says element j of the first
    /// source equals some element of the second; bit k of the odd one, the same for element k of
    /// the second. No write mask and no zeroing: EVEX.aaa or EVEX.z set is #UD.
    fn intersect_into_mask_pair(&mut self, i: &Insn, next: u64, wide: bool) -> ExecResult<()> {
        if i.evex_mask != 0 || i.evex_zeroing {
            return Err(Exception::invalid_opcode());
        }
        let size = if wide { 8 } else { 4 };
        let bytes = self.vector_bytes(i, Rounding::None)?;
        let second = self.fetch_rm(i, next, RmShape::full(bytes, size))?;
        let first = self.zmm(i.vex_v as usize);
        let count = bytes / size;
        let (mut first_matches, mut second_matches) = (0u64, 0u64);
        for j in 0..count {
            for k in 0..count {
                if element(&first, size, j) == element(&second, size, k) {
                    first_matches |= 1 << j;
                    second_matches |= 1 << k;
                }
            }
        }
        let even = (i.reg & MASK_PAIR_EVEN) as usize;
        self.opmask[even] = first_matches;
        self.opmask[even + 1] = second_matches;
        Ok(())
    }
}

/// One element of a double-width shift: `left` is vpshld (the high half of (a:b) << n), the
/// other vpshrd (the low half of (b:a) >> n).
fn funnel(left: bool, a: u64, b: u64, n: u32, bits: u32) -> u64 {
    if n == 0 {
        return a;
    }
    let (a, b) = (
        a & size_mask(bits as usize / 8),
        b & size_mask(bits as usize / 8),
    );
    let value = if left {
        a << n | b >> (bits - n)
    } else {
        a >> n | b << (bits - n)
    };
    value & size_mask(bits as usize / 8)
}
