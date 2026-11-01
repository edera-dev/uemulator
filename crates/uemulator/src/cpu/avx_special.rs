//! VEX instructions that are not a plain lane-wise extension of an SSE instruction: mask moves,
//! broadcasts, permutes, 128-bit inserts/extracts, widening and narrowing conversions, variable
//! shifts and blends. They share operand plumbing with `avx.rs`; the arithmetic that has a legacy
//! counterpart (conversions, shifts) is delegated to the legacy kernel on explicit registers.

use super::decode::{EVEX_OP_FLAG, Insn, VEX_OP_FLAG};
use super::exception::Exception;
use super::sse::Prefix;
use super::sse::evex::misc::{VNNI_FIRST_OPCODE, VNNI_LAST_OPCODE, vnni_dword};
use super::sse::ext::{Op, decode_op};
use super::{Cpu, ExecResult};

const OPCODE_MAP_0F: u16 = 1;
const OPCODE_MAP_0F38: u16 = 2;
const OPCODE_MAP_0F3A: u16 = 3;
pub(super) const LANE_BYTES: usize = 16;
pub(super) const YMM_BYTES: usize = 32;
/// Scratch registers handed to the legacy kernels; their contents are saved and restored.
const SCRATCH_SOURCE: usize = 14;
const SCRATCH_DEST: usize = 15;

const SIGN_BITS_32: u128 = 0x8000_0000_8000_0000_8000_0000_8000_0000;
const SIGN_BITS_64: u128 = 0x8000_0000_0000_0000_8000_0000_0000_0000;

fn undefined<T>() -> ExecResult<T> {
    Err(Exception::invalid_opcode())
}

/// Bit `k` of the result is the top bit of lane `k` (`width` bits wide) of `v`.
fn sign_mask(v: u128, width: u32) -> u64 {
    let mut mask = 0;
    for lane in 0..128 / width {
        mask |= (((v >> (lane * width + width - 1)) & 1) as u64) << lane;
    }
    mask
}

fn lane_u32(v: u128, k: usize) -> u32 {
    (v >> (32 * k)) as u32
}

fn lane_u64(v: u128, k: usize) -> u64 {
    (v >> (64 * k)) as u64
}

fn dwords(v: [u32; 4]) -> u128 {
    v.iter()
        .enumerate()
        .fold(0, |acc, (k, &d)| acc | (d as u128) << (32 * k))
}

fn qwords(v: [u64; 2]) -> u128 {
    v[0] as u128 | (v[1] as u128) << 64
}

/// Broadcast of the low `bytes` bytes of `v` to fill 128 bits.
fn replicate(v: u128, bytes: usize) -> u128 {
    let bits = bytes * 8;
    let element = v & ((1u128 << bits) - 1);
    (0..128 / bits).fold(0, |acc, k| acc | element << (k * bits))
}

impl Cpu {
    pub(super) fn vex_address(&self, i: &Insn, next: u64) -> ExecResult<u64> {
        self.linear_address(i, self.effective_address(i, next))
    }

    /// `bytes` of memory at `address` as a little-endian integer (at most 16, no alignment rule).
    pub(super) fn read_partial(&mut self, address: u64, bytes: usize) -> ExecResult<u128> {
        match bytes {
            LANE_BYTES => self.read128(address),
            n => Ok(self.read(address, n as u8)? as u128),
        }
    }

    /// The r/m operand as (low lane, high lane): a register, or 16/32 bytes of memory.
    pub(super) fn vector_operand(
        &mut self,
        i: &Insn,
        next: u64,
        bytes: usize,
    ) -> ExecResult<(u128, u128)> {
        if i.md == 3 {
            return Ok((self.xmm[i.rm as usize], self.ymm_hi[i.rm as usize]));
        }
        let address = self.vex_address(i, next)?;
        let low = self.read_partial(address, bytes.min(LANE_BYTES))?;
        let high = if bytes == YMM_BYTES {
            self.read128(address + LANE_BYTES as u64)?
        } else {
            0
        };
        Ok((low, high))
    }

    /// Writes a destination register; without `wide` the upper half is zeroed (VEX.128 rule).
    pub(super) fn set_vector(&mut self, register: usize, low: u128, high: u128, wide: bool) {
        self.xmm[register] = low;
        self.ymm_hi[register] = if wide { high } else { 0 };
        self.clear_zmm_upper(register);
    }

    /// Runs the legacy SSE kernel for `i` with explicit register operands and returns the new
    /// destination value. `destination` is the kernel's old-destination operand, `source` its
    /// r/m operand (always as a register here; callers read memory themselves).
    pub(super) fn run_legacy_registers(
        &mut self,
        i: &Insn,
        destination: u128,
        source: u128,
    ) -> ExecResult<u128> {
        let saved = (self.xmm[SCRATCH_SOURCE], self.xmm[SCRATCH_DEST]);
        self.xmm[SCRATCH_SOURCE] = source;
        self.xmm[SCRATCH_DEST] = destination;
        let mut legacy = *i;
        legacy.vex = false;
        legacy.op = i.op & !(VEX_OP_FLAG | EVEX_OP_FLAG);
        legacy.md = 3;
        legacy.reg = SCRATCH_DEST as u8;
        legacy.rm = SCRATCH_SOURCE as u8;
        self.vex_active = true;
        let outcome = self.sse(&legacy, 0);
        self.vex_active = false;
        let result = self.xmm[SCRATCH_DEST];
        (self.xmm[SCRATCH_SOURCE], self.xmm[SCRATCH_DEST]) = saved;
        match outcome {
            Ok(true) => Ok(result),
            Ok(false) => Err(self.unimplemented_opcode(i)),
            Err(error) => Err(error),
        }
    }

    /// Like `run_legacy_registers` for the immediate shift groups (`0F 71..73`), where ModRM.reg
    /// is the operation and the single register operand is both source and destination.
    fn run_legacy_shift_immediate(&mut self, i: &Insn, value: u128) -> ExecResult<u128> {
        let saved = self.xmm[SCRATCH_DEST];
        self.xmm[SCRATCH_DEST] = value;
        let mut legacy = *i;
        legacy.vex = false;
        legacy.op = i.op & !VEX_OP_FLAG;
        legacy.md = 3;
        legacy.rm = SCRATCH_DEST as u8;
        self.vex_active = true;
        let outcome = self.sse(&legacy, 0);
        self.vex_active = false;
        let result = self.xmm[SCRATCH_DEST];
        self.xmm[SCRATCH_DEST] = saved;
        match outcome {
            Ok(true) => Ok(result),
            Ok(false) => Err(self.unimplemented_opcode(i)),
            Err(error) => Err(error),
        }
    }

    /// vpgather*/vgather*: elements are loaded one at a time under the mask, and the mask bit of each
    /// completed element is cleared, so a fault leaves a restartable partial state: the registers
    /// are written back before the fault is reported, as on hardware.
    fn exec_gather(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let opcode = (i.op & 0xff) as u8;
        let wide_elements = i.opsize == 8;
        let element_bytes = if wide_elements { 8 } else { 4 };
        // 90/92: dword indices; 91/93: qword indices
        let index_bytes = if opcode & 1 == 0 { 4 } else { 8 };
        let (reg, mask_reg, index_reg) = (i.reg as usize, i.vex_v as usize, i.index as usize);
        // the destination, mask and index registers must all differ
        if i.md == 3
            || i.index == super::decode::NO_REG
            || reg == mask_reg
            || reg == index_reg
            || mask_reg == index_reg
        {
            return undefined();
        }
        let vector_bytes = if i.vex_l { YMM_BYTES } else { LANE_BYTES };
        let count = vector_bytes / element_bytes.max(index_bytes);
        let destination_bytes = count * element_bytes;
        let base = if i.rip_rel {
            next
        } else if i.base != super::decode::NO_REG {
            self.gpr[i.base as usize]
        } else {
            0
        };
        let base = base.wrapping_add(i.disp as u64);
        let mut dest = [self.xmm[reg], self.ymm_hi[reg]];
        let mut mask = [self.xmm[mask_reg], self.ymm_hi[mask_reg]];
        let indices = [self.xmm[index_reg], self.ymm_hi[index_reg]];
        // The mask is normalised first: each element becomes all ones or all zeros by its sign bit.
        for k in 0..count {
            let (half, shift) = (
                k * element_bytes / LANE_BYTES,
                (k * element_bytes % LANE_BYTES) * 8,
            );
            let element_mask = ((1u128 << (element_bytes * 8)) - 1) << shift;
            let set = mask[half] >> (shift + element_bytes * 8 - 1) & 1 != 0;
            mask[half] = (mask[half] & !element_mask) | if set { element_mask } else { 0 };
        }
        let mut failure = None;
        for k in 0..count {
            let (half, shift) = (
                k * element_bytes / LANE_BYTES,
                (k * element_bytes % LANE_BYTES) * 8,
            );
            if mask[half] >> shift & 1 == 0 {
                continue;
            }
            let index_half = k * index_bytes / LANE_BYTES;
            let index_shift = (k * index_bytes % LANE_BYTES) * 8;
            let raw = (indices[index_half] >> index_shift) as u64;
            let index = if index_bytes == 4 {
                raw as u32 as i32 as i64
            } else {
                raw as i64
            };
            let mut address = base.wrapping_add((index << i.scale).wrapping_mul(1) as u64);
            if i.addr32 {
                address &= 0xffff_ffff;
            }
            let linear = self.linear_address(i, address)?;
            match self.read(linear, element_bytes as u8) {
                Ok(value) => {
                    let element_mask = ((1u128 << (element_bytes * 8)) - 1) << shift;
                    dest[half] = (dest[half] & !element_mask) | ((value as u128) << shift);
                    mask[half] &= !element_mask;
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if failure.is_none() {
            // completed: everything above the gathered elements is zeroed, in both registers
            mask = [0; 2];
            if destination_bytes <= LANE_BYTES {
                dest[1] = 0;
                if destination_bytes < LANE_BYTES {
                    dest[0] &= (1u128 << (destination_bytes * 8)) - 1;
                }
            }
        }
        (self.xmm[reg], self.ymm_hi[reg]) = (dest[0], dest[1]);
        (self.xmm[mask_reg], self.ymm_hi[mask_reg]) = (mask[0], mask[1]);
        self.clear_zmm_upper(reg);
        self.clear_zmm_upper(mask_reg);
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `Ok(false)` means the opcode is not implemented (the caller raises the logged #UD).
    pub(super) fn exec_vex_special(
        &mut self,
        i: &Insn,
        next: u64,
        prefix: Prefix,
    ) -> ExecResult<bool> {
        use Prefix::*;
        let op = i.op & !VEX_OP_FLAG;
        let map = op >> 8;
        let opcode = (op & 0xff) as u8;
        let l = i.vex_l;
        let wide_w = i.opsize == 8;
        let reg = i.reg as usize;
        let rm = i.rm as usize;
        let vvvv = i.vex_v as usize;
        let operand_bytes = if l { YMM_BYTES } else { LANE_BYTES };
        let immediate = i.imm as u8;
        macro_rules! gate {
            () => {
                self.vex_gate()?
            };
        }
        macro_rules! require {
            ($condition:expr) => {
                if !$condition {
                    return undefined();
                }
            };
        }
        match (map, opcode, prefix) {
            // movmskps/movmskpd: sign bits of each lane into a general register
            (OPCODE_MAP_0F, 0x50, Np | P66) => {
                gate!();
                require!(i.md == 3 && i.vex_v == 0);
                let width = if prefix == Np { 32 } else { 64 };
                let mut mask = sign_mask(self.xmm[rm], width);
                if l {
                    mask |= sign_mask(self.ymm_hi[rm], width) << (128 / width);
                }
                self.write_gpr(i.reg, 4, true, mask);
            }
            (OPCODE_MAP_0F, 0xd7, P66) => {
                gate!();
                require!(i.md == 3 && i.vex_v == 0);
                let mut mask = sign_mask(self.xmm[rm], 8);
                if l {
                    mask |= sign_mask(self.ymm_hi[rm], 8) << 16;
                }
                self.write_gpr(i.reg, 4, true, mask);
            }
            // cvtps2pd / cvtdq2pd widen; cvtpd2ps / cvt(t)pd2dq narrow. Each half goes through
            // the legacy kernel, which owns rounding, exceptions and MXCSR.
            (OPCODE_MAP_0F, 0x5a, Np) | (OPCODE_MAP_0F, 0xe6, F3) => {
                gate!();
                require!(i.vex_v == 0);
                let (source, _) = self.vector_operand(i, next, if l { 16 } else { 8 })?;
                let low = self.run_legacy_registers(i, 0, source)?;
                let high = if l {
                    self.run_legacy_registers(i, 0, source >> 64)?
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            (OPCODE_MAP_0F, 0x5a, P66) | (OPCODE_MAP_0F, 0xe6, P66 | F2) => {
                gate!();
                require!(i.vex_v == 0);
                let (low, high) = self.vector_operand(i, next, operand_bytes)?;
                let mut result = self.run_legacy_registers(i, 0, low)? & u64::MAX as u128;
                if l {
                    result |= (self.run_legacy_registers(i, 0, high)? & u64::MAX as u128) << 64;
                }
                self.set_vector(reg, result, 0, false);
            }
            // psllw/pslld/psllq/psrlw/psrld/psrlq/psraw/psrad by the low quadword of an xmm/m128
            (OPCODE_MAP_0F, 0xd1..=0xd3 | 0xe1 | 0xe2 | 0xf1..=0xf3, P66) => {
                gate!();
                let count = if i.md == 3 {
                    self.xmm[rm]
                } else {
                    let address = self.vex_address(i, next)?;
                    self.read128(address)?
                };
                let low = self.run_legacy_registers(i, self.xmm[vvvv], count)?;
                let high = if l {
                    self.run_legacy_registers(i, self.ymm_hi[vvvv], count)?
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            // shifts by immediate: destination is vvvv, source is the r/m register
            (OPCODE_MAP_0F, 0x71..=0x73, P66) => {
                gate!();
                require!(i.md == 3);
                let low = self.run_legacy_shift_immediate(i, self.xmm[rm])?;
                let high = if l {
                    self.run_legacy_shift_immediate(i, self.ymm_hi[rm])?
                } else {
                    0
                };
                self.set_vector(vvvv, low, high, l);
            }
            (OPCODE_MAP_0F38, 0x17, P66) => {
                gate!();
                require!(i.vex_v == 0);
                let (src_low, src_high) = self.vector_operand(i, next, operand_bytes)?;
                let (dst_low, dst_high) = (self.xmm[reg], self.ymm_hi[reg]);
                let (and, and_not) = if l {
                    (
                        (dst_low & src_low) | (dst_high & src_high),
                        (!dst_low & src_low) | (!dst_high & src_high),
                    )
                } else {
                    (dst_low & src_low, !dst_low & src_low)
                };
                self.set_eflags_arith(
                    if and == 0 { super::flags::ZF } else { 0 }
                        | if and_not == 0 { super::flags::CF } else { 0 },
                );
            }
            // vtestps/vtestpd: ptest on the sign bits only
            (OPCODE_MAP_0F38, 0x0e | 0x0f, P66) => {
                gate!();
                require!(i.vex_v == 0);
                let signs = if opcode == 0x0e {
                    SIGN_BITS_32
                } else {
                    SIGN_BITS_64
                };
                let (src_low, src_high) = self.vector_operand(i, next, operand_bytes)?;
                let (dst_low, dst_high) = (self.xmm[reg], self.ymm_hi[reg]);
                let mut and = dst_low & src_low & signs;
                let mut and_not = !dst_low & src_low & signs;
                if l {
                    and |= dst_high & src_high & signs;
                    and_not |= !dst_high & src_high & signs;
                }
                self.set_eflags_arith(
                    if and == 0 { super::flags::ZF } else { 0 }
                        | if and_not == 0 { super::flags::CF } else { 0 },
                );
            }
            // vpermilps/vpermilpd with a variable selector (the r/m operand)
            (OPCODE_MAP_0F38, 0x0c | 0x0d, P66) => {
                gate!();
                let (sel_low, sel_high) = self.vector_operand(i, next, operand_bytes)?;
                let permute = |source: u128, selector: u128| {
                    if opcode == 0x0c {
                        dwords(core::array::from_fn(|k| {
                            lane_u32(source, (lane_u32(selector, k) & 3) as usize)
                        }))
                    } else {
                        qwords(core::array::from_fn(|k| {
                            lane_u64(source, ((lane_u64(selector, k) >> 1) & 1) as usize)
                        }))
                    }
                };
                let low = permute(self.xmm[vvvv], sel_low);
                let high = if l {
                    permute(self.ymm_hi[vvvv], sel_high)
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            (OPCODE_MAP_0F3A, 0x04 | 0x05, P66) => {
                gate!();
                require!(i.vex_v == 0);
                let (src_low, src_high) = self.vector_operand(i, next, operand_bytes)?;
                let permute = |source: u128, immediate: u8| {
                    if opcode == 0x04 {
                        dwords(core::array::from_fn(|k| {
                            lane_u32(source, ((immediate >> (2 * k)) & 3) as usize)
                        }))
                    } else {
                        qwords(core::array::from_fn(|k| {
                            lane_u64(source, ((immediate >> k) & 1) as usize)
                        }))
                    }
                };
                let low = permute(src_low, immediate);
                // vpermilpd uses imm bits 3:2 for the upper lane
                let high = if l {
                    permute(
                        src_high,
                        if opcode == 0x04 {
                            immediate
                        } else {
                            immediate >> 2
                        },
                    )
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            // vpermq / vpermpd: 256-bit only, W = 1
            (OPCODE_MAP_0F3A, 0x00 | 0x01, P66) => {
                gate!();
                require!(l && wide_w && i.vex_v == 0);
                let (src_low, src_high) = self.vector_operand(i, next, YMM_BYTES)?;
                let source = [
                    lane_u64(src_low, 0),
                    lane_u64(src_low, 1),
                    lane_u64(src_high, 0),
                    lane_u64(src_high, 1),
                ];
                let pick = |k: usize| source[((immediate >> (2 * k)) & 3) as usize];
                self.set_vector(
                    reg,
                    qwords([pick(0), pick(1)]),
                    qwords([pick(2), pick(3)]),
                    true,
                );
            }
            // vpermps / vpermd: indices in vvvv, data in r/m
            (OPCODE_MAP_0F38, 0x16 | 0x36, P66) => {
                gate!();
                require!(l && !wide_w);
                let (src_low, src_high) = self.vector_operand(i, next, YMM_BYTES)?;
                let source: [u32; 8] = core::array::from_fn(|k| {
                    if k < 4 {
                        lane_u32(src_low, k)
                    } else {
                        lane_u32(src_high, k - 4)
                    }
                });
                let (index_low, index_high) = (self.xmm[vvvv], self.ymm_hi[vvvv]);
                let pick = |k: usize| {
                    let index = if k < 4 {
                        lane_u32(index_low, k)
                    } else {
                        lane_u32(index_high, k - 4)
                    };
                    source[(index & 7) as usize]
                };
                self.set_vector(
                    reg,
                    dwords([pick(0), pick(1), pick(2), pick(3)]),
                    dwords([pick(4), pick(5), pick(6), pick(7)]),
                    true,
                );
            }
            // vperm2f128 / vperm2i128: each result half picks one of the four source halves
            (OPCODE_MAP_0F3A, 0x06 | 0x46, P66) => {
                gate!();
                require!(l && !wide_w);
                let (src2_low, src2_high) = self.vector_operand(i, next, YMM_BYTES)?;
                let halves = [self.xmm[vvvv], self.ymm_hi[vvvv], src2_low, src2_high];
                let select = |control: u8| {
                    if control & 0x8 != 0 {
                        0
                    } else {
                        halves[(control & 3) as usize]
                    }
                };
                self.set_vector(reg, select(immediate & 0xf), select(immediate >> 4), true);
            }
            // vinsertf128/vinserti128 and vextractf128/vextracti128
            (OPCODE_MAP_0F3A, 0x18 | 0x38, P66) => {
                gate!();
                require!(l && !wide_w);
                let (insert, _) = self.vector_operand(i, next, LANE_BYTES)?;
                let (mut low, mut high) = (self.xmm[vvvv], self.ymm_hi[vvvv]);
                if immediate & 1 == 0 {
                    low = insert;
                } else {
                    high = insert;
                }
                self.set_vector(reg, low, high, true);
            }
            (OPCODE_MAP_0F3A, 0x19 | 0x39, P66) => {
                gate!();
                require!(l && !wide_w && i.vex_v == 0);
                let half = if immediate & 1 == 0 {
                    self.xmm[reg]
                } else {
                    self.ymm_hi[reg]
                };
                if i.md == 3 {
                    self.set_vector(rm, half, 0, false);
                } else {
                    let address = self.vex_address(i, next)?;
                    self.write128(address, half)?;
                }
            }
            // broadcasts
            (OPCODE_MAP_0F38, 0x18 | 0x19 | 0x58 | 0x59 | 0x78 | 0x79, P66) => {
                gate!();
                require!(i.vex_v == 0 && !wide_w);
                // vbroadcastsd exists only as a 256-bit instruction
                require!(opcode != 0x19 || l);
                let bytes = match opcode {
                    0x18 | 0x58 => 4,
                    0x19 | 0x59 => 8,
                    0x78 => 1,
                    _ => 2,
                };
                let element = if i.md == 3 {
                    self.xmm[rm]
                } else {
                    let address = self.vex_address(i, next)?;
                    self.read_partial(address, bytes)?
                };
                let filled = replicate(element, bytes);
                self.set_vector(reg, filled, filled, l);
            }
            // vbroadcastf128/vbroadcasti128: memory only, 256-bit only
            (OPCODE_MAP_0F38, 0x1a | 0x5a, P66) => {
                gate!();
                require!(l && i.md != 3 && i.vex_v == 0 && !wide_w);
                let address = self.vex_address(i, next)?;
                let value = self.read128(address)?;
                self.set_vector(reg, value, value, true);
            }
            // pmovsx*/pmovzx*: 256-bit forms widen twice as many elements
            (OPCODE_MAP_0F38, 0x20..=0x25 | 0x30..=0x35, P66) => {
                gate!();
                require!(i.vex_v == 0);
                let Some(Op::Widen {
                    signed,
                    from_bits,
                    to_bits,
                }) = decode_op(map, opcode, P66)
                else {
                    return Ok(false);
                };
                let ratio = (to_bits / from_bits) as usize;
                let bytes = operand_bytes / ratio;
                let (source, _) = self.vector_operand(i, next, bytes)?;
                let widen = |v: u128| {
                    let mut result = 0u128;
                    for lane in 0..(128 / to_bits) {
                        let raw = (v >> (lane * from_bits)) & ((1u128 << from_bits) - 1);
                        let value = if signed && raw >> (from_bits - 1) & 1 != 0 {
                            raw | !((1u128 << from_bits) - 1)
                        } else {
                            raw
                        };
                        result |= (value & ((1u128 << to_bits) - 1)) << (lane * to_bits);
                    }
                    result
                };
                let low = widen(source);
                // the upper result lane takes the next 128/ratio bits of the source
                let high = if l { widen(source >> (128 / ratio)) } else { 0 };
                self.set_vector(reg, low, high, l);
            }
            // vblendvps/vblendvpd/vpblendvb: the mask register is in imm8[7:4]
            (OPCODE_MAP_0F3A, 0x4a..=0x4c, P66) => {
                gate!();
                require!(!wide_w);
                let legacy = match opcode {
                    0x4a => 0x14,
                    0x4b => 0x15,
                    _ => 0x10,
                };
                let Some(Op::WithMask(blend)) = decode_op(OPCODE_MAP_0F38, legacy, P66) else {
                    return Ok(false);
                };
                let mask_register = (immediate >> 4) as usize;
                let (src_low, src_high) = self.vector_operand(i, next, operand_bytes)?;
                let low = blend(self.xmm[vvvv], src_low, self.xmm[mask_register]);
                let high = if l {
                    blend(self.ymm_hi[vvvv], src_high, self.ymm_hi[mask_register])
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            // vpblendd: one immediate bit per dword
            (OPCODE_MAP_0F3A, 0x02, P66) => {
                gate!();
                require!(!wide_w);
                let (src_low, src_high) = self.vector_operand(i, next, operand_bytes)?;
                let blend = |first: u128, second: u128, bits: u8| {
                    dwords(core::array::from_fn(|k| {
                        if bits >> k & 1 != 0 {
                            lane_u32(second, k)
                        } else {
                            lane_u32(first, k)
                        }
                    }))
                };
                let low = blend(self.xmm[vvvv], src_low, immediate & 0xf);
                let high = if l {
                    blend(self.ymm_hi[vvvv], src_high, immediate >> 4)
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            // vpsrlv/vpsllv (W picks dword or qword elements) and vpsravd (dwords only)
            (OPCODE_MAP_0F38, 0x45..=0x47, P66) => {
                gate!();
                require!(opcode != 0x46 || !wide_w);
                let (count_low, count_high) = self.vector_operand(i, next, operand_bytes)?;
                let shift = |value: u128, counts: u128| {
                    if wide_w {
                        qwords(core::array::from_fn(|k| {
                            let (v, c) = (lane_u64(value, k), lane_u64(counts, k));
                            if c >= 64 {
                                0
                            } else if opcode == 0x45 {
                                v >> c
                            } else {
                                v << c
                            }
                        }))
                    } else {
                        dwords(core::array::from_fn(|k| {
                            let (v, c) = (lane_u32(value, k), lane_u32(counts, k));
                            match opcode {
                                0x45 => {
                                    if c >= 32 {
                                        0
                                    } else {
                                        v >> c
                                    }
                                }
                                0x47 => {
                                    if c >= 32 {
                                        0
                                    } else {
                                        v << c
                                    }
                                }
                                _ => ((v as i32) >> c.min(31)) as u32,
                            }
                        }))
                    }
                };
                let low = shift(self.xmm[vvvv], count_low);
                let high = if l {
                    shift(self.ymm_hi[vvvv], count_high)
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            // vldmxcsr / vstmxcsr (VEX.0F AE /2, /3): memory operand only, 128-bit, vvvv unused
            (OPCODE_MAP_0F, 0xae, Np) => {
                gate!();
                require!(i.md != 3 && !l && i.vex_v == 0 && matches!(i.reg & 7, 2 | 3));
                let address = self.vex_address(i, next)?;
                if i.reg & 7 == 2 {
                    let value = self.read(address, 4)? as u32;
                    if value & !super::fpu::MXCSR_VALID_BITS != 0 {
                        return Err(Exception::general_protection(0));
                    }
                    self.mxcsr = value;
                } else {
                    self.write(address, 4, self.mxcsr as u64)?;
                }
            }
            // vmaskmovps/vmaskmovpd and vpmaskmovd/vpmaskmovq: per-element masked loads and stores
            // that never touch (or fault on) a masked-out element
            (OPCODE_MAP_0F38, 0x2c..=0x2f | 0x8c | 0x8e, P66) => {
                gate!();
                require!(i.md != 3);
                let element_bytes = match opcode {
                    0x2c | 0x2e => 4,
                    0x2d | 0x2f => 8,
                    _ => {
                        if wide_w {
                            8
                        } else {
                            4
                        }
                    }
                };
                // the vmaskmov forms are W0 only
                require!(!wide_w || matches!(opcode, 0x8c | 0x8e));
                let address = self.vex_address(i, next)?;
                let (mask_low, mask_high) = (self.xmm[vvvv], self.ymm_hi[vvvv]);
                let count = operand_bytes / element_bytes;
                let active = |k: usize| {
                    let source = if k * element_bytes < LANE_BYTES {
                        mask_low
                    } else {
                        mask_high
                    };
                    let shift = (k * element_bytes % LANE_BYTES) * 8 + element_bytes * 8 - 1;
                    source >> shift & 1 != 0
                };
                if matches!(opcode, 0x2c | 0x2d | 0x8c) {
                    let mut halves = [0u128; 2];
                    for k in 0..count {
                        if active(k) {
                            let value = self
                                .read(address + (k * element_bytes) as u64, element_bytes as u8)?;
                            halves[k * element_bytes / LANE_BYTES] |=
                                (value as u128) << ((k * element_bytes % LANE_BYTES) * 8);
                        }
                    }
                    self.set_vector(reg, halves[0], halves[1], l);
                } else {
                    for k in 0..count {
                        if active(k) {
                            let source = if k * element_bytes < LANE_BYTES {
                                self.xmm[reg]
                            } else {
                                self.ymm_hi[reg]
                            };
                            let value = (source >> ((k * element_bytes % LANE_BYTES) * 8)) as u64;
                            self.write(
                                address + (k * element_bytes) as u64,
                                element_bytes as u8,
                                value,
                            )?;
                        }
                    }
                }
            }
            // gathers (VSIB addressing)
            (OPCODE_MAP_0F38, 0x90..=0x93, P66) => {
                gate!();
                self.exec_gather(i, next)?;
            }
            // FMA3: 0F38 96-9F, A6-AF, B6-BF
            (OPCODE_MAP_0F38, 0x96..=0x9f | 0xa6..=0xaf | 0xb6..=0xbf, P66) => {
                self.exec_fma(i, next)?;
            }
            (OPCODE_MAP_0F38, 0x13, P66) | (OPCODE_MAP_0F3A, 0x1d, P66) => {
                self.exec_f16c(i, next)?;
            }
            // AVX-VNNI: the VEX forms of vpdpbusd(s)/vpdpwssd(s), W0 only, the destination as
            // accumulator; the arithmetic is shared with the EVEX (AVX512_VNNI) forms
            (OPCODE_MAP_0F38, VNNI_FIRST_OPCODE..=VNNI_LAST_OPCODE, P66) => {
                gate!();
                require!(!wide_w);
                let (second_low, second_high) = self.vector_operand(i, next, operand_bytes)?;
                let accumulate = |old: u128, first: u128, second: u128| {
                    dwords(core::array::from_fn(|k| {
                        vnni_dword(
                            opcode,
                            lane_u32(old, k),
                            lane_u32(first, k),
                            lane_u32(second, k),
                        )
                    }))
                };
                let low = accumulate(self.xmm[reg], self.xmm[vvvv], second_low);
                let high = if l {
                    accumulate(self.ymm_hi[reg], self.ymm_hi[vvvv], second_high)
                } else {
                    0
                };
                self.set_vector(reg, low, high, l);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "avx_special_tests.rs"]
mod avx_special_tests;
