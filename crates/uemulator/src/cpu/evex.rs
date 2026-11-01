//! AVX-512 (EVEX-encoded) instructions: register file, operand fetch, write masking and the
//! instructions that are an SSE/AVX instruction on four 128-bit lanes. See
//! `docs/plans/14-avx-and-virtualization.md` for what is and is not implemented.
//!
//! Register file: `xmm`/`ymm_hi` keep ZMM0-15 bits 255:0 as before, `zmm_hi` holds bits 511:256 and
//! `zmm_extra` all of ZMM16-31, so the JIT's view of `xmm` and every VEX path are untouched. A
//! `Vector` is the register as four 128-bit lanes, lane 0 least significant; the legacy SSE kernels
//! (the pure 128-bit functions in `sse.rs`/`sse_ext.rs`) run on one lane at a time, which gives the
//! per-lane semantics of the 512-bit forms for free.
//!
//! Every handler computes its whole result before writing anything, so a fault (memory or #XM)
//! leaves registers untouched; memory stores probe every page they will write first.
//!
//! This is a child of `sse` so it can reach the soft-float engine behind the FP handlers.

use super::*;
use crate::cpu::avx_special::LANE_BYTES;
use crate::cpu::decode::{EVEX_OP_FLAG, Insn, VEX_OP_FLAG};
use crate::cpu::sys::CR4_OSXSAVE;
use crate::cpu::xsave::{XCR0_AVX, XCR0_AVX512, XCR0_SSE};

#[path = "avx512_data.rs"]
mod data;
#[path = "avx512_fp.rs"]
mod fp;
#[path = "avx512_gather.rs"]
mod gather;
#[path = "avx512_int.rs"]
mod int;
#[path = "avx512_lanes.rs"]
mod lanes;
#[path = "avx512_mask.rs"]
pub(in crate::cpu) mod mask;
#[path = "avx512_misc.rs"]
pub(in crate::cpu) mod misc;

/// A vector register as four 128-bit lanes, lane 0 least significant.
pub(in crate::cpu) type Vector = [u128; 4];
pub(in crate::cpu) const VECTOR_BYTES: usize = 4 * LANE_BYTES;
const LANES: usize = 4;
/// Registers ZMM0-15 live in `xmm`/`ymm_hi`/`zmm_hi`, ZMM16-31 in `zmm_extra`.
const LEGACY_REGISTER_COUNT: usize = 16;
/// EVEX.L'L values.
const LENGTH_128: u8 = 0;
const LENGTH_256: u8 = 1;
const LENGTH_512: u8 = 2;

pub(in crate::cpu) fn size_mask(size: usize) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1 << (size * 8)) - 1
    }
}

/// Element `index` of `size` bytes (1, 2, 4 or 8) of `v`.
pub(in crate::cpu) fn element(v: &Vector, size: usize, index: usize) -> u64 {
    let bit = index * size * 8;
    (v[bit / 128] >> (bit % 128)) as u64 & size_mask(size)
}

pub(in crate::cpu) fn set_element(v: &mut Vector, size: usize, index: usize, value: u64) {
    let bit = index * size * 8;
    let shift = bit % 128;
    let keep = !((size_mask(size) as u128) << shift);
    v[bit / 128] = v[bit / 128] & keep | ((value & size_mask(size)) as u128) << shift;
}

pub(in crate::cpu) fn signed(value: u64, size: usize) -> i64 {
    let unused = 64 - size as u32 * 8;
    ((value << unused) as i64) >> unused
}

/// Builds a vector of `bytes` bytes element by element.
pub(in crate::cpu) fn build(
    bytes: usize,
    element_bytes: usize,
    mut f: impl FnMut(usize) -> u64,
) -> Vector {
    let mut out = [0u128; LANES];
    for k in 0..bytes / element_bytes {
        set_element(&mut out, element_bytes, k, f(k));
    }
    out
}

/// The legacy opcode space as the SSE decoder keys it.
fn prefix_of(i: &Insn) -> Prefix {
    match (i.rep, i.p66) {
        (1, _) => Prefix::F3,
        (2, _) => Prefix::F2,
        (_, true) => Prefix::P66,
        _ => Prefix::Np,
    }
}

/// How an FP instruction reacts to EVEX.b on a register operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cpu) enum Rounding {
    /// EVEX.b with a register operand is #UD.
    None,
    /// `{sae}`: all exceptions suppressed, rounding stays as MXCSR says.
    Sae,
    /// `{er}`: static rounding mode from L'L, exceptions suppressed.
    Embedded,
}

/// How the r/m operand is fetched; see `Cpu::fetch_rm`.
#[derive(Clone, Copy)]
pub(in crate::cpu) struct RmShape {
    /// Bytes of the operand (the vector length, or less for half/quarter/scalar sources).
    pub bytes: usize,
    /// Element size, for broadcast and for the disp8 scale of a broadcast.
    pub element: usize,
    pub broadcast: bool,
    /// Elements whose bit is clear in this mask (in units of `element`) are not read, so they
    /// cannot fault; they read as zero.
    pub suppress: Option<u64>,
}

impl RmShape {
    pub(in crate::cpu) fn full(bytes: usize, element: usize) -> Self {
        RmShape {
            bytes,
            element,
            broadcast: true,
            suppress: None,
        }
    }
}

/// How `exec_vector_op` handles the operands of one instruction.
#[derive(Clone, Copy)]
pub(in crate::cpu) struct OpShape {
    /// Element size of the result, which the write mask applies to.
    pub element: usize,
    /// Element size of a broadcast memory operand when it differs from `element`.
    pub rm_element: Option<usize>,
    /// The r/m operand is `vector bytes / n` bytes (half, quarter, eighth sources).
    pub rm_bytes: Option<usize>,
    pub broadcast: bool,
    /// Masked-out elements of a memory operand are not read. Only right when source element `k`
    /// feeds result element `k`.
    pub suppress: bool,
    /// No vvvv operand (it must encode 1111).
    pub unary: bool,
    pub rounding: Rounding,
    /// Destination register when it is not ModRM.reg (the shift-by-immediate groups use vvvv).
    pub dest: Option<usize>,
}

impl OpShape {
    /// Element-wise operation: result element `k` comes from source elements `k`.
    pub(in crate::cpu) fn elementwise(element: usize) -> Self {
        OpShape {
            element,
            rm_element: None,
            rm_bytes: None,
            broadcast: true,
            suppress: true,
            unary: false,
            rounding: Rounding::None,
            dest: None,
        }
    }

    pub(in crate::cpu) fn unary(self) -> Self {
        OpShape {
            unary: true,
            ..self
        }
    }

    /// Any source element can reach any result element (permutes), so nothing is suppressed.
    pub(in crate::cpu) fn crossing(self) -> Self {
        OpShape {
            suppress: false,
            ..self
        }
    }
}

/// What the instruction table knows about a lane-wise instruction.
#[derive(Clone, Copy)]
struct LaneSpec {
    /// Element size the write mask applies to.
    element: usize,
    /// Scalar: one element, the rest of the low lane comes from vvvv.
    scalar: bool,
    /// No vvvv operand (it must encode 1111).
    unary: bool,
    rounding: Rounding,
    /// The r/m operand supplies a 128-bit shift count for every lane, not a vector.
    count_operand: bool,
}

const fn lane(element: usize, rounding: Rounding) -> LaneSpec {
    LaneSpec {
        element,
        scalar: false,
        unary: false,
        rounding,
        count_operand: false,
    }
}

/// The instructions that run unchanged, 128 bits at a time, through the legacy SSE kernels. The
/// element size is only for masking and broadcast: the kernels know their own.
fn lane_spec(map: u16, opcode: u8, prefix: Prefix, wide: bool) -> Option<LaneSpec> {
    use Prefix::*;
    let packed = |p: Prefix| if p == Np { 4 } else { 8 };
    let wide_size = if wide { 8 } else { 4 };
    let binary = |element, rounding| Some(lane(element, rounding));
    let unary = |element, rounding| {
        Some(LaneSpec {
            unary: true,
            ..lane(element, rounding)
        })
    };
    let scalar = |element, rounding| {
        Some(LaneSpec {
            scalar: true,
            ..lane(element, rounding)
        })
    };
    let count = |element| {
        Some(LaneSpec {
            count_operand: true,
            ..lane(element, Rounding::None)
        })
    };
    use Rounding::{Embedded, None as NoRounding, Sae};
    match (map, opcode, prefix) {
        (1, 0x14 | 0x15, Np | P66) => binary(packed(prefix), NoRounding),
        (1, 0x51, Np | P66) => unary(packed(prefix), Embedded),
        (1, 0x51, F3) => scalar(4, Embedded),
        (1, 0x51, F2) => scalar(8, Embedded),
        (1, 0x58 | 0x59 | 0x5c | 0x5e, Np | P66) => binary(packed(prefix), Embedded),
        (1, 0x58 | 0x59 | 0x5c | 0x5e, F3) => scalar(4, Embedded),
        (1, 0x58 | 0x59 | 0x5c | 0x5e, F2) => scalar(8, Embedded),
        (1, 0x5d | 0x5f, Np | P66) => binary(packed(prefix), Sae),
        (1, 0x5d | 0x5f, F3) => scalar(4, Sae),
        (1, 0x5d | 0x5f, F2) => scalar(8, Sae),
        (1, 0x54..=0x57, Np | P66) => binary(packed(prefix), NoRounding),
        (1, 0x5b, Np | P66) if !wide => unary(4, Embedded),
        (1, 0x5b, F3) if !wide => unary(4, Sae),
        (1, 0xc6, Np | P66) => binary(packed(prefix), NoRounding),
        (1, 0x12, F2) => unary(8, NoRounding),
        (1, 0x12, F3) | (1, 0x16, F3) => unary(4, NoRounding),
        (1, 0x70, P66) => unary(4, NoRounding),
        (1, 0x70, F3 | F2) => unary(2, NoRounding),
        (1, 0xdb | 0xdf | 0xeb | 0xef, P66) => binary(wide_size, NoRounding),
        (1, 0xd1 | 0xe1 | 0xf1, P66) => count(2),
        (1, 0xd2 | 0xe2 | 0xf2, P66) if !(opcode == 0xe2 && wide) => count(4),
        (1, 0xd3 | 0xf3, P66) => count(8),
        (1, 0x60 | 0x63 | 0x67 | 0x68 | 0xd8 | 0xda | 0xdc | 0xde | 0xe0 | 0xe8, P66)
        | (1, 0xec | 0xf8 | 0xfc, P66) => binary(1, NoRounding),
        (1, 0x61 | 0x69 | 0x6b | 0xd5 | 0xd9 | 0xdd | 0xe3 | 0xe4 | 0xe5, P66)
        | (1, 0xe9 | 0xea | 0xed | 0xee | 0xf9 | 0xfd, P66) => binary(2, NoRounding),
        (1, 0x62 | 0x6a | 0xf5 | 0xfa | 0xfe, P66) => binary(4, NoRounding),
        (1, 0x6c | 0x6d | 0xd4 | 0xf4 | 0xfb, P66) => binary(8, NoRounding),
        (2, 0x00, P66) => binary(1, NoRounding),
        (2, 0x04 | 0x0b | 0x2b, P66) => binary(2, NoRounding),
        (2, 0x1c | 0x38 | 0x3c, P66) => unary_or_binary(opcode, 1),
        (2, 0x1d | 0x3a | 0x3e, P66) => unary_or_binary(opcode, 2),
        (2, 0x1e, P66) => unary(4, NoRounding),
        (2, 0x28, P66) => binary(8, NoRounding),
        (2, 0x39 | 0x3b | 0x3d | 0x3f, P66) if !wide => binary(4, NoRounding),
        (2, 0x40, P66) if !wide => binary(4, NoRounding),
        (3, 0x0f, P66) => binary(1, NoRounding),
        // psadbw, AES rounds and carry-less multiply have no write mask in EVEX; the element size
        // only keeps the generic commit in range
        (1, 0xf6, P66) | (2, 0xdc..=0xdf, P66) | (3, 0x44, P66) => binary(8, NoRounding),
        _ => None,
    }
}

/// pabsb/pabsw are unary, their min/max neighbours binary; one match arm above covers both.
fn unary_or_binary(opcode: u8, element: usize) -> Option<LaneSpec> {
    let unary = matches!(opcode, 0x1c | 0x1d);
    Some(LaneSpec {
        unary,
        ..lane(element, Rounding::None)
    })
}

impl Cpu {
    pub(in crate::cpu) fn zmm(&self, n: usize) -> Vector {
        if n < LEGACY_REGISTER_COUNT {
            [
                self.xmm[n],
                self.ymm_hi[n],
                self.zmm_hi[n][0],
                self.zmm_hi[n][1],
            ]
        } else {
            self.zmm_extra[n - LEGACY_REGISTER_COUNT]
        }
    }

    pub(in crate::cpu) fn set_zmm(&mut self, n: usize, v: Vector) {
        if n < LEGACY_REGISTER_COUNT {
            self.xmm[n] = v[0];
            self.ymm_hi[n] = v[1];
            self.zmm_hi[n] = [v[2], v[3]];
        } else {
            self.zmm_extra[n - LEGACY_REGISTER_COUNT] = v;
        }
    }

    /// Zeroes bits 511:256 of ZMM0-15, which every VEX-encoded write to a register does.
    pub(in crate::cpu) fn clear_zmm_upper(&mut self, n: usize) {
        if n < LEGACY_REGISTER_COUNT {
            self.zmm_hi[n] = [0; 2];
        }
    }

    /// #UD unless the OS enabled the AVX-512 state (CR4.OSXSAVE, XCR0 bits 7:5 and 2:1), then #NM
    /// under CR0.TS.
    pub(in crate::cpu) fn evex_gate(&self) -> ExecResult<()> {
        const NEEDED: u64 = XCR0_SSE | XCR0_AVX | XCR0_AVX512;
        if self.cr4 & CR4_OSXSAVE == 0 || self.xcr0 & NEEDED != NEEDED {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        Ok(())
    }

    /// Vector length in bytes from EVEX.L'L; 3 is reserved. With embedded rounding on a register
    /// operand L'L carries the rounding mode and the operation is 512 bits wide.
    pub(in crate::cpu) fn vector_bytes(&self, i: &Insn, rounding: Rounding) -> ExecResult<usize> {
        if i.evex_b && i.md == 3 {
            return match rounding {
                Rounding::None => Err(Exception::invalid_opcode()),
                _ => Ok(VECTOR_BYTES),
            };
        }
        match i.evex_ll {
            LENGTH_128 => Ok(LANE_BYTES),
            LENGTH_256 => Ok(2 * LANE_BYTES),
            LENGTH_512 => Ok(VECTOR_BYTES),
            _ => Err(Exception::invalid_opcode()),
        }
    }

    /// The write mask: all ones when EVEX.aaa is 0 (k0 cannot mask).
    pub(in crate::cpu) fn write_mask(&self, i: &Insn) -> u64 {
        if i.evex_mask == 0 {
            u64::MAX
        } else {
            self.opmask[i.evex_mask as usize]
        }
    }

    /// Stores `result` into `dest`: elements whose mask bit is set take the new value, the others
    /// keep the old one (merging) or become zero (EVEX.z); everything above `bytes` is zeroed.
    pub(in crate::cpu) fn commit_masked(
        &mut self,
        i: &Insn,
        dest: usize,
        element_bytes: usize,
        bytes: usize,
        result: &Vector,
    ) {
        let masked = i.evex_mask != 0;
        if !masked && bytes.is_multiple_of(LANE_BYTES) {
            // Every element takes the new value: copy whole lanes. This is nearly every vector
            // instruction a guest runs, and the element loop below costs several calls per byte.
            let mut out = [0u128; LANES];
            out[..bytes / LANE_BYTES].copy_from_slice(&result[..bytes / LANE_BYTES]);
            self.set_zmm(dest, out);
            return;
        }
        let mask = self.write_mask(i);
        let old = if masked && !i.evex_zeroing {
            self.zmm(dest)
        } else {
            [0; LANES]
        };
        let mut out = [0u128; LANES];
        for k in 0..bytes / element_bytes {
            let value = if mask >> k & 1 == 1 {
                element(result, element_bytes, k)
            } else {
                element(&old, element_bytes, k)
            };
            set_element(&mut out, element_bytes, k, value);
        }
        self.set_zmm(dest, out);
    }

    /// Scalar write: only element 0 is masked; the rest of the low lane is `result`'s.
    pub(in crate::cpu) fn commit_scalar(
        &mut self,
        i: &Insn,
        dest: usize,
        element_bytes: usize,
        result: u128,
    ) {
        let mut low = result;
        if i.evex_mask != 0 && self.opmask[i.evex_mask as usize] & 1 == 0 {
            let keep = if i.evex_zeroing {
                0
            } else {
                self.zmm(dest)[0] & size_mask(element_bytes) as u128
            };
            low = low & !(size_mask(element_bytes) as u128) | keep;
        }
        self.set_zmm(dest, [low, 0, 0, 0]);
    }

    /// Linear address of the memory operand, with the compressed disp8 scaled by `scale`.
    pub(in crate::cpu) fn evex_address(
        &self,
        i: &Insn,
        next: u64,
        scale: usize,
    ) -> ExecResult<u64> {
        let mut scaled = *i;
        if i.evex_disp8 {
            scaled.disp = i.disp.wrapping_mul(scale as i64);
        }
        self.linear_address(&scaled, self.effective_address(&scaled, next))
    }

    /// Reads `bytes` bytes (at most 64) at `address` as a vector, element by element when some
    /// elements are suppressed.
    pub(in crate::cpu) fn load_vector(
        &mut self,
        address: u64,
        bytes: usize,
        element_bytes: usize,
        suppress: Option<u64>,
    ) -> ExecResult<Vector> {
        let mut out = [0u128; LANES];
        match suppress {
            None => {
                for (lane, slot) in out.iter_mut().enumerate().take(bytes.div_ceil(LANE_BYTES)) {
                    let at = address.wrapping_add((lane * LANE_BYTES) as u64);
                    let remaining = bytes - lane * LANE_BYTES;
                    *slot = if remaining >= LANE_BYTES {
                        self.read128(at)?
                    } else {
                        self.read(at, remaining as u8)? as u128
                    };
                }
            }
            Some(mask) => {
                for k in 0..bytes / element_bytes {
                    if mask >> k & 1 == 1 {
                        let at = address.wrapping_add((k * element_bytes) as u64);
                        let value = self.read(at, element_bytes as u8)?;
                        set_element(&mut out, element_bytes, k, value);
                    }
                }
            }
        }
        Ok(out)
    }

    /// The r/m operand: a register, or memory with compressed disp8, broadcast and fault
    /// suppression. A register comes back whole; callers use the part they need.
    pub(in crate::cpu) fn fetch_rm(
        &mut self,
        i: &Insn,
        next: u64,
        shape: RmShape,
    ) -> ExecResult<Vector> {
        if i.md == 3 {
            return Ok(self.zmm(i.rm as usize));
        }
        let broadcast = i.evex_b;
        if broadcast && !shape.broadcast {
            return Err(Exception::invalid_opcode());
        }
        let scale = if broadcast {
            shape.element
        } else {
            shape.bytes
        };
        let address = self.evex_address(i, next, scale)?;
        if broadcast {
            let unmasked = match shape.suppress {
                Some(mask) => mask != 0,
                None => true,
            };
            let value = if unmasked {
                self.read(address, shape.element as u8)?
            } else {
                0
            };
            let mut out = [0u128; LANES];
            for k in 0..shape.bytes / shape.element {
                set_element(&mut out, shape.element, k, value);
            }
            return Ok(out);
        }
        self.load_vector(address, shape.bytes, shape.element, shape.suppress)
    }

    /// Writes `bytes` of `v` to `address`, only the elements whose mask bit is set. Every page
    /// involved is probed first, so a fault leaves memory untouched.
    pub(in crate::cpu) fn store_vector(
        &mut self,
        address: u64,
        v: &Vector,
        bytes: usize,
        element_bytes: usize,
        mask: u64,
    ) -> ExecResult<()> {
        for k in 0..bytes / element_bytes {
            if mask >> k & 1 == 1 {
                let at = address.wrapping_add((k * element_bytes) as u64);
                self.translate(at, mem::AccessKind::Write, false)?;
                let last = at.wrapping_add(element_bytes as u64 - 1);
                self.translate(last, mem::AccessKind::Write, false)?;
            }
        }
        for k in 0..bytes / element_bytes {
            if mask >> k & 1 == 1 {
                let at = address.wrapping_add((k * element_bytes) as u64);
                self.write(at, element_bytes as u8, element(v, element_bytes, k))?;
            }
        }
        Ok(())
    }

    /// Runs the legacy 128-bit kernel for `i` on one lane. `old` is the kernel's destination
    /// operand (vvvv for binary operations).
    fn legacy_lane(&mut self, i: &Insn, old: u128, source: u128) -> ExecResult<u128> {
        self.run_legacy_registers(i, old, source)
    }

    /// Embedded rounding / SAE: runs `body` with MXCSR rounding replaced and every exception
    /// masked, then puts MXCSR back, so nothing is raised or recorded.
    pub(in crate::cpu) fn with_static_rounding<T>(
        &mut self,
        i: &Insn,
        rounding: Rounding,
        body: impl FnOnce(&mut Cpu) -> ExecResult<T>,
    ) -> ExecResult<T> {
        if !(i.evex_b && i.md == 3) {
            return body(self);
        }
        let saved = self.mxcsr;
        let mut temporary = saved | (MXCSR_EXCEPTION_FLAGS << MXCSR_MASK_SHIFT);
        if rounding == Rounding::Embedded {
            temporary &= !(MXCSR_RC_MASK << MXCSR_RC_SHIFT);
            temporary |= (i.evex_ll as u32 & MXCSR_RC_MASK) << MXCSR_RC_SHIFT;
        }
        self.mxcsr = temporary;
        let outcome = body(self);
        self.mxcsr = saved;
        outcome
    }

    /// Entry for every EVEX-encoded instruction.
    pub(in crate::cpu) fn exec_evex(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        self.evex_gate()?;
        let op = i.op & !EVEX_OP_FLAG;
        let map = op >> 8;
        let opcode = (op & 0xff) as u8;
        let prefix = prefix_of(i);
        let wide = i.opsize == 8;
        let handled = if let Some(spec) = lane_spec(map, opcode, prefix, wide) {
            self.exec_lane_wise(i, next, spec)?;
            true
        } else {
            self.exec_evex_data(i, next, map, opcode, prefix, wide)?
                || (prefix == Prefix::P66 && self.exec_evex_gather(i, next, map, opcode, wide)?)
                || self.exec_evex_int(i, next, map, opcode, prefix, wide)?
        };
        if !handled {
            return Err(self.unimplemented_opcode(i));
        }
        self.rip = next;
        Ok(())
    }

    /// The generic vector instruction: fetches vvvv and r/m (register, memory or broadcast, fault
    /// suppression per `shape`), runs `body(old destination, first source, second source, vector
    /// bytes)` and commits the result under the write mask.
    pub(in crate::cpu) fn exec_vector_op(
        &mut self,
        i: &Insn,
        next: u64,
        shape: OpShape,
        body: impl FnOnce(&mut Cpu, &Vector, &Vector, &Vector, usize) -> ExecResult<Vector>,
    ) -> ExecResult<()> {
        if shape.unary && i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = self.vector_bytes(i, shape.rounding)?;
        let mask = self.write_mask(i);
        let rm_shape = RmShape {
            bytes: shape.rm_bytes.map_or(bytes, |divide| bytes / divide),
            element: shape.rm_element.unwrap_or(shape.element),
            broadcast: shape.broadcast,
            suppress: (i.evex_mask != 0 && shape.suppress).then_some(mask),
        };
        let second = self.fetch_rm(i, next, rm_shape)?;
        let first = if shape.unary {
            [0; LANES]
        } else {
            self.zmm(i.vex_v as usize)
        };
        let dest = shape.dest.unwrap_or(i.reg as usize);
        let old = self.zmm(dest);
        let rounding = shape.rounding;
        let result =
            self.with_static_rounding(i, rounding, |cpu| body(cpu, &old, &first, &second, bytes))?;
        self.commit_masked(i, dest, shape.element, bytes, &result);
        Ok(())
    }

    fn exec_lane_wise(&mut self, i: &Insn, next: u64, spec: LaneSpec) -> ExecResult<()> {
        if spec.unary && i.vex_v != 0 {
            return Err(Exception::invalid_opcode());
        }
        let bytes = if spec.scalar {
            LANE_BYTES
        } else {
            self.vector_bytes(i, spec.rounding)?
        };
        if spec.scalar && i.evex_b && i.md == 3 && spec.rounding == Rounding::None {
            return Err(Exception::invalid_opcode());
        }
        let dest = i.reg as usize;
        let mask = self.write_mask(i);
        let source = if spec.count_operand {
            // A shift count is always 128 bits, whatever the vector length.
            let shape = RmShape {
                bytes: LANE_BYTES,
                element: LANE_BYTES,
                broadcast: false,
                suppress: None,
            };
            let count = self.fetch_rm(i, next, shape)?[0];
            [count; LANES]
        } else {
            let shape = if spec.scalar {
                RmShape {
                    bytes: spec.element,
                    element: spec.element,
                    broadcast: false,
                    suppress: None,
                }
            } else {
                RmShape {
                    suppress: (i.evex_mask != 0).then_some(mask),
                    ..RmShape::full(bytes, spec.element)
                }
            };
            self.fetch_rm(i, next, shape)?
        };
        let first = self.zmm(i.vex_v as usize);
        let rounding = spec.rounding;
        let result = self.with_static_rounding(i, rounding, |cpu| {
            let mut result = [0u128; LANES];
            for k in 0..bytes / LANE_BYTES {
                // Skip lanes whose elements are all masked: nothing observable (not even an
                // exception flag) may come from them.
                let lane_elements = LANE_BYTES / spec.element;
                let lane_mask = (mask >> (k * lane_elements)) & ((1 << lane_elements) - 1);
                if lane_mask == 0 && !spec.scalar {
                    continue;
                }
                let old = if spec.unary { 0 } else { first[k] };
                result[k] = cpu.legacy_lane(i, old, source[k])?;
            }
            Ok(result)
        })?;
        if spec.scalar {
            self.commit_scalar(i, dest, spec.element, result[0]);
        } else {
            self.commit_masked(i, dest, spec.element, bytes, &result);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "evex_tests.rs"]
mod tests;
