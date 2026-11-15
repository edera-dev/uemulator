//! BMI1 and BMI2: the VEX-encoded general-purpose-register instructions (andn, bextr, blsi, blsmsk,
//! blsr, bzhi, mulx, pdep, pext, rorx, sarx, shlx, shrx) and, in `exec.rs`, tzcnt. None of them
//! touches a vector register, so they need no XSAVE state and no CR4/XCR0 enabling: VEX decodes
//! as soon as the CPUID bits say the instructions exist.
//!
//! `decode` hands these over with `Insn::vex` set and `op = VEX_OP_FLAG | (map << 8) | opcode`.
//! Anything else VEX-encoded is #UD (logged once) until AVX is implemented.

use super::decode::{Insn, VEX_OP_FLAG};
use super::exception::Exception;
use super::flags::{CF, PF, SF, ZF, mask};
use super::sys::RDX;
use super::{Cpu, ExecResult};

const OP_ANDN: u16 = 0x2f2;
/// Group 17: /1 blsr, /2 blsmsk, /3 blsi.
const OP_GROUP17: u16 = 0x2f3;
const OP_BZHI_PEXT_PDEP: u16 = 0x2f5;
const OP_MULX: u16 = 0x2f6;
/// bextr (no prefix), shlx (66), sarx (F3), shrx (F2).
const OP_BEXTR_SHIFTS: u16 = 0x2f7;
const OP_RORX: u16 = 0x3f0;

const GROUP17_BLSR: u8 = 1;
const GROUP17_BLSMSK: u8 = 2;
const GROUP17_BLSI: u8 = 3;

/// `Insn::rep` values (the implied-prefix field for VEX): none, F3, F2.
const REP_NONE: u8 = 0;
const REP_F3: u8 = 1;
const REP_F2: u8 = 2;

/// bextr's control operand: start in bits 7:0, length in bits 15:8.
const BEXTR_FIELD_MASK: u64 = 0xff;
const BEXTR_LENGTH_SHIFT: u32 = 8;

/// ZF, SF and PF for a result of `size` bytes; CF, OF and AF are the caller's business.
fn result_flags(result: u64, size: u8) -> u64 {
    let sign_bit = 1u64 << (size as u32 * 8 - 1);
    let mut flags = 0;
    if result == 0 {
        flags |= ZF;
    }
    if result & sign_bit != 0 {
        flags |= SF;
    }
    if (result as u8).count_ones().is_multiple_of(2) {
        flags |= PF;
    }
    flags
}

/// Deposits the low bits of `source` into the positions set in `mask`, lowest first.
pub(super) fn parallel_deposit(source: u64, mask: u64) -> u64 {
    let mut result = 0;
    let mut remaining = mask;
    let mut next_source_bit = 0;
    while remaining != 0 {
        let lowest = remaining.isolate_lowest_one();
        if source & (1 << next_source_bit) != 0 {
            result |= lowest;
        }
        next_source_bit += 1;
        remaining &= remaining - 1;
    }
    result
}

/// Gathers the bits of `source` selected by `mask` into the low bits of the result.
pub(super) fn parallel_extract(source: u64, mask: u64) -> u64 {
    let mut result = 0;
    let mut remaining = mask;
    let mut next_result_bit = 0;
    while remaining != 0 {
        let lowest = remaining.isolate_lowest_one();
        if source & lowest != 0 {
            result |= 1 << next_result_bit;
        }
        next_result_bit += 1;
        remaining &= remaining - 1;
    }
    result
}

impl Cpu {
    /// Executes a VEX-encoded instruction; sets RIP itself like the other group handlers.
    pub(super) fn exec_vex(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let size = i.opsize;
        let operand_mask = mask(size);
        let bits = size as u32 * 8;
        let op = i.op & !VEX_OP_FLAG;
        // BMI opcodes are all in map 2/3 with specific numbers; anything else is a vector form.
        if !matches!(
            op,
            OP_ANDN | OP_GROUP17 | OP_BZHI_PEXT_PDEP | OP_MULX | OP_BEXTR_SHIFTS | OP_RORX
        ) {
            return self.exec_vex_vector(i, next);
        }
        // All the BMI forms are LZ (VEX.L = 0).
        if i.vex_l {
            return Err(Exception::invalid_opcode());
        }
        let vvvv = self.read_gpr(i.vex_v, size, true);
        match (op, i.rep, i.p66) {
            (OP_ANDN, REP_NONE, false) => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)?;
                let result = !vvvv & source & operand_mask;
                self.write_gpr(i.reg, size, true, result);
                self.set_eflags_arith(result_flags(result, size));
            }
            (OP_GROUP17, REP_NONE, false) => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let (result, carry) = match i.reg & 7 {
                    GROUP17_BLSR => (source & source.wrapping_sub(1), source == 0),
                    GROUP17_BLSMSK => (source ^ source.wrapping_sub(1), source == 0),
                    GROUP17_BLSI => (source.isolate_lowest_one(), source != 0),
                    _ => return Err(Exception::invalid_opcode()),
                };
                let result = result & operand_mask;
                self.write_gpr(i.vex_v, size, true, result);
                self.set_eflags_arith(result_flags(result, size) | if carry { CF } else { 0 });
            }
            (OP_BZHI_PEXT_PDEP, REP_NONE, false) => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let index = (vvvv & BEXTR_FIELD_MASK) as u32;
                let result = if index < bits {
                    source & ((1u64 << index) - 1)
                } else {
                    source
                };
                self.write_gpr(i.reg, size, true, result);
                let carry = if index >= bits { CF } else { 0 };
                self.set_eflags_arith(result_flags(result, size) | carry);
            }
            (OP_BZHI_PEXT_PDEP, prefix @ (REP_F3 | REP_F2), false) => {
                let loc = self.modrm_location(i, next)?;
                let selector = self.load_operand(loc, size, i)? & operand_mask;
                let source = vvvv & operand_mask;
                let result = if prefix == REP_F3 {
                    parallel_extract(source, selector)
                } else {
                    parallel_deposit(source, selector)
                };
                self.write_gpr(i.reg, size, true, result);
            }
            (OP_MULX, REP_F2, false) => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let multiplier = self.gpr[RDX] & operand_mask;
                let product = source as u128 * multiplier as u128;
                // The high half goes to ModRM.reg and the low half to vvvv. When both name the same
                // register the high half wins, so the low half is written first.
                self.write_gpr(i.vex_v, size, true, product as u64 & operand_mask);
                self.write_gpr(i.reg, size, true, (product >> bits) as u64 & operand_mask);
            }
            (OP_BEXTR_SHIFTS, REP_NONE, false) => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let start = (vvvv & BEXTR_FIELD_MASK) as u32;
                let length = ((vvvv >> BEXTR_LENGTH_SHIFT) & BEXTR_FIELD_MASK) as u32;
                let shifted = if start < bits { source >> start } else { 0 };
                let result = if length < 64 {
                    shifted & ((1u64 << length) - 1)
                } else {
                    shifted
                };
                self.write_gpr(i.reg, size, true, result);
                // CF and OF are cleared; ZF reports the result.
                self.set_eflags_arith(if result == 0 { ZF } else { 0 });
            }
            (OP_BEXTR_SHIFTS, shift_prefix, shl) if (shift_prefix == REP_NONE) == shl => {
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let count = (vvvv as u32) & (bits - 1);
                let result = if shl {
                    source << count
                } else if shift_prefix == REP_F2 {
                    source >> count
                } else {
                    // sarx: sign-extend from the operand size, shift, truncate back
                    (((source << (64 - bits)) as i64) >> (64 - bits) >> count) as u64
                } & operand_mask;
                self.write_gpr(i.reg, size, true, result);
            }
            (OP_RORX, REP_F2, false) => {
                // vvvv is unused and must be 1111 (decoded as 0).
                if i.vex_v != 0 {
                    return Err(Exception::invalid_opcode());
                }
                let loc = self.modrm_location(i, next)?;
                let source = self.load_operand(loc, size, i)? & operand_mask;
                let count = (i.imm as u32) & (bits - 1);
                let result = if count == 0 {
                    source
                } else {
                    ((source >> count) | (source << (bits - count))) & operand_mask
                };
                self.write_gpr(i.reg, size, true, result);
            }
            _ => return self.exec_vex_vector(i, next),
        }
        self.rip = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::flags::{ARITH, OF};
    use crate::testkit::{Harness, RDI, SCRATCH, Stop};

    const HLT: u8 = 0xf4;
    const RAX_REG: usize = 0;
    const RCX_REG: usize = 1;
    const RBX_REG: usize = 3;
    const R9: usize = 9;
    const R10: usize = 10;
    const DATA: u64 = SCRATCH + 0x1000;

    // Each constant is the assembly in its name, from hack/asm.sh, followed by hlt. Operand order
    // is Intel syntax: for three-operand forms (dest, first source, second source) the VEX.vvvv
    // register is the middle one except for bextr/bzhi/shifts, where it is the last.
    const ANDN64: &[u8] = &[0xc4, 0xe2, 0xe0, 0xf2, 0xc1, HLT]; // andn rax, rbx, rcx
    const ANDN32: &[u8] = &[0xc4, 0xe2, 0x60, 0xf2, 0xc1, HLT]; // andn eax, ebx, ecx
    const ANDN_MEM: &[u8] = &[0xc4, 0x62, 0xa8, 0xf2, 0x0f, HLT]; // andn r9, r10, [rdi]
    const BLSR64: &[u8] = &[0xc4, 0xe2, 0xf8, 0xf3, 0xc9, HLT]; // blsr rax, rcx
    const BLSR32: &[u8] = &[0xc4, 0xe2, 0x78, 0xf3, 0xc9, HLT]; // blsr eax, ecx
    const BLSMSK64: &[u8] = &[0xc4, 0xe2, 0xf8, 0xf3, 0xd1, HLT]; // blsmsk rax, rcx
    const BLSI64: &[u8] = &[0xc4, 0xe2, 0xf8, 0xf3, 0xd9, HLT]; // blsi rax, rcx
    const BEXTR64: &[u8] = &[0xc4, 0xe2, 0xe0, 0xf7, 0xc1, HLT]; // bextr rax, rcx, rbx
    const BEXTR32: &[u8] = &[0xc4, 0xe2, 0x60, 0xf7, 0xc1, HLT]; // bextr eax, ecx, ebx
    const BZHI64: &[u8] = &[0xc4, 0xe2, 0xe0, 0xf5, 0xc1, HLT]; // bzhi rax, rcx, rbx
    const BZHI32: &[u8] = &[0xc4, 0xe2, 0x60, 0xf5, 0xc1, HLT]; // bzhi eax, ecx, ebx
    const PDEP64: &[u8] = &[0xc4, 0xe2, 0xe3, 0xf5, 0xc1, HLT]; // pdep rax, rbx, rcx
    const PEXT64: &[u8] = &[0xc4, 0xe2, 0xe2, 0xf5, 0xc1, HLT]; // pext rax, rbx, rcx
    const PDEP32: &[u8] = &[0xc4, 0xe2, 0x63, 0xf5, 0xc1, HLT]; // pdep eax, ebx, ecx
    const MULX64: &[u8] = &[0xc4, 0xe2, 0xe3, 0xf6, 0xc1, HLT]; // mulx rax, rbx, rcx
    const MULX32: &[u8] = &[0xc4, 0xe2, 0x63, 0xf6, 0xc1, HLT]; // mulx eax, ebx, ecx
    const MULX_MEM: &[u8] = &[0xc4, 0x62, 0xab, 0xf6, 0x0f, HLT]; // mulx r9, r10, [rdi]
    const RORX64: &[u8] = &[0xc4, 0xe3, 0xfb, 0xf0, 0xc1, 0x05, HLT]; // rorx rax, rcx, 5
    const RORX32: &[u8] = &[0xc4, 0xe3, 0x7b, 0xf0, 0xc1, 0x03, HLT]; // rorx eax, ecx, 3
    const SARX64: &[u8] = &[0xc4, 0xe2, 0xe2, 0xf7, 0xc1, HLT]; // sarx rax, rcx, rbx
    const SHLX64: &[u8] = &[0xc4, 0xe2, 0xe1, 0xf7, 0xc1, HLT]; // shlx rax, rcx, rbx
    const SHRX64: &[u8] = &[0xc4, 0xe2, 0xe3, 0xf7, 0xc1, HLT]; // shrx rax, rcx, rbx
    const SARX32: &[u8] = &[0xc4, 0xe2, 0x62, 0xf7, 0xc1, HLT]; // sarx eax, ecx, ebx
    const SHLX32: &[u8] = &[0xc4, 0xe2, 0x61, 0xf7, 0xc1, HLT]; // shlx eax, ecx, ebx
    const SHRX32: &[u8] = &[0xc4, 0xe2, 0x63, 0xf7, 0xc1, HLT]; // shrx eax, ecx, ebx
    const TZCNT64: &[u8] = &[0xf3, 0x48, 0x0f, 0xbc, 0xc1, HLT]; // tzcnt rax, rcx
    const TZCNT32: &[u8] = &[0xf3, 0x0f, 0xbc, 0xc1, HLT]; // tzcnt eax, ecx
    const TZCNT16: &[u8] = &[0x66, 0xf3, 0x0f, 0xbc, 0xc1, HLT]; // tzcnt ax, cx

    const VALUES: [u64; 14] = [
        0,
        1,
        2,
        0x80,
        0xff,
        0x8000_0000,
        0xffff_ffff,
        0x1_0000_0000,
        0x8000_0000_0000_0000,
        u64::MAX,
        0x1234_5678_9abc_def0,
        0xf0f0_f0f0_f0f0_f0f0,
        0x5555_5555_5555_5555,
        0xdead_beef_0bad_f00d,
    ];

    /// Runs `code` with RAX=junk, RBX=`b`, RCX=`c`, RDX=`d`, flags preset to all-ones so a missing
    /// flag write shows.
    fn run(code: &[u8], b: u64, c: u64, d: u64) -> Harness {
        let mut h = Harness::new(code);
        h.cpu.gpr[RAX_REG] = 0xaaaa_aaaa_aaaa_aaaa;
        h.cpu.gpr[RBX_REG] = b;
        h.cpu.gpr[RCX_REG] = c;
        h.cpu.gpr[2] = d;
        h.cpu.set_eflags_arith(ARITH);
        assert_eq!(h.run(100), Stop::Halted);
        h
    }

    fn flags(h: &Harness) -> u64 {
        h.cpu.eflags_full() & ARITH
    }

    fn rax(h: &Harness) -> u64 {
        h.cpu.gpr[RAX_REG]
    }

    fn zsp(result: u64, bits: u32) -> u64 {
        let mut f = 0;
        if result == 0 {
            f |= ZF;
        }
        if result >> (bits - 1) & 1 != 0 {
            f |= SF;
        }
        if (result as u8).count_ones().is_multiple_of(2) {
            f |= PF;
        }
        f
    }

    fn fault(vector: u8) -> Stop {
        Stop::Fault {
            vector,
            error_code: None,
            software: false,
        }
    }

    #[test]
    fn andn_complements_the_vvvv_operand() {
        for &b in &VALUES {
            for &c in &VALUES {
                let h = run(ANDN64, b, c, 0);
                assert_eq!(rax(&h), !b & c, "{b:#x} {c:#x}");
                assert_eq!(flags(&h) & (CF | OF), 0);
                assert_eq!(flags(&h) & (ZF | SF), zsp(!b & c, 64) & (ZF | SF));
                let h = run(ANDN32, b, c, 0);
                assert_eq!(rax(&h), (!(b as u32) & c as u32) as u64);
            }
        }
    }

    #[test]
    fn andn_reads_memory_and_uses_the_extended_registers() {
        let mut h = Harness::new(ANDN_MEM);
        h.cpu.gpr[RDI] = DATA;
        h.write_u64(DATA, 0xff00_ff00_ff00_ff00);
        h.cpu.gpr[R10] = 0x0f0f_0f0f_0f0f_0f0f;
        assert_eq!(h.run(100), Stop::Halted);
        assert_eq!(h.cpu.gpr[R9], 0xf000_f000_f000_f000);
    }

    #[test]
    fn bit_deposit_and_extract_cover_every_single_bit_and_zero() {
        assert_eq!(parallel_deposit(u64::MAX, 0), 0);
        assert_eq!(parallel_extract(u64::MAX, 0), 0);
        for bit in 0..u64::BITS {
            let mask = 1u64 << bit;
            assert_eq!(parallel_deposit(1, mask), mask);
            assert_eq!(parallel_deposit(2, mask), 0);
            assert_eq!(parallel_extract(mask, mask), 1);
            assert_eq!(parallel_extract(!mask, mask), 0);
        }
    }

    #[test]
    fn blsr_blsmsk_blsi_match_their_definitions_and_flags() {
        for &c in &VALUES {
            let h = run(BLSR64, 0, c, 0);
            let expected = c & c.wrapping_sub(1);
            assert_eq!(rax(&h), expected);
            assert_eq!(flags(&h) & CF != 0, c == 0);
            assert_eq!(flags(&h) & OF, 0);
            assert_eq!(flags(&h) & (ZF | SF), zsp(expected, 64) & (ZF | SF));

            let h = run(BLSMSK64, 0, c, 0);
            assert_eq!(rax(&h), c ^ c.wrapping_sub(1));
            assert_eq!(flags(&h) & CF != 0, c == 0);
            // blsmsk always clears ZF: the result has at least bit 0 set unless c == 0 gives all ones
            assert_eq!(flags(&h) & ZF, 0);

            let h = run(BLSI64, 0, c, 0);
            assert_eq!(
                rax(&h),
                if c == 0 {
                    0
                } else {
                    1u64 << c.trailing_zeros()
                }
            );
            assert_eq!(flags(&h) & CF != 0, c != 0);

            let h = run(BLSR32, 0, c, 0);
            let c32 = c as u32;
            assert_eq!(rax(&h), (c32 & c32.wrapping_sub(1)) as u64);
            assert_eq!(flags(&h) & CF != 0, c32 == 0);
        }
    }

    #[test]
    fn bextr_extracts_start_and_length() {
        let bextr = |code: &[u8], value: u64, start: u64, length: u64| {
            rax(&run(code, start | length << 8, value, 0))
        };
        assert_eq!(bextr(BEXTR64, 0x1234_5678_9abc_def0, 8, 16), 0xbcde);
        assert_eq!(bextr(BEXTR64, u64::MAX, 60, 64), 0xf);
        assert_eq!(bextr(BEXTR64, u64::MAX, 0, 64), u64::MAX);
        assert_eq!(
            bextr(BEXTR64, u64::MAX, 64, 8),
            0,
            "start beyond the operand"
        );
        assert_eq!(bextr(BEXTR64, u64::MAX, 255, 255), 0);
        assert_eq!(bextr(BEXTR64, u64::MAX, 4, 0), 0, "zero length");
        assert_eq!(bextr(BEXTR32, 0xffff_ffff_8765_4321, 4, 8), 0x32);
        assert_eq!(
            bextr(BEXTR32, u64::MAX, 32, 8),
            0,
            "32-bit start 32 is out of range"
        );
        let h = run(BEXTR64, 4 | 8 << 8, 0xf0, 0);
        assert_eq!(flags(&h) & (CF | OF | ZF), 0);
        let h = run(BEXTR64, 4 | 8 << 8, 0, 0);
        assert_eq!(flags(&h) & (CF | OF | ZF), ZF);
    }

    #[test]
    fn bzhi_clears_bits_from_the_index_up_and_reports_overflow_in_cf() {
        for &value in &VALUES {
            for index in [0u64, 1, 7, 31, 32, 33, 63, 64, 65, 255, 0x1_0000_0003] {
                let h = run(BZHI64, index, value, 0);
                let n = index & 0xff;
                let expected = if n < 64 {
                    value & ((1u64 << n) - 1)
                } else {
                    value
                };
                assert_eq!(rax(&h), expected, "{value:#x} index {index}");
                assert_eq!(flags(&h) & CF != 0, n > 63);
                assert_eq!(flags(&h) & OF, 0);
                assert_eq!(flags(&h) & (ZF | SF), zsp(expected, 64) & (ZF | SF));
                let h = run(BZHI32, index, value, 0);
                let value32 = value as u32;
                let expected = if n < 32 {
                    value32 & ((1u32 << n) - 1)
                } else {
                    value32
                };
                assert_eq!(rax(&h), expected as u64);
                assert_eq!(flags(&h) & CF != 0, n > 31);
            }
        }
    }

    #[test]
    fn pext_and_pdep_follow_the_manual_examples_and_invert_each_other() {
        // pext gathers the high nibble of every byte; pdep scatters them back.
        assert_eq!(rax(&run(PEXT64, 0x1234_5678, 0xf0f0_f0f0, 0)), 0x1357);
        assert_eq!(rax(&run(PDEP64, 0x1357, 0xf0f0_f0f0, 0)), 0x1030_5070);
        assert_eq!(rax(&run(PEXT64, u64::MAX, u64::MAX, 0)), u64::MAX);
        assert_eq!(
            rax(&run(PDEP64, 0x8000_0000_0000_0000, u64::MAX, 0)),
            0x8000_0000_0000_0000
        );
        assert_eq!(rax(&run(PEXT64, 0xdead_beef, 0, 0)), 0);
        assert_eq!(
            rax(&run(
                PDEP32,
                0xffff_ffff_ffff_ffff,
                0xffff_ffff_0f0f_0f0f,
                0
            )),
            0x0f0f_0f0f
        );
        for &mask in &VALUES {
            for &source in &VALUES {
                let extracted = rax(&run(PEXT64, source, mask, 0));
                let deposited = rax(&run(PDEP64, extracted, mask, 0));
                assert_eq!(deposited, source & mask, "{source:#x} {mask:#x}");
                // reference: walk the mask bit by bit
                let mut expected = 0u64;
                let mut out = 0;
                for bit in 0..64 {
                    if mask >> bit & 1 != 0 {
                        expected |= (source >> bit & 1) << out;
                        out += 1;
                    }
                }
                assert_eq!(extracted, expected);
            }
        }
    }

    #[test]
    fn mulx_multiplies_by_rdx_without_touching_flags() {
        let h = run(MULX64, 0, u64::MAX, u64::MAX);
        // rax = high, rbx = low
        assert_eq!(rax(&h), u64::MAX - 1);
        assert_eq!(h.cpu.gpr[RBX_REG], 1);
        assert_eq!(flags(&h), ARITH, "mulx leaves every flag alone");
        let h = run(MULX64, 0, 3, 5);
        assert_eq!((rax(&h), h.cpu.gpr[RBX_REG]), (0, 15));
        for &c in &VALUES {
            for &d in &VALUES {
                let h = run(MULX64, 0, c, d);
                let product = c as u128 * d as u128;
                assert_eq!(rax(&h), (product >> 64) as u64);
                assert_eq!(h.cpu.gpr[RBX_REG], product as u64);
                let h = run(MULX32, 0, c, d);
                let product = (c as u32 as u64) * (d as u32 as u64);
                assert_eq!(rax(&h), product >> 32);
                assert_eq!(h.cpu.gpr[RBX_REG], product & 0xffff_ffff);
            }
        }
    }

    #[test]
    fn mulx_with_one_register_as_both_destinations_keeps_the_high_half() {
        // mulx rax, rax, rcx
        let h = run(&[0xc4, 0xe2, 0xfb, 0xf6, 0xc1, HLT], 0, u64::MAX, 2);
        assert_eq!(rax(&h), 1);
    }

    #[test]
    fn mulx_reads_memory_and_uses_extended_registers() {
        let mut h = Harness::new(MULX_MEM);
        h.cpu.gpr[RDI] = DATA;
        h.write_u64(DATA, 1 << 40);
        h.cpu.gpr[2] = 1 << 40;
        assert_eq!(h.run(100), Stop::Halted);
        assert_eq!((h.cpu.gpr[R9], h.cpu.gpr[R10]), (1 << 16, 0));
    }

    #[test]
    fn variable_shifts_mask_the_count_and_leave_flags_alone() {
        for &value in &VALUES {
            for count in [0u64, 1, 31, 32, 63, 64, 65, 0x1_0000_0001] {
                let h = run(SHLX64, count, value, 0);
                assert_eq!(rax(&h), value << (count & 63));
                assert_eq!(flags(&h), ARITH);
                assert_eq!(rax(&run(SHRX64, count, value, 0)), value >> (count & 63));
                assert_eq!(
                    rax(&run(SARX64, count, value, 0)),
                    ((value as i64) >> (count & 63)) as u64
                );
                let v32 = value as u32;
                assert_eq!(
                    rax(&run(SHLX32, count, value, 0)),
                    (v32 << (count & 31)) as u64
                );
                assert_eq!(
                    rax(&run(SHRX32, count, value, 0)),
                    (v32 >> (count & 31)) as u64
                );
                assert_eq!(
                    rax(&run(SARX32, count, value, 0)),
                    ((v32 as i32) >> (count & 31)) as u32 as u64
                );
            }
        }
    }

    #[test]
    fn rorx_rotates_by_an_immediate_without_flags() {
        let h = run(RORX64, 0, 0x8000_0000_0000_0001, 0);
        assert_eq!(rax(&h), 0x0c00_0000_0000_0000);
        assert_eq!(flags(&h), ARITH);
        let h = run(RORX32, 0, 0xffff_ffff_0000_0009, 0);
        assert_eq!(rax(&h), 0x2000_0001);
        // count 0 and a count above the width wrap
        assert_eq!(
            rax(&run(
                &[0xc4, 0xe3, 0x7b, 0xf0, 0xc1, 0x20, HLT],
                0,
                0x1234_5678,
                0
            )),
            0x1234_5678
        );
    }

    #[test]
    fn rorx_with_vvvv_in_use_is_undefined() {
        // rorx rax, rcx, 5 with vvvv = 0 (encoded 0b1111 inverted is 0): flip a vvvv bit
        let (_, stop) = {
            let mut h = Harness::new(&[0xc4, 0xe3, 0xc3, 0xf0, 0xc1, 0x05, HLT]);
            let stop = h.run(10);
            (h, stop)
        };
        assert_eq!(stop, fault(6));
    }

    #[test]
    fn tzcnt_counts_trailing_zeros_and_reports_zero_in_cf() {
        for &c in &VALUES {
            let h = run(TZCNT64, 0, c, 0);
            let expected = if c == 0 {
                64
            } else {
                c.trailing_zeros() as u64
            };
            assert_eq!(rax(&h), expected);
            assert_eq!(flags(&h) & CF != 0, c == 0);
            assert_eq!(flags(&h) & ZF != 0, expected == 0);
            let h = run(TZCNT32, 0, c, 0);
            let c32 = c as u32;
            assert_eq!(
                rax(&h),
                if c32 == 0 { 32 } else { c32.trailing_zeros() } as u64
            );
            assert_eq!(flags(&h) & CF != 0, c32 == 0);
        }
        let h = run(TZCNT16, 0, 0xffff_0000, 0);
        assert_eq!(rax(&h) & 0xffff, 16, "16-bit source zero gives 16");
        assert_eq!(
            rax(&h) >> 16,
            0xaaaa_aaaa_aaaa_aaaa >> 16,
            "16-bit result keeps the upper bits"
        );
    }

    #[test]
    fn plain_bsf_is_unchanged_without_the_prefix() {
        // bsf rax, rcx with rcx = 0 leaves rax alone and sets ZF
        let h = run(&[0x48, 0x0f, 0xbc, 0xc1, HLT], 0, 0, 0);
        assert_eq!(rax(&h), 0xaaaa_aaaa_aaaa_aaaa);
        assert_ne!(flags(&h) & ZF, 0);
    }

    #[test]
    fn unsupported_vex_instructions_and_forms_are_undefined() {
        // vzeroupper (AVX), andn with VEX.L set, andn with a 66 prefix, VEX with a legacy
        // prefix in front, an unknown map
        for code in [
            &[0xc5, 0xf8, 0x77, HLT][..],
            &[0xc4, 0xe2, 0xe4, 0xf2, 0xc1, HLT],
            &[0xc4, 0xe2, 0xe1, 0xf2, 0xc1, HLT],
            &[0x66, 0xc4, 0xe2, 0xe0, 0xf2, 0xc1, HLT],
            &[0x48, 0xc4, 0xe2, 0xe0, 0xf2, 0xc1, HLT],
            &[0xc4, 0xe4, 0xe0, 0xf2, 0xc1, HLT],
        ] {
            let mut h = Harness::new(code);
            assert_eq!(h.run(10), fault(6), "{code:02x?}");
        }
    }

    #[test]
    fn the_extended_registers_in_vex_r_x_b_work() {
        // andn r9, r10, [rdi + rcx*1]? use the vex.x path: andn r9, r10, [rdi+r8*2]
        // c4 22 a8 f2 0c 47  (X inverted bit cleared selects r8 as index)
        let mut h = Harness::new(&[0xc4, 0x22, 0xa8, 0xf2, 0x0c, 0x47, HLT]);
        h.cpu.gpr[RDI] = DATA;
        h.cpu.gpr[8] = 8;
        h.write_u64(DATA + 16, 0xffff);
        h.cpu.gpr[R10] = 0xff;
        assert_eq!(h.run(10), Stop::Halted);
        assert_eq!(h.cpu.gpr[R9], 0xff00);
    }
}
