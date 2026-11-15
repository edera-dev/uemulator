pub(crate) const CF: u64 = 1;
pub(crate) const PF: u64 = 1 << 2;
pub(crate) const AF: u64 = 1 << 4;
pub(crate) const ZF: u64 = 1 << 6;
pub(crate) const SF: u64 = 1 << 7;
pub(crate) const TF: u64 = 1 << 8;
pub(crate) const IF: u64 = 1 << 9;
pub(crate) const DF: u64 = 1 << 10;
pub(crate) const OF: u64 = 1 << 11;
/// Two-bit field (bits 12-13), not a single flag.
pub(crate) const IOPL: u64 = 3 << 12;
pub(crate) const NT: u64 = 1 << 14;
pub(crate) const RF: u64 = 1 << 16;
pub(crate) const VM: u64 = 1 << 17;
pub(crate) const AC: u64 = 1 << 18;
/// Virtual interrupt flag and virtual interrupt pending (`vme.rs`).
pub(crate) const VIF: u64 = 1 << 19;
pub(crate) const VIP: u64 = 1 << 20;
/// Bit 1 reads as 1 in every EFLAGS value.
pub(crate) const RESERVED_ONE: u64 = 1 << 1;
/// The six status flags that `LazyFlagsOp` can defer; everything else in EFLAGS is kept eagerly.
pub(crate) const ARITH: u64 = CF | PF | AF | ZF | SF | OF;

/// How to derive the `ARITH` flags from `Cpu::lazy_dst/lazy_src/lazy_src2` when they are needed.
///
/// `lazy_dst` holds the operation's *result* (not the old destination operand) and `lazy_src`
/// the second operand. `Adc`/`Sbb` take the carry-in from bit 0 of `lazy_src2`; `Inc`/`Dec`
/// keep the untouched CF in bit 0 of `lazy_src`. `Eflags` means the flags are already
/// materialized in `Cpu::eflags`.
///
/// The JIT writes the discriminant with a 16-bit store together with `Cpu::lazy_size`, so it
/// must stay `repr(u8)`, and the variants it emits (Add, Sub, Logic, Inc, Dec, SzpCfOf) must not
/// be renumbered without checking `jit/emit*.rs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub(crate) enum LazyFlagsOp {
    #[default]
    Eflags,
    Add,
    Adc,
    Sub,
    Sbb,
    Logic,
    Inc,
    Dec,
    /// SF/ZF/PF from the result, AF clear, and CF and OF taken from `lazy_src` bits 0 and 1
    /// (worked out by the generated code). That is exactly what a shift by a non-zero count
    /// (`Cpu::shift_rotate`) and `imul` (`Cpu::imul_truncated`) leave, which the interpreter
    /// materializes eagerly: only the JIT produces this op.
    #[cfg_attr(
        not(all(
            feature = "jit",
            any(
                target_arch = "aarch64",
                target_arch = "x86_64",
                target_arch = "wasm32"
            )
        )),
        allow(dead_code)
    )]
    SzpCfOf,
}

/// Bit positions inside `lazy_src` for `LazyFlagsOp::SzpCfOf`.
pub(crate) const SZP_CF_OF_SRC_CF_BIT: u32 = 0;
pub(crate) const SZP_CF_OF_SRC_OF_BIT: u32 = 1;

/// Value mask for an operand `size` in bytes; anything other than 1, 2 or 4 means 8.
pub(crate) fn mask(size: u8) -> u64 {
    match size {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

/// Sign bit of an operand `size` bytes wide.
pub(crate) fn sign(size: u8) -> u64 {
    1u64 << (size as u32 * 8 - 1)
}

/// `res` must already be truncated to `size`. PF looks at the low byte only, whatever the width.
fn szp(res: u64, size: u8) -> u64 {
    let mut f = 0;
    if res == 0 {
        f |= ZF;
    }
    if res & sign(size) != 0 {
        f |= SF;
    }
    if (res as u8).count_ones() & 1 == 0 {
        f |= PF;
    }
    f
}

/// AF is the carry out of bit 3, so INC/DEC derive it from the low nibble of the result.
const NIBBLE_MASK: u64 = 0xf;

/// Materializes the `ARITH` flags for a deferred operation; the single source of truth that
/// the interpreter and the JIT (via helper calls) share. `eflags` is only read for `Eflags`.
pub(crate) fn compute(
    op: LazyFlagsOp,
    size: u8,
    dst: u64,
    src: u64,
    src2: u64,
    eflags: u64,
) -> u64 {
    let width_mask = mask(size);
    let result = dst & width_mask;
    let b = src & width_mask;
    let sign_bit = sign(size);
    match op {
        LazyFlagsOp::Eflags => eflags & ARITH,
        LazyFlagsOp::Add | LazyFlagsOp::Adc => {
            let carry_in = if op == LazyFlagsOp::Adc { src2 & 1 } else { 0 };
            // first operand, recovered from the result
            let a = result.wrapping_sub(b).wrapping_sub(carry_in) & width_mask;
            // with a carry-in, a + b + 1 wraps to exactly `b` when a == !0, hence <= and not <
            let carry = if carry_in != 0 {
                result <= b
            } else {
                result < b
            };
            let mut f = szp(result, size);
            f |= carry as u64;
            f |= (a ^ b ^ result) & AF;
            if (!(a ^ b) & (a ^ result)) & sign_bit != 0 {
                f |= OF;
            }
            f
        }
        LazyFlagsOp::Sub | LazyFlagsOp::Sbb => {
            let carry_in = if op == LazyFlagsOp::Sbb { src2 & 1 } else { 0 };
            let a = result.wrapping_add(b).wrapping_add(carry_in) & width_mask;
            let borrow = if carry_in != 0 { a <= b } else { a < b };
            let mut f = szp(result, size);
            f |= borrow as u64;
            f |= (a ^ b ^ result) & AF;
            if ((a ^ b) & (a ^ result)) & sign_bit != 0 {
                f |= OF;
            }
            f
        }
        // CF and OF cleared; AF is architecturally undefined and left clear.
        LazyFlagsOp::Logic => szp(result, size),
        // INC/DEC leave CF alone: it rides along in bit 0 of `src`.
        LazyFlagsOp::Inc => {
            let mut f = szp(result, size) | (src & 1);
            if result & NIBBLE_MASK == 0 {
                f |= AF;
            }
            if result == sign_bit {
                f |= OF;
            }
            f
        }
        LazyFlagsOp::SzpCfOf => {
            let mut f = szp(result, size);
            if src >> SZP_CF_OF_SRC_CF_BIT & 1 != 0 {
                f |= CF;
            }
            if src >> SZP_CF_OF_SRC_OF_BIT & 1 != 0 {
                f |= OF;
            }
            f
        }
        LazyFlagsOp::Dec => {
            let mut f = szp(result, size) | (src & 1);
            if result & NIBBLE_MASK == NIBBLE_MASK {
                f |= AF;
            }
            if result == sign_bit.wrapping_sub(1) & width_mask {
                f |= OF;
            }
            f
        }
    }
}

/// SF/ZF/PF of `res` truncated to `size`, for instructions that set only those.
pub(crate) fn szp_flags(res: u64, size: u8) -> u64 {
    szp(res & mask(size), size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{Harness, RAX, assert_jit_matches_interpreter};

    const SIZES: [u8; 4] = [1, 2, 4, 8];
    const NOISE_ABOVE_RESULT: u64 = 0xdead_beef_cafe_f00d;
    /// Operand values around every width's sign, carry and nibble boundaries.
    const INTERESTING: [u64; 24] = [
        0,
        1,
        2,
        0x7,
        0x8,
        0xf,
        0x10,
        0x11,
        0x3f,
        0x7f,
        0x80,
        0x81,
        0xff,
        0x7fff,
        0x8000,
        0xffff,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        0xffff_ffff_ffff_fffe,
        u64::MAX,
        0x1234_5678_9abc_def0,
    ];

    fn truncate(value: u64, size: u8) -> u64 {
        match size {
            1 => value & 0xff,
            2 => value & 0xffff,
            4 => value & 0xffff_ffff,
            _ => value,
        }
    }

    fn sign_extend(value: u64, size: u8) -> i128 {
        let bits = size as u32 * 8;
        let shift = 128 - bits;
        (((value as u128) << shift) as i128) >> shift
    }

    /// ZF, SF and PF of a result, computed bit by bit.
    fn result_flags(result: u64, size: u8) -> u64 {
        let bits = size as u32 * 8;
        let mut flags = 0;
        if result == 0 {
            flags |= ZF;
        }
        if (result >> (bits - 1)) & 1 == 1 {
            flags |= SF;
        }
        let ones = (0..8).filter(|bit| (result >> bit) & 1 == 1).count();
        if ones % 2 == 0 {
            flags |= PF;
        }
        flags
    }

    /// `a + b + carry` with the flags the SDM defines for ADD and ADC, derived with wide
    /// integers instead of the bit tricks `compute` uses.
    fn reference_add(a: u64, b: u64, carry: u64, size: u8) -> (u64, u64) {
        let (a, b) = (truncate(a, size), truncate(b, size));
        let result = truncate(a.wrapping_add(b).wrapping_add(carry), size);
        let mut flags = result_flags(result, size);
        if (a as u128 + b as u128 + carry as u128) >> (size as u32 * 8) != 0 {
            flags |= CF;
        }
        if (a & 0xf) + (b & 0xf) + carry > 0xf {
            flags |= AF;
        }
        if sign_extend(a, size) + sign_extend(b, size) + carry as i128 != sign_extend(result, size)
        {
            flags |= OF;
        }
        (result, flags)
    }

    /// `a - b - borrow` with the flags the SDM defines for SUB, SBB and CMP.
    fn reference_sub(a: u64, b: u64, borrow: u64, size: u8) -> (u64, u64) {
        let (a, b) = (truncate(a, size), truncate(b, size));
        let result = truncate(a.wrapping_sub(b).wrapping_sub(borrow), size);
        let mut flags = result_flags(result, size);
        if (a as u128) < b as u128 + borrow as u128 {
            flags |= CF;
        }
        if (a & 0xf) < (b & 0xf) + borrow {
            flags |= AF;
        }
        if sign_extend(a, size) - sign_extend(b, size) - borrow as i128 != sign_extend(result, size)
        {
            flags |= OF;
        }
        (result, flags)
    }

    #[test]
    fn sub_flags() {
        // 0 - 1 (8-bit): borrow
        let f = compute(LazyFlagsOp::Sub, 1, 0xff, 1, 0, 0);
        assert_ne!(f & (CF | SF), 0);
        // 5 - 1: no borrow
        let f = compute(LazyFlagsOp::Sub, 4, 4, 1, 0, 0);
        assert_eq!(f & (CF | ZF | OF), 0);
        // 0x80 - 1 (8-bit): signed overflow
        let f = compute(LazyFlagsOp::Sub, 1, 0x7f, 1, 0, 0);
        assert_ne!(f & OF, 0);
        // 1 - 1
        let f = compute(LazyFlagsOp::Sub, 4, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), ZF);
    }

    #[test]
    fn add_flags() {
        let f = compute(LazyFlagsOp::Add, 1, 0, 1, 0, 0);
        assert_eq!(f & (CF | ZF), CF | ZF);
        let f = compute(LazyFlagsOp::Add, 1, 0x80, 0x7f, 0, 0);
        assert_ne!(f & OF, 0);
        let f = compute(LazyFlagsOp::Adc, 1, 0, 0xff, 1, 0);
        assert_ne!(f & CF, 0);
    }

    /// What `compute` is given in `lazy_dst`: the result, with junk above its width.
    fn dirty(value: u64, size: u8) -> u64 {
        if size == 8 {
            value
        } else {
            value | (NOISE_ABOVE_RESULT << (size as u32 * 8))
        }
    }

    fn describe(flags: u64) -> String {
        let names = [
            (CF, "CF"),
            (PF, "PF"),
            (AF, "AF"),
            (ZF, "ZF"),
            (SF, "SF"),
            (OF, "OF"),
        ];
        let set: Vec<&str> = names
            .iter()
            .filter(|(bit, _)| flags & bit != 0)
            .map(|(_, name)| *name)
            .collect();
        format!("{:#x} [{}]", flags, set.join("|"))
    }

    mod helpers {
        use super::*;

        #[test]
        fn mask_covers_exactly_the_operand_width() {
            assert_eq!(mask(1), 0xff);
            assert_eq!(mask(2), 0xffff);
            assert_eq!(mask(4), 0xffff_ffff);
            assert_eq!(mask(8), u64::MAX);
        }

        #[test]
        fn mask_treats_unknown_sizes_as_64_bit() {
            for size in [0, 3, 5, 16] {
                assert_eq!(mask(size), u64::MAX, "size {size}");
            }
        }

        #[test]
        fn sign_is_the_top_bit_of_the_width() {
            assert_eq!(sign(1), 0x80);
            assert_eq!(sign(2), 0x8000);
            assert_eq!(sign(4), 0x8000_0000);
            assert_eq!(sign(8), 1 << 63);
        }

        #[test]
        fn status_flag_constants_are_the_architectural_bit_positions() {
            assert_eq!(CF, 0x1);
            assert_eq!(PF, 0x4);
            assert_eq!(AF, 0x10);
            assert_eq!(ZF, 0x40);
            assert_eq!(SF, 0x80);
            assert_eq!(TF, 0x100);
            assert_eq!(IF, 0x200);
            assert_eq!(DF, 0x400);
            assert_eq!(OF, 0x800);
            assert_eq!(IOPL, 0x3000);
            assert_eq!(NT, 0x4000);
            assert_eq!(RF, 0x1_0000);
            assert_eq!(VM, 0x2_0000);
            assert_eq!(AC, 0x4_0000);
            assert_eq!(RESERVED_ONE, 0x2);
            assert_eq!(ARITH, 0x8d5);
        }

        #[test]
        fn lazy_op_is_one_byte_and_defaults_to_materialized() {
            assert_eq!(core::mem::size_of::<LazyFlagsOp>(), 1);
            assert_eq!(LazyFlagsOp::default(), LazyFlagsOp::Eflags);
        }
    }

    mod parity_and_szp {
        use super::*;

        #[test]
        fn parity_flag_is_set_for_every_byte_with_an_even_number_of_ones() {
            for byte in 0..=255u64 {
                let flags = szp_flags(byte, 1);
                assert_eq!(
                    flags & PF != 0,
                    byte.count_ones() % 2 == 0,
                    "byte {byte:#04x}"
                );
            }
        }

        #[test]
        fn parity_looks_only_at_the_low_byte_whatever_the_width() {
            for size in [2u8, 4, 8] {
                for byte in 0..=255u64 {
                    for high in [0u64, 0x01, 0xff, 0x5a] {
                        let value = truncate((high << 8) | byte, size);
                        assert_eq!(
                            szp_flags(value, size) & PF != 0,
                            byte.count_ones() % 2 == 0,
                            "size {size} value {value:#x}"
                        );
                    }
                }
            }
        }

        #[test]
        fn szp_flags_sets_zero_only_for_a_zero_result() {
            for size in SIZES {
                assert_eq!(szp_flags(0, size), ZF | PF, "size {size}");
                assert_eq!(szp_flags(1, size) & ZF, 0, "size {size}");
            }
            // A set bit above the operand width is truncated away first.
            assert_eq!(szp_flags(0x100, 1), ZF | PF);
            assert_eq!(szp_flags(0x1_0000, 2), ZF | PF);
            assert_eq!(szp_flags(0x1_0000_0000, 4), ZF | PF);
        }

        #[test]
        fn szp_flags_sets_sign_from_the_top_bit_of_the_width() {
            assert_eq!(szp_flags(0x80, 1) & SF, SF);
            assert_eq!(szp_flags(0x80, 2) & SF, 0);
            assert_eq!(szp_flags(0x8000, 2) & SF, SF);
            assert_eq!(szp_flags(0x8000, 4) & SF, 0);
            assert_eq!(szp_flags(0x8000_0000, 4) & SF, SF);
            assert_eq!(szp_flags(0x8000_0000, 8) & SF, 0);
            assert_eq!(szp_flags(1 << 63, 8) & SF, SF);
            assert_eq!(szp_flags(0x7f, 1) & SF, 0);
        }

        #[test]
        fn szp_flags_never_reports_carry_overflow_or_adjust() {
            for size in SIZES {
                for value in INTERESTING {
                    assert_eq!(szp_flags(value, size) & (CF | AF | OF), 0);
                }
            }
        }

        #[test]
        fn szp_flags_agrees_with_the_bitwise_reference() {
            for size in SIZES {
                for value in INTERESTING {
                    let truncated = truncate(value, size);
                    assert_eq!(
                        szp_flags(value, size),
                        result_flags(truncated, size),
                        "size {size} value {value:#x}"
                    );
                }
            }
        }
    }

    mod compute_known_vectors {
        use super::*;

        #[test]
        fn eflags_op_returns_only_the_arithmetic_bits_of_the_given_eflags() {
            assert_eq!(compute(LazyFlagsOp::Eflags, 4, 1, 2, 3, u64::MAX), ARITH);
            assert_eq!(
                compute(
                    LazyFlagsOp::Eflags,
                    8,
                    0,
                    0,
                    0,
                    IF | DF | CF | ZF | RESERVED_ONE
                ),
                CF | ZF
            );
            assert_eq!(compute(LazyFlagsOp::Eflags, 1, 0, 0, 0, 0), 0);
        }

        #[test]
        fn add_0x7f_plus_1_in_8_bits_overflows_into_the_sign() {
            // 0x7f + 1 = 0x80: SF, signed overflow, carry out of bit 3. 0x80 has one bit: PF clear.
            assert_eq!(compute(LazyFlagsOp::Add, 1, 0x80, 1, 0, 0), SF | OF | AF);
        }

        #[test]
        fn add_0xff_plus_1_in_8_bits_wraps_to_zero() {
            assert_eq!(compute(LazyFlagsOp::Add, 1, 0, 1, 0, 0), CF | ZF | PF | AF);
        }

        #[test]
        fn add_0x7fffffff_plus_1_in_32_bits() {
            assert_eq!(
                compute(LazyFlagsOp::Add, 4, 0x8000_0000, 1, 0, 0),
                SF | OF | AF | PF
            );
        }

        #[test]
        fn add_max_plus_1_in_64_bits_wraps_to_zero() {
            assert_eq!(compute(LazyFlagsOp::Add, 8, 0, 1, 0, 0), CF | ZF | PF | AF);
        }

        #[test]
        fn add_of_two_negatives_without_unsigned_overflow_is_not_flagged_as_carry() {
            // 0x80 + 0x80 = 0x100: carry and overflow, zero low byte.
            assert_eq!(
                compute(LazyFlagsOp::Add, 1, 0, 0x80, 0, 0),
                CF | OF | ZF | PF
            );
            // 0x40 + 0x40 = 0x80: overflow only.
            assert_eq!(compute(LazyFlagsOp::Add, 1, 0x80, 0x40, 0, 0), OF | SF);
        }

        #[test]
        fn adc_0xff_plus_0xff_plus_carry_in_8_bits() {
            // 0xff + 0xff + 1 = 0x1ff: low byte 0xff.
            assert_eq!(
                compute(LazyFlagsOp::Adc, 1, 0xff, 0xff, 1, 0),
                CF | AF | SF | PF
            );
        }

        #[test]
        fn adc_with_carry_in_makes_0x7f_plus_0_overflow() {
            assert_eq!(compute(LazyFlagsOp::Adc, 1, 0x80, 0, 1, 0), SF | OF | AF);
        }

        #[test]
        fn adc_without_carry_in_behaves_like_add() {
            assert_eq!(
                compute(LazyFlagsOp::Adc, 1, 0, 0xff, 0, 0),
                compute(LazyFlagsOp::Add, 1, 0, 0xff, 0, 0)
            );
        }

        #[test]
        fn adc_carry_out_when_the_first_operand_is_all_ones() {
            // a = 0xff, b = 0, carry-in 1: 0xff + 0 + 1 wraps to 0 with a carry.
            assert_eq!(compute(LazyFlagsOp::Adc, 1, 0, 0, 1, 0), CF | ZF | PF | AF);
        }

        #[test]
        fn adc_uses_only_bit_zero_of_the_carry_operand() {
            assert_eq!(
                compute(LazyFlagsOp::Adc, 1, 0x05, 0x04, 0xfe, 0),
                compute(LazyFlagsOp::Adc, 1, 0x05, 0x04, 0, 0)
            );
            assert_eq!(
                compute(LazyFlagsOp::Adc, 1, 0x06, 0x04, 0xff, 0),
                compute(LazyFlagsOp::Adc, 1, 0x06, 0x04, 1, 0)
            );
        }

        #[test]
        fn sub_0_minus_1_in_8_bits_borrows() {
            assert_eq!(
                compute(LazyFlagsOp::Sub, 1, 0xff, 1, 0, 0),
                CF | AF | SF | PF
            );
        }

        #[test]
        fn sub_0x80_minus_1_in_8_bits_overflows() {
            assert_eq!(compute(LazyFlagsOp::Sub, 1, 0x7f, 1, 0, 0), OF | AF);
        }

        #[test]
        fn sub_equal_operands_is_zero_with_no_borrow() {
            for size in SIZES {
                assert_eq!(
                    compute(LazyFlagsOp::Sub, size, 0, 0x1234, 0, 0),
                    ZF | PF,
                    "size {size}"
                );
            }
        }

        #[test]
        fn sub_0x8000_minus_1_in_16_bits() {
            assert_eq!(compute(LazyFlagsOp::Sub, 2, 0x7fff, 1, 0, 0), OF | AF | PF);
        }

        #[test]
        fn sub_of_a_larger_from_a_smaller_unsigned_value_sets_carry_without_overflow() {
            // 1 - 2 = -1 (0xffffffff): borrow, sign, no overflow.
            assert_eq!(
                compute(LazyFlagsOp::Sub, 4, 0xffff_ffff, 2, 0, 0),
                CF | AF | SF | PF
            );
        }

        #[test]
        fn sbb_0_minus_0xff_minus_borrow_in_8_bits() {
            assert_eq!(
                compute(LazyFlagsOp::Sbb, 1, 0, 0xff, 1, 0),
                CF | AF | ZF | PF
            );
        }

        #[test]
        fn sbb_of_equal_operands_with_borrow_in_borrows() {
            // 0xff - 0xff - 1 = 0xff.
            assert_eq!(
                compute(LazyFlagsOp::Sbb, 1, 0xff, 0xff, 1, 0),
                CF | AF | SF | PF
            );
        }

        #[test]
        fn sbb_without_borrow_in_behaves_like_sub() {
            assert_eq!(
                compute(LazyFlagsOp::Sbb, 4, 0x7fff_ffff, 0xffff_ffff, 0, 0),
                compute(LazyFlagsOp::Sub, 4, 0x7fff_ffff, 0xffff_ffff, 0, 0)
            );
        }

        #[test]
        fn sbb_uses_only_bit_zero_of_the_borrow_operand() {
            assert_eq!(
                compute(LazyFlagsOp::Sbb, 1, 0x10, 0x01, 0xfe, 0),
                compute(LazyFlagsOp::Sbb, 1, 0x10, 0x01, 0, 0)
            );
        }

        #[test]
        fn logic_clears_carry_overflow_and_adjust() {
            for size in SIZES {
                assert_eq!(
                    compute(LazyFlagsOp::Logic, size, 0, 0xff, 1, u64::MAX),
                    ZF | PF
                );
                // The sign bit alone: the low byte is 0x80 (odd parity) only for byte operands.
                let parity = if size == 1 { 0 } else { PF };
                assert_eq!(
                    compute(LazyFlagsOp::Logic, size, sign(size), 0xff, 1, u64::MAX),
                    SF | parity,
                    "size {size}"
                );
            }
        }

        #[test]
        fn logic_ignores_the_second_operand_and_junk_above_the_width() {
            assert_eq!(
                compute(LazyFlagsOp::Logic, 1, dirty(0x81, 1), 0xffff, 0xff, 0),
                SF | PF
            );
        }

        #[test]
        fn szp_cf_of_takes_cf_and_of_from_src_and_the_rest_from_the_result() {
            assert_eq!(compute(LazyFlagsOp::SzpCfOf, 4, 0, 0, 0, 0), ZF | PF);
            assert_eq!(compute(LazyFlagsOp::SzpCfOf, 4, 0, 1, 0, 0), ZF | PF | CF);
            assert_eq!(compute(LazyFlagsOp::SzpCfOf, 4, 0, 2, 0, 0), ZF | PF | OF);
            assert_eq!(
                compute(LazyFlagsOp::SzpCfOf, 1, dirty(0x80, 1), 3, 0, 0),
                SF | CF | OF
            );
            // AF stays clear whatever the result.
            assert_eq!(compute(LazyFlagsOp::SzpCfOf, 1, 0x10, 0, 0, 0) & AF, 0);
        }

        #[test]
        fn inc_keeps_the_carry_given_in_bit_zero_of_src() {
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0x02, 1, 0, 0), CF);
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0x02, 0, 0, 0), 0);
        }

        #[test]
        fn inc_of_the_largest_positive_value_overflows() {
            assert_eq!(
                compute(LazyFlagsOp::Inc, 1, 0x80, 1, 0, 0),
                CF | AF | SF | OF
            );
            assert_eq!(
                compute(LazyFlagsOp::Inc, 8, 1 << 63, 0, 0, 0),
                AF | SF | OF | PF
            );
        }

        #[test]
        fn inc_of_all_ones_wraps_to_zero_without_touching_carry() {
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0, 0, 0, 0), ZF | PF | AF);
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0, 1, 0, 0), CF | ZF | PF | AF);
        }

        #[test]
        fn inc_adjust_flag_follows_the_low_nibble_wrap() {
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0x10, 0, 0, 0) & AF, AF);
            assert_eq!(compute(LazyFlagsOp::Inc, 1, 0x0f, 0, 0, 0) & AF, 0);
        }

        #[test]
        fn dec_of_the_most_negative_value_overflows() {
            assert_eq!(compute(LazyFlagsOp::Dec, 1, 0x7f, 0, 0, 0), OF | AF);
            assert_eq!(compute(LazyFlagsOp::Dec, 1, 0x7f, 1, 0, 0), CF | OF | AF);
        }

        #[test]
        fn dec_of_one_is_zero_without_adjust() {
            assert_eq!(compute(LazyFlagsOp::Dec, 1, 0, 0, 0, 0), ZF | PF);
        }

        #[test]
        fn dec_of_zero_borrows_through_the_nibble_but_not_into_carry() {
            assert_eq!(compute(LazyFlagsOp::Dec, 1, 0xff, 0, 0, 0), SF | PF | AF);
            assert_eq!(
                compute(LazyFlagsOp::Dec, 1, 0xff, 1, 0, 0),
                CF | SF | PF | AF
            );
        }

        #[test]
        fn inc_and_dec_use_only_bit_zero_of_src_for_carry() {
            assert_eq!(
                compute(LazyFlagsOp::Inc, 2, 5, 0xfe, 0, 0),
                compute(LazyFlagsOp::Inc, 2, 5, 0, 0, 0)
            );
            assert_eq!(
                compute(LazyFlagsOp::Dec, 2, 5, 0xff, 0, 0),
                compute(LazyFlagsOp::Dec, 2, 5, 1, 0, 0)
            );
        }
    }

    mod compute_against_reference {
        use super::*;

        fn check(op: LazyFlagsOp, size: u8, a: u64, b: u64, carry: u64) {
            let (result, expected) = match op {
                LazyFlagsOp::Add | LazyFlagsOp::Adc => reference_add(a, b, carry, size),
                _ => reference_sub(a, b, carry, size),
            };
            let actual = compute(
                op,
                size,
                dirty(result, size),
                dirty(b, size),
                carry | 0xfe,
                u64::MAX,
            );
            assert_eq!(
                actual,
                expected,
                "{op:?} size {size}: {a:#x}, {b:#x}, carry {carry}: got {}, want {}",
                describe(actual),
                describe(expected)
            );
        }

        #[test]
        fn add_matches_the_reference_for_every_width_and_operand_pair() {
            for size in SIZES {
                for a in INTERESTING {
                    for b in INTERESTING {
                        check(LazyFlagsOp::Add, size, a, b, 0);
                    }
                }
            }
        }

        #[test]
        fn adc_matches_the_reference_for_every_width_operand_pair_and_carry() {
            for size in SIZES {
                for a in INTERESTING {
                    for b in INTERESTING {
                        for carry in [0, 1] {
                            check(LazyFlagsOp::Adc, size, a, b, carry);
                        }
                    }
                }
            }
        }

        #[test]
        fn sub_matches_the_reference_for_every_width_and_operand_pair() {
            for size in SIZES {
                for a in INTERESTING {
                    for b in INTERESTING {
                        check(LazyFlagsOp::Sub, size, a, b, 0);
                    }
                }
            }
        }

        #[test]
        fn sbb_matches_the_reference_for_every_width_operand_pair_and_borrow() {
            for size in SIZES {
                for a in INTERESTING {
                    for b in INTERESTING {
                        for borrow in [0, 1] {
                            check(LazyFlagsOp::Sbb, size, a, b, borrow);
                        }
                    }
                }
            }
        }

        #[test]
        fn logic_matches_the_reference_for_every_width_and_value() {
            for size in SIZES {
                for value in INTERESTING {
                    let result = truncate(value, size);
                    let actual = compute(
                        LazyFlagsOp::Logic,
                        size,
                        dirty(result, size),
                        NOISE_ABOVE_RESULT,
                        u64::MAX,
                        u64::MAX,
                    );
                    assert_eq!(actual, result_flags(result, size), "size {size} {value:#x}");
                }
            }
        }

        #[test]
        fn inc_matches_add_one_with_the_carry_flag_preserved() {
            for size in SIZES {
                for a in INTERESTING {
                    for carry_in in [0u64, 1] {
                        let (result, add_flags) = reference_add(a, 1, 0, size);
                        let expected = (add_flags & !CF) | carry_in;
                        let actual = compute(
                            LazyFlagsOp::Inc,
                            size,
                            dirty(result, size),
                            carry_in | 0xfe,
                            0,
                            u64::MAX,
                        );
                        assert_eq!(
                            actual,
                            expected,
                            "inc size {size} {a:#x} cf {carry_in}: got {}, want {}",
                            describe(actual),
                            describe(expected)
                        );
                    }
                }
            }
        }

        #[test]
        fn dec_matches_subtract_one_with_the_carry_flag_preserved() {
            for size in SIZES {
                for a in INTERESTING {
                    for carry_in in [0u64, 1] {
                        let (result, sub_flags) = reference_sub(a, 1, 0, size);
                        let expected = (sub_flags & !CF) | carry_in;
                        let actual = compute(
                            LazyFlagsOp::Dec,
                            size,
                            dirty(result, size),
                            carry_in | 0xfe,
                            0,
                            u64::MAX,
                        );
                        assert_eq!(
                            actual,
                            expected,
                            "dec size {size} {a:#x} cf {carry_in}: got {}, want {}",
                            describe(actual),
                            describe(expected)
                        );
                    }
                }
            }
        }

        #[test]
        fn results_only_ever_contain_the_six_status_flags() {
            for op in [
                LazyFlagsOp::Add,
                LazyFlagsOp::Adc,
                LazyFlagsOp::Sub,
                LazyFlagsOp::Sbb,
                LazyFlagsOp::Logic,
                LazyFlagsOp::Inc,
                LazyFlagsOp::Dec,
            ] {
                for size in SIZES {
                    for value in INTERESTING {
                        let flags = compute(op, size, value, value ^ 0x55, 1, u64::MAX);
                        assert_eq!(flags & !ARITH, 0, "{op:?} size {size}");
                    }
                }
            }
        }

        #[test]
        fn compare_is_a_subtract_so_equal_values_set_only_zero_and_parity() {
            for size in SIZES {
                for value in INTERESTING {
                    let (result, expected) = reference_sub(value, value, 0, size);
                    assert_eq!(result, 0);
                    assert_eq!(expected, ZF | PF);
                }
            }
        }
    }

    mod through_instructions {
        use super::*;

        fn flags_after(code: &[u8]) -> u64 {
            let mut harness = Harness::new(code);
            harness.run_to_halt();
            harness.cpu.gpr[RAX] & ARITH
        }

        fn check(code: &[u8], expected: u64) {
            let actual = flags_after(code);
            assert_eq!(
                actual,
                expected,
                "{} vs {}",
                describe(actual),
                describe(expected)
            );
            assert_jit_matches_interpreter(code, |_| {});
        }

        #[test]
        fn add_al_overflowing_into_the_sign() {
            // mov al, 0x7f; add al, 1; pushfq; pop rax; hlt
            check(&[0xb0, 0x7f, 0x04, 0x01, 0x9c, 0x58, 0xf4], SF | OF | AF);
        }

        #[test]
        fn adc_al_with_carry_in_wraps_to_zero() {
            // stc; mov al, 0xff; adc al, 0; pushfq; pop rax; hlt
            check(
                &[0xf9, 0xb0, 0xff, 0x14, 0x00, 0x9c, 0x58, 0xf4],
                CF | ZF | PF | AF,
            );
        }

        #[test]
        fn sbb_al_with_borrow_in() {
            // stc; mov al, 0; sbb al, 0xff; pushfq; pop rax; hlt
            check(
                &[0xf9, 0xb0, 0x00, 0x1c, 0xff, 0x9c, 0x58, 0xf4],
                CF | AF | ZF | PF,
            );
        }

        #[test]
        fn sbb_al_without_borrow_in() {
            // clc; mov al, 0; sbb al, 0xff; pushfq; pop rax; hlt
            check(&[0xf8, 0xb0, 0x00, 0x1c, 0xff, 0x9c, 0x58, 0xf4], CF | AF);
        }

        #[test]
        fn inc_al_keeps_the_carry_flag() {
            // stc; mov al, 0x7f; inc al; pushfq; pop rax; hlt
            check(
                &[0xf9, 0xb0, 0x7f, 0xfe, 0xc0, 0x9c, 0x58, 0xf4],
                CF | AF | SF | OF,
            );
        }

        #[test]
        fn dec_al_leaves_carry_clear() {
            // clc; mov al, 0x80; dec al; pushfq; pop rax; hlt
            check(&[0xf8, 0xb0, 0x80, 0xfe, 0xc8, 0x9c, 0x58, 0xf4], OF | AF);
        }

        #[test]
        fn and_clears_carry_overflow_and_adjust() {
            // mov al, 0xf0; and al, 0x0f; pushfq; pop rax; hlt
            check(&[0xb0, 0xf0, 0x24, 0x0f, 0x9c, 0x58, 0xf4], ZF | PF);
        }

        #[test]
        fn cmp_eax_of_the_most_negative_value_with_one() {
            // mov eax, 0x80000000; cmp eax, 1; pushfq; pop rax; hlt
            check(
                &[
                    0xb8, 0x00, 0x00, 0x00, 0x80, 0x83, 0xf8, 0x01, 0x9c, 0x58, 0xf4,
                ],
                OF | AF | PF,
            );
        }

        #[test]
        fn add_rax_to_itself_for_the_most_negative_value() {
            // mov rax, 0x8000000000000000; add rax, rax; pushfq; pop rax; hlt
            check(
                &[
                    0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48, 0x01, 0xc0,
                    0x9c, 0x58, 0xf4,
                ],
                CF | OF | ZF | PF,
            );
        }
    }
}
