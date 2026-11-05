//! Minimal aarch64 assembler: just the instruction forms `emit.rs` and the prologue in
//! `arch_a64.rs` use. Each opcode literal is the fixed bits of the instruction with every
//! register and immediate field zero; the fields are or-ed in at the `*_SHIFT` positions below.
//! `is64 = false` selects the W-register (32-bit) form by leaving `SF` clear.
//!
//! Register arguments are plain numbers 0..=31. Number 31 is `xzr` or `sp` depending on the
//! instruction, which is why `mov_sp` exists next to `mov`.

#![allow(dead_code)]

pub(crate) const NOP_WORD: u32 = 0xd503_201f;

pub(crate) const XZR: u32 = 31;
pub(crate) const SP: u32 = 31;

/// A64 condition field values. Flipping bit 0 inverts a condition (`cset` relies on this).
pub(crate) const EQ: u32 = 0;
pub(crate) const NE: u32 = 1;
pub(crate) const CS: u32 = 2;
pub(crate) const CC: u32 = 3;
pub(crate) const MI: u32 = 4;
pub(crate) const PL: u32 = 5;
pub(crate) const VS: u32 = 6;
pub(crate) const VC: u32 = 7;
pub(crate) const HI: u32 = 8;
pub(crate) const LS: u32 = 9;
pub(crate) const GE: u32 = 10;
pub(crate) const LT: u32 = 11;
pub(crate) const GT: u32 = 12;
pub(crate) const LE: u32 = 13;

/// Instruction-word field positions.
const RN_SHIFT: u32 = 5;
const RM_SHIFT: u32 = 16;
/// Bit 31: 64-bit (X register) form.
const SF: u32 = 1 << 31;
const COND_SHIFT: u32 = 12;
const IMM12_SHIFT: u32 = 10;
const IMM12_MASK: u32 = 0xfff;
/// Bytes in a 128-bit vector register; the scale of `ldr_q`/`str_q` offsets.
const VECTOR_BYTES: u32 = 16;
const VEC_SIZE_SHIFT: u32 = 22;
/// immh:immb of the vector shifts, and imm5 of `umov`.
const VEC_SHIFT_IMMEDIATE_SHIFT: u32 = 16;

/// Bases of the three-register vector forms `Asm::vec_op` takes (Q = 1, 128-bit).
pub const VEC_AND: u32 = 0x4e20_1c00;
pub const VEC_BIC: u32 = 0x4e60_1c00;
pub const VEC_ORR: u32 = 0x4ea0_1c00;
pub const VEC_EOR: u32 = 0x6e20_1c00;
pub const VEC_CMEQ: u32 = 0x6e20_8c00;
pub const VEC_UMIN: u32 = 0x6e20_6c00;
pub const VEC_UMAX: u32 = 0x6e20_6400;
pub const VEC_ADD: u32 = 0x4e20_8400;
pub const VEC_SUB: u32 = 0x6e20_8400;
/// Lanes become all ones where `vn & vm` is not zero.
pub const VEC_CMTST: u32 = 0x4e20_8c00;
/// `not vd.16b, vn.16b`: `vm` is zero.
pub const VEC_NOT: u32 = 0x6e20_5800;
/// `bsl vd, vn, vm`: vd = (vd & vn) | (!vd & vm), so vd is the select mask.
pub const VEC_BSL: u32 = 0x6e60_1c00;
/// Unsigned and signed byte or wider lane compares: vn > vm (`CMHI`, `CMGT`) and vn >= vm
/// (`CMHS`, `CMGE`).
pub const VEC_CMHI: u32 = 0x6e20_3400;
pub const VEC_CMHS: u32 = 0x6e20_3c00;
pub const VEC_CMGT: u32 = 0x4e20_3400;
pub const VEC_CMGE: u32 = 0x4e20_3c00;
/// Shift amount of the shifted-register add.
const IMM6_SHIFT: u32 = 10;
/// MOVZ/MOVK/MOVN: 16-bit immediate, and which halfword it replaces (`hw` * 16 bits).
const IMM16_SHIFT: u32 = 5;
const HW_SHIFT: u32 = 21;
/// Bitfield moves: rotate amount and index of the last bit of the field.
const IMMR_SHIFT: u32 = 16;
const IMMS_SHIFT: u32 = 10;
/// N:immr:imms of a logical (bitmask) immediate sits here as one 13-bit field.
const BITMASK_IMM_SHIFT: u32 = 10;
/// Load/store pair: second register and the signed 7-bit offset, scaled by 8.
const RT2_SHIFT: u32 = 10;
const IMM7_SHIFT: u32 = 15;
const IMM7_MASK: u32 = 0x7f;
const PAIR_OFFSET_SCALE: i32 = 8;

/// PC-relative branch immediates count instructions, not bytes.
const IMM26_MASK: u32 = 0x03ff_ffff;
const IMM19_MASK: u32 = 0x7_ffff;
const IMM19_SHIFT: u32 = 5;
const IMM26_SIGNED_LIMIT: i64 = 1 << 25;

/// Unsigned 12-bit immediates of add/sub and of the scaled load/store offset.
pub(crate) const IMM12_LIMIT: u32 = 1 << 12;
const XREG_BITS: u32 = 64;
const WREG_BITS: u32 = 32;

/// Index into `Asm::labels`.
pub(crate) type Label = usize;

/// Which immediate field of a branch `Asm::finish` patches once the label's position is known.
#[derive(Clone, Copy)]
enum Fixup {
    /// `b`: 26-bit offset from bit 0.
    Imm26,
    /// `b.cond`, `cbz`, `cbnz`, `ldr` (literal): 19-bit offset from bit 5.
    Imm19,
    /// `adr`: a 21-bit byte offset split over bits 5..24 (immhi) and 29..30 (immlo). Labels are
    /// word aligned, so immlo is always zero.
    Adr,
}

pub(crate) struct Asm {
    pub code: Vec<u32>,
    /// Address the code will run at. Must be where the words are finally written, because
    /// `here` and `b_abs` compute pc-relative offsets against it.
    base: u64,
    /// Position (in words) each label was bound to.
    labels: Vec<Option<usize>>,
    /// Branches emitted before their target was bound: (word index, label, field to patch).
    fixes: Vec<(usize, Label, Fixup)>,
    /// 64-bit constants to place after the code (see `literal`): the label bound to the first of
    /// a run (if any) and the value.
    pool: Vec<(Option<Label>, u64)>,
}

/// Encodes `val` as a logical immediate (the 13-bit N:immr:imms field) for `and`/`orr`/`eor`/
/// `ands`, or `None` if it is not representable. Representable values are a run of ones,
/// rotated, replicated across the register in 2, 4, ..., 64-bit elements. Zero and all-ones
/// are never encodable. For `!is64` only the low 32 bits count.
pub(crate) fn encode_bitmask(val: u64, is64: bool) -> Option<u32> {
    let register_bits: u32 = if is64 { XREG_BITS } else { WREG_BITS };
    let mut pattern = if is64 { val } else { val & 0xffff_ffff };
    if pattern == 0 || pattern == if is64 { u64::MAX } else { 0xffff_ffff } {
        return None;
    }
    if !is64 {
        pattern |= pattern << 32;
    }
    // Smallest element size whose halves are equal all the way down.
    let mut element_size = register_bits;
    loop {
        element_size /= 2;
        let half_mask = (1u64 << element_size) - 1;
        if (pattern & half_mask) != ((pattern >> element_size) & half_mask) {
            element_size *= 2;
            break;
        }
        if element_size <= 2 {
            break;
        }
    }
    let element_mask = u64::MAX >> (XREG_BITS - element_size);
    pattern &= element_mask;
    // True if `v` is one contiguous run of ones (possibly shifted up from bit 0).
    let is_run_of_ones = |v: u64| {
        v != 0 && {
            let filled = (v - 1) | v;
            filled & filled.wrapping_add(1) == 0
        }
    };
    let (rotation_source, run_length);
    if is_run_of_ones(pattern) {
        rotation_source = pattern.trailing_zeros();
        run_length = (pattern >> rotation_source).trailing_ones();
    } else {
        // The run wraps around the element: extend with ones above it and look at the zeros.
        pattern |= !element_mask;
        if !is_run_of_ones(!pattern) {
            return None;
        }
        let leading_ones = pattern.leading_ones();
        rotation_source = XREG_BITS - leading_ones;
        run_length = leading_ones + pattern.trailing_ones() - (XREG_BITS - element_size);
    }
    let immr = (element_size - rotation_source) & (element_size - 1);
    // imms holds run_length - 1 in its low bits; the bits above encode the element size
    // (0xxxxx for 32-bit elements, 10xxxx for 16, ...). N is set only for 64-bit elements,
    // which is the inverse of bit 6 of this value.
    let mut size_prefixed_imms = !(element_size - 1) << 1;
    size_prefixed_imms |= run_length - 1;
    let n = ((size_prefixed_imms >> 6) & 1) ^ 1;
    Some((n << 12) | (immr << 6) | (size_prefixed_imms & 0x3f))
}

impl Asm {
    pub fn new(base: u64) -> Asm {
        Asm {
            code: Vec::new(),
            base,
            labels: Vec::new(),
            fixes: Vec::new(),
            pool: Vec::new(),
        }
    }

    /// Address the first instruction is assembled for.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Address of the next instruction to be emitted.
    pub fn here(&self) -> u64 {
        self.base + self.code.len() as u64 * 4
    }

    pub fn emit(&mut self, word: u32) {
        self.code.push(word);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label] = Some(self.code.len());
    }

    /// Patches every branch to a label. The offsets are not range-checked: `b.cond`/`cbz` reach
    /// only +-1 MiB and `b` +-128 MiB, far more than one block (at most `MAX_INSNS` instructions
    /// plus their out-of-line slow paths) ever spans. Panics on a label that was never bound.
    pub fn finish(&mut self) {
        self.flush_pool();
        for &(at, label, fixup) in &self.fixes {
            let offset = self.labels[label].expect("unbound label") as i64 - at as i64;
            match fixup {
                Fixup::Imm26 => self.code[at] |= (offset as u32) & IMM26_MASK,
                Fixup::Imm19 | Fixup::Adr => {
                    self.code[at] |= ((offset as u32) & IMM19_MASK) << IMM19_SHIFT
                }
            }
        }
        self.fixes.clear();
    }

    /// Appends the literal pool, 8-byte aligned, and binds its labels. The code buffer is placed
    /// at a 16-byte boundary, so an even word index is an 8-byte aligned address.
    fn flush_pool(&mut self) {
        if self.pool.is_empty() {
            return;
        }
        if self.code.len() % 2 == 1 {
            self.emit(NOP_WORD);
        }
        for (label, value) in std::mem::take(&mut self.pool) {
            if let Some(label) = label {
                self.bind(label);
            }
            self.emit(value as u32);
            self.emit((value >> 32) as u32);
        }
    }

    /// A 64-bit constant stored after the code; the label addresses it for `adr` and
    /// `ldr_literal`. Both reach +-1 MiB, far more than one block.
    pub fn literal(&mut self, value: u64) -> Label {
        let label = self.label();
        self.pool.push((Some(label), value));
        label
    }

    /// Two consecutive 64-bit constants; the label addresses the first.
    pub fn literal_pair(&mut self, first: u64, second: u64) -> Label {
        let label = self.label();
        self.pool.push((Some(label), first));
        self.pool.push((None, second));
        label
    }

    /// `adr rd, label`: the address of a word-aligned label.
    pub fn adr(&mut self, rd: u32, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Adr));
        self.emit(0x1000_0000 | rd);
    }

    /// `ldr xt, label`: loads the 64-bit constant at a word-aligned label.
    pub fn ldr_literal(&mut self, rt: u32, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(0x5800_0000 | rt);
    }

    pub fn b(&mut self, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Imm26));
        self.emit(0x1400_0000);
    }

    pub fn b_cond(&mut self, cond: u32, label: Label) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(0x5400_0000 | cond);
    }

    pub fn cbz(&mut self, rt: u32, label: Label, is64: bool) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(if is64 { 0xb400_0000 } else { 0x3400_0000 } | rt);
    }

    pub fn cbnz(&mut self, rt: u32, label: Label, is64: bool) {
        self.fixes.push((self.code.len(), label, Fixup::Imm19));
        self.emit(if is64 { 0xb500_0000 } else { 0x3500_0000 } | rt);
    }

    /// Direct branch to an absolute address outside this `Asm` (the exit stubs in the prologue).
    /// Reach is +-128 MiB; the JIT code buffer (`CODE_SIZE` in `native.rs`) is exactly that big,
    /// so a block can always reach the stubs at its start.
    pub fn b_abs(&mut self, target: u64) {
        let offset = (target as i64 - self.here() as i64) / 4;
        assert!(offset.abs() < IMM26_SIGNED_LIMIT, "branch out of range");
        self.emit(0x1400_0000 | (offset as u32 & IMM26_MASK));
    }

    pub fn ret(&mut self) {
        self.emit(0xd65f_03c0);
    }

    /// `ret xn`: return through a register other than the link register.
    pub fn ret_reg(&mut self, rn: u32) {
        self.emit(0xd65f_0000 | rn << RN_SHIFT);
    }

    /// Direct call to an absolute address outside this `Asm`, reach +-128 MiB like `b_abs`.
    pub fn bl_abs(&mut self, target: u64) {
        let offset = (target as i64 - self.here() as i64) / 4;
        assert!(offset.abs() < IMM26_SIGNED_LIMIT, "branch out of range");
        self.emit(0x9400_0000 | (offset as u32 & IMM26_MASK));
    }

    pub fn blr(&mut self, rn: u32) {
        self.emit(0xd63f_0000 | rn << RN_SHIFT);
    }

    pub fn br(&mut self, rn: u32) {
        self.emit(0xd61f_0000 | rn << RN_SHIFT);
    }

    /// `hw` is the halfword index 0..=3 the 16 bits land in.
    pub fn movz(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xd280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    pub fn movk(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0xf280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    /// Writes the bitwise NOT of the shifted immediate.
    pub fn movn(&mut self, rd: u32, imm: u32, hw: u32) {
        self.emit(0x9280_0000 | hw << HW_SHIFT | (imm & 0xffff) << IMM16_SHIFT | rd);
    }

    /// Loads any 64-bit constant: one `movn` when the top 48 bits are all ones, otherwise a
    /// `movz` for the first non-zero halfword and a `movk` for each further one.
    pub fn mov_imm(&mut self, rd: u32, v: u64) {
        if v == 0 {
            self.movz(rd, 0, 0);
            return;
        }
        if !v >> 16 == 0 {
            self.movn(rd, (!v) as u32, 0);
            return;
        }
        let mut first = true;
        for hw in 0..4 {
            let part = ((v >> (16 * hw)) & 0xffff) as u32;
            if part != 0 {
                if first {
                    self.movz(rd, part, hw);
                    first = false;
                } else {
                    self.movk(rd, part, hw);
                }
            }
        }
    }

    /// `orr rd, xzr, rm`. Register 31 here is `xzr`, so this cannot move from or to `sp`.
    pub fn mov(&mut self, rd: u32, rm: u32) {
        self.emit(0xaa00_03e0 | rm << RM_SHIFT | rd);
    }

    /// `add rd, rn, #0`, the form that treats register 31 as `sp`.
    pub fn mov_sp(&mut self, rd: u32, rn: u32) {
        self.emit(0x9100_0000 | rn << RN_SHIFT | rd);
    }

    /// Unshifted 12-bit immediate only; callers must range-check (see `Gen::effective_address`).
    fn addsub_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        assert!(imm < IMM12_LIMIT);
        self.emit(sf | op | imm << IMM12_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn add_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x1100_0000, rd, rn, imm, is64);
    }
    pub fn sub_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x5100_0000, rd, rn, imm, is64);
    }
    pub fn adds_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x3100_0000, rd, rn, imm, is64);
    }
    pub fn subs_imm(&mut self, rd: u32, rn: u32, imm: u32, is64: bool) {
        self.addsub_imm(0x7100_0000, rd, rn, imm, is64);
    }

    fn reg3(&mut self, op: u32, rd: u32, rn: u32, rm: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        self.emit(sf | op | rm << RM_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn add(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0b00_0000, rd, rn, rm, is64);
    }
    pub fn adds(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2b00_0000, rd, rn, rm, is64);
    }
    /// `adcs`: rn + rm + C, setting NZCV.
    pub fn adcs(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x3a00_0000, rd, rn, rm, is64);
    }
    /// `sbcs`: rn - rm - !C, setting NZCV (C is NOT borrow, as for `subs`).
    pub fn sbcs(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x7a00_0000, rd, rn, rm, is64);
    }
    pub fn sub(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4b00_0000, rd, rn, rm, is64);
    }
    pub fn subs(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6b00_0000, rd, rn, rm, is64);
    }
    pub fn and(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x0a00_0000, rd, rn, rm, is64);
    }
    pub fn ands(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x6a00_0000, rd, rn, rm, is64);
    }
    pub fn orr(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x2a00_0000, rd, rn, rm, is64);
    }
    pub fn eor(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x4a00_0000, rd, rn, rm, is64);
    }
    /// High 64 bits of the signed 128-bit product.
    pub fn smulh(&mut self, rd: u32, rn: u32, rm: u32) {
        self.reg3(0x9b40_7c00, rd, rn, rm, false);
    }
    /// High 64 bits of the unsigned 128-bit product.
    pub fn umulh(&mut self, rd: u32, rn: u32, rm: u32) {
        self.reg3(0x9bc0_7c00, rd, rn, rm, false);
    }
    /// Counts leading zero bits of the 32- or 64-bit operand (the width when it is zero).
    pub fn clz(&mut self, rd: u32, rn: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        self.emit(sf | 0x5ac0_1000 | rn << RN_SHIFT | rd);
    }
    /// Reverses the bits of the 32- or 64-bit operand.
    pub fn rbit(&mut self, rd: u32, rn: u32, is64: bool) {
        let sf = if is64 { SF } else { 0 };
        self.emit(sf | 0x5ac0_0000 | rn << RN_SHIFT | rd);
    }
    pub fn mul(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1b00_7c00, rd, rn, rm, is64);
    }
    pub fn lslv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2000, rd, rn, rm, is64);
    }
    pub fn lsrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2400, rd, rn, rm, is64);
    }
    pub fn asrv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2800, rd, rn, rm, is64);
    }
    pub fn rorv(&mut self, rd: u32, rn: u32, rm: u32, is64: bool) {
        self.reg3(0x1ac0_2c00, rd, rn, rm, is64);
    }
    /// `extr rd, rn, rm, #lsb`: the low 32 or 64 bits of the concatenation rn:rm shifted right
    /// by `lsb`.
    pub fn extr(&mut self, rd: u32, rn: u32, rm: u32, lsb: u32, is64: bool) {
        let opcode = if is64 { 0x93c0_0000 } else { 0x1380_0000 };
        self.emit(opcode | rm << RM_SHIFT | lsb << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }

    /// `ror rd, rn, #amount`, the alias of `extr rd, rn, rn, #amount`.
    pub fn ror_imm(&mut self, rd: u32, rn: u32, amount: u32, is64: bool) {
        self.extr(rd, rn, rn, amount, is64);
    }

    /// 64-bit `add rd, rn, rm, lsl #shift`.
    pub fn add_lsl(&mut self, rd: u32, rn: u32, rm: u32, shift: u32) {
        self.emit(0x8b00_0000 | rm << RM_SHIFT | shift << IMM6_SHIFT | rn << RN_SHIFT | rd);
    }

    pub fn cmp(&mut self, rn: u32, rm: u32, is64: bool) {
        self.subs(XZR, rn, rm, is64);
    }
    pub fn cmp_imm(&mut self, rn: u32, imm: u32, is64: bool) {
        self.subs_imm(XZR, rn, imm, is64);
    }
    pub fn tst(&mut self, rn: u32, rm: u32, is64: bool) {
        self.ands(XZR, rn, rm, is64);
    }

    /// Returns false, emitting nothing, when `imm` is not a valid bitmask immediate; the
    /// caller must then materialise the constant in a register itself.
    #[must_use]
    fn logic_imm(&mut self, op: u32, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        match encode_bitmask(imm, is64) {
            Some(bitmask) => {
                let sf = if is64 { SF } else { 0 };
                self.emit(sf | op | bitmask << BITMASK_IMM_SHIFT | rn << RN_SHIFT | rd);
                true
            }
            None => false,
        }
    }

    /// See `logic_imm` for the meaning of the result.
    #[must_use]
    pub fn and_imm(&mut self, rd: u32, rn: u32, imm: u64, is64: bool) -> bool {
        self.logic_imm(0x1200_0000, rd, rn, imm, is64)
    }

    // The shifts and extensions below are all aliases of these three bitfield moves, with
    // `immr` the rotate amount and `imms` the index of the field's last bit.
    pub fn ubfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0xd340_0000 } else { 0x5300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }
    pub fn sbfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0x9340_0000 } else { 0x1300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }
    pub fn bfm(&mut self, rd: u32, rn: u32, immr: u32, imms: u32, is64: bool) {
        let opcode = if is64 { 0xb340_0000 } else { 0x3300_0000 };
        self.emit(opcode | immr << IMMR_SHIFT | imms << IMMS_SHIFT | rn << RN_SHIFT | rd);
    }

    /// 64-bit immediate shifts.
    pub fn lsl_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm(
            rd,
            rn,
            (XREG_BITS - shift) & (XREG_BITS - 1),
            XREG_BITS - 1 - shift,
            true,
        );
    }
    pub fn lsr_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.ubfm(rd, rn, shift, XREG_BITS - 1, true);
    }
    pub fn asr_imm(&mut self, rd: u32, rn: u32, shift: u32) {
        self.sbfm(rd, rn, shift, XREG_BITS - 1, true);
    }
    /// `ubfx rd, rn, #lsb, #width`: the field zero-extended into `rd`.
    pub fn ubfx(&mut self, rd: u32, rn: u32, lsb: u32, width: u32, is64: bool) {
        self.ubfm(rd, rn, lsb, lsb + width - 1, is64);
    }

    /// Immediate shifts at either register width (the 64-bit-only `lsl_imm` family above is the
    /// common case; these exist for 8/16/32-bit operand sizes done in W registers).
    pub fn lsl_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.ubfm(rd, rn, (bits - shift) & (bits - 1), bits - 1 - shift, is64);
    }
    pub fn lsr_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.ubfm(rd, rn, shift, bits - 1, is64);
    }
    pub fn asr_imm_sized(&mut self, rd: u32, rn: u32, shift: u32, is64: bool) {
        let bits = if is64 { XREG_BITS } else { WREG_BITS };
        self.sbfm(rd, rn, shift, bits - 1, is64);
    }
    pub fn uxtb(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 7, false);
    }
    pub fn uxth(&mut self, rd: u32, rn: u32) {
        self.ubfm(rd, rn, 0, 15, false);
    }
    /// 32-bit `mov`, which zeroes the upper half of the destination.
    pub fn uxtw(&mut self, rd: u32, rn: u32) {
        self.emit(0x2a00_03e0 | rn << RM_SHIFT | rd);
    }
    pub fn sxtb(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 7, is64);
    }
    pub fn sxth(&mut self, rd: u32, rn: u32, is64: bool) {
        self.sbfm(rd, rn, 0, 15, is64);
    }
    pub fn sxtw(&mut self, rd: u32, rn: u32) {
        self.sbfm(rd, rn, 0, 31, true);
    }

    /// Inserts the low `width` bits of `rn` into `rd` at bit `lsb`, keeping the other bits.
    pub fn bfi(&mut self, rd: u32, rn: u32, lsb: u32, width: u32, is64: bool) {
        let last_bit = if is64 { XREG_BITS - 1 } else { WREG_BITS - 1 };
        self.bfm(rd, rn, (last_bit + 1 - lsb) & last_bit, width - 1, is64);
    }

    /// `csinc rd, xzr, xzr, !cond`: rd = cond ? 1 : 0.
    pub fn cset(&mut self, rd: u32, cond: u32) {
        self.emit(0x9a9f_07e0 | ((cond ^ 1) << COND_SHIFT) | rd);
    }

    pub fn csel(&mut self, rd: u32, rn: u32, rm: u32, cond: u32) {
        self.emit(0x9a80_0000 | rm << RM_SHIFT | cond << COND_SHIFT | rn << RN_SHIFT | rd);
    }

    /// Unsigned-offset load/store: the encoded immediate is `off >> scale`, so `off` must be a
    /// multiple of the access size (`1 << scale`) and below `4096 << scale`.
    fn ldst_imm(&mut self, base: u32, scale: u32, rt: u32, rn: u32, off: u32) {
        assert!(
            off.is_multiple_of(1 << scale) && (off >> scale) < IMM12_LIMIT,
            "bad ldst offset {off}"
        );
        self.emit(base | (off >> scale) << IMM12_SHIFT | rn << RN_SHIFT | rt);
    }

    pub fn ldr(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf940_0000, 3, rt, rn, off);
    }
    pub fn str(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xf900_0000, 3, rt, rn, off);
    }
    pub fn ldr_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb940_0000, 2, rt, rn, off);
    }
    pub fn str_w(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0xb900_0000, 2, rt, rn, off);
    }
    pub fn ldrh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7940_0000, 1, rt, rn, off);
    }
    pub fn strh(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x7900_0000, 1, rt, rn, off);
    }
    pub fn ldrb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3940_0000, 0, rt, rn, off);
    }
    pub fn strb(&mut self, rt: u32, rn: u32, off: u32) {
        self.ldst_imm(0x3900_0000, 0, rt, rn, off);
    }

    /// Register-offset form `[rn, rm]`: no shift, no extend.
    fn ldst_reg(&mut self, base: u32, rt: u32, rn: u32, rm: u32) {
        self.emit(base | rm << RM_SHIFT | rn << RN_SHIFT | rt);
    }

    /// Loads `size` bytes from `[rn + rm]`. Unsigned loads zero-extend; signed ones sign-extend
    /// to 64 bits (`ldrsb`/`ldrsh`/`ldrsw` into an X register). A signed 8-byte load is a plain
    /// `ldr`, and a signed size other than 1 or 2 is treated as 4.
    pub fn ldr_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8, signed: bool) {
        let base = match (size, signed) {
            (1, false) => 0x3860_6800,
            (2, false) => 0x7860_6800,
            (4, false) => 0xb860_6800,
            (8, _) => 0xf860_6800,
            (1, true) => 0x38a0_6800,
            (2, true) => 0x78a0_6800,
            _ => 0xb8a0_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    /// Stores the low `size` bytes of `rt` to `[rn + rm]`; any size other than 1, 2 or 4
    /// stores 8.
    pub fn str_r(&mut self, rt: u32, rn: u32, rm: u32, size: u8) {
        let base = match size {
            1 => 0x3820_6800,
            2 => 0x7820_6800,
            4 => 0xb820_6800,
            _ => 0xf820_6800,
        };
        self.ldst_reg(base, rt, rn, rm);
    }

    /// Load-acquire of `size` bytes from `[rn]`, zero-extended: `ldarb`/`ldarh`/`ldar`. The
    /// address must be naturally aligned, or the access faults. Any size other than 1, 2 or 4
    /// loads 8.
    pub fn ldar(&mut self, rt: u32, rn: u32, size: u8) {
        let base = match size {
            1 => 0x08df_fc00,
            2 => 0x48df_fc00,
            4 => 0x88df_fc00,
            _ => 0xc8df_fc00,
        };
        self.emit(base | rn << RN_SHIFT | rt);
    }

    /// Store-release of the low `size` bytes of `rt` to `[rn]`; the counterpart of `ldar`.
    pub fn stlr(&mut self, rt: u32, rn: u32, size: u8) {
        let base = match size {
            1 => 0x089f_fc00,
            2 => 0x489f_fc00,
            4 => 0x889f_fc00,
            _ => 0xc89f_fc00,
        };
        self.emit(base | rn << RN_SHIFT | rt);
    }

    /// 64-bit `stp`/`ldp`. `imm` is a byte offset, a multiple of 8 in -512..=504.
    fn pair_imm7(imm: i32) -> u32 {
        ((imm / PAIR_OFFSET_SCALE) as u32 & IMM7_MASK) << IMM7_SHIFT
    }

    /// Pre-indexed: the base register is updated before the store.
    pub fn stp_pre(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa980_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    pub fn stp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa900_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    pub fn ldp_off(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa940_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }
    /// `csinc rd, rn, rn, ne`: rd = eq ? rn + 1 : rn.
    pub fn cinc_eq(&mut self, rd: u32, rn: u32) {
        self.emit(0x9a80_0400 | rn << RM_SHIFT | (NE << COND_SHIFT) | rn << RN_SHIFT | rd);
    }
    /// Read/write the NZCV condition flags (system register S3_3_C4_C2_0).
    pub fn mrs_nzcv(&mut self, rt: u32) {
        self.emit(0xd53b_4200 | rt);
    }
    pub fn msr_nzcv(&mut self, rt: u32) {
        self.emit(0xd51b_4200 | rt);
    }
    /// Post-indexed: the base register is updated after the load.
    pub fn ldp_post(&mut self, rt: u32, rt2: u32, rn: u32, imm: i32) {
        self.emit(0xa8c0_0000 | Self::pair_imm7(imm) | rt2 << RT2_SHIFT | rn << RN_SHIFT | rt);
    }

    /// 128-bit `ldr`/`str` of vector register `vt`; `off` is a byte offset, a multiple of 16 below
    /// 64 KiB.
    pub fn ldr_q(&mut self, vt: u32, rn: u32, off: u32) {
        self.emit(0x3dc0_0000 | Self::q_offset(off) | rn << RN_SHIFT | vt);
    }
    pub fn str_q(&mut self, vt: u32, rn: u32, off: u32) {
        self.emit(0x3d80_0000 | Self::q_offset(off) | rn << RN_SHIFT | vt);
    }
    fn q_offset(off: u32) -> u32 {
        assert!(
            off.is_multiple_of(VECTOR_BYTES) && off / VECTOR_BYTES <= IMM12_MASK,
            "q offset {off}"
        );
        (off / VECTOR_BYTES) << IMM12_SHIFT
    }

    /// A three-register vector operation on the full 128 bits: one of the `VEC_*` bases, `size`
    /// the lane width as log2 of the byte count (0 = 16 x 8-bit .. 3 = 2 x 64-bit; the
    /// bitwise operations take 0). `vd = vn op vm`; `VEC_BIC` is `vn & !vm`.
    pub fn vec_op(&mut self, base: u32, size: u32, vd: u32, vn: u32, vm: u32) {
        self.emit(base | size << VEC_SIZE_SHIFT | vm << RM_SHIFT | vn << RN_SHIFT | vd);
    }

    /// `ushr vd.16b, vn.16b, #shift` and the accumulating `usra`, per lane of `lane_bits` (8, 16,
    /// 32 or 64) bits, shift 1..=lane_bits.
    pub fn vec_ushr(&mut self, vd: u32, vn: u32, lane_bits: u32, shift: u32) {
        self.vec_shift_right(0x6f00_0400, vd, vn, lane_bits, shift);
    }
    pub fn vec_usra(&mut self, vd: u32, vn: u32, lane_bits: u32, shift: u32) {
        self.vec_shift_right(0x6f00_1400, vd, vn, lane_bits, shift);
    }
    fn vec_shift_right(&mut self, base: u32, vd: u32, vn: u32, lane_bits: u32, shift: u32) {
        assert!(lane_bits.is_power_of_two() && (1..=lane_bits).contains(&shift));
        // immh:immb is `2 * lane_bits - shift`; the position of its top set bit gives the lane size.
        let immediate = 2 * lane_bits - shift;
        self.emit(base | immediate << VEC_SHIFT_IMMEDIATE_SHIFT | vn << RN_SHIFT | vd);
    }

    /// `umov wd, vn.b[index]`.
    /// `fmov dd, xn`: the low 64 bits of vector register `vd` from a general register, the rest
    /// zeroed.
    pub fn fmov_d_x(&mut self, vd: u32, xn: u32) {
        self.emit(0x9e67_0000 | xn << RN_SHIFT | vd);
    }
    /// `mov vd.d[1], xn`: the high 64 bits of `vd` from a general register.
    pub fn ins_d1(&mut self, vd: u32, xn: u32) {
        self.emit(0x4e18_1c00 | xn << RN_SHIFT | vd);
    }
    pub fn umov_byte(&mut self, rd: u32, vn: u32, index: u32) {
        assert!(index < VECTOR_BYTES);
        self.emit(
            0x0e00_3c00 | (index << 1 | 1) << VEC_SHIFT_IMMEDIATE_SHIFT | vn << RN_SHIFT | rd,
        );
    }

    /// `sha256su0 vd.4s, vn.4s`: the first half of SHA-256's message schedule, which is exactly
    /// x86's sha256msg1 with the destination in `vd` and the source in `vn` (both lane 0 low).
    pub fn sha256su0(&mut self, vd: u32, vn: u32) {
        self.emit(0x5e28_2800 | vn << RN_SHIFT | vd);
    }
    /// `sha256su1 vd.4s, vn.4s, vm.4s`: `vd + sigma1(...)` like x86's sha256msg2, plus the
    /// `W[t-7]` words taken from `vn` and lane 0 of `vm`, which a caller wanting sha256msg2 zeroes.
    pub fn sha256su1(&mut self, vd: u32, vn: u32, vm: u32) {
        self.emit(0x5e00_6000 | vm << RM_SHIFT | vn << RN_SHIFT | vd);
    }
    /// `ext vd.16b, vn.16b, vm.16b, #index`: bytes `index..16` of `vn` followed by bytes
    /// `0..index` of `vm`, the 32-byte `vm:vn` shifted right by `index` bytes.
    pub fn ext_16b(&mut self, vd: u32, vn: u32, vm: u32, index: u32) {
        const EXT_INDEX_SHIFT: u32 = 11;
        assert!(index < VECTOR_BYTES);
        self.emit(0x6e00_0000 | vm << RM_SHIFT | index << EXT_INDEX_SHIFT | vn << RN_SHIFT | vd);
    }
    /// `movi vd.2d, #0`.
    pub fn vec_zero(&mut self, vd: u32) {
        self.emit(0x6f00_e400 | vd);
    }
    /// `mov vd.s[0], wzr`: clears the low dword of `vd` and keeps the rest.
    pub fn clear_low_dword(&mut self, vd: u32) {
        const ZERO_REGISTER: u32 = 31;
        self.emit(0x4e04_1c00 | ZERO_REGISTER << RN_SHIFT | vd);
    }
    /// `movi vd.16b, #imm`: `imm` in every byte.
    pub fn vec_splat_byte(&mut self, vd: u32, imm: u8) {
        const ABC_SHIFT: u32 = 16;
        const DEFGH_SHIFT: u32 = 5;
        const DEFGH_BITS: u32 = 5;
        let imm = imm as u32;
        self.emit(
            0x4f00_e400
                | (imm >> DEFGH_BITS) << ABC_SHIFT
                | (imm & ((1 << DEFGH_BITS) - 1)) << DEFGH_SHIFT
                | vd,
        );
    }
    /// `tbl vd.16b, {vn.16b}, vm.16b`: byte `k` of `vd` is byte `vm[k]` of `vn` (0 past 15).
    pub fn tbl(&mut self, vd: u32, vn: u32, vm: u32) {
        self.emit(0x4e00_0000 | vm << RM_SHIFT | vn << RN_SHIFT | vd);
    }

    /// `dmb ishld`: loads before it complete before any later load or store. A plain vector
    /// load plus this is an acquire, which `ldar` cannot give a 128-bit access.
    pub fn dmb_ishld(&mut self) {
        self.emit(0xd503_39bf);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::sys::ExecMem;

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        f(&mut a);
        a.ret();
        a.finish();
        m.write(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    /// The words clang produces for the same instructions.
    #[test]
    fn vector_instructions_encode_like_the_assembler() {
        let words = |build: &dyn Fn(&mut Asm)| {
            let mut a = Asm::new(0);
            build(&mut a);
            a.code.clone()
        };
        assert_eq!(
            words(&|a| {
                a.ldr_q(0, 19, 1296); // ldr q0, [x19, #1296]
                a.ldr_q(1, 19, 1312);
                a.str_q(0, 19, 1296);
                a.str_q(1, 19, 1328);
            }),
            [0x3dc1_4660, 0x3dc1_4a61, 0x3d81_4660, 0x3d81_4e61]
        );
        assert_eq!(
            words(&|a| {
                a.vec_op(VEC_AND, 0, 0, 0, 1);
                a.vec_op(VEC_BIC, 0, 0, 1, 0); // bic v0.16b, v1.16b, v0.16b
                a.vec_op(VEC_ORR, 0, 0, 0, 1);
                a.vec_op(VEC_EOR, 0, 0, 0, 1);
                a.vec_op(VEC_CMEQ, 0, 0, 0, 1);
                a.vec_op(VEC_CMEQ, 1, 0, 0, 1);
                a.vec_op(VEC_CMEQ, 2, 0, 0, 1);
                a.vec_op(VEC_UMIN, 0, 0, 0, 1);
                a.vec_op(VEC_UMAX, 0, 0, 0, 1);
                a.vec_op(VEC_ADD, 0, 0, 0, 1);
                a.vec_op(VEC_ADD, 1, 0, 0, 1);
                a.vec_op(VEC_ADD, 2, 0, 0, 1);
                a.vec_op(VEC_ADD, 3, 0, 0, 1);
                a.vec_op(VEC_SUB, 0, 0, 0, 1);
                a.vec_op(VEC_SUB, 1, 0, 0, 1);
                a.vec_op(VEC_SUB, 2, 0, 0, 1);
                a.vec_op(VEC_SUB, 3, 0, 0, 1);
                a.vec_op(VEC_CMTST, 0, 0, 1, 2); // cmtst v0.16b, v1.16b, v2.16b
                a.vec_op(VEC_NOT, 0, 0, 1, 0); // not v0.16b, v1.16b
                a.fmov_d_x(2, 5); // fmov d2, x5
                a.ins_d1(2, 7); // mov v2.d[1], x7
                a.vec_op(VEC_BSL, 0, 2, 0, 3); // bsl v2.16b, v0.16b, v3.16b
                a.vec_op(VEC_CMHI, 0, 0, 1, 2); // cmhi v0.16b, v1.16b, v2.16b
                a.vec_op(VEC_CMHS, 0, 0, 1, 2);
                a.vec_op(VEC_CMGT, 0, 0, 1, 2);
                a.vec_op(VEC_CMGE, 0, 0, 1, 2);
            }),
            [
                0x4e21_1c00,
                0x4e60_1c20,
                0x4ea1_1c00,
                0x6e21_1c00,
                0x6e21_8c00,
                0x6e61_8c00,
                0x6ea1_8c00,
                0x6e21_6c00,
                0x6e21_6400,
                0x4e21_8400,
                0x4e61_8400,
                0x4ea1_8400,
                0x4ee1_8400,
                0x6e21_8400,
                0x6e61_8400,
                0x6ea1_8400,
                0x6ee1_8400,
                0x4e22_8c20,
                0x6e20_5820,
                0x9e67_00a2,
                0x4e18_1ce2,
                0x6e63_1c02,
                0x6e22_3420,
                0x6e22_3c20,
                0x4e22_3420,
                0x4e22_3c20,
            ]
        );
        assert_eq!(
            words(&|a| {
                a.vec_ushr(1, 1, 8, 7);
                a.vec_usra(1, 1, 16, 7);
                a.vec_usra(1, 1, 32, 14);
                a.vec_usra(1, 1, 64, 28);
                a.umov_byte(9, 1, 0);
                a.umov_byte(10, 1, 8);
                a.dmb_ishld();
            }),
            [
                0x6f09_0421,
                0x6f19_1421,
                0x6f32_1421,
                0x6f64_1421,
                0x0e01_3c29,
                0x0e11_3c2a,
                0xd503_39bf,
            ]
        );
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run(|a| a.add(0, 0, 1, true), 40, 2), 42);
        assert_eq!(run(|a| a.sub(0, 0, 1, true), 40, 2), 38);
        assert_eq!(
            run(|a| a.mov_imm(0, 0x1234_5678_9abc_def0), 0, 0),
            0x1234_5678_9abc_def0
        );
        assert_eq!(run(|a| a.mov_imm(0, u64::MAX - 5), 0, 0), u64::MAX - 5);
        assert_eq!(
            run(|a| a.mov_imm(0, 0xffff_0000_0000_1234), 0, 0),
            0xffff_0000_0000_1234
        );
        assert_eq!(run(|a| a.lsl_imm(0, 0, 4), 3, 0), 48);
        assert_eq!(run(|a| a.lsr_imm(0, 0, 4), 0xf0, 0), 0xf);
        assert_eq!(
            run(|a| a.asr_imm(0, 0, 4), (-32i64) as u64, 0),
            (-2i64) as u64
        );
        assert_eq!(run(|a| a.mul(0, 0, 1, true), 6, 7), 42);
    }

    #[test]
    fn logic_immediates() {
        for &v in &[
            0xffu64,
            0x3ff,
            0xffff_ffff,
            !0xfffu64,
            0xff00,
            0x5555_5555_5555_5555,
            0x8000_0000_0000_0001,
        ] {
            let got = run(
                |a| {
                    assert!(a.and_imm(0, 0, v, true), "{v:#x}");
                },
                u64::MAX,
                0,
            );
            assert_eq!(got, v, "{v:#x}");
        }
        assert!(encode_bitmask(0x1234, true).is_none());
        assert!(encode_bitmask(0, true).is_none());
    }

    #[test]
    fn branches_and_flags() {
        let r = run(
            |a| {
                let l = a.label();
                a.cmp(0, 1, true);
                a.mov_imm(0, 1);
                a.b_cond(LT, l);
                a.mov_imm(0, 2);
                a.bind(l);
            },
            1,
            5,
        );
        assert_eq!(r, 1);
        assert_eq!(
            run(
                |a| {
                    a.adds(2, 0, 1, true);
                    a.cset(0, CS);
                },
                u64::MAX,
                1
            ),
            1
        );
    }

    #[test]
    fn loads_stores() {
        let mut buf = [0u64; 4];
        let p = buf.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(2, 0xdead_beef_1234_5678);
                a.str(2, 0, 8);
            },
            p,
            0,
        );
        assert_eq!(buf[1], 0xdead_beef_1234_5678);
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 4, false);
                },
                p,
                0
            ),
            0x1234_5678
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 1, true);
                },
                p,
                0
            ),
            0x78
        );
        run(
            |a| {
                a.mov_imm(2, 0xab);
                a.mov_imm(3, 16);
                a.str_r(2, 0, 3, 1);
            },
            p,
            0,
        );
        assert_eq!(buf[2] & 0xff, 0xab);
    }
}

#[cfg(test)]
mod encoding_tests {
    //! Encoder tests for the aarch64 assembler. Golden words come from `clang -target aarch64`
    //! (assembling the instruction named next to each case), never from this encoder. The bitmask
    //! immediate encoder is also checked against an independent decoder of the architecture's
    //! `DecodeBitMasks` for every valid encoding.

    use super::*;
    use crate::sys::ExecMem;

    /// `nop`
    const NOP_WORD: u32 = 0xd503_201f;
    const BASE_ADDRESS: u64 = 0x1000_0000;

    /// An assembly line, the encoder call it must match, and the bytes the assembler produced.
    type EncodeCase = (&'static str, fn(&mut Asm), &'static [u32]);

    fn nop(a: &mut Asm) {
        a.emit(NOP_WORD);
    }

    fn assemble(build: impl FnOnce(&mut Asm)) -> Vec<u32> {
        let mut a = Asm::new(BASE_ADDRESS);
        build(&mut a);
        a.finish();
        a.code
    }

    #[track_caller]
    fn check_encodings(cases: &[EncodeCase]) {
        for (assembly, encode, expected) in cases {
            let mut a = Asm::new(BASE_ADDRESS);
            encode(&mut a);
            assert_eq!(a.code.as_slice(), *expected, "`{assembly}`");
        }
    }

    fn run(f: impl FnOnce(&mut Asm), a0: u64, a1: u64) -> u64 {
        let _guard = crate::jit::serialize_jit_tests();
        let m = ExecMem::new(16384).unwrap();
        let mut a = Asm::new(m.base());
        f(&mut a);
        a.ret();
        a.finish();
        m.write(0, &a.code);
        let fun: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(m.base() as usize) };
        fun(a0, a1)
    }

    /// Add/adds/sub/subs/and/ands/orr/eor/mul and the register shifts, 64- and 32-bit, with the zero register in every position.
#[rustfmt::skip]
#[test]
fn register_arithmetic_and_logic_encodings_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add x0, x1, x2", |a| { a.add(0, 1, 2, true); }, &[0x8b020020]),
        ("add x28, x29, x30", |a| { a.add(28, 29, 30, true); }, &[0x8b1e03bc]),
        ("add xzr, x3, x4", |a| { a.add(31, 3, 4, true); }, &[0x8b04007f]),
        ("add x5, xzr, x6", |a| { a.add(5, 31, 6, true); }, &[0x8b0603e5]),
        ("add x7, x8, xzr", |a| { a.add(7, 8, 31, true); }, &[0x8b1f0107]),
        ("adds x0, x1, x2", |a| { a.adds(0, 1, 2, true); }, &[0xab020020]),
        ("adcs x0, x1, x2", |a| { a.adcs(0, 1, 2, true); }, &[0xba020020]),
        ("adcs w0, w1, w2", |a| { a.adcs(0, 1, 2, false); }, &[0x3a020020]),
        ("sbcs x0, x1, x2", |a| { a.sbcs(0, 1, 2, true); }, &[0xfa020020]),
        ("sbcs w3, w4, w5", |a| { a.sbcs(3, 4, 5, false); }, &[0x7a050083]),
        ("adds x28, x29, x30", |a| { a.adds(28, 29, 30, true); }, &[0xab1e03bc]),
        ("adds xzr, x3, x4", |a| { a.adds(31, 3, 4, true); }, &[0xab04007f]),
        ("adds x5, xzr, x6", |a| { a.adds(5, 31, 6, true); }, &[0xab0603e5]),
        ("adds x7, x8, xzr", |a| { a.adds(7, 8, 31, true); }, &[0xab1f0107]),
        ("sub x0, x1, x2", |a| { a.sub(0, 1, 2, true); }, &[0xcb020020]),
        ("sub x28, x29, x30", |a| { a.sub(28, 29, 30, true); }, &[0xcb1e03bc]),
        ("sub xzr, x3, x4", |a| { a.sub(31, 3, 4, true); }, &[0xcb04007f]),
        ("sub x5, xzr, x6", |a| { a.sub(5, 31, 6, true); }, &[0xcb0603e5]),
        ("sub x7, x8, xzr", |a| { a.sub(7, 8, 31, true); }, &[0xcb1f0107]),
        ("subs x0, x1, x2", |a| { a.subs(0, 1, 2, true); }, &[0xeb020020]),
        ("subs x28, x29, x30", |a| { a.subs(28, 29, 30, true); }, &[0xeb1e03bc]),
        ("subs xzr, x3, x4", |a| { a.subs(31, 3, 4, true); }, &[0xeb04007f]),
        ("subs x5, xzr, x6", |a| { a.subs(5, 31, 6, true); }, &[0xeb0603e5]),
        ("subs x7, x8, xzr", |a| { a.subs(7, 8, 31, true); }, &[0xeb1f0107]),
        ("and x0, x1, x2", |a| { a.and(0, 1, 2, true); }, &[0x8a020020]),
        ("and x28, x29, x30", |a| { a.and(28, 29, 30, true); }, &[0x8a1e03bc]),
        ("and xzr, x3, x4", |a| { a.and(31, 3, 4, true); }, &[0x8a04007f]),
        ("and x5, xzr, x6", |a| { a.and(5, 31, 6, true); }, &[0x8a0603e5]),
        ("and x7, x8, xzr", |a| { a.and(7, 8, 31, true); }, &[0x8a1f0107]),
        ("ands x0, x1, x2", |a| { a.ands(0, 1, 2, true); }, &[0xea020020]),
        ("ands x28, x29, x30", |a| { a.ands(28, 29, 30, true); }, &[0xea1e03bc]),
        ("ands xzr, x3, x4", |a| { a.ands(31, 3, 4, true); }, &[0xea04007f]),
        ("ands x5, xzr, x6", |a| { a.ands(5, 31, 6, true); }, &[0xea0603e5]),
        ("ands x7, x8, xzr", |a| { a.ands(7, 8, 31, true); }, &[0xea1f0107]),
        ("orr x0, x1, x2", |a| { a.orr(0, 1, 2, true); }, &[0xaa020020]),
        ("orr x28, x29, x30", |a| { a.orr(28, 29, 30, true); }, &[0xaa1e03bc]),
        ("orr xzr, x3, x4", |a| { a.orr(31, 3, 4, true); }, &[0xaa04007f]),
        ("orr x5, xzr, x6", |a| { a.orr(5, 31, 6, true); }, &[0xaa0603e5]),
        ("orr x7, x8, xzr", |a| { a.orr(7, 8, 31, true); }, &[0xaa1f0107]),
        ("eor x0, x1, x2", |a| { a.eor(0, 1, 2, true); }, &[0xca020020]),
        ("eor x28, x29, x30", |a| { a.eor(28, 29, 30, true); }, &[0xca1e03bc]),
        ("eor xzr, x3, x4", |a| { a.eor(31, 3, 4, true); }, &[0xca04007f]),
        ("eor x5, xzr, x6", |a| { a.eor(5, 31, 6, true); }, &[0xca0603e5]),
        ("eor x7, x8, xzr", |a| { a.eor(7, 8, 31, true); }, &[0xca1f0107]),
        ("mul x0, x1, x2", |a| { a.mul(0, 1, 2, true); }, &[0x9b027c20]),
        ("mul x28, x29, x30", |a| { a.mul(28, 29, 30, true); }, &[0x9b1e7fbc]),
        ("mul xzr, x3, x4", |a| { a.mul(31, 3, 4, true); }, &[0x9b047c7f]),
        ("mul x5, xzr, x6", |a| { a.mul(5, 31, 6, true); }, &[0x9b067fe5]),
        ("mul x7, x8, xzr", |a| { a.mul(7, 8, 31, true); }, &[0x9b1f7d07]),
        ("lsl x0, x1, x2", |a| { a.lslv(0, 1, 2, true); }, &[0x9ac22020]),
        ("lsl x28, x29, x30", |a| { a.lslv(28, 29, 30, true); }, &[0x9ade23bc]),
        ("lsl xzr, x3, x4", |a| { a.lslv(31, 3, 4, true); }, &[0x9ac4207f]),
        ("lsl x5, xzr, x6", |a| { a.lslv(5, 31, 6, true); }, &[0x9ac623e5]),
        ("lsl x7, x8, xzr", |a| { a.lslv(7, 8, 31, true); }, &[0x9adf2107]),
        ("lsr x0, x1, x2", |a| { a.lsrv(0, 1, 2, true); }, &[0x9ac22420]),
        ("lsr x28, x29, x30", |a| { a.lsrv(28, 29, 30, true); }, &[0x9ade27bc]),
        ("lsr xzr, x3, x4", |a| { a.lsrv(31, 3, 4, true); }, &[0x9ac4247f]),
        ("lsr x5, xzr, x6", |a| { a.lsrv(5, 31, 6, true); }, &[0x9ac627e5]),
        ("lsr x7, x8, xzr", |a| { a.lsrv(7, 8, 31, true); }, &[0x9adf2507]),
        ("asr x0, x1, x2", |a| { a.asrv(0, 1, 2, true); }, &[0x9ac22820]),
        ("asr x28, x29, x30", |a| { a.asrv(28, 29, 30, true); }, &[0x9ade2bbc]),
        ("asr xzr, x3, x4", |a| { a.asrv(31, 3, 4, true); }, &[0x9ac4287f]),
        ("asr x5, xzr, x6", |a| { a.asrv(5, 31, 6, true); }, &[0x9ac62be5]),
        ("asr x7, x8, xzr", |a| { a.asrv(7, 8, 31, true); }, &[0x9adf2907]),
        ("add w0, w1, w2", |a| { a.add(0, 1, 2, false); }, &[0x0b020020]),
        ("add w28, w29, w30", |a| { a.add(28, 29, 30, false); }, &[0x0b1e03bc]),
        ("add wzr, w3, w4", |a| { a.add(31, 3, 4, false); }, &[0x0b04007f]),
        ("add w5, wzr, w6", |a| { a.add(5, 31, 6, false); }, &[0x0b0603e5]),
        ("add w7, w8, wzr", |a| { a.add(7, 8, 31, false); }, &[0x0b1f0107]),
        ("adds w0, w1, w2", |a| { a.adds(0, 1, 2, false); }, &[0x2b020020]),
        ("adds w28, w29, w30", |a| { a.adds(28, 29, 30, false); }, &[0x2b1e03bc]),
        ("adds wzr, w3, w4", |a| { a.adds(31, 3, 4, false); }, &[0x2b04007f]),
        ("adds w5, wzr, w6", |a| { a.adds(5, 31, 6, false); }, &[0x2b0603e5]),
        ("adds w7, w8, wzr", |a| { a.adds(7, 8, 31, false); }, &[0x2b1f0107]),
        ("sub w0, w1, w2", |a| { a.sub(0, 1, 2, false); }, &[0x4b020020]),
        ("sub w28, w29, w30", |a| { a.sub(28, 29, 30, false); }, &[0x4b1e03bc]),
        ("sub wzr, w3, w4", |a| { a.sub(31, 3, 4, false); }, &[0x4b04007f]),
        ("sub w5, wzr, w6", |a| { a.sub(5, 31, 6, false); }, &[0x4b0603e5]),
        ("sub w7, w8, wzr", |a| { a.sub(7, 8, 31, false); }, &[0x4b1f0107]),
        ("subs w0, w1, w2", |a| { a.subs(0, 1, 2, false); }, &[0x6b020020]),
        ("subs w28, w29, w30", |a| { a.subs(28, 29, 30, false); }, &[0x6b1e03bc]),
        ("subs wzr, w3, w4", |a| { a.subs(31, 3, 4, false); }, &[0x6b04007f]),
        ("subs w5, wzr, w6", |a| { a.subs(5, 31, 6, false); }, &[0x6b0603e5]),
        ("subs w7, w8, wzr", |a| { a.subs(7, 8, 31, false); }, &[0x6b1f0107]),
        ("and w0, w1, w2", |a| { a.and(0, 1, 2, false); }, &[0x0a020020]),
        ("and w28, w29, w30", |a| { a.and(28, 29, 30, false); }, &[0x0a1e03bc]),
        ("and wzr, w3, w4", |a| { a.and(31, 3, 4, false); }, &[0x0a04007f]),
        ("and w5, wzr, w6", |a| { a.and(5, 31, 6, false); }, &[0x0a0603e5]),
        ("and w7, w8, wzr", |a| { a.and(7, 8, 31, false); }, &[0x0a1f0107]),
        ("ands w0, w1, w2", |a| { a.ands(0, 1, 2, false); }, &[0x6a020020]),
        ("ands w28, w29, w30", |a| { a.ands(28, 29, 30, false); }, &[0x6a1e03bc]),
        ("ands wzr, w3, w4", |a| { a.ands(31, 3, 4, false); }, &[0x6a04007f]),
        ("ands w5, wzr, w6", |a| { a.ands(5, 31, 6, false); }, &[0x6a0603e5]),
        ("ands w7, w8, wzr", |a| { a.ands(7, 8, 31, false); }, &[0x6a1f0107]),
        ("orr w0, w1, w2", |a| { a.orr(0, 1, 2, false); }, &[0x2a020020]),
        ("orr w28, w29, w30", |a| { a.orr(28, 29, 30, false); }, &[0x2a1e03bc]),
        ("orr wzr, w3, w4", |a| { a.orr(31, 3, 4, false); }, &[0x2a04007f]),
        ("orr w5, wzr, w6", |a| { a.orr(5, 31, 6, false); }, &[0x2a0603e5]),
        ("orr w7, w8, wzr", |a| { a.orr(7, 8, 31, false); }, &[0x2a1f0107]),
        ("eor w0, w1, w2", |a| { a.eor(0, 1, 2, false); }, &[0x4a020020]),
        ("eor w28, w29, w30", |a| { a.eor(28, 29, 30, false); }, &[0x4a1e03bc]),
        ("eor wzr, w3, w4", |a| { a.eor(31, 3, 4, false); }, &[0x4a04007f]),
        ("eor w5, wzr, w6", |a| { a.eor(5, 31, 6, false); }, &[0x4a0603e5]),
        ("eor w7, w8, wzr", |a| { a.eor(7, 8, 31, false); }, &[0x4a1f0107]),
        ("mul w0, w1, w2", |a| { a.mul(0, 1, 2, false); }, &[0x1b027c20]),
        ("mul w28, w29, w30", |a| { a.mul(28, 29, 30, false); }, &[0x1b1e7fbc]),
        ("mul wzr, w3, w4", |a| { a.mul(31, 3, 4, false); }, &[0x1b047c7f]),
        ("mul w5, wzr, w6", |a| { a.mul(5, 31, 6, false); }, &[0x1b067fe5]),
        ("mul w7, w8, wzr", |a| { a.mul(7, 8, 31, false); }, &[0x1b1f7d07]),
        ("lsl w0, w1, w2", |a| { a.lslv(0, 1, 2, false); }, &[0x1ac22020]),
        ("lsl w28, w29, w30", |a| { a.lslv(28, 29, 30, false); }, &[0x1ade23bc]),
        ("lsl wzr, w3, w4", |a| { a.lslv(31, 3, 4, false); }, &[0x1ac4207f]),
        ("lsl w5, wzr, w6", |a| { a.lslv(5, 31, 6, false); }, &[0x1ac623e5]),
        ("lsl w7, w8, wzr", |a| { a.lslv(7, 8, 31, false); }, &[0x1adf2107]),
        ("lsr w0, w1, w2", |a| { a.lsrv(0, 1, 2, false); }, &[0x1ac22420]),
        ("lsr w28, w29, w30", |a| { a.lsrv(28, 29, 30, false); }, &[0x1ade27bc]),
        ("lsr wzr, w3, w4", |a| { a.lsrv(31, 3, 4, false); }, &[0x1ac4247f]),
        ("lsr w5, wzr, w6", |a| { a.lsrv(5, 31, 6, false); }, &[0x1ac627e5]),
        ("lsr w7, w8, wzr", |a| { a.lsrv(7, 8, 31, false); }, &[0x1adf2507]),
        ("asr w0, w1, w2", |a| { a.asrv(0, 1, 2, false); }, &[0x1ac22820]),
        ("asr w28, w29, w30", |a| { a.asrv(28, 29, 30, false); }, &[0x1ade2bbc]),
        ("asr wzr, w3, w4", |a| { a.asrv(31, 3, 4, false); }, &[0x1ac4287f]),
        ("asr w5, wzr, w6", |a| { a.asrv(5, 31, 6, false); }, &[0x1ac62be5]),
        ("asr w7, w8, wzr", |a| { a.asrv(7, 8, 31, false); }, &[0x1adf2907]),
    ];
    check_encodings(cases);
}

    /// 12-bit immediate forms; register 31 is sp for add/sub and xzr for the flag-setting forms.
#[rustfmt::skip]
#[test]
fn immediate_add_and_subtract_encodings_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add x0, x1, #0", |a| { a.add_imm(0, 1, 0, true); }, &[0x91000020]),
        ("add x1, x2, #1", |a| { a.add_imm(1, 2, 1, true); }, &[0x91000441]),
        ("add x28, x29, #4095", |a| { a.add_imm(28, 29, 4095, true); }, &[0x913fffbc]),
        ("add sp, sp, #16", |a| { a.add_imm(31, 31, 16, true); }, &[0x910043ff]),
        ("add x3, sp, #8", |a| { a.add_imm(3, 31, 8, true); }, &[0x910023e3]),
        ("sub x0, x1, #0", |a| { a.sub_imm(0, 1, 0, true); }, &[0xd1000020]),
        ("sub x1, x2, #1", |a| { a.sub_imm(1, 2, 1, true); }, &[0xd1000441]),
        ("sub x28, x29, #4095", |a| { a.sub_imm(28, 29, 4095, true); }, &[0xd13fffbc]),
        ("sub sp, sp, #16", |a| { a.sub_imm(31, 31, 16, true); }, &[0xd10043ff]),
        ("sub x3, sp, #8", |a| { a.sub_imm(3, 31, 8, true); }, &[0xd10023e3]),
        ("adds x0, x1, #0", |a| { a.adds_imm(0, 1, 0, true); }, &[0xb1000020]),
        ("adds x1, x2, #1", |a| { a.adds_imm(1, 2, 1, true); }, &[0xb1000441]),
        ("adds x28, x29, #4095", |a| { a.adds_imm(28, 29, 4095, true); }, &[0xb13fffbc]),
        ("adds xzr, sp, #16", |a| { a.adds_imm(31, 31, 16, true); }, &[0xb10043ff]),
        ("adds x3, sp, #8", |a| { a.adds_imm(3, 31, 8, true); }, &[0xb10023e3]),
        ("subs x0, x1, #0", |a| { a.subs_imm(0, 1, 0, true); }, &[0xf1000020]),
        ("subs x1, x2, #1", |a| { a.subs_imm(1, 2, 1, true); }, &[0xf1000441]),
        ("subs x28, x29, #4095", |a| { a.subs_imm(28, 29, 4095, true); }, &[0xf13fffbc]),
        ("subs xzr, sp, #16", |a| { a.subs_imm(31, 31, 16, true); }, &[0xf10043ff]),
        ("subs x3, sp, #8", |a| { a.subs_imm(3, 31, 8, true); }, &[0xf10023e3]),
        ("add w0, w1, #0", |a| { a.add_imm(0, 1, 0, false); }, &[0x11000020]),
        ("add w1, w2, #1", |a| { a.add_imm(1, 2, 1, false); }, &[0x11000441]),
        ("add w28, w29, #4095", |a| { a.add_imm(28, 29, 4095, false); }, &[0x113fffbc]),
        ("add wsp, wsp, #16", |a| { a.add_imm(31, 31, 16, false); }, &[0x110043ff]),
        ("add w3, wsp, #8", |a| { a.add_imm(3, 31, 8, false); }, &[0x110023e3]),
        ("sub w0, w1, #0", |a| { a.sub_imm(0, 1, 0, false); }, &[0x51000020]),
        ("sub w1, w2, #1", |a| { a.sub_imm(1, 2, 1, false); }, &[0x51000441]),
        ("sub w28, w29, #4095", |a| { a.sub_imm(28, 29, 4095, false); }, &[0x513fffbc]),
        ("sub wsp, wsp, #16", |a| { a.sub_imm(31, 31, 16, false); }, &[0x510043ff]),
        ("sub w3, wsp, #8", |a| { a.sub_imm(3, 31, 8, false); }, &[0x510023e3]),
        ("adds w0, w1, #0", |a| { a.adds_imm(0, 1, 0, false); }, &[0x31000020]),
        ("adds w1, w2, #1", |a| { a.adds_imm(1, 2, 1, false); }, &[0x31000441]),
        ("adds w28, w29, #4095", |a| { a.adds_imm(28, 29, 4095, false); }, &[0x313fffbc]),
        ("adds wzr, wsp, #16", |a| { a.adds_imm(31, 31, 16, false); }, &[0x310043ff]),
        ("adds w3, wsp, #8", |a| { a.adds_imm(3, 31, 8, false); }, &[0x310023e3]),
        ("subs w0, w1, #0", |a| { a.subs_imm(0, 1, 0, false); }, &[0x71000020]),
        ("subs w1, w2, #1", |a| { a.subs_imm(1, 2, 1, false); }, &[0x71000441]),
        ("subs w28, w29, #4095", |a| { a.subs_imm(28, 29, 4095, false); }, &[0x713fffbc]),
        ("subs wzr, wsp, #16", |a| { a.subs_imm(31, 31, 16, false); }, &[0x710043ff]),
        ("subs w3, wsp, #8", |a| { a.subs_imm(3, 31, 8, false); }, &[0x710023e3]),
    ];
    check_encodings(cases);
}

    /// Cmp, cmp with immediate and tst are subs/ands into the zero register.
#[rustfmt::skip]
#[test]
fn compare_and_test_aliases_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("cmp x0, x1", |a| { a.cmp(0, 1, true); }, &[0xeb01001f]),
        ("tst x0, x1", |a| { a.tst(0, 1, true); }, &[0xea01001f]),
        ("cmp x30, x29", |a| { a.cmp(30, 29, true); }, &[0xeb1d03df]),
        ("tst x30, x29", |a| { a.tst(30, 29, true); }, &[0xea1d03df]),
        ("cmp xzr, x2", |a| { a.cmp(31, 2, true); }, &[0xeb0203ff]),
        ("tst xzr, x2", |a| { a.tst(31, 2, true); }, &[0xea0203ff]),
        ("cmp x0, #0", |a| { a.cmp_imm(0, 0, true); }, &[0xf100001f]),
        ("cmp x30, #4095", |a| { a.cmp_imm(30, 4095, true); }, &[0xf13fffdf]),
        ("cmp x1, #1", |a| { a.cmp_imm(1, 1, true); }, &[0xf100043f]),
        ("cmp w0, w1", |a| { a.cmp(0, 1, false); }, &[0x6b01001f]),
        ("tst w0, w1", |a| { a.tst(0, 1, false); }, &[0x6a01001f]),
        ("cmp w30, w29", |a| { a.cmp(30, 29, false); }, &[0x6b1d03df]),
        ("tst w30, w29", |a| { a.tst(30, 29, false); }, &[0x6a1d03df]),
        ("cmp wzr, w2", |a| { a.cmp(31, 2, false); }, &[0x6b0203ff]),
        ("tst wzr, w2", |a| { a.tst(31, 2, false); }, &[0x6a0203ff]),
        ("cmp w0, #0", |a| { a.cmp_imm(0, 0, false); }, &[0x7100001f]),
        ("cmp w30, #4095", |a| { a.cmp_imm(30, 4095, false); }, &[0x713fffdf]),
        ("cmp w1, #1", |a| { a.cmp_imm(1, 1, false); }, &[0x7100043f]),
    ];
    check_encodings(cases);
}

    /// 64-bit add with a left-shifted register operand.
#[rustfmt::skip]
#[test]
fn shifted_register_add_matches_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("add x0, x1, x7, lsl #0", |a| { a.add_lsl(0, 1, 7, 0); }, &[0x8b070020]),
        ("add x2, x3, x7, lsl #4", |a| { a.add_lsl(2, 3, 7, 4); }, &[0x8b071062]),
        ("add x28, x29, x7, lsl #63", |a| { a.add_lsl(28, 29, 7, 63); }, &[0x8b07ffbc]),
        ("add x5, x6, x7, lsl #1", |a| { a.add_lsl(5, 6, 7, 1); }, &[0x8b0704c5]),
    ];
    check_encodings(cases);
}

    /// 64-bit lsl/lsr/asr by an immediate are bitfield-move aliases.
#[rustfmt::skip]
#[test]
fn immediate_shifts_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("lsl x0, x1, #1", |a| { a.lsl_imm(0, 1, 1); }, &[0xd37ff820]),
        ("lsr x0, x1, #1", |a| { a.lsr_imm(0, 1, 1); }, &[0xd341fc20]),
        ("asr x0, x1, #1", |a| { a.asr_imm(0, 1, 1); }, &[0x9341fc20]),
        ("lsl x2, x3, #4", |a| { a.lsl_imm(2, 3, 4); }, &[0xd37cec62]),
        ("lsr x2, x3, #4", |a| { a.lsr_imm(2, 3, 4); }, &[0xd344fc62]),
        ("asr x2, x3, #4", |a| { a.asr_imm(2, 3, 4); }, &[0x9344fc62]),
        ("lsl x28, x29, #63", |a| { a.lsl_imm(28, 29, 63); }, &[0xd34103bc]),
        ("lsr x28, x29, #63", |a| { a.lsr_imm(28, 29, 63); }, &[0xd37fffbc]),
        ("asr x28, x29, #63", |a| { a.asr_imm(28, 29, 63); }, &[0x937fffbc]),
        ("lsl x5, x6, #32", |a| { a.lsl_imm(5, 6, 32); }, &[0xd3607cc5]),
        ("lsr x5, x6, #32", |a| { a.lsr_imm(5, 6, 32); }, &[0xd360fcc5]),
        ("asr x5, x6, #32", |a| { a.asr_imm(5, 6, 32); }, &[0x9360fcc5]),
    ];
    check_encodings(cases);
}

    /// Uxtb/uxth/uxtw and sxtb/sxth/sxtw in both widths.
#[rustfmt::skip]
#[test]
fn zero_and_sign_extensions_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("uxtb w0, w1", |a| { a.uxtb(0, 1); }, &[0x53001c20]),
        ("uxth w0, w1", |a| { a.uxth(0, 1); }, &[0x53003c20]),
        ("sxtb x0, w1", |a| { a.sxtb(0, 1, true); }, &[0x93401c20]),
        ("sxtb w0, w1", |a| { a.sxtb(0, 1, false); }, &[0x13001c20]),
        ("sxth x0, w1", |a| { a.sxth(0, 1, true); }, &[0x93403c20]),
        ("sxth w0, w1", |a| { a.sxth(0, 1, false); }, &[0x13003c20]),
        ("sxtw x0, w1", |a| { a.sxtw(0, 1); }, &[0x93407c20]),
        ("mov w0, w1", |a| { a.uxtw(0, 1); }, &[0x2a0103e0]),
        ("uxtb w28, w29", |a| { a.uxtb(28, 29); }, &[0x53001fbc]),
        ("uxth w28, w29", |a| { a.uxth(28, 29); }, &[0x53003fbc]),
        ("sxtb x28, w29", |a| { a.sxtb(28, 29, true); }, &[0x93401fbc]),
        ("sxtb w28, w29", |a| { a.sxtb(28, 29, false); }, &[0x13001fbc]),
        ("sxth x28, w29", |a| { a.sxth(28, 29, true); }, &[0x93403fbc]),
        ("sxth w28, w29", |a| { a.sxth(28, 29, false); }, &[0x13003fbc]),
        ("sxtw x28, w29", |a| { a.sxtw(28, 29); }, &[0x93407fbc]),
        ("mov w28, w29", |a| { a.uxtw(28, 29); }, &[0x2a1d03fc]),
        ("uxtb w5, wzr", |a| { a.uxtb(5, 31); }, &[0x53001fe5]),
        ("uxth w5, wzr", |a| { a.uxth(5, 31); }, &[0x53003fe5]),
        ("sxtb x5, wzr", |a| { a.sxtb(5, 31, true); }, &[0x93401fe5]),
        ("sxtb w5, wzr", |a| { a.sxtb(5, 31, false); }, &[0x13001fe5]),
        ("sxth x5, wzr", |a| { a.sxth(5, 31, true); }, &[0x93403fe5]),
        ("sxth w5, wzr", |a| { a.sxth(5, 31, false); }, &[0x13003fe5]),
        ("sxtw x5, wzr", |a| { a.sxtw(5, 31); }, &[0x93407fe5]),
        ("mov w5, wzr", |a| { a.uxtw(5, 31); }, &[0x2a1f03e5]),
    ];
    check_encodings(cases);
}

    /// Ubfm/sbfm/bfm and the bfi alias.
#[rustfmt::skip]
#[test]
fn bitfield_moves_and_insert_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("ubfm x0, x1, #0, #7", |a| { a.ubfm(0, 1, 0, 7, true); }, &[0xd3401c20]),
        ("sbfm x0, x1, #0, #7", |a| { a.sbfm(0, 1, 0, 7, true); }, &[0x93401c20]),
        ("bfm x0, x1, #0, #7", |a| { a.bfm(0, 1, 0, 7, true); }, &[0xb3401c20]),
        ("ubfm x2, x3, #8, #15", |a| { a.ubfm(2, 3, 8, 15, true); }, &[0xd3483c62]),
        ("sbfm x2, x3, #8, #15", |a| { a.sbfm(2, 3, 8, 15, true); }, &[0x93483c62]),
        ("bfm x2, x3, #8, #15", |a| { a.bfm(2, 3, 8, 15, true); }, &[0xb3483c62]),
        ("ubfm x4, x5, #3, #9", |a| { a.ubfm(4, 5, 3, 9, true); }, &[0xd34324a4]),
        ("sbfm x4, x5, #3, #9", |a| { a.sbfm(4, 5, 3, 9, true); }, &[0x934324a4]),
        ("bfm x4, x5, #3, #9", |a| { a.bfm(4, 5, 3, 9, true); }, &[0xb34324a4]),
        ("bfi x0, x1, #8, #8", |a| { a.bfi(0, 1, 8, 8, true); }, &[0xb3781c20]),
        ("bfi x2, x3, #0, #16", |a| { a.bfi(2, 3, 0, 16, true); }, &[0xb3403c62]),
        ("bfi x4, x5, #16, #1", |a| { a.bfi(4, 5, 16, 1, true); }, &[0xb37000a4]),
        ("ubfm w0, w1, #0, #7", |a| { a.ubfm(0, 1, 0, 7, false); }, &[0x53001c20]),
        ("sbfm w0, w1, #0, #7", |a| { a.sbfm(0, 1, 0, 7, false); }, &[0x13001c20]),
        ("bfm w0, w1, #0, #7", |a| { a.bfm(0, 1, 0, 7, false); }, &[0x33001c20]),
        ("ubfm w2, w3, #8, #15", |a| { a.ubfm(2, 3, 8, 15, false); }, &[0x53083c62]),
        ("sbfm w2, w3, #8, #15", |a| { a.sbfm(2, 3, 8, 15, false); }, &[0x13083c62]),
        ("bfm w2, w3, #8, #15", |a| { a.bfm(2, 3, 8, 15, false); }, &[0x33083c62]),
        ("ubfm w4, w5, #3, #9", |a| { a.ubfm(4, 5, 3, 9, false); }, &[0x530324a4]),
        ("sbfm w4, w5, #3, #9", |a| { a.sbfm(4, 5, 3, 9, false); }, &[0x130324a4]),
        ("bfm w4, w5, #3, #9", |a| { a.bfm(4, 5, 3, 9, false); }, &[0x330324a4]),
        ("bfi w0, w1, #8, #8", |a| { a.bfi(0, 1, 8, 8, false); }, &[0x33181c20]),
        ("bfi w2, w3, #0, #16", |a| { a.bfi(2, 3, 0, 16, false); }, &[0x33003c62]),
        ("bfi w4, w5, #16, #1", |a| { a.bfi(4, 5, 16, 1, false); }, &[0x331000a4]),
    ];
    check_encodings(cases);
}

    /// The SHA-256 schedule instructions and the byte permutes behind the inline sha256msg1/2,
    /// palignr and pshufd.
#[rustfmt::skip]
#[test]
fn whole_vector_encodings_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("sha256su0 v0.4s, v1.4s", |a| { a.sha256su0(0, 1); }, &[0x5e282820]),
        ("sha256su1 v0.4s, v2.4s, v1.4s", |a| { a.sha256su1(0, 2, 1); }, &[0x5e016040]),
        ("ext v0.16b, v1.16b, v2.16b, #5", |a| { a.ext_16b(0, 1, 2, 5); }, &[0x6e022820]),
        ("movi v2.2d, #0", |a| { a.vec_zero(2); }, &[0x6f00e402]),
        ("movi v2.16b, #0x8f", |a| { a.vec_splat_byte(2, 0x8f); }, &[0x4f04e5e2]),
        ("mov v1.s[0], wzr", |a| { a.clear_low_dword(1); }, &[0x4e041fe1]),
        ("tbl v0.16b, {v1.16b}, v2.16b", |a| { a.tbl(0, 1, 2); }, &[0x4e020020]),
    ];
    check_encodings(cases);
}

    /// Movz/movk/movn, register mov and the sp-capable mov.
#[rustfmt::skip]
#[test]
fn move_encodings_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("movz x0, #0x0, lsl #0", |a| { a.movz(0, 0x0, 0); }, &[0xd2800000]),
        ("movk x0, #0x0, lsl #0", |a| { a.movk(0, 0x0, 0); }, &[0xf2800000]),
        ("movn x0, #0x0, lsl #0", |a| { a.movn(0, 0x0, 0); }, &[0x92800000]),
        ("movz x1, #0x1234, lsl #16", |a| { a.movz(1, 0x1234, 1); }, &[0xd2a24681]),
        ("movk x1, #0x1234, lsl #16", |a| { a.movk(1, 0x1234, 1); }, &[0xf2a24681]),
        ("movn x1, #0x1234, lsl #16", |a| { a.movn(1, 0x1234, 1); }, &[0x92a24681]),
        ("movz x28, #0xffff, lsl #48", |a| { a.movz(28, 0xffff, 3); }, &[0xd2fffffc]),
        ("movk x28, #0xffff, lsl #48", |a| { a.movk(28, 0xffff, 3); }, &[0xf2fffffc]),
        ("movn x28, #0xffff, lsl #48", |a| { a.movn(28, 0xffff, 3); }, &[0x92fffffc]),
        ("movz x5, #0xbeef, lsl #32", |a| { a.movz(5, 0xbeef, 2); }, &[0xd2d7dde5]),
        ("movk x5, #0xbeef, lsl #32", |a| { a.movk(5, 0xbeef, 2); }, &[0xf2d7dde5]),
        ("movn x5, #0xbeef, lsl #32", |a| { a.movn(5, 0xbeef, 2); }, &[0x92d7dde5]),
        ("mov x0, x1", |a| { a.mov(0, 1); }, &[0xaa0103e0]),
        ("mov x28, x29", |a| { a.mov(28, 29); }, &[0xaa1d03fc]),
        ("mov x2, xzr", |a| { a.mov(2, 31); }, &[0xaa1f03e2]),
        ("orr xzr, xzr, x5", |a| { a.mov(31, 5); }, &[0xaa0503ff]),
        ("mov x0, sp", |a| { a.mov_sp(0, 31); }, &[0x910003e0]),
        ("mov sp, x0", |a| { a.mov_sp(31, 0); }, &[0x9100001f]),
        ("mov x29, sp", |a| { a.mov_sp(29, 31); }, &[0x910003fd]),
    ];
    check_encodings(cases);
}

    /// Cset, csel for every condition and the cinc-on-equal helper.
#[rustfmt::skip]
#[test]
fn conditional_selects_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("cset x3, eq", |a| { a.cset(3, EQ); }, &[0x9a9f17e3]),
        ("csel x4, x5, x6, eq", |a| { a.csel(4, 5, 6, EQ); }, &[0x9a8600a4]),
        ("cset x3, ne", |a| { a.cset(3, NE); }, &[0x9a9f07e3]),
        ("csel x4, x5, x6, ne", |a| { a.csel(4, 5, 6, NE); }, &[0x9a8610a4]),
        ("cset x3, cs", |a| { a.cset(3, CS); }, &[0x9a9f37e3]),
        ("csel x4, x5, x6, cs", |a| { a.csel(4, 5, 6, CS); }, &[0x9a8620a4]),
        ("cset x3, cc", |a| { a.cset(3, CC); }, &[0x9a9f27e3]),
        ("csel x4, x5, x6, cc", |a| { a.csel(4, 5, 6, CC); }, &[0x9a8630a4]),
        ("cset x3, mi", |a| { a.cset(3, MI); }, &[0x9a9f57e3]),
        ("csel x4, x5, x6, mi", |a| { a.csel(4, 5, 6, MI); }, &[0x9a8640a4]),
        ("cset x3, pl", |a| { a.cset(3, PL); }, &[0x9a9f47e3]),
        ("csel x4, x5, x6, pl", |a| { a.csel(4, 5, 6, PL); }, &[0x9a8650a4]),
        ("cset x3, vs", |a| { a.cset(3, VS); }, &[0x9a9f77e3]),
        ("csel x4, x5, x6, vs", |a| { a.csel(4, 5, 6, VS); }, &[0x9a8660a4]),
        ("cset x3, vc", |a| { a.cset(3, VC); }, &[0x9a9f67e3]),
        ("csel x4, x5, x6, vc", |a| { a.csel(4, 5, 6, VC); }, &[0x9a8670a4]),
        ("cset x3, hi", |a| { a.cset(3, HI); }, &[0x9a9f97e3]),
        ("csel x4, x5, x6, hi", |a| { a.csel(4, 5, 6, HI); }, &[0x9a8680a4]),
        ("cset x3, ls", |a| { a.cset(3, LS); }, &[0x9a9f87e3]),
        ("csel x4, x5, x6, ls", |a| { a.csel(4, 5, 6, LS); }, &[0x9a8690a4]),
        ("cset x3, ge", |a| { a.cset(3, GE); }, &[0x9a9fb7e3]),
        ("csel x4, x5, x6, ge", |a| { a.csel(4, 5, 6, GE); }, &[0x9a86a0a4]),
        ("cset x3, lt", |a| { a.cset(3, LT); }, &[0x9a9fa7e3]),
        ("csel x4, x5, x6, lt", |a| { a.csel(4, 5, 6, LT); }, &[0x9a86b0a4]),
        ("cset x3, gt", |a| { a.cset(3, GT); }, &[0x9a9fd7e3]),
        ("csel x4, x5, x6, gt", |a| { a.csel(4, 5, 6, GT); }, &[0x9a86c0a4]),
        ("cset x3, le", |a| { a.cset(3, LE); }, &[0x9a9fc7e3]),
        ("csel x4, x5, x6, le", |a| { a.csel(4, 5, 6, LE); }, &[0x9a86d0a4]),
        ("cinc x1, x1, eq", |a| { a.cinc_eq(1, 1); }, &[0x9a811421]),
        ("csinc x7, x9, x9, ne", |a| { a.cinc_eq(7, 9); }, &[0x9a891527]),
    ];
    check_encodings(cases);
}

    /// Every width at offset zero, one element and the largest scaled offset, with sp as the base.
#[rustfmt::skip]
#[test]
fn unsigned_offset_loads_and_stores_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("ldr x0, [x1, #0]", |a| { a.ldr(0, 1, 0); }, &[0xf9400020]),
        ("ldr x0, [x1, #8]", |a| { a.ldr(0, 1, 8); }, &[0xf9400420]),
        ("ldr x0, [x1, #32760]", |a| { a.ldr(0, 1, 32760); }, &[0xf97ffc20]),
        ("ldr x0, [sp, #8]", |a| { a.ldr(0, 31, 8); }, &[0xf94007e0]),
        ("str x2, [x3, #0]", |a| { a.str(2, 3, 0); }, &[0xf9000062]),
        ("str x2, [x3, #16]", |a| { a.str(2, 3, 16); }, &[0xf9000862]),
        ("str x2, [x3, #32760]", |a| { a.str(2, 3, 32760); }, &[0xf93ffc62]),
        ("str x2, [sp, #8]", |a| { a.str(2, 31, 8); }, &[0xf90007e2]),
        ("ldr w4, [x5, #0]", |a| { a.ldr_w(4, 5, 0); }, &[0xb94000a4]),
        ("ldr w4, [x5, #4]", |a| { a.ldr_w(4, 5, 4); }, &[0xb94004a4]),
        ("ldr w4, [x5, #16380]", |a| { a.ldr_w(4, 5, 16380); }, &[0xb97ffca4]),
        ("ldr w4, [sp, #4]", |a| { a.ldr_w(4, 31, 4); }, &[0xb94007e4]),
        ("str w6, [x7, #0]", |a| { a.str_w(6, 7, 0); }, &[0xb90000e6]),
        ("str w6, [x7, #8]", |a| { a.str_w(6, 7, 8); }, &[0xb90008e6]),
        ("str w6, [x7, #16380]", |a| { a.str_w(6, 7, 16380); }, &[0xb93ffce6]),
        ("str w6, [sp, #4]", |a| { a.str_w(6, 31, 4); }, &[0xb90007e6]),
        ("ldrh w8, [x9, #0]", |a| { a.ldrh(8, 9, 0); }, &[0x79400128]),
        ("ldrh w8, [x9, #2]", |a| { a.ldrh(8, 9, 2); }, &[0x79400528]),
        ("ldrh w8, [x9, #8190]", |a| { a.ldrh(8, 9, 8190); }, &[0x797ffd28]),
        ("ldrh w8, [sp, #2]", |a| { a.ldrh(8, 31, 2); }, &[0x794007e8]),
        ("strh w10, [x11, #0]", |a| { a.strh(10, 11, 0); }, &[0x7900016a]),
        ("strh w10, [x11, #2]", |a| { a.strh(10, 11, 2); }, &[0x7900056a]),
        ("strh w10, [x11, #8190]", |a| { a.strh(10, 11, 8190); }, &[0x793ffd6a]),
        ("strh w10, [sp, #2]", |a| { a.strh(10, 31, 2); }, &[0x790007ea]),
        ("ldrb w12, [x13, #0]", |a| { a.ldrb(12, 13, 0); }, &[0x394001ac]),
        ("ldrb w12, [x13, #1]", |a| { a.ldrb(12, 13, 1); }, &[0x394005ac]),
        ("ldrb w12, [x13, #4095]", |a| { a.ldrb(12, 13, 4095); }, &[0x397ffdac]),
        ("ldrb w12, [sp, #1]", |a| { a.ldrb(12, 31, 1); }, &[0x394007ec]),
        ("strb w14, [x15, #0]", |a| { a.strb(14, 15, 0); }, &[0x390001ee]),
        ("strb w14, [x15, #1]", |a| { a.strb(14, 15, 1); }, &[0x390005ee]),
        ("strb w14, [x15, #4095]", |a| { a.strb(14, 15, 4095); }, &[0x393ffdee]),
        ("strb w14, [sp, #1]", |a| { a.strb(14, 31, 1); }, &[0x390007ee]),
    ];
    check_encodings(cases);
}

    /// The TLB fast path accesses: every size, signed and unsigned.
#[rustfmt::skip]
#[test]
fn register_offset_loads_and_stores_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("ldrb w0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 1, false); }, &[0x38626820]),
        ("ldrh w0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 2, false); }, &[0x78626820]),
        ("ldr w0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 4, false); }, &[0xb8626820]),
        ("ldr x0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 8, false); }, &[0xf8626820]),
        ("ldrsb x0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 1, true); }, &[0x38a26820]),
        ("ldrsh x0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 2, true); }, &[0x78a26820]),
        ("ldrsw x0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 4, true); }, &[0xb8a26820]),
        ("ldr x0, [x1, x2]", |a| { a.ldr_r(0, 1, 2, 8, true); }, &[0xf8626820]),
        ("strb w6, [x4, x5]", |a| { a.str_r(6, 4, 5, 1); }, &[0x38256886]),
        ("strh w6, [x4, x5]", |a| { a.str_r(6, 4, 5, 2); }, &[0x78256886]),
        ("str w6, [x4, x5]", |a| { a.str_r(6, 4, 5, 4); }, &[0xb8256886]),
        ("str x6, [x4, x5]", |a| { a.str_r(6, 4, 5, 8); }, &[0xf8256886]),
        ("ldr x28, [x29, x30]", |a| { a.ldr_r(28, 29, 30, 8, false); }, &[0xf87e6bbc]),
    ];
    check_encodings(cases);
}

    /// The prologue and epilogue forms with the smallest and largest offsets.
#[rustfmt::skip]
#[test]
fn pair_loads_and_stores_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("stp x29, x30, [sp, #-80]!", |a| { a.stp_pre(29, 30, SP, -80); }, &[0xa9bb7bfd]),
        ("stp x19, x20, [sp, #16]!", |a| { a.stp_pre(19, 20, SP, 16); }, &[0xa98153f3]),
        ("stp x22, x23, [sp, #504]!", |a| { a.stp_pre(22, 23, SP, 504); }, &[0xa99fdff6]),
        ("stp x0, x1, [sp, #-512]!", |a| { a.stp_pre(0, 1, SP, -512); }, &[0xa9a007e0]),
        ("stp x2, x3, [sp, #0]!", |a| { a.stp_pre(2, 3, SP, 0); }, &[0xa9800fe2]),
        ("stp x29, x30, [sp, #-80]", |a| { a.stp_off(29, 30, SP, -80); }, &[0xa93b7bfd]),
        ("stp x19, x20, [sp, #16]", |a| { a.stp_off(19, 20, SP, 16); }, &[0xa90153f3]),
        ("stp x22, x23, [sp, #504]", |a| { a.stp_off(22, 23, SP, 504); }, &[0xa91fdff6]),
        ("stp x0, x1, [sp, #-512]", |a| { a.stp_off(0, 1, SP, -512); }, &[0xa92007e0]),
        ("stp x2, x3, [sp, #0]", |a| { a.stp_off(2, 3, SP, 0); }, &[0xa9000fe2]),
        ("ldp x29, x30, [sp, #-80]", |a| { a.ldp_off(29, 30, SP, -80); }, &[0xa97b7bfd]),
        ("ldp x19, x20, [sp, #16]", |a| { a.ldp_off(19, 20, SP, 16); }, &[0xa94153f3]),
        ("ldp x22, x23, [sp, #504]", |a| { a.ldp_off(22, 23, SP, 504); }, &[0xa95fdff6]),
        ("ldp x0, x1, [sp, #-512]", |a| { a.ldp_off(0, 1, SP, -512); }, &[0xa96007e0]),
        ("ldp x2, x3, [sp, #0]", |a| { a.ldp_off(2, 3, SP, 0); }, &[0xa9400fe2]),
        ("ldp x29, x30, [sp], #80", |a| { a.ldp_post(29, 30, SP, 80); }, &[0xa8c57bfd]),
        ("ldp x19, x20, [sp], #504", |a| { a.ldp_post(19, 20, SP, 504); }, &[0xa8dfd3f3]),
        ("ldp x0, x1, [sp], #0", |a| { a.ldp_post(0, 1, SP, 0); }, &[0xa8c007e0]),
    ];
    check_encodings(cases);
}

    /// Ret, blr, br, mrs/msr NZCV.
#[rustfmt::skip]
#[test]
fn returns_indirect_branches_and_flag_transfers_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("ret", |a| { a.ret(); }, &[0xd65f03c0]),
        ("blr x16", |a| { a.blr(16); }, &[0xd63f0200]),
        ("blr x0", |a| { a.blr(0); }, &[0xd63f0000]),
        ("br x1", |a| { a.br(1); }, &[0xd61f0020]),
        ("br x30", |a| { a.br(30); }, &[0xd61f03c0]),
        ("mrs x23, nzcv", |a| { a.mrs_nzcv(23); }, &[0xd53b4217]),
        ("mrs x0, nzcv", |a| { a.mrs_nzcv(0); }, &[0xd53b4200]),
        ("msr nzcv, x23", |a| { a.msr_nzcv(23); }, &[0xd51b4217]),
        ("msr nzcv, x0", |a| { a.msr_nzcv(0); }, &[0xd51b4200]),
    ];
    check_encodings(cases);
}

    /// Values the TLB probe and ordinary masks use.
#[rustfmt::skip]
#[test]
fn and_with_bitmask_immediate_matches_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("and x1, x2, #0xff", |a| { assert!(a.and_imm(1, 2, 0xff, true)); }, &[0x92401c41]),
        ("and x1, x2, #0x3ff", |a| { assert!(a.and_imm(1, 2, 0x3ff, true)); }, &[0x92402441]),
        ("and x1, x2, #0xffffffff", |a| { assert!(a.and_imm(1, 2, 0xffffffff, true)); }, &[0x92407c41]),
        ("and x1, x2, #0xffff0000", |a| { assert!(a.and_imm(1, 2, 0xffff0000, true)); }, &[0x92703c41]),
        ("and x1, x2, #0x5555555555555555", |a| { assert!(a.and_imm(1, 2, 0x5555555555555555, true)); }, &[0x9200f041]),
        ("and x1, x2, #0x8000000000000001", |a| { assert!(a.and_imm(1, 2, 0x8000000000000001, true)); }, &[0x92410441]),
        ("and x1, x2, #0xfffffffffffff000", |a| { assert!(a.and_imm(1, 2, 0xfffffffffffff000, true)); }, &[0x9274cc41]),
        ("and x1, x2, #0xf0f0f0f0f0f0f0f0", |a| { assert!(a.and_imm(1, 2, 0xf0f0f0f0f0f0f0f0, true)); }, &[0x9204cc41]),
        ("and x1, x2, #0xff00ff00ff00ff", |a| { assert!(a.and_imm(1, 2, 0xff00ff00ff00ff, true)); }, &[0x92009c41]),
        ("and x1, x2, #0xffff0000ffff", |a| { assert!(a.and_imm(1, 2, 0xffff0000ffff, true)); }, &[0x92003c41]),
        ("and x1, x2, #0x7ffffffffffffffe", |a| { assert!(a.and_imm(1, 2, 0x7ffffffffffffffe, true)); }, &[0x927ff441]),
        ("and w3, w4, #0xff", |a| { assert!(a.and_imm(3, 4, 0xff, false)); }, &[0x12001c83]),
        ("and w3, w4, #0xffff0000", |a| { assert!(a.and_imm(3, 4, 0xffff0000, false)); }, &[0x12103c83]),
        ("and w3, w4, #0x55555555", |a| { assert!(a.and_imm(3, 4, 0x55555555, false)); }, &[0x1200f083]),
        ("and w3, w4, #0xf0f0f0f0", |a| { assert!(a.and_imm(3, 4, 0xf0f0f0f0, false)); }, &[0x1204cc83]),
        ("and w3, w4, #0x80000001", |a| { assert!(a.and_imm(3, 4, 0x80000001, false)); }, &[0x12010483]),
        ("and w3, w4, #0x3ff", |a| { assert!(a.and_imm(3, 4, 0x3ff, false)); }, &[0x12002483]),
        ("and w3, w4, #0x7ffffffe", |a| { assert!(a.and_imm(3, 4, 0x7ffffffe, false)); }, &[0x121f7483]),
        ("and w3, w4, #0xfffffff0", |a| { assert!(a.and_imm(3, 4, 0xfffffff0, false)); }, &[0x121c6c83]),
    ];
    check_encodings(cases);
}

    /// `(value, is64, N:immr:imms the assembler picks, or None when it rejects the value)`
#[rustfmt::skip]
const BITMASK_CASES: &[(u64, bool, Option<u32>)] = &[
    (0x0, false, None),
    (0x0, true, None),
    (0x1, false, Some(0x0)),
    (0x1, true, Some(0x1000)),
    (0x2, false, Some(0x7c0)),
    (0x2, true, Some(0x1fc0)),
    (0x3, false, Some(0x1)),
    (0x3, true, Some(0x1001)),
    (0x5, true, None),
    (0x6, false, Some(0x7c1)),
    (0x6, true, Some(0x1fc1)),
    (0x7, false, Some(0x2)),
    (0x7, true, Some(0x1002)),
    (0xe, false, Some(0x7c2)),
    (0xe, true, Some(0x1fc2)),
    (0x101, false, None),
    (0x101, true, None),
    (0x1234, false, None),
    (0x1234, true, None),
    (0x7fff, false, Some(0xe)),
    (0xfffe, false, Some(0x7ce)),
    (0xffff, false, Some(0xf)),
    (0x10000, false, Some(0x400)),
    (0x10001, false, Some(0x20)),
    (0x1fffe, false, Some(0x7cf)),
    (0x1ffff, false, Some(0x10)),
    (0x20002, false, Some(0x3e0)),
    (0x30000, false, Some(0x401)),
    (0x30003, false, Some(0x21)),
    (0x3fffe, false, Some(0x7d0)),
    (0x60006, false, Some(0x3e1)),
    (0x70000, false, Some(0x402)),
    (0x70007, false, Some(0x22)),
    (0xe000e, false, Some(0x3e2)),
    (0x7f007f, false, Some(0x26)),
    (0xfe00fe, false, Some(0x3e6)),
    (0xff00ff, false, Some(0x27)),
    (0x1000100, false, Some(0x220)),
    (0x1010101, false, Some(0x30)),
    (0x1fe01fe, false, Some(0x3e7)),
    (0x1ff01ff, false, Some(0x28)),
    (0x2020202, false, Some(0x1f0)),
    (0x3000300, false, Some(0x221)),
    (0x3030303, false, Some(0x31)),
    (0x3fe03fe, false, Some(0x3e8)),
    (0x4040404, false, Some(0x1b0)),
    (0x6060606, false, Some(0x1f1)),
    (0x7000700, false, Some(0x222)),
    (0x7070707, false, Some(0x32)),
    (0x8080808, false, Some(0x170)),
    (0xc0c0c0c, false, Some(0x1b1)),
    (0xe0e0e0e, false, Some(0x1f2)),
    (0xf0f0f0f, false, Some(0x33)),
    (0x10101010, false, Some(0x130)),
    (0x11111111, false, Some(0x38)),
    (0x18181818, false, Some(0x171)),
    (0x1c1c1c1c, false, Some(0x1b2)),
    (0x1e1e1e1e, false, Some(0x1f3)),
    (0x1f1f1f1f, false, Some(0x34)),
    (0x20000000, false, Some(0xc0)),
    (0x20002000, false, Some(0xe0)),
    (0x20202020, false, Some(0xf0)),
    (0x22222222, false, Some(0xf8)),
    (0x30303030, false, Some(0x131)),
    (0x33333333, false, Some(0x39)),
    (0x38383838, false, Some(0x172)),
    (0x3c3c3c3c, false, Some(0x1b3)),
    (0x3e3e3e3e, false, Some(0x1f4)),
    (0x3f3f3f3f, false, Some(0x35)),
    (0x3fff3fff, false, Some(0x2d)),
    (0x3fffffff, false, Some(0x1d)),
    (0x40404040, false, Some(0xb0)),
    (0x44444444, false, Some(0xb8)),
    (0x55555555, false, Some(0x3c)),
    (0x60000000, false, Some(0xc1)),
    (0x60006000, false, Some(0xe1)),
    (0x60606060, false, Some(0xf1)),
    (0x66666666, false, Some(0xf9)),
    (0x70707070, false, Some(0x132)),
    (0x77777777, false, Some(0x3a)),
    (0x78787878, false, Some(0x173)),
    (0x7c7c7c7c, false, Some(0x1b4)),
    (0x7e7e7e7e, false, Some(0x1f5)),
    (0x7f007f00, false, Some(0x226)),
    (0x7f7f7f7f, false, Some(0x36)),
    (0x7ffe7ffe, false, Some(0x3ed)),
    (0x7fff0000, false, Some(0x40e)),
    (0x7fff7fff, false, Some(0x2e)),
    (0x7ffffffe, false, Some(0x7dd)),
    (0x7fffffff, false, Some(0x1e)),
    (0x7fffffff, true, Some(0x101e)),
    (0x80000000, false, Some(0x40)),
    (0x80000001, false, Some(0x41)),
    (0x80000003, false, Some(0x42)),
    (0x80003fff, false, Some(0x4e)),
    (0x80007fff, false, Some(0x4f)),
    (0x80008000, false, Some(0x60)),
    (0x8000ffff, false, Some(0x50)),
    (0x80018001, false, Some(0x61)),
    (0x80038003, false, Some(0x62)),
    (0x803f803f, false, Some(0x66)),
    (0x807f807f, false, Some(0x67)),
    (0x80808080, false, Some(0x70)),
    (0x80ff80ff, false, Some(0x68)),
    (0x81818181, false, Some(0x71)),
    (0x83838383, false, Some(0x72)),
    (0x87878787, false, Some(0x73)),
    (0x88888888, false, Some(0x78)),
    (0x8f8f8f8f, false, Some(0x74)),
    (0x99999999, false, Some(0x79)),
    (0x9f9f9f9f, false, Some(0x75)),
    (0x9fff9fff, false, Some(0x6d)),
    (0x9fffffff, false, Some(0x5d)),
    (0xaaaaaaaa, false, Some(0x7c)),
    (0xbbbbbbbb, false, Some(0x7a)),
    (0xbfbfbfbf, false, Some(0x76)),
    (0xbfffbfff, false, Some(0x6e)),
    (0xbfffffff, false, Some(0x5e)),
    (0xc0c0c0c0, false, Some(0xb1)),
    (0xc1c1c1c1, false, Some(0xb2)),
    (0xc3c3c3c3, false, Some(0xb3)),
    (0xc7c7c7c7, false, Some(0xb4)),
    (0xcccccccc, false, Some(0xb9)),
    (0xcfcfcfcf, false, Some(0xb5)),
    (0xdddddddd, false, Some(0xba)),
    (0xdeadbeef, false, None),
    (0xdeadbeef, true, None),
    (0xdfdfdfdf, false, Some(0xb6)),
    (0xe0000000, false, Some(0xc2)),
    (0xe0000fff, false, Some(0xce)),
    (0xe0001fff, false, Some(0xcf)),
    (0xe0003fff, false, Some(0xd0)),
    (0xe000e000, false, Some(0xe2)),
    (0xe00fe00f, false, Some(0xe6)),
    (0xe01fe01f, false, Some(0xe7)),
    (0xe03fe03f, false, Some(0xe8)),
    (0xe0e0e0e0, false, Some(0xf2)),
    (0xe1e1e1e1, false, Some(0xf3)),
    (0xe3e3e3e3, false, Some(0xf4)),
    (0xe7e7e7e7, false, Some(0xf5)),
    (0xe7ffe7ff, false, Some(0xed)),
    (0xe7ffffff, false, Some(0xdd)),
    (0xeeeeeeee, false, Some(0xfa)),
    (0xefefefef, false, Some(0xf6)),
    (0xefffefff, false, Some(0xee)),
    (0xefffffff, false, Some(0xde)),
    (0xf0f0f0f0, false, Some(0x133)),
    (0xf1f1f1f1, false, Some(0x134)),
    (0xf3f3f3f3, false, Some(0x135)),
    (0xf7f7f7f7, false, Some(0x136)),
    (0xf8f8f8f8, false, Some(0x174)),
    (0xf9f9f9f9, false, Some(0x175)),
    (0xfbfbfbfb, false, Some(0x176)),
    (0xfcfcfcfc, false, Some(0x1b5)),
    (0xfdfdfdfd, false, Some(0x1b6)),
    (0xfefefefe, false, Some(0x1f6)),
    (0xff00ff00, false, Some(0x227)),
    (0xff01ff01, false, Some(0x228)),
    (0xff3fff3f, false, Some(0x22d)),
    (0xff7fff7f, false, Some(0x22e)),
    (0xfffefffe, false, Some(0x3ee)),
    (0xffff0000, false, Some(0x40f)),
    (0xffff0001, false, Some(0x410)),
    (0xffff3fff, false, Some(0x41d)),
    (0xffff7fff, false, Some(0x41e)),
    (0xfffffffe, false, Some(0x7de)),
    (0xfffffffe, true, Some(0x1fde)),
    (0xffffffff, false, None),
    (0xffffffff, true, Some(0x101f)),
    (0x100000000, true, Some(0x1800)),
    (0x100000001, true, Some(0x0)),
    (0x1fffffffe, true, Some(0x1fdf)),
    (0x1ffffffff, true, Some(0x1020)),
    (0x200000002, true, Some(0x7c0)),
    (0x300000000, true, Some(0x1801)),
    (0x300000003, true, Some(0x1)),
    (0x3fffffffe, true, Some(0x1fe0)),
    (0x600000006, true, Some(0x7c1)),
    (0x700000000, true, Some(0x1802)),
    (0x700000007, true, Some(0x2)),
    (0xe0000000e, true, Some(0x7c2)),
    (0x7fff00007fff, true, Some(0xe)),
    (0xfffe0000fffe, true, Some(0x7ce)),
    (0xffff0000ffff, true, Some(0xf)),
    (0x1000000010000, true, Some(0x400)),
    (0x1000100010001, true, Some(0x20)),
    (0x1fffe0001fffe, true, Some(0x7cf)),
    (0x1ffff0001ffff, true, Some(0x10)),
    (0x2000200020002, true, Some(0x3e0)),
    (0x3000000030000, true, Some(0x401)),
    (0x3000300030003, true, Some(0x21)),
    (0x3fffe0003fffe, true, Some(0x7d0)),
    (0x6000600060006, true, Some(0x3e1)),
    (0x7000000070000, true, Some(0x402)),
    (0x7000700070007, true, Some(0x22)),
    (0xe000e000e000e, true, Some(0x3e2)),
    (0x7f007f007f007f, true, Some(0x26)),
    (0xfe00fe00fe00fe, true, Some(0x3e6)),
    (0xff00ff00ff00ff, true, Some(0x27)),
    (0x100010001000100, true, Some(0x220)),
    (0x101010101010101, true, Some(0x30)),
    (0x123456789abcdef, true, None),
    (0x1fe01fe01fe01fe, true, Some(0x3e7)),
    (0x1ff01ff01ff01ff, true, Some(0x28)),
    (0x202020202020202, true, Some(0x1f0)),
    (0x300030003000300, true, Some(0x221)),
    (0x303030303030303, true, Some(0x31)),
    (0x3fe03fe03fe03fe, true, Some(0x3e8)),
    (0x404040404040404, true, Some(0x1b0)),
    (0x606060606060606, true, Some(0x1f1)),
    (0x700070007000700, true, Some(0x222)),
    (0x707070707070707, true, Some(0x32)),
    (0x808080808080808, true, Some(0x170)),
    (0xc0c0c0c0c0c0c0c, true, Some(0x1b1)),
    (0xe0e0e0e0e0e0e0e, true, Some(0x1f2)),
    (0xf0f0f0f0f0f0f0f, true, Some(0x33)),
    (0x1010101010101010, true, Some(0x130)),
    (0x1111111111111111, true, Some(0x38)),
    (0x1818181818181818, true, Some(0x171)),
    (0x1c1c1c1c1c1c1c1c, true, Some(0x1b2)),
    (0x1e1e1e1e1e1e1e1e, true, Some(0x1f3)),
    (0x1f1f1f1f1f1f1f1f, true, Some(0x34)),
    (0x2000000000000000, true, Some(0x10c0)),
    (0x2000000020000000, true, Some(0xc0)),
    (0x2000200020002000, true, Some(0xe0)),
    (0x2020202020202020, true, Some(0xf0)),
    (0x2222222222222222, true, Some(0xf8)),
    (0x3030303030303030, true, Some(0x131)),
    (0x3333333333333333, true, Some(0x39)),
    (0x3838383838383838, true, Some(0x172)),
    (0x3c3c3c3c3c3c3c3c, true, Some(0x1b3)),
    (0x3e3e3e3e3e3e3e3e, true, Some(0x1f4)),
    (0x3f3f3f3f3f3f3f3f, true, Some(0x35)),
    (0x3fff3fff3fff3fff, true, Some(0x2d)),
    (0x3fffffff3fffffff, true, Some(0x1d)),
    (0x3fffffffffffffff, true, Some(0x103d)),
    (0x4040404040404040, true, Some(0xb0)),
    (0x4444444444444444, true, Some(0xb8)),
    (0x5555555555555555, true, Some(0x3c)),
    (0x6000000000000000, true, Some(0x10c1)),
    (0x6000000060000000, true, Some(0xc1)),
    (0x6000600060006000, true, Some(0xe1)),
    (0x6060606060606060, true, Some(0xf1)),
    (0x6666666666666666, true, Some(0xf9)),
    (0x7070707070707070, true, Some(0x132)),
    (0x7777777777777777, true, Some(0x3a)),
    (0x7878787878787878, true, Some(0x173)),
    (0x7c7c7c7c7c7c7c7c, true, Some(0x1b4)),
    (0x7e7e7e7e7e7e7e7e, true, Some(0x1f5)),
    (0x7f007f007f007f00, true, Some(0x226)),
    (0x7f7f7f7f7f7f7f7f, true, Some(0x36)),
    (0x7ffe7ffe7ffe7ffe, true, Some(0x3ed)),
    (0x7fff00007fff0000, true, Some(0x40e)),
    (0x7fff7fff7fff7fff, true, Some(0x2e)),
    (0x7ffffffe7ffffffe, true, Some(0x7dd)),
    (0x7fffffff00000000, true, Some(0x181e)),
    (0x7fffffff7fffffff, true, Some(0x1e)),
    (0x7ffffffffffffffe, true, Some(0x1ffd)),
    (0x7fffffffffffffff, true, Some(0x103e)),
    (0x8000000000000000, true, Some(0x1040)),
    (0x8000000000000001, true, Some(0x1041)),
    (0x8000000000000003, true, Some(0x1042)),
    (0x8000000000000004, true, None),
    (0x800000003fffffff, true, Some(0x105e)),
    (0x800000007fffffff, true, Some(0x105f)),
    (0x8000000080000000, true, Some(0x40)),
    (0x80000000ffffffff, true, Some(0x1060)),
    (0x8000000180000001, true, Some(0x41)),
    (0x8000000380000003, true, Some(0x42)),
    (0x80003fff80003fff, true, Some(0x4e)),
    (0x80007fff80007fff, true, Some(0x4f)),
    (0x8000800080008000, true, Some(0x60)),
    (0x8000ffff8000ffff, true, Some(0x50)),
    (0x8001800180018001, true, Some(0x61)),
    (0x8003800380038003, true, Some(0x62)),
    (0x803f803f803f803f, true, Some(0x66)),
    (0x807f807f807f807f, true, Some(0x67)),
    (0x8080808080808080, true, Some(0x70)),
    (0x80ff80ff80ff80ff, true, Some(0x68)),
    (0x8181818181818181, true, Some(0x71)),
    (0x8383838383838383, true, Some(0x72)),
    (0x8787878787878787, true, Some(0x73)),
    (0x8888888888888888, true, Some(0x78)),
    (0x8f8f8f8f8f8f8f8f, true, Some(0x74)),
    (0x9999999999999999, true, Some(0x79)),
    (0x9f9f9f9f9f9f9f9f, true, Some(0x75)),
    (0x9fff9fff9fff9fff, true, Some(0x6d)),
    (0x9fffffff9fffffff, true, Some(0x5d)),
    (0x9fffffffffffffff, true, Some(0x107d)),
    (0xaaaaaaaaaaaaaaaa, true, Some(0x7c)),
    (0xbbbbbbbbbbbbbbbb, true, Some(0x7a)),
    (0xbfbfbfbfbfbfbfbf, true, Some(0x76)),
    (0xbfffbfffbfffbfff, true, Some(0x6e)),
    (0xbfffffffbfffffff, true, Some(0x5e)),
    (0xbfffffffffffffff, true, Some(0x107e)),
    (0xc0c0c0c0c0c0c0c0, true, Some(0xb1)),
    (0xc1c1c1c1c1c1c1c1, true, Some(0xb2)),
    (0xc3c3c3c3c3c3c3c3, true, Some(0xb3)),
    (0xc7c7c7c7c7c7c7c7, true, Some(0xb4)),
    (0xcccccccccccccccc, true, Some(0xb9)),
    (0xcfcfcfcfcfcfcfcf, true, Some(0xb5)),
    (0xdddddddddddddddd, true, Some(0xba)),
    (0xdfdfdfdfdfdfdfdf, true, Some(0xb6)),
    (0xe000000000000000, true, Some(0x10c2)),
    (0xe00000000fffffff, true, Some(0x10de)),
    (0xe00000001fffffff, true, Some(0x10df)),
    (0xe00000003fffffff, true, Some(0x10e0)),
    (0xe0000000e0000000, true, Some(0xc2)),
    (0xe0000fffe0000fff, true, Some(0xce)),
    (0xe0001fffe0001fff, true, Some(0xcf)),
    (0xe0003fffe0003fff, true, Some(0xd0)),
    (0xe000e000e000e000, true, Some(0xe2)),
    (0xe00fe00fe00fe00f, true, Some(0xe6)),
    (0xe01fe01fe01fe01f, true, Some(0xe7)),
    (0xe03fe03fe03fe03f, true, Some(0xe8)),
    (0xe0e0e0e0e0e0e0e0, true, Some(0xf2)),
    (0xe1e1e1e1e1e1e1e1, true, Some(0xf3)),
    (0xe3e3e3e3e3e3e3e3, true, Some(0xf4)),
    (0xe7e7e7e7e7e7e7e7, true, Some(0xf5)),
    (0xe7ffe7ffe7ffe7ff, true, Some(0xed)),
    (0xe7ffffffe7ffffff, true, Some(0xdd)),
    (0xe7ffffffffffffff, true, Some(0x10fd)),
    (0xeeeeeeeeeeeeeeee, true, Some(0xfa)),
    (0xefefefefefefefef, true, Some(0xf6)),
    (0xefffefffefffefff, true, Some(0xee)),
    (0xefffffffefffffff, true, Some(0xde)),
    (0xefffffffffffffff, true, Some(0x10fe)),
    (0xf0f0f0f0f0f0f0f0, true, Some(0x133)),
    (0xf1f1f1f1f1f1f1f1, true, Some(0x134)),
    (0xf3f3f3f3f3f3f3f3, true, Some(0x135)),
    (0xf7f7f7f7f7f7f7f7, true, Some(0x136)),
    (0xf8f8f8f8f8f8f8f8, true, Some(0x174)),
    (0xf9f9f9f9f9f9f9f9, true, Some(0x175)),
    (0xfbfbfbfbfbfbfbfb, true, Some(0x176)),
    (0xfcfcfcfcfcfcfcfc, true, Some(0x1b5)),
    (0xfdfdfdfdfdfdfdfd, true, Some(0x1b6)),
    (0xfefefefefefefefe, true, Some(0x1f6)),
    (0xff00ff00ff00ff00, true, Some(0x227)),
    (0xff01ff01ff01ff01, true, Some(0x228)),
    (0xff3fff3fff3fff3f, true, Some(0x22d)),
    (0xff7fff7fff7fff7f, true, Some(0x22e)),
    (0xfffefffefffefffe, true, Some(0x3ee)),
    (0xffff0000ffff0000, true, Some(0x40f)),
    (0xffff0001ffff0001, true, Some(0x410)),
    (0xffff3fffffff3fff, true, Some(0x41d)),
    (0xffff7fffffff7fff, true, Some(0x41e)),
    (0xfffffffefffffffe, true, Some(0x7de)),
    (0xffffffff00000000, true, Some(0x181f)),
    (0xffffffff00000001, true, Some(0x1820)),
    (0xffffffff3fffffff, true, Some(0x183d)),
    (0xffffffff7fffffff, true, Some(0x183e)),
    (0xfffffffffffffffe, true, Some(0x1ffe)),
    (0xffffffffffffffff, true, None),
];

    #[test]
    fn forward_branch_is_patched_with_the_distance_in_words() {
        // b 1f; nop; nop; 1: ret
        let expected: &[u32] = &[0x14000003, 0xd503201f, 0xd503201f, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b(target);
                nop(a);
                nop(a);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn backward_branch_is_patched_with_a_negative_distance() {
        // 1: nop; nop; b 1b
        let expected: &[u32] = &[0xd503201f, 0xd503201f, 0x17fffffe];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                nop(a);
                nop(a);
                a.b(target);
            }),
            expected
        );
    }

    #[test]
    fn forward_conditional_branch_carries_the_condition_and_distance() {
        // b.eq 1f; nop; 1: ret
        let expected: &[u32] = &[0x54000040, 0xd503201f, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b_cond(EQ, target);
                nop(a);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn backward_conditional_branch_sign_wraps_into_19_bits() {
        // 1: nop; b.ne 1b
        let expected: &[u32] = &[0xd503201f, 0x54ffffe1];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                nop(a);
                a.b_cond(NE, target);
            }),
            expected
        );
    }

    #[test]
    fn forward_cbz_on_a_64_bit_register() {
        // cbz x3, 1f; nop; nop; nop; 1: ret
        let expected: &[u32] = &[0xb4000083, 0xd503201f, 0xd503201f, 0xd503201f, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.cbz(3, target, true);
                nop(a);
                nop(a);
                nop(a);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn forward_cbnz_on_a_32_bit_register() {
        // cbnz w4, 1f; nop; 1: ret
        let expected: &[u32] = &[0x35000044, 0xd503201f, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.cbnz(4, target, false);
                nop(a);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn backward_cbz_on_a_32_bit_register() {
        // 1: nop; cbz w5, 1b
        let expected: &[u32] = &[0xd503201f, 0x34ffffe5];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                nop(a);
                a.cbz(5, target, false);
            }),
            expected
        );
    }

    #[test]
    fn backward_cbnz_on_a_64_bit_register() {
        // 1: nop; nop; cbnz x6, 1b
        let expected: &[u32] = &[0xd503201f, 0xd503201f, 0xb5ffffc6];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                nop(a);
                nop(a);
                a.cbnz(6, target, true);
            }),
            expected
        );
    }

    #[test]
    fn branch_to_the_next_instruction_has_distance_one() {
        // b 1f; 1: ret
        let expected: &[u32] = &[0x14000001, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b(target);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn branch_to_itself_has_distance_zero() {
        // 1: b 1b
        let expected: &[u32] = &[0x14000000];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                a.b(target);
            }),
            expected
        );
    }

    #[test]
    fn a_label_bound_after_the_last_instruction_is_the_end_of_the_code() {
        // b 1f; nop; 1:
        let expected: &[u32] = &[0x14000002, 0xd503201f];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b(target);
                nop(a);
                a.bind(target);
            }),
            expected
        );
    }

    #[test]
    fn two_branches_can_share_a_label() {
        // b.lt 1f; b.ge 1f; nop; 1: ret
        let expected: &[u32] = &[0x5400006b, 0x5400004a, 0xd503201f, 0xd65f03c0];
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b_cond(LT, target);
                a.b_cond(GE, target);
                nop(a);
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn a_forward_branch_over_a_thousand_instructions() {
        // b 1f; 1000 x nop; 1: ret
        let expected: Vec<u32> =
            [&[0x140003e9][..], &[NOP_WORD; 1000][..], &[0xd65f03c0][..]].concat();
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b(target);
                for _ in 0..1000 {
                    nop(a);
                }
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn a_backward_branch_over_a_thousand_instructions() {
        // 1: 1000 x nop; b 1b
        let expected: Vec<u32> = [&[][..], &[NOP_WORD; 1000][..], &[0x17fffc18][..]].concat();
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.bind(target);
                for _ in 0..1000 {
                    nop(a);
                }
                a.b(target);
            }),
            expected
        );
    }

    #[test]
    fn a_conditional_branch_over_a_thousand_instructions() {
        // b.hi 1f; 1000 x nop; 1: ret
        let expected: Vec<u32> =
            [&[0x54007d28][..], &[NOP_WORD; 1000][..], &[0xd65f03c0][..]].concat();
        assert_eq!(
            assemble(|a| {
                let target = a.label();
                a.b_cond(HI, target);
                for _ in 0..1000 {
                    nop(a);
                }
                a.bind(target);
                a.ret();
            }),
            expected
        );
    }

    #[test]
    fn bitmask_encoder_agrees_with_the_assembler_on_every_case() {
        for &(value, is64, expected) in BITMASK_CASES {
            assert_eq!(
                encode_bitmask(value, is64),
                expected,
                "{value:#x} ({} bit)",
                if is64 { 64 } else { 32 }
            );
        }
    }

    /// The architecture's `DecodeBitMasks` for a logical immediate: the value N:immr:imms stands for.
    /// Returns `None` for the reserved encodings and for the non-canonical ones whose `immr` has bits
    /// above the element size (hardware ignores them, assemblers write zeros).
    fn decode_bitmask(n: u32, immr: u32, imms: u32, is64: bool) -> Option<u64> {
        let combined = (n << 6) | (!imms & 0x3f);
        if combined == 0 {
            return None;
        }
        let highest_bit = 31 - combined.leading_zeros();
        let element_size = 1u32 << highest_bit;
        if element_size < 2 || (!is64 && element_size > 32) {
            return None;
        }
        let levels = element_size - 1;
        let run_length = (imms & levels) + 1;
        if run_length == element_size {
            return None;
        }
        if immr & !levels != 0 {
            return None;
        }
        let rotation = immr;
        let run = (1u64 << run_length) - 1;
        let element_mask = if element_size == 64 {
            u64::MAX
        } else {
            (1u64 << element_size) - 1
        };
        let rotated = if rotation == 0 {
            run
        } else {
            ((run >> rotation) | (run << (element_size - rotation))) & element_mask
        };
        let mut value = 0u64;
        let register_bits = if is64 { 64 } else { 32 };
        let mut shift = 0;
        while shift < register_bits {
            value |= rotated << shift;
            shift += element_size;
        }
        Some(value)
    }

    #[test]
    fn every_valid_64_bit_logical_immediate_round_trips_through_the_encoder() {
        let mut valid = 0;
        for n in 0..2 {
            for immr in 0..64 {
                for imms in 0..64 {
                    let Some(value) = decode_bitmask(n, immr, imms, true) else {
                        continue;
                    };
                    valid += 1;
                    let field = (n << 12) | (immr << 6) | imms;
                    assert_eq!(encode_bitmask(value, true), Some(field), "{value:#x}");
                }
            }
        }
        assert_eq!(
            valid, 5334,
            "the architecture defines 5334 distinct 64-bit logical immediates"
        );
    }

    #[test]
    fn every_valid_32_bit_logical_immediate_round_trips_through_the_encoder() {
        let mut valid = 0;
        for immr in 0..32 {
            for imms in 0..64 {
                let Some(value) = decode_bitmask(0, immr, imms, false) else {
                    continue;
                };
                valid += 1;
                let field = (immr << 6) | imms;
                assert_eq!(encode_bitmask(value, false), Some(field), "{value:#x}");
            }
        }
        assert_eq!(
            valid, 1302,
            "the architecture defines 1302 distinct 32-bit logical immediates"
        );
    }

    #[test]
    fn bitmask_encoder_rejects_zero_and_all_ones_in_both_widths() {
        for is64 in [true, false] {
            assert_eq!(encode_bitmask(0, is64), None);
        }
        assert_eq!(encode_bitmask(u64::MAX, true), None);
        assert_eq!(encode_bitmask(0xffff_ffff, false), None);
    }

    #[test]
    fn bitmask_encoder_ignores_the_upper_half_for_32_bit_operations() {
        assert_eq!(
            encode_bitmask(0xdead_beef_0000_00ff, false),
            encode_bitmask(0xff, false)
        );
        assert_eq!(
            encode_bitmask(0xffff_ffff_0000_0000, false),
            None,
            "only zeros remain"
        );
    }

    #[test]
    fn bitmask_encoder_rejects_values_that_are_not_a_rotated_run() {
        for value in [0x1234u64, 0x5, 0x101, 0xdead_beef, 0x0123_4567_89ab_cdef] {
            assert_eq!(encode_bitmask(value, true), None, "{value:#x}");
        }
    }

    #[test]
    fn and_imm_emits_nothing_and_returns_false_for_an_unencodable_value() {
        for value in [0u64, u64::MAX, 0x1234] {
            let mut a = Asm::new(BASE_ADDRESS);
            assert!(!a.and_imm(0, 1, value, true), "{value:#x}");
            assert!(a.code.is_empty(), "{value:#x}");
        }
    }

    #[test]
    fn the_tlb_probe_constants_are_encodable_bitmask_immediates() {
        // The inline probe masks the slot index and the page address with these (emit.rs).
        let mut a = Asm::new(BASE_ADDRESS);
        assert!(a.and_imm(3, 3, (crate::cpu::TLB_ENTRIES_PER_MODE - 1) as u64, true));
        assert!(a.and_imm(5, 2, !((1u64 << 12) - 1), true));
        assert_eq!(a.code.len(), 2);
    }

    #[test]
    fn here_advances_four_bytes_per_instruction_from_the_base() {
        let mut a = Asm::new(BASE_ADDRESS);
        assert_eq!(a.here(), BASE_ADDRESS);
        nop(&mut a);
        nop(&mut a);
        assert_eq!(a.here(), BASE_ADDRESS + 8);
    }

    #[test]
    fn absolute_branches_encode_the_distance_to_the_target() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.b_abs(BASE_ADDRESS + 0x100);
        assert_eq!(a.code, [0x1400_0000 | (0x100 / 4)]);
        let mut a = Asm::new(BASE_ADDRESS);
        nop(&mut a);
        a.b_abs(BASE_ADDRESS);
        assert_eq!(
            a.code[1],
            0x1400_0000 | (0x03ff_ffff & (-1i32) as u32),
            "back to the start, one word away"
        );
    }

    #[test]
    fn absolute_branches_reach_just_under_128_mib_in_both_directions() {
        const LIMIT_BYTES: u64 = 128 << 20;
        let mut a = Asm::new(BASE_ADDRESS);
        a.b_abs(BASE_ADDRESS + LIMIT_BYTES - 4);
        assert_eq!(a.code[0], 0x1400_0000 | 0x01ff_ffff);
        let mut a = Asm::new(BASE_ADDRESS);
        a.b_abs(BASE_ADDRESS - LIMIT_BYTES + 4);
        assert_eq!(a.code[0], 0x1400_0000 | 0x0200_0001);
    }

    #[test]
    #[should_panic(expected = "branch out of range")]
    fn absolute_branch_beyond_128_mib_forward_panics() {
        Asm::new(BASE_ADDRESS).b_abs(BASE_ADDRESS + (128 << 20));
    }

    #[test]
    #[should_panic(expected = "branch out of range")]
    fn absolute_branch_beyond_128_mib_backward_panics() {
        Asm::new(BASE_ADDRESS).b_abs(BASE_ADDRESS - (128 << 20));
    }

    #[test]
    #[should_panic(expected = "unbound label")]
    fn finishing_with_a_branch_to_an_unbound_label_panics() {
        let mut a = Asm::new(BASE_ADDRESS);
        let never_bound = a.label();
        a.b(never_bound);
        a.finish();
    }

    #[test]
    fn finish_patches_each_branch_once() {
        let mut a = Asm::new(BASE_ADDRESS);
        let target = a.label();
        a.b(target);
        a.bind(target);
        a.finish();
        let patched = a.code.clone();
        a.finish();
        assert_eq!(
            a.code, patched,
            "a second finish must not or the offsets in again"
        );
    }

    #[test]
    fn an_unreferenced_label_needs_no_binding() {
        let mut a = Asm::new(BASE_ADDRESS);
        let _unused = a.label();
        a.ret();
        a.finish();
        assert_eq!(a.code, [0xd65f_03c0]);
    }

    #[test]
    fn immediate_arithmetic_accepts_the_largest_12_bit_value() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.add_imm(0, 1, 4095, true);
        a.sub_imm(0, 1, 4095, false);
        assert_eq!(a.code, [0x913f_fc20, 0x513f_fc20]);
    }

    #[test]
    #[should_panic]
    fn immediate_add_rejects_4096() {
        Asm::new(BASE_ADDRESS).add_imm(0, 1, 4096, true);
    }

    #[test]
    #[should_panic]
    fn immediate_subtract_rejects_4096() {
        Asm::new(BASE_ADDRESS).sub_imm(0, 1, 4096, true);
    }

    #[test]
    #[should_panic]
    fn flag_setting_immediate_forms_reject_4096() {
        Asm::new(BASE_ADDRESS).subs_imm(31, 1, 4096, true);
    }

    #[test]
    #[should_panic(expected = "bad ldst offset")]
    fn eight_byte_load_rejects_an_offset_past_the_scaled_immediate() {
        Asm::new(BASE_ADDRESS).ldr(0, 1, 32768);
    }

    #[test]
    #[should_panic(expected = "bad ldst offset")]
    fn eight_byte_store_rejects_an_unaligned_offset() {
        Asm::new(BASE_ADDRESS).str(0, 1, 4);
    }

    #[test]
    #[should_panic(expected = "bad ldst offset")]
    fn four_byte_load_rejects_an_offset_past_the_scaled_immediate() {
        Asm::new(BASE_ADDRESS).ldr_w(0, 1, 16384);
    }

    #[test]
    #[should_panic(expected = "bad ldst offset")]
    fn halfword_load_rejects_an_odd_offset() {
        Asm::new(BASE_ADDRESS).ldrh(0, 1, 1);
    }

    #[test]
    #[should_panic(expected = "bad ldst offset")]
    fn byte_store_rejects_an_offset_past_4095() {
        Asm::new(BASE_ADDRESS).strb(0, 1, 4096);
    }

    #[test]
    fn mov_imm_picks_movn_when_the_top_48_bits_are_ones() {
        let mut a = Asm::new(BASE_ADDRESS);
        a.mov_imm(0, u64::MAX);
        a.mov_imm(1, u64::MAX - 1);
        a.mov_imm(2, 0xffff_ffff_ffff_8000);
        assert_eq!(a.code, [0x9280_0000, 0x9280_0021, 0x928f_ffe2]);
    }

    #[test]
    fn mov_imm_uses_one_instruction_per_nonzero_halfword() {
        for (value, instructions) in [
            (0u64, 1),
            (1, 1),
            (0xffff, 1),
            (0x1_0000, 1),
            (0x1_0001, 2),
            (0xffff_0000_0000, 1),
            (0x1234_0000_5678, 2),
            (0x1234_5678_9abc_def0, 4),
            (0x8000_0000_0000_0000, 1),
        ] {
            let mut a = Asm::new(BASE_ADDRESS);
            a.mov_imm(0, value);
            assert_eq!(a.code.len(), instructions, "{value:#x}");
        }
    }

    #[test]
    fn mov_imm_materialises_every_pattern_of_interesting_halfwords() {
        const HALFWORDS: [u64; 4] = [0, 0xffff, 1, 0x8000];
        for a in HALFWORDS {
            for b in HALFWORDS {
                for c in HALFWORDS {
                    for d in HALFWORDS {
                        let value = a | b << 16 | c << 32 | d << 48;
                        assert_eq!(run(|asm| asm.mov_imm(0, value), 0, 0), value, "{value:#x}");
                    }
                }
            }
        }
    }

    #[test]
    fn immediate_shifts_and_extensions_compute_the_right_values_natively() {
        assert_eq!(run(|a| a.lsl_imm(0, 0, 63), 1, 0), 1 << 63);
        assert_eq!(run(|a| a.lsr_imm(0, 0, 63), 1 << 63, 0), 1);
        assert_eq!(run(|a| a.asr_imm(0, 0, 63), 1 << 63, 0), u64::MAX);
        assert_eq!(run(|a| a.uxtb(0, 0), 0x1234_5678_9abc_def0, 0), 0xf0);
        assert_eq!(run(|a| a.uxth(0, 0), 0x1234_5678_9abc_def0, 0), 0xdef0);
        assert_eq!(run(|a| a.uxtw(0, 0), 0x1234_5678_9abc_def0, 0), 0x9abc_def0);
        assert_eq!(run(|a| a.sxtb(0, 0, true), 0x80, 0), 0xffff_ffff_ffff_ff80);
        assert_eq!(
            run(|a| a.sxtb(0, 0, false), 0x80, 0),
            0xffff_ff80,
            "the 32-bit form clears the upper half"
        );
        assert_eq!(
            run(|a| a.sxth(0, 0, true), 0x8000, 0),
            0xffff_ffff_ffff_8000
        );
        assert_eq!(run(|a| a.sxtw(0, 0), 0x8000_0000, 0), 0xffff_ffff_8000_0000);
    }

    #[test]
    fn bit_insert_replaces_only_the_selected_field() {
        assert_eq!(
            run(|a| a.bfi(0, 1, 8, 8, true), 0xffff_ffff_ffff_ffff, 0x12),
            0xffff_ffff_ffff_12ff
        );
        assert_eq!(
            run(|a| a.bfi(0, 1, 16, 16, false), 0xffff_ffff, 0x1234_5678),
            0x5678_ffff
        );
    }

    #[test]
    fn conditional_set_and_select_follow_the_flags() {
        // cmp x0, x1 then cset/csel on each condition, compared with a software evaluation
        for (x, y) in [
            (1u64, 2u64),
            (2, 1),
            (5, 5),
            (0, u64::MAX),
            (u64::MAX, 0),
            (1 << 63, 1),
            (i64::MAX as u64, u64::MAX),
        ] {
            for (condition, holds) in [
                (EQ, x == y),
                (NE, x != y),
                (CS, x >= y),
                (CC, x < y),
                (HI, x > y),
                (LS, x <= y),
                (GE, (x as i64) >= (y as i64)),
                (LT, (x as i64) < (y as i64)),
                (GT, (x as i64) > (y as i64)),
                (LE, (x as i64) <= (y as i64)),
            ] {
                let set = run(
                    |a| {
                        a.cmp(0, 1, true);
                        a.cset(0, condition);
                    },
                    x,
                    y,
                );
                assert_eq!(
                    set, holds as u64,
                    "cset {condition} after cmp {x:#x}, {y:#x}"
                );
                let selected = run(
                    |a| {
                        a.cmp(0, 1, true);
                        a.mov_imm(2, 7);
                        a.mov_imm(3, 9);
                        a.csel(0, 2, 3, condition);
                    },
                    x,
                    y,
                );
                assert_eq!(
                    selected,
                    if holds { 7 } else { 9 },
                    "csel {condition} after cmp {x:#x}, {y:#x}"
                );
            }
        }
    }

    #[test]
    fn flags_round_trip_through_mrs_and_msr() {
        const N: u64 = 1 << 31;
        const Z: u64 = 1 << 30;
        const C: u64 = 1 << 29;
        const V: u64 = 1 << 28;
        for nzcv in [0, N, Z, C, V, N | Z | C | V, Z | C] {
            // msr nzcv, x0; mrs x0, nzcv
            assert_eq!(
                run(
                    |a| {
                        a.msr_nzcv(0);
                        a.mrs_nzcv(0);
                    },
                    nzcv,
                    0
                ),
                nzcv
            );
        }
    }

    #[test]
    fn cinc_adds_one_only_when_equal() {
        assert_eq!(
            run(
                |a| {
                    a.cmp(0, 1, true);
                    a.cinc_eq(0, 0);
                },
                5,
                5
            ),
            6
        );
        assert_eq!(
            run(
                |a| {
                    a.cmp(0, 1, true);
                    a.cinc_eq(0, 0);
                },
                5,
                6
            ),
            5
        );
    }

    #[test]
    fn pair_stores_and_loads_move_two_registers_with_each_addressing_mode() {
        let mut buffer = [0u64; 8];
        let base = buffer.as_mut_ptr() as u64;
        // stp x2, x3, [x0, #16]; ldp x4, x5, [x0, #16]; add x0, x4, x5
        let sum = run(
            |a| {
                a.mov_imm(2, 11);
                a.mov_imm(3, 31);
                a.stp_off(2, 3, 0, 16);
                a.ldp_off(4, 5, 0, 16);
                a.add(0, 4, 5, true);
            },
            base,
            0,
        );
        assert_eq!(sum, 42);
        assert_eq!((buffer[2], buffer[3]), (11, 31));
    }

    #[test]
    fn register_offset_accesses_extend_as_documented() {
        let buffer: [u8; 16] = [
            0x80, 0x80, 0x00, 0x80, 0xff, 0xff, 0xff, 0x80, 1, 2, 3, 4, 5, 6, 7, 8,
        ];
        let base = buffer.as_ptr() as u64;
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 0);
                    a.ldr_r(0, 0, 2, 1, true);
                },
                base,
                0
            ),
            0xffff_ffff_ffff_ff80
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 0);
                    a.ldr_r(0, 0, 2, 2, true);
                },
                base,
                0
            ),
            0xffff_ffff_ffff_8080
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 4);
                    a.ldr_r(0, 0, 2, 4, true);
                },
                base,
                0
            ),
            0xffff_ffff_80ff_ffff
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 4);
                    a.ldr_r(0, 0, 2, 4, false);
                },
                base,
                0
            ),
            0x80ff_ffff
        );
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 8);
                    a.ldr_r(0, 0, 2, 8, false);
                },
                base,
                0
            ),
            0x0807_0605_0403_0201
        );
    }

    #[test]
    fn the_ldr_r_size_argument_selects_the_access_width_and_odd_sizes_load_four_bytes_signed() {
        // Documented fallthrough: a signed size other than 1, 2 or 8 is a sign-extending 4-byte load.
        let buffer: [u8; 8] = [0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0];
        let base = buffer.as_ptr() as u64;
        assert_eq!(
            run(
                |a| {
                    a.mov_imm(2, 0);
                    a.ldr_r(0, 0, 2, 3, true);
                },
                base,
                0
            ),
            u64::MAX
        );
    }

    #[test]
    fn acquire_loads_and_release_stores_move_the_selected_width() {
        let mut buffer = [0xffu8; 16];
        let base = buffer.as_mut_ptr() as u64;
        for (size, written) in [(1u8, 1usize), (2, 2), (4, 4), (8, 8)] {
            buffer.fill(0xff);
            run(
                |a| {
                    a.mov_imm(2, 0x1122_3344_5566_7788);
                    a.stlr(2, 0, size);
                },
                base,
                0,
            );
            assert_eq!(
                buffer[..written],
                0x1122_3344_5566_7788u64.to_le_bytes()[..written]
            );
            assert!(
                buffer[written..].iter().all(|&byte| byte == 0xff),
                "size {size}"
            );
        }
        buffer[..8].copy_from_slice(&0x8877_6655_4433_2211u64.to_le_bytes());
        for (size, expected) in [
            (1u8, 0x11u64),
            (2, 0x2211),
            (4, 0x4433_2211),
            (8, 0x8877_6655_4433_2211),
        ] {
            assert_eq!(
                run(|a| a.ldar(0, 0, size), base, 0),
                expected,
                "size {size}"
            );
        }
    }

    #[test]
    fn store_register_offset_writes_only_the_selected_width() {
        let mut buffer = [0xffu8; 16];
        let base = buffer.as_mut_ptr() as u64;
        run(
            |a| {
                a.mov_imm(2, 0x1122_3344_5566_7788);
                a.mov_imm(3, 0);
                a.str_r(2, 0, 3, 1);
            },
            base,
            0,
        );
        assert_eq!(&buffer[..2], &[0x88, 0xff]);
        run(
            |a| {
                a.mov_imm(2, 0x1122_3344_5566_7788);
                a.mov_imm(3, 4);
                a.str_r(2, 0, 3, 2);
            },
            base,
            0,
        );
        assert_eq!(&buffer[4..7], &[0x88, 0x77, 0xff]);
        run(
            |a| {
                a.mov_imm(2, 0x1122_3344_5566_7788);
                a.mov_imm(3, 8);
                a.str_r(2, 0, 3, 4);
            },
            base,
            0,
        );
        assert_eq!(&buffer[8..13], &[0x88, 0x77, 0x66, 0x55, 0xff]);
    }

    /// The bitfield extract and the W-register immediate shifts used for 8/16/32-bit operands.
#[rustfmt::skip]
#[test]
fn bitfield_extract_and_sized_shift_encodings_match_the_assembler() {
    let cases: &[EncodeCase] = &[
        ("ubfx x0, x1, #8, #8", |a| { a.ubfx(0, 1, 8, 8, true); }, &[0xd3483c20]),
        ("ubfx w2, w3, #1, #1", |a| { a.ubfx(2, 3, 1, 1, false); }, &[0x53010462]),
        ("ubfx x4, x5, #63, #1", |a| { a.ubfx(4, 5, 63, 1, true); }, &[0xd37ffca4]),
        ("lsl w0, w1, #24", |a| { a.lsl_imm_sized(0, 1, 24, false); }, &[0x53081c20]),
        ("lsl w0, w1, #16", |a| { a.lsl_imm_sized(0, 1, 16, false); }, &[0x53103c20]),
        ("lsl w2, w3, #1", |a| { a.lsl_imm_sized(2, 3, 1, false); }, &[0x531f7862]),
        ("lsl x0, x1, #5", |a| { a.lsl_imm_sized(0, 1, 5, true); }, &[0xd37be820]),
        ("lsr w0, w1, #24", |a| { a.lsr_imm_sized(0, 1, 24, false); }, &[0x53187c20]),
        ("lsr w4, w5, #31", |a| { a.lsr_imm_sized(4, 5, 31, false); }, &[0x531f7ca4]),
        ("asr w0, w1, #31", |a| { a.asr_imm_sized(0, 1, 31, false); }, &[0x131f7c20]),
        ("asr w2, w3, #1", |a| { a.asr_imm_sized(2, 3, 1, false); }, &[0x13017c62]),
        ("asr x6, x7, #63", |a| { a.asr_imm_sized(6, 7, 63, true); }, &[0x937ffce6]),
        ("ror x0, x1, #5", |a| { a.ror_imm(0, 1, 5, true); }, &[0x93c11420]),
        ("ror w2, w3, #31", |a| { a.ror_imm(2, 3, 31, false); }, &[0x13837c62]),
        ("extr x4, x5, x5, #63", |a| { a.ror_imm(4, 5, 63, true); }, &[0x93c5fca4]),
        ("rorv x0, x1, x2", |a| { a.rorv(0, 1, 2, true); }, &[0x9ac22c20]),
        ("rorv w3, w4, w5", |a| { a.rorv(3, 4, 5, false); }, &[0x1ac52c83]),
        ("smulh x0, x1, x2", |a| { a.smulh(0, 1, 2); }, &[0x9b427c20]),
        ("smulh x28, x29, x30", |a| { a.smulh(28, 29, 30); }, &[0x9b5e7fbc]),
        ("smulh x3, x3, x4", |a| { a.smulh(3, 3, 4); }, &[0x9b447c63]),
        ("umulh x0, x1, x2", |a| { a.umulh(0, 1, 2); }, &[0x9bc27c20]),
        ("umulh x28, x29, x30", |a| { a.umulh(28, 29, 30); }, &[0x9bde7fbc]),
        ("extr x0, x1, x2, #5", |a| { a.extr(0, 1, 2, 5, true); }, &[0x93c21420]),
        ("extr w28, w29, w30, #31", |a| { a.extr(28, 29, 30, 31, false); }, &[0x139e7fbc]),
        ("extr x3, x3, x4, #63", |a| { a.extr(3, 3, 4, 63, true); }, &[0x93c4fc63]),
        ("clz x0, x1", |a| { a.clz(0, 1, true); }, &[0xdac01020]),
        ("clz w28, w29", |a| { a.clz(28, 29, false); }, &[0x5ac013bc]),
        ("rbit x0, x1", |a| { a.rbit(0, 1, true); }, &[0xdac00020]),
        ("rbit w28, w29", |a| { a.rbit(28, 29, false); }, &[0x5ac003bc]),
    ];
    check_encodings(cases);
}

    #[test]
    fn adr_ldr_literal_bl_and_ret_register_encodings_match_the_assembler() {
        // `l0: adr x7, l0; nop; adr x7, l1; nop; nop; l1: nop; bl l1; ret x28; ldr x1, l1`
        let words = assemble(|a| {
            let l0 = a.label();
            a.bind(l0);
            a.adr(7, l0);
            nop(a);
            let l1 = a.label();
            a.adr(7, l1);
            nop(a);
            nop(a);
            a.bind(l1);
            nop(a);
            a.bl_abs(BASE_ADDRESS + 5 * 4);
            a.ret_reg(28);
            a.ldr_literal(1, l1);
        });
        assert_eq!(
            words,
            [
                0x1000_0007,
                NOP_WORD,
                0x1000_0067,
                NOP_WORD,
                NOP_WORD,
                NOP_WORD,
                0x97ff_ffff,
                0xd65f_0380,
                0x58ff_ffa1,
            ]
        );
    }

    #[test]
    fn literals_land_after_the_code_aligned_and_are_reachable() {
        let mut a = Asm::new(BASE_ADDRESS);
        let pair = a.literal_pair(0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00);
        let single = a.literal(0xdead_beef_0bad_f00d);
        a.adr(7, pair);
        a.ldr_literal(1, single);
        a.finish();
        // Two instructions of code (adr, ldr; already an even count, so no pad), then three 64-bit
        // constants.
        assert_eq!(a.code.len(), 2 + 6);
        assert_eq!(
            &a.code[2..],
            [
                0x5566_7788,
                0x1122_3344,
                0xddee_ff00,
                0x99aa_bbcc,
                0x0bad_f00d,
                0xdead_beef,
            ]
        );
        // adr x7 -> word 2 (offset 2 words), ldr x1 -> word 6 (offset 5 words)
        assert_eq!(a.code[0], 0x1000_0007 | 2 << 5);
        assert_eq!(a.code[1], 0x5800_0001 | 5 << 5);
    }

    #[test]
    fn an_odd_code_length_is_padded_before_the_pool() {
        let mut a = Asm::new(BASE_ADDRESS);
        let value = a.literal(7);
        a.adr(0, value);
        a.finish();
        assert_eq!(
            a.code.len(),
            4,
            "adr, pad, and the two words of the constant"
        );
        assert_eq!(a.code[1], NOP_WORD, "the pad is a nop");
        assert_eq!(
            a.code[0],
            0x1000_0000 | 2 << 5,
            "the constant is two words ahead"
        );
        assert_eq!(&a.code[2..], [7, 0]);
    }
}
