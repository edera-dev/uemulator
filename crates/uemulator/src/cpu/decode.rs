use super::{SEG_CS, SEG_DS, SEG_ES, SEG_FS, SEG_GS, SEG_SS};

/// What decoding depends on: the code segment's default sizes, and whether REX and the 64-bit
/// opcode map apply. Derived from CS by `Cpu::code_mode`; the decode cache is keyed on it because
/// the same bytes decode differently in each.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum CodeMode {
    /// 64-bit mode (CS.L with EFER.LMA): 32-bit operands and 64-bit addresses by default.
    #[default]
    Long64,
    /// A 32-bit code segment (protected or compatibility mode): CS.D set.
    Bits32,
    /// A 16-bit code segment, which includes real mode and virtual-8086 mode: CS.D clear.
    Bits16,
}

/// One decoded instruction. Cached per guest physical address in `Cpu::decode_cache` and
/// consumed by the interpreter (`exec.rs`) and the JIT front ends (`jit/`), so field meanings
/// are shared contracts.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Insn {
    /// Total encoded length in bytes, prefixes included (at most `MAX_INSN_LEN`).
    pub len: u8,
    /// `(map << 8) | opcode`, where map is 0 for one-byte opcodes, 1 for `0F xx`, 2 for
    /// `0F 38 xx` and 3 for `0F 3A xx`. The JIT uses `0xffff` for an undecodable instruction.
    pub op: u16,
    /// Last repeat prefix seen: 0 none, 1 = `F3` (rep/repe), 2 = `F2` (repne). For SSE opcodes
    /// the same prefixes select the scalar/other variant instead.
    pub rep: u8,
    pub lock: bool,
    /// `66` prefix seen. Kept even when REX.W overrides it in `opsize`, because push/pop and
    /// SSE opcodes look at the prefix itself.
    pub p66: bool,
    /// Raw REX byte (`0x40..=0x4f`), or 0 when absent.
    pub rex: u8,
    /// Distinguishes a bare `0x40` REX from none: it changes which byte register `spl/bpl/sil/dil`
    /// vs `ah/ch/dh/bh` is addressed by register numbers 4-7.
    pub rex_present: bool,
    /// Operand size in bytes from REX.W and `66` only (8, 2 or 4). Byte-sized opcodes and opcodes
    /// with a fixed or default-64-bit size (push/pop, near branches) must not use this blindly.
    pub opsize: u8,
    /// The effective address size is 32 bits: a `67` prefix in 64-bit or 16-bit code, or no prefix
    /// in 32-bit code. Also gives `Moffs` four bytes.
    pub addr32: bool,
    /// The effective address size is 16 bits (`67` in 32-bit code, no prefix in 16-bit code):
    /// ModRM uses the `[bx+si]` table and the sum wraps at 64 KiB. Neither flag means 64 bits.
    pub addr16: bool,
    /// The memory operand goes through SS unless a segment-override prefix says otherwise
    /// (`bp`/`esp`/`ebp` based addressing outside 64-bit mode); DS otherwise.
    pub ss_default: bool,
    /// The mode this was decoded in; sizes the stack and branch operands (see `stack_operand_size`).
    pub mode: CodeMode,
    /// Segment-override prefix as an index into `Cpu::seg` (`SEG_ES`..`SEG_GS`), or `NO_SEG`.
    /// The last override wins. The JIT also indexes `Cpu::seg` with it directly.
    pub seg: u8,
    pub has_modrm: bool,
    pub modrm: u8,
    /// ModRM.mod: 3 means register operand, otherwise memory.
    pub md: u8,
    /// ModRM.reg extended with REX.R (bit 3). For group opcodes the opcode extension is
    /// `reg & 7`.
    pub reg: u8,
    /// ModRM.rm extended with REX.B. Only a register number when `md == 3`; memory operands use
    /// `base`/`index`.
    pub rm: u8,
    /// Memory base register (REX.B applied), or `NO_REG` for RIP-relative and disp32-only forms.
    pub base: u8,
    /// SIB index register (REX.X applied), or `NO_REG` when there is no index.
    pub index: u8,
    /// SIB scale as log2 (0..=3).
    pub scale: u8,
    /// ModRM.mod == 0 with rm == 5 and no SIB: `disp` is relative to the next instruction.
    pub rip_rel: bool,
    /// Sign-extended displacement.
    pub disp: i64,
    /// Sign-extended immediate, except `ret imm16` and `enter` sizes, which are zero-extended.
    /// Also holds relative branch offsets (relative to the next instruction) and the absolute
    /// address of `Moffs` forms.
    pub imm: i64,
    /// Nesting level of `enter` (second immediate).
    pub imm2: u16,
    /// Encoded with a VEX prefix (`c4`/`c5`). `op` then carries `VEX_OP_FLAG` on top of
    /// `(map << 8) | opcode`, so no legacy opcode match (interpreter or JIT) can claim it. `p66`
    /// and `rep` hold the implied-prefix field `pp`, `rex` the inverted-back R/X/B/W bits (with
    /// `rex_present` still false: VEX never selects `spl/bpl/sil/dil`), and `opsize` is 8 with
    /// VEX.W and 4 otherwise, because `pp = 66` is an opcode extension for VEX, not an operand
    /// size.
    pub vex: bool,
    /// VEX.L: 256-bit vector length.
    pub vex_l: bool,
    /// VEX.vvvv, already un-inverted (0..=15; 0..=31 with EVEX.V'): the extra register operand.
    pub vex_v: u8,
    /// Encoded with an EVEX prefix (`62`, AVX-512). `op` then carries `EVEX_OP_FLAG`, `vex` stays
    /// false, and `reg`, `rm` (register form), `vex_v` and a VSIB `index` are 5 bits wide. `p66`,
    /// `rep` and `opsize` follow the VEX conventions. `disp` holds the raw disp8 for `md == 1`:
    /// the executor scales it by the tuple size (`evex_disp8`).
    pub evex: bool,
    /// EVEX.aaa: opmask register selecting the elements to write (0 = no masking).
    pub evex_mask: u8,
    /// EVEX.z: zero the masked-out elements instead of keeping them.
    pub evex_zeroing: bool,
    /// EVEX.b: broadcast with a memory operand; embedded rounding / suppress-all-exceptions with a
    /// register operand (then `evex_ll` is the rounding mode, not a vector length).
    pub evex_b: bool,
    /// EVEX.L'L: 0 = 128, 1 = 256, 2 = 512 bits, 3 reserved (rounding mode for `evex_b` forms).
    pub evex_ll: u8,
    /// `md == 1`: `disp` is a compressed disp8 (multiplied by the memory tuple size).
    pub evex_disp8: bool,
}

/// Set in `Insn::op` for VEX-encoded instructions; see `Insn::vex`.
pub(crate) const VEX_OP_FLAG: u16 = 0x1000;
/// Set in `Insn::op` for EVEX-encoded instructions; see `Insn::evex`.
pub(crate) const EVEX_OP_FLAG: u16 = 0x2000;

#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
/// Opcodes whose ModRM.reg field selects the operation (shift group, `f7` group, `0f ba`, ...):
/// `Insn::histogram_key` keeps the selector for these only.
const GROUP_OPCODES: &[u16] = &[
    0x080, 0x081, 0x083, 0x08f, 0x0c0, 0x0c1, 0x0c6, 0x0c7, 0x0d0, 0x0d1, 0x0d2, 0x0d3, 0x0f6,
    0x0f7, 0x0fe, 0x0ff, 0x101, 0x118, 0x171, 0x172, 0x173, 0x1ae, 0x1ba, 0x1c7,
];
const HISTOGRAM_GROUP_SHIFT: u32 = 16;
const HISTOGRAM_REP_SHIFT: u32 = 20;
const HISTOGRAM_P66_SHIFT: u32 = 22;
const HISTOGRAM_VEX_SHIFT: u32 = 23;
const HISTOGRAM_OPSIZE_SHIFT: u32 = 24;
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
const GROUP_SELECTOR_MASK: u8 = 7;

#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
impl Insn {
    /// What kind of instruction this is, for counting how often each reaches the interpreter:
    /// opcode (EVEX flag included), group selector, repeat prefix, `66`, VEX/EVEX and operand size,
    /// but not registers, addresses or immediates. `histogram_label` turns it back into text.
    pub(crate) fn histogram_key(&self) -> u64 {
        let group = if self.has_modrm && GROUP_OPCODES.contains(&(self.op & 0x1fff)) {
            (self.reg & GROUP_SELECTOR_MASK) as u64 + 1
        } else {
            0
        };
        self.op as u64
            | group << HISTOGRAM_GROUP_SHIFT
            | (self.rep as u64) << HISTOGRAM_REP_SHIFT
            | (self.p66 as u64) << HISTOGRAM_P66_SHIFT
            | ((self.vex || self.evex) as u64) << HISTOGRAM_VEX_SHIFT
            | (self.opsize as u64) << HISTOGRAM_OPSIZE_SHIFT
    }
}

/// `Insn::histogram_key` as text: `op=0x1ba/4 rep=1 p66 vex size=4`.
pub(crate) fn histogram_label(key: u64) -> alloc::string::String {
    use alloc::format;
    let op = key & 0xffff;
    let group = (key >> HISTOGRAM_GROUP_SHIFT) & 0xf;
    let mut label = format!("op={:#05x}", op & 0x1fff);
    if op & EVEX_OP_FLAG as u64 != 0 {
        label += " evex";
    }
    if group != 0 {
        label += &format!("/{}", group - 1);
    }
    match (key >> HISTOGRAM_REP_SHIFT) & 3 {
        0 => {}
        rep => label += &format!(" rep={rep}"),
    }
    if (key >> HISTOGRAM_P66_SHIFT) & 1 != 0 {
        label += " p66";
    }
    if (key >> HISTOGRAM_VEX_SHIFT) & 1 != 0 {
        label += " vex";
    }
    label + &format!(" size={}", key >> HISTOGRAM_OPSIZE_SHIFT)
}

/// Marks an absent `base`/`index` register.
pub(crate) const NO_REG: u8 = 0xff;
/// No segment-override prefix.
pub(crate) const NO_SEG: u8 = 0xff;

/// Architectural instruction length limit. A longer encoding is #GP(0), raised by
/// `Cpu::fetch_decode` when the 15-byte window is used up without a complete instruction (see
/// `DecodeErr::Truncated`).
const MAX_INSN_LEN: usize = 15;

const PREFIX_OPERAND_SIZE: u8 = 0x66;
const PREFIX_ADDRESS_SIZE: u8 = 0x67;
const PREFIX_LOCK: u8 = 0xf0;
const PREFIX_REPNE: u8 = 0xf2;
const PREFIX_REP: u8 = 0xf3;
/// `Insn::rep` after an F2 prefix (F3 is 1).
const REP_F2: u8 = 2;
/// 0F 20 and 0F 22: mov from and to a control register; CR8 is the one AltMovCr8 reaches.
const OPCODE_MOV_FROM_CR: u8 = 0x20;
const OPCODE_MOV_TO_CR: u8 = 0x22;
const CR8: u8 = 8;
/// 0F 78: the SSE4A forms that carry two immediates.
const OPCODE_SSE4A_IMMEDIATE_FORMS: u8 = 0x78;
const PREFIX_SEG_ES: u8 = 0x26;
const PREFIX_SEG_CS: u8 = 0x2e;
const PREFIX_SEG_SS: u8 = 0x36;
const PREFIX_SEG_DS: u8 = 0x3e;
const PREFIX_SEG_FS: u8 = 0x64;
const PREFIX_SEG_GS: u8 = 0x65;

const VEX_THREE_BYTE: u8 = 0xc4;
const VEX_TWO_BYTE: u8 = 0xc5;
const EVEX_PREFIX: u8 = 0x62;
/// EVEX payload bits. P0 is `R X B R' 0 mmm` (R, X, B, R' inverted), P1 is `W vvvv 1 pp` and P2 is
/// `z L'L b V' aaa` (V' inverted). The fixed bits must hold their value or the encoding is #UD.
const EVEX_R: u8 = 0x80;
const EVEX_X: u8 = 0x40;
const EVEX_B: u8 = 0x20;
const EVEX_R_PRIME: u8 = 0x10;
const EVEX_P0_MUST_BE_ZERO: u8 = 0x08;
const EVEX_MAP_MASK: u8 = 0x07;
const EVEX_P1_MUST_BE_ONE: u8 = 0x04;
const EVEX_Z: u8 = 0x80;
const EVEX_LL_SHIFT: u32 = 5;
const EVEX_LL_MASK: u8 = 3;
const EVEX_BROADCAST: u8 = 0x10;
const EVEX_V_PRIME: u8 = 0x08;
const EVEX_AAA_MASK: u8 = 7;
/// VEX payload bits (stored inverted for R, X, B and vvvv).
const VEX_R: u8 = 0x80;
const VEX_X: u8 = 0x40;
const VEX_B: u8 = 0x20;
const VEX_MAP_MASK: u8 = 0x1f;
const VEX_W: u8 = 0x80;
const VEX_VVVV_SHIFT: u32 = 3;
const VEX_VVVV_MASK: u8 = 0xf;
const VEX_L: u8 = 0x04;
const VEX_PP_MASK: u8 = 0x03;
const VEX_PP_66: u8 = 1;
const VEX_PP_F3: u8 = 2;
const VEX_PP_F2: u8 = 3;

const REX_FIRST: u8 = 0x40;
const REX_LAST: u8 = 0x4f;
const REX_W: u8 = 8;
const REX_R: u8 = 4;
const REX_X: u8 = 2;
const REX_B: u8 = 1;

const OPCODE_ESCAPE_0F: u8 = 0x0f;
const OPCODE_ESCAPE_0F38: u8 = 0x38;
const OPCODE_ESCAPE_0F3A: u8 = 0x3a;

/// Opcode map numbers, stored in the high byte of `Insn::op`.
const MAP_ONE_BYTE: u16 = 0;
const MAP_0F: u16 = 1;
const MAP_0F38: u16 = 2;
const MAP_0F3A: u16 = 3;

/// ModRM.rm value that selects a SIB byte (mod != 3), and the rm/base value that means
/// disp32 (mod == 0) instead of a register.
const RM_SIB_FOLLOWS: u8 = 4;
const RM_DISP32_OR_RIP: u8 = 5;
const SIB_BASE_DISP32: u8 = 5;
/// SIB.index value meaning "no index" (before REX.X is applied, so r12 remains usable).
const SIB_INDEX_NONE: u8 = 4;

const MODRM_MOD_DISP8: u8 = 1;
const MODRM_MOD_DISP32: u8 = 2;
const MODRM_MOD_REGISTER: u8 = 3;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecodeErr {
    /// The byte slice ended in the middle of the instruction. The caller supplies more bytes
    /// (an instruction may straddle a page boundary) and retries; at the full 15 bytes it is #UD.
    Truncated,
    /// Not decodable as an instruction in 64-bit mode (opcode removed in long mode, or an
    /// undefined encoding). Becomes #UD. An over-long encoding is not this case: `fetch_decode`
    /// turns it into #GP(0).
    Invalid,
}

/// Immediate operand encodings, named after the SDM operand-type letters.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Imm {
    None,
    I8,
    I16,
    /// 16 bits with a 66 prefix, else 32 bits (sign-extended under REX.W).
    Iz,
    /// Full operand size, up to 64 bits (`mov r64, imm64`).
    Iv,
    /// Absolute address operand of `mov al/eax, [moffs]`: 8 bytes, or 4 with a 67 prefix.
    Moffs,
    Enter,
    Rel8,
    /// 32 bits, or 16 with a 66 prefix in 32-bit code or no prefix in 16-bit code (`opsize` 2).
    Rel32,
    /// `call/jmp ptr16:16/32`: an offset of `opsize` bytes (`imm`) then a selector (`imm2`).
    FarPtr,
    /// Two imm8s, the first in `imm` and the second in `imm2` (SSE4A extrq/insertq).
    TwoI8,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> Result<u8, DecodeErr> {
        let v = *self.bytes.get(self.pos).ok_or(DecodeErr::Truncated)?;
        self.pos += 1;
        if self.pos > MAX_INSN_LEN {
            return Err(DecodeErr::Invalid);
        }
        Ok(v)
    }

    /// Reads `n` little-endian bytes and sign-extends from that width (8 bytes is returned as is).
    fn int(&mut self, n: usize) -> Result<i64, DecodeErr> {
        let mut v: u64 = 0;
        for i in 0..n {
            v |= (self.u8()? as u64) << (8 * i);
        }
        Ok(match n {
            1 => v as u8 as i8 as i64,
            2 => v as u16 as i16 as i64,
            4 => v as u32 as i32 as i64,
            _ => v as i64,
        })
    }
}

fn imm_kind(map: u16, opcode: u8) -> Imm {
    match map {
        MAP_ONE_BYTE => match opcode {
            // ALU group opcodes 00-3f: low three bits 4 and 5 are `op al, imm8` / `op eax, imm32`.
            0x00..=0x3f => match opcode & 7 {
                4 => Imm::I8,
                5 => Imm::Iz,
                _ => Imm::None,
            },
            // 68/69 push/imul imm32, 81 group 1, c7 mov, a9 test eax
            0x68 | 0x69 | 0x81 | 0xc7 | 0xa9 => Imm::Iz,
            // 6a/6b push/imul imm8, 80/83 group 1, c0/c1 shift groups, c6 mov, a8 test al, cd int
            // d4/d5 aam/aad (not in 64-bit mode) and 82 (group 1 alias) also take imm8
            0x6a | 0x6b | 0x80 | 0x82 | 0x83 | 0xc0 | 0xc1 | 0xc6 | 0xa8 | 0xcd | 0xd4 | 0xd5 => {
                Imm::I8
            }
            0x9a | 0xea => Imm::FarPtr,
            // jcc, loopne/loope/loop/jrcxz, jmp short
            0x70..=0x7f | 0xe0..=0xe3 | 0xeb => Imm::Rel8,
            0xe8 | 0xe9 => Imm::Rel32,
            0xa0..=0xa3 => Imm::Moffs,
            // mov r8, imm8 and in/out with imm8 port
            0xb0..=0xb7 | 0xe4..=0xe7 => Imm::I8,
            0xb8..=0xbf => Imm::Iv,
            // ret/retf imm16
            0xc2 | 0xca => Imm::I16,
            0xc8 => Imm::Enter,
            // f6/f7 /0 and /1 (test) take an immediate; `decode` handles that using ModRM.reg.
            _ => Imm::None,
        },
        MAP_0F => match opcode {
            // jcc rel32
            0x80..=0x8f => Imm::Rel32,
            // pshuf* (70), psrl/psra/psll groups (71-73), shld/shrd imm8, bt group,
            // cmpps, pinsrw, pextrw, shufps
            0x70..=0x73 | 0xa4 | 0xac | 0xba | 0xc2 | 0xc4..=0xc6 => Imm::I8,
            _ => Imm::None,
        },
        MAP_0F3A => Imm::I8,
        _ => Imm::None,
    }
}

fn has_modrm(map: u16, opcode: u8) -> bool {
    match map {
        MAP_ONE_BYTE => match opcode {
            // ALU group: only the four `op r/m, r` / `op r, r/m` forms of each row take ModRM
            0x00..=0x3f => opcode & 7 < 4,
            // 62/c4/c5 are rejected earlier as EVEX/VEX; 63 movsxd, 69/6b imul
            0x62 | 0x63 | 0x69 | 0x6b | 0x80..=0x8f | 0xc0 | 0xc1 | 0xc4 | 0xc5 | 0xc6 | 0xc7 => {
                true
            }
            0xd0..=0xd3 | 0xd8..=0xdf | 0xf6 | 0xf7 | 0xfe | 0xff => true,
            _ => false,
        },
        // 0F opcodes without ModRM: syscall/clts/sysret/invd/wbinvd (05-09), ud2, femms,
        // wrmsr/rdtsc/rdmsr/rdpmc/sysenter/sysexit/getsec (30-37), emms, jcc rel32, push/pop fs/gs,
        // cpuid, rsm, bswap.
        MAP_0F => !matches!(
            opcode,
            0x05..=0x09 | 0x0b | 0x0e | 0x30..=0x37 | 0x77 | 0x80..=0x8f | 0xa0..=0xa2
                | 0xa8..=0xaa | 0xc8..=0xcf
        ),
        _ => true,
    }
}

/// `0F 38` opcodes whose memory operand has a vector index (VSIB): the VEX gathers 90-93, and with
/// EVEX also the scatters a0-a3.
fn is_gather_opcode(opcode: u8, evex: bool) -> bool {
    (0x90..=0x93).contains(&opcode) || (evex && (0xa0..=0xa3).contains(&opcode))
}

/// One-byte opcodes that are invalid in 64-bit mode, plus EVEX (`62`), which is not supported.
/// `c4`/`c5` are `les`/`lds` outside 64-bit mode and VEX prefixes here; `decode` consumes them
/// before this check.
fn invalid_in_long_mode(opcode: u8) -> bool {
    matches!(
        opcode,
        0x06 | 0x07 // push/pop es
            | 0x0e // push cs
            | 0x16 | 0x17 // push/pop ss
            | 0x1e | 0x1f // push/pop ds
            | 0x27 | 0x2f | 0x37 | 0x3f // daa das aaa aas
            | 0x60 | 0x61 // pusha popa
            | 0x62 // bound (EVEX, not supported)
            | 0x82 // group 1 alias
            | 0x9a // far call
            | 0xce // into
            | 0xd4 | 0xd5 | 0xd6 // aam aad salc
            | 0xea // far jmp
    )
}

/// Decodes in 64-bit mode, which is the only mode the JIT compiles and the one most tests use.
#[cfg(any(test, feature = "jit"))]
pub(crate) fn decode(bytes: &[u8]) -> Result<Insn, DecodeErr> {
    decode_in_mode(bytes, CodeMode::Long64)
}

/// The `[bx+si]` ... `[bx]` table of 16-bit addressing, indexed by ModRM.rm: (base, index) as
/// register numbers, `NO_REG` where absent. Mod 0 with rm 6 is `disp16` alone instead of `[bp]`.
const ADDRESS16_REGISTERS: [(u8, u8); 8] = [
    (3, 6),
    (3, 7),
    (5, 6),
    (5, 7),
    (6, NO_REG),
    (7, NO_REG),
    (5, NO_REG),
    (3, NO_REG),
];
const RM_ADDRESS16_DISP16: u8 = 6;
/// Register numbers whose use as a base selects SS as the default segment (rsp, rbp).
const REG_RSP: u8 = 4;
const REG_RBP: u8 = 5;

pub(crate) fn decode_in_mode(bytes: &[u8], mode: CodeMode) -> Result<Insn, DecodeErr> {
    let long = mode == CodeMode::Long64;
    let mut r = Reader { bytes, pos: 0 };
    let mut i = Insn {
        seg: NO_SEG,
        base: NO_REG,
        index: NO_REG,
        mode,
        ..Default::default()
    };
    let mut address_prefix = false;
    let mut rex = 0u8;
    // In 64-bit mode 40-4f are always REX (never inc/dec). A REX only counts when it directly
    // precedes the opcode: a legacy prefix after it drops it, and of two REX bytes in a row the
    // last one wins. Elsewhere they are inc/dec opcodes.
    let op = loop {
        let b = r.u8()?;
        if long && (REX_FIRST..=REX_LAST).contains(&b) {
            rex = b;
            continue;
        }
        match b {
            PREFIX_OPERAND_SIZE => i.p66 = true,
            PREFIX_ADDRESS_SIZE => address_prefix = true,
            PREFIX_LOCK => i.lock = true,
            PREFIX_REPNE => i.rep = REP_F2,
            PREFIX_REP => i.rep = 1,
            PREFIX_SEG_ES => i.seg = SEG_ES as u8,
            PREFIX_SEG_CS => i.seg = SEG_CS as u8,
            PREFIX_SEG_SS => i.seg = SEG_SS as u8,
            PREFIX_SEG_DS => i.seg = SEG_DS as u8,
            PREFIX_SEG_FS => i.seg = SEG_FS as u8,
            PREFIX_SEG_GS => i.seg = SEG_GS as u8,
            _ => break b,
        }
        rex = 0;
    };
    match (mode, address_prefix) {
        (CodeMode::Long64, true) | (CodeMode::Bits16, true) | (CodeMode::Bits32, false) => {
            i.addr32 = true
        }
        (CodeMode::Bits16, false) | (CodeMode::Bits32, true) => i.addr16 = true,
        (CodeMode::Long64, false) => {}
    }
    let mut map = MAP_ONE_BYTE;
    let mut opb = op;
    // Outside 64-bit mode c4/c5 are les/lds unless ModRM.mod would be 3, which those cannot
    // encode (they need memory): that is VEX. The byte to look at is the one after the opcode,
    // `r.pos`, not the instruction's second byte: behind a prefix that would be the opcode.
    let is_vex = (op == VEX_THREE_BYTE || op == VEX_TWO_BYTE)
        && (long
            || bytes
                .get(r.pos)
                .is_some_and(|&b| b >> 6 == MODRM_MOD_REGISTER));
    // Registers 16-31 (EVEX.R', V' and X on a register operand); added once ModRM is known.
    let (mut high_reg, mut high_rm, mut high_vvvv) = (0u8, 0u8, 0u8);
    // Outside 64-bit mode 62 is `bound`.
    if long && op == EVEX_PREFIX {
        // Same restriction as VEX: legacy operand-size, repeat, lock and REX prefixes are #UD.
        if i.p66 || i.rep != 0 || i.lock || rex != 0 {
            return Err(DecodeErr::Invalid);
        }
        let (p0, p1, p2) = (r.u8()?, r.u8()?, r.u8()?);
        let evex_map = (p0 & EVEX_MAP_MASK) as u16;
        if p0 & EVEX_P0_MUST_BE_ZERO != 0
            || p1 & EVEX_P1_MUST_BE_ONE == 0
            || !(MAP_0F..=MAP_0F3A).contains(&evex_map)
        {
            return Err(DecodeErr::Invalid);
        }
        let mut synthesized = REX_FIRST;
        for (bit, rex_bit) in [(EVEX_R, REX_R), (EVEX_X, REX_X), (EVEX_B, REX_B)] {
            if p0 & bit == 0 {
                synthesized |= rex_bit;
            }
        }
        let w = p1 & VEX_W != 0;
        if w {
            synthesized |= REX_W;
        }
        rex = synthesized;
        high_reg = (p0 & EVEX_R_PRIME == 0) as u8;
        high_rm = (p0 & EVEX_X == 0) as u8;
        high_vvvv = (p2 & EVEX_V_PRIME == 0) as u8;
        i.evex = true;
        i.vex_v = (!(p1 >> VEX_VVVV_SHIFT) & VEX_VVVV_MASK) | high_vvvv << 4;
        match p1 & VEX_PP_MASK {
            VEX_PP_66 => i.p66 = true,
            VEX_PP_F3 => i.rep = 1,
            VEX_PP_F2 => i.rep = 2,
            _ => {}
        }
        i.evex_zeroing = p2 & EVEX_Z != 0;
        i.evex_ll = (p2 >> EVEX_LL_SHIFT) & EVEX_LL_MASK;
        i.evex_b = p2 & EVEX_BROADCAST != 0;
        i.evex_mask = p2 & EVEX_AAA_MASK;
        map = evex_map;
        opb = r.u8()?;
        i.rex = rex;
        i.opsize = if w { 8 } else { 4 };
        i.op = EVEX_OP_FLAG | (map << 8) | opb as u16;
    } else if is_vex {
        // A legacy operand-size, repeat, lock or REX prefix before VEX is #UD (SDM 2.3.2).
        if i.p66 || i.rep != 0 || i.lock || rex != 0 {
            return Err(DecodeErr::Invalid);
        }
        let first = r.u8()?;
        let (inverted_rxb, vex_map, payload) = if op == VEX_TWO_BYTE {
            // R is the only extension bit; X, B and W are implied 0 and the map is 0F.
            (first & VEX_R | VEX_X | VEX_B, MAP_0F, first)
        } else {
            let second = r.u8()?;
            (
                first & (VEX_R | VEX_X | VEX_B),
                (first & VEX_MAP_MASK) as u16,
                second,
            )
        };
        if !(MAP_0F..=MAP_0F3A).contains(&vex_map) {
            return Err(DecodeErr::Invalid);
        }
        let w = op == VEX_THREE_BYTE && payload & VEX_W != 0;
        let mut synthesized = REX_FIRST;
        // R, X and B are stored inverted: a clear bit means the extension is in use.
        for (bit, rex_bit) in [(VEX_R, REX_R), (VEX_X, REX_X), (VEX_B, REX_B)] {
            if inverted_rxb & bit == 0 {
                synthesized |= rex_bit;
            }
        }
        if w {
            synthesized |= REX_W;
        }
        // Registers 8-15 do not exist outside 64-bit mode: the extension bits are ignored.
        rex = if long {
            synthesized
        } else {
            synthesized & REX_W | REX_FIRST
        };
        i.vex = true;
        i.vex_v = !(payload >> VEX_VVVV_SHIFT) & VEX_VVVV_MASK;
        if !long {
            i.vex_v &= 7;
        }
        i.vex_l = payload & VEX_L != 0;
        match payload & VEX_PP_MASK {
            VEX_PP_66 => i.p66 = true,
            VEX_PP_F3 => i.rep = 1,
            VEX_PP_F2 => i.rep = 2,
            _ => {}
        }
        map = vex_map;
        opb = r.u8()?;
        i.rex = rex;
        i.opsize = if w { 8 } else { 4 };
        i.op = VEX_OP_FLAG | (map << 8) | opb as u16;
    } else {
        i.rex = rex;
        i.rex_present = rex != 0;
        let rexw = rex & REX_W != 0;
        if op == OPCODE_ESCAPE_0F {
            opb = r.u8()?;
            map = MAP_0F;
            if opb == OPCODE_ESCAPE_0F38 || opb == OPCODE_ESCAPE_0F3A {
                map = if opb == OPCODE_ESCAPE_0F38 {
                    MAP_0F38
                } else {
                    MAP_0F3A
                };
                opb = r.u8()?;
            }
        }
        i.op = (map << 8) | opb as u16;
        i.opsize = match (rexw, i.p66, mode) {
            (true, _, _) => 8,
            (false, true, CodeMode::Bits16) => 4,
            (false, true, _) => 2,
            (false, false, CodeMode::Bits16) => 2,
            (false, false, _) => 4,
        };
    }

    if long && map == MAP_ONE_BYTE && invalid_in_long_mode(opb) {
        return Err(DecodeErr::Invalid);
    }

    if has_modrm(map, opb) || i.evex {
        let m = r.u8()?;
        i.has_modrm = true;
        i.modrm = m;
        i.md = m >> 6;
        // REX.R/X/B extend the 3-bit fields to 4 bits; shift moves the REX bit to bit 3.
        i.reg = ((m >> 3) & 7) | ((i.rex & REX_R) << 1);
        i.rm = (m & 7) | ((i.rex & REX_B) << 3);
        i.reg |= high_reg << 4;
        if i.md == MODRM_MOD_REGISTER {
            i.rm |= high_rm << 4;
        } else if i.addr16 {
            let (base, index) = ADDRESS16_REGISTERS[(m & 7) as usize];
            i.base = base;
            i.index = index;
            if i.md == 0 && m & 7 == RM_ADDRESS16_DISP16 {
                i.base = NO_REG;
                i.disp = r.int(2)?;
            }
            match i.md {
                MODRM_MOD_DISP8 => i.disp = r.int(1)?,
                MODRM_MOD_DISP32 => i.disp = r.int(2)?,
                _ => {}
            }
        } else {
            let mut base_field = m & 7;
            if base_field == RM_SIB_FOLLOWS {
                let sib = r.u8()?;
                i.scale = sib >> 6;
                let index_field = ((sib >> 3) & 7) | ((i.rex & REX_X) << 2);
                // VSIB (the VEX gathers, 0F38 90-93) has no "no index" encoding: index 4 is xmm4/ymm4
                let vsib = (i.vex || i.evex) && map == MAP_0F38 && is_gather_opcode(opb, i.evex);
                if index_field != SIB_INDEX_NONE || vsib {
                    // EVEX.V' extends a VSIB index to zmm16-31; X already did bit 3.
                    i.index = index_field | if vsib { high_vvvv << 4 } else { 0 };
                }
                base_field = sib & 7;
                // mod 0 with SIB base 5 means no base register, disp32 follows (index may remain)
                if base_field == SIB_BASE_DISP32 && i.md == 0 {
                    i.disp = r.int(4)?;
                } else {
                    i.base = base_field | ((i.rex & REX_B) << 3);
                }
            } else if base_field == RM_DISP32_OR_RIP && i.md == 0 {
                // Only 64-bit mode makes this RIP-relative; elsewhere it is an absolute disp32.
                i.rip_rel = long;
                i.disp = r.int(4)?;
            } else {
                i.base = base_field | ((i.rex & REX_B) << 3);
            }
            match i.md {
                MODRM_MOD_DISP8 => {
                    i.disp = r.int(1)?;
                    i.evex_disp8 = i.evex;
                }
                MODRM_MOD_DISP32 => i.disp = r.int(4)?,
                _ => {}
            }
        }
    }

    let legacy_0f = map == MAP_0F && !i.vex && !i.evex;
    // AMD AltMovCr8 (leaf 0x8000_0001 ECX bit 4, CR8_LEGACY): `lock mov cr0` is `mov cr8`, so
    // code without REX (32-bit mode) can reach the task priority. Rewriting it here gives the SVM
    // intercepts and `mov_cr_dr` the register they must see.
    if legacy_0f && matches!(opb, OPCODE_MOV_FROM_CR | OPCODE_MOV_TO_CR) && i.lock && i.reg == 0 {
        i.reg = CR8;
        i.lock = false;
    }
    let mut kind = imm_kind(map, opb);
    // SSE4A extrq (66) and insertq (F2) with immediates: length then index. Without either
    // prefix 0F 78 is Intel's vmread, #UD here.
    if legacy_0f && opb == OPCODE_SSE4A_IMMEDIATE_FORMS && (i.p66 || i.rep == REP_F2) {
        kind = Imm::TwoI8;
    }
    // group 3: only /0 and /1 (test) carry an immediate; not/neg/mul/div do not
    if map == MAP_ONE_BYTE && (opb == 0xf6 || opb == 0xf7) && ((i.modrm >> 3) & 7) < 2 {
        kind = if opb == 0xf6 { Imm::I8 } else { Imm::Iz };
    }
    match kind {
        Imm::None => {}
        Imm::I8 | Imm::Rel8 => i.imm = r.int(1)?,
        Imm::I16 => i.imm = r.int(2)? & 0xffff,
        Imm::Iz => i.imm = r.int(if i.opsize == 2 { 2 } else { 4 })?,
        Imm::Iv => i.imm = r.int(i.opsize as usize)?,
        Imm::Rel32 => i.imm = r.int(if i.opsize == 2 && !long { 2 } else { 4 })?,
        Imm::FarPtr => {
            i.imm = r.int(i.opsize as usize)? & if i.opsize == 2 { 0xffff } else { 0xffff_ffff };
            i.imm2 = r.int(2)? as u16;
        }
        Imm::Moffs => {
            i.imm = r.int(if i.addr16 {
                2
            } else if i.addr32 {
                4
            } else {
                8
            })?
        }
        Imm::Enter => {
            i.imm = r.int(2)? & 0xffff;
            i.imm2 = r.int(1)? as u8 as u16;
        }
        Imm::TwoI8 => {
            i.imm = r.int(1)?;
            i.imm2 = r.int(1)? as u8 as u16;
        }
    }
    i.ss_default = !long && (i.base == REG_RSP || i.base == REG_RBP);
    i.len = r.pos as u8;
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        let cases: &[(&[u8], u8)] = &[
            (&[0x90], 1),
            (&[0x48, 0x89, 0xe5], 3),
            (&[0x48, 0xc7, 0xc0, 1, 0, 0, 0], 7),
            (&[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8], 10),
            (&[0x8b, 0x04, 0x25, 0, 0, 0, 0], 7),
            (&[0x48, 0x8d, 0x05, 0, 0, 0, 0], 7),
            (&[0xf3, 0x0f, 0x1e, 0xfa], 4),
            (&[0x0f, 0x1f, 0x44, 0x00, 0x00], 5),
            (&[0x66, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0], 9),
            (&[0xf7, 0x45, 0xfc, 1, 0, 0, 0], 7),
            (&[0xf7, 0xd0], 2),
            (&[0xe8, 0, 0, 0, 0], 5),
            (&[0x0f, 0x84, 0, 0, 0, 0], 6),
            (&[0x66, 0x83, 0xc0, 1], 4),
            (&[0xc8, 0x10, 0, 0], 4),
        ];
        for (b, n) in cases {
            assert_eq!(decode(b).unwrap().len, *n, "{b:02x?}");
        }
    }

    #[test]
    fn truncated() {
        assert_eq!(decode(&[0x48]).unwrap_err(), DecodeErr::Truncated);
        assert_eq!(decode(&[0x8b, 0x04]).unwrap_err(), DecodeErr::Truncated);
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    fn decoded(bytes: &[u8]) -> Insn {
        decode(bytes).unwrap_or_else(|error| panic!("{bytes:02x?} failed to decode: {error:?}"))
    }

    /// The assembler's byte count is the independent length oracle: the decoder must consume exactly
    /// the bytes of one instruction, and every shorter prefix of it must report `Truncated` (x86
    /// instruction encodings are prefix-free, so the decoder can never settle early).
    #[track_caller]
    fn assert_length(bytes: &[u8]) {
        assert_eq!(decoded(bytes).len as usize, bytes.len(), "{bytes:02x?}");
        for shorter in 0..bytes.len() {
            assert_eq!(
                decode(&bytes[..shorter]).err(),
                Some(DecodeErr::Truncated),
                "{:02x?} cut to {shorter} bytes",
                bytes
            );
        }
    }

    /// Trailing bytes after the instruction must not influence the result.
    #[track_caller]
    fn assert_ignores_trailing_bytes(bytes: &[u8]) {
        let mut longer = bytes.to_vec();
        longer.extend_from_slice(&[0x90; 4]);
        assert_eq!(decoded(&longer).len as usize, bytes.len(), "{bytes:02x?}");
    }

    type PrefixCheck = fn(&Insn) -> bool;

    const SEG_NAMES_AND_PREFIXES: [(u8, usize); 6] = [
        (0x26, SEG_ES),
        (0x2e, SEG_CS),
        (0x36, SEG_SS),
        (0x3e, SEG_DS),
        (0x64, SEG_FS),
        (0x65, SEG_GS),
    ];

    mod vex {
        use super::*;

        #[test]
        fn three_byte_vex_fields_are_unpacked_and_un_inverted() {
            // andn rax, rbx, rcx: c4 e2 e0 f2 c1 (R,X,B inverted = 1, map 2, W=1, vvvv=~1100)
            let i = decoded(&[0xc4, 0xe2, 0xe0, 0xf2, 0xc1]);
            assert!(i.vex && !i.vex_l);
            assert_eq!(i.len, 5);
            assert_eq!(i.op, VEX_OP_FLAG | 0x2f2);
            assert_eq!((i.vex_v, i.opsize), (3, 8));
            assert_eq!((i.reg, i.rm), (0, 1));
            assert_eq!((i.rep, i.p66), (0, false));
            // andn r9, r10, [rdi]: R and B extensions in use (inverted bits clear), vvvv = r10
            let i = decoded(&[0xc4, 0x62, 0xa8, 0xf2, 0x0f]);
            assert_eq!((i.vex_v, i.reg, i.base), (10, 9, 7));
            assert_eq!(i.rex & 0xf, REX_W | REX_R);
            assert!(
                !i.rex_present,
                "vex never selects the spl/bpl/sil/dil byte registers"
            );
        }

        #[test]
        fn x_and_b_extend_the_memory_operand() {
            // c4 22 a8 f2 0c 47: andn r9, r10, [rdi + r8*2]
            let i = decoded(&[0xc4, 0x22, 0xa8, 0xf2, 0x0c, 0x47]);
            assert_eq!((i.base, i.index, i.scale), (7, 8, 1));
        }

        #[test]
        fn two_byte_vex_implies_map_0f_and_no_x_b_w() {
            // vzeroupper: c5 f8 77 (no ModRM), vmovups xmm0,[rax]: c5 f8 10 00
            let i = decoded(&[0xc5, 0xf8, 0x77]);
            assert_eq!((i.len, i.op, i.has_modrm), (3, VEX_OP_FLAG | 0x177, false));
            let i = decoded(&[0xc5, 0xf8, 0x10, 0x00]);
            assert_eq!(
                (i.len, i.op, i.vex_v, i.vex_l),
                (4, VEX_OP_FLAG | 0x110, 0, false)
            );
            // vex.R inverted clear: vmovups xmm8, xmm0 is c5 78 10 c0
            let i = decoded(&[0xc5, 0x78, 0x10, 0xc0]);
            assert_eq!(i.reg, 8);
            // L bit and a non-trivial vvvv: c5 fc 58 c1 = vaddps ymm0, ymm0, ymm1
            let i = decoded(&[0xc5, 0xfc, 0x58, 0xc1]);
            assert!(i.vex_l);
            assert_eq!(i.vex_v, 0);
            let i = decoded(&[0xc5, 0xdc, 0x58, 0xc1]);
            assert_eq!(i.vex_v, 4);
        }

        #[test]
        fn pp_selects_the_implied_prefix_without_changing_operand_size() {
            // shlx (66), sarx (F3), shrx (F2), all with W=0
            let i = decoded(&[0xc4, 0xe2, 0x61, 0xf7, 0xc1]);
            assert_eq!((i.p66, i.rep, i.opsize), (true, 0, 4));
            let i = decoded(&[0xc4, 0xe2, 0x62, 0xf7, 0xc1]);
            assert_eq!((i.p66, i.rep, i.opsize), (false, 1, 4));
            let i = decoded(&[0xc4, 0xe2, 0x63, 0xf7, 0xc1]);
            assert_eq!((i.p66, i.rep, i.opsize), (false, 2, 4));
        }

        #[test]
        fn immediates_follow_the_map() {
            // rorx rax, rcx, 5 (map 3 always has imm8)
            assert_eq!(decoded(&[0xc4, 0xe3, 0xfb, 0xf0, 0xc1, 0x05]).imm, 5);
            assert_length(&[0xc4, 0xe3, 0xfb, 0xf0, 0xc1, 0x05]);
            assert_length(&[0xc5, 0xf9, 0x70, 0xc1, 0x1b]); // vpshufd xmm0, xmm1, 0x1b
        }

        #[test]
        fn lengths_and_truncation() {
            assert_length(&[0xc4, 0xe2, 0xe0, 0xf2, 0xc1]);
            assert_length(&[0xc4, 0x62, 0xa8, 0xf2, 0x0f]);
            assert_length(&[0xc5, 0xf8, 0x77]);
            assert_length(&[0xc5, 0xf8, 0x10, 0x84, 0x24, 0x00, 0x01, 0x00, 0x00]);
        }

        #[test]
        fn reserved_encodings_are_invalid() {
            // maps 0 and 4+ do not exist
            for map in [0u8, 4, 5, 0x1f] {
                assert_eq!(
                    decode(&[0xc4, 0xe0 | map, 0xe0, 0xf2, 0xc1, 0, 0, 0]).err(),
                    Some(DecodeErr::Invalid),
                    "map {map}"
                );
            }
            // legacy 66/F2/F3/F0/REX in front of VEX is #UD; segment and address-size prefixes are fine
            for prefix in [0x66u8, 0xf2, 0xf3, 0xf0, 0x48, 0x40] {
                assert_eq!(
                    decode(&[prefix, 0xc4, 0xe2, 0xe0, 0xf2, 0xc1, 0, 0]).err(),
                    Some(DecodeErr::Invalid),
                    "{prefix:#x}"
                );
            }
            assert_eq!(
                decoded(&[0x64, 0x67, 0xc4, 0xe2, 0xe0, 0xf2, 0x07]).seg as usize,
                SEG_FS
            );
        }

        #[test]
        fn vex_ops_cannot_collide_with_legacy_ops() {
            // vex 0f38 f6 (mulx) vs legacy 0f 38 f6 (adcx): different `op`
            let vex = decoded(&[0xc4, 0xe2, 0xe3, 0xf6, 0xc1]);
            let legacy = decoded(&[0x66, 0x0f, 0x38, 0xf6, 0xc1]);
            assert_ne!(vex.op, legacy.op);
            assert!(!legacy.vex);
        }
    }

    mod prefixes {
        use super::*;

        #[test]
        fn no_prefix_defaults() {
            let nop = decoded(&[0x90]);
            assert_eq!(nop.len, 1);
            assert_eq!(nop.op, 0x90);
            assert_eq!(nop.rep, 0);
            assert!(!nop.lock);
            assert!(!nop.p66);
            assert_eq!(nop.rex, 0);
            assert!(!nop.rex_present);
            assert_eq!(nop.opsize, 4);
            assert!(!nop.addr32);
            assert_eq!(nop.seg, NO_SEG);
            assert!(!nop.has_modrm);
            assert_eq!(nop.base, NO_REG);
            assert_eq!(nop.index, NO_REG);
        }

        #[test]
        fn operand_size_prefix_selects_16_bits() {
            let i = decoded(&[0x66, 0x90]);
            assert!(i.p66);
            assert_eq!(i.opsize, 2);
            assert_eq!(i.len, 2);
        }

        #[test]
        fn address_size_prefix_is_recorded() {
            let i = decoded(&[0x67, 0x90]);
            assert!(i.addr32);
            assert_eq!(i.opsize, 4);
        }

        #[test]
        fn lock_prefix_is_recorded() {
            let i = decoded(&[0xf0, 0x90]);
            assert!(i.lock);
            assert_eq!(i.rep, 0);
        }

        #[test]
        fn rep_prefixes_select_their_code() {
            assert_eq!(decoded(&[0xf3, 0x90]).rep, 1);
            assert_eq!(decoded(&[0xf2, 0x90]).rep, 2);
        }

        #[test]
        fn last_repeat_prefix_wins() {
            assert_eq!(decoded(&[0xf3, 0xf2, 0x90]).rep, 2);
            assert_eq!(decoded(&[0xf2, 0xf3, 0x90]).rep, 1);
        }

        #[test]
        fn each_segment_override_maps_to_its_register_index() {
            for (prefix, expected) in SEG_NAMES_AND_PREFIXES {
                let i = decoded(&[prefix, 0x90]);
                assert_eq!(i.seg as usize, expected, "prefix {prefix:#x}");
            }
        }

        #[test]
        fn last_segment_override_wins() {
            assert_eq!(decoded(&[0x64, 0x65, 0x90]).seg as usize, SEG_GS);
            assert_eq!(decoded(&[0x65, 0x64, 0x90]).seg as usize, SEG_FS);
            assert_eq!(decoded(&[0x2e, 0x64, 0x3e, 0x90]).seg as usize, SEG_DS);
        }

        #[test]
        fn legacy_prefixes_are_order_independent() {
            let orders: [[u8; 5]; 4] = [
                [0x66, 0x67, 0xf0, 0xf3, 0x64],
                [0x64, 0xf3, 0xf0, 0x67, 0x66],
                [0xf0, 0x64, 0x66, 0xf3, 0x67],
                [0x67, 0x66, 0x64, 0xf0, 0xf3],
            ];
            for order in orders {
                let mut bytes = order.to_vec();
                bytes.push(0x90);
                let i = decoded(&bytes);
                assert!(i.p66 && i.addr32 && i.lock, "{order:02x?}");
                assert_eq!(i.rep, 1, "{order:02x?}");
                assert_eq!(i.seg as usize, SEG_FS, "{order:02x?}");
                assert_eq!(i.len, 6, "{order:02x?}");
            }
        }

        #[test]
        fn repeated_operand_size_prefixes_still_mean_16_bits() {
            let i = decoded(&[0x66, 0x66, 0x66, 0x90]);
            assert_eq!(i.opsize, 2);
            assert_eq!(i.len, 4);
        }

        #[test]
        fn prefix_before_two_byte_opcode_keeps_the_map() {
            let i = decoded(&[0x66, 0x0f, 0xef, 0xc1]);
            assert_eq!(i.op, 0x1ef);
            assert!(i.p66);
            assert_eq!(i.len, 4);
        }

        #[test]
        fn rex_must_immediately_precede_the_opcode() {
            // 66 48 89 c0: operand-size prefix first, so REX.W applies and wins over 66.
            let i = decoded(&[0x66, 0x48, 0x89, 0xc0]);
            assert_eq!(i.rex, 0x48);
            assert!(i.rex_present);
            assert_eq!(i.opsize, 8);
            assert!(
                i.p66,
                "66 stays visible to push/pop and SSE even when REX.W overrides it"
            );
            assert_eq!(i.op, 0x89);
            assert_eq!(i.len, 4);
        }

        #[test]
        fn rex_followed_by_operand_size_prefix_is_dropped() {
            // 48 66 89 c0 is `mov ax, ax`: the REX is not adjacent to the opcode, so it is ignored.
            let i = decoded(&[0x48, 0x66, 0x89, 0xc0]);
            assert_eq!(i.op, 0x89);
            assert_eq!(i.rex, 0);
            assert!(!i.rex_present);
            assert_eq!(i.opsize, 2);
            assert!(i.p66);
            assert_eq!(i.len, 4);
            assert!(i.has_modrm);
            assert_eq!((i.reg, i.rm), (0, 0));
        }

        #[test]
        fn rex_followed_by_each_legacy_prefix_is_dropped_and_prefix_applies() {
            let cases: [(u8, PrefixCheck); 11] = [
                (0x66, |i| i.p66),
                (0x67, |i| i.addr32),
                (0xf0, |i| i.lock),
                (0xf2, |i| i.rep == 2),
                (0xf3, |i| i.rep == 1),
                (0x26, |i| i.seg as usize == SEG_ES),
                (0x2e, |i| i.seg as usize == SEG_CS),
                (0x36, |i| i.seg as usize == SEG_SS),
                (0x3e, |i| i.seg as usize == SEG_DS),
                (0x64, |i| i.seg as usize == SEG_FS),
                (0x65, |i| i.seg as usize == SEG_GS),
            ];
            for (prefix, applied) in cases {
                // REX.WRXB then the legacy prefix then `add eax, ebx` (01 d8).
                let i = decoded(&[0x4f, prefix, 0x01, 0xd8]);
                assert!(applied(&i), "prefix {prefix:#x} was not applied");
                assert_eq!(i.rex, 0, "prefix {prefix:#x}");
                assert!(!i.rex_present, "prefix {prefix:#x}");
                assert_eq!(
                    (i.reg, i.rm),
                    (3, 0),
                    "prefix {prefix:#x}: REX.R/B must not leak"
                );
                assert_eq!(i.len, 4, "prefix {prefix:#x}");
            }
        }

        #[test]
        fn dropped_rex_does_not_widen_the_operand_or_extend_registers() {
            // 4c 66 01 d8: REX.WR dropped, so this is `add ax, bx`, not `add r8, rbx`.
            let i = decoded(&[0x4c, 0x66, 0x01, 0xd8]);
            assert_eq!(i.opsize, 2);
            assert_eq!(i.reg, 3);
            assert_eq!(i.rm, 0);
        }

        #[test]
        fn rex_before_legacy_prefix_before_a_new_rex_uses_the_new_rex() {
            // 48 66 4c 01 c8: only the REX adjacent to the opcode (4c: W and R) counts.
            let i = decoded(&[0x48, 0x66, 0x4c, 0x01, 0xc8]);
            assert_eq!(i.rex, 0x4c);
            assert_eq!(i.opsize, 8);
            assert_eq!(i.reg, 9);
            assert_eq!(i.rm, 0);
            assert_eq!(i.len, 5);
        }

        #[test]
        fn consecutive_rex_bytes_use_the_last_one() {
            let i = decoded(&[0x48, 0x41, 0x01, 0xd8]);
            assert_eq!(i.rex, 0x41);
            assert_eq!(
                i.opsize, 4,
                "the earlier REX.W is not adjacent to the opcode"
            );
            assert_eq!(i.rm, 8);
        }

        #[test]
        fn dropped_rex_still_counts_toward_the_15_byte_limit() {
            // 14 prefix bytes (REX then 13 x 66) plus the opcode make exactly 15 bytes.
            let mut bytes = vec![0x48];
            bytes.extend(std::iter::repeat_n(0x66, 13));
            bytes.push(0x90);
            assert_eq!(bytes.len(), 15);
            let i = decoded(&bytes);
            assert_eq!(i.len, 15);
            assert!(!i.rex_present);
            bytes.insert(1, 0x66);
            assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
        }

        #[test]
        fn mandatory_prefix_then_rex_keeps_the_rex() {
            // F3 48 0F B8 C1 (popcnt rax, rcx style): mandatory prefix before REX is the valid order.
            let i = decoded(&[0xf3, 0x48, 0x0f, 0xb8, 0xc1]);
            assert_eq!(i.rep, 1);
            assert_eq!(i.rex, 0x48);
            assert_eq!(i.op, 0x1b8);
        }

        #[test]
        fn rex_before_mandatory_prefix_drops_the_rex() {
            let i = decoded(&[0x48, 0xf3, 0x0f, 0xb8, 0xc1]);
            assert_eq!(i.rep, 1);
            assert_eq!(i.rex, 0);
            assert_eq!(i.opsize, 4);
        }
    }

    mod rex {
        use super::*;

        #[test]
        fn bare_rex_is_present_without_bits() {
            let i = decoded(&[0x40, 0x88, 0xe6]);
            assert!(i.rex_present);
            assert_eq!(i.rex, 0x40);
            assert_eq!(i.opsize, 4);
            assert_eq!((i.reg, i.rm), (4, 6));
        }

        #[test]
        fn every_rex_value_is_a_prefix_never_inc_or_dec() {
            for rex in 0x40..=0x4fu8 {
                let i = decoded(&[rex, 0x90]);
                assert_eq!(i.rex, rex);
                assert!(i.rex_present);
                assert_eq!(i.op, 0x90);
                assert_eq!(i.len, 2);
                assert_eq!(i.opsize, if rex & 8 != 0 { 8 } else { 4 });
            }
        }

        #[test]
        fn rex_w_wins_over_operand_size_prefix() {
            let i = decoded(&[0x66, 0x48, 0x90]);
            assert_eq!(i.opsize, 8);
            assert!(i.p66);
        }

        #[test]
        fn rex_r_extends_modrm_reg() {
            // mov rax, r9
            let bytes = &[0x4c, 0x89, 0xc8];
            let i = decoded(bytes);
            assert_eq!(i.reg, 9);
            assert_eq!(i.rm, 0);
            assert_eq!(i.rex & REX_R, REX_R);
        }

        #[test]
        fn rex_b_extends_modrm_rm_in_register_form() {
            // mov r9, rax
            let bytes = &[0x49, 0x89, 0xc1];
            let i = decoded(bytes);
            assert_eq!(i.rm, 9);
            assert_eq!(i.reg, 0);
        }

        #[test]
        fn rex_b_extends_the_memory_base() {
            // mov rax, [r9 + 8]
            let bytes = &[0x49, 0x8b, 0x41, 0x08];
            let i = decoded(bytes);
            assert_eq!(i.base, 9);
            assert_eq!(i.index, NO_REG);
            assert_eq!(i.disp, 8);
        }

        #[test]
        fn rex_x_extends_the_sib_index() {
            // mov rax, [rbx + r10*4]
            let bytes = &[0x4a, 0x8b, 0x04, 0x93];
            let i = decoded(bytes);
            assert_eq!(i.base, 3);
            assert_eq!(i.index, 10);
            assert_eq!(i.scale, 2);
        }

        #[test]
        fn rex_b_extends_the_sib_base() {
            // mov rax, [r11 + rcx*2]
            let bytes = &[0x49, 0x8b, 0x04, 0x4b];
            let i = decoded(bytes);
            assert_eq!(i.base, 11);
            assert_eq!(i.index, 1);
            assert_eq!(i.scale, 1);
        }

        #[test]
        fn all_rex_bits_together() {
            // mov r15, [r14 + r13*8 + 0x20]
            let bytes = &[0x4f, 0x8b, 0x7c, 0xee, 0x20];
            let i = decoded(bytes);
            assert_eq!(i.reg, 15);
            assert_eq!(i.base, 14);
            assert_eq!(i.index, 13);
            assert_eq!(i.scale, 3);
            assert_eq!(i.disp, 0x20);
            assert_eq!(i.opsize, 8);
        }

        #[test]
        fn rex_before_two_byte_opcode() {
            let i = decoded(&[0x4c, 0x0f, 0xaf, 0xc1]);
            assert_eq!(i.op, 0x1af);
            assert_eq!(i.reg, 8);
            assert_eq!(i.rm, 1);
            assert_eq!(i.opsize, 8);
        }

        #[test]
        fn rex_before_three_byte_opcode() {
            let i = decoded(&[0x48, 0x0f, 0x38, 0xf0, 0xc1]);
            assert_eq!(i.op, 0x2f0);
            assert_eq!(i.rex, 0x48);
        }

        #[test]
        fn low_byte_register_numbers_need_rex_to_mean_spl_through_dil() {
            // Both `mov dh, al` forms are 88 /r with reg = 6; only the bare REX changes the meaning,
            // which the executor reads from rex_present.
            let without = decoded(&[0x88, 0xc6]);
            let with = decoded(&[0x40, 0x88, 0xc6]);
            assert!(!without.rex_present);
            assert!(with.rex_present);
            assert_eq!(without.rm, with.rm);
        }
    }

    mod opcode_maps {
        use super::*;

        #[test]
        fn one_byte_map_has_map_zero() {
            assert_eq!(decoded(&[0x90]).op >> 8, 0);
            assert_eq!(decoded(&[0xff, 0xc0]).op, 0xff);
        }

        #[test]
        fn two_byte_map_sets_map_one() {
            let i = decoded(&[0x0f, 0xa2]);
            assert_eq!(i.op, 0x1a2);
            assert_eq!(i.len, 2);
            assert!(!i.has_modrm);
        }

        #[test]
        fn three_byte_0f38_map_sets_map_two() {
            let i = decoded(&[0x66, 0x0f, 0x38, 0x00, 0xc1]);
            assert_eq!(i.op, 0x200);
            assert!(i.has_modrm);
            assert_eq!(i.len, 5);
        }

        #[test]
        fn three_byte_0f3a_map_sets_map_three_and_takes_an_imm8() {
            let i = decoded(&[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x05]);
            assert_eq!(i.op, 0x30f);
            assert_eq!(i.imm, 5);
            assert_eq!(i.len, 6);
        }

        #[test]
        fn every_0f3a_opcode_has_modrm_and_imm8() {
            for opcode in 0..=0xffu8 {
                let i = decoded(&[0x66, 0x0f, 0x3a, opcode, 0xc1, 0x7f]);
                assert_eq!(i.len, 6, "0f 3a {opcode:#x}");
                assert_eq!(i.imm, 0x7f, "0f 3a {opcode:#x}");
            }
        }

        #[test]
        fn every_0f38_opcode_has_modrm_and_no_immediate() {
            for opcode in 0..=0xffu8 {
                let i = decoded(&[0x66, 0x0f, 0x38, opcode, 0xc1]);
                assert_eq!(i.len, 5, "0f 38 {opcode:#x}");
                assert_eq!(i.imm, 0, "0f 38 {opcode:#x}");
            }
        }

        #[test]
        fn two_byte_opcodes_without_modrm() {
            let no_modrm: Vec<u8> = (0x05..=0x09)
                .chain([0x0b, 0x0e])
                .chain(0x30..=0x37)
                .chain([0x77])
                .chain(0xa0..=0xa2)
                .chain(0xa8..=0xaa)
                .chain(0xc8..=0xcf)
                .collect();
            for opcode in no_modrm {
                let i = decoded(&[0x0f, opcode, 0x90, 0x90, 0x90, 0x90]);
                assert_eq!(i.len, 2, "0f {opcode:#x}");
                assert!(!i.has_modrm, "0f {opcode:#x}");
            }
        }

        #[test]
        fn two_byte_opcodes_with_modrm() {
            for opcode in [
                0x00u8, 0x01, 0x10, 0x20, 0x28, 0x40, 0x6f, 0x7f, 0xa3, 0xaf, 0xb6, 0xc7,
            ] {
                let i = decoded(&[0x0f, opcode, 0xc1, 0x90, 0x90, 0x90]);
                assert!(i.has_modrm, "0f {opcode:#x}");
                assert_eq!(i.modrm, 0xc1);
            }
        }

        #[test]
        fn jcc_rel32_takes_four_bytes() {
            for cc in 0x80..=0x8fu8 {
                let i = decoded(&[0x0f, cc, 0xfe, 0xff, 0xff, 0xff]);
                assert_eq!(i.len, 6);
                assert_eq!(i.imm, -2);
            }
        }

        #[test]
        fn bswap_has_no_modrm_and_keeps_rex_b() {
            // bswap r9
            let bytes = &[0x49, 0x0f, 0xc9];
            let i = decoded(bytes);
            assert_eq!(i.op, 0x1c9);
            assert_eq!(i.len as usize, bytes.len());
            assert!(!i.has_modrm);
            assert_eq!(i.rex & REX_B, REX_B);
        }

        #[test]
        fn escape_alone_is_truncated() {
            assert_eq!(decode(&[0x0f]).err(), Some(DecodeErr::Truncated));
            assert_eq!(decode(&[0x0f, 0x38]).err(), Some(DecodeErr::Truncated));
            assert_eq!(decode(&[0x0f, 0x3a]).err(), Some(DecodeErr::Truncated));
            assert_eq!(
                decode(&[0x0f, 0x38, 0x00]).err(),
                Some(DecodeErr::Truncated)
            );
            assert_eq!(
                decode(&[0x0f, 0x3a, 0x0f, 0xc1]).err(),
                Some(DecodeErr::Truncated)
            );
        }
    }

    mod modrm_and_sib {
        use super::*;

        #[test]
        fn register_direct_form() {
            // add rbx, rcx
            let bytes = &[0x48, 0x01, 0xcb];
            let i = decoded(bytes);
            assert!(i.has_modrm);
            assert_eq!(i.md, 3);
            assert_eq!(i.modrm >> 6, 3);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.index, NO_REG);
            assert!(!i.rip_rel);
            assert_eq!(i.disp, 0);
        }

        #[test]
        fn base_only() {
            // mov rax, [rbx]
            let bytes = &[0x48, 0x8b, 0x03];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base, i.index, i.disp), (0, 3, NO_REG, 0));
            assert_eq!(i.len, 3);
        }

        #[test]
        fn base_plus_disp8_sign_extends() {
            // mov rax, [rbx - 0x80]
            let bytes = &[0x48, 0x8b, 0x43, 0x80];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base), (1, 3));
            assert_eq!(i.disp, -0x80);
            // mov rax, [rbx + 0x7f]
            let positive = decoded(&[0x48, 0x8b, 0x43, 0x7f]);
            assert_eq!(positive.disp, 0x7f);
        }

        #[test]
        fn base_plus_disp32_sign_extends() {
            // mov rax, [rbx - 0x12345]
            let bytes = &[0x48, 0x8b, 0x83, 0xbb, 0xdc, 0xfe, 0xff];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base), (2, 3));
            assert_eq!(i.disp, -0x12345);
            // mov rax, [rbx + 0x7fffffff]
            let high = decoded(&[0x48, 0x8b, 0x83, 0xff, 0xff, 0xff, 0x7f]);
            assert_eq!(high.disp, 0x7fff_ffff);
        }

        #[test]
        fn rsp_base_needs_a_sib_byte() {
            // mov rax, [rsp]
            let bytes = &[0x48, 0x8b, 0x04, 0x24];
            let i = decoded(bytes);
            assert_eq!(i.base, 4);
            assert_eq!(i.index, NO_REG);
            assert_eq!(i.len, 4);
            // mov rax, [rsp + 0x10]
            let with_disp = decoded(&[0x48, 0x8b, 0x44, 0x24, 0x10]);
            assert_eq!(
                (with_disp.base, with_disp.disp, with_disp.len),
                (4, 0x10, 5)
            );
        }

        #[test]
        fn r12_base_needs_a_sib_byte_like_rsp() {
            // mov rax, [r12]
            let bytes = &[0x49, 0x8b, 0x04, 0x24];
            let i = decoded(bytes);
            assert_eq!(i.base, 12);
            assert_eq!(i.index, NO_REG);
            assert_eq!(i.len, 4);
        }

        #[test]
        fn rbp_base_is_encoded_with_a_zero_disp8() {
            // mov rax, [rbp]
            let bytes = &[0x48, 0x8b, 0x45, 0x00];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base, i.disp), (1, 5, 0));
            assert!(!i.rip_rel);
        }

        #[test]
        fn r13_base_is_encoded_with_a_zero_disp8() {
            // mov rax, [r13]
            let bytes = &[0x49, 0x8b, 0x45, 0x00];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base, i.disp), (1, 13, 0));
            assert!(!i.rip_rel);
        }

        #[test]
        fn rip_relative_has_no_base_and_a_signed_disp32() {
            // mov rax, [rip + 0x100]
            let bytes = &[0x48, 0x8b, 0x05, 0x00, 0x01, 0x00, 0x00];
            let i = decoded(bytes);
            assert!(i.rip_rel);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.index, NO_REG);
            assert_eq!(i.disp, 0x100);
            assert_eq!(i.md, 0);
            // mov rax, [rip - 0x20]
            let backwards = decoded(&[0x48, 0x8b, 0x05, 0xe0, 0xff, 0xff, 0xff]);
            assert_eq!(backwards.disp, -0x20);
            assert!(backwards.rip_rel);
        }

        #[test]
        fn rip_relative_ignores_rex_b() {
            // mod=00 rm=101 is RIP-relative even when REX.B would make the register r13.
            let i = decoded(&[0x49, 0x8b, 0x05, 0x10, 0, 0, 0]);
            assert!(i.rip_rel);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.len, 7);
        }

        #[test]
        fn rip_relative_with_a_trailing_immediate_counts_both() {
            // add dword ptr [rip + 0x10], 5
            let bytes = &[0x83, 0x05, 0x10, 0x00, 0x00, 0x00, 0x05];
            let i = decoded(bytes);
            assert!(i.rip_rel);
            assert_eq!(i.imm, 5);
            assert_eq!(i.len as usize, bytes.len());
        }

        #[test]
        fn absolute_disp32_through_sib() {
            // mov eax, [0x12345678]
            let bytes = &[0x8b, 0x04, 0x25, 0x78, 0x56, 0x34, 0x12];
            let i = decoded(bytes);
            assert!(!i.rip_rel);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.index, NO_REG);
            assert_eq!(i.disp, 0x1234_5678);
        }

        #[test]
        fn scaled_index_without_base_keeps_the_index() {
            // mov eax, [rcx*8 + 0x1000]
            let bytes = &[0x8b, 0x04, 0xcd, 0x00, 0x10, 0x00, 0x00];
            let i = decoded(bytes);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.index, 1);
            assert_eq!(i.scale, 3);
            assert_eq!(i.disp, 0x1000);
        }

        #[test]
        fn sib_base_5_with_mod_0_has_no_base_even_for_r13() {
            // 4b 8b 04 ed disp32: REX.XB, SIB base field 5 with mod 0 means "no base".
            let i = decoded(&[0x4b, 0x8b, 0x04, 0xed, 0x10, 0, 0, 0]);
            assert_eq!(i.base, NO_REG);
            assert_eq!(i.index, 13);
            assert_eq!(i.scale, 3);
            assert_eq!(i.disp, 0x10);
        }

        #[test]
        fn sib_base_5_with_mod_1_is_rbp() {
            // mov rax, [rbp + rcx*2 + 8]
            let bytes = &[0x48, 0x8b, 0x44, 0x4d, 0x08];
            let i = decoded(bytes);
            assert_eq!((i.md, i.base, i.index, i.scale, i.disp), (1, 5, 1, 1, 8));
        }

        #[test]
        fn sib_index_4_means_none_but_r12_index_is_real() {
            // .byte 0x48, 0x8b, 0x04, 0x20
            let none = decoded(&[0x48, 0x8b, 0x04, 0x20]);
            assert_eq!(none.index, NO_REG);
            // mov rax, [rax + r12*2]
            let r12 = decoded(&[0x4a, 0x8b, 0x04, 0x60]);
            assert_eq!(r12.index, 12);
            assert_eq!(r12.scale, 1);
        }

        #[test]
        fn all_four_scales() {
            for log2 in 0..4u8 {
                let sib = (log2 << 6) | (1 << 3) | 3;
                let i = decoded(&[0x48, 0x8b, 0x04, sib]);
                assert_eq!(i.scale, log2);
                assert_eq!((i.base, i.index), (3, 1));
            }
        }

        #[test]
        fn modrm_reg_field_is_the_group_extension() {
            // ff /4 (jmp r/m64) and ff /6 (push r/m64): reg & 7 selects the operation.
            // jmp rax
            let jmp = decoded(&[0xff, 0xe0]);
            assert_eq!(jmp.reg & 7, 4);
            // push qword ptr [rax]
            let push = decoded(&[0xff, 0x30]);
            assert_eq!(push.reg & 7, 6);
        }

        #[test]
        fn modrm_fields_are_all_exposed() {
            let i = decoded(&[0x8b, 0b10_011_100, 0b11_010_001, 1, 0, 0, 0]);
            assert_eq!(i.modrm, 0b10_011_100);
            assert_eq!(i.md, 2);
            assert_eq!(i.reg, 3);
            assert_eq!(i.base, 1);
            assert_eq!(i.index, 2);
            assert_eq!(i.scale, 3);
            assert_eq!(i.disp, 1);
        }

        #[test]
        fn address_size_prefix_does_not_change_the_encoding_length() {
            assert_eq!(decoded(&[0x67, 0x8b, 0x04, 0x24]).len, 4);
            assert_eq!(decoded(&[0x67, 0x8b, 0x05, 1, 0, 0, 0]).len, 7);
        }
    }

    mod immediates {
        use super::*;

        #[test]
        fn imm8_is_sign_extended() {
            // add rax, -1
            let i = decoded(&[0x48, 0x83, 0xc0, 0xff]);
            assert_eq!(i.imm, -1);
            let max = decoded(&[0x83, 0xc0, 0x7f]);
            assert_eq!(max.imm, 0x7f);
            let min = decoded(&[0x83, 0xc0, 0x80]);
            assert_eq!(min.imm, -128);
        }

        #[test]
        fn iz_is_32_bits_without_prefix_and_sign_extended() {
            // add eax, 0x12345678
            let i = decoded(&[0x05, 0x78, 0x56, 0x34, 0x12]);
            assert_eq!(i.imm, 0x1234_5678);
            assert_eq!(i.len, 5);
            let negative = decoded(&[0x05, 0x00, 0x00, 0x00, 0x80]);
            assert_eq!(negative.imm, i64::from(i32::MIN));
        }

        #[test]
        fn iz_is_16_bits_with_operand_size_prefix() {
            // add ax, 0x1234
            let bytes = &[0x66, 0x05, 0x34, 0x12];
            let i = decoded(bytes);
            assert_eq!(i.imm, 0x1234);
            assert_eq!(i.len, 4);
            let negative = decoded(&[0x66, 0x05, 0x00, 0x80]);
            assert_eq!(negative.imm, -0x8000);
        }

        #[test]
        fn iz_stays_32_bits_with_rex_w_and_sign_extends_to_64() {
            // add rax, 0x12345678
            let bytes = &[0x48, 0x05, 0x78, 0x56, 0x34, 0x12];
            let i = decoded(bytes);
            assert_eq!(i.imm, 0x1234_5678);
            assert_eq!(i.len, 6);
            // mov rax, -2
            let negative = decoded(&[0x48, 0xc7, 0xc0, 0xfe, 0xff, 0xff, 0xff]);
            assert_eq!(negative.imm, -2);
        }

        #[test]
        fn rex_w_with_operand_size_prefix_uses_a_32_bit_immediate() {
            // opsize is 8 under REX.W even with 66, so Iz is 4 bytes.
            let i = decoded(&[0x66, 0x48, 0x05, 1, 0, 0, 0]);
            assert_eq!(i.len, 7);
            assert_eq!(i.imm, 1);
        }

        #[test]
        fn mov_r64_imm64() {
            // movabs rax, 0x1122334455667788
            let bytes = &[0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11];
            let i = decoded(bytes);
            assert_eq!(i.len, 10);
            assert_eq!(i.imm, 0x1122_3344_5566_7788);
            assert_eq!(i.op, 0xb8);
        }

        #[test]
        fn mov_r32_imm32_and_r16_imm16() {
            // mov ecx, 0xdeadbeef
            let dword = decoded(&[0xb9, 0xef, 0xbe, 0xad, 0xde]);
            assert_eq!(dword.len, 5);
            assert_eq!(dword.imm, i64::from(0xdead_beefu32 as i32));
            // mov cx, 0x1234
            let word = decoded(&[0x66, 0xb9, 0x34, 0x12]);
            assert_eq!(word.len, 4);
            assert_eq!(word.imm, 0x1234);
        }

        #[test]
        fn mov_r8_imm8_for_all_byte_registers() {
            for opcode in 0xb0..=0xb7u8 {
                let i = decoded(&[opcode, 0x80]);
                assert_eq!(i.len, 2);
                assert_eq!(i.imm, -128);
            }
        }

        #[test]
        fn mov_r_imm_with_rex_b_reaches_r8_to_r15() {
            // mov r8d, 5
            let bytes = &[0x41, 0xb8, 0x05, 0x00, 0x00, 0x00];
            let i = decoded(bytes);
            assert_eq!(i.rex & REX_B, REX_B);
            assert_eq!(i.op, 0xb8);
            assert_eq!(i.imm, 5);
        }

        #[test]
        fn moffs_is_eight_bytes_in_64_bit_mode() {
            // movabs al, [0x1122334455667788]
            let bytes = &[0xa0, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11];
            let i = decoded(bytes);
            assert_eq!(i.op, 0xa0);
            assert_eq!(i.len, 9);
            assert_eq!(i.imm, 0x1122_3344_5566_7788);
            // movabs [0x1122334455667788], rax
            let store = decoded(&[0x48, 0xa3, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
            assert_eq!(store.op, 0xa3);
            assert_eq!(store.len, 10);
        }

        #[test]
        fn moffs_shrinks_to_four_bytes_with_the_address_size_prefix() {
            let i = decoded(&[0x67, 0xa1, 0x78, 0x56, 0x34, 0x12]);
            assert!(i.addr32);
            assert_eq!(i.len, 6);
            assert_eq!(i.imm, 0x1234_5678);
            let with_rex = decoded(&[0x48, 0x67, 0xa1, 0x78, 0x56, 0x34, 0x12]);
            assert_eq!(with_rex.len, 7, "the dropped REX still occupies a byte");
            assert_eq!(with_rex.imm, 0x1234_5678);
            assert!(!with_rex.rex_present);
        }

        #[test]
        fn ret_imm16_is_zero_extended() {
            // ret 0xffff
            let i = decoded(&[0xc2, 0xff, 0xff]);
            assert_eq!(i.op, 0xc2);
            assert_eq!(i.imm, 0xffff);
            assert_eq!(i.len, 3);
            let far = decoded(&[0xca, 0x00, 0x80]);
            assert_eq!(far.imm, 0x8000);
        }

        #[test]
        fn enter_has_a_zero_extended_size_and_a_nesting_level() {
            let i = decoded(&[0xc8, 0x00, 0x80, 0x1f]);
            assert_eq!(i.len, 4);
            assert_eq!(i.imm, 0x8000);
            assert_eq!(i.imm2, 0x1f);
            let max_level = decoded(&[0xc8, 0x10, 0x00, 0xff]);
            assert_eq!(max_level.imm2, 0xff);
        }

        #[test]
        fn relative_branches_are_signed() {
            let backward = decoded(&[0xeb, 0xfe]);
            assert_eq!(backward.imm, -2);
            let call = decoded(&[0xe8, 0xfb, 0xff, 0xff, 0xff]);
            assert_eq!((call.op, call.imm, call.len), (0xe8, -5, 5));
            let jmp = decoded(&[0xe9, 0x00, 0x01, 0x00, 0x00]);
            assert_eq!((jmp.imm, jmp.len), (0x100, 5));
        }

        #[test]
        fn jcc_short_and_loop_forms_take_rel8() {
            for opcode in (0x70..=0x7fu8).chain(0xe0..=0xe3) {
                let i = decoded(&[opcode, 0x80]);
                assert_eq!(i.len, 2, "{opcode:#x}");
                assert_eq!(i.imm, -128, "{opcode:#x}");
            }
        }

        #[test]
        fn rel32_ignores_the_operand_size_prefix() {
            let i = decoded(&[0x66, 0xe8, 1, 0, 0, 0]);
            assert_eq!(i.len, 6);
            assert_eq!(i.imm, 1);
        }

        #[test]
        fn push_imm_forms() {
            // push 0x12345678
            let imm32 = decoded(&[0x68, 0x78, 0x56, 0x34, 0x12]);
            assert_eq!((imm32.op, imm32.len, imm32.imm), (0x68, 5, 0x1234_5678));
            // push -2
            let imm8 = decoded(&[0x6a, 0xfe]);
            assert_eq!((imm8.op, imm8.len, imm8.imm), (0x6a, 2, -2));
            let imm16 = decoded(&[0x66, 0x68, 0x34, 0x12]);
            assert_eq!((imm16.len, imm16.imm), (4, 0x1234));
        }

        #[test]
        fn imul_imm_forms() {
            // imul rax, rbx, 0x1000
            let imm32 = decoded(&[0x48, 0x69, 0xc3, 0x00, 0x10, 0x00, 0x00]);
            assert_eq!(imm32.op, 0x69);
            assert_eq!(imm32.imm, 0x1000);
            // imul rax, rbx, 5
            let imm8 = decoded(&[0x48, 0x6b, 0xc3, 0x05]);
            assert_eq!(imm8.op, 0x6b);
            assert_eq!(imm8.imm, 5);
        }

        #[test]
        fn group1_immediate_forms() {
            // add byte ptr [rax], 1
            let eighty = decoded(&[0x80, 0x00, 0x01]);
            assert_eq!((eighty.op, eighty.imm), (0x80, 1));
            // add dword ptr [rax], 0x1000
            let eighty_one = decoded(&[0x81, 0x00, 0x00, 0x10, 0x00, 0x00]);
            assert_eq!((eighty_one.op, eighty_one.imm), (0x81, 0x1000));
            // add dword ptr [rax], -1
            let eighty_three = decoded(&[0x83, 0x00, 0xff]);
            assert_eq!((eighty_three.op, eighty_three.imm), (0x83, -1));
        }

        #[test]
        fn shift_group_immediates() {
            // shl rax, 5
            let by_imm = decoded(&[0x48, 0xc1, 0xe0, 0x05]);
            assert_eq!((by_imm.op, by_imm.imm, by_imm.len), (0xc1, 5, 4));
            // shl al, 3
            let byte = decoded(&[0xc0, 0xe0, 0x03]);
            assert_eq!((byte.op, byte.imm), (0xc0, 3));
            // shl rax, 1
            let by_one = decoded(&[0x48, 0xd1, 0xe0]);
            assert_eq!(by_one.op, 0xd1);
            assert_eq!(by_one.len, 3);
            // shl rax, cl
            let by_cl = decoded(&[0x48, 0xd3, 0xe0]);
            assert_eq!(by_cl.op, 0xd3);
            assert_eq!(by_cl.len, 3);
        }

        #[test]
        fn mov_rm_imm_forms() {
            // mov byte ptr [rax], 7
            let byte = decoded(&[0xc6, 0x00, 0x07]);
            assert_eq!((byte.op, byte.imm, byte.len), (0xc6, 7, 3));
            // mov dword ptr [rax], 0x11223344
            let dword = decoded(&[0xc7, 0x00, 0x44, 0x33, 0x22, 0x11]);
            assert_eq!((dword.op, dword.imm, dword.len), (0xc7, 0x1122_3344, 6));
            // mov word ptr [rax], 0x1122
            let word = decoded(&[0x66, 0xc7, 0x00, 0x22, 0x11]);
            assert_eq!((word.imm, word.len), (0x1122, 5));
        }

        #[test]
        fn test_group3_takes_an_immediate_only_for_slash_0_and_1() {
            // test byte ptr [rax], 0x55
            let test_byte = decoded(&[0xf6, 0x00, 0x55]);
            assert_eq!(
                (test_byte.op, test_byte.imm, test_byte.len),
                (0xf6, 0x55, 3)
            );
            // test dword ptr [rax], 0x55667788
            let test_dword = decoded(&[0xf7, 0x00, 0x88, 0x77, 0x66, 0x55]);
            assert_eq!(
                (test_dword.op, test_dword.imm, test_dword.len),
                (0xf7, 0x5566_7788, 6)
            );
            let test_word = decoded(&[0x66, 0xf7, 0x00, 0x34, 0x12]);
            assert_eq!((test_word.imm, test_word.len), (0x1234, 5));
            // /1 is an undocumented alias of /0 and also carries the immediate.
            let alias = decoded(&[0xf6, 0x08, 0x55]);
            assert_eq!((alias.imm, alias.len), (0x55, 3));
            for slash in 2..8u8 {
                let modrm = 0xc0 | (slash << 3);
                let byte = decoded(&[0xf6, modrm]);
                assert_eq!(byte.len, 2, "f6 /{slash}");
                assert_eq!(byte.imm, 0, "f6 /{slash}");
                let dword = decoded(&[0xf7, modrm]);
                assert_eq!(dword.len, 2, "f7 /{slash}");
            }
        }

        #[test]
        fn accumulator_alu_immediates() {
            for row in 0..8u8 {
                let base = row << 3;
                let byte = decoded(&[base + 4, 0x80]);
                assert_eq!((byte.len, byte.imm), (2, -128), "{base:#x}+4");
                assert!(!byte.has_modrm);
                let dword = decoded(&[base + 5, 1, 0, 0, 0]);
                assert_eq!((dword.len, dword.imm), (5, 1), "{base:#x}+5");
                assert!(!dword.has_modrm);
            }
        }

        #[test]
        fn test_al_and_eax_immediates() {
            assert_eq!(decoded(&[0xa8, 0x01]).len, 2);
            assert_eq!(decoded(&[0xa9, 1, 0, 0, 0]).len, 5);
            assert_eq!(decoded(&[0x66, 0xa9, 1, 0]).len, 4);
        }

        #[test]
        fn io_immediate_port_forms() {
            for opcode in [0xe4u8, 0xe5, 0xe6, 0xe7] {
                let i = decoded(&[opcode, 0x80]);
                assert_eq!(i.len, 2);
                assert_eq!(i.imm, -128);
            }
            assert_eq!(decoded(&[0xec]).len, 1);
        }

        #[test]
        fn int_imm8() {
            // int 0x80
            let i = decoded(&[0xcd, 0x80]);
            assert_eq!((i.op, i.len), (0xcd, 2));
            // int3
            assert_eq!(decoded(&[0xcc]).len, 1);
        }

        #[test]
        fn two_byte_map_immediates() {
            // pshufd xmm0, xmm1, 0x1b
            let pshufd = decoded(&[0x66, 0x0f, 0x70, 0xc1, 0x1b]);
            assert_eq!((pshufd.op, pshufd.imm), (0x170, 0x1b));
            // psllw xmm0, 3
            let shift = decoded(&[0x66, 0x0f, 0x71, 0xf0, 0x03]);
            assert_eq!((shift.op, shift.imm), (0x171, 3));
            // shld rax, rbx, 4
            let shld = decoded(&[0x48, 0x0f, 0xa4, 0xd8, 0x04]);
            assert_eq!((shld.op, shld.imm), (0x1a4, 4));
            // shld rax, rbx, cl
            let shld_cl = decoded(&[0x48, 0x0f, 0xa5, 0xd8]);
            assert_eq!((shld_cl.op, shld_cl.len), (0x1a5, 4));
            // shrd rax, rbx, 4
            let shrd = decoded(&[0x48, 0x0f, 0xac, 0xd8, 0x04]);
            assert_eq!((shrd.op, shrd.imm), (0x1ac, 4));
            // bt rax, 3
            let bt = decoded(&[0x48, 0x0f, 0xba, 0xe0, 0x03]);
            assert_eq!((bt.op, bt.imm), (0x1ba, 3));
            // cmpps xmm0, xmm1, 2
            let cmpps = decoded(&[0x0f, 0xc2, 0xc1, 0x02]);
            assert_eq!((cmpps.op, cmpps.imm), (0x1c2, 2));
            // pinsrw xmm0, eax, 3
            let pinsrw = decoded(&[0x66, 0x0f, 0xc4, 0xc0, 0x03]);
            assert_eq!((pinsrw.op, pinsrw.imm), (0x1c4, 3));
            // pextrw eax, xmm1, 3
            let pextrw = decoded(&[0x66, 0x0f, 0xc5, 0xc1, 0x03]);
            assert_eq!((pextrw.op, pextrw.imm), (0x1c5, 3));
            // shufps xmm0, xmm1, 0xe4
            let shufps = decoded(&[0x0f, 0xc6, 0xc1, 0xe4]);
            assert_eq!(shufps.op, 0x1c6);
            assert_eq!(shufps.imm, -28, "0xe4 sign-extended");
        }

        #[test]
        fn two_byte_opcodes_next_to_immediate_ones_have_none() {
            for opcode in [0x6fu8, 0x74, 0x75, 0x76, 0xa5, 0xad, 0xbb, 0xc1, 0xc3, 0xc7] {
                let i = decoded(&[0x0f, opcode, 0xc1, 0x55]);
                assert_eq!(i.len, 3, "0f {opcode:#x}");
                assert_eq!(i.imm, 0, "0f {opcode:#x}");
            }
        }

        #[test]
        fn memory_operand_before_the_immediate() {
            // mov dword ptr [rbx + rcx*4 + 0x10], 0x11223344
            let bytes = &[0xc7, 0x44, 0x8b, 0x10, 0x44, 0x33, 0x22, 0x11];
            let i = decoded(bytes);
            assert_eq!(i.disp, 0x10);
            assert_eq!(i.imm, 0x1122_3344);
            assert_eq!(i.len as usize, bytes.len());
        }
    }

    mod invalid_encodings {
        use super::*;

        const INVALID_IN_64_BIT_MODE: [u8; 21] = [
            0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f, 0x60, 0x61, 0x62,
            0x82, 0x9a, 0xce, 0xd4, 0xd5, 0xd6, 0xea,
        ];

        #[test]
        fn opcodes_removed_in_long_mode_are_invalid() {
            for opcode in &INVALID_IN_64_BIT_MODE {
                assert_eq!(
                    decode(&[*opcode, 0xc0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]).err(),
                    Some(DecodeErr::Invalid),
                    "{opcode:#x}"
                );
            }
        }

        #[test]
        fn invalid_opcodes_stay_invalid_behind_prefixes() {
            for prefix in [&[0x66u8][..], &[0x48], &[0xf0], &[0x67, 0x66], &[0x2e]] {
                let mut bytes = prefix.to_vec();
                bytes.extend_from_slice(&[0x06, 0x90, 0x90, 0x90]);
                assert_eq!(
                    decode(&bytes).err(),
                    Some(DecodeErr::Invalid),
                    "{prefix:02x?}"
                );
            }
        }

        #[test]
        fn invalid_one_byte_opcodes_are_valid_in_the_two_byte_map() {
            // 0f 06 is clts, 0f 07 sysret, 0f 0e femms, 0f 60 punpcklbw, 0f 82 jb: only the one-byte
            // map entries were removed.
            for opcode in [
                0x06u8, 0x07, 0x0e, 0x60, 0x61, 0x62, 0x82, 0xc4, 0xc5, 0xd4, 0xd5, 0xd6,
            ] {
                assert!(
                    decode(&[0x0f, opcode, 0xc0, 0x90, 0x90, 0x90, 0x90, 0x90]).is_ok(),
                    "0f {opcode:#x}"
                );
            }
        }

        #[test]
        fn evex_decodes_instead_of_being_bound() {
            // vmovaps zmm0, zmm1: 62 is EVEX in 64-bit mode, never `bound`
            let evex = [0x62, 0xf1, 0x7c, 0x48, 0x28, 0xc1];
            let i = decode(&evex).unwrap();
            assert!(i.evex);
            assert_eq!(i.len, 6);
        }

        #[test]
        fn neighbouring_valid_opcodes_still_decode() {
            for opcode in [0x63u8, 0x9b, 0xcc, 0xcf, 0xd7, 0xd8, 0xe8] {
                let result = decode(&[opcode, 0xc0, 0, 0, 0, 0, 0, 0]);
                assert!(result.is_ok(), "{opcode:#x} unexpectedly {result:?}");
            }
        }
    }

    mod length_limit_and_truncation {
        use super::*;

        #[test]
        fn empty_input_is_truncated() {
            assert_eq!(decode(&[]).err(), Some(DecodeErr::Truncated));
        }

        #[test]
        fn prefix_only_is_truncated() {
            for prefix in [
                0x66u8, 0x67, 0xf0, 0xf2, 0xf3, 0x26, 0x2e, 0x36, 0x3e, 0x64, 0x65, 0x40, 0x4f,
            ] {
                assert_eq!(
                    decode(&[prefix]).err(),
                    Some(DecodeErr::Truncated),
                    "{prefix:#x}"
                );
            }
        }

        #[test]
        fn missing_modrm_sib_disp_and_immediate_bytes_are_truncated() {
            let cases: &[&[u8]] = &[
                &[0x8b],
                &[0x8b, 0x04],
                &[0x8b, 0x44, 0x24],
                &[0x8b, 0x84, 0x24, 0, 0, 0],
                &[0x8b, 0x05, 0, 0, 0],
                &[0x83, 0xc0],
                &[0x81, 0xc0, 1, 0, 0],
                &[0x66, 0x81, 0xc0, 1],
                &[0xb8, 1, 0, 0],
                &[0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7],
                &[0xa1, 1, 2, 3, 4, 5, 6, 7],
                &[0xc2, 0x10],
                &[0xc8, 0x10, 0x00],
                &[0xe9, 0, 0, 0],
                &[0xeb],
                &[0x0f, 0x84, 0, 0, 0],
                &[0xf6, 0x00],
                &[0xf7, 0xc0, 1, 0, 0],
            ];
            for bytes in cases {
                assert_eq!(
                    decode(bytes).err(),
                    Some(DecodeErr::Truncated),
                    "{bytes:02x?}"
                );
            }
        }

        #[test]
        fn exactly_fifteen_bytes_decode() {
            // fs: addr32 lock add qword ptr [rax*8 + 0x12345678], 1 -- REX, opcode, modrm, SIB,
            // disp32, imm32 plus three legacy prefixes.
            let bytes = [
                0x64, 0x67, 0xf0, 0x48, 0x81, 0x84, 0xc0, 0x78, 0x56, 0x34, 0x12, 0x01, 0x00, 0x00,
                0x00,
            ];
            assert_eq!(bytes.len(), 15);
            let i = decoded(&bytes);
            assert_eq!(i.len, 15);
            assert!(i.lock && i.addr32);
            assert_eq!(i.seg as usize, SEG_FS);
            assert_eq!(i.disp, 0x1234_5678);
            assert_eq!(i.imm, 1);
        }

        #[test]
        fn a_sixteenth_byte_makes_it_invalid() {
            let mut bytes = vec![
                0x2e, 0x64, 0x67, 0xf0, 0x48, 0x81, 0x84, 0xc0, 0x78, 0x56, 0x34, 0x12, 1, 0, 0, 0,
            ];
            assert_eq!(bytes.len(), 16);
            assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
            bytes.truncate(15);
            assert_eq!(decode(&bytes).err(), Some(DecodeErr::Truncated));
        }

        #[test]
        fn fifteen_prefixes_need_a_sixteenth_byte_to_become_invalid() {
            let mut bytes = vec![0x66u8; 15];
            assert_eq!(decode(&bytes).err(), Some(DecodeErr::Truncated));
            bytes.push(0x90);
            assert_eq!(decode(&bytes).err(), Some(DecodeErr::Invalid));
        }

        #[test]
        fn fourteen_prefixes_and_an_opcode_is_the_longest_nop() {
            let mut bytes = vec![0x66u8; 14];
            bytes.push(0x90);
            assert_eq!(decoded(&bytes).len, 15);
        }

        #[test]
        fn decoding_stops_at_the_instruction_end() {
            assert_eq!(decoded(&[0x90, 0xff, 0xff, 0xff]).len, 1);
            assert_eq!(decoded(&[0x48, 0x89, 0xe5, 0xff, 0xff]).len, 3);
            assert_eq!(decoded(&[0xc3, 0x00]).len, 1);
        }
    }

    mod instruction_lengths {
        use super::*;

        /// Encodings assembled by clang; the decoder must agree with its byte counts.
        const ASSEMBLED: &[&[u8]] = &[
            // nop
            &[0x90],
            // ret
            &[0xc3],
            // hlt
            &[0xf4],
            // cli
            &[0xfa],
            // sti
            &[0xfb],
            // pause
            &[0xf3, 0x90],
            // leave
            &[0xc9],
            // cpuid
            &[0x0f, 0xa2],
            // rdtsc
            &[0x0f, 0x31],
            // syscall
            &[0x0f, 0x05],
            // sysretq
            &[0x48, 0x0f, 0x07],
            // swapgs
            &[0x0f, 0x01, 0xf8],
            // iretq
            &[0x48, 0xcf],
            // push rax
            &[0x50],
            // push r15
            &[0x41, 0x57],
            // pop rbx
            &[0x5b],
            // pop r12
            &[0x41, 0x5c],
            // mov rax, rbx
            &[0x48, 0x89, 0xd8],
            // mov eax, ebx
            &[0x89, 0xd8],
            // mov ax, bx
            &[0x66, 0x89, 0xd8],
            // mov al, bl
            &[0x88, 0xd8],
            // mov sil, dil
            &[0x40, 0x88, 0xfe],
            // mov r8b, r15b
            &[0x45, 0x88, 0xf8],
            // mov rax, [rbx]
            &[0x48, 0x8b, 0x03],
            // mov [rbx], rax
            &[0x48, 0x89, 0x03],
            // mov rax, [rbx + 8]
            &[0x48, 0x8b, 0x43, 0x08],
            // mov rax, [rbx + 0x1000]
            &[0x48, 0x8b, 0x83, 0x00, 0x10, 0x00, 0x00],
            // mov rax, [rbx + rcx]
            &[0x48, 0x8b, 0x04, 0x0b],
            // mov rax, [rbx + rcx*8 + 0x10]
            &[0x48, 0x8b, 0x44, 0xcb, 0x10],
            // mov rax, [rbx + rcx*8 + 0x1000]
            &[0x48, 0x8b, 0x84, 0xcb, 0x00, 0x10, 0x00, 0x00],
            // mov rax, fs:[0x28]
            &[0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
            // mov rax, gs:[rax]
            &[0x65, 0x48, 0x8b, 0x00],
            // mov eax, [rip + 0x1234]
            &[0x8b, 0x05, 0x34, 0x12, 0x00, 0x00],
            // lea rax, [rip + 0x1234]
            &[0x48, 0x8d, 0x05, 0x34, 0x12, 0x00, 0x00],
            // lea r9, [r10 + r11*4 - 8]
            &[0x4f, 0x8d, 0x4c, 0x9a, 0xf8],
            // movzx eax, byte ptr [rax]
            &[0x0f, 0xb6, 0x00],
            // movzx eax, word ptr [rax]
            &[0x0f, 0xb7, 0x00],
            // movsx rax, byte ptr [rax]
            &[0x48, 0x0f, 0xbe, 0x00],
            // movsxd rax, dword ptr [rax]
            &[0x48, 0x63, 0x00],
            // add rax, 1
            &[0x48, 0x83, 0xc0, 0x01],
            // add rax, 0x1000
            &[0x48, 0x05, 0x00, 0x10, 0x00, 0x00],
            // sub dword ptr [rsp + 8], 1
            &[0x83, 0x6c, 0x24, 0x08, 0x01],
            // and eax, 0xffff
            &[0x25, 0xff, 0xff, 0x00, 0x00],
            // or byte ptr [rax], 0x80
            &[0x80, 0x08, 0x80],
            // xor eax, eax
            &[0x31, 0xc0],
            // cmp rax, [rbx]
            &[0x48, 0x3b, 0x03],
            // test rax, rax
            &[0x48, 0x85, 0xc0],
            // test al, 1
            &[0xa8, 0x01],
            // test eax, 0x80000000
            &[0xa9, 0x00, 0x00, 0x00, 0x80],
            // inc qword ptr [rax]
            &[0x48, 0xff, 0x00],
            // dec ecx
            &[0xff, 0xc9],
            // neg rax
            &[0x48, 0xf7, 0xd8],
            // not rax
            &[0x48, 0xf7, 0xd0],
            // mul rbx
            &[0x48, 0xf7, 0xe3],
            // imul rax, rbx
            &[0x48, 0x0f, 0xaf, 0xc3],
            // imul rax, [rbx], 10
            &[0x48, 0x6b, 0x03, 0x0a],
            // div rcx
            &[0x48, 0xf7, 0xf1],
            // idiv dword ptr [rcx]
            &[0xf7, 0x39],
            // shl rax, cl
            &[0x48, 0xd3, 0xe0],
            // shr rax, 3
            &[0x48, 0xc1, 0xe8, 0x03],
            // sar eax, 1
            &[0xd1, 0xf8],
            // rol rax, 8
            &[0x48, 0xc1, 0xc0, 0x08],
            // ror dword ptr [rax], cl
            &[0xd3, 0x08],
            // shld rax, rbx, 4
            &[0x48, 0x0f, 0xa4, 0xd8, 0x04],
            // bt rax, rbx
            &[0x48, 0x0f, 0xa3, 0xd8],
            // bts qword ptr [rax], 5
            &[0x48, 0x0f, 0xba, 0x28, 0x05],
            // bsf rax, rbx
            &[0x48, 0x0f, 0xbc, 0xc3],
            // bsr rax, [rbx]
            &[0x48, 0x0f, 0xbd, 0x03],
            // bswap eax
            &[0x0f, 0xc8],
            // bswap rax
            &[0x48, 0x0f, 0xc8],
            // xchg rax, rbx
            &[0x48, 0x93],
            // xchg [rax], rbx
            &[0x48, 0x87, 0x18],
            // xadd [rax], rbx
            &[0x48, 0x0f, 0xc1, 0x18],
            // cmpxchg [rax], rbx
            &[0x48, 0x0f, 0xb1, 0x18],
            // lock cmpxchg [rax], rbx
            &[0xf0, 0x48, 0x0f, 0xb1, 0x18],
            // cmpxchg8b [rax]
            &[0x0f, 0xc7, 0x08],
            // cmovne rax, rbx
            &[0x48, 0x0f, 0x45, 0xc3],
            // sete al
            &[0x0f, 0x94, 0xc0],
            // call rax
            &[0xff, 0xd0],
            // call [rax + 8]
            &[0xff, 0x50, 0x08],
            // jmp [rax*8 + 0x1000]
            &[0xff, 0x24, 0xc5, 0x00, 0x10, 0x00, 0x00],
            // push qword ptr [rax]
            &[0xff, 0x30],
            // push 5
            &[0x6a, 0x05],
            // push 0x1000
            &[0x68, 0x00, 0x10, 0x00, 0x00],
            // ret 8
            &[0xc2, 0x08, 0x00],
            // enter 0x10, 2
            &[0xc8, 0x10, 0x00, 0x02],
            // int 0x80
            &[0xcd, 0x80],
            // rep movsb
            &[0xf3, 0xa4],
            // rep stosq
            &[0xf3, 0x48, 0xab],
            // repne scasb
            &[0xf2, 0xae],
            // movsq
            &[0x48, 0xa5],
            // cmpsb
            &[0xa6],
            // lodsd
            &[0xad],
            // in al, dx
            &[0xec],
            // out dx, al
            &[0xee],
            // in eax, 0x60
            &[0xe5, 0x60],
            // out 0x80, al
            &[0xe6, 0x80],
            // cwde
            &[0x98],
            // cdqe
            &[0x48, 0x98],
            // cdq
            &[0x99],
            // cqo
            &[0x48, 0x99],
            // lahf
            &[0x9f],
            // sahf
            &[0x9e],
            // pushfq
            &[0x9c],
            // popfq
            &[0x9d],
            // std
            &[0xfd],
            // cld
            &[0xfc],
            // clc
            &[0xf8],
            // stc
            &[0xf9],
            // cmc
            &[0xf5],
            // lgdt [rax]
            &[0x0f, 0x01, 0x10],
            // lidt [rax]
            &[0x0f, 0x01, 0x18],
            // sgdt [rax]
            &[0x0f, 0x01, 0x00],
            // ltr ax
            &[0x0f, 0x00, 0xd8],
            // lldt word ptr [rax]
            &[0x0f, 0x00, 0x10],
            // verr ax
            &[0x0f, 0x00, 0xe0],
            // invlpg [rax]
            &[0x0f, 0x01, 0x38],
            // mov cr3, rax
            &[0x0f, 0x22, 0xd8],
            // mov rax, cr0
            &[0x0f, 0x20, 0xc0],
            // mov r9, cr8
            &[0x45, 0x0f, 0x20, 0xc1],
            // mov dr7, rax
            &[0x0f, 0x23, 0xf8],
            // rdmsr
            &[0x0f, 0x32],
            // wrmsr
            &[0x0f, 0x30],
            // rdpmc
            &[0x0f, 0x33],
            // wbinvd
            &[0x0f, 0x09],
            // lmsw ax
            &[0x0f, 0x01, 0xf0],
            // smsw eax
            &[0x0f, 0x01, 0xe0],
            // clflush [rax]
            &[0x0f, 0xae, 0x38],
            // mfence
            &[0x0f, 0xae, 0xf0],
            // lfence
            &[0x0f, 0xae, 0xe8],
            // sfence
            &[0x0f, 0xae, 0xf8],
            // prefetcht0 [rax]
            &[0x0f, 0x18, 0x08],
            // endbr64
            &[0xf3, 0x0f, 0x1e, 0xfa],
            // ud2
            &[0x0f, 0x0b],
            // fld1
            &[0xd9, 0xe8],
            // fldz
            &[0xd9, 0xee],
            // fld qword ptr [rax]
            &[0xdd, 0x00],
            // fld tbyte ptr [rax + 0x10]
            &[0xdb, 0x68, 0x10],
            // fstp st(3)
            &[0xdd, 0xdb],
            // fadd st, st(2)
            &[0xd8, 0xc2],
            // fxch st(1)
            &[0xd9, 0xc9],
            // fnstcw [rax]
            &[0xd9, 0x38],
            // fldcw [rax]
            &[0xd9, 0x28],
            // fnstsw ax
            &[0xdf, 0xe0],
            // fxsave [rax]
            &[0x0f, 0xae, 0x00],
            // fxrstor [rax + rcx*8]
            &[0x0f, 0xae, 0x0c, 0xc8],
            // fucomip st, st(1)
            &[0xdf, 0xe9],
            // fcmovb st, st(1)
            &[0xda, 0xc1],
            // movaps xmm0, [rax]
            &[0x0f, 0x28, 0x00],
            // movdqu xmm1, [rax + 0x10]
            &[0xf3, 0x0f, 0x6f, 0x48, 0x10],
            // movdqa [rax], xmm15
            &[0x66, 0x44, 0x0f, 0x7f, 0x38],
            // movups xmm8, xmm9
            &[0x45, 0x0f, 0x10, 0xc1],
            // movss xmm0, [rax]
            &[0xf3, 0x0f, 0x10, 0x00],
            // movsd xmm0, [rax]
            &[0xf2, 0x0f, 0x10, 0x00],
            // movq xmm0, rax
            &[0x66, 0x48, 0x0f, 0x6e, 0xc0],
            // movd eax, xmm0
            &[0x66, 0x0f, 0x7e, 0xc0],
            // pxor xmm0, xmm0
            &[0x66, 0x0f, 0xef, 0xc0],
            // pshufd xmm0, xmm1, 0
            &[0x66, 0x0f, 0x70, 0xc1, 0x00],
            // pcmpeqb xmm0, [rax]
            &[0x66, 0x0f, 0x74, 0x00],
            // pmovmskb eax, xmm0
            &[0x66, 0x0f, 0xd7, 0xc0],
            // addsd xmm0, xmm1
            &[0xf2, 0x0f, 0x58, 0xc1],
            // cvtsi2sd xmm0, rax
            &[0xf2, 0x48, 0x0f, 0x2a, 0xc0],
            // cvttsd2si rax, xmm0
            &[0xf2, 0x48, 0x0f, 0x2c, 0xc0],
            // ucomisd xmm0, xmm1
            &[0x66, 0x0f, 0x2e, 0xc1],
            // sqrtpd xmm0, xmm1
            &[0x66, 0x0f, 0x51, 0xc1],
            // ldmxcsr [rax]
            &[0x0f, 0xae, 0x10],
            // stmxcsr [rax]
            &[0x0f, 0xae, 0x18],
            // movnti [rax], ecx
            &[0x0f, 0xc3, 0x08],
            // movntdq [rax], xmm0
            &[0x66, 0x0f, 0xe7, 0x00],
            // emms
            &[0x0f, 0x77],
            // pxor mm0, mm1
            &[0x0f, 0xef, 0xc1],
            // popcnt rax, rbx
            &[0xf3, 0x48, 0x0f, 0xb8, 0xc3],
            // lzcnt rax, rbx
            &[0xf3, 0x48, 0x0f, 0xbd, 0xc3],
            // tzcnt eax, ebx
            &[0xf3, 0x0f, 0xbc, 0xc3],
            // crc32 eax, bl
            &[0xf2, 0x0f, 0x38, 0xf0, 0xc3],
            // pshufb xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x00, 0xc1],
            // palignr xmm0, xmm1, 4
            &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04],
            // movbe eax, [rax]
            &[0x0f, 0x38, 0xf0, 0x00],
            // xgetbv
            &[0x0f, 0x01, 0xd0],
            // rdtscp
            &[0x0f, 0x01, 0xf9],
            // monitor
            &[0x0f, 0x01, 0xc8],
            // mwait
            &[0x0f, 0x01, 0xc9],
            // lock add dword ptr [rax], 1
            &[0xf0, 0x83, 0x00, 0x01],
            // data16 nop word ptr cs:[rax + rax]
            &[0x66, 0x2e, 0x66, 0x0f, 0x1f, 0x04, 0x00],
            // nop dword ptr [rax + rax]
            &[0x0f, 0x1f, 0x04, 0x00],
            // nop word ptr [rax + rax]
            &[0x66, 0x0f, 0x1f, 0x04, 0x00],
            // The instruction-set extensions CPUID advertises (POPCNT through AES-NI, MOVBE, CRC32,
            // FSGSBASE, fisttp): all plain non-branching instructions with ordinary ModRM operands.
            // popcnt rax, rbx
            &[0xf3, 0x48, 0x0f, 0xb8, 0xc3],
            // popcnt eax, [rsi + rdi*4 + 0x10]
            &[0xf3, 0x0f, 0xb8, 0x44, 0xbe, 0x10],
            // popcnt ax, [rip + 0x100]
            &[0x66, 0xf3, 0x0f, 0xb8, 0x05, 0x00, 0x01, 0x00, 0x00],
            // lzcnt r9, r10
            &[0xf3, 0x4d, 0x0f, 0xbd, 0xca],
            // cmpxchg16b [rsi]
            &[0x48, 0x0f, 0xc7, 0x0e],
            // cmpxchg16b [r8 + r9*8 + 0x12345678]
            &[0x4b, 0x0f, 0xc7, 0x8c, 0xc8, 0x78, 0x56, 0x34, 0x12],
            // lock cmpxchg16b [rsp]
            &[0xf0, 0x48, 0x0f, 0xc7, 0x0c, 0x24],
            // movddup xmm0, xmm1
            &[0xf2, 0x0f, 0x12, 0xc1],
            // movddup xmm9, [rax + 8]
            &[0xf2, 0x44, 0x0f, 0x12, 0x48, 0x08],
            // movsldup xmm0, [rip + 0x20]
            &[0xf3, 0x0f, 0x12, 0x05, 0x20, 0x00, 0x00, 0x00],
            // movshdup xmm1, xmm2
            &[0xf3, 0x0f, 0x16, 0xca],
            // lddqu xmm0, [rsi + rcx*2]
            &[0xf2, 0x0f, 0xf0, 0x04, 0x4e],
            // addsubps xmm0, xmm1
            &[0xf2, 0x0f, 0xd0, 0xc1],
            // haddpd xmm0, [rdi]
            &[0x66, 0x0f, 0x7c, 0x07],
            // hsubps xmm10, xmm11
            &[0xf2, 0x45, 0x0f, 0x7d, 0xd3],
            // pshufb xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x00, 0xc1],
            // pshufb xmm8, [rsi + 4]
            &[0x66, 0x44, 0x0f, 0x38, 0x00, 0x46, 0x04],
            // phaddw xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x01, 0xc1],
            // pabsd xmm2, [r12]
            &[0x66, 0x41, 0x0f, 0x38, 0x1e, 0x14, 0x24],
            // palignr xmm0, xmm1, 4
            &[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04],
            // palignr xmm9, [rax + rbx*8 + 0x1000], 0xff
            &[
                0x66, 0x44, 0x0f, 0x3a, 0x0f, 0x8c, 0xd8, 0x00, 0x10, 0x00, 0x00, 0xff,
            ],
            // pblendvb xmm1, xmm2, xmm0
            &[0x66, 0x0f, 0x38, 0x10, 0xca],
            // blendvps xmm1, [rsi], xmm0
            &[0x66, 0x0f, 0x38, 0x14, 0x0e],
            // pblendw xmm0, xmm1, 0x55
            &[0x66, 0x0f, 0x3a, 0x0e, 0xc1, 0x55],
            // pmulld xmm0, [rsi]
            &[0x66, 0x0f, 0x38, 0x40, 0x06],
            // pminud xmm3, xmm4
            &[0x66, 0x0f, 0x38, 0x3b, 0xdc],
            // ptest xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x17, 0xc1],
            // pmovsxbw xmm0, [rsi]
            &[0x66, 0x0f, 0x38, 0x20, 0x06],
            // pmovzxdq xmm5, xmm6
            &[0x66, 0x0f, 0x38, 0x35, 0xee],
            // packusdw xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x2b, 0xc1],
            // pextrb eax, xmm1, 3
            &[0x66, 0x0f, 0x3a, 0x14, 0xc8, 0x03],
            // pextrd [rsi + 8], xmm1, 2
            &[0x66, 0x0f, 0x3a, 0x16, 0x4e, 0x08, 0x02],
            // pextrq rax, xmm1, 1
            &[0x66, 0x48, 0x0f, 0x3a, 0x16, 0xc8, 0x01],
            // pextrq [r8], xmm9, 1
            &[0x66, 0x4d, 0x0f, 0x3a, 0x16, 0x08, 0x01],
            // extractps eax, xmm1, 2
            &[0x66, 0x0f, 0x3a, 0x17, 0xc8, 0x02],
            // pinsrb xmm0, eax, 5
            &[0x66, 0x0f, 0x3a, 0x20, 0xc0, 0x05],
            // pinsrq xmm0, rax, 1
            &[0x66, 0x48, 0x0f, 0x3a, 0x22, 0xc0, 0x01],
            // pinsrd xmm9, [rip + 0x44], 3
            &[
                0x66, 0x44, 0x0f, 0x3a, 0x22, 0x0d, 0x44, 0x00, 0x00, 0x00, 0x03,
            ],
            // insertps xmm0, xmm1, 0x10
            &[0x66, 0x0f, 0x3a, 0x21, 0xc1, 0x10],
            // roundps xmm0, xmm1, 9
            &[0x66, 0x0f, 0x3a, 0x08, 0xc1, 0x09],
            // roundsd xmm0, [rsi], 4
            &[0x66, 0x0f, 0x3a, 0x0b, 0x06, 0x04],
            // dpps xmm0, xmm1, 0xff
            &[0x66, 0x0f, 0x3a, 0x40, 0xc1, 0xff],
            // dppd xmm0, [rsi + 0x7f], 0x31
            &[0x66, 0x0f, 0x3a, 0x41, 0x46, 0x7f, 0x31],
            // mpsadbw xmm0, xmm1, 5
            &[0x66, 0x0f, 0x3a, 0x42, 0xc1, 0x05],
            // phminposuw xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x41, 0xc1],
            // movntdqa xmm0, [rsi]
            &[0x66, 0x0f, 0x38, 0x2a, 0x06],
            // pcmpgtq xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0x37, 0xc1],
            // pcmpistri xmm0, xmm1, 0x1a
            &[0x66, 0x0f, 0x3a, 0x63, 0xc1, 0x1a],
            // pcmpistrm xmm0, [rsi], 0x40
            &[0x66, 0x0f, 0x3a, 0x62, 0x06, 0x40],
            // pcmpestri xmm0, xmm1, 0x0c
            &[0x66, 0x0f, 0x3a, 0x61, 0xc1, 0x0c],
            // pcmpestrm xmm9, [r10 + 0x10], 0x46
            &[0x66, 0x45, 0x0f, 0x3a, 0x60, 0x4a, 0x10, 0x46],
            // crc32 eax, bl
            &[0xf2, 0x0f, 0x38, 0xf0, 0xc3],
            // crc32 rax, rbx
            &[0xf2, 0x48, 0x0f, 0x38, 0xf1, 0xc3],
            // crc32 eax, bx
            &[0x66, 0xf2, 0x0f, 0x38, 0xf1, 0xc3],
            // crc32 eax, byte ptr [rsi]
            &[0xf2, 0x0f, 0x38, 0xf0, 0x06],
            // crc32 r9, qword ptr [rip + 0x80]
            &[0xf2, 0x4c, 0x0f, 0x38, 0xf1, 0x0d, 0x80, 0x00, 0x00, 0x00],
            // movbe eax, [rsi]
            &[0x0f, 0x38, 0xf0, 0x06],
            // movbe [r8 + rax*2], r9
            &[0x4d, 0x0f, 0x38, 0xf1, 0x0c, 0x40],
            // movbe ax, [rsp + 0x10]
            &[0x66, 0x0f, 0x38, 0xf0, 0x44, 0x24, 0x10],
            // pclmulqdq xmm0, xmm1, 0x11
            &[0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x11],
            // pclmulqdq xmm0, [rsi], 0
            &[0x66, 0x0f, 0x3a, 0x44, 0x06, 0x00],
            // aesenc xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0xdc, 0xc1],
            // aesenclast xmm8, [rdi + 0x20]
            &[0x66, 0x44, 0x0f, 0x38, 0xdd, 0x47, 0x20],
            // aesdec xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0xde, 0xc1],
            // aesdeclast xmm0, xmm1
            &[0x66, 0x0f, 0x38, 0xdf, 0xc1],
            // aesimc xmm0, [rsi]
            &[0x66, 0x0f, 0x38, 0xdb, 0x06],
            // aeskeygenassist xmm0, xmm1, 0x36
            &[0x66, 0x0f, 0x3a, 0xdf, 0xc1, 0x36],
            // rdfsbase rax
            &[0xf3, 0x48, 0x0f, 0xae, 0xc0],
            // wrgsbase r9d
            &[0xf3, 0x41, 0x0f, 0xae, 0xd9],
            // fisttp word ptr [rsi]
            &[0xdf, 0x0e],
            // fisttp dword ptr [rsp + 8]
            &[0xdb, 0x4c, 0x24, 0x08],
            // fisttp qword ptr [rip + 0x10]
            &[0xdd, 0x0d, 0x10, 0x00, 0x00, 0x00],
        ];

        #[test]
        fn extension_instructions_decode_to_their_opcode_map_and_immediate() {
            // (bytes, op as map << 8 | opcode, imm, mandatory prefix flags: rep, p66)
            let cases: &[(&[u8], u16, i64, u8, bool)] = &[
                (&[0xf3, 0x48, 0x0f, 0xb8, 0xc3], 0x1b8, 0, 1, false),
                (&[0x48, 0x0f, 0xc7, 0x0e], 0x1c7, 0, 0, false),
                (&[0xf2, 0x0f, 0x12, 0xc1], 0x112, 0, 2, false),
                (&[0x66, 0x0f, 0x38, 0x00, 0xc1], 0x200, 0, 0, true),
                (&[0x66, 0x0f, 0x3a, 0x0f, 0xc1, 0x04], 0x30f, 4, 0, true),
                (&[0x66, 0x0f, 0x3a, 0x63, 0xc1, 0x1a], 0x363, 0x1a, 0, true),
                (&[0xf2, 0x48, 0x0f, 0x38, 0xf1, 0xc1], 0x2f1, 0, 2, false),
                (&[0x0f, 0x38, 0xf0, 0x06], 0x2f0, 0, 0, false),
                (&[0x66, 0x0f, 0x3a, 0x44, 0xc1, 0x11], 0x344, 0x11, 0, true),
                (&[0x66, 0x0f, 0x38, 0xdc, 0xc1], 0x2dc, 0, 0, true),
                (&[0xf3, 0x48, 0x0f, 0xae, 0xc0], 0x1ae, 0, 1, false),
                (&[0xdb, 0x0e], 0xdb, 0, 0, false),
            ];
            for &(bytes, op, imm, rep, p66) in cases {
                let i = decoded(bytes);
                assert_eq!(i.op, op, "{bytes:02x?}");
                assert_eq!(i.imm, imm, "{bytes:02x?}");
                assert_eq!((i.rep, i.p66), (rep, p66), "{bytes:02x?}");
                assert_eq!(i.len as usize, bytes.len(), "{bytes:02x?}");
                assert!(i.has_modrm, "{bytes:02x?}");
            }
        }

        #[test]
        fn every_assembled_instruction_decodes_to_exactly_its_length() {
            for bytes in ASSEMBLED {
                assert_length(bytes);
            }
        }

        #[test]
        fn trailing_bytes_are_never_consumed() {
            for bytes in ASSEMBLED {
                assert_ignores_trailing_bytes(bytes);
            }
        }

        #[test]
        fn decoding_is_deterministic() {
            for bytes in ASSEMBLED {
                let first = decoded(bytes);
                let second = decoded(bytes);
                assert_eq!(first.len, second.len);
                assert_eq!(first.op, second.op);
                assert_eq!(first.imm, second.imm);
                assert_eq!(first.disp, second.disp);
            }
        }
    }

    mod exhaustive {
        use super::*;

        /// Padding long enough for any instruction: ModRM c0 (register form) then zeros.
        fn padded(prefix: &[u8], opcode: &[u8]) -> Vec<u8> {
            let mut bytes = prefix.to_vec();
            bytes.extend_from_slice(opcode);
            bytes.push(0xc0);
            bytes.resize(MAX_INSN_LEN, 0);
            bytes
        }

        /// Opcode bytes that are prefixes or escapes rather than opcodes.
        fn is_prefix_or_escape(opcode: u8) -> bool {
            matches!(
                opcode,
                0x26 | 0x2e | 0x36 | 0x3e | 0x40
                    ..=0x4f | 0x64 | 0x65 | 0x66 | 0x67 | 0x0f | 0xf0 | 0xf2 | 0xf3
                    // VEX escapes: their own tests are in `vex`
                    | 0xc4 | 0xc5
            )
        }

        #[test]
        fn exactly_the_documented_one_byte_opcodes_are_invalid_in_long_mode() {
            let invalid: [u8; 21] = [
                0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f, 0x60, 0x61, 0x62,
                0x82, 0x9a, 0xce, 0xd4, 0xd5, 0xd6, 0xea,
            ];
            for opcode in (0..=0xffu8).filter(|&opcode| !is_prefix_or_escape(opcode)) {
                let result = decode(&padded(&[], &[opcode]));
                assert_eq!(
                    result.is_err(),
                    invalid.contains(&opcode),
                    "opcode {opcode:#04x}: {result:?}"
                );
            }
        }

        #[test]
        fn every_two_byte_opcode_decodes() {
            for opcode in 0..=0xffu8 {
                if opcode == 0x38 || opcode == 0x3a {
                    continue;
                }
                let result = decode(&padded(&[], &[0x0f, opcode]));
                assert!(result.is_ok(), "0f {opcode:#04x}: {result:?}");
            }
        }

        #[test]
        fn every_opcode_decodes_the_same_with_every_legacy_prefix_in_front() {
            for prefix in [0x66u8, 0x67, 0xf0, 0xf2, 0xf3, 0x2e, 0x64, 0x65] {
                for opcode in (0..=0xffu8).filter(|&opcode| !is_prefix_or_escape(opcode)) {
                    let plain = decode(&padded(&[], &[opcode]));
                    let prefixed = decode(&padded(&[prefix], &[opcode]));
                    match (plain, prefixed) {
                        (Err(a), Err(b)) => assert_eq!(a, b),
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a.op, b.op, "{prefix:#x} {opcode:#x}");
                            // 66 resizes immediates and 67 resizes the moffs operand; nothing else moves
                            let moffs = (0xa0..=0xa3).contains(&opcode);
                            if prefix != 0x66 && !(prefix == 0x67 && moffs) {
                                assert_eq!(a.len + 1, b.len, "{prefix:#x} {opcode:#x}");
                            }
                        }
                        (a, b) => panic!("{prefix:#x} {opcode:#x}: {a:?} vs {b:?}"),
                    }
                }
            }
        }

        /// Length of `8b /r` (mov r32, r/m32) computed from the encoding rules alone.
        fn expected_mov_length(modrm: u8, sib: u8) -> usize {
            let (mode, rm) = (modrm >> 6, modrm & 7);
            if mode == 3 {
                return 2;
            }
            let has_sib = rm == 4;
            let displacement = match mode {
                1 => 1,
                2 => 4,
                _ if rm == 5 => 4,
                _ if has_sib && sib & 7 == 5 => 4,
                _ => 0,
            };
            2 + has_sib as usize + displacement
        }

        #[test]
        fn every_modrm_and_sib_combination_has_the_architectural_length() {
            for modrm in 0..=0xffu8 {
                for sib in 0..=0xffu8 {
                    let mut bytes = vec![0x8b, modrm, sib];
                    bytes.extend_from_slice(&[0x11; 4]);
                    let i = decode(&bytes).unwrap();
                    assert_eq!(
                        i.len as usize,
                        expected_mov_length(modrm, sib),
                        "modrm {modrm:#04x} sib {sib:#04x}"
                    );
                    if modrm >> 6 != 3 && modrm & 7 == 4 {
                        assert_eq!(i.scale, sib >> 6);
                    }
                }
            }
        }

        #[test]
        fn modrm_forms_are_independent_of_rex_bits_for_length() {
            for rex in 0x40..=0x4fu8 {
                for modrm in 0..=0xffu8 {
                    let mut bytes = vec![rex, 0x8b, modrm, 0x24];
                    bytes.extend_from_slice(&[0x22; 4]);
                    let i = decode(&bytes).unwrap();
                    assert_eq!(
                        i.len as usize,
                        1 + expected_mov_length(modrm, 0x24),
                        "rex {rex:#04x} modrm {modrm:#04x}"
                    );
                }
            }
        }

        #[test]
        fn every_truncation_of_every_modrm_form_is_truncated_not_invalid() {
            for modrm in 0..=0xffu8 {
                let full = [0x8b, modrm, 0x25, 0x10, 0x20, 0x30, 0x40];
                let length = expected_mov_length(modrm, 0x25);
                for cut in 0..length {
                    assert_eq!(
                        decode(&full[..cut]).err(),
                        Some(DecodeErr::Truncated),
                        "modrm {modrm:#04x} cut at {cut}"
                    );
                }
            }
        }
    }
}
