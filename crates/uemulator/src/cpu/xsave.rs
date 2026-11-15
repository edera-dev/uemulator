//! XSAVE feature set: xgetbv/xsetbv, xsave/xrstor/xsaveopt (standard, non-compacted format), and
//! the rdrand/rdseed generator. x87 and SSE are the only state components; the legacy 512 bytes
//! are the fxsave image (`fpu.rs`), followed by the 64-byte header.

use super::exception::Exception;
use super::flags::CF;
use super::fpu::{FXSAVE_AREA_SIZE, LegacyRestore};
use super::sys::{CR0_TS, CR4_OSXSAVE, RAX, RCX, RDX};
use super::{Cpu, ExecResult, FCW_RESET, MXCSR_RESET, mem};

pub(crate) const XCR0_X87: u64 = 1 << 0;
pub(crate) const XCR0_SSE: u64 = 1 << 1;
/// YMM_Hi128: the upper halves of YMM0-15 (`Cpu::ymm_hi`).
pub(crate) const XCR0_AVX: u64 = 1 << 2;
/// Opmask: k0-k7 (`Cpu::opmask`).
pub(super) const XCR0_OPMASK: u64 = 1 << 5;
/// ZMM_Hi256: bits 511:256 of ZMM0-15 (`Cpu::zmm_hi`).
pub(super) const XCR0_ZMM_HI256: u64 = 1 << 6;
/// Hi16_ZMM: ZMM16-31 (`Cpu::zmm_extra`).
pub(super) const XCR0_HI16_ZMM: u64 = 1 << 7;
/// The three AVX-512 components; xsetbv accepts all of them or none.
pub(crate) const XCR0_AVX512: u64 = XCR0_OPMASK | XCR0_ZMM_HI256 | XCR0_HI16_ZMM;
/// Components xsetbv accepts. Extending this also means extending CPUID leaf 0xD and the save
/// area layout below, or the kernel will size its buffers for state the instructions skip.
pub(crate) const XCR0_SUPPORTED: u64 = XCR0_X87 | XCR0_SSE | XCR0_AVX | XCR0_AVX512;

/// The XSAVE area is the legacy image plus a 64-byte header: XSTATE_BV (8 bytes), XCOMP_BV (8)
/// and 48 reserved bytes, all of which xrstor requires to be zero except XSTATE_BV.
pub(super) const XSAVE_HEADER_SIZE: u64 = 64;
pub(super) const XSAVE_AREA_SIZE: u64 = FXSAVE_AREA_SIZE + XSAVE_HEADER_SIZE;

/// One extended-state component in the standard (non-compacted) format. `offset` is what
/// CPUID.0xD.n:EBX reports; the standard format places component n after the previous one, but
/// the opmask component starts at 1088 because Intel aligns components to the sizes the
/// hardware gave them, and software hard-codes these numbers.
pub(super) struct Component {
    pub bit: u64,
    /// The component number (CPUID.0xD subleaf).
    pub index: u32,
    pub offset: u64,
    pub size: u64,
}

/// The AVX component sits right after the header and holds YMM0_H..YMM15_H, 16 bytes each.
pub(super) const AVX_STATE_OFFSET: u64 = XSAVE_AREA_SIZE;
pub(super) const AVX_STATE_SIZE: u64 = 256;
const OPMASK_STATE_OFFSET: u64 = 1088;
const OPMASK_STATE_SIZE: u64 = 64;
const ZMM_HI256_STATE_OFFSET: u64 = 1152;
const ZMM_HI256_STATE_SIZE: u64 = 512;
const HI16_ZMM_STATE_OFFSET: u64 = 1664;
const HI16_ZMM_STATE_SIZE: u64 = 1024;
pub(super) const COMPONENTS: [Component; 4] = [
    Component {
        bit: XCR0_AVX,
        index: 2,
        offset: AVX_STATE_OFFSET,
        size: AVX_STATE_SIZE,
    },
    Component {
        bit: XCR0_OPMASK,
        index: 5,
        offset: OPMASK_STATE_OFFSET,
        size: OPMASK_STATE_SIZE,
    },
    Component {
        bit: XCR0_ZMM_HI256,
        index: 6,
        offset: ZMM_HI256_STATE_OFFSET,
        size: ZMM_HI256_STATE_SIZE,
    },
    Component {
        bit: XCR0_HI16_ZMM,
        index: 7,
        offset: HI16_ZMM_STATE_OFFSET,
        size: HI16_ZMM_STATE_SIZE,
    },
];
const YMM_HIGH_STRIDE: u64 = 16;
const ZMM_HI256_STRIDE: u64 = 32;
const HI16_ZMM_STRIDE: u64 = 64;
const OPMASK_STRIDE: u64 = 8;
const XSAVE_ALIGNMENT: u64 = 64;
const XSTATE_BV_OFFSET: u64 = FXSAVE_AREA_SIZE;
const XCOMP_BV_OFFSET: u64 = XSTATE_BV_OFFSET + 8;
const XSAVE_HEADER_RESERVED_OFFSET: u64 = XCOMP_BV_OFFSET + 8;
const XSAVE_HEADER_RESERVED_QWORDS: u64 = 6;
/// XCOMP_BV bit 63: the area is in the compacted format (xsavec/xsaves wrote it). The remaining
/// bits then name the components present, which are stored back to back with no gaps.
const XCOMP_BV_COMPACTED: u64 = 1 << 63;
/// Supervisor state components IA32_XSS could enable (CET, PT, ...). None are implemented, so the
/// MSR reads zero and only zero can be written.
pub(super) const XSS_SUPPORTED: u64 = 0;
/// XGETBV's ECX selector for XINUSE (CPUID.0xD.1:EAX bit 2).
const XGETBV_XINUSE: u32 = 1;

/// splitmix64 constants (Steele, Lea, Flood; also what Java's SplittableRandom uses).
const SPLITMIX_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLY_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLY_2: u64 = 0x94d0_49bb_1331_11eb;

/// Where a component sits in the XSAVE area. The standard format puts each at its fixed offset;
/// the compacted one packs the components named in XCOMP_BV in component order, so an offset
/// depends on which lower components are present. None of ours asks for 64-byte alignment
/// (CPUID.0xD.n:ECX bit 1 is clear), so the packing is just a running sum.
#[derive(Clone, Copy)]
enum Layout {
    Standard,
    Compacted { components: u64 },
}

impl Layout {
    fn offset(self, part: &Component) -> u64 {
        match self {
            Layout::Standard => part.offset,
            Layout::Compacted { components } => {
                XSAVE_AREA_SIZE
                    + COMPONENTS
                        .iter()
                        .filter(|lower| lower.index < part.index && components & lower.bit != 0)
                        .map(|lower| lower.size)
                        .sum::<u64>()
            }
        }
    }
}

/// One step of SplitMix64: the deterministic generator behind rdrand and, under `--verify`,
/// virtio-rng.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MULTIPLY_1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MULTIPLY_2);
    z ^ (z >> 31)
}

impl Cpu {
    /// Next value of the rdrand/rdseed generator. Deterministic given `MachineConfig::random_seed`.
    pub(super) fn next_random(&mut self) -> u64 {
        splitmix64(&mut self.rng_state)
    }

    /// 0F C7 /6 (rdrand) and /7 (rdseed), register forms only. Always succeeds: CF=1 and the other
    /// arithmetic flags cleared, as the SDM describes for a valid result.
    pub(super) fn rdrand(&mut self, register: u8, size: u8) {
        let value = self.next_random();
        self.write_gpr(register, size, true, value);
        self.set_eflags_arith(CF);
    }

    /// 0F 01 D0. #UD unless the OS enabled XSAVE (CR4.OSXSAVE); any privilege level may read.
    /// ECX=1 is XINUSE ANDed with XCR0: the components not in their initial configuration.
    pub(super) fn xgetbv(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        let value = match self.gpr[RCX] as u32 {
            0 => self.xcr0,
            XGETBV_XINUSE => self.xcr0 & self.xinuse(),
            _ => return Err(Exception::general_protection(0)),
        };
        self.gpr[RAX] = value & 0xffff_ffff;
        self.gpr[RDX] = value >> 32;
        Ok(())
    }

    /// The state components that differ from their initial configuration (SDM 13.6): software
    /// uses it to skip saving state that is still at its reset value.
    fn xinuse(&self) -> u64 {
        let mut in_use = 0;
        let x87_initial = self.fcw == FCW_RESET && self.fsw == 0 && self.ftw == 0;
        if !x87_initial {
            in_use |= XCR0_X87;
        }
        if self.mxcsr != MXCSR_RESET || self.xmm.iter().any(|&register| register != 0) {
            in_use |= XCR0_SSE;
        }
        if self.ymm_hi.iter().any(|&half| half != 0) {
            in_use |= XCR0_AVX;
        }
        if self.opmask.iter().any(|&mask| mask != 0) {
            in_use |= XCR0_OPMASK;
        }
        if self.zmm_hi.iter().flatten().any(|&half| half != 0) {
            in_use |= XCR0_ZMM_HI256;
        }
        if self.zmm_extra.iter().flatten().any(|&lane| lane != 0) {
            in_use |= XCR0_HI16_ZMM;
        }
        in_use
    }

    /// 0F 01 D1. CPL 0 only; x87 must stay enabled, unknown components and the AVX-without-SSE
    /// combination are #GP.
    pub(super) fn xsetbv(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        if self.gpr[RCX] as u32 != 0 {
            return Err(Exception::general_protection(0));
        }
        let value = (self.gpr[RDX] << 32) | (self.gpr[RAX] & 0xffff_ffff);
        if value & XCR0_X87 == 0
            || value & !XCR0_SUPPORTED != 0
            || (value & XCR0_AVX != 0 && value & XCR0_SSE == 0)
            // The AVX-512 components come together and need AVX state to hang their upper halves on.
            || (value & XCR0_AVX512 != 0
                && (value & XCR0_AVX512 != XCR0_AVX512 || value & XCR0_AVX == 0))
        {
            return Err(Exception::general_protection(0));
        }
        self.xcr0 = value;
        Ok(())
    }

    /// The checks xsave/xsaveopt/xrstor share, in SDM order: #UD without OSXSAVE, #NM under
    /// CR0.TS, #GP(0) for a misaligned area. Returns the requested-feature bitmap, EDX:EAX
    /// masked by XCR0.
    fn xsave_prologue(&self, address: u64) -> ExecResult<u64> {
        if self.cr4 & CR4_OSXSAVE == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.cr0 & CR0_TS != 0 {
            return Err(Exception::device_not_available());
        }
        if address & (XSAVE_ALIGNMENT - 1) != 0 {
            return Err(Exception::general_protection(0));
        }
        let mask = (self.gpr[RDX] << 32) | (self.gpr[RAX] & 0xffff_ffff);
        Ok(mask & self.xcr0)
    }

    /// xsave and xsaveopt (REX.W forms are identical here: the x87 pointers are not stored).
    /// Components in the bitmap are always written and always marked in use in XSTATE_BV; xsaveopt's
    /// "skip components in their initial state" is an optimization software cannot observe except
    /// through XSTATE_BV, which saying "in use" keeps correct.
    pub(super) fn xsave(&mut self, address: u64) -> ExecResult<()> {
        let requested = self.xsave_prologue(address)?;
        let size = enabled_area_size(self.xcr0) as u64;
        self.translate(address + size - 1, mem::AccessKind::Write, false)?;
        self.save_legacy_region(
            address,
            requested & XCR0_X87 != 0,
            requested & XCR0_SSE != 0,
        )?;
        self.save_vector_components(address, requested, Layout::Standard)?;
        let previous = self.read(address + XSTATE_BV_OFFSET, 8)?;
        self.write(
            address + XSTATE_BV_OFFSET,
            8,
            (previous & !requested) | requested,
        )
    }

    /// xrstor: the standard format, or the compacted one when XCOMP_BV bit 63 is set. The whole
    /// header is validated and every byte of the components read before any register changes; a
    /// component the bitmap names but XSTATE_BV does not is reset to its initial state instead of
    /// loaded.
    pub(super) fn xrstor(&mut self, address: u64) -> ExecResult<()> {
        self.xrstor_area(address, false)
    }

    /// xrstors: CPL 0 only, and it insists on the compacted format.
    pub(super) fn xrstors(&mut self, address: u64) -> ExecResult<()> {
        self.xrstor_area(address, true)
    }

    fn xrstor_area(&mut self, address: u64, supervisor: bool) -> ExecResult<()> {
        let requested = self.xsave_prologue(address)?;
        if supervisor {
            self.need_cpl0()?;
        }
        let state_bv = self.read(address + XSTATE_BV_OFFSET, 8)?;
        let compaction = self.read(address + XCOMP_BV_OFFSET, 8)?;
        let compacted = compaction & XCOMP_BV_COMPACTED != 0;
        let components = compaction & !XCOMP_BV_COMPACTED;
        let valid = if compacted {
            // the components present must be ones the OS enabled, and only they can be in use
            components & !(self.xcr0 | XSS_SUPPORTED) == 0 && state_bv & !components == 0
        } else {
            !supervisor && compaction == 0 && state_bv & !self.xcr0 == 0
        };
        if !valid {
            return Err(Exception::general_protection(0));
        }
        for k in 0..XSAVE_HEADER_RESERVED_QWORDS {
            if self.read(address + XSAVE_HEADER_RESERVED_OFFSET + 8 * k, 8)? != 0 {
                return Err(Exception::general_protection(0));
            }
        }
        let layout = if compacted {
            Layout::Compacted { components }
        } else {
            Layout::Standard
        };
        let component = |bit: u64| {
            if requested & bit == 0 {
                LegacyRestore::Skip
            } else if state_bv & bit != 0 {
                LegacyRestore::Load
            } else {
                LegacyRestore::Init
            }
        };
        // Read every vector component before anything is committed, like the legacy region.
        let mut pending = VectorState {
            ymm_hi: self.ymm_hi,
            opmask: self.opmask,
            zmm_hi: self.zmm_hi,
            zmm_extra: self.zmm_extra,
        };
        for part in &COMPONENTS {
            match component(part.bit) {
                // `component` never asks for MxcsrOnly, which only fast fxrstor uses
                LegacyRestore::Skip | LegacyRestore::MxcsrOnly => {}
                LegacyRestore::Init => pending.initialize(part.bit),
                LegacyRestore::Load => self.load_component(address, part, layout, &mut pending)?,
            }
        }
        self.restore_legacy_region(address, component(XCR0_X87), component(XCR0_SSE))?;
        self.ymm_hi = pending.ymm_hi;
        self.opmask = pending.opmask;
        self.zmm_hi = pending.zmm_hi;
        self.zmm_extra = pending.zmm_extra;
        Ok(())
    }

    /// xsavec: like xsave but the components are packed (see `Layout`), XCOMP_BV records which
    /// ones are present, and the header's other bytes are cleared. Every requested component is
    /// stored and marked in use; real hardware leaves out those in their initial state, which
    /// software cannot tell from here.
    pub(super) fn xsavec(&mut self, address: u64) -> ExecResult<()> {
        self.xsave_compacted(address, false)
    }

    /// xsaves: xsavec at CPL 0, which could also save supervisor state if any were enabled.
    pub(super) fn xsaves(&mut self, address: u64) -> ExecResult<()> {
        self.xsave_compacted(address, true)
    }

    fn xsave_compacted(&mut self, address: u64, supervisor: bool) -> ExecResult<()> {
        let requested = self.xsave_prologue(address)?;
        if supervisor {
            self.need_cpl0()?;
        }
        self.translate(
            address + compacted_area_size(requested) as u64 - 1,
            mem::AccessKind::Write,
            false,
        )?;
        self.save_legacy_region(
            address,
            requested & XCR0_X87 != 0,
            requested & XCR0_SSE != 0,
        )?;
        let layout = Layout::Compacted {
            components: requested,
        };
        self.save_vector_components(address, requested, layout)?;
        self.write(address + XSTATE_BV_OFFSET, 8, requested)?;
        self.write(address + XCOMP_BV_OFFSET, 8, XCOMP_BV_COMPACTED | requested)?;
        for k in 0..XSAVE_HEADER_RESERVED_QWORDS {
            self.write(address + XSAVE_HEADER_RESERVED_OFFSET + 8 * k, 8, 0)?;
        }
        Ok(())
    }

    fn save_vector_components(
        &mut self,
        address: u64,
        requested: u64,
        layout: Layout,
    ) -> ExecResult<()> {
        if requested & XCR0_AVX != 0 {
            for k in 0..self.ymm_hi.len() {
                let at = address + layout.offset(&COMPONENTS[0]) + k as u64 * YMM_HIGH_STRIDE;
                self.write128(at, self.ymm_hi[k])?;
            }
        }
        if requested & XCR0_OPMASK != 0 {
            for k in 0..self.opmask.len() {
                let at = address + layout.offset(&COMPONENTS[1]) + k as u64 * OPMASK_STRIDE;
                self.write(at, 8, self.opmask[k])?;
            }
        }
        if requested & XCR0_ZMM_HI256 != 0 {
            for k in 0..self.zmm_hi.len() {
                let at = address + layout.offset(&COMPONENTS[2]) + k as u64 * ZMM_HI256_STRIDE;
                self.write128(at, self.zmm_hi[k][0])?;
                self.write128(at + 16, self.zmm_hi[k][1])?;
            }
        }
        if requested & XCR0_HI16_ZMM != 0 {
            for k in 0..self.zmm_extra.len() {
                let at = address + layout.offset(&COMPONENTS[3]) + k as u64 * HI16_ZMM_STRIDE;
                for (lane, value) in self.zmm_extra[k].into_iter().enumerate() {
                    self.write128(at + 16 * lane as u64, value)?;
                }
            }
        }
        Ok(())
    }

    fn load_component(
        &mut self,
        address: u64,
        part: &Component,
        layout: Layout,
        pending: &mut VectorState,
    ) -> ExecResult<()> {
        let base = address + layout.offset(part);
        match part.bit {
            XCR0_AVX => {
                for k in 0..pending.ymm_hi.len() {
                    pending.ymm_hi[k] = self.read128(base + k as u64 * YMM_HIGH_STRIDE)?;
                }
            }
            XCR0_OPMASK => {
                for k in 0..pending.opmask.len() {
                    pending.opmask[k] = self.read(base + k as u64 * OPMASK_STRIDE, 8)?;
                }
            }
            XCR0_ZMM_HI256 => {
                for k in 0..pending.zmm_hi.len() {
                    let at = base + k as u64 * ZMM_HI256_STRIDE;
                    pending.zmm_hi[k] = [self.read128(at)?, self.read128(at + 16)?];
                }
            }
            _ => {
                for k in 0..pending.zmm_extra.len() {
                    let at = base + k as u64 * HI16_ZMM_STRIDE;
                    for lane in 0..4 {
                        pending.zmm_extra[k][lane] = self.read128(at + 16 * lane as u64)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The vector registers xrstor can replace, staged so a fault while reading leaves the real
/// ones alone.
struct VectorState {
    ymm_hi: [u128; 16],
    opmask: [u64; 8],
    zmm_hi: [[u128; 2]; 16],
    zmm_extra: [[u128; 4]; 16],
}

impl VectorState {
    fn initialize(&mut self, bit: u64) {
        match bit {
            XCR0_AVX => self.ymm_hi = [0; 16],
            XCR0_OPMASK => self.opmask = [0; 8],
            XCR0_ZMM_HI256 => self.zmm_hi = [[0; 2]; 16],
            _ => self.zmm_extra = [[0; 4]; 16],
        }
    }
}

/// CPUID.0xD.0:EBX: bytes the XSAVE area needs for the components `xcr0` enables.
pub(super) fn enabled_area_size(xcr0: u64) -> u32 {
    // The header is always part of the area; the highest enabled component ends it.
    COMPONENTS
        .iter()
        .filter(|part| xcr0 & part.bit != 0)
        .map(|part| part.offset + part.size)
        .fold(XSAVE_AREA_SIZE, u64::max) as u32
}

/// CPUID.0xD.1:EBX: bytes the compacted format needs for the components `xcr0` enables, which is
/// the header plus exactly those components with no gaps.
pub(super) fn compacted_area_size(xcr0: u64) -> u32 {
    (XSAVE_AREA_SIZE
        + COMPONENTS
            .iter()
            .filter(|part| xcr0 & part.bit != 0)
            .map(|part| part.size)
            .sum::<u64>()) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{CODE_BASE, Harness, RBX, RDI, SCRATCH, Stop};

    const AREA: u64 = SCRATCH + 0x1000;
    const HLT: u8 = 0xf4;
    const XGETBV: &[u8] = &[0x0f, 0x01, 0xd0, HLT];
    const XSETBV: &[u8] = &[0x0f, 0x01, 0xd1, HLT];
    // xsave/xrstor/xsaveopt [rdi]
    const XSAVE_RDI: &[u8] = &[0x0f, 0xae, 0x27, HLT];
    const XRSTOR_RDI: &[u8] = &[0x0f, 0xae, 0x2f, HLT];
    const XSAVEOPT_RDI: &[u8] = &[0x0f, 0xae, 0x37, HLT];
    const RDRAND_RAX: &[u8] = &[0x48, 0x0f, 0xc7, 0xf0, HLT];
    const RDRAND_EAX: &[u8] = &[0x0f, 0xc7, 0xf0, HLT];
    const RDRAND_AX: &[u8] = &[0x66, 0x0f, 0xc7, 0xf0, HLT];
    const RDSEED_RAX: &[u8] = &[0x48, 0x0f, 0xc7, 0xf8, HLT];
    const CR4_FXSR_XSAVE: u64 = super::super::sys::CR4_OSFXSR | CR4_OSXSAVE;

    fn run(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
        let mut h = Harness::new(code);
        h.cpu.cr4 |= CR4_FXSR_XSAVE;
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        h.cpu.gpr[RDI] = AREA;
        setup(&mut h);
        let stop = h.run(100);
        (h, stop)
    }

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    fn pattern(k: usize) -> u128 {
        0x0101_0101_0101_0101_0101_0101_0101_0101u128.wrapping_mul(k as u128 + 0x11)
    }

    #[test]
    fn xgetbv_reads_xcr0_into_edx_eax() {
        let (h, stop) = run(XGETBV, |h| {
            h.cpu.gpr[RCX] = 0;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RDX]), (3, 0));
    }

    #[test]
    fn xgetbv_is_undefined_without_osxsave_and_faults_on_other_registers() {
        let (_, stop) = run(XGETBV, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
        assert_eq!(stop, fault(6, None));
        let (_, stop) = run(XGETBV, |h| h.cpu.gpr[RCX] = 2);
        assert_eq!(stop, fault(13, Some(0)));
    }

    #[test]
    fn xsetbv_writes_xcr0_and_validates_it() {
        let write = |value: u64| {
            run(XSETBV, move |h| {
                h.cpu.gpr[RCX] = 0;
                h.cpu.gpr[RAX] = value & 0xffff_ffff;
                h.cpu.gpr[RDX] = value >> 32;
            })
        };
        let (h, stop) = write(XCR0_X87);
        assert_eq!((stop, h.cpu.xcr0), (Stop::Halted, XCR0_X87));
        let (h, stop) = write(XCR0_X87 | XCR0_SSE | XCR0_AVX);
        assert_eq!((stop, h.cpu.xcr0), (Stop::Halted, 7));
        // x87 cannot be disabled, unsupported components and high garbage are #GP
        for bad in [XCR0_SSE, 0, 0b101, 0b1000 | 1, 1 << 63 | 1, 1 << 32 | 1] {
            let (h, stop) = write(bad);
            assert_eq!(stop, fault(13, Some(0)), "value {bad:#x}");
            assert_eq!(h.cpu.xcr0, 3);
        }
        let (_, stop) = run(XSETBV, |h| {
            h.cpu.gpr[RCX] = 1;
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, fault(13, Some(0)));
        let (_, stop) = run(XSETBV, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
        assert_eq!(stop, fault(6, None));
    }

    #[test]
    fn xsave_writes_the_fxsave_image_and_marks_components_in_use() {
        let (h, stop) = run(XSAVE_RDI, |h| {
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
            h.cpu.fcw = 0x027f;
            for k in 0..16 {
                h.cpu.xmm[k] = pattern(k);
            }
        });
        assert_eq!(stop, Stop::Halted);
        let image = h.read(AREA, XSAVE_AREA_SIZE as usize);
        assert_eq!(u16::from_le_bytes([image[0], image[1]]), 0x027f);
        for k in 0..16 {
            let at = 160 + 16 * k;
            assert_eq!(
                u128::from_le_bytes(image[at..at + 16].try_into().unwrap()),
                pattern(k)
            );
        }
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 3);
        assert_eq!(h.read_u64(AREA + XCOMP_BV_OFFSET), 0);
    }

    #[test]
    fn xsave_with_a_partial_mask_leaves_the_other_component_alone() {
        let (h, _) = run(XSAVE_RDI, |h| {
            h.write(AREA, &[0xaa; XSAVE_AREA_SIZE as usize]);
            h.write_u64(AREA + XSTATE_BV_OFFSET, 0b10);
            h.cpu.gpr[RAX] = XCR0_X87;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[0] = 7;
        });
        // x87 saved (FCW is the reset value), SSE bytes untouched, XSTATE_BV keeps bit 1
        assert_eq!(h.read(AREA, 2), 0x037fu16.to_le_bytes());
        assert_eq!(h.read(AREA + 160, 16), [0xaa; 16]);
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 0b11);
    }

    #[test]
    fn xsaveopt_matches_xsave() {
        let setup = |h: &mut Harness| {
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[3] = pattern(3);
        };
        let (a, _) = run(XSAVE_RDI, setup);
        let (b, _) = run(XSAVEOPT_RDI, setup);
        assert_eq!(
            a.read(AREA, XSAVE_AREA_SIZE as usize),
            b.read(AREA, XSAVE_AREA_SIZE as usize)
        );
    }

    #[test]
    fn xrstor_round_trips_x87_and_sse_state() {
        let (saved, _) = run(XSAVE_RDI, |h| {
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.cpu.fcw = 0x0c7f;
            h.cpu.mxcsr = 0x1fa0;
            for k in 0..16 {
                h.cpu.xmm[k] = pattern(k);
            }
        });
        let image = saved.read(AREA, XSAVE_AREA_SIZE as usize);
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &image);
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.fcw, h.cpu.mxcsr), (0x0c7f, 0x1fa0));
        for k in 0..16 {
            assert_eq!(h.cpu.xmm[k], pattern(k));
        }
    }

    #[test]
    fn xrstor_initializes_components_whose_state_bit_is_clear() {
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
            h.write(AREA + 24, &0x1f80u32.to_le_bytes());
            h.cpu.fcw = 0x0c7f;
            h.cpu.xmm[5] = 9;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!((h.cpu.fcw, h.cpu.mxcsr, h.cpu.xmm[5]), (0x037f, 0x1f80, 0));
    }

    #[test]
    fn xrstor_skips_components_outside_the_mask() {
        let (h, _) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.cpu.gpr[RAX] = XCR0_X87;
            h.cpu.gpr[RDX] = 0;
            h.cpu.xmm[5] = 9;
            h.cpu.fcw = 0x0c7f;
        });
        assert_eq!((h.cpu.fcw, h.cpu.xmm[5]), (0x037f, 9));
    }

    #[test]
    fn xrstor_rejects_bad_headers_without_touching_state() {
        for (offset, value) in [
            (XSTATE_BV_OFFSET, 4u64),
            // XCOMP_BV bits without the compaction bit, or naming a component XCR0 lacks
            (XCOMP_BV_OFFSET, 1),
            (XCOMP_BV_OFFSET, XCOMP_BV_COMPACTED | XCR0_AVX),
            (XSAVE_HEADER_RESERVED_OFFSET, 1),
            (XSAVE_HEADER_RESERVED_OFFSET + 40, 1),
        ] {
            let (h, stop) = run(XRSTOR_RDI, |h| {
                h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
                h.write_u64(AREA + offset, value);
                h.cpu.gpr[RAX] = 3;
                h.cpu.gpr[RDX] = 0;
                h.cpu.xmm[1] = 5;
            });
            assert_eq!(stop, fault(13, Some(0)), "offset {offset:#x}");
            assert_eq!(h.cpu.xmm[1], 5);
        }
    }

    #[test]
    fn xrstor_rejects_a_reserved_mxcsr_bit() {
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.write(AREA + 24, &0x0001_0000u32.to_le_bytes());
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, fault(13, Some(0)));
        assert_eq!(h.cpu.mxcsr, 0x1f80);
    }

    #[test]
    fn xsave_family_faults_follow_the_sdm_order() {
        for code in [XSAVE_RDI, XRSTOR_RDI, XSAVEOPT_RDI] {
            let (_, stop) = run(code, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
            assert_eq!(stop, fault(6, None), "no OSXSAVE");
            let (_, stop) = run(code, |h| h.cpu.cr0 |= CR0_TS);
            assert_eq!(stop, fault(7, None), "CR0.TS");
            let (_, stop) = run(code, |h| h.cpu.gpr[RDI] = AREA + 16);
            assert_eq!(stop, fault(13, Some(0)), "16-byte aligned is not enough");
        }
    }

    #[test]
    fn xsave_mask_is_limited_to_xcr0() {
        let (h, _) = run(XSAVE_RDI, |h| {
            h.cpu.xcr0 = XCR0_X87;
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
        });
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), 1);
    }

    #[test]
    fn rdrand_sets_cf_and_clears_the_other_arithmetic_flags() {
        let (h, stop) = run(RDRAND_RAX, |h| {
            h.cpu.set_eflags_arith(super::super::flags::ARITH & !CF)
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.eflags_full() & super::super::flags::ARITH, CF);
        assert_ne!(h.cpu.gpr[RAX], 0);
    }

    #[test]
    fn rdrand_and_rdseed_follow_the_seed_and_the_operand_size() {
        let (a, _) = run(RDRAND_RAX, |h| h.cpu.rng_state = 1);
        let (b, _) = run(RDSEED_RAX, |h| h.cpu.rng_state = 1);
        let (c, _) = run(RDRAND_RAX, |h| h.cpu.rng_state = 2);
        assert_eq!(a.cpu.gpr[RAX], b.cpu.gpr[RAX]);
        assert_ne!(a.cpu.gpr[RAX], c.cpu.gpr[RAX]);
        // 32-bit writes zero the upper half, 16-bit writes keep it
        let (d, _) = run(RDRAND_EAX, |h| {
            h.cpu.rng_state = 1;
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(d.cpu.gpr[RAX], a.cpu.gpr[RAX] & 0xffff_ffff);
        let (e, _) = run(RDRAND_AX, |h| {
            h.cpu.rng_state = 1;
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(e.cpu.gpr[RAX], (u64::MAX << 16) | (a.cpu.gpr[RAX] & 0xffff));
    }

    #[test]
    fn the_generator_does_not_repeat_over_a_short_run() {
        let mut h = Harness::new(&[HLT]);
        let values: Vec<u64> = (0..1000).map(|_| h.cpu.next_random()).collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), values.len());
        // splitmix64 reference output for seed 0 (first value)
        let mut reference = Harness::new(&[HLT]);
        reference.cpu.rng_state = 0;
        assert_eq!(reference.cpu.next_random(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn rdrand_with_a_register_form_prefix_or_memory_operand_is_undefined() {
        // rdrand [rdi]: memory form of /6
        let (_, stop) = run(&[0x0f, 0xc7, 0x37, HLT], |_| {});
        assert_eq!(stop, fault(6, None));
        let _ = CODE_BASE;
    }

    const XSAVEC_RDI: &[u8] = &[0x0f, 0xc7, 0x27, HLT];
    const XSAVES_RDI: &[u8] = &[0x0f, 0xc7, 0x2f, HLT];
    const XRSTORS_RDI: &[u8] = &[0x0f, 0xc7, 0x1f, HLT];
    const RDPID_RAX: &[u8] = &[0xf3, 0x0f, 0xc7, 0xf8, HLT];
    const ALL_COMPONENTS: u64 = XCR0_SUPPORTED;
    const COMPACT_AVX: u64 = XSAVE_AREA_SIZE;
    const COMPACT_OPMASK: u64 = COMPACT_AVX + AVX_STATE_SIZE;
    const COMPACT_ZMM_HI256: u64 = COMPACT_OPMASK + OPMASK_STATE_SIZE;
    const COMPACT_HI16_ZMM: u64 = COMPACT_ZMM_HI256 + ZMM_HI256_STATE_SIZE;

    /// A harness with every component enabled and each of them holding a recognisable pattern.
    fn fill_vector_state(h: &mut Harness) {
        h.cpu.xcr0 = ALL_COMPONENTS;
        h.cpu.gpr[RAX] = u64::MAX;
        h.cpu.gpr[RDX] = u64::MAX;
        h.cpu.fcw = 0x027f;
        h.cpu.mxcsr = 0x1fa0;
        for k in 0..16 {
            h.cpu.xmm[k] = pattern(k);
            h.cpu.ymm_hi[k] = pattern(k + 100);
            h.cpu.zmm_hi[k] = [pattern(k + 200), pattern(k + 300)];
            h.cpu.zmm_extra[k] = [
                pattern(k + 400),
                pattern(k + 500),
                pattern(k + 600),
                pattern(k + 700),
            ];
        }
        for k in 0..8 {
            h.cpu.opmask[k] = 0x1111_0000_0000_0000u64 * (k as u64 + 1) + k as u64;
        }
    }

    #[test]
    fn xsavec_packs_the_components_behind_the_header() {
        let (h, stop) = run(XSAVEC_RDI, fill_vector_state);
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.read_u64(AREA + XSTATE_BV_OFFSET), ALL_COMPONENTS);
        assert_eq!(
            h.read_u64(AREA + XCOMP_BV_OFFSET),
            XCOMP_BV_COMPACTED | ALL_COMPONENTS
        );
        for k in 0..XSAVE_HEADER_RESERVED_QWORDS {
            assert_eq!(h.read_u64(AREA + XSAVE_HEADER_RESERVED_OFFSET + 8 * k), 0);
        }
        let u128_at = |at: u64| u128::from_le_bytes(h.read(AREA + at, 16).try_into().unwrap());
        // the legacy region is where fxsave puts it
        assert_eq!(u128_at(160 + 16 * 3), pattern(3));
        assert_eq!(h.read_u64(AREA + 24) as u32, 0x1fa0);
        assert_eq!(u128_at(COMPACT_AVX + 16 * 5), pattern(105));
        assert_eq!(
            h.read_u64(AREA + COMPACT_OPMASK + 8 * 2),
            0x3333_0000_0000_0000 + 2
        );
        assert_eq!(u128_at(COMPACT_ZMM_HI256 + 32 * 7), pattern(207));
        assert_eq!(u128_at(COMPACT_ZMM_HI256 + 32 * 7 + 16), pattern(307));
        assert_eq!(u128_at(COMPACT_HI16_ZMM + 64 * 15 + 48), pattern(715));
        // and the standard format would have used more room
        assert_eq!(
            compacted_area_size(ALL_COMPONENTS) as u64,
            COMPACT_HI16_ZMM + HI16_ZMM_STATE_SIZE
        );
        assert!(compacted_area_size(ALL_COMPONENTS) < enabled_area_size(ALL_COMPONENTS));
    }

    #[test]
    fn xsavec_offsets_depend_on_which_components_are_requested() {
        // SSE and Hi16_ZMM only: the Hi16 state sits right behind the header
        let (h, stop) = run(XSAVEC_RDI, |h| {
            fill_vector_state(h);
            h.cpu.gpr[RAX] = XCR0_SSE | XCR0_HI16_ZMM;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(
            h.read_u64(AREA + XCOMP_BV_OFFSET),
            XCOMP_BV_COMPACTED | XCR0_SSE | XCR0_HI16_ZMM
        );
        let first = u128::from_le_bytes(h.read(AREA + XSAVE_AREA_SIZE, 16).try_into().unwrap());
        assert_eq!(first, pattern(400));
        assert_eq!(
            compacted_area_size(XCR0_SSE | XCR0_HI16_ZMM) as u64,
            XSAVE_AREA_SIZE + HI16_ZMM_STATE_SIZE
        );
    }

    #[test]
    fn xrstor_loads_what_xsavec_stored() {
        let (saved, _) = run(XSAVEC_RDI, fill_vector_state);
        let image = saved.read(AREA, enabled_area_size(ALL_COMPONENTS) as usize);
        for code in [XRSTOR_RDI, XRSTORS_RDI] {
            let (h, stop) = run(code, |h| {
                h.cpu.xcr0 = ALL_COMPONENTS;
                h.write(AREA, &image);
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
            });
            assert_eq!(stop, Stop::Halted, "{code:02x?}");
            assert_eq!((h.cpu.fcw, h.cpu.mxcsr), (0x027f, 0x1fa0));
            for k in 0..16 {
                assert_eq!(h.cpu.xmm[k], pattern(k));
                assert_eq!(h.cpu.ymm_hi[k], pattern(k + 100));
                assert_eq!(h.cpu.zmm_hi[k], [pattern(k + 200), pattern(k + 300)]);
                assert_eq!(h.cpu.zmm_extra[k][3], pattern(k + 700));
            }
            assert_eq!(h.cpu.opmask[5], 0x6666_0000_0000_0000 + 5);
        }
    }

    #[test]
    fn xrstor_of_a_compacted_area_resets_components_whose_state_bit_is_clear() {
        let (saved, _) = run(XSAVEC_RDI, fill_vector_state);
        let mut image = saved.read(AREA, enabled_area_size(ALL_COMPONENTS) as usize);
        // clear XSTATE_BV bit 2 (AVX): the component is still present in XCOMP_BV
        let bv = (ALL_COMPONENTS & !XCR0_AVX).to_le_bytes();
        image[XSTATE_BV_OFFSET as usize..][..8].copy_from_slice(&bv);
        let (h, stop) = run(XRSTOR_RDI, |h| {
            h.cpu.xcr0 = ALL_COMPONENTS;
            h.write(AREA, &image);
            h.cpu.gpr[RAX] = u64::MAX;
            h.cpu.gpr[RDX] = u64::MAX;
            h.cpu.ymm_hi[0] = 77;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.ymm_hi, [0; 16]);
        assert_eq!(h.cpu.zmm_hi[3], [pattern(203), pattern(303)]);
    }

    #[test]
    fn xrstors_needs_cpl_0_and_the_compacted_format() {
        // The harness runs code in supervisor pages, so the CPL check is made on the handlers.
        let mut h = Harness::new(&[HLT]);
        h.cpu.cr4 |= CR4_FXSR_XSAVE;
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
        h.write_u64(AREA + XCOMP_BV_OFFSET, XCOMP_BV_COMPACTED);
        h.cpu.gpr[RAX] = 3;
        h.cpu.gpr[RDX] = 0;
        assert!(h.cpu.xrstors(AREA).is_ok());
        assert!(h.cpu.xsaves(AREA).is_ok());
        h.cpu.cpl = 3;
        for result in [h.cpu.xrstors(AREA), h.cpu.xsaves(AREA)] {
            assert!(matches!(
                result,
                Err(Exception::Fault {
                    vector: 13,
                    error_code: Some(0),
                    ..
                })
            ));
        }
        let (_, stop) = run(XRSTORS_RDI, |h| {
            h.write(AREA, &[0; XSAVE_AREA_SIZE as usize]);
            h.cpu.gpr[RAX] = 3;
            h.cpu.gpr[RDX] = 0;
        });
        assert_eq!(stop, fault(13, Some(0)), "a standard-format area");
    }

    #[test]
    fn xsaves_matches_xsavec_while_no_supervisor_state_exists() {
        let (a, _) = run(XSAVEC_RDI, fill_vector_state);
        let (b, _) = run(XSAVES_RDI, fill_vector_state);
        let size = compacted_area_size(ALL_COMPONENTS) as usize;
        assert_eq!(a.read(AREA, size), b.read(AREA, size));
    }

    #[test]
    fn xrstor_rejects_bad_compacted_headers() {
        let (saved, _) = run(XSAVEC_RDI, fill_vector_state);
        let good = saved.read(AREA, enabled_area_size(ALL_COMPONENTS) as usize);
        let cases: [(&str, u64, u64); 4] = [
            (
                "component outside XCR0",
                XCOMP_BV_OFFSET,
                XCOMP_BV_COMPACTED | 1 << 9,
            ),
            ("XSTATE_BV beyond XCOMP_BV", XSTATE_BV_OFFSET, 1 << 8),
            ("reserved header bytes", XSAVE_HEADER_RESERVED_OFFSET + 8, 1),
            (
                "state bit for an absent component",
                XCOMP_BV_OFFSET,
                XCOMP_BV_COMPACTED | 3,
            ),
        ];
        for (what, offset, value) in cases {
            let mut image = good.clone();
            image[offset as usize..][..8].copy_from_slice(&value.to_le_bytes());
            let (h, stop) = run(XRSTOR_RDI, |h| {
                h.cpu.xcr0 = ALL_COMPONENTS;
                h.write(AREA, &image);
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
                h.cpu.xmm[1] = 5;
            });
            assert_eq!(stop, fault(13, Some(0)), "{what}");
            assert_eq!(h.cpu.xmm[1], 5, "{what}");
        }
    }

    #[test]
    fn xsavec_and_xrstors_follow_the_xsave_fault_order() {
        for code in [XSAVEC_RDI, XSAVES_RDI, XRSTORS_RDI] {
            let (_, stop) = run(code, |h| h.cpu.cr4 &= !CR4_OSXSAVE);
            assert_eq!(stop, fault(6, None), "no OSXSAVE");
            let (_, stop) = run(code, |h| h.cpu.cr0 |= CR0_TS);
            assert_eq!(stop, fault(7, None), "CR0.TS");
            let (_, stop) = run(code, |h| h.cpu.gpr[RDI] = AREA + 32);
            assert_eq!(stop, fault(13, Some(0)), "32-byte aligned is not enough");
        }
    }

    #[test]
    fn xgetbv_1_reports_the_components_that_left_their_initial_state() {
        let xgetbv_1 = |setup: &dyn Fn(&mut Harness)| {
            let (h, stop) = run(XGETBV, |h| {
                h.cpu.xcr0 = ALL_COMPONENTS;
                h.cpu.gpr[RCX] = 1;
                setup(h);
            });
            assert_eq!(stop, Stop::Halted);
            h.cpu.gpr[RAX] | h.cpu.gpr[RDX] << 32
        };
        assert_eq!(xgetbv_1(&|_| {}), 0, "everything at reset");
        assert_eq!(xgetbv_1(&|h| h.cpu.fcw = 0x027f), XCR0_X87);
        assert_eq!(xgetbv_1(&|h| h.cpu.xmm[3] = 1), XCR0_SSE);
        assert_eq!(xgetbv_1(&|h| h.cpu.mxcsr = 0x9fc0), XCR0_SSE);
        assert_eq!(xgetbv_1(&|h| h.cpu.ymm_hi[15] = 1), XCR0_AVX);
        assert_eq!(xgetbv_1(&|h| h.cpu.opmask[7] = 1), XCR0_OPMASK);
        assert_eq!(xgetbv_1(&|h| h.cpu.zmm_hi[0][1] = 1), XCR0_ZMM_HI256);
        assert_eq!(xgetbv_1(&|h| h.cpu.zmm_extra[15][3] = 1), XCR0_HI16_ZMM);
        // only components XCR0 enables are reported
        assert_eq!(
            xgetbv_1(&|h| {
                h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
                h.cpu.ymm_hi[0] = 1;
            }),
            0
        );
        let (_, stop) = run(XGETBV, |h| h.cpu.gpr[RCX] = 2);
        assert_eq!(stop, fault(13, Some(0)));
    }

    #[test]
    fn cpuid_describes_the_compacted_format() {
        let mut h = Harness::new(&[HLT]);
        h.cpu.xcr0 = ALL_COMPONENTS;
        h.cpu.gpr[RAX] = 0xd;
        h.cpu.gpr[RCX] = 1;
        h.cpu.cpuid();
        assert_eq!(
            h.cpu.gpr[RAX] as u32, 0xf,
            "xsaveopt, xsavec, xgetbv1, xsaves"
        );
        assert_eq!(h.cpu.gpr[RBX], COMPACT_HI16_ZMM + HI16_ZMM_STATE_SIZE);
        assert_eq!((h.cpu.gpr[RCX], h.cpu.gpr[RDX]), (0, 0), "no IA32_XSS bits");
        h.cpu.xcr0 = XCR0_X87 | XCR0_SSE;
        h.cpu.gpr[RAX] = 0xd;
        h.cpu.gpr[RCX] = 1;
        h.cpu.cpuid();
        assert_eq!(h.cpu.gpr[RBX], XSAVE_AREA_SIZE);
        // subleaf 2: no supervisor bit and no 64-byte alignment, which the packing relies on
        h.cpu.gpr[RAX] = 0xd;
        h.cpu.gpr[RCX] = 2;
        h.cpu.cpuid();
        assert_eq!(h.cpu.gpr[RCX], 0);
    }

    #[test]
    fn ia32_xss_reads_zero_and_accepts_only_zero() {
        use crate::cpu::sys::MSR_XSS;
        let mut h = Harness::new(&[HLT]);
        assert_eq!(h.cpu.rdmsr(MSR_XSS).unwrap(), 0);
        assert!(h.cpu.wrmsr(MSR_XSS, 0).is_ok());
        assert!(h.cpu.wrmsr(MSR_XSS, 1 << 11).is_err());
    }

    #[test]
    fn rdpid_returns_tsc_aux() {
        let (h, stop) = run(RDPID_RAX, |h| {
            h.cpu.tsc_aux = 0xdead_beef_0000_0007;
            h.cpu.gpr[RAX] = u64::MAX;
        });
        assert_eq!(stop, Stop::Halted);
        assert_eq!(h.cpu.gpr[RAX], 7, "the low 32 bits, zero-extended");
    }
}
