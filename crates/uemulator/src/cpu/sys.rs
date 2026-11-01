use super::decode::*;
use super::exception::{Exception, VECTOR_SEGMENT_NOT_PRESENT};
use super::exec::Loc;
use super::flags::*;
use super::fpu::MXCSR_VALID_BITS;
use super::*;
use crate::dev::apic;

pub(crate) const MSR_TSC: u32 = 0x10;
pub(crate) const MSR_APIC_BASE: u32 = 0x1b;
pub(crate) const MSR_SYSENTER_CS: u32 = 0x174;
pub(crate) const MSR_SYSENTER_ESP: u32 = 0x175;
pub(crate) const MSR_SYSENTER_EIP: u32 = 0x176;
pub(crate) const MSR_PAT: u32 = 0x277;
pub(crate) const MSR_EFER: u32 = 0xc000_0080;
pub(crate) const MSR_STAR: u32 = 0xc000_0081;
pub(crate) const MSR_LSTAR: u32 = 0xc000_0082;
pub(crate) const MSR_CSTAR: u32 = 0xc000_0083;
pub(crate) const MSR_SFMASK: u32 = 0xc000_0084;
pub(crate) const MSR_FS_BASE: u32 = 0xc000_0100;
pub(crate) const MSR_GS_BASE: u32 = 0xc000_0101;
pub(crate) const MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
pub(crate) const MSR_TSC_AUX: u32 = 0xc000_0103;
/// The cache line size CLZERO clears, and leaf 1 EBX reports in units of 8 bytes.
const CLZERO_LINE_BYTES: u64 = 64;
/// The legacy AMD performance counters, PERFCTR0-3 (0xC0010004-7), which `rdpmc` reads by index.
/// They hold whatever was written and never count; see `Cpu::new`.
pub(crate) const MSR_K7_PERFCTR0: u32 = 0xc001_0004;
const K7_PERF_COUNTERS: u32 = 4;
/// The counters are 48 bits wide.
const PERF_COUNTER_MASK: u64 = (1 << 48) - 1;
/// IA32_XSS: which supervisor state components xsaves/xrstors handle (none here).
pub(crate) const MSR_XSS: u32 = 0xda0;
pub(crate) const MSR_SPEC_CTRL: u32 = 0x48;
pub(crate) const MSR_PRED_CMD: u32 = 0x49;
pub(crate) const MSR_ARCH_CAPABILITIES: u32 = 0x10a;
pub(crate) const MSR_FLUSH_CMD: u32 = 0x10b;

/// IA32_SPEC_CTRL bits: IBRS, STIBP, SSBD. Any other bit is reserved and #GPs on write.
const SPEC_CTRL_WRITABLE: u64 = 0b111;
/// IA32_PRED_CMD bit 0 (IBPB) and IA32_FLUSH_CMD bit 0 (L1D flush): write-only command MSRs.
/// The emulator has no predictors or L1D to flush, so accepting the write is the whole job.
const COMMAND_MSR_WRITABLE: u64 = 1;

// IA32_ARCH_CAPABILITIES bits. The emulated CPU executes strictly in order with no speculation,
// so it states that it is immune to every speculative-execution vulnerability Linux knows about.
// That makes the kernel pick "Not affected" everywhere (no KPTI, no VERW on exit, no retpolines
// or return thunks, no BHB clearing loop on syscall entry), which is the point: those
// mitigations cost guest instructions the emulator would then have to run. Add a bit here only
// together with the kernel check it silences; a bit that is *not* a "_NO" bit (RSBA, bit 2)
// must stay clear because it marks the CPU as vulnerable.
const ARCH_CAP_RDCL_NO: u64 = 1 << 0;
const ARCH_CAP_IBRS_ALL: u64 = 1 << 1;
const ARCH_CAP_SKIP_L1DFL_VMENTRY: u64 = 1 << 3;
const ARCH_CAP_SSB_NO: u64 = 1 << 4;
const ARCH_CAP_MDS_NO: u64 = 1 << 5;
const ARCH_CAP_PSCHANGE_MC_NO: u64 = 1 << 6;
const ARCH_CAP_TAA_NO: u64 = 1 << 8;
const ARCH_CAP_SBDR_SSDP_NO: u64 = 1 << 13;
const ARCH_CAP_FBSDP_NO: u64 = 1 << 14;
const ARCH_CAP_PSDP_NO: u64 = 1 << 15;
const ARCH_CAP_BHI_NO: u64 = 1 << 20;
const ARCH_CAP_PBRSB_NO: u64 = 1 << 24;
const ARCH_CAP_GDS_NO: u64 = 1 << 26;
const ARCH_CAP_RFDS_NO: u64 = 1 << 27;
const ARCH_CAP_ITS_NO: u64 = 1 << 62;
const ARCH_CAPABILITIES: u64 = ARCH_CAP_RDCL_NO
    | ARCH_CAP_IBRS_ALL
    | ARCH_CAP_SKIP_L1DFL_VMENTRY
    | ARCH_CAP_SSB_NO
    | ARCH_CAP_MDS_NO
    | ARCH_CAP_PSCHANGE_MC_NO
    | ARCH_CAP_TAA_NO
    | ARCH_CAP_SBDR_SSDP_NO
    | ARCH_CAP_FBSDP_NO
    | ARCH_CAP_PSDP_NO
    | ARCH_CAP_BHI_NO
    | ARCH_CAP_PBRSB_NO
    | ARCH_CAP_GDS_NO
    | ARCH_CAP_RFDS_NO
    | ARCH_CAP_ITS_NO;

// CR0_PE/MP/ET/PG, CR4_PAE and EFER_LME/LMA/NXE live in cpu/mod.rs.
pub(crate) const CR0_EM: u64 = 1 << 2;
pub(crate) const CR0_TS: u64 = 1 << 3;
const CR0_NW: u64 = 1 << 29;
const CR0_CD: u64 = 1 << 30;

pub(crate) const CR4_TSD: u64 = 1 << 2;
pub(crate) const CR4_DE: u64 = 1 << 3;
/// Machine-check enable: advertised with the (inert) machine-check architecture in `mca.rs`.
pub(crate) const CR4_MCE: u64 = 1 << 6;
pub(crate) const CR4_PSE: u64 = 1 << 4;
pub(crate) const CR4_PGE: u64 = 1 << 7;
pub(crate) const CR4_PCE: u64 = 1 << 8;
pub(crate) const CR4_OSFXSR: u64 = 1 << 9;
pub(crate) const CR4_OSXMMEXCPT: u64 = 1 << 10;
/// Enables rdfsbase/rdgsbase/wrfsbase/wrgsbase (advertised in CPUID.7.0:EBX).
pub(crate) const CR4_FSGSBASE: u64 = 1 << 16;
/// Enables xgetbv/xsetbv/xsave/xrstor; also mirrored into CPUID.1:ECX bit 27.
pub(crate) const CR4_OSXSAVE: u64 = 1 << 18;
/// User-mode instruction prevention: sgdt/sidt/sldt/str/smsw are #GP outside CPL 0.
pub(crate) const CR4_UMIP: u64 = 1 << 11;
/// Process-context identifiers: CR3[11:0] is a PCID and bit 63 a no-flush hint.
pub(crate) const CR4_PCIDE: u64 = 1 << 17;
/// Supervisor-mode execution prevention: no instruction fetches from user pages at CPL < 3.
pub(crate) const CR4_SMEP: u64 = 1 << 20;
/// Supervisor-mode access prevention: no data accesses to user pages at CPL < 3 unless EFLAGS.AC.
pub(crate) const CR4_SMAP: u64 = 1 << 21;
/// CR4 bits a guest may set. Everything else (VME, PVI, UMIP, LA57, PCIDE, OSXSAVE, SMEP,
/// SMAP...) is rejected with #GP because CPUID does not advertise it.
const CR4_SUPPORTED: u64 = CR4_TSD
    | CR4_DE
    | CR4_PSE
    | CR4_PAE
    | CR4_MCE
    | CR4_PGE
    | CR4_PCE
    | CR4_OSFXSR
    | CR4_OSXMMEXCPT
    | CR4_FSGSBASE
    | CR4_OSXSAVE
    | CR4_UMIP
    | CR4_PCIDE
    | CR4_SMEP
    | CR4_SMAP;

pub(super) const CR4_SUPPORTED_BITS: u64 = CR4_SUPPORTED;

/// Address widths reported by CPUID 0x8000_0008. CR3 and the canonical-address checks follow them.
pub(super) const PHYSICAL_ADDRESS_BITS: u32 = 40;
pub(super) const VIRTUAL_ADDRESS_BITS: u32 = 48;

/// CR3 bits kept on write: physical address bits 51:12 plus PWT (bit 3) and PCD (bit 4).
/// Everything else (bits 2:0 and 11:5, the PCID field) is ignored without CR4.PCIDE.
const CR3_MASK: u64 = 0x000f_ffff_ffff_f018;
/// Bits at or above the physical address width are reserved in CR3: writing one is #GP(0).
/// CR3 bit 63 with CR4.PCIDE: do not invalidate the new PCID's translations.
const CR3_NO_FLUSH: u64 = 1 << 63;
/// CR3 bits kept with CR4.PCIDE: the address plus the whole 12-bit PCID.
const CR3_MASK_PCID: u64 = 0x000f_ffff_ffff_ffff;
/// Outside IA-32e mode CR3 is 32 bits wide: 32-bit paging uses bits 31:12 (PWT, PCD in 4:3), PAE
/// paging the 32-byte-aligned PDPT address in bits 31:5.
const CR3_NON_LONG_RESERVED: u64 = !0u64 << 32;
const CR3_MASK_LEGACY: u64 = 0xffff_f018;
const CR3_MASK_PAE: u64 = 0xffff_ffe0 | 0x18;
pub(super) const CR3_RESERVED_BITS: u64 = CR3_RESERVED;
const CR3_RESERVED: u64 = !0u64 << PHYSICAL_ADDRESS_BITS;
/// CR8 holds the 4-bit task priority; it is stored but no APIC consumes it. Higher bits are
/// reserved.
const CR8_MASK: u64 = 0xf;
/// CR8 is TPR[7:4] (SDM 10.8.6.1).
const CR8_TPR_SHIFT: u32 = 4;

/// DR6 bits software may write: B0-B3 (3:0), BD (13), BS (14), BT (15). Bits 31:16 and 11:4
/// always read as one (no RTM, no bus-lock detect) and bit 12 as zero.
const DR6_WRITABLE: u64 = 0xe00f;
const DR6_FIXED_ONE: u64 = 0xffff_0ff0;
/// DR7 bits software may write: L0-G3, LE, GE (9:0), GD (13) and the RW/LEN fields (31:16).
/// Bit 10 always reads as one; bits 15:14 and 12:11 as zero.
const DR7_WRITABLE: u64 = 0xffff_23ff;
const DR7_FIXED_ONE: u64 = 0x400;
/// Debug register numbers. DR4 and DR5 alias DR6 and DR7 when CR4.DE is clear and are #UD
/// otherwise; DR8-DR15 do not exist.
const DR4: usize = 4;
const DR5: usize = 5;
const DR6: usize = 6;
const DR7: usize = 7;
const DEBUG_REGISTER_COUNT: usize = 8;

pub(crate) const EFER_SCE: u64 = 1 << 0;
/// wrmsr to EFER faults if any bit outside this set is 1. LMA is accepted in the value but
/// never stored: it is recomputed from CR0.PG and EFER.LME on CR0 writes.
const EFER_WRMSR_ALLOWED: u64 = EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE | super::svm::EFER_SVME;
const EFER_WRMSR_STORED: u64 = EFER_SCE | EFER_LME | EFER_NXE | super::svm::EFER_SVME;

const EFLAGS_ID: u64 = 1 << 21;
/// Bit 1 of EFLAGS always reads as 1.
const EFLAGS_ALWAYS_ONE: u64 = 2;
/// Flags sysret restores from r11. IOPL, RF and VM are deliberately not restored.
const SYSRET_RESTORED_FLAGS: u64 = CF | PF | AF | ZF | SF | TF | IF | DF | OF | NT | AC | EFLAGS_ID;

/// Low two bits of a segment selector.
const SELECTOR_RPL_MASK: u16 = 3;
const SELECTOR_RPL_USER: u16 = 3;

/// STAR holds the sysret selector base in bits 63:48 and the syscall selector base in 47:32.
const STAR_SYSCALL_SELECTOR_SHIFT: u32 = 32;
const STAR_SYSRET_SELECTOR_SHIFT: u32 = 48;

const DESC_LIMIT_HIGH_MASK: u32 = 0xf_0000;
/// Descriptor type field (bits 11:8 of the upper dword) values of the system descriptors LDT
/// and 64-bit TSS accept; a busy TSS is the available type with bit 1 set.
const DESC_TYPE_SHIFT: u32 = 8;
const DESC_TYPE_MASK: u32 = 0xf;
const SYSTEM_TYPE_LDT: u32 = 2;
const SYSTEM_TYPE_TSS64_AVAILABLE: u32 = 9;
/// The "busy" bit of the TSS type, in its place within the descriptor's upper dword.
const SYSTEM_TYPE_BUSY_BIT: u32 = 2 << DESC_TYPE_SHIFT;
/// Conforming bit of a code segment type.
const SEG_TYPE_CONFORMING: u32 = 1 << 10;
/// Descriptors are 8 bytes, system descriptors (LDT, TSS) 16 in long mode.
const DESCRIPTOR_BYTES: u64 = 8;
/// lgdt/lidt with a 16-bit operand size use only the low 24 bits of the base.
const TABLE_BASE_24_MASK: u64 = 0x00ff_ffff;
const SYSTEM_DESCRIPTOR_BYTES: u64 = 16;
/// Byte offset of the upper dword of a descriptor, where type and attributes live.
const DESC_HIGH_DWORD_OFFSET: u64 = 4;
/// Selector fields: table indicator (set selects the LDT) and the descriptor byte offset.
const SELECTOR_TABLE_INDICATOR: u16 = 1 << 2;
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
const FLAT_LIMIT: u32 = 0xffff_ffff;
const DESC_DPL_USER: u32 = SEG_DPL_MASK << SEG_DPL_SHIFT;
/// Flat segments installed by syscall/sysret (base 0, 4 GiB limit).
const USER_CODE64_FLAGS: u32 = SEG_FLAGS_CODE64 | DESC_DPL_USER;
/// A flat 32-bit ring-0 code segment, as sysenter installs.
pub(crate) const SEG_FLAGS_CODE32: u32 = SEG_GRANULARITY
    | SEG_DEFAULT_BIG
    | SEG_PRESENT
    | SEG_NON_SYSTEM
    | SEG_TYPE_EXECUTABLE
    | SEG_TYPE_READ_WRITE;
const USER_DATA_FLAGS: u32 = SEG_FLAGS_FLAT_DATA | DESC_DPL_USER;

/// `Cpu::log_once` keys for faulting MSR accesses, offset by the 32-bit MSR number so each base
/// owns a 2^32-wide range. The other users are `unimplemented_opcode` in exec.rs (small keys from
/// 0x4000_0000) and the unmapped-access keys in mem.rs (bits 62 and 61 plus a page number), so
/// these two bases sit below those in bits 60 and 59.
const LOG_KEY_RDMSR_FAULT: u64 = 1 << 60;
const LOG_KEY_WRMSR_FAULT: u64 = 1 << 59;

/// Indices into `Cpu::gpr`: hardware register numbers, so RCX is 1 and RBX is 3.
pub(super) const RAX: usize = 0;
pub(super) const RCX: usize = 1;
pub(super) const RDX: usize = 2;
pub(super) const RBX: usize = 3;
const R11: usize = 11;

/// Whether bits 63:47 of `address` are all copies of bit 47.
pub(super) fn is_canonical(address: u64) -> bool {
    let sign_extension = (address as i64) >> (VIRTUAL_ADDRESS_BITS - 1);
    sign_extension == 0 || sign_extension == -1
}

impl Cpu {
    /// #GP(0) unless CPL is 0.
    pub fn need_cpl0(&self) -> ExecResult<()> {
        if self.cpl != 0 {
            Err(Exception::general_protection(0))
        } else {
            Ok(())
        }
    }

    /// Unknown MSRs raise #GP(0) (the kernel probes with `rdmsr_safe`); each is logged once.
    pub fn rdmsr(&mut self, msr: u32) -> ExecResult<u64> {
        Ok(match msr {
            MSR_TSC => self.tsc(),
            MSR_APIC_BASE => self.dev.lock().lapic().base_msr(),
            apic::X2APIC_MSR_FIRST..=apic::X2APIC_MSR_LAST => {
                let now = self.now_ns();
                match self.dev.lock().apic_msr_read(msr, now) {
                    Some(value) => value,
                    None => return Err(Exception::general_protection(0)),
                }
            }
            apic::MSR_TSC_DEADLINE => self.dev.lock().lapic().tsc_deadline(),
            MSR_PAT => self.pat,
            _ if mtrr::Mtrr::owns(msr) => self.mtrr.read(msr).unwrap_or(0),
            _ if mca::Mca::owns(msr) => self.mca.read(msr).unwrap_or(0),
            MSR_EFER => self.efer,
            MSR_STAR => self.star,
            MSR_LSTAR => self.lstar,
            MSR_CSTAR => self.cstar,
            MSR_SFMASK => self.sfmask,
            // FS/GS bases live in the segment cache, not in separate fields
            MSR_FS_BASE => self.seg[SEG_FS].base,
            MSR_GS_BASE => self.seg[SEG_GS].base,
            MSR_KERNEL_GS_BASE => self.kgsbase,
            MSR_TSC_AUX => self.tsc_aux,
            MSR_XSS => xsave::XSS_SUPPORTED,
            super::svm::MSR_VM_CR => self.svm.vm_cr,
            super::svm::MSR_VM_HSAVE_PA => self.svm.hsave_pa,
            MSR_SPEC_CTRL => self.spec_ctrl,
            MSR_ARCH_CAPABILITIES => ARCH_CAPABILITIES,
            MSR_SYSENTER_CS..=MSR_SYSENTER_EIP => self.sysenter[(msr - MSR_SYSENTER_CS) as usize],
            _ => match self.msr_misc.get(&msr) {
                Some(v) => *v,
                None => {
                    let rip = self.rip;
                    self.log_once(LOG_KEY_RDMSR_FAULT + msr as u64, || {
                        format!("rdmsr {msr:#x} -> #GP at rip {rip:#x}")
                    });
                    return Err(Exception::general_protection(0));
                }
            },
        })
    }

    /// Writes to MSRs in `msr_misc` are stored and read back but have no effect. Unknown MSRs
    /// raise #GP(0).
    pub fn wrmsr(&mut self, msr: u32, v: u64) -> ExecResult<()> {
        // Addresses that the CPU will later jump or index through must be canonical at write
        // time, or swapgs/syscall/sysenter would fault in the kernel entry path instead.
        let holds_address = matches!(
            msr,
            MSR_FS_BASE
                | MSR_GS_BASE
                | MSR_KERNEL_GS_BASE
                | MSR_LSTAR
                | MSR_CSTAR
                | MSR_SYSENTER_ESP
                | MSR_SYSENTER_EIP
        );
        if holds_address && !is_canonical(v) {
            return Err(Exception::general_protection(0));
        }
        match msr {
            MSR_APIC_BASE => {
                if !self.dev.lock().lapic_mut().set_base_msr(v) {
                    return Err(Exception::general_protection(0));
                }
            }
            apic::X2APIC_MSR_FIRST..=apic::X2APIC_MSR_LAST => {
                let now = self.now_ns();
                if !self.dev.lock().apic_msr_write(msr, v, now) {
                    return Err(Exception::general_protection(0));
                }
            }
            apic::MSR_TSC_DEADLINE => {
                let now = self.now_ns();
                self.dev.lock().lapic_mut().set_tsc_deadline(v, now);
            }
            MSR_PAT => {
                if !mtrr::pat_is_valid(v) {
                    return Err(Exception::general_protection(0));
                }
                self.pat = v;
            }
            _ if mtrr::Mtrr::owns(msr) => {
                if !self.mtrr.write(msr, v) {
                    return Err(Exception::general_protection(0));
                }
            }
            _ if mca::Mca::owns(msr) => {
                if !self.mca.write(msr, v) {
                    return Err(Exception::general_protection(0));
                }
            }
            MSR_EFER => {
                // SVME cannot be set while the firmware has disabled SVM in VM_CR
                if v & !EFER_WRMSR_ALLOWED != 0
                    || (v & super::svm::EFER_SVME != 0
                        && self.svm.vm_cr & super::svm::VM_CR_SVMDIS != 0)
                {
                    return Err(Exception::general_protection(0));
                }
                // LME can only change while paging is off (SDM 9.8.5).
                if (v ^ self.efer) & EFER_LME != 0 && self.cr0 & CR0_PG != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.efer = (self.efer & !EFER_WRMSR_STORED) | (v & EFER_WRMSR_STORED);
            }
            MSR_STAR => self.star = v,
            MSR_LSTAR => self.lstar = v,
            MSR_CSTAR => self.cstar = v,
            MSR_SFMASK => self.sfmask = v,
            MSR_FS_BASE => self.seg[SEG_FS].base = v,
            MSR_GS_BASE => self.seg[SEG_GS].base = v,
            MSR_KERNEL_GS_BASE => self.kgsbase = v,
            MSR_TSC_AUX => self.tsc_aux = v,
            // no supervisor state component is implemented, so the only value is "none"
            MSR_XSS => {
                if v & !xsave::XSS_SUPPORTED != 0 {
                    return Err(Exception::general_protection(0));
                }
            }
            super::svm::MSR_VM_CR => {
                if v & !super::svm::VM_CR_WRITABLE != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.svm.vm_cr = v;
            }
            super::svm::MSR_VM_HSAVE_PA => {
                // a 4 KiB aligned physical address inside the address space
                if v & (PAGE_SIZE - 1) != 0 || v >> PHYSICAL_ADDRESS_BITS != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.svm.hsave_pa = v;
            }
            MSR_SPEC_CTRL => {
                if v & !SPEC_CTRL_WRITABLE != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.spec_ctrl = v;
            }
            MSR_PRED_CMD | MSR_FLUSH_CMD => {
                if v & !COMMAND_MSR_WRITABLE != 0 {
                    return Err(Exception::general_protection(0));
                }
            }
            // the TSC follows the virtual clock and cannot be set
            MSR_TSC => {}
            MSR_SYSENTER_CS..=MSR_SYSENTER_EIP => {
                self.sysenter[(msr - MSR_SYSENTER_CS) as usize] = v
            }
            _ if self.msr_misc.contains_key(&msr) => {
                self.msr_misc.insert(msr, v);
            }
            _ => {
                let rip = self.rip;
                self.log_once(LOG_KEY_WRMSR_FAULT + msr as u64, || {
                    format!("wrmsr {msr:#x} <- {v:#x} -> #GP at rip {rip:#x}")
                });
                return Err(Exception::general_protection(0));
            }
        }
        Ok(())
    }

    /// 0F 20..23: mov to/from control and debug registers. ModRM.reg is the CR/DR number
    /// (REX.R makes CR8 or DR8-DR15) and ModRM.rm the GPR; mod is ignored. Nothing triggers
    /// hardware breakpoints, so the debug registers are validated storage.
    pub fn mov_cr_dr(&mut self, i: &Insn) -> ExecResult<()> {
        const OP_MOV_FROM_CR: u16 = 0x120;
        const OP_MOV_FROM_DR: u16 = 0x121;
        const OP_MOV_TO_CR: u16 = 0x122;
        self.need_cpl0()?;
        let register = i.reg as usize;
        let gpr = i.rm as usize;
        match i.op {
            OP_MOV_FROM_CR => {
                let value = match register {
                    0 => self.cr0,
                    2 => self.cr2,
                    3 => self.cr3,
                    4 => self.cr4,
                    8 => (self.dev.lock().lapic().tpr() >> CR8_TPR_SHIFT) as u64,
                    _ => return Err(Exception::invalid_opcode()),
                };
                // Outside IA-32e mode the access is 32 bits wide and clears the upper half.
                self.gpr[gpr] = if self.efer & EFER_LMA != 0 {
                    value
                } else {
                    value & 0xffff_ffff
                };
            }
            OP_MOV_TO_CR => {
                let v = if self.efer & EFER_LMA != 0 {
                    self.gpr[gpr]
                } else {
                    self.gpr[gpr] & 0xffff_ffff
                };
                match register {
                    0 => self.write_cr0(v)?,
                    2 => self.cr2 = v,
                    3 => self.write_cr3(v)?,
                    4 => {
                        // PAE cannot be turned off under IA-32e paging. PCIDE needs IA-32e
                        // mode and may only be set while the PCID field of CR3 is zero.
                        let long_mode = self.efer & EFER_LMA != 0;
                        if v & !CR4_SUPPORTED != 0
                            || (long_mode && v & CR4_PAE == 0)
                            || (v & CR4_PCIDE != 0
                                && (!long_mode
                                    || (self.cr4 & CR4_PCIDE == 0 && self.cr3 & 0xfff != 0)))
                        {
                            return Err(Exception::general_protection(0));
                        }
                        let mode_before = self.paging_mode();
                        let pdpte = self.pdptes_if_pae_paging(self.cr0, v, self.cr3)?;
                        self.cr4 = v;
                        if let Some(pdpte) = pdpte {
                            self.pdpte = pdpte;
                        }
                        self.paging_changed(mode_before);
                    }
                    8 => {
                        if v & !CR8_MASK != 0 {
                            return Err(Exception::general_protection(0));
                        }
                        self.dev
                            .lock()
                            .lapic_mut()
                            .set_tpr((v as u8) << CR8_TPR_SHIFT);
                    }
                    _ => return Err(Exception::invalid_opcode()),
                }
            }
            OP_MOV_FROM_DR => {
                let index = self.debug_register_index(register)?;
                self.gpr[gpr] = if self.efer & EFER_LMA != 0 {
                    self.dr[index]
                } else {
                    self.dr[index] & 0xffff_ffff
                };
            }
            _ => {
                let index = self.debug_register_index(register)?;
                let v = self.gpr[gpr];
                self.write_debug_register(index, v)?;
            }
        }
        Ok(())
    }

    /// Validates a CR0 write (SDM Vol. 3A 2.5 and 10.8.5) before anything changes. LMA is
    /// derived here: long mode is active exactly when paging is on with EFER.LME set.
    fn write_cr0(&mut self, v: u64) -> ExecResult<()> {
        let paging = v & CR0_PG != 0;
        let upper_half_set = v >> 32 != 0;
        let not_write_through_without_cache_disable = v & CR0_NW != 0 && v & CR0_CD == 0;
        let paging_without_protection = paging && v & CR0_PE == 0;
        // Paging is switched off from compatibility mode (that is how software leaves IA-32e
        // mode); doing it from 64-bit code would leave that code running unpaged.
        let leaving_long_mode = self.efer & EFER_LMA != 0 && !paging && self.seg[SEG_CS].long();
        let entering_long_mode_without_pae =
            paging && self.efer & EFER_LME != 0 && self.cr4 & CR4_PAE == 0;
        // Activating IA-32e mode must happen from compatibility or legacy code: with CS.L set the
        // CPU would be in 64-bit mode on an unmapped instruction stream (SDM 9.8.5.4).
        let entering_long_mode_from_64bit_code = paging
            && self.efer & EFER_LME != 0
            && self.efer & EFER_LMA == 0
            && self.seg[SEG_CS].long();
        // PCIDs only exist in IA-32e mode.
        let leaving_paging_with_pcid = !paging && self.cr4 & CR4_PCIDE != 0;
        if upper_half_set
            || not_write_through_without_cache_disable
            || paging_without_protection
            || leaving_long_mode
            || entering_long_mode_without_pae
            || entering_long_mode_from_64bit_code
            || leaving_paging_with_pcid
        {
            return Err(Exception::general_protection(0));
        }
        let mode_before = self.paging_mode();
        let long_mode_after = paging && self.efer & EFER_LME != 0;
        let pdpte = if long_mode_after {
            None
        } else {
            self.pdptes_if_pae_paging(v, self.cr4, self.cr3)?
        };
        self.cr0 = v | CR0_ET;
        if long_mode_after {
            self.efer |= EFER_LMA;
        } else {
            self.efer &= !EFER_LMA;
        }
        if let Some(pdpte) = pdpte {
            self.pdpte = pdpte;
        }
        self.paging_changed(mode_before);
        Ok(())
    }

    /// The PDPT entries to latch when `cr0`/`cr4`/`cr3` select PAE paging (and, as a side
    /// effect of validating them, the #GP for reserved bits); `None` for any other format.
    /// Long mode is never PAE paging, whatever CR4.PAE says: callers pass the CR0/CR4 they are
    /// about to install, and LMA is derived from them only in `write_cr0`.
    fn pdptes_if_pae_paging(
        &mut self,
        cr0: u64,
        cr4: u64,
        cr3: u64,
    ) -> ExecResult<Option<[u64; 4]>> {
        let pae_paging = cr0 & CR0_PG != 0 && cr4 & CR4_PAE != 0 && self.efer & EFER_LMA == 0;
        if pae_paging {
            self.read_pdptes(cr3).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Every paging-relevant control register write drops cached translations; a change of
    /// format also drops decoded instructions, since they were found through the old one.
    fn paging_changed(&mut self, mode_before: super::mem::PagingMode) {
        self.flush_tlb();
        if self.paging_mode() != mode_before {
            self.flush_decode();
        }
    }

    /// MOV to CR3. The register is 32 bits wide outside IA-32e mode, where PAE paging also
    /// latches the four PDPT entries now (#GP if one is malformed).
    pub(crate) fn write_cr3(&mut self, v: u64) -> ExecResult<()> {
        if self.efer & EFER_LMA != 0 {
            // With CR4.PCIDE, bits 11:0 are the PCID (not PWT/PCD) and bit 63 asks
            // not to flush; the TLB is untagged, so the hint is ignored (flushing
            // more than asked is always allowed).
            let pcide = self.cr4 & CR4_PCIDE != 0;
            let reserved = if pcide {
                CR3_RESERVED & !CR3_NO_FLUSH
            } else {
                CR3_RESERVED
            };
            if v & reserved != 0 {
                return Err(Exception::general_protection(0));
            }
            self.cr3 = v & if pcide { CR3_MASK_PCID } else { CR3_MASK };
            self.flush_tlb();
            return Ok(());
        }
        if v & CR3_NON_LONG_RESERVED != 0 {
            return Err(Exception::general_protection(0));
        }
        let mode = self.paging_mode();
        if mode == super::mem::PagingMode::Pae {
            self.pdpte = self.read_pdptes(v)?;
            self.cr3 = v & CR3_MASK_PAE;
        } else {
            self.cr3 = v & CR3_MASK_LEGACY;
        }
        self.flush_tlb();
        Ok(())
    }

    /// Maps a MOV DR register number to its slot in `dr`: DR4/DR5 alias DR6/DR7 unless CR4.DE
    /// makes them #UD, and numbers 8-15 (REX.R) do not exist.
    fn debug_register_index(&self, register: usize) -> ExecResult<usize> {
        match register {
            DR4 | DR5 if self.cr4 & CR4_DE != 0 => Err(Exception::invalid_opcode()),
            DR4 | DR5 => Ok(register + (DR6 - DR4)),
            _ if register < DEBUG_REGISTER_COUNT => Ok(register),
            _ => Err(Exception::invalid_opcode()),
        }
    }

    fn write_debug_register(&mut self, index: usize, v: u64) -> ExecResult<()> {
        match index {
            DR6 | DR7 => {
                if v >> 32 != 0 {
                    return Err(Exception::general_protection(0));
                }
                self.dr[index] = if index == DR6 {
                    (v & DR6_WRITABLE) | DR6_FIXED_ONE
                } else {
                    (v & DR7_WRITABLE) | DR7_FIXED_ONE
                };
            }
            _ => self.dr[index] = v,
        }
        Ok(())
    }

    /// `next` is the address of the following instruction. SYSCALL loads CS from STAR[47:32]
    /// and SS from that plus 8, always flat, and masks RFLAGS with SFMASK.
    pub fn do_syscall(&mut self, next: u64) -> ExecResult<()> {
        if self.efer & EFER_SCE == 0 {
            return Err(Exception::invalid_opcode());
        }
        // SYSCALL only exists in IA-32e mode here (legacy-mode SYSCALL is not implemented); from
        // compatibility mode it enters 64-bit code at CSTAR instead of LSTAR.
        if self.efer & EFER_LMA == 0 {
            return Err(Exception::invalid_opcode());
        }
        let from_compatibility_mode = !self.is_long64();
        let rflags = self.eflags_full();
        self.gpr[RCX] = next;
        self.gpr[R11] = rflags & !RF;
        let selector_base = (self.star >> STAR_SYSCALL_SELECTOR_SHIFT) as u16 & !SELECTOR_RPL_MASK;
        self.seg[SEG_CS] = Seg {
            sel: selector_base,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_CODE64,
        };
        self.seg[SEG_SS] = Seg {
            sel: selector_base + 8,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_FLAT_DATA,
        };
        self.eflags = (rflags & !self.sfmask & !RF) | EFLAGS_ALWAYS_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.set_cpl(0);
        self.rip = if from_compatibility_mode {
            self.cstar
        } else {
            self.lstar
        };
        Ok(())
    }

    /// SYSENTER, legal in protected mode only: this emulator identifies as AMD hardware, where it
    /// (like SYSEXIT) is #UD in IA-32e mode and #GP(0) in real mode. CS = SYSENTER_CS, SS = CS + 8,
    /// both flat ring-0 segments; ESP and EIP come from the SYSENTER MSRs; VM, IF and RF are cleared.
    pub fn do_sysenter(&mut self) -> ExecResult<()> {
        const SYSENTER_STACK_OFFSET: u16 = 8;
        const SELECTOR_NULL_MASK: u64 = 0xfffc;
        if self.efer & EFER_LMA != 0 {
            return Err(Exception::invalid_opcode());
        }
        let [cs, esp, eip] = self.sysenter;
        if self.cr0 & CR0_PE == 0 || cs & SELECTOR_NULL_MASK == 0 {
            return Err(Exception::general_protection(0));
        }
        let selector = cs as u16 & !SELECTOR_RPL_MASK;
        self.seg[SEG_CS] = Seg {
            sel: selector,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_CODE32,
        };
        self.seg[SEG_SS] = Seg {
            sel: selector + SYSENTER_STACK_OFFSET,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_FLAT_DATA,
        };
        self.materialize_lazy_flags();
        self.eflags &= !(VM | IF | RF);
        self.set_cpl(0);
        self.gpr[GPR_RSP] = esp & 0xffff_ffff;
        self.rip = eip & 0xffff_ffff;
        Ok(())
    }

    /// SYSEXIT: back to ring 3 with CS = SYSENTER_CS + 16 and SS = SYSENTER_CS + 24, EIP from EDX
    /// and ESP from ECX.
    pub fn do_sysexit(&mut self) -> ExecResult<()> {
        const SYSEXIT_CODE_OFFSET: u16 = 16;
        const SYSEXIT_STACK_OFFSET: u16 = 24;
        const SELECTOR_NULL_MASK: u64 = 0xfffc;
        if self.efer & EFER_LMA != 0 {
            return Err(Exception::invalid_opcode());
        }
        let cs = self.sysenter[0];
        if self.cr0 & CR0_PE == 0 || self.cpl != 0 || cs & SELECTOR_NULL_MASK == 0 {
            return Err(Exception::general_protection(0));
        }
        let selector = cs as u16 & !SELECTOR_RPL_MASK;
        self.seg[SEG_CS] = Seg {
            sel: (selector + SYSEXIT_CODE_OFFSET) | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: SEG_FLAGS_CODE32 | DESC_DPL_USER,
        };
        self.seg[SEG_SS] = Seg {
            sel: (selector + SYSEXIT_STACK_OFFSET) | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: USER_DATA_FLAGS,
        };
        self.set_cpl(3);
        self.rip = self.gpr[RDX] & 0xffff_ffff;
        self.gpr[GPR_RSP] = self.gpr[RCX] & 0xffff_ffff;
        Ok(())
    }

    /// SYSRET. With REX.W (`sysretq`) it returns to 64-bit mode (CS = STAR[63:48] plus 16); without
    /// it (`sysretl`) to compatibility mode (CS = STAR[63:48], a flat 32-bit code segment). SS is
    /// STAR[63:48] plus 8 either way. Both use RPL 3 and take RIP from RCX and the flags from R11.
    pub fn do_sysret(&mut self, i: &Insn) -> ExecResult<()> {
        if self.efer & EFER_SCE == 0 || self.efer & EFER_LMA == 0 {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()?;
        let selector_base = (self.star >> STAR_SYSRET_SELECTOR_SHIFT) as u16 & !SELECTOR_RPL_MASK;
        let (code_selector, code_flags, target) = if i.opsize == 8 {
            (selector_base + 16, USER_CODE64_FLAGS, self.gpr[RCX])
        } else {
            (
                selector_base,
                SEG_FLAGS_CODE32 | DESC_DPL_USER,
                self.gpr[RCX] & 0xffff_ffff,
            )
        };
        self.seg[SEG_CS] = Seg {
            sel: code_selector | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: code_flags,
        };
        self.seg[SEG_SS] = Seg {
            sel: (selector_base + 8) | SELECTOR_RPL_USER,
            base: 0,
            limit: FLAT_LIMIT,
            flags: USER_DATA_FLAGS,
        };
        let r11 = self.gpr[R11];
        self.eflags = (r11 & SYSRET_RESTORED_FLAGS) | EFLAGS_ALWAYS_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.set_cpl(3);
        self.rip = target;
        Ok(())
    }

    /// Base and byte-granular limit of a 16-byte system descriptor (LDT/TSS) given as two
    /// quadwords. Base bits 63:32 come from the second quadword.
    fn system_descriptor_base_limit(lo: u64, hi: u64) -> (u64, u32) {
        let dword0 = lo as u32;
        let dword1 = (lo >> 32) as u32;
        let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64
            | ((hi & 0xffff_ffff) << 32);
        let mut limit = (dword0 & 0xffff) | (dword1 & DESC_LIMIT_HIGH_MASK);
        if dword1 & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        (base, limit)
    }

    /// LLDT: the descriptor is a 16-byte system descriptor in the GDT. A null selector marks the
    /// LDT invalid but still becomes the visible selector, which SLDT reads back.
    fn lldt(&mut self, sel: u16) -> ExecResult<()> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            self.ldtr = Seg {
                sel,
                ..Seg::default()
            };
            return Ok(());
        }
        let seg = self.load_system_descriptor(sel, SYSTEM_TYPE_LDT)?;
        self.ldtr = seg;
        Ok(())
    }

    /// LTR: only an available 64-bit TSS loads, and doing so marks it busy in the GDT.
    fn ltr(&mut self, sel: u16) -> ExecResult<()> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            return Err(Exception::general_protection(0));
        }
        let mut seg = self.load_system_descriptor(sel, SYSTEM_TYPE_TSS64_AVAILABLE)?;
        let high_dword_address =
            self.gdtr.base + (sel & SELECTOR_OFFSET_MASK) as u64 + DESC_HIGH_DWORD_OFFSET;
        // The write comes before the TR update so a faulting store leaves TR as it was.
        let high_dword = self.read_sys(high_dword_address, 4)?;
        self.write_sys(
            high_dword_address,
            4,
            high_dword | SYSTEM_TYPE_BUSY_BIT as u64,
        )?;
        seg.flags |= SYSTEM_TYPE_BUSY_BIT;
        self.tr = seg;
        Ok(())
    }

    /// Reads the 16-byte GDT descriptor for `sel` and checks it is a present system descriptor
    /// of `required_type`: #GP(sel) for a selector in the LDT, outside the GDT limit (the whole
    /// 16 bytes must fit) or of another type, then #NP(sel). Error codes drop the RPL bits.
    fn load_system_descriptor(&mut self, sel: u16, required_type: u32) -> ExecResult<Seg> {
        let error_code = (sel & !SELECTOR_RPL_MASK) as u32;
        let offset = (sel & SELECTOR_OFFSET_MASK) as u64;
        // System descriptors are 16 bytes in IA-32e mode and 8 elsewhere.
        let descriptor_bytes = if self.efer & EFER_LMA != 0 {
            SYSTEM_DESCRIPTOR_BYTES
        } else {
            DESCRIPTOR_BYTES
        };
        if sel & SELECTOR_TABLE_INDICATOR != 0
            || offset + descriptor_bytes - 1 > self.gdtr.limit as u64
        {
            return Err(Exception::general_protection(error_code));
        }
        let (lo, hi) = self.read_desc(sel)?;
        let (base, limit) = Self::system_descriptor_base_limit(lo, hi);
        let seg = Seg {
            sel,
            base,
            limit,
            flags: ((lo >> 32) as u32) & SEG_DESCRIPTOR_FLAGS_MASK,
        };
        let descriptor_type = (seg.flags >> DESC_TYPE_SHIFT) & DESC_TYPE_MASK;
        if seg.flags & SEG_NON_SYSTEM != 0 || descriptor_type != required_type {
            return Err(Exception::general_protection(error_code));
        }
        if !seg.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(error_code),
            ));
        }
        Ok(seg)
    }

    /// VERR/VERW: whether the segment named by `sel` can be read (or written) at the current
    /// privilege. Never faults on a bad selector (null, outside the table, system descriptor);
    /// only the descriptor fetch itself can fault.
    fn segment_accessible(&mut self, sel: u16, for_write: bool) -> ExecResult<bool> {
        if sel & !SELECTOR_RPL_MASK == 0 {
            return Ok(false);
        }
        let table_limit = if sel & SELECTOR_TABLE_INDICATOR != 0 {
            self.ldtr.limit
        } else {
            self.gdtr.limit as u32
        };
        if (sel & SELECTOR_OFFSET_MASK) as u64 + DESCRIPTOR_BYTES - 1 > table_limit as u64 {
            return Ok(false);
        }
        let (lo, _) = self.read_desc(sel)?;
        let flags = ((lo >> 32) as u32) & SEG_DESCRIPTOR_FLAGS_MASK;
        if flags & SEG_NON_SYSTEM == 0 {
            return Ok(false);
        }
        let descriptor_dpl = ((flags >> SEG_DPL_SHIFT) & SEG_DPL_MASK) as u8;
        let requestor_rpl = (sel & SELECTOR_RPL_MASK) as u8;
        let privileged_enough = descriptor_dpl >= self.cpl && descriptor_dpl >= requestor_rpl;
        let executable = flags & SEG_TYPE_EXECUTABLE != 0;
        // Bit 9 is W for data and R for code.
        let read_write = flags & SEG_TYPE_READ_WRITE != 0;
        Ok(if for_write {
            !executable && read_write && privileged_enough
        } else if executable {
            // Conforming code is readable from any privilege level.
            read_write && (flags & SEG_TYPE_CONFORMING != 0 || privileged_enough)
        } else {
            privileged_enough
        })
    }

    /// Bytes of base address in the memory image of sgdt/sidt/lgdt/lidt: 8 in IA-32e mode, else 4
    /// with a 32-bit operand size and 3 (a 24-bit base) with a 16-bit one.
    fn table_register_base_bytes(&self, i: &Insn) -> u8 {
        if self.efer & EFER_LMA != 0 {
            8
        } else if i.opsize == 2 {
            3
        } else {
            4
        }
    }

    /// LAR (`0F 02`) and LSL (`0F 03`): read a descriptor's access rights or limit through a
    /// selector, setting ZF when the selector is usable at the current privilege and leaving the
    /// destination alone (and ZF clear) when it is not. Both are #UD outside protected mode.
    pub(super) fn load_access_rights_or_limit(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        const AR_VISIBLE_MASK: u64 = 0x00ff_ff00;
        if self.cr0 & CR0_PE == 0 || self.eflags & VM != 0 {
            return Err(Exception::invalid_opcode());
        }
        let loc = self.modrm_location(i, next)?;
        let selector = self.load_operand(loc, 2, i)? as u16;
        let wants_limit = i.op == 0x103;
        let flags = self.eflags_full();
        let usable = self.descriptor_for_lar_lsl(selector, wants_limit)?;
        match usable {
            Some((high_dword, limit)) => {
                let value = if wants_limit {
                    limit as u64
                } else {
                    high_dword as u64 & AR_VISIBLE_MASK
                };
                self.write_gpr(i.reg, i.opsize, true, value);
                self.set_eflags_arith(flags | ZF);
            }
            None => self.set_eflags_arith(flags & !ZF),
        }
        self.rip = next;
        Ok(())
    }

    /// The upper descriptor dword and the scaled limit of `selector` when LAR/LSL accept it.
    fn descriptor_for_lar_lsl(
        &mut self,
        selector: u16,
        wants_limit: bool,
    ) -> ExecResult<Option<(u32, u32)>> {
        const LAR_TYPES: [u32; 8] = [1, 2, 3, 4, 5, 9, 0xb, 0xc];
        const LSL_TYPES: [u32; 5] = [1, 2, 3, 9, 0xb];
        if selector & !SELECTOR_RPL_MASK == 0 {
            return Ok(None);
        }
        let table_limit = if selector & SELECTOR_TABLE_INDICATOR != 0 {
            self.ldtr.limit
        } else {
            self.gdtr.limit as u32
        };
        if (selector & SELECTOR_OFFSET_MASK) as u64 + DESCRIPTOR_BYTES - 1 > table_limit as u64 {
            return Ok(None);
        }
        let (lo, _) = self.read_desc(selector)?;
        let high = (lo >> 32) as u32;
        let flags = high & SEG_DESCRIPTOR_FLAGS_MASK;
        let dpl = ((flags >> SEG_DPL_SHIFT) & SEG_DPL_MASK) as u8;
        let conforming_code = flags & SEG_NON_SYSTEM != 0
            && flags & SEG_TYPE_EXECUTABLE != 0
            && flags & SEG_TYPE_CONFORMING != 0;
        let privileged_enough =
            conforming_code || (dpl >= self.cpl && dpl >= (selector & SELECTOR_RPL_MASK) as u8);
        let type_field = (flags >> DESC_TYPE_SHIFT) & DESC_TYPE_MASK;
        let type_ok = flags & SEG_NON_SYSTEM != 0
            || if wants_limit {
                LSL_TYPES.contains(&type_field)
            } else {
                LAR_TYPES.contains(&type_field)
            };
        if !privileged_enough || !type_ok {
            return Ok(None);
        }
        let mut limit = (lo as u32 & 0xffff) | (high & DESC_LIMIT_HIGH_MASK);
        if high & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        Ok(Some((high, limit)))
    }

    /// 0F 00 (group 6: sldt, str, lldt, ltr, verr, verw) and 0F 01 (group 7: sgdt, sidt,
    /// lgdt, lidt, smsw, lmsw, invlpg, swapgs), selected by ModRM.reg.
    pub fn sys_group(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        const OP_GROUP6: u16 = 0x100;
        const MODRM_SWAPGS: u8 = 0xf8;
        const MODRM_XGETBV: u8 = 0xd0;
        const MODRM_XSETBV: u8 = 0xd1;
        const MODRM_RDTSCP: u8 = 0xf9;
        const MODRM_VMRUN: u8 = 0xd8;
        const MODRM_VMLOAD: u8 = 0xda;
        const MODRM_VMSAVE: u8 = 0xdb;
        const MODRM_STGI: u8 = 0xdc;
        const MODRM_CLGI: u8 = 0xdd;
        const MODRM_INVLPGA: u8 = 0xdf;
        const MODRM_MONITOR: u8 = 0xc8;
        const MODRM_MWAIT: u8 = 0xc9;
        const MODRM_CLAC: u8 = 0xca;
        const MODRM_STAC: u8 = 0xcb;
        const MODRM_CLZERO: u8 = 0xfc;
        let reg = i.reg & 7;
        if i.op == OP_GROUP6 {
            let loc = self.modrm_location(i, next)?;
            // a register destination takes the operand size, memory is always 16 bits
            let size = if i.md == 3 { i.opsize } else { 2 };
            match reg {
                // sldt
                0 => {
                    self.umip_check()?;
                    let v = self.ldtr.sel as u64;
                    self.store_operand(loc, size, i, v)?;
                }
                // str
                1 => {
                    self.umip_check()?;
                    let v = self.tr.sel as u64;
                    self.store_operand(loc, size, i, v)?;
                }
                2 | 3 => {
                    self.need_cpl0()?;
                    let sel = self.load_operand(loc, 2, i)? as u16;
                    if reg == 2 {
                        self.lldt(sel)?;
                    } else {
                        self.ltr(sel)?;
                    }
                }
                4 | 5 => {
                    let sel = self.load_operand(loc, 2, i)? as u16;
                    let accessible = self.segment_accessible(sel, reg == 5)?;
                    let rflags = self.eflags_full();
                    self.set_eflags_arith((rflags & !ZF) | if accessible { ZF } else { 0 });
                }
                _ => return Err(Exception::invalid_opcode()),
            }
            self.rip = next;
            return Ok(());
        }
        // register forms of group 7 are separate instructions, except smsw (/4) and lmsw (/6)
        if i.md == 3 && reg != 4 && reg != 6 {
            match i.modrm {
                MODRM_SWAPGS => {
                    self.need_cpl0()?;
                    core::mem::swap(&mut self.seg[SEG_GS].base, &mut self.kgsbase);
                }
                // SVM instructions set RIP themselves (vmrun loads the guest's)
                MODRM_VMRUN => return self.vmrun(next),
                MODRM_VMLOAD => return self.vmload_vmsave(false, next),
                MODRM_VMSAVE => return self.vmload_vmsave(true, next),
                MODRM_STGI => return self.stgi_clgi(true, next),
                MODRM_CLGI => return self.stgi_clgi(false, next),
                MODRM_INVLPGA => return self.invlpga(next),
                MODRM_XGETBV => self.xgetbv()?,
                MODRM_MONITOR => self.monitor()?,
                MODRM_MWAIT => self.mwait()?,
                // clac/stac: #UD outside CPL 0 (not #GP), unlike most privileged instructions
                MODRM_CLAC | MODRM_STAC => {
                    if self.cpl != 0 {
                        return Err(Exception::invalid_opcode());
                    }
                    if i.modrm == MODRM_STAC {
                        self.eflags |= AC;
                    } else {
                        self.eflags &= !AC;
                    }
                }
                MODRM_XSETBV => self.xsetbv()?,
                MODRM_CLZERO => self.clzero(i)?,
                // Same CR4.TSD rule as rdtsc; ECX = IA32_TSC_AUX (zero-extended), EDX:EAX = TSC.
                MODRM_RDTSCP => {
                    if self.cr4 & CR4_TSD != 0 && self.cpl != 0 {
                        return Err(Exception::general_protection(0));
                    }
                    let tsc = self.tsc();
                    self.gpr[RAX] = tsc & 0xffff_ffff;
                    self.gpr[RDX] = tsc >> 32;
                    self.gpr[RCX] = self.tsc_aux & 0xffff_ffff;
                }
                _ => return Err(self.unimplemented_opcode(i)),
            }
            self.rip = next;
            return Ok(());
        }
        let loc = self.modrm_location(i, next)?;
        match reg {
            // sgdt, sidt: 2-byte limit then 8-byte base
            0 | 1 => {
                let addr = match loc {
                    Loc::Mem(addr) => addr,
                    _ => return Err(Exception::invalid_opcode()),
                };
                self.umip_check()?;
                let table = if reg == 0 { self.gdtr } else { self.idtr };
                self.write(addr, 2, table.limit as u64)?;
                match self.table_register_base_bytes(i) {
                    8 => self.write(addr + 2, 8, table.base)?,
                    4 => self.write(addr + 2, 4, table.base)?,
                    // 16-bit operand size stores a 24-bit base, with the top byte zero
                    _ => self.write(addr + 2, 4, table.base & TABLE_BASE_24_MASK)?,
                }
            }
            // lgdt, lidt
            2 | 3 => {
                self.need_cpl0()?;
                let addr = match loc {
                    Loc::Mem(addr) => addr,
                    _ => return Err(Exception::invalid_opcode()),
                };
                let limit = self.read(addr, 2)? as u16;
                let base = match self.table_register_base_bytes(i) {
                    8 => self.read(addr + 2, 8)?,
                    4 => self.read(addr + 2, 4)?,
                    _ => self.read(addr + 2, 4)? & TABLE_BASE_24_MASK,
                };
                let table = DescriptorTable { base, limit };
                if reg == 2 {
                    self.gdtr = table;
                } else {
                    self.idtr = table;
                }
            }
            // smsw: the store truncates to the operand size, so a register destination of 32 or 64
            // bits receives CR0 itself
            4 => {
                self.umip_check()?;
                let size = if i.md == 3 { i.opsize } else { 2 };
                let v = self.cr0;
                self.store_operand(loc, size, i, v)?;
            }
            // lmsw loads MP, EM, TS and can set PE but never clear it
            6 => {
                const LMSW_MASK: u64 = CR0_MP | CR0_EM | CR0_TS;
                self.need_cpl0()?;
                let v = self.load_operand(loc, 2, i)?;
                self.cr0 =
                    (self.cr0 & !LMSW_MASK) | (v & LMSW_MASK) | (self.cr0 & CR0_PE) | (v & CR0_PE);
            }
            // invlpg
            7 => {
                self.need_cpl0()?;
                if let Loc::Mem(addr) = loc {
                    self.invlpg(addr);
                } else {
                    return Err(Exception::invalid_opcode());
                }
            }
            _ => return Err(Exception::invalid_opcode()),
        }
        self.rip = next;
        Ok(())
    }

    /// 0F 01 FC (AMD): zeroes the cache line holding the address in rAX (DS-relative, truncated to
    /// the address size). Allowed at any CPL; the line is aligned down, so it never crosses a
    /// page and one probe covers the whole write.
    fn clzero(&mut self, i: &Insn) -> ExecResult<()> {
        let offset = if i.addr32 {
            self.gpr[RAX] & 0xffff_ffff
        } else {
            self.gpr[RAX]
        };
        let line = self.linear_address(i, offset & !(CLZERO_LINE_BYTES - 1))?;
        self.translate(line, mem::AccessKind::Write, false)?;
        for qword in 0..CLZERO_LINE_BYTES / 8 {
            self.write(line + 8 * qword, 8, 0)?;
        }
        Ok(())
    }

    /// 0F 33: ECX selects a counter. Allowed at CPL > 0 only with CR4.PCE; a counter that does not
    /// exist is #GP(0). Linux's AMD PMU driver reads the counters it has enabled with this (also at
    /// shutdown, from `perf_reboot`), so it must exist whenever the counter MSRs do.
    pub(super) fn rdpmc(&mut self) -> ExecResult<()> {
        let counter = self.gpr[RCX] as u32;
        if (self.cr4 & CR4_PCE == 0 && self.cpl != 0) || counter >= K7_PERF_COUNTERS {
            return Err(Exception::general_protection(0));
        }
        let value = self.rdmsr(MSR_K7_PERFCTR0 + counter)? & PERF_COUNTER_MASK;
        self.set_edx_eax(value);
        Ok(())
    }

    /// F3 0F AE /0../3 with a register operand: rdfsbase, rdgsbase, wrfsbase, wrgsbase. They exist
    /// only while CR4.FSGSBASE is set (#UD otherwise) and work at any CPL. The operand is 32 bits
    /// without REX.W (a read zero-extends, a write zero-extends into the base) and 64 bits with it;
    /// writing a non-canonical 64-bit base is #GP(0), as for wrmsr.
    fn fs_gs_base(&mut self, i: &Insn) -> ExecResult<()> {
        const OP_READ_FS: u8 = 0;
        const OP_READ_GS: u8 = 1;
        const OP_WRITE_FS: u8 = 2;
        if self.cr4 & CR4_FSGSBASE == 0 {
            return Err(Exception::invalid_opcode());
        }
        let operation = i.reg & 7;
        let segment = if operation == OP_READ_FS || operation == OP_WRITE_FS {
            SEG_FS
        } else {
            SEG_GS
        };
        let size = if i.opsize == 8 { 8 } else { 4 };
        if operation == OP_READ_FS || operation == OP_READ_GS {
            let base = self.seg[segment].base;
            self.write_gpr(i.rm, size, true, base);
        } else {
            let base = self.read_gpr(i.rm, size, true);
            if !is_canonical(base) {
                return Err(Exception::general_protection(0));
            }
            self.seg[segment].base = base;
        }
        Ok(())
    }

    /// 0F AE: fxsave, fxrstor, ldmxcsr, stmxcsr, clflush (memory forms) and lfence/mfence/sfence
    /// (register forms /5../7, no-ops), and with an F3 prefix the register forms /0../3 that read
    /// and write the FS and GS bases.
    pub fn group_ae(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        const REP_REPE: u8 = 1;
        let reg = i.reg & 7;
        if i.md == 3 {
            if i.rep == REP_REPE && reg < 4 {
                self.fs_gs_base(i)?;
            } else if i.rep != 0 || !(5..=7).contains(&reg) {
                return Err(Exception::invalid_opcode());
            }
        } else {
            // ldmxcsr/stmxcsr obey the SSE enable rules (#UD for CR0.EM or CR4.OSFXSR clear, then
            // #NM for CR0.TS) before touching memory; fxsave/fxrstor have their own checks.
            if matches!(reg, 2 | 3) {
                self.sse_gate()?;
            }
            let addr = self.linear_address(i, self.effective_address(i, next))?;
            match reg {
                0 => self.fxsave(addr)?,
                1 => self.fxrstor(addr)?,
                // xsave, xrstor, xsaveopt (REX.W selects the 64-bit pointer layout, identical here)
                4 => self.xsave(addr)?,
                5 => self.xrstor(addr)?,
                // 66 0F AE /6 is clwb, not xsaveopt: a cache-line write back, which like clflush
                // only needs the access check here
                6 if i.p66 => {
                    self.read(addr, 1)?;
                }
                6 => self.xsave(addr)?,
                2 => {
                    let v = self.read(addr, 4)? as u32;
                    if v & !MXCSR_VALID_BITS != 0 {
                        return Err(Exception::general_protection(0));
                    }
                    self.mxcsr = v;
                }
                3 => {
                    let v = self.mxcsr as u64;
                    self.write(addr, 4, v)?;
                }
                // clflush: only the access check (and its fault) is performed
                7 => {
                    self.read(addr, 1)?;
                }
                _ => return Err(Exception::invalid_opcode()),
            }
        }
        self.rip = next;
        Ok(())
    }
}

#[cfg(test)]
mod sys_tests {
    use super::*;
    use crate::cpu::exception::{
        VECTOR_GENERAL_PROTECTION, VECTOR_INVALID_OPCODE, VECTOR_PAGE_FAULT,
    };
    use crate::testkit::{CODE_BASE, Harness, R8, R9, RDI, RSI, RSP, SCRATCH, Stop};

    /// Guest memory the snippets use: the descriptor table, then scratch data and an output area.
    const GDT: u64 = SCRATCH;
    const DATA: u64 = SCRATCH + 0x1000;
    const OUT: u64 = SCRATCH + 0x2000;
    /// A 2 MiB region of the boot identity map that tests may remap or unmap.
    const REMAP_REGION: u64 = 0xa0_0000;
    const OTHER_REGION: u64 = 0xc0_0000;

    const PTE_USER: u64 = 4;
    const PTE_PRESENT_WRITABLE_HUGE: u64 = 0x83;
    const PTE_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;

    const ACCESS_PRESENT: u8 = 0x80;

    const CR0_CD_BIT: u64 = 1 << 30;
    const CR0_NW_BIT: u64 = 1 << 29;
    const CR0_WP_BIT: u64 = 1 << 16;
    const CR0_AM_BIT: u64 = 1 << 18;
    const CR0_NE_BIT: u64 = 1 << 5;
    const CR0_ET_BIT: u64 = 1 << 4;
    const CR0_MP_BIT: u64 = 1 << 1;
    const CR0_PE_BIT: u64 = 1;
    const CR0_PG_BIT: u64 = 1 << 31;
    const CR0_EM_BIT: u64 = 1 << 2;
    const CR0_TS_BIT: u64 = 1 << 3;
    const BOOT_CR0: u64 = CR0_PG_BIT | CR0_NE_BIT | CR0_ET_BIT | CR0_MP_BIT | CR0_PE_BIT;
    const BOOT_CR4: u64 = 1 << 5;

    const EFER_LME_BIT: u64 = 1 << 8;
    const EFER_LMA_BIT: u64 = 1 << 10;
    const EFER_SCE_BIT: u64 = 1;
    const EFER_NXE_BIT: u64 = 1 << 11;

    const TEST_MSR_PAT: u32 = 0x277;
    const TEST_MSR_STAR: u32 = 0xc000_0081;
    const TEST_MSR_LSTAR: u32 = 0xc000_0082;
    const TEST_MSR_CSTAR: u32 = 0xc000_0083;
    const TEST_MSR_SFMASK: u32 = 0xc000_0084;
    const TEST_MSR_FS_BASE: u32 = 0xc000_0100;
    const TEST_MSR_GS_BASE: u32 = 0xc000_0101;
    const TEST_MSR_KERNEL_GS_BASE: u32 = 0xc000_0102;
    const TEST_MSR_TSC_AUX: u32 = 0xc000_0103;
    const TEST_MSR_K7_PERFCTR0: u32 = 0xc001_0004;
    const TEST_MSR_SYSENTER_CS: u32 = 0x174;
    const TEST_MSR_SYSENTER_ESP: u32 = 0x175;
    const TEST_MSR_SYSENTER_EIP: u32 = 0x176;
    const TEST_MSR_APIC_BASE: u32 = 0x1b;
    const TEST_MSR_EFER: u32 = 0xc000_0080;
    const TEST_MSR_TSC: u32 = 0x10;

    fn fault(vector: u8, error_code: Option<u32>) -> Stop {
        Stop::Fault {
            vector,
            error_code,
            software: false,
        }
    }

    fn general_protection(error_code: u32) -> Stop {
        fault(VECTOR_GENERAL_PROTECTION, Some(error_code))
    }

    fn invalid_opcode() -> Stop {
        fault(VECTOR_INVALID_OPCODE, None)
    }

    fn not_present(error_code: u32) -> Stop {
        fault(
            crate::cpu::exception::VECTOR_SEGMENT_NOT_PRESENT,
            Some(error_code),
        )
    }

    fn run_with(code: &[u8], setup: impl FnOnce(&mut Harness)) -> (Harness, Stop) {
        let mut harness = Harness::new(code);
        setup(&mut harness);
        let stop = harness.run(1000);
        (harness, stop)
    }

    /// Runs to `hlt`; any other outcome fails with the stop reason.
    #[track_caller]
    fn run_ok(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
        let (harness, stop) = run_with(code, setup);
        assert_eq!(stop, Stop::Halted, "{}", harness.cpu.dump_regs());
        harness
    }

    /// Runs `code` (whose first instruction is the one under test) and expects exactly `expected`
    /// with `rip` left on that instruction.
    #[track_caller]
    fn assert_faults(code: &[u8], setup: impl FnOnce(&mut Harness), expected: Stop) -> Harness {
        let (harness, stop) = run_with(code, setup);
        assert_eq!(stop, expected, "{}", harness.cpu.dump_regs());
        assert_eq!(
            harness.cpu.rip, CODE_BASE,
            "a fault must leave rip on the instruction"
        );
        harness
    }

    #[track_caller]
    fn assert_general_protection_zero(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
        assert_faults(code, setup, general_protection(0))
    }

    #[track_caller]
    fn assert_invalid_opcode(code: &[u8], setup: impl FnOnce(&mut Harness)) -> Harness {
        assert_faults(code, setup, invalid_opcode())
    }

    fn fault_parts<T: std::fmt::Debug>(result: ExecResult<T>) -> (u8, Option<u32>) {
        match result {
            Err(Exception::Fault {
                vector, error_code, ..
            }) => (vector, error_code),
            other => panic!("expected a fault, got {other:?}"),
        }
    }

    #[track_caller]
    fn assert_gp0<T: std::fmt::Debug>(result: ExecResult<T>) {
        assert_eq!(fault_parts(result), (VECTOR_GENERAL_PROTECTION, Some(0)));
    }

    fn pd_entry_address(harness: &Harness, linear: u64) -> u64 {
        let read = |address: u64| harness.read_u64(address) & PTE_ADDRESS_MASK;
        let pml4 = harness.cpu.cr3 & PTE_ADDRESS_MASK;
        let pdpt = read(pml4 + ((linear >> 39) & 0x1ff) * 8);
        let pd = read(pdpt + ((linear >> 30) & 0x1ff) * 8);
        pd + ((linear >> 21) & 0x1ff) * 8
    }

    /// Makes the 2 MiB region containing `linear` not present, so accesses there page-fault.
    fn unmap_region(harness: &mut Harness, linear: u64) {
        let entry = pd_entry_address(harness, linear);
        harness.write_u64(entry, 0);
        harness.cpu.flush_tlb();
    }

    /// Marks the whole boot identity map user-accessible, so the harness can fetch and touch memory
    /// at CPL 3, then switches to CPL 3.
    fn user_mode(harness: &mut Harness) {
        let pml4 = harness.cpu.cr3 & PTE_ADDRESS_MASK;
        let pml4_entry = harness.read_u64(pml4);
        harness.write_u64(pml4, pml4_entry | PTE_USER);
        let pdpt = pml4_entry & PTE_ADDRESS_MASK;
        for gib in 0..4 {
            let pdpt_entry = harness.read_u64(pdpt + gib * 8);
            harness.write_u64(pdpt + gib * 8, pdpt_entry | PTE_USER);
            let pd = pdpt_entry & PTE_ADDRESS_MASK;
            for slot in 0..512 {
                let entry = harness.read_u64(pd + slot * 8);
                harness.write_u64(pd + slot * 8, entry | PTE_USER);
            }
        }
        harness.cpu.flush_tlb();
        harness.cpu.set_cpl(3);
    }

    fn descriptor(base: u32, limit: u32, access: u8, flags: u8) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..2].copy_from_slice(&(limit as u16).to_le_bytes());
        bytes[2..4].copy_from_slice(&(base as u16).to_le_bytes());
        bytes[4] = (base >> 16) as u8;
        bytes[5] = access;
        bytes[6] = (flags << 4) | ((limit >> 16) as u8 & 0xf);
        bytes[7] = (base >> 24) as u8;
        bytes
    }

    fn system_descriptor(base: u64, limit: u32, access: u8, flags: u8) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&descriptor(base as u32, limit, access, flags));
        bytes[8..12].copy_from_slice(&((base >> 32) as u32).to_le_bytes());
        bytes
    }

    const SEL_LDT: u16 = 0x08;
    const SEL_TSS: u16 = 0x18;
    const SEL_TSS_BUSY: u16 = 0x28;
    const SEL_TSS_ABSENT: u16 = 0x38;
    const SEL_LDT_ABSENT: u16 = 0x48;
    const SEL_DATA_RW_DPL0: u16 = 0x58;
    const SEL_DATA_RW_DPL3: u16 = 0x60;
    const SEL_DATA_RO_DPL0: u16 = 0x68;
    const SEL_CODE_X_DPL0: u16 = 0x70;
    const SEL_CODE_XR_DPL0: u16 = 0x78;
    const SEL_CODE_XR_CONFORMING_DPL0: u16 = 0x80;
    const SEL_CODE_XR_DPL3: u16 = 0x88;
    const SEL_LDT_GRANULAR: u16 = 0x90;
    const SEL_TSS_HIGH_BASE: u16 = 0xa0;
    const GDT_LIMIT: u16 = 0xaf;

    const LDT_BASE: u64 = 0x1234_5000;
    const LDT_LIMIT: u32 = 0xff;
    const TSS_BASE: u64 = 0xffff_8000_1234_5000;
    const TSS_LIMIT: u32 = 0x67;
    const ACCESS_LDT: u8 = 0x82;
    const ACCESS_TSS_AVAILABLE: u8 = 0x89;
    const ACCESS_TSS_BUSY: u8 = 0x8b;
    const GRANULARITY_FLAG: u8 = 8;

    fn install_gdt(harness: &mut Harness) {
        let entries: [(u16, Vec<u8>); 15] = [
            (
                SEL_LDT,
                system_descriptor(LDT_BASE, LDT_LIMIT, ACCESS_LDT, 0).to_vec(),
            ),
            (
                SEL_TSS,
                system_descriptor(TSS_BASE, TSS_LIMIT, ACCESS_TSS_AVAILABLE, 0).to_vec(),
            ),
            (
                SEL_TSS_BUSY,
                system_descriptor(TSS_BASE, TSS_LIMIT, ACCESS_TSS_BUSY, 0).to_vec(),
            ),
            (
                SEL_TSS_ABSENT,
                system_descriptor(
                    TSS_BASE,
                    TSS_LIMIT,
                    ACCESS_TSS_AVAILABLE & !ACCESS_PRESENT,
                    0,
                )
                .to_vec(),
            ),
            (
                SEL_LDT_ABSENT,
                system_descriptor(LDT_BASE, LDT_LIMIT, ACCESS_LDT & !ACCESS_PRESENT, 0).to_vec(),
            ),
            (SEL_DATA_RW_DPL0, descriptor(0, 0xfffff, 0x92, 0xc).to_vec()),
            (SEL_DATA_RW_DPL3, descriptor(0, 0xfffff, 0xf2, 0xc).to_vec()),
            (SEL_DATA_RO_DPL0, descriptor(0, 0xfffff, 0x90, 0xc).to_vec()),
            (SEL_CODE_X_DPL0, descriptor(0, 0xfffff, 0x98, 0xa).to_vec()),
            (SEL_CODE_XR_DPL0, descriptor(0, 0xfffff, 0x9a, 0xa).to_vec()),
            (
                SEL_CODE_XR_CONFORMING_DPL0,
                descriptor(0, 0xfffff, 0x9e, 0xa).to_vec(),
            ),
            (SEL_CODE_XR_DPL3, descriptor(0, 0xfffff, 0xfa, 0xa).to_vec()),
            (
                SEL_LDT_GRANULAR,
                system_descriptor(LDT_BASE, 0xf, ACCESS_LDT, GRANULARITY_FLAG).to_vec(),
            ),
            (
                SEL_TSS_HIGH_BASE,
                system_descriptor(TSS_BASE, TSS_LIMIT, ACCESS_TSS_AVAILABLE, 0).to_vec(),
            ),
            (0x00, vec![0; 8]),
        ];
        for (selector, bytes) in entries {
            harness.write(GDT + selector as u64, &bytes);
        }
        harness.cpu.gdtr = DescriptorTable {
            base: GDT,
            limit: GDT_LIMIT,
        };
    }

    fn cpuid(harness: &mut Harness, leaf: u32, subleaf: u32) -> [u32; 4] {
        harness.cpu.gpr[RAX] = 0xdead_beef_0000_0000 | leaf as u64;
        harness.cpu.gpr[RCX] = 0xcafe_f00d_0000_0000 | subleaf as u64;
        harness.cpu.gpr[RBX] = u64::MAX;
        harness.cpu.gpr[RDX] = u64::MAX;
        harness.cpu.cpuid();
        let registers = [RAX, RBX, RCX, RDX].map(|index| harness.cpu.gpr[index]);
        for value in registers {
            assert_eq!(
                value >> 32,
                0,
                "cpuid leaf {leaf:#x} must zero-extend into 64-bit registers"
            );
        }
        registers.map(|value| value as u32)
    }

    fn bits(positions: &[u32]) -> u32 {
        positions
            .iter()
            .fold(0, |word, position| word | 1 << position)
    }

    fn blank() -> Harness {
        Harness::new(&[crate::testkit::HLT])
    }

    mod cpuid_instruction {
        use super::*;

        #[test]
        fn leaf_1_advertises_mmx_and_the_register_forms_execute() {
            // glibc's baseline ISA level needs the bit: clearing it stops every Debian binary at
            // startup, so the instructions behind it must work (full coverage is in the tests in sse.rs).
            let mut h = blank();
            let [_, _, _, edx] = cpuid(&mut h, 1, 0);
            assert_ne!(edx & (1 << 23), 0);
            let [_, _, _, extended_edx] = cpuid(&mut h, 0x8000_0001, 0);
            assert_ne!(extended_edx & (1 << 23), 0, "the AMD alias of MMX");
            // movq mm0, mm1; paddb mm0, mm1; movd eax, mm0; hlt
            let h = run_ok(
                &[0x0f, 0x6f, 0xc1, 0x0f, 0xfc, 0xc1, 0x0f, 0x7e, 0xc0, 0xf4],
                |h| {
                    h.cpu.st[1][..8].copy_from_slice(&0x0000_0000_0102_0304u64.to_le_bytes());
                },
            );
            assert_eq!(h.cpu.gpr[RAX], 0x0204_0608, "mm0 = mm1 + mm1, bytewise");
        }

        #[test]
        fn prefetchw_is_a_hint_that_never_faults() {
            // prefetchw [rax]; prefetch [rax]; hlt, with rax pointing at nothing
            let mut h = Harness::new(&[0x0f, 0x0d, 0x08, 0x0f, 0x0d, 0x00, 0xf4]);
            h.cpu.gpr[RAX] = 0x1234_5678_9000;
            assert_eq!(h.run(10), Stop::Halted);
        }

        #[test]
        fn cpuid_instruction_fills_eax_ebx_ecx_edx() {
            // cpuid; hlt
            let h = run_ok(&[0x0f, 0xa2, 0xf4], |h| h.cpu.gpr[RAX] = 0);
            assert_eq!(h.cpu.gpr[RAX], 0x10);
            assert_eq!(h.cpu.gpr[RBX], u32::from_le_bytes(*b"Auth") as u64);
            assert_eq!(h.cpu.gpr[RDX], u32::from_le_bytes(*b"enti") as u64);
            assert_eq!(h.cpu.gpr[RCX], u32::from_le_bytes(*b"cAMD") as u64);
        }

        #[test]
        fn cpuid_is_unprivileged() {
            // cpuid; hlt
            let (h, stop) = run_with(&[0x0f, 0xa2, 0xf4], |h| {
                user_mode(h);
                h.cpu.gpr[RAX] = 1;
            });
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: 13,
                    error_code: Some(0),
                    software: false
                },
                "hlt at CPL 3 is the only fault"
            );
            assert_eq!(h.cpu.rip, CODE_BASE + 2);
            assert_eq!(h.cpu.gpr[RAX], 0x00b0_0f21);
        }

        #[test]
        fn jit_and_interpreter_agree_on_cpuid() {
            for leaf in [
                0u64,
                1,
                7,
                0xb,
                0xd,
                0x8000_0001,
                0x8000_001d,
                0x8000_001e,
                0x8000_0026,
            ] {
                // cpuid; hlt
                crate::testkit::assert_jit_matches_interpreter(&[0x0f, 0xa2, 0xf4], |h| {
                    h.cpu.gpr[RAX] = leaf
                });
            }
        }
    }

    mod msrs {
        use super::*;

        #[test]
        fn msr_fault_log_keys_never_collide() {
            // Sampled MSR numbers including both extremes of the 32-bit space and values that used to
            // make the two bases overlap (0x5000_0000 + msr reaching 0x6000_0000 + other).
            let msrs = [
                0u64,
                1,
                0x277,
                0x1000_0000,
                0x4000_0000,
                0x4fff_ffff,
                0x5000_0000,
                0x1000_0000 + 0x277,
                0xc000_0080,
                0xffff_fffe,
                0xffff_ffff,
            ];
            let mut seen = std::collections::HashSet::new();
            for msr in msrs {
                for base in [LOG_KEY_RDMSR_FAULT, LOG_KEY_WRMSR_FAULT] {
                    assert!(seen.insert(base + msr), "key {:#x} repeats", base + msr);
                }
            }
            // Both ranges must also stay clear of the unmapped-access keys in mem.rs (bits 62 and 61)
            // and the unimplemented-opcode keys in exec.rs (0x4000_0000 plus a small number).
            for base in [LOG_KEY_RDMSR_FAULT, LOG_KEY_WRMSR_FAULT] {
                assert!(base + 0xffff_ffff < 1 << 61);
                assert!(base > 0x4000_0000 + 0x10_000);
            }
            const { assert!(LOG_KEY_WRMSR_FAULT + 0xffff_ffff < LOG_KEY_RDMSR_FAULT) };
        }

        const NON_CANONICAL: [u64; 4] = [
            0x0000_8000_0000_0000,
            0x0001_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            0xffff_7fff_ffff_ffff,
        ];
        const CANONICAL: [u64; 5] = [
            0,
            0x0000_7fff_ffff_ffff,
            0xffff_8000_0000_0000,
            0xffff_ffff_ffff_ffff,
            0xffff_ffff_8123_4567,
        ];
        const ADDRESS_MSRS: [u32; 7] = [
            TEST_MSR_FS_BASE,
            TEST_MSR_GS_BASE,
            TEST_MSR_KERNEL_GS_BASE,
            TEST_MSR_LSTAR,
            TEST_MSR_CSTAR,
            TEST_MSR_SYSENTER_EIP,
            TEST_MSR_SYSENTER_ESP,
        ];

        #[test]
        fn pat_round_trips_valid_memory_types_and_rejects_reserved_ones() {
            const VALID_PAT: u64 = 0x0106_0504_0007_0600;
            let mut h = blank();
            h.cpu.wrmsr(TEST_MSR_PAT, VALID_PAT).unwrap();
            assert_eq!(h.cpu.rdmsr(TEST_MSR_PAT).unwrap(), VALID_PAT);
            for invalid in [2, 3, 8, 3 << 56, 0x0706_0504_0302_0100] {
                assert_gp0(h.cpu.wrmsr(TEST_MSR_PAT, invalid));
            }
            assert_eq!(h.cpu.rdmsr(TEST_MSR_PAT).unwrap(), VALID_PAT);
        }

        #[test]
        fn plain_registers_round_trip_through_wrmsr_and_rdmsr() {
            let mut h = blank();
            for msr in [
                TEST_MSR_STAR,
                TEST_MSR_SFMASK,
                TEST_MSR_TSC_AUX,
                TEST_MSR_SYSENTER_CS,
            ] {
                h.cpu.wrmsr(msr, 0x1122_3344_5566_7788).unwrap();
                assert_eq!(
                    h.cpu.rdmsr(msr).unwrap(),
                    0x1122_3344_5566_7788,
                    "msr {msr:#x}"
                );
            }
        }

        #[test]
        fn address_registers_round_trip_canonical_values() {
            let mut h = blank();
            for msr in ADDRESS_MSRS {
                for value in CANONICAL {
                    h.cpu
                        .wrmsr(msr, value)
                        .unwrap_or_else(|_| panic!("msr {msr:#x} rejected {value:#x}"));
                    assert_eq!(h.cpu.rdmsr(msr).unwrap(), value, "msr {msr:#x}");
                }
            }
        }

        #[test]
        fn fs_and_gs_base_live_in_the_segment_cache() {
            let mut h = blank();
            h.cpu
                .wrmsr(TEST_MSR_FS_BASE, 0xffff_8000_0000_1000)
                .unwrap();
            h.cpu.wrmsr(TEST_MSR_GS_BASE, 0x2000).unwrap();
            assert_eq!(h.cpu.seg[SEG_FS].base, 0xffff_8000_0000_1000);
            assert_eq!(h.cpu.seg[SEG_GS].base, 0x2000);
            assert_eq!(
                h.cpu.rdmsr(TEST_MSR_FS_BASE).unwrap(),
                0xffff_8000_0000_1000
            );
            assert_eq!(h.cpu.rdmsr(TEST_MSR_GS_BASE).unwrap(), 0x2000);
        }

        #[test]
        fn non_canonical_values_raise_gp_for_every_address_msr_and_change_nothing() {
            for msr in ADDRESS_MSRS {
                for value in NON_CANONICAL {
                    let mut h = blank();
                    let before = h.cpu.rdmsr(msr).unwrap();
                    assert_gp0(h.cpu.wrmsr(msr, value));
                    assert_eq!(
                        h.cpu.rdmsr(msr).unwrap(),
                        before,
                        "msr {msr:#x} after a faulting write of {value:#x}"
                    );
                }
            }
        }

        #[test]
        fn non_address_msrs_accept_any_value() {
            let mut h = blank();
            for msr in [
                TEST_MSR_STAR,
                TEST_MSR_SFMASK,
                TEST_MSR_TSC_AUX,
                TEST_MSR_SYSENTER_CS,
            ] {
                for value in NON_CANONICAL {
                    h.cpu.wrmsr(msr, value).unwrap();
                }
            }
        }

        #[test]
        fn wrmsr_instruction_faults_on_a_non_canonical_lstar() {
            // wrmsr; hlt
            let h = assert_general_protection_zero(&[0x0f, 0x30, 0xf4], |h| {
                h.cpu.gpr[RCX] = TEST_MSR_LSTAR as u64;
                h.cpu.gpr[RAX] = 0;
                h.cpu.gpr[RDX] = 0x0001_0000;
            });
            assert_eq!(h.cpu.lstar, 0);
        }

        #[test]
        fn swapgs_exchanges_the_gs_base_with_kernel_gs_base() {
            // swapgs; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf8, 0xf4], |h| {
                h.cpu.seg[SEG_GS].base = 0x1111;
                h.cpu.kgsbase = 0x2222;
            });
            assert_eq!(h.cpu.seg[SEG_GS].base, 0x2222);
            assert_eq!(h.cpu.kgsbase, 0x1111);
        }

        #[test]
        fn swapgs_twice_restores_both_bases() {
            // swapgs; swapgs; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf8, 0x0f, 0x01, 0xf8, 0xf4], |h| {
                h.cpu.seg[SEG_GS].base = 0x1111;
                h.cpu.kgsbase = 0x2222;
            });
            assert_eq!((h.cpu.seg[SEG_GS].base, h.cpu.kgsbase), (0x1111, 0x2222));
        }

        #[test]
        fn swapgs_at_cpl3_is_gp0_and_swaps_nothing() {
            // swapgs; hlt
            let h = assert_general_protection_zero(&[0x0f, 0x01, 0xf8, 0xf4], |h| {
                user_mode(h);
                h.cpu.seg[SEG_GS].base = 0x1111;
                h.cpu.kgsbase = 0x2222;
            });
            assert_eq!((h.cpu.seg[SEG_GS].base, h.cpu.kgsbase), (0x1111, 0x2222));
        }

        #[test]
        fn swapgs_leaves_the_gs_selector_alone() {
            // swapgs; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf8, 0xf4], |h| h.cpu.seg[SEG_GS].sel = 0x2b);
            assert_eq!(h.cpu.seg[SEG_GS].sel, 0x2b);
        }

        #[test]
        fn sysenter_registers_follow_the_msr_numbers() {
            let mut h = blank();
            h.cpu.wrmsr(TEST_MSR_SYSENTER_CS, 8).unwrap();
            h.cpu.wrmsr(TEST_MSR_SYSENTER_ESP, 0x1000).unwrap();
            h.cpu.wrmsr(TEST_MSR_SYSENTER_EIP, 0x2000).unwrap();
            assert_eq!(h.cpu.sysenter, [8, 0x1000, 0x2000]);
        }

        #[test]
        fn pat_resets_to_the_power_on_value() {
            let mut h = blank();
            // PA0=WB(6) PA1=WT(4) PA2=UC-(7) PA3=UC(0), repeated for PA4..PA7.
            assert_eq!(h.cpu.rdmsr(TEST_MSR_PAT).unwrap(), 0x0007_0406_0007_0406);
        }

        #[test]
        fn tsc_msr_follows_the_virtual_clock_and_ignores_writes() {
            let mut h = blank();
            h.cpu.icount = 1000;
            assert_eq!(h.cpu.rdmsr(TEST_MSR_TSC).unwrap(), 1000 * NS_PER_INSN);
            h.cpu.wrmsr(TEST_MSR_TSC, 5).unwrap();
            assert_eq!(h.cpu.rdmsr(TEST_MSR_TSC).unwrap(), 1000 * NS_PER_INSN);
        }

        #[test]
        fn apic_base_reads_the_default_enabled_bsp_state() {
            let mut h = blank();
            assert_eq!(h.cpu.rdmsr(TEST_MSR_APIC_BASE).unwrap(), 0xfee0_0900);
        }

        #[test]
        fn apic_base_rejects_x2apic_without_enable_and_reserved_bits() {
            let mut h = blank();
            assert_gp0(h.cpu.wrmsr(TEST_MSR_APIC_BASE, 0xfee0_0400));
            assert_gp0(h.cpu.wrmsr(TEST_MSR_APIC_BASE, 0xfee0_0900 | (1 << 62)));
        }

        #[test]
        fn x2apic_registers_fault_until_x2apic_mode_is_enabled() {
            const X2APIC_VERSION: u32 = 0x803;
            let mut h = blank();
            assert_gp0(h.cpu.rdmsr(X2APIC_VERSION));
            h.cpu.wrmsr(TEST_MSR_APIC_BASE, 0xfee0_0d00).unwrap();
            assert_eq!(h.cpu.rdmsr(X2APIC_VERSION).unwrap(), 0x0006_0014);
        }

        #[test]
        fn x2apic_rejects_a_write_with_upper_bits_and_reserved_registers() {
            let mut h = blank();
            h.cpu.wrmsr(TEST_MSR_APIC_BASE, 0xfee0_0d00).unwrap();
            assert_gp0(h.cpu.wrmsr(0x80b, 1 << 32));
            assert_gp0(h.cpu.wrmsr(0x803, 0));
            assert_gp0(h.cpu.rdmsr(0x8ff));
        }

        #[test]
        fn tsc_deadline_msr_round_trips() {
            const MSR_TSC_DEADLINE: u32 = 0x6e0;
            let mut h = blank();
            h.cpu.wrmsr(MSR_TSC_DEADLINE, 12345).unwrap();
            assert_eq!(h.cpu.rdmsr(MSR_TSC_DEADLINE).unwrap(), 12345);
        }

        #[test]
        fn mtrr_msrs_validate_memory_types() {
            let mut h = blank();
            assert_eq!(h.cpu.rdmsr(0xfe).unwrap(), 0x508);
            h.cpu.wrmsr(0x2ff, 0xc06).unwrap();
            assert_eq!(h.cpu.rdmsr(0x2ff).unwrap(), 0xc06);
            assert_gp0(h.cpu.wrmsr(0x2ff, 2));
            assert_gp0(h.cpu.wrmsr(0x250, 0x0302));
            assert_gp0(h.cpu.wrmsr(0xfe, 0));
        }

        #[test]
        fn machine_check_banks_accept_initialization_and_never_report() {
            let mut h = blank();
            assert_eq!(h.cpu.rdmsr(0x179).unwrap() & 0xff, 6);
            h.cpu.wrmsr(0x17b, u64::MAX).unwrap();
            h.cpu.wrmsr(0x400, u64::MAX).unwrap();
            h.cpu.wrmsr(0x401, 0).unwrap();
            assert_eq!(h.cpu.rdmsr(0x401).unwrap(), 0);
        }

        #[test]
        fn amd_memory_typing_msrs_report_the_top_of_ram() {
            let mut h = blank();
            let ram = h.cpu.ram.len() as u64;
            assert_eq!(h.cpu.rdmsr(0xc001_001a).unwrap(), ram);
            assert_eq!(h.cpu.rdmsr(0xc001_001d).unwrap(), 0);
        }

        #[test]
        fn unknown_msrs_raise_gp0_on_read_and_write() {
            let mut h = blank();
            for msr in [
                0x0,
                0x1,
                0x8c,
                0x4b56_4d00,
                0xc000_0200,
                0xdead_beef,
                0x4000_0000,
                0xffff_ffff,
            ] {
                assert_gp0(h.cpu.rdmsr(msr));
                assert_gp0(h.cpu.wrmsr(msr, 1));
            }
        }

        #[test]
        fn svm_msrs_and_efer_svme() {
            let mut h = blank();
            // EFER.SVME is writable and preserved
            let efer = h.cpu.rdmsr(TEST_MSR_EFER).unwrap();
            h.cpu.wrmsr(TEST_MSR_EFER, efer | (1 << 12)).unwrap();
            assert_ne!(h.cpu.efer & (1 << 12), 0);
            // VM_CR takes its five defined bits; VM_HSAVE_PA needs a 4 KiB aligned physical address
            h.cpu.wrmsr(0xc001_0114, 0b1_1111).unwrap();
            assert_eq!(h.cpu.rdmsr(0xc001_0114).unwrap(), 0b1_1111);
            assert_gp0(h.cpu.wrmsr(0xc001_0114, 1 << 5));
            h.cpu.wrmsr(0xc001_0117, 0x20_0000).unwrap();
            assert_eq!(h.cpu.rdmsr(0xc001_0117).unwrap(), 0x20_0000);
            assert_gp0(h.cpu.wrmsr(0xc001_0117, 0x20_0008));
            assert_gp0(h.cpu.wrmsr(0xc001_0117, 1 << 45));
            // with VM_CR.SVMDIS set (firmware disabled SVM) EFER.SVME cannot be turned on
            let mut h = blank();
            h.cpu.wrmsr(0xc001_0114, 1 << 4).unwrap();
            let efer = h.cpu.rdmsr(TEST_MSR_EFER).unwrap();
            assert_gp0(h.cpu.wrmsr(TEST_MSR_EFER, efer | (1 << 12)));
        }

        #[test]
        fn spec_ctrl_stores_ibrs_stibp_ssbd_and_rejects_reserved_bits() {
            let mut h = blank();
            assert_eq!(h.cpu.rdmsr(MSR_SPEC_CTRL).unwrap(), 0);
            h.cpu.wrmsr(MSR_SPEC_CTRL, 0b111).unwrap();
            assert_eq!(h.cpu.rdmsr(MSR_SPEC_CTRL).unwrap(), 0b111);
            assert_gp0(h.cpu.wrmsr(MSR_SPEC_CTRL, 1 << 3));
            assert_eq!(h.cpu.rdmsr(MSR_SPEC_CTRL).unwrap(), 0b111);
        }

        #[test]
        fn command_msrs_are_write_only_and_take_bit_zero() {
            let mut h = blank();
            for msr in [MSR_PRED_CMD, MSR_FLUSH_CMD] {
                h.cpu.wrmsr(msr, 1).unwrap();
                h.cpu.wrmsr(msr, 0).unwrap();
                assert_gp0(h.cpu.wrmsr(msr, 2));
                assert_gp0(h.cpu.rdmsr(msr));
            }
        }

        #[test]
        fn arch_capabilities_claims_immunity_and_is_read_only() {
            let mut h = blank();
            let value = h.cpu.rdmsr(MSR_ARCH_CAPABILITIES).unwrap();
            // RSBA (bit 2) marks the CPU as vulnerable and must stay clear.
            assert_eq!(value & (1 << 2), 0);
            for bit in [
                ARCH_CAP_RDCL_NO,
                ARCH_CAP_IBRS_ALL,
                ARCH_CAP_SSB_NO,
                ARCH_CAP_MDS_NO,
                ARCH_CAP_BHI_NO,
            ] {
                assert_ne!(value & bit, 0);
            }
            assert_gp0(h.cpu.wrmsr(MSR_ARCH_CAPABILITIES, 0));
        }

        #[test]
        fn speculation_control_cpuid_bits_match_the_msrs() {
            let mut h = blank();
            let [_, _, _, edx] = cpuid(&mut h, 7, 0);
            // FSRM, SPEC_CTRL, STIBP, L1D_FLUSH, ARCH_CAPABILITIES, SSBD
            assert_eq!(edx, bits(&[4, 26, 27, 28, 29, 31]));
            let [_, ebx, _, _] = cpuid(&mut h, 0x8000_0008, 0);
            // IBPB, IBRS, STIBP and SSBD in the AMD spelling
            for bit in [12, 14, 15, 24] {
                assert_ne!(ebx & (1 << bit), 0, "bit {bit}");
            }
        }

        /// Linux's `VULNBL_AMD` names families 0x15-0x17 for RETBLEED (a return thunk on every
        /// `ret`) and 0x19/0x1a for SRSO; the model has family 0x1a, so it must also say SRSO_NO.
        #[test]
        fn zen_5_says_it_has_neither_retbleed_nor_srso() {
            let mut h = blank();
            let [eax, _, _, _] = cpuid(&mut h, 1, 0);
            let family = ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff);
            assert_eq!(family, 0x1a);
            let [srso_no, ..] = cpuid(&mut h, 0x8000_0021, 0);
            assert_ne!(srso_no & (1 << 29), 0);
        }

        #[test]
        fn stored_only_msrs_read_back_what_was_written() {
            let mut h = blank();
            let keys: Vec<u32> = h.cpu.msr_misc.keys().copied().collect();
            assert!(!keys.is_empty());
            for key in keys {
                h.cpu.wrmsr(key, 0xabcd).unwrap();
                assert_eq!(h.cpu.rdmsr(key).unwrap(), 0xabcd);
            }
        }

        #[test]
        fn a_faulting_unknown_msr_access_is_logged_once_per_direction() {
            let mut h = blank();
            let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
            let sink = log.clone();
            h.cpu.log = Some(std::sync::Arc::new(move |message: &str| {
                sink.lock().unwrap().push(message.to_string())
            }));
            for _ in 0..3 {
                let _ = h.cpu.rdmsr(0xdead_beef);
            }
            for _ in 0..3 {
                let _ = h.cpu.wrmsr(0xdead_beef, 1);
            }
            let messages = log.lock().unwrap();
            assert_eq!(
                messages.len(),
                2,
                "one rdmsr line and one wrmsr line: {messages:?}"
            );
            assert!(messages[0].contains("rdmsr") && messages[0].contains("0xdeadbeef"));
            assert!(messages[1].contains("wrmsr"));
        }

        #[test]
        fn efer_stores_sce_lme_and_nxe_only() {
            let mut h = blank();
            h.cpu.efer = 0;
            h.cpu.cr0 &= !CR0_PG;
            h.cpu
                .wrmsr(TEST_MSR_EFER, EFER_SCE_BIT | EFER_LME_BIT | EFER_NXE_BIT)
                .unwrap();
            assert_eq!(h.cpu.efer, EFER_SCE_BIT | EFER_LME_BIT | EFER_NXE_BIT);
            h.cpu.wrmsr(TEST_MSR_EFER, 0).unwrap();
            assert_eq!(h.cpu.efer, 0);
        }

        #[test]
        fn efer_lma_in_a_write_is_accepted_but_not_stored() {
            let mut h = blank();
            h.cpu.efer = EFER_LME_BIT;
            h.cpu
                .wrmsr(TEST_MSR_EFER, EFER_LME_BIT | EFER_LMA_BIT)
                .unwrap();
            assert_eq!(
                h.cpu.efer & EFER_LMA_BIT,
                0,
                "LMA is derived from CR0.PG and LME"
            );
            h.cpu.efer = EFER_LME_BIT | EFER_LMA_BIT;
            h.cpu.wrmsr(TEST_MSR_EFER, EFER_LME_BIT).unwrap();
            assert_eq!(
                h.cpu.efer,
                EFER_LME_BIT | EFER_LMA_BIT,
                "a write cannot clear LMA either"
            );
        }

        #[test]
        fn efer_reserved_bits_raise_gp0() {
            let mut h = blank();
            for bit in [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 15, 32, 63] {
                let before = h.cpu.efer;
                assert_gp0(h.cpu.wrmsr(TEST_MSR_EFER, 1u64 << bit));
                assert_eq!(h.cpu.efer, before, "bit {bit}");
            }
        }

        #[test]
        fn rdmsr_instruction_splits_the_value_into_edx_eax() {
            // rdmsr; hlt
            let h = run_ok(&[0x0f, 0x32, 0xf4], |h| {
                h.cpu.pat = 0x1122_3344_5566_7788;
                h.cpu.gpr[RCX] = TEST_MSR_PAT as u64;
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
            });
            assert_eq!(h.cpu.gpr[RAX], 0x5566_7788);
            assert_eq!(h.cpu.gpr[RDX], 0x1122_3344);
        }

        #[test]
        fn wrmsr_instruction_combines_edx_eax_and_ignores_the_upper_register_halves() {
            // wrmsr; hlt
            let h = run_ok(&[0x0f, 0x30, 0xf4], |h| {
                h.cpu.gpr[RCX] = TEST_MSR_SFMASK as u64;
                h.cpu.gpr[RAX] = 0xffff_ffff_5566_7788;
                h.cpu.gpr[RDX] = 0xaaaa_aaaa_1122_3344;
            });
            assert_eq!(h.cpu.sfmask, 0x1122_3344_5566_7788);
        }

        #[test]
        fn only_ecx_selects_the_msr() {
            // rdmsr; hlt
            let h = run_ok(&[0x0f, 0x32, 0xf4], |h| {
                h.cpu.pat = 0x77;
                h.cpu.gpr[RCX] = 0x1_0000_0000 | TEST_MSR_PAT as u64;
            });
            assert_eq!(h.cpu.gpr[RAX], 0x77);
        }

        #[test]
        fn rdmsr_of_an_unknown_msr_faults_without_touching_registers() {
            // rdmsr; hlt
            let h = assert_general_protection_zero(&[0x0f, 0x32, 0xf4], |h| {
                h.cpu.gpr[RCX] = 0xdead_beef;
                h.cpu.gpr[RAX] = 0x1234;
                h.cpu.gpr[RDX] = 0x5678;
            });
            assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RDX]), (0x1234, 0x5678));
        }

        #[test]
        fn rdmsr_and_wrmsr_at_cpl3_are_gp0() {
            // rdmsr; hlt
            // wrmsr; hlt
            for code in [&[0x0f, 0x32, 0xf4], &[0x0f, 0x30, 0xf4]] {
                assert_general_protection_zero(code, |h| {
                    user_mode(h);
                    h.cpu.gpr[RCX] = TEST_MSR_PAT as u64;
                });
            }
        }

        #[test]
        fn privilege_is_checked_before_the_msr_is_looked_up() {
            // rdmsr; hlt
            assert_general_protection_zero(&[0x0f, 0x32, 0xf4], |h| {
                user_mode(h);
                h.cpu.gpr[RCX] = TEST_MSR_APIC_BASE as u64;
            });
        }

        #[test]
        fn sysenter_and_sysexit_are_not_implemented_in_long_mode() {
            let codes: [&[u8]; 2] = [
                // sysenter; hlt
                &[0x0f, 0x34, 0xf4],
                // .byte 0x0f, 0x35; hlt
                &[0x0f, 0x35, 0xf4],
            ];
            for code in codes {
                assert_invalid_opcode(code, |_| {});
            }
        }

        #[test]
        fn rdpmc_reads_the_legacy_counters_by_index() {
            // rdpmc; hlt
            const RDPMC: &[u8] = &[0x0f, 0x33, 0xf4];
            for counter in 0..4u32 {
                let value = 0xffff_aaaa_bbbb_cccc_u64 ^ counter as u64;
                let h = run_ok(RDPMC, |h| {
                    h.cpu.wrmsr(TEST_MSR_K7_PERFCTR0 + counter, value).unwrap();
                    h.cpu.gpr[RCX] = 0xdead_0000_0000_0000 | counter as u64;
                });
                assert_eq!(h.cpu.gpr[RAX], value & 0xffff_ffff, "counter {counter}");
                assert_eq!(
                    h.cpu.gpr[RDX],
                    (value >> 32) & 0xffff,
                    "counter {counter}: 48 bits"
                );
            }
        }

        #[test]
        fn rdpmc_needs_an_existing_counter_and_cr4_pce_outside_cpl_0() {
            const RDPMC: &[u8] = &[0x0f, 0x33, 0xf4];
            assert_general_protection_zero(RDPMC, |h| h.cpu.gpr[RCX] = 4);
            assert_general_protection_zero(RDPMC, |h| h.cpu.gpr[RCX] = 1 << 31);
            // at CPL 3 without CR4.PCE it faults on the instruction; with PCE only the hlt does
            assert_general_protection_zero(RDPMC, user_mode);
            let (h, stop) = run_with(RDPMC, |h| {
                user_mode(h);
                h.cpu.cr4 |= CR4_PCE;
            });
            assert_eq!(stop, general_protection(0));
            assert_eq!(
                h.cpu.rip,
                CODE_BASE + 2,
                "rdpmc ran, the hlt behind it faulted"
            );
        }
    }

    mod protection_features {
        use super::*;

        const SGDT: &[u8] = &[0x0f, 0x01, 0x06, 0xf4];
        const SIDT: &[u8] = &[0x0f, 0x01, 0x0e, 0xf4];
        const SMSW: &[u8] = &[0x0f, 0x01, 0xe0, 0xf4];
        const SLDT: &[u8] = &[0x0f, 0x00, 0xc0, 0xf4];
        const STR: &[u8] = &[0x0f, 0x00, 0xc8, 0xf4];
        const MONITOR: &[u8] = &[0x0f, 0x01, 0xc8, 0xf4];
        const MWAIT: &[u8] = &[0x0f, 0x01, 0xc9, 0xf4];
        const MONITOR_MWAIT_MOV: &[u8] =
            &[0x0f, 0x01, 0xc8, 0x0f, 0x01, 0xc9, 0xb8, 5, 0, 0, 0, 0xf4];
        const CLAC: &[u8] = &[0x0f, 0x01, 0xca, 0xf4];
        const STAC: &[u8] = &[0x0f, 0x01, 0xcb, 0xf4];
        const LOAD_RSI: &[u8] = &[0x48, 0x8b, 0x06, 0xf4];
        const STAC_LOAD_CLAC_LOAD: &[u8] = &[
            0x0f, 0x01, 0xcb, 0x48, 0x8b, 0x06, 0x0f, 0x01, 0xca, 0x48, 0x8b, 0x06, 0xf4,
        ];
        const INVPCID: &[u8] = &[0x66, 0x0f, 0x38, 0x82, 0x06, 0xf4];
        const INVPCID_REGISTER: &[u8] = &[0x66, 0x0f, 0x38, 0x82, 0xc6, 0xf4];
        const WRITE_CR3: &[u8] = &[0x0f, 0x22, 0xd8, 0xf4];
        const CR4_UMIP_BIT: u64 = 1 << 11;
        const CR4_PCIDE_BIT: u64 = 1 << 17;
        const CR4_SMEP_BIT: u64 = 1 << 20;
        const CR4_SMAP_BIT: u64 = 1 << 21;
        const AC_BIT: u64 = 1 << 18;
        const IF_BIT: u64 = 1 << 9;

        fn page_fault(code: u32) -> Stop {
            Stop::Fault {
                vector: 14,
                error_code: Some(code),
                software: false,
            }
        }

        #[test]
        fn umip_blocks_the_descriptor_table_reads_outside_cpl0() {
            for code in [SGDT, SIDT, SMSW, SLDT, STR] {
                // without UMIP they work at CPL 3: the only fault is the final hlt
                let (h, stop) = run_with(code, |h| {
                    user_mode(h);
                    h.cpu.gpr[RSI] = DATA;
                });
                assert_eq!(stop, general_protection(0));
                assert_eq!(
                    h.cpu.rip,
                    CODE_BASE + 3,
                    "stopped at hlt, not the instruction"
                );
                assert_general_protection_zero(code, |h| {
                    user_mode(h);
                    h.cpu.cr4 |= CR4_UMIP_BIT;
                    h.cpu.gpr[RSI] = DATA;
                });
                // CPL 0 is exempt
                run_ok(code, |h| {
                    h.cpu.cr4 |= CR4_UMIP_BIT;
                    h.cpu.gpr[RSI] = DATA;
                });
            }
        }

        #[test]
        fn smap_faults_supervisor_data_accesses_to_user_pages_unless_ac() {
            let setup = |h: &mut Harness| {
                user_mode(h);
                h.cpu.set_cpl(0);
                h.cpu.cr4 |= CR4_SMAP_BIT;
                h.cpu.gpr[RSI] = DATA;
            };
            let h = assert_faults(LOAD_RSI, setup, page_fault(1));
            assert_eq!(h.cpu.cr2, DATA);
            // AC set: allowed
            run_ok(LOAD_RSI, |h| {
                setup(h);
                h.cpu.eflags |= AC_BIT;
            });
            // without SMAP the same access is fine
            run_ok(LOAD_RSI, |h| {
                setup(h);
                h.cpu.cr4 &= !CR4_SMAP_BIT;
            });
        }

        #[test]
        fn stac_and_clac_toggle_ac_and_a_cached_translation_cannot_outlive_clac() {
            // stac; mov rax,[rsi]; clac; mov rax,[rsi]: the second load must fault even though the
            // first one just walked the same page.
            let (h, stop) = run_with(STAC_LOAD_CLAC_LOAD, |h| {
                user_mode(h);
                h.cpu.set_cpl(0);
                h.cpu.cr4 |= CR4_SMAP_BIT;
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!(stop, page_fault(1));
            assert_eq!(h.cpu.eflags & AC_BIT, 0);
            let h = run_ok(STAC, |h| h.cpu.eflags &= !AC_BIT);
            assert_ne!(h.cpu.eflags & AC_BIT, 0);
            let h = run_ok(CLAC, |h| h.cpu.eflags |= AC_BIT);
            assert_eq!(h.cpu.eflags & AC_BIT, 0);
        }

        #[test]
        fn stac_and_clac_are_ud_outside_cpl0() {
            for code in [STAC, CLAC] {
                assert_invalid_opcode(code, user_mode);
            }
        }

        #[test]
        fn smep_faults_supervisor_fetches_from_user_pages() {
            // the code page itself is user-accessible here
            let h = assert_faults(
                SGDT,
                |h| {
                    user_mode(h);
                    h.cpu.set_cpl(0);
                    h.cpu.cr4 |= CR4_SMEP_BIT;
                },
                page_fault(0x11),
            );
            assert_eq!(h.cpu.cr2, CODE_BASE);
            // user fetches from user pages are unaffected
            let (h, stop) = run_with(SGDT, |h| {
                user_mode(h);
                h.cpu.cr4 |= CR4_SMEP_BIT;
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!(stop, general_protection(0), "only the hlt faults");
            assert_eq!(h.cpu.rip, CODE_BASE + 3);
        }

        #[test]
        fn pcid_cr3_keeps_the_pcid_and_ignores_the_no_flush_bit() {
            let cr3 = {
                let h = Harness::new(&[0xf4]);
                h.cpu.cr3
            };
            let h = run_ok(WRITE_CR3, |h| {
                h.cpu.cr3 &= !0xfff;
                h.cpu.cr4 |= CR4_PCIDE_BIT;
                h.cpu.gpr[RAX] = (cr3 & !0xfff) | 5 | (1 << 63);
            });
            assert_eq!(h.cpu.cr3, (cr3 & !0xfff) | 5);
            // without PCIDE bit 63 is reserved
            assert_general_protection_zero(WRITE_CR3, |h| {
                h.cpu.gpr[RAX] = h.cpu.cr3 | (1 << 63);
            });
        }

        #[test]
        fn pcide_cannot_be_set_while_cr3_holds_a_pcid() {
            let mut h = blank();
            h.cpu.cr3 |= 5;
            assert_gp0(h.cpu.wrmsr(0xdead, 0));
            let value = h.cpu.cr4 | CR4_PCIDE_BIT;
            h.cpu.gpr[RAX] = value;
            h.write(CODE_BASE, &[0x0f, 0x22, 0xe0, 0xf4]);
            h.cpu.rip = CODE_BASE;
            assert_eq!(h.run(10), general_protection(0));
            assert_eq!(h.cpu.cr4 & CR4_PCIDE_BIT, 0);
        }

        fn invpcid_setup(kind: u64, low: u64, linear: u64) -> impl FnOnce(&mut Harness) {
            move |h| {
                h.cpu.gpr[RAX] = kind;
                h.cpu.gpr[RSI] = DATA;
                h.write_u64(DATA, low);
                h.write_u64(DATA + 8, linear);
            }
        }

        #[test]
        fn invpcid_validates_its_operands_and_flushes_the_tlb() {
            for kind in 0..=3 {
                let h = run_ok(INVPCID, |h| {
                    invpcid_setup(kind, 3, DATA)(h);
                    // fill a TLB entry first
                    h.cpu.read(DATA, 8).unwrap();
                });
                assert!(
                    h.cpu.tlb.iter().all(|e| e.tag_r == !0 || e.tag_r != DATA),
                    "type {kind}"
                );
            }
            assert_general_protection_zero(INVPCID, invpcid_setup(4, 0, 0));
            assert_general_protection_zero(INVPCID, invpcid_setup(0, 1 << 12, 0));
            assert_general_protection_zero(INVPCID, invpcid_setup(0, 0, 1 << 47));
            // a non-canonical address only matters for type 0
            run_ok(INVPCID, invpcid_setup(1, 0, 1 << 47));
            assert_general_protection_zero(INVPCID, |h| {
                invpcid_setup(0, 0, DATA)(h);
                user_mode(h);
            });
            assert_invalid_opcode(INVPCID_REGISTER, |h| h.cpu.gpr[RAX] = 0);
        }

        #[test]
        fn monitor_and_mwait_are_privileged_and_check_ecx() {
            for code in [MONITOR, MWAIT] {
                assert_general_protection_zero(code, user_mode);
            }
            assert_general_protection_zero(MONITOR, |h| h.cpu.gpr[RCX] = 1);
            assert_general_protection_zero(MONITOR, |h| h.cpu.gpr[RDX] = 1);
            assert_general_protection_zero(MWAIT, |h| h.cpu.gpr[RCX] = 2);
        }

        #[test]
        fn mwait_waits_like_hlt_only_with_interrupts_enabled_and_a_monitor_armed() {
            let enabled = run_ok(MONITOR_MWAIT_MOV, |h| {
                h.cpu.set_eflags_arith(0);
                h.cpu.eflags |= IF_BIT;
            });
            assert!(enabled.cpu.halted);
            assert_ne!(enabled.cpu.gpr[RAX], 5, "stopped at mwait");
            let masked = run_ok(MONITOR_MWAIT_MOV, |h| h.cpu.eflags &= !IF_BIT);
            assert_eq!(masked.cpu.gpr[RAX], 5, "returns at once with IF=0");
            // no monitor armed: returns
            let unarmed = run_ok(&[0x0f, 0x01, 0xc9, 0xb8, 5, 0, 0, 0, 0xf4], |h| {
                h.cpu.eflags |= IF_BIT
            });
            assert_eq!(unarmed.cpu.gpr[RAX], 5);
        }
    }

    mod rdtsc {
        use super::*;

        const RDTSCP: &[u8] = &[0x0f, 0x01, 0xf9, 0xf4];

        #[test]
        fn rdtscp_returns_the_time_stamp_and_tsc_aux() {
            let h = run_ok(RDTSCP, |h| {
                h.cpu.tsc_aux = 0x1_2345_6789;
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
            });
            assert_eq!(
                h.cpu.gpr[RCX], 0x2345_6789,
                "TSC_AUX zero-extended into ECX"
            );
            assert_eq!(h.cpu.gpr[RAX] >> 32, 0);
            assert_eq!(h.cpu.gpr[RDX] >> 32, 0);
            assert_ne!(h.cpu.gpr[RAX] | h.cpu.gpr[RDX], 0);
        }

        #[test]
        fn rdtscp_obeys_cr4_tsd_like_rdtsc() {
            let (_, stop) = run_with(RDTSCP, |h| {
                user_mode(h);
                h.cpu.cr4 |= CR4_TSD;
            });
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: 13,
                    error_code: Some(0),
                    software: false
                }
            );
            let (h, stop) = run_with(RDTSCP, |h| h.cpu.cr4 |= CR4_TSD);
            assert_eq!(stop, Stop::Halted, "CPL 0 is exempt");
            assert_ne!(h.cpu.rip, 0);
        }

        #[test]
        fn extended_leaf_1_advertises_rdtscp() {
            let mut h = blank();
            let [_, _, _, edx] = cpuid(&mut h, 0x8000_0001, 0);
            assert_ne!(edx & (1 << 27), 0);
        }

        #[test]
        fn rdtsc_returns_the_virtual_clock_in_edx_eax_and_zero_extends() {
            // rdtsc; hlt
            let h = run_ok(&[0x0f, 0x31, 0xf4], |h| {
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RDX] = u64::MAX;
                // The first instruction executes at icount + 1, so the clock reads 2^32 - 1.
                h.cpu.icount = 0xffff_fffe;
            });
            assert_eq!(h.cpu.gpr[RAX], 0xffff_ffff * NS_PER_INSN);
            assert_eq!(
                h.cpu.gpr[RDX], 0,
                "EDX is the high half and the register is zero-extended"
            );
        }

        #[test]
        fn rdtsc_splits_a_64_bit_value_across_edx_and_eax() {
            // rdtsc; hlt
            let h = run_ok(&[0x0f, 0x31, 0xf4], |h| {
                h.cpu.icount = 0x1234_5678_9abc_def0
            });
            let tsc = (h.cpu.gpr[RDX] << 32) | h.cpu.gpr[RAX];
            assert_eq!(tsc, 0x1234_5678_9abc_def1 * NS_PER_INSN);
            assert_eq!(h.cpu.gpr[RDX], 0x1234_5678);
        }

        #[test]
        fn consecutive_reads_advance_by_the_instructions_between_them() {
            // rdtsc; mov rsi, rax; nop; rdtsc; hlt
            let h = run_ok(
                &[0x0f, 0x31, 0x48, 0x89, 0xc6, 0x90, 0x0f, 0x31, 0xf4],
                |_| {},
            );
            assert_eq!(h.cpu.gpr[RSI], NS_PER_INSN);
            assert_eq!(h.cpu.gpr[RAX], 4 * NS_PER_INSN);
        }

        #[test]
        fn idle_time_advances_the_tsc() {
            // rdtsc; hlt
            let h = run_ok(&[0x0f, 0x31, 0xf4], |h| h.cpu.idle_skipped_ns = 10_000);
            assert_eq!(h.cpu.gpr[RAX], NS_PER_INSN + 10_000);
        }

        #[test]
        fn cr4_tsd_makes_rdtsc_privileged() {
            // rdtsc; hlt
            assert_general_protection_zero(&[0x0f, 0x31, 0xf4], |h| {
                user_mode(h);
                h.cpu.cr4 |= CR4_TSD;
            });
        }

        #[test]
        fn cr4_tsd_does_not_affect_cpl0() {
            // rdtsc; hlt
            run_ok(&[0x0f, 0x31, 0xf4], |h| h.cpu.cr4 |= CR4_TSD);
        }

        #[test]
        fn rdtsc_is_allowed_at_cpl3_without_tsd() {
            // rdtsc; hlt
            let (h, stop) = run_with(&[0x0f, 0x31, 0xf4], user_mode);
            assert_eq!(
                h.cpu.rip,
                CODE_BASE + 2,
                "rdtsc retired, then hlt faulted: {stop:?}"
            );
            assert_ne!(h.cpu.gpr[RAX], 0);
        }

        #[test]
        fn jit_and_interpreter_agree_on_rdtsc() {
            crate::testkit::assert_jit_matches_interpreter(
                // rdtsc; mov rsi, rax; nop; nop; rdtsc; sub rax, rsi; hlt
                &[
                    0x0f, 0x31, 0x48, 0x89, 0xc6, 0x90, 0x90, 0x0f, 0x31, 0x48, 0x29, 0xf0, 0xf4,
                ],
                |_| {},
            );
        }
    }

    mod control_registers {
        use super::*;
        use crate::cpu::mem::PagingMode;
        use crate::cpu::test_support::LegacyTables;

        // mov cr0, rax; hlt
        const WRITE_CR0: &[u8] = &[0x0f, 0x22, 0xc0, 0xf4];
        // mov rax, cr0; hlt
        const READ_CR0: &[u8] = &[0x0f, 0x20, 0xc0, 0xf4];
        // mov cr3, rax; hlt
        const WRITE_CR3: &[u8] = &[0x0f, 0x22, 0xd8, 0xf4];
        // mov cr4, rax; hlt
        const WRITE_CR4: &[u8] = &[0x0f, 0x22, 0xe0, 0xf4];

        fn write_cr0(value: u64) -> (Harness, Stop) {
            run_with(WRITE_CR0, |h| h.cpu.gpr[RAX] = value)
        }

        /// CPU state as if 64-bit paging had not been entered yet: protected mode, no PG, LME set.
        fn not_yet_in_long_mode(h: &mut Harness) {
            h.cpu.cr0 = CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT;
            h.cpu.efer = EFER_LME_BIT;
            h.cpu.cr4 = 0;
        }

        #[test]
        fn reading_cr0_returns_the_boot_value() {
            let h = run_ok(READ_CR0, |_| {});
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR0);
        }

        #[test]
        fn writing_the_same_cr0_is_accepted() {
            let (h, stop) = write_cr0(BOOT_CR0);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn cr0_accepts_write_protect_and_alignment_mask() {
            let (h, stop) = write_cr0(BOOT_CR0 | CR0_WP_BIT | CR0_AM_BIT);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_WP_BIT | CR0_AM_BIT);
        }

        #[test]
        fn cr0_extension_type_reads_back_as_one() {
            let (h, stop) = write_cr0(BOOT_CR0 & !CR0_ET_BIT);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn cr0_ts_and_em_are_stored() {
            let (h, _) = write_cr0(BOOT_CR0 | CR0_TS_BIT | CR0_EM_BIT);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_TS_BIT | CR0_EM_BIT);
        }

        #[test]
        fn setting_any_of_the_upper_32_bits_of_cr0_is_gp0() {
            for bit in 32..64 {
                let h = assert_general_protection_zero(WRITE_CR0, |h| {
                    h.cpu.gpr[RAX] = BOOT_CR0 | 1 << bit
                });
                assert_eq!(h.cpu.cr0, BOOT_CR0, "bit {bit} must not reach CR0");
            }
        }

        #[test]
        fn not_write_through_without_cache_disable_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR0, |h| {
                h.cpu.gpr[RAX] = BOOT_CR0 | CR0_NW_BIT
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn cache_disable_alone_and_with_not_write_through_are_accepted() {
            let (h, stop) = write_cr0(BOOT_CR0 | CR0_CD_BIT);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_CD_BIT);
            let (h, stop) = write_cr0(BOOT_CR0 | CR0_CD_BIT | CR0_NW_BIT);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_CD_BIT | CR0_NW_BIT);
        }

        #[test]
        fn paging_without_protection_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR0, |h| {
                not_yet_in_long_mode(h);
                h.cpu.efer = 0;
                h.cpu.gpr[RAX] = CR0_PG_BIT | CR0_ET_BIT;
            });
            assert_eq!(h.cpu.cr0 & CR0_PG_BIT, 0);
        }

        #[test]
        fn clearing_pe_while_keeping_pg_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR0, |h| {
                h.cpu.gpr[RAX] = BOOT_CR0 & !CR0_PE_BIT
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn clearing_pg_in_long_mode_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR0, |h| {
                h.cpu.gpr[RAX] = BOOT_CR0 & !CR0_PG_BIT
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0);
            assert_ne!(h.cpu.efer & EFER_LMA_BIT, 0, "long mode stays active");
        }

        #[test]
        fn clearing_pg_from_a_compatibility_mode_segment_leaves_long_mode() {
            // The documented way out of IA-32e mode: clear PG while CS.L is 0.
            let (h, stop) = run_with(WRITE_CR0, |h| {
                h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
                h.cpu.gpr[RAX] = CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0 & CR0_PG_BIT, 0);
            assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0, "LMA drops with PG");
            assert_ne!(h.cpu.efer & EFER_LME_BIT, 0, "LME is untouched");
        }

        #[test]
        fn clearing_both_pg_and_pe_in_long_mode_is_gp0() {
            assert_general_protection_zero(WRITE_CR0, |h| h.cpu.gpr[RAX] = CR0_ET_BIT);
        }

        #[test]
        fn clearing_pg_outside_long_mode_is_accepted() {
            let (h, stop) = run_with(WRITE_CR0, |h| {
                LegacyTables::new(&mut h.cpu).identity_map(&mut h.cpu, 32);
                h.cpu.gpr[RAX] = CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT);
            assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0);
        }

        #[test]
        fn enabling_paging_with_lme_and_pae_activates_long_mode() {
            let (h, stop) = run_with(WRITE_CR0, |h| {
                not_yet_in_long_mode(h);
                h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
                h.cpu.cr4 = BOOT_CR4;
                h.cpu.gpr[RAX] = CR0_PG_BIT | CR0_PE_BIT | CR0_ET_BIT;
            });
            assert_eq!(stop, Stop::Halted);
            assert_ne!(h.cpu.efer & EFER_LMA_BIT, 0, "LMA follows PG && LME");
        }

        #[test]
        fn enabling_paging_without_lme_leaves_long_mode_inactive() {
            let (h, stop) = run_with(WRITE_CR0, |h| {
                not_yet_in_long_mode(h);
                // 32-bit paging with the first 16 MiB identity mapped, so code keeps running.
                let mut tables = LegacyTables::new(&mut h.cpu);
                tables.identity_map(&mut h.cpu, 32);
                h.cpu.cr0 = CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT;
                h.cpu.flush_tlb();
                h.cpu.gpr[RAX] = CR0_PG_BIT | CR0_PE_BIT | CR0_ET_BIT;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0);
            assert_eq!(h.cpu.paging_mode(), PagingMode::Legacy);
        }

        #[test]
        fn enabling_paging_with_lme_but_without_pae_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR0, |h| {
                not_yet_in_long_mode(h);
                h.cpu.gpr[RAX] = CR0_PG_BIT | CR0_PE_BIT | CR0_ET_BIT;
            });
            assert_eq!(h.cpu.cr0 & CR0_PG_BIT, 0);
            assert_eq!(h.cpu.efer & EFER_LMA_BIT, 0);
        }

        #[test]
        fn a_cr0_write_flushes_the_tlb() {
            let mut h = Harness::new(WRITE_CR0);
            h.cpu.gpr[RAX] = BOOT_CR0;
            assert_eq!(h.run(1), Stop::Limit);
            assert!(
                h.cpu
                    .tlb
                    .iter()
                    .all(|entry| entry.tag_r == TLB_TAG_INVALID && entry.tag_x == TLB_TAG_INVALID)
            );
        }

        #[test]
        fn a_faulting_cr0_write_does_not_flush_the_tlb() {
            let mut h = Harness::new(WRITE_CR0);
            h.cpu.gpr[RAX] = BOOT_CR0 | 1 << 40;
            h.cpu.read(DATA, 8).unwrap();
            assert!(h.cpu.tlb.iter().any(|entry| entry.tag_r != TLB_TAG_INVALID));
            assert!(matches!(h.run(1), Stop::Fault { .. }));
            assert!(h.cpu.tlb.iter().any(|entry| entry.tag_r != TLB_TAG_INVALID));
        }

        #[test]
        fn cr0_access_at_cpl3_is_gp0() {
            for code in [WRITE_CR0, READ_CR0] {
                assert_general_protection_zero(code, |h| {
                    user_mode(h);
                    h.cpu.gpr[RAX] = BOOT_CR0;
                });
            }
        }

        #[test]
        fn the_general_register_comes_from_modrm_rm_with_rex_b() {
            // mov r8, cr0; hlt
            let h = run_ok(&[0x41, 0x0f, 0x20, 0xc0, 0xf4], |_| {});
            assert_eq!(h.cpu.gpr[R8], BOOT_CR0);
            // mov cr0, r9; hlt
            let (h, stop) = run_with(&[0x41, 0x0f, 0x22, 0xc1, 0xf4], |h| {
                h.cpu.gpr[R9] = BOOT_CR0 | CR0_WP_BIT
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_WP_BIT);
        }

        #[test]
        fn the_modrm_mod_field_is_ignored_for_control_register_moves() {
            // 0f 20 00 has mod=00 but still names rax as the register operand.
            // .byte 0x0f, 0x20, 0x00; hlt
            let h = run_ok(&[0x0f, 0x20, 0x00, 0xf4], |_| {});
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR0);
        }

        #[test]
        fn reading_a_control_register_writes_all_64_bits() {
            // mov rax, cr4; hlt
            let h = run_ok(&[0x0f, 0x20, 0xe0, 0xf4], |h| h.cpu.gpr[RAX] = u64::MAX);
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR4);
        }

        #[test]
        fn cr2_holds_any_value() {
            for value in [0, 0x1234, 0xffff_8000_dead_beef, u64::MAX] {
                // mov cr2, rax; mov rbx, cr2; hlt
                let h = run_ok(&[0x0f, 0x22, 0xd0, 0x0f, 0x20, 0xd3, 0xf4], |h| {
                    h.cpu.gpr[RAX] = value
                });
                assert_eq!(h.cpu.cr2, value);
                assert_eq!(h.cpu.gpr[RBX], value);
            }
        }

        #[test]
        fn cr3_write_keeps_pwt_and_pcd_and_the_page_table_address() {
            let mut h = Harness::new(WRITE_CR3);
            let original = h.cpu.cr3;
            h.cpu.gpr[RAX] = original | 0x18;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cr3, original | 0x18);
        }

        #[test]
        fn cr3_ignores_the_low_pcid_style_bits() {
            let mut h = Harness::new(WRITE_CR3);
            let original = h.cpu.cr3;
            // bits 2:0 and 11:5 are ignored without CR4.PCIDE
            h.cpu.gpr[RAX] = original | 0xfe7;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cr3, original);
        }

        #[test]
        fn cr3_accepts_the_highest_valid_physical_address_bit() {
            let mut h = Harness::new(WRITE_CR3);
            h.cpu.gpr[RAX] = 1 << 39;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cr3, 1 << 39);
        }

        #[test]
        fn cr3_bits_above_the_advertised_physical_address_width_are_gp0() {
            for bit in 40..64 {
                let mut h = Harness::new(WRITE_CR3);
                let original = h.cpu.cr3;
                h.cpu.gpr[RAX] = original | 1 << bit;
                assert_eq!(
                    h.run(1),
                    general_protection(0),
                    "bit {bit} is reserved: CPUID 0x80000008 reports 40 physical bits"
                );
                assert_eq!(h.cpu.cr3, original, "bit {bit}");
                assert_eq!(h.cpu.rip, CODE_BASE);
            }
        }

        #[test]
        fn a_cr3_write_flushes_the_tlb() {
            let mut h = Harness::new(WRITE_CR3);
            h.cpu.gpr[RAX] = h.cpu.cr3;
            assert_eq!(h.run(1), Stop::Limit);
            assert!(h.cpu.tlb.iter().all(|entry| entry.tag_r == TLB_TAG_INVALID));
        }

        #[test]
        fn cr3_round_trips_through_a_read() {
            // mov rax, cr3; mov cr3, rax; mov rbx, cr3; hlt
            let h = run_ok(
                &[0x0f, 0x20, 0xd8, 0x0f, 0x22, 0xd8, 0x0f, 0x20, 0xdb, 0xf4],
                |_| {},
            );
            assert_eq!(h.cpu.gpr[RBX], h.cpu.gpr[RAX]);
        }

        #[test]
        fn cr3_access_at_cpl3_is_gp0() {
            assert_general_protection_zero(WRITE_CR3, user_mode);
            // mov rax, cr3; hlt
            assert_general_protection_zero(&[0x0f, 0x20, 0xd8, 0xf4], user_mode);
        }

        #[test]
        fn cr4_reads_the_boot_value() {
            // mov rax, cr4; hlt
            let h = run_ok(&[0x0f, 0x20, 0xe0, 0xf4], |_| {});
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR4);
        }

        #[test]
        fn cr4_accepts_the_supported_bits() {
            for (bit, name) in [
                (2, "TSD"),
                (3, "DE"),
                (4, "PSE"),
                (7, "PGE"),
                (8, "PCE"),
                (9, "OSFXSR"),
                (10, "OSXMMEXCPT"),
                (16, "FSGSBASE"),
            ] {
                let (h, stop) = run_with(WRITE_CR4, |h| h.cpu.gpr[RAX] = BOOT_CR4 | 1 << bit);
                assert_eq!(stop, Stop::Halted, "{name}");
                assert_eq!(h.cpu.cr4, BOOT_CR4 | 1 << bit, "{name}");
            }
        }

        #[test]
        fn cr4_unsupported_bits_are_gp0_and_leave_cr4_alone() {
            for (bit, name) in [
                (0, "VME"),
                (1, "PVI"),
                (12, "LA57"),
                (13, "VMXE"),
                (14, "SMXE"),
                (19, "reserved"),
                (22, "PKE"),
                (32, "upper half"),
                (63, "upper half"),
            ] {
                let h = assert_general_protection_zero(WRITE_CR4, |h| {
                    h.cpu.gpr[RAX] = BOOT_CR4 | 1 << bit
                });
                assert_eq!(h.cpu.cr4, BOOT_CR4, "{name}");
            }
        }

        #[test]
        fn clearing_pae_while_long_mode_is_active_is_gp0() {
            let h = assert_general_protection_zero(WRITE_CR4, |h| h.cpu.gpr[RAX] = 0);
            assert_eq!(h.cpu.cr4, BOOT_CR4);
        }

        #[test]
        fn clearing_pae_outside_long_mode_is_accepted() {
            let (h, stop) = run_with(WRITE_CR4, |h| {
                h.cpu.efer = 0;
                h.cpu.cr0 = CR0_PE_BIT | CR0_ET_BIT | CR0_NE_BIT;
                h.cpu.gpr[RAX] = 0;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.cr4, 0);
        }

        #[test]
        fn a_cr4_write_flushes_the_tlb() {
            let mut h = Harness::new(WRITE_CR4);
            h.cpu.gpr[RAX] = BOOT_CR4 | CR4_PGE;
            assert_eq!(h.run(1), Stop::Limit);
            assert!(h.cpu.tlb.iter().all(|entry| entry.tag_r == TLB_TAG_INVALID));
        }

        #[test]
        fn cr4_access_at_cpl3_is_gp0() {
            assert_general_protection_zero(WRITE_CR4, user_mode);
            // mov rax, cr4; hlt
            assert_general_protection_zero(&[0x0f, 0x20, 0xe0, 0xf4], user_mode);
        }

        #[test]
        fn cr8_holds_a_4_bit_task_priority() {
            for value in [0, 1, 0xf] {
                // mov cr8, rax; mov rbx, cr8; hlt
                let h = run_ok(
                    &[0x44, 0x0f, 0x22, 0xc0, 0x44, 0x0f, 0x20, 0xc3, 0xf4],
                    |h| h.cpu.gpr[RAX] = value,
                );
                assert_eq!(h.cpu.gpr[RBX], value);
            }
        }

        #[test]
        fn cr8_reserved_bits_are_gp0() {
            for bit in [4, 5, 31, 32, 63] {
                // mov cr8, rax; hlt
                let h = assert_general_protection_zero(&[0x44, 0x0f, 0x22, 0xc0, 0xf4], |h| {
                    h.cpu.gpr[RAX] = 1 << bit
                });
                assert_eq!(h.cpu.dev.lapic().tpr(), 0, "bit {bit}");
            }
        }

        #[test]
        fn nonexistent_control_registers_are_ud() {
            const WRITES: [&[u8]; 6] = [
                // mov cr1, rax; hlt
                &[0x0f, 0x22, 0xc8, 0xf4],
                // mov cr5, rax; hlt
                &[0x0f, 0x22, 0xe8, 0xf4],
                // mov cr6, rax; hlt
                &[0x0f, 0x22, 0xf0, 0xf4],
                // mov cr7, rax; hlt
                &[0x0f, 0x22, 0xf8, 0xf4],
                // mov cr9, rax; hlt
                &[0x44, 0x0f, 0x22, 0xc8, 0xf4],
                // mov cr15, rax; hlt
                &[0x44, 0x0f, 0x22, 0xf8, 0xf4],
            ];
            const READS: [&[u8]; 3] = [
                // mov rax, cr1; hlt
                &[0x0f, 0x20, 0xc8, 0xf4],
                // mov rax, cr5; hlt
                &[0x0f, 0x20, 0xe8, 0xf4],
                // mov rax, cr9; hlt
                &[0x44, 0x0f, 0x20, 0xc8, 0xf4],
            ];
            for code in WRITES.into_iter().chain(READS) {
                assert_invalid_opcode(code, |h| h.cpu.gpr[RAX] = 5);
            }
        }

        #[test]
        fn a_ud_control_register_read_leaves_the_destination_alone() {
            // mov rax, cr1; hlt
            let h = assert_invalid_opcode(&[0x0f, 0x20, 0xc8, 0xf4], |h| h.cpu.gpr[RAX] = 0x55);
            assert_eq!(h.cpu.gpr[RAX], 0x55);
        }

        #[test]
        fn jit_and_interpreter_agree_on_control_register_moves() {
            crate::testkit::assert_jit_matches_interpreter(
                // mov rax, cr0; mov rbx, cr4; mov rcx, cr3; mov cr8, rdx; mov rsi, cr8; hlt
                &[
                    0x0f, 0x20, 0xc0, 0x0f, 0x20, 0xe3, 0x0f, 0x20, 0xd9, 0x44, 0x0f, 0x22, 0xc2,
                    0x44, 0x0f, 0x20, 0xc6, 0xf4,
                ],
                |h| h.cpu.gpr[RDX] = 7,
            );
        }
    }

    mod debug_registers {
        use super::*;

        const DR6_RESET_VALUE: u64 = 0xffff_0ff0;
        const DR7_RESET_VALUE: u64 = 0x400;
        const READ_DR: [&[u8]; 8] = [
            // mov rax, dr0; hlt
            &[0x0f, 0x21, 0xc0, 0xf4],
            // mov rax, dr1; hlt
            &[0x0f, 0x21, 0xc8, 0xf4],
            // mov rax, dr2; hlt
            &[0x0f, 0x21, 0xd0, 0xf4],
            // mov rax, dr3; hlt
            &[0x0f, 0x21, 0xd8, 0xf4],
            // mov rax, dr4; hlt
            &[0x0f, 0x21, 0xe0, 0xf4],
            // mov rax, dr5; hlt
            &[0x0f, 0x21, 0xe8, 0xf4],
            // mov rax, dr6; hlt
            &[0x0f, 0x21, 0xf0, 0xf4],
            // mov rax, dr7; hlt
            &[0x0f, 0x21, 0xf8, 0xf4],
        ];
        const WRITE_DR: [&[u8]; 8] = [
            // mov dr0, rax; hlt
            &[0x0f, 0x23, 0xc0, 0xf4],
            // mov dr1, rax; hlt
            &[0x0f, 0x23, 0xc8, 0xf4],
            // mov dr2, rax; hlt
            &[0x0f, 0x23, 0xd0, 0xf4],
            // mov dr3, rax; hlt
            &[0x0f, 0x23, 0xd8, 0xf4],
            // mov dr4, rax; hlt
            &[0x0f, 0x23, 0xe0, 0xf4],
            // mov dr5, rax; hlt
            &[0x0f, 0x23, 0xe8, 0xf4],
            // mov dr6, rax; hlt
            &[0x0f, 0x23, 0xf0, 0xf4],
            // mov dr7, rax; hlt
            &[0x0f, 0x23, 0xf8, 0xf4],
        ];
        const READ_RESERVED_DR: [&[u8]; 8] = [
            // mov rax, dr8; hlt
            &[0x44, 0x0f, 0x21, 0xc0, 0xf4],
            // mov rax, dr9; hlt
            &[0x44, 0x0f, 0x21, 0xc8, 0xf4],
            // mov rax, dr10; hlt
            &[0x44, 0x0f, 0x21, 0xd0, 0xf4],
            // mov rax, dr11; hlt
            &[0x44, 0x0f, 0x21, 0xd8, 0xf4],
            // mov rax, dr12; hlt
            &[0x44, 0x0f, 0x21, 0xe0, 0xf4],
            // mov rax, dr13; hlt
            &[0x44, 0x0f, 0x21, 0xe8, 0xf4],
            // mov rax, dr14; hlt
            &[0x44, 0x0f, 0x21, 0xf0, 0xf4],
            // mov rax, dr15; hlt
            &[0x44, 0x0f, 0x21, 0xf8, 0xf4],
        ];
        const WRITE_RESERVED_DR: [&[u8]; 8] = [
            // mov dr8, rax; hlt
            &[0x44, 0x0f, 0x23, 0xc0, 0xf4],
            // mov dr9, rax; hlt
            &[0x44, 0x0f, 0x23, 0xc8, 0xf4],
            // mov dr10, rax; hlt
            &[0x44, 0x0f, 0x23, 0xd0, 0xf4],
            // mov dr11, rax; hlt
            &[0x44, 0x0f, 0x23, 0xd8, 0xf4],
            // mov dr12, rax; hlt
            &[0x44, 0x0f, 0x23, 0xe0, 0xf4],
            // mov dr13, rax; hlt
            &[0x44, 0x0f, 0x23, 0xe8, 0xf4],
            // mov dr14, rax; hlt
            &[0x44, 0x0f, 0x23, 0xf0, 0xf4],
            // mov dr15, rax; hlt
            &[0x44, 0x0f, 0x23, 0xf8, 0xf4],
        ];

        fn write_dr(register: usize, value: u64) -> (Harness, Stop) {
            run_with(WRITE_DR[register], |h| h.cpu.gpr[RAX] = value)
        }

        fn read_dr(register: usize, setup: impl FnOnce(&mut Harness)) -> u64 {
            run_ok(READ_DR[register], setup).cpu.gpr[RAX]
        }

        #[test]
        fn power_on_values_of_dr6_and_dr7() {
            assert_eq!(read_dr(6, |_| {}), DR6_RESET_VALUE);
            assert_eq!(read_dr(7, |_| {}), DR7_RESET_VALUE);
            for register in 0..4 {
                assert_eq!(read_dr(register, |_| {}), 0, "dr{register}");
            }
        }

        #[test]
        fn address_registers_hold_any_64_bit_value() {
            for register in 0..4 {
                for value in [
                    0,
                    0x1000,
                    0xffff_ffff_ffff_ffff,
                    0x8000_0000_0000_0000,
                    0x1234_5678_9abc_def0,
                ] {
                    let (h, stop) = write_dr(register, value);
                    assert_eq!(stop, Stop::Halted, "dr{register}");
                    assert_eq!(h.cpu.dr[register], value, "dr{register}");
                    assert_eq!(read_dr(register, |h| h.cpu.dr[register] = value), value);
                }
            }
        }

        #[test]
        fn dr6_write_forces_the_reserved_bits() {
            for (written, expected) in [
                (0u64, 0xffff_0ff0u64),
                (0xf, 0xffff_0fff),
                (0xffff_ffff, 0xffff_efff),
                (1 << 13, 0xffff_2ff0),
                (1 << 14, 0xffff_4ff0),
                (1 << 15, 0xffff_8ff0),
                (1 << 12, 0xffff_0ff0),
                (0xffff_0000, 0xffff_0ff0),
                (0x0000_0ff0, 0xffff_0ff0),
            ] {
                let (h, stop) = write_dr(6, written);
                assert_eq!(stop, Stop::Halted);
                assert_eq!(h.cpu.dr[6], expected, "wrote {written:#x}");
            }
        }

        #[test]
        fn dr7_write_forces_the_reserved_bits() {
            for (written, expected) in [
                (0u64, 0x400u64),
                (1, 0x401),
                (0x3ff, 0x7ff),
                (0xffff_ffff, 0xffff_27ff),
                (0x0003_0000, 0x0003_0400),
                (1 << 13, 0x2400),
                (0xd800, 0x400),
            ] {
                let (h, stop) = write_dr(7, written);
                assert_eq!(stop, Stop::Halted);
                assert_eq!(h.cpu.dr[7], expected, "wrote {written:#x}");
            }
        }

        #[test]
        fn dr6_and_dr7_reject_the_upper_32_bits() {
            for register in [6, 7] {
                for bit in [32, 40, 63] {
                    let before = if register == 6 {
                        DR6_RESET_VALUE
                    } else {
                        DR7_RESET_VALUE
                    };
                    let h = assert_general_protection_zero(WRITE_DR[register], |h| {
                        h.cpu.gpr[RAX] = 1 << bit
                    });
                    assert_eq!(h.cpu.dr[register], before, "dr{register} bit {bit}");
                }
            }
        }

        #[test]
        fn dr0_to_dr3_do_not_reject_the_upper_bits() {
            for register in 0..4 {
                let (h, stop) = write_dr(register, 1 << 63);
                assert_eq!(stop, Stop::Halted);
                assert_eq!(h.cpu.dr[register], 1 << 63);
            }
        }

        #[test]
        fn dr4_and_dr5_alias_dr6_and_dr7_without_cr4_de() {
            assert_eq!(read_dr(4, |h| h.cpu.dr[6] = 0xffff_1ff1), 0xffff_1ff1);
            assert_eq!(read_dr(5, |h| h.cpu.dr[7] = 0x455), 0x455);
            let (h, stop) = write_dr(4, 0xf);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(
                h.cpu.dr[6], 0xffff_0fff,
                "a write through dr4 is a dr6 write"
            );
            assert_eq!(h.cpu.dr[4], 0, "the raw slot stays unused");
            let (h, stop) = write_dr(5, 1);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.dr[7], 0x401);
            assert_eq!(h.cpu.dr[5], 0);
        }

        #[test]
        fn dr4_and_dr5_writes_through_the_alias_get_the_same_validation() {
            let h = assert_general_protection_zero(WRITE_DR[4], |h| h.cpu.gpr[RAX] = 1 << 33);
            assert_eq!(h.cpu.dr[6], DR6_RESET_VALUE);
            let h = assert_general_protection_zero(WRITE_DR[5], |h| h.cpu.gpr[RAX] = 1 << 33);
            assert_eq!(h.cpu.dr[7], DR7_RESET_VALUE);
        }

        #[test]
        fn dr4_and_dr5_are_ud_with_cr4_de() {
            for register in [4, 5] {
                let enable_de = |h: &mut Harness| h.cpu.cr4 |= CR4_DE;
                let h = assert_invalid_opcode(READ_DR[register], |h| {
                    enable_de(h);
                    h.cpu.gpr[RAX] = 0x77;
                });
                assert_eq!(h.cpu.gpr[RAX], 0x77, "dr{register} read must not write rax");
                let h = assert_invalid_opcode(WRITE_DR[register], |h| {
                    enable_de(h);
                    h.cpu.gpr[RAX] = 0x77;
                });
                assert_eq!(
                    (h.cpu.dr[6], h.cpu.dr[7]),
                    (DR6_RESET_VALUE, DR7_RESET_VALUE)
                );
            }
        }

        #[test]
        fn cr4_de_does_not_affect_the_real_debug_registers() {
            let (h, stop) = run_with(WRITE_DR[6], |h| {
                h.cpu.cr4 |= CR4_DE;
                h.cpu.gpr[RAX] = 1;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.dr[6], 0xffff_0ff1);
        }

        #[test]
        fn dr8_to_dr15_are_ud_for_reads_and_writes() {
            for (index, code) in READ_RESERVED_DR
                .iter()
                .chain(&WRITE_RESERVED_DR)
                .enumerate()
            {
                let h = assert_invalid_opcode(code, |h| h.cpu.gpr[RAX] = 0x1234);
                assert_eq!(h.cpu.gpr[RAX], 0x1234, "case {index}");
                assert_eq!(
                    h.cpu.dr[..4],
                    [0; 4],
                    "case {index}: no aliasing onto dr0-dr3"
                );
                assert_eq!(
                    (h.cpu.dr[6], h.cpu.dr[7]),
                    (DR6_RESET_VALUE, DR7_RESET_VALUE),
                    "case {index}"
                );
            }
        }

        #[test]
        fn debug_register_access_at_cpl3_is_gp0() {
            for code in [READ_DR[0], WRITE_DR[0], READ_DR[7], WRITE_DR[7]] {
                assert_general_protection_zero(code, user_mode);
            }
        }

        #[test]
        fn the_general_register_comes_from_modrm_rm_with_rex_b() {
            // mov dr0, r9; hlt
            let (h, stop) = run_with(&[0x41, 0x0f, 0x23, 0xc1, 0xf4], |h| h.cpu.gpr[R9] = 0xfeed);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.dr[0], 0xfeed);
            // mov r8, dr0; hlt
            let h = run_ok(&[0x41, 0x0f, 0x21, 0xc0, 0xf4], |h| h.cpu.dr[0] = 0xbeef);
            assert_eq!(h.cpu.gpr[R8], 0xbeef);
        }

        #[test]
        fn writing_debug_registers_does_not_flush_the_tlb() {
            let mut h = Harness::new(WRITE_DR[0]);
            h.cpu.read(DATA, 8).unwrap();
            assert_eq!(h.run(1), Stop::Limit);
            assert!(h.cpu.tlb.iter().any(|entry| entry.tag_r != TLB_TAG_INVALID));
        }
    }

    mod descriptor_registers {
        use super::*;

        // lldt ax; hlt
        const LLDT_AX: &[u8] = &[0x0f, 0x00, 0xd0, 0xf4];
        // ltr ax; hlt
        const LTR_AX: &[u8] = &[0x0f, 0x00, 0xd8, 0xf4];
        // lldt word ptr [rsi]; hlt
        const LLDT_MEMORY: &[u8] = &[0x0f, 0x00, 0x16, 0xf4];
        // ltr word ptr [rsi]; hlt
        const LTR_MEMORY: &[u8] = &[0x0f, 0x00, 0x1e, 0xf4];

        fn load_ldt(selector: u16) -> (Harness, Stop) {
            run_with(LLDT_AX, |h| {
                install_gdt(h);
                h.cpu.gpr[RAX] = selector as u64;
            })
        }

        fn load_tr(selector: u16) -> (Harness, Stop) {
            run_with(LTR_AX, |h| {
                install_gdt(h);
                h.cpu.gpr[RAX] = selector as u64;
            })
        }

        fn descriptor_type(flags: u32) -> u32 {
            (flags >> 8) & 0xf
        }

        #[test]
        fn the_test_descriptor_builder_matches_a_hand_encoded_tss_descriptor() {
            // limit 0x67, base 0xffff8000_12345000, present available 64-bit TSS.
            let by_hand = [
                0x67, 0x00, 0x00, 0x50, 0x34, 0x89, 0x00, 0x12, 0x00, 0x80, 0xff, 0xff, 0x00, 0x00,
                0x00, 0x00,
            ];
            assert_eq!(
                system_descriptor(TSS_BASE, TSS_LIMIT, ACCESS_TSS_AVAILABLE, 0),
                by_hand
            );
        }

        #[test]
        fn sldt_and_str_start_at_zero() {
            // sldt eax; str ebx; hlt
            let h = run_ok(&[0x0f, 0x00, 0xc0, 0x0f, 0x00, 0xcb, 0xf4], |h| {
                h.cpu.gpr[RAX] = u64::MAX;
                h.cpu.gpr[RBX] = u64::MAX;
            });
            assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RBX]), (0, 0));
        }

        #[test]
        fn sldt_register_form_follows_the_operand_size() {
            let setup = |h: &mut Harness| {
                h.cpu.ldtr.sel = 0x1238;
                h.cpu.gpr[RAX] = 0xdead_beef_cafe_f00d;
            };
            // sldt ax; hlt
            let h = run_ok(&[0x66, 0x0f, 0x00, 0xc0, 0xf4], setup);
            assert_eq!(
                h.cpu.gpr[RAX], 0xdead_beef_cafe_1238,
                "16-bit form keeps the upper bits"
            );
            // sldt eax; hlt
            let h = run_ok(&[0x0f, 0x00, 0xc0, 0xf4], setup);
            assert_eq!(h.cpu.gpr[RAX], 0x1238, "32-bit form zero-extends");
            // sldt rax; hlt
            let h = run_ok(&[0x48, 0x0f, 0x00, 0xc0, 0xf4], setup);
            assert_eq!(h.cpu.gpr[RAX], 0x1238);
        }

        #[test]
        fn str_register_form_follows_the_operand_size() {
            let setup = |h: &mut Harness| {
                h.cpu.tr.sel = 0x2468;
                h.cpu.gpr[RAX] = 0xdead_beef_cafe_f00d;
            };
            // str ax; hlt
            let h = run_ok(&[0x66, 0x0f, 0x00, 0xc8, 0xf4], setup);
            assert_eq!(h.cpu.gpr[RAX], 0xdead_beef_cafe_2468);
            // str eax; hlt
            let h = run_ok(&[0x0f, 0x00, 0xc8, 0xf4], setup);
            assert_eq!(h.cpu.gpr[RAX], 0x2468);
            // str rax; hlt
            let h = run_ok(&[0x48, 0x0f, 0x00, 0xc8, 0xf4], setup);
            assert_eq!(h.cpu.gpr[RAX], 0x2468);
        }

        #[test]
        fn sldt_and_str_memory_forms_store_exactly_two_bytes() {
            // sldt word ptr [rdi]; str word ptr [rdi + 2]; hlt
            let h = run_ok(&[0x0f, 0x00, 0x07, 0x0f, 0x00, 0x4f, 0x02, 0xf4], |h| {
                h.write(OUT, &[0xaa; 8]);
                h.cpu.ldtr.sel = 0x1234;
                h.cpu.tr.sel = 0x5678;
                h.cpu.gpr[RDI] = OUT;
            });
            assert_eq!(
                h.read(OUT, 8),
                [0x34, 0x12, 0x78, 0x56, 0xaa, 0xaa, 0xaa, 0xaa]
            );
        }

        #[test]
        fn sldt_and_str_are_allowed_at_cpl3() {
            // sldt ax; str bx; hlt
            let mut h = Harness::new(&[0x66, 0x0f, 0x00, 0xc0, 0x66, 0x0f, 0x00, 0xcb, 0xf4]);
            h.cpu.ldtr.sel = 0x18;
            h.cpu.tr.sel = 0x28;
            user_mode(&mut h);
            assert_eq!(h.run(2), Stop::Limit);
            assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RBX]), (0x18, 0x28));
        }

        #[test]
        fn lldt_loads_the_ldt_descriptor() {
            let (h, stop) = load_ldt(SEL_LDT);
            assert_eq!(stop, Stop::Halted);
            let ldtr = h.cpu.ldtr;
            assert_eq!(ldtr.sel, SEL_LDT);
            assert_eq!(ldtr.base, LDT_BASE);
            assert_eq!(ldtr.limit, LDT_LIMIT);
            assert!(ldtr.present());
            assert_eq!(ldtr.flags & SEG_NON_SYSTEM, 0, "a system descriptor");
            assert_eq!(descriptor_type(ldtr.flags), 2);
            assert_eq!(ldtr.dpl(), 0);
        }

        #[test]
        fn lldt_scales_a_page_granular_limit() {
            let (h, stop) = load_ldt(SEL_LDT_GRANULAR);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.ldtr.limit, (0xf << 12) | 0xfff);
        }

        #[test]
        fn lldt_decodes_the_upper_base_half_of_the_16_byte_descriptor() {
            let (h, stop) = run_with(LLDT_AX, |h| {
                install_gdt(h);
                h.write(
                    GDT + SEL_LDT as u64,
                    &system_descriptor(0xffff_8000_1234_5000, 0xff, ACCESS_LDT, 0),
                );
                h.cpu.gpr[RAX] = SEL_LDT as u64;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.ldtr.base, 0xffff_8000_1234_5000);
        }

        #[test]
        fn lldt_keeps_the_rpl_bits_in_the_visible_selector_but_ignores_them_for_lookup() {
            let (h, stop) = load_ldt(SEL_LDT | 3);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.ldtr.sel, SEL_LDT | 3);
            assert_eq!(h.cpu.ldtr.base, LDT_BASE);
        }

        #[test]
        fn lldt_reads_a_memory_operand() {
            let (h, stop) = run_with(LLDT_MEMORY, |h| {
                install_gdt(h);
                h.write(DATA, &SEL_LDT.to_le_bytes());
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.ldtr.sel, SEL_LDT);
        }

        #[test]
        fn lldt_memory_operand_page_fault_leaves_ldtr_alone() {
            let (h, stop) = run_with(LLDT_MEMORY, |h| {
                install_gdt(h);
                h.cpu.ldtr.sel = 0x99;
                unmap_region(h, REMAP_REGION);
                h.cpu.gpr[RSI] = REMAP_REGION + 0x10;
            });
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: VECTOR_PAGE_FAULT,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(h.cpu.cr2, REMAP_REGION + 0x10);
            assert_eq!(h.cpu.ldtr.sel, 0x99);
        }

        #[test]
        fn lldt_with_a_null_selector_invalidates_the_ldt() {
            for selector in 0u16..4 {
                let (h, stop) = run_with(LLDT_AX, |h| {
                    install_gdt(h);
                    h.cpu.ldtr = Seg {
                        sel: SEL_LDT,
                        base: LDT_BASE,
                        limit: LDT_LIMIT,
                        flags: SEG_PRESENT,
                    };
                    h.cpu.gpr[RAX] = selector as u64;
                });
                assert_eq!(stop, Stop::Halted, "selector {selector}");
                let ldtr = h.cpu.ldtr;
                assert_eq!(
                    (ldtr.base, ldtr.limit, ldtr.flags),
                    (0, 0, 0),
                    "selector {selector}"
                );
                assert_eq!(
                    ldtr.sel, selector,
                    "sldt must read back the null selector that was loaded"
                );
            }
        }

        #[test]
        fn lldt_after_a_valid_load_can_be_replaced_by_another() {
            // lldt ax; mov ax, 0x90; lldt ax; hlt
            let (h, stop) = run_with(
                &[
                    0x0f, 0x00, 0xd0, 0x66, 0xb8, 0x90, 0x00, 0x0f, 0x00, 0xd0, 0xf4,
                ],
                |h| {
                    install_gdt(h);
                    h.cpu.gpr[RAX] = SEL_LDT as u64;
                },
            );
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.ldtr.sel, SEL_LDT_GRANULAR);
        }

        #[test]
        fn lldt_selector_in_the_ldt_is_gp_with_the_selector_as_error_code() {
            let (h, stop) = load_ldt(0x0c);
            assert_eq!(stop, general_protection(0x0c));
            assert_eq!(h.cpu.rip, CODE_BASE);
            let (_, stop) = load_ldt(0x0f);
            assert_eq!(
                stop,
                general_protection(0x0c),
                "RPL bits are not part of the error code"
            );
        }

        #[test]
        fn lldt_beyond_the_gdt_limit_is_gp() {
            let (_, stop) = load_ldt(0xb0);
            assert_eq!(stop, general_protection(0xb0));
            let (_, stop) = load_ldt(0xfff8);
            assert_eq!(stop, general_protection(0xfff8));
        }

        #[test]
        fn lldt_needs_the_whole_16_byte_descriptor_inside_the_gdt_limit() {
            let (h, stop) = run_with(LLDT_AX, |h| {
                install_gdt(h);
                h.cpu.gdtr.limit = SEL_LDT + 7;
                h.cpu.ldtr.sel = 0x99;
                h.cpu.gpr[RAX] = SEL_LDT as u64;
            });
            assert_eq!(stop, general_protection(SEL_LDT as u32));
            assert_eq!(h.cpu.ldtr.sel, 0x99);
            let (_, stop) = run_with(LLDT_AX, |h| {
                install_gdt(h);
                h.cpu.gdtr.limit = SEL_LDT + 15;
                h.cpu.gpr[RAX] = SEL_LDT as u64;
            });
            assert_eq!(stop, Stop::Halted, "limit is inclusive");
        }

        #[test]
        fn lldt_of_a_code_or_data_descriptor_is_gp() {
            for selector in [SEL_DATA_RW_DPL0, SEL_CODE_XR_DPL0, SEL_DATA_RO_DPL0] {
                let (h, stop) = load_ldt(selector);
                assert_eq!(
                    stop,
                    general_protection(selector as u32),
                    "selector {selector:#x}"
                );
                assert_eq!(h.cpu.ldtr.sel, 0, "selector {selector:#x}");
            }
        }

        #[test]
        fn lldt_of_a_tss_descriptor_is_gp() {
            for selector in [SEL_TSS, SEL_TSS_BUSY] {
                let (_, stop) = load_ldt(selector);
                assert_eq!(
                    stop,
                    general_protection(selector as u32),
                    "selector {selector:#x}"
                );
            }
        }

        #[test]
        fn lldt_of_a_non_present_ldt_is_np() {
            let (h, stop) = load_ldt(SEL_LDT_ABSENT);
            assert_eq!(stop, not_present(SEL_LDT_ABSENT as u32));
            assert_eq!(h.cpu.ldtr.sel, 0);
            let (_, stop) = load_ldt(SEL_LDT_ABSENT | 3);
            assert_eq!(
                stop,
                not_present(SEL_LDT_ABSENT as u32),
                "RPL is stripped from the error code"
            );
        }

        #[test]
        fn the_type_check_comes_before_the_present_check() {
            let (_, stop) = run_with(LLDT_AX, |h| {
                install_gdt(h);
                h.write(
                    GDT + SEL_TSS_ABSENT as u64,
                    &system_descriptor(0, 0x67, ACCESS_TSS_AVAILABLE & !ACCESS_PRESENT, 0),
                );
                h.cpu.gpr[RAX] = SEL_TSS_ABSENT as u64;
            });
            assert_eq!(stop, general_protection(SEL_TSS_ABSENT as u32));
        }

        #[test]
        fn lldt_at_cpl3_is_gp0_before_anything_is_read() {
            let (h, stop) = run_with(LLDT_AX, |h| {
                install_gdt(h);
                user_mode(h);
                h.cpu.gpr[RAX] = SEL_LDT as u64;
            });
            assert_eq!(stop, general_protection(0));
            assert_eq!(h.cpu.ldtr.sel, 0);
            let (_, stop) = run_with(LLDT_AX, |h| {
                user_mode(h);
                h.cpu.gpr[RAX] = 0xffff;
            });
            assert_eq!(
                stop,
                general_protection(0),
                "even a bad selector reports the privilege fault"
            );
        }

        #[test]
        fn ltr_loads_the_task_register() {
            let (h, stop) = load_tr(SEL_TSS);
            assert_eq!(stop, Stop::Halted);
            let tr = h.cpu.tr;
            assert_eq!(tr.sel, SEL_TSS);
            assert_eq!(tr.base, TSS_BASE);
            assert_eq!(tr.limit, TSS_LIMIT);
            assert!(tr.present());
            assert_eq!(tr.flags & SEG_NON_SYSTEM, 0);
        }

        #[test]
        fn ltr_marks_the_tss_busy_in_the_gdt_and_the_cached_descriptor() {
            let (h, stop) = load_tr(SEL_TSS);
            assert_eq!(stop, Stop::Halted);
            assert_eq!(
                descriptor_type(h.cpu.tr.flags),
                0xb,
                "cached type is busy 64-bit TSS"
            );
            let after = h.read(GDT + SEL_TSS as u64, 16);
            let mut expected = system_descriptor(TSS_BASE, TSS_LIMIT, ACCESS_TSS_AVAILABLE, 0);
            expected[5] = ACCESS_TSS_BUSY;
            assert_eq!(
                after, expected,
                "only the type byte changes: 0x89 becomes 0x8b"
            );
        }

        #[test]
        fn ltr_does_not_touch_neighbouring_descriptors() {
            let (h, _) = run_with(LTR_AX, |h| {
                install_gdt(h);
                h.cpu.gpr[RAX] = SEL_TSS as u64;
            });
            let mut reference = Harness::new(&[crate::testkit::HLT]);
            install_gdt(&mut reference);
            for offset in (0..GDT_LIMIT as u64 + 1)
                .filter(|o| !(SEL_TSS as u64..SEL_TSS as u64 + 16).contains(o))
            {
                assert_eq!(
                    h.read(GDT + offset, 1),
                    reference.read(GDT + offset, 1),
                    "offset {offset:#x}"
                );
            }
        }

        #[test]
        fn loading_the_same_tss_twice_is_gp_because_it_is_now_busy() {
            // ltr ax; ltr ax; hlt
            let (h, stop) = run_with(&[0x0f, 0x00, 0xd8, 0x0f, 0x00, 0xd8, 0xf4], |h| {
                install_gdt(h);
                h.cpu.gpr[RAX] = SEL_TSS as u64;
            });
            assert_eq!(stop, general_protection(SEL_TSS as u32));
            assert_eq!(
                h.cpu.rip,
                CODE_BASE + 3,
                "the second ltr is the faulting one"
            );
            assert_eq!(h.cpu.tr.sel, SEL_TSS, "the first load stays in effect");
        }

        #[test]
        fn ltr_of_a_busy_tss_is_gp() {
            let (h, stop) = load_tr(SEL_TSS_BUSY);
            assert_eq!(stop, general_protection(SEL_TSS_BUSY as u32));
            assert_eq!(h.cpu.tr.sel, 0);
        }

        #[test]
        fn ltr_with_a_null_selector_is_gp0() {
            for selector in 0u16..4 {
                let (h, stop) = load_tr(selector);
                assert_eq!(stop, general_protection(0), "selector {selector}");
                assert_eq!(h.cpu.tr.sel, 0);
            }
        }

        #[test]
        fn ltr_selector_in_the_ldt_is_gp() {
            let (_, stop) = load_tr(0x1c);
            assert_eq!(stop, general_protection(0x1c));
        }

        #[test]
        fn ltr_beyond_the_gdt_limit_is_gp() {
            let (_, stop) = load_tr(0xb0);
            assert_eq!(stop, general_protection(0xb0));
        }

        #[test]
        fn ltr_needs_the_whole_16_byte_descriptor_inside_the_gdt_limit() {
            let (h, stop) = run_with(LTR_AX, |h| {
                install_gdt(h);
                h.cpu.gdtr.limit = SEL_TSS + 7;
                h.cpu.gpr[RAX] = SEL_TSS as u64;
            });
            assert_eq!(stop, general_protection(SEL_TSS as u32));
            assert_eq!(
                h.read(GDT + SEL_TSS as u64 + 5, 1),
                [ACCESS_TSS_AVAILABLE],
                "still available"
            );
        }

        #[test]
        fn ltr_of_the_wrong_descriptor_kind_is_gp_and_changes_nothing() {
            for selector in [SEL_LDT, SEL_DATA_RW_DPL0, SEL_CODE_XR_DPL0] {
                let (h, stop) = load_tr(selector);
                assert_eq!(
                    stop,
                    general_protection(selector as u32),
                    "selector {selector:#x}"
                );
                assert_eq!(h.cpu.tr.sel, 0);
                let mut reference = Harness::new(&[crate::testkit::HLT]);
                install_gdt(&mut reference);
                assert_eq!(
                    h.read(GDT, GDT_LIMIT as usize + 1),
                    reference.read(GDT, GDT_LIMIT as usize + 1)
                );
            }
        }

        #[test]
        fn ltr_of_a_non_present_tss_is_np_and_does_not_set_busy() {
            let (h, stop) = load_tr(SEL_TSS_ABSENT);
            assert_eq!(stop, not_present(SEL_TSS_ABSENT as u32));
            assert_eq!(
                h.read(GDT + SEL_TSS_ABSENT as u64 + 5, 1),
                [ACCESS_TSS_AVAILABLE & !ACCESS_PRESENT]
            );
            assert_eq!(h.cpu.tr.sel, 0);
        }

        #[test]
        fn ltr_error_codes_drop_the_rpl_bits() {
            let (_, stop) = load_tr(SEL_TSS_BUSY | 3);
            assert_eq!(stop, general_protection(SEL_TSS_BUSY as u32));
            let (_, stop) = load_tr(SEL_TSS_ABSENT | 2);
            assert_eq!(stop, not_present(SEL_TSS_ABSENT as u32));
        }

        #[test]
        fn ltr_accepts_a_selector_with_rpl_and_reports_it_back_through_str() {
            // ltr ax; str bx; hlt
            let (h, stop) = run_with(&[0x0f, 0x00, 0xd8, 0x66, 0x0f, 0x00, 0xcb, 0xf4], |h| {
                install_gdt(h);
                h.cpu.gpr[RAX] = (SEL_TSS | 3) as u64;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.gpr[RBX], (SEL_TSS | 3) as u64);
        }

        #[test]
        fn ltr_reads_a_memory_operand() {
            let (h, stop) = run_with(LTR_MEMORY, |h| {
                install_gdt(h);
                h.write(DATA, &SEL_TSS.to_le_bytes());
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!(stop, Stop::Halted);
            assert_eq!(h.cpu.tr.sel, SEL_TSS);
        }

        #[test]
        fn ltr_at_cpl3_is_gp0_and_leaves_the_tss_available() {
            let (h, stop) = run_with(LTR_AX, |h| {
                install_gdt(h);
                user_mode(h);
                h.cpu.gpr[RAX] = SEL_TSS as u64;
            });
            assert_eq!(stop, general_protection(0));
            assert_eq!(h.read(GDT + SEL_TSS as u64 + 5, 1), [ACCESS_TSS_AVAILABLE]);
        }
    }

    mod segment_verification {
        use super::*;

        // verr ax; hlt
        const VERR_AX: &[u8] = &[0x0f, 0x00, 0xe0, 0xf4];
        // verw ax; hlt
        const VERW_AX: &[u8] = &[0x0f, 0x00, 0xe8, 0xf4];
        const LDT_TABLE: u64 = SCRATCH + 0x800;
        const LDT_SELECTOR: u16 = 4;

        /// Result of one verr/verw at `cpl` for `selector`: whether ZF ended up set.
        fn verify(selector: u16, cpl: u8, write: bool) -> bool {
            let mut h = Harness::new(if write { VERW_AX } else { VERR_AX });
            install_gdt(&mut h);
            if cpl == 3 {
                user_mode(&mut h);
            }
            h.cpu.gpr[RAX] = selector as u64;
            h.cpu.set_eflags_arith(0);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(
                h.cpu.rip,
                CODE_BASE + 3,
                "verr/verw never fault on a bad selector"
            );
            h.cpu.eflags_full() & ZF != 0
        }

        fn readable(selector: u16, cpl: u8) -> bool {
            verify(selector, cpl, false)
        }

        fn writable(selector: u16, cpl: u8) -> bool {
            verify(selector, cpl, true)
        }

        #[test]
        fn a_null_selector_is_neither_readable_nor_writable() {
            for selector in 0..4 {
                assert!(!readable(selector, 0), "{selector}");
                assert!(!writable(selector, 0), "{selector}");
            }
        }

        #[test]
        fn writable_data_is_readable_and_writable() {
            assert!(readable(SEL_DATA_RW_DPL0, 0));
            assert!(writable(SEL_DATA_RW_DPL0, 0));
        }

        #[test]
        fn read_only_data_is_readable_but_not_writable() {
            assert!(readable(SEL_DATA_RO_DPL0, 0));
            assert!(!writable(SEL_DATA_RO_DPL0, 0));
        }

        #[test]
        fn execute_only_code_is_neither() {
            assert!(!readable(SEL_CODE_X_DPL0, 0));
            assert!(!writable(SEL_CODE_X_DPL0, 0));
        }

        #[test]
        fn readable_code_is_readable_but_never_writable() {
            for selector in [
                SEL_CODE_XR_DPL0,
                SEL_CODE_XR_CONFORMING_DPL0,
                SEL_CODE_XR_DPL3,
            ] {
                assert!(readable(selector, 0), "{selector:#x}");
                assert!(!writable(selector, 0), "{selector:#x}");
            }
        }

        #[test]
        fn system_descriptors_are_not_accessible() {
            for selector in [SEL_LDT, SEL_TSS, SEL_TSS_BUSY, SEL_LDT_GRANULAR] {
                assert!(!readable(selector, 0), "{selector:#x}");
                assert!(!writable(selector, 0), "{selector:#x}");
            }
        }

        #[test]
        fn selectors_beyond_the_gdt_limit_are_not_accessible_and_do_not_fault() {
            for selector in [0xb0, 0x1000, 0xfff8] {
                assert!(!readable(selector, 0), "{selector:#x}");
                assert!(!writable(selector, 0), "{selector:#x}");
            }
        }

        #[test]
        fn an_ldt_selector_with_no_ldt_is_not_accessible() {
            assert!(!readable(LDT_SELECTOR, 0));
            assert!(!writable(LDT_SELECTOR | 8, 0));
        }

        #[test]
        fn dpl_is_checked_against_the_larger_of_cpl_and_rpl() {
            // CPL 0 with an RPL 3 selector: DPL 0 data is too privileged, DPL 3 data is fine.
            assert!(!readable(SEL_DATA_RW_DPL0 | 3, 0));
            assert!(!writable(SEL_DATA_RW_DPL0 | 3, 0));
            assert!(readable(SEL_DATA_RW_DPL3 | 3, 0));
            assert!(writable(SEL_DATA_RW_DPL3 | 3, 0));
            // CPL 3: DPL 0 is off limits even with RPL 0.
            assert!(!readable(SEL_DATA_RW_DPL0, 3));
            assert!(!writable(SEL_DATA_RW_DPL0, 3));
            assert!(readable(SEL_DATA_RW_DPL3, 3));
            assert!(writable(SEL_DATA_RW_DPL3, 3));
        }

        #[test]
        fn nonconforming_readable_code_obeys_the_dpl_rule() {
            assert!(!readable(SEL_CODE_XR_DPL0 | 3, 0));
            assert!(!readable(SEL_CODE_XR_DPL0, 3));
            assert!(readable(SEL_CODE_XR_DPL3, 3));
            assert!(readable(SEL_CODE_XR_DPL3 | 3, 0));
        }

        #[test]
        fn conforming_readable_code_ignores_the_dpl_rule() {
            assert!(readable(SEL_CODE_XR_CONFORMING_DPL0 | 3, 0));
            assert!(readable(SEL_CODE_XR_CONFORMING_DPL0, 3));
            assert!(readable(SEL_CODE_XR_CONFORMING_DPL0 | 3, 3));
        }

        #[test]
        fn selectors_in_the_ldt_use_the_ldt_descriptors() {
            let mut h = Harness::new(VERR_AX);
            install_gdt(&mut h);
            h.write(LDT_TABLE, &[0; 8]);
            h.write(LDT_TABLE + 8, &descriptor(0, 0xfffff, 0x92, 0xc));
            h.write(LDT_TABLE + 16, &descriptor(0, 0xfffff, 0x98, 0xa));
            h.cpu.ldtr = Seg {
                sel: 0x10,
                base: LDT_TABLE,
                limit: 0x17,
                flags: SEG_PRESENT,
            };
            for (selector, expected) in [(0x0c, true), (0x14, false), (0x1c, false)] {
                h.cpu.gpr[RAX] = selector as u64;
                h.cpu.rip = CODE_BASE;
                h.cpu.set_eflags_arith(0);
                assert_eq!(h.run(1), Stop::Limit);
                assert_eq!(
                    h.cpu.eflags_full() & ZF != 0,
                    expected,
                    "selector {selector:#x}"
                );
            }
        }

        #[test]
        fn only_zf_changes() {
            let mut h = Harness::new(VERR_AX);
            install_gdt(&mut h);
            h.cpu.gpr[RAX] = SEL_DATA_RW_DPL0 as u64;
            h.cpu.set_eflags_arith(CF | SF | OF | AF | PF);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.eflags_full() & ARITH, CF | SF | OF | AF | PF | ZF);
            let mut h = Harness::new(VERR_AX);
            install_gdt(&mut h);
            h.cpu.gpr[RAX] = SEL_CODE_X_DPL0 as u64;
            h.cpu.set_eflags_arith(CF | ZF | OF | PF);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(
                h.cpu.eflags_full() & ARITH,
                CF | OF | PF,
                "a failed check clears ZF"
            );
        }

        #[test]
        fn the_selector_can_come_from_memory() {
            // verw word ptr [rsi]; hlt
            let mut h = Harness::new(&[0x0f, 0x00, 0x2e, 0xf4]);
            install_gdt(&mut h);
            h.write(DATA, &SEL_DATA_RW_DPL0.to_le_bytes());
            h.cpu.gpr[RSI] = DATA;
            h.cpu.set_eflags_arith(0);
            assert_eq!(h.run(1), Stop::Limit);
            assert_ne!(h.cpu.eflags_full() & ZF, 0);
            h.write(DATA, &SEL_DATA_RO_DPL0.to_le_bytes());
            h.cpu.rip = CODE_BASE;
            h.cpu.set_eflags_arith(ZF);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.eflags_full() & ZF, 0);
        }

        #[test]
        fn a_memory_operand_page_fault_is_reported() {
            // verr word ptr [rsi]; hlt
            let (h, stop) = run_with(&[0x0f, 0x00, 0x26, 0xf4], |h| {
                install_gdt(h);
                unmap_region(h, REMAP_REGION);
                h.cpu.gpr[RSI] = REMAP_REGION;
            });
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: VECTOR_PAGE_FAULT,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(h.cpu.cr2, REMAP_REGION);
        }

        #[test]
        fn only_the_low_16_bits_of_a_register_operand_are_the_selector() {
            let mut h = Harness::new(VERR_AX);
            install_gdt(&mut h);
            h.cpu.gpr[RAX] = 0xffff_ffff_0000_0000 | SEL_DATA_RW_DPL0 as u64;
            h.cpu.set_eflags_arith(0);
            assert_eq!(h.run(1), Stop::Limit);
            assert_ne!(h.cpu.eflags_full() & ZF, 0);
        }
    }

    mod descriptor_tables {
        use super::*;

        // lgdt [rsi]; hlt
        const LGDT: &[u8] = &[0x0f, 0x01, 0x16, 0xf4];
        // lidt [rsi]; hlt
        const LIDT: &[u8] = &[0x0f, 0x01, 0x1e, 0xf4];
        // sgdt [rdi]; hlt
        const SGDT: &[u8] = &[0x0f, 0x01, 0x07, 0xf4];
        // sidt [rdi]; hlt
        const SIDT: &[u8] = &[0x0f, 0x01, 0x0f, 0xf4];
        const TABLE_BASE: u64 = 0xffff_8000_1234_5678;

        fn pseudo_descriptor(limit: u16, base: u64) -> [u8; 10] {
            let mut bytes = [0u8; 10];
            bytes[..2].copy_from_slice(&limit.to_le_bytes());
            bytes[2..].copy_from_slice(&base.to_le_bytes());
            bytes
        }

        #[test]
        fn sgdt_stores_a_2_byte_limit_then_an_8_byte_base() {
            let h = run_ok(SGDT, |h| {
                h.write(OUT, &[0xaa; 16]);
                h.cpu.gdtr = DescriptorTable {
                    base: TABLE_BASE,
                    limit: 0x1ff,
                };
                h.cpu.gpr[RDI] = OUT;
            });
            let mut expected = pseudo_descriptor(0x1ff, TABLE_BASE).to_vec();
            expected.extend_from_slice(&[0xaa; 6]);
            assert_eq!(h.read(OUT, 16), expected, "exactly ten bytes are written");
        }

        #[test]
        fn sidt_stores_the_idt_register() {
            let h = run_ok(SIDT, |h| {
                h.write(OUT, &[0xaa; 16]);
                h.cpu.idtr = DescriptorTable {
                    base: TABLE_BASE,
                    limit: 0xfff,
                };
                h.cpu.gpr[RDI] = OUT;
            });
            assert_eq!(h.read(OUT, 10), pseudo_descriptor(0xfff, TABLE_BASE));
            assert_eq!(h.read(OUT + 10, 6), [0xaa; 6]);
        }

        #[test]
        fn lgdt_and_lidt_load_limit_and_base() {
            let h = run_ok(LGDT, |h| {
                h.write(DATA, &pseudo_descriptor(0x7f, TABLE_BASE));
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!((h.cpu.gdtr.limit, h.cpu.gdtr.base), (0x7f, TABLE_BASE));
            let h = run_ok(LIDT, |h| {
                h.write(DATA, &pseudo_descriptor(0xfff, 0x1000));
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!((h.cpu.idtr.limit, h.cpu.idtr.base), (0xfff, 0x1000));
        }

        #[test]
        fn lgdt_does_not_disturb_the_idtr_and_vice_versa() {
            let h = run_ok(LGDT, |h| {
                h.write(DATA, &pseudo_descriptor(0x7f, 0x2000));
                h.cpu.idtr = DescriptorTable {
                    base: 0x4000,
                    limit: 9,
                };
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!((h.cpu.idtr.base, h.cpu.idtr.limit), (0x4000, 9));
            let h = run_ok(LIDT, |h| {
                h.write(DATA, &pseudo_descriptor(0x7f, 0x2000));
                h.cpu.gpr[RSI] = DATA;
            });
            let boot_gdtr = Harness::new(LIDT).cpu.gdtr;
            assert_eq!(
                (h.cpu.gdtr.base, h.cpu.gdtr.limit),
                (boot_gdtr.base, boot_gdtr.limit)
            );
            assert_eq!((h.cpu.idtr.base, h.cpu.idtr.limit), (0x2000, 0x7f));
        }

        #[test]
        fn lgdt_then_sgdt_round_trips() {
            // lgdt [rsi]; sgdt [rdi]; hlt
            let h = run_ok(&[0x0f, 0x01, 0x16, 0x0f, 0x01, 0x07, 0xf4], |h| {
                h.write(DATA, &pseudo_descriptor(0x1234, 0x5678_9000));
                h.cpu.gpr[RSI] = DATA;
                h.cpu.gpr[RDI] = OUT;
            });
            assert_eq!(h.read(OUT, 10), pseudo_descriptor(0x1234, 0x5678_9000));
        }

        #[test]
        fn lgdt_reads_exactly_ten_bytes() {
            // The operand ends right below an unmapped 2 MiB region: an 11th byte read would fault.
            let h = run_ok(LGDT, |h| {
                unmap_region(h, REMAP_REGION);
                h.write(REMAP_REGION - 10, &pseudo_descriptor(0x55, 0x6000));
                h.cpu.gpr[RSI] = REMAP_REGION - 10;
            });
            assert_eq!((h.cpu.gdtr.limit, h.cpu.gdtr.base), (0x55, 0x6000));
        }

        #[test]
        fn sgdt_writes_exactly_ten_bytes() {
            let h = run_ok(SGDT, |h| {
                unmap_region(h, REMAP_REGION);
                h.cpu.gpr[RDI] = REMAP_REGION - 10;
            });
            assert_eq!(
                h.read(REMAP_REGION - 10, 10),
                pseudo_descriptor(h.cpu.gdtr.limit, h.cpu.gdtr.base)
            );
        }

        #[test]
        fn lgdt_page_fault_leaves_gdtr_alone() {
            let (h, stop) = run_with(LGDT, |h| {
                unmap_region(h, REMAP_REGION);
                h.write(REMAP_REGION - 4, &[0xff; 4]);
                h.cpu.gpr[RSI] = REMAP_REGION - 4;
            });
            let before = Harness::new(LGDT).cpu.gdtr;
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: VECTOR_PAGE_FAULT,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(
                (h.cpu.gdtr.base, h.cpu.gdtr.limit),
                (before.base, before.limit)
            );
        }

        #[test]
        fn lgdt_and_lidt_at_cpl3_are_gp0() {
            for code in [LGDT, LIDT] {
                let h = assert_general_protection_zero(code, |h| {
                    user_mode(h);
                    h.write(DATA, &pseudo_descriptor(1, 2));
                    h.cpu.gpr[RSI] = DATA;
                });
                assert_ne!(h.cpu.idtr.base, 2);
                assert_ne!(h.cpu.gdtr.base, 2);
            }
        }

        #[test]
        fn sgdt_and_sidt_work_at_cpl3() {
            for (code, expected) in [
                (SGDT, pseudo_descriptor(0x22, 0x1111)),
                (SIDT, pseudo_descriptor(0x44, 0x3333)),
            ] {
                let mut h = Harness::new(code);
                user_mode(&mut h);
                h.cpu.gpr[RDI] = OUT;
                h.cpu.gdtr = DescriptorTable {
                    base: 0x1111,
                    limit: 0x22,
                };
                h.cpu.idtr = DescriptorTable {
                    base: 0x3333,
                    limit: 0x44,
                };
                assert_eq!(h.run(1), Stop::Limit);
                assert_eq!(h.read(OUT, 10), expected);
            }
        }

        #[test]
        fn register_forms_of_group_7_are_not_table_loads() {
            // 0f 01 d0 is xgetbv, 0f 01 d8 vmrun, 0f 01 c1 vmcall: none implemented.
            for code in [
                // xgetbv; hlt
                &[0x0f, 0x01, 0xd0, 0xf4],
                // .byte 0x0f, 0x01, 0xd8; hlt
                &[0x0f, 0x01, 0xd8, 0xf4],
                // .byte 0x0f, 0x01, 0xc1; hlt
                &[0x0f, 0x01, 0xc1, 0xf4],
                // .byte 0x0f, 0x01, 0xc0; hlt
                &[0x0f, 0x01, 0xc0, 0xf4],
            ] {
                assert_invalid_opcode(code, |_| {});
            }
        }
    }

    mod machine_status_word {
        use super::*;

        // smsw eax; hlt
        const SMSW_EAX: &[u8] = &[0x0f, 0x01, 0xe0, 0xf4];

        #[test]
        fn smsw_32_bit_register_form_returns_cr0_low_dword_zero_extended() {
            let h = run_ok(SMSW_EAX, |h| h.cpu.gpr[RAX] = u64::MAX);
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR0);
        }

        #[test]
        fn smsw_64_bit_register_form_returns_all_of_cr0() {
            // smsw rax; hlt
            let h = run_ok(&[0x48, 0x0f, 0x01, 0xe0, 0xf4], |h| {
                h.cpu.cr0 = BOOT_CR0 | 1 << 32
            });
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR0 | 1 << 32);
        }

        #[test]
        fn smsw_16_bit_register_form_keeps_the_upper_bits() {
            // smsw ax; hlt
            let h = run_ok(&[0x66, 0x0f, 0x01, 0xe0, 0xf4], |h| {
                h.cpu.gpr[RAX] = 0xdead_beef_cafe_0000
            });
            assert_eq!(h.cpu.gpr[RAX], 0xdead_beef_cafe_0000 | (BOOT_CR0 & 0xffff));
        }

        #[test]
        fn smsw_memory_form_stores_two_bytes() {
            // smsw word ptr [rdi]; hlt
            let h = run_ok(&[0x0f, 0x01, 0x27, 0xf4], |h| {
                h.write(OUT, &[0xaa; 8]);
                h.cpu.gpr[RDI] = OUT;
            });
            assert_eq!(
                h.read(OUT, 8),
                [0x33, 0x00, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa]
            );
        }

        #[test]
        fn smsw_is_allowed_at_cpl3() {
            let mut h = Harness::new(SMSW_EAX);
            user_mode(&mut h);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RAX], BOOT_CR0);
        }

        #[test]
        fn lmsw_loads_mp_em_and_ts() {
            // lmsw ax; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf0, 0xf4], |h| h.cpu.gpr[RAX] = 0xf);
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_EM_BIT | CR0_TS_BIT);
        }

        #[test]
        fn lmsw_can_clear_mp_but_never_pe() {
            // lmsw ax; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf0, 0xf4], |h| h.cpu.gpr[RAX] = 0);
            assert_eq!(
                h.cpu.cr0,
                BOOT_CR0 & !CR0_MP_BIT,
                "PE stays set, MP follows the operand"
            );
        }

        #[test]
        fn lmsw_clears_em_and_ts_when_the_operand_has_them_clear() {
            // lmsw ax; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf0, 0xf4], |h| {
                h.cpu.cr0 |= CR0_EM_BIT | CR0_TS_BIT;
                h.cpu.gpr[RAX] = CR0_MP_BIT | CR0_PE_BIT;
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn lmsw_only_touches_the_low_four_bits_and_ignores_the_rest_of_the_operand() {
            // lmsw ax; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf0, 0xf4], |h| {
                h.cpu.gpr[RAX] = 0xffff_ffff_ffff_fff1
            });
            assert_eq!(
                h.cpu.cr0,
                BOOT_CR0 & !CR0_MP_BIT,
                "ET, NE, PG and every other bit are untouched"
            );
        }

        #[test]
        fn lmsw_does_not_touch_paging_or_write_protect() {
            // lmsw ax; hlt
            let h = run_ok(&[0x0f, 0x01, 0xf0, 0xf4], |h| {
                h.cpu.cr0 |= CR0_WP_BIT;
                h.cpu.gpr[RAX] = 0;
            });
            assert_eq!(
                h.cpu.cr0 & (CR0_PG_BIT | CR0_WP_BIT),
                CR0_PG_BIT | CR0_WP_BIT
            );
        }

        #[test]
        fn lmsw_reads_a_memory_operand() {
            // lmsw word ptr [rsi]; hlt
            let h = run_ok(&[0x0f, 0x01, 0x36, 0xf4], |h| {
                h.write(DATA, &0x000cu16.to_le_bytes());
                h.cpu.gpr[RSI] = DATA;
            });
            assert_eq!(
                h.cpu.cr0,
                (BOOT_CR0 & !CR0_MP_BIT) | CR0_EM_BIT | CR0_TS_BIT
            );
        }

        #[test]
        fn lmsw_at_cpl3_is_gp0() {
            // lmsw ax; hlt
            let h = assert_general_protection_zero(&[0x0f, 0x01, 0xf0, 0xf4], |h| {
                user_mode(h);
                h.cpu.gpr[RAX] = 0xf;
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn clts_clears_ts_and_only_ts() {
            // clts; hlt
            let h = run_ok(&[0x0f, 0x06, 0xf4], |h| {
                h.cpu.cr0 |= CR0_TS_BIT | CR0_EM_BIT
            });
            assert_eq!(h.cpu.cr0, BOOT_CR0 | CR0_EM_BIT);
        }

        #[test]
        fn clts_with_ts_already_clear_changes_nothing() {
            // clts; hlt
            let h = run_ok(&[0x0f, 0x06, 0xf4], |_| {});
            assert_eq!(h.cpu.cr0, BOOT_CR0);
        }

        #[test]
        fn clts_at_cpl3_is_gp0() {
            // clts; hlt
            let h = assert_general_protection_zero(&[0x0f, 0x06, 0xf4], |h| {
                user_mode(h);
                h.cpu.cr0 |= CR0_TS_BIT;
            });
            assert_ne!(h.cpu.cr0 & CR0_TS_BIT, 0);
        }

        #[test]
        fn jit_and_interpreter_agree_on_smsw_lmsw_clts() {
            crate::testkit::assert_jit_matches_interpreter(
                // smsw eax; mov ebx, 0xc; lmsw bx; smsw ecx; clts; smsw edx; hlt
                &[
                    0x0f, 0x01, 0xe0, 0xbb, 0x0c, 0x00, 0x00, 0x00, 0x0f, 0x01, 0xf3, 0x0f, 0x01,
                    0xe1, 0x0f, 0x06, 0x0f, 0x01, 0xe2, 0xf4,
                ],
                |_| {},
            );
        }
    }

    mod cache_and_tlb_control {
        use super::*;

        #[test]
        fn invlpg_drops_a_stale_translation() {
            // Read through a mapping, repoint the page-directory entry without invalidating, and
            // read again: the TLB still serves the old frame until invlpg.
            // mov al, [rsi]; mov bl, [rsi]; invlpg [rsi]; mov dl, [rsi]; hlt
            let mut h = Harness::new(&[0x8a, 0x06, 0x8a, 0x1e, 0x0f, 0x01, 0x3e, 0x8a, 0x16, 0xf4]);
            h.write(REMAP_REGION + 0x10, &[0x11]);
            h.write(OTHER_REGION + 0x10, &[0x22]);
            h.cpu.gpr[RSI] = REMAP_REGION + 0x10;
            let entry = pd_entry_address(&h, REMAP_REGION);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RAX] & 0xff, 0x11);
            h.write_u64(entry, OTHER_REGION | PTE_PRESENT_WRITABLE_HUGE);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RBX] & 0xff, 0x11, "still the stale translation");
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RDX] & 0xff, 0x22, "invlpg forced a fresh walk");
        }

        #[test]
        fn invlpg_only_reads_nothing_and_does_not_fault_on_an_unmapped_address() {
            // invlpg [rsi]; hlt
            let h = run_ok(&[0x0f, 0x01, 0x3e, 0xf4], |h| {
                unmap_region(h, REMAP_REGION);
                h.cpu.gpr[RSI] = REMAP_REGION;
            });
            assert_eq!(h.cpu.rip, CODE_BASE + 4);
        }

        #[test]
        fn invlpg_at_cpl3_is_gp0() {
            // invlpg [rsi]; hlt
            assert_general_protection_zero(&[0x0f, 0x01, 0x3e, 0xf4], user_mode);
        }

        #[test]
        fn wbinvd_and_invd_are_privileged_no_ops() {
            // wbinvd; hlt
            // invd; hlt
            for code in [&[0x0f, 0x09, 0xf4], &[0x0f, 0x08, 0xf4]] {
                let reference = Harness::new(code).snapshot();
                let h = run_ok(code, |_| {});
                let mut expected = reference;
                expected.rip = h.cpu.rip;
                expected.icount = h.cpu.icount;
                assert_eq!(h.snapshot(), expected);
            }
        }

        #[test]
        fn wbnoinvd_is_wbinvd_without_the_invalidate() {
            // f3 0f 09 (wbnoinvd); hlt: privileged like wbinvd, and nothing else happens
            run_ok(&[0xf3, 0x0f, 0x09, 0xf4], |_| {});
            assert_general_protection_zero(&[0xf3, 0x0f, 0x09, 0xf4], user_mode);
        }

        #[test]
        fn clzero_zeroes_the_aligned_cache_line_around_rax_at_any_cpl() {
            // clzero; hlt
            const CLZERO: &[u8] = &[0x0f, 0x01, 0xfc, 0xf4];
            const CLZERO_ADDRESS32: &[u8] = &[0x67, 0x0f, 0x01, 0xfc, 0xf4];
            let fill = |h: &mut Harness| h.write(DATA, &[0xaa; 192]);
            let h = run_ok(CLZERO, |h| {
                fill(h);
                h.cpu.gpr[RAX] = DATA + 64 + 0x25;
            });
            let bytes = h.read(DATA, 192);
            assert_eq!(bytes[..64], [0xaa; 64]);
            assert_eq!(bytes[64..128], [0; 64], "exactly the line holding rax");
            assert_eq!(bytes[128..], [0xaa; 64]);
            // the 0x67 prefix truncates the address to 32 bits
            let h = run_ok(CLZERO_ADDRESS32, |h| {
                fill(h);
                h.cpu.gpr[RAX] = 0xffff_ffff_0000_0000 | (DATA + 128);
            });
            assert_eq!(h.read(DATA + 128, 64), [0; 64]);
            assert_eq!(h.read(DATA + 64, 64), [0xaa; 64]);
            // unprivileged, unlike wbinvd
            // (the hlt behind it is what faults at CPL 3)
            let (h, stop) = run_with(CLZERO, |h| {
                user_mode(h);
                fill(h);
                h.cpu.gpr[RAX] = DATA;
            });
            assert_eq!(stop, general_protection(0));
            assert_eq!(h.cpu.rip, CODE_BASE + 3, "stopped at the hlt, after clzero");
            assert_eq!(h.read(DATA, 64), [0; 64]);
        }

        #[test]
        fn clzero_faults_like_a_store() {
            const CLZERO: &[u8] = &[0x0f, 0x01, 0xfc, 0xf4];
            let (_, stop) = run_with(CLZERO, |h| h.cpu.gpr[RAX] = 0x4000_0000_0000);
            assert!(
                matches!(
                    stop,
                    Stop::Fault {
                        vector: 14 | 13,
                        ..
                    }
                ),
                "{stop:?}"
            );
        }

        #[test]
        fn wbinvd_and_invd_at_cpl3_are_gp0() {
            // wbinvd; hlt
            // invd; hlt
            for code in [&[0x0f, 0x09, 0xf4], &[0x0f, 0x08, 0xf4]] {
                assert_general_protection_zero(code, user_mode);
            }
        }
    }

    mod group_15 {
        use super::*;

        #[test]
        fn fences_are_no_ops_at_any_privilege() {
            // lfence
            // mfence
            // sfence
            for code in [
                &[0x0f, 0xae, 0xe8],
                &[0x0f, 0xae, 0xf0],
                &[0x0f, 0xae, 0xf8],
            ] {
                let mut h = Harness::new(code);
                user_mode(&mut h);
                assert_eq!(h.run(1), Stop::Limit);
                assert_eq!(h.cpu.rip, CODE_BASE + 3);
            }
        }

        #[test]
        fn clflush_touches_nothing_but_reports_faults() {
            // clflush [rsi]; hlt
            let h = run_ok(&[0x0f, 0xae, 0x3e, 0xf4], |h| h.cpu.gpr[RSI] = DATA);
            assert_eq!(h.cpu.rip, CODE_BASE + 4);
            // clflush [rsi]; hlt
            let (h, stop) = run_with(&[0x0f, 0xae, 0x3e, 0xf4], |h| {
                unmap_region(h, REMAP_REGION);
                h.cpu.gpr[RSI] = REMAP_REGION;
            });
            assert_eq!(
                stop,
                Stop::Fault {
                    vector: VECTOR_PAGE_FAULT,
                    error_code: Some(0),
                    software: false
                }
            );
            assert_eq!(h.cpu.rip, CODE_BASE);
        }

        #[test]
        fn clflush_is_allowed_at_cpl3() {
            // clflush [rsi]; hlt
            let mut h = Harness::new(&[0x0f, 0xae, 0x3e, 0xf4]);
            user_mode(&mut h);
            h.cpu.gpr[RSI] = DATA;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.rip, CODE_BASE + 3);
        }

        #[test]
        fn register_forms_other_than_the_fences_are_ud() {
            let codes: [&[u8]; 4] = [
                // .byte 0x0f, 0xae, 0xc0; hlt
                &[0x0f, 0xae, 0xc0, 0xf4],
                // .byte 0x0f, 0xae, 0xe0; hlt
                &[0x0f, 0xae, 0xe0, 0xf4],
                // rdfsbase rax; hlt
                &[0xf3, 0x48, 0x0f, 0xae, 0xc0, 0xf4],
                // .byte 0xf3, 0x0f, 0xae, 0xe8; hlt
                &[0xf3, 0x0f, 0xae, 0xe8, 0xf4],
            ];
            for code in codes {
                assert_invalid_opcode(code, |_| {});
            }
        }

        #[test]
        fn xsave_family_memory_forms_are_ud_while_cr4_osxsave_is_clear() {
            for code in [
                // .byte 0x0f, 0xae, 0x26; hlt
                &[0x0f, 0xae, 0x26, 0xf4],
                // .byte 0x0f, 0xae, 0x2e; hlt
                &[0x0f, 0xae, 0x2e, 0xf4],
                // .byte 0x0f, 0xae, 0x36; hlt
                &[0x0f, 0xae, 0x36, 0xf4],
            ] {
                assert_invalid_opcode(code, |h| h.cpu.gpr[RSI] = DATA);
            }
        }

        #[test]
        fn ldmxcsr_loads_valid_values() {
            for value in [0x1f80u32, 0xffff, 0, 0x1f80 | 1 << 15] {
                // ldmxcsr [rsi]; hlt
                let h = run_ok(&[0x0f, 0xae, 0x16, 0xf4], |h| {
                    h.write(DATA, &value.to_le_bytes());
                    h.cpu.gpr[RSI] = DATA;
                    h.cpu.cr4 |= CR4_OSFXSR;
                });
                assert_eq!(h.cpu.mxcsr, value);
            }
        }

        #[test]
        fn ldmxcsr_reserved_bits_are_gp0_and_leave_mxcsr_alone() {
            for bit in 16..32 {
                // ldmxcsr [rsi]; hlt
                let h = assert_general_protection_zero(&[0x0f, 0xae, 0x16, 0xf4], |h| {
                    h.write(DATA, &(0x1f80u32 | 1 << bit).to_le_bytes());
                    h.cpu.gpr[RSI] = DATA;
                    h.cpu.cr4 |= CR4_OSFXSR;
                });
                assert_eq!(h.cpu.mxcsr, 0x1f80, "bit {bit}");
            }
        }

        #[test]
        fn stmxcsr_stores_four_bytes() {
            // stmxcsr [rdi]; hlt
            let h = run_ok(&[0x0f, 0xae, 0x1f, 0xf4], |h| {
                h.write(OUT, &[0xaa; 8]);
                h.cpu.mxcsr = 0x1234;
                h.cpu.gpr[RDI] = OUT;
                h.cpu.cr4 |= CR4_OSFXSR;
            });
            assert_eq!(h.read(OUT, 8), [0x34, 0x12, 0, 0, 0xaa, 0xaa, 0xaa, 0xaa]);
        }
    }

    mod syscall_and_sysret {
        use super::*;

        const USER_BASE_SELECTOR: u64 = 0x23;
        const KERNEL_SELECTOR: u64 = 0x10;
        const STAR_VALUE: u64 = (USER_BASE_SELECTOR << 48) | (KERNEL_SELECTOR << 32);
        const ENTRY: u64 = CODE_BASE + 0x400;
        const USER_CODE: u64 = CODE_BASE + 0x100;
        const SYSCALL_MASK: u64 = IF | TF | DF | AC | NT;
        // syscall; hlt
        const SYSCALL: &[u8] = &[0x0f, 0x05, 0xf4];
        // sysretq; hlt
        const SYSRETQ: &[u8] = &[0x48, 0x0f, 0x07, 0xf4];

        fn enable(h: &mut Harness) {
            h.cpu.efer |= EFER_SCE_BIT;
            h.cpu.star = STAR_VALUE;
            h.cpu.lstar = ENTRY;
            h.cpu.sfmask = SYSCALL_MASK;
        }

        #[test]
        fn syscall_is_ud_without_efer_sce() {
            let h = assert_invalid_opcode(SYSCALL, |h| {
                h.cpu.star = STAR_VALUE;
                h.cpu.lstar = ENTRY;
                h.cpu.gpr[RCX] = 0x1234;
            });
            assert_eq!(h.cpu.gpr[RCX], 0x1234);
        }

        #[test]
        fn syscall_saves_the_return_address_and_flags_and_jumps_to_lstar() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            user_mode(&mut h);
            let before = RESERVED_ONE | CF | ZF | IF | TF | DF | AC | NT | RF;
            h.cpu.eflags = before;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(
                h.cpu.gpr[RCX],
                CODE_BASE + 2,
                "rcx = address of the next instruction"
            );
            assert_eq!(h.cpu.gpr[R11], before & !RF, "r11 = rflags with RF cleared");
            assert_eq!(h.cpu.rip, ENTRY);
        }

        #[test]
        fn syscall_masks_rflags_with_sfmask_and_clears_rf() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            h.cpu.eflags = RESERVED_ONE | CF | ZF | IF | TF | DF | AC | NT | RF;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | CF | ZF);
        }

        #[test]
        fn syscall_with_a_zero_sfmask_keeps_the_flags() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            h.cpu.sfmask = 0;
            h.cpu.eflags = RESERVED_ONE | IF | DF | SF;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.eflags_full(), RESERVED_ONE | IF | DF | SF);
        }

        #[test]
        fn syscall_loads_flat_kernel_segments_from_star() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            user_mode(&mut h);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cpl, 0);
            let cs = h.cpu.seg[SEG_CS];
            let ss = h.cpu.seg[SEG_SS];
            assert_eq!(cs.sel, 0x10, "STAR[47:32] without RPL bits");
            assert_eq!(ss.sel, 0x18, "CS selector + 8");
            assert!(cs.is_code() && cs.long() && cs.present());
            assert_eq!((cs.dpl(), cs.base, cs.limit), (0, 0, 0xffff_ffff));
            assert!(ss.present() && !ss.is_code());
            assert_ne!(ss.flags & SEG_TYPE_READ_WRITE, 0, "writable data");
            assert_eq!((ss.dpl(), ss.base, ss.limit), (0, 0, 0xffff_ffff));
        }

        #[test]
        fn syscall_strips_rpl_bits_from_the_star_selector() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            h.cpu.star = (USER_BASE_SELECTOR << 48) | (0x13 << 32);
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.seg[SEG_CS].sel, 0x10);
            assert_eq!(h.cpu.seg[SEG_SS].sel, 0x18);
        }

        #[test]
        fn syscall_does_not_touch_the_stack_pointer() {
            let mut h = Harness::new(SYSCALL);
            enable(&mut h);
            let rsp = h.cpu.gpr[RSP];
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RSP], rsp);
        }

        #[test]
        fn sysret_is_ud_without_efer_sce() {
            assert_invalid_opcode(SYSRETQ, |_| {});
        }

        #[test]
        fn sysret_at_cpl3_is_gp0() {
            let h = assert_general_protection_zero(SYSRETQ, |h| {
                enable(h);
                user_mode(h);
                h.cpu.gpr[RCX] = USER_CODE;
            });
            assert_eq!(h.cpu.cpl, 3);
        }

        #[test]
        fn sysret_returns_to_cpl3_with_user_segments_and_rcx_as_rip() {
            let mut h = Harness::new(SYSRETQ);
            enable(&mut h);
            h.cpu.gpr[RCX] = USER_CODE;
            h.cpu.gpr[R11] = RESERVED_ONE | IF;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cpl, 3);
            assert_eq!(h.cpu.rip, USER_CODE);
            let cs = h.cpu.seg[SEG_CS];
            let ss = h.cpu.seg[SEG_SS];
            assert_eq!(cs.sel, 0x33, "STAR[63:48] + 16, RPL 3");
            assert_eq!(ss.sel, 0x2b, "STAR[63:48] + 8, RPL 3");
            assert!(cs.is_code() && cs.long() && cs.present());
            assert_eq!((cs.dpl(), cs.base, cs.limit), (3, 0, 0xffff_ffff));
            assert!(ss.present() && !ss.is_code());
            assert_eq!((ss.dpl(), ss.base, ss.limit), (3, 0, 0xffff_ffff));
        }

        #[test]
        fn sysret_restores_flags_from_r11_except_iopl_rf_and_vm() {
            let mut h = Harness::new(SYSRETQ);
            enable(&mut h);
            h.cpu.gpr[RCX] = USER_CODE;
            h.cpu.gpr[R11] = u64::MAX;
            assert_eq!(h.run(1), Stop::Limit);
            // CF PF AF ZF SF TF IF DF OF NT AC ID, plus the always-set bit 1.
            let restored = [0, 2, 4, 6, 7, 8, 9, 10, 11, 14, 18, 21]
                .iter()
                .fold(2u64, |mask, bit| mask | 1 << bit);
            assert_eq!(h.cpu.eflags_full(), restored);
        }

        #[test]
        fn sysret_forces_the_always_one_flag_bit_even_if_r11_clears_it() {
            let mut h = Harness::new(SYSRETQ);
            enable(&mut h);
            h.cpu.gpr[RCX] = USER_CODE;
            h.cpu.gpr[R11] = 0;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.eflags_full(), 2);
        }

        #[test]
        fn sysret_without_rex_w_returns_to_compatibility_mode() {
            // .byte 0x0f, 0x07; hlt
            let mut h = Harness::new(&[0x0f, 0x07, 0xf4]);
            enable(&mut h);
            h.cpu.gpr[RCX] = USER_CODE;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.cpl, 3);
            assert_eq!(h.cpu.rip, USER_CODE);
            assert_eq!(h.cpu.code_mode(), crate::cpu::decode::CodeMode::Bits32);
            let cs = h.cpu.seg[SEG_CS];
            assert_eq!(cs.flags & SEG_LONG, 0, "CS.L is clear: compatibility mode");
            assert_ne!(cs.flags & SEG_DEFAULT_BIG, 0);
        }

        #[test]
        fn syscall_from_compatibility_mode_enters_at_cstar() {
            let mut h = Harness::new(&[0x0f, 0x05, 0xf4]);
            enable(&mut h);
            h.cpu.cstar = 0x2345_6000;
            h.cpu.lstar = 0x1111_1000;
            h.cpu.seg[SEG_CS].flags &= !SEG_LONG;
            h.cpu.seg[SEG_CS].flags |= SEG_DEFAULT_BIG;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.rip, 0x2345_6000);
            assert_eq!(h.cpu.gpr[RCX], CODE_BASE + 2);
            assert!(h.cpu.is_long64(), "the entry is always 64-bit code");
        }

        #[test]
        fn sysret_does_not_touch_rsp_or_rcx_or_r11() {
            let mut h = Harness::new(SYSRETQ);
            enable(&mut h);
            h.cpu.gpr[RCX] = USER_CODE;
            h.cpu.gpr[R11] = 0x202;
            let rsp = h.cpu.gpr[RSP];
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(
                (h.cpu.gpr[RSP], h.cpu.gpr[RCX], h.cpu.gpr[R11]),
                (rsp, USER_CODE, 0x202)
            );
        }

        #[test]
        fn a_user_mode_round_trip_through_the_kernel() {
            let mut h = Harness::new(SYSRETQ);
            // user code: syscall; kernel entry: hlt
            // syscall
            h.write(USER_CODE, &[0x0f, 0x05]);
            h.write(ENTRY, &[crate::testkit::HLT]);
            enable(&mut h);
            user_mode(&mut h);
            h.cpu.set_cpl(0);
            h.cpu.gpr[RCX] = USER_CODE;
            h.cpu.gpr[R11] = RESERVED_ONE | IF | CF;
            assert_eq!(h.run(10), Stop::Halted);
            assert_eq!(h.cpu.cpl, 0, "back in the kernel");
            assert_eq!(h.cpu.rip, ENTRY + 1);
            assert_eq!(h.cpu.gpr[RCX], USER_CODE + 2, "the user return address");
            assert_eq!(
                h.cpu.gpr[R11] & (IF | CF),
                IF | CF,
                "user flags saved in r11"
            );
            assert_eq!(h.cpu.eflags_full() & IF, 0, "sfmask cleared IF on entry");
        }
    }

    mod advertised_features_are_backed_by_instructions {
        use super::*;

        #[test]
        fn cx8_cmpxchg8b_swaps_on_match() {
            // cmpxchg8b [rsi]; hlt
            let h = run_ok(&[0x0f, 0xc7, 0x0e, 0xf4], |h| {
                h.write_u64(DATA, 0x1122_3344_5566_7788);
                h.cpu.gpr[RSI] = DATA;
                h.cpu.gpr[RAX] = 0x5566_7788;
                h.cpu.gpr[RDX] = 0x1122_3344;
                h.cpu.gpr[RBX] = 0xeeff_0011;
                h.cpu.gpr[RCX] = 0xaabb_ccdd;
            });
            assert_eq!(h.read_u64(DATA), 0xaabb_ccdd_eeff_0011);
            assert_ne!(h.cpu.eflags_full() & ZF, 0);
        }

        #[test]
        fn cx8_cmpxchg8b_loads_the_memory_value_on_mismatch() {
            // cmpxchg8b [rsi]; hlt
            let h = run_ok(&[0x0f, 0xc7, 0x0e, 0xf4], |h| {
                h.write_u64(DATA, 0x1122_3344_5566_7788);
                h.cpu.gpr[RSI] = DATA;
                h.cpu.gpr[RAX] = 1;
                h.cpu.gpr[RDX] = 2;
                h.cpu.gpr[RBX] = 3;
                h.cpu.gpr[RCX] = 4;
            });
            assert_eq!(h.read_u64(DATA), 0x1122_3344_5566_7788, "memory unchanged");
            assert_eq!((h.cpu.gpr[RAX], h.cpu.gpr[RDX]), (0x5566_7788, 0x1122_3344));
            assert_eq!(h.cpu.eflags_full() & ZF, 0);
        }

        #[test]
        fn cmov_moves_when_the_condition_holds() {
            // cmovne eax, ebx; hlt
            let h = run_ok(&[0x0f, 0x45, 0xc3, 0xf4], |h| {
                h.cpu.gpr[RAX] = 1;
                h.cpu.gpr[RBX] = 2;
                h.cpu.set_eflags_arith(0);
            });
            assert_eq!(h.cpu.gpr[RAX], 2);
        }

        #[test]
        fn cmov_keeps_the_destination_when_it_does_not_but_zero_extends_32_bit_forms() {
            // cmovne eax, ebx; hlt
            let h = run_ok(&[0x0f, 0x45, 0xc3, 0xf4], |h| {
                h.cpu.gpr[RAX] = 0xffff_ffff_0000_0001;
                h.cpu.gpr[RBX] = 2;
                h.cpu.set_eflags_arith(ZF);
            });
            assert_eq!(h.cpu.gpr[RAX], 1);
        }

        #[test]
        fn lahf_and_sahf_move_the_five_status_flags_through_ah() {
            // lahf; hlt
            let h = run_ok(&[0x9f, 0xf4], |h| {
                h.cpu.set_eflags_arith(CF | PF | AF | ZF | SF | OF)
            });
            assert_eq!(
                (h.cpu.gpr[RAX] >> 8) & 0xff,
                0xd7,
                "SF ZF 0 AF 0 PF 1 CF; OF is not included"
            );
            // sahf; hlt
            let h = run_ok(&[0x9e, 0xf4], |h| {
                h.cpu.gpr[RAX] = 0xff00;
                h.cpu.set_eflags_arith(OF);
            });
            assert_eq!(
                h.cpu.eflags_full() & ARITH,
                CF | PF | AF | ZF | SF | OF,
                "OF is left alone"
            );
            // sahf; hlt
            let h = run_ok(&[0x9e, 0xf4], |h| {
                h.cpu.gpr[RAX] = 0x0000;
                h.cpu.set_eflags_arith(CF | PF | AF | ZF | SF);
            });
            assert_eq!(h.cpu.eflags_full() & ARITH, 0);
        }

        #[test]
        fn de_makes_dr4_and_dr5_invalid() {
            // mov rax, dr4; hlt
            assert_invalid_opcode(&[0x0f, 0x21, 0xe0, 0xf4], |h| h.cpu.cr4 |= CR4_DE);
        }

        #[test]
        fn long_mode_is_active_after_boot() {
            // mov ecx, 0xc0000080; rdmsr; hlt
            let h = run_ok(&[0xb9, 0x80, 0x00, 0x00, 0xc0, 0x0f, 0x32, 0xf4], |_| {});
            assert_eq!(
                h.cpu.gpr[RAX] & (EFER_LME_BIT | EFER_LMA_BIT),
                EFER_LME_BIT | EFER_LMA_BIT
            );
            assert!(h.cpu.seg[SEG_CS].long());
        }
    }

    mod fs_gs_base {
        use super::*;
        use crate::testkit::assert_jit_matches_interpreter;

        const HIGH_HALF_BASE: u64 = 0xffff_8000_0000_1000;
        const NON_CANONICAL: u64 = 0x0001_0000_0000_0000;

        // rdfsbase rax
        const RDFSBASE_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xae, 0xc0, 0xf4];
        // rdfsbase eax
        const RDFSBASE_32: &[u8] = &[0xf3, 0x0f, 0xae, 0xc0, 0xf4];
        // rdgsbase rax
        const RDGSBASE_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xae, 0xc8, 0xf4];
        // rdgsbase eax
        const RDGSBASE_32: &[u8] = &[0xf3, 0x0f, 0xae, 0xc8, 0xf4];
        // wrfsbase rbx
        const WRFSBASE_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xae, 0xd3, 0xf4];
        // wrfsbase ebx
        const WRFSBASE_32: &[u8] = &[0xf3, 0x0f, 0xae, 0xd3, 0xf4];
        // wrgsbase rbx
        const WRGSBASE_64: &[u8] = &[0xf3, 0x48, 0x0f, 0xae, 0xdb, 0xf4];
        // wrgsbase ebx
        const WRGSBASE_32: &[u8] = &[0xf3, 0x0f, 0xae, 0xdb, 0xf4];
        // rdfsbase r9
        const RDFSBASE_R9: &[u8] = &[0xf3, 0x49, 0x0f, 0xae, 0xc1, 0xf4];
        // wrfsbase r9
        const WRFSBASE_R9: &[u8] = &[0xf3, 0x49, 0x0f, 0xae, 0xd1, 0xf4];
        // wrfsbase rbx; mov rax, fs:[0]
        const WRFSBASE_THEN_FS_LOAD: &[u8] = &[
            0xf3, 0x48, 0x0f, 0xae, 0xd3, 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00,
            0xf4,
        ];
        // wrgsbase rbx; mov rax, gs:[0]
        const WRGSBASE_THEN_GS_LOAD: &[u8] = &[
            0xf3, 0x48, 0x0f, 0xae, 0xdb, 0x65, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00,
            0xf4,
        ];

        fn enabled(h: &mut Harness) {
            h.cpu.cr4 |= CR4_FSGSBASE;
        }

        #[test]
        fn reads_return_the_current_bases_at_the_operand_size() {
            let read = |code: &[u8], fs: u64, gs: u64| {
                let h = run_ok(code, |h| {
                    enabled(h);
                    h.cpu.seg[SEG_FS].base = fs;
                    h.cpu.seg[SEG_GS].base = gs;
                    h.cpu.gpr[RAX] = u64::MAX;
                });
                h.cpu.gpr[RAX]
            };
            assert_eq!(read(RDFSBASE_64, HIGH_HALF_BASE, 0x2000), HIGH_HALF_BASE);
            assert_eq!(read(RDGSBASE_64, HIGH_HALF_BASE, 0x2000), 0x2000);
            // The 32-bit forms return the low half and clear the upper half of the register.
            assert_eq!(read(RDFSBASE_32, HIGH_HALF_BASE, 0x2000), 0x1000);
            assert_eq!(
                read(RDGSBASE_32, 0x1000, 0xffff_8000_dead_beef),
                0xdead_beef
            );
            let h = run_ok(RDFSBASE_R9, |h| {
                enabled(h);
                h.cpu.seg[SEG_FS].base = 0x1234;
            });
            assert_eq!(h.cpu.gpr[R9], 0x1234, "REX.B selects r9");
        }

        #[test]
        fn writes_set_the_bases_and_the_32_bit_forms_zero_extend() {
            let write = |code: &[u8], value: u64| {
                let h = run_ok(code, |h| {
                    enabled(h);
                    h.cpu.gpr[RBX] = value;
                    h.cpu.seg[SEG_FS].base = 0x7777;
                    h.cpu.seg[SEG_GS].base = 0x8888;
                });
                (h.cpu.seg[SEG_FS].base, h.cpu.seg[SEG_GS].base)
            };
            assert_eq!(write(WRFSBASE_64, HIGH_HALF_BASE), (HIGH_HALF_BASE, 0x8888));
            assert_eq!(write(WRGSBASE_64, 0x4000), (0x7777, 0x4000));
            assert_eq!(
                write(WRFSBASE_32, 0xffff_ffff_8765_4321),
                (0x8765_4321, 0x8888)
            );
            assert_eq!(
                write(WRGSBASE_32, 0xffff_ffff_8765_4321),
                (0x7777, 0x8765_4321)
            );
            let h = run_ok(WRFSBASE_R9, |h| {
                enabled(h);
                h.cpu.gpr[R9] = 0x5555;
            });
            assert_eq!(h.cpu.seg[SEG_FS].base, 0x5555);
        }

        #[test]
        fn a_non_canonical_base_is_gp0_and_changes_nothing() {
            for code in [WRFSBASE_64, WRGSBASE_64] {
                let h = assert_general_protection_zero(code, |h| {
                    enabled(h);
                    h.cpu.gpr[RBX] = NON_CANONICAL;
                    h.cpu.seg[SEG_FS].base = 0x7777;
                    h.cpu.seg[SEG_GS].base = 0x8888;
                });
                assert_eq!(
                    (h.cpu.seg[SEG_FS].base, h.cpu.seg[SEG_GS].base),
                    (0x7777, 0x8888)
                );
            }
            // A 32-bit write cannot be non-canonical.
            run_ok(WRFSBASE_32, |h| {
                enabled(h);
                h.cpu.gpr[RBX] = NON_CANONICAL | 0x10;
            });
        }

        #[test]
        fn without_cr4_fsgsbase_all_four_are_invalid_opcode() {
            for code in [
                RDFSBASE_64,
                RDGSBASE_32,
                WRFSBASE_64,
                WRGSBASE_32,
                RDFSBASE_R9,
            ] {
                assert_invalid_opcode(code, |_| {});
            }
        }

        #[test]
        fn they_work_in_user_mode() {
            // hlt faults at CPL 3, so run exactly the one instruction.
            let mut h = Harness::new(WRFSBASE_64);
            enabled(&mut h);
            user_mode(&mut h);
            h.cpu.gpr[RBX] = 0x6000;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.rip, CODE_BASE + WRFSBASE_64.len() as u64 - 1);
            assert_eq!(h.cpu.seg[SEG_FS].base, 0x6000);
            let mut h = Harness::new(RDGSBASE_64);
            enabled(&mut h);
            user_mode(&mut h);
            h.cpu.seg[SEG_GS].base = 0x9000;
            assert_eq!(h.run(1), Stop::Limit);
            assert_eq!(h.cpu.gpr[RAX], 0x9000);
        }

        #[test]
        fn other_register_forms_of_group_15_stay_invalid() {
            // F3 0F AE /4 (ptwrite), F2 0F AE /0 and the unprefixed register /0.
            for code in [
                &[0xf3, 0x0f, 0xae, 0xe0, 0xf4][..],
                &[0xf2, 0x0f, 0xae, 0xc0, 0xf4],
                &[0x0f, 0xae, 0xc0, 0xf4],
            ] {
                assert_invalid_opcode(code, enabled);
            }
        }

        #[test]
        fn fs_and_gs_relative_accesses_see_the_new_base() {
            for (code, name) in [(WRFSBASE_THEN_FS_LOAD, "fs"), (WRGSBASE_THEN_GS_LOAD, "gs")] {
                let h = run_ok(code, |h| {
                    enabled(h);
                    h.cpu.gpr[RBX] = DATA;
                    h.write_u64(DATA, 0x1122_3344_5566_7788);
                });
                assert_eq!(h.cpu.gpr[RAX], 0x1122_3344_5566_7788, "{name}");
                assert_jit_matches_interpreter(code, |h| {
                    enabled(h);
                    h.cpu.gpr[RBX] = DATA;
                    h.write_u64(DATA, 0x1122_3344_5566_7788);
                });
            }
        }

        #[test]
        fn the_jit_matches_for_reads_and_writes() {
            for code in [RDFSBASE_64, RDGSBASE_32, WRFSBASE_32, WRGSBASE_64] {
                assert_jit_matches_interpreter(code, |h| {
                    enabled(h);
                    h.cpu.gpr[RBX] = 0xffff_8000_0000_2000;
                    h.cpu.seg[SEG_FS].base = 0x1111;
                    h.cpu.seg[SEG_GS].base = 0xffff_8000_dead_0000;
                });
            }
            assert_jit_matches_interpreter(WRFSBASE_64, |h| {
                enabled(h);
                h.cpu.gpr[RBX] = NON_CANONICAL;
            });
            assert_jit_matches_interpreter(RDFSBASE_64, |_| {});
        }
    }
}
