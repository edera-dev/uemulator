//! AMD SVM for nested guests: `vmrun`, `vmload`, `vmsave`, `stgi`, `clgi`, `invlpga`, the VMCB
//! intercepts, event injection, nested paging (see `mem.rs`) and #VMEXIT.
//!
//! The emulator has exactly one CPU state, so entering a guest is a *state swap*, not a second
//! CPU: `vmrun` saves the host-visible registers into `Guest::host`, loads the VMCB's state into
//! the one `Cpu`, and records the intercepts; every intercept is then a check in the normal
//! execution paths (`exec`, `after_step`, `before_step`, the page walk), and a #VMEXIT swaps the
//! host back. Layout and exit codes follow the AMD APM vol. 2 ch. 15 and are cross-checked against
//! the hardware driver in `uemulator-uefi/src/hv/svm.rs`, which runs on real SVM.
//!
//! Scope: the guest may be in any mode the CPU runs (real, virtual-8086, 16/32-bit protected, 32-bit
//! and PAE paging, compatibility and 64-bit mode); the checks are the APM's. A guest that itself
//! executes `vmrun` is not supported (it is intercepted like any other VMRUN, and without that
//! intercept the instruction is #UD).

use super::decode::Insn;
use super::exception::Exception;
use super::flags::*;
use super::sys::{CR0_TS, EFER_AIBRSE, EFER_SCE, RAX, RCX, RDX};
use super::*;

pub(crate) const MSR_VM_CR: u32 = 0xc001_0114;
pub(crate) const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
/// EFER.SVME: enables the SVM instructions.
pub(crate) const EFER_SVME: u64 = 1 << 12;
/// VM_CR bits software may write: DPD, R_INIT, DIS_A20M, LOCK, SVMDIS.
pub(super) const VM_CR_WRITABLE: u64 = 0b1_1111;
pub(super) const VM_CR_SVMDIS: u64 = 1 << 4;

// VMCB control area offsets.
const VMCB_INTERCEPT_CR: u64 = 0x00;
const VMCB_INTERCEPT_DR: u64 = 0x04;
const VMCB_INTERCEPT_EXCEPTIONS: u64 = 0x08;
const VMCB_INTERCEPT3: u64 = 0x0c;
const VMCB_INTERCEPT4: u64 = 0x10;
const VMCB_IOPM_BASE: u64 = 0x40;
const VMCB_MSRPM_BASE: u64 = 0x48;
const VMCB_TSC_OFFSET: u64 = 0x50;
const VMCB_ASID: u64 = 0x58;
const VMCB_V_INTR: u64 = 0x60;
const VMCB_INTERRUPT_SHADOW: u64 = 0x68;
const VMCB_EXITCODE: u64 = 0x70;
const VMCB_EXITINFO1: u64 = 0x78;
const VMCB_EXITINFO2: u64 = 0x80;
const VMCB_EXIT_INT_INFO: u64 = 0x88;
const VMCB_EXIT_INT_INFO_ERROR: u64 = 0x8c;
const VMCB_NP_ENABLE: u64 = 0x90;
const VMCB_EVENTINJ: u64 = 0xa8;
const VMCB_EVENTINJ_ERROR: u64 = 0xac;
const VMCB_N_CR3: u64 = 0xb0;
const VMCB_NEXT_RIP: u64 = 0xc8;

// State save area (starts at 0x400); each segment is selector u16, attrib u16, limit u32, base u64.
const SAVE_ES: u64 = 0x400;
const SAVE_CS: u64 = 0x410;
const SAVE_SS: u64 = 0x420;
const SAVE_DS: u64 = 0x430;
const SAVE_FS: u64 = 0x440;
const SAVE_GS: u64 = 0x450;
const SAVE_GDTR: u64 = 0x460;
const SAVE_LDTR: u64 = 0x470;
const SAVE_IDTR: u64 = 0x480;
const SAVE_TR: u64 = 0x490;
const SAVE_CPL: u64 = 0x4cb;
const SAVE_EFER: u64 = 0x4d0;
const SAVE_CR4: u64 = 0x548;
const SAVE_CR3: u64 = 0x550;
const SAVE_CR0: u64 = 0x558;
const SAVE_DR7: u64 = 0x560;
const SAVE_DR6: u64 = 0x568;
const SAVE_RFLAGS: u64 = 0x570;
const SAVE_RIP: u64 = 0x578;
const SAVE_STAR: u64 = 0x600;
const SAVE_LSTAR: u64 = 0x608;
const SAVE_CSTAR: u64 = 0x610;
const SAVE_SFMASK: u64 = 0x618;
const SAVE_KERNEL_GS_BASE: u64 = 0x620;
const SAVE_SYSENTER_CS: u64 = 0x628;
const SAVE_SYSENTER_ESP: u64 = 0x630;
const SAVE_SYSENTER_EIP: u64 = 0x638;
const SAVE_CR2: u64 = 0x640;
const SAVE_RSP: u64 = 0x5d8;
const SAVE_RAX: u64 = 0x5f8;
const SAVE_G_PAT: u64 = 0x668;

const SEGMENT_SELECTOR: u64 = 0;
const SEGMENT_ATTRIB: u64 = 2;
const SEGMENT_LIMIT: u64 = 4;
const SEGMENT_BASE: u64 = 8;

// Intercept vector 3 (VMCB 0x0c).
const INTERCEPT_INTR: u32 = 1 << 0;
const INTERCEPT_VINTR: u32 = 1 << 4;
const INTERCEPT_CR0_SELECTIVE_WRITE: u32 = 1 << 5;
const INTERCEPT_IDTR_READ: u32 = 1 << 6;
const INTERCEPT_GDTR_READ: u32 = 1 << 7;
const INTERCEPT_LDTR_READ: u32 = 1 << 8;
const INTERCEPT_TR_READ: u32 = 1 << 9;
const INTERCEPT_IDTR_WRITE: u32 = 1 << 10;
const INTERCEPT_GDTR_WRITE: u32 = 1 << 11;
const INTERCEPT_LDTR_WRITE: u32 = 1 << 12;
const INTERCEPT_TR_WRITE: u32 = 1 << 13;
const INTERCEPT_RDTSC: u32 = 1 << 14;
const INTERCEPT_RDPMC: u32 = 1 << 15;
const INTERCEPT_PUSHF: u32 = 1 << 16;
const INTERCEPT_POPF: u32 = 1 << 17;
const INTERCEPT_CPUID: u32 = 1 << 18;
const INTERCEPT_IRET: u32 = 1 << 20;
const INTERCEPT_INT: u32 = 1 << 21;
const INTERCEPT_INVD: u32 = 1 << 22;
const INTERCEPT_PAUSE: u32 = 1 << 23;
const INTERCEPT_HLT: u32 = 1 << 24;
const INTERCEPT_INVLPG: u32 = 1 << 25;
const INTERCEPT_INVLPGA: u32 = 1 << 26;
const INTERCEPT_IOIO_PROT: u32 = 1 << 27;
const INTERCEPT_MSR_PROT: u32 = 1 << 28;
const INTERCEPT_SHUTDOWN: u32 = 1 << 31;
// Intercept vector 4 (VMCB 0x10).
const INTERCEPT_VMRUN: u32 = 1 << 0;
const INTERCEPT_VMMCALL: u32 = 1 << 1;
const INTERCEPT_VMLOAD: u32 = 1 << 2;
const INTERCEPT_VMSAVE: u32 = 1 << 3;
const INTERCEPT_STGI: u32 = 1 << 4;
const INTERCEPT_CLGI: u32 = 1 << 5;
const INTERCEPT_SKINIT: u32 = 1 << 6;
const INTERCEPT_RDTSCP: u32 = 1 << 7;
const INTERCEPT_WBINVD: u32 = 1 << 9;
const INTERCEPT_MONITOR: u32 = 1 << 10;
const INTERCEPT_MWAIT: u32 = 1 << 11;
const INTERCEPT_XSETBV: u32 = 1 << 13;

// #VMEXIT codes.
pub(crate) const VMEXIT_CR_READ: u64 = 0x00;
pub(crate) const VMEXIT_CR_WRITE: u64 = 0x10;
pub(crate) const VMEXIT_DR_READ: u64 = 0x20;
pub(crate) const VMEXIT_DR_WRITE: u64 = 0x30;
pub(crate) const VMEXIT_EXCEPTION: u64 = 0x40;
pub(crate) const VMEXIT_INTR: u64 = 0x60;
pub(crate) const VMEXIT_VINTR: u64 = 0x64;
pub(crate) const VMEXIT_CR0_SELECTIVE_WRITE: u64 = 0x65;
const VMEXIT_IDTR_READ: u64 = 0x66;
const VMEXIT_GDTR_READ: u64 = 0x67;
const VMEXIT_LDTR_READ: u64 = 0x68;
const VMEXIT_TR_READ: u64 = 0x69;
const VMEXIT_IDTR_WRITE: u64 = 0x6a;
const VMEXIT_GDTR_WRITE: u64 = 0x6b;
const VMEXIT_LDTR_WRITE: u64 = 0x6c;
const VMEXIT_TR_WRITE: u64 = 0x6d;
const VMEXIT_RDTSC: u64 = 0x6e;
const VMEXIT_RDPMC: u64 = 0x6f;
const VMEXIT_PUSHF: u64 = 0x70;
const VMEXIT_POPF: u64 = 0x71;
const VMEXIT_CPUID: u64 = 0x72;
const VMEXIT_IRET: u64 = 0x74;
const VMEXIT_INT: u64 = 0x75;
const VMEXIT_INVD: u64 = 0x76;
const VMEXIT_PAUSE: u64 = 0x77;
const VMEXIT_HLT: u64 = 0x78;
const VMEXIT_INVLPG: u64 = 0x79;
const VMEXIT_INVLPGA: u64 = 0x7a;
const VMEXIT_IOIO: u64 = 0x7b;
const VMEXIT_MSR: u64 = 0x7c;
const VMEXIT_SHUTDOWN: u64 = 0x7f;
const VMEXIT_VMRUN: u64 = 0x80;
const VMEXIT_VMMCALL: u64 = 0x81;
const VMEXIT_VMLOAD: u64 = 0x82;
const VMEXIT_VMSAVE: u64 = 0x83;
const VMEXIT_STGI: u64 = 0x84;
const VMEXIT_CLGI: u64 = 0x85;
const VMEXIT_SKINIT: u64 = 0x86;
const VMEXIT_RDTSCP: u64 = 0x87;
const VMEXIT_WBINVD: u64 = 0x89;
const VMEXIT_MONITOR: u64 = 0x8a;
const VMEXIT_MWAIT: u64 = 0x8b;
const VMEXIT_XSETBV: u64 = 0x8d;
pub(crate) const VMEXIT_NPF: u64 = 0x400;
pub(crate) const VMEXIT_INVALID: u64 = u64::MAX;

// IOIO exit information (EXITINFO1).
const IOIO_IN: u64 = 1 << 0;
const IOIO_STRING: u64 = 1 << 2;
const IOIO_REP: u64 = 1 << 3;
const IOIO_SIZE_8: u64 = 1 << 4;
const IOIO_SIZE_16: u64 = 1 << 5;
const IOIO_SIZE_32: u64 = 1 << 6;
const IOIO_ADDRESS_64: u64 = 1 << 9;
const IOIO_PORT_SHIFT: u32 = 16;

/// The I/O permission map covers ports 0..=0xffff, one bit each.
const IOPM_BITS_PER_PORT: u64 = 1;
/// MSR permission map: two bits per MSR (read, write) in three 2 KiB ranges.
const MSRPM_RANGES: [(u32, u64); 3] = [(0, 0), (0xc000_0000, 0x800), (0xc001_0000, 0x1000)];
const MSRPM_RANGE_MSRS: u32 = 0x2000;

// EVENTINJ.
const EVENTINJ_VALID: u32 = 1 << 31;
const EVENTINJ_ERROR_VALID: u32 = 1 << 11;
const EVENTINJ_VECTOR_MASK: u32 = 0xff;
const EVENTINJ_TYPE_SHIFT: u32 = 8;
const EVENTINJ_TYPE_MASK: u32 = 7;
pub(super) const EVENT_TYPE_INTERRUPT: u32 = 0;
const EVENT_TYPE_NMI: u32 = 2;
pub(super) const EVENT_TYPE_EXCEPTION: u32 = 3;
pub(super) const EVENT_TYPE_SOFTWARE: u32 = 4;

// V_INTR.
const V_IRQ: u64 = 1 << 8;
const V_PRIORITY_SHIFT: u32 = 16;
const V_PRIORITY_MASK: u64 = 0xf;
const V_IGNORE_TPR: u64 = 1 << 20;
const V_VECTOR_SHIFT: u32 = 32;

const INTERRUPT_SHADOW: u64 = 1;
const NP_ENABLE: u64 = 1;
const VECTOR_INTERCEPT_BITMAP_BITS: u32 = 32;
const VECTOR_BREAKPOINT_EXCEPTION: u8 = 3;
const CR0_SELECTIVE_IGNORED: u64 = CR0_TS | 2;
const CR0_CD_BIT: u64 = 1 << 30;
const CR0_NW_BIT: u64 = 1 << 29;
const CR0_PG_BIT: u64 = 1 << 31;
const CR0_PE_BIT: u64 = 1;
/// EFER bits a VMCB may hold; any other is a consistency failure (VMEXIT_INVALID). AIBRSE is in
/// the set because a KVM guest that sees automatic IBRS turns it on and KVM passes its EFER on.
const EFER_KNOWN: u64 = EFER_SCE | EFER_LME | EFER_LMA | EFER_NXE | EFER_SVME | EFER_AIBRSE;
const DR_HIGH_BITS: u64 = !0u64 << 32;
/// Number of ASIDs reported in CPUID 0x8000000A; any non-zero ASID in a VMCB is accepted.
pub(crate) const SVM_ASIDS: u32 = 64;

/// CPUID 0x8000000A EDX: nested paging, next-RIP save, VMCB clean bits (every VMRUN reloads the
/// whole VMCB, which clean bits permit), flush-by-ASID (every VMRUN flushes the whole TLB).
pub(crate) const SVM_FEATURES: u32 = (1 << 0) | (1 << 3) | (1 << 5) | (1 << 6);

/// What `vmrun` saved of the host, and restores at #VMEXIT. The other registers (including all
/// GPRs but RAX/RSP) are deliberately *not* here: the hypervisor swaps those itself.
struct Host {
    rip: u64,
    rsp: u64,
    rax: u64,
    eflags: u64,
    cr0: u64,
    cr3: u64,
    cr4: u64,
    efer: u64,
    seg: [Seg; 4],
    gdtr: DescriptorTable,
    idtr: DescriptorTable,
    dr6: u64,
    dr7: u64,
    pdpte: [u64; 4],
}

/// The guest being run: the intercepts read from its VMCB at `vmrun` time (the hypervisor cannot
/// change them while the guest runs) and the host state to return to.
pub(crate) struct Guest {
    vmcb: u64,
    intercept_cr: u32,
    intercept_dr: u32,
    intercept_exceptions: u32,
    intercept3: u32,
    intercept4: u32,
    iopm: u64,
    msrpm: u64,
    tsc_offset: u64,
    /// Nested page table root, when nested paging is on.
    pub(crate) nested_cr3: Option<u64>,
    host: Host,
    /// Virtual interrupt (V_IRQ): vector, priority, and whether the guest's CR8 is ignored.
    virtual_irq: bool,
    virtual_vector: u8,
    virtual_priority: u8,
    ignore_tpr: bool,
    virtual_tpr: u8,
}

/// SVM state of the CPU.
pub(crate) struct Svm {
    pub(crate) hsave_pa: u64,
    pub(crate) vm_cr: u64,
    /// Global interrupt flag: clear while the host handles a #VMEXIT (`clgi`).
    pub(crate) gif: bool,
    pub(crate) guest: Option<Box<Guest>>,
    /// The VMCB of the guest that exited last, so an exit that happened while delivering an event
    /// can still report that event (`exit_int_info`).
    last_vmcb: u64,
}

impl Svm {
    pub(crate) fn new() -> Self {
        Svm {
            hsave_pa: 0,
            vm_cr: 0,
            gif: true,
            guest: None,
            last_vmcb: 0,
        }
    }
}

fn attrib_to_flags(attrib: u16) -> u32 {
    (((attrib & 0xff) as u32) << 8) | ((((attrib >> 8) & 0xf) as u32) << 20)
}

fn flags_to_attrib(flags: u32) -> u16 {
    (((flags >> 8) & 0xff) | (((flags >> 20) & 0xf) << 8)) as u16
}

impl Cpu {
    pub(crate) fn svm_enabled(&self) -> bool {
        self.efer & EFER_SVME != 0
    }

    fn vmcb_read(&mut self, base: u64, offset: u64, size: u8) -> u64 {
        self.phys_read(base + offset, size)
    }

    fn vmcb_write(&mut self, base: u64, offset: u64, size: u8, value: u64) {
        self.phys_write(base + offset, size, value);
    }

    fn vmcb_segment(&mut self, base: u64, offset: u64) -> Seg {
        let at = base + offset;
        Seg {
            sel: self.phys_read(at + SEGMENT_SELECTOR, 2) as u16,
            flags: attrib_to_flags(self.phys_read(at + SEGMENT_ATTRIB, 2) as u16),
            limit: self.phys_read(at + SEGMENT_LIMIT, 4) as u32,
            base: self.phys_read(at + SEGMENT_BASE, 8),
        }
    }

    fn vmcb_set_segment(&mut self, base: u64, offset: u64, segment: &Seg) {
        let at = base + offset;
        self.phys_write(at + SEGMENT_SELECTOR, 2, segment.sel as u64);
        self.phys_write(
            at + SEGMENT_ATTRIB,
            2,
            flags_to_attrib(segment.flags) as u64,
        );
        self.phys_write(at + SEGMENT_LIMIT, 4, segment.limit as u64);
        self.phys_write(at + SEGMENT_BASE, 8, segment.base);
    }

    fn vmcb_table(&mut self, base: u64, offset: u64) -> DescriptorTable {
        DescriptorTable {
            limit: self.phys_read(base + offset + SEGMENT_LIMIT, 4) as u16,
            base: self.phys_read(base + offset + SEGMENT_BASE, 8),
        }
    }

    fn vmcb_set_table(&mut self, base: u64, offset: u64, table: &DescriptorTable) {
        self.phys_write(base + offset + SEGMENT_LIMIT, 4, table.limit as u64);
        self.phys_write(base + offset + SEGMENT_BASE, 8, table.base);
    }

    /// Preconditions every SVM instruction shares: #UD with EFER.SVME clear, #GP(0) outside CPL 0.
    fn svm_instruction_checks(&self) -> ExecResult<()> {
        if !self.svm_enabled() {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()
    }

    /// A VMCB (or VMLOAD/VMSAVE area) address from rAX: 4 KiB aligned and inside the physical
    /// address space.
    fn vmcb_address(&self) -> ExecResult<u64> {
        let address = self.gpr[RAX];
        if address & (PAGE_SIZE - 1) != 0 || address >> super::sys::PHYSICAL_ADDRESS_BITS != 0 {
            return Err(Exception::general_protection(0));
        }
        Ok(address)
    }

    /// 0F 01 DA / DB: loads or stores the state `vmrun` leaves alone: FS, GS, TR, LDTR,
    /// KernelGSbase, STAR, LSTAR, CSTAR, SFMASK and the SYSENTER MSRs.
    pub(super) fn vmload_vmsave(&mut self, save: bool, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        let vmcb = self.vmcb_address()?;
        if save {
            for (offset, segment) in [
                (SAVE_FS, self.seg[SEG_FS]),
                (SAVE_GS, self.seg[SEG_GS]),
                (SAVE_TR, self.tr),
                (SAVE_LDTR, self.ldtr),
            ] {
                self.vmcb_set_segment(vmcb, offset, &segment);
            }
            for (offset, value) in [
                (SAVE_KERNEL_GS_BASE, self.kgsbase),
                (SAVE_STAR, self.star),
                (SAVE_LSTAR, self.lstar),
                (SAVE_CSTAR, self.cstar),
                (SAVE_SFMASK, self.sfmask),
                (SAVE_SYSENTER_CS, self.sysenter[0]),
                (SAVE_SYSENTER_ESP, self.sysenter[1]),
                (SAVE_SYSENTER_EIP, self.sysenter[2]),
            ] {
                self.vmcb_write(vmcb, offset, 8, value);
            }
        } else {
            self.seg[SEG_FS] = self.vmcb_segment(vmcb, SAVE_FS);
            self.seg[SEG_GS] = self.vmcb_segment(vmcb, SAVE_GS);
            self.tr = self.vmcb_segment(vmcb, SAVE_TR);
            self.ldtr = self.vmcb_segment(vmcb, SAVE_LDTR);
            self.kgsbase = self.vmcb_read(vmcb, SAVE_KERNEL_GS_BASE, 8);
            self.star = self.vmcb_read(vmcb, SAVE_STAR, 8);
            self.lstar = self.vmcb_read(vmcb, SAVE_LSTAR, 8);
            self.cstar = self.vmcb_read(vmcb, SAVE_CSTAR, 8);
            self.sfmask = self.vmcb_read(vmcb, SAVE_SFMASK, 8);
            for (k, offset) in [SAVE_SYSENTER_CS, SAVE_SYSENTER_ESP, SAVE_SYSENTER_EIP]
                .into_iter()
                .enumerate()
            {
                self.sysenter[k] = self.vmcb_read(vmcb, offset, 8);
            }
        }
        self.rip = next;
        Ok(())
    }

    /// 0F 01 DC / DD.
    pub(super) fn stgi_clgi(&mut self, set: bool, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        self.svm.gif = set;
        self.rip = next;
        Ok(())
    }

    /// 0F 01 DF: invalidates the TLB (the TLB here is not tagged with ASIDs, so everything goes).
    pub(super) fn invlpga(&mut self, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        self.flush_tlb();
        self.rip = next;
        Ok(())
    }

    /// 0F 01 D8. Saves the host state, loads the guest state from the VMCB at rAX, checks the
    /// VMCB for consistency (a bad one makes the instruction complete with VMEXIT_INVALID in the
    /// VMCB and the host state untouched), injects any pending event and starts running the guest.
    pub(super) fn vmrun(&mut self, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        let vmcb = self.vmcb_address()?;
        let intercept3 = self.vmcb_read(vmcb, VMCB_INTERCEPT3, 4) as u32;
        let intercept4 = self.vmcb_read(vmcb, VMCB_INTERCEPT4, 4) as u32;
        let guest_efer = self.vmcb_read(vmcb, SAVE_EFER, 8);
        let guest_cr0 = self.vmcb_read(vmcb, SAVE_CR0, 8);
        let guest_cr3 = self.vmcb_read(vmcb, SAVE_CR3, 8);
        let guest_cr4 = self.vmcb_read(vmcb, SAVE_CR4, 8);
        let guest_dr6 = self.vmcb_read(vmcb, SAVE_DR6, 8);
        let guest_dr7 = self.vmcb_read(vmcb, SAVE_DR7, 8);
        let asid = self.vmcb_read(vmcb, VMCB_ASID, 4);
        let cs = self.vmcb_segment(vmcb, SAVE_CS);
        let long_mode_paging = guest_efer & EFER_LME != 0 && guest_cr0 & CR0_PG_BIT != 0;
        // Consistency checks (APM 15.5.1).
        let invalid = guest_efer & EFER_SVME == 0
            || guest_efer & !EFER_KNOWN != 0
            || (guest_cr0 & CR0_CD_BIT == 0 && guest_cr0 & CR0_NW_BIT != 0)
            || guest_cr0 >> 32 != 0
            || guest_cr3 & super::sys::CR3_RESERVED_BITS != 0
            || guest_cr4 & !super::sys::CR4_SUPPORTED_BITS != 0
            || guest_dr6 & DR_HIGH_BITS != 0
            || guest_dr7 & DR_HIGH_BITS != 0
            || intercept4 & INTERCEPT_VMRUN == 0
            || asid == 0
            || (long_mode_paging
                && (guest_cr0 & CR0_PE_BIT == 0
                    || guest_cr4 & CR4_PAE == 0
                    || (cs.long() && cs.flags & SEG_DEFAULT_BIG != 0)));
        if invalid {
            self.vmcb_write(vmcb, VMCB_EXITCODE, 8, VMEXIT_INVALID);
            self.rip = next;
            return Ok(());
        }
        let host = Host {
            rip: next,
            rsp: self.gpr[GPR_RSP],
            rax: self.gpr[RAX],
            eflags: self.eflags_full(),
            cr0: self.cr0,
            cr3: self.cr3,
            cr4: self.cr4,
            efer: self.efer,
            seg: [
                self.seg[SEG_ES],
                self.seg[SEG_CS],
                self.seg[SEG_SS],
                self.seg[SEG_DS],
            ],
            gdtr: self.gdtr,
            idtr: self.idtr,
            dr6: self.dr[6],
            dr7: self.dr[7],
            pdpte: self.pdpte,
        };
        let v_intr = self.vmcb_read(vmcb, VMCB_V_INTR, 8);
        let nested = self.vmcb_read(vmcb, VMCB_NP_ENABLE, 8) & NP_ENABLE != 0;
        let guest = Guest {
            vmcb,
            intercept_cr: self.vmcb_read(vmcb, VMCB_INTERCEPT_CR, 4) as u32,
            intercept_dr: self.vmcb_read(vmcb, VMCB_INTERCEPT_DR, 4) as u32,
            intercept_exceptions: self.vmcb_read(vmcb, VMCB_INTERCEPT_EXCEPTIONS, 4) as u32,
            intercept3,
            intercept4,
            iopm: self.vmcb_read(vmcb, VMCB_IOPM_BASE, 8),
            msrpm: self.vmcb_read(vmcb, VMCB_MSRPM_BASE, 8),
            tsc_offset: self.vmcb_read(vmcb, VMCB_TSC_OFFSET, 8),
            nested_cr3: nested.then(|| self.vmcb_read(vmcb, VMCB_N_CR3, 8)),
            host,
            virtual_irq: v_intr & V_IRQ != 0,
            virtual_vector: (v_intr >> V_VECTOR_SHIFT) as u8,
            virtual_priority: ((v_intr >> V_PRIORITY_SHIFT) & V_PRIORITY_MASK) as u8,
            ignore_tpr: v_intr & V_IGNORE_TPR != 0,
            virtual_tpr: v_intr as u8,
        };
        if self.trace.is_some() {
            let guest_rip = self.vmcb_read(vmcb, SAVE_RIP, 8);
            if let Some(log) = &self.log {
                log(&format!(
                    "svm: vmrun vmcb={vmcb:#x} rip={guest_rip:#x} cr0={guest_cr0:#x} efer={guest_efer:#x} npt={nested} cs={:#x}:{:#x} limit={:#x} flags={:#x}",
                    cs.sel, cs.base, cs.limit, cs.flags
                ));
            }
        }
        // Load the guest.
        self.seg[SEG_ES] = self.vmcb_segment(vmcb, SAVE_ES);
        self.seg[SEG_CS] = cs;
        self.seg[SEG_SS] = self.vmcb_segment(vmcb, SAVE_SS);
        self.seg[SEG_DS] = self.vmcb_segment(vmcb, SAVE_DS);
        self.gdtr = self.vmcb_table(vmcb, SAVE_GDTR);
        self.idtr = self.vmcb_table(vmcb, SAVE_IDTR);
        let cpl = self.vmcb_read(vmcb, SAVE_CPL, 1) as u8 & 3;
        self.set_cpl(cpl);
        // Long mode is active exactly when paging is on with LME set.
        self.efer = if long_mode_paging {
            guest_efer | EFER_LMA
        } else {
            guest_efer & !EFER_LMA
        };
        self.cr0 = guest_cr0;
        self.cr3 = guest_cr3;
        self.cr4 = guest_cr4;
        self.cr2 = self.vmcb_read(vmcb, SAVE_CR2, 8);
        self.dr[6] = guest_dr6;
        self.dr[7] = guest_dr7;
        let rflags = self.vmcb_read(vmcb, SAVE_RFLAGS, 8);
        self.eflags = (rflags & !(VM | RF)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.rip = self.vmcb_read(vmcb, SAVE_RIP, 8);
        self.gpr[GPR_RSP] = self.vmcb_read(vmcb, SAVE_RSP, 8);
        self.gpr[RAX] = self.vmcb_read(vmcb, SAVE_RAX, 8);
        if nested {
            self.pat = self.vmcb_read(vmcb, SAVE_G_PAT, 8);
        }
        self.interrupt_shadow =
            self.vmcb_read(vmcb, VMCB_INTERRUPT_SHADOW, 8) & INTERRUPT_SHADOW != 0;
        self.halted = false;
        self.svm.gif = true;
        self.svm.guest = Some(Box::new(guest));
        self.flush_tlb();
        self.flush_decode();
        // PAE paging latches the guest's four PDPT entries now (through nested paging, if on).
        if self.paging_mode() == super::mem::PagingMode::Pae {
            match self.read_pdptes(guest_cr3) {
                Ok(entries) => self.pdpte = entries,
                Err(_) => return Err(self.vmexit(VMEXIT_INVALID, 0, 0, next)),
            }
        }
        // Event injection happens after the state is loaded, as if the event arrived just before
        // the guest's first instruction.
        let event = self.vmcb_read(vmcb, VMCB_EVENTINJ, 4) as u32;
        if event & EVENTINJ_VALID != 0 {
            let error = self.vmcb_read(vmcb, VMCB_EVENTINJ_ERROR, 4) as u32;
            let vector = (event & EVENTINJ_VECTOR_MASK) as u8;
            let error_code = (event & EVENTINJ_ERROR_VALID != 0).then_some(error);
            let reason = match (event >> EVENTINJ_TYPE_SHIFT) & EVENTINJ_TYPE_MASK {
                EVENT_TYPE_INTERRUPT => self.deliver_hardware_interrupt(vector),
                EVENT_TYPE_NMI => self.deliver_event(2, None, false),
                EVENT_TYPE_EXCEPTION => self.deliver_event(vector, error_code, false),
                EVENT_TYPE_SOFTWARE => self.deliver_event(vector, None, true),
                _ => None,
            };
            if let Some(reason) = reason {
                return Err(Exception::Exit(reason));
            }
        }
        Ok(())
    }

    /// Ends the guest run: saves the guest state and the exit information into the VMCB, restores
    /// the host, clears GIF, and returns the marker that tells the run loop the instruction has
    /// been fully accounted for. `next_rip` is the address after the intercepted instruction (0
    /// when meaningless).
    pub(crate) fn vmexit(&mut self, code: u64, info1: u64, info2: u64, next_rip: u64) -> Exception {
        let guest = self.svm.guest.take().expect("vmexit outside a guest");
        let vmcb = guest.vmcb;
        self.svm.last_vmcb = vmcb;
        if let (Some(log), Some(_)) = (&self.log, &self.trace) {
            log(&format!(
                "svm: #VMEXIT {code:#x} info1={info1:#x} info2={info2:#x} rip={:#x} nrip={next_rip:#x} cs={:#x}:{:#x} flags={:#x} rax={:#x} cr0={:#x} cpl={}",
                self.rip,
                self.seg[SEG_CS].sel,
                self.seg[SEG_CS].base,
                self.seg[SEG_CS].flags,
                self.gpr[0],
                self.cr0,
                self.cpl
            ));
        }
        for (offset, segment) in [
            (SAVE_ES, self.seg[SEG_ES]),
            (SAVE_CS, self.seg[SEG_CS]),
            (SAVE_SS, self.seg[SEG_SS]),
            (SAVE_DS, self.seg[SEG_DS]),
        ] {
            self.vmcb_set_segment(vmcb, offset, &segment);
        }
        let (gdtr, idtr) = (self.gdtr, self.idtr);
        self.vmcb_set_table(vmcb, SAVE_GDTR, &gdtr);
        self.vmcb_set_table(vmcb, SAVE_IDTR, &idtr);
        let (cpl, efer, cr4, cr3, cr0) = (self.cpl, self.efer, self.cr4, self.cr3, self.cr0);
        self.vmcb_write(vmcb, SAVE_CPL, 1, cpl as u64);
        // LMA is read-only and hardware-maintained; it is saved as the processor had it.
        self.vmcb_write(vmcb, SAVE_EFER, 8, efer);
        self.vmcb_write(vmcb, SAVE_CR4, 8, cr4);
        self.vmcb_write(vmcb, SAVE_CR3, 8, cr3);
        self.vmcb_write(vmcb, SAVE_CR0, 8, cr0);
        self.vmcb_write(vmcb, SAVE_CR2, 8, self.cr2);
        self.vmcb_write(vmcb, SAVE_DR7, 8, self.dr[7]);
        self.vmcb_write(vmcb, SAVE_DR6, 8, self.dr[6]);
        let rflags = self.eflags_full();
        self.vmcb_write(vmcb, SAVE_RFLAGS, 8, rflags);
        self.vmcb_write(vmcb, SAVE_RIP, 8, self.rip);
        self.vmcb_write(vmcb, SAVE_RSP, 8, self.gpr[GPR_RSP]);
        self.vmcb_write(vmcb, SAVE_RAX, 8, self.gpr[RAX]);
        if guest.nested_cr3.is_some() {
            self.vmcb_write(vmcb, SAVE_G_PAT, 8, self.pat);
        }
        self.vmcb_write(
            vmcb,
            VMCB_INTERRUPT_SHADOW,
            8,
            self.interrupt_shadow as u64 * INTERRUPT_SHADOW,
        );
        // the virtual interrupt that is still pending goes back into V_INTR
        let v_intr = (guest.virtual_tpr as u64)
            | (guest.virtual_irq as u64 * V_IRQ)
            | ((guest.virtual_priority as u64) << V_PRIORITY_SHIFT)
            | (guest.ignore_tpr as u64 * V_IGNORE_TPR)
            | ((guest.virtual_vector as u64) << V_VECTOR_SHIFT);
        let preserved = self.vmcb_read(vmcb, VMCB_V_INTR, 8)
            & !(0xff
                | V_IRQ
                | (V_PRIORITY_MASK << V_PRIORITY_SHIFT)
                | V_IGNORE_TPR
                | (0xff << V_VECTOR_SHIFT));
        self.vmcb_write(vmcb, VMCB_V_INTR, 8, preserved | v_intr);
        self.vmcb_write(vmcb, VMCB_EVENTINJ, 4, 0);
        self.vmcb_write(vmcb, VMCB_EXIT_INT_INFO, 4, 0);
        self.vmcb_write(vmcb, VMCB_EXITCODE, 8, code);
        self.vmcb_write(vmcb, VMCB_EXITINFO1, 8, info1);
        self.vmcb_write(vmcb, VMCB_EXITINFO2, 8, info2);
        self.vmcb_write(vmcb, VMCB_NEXT_RIP, 8, next_rip);
        // restore the host
        let host = guest.host;
        self.seg[SEG_ES] = host.seg[0];
        self.seg[SEG_CS] = host.seg[1];
        self.seg[SEG_SS] = host.seg[2];
        self.seg[SEG_DS] = host.seg[3];
        self.gdtr = host.gdtr;
        self.idtr = host.idtr;
        self.set_cpl(0);
        self.efer = host.efer;
        self.cr0 = host.cr0;
        self.cr3 = host.cr3;
        self.cr4 = host.cr4;
        self.dr[6] = host.dr6;
        self.dr[7] = host.dr7;
        self.pdpte = host.pdpte;
        self.eflags = (host.eflags & !(VM | RF)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.rip = host.rip;
        self.gpr[GPR_RSP] = host.rsp;
        self.gpr[RAX] = host.rax;
        self.interrupt_shadow = false;
        self.halted = false;
        self.svm.gif = false;
        self.flush_tlb();
        self.flush_decode();
        Exception::Vmexit
    }

    /// Records the event whose delivery a #VMEXIT just interrupted (a nested page fault while
    /// pushing the frame): `exit_int_info`, so the hypervisor can inject it again after fixing the
    /// cause. `kind` is the SVM event type (interrupt, NMI, exception or software interrupt).
    pub(crate) fn svm_event_in_flight(&mut self, vector: u8, error_code: Option<u32>, kind: u32) {
        let vmcb = self.svm.last_vmcb;
        let mut info = EVENTINJ_VALID | (kind << EVENTINJ_TYPE_SHIFT) | vector as u32;
        if let Some(code) = error_code {
            info |= EVENTINJ_ERROR_VALID;
            self.vmcb_write(vmcb, VMCB_EXIT_INT_INFO_ERROR, 4, code as u64);
        }
        self.vmcb_write(vmcb, VMCB_EXIT_INT_INFO, 4, info as u64);
    }

    /// The guest's time stamp counter is the host's plus the VMCB offset.
    pub(crate) fn svm_tsc_offset(&self) -> u64 {
        self.svm.guest.as_ref().map_or(0, |guest| guest.tsc_offset)
    }

    /// Whether the running guest intercepts exception `vector`; performs the #VMEXIT if so.
    /// Returns true when it did, so the caller must not deliver the exception to the guest.
    pub(crate) fn svm_exception_exit(
        &mut self,
        vector: u8,
        error_code: Option<u32>,
        software: bool,
    ) -> bool {
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        // `int n` is not an exception; `int3` is, for the #BP intercept.
        if (software && vector != VECTOR_BREAKPOINT_EXCEPTION)
            || vector as u32 >= VECTOR_INTERCEPT_BITMAP_BITS
            || guest.intercept_exceptions >> vector & 1 == 0
        {
            return false;
        }
        let info2 = if vector == super::exception::VECTOR_PAGE_FAULT {
            self.cr2
        } else {
            0
        };
        let next = self.rip;
        self.vmexit(
            VMEXIT_EXCEPTION + vector as u64,
            error_code.unwrap_or(0) as u64,
            info2,
            next,
        );
        true
    }

    /// A triple fault inside a guest that intercepts shutdown is a #VMEXIT, not a machine reset.
    pub(crate) fn svm_shutdown_exit(&mut self) -> bool {
        match &self.svm.guest {
            Some(guest) if guest.intercept3 & INTERCEPT_SHUTDOWN != 0 => {
                let next = self.rip;
                self.vmexit(VMEXIT_SHUTDOWN, 0, 0, next);
                true
            }
            _ => false,
        }
    }

    /// Housekeeping before each guest instruction: physical interrupts (INTR intercept) and virtual
    /// interrupts. Returns true when a #VMEXIT happened.
    pub(crate) fn svm_before_step(&mut self) -> bool {
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        if self.dev.lock().interrupt_pending() && guest.intercept3 & INTERCEPT_INTR != 0 {
            let next = self.rip;
            self.vmexit(VMEXIT_INTR, 0, 0, next);
            return true;
        }
        if guest.virtual_irq
            && self.eflags & IF != 0
            && !self.interrupt_shadow
            && (guest.ignore_tpr || guest.virtual_priority > guest.virtual_tpr >> 4)
        {
            if guest.intercept3 & INTERCEPT_VINTR != 0 {
                let next = self.rip;
                self.vmexit(VMEXIT_VINTR, 0, 0, next);
                return true;
            }
            let vector = guest.virtual_vector;
            if let Some(guest) = &mut self.svm.guest {
                guest.virtual_irq = false;
            }
            self.halted = false;
            // A failure to deliver is reported by the next step like any other fault.
            let _ = self.deliver_hardware_interrupt(vector);
        }
        false
    }

    fn port_is_intercepted(&mut self, port: u16, size: u8) -> bool {
        let iopm = self.svm.guest.as_ref().map_or(0, |guest| guest.iopm);
        (0..size as u64).any(|k| {
            let bit = (port as u64 + k) * IOPM_BITS_PER_PORT;
            self.phys_read(iopm + bit / 8, 1) >> (bit % 8) & 1 != 0
        })
    }

    fn msr_is_intercepted(&mut self, msr: u32, write: bool) -> bool {
        let msrpm = self.svm.guest.as_ref().map_or(0, |guest| guest.msrpm);
        for (start, offset) in MSRPM_RANGES {
            if msr >= start && msr - start < MSRPM_RANGE_MSRS {
                let bit = (msr - start) as u64 * 2 + write as u64;
                return self.phys_read(msrpm + offset + bit / 8, 1) >> (bit % 8) & 1 != 0;
            }
        }
        // MSRs outside the three ranges always exit
        true
    }

    /// The instruction intercepts. Called from `exec` before the instruction does anything, while
    /// RIP still points at it, so the guest state saved by the exit is "before the instruction".
    pub(crate) fn svm_intercept(&mut self, i: &Insn, next: u64) -> ExecResult<()> {
        let Some(guest) = &self.svm.guest else {
            return Ok(());
        };
        let (cr_bits, dr_bits, v3, v4) = (
            guest.intercept_cr,
            guest.intercept_dr,
            guest.intercept3,
            guest.intercept4,
        );
        let op = i.op;
        // (exit code, whether intercepted)
        let simple = |code: u64, hit: bool| hit.then_some((code, 0u64, 0u64));
        let reg = (i.reg & 7) as u32;
        let hit = match op {
            0x1a2 => simple(VMEXIT_CPUID, v3 & INTERCEPT_CPUID != 0),
            0xf4 => simple(VMEXIT_HLT, v3 & INTERCEPT_HLT != 0),
            0x131 => simple(VMEXIT_RDTSC, v3 & INTERCEPT_RDTSC != 0),
            0x133 => simple(VMEXIT_RDPMC, v3 & INTERCEPT_RDPMC != 0),
            0x9c => simple(VMEXIT_PUSHF, v3 & INTERCEPT_PUSHF != 0),
            0x9d => simple(VMEXIT_POPF, v3 & INTERCEPT_POPF != 0),
            0xcf => simple(VMEXIT_IRET, v3 & INTERCEPT_IRET != 0),
            0xcd => simple(VMEXIT_INT, v3 & INTERCEPT_INT != 0),
            0x108 => simple(VMEXIT_INVD, v3 & INTERCEPT_INVD != 0),
            0x109 => simple(VMEXIT_WBINVD, v4 & INTERCEPT_WBINVD != 0),
            // pause is `f3 90`
            0x90 if i.rep == 1 => simple(VMEXIT_PAUSE, v3 & INTERCEPT_PAUSE != 0),
            0x132 | 0x130 => {
                let write = op == 0x130;
                if v3 & INTERCEPT_MSR_PROT != 0
                    && self.msr_is_intercepted(self.gpr[RCX] as u32, write)
                {
                    Some((VMEXIT_MSR, write as u64, 0))
                } else {
                    None
                }
            }
            // in/out and ins/outs
            0xe4..=0xe7 | 0xec..=0xef | 0x6c..=0x6f => {
                if v3 & INTERCEPT_IOIO_PROT == 0 {
                    None
                } else {
                    let byte = op & 1 == 0;
                    let size = if byte {
                        1
                    } else if i.opsize == 2 {
                        2
                    } else {
                        4
                    };
                    let string = (0x6c..=0x6f).contains(&op);
                    let input = matches!(op, 0xe4 | 0xe5 | 0xec | 0xed | 0x6c | 0x6d);
                    let port = if matches!(op, 0xe4..=0xe7) {
                        i.imm as u8 as u16
                    } else {
                        self.gpr[RDX] as u16
                    };
                    if self.port_is_intercepted(port, size) {
                        let info1 = (port as u64) << IOIO_PORT_SHIFT
                            | if input { IOIO_IN } else { 0 }
                            | if string { IOIO_STRING } else { 0 }
                            | if i.rep != 0 { IOIO_REP } else { 0 }
                            | match size {
                                1 => IOIO_SIZE_8,
                                2 => IOIO_SIZE_16,
                                _ => IOIO_SIZE_32,
                            }
                            | IOIO_ADDRESS_64;
                        Some((VMEXIT_IOIO, info1, next))
                    } else {
                        None
                    }
                }
            }
            // mov from/to control and debug registers
            0x120 => (cr_bits >> (i.reg & 15) & 1 != 0).then_some((
                VMEXIT_CR_READ + (i.reg & 15) as u64,
                0,
                0,
            )),
            0x122 => {
                let number = (i.reg & 15) as u32;
                if cr_bits >> (16 + number) & 1 != 0 {
                    Some((VMEXIT_CR_WRITE + number as u64, 0, 0))
                } else if number == 0
                    && v3 & INTERCEPT_CR0_SELECTIVE_WRITE != 0
                    && (self.gpr[i.rm as usize] ^ self.cr0) & !CR0_SELECTIVE_IGNORED != 0
                {
                    Some((VMEXIT_CR0_SELECTIVE_WRITE, 0, 0))
                } else {
                    None
                }
            }
            0x121 => (dr_bits >> (i.reg & 15) & 1 != 0).then_some((
                VMEXIT_DR_READ + (i.reg & 15) as u64,
                0,
                0,
            )),
            0x123 => (dr_bits >> (16 + (i.reg & 15) as u32) & 1 != 0).then_some((
                VMEXIT_DR_WRITE + (i.reg & 15) as u64,
                0,
                0,
            )),
            // group 6: sldt (/0), str (/1) read; lldt (/2), ltr (/3) write
            0x100 => match reg {
                0 => simple(VMEXIT_LDTR_READ, v3 & INTERCEPT_LDTR_READ != 0),
                1 => simple(VMEXIT_TR_READ, v3 & INTERCEPT_TR_READ != 0),
                2 => simple(VMEXIT_LDTR_WRITE, v3 & INTERCEPT_LDTR_WRITE != 0),
                3 => simple(VMEXIT_TR_WRITE, v3 & INTERCEPT_TR_WRITE != 0),
                _ => None,
            },
            // group 7
            0x101 if i.md != 3 => match reg {
                0 => simple(VMEXIT_GDTR_READ, v3 & INTERCEPT_GDTR_READ != 0),
                1 => simple(VMEXIT_IDTR_READ, v3 & INTERCEPT_IDTR_READ != 0),
                2 => simple(VMEXIT_GDTR_WRITE, v3 & INTERCEPT_GDTR_WRITE != 0),
                3 => simple(VMEXIT_IDTR_WRITE, v3 & INTERCEPT_IDTR_WRITE != 0),
                7 => simple(VMEXIT_INVLPG, v3 & INTERCEPT_INVLPG != 0),
                _ => None,
            },
            0x101 => match i.modrm {
                0xc8 => simple(VMEXIT_MONITOR, v4 & INTERCEPT_MONITOR != 0),
                0xc9 => simple(VMEXIT_MWAIT, v4 & INTERCEPT_MWAIT != 0),
                0xd1 => simple(VMEXIT_XSETBV, v4 & INTERCEPT_XSETBV != 0),
                0xd8 => simple(VMEXIT_VMRUN, v4 & INTERCEPT_VMRUN != 0),
                0xd9 => simple(VMEXIT_VMMCALL, v4 & INTERCEPT_VMMCALL != 0),
                0xda => simple(VMEXIT_VMLOAD, v4 & INTERCEPT_VMLOAD != 0),
                0xdb => simple(VMEXIT_VMSAVE, v4 & INTERCEPT_VMSAVE != 0),
                0xdc => simple(VMEXIT_STGI, v4 & INTERCEPT_STGI != 0),
                0xdd => simple(VMEXIT_CLGI, v4 & INTERCEPT_CLGI != 0),
                0xde => simple(VMEXIT_SKINIT, v4 & INTERCEPT_SKINIT != 0),
                0xdf => simple(VMEXIT_INVLPGA, v3 & INTERCEPT_INVLPGA != 0),
                0xf9 => simple(VMEXIT_RDTSCP, v4 & INTERCEPT_RDTSCP != 0),
                _ => None,
            },
            _ => None,
        };
        match hit {
            Some((code, info1, info2)) => Err(self.vmexit(code, info1, info2, next)),
            None => Ok(()),
        }
    }
}

// Nested page fault error code bits (EXITINFO1).
const NPF_PRESENT: u64 = 1 << 0;
const NPF_WRITE: u64 = 1 << 1;
const NPF_USER: u64 = 1 << 2;
const NPF_RESERVED: u64 = 1 << 3;
const NPF_FETCH: u64 = 1 << 4;
/// The fault happened on the final guest-physical address / while walking the guest's tables.
const NPF_FINAL_ADDRESS: u64 = 1 << 32;
const NPF_GUEST_TABLE_WALK: u64 = 1 << 33;
const NPT_PRESENT: u64 = 1 << 0;
const NPT_WRITABLE: u64 = 1 << 1;
const NPT_USER: u64 = 1 << 2;
const NPT_ACCESSED: u64 = 1 << 5;
const NPT_DIRTY: u64 = 1 << 6;
const NPT_PAGE_SIZE: u64 = 1 << 7;
const NPT_NO_EXECUTE: u64 = 1 << 63;
const NPT_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const NPT_LEVELS: u32 = 4;
const NPT_INDEX_BITS: u32 = 9;
const NPT_PAGE_SHIFT: u32 = 12;

impl Cpu {
    /// Guest-physical to host-physical through the nested page tables; the identity outside a
    /// guest or without nested paging. A fault is a #VMEXIT(NPF), reported as `Exception::Vmexit`.
    /// The nested tables always treat the access as a user-mode access, so the User bit must be
    /// set at every level.
    pub(crate) fn nested_translate(
        &mut self,
        guest_physical: u64,
        access: super::mem::AccessKind,
        guest_table_walk: bool,
    ) -> ExecResult<u64> {
        let Some(root) = self.svm.guest.as_ref().and_then(|guest| guest.nested_cr3) else {
            return Ok(guest_physical);
        };
        let write = access == super::mem::AccessKind::Write;
        let fetch = access == super::mem::AccessKind::Exec;
        let nx_enabled = self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.host.efer & EFER_NXE != 0);
        let fault = |cpu: &mut Cpu, present: bool, reserved: bool| {
            let info = (if present { NPF_PRESENT } else { 0 })
                | (if write { NPF_WRITE } else { 0 })
                | NPF_USER
                | (if reserved { NPF_RESERVED } else { 0 })
                | (if fetch { NPF_FETCH } else { 0 })
                | if guest_table_walk {
                    NPF_GUEST_TABLE_WALK
                } else {
                    NPF_FINAL_ADDRESS
                };
            Err(cpu.vmexit(VMEXIT_NPF, info, guest_physical, 0))
        };
        let mut table = root & NPT_ADDRESS_MASK;
        let mut writable = true;
        let mut no_execute = false;
        for level in (0..NPT_LEVELS).rev() {
            let shift = NPT_PAGE_SHIFT + NPT_INDEX_BITS * level;
            let index = (guest_physical >> shift) & ((1 << NPT_INDEX_BITS) - 1);
            let entry_address = table + index * 8;
            if !self.ram.contains(entry_address, 8) {
                return fault(self, false, false);
            }
            let mut entry = self.phys_read(entry_address, 8);
            if entry & NPT_PRESENT == 0 {
                return fault(self, false, false);
            }
            let leaf = level == 0 || (entry & NPT_PAGE_SIZE != 0 && level <= 2);
            let alignment_mask = (1u64 << shift) - 1;
            if (!nx_enabled && entry & NPT_NO_EXECUTE != 0)
                || (level == NPT_LEVELS - 1 && entry & NPT_PAGE_SIZE != 0)
                || (leaf && level > 0 && entry & alignment_mask & NPT_ADDRESS_MASK != 0)
            {
                return fault(self, true, true);
            }
            if entry & NPT_USER == 0 {
                return fault(self, true, false);
            }
            writable &= entry & NPT_WRITABLE != 0;
            no_execute |= entry & NPT_NO_EXECUTE != 0;
            if leaf {
                if (write && !writable) || (fetch && no_execute && nx_enabled) {
                    return fault(self, true, false);
                }
                let updated = entry | NPT_ACCESSED | if write { NPT_DIRTY } else { 0 };
                if updated != entry {
                    // A nested hypervisor on another vCPU may be changing the entry (clearing
                    // Dirty for dirty logging, say): update it as hardware does, and walk again
                    // if it changed under us.
                    if !self.update_ram_u64(entry_address, entry, updated) {
                        return self.nested_translate(guest_physical, access, guest_table_walk);
                    }
                    entry = updated;
                }
                let base = entry & NPT_ADDRESS_MASK & !alignment_mask;
                return Ok(base | (guest_physical & alignment_mask));
            }
            if entry & NPT_ACCESSED == 0
                && !self.update_ram_u64(entry_address, entry, entry | NPT_ACCESSED)
            {
                return self.nested_translate(guest_physical, access, guest_table_walk);
            }
            table = entry & NPT_ADDRESS_MASK;
        }
        fault(self, false, false)
    }
}

#[cfg(test)]
mod tests {
    //! Host snippet: `vmrun; hlt`. The guest runs on the host's own page tables and GDT (the VMCB
    //! is filled from the live CPU state), so guest snippets can be plain `testkit` code. The tests
    //! drive the real run-loop pieces (`before_step`, `step`, `after_step`), because the exception
    //! and interrupt intercepts live there, not in `Harness::run`.
    use super::*;
    use crate::cpu::BeforeStep;
    use crate::cpu::mem::AccessKind;
    use crate::cpu::test_support::{LegacyTables, PTE_P, PaeTables};
    use crate::testkit::RBX;
    use crate::testkit::{CODE_BASE, Harness, SCRATCH};

    /// Far from the start of the table pool (0x20_0000), which the guest page-table helpers fill.
    const VMCB: u64 = 0x3f_0000;
    const IOPM: u64 = 0x21_0000;
    const MSRPM: u64 = 0x25_0000;
    const NPT: u64 = 0x28_0000;
    const IDT: u64 = 0x60_0000;
    const GUEST_STACK: u64 = 0x70_0000;
    const GUEST_CODE: u64 = CODE_BASE + 0x1000;
    const HOST_CODE: &[u8] = &[0x0f, 0x01, 0xd8, 0xf4]; // vmrun; hlt
    const ASM_CPUID: &[u8] = &[0x0f, 0xa2];
    const ASM_HLT: &[u8] = &[0xf4];
    const UNMAPPED: u64 = 0x2_0000_0000;

    thread_local! {
        /// Set by `under_jit`: every rig built on this thread runs on the host JIT (if there is one).
        static USE_JIT: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    }

    /// Runs `test` with the JIT enabled in every rig it builds, so the intercepts, state swaps and
    /// nested translation are also exercised through compiled blocks and the `exec` helper.
    fn under_jit(test: fn()) {
        USE_JIT.with(|flag| flag.set(true));
        test();
        USE_JIT.with(|flag| flag.set(false));
    }

    fn put(h: &mut Harness, offset: u64, size: usize, value: u64) {
        h.write(VMCB + offset, &value.to_le_bytes()[..size]);
    }

    fn get(h: &Harness, offset: u64, size: usize) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[..size].copy_from_slice(&h.read(VMCB + offset, size));
        u64::from_le_bytes(bytes)
    }

    fn put_segment(h: &mut Harness, offset: u64, segment: Seg) {
        put(h, offset + SEGMENT_SELECTOR, 2, segment.sel as u64);
        put(
            h,
            offset + SEGMENT_ATTRIB,
            2,
            flags_to_attrib(segment.flags) as u64,
        );
        put(h, offset + SEGMENT_LIMIT, 4, segment.limit as u64);
        put(h, offset + SEGMENT_BASE, 8, segment.base);
    }

    /// A host that is about to `vmrun` a guest that executes `guest_code`.
    fn rig(guest_code: &[u8]) -> Harness {
        rig_prepared(guest_code, |_| ()).0
    }

    /// `rig` with a step that runs on the fresh harness before the VMCB is written: guest page
    /// tables built by the paging test helpers live in the pool that holds the VMCB, so they must
    /// go in first.
    fn rig_prepared<T>(guest_code: &[u8], prepare: impl FnOnce(&mut Harness) -> T) -> (Harness, T) {
        let mut h = Harness::new(HOST_CODE);
        let prepared = prepare(&mut h);
        h.cpu.efer |= EFER_SVME;
        h.write(GUEST_CODE, guest_code);
        let segments = [SEG_ES, SEG_CS, SEG_SS, SEG_DS].map(|index| h.cpu.seg[index]);
        for (offset, segment) in [SAVE_ES, SAVE_CS, SAVE_SS, SAVE_DS]
            .into_iter()
            .zip(segments)
        {
            put_segment(&mut h, offset, segment);
        }
        let (gdtr, idtr) = (h.cpu.gdtr, h.cpu.idtr);
        put(&mut h, SAVE_GDTR + SEGMENT_LIMIT, 4, gdtr.limit as u64);
        put(&mut h, SAVE_GDTR + SEGMENT_BASE, 8, gdtr.base);
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, idtr.limit as u64);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, idtr.base);
        let (efer, cr0, cr3, cr4) = (h.cpu.efer, h.cpu.cr0, h.cpu.cr3, h.cpu.cr4);
        put(&mut h, SAVE_EFER, 8, efer);
        put(&mut h, SAVE_CR0, 8, cr0);
        put(&mut h, SAVE_CR3, 8, cr3);
        put(&mut h, SAVE_CR4, 8, cr4);
        put(&mut h, SAVE_DR7, 8, 0x400);
        put(&mut h, SAVE_DR6, 8, 0xffff_0ff0);
        put(&mut h, SAVE_RFLAGS, 8, 2);
        put(&mut h, SAVE_RIP, 8, GUEST_CODE);
        put(&mut h, SAVE_RSP, 8, GUEST_STACK);
        put(&mut h, SAVE_RAX, 8, 0x1234);
        put(&mut h, VMCB_ASID, 4, 1);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_HLT) as u64,
        );
        put(&mut h, VMCB_INTERCEPT4, 4, INTERCEPT_VMRUN as u64);
        h.cpu.gpr[RAX] = VMCB;
        if USE_JIT.with(|flag| flag.get()) {
            h.enable_jit();
        }
        (h, prepared)
    }

    /// Runs the real loop until the host halts (or the budget ends).
    fn run(h: &mut Harness) {
        for _ in 0..200_000 {
            if h.cpu.halted && h.cpu.svm.guest.is_none() {
                return;
            }
            match h.cpu.before_step() {
                BeforeStep::Exit(reason) => panic!("machine exit: {reason:?}"),
                BeforeStep::Idled => continue,
                BeforeStep::Execute => {}
                BeforeStep::Yield | BeforeStep::Halted => unreachable!("one vCPU"),
            }
            // as in `Cpu::run`: the shadow of `sti`/`mov ss` covers exactly one instruction
            h.cpu.interrupt_shadow = false;
            h.cpu.icount += 1;
            let result = if h.cpu.jit.is_some() && h.cpu.is_long64() {
                h.cpu.step_block()
            } else {
                h.cpu.step()
            };
            if let Some(reason) = h.cpu.after_step(result) {
                panic!("machine exit: {reason:?}\n{}", h.cpu.dump_regs());
            }
        }
        panic!("budget exhausted\n{}", h.cpu.dump_regs());
    }

    #[test]
    fn a_long_mode_guest_exits_with_efer_lma_saved() {
        // Xen without intercepted CR0 writes learns that the guest entered long mode from this
        // bit; with it clear it truncates the guest's RIP to 32 bits after every emulated
        // instruction.
        let mut h = rig(&[0x0f, 0xa2, 0xf4]);
        run(&mut h);
        assert_eq!(get(&h, SAVE_EFER, 8) & EFER_LMA, EFER_LMA);
    }

    #[test]
    fn a_guest_with_automatic_ibrs_in_efer_runs_and_keeps_it() {
        let mut h = rig(&[0x0f, 0xa2, 0xf4]);
        let guest_efer = get(&h, SAVE_EFER, 8);
        put(&mut h, SAVE_EFER, 8, guest_efer | EFER_AIBRSE);
        run(&mut h);
        assert_eq!(
            get(&h, VMCB_EXITCODE, 8),
            VMEXIT_CPUID,
            "not VMEXIT_INVALID"
        );
        assert_ne!(get(&h, SAVE_EFER, 8) & EFER_AIBRSE, 0);
        assert_eq!(h.cpu.efer & EFER_AIBRSE, 0, "the host's EFER comes back");
    }

    #[test]
    fn a_guest_cpuid_exits_with_the_intercept_code_and_the_state_before_the_instruction() {
        let mut h = rig(&[0xb8, 7, 0, 0, 0, 0x0f, 0xa2, 0xf4]); // mov eax, 7; cpuid; hlt
        h.cpu.gpr[GPR_RSP] = 0x1111_0000;
        let host_rsp = h.cpu.gpr[GPR_RSP];
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE + 5,
            "RIP is the cpuid, not after it"
        );
        assert_eq!(get(&h, VMCB_NEXT_RIP, 8), GUEST_CODE + 7);
        assert_eq!(get(&h, SAVE_RAX, 8), 7);
        assert_eq!(get(&h, SAVE_RSP, 8), GUEST_STACK);
        // host state is back: RAX is the VMCB address again, RSP and RIP resumed after vmrun
        assert_eq!(h.cpu.gpr[RAX], VMCB);
        assert_eq!(h.cpu.gpr[GPR_RSP], host_rsp);
        assert_eq!(
            h.cpu.rip,
            CODE_BASE + 4,
            "the host halted on the hlt after vmrun"
        );
        assert!(h.cpu.svm.guest.is_none());
        assert!(!h.cpu.svm.gif, "GIF is clear after a #VMEXIT");
        assert_eq!(h.cpu.cpl, 0);
    }

    #[test]
    fn guest_gprs_are_shared_with_the_host_apart_from_rax_and_rsp() {
        let mut h = rig(&[0x48, 0xc7, 0xc3, 0x55, 0, 0, 0, 0x0f, 0xa2]); // mov rbx, 0x55; cpuid
        h.cpu.gpr[3] = 0x99;
        run(&mut h);
        assert_eq!(
            h.cpu.gpr[3], 0x55,
            "the hypervisor sees the guest's rbx: it swaps GPRs itself"
        );
    }

    #[test]
    fn hlt_is_intercepted_only_when_asked() {
        let mut h = rig(ASM_HLT);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_HLT);
    }

    #[test]
    fn instruction_intercept_table() {
        // (guest code, intercept3 bit, intercept4 bit, exit code)
        let cases: &[(&[u8], u32, u32, u64)] = &[
            (&[0x0f, 0x31], INTERCEPT_RDTSC, 0, VMEXIT_RDTSC),
            (&[0x0f, 0x33], INTERCEPT_RDPMC, 0, VMEXIT_RDPMC),
            (&[0x9c], INTERCEPT_PUSHF, 0, VMEXIT_PUSHF),
            (&[0x9d], INTERCEPT_POPF, 0, VMEXIT_POPF),
            (&[0xcd, 0x80], INTERCEPT_INT, 0, VMEXIT_INT),
            (&[0x0f, 0x08], INTERCEPT_INVD, 0, VMEXIT_INVD),
            (&[0xf3, 0x90], INTERCEPT_PAUSE, 0, VMEXIT_PAUSE),
            (&[0x0f, 0x09], 0, INTERCEPT_WBINVD, VMEXIT_WBINVD),
            (&[0x0f, 0x01, 0xc8], 0, INTERCEPT_MONITOR, VMEXIT_MONITOR),
            (&[0x0f, 0x01, 0xc9], 0, INTERCEPT_MWAIT, VMEXIT_MWAIT),
            (&[0x0f, 0x01, 0xd1], 0, INTERCEPT_XSETBV, VMEXIT_XSETBV),
            (&[0x0f, 0x01, 0xf9], 0, INTERCEPT_RDTSCP, VMEXIT_RDTSCP),
            (&[0x0f, 0x01, 0xd9], 0, INTERCEPT_VMMCALL, VMEXIT_VMMCALL),
            (&[0x0f, 0x01, 0xda], 0, INTERCEPT_VMLOAD, VMEXIT_VMLOAD),
            (&[0x0f, 0x01, 0xdb], 0, INTERCEPT_VMSAVE, VMEXIT_VMSAVE),
            (&[0x0f, 0x01, 0xdc], 0, INTERCEPT_STGI, VMEXIT_STGI),
            (&[0x0f, 0x01, 0xdd], 0, INTERCEPT_CLGI, VMEXIT_CLGI),
            (&[0x0f, 0x01, 0xde], 0, INTERCEPT_SKINIT, VMEXIT_SKINIT),
            (&[0x0f, 0x01, 0xdf], INTERCEPT_INVLPGA, 0, VMEXIT_INVLPGA),
            (&[0x0f, 0x01, 0xd8], 0, 0, VMEXIT_VMRUN),
            (
                &[0x0f, 0x01, 0x3d, 0, 0, 0, 0],
                INTERCEPT_INVLPG,
                0,
                VMEXIT_INVLPG,
            ),
        ];
        for &(code, v3, v4, expected) in cases {
            let mut h = rig(code);
            put(&mut h, VMCB_INTERCEPT3, 4, v3 as u64);
            put(&mut h, VMCB_INTERCEPT4, 4, (INTERCEPT_VMRUN | v4) as u64);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), expected, "{code:02x?}");
            assert_eq!(
                get(&h, SAVE_RIP, 8),
                GUEST_CODE,
                "{code:02x?}: intercepts leave RIP on the instruction"
            );
        }
    }

    #[test]
    fn cr_and_dr_moves_are_intercepted_per_register() {
        // mov cr3, rax / mov rax, cr0 / mov dr7, rax / mov rax, dr6
        let write_cr3 = [0x0f, 0x22, 0xd8];
        let read_cr0 = [0x0f, 0x20, 0xc0];
        let write_dr7 = [0x0f, 0x23, 0xf8];
        let read_dr6 = [0x0f, 0x21, 0xf0];
        for (code, cr, dr, expected) in [
            (write_cr3, 1 << (16 + 3), 0, VMEXIT_CR_WRITE + 3),
            (read_cr0, 1 << 0, 0, VMEXIT_CR_READ),
            (write_dr7, 0, 1 << (16 + 7), VMEXIT_DR_WRITE + 7),
            (read_dr6, 0, 1 << 6, VMEXIT_DR_READ + 6),
        ] {
            let mut h = rig(&code);
            put(&mut h, VMCB_INTERCEPT_CR, 4, cr);
            put(&mut h, VMCB_INTERCEPT_DR, 4, dr);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), expected, "{code:02x?}");
        }
        // an unrelated register does not exit: writing cr3 with only the cr0 write bit set
        let mut h = rig(&[0x0f, 0x22, 0xd8, 0x0f, 0xa2]);
        put(&mut h, VMCB_INTERCEPT_CR, 4, 1 << 16);
        let cr3 = h.cpu.cr3;
        put(&mut h, SAVE_RAX, 8, cr3);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
    }

    #[test]
    fn cr0_selective_write_ignores_ts_and_mp_only() {
        // mov cr0, rax with rax = current cr0 ^ TS: no exit; ^ CD: exit
        let mut h = rig(&[0x0f, 0x22, 0xc0, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_CR0_SELECTIVE_WRITE) as u64,
        );
        let cr0 = h.cpu.cr0;
        put(&mut h, SAVE_RAX, 8, cr0 ^ CR0_TS);
        run(&mut h);
        assert_eq!(
            get(&h, VMCB_EXITCODE, 8),
            VMEXIT_CPUID,
            "TS alone is not intercepted"
        );
        let mut h = rig(&[0x0f, 0x22, 0xc0, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_CR0_SELECTIVE_WRITE) as u64,
        );
        let cr0 = h.cpu.cr0;
        put(&mut h, SAVE_RAX, 8, cr0 ^ CR0_CD_BIT);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CR0_SELECTIVE_WRITE);
    }

    #[test]
    fn port_io_follows_the_permission_map_and_reports_the_access() {
        // out 0x80, al; in al, 0x81; cpuid
        let mut h = rig(&[0xe6, 0x80, 0xe4, 0x81, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_IOIO_PROT) as u64,
        );
        put(&mut h, VMCB_IOPM_BASE, 8, IOPM);
        h.write(IOPM, &[0u8; 0x3000]);
        h.write(IOPM + 0x80 / 8, &[1 << (0x80 % 8) | 1 << (0x81 % 8)]);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_IOIO);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(info >> IOIO_PORT_SHIFT, 0x80);
        assert_eq!(info & IOIO_IN, 0, "out");
        assert_ne!(info & IOIO_SIZE_8, 0);
        assert_eq!(
            get(&h, VMCB_EXITINFO2, 8),
            GUEST_CODE + 2,
            "info2 is the next instruction"
        );
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        // an unmapped port passes through; a 16-bit access straddling a set bit exits
        let mut h = rig(&[0x66, 0xe5, 0x7f, 0x0f, 0xa2]); // in ax, 0x7f
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_IOIO_PROT) as u64,
        );
        put(&mut h, VMCB_IOPM_BASE, 8, IOPM);
        h.write(IOPM, &[0u8; 0x3000]);
        h.write(IOPM + 0x80 / 8, &[1 << (0x80 % 8)]);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_IOIO);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(
            (
                info >> IOIO_PORT_SHIFT,
                info & IOIO_IN != 0,
                info & IOIO_SIZE_16 != 0
            ),
            (0x7f, true, true)
        );
        let mut h = rig(&[0xe6, 0x82, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_IOIO_PROT) as u64,
        );
        put(&mut h, VMCB_IOPM_BASE, 8, IOPM);
        h.write(IOPM, &[0u8; 0x3000]);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
    }

    #[test]
    fn rdmsr_and_wrmsr_follow_the_msr_permission_map() {
        const EFER: u32 = 0xc000_0080;
        let efer_bit = (EFER - 0xc000_0000) as u64 * 2;
        // rdmsr (ecx = EFER); wrmsr; cpuid
        let code = [0x0f, 0x32, 0x0f, 0x30, 0x0f, 0xa2];
        // read intercepted
        let mut h = rig(&code);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_MSR_PROT) as u64,
        );
        put(&mut h, VMCB_MSRPM_BASE, 8, MSRPM);
        h.write(MSRPM, &[0u8; 0x2000]);
        h.write(MSRPM + 0x800 + efer_bit / 8, &[1 << (efer_bit % 8)]);
        h.cpu.gpr[RCX] = EFER as u64;
        run(&mut h);
        assert_eq!(
            (get(&h, VMCB_EXITCODE, 8), get(&h, VMCB_EXITINFO1, 8)),
            (VMEXIT_MSR, 0)
        );
        // write bit is the odd one
        let mut h = rig(&[0x0f, 0x30, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_MSR_PROT) as u64,
        );
        put(&mut h, VMCB_MSRPM_BASE, 8, MSRPM);
        h.write(MSRPM, &[0u8; 0x2000]);
        h.write(
            MSRPM + 0x800 + (efer_bit + 1) / 8,
            &[1 << ((efer_bit + 1) % 8)],
        );
        h.cpu.gpr[RCX] = EFER as u64;
        run(&mut h);
        assert_eq!(
            (get(&h, VMCB_EXITCODE, 8), get(&h, VMCB_EXITINFO1, 8)),
            (VMEXIT_MSR, 1)
        );
        // an MSR outside the three ranges always exits
        let mut h = rig(&code);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_MSR_PROT) as u64,
        );
        put(&mut h, VMCB_MSRPM_BASE, 8, MSRPM);
        h.write(MSRPM, &[0u8; 0x2000]);
        h.cpu.gpr[RCX] = 0x4000_0000;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_MSR);
        // a clear map passes the access to the guest (rdmsr of a known MSR works)
        let mut h = rig(&[0x0f, 0x32, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_MSR_PROT) as u64,
        );
        put(&mut h, VMCB_MSRPM_BASE, 8, MSRPM);
        h.write(MSRPM, &[0u8; 0x2000]);
        h.cpu.gpr[RCX] = EFER as u64;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
    }

    #[test]
    fn intercepted_exceptions_exit_with_their_error_code_and_cr2() {
        // ud2 with #UD (6) intercepted
        let mut h = rig(&[0x0f, 0x0b]);
        put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 6);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_EXCEPTION + 6);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        // a page fault: mov rax, [UNMAPPED]
        let mut h = rig(&[0x48, 0xa1, 0, 0, 0, 0, 2, 0, 0, 0]);
        put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 14);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_EXCEPTION + 14);
        assert_eq!(
            get(&h, VMCB_EXITINFO1, 8),
            0,
            "not present, read, supervisor"
        );
        assert_eq!(
            get(&h, VMCB_EXITINFO2, 8),
            UNMAPPED,
            "info2 is the faulting address"
        );
        // an exception that is not intercepted is delivered to the guest: with an empty IDT the
        // guest takes #GP then #DF then triple-faults, which (with shutdown intercepted) exits
        let mut h = rig(&[0x0f, 0x0b]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_SHUTDOWN) as u64,
        );
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_SHUTDOWN);
    }

    fn consistency_failure(mutate: impl FnOnce(&mut Harness)) {
        let mut h = rig(&[0x0f, 0xa2]);
        mutate(&mut h);
        let rsp = h.cpu.gpr[GPR_RSP];
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INVALID);
        assert!(h.cpu.svm.guest.is_none(), "the guest never started");
        assert_eq!(h.cpu.gpr[GPR_RSP], rsp);
        assert_eq!(h.cpu.rip, CODE_BASE + 4, "the host continues after vmrun");
    }

    #[test]
    fn an_inconsistent_vmcb_is_vmexit_invalid_and_the_host_carries_on() {
        consistency_failure(|h| put(h, SAVE_EFER, 8, EFER_LME | EFER_LMA)); // SVME clear
        consistency_failure(|h| put(h, VMCB_ASID, 4, 0));
        consistency_failure(|h| put(h, VMCB_INTERCEPT4, 4, 0)); // VMRUN not intercepted
        consistency_failure(|h| put(h, SAVE_CR0, 8, 1 << 32));
        consistency_failure(|h| {
            let cr0 = h.cpu.cr0;
            put(h, SAVE_CR0, 8, (cr0 | CR0_NW_BIT) & !CR0_CD_BIT);
        });
        consistency_failure(|h| put(h, SAVE_DR7, 8, 1 << 32));
        consistency_failure(|h| put(h, SAVE_CR3, 8, 1 << 50));
        consistency_failure(|h| {
            let cr4 = h.cpu.cr4;
            put(h, SAVE_CR4, 8, cr4 | (1 << 63));
        });
        // long mode needs PAE and protection
        consistency_failure(|h| {
            let cr4 = h.cpu.cr4;
            put(h, SAVE_CR4, 8, cr4 & !CR4_PAE);
        });
        consistency_failure(|h| {
            let cr0 = h.cpu.cr0;
            put(h, SAVE_CR0, 8, cr0 & !CR0_PE_BIT);
        });
    }

    /// Replaces the guest's mode: control registers, EFER and flat segments for `bits`-bit code.
    fn set_guest_mode(h: &mut Harness, efer: u64, cr0: u64, cr3: u64, cr4: u64, code_flags: u32) {
        put(h, SAVE_EFER, 8, efer | EFER_SVME);
        put(h, SAVE_CR0, 8, cr0);
        put(h, SAVE_CR3, 8, cr3);
        put(h, SAVE_CR4, 8, cr4);
        let code = Seg {
            sel: 8,
            base: 0,
            limit: SEG_LIMIT_4GIB,
            flags: code_flags,
        };
        put_segment(h, SAVE_CS, code);
        let data = Seg {
            sel: 16,
            base: 0,
            limit: SEG_LIMIT_4GIB,
            flags: SEG_FLAGS_FLAT_DATA,
        };
        for offset in [SAVE_ES, SAVE_SS, SAVE_DS] {
            put_segment(h, offset, data);
        }
    }

    const FLAT_CODE32: u32 = super::super::sys::SEG_FLAGS_CODE32;
    /// mov eax, 0x5aa5; cpuid
    const GUEST_MOV_CPUID_32: &[u8] = &[0xb8, 0xa5, 0x5a, 0, 0, 0x0f, 0xa2];

    fn assert_guest_exited_on_cpuid(h: &Harness, rip: u64) {
        assert_eq!(get(h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(
            get(h, SAVE_RAX, 8),
            0x5aa5,
            "the guest ran its first instruction"
        );
        assert_eq!(get(h, SAVE_RIP, 8), rip, "stopped on the cpuid");
        assert!(h.cpu.svm.guest.is_none());
        assert!(h.cpu.is_long64(), "the host is back in 64-bit mode");
    }

    #[test]
    fn a_protected_mode_guest_without_paging_runs() {
        let mut h = rig(GUEST_MOV_CPUID_32);
        set_guest_mode(&mut h, 0, CR0_PE_BIT, 0, 0, FLAT_CODE32);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, GUEST_CODE + 5);
    }

    #[test]
    fn a_real_mode_guest_runs_with_sixteen_bit_decoding() {
        // mov eax, 0x5aa5 (operand-size prefix); cpuid
        let mut h = rig(&[0x66, 0xb8, 0xa5, 0x5a, 0, 0, 0x0f, 0xa2]);
        h.write(0x7c00, &[0x66, 0xb8, 0xa5, 0x5a, 0, 0, 0x0f, 0xa2]);
        set_guest_mode(&mut h, 0, 0, 0, 0, 0);
        let code = crate::cpu::modes::real_mode_segment(0, 0);
        put_segment(
            &mut h,
            SAVE_CS,
            Seg {
                flags: code.flags | SEG_TYPE_EXECUTABLE,
                ..code
            },
        );
        for offset in [SAVE_ES, SAVE_SS, SAVE_DS] {
            put_segment(&mut h, offset, code);
        }
        put(&mut h, SAVE_RIP, 8, 0x7c00);
        put(&mut h, SAVE_RSP, 8, 0x7000);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, 0x7c06);
    }

    /// Builds guest tables with `build` and hands back the control registers they imply, leaving
    /// the CPU as it was.
    fn guest_registers(h: &mut Harness, build: impl FnOnce(&mut Cpu)) -> (u64, u64, u64, u64) {
        let host = (h.cpu.efer, h.cpu.cr0, h.cpu.cr3, h.cpu.cr4, h.cpu.pdpte);
        build(&mut h.cpu);
        let guest = (h.cpu.efer, h.cpu.cr0, h.cpu.cr3, h.cpu.cr4);
        (h.cpu.efer, h.cpu.cr0, h.cpu.cr3, h.cpu.cr4, h.cpu.pdpte) = host;
        h.cpu.flush_tlb();
        guest
    }

    #[test]
    fn a_guest_with_thirty_two_bit_paging_runs() {
        let (mut h, guest) = rig_prepared(GUEST_MOV_CPUID_32, |h| {
            guest_registers(h, |cpu| LegacyTables::new(cpu).identity_map(cpu, 32))
        });
        set_guest_mode(&mut h, guest.0, guest.1, guest.2, guest.3, FLAT_CODE32);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, GUEST_CODE + 5);
    }

    #[test]
    fn a_pae_guest_latches_its_pdptes_and_the_host_gets_its_own_back() {
        let (mut h, guest) = rig_prepared(GUEST_MOV_CPUID_32, |h| {
            guest_registers(h, |cpu| PaeTables::new(cpu).identity_map(cpu, 32))
        });
        let host_pdpte = h.cpu.pdpte;
        set_guest_mode(&mut h, guest.0, guest.1, guest.2, guest.3, FLAT_CODE32);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, GUEST_CODE + 5);
        assert_eq!(h.cpu.pdpte, host_pdpte);
    }

    #[test]
    fn a_pae_guest_with_a_reserved_pdpte_bit_is_vmexit_invalid() {
        let (mut h, guest) = rig_prepared(GUEST_MOV_CPUID_32, |h| {
            guest_registers(h, |cpu| {
                let tables = PaeTables::new(cpu);
                cpu.ram[tables.pdpt as usize..][..8]
                    .copy_from_slice(&(PTE_P | 1 << 63).to_le_bytes());
            })
        });
        set_guest_mode(&mut h, guest.0, guest.1, guest.2, guest.3, FLAT_CODE32);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INVALID);
        assert!(h.cpu.svm.guest.is_none());
        assert!(h.cpu.is_long64());
    }

    #[test]
    fn a_compatibility_mode_guest_runs_under_long_mode_paging() {
        let mut h = rig(GUEST_MOV_CPUID_32);
        let (efer, cr0, cr3, cr4) = (h.cpu.efer, h.cpu.cr0, h.cpu.cr3, h.cpu.cr4);
        set_guest_mode(&mut h, efer, cr0, cr3, cr4, FLAT_CODE32);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, GUEST_CODE + 5);
    }

    #[test]
    fn svm_instructions_need_svme_and_cpl_zero_and_an_aligned_vmcb() {
        // #UD without EFER.SVME
        let mut h = Harness::new(HOST_CODE);
        h.cpu.gpr[RAX] = VMCB;
        assert!(matches!(
            h.run(5),
            crate::testkit::Stop::Fault { vector: 6, .. }
        ));
        // #GP for a misaligned address, or one above the physical address space
        for address in [VMCB + 8, 1 << 45] {
            let mut h = rig(&[0xf4]);
            h.cpu.gpr[RAX] = address;
            assert!(
                matches!(h.run(5), crate::testkit::Stop::Fault { vector: 13, .. }),
                "{address:#x}"
            );
        }
        // #GP outside CPL 0 (called directly: the host snippet's page is supervisor-only)
        let mut h = rig(&[0xf4]);
        h.cpu.set_cpl(3);
        for result in [
            h.cpu.vmrun(0),
            h.cpu.vmload_vmsave(false, 0),
            h.cpu.stgi_clgi(true, 0),
            h.cpu.invlpga(0),
        ] {
            assert!(matches!(result, Err(Exception::Fault { vector: 13, .. })));
        }
        // and the two that need no VMCB: stgi, clgi
        for code in [[0x0f, 0x01, 0xdc, 0xf4], [0x0f, 0x01, 0xdd, 0xf4]] {
            let mut h = Harness::new(&code);
            assert!(matches!(
                h.run(5),
                crate::testkit::Stop::Fault { vector: 6, .. }
            ));
        }
    }

    #[test]
    fn stgi_and_clgi_toggle_the_global_interrupt_flag_and_gate_interrupt_delivery() {
        let mut h = Harness::new(&[0x0f, 0x01, 0xdd, 0x0f, 0x01, 0xdc, 0xf4]); // clgi; stgi; hlt
        h.cpu.efer |= EFER_SVME;
        h.cpu.step().unwrap();
        assert!(!h.cpu.svm.gif);
        h.cpu.step().unwrap();
        assert!(h.cpu.svm.gif);
        // with GIF clear and IF set, a pending interrupt is not taken
        let mut h = Harness::new(&[0x90, 0x90, 0xf4]);
        init_pic(&mut h);
        h.cpu.efer |= EFER_SVME;
        h.cpu.eflags |= IF;
        h.cpu.svm.gif = false;
        h.cpu.dev.pic.raise(1);
        let rsp = h.cpu.gpr[GPR_RSP];
        let _ = h.cpu.before_step();
        assert_eq!(h.cpu.gpr[GPR_RSP], rsp, "no interrupt frame was pushed");
        assert!(h.cpu.dev.pic.has_pending());
    }

    fn init_pic(h: &mut Harness) {
        for (port, value) in [
            (0x20, 0x11),
            (0x21, 0x20),
            (0x21, 4),
            (0x21, 1),
            (0xa0, 0x11),
            (0xa1, 0x28),
            (0xa1, 2),
            (0xa1, 1),
            (0x21, 0),
            (0xa1, 0),
        ] {
            h.cpu.dev.io_out(port, 1, value, 0);
        }
    }

    #[test]
    fn a_physical_interrupt_exits_when_intercepted() {
        let mut h = rig(&[0x90, 0x90, 0x90, 0x0f, 0xa2]);
        init_pic(&mut h);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_INTR) as u64,
        );
        h.cpu.dev.pic.raise(1);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INTR);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE,
            "taken before the first instruction"
        );
        assert!(
            h.cpu.dev.pic.has_pending(),
            "the interrupt stays pending for the host"
        );
    }

    #[test]
    fn vmload_and_vmsave_move_the_state_vmrun_leaves_alone() {
        let mut h = Harness::new(&[0x0f, 0x01, 0xdb, 0x0f, 0x01, 0xda, 0xf4]); // vmsave; vmload; hlt
        h.cpu.efer |= EFER_SVME;
        h.cpu.gpr[RAX] = VMCB;
        h.cpu.seg[SEG_FS].base = 0x1111_2222;
        h.cpu.seg[SEG_GS].base = 0x3333_4444;
        h.cpu.kgsbase = 0x5555;
        h.cpu.star = 0x66;
        h.cpu.lstar = 0xffff_ffff_8000_0000;
        h.cpu.cstar = 0x88;
        h.cpu.sfmask = 0x99;
        h.cpu.sysenter = [1, 2, 3];
        h.cpu.tr = Seg {
            sel: 0x40,
            base: 0x7000,
            limit: 0x67,
            flags: 0x8b00,
        };
        h.cpu.ldtr = Seg {
            sel: 0x50,
            base: 0x8000,
            limit: 0xf,
            flags: 0x8200,
        };
        h.cpu.step().unwrap();
        assert_eq!(get(&h, SAVE_FS + SEGMENT_BASE, 8), 0x1111_2222);
        assert_eq!(get(&h, SAVE_KERNEL_GS_BASE, 8), 0x5555);
        assert_eq!(get(&h, SAVE_LSTAR, 8), 0xffff_ffff_8000_0000);
        assert_eq!(get(&h, SAVE_SYSENTER_EIP, 8), 3);
        assert_eq!(get(&h, SAVE_TR + SEGMENT_SELECTOR, 2), 0x40);
        assert_eq!(get(&h, SAVE_TR + SEGMENT_ATTRIB, 2), 0x8b);
        // scramble the live state, then load it back
        h.cpu.seg[SEG_FS].base = 0;
        h.cpu.kgsbase = 0;
        h.cpu.lstar = 0;
        h.cpu.tr = Seg::default();
        h.cpu.sysenter = [0; 3];
        h.cpu.step().unwrap();
        assert_eq!(h.cpu.seg[SEG_FS].base, 0x1111_2222);
        assert_eq!(
            (h.cpu.kgsbase, h.cpu.star, h.cpu.cstar, h.cpu.sfmask),
            (0x5555, 0x66, 0x88, 0x99)
        );
        assert_eq!(h.cpu.sysenter, [1, 2, 3]);
        assert_eq!(
            (h.cpu.tr.sel, h.cpu.tr.base, h.cpu.tr.limit, h.cpu.tr.flags),
            (0x40, 0x7000, 0x67, 0x8b00)
        );
        assert_eq!(h.cpu.ldtr.flags, 0x8200);
    }

    fn install_idt_gate(h: &mut Harness, vector: u8, handler: u64, selector: u16) {
        let low = (handler & 0xffff)
            | ((selector as u64) << 16)
            | (0x8eu64 << 40)
            | (((handler >> 16) & 0xffff) << 48);
        h.write(IDT + vector as u64 * 16, &low.to_le_bytes());
        h.write(IDT + vector as u64 * 16 + 8, &(handler >> 32).to_le_bytes());
    }

    #[test]
    fn event_injection_delivers_through_the_guest_idt_before_the_first_instruction() {
        // handler: mov ebx, 0x99; cpuid. The guest's own code would be `hlt` (intercepted).
        let handler = GUEST_CODE + 0x100;
        for (event, expected_stack_words) in [
            (
                EVENTINJ_VALID | (EVENT_TYPE_INTERRUPT << EVENTINJ_TYPE_SHIFT) | 0x30,
                5u64,
            ),
            (
                EVENTINJ_VALID
                    | (EVENT_TYPE_EXCEPTION << EVENTINJ_TYPE_SHIFT)
                    | 0x31
                    | EVENTINJ_ERROR_VALID,
                6,
            ),
            (
                EVENTINJ_VALID | (EVENT_TYPE_SOFTWARE << EVENTINJ_TYPE_SHIFT) | 0x30,
                5,
            ),
        ] {
            let mut h = rig(ASM_HLT);
            h.write(handler, &[0xbb, 0x99, 0, 0, 0, 0x0f, 0xa2]);
            let cs = h.cpu.seg[SEG_CS].sel;
            install_idt_gate(&mut h, 0x30, handler, cs);
            install_idt_gate(&mut h, 0x31, handler, cs);
            put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
            put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
            put(&mut h, VMCB_EVENTINJ, 4, event as u64);
            put(&mut h, VMCB_EVENTINJ_ERROR, 4, 0xe);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID, "{event:#x}");
            assert_eq!(
                get(&h, SAVE_RIP, 8),
                handler + 5,
                "the guest ran its handler"
            );
            assert_eq!(h.cpu.gpr[3], 0x99);
            assert_eq!(
                get(&h, SAVE_RSP, 8),
                (GUEST_STACK & !0xf) - expected_stack_words * 8,
                "event {event:#x}"
            );
            assert_eq!(get(&h, VMCB_EVENTINJ, 4), 0, "the valid bit is consumed");
        }
    }

    #[test]
    fn the_tsc_offset_applies_to_guest_reads_only() {
        // rdtsc; mov ebx, eax; mov ecx, edx; cpuid (rax is the host's again after the exit)
        let mut h = rig(&[0x0f, 0x31, 0x89, 0xc3, 0x89, 0xd1, 0x0f, 0xa2]);
        put(&mut h, VMCB_TSC_OFFSET, 8, 1_000_000_000);
        run(&mut h);
        let guest_tsc = (h.cpu.gpr[1] << 32) | (h.cpu.gpr[3] & 0xffff_ffff);
        assert!(guest_tsc >= 1_000_000_000, "{guest_tsc}");
        assert!(
            h.cpu.tsc() < 1_000_000_000,
            "the host's counter is unchanged"
        );
    }

    #[test]
    fn a_pending_virtual_interrupt_is_injected_when_the_guest_can_take_it() {
        let handler = GUEST_CODE + 0x100;
        // sti; nop; jmp +0; nop; hlt: the jmp ends a JIT block, which is the granularity at which
        // interrupts are taken, so the injection point after `sti`'s shadow exists in both modes
        let mut h = rig(&[0xfb, 0x90, 0xeb, 0x00, 0x90, 0xf4]);
        h.write(handler, &[0x0f, 0xa2]); // cpuid
        let cs = h.cpu.seg[SEG_CS].sel;
        install_idt_gate(&mut h, 0x40, handler, cs);
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
        put(
            &mut h,
            VMCB_V_INTR,
            8,
            V_IRQ | V_IGNORE_TPR | (0x40u64 << V_VECTOR_SHIFT),
        );
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            handler,
            "the virtual interrupt ran the handler"
        );
        assert_eq!(get(&h, VMCB_V_INTR, 8) & V_IRQ, 0, "V_IRQ is consumed");
        // with the VINTR intercept the exit comes first, once IF is set
        let mut h = rig(&[0xfb, 0x90, 0xeb, 0x00, 0x90, 0xf4]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_HLT | INTERCEPT_VINTR) as u64,
        );
        put(
            &mut h,
            VMCB_V_INTR,
            8,
            V_IRQ | V_IGNORE_TPR | (0x40u64 << V_VECTOR_SHIFT),
        );
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_VINTR);
        assert_ne!(get(&h, VMCB_V_INTR, 8) & V_IRQ, 0, "still pending");
    }

    // ---- nested paging ----

    const NESTED_REMAPPED_GPA: u64 = 0x80_0000;
    const NESTED_REMAPPED_HPA: u64 = 0x90_0000;

    /// Identity nested tables for the first 1 GiB (2 MiB pages), except that the 2 MiB page at
    /// `NESTED_REMAPPED_GPA` is split into 4 KiB pages and its first page points elsewhere.
    fn build_nested_tables(h: &mut Harness) {
        let pml4 = NPT;
        let pdpt = NPT + 0x1000;
        let pd = NPT + 0x2000;
        let pt = NPT + 0x3000;
        let user_rw_present = 7u64;
        h.write(pml4, &(pdpt | user_rw_present).to_le_bytes());
        h.write(pdpt, &(pd | user_rw_present).to_le_bytes());
        for k in 0..512u64 {
            let entry = if k * 0x20_0000 == NESTED_REMAPPED_GPA {
                pt | user_rw_present
            } else {
                (k * 0x20_0000) | user_rw_present | (1 << 7)
            };
            h.write(pd + k * 8, &entry.to_le_bytes());
        }
        for k in 0..512u64 {
            let host = if k == 0 {
                NESTED_REMAPPED_HPA
            } else {
                NESTED_REMAPPED_GPA + k * 0x1000
            };
            h.write(pt + k * 8, &(host | user_rw_present).to_le_bytes());
        }
    }

    fn nested_rig(guest_code: &[u8]) -> Harness {
        let mut h = rig(guest_code);
        build_nested_tables(&mut h);
        put(&mut h, VMCB_NP_ENABLE, 8, NP_ENABLE);
        put(&mut h, VMCB_N_CR3, 8, NPT);
        put(&mut h, SAVE_G_PAT, 8, 0x0007_0406_0007_0406);
        h
    }

    #[test]
    fn nested_paging_remaps_guest_physical_addresses() {
        // mov rax, [0x800000]; cpuid : the guest's identity-mapped page 0x800000 is guest-physical
        // 0x800000, which the nested tables send to 0x900000.
        let mut h = nested_rig(&[0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2]);
        h.write(NESTED_REMAPPED_HPA, &0xfeed_face_cafe_beefu64.to_le_bytes());
        h.write(NESTED_REMAPPED_GPA, &0x1111_1111_1111_1111u64.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(get(&h, SAVE_RAX, 8), 0xfeed_face_cafe_beef);
    }

    #[test]
    fn a_guest_without_paging_still_goes_through_the_nested_tables() {
        // Real mode, CS.base = the remapped guest-physical page; the code exists only at the
        // host-physical page behind it, so the fetch must go through nCR3.
        let mut h = nested_rig(ASM_HLT);
        h.write(
            NESTED_REMAPPED_HPA,
            &[0x66, 0xb8, 0xa5, 0x5a, 0, 0, 0x0f, 0xa2],
        );
        set_guest_mode(&mut h, 0, 0, 0, 0, 0);
        let code = crate::cpu::modes::real_mode_segment(0, 0);
        let code_segment = Seg {
            sel: (NESTED_REMAPPED_GPA >> 4) as u16,
            base: NESTED_REMAPPED_GPA,
            flags: code.flags | SEG_TYPE_EXECUTABLE,
            ..code
        };
        put_segment(&mut h, SAVE_CS, code_segment);
        for offset in [SAVE_ES, SAVE_SS, SAVE_DS] {
            put_segment(&mut h, offset, code);
        }
        put(&mut h, SAVE_RIP, 8, 0);
        put(&mut h, SAVE_RSP, 8, 0x7000);
        run(&mut h);
        assert_guest_exited_on_cpuid(&h, 6);
    }

    #[test]
    fn nested_page_faults_exit_with_the_gpa_and_error_code() {
        // unmap the remapped page in the nested tables, then read it
        let mut h = nested_rig(&[0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2]);
        h.write(NPT + 0x3000, &0u64.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), NESTED_REMAPPED_GPA);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(info & NPF_PRESENT, 0);
        assert_ne!(info & NPF_USER, 0);
        assert_ne!(info & NPF_FINAL_ADDRESS, 0);
        assert_eq!(info & NPF_WRITE, 0);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE,
            "the access restarts after the host fixes the tables"
        );
        // a write to a read-only nested page is a protection fault with the write bit
        let mut h = nested_rig(&[0x48, 0xa3, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2]); // mov [0x800000], rax
        h.write(NPT + 0x3000, &(NESTED_REMAPPED_HPA | 5).to_le_bytes()); // present, user, not writable
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(
            (info & NPF_PRESENT != 0, info & NPF_WRITE != 0),
            (true, true)
        );
    }

    #[test]
    fn a_nested_fault_in_the_middle_of_a_block_saves_the_guest_and_restores_the_host() {
        // The #VMEXIT happens inside the memory access, with earlier instructions of the same
        // block already executed (a JIT block keeps registers in host registers and must not
        // write them back over the host state the exit just restored).
        let mut h = nested_rig(&[
            0xbb, 0x11, 0, 0, 0, // mov ebx, 0x11
            0xb8, 0x22, 0, 0, 0, // mov eax, 0x22
            0x48, 0x89, 0x03, // mov [rbx], rax: placeholder, rewritten below
            0x0f, 0xa2,
        ]);
        h.write(
            GUEST_CODE,
            &[
                0xbb, 0, 0, 0x80, 0, // mov ebx, 0x800000
                0xb8, 0x22, 0, 0, 0, // mov eax, 0x22
                0x48, 0x89, 0x03, // mov [rbx], rax
                0x0f, 0xa2,
            ],
        );
        h.write(NPT + 0x3000, &0u64.to_le_bytes());
        let host_rsp = h.cpu.gpr[GPR_RSP];
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 10);
        assert_eq!(get(&h, SAVE_RAX, 8), 0x22);
        assert_eq!(h.cpu.gpr[RBX], NESTED_REMAPPED_GPA);
        assert_eq!(h.cpu.gpr[RAX], VMCB, "the host's RAX is back");
        assert_eq!(h.cpu.gpr[GPR_RSP], host_rsp);
        assert_eq!(h.cpu.rip, CODE_BASE + 4, "the host resumes after vmrun");
    }

    #[test]
    fn a_nested_fault_while_delivering_an_event_reports_it_in_exit_int_info() {
        let handler = GUEST_CODE + 0x100;
        for (event, expected) in [
            (
                EVENTINJ_VALID | (EVENT_TYPE_INTERRUPT << EVENTINJ_TYPE_SHIFT) | 0x30,
                EVENTINJ_VALID | (EVENT_TYPE_INTERRUPT << EVENTINJ_TYPE_SHIFT) | 0x30,
            ),
            (
                EVENTINJ_VALID
                    | (EVENT_TYPE_EXCEPTION << EVENTINJ_TYPE_SHIFT)
                    | 0x31
                    | EVENTINJ_ERROR_VALID,
                EVENTINJ_VALID
                    | (EVENT_TYPE_EXCEPTION << EVENTINJ_TYPE_SHIFT)
                    | 0x31
                    | EVENTINJ_ERROR_VALID,
            ),
        ] {
            let mut h = nested_rig(ASM_HLT);
            h.write(handler, &[0x0f, 0xa2]);
            let cs = h.cpu.seg[SEG_CS].sel;
            install_idt_gate(&mut h, 0x30, handler, cs);
            install_idt_gate(&mut h, 0x31, handler, cs);
            put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
            put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
            // the frame is pushed onto the page the nested tables do not map
            h.write(NPT + 0x3000, &0u64.to_le_bytes());
            put(&mut h, SAVE_RSP, 8, NESTED_REMAPPED_GPA + 0x800);
            put(&mut h, VMCB_EVENTINJ, 4, event as u64);
            put(&mut h, VMCB_EVENTINJ_ERROR, 4, 0xe);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF, "{event:#x}");
            assert_eq!(
                get(&h, VMCB_EXIT_INT_INFO, 4),
                expected as u64,
                "{event:#x}"
            );
            if event & EVENTINJ_ERROR_VALID != 0 {
                assert_eq!(get(&h, VMCB_EXIT_INT_INFO_ERROR, 4), 0xe);
            }
        }
    }

    #[test]
    fn a_nested_fault_while_walking_the_guest_page_tables_is_flagged() {
        // the guest's PML4 lives in a page the nested tables do not map: the very first
        // instruction fetch already walks it
        let mut h = nested_rig(&[0x0f, 0xa2]);
        let cr3 = h.cpu.cr3 & !0xfff;
        // cut the nested mapping of the page that holds the guest's top-level table
        let pml4_page = cr3 & !0x1f_ffff;
        assert_ne!(
            pml4_page, NESTED_REMAPPED_GPA,
            "the test needs a different 2 MiB region"
        );
        let index = pml4_page >> 21;
        h.write(NPT + 0x2000 + index * 8, &0u64.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_ne!(get(&h, VMCB_EXITINFO1, 8) & NPF_GUEST_TABLE_WALK, 0);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8) & !0xfff, cr3 & !0xfff);
    }

    #[test]
    fn nested_translation_sets_accessed_and_dirty_bits() {
        let mut h = nested_rig(&[0x48, 0xa3, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2]);
        run(&mut h);
        let entry = u64::from_le_bytes(h.read(NPT + 0x3000, 8).try_into().unwrap());
        assert_ne!(entry & NPT_ACCESSED, 0);
        assert_ne!(entry & NPT_DIRTY, 0, "the guest wrote through it");
        let _ = (SCRATCH, AccessKind::Read);
    }

    #[test]
    fn without_nested_paging_the_translation_is_the_identity() {
        let mut h = rig(ASM_CPUID);
        assert_eq!(
            h.cpu
                .nested_translate(0x1234_5000, AccessKind::Read, false)
                .unwrap(),
            0x1234_5000
        );
    }

    /// The same scenarios on the host JIT, one test each so a failure names its scenario.
    macro_rules! jit_scenarios {
        ($($name:ident),* $(,)?) => {
            mod jit {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        let _serial = crate::jit::serialize_jit_tests();
                        under_jit(super::$name);
                    }
                )*
            }
        };
    }
    jit_scenarios!(
        a_guest_cpuid_exits_with_the_intercept_code_and_the_state_before_the_instruction,
        guest_gprs_are_shared_with_the_host_apart_from_rax_and_rsp,
        instruction_intercept_table,
        cr_and_dr_moves_are_intercepted_per_register,
        port_io_follows_the_permission_map_and_reports_the_access,
        rdmsr_and_wrmsr_follow_the_msr_permission_map,
        intercepted_exceptions_exit_with_their_error_code_and_cr2,
        an_inconsistent_vmcb_is_vmexit_invalid_and_the_host_carries_on,
        event_injection_delivers_through_the_guest_idt_before_the_first_instruction,
        nested_paging_remaps_guest_physical_addresses,
        nested_page_faults_exit_with_the_gpa_and_error_code,
        a_nested_fault_in_the_middle_of_a_block_saves_the_guest_and_restores_the_host,
        a_pending_virtual_interrupt_is_injected_when_the_guest_can_take_it,
    );
}
