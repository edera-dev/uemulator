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
use super::mem::AccessKind;
use super::sys::{CR0_TS, CR4_MCE, EFER_AIBRSE, EFER_SCE, RAX, RCX, RDX};
use super::*;
use crate::snapshot::Snapshot;

#[path = "svm_avic.rs"]
mod avic;
pub(crate) use avic::AVIC_MMIO_WINDOW;
use avic::{Avic, PendingTrap, V_INTR_MASKING};

pub(crate) const MSR_VM_CR: u32 = 0xc001_0114;
pub(crate) const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
/// SVM-Lock key (APM 15.31): write-only, reads as zero.
pub(crate) const MSR_SVM_KEY: u32 = 0xc001_0118;
/// AVIC doorbell (APM 15.29.8): the host APIC id of a vCPU to look at its backing page.
pub(crate) const MSR_AVIC_DOORBELL: u32 = 0xc001_011b;
/// TSC_RATIO (APM 15.30.5): the guest's TSC is the host's times this 8.32 fixed-point ratio,
/// plus the VMCB's TSC offset. Applies only while a guest runs.
pub(crate) const MSR_TSC_RATIO: u32 = 0xc000_0104;
/// 1.0, the reset value.
pub(crate) const TSC_RATIO_ONE: u64 = 1 << 32;
const TSC_RATIO_FRACTION_BITS: u32 = 32;
const TSC_RATIO_RESERVED: u64 = !0 << 40;
/// The branch record MSRs (LastBranchFromIP, LastBranchToIP, LastExceptionFromIP,
/// LastExceptionToIP). Read-only, and nothing records branches, so they only ever change through
/// LBR virtualization: VMRUN loads a guest's from its VMCB and #VMEXIT puts the host's back.
pub(crate) const MSR_LAST_BRANCH_FROM_IP: u32 = 0x1db;
pub(crate) const MSR_LAST_EXCEPTION_TO_IP: u32 = 0x1de;
#[cfg(test)]
const MSR_LAST_EXCEPTION_FROM_IP: u32 = 0x1dd;
const LAST_BRANCH_MSRS: usize = 4;
/// EFER.SVME: enables the SVM instructions.
pub(crate) const EFER_SVME: u64 = 1 << 12;
/// VM_CR bits software may write: DPD, R_INIT, DIS_A20M, LOCK, SVMDIS.
pub(super) const VM_CR_WRITABLE: u64 = 0b1_1111;
/// VM_CR.R_INIT: INIT becomes #SX with error code `SX_INIT_REDIRECTED` (APM 15.29).
pub(super) const VM_CR_R_INIT: u64 = 1 << 1;
/// #SX, and its error code for a redirected INIT.
pub(super) const VECTOR_SECURITY_EXCEPTION: u8 = 30;
pub(super) const SX_INIT_REDIRECTED: u32 = 1;
pub(super) const VM_CR_LOCK: u64 = 1 << 3;
pub(super) const VM_CR_SVMDIS: u64 = 1 << 4;

// VMCB control area offsets.
const VMCB_INTERCEPT_CR: u64 = 0x00;
const VMCB_INTERCEPT_DR: u64 = 0x04;
const VMCB_INTERCEPT_EXCEPTIONS: u64 = 0x08;
const VMCB_INTERCEPT3: u64 = 0x0c;
const VMCB_INTERCEPT4: u64 = 0x10;
/// Two words: the pause filter threshold (cycles) and count (APM 15.14.4).
const VMCB_PAUSE_FILTER_THRESHOLD: u64 = 0x3c;
const VMCB_PAUSE_FILTER_COUNT: u64 = 0x3e;
/// Intercept vector 5: INVLPGB (bit 0), INVLPGB with illegal operands (1), INVPCID (2),
/// MCOMMIT (3), TLBSYNC (4).
const VMCB_INTERCEPT5: u64 = 0x14;
const INTERCEPT_INVLPGB: u32 = 1 << 0;
const INTERCEPT_INVLPGB_ILLEGAL: u32 = 1 << 1;
const INTERCEPT_MCOMMIT: u32 = 1 << 3;
const INTERCEPT_TLBSYNC: u32 = 1 << 4;
const VMEXIT_INVLPGB: u64 = 0xa0;
const VMEXIT_INVLPGB_ILLEGAL: u64 = 0xa1;
const VMEXIT_MCOMMIT: u64 = 0xa3;
const VMEXIT_TLBSYNC: u64 = 0xa4;
const VMCB_IOPM_BASE: u64 = 0x40;
const VMCB_MSRPM_BASE: u64 = 0x48;
const VMCB_TSC_OFFSET: u64 = 0x50;
const VMCB_ASID: u64 = 0x58;
/// One byte: which translations VMRUN flushes before it starts the guest (APM 15.16.1).
const VMCB_TLB_CONTROL: u64 = 0x5c;
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
/// LBR virtualization (bit 0) and virtualized VMSAVE/VMLOAD (bit 1, not supported).
const VMCB_VIRTUAL_EXTENSIONS: u64 = 0xb8;
const LBR_VIRTUALIZATION_ENABLE: u64 = 1 << 0;
/// Virtual VMLOAD/VMSAVE (APM 15.33.1): with nested paging, a guest `vmload`/`vmsave` the VMCB
/// does not intercept takes rAX as a guest-physical address and goes through the nested tables.
const VIRTUAL_VMLOAD_VMSAVE_ENABLE: u64 = 1 << 1;
const VMCB_NEXT_RIP: u64 = 0xc8;
/// Decode assists (APM 15.33.4): how many bytes of the guest's instruction a #PF intercept or a
/// nested page fault fetched (0 to 15), and the bytes.
const VMCB_INSTRUCTION_BYTES_FETCHED: u64 = 0xd0;
const VMCB_INSTRUCTION_BYTES: u64 = 0xd1;
const INSTRUCTION_BYTES_MAX: usize = 15;

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
/// The supervisor CET state VMRUN loads and #VMEXIT saves (`cet.rs`); U_CET and the PLn_SSP
/// MSRs are not in the VMCB, a hypervisor switches them itself (KVM through xsaves).
const SAVE_S_CET: u64 = 0x5e0;
const SAVE_SSP: u64 = 0x5e8;
const SAVE_ISST_ADDR: u64 = 0x5f0;
const SAVE_G_PAT: u64 = 0x668;
/// DEBUGCTL and the four branch record MSRs, in `MSR_LAST_BRANCH_FROM_IP` order: swapped only
/// under LBR virtualization.
const SAVE_DEBUGCTL: u64 = 0x670;
const SAVE_LAST_BRANCHES: [u64; LAST_BRANCH_MSRS] = [0x678, 0x680, 0x688, 0x690];
/// The guest's SPEC_CTRL, which VMRUN loads and #VMEXIT saves (SpecCtrl virtualization: the
/// hypervisor no longer swaps the MSR itself).
const SAVE_SPEC_CTRL: u64 = 0x6e0;

const SEGMENT_SELECTOR: u64 = 0;
const SEGMENT_ATTRIB: u64 = 2;
const SEGMENT_LIMIT: u64 = 4;
const SEGMENT_BASE: u64 = 8;

// Intercept vector 3 (VMCB 0x0c).
const INTERCEPT_INTR: u32 = 1 << 0;
const INTERCEPT_NMI: u32 = 1 << 1;
/// Accepted and never taken: nothing raises an SMI (there is no SMM), so only the tests name it.
#[cfg(test)]
const INTERCEPT_SMI: u32 = 1 << 2;
const INTERCEPT_INIT: u32 = 1 << 3;
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
const INTERCEPT_TASK_SWITCH: u32 = 1 << 29;
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
const INTERCEPT_RDPRU: u32 = 1 << 14;

// #VMEXIT codes.
pub(crate) const VMEXIT_CR_READ: u64 = 0x00;
pub(crate) const VMEXIT_CR_WRITE: u64 = 0x10;
pub(crate) const VMEXIT_DR_READ: u64 = 0x20;
pub(crate) const VMEXIT_DR_WRITE: u64 = 0x30;
pub(crate) const VMEXIT_EXCEPTION: u64 = 0x40;
pub(crate) const VMEXIT_INTR: u64 = 0x60;
const VMEXIT_NMI: u64 = 0x61;
#[cfg(test)]
const VMEXIT_SMI: u64 = 0x62;
const VMEXIT_INIT: u64 = 0x63;
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
const VMEXIT_TASK_SWITCH: u64 = 0x7d;
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
const VMEXIT_RDPRU: u64 = 0x8e;
pub(crate) const VMEXIT_NPF: u64 = 0x400;
pub(crate) const VMEXIT_INVALID: u64 = u64::MAX;

// Task switch exit information (EXITINFO2; EXITINFO1 is the target TSS selector). A switch
// through a task gate during event delivery sets neither reason bit and reports the event in
// EXITINTINFO.
const TASK_SWITCH_BY_IRET: u64 = 1 << 36;
const TASK_SWITCH_BY_JUMP: u64 = 1 << 38;
const TASK_SWITCH_ERROR_CODE_VALID: u64 = 1 << 44;
/// Decode assists: EXITINFO1 of a CR access exit holds the GPR number in bits 3:0, valid only when
/// this bit says it was a MOV (LMSW, CLTS and SMSW leave it clear). DR exits have the number alone.
const CR_ACCESS_GPR_VALID: u64 = 1 << 63;
const ACCESS_GPR_MASK: u64 = 0xf;

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
pub(super) const EVENT_TYPE_NMI: u32 = 2;
pub(super) const EVENT_TYPE_EXCEPTION: u32 = 3;
pub(super) const EVENT_TYPE_SOFTWARE: u32 = 4;

// V_INTR.
const V_IRQ: u64 = 1 << 8;
/// vGIF (APM 15.33.2): with V_GIF_ENABLE, a guest `stgi`/`clgi` the VMCB does not intercept
/// sets/clears V_GIF instead of GIF, and virtual interrupts and NMIs wait for V_GIF.
const V_GIF: u64 = 1 << 9;
const V_GIF_ENABLE: u64 = 1 << 25;
/// vNMI (APM 15.21.10): with V_NMI_ENABLE the hypervisor queues an NMI with V_NMI_PENDING and
/// the processor delivers it when the guest's NMIs are unmasked; V_NMI_BLOCKING is the guest's
/// NMI mask, set by the delivery (or by an NMI injected through EVENTINJ) and cleared by `iret`.
const V_NMI_PENDING: u64 = 1 << 11;
const V_NMI_BLOCKING: u64 = 1 << 12;
const V_NMI_ENABLE: u64 = 1 << 26;
const V_PRIORITY_SHIFT: u32 = 16;
const V_PRIORITY_MASK: u64 = 0xf;
const V_IGNORE_TPR: u64 = 1 << 20;
/// V_TPR holds the guest's CR8 value (bits 3:0; 7:4 are reserved), which V_INTR_PRIO is compared
/// with.
const V_TPR_MASK: u8 = 0xf;
/// CR8 is TPR bits 7:4.
const CR8_TPR_SHIFT: u32 = 4;
const V_VECTOR_SHIFT: u32 = 32;

// TLB_CONTROL. Every value but `NOTHING` empties the guest's TLB here; `FLUSH_ALL` also empties
// the host's.
const TLB_CONTROL_NOTHING: u64 = 0;
const TLB_CONTROL_FLUSH_ALL: u64 = 1;
#[cfg(test)]
const TLB_CONTROL_FLUSH_GUEST: u64 = 3;
#[cfg(test)]
const TLB_CONTROL_FLUSH_GUEST_NON_GLOBAL: u64 = 7;

const INTERRUPT_SHADOW: u64 = 1;
// VMCB 0x90: nested paging and the controls beside it (bits 1, 2 and 5 are SEV's).
const NP_ENABLE: u64 = 1;
/// GMET: under nested paging a guest supervisor access is a supervisor access in the nested
/// tables too, with SMEP's rule for fetches (`nested_walk`).
const GMET_ENABLE: u64 = 1 << 3;
/// ROGPT: a guest page-table access is a nested write only when it sets an Accessed or Dirty
/// bit (`nested_walk`).
const ROGPT_ENABLE: u64 = 1 << 6;
/// INVLPGB and TLBSYNC may run in the guest (else they are #UD there, intercepts aside).
const INVLPGB_ENABLE: u64 = 1 << 7;
/// The VMCB 0x90 controls that change what a nested walk allows, so a guest TLB made under
/// others is not kept (`GuestAddressSpace`).
const NESTED_WALK_CONTROLS: u64 = GMET_ENABLE | ROGPT_ENABLE;
const VECTOR_INTERCEPT_BITMAP_BITS: u32 = 32;
const VECTOR_BREAKPOINT_EXCEPTION: u8 = 3;
const CR0_SELECTIVE_IGNORED: u64 = CR0_TS | 2;
const CR0_CD_BIT: u64 = 1 << 30;
const CR0_NW_BIT: u64 = 1 << 29;
const CR0_PG_BIT: u64 = 1 << 31;
const CR0_PE_BIT: u64 = 1;
/// What `lmsw` loads (it can also set PE, never clear it).
const LMSW_BITS: u64 = CR0_PE_BIT | 0b1110;
/// CR intercept vector bits for CR0: reads in the low half, writes in the high half.
const CR_INTERCEPT_CR0_READ: u32 = 1 << 0;
const CR_INTERCEPT_CR0_WRITE: u32 = 1 << 16;
/// Group 7 (0F 01) ModRM.reg of smsw and lmsw.
const GROUP7_SMSW: u32 = 4;
const GROUP7_LMSW: u32 = 6;
/// EFER bits a VMCB may hold; any other is a consistency failure (VMEXIT_INVALID). AIBRSE is in
/// the set because a KVM guest that sees automatic IBRS turns it on and KVM passes its EFER on;
/// FFXSR, TCE and MCOMMIT likewise, being EFER bits the CPU accepts.
const EFER_KNOWN: u64 = EFER_SCE
    | EFER_LME
    | EFER_LMA
    | EFER_NXE
    | EFER_SVME
    | EFER_AIBRSE
    | super::sys::EFER_FFXSR
    | super::sys::EFER_TCE
    | super::sys::EFER_MCOMMIT;
const DR_HIGH_BITS: u64 = !0u64 << 32;
/// Number of ASIDs reported in CPUID 0x8000000A, Turin's. Nothing is sized by it: any non-zero
/// ASID in a VMCB is accepted, and only the last guest's TLB is kept (`ParkedTlb`).
pub(crate) const SVM_ASIDS: u32 = 32768;
/// The host's ASID; a VMCB may not use it.
const HOST_ASID: u32 = 0;

const SVM_FEATURE_NESTED_PAGING: u32 = 1 << 0;
const SVM_FEATURE_LBR_VIRTUALIZATION: u32 = 1 << 1;
const SVM_FEATURE_SVM_LOCK: u32 = 1 << 2;
const SVM_FEATURE_NEXT_RIP_SAVE: u32 = 1 << 3;
const SVM_FEATURE_TSC_RATIO: u32 = 1 << 4;
const SVM_FEATURE_VMCB_CLEAN_BITS: u32 = 1 << 5;
const SVM_FEATURE_FLUSH_BY_ASID: u32 = 1 << 6;
const SVM_FEATURE_DECODE_ASSISTS: u32 = 1 << 7;
const SVM_FEATURE_PAUSE_FILTER: u32 = 1 << 10;
const SVM_FEATURE_PAUSE_FILTER_THRESHOLD: u32 = 1 << 12;
const SVM_FEATURE_AVIC: u32 = 1 << 13;
const SVM_FEATURE_V_VMSAVE_VMLOAD: u32 = 1 << 15;
const SVM_FEATURE_VGIF: u32 = 1 << 16;
const SVM_FEATURE_GMET: u32 = 1 << 17;
const SVM_FEATURE_X2AVIC: u32 = 1 << 18;
const SVM_FEATURE_SPEC_CTRL: u32 = 1 << 20;
const SVM_FEATURE_ROGPT: u32 = 1 << 21;
/// A machine check in a guest with CR4.MCE clear, while the host's is set, is a #VMEXIT(#MC)
/// instead of a shutdown (`svm_machine_check_exit`).
const SVM_FEATURE_HOST_MCE_OVERRIDE: u32 = 1 << 23;
/// TlbiCtl: the VMCB's INVLPGB enable (and the TLBSYNC intercept); `invlpgb.rs`.
const SVM_FEATURE_TLBI_CONTROL: u32 = 1 << 24;
const SVM_FEATURE_VNMI: u32 = 1 << 25;
/// SVME_ADDR_CHK: a guest's VMRUN/VMLOAD/VMSAVE intercept is checked before rAX is (an address
/// the processor refuses is #GP only when the instruction is not intercepted); without the bit
/// KVM intercepts #GP to catch the old order. `svm_intercept` runs before the instruction does
/// anything, which is the new order.
const SVM_FEATURE_SVME_ADDR_CHK: u32 = 1 << 28;
/// CPUID 0x8000000A EDX. VMCB clean bits: every VMRUN reloads the whole VMCB, which clean bits
/// permit. Flush-by-ASID: any non-zero TLB_CONTROL empties the guest's whole TLB (`ParkedTlb`).
pub(crate) const SVM_FEATURES: u32 = SVM_FEATURE_NESTED_PAGING
    | SVM_FEATURE_LBR_VIRTUALIZATION
    | SVM_FEATURE_SVM_LOCK
    | SVM_FEATURE_NEXT_RIP_SAVE
    | SVM_FEATURE_TSC_RATIO
    | SVM_FEATURE_VMCB_CLEAN_BITS
    | SVM_FEATURE_FLUSH_BY_ASID
    | SVM_FEATURE_DECODE_ASSISTS
    | SVM_FEATURE_PAUSE_FILTER
    | SVM_FEATURE_PAUSE_FILTER_THRESHOLD
    | SVM_FEATURE_AVIC
    | SVM_FEATURE_V_VMSAVE_VMLOAD
    | SVM_FEATURE_VGIF
    | SVM_FEATURE_GMET
    | SVM_FEATURE_X2AVIC
    | SVM_FEATURE_SPEC_CTRL
    | SVM_FEATURE_ROGPT
    | SVM_FEATURE_HOST_MCE_OVERRIDE
    | SVM_FEATURE_TLBI_CONTROL
    | SVM_FEATURE_VNMI
    | SVM_FEATURE_SVME_ADDR_CHK;

/// What `vmrun` saved of the host, and restores at #VMEXIT. The other registers (including all
/// GPRs but RAX/RSP) are deliberately *not* here: the hypervisor swaps those itself.
#[derive(Snapshot)]
#[snapshot(new)]
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
    /// DEBUGCTL and the branch records, put back at #VMEXIT under LBR virtualization.
    debugctl: u64,
    last_branches: [u64; LAST_BRANCH_MSRS],
    /// The host's NMI mask. The guest has its own (`Cpu::nmi_blocked` while it runs): an NMI
    /// injected into it must not mask the host's, nor the host's NMI handler the guest's.
    nmi_blocked: bool,
    spec_ctrl: u64,
    /// S_CET, SSP and ISST_ADDR.
    shadow_stack: [u64; 3],
}

/// The guest being run: the intercepts read from its VMCB at `vmrun` time (the hypervisor cannot
/// change them while the guest runs) and the host state to return to.
#[derive(Snapshot)]
#[snapshot(new)]
pub(crate) struct Guest {
    vmcb: u64,
    pub(super) asid: u32,
    intercept_cr: u32,
    intercept_dr: u32,
    intercept_exceptions: u32,
    intercept3: u32,
    intercept4: u32,
    intercept5: u32,
    iopm: u64,
    msrpm: u64,
    tsc_offset: u64,
    /// Nested page table root, when nested paging is on.
    pub(crate) nested_cr3: Option<u64>,
    /// GMET and ROGPT, from VMCB 0x90 (only with nested paging).
    nested_walk_controls: u64,
    /// INVLPGB and TLBSYNC are allowed in the guest (VMCB 0x90 bit 7).
    pub(super) invlpgb_enabled: bool,
    host: Host,
    /// Virtual interrupt (V_IRQ): vector, priority, and whether the guest's CR8 is ignored.
    virtual_irq: bool,
    virtual_vector: u8,
    virtual_priority: u8,
    ignore_tpr: bool,
    virtual_tpr: u8,
    /// The VMCB's DEBUGCTL and branch records are the guest's own (swapped at VMRUN and #VMEXIT).
    lbr_virtualization: bool,
    pause_filter: PauseFilter,
    /// Guest `vmload`/`vmsave` addresses are guest-physical (needs nested paging).
    virtual_vmload_vmsave: bool,
    /// V_GIF, when vGIF is enabled.
    virtual_gif: Option<bool>,
    /// V_NMI_PENDING, when vNMI is enabled (the mask is `Cpu::nmi_blocked`).
    virtual_nmi_pending: Option<bool>,
    /// V_INTR_MASKING: the guest's CR8 is `virtual_tpr` (with AVIC, the backing page's TPR).
    interrupt_masking: bool,
    avic: Option<Avic>,
    /// An AVIC exit owed after the instruction that caused it.
    avic_trap: Option<PendingTrap>,
}

/// The PAUSE intercept's filter (APM 15.14.4): a guest spinning on `pause` exits only after
/// `count` of them, and with a threshold, only if they come less than `threshold` cycles apart
/// (a pause after a longer gap restarts the count: the guest is not spinning). A count of 0 is no
/// filter. KVM uses this to tell a vCPU waiting for a preempted lock holder from one that merely
/// pauses now and then.
#[derive(Snapshot)]
#[snapshot(new)]
struct PauseFilter {
    count: u16,
    threshold: u16,
    /// Pauses left before the one that exits; loaded from `count` at VMRUN.
    remaining: u16,
    /// Host time (one cycle per nanosecond) of the last pause, or of VMRUN.
    last_pause_ns: u64,
}

/// SVM state of the CPU.
#[derive(Snapshot)]
pub(crate) struct Svm {
    pub(crate) hsave_pa: u64,
    pub(crate) vm_cr: u64,
    /// Global interrupt flag: clear while the host handles a #VMEXIT (`clgi`).
    pub(crate) gif: bool,
    /// SVM-Lock key: `MSR_SVM_KEY` written while unlocked, compared while locked.
    key: u64,
    tsc_ratio: u64,
    /// DEBUGCTL lives in `Cpu::debug`; these are the other LBR MSRs (`MSR_LAST_BRANCH_FROM_IP`
    /// on).
    pub(crate) last_branches: [u64; LAST_BRANCH_MSRS],
    pub(crate) guest: Option<Box<Guest>>,
    /// The VMCB of the guest that exited last, so an exit that happened while delivering an event
    /// can still report that event (`exit_int_info`).
    last_vmcb: u64,
    /// The soft TLB of the side that is not running: the host's while a guest runs, the last
    /// guest's while the host runs. `None` until the first VMRUN.
    #[snapshot(skip)]
    pub(crate) parked_tlb: Option<Box<ParkedTlb>>,
    /// Nested-table leaves the walker resolved for the running guest. `None` until the first
    /// nested walk.
    #[snapshot(skip)]
    nested_leaves: Option<Box<NestedLeafCache>>,
    /// Set while a #VMEXIT fetches the guest's instruction bytes (decode assists): a nested
    /// fault then only ends the fetch, it must not exit (again).
    #[snapshot(skip)]
    fetching_instruction_bytes: bool,
}

impl Svm {
    /// The state after INIT: no guest and GIF set, but the MSRs keep their values, except that
    /// VM_CR.SVMDIS clears unless VM_CR.LOCK is set (APM 15.30.1).
    pub(crate) fn after_init(&self) -> Self {
        let vm_cr = if self.vm_cr & VM_CR_LOCK != 0 {
            self.vm_cr
        } else {
            self.vm_cr & !VM_CR_SVMDIS
        };
        Svm {
            hsave_pa: self.hsave_pa,
            vm_cr,
            key: self.key,
            tsc_ratio: self.tsc_ratio,
            last_branches: self.last_branches,
            ..Svm::new()
        }
    }

    pub(crate) fn tsc_ratio(&self) -> u64 {
        self.tsc_ratio
    }

    pub(crate) fn write_tsc_ratio(&mut self, ratio: u64) -> ExecResult<()> {
        if ratio & TSC_RATIO_RESERVED != 0 {
            return Err(Exception::general_protection(0));
        }
        self.tsc_ratio = ratio;
        Ok(())
    }

    /// SVM_KEY (APM 15.31): unlocked, a write sets the key; locked, writing the key (if one was
    /// set: a lock taken with no key holds until reset) clears VM_CR.LOCK. Anything else is
    /// ignored.
    pub(crate) fn write_key(&mut self, value: u64) {
        if self.vm_cr & VM_CR_LOCK == 0 {
            self.key = value;
        } else if self.key != 0 && value == self.key {
            self.vm_cr &= !VM_CR_LOCK;
        }
    }

    pub(crate) fn new() -> Self {
        Svm {
            hsave_pa: 0,
            vm_cr: 0,
            gif: true,
            key: 0,
            tsc_ratio: TSC_RATIO_ONE,
            last_branches: [0; LAST_BRANCH_MSRS],
            guest: None,
            last_vmcb: 0,
            parked_tlb: None,
            nested_leaves: None,
            fetching_instruction_bytes: false,
        }
    }
}

/// A soft TLB (both CPL halves and the large-page cache) set aside by VMRUN or #VMEXIT.
///
/// With ASIDs, hardware flushes nothing at VMRUN or #VMEXIT (APM 15.16): host translations
/// (ASID 0) survive the guest's run and the guest's survive the host's exit handling, and the
/// hypervisor asks for a flush with TLB_CONTROL or a new ASID when it changes the guest's nested
/// tables. Flushing both ways cost every exit a refill of the host's working set and, worse, the
/// guest a two-dimensional walk (up to 24 table reads) per page it touches again; a PVH dom0's
/// PIT calibration alone is 100k exits. So the two sides swap TLBs instead (a pointer swap, then
/// `tlb_active_base` follows).
///
/// The parked entries are live state, not a snapshot: everything that revokes write tags
/// (`Cpu::revoke_write_tags_for`, `revoke_stale_write_tags`) must reach them too, or a guest
/// store to a page that became code while it was out skips `code_written`.
pub(crate) struct ParkedTlb {
    pub(crate) entries: Vec<TlbEntry>,
    /// `Cpu::tlb_filled` and `Cpu::tlb_global` for `entries`.
    filled: Vec<u64>,
    global: Vec<u64>,
    pub(crate) large_pages: [LargePage; LARGE_PAGE_CACHE_ENTRIES],
    owner: TlbOwner,
}

/// Which guests' translations an invalidation reaches, by ASID.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum GuestTlbs {
    None,
    All,
    Asid(u32),
}

impl GuestTlbs {
    pub(crate) fn covers(self, asid: u32) -> bool {
        match self {
            GuestTlbs::None => false,
            GuestTlbs::All => true,
            GuestTlbs::Asid(target) => target == asid,
        }
    }
}

/// Whose translations a `ParkedTlb` holds.
#[derive(Clone, Copy, PartialEq)]
enum TlbOwner {
    /// Nobody's: the entries must not be used.
    Nobody,
    Host,
    Guest(GuestAddressSpace),
}

/// What a guest's translations were made under. A guest that comes back with anything here
/// different gets an empty TLB. Only the ASID and the nested tables matter architecturally; the
/// paging state is there because the hypervisor may rewrite it in the VMCB (a shadow-paging
/// hypervisor swaps CR3), and a needless flush costs time, never correctness.
#[derive(Clone, Copy, PartialEq)]
struct GuestAddressSpace {
    asid: u32,
    nested_cr3: Option<u64>,
    nested_walk_controls: u64,
    cr0: u64,
    cr3: u64,
    cr4: u64,
    efer: u64,
}

/// Leaves of the nested page tables the walker already resolved, so a guest's page walk (up to
/// five nested walks: one per guest table and one for the page) mostly reads the guest's own
/// tables. A TLB that caches nested translations is what the architecture describes, so it lives
/// exactly as long as the guest's TLB: emptied whenever that is (a VMRUN that does not keep it,
/// INVLPGA), and never consulted for another nested CR3. A hypervisor that changes the nested
/// tables without asking for a flush sees the old translation, as on hardware.
///
/// 4 KiB leaves are direct-mapped by guest-physical page; 2 MiB and 1 GiB leaves (Xen maps a
/// dom0 with them) go round robin in a short list, like `Cpu::large_pages`.
pub(crate) struct NestedLeafCache {
    root: u64,
    small: [NestedLeaf; NESTED_SMALL_LEAVES],
    large: [NestedLeaf; NESTED_LARGE_LEAVES],
    next_large: usize,
}

const NESTED_SMALL_LEAVES: usize = 256;
const NESTED_LARGE_LEAVES: usize = 8;

#[derive(Clone, Copy)]
struct NestedLeaf {
    /// Guest-physical address of the leaf's first byte; `NO_NESTED_LEAF` when unused.
    guest_base: u64,
    host_base: u64,
    /// log2 of the leaf size.
    shift: u32,
    /// Writable at every level and Dirty already.
    store: bool,
    /// Not no-execute at any level (or NX off in the host's EFER).
    executable: bool,
    /// User at every level. Without GMET every access needs it; with GMET only user ones do,
    /// and a supervisor fetch needs it clear.
    user: bool,
}

impl NestedLeaf {
    /// What the leaf allows an access of the given privilege (`supervisor` only under GMET).
    fn rights(&self, supervisor: bool) -> NestedRights {
        NestedRights {
            store: self.store,
            fetch: self.executable && !(supervisor && self.user),
        }
    }
}

/// Never a leaf's aligned guest-physical base.
const NO_NESTED_LEAF: u64 = u64::MAX;
const EMPTY_NESTED_LEAF: NestedLeaf = NestedLeaf {
    guest_base: NO_NESTED_LEAF,
    host_base: 0,
    shift: NPT_PAGE_SHIFT,
    store: true,
    executable: true,
    user: true,
};

impl NestedLeafCache {
    fn empty() -> Box<Self> {
        Box::new(NestedLeafCache {
            root: NO_NESTED_LEAF,
            small: [EMPTY_NESTED_LEAF; NESTED_SMALL_LEAVES],
            large: [EMPTY_NESTED_LEAF; NESTED_LARGE_LEAVES],
            next_large: 0,
        })
    }

    fn small_slot(guest_physical: u64) -> usize {
        (guest_physical >> NPT_PAGE_SHIFT) as usize & (NESTED_SMALL_LEAVES - 1)
    }

    /// The cached leaf covering `guest_physical` under `root`, unless the access needs the walk:
    /// a store the leaf does not allow yet (not writable, or Dirty still to be set), a fetch it
    /// forbids, or a user access to a supervisor leaf. The walk then sets the bit or raises the
    /// nested fault. `supervisor` is a guest supervisor access under GMET.
    fn lookup(
        &self,
        root: u64,
        guest_physical: u64,
        access: super::mem::AccessKind,
        supervisor: bool,
    ) -> Option<(u64, NestedRights)> {
        if self.root != root {
            return None;
        }
        let covers = |leaf: &&NestedLeaf| {
            leaf.guest_base != NO_NESTED_LEAF
                && guest_physical >> leaf.shift == leaf.guest_base >> leaf.shift
        };
        let leaf = Some(&self.small[Self::small_slot(guest_physical)])
            .filter(covers)
            .or_else(|| self.large.iter().find(covers))?;
        let rights = leaf.rights(supervisor);
        let allowed = match access {
            super::mem::AccessKind::Write => rights.store,
            super::mem::AccessKind::Exec => rights.fetch,
            super::mem::AccessKind::Read => true,
        };
        if !allowed || !(supervisor || leaf.user) {
            return None;
        }
        let offset = guest_physical & ((1u64 << leaf.shift) - 1);
        Some((leaf.host_base | offset, rights))
    }

    fn insert(&mut self, root: u64, leaf: NestedLeaf) {
        if self.root != root {
            self.forget();
            self.root = root;
        }
        if leaf.shift == NPT_PAGE_SHIFT {
            self.small[Self::small_slot(leaf.guest_base)] = leaf;
            return;
        }
        // The same leaf again (a store re-walking a clean one) takes its old slot.
        let slot = self
            .large
            .iter()
            .position(|cached| cached.shift == leaf.shift && cached.guest_base == leaf.guest_base)
            .unwrap_or_else(|| {
                let slot = self.next_large;
                self.next_large = (slot + 1) % NESTED_LARGE_LEAVES;
                slot
            });
        self.large[slot] = leaf;
    }

    fn forget(&mut self) {
        self.small = [EMPTY_NESTED_LEAF; NESTED_SMALL_LEAVES];
        self.large = [EMPTY_NESTED_LEAF; NESTED_LARGE_LEAVES];
        self.root = NO_NESTED_LEAF;
    }
}

impl ParkedTlb {
    pub(super) fn entries_and_filled(&mut self) -> (&mut Vec<TlbEntry>, &Vec<u64>) {
        (&mut self.entries, &self.filled)
    }

    fn empty() -> Box<Self> {
        Box::new(ParkedTlb {
            entries: vec![TLB_EMPTY; 2 * TLB_ENTRIES_PER_MODE],
            filled: vec![0; TLB_FILLED_WORDS],
            global: vec![0; TLB_FILLED_WORDS],
            large_pages: [LARGE_PAGE_EMPTY; LARGE_PAGE_CACHE_ENTRIES],
            owner: TlbOwner::Nobody,
        })
    }
}

/// Where `exit_histogram_key` puts the detail (port and direction) above the exit code.
const EXIT_KEY_DETAIL_SHIFT: u32 = 32;

/// The `--debug` exit histogram's key: the exit code, and for I/O exits also the port and the
/// direction, which is what tells a calibration loop from device traffic.
pub(crate) fn exit_histogram_key(code: u64, info1: u64) -> u64 {
    if code == VMEXIT_IOIO {
        let port_and_direction = (info1 >> IOIO_PORT_SHIFT) << 1 | (info1 & IOIO_IN);
        code | port_and_direction << EXIT_KEY_DETAIL_SHIFT
    } else {
        code & u32::MAX as u64
    }
}

pub(crate) fn exit_histogram_label(key: u64) -> String {
    let code = key & u32::MAX as u64;
    let detail = key >> EXIT_KEY_DETAIL_SHIFT;
    match code {
        VMEXIT_IOIO => format!(
            "ioio port={:#x} {}",
            detail >> 1,
            if detail & 1 != 0 { "in" } else { "out" }
        ),
        VMEXIT_NPF => "npf".into(),
        code if (VMEXIT_EXCEPTION..VMEXIT_INTR).contains(&code) => {
            format!("exception {}", code - VMEXIT_EXCEPTION)
        }
        code => format!("code={code:#x}"),
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

    /// VM_CR (APM 15.30.1): while LOCK is set, writes to LOCK and SVMDIS are ignored; SVMDIS
    /// cannot be set under EFER.SVME, locked or not.
    pub(crate) fn write_vm_cr(&mut self, value: u64) -> ExecResult<()> {
        if value & !VM_CR_WRITABLE != 0 || (value & VM_CR_SVMDIS != 0 && self.svm_enabled()) {
            return Err(Exception::general_protection(0));
        }
        let frozen = if self.svm.vm_cr & VM_CR_LOCK != 0 {
            VM_CR_LOCK | VM_CR_SVMDIS
        } else {
            0
        };
        self.svm.vm_cr = (self.svm.vm_cr & frozen) | (value & !frozen);
        Ok(())
    }

    /// DEBUGCTL is `DebugState::debugctl` (`debug.rs`, which acts on BTF and bus-lock detect);
    /// LBR virtualization swaps it with the VMCB's.
    fn debugctl(&self) -> u64 {
        self.debug.debugctl
    }

    fn set_debugctl(&mut self, value: u64) {
        self.debug.debugctl = value;
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
        let address = self.vmcb_address()?;
        let vmcb = if self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.virtual_vmload_vmsave)
        {
            let access = if save {
                AccessKind::Write
            } else {
                AccessKind::Read
            };
            self.nested_translate(address, access, false)?
        } else {
            address
        };
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

    /// 0F 01 DC / DD. In a guest with vGIF they move V_GIF (the VMCB intercepts them otherwise).
    pub(super) fn stgi_clgi(&mut self, set: bool, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        match self.svm.guest.as_deref_mut() {
            Some(guest) if guest.virtual_gif.is_some() => guest.virtual_gif = Some(set),
            _ => self.svm.gif = set,
        }
        self.rip = next;
        Ok(())
    }

    /// 0F 01 DF: invalidates the page at rAX in ASID ECX. ASID 0 is the host's own TLB, which is
    /// the running one; any other drops the parked guest TLB whole (more than the one page, which
    /// is allowed: it is a cache).
    pub(super) fn invlpga(&mut self, next: u64) -> ExecResult<()> {
        self.svm_instruction_checks()?;
        if self.gpr[RCX] as u32 == HOST_ASID {
            self.invlpg(self.gpr[RAX]);
        } else {
            if let Some(parked) = self.svm.parked_tlb.as_deref_mut()
                && matches!(parked.owner, TlbOwner::Guest(_))
            {
                parked.owner = TlbOwner::Nobody;
            }
            self.forget_nested_leaves();
        }
        self.rip = next;
        Ok(())
    }

    /// Makes the parked TLB unusable if it holds the host's translations (`host`) or those of a
    /// guest `guests` covers: the next world switch starts that side empty. With `nested` the
    /// cached nested-table leaves go too (only the hypervisor can have changed those). For
    /// INVLPGB, which may drop more than it was asked to.
    pub(super) fn drop_parked_tlb(&mut self, host: bool, guests: GuestTlbs, nested: bool) {
        if let Some(parked) = self.svm.parked_tlb.as_deref_mut() {
            let matches = match parked.owner {
                TlbOwner::Nobody => false,
                TlbOwner::Host => host,
                TlbOwner::Guest(space) => guests.covers(space.asid),
            };
            if matches {
                parked.owner = TlbOwner::Nobody;
            }
        }
        if nested && guests != GuestTlbs::None {
            self.forget_nested_leaves();
        }
    }

    /// Makes whatever TLB is parked, the host's or a guest's, unusable: for state both sides
    /// share that just changed (PKRU, which VMRUN leaves alone).
    pub(super) fn forget_parked_tlb(&mut self) {
        if let Some(parked) = self.svm.parked_tlb.as_deref_mut() {
            parked.owner = TlbOwner::Nobody;
        }
    }

    /// Empties `NestedLeafCache`, together with the guest TLB it belongs to.
    fn forget_nested_leaves(&mut self) {
        if let Some(cache) = self.svm.nested_leaves.as_deref_mut() {
            cache.forget();
        }
    }

    /// Swaps the running TLB with the parked one; `tlb_active_base` follows the new array.
    fn swap_parked_tlb(&mut self, parked: &mut ParkedTlb) {
        core::mem::swap(&mut self.tlb, &mut parked.entries);
        core::mem::swap(&mut self.tlb_filled, &mut parked.filled);
        core::mem::swap(&mut self.tlb_global, &mut parked.global);
        core::mem::swap(&mut self.large_pages, &mut parked.large_pages);
        self.set_cpl(self.cpl);
    }

    /// VMRUN, with the guest's state loaded: parks the host's TLB and takes the guest's back if
    /// it was made in `space` and TLB_CONTROL asks for no flush, else starts the guest empty.
    fn enter_guest_tlb(&mut self, space: GuestAddressSpace, tlb_control: u64) {
        let mut parked = self.svm.parked_tlb.take().unwrap_or_else(ParkedTlb::empty);
        let keep = tlb_control == TLB_CONTROL_NOTHING && parked.owner == TlbOwner::Guest(space);
        self.swap_parked_tlb(&mut parked);
        if !keep {
            self.flush_tlb();
            self.forget_nested_leaves();
        }
        parked.owner = if tlb_control == TLB_CONTROL_FLUSH_ALL {
            TlbOwner::Nobody
        } else {
            TlbOwner::Host
        };
        self.svm.parked_tlb = Some(parked);
    }

    /// #VMEXIT, with the host's state restored: parks the guest's TLB (made in `space`) for its
    /// next VMRUN and takes the host's back.
    fn leave_guest_tlb(&mut self, space: GuestAddressSpace) {
        let mut parked = self.svm.parked_tlb.take().unwrap_or_else(ParkedTlb::empty);
        let host = parked.owner == TlbOwner::Host;
        self.swap_parked_tlb(&mut parked);
        if !host {
            self.flush_tlb();
        }
        parked.owner = TlbOwner::Guest(space);
        self.svm.parked_tlb = Some(parked);
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
        let guest_s_cet = self.vmcb_read(vmcb, SAVE_S_CET, 8);
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
            || (guest_cr4 & super::cet::CR4_CET != 0 && guest_cr0 & CR0_WP == 0)
            || guest_s_cet & !super::cet::CET_CONTROL_WRITABLE != 0
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
            debugctl: self.debugctl(),
            last_branches: self.svm.last_branches,
            nmi_blocked: self.nmi_blocked,
            spec_ctrl: self.spec_ctrl,
            shadow_stack: [self.cet.s_cet, self.cet.ssp, self.cet.isst_addr],
        };
        let v_intr = self.vmcb_read(vmcb, VMCB_V_INTR, 8);
        let nested_controls = self.vmcb_read(vmcb, VMCB_NP_ENABLE, 8);
        let nested = nested_controls & NP_ENABLE != 0;
        let virtual_extensions = self.vmcb_read(vmcb, VMCB_VIRTUAL_EXTENSIONS, 8);
        let lbr_virtualization = virtual_extensions & LBR_VIRTUALIZATION_ENABLE != 0;
        let pause_filter_count = self.vmcb_read(vmcb, VMCB_PAUSE_FILTER_COUNT, 2) as u16;
        let pause_filter = PauseFilter {
            count: pause_filter_count,
            threshold: self.vmcb_read(vmcb, VMCB_PAUSE_FILTER_THRESHOLD, 2) as u16,
            remaining: pause_filter_count,
            last_pause_ns: self.now_ns(),
        };
        let guest = Guest {
            vmcb,
            asid: asid as u32,
            intercept_cr: self.vmcb_read(vmcb, VMCB_INTERCEPT_CR, 4) as u32,
            intercept_dr: self.vmcb_read(vmcb, VMCB_INTERCEPT_DR, 4) as u32,
            intercept_exceptions: self.vmcb_read(vmcb, VMCB_INTERCEPT_EXCEPTIONS, 4) as u32,
            intercept3,
            intercept4,
            intercept5: self.vmcb_read(vmcb, VMCB_INTERCEPT5, 4) as u32,
            iopm: self.vmcb_read(vmcb, VMCB_IOPM_BASE, 8),
            msrpm: self.vmcb_read(vmcb, VMCB_MSRPM_BASE, 8),
            tsc_offset: self.vmcb_read(vmcb, VMCB_TSC_OFFSET, 8),
            nested_cr3: nested.then(|| self.vmcb_read(vmcb, VMCB_N_CR3, 8)),
            nested_walk_controls: if nested {
                nested_controls & NESTED_WALK_CONTROLS
            } else {
                0
            },
            invlpgb_enabled: nested_controls & INVLPGB_ENABLE != 0,
            host,
            virtual_irq: v_intr & V_IRQ != 0,
            virtual_vector: (v_intr >> V_VECTOR_SHIFT) as u8,
            virtual_priority: ((v_intr >> V_PRIORITY_SHIFT) & V_PRIORITY_MASK) as u8,
            ignore_tpr: v_intr & V_IGNORE_TPR != 0,
            virtual_tpr: v_intr as u8,
            lbr_virtualization,
            pause_filter,
            virtual_vmload_vmsave: nested && virtual_extensions & VIRTUAL_VMLOAD_VMSAVE_ENABLE != 0,
            virtual_gif: (v_intr & V_GIF_ENABLE != 0).then_some(v_intr & V_GIF != 0),
            virtual_nmi_pending: (v_intr & V_NMI_ENABLE != 0)
                .then_some(v_intr & V_NMI_PENDING != 0),
            interrupt_masking: v_intr & V_INTR_MASKING != 0,
            avic: Avic::from_vmcb(self, vmcb, v_intr, nested),
            avic_trap: None,
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
        self.cet.s_cet = guest_s_cet;
        self.cet.ssp = self.vmcb_read(vmcb, SAVE_SSP, 8);
        self.cet.isst_addr = self.vmcb_read(vmcb, SAVE_ISST_ADDR, 8);
        if nested {
            self.pat = self.vmcb_read(vmcb, SAVE_G_PAT, 8);
        }
        if lbr_virtualization {
            let debugctl = self.vmcb_read(vmcb, SAVE_DEBUGCTL, 8);
            self.set_debugctl(debugctl);
            for (k, offset) in SAVE_LAST_BRANCHES.into_iter().enumerate() {
                self.svm.last_branches[k] = self.vmcb_read(vmcb, offset, 8);
            }
        }
        self.spec_ctrl = self.vmcb_read(vmcb, SAVE_SPEC_CTRL, 8);
        // Without vNMI nothing records the guest's NMI mask across exits: the hypervisor tracks
        // it (KVM sets the IRET intercept after injecting an NMI).
        self.nmi_blocked = v_intr & V_NMI_ENABLE != 0 && v_intr & V_NMI_BLOCKING != 0;
        self.interrupt_shadow =
            self.vmcb_read(vmcb, VMCB_INTERRUPT_SHADOW, 8) & INTERRUPT_SHADOW != 0;
        self.halted = false;
        self.svm.gif = true;
        let space = GuestAddressSpace {
            asid: guest.asid,
            nested_cr3: guest.nested_cr3,
            nested_walk_controls: guest.nested_walk_controls,
            cr0: self.cr0,
            cr3: self.cr3,
            cr4: self.cr4,
            efer: self.efer,
        };
        self.svm.guest = Some(Box::new(guest));
        // Counters with HostOnly or GuestOnly set switch sides with the CPU.
        self.pmu_mode_changed();
        let tlb_control = self.vmcb_read(vmcb, VMCB_TLB_CONTROL, 1);
        self.enter_guest_tlb(space, tlb_control);
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
                EVENT_TYPE_NMI => self.deliver_virtual_nmi(),
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
        let instruction_bytes = (code == VMEXIT_NPF
            || code == VMEXIT_EXCEPTION + super::exception::VECTOR_PAGE_FAULT as u64)
            .then(|| self.guest_instruction_bytes());
        let guest = self.svm.guest.take().expect("vmexit outside a guest");
        let vmcb = guest.vmcb;
        self.svm.last_vmcb = vmcb;
        if let Some(histogram) = self.svm_exit_histogram.as_mut() {
            *histogram
                .entry(exit_histogram_key(code, info1))
                .or_default() += 1;
        }
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
        // What the guest's TLB holds now was made under this; the VMCB keeps the same values, so
        // a VMRUN of an unchanged VMCB finds them equal (`enter_guest_tlb`).
        let space = GuestAddressSpace {
            asid: guest.asid,
            nested_cr3: guest.nested_cr3,
            nested_walk_controls: guest.nested_walk_controls,
            cr0,
            cr3,
            cr4,
            efer,
        };
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
        self.vmcb_write(vmcb, SAVE_S_CET, 8, self.cet.s_cet);
        self.vmcb_write(vmcb, SAVE_SSP, 8, self.cet.ssp);
        self.vmcb_write(vmcb, SAVE_ISST_ADDR, 8, self.cet.isst_addr);
        if guest.nested_cr3.is_some() {
            self.vmcb_write(vmcb, SAVE_G_PAT, 8, self.pat);
        }
        self.vmcb_write(vmcb, SAVE_SPEC_CTRL, 8, self.spec_ctrl);
        if guest.lbr_virtualization {
            self.vmcb_write(vmcb, SAVE_DEBUGCTL, 8, self.debugctl());
            for (k, offset) in SAVE_LAST_BRANCHES.into_iter().enumerate() {
                self.vmcb_write(vmcb, offset, 8, self.svm.last_branches[k]);
            }
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
            | ((guest.virtual_vector as u64) << V_VECTOR_SHIFT)
            | ((guest.virtual_gif == Some(true)) as u64 * V_GIF)
            | ((guest.virtual_nmi_pending == Some(true)) as u64 * V_NMI_PENDING)
            | ((guest.virtual_nmi_pending.is_some() && self.nmi_blocked) as u64 * V_NMI_BLOCKING);
        let mut rebuilt = 0xff
            | V_IRQ
            | (V_PRIORITY_MASK << V_PRIORITY_SHIFT)
            | V_IGNORE_TPR
            | (0xff << V_VECTOR_SHIFT);
        if guest.virtual_gif.is_some() {
            rebuilt |= V_GIF;
        }
        if guest.virtual_nmi_pending.is_some() {
            rebuilt |= V_NMI_PENDING | V_NMI_BLOCKING;
        }
        let preserved = self.vmcb_read(vmcb, VMCB_V_INTR, 8) & !rebuilt;
        self.vmcb_write(vmcb, VMCB_V_INTR, 8, preserved | v_intr);
        self.vmcb_write(vmcb, VMCB_EVENTINJ, 4, 0);
        self.vmcb_write(vmcb, VMCB_EXIT_INT_INFO, 4, 0);
        self.vmcb_write(vmcb, VMCB_EXITCODE, 8, code);
        self.vmcb_write(vmcb, VMCB_EXITINFO1, 8, info1);
        self.vmcb_write(vmcb, VMCB_EXITINFO2, 8, info2);
        self.vmcb_write(vmcb, VMCB_NEXT_RIP, 8, next_rip);
        if let Some((bytes, count)) = instruction_bytes {
            self.vmcb_write(vmcb, VMCB_INSTRUCTION_BYTES_FETCHED, 1, count as u64);
            for (k, &byte) in bytes[..count].iter().enumerate() {
                self.vmcb_write(vmcb, VMCB_INSTRUCTION_BYTES + k as u64, 1, byte as u64);
            }
        }
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
        if guest.lbr_virtualization {
            self.set_debugctl(host.debugctl);
            self.svm.last_branches = host.last_branches;
        }
        self.nmi_blocked = host.nmi_blocked;
        self.spec_ctrl = host.spec_ctrl;
        self.eflags = (host.eflags & !(VM | RF)) | RESERVED_ONE;
        self.lazy_op = LazyFlagsOp::Eflags;
        self.rip = host.rip;
        self.gpr[GPR_RSP] = host.rsp;
        self.gpr[RAX] = host.rax;
        [self.cet.s_cet, self.cet.ssp, self.cet.isst_addr] = host.shadow_stack;
        self.interrupt_shadow = false;
        self.halted = false;
        self.svm.gif = false;
        self.leave_guest_tlb(space);
        self.flush_decode();
        Exception::Vmexit
    }

    /// Decode assists: up to 15 bytes of the guest instruction at CS:rIP, fetched as the guest
    /// would (its paging, then the nested tables), for a #PF intercept or a nested page fault, so
    /// the hypervisor can emulate an MMIO access without walking the guest's tables itself. The
    /// fetch stops at the first byte it cannot translate (0 bytes when the fault was the fetch's
    /// own). CR2 is the guest's fault address and must survive the fetch's own page faults.
    fn guest_instruction_bytes(&mut self) -> ([u8; INSTRUCTION_BYTES_MAX], usize) {
        let mut bytes = [0u8; INSTRUCTION_BYTES_MAX];
        let mut count = 0;
        let Ok(start) = self.fetch_linear() else {
            return (bytes, 0);
        };
        let cr2 = self.cr2;
        self.svm.fetching_instruction_bytes = true;
        while count < INSTRUCTION_BYTES_MAX {
            let Ok(physical) =
                self.translate(start.wrapping_add(count as u64), AccessKind::Exec, false)
            else {
                break;
            };
            let in_page = (PAGE_SIZE - (physical & (PAGE_SIZE - 1))) as usize;
            let length = in_page.min(INSTRUCTION_BYTES_MAX - count);
            let Some(source) = self.ram.slice(physical, length) else {
                break;
            };
            bytes[count..count + length].copy_from_slice(source);
            count += length;
        }
        self.svm.fetching_instruction_bytes = false;
        self.cr2 = cr2;
        (bytes, count)
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

    /// The time stamp counter as the running side sees it: `host_tsc` for the host, scaled by
    /// TSC_RATIO and then offset by the VMCB for a guest.
    pub(crate) fn svm_visible_tsc(&self, host_tsc: u64) -> u64 {
        match &self.svm.guest {
            None => host_tsc,
            Some(guest) => {
                let scaled =
                    (host_tsc as u128 * self.svm.tsc_ratio as u128) >> TSC_RATIO_FRACTION_BITS;
                (scaled as u64).wrapping_add(guest.tsc_offset)
            }
        }
    }

    /// The host TSC at which the running side reads `visible_tsc`: `svm_visible_tsc` undone. A
    /// ratio that does not divide evenly rounds down, so a guest's write can land a tick low.
    pub(crate) fn svm_host_tsc_for(&self, visible_tsc: u64) -> u64 {
        match &self.svm.guest {
            None => visible_tsc,
            Some(guest) => {
                let scaled = visible_tsc.wrapping_sub(guest.tsc_offset) as u128;
                // a zero ratio stops the guest's TSC at its offset: no host value moves it
                let ratio = self.svm.tsc_ratio.max(1) as u128;
                ((scaled << TSC_RATIO_FRACTION_BITS) / ratio) as u64
            }
        }
    }

    /// The pause filter's verdict on a guest `pause` under the PAUSE intercept: true when it
    /// exits.
    fn pause_filter_exhausted(&mut self) -> bool {
        let now = self.now_ns();
        let Some(guest) = self.svm.guest.as_deref_mut() else {
            return false;
        };
        let filter = &mut guest.pause_filter;
        if filter.count == 0 {
            return true;
        }
        if filter.threshold != 0
            && now.saturating_sub(filter.last_pause_ns) > filter.threshold as u64
        {
            filter.remaining = filter.count;
        }
        filter.last_pause_ns = now;
        filter.remaining = filter.remaining.saturating_sub(1);
        filter.remaining == 0
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

    /// A machine check while a guest runs: a #VMEXIT(#MC) when the guest intercepts #MC, or,
    /// with HOST_MCE_OVERRIDE, when the guest has CR4.MCE clear and the host has it set (APM
    /// 15.12.x: the host keeps machine checks a guest would otherwise turn into a shutdown).
    /// Returns true when it exited.
    pub(crate) fn svm_machine_check_exit(&mut self) -> bool {
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        let overridden = guest.host.cr4 & CR4_MCE != 0 && self.cr4 & CR4_MCE == 0;
        if self.svm_exception_exit(super::exception::VECTOR_MACHINE_CHECK, None, false) {
            return true;
        }
        if overridden {
            let next = self.rip;
            self.vmexit(
                VMEXIT_EXCEPTION + super::exception::VECTOR_MACHINE_CHECK as u64,
                0,
                0,
                next,
            );
            return true;
        }
        false
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

    /// An INIT for a guest that intercepts it: performs the #VMEXIT and leaves the INIT pending.
    /// Returns true when it did.
    pub(crate) fn svm_init_exit(&mut self) -> bool {
        let intercepted = self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.intercept3 & INTERCEPT_INIT != 0);
        let index = self.cpu_index as usize;
        if !intercepted || !self.dev.lock().lapic_of(index).init_pending() {
            return false;
        }
        let next = self.rip;
        self.vmexit(VMEXIT_INIT, 0, 0, next);
        true
    }

    /// The task switch intercept, at the start of a switch to the TSS `selector`. `return_eip`
    /// is the address after the instruction that switches (meaningless for a gate), and
    /// `error_code` the one an exception delivered through a task gate pushes. A gate's event
    /// goes into EXITINTINFO on the way out (`deliver_chain`), which is how KVM and Xen tell a
    /// gate from a `call`.
    pub(super) fn svm_task_switch_intercept(
        &mut self,
        selector: u16,
        source: super::legacy_flow::TaskSwitchSource,
        return_eip: u64,
        error_code: Option<u32>,
    ) -> ExecResult<()> {
        use super::legacy_flow::TaskSwitchSource;
        if !self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.intercept3 & INTERCEPT_TASK_SWITCH != 0)
        {
            return Ok(());
        }
        let reason = match source {
            TaskSwitchSource::Return => TASK_SWITCH_BY_IRET,
            TaskSwitchSource::Jump => TASK_SWITCH_BY_JUMP,
            TaskSwitchSource::Call | TaskSwitchSource::Gate => 0,
        };
        let error = error_code.map_or(0, |code| TASK_SWITCH_ERROR_CODE_VALID | code as u64);
        let next_rip = if source == TaskSwitchSource::Gate {
            0
        } else {
            return_eip
        };
        Err(self.vmexit(
            VMEXIT_TASK_SWITCH,
            selector as u64,
            reason | error,
            next_rip,
        ))
    }

    /// Housekeeping before each guest instruction: physical NMIs and interrupts (the NMI and INTR
    /// intercepts) and virtual interrupts. Returns true when a #VMEXIT happened.
    ///
    /// An intercepted NMI or interrupt is the host's: it stays pending, the exit clears GIF, and
    /// the host takes it once it sets GIF again (APM 15.17). The NMI comes first, as it would be
    /// delivered first. Without its intercept a physical NMI goes to the guest through the guest's
    /// IDT (`before_step`), which is what hardware does, and which a hypervisor that always sets
    /// the intercept (KVM, Xen) never expects.
    pub(crate) fn svm_before_step(&mut self) -> bool {
        let Some(guest) = self.svm.guest.as_deref_mut() else {
            return false;
        };
        if let Some(trap) = guest.avic_trap.take() {
            let next = self.rip;
            self.vmexit(trap.code, trap.info1, trap.info2, next);
            return true;
        }
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        // The intercept is about the host's NMIs: the guest's own mask does not hold them back.
        if guest.intercept3 & INTERCEPT_NMI != 0 && !guest.host.nmi_blocked && self.nmi_arrived() {
            let next = self.rip;
            self.vmexit(VMEXIT_NMI, 0, 0, next);
            return true;
        }
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        if self.dev.lock().interrupt_pending() && guest.intercept3 & INTERCEPT_INTR != 0 {
            let next = self.rip;
            self.vmexit(VMEXIT_INTR, 0, 0, next);
            return true;
        }
        let virtual_gif = guest.virtual_gif.unwrap_or(true);
        if virtual_gif
            && guest.virtual_nmi_pending == Some(true)
            && !self.nmi_blocked
            && !self.interrupt_shadow
        {
            if let Some(guest) = &mut self.svm.guest {
                guest.virtual_nmi_pending = Some(false);
            }
            self.halted = false;
            // A failure to deliver is reported by the next step like any other fault.
            let _ = self.deliver_virtual_nmi();
            return false;
        }
        if virtual_gif
            && guest.avic.is_some()
            && self.eflags & IF != 0
            && !self.interrupt_shadow
            && let Some(vector) = self.avic_accept_interrupt()
        {
            self.halted = false;
            let _ = self.deliver_hardware_interrupt(vector);
            return false;
        }
        let Some(guest) = &self.svm.guest else {
            return false;
        };
        if virtual_gif
            && guest.virtual_irq
            && self.eflags & IF != 0
            && !self.interrupt_shadow
            && (guest.ignore_tpr || guest.virtual_priority > guest.virtual_tpr & V_TPR_MASK)
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

    /// MOV from CR8 in a guest with V_INTR_MASKING: its virtual TPR (`None` outside one, where
    /// CR8 is the local APIC's TPR).
    pub(crate) fn svm_guest_cr8(&mut self) -> Option<u64> {
        let guest = self
            .svm
            .guest
            .as_ref()
            .filter(|guest| guest.interrupt_masking)?;
        if guest.avic.is_some() {
            return Some(self.avic_tpr() as u64 >> CR8_TPR_SHIFT);
        }
        Some((guest.virtual_tpr & V_TPR_MASK) as u64)
    }

    /// MOV to CR8 in a guest with V_INTR_MASKING; false outside one.
    pub(crate) fn svm_set_guest_cr8(&mut self, cr8: u64) -> bool {
        let Some(guest) = self
            .svm
            .guest
            .as_deref_mut()
            .filter(|guest| guest.interrupt_masking)
        else {
            return false;
        };
        guest.virtual_tpr = cr8 as u8;
        if guest.avic.is_some() {
            self.avic_set_tpr((cr8 as u8) << CR8_TPR_SHIFT);
        }
        true
    }

    /// Delivers an NMI the hypervisor gave the guest (EVENTINJ or vNMI) through the guest's IDT;
    /// it masks the guest's NMIs until `iret` (APM 15.20), unless the delivery itself exited.
    fn deliver_virtual_nmi(&mut self) -> Option<ExitReason> {
        let reason = self.deliver_event(NMI_VECTOR, None, false);
        if self.svm.guest.is_some() {
            self.nmi_blocked = true;
        }
        reason
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
            // decode assists: the vector
            0xcd => (v3 & INTERCEPT_INT != 0).then_some((VMEXIT_INT, i.imm as u8 as u64, 0)),
            0x108 => simple(VMEXIT_INVD, v3 & INTERCEPT_INVD != 0),
            0x109 => simple(VMEXIT_WBINVD, v4 & INTERCEPT_WBINVD != 0),
            // pause is `f3 90`
            0x90 if i.rep == 1 => simple(
                VMEXIT_PAUSE,
                v3 & INTERCEPT_PAUSE != 0 && self.pause_filter_exhausted(),
            ),
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
            // mov from/to control and debug registers; decode assists name the GPR
            0x120 => (cr_bits >> (i.reg & 15) & 1 != 0).then_some((
                VMEXIT_CR_READ + (i.reg & 15) as u64,
                CR_ACCESS_GPR_VALID | (i.rm as u64 & ACCESS_GPR_MASK),
                0,
            )),
            0x122 => {
                let number = (i.reg & 15) as u32;
                let gpr = CR_ACCESS_GPR_VALID | (i.rm as u64 & ACCESS_GPR_MASK);
                if cr_bits >> (16 + number) & 1 != 0 {
                    Some((VMEXIT_CR_WRITE + number as u64, gpr, 0))
                } else if number == 0
                    && v3 & INTERCEPT_CR0_SELECTIVE_WRITE != 0
                    && (self.gpr[i.rm as usize] ^ self.cr0) & !CR0_SELECTIVE_IGNORED != 0
                {
                    Some((VMEXIT_CR0_SELECTIVE_WRITE, gpr, 0))
                } else {
                    None
                }
            }
            0x121 => (dr_bits >> (i.reg & 15) & 1 != 0).then_some((
                VMEXIT_DR_READ + (i.reg & 15) as u64,
                i.rm as u64 & ACCESS_GPR_MASK,
                0,
            )),
            0x123 => (dr_bits >> (16 + (i.reg & 15) as u32) & 1 != 0).then_some((
                VMEXIT_DR_WRITE + (i.reg & 15) as u64,
                i.rm as u64 & ACCESS_GPR_MASK,
                0,
            )),
            // clts writes CR0 (only TS, which the selective intercept ignores)
            0x106 => simple(VMEXIT_CR_WRITE, cr_bits & CR_INTERCEPT_CR0_WRITE != 0),
            // smsw reads CR0, lmsw writes it (MP, EM, TS, and PE only upwards); neither names a GPR
            0x101 if reg == GROUP7_SMSW => {
                simple(VMEXIT_CR_READ, cr_bits & CR_INTERCEPT_CR0_READ != 0)
            }
            0x101 if reg == GROUP7_LMSW => {
                if cr_bits & CR_INTERCEPT_CR0_WRITE != 0 {
                    Some((VMEXIT_CR_WRITE, 0, 0))
                } else if v3 & INTERCEPT_CR0_SELECTIVE_WRITE != 0 {
                    let location = self.modrm_location(i, next)?;
                    let msw = self.load_operand(location, 2, i)?;
                    let loaded =
                        (self.cr0 & !LMSW_BITS) | (msw & LMSW_BITS) | (self.cr0 & CR0_PE_BIT);
                    ((loaded ^ self.cr0) & !CR0_SELECTIVE_IGNORED != 0).then_some((
                        VMEXIT_CR0_SELECTIVE_WRITE,
                        0,
                        0,
                    ))
                } else {
                    None
                }
            }
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
                // decode assists: the linear address
                7 if v3 & INTERCEPT_INVLPG != 0 => match self.modrm_location(i, next)? {
                    super::exec::Loc::Mem(linear) => Some((VMEXIT_INVLPG, linear, 0)),
                    _ => None,
                },
                _ => None,
            },
            0x101 => match i.modrm {
                // monitorx and mwaitx share the MONITOR and MWAIT intercepts (APM vol. 2 15.9)
                0xc8 => simple(VMEXIT_MONITOR, v4 & INTERCEPT_MONITOR != 0),
                0xfa if i.rep == 0 => simple(VMEXIT_MONITOR, v4 & INTERCEPT_MONITOR != 0),
                0xc9 | 0xfb => simple(VMEXIT_MWAIT, v4 & INTERCEPT_MWAIT != 0),
                0xfd => simple(VMEXIT_RDPRU, v4 & INTERCEPT_RDPRU != 0),
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
                super::invlpgb::MODRM_INVLPGB => {
                    let v5 = self.svm.guest.as_ref().map_or(0, |guest| guest.intercept5);
                    if v5 & INTERCEPT_INVLPGB != 0 {
                        Some((VMEXIT_INVLPGB, 0, 0))
                    } else if v5 & INTERCEPT_INVLPGB_ILLEGAL != 0 && self.invlpgb_operands_illegal()
                    {
                        Some((VMEXIT_INVLPGB_ILLEGAL, 0, 0))
                    } else {
                        None
                    }
                }
                super::invlpgb::MODRM_TLBSYNC => {
                    let v5 = self.svm.guest.as_ref().map_or(0, |guest| guest.intercept5);
                    simple(VMEXIT_TLBSYNC, v5 & INTERCEPT_TLBSYNC != 0)
                }
                // F3 0F 01 FA is mcommit; the intercept comes before its EFER.MCOMMIT #UD
                0xfa => {
                    let v5 = self.svm.guest.as_ref().map_or(0, |guest| guest.intercept5);
                    simple(VMEXIT_MCOMMIT, v5 & INTERCEPT_MCOMMIT != 0)
                }
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
/// The nested tables have the host's depth: 4 levels, or 5 when the host ran VMRUN with
/// CR4.LA57 (APM 15.25.5; KVM sizes its NPT by its own LA57). Only the PDPT and PD take PS.
const NPT_LEVELS: u32 = 4;
const NPT_LEVELS_LA57: u32 = 5;
const NPT_PDPT_LEVEL: u32 = 2;
const NPT_INDEX_BITS: u32 = 9;
const NPT_PAGE_SHIFT: u32 = 12;

/// What the nested leaf behind a guest-physical page allows beyond the access that walked it.
/// The soft TLB keeps one entry per page for both levels, so a guest walk folds these into the
/// flags it caches (`Cpu::restrict_to_nested`): a page filled by a read must not take a store the
/// nested tables refuse or one that has to set their Dirty bit, nor a fetch they forbid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct NestedRights {
    /// Writable at every level and Dirty already: a store needs no walk.
    pub(crate) store: bool,
    /// Not no-execute at any level (or NX not enabled in the host's EFER).
    pub(crate) fetch: bool,
}

/// Outside a guest, or without nested paging.
pub(crate) const ALL_NESTED_RIGHTS: NestedRights = NestedRights {
    store: true,
    fetch: true,
};

impl Cpu {
    /// Guest-physical to host-physical through the nested page tables; the identity outside a
    /// guest or without nested paging. A fault is a #VMEXIT(NPF), reported as `Exception::Vmexit`.
    /// For the guest's page-table walks (`guest_table_walk`) and the processor's own accesses
    /// (VMLOAD/VMSAVE), which are supervisor accesses; `nested_translate_with_rights` takes the
    /// privilege of the guest's own.
    pub(crate) fn nested_translate(
        &mut self,
        guest_physical: u64,
        access: super::mem::AccessKind,
        guest_table_walk: bool,
    ) -> ExecResult<u64> {
        self.nested_walk(guest_physical, access, guest_table_walk, false)
            .map(|(address, _)| address)
    }

    /// `nested_translate` of the page a guest access ends at (`user` if the guest made it as a
    /// user access), with what the nested leaf allows. Under AVIC the APIC BAR page, once the
    /// nested tables allow the access, is the backing page's window (`AVIC_MMIO_WINDOW`)
    /// instead.
    pub(crate) fn nested_translate_with_rights(
        &mut self,
        guest_physical: u64,
        access: super::mem::AccessKind,
        user: bool,
    ) -> ExecResult<(u64, NestedRights)> {
        let translated = self.nested_walk(guest_physical, access, false, user)?;
        if self.avic_mmio_page() == Some(guest_physical & !(PAGE_SIZE - 1)) {
            return Ok((
                AVIC_MMIO_WINDOW | (guest_physical & (PAGE_SIZE - 1)),
                translated.1,
            ));
        }
        Ok(translated)
    }

    /// The nested walk proper. Without GMET every access is a user access in the nested tables
    /// (the User bit must be set at every level); with GMET a guest supervisor access (`user`
    /// clear, and every guest page-table access) is a supervisor one, which may use a page
    /// whose leaf is not User but may not fetch from one that is: SMEP in the nested tables, the
    /// way the APM describes GMET ("guest mode execute trap") for hypervisors that keep the
    /// guest kernel from running code out of user pages. My reading: the APM does not spell out
    /// what a supervisor data access to a supervisor nested page does; letting it through is
    /// what makes such pages usable for the kernel code they are meant to hold.
    ///
    /// A guest page-table access is a write in the nested tables, which must be writable and
    /// get Dirty, whether or not the guest entry's Accessed and Dirty bits need setting (as KVM's
    /// walker treats it, `PFERR_WRITE_MASK` in `paging_tmpl.h`). ROGPT relaxes that to a read
    /// unless an A/D update is due: the walkers in `mem.rs` already ask for a write only then.
    fn nested_walk(
        &mut self,
        guest_physical: u64,
        access: super::mem::AccessKind,
        guest_table_walk: bool,
        user: bool,
    ) -> ExecResult<(u64, NestedRights)> {
        let Some((root, controls)) = self
            .svm
            .guest
            .as_ref()
            .and_then(|guest| Some((guest.nested_cr3?, guest.nested_walk_controls)))
        else {
            return Ok((guest_physical, ALL_NESTED_RIGHTS));
        };
        let access = if guest_table_walk && controls & ROGPT_ENABLE == 0 {
            super::mem::AccessKind::Write
        } else {
            access
        };
        let supervisor = controls & GMET_ENABLE != 0 && !user;
        if let Some(hit) = self
            .svm
            .nested_leaves
            .as_deref()
            .and_then(|cache| cache.lookup(root, guest_physical, access, supervisor))
        {
            return Ok(hit);
        }
        let write = access == super::mem::AccessKind::Write;
        let fetch = access == super::mem::AccessKind::Exec;
        let nx_enabled = self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.host.efer & EFER_NXE != 0);
        let levels = if self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| guest.host.cr4 & super::sys::CR4_LA57 != 0)
        {
            NPT_LEVELS_LA57
        } else {
            NPT_LEVELS
        };
        let fault = |cpu: &mut Cpu, present: bool, reserved: bool| {
            if cpu.svm.fetching_instruction_bytes {
                return Err(Exception::general_protection(0));
            }
            let info = (if present { NPF_PRESENT } else { 0 })
                | (if write { NPF_WRITE } else { 0 })
                | (if supervisor { 0 } else { NPF_USER })
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
        let mut user_leaf = true;
        for level in (0..levels).rev() {
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
            let leaf = level == 0 || (entry & NPT_PAGE_SIZE != 0 && level <= NPT_PDPT_LEVEL);
            let alignment_mask = (1u64 << shift) - 1;
            if (!nx_enabled && entry & NPT_NO_EXECUTE != 0)
                || (level > NPT_PDPT_LEVEL && entry & NPT_PAGE_SIZE != 0)
                || (leaf && level > 0 && entry & alignment_mask & NPT_ADDRESS_MASK != 0)
            {
                return fault(self, true, true);
            }
            if entry & NPT_USER == 0 && !supervisor {
                return fault(self, true, false);
            }
            writable &= entry & NPT_WRITABLE != 0;
            no_execute |= entry & NPT_NO_EXECUTE != 0;
            user_leaf &= entry & NPT_USER != 0;
            if leaf {
                if (write && !writable)
                    || (fetch && no_execute && nx_enabled)
                    || (fetch && supervisor && user_leaf)
                {
                    return fault(self, true, false);
                }
                let updated = entry | NPT_ACCESSED | if write { NPT_DIRTY } else { 0 };
                if updated != entry {
                    // A nested hypervisor on another vCPU may be changing the entry (clearing
                    // Dirty for dirty logging, say): update it as hardware does, and walk again
                    // if it changed under us.
                    if !self.update_ram_u64(entry_address, entry, updated) {
                        return self.nested_walk(guest_physical, access, guest_table_walk, user);
                    }
                    entry = updated;
                }
                let base = entry & NPT_ADDRESS_MASK & !alignment_mask;
                let cached = NestedLeaf {
                    guest_base: guest_physical & !alignment_mask,
                    host_base: base,
                    shift,
                    store: writable && entry & NPT_DIRTY != 0,
                    executable: !(no_execute && nx_enabled),
                    user: user_leaf,
                };
                self.svm
                    .nested_leaves
                    .get_or_insert_with(NestedLeafCache::empty)
                    .insert(root, cached);
                return Ok((
                    base | (guest_physical & alignment_mask),
                    cached.rights(supervisor),
                ));
            }
            if entry & NPT_ACCESSED == 0
                && !self.update_ram_u64(entry_address, entry, entry | NPT_ACCESSED)
            {
                return self.nested_walk(guest_physical, access, guest_table_walk, user);
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
    use crate::cpu::debug::MSR_DEBUGCTL;
    use crate::cpu::mem::AccessKind;
    use crate::cpu::sys::{CR0_EM, MSR_EFER};
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
    /// `flush_tlb` empties only the slots filled since the last flush, so which slots those are
    /// has to travel with the TLB when VMRUN and #VMEXIT swap it with the parked one. A record
    /// left behind would have the host's flush clear the guest's slots and leave its own.
    #[test]
    fn a_flush_after_a_world_switch_empties_the_slots_that_side_filled() {
        const HOST_PAGE: u64 = 0x5000;
        const GUEST_PAGE: u64 = 0x7000;
        let mut h = Harness::new(&[0xf4]);
        let space = GuestAddressSpace {
            asid: 1,
            nested_cr3: None,
            nested_walk_controls: 0,
            cr0: 0,
            cr3: 0,
            cr4: 0,
            efer: 0,
        };
        let cached = |h: &Harness, page: u64| h.cpu.tlb.iter().any(|entry| entry.tag_r == page);
        h.cpu.translate(HOST_PAGE, AccessKind::Read, true).unwrap();
        assert!(cached(&h, HOST_PAGE));
        h.cpu.enter_guest_tlb(space, TLB_CONTROL_NOTHING);
        h.cpu.translate(GUEST_PAGE, AccessKind::Read, true).unwrap();
        h.cpu.leave_guest_tlb(space);
        assert!(cached(&h, HOST_PAGE));
        h.cpu.flush_tlb();
        assert!(!cached(&h, HOST_PAGE), "the host's flush left its own page");
        h.cpu.enter_guest_tlb(space, TLB_CONTROL_NOTHING);
        assert!(cached(&h, GUEST_PAGE), "the guest's TLB did not come back");
        h.cpu.flush_tlb();
        assert!(
            !cached(&h, GUEST_PAGE),
            "the guest's flush left its own page"
        );
    }

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
            let result = h.cpu.dispatch_step();
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
    fn vmrun_loads_the_guest_shadow_stack_state_and_vmexit_gives_the_host_its_own_back() {
        // rdsspq rax; cpuid; hlt
        let mut h = rig(&[0xf3, 0x48, 0x0f, 0x1e, 0xc8, 0x0f, 0xa2, 0xf4]);
        let (cr0, cr4) = (get(&h, SAVE_CR0, 8), get(&h, SAVE_CR4, 8));
        put(&mut h, SAVE_CR0, 8, cr0 | CR0_WP);
        put(&mut h, SAVE_CR4, 8, cr4 | super::super::cet::CR4_CET);
        put(&mut h, SAVE_S_CET, 8, super::super::cet::CET_SH_STK_EN);
        put(&mut h, SAVE_SSP, 8, 0x1234_5000);
        put(&mut h, SAVE_ISST_ADDR, 8, 0x5000);
        h.cpu.cet.ssp = 0x9990;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(get(&h, SAVE_RAX, 8), 0x1234_5000, "the guest read its SSP");
        assert_eq!(get(&h, SAVE_SSP, 8), 0x1234_5000);
        assert_eq!(get(&h, SAVE_ISST_ADDR, 8), 0x5000);
        assert_eq!(h.cpu.cet.ssp, 0x9990);
        assert_eq!(h.cpu.cet.s_cet, 0);
        assert_eq!(h.cpu.cet.isst_addr, 0);

        let mut h = rig(&[0x0f, 0xa2, 0xf4]);
        put(&mut h, SAVE_S_CET, 8, 1 << 2);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INVALID, "an IBT bit");
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
            // monitorx and mwaitx use the MONITOR and MWAIT intercepts
            (&[0x0f, 0x01, 0xfa], 0, INTERCEPT_MONITOR, VMEXIT_MONITOR),
            (&[0x0f, 0x01, 0xfb], 0, INTERCEPT_MWAIT, VMEXIT_MWAIT),
            (&[0x0f, 0x01, 0xfd], 0, INTERCEPT_RDPRU, VMEXIT_RDPRU),
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
        // lock mov cr0, rax is a CR8 write (AltMovCr8), intercepted as one
        let mut h = rig(&[0xf0, 0x0f, 0x22, 0xc0]);
        put(&mut h, VMCB_INTERCEPT_CR, 4, 1 << (16 + 8));
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CR_WRITE + 8);
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
        consistency_failure(|h| put(h, SAVE_CR3, 8, 1 << 52));
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

    /// A 32-bit protected-mode guest with its own GDT (flat code at 8 and data at 16, as
    /// `set_guest_mode` loads them, the running task's TSS at `TSS_RUNNING`, an available one at
    /// `TSS_TARGET`) and a 32-bit IDT whose vector 13 is a task gate to `TSS_TARGET`.
    fn task_switch_rig(guest_code: &[u8]) -> Harness {
        use crate::cpu::test_support::descriptor;
        const GDT: u64 = 0x61_0000;
        const IDT32: u64 = 0x62_0000;
        const TSS_TYPE_32_AVAILABLE: u8 = 0x89;
        const TSS_TYPE_32_BUSY: u8 = 0x8b;
        const TASK_GATE: u64 = 0x85 << 40;
        const FLAT_4GIB: u8 = 0xc;
        let mut h = rig(guest_code);
        set_guest_mode(&mut h, 0, CR0_PE_BIT, 0, 0, FLAT_CODE32);
        for (selector, entry) in [
            (8, descriptor(0, 0xfffff, 0x9a, FLAT_4GIB)),
            (16, descriptor(0, 0xfffff, 0x92, FLAT_4GIB)),
            (
                TSS_RUNNING,
                descriptor(TSS_RUNNING_BASE as u32, 0x67, TSS_TYPE_32_BUSY, 0),
            ),
            (
                TSS_TARGET,
                descriptor(TSS_TARGET_BASE as u32, 0x67, TSS_TYPE_32_AVAILABLE, 0),
            ),
        ] {
            h.write(GDT + selector as u64, &entry.to_le_bytes());
        }
        put(&mut h, SAVE_GDTR + SEGMENT_LIMIT, 4, 0x27);
        put(&mut h, SAVE_GDTR + SEGMENT_BASE, 8, GDT);
        let gate = TASK_GATE | ((TSS_TARGET as u64) << 16);
        h.write(IDT32 + 13 * 8, &gate.to_le_bytes());
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0xff);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT32);
        // VMRUN leaves TR alone (VMLOAD loads it), so the guest runs on the live one
        h.cpu.tr = Seg {
            sel: TSS_RUNNING,
            base: TSS_RUNNING_BASE,
            limit: 0x67,
            flags: (TSS_TYPE_32_BUSY as u32) << 8,
        };
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_TASK_SWITCH) as u64,
        );
        h
    }

    const TSS_RUNNING: u16 = 0x18;
    const TSS_TARGET: u16 = 0x20;
    const TSS_RUNNING_BASE: u64 = 0x63_0000;
    const TSS_TARGET_BASE: u64 = 0x63_1000;

    fn assert_no_task_switch(h: &Harness) {
        assert_eq!(h.cpu.tr.sel, TSS_RUNNING, "the switch did not happen");
        assert_eq!(
            h.read(TSS_TARGET_BASE, 4),
            [0; 4],
            "nothing was written to the incoming TSS"
        );
    }

    #[test]
    fn a_task_switch_by_jump_exits_before_switching() {
        let jump_to_tss = [0xea, 0, 0, 0, 0, TSS_TARGET as u8, 0]; // jmp far TSS_TARGET:0
        let mut h = task_switch_rig(&jump_to_tss);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_TASK_SWITCH);
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), TSS_TARGET as u64);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), TASK_SWITCH_BY_JUMP);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        assert_eq!(get(&h, VMCB_NEXT_RIP, 8), GUEST_CODE + 7);
        assert_no_task_switch(&h);
        // without the intercept the guest switches tasks
        let mut h = task_switch_rig(&jump_to_tss);
        put(&mut h, VMCB_INTERCEPT3, 4, INTERCEPT_CPUID as u64);
        let mut tss = [0u8; 0x68];
        tss[32..36].copy_from_slice(&((GUEST_CODE + 0x100) as u32).to_le_bytes());
        tss[36..40].copy_from_slice(&2u32.to_le_bytes());
        for offset in [72, 80, 84] {
            tss[offset..offset + 4].copy_from_slice(&16u32.to_le_bytes());
        }
        tss[76..80].copy_from_slice(&8u32.to_le_bytes());
        h.write(TSS_TARGET_BASE, &tss);
        h.write(GUEST_CODE + 0x100, ASM_CPUID);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 0x100);
        assert_eq!(h.cpu.tr.sel, TSS_TARGET);
    }

    #[test]
    fn a_task_switch_by_iret_exits_with_the_linked_tss() {
        let mut h = task_switch_rig(&[0xcf]); // iret
        put(&mut h, SAVE_RFLAGS, 8, RESERVED_ONE | NT);
        h.write(TSS_RUNNING_BASE, &TSS_TARGET.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_TASK_SWITCH);
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), TSS_TARGET as u64);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), TASK_SWITCH_BY_IRET);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        assert_eq!(get(&h, VMCB_NEXT_RIP, 8), GUEST_CODE + 1);
        assert_no_task_switch(&h);
    }

    #[test]
    fn a_task_switch_through_a_gate_reports_the_event_and_its_error_code() {
        const GENERAL_PROTECTION: u32 = 13;
        let mut h = task_switch_rig(ASM_CPUID);
        let event = EVENTINJ_VALID
            | (EVENT_TYPE_EXCEPTION << EVENTINJ_TYPE_SHIFT)
            | GENERAL_PROTECTION
            | EVENTINJ_ERROR_VALID;
        put(&mut h, VMCB_EVENTINJ, 4, event as u64);
        put(&mut h, VMCB_EVENTINJ_ERROR, 4, 0x1234);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_TASK_SWITCH);
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), TSS_TARGET as u64);
        assert_eq!(
            get(&h, VMCB_EXITINFO2, 8),
            TASK_SWITCH_ERROR_CODE_VALID | 0x1234
        );
        assert_eq!(get(&h, VMCB_EXIT_INT_INFO, 4), event as u64);
        assert_eq!(get(&h, VMCB_EXIT_INT_INFO_ERROR, 4), 0x1234);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE, "the guest ran nothing");
        assert_no_task_switch(&h);
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
        for address in [VMCB + 8, 1 << 52] {
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

    /// Executes the host's `vmrun`, so that what the test raises next arrives while the guest runs.
    fn enter_guest(h: &mut Harness) {
        h.cpu.step().unwrap();
        assert!(h.cpu.svm.guest.is_some(), "vmrun failed");
    }

    /// Raises an NMI at the (only) local APIC, as an NMI IPI or the LINT1 pin would.
    fn raise_nmi(h: &mut Harness) {
        use crate::dev::apic::{DELIVERY_NMI, Interrupt};
        h.cpu.dev.lock().lapic_of(0).accept_addressed(&Interrupt {
            vector: 0,
            delivery_mode: DELIVERY_NMI,
            level_triggered: false,
            destination: 0,
            logical: false,
        });
    }

    #[test]
    fn an_nmi_for_the_host_exits_the_guest_when_intercepted() {
        // KVM always sets the NMI intercept: an NMI that arrives while its guest runs is the
        // host's, and must not run the guest's NMI handler. The handler here would stop on its
        // cpuid with rbx = 0x77.
        let handler = GUEST_CODE + 0x100;
        let mut h = rig(&[0x90, 0x90, 0x90, 0x0f, 0xa2]);
        h.write(handler, &[0xbb, 0x77, 0, 0, 0, 0x0f, 0xa2]);
        let cs = h.cpu.seg[SEG_CS].sel;
        install_idt_gate(&mut h, NMI_VECTOR, handler, cs);
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_NMI) as u64,
        );
        h.cpu.gpr[3] = 0;
        enter_guest(&mut h);
        raise_nmi(&mut h);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NMI);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE,
            "taken before the first instruction"
        );
        assert_eq!(h.cpu.gpr[3], 0, "the guest's NMI handler did not run");
        assert!(
            h.cpu.dev.lock().nmi_pending(),
            "the NMI stays pending for the host, held while GIF is clear"
        );
        assert!(!h.cpu.nmi_blocked);
        // without the intercept the NMI is the guest's
        let mut h = rig(&[0x90, 0x90, 0x90, 0x0f, 0xa2]);
        h.write(handler, &[0xbb, 0x77, 0, 0, 0, 0x0f, 0xa2]);
        install_idt_gate(&mut h, NMI_VECTOR, handler, cs);
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
        enter_guest(&mut h);
        raise_nmi(&mut h);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], 0x77, "the guest took its NMI");
    }

    fn raise_init(h: &mut Harness) {
        use crate::dev::apic::{DELIVERY_INIT, Interrupt};
        h.cpu.dev.lock().lapic_of(0).accept_addressed(&Interrupt {
            vector: 0,
            delivery_mode: DELIVERY_INIT,
            level_triggered: false,
            destination: 0,
            logical: false,
        });
    }

    #[test]
    fn an_init_exits_the_guest_when_intercepted_and_waits_for_gif() {
        const HSAVE: u64 = 0x3e_0000;
        let mut h = rig(&[0x90, 0x90, 0x0f, 0xa2]);
        h.cpu.wrmsr(MSR_VM_HSAVE_PA, HSAVE).unwrap();
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_INIT | INTERCEPT_SMI) as u64,
        );
        enter_guest(&mut h);
        raise_init(&mut h);
        h.cpu.apply_start_signals();
        assert!(h.cpu.svm.guest.is_none());
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INIT);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        assert!(!h.cpu.wait_for_sipi, "the host is not reset");
        // held while GIF is clear: the hypervisor's exit handler runs undisturbed
        h.cpu.apply_start_signals();
        assert!(!h.cpu.wait_for_sipi);
        assert!(h.cpu.dev.lock().lapic_of(0).start_signal_pending());
        // `stgi` lets it in
        h.cpu.svm.gif = true;
        h.cpu.apply_start_signals();
        assert!(h.cpu.wait_for_sipi, "INIT reset the processor");
        assert!(!h.cpu.dev.lock().lapic_of(0).start_signal_pending());
        assert_eq!(
            h.cpu.rdmsr(MSR_VM_HSAVE_PA).unwrap(),
            HSAVE,
            "INIT keeps the MSRs"
        );
        // without the intercept the INIT resets the processor out of the guest
        let mut h = rig(&[0x90, 0x90, 0x0f, 0xa2]);
        enter_guest(&mut h);
        raise_init(&mut h);
        h.cpu.apply_start_signals();
        assert!(h.cpu.svm.guest.is_none());
        assert!(h.cpu.wait_for_sipi);
        assert_ne!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INIT);
        assert_ne!(get(&h, VMCB_EXITCODE, 8), VMEXIT_SMI);
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

    /// rdtsc; mov ebx, eax; mov ecx, edx; cpuid: the guest's TSC ends up in ecx:ebx.
    const GUEST_READS_TSC: &[u8] = &[0x0f, 0x31, 0x89, 0xc3, 0x89, 0xd1, 0x0f, 0xa2];
    const ONE_SECOND_NS: u64 = 1_000_000_000;

    /// What the guest read with `GUEST_READS_TSC`, the host clock starting at one second.
    fn guest_tsc_with_ratio(ratio: u64, offset: u64) -> (u64, Harness) {
        let mut h = rig(GUEST_READS_TSC);
        h.cpu.time_base_ns = ONE_SECOND_NS;
        h.cpu.wrmsr(MSR_TSC_RATIO, ratio).unwrap();
        put(&mut h, VMCB_TSC_OFFSET, 8, offset);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        ((h.cpu.gpr[1] << 32) | (h.cpu.gpr[3] & 0xffff_ffff), h)
    }

    #[test]
    fn the_tsc_ratio_scales_guest_reads_before_the_offset() {
        let mut h = Harness::new(ASM_HLT);
        assert_eq!(h.cpu.rdmsr(MSR_TSC_RATIO).unwrap(), TSC_RATIO_ONE);
        assert!(
            h.cpu.wrmsr(MSR_TSC_RATIO, 1 << 40).is_err(),
            "bits 63:40 are reserved"
        );
        // a guest whose TSC runs twice as fast; the host's own is untouched
        let (doubled, h) = guest_tsc_with_ratio(2 * TSC_RATIO_ONE, 0);
        assert!(
            (2 * ONE_SECOND_NS..2 * ONE_SECOND_NS + 1000).contains(&doubled),
            "{doubled}"
        );
        assert!(h.cpu.tsc() < ONE_SECOND_NS + 1000, "{}", h.cpu.tsc());
        // half speed, then the offset: (TSC * ratio) + offset
        let (halved, _) = guest_tsc_with_ratio(TSC_RATIO_ONE / 2, 7 * ONE_SECOND_NS);
        assert!(
            (ONE_SECOND_NS / 2 + 7 * ONE_SECOND_NS..ONE_SECOND_NS / 2 + 7 * ONE_SECOND_NS + 1000)
                .contains(&halved),
            "{halved}"
        );
    }

    /// A guest's write of the TSC lands on the one real counter, so the guest then reads what it
    /// wrote through its ratio and offset (`Cpu::svm_host_tsc_for`).
    #[test]
    fn a_guest_tsc_write_is_read_back_through_the_ratio_and_offset() {
        const WRITTEN: u64 = 0x10_0000_0000;
        // mov ecx, TSC; xor eax, eax; mov edx, WRITTEN >> 32; wrmsr; rdtsc; mov ebx, eax;
        // mov ecx, edx; cpuid
        const GUEST_WRITES_TSC: &[u8] = &[
            0xb9, 0x10, 0, 0, 0, 0x31, 0xc0, 0xba, 0x10, 0, 0, 0, 0x0f, 0x30, 0x0f, 0x31, 0x89,
            0xc3, 0x89, 0xd1, 0x0f, 0xa2,
        ];
        let mut h = rig(GUEST_WRITES_TSC);
        h.cpu.time_base_ns = ONE_SECOND_NS;
        h.cpu.wrmsr(MSR_TSC_RATIO, 2 * TSC_RATIO_ONE).unwrap();
        put(&mut h, VMCB_TSC_OFFSET, 8, 7 * ONE_SECOND_NS);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        let read = (h.cpu.gpr[1] << 32) | (h.cpu.gpr[3] & 0xffff_ffff);
        assert!((WRITTEN..WRITTEN + 100).contains(&read), "{read:#x}");
        // the host's counter moved to where the guest's ratio and offset put it
        let host = (WRITTEN - 7 * ONE_SECOND_NS) / 2;
        assert!(
            (host..host + 1000).contains(&h.cpu.tsc()),
            "{}",
            h.cpu.tsc()
        );
    }

    /// The guest reads DEBUGCTL into ebx and LastExceptionFromIP into esi, clears DEBUGCTL, and
    /// stops on a cpuid.
    const GUEST_USES_DEBUGCTL: &[u8] = &[
        0xb9, 0xd9, 0x01, 0, 0, // mov ecx, DEBUGCTL
        0x0f, 0x32, // rdmsr
        0x89, 0xc3, // mov ebx, eax
        0xb9, 0xdd, 0x01, 0, 0, // mov ecx, LastExceptionFromIP
        0x0f, 0x32, // rdmsr
        0x89, 0xc6, // mov esi, eax
        0xb9, 0xd9, 0x01, 0, 0, // mov ecx, DEBUGCTL
        0x31, 0xc0, // xor eax, eax
        0x31, 0xd2, // xor edx, edx
        0x0f, 0x30, // wrmsr
        0x0f, 0xa2, // cpuid
    ];
    const HOST_DEBUGCTL: u64 = 1;
    const GUEST_DEBUGCTL: u64 = 2;

    fn lbr_rig(virtualized: bool) -> Harness {
        let mut h = rig(GUEST_USES_DEBUGCTL);
        h.cpu.wrmsr(MSR_DEBUGCTL, HOST_DEBUGCTL).unwrap();
        if virtualized {
            put(
                &mut h,
                VMCB_VIRTUAL_EXTENSIONS,
                8,
                LBR_VIRTUALIZATION_ENABLE,
            );
        }
        put(&mut h, SAVE_DEBUGCTL, 8, GUEST_DEBUGCTL);
        for (k, offset) in SAVE_LAST_BRANCHES.into_iter().enumerate() {
            put(&mut h, offset, 8, 0x1111 * (k as u64 + 1));
        }
        h
    }

    #[test]
    fn lbr_virtualization_gives_the_guest_the_vmcbs_debugctl_and_branch_records() {
        let mut h = lbr_rig(true);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], GUEST_DEBUGCTL, "the guest's DEBUGCTL");
        assert_eq!(h.cpu.gpr[6], 0x3333, "the guest's LastExceptionFromIP");
        assert_eq!(get(&h, SAVE_DEBUGCTL, 8), 0, "the guest's write went back");
        assert_eq!(get(&h, SAVE_LAST_BRANCHES[0], 8), 0x1111);
        assert_eq!(h.cpu.rdmsr(MSR_DEBUGCTL).unwrap(), HOST_DEBUGCTL);
        assert_eq!(h.cpu.rdmsr(MSR_LAST_EXCEPTION_FROM_IP).unwrap(), 0);
        // without it the guest shares the host's, and the VMCB's are left alone
        let mut h = lbr_rig(false);
        run(&mut h);
        assert_eq!(h.cpu.gpr[3], HOST_DEBUGCTL);
        assert_eq!(h.cpu.gpr[6], 0);
        assert_eq!(get(&h, SAVE_DEBUGCTL, 8), GUEST_DEBUGCTL);
        assert_eq!(
            h.cpu.rdmsr(MSR_DEBUGCTL).unwrap(),
            0,
            "the guest cleared it"
        );
    }

    #[test]
    fn the_branch_record_msrs_are_read_only() {
        let mut h = Harness::new(ASM_HLT);
        for msr in MSR_LAST_BRANCH_FROM_IP..=MSR_LAST_EXCEPTION_TO_IP {
            assert_eq!(h.cpu.rdmsr(msr).unwrap(), 0, "{msr:#x}");
            assert!(h.cpu.wrmsr(msr, 1).is_err(), "{msr:#x}");
        }
    }

    #[test]
    fn svm_lock_freezes_svmdis_until_the_key_unlocks_it() {
        let mut h = Harness::new(ASM_HLT);
        h.cpu.efer &= !EFER_SVME;
        h.cpu.wrmsr(MSR_SVM_KEY, 0x55).unwrap();
        assert_eq!(
            h.cpu.rdmsr(MSR_SVM_KEY).unwrap(),
            0,
            "the key reads as zero"
        );
        h.cpu.wrmsr(MSR_VM_CR, VM_CR_LOCK | VM_CR_SVMDIS).unwrap();
        h.cpu.wrmsr(MSR_VM_CR, 0).unwrap();
        assert_eq!(
            h.cpu.rdmsr(MSR_VM_CR).unwrap(),
            VM_CR_LOCK | VM_CR_SVMDIS,
            "writes to LOCK and SVMDIS are ignored while locked"
        );
        assert!(h.cpu.wrmsr(MSR_EFER, h.cpu.efer | EFER_SVME).is_err());
        // INIT keeps a locked SVMDIS
        h.cpu.init();
        assert_eq!(h.cpu.rdmsr(MSR_VM_CR).unwrap(), VM_CR_LOCK | VM_CR_SVMDIS);
        h.cpu.wrmsr(MSR_SVM_KEY, 0x54).unwrap();
        assert_ne!(h.cpu.rdmsr(MSR_VM_CR).unwrap() & VM_CR_LOCK, 0, "wrong key");
        h.cpu.wrmsr(MSR_SVM_KEY, 0x55).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_VM_CR).unwrap(), VM_CR_SVMDIS, "unlocked");
        // unlocked, INIT clears SVMDIS
        h.cpu.init();
        assert_eq!(h.cpu.rdmsr(MSR_VM_CR).unwrap(), 0);
        // locked without a key, only a reset unlocks it
        let mut h = Harness::new(ASM_HLT);
        h.cpu.wrmsr(MSR_VM_CR, VM_CR_LOCK).unwrap();
        h.cpu.wrmsr(MSR_SVM_KEY, 0).unwrap();
        assert_eq!(h.cpu.rdmsr(MSR_VM_CR).unwrap(), VM_CR_LOCK);
        // SVMDIS cannot be set under a running EFER.SVME
        let mut h = Harness::new(ASM_HLT);
        h.cpu.efer |= EFER_SVME;
        assert!(h.cpu.wrmsr(MSR_VM_CR, VM_CR_SVMDIS).is_err());
    }

    const ASM_PAUSE: [u8; 2] = [0xf3, 0x90];
    const ASM_INC_EBX: [u8; 2] = [0xff, 0xc3];

    fn pause_rig(code: &[u8], count: u64, threshold: u64) -> Harness {
        let mut h = rig(code);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_PAUSE) as u64,
        );
        put(&mut h, VMCB_PAUSE_FILTER_COUNT, 2, count);
        put(&mut h, VMCB_PAUSE_FILTER_THRESHOLD, 2, threshold);
        h.cpu.gpr[3] = 0;
        h
    }

    #[test]
    fn the_pause_filter_count_lets_pauses_through_until_it_runs_out() {
        let code = [
            ASM_PAUSE,
            ASM_INC_EBX,
            ASM_PAUSE,
            ASM_INC_EBX,
            ASM_PAUSE,
            ASM_INC_EBX,
        ]
        .concat();
        let mut h = pause_rig(&[&code[..], ASM_CPUID].concat(), 3, 0);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_PAUSE);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 8, "on the third pause");
        assert_eq!(h.cpu.gpr[3], 2);
        // the counter is reloaded by every VMRUN
        let mut h = pause_rig(&[&code[..], ASM_CPUID].concat(), 3, 0);
        run(&mut h);
        put(&mut h, SAVE_RIP, 8, GUEST_CODE);
        h.cpu.svm.gif = true;
        h.cpu.halted = false;
        h.cpu.rip = CODE_BASE;
        h.cpu.gpr[RAX] = VMCB;
        run(&mut h);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 8);
        assert_eq!(h.cpu.gpr[3], 4);
    }

    #[test]
    fn the_pause_filter_threshold_restarts_the_count_after_a_slow_pause() {
        // pause; mov ecx, 1000; loop $; pause; cpuid
        let spaced = [
            &ASM_PAUSE[..],
            &[0xb9, 0xe8, 0x03, 0, 0, 0xe2, 0xfe],
            &ASM_PAUSE,
            ASM_CPUID,
        ]
        .concat();
        let mut h = pause_rig(&spaced, 2, 100);
        run(&mut h);
        assert_eq!(
            get(&h, VMCB_EXITCODE, 8),
            VMEXIT_CPUID,
            "the second pause came too late to count"
        );
        let mut h = pause_rig(&spaced, 2, 0);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_PAUSE, "no threshold");
        let back_to_back = [&ASM_PAUSE[..], &ASM_PAUSE, ASM_CPUID].concat();
        let mut h = pause_rig(&back_to_back, 2, 100);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_PAUSE, "a spin loop");
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 2);
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

    /// A host on 5-level paging (its PML5's first entry is the old PML4) gets 5-level nested
    /// tables: the same identity tables under a PML5. The guest inherits LA57 from the host's
    /// state and walks five levels of its own, each table through the nested ones.
    #[test]
    fn nested_tables_have_five_levels_when_the_host_runs_with_la57() {
        const NESTED_PML5: u64 = NPT + 0x4000;
        const HOST_PML5: u64 = NPT + 0x5000;
        const PRESENT_WRITABLE_USER: u64 = 7;
        let read_then_cpuid = [0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2];
        let (mut h, ()) = rig_prepared(&read_then_cpuid, |h| {
            let pml4 = h.cpu.cr3 & NPT_ADDRESS_MASK;
            h.write(HOST_PML5, &(pml4 | PRESENT_WRITABLE_USER).to_le_bytes());
            h.cpu.cr3 = HOST_PML5;
            h.cpu.cr4 |= super::super::sys::CR4_LA57;
            h.cpu.flush_tlb();
        });
        build_nested_tables(&mut h);
        h.write(NESTED_PML5, &(NPT | PRESENT_WRITABLE_USER).to_le_bytes());
        put(&mut h, VMCB_NP_ENABLE, 8, NP_ENABLE);
        put(&mut h, VMCB_N_CR3, 8, NESTED_PML5);
        put(&mut h, SAVE_G_PAT, 8, 0x0007_0406_0007_0406);
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

    /// mov [0x801000], rax; mov rax, [0x800000]; mov [0x800000], rax; cpuid. The first store
    /// makes the guest's 2 MiB leaf dirty, so the read fills a TLB entry the guest's own tables
    /// would let a store use.
    const DIRTY_LEAF_THEN_READ_THEN_WRITE: &[u8] = &[
        0x48, 0xa3, 0x00, 0x10, 0x80, 0, 0, 0, 0, 0, 0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x48,
        0xa3, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2,
    ];

    /// The soft TLB holds one entry per page for both levels. Filled by a read, it must not let
    /// a store through that the nested tables refuse: hypervisors write-protect guest memory in
    /// the nested tables (Xen's log-dirty mode, read-only p2m types) and rely on the fault.
    #[test]
    fn a_store_after_a_read_still_faults_on_a_read_only_nested_page() {
        let mut h = nested_rig(DIRTY_LEAF_THEN_READ_THEN_WRITE);
        h.write(NPT + 0x3000, &(NESTED_REMAPPED_HPA | 5).to_le_bytes()); // present, user, not writable
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), NESTED_REMAPPED_GPA);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(
            (info & NPF_PRESENT != 0, info & NPF_WRITE != 0),
            (true, true)
        );
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE + 20,
            "the third instruction"
        );
    }

    /// And a fetch from a page the nested tables mark no-execute, after a read cached it.
    #[test]
    fn a_fetch_after_a_read_still_faults_on_a_no_execute_nested_page() {
        // mov rax, [0x800000]; mov rax, 0x800000; jmp rax
        let mut h = nested_rig(&[
            0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x80, 0x00, 0x48, 0xc7, 0xc0, 0x00, 0x00, 0x80,
            0x00, 0xff, 0xe0,
        ]);
        // Nested NX counts only with NX on in the host's EFER.
        h.cpu.efer |= EFER_NXE;
        h.write(
            NPT + 0x3000,
            &(NESTED_REMAPPED_HPA | 7 | NPT_NO_EXECUTE).to_le_bytes(),
        );
        h.write(NESTED_REMAPPED_HPA, ASM_CPUID);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), NESTED_REMAPPED_GPA);
        assert_ne!(get(&h, VMCB_EXITINFO1, 8) & NPF_FETCH, 0);
    }

    /// Likewise a store must reach the nested walk that sets the nested Dirty bit, even after a
    /// read cached the page.
    #[test]
    fn a_store_after_a_read_sets_the_nested_dirty_bit() {
        let mut h = nested_rig(DIRTY_LEAF_THEN_READ_THEN_WRITE);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        let entry = u64::from_le_bytes(h.read(NPT + 0x3000, 8).try_into().unwrap());
        assert_ne!(entry & NPT_DIRTY, 0);
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
            // an NMI is reported as one (type 2), not as exception 2: the hypervisor re-injects
            // what it reads here
            (
                EVENTINJ_VALID | (EVENT_TYPE_NMI << EVENTINJ_TYPE_SHIFT) | NMI_VECTOR as u32,
                EVENTINJ_VALID | (EVENT_TYPE_NMI << EVENTINJ_TYPE_SHIFT) | NMI_VECTOR as u32,
            ),
        ] {
            let mut h = nested_rig(ASM_HLT);
            h.write(handler, &[0x0f, 0xa2]);
            let cs = h.cpu.seg[SEG_CS].sel;
            install_idt_gate(&mut h, NMI_VECTOR, handler, cs);
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

    /// Host-physical page the second run's nested tables send `NESTED_REMAPPED_GPA` to.
    const REMAPPED_AGAIN_HPA: u64 = 0xa0_0000;
    /// mov rax, [0x800000]; cpuid; mov rax, [0x800000]; cpuid
    const READ_REMAPPED_TWICE: &[u8] = &[
        0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2, 0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0,
        0x0f, 0xa2,
    ];
    /// vmrun; mov rcx, [VMCB + NEXT_RIP]; mov [VMCB + SAVE_RIP], rcx; mov [nested PTE], r10
    /// mov dword ptr [r8], r9d; vmrun; hlt: runs the guest to its first cpuid, skips it the way a
    /// hypervisor does, remaps the page the guest read through the nested tables without a
    /// flush, writes r9d to r8 (a VMCB field), and runs the guest again.
    const REMAP_BETWEEN_RUNS: &[u8] = &[
        0x0f, 0x01, 0xd8, 0x48, 0x8b, 0x0c, 0x25, 0xc8, 0x00, 0x3f, 0x00, 0x48, 0x89, 0x0c, 0x25,
        0x78, 0x05, 0x3f, 0x00, 0x4c, 0x89, 0x14, 0x25, 0x00, 0x30, 0x28, 0x00, 0x45, 0x89, 0x08,
        0x0f, 0x01, 0xd8, 0xf4,
    ];
    const BEFORE_REMAP: u64 = 0xfeed_face_cafe_beef;
    const AFTER_REMAP: u64 = 0x0123_4567_89ab_cdef;

    /// What the guest's second read returns after the host remapped the page between its two
    /// runs and stored `value` at VMCB `offset` (`size` bytes).
    fn second_read_after_remap(tlb_control: u64, offset: u64, value: u32) -> u64 {
        let mut h = nested_rig(READ_REMAPPED_TWICE);
        h.write(CODE_BASE, REMAP_BETWEEN_RUNS);
        h.write(NESTED_REMAPPED_HPA, &BEFORE_REMAP.to_le_bytes());
        h.write(REMAPPED_AGAIN_HPA, &AFTER_REMAP.to_le_bytes());
        put(&mut h, VMCB_TLB_CONTROL, 1, tlb_control);
        h.cpu.gpr[8] = VMCB + offset;
        h.cpu.gpr[9] = value as u64;
        h.cpu.gpr[10] = REMAPPED_AGAIN_HPA | 7;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 22, "the second cpuid");
        get(&h, SAVE_RAX, 8)
    }

    /// APM 15.16: with ASIDs the TLB is not flushed by VMRUN or #VMEXIT; the hypervisor asks for
    /// a flush with TLB_CONTROL or moves the guest to a new ASID. A guest that comes back with
    /// its ASID and nothing asked keeps its translations, stale ones included.
    #[test]
    fn a_guest_keeps_its_translations_across_an_exit_unless_a_flush_is_asked_for() {
        assert_eq!(
            second_read_after_remap(TLB_CONTROL_NOTHING, VMCB_ASID, 1),
            BEFORE_REMAP
        );
        for flush in [
            TLB_CONTROL_FLUSH_ALL,
            TLB_CONTROL_FLUSH_GUEST,
            TLB_CONTROL_FLUSH_GUEST_NON_GLOBAL,
        ] {
            assert_eq!(
                second_read_after_remap(flush, VMCB_ASID, 1),
                AFTER_REMAP,
                "TLB_CONTROL {flush}"
            );
        }
        assert_eq!(
            second_read_after_remap(TLB_CONTROL_NOTHING, VMCB_ASID, 2),
            AFTER_REMAP,
            "a new ASID"
        );
        // A guest CR3 the hypervisor changed is taken for another address space (here the same
        // tables with PWT set; more flushing than the architecture asks for, never less).
        const CR3_PWT: u64 = 1 << 3;
        let cr3 = rig(ASM_CPUID).cpu.cr3;
        assert_eq!(
            second_read_after_remap(TLB_CONTROL_NOTHING, SAVE_CR3, (cr3 | CR3_PWT) as u32),
            AFTER_REMAP,
            "another guest CR3"
        );
    }

    #[test]
    fn invlpga_drops_the_guests_translations() {
        // vmrun; mov rcx, [VMCB + NEXT_RIP]; mov [VMCB + SAVE_RIP], rcx; mov [nested PTE], r10
        // mov eax, 0x800000; mov ecx, 1; invlpga; mov eax, 0x3f0000; vmrun; hlt
        let host = [
            0x0f, 0x01, 0xd8, 0x48, 0x8b, 0x0c, 0x25, 0xc8, 0x00, 0x3f, 0x00, 0x48, 0x89, 0x0c,
            0x25, 0x78, 0x05, 0x3f, 0x00, 0x4c, 0x89, 0x14, 0x25, 0x00, 0x30, 0x28, 0x00, 0xb8,
            0x00, 0x00, 0x80, 0x00, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x0f, 0x01, 0xdf, 0xb8, 0x00,
            0x00, 0x3f, 0x00, 0x0f, 0x01, 0xd8, 0xf4,
        ];
        let mut h = nested_rig(READ_REMAPPED_TWICE);
        h.write(CODE_BASE, &host);
        h.write(NESTED_REMAPPED_HPA, &BEFORE_REMAP.to_le_bytes());
        h.write(REMAPPED_AGAIN_HPA, &AFTER_REMAP.to_le_bytes());
        h.cpu.gpr[10] = REMAPPED_AGAIN_HPA | 7;
        run(&mut h);
        assert_eq!(get(&h, SAVE_RAX, 8), AFTER_REMAP);
    }

    /// The guest's translations wait in a second TLB while the host runs. A page the host starts
    /// translating as code then must lose its write tags there too, or the guest's next store to
    /// it skips `code_written` and stale translated code survives.
    #[test]
    fn a_page_flagged_as_code_while_the_guest_is_out_loses_the_guests_write_tags() {
        // mov [0x800000], rax; cpuid; mov [0x800000], rax; cpuid
        let guest = [
            0x48, 0xa3, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2, 0x48, 0xa3, 0, 0, 0x80, 0, 0, 0, 0,
            0, 0x0f, 0xa2,
        ];
        // vmrun; mov rcx, [VMCB + NEXT_RIP]; mov [VMCB + SAVE_RIP], rcx; hlt; vmrun; hlt
        let host = [
            0x0f, 0x01, 0xd8, 0x48, 0x8b, 0x0c, 0x25, 0xc8, 0x00, 0x3f, 0x00, 0x48, 0x89, 0x0c,
            0x25, 0x78, 0x05, 0x3f, 0x00, 0xf4, 0x0f, 0x01, 0xd8, 0xf4,
        ];
        let mut h = nested_rig(&guest);
        h.write(CODE_BASE, &host);
        run(&mut h);
        let page = (NESTED_REMAPPED_HPA >> PAGE_SHIFT) as usize;
        h.cpu.mark_code(NESTED_REMAPPED_HPA);
        assert_ne!(h.cpu.page_has_code.get(page), 0);
        h.cpu.halted = false;
        run(&mut h);
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 22, "the second cpuid");
        assert_eq!(
            h.cpu.page_has_code.get(page),
            0,
            "the guest's store went through code_written"
        );
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

    // ---- decode assists ----

    /// Runs `code` with the CR intercept bits `cr` and DR bits `dr`; returns (exit code, info1).
    fn cr_dr_exit(code: &[u8], cr: u32, dr: u32) -> (u64, u64) {
        let mut h = rig(code);
        put(&mut h, VMCB_INTERCEPT_CR, 4, cr as u64);
        put(&mut h, VMCB_INTERCEPT_DR, 4, dr as u64);
        run(&mut h);
        (get(&h, VMCB_EXITCODE, 8), get(&h, VMCB_EXITINFO1, 8))
    }

    #[test]
    fn decode_assists_name_the_gpr_of_a_cr_or_dr_move() {
        // mov cr3, rbx / mov rax, cr0 / mov cr0, r9
        let cases: &[(&[u8], u32, u64, u64)] = &[
            (&[0x0f, 0x22, 0xdb], 1 << (16 + 3), VMEXIT_CR_WRITE + 3, 3),
            (&[0x0f, 0x20, 0xc0], 1, VMEXIT_CR_READ, 0),
            (&[0x41, 0x0f, 0x22, 0xc1], 1 << 16, VMEXIT_CR_WRITE, 9),
        ];
        for &(code, cr, exit, gpr) in cases {
            assert_eq!(
                cr_dr_exit(code, cr, 0),
                (exit, CR_ACCESS_GPR_VALID | gpr),
                "{code:02x?}"
            );
        }
        // mov dr7, rcx / mov r10, dr6
        assert_eq!(
            cr_dr_exit(&[0x0f, 0x23, 0xf9], 0, 1 << (16 + 7)),
            (VMEXIT_DR_WRITE + 7, 1)
        );
        assert_eq!(
            cr_dr_exit(&[0x41, 0x0f, 0x21, 0xf2], 0, 1 << 6),
            (VMEXIT_DR_READ + 6, 10)
        );
        // the CR0 selective write reports the GPR too: mov cr0, rbx
        let mut h = rig(&[0x0f, 0x22, 0xc3, 0x0f, 0xa2]);
        put(
            &mut h,
            VMCB_INTERCEPT3,
            4,
            (INTERCEPT_CPUID | INTERCEPT_CR0_SELECTIVE_WRITE) as u64,
        );
        h.cpu.gpr[3] = h.cpu.cr0 ^ CR0_CD_BIT;
        run(&mut h);
        assert_eq!(
            (get(&h, VMCB_EXITCODE, 8), get(&h, VMCB_EXITINFO1, 8)),
            (VMEXIT_CR0_SELECTIVE_WRITE, CR_ACCESS_GPR_VALID | 3)
        );
    }

    #[test]
    fn clts_lmsw_and_smsw_are_cr0_accesses_without_a_gpr() {
        const CLTS: &[u8] = &[0x0f, 0x06];
        const LMSW_AX: &[u8] = &[0x0f, 0x01, 0xf0];
        const SMSW_EAX: &[u8] = &[0x0f, 0x01, 0xe0];
        for code in [CLTS, LMSW_AX] {
            assert_eq!(
                cr_dr_exit(code, 1 << 16, 0),
                (VMEXIT_CR_WRITE, 0),
                "{code:02x?}"
            );
        }
        assert_eq!(cr_dr_exit(SMSW_EAX, 1, 0), (VMEXIT_CR_READ, 0));
        // the selective write intercept sees lmsw changing EM, and not clts or lmsw changing TS
        let selective = |code: &[u8], ax: u64| {
            let mut h = rig(&[code, ASM_CPUID].concat());
            put(
                &mut h,
                VMCB_INTERCEPT3,
                4,
                (INTERCEPT_CPUID | INTERCEPT_CR0_SELECTIVE_WRITE) as u64,
            );
            put(&mut h, SAVE_RAX, 8, ax);
            run(&mut h);
            (get(&h, VMCB_EXITCODE, 8), get(&h, VMCB_EXITINFO1, 8))
        };
        let msw = Harness::new(HOST_CODE).cpu.cr0 & 0xf;
        assert_eq!(
            selective(LMSW_AX, msw ^ CR0_EM),
            (VMEXIT_CR0_SELECTIVE_WRITE, 0)
        );
        assert_eq!(selective(LMSW_AX, msw ^ CR0_TS).0, VMEXIT_CPUID);
        assert_eq!(selective(CLTS, 0).0, VMEXIT_CPUID);
    }

    #[test]
    fn decode_assists_give_invlpg_its_linear_address_and_int_its_vector() {
        // invlpg fs:[rbx + 0x10]
        let mut h = rig(&[0x64, 0x0f, 0x01, 0x7b, 0x10]);
        put(&mut h, VMCB_INTERCEPT3, 4, INTERCEPT_INVLPG as u64);
        h.cpu.seg[SEG_FS].base = 0x1_0000;
        h.cpu.gpr[3] = 0x5000;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INVLPG);
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), 0x1_5010);
        let mut h = rig(&[0xcd, 0x80]);
        put(&mut h, VMCB_INTERCEPT3, 4, INTERCEPT_INT as u64);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_INT);
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), 0x80);
    }

    fn instruction_bytes(h: &Harness) -> Vec<u8> {
        let count = get(h, VMCB_INSTRUCTION_BYTES_FETCHED, 1) as usize;
        h.read(VMCB + VMCB_INSTRUCTION_BYTES, count)
    }

    #[test]
    fn a_page_fault_intercept_carries_the_instruction_bytes() {
        // mov rax, [UNMAPPED]; then filler, so a full 15 bytes are there to fetch
        let code = [
            0x48, 0xa1, 0, 0, 0, 0, 2, 0, 0, 0, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
        ];
        let mut h = rig(&code);
        put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 14);
        put(&mut h, VMCB_INSTRUCTION_BYTES_FETCHED, 1, 0xee);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_EXCEPTION + 14);
        assert_eq!(instruction_bytes(&h), code[..INSTRUCTION_BYTES_MAX]);
        assert_eq!(
            get(&h, SAVE_CR2, 8),
            UNMAPPED,
            "fetching the bytes kept CR2"
        );
    }

    #[test]
    fn a_nested_page_fault_carries_the_instruction_bytes_unless_the_fetch_faulted() {
        // the data access faults: mov rax, [0x800000]; cpuid
        let code = [0x48, 0xa1, 0, 0, 0x80, 0, 0, 0, 0, 0, 0x0f, 0xa2];
        let mut h = nested_rig(&code);
        h.write(NPT + 0x3000, &0u64.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        let bytes = instruction_bytes(&h);
        assert_eq!(bytes.len(), INSTRUCTION_BYTES_MAX);
        assert_eq!(bytes[..code.len()], code);
        // the fetch itself faults: mov eax, 0x800000; jmp rax
        let mut h = nested_rig(&[0xb8, 0, 0, 0x80, 0, 0xff, 0xe0]);
        h.write(NPT + 0x3000, &0u64.to_le_bytes());
        put(&mut h, VMCB_INSTRUCTION_BYTES_FETCHED, 1, 0xee);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_ne!(get(&h, VMCB_EXITINFO1, 8) & NPF_FETCH, 0);
        assert_eq!(get(&h, VMCB_INSTRUCTION_BYTES_FETCHED, 1), 0);
    }

    #[test]
    fn an_intercepted_svm_instruction_exits_before_its_address_is_checked() {
        // SVME_ADDR_CHK: vmload with a misaligned rAX exits rather than raising #GP
        let mut h = rig(&[0x0f, 0x01, 0xda]);
        put(
            &mut h,
            VMCB_INTERCEPT4,
            4,
            (INTERCEPT_VMRUN | INTERCEPT_VMLOAD) as u64,
        );
        put(&mut h, SAVE_RAX, 8, 0x1234_5678);
        put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 13);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_VMLOAD);
    }

    // ---- vGIF, vNMI, virtual VMLOAD/VMSAVE, SPEC_CTRL ----

    /// A guest with an IDT whose gates `vectors` all lead to `handler` (code at GUEST_CODE + 0x100).
    fn rig_with_handler(guest_code: &[u8], handler_code: &[u8], vectors: &[u8]) -> Harness {
        let handler = GUEST_CODE + 0x100;
        let mut h = rig(guest_code);
        h.write(handler, handler_code);
        let cs = h.cpu.seg[SEG_CS].sel;
        for &vector in vectors {
            install_idt_gate(&mut h, vector, handler, cs);
        }
        put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0xfff);
        put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
        put(&mut h, SAVE_RFLAGS, 8, 2 | IF);
        h.cpu.gpr[3] = 0;
        h
    }

    /// inc ebx; stgi; jmp $+2; nop; hlt. The jmp ends a JIT block, so the boundary after `stgi`
    /// exists in both modes.
    const INC_THEN_STGI: &[u8] = &[0xff, 0xc3, 0x0f, 0x01, 0xdc, 0xeb, 0x00, 0x90, 0xf4];

    #[test]
    fn vgif_lets_the_guest_toggle_v_gif_without_an_exit() {
        for (code, v_gif) in [
            (
                &[
                    0x0f, 0x01, 0xdd, 0x0f, 0x01, 0xdc, 0x0f, 0x01, 0xdd, 0x0f, 0xa2,
                ][..],
                false,
            ),
            (&[0x0f, 0x01, 0xdd, 0x0f, 0x01, 0xdc, 0x0f, 0xa2][..], true),
        ] {
            let mut h = rig(code);
            put(&mut h, VMCB_V_INTR, 8, V_GIF_ENABLE | V_GIF);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID, "{code:02x?}");
            assert_eq!(get(&h, VMCB_V_INTR, 8) & V_GIF != 0, v_gif, "{code:02x?}");
        }
    }

    #[test]
    fn a_virtual_interrupt_and_a_virtual_nmi_wait_for_v_gif() {
        for (v_intr, vector) in [
            (V_IRQ | V_IGNORE_TPR | (0x40u64 << V_VECTOR_SHIFT), 0x40),
            (V_NMI_ENABLE | V_NMI_PENDING, NMI_VECTOR),
        ] {
            let mut h = rig_with_handler(INC_THEN_STGI, ASM_CPUID, &[vector]);
            put(&mut h, VMCB_V_INTR, 8, V_GIF_ENABLE | v_intr);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID, "{vector:#x}");
            assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 0x100, "{vector:#x}");
            assert_eq!(h.cpu.gpr[3], 1, "{vector:#x}: delivered only after stgi");
            assert!(get(&h, VMCB_V_INTR, 8) & V_GIF != 0);
        }
    }

    /// Pushes an iret frame for the next label and runs `iretq` (unmasking NMIs), then:
    /// nop; jmp $+2; cpuid.
    const IRET_THEN_CPUID: &[u8] = &[
        0x48, 0x89, 0xe0, // mov rax, rsp
        0x8c, 0xd1, // mov ecx, ss
        0x51, // push rcx
        0x50, // push rax
        0x9c, // pushfq
        0x8c, 0xc9, // mov ecx, cs
        0x51, // push rcx
        0x48, 0x8d, 0x0d, 3, 0, 0, 0,    // lea rcx, [rip + 3]
        0x51, // push rcx
        0x48, 0xcf, // iretq
        0x90, 0xeb, 0x00, 0x0f, 0xa2, // nop; jmp $+2; cpuid
    ];
    /// inc ebx; iretq
    const INC_THEN_IRET: &[u8] = &[0xff, 0xc3, 0x48, 0xcf];

    #[test]
    fn a_virtual_nmi_is_delivered_when_the_guest_unmasks_nmis() {
        let mut h = rig_with_handler(IRET_THEN_CPUID, INC_THEN_IRET, &[NMI_VECTOR]);
        put(
            &mut h,
            VMCB_V_INTR,
            8,
            V_NMI_ENABLE | V_NMI_PENDING | V_NMI_BLOCKING,
        );
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], 1, "the NMI ran once, after the first iret");
        let v_intr = get(&h, VMCB_V_INTR, 8);
        assert_eq!(v_intr & (V_NMI_PENDING | V_NMI_BLOCKING), 0, "{v_intr:#x}");
        // taken at once when unmasked; the handler's own state is in V_NMI_BLOCKING at the exit
        let mut h = rig_with_handler(ASM_HLT, &[0xff, 0xc3, 0x0f, 0xa2], &[NMI_VECTOR]);
        put(&mut h, VMCB_V_INTR, 8, V_NMI_ENABLE | V_NMI_PENDING);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], 1);
        let v_intr = get(&h, VMCB_V_INTR, 8);
        assert_eq!(v_intr & (V_NMI_PENDING | V_NMI_BLOCKING), V_NMI_BLOCKING);
        // without V_NMI_ENABLE the bits mean nothing
        let mut h = rig_with_handler(ASM_HLT, ASM_CPUID, &[NMI_VECTOR]);
        put(&mut h, VMCB_V_INTR, 8, V_NMI_PENDING);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_HLT);
    }

    #[test]
    fn an_nmi_injected_through_eventinj_masks_the_guests_nmis() {
        // handler: inc ebx; nop; jmp $+2; nop; cpuid. A physical NMI (not intercepted) waits
        // behind the injected one instead of re-entering the handler.
        let handler = &[0xff, 0xc3, 0x90, 0xeb, 0x00, 0x90, 0x0f, 0xa2];
        let event = (EVENTINJ_VALID | (EVENT_TYPE_NMI << EVENTINJ_TYPE_SHIFT)) as u64;
        let mut h = rig_with_handler(ASM_HLT, handler, &[NMI_VECTOR]);
        put(&mut h, VMCB_EVENTINJ, 4, event);
        enter_guest(&mut h);
        raise_nmi(&mut h);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], 1, "the handler ran once");
        assert!(!h.cpu.nmi_blocked, "the host's NMIs are its own");
        assert!(h.cpu.dev.lock().nmi_pending());
        // under vNMI the mask is reported in V_NMI_BLOCKING
        let mut h = rig_with_handler(ASM_HLT, handler, &[NMI_VECTOR]);
        put(&mut h, VMCB_EVENTINJ, 4, event);
        put(&mut h, VMCB_V_INTR, 8, V_NMI_ENABLE);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_ne!(get(&h, VMCB_V_INTR, 8) & V_NMI_BLOCKING, 0);
    }

    /// mov eax, 0x800000; vmsave or vmload; cpuid, under the identity-plus-one-remap nested
    /// tables, with `virtual_vmload_vmsave` set in the VMCB or not.
    fn virtual_vmsave_rig(save: bool, virtual_vmload_vmsave: bool) -> Harness {
        let instruction = if save { 0xdb } else { 0xda };
        let mut h = nested_rig(&[0xb8, 0, 0, 0x80, 0, 0x0f, 0x01, instruction, 0x0f, 0xa2]);
        if virtual_vmload_vmsave {
            put(
                &mut h,
                VMCB_VIRTUAL_EXTENSIONS,
                8,
                VIRTUAL_VMLOAD_VMSAVE_ENABLE,
            );
        }
        h
    }

    #[test]
    fn virtual_vmload_and_vmsave_take_a_guest_physical_address() {
        const FS_BASE: u64 = 0x1234_0000;
        let fs_base_at = |address: u64| address + SAVE_FS + SEGMENT_BASE;
        let mut h = virtual_vmsave_rig(true, true);
        h.cpu.seg[SEG_FS].base = FS_BASE;
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.read_u64(fs_base_at(NESTED_REMAPPED_HPA)), FS_BASE);
        assert_eq!(h.read_u64(fs_base_at(NESTED_REMAPPED_GPA)), 0);
        // without the VMCB bit the address is the host's
        let mut h = virtual_vmsave_rig(true, false);
        h.cpu.seg[SEG_FS].base = FS_BASE;
        run(&mut h);
        assert_eq!(h.read_u64(fs_base_at(NESTED_REMAPPED_GPA)), FS_BASE);
        // vmload reads through the nested tables too
        let mut h = virtual_vmsave_rig(false, true);
        h.write(fs_base_at(NESTED_REMAPPED_HPA), &FS_BASE.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.seg[SEG_FS].base, FS_BASE);
        // and an unmapped guest-physical page is a nested page fault
        let mut h = virtual_vmsave_rig(true, true);
        h.write(NPT + 0x3000, &0u64.to_le_bytes());
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8), NESTED_REMAPPED_GPA);
        assert_ne!(get(&h, VMCB_EXITINFO1, 8) & NPF_WRITE, 0);
    }

    #[test]
    fn the_guests_spec_ctrl_comes_from_the_vmcb_and_goes_back_to_it() {
        const SSBD: u64 = 1 << 2;
        const IBRS: u64 = 1 << 0;
        // mov ecx, 0x48; rdmsr; mov ebx, eax; mov eax, 5; xor edx, edx; wrmsr; cpuid
        let mut h = rig(&[
            0xb9, 0x48, 0, 0, 0, 0x0f, 0x32, 0x89, 0xc3, 0xb8, 5, 0, 0, 0, 0x31, 0xd2, 0x0f, 0x30,
            0x0f, 0xa2,
        ]);
        h.cpu.spec_ctrl = IBRS;
        put(&mut h, SAVE_SPEC_CTRL, 8, SSBD);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(h.cpu.gpr[3], SSBD, "the guest read its own");
        assert_eq!(get(&h, SAVE_SPEC_CTRL, 8), SSBD | IBRS);
        assert_eq!(h.cpu.spec_ctrl, IBRS, "the host's is back");
    }

    #[test]
    fn with_r_init_an_init_is_a_security_exception_instead_of_a_reset() {
        // the host: #SX through its IDT, error code 1, nothing reset
        let handler = CODE_BASE + 0x800;
        let mut h = rig(ASM_HLT);
        let cs = h.cpu.seg[SEG_CS].sel;
        install_idt_gate(&mut h, VECTOR_SECURITY_EXCEPTION, handler, cs);
        h.cpu.idtr = DescriptorTable {
            base: IDT,
            limit: 0x4ff,
        };
        h.cpu.write_vm_cr(VM_CR_R_INIT).unwrap();
        raise_init(&mut h);
        assert!(h.cpu.apply_start_signals().is_none());
        assert!(!h.cpu.wait_for_sipi, "not reset");
        assert_eq!(h.cpu.rip, handler);
        let rsp = h.cpu.gpr[GPR_RSP];
        assert_eq!(h.read_u64(rsp), SX_INIT_REDIRECTED as u64, "the error code");
        assert!(!h.cpu.dev.lock().lapic_of(0).start_signal_pending());
        // a guest can intercept it like any exception
        let mut h = rig(&[0x90, 0x0f, 0xa2]);
        h.cpu.write_vm_cr(VM_CR_R_INIT).unwrap();
        put(
            &mut h,
            VMCB_INTERCEPT_EXCEPTIONS,
            4,
            1 << VECTOR_SECURITY_EXCEPTION,
        );
        enter_guest(&mut h);
        raise_init(&mut h);
        assert!(h.cpu.apply_start_signals().is_none());
        assert_eq!(
            get(&h, VMCB_EXITCODE, 8),
            VMEXIT_EXCEPTION + VECTOR_SECURITY_EXCEPTION as u64
        );
        assert_eq!(get(&h, VMCB_EXITINFO1, 8), SX_INIT_REDIRECTED as u64);
        assert!(!h.cpu.wait_for_sipi);
    }

    // ---- AVIC, x2AVIC, V_INTR_MASKING ----

    use super::avic::*;

    const BACKING: u64 = 0x3a_0000;
    const PHYSICAL_TABLE: u64 = 0x3b_0000;
    const LOGICAL_TABLE: u64 = 0x3c_0000;
    const TARGET_BACKING: u64 = 0x3d_0000;
    /// Guest-physical (and, through the guest's identity paging, linear) APIC BAR: the second
    /// 4 KiB page of the nested tables' split 2 MiB page, mapped to itself.
    const APIC_BAR: u64 = NESTED_REMAPPED_GPA + 0x1000;
    const TARGET_INDEX: u64 = 3;

    /// `rig_with_handler` with nested paging and AVIC (x2AVIC with `x2apic`) on, IF clear.
    fn avic_rig(code: &[u8], handler: &[u8], vectors: &[u8], x2apic: bool) -> Harness {
        let mut h = rig_with_handler(code, handler, vectors);
        build_nested_tables(&mut h);
        put(&mut h, VMCB_NP_ENABLE, 8, NP_ENABLE);
        put(&mut h, VMCB_N_CR3, 8, NPT);
        put(&mut h, SAVE_G_PAT, 8, 0x0007_0406_0007_0406);
        put(&mut h, SAVE_RFLAGS, 8, 2);
        let x2avic = if x2apic { X2AVIC_ENABLE } else { 0 };
        put(
            &mut h,
            VMCB_V_INTR,
            8,
            AVIC_ENABLE | V_INTR_MASKING | x2avic,
        );
        put(&mut h, VMCB_AVIC_APIC_BAR, 8, APIC_BAR);
        put(&mut h, VMCB_AVIC_BACKING_PAGE, 8, BACKING);
        put(&mut h, VMCB_AVIC_LOGICAL_TABLE, 8, LOGICAL_TABLE);
        put(&mut h, VMCB_AVIC_PHYSICAL_TABLE, 8, PHYSICAL_TABLE | 7);
        h
    }

    fn read_u32(h: &Harness, address: u64) -> u32 {
        u32::from_le_bytes(h.read(address, 4).try_into().unwrap())
    }

    fn backing(h: &Harness, page: u64, offset: u64) -> u32 {
        read_u32(h, page + offset)
    }

    fn set_backing(h: &mut Harness, page: u64, offset: u64, value: u32) {
        h.write(page + offset, &value.to_le_bytes());
    }

    fn irr_has(h: &Harness, page: u64, vector: u8) -> bool {
        backing(h, page, APIC_IRR + (vector as u64 / 32) * 0x10) >> (vector % 32) & 1 != 0
    }

    /// mov dword [APIC_BAR + offset], value
    fn apic_store(offset: u64, value: u32) -> Vec<u8> {
        let address = (APIC_BAR + offset) as u32;
        [
            &[0xc7, 0x04, 0x25][..],
            &address.to_le_bytes(),
            &value.to_le_bytes(),
        ]
        .concat()
    }

    /// mov <reg>, dword [APIC_BAR + offset], with `modrm` 0x04 (eax), 0x0c (ecx)...
    fn apic_load(modrm: u8, offset: u64) -> Vec<u8> {
        let address = (APIC_BAR + offset) as u32;
        [&[0x8b, modrm, 0x25][..], &address.to_le_bytes()].concat()
    }

    fn exit(h: &Harness) -> (u64, u64, u64) {
        (
            get(h, VMCB_EXITCODE, 8),
            get(h, VMCB_EXITINFO1, 8),
            get(h, VMCB_EXITINFO2, 8),
        )
    }

    #[test]
    fn avic_reads_come_from_the_backing_page_and_the_current_count_faults() {
        // mov eax, [ID]; mov ebx, eax; mov ecx, [current count]; cpuid
        let code = [
            apic_load(0x04, APIC_ID),
            vec![0x89, 0xc3],
            apic_load(0x0c, 0x390),
            ASM_CPUID.to_vec(),
        ]
        .concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        set_backing(&mut h, BACKING, APIC_ID, 0x0500_0000);
        run(&mut h);
        assert_eq!(exit(&h), (VMEXIT_AVIC_NOACCEL, 0x390, 0));
        assert_eq!(
            h.cpu.gpr[3], 0x0500_0000,
            "the ID came from the backing page"
        );
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 9, "a fault: on the load");
    }

    #[test]
    fn avic_trap_writes_land_then_exit_and_fault_writes_exit_first() {
        // the SVR: written, then the exit with RIP past the store (not after the nop)
        let code = [apic_store(APIC_SVR, 0x1ff), vec![0x90], ASM_CPUID.to_vec()].concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        run(&mut h);
        assert_eq!(exit(&h), (VMEXIT_AVIC_NOACCEL, APIC_SVR | NOACCEL_WRITE, 0));
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 11);
        assert_eq!(backing(&h, BACKING, APIC_SVR), 0x1ff);
        assert_eq!(
            read_u32(&h, APIC_BAR + APIC_SVR),
            0,
            "not the page the BAR maps"
        );
        // the PPR is read-only: the exit comes first and nothing is written
        let code = [apic_store(APIC_PPR, 0xff), ASM_CPUID.to_vec()].concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        run(&mut h);
        assert_eq!(exit(&h), (VMEXIT_AVIC_NOACCEL, APIC_PPR | NOACCEL_WRITE, 0));
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        assert_eq!(backing(&h, BACKING, APIC_PPR), 0);
    }

    /// inc ebx; EOI; iretq
    fn counting_eoi_handler() -> Vec<u8> {
        [vec![0xff, 0xc3], apic_store(APIC_EOI, 0), vec![0x48, 0xcf]].concat()
    }

    #[test]
    fn avic_delivers_irr_vectors_above_the_ppr_and_an_edge_eoi_retires_them() {
        // sti; nop; jmp $+2; cpuid
        let code = [0xfb, 0x90, 0xeb, 0x00, 0x0f, 0xa2];
        for (tpr, delivered) in [(0x50, 1), (0, 2)] {
            let mut h = avic_rig(&code, &counting_eoi_handler(), &[0x41, 0x62], false);
            set_backing(&mut h, BACKING, APIC_TPR, tpr);
            set_backing(&mut h, BACKING, APIC_IRR + 0x20, 1 << 1); // 0x41
            set_backing(&mut h, BACKING, APIC_IRR + 0x30, 1 << 2); // 0x62
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID, "tpr {tpr:#x}");
            assert_eq!(h.cpu.gpr[3], delivered, "tpr {tpr:#x}");
            assert!(!irr_has(&h, BACKING, 0x62));
            assert_eq!(irr_has(&h, BACKING, 0x41), tpr != 0, "tpr {tpr:#x}");
            for word in 0..8 {
                assert_eq!(
                    backing(&h, BACKING, APIC_ISR + word * 0x10),
                    0,
                    "EOI retired it"
                );
            }
            assert_eq!(backing(&h, BACKING, APIC_PPR), tpr);
        }
    }

    #[test]
    fn a_level_triggered_eoi_traps_and_leaves_the_isr_to_the_hypervisor() {
        let code = [apic_store(APIC_EOI, 0), vec![0x90], ASM_CPUID.to_vec()].concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        set_backing(&mut h, BACKING, APIC_ISR + 0x30, 1 << 2); // 0x62
        set_backing(&mut h, BACKING, APIC_TMR + 0x30, 1 << 2);
        run(&mut h);
        assert_eq!(
            exit(&h),
            (VMEXIT_AVIC_NOACCEL, APIC_EOI | NOACCEL_WRITE, 0x62)
        );
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 11);
        assert_eq!(backing(&h, BACKING, APIC_ISR + 0x30), 1 << 2);
    }

    /// The guest writes ICR high, then ICR low, then runs cpuid. Returns the harness after the
    /// exit.
    fn ipi(high: u32, low: u32, prepare: impl FnOnce(&mut Harness)) -> Harness {
        let code = [
            apic_store(APIC_ICR_HIGH, high),
            apic_store(APIC_ICR_LOW, low),
            ASM_CPUID.to_vec(),
        ]
        .concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        prepare(&mut h);
        run(&mut h);
        h
    }

    fn target_entry(h: &mut Harness, running: bool) {
        let running = if running {
            PHYSICAL_ENTRY_IS_RUNNING
        } else {
            0
        };
        let entry = TARGET_BACKING | PHYSICAL_ENTRY_VALID | running;
        h.write(PHYSICAL_TABLE + TARGET_INDEX * 8, &entry.to_le_bytes());
    }

    #[test]
    fn avic_virtualizes_fixed_ipis_and_exits_for_the_rest() {
        const VECTOR: u32 = 0x46;
        const SELF: u32 = 1 << 18;
        const LOGICAL: u32 = 1 << 11;
        const NMI: u32 = 4 << 8;
        let physical = (TARGET_INDEX as u32) << 24;
        // to itself
        let h = ipi(0, SELF | VECTOR, |_| ());
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert!(irr_has(&h, BACKING, VECTOR as u8));
        // to a running vCPU: no exit
        let h = ipi(physical, VECTOR, |h| target_entry(h, true));
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert!(irr_has(&h, TARGET_BACKING, VECTOR as u8));
        // to one that is not running: delivered, then the hypervisor is told to wake it
        let h = ipi(physical, VECTOR, |h| target_entry(h, false));
        assert_eq!(
            exit(&h),
            (
                VMEXIT_AVIC_INCOMPLETE_IPI,
                (physical as u64) << 32 | VECTOR as u64,
                IPI_FAILURE_TARGET_NOT_RUNNING << 32 | TARGET_INDEX
            )
        );
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 22, "a trap");
        assert!(irr_has(&h, TARGET_BACKING, VECTOR as u8));
        // logical (flat): logical id bit 1 is guest APIC 3
        let h = ipi(0x02 << 24, LOGICAL | VECTOR, |h| {
            target_entry(h, true);
            set_backing(h, BACKING, APIC_DFR, 0xffff_ffff);
            let entry = (LOGICAL_ENTRY_VALID | TARGET_INDEX) as u32;
            h.write(LOGICAL_TABLE + 4, &entry.to_le_bytes());
        });
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert!(irr_has(&h, TARGET_BACKING, VECTOR as u8));
        // nothing is delivered for an invalid target, an NMI or a reserved vector
        for (low, prepare, cause) in [
            (VECTOR, false, IPI_FAILURE_INVALID_TARGET),
            (NMI | VECTOR, true, IPI_FAILURE_INVALID_INT_TYPE),
            (5, true, IPI_FAILURE_INVALID_VECTOR),
        ] {
            let h = ipi(physical, low, |h| {
                if prepare {
                    target_entry(h, true)
                }
            });
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_AVIC_INCOMPLETE_IPI);
            assert_eq!(get(&h, VMCB_EXITINFO2, 8) >> 32, cause, "{low:#x}");
            assert!(!irr_has(&h, TARGET_BACKING, low as u8), "{low:#x}");
        }
    }

    #[test]
    fn x2avic_serves_the_x2apic_msrs_from_the_backing_page() {
        let code = [
            &[0xb9, 0x02, 0x08, 0, 0][..],        // mov ecx, 0x802 (ID)
            &[0x0f, 0x32],                        // rdmsr
            &[0x89, 0xc3],                        // mov ebx, eax
            &[0xb9, 0x30, 0x08, 0, 0],            // mov ecx, 0x830 (ICR)
            &[0xb8, 0x46, 0, 0, 0],               // mov eax, 0x46
            &[0xba, TARGET_INDEX as u8, 0, 0, 0], // mov edx, 3
            &[0x0f, 0x30],                        // wrmsr
            &[0xb9, 0x39, 0x08, 0, 0],            // mov ecx, 0x839 (current count)
            &[0x0f, 0x32],                        // rdmsr
            ASM_CPUID,
        ]
        .concat();
        let mut h = avic_rig(&code, ASM_CPUID, &[], true);
        set_backing(&mut h, BACKING, APIC_ID, 7);
        target_entry(&mut h, true);
        run(&mut h);
        assert_eq!(h.cpu.gpr[3], 7);
        assert!(irr_has(&h, TARGET_BACKING, 0x46));
        assert_eq!(backing(&h, BACKING, APIC_ICR_HIGH), TARGET_INDEX as u32);
        assert_eq!(exit(&h), (VMEXIT_AVIC_NOACCEL, 0x390, 0));
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE + 31);
    }

    #[test]
    fn with_v_intr_masking_the_guests_cr8_is_its_virtual_tpr() {
        // mov eax, 7; mov cr8, rax; mov rbx, cr8; cpuid
        let code = [
            0xb8, 7, 0, 0, 0, 0x44, 0x0f, 0x22, 0xc0, 0x44, 0x0f, 0x20, 0xc3, 0x0f, 0xa2,
        ];
        let mut h = rig(&code);
        put(&mut h, VMCB_V_INTR, 8, V_INTR_MASKING);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(get(&h, VMCB_V_INTR, 8) & 0xff, 7);
        assert_eq!(h.cpu.gpr[3], 7);
        assert_eq!(
            h.cpu.dev.lock().lapic().tpr(),
            0,
            "the host's TPR is untouched"
        );
        // under AVIC it is the backing page's TPR
        let mut h = avic_rig(&code, ASM_CPUID, &[], false);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        assert_eq!(backing(&h, BACKING, APIC_TPR), 0x70);
        assert_eq!(h.cpu.gpr[3], 7);
    }

    #[test]
    fn a_virtual_interrupt_waits_for_a_priority_above_v_tpr() {
        for (priority, exit_code) in [(3u64, VMEXIT_HLT), (6, VMEXIT_CPUID)] {
            let mut h = rig_with_handler(ASM_HLT, ASM_CPUID, &[0x40]);
            put(
                &mut h,
                VMCB_V_INTR,
                8,
                5 | V_IRQ | (priority << V_PRIORITY_SHIFT) | (0x40u64 << V_VECTOR_SHIFT),
            );
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), exit_code, "priority {priority}");
        }
    }

    #[test]
    fn the_avic_doorbell_is_write_only() {
        let mut h = rig(ASM_HLT);
        assert!(h.cpu.wrmsr(MSR_AVIC_DOORBELL, 1).is_ok());
        assert!(h.cpu.rdmsr(MSR_AVIC_DOORBELL).is_err());
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
        a_store_after_a_read_still_faults_on_a_read_only_nested_page,
        a_store_after_a_read_sets_the_nested_dirty_bit,
        a_fetch_after_a_read_still_faults_on_a_no_execute_nested_page,
        a_pending_virtual_interrupt_is_injected_when_the_guest_can_take_it,
        a_guest_keeps_its_translations_across_an_exit_unless_a_flush_is_asked_for,
        invlpga_drops_the_guests_translations,
        a_page_flagged_as_code_while_the_guest_is_out_loses_the_guests_write_tags,
        an_nmi_for_the_host_exits_the_guest_when_intercepted,
        the_tsc_ratio_scales_guest_reads_before_the_offset,
        a_guest_tsc_write_is_read_back_through_the_ratio_and_offset,
        lbr_virtualization_gives_the_guest_the_vmcbs_debugctl_and_branch_records,
        the_pause_filter_count_lets_pauses_through_until_it_runs_out,
        the_pause_filter_threshold_restarts_the_count_after_a_slow_pause,
        decode_assists_name_the_gpr_of_a_cr_or_dr_move,
        clts_lmsw_and_smsw_are_cr0_accesses_without_a_gpr,
        decode_assists_give_invlpg_its_linear_address_and_int_its_vector,
        a_page_fault_intercept_carries_the_instruction_bytes,
        a_nested_page_fault_carries_the_instruction_bytes_unless_the_fetch_faulted,
        vgif_lets_the_guest_toggle_v_gif_without_an_exit,
        a_virtual_interrupt_and_a_virtual_nmi_wait_for_v_gif,
        a_virtual_nmi_is_delivered_when_the_guest_unmasks_nmis,
        an_nmi_injected_through_eventinj_masks_the_guests_nmis,
        virtual_vmload_and_vmsave_take_a_guest_physical_address,
        the_guests_spec_ctrl_comes_from_the_vmcb_and_goes_back_to_it,
        avic_reads_come_from_the_backing_page_and_the_current_count_faults,
        avic_trap_writes_land_then_exit_and_fault_writes_exit_first,
        avic_delivers_irr_vectors_above_the_ppr_and_an_edge_eoi_retires_them,
        a_level_triggered_eoi_traps_and_leaves_the_isr_to_the_hypervisor,
        avic_virtualizes_fixed_ipis_and_exits_for_the_rest,
        x2avic_serves_the_x2apic_msrs_from_the_backing_page,
        with_v_intr_masking_the_guests_cr8_is_its_virtual_tpr,
    );
    #[test]
    fn mcommit_exits_when_intercepted_and_is_ud_without_efer_mcommit_otherwise() {
        // mcommit; cpuid
        const MCOMMIT: &[u8] = &[0xf3, 0x0f, 0x01, 0xfa, 0x0f, 0xa2];
        for (intercept5, guest_efer_mcommit, expected) in [
            (INTERCEPT_MCOMMIT, false, VMEXIT_MCOMMIT),
            (0, false, VMEXIT_EXCEPTION + 6),
            (0, true, VMEXIT_CPUID),
        ] {
            let mut h = rig(MCOMMIT);
            put(&mut h, VMCB_INTERCEPT5, 4, intercept5 as u64);
            put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 6);
            if guest_efer_mcommit {
                let efer = get(&h, SAVE_EFER, 8);
                put(&mut h, SAVE_EFER, 8, efer | crate::cpu::sys::EFER_MCOMMIT);
            }
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), expected, "{intercept5:#x}");
        }
    }

    #[test]
    fn invlpgb_and_tlbsync_exit_when_intercepted_and_are_ud_in_a_guest_otherwise() {
        // invlpgb / tlbsync, then cpuid
        const INVLPGB: &[u8] = &[0x0f, 0x01, 0xfe, 0x0f, 0xa2];
        const TLBSYNC: &[u8] = &[0x0f, 0x01, 0xff, 0x0f, 0xa2];
        const TOO_MANY_PAGES: u64 = 8;
        for (code, intercept5, rcx, expected) in [
            (INVLPGB, INTERCEPT_INVLPGB, 0, VMEXIT_INVLPGB),
            (
                INVLPGB,
                INTERCEPT_INVLPGB_ILLEGAL,
                TOO_MANY_PAGES,
                VMEXIT_INVLPGB_ILLEGAL,
            ),
            (INVLPGB, INTERCEPT_INVLPGB_ILLEGAL, 0, VMEXIT_EXCEPTION + 6),
            (TLBSYNC, INTERCEPT_TLBSYNC, 0, VMEXIT_TLBSYNC),
            (TLBSYNC, 0, 0, VMEXIT_EXCEPTION + 6),
        ] {
            let mut h = rig(code);
            put(&mut h, VMCB_INTERCEPT5, 4, intercept5 as u64);
            put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 6);
            h.cpu.gpr[RCX] = rcx;
            run(&mut h);
            assert_eq!(
                get(&h, VMCB_EXITCODE, 8),
                expected,
                "{code:02x?} {intercept5:#x}"
            );
            assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
        }
    }

    #[test]
    fn a_guest_data_breakpoint_follows_the_vmcb_dr7_and_the_host_gets_its_own_back() {
        // mov dword [rbx], eax; cpuid
        let mut h = rig(&[0x89, 0x03, 0x0f, 0xa2]);
        const WATCHED: u64 = crate::testkit::SCRATCH;
        // L0, R/W 01 (write), LEN 11 (four bytes)
        const GUEST_DR7: u64 = 0x400 | 1 | (1 | 3 << 2) << 16;
        h.cpu.dr[0] = WATCHED;
        h.cpu.gpr[RBX] = WATCHED;
        put(&mut h, SAVE_DR7, 8, GUEST_DR7);
        put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 1);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_EXCEPTION + 1);
        assert_eq!(
            get(&h, SAVE_RIP, 8),
            GUEST_CODE + 2,
            "a trap: after the store"
        );
        assert_eq!(get(&h, SAVE_DR6, 8), 0xffff_0ff1, "B0 in the guest's DR6");
        assert_eq!(get(&h, SAVE_DR7, 8), GUEST_DR7);
        assert_eq!(h.cpu.dr[7], 0x400, "the host's DR7 is back");
        assert_eq!(h.cpu.dr[6], 0xffff_0ff0);
    }

    // ---- GMET, ROGPT, HOST_MCE_OVERRIDE, INVLPGB in a guest ----

    /// The nested PD entry of the 2 MiB region holding `guest_physical`, an identity large leaf
    /// with `flags` (present 1, writable 2, user 4).
    fn set_nested_region(h: &mut Harness, guest_physical: u64, flags: u64) {
        let region = guest_physical & !0x1f_ffff;
        assert_ne!(
            region, NESTED_REMAPPED_GPA,
            "that region is split into 4 KiB pages"
        );
        let entry = region | flags | NPT_PAGE_SIZE;
        h.write(NPT + 0x2000 + (region >> 21) * 8, &entry.to_le_bytes());
    }

    const NPT_PRESENT_WRITABLE: u64 = 3;
    const NPT_PRESENT_USER: u64 = 5;

    #[test]
    fn gmet_makes_a_supervisor_fetch_from_a_user_nested_page_a_nested_fault() {
        let mut h = nested_rig(ASM_CPUID);
        put(&mut h, VMCB_NP_ENABLE, 8, NP_ENABLE | GMET_ENABLE);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        assert_eq!(get(&h, VMCB_EXITINFO2, 8) & !0xfff, GUEST_CODE & !0xfff);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(
            info & (NPF_PRESENT | NPF_WRITE | NPF_USER | NPF_FETCH | NPF_FINAL_ADDRESS),
            NPF_PRESENT | NPF_FETCH | NPF_FINAL_ADDRESS,
            "a supervisor fetch, refused: {info:#x}"
        );
        assert_eq!(get(&h, SAVE_RIP, 8), GUEST_CODE);
    }

    /// Under GMET the guest kernel runs (and reads, and walks its tables) from nested pages that
    /// are not User; without GMET those pages are closed to it like to anything else.
    #[test]
    fn under_gmet_supervisor_code_runs_from_a_supervisor_nested_page() {
        for (controls, expected) in [
            (NP_ENABLE | GMET_ENABLE, VMEXIT_CPUID),
            (NP_ENABLE, VMEXIT_NPF),
        ] {
            let mut h = nested_rig(ASM_CPUID);
            set_nested_region(&mut h, GUEST_CODE, NPT_PRESENT_WRITABLE);
            put(&mut h, VMCB_NP_ENABLE, 8, controls);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), expected, "{controls:#x}");
        }
    }

    /// A read caches the User page under GMET; the jump into it must still be refused, on either
    /// engine (the soft TLB keeps one entry per page for reads and fetches).
    #[test]
    fn under_gmet_a_fetch_after_a_read_of_a_user_page_still_faults() {
        fn check() {
            // mov rax, [0x800000]; mov rax, 0x800000; jmp rax
            let mut h = nested_rig(&[
                0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x80, 0x00, 0x48, 0xc7, 0xc0, 0x00, 0x00, 0x80,
                0x00, 0xff, 0xe0,
            ]);
            set_nested_region(&mut h, GUEST_CODE, NPT_PRESENT_WRITABLE);
            put(&mut h, VMCB_NP_ENABLE, 8, NP_ENABLE | GMET_ENABLE);
            h.write(NESTED_REMAPPED_HPA, ASM_CPUID);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
            assert_eq!(get(&h, VMCB_EXITINFO2, 8), NESTED_REMAPPED_GPA);
            let info = get(&h, VMCB_EXITINFO1, 8);
            assert_ne!(info & NPF_FETCH, 0);
            assert_eq!(info & NPF_USER, 0);
        }
        check();
        under_jit(check);
    }

    /// Sets Accessed in every guest page-table entry on the way to `linear` (the guest runs on
    /// the host's tables, 4-level).
    fn mark_accessed_along(h: &mut Harness, linear: u64) {
        const ACCESSED: u64 = 1 << 5;
        let mut table = h.cpu.cr3 & NPT_ADDRESS_MASK;
        for level in (0..NPT_LEVELS).rev() {
            let address = table + ((linear >> (12 + 9 * level)) & 0x1ff) * 8;
            let entry = u64::from_le_bytes(h.read(address, 8).try_into().unwrap());
            h.write(address, &(entry | ACCESSED).to_le_bytes());
            if level == 0 || entry & NPT_PAGE_SIZE != 0 {
                return;
            }
            table = entry & NPT_ADDRESS_MASK;
        }
    }

    /// The guest's top-level table lies in a 2 MiB region the nested tables map read-only (the
    /// code is fetched from it too, which reading allows).
    fn rig_with_read_only_guest_tables(controls: u64) -> Harness {
        let mut h = nested_rig(ASM_CPUID);
        let tables = h.cpu.cr3 & NPT_ADDRESS_MASK;
        set_nested_region(&mut h, tables, NPT_PRESENT_USER);
        put(&mut h, VMCB_NP_ENABLE, 8, controls);
        h
    }

    #[test]
    fn without_rogpt_a_guest_table_walk_needs_writable_nested_pages() {
        let mut h = rig_with_read_only_guest_tables(NP_ENABLE);
        mark_accessed_along(&mut h, GUEST_CODE);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_eq!(
            info & (NPF_PRESENT | NPF_WRITE | NPF_GUEST_TABLE_WALK),
            NPF_PRESENT | NPF_WRITE | NPF_GUEST_TABLE_WALK,
            "{info:#x}"
        );
        assert_eq!(
            get(&h, VMCB_EXITINFO2, 8) & !0xfff,
            h.cpu.cr3 & NPT_ADDRESS_MASK
        );
    }

    #[test]
    fn with_rogpt_a_guest_table_walk_writes_only_to_set_accessed() {
        let mut h = rig_with_read_only_guest_tables(NP_ENABLE | ROGPT_ENABLE);
        mark_accessed_along(&mut h, GUEST_CODE);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        // The same, with the top-level entry's Accessed bit still to set. The host's own fetch of
        // vmrun walks the same tables, so the bit is cleared once the guest runs.
        let mut h = rig_with_read_only_guest_tables(NP_ENABLE | ROGPT_ENABLE);
        mark_accessed_along(&mut h, GUEST_CODE);
        enter_guest(&mut h);
        let pml4_entry = (h.cpu.cr3 & NPT_ADDRESS_MASK) + ((GUEST_CODE >> 39) & 0x1ff) * 8;
        let entry = u64::from_le_bytes(h.read(pml4_entry, 8).try_into().unwrap());
        h.write(pml4_entry, &(entry & !(1 << 5)).to_le_bytes());
        h.cpu.flush_tlb();
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_NPF);
        let info = get(&h, VMCB_EXITINFO1, 8);
        assert_ne!(info & NPF_WRITE, 0);
        assert_ne!(info & NPF_GUEST_TABLE_WALK, 0);
    }

    #[test]
    fn host_mce_override_and_the_machine_check_intercept() {
        use crate::cpu::mca::MachineCheck;
        const MSR_MCG_CTL: u32 = 0x17b;
        const MSR_SMCA_BANK2_CTL: u32 = 0xc000_2020;
        const STATUS_UNCORRECTED_ENABLED: u64 = 1 << 61 | 1 << 60;
        const VECTOR_MACHINE_CHECK: u8 = 18;
        let error = MachineCheck {
            bank: 2,
            status: STATUS_UNCORRECTED_ENABLED,
            address: 0,
        };
        enum Outcome {
            Exit(u64),
            Delivered,
            Shutdown,
        }
        // (host CR4.MCE, guest CR4.MCE, #MC intercepted, shutdown intercepted)
        for (host_mce, guest_mce, intercepted, shutdown_intercepted, outcome) in [
            (
                true,
                false,
                false,
                false,
                Outcome::Exit(VMEXIT_EXCEPTION + 18),
            ),
            (
                true,
                true,
                true,
                false,
                Outcome::Exit(VMEXIT_EXCEPTION + 18),
            ),
            (
                false,
                true,
                true,
                false,
                Outcome::Exit(VMEXIT_EXCEPTION + 18),
            ),
            (true, true, false, false, Outcome::Delivered),
            (false, false, false, true, Outcome::Exit(VMEXIT_SHUTDOWN)),
            (false, false, false, false, Outcome::Shutdown),
        ] {
            let mut h = rig(ASM_CPUID);
            let handler = GUEST_CODE + 0x100;
            let cs = h.cpu.seg[SEG_CS].sel;
            install_idt_gate(&mut h, VECTOR_MACHINE_CHECK, handler, cs);
            put(&mut h, SAVE_IDTR + SEGMENT_LIMIT, 4, 0x4ff);
            put(&mut h, SAVE_IDTR + SEGMENT_BASE, 8, IDT);
            h.cpu.wrmsr(MSR_MCG_CTL, u64::MAX).unwrap();
            h.cpu.wrmsr(MSR_SMCA_BANK2_CTL, u64::MAX).unwrap();
            let mce = |on: bool| if on { CR4_MCE } else { 0 };
            h.cpu.cr4 = (h.cpu.cr4 & !CR4_MCE) | mce(host_mce);
            let guest_cr4 = get(&h, SAVE_CR4, 8);
            put(&mut h, SAVE_CR4, 8, (guest_cr4 & !CR4_MCE) | mce(guest_mce));
            put(
                &mut h,
                VMCB_INTERCEPT_EXCEPTIONS,
                4,
                if intercepted {
                    1 << VECTOR_MACHINE_CHECK
                } else {
                    0
                },
            );
            if shutdown_intercepted {
                let intercept3 = get(&h, VMCB_INTERCEPT3, 4);
                put(
                    &mut h,
                    VMCB_INTERCEPT3,
                    4,
                    intercept3 | INTERCEPT_SHUTDOWN as u64,
                );
            }
            enter_guest(&mut h);
            let reason = h.cpu.machine_check(error);
            let case = (host_mce, guest_mce, intercepted, shutdown_intercepted);
            match outcome {
                Outcome::Exit(code) => {
                    assert!(reason.is_none(), "{case:?}");
                    assert!(h.cpu.svm.guest.is_none(), "{case:?}");
                    assert_eq!(get(&h, VMCB_EXITCODE, 8), code, "{case:?}");
                }
                Outcome::Delivered => {
                    assert!(reason.is_none(), "{case:?}");
                    assert!(h.cpu.svm.guest.is_some(), "{case:?}");
                    assert_eq!(h.cpu.rip, handler, "{case:?}");
                }
                Outcome::Shutdown => {
                    assert_eq!(reason, Some(ExitReason::TripleFault), "{case:?}")
                }
            }
        }
    }

    const STALE_VIRTUAL: u64 = 0x0000_2000_0003_5000;
    const OLD_PAGE: u64 = SCRATCH + 0x4000;
    const NEW_PAGE: u64 = SCRATCH + 0x8000;
    const OLD_VALUE: u32 = 0x11;
    const NEW_VALUE: u32 = 0x22;
    // mov edx, [rbx]; vmrun; mov edx, [rbx]; hlt
    const HOST_LOAD_AROUND_VMRUN: &[u8] = &[0x8b, 0x13, 0x0f, 0x01, 0xd8, 0x8b, 0x13, 0xf4];
    // mov edx, [rbx]; mov [rdi], rsi; invlpgb; tlbsync; mov ecx, [rbx]; cpuid (ecx must be 0 at
    // invlpgb: it counts the extra pages)
    const GUEST_REMAP_THEN_INVLPGB: &[u8] = &[
        0x8b, 0x13, 0x48, 0x89, 0x37, 0x0f, 0x01, 0xfe, 0x0f, 0x01, 0xff, 0x8b, 0x0b, 0x0f, 0xa2,
    ];
    // the same with six nops instead of invlpgb; tlbsync
    const GUEST_REMAP_WITHOUT_INVLPGB: &[u8] = &[
        0x8b, 0x13, 0x48, 0x89, 0x37, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x8b, 0x0b, 0x0f, 0xa2,
    ];

    /// Host and guest share 4-level tables mapping `STALE_VIRTUAL` to `OLD_PAGE`. The host loads
    /// it before VMRUN and again after the #VMEXIT (into edx); the guest loads it (caching the
    /// translation), points the entry at `NEW_PAGE` (rdi, rsi), runs `guest_code`'s middle and
    /// loads it again (into ecx). Returns (the guest's second load, the host's second load).
    fn guest_remaps_a_page(guest_code: &[u8], controls: u64, nested: bool) -> (u64, u64) {
        use crate::cpu::test_support::{GDT_BASE, PTE_PRESENT_WRITABLE_USER, PageTables};
        use crate::testkit::{RDI, RSI};
        let flags = PTE_PRESENT_WRITABLE_USER;
        let (mut h, entry) = rig_prepared(guest_code, |h| {
            let mut tables = PageTables::new(&mut h.cpu);
            let code_page = CODE_BASE & !0xfff;
            for page in [code_page, code_page + 0x1000, GDT_BASE & !0xfff] {
                tables.map_4k(&mut h.cpu, page, page, flags);
            }
            let entry = tables.map_4k(&mut h.cpu, STALE_VIRTUAL, OLD_PAGE, flags);
            // the guest rewrites the entry through an identity mapping of its table
            tables.map_4k(&mut h.cpu, entry & !0xfff, entry & !0xfff, flags);
            h.write(OLD_PAGE, &OLD_VALUE.to_le_bytes());
            h.write(NEW_PAGE, &NEW_VALUE.to_le_bytes());
            h.cpu.flush_tlb();
            entry
        });
        if nested {
            build_nested_tables(&mut h);
            put(&mut h, VMCB_N_CR3, 8, NPT);
            put(&mut h, SAVE_G_PAT, 8, 0x0007_0406_0007_0406);
        }
        put(&mut h, VMCB_NP_ENABLE, 8, controls);
        h.write(CODE_BASE, HOST_LOAD_AROUND_VMRUN);
        h.cpu.gpr[RBX] = STALE_VIRTUAL;
        h.cpu.gpr[RDI] = entry;
        h.cpu.gpr[RSI] = NEW_PAGE | flags;
        h.cpu.gpr[RCX] = 0;
        put(&mut h, SAVE_RAX, 8, STALE_VIRTUAL | 1);
        run(&mut h);
        assert_eq!(get(&h, VMCB_EXITCODE, 8), VMEXIT_CPUID);
        (h.cpu.gpr[RCX], h.cpu.gpr[RDX])
    }

    #[test]
    fn invlpgb_in_a_guest_drops_its_own_translation_and_leaves_the_hosts() {
        for nested in [false, true] {
            let controls = INVLPGB_ENABLE | if nested { NP_ENABLE } else { 0 };
            assert_eq!(
                guest_remaps_a_page(GUEST_REMAP_WITHOUT_INVLPGB, controls, nested),
                (OLD_VALUE as u64, OLD_VALUE as u64),
                "nested {nested}: both translations are stale without it"
            );
            assert_eq!(
                guest_remaps_a_page(GUEST_REMAP_THEN_INVLPGB, controls, nested),
                (NEW_VALUE as u64, OLD_VALUE as u64),
                "nested {nested}: the guest's goes, the host's stays"
            );
        }
    }

    #[test]
    fn invlpgb_in_a_guest_is_ud_unless_the_vmcb_enables_it() {
        // invlpgb; tlbsync; cpuid
        const BOTH: &[u8] = &[0x0f, 0x01, 0xfe, 0x0f, 0x01, 0xff, 0x0f, 0xa2];
        for (controls, expected) in [(INVLPGB_ENABLE, VMEXIT_CPUID), (0, VMEXIT_EXCEPTION + 6)] {
            let mut h = rig(BOTH);
            put(&mut h, VMCB_NP_ENABLE, 8, controls);
            put(&mut h, VMCB_INTERCEPT_EXCEPTIONS, 4, 1 << 6);
            run(&mut h);
            assert_eq!(get(&h, VMCB_EXITCODE, 8), expected, "{controls:#x}");
        }
    }
}
