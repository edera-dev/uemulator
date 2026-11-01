//! Intel VMX backend with EPT and unrestricted guest. Field encodings, capability MSRs and bit
//! positions are from the Intel SDM Vol. 3 (VMCS fields, Appendix A and B). Never run on real
//! hardware yet (docs/plans/12-uefi.md). Pointers are used as physical addresses throughout: UEFI runs
//! with an identity map.

use alloc::format;
use alloc::string::String;
#[cfg(target_os = "uefi")]
use core::arch::asm;
#[cfg(target_os = "uefi")]
use uemulator::BootState;

use super::arch;
#[cfg(target_os = "uefi")]
use super::arch::{DescriptorTableRegister, Gprs};
use super::{Exit, PAGE_SIZE};
#[cfg(target_os = "uefi")]
use super::{Vcpu, alloc_pages};

const MSR_VMX_BASIC: u32 = 0x480;
const MSR_VMX_PINBASED: u32 = 0x481;
const MSR_VMX_PROCBASED: u32 = 0x482;
const MSR_VMX_EXIT: u32 = 0x483;
const MSR_VMX_ENTRY: u32 = 0x484;
const MSR_VMX_CR0_FIXED0: u32 = 0x486;
const MSR_VMX_CR0_FIXED1: u32 = 0x487;
const MSR_VMX_CR4_FIXED0: u32 = 0x488;
const MSR_VMX_CR4_FIXED1: u32 = 0x489;
const MSR_VMX_PROCBASED2: u32 = 0x48b;
const MSR_VMX_EPT_VPID_CAP: u32 = 0x48c;
const MSR_VMX_TRUE_PINBASED: u32 = 0x48d;
const MSR_VMX_TRUE_PROCBASED: u32 = 0x48e;
const MSR_VMX_TRUE_EXIT: u32 = 0x48f;
const MSR_VMX_TRUE_ENTRY: u32 = 0x490;

/// IA32_VMX_BASIC bits 30:0 are the VMCS revision identifier; bit 31 of the region header must
/// stay clear.
const VMX_BASIC_REVISION_MASK: u32 = 0x7fff_ffff;
/// IA32_VMX_BASIC bit 55: the IA32_VMX_TRUE_* control MSRs exist.
const VMX_BASIC_TRUE_CONTROLS: u64 = 1 << 55;

/// IA32_VMX_EPT_VPID_CAP bits: 4-level page walk, write-back EPT memory type, 2 MiB pages.
const EPT_CAP_PAGE_WALK_4: u64 = 1 << 6;
const EPT_CAP_WRITE_BACK: u64 = 1 << 14;
const EPT_CAP_2MIB_PAGES: u64 = 1 << 16;

/// IA32_FEATURE_CONTROL: lock bit, and "VMXON outside SMX" enable.
const FEATURE_LOCK: u64 = 1 << 0;
const FEATURE_VMXON_OUTSIDE_SMX: u64 = 1 << 2;
const CR4_VMXE: u64 = 1 << 13;

// VMCS field encodings, named as in the SDM. The `*_ES_*` ones are the first of a group of
// per-segment fields 2 apart, in the order of `SEGMENT_*` below.
const VMCS_MSR_BITMAP_ADDRESS: u64 = 0x2004;
const VMCS_VMEXIT_MSR_STORE_ADDRESS: u64 = 0x2006;
const VMCS_VMEXIT_MSR_LOAD_ADDRESS: u64 = 0x2008;
const VMCS_VMENTRY_MSR_LOAD_ADDRESS: u64 = 0x200a;
const VMCS_EPT_POINTER: u64 = 0x201a;
const VMCS_GUEST_PHYSICAL_ADDRESS: u64 = 0x2400;
const VMCS_LINK_POINTER: u64 = 0x2800;
const VMCS_GUEST_IA32_DEBUGCTL: u64 = 0x2802;
const VMCS_GUEST_IA32_PAT: u64 = 0x2804;
const VMCS_GUEST_IA32_EFER: u64 = 0x2806;
const VMCS_HOST_IA32_PAT: u64 = 0x2c00;
const VMCS_HOST_IA32_EFER: u64 = 0x2c02;
const VMCS_PIN_BASED_CONTROLS: u64 = 0x4000;
const VMCS_PRIMARY_PROC_BASED_CONTROLS: u64 = 0x4002;
const VMCS_EXCEPTION_BITMAP: u64 = 0x4004;
const VMCS_PAGE_FAULT_ERROR_CODE_MASK: u64 = 0x4006;
const VMCS_PAGE_FAULT_ERROR_CODE_MATCH: u64 = 0x4008;
const VMCS_CR3_TARGET_COUNT: u64 = 0x400a;
const VMCS_VMEXIT_CONTROLS: u64 = 0x400c;
const VMCS_VMEXIT_MSR_STORE_COUNT: u64 = 0x400e;
const VMCS_VMEXIT_MSR_LOAD_COUNT: u64 = 0x4010;
const VMCS_VMENTRY_CONTROLS: u64 = 0x4012;
const VMCS_VMENTRY_MSR_LOAD_COUNT: u64 = 0x4014;
const VMCS_VMENTRY_INTERRUPTION_INFO: u64 = 0x4016;
const VMCS_VMENTRY_EXCEPTION_ERROR_CODE: u64 = 0x4018;
const VMCS_VMENTRY_INSTRUCTION_LENGTH: u64 = 0x401a;
const VMCS_SECONDARY_PROC_BASED_CONTROLS: u64 = 0x401e;
const VMCS_VM_INSTRUCTION_ERROR: u64 = 0x4400;
const VMCS_EXIT_REASON: u64 = 0x4402;
const VMCS_IDT_VECTORING_INFO: u64 = 0x4408;
const VMCS_IDT_VECTORING_ERROR_CODE: u64 = 0x440a;
const VMCS_VMEXIT_INSTRUCTION_LENGTH: u64 = 0x440c;
const VMCS_CR0_GUEST_HOST_MASK: u64 = 0x6000;
const VMCS_CR4_GUEST_HOST_MASK: u64 = 0x6002;
const VMCS_CR0_READ_SHADOW: u64 = 0x6004;
const VMCS_CR4_READ_SHADOW: u64 = 0x6006;
const VMCS_EXIT_QUALIFICATION: u64 = 0x6400;

const VMCS_GUEST_ES_SELECTOR: u64 = 0x0800;
const VMCS_HOST_ES_SELECTOR: u64 = 0x0c00;
/// The limit group continues after the segment registers with GDTR and IDTR.
const VMCS_GUEST_ES_LIMIT: u64 = 0x4800;
const VMCS_GUEST_ES_ACCESS_RIGHTS: u64 = 0x4814;
/// The base group continues after the segment registers with GDTR and IDTR.
const VMCS_GUEST_ES_BASE: u64 = 0x6806;
const VMCS_GUEST_INTERRUPTIBILITY_STATE: u64 = 0x4824;
const VMCS_GUEST_ACTIVITY_STATE: u64 = 0x4826;
const VMCS_GUEST_IA32_SYSENTER_CS: u64 = 0x482a;
const VMCS_HOST_IA32_SYSENTER_CS: u64 = 0x4c00;
const VMCS_GUEST_CR0: u64 = 0x6800;
const VMCS_GUEST_CR3: u64 = 0x6802;
const VMCS_GUEST_CR4: u64 = 0x6804;
const VMCS_GUEST_DR7: u64 = 0x681a;
const VMCS_GUEST_RSP: u64 = 0x681c;
const VMCS_GUEST_RIP: u64 = 0x681e;
const VMCS_GUEST_RFLAGS: u64 = 0x6820;
const VMCS_GUEST_PENDING_DEBUG_EXCEPTIONS: u64 = 0x6822;
const VMCS_GUEST_IA32_SYSENTER_ESP: u64 = 0x6824;
const VMCS_GUEST_IA32_SYSENTER_EIP: u64 = 0x6826;
const VMCS_HOST_CR0: u64 = 0x6c00;
const VMCS_HOST_CR3: u64 = 0x6c02;
const VMCS_HOST_CR4: u64 = 0x6c04;
const VMCS_HOST_FS_BASE: u64 = 0x6c06;
const VMCS_HOST_GS_BASE: u64 = 0x6c08;
const VMCS_HOST_TR_BASE: u64 = 0x6c0a;
const VMCS_HOST_GDTR_BASE: u64 = 0x6c0c;
const VMCS_HOST_IDTR_BASE: u64 = 0x6c0e;
const VMCS_HOST_IA32_SYSENTER_ESP: u64 = 0x6c10;
const VMCS_HOST_IA32_SYSENTER_EIP: u64 = 0x6c12;
/// Read by the `uemu_vmx_enter` stub in `arch.rs`, which writes the host RSP itself.
pub(super) const VMCS_HOST_RSP: u64 = 0x6c14;
const VMCS_HOST_RIP: u64 = 0x6c16;

/// Index of each segment register within the per-segment VMCS field groups.
const SEGMENT_ES: u64 = 0;
const SEGMENT_CS: u64 = 1;
const SEGMENT_SS: u64 = 2;
const SEGMENT_DS: u64 = 3;
const SEGMENT_FS: u64 = 4;
const SEGMENT_GS: u64 = 5;
const SEGMENT_LDTR: u64 = 6;
const SEGMENT_TR: u64 = 7;
const SEGMENT_GDTR: u64 = 8;
const SEGMENT_IDTR: u64 = 9;
/// Fields of one group are 2 apart.
const VMCS_GROUP_STRIDE: u64 = 2;

// Guest segment access rights: 3:0 type, 4 S, 7 P, 13 L, 14 D/B, 15 G, 16 unusable (a different
// packing from SVM's).
const AR_TYPE_CODE_EXEC_READ_ACCESSED: u64 = 0xb;
const AR_TYPE_DATA_READ_WRITE_ACCESSED: u64 = 0x3;
const AR_TYPE_BUSY_TSS: u64 = 0xb;
const AR_S: u64 = 1 << 4;
const AR_P: u64 = 1 << 7;
const AR_L: u64 = 1 << 13;
const AR_DB: u64 = 1 << 14;
const AR_G: u64 = 1 << 15;
const AR_UNUSABLE: u64 = 1 << 16;
const AR_CODE_64BIT: u64 = AR_TYPE_CODE_EXEC_READ_ACCESSED | AR_S | AR_P | AR_L | AR_G;
const AR_DATA_FLAT: u64 = AR_TYPE_DATA_READ_WRITE_ACCESSED | AR_S | AR_P | AR_DB | AR_G;
const AR_TSS: u64 = AR_TYPE_BUSY_TSS | AR_P;

// Pin-based controls.
const PIN_EXTERNAL_INTERRUPT_EXITING: u32 = 1 << 0;
// Primary processor-based controls.
const PROC_INTERRUPT_WINDOW_EXITING: u32 = 1 << 2;
const PROC_HLT_EXITING: u32 = 1 << 7;
const PROC_UNCONDITIONAL_IO_EXITING: u32 = 1 << 24;
const PROC_USE_MSR_BITMAPS: u32 = 1 << 28;
const PROC_ACTIVATE_SECONDARY_CONTROLS: u32 = 1 << 31;
// Secondary processor-based controls.
const PROC2_ENABLE_EPT: u32 = 1 << 1;
const PROC2_UNRESTRICTED_GUEST: u32 = 1 << 7;
// VM-exit controls.
const EXIT_SAVE_DEBUG_CONTROLS: u32 = 1 << 2;
const EXIT_HOST_ADDRESS_SPACE_SIZE: u32 = 1 << 9;
const EXIT_SAVE_IA32_PAT: u32 = 1 << 18;
const EXIT_LOAD_IA32_PAT: u32 = 1 << 19;
const EXIT_SAVE_IA32_EFER: u32 = 1 << 20;
const EXIT_LOAD_IA32_EFER: u32 = 1 << 21;
// VM-entry controls.
const ENTRY_LOAD_DEBUG_CONTROLS: u32 = 1 << 2;
const ENTRY_IA32E_MODE_GUEST: u32 = 1 << 9;
const ENTRY_LOAD_IA32_PAT: u32 = 1 << 14;
const ENTRY_LOAD_IA32_EFER: u32 = 1 << 15;

/// EPT pointer: bits 2:0 memory type (6 = write-back), bits 5:3 page-walk length minus one.
const EPTP_MEMORY_TYPE_WRITE_BACK: u64 = 6;
const EPTP_PAGE_WALK_LENGTH_SHIFT: u32 = 3;
const EPTP_PAGE_WALK_LENGTH_4_MINUS_1: u64 = 3;

// Basic exit reasons.
const REASON_EXTERNAL_INTERRUPT: u64 = 1;
const REASON_TRIPLE_FAULT: u64 = 2;
const REASON_INTERRUPT_WINDOW: u64 = 7;
const REASON_CPUID: u64 = 10;
const REASON_HLT: u64 = 12;
const REASON_INVD: u64 = 13;
const REASON_VMCALL: u64 = 18;
const REASON_VMXON: u64 = 27;
const REASON_IO_INSTRUCTION: u64 = 30;
const REASON_RDMSR: u64 = 31;
const REASON_WRMSR: u64 = 32;
const REASON_EPT_VIOLATION: u64 = 48;
const REASON_XSETBV: u64 = 55;
/// Bit 31 of the exit reason: the exit is a failed VM entry. Bits 15:0 are the basic reason.
const EXIT_REASON_ENTRY_FAILURE: u64 = 1 << 31;
const EXIT_REASON_BASIC_MASK: u64 = 0xffff;

// Exit qualification for an I/O instruction: 2:0 size minus one, 3 direction (1 = IN), 4 string,
// 31:16 port.
const IO_QUAL_SIZE_MASK: u64 = 0b111;
const IO_QUAL_DIRECTION_IN: u64 = 1 << 3;
const IO_QUAL_STRING: u64 = 1 << 4;
const IO_QUAL_PORT_SHIFT: u32 = 16;
/// Exit qualification for an EPT violation: bit 1 is a data write.
const EPT_VIOLATION_QUAL_WRITE: u64 = 1 << 1;

/// Interruption-information types 4 (software interrupt), 5 (privileged software exception) and
/// 6 (software exception) are delivered with an instruction length; the range is bounded by the
/// first and last.
const EVENT_TYPE_MASK: u32 = 0b111;
const EVENT_TYPE_SOFTWARE_INTERRUPT: u32 = 4;
const EVENT_TYPE_SOFTWARE_EXCEPTION: u32 = 6;

/// Interruptibility state: blocking by STI (bit 0) and by MOV SS (bit 1).
const INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS: u64 = 0b11;

/// The MSR bitmap is 4 KiB: read bitmaps for the low and high MSR ranges, then the two write
/// bitmaps. Each range covers 0x2000 MSRs, so 1 KiB.
const MSR_BITMAP_HIGH_RANGE_OFFSET: usize = 1024;
const MSR_BITMAP_WRITE_OFFSET: usize = 2048;

/// One page of guest entries and one of host entries.
const MSR_LIST_PAGES: usize = 2;
/// Each MSR-list entry is 16 bytes: the MSR index (upper half reserved), then the value.
const MSR_LIST_ENTRY_U64S: usize = 2;

/// Limit of a minimal 64-bit TSS (104 bytes, minus one).
const TSS_LIMIT: u64 = 0x67;
/// Descriptor access byte: present, type 9 (available 64-bit TSS).
const TSS_ACCESS: u64 = 0x89;
const SELECTOR_RPL_AND_TABLE_MASK: u16 = 0b111;
// Bit positions inside the low qword of a 64-bit TSS descriptor.
const DESCRIPTOR_BASE_15_0_POS: u32 = 16;
const DESCRIPTOR_BASE_23_16_POS: u32 = 32;
const DESCRIPTOR_ACCESS_POS: u32 = 40;
const DESCRIPTOR_BASE_31_24_POS: u32 = 56;
// Where those pieces sit in the base address itself.
const BASE_23_16_SHIFT: u32 = 16;
const BASE_31_24_SHIFT: u32 = 24;
const BASE_63_32_SHIFT: u32 = 32;
/// A system descriptor takes two GDT slots in long mode.
const SYSTEM_DESCRIPTOR_SIZE: usize = 16;
/// A GDT limit is 16 bits, so the table is at most 64 KiB.
const MAX_GDT_BYTES: usize = 1 << 16;

/// Guest MSRs the CPU does not switch by itself; they live in the VM-entry load and VM-exit store
/// lists, with the host's values restored from the VM-exit load list.
const SWITCHED_MSRS: [u32; 5] = [
    arch::MSR_STAR,
    arch::MSR_LSTAR,
    arch::MSR_CSTAR,
    arch::MSR_SFMASK,
    arch::MSR_KERNEL_GS_BASE,
];

/// Field of segment `segment` in the per-segment group starting at `group`.
fn segment_field(group: u64, segment: u64) -> u64 {
    group + VMCS_GROUP_STRIDE * segment
}

#[cfg(target_os = "uefi")]
// vmwrite and vmread report failure in CF/ZF, which is not checked: a failing field access here
// would be a wrong encoding, and the VM entry then fails visibly. They modify flags, so no
// `preserves_flags`.
#[cfg(target_os = "uefi")]
#[inline]
fn vmwrite(field: u64, value: u64) {
    unsafe {
        asm!("vmwrite {f}, {v}", f = in(reg) field, v = in(reg) value, options(nostack));
    }
}

#[cfg(target_os = "uefi")]
#[inline]
fn vmread(field: u64) -> u64 {
    let v: u64;
    unsafe {
        asm!("vmread {v}, {f}", f = in(reg) field, v = out(reg) v, options(nostack));
    }
    v
}

/// Runs VMXON (`"on"`), VMCLEAR (`"clear"`) or, for anything else, VMPTRLD on the region at
/// physical address `pa`. Each takes a pointer to a memory operand holding that address, hence
/// the local. `setna` turns VMfailInvalid (CF) or VMfailValid (ZF) into a nonzero byte.
#[cfg(target_os = "uefi")]
fn vm_op(op: &str, pa: u64) -> bool {
    let failed: u8;
    unsafe {
        let addr = &pa as *const u64;
        match op {
            "on" => {
                asm!("vmxon [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
            "clear" => {
                asm!("vmclear [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
            _ => {
                asm!("vmptrld [{a}]", "setna {failed}", a = in(reg) addr, failed = out(reg_byte) failed, options(nostack))
            }
        }
    }
    failed == 0
}

#[cfg(target_os = "uefi")]
pub fn usable() -> bool {
    // The MSRs below do not exist without VMX, so this is checked before reading them.
    if arch::cpuid(arch::CPUID_LEAF_FEATURES, 0)[2] & arch::CPUID_1_ECX_VMX == 0 {
        return false;
    }
    usable_with(arch::rdmsr)
}

/// Whether the CPU (which has VMX) can run the guest this way, given a way to read MSRs. Each one
/// is read only once the checks before it have passed, as a missing MSR raises #GP.
fn usable_with(read_msr: impl Fn(u32) -> u64) -> bool {
    let feature_control = read_msr(arch::MSR_FEATURE_CONTROL);
    if feature_control & FEATURE_LOCK != 0 && feature_control & FEATURE_VMXON_OUTSIDE_SMX == 0 {
        return false;
    }
    // Without the TRUE control MSRs, CR3 accesses would be forced to exit.
    if read_msr(MSR_VMX_BASIC) & VMX_BASIC_TRUE_CONTROLS == 0 {
        return false;
    }
    let proc = read_msr(MSR_VMX_PROCBASED);
    let proc2 = read_msr(MSR_VMX_PROCBASED2);
    let ept_vpid_cap = read_msr(MSR_VMX_EPT_VPID_CAP);
    let ept_caps = EPT_CAP_PAGE_WALK_4 | EPT_CAP_WRITE_BACK | EPT_CAP_2MIB_PAGES;
    let proc2_wanted = PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST;
    // Secondary controls with EPT and unrestricted guest, 4-level walks, 2 MiB pages, WB memory.
    // The high half of a control MSR says which bits may be set.
    (proc >> 32) as u32 & PROC_ACTIVATE_SECONDARY_CONTROLS != 0
        && (proc2 >> 32) as u32 & proc2_wanted == proc2_wanted
        && ept_vpid_cap & ept_caps == ept_caps
}

/// Applies the capability MSR `msr` to a control word: bits set in its low half must be one, bits
/// clear in its high half cannot be set.
#[cfg(target_os = "uefi")]
fn adjust(msr: u32, wanted: u32) -> Result<u32, String> {
    adjust_with(msr, arch::rdmsr(msr), wanted)
}

/// `adjust` for the value `capability` of the MSR (`msr` only names it in the error).
fn adjust_with(msr: u32, capability: u64, wanted: u32) -> Result<u32, String> {
    let (must_be_one, may_be_one) = (capability as u32, (capability >> 32) as u32);
    if wanted & !may_be_one != 0 {
        return Err(format!(
            "VMX control MSR {msr:#x} cannot enable {:#x}",
            wanted & !may_be_one
        ));
    }
    Ok(wanted | must_be_one)
}

/// VMXON requires CR0 and CR4 to satisfy the fixed-bit MSRs: a bit set in FIXED0 must be one, a
/// bit clear in FIXED1 must be zero.
fn apply_fixed_bits(value: u64, fixed0: u64, fixed1: u64) -> u64 {
    (value | fixed0) & fixed1
}

/// The EPT pointer for the top-level table at `root`.
fn ept_pointer(root: u64) -> u64 {
    root | EPTP_MEMORY_TYPE_WRITE_BACK
        | (EPTP_PAGE_WALK_LENGTH_4_MINUS_1 << EPTP_PAGE_WALK_LENGTH_SHIFT)
}

#[cfg(target_os = "uefi")]
pub struct Vmx {
    gprs: Gprs,
    /// Primary processor-based controls without the interrupt-window bit, which `run` toggles.
    proc_controls: u32,
    /// VMLAUNCH has succeeded; later entries use VMRESUME.
    launched: bool,
    /// rip after the instruction behind the last exit, for `skip`.
    next_rip: u64,
    /// Length of that instruction.
    insn_len: u64,
    /// Interruption information (without the valid bit) and error code to inject on the next
    /// entry.
    event: Option<(u32, Option<u32>)>,
    /// Ask for an exit as soon as the guest can take an interrupt.
    window: bool,
    /// A VM exit leaves the GDTR and IDTR limits at 0xffff, so `run` reloads these.
    host_gdt: DescriptorTableRegister,
    host_idt: DescriptorTableRegister,
}

#[cfg(target_os = "uefi")]
impl Vmx {
    pub fn new(ept_root: u64, boot: &BootState) -> Result<Self, String> {
        let basic = arch::rdmsr(MSR_VMX_BASIC);
        let feature_control = arch::rdmsr(arch::MSR_FEATURE_CONTROL);
        // Firmware that left the MSR unlocked: enable VMXON and lock it, as VMXON needs. A locked
        // MSR with VMX off was already refused by `usable`.
        if feature_control & FEATURE_LOCK == 0 {
            arch::wrmsr(
                arch::MSR_FEATURE_CONTROL,
                feature_control | FEATURE_LOCK | FEATURE_VMXON_OUTSIDE_SMX,
            );
        }
        let fixed = |value: u64, fixed0: u32, fixed1: u32| {
            apply_fixed_bits(value, arch::rdmsr(fixed0), arch::rdmsr(fixed1))
        };
        arch::set_cr0(fixed(arch::cr0(), MSR_VMX_CR0_FIXED0, MSR_VMX_CR0_FIXED1));
        arch::set_cr4(fixed(
            arch::cr4() | CR4_VMXE,
            MSR_VMX_CR4_FIXED0,
            MSR_VMX_CR4_FIXED1,
        ));

        let vmxon_region = alloc_pages(1)?;
        let vmcs = alloc_pages(1)?;
        let msr_bitmap = alloc_pages(1)?;
        // Page 0: guest values (VM-entry load, VM-exit store). Page 1: host values (VM-exit load).
        let msr_list = alloc_pages(MSR_LIST_PAGES)?;
        // The first dword of the VMXON region and of the VMCS is the revision identifier.
        let revision = (basic as u32) & VMX_BASIC_REVISION_MASK;
        unsafe {
            (vmxon_region as *mut u32).write(revision);
            (vmcs as *mut u32).write(revision);
            // All ones: every MSR access exits unless cleared below.
            core::ptr::write_bytes(msr_bitmap, 0xff, PAGE_SIZE);
        }
        if !vm_op("on", vmxon_region as u64) {
            return Err("VMXON failed".into());
        }
        if !vm_op("clear", vmcs as u64) || !vm_op("ptrld", vmcs as u64) {
            return Err("VMCLEAR/VMPTRLD failed".into());
        }

        let bitmap = unsafe { core::slice::from_raw_parts_mut(msr_bitmap, PAGE_SIZE) };
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(bitmap, msr);
        }

        // The TRUE capability MSRs report which "default1" control bits can really be cleared.
        let true_controls = basic & VMX_BASIC_TRUE_CONTROLS != 0;
        let pick = |plain: u32, true_msr: u32| if true_controls { true_msr } else { plain };
        let pin = adjust(
            pick(MSR_VMX_PINBASED, MSR_VMX_TRUE_PINBASED),
            PIN_EXTERNAL_INTERRUPT_EXITING,
        )?;
        let proc = adjust(
            pick(MSR_VMX_PROCBASED, MSR_VMX_TRUE_PROCBASED),
            PROC_HLT_EXITING
                | PROC_UNCONDITIONAL_IO_EXITING
                | PROC_USE_MSR_BITMAPS
                | PROC_ACTIVATE_SECONDARY_CONTROLS,
        )?;
        let proc2 = adjust(
            MSR_VMX_PROCBASED2,
            PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST,
        )?;
        let exit = adjust(
            pick(MSR_VMX_EXIT, MSR_VMX_TRUE_EXIT),
            EXIT_SAVE_DEBUG_CONTROLS
                | EXIT_HOST_ADDRESS_SPACE_SIZE
                | EXIT_SAVE_IA32_PAT
                | EXIT_LOAD_IA32_PAT
                | EXIT_SAVE_IA32_EFER
                | EXIT_LOAD_IA32_EFER,
        )?;
        let entry = adjust(
            pick(MSR_VMX_ENTRY, MSR_VMX_TRUE_ENTRY),
            ENTRY_LOAD_DEBUG_CONTROLS
                | ENTRY_IA32E_MODE_GUEST
                | ENTRY_LOAD_IA32_PAT
                | ENTRY_LOAD_IA32_EFER,
        )?;

        vmwrite(VMCS_PIN_BASED_CONTROLS, pin as u64);
        vmwrite(VMCS_PRIMARY_PROC_BASED_CONTROLS, proc as u64);
        vmwrite(VMCS_SECONDARY_PROC_BASED_CONTROLS, proc2 as u64);
        vmwrite(VMCS_VMEXIT_CONTROLS, exit as u64);
        vmwrite(VMCS_VMENTRY_CONTROLS, entry as u64);
        vmwrite(VMCS_MSR_BITMAP_ADDRESS, msr_bitmap as u64);
        vmwrite(VMCS_EPT_POINTER, ept_pointer(ept_root));
        vmwrite(VMCS_EXCEPTION_BITMAP, 0);
        vmwrite(VMCS_PAGE_FAULT_ERROR_CODE_MASK, 0);
        vmwrite(VMCS_PAGE_FAULT_ERROR_CODE_MATCH, 0);
        vmwrite(VMCS_CR3_TARGET_COUNT, 0);
        // All ones: there is no shadow VMCS.
        vmwrite(VMCS_LINK_POINTER, u64::MAX);

        let host_list = unsafe { msr_list.add(PAGE_SIZE) };
        for (i, msr) in SWITCHED_MSRS.iter().enumerate() {
            unsafe {
                let guest_entry = (msr_list as *mut u64).add(i * MSR_LIST_ENTRY_U64S);
                guest_entry.write(*msr as u64);
                guest_entry.add(1).write(0);
                let host_entry = (host_list as *mut u64).add(i * MSR_LIST_ENTRY_U64S);
                host_entry.write(*msr as u64);
                host_entry.add(1).write(arch::rdmsr(*msr));
            }
        }
        vmwrite(VMCS_VMENTRY_MSR_LOAD_ADDRESS, msr_list as u64);
        vmwrite(VMCS_VMENTRY_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(VMCS_VMEXIT_MSR_STORE_ADDRESS, msr_list as u64);
        vmwrite(VMCS_VMEXIT_MSR_STORE_COUNT, SWITCHED_MSRS.len() as u64);
        vmwrite(VMCS_VMEXIT_MSR_LOAD_ADDRESS, host_list as u64);
        vmwrite(VMCS_VMEXIT_MSR_LOAD_COUNT, SWITCHED_MSRS.len() as u64);

        let host_gdt = arch::sgdt();
        let host_idt = arch::sidt();
        let task_register = host_task_register(&host_gdt)?;
        // Read again: `host_task_register` may have loaded a new GDT.
        let host_gdt = arch::sgdt();
        let tr_base = tss_base(&host_gdt, task_register);
        for (i, selector) in [
            arch::es(),
            arch::cs(),
            arch::ss(),
            arch::ds(),
            arch::fs(),
            arch::gs(),
            task_register,
        ]
        .into_iter()
        .enumerate()
        {
            // The host group is ES, CS, SS, DS, FS, GS, TR (no LDTR). Its selectors must have
            // RPL and TI clear.
            vmwrite(
                segment_field(VMCS_HOST_ES_SELECTOR, i as u64),
                (selector & !SELECTOR_RPL_AND_TABLE_MASK) as u64,
            );
        }
        vmwrite(VMCS_HOST_CR0, arch::cr0());
        vmwrite(VMCS_HOST_CR3, arch::cr3());
        vmwrite(VMCS_HOST_CR4, arch::cr4());
        vmwrite(VMCS_HOST_FS_BASE, arch::rdmsr(arch::MSR_FS_BASE));
        vmwrite(VMCS_HOST_GS_BASE, arch::rdmsr(arch::MSR_GS_BASE));
        vmwrite(VMCS_HOST_TR_BASE, tr_base);
        let (gdtr_base, idtr_base) = (host_gdt.base, host_idt.base);
        vmwrite(VMCS_HOST_GDTR_BASE, gdtr_base);
        vmwrite(VMCS_HOST_IDTR_BASE, idtr_base);
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_CS,
            arch::rdmsr(arch::MSR_SYSENTER_CS),
        );
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_ESP,
            arch::rdmsr(arch::MSR_SYSENTER_ESP),
        );
        vmwrite(
            VMCS_HOST_IA32_SYSENTER_EIP,
            arch::rdmsr(arch::MSR_SYSENTER_EIP),
        );
        vmwrite(VMCS_HOST_IA32_PAT, arch::rdmsr(arch::MSR_PAT));
        vmwrite(VMCS_HOST_IA32_EFER, arch::rdmsr(arch::MSR_EFER));
        // The host RSP is written by the entry stub on every entry.
        vmwrite(
            VMCS_HOST_RIP,
            arch::uemu_vmx_exit as *const () as usize as u64,
        );

        // Guest state: the 64-bit Linux entry.
        let cr0 = fixed(
            arch::GUEST_ENTRY_CR0,
            MSR_VMX_CR0_FIXED0,
            MSR_VMX_CR0_FIXED1,
        );
        let cr4_fixed0 = arch::rdmsr(MSR_VMX_CR4_FIXED0);
        vmwrite(VMCS_GUEST_CR0, cr0);
        vmwrite(VMCS_GUEST_CR3, boot.cr3);
        vmwrite(VMCS_GUEST_CR4, arch::GUEST_ENTRY_CR4 | cr4_fixed0);
        // VMXE is the host's: the guest reads it as clear and its writes of it would exit.
        vmwrite(VMCS_CR0_GUEST_HOST_MASK, 0);
        vmwrite(VMCS_CR0_READ_SHADOW, cr0);
        vmwrite(VMCS_CR4_GUEST_HOST_MASK, CR4_VMXE);
        vmwrite(
            VMCS_CR4_READ_SHADOW,
            arch::GUEST_ENTRY_CR4 | (cr4_fixed0 & !CR4_VMXE),
        );
        vmwrite(VMCS_GUEST_DR7, arch::DR7_RESET);
        vmwrite(VMCS_GUEST_RSP, 0);
        vmwrite(VMCS_GUEST_RIP, boot.rip);
        vmwrite(VMCS_GUEST_RFLAGS, arch::RFLAGS_RESERVED_ONE);
        vmwrite(VMCS_GUEST_IA32_EFER, arch::EFER_LME | arch::EFER_LMA);
        vmwrite(VMCS_GUEST_IA32_PAT, arch::PAT_RESET);
        vmwrite(VMCS_GUEST_IA32_DEBUGCTL, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_CS, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_ESP, 0);
        vmwrite(VMCS_GUEST_IA32_SYSENTER_EIP, 0);
        vmwrite(VMCS_GUEST_INTERRUPTIBILITY_STATE, 0);
        vmwrite(VMCS_GUEST_ACTIVITY_STATE, 0);
        vmwrite(VMCS_GUEST_PENDING_DEBUG_EXCEPTIONS, 0);
        let set_segment =
            |segment: u64, selector: u64, limit: u64, access_rights: u64, base: u64| {
                vmwrite(segment_field(VMCS_GUEST_ES_SELECTOR, segment), selector);
                vmwrite(segment_field(VMCS_GUEST_ES_LIMIT, segment), limit);
                vmwrite(
                    segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, segment),
                    access_rights,
                );
                vmwrite(segment_field(VMCS_GUEST_ES_BASE, segment), base);
            };
        let flat_limit = arch::FLAT_SEGMENT_LIMIT as u64;
        let ds = boot.ds as u64;
        set_segment(SEGMENT_ES, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_CS, boot.cs as u64, flat_limit, AR_CODE_64BIT, 0);
        set_segment(SEGMENT_SS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_DS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_FS, ds, flat_limit, AR_DATA_FLAT, 0);
        set_segment(SEGMENT_GS, ds, flat_limit, AR_DATA_FLAT, 0);
        let system_limit = arch::SYSTEM_SEGMENT_LIMIT as u64;
        set_segment(SEGMENT_LDTR, 0, system_limit, AR_UNUSABLE, 0);
        set_segment(SEGMENT_TR, 0, system_limit, AR_TSS, 0);
        // GDTR and IDTR have only a limit and a base field, no selector or access rights.
        vmwrite(
            segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_GDTR),
            boot.gdt_limit as u64,
        );
        vmwrite(
            segment_field(VMCS_GUEST_ES_BASE, SEGMENT_GDTR),
            boot.gdt_base,
        );
        vmwrite(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_IDTR), 0);
        vmwrite(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_IDTR), 0);

        Ok(Self {
            gprs: Gprs {
                rsi: boot.rsi,
                ..Gprs::default()
            },
            proc_controls: proc,
            launched: false,
            next_rip: 0,
            insn_len: 0,
            event: None,
            window: false,
            host_gdt,
            host_idt,
        })
    }
}

/// Where the copied GDT puts its new TSS descriptor, and how much memory the copy needs.
#[derive(Debug, PartialEq, Eq)]
struct GdtLayout {
    /// Byte offset of the descriptor, 16-aligned past the old table. It is also the selector.
    descriptor_offset: usize,
    /// 4 KiB pages for the old table, any padding and the descriptor.
    pages: usize,
    /// Limit of the extended table.
    limit: u16,
}

/// Layout of a copy of the GDT with limit `old_limit` plus one system descriptor. Fails when the
/// descriptor would not fit under the 64 KiB a GDT can span.
fn gdt_layout(old_limit: u16) -> Result<GdtLayout, String> {
    let old_len = old_limit as usize + 1;
    let descriptor_offset = old_len.next_multiple_of(SYSTEM_DESCRIPTOR_SIZE);
    let end = descriptor_offset + SYSTEM_DESCRIPTOR_SIZE;
    if end > MAX_GDT_BYTES {
        return Err(format!(
            "the firmware GDT ({old_len} bytes) has no room for a task state descriptor"
        ));
    }
    Ok(GdtLayout {
        descriptor_offset,
        pages: end.div_ceil(PAGE_SIZE),
        limit: (end - 1) as u16,
    })
}

/// The two qwords of an available 64-bit TSS descriptor of limit `TSS_LIMIT` at `base`. Low qword:
/// limit 15:0, base 23:0 split around the access byte, base 31:24. High qword: base 63:32.
fn tss_descriptor(base: u64) -> [u64; 2] {
    let low = TSS_LIMIT
        | (base & u16::MAX as u64) << DESCRIPTOR_BASE_15_0_POS
        | ((base >> BASE_23_16_SHIFT) & u8::MAX as u64) << DESCRIPTOR_BASE_23_16_POS
        | TSS_ACCESS << DESCRIPTOR_ACCESS_POS
        | ((base >> BASE_31_24_SHIFT) & u8::MAX as u64) << DESCRIPTOR_BASE_31_24_POS;
    [low, base >> BASE_63_32_SHIFT]
}

/// The host needs a task register with a valid TSS for VM exits to load. The firmware sets one up
/// on some platforms; where it did not, a copy of its GDT gets a TSS descriptor.
#[cfg(target_os = "uefi")]
fn host_task_register(gdt: &DescriptorTableRegister) -> Result<u16, String> {
    let selector = arch::task_register_selector();
    if selector != 0 {
        return Ok(selector);
    }
    let old_len = gdt.limit as usize + 1;
    let layout = gdt_layout(gdt.limit)?;
    // The copied GDT and its new descriptor go in the first pages, the (zeroed) TSS in the last.
    let page = alloc_pages(layout.pages + 1)?;
    let tss = unsafe { page.add(layout.pages * PAGE_SIZE) };
    let base = tss as u64;
    unsafe {
        core::ptr::copy_nonoverlapping(gdt.base as *const u8, page, old_len);
        let descriptor = page.add(layout.descriptor_offset) as *mut u64;
        let [low, high] = tss_descriptor(base);
        descriptor.write(low);
        descriptor.add(1).write(high);
    }
    let new_gdt = DescriptorTableRegister {
        limit: layout.limit,
        base: page as u64,
    };
    arch::lgdt(&new_gdt);
    // The selector is the byte offset of the descriptor: index in bits 15:3, TI and RPL zero.
    let selector = layout.descriptor_offset as u16;
    // No `nomem`: LTR reads the GDT and marks the TSS descriptor busy.
    unsafe { asm!("ltr {:x}", in(reg) selector, options(nostack)) };
    Ok(selector)
}

/// Base address of the 64-bit TSS that `selector` names in `gdt`.
#[cfg(target_os = "uefi")]
fn tss_base(gdt: &DescriptorTableRegister, selector: u16) -> u64 {
    let descriptor = (gdt.base + (selector & !SELECTOR_RPL_AND_TABLE_MASK) as u64) as *const u64;
    let (low, high) = unsafe {
        (
            descriptor.read_unaligned(),
            descriptor.add(1).read_unaligned(),
        )
    };
    tss_base_from_descriptor(low, high)
}

/// The base address in the two qwords of a 64-bit TSS descriptor.
fn tss_base_from_descriptor(low: u64, high: u64) -> u64 {
    // Base 23:0 is contiguous from bit 16 of the low qword.
    ((low >> DESCRIPTOR_BASE_15_0_POS) & ((1 << BASE_31_24_SHIFT) - 1))
        | ((low >> DESCRIPTOR_BASE_31_24_POS) << BASE_31_24_SHIFT)
        | (high << BASE_63_32_SHIFT)
}

/// Clears the read and write intercept bits of `msr` in the MSR bitmap.
fn pass_through(bitmap: &mut [u8], msr: u32) {
    let (base, index) = match msr {
        0..=arch::MSR_LOW_RANGE_END => (0, msr),
        arch::MSR_HIGH_RANGE_START..=arch::MSR_HIGH_RANGE_END => (
            MSR_BITMAP_HIGH_RANGE_OFFSET,
            msr - arch::MSR_HIGH_RANGE_START,
        ),
        _ => return,
    };
    let (byte, bit) = (index as usize / 8, index % 8);
    bitmap[base + byte] &= !(1 << bit);
    bitmap[MSR_BITMAP_WRITE_OFFSET + base + byte] &= !(1 << bit);
}

/// A port I/O exit for INS or OUTS, which this backend does not emulate.
#[derive(Debug, PartialEq, Eq)]
struct StringIo {
    port: u16,
}

/// Maps a basic exit reason to what the event loop handles. `guest_physical_address` is only read
/// for an EPT violation.
fn decode_exit(
    reason: u64,
    qualification: u64,
    guest_physical_address: impl FnOnce() -> u64,
) -> Result<Exit, StringIo> {
    Ok(match reason {
        REASON_EXTERNAL_INTERRUPT => Exit::HostInterrupt,
        REASON_INTERRUPT_WINDOW => Exit::IntWindow,
        REASON_CPUID => Exit::Cpuid,
        REASON_HLT => Exit::Hlt,
        REASON_INVD => Exit::Skip,
        REASON_RDMSR => Exit::Rdmsr,
        REASON_WRMSR => Exit::Wrmsr,
        REASON_TRIPLE_FAULT => Exit::TripleFault,
        // VMX instructions (VMCALL through VMXON) and XSETBV.
        REASON_VMCALL..=REASON_VMXON | REASON_XSETBV => Exit::Undefined,
        REASON_IO_INSTRUCTION => {
            let port = (qualification >> IO_QUAL_PORT_SHIFT) as u16;
            if qualification & IO_QUAL_STRING != 0 {
                return Err(StringIo { port });
            }
            Exit::Io {
                port,
                size: (qualification & IO_QUAL_SIZE_MASK) as u8 + 1,
                write: qualification & IO_QUAL_DIRECTION_IN == 0,
            }
        }
        REASON_EPT_VIOLATION => Exit::Fault {
            gpa: guest_physical_address(),
            write: qualification & EPT_VIOLATION_QUAL_WRITE != 0,
        },
        other => Exit::Other(other),
    })
}

/// The event that was being delivered when a VM exit of `reason` happened, to deliver again:
/// interruption information (without the valid bit) and error code. An EPT violation is not
/// reported this way, as the access is retried by the guest itself.
fn event_to_redeliver(
    reason: u64,
    vectoring_info: u32,
    vectoring_error_code: impl FnOnce() -> u32,
) -> Option<(u32, Option<u32>)> {
    if vectoring_info & arch::EVENT_VALID == 0 || reason == REASON_EPT_VIOLATION {
        return None;
    }
    let error_code =
        (vectoring_info & arch::EVENT_ERROR_CODE_VALID != 0).then(vectoring_error_code);
    Some((vectoring_info & !arch::EVENT_VALID, error_code))
}

/// Software interrupts and exceptions are delivered with an instruction length.
fn carries_instruction_length(interruption_info: u32) -> bool {
    matches!(
        (interruption_info >> arch::EVENT_TYPE_SHIFT) & EVENT_TYPE_MASK,
        EVENT_TYPE_SOFTWARE_INTERRUPT..=EVENT_TYPE_SOFTWARE_EXCEPTION
    )
}

#[cfg(target_os = "uefi")]
impl Vcpu for Vmx {
    fn gprs(&mut self) -> &mut Gprs {
        &mut self.gprs
    }

    fn run(&mut self) -> Result<Exit, String> {
        match self.event.take() {
            Some((info, error_code)) => {
                if let Some(error_code) = error_code {
                    vmwrite(VMCS_VMENTRY_EXCEPTION_ERROR_CODE, error_code as u64);
                }
                if carries_instruction_length(info) {
                    vmwrite(VMCS_VMENTRY_INSTRUCTION_LENGTH, self.insn_len);
                }
                vmwrite(
                    VMCS_VMENTRY_INTERRUPTION_INFO,
                    (info | arch::EVENT_VALID) as u64,
                );
            }
            None => vmwrite(VMCS_VMENTRY_INTERRUPTION_INFO, 0),
        }
        let controls = if self.window {
            self.proc_controls | PROC_INTERRUPT_WINDOW_EXITING
        } else {
            self.proc_controls & !PROC_INTERRUPT_WINDOW_EXITING
        };
        vmwrite(VMCS_PRIMARY_PROC_BASED_CONTROLS, controls as u64);

        let failed = unsafe { arch::uemu_vmx_enter(&mut self.gprs, self.launched as u32) };
        arch::lgdt(&self.host_gdt);
        arch::lidt(&self.host_idt);
        if failed != 0 {
            return Err(format!(
                "VMLAUNCH/VMRESUME failed, instruction error {}",
                vmread(VMCS_VM_INSTRUCTION_ERROR)
            ));
        }
        self.launched = true;

        let exit_reason = vmread(VMCS_EXIT_REASON);
        let qualification = vmread(VMCS_EXIT_QUALIFICATION);
        let rip = vmread(VMCS_GUEST_RIP);
        self.insn_len = vmread(VMCS_VMEXIT_INSTRUCTION_LENGTH);
        self.next_rip = rip + self.insn_len;
        if exit_reason & EXIT_REASON_ENTRY_FAILURE != 0 {
            return Err(format!(
                "VM entry failed: reason {:#x} qualification {qualification:#x} (rip {rip:#x})",
                exit_reason & EXIT_REASON_BASIC_MASK
            ));
        }
        let reason = exit_reason & EXIT_REASON_BASIC_MASK;
        // An event that was being delivered when the exit happened is delivered again.
        let vectoring = vmread(VMCS_IDT_VECTORING_INFO) as u32;
        if let Some(event) = event_to_redeliver(reason, vectoring, || {
            vmread(VMCS_IDT_VECTORING_ERROR_CODE) as u32
        }) {
            self.event = Some(event);
        }
        decode_exit(reason, qualification, || {
            vmread(VMCS_GUEST_PHYSICAL_ADDRESS)
        })
        .map_err(|StringIo { port }| {
            format!("string port I/O (port {port:#x}) at rip {rip:#x} is not supported")
        })
    }

    fn skip(&mut self) {
        vmwrite(VMCS_GUEST_RIP, self.next_rip);
        let state = vmread(VMCS_GUEST_INTERRUPTIBILITY_STATE);
        vmwrite(
            VMCS_GUEST_INTERRUPTIBILITY_STATE,
            state & !INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS,
        );
    }

    fn rip(&self) -> u64 {
        vmread(VMCS_GUEST_RIP)
    }

    fn interrupts_enabled(&self) -> bool {
        vmread(VMCS_GUEST_RFLAGS) & arch::RFLAGS_IF != 0
    }

    fn irq_ok(&self) -> bool {
        self.interrupts_enabled()
            && vmread(VMCS_GUEST_INTERRUPTIBILITY_STATE)
                & INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS
                == 0
            && self.event.is_none()
    }

    fn inject_irq(&mut self, vector: u8) {
        self.event = Some((vector as u32, None));
    }

    fn window(&mut self, on: bool) {
        self.window = on;
    }

    fn inject_gp(&mut self) {
        self.event = Some((
            arch::exception_event(arch::VECTOR_GENERAL_PROTECTION, true),
            Some(0),
        ));
    }

    fn inject_ud(&mut self) {
        self.event = Some((
            arch::exception_event(arch::VECTOR_INVALID_OPCODE, false),
            None,
        ));
    }

    fn efer(&self) -> u64 {
        vmread(VMCS_GUEST_IA32_EFER)
    }

    fn set_efer(&mut self, efer: u64) {
        vmwrite(VMCS_GUEST_IA32_EFER, efer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    // VMCS field encodings (SDM Vol. 3, Appendix B).

    #[test]
    fn guest_segment_selector_fields_are_two_apart_from_0x800() {
        let expected = [0x800, 0x802, 0x804, 0x806, 0x808, 0x80a, 0x80c, 0x80e];
        let segments = [
            SEGMENT_ES,
            SEGMENT_CS,
            SEGMENT_SS,
            SEGMENT_DS,
            SEGMENT_FS,
            SEGMENT_GS,
            SEGMENT_LDTR,
            SEGMENT_TR,
        ];
        for (segment, field) in segments.into_iter().zip(expected) {
            assert_eq!(segment_field(VMCS_GUEST_ES_SELECTOR, segment), field);
        }
    }

    #[test]
    fn guest_limit_fields_continue_with_gdtr_and_idtr() {
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_ES), 0x4800);
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_CS), 0x4802);
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_LDTR), 0x480c);
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_TR), 0x480e);
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_GDTR), 0x4810);
        assert_eq!(segment_field(VMCS_GUEST_ES_LIMIT, SEGMENT_IDTR), 0x4812);
    }

    #[test]
    fn guest_access_rights_fields_are_at_0x4814_and_up() {
        assert_eq!(
            segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, SEGMENT_ES),
            0x4814
        );
        assert_eq!(
            segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, SEGMENT_CS),
            0x4816
        );
        assert_eq!(
            segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, SEGMENT_FS),
            0x481c
        );
        assert_eq!(
            segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, SEGMENT_LDTR),
            0x4820
        );
        assert_eq!(
            segment_field(VMCS_GUEST_ES_ACCESS_RIGHTS, SEGMENT_TR),
            0x4822
        );
    }

    #[test]
    fn guest_base_fields_continue_with_gdtr_and_idtr() {
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_ES), 0x6806);
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_CS), 0x6808);
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_FS), 0x680e);
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_TR), 0x6814);
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_GDTR), 0x6816);
        assert_eq!(segment_field(VMCS_GUEST_ES_BASE, SEGMENT_IDTR), 0x6818);
    }

    #[test]
    fn host_selector_fields_end_with_tr_at_0xc0c() {
        // The host group has no LDTR, so TR is the seventh entry.
        let selectors = [0xc00, 0xc02, 0xc04, 0xc06, 0xc08, 0xc0a, 0xc0c];
        for (index, field) in selectors.into_iter().enumerate() {
            assert_eq!(segment_field(VMCS_HOST_ES_SELECTOR, index as u64), field);
        }
    }

    #[test]
    fn segment_indices_follow_the_sdm_order() {
        let order = [
            SEGMENT_ES,
            SEGMENT_CS,
            SEGMENT_SS,
            SEGMENT_DS,
            SEGMENT_FS,
            SEGMENT_GS,
            SEGMENT_LDTR,
            SEGMENT_TR,
            SEGMENT_GDTR,
            SEGMENT_IDTR,
        ];
        for (expected, actual) in order.into_iter().enumerate() {
            assert_eq!(actual, expected as u64);
        }
    }

    #[test]
    fn control_and_read_only_fields_have_their_sdm_encodings() {
        let fields = [
            (VMCS_PIN_BASED_CONTROLS, 0x4000),
            (VMCS_PRIMARY_PROC_BASED_CONTROLS, 0x4002),
            (VMCS_EXCEPTION_BITMAP, 0x4004),
            (VMCS_PAGE_FAULT_ERROR_CODE_MASK, 0x4006),
            (VMCS_PAGE_FAULT_ERROR_CODE_MATCH, 0x4008),
            (VMCS_CR3_TARGET_COUNT, 0x400a),
            (VMCS_VMEXIT_CONTROLS, 0x400c),
            (VMCS_VMEXIT_MSR_STORE_COUNT, 0x400e),
            (VMCS_VMEXIT_MSR_LOAD_COUNT, 0x4010),
            (VMCS_VMENTRY_CONTROLS, 0x4012),
            (VMCS_VMENTRY_MSR_LOAD_COUNT, 0x4014),
            (VMCS_VMENTRY_INTERRUPTION_INFO, 0x4016),
            (VMCS_VMENTRY_EXCEPTION_ERROR_CODE, 0x4018),
            (VMCS_VMENTRY_INSTRUCTION_LENGTH, 0x401a),
            (VMCS_SECONDARY_PROC_BASED_CONTROLS, 0x401e),
            (VMCS_VM_INSTRUCTION_ERROR, 0x4400),
            (VMCS_EXIT_REASON, 0x4402),
            (VMCS_IDT_VECTORING_INFO, 0x4408),
            (VMCS_IDT_VECTORING_ERROR_CODE, 0x440a),
            (VMCS_VMEXIT_INSTRUCTION_LENGTH, 0x440c),
            (VMCS_EXIT_QUALIFICATION, 0x6400),
            (VMCS_GUEST_PHYSICAL_ADDRESS, 0x2400),
        ];
        for (field, encoding) in fields {
            assert_eq!(field, encoding);
        }
    }

    #[test]
    fn address_and_64_bit_fields_have_their_sdm_encodings() {
        let fields = [
            (VMCS_MSR_BITMAP_ADDRESS, 0x2004),
            (VMCS_VMEXIT_MSR_STORE_ADDRESS, 0x2006),
            (VMCS_VMEXIT_MSR_LOAD_ADDRESS, 0x2008),
            (VMCS_VMENTRY_MSR_LOAD_ADDRESS, 0x200a),
            (VMCS_EPT_POINTER, 0x201a),
            (VMCS_LINK_POINTER, 0x2800),
            (VMCS_GUEST_IA32_DEBUGCTL, 0x2802),
            (VMCS_GUEST_IA32_PAT, 0x2804),
            (VMCS_GUEST_IA32_EFER, 0x2806),
            (VMCS_HOST_IA32_PAT, 0x2c00),
            (VMCS_HOST_IA32_EFER, 0x2c02),
        ];
        for (field, encoding) in fields {
            assert_eq!(field, encoding);
        }
    }

    #[test]
    fn guest_state_fields_have_their_sdm_encodings() {
        let fields = [
            (VMCS_GUEST_CR0, 0x6800),
            (VMCS_GUEST_CR3, 0x6802),
            (VMCS_GUEST_CR4, 0x6804),
            (VMCS_GUEST_DR7, 0x681a),
            (VMCS_GUEST_RSP, 0x681c),
            (VMCS_GUEST_RIP, 0x681e),
            (VMCS_GUEST_RFLAGS, 0x6820),
            (VMCS_GUEST_PENDING_DEBUG_EXCEPTIONS, 0x6822),
            (VMCS_GUEST_IA32_SYSENTER_ESP, 0x6824),
            (VMCS_GUEST_IA32_SYSENTER_EIP, 0x6826),
            (VMCS_GUEST_INTERRUPTIBILITY_STATE, 0x4824),
            (VMCS_GUEST_ACTIVITY_STATE, 0x4826),
            (VMCS_GUEST_IA32_SYSENTER_CS, 0x482a),
            (VMCS_CR0_GUEST_HOST_MASK, 0x6000),
            (VMCS_CR4_GUEST_HOST_MASK, 0x6002),
            (VMCS_CR0_READ_SHADOW, 0x6004),
            (VMCS_CR4_READ_SHADOW, 0x6006),
        ];
        for (field, encoding) in fields {
            assert_eq!(field, encoding);
        }
    }

    #[test]
    fn host_state_fields_have_their_sdm_encodings() {
        let fields = [
            (VMCS_HOST_CR0, 0x6c00),
            (VMCS_HOST_CR3, 0x6c02),
            (VMCS_HOST_CR4, 0x6c04),
            (VMCS_HOST_FS_BASE, 0x6c06),
            (VMCS_HOST_GS_BASE, 0x6c08),
            (VMCS_HOST_TR_BASE, 0x6c0a),
            (VMCS_HOST_GDTR_BASE, 0x6c0c),
            (VMCS_HOST_IDTR_BASE, 0x6c0e),
            (VMCS_HOST_IA32_SYSENTER_ESP, 0x6c10),
            (VMCS_HOST_IA32_SYSENTER_EIP, 0x6c12),
            (VMCS_HOST_RSP, 0x6c14),
            (VMCS_HOST_RIP, 0x6c16),
            (VMCS_HOST_IA32_SYSENTER_CS, 0x4c00),
        ];
        for (field, encoding) in fields {
            assert_eq!(field, encoding);
        }
    }

    // Segment access rights.

    #[test]
    fn long_mode_code_segment_access_rights() {
        // Type 0xb (execute/read, accessed), S, P, L and G.
        assert_eq!(AR_CODE_64BIT, 0xa09b);
        assert_eq!(AR_CODE_64BIT & AR_DB, 0, "D must be clear when L is set");
    }

    #[test]
    fn flat_data_segment_access_rights() {
        // Type 3 (read/write, accessed), S, P, D/B and G.
        assert_eq!(AR_DATA_FLAT, 0xc093);
    }

    #[test]
    fn busy_tss_access_rights_have_no_s_bit() {
        assert_eq!(AR_TSS, 0x8b);
        assert_eq!(AR_TSS & AR_S, 0);
    }

    #[test]
    fn unusable_is_bit_16() {
        assert_eq!(AR_UNUSABLE, 0x1_0000);
    }

    // Control bits and exit reasons.

    #[test]
    fn control_bits_are_at_their_sdm_positions() {
        assert_eq!(PIN_EXTERNAL_INTERRUPT_EXITING, 1 << 0);
        assert_eq!(PROC_INTERRUPT_WINDOW_EXITING, 1 << 2);
        assert_eq!(PROC_HLT_EXITING, 1 << 7);
        assert_eq!(PROC_UNCONDITIONAL_IO_EXITING, 1 << 24);
        assert_eq!(PROC_USE_MSR_BITMAPS, 1 << 28);
        assert_eq!(PROC_ACTIVATE_SECONDARY_CONTROLS, 1 << 31);
        assert_eq!(PROC2_ENABLE_EPT, 1 << 1);
        assert_eq!(PROC2_UNRESTRICTED_GUEST, 1 << 7);
        assert_eq!(EXIT_SAVE_DEBUG_CONTROLS, 1 << 2);
        assert_eq!(EXIT_HOST_ADDRESS_SPACE_SIZE, 1 << 9);
        assert_eq!(EXIT_SAVE_IA32_PAT, 1 << 18);
        assert_eq!(EXIT_LOAD_IA32_PAT, 1 << 19);
        assert_eq!(EXIT_SAVE_IA32_EFER, 1 << 20);
        assert_eq!(EXIT_LOAD_IA32_EFER, 1 << 21);
        assert_eq!(ENTRY_LOAD_DEBUG_CONTROLS, 1 << 2);
        assert_eq!(ENTRY_IA32E_MODE_GUEST, 1 << 9);
        assert_eq!(ENTRY_LOAD_IA32_PAT, 1 << 14);
        assert_eq!(ENTRY_LOAD_IA32_EFER, 1 << 15);
    }

    #[test]
    fn basic_exit_reasons_are_the_sdm_numbers() {
        assert_eq!(REASON_EXTERNAL_INTERRUPT, 1);
        assert_eq!(REASON_TRIPLE_FAULT, 2);
        assert_eq!(REASON_INTERRUPT_WINDOW, 7);
        assert_eq!(REASON_CPUID, 10);
        assert_eq!(REASON_HLT, 12);
        assert_eq!(REASON_INVD, 13);
        assert_eq!(REASON_VMCALL, 18);
        assert_eq!(REASON_VMXON, 27);
        assert_eq!(REASON_IO_INSTRUCTION, 30);
        assert_eq!(REASON_RDMSR, 31);
        assert_eq!(REASON_WRMSR, 32);
        assert_eq!(REASON_EPT_VIOLATION, 48);
        assert_eq!(REASON_XSETBV, 55);
        assert_eq!(EXIT_REASON_ENTRY_FAILURE, 0x8000_0000);
    }

    // Exit decoding.

    fn decode(reason: u64, qualification: u64) -> Result<Exit, StringIo> {
        decode_exit(reason, qualification, || 0)
    }

    #[test]
    fn simple_reasons_decode_to_their_exits() {
        assert_eq!(
            decode(REASON_EXTERNAL_INTERRUPT, 0),
            Ok(Exit::HostInterrupt)
        );
        assert_eq!(decode(REASON_INTERRUPT_WINDOW, 0), Ok(Exit::IntWindow));
        assert_eq!(decode(REASON_CPUID, 0), Ok(Exit::Cpuid));
        assert_eq!(decode(REASON_HLT, 0), Ok(Exit::Hlt));
        assert_eq!(decode(REASON_INVD, 0), Ok(Exit::Skip));
        assert_eq!(decode(REASON_RDMSR, 0), Ok(Exit::Rdmsr));
        assert_eq!(decode(REASON_WRMSR, 0), Ok(Exit::Wrmsr));
        assert_eq!(decode(REASON_TRIPLE_FAULT, 0), Ok(Exit::TripleFault));
    }

    #[test]
    fn vmx_instructions_and_xsetbv_raise_invalid_opcode() {
        // VMCALL, VMCLEAR, VMLAUNCH, VMPTRLD, VMPTRST, VMREAD, VMRESUME, VMWRITE, VMXOFF, VMXON.
        for reason in 18..=27 {
            assert_eq!(decode(reason, 0), Ok(Exit::Undefined), "reason {reason}");
        }
        assert_eq!(decode(REASON_XSETBV, 0), Ok(Exit::Undefined));
    }

    #[test]
    fn reasons_next_to_the_undefined_ones_are_not_swallowed() {
        assert_eq!(decode(17, 0), Ok(Exit::Other(17)));
        assert_eq!(decode(28, 0), Ok(Exit::Other(28)));
        assert_eq!(decode(54, 0), Ok(Exit::Other(54)));
        assert_eq!(decode(56, 0), Ok(Exit::Other(56)));
    }

    #[test]
    fn unhandled_reasons_are_reported_with_their_number() {
        // Exception or NMI, task switch, CR access, MOV DR, MWAIT, MONITOR, PAUSE, APIC access.
        for reason in [0, 9, 28, 29, 36, 39, 40, 44, 49] {
            assert_eq!(
                decode(reason, 0),
                Ok(Exit::Other(reason)),
                "reason {reason}"
            );
        }
    }

    #[test]
    fn io_exit_qualification_gives_size_direction_and_port() {
        // OUT DX, AL to port 0x3f8: size field 0 (1 byte), direction 0.
        assert_eq!(
            decode(REASON_IO_INSTRUCTION, 0x3f8 << 16),
            Ok(Exit::Io {
                port: 0x3f8,
                size: 1,
                write: true
            })
        );
        // IN AX, DX from port 0x60: size 1 (2 bytes), direction IN.
        assert_eq!(
            decode(REASON_IO_INSTRUCTION, (0x60 << 16) | 0b1001),
            Ok(Exit::Io {
                port: 0x60,
                size: 2,
                write: false
            })
        );
        // OUT DX, EAX: size 3 (4 bytes).
        assert_eq!(
            decode(REASON_IO_INSTRUCTION, (0xcf8 << 16) | 0b011),
            Ok(Exit::Io {
                port: 0xcf8,
                size: 4,
                write: true
            })
        );
    }

    #[test]
    fn io_exit_uses_the_whole_port_field() {
        let Ok(Exit::Io { port, .. }) = decode(REASON_IO_INSTRUCTION, 0xffff << 16) else {
            panic!("not an I/O exit");
        };
        assert_eq!(port, 0xffff);
    }

    #[test]
    fn io_exit_ignores_qualification_bits_above_the_port() {
        let qualification = (0x80 << 16) | (0xabcd << 32);
        assert_eq!(
            decode(REASON_IO_INSTRUCTION, qualification),
            Ok(Exit::Io {
                port: 0x80,
                size: 1,
                write: true
            })
        );
    }

    #[test]
    fn string_io_is_refused_with_its_port() {
        // INSB from port 0x1f0: direction IN and the string bit.
        let qualification = (0x1f0 << 16) | (1 << 4) | (1 << 3);
        assert_eq!(
            decode(REASON_IO_INSTRUCTION, qualification),
            Err(StringIo { port: 0x1f0 })
        );
    }

    #[test]
    fn ept_violation_reports_the_address_and_whether_it_was_a_write() {
        assert_eq!(
            decode_exit(REASON_EPT_VIOLATION, 0b10, || 0xfee0_0000),
            Ok(Exit::Fault {
                gpa: 0xfee0_0000,
                write: true
            })
        );
        assert_eq!(
            decode_exit(REASON_EPT_VIOLATION, 0b01, || 0x1000),
            Ok(Exit::Fault {
                gpa: 0x1000,
                write: false
            })
        );
    }

    #[test]
    fn the_guest_physical_address_is_only_read_for_ept_violations() {
        let reads = Cell::new(0);
        for reason in [REASON_CPUID, REASON_IO_INSTRUCTION, REASON_HLT, 99] {
            let _ = decode_exit(reason, 0, || {
                reads.set(reads.get() + 1);
                0
            });
        }
        assert_eq!(reads.get(), 0);
        let _ = decode_exit(REASON_EPT_VIOLATION, 0, || {
            reads.set(reads.get() + 1);
            0
        });
        assert_eq!(reads.get(), 1);
    }

    // Event re-delivery.

    const VECTORING_EXTERNAL_INTERRUPT_0X20: u32 = arch::EVENT_VALID | 0x20;

    #[test]
    fn an_event_that_was_not_being_delivered_is_not_redelivered() {
        assert_eq!(event_to_redeliver(REASON_CPUID, 0x20, || 0), None);
        assert_eq!(event_to_redeliver(REASON_CPUID, 0, || 0), None);
    }

    #[test]
    fn an_interrupted_delivery_is_redelivered_without_the_valid_bit() {
        assert_eq!(
            event_to_redeliver(REASON_CPUID, VECTORING_EXTERNAL_INTERRUPT_0X20, || {
                panic!("no error code expected")
            }),
            Some((0x20, None))
        );
    }

    #[test]
    fn a_redelivered_exception_keeps_its_error_code() {
        let info = arch::EVENT_VALID | arch::exception_event(arch::VECTOR_GENERAL_PROTECTION, true);
        assert_eq!(
            event_to_redeliver(REASON_HLT, info, || 0x18),
            Some((
                arch::exception_event(arch::VECTOR_GENERAL_PROTECTION, true),
                Some(0x18)
            ))
        );
    }

    #[test]
    fn the_error_code_is_only_read_when_the_event_has_one() {
        let reads = Cell::new(0);
        let _ = event_to_redeliver(REASON_CPUID, VECTORING_EXTERNAL_INTERRUPT_0X20, || {
            reads.set(reads.get() + 1);
            0
        });
        assert_eq!(reads.get(), 0);
    }

    #[test]
    fn an_ept_violation_does_not_redeliver() {
        assert_eq!(
            event_to_redeliver(
                REASON_EPT_VIOLATION,
                VECTORING_EXTERNAL_INTERRUPT_0X20,
                || 0
            ),
            None
        );
    }

    #[test]
    fn only_software_event_types_carry_an_instruction_length() {
        let info = |event_type: u32| arch::EVENT_VALID | (event_type << arch::EVENT_TYPE_SHIFT) | 3;
        let expected = [
            (0, false), // external interrupt
            (1, false), // reserved
            (2, false), // NMI
            (3, false), // hardware exception
            (4, true),  // software interrupt
            (5, true),  // privileged software exception
            (6, true),  // software exception
            (7, false), // other event
        ];
        for (event_type, carries) in expected {
            assert_eq!(
                carries_instruction_length(info(event_type)),
                carries,
                "type {event_type}"
            );
        }
    }

    // Capability checks.

    const ALLOWED_1_SHIFT: u32 = 32;

    /// A host that passes every check in `usable_with`.
    fn capable_host() -> Vec<(u32, u64)> {
        vec![
            (
                arch::MSR_FEATURE_CONTROL,
                FEATURE_LOCK | FEATURE_VMXON_OUTSIDE_SMX,
            ),
            (MSR_VMX_BASIC, VMX_BASIC_TRUE_CONTROLS),
            (
                MSR_VMX_PROCBASED,
                (PROC_ACTIVATE_SECONDARY_CONTROLS as u64) << ALLOWED_1_SHIFT,
            ),
            (
                MSR_VMX_PROCBASED2,
                ((PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST) as u64) << ALLOWED_1_SHIFT,
            ),
            (
                MSR_VMX_EPT_VPID_CAP,
                EPT_CAP_PAGE_WALK_4 | EPT_CAP_WRITE_BACK | EPT_CAP_2MIB_PAGES,
            ),
        ]
    }

    fn set(host: &mut [(u32, u64)], msr: u32, value: u64) {
        host.iter_mut().find(|(m, _)| *m == msr).unwrap().1 = value;
    }

    fn usable_on(host: &[(u32, u64)]) -> bool {
        usable_with(|msr| host.iter().find(|(m, _)| *m == msr).map_or(0, |(_, v)| *v))
    }

    #[test]
    fn a_cpu_with_ept_unrestricted_guest_and_true_controls_is_usable() {
        assert!(usable_on(&capable_host()));
    }

    #[test]
    fn an_unlocked_feature_control_msr_is_usable_as_the_driver_locks_it() {
        let mut host = capable_host();
        set(&mut host, arch::MSR_FEATURE_CONTROL, 0);
        assert!(usable_on(&host));
    }

    #[test]
    fn firmware_that_locked_vmx_off_makes_it_unusable() {
        let mut host = capable_host();
        set(&mut host, arch::MSR_FEATURE_CONTROL, FEATURE_LOCK);
        assert!(!usable_on(&host));
    }

    #[test]
    fn locked_with_vmx_enabled_only_inside_smx_is_unusable() {
        let mut host = capable_host();
        set(
            &mut host,
            arch::MSR_FEATURE_CONTROL,
            FEATURE_LOCK | (1 << 1),
        );
        assert!(!usable_on(&host));
    }

    #[test]
    fn missing_true_control_msrs_make_it_unusable() {
        let mut host = capable_host();
        set(&mut host, MSR_VMX_BASIC, 0);
        assert!(!usable_on(&host));
    }

    #[test]
    fn missing_secondary_controls_make_it_unusable() {
        let mut host = capable_host();
        set(&mut host, MSR_VMX_PROCBASED, 0);
        assert!(!usable_on(&host));
    }

    #[test]
    fn it_needs_the_may_be_one_half_not_the_must_be_one_half() {
        let mut host = capable_host();
        // The bit in the low (must-be-one) half does not mean it is allowed.
        set(
            &mut host,
            MSR_VMX_PROCBASED,
            PROC_ACTIVATE_SECONDARY_CONTROLS as u64,
        );
        assert!(!usable_on(&host));
    }

    #[test]
    fn ept_and_unrestricted_guest_are_each_required() {
        for missing in [PROC2_ENABLE_EPT, PROC2_UNRESTRICTED_GUEST] {
            let mut host = capable_host();
            let both = PROC2_ENABLE_EPT | PROC2_UNRESTRICTED_GUEST;
            set(
                &mut host,
                MSR_VMX_PROCBASED2,
                ((both & !missing) as u64) << ALLOWED_1_SHIFT,
            );
            assert!(!usable_on(&host), "without {missing:#x}");
        }
    }

    #[test]
    fn each_ept_capability_is_required() {
        for missing in [EPT_CAP_PAGE_WALK_4, EPT_CAP_WRITE_BACK, EPT_CAP_2MIB_PAGES] {
            let mut host = capable_host();
            let all = EPT_CAP_PAGE_WALK_4 | EPT_CAP_WRITE_BACK | EPT_CAP_2MIB_PAGES;
            set(&mut host, MSR_VMX_EPT_VPID_CAP, all & !missing);
            assert!(!usable_on(&host), "without {missing:#x}");
        }
    }

    #[test]
    fn msrs_are_not_read_once_an_earlier_check_has_failed() {
        // A missing MSR raises #GP, so the later ones may only be read when the checks that say
        // they exist have passed.
        let mut host = capable_host();
        set(&mut host, arch::MSR_FEATURE_CONTROL, FEATURE_LOCK);
        let read = RefCell::new(Vec::new());
        let result = usable_with(|msr| {
            read.borrow_mut().push(msr);
            host.iter().find(|(m, _)| *m == msr).map_or(0, |(_, v)| *v)
        });
        assert!(!result);
        assert_eq!(*read.borrow(), [arch::MSR_FEATURE_CONTROL]);

        set(
            &mut host,
            arch::MSR_FEATURE_CONTROL,
            FEATURE_LOCK | FEATURE_VMXON_OUTSIDE_SMX,
        );
        set(&mut host, MSR_VMX_BASIC, 0);
        read.borrow_mut().clear();
        let result = usable_with(|msr| {
            read.borrow_mut().push(msr);
            host.iter().find(|(m, _)| *m == msr).map_or(0, |(_, v)| *v)
        });
        assert!(!result);
        assert_eq!(*read.borrow(), [arch::MSR_FEATURE_CONTROL, MSR_VMX_BASIC]);
    }

    #[test]
    fn capability_msr_numbers_are_the_sdm_ones() {
        assert_eq!(MSR_VMX_BASIC, 0x480);
        assert_eq!(MSR_VMX_PINBASED, 0x481);
        assert_eq!(MSR_VMX_PROCBASED, 0x482);
        assert_eq!(MSR_VMX_EXIT, 0x483);
        assert_eq!(MSR_VMX_ENTRY, 0x484);
        assert_eq!(MSR_VMX_CR0_FIXED0, 0x486);
        assert_eq!(MSR_VMX_CR0_FIXED1, 0x487);
        assert_eq!(MSR_VMX_CR4_FIXED0, 0x488);
        assert_eq!(MSR_VMX_CR4_FIXED1, 0x489);
        assert_eq!(MSR_VMX_PROCBASED2, 0x48b);
        assert_eq!(MSR_VMX_EPT_VPID_CAP, 0x48c);
        assert_eq!(MSR_VMX_TRUE_PINBASED, 0x48d);
        assert_eq!(MSR_VMX_TRUE_PROCBASED, 0x48e);
        assert_eq!(MSR_VMX_TRUE_EXIT, 0x48f);
        assert_eq!(MSR_VMX_TRUE_ENTRY, 0x490);
    }

    #[test]
    fn adjust_sets_the_bits_the_cpu_insists_on() {
        // Must-be-one 0x16 in the low half, everything allowed in the high half.
        let capability = (u32::MAX as u64) << 32 | 0x16;
        assert_eq!(
            adjust_with(MSR_VMX_TRUE_ENTRY, capability, 0x100),
            Ok(0x116)
        );
    }

    #[test]
    fn adjust_keeps_wanted_bits_that_are_already_forced() {
        let capability = (u32::MAX as u64) << 32 | 0x16;
        assert_eq!(adjust_with(MSR_VMX_TRUE_ENTRY, capability, 0x6), Ok(0x16));
    }

    #[test]
    fn adjust_rejects_a_bit_the_cpu_cannot_set() {
        let capability = 0x0000_00ff_0000_0000u64;
        let error = adjust_with(MSR_VMX_PROCBASED, capability, 0x1_01).unwrap_err();
        assert!(error.contains("0x482"), "{error}");
        assert!(error.contains("0x100"), "{error}");
    }

    #[test]
    fn adjust_with_nothing_wanted_returns_the_forced_bits() {
        assert_eq!(adjust_with(0, 0xff, 0), Ok(0xff));
        assert_eq!(adjust_with(0, (0xffu64 << 32) | 0x5, 0), Ok(0x5));
        assert_eq!(adjust_with(0, 0xff00_0000_0000_0000, 0), Ok(0));
    }

    #[test]
    fn fixed_bits_force_ones_and_clear_zeros() {
        // Bits 0 and 4 forced on; only bits 0-7 may be set.
        assert_eq!(apply_fixed_bits(0, 0x11, 0xff), 0x11);
        assert_eq!(apply_fixed_bits(0xffff, 0x11, 0xff), 0xff);
        assert_eq!(apply_fixed_bits(0x2e, 0x11, 0xff), 0x3f);
    }

    #[test]
    fn fixed_bits_of_a_typical_cr4_keep_pae_and_force_vmxe() {
        let fixed0 = CR4_VMXE;
        let fixed1 = 0x0037_27ff;
        let cr4 = apply_fixed_bits(arch::CR4_PAE, fixed0, fixed1);
        assert_eq!(cr4, arch::CR4_PAE | CR4_VMXE);
    }

    #[test]
    fn ept_pointer_is_write_back_with_a_four_level_walk() {
        assert_eq!(ept_pointer(0x1000), 0x1000 | 6 | (3 << 3));
        assert_eq!(ept_pointer(0x1000), 0x101e);
    }

    #[test]
    fn ept_pointer_keeps_a_high_root_address() {
        let root = 0x0000_1234_5678_9000;
        assert_eq!(ept_pointer(root) & !0xfff, root);
    }

    // Host task register: the GDT copy.

    #[test]
    fn a_small_gdt_fits_one_page_and_the_descriptor_follows_it() {
        // 0x7f is a limit of 128 bytes, eight descriptors.
        assert_eq!(
            gdt_layout(0x7f),
            Ok(GdtLayout {
                descriptor_offset: 0x80,
                pages: 1,
                limit: 0x8f
            })
        );
    }

    #[test]
    fn the_descriptor_offset_is_aligned_to_sixteen_bytes() {
        // 40 bytes of GDT (limit 0x27, five descriptors) pads to 48.
        let layout = gdt_layout(0x27).unwrap();
        assert_eq!(layout.descriptor_offset, 48);
        assert_eq!(layout.descriptor_offset % SYSTEM_DESCRIPTOR_SIZE, 0);
        assert_eq!(layout.limit, 63);
    }

    #[test]
    fn a_one_byte_gdt_is_padded_to_the_first_slot_after_it() {
        let layout = gdt_layout(0).unwrap();
        assert_eq!(layout.descriptor_offset, 16);
        assert_eq!(layout.limit, 31);
    }

    #[test]
    fn a_gdt_that_fills_a_page_needs_a_second_one_for_the_descriptor() {
        // Limit 0xfff is exactly 4096 bytes; the descriptor lies entirely in the next page. One
        // page, as the copy used to be given, would have been overrun.
        let layout = gdt_layout(0xfff).unwrap();
        assert_eq!(layout.descriptor_offset, PAGE_SIZE);
        assert_eq!(layout.pages, 2);
        assert_eq!(layout.limit, 0x100f);
    }

    #[test]
    fn a_gdt_whose_descriptor_straddles_a_page_boundary_gets_both_pages() {
        // 4090 bytes pad to 4096 for the descriptor, but 4080 bytes put it at 4080..4096.
        assert_eq!(gdt_layout(4080 - 1).unwrap().pages, 1);
        assert_eq!(gdt_layout(4081 - 1).unwrap().pages, 2);
    }

    #[test]
    fn the_largest_gdt_that_leaves_room_still_works() {
        // 65520 bytes (limit 0xffef) plus the descriptor is exactly 64 KiB.
        assert_eq!(
            gdt_layout(0xffef),
            Ok(GdtLayout {
                descriptor_offset: 0xfff0,
                pages: 16,
                limit: 0xffff
            })
        );
    }

    #[test]
    fn a_gdt_with_no_room_for_a_descriptor_is_refused() {
        // Limit 0xffff is the full 64 KiB, and 0xfff0..=0xffee pad past the end.
        for limit in [0xffff, 0xfff0, 0xffef + 1] {
            let error = gdt_layout(limit).unwrap_err();
            assert!(error.contains("no room"), "limit {limit:#x}: {error}");
        }
    }

    #[test]
    fn every_accepted_layout_has_its_descriptor_inside_the_allocation_and_the_limit() {
        for limit in (0..=u16::MAX)
            .step_by(7)
            .chain([0xffef, 0xfff0, 0x0fff, 0x1000])
        {
            let Ok(layout) = gdt_layout(limit) else {
                continue;
            };
            let end = layout.descriptor_offset + SYSTEM_DESCRIPTOR_SIZE;
            assert!(end <= layout.pages * PAGE_SIZE, "limit {limit:#x}");
            assert_eq!(layout.limit as usize + 1, end, "limit {limit:#x}");
            assert!(
                layout.descriptor_offset > limit as usize,
                "limit {limit:#x}"
            );
            assert!(layout.descriptor_offset <= u16::MAX as usize);
        }
    }

    // TSS descriptor.

    #[test]
    fn tss_descriptor_of_base_zero_is_limit_and_access_byte() {
        assert_eq!(tss_descriptor(0), [0x0000_8900_0000_0067, 0]);
    }

    #[test]
    fn tss_descriptor_scatters_the_base_as_the_sdm_describes() {
        // Base bytes, low to high: f0 de bc 9a 78 56 34 12. The descriptor bytes are limit
        // 67 00, base 15:0 (f0 de), base 23:16 (bc), access 89, flags 00, base 31:24 (9a).
        let [low, high] = tss_descriptor(0x1234_5678_9abc_def0);
        assert_eq!(low, 0x9a00_89bc_def0_0067);
        assert_eq!(high, 0x1234_5678);
    }

    #[test]
    fn tss_descriptor_limit_is_a_104_byte_tss_and_type_is_available_64_bit() {
        assert_eq!(TSS_LIMIT + 1, 104);
        assert_eq!(TSS_ACCESS & 0xf, 9, "type: available 64-bit TSS");
        assert_ne!(TSS_ACCESS & 0x80, 0, "present");
        assert_eq!(TSS_ACCESS & 0x10, 0, "system segment");
    }

    #[test]
    fn the_base_survives_a_round_trip_through_the_descriptor() {
        for base in [
            0,
            0x1000,
            0x0000_0000_dead_b000,
            0x0000_0001_0000_0000,
            0x0000_7fff_ffff_f000,
            0xffff_ffff_ffff_ffff,
            0xaaaa_5555_aaaa_5555,
            0x0000_0000_00ff_0000,
            0x0000_0000_ff00_0000,
        ] {
            let [low, high] = tss_descriptor(base);
            assert_eq!(tss_base_from_descriptor(low, high), base, "base {base:#x}");
        }
    }

    #[test]
    fn reserved_upper_bits_of_the_high_qword_do_not_leak_into_the_base() {
        let [low, high] = tss_descriptor(0x0000_0001_2345_6000);
        assert_eq!(
            tss_base_from_descriptor(low, high | (0xffff_ffff << 32)),
            0x0000_0001_2345_6000
        );
    }

    #[test]
    fn limit_and_flag_bits_of_the_low_qword_do_not_leak_into_the_base() {
        let [low, high] = tss_descriptor(0x4000);
        let noisy = low | (0xf << 48) | (0xf << 52);
        assert_eq!(tss_base_from_descriptor(noisy, high), 0x4000);
    }

    #[test]
    fn selectors_drop_requested_privilege_and_table_indicator() {
        assert_eq!(SELECTOR_RPL_AND_TABLE_MASK, 0b111);
        assert_eq!(0x0e3u16 & !SELECTOR_RPL_AND_TABLE_MASK, 0x0e0);
    }

    // MSR bitmap.

    const BITMAP_BYTES: usize = 4096;

    fn full_bitmap() -> Vec<u8> {
        vec![0xff; BITMAP_BYTES]
    }

    fn cleared_bits(bitmap: &[u8]) -> Vec<(usize, u8)> {
        let mut out = Vec::new();
        for (byte, value) in bitmap.iter().enumerate() {
            for bit in 0..8 {
                if value & (1 << bit) == 0 {
                    out.push((byte, bit));
                }
            }
        }
        out
    }

    #[test]
    fn a_low_msr_clears_its_read_and_write_bits() {
        let mut bitmap = full_bitmap();
        // MSR 0x10: bit 0 of byte 2 in the read bitmap and the write bitmap 2048 bytes on.
        pass_through(&mut bitmap, arch::MSR_TSC);
        assert_eq!(cleared_bits(&bitmap), [(2, 0), (2048 + 2, 0)]);
    }

    #[test]
    fn a_low_msr_in_the_middle_of_a_byte_clears_the_right_bit() {
        let mut bitmap = full_bitmap();
        // MSR 0x277 (PAT): byte 0x4e, bit 7.
        pass_through(&mut bitmap, arch::MSR_PAT);
        assert_eq!(cleared_bits(&bitmap), [(0x4e, 7), (2048 + 0x4e, 7)]);
    }

    #[test]
    fn a_high_msr_uses_the_second_kilobyte_of_each_bitmap() {
        let mut bitmap = full_bitmap();
        // MSR 0xc0000081 is index 0x81 of the high range: byte 16, bit 1.
        pass_through(&mut bitmap, arch::MSR_STAR);
        assert_eq!(cleared_bits(&bitmap), [(1024 + 16, 1), (3072 + 16, 1)]);
    }

    #[test]
    fn the_last_msr_of_each_range_is_the_last_bit_of_its_kilobyte() {
        let mut bitmap = full_bitmap();
        pass_through(&mut bitmap, arch::MSR_LOW_RANGE_END);
        pass_through(&mut bitmap, arch::MSR_HIGH_RANGE_END);
        assert_eq!(
            cleared_bits(&bitmap),
            [
                (1023, 7),
                (1024 + 1023, 7),
                (2048 + 1023, 7),
                (3072 + 1023, 7)
            ]
        );
    }

    #[test]
    fn msrs_outside_both_ranges_leave_the_bitmap_alone() {
        for msr in [
            arch::MSR_LOW_RANGE_END + 1,
            arch::MSR_HIGH_RANGE_START - 1,
            arch::MSR_HIGH_RANGE_END + 1,
            0x4000_0000,
            0xc001_0000,
            u32::MAX,
        ] {
            let mut bitmap = full_bitmap();
            pass_through(&mut bitmap, msr);
            assert_eq!(bitmap, full_bitmap(), "msr {msr:#x}");
        }
    }

    #[test]
    fn passing_through_the_same_msr_twice_changes_nothing_more() {
        let mut once = full_bitmap();
        pass_through(&mut once, arch::MSR_LSTAR);
        let mut twice = once.clone();
        pass_through(&mut twice, arch::MSR_LSTAR);
        assert_eq!(once, twice);
    }

    #[test]
    fn all_pass_through_msrs_clear_two_bits_each_and_efer_still_exits() {
        let mut bitmap = full_bitmap();
        for msr in arch::PASS_THROUGH_MSRS {
            pass_through(&mut bitmap, msr);
        }
        assert_eq!(
            cleared_bits(&bitmap).len(),
            2 * arch::PASS_THROUGH_MSRS.len()
        );
        // EFER is index 0x80 of the high range: byte 16, bit 0.
        assert_eq!(bitmap[1024 + 16] & 1, 1, "EFER reads must exit");
        assert_eq!(bitmap[3072 + 16] & 1, 1, "EFER writes must exit");
    }

    // MSR lists and consistency.

    #[test]
    fn switched_msrs_are_passed_through_so_the_lists_do_the_switching() {
        for msr in SWITCHED_MSRS {
            assert!(arch::PASS_THROUGH_MSRS.contains(&msr), "{msr:#x}");
        }
    }

    #[test]
    fn switched_msrs_are_distinct_and_fit_one_list_page() {
        for (i, a) in SWITCHED_MSRS.iter().enumerate() {
            assert!(!SWITCHED_MSRS[i + 1..].contains(a));
        }
        let entry_bytes = MSR_LIST_ENTRY_U64S * 8;
        assert_eq!(entry_bytes, 16);
        assert!(SWITCHED_MSRS.len() * entry_bytes <= PAGE_SIZE);
        assert!(SWITCHED_MSRS.len() <= 512, "recommended maximum list size");
    }

    #[test]
    fn the_list_pages_hold_a_guest_page_then_a_host_page() {
        assert_eq!(MSR_LIST_PAGES, 2);
    }

    #[test]
    fn interruptibility_mask_covers_sti_and_mov_ss_blocking() {
        assert_eq!(INTERRUPTIBILITY_BLOCKING_BY_STI_AND_MOV_SS, 0b11);
    }

    #[test]
    fn exit_reason_masks_split_the_basic_reason_from_the_failure_flag() {
        let exit_reason = EXIT_REASON_ENTRY_FAILURE | 33;
        assert_eq!(exit_reason & EXIT_REASON_BASIC_MASK, 33);
        assert_ne!(exit_reason & EXIT_REASON_ENTRY_FAILURE, 0);
    }
}
