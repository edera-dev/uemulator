use super::flags::*;
use super::{
    Cpu, EFER_LMA, ExecResult, GPR_RSP, SEG_CS, SEG_DEFAULT_BIG, SEG_DESCRIPTOR_FLAGS_MASK,
    SEG_DPL_SHIFT, SEG_FLAGS_FLAT_DATA, SEG_GRANULARITY, SEG_LIMIT_4GIB, SEG_SS, Seg, USER_CPL,
};
use crate::machine::ExitReason;

pub(crate) const VECTOR_DIVIDE_ERROR: u8 = 0;
pub(crate) const VECTOR_INVALID_OPCODE: u8 = 6;
pub(crate) const VECTOR_DEVICE_NOT_AVAILABLE: u8 = 7;
pub(crate) const VECTOR_DOUBLE_FAULT: u8 = 8;
pub(crate) const VECTOR_INVALID_TSS: u8 = 10;
pub(crate) const VECTOR_SEGMENT_NOT_PRESENT: u8 = 11;
pub(crate) const VECTOR_STACK_FAULT: u8 = 12;
pub(crate) const VECTOR_GENERAL_PROTECTION: u8 = 13;
pub(crate) const VECTOR_PAGE_FAULT: u8 = 14;
pub(crate) const VECTOR_MACHINE_CHECK: u8 = 18;

/// Long-mode IDT entries are 16 bytes.
const IDT_ENTRY_BYTES: u64 = 16;
/// Error codes for faults caused by an IDT entry are `vector * 8` with this bit (IDT) set.
const IDT_ERROR_CODE_FLAG: u32 = 2;
/// Clears the RPL bits of a selector, leaving index and table indicator, which is the form that
/// segment-related error codes use.
const SELECTOR_ERROR_CODE_MASK: u16 = 0xfffc;
const SELECTOR_RPL_MASK: u16 = 3;
/// Code segment type bit "conforming" (bit 10 of the descriptor's upper dword).
const SEG_TYPE_CONFORMING: u32 = 1 << 10;
/// Selector bits 3-15 pre-multiplied by 8: the byte offset of the descriptor in its table.
const SELECTOR_OFFSET_MASK: u16 = 0xfff8;
/// Selector table indicator: set selects the LDT, clear the GDT.
const SELECTOR_TI_LDT: u16 = 1 << 2;

const GATE_PRESENT: u8 = 1 << 7;
const GATE_DPL_SHIFT: u32 = 5;
const GATE_TYPE_MASK: u8 = 0xf;
const GATE_TYPE_INTERRUPT: u8 = 0xe;
const GATE_TYPE_TRAP: u8 = 0xf;
const GATE_IST_MASK: u64 = 7;
/// The S bit of a descriptor (bit 12 of its upper dword) clear means a system descriptor,
/// which is 16 bytes wide in long mode.
const DESCRIPTOR_S_BIT: u64 = 1 << 44;

/// Offsets into the 64-bit TSS: RSP0 (then RSP1, RSP2) and IST1 (then IST2..7), 8 bytes each.
const TSS64_RSP0_OFFSET: u64 = 4;
const TSS64_IST1_OFFSET: u64 = 36;
const TSS64_SLOT_BYTES: u64 = 8;
/// Interrupt delivery in 64-bit mode aligns the new stack pointer down to 16 bytes.
const STACK_ALIGN_MASK: u64 = !0xf;
/// Upper bound on delivery attempts, so a cycle of benign faults raised by delivery itself
/// ends as a triple fault instead of looping.
const MAX_DELIVERY_ATTEMPTS: usize = 4;

/// Why an instruction or memory access stopped executing normally.
#[derive(Debug)]
pub(crate) enum Exception {
    /// A guest-visible exception, delivered through the IDT.
    Fault {
        vector: u8,
        /// Error code to push; `None` for vectors that have none.
        error_code: Option<u32>,
        /// Raised by `int n`/`int3` rather than by a fault, so delivery also checks the gate's
        /// DPL against CPL.
        software: bool,
    },
    /// Not a guest exception: ends the run loop with this reason.
    Exit(ExitReason),
    /// An SVM #VMEXIT already switched the CPU back to the host: the instruction that raised it is
    /// fully accounted for and nothing is delivered.
    Vmexit,
    /// A locked read-modify-write lost its compare-and-swap to another vCPU, or its operand needs
    /// the other vCPUs stopped. `Cpu::exec` catches it, puts the registers back and runs the
    /// instruction again; it never reaches the run loop.
    AtomicRetry,
}

impl Exception {
    pub fn fault(vector: u8, error_code: Option<u32>) -> Exception {
        Exception::Fault {
            vector,
            error_code,
            software: false,
        }
    }
    pub fn stack_fault(code: u32) -> Exception {
        Exception::fault(VECTOR_STACK_FAULT, Some(code))
    }
    pub fn general_protection(code: u32) -> Exception {
        Exception::fault(VECTOR_GENERAL_PROTECTION, Some(code))
    }
    pub fn invalid_opcode() -> Exception {
        Exception::fault(VECTOR_INVALID_OPCODE, None)
    }
    pub fn divide_error() -> Exception {
        Exception::fault(VECTOR_DIVIDE_ERROR, None)
    }
    pub fn device_not_available() -> Exception {
        Exception::fault(VECTOR_DEVICE_NOT_AVAILABLE, None)
    }
}

/// Classes of exceptions for double-fault detection (SDM Vol. 3A, "Interrupt 8 - Double Fault
/// Exception", Table 6-5). Software interrupts and external interrupts are always benign.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FaultClass {
    Benign,
    /// #DE, #TS, #NP, #SS, #GP.
    Contributory,
    PageFault,
}

fn fault_class(vector: u8) -> FaultClass {
    match vector {
        VECTOR_DIVIDE_ERROR
        | VECTOR_INVALID_TSS
        | VECTOR_SEGMENT_NOT_PRESENT
        | VECTOR_STACK_FAULT
        | VECTOR_GENERAL_PROTECTION => FaultClass::Contributory,
        VECTOR_PAGE_FAULT => FaultClass::PageFault,
        _ => FaultClass::Benign,
    }
}

/// A second fault while delivering the first becomes #DF only for contributory-then-contributory,
/// #PF-then-contributory and #PF-then-#PF. Contributory-then-#PF is handled serially: the #PF is
/// simply delivered next.
fn escalates_to_double_fault(first: FaultClass, second: FaultClass) -> bool {
    matches!(
        (first, second),
        (FaultClass::Contributory, FaultClass::Contributory)
            | (FaultClass::PageFault, FaultClass::Contributory)
            | (FaultClass::PageFault, FaultClass::PageFault)
    )
}

fn idt_error_code(vector: u8) -> u32 {
    vector as u32 * 8 + IDT_ERROR_CODE_FLAG
}

impl Cpu {
    /// Delivers an exception (or, with `software`, an `int n`/`int3`) through the IDT. If delivery
    /// itself faults, the new fault replaces the old one, or becomes #DF per the SDM's table
    /// (`escalates_to_double_fault`); a fault while delivering #DF is a triple fault. Returns
    /// `Some` only when the run must end.
    pub fn deliver_event(
        &mut self,
        vector: u8,
        error_code: Option<u32>,
        software: bool,
    ) -> Option<ExitReason> {
        self.deliver_chain(vector, error_code, software, !software)
    }

    /// Delivers an external interrupt. Like `int n` it is not an exception, so a fault while
    /// delivering it never counts the interrupt's own vector as contributory.
    pub fn deliver_hardware_interrupt(&mut self, vector: u8) -> Option<ExitReason> {
        self.deliver_chain(vector, None, false, false)
    }

    fn deliver_chain(
        &mut self,
        vector: u8,
        error_code: Option<u32>,
        software: bool,
        first_is_exception: bool,
    ) -> Option<ExitReason> {
        let (mut vector, mut error_code, mut software) = (vector, error_code, software);
        let mut class = if first_is_exception {
            fault_class(vector)
        } else {
            FaultClass::Benign
        };
        for _ in 0..MAX_DELIVERY_ATTEMPTS {
            match self.deliver(vector, error_code, software) {
                Ok(()) => return None,
                Err(Exception::Exit(reason)) => return Some(reason),
                // a nested-paging fault while delivering switched to the host already; the event
                // being delivered is reported to the hypervisor so it can inject it again
                Err(Exception::AtomicRetry) => unreachable!("exec retries locked instructions"),
                Err(Exception::Vmexit) => {
                    let kind = if software {
                        super::svm::EVENT_TYPE_SOFTWARE
                    } else if class != FaultClass::Benign || first_is_exception {
                        super::svm::EVENT_TYPE_EXCEPTION
                    } else {
                        super::svm::EVENT_TYPE_INTERRUPT
                    };
                    self.svm_event_in_flight(vector, error_code, kind);
                    return None;
                }
                Err(Exception::Fault {
                    vector: nested_vector,
                    error_code: nested_error,
                    ..
                }) => {
                    if vector == VECTOR_DOUBLE_FAULT {
                        return Some(ExitReason::TripleFault);
                    }
                    software = false;
                    let nested_class = fault_class(nested_vector);
                    if escalates_to_double_fault(class, nested_class) {
                        vector = VECTOR_DOUBLE_FAULT;
                        error_code = Some(0);
                        class = FaultClass::Benign;
                    } else {
                        vector = nested_vector;
                        error_code = nested_error;
                        class = nested_class;
                    }
                }
            }
        }
        Some(ExitReason::TripleFault)
    }

    /// 64-bit-mode interrupt/trap gate delivery. Nothing is committed (RSP, CS, SS, RIP, flags)
    /// until after the last push, so a fault while pushing leaves the CPU state as it was and
    /// the nested fault is delivered instead.
    fn deliver(&mut self, vector: u8, error_code: Option<u32>, software: bool) -> ExecResult<()> {
        if self.efer & EFER_LMA == 0 {
            return self.deliver_legacy(vector, error_code, software);
        }
        let entry_offset = vector as u64 * IDT_ENTRY_BYTES;
        if entry_offset + (IDT_ENTRY_BYTES - 1) > self.idtr.limit as u64 {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        // The bases come straight from the guest (lidt/lgdt/ltr), so the sums may wrap.
        let entry_address = self.idtr.base.wrapping_add(entry_offset);
        let lo = self.read_sys(entry_address, 8)?;
        let hi = self.read_sys(entry_address.wrapping_add(8), 8)?;
        // lo: offset[15:0], selector, IST in bits 32-34, attributes byte, offset[31:16];
        // hi: offset[63:32].
        let selector = (lo >> 16) as u16;
        let ist = ((lo >> 32) & GATE_IST_MASK) as usize;
        let attributes = (lo >> 40) as u8;
        let gate_dpl = (attributes >> GATE_DPL_SHIFT) & 3;
        let gate_type = attributes & GATE_TYPE_MASK;
        // Same order as the SDM's INT pseudocode: type, software-interrupt DPL, then present.
        if gate_type != GATE_TYPE_INTERRUPT && gate_type != GATE_TYPE_TRAP {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        if software && gate_dpl < self.cpl {
            return Err(Exception::general_protection(idt_error_code(vector)));
        }
        if attributes & GATE_PRESENT == 0 {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(idt_error_code(vector)),
            ));
        }
        let target = (lo & 0xffff) | ((lo >> 32) & 0xffff_0000) | (hi << 32);
        if selector & SELECTOR_ERROR_CODE_MASK == 0 {
            return Err(Exception::general_protection(0));
        }
        let mut new_cs = self.load_desc_seg(selector, false)?;
        let selector_error_code = (selector & SELECTOR_ERROR_CODE_MASK) as u32;
        if !new_cs.is_code() || new_cs.dpl() > self.cpl {
            return Err(Exception::general_protection(selector_error_code));
        }
        if !new_cs.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some(selector_error_code),
            ));
        }
        // A 64-bit code segment has L set and D clear; both set is reserved.
        if !new_cs.long() || new_cs.flags & SEG_DEFAULT_BIG != 0 {
            return Err(Exception::general_protection(selector_error_code));
        }
        // A conforming handler runs at the caller's privilege level whatever its DPL; only a
        // non-conforming one moves to its DPL (which is at most the current CPL, checked above).
        let new_cpl = if new_cs.flags & SEG_TYPE_CONFORMING != 0 {
            self.cpl
        } else {
            new_cs.dpl()
        };
        // The CS selector's RPL always equals the CPL.
        new_cs.sel = (selector & !SELECTOR_RPL_MASK) | new_cpl as u16;
        let old_ss = self.seg[SEG_SS].sel as u64;
        let old_rsp = self.gpr[GPR_RSP];
        let rflags = self.eflags_full();
        let mut stack_pointer = if ist != 0 {
            self.tss_read(TSS64_IST1_OFFSET + (ist as u64 - 1) * TSS64_SLOT_BYTES)?
        } else if new_cpl < self.cpl {
            self.tss_read(TSS64_RSP0_OFFSET + new_cpl as u64 * TSS64_SLOT_BYTES)?
        } else {
            old_rsp
        };
        stack_pointer &= STACK_ALIGN_MASK;
        let old_cs = self.seg[SEG_CS].sel as u64;
        // The pushes are supervisor accesses, except when the handler itself runs in ring 3
        // (a conforming or same-level target, so the interrupted code was in ring 3 too): then
        // the stack must be user-accessible, as for any other ring 3 access.
        let user_stack = new_cpl == USER_CPL;
        let mut push = |cpu: &mut Cpu, value: u64| -> ExecResult<()> {
            stack_pointer = stack_pointer.wrapping_sub(8);
            if user_stack {
                cpu.write_stack(stack_pointer, 8, value)
            } else {
                cpu.write_sys_stack(stack_pointer, 8, value)
            }
        };
        // 64-bit mode always pushes all five words, with or without a privilege change.
        push(self, old_ss)?;
        push(self, old_rsp)?;
        push(self, rflags)?;
        push(self, old_cs)?;
        let rip = self.rip;
        push(self, rip)?;
        if let Some(error_code) = error_code {
            push(self, error_code as u64)?;
        }
        self.gpr[GPR_RSP] = stack_pointer;
        if new_cpl != self.cpl {
            // Hardware loads a null SS with RPL = new CPL; modeled as a flat data segment of
            // that DPL.
            self.seg[SEG_SS] = Seg {
                sel: new_cpl as u16,
                base: 0,
                limit: SEG_LIMIT_4GIB,
                flags: SEG_FLAGS_FLAT_DATA | ((new_cpl as u32) << SEG_DPL_SHIFT),
            };
        }
        self.seg[SEG_CS] = new_cs;
        self.set_cpl(new_cpl);
        self.rip = target;
        // Resolve lazy flags first: the masking below edits `eflags` directly.
        self.materialize_lazy_flags();
        self.eflags &= !(TF | NT | RF | VM | AC);
        // Only interrupt gates mask interrupts; trap gates leave IF alone.
        if gate_type == GATE_TYPE_INTERRUPT {
            self.eflags &= !IF;
        }
        Ok(())
    }

    /// Reads a qword from the 64-bit TSS; an offset past the TR limit is #TS.
    fn tss_read(&mut self, offset: u64) -> ExecResult<u64> {
        if offset + 7 > self.tr.limit as u64 {
            return Err(Exception::fault(
                VECTOR_INVALID_TSS,
                Some((self.tr.sel & SELECTOR_ERROR_CODE_MASK) as u32),
            ));
        }
        self.read_sys(self.tr.base.wrapping_add(offset), 8)
    }

    /// Returns the descriptor's low qword and, for system descriptors (16 bytes in long mode)
    /// whose second half is inside the table limit, the high qword (otherwise 0).
    pub fn read_desc(&mut self, selector: u16) -> ExecResult<(u64, u64)> {
        let offset = (selector & SELECTOR_OFFSET_MASK) as u64;
        let (table_base, table_limit) = if selector & SELECTOR_TI_LDT != 0 {
            (self.ldtr.base, self.ldtr.limit as u64)
        } else {
            (self.gdtr.base, self.gdtr.limit as u64)
        };
        if offset + 7 > table_limit {
            return Err(Exception::general_protection(
                (selector & SELECTOR_ERROR_CODE_MASK) as u32,
            ));
        }
        let lo = self.read_sys(table_base.wrapping_add(offset), 8)?;
        let hi = if lo & DESCRIPTOR_S_BIT == 0
            && offset + 15 <= table_limit
            && self.efer & EFER_LMA != 0
        {
            self.read_sys(table_base.wrapping_add(offset + 8), 8)?
        } else {
            0
        };
        Ok((lo, hi))
    }

    /// Builds a `Seg` from an 8-byte descriptor. Does no type or privilege checks (callers
    /// do), only the present check when asked, which raises #NP.
    pub fn load_desc_seg(&mut self, selector: u16, require_present: bool) -> ExecResult<Seg> {
        let (lo, _) = self.read_desc(selector)?;
        let dword0 = lo as u32;
        let dword1 = (lo >> 32) as u32;
        let base = ((dword0 >> 16) | ((dword1 & 0xff) << 16) | (dword1 & 0xff00_0000)) as u64;
        let mut limit = (dword0 & 0xffff) | (dword1 & 0xf_0000);
        if dword1 & SEG_GRANULARITY != 0 {
            limit = (limit << 12) | 0xfff;
        }
        let seg = Seg {
            sel: selector,
            base,
            limit,
            flags: dword1 & SEG_DESCRIPTOR_FLAGS_MASK,
        };
        if require_present && !seg.present() {
            return Err(Exception::fault(
                VECTOR_SEGMENT_NOT_PRESENT,
                Some((selector & SELECTOR_ERROR_CODE_MASK) as u32),
            ));
        }
        Ok(seg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::test_support::*;
    use crate::cpu::{TLB_ENTRIES_PER_MODE, TlbEntry};
    use crate::testkit::{HLT, Harness, STACK_TOP};

    const OLD_RIP: u64 = 0x0000_7000_1234_5678;
    const USER_RSP: u64 = 0x7f_0008;
    const RSP0: u64 = 0x70_0000;
    const RSP1: u64 = 0x71_0000;
    const RSP2: u64 = 0x72_0008;
    const IST_STACKS: [u64; 7] = [
        0x61_0000, 0x62_0008, 0x63_0000, 0x64_0000, 0x65_0000, 0x66_0000, 0x67_0000,
    ];
    /// A 2 MiB region of the boot identity map that `unmap_boot_identity_2m` makes not present.
    const HOLE: u64 = 0x180_0000;
    const HOLE_END: u64 = HOLE + TWO_MIB;
    const SUPERVISOR_ONLY_FLAGS: u64 = PTE_P | PTE_W;
    const FRAME_WORDS: u64 = 5;
    const WORD: u64 = 8;
    const FRAME_BYTES: u64 = FRAME_WORDS * WORD;
    const NOT_CANONICAL: u64 = 0x0000_8000_0000_0000;
    /// A stack pointer whose first push lands in the non-canonical range (pushes go downward, so the
    /// first non-canonical address itself would push to a canonical one).
    const NON_CANONICAL_STACK: u64 = NOT_CANONICAL + 0x1000;
    /// Every flag the CPU clears on entry to a handler, plus a few it must keep.
    const CLEARED_ON_ENTRY: u64 = TF | NT | RF | VM;

    fn base() -> Harness {
        let mut h = Harness::new(&[HLT]);
        install_idt(&mut h);
        install_tss(&mut h, [RSP0, RSP1, RSP2], IST_STACKS);
        h.cpu.rip = OLD_RIP;
        h.cpu.eflags = RESERVED_ONE | IF;
        h
    }

    fn user_base() -> Harness {
        let mut h = base();
        enter_user_mode(&mut h);
        h.cpu.gpr[GPR_RSP] = USER_RSP;
        h
    }

    /// Switches to page tables that map just what delivery touches. The user stack page is
    /// user-accessible when `user_stack_flags` says so; everything else is supervisor-only.
    fn with_user_stack_paged(h: &mut Harness, user_stack_flags: u64) {
        const SUPERVISOR: u64 = PTE_P | PTE_W;
        let mut tables = PageTables::new(&mut h.cpu);
        // The user pages go first: the shared upper-level entries are created with their flags, and
        // supervisor-only upper entries would make the stack inaccessible from ring 3 regardless.
        for (page, flags) in [
            ((USER_RSP & !0xfff) - PAGE, user_stack_flags),
            (USER_RSP & !0xfff, user_stack_flags),
            (GDT_BASE, SUPERVISOR),
            (IDT_BASE, SUPERVISOR),
            (TSS_BASE, SUPERVISOR),
            (RSP0 - PAGE, SUPERVISOR),
        ] {
            let upper = flags | PTE_U;
            tables.map(&mut h.cpu, page, page, &[upper, upper, upper, flags]);
        }
    }

    fn aligned_below(stack: u64, bytes: u64) -> u64 {
        (stack & !0xf) - bytes
    }

    #[derive(Debug, PartialEq, Eq)]
    struct State {
        rip: u64,
        rsp: u64,
        cs: u16,
        ss: u16,
        cpl: u8,
        eflags: u64,
    }

    fn state(h: &Harness) -> State {
        State {
            rip: h.cpu.rip,
            rsp: h.cpu.gpr[GPR_RSP],
            cs: h.cpu.seg[SEG_CS].sel,
            ss: h.cpu.seg[SEG_SS].sel,
            cpl: h.cpu.cpl,
            eflags: h.cpu.eflags_full(),
        }
    }

    mod constructors_and_codes {
        use super::*;

        #[test]
        fn vector_constants_are_the_architectural_numbers() {
            assert_eq!(VECTOR_DIVIDE_ERROR, 0);
            assert_eq!(VECTOR_INVALID_OPCODE, 6);
            assert_eq!(VECTOR_DEVICE_NOT_AVAILABLE, 7);
            assert_eq!(VECTOR_DOUBLE_FAULT, 8);
            assert_eq!(VECTOR_INVALID_TSS, 10);
            assert_eq!(VECTOR_SEGMENT_NOT_PRESENT, 11);
            assert_eq!(VECTOR_STACK_FAULT, 12);
            assert_eq!(VECTOR_GENERAL_PROTECTION, 13);
            assert_eq!(VECTOR_PAGE_FAULT, 14);
            assert_eq!(VECTOR_MACHINE_CHECK, 18);
        }

        #[test]
        fn constructors_build_the_expected_faults() {
            assert_eq!(
                fault_of::<()>(Err(Exception::fault(33, Some(7)))),
                (33, Some(7))
            );
            assert_eq!(fault_of::<()>(Err(Exception::fault(33, None))), (33, None));
            assert_eq!(
                fault_of::<()>(Err(Exception::stack_fault(5))),
                (12, Some(5))
            );
            assert_eq!(
                fault_of::<()>(Err(Exception::general_protection(9))),
                (13, Some(9))
            );
            assert_eq!(fault_of::<()>(Err(Exception::invalid_opcode())), (6, None));
            assert_eq!(fault_of::<()>(Err(Exception::divide_error())), (0, None));
            assert_eq!(
                fault_of::<()>(Err(Exception::device_not_available())),
                (7, None)
            );
        }

        #[test]
        fn constructed_faults_are_not_software_interrupts() {
            for exception in [
                Exception::fault(3, None),
                Exception::stack_fault(0),
                Exception::general_protection(0),
                Exception::invalid_opcode(),
                Exception::divide_error(),
                Exception::device_not_available(),
            ] {
                assert!(matches!(
                    exception,
                    Exception::Fault {
                        software: false,
                        ..
                    }
                ));
            }
        }

        #[test]
        fn idt_error_codes_are_vector_times_eight_with_the_idt_bit() {
            assert_eq!(idt_error_code(0), 2);
            assert_eq!(idt_error_code(1), 0xa);
            assert_eq!(idt_error_code(13), 106);
            assert_eq!(idt_error_code(0x80), 0x402);
            assert_eq!(idt_error_code(255), 0x7fa);
        }

        #[test]
        fn fault_classes_follow_the_sdm() {
            for vector in 0..=255u8 {
                let expected = match vector {
                    0 | 10 | 11 | 12 | 13 => FaultClass::Contributory,
                    14 => FaultClass::PageFault,
                    _ => FaultClass::Benign,
                };
                assert_eq!(fault_class(vector), expected, "vector {vector}");
            }
        }

        #[test]
        fn double_fault_table_matches_sdm_table_6_5() {
            use FaultClass::*;
            let table = [
                (Benign, Benign, false),
                (Benign, Contributory, false),
                (Benign, PageFault, false),
                (Contributory, Benign, false),
                (Contributory, Contributory, true),
                (Contributory, PageFault, false),
                (PageFault, Benign, false),
                (PageFault, Contributory, true),
                (PageFault, PageFault, true),
            ];
            for (first, second, escalates) in table {
                assert_eq!(
                    escalates_to_double_fault(first, second),
                    escalates,
                    "{first:?} then {second:?}"
                );
            }
        }
    }

    mod descriptors {
        use super::*;

        const SYSTEM_DESCRIPTOR_TYPE_TSS_AVAILABLE: u8 = 0x89;

        fn harness() -> Harness {
            let mut h = base();
            install_test_gdt(&mut h);
            h
        }

        #[test]
        fn read_desc_returns_the_gdt_qword_and_no_high_half_for_code_and_data() {
            let mut h = harness();
            let expected = h.read_u64(GDT_BASE + KERNEL_CS as u64);
            // The qword after a code descriptor is some other descriptor; it must not be read as a
            // high half.
            assert_eq!(h.cpu.read_desc(KERNEL_CS).unwrap(), (expected, 0));
        }

        #[test]
        fn read_desc_ignores_the_rpl_bits_of_the_selector() {
            let mut h = harness();
            let expected = h.read_u64(GDT_BASE + KERNEL_CS as u64);
            for rpl in 0..4 {
                assert_eq!(h.cpu.read_desc(KERNEL_CS | rpl).unwrap().0, expected);
            }
        }

        #[test]
        fn read_desc_returns_both_halves_of_a_system_descriptor() {
            let mut h = harness();
            let low = descriptor(0x1234_5678, 0x67, SYSTEM_DESCRIPTOR_TYPE_TSS_AVAILABLE, 0);
            set_gdt_entry(&mut h, TSS_SELECTOR, low);
            set_gdt_entry(&mut h, TSS_SELECTOR + 8, 0xffff_8000_dead_beef);
            assert_eq!(
                h.cpu.read_desc(TSS_SELECTOR).unwrap(),
                (low, 0xffff_8000_dead_beef)
            );
        }

        #[test]
        fn read_desc_gives_zero_for_a_high_half_outside_the_limit() {
            let mut h = harness();
            let low = descriptor(0x1234_5678, 0x67, SYSTEM_DESCRIPTOR_TYPE_TSS_AVAILABLE, 0);
            set_gdt_entry(&mut h, TSS_SELECTOR, low);
            h.write_u64(GDT_BASE + TSS_SELECTOR as u64 + 8, 0x1111);
            h.cpu.gdtr.limit = TSS_SELECTOR + 7;
            assert_eq!(h.cpu.read_desc(TSS_SELECTOR).unwrap(), (low, 0));
            h.cpu.gdtr.limit = TSS_SELECTOR + 15;
            assert_eq!(h.cpu.read_desc(TSS_SELECTOR).unwrap(), (low, 0x1111));
        }

        #[test]
        fn read_desc_limit_boundary_is_the_last_byte_of_the_descriptor() {
            let mut h = harness();
            h.cpu.gdtr.limit = KERNEL_CS + 7;
            h.cpu.read_desc(KERNEL_CS).unwrap();
            h.cpu.gdtr.limit = KERNEL_CS + 6;
            assert_eq!(
                fault_of(h.cpu.read_desc(KERNEL_CS)),
                (VECTOR_GENERAL_PROTECTION, Some(KERNEL_CS as u32))
            );
        }

        #[test]
        fn read_desc_error_code_drops_rpl_but_keeps_the_table_bit() {
            let mut h = harness();
            h.cpu.gdtr.limit = 0x1f;
            assert_eq!(
                fault_of(h.cpu.read_desc(0x1003)),
                (VECTOR_GENERAL_PROTECTION, Some(0x1000))
            );
            h.cpu.ldtr = Seg {
                sel: 0x38,
                base: SCRATCH_FOR_LDT,
                limit: 0xf,
                flags: 0,
            };
            assert_eq!(
                fault_of(h.cpu.read_desc(0x1007)),
                (VECTOR_GENERAL_PROTECTION, Some(0x1004))
            );
        }

        const SCRATCH_FOR_LDT: u64 = 0x50_0000;

        #[test]
        fn selector_with_the_ldt_bit_reads_the_ldt_not_the_gdt() {
            let mut h = harness();
            h.cpu.ldtr = Seg {
                sel: 0x38,
                base: SCRATCH_FOR_LDT,
                limit: 0x17,
                flags: 0,
            };
            let marker = 0x00cf_fa00_0000_ffff;
            h.write_u64(SCRATCH_FOR_LDT + 0x10, marker);
            assert_eq!(h.cpu.read_desc(0x10 | 0x4).unwrap().0, marker);
            assert_ne!(h.cpu.read_desc(0x10).unwrap().0, marker);
            // The LDT limit applies, not the GDT's.
            assert_eq!(
                fault_of(h.cpu.read_desc(0x18 | 0x4 | 0x8)),
                (VECTOR_GENERAL_PROTECTION, Some(0x1c))
            );
        }

        #[test]
        fn null_selector_reads_the_null_descriptor() {
            let mut h = harness();
            assert_eq!(h.cpu.read_desc(0).unwrap(), (0, 0));
        }

        #[test]
        fn load_desc_seg_assembles_base_limit_and_flags() {
            let mut h = harness();
            // base 0x1234_5678, limit 0xabcde, access 0x92 (present data), flags 0x4 (D/B).
            set_gdt_entry(&mut h, 0x78, descriptor(0x1234_5678, 0xa_bcde, 0x92, 0x4));
            let seg = h.cpu.load_desc_seg(0x7b, true).unwrap();
            assert_eq!(seg.sel, 0x7b);
            assert_eq!(seg.base, 0x1234_5678);
            assert_eq!(seg.limit, 0xa_bcde);
            assert_eq!(seg.flags, 0x0040_9200);
        }

        #[test]
        fn load_desc_seg_scales_the_limit_by_granularity() {
            let mut h = harness();
            set_gdt_entry(&mut h, 0x78, descriptor(0, 0xf_ffff, 0x92, 0x8));
            assert_eq!(h.cpu.load_desc_seg(0x78, true).unwrap().limit, 0xffff_ffff);
            set_gdt_entry(&mut h, 0x78, descriptor(0, 0, 0x92, 0x8));
            assert_eq!(h.cpu.load_desc_seg(0x78, true).unwrap().limit, 0xfff);
            set_gdt_entry(&mut h, 0x78, descriptor(0, 0x1_2345, 0x92, 0x8));
            assert_eq!(h.cpu.load_desc_seg(0x78, true).unwrap().limit, 0x1234_5fff);
            set_gdt_entry(&mut h, 0x78, descriptor(0, 0x1_2345, 0x92, 0x0));
            assert_eq!(h.cpu.load_desc_seg(0x78, true).unwrap().limit, 0x1_2345);
        }

        #[test]
        fn load_desc_seg_keeps_only_the_architectural_flag_bits() {
            let mut h = harness();
            // Limit bits 19:16 sit in the byte that also holds AVL/L/D/G: they must not leak.
            set_gdt_entry(&mut h, 0x78, descriptor(0xffff_ffff, 0xf_ffff, 0x9a, 0xa));
            let seg = h.cpu.load_desc_seg(0x78, true).unwrap();
            assert_eq!(seg.flags & !SEG_DESCRIPTOR_FLAGS_MASK, 0);
            assert_eq!(seg.flags, 0x00a0_9a00);
            assert_eq!(seg.base, 0xffff_ffff);
        }

        #[test]
        fn load_desc_seg_with_require_present_raises_np_with_the_selector() {
            let mut h = harness();
            assert_eq!(
                fault_of(h.cpu.load_desc_seg(NOT_PRESENT_CS | 3, true)),
                (VECTOR_SEGMENT_NOT_PRESENT, Some(NOT_PRESENT_CS as u32))
            );
        }

        #[test]
        fn load_desc_seg_without_require_present_returns_the_absent_descriptor() {
            let mut h = harness();
            let seg = h.cpu.load_desc_seg(NOT_PRESENT_CS, false).unwrap();
            assert!(!seg.present());
            assert!(seg.is_code());
        }

        #[test]
        fn load_desc_seg_reports_bounds_errors_before_presence() {
            let mut h = harness();
            h.cpu.gdtr.limit = 0x1f;
            assert_eq!(
                fault_of(h.cpu.load_desc_seg(0x40, true)),
                (VECTOR_GENERAL_PROTECTION, Some(0x40))
            );
        }
    }

    mod gate_delivery {
        use super::*;

        #[test]
        fn interrupt_gate_transfers_to_the_handler_and_reports_no_exit() {
            let mut h = base();
            assert!(h.cpu.deliver_event(0x20, None, false).is_none());
            assert_eq!(h.cpu.rip, handler_for(0x20));
            assert_eq!(h.cpu.seg[SEG_CS].sel, KERNEL_CS);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn vector_zero_and_vector_255_use_the_first_and_last_idt_entry() {
            for vector in [0u8, 255] {
                let mut h = base();
                assert!(h.cpu.deliver_event(vector, None, false).is_none());
                assert_eq!(h.cpu.rip, handler_for(vector), "vector {vector}");
            }
        }

        #[test]
        fn handler_address_is_assembled_from_the_three_gate_fields() {
            let mut h = base();
            let handler = 0xffff_8000_1234_abcd;
            set_gate(&mut h, 0x30, Gate::interrupt(handler));
            h.cpu.deliver_event(0x30, None, false);
            assert_eq!(h.cpu.rip, handler);
            let handler = 0x0123_4567_89ab_cdef;
            set_gate(&mut h, 0x31, Gate::interrupt(handler));
            h.cpu.deliver_event(0x31, None, false);
            assert_eq!(h.cpu.rip, handler);
        }

        #[test]
        fn frame_holds_rip_cs_rflags_rsp_ss_in_that_order_below_the_aligned_stack() {
            let mut h = base();
            h.cpu.eflags = RESERVED_ONE | IF | DF | IOPL;
            h.cpu.deliver_event(0x20, None, false);
            let rsp = h.cpu.gpr[GPR_RSP];
            assert_eq!(rsp, aligned_below(STACK_TOP, FRAME_BYTES));
            assert_eq!(
                read_frame(&h, rsp, false),
                Frame {
                    error_code: None,
                    rip: OLD_RIP,
                    cs: KERNEL_CS as u64,
                    rflags: RESERVED_ONE | IF | DF | IOPL,
                    rsp: STACK_TOP,
                    ss: KERNEL_DS as u64,
                }
            );
        }

        #[test]
        fn error_code_is_pushed_last_when_present() {
            let mut h = base();
            h.cpu
                .deliver_event(VECTOR_GENERAL_PROTECTION, Some(0x1234), false);
            let rsp = h.cpu.gpr[GPR_RSP];
            assert_eq!(rsp, aligned_below(STACK_TOP, FRAME_BYTES + WORD));
            let frame = read_frame(&h, rsp, true);
            assert_eq!(frame.error_code, Some(0x1234));
            assert_eq!(frame.rip, OLD_RIP);
        }

        #[test]
        fn error_code_is_zero_extended_to_a_qword() {
            let mut h = base();
            h.cpu
                .deliver_event(VECTOR_PAGE_FAULT, Some(0xffff_ffff), false);
            let rsp = h.cpu.gpr[GPR_RSP];
            assert_eq!(h.read_u64(rsp), 0xffff_ffff);
        }

        #[test]
        fn stack_pointer_is_aligned_down_to_sixteen_bytes_but_the_old_one_is_pushed_as_is() {
            for misalignment in 0..16 {
                let mut h = base();
                let old_rsp = STACK_TOP - misalignment;
                h.cpu.gpr[GPR_RSP] = old_rsp;
                h.cpu.deliver_event(0x20, None, false);
                let rsp = h.cpu.gpr[GPR_RSP];
                assert_eq!(
                    rsp,
                    aligned_below(old_rsp, FRAME_BYTES),
                    "misalignment {misalignment}"
                );
                assert_eq!(rsp % 16, 8, "the frame leaves rsp 8 mod 16");
                assert_eq!(read_frame(&h, rsp, false).rsp, old_rsp);
            }
        }

        #[test]
        fn pushed_rflags_resolves_the_deferred_arithmetic_flags() {
            let mut h = base();
            // 0xff + 1 = 0 in 8 bits: CF, ZF, PF, AF.
            h.cpu.set_lazy_flags(LazyFlagsOp::Add, 1, 0, 1, 0);
            h.cpu.deliver_event(0x20, None, false);
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.rflags & ARITH, CF | ZF | PF | AF);
            assert_eq!(frame.rflags & IF, IF);
            assert_eq!(h.cpu.lazy_op, LazyFlagsOp::Eflags);
            assert_eq!(h.cpu.eflags_full() & ARITH, CF | ZF | PF | AF);
        }

        #[test]
        fn interrupt_gate_clears_if_and_the_other_entry_flags_but_pushes_the_old_value() {
            let mut h = base();
            let all = RESERVED_ONE | IF | TF | NT | RF | VM | DF | IOPL | CF;
            h.cpu.eflags = all;
            h.cpu.deliver_event(0x20, None, false);
            let after = h.cpu.eflags_full();
            assert_eq!(after & (IF | CLEARED_ON_ENTRY), 0);
            assert_eq!(after & (DF | IOPL | CF), DF | IOPL | CF);
            assert_eq!(read_frame(&h, h.cpu.gpr[GPR_RSP], false).rflags, all);
        }

        #[test]
        fn trap_gate_leaves_if_set_but_clears_tf_nt_rf_vm() {
            let mut h = base();
            set_gate(&mut h, 0x21, Gate::trap(handler_for(0x21)));
            h.cpu.eflags = RESERVED_ONE | IF | TF | NT | RF | VM;
            h.cpu.deliver_event(0x21, None, false);
            let after = h.cpu.eflags_full();
            assert_eq!(after & IF, IF);
            assert_eq!(after & CLEARED_ON_ENTRY, 0);
        }

        #[test]
        fn interrupt_gate_with_if_already_clear_stays_clear() {
            let mut h = base();
            h.cpu.eflags = RESERVED_ONE;
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.eflags_full() & IF, 0);
        }

        #[test]
        fn delivery_leaves_the_general_registers_alone() {
            let mut h = base();
            for (i, reg) in h.cpu.gpr.iter_mut().enumerate() {
                if i != GPR_RSP {
                    *reg = 0x1111_0000 + i as u64;
                }
            }
            h.cpu.deliver_event(0x20, None, false);
            for (i, reg) in h.cpu.gpr.iter().enumerate() {
                if i != GPR_RSP {
                    assert_eq!(*reg, 0x1111_0000 + i as u64);
                }
            }
        }

        #[test]
        fn idt_limit_must_cover_the_whole_sixteen_byte_entry() {
            let mut h = base();
            let vector = 0x40u8;
            h.cpu.idtr.limit = vector as u16 * 16 + 15;
            h.cpu.deliver(vector, None, false).unwrap();
            h.cpu.idtr.limit = vector as u16 * 16 + 14;
            assert_eq!(
                fault_of(h.cpu.deliver(vector, None, false)),
                (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(vector)))
            );
        }

        #[test]
        fn empty_idt_limit_faults_every_vector() {
            let mut h = base();
            h.cpu.idtr.limit = 0;
            assert_eq!(
                fault_of(h.cpu.deliver(0, None, false)),
                (VECTOR_GENERAL_PROTECTION, Some(2))
            );
        }

        #[test]
        fn missing_gate_raises_np_with_the_idt_error_code() {
            let mut h = base();
            set_gate(
                &mut h,
                0x41,
                Gate {
                    present: false,
                    ..Gate::interrupt(handler_for(0x41))
                },
            );
            assert_eq!(
                fault_of(h.cpu.deliver(0x41, None, false)),
                (VECTOR_SEGMENT_NOT_PRESENT, Some(idt_error_code(0x41)))
            );
        }

        #[test]
        fn only_interrupt_and_trap_gate_types_are_accepted() {
            for gate_type in 0..16u8 {
                let mut h = base();
                set_gate(
                    &mut h,
                    0x42,
                    Gate {
                        gate_type,
                        ..Gate::interrupt(handler_for(0x42))
                    },
                );
                let result = h.cpu.deliver(0x42, None, false);
                if gate_type == TYPE_INTERRUPT_GATE || gate_type == TYPE_TRAP_GATE {
                    result.unwrap();
                } else {
                    assert_eq!(
                        fault_of(result),
                        (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(0x42))),
                        "type {gate_type:#x}"
                    );
                }
            }
        }

        #[test]
        fn gate_type_is_checked_before_presence() {
            let mut h = base();
            set_gate(
                &mut h,
                0x43,
                Gate {
                    present: false,
                    gate_type: 0,
                    ..Gate::interrupt(handler_for(0x43))
                },
            );
            assert_eq!(
                fault_of(h.cpu.deliver(0x43, None, false)),
                (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(0x43)))
            );
        }

        #[test]
        fn descriptor_table_bases_near_the_top_of_the_address_space_do_not_overflow() {
            // lidt/lgdt/ltr accept any 64-bit base: the entry address arithmetic must wrap, not panic.
            let mut h = base();
            h.cpu.idtr.base = u64::MAX - 7;
            assert!(matches!(
                h.cpu.deliver_event(0x20, None, false),
                Some(ExitReason::TripleFault)
            ));
            let mut h = base();
            h.cpu.gdtr.base = u64::MAX - 7;
            // The wrapped address lands on page 0, which the identity map covers: no fault either.
            assert_eq!(h.cpu.read_desc(KERNEL_CS).unwrap(), (0, 0));
            h.cpu.tr.base = u64::MAX - 3;
            h.cpu.tr.limit = 0xffff;
            assert_eq!(h.cpu.tss_read(8).unwrap(), 0);
        }

        #[test]
        fn idt_entry_in_unmapped_memory_page_faults() {
            let mut h = base();
            unmap_boot_identity_2m(&mut h, HOLE);
            h.cpu.idtr.base = HOLE + 0x1000;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_PAGE_FAULT, Some(0))
            );
            assert_eq!(h.cpu.cr2, HOLE + 0x1000 + 0x20 * 16);
        }
    }

    mod soft_interrupt_privilege {
        use super::*;

        #[test]
        fn software_interrupt_through_a_dpl0_gate_from_ring_three_is_gp() {
            let mut h = user_base();
            assert_eq!(
                fault_of(h.cpu.deliver(0x80, None, true)),
                (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(0x80)))
            );
        }

        #[test]
        fn software_interrupt_through_a_dpl3_gate_from_ring_three_works() {
            let mut h = user_base();
            set_gate(
                &mut h,
                0x80,
                Gate {
                    dpl: 3,
                    ..Gate::interrupt(handler_for(0x80))
                },
            );
            assert!(h.cpu.deliver_event(0x80, None, true).is_none());
            assert_eq!(h.cpu.rip, handler_for(0x80));
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn hardware_events_ignore_the_gate_dpl() {
            let mut h = user_base();
            assert!(h.cpu.deliver_event(0x20, None, false).is_none());
            assert_eq!(h.cpu.rip, handler_for(0x20));
        }

        #[test]
        fn gate_dpl_is_compared_with_cpl_for_every_level() {
            for cpl in 0..4u8 {
                for dpl in 0..4u8 {
                    let mut h = base();
                    h.cpu.set_cpl(cpl);
                    set_gate(
                        &mut h,
                        0x80,
                        Gate {
                            dpl,
                            ..Gate::interrupt(handler_for(0x80))
                        },
                    );
                    // Delivery to a DPL0 code segment from an outer ring needs a stack in the TSS.
                    let result = h.cpu.deliver(0x80, None, true);
                    if dpl < cpl {
                        assert_eq!(
                            fault_of(result),
                            (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(0x80))),
                            "cpl {cpl} gate dpl {dpl}"
                        );
                    } else {
                        result.unwrap();
                    }
                }
            }
        }

        #[test]
        fn dpl_check_comes_before_the_present_check() {
            let mut h = user_base();
            set_gate(
                &mut h,
                0x80,
                Gate {
                    present: false,
                    ..Gate::interrupt(handler_for(0x80))
                },
            );
            assert_eq!(
                fault_of(h.cpu.deliver(0x80, None, true)),
                (VECTOR_GENERAL_PROTECTION, Some(idt_error_code(0x80)))
            );
        }

        #[test]
        fn software_interrupt_at_ring_zero_through_a_dpl0_gate_works() {
            let mut h = base();
            assert!(h.cpu.deliver_event(0x80, None, true).is_none());
            assert_eq!(h.cpu.rip, handler_for(0x80));
        }
    }

    mod target_segment_checks {
        use super::*;

        fn gate_to(h: &mut Harness, selector: u16) {
            set_gate(
                h,
                0x50,
                Gate {
                    selector,
                    ..Gate::interrupt(handler_for(0x50))
                },
            );
        }

        fn assert_fault_leaves_state(h: &mut Harness, expected: (u8, Option<u32>)) {
            let before = state(h);
            assert_eq!(fault_of(h.cpu.deliver(0x50, None, false)), expected);
            assert_eq!(
                state(h),
                before,
                "a faulting delivery must not commit anything"
            );
        }

        #[test]
        fn null_selector_is_gp_zero() {
            for selector in [0u16, 1, 2, 3] {
                let mut h = base();
                gate_to(&mut h, selector);
                assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(0)));
            }
        }

        #[test]
        fn null_selector_is_gp_zero_even_if_gdt_entry_zero_holds_a_valid_code_descriptor() {
            let mut h = base();
            h.write_u64(GDT_BASE, 0x00af_9a00_0000_ffff);
            gate_to(&mut h, 0);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(0)));
        }

        #[test]
        fn selector_beyond_the_gdt_limit_is_gp_with_the_selector() {
            let mut h = base();
            gate_to(&mut h, 0x1000);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(0x1000)));
        }

        #[test]
        fn data_segment_target_is_gp_with_the_selector() {
            let mut h = base();
            gate_to(&mut h, KERNEL_DS);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(KERNEL_DS as u32)));
        }

        #[test]
        fn system_segment_target_is_gp_with_the_selector() {
            let mut h = base();
            set_gdt_entry(&mut h, TSS_SELECTOR, descriptor(0x1000, 0x67, 0x89, 0));
            gate_to(&mut h, TSS_SELECTOR);
            assert_fault_leaves_state(
                &mut h,
                (VECTOR_GENERAL_PROTECTION, Some(TSS_SELECTOR as u32)),
            );
        }

        #[test]
        fn not_present_code_segment_is_np_with_the_selector() {
            let mut h = base();
            gate_to(&mut h, NOT_PRESENT_CS);
            assert_fault_leaves_state(
                &mut h,
                (VECTOR_SEGMENT_NOT_PRESENT, Some(NOT_PRESENT_CS as u32)),
            );
        }

        #[test]
        fn selector_error_codes_drop_the_rpl() {
            let mut h = base();
            gate_to(&mut h, NOT_PRESENT_CS | 3);
            assert_fault_leaves_state(
                &mut h,
                (VECTOR_SEGMENT_NOT_PRESENT, Some(NOT_PRESENT_CS as u32)),
            );
            gate_to(&mut h, KERNEL_DS | 2);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(KERNEL_DS as u32)));
        }

        #[test]
        fn compatibility_mode_code_segment_is_gp() {
            let mut h = base();
            gate_to(&mut h, COMPAT_CS);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(COMPAT_CS as u32)));
        }

        #[test]
        fn code_segment_with_both_long_and_default_size_set_is_gp() {
            let mut h = base();
            gate_to(&mut h, BIG_LONG_CS);
            assert_fault_leaves_state(
                &mut h,
                (VECTOR_GENERAL_PROTECTION, Some(BIG_LONG_CS as u32)),
            );
        }

        #[test]
        fn non_conforming_code_segment_less_privileged_than_cpl_is_gp() {
            let mut h = base();
            gate_to(&mut h, DPL3_NONCONFORMING_CS);
            assert_fault_leaves_state(
                &mut h,
                (
                    VECTOR_GENERAL_PROTECTION,
                    Some(DPL3_NONCONFORMING_CS as u32),
                ),
            );
        }

        #[test]
        fn conforming_code_segment_less_privileged_than_cpl_is_gp() {
            let mut h = base();
            gate_to(&mut h, CONFORMING_CS_DPL3);
            assert_fault_leaves_state(
                &mut h,
                (VECTOR_GENERAL_PROTECTION, Some(CONFORMING_CS_DPL3 as u32)),
            );
        }

        #[test]
        fn type_problems_are_reported_before_a_missing_present_bit() {
            let mut h = base();
            // A not-present data descriptor: it is the wrong type, which wins over not present.
            set_gdt_entry(&mut h, 0x78, descriptor(0, 0xfffff, 0x12, 0xc));
            gate_to(&mut h, 0x78);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(0x78)));
        }

        #[test]
        fn privilege_problems_are_reported_before_a_missing_present_bit() {
            let mut h = base();
            // Not present and DPL 3 from ring 0: DPL > CPL is checked first.
            set_gdt_entry(
                &mut h,
                0x78,
                descriptor(0, 0xfffff, 0x9a & !0x80 | 0x60, 0xa),
            );
            gate_to(&mut h, 0x78);
            assert_fault_leaves_state(&mut h, (VECTOR_GENERAL_PROTECTION, Some(0x78)));
        }
    }

    mod privilege_and_stacks {
        use super::*;

        #[test]
        fn user_to_kernel_switches_to_rsp0_and_records_the_user_context() {
            let mut h = user_base();
            h.cpu.eflags = RESERVED_ONE | IF | ZF;
            assert!(h.cpu.deliver_event(0x20, None, false).is_none());
            assert_eq!(h.cpu.cpl, 0);
            let rsp = h.cpu.gpr[GPR_RSP];
            assert_eq!(rsp, aligned_below(RSP0, FRAME_BYTES));
            assert_eq!(
                read_frame(&h, rsp, false),
                Frame {
                    error_code: None,
                    rip: OLD_RIP,
                    cs: USER_CS as u64,
                    rflags: RESERVED_ONE | IF | ZF,
                    rsp: USER_RSP,
                    ss: USER_DS as u64,
                }
            );
        }

        #[test]
        fn user_to_kernel_loads_cs_and_a_null_ss_with_rpl_equal_to_the_new_cpl() {
            let mut h = user_base();
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.seg[SEG_CS].sel, KERNEL_CS);
            assert_eq!(h.cpu.seg[SEG_SS].sel, 0);
            assert_eq!(h.cpu.seg[SEG_SS].dpl(), 0);
            assert!(h.cpu.seg[SEG_SS].present());
        }

        #[test]
        fn privilege_change_switches_the_tlb_half() {
            let mut h = user_base();
            let user_half = h.cpu.tlb.as_ptr() as u64
                + (TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>()) as u64;
            assert_eq!(h.cpu.tlb_active_base, user_half);
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.tlb_active_base, h.cpu.tlb.as_ptr() as u64);
        }

        #[test]
        fn rsp0_is_aligned_down_like_any_other_new_stack() {
            let mut h = user_base();
            install_tss(&mut h, [RSP0 + 0xc, 0, 0], [0; 7]);
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.gpr[GPR_RSP], aligned_below(RSP0, FRAME_BYTES));
        }

        #[test]
        fn same_privilege_delivery_stays_on_the_current_stack_and_keeps_ss() {
            let mut h = base();
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.seg[SEG_SS].sel, KERNEL_DS);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn dpl1_non_conforming_target_from_ring_three_uses_rsp1_and_cpl_one() {
            let mut h = user_base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: DPL1_CS,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.cpl, 1);
            assert_eq!(h.cpu.gpr[GPR_RSP], aligned_below(RSP1, FRAME_BYTES));
            assert_eq!(h.cpu.seg[SEG_CS].sel, DPL1_CS);
            assert_eq!(h.cpu.seg[SEG_SS].sel, 1);
            assert_eq!(h.cpu.seg[SEG_SS].dpl(), 1);
        }

        #[test]
        fn non_conforming_target_at_the_current_level_does_not_switch_stacks() {
            let mut h = user_base();
            with_user_stack_paged(&mut h, PTE_PRESENT_WRITABLE_USER);
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: DPL3_NONCONFORMING_CS,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.cpl, 3);
            assert_eq!(h.cpu.gpr[GPR_RSP], aligned_below(USER_RSP, FRAME_BYTES));
            assert_eq!(h.cpu.seg[SEG_SS].sel, USER_DS);
            assert_eq!(h.cpu.seg[SEG_CS].sel, DPL3_NONCONFORMING_CS | 3);
        }

        #[test]
        fn conforming_target_keeps_the_current_privilege_level() {
            let mut h = user_base();
            with_user_stack_paged(&mut h, PTE_PRESENT_WRITABLE_USER);
            // The TSS must not even be consulted: no stack switch happens.
            h.cpu.tr.limit = 0;
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: CONFORMING_CS_DPL0,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            assert!(h.cpu.deliver_event(0x20, None, false).is_none());
            assert_eq!(
                h.cpu.cpl, 3,
                "a conforming handler runs at the caller's CPL"
            );
            assert_eq!(h.cpu.gpr[GPR_RSP], aligned_below(USER_RSP, FRAME_BYTES));
            assert_eq!(
                h.cpu.seg[SEG_SS].sel, USER_DS,
                "SS is not replaced without a stack switch"
            );
            assert_eq!(h.cpu.seg[SEG_CS].sel, CONFORMING_CS_DPL0 | 3);
            assert_eq!(
                h.cpu.tlb_active_base,
                h.cpu.tlb.as_ptr() as u64
                    + (TLB_ENTRIES_PER_MODE * core::mem::size_of::<TlbEntry>()) as u64
            );
        }

        #[test]
        fn ring_three_handler_pushes_are_checked_as_user_accesses() {
            // A conforming handler keeps CPL 3, so the frame goes to the user stack with user
            // permissions: a supervisor-only stack page faults with the U bit.
            let mut h = user_base();
            with_user_stack_paged(&mut h, SUPERVISOR_ONLY_FLAGS);
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: CONFORMING_CS_DPL0,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            let before = state(&h);
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_PAGE_FAULT, Some(PF_P | PF_W | PF_U))
            );
            assert_eq!(state(&h), before);
        }

        #[test]
        fn privilege_changing_pushes_stay_supervisor_accesses() {
            // To ring 0 the frame goes to the kernel stack from the TSS, which is supervisor-only.
            let mut h = user_base();
            with_user_stack_paged(&mut h, SUPERVISOR_ONLY_FLAGS);
            h.cpu.deliver(0x20, None, false).unwrap();
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn conforming_target_at_ring_zero_stays_at_ring_zero() {
            let mut h = base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: CONFORMING_CS_DPL0,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.cpl, 0);
            assert_eq!(h.cpu.seg[SEG_CS].sel, CONFORMING_CS_DPL0);
        }

        #[test]
        fn conforming_target_with_dpl_equal_to_cpl_is_accepted() {
            let mut h = user_base();
            with_user_stack_paged(&mut h, PTE_PRESENT_WRITABLE_USER);
            set_gate(
                &mut h,
                0x20,
                Gate {
                    selector: CONFORMING_CS_DPL3,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(h.cpu.cpl, 3);
        }

        #[test]
        fn cs_selector_rpl_is_forced_to_the_new_cpl() {
            for rpl in 0..4u16 {
                let mut h = base();
                set_gate(
                    &mut h,
                    0x20,
                    Gate {
                        selector: KERNEL_CS | rpl,
                        ..Gate::interrupt(handler_for(0x20))
                    },
                );
                h.cpu.deliver_event(0x20, None, false);
                assert_eq!(h.cpu.seg[SEG_CS].sel, KERNEL_CS, "gate selector RPL {rpl}");
            }
        }

        #[test]
        fn tss_limit_must_cover_the_rsp_slot() {
            let mut h = user_base();
            // RSP0 occupies bytes 4..=11.
            h.cpu.tr.limit = 11;
            h.cpu.deliver(0x20, None, false).unwrap();
            let mut h = user_base();
            h.cpu.tr.limit = 10;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_INVALID_TSS, Some((TSS_SELECTOR & !3) as u32))
            );
        }

        #[test]
        fn each_ist_slot_selects_its_own_stack() {
            for ist in 1..=7u8 {
                let mut h = base();
                set_gate(
                    &mut h,
                    0x20,
                    Gate {
                        ist,
                        ..Gate::interrupt(handler_for(0x20))
                    },
                );
                h.cpu.deliver_event(0x20, None, false);
                assert_eq!(
                    h.cpu.gpr[GPR_RSP],
                    aligned_below(IST_STACKS[ist as usize - 1], FRAME_BYTES),
                    "IST{ist}"
                );
                assert_eq!(
                    read_frame(&h, h.cpu.gpr[GPR_RSP], false).rsp,
                    STACK_TOP,
                    "the interrupted stack pointer is saved"
                );
            }
        }

        #[test]
        fn ist_at_the_same_privilege_still_pushes_the_old_ss_and_rsp() {
            let mut h = base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    ist: 3,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], false);
            assert_eq!(frame.ss, KERNEL_DS as u64);
            assert_eq!(frame.cs, KERNEL_CS as u64);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn ist_takes_priority_over_the_privilege_based_stack() {
            let mut h = user_base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    ist: 1,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.deliver_event(0x20, None, false);
            assert_eq!(
                h.cpu.gpr[GPR_RSP],
                aligned_below(IST_STACKS[0], FRAME_BYTES)
            );
        }

        #[test]
        fn tss_limit_must_cover_the_ist_slot() {
            // IST7 occupies bytes 84..=91.
            let mut h = base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    ist: 7,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.tr.limit = 91;
            h.cpu.deliver(0x20, None, false).unwrap();
            let mut h = base();
            set_gate(
                &mut h,
                0x20,
                Gate {
                    ist: 7,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            h.cpu.tr.limit = 90;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_INVALID_TSS, Some((TSS_SELECTOR & !3) as u32))
            );
        }

        #[test]
        fn invalid_tss_error_code_drops_the_rpl_of_tr() {
            let mut h = user_base();
            h.cpu.tr.sel = TSS_SELECTOR | 3;
            h.cpu.tr.limit = 0;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_INVALID_TSS, Some(TSS_SELECTOR as u32))
            );
        }
    }

    mod atomicity_and_stack_faults {
        use super::*;

        #[test]
        fn push_fault_on_the_first_word_leaves_the_cpu_untouched() {
            let mut h = base();
            unmap_boot_identity_2m(&mut h, HOLE);
            h.cpu.gpr[GPR_RSP] = HOLE + 0x1000;
            let before = state(&h);
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_PAGE_FAULT, Some(PF_W))
            );
            assert_eq!(state(&h), before);
            assert_eq!(h.cpu.cr2, aligned_below(HOLE + 0x1000, WORD));
        }

        #[test]
        fn push_fault_after_some_words_were_written_still_commits_nothing() {
            let mut h = base();
            unmap_boot_identity_2m(&mut h, HOLE);
            // Two words fit above the hole; the third push crosses into it.
            h.cpu.gpr[GPR_RSP] = HOLE_END + 0x10;
            let before = state(&h);
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_PAGE_FAULT, Some(PF_W))
            );
            assert_eq!(state(&h), before);
            assert_eq!(h.cpu.cr2, HOLE_END - WORD);
            // The words pushed before the fault stay in memory: the first push is the old SS.
            assert_eq!(h.read_u64(HOLE_END + 0x10 - WORD), KERNEL_DS as u64);
        }

        #[test]
        fn failed_privilege_change_keeps_the_user_context() {
            let mut h = user_base();
            unmap_boot_identity_2m(&mut h, HOLE);
            install_tss(&mut h, [HOLE + 0x1000, 0, 0], [0; 7]);
            let before = state(&h);
            h.cpu.deliver(0x20, None, false).unwrap_err();
            assert_eq!(state(&h), before);
            assert_eq!(h.cpu.cpl, 3);
            assert_eq!(h.cpu.seg[SEG_SS].sel, USER_DS);
        }

        #[test]
        fn non_canonical_ist_stack_is_a_stack_fault_with_code_zero() {
            let mut h = base();
            install_tss(
                &mut h,
                [RSP0, 0, 0],
                [NON_CANONICAL_STACK, 0, 0, 0, 0, 0, 0],
            );
            set_gate(
                &mut h,
                0x20,
                Gate {
                    ist: 1,
                    ..Gate::interrupt(handler_for(0x20))
                },
            );
            let before = state(&h);
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_STACK_FAULT, Some(0))
            );
            assert_eq!(state(&h), before);
        }

        #[test]
        fn non_canonical_rsp0_is_a_stack_fault() {
            let mut h = user_base();
            install_tss(&mut h, [NOT_CANONICAL + 0x10, 0, 0], [0; 7]);
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_STACK_FAULT, Some(0))
            );
        }

        #[test]
        fn non_canonical_current_stack_pointer_is_a_stack_fault() {
            let mut h = base();
            h.cpu.gpr[GPR_RSP] = NOT_CANONICAL + 0x40;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_STACK_FAULT, Some(0))
            );
        }

        #[test]
        fn stack_crossing_into_the_non_canonical_hole_is_a_stack_fault() {
            let mut h = base();
            // Stack at the bottom of the high half: the first push goes below it, into the hole.
            h.cpu.gpr[GPR_RSP] = 0xffff_8000_0000_0000;
            assert_eq!(
                fault_of(h.cpu.deliver(0x20, None, false)),
                (VECTOR_STACK_FAULT, Some(0))
            );
        }
    }

    mod nested_faults {
        use super::*;

        /// Ways to make delivering `vector` raise a chosen nested exception.
        #[derive(Clone, Copy, Debug)]
        enum Break {
            /// Gate type is not interrupt/trap: #GP(idt).
            InvalidType,
            /// Gate is not present: #NP(idt).
            NotPresent,
            /// Gate uses IST1 but the TSS is too small: #TS(tr).
            TssTooSmall,
            /// Gate uses IST1 pointing into unmapped memory: #PF.
            IstUnmapped,
            /// Gate uses IST1 holding a non-canonical address: #SS(0).
            IstNonCanonical,
        }

        const ALL_BREAKS: [Break; 5] = [
            Break::InvalidType,
            Break::NotPresent,
            Break::TssTooSmall,
            Break::IstUnmapped,
            Break::IstNonCanonical,
        ];

        fn nested_vector(how: Break) -> u8 {
            match how {
                Break::InvalidType => VECTOR_GENERAL_PROTECTION,
                Break::NotPresent => VECTOR_SEGMENT_NOT_PRESENT,
                Break::TssTooSmall => VECTOR_INVALID_TSS,
                Break::IstUnmapped => VECTOR_PAGE_FAULT,
                Break::IstNonCanonical => VECTOR_STACK_FAULT,
            }
        }

        fn nested_error_code(how: Break, vector: u8) -> u32 {
            match how {
                Break::InvalidType | Break::NotPresent => idt_error_code(vector),
                Break::TssTooSmall => (TSS_SELECTOR & !3) as u32,
                Break::IstUnmapped => PF_W,
                Break::IstNonCanonical => 0,
            }
        }

        fn broken(vector: u8, how: Break) -> Harness {
            let mut h = base();
            unmap_boot_identity_2m(&mut h, HOLE);
            let gate = Gate::interrupt(handler_for(vector));
            match how {
                Break::InvalidType => set_gate(
                    &mut h,
                    vector,
                    Gate {
                        gate_type: 0x5,
                        ..gate
                    },
                ),
                Break::NotPresent => set_gate(
                    &mut h,
                    vector,
                    Gate {
                        present: false,
                        ..gate
                    },
                ),
                Break::TssTooSmall => {
                    set_gate(&mut h, vector, Gate { ist: 1, ..gate });
                    h.cpu.tr.limit = 10;
                }
                Break::IstUnmapped => {
                    set_gate(&mut h, vector, Gate { ist: 1, ..gate });
                    install_tss(&mut h, [RSP0, 0, 0], [HOLE + 0x1000, 0, 0, 0, 0, 0, 0]);
                }
                Break::IstNonCanonical => {
                    set_gate(&mut h, vector, Gate { ist: 1, ..gate });
                    install_tss(
                        &mut h,
                        [RSP0, 0, 0],
                        [NON_CANONICAL_STACK, 0, 0, 0, 0, 0, 0],
                    );
                }
            }
            h
        }

        fn pushed_error_code(h: &Harness) -> u64 {
            h.read_u64(h.cpu.gpr[GPR_RSP])
        }

        /// Delivers `first` through a gate broken in way `how` and checks that the CPU ended up in
        /// the handler of `expected_vector` with the matching error code.
        fn check_pair(first: u8, how: Break, expected_vector: u8, expected_error: u32) {
            let mut h = broken(first, how);
            let result = h.cpu.deliver_event(first, Some(0x55), false);
            let context = format!("{first} then {} ({how:?})", nested_vector(how));
            assert!(result.is_none(), "{context}: {result:?}");
            assert_eq!(h.cpu.rip, handler_for(expected_vector), "{context}");
            assert_eq!(pushed_error_code(&h), expected_error as u64, "{context}");
        }

        #[test]
        fn contributory_followed_by_contributory_is_a_double_fault() {
            for first in [0u8, 10, 11, 12, 13] {
                for how in ALL_BREAKS {
                    let nested = nested_vector(how);
                    if nested == VECTOR_PAGE_FAULT {
                        continue;
                    }
                    check_pair(first, how, VECTOR_DOUBLE_FAULT, 0);
                }
            }
        }

        #[test]
        fn contributory_followed_by_page_fault_is_handled_serially() {
            for first in [0u8, 10, 11, 12, 13] {
                check_pair(
                    first,
                    Break::IstUnmapped,
                    VECTOR_PAGE_FAULT,
                    nested_error_code(Break::IstUnmapped, first),
                );
            }
        }

        #[test]
        fn page_fault_followed_by_anything_that_can_fault_is_a_double_fault() {
            for how in ALL_BREAKS {
                check_pair(VECTOR_PAGE_FAULT, how, VECTOR_DOUBLE_FAULT, 0);
            }
        }

        #[test]
        fn benign_exceptions_are_followed_serially_by_whatever_faults() {
            for first in [1u8, 3, 4, 6, 7, 9, 16, 32, 128, 255] {
                for how in ALL_BREAKS {
                    check_pair(
                        first,
                        how,
                        nested_vector(how),
                        nested_error_code(how, first),
                    );
                }
            }
        }

        #[test]
        fn page_fault_error_code_and_cr2_of_the_nested_fault_are_delivered() {
            let mut h = broken(VECTOR_GENERAL_PROTECTION, Break::IstUnmapped);
            h.cpu
                .deliver_event(VECTOR_GENERAL_PROTECTION, Some(0), false);
            assert_eq!(h.cpu.rip, handler_for(VECTOR_PAGE_FAULT));
            assert_eq!(h.cpu.cr2, aligned_below(HOLE + 0x1000, WORD));
        }

        #[test]
        fn fault_while_delivering_a_double_fault_is_a_triple_fault() {
            for how in ALL_BREAKS {
                let mut h = broken(VECTOR_DOUBLE_FAULT, how);
                assert!(
                    matches!(
                        h.cpu.deliver_event(VECTOR_DOUBLE_FAULT, Some(0), false),
                        Some(ExitReason::TripleFault)
                    ),
                    "{how:?}"
                );
            }
        }

        #[test]
        fn double_fault_that_delivers_cleanly_is_not_a_triple_fault() {
            let mut h = base();
            assert!(
                h.cpu
                    .deliver_event(VECTOR_DOUBLE_FAULT, Some(0), false)
                    .is_none()
            );
            assert_eq!(h.cpu.rip, handler_for(VECTOR_DOUBLE_FAULT));
        }

        #[test]
        fn contributory_pair_whose_double_fault_also_fails_is_a_triple_fault() {
            let mut h = broken(VECTOR_GENERAL_PROTECTION, Break::NotPresent);
            set_gate(
                &mut h,
                VECTOR_DOUBLE_FAULT,
                Gate {
                    present: false,
                    ..Gate::interrupt(handler_for(VECTOR_DOUBLE_FAULT))
                },
            );
            assert!(matches!(
                h.cpu
                    .deliver_event(VECTOR_GENERAL_PROTECTION, Some(0), false),
                Some(ExitReason::TripleFault)
            ));
        }

        #[test]
        fn unreadable_idt_ends_in_a_triple_fault() {
            let mut h = base();
            unmap_boot_identity_2m(&mut h, HOLE);
            h.cpu.idtr.base = HOLE;
            assert!(matches!(
                h.cpu.deliver_event(VECTOR_INVALID_OPCODE, None, false),
                Some(ExitReason::TripleFault)
            ));
        }

        #[test]
        fn empty_idt_ends_in_a_triple_fault() {
            let mut h = base();
            h.cpu.idtr.limit = 0;
            assert!(matches!(
                h.cpu.deliver_event(VECTOR_INVALID_OPCODE, None, false),
                Some(ExitReason::TripleFault)
            ));
        }

        #[test]
        fn nested_fault_clears_the_software_flag_for_later_attempts() {
            // `int 0x80` from ring 3 through a DPL0 gate is #GP; that #GP is then delivered as a
            // hardware exception, which ignores the gate DPL of vector 13.
            let mut h = user_base();
            assert!(h.cpu.deliver_event(0x80, None, true).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
            assert_eq!(pushed_error_code(&h), idt_error_code(0x80) as u64);
            assert_eq!(h.cpu.cpl, 0);
        }

        #[test]
        fn software_interrupt_is_not_classified_as_a_contributory_exception() {
            // `int 13` through a not-present gate: the #NP is handled serially because INT n is not
            // an exception (QEMU's check_exception is only run for non-INT events).
            let mut h = broken(VECTOR_GENERAL_PROTECTION, Break::NotPresent);
            assert!(
                h.cpu
                    .deliver_event(VECTOR_GENERAL_PROTECTION, None, true)
                    .is_none()
            );
            assert_eq!(h.cpu.rip, handler_for(VECTOR_SEGMENT_NOT_PRESENT));
        }

        #[test]
        fn hardware_interrupt_is_not_classified_as_a_contributory_exception() {
            let mut h = broken(VECTOR_STACK_FAULT, Break::NotPresent);
            assert!(
                h.cpu
                    .deliver_hardware_interrupt(VECTOR_STACK_FAULT)
                    .is_none()
            );
            assert_eq!(h.cpu.rip, handler_for(VECTOR_SEGMENT_NOT_PRESENT));
        }

        #[test]
        fn hardware_interrupt_whose_fault_chain_continues_follows_the_exception_rules() {
            // The #NP raised for the interrupt is an exception: when its own gate is not present
            // either, contributory + contributory is a double fault.
            let mut h = broken(VECTOR_STACK_FAULT, Break::NotPresent);
            set_gate(
                &mut h,
                VECTOR_SEGMENT_NOT_PRESENT,
                Gate {
                    present: false,
                    ..Gate::interrupt(handler_for(VECTOR_SEGMENT_NOT_PRESENT))
                },
            );
            assert!(
                h.cpu
                    .deliver_hardware_interrupt(VECTOR_STACK_FAULT)
                    .is_none()
            );
            assert_eq!(h.cpu.rip, handler_for(VECTOR_DOUBLE_FAULT));
        }

        #[test]
        fn hardware_interrupt_delivers_without_an_error_code() {
            let mut h = base();
            assert!(h.cpu.deliver_hardware_interrupt(0x30).is_none());
            assert_eq!(h.cpu.rip, handler_for(0x30));
            assert_eq!(h.cpu.gpr[GPR_RSP], aligned_below(STACK_TOP, FRAME_BYTES));
        }

        #[test]
        fn longest_legal_fault_chain_needs_all_four_delivery_attempts() {
            // int3's gate is not present (#NP, serial after a benign event); the #NP gate's stack
            // is unmapped (#PF, serial after a contributory fault); the #PF gate is invalid (#GP,
            // which after a #PF is a double fault); the #DF gate finally works.
            let mut h = broken(3, Break::NotPresent);
            let page_fault_gate = Gate::interrupt(handler_for(VECTOR_PAGE_FAULT));
            set_gate(
                &mut h,
                VECTOR_PAGE_FAULT,
                Gate {
                    gate_type: 0x5,
                    ..page_fault_gate
                },
            );
            set_gate(
                &mut h,
                VECTOR_SEGMENT_NOT_PRESENT,
                Gate {
                    ist: 1,
                    ..Gate::interrupt(handler_for(VECTOR_SEGMENT_NOT_PRESENT))
                },
            );
            install_tss(&mut h, [RSP0, 0, 0], [HOLE + 0x1000, 0, 0, 0, 0, 0, 0]);
            assert!(h.cpu.deliver_event(3, None, false).is_none());
            assert_eq!(h.cpu.rip, handler_for(VECTOR_DOUBLE_FAULT));
        }

        #[test]
        fn the_nested_fault_replaces_the_original_error_code() {
            // Benign first vector 3 with a broken gate: the #GP's code, not the original, is pushed.
            let mut h = broken(3, Break::InvalidType);
            h.cpu.deliver_event(3, Some(0x1234), false);
            assert_eq!(h.cpu.rip, handler_for(VECTOR_GENERAL_PROTECTION));
            assert_eq!(pushed_error_code(&h), idt_error_code(3) as u64);
        }

        #[test]
        fn nested_fault_frame_records_the_original_interrupted_context() {
            let mut h = broken(3, Break::NotPresent);
            h.cpu.deliver_event(3, None, false);
            let frame = read_frame(&h, h.cpu.gpr[GPR_RSP], true);
            assert_eq!(frame.rip, OLD_RIP);
            assert_eq!(frame.rsp, STACK_TOP);
        }
    }
}
