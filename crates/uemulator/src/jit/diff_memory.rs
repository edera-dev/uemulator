//! Memory access paths: faults at every offset of a page, accesses that straddle a page
//! boundary, non-canonical and unbacked addresses, read-only pages, stack faults, and TLB refills.
//! The expected faults are derived from the page layout below (the architecture's rules), not
//! from what the interpreter returns; the JIT must also agree with the interpreter on the exact
//! state the fault leaves behind.

use super::diff_tests::*;
use crate::testkit::*;

const PAGE_BYTES: u64 = 4096;
const PTE_PRESENT: u64 = 1;
const PTE_WRITABLE: u64 = 2;
const CR0_WRITE_PROTECT: u64 = 1 << 16;
/// A page table (free RAM) that maps the 2 MiB around `SCRATCH` with 4 KiB pages, so single pages
/// can be made absent or read-only. `boot::load` identity-maps everything with 2 MiB pages.
const PAGE_TABLE: u64 = 0x30_0000;
/// `boot::load` lays out PML4, PDPT and the first page directory back to back from `cr3`.
const FIRST_PAGE_DIRECTORY_OFFSET: u64 = 2 * PAGE_BYTES;
const PDE_BYTES: u64 = 8;
const HUGE_PAGE_SHIFT: u32 = 21;

/// Five consecutive pages of `SCRATCH` with different permissions, in address order.
const RW_BEFORE_UNMAPPED: u64 = SCRATCH + 0x10_000;
const UNMAPPED: u64 = SCRATCH + 0x11_000;
const RW_AFTER_UNMAPPED: u64 = SCRATCH + 0x12_000;
const READ_ONLY: u64 = SCRATCH + 0x13_000;
const RW_AFTER_READ_ONLY: u64 = SCRATCH + 0x14_000;
const LAYOUT_PAGES: [u64; 5] = [
    RW_BEFORE_UNMAPPED,
    UNMAPPED,
    RW_AFTER_UNMAPPED,
    READ_ONLY,
    RW_AFTER_READ_ONLY,
];

/// Page offsets worth trying: the start, an aligned qword inside, and every position within 8
/// bytes of the end, where 2/4/8-byte accesses straddle into the next page.
const PAGE_OFFSETS: [u64; 8] = [0, 8, 0xff8, 0xffb, 0xffc, 0xffd, 0xffe, 0xfff];

fn pte_address(linear: u64) -> usize {
    (PAGE_TABLE + (linear - (SCRATCH & !((1 << HUGE_PAGE_SHIFT) - 1))) / PAGE_BYTES * 8) as usize
}

/// Replaces the 2 MiB mapping that covers `SCRATCH` by a page table with identical translations,
/// then edits individual entries.
fn map_scratch_with_small_pages(harness: &mut Harness) {
    let region_base = SCRATCH & !((1 << HUGE_PAGE_SHIFT) - 1);
    for page in 0..PAGE_BYTES / 8 {
        let entry = (region_base + page * PAGE_BYTES) | PTE_PRESENT | PTE_WRITABLE;
        harness.write_u64(PAGE_TABLE + page * 8, entry);
    }
    let directory_entry =
        harness.cpu.cr3 + FIRST_PAGE_DIRECTORY_OFFSET + (SCRATCH >> HUGE_PAGE_SHIFT) * PDE_BYTES;
    harness.write_u64(directory_entry, PAGE_TABLE | PTE_PRESENT | PTE_WRITABLE);
}

fn set_page_flags(harness: &mut Harness, linear: u64, flags: u64) {
    let entry = (linear & !(PAGE_BYTES - 1)) | flags;
    harness.cpu.ram[pte_address(linear)..][..8].copy_from_slice(&entry.to_le_bytes());
}

/// Seeds registers and memory, maps the layout pages, and gives each page distinct contents.
fn layout_setup(harness: &mut Harness) {
    seed_registers_and_memory(harness);
    map_scratch_with_small_pages(harness);
    set_page_flags(harness, UNMAPPED, 0);
    set_page_flags(harness, READ_ONLY, PTE_PRESENT);
    harness.cpu.cr0 |= CR0_WRITE_PROTECT;
    for (index, page) in LAYOUT_PAGES.iter().enumerate() {
        let pattern: Vec<u8> = (0..PAGE_BYTES)
            .map(|byte| {
                (byte as u8)
                    .wrapping_mul(7)
                    .wrapping_add(index as u8 * 0x31)
            })
            .collect();
        harness.write(*page, &pattern);
    }
    harness.cpu.flush_tlb();
}

#[derive(Clone, Copy, PartialEq)]
enum Permission {
    ReadWrite,
    Absent,
    ReadOnly,
}

fn permission(linear: u64) -> Permission {
    match linear & !(PAGE_BYTES - 1) {
        UNMAPPED => Permission::Absent,
        READ_ONLY => Permission::ReadOnly,
        _ => Permission::ReadWrite,
    }
}

/// The page fault the architecture raises for a `size`-byte access at `address`: error code and
/// CR2. The pages are checked in address order and a straddling access reports the first page
/// that fails, with CR2 at the first byte of that page (or the access address on the first).
fn expected_page_fault(address: u64, size: u64, writes: bool) -> Option<(u32, u64)> {
    let first_page = address & !(PAGE_BYTES - 1);
    let last_page = (address + size - 1) & !(PAGE_BYTES - 1);
    for page in [first_page, last_page] {
        let faulting_address = address.max(page);
        match permission(page) {
            Permission::Absent => return Some((if writes { 2 } else { 0 }, faulting_address)),
            Permission::ReadOnly if writes => return Some((3, faulting_address)),
            _ => {}
        }
        if first_page == last_page {
            break;
        }
    }
    None
}

struct Access {
    code: &'static [u8],
    /// Bytes of `code` before the instruction that touches memory (setup instructions).
    leading_bytes: u64,
    size: u64,
    writes: bool,
}

/// Runs prelude, the access with `rsi = address`, epilogue on both engines and checks the
/// outcome against the page layout.
#[track_caller]
fn check_access(access: &Access, address: u64) {
    let code = [&PRELUDE[..], access.code, &EPILOGUE[..]].concat();
    let result = run_checked(&code, |harness| {
        layout_setup(harness);
        harness.cpu.gpr[RSI] = address;
    });
    let fault_expectation = expected_page_fault(address, access.size, access.writes);
    let context = format!(
        "access {:02x?} at {address:#x}: {}",
        access.code,
        result.interpreter.cpu.dump_regs()
    );
    let interpreter = &result.interpreter;
    match fault_expectation {
        None => {
            assert_eq!(result.interpreter_stop, Stop::Halted, "{context}");
            assert_eq!(interpreter.cpu.gpr[R11], SEED_GPRS_R11 + 1, "epilogue ran");
        }
        Some((error_code, cr2)) => {
            assert_eq!(
                result.interpreter_stop,
                fault(14, Some(error_code)),
                "{context}"
            );
            assert_eq!(interpreter.cpu.cr2, cr2, "{context}");
            assert_eq!(
                interpreter.cpu.rip,
                CODE_BASE + PRELUDE.len() as u64 + access.leading_bytes,
                "rip stays on the faulting instruction: {context}"
            );
            assert_eq!(interpreter.cpu.gpr[R9], SEED_GPRS_R9 + 1, "prelude ran");
            assert_eq!(
                interpreter.cpu.gpr[R11], SEED_GPRS_R11,
                "epilogue did not run"
            );
            if access.writes {
                let mut untouched = Harness::new(&code);
                layout_setup(&mut untouched);
                for page in LAYOUT_PAGES {
                    assert_eq!(
                        interpreter.read(page, PAGE_BYTES as usize),
                        untouched.read(page, PAGE_BYTES as usize),
                        "a faulting store must not modify memory (page {page:#x}): {context}"
                    );
                }
            }
        }
    }
}

const SEED_GPRS_R9: u64 = 1;
const SEED_GPRS_R11: u64 = 0x8000_0000;

fn all_layout_addresses() -> Vec<u64> {
    LAYOUT_PAGES
        .iter()
        .flat_map(|page| PAGE_OFFSETS.iter().map(move |offset| page + offset))
        .collect()
}

macro_rules! access_tests {
    ($($name:ident => ($code:expr, $leading:expr, $size:expr, $writes:expr),)*) => {
        $(#[test]
        fn $name() {
            let access = Access { code: &$code, leading_bytes: $leading, size: $size, writes: $writes };
            for address in all_layout_addresses() {
                check_access(&access, address);
            }
        })*
    };
}

access_tests! {
    load_byte_at_every_page_offset => (LOAD_BYTE, 0, 1, false),
    load_word_at_every_page_offset => (LOAD_WORD, 0, 2, false),
    load_dword_at_every_page_offset => (LOAD_DWORD, 0, 4, false),
    load_qword_at_every_page_offset => (LOAD_QWORD, 0, 8, false),
    movzx_word_load_at_every_page_offset => (MOVZX_WORD, 0, 2, false),
    movsx_byte_load_at_every_page_offset => (MOVSX_BYTE, 0, 1, false),
    add_from_memory_at_every_page_offset => (ADD_FROM_MEMORY_QWORD, 0, 8, false),
    cmp_memory_operand_at_every_page_offset => (CMP_MEMORY_DWORD, 0, 4, false),
    test_memory_operand_at_every_page_offset => (TEST_MEMORY_BYTE, 0, 1, false),
    high_byte_load_at_every_page_offset => (MOVZX_FROM_MEMORY_BYTE_TO_HIGH_REGISTER, 0, 1, false),
    store_byte_at_every_page_offset => (STORE_BYTE, 0, 1, true),
    store_word_at_every_page_offset => (STORE_WORD, 0, 2, true),
    store_dword_at_every_page_offset => (STORE_DWORD, 0, 4, true),
    store_qword_at_every_page_offset => (STORE_QWORD, 0, 8, true),
    store_immediate_dword_at_every_page_offset => (STORE_IMM_DWORD, 0, 4, true),
    store_immediate_qword_at_every_page_offset => (STORE_IMM_QWORD, 0, 8, true),
    store_high_byte_register_at_every_page_offset => (STORE_HIGH_BYTE_REGISTER, 0, 1, true),
    stosq_at_every_page_offset => (STOSQ_TO_RSI, 3, 8, true),
    setcc_to_memory_at_every_page_offset => (SETE_TO_MEMORY, 0, 1, true),
    pop_to_memory_at_every_page_offset => (POP_MEMORY, 0, 8, true),
    push_memory_at_every_page_offset => (PUSH_MEMORY_QWORD, 0, 8, false),
    cmov_from_memory_at_every_page_offset => (CMOVNE_FROM_MEMORY_DWORD, 0, 4, false),
    imul_from_memory_at_every_page_offset => (IMUL_FROM_MEMORY_DWORD, 0, 4, false),
    imul_immediate_from_memory_at_every_page_offset => (IMUL_IMMEDIATE_FROM_MEMORY_QWORD, 0, 8, false),
    test_immediate_memory_byte_at_every_page_offset => (TEST_IMMEDIATE_MEMORY_BYTE, 0, 1, false),
}

/// Read-modify-write instructions: the access is a write for permission purposes, and a straddling
/// one must not leave half of its result behind. Only the engines' agreement is checked, plus
/// that the fault is a page fault exactly when some page is absent or read-only.
macro_rules! read_modify_write_tests {
    ($($name:ident => ($code:expr, $size:expr),)*) => {
        $(#[test]
        fn $name() {
            for address in all_layout_addresses() {
                let code = [&PRELUDE[..], &$code[..], &EPILOGUE[..]].concat();
                let result = run_checked(&code, |harness| {
                    layout_setup(harness);
                    harness.cpu.gpr[RSI] = address;
                });
                let faults = expected_page_fault(address, $size, true).is_some();
                match result.interpreter_stop {
                    Stop::Halted => assert!(!faults, "{address:#x} should fault"),
                    Stop::Fault { vector: 14, .. } => assert!(faults, "{address:#x} must not fault"),
                    ref other => panic!("{address:#x}: {other:?}"),
                }
            }
        })*
    };
}

read_modify_write_tests! {
    add_to_memory_at_every_page_offset => (ADD_TO_MEMORY_QWORD, 8),
    inc_memory_at_every_page_offset => (INC_MEMORY_DWORD, 4),
    xchg_memory_at_every_page_offset => (XCHG_MEMORY_DWORD, 4),
    cmpxchg_memory_at_every_page_offset => (CMPXCHG_MEMORY_QWORD, 8),
    lock_add_memory_at_every_page_offset => (LOCK_ADD_MEMORY_WORD, 2),
    shl_memory_at_every_page_offset => (SHL_MEMORY_QWORD, 8),
    adc_memory_at_every_page_offset => (ADC_MEMORY_DWORD, 4),
    neg_memory_at_every_page_offset => (NEG_MEMORY_WORD, 2),
    not_memory_at_every_page_offset => (NOT_MEMORY_BYTE, 1),
    bts_memory_at_every_page_offset => (BTS_MEMORY_IMM, 8),
    add_register_to_memory_byte_at_every_page_offset => (ADD_REGISTER_TO_MEMORY_BYTE, 1),
    add_high_register_to_memory_byte_at_every_page_offset => (ADD_HIGH_REGISTER_TO_MEMORY_BYTE, 1),
    or_immediate_to_memory_word_at_every_page_offset => (OR_IMMEDIATE_TO_MEMORY_WORD, 2),
    xor_register_to_memory_word_at_every_page_offset => (XOR_REGISTER_TO_MEMORY_WORD, 2),
    sub_immediate_from_memory_dword_at_every_page_offset => (SUB_IMMEDIATE_FROM_MEMORY_DWORD, 4),
    and_immediate_to_memory_byte_at_every_page_offset => (AND_IMMEDIATE_TO_MEMORY_BYTE, 1),
    inc_memory_byte_at_every_page_offset => (INC_MEMORY_BYTE, 1),
    dec_memory_word_at_every_page_offset => (DEC_MEMORY_WORD, 2),
    shr_immediate_memory_dword_at_every_page_offset => (SHR_IMMEDIATE_MEMORY_DWORD, 4),
    sar_cl_memory_qword_at_every_page_offset => (SAR_CL_MEMORY_QWORD, 8),
    neg_memory_dword_at_every_page_offset => (NEG_MEMORY_DWORD, 4),
    not_memory_qword_at_every_page_offset => (NOT_MEMORY_QWORD, 8),
}

/// cmpxchg8b: the engines must agree at every offset. Whether a failed comparison still counts as
/// a write for permission purposes is not asserted (see the report notes).
#[test]
fn cmpxchg8b_at_every_page_offset_agrees_between_engines() {
    for address in all_layout_addresses() {
        let code = [&PRELUDE[..], &CMPXCHG8B_MEMORY[..], &EPILOGUE[..]].concat();
        run_checked(&code, |harness| {
            layout_setup(harness);
            harness.cpu.gpr[RSI] = address;
        });
    }
}

#[test]
fn a_write_to_a_read_only_page_succeeds_when_cr0_wp_is_clear() {
    let code = [&PRELUDE[..], &STORE_QWORD[..], &EPILOGUE[..]].concat();
    let result = check_with(&code, |harness| {
        layout_setup(harness);
        harness.cpu.cr0 &= !CR0_WRITE_PROTECT;
        harness.cpu.gpr[RSI] = READ_ONLY + 0x20;
    });
    assert_eq!(
        result.interpreter.read_u64(READ_ONLY + 0x20),
        SEED_RAX,
        "supervisor writes ignore the read-only bit without CR0.WP"
    );
}

const SEED_RAX: u64 = 0x8899_aabb_ccdd_eeff;

/// The WebAssembly generator keeps the last flag writer's record pending and stores it only on
/// the way out (`emit_wasm::Gen::pending_record`). A read-modify-write whose load succeeds and
/// whose store faults computes its result into the locals that pending record lives in, so the
/// exception must still see the flags from before it: the cmp's (CF and SF set, ZF clear).
#[test]
fn a_store_that_faults_after_its_load_leaves_the_flags_of_the_instruction_before() {
    // mov rcx, [rsi] (warms the TLB entry the load hits); cmp rax, rbx; <op>; hlt
    const WARM_THEN_CMP: [u8; 6] = [0x48, 0x8b, 0x0e, 0x48, 0x39, 0xd8];
    let forms: [&[u8]; 6] = [
        &[0x48, 0x01, 0x0e], // add [rsi], rcx
        &[0x48, 0xff, 0x06], // inc qword ptr [rsi]
        &[0x48, 0xf7, 0x1e], // neg qword ptr [rsi]
        &[0x83, 0x2e, 0x05], // sub dword ptr [rsi], 5
        &[0x08, 0x06],       // or byte ptr [rsi], al
        &[0x48, 0x89, 0x0e], // mov [rsi], rcx: no load, the store faults straight away
    ];
    for form in forms {
        let code = [&WARM_THEN_CMP[..], form, &[0xf4]].concat();
        check_stop_with(
            &code,
            |harness| {
                layout_setup(harness);
                harness.cpu.gpr[RSI] = READ_ONLY + 0x30;
                harness.cpu.gpr[RAX] = 1;
                harness.cpu.gpr[RBX] = 2;
            },
            fault(14, Some(PF_PRESENT | PF_WRITE)),
        );
    }
}

/// #PF error code bits (SDM 4.7): the page was present, the access was a write.
const PF_PRESENT: u32 = 1;
const PF_WRITE: u32 = 2;

#[test]
fn a_read_only_page_can_be_read_and_the_data_matches_memory() {
    let code = [&PRELUDE[..], &LOAD_QWORD[..], &EPILOGUE[..]].concat();
    let result = check_with(&code, |harness| {
        layout_setup(harness);
        harness.cpu.gpr[RSI] = READ_ONLY + 0x40;
    });
    let expected = u64::from_le_bytes(
        (0..8u64)
            .map(|byte| ((0x40 + byte) as u8).wrapping_mul(7).wrapping_add(3 * 0x31))
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap(),
    );
    assert_eq!(result.interpreter.cpu.gpr[RAX], expected);
}

/// Addresses outside the 48-bit canonical range raise #GP(0) for every kind of access and leave
/// CR2 alone.
const NON_CANONICAL_ADDRESSES: [u64; 5] = [
    0x0000_8000_0000_0000,
    0xffff_7fff_ffff_ffff,
    0x8000_0000_0000_0000,
    0x1234_5678_9abc_def0,
    0x0001_0000_0000_0000,
];

macro_rules! non_canonical_tests {
    ($($name:ident => $code:expr,)*) => {
        $(#[test]
        fn $name() {
            for address in NON_CANONICAL_ADDRESSES {
                let code = [&PRELUDE[..], &$code[..], &EPILOGUE[..]].concat();
                let result = check_stop_with(
                    &code,
                    |harness| {
                        seed_registers_and_memory(harness);
                        harness.cpu.gpr[RSI] = address;
                    },
                    fault(13, Some(0)),
                );
                assert_eq!(result.interpreter.cpu.cr2, 0, "#GP does not set CR2");
                assert_eq!(result.interpreter.cpu.rip, CODE_BASE + PRELUDE.len() as u64);
            }
        })*
    };
}

non_canonical_tests! {
    non_canonical_byte_load_raises_general_protection => LOAD_BYTE,
    non_canonical_qword_load_raises_general_protection => LOAD_QWORD,
    non_canonical_qword_store_raises_general_protection => STORE_QWORD,
    non_canonical_byte_store_raises_general_protection => STORE_BYTE,
    non_canonical_read_modify_write_raises_general_protection => ADD_TO_MEMORY_QWORD,
    non_canonical_cmp_raises_general_protection => CMP_MEMORY_DWORD,
    non_canonical_push_memory_raises_general_protection => PUSH_MEMORY,
    non_canonical_indirect_call_raises_general_protection => CALL_INDIRECT_MEMORY,
    non_canonical_movsx_raises_general_protection => MOVSX_BYTE,
}

#[test]
fn an_access_straddling_the_canonical_boundary_faults() {
    // The last canonical page is not mapped, so the first page already fails.
    let code = [&PRELUDE[..], &LOAD_QWORD[..], &EPILOGUE[..]].concat();
    for address in [0x0000_7fff_ffff_fffc_u64, 0xffff_8000_0000_0000 - 4] {
        let result = run_checked(&code, |harness| {
            seed_registers_and_memory(harness);
            harness.cpu.gpr[RSI] = address;
        });
        assert!(
            matches!(
                result.interpreter_stop,
                Stop::Fault {
                    vector: 13 | 14,
                    ..
                }
            ),
            "{address:#x}: {:?}",
            result.interpreter_stop
        );
    }
}

#[test]
fn addresses_beyond_the_four_gib_identity_map_raise_page_faults() {
    for (access, writes) in [(&LOAD_QWORD[..], false), (&STORE_DWORD[..], true)] {
        let code = [&PRELUDE[..], access, &EPILOGUE[..]].concat();
        let address = 0x1_0000_0000_u64;
        let result = check_stop_with(
            &code,
            |harness| {
                seed_registers_and_memory(harness);
                harness.cpu.gpr[RSI] = address;
            },
            fault(14, Some(if writes { 2 } else { 0 })),
        );
        assert_eq!(result.interpreter.cpu.cr2, address);
    }
}

#[test]
fn an_access_straddling_the_four_gib_boundary_faults_on_the_second_page() {
    // The last identity-mapped page has no RAM behind it (reads return ones, writes vanish), so
    // the first page is fine and the second page is not present.
    let code = [&PRELUDE[..], &LOAD_QWORD[..], &EPILOGUE[..]].concat();
    let result = check_stop_with(
        &code,
        |harness| {
            seed_registers_and_memory(harness);
            harness.cpu.gpr[RSI] = 0xffff_fffc;
        },
        fault(14, Some(0)),
    );
    assert_eq!(result.interpreter.cpu.cr2, 0x1_0000_0000);
}

/// Physical addresses with no RAM behind them (identity-mapped): loads must give the same values
/// on both engines and stores must have the same (device or no) effect.
#[test]
fn accesses_to_unbacked_physical_addresses_agree() {
    for address in [
        0x2000_0000_u64,
        0xfee0_0000,
        0xfed0_0000,
        0x1ff_fff8,
        0x1ff_fffc,
        0x1ff_fffd,
        0x1ff_ffff,
        0x200_0000,
    ] {
        for access in [
            &LOAD_BYTE[..],
            &LOAD_WORD,
            &LOAD_DWORD,
            &LOAD_QWORD,
            &STORE_DWORD,
            &STORE_QWORD,
            &CMP_MEMORY_DWORD,
            &ADD_TO_MEMORY_QWORD,
        ] {
            let code = [&PRELUDE[..], access, &EPILOGUE[..]].concat();
            run_checked(&code, |harness| {
                seed_registers_and_memory(harness);
                harness.cpu.gpr[RSI] = address;
            });
        }
    }
}

macro_rules! stack_fault_tests {
    ($($name:ident => ($code:expr, $stack_pointer:expr, $error_code:expr, $cr2:expr),)*) => {
        $(#[test]
        fn $name() {
            let code = [&PRELUDE[..], &$code[..], &EPILOGUE[..]].concat();
            let stack_pointer: u64 = $stack_pointer;
            let result = check_stop_with(
                &code,
                |harness| {
                    layout_setup(harness);
                    harness.cpu.gpr[RSP] = stack_pointer;
                    harness.cpu.gpr[RBP] = stack_pointer;
                    harness.cpu.gpr[RSI] = RW_BEFORE_UNMAPPED + 0x100;
                },
                fault(14, Some($error_code)),
            );
            assert_eq!(result.interpreter.cpu.cr2, $cr2);
            assert_eq!(
                result.interpreter.cpu.gpr[RSP], stack_pointer,
                "rsp is unchanged by a faulting stack access"
            );
            assert_eq!(result.interpreter.cpu.rip, CODE_BASE + PRELUDE.len() as u64);
        })*
    };
}

stack_fault_tests! {
    push_below_an_unmapped_page_faults_with_rsp_unchanged => (PUSH_RAX, UNMAPPED + 8, 2, UNMAPPED),
    push_straddling_into_an_unmapped_page_faults => (PUSH_RAX, UNMAPPED + 4, 2, UNMAPPED),
    pop_from_an_unmapped_page_faults_with_rsp_unchanged => (POP_RAX, UNMAPPED, 0, UNMAPPED),
    pop_straddling_into_an_unmapped_page_faults => (POP_RAX, UNMAPPED - 4, 0, UNMAPPED),
    call_with_an_unwritable_stack_faults_before_changing_rsp => (CALL_REL32_TO_NEXT, UNMAPPED + 8, 2, UNMAPPED),
    ret_from_an_unmapped_stack_faults_with_rsp_unchanged => (RET_NEAR, UNMAPPED, 0, UNMAPPED),
    ret_imm16_from_an_unmapped_stack_faults_with_rsp_unchanged => (RET_IMM16, UNMAPPED, 0, UNMAPPED),
    pushfq_to_an_unwritable_stack_faults => (PUSHFQ, UNMAPPED + 8, 2, UNMAPPED),
    popfq_from_an_unmapped_stack_faults => (POPFQ, UNMAPPED, 0, UNMAPPED),
    push_to_a_read_only_stack_faults => (PUSH_RAX, READ_ONLY + 8, 3, READ_ONLY),
    push_straddling_into_a_read_only_page_faults => (PUSH_RAX, READ_ONLY + 4, 3, READ_ONLY),
    push_memory_operand_with_unmapped_stack_faults_before_rsp_changes => (PUSH_MEMORY, UNMAPPED + 8, 2, UNMAPPED),
}

#[test]
fn leave_with_an_unmapped_frame_pointer_keeps_rbp() {
    let code = [&PRELUDE[..], &LEAVE[..], &EPILOGUE[..]].concat();
    let result = check_stop_with(
        &code,
        |harness| {
            layout_setup(harness);
            harness.cpu.gpr[RBP] = UNMAPPED + 0x10;
        },
        fault(14, Some(0)),
    );
    assert_eq!(result.interpreter.cpu.gpr[RBP], UNMAPPED + 0x10);
}

#[test]
fn pop_to_a_faulting_memory_destination_leaves_rsp_unchanged() {
    let code = [&PRELUDE[..], &POP_MEMORY[..], &EPILOGUE[..]].concat();
    let stack_pointer = RW_AFTER_UNMAPPED + 0x100;
    let result = check_stop_with(
        &code,
        |harness| {
            layout_setup(harness);
            harness.cpu.gpr[RSP] = stack_pointer;
            harness.cpu.gpr[RSI] = UNMAPPED + 0x10;
        },
        fault(14, Some(2)),
    );
    assert_eq!(result.interpreter.cpu.gpr[RSP], stack_pointer);
}

#[test]
fn faulting_load_leaves_the_destination_register_untouched() {
    for load in [
        &LOAD_BYTE[..],
        &LOAD_WORD,
        &LOAD_DWORD,
        &LOAD_QWORD,
        &MOVSX_BYTE,
        &MOVZX_WORD,
    ] {
        let code = [&PRELUDE[..], load, &EPILOGUE[..]].concat();
        let result = check_stop_with(
            &code,
            |harness| {
                layout_setup(harness);
                harness.cpu.gpr[RSI] = UNMAPPED + 0x10;
            },
            fault(14, Some(0)),
        );
        assert_eq!(result.interpreter.cpu.gpr[RAX], SEED_RAX);
    }
}

#[test]
fn a_fault_after_inline_instructions_reports_their_effects_and_count() {
    // Three inline instructions retire, then the load faults: the engines must agree on the
    // instruction count too (Snapshot.icount), which counts the faulting instruction.
    let code = [
        &PRELUDE[..],
        &[0x48, 0x01, 0xc8][..],
        &LOAD_QWORD[..],
        &EPILOGUE[..],
    ]
    .concat();
    let result = check_stop_with(
        &code,
        |harness| {
            layout_setup(harness);
            harness.cpu.gpr[RSI] = UNMAPPED;
        },
        fault(14, Some(0)),
    );
    assert_eq!(result.interpreter.cpu.icount, 4);
}

/// The TLB has `TLB_ENTRIES_PER_MODE` (1024) entries per privilege level. The sweeps visit
/// 496 + 752 = 1248 distinct pages of free RAM (away from the page tables below 1 MiB, the fake
/// kernel at 16 MiB and the stack below 8 MiB), so every slot is refilled at least once.
const _: () = assert!(496 + 752 > crate::cpu::TLB_ENTRIES_PER_MODE);

#[test]
fn loads_sweeping_more_pages_than_the_tlb_holds_agree() {
    check(&SWEEP_LOAD_LOW);
    check(&SWEEP_LOAD_HIGH);
}

#[test]
fn stores_sweeping_more_pages_than_the_tlb_holds_agree() {
    check(&SWEEP_STORE_LOW);
    check(&SWEEP_STORE_HIGH);
}

#[test]
fn read_modify_write_sweeping_more_pages_than_the_tlb_holds_agree() {
    check(&SWEEP_READ_MODIFY_WRITE_HIGH);
}

#[test]
fn values_stored_through_a_full_tlb_sweep_read_back_the_same() {
    let result = check(&SWEEP_STORE_THEN_LOAD_BACK);
    // sum of rcx over 752..=1
    assert_eq!(result.interpreter.cpu.gpr[RDX], 752 * 753 / 2);
}

#[test]
fn two_pages_that_share_a_tlb_slot_evict_each_other_correctly() {
    // Pages 4 MiB apart map to the same slot of a 1024-entry TLB.
    check(&TLB_CONFLICT_PING_PONG);
}

#[test]
fn eight_pages_sharing_one_tlb_slot_are_accessed_round_robin() {
    check(&TLB_CONFLICT_EIGHT_WAY);
}

fn remap_setup(harness: &mut Harness) {
    layout_setup(harness);
    harness.cpu.gpr[RSI] = RW_AFTER_UNMAPPED;
    harness.cpu.gpr[RDI] = pte_address(RW_AFTER_UNMAPPED) as u64;
    harness.cpu.gpr[RDX] = READ_ONLY_FRAME | PTE_PRESENT | PTE_WRITABLE;
}

/// Frame the remapped entry is pointed at; its contents differ from the page's original frame.
const READ_ONLY_FRAME: u64 = READ_ONLY;

#[test]
fn invlpg_makes_a_remapped_page_visible() {
    let result = check_with(&READ_REMAP_FLUSH_READ, remap_setup);
    assert_ne!(
        result.interpreter.cpu.gpr[RAX], result.interpreter.cpu.gpr[RCX],
        "the load after invlpg sees the frame the entry now points to"
    );
}

#[test]
fn reloading_cr3_makes_a_remapped_page_visible() {
    let result = check_with(&READ_REMAP_RELOAD_CR3_READ, remap_setup);
    assert_ne!(
        result.interpreter.cpu.gpr[RAX],
        result.interpreter.cpu.gpr[RCX]
    );
}

#[test]
fn a_store_after_invlpg_lands_in_the_new_frame() {
    let result = check_with(&WRITE_THEN_REMAP_AND_FLUSH, remap_setup);
    // The first store went to the old frame, the second one (after the flush) to the new frame.
    assert_eq!(result.interpreter.read_u64(RW_AFTER_UNMAPPED), SEED_RAX);
    assert_eq!(result.interpreter.cpu.gpr[R8], 4);
}

// mov rsi, 0x100000; mov ecx, 496; 1: add rdx, [rsi]; add rsi, 4096; dec ecx; jnz 1b
// hlt
const SWEEP_LOAD_LOW: [u8; 27] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x10, 0x00, 0xb9, 0xf0, 0x01, 0x00, 0x00, 0x48, 0x03, 0x16, 0x48,
    0x81, 0xc6, 0x00, 0x10, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf2, 0xf4,
];

// mov rsi, 0x500000; mov ecx, 752; 1: add rdx, [rsi]; add rsi, 4096; dec ecx; jnz 1b
// hlt
const SWEEP_LOAD_HIGH: [u8; 27] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x50, 0x00, 0xb9, 0xf0, 0x02, 0x00, 0x00, 0x48, 0x03, 0x16, 0x48,
    0x81, 0xc6, 0x00, 0x10, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf2, 0xf4,
];

// mov rsi, 0x100000; mov ecx, 496; mov rdx, 5; 1: mov [rsi], rdx; add rdx, rcx
// add rsi, 4096; dec ecx; jnz 1b; hlt
const SWEEP_STORE_LOW: [u8; 37] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x10, 0x00, 0xb9, 0xf0, 0x01, 0x00, 0x00, 0x48, 0xc7, 0xc2, 0x05,
    0x00, 0x00, 0x00, 0x48, 0x89, 0x16, 0x48, 0x01, 0xca, 0x48, 0x81, 0xc6, 0x00, 0x10, 0x00, 0x00,
    0xff, 0xc9, 0x75, 0xef, 0xf4,
];

// mov rsi, 0x500000; mov ecx, 752; mov rdx, 5; 1: mov [rsi], rdx; add rdx, rcx
// add rsi, 4096; dec ecx; jnz 1b; hlt
const SWEEP_STORE_HIGH: [u8; 37] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x50, 0x00, 0xb9, 0xf0, 0x02, 0x00, 0x00, 0x48, 0xc7, 0xc2, 0x05,
    0x00, 0x00, 0x00, 0x48, 0x89, 0x16, 0x48, 0x01, 0xca, 0x48, 0x81, 0xc6, 0x00, 0x10, 0x00, 0x00,
    0xff, 0xc9, 0x75, 0xef, 0xf4,
];

// mov rsi, 0x500000; mov ecx, 752; 1: add [rsi + 8], rcx; add rsi, 4096; dec ecx
// jnz 1b; hlt
const SWEEP_READ_MODIFY_WRITE_HIGH: [u8; 28] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x50, 0x00, 0xb9, 0xf0, 0x02, 0x00, 0x00, 0x48, 0x01, 0x4e, 0x08,
    0x48, 0x81, 0xc6, 0x00, 0x10, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf1, 0xf4,
];

// mov rsi, 0x500000; mov ecx, 752; 1: mov [rsi + 16], rcx; add rsi, 4096; dec ecx
// jnz 1b; mov rsi, 0x500000; mov ecx, 752; xor edx, edx; 2: add rdx, [rsi + 16]
// add rsi, 4096; dec ecx; jnz 2b; hlt
const SWEEP_STORE_THEN_LOAD_BACK: [u8; 57] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x50, 0x00, 0xb9, 0xf0, 0x02, 0x00, 0x00, 0x48, 0x89, 0x4e, 0x10,
    0x48, 0x81, 0xc6, 0x00, 0x10, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf1, 0x48, 0xc7, 0xc6, 0x00, 0x00,
    0x50, 0x00, 0xb9, 0xf0, 0x02, 0x00, 0x00, 0x31, 0xd2, 0x48, 0x03, 0x56, 0x10, 0x48, 0x81, 0xc6,
    0x00, 0x10, 0x00, 0x00, 0xff, 0xc9, 0x75, 0xf1, 0xf4,
];

// mov rsi, 0x600000; mov rdi, 0xa00000; mov ecx, 400; 1: mov [rsi], rcx
// add rdx, [rdi]; mov [rdi + 8], rcx; add rdx, [rsi + 8]; dec ecx; jnz 1b; hlt
const TLB_CONFLICT_PING_PONG: [u8; 38] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x60, 0x00, 0x48, 0xc7, 0xc7, 0x00, 0x00, 0xa0, 0x00, 0xb9, 0x90,
    0x01, 0x00, 0x00, 0x48, 0x89, 0x0e, 0x48, 0x03, 0x17, 0x48, 0x89, 0x4f, 0x08, 0x48, 0x03, 0x56,
    0x08, 0xff, 0xc9, 0x75, 0xee, 0xf4,
];

// mov rsi, 0x500000; mov r8d, 100; 1: mov ecx, 8; mov rdi, rsi; 2: add rdx, [rdi]
// mov [rdi + 8], rdx; add rdi, 0x400000; dec ecx; jnz 2b; dec r8d; jnz 1b; hlt
const TLB_CONFLICT_EIGHT_WAY: [u8; 45] = [
    0x48, 0xc7, 0xc6, 0x00, 0x00, 0x50, 0x00, 0x41, 0xb8, 0x64, 0x00, 0x00, 0x00, 0xb9, 0x08, 0x00,
    0x00, 0x00, 0x48, 0x89, 0xf7, 0x48, 0x03, 0x17, 0x48, 0x89, 0x57, 0x08, 0x48, 0x81, 0xc7, 0x00,
    0x00, 0x40, 0x00, 0xff, 0xc9, 0x75, 0xee, 0x41, 0xff, 0xc8, 0x75, 0xe1, 0xf4,
];

// mov rax, [rsi]; mov [rdi], rdx; invlpg [rsi]; mov rcx, [rsi]; hlt
const READ_REMAP_FLUSH_READ: [u8; 13] = [
    0x48, 0x8b, 0x06, 0x48, 0x89, 0x17, 0x0f, 0x01, 0x3e, 0x48, 0x8b, 0x0e, 0xf4,
];

// mov rax, [rsi]; mov [rdi], rdx; mov r8, cr3; mov cr3, r8; mov rcx, [rsi]; hlt
const READ_REMAP_RELOAD_CR3_READ: [u8; 18] = [
    0x48, 0x8b, 0x06, 0x48, 0x89, 0x17, 0x41, 0x0f, 0x20, 0xd8, 0x41, 0x0f, 0x22, 0xd8, 0x48, 0x8b,
    0x0e, 0xf4,
];

// mov [rsi], rax; mov [rdi], rdx; invlpg [rsi]; mov [rsi], rcx; mov r8, [rsi]; hlt
const WRITE_THEN_REMAP_AND_FLUSH: [u8; 16] = [
    0x48, 0x89, 0x06, 0x48, 0x89, 0x17, 0x0f, 0x01, 0x3e, 0x48, 0x89, 0x0e, 0x4c, 0x8b, 0x06, 0xf4,
];

// add r9, 1; add r10, 2
const PRELUDE: [u8; 8] = [0x49, 0x83, 0xc1, 0x01, 0x49, 0x83, 0xc2, 0x02];

// add r11, 1; hlt
const EPILOGUE: [u8; 5] = [0x49, 0x83, 0xc3, 0x01, 0xf4];

// mov al, [rsi]
const LOAD_BYTE: [u8; 2] = [0x8a, 0x06];

// mov ax, [rsi]
const LOAD_WORD: [u8; 3] = [0x66, 0x8b, 0x06];

// mov eax, [rsi]
const LOAD_DWORD: [u8; 2] = [0x8b, 0x06];

// mov rax, [rsi]
const LOAD_QWORD: [u8; 3] = [0x48, 0x8b, 0x06];

// movzx eax, word ptr [rsi]
const MOVZX_WORD: [u8; 3] = [0x0f, 0xb7, 0x06];

// movsx rax, byte ptr [rsi]
const MOVSX_BYTE: [u8; 4] = [0x48, 0x0f, 0xbe, 0x06];

// add rax, [rsi]
const ADD_FROM_MEMORY_QWORD: [u8; 3] = [0x48, 0x03, 0x06];

// cmp [rsi], eax
const CMP_MEMORY_DWORD: [u8; 2] = [0x39, 0x06];

// test [rsi], al
const TEST_MEMORY_BYTE: [u8; 2] = [0x84, 0x06];

// mov [rsi], al
const STORE_BYTE: [u8; 2] = [0x88, 0x06];

// mov [rsi], ax
const STORE_WORD: [u8; 3] = [0x66, 0x89, 0x06];

// mov [rsi], eax
const STORE_DWORD: [u8; 2] = [0x89, 0x06];

// mov [rsi], rax
const STORE_QWORD: [u8; 3] = [0x48, 0x89, 0x06];

// mov dword ptr [rsi], 0x11223344
const STORE_IMM_DWORD: [u8; 6] = [0xc7, 0x06, 0x44, 0x33, 0x22, 0x11];

// mov qword ptr [rsi], -2
const STORE_IMM_QWORD: [u8; 7] = [0x48, 0xc7, 0x06, 0xfe, 0xff, 0xff, 0xff];

// add [rsi], rax
const ADD_TO_MEMORY_QWORD: [u8; 3] = [0x48, 0x01, 0x06];

// inc dword ptr [rsi]
const INC_MEMORY_DWORD: [u8; 2] = [0xff, 0x06];

// xchg [rsi], eax
const XCHG_MEMORY_DWORD: [u8; 2] = [0x87, 0x06];

// cmpxchg [rsi], rcx
const CMPXCHG_MEMORY_QWORD: [u8; 4] = [0x48, 0x0f, 0xb1, 0x0e];

// lock add [rsi], ax
const LOCK_ADD_MEMORY_WORD: [u8; 4] = [0x66, 0xf0, 0x01, 0x06];

// shl qword ptr [rsi], 1
const SHL_MEMORY_QWORD: [u8; 3] = [0x48, 0xd1, 0x26];

// mov ah, [rsi]
const MOVZX_FROM_MEMORY_BYTE_TO_HIGH_REGISTER: [u8; 2] = [0x8a, 0x26];

// mov [rsi], ah
const STORE_HIGH_BYTE_REGISTER: [u8; 2] = [0x88, 0x26];

// push rax
const PUSH_RAX: [u8; 1] = [0x50];

// pop rax
const POP_RAX: [u8; 1] = [0x58];

// push qword ptr [rsi]
const PUSH_MEMORY: [u8; 2] = [0xff, 0x36];

// pop qword ptr [rsi]
const POP_MEMORY: [u8; 2] = [0x8f, 0x06];

// call 1f; 1: nop
const CALL_REL32_TO_NEXT: [u8; 6] = [0xe8, 0x00, 0x00, 0x00, 0x00, 0x90];

// ret
const RET_NEAR: [u8; 1] = [0xc3];

// ret 8
const RET_IMM16: [u8; 3] = [0xc2, 0x08, 0x00];

// leave
const LEAVE: [u8; 1] = [0xc9];

// pushfq
const PUSHFQ: [u8; 1] = [0x9c];

// popfq
const POPFQ: [u8; 1] = [0x9d];

// call [rsi]
const CALL_INDIRECT_MEMORY: [u8; 2] = [0xff, 0x16];

// cmpxchg8b [rsi]
const CMPXCHG8B_MEMORY: [u8; 3] = [0x0f, 0xc7, 0x0e];

// mov rdi, rsi; stosq
const STOSQ_TO_RSI: [u8; 5] = [0x48, 0x89, 0xf7, 0x48, 0xab];

// adc [rsi], eax
const ADC_MEMORY_DWORD: [u8; 2] = [0x11, 0x06];

// neg word ptr [rsi]
const NEG_MEMORY_WORD: [u8; 3] = [0x66, 0xf7, 0x1e];

// not byte ptr [rsi]
const NOT_MEMORY_BYTE: [u8; 2] = [0xf6, 0x16];

// bts qword ptr [rsi], 3
const BTS_MEMORY_IMM: [u8; 5] = [0x48, 0x0f, 0xba, 0x2e, 0x03];

// push qword ptr [rsi]
const PUSH_MEMORY_QWORD: [u8; 2] = [0xff, 0x36];
// cmovne eax, [rsi]
const CMOVNE_FROM_MEMORY_DWORD: [u8; 3] = [0x0f, 0x45, 0x06];
// imul eax, [rsi]
const IMUL_FROM_MEMORY_DWORD: [u8; 3] = [0x0f, 0xaf, 0x06];
// imul rax, [rsi], 7
const IMUL_IMMEDIATE_FROM_MEMORY_QWORD: [u8; 4] = [0x48, 0x6b, 0x06, 0x07];
// test byte ptr [rsi], 4
const TEST_IMMEDIATE_MEMORY_BYTE: [u8; 3] = [0xf6, 0x06, 0x04];
// add byte ptr [rsi], al
const ADD_REGISTER_TO_MEMORY_BYTE: [u8; 2] = [0x00, 0x06];
// add [rsi], ah
const ADD_HIGH_REGISTER_TO_MEMORY_BYTE: [u8; 2] = [0x00, 0x26];
// or word ptr [rsi], 0x1234
const OR_IMMEDIATE_TO_MEMORY_WORD: [u8; 5] = [0x66, 0x81, 0x0e, 0x34, 0x12];
// xor [rsi], ax
const XOR_REGISTER_TO_MEMORY_WORD: [u8; 3] = [0x66, 0x31, 0x06];
// sub dword ptr [rsi], 5
const SUB_IMMEDIATE_FROM_MEMORY_DWORD: [u8; 3] = [0x83, 0x2e, 0x05];
// and byte ptr [rsi], 0x7f
const AND_IMMEDIATE_TO_MEMORY_BYTE: [u8; 3] = [0x80, 0x26, 0x7f];
// inc byte ptr [rsi]
const INC_MEMORY_BYTE: [u8; 2] = [0xfe, 0x06];
// dec word ptr [rsi]
const DEC_MEMORY_WORD: [u8; 3] = [0x66, 0xff, 0x0e];
// shr dword ptr [rsi], 3
const SHR_IMMEDIATE_MEMORY_DWORD: [u8; 3] = [0xc1, 0x2e, 0x03];
// sar qword ptr [rsi], cl
const SAR_CL_MEMORY_QWORD: [u8; 3] = [0x48, 0xd3, 0x3e];
// neg dword ptr [rsi]
const NEG_MEMORY_DWORD: [u8; 2] = [0xf7, 0x1e];
// not qword ptr [rsi]
const NOT_MEMORY_QWORD: [u8; 3] = [0x48, 0xf7, 0x16];

// sete [rsi]
const SETE_TO_MEMORY: [u8; 3] = [0x0f, 0x94, 0x06];

// enter 16, 0
const ENTER_16: [u8; 4] = [0xc8, 0x10, 0x00, 0x00];
// push 5
const PUSH_IMM: [u8; 2] = [0x6a, 0x05];
// call rax
const CALL_RAX: [u8; 2] = [0xff, 0xd0];

/// Stack pointers whose implicit accesses (at `rsp` or `rsp - 8`) are non-canonical.
const NON_CANONICAL_STACK_POINTERS: [u64; 3] = [
    0x0000_8000_0000_0008,
    0x8000_0000_0000_0000,
    0x1234_5678_9abc_def0,
];
/// Vector of #SS.
const STACK_FAULT_VECTOR: u8 = 12;

macro_rules! non_canonical_stack_tests {
    ($($name:ident => $code:expr,)*) => {
        $(#[test]
        fn $name() {
            for stack_pointer in NON_CANONICAL_STACK_POINTERS {
                let code = [&PRELUDE[..], &$code[..], &EPILOGUE[..]].concat();
                let result = check_stop_with(
                    &code,
                    |harness| {
                        seed_registers_and_memory(harness);
                        harness.cpu.gpr[RSP] = stack_pointer;
                        harness.cpu.gpr[RBP] = stack_pointer;
                    },
                    fault(STACK_FAULT_VECTOR, Some(0)),
                );
                assert_eq!(result.interpreter.cpu.gpr[RSP], stack_pointer, "rsp is unchanged");
                assert_eq!(result.interpreter.cpu.rip, CODE_BASE + PRELUDE.len() as u64);
                assert_eq!(result.interpreter.cpu.cr2, 0);
            }
        })*
    };
}

non_canonical_stack_tests! {
    push_register_with_a_non_canonical_stack_raises_stack_fault => PUSH_RAX,
    push_immediate_with_a_non_canonical_stack_raises_stack_fault => PUSH_IMM,
    pop_register_with_a_non_canonical_stack_raises_stack_fault => POP_RAX,
    call_with_a_non_canonical_stack_raises_stack_fault => CALL_REL32_TO_NEXT,
    indirect_call_with_a_non_canonical_stack_raises_stack_fault => CALL_RAX,
    ret_with_a_non_canonical_stack_raises_stack_fault => RET_NEAR,
    ret_imm16_with_a_non_canonical_stack_raises_stack_fault => RET_IMM16,
    leave_with_a_non_canonical_frame_pointer_raises_stack_fault => LEAVE,
    enter_with_a_non_canonical_stack_raises_stack_fault => ENTER_16,
    pushfq_with_a_non_canonical_stack_raises_stack_fault => PUSHFQ,
    popfq_with_a_non_canonical_stack_raises_stack_fault => POPFQ,
}
