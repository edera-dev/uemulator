//! Paging features the JIT's memory fast path never looks at directly, so they must reach it
//! through the soft TLB: 5-level paging with 57-bit addresses, and protection keys, whose
//! restrictions (and every PKRU change by wrpkru or xrstor) must stop a cached translation from
//! being used. Expected stops restate the architecture rather than what the interpreter does.

use super::diff_tests::*;
use crate::testkit::*;

const PTE_PRESENT: u64 = 1;
const PTE_WRITABLE: u64 = 2;
const PTE_USER: u64 = 4;
const PTE_KEY_SHIFT: u32 = 59;
const PAGE_BYTES: u64 = 4096;
/// `boot::load` lays out PML4, PDPT and the first page directory back to back from `cr3`, and
/// maps the first GiBs with 2 MiB leaves.
const PDPT_OFFSET: u64 = PAGE_BYTES;
const FIRST_PAGE_DIRECTORY_OFFSET: u64 = 2 * PAGE_BYTES;
const HUGE_PAGE_SHIFT: u32 = 21;
const CR4_LA57: u64 = 1 << 12;
const CR4_PKE: u64 = 1 << 22;
const CR0_WRITE_PROTECT: u64 = 1 << 16;
const PAGE_FAULT: u8 = 14;
const GENERAL_PROTECTION: u8 = 13;
const PF_PRESENT: u32 = 1;
const PF_WRITE: u32 = 2;
const PF_PROTECTION_KEY: u32 = 1 << 5;
/// Free RAM below `SCRATCH` for the PML5.
const PML5: u64 = 0x3f_f000;
/// PML5 index 1 aliases index 0, so this reaches `SCRATCH` through a 57-bit address.
const SCRATCH_ABOVE_48_BITS: u64 = (1 << 48) + SCRATCH;
const KEY: u64 = 1;
const PKRU_ACCESS_DISABLE: u64 = 1 << (2 * KEY);
const PKRU_WRITE_DISABLE: u64 = 2 << (2 * KEY);

// mov rax, [rbx]
const LOAD_RAX: [u8; 3] = [0x48, 0x8b, 0x03];
// mov rsi, [rbx + 8]
const LOAD_RSI: [u8; 4] = [0x48, 0x8b, 0x73, 0x08];
// mov [rbx + 16], rax
const STORE_RAX: [u8; 4] = [0x48, 0x89, 0x43, 0x10];
// mov rcx, [rdi]
const LOAD_RCX_RDI: [u8; 3] = [0x48, 0x8b, 0x0f];
// xor ecx, ecx; xor edx, edx
const CLEAR_ECX_EDX: [u8; 4] = [0x31, 0xc9, 0x31, 0xd2];
// mov eax, imm32
const MOV_EAX: u8 = 0xb8;
const WRPKRU: [u8; 3] = [0x0f, 0x01, 0xef];
const RDPKRU: [u8; 3] = [0x0f, 0x01, 0xee];

fn wrpkru(value: u64) -> Vec<u8> {
    let mut code = vec![MOV_EAX];
    code.extend_from_slice(&(value as u32).to_le_bytes());
    code.extend_from_slice(&CLEAR_ECX_EDX);
    code.extend_from_slice(&WRPKRU);
    code
}

fn program(pieces: &[&[u8]]) -> Vec<u8> {
    let mut code = pieces.concat();
    code.push(HLT);
    code
}

/// 5-level paging over the boot identity map: PML5 entries 0 and 1 both point at the PML4.
fn five_level(harness: &mut Harness) {
    seed_registers_and_memory(harness);
    let pml4 = harness.cpu.cr3 & !(PAGE_BYTES - 1);
    harness.write_u64(PML5, pml4 | PTE_PRESENT | PTE_WRITABLE);
    harness.write_u64(PML5 + 8, pml4 | PTE_PRESENT | PTE_WRITABLE);
    harness.cpu.cr4 |= CR4_LA57;
    harness.cpu.cr3 = PML5;
    harness.cpu.flush_tlb();
    harness.cpu.gpr[RBX] = SCRATCH_ABOVE_48_BITS + 0x40;
}

#[test]
fn loads_and_stores_through_a_57_bit_address() {
    let result = check_with(&program(&[&LOAD_RAX, &STORE_RAX, &LOAD_RSI]), five_level);
    let value = result.interpreter.read_u64(SCRATCH + 0x40);
    assert_eq!(result.interpreter.read_u64(SCRATCH + 0x50), value);
}

#[test]
fn a_48_bit_non_canonical_address_is_a_page_fault_under_la57() {
    // PML4 index 256 of the aliased hierarchy is empty: not present, not #GP.
    check_stop_with(
        &program(&[&LOAD_RAX, &LOAD_RCX_RDI]),
        |harness| {
            five_level(harness);
            harness.cpu.gpr[RDI] = 0x0000_8000_0000_0000;
        },
        fault(PAGE_FAULT, Some(0)),
    );
}

#[test]
fn a_57_bit_non_canonical_address_is_gp0() {
    check_stop_with(
        &program(&[&LOAD_RAX, &LOAD_RCX_RDI]),
        |harness| {
            five_level(harness);
            harness.cpu.gpr[RDI] = 0x0100_0000_0000_0000;
        },
        fault(GENERAL_PROTECTION, Some(0)),
    );
}

/// The data the keyed tests touch. Not in `SCRATCH`'s first page: that shares its soft-TLB slot
/// with the code page, so every fetch would evict it and a stale translation could never be
/// seen, which is what the tests must show does not happen.
const KEYED_DATA: u64 = SCRATCH + 0x5040;

/// `SCRATCH`'s 2 MiB leaf becomes a user page with key `KEY` (the upper levels user too), CR4.PKE
/// is on and PKRU starts at `pkru`. Code stays on a supervisor page.
fn keyed(harness: &mut Harness, pkru: u64) {
    seed_registers_and_memory(harness);
    let cr3 = harness.cpu.cr3 & !(PAGE_BYTES - 1);
    for table_entry in [cr3, cr3 + PDPT_OFFSET] {
        let entry = harness.read_u64(table_entry);
        harness.write_u64(table_entry, entry | PTE_USER);
    }
    let pde = cr3 + FIRST_PAGE_DIRECTORY_OFFSET + (SCRATCH >> HUGE_PAGE_SHIFT) * 8;
    let entry = harness.read_u64(pde);
    harness.write_u64(pde, entry | PTE_USER | KEY << PTE_KEY_SHIFT);
    harness.cpu.cr4 |= CR4_PKE;
    harness.cpu.pkru = pkru as u32;
    harness.cpu.flush_tlb();
    harness.cpu.gpr[RBX] = KEYED_DATA;
}

#[test]
fn wrpkru_access_disable_stops_a_load_from_a_page_the_block_already_read() {
    check_stop_with(
        &program(&[&LOAD_RAX, &wrpkru(PKRU_ACCESS_DISABLE), &LOAD_RSI]),
        |harness| keyed(harness, 0),
        fault(PAGE_FAULT, Some(PF_PRESENT | PF_PROTECTION_KEY)),
    );
}

#[test]
fn wrpkru_write_disable_stops_a_store_under_cr0_wp_but_not_a_load() {
    check_stop_with(
        &program(&[
            &STORE_RAX,
            &wrpkru(PKRU_WRITE_DISABLE),
            &LOAD_RSI,
            &STORE_RAX,
        ]),
        |harness| {
            keyed(harness, 0);
            harness.cpu.cr0 |= CR0_WRITE_PROTECT;
        },
        fault(PAGE_FAULT, Some(PF_PRESENT | PF_WRITE | PF_PROTECTION_KEY)),
    );
}

#[test]
fn write_disable_lets_a_supervisor_store_through_without_cr0_wp() {
    check_with(&program(&[&LOAD_RAX, &STORE_RAX]), |harness| {
        keyed(harness, PKRU_WRITE_DISABLE);
        harness.cpu.cr0 &= !CR0_WRITE_PROTECT;
    });
}

#[test]
fn wrpkru_lifting_the_restriction_lets_the_access_through() {
    let result = check_with(
        &program(&[&wrpkru(0), &LOAD_RAX, &STORE_RAX, &RDPKRU]),
        |harness| keyed(harness, PKRU_ACCESS_DISABLE | PKRU_WRITE_DISABLE),
    );
    assert_eq!(result.interpreter.cpu.pkru, 0);
}

#[test]
fn rdpkru_reads_back_what_wrpkru_wrote() {
    let result = check_with(&program(&[&wrpkru(0x5555_5554), &RDPKRU]), |harness| {
        keyed(harness, 0)
    });
    assert_eq!(result.interpreter.cpu.gpr[RAX], 0x5555_5554);
}

#[test]
fn xrstor_loading_pkru_mid_block_stops_the_next_load() {
    // xor edx, edx; mov eax, XCR0_PKRU; xrstor [rdi]: xrstor runs in the interpreter helper and
    // does not end the block, so the load after it is compiled with the inline TLB probe.
    const XCR0_PKRU: u64 = 1 << 9;
    const CR4_OSFXSR_AND_OSXSAVE: u64 = (1 << 9) | (1 << 18);
    const AREA: u64 = SCRATCH + 0x1000;
    const XSTATE_BV: u64 = 512;
    const HEADER_BYTES: usize = 64;
    const PKRU_OFFSET: u64 = 2688;
    let mut restore = vec![0x31, 0xd2, MOV_EAX];
    restore.extend_from_slice(&(XCR0_PKRU as u32).to_le_bytes());
    restore.extend_from_slice(&[0x0f, 0xae, 0x2f]);
    check_stop_with(
        &program(&[&LOAD_RAX, &restore, &LOAD_RSI]),
        |harness| {
            keyed(harness, 0);
            harness.cpu.cr4 |= CR4_OSFXSR_AND_OSXSAVE;
            harness.cpu.xcr0 |= XCR0_PKRU;
            harness.cpu.gpr[RDI] = AREA;
            harness.write(AREA + XSTATE_BV, &[0; HEADER_BYTES]);
            harness.write_u64(AREA + XSTATE_BV, XCR0_PKRU);
            harness.write_u64(AREA + PKRU_OFFSET, PKRU_ACCESS_DISABLE);
        },
        fault(PAGE_FAULT, Some(PF_PRESENT | PF_PROTECTION_KEY)),
    );
}
