//! The dirty log (`ram::DirtyLog`) behind delta snapshots must see every write to guest RAM: a
//! store it misses leaves a delta silently wrong. One test per way a write reaches RAM, each
//! built so that it fails if that path stops marking. The threaded case is in `smp_tests.rs`,
//! and `machine/suspend/tests.rs` checks whole machines (with `check_dirty_tracking`).

use super::test_support::*;
use super::*;
use crate::dev::guestmem::GuestMem;
use crate::testkit::{Harness, RAX, RDI, RSI, SCRATCH};

const FIRST: u64 = SCRATCH;
const SECOND: u64 = SCRATCH + 0x3000;
const UNTOUCHED: u64 = SCRATCH + 0x8000;
const PARENT: [u8; 16] = [7; 16];
/// mov [rdi], rax; mov [rsi], rax; hlt
const STORE_TWICE: [u8; 7] = [0x48, 0x89, 0x07, 0x48, 0x89, 0x06, 0xf4];
/// mov rbx, [rdi]; mov [rdi], rax; hlt
const LOAD_THEN_STORE: [u8; 7] = [0x48, 0x8b, 0x1f, 0x48, 0x89, 0x07, 0xf4];
/// mov ecx, 1024; rep stosq; hlt: two pages through `rep_bulk`'s host fill.
const FILL_TWO_PAGES: [u8; 9] = [0xb9, 0x00, 0x04, 0x00, 0x00, 0xf3, 0x48, 0xab, 0xf4];

fn page(address: u64) -> usize {
    (address >> PAGE_SHIFT) as usize
}

fn dirty(cpu: &Cpu, address: u64) -> bool {
    cpu.page_has_code.dirty_log().is_dirty(page(address))
}

/// What a checkpoint does to the one vCPU: a new generation, and no write tag left over.
fn arm(cpu: &mut Cpu) {
    cpu.page_has_code.dirty_log().arm(PARENT);
    cpu.revoke_all_write_tags();
}

fn rerun(h: &mut Harness) {
    h.cpu.rip = crate::testkit::CODE_BASE;
    h.cpu.halted = false;
    h.run_to_halt();
}

/// Stores through the soft TLB: the first run leaves write tags for both pages; once armed, the
/// same stores (the JIT's inline ones, the interpreter's fast path) must take the slow path.
fn stores_mark_after_arming(jit: bool) {
    let mut h = Harness::new(&STORE_TWICE);
    if jit && !h.enable_jit() {
        return;
    }
    h.cpu.gpr[RDI] = FIRST;
    h.cpu.gpr[RSI] = SECOND;
    h.cpu.gpr[RAX] = 1;
    h.run_to_halt();
    assert!(!dirty(&h.cpu, FIRST), "the log is not armed yet");
    arm(&mut h.cpu);
    h.cpu.gpr[RAX] = 2;
    rerun(&mut h);
    assert_eq!(h.read_u64(FIRST), 2);
    assert!(dirty(&h.cpu, FIRST) && dirty(&h.cpu, SECOND));
    assert!(!dirty(&h.cpu, UNTOUCHED));
    // The tag comes back once the page is marked: the next stores are fast again.
    let slot = &h.cpu.tlb[page(FIRST) & (TLB_ENTRIES_PER_MODE - 1)];
    assert_eq!(slot.tag_w, FIRST);
}

#[test]
fn interpreter_stores_mark_their_pages() {
    stores_mark_after_arming(false);
}

#[test]
fn jit_stores_mark_their_pages() {
    stores_mark_after_arming(true);
}

/// A load fills the TLB entry first; it must not hand out the write tag of a clean page that the
/// store after it would then use without marking. The first run sets the page's Dirty bit, so
/// only the dirty log can withhold the tag.
#[test]
fn a_load_does_not_grant_a_clean_page_its_write_tag() {
    for jit in [false, true] {
        let mut h = Harness::new(&LOAD_THEN_STORE);
        if jit && !h.enable_jit() {
            continue;
        }
        h.cpu.gpr[RDI] = FIRST;
        h.run_to_halt();
        arm(&mut h.cpu);
        h.cpu.flush_tlb();
        rerun(&mut h);
        assert!(dirty(&h.cpu, FIRST), "jit {jit}");
    }
}

#[test]
fn rep_stos_marks_every_page_it_fills() {
    for jit in [false, true] {
        let mut h = Harness::new(&FILL_TWO_PAGES);
        if jit && !h.enable_jit() {
            continue;
        }
        h.cpu.gpr[RDI] = FIRST;
        h.cpu.gpr[RAX] = 0x5a;
        arm(&mut h.cpu);
        h.run_to_halt();
        assert!(
            dirty(&h.cpu, FIRST) && dirty(&h.cpu, FIRST + 0x1000),
            "jit {jit}"
        );
        assert!(!dirty(&h.cpu, FIRST + 0x2000), "jit {jit}");
    }
}

/// The walker's Accessed and Dirty updates write the page tables.
#[test]
fn page_table_accessed_and_dirty_updates_mark_the_table() {
    const VA: u64 = 0x7000_0000;
    let mut h = Harness::new(&[crate::testkit::HLT]);
    let mut tables = PageTables::new(&mut h.cpu);
    let leaf = tables.map_4k(&mut h.cpu, VA, FIRST, PTE_PRESENT_WRITABLE_USER);
    arm(&mut h.cpu);
    h.cpu.translate(VA, mem::AccessKind::Read, false).unwrap();
    assert_ne!(read_pte(&h.cpu, leaf) & PTE_A, 0);
    assert!(
        dirty(&h.cpu, leaf),
        "setting Accessed did not mark the table"
    );
    assert!(!dirty(&h.cpu, FIRST), "a load marked its page");
    h.cpu.translate(VA, mem::AccessKind::Write, false).unwrap();
    assert!(dirty(&h.cpu, FIRST));
}

/// Writes by physical address: what the VMCB, the AVIC and shadow stacks use.
#[test]
fn physical_writes_mark_their_pages() {
    let mut h = Harness::new(&[crate::testkit::HLT]);
    arm(&mut h.cpu);
    h.cpu.phys_write(SECOND + 8, 8, 1);
    assert!(dirty(&h.cpu, SECOND));
    assert!(!dirty(&h.cpu, FIRST));
}

/// Device DMA bypasses every CPU path; `GuestMem` marks for it.
#[test]
fn device_writes_mark_every_page_they_touch() {
    let mut h = Harness::new(&[crate::testkit::HLT]);
    arm(&mut h.cpu);
    let pages = h.cpu.page_has_code.clone();
    let mut mem = GuestMem::for_pages(&mut h.cpu.ram, &pages);
    // Straddles two pages.
    assert!(mem.write(FIRST + 0xffc, &[1; 8]));
    assert!(crate::dev::iommu::PhysWrite::fetch_or_u32(
        &mut mem, SECOND, 1
    ));
    drop(mem);
    assert!(dirty(&h.cpu, FIRST) && dirty(&h.cpu, FIRST + 0x1000) && dirty(&h.cpu, SECOND));
    assert!(!dirty(&h.cpu, UNTOUCHED));
}

#[test]
fn an_unarmed_log_records_nothing_and_has_no_parent() {
    let mut h = Harness::new(&STORE_TWICE);
    h.cpu.gpr[RDI] = FIRST;
    h.cpu.gpr[RSI] = SECOND;
    h.run_to_halt();
    let log = h.cpu.page_has_code.dirty_log();
    assert!(!log.is_dirty(page(FIRST)));
    assert_eq!(log.parent(), None);
    log.arm(PARENT);
    assert_eq!(log.parent(), Some(PARENT));
    log.disarm();
    assert_eq!(log.parent(), None);
}
