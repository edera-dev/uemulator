//! The code cache several vCPUs share: what one handle translates the others find, what one
//! invalidates or flushes the others stop using, links patched by one are followed by another,
//! and the bookkeeping that keeps all of that cheap. Single-threaded: the interleavings that need
//! real threads are in `cpu/smp_tests.rs`, and the steps of `translate` that a second vCPU can
//! slip into are pinned down with the hooks of this module's parent.

use super::*;
use crate::jit::serialize_jit_tests;
use crate::testkit::*;

const MOV_EAX_ONE: [u8; 6] = [0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3];
const NOP: u8 = 0x90;
/// mov ecx, 100; l: sub ecx, 1; jnz l; hlt
const COUNTDOWN_LOOP: [u8; 11] = [
    0xb9, 0x64, 0x00, 0x00, 0x00, 0x83, 0xe9, 0x01, 0x75, 0xfb, 0xf4,
];
const COUNTDOWN_LOOP_INSNS: u64 = 202;
const PAGE: u64 = PAGE_SIZE;

fn cpu_running(code: &[u8]) -> Cpu {
    Harness::new(code).cpu
}

fn new_jit() -> Jit {
    Jit::new(true).expect("the host has a JIT")
}

fn translate_at_code_base(jit: &mut Jit, cpu: &mut Cpu) -> u64 {
    jit.translate(cpu, CODE_BASE, CODE_BASE).unwrap()
}

fn step_once(cpu: &mut Cpu) {
    cpu.interrupt_shadow = false;
    cpu.icount += 1;
    cpu.step_block().unwrap();
}

fn warm_up_loop(cpu: &mut Cpu) {
    cpu.next_event = 20;
    while cpu.icount < 20 {
        step_once(cpu);
    }
    cpu.next_event = u64::MAX;
}

fn run_to_halt(cpu: &mut Cpu) {
    for _ in 0..1000 {
        step_once(cpu);
        if cpu.halted {
            return;
        }
    }
    panic!("never halted: {}", cpu.dump_regs());
}

#[test]
fn a_block_translated_through_one_handle_is_found_through_another() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut first = new_jit();
    let mut second = first.sibling();
    let mut third = second.sibling();
    let code = translate_at_code_base(&mut first, &mut cpu);
    assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert_eq!(third.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert_eq!(first.stats().translated, 1);
}

#[test]
fn two_handles_translating_the_same_code_register_one_block() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut first = new_jit();
    let mut second = first.sibling();
    let code = translate_at_code_base(&mut first, &mut cpu);
    // The second handle's jump cache is empty, but translating finds the registered block.
    assert_eq!(translate_at_code_base(&mut second, &mut cpu), code);
    assert_eq!(first.state().blocks.len(), 1);
    assert_eq!(first.stats().translated, 1);
}

#[test]
fn the_same_code_on_another_physical_page_is_another_block() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let other = CODE_BASE + PAGE;
    cpu.ram[other as usize..][..MOV_EAX_ONE.len()].copy_from_slice(&MOV_EAX_ONE);
    let mut jit = new_jit();
    let first = translate_at_code_base(&mut jit, &mut cpu);
    let second = jit.translate(&mut cpu, other, other).unwrap();
    assert_ne!(first, second);
    assert_eq!(jit.state().blocks.len(), 2);
}

#[test]
fn blocks_at_different_offsets_of_one_physical_page_are_different_blocks() {
    let _guard = serialize_jit_tests();
    let code = [&[NOP; 8][..], &MOV_EAX_ONE[..]].concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    let from_start = translate_at_code_base(&mut jit, &mut cpu);
    let from_middle = jit
        .translate(&mut cpu, CODE_BASE + 4, CODE_BASE + 4)
        .unwrap();
    assert_ne!(from_start, from_middle);
    assert_eq!(jit.state().blocks.len(), 2);
    assert_eq!(jit.state().page_blocks[&(CODE_BASE >> PAGE_SHIFT)].len(), 2);
}

#[test]
fn many_virtual_aliases_of_one_block_cost_one_translation() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut jit = new_jit();
    let code = translate_at_code_base(&mut jit, &mut cpu);
    for alias in 1..200u64 {
        let rip = (alias << 30) | (CODE_BASE & PAGE_MASK);
        assert_eq!(jit.lookup(rip, CODE_BASE), Some(code), "alias {alias}");
    }
    assert_eq!(jit.stats().translated, 1);
    assert_eq!(jit.state().blocks.len(), 1);
}

#[test]
fn a_jump_cache_slot_naming_an_invalidated_block_is_not_used_by_its_owner() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut writer = new_jit();
    let mut reader = writer.sibling();
    let old = translate_at_code_base(&mut writer, &mut cpu);
    assert_eq!(reader.lookup(CODE_BASE, CODE_BASE), Some(old));
    writer.invalidate_page(CODE_BASE >> PAGE_SHIFT, &cpu.page_has_code);
    // The reader's slot is untouched (it cannot be found from the block), but not trusted.
    assert_ne!(reader.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, 0);
    assert_eq!(reader.lookup(CODE_BASE, CODE_BASE), None);
    let new = translate_at_code_base(&mut reader, &mut cpu);
    assert_ne!(new, old, "a new block, not the invalidated one");
    assert_eq!(writer.lookup(CODE_BASE, CODE_BASE), Some(new));
}

#[test]
fn a_flush_marks_every_block_invalid_for_every_handle_before_anyone_syncs() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut flusher = new_jit();
    let mut other = flusher.sibling();
    translate_at_code_base(&mut flusher, &mut cpu);
    assert!(other.lookup(CODE_BASE, CODE_BASE).is_some());
    flusher.reset();
    assert!(!flusher.block_is_valid(0));
    // `other` has not synced yet: its slot is still there, and still refused.
    assert_ne!(other.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, 0);
    assert_eq!(other.lookup(CODE_BASE, CODE_BASE), None);
}

#[test]
fn each_flush_is_seen_once_by_each_handle() {
    let _guard = serialize_jit_tests();
    let mut jit = new_jit();
    let mut sibling = jit.sibling();
    assert_eq!(sibling.seen_epoch, 0);
    jit.reset();
    jit.reset();
    assert_eq!(jit.shared.epoch.load(Ordering::Relaxed), 2);
    sibling.sync();
    assert_eq!(sibling.seen_epoch, 2);
    sibling.jump_cache[5].code_addr = 7;
    sibling.sync();
    assert_eq!(
        sibling.jump_cache[5].code_addr, 7,
        "nothing new, nothing cleared"
    );
}

#[test]
fn a_handle_made_after_a_flush_starts_in_the_current_epoch() {
    let _guard = serialize_jit_tests();
    let mut jit = new_jit();
    jit.reset();
    let late = jit.sibling();
    assert_eq!(late.seen_epoch, 1);
}

#[test]
fn the_cache_outlives_the_handle_that_made_it() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut first = new_jit();
    let mut second = first.sibling();
    let code = translate_at_code_base(&mut first, &mut cpu);
    drop(first);
    assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(code));
    let mut third = second.sibling();
    drop(second);
    assert_eq!(third.lookup(CODE_BASE, CODE_BASE), Some(code));
}

#[test]
fn siblings_have_their_own_jump_cache_and_share_the_validity_table() {
    let _guard = serialize_jit_tests();
    let jit = new_jit();
    let sibling = jit.sibling();
    assert_ne!(jit.jump_cache_base(), sibling.jump_cache_base());
    assert_eq!(jit.valid_base(), sibling.valid_base());
    assert_eq!(jit.enter_trampoline(), sibling.enter_trampoline());
}

#[test]
fn a_jump_cache_fill_by_one_handle_does_not_appear_in_another() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut first = new_jit();
    let second = first.sibling();
    translate_at_code_base(&mut first, &mut cpu);
    assert_ne!(first.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, 0);
    assert_eq!(second.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, 0);
}

#[test]
fn turning_on_ordered_memory_drops_blocks_translated_without_it() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    jit.set_ordered_memory(true);
    assert!(jit.state().blocks.is_empty());
    assert_eq!(jit.stats().resets, 1);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), None);
    translate_at_code_base(&mut jit, &mut cpu);
    jit.set_ordered_memory(true);
    assert_eq!(jit.stats().resets, 1, "already on: nothing to drop");
    assert_eq!(jit.state().blocks.len(), 1);
}

#[test]
fn ordered_memory_changes_the_code_of_a_block_that_loads_on_aarch64_only() {
    let _guard = serialize_jit_tests();
    // mov rax, [rbx]; hlt
    let code = [0x48, 0x8b, 0x03, 0xf4];
    let mut cpu = cpu_running(&code);
    let mut plain = new_jit();
    let mut ordered = new_jit();
    ordered.set_ordered_memory(true);
    translate_at_code_base(&mut plain, &mut cpu);
    translate_at_code_base(&mut ordered, &mut cpu);
    let size = |jit: &Jit| jit.state().used_bytes - jit.shared.blocks_start;
    if cfg!(target_arch = "aarch64") {
        assert_ne!(size(&plain), size(&ordered), "ldar and the alignment check");
    } else {
        assert_eq!(size(&plain), size(&ordered), "x86-64 is already ordered");
    }
}

#[test]
fn a_block_limit_flushes_when_the_next_block_would_exceed_it() {
    let _guard = serialize_jit_tests();
    let code = [
        &[NOP; 4][..],
        &[0xeb, 0x00][..],
        &[NOP; 4][..],
        &[0xeb, 0x00][..],
        &[0xf4][..],
    ]
    .concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    jit.set_block_limit(2);
    translate_at_code_base(&mut jit, &mut cpu);
    jit.translate(&mut cpu, CODE_BASE + 6, CODE_BASE + 6)
        .unwrap();
    assert_eq!(jit.stats().resets, 0);
    jit.translate(&mut cpu, CODE_BASE + 12, CODE_BASE + 12)
        .unwrap();
    assert_eq!(jit.stats().resets, 1);
    assert_eq!(jit.state().blocks.len(), 1);
}

#[test]
fn a_flush_resets_the_bump_pointer_and_the_link_tables() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    let mut jit = new_jit();
    cpu.jit = None;
    let mut handle = Some(Box::new(jit.sibling()));
    std::mem::swap(&mut cpu.jit, &mut handle);
    warm_up_loop(&mut cpu);
    assert_ne!(jit.link_state_len(), (0, 0));
    jit.reset();
    assert_eq!(jit.link_state_len(), (0, 0));
    assert_eq!(jit.state().used_bytes, jit.shared.blocks_start);
}

#[test]
fn invalidating_a_page_counts_even_when_nothing_was_translated_from_it() {
    let _guard = serialize_jit_tests();
    let cpu = cpu_running(&MOV_EAX_ONE);
    let jit = new_jit();
    let before = jit.shared.invalidations.load(Ordering::Relaxed);
    jit.invalidate_page(0x77, &cpu.page_has_code);
    jit.invalidate_page(0x78, &cpu.page_has_code);
    assert_eq!(jit.shared.invalidations.load(Ordering::Relaxed), before + 2);
}

#[test]
fn invalidating_a_page_clears_only_that_pages_flag() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    let page = (CODE_BASE >> PAGE_SHIFT) as usize;
    cpu.page_has_code.set(page + 1, 1);
    cpu.page_has_code.set(page - 1, 1);
    jit.invalidate_page(page as u64, &cpu.page_has_code);
    assert_eq!(cpu.page_has_code.get(page), 0);
    assert_eq!(cpu.page_has_code.get(page + 1), 1);
    assert_eq!(cpu.page_has_code.get(page - 1), 1);
}

#[test]
fn another_handle_translating_inside_the_window_is_found_not_duplicated() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut jit = new_jit();
    let mut rival = jit.sibling();
    let rival_code = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = rival_code.clone();
    BEFORE_INSTALL.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move |cpu: &mut Cpu| {
            seen.set(translate_at_code_base(&mut rival, cpu));
        }));
    });
    let code = translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(code, rival_code.get(), "the block the rival registered");
    assert_eq!(jit.state().blocks.len(), 1);
    assert_eq!(jit.stats().translated, 1);
}

#[test]
fn a_flush_while_another_handle_decodes_leaves_that_translation_correct() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut jit = new_jit();
    let mut flusher = jit.sibling();
    BEFORE_INSTALL.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move |_: &mut Cpu| flusher.reset()));
    });
    let code = translate_at_code_base(&mut jit, &mut cpu);
    assert_eq!(jit.lookup(CODE_BASE, CODE_BASE), Some(code));
    assert!(jit.state().used_bytes > jit.shared.blocks_start);
    assert_eq!(jit.state().blocks.len(), 1);
}

#[test]
fn links_patched_by_one_handle_are_followed_by_blocks_another_runs() {
    let _guard = serialize_jit_tests();
    let mut warm = cpu_running(&COUNTDOWN_LOOP);
    let jit = new_jit();
    let sibling = jit.sibling();
    warm.jit = Some(Box::new(jit));
    warm_up_loop(&mut warm);
    let translated = warm.jit.as_ref().unwrap().stats().translated;
    let linked = warm.jit.as_ref().unwrap().stats().linked;
    assert!(linked >= 1);
    // A second CPU with the same code runs it to the end on the other handle.
    let mut other = cpu_running(&COUNTDOWN_LOOP);
    other.jit = Some(Box::new(sibling));
    run_to_halt(&mut other);
    assert_eq!(other.gpr[1], 0);
    assert_eq!(other.icount, COUNTDOWN_LOOP_INSNS);
    assert_eq!(
        other.jit.as_ref().unwrap().stats().translated,
        translated,
        "nothing new: the loop's block runs on past its jnz to the final `hlt`"
    );
    assert!(
        other.jit.as_ref().unwrap().stats().barriers >= 1,
        "the blocks the other handle wrote are entered after an instruction barrier"
    );
}

/// A lone handle wrote every block it runs, so taking them from the block map after a jump
/// cache miss needs no instruction barrier beyond its first (`Jit::barrier_if_foreign`).
#[test]
fn a_lone_handle_runs_its_own_blocks_without_instruction_barriers() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    cpu.jit = Some(Box::new(new_jit()));
    warm_up_loop(&mut cpu);
    // Forget the jump cache, so every block comes from the block map again.
    let jit = cpu.jit.as_mut().unwrap();
    jit.jump_cache.fill(JumpEntry::default());
    let before = jit.stats().barriers;
    run_to_halt(&mut cpu);
    assert_eq!(cpu.gpr[1], 0);
    assert_eq!(cpu.jit.as_ref().unwrap().stats().barriers, before);
}

#[test]
fn a_block_waiting_for_its_destination_is_linked_when_another_handle_translates_it() {
    let _guard = serialize_jit_tests();
    // jmp +8 (rel32); 8 nops; hlt
    let code = [
        &[0xe9, 0x08, 0x00, 0x00, 0x00][..],
        &[NOP; 8][..],
        &[0xf4][..],
    ]
    .concat();
    let mut cpu = cpu_running(&code);
    let mut first = new_jit();
    let mut second = first.sibling();
    translate_at_code_base(&mut first, &mut cpu);
    assert_eq!(
        first.link_state_len(),
        (0, 1),
        "the destination does not exist yet"
    );
    let target = CODE_BASE + 13;
    second.translate(&mut cpu, target, target).unwrap();
    assert_eq!(first.link_state_len(), (1, 0));
    assert_eq!(first.stats().linked, 1);
}

#[test]
fn invalidating_a_destination_through_another_handle_unlinks_every_slot_into_it() {
    let _guard = serialize_jit_tests();
    let code = [
        &[0xe9, 0x08, 0x00, 0x00, 0x00][..],
        &[NOP; 8][..],
        &[0xf4][..],
    ]
    .concat();
    let mut cpu = cpu_running(&code);
    let mut first = new_jit();
    let second = first.sibling();
    translate_at_code_base(&mut first, &mut cpu);
    first
        .translate(&mut cpu, CODE_BASE + 13, CODE_BASE + 13)
        .unwrap();
    assert_eq!(first.link_state_len(), (1, 0));
    second.invalidate_page(CODE_BASE >> PAGE_SHIFT, &cpu.page_has_code);
    assert_eq!(first.stats().unlinked, 1);
    assert_eq!(
        first.link_state_len(),
        (0, 1),
        "waiting for the next translation"
    );
}

#[test]
fn every_link_slot_has_its_patched_word_aligned_wherever_the_block_starts() {
    let _guard = serialize_jit_tests();
    // Blocks that end in a conditional and an unconditional jump at every alignment of the
    // preceding code: the slot's word is patched while other threads may run it, so it must be
    // one aligned 32-bit store.
    for padding in 0..16usize {
        let code = [
            &vec![NOP; padding][..],
            &[0x75, 0x02][..],
            &[0xeb, 0x00][..],
            &[0xf4][..],
        ]
        .concat();
        let mut cpu = cpu_running(&code);
        let mut jit = new_jit();
        translate_at_code_base(&mut jit, &mut cpu);
        let state = jit.state();
        let sites: Vec<_> = state
            .pending
            .values()
            .chain(state.incoming.values())
            .flatten()
            .collect();
        assert!(!sites.is_empty(), "padding {padding}");
        for site in sites {
            let word =
                jit.shared.exec_mem.base() + (site.exec_offset + arch::LINK_WORD_OFFSET) as u64;
            assert_eq!(word % 4, 0, "padding {padding}");
        }
    }
}

/// A one-instruction `mov eax, imm32` block that straddles the end of `first` and continues in
/// `second`, and an ordinary block that jumps to it.
fn straddler_with_a_jump_to_it() -> (Cpu, crate::cpu::test_support::PageTables, u64, u64) {
    use crate::cpu::test_support::{PTE_PRESENT_WRITABLE_USER, PageTables};
    const VIRTUAL_PAGE: u64 = 0x0000_2000_0000_0000;
    let (first, second) = (SCRATCH, SCRATCH + 2 * PAGE);
    let mut cpu = cpu_running(&[0xf4]);
    let mut tables = PageTables::new(&mut cpu);
    for (virtual_page, physical) in [(VIRTUAL_PAGE, first), (VIRTUAL_PAGE + PAGE, second)] {
        tables.map_4k(&mut cpu, virtual_page, physical, PTE_PRESENT_WRITABLE_USER);
    }
    // At the start of the first page: jmp rel32 to the straddling instruction two bytes before its end.
    let target = PAGE - 2;
    let rel = (target - 5) as u32;
    cpu.ram[first as usize] = 0xe9;
    cpu.ram[first as usize + 1..][..4].copy_from_slice(&rel.to_le_bytes());
    let mov = [0xb8u8, 0x44, 0x33, 0x22, 0x11];
    cpu.ram[(first + target) as usize..][..2].copy_from_slice(&mov[..2]);
    cpu.ram[second as usize..][..3].copy_from_slice(&mov[2..]);
    (cpu, tables, VIRTUAL_PAGE, first)
}

#[test]
fn a_link_into_a_straddling_instruction_is_never_patched() {
    let _guard = serialize_jit_tests();
    let (mut cpu, _tables, rip, phys) = straddler_with_a_jump_to_it();
    let mut jit = new_jit();
    jit.translate(&mut cpu, rip, phys).unwrap();
    assert_eq!(jit.link_state_len(), (0, 1));
    let straddler = rip + PAGE - 2;
    cpu.rip = straddler;
    jit.translate(&mut cpu, straddler, phys + PAGE - 2).unwrap();
    assert_eq!(
        jit.link_state_len(),
        (0, 1),
        "the slot waits: the stub checks the second page, a direct jump would not"
    );
    assert_eq!(jit.stats().linked, 0);
}

#[test]
fn a_straddling_block_is_dropped_when_either_of_its_pages_is_written() {
    let _guard = serialize_jit_tests();
    for second_page_written in [false, true] {
        let (mut cpu, _tables, rip, phys) = straddler_with_a_jump_to_it();
        let mut jit = new_jit();
        let straddler = rip + PAGE - 2;
        cpu.rip = straddler;
        let physical = phys + PAGE - 2;
        let code = jit.translate(&mut cpu, straddler, physical).unwrap();
        assert_eq!(
            jit.lookup_straddler(&mut cpu, straddler, physical),
            Some(code)
        );
        let page = if second_page_written {
            (SCRATCH + 2 * PAGE) >> PAGE_SHIFT
        } else {
            phys >> PAGE_SHIFT
        };
        jit.invalidate_page(page, &cpu.page_has_code);
        assert_eq!(
            jit.lookup_straddler(&mut cpu, straddler, physical),
            None,
            "second page written: {second_page_written}"
        );
    }
}

#[test]
fn a_straddling_slot_is_refused_for_another_rip_or_physical_address() {
    let _guard = serialize_jit_tests();
    let (mut cpu, _tables, rip, phys) = straddler_with_a_jump_to_it();
    let mut jit = new_jit();
    let straddler = rip + PAGE - 2;
    cpu.rip = straddler;
    let physical = phys + PAGE - 2;
    jit.translate(&mut cpu, straddler, physical).unwrap();
    assert!(
        jit.lookup_straddler(&mut cpu, straddler, physical)
            .is_some()
    );
    assert_eq!(
        jit.lookup_straddler(&mut cpu, straddler, physical + 0x1000),
        None
    );
    assert_eq!(
        jit.lookup_straddler(&mut cpu, straddler ^ (1 << 40), physical),
        None
    );
    assert_eq!(
        jit.lookup_straddler(&mut cpu, rip, phys),
        None,
        "a different block's slot"
    );
}

#[test]
fn a_straddling_slot_is_refused_when_the_next_page_stops_being_mapped() {
    let _guard = serialize_jit_tests();
    let (mut cpu, mut tables, rip, phys) = straddler_with_a_jump_to_it();
    let mut jit = new_jit();
    let straddler = rip + PAGE - 2;
    cpu.rip = straddler;
    let physical = phys + PAGE - 2;
    jit.translate(&mut cpu, straddler, physical).unwrap();
    assert!(
        jit.lookup_straddler(&mut cpu, straddler, physical)
            .is_some()
    );
    // Not present any more: the walk fails, which counts as "not the same page".
    tables.map_4k(&mut cpu, rip + PAGE, SCRATCH + 2 * PAGE, 0);
    cpu.flush_tlb();
    assert_eq!(jit.lookup_straddler(&mut cpu, straddler, physical), None);
}

#[test]
fn the_validity_table_starts_clear_and_marks_each_block_as_it_is_made() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&[&[NOP; 4][..], &[0xeb, 0x00][..], &[0xf4][..]].concat());
    let mut jit = new_jit();
    assert!(!jit.block_is_valid(0));
    translate_at_code_base(&mut jit, &mut cpu);
    assert!(jit.block_is_valid(0));
    assert!(!jit.block_is_valid(1));
    jit.translate(&mut cpu, CODE_BASE + 6, CODE_BASE + 6)
        .unwrap();
    assert!(jit.block_is_valid(1));
}

#[test]
fn blocks_are_listed_under_the_page_they_were_decoded_from_in_translation_order() {
    let _guard = serialize_jit_tests();
    let code = [
        &[NOP; 2][..],
        &[0xeb, 0x00][..],
        &[NOP; 2][..],
        &[0xeb, 0x00][..],
        &[0xf4][..],
    ]
    .concat();
    let mut cpu = cpu_running(&code);
    let mut jit = new_jit();
    translate_at_code_base(&mut jit, &mut cpu);
    jit.translate(&mut cpu, CODE_BASE + 4, CODE_BASE + 4)
        .unwrap();
    assert_eq!(jit.state().page_blocks[&(CODE_BASE >> PAGE_SHIFT)], [0, 1]);
}

#[test]
fn a_translated_block_survives_the_handle_that_translated_it_being_dropped_mid_run() {
    let _guard = serialize_jit_tests();
    let mut warm = cpu_running(&COUNTDOWN_LOOP);
    let jit = new_jit();
    let survivor = jit.sibling();
    warm.jit = Some(Box::new(jit));
    warm_up_loop(&mut warm);
    warm.jit = None;
    let mut other = cpu_running(&COUNTDOWN_LOOP);
    other.jit = Some(Box::new(survivor));
    run_to_halt(&mut other);
    assert_eq!(other.gpr[1], 0);
}

#[test]
fn a_cpu_running_on_a_sibling_handle_stops_using_blocks_after_a_flush_elsewhere() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    let jit = new_jit();
    let mut flusher = jit.sibling();
    cpu.jit = Some(Box::new(jit));
    warm_up_loop(&mut cpu);
    // Another vCPU fills the buffer and flushes while this one is between blocks.
    flusher.reset();
    run_to_halt(&mut cpu);
    assert_eq!(cpu.gpr[1], 0);
    assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
    assert_eq!(cpu.jit.as_ref().unwrap().stats().resets, 1);
}

#[test]
fn a_chain_does_not_enter_a_block_another_handle_invalidated() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&COUNTDOWN_LOOP);
    let jit = new_jit();
    let writer = jit.sibling();
    cpu.jit = Some(Box::new(jit));
    warm_up_loop(&mut cpu);
    let translated = cpu.jit.as_ref().unwrap().stats().translated;
    // Another vCPU wrote the code page while this one was between blocks: the slots and links
    // this one holds name blocks that must not run again.
    writer.invalidate_page(CODE_BASE >> PAGE_SHIFT, &cpu.page_has_code);
    run_to_halt(&mut cpu);
    assert_eq!(cpu.gpr[1], 0);
    assert_eq!(cpu.icount, COUNTDOWN_LOOP_INSNS);
    assert!(
        cpu.jit.as_ref().unwrap().stats().translated > translated,
        "the loop was translated again, not chained into"
    );
}

#[test]
fn two_blocks_whose_rips_collide_in_the_jump_cache_are_both_found() {
    let _guard = serialize_jit_tests();
    // The same code at two physical pages, reached at two virtual addresses with one hash.
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let other = CODE_BASE + PAGE;
    cpu.ram[other as usize..][..MOV_EAX_ONE.len()].copy_from_slice(&MOV_EAX_ONE);
    let mut jit = new_jit();
    let first_rip = CODE_BASE;
    let mut second_rip = other;
    // Find a rip with the same page offset and slot as the first, on the other physical page.
    while jump_cache_hash(second_rip) != jump_cache_hash(first_rip) {
        // Moves the bits the hash folds in (`rip >> 16`) and leaves the page offset alone.
        second_rip += 1 << JUMP_CACHE_BITS;
    }
    let first = jit.translate(&mut cpu, first_rip, CODE_BASE).unwrap();
    let second = jit.translate(&mut cpu, second_rip, other).unwrap();
    assert_ne!(first, second);
    // The slot now belongs to the second; the first is found again through the block map.
    assert_eq!(jit.lookup(second_rip, other), Some(second));
    assert_eq!(jit.lookup(first_rip, CODE_BASE), Some(first));
    assert_eq!(
        jit.stats().translated,
        2,
        "the eviction cost no translation"
    );
}

#[test]
fn a_lookup_refills_the_slot_so_the_next_one_needs_no_lock() {
    let _guard = serialize_jit_tests();
    let mut cpu = cpu_running(&MOV_EAX_ONE);
    let mut first = new_jit();
    let mut second = first.sibling();
    let code = translate_at_code_base(&mut first, &mut cpu);
    assert_eq!(second.jump_cache[jump_cache_hash(CODE_BASE)].code_addr, 0);
    assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(code));
    let slot = second.jump_cache[jump_cache_hash(CODE_BASE)];
    assert_eq!(
        (slot.rip, slot.phys_addr, slot.code_addr),
        (CODE_BASE, CODE_BASE, code)
    );
    // Holding the lock elsewhere cannot block a hit on the slot.
    let shared = second.shared.clone();
    let _held = shared.lock();
    assert_eq!(second.lookup(CODE_BASE, CODE_BASE), Some(code));
}

#[test]
fn a_straddling_slot_keeps_the_second_page_number_above_the_block_index() {
    let _guard = serialize_jit_tests();
    let (mut cpu, _tables, rip, phys) = straddler_with_a_jump_to_it();
    let mut jit = new_jit();
    // A plain block first, so the straddler's index is not zero.
    jit.translate(&mut cpu, rip, phys).unwrap();
    let straddler = rip + PAGE - 2;
    cpu.rip = straddler;
    jit.translate(&mut cpu, straddler, phys + PAGE - 2).unwrap();
    let slot = jit.jump_cache[jump_cache_hash(straddler)];
    assert_ne!(slot.phys_addr & STRADDLER_TAG, 0);
    assert_eq!(slot.block & SLOT_BLOCK_MASK, 1);
    assert_eq!(
        slot.block >> SLOT_BLOCK_BITS,
        (SCRATCH + 2 * PAGE) >> PAGE_SHIFT
    );
}

#[test]
fn the_slot_encoding_has_room_for_every_block_and_every_physical_page() {
    const {
        assert!(MAX_BLOCKS as u64 <= 1 << SLOT_BLOCK_BITS);
        // 52-bit physical addresses: the page number fits above the block index.
        assert!((1u64 << 40) - 1 <= u64::MAX >> SLOT_BLOCK_BITS);
    }
    assert_eq!(
        STRADDLER_TAG & ((1 << 52) - 1),
        0,
        "no physical address has the tag"
    );
}
