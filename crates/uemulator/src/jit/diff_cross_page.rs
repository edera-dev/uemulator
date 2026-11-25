//! Direct links between pages (`Gen::fixed_exit`, `LinkSite::phys_page`). A block that jumps to a
//! fixed address in another page is linked to the block that address mapped to when the jumping
//! block was translated, behind a check of the exec TLB: the link is taken only while the target
//! page still maps to the same host page. Each case runs a loop whose two halves are in different
//! pages, warms it up so the links exist, then changes something under it (the mapping of the
//! second page, its code) and compares the rest of the run with the interpreter's.

use crate::cpu::test_support::*;
use crate::testkit::*;

/// The two virtual pages of the loop, far enough apart that their soft-TLB slots differ.
const PAGE_A: u64 = 0x0000_3000_0000_0000;
const PAGE_B: u64 = PAGE_A + 0x5000;
/// Physical frames: the first half, the second half, and another second half to remap to.
const FRAME_A: u64 = SCRATCH + 0x6000;
const FRAME_B: u64 = SCRATCH + 0x7000;
const FRAME_B_OTHER: u64 = SCRATCH + 0x8000;

const JMP_REL32: u8 = 0xe9;
const JNZ_REL32: [u8; 2] = [0x0f, 0x85];
const JMP_LENGTH: u64 = 5;

fn rel32(from_next: u64, to: u64) -> [u8; 4] {
    (to.wrapping_sub(from_next) as i32).to_le_bytes()
}

/// `jmp PAGE_B` at `PAGE_A`.
fn first_half() -> Vec<u8> {
    [&[JMP_REL32][..], &rel32(PAGE_A + JMP_LENGTH, PAGE_B)].concat()
}

/// At `PAGE_B`: `add rax, increment; dec r8; jnz PAGE_A; hlt`.
fn second_half(increment: u8) -> Vec<u8> {
    let add = [0x48, 0x83, 0xc0, increment];
    let dec_r8 = [0x49, 0xff, 0xc8];
    let jnz_next = PAGE_B + (add.len() + dec_r8.len() + 6) as u64;
    [
        &add[..],
        &dec_r8,
        &JNZ_REL32,
        &rel32(jnz_next, PAGE_A),
        &[HLT],
    ]
    .concat()
}

struct Rig {
    harness: Harness,
    tables: PageTables,
}

fn rig(jit: bool) -> Option<Rig> {
    let mut harness = Harness::new(&[HLT]);
    if jit && !harness.enable_jit() {
        return None;
    }
    let mut tables = PageTables::new(&mut harness.cpu);
    tables.map_4k(&mut harness.cpu, PAGE_A, FRAME_A, PTE_PRESENT_WRITABLE_USER);
    tables.map_4k(&mut harness.cpu, PAGE_B, FRAME_B, PTE_PRESENT_WRITABLE_USER);
    let stack_page = STACK_TOP - PAGE;
    tables.map_4k(
        &mut harness.cpu,
        stack_page,
        stack_page,
        PTE_PRESENT_WRITABLE_USER,
    );
    harness.write(FRAME_A, &first_half());
    harness.write(FRAME_B, &second_half(1));
    harness.write(FRAME_B_OTHER, &second_half(0x10));
    Some(Rig { harness, tables })
}

/// Runs the loop `iterations` times from `PAGE_A` to its `hlt`; returns how many times the run
/// loop entered generated code (or stepped the interpreter).
fn run_loop(harness: &mut Harness, iterations: u64) -> u64 {
    harness.cpu.halted = false;
    harness.cpu.rip = PAGE_A;
    harness.cpu.gpr[R8] = iterations;
    harness.cpu.next_event = u64::MAX;
    let mut entries = 0;
    while !harness.cpu.halted {
        harness.cpu.interrupt_shadow = false;
        harness.cpu.icount += 1;
        harness.cpu.dispatch_step().unwrap();
        entries += 1;
        assert!(entries < 100_000, "{}", harness.cpu.dump_regs());
    }
    entries
}

fn stats(harness: &Harness) -> crate::jit::JitStats {
    harness.cpu.jit.as_ref().unwrap().stats()
}

#[track_caller]
fn assert_same(jitted: &Harness, interpreted: &Harness) {
    assert_eq!(jitted.snapshot(), interpreted.snapshot());
}

#[test]
fn a_loop_across_two_pages_runs_through_its_links_without_the_jump_cache() {
    let mut interpreted = rig(false).unwrap();
    let Some(mut jitted) = rig(true) else {
        return;
    };
    run_loop(&mut interpreted.harness, 20);
    run_loop(&mut jitted.harness, 20);
    assert_same(&jitted.harness, &interpreted.harness);
    // Both exits are linked: `jmp PAGE_B` (once PAGE_B's block exists and its page is in the
    // TLB) and `jnz PAGE_A`.
    assert!(
        stats(&jitted.harness).cross_page_linked >= 2,
        "{:?}",
        stats(&jitted.harness)
    );
    // The chain stub would miss on every transition now and go back to the run loop.
    jitted.harness.cpu.jit.as_mut().unwrap().clear_jump_cache();
    let translated = stats(&jitted.harness).translated;
    let entries = run_loop(&mut jitted.harness, 1000);
    run_loop(&mut interpreted.harness, 1000);
    assert_same(&jitted.harness, &interpreted.harness);
    assert_eq!(entries, 1, "the whole loop ran through its links");
    assert_eq!(stats(&jitted.harness).translated, translated);
}

/// Remapping the second page (and invalidating its TLB entry, as the guest must) makes the link's
/// check fail, so the loop runs the code the page maps to now.
#[test]
fn a_link_is_not_taken_once_its_target_page_maps_elsewhere() {
    let mut interpreted = rig(false).unwrap();
    let Some(mut jitted) = rig(true) else {
        return;
    };
    for rig in [&mut interpreted, &mut jitted] {
        run_loop(&mut rig.harness, 20);
        rig.tables.map_4k(
            &mut rig.harness.cpu,
            PAGE_B,
            FRAME_B_OTHER,
            PTE_PRESENT_WRITABLE_USER,
        );
        rig.harness.cpu.invlpg(PAGE_B);
        run_loop(&mut rig.harness, 20);
    }
    assert_same(&jitted.harness, &interpreted.harness);
    assert_eq!(jitted.harness.cpu.gpr[RAX], 20 + 20 * 0x10);
    // And back: the first links are still there, and right again.
    for rig in [&mut interpreted, &mut jitted] {
        rig.tables.map_4k(
            &mut rig.harness.cpu,
            PAGE_B,
            FRAME_B,
            PTE_PRESENT_WRITABLE_USER,
        );
        rig.harness.cpu.invlpg(PAGE_B);
        run_loop(&mut rig.harness, 20);
    }
    assert_same(&jitted.harness, &interpreted.harness);
    assert_eq!(jitted.harness.cpu.gpr[RAX], 40 + 20 * 0x10);
}

/// The TLB entry is gone (a flush, a CR3 write) but the page maps where it did: the check fails
/// until the run loop refills the entry, then links are taken again.
#[test]
fn a_flushed_tlb_sends_the_link_through_the_run_loop_once() {
    let mut interpreted = rig(false).unwrap();
    let Some(mut jitted) = rig(true) else {
        return;
    };
    for rig in [&mut interpreted, &mut jitted] {
        run_loop(&mut rig.harness, 20);
        rig.harness.cpu.flush_tlb();
    }
    run_loop(&mut interpreted.harness, 200);
    let entries = run_loop(&mut jitted.harness, 200);
    assert_same(&jitted.harness, &interpreted.harness);
    assert!(entries <= 4, "{entries}");
}

/// A write to the target page invalidates its blocks, which unlinks every slot leading into them;
/// the next pass translates the new code and links to it.
#[test]
fn writing_the_target_page_unlinks_the_cross_page_links_into_it() {
    let mut interpreted = rig(false).unwrap();
    let Some(mut jitted) = rig(true) else {
        return;
    };
    for rig in [&mut interpreted, &mut jitted] {
        run_loop(&mut rig.harness, 20);
    }
    let before = stats(&jitted.harness);
    for rig in [&mut interpreted, &mut jitted] {
        // What a guest store to the page does after writing it.
        rig.harness.write(FRAME_B, &second_half(3));
        rig.harness.cpu.code_written((FRAME_B >> 12) as usize);
        rig.harness.cpu.block_exit_requested = false;
    }
    assert!(stats(&jitted.harness).unlinked > before.unlinked);
    for rig in [&mut interpreted, &mut jitted] {
        run_loop(&mut rig.harness, 20);
    }
    assert_same(&jitted.harness, &interpreted.harness);
    assert_eq!(jitted.harness.cpu.gpr[RAX], 20 + 20 * 3);
    assert!(stats(&jitted.harness).cross_page_linked > before.cross_page_linked);
}
