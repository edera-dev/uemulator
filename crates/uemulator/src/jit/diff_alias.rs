//! Position independence. A translated block is keyed by physical address alone and runs at every
//! virtual address its page is mapped at, so everything it computes from `rip` (return addresses,
//! `rip`-relative operands, branch targets, the `rip` a fault reports) has to come from the
//! address of the page it is running in, not from the one it was translated at. Each case maps
//! one physical page at two virtual addresses and checks that one translation serves both.

use crate::cpu::test_support::*;
use crate::testkit::*;

/// Two virtual addresses of the same physical page. They differ in a few bits, so xor-ing an
/// address with `ALIAS_DELTA` switches between them. The page numbers also differ in their low
/// bits: the soft TLB is direct-mapped on them, and aliases sharing a slot would evict each other
/// and keep the chain stub from ever hitting, which is the path these tests are about.
pub(super) const ALIAS_A: u64 = 0x0000_2000_0000_0000;
pub(super) const ALIAS_B: u64 = 0x0000_2100_0000_1000;
const ALIAS_DELTA: u64 = ALIAS_A ^ ALIAS_B;
const CODE_PAGE: u64 = SCRATCH + 0x2000;
/// Reads here page-fault: nothing maps it.
const UNMAPPED: u64 = 0x0000_5000_0000_0000;

const PAGE_FAULT: u8 = 14;

/// The code of `a_loop_that_bounces_between_the_aliases_...`. Blocks start at 0x00 (jumps to the
/// other alias with an indirect `jmp`, the one transfer that always goes through the chain stub),
/// 0x20 (counts down, loops to 0x00 inside the page) and 0x25 (records where it ended).
/// ```text
/// 00: lea rax, [rip]        ; this alias + 7
/// 07: xor rax, r10          ; the other alias + 7
/// 0a: add rax, 0x19
/// 0e: jmp rax               ; the other alias + 0x20
/// 20: dec r8
/// 23: jnz 00
/// 25: lea rbx, [rip]        ; this alias + 0x2c
/// 2c: hlt
/// ```
const BOUNCE: &[(u64, &[u8])] = &[
    (0x00, &[0x48, 0x8d, 0x05, 0, 0, 0, 0]),
    (0x07, &[0x4c, 0x31, 0xd0]),
    (0x0a, &[0x48, 0x83, 0xc0, 0x19]),
    (0x0e, &[0xff, 0xe0]),
    (0x20, &[0x49, 0xff, 0xc8]),
    (0x23, &[0x75, 0xdb]),
    (0x25, &[0x48, 0x8d, 0x1d, 0, 0, 0, 0]),
    (0x2c, &[0xf4]),
];
const BOUNCE_END_OFFSET: u64 = 0x2c;

/// ```text
/// 00: call 20
/// 05: lea rbx, [rip]        ; this alias + 0x0c
/// 0c: test rdi, rdi
/// 0f: jnz 18
/// 11: mov ecx, 1
/// 16: hlt
/// 18: mov rax, [rsi]        ; faults when rsi is unmapped
/// 1b: hlt
/// 20: pop rax               ; the return address
/// 21: push rax
/// 22: ret
/// ```
const CALL_AND_FAULT: &[(u64, &[u8])] = &[
    (0x00, &[0xe8, 0x1b, 0, 0, 0]),
    (0x05, &[0x48, 0x8d, 0x1d, 0, 0, 0, 0]),
    (0x0c, &[0x48, 0x85, 0xff]),
    (0x0f, &[0x75, 0x07]),
    (0x11, &[0xb9, 1, 0, 0, 0]),
    (0x16, &[0xf4]),
    (0x18, &[0x48, 0x8b, 0x06]),
    (0x1b, &[0xf4]),
    (0x20, &[0x58, 0x50, 0xc3]),
];

/// A harness whose paging maps `CODE_PAGE` at both aliases (and the stack), holding `program`.
fn aliased(program: &[(u64, &[u8])], jit: bool) -> Option<Harness> {
    let mut harness = Harness::new(&[HLT]);
    if jit && !harness.enable_jit() {
        return None;
    }
    let mut tables = PageTables::new(&mut harness.cpu);
    for alias in [ALIAS_A, ALIAS_B] {
        tables.map_4k(
            &mut harness.cpu,
            alias,
            CODE_PAGE,
            PTE_PRESENT_WRITABLE_USER,
        );
    }
    let stack_page = STACK_TOP - PAGE;
    tables.map_4k(
        &mut harness.cpu,
        stack_page,
        stack_page,
        PTE_PRESENT_WRITABLE_USER,
    );
    for (offset, bytes) in program {
        harness.write(CODE_PAGE + offset, bytes);
    }
    Some(harness)
}

fn translated(harness: &Harness) -> u64 {
    harness.cpu.jit.as_ref().unwrap().stats().translated
}

fn bounce(harness: &mut Harness, count: u64) -> Stop {
    harness.cpu.rip = ALIAS_A;
    harness.cpu.gpr[R8] = count;
    harness.cpu.gpr[R10] = ALIAS_DELTA;
    harness.run(1_000_000)
}

/// The chain stub is what moves the page base from one alias to the other here: the first visit
/// to each alias's block goes through the run loop, the later ones through the stub. A stale page
/// base sends the `lea` at 0x00 back to the wrong alias and the loop ends somewhere else.
#[test]
fn a_loop_that_bounces_between_the_aliases_ends_where_the_interpreter_ends() {
    for count in [6, 7] {
        let mut interpreted = aliased(BOUNCE, false).unwrap();
        assert_eq!(bounce(&mut interpreted, count), Stop::Halted);
        let Some(mut jitted) = aliased(BOUNCE, true) else {
            return;
        };
        assert_eq!(bounce(&mut jitted, count), Stop::Halted);
        let expected_alias = if count % 2 == 0 { ALIAS_A } else { ALIAS_B };
        assert_eq!(
            interpreted.cpu.gpr[RBX],
            expected_alias + BOUNCE_END_OFFSET,
            "the oracle"
        );
        assert_eq!(
            jitted.cpu.gpr[RBX], interpreted.cpu.gpr[RBX],
            "count {count}"
        );
        assert_eq!(jitted.cpu.gpr[R8], 0);
        assert_eq!(jitted.cpu.gpr[RAX], interpreted.cpu.gpr[RAX]);
        // The WebAssembly backend runs the block at 0x20 on past its jnz, so 0x25 is no block
        // of its own there (`blocks::decode_block`).
        let blocks = if cfg!(target_arch = "wasm32") { 2 } else { 3 };
        assert_eq!(
            translated(&jitted),
            blocks,
            "both aliases ran the blocks translated for the first"
        );
    }
}

/// Runs `CALL_AND_FAULT` at `alias` down the path `fault` picks and returns the harness's stop.
fn call_and_fault_at(harness: &mut Harness, alias: u64, fault: bool) -> Stop {
    harness.cpu.halted = false;
    harness.cpu.rip = alias;
    harness.cpu.gpr[RSP] = STACK_TOP;
    harness.cpu.gpr[RDI] = fault as u64;
    harness.cpu.gpr[RSI] = UNMAPPED;
    harness.cpu.gpr[RAX] = 0;
    harness.cpu.gpr[RBX] = 0;
    harness.cpu.gpr[RCX] = 0;
    harness.run(1_000_000)
}

#[test]
fn return_addresses_rip_relative_operands_and_fault_rips_follow_the_alias_being_run() {
    let Some(mut harness) = aliased(CALL_AND_FAULT, true) else {
        return;
    };
    let mut oracle = aliased(CALL_AND_FAULT, false).unwrap();
    let mut blocks = None;
    // Each alias twice, and the translation made for the first must serve every later run.
    for alias in [ALIAS_A, ALIAS_B, ALIAS_B, ALIAS_A] {
        for fault in [false, true] {
            let stop = call_and_fault_at(&mut harness, alias, fault);
            let expected_stop = call_and_fault_at(&mut oracle, alias, fault);
            assert_eq!(stop, expected_stop, "alias {alias:#x} fault {fault}");
            if fault {
                assert!(
                    matches!(
                        stop,
                        Stop::Fault {
                            vector: PAGE_FAULT,
                            ..
                        }
                    ),
                    "{stop:?}"
                );
                assert_eq!(harness.cpu.rip, alias + 0x18, "the faulting instruction");
                assert_eq!(harness.cpu.cr2, UNMAPPED);
            } else {
                assert_eq!(stop, Stop::Halted);
                assert_eq!(
                    harness.cpu.gpr[RAX],
                    alias + 0x05,
                    "the pushed return address"
                );
                assert_eq!(harness.cpu.gpr[RBX], alias + 0x0c, "a rip-relative lea");
                assert_eq!(harness.cpu.gpr[RCX], 1);
            }
            assert_eq!(harness.cpu.gpr[RAX], oracle.cpu.gpr[RAX]);
            assert_eq!(harness.cpu.gpr[RBX], oracle.cpu.gpr[RBX]);
            assert_eq!(harness.cpu.rip, oracle.cpu.rip);
        }
        match blocks {
            None => blocks = Some(translated(&harness)),
            Some(count) => assert_eq!(translated(&harness), count, "alias {alias:#x}"),
        }
    }
}

/// Extra virtual pages next to the code page, at `alias + delta` for both aliases.
pub(super) struct Extra {
    pub delta: i64,
    /// What the page holds from its start.
    pub bytes: &'static [u8],
    /// What alias B's page holds instead, if it should be another physical page than alias A's.
    pub alias_b_bytes: Option<&'static [u8]>,
}

pub(super) struct Init {
    pub register: usize,
    pub value: u64,
}

/// One program for the aliased code page: where it starts in the page, the extra pages it
/// reaches, and the registers it starts with (`rsp` is `STACK_TOP` unless given).
pub(super) struct Case {
    pub start: u64,
    pub code: &'static [u8],
    pub extras: &'static [Extra],
    pub init: &'static [Init],
    /// Every run after the first translates nothing new (false when the program rewrites itself
    /// or reads a page that differs per alias).
    pub same_translation: bool,
}

/// RAM the cases may use freely (`rep stosb` and friends), identity mapped.
const SCRATCH_PAGES: u64 = 4;
const EXTRA_POOL: u64 = SCRATCH + 0x20000;
const EXTRA_STRIDE: u64 = 0x2000;

fn extra_physical(index: usize, alias_b: bool) -> u64 {
    EXTRA_POOL + index as u64 * EXTRA_STRIDE + if alias_b { PAGE } else { 0 }
}

fn aliased_case(case: &Case, jit: bool) -> Option<Harness> {
    let mut harness = Harness::new(&[HLT]);
    if jit && !harness.enable_jit() {
        return None;
    }
    let mut tables = PageTables::new(&mut harness.cpu);
    let flags = PTE_PRESENT_WRITABLE_USER;
    for (alias, alias_b) in [(ALIAS_A, false), (ALIAS_B, true)] {
        tables.map_4k(&mut harness.cpu, alias, CODE_PAGE, flags);
        for (index, extra) in case.extras.iter().enumerate() {
            let physical = match extra.alias_b_bytes {
                Some(_) => extra_physical(index, alias_b),
                None => extra_physical(index, false),
            };
            let virtual_page = alias.wrapping_add(extra.delta as u64);
            tables.map_4k(&mut harness.cpu, virtual_page, physical, flags);
        }
    }
    let stack_page = STACK_TOP - PAGE;
    tables.map_4k(&mut harness.cpu, stack_page, stack_page, flags);
    for page in 0..SCRATCH_PAGES {
        let address = SCRATCH + page * PAGE;
        tables.map_4k(&mut harness.cpu, address, address, flags);
    }
    harness.write(CODE_PAGE + case.start, case.code);
    for (index, extra) in case.extras.iter().enumerate() {
        harness.write(extra_physical(index, false), extra.bytes);
        if let Some(bytes) = extra.alias_b_bytes {
            harness.write(extra_physical(index, true), bytes);
        }
    }
    Some(harness)
}

fn start_case(harness: &mut Harness, case: &Case, alias: u64) {
    harness.cpu.halted = false;
    harness.cpu.gpr = [0; 16];
    harness.cpu.gpr[RSP] = STACK_TOP;
    for init in case.init {
        harness.cpu.gpr[init.register] = init.value;
    }
    harness.cpu.rip = alias + case.start;
}

/// The bytes a case can have changed: the code page, the extra pages and the scratch data.
fn case_memory(harness: &Harness, case: &Case) -> Vec<Vec<u8>> {
    let mut regions = vec![harness.read(CODE_PAGE, PAGE as usize)];
    for (index, extra) in case.extras.iter().enumerate() {
        regions.push(harness.read(extra_physical(index, false), PAGE as usize));
        if extra.alias_b_bytes.is_some() {
            regions.push(harness.read(extra_physical(index, true), PAGE as usize));
        }
    }
    regions.push(harness.read(SCRATCH, (SCRATCH_PAGES * PAGE) as usize));
    regions
}

/// Runs `case` at alias A, B, B and A again on the interpreter and on the JIT. Every run must
/// stop the same way and leave the same registers, `rip` and memory: the oracle computes every
/// address from the alias it runs at, so a block that kept an address from the alias it was
/// translated at (or from the one it last came from) differs. With `same_translation` the first
/// run translates all there is and the others find it.
pub(super) fn check_aliases(case: &Case) {
    check_aliases_with(case, &|_| {});
}

/// `check_aliases` with `setup` run on both CPUs before every run (registers, flags and memory
/// beyond what `case` says).
pub(super) fn check_aliases_with(case: &Case, setup: &dyn Fn(&mut Harness)) {
    let mut oracle = aliased_case(case, false).unwrap();
    let Some(mut jitted) = aliased_case(case, true) else {
        return;
    };
    let mut translated_after_first = None;
    for (run, alias) in [ALIAS_A, ALIAS_B, ALIAS_B, ALIAS_A].into_iter().enumerate() {
        start_case(&mut oracle, case, alias);
        start_case(&mut jitted, case, alias);
        setup(&mut oracle);
        setup(&mut jitted);
        let expected = oracle.run(1_000_000);
        let actual = jitted.run(1_000_000);
        let place = format!("run {run} at alias {alias:#x}");
        assert_eq!(actual, expected, "{place}");
        assert_eq!(jitted.cpu.gpr, oracle.cpu.gpr, "{place}");
        assert_eq!(jitted.cpu.rip, oracle.cpu.rip, "{place}");
        assert_eq!(
            jitted.cpu.eflags_full() & 0x8d5,
            oracle.cpu.eflags_full() & 0x8d5,
            "{place}"
        );
        assert_eq!(
            case_memory(&jitted, case),
            case_memory(&oracle, case),
            "{place}"
        );
        if case.same_translation {
            let count = translated(&jitted);
            match translated_after_first {
                None => translated_after_first = Some(count),
                Some(first) => assert_eq!(count, first, "{place} translated again"),
            }
        }
    }
}
