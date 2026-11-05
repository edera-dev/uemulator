//! Differential corpus: every snippet runs on the interpreter (the oracle) and on the host JIT
//! and must leave identical architectural state, instruction count and RAM. On a build without a
//! JIT the comparison is skipped and only the interpreter-side expectations (halt, fault) run.
//! Snippet bytes come from `hack/asm.sh`; the assembly is in the comment above each array.

use crate::boot::KERNEL_LOAD;
use crate::cpu::{SEG_FS, SEG_GS};
use crate::testkit::*;

/// Bytes of RAM at `SCRATCH` filled with pseudo-random data before every run, so loads see
/// distinguishable values and a store to the wrong address shows up in the RAM comparison.
const SEEDED_BYTES: usize = 64 << 10;
const SEED_DATA_STATE: u64 = 0x9e37_79b9_7f4a_7c15;

/// Deterministic xorshift64* generator; the same seeds give the same snippets everywhere.
pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

/// Register values that exercise sign, carry and width boundaries; the pointer-like ones aim into
/// seeded scratch memory so a snippet may use them as base registers.
const SEED_GPRS: [u64; 16] = [
    0x8899_aabb_ccdd_eeff,
    0x0000_0000_0000_0004,
    0xffff_ffff_8000_0000,
    SCRATCH + 0x100,
    STACK_TOP,
    SCRATCH + 0x200,
    SCRATCH + 0x300,
    SCRATCH + 0x400,
    0x7fff_ffff_ffff_ffff,
    0x0000_0000_0000_0001,
    u64::MAX,
    0x0000_0000_8000_0000,
    SCRATCH + 0x800,
    0x00ff_00ff_00ff_00ff,
    0xfedc_ba98_7654_3210,
    0,
];

/// FS and GS bases: inside the seeded scratch data, so `fs:[...]` and `gs:[...]` accesses land on
/// data the base register forms also see.
pub(super) const SEED_FS_BASE: u64 = SCRATCH + 0x2000;
pub(super) const SEED_GS_BASE: u64 = SCRATCH + 0x3000;

pub(super) fn seed_registers_and_memory(harness: &mut Harness) {
    harness.cpu.gpr = SEED_GPRS;
    harness.cpu.seg[SEG_FS].base = SEED_FS_BASE;
    harness.cpu.seg[SEG_GS].base = SEED_GS_BASE;
    let mut rng = Rng(SEED_DATA_STATE);
    let mut data = vec![0u8; SEEDED_BYTES];
    for chunk in data.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes()[..chunk.len()]);
    }
    harness.write(SCRATCH, &data);
}

#[allow(clippy::single_range_in_vec_init)] // one window covering all of RAM
const ALL_RAM: &[std::ops::Range<usize>] = &[0..HARNESS_RAM_BYTES];
/// Low memory up to just above the stack (page tables, scratch data, stack) and the first
/// megabyte of the fake kernel, where the snippet itself lives.
pub(super) const HOT_RAM: &[std::ops::Range<usize>] = &[
    0..0x90_0000,
    KERNEL_LOAD as usize..KERNEL_LOAD as usize + 0x10_0000,
];

/// What differs between the interpreter and the JIT run, or `None` if they agree on everything
/// observable. Without a JIT there is nothing to compare.
pub(super) fn difference(result: &Differential) -> Option<String> {
    difference_in(result, ALL_RAM)
}

/// Like `difference`, comparing only the given RAM ranges. Comparing all 32 MiB costs about as
/// much as running the snippet, so bulk tests compare the windows snippets can reach (`HOT_RAM`).
pub(super) fn difference_in(
    result: &Differential,
    compared_ram: &[std::ops::Range<usize>],
) -> Option<String> {
    let (jit, jit_stop) = result.jit.as_ref()?;
    // A module the engine refuses leaves its blocks to the interpreter, which would agree with
    // itself here: a generator bug must not pass as a match.
    #[cfg(target_arch = "wasm32")]
    if let Some(compiler) = jit.cpu.jit.as_ref()
        && compiler.stats().compile_failures != 0
    {
        return Some(format!(
            "the engine refused a generated module: {:?}",
            compiler.stats()
        ));
    }
    if jit_stop != &result.interpreter_stop {
        return Some(format!(
            "stop reason: jit {jit_stop:?}, interpreter {:?}",
            result.interpreter_stop
        ));
    }
    let (jit_state, oracle_state) = (jit.snapshot(), result.interpreter.snapshot());
    if jit_state != oracle_state {
        return Some(format!(
            "state:\n  jit         {jit_state:x?}\n  interpreter {oracle_state:x?}"
        ));
    }
    if jit.cpu.cr2 != result.interpreter.cpu.cr2 {
        return Some(format!(
            "cr2: jit {:#x}, interpreter {:#x}",
            jit.cpu.cr2, result.interpreter.cpu.cr2
        ));
    }
    // Slice equality is a memcmp, ~100x faster than locating the offset byte by byte, which
    // only happens on the failure path.
    for window in compared_ram {
        let (jit_ram, oracle_ram) = (
            &jit.cpu.ram[window.clone()],
            &result.interpreter.cpu.ram[window.clone()],
        );
        if jit_ram != oracle_ram {
            let offset = jit_ram
                .iter()
                .zip(oracle_ram)
                .position(|(a, b)| a != b)
                .unwrap();
            return Some(format!("RAM differs first at {:#x}", window.start + offset));
        }
    }
    None
}

/// `run_both` (see `testkit`) with JIT execution serialized where the host needs it
/// (`serialize_jit_tests`). Returns once both CPUs are done; the JIT's code buffer is released
/// when the result is dropped.
pub(super) fn run_serialized(code: &[u8], setup: impl Fn(&mut Harness)) -> Differential {
    let _guard = super::serialize_jit_tests();
    run_both(code, setup)
}

/// Runs `code` on both engines and asserts they agree on everything observable.
#[track_caller]
pub(super) fn run_checked(code: &[u8], setup: impl Fn(&mut Harness)) -> Differential {
    let result = run_serialized(code, setup);
    if let Some(difference) = difference(&result) {
        panic!("jit differs from the interpreter, {difference}");
    }
    result
}

/// Both engines agree and the snippet runs to its `hlt`.
#[track_caller]
pub(super) fn check_with(code: &[u8], setup: impl Fn(&mut Harness)) -> Differential {
    let result = run_checked(code, setup);
    assert_eq!(
        result.interpreter_stop,
        Stop::Halted,
        "{}",
        result.interpreter.cpu.dump_regs()
    );
    result
}

#[track_caller]
pub(super) fn check(code: &[u8]) -> Differential {
    check_with(code, seed_registers_and_memory)
}

/// Both engines agree and stop for `expected` (a fault, a limit, ...) instead of halting.
#[track_caller]
pub(super) fn check_stop(code: &[u8], expected: Stop) -> Differential {
    check_stop_with(code, seed_registers_and_memory, expected)
}

#[track_caller]
pub(super) fn check_stop_with(
    code: &[u8],
    setup: impl Fn(&mut Harness),
    expected: Stop,
) -> Differential {
    let result = run_checked(code, setup);
    assert_eq!(
        result.interpreter_stop,
        expected,
        "{}",
        result.interpreter.cpu.dump_regs()
    );
    result
}

/// A snippet assembled from literal bytes and long runs of one byte (padding that pushes code
/// across a page boundary).
pub(super) enum Part<'a> {
    Bytes(&'a [u8]),
    Fill(u8, usize),
}

pub(super) fn parts(parts: &[Part]) -> Vec<u8> {
    let mut code = Vec::new();
    for part in parts {
        match part {
            Part::Bytes(bytes) => code.extend_from_slice(bytes),
            Part::Fill(byte, count) => code.resize(code.len() + count, *byte),
        }
    }
    code
}

/// The undelivered fault a stopped run reports.
pub(super) fn fault(vector: u8, error_code: Option<u32>) -> Stop {
    Stop::Fault {
        vector,
        error_code,
        software: false,
    }
}

/// A software interrupt or breakpoint trap (`int n`, `int3`).
pub(super) fn trap(vector: u8) -> Stop {
    Stop::Fault {
        vector,
        error_code: None,
        software: true,
    }
}

#[test]
fn the_host_has_a_jit_so_the_corpus_is_not_vacuous() {
    let result = check(&[0xf4]);
    assert_eq!(
        result.jit.is_some(),
        cfg!(all(
            feature = "jit",
            any(
                target_arch = "aarch64",
                target_arch = "x86_64",
                target_arch = "wasm32"
            )
        ))
    );
}

#[test]
fn two_rdtsc_in_one_block_leave_icount_and_clock_exact() {
    // nop; rdtsc; mov r8, rax; mov r9, rdx; nop; rdtsc; hlt
    check(&[
        0x90, 0x0f, 0x31, 0x49, 0x89, 0xc0, 0x49, 0x89, 0xd1, 0x90, 0x0f, 0x31, 0xf4,
    ]);
}

#[test]
fn three_rdtsc_in_one_block_leave_icount_and_clock_exact() {
    // rdtsc; mov r8, rax; rdtsc; mov r9, rax; nop; nop; rdtsc; hlt
    check(&[
        0x0f, 0x31, 0x49, 0x89, 0xc0, 0x0f, 0x31, 0x49, 0x89, 0xc1, 0x90, 0x90, 0x0f, 0x31, 0xf4,
    ]);
}

#[test]
fn rdtsc_then_pit_port_read_in_one_block_sees_exact_clock() {
    // mov al, 0x34; out 0x43, al; xor eax, eax; out 0x40, al; out 0x40, al; nop; rdtsc
    // nop; in al, 0x40; mov r8, rax; hlt
    check(&[
        0xb0, 0x34, 0xe6, 0x43, 0x31, 0xc0, 0xe6, 0x40, 0xe6, 0x40, 0x90, 0x0f, 0x31, 0x90, 0xe4,
        0x40, 0x49, 0x89, 0xc0, 0xf4,
    ]);
}

#[test]
fn rdmsr_tsc_after_other_instructions_reads_exact_clock() {
    // mov ecx, 0x10; nop; nop; rdmsr; hlt
    check(&[0xb9, 0x10, 0x00, 0x00, 0x00, 0x90, 0x90, 0x0f, 0x32, 0xf4]);
}

#[test]
fn rdtsc_after_rdmsr_tsc_in_one_block_reads_exact_clock() {
    // mov ecx, 0x10; rdmsr; mov r8, rax; nop; rdtsc; hlt
    check(&[
        0xb9, 0x10, 0x00, 0x00, 0x00, 0x0f, 0x32, 0x49, 0x89, 0xc0, 0x90, 0x0f, 0x31, 0xf4,
    ]);
}

#[test]
fn enter_level0_then_leave_restores_rsp_and_rbp() {
    // mov rbp, 0x1234; enter 32, 0; mov rax, rbp; mov rcx, rsp; leave; hlt
    check(&[
        0x48, 0xc7, 0xc5, 0x34, 0x12, 0x00, 0x00, 0xc8, 0x20, 0x00, 0x00, 0x48, 0x89, 0xe8, 0x48,
        0x89, 0xe1, 0xc9, 0xf4,
    ]);
}

#[test]
fn enter_between_alu_instructions_continues_the_block() {
    // add rax, rcx; enter 16, 0; sub rax, rdx; leave; xor rbx, rbx; hlt
    check(&[
        0x48, 0x01, 0xc8, 0xc8, 0x10, 0x00, 0x00, 0x48, 0x29, 0xd0, 0xc9, 0x48, 0x31, 0xdb, 0xf4,
    ]);
}

#[test]
fn enter_with_unmapped_stack_faults_the_same() {
    // mov rsp, 0x100000000; enter 16, 0; hlt
    check(&[
        0x48, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc8, 0x10, 0x00, 0x00, 0xf4,
    ]);
}

/// Tests that look inside the JIT (block counts, inline coverage), so they exist only where
/// there is one.
#[cfg(all(
    feature = "jit",
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )
))]
mod translation {
    use super::*;
    use crate::jit::JitStats;

    fn stats_of(result: &Differential) -> JitStats {
        let (jit, _) = result.jit.as_ref().expect("this host has a JIT");
        jit.cpu.jit.as_ref().unwrap().stats()
    }

    #[test]
    fn enter_does_not_end_a_block() {
        // one block if enter continues it, two if it ended it
        let result = check(&ENTER_THEN_PUSH_POP);
        assert_eq!(stats_of(&result).translated, 1);
    }

    /// The WebAssembly backend decodes past a conditional branch within the page
    /// (`blocks::decode_block`), so its blocks there run on to the next block end.
    const RUNS_ON_PAST_JCC: bool = cfg!(target_arch = "wasm32");

    #[test]
    fn alu_and_conditional_jump_are_lowered_inline_and_hlt_runs_through_the_helper() {
        let result = check(&INLINE_ALU_THEN_JCC);
        let stats = stats_of(&result);
        // [add, cmp, jb] and [hlt]; or [add, cmp, jb, hlt], whose jb to the hlt is taken with
        // the seeded registers and lands on [hlt] as well
        assert_eq!(stats.translated, 2, "{stats:?}");
        assert_eq!(stats.total, if RUNS_ON_PAST_JCC { 5 } else { 4 });
        assert_eq!(stats.inlined, 3);
    }

    #[test]
    fn a_16_bit_adc_and_cpuid_are_not_lowered_inline_but_a_byte_move_is() {
        let result = check(&HELPER_ONLY_FORMS);
        let stats = stats_of(&result);
        assert_eq!(stats.translated, 1);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.inlined, 1, "only mov al, bl");
    }

    #[test]
    fn a_loop_is_translated_once_however_often_it_runs() {
        let result = check(&COUNTED_LOOP_1000);
        // [mov, add, dec, jnz] on the way in, [add, dec, jnz] for every back edge, [hlt]; where
        // blocks run on past the jnz, the second ends at the hlt and there is no third
        let blocks = if RUNS_ON_PAST_JCC { 2 } else { 3 };
        assert_eq!(stats_of(&result).translated, blocks);
        assert!(result.interpreter.cpu.icount > 3000);
    }

    #[test]
    fn straight_line_code_splits_into_blocks_of_at_most_48_instructions() {
        let result = check(&straight_line_100_nops());
        let stats = stats_of(&result);
        // 101 instructions: 48 + 48 + [4 nops, hlt]
        assert_eq!(stats.translated, 3);
        assert_eq!(stats.total, 101);
        assert_eq!(stats.inlined, 100, "every nop is inline, hlt is not");
    }

    #[test]
    fn inline_adds_are_lowered_without_the_helper() {
        let code = [[ADD_RAX_1; 100].concat(), vec![HLT]].concat();
        let result = check(&code);
        let stats = stats_of(&result);
        assert_eq!(stats.inlined, 100);
        assert_eq!(result.interpreter.cpu.gpr[RAX], SEED_RAX + 100);
    }

    const SEED_RAX: u64 = 0x8899_aabb_ccdd_eeff;

    #[test]
    fn lea_mov_push_pop_and_extensions_are_lowered_inline() {
        let result = check(&LEA_MOV_PUSH_POP_MOVZX_INLINE);
        let stats = stats_of(&result);
        assert_eq!(stats.total, 7);
        assert_eq!(stats.inlined, 6);
    }

    #[test]
    fn undecodable_bytes_become_a_single_marker_instruction_that_faults_when_reached() {
        let result = check_stop(&ONE_UNDECODABLE_BYTE, fault(6, None));
        let stats = stats_of(&result);
        assert_eq!(stats.translated, 1);
        assert_eq!(stats.total, 1);
        assert_eq!(stats.inlined, 0);
    }

    #[test]
    fn call_and_ret_each_end_their_own_block() {
        let result = check(&CALL_RET_PAIR);
        let stats = stats_of(&result);
        assert_eq!(stats.translated, 3, "[call], [ret], [hlt]");
        assert_eq!(stats.inlined, 2);
    }

    #[test]
    fn a_block_ends_at_the_end_of_its_page() {
        let result = check(&nop_on_the_last_byte_of_a_page());
        // [jmp], [nop] alone on the first page, [mov, hlt] on the next
        assert_eq!(stats_of(&result).translated, 3);
    }

    #[test]
    fn running_the_corpus_snippets_never_resets_the_code_buffer() {
        let result = check(&COUNTED_LOOP_1000);
        assert_eq!(stats_of(&result).resets, 0);
    }

    // enter 16, 0; push rax; pop rbx; leave; hlt
    const ENTER_THEN_PUSH_POP: [u8; 8] = [0xc8, 0x10, 0x00, 0x00, 0x50, 0x5b, 0xc9, 0xf4];

    // add rax, rcx; cmp rax, rdx; jb 1f; 1: hlt
    const INLINE_ALU_THEN_JCC: [u8; 9] = [0x48, 0x01, 0xc8, 0x48, 0x39, 0xd0, 0x72, 0x00, 0xf4];

    // adc ax, cx; mov al, bl; cpuid; hlt
    const HELPER_ONLY_FORMS: [u8; 8] = [0x66, 0x11, 0xc8, 0x88, 0xd8, 0x0f, 0xa2, 0xf4];

    // mov ecx, 1000; 1: add rax, rcx; dec rcx; jnz 1b; hlt
    const COUNTED_LOOP_1000: [u8; 14] = [
        0xb9, 0xe8, 0x03, 0x00, 0x00, 0x48, 0x01, 0xc8, 0x48, 0xff, 0xc9, 0x75, 0xf8, 0xf4,
    ];

    // .rept 100; nop; .endr; hlt
    fn straight_line_100_nops() -> Vec<u8> {
        parts(&[Part::Fill(0x90, 100), Part::Bytes(&[0xf4])])
    }

    // add rax, 1
    const ADD_RAX_1: [u8; 4] = [0x48, 0x83, 0xc0, 0x01];

    // jmp f; .org 3583; f: nop; mov eax, 1; hlt
    fn nop_on_the_last_byte_of_a_page() -> Vec<u8> {
        parts(&[
            Part::Bytes(&[0xe9, 0xfa, 0x0d]),
            Part::Fill(0x00, 3580),
            Part::Bytes(&[0x90, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xf4]),
        ])
    }

    // lea rax, [rbx + rcx*4 + 8]; mov rdx, rax; push rdx; pop rsi
    // movzx edi, byte ptr [rbx]; movsxd r8, dword ptr [rbx]; hlt
    const LEA_MOV_PUSH_POP_MOVZX_INLINE: [u8; 17] = [
        0x48, 0x8d, 0x44, 0x8b, 0x08, 0x48, 0x89, 0xc2, 0x52, 0x5e, 0x0f, 0xb6, 0x3b, 0x4c, 0x63,
        0x03, 0xf4,
    ];

    // .byte 0x06
    const ONE_UNDECODABLE_BYTE: [u8; 1] = [0x06];

    // call f; hlt; f: ret
    const CALL_RET_PAIR: [u8; 7] = [0xe8, 0x01, 0x00, 0x00, 0x00, 0xf4, 0xc3];
}
