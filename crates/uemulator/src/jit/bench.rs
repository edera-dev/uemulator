//! Microbenchmarks of the JIT, one guest loop per cost the generators pay: register ALU, loads
//! and stores, branches inside a block, call/ret, indirect jumps, `rep stos`, glibc-shaped
//! AVX2 and AVX-512 string loops, and SSE/AVX-512 floating point (packed multiply-add, square
//! root and divide, conversions, FMA). All `#[ignore]`;
//! run them on each backend with
//!
//!   cargo test --release -p uemulator --lib jit::bench -- --ignored --nocapture --test-threads=1
//!
//! (add `--target x86_64-apple-darwin` for the x86-64 backend under Rosetta, or
//! `--target wasm32-wasip1` for the WebAssembly one under Node). Each prints `bench <name>
//! <backend> <ns per guest instruction> <guest instructions>`. The loops run in slices of
//! `EVENT_SLICE` instructions, like `Cpu::run` between event checks, so a chain that stops at the
//! deadline and the run loop's re-entry are part of what is measured. Before timing, a short
//! run of the same loop is checked against the interpreter, so a fast but wrong backend does not
//! report a number.

use crate::testkit::*;
use std::time::Instant;

/// `Cpu::run`'s `EVENT_INTERVAL`: how far a chain of blocks runs before the run loop takes over.
const EVENT_SLICE: u64 = 4096;
const CHECK_ITERATIONS: u64 = 2000;
/// About 0.1 to 0.3 s natively per loop.
const TIMED_ITERATIONS: u64 = 10_000_000;
const R10: usize = 10;
const DATA: u64 = SCRATCH + 0x3000;
/// A linear page in `CODE_BASE`'s soft-TLB slot, so every data access through it evicts the code
/// page's exec tag. Identity-mapped RAM is too small for one (the slots repeat every 16 MiB), so
/// `setup` maps the 2 MiB around it onto `ALIAS_FRAME`, free harness RAM.
const ALIAS_DATA: u64 = 0x4100_0000;
const ALIAS_FRAME: u64 = 0x120_0000;
const PAGE_BYTES: u64 = 4096;
const _: () = assert!(
    (ALIAS_DATA / PAGE_BYTES) % crate::cpu::TLB_ENTRIES_PER_MODE as u64
        == (CODE_BASE / PAGE_BYTES) % crate::cpu::TLB_ENTRIES_PER_MODE as u64
);
/// `boot::load` lays out PML4, PDPT and the page directories back to back from `cr3`.
const FIRST_PAGE_DIRECTORY_OFFSET: u64 = 2 * PAGE_BYTES;
const LARGE_PAGE_SHIFT: u32 = 21;
const PDE_PRESENT_WRITABLE_LARGE: u64 = 1 | 2 | 1 << 7;
/// A page of its own for the far call target, so call and ret cross pages.
const FAR_TARGET: u64 = SCRATCH + 0x8_0000;
/// `cpu::sys`'s CR4 bits, which `cpu` re-exports only to the JIT builds; this module also builds
/// without the `jit` feature.
const CR4_OSFXSR: u64 = 1 << 9;
const CR4_OSXSAVE: u64 = 1 << 18;

// 1: add rax, rbx; xor rdx, rax; add rbx, 3; dec rcx; jnz 1b; hlt
const REGISTER_LOOP: &[u8] = &[
    0x48, 0x01, 0xd8, 0x48, 0x31, 0xc2, 0x48, 0x83, 0xc3, 0x03, 0x48, 0xff, 0xc9, 0x75, 0xf1, 0xf4,
];

// 2: mov r8, 512; 1: mov rax, [rsi + r8*8]; add [rdi + r8*8], rax; dec r8; jnz 1b;
// dec rcx; jnz 2b; hlt   (rcx counts 512-iteration passes)
const LOAD_STORE_LOOP: &[u8] = &[
    0x49, 0xc7, 0xc0, 0x00, 0x02, 0x00, 0x00, 0x4a, 0x8b, 0x04, 0xc6, 0x4a, 0x01, 0x04, 0xc7, 0x49,
    0xff, 0xc8, 0x75, 0xf3, 0x48, 0xff, 0xc9, 0x75, 0xe7, 0xf4,
];
const LOAD_STORE_INNER: u64 = 512;

// 1: test rax, 1; jnz 2f; add rbx, 1; 2: cmp rbx, rdx; je 3f; add rdx, 2;
// 3: test rbx, 0x40000000; jnz 4f; xor rsi, rbx; 4: inc rax; inc rax; dec rcx; jnz 1b; hlt
const BRANCHY_LOOP: &[u8] = &[
    0x48, 0xa9, 0x01, 0x00, 0x00, 0x00, 0x75, 0x04, 0x48, 0x83, 0xc3, 0x01, 0x48, 0x39, 0xd3, 0x74,
    0x04, 0x48, 0x83, 0xc2, 0x02, 0x48, 0xf7, 0xc3, 0x00, 0x00, 0x00, 0x40, 0x75, 0x03, 0x48, 0x31,
    0xde, 0x48, 0xff, 0xc0, 0x48, 0xff, 0xc0, 0x48, 0xff, 0xc9, 0x75, 0xd4, 0xf4,
];

// 1: call 2f; dec rcx; jnz 1b; hlt; 2: add rax, 1; ret
const CALL_RET_LOOP: &[u8] = &[
    0xe8, 0x06, 0x00, 0x00, 0x00, 0x48, 0xff, 0xc9, 0x75, 0xf6, 0xf4, 0x48, 0x83, 0xc0, 0x01, 0xc3,
];

// 1: call r10; dec rcx; jnz 1b; hlt
const FAR_CALL_LOOP: &[u8] = &[0x41, 0xff, 0xd2, 0x48, 0xff, 0xc9, 0x75, 0xf8, 0xf4];
// add rax, 1; ret
const FAR_FUNCTION: &[u8] = &[0x48, 0x83, 0xc0, 0x01, 0xc3];

// Fills a table at [rsi] with four targets, then cycles `jmp [rsi + (rax & 3) * 8]` through them.
const INDIRECT_LOOP: &[u8] = &[
    0x4c, 0x8d, 0x0d, 0x2f, 0x00, 0x00, 0x00, 0x4c, 0x89, 0x0e, 0x4c, 0x8d, 0x0d, 0x2b, 0x00, 0x00,
    0x00, 0x4c, 0x89, 0x4e, 0x08, 0x4c, 0x8d, 0x0d, 0x26, 0x00, 0x00, 0x00, 0x4c, 0x89, 0x4e, 0x10,
    0x4c, 0x8d, 0x0d, 0x21, 0x00, 0x00, 0x00, 0x4c, 0x89, 0x4e, 0x18, 0x31, 0xc0, 0x48, 0x89, 0xc2,
    0x83, 0xe2, 0x03, 0xff, 0x24, 0xd6, 0x48, 0x83, 0xc3, 0x01, 0xeb, 0x10, 0x48, 0x83, 0xc3, 0x02,
    0xeb, 0x0a, 0x48, 0x83, 0xc3, 0x03, 0xeb, 0x04, 0x48, 0x83, 0xc3, 0x04, 0x48, 0xff, 0xc0, 0x48,
    0xff, 0xc9, 0x75, 0xd9, 0xf4,
];

// 1: mov rdi, rsi; mov r8, rcx; mov ecx, 64; rep stosb; mov rcx, r8; add rbx, 1; dec rcx;
// jnz 1b; hlt
const REP_STOS_LOOP: &[u8] = &[
    0x48, 0x89, 0xf7, 0x49, 0x89, 0xc8, 0xb9, 0x40, 0x00, 0x00, 0x00, 0xf3, 0xaa, 0x4c, 0x89, 0xc1,
    0x48, 0x83, 0xc3, 0x01, 0x48, 0xff, 0xc9, 0x75, 0xe7, 0xf4,
];

// glibc's strlen-avx2 core: 32 bytes a step through vpcmpeqb against zero and vpmovmskb, over
// 4 KiB without a zero. vpxor xmm0, xmm0, xmm0; 2: mov rdx, rsi; mov r8d, 128;
// 1: vpcmpeqb ymm1, ymm0, [rdx]; vpmovmskb eax, ymm1; test eax, eax; jnz 3f; add rdx, 32;
// dec r8d; jnz 1b; 3: dec rcx; jnz 2b; hlt
const AVX2_STRLEN_LOOP: &[u8] = &[
    0xc5, 0xf9, 0xef, 0xc0, 0x48, 0x89, 0xf2, 0x41, 0xb8, 0x80, 0x00, 0x00, 0x00, 0xc5, 0xfd, 0x74,
    0x0a, 0xc5, 0xfd, 0xd7, 0xc1, 0x85, 0xc0, 0x75, 0x09, 0x48, 0x83, 0xc2, 0x20, 0x41, 0xff, 0xc8,
    0x75, 0xeb, 0x48, 0xff, 0xc9, 0x75, 0xdd, 0xf4,
];

// A 4 KiB copy 32 bytes a step. 2: xor r8d, r8d; 1: vmovdqu ymm0, [rsi + r8];
// vmovdqu [rdi + r8], ymm0; add r8, 32; cmp r8, 4096; jb 1b; dec rcx; jnz 2b; hlt
const AVX2_MEMCPY_LOOP: &[u8] = &[
    0x45, 0x31, 0xc0, 0xc4, 0xa1, 0x7e, 0x6f, 0x04, 0x06, 0xc4, 0xa1, 0x7e, 0x7f, 0x04, 0x07, 0x49,
    0x83, 0xc0, 0x20, 0x49, 0x81, 0xf8, 0x00, 0x10, 0x00, 0x00, 0x72, 0xe7, 0x48, 0xff, 0xc9, 0x75,
    0xdf, 0xf4,
];

// glibc's strlen-evex core: 64 bytes a step through vpcmpeqb into an opmask and kmovq.
// vpxorq zmm16, zmm16, zmm16; 2: mov rdx, rsi; mov r8d, 64; 1: vpcmpeqb k0, zmm16, [rdx];
// kmovq rax, k0; test rax, rax; jnz 3f; add rdx, 64; dec r8d; jnz 1b; 3: dec rcx; jnz 2b; hlt
const EVEX_STRLEN_LOOP: &[u8] = &[
    0x62, 0xa1, 0xfd, 0x40, 0xef, 0xc0, 0x48, 0x89, 0xf2, 0x41, 0xb8, 0x40, 0x00, 0x00, 0x00, 0x62,
    0xf1, 0x7d, 0x40, 0x74, 0x02, 0xc4, 0xe1, 0xfb, 0x93, 0xc0, 0x48, 0x85, 0xc0, 0x75, 0x09, 0x48,
    0x83, 0xc2, 0x40, 0x41, 0xff, 0xc8, 0x75, 0xe7, 0x48, 0xff, 0xc9, 0x75, 0xd9, 0xf4,
];

// memchr-evex's shape: the byte broadcast from a general register, 64 bytes a step, and a tail
// through a masked zeroing load. 2: mov eax, 0x11; vpbroadcastb zmm17, eax; mov rdx, rsi;
// mov r8d, 63; 1: vpcmpeqb k1, zmm17, [rdx]; kmovq r9, k1; test r9, r9; jnz 3f; add rdx, 64;
// dec r8d; jnz 1b; mov r9d, 0xffff; kmovq k2, r9; vmovdqu8 zmm18 {k2} {z}, [rdx];
// vpcmpeqb k1 {k2}, zmm17, zmm18; 3: dec rcx; jnz 2b; hlt
const EVEX_MEMCHR_LOOP: &[u8] = &[
    0xb8, 0x11, 0x00, 0x00, 0x00, 0x62, 0xe2, 0x7d, 0x48, 0x7a, 0xc8, 0x48, 0x89, 0xf2, 0x41, 0xb8,
    0x3f, 0x00, 0x00, 0x00, 0x62, 0xf1, 0x75, 0x40, 0x74, 0x0a, 0xc4, 0x61, 0xfb, 0x93, 0xc9, 0x4d,
    0x85, 0xc9, 0x75, 0x20, 0x48, 0x83, 0xc2, 0x40, 0x41, 0xff, 0xc8, 0x75, 0xe7, 0x41, 0xb9, 0xff,
    0xff, 0x00, 0x00, 0xc4, 0xc1, 0xfb, 0x92, 0xd1, 0x62, 0xe1, 0x7f, 0xca, 0x6f, 0x12, 0x62, 0xb1,
    0x75, 0x42, 0x74, 0xca, 0x48, 0xff, 0xc9, 0x75, 0xb7, 0xf4,
];
/// Inner steps of the string loops: 4 KiB at 32 or 64 bytes a step.
const STRING_INNER: u64 = 128;
/// What the string loops scan: no zero byte and no 0x11.
const STRING_FILL: u8 = 0x5a;
const STRING_BYTES: usize = 4096;

// 1: movaps xmm0, xmm6; mulps xmm0, xmm1; addps xmm0, xmm2; movapd xmm3, xmm7; mulpd xmm3, xmm4;
// addpd xmm3, xmm5; dec rcx; jnz 1b; hlt   (1.1 * 0.3 + 0.7, inexact every time)
const PACKED_FP_LOOP: &[u8] = &[
    0x0f, 0x28, 0xc6, 0x0f, 0x59, 0xc1, 0x0f, 0x58, 0xc2, 0x66, 0x0f, 0x28, 0xdf, 0x66, 0x0f, 0x59,
    0xdc, 0x66, 0x0f, 0x58, 0xdd, 0x48, 0xff, 0xc9, 0x75, 0xe6, 0xf4,
];

// 1: minps xmm0, xmm1; maxps xmm0, xmm2; minpd xmm3, xmm4; maxpd xmm3, xmm5; dec rcx; jnz 1b; hlt
// The same dispatch as the arithmetic with almost no work per lane: the floor the lane math sits
// on.
const MIN_MAX_LOOP: &[u8] = &[
    0x0f, 0x5d, 0xc1, 0x0f, 0x5f, 0xc2, 0x66, 0x0f, 0x5d, 0xdc, 0x66, 0x0f, 0x5f, 0xdd, 0x48, 0xff,
    0xc9, 0x75, 0xed, 0xf4,
];

// 1: sqrtps xmm0, xmm1; movaps xmm2, xmm6; divps xmm2, xmm0; sqrtpd xmm3, xmm4;
// movapd xmm5, xmm7; divpd xmm5, xmm3; dec rcx; jnz 1b; hlt
const SQRT_DIV_LOOP: &[u8] = &[
    0x0f, 0x51, 0xc1, 0x0f, 0x28, 0xd6, 0x0f, 0x5e, 0xd0, 0x66, 0x0f, 0x51, 0xdc, 0x66, 0x0f, 0x28,
    0xef, 0x66, 0x0f, 0x5e, 0xeb, 0x48, 0xff, 0xc9, 0x75, 0xe6, 0xf4,
];

// 1: cvtdq2ps xmm0, xmm1; cvtps2dq xmm2, xmm0; cvtsi2sd xmm3, rcx; cvttsd2si rax, xmm3;
// cvtss2sd xmm4, xmm0; cvtsd2ss xmm5, xmm3; dec rcx; jnz 1b; hlt
const CONVERT_LOOP: &[u8] = &[
    0x0f, 0x5b, 0xc1, 0x66, 0x0f, 0x5b, 0xd0, 0xf2, 0x48, 0x0f, 0x2a, 0xd9, 0xf2, 0x48, 0x0f, 0x2c,
    0xc3, 0xf3, 0x0f, 0x5a, 0xe0, 0xf2, 0x0f, 0x5a, 0xeb, 0x48, 0xff, 0xc9, 0x75, 0xe2, 0xf4,
];

// 1: vfmadd231ps zmm0, zmm1, zmm2; vmulps zmm3, zmm0, zmm1; vaddps zmm3, zmm3, zmm2;
// vfmadd231sd xmm4, xmm5, xmm6; dec rcx; jnz 1b; hlt
const FMA_LOOP: &[u8] = &[
    0x62, 0xf2, 0x75, 0x48, 0xb8, 0xc2, 0x62, 0xf1, 0x7c, 0x48, 0x59, 0xd9, 0x62, 0xf1, 0x64, 0x48,
    0x58, 0xda, 0xc4, 0xe2, 0xd1, 0xb9, 0xe6, 0x48, 0xff, 0xc9, 0x75, 0xe4, 0xf4,
];

/// Every 32-bit lane of a 128-bit register set to `bits`.
fn singles(bits: u32) -> u128 {
    u128::from(bits) * 0x0000_0001_0000_0001_0000_0001_0000_0001
}

/// Both 64-bit lanes set to `bits`.
fn doubles(bits: u64) -> u128 {
    u128::from(bits) << 64 | u128::from(bits)
}

/// Normal operands for the floating-point loops, chosen so that every product and sum is inexact:
/// singles in xmm0-2 and 6, doubles in xmm3-5 and 7 (xmm1 doubles as the integers the conversions
/// start from), AVX-512 enabled with zmm0-2 filled.
fn setup_floating_point(harness: &mut Harness) {
    let cpu = &mut harness.cpu;
    cpu.cr4 |= CR4_OSFXSR | CR4_OSXSAVE;
    cpu.xcr0 = crate::cpu::xsave::XCR0_SUPPORTED;
    cpu.xmm[0] = singles(1.5f32.to_bits());
    cpu.xmm[1] = singles(0.3f32.to_bits());
    cpu.xmm[2] = singles(0.7f32.to_bits());
    cpu.xmm[3] = doubles(1.5f64.to_bits());
    cpu.xmm[4] = doubles(0.3f64.to_bits());
    cpu.xmm[5] = doubles(0.7f64.to_bits());
    cpu.xmm[6] = singles(1.1f32.to_bits());
    cpu.xmm[7] = doubles(1.1f64.to_bits());
    for register in 0..3 {
        cpu.zmm_hi[register] = [cpu.xmm[register]; 2];
    }
}

fn backend() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "wasm"
    }
}

fn setup(harness: &mut Harness, iterations: u64, data: u64) {
    harness.cpu.gpr[RCX] = iterations;
    harness.cpu.gpr[RBX] = 7;
    harness.cpu.gpr[RSI] = data;
    harness.cpu.gpr[RDI] = SCRATCH + 0x1_0000;
    harness.cpu.gpr[R10] = FAR_TARGET;
    harness.write(FAR_TARGET, FAR_FUNCTION);
    // The vector loops need the AVX and AVX-512 state enabled, and data without the bytes they
    // search for.
    harness.cpu.cr4 |= super::diff_evex::CR4_OSXSAVE;
    harness.cpu.xcr0 = crate::cpu::xsave::XCR0_SUPPORTED;
    if data == DATA {
        harness.write(DATA, &[STRING_FILL; STRING_BYTES]);
    }
    if data == ALIAS_DATA {
        let directory_entry =
            harness.cpu.cr3 + FIRST_PAGE_DIRECTORY_OFFSET + (ALIAS_DATA >> LARGE_PAGE_SHIFT) * 8;
        harness.write_u64(directory_entry, ALIAS_FRAME | PDE_PRESENT_WRITABLE_LARGE);
        harness.cpu.flush_tlb();
    }
}

fn run_sliced(harness: &mut Harness) {
    loop {
        match harness.run(EVENT_SLICE) {
            Stop::Limit => {}
            Stop::Halted => return,
            other => panic!("benchmark stopped: {other:?}\n{}", harness.cpu.dump_regs()),
        }
    }
}

fn bench(name: &str, code: &[u8], iterations: u64) {
    bench_with_data(name, code, iterations, DATA);
}

fn bench_with_data(name: &str, code: &[u8], iterations: u64, data: u64) {
    bench_with_setup(name, code, iterations, data, |_| {});
}

fn bench_with_setup(
    name: &str,
    code: &[u8],
    iterations: u64,
    data: u64,
    extra_setup: impl Fn(&mut Harness),
) {
    let mut interpreted = Harness::new(code);
    setup(&mut interpreted, CHECK_ITERATIONS, data);
    extra_setup(&mut interpreted);
    run_sliced(&mut interpreted);
    let mut compiled = Harness::new(code);
    if !compiled.enable_jit() {
        println!("bench {name} {} no-jit", backend());
        return;
    }
    setup(&mut compiled, CHECK_ITERATIONS, data);
    extra_setup(&mut compiled);
    run_sliced(&mut compiled);
    assert_eq!(
        compiled.cpu.gpr, interpreted.cpu.gpr,
        "{name}: JIT differs from the interpreter"
    );
    assert_eq!(
        (compiled.cpu.xmm, compiled.cpu.zmm_hi, compiled.cpu.mxcsr),
        (
            interpreted.cpu.xmm,
            interpreted.cpu.zmm_hi,
            interpreted.cpu.mxcsr
        ),
        "{name}: JIT vector state differs from the interpreter"
    );
    assert_eq!(
        compiled.cpu.icount, interpreted.cpu.icount,
        "{name}: retired counts differ"
    );

    let mut timed = Harness::new(code);
    timed.enable_jit();
    setup(&mut timed, iterations, data);
    extra_setup(&mut timed);
    let started = Instant::now();
    run_sliced(&mut timed);
    let elapsed = started.elapsed();
    let instructions = timed.cpu.icount;
    assert_eq!(
        timed.cpu.gpr[RCX],
        0,
        "{name}: stopped early\n{}",
        timed.cpu.dump_regs()
    );
    println!(
        "bench {name:<12} {:<8} {:>8.3} ns/iteration {:>7.3} ns/insn {instructions:>11} insns",
        backend(),
        elapsed.as_nanos() as f64 / iterations as f64,
        elapsed.as_nanos() as f64 / instructions as f64
    );
}

#[test]
#[ignore]
fn register_loop() {
    bench("register", REGISTER_LOOP, TIMED_ITERATIONS);
}

#[test]
#[ignore]
fn load_store_loop() {
    bench(
        "load-store",
        LOAD_STORE_LOOP,
        TIMED_ITERATIONS / LOAD_STORE_INNER,
    );
}

#[test]
#[ignore]
fn branchy_loop() {
    bench("branchy", BRANCHY_LOOP, TIMED_ITERATIONS);
}

#[test]
#[ignore]
fn call_ret_loop() {
    bench("call-ret", CALL_RET_LOOP, TIMED_ITERATIONS);
}

#[test]
#[ignore]
fn far_call_loop() {
    bench("far-call", FAR_CALL_LOOP, TIMED_ITERATIONS);
}

#[test]
#[ignore]
fn indirect_loop() {
    bench("indirect", INDIRECT_LOOP, TIMED_ITERATIONS);
}

#[test]
#[ignore]
fn rep_stos_loop() {
    bench("rep-stos", REP_STOS_LOOP, TIMED_ITERATIONS / 4);
}

/// The indirect loop with its jump table on a page that shares the code page's soft-TLB slot.
#[test]
#[ignore]
fn aliased_indirect_loop() {
    bench_with_data("tlb-alias", INDIRECT_LOOP, TIMED_ITERATIONS, ALIAS_DATA);
}

#[test]
#[ignore]
fn avx2_strlen_loop() {
    bench(
        "avx2-strlen",
        AVX2_STRLEN_LOOP,
        TIMED_ITERATIONS / STRING_INNER,
    );
}

#[test]
#[ignore]
fn packed_fp_loop() {
    bench_with_setup(
        "packed-fp",
        PACKED_FP_LOOP,
        TIMED_ITERATIONS / 4,
        DATA,
        setup_floating_point,
    );
}

#[test]
#[ignore]
fn avx2_memcpy_loop() {
    bench(
        "avx2-memcpy",
        AVX2_MEMCPY_LOOP,
        TIMED_ITERATIONS / STRING_INNER,
    );
}

#[test]
#[ignore]
fn min_max_loop() {
    bench_with_setup(
        "min-max",
        MIN_MAX_LOOP,
        TIMED_ITERATIONS / 4,
        DATA,
        setup_floating_point,
    );
}

#[test]
#[ignore]
fn evex_strlen_loop() {
    bench(
        "evex-strlen",
        EVEX_STRLEN_LOOP,
        TIMED_ITERATIONS / STRING_INNER,
    );
}

#[test]
#[ignore]
fn sqrt_div_loop() {
    bench_with_setup(
        "sqrt-div",
        SQRT_DIV_LOOP,
        TIMED_ITERATIONS / 8,
        DATA,
        setup_floating_point,
    );
}

#[test]
#[ignore]
fn evex_memchr_loop() {
    bench(
        "evex-memchr",
        EVEX_MEMCHR_LOOP,
        TIMED_ITERATIONS / STRING_INNER,
    );
}

#[test]
#[ignore]
fn convert_loop() {
    bench_with_setup(
        "convert",
        CONVERT_LOOP,
        TIMED_ITERATIONS / 4,
        DATA,
        setup_floating_point,
    );
}

#[test]
#[ignore]
fn fma_loop() {
    bench_with_setup(
        "fma",
        FMA_LOOP,
        TIMED_ITERATIONS / 8,
        DATA,
        setup_floating_point,
    );
}
