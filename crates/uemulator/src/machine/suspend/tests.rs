use super::*;
use crate::snapshot::{MemorySnapshot, SnapshotId};
use crate::testkit::fake_bzimage;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

/// Prints 'S', runs 40 million iterations of integer, memory, SSE and x87 work, then prints the
/// result as 16 hex digits and a newline and halts with interrupts off (a shutdown). Any state a
/// suspend loses changes the digits; the work is long enough that a suspend lands inside it.
///
/// ```text
/// mov rax, cr4; or eax, 0x600; mov cr4, rax   (OSFXSR, OSXMMEXCPT: SSE is off at entry)
/// mov dx, 0x3f8; mov al, 'S'; out dx, al
/// xor eax, eax; movabs rbx, 0x9e3779b97f4a7c15; mov r8d, 40000000; mov r9d, 0x300000
/// pxor xmm0, xmm0; fld1
/// 2: imul rax, rbx; add rax, r8; rol rax, 13; mov [r9], rax; add rax, [r9 + 8]
///    mov [r9 + 8], rax; movq xmm1, rax; paddq xmm0, xmm1; fadd st(0), st(0); fsqrt
///    dec r8d; jnz 2b
/// fistp qword ptr [r9 + 16]; add rax, [r9 + 16]; movq rdi, xmm0; xor rax, rdi
/// mov esi, 16
/// 3: rol rax, 4; mov edi, eax; and edi, 15; add edi, '0'; cmp edi, '9'; jbe 4f; add edi, 7
/// 4: mov r10, rax; mov eax, edi; out dx, al; mov rax, r10; dec esi; jnz 3b
/// mov al, 10; out dx, al; cli; hlt
/// ```
const CHECKSUM_GUEST: [u8; 147] = [
    0x0f, 0x20, 0xe0, 0x0d, 0x00, 0x06, 0x00, 0x00, 0x0f, 0x22, 0xe0, 0x66, 0xba, 0xf8, 0x03, 0xb0,
    0x53, 0xee, 0x31, 0xc0, 0x48, 0xbb, 0x15, 0x7c, 0x4a, 0x7f, 0xb9, 0x79, 0x37, 0x9e, 0x41, 0xb8,
    0x00, 0x5a, 0x62, 0x02, 0x41, 0xb9, 0x00, 0x00, 0x30, 0x00, 0x66, 0x0f, 0xef, 0xc0, 0xd9, 0xe8,
    0x48, 0x0f, 0xaf, 0xc3, 0x4c, 0x01, 0xc0, 0x48, 0xc1, 0xc0, 0x0d, 0x49, 0x89, 0x01, 0x49, 0x03,
    0x41, 0x08, 0x49, 0x89, 0x41, 0x08, 0x66, 0x48, 0x0f, 0x6e, 0xc8, 0x66, 0x0f, 0xd4, 0xc1, 0xd8,
    0xc0, 0xd9, 0xfa, 0x41, 0xff, 0xc8, 0x75, 0xd8, 0x41, 0xdf, 0x79, 0x10, 0x49, 0x03, 0x41, 0x10,
    0x66, 0x48, 0x0f, 0x7e, 0xc7, 0x48, 0x31, 0xf8, 0xbe, 0x10, 0x00, 0x00, 0x00, 0x48, 0xc1, 0xc0,
    0x04, 0x89, 0xc7, 0x83, 0xe7, 0x0f, 0x83, 0xc7, 0x30, 0x83, 0xff, 0x39, 0x76, 0x03, 0x83, 0xc7,
    0x07, 0x49, 0x89, 0xc2, 0x89, 0xf8, 0xee, 0x4c, 0x89, 0xd0, 0xff, 0xce, 0x75, 0xdf, 0xb0, 0x0a,
    0xee, 0xfa, 0xf4,
];
/// The marker `CHECKSUM_GUEST` prints before its loop, and the length of its whole output.
const STARTED: u8 = b'S';
const CHECKSUM_OUTPUT_BYTES: usize = 1 + 16 + 1;
/// mov dx, 0x3f8; mov al, 'k'; out dx, al; sti; hlt: halted for good with nothing pending, so a
/// machine resumed from it runs no instruction before a second suspend.
const PRINT_THEN_STI_HALT: [u8; 9] = [0x66, 0xba, 0xf8, 0x03, 0xb0, 0x6b, 0xee, 0xfb, 0xf4];
/// Only a hung guest gets here.
const LIMIT: Duration = Duration::from_secs(60);
const MEMORY_MIB: usize = 64;

/// A sink whose bytes the test can still read after the machine took it.
#[derive(Clone, Default)]
struct Shared {
    bytes: Arc<StdMutex<Vec<u8>>>,
    finished: Arc<AtomicBool>,
}

impl SnapshotSink for Shared {
    fn write(&mut self, bytes: &[u8]) -> Result<(), SnapshotError> {
        self.bytes.lock().unwrap().extend_from_slice(bytes);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        self.finished.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// A source that records whether the restore declared it consumed.
struct Consumed {
    inner: MemorySnapshot,
    finished: Arc<AtomicBool>,
}

impl SnapshotSource for Consumed {
    fn read(&mut self, buffer: &mut [u8]) -> Result<usize, SnapshotError> {
        self.inner.read(buffer)
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        self.finished.store(true, Ordering::SeqCst);
        Ok(())
    }
}

struct Run {
    exit: ExitReason,
    output: Vec<u8>,
}

fn config(kernel: &[u8], jit: JitMode) -> MachineConfig {
    MachineConfig::new()
        .clock(ClockMode::Virtual)
        .rng(false)
        .kernel(fake_bzimage(kernel))
        .memory_mib(MEMORY_MIB)
        .jit(jit)
        .reboot(false)
}

/// Starts `config`; once the console shows `suspend_after` (if given) suspends the machine.
fn run(config: MachineConfig, suspend_after: Option<u8>) -> Run {
    let mut machine = EmulatedMachine::new(config).expect("valid config");
    let output = Arc::new(StdMutex::new(Vec::new()));
    let sink = output.clone();
    machine
        .console()
        .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
    let handle = machine.stop_handle();
    machine.start().expect("starts");
    let deadline = Instant::now() + LIMIT;
    if let Some(marker) = suspend_after {
        while !output.lock().unwrap().contains(&marker) {
            assert!(
                Instant::now() < deadline,
                "the guest never printed its marker"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        handle.suspend();
    }
    let watchdog = {
        let handle = handle.clone();
        std::thread::spawn(move || {
            std::thread::sleep(LIMIT);
            handle.stop();
        })
    };
    let exit = machine.wait();
    drop(watchdog);
    let output = output.lock().unwrap().clone();
    Run { exit, output }
}

fn suspend_mid_checksum(config: MachineConfig) -> (Vec<u8>, Vec<u8>) {
    let sink = Shared::default();
    let run = run(config.suspend_to(sink.clone()), Some(STARTED));
    assert_eq!(run.exit, ExitReason::Suspended);
    assert_eq!(
        run.output,
        [STARTED],
        "the guest finished before the suspend; lengthen its loop"
    );
    assert!(sink.finished.load(Ordering::SeqCst));
    let bytes = sink.bytes.lock().unwrap().clone();
    (bytes, run.output)
}

fn resume(config: MachineConfig, bytes: Vec<u8>) -> (Run, bool) {
    let finished = Arc::new(AtomicBool::new(false));
    let source = Consumed {
        inner: MemorySnapshot::new(bytes),
        finished: finished.clone(),
    };
    let run = run(config.resume_from(source), None);
    (run, finished.load(Ordering::SeqCst))
}

fn reference_output(jit: JitMode) -> Vec<u8> {
    let run = run(config(&CHECKSUM_GUEST, jit), None);
    assert_eq!(run.exit, ExitReason::Shutdown);
    assert_eq!(run.output.len(), CHECKSUM_OUTPUT_BYTES);
    run.output
}

#[test]
fn a_machine_suspended_mid_computation_resumes_to_the_result_an_uninterrupted_one_gets() {
    for jit in [JitMode::Off, JitMode::Auto] {
        let _turn = (jit != JitMode::Off).then(crate::jit::serialize_jit_tests);
        let expected = reference_output(jit);
        let (snapshot, before) = suspend_mid_checksum(config(&CHECKSUM_GUEST, jit));
        let (after, consumed) = resume(config(&CHECKSUM_GUEST, jit), snapshot);
        assert_eq!(after.exit, ExitReason::Shutdown, "{jit:?}");
        assert!(consumed, "the source was not told the restore finished");
        let mut whole = before;
        whole.extend_from_slice(&after.output);
        assert_eq!(
            String::from_utf8_lossy(&whole),
            String::from_utf8_lossy(&expected),
            "{jit:?}"
        );
    }
}

/// The interpreter's snapshot resumes under the JIT and the other way round: the snapshot
/// carries no generated code.
#[test]
fn a_snapshot_resumes_with_the_jit_on_or_off_whichever_took_it() {
    let _turn = crate::jit::serialize_jit_tests();
    let expected = reference_output(JitMode::Off);
    for (taken, resumed) in [(JitMode::Off, JitMode::Auto), (JitMode::Auto, JitMode::Off)] {
        let (snapshot, mut whole) = suspend_mid_checksum(config(&CHECKSUM_GUEST, taken));
        let (after, _) = resume(config(&CHECKSUM_GUEST, resumed), snapshot);
        assert_eq!(after.exit, ExitReason::Shutdown);
        whole.extend_from_slice(&after.output);
        assert_eq!(whole, expected, "{taken:?} then {resumed:?}");
    }
}

/// The native engine runs vCPUs on threads, the WebKit one time-shares them: a snapshot from
/// either resumes in the other.
#[test]
fn a_snapshot_moves_between_threaded_and_time_shared_vcpus() {
    const CPUS: usize = 2;
    let expected = reference_output(JitMode::Off);
    let smp = |mode: SmpMode| {
        config(&CHECKSUM_GUEST, JitMode::Off)
            .cpus(CPUS)
            .smp(mode)
            .clock(if mode == SmpMode::Threads {
                ClockMode::Realtime
            } else {
                ClockMode::Virtual
            })
    };
    for (taken, resumed) in [
        (SmpMode::Threads, SmpMode::Timeshare),
        (SmpMode::Timeshare, SmpMode::Threads),
    ] {
        let (snapshot, mut whole) = suspend_mid_checksum(smp(taken));
        let (after, _) = resume(smp(resumed), snapshot);
        assert_eq!(
            after.exit,
            ExitReason::Shutdown,
            "{taken:?} then {resumed:?}"
        );
        whole.extend_from_slice(&after.output);
        assert_eq!(whole, expected, "{taken:?} then {resumed:?}");
    }
}

/// Every device a machine can have, so the byte comparison below covers each one's state.
fn every_device(kernel: &[u8]) -> MachineConfig {
    config(kernel, JitMode::Off)
        .disk(Disk::memory(vec![0; 1 << 20]))
        .net([2, 0, 0, 0, 0, 1])
        .tpm(true)
        .key_value_store(MemoryKeyValueStore::new())
        .display(DisplayMode::Framebuffer)
        .input(true)
        .rng(true)
        .random_seed(1)
}

/// What is saved is exactly what is restored: a machine resumed from a snapshot and suspended
/// again before it runs anything gives the same bytes. A field the restore misses, or one
/// `settle` changes, shows up here.
#[test]
fn resuming_and_suspending_again_without_running_gives_the_same_snapshot() {
    let first = Shared::default();
    let run = run(
        every_device(&PRINT_THEN_STI_HALT).suspend_to(first.clone()),
        Some(b'k'),
    );
    assert_eq!(run.exit, ExitReason::Suspended);
    let first = first.bytes.lock().unwrap().clone();
    let second = Shared::default();
    let config = every_device(&PRINT_THEN_STI_HALT)
        .resume_from(MemorySnapshot::new(first.clone()))
        .suspend_to(second.clone());
    let mut machine = EmulatedMachine::new(config).unwrap();
    // Before the start: the resumed vCPU is halted with nothing pending, so the run notices the
    // request before anything executes.
    machine.suspend();
    machine.start().unwrap();
    assert_eq!(machine.wait(), ExitReason::Suspended);
    let second = without_ids(second.bytes.lock().unwrap().clone());
    let first = without_ids(first);
    assert_eq!(first.len(), second.len());
    let differs = first.iter().zip(&second).position(|(a, b)| a != b);
    assert_eq!(
        differs, None,
        "the snapshots differ from byte {differs:?} on"
    );
}

#[test]
fn a_snapshot_of_another_machine_is_refused_before_anything_runs() {
    let (snapshot, _) = suspend_mid_checksum(config(&CHECKSUM_GUEST, JitMode::Off));
    let (run, consumed) = resume(
        config(&CHECKSUM_GUEST, JitMode::Off).memory_mib(MEMORY_MIB * 2),
        snapshot,
    );
    let ExitReason::ResumeFailed(message) = run.exit else {
        panic!("resumed into a bigger machine: {:?}", run.exit);
    };
    assert!(message.contains("another machine"), "{message}");
    assert!(run.output.is_empty());
    assert!(
        !consumed,
        "a failed restore told the host to drop the snapshot"
    );
}

#[test]
fn a_damaged_snapshot_is_refused_and_nothing_runs() {
    let (snapshot, _) = suspend_mid_checksum(config(&CHECKSUM_GUEST, JitMode::Off));
    for damage in [
        |bytes: &mut Vec<u8>| bytes.truncate(bytes.len() / 2),
        |bytes: &mut Vec<u8>| {
            let middle = bytes.len() / 2;
            bytes[middle] ^= 1;
        },
        |bytes: &mut Vec<u8>| bytes.push(0),
    ] {
        let mut bytes = snapshot.clone();
        damage(&mut bytes);
        let (run, consumed) = resume(config(&CHECKSUM_GUEST, JitMode::Off), bytes);
        assert!(
            matches!(run.exit, ExitReason::ResumeFailed(_)),
            "{:?}",
            run.exit
        );
        assert!(run.output.is_empty());
        assert!(!consumed);
    }
}

#[test]
fn suspend_without_a_sink_is_a_plain_stop() {
    let run = run(config(&CHECKSUM_GUEST, JitMode::Off), Some(STARTED));
    assert_eq!(run.exit, ExitReason::Stopped);
}

#[test]
fn suspend_and_resume_refuse_verify_jit() {
    let config = config(&CHECKSUM_GUEST, JitMode::Off)
        .verify_jit(true)
        .suspend_to(Shared::default());
    assert!(matches!(
        EmulatedMachine::new(config),
        Err(Error::BadConfig(_))
    ));
}

/// Where a format-2 header keeps the snapshot's id, then its parent's.
const ID_AT: usize = MAGIC.len() + 4 + 1;
const IDS_END: usize = ID_AT + 32;

/// `bytes` with the id and parent fields blanked and without the checksum, which covers them:
/// two snapshots of one state differ only there.
fn without_ids(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes[ID_AT..IDS_END].fill(0);
    bytes.truncate(bytes.len() - 4);
    bytes
}

/// Fills 32 pages at 4 MiB with 0x11 and prints '1'; spins long enough for a checkpoint; then
/// rewrites four of them with a pattern, zeroes one, stores one byte into another and adds to a
/// fifth with a locked instruction, prints '2', and halts for good with nothing pending.
///
/// ```text
/// mov dx, 0x3f8; mov edi, 0x400000; mov ecx, 16384; movabs rax, 0x1111111111111111; rep stosq
/// mov al, '1'; out dx, al; mov ecx, 30000000; 2: dec ecx; jnz 2b
/// mov edi, 0x410000; mov ecx, 2048; movabs rbx, 0x9e3779b97f4a7c15; mov rax, rbx
/// 3: mov [rdi], rax; add rax, rbx; add rdi, 8; dec ecx; jnz 3b
/// mov edi, 0x404000; mov ecx, 512; xor eax, eax; rep stosq
/// mov edi, 0x40c123; mov byte ptr [rdi], 0x5a; mov edi, 0x40d008; lock add qword ptr [rdi], 7
/// mov al, '2'; out dx, al; sti; hlt
/// ```
const DELTA_GUEST: [u8; 114] = [
    0x66, 0xba, 0xf8, 0x03, 0xbf, 0x00, 0x00, 0x40, 0x00, 0xb9, 0x00, 0x40, 0x00, 0x00, 0x48, 0xb8,
    0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0xf3, 0x48, 0xab, 0xb0, 0x31, 0xee, 0xb9, 0x80,
    0xc3, 0xc9, 0x01, 0xff, 0xc9, 0x75, 0xfc, 0xbf, 0x00, 0x00, 0x41, 0x00, 0xb9, 0x00, 0x08, 0x00,
    0x00, 0x48, 0xbb, 0x15, 0x7c, 0x4a, 0x7f, 0xb9, 0x79, 0x37, 0x9e, 0x48, 0x89, 0xd8, 0x48, 0x89,
    0x07, 0x48, 0x01, 0xd8, 0x48, 0x83, 0xc7, 0x08, 0xff, 0xc9, 0x75, 0xf2, 0xbf, 0x00, 0x40, 0x40,
    0x00, 0xb9, 0x00, 0x02, 0x00, 0x00, 0x31, 0xc0, 0xf3, 0x48, 0xab, 0xbf, 0x23, 0xc1, 0x40, 0x00,
    0xc6, 0x07, 0x5a, 0xbf, 0x08, 0xd0, 0x40, 0x00, 0xf0, 0x48, 0x83, 0x07, 0x07, 0xb0, 0x32, 0xee,
    0xfb, 0xf4,
];

/// A machine that is running, and what its console printed so far.
struct Running {
    machine: EmulatedMachine,
    output: Arc<StdMutex<Vec<u8>>>,
    handle: StopHandle,
}

fn start(config: MachineConfig) -> Running {
    let mut machine = EmulatedMachine::new(config).expect("valid config");
    let output = Arc::new(StdMutex::new(Vec::new()));
    let sink = output.clone();
    machine
        .console()
        .on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
    let handle = machine.stop_handle();
    machine.start().expect("starts");
    Running {
        machine,
        output,
        handle,
    }
}

impl Running {
    fn wait_for(&self, marker: u8) {
        let deadline = Instant::now() + LIMIT;
        while !self.output.lock().unwrap().contains(&marker) {
            assert!(
                Instant::now() < deadline,
                "the guest never printed {marker}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn output(&self) -> Vec<u8> {
        self.output.lock().unwrap().clone()
    }

    /// Takes a checkpoint and waits for it.
    fn checkpoint(&self) -> Result<(SnapshotInfo, Vec<u8>), SnapshotError> {
        let sink = Shared::default();
        let (answer, answered) = std::sync::mpsc::channel();
        self.handle
            .checkpoint(sink.clone(), move |result| answer.send(result).unwrap());
        let info = answered
            .recv_timeout(LIMIT)
            .expect("the checkpoint was answered")?;
        assert!(sink.finished.load(Ordering::SeqCst));
        let bytes = sink.bytes.lock().unwrap().clone();
        Ok((info, bytes))
    }
}

fn chain(snapshots: &[&Vec<u8>]) -> Vec<Box<dyn SnapshotSource>> {
    snapshots
        .iter()
        .map(|bytes| Box::new(MemorySnapshot::new((*bytes).clone())) as Box<dyn SnapshotSource>)
        .collect()
}

/// A checkpoint, a delta after the guest rewrote some pages, and a suspend of the same moment:
/// the chain of the two checkpoints, resumed and suspended again before anything runs, gives the
/// suspend's bytes. Every write the delta missed, or a page it restored wrongly, shows up. The
/// dirty-log check runs too, so a store path that skips the log fails the second checkpoint.
#[test]
fn a_delta_chain_restores_exactly_the_machine_it_was_taken_from() {
    for (jit, cpus) in [(JitMode::Off, 1), (JitMode::Auto, 1), (JitMode::Off, 2)] {
        let _turn = (jit != JitMode::Off).then(crate::jit::serialize_jit_tests);
        let machine = || {
            config(&DELTA_GUEST, jit)
                .cpus(cpus)
                .smp(SmpMode::Timeshare)
                .check_dirty_tracking(true)
        };
        let suspended = Shared::default();
        let running = start(machine().suspend_to(suspended.clone()));
        running.wait_for(b'1');
        let (base, base_bytes) = running.checkpoint().expect("the first checkpoint");
        assert_eq!(base.parent, None, "the first checkpoint is full");
        assert!(
            !running.output().contains(&b'2'),
            "the guest finished before the checkpoint; lengthen its loop"
        );
        running.wait_for(b'2');
        let (delta, delta_bytes) = running.checkpoint().expect("the delta");
        assert_eq!(delta.parent, Some(base.id), "{jit:?} {cpus}");
        running.handle.suspend();
        let mut running = running;
        assert_eq!(running.machine.wait(), ExitReason::Suspended);
        let suspended = suspended.bytes.lock().unwrap().clone();

        // Resumed and suspended at once: through the same `settle` both ways, which re-anchors
        // the inactive vCPUs' counters, so the two are compared after it.
        let resume_then_suspend = |sources: Vec<Box<dyn SnapshotSource>>| {
            let again = Shared::default();
            let mut resumed = EmulatedMachine::new(
                machine()
                    .resume_from_chain(sources)
                    .suspend_to(again.clone()),
            )
            .unwrap();
            resumed.suspend();
            resumed.start().unwrap();
            assert_eq!(resumed.wait(), ExitReason::Suspended);
            again.bytes.lock().unwrap().clone()
        };
        let again = resume_then_suspend(chain(&[&base_bytes, &delta_bytes]));
        let expected = resume_then_suspend(chain(&[&suspended]));
        if cpus == 1 {
            assert_eq!(without_ids(expected.clone()), without_ids(suspended));
        }
        assert_eq!(again.len(), expected.len(), "{jit:?} {cpus}");
        let (again, expected) = (without_ids(again), without_ids(expected));
        let differs = again.iter().zip(&expected).position(|(a, b)| a != b);
        assert_eq!(
            differs, None,
            "{jit:?} {cpus}: differs from byte {differs:?} on"
        );
    }
}

/// Checkpoints taken during the computation change nothing the guest computes, and the chain of
/// them resumes to the same result, on threads and time-shared, with the interpreter and the JIT.
#[test]
fn checkpoints_leave_a_computation_alone_and_its_chain_resumes_it() {
    let expected = reference_output(JitMode::Off);
    for (jit, cpus, smp) in [
        (JitMode::Off, 1, SmpMode::Auto),
        (JitMode::Auto, 1, SmpMode::Auto),
        (JitMode::Off, 2, SmpMode::Timeshare),
        (JitMode::Off, 2, SmpMode::Threads),
        (JitMode::Auto, 2, SmpMode::Threads),
    ] {
        let _turn = (jit != JitMode::Off).then(crate::jit::serialize_jit_tests);
        let machine = || {
            config(&CHECKSUM_GUEST, jit)
                .cpus(cpus)
                .smp(smp)
                .clock(if smp == SmpMode::Threads {
                    ClockMode::Realtime
                } else {
                    ClockMode::Virtual
                })
                .check_dirty_tracking(true)
        };
        let mut running = start(machine());
        running.wait_for(STARTED);
        let mut taken = Vec::new();
        for _ in 0..3 {
            taken.push(running.checkpoint().expect("a checkpoint"));
        }
        assert_eq!(
            running.output(),
            [STARTED],
            "the guest finished before the checkpoints; lengthen its loop"
        );
        for pair in taken.windows(2) {
            assert_eq!(pair[1].0.parent, Some(pair[0].0.id));
        }
        assert_eq!(running.machine.wait(), ExitReason::Shutdown);
        assert_eq!(running.output(), expected, "{jit:?} {cpus} {smp:?}");

        let snapshots: Vec<&Vec<u8>> = taken.iter().map(|(_, bytes)| bytes).collect();
        let resumed = run(machine().resume_from_chain(chain(&snapshots)), None);
        assert_eq!(resumed.exit, ExitReason::Shutdown, "{jit:?} {cpus} {smp:?}");
        let mut whole = vec![STARTED];
        whole.extend_from_slice(&resumed.output);
        assert_eq!(whole, expected, "{jit:?} {cpus} {smp:?}");
    }
}

/// A chain whose pieces do not fit together is refused before anything runs.
#[test]
fn a_chain_that_does_not_fit_together_is_refused() {
    let running = start(config(&DELTA_GUEST, JitMode::Off));
    running.wait_for(b'1');
    let (_, first) = running.checkpoint().unwrap();
    let (_, second) = running.checkpoint().unwrap();
    let (_, third) = running.checkpoint().unwrap();
    running.handle.stop();
    for (pieces, why) in [
        (vec![&second], "must be restored before it"),
        (vec![&first, &first], "not a delta"),
        (vec![&first, &third], "is a delta of"),
    ] {
        let resumed = run(
            config(&DELTA_GUEST, JitMode::Off).resume_from_chain(chain(&pieces)),
            None,
        );
        let ExitReason::ResumeFailed(message) = &resumed.exit else {
            panic!("resumed: {:?}", resumed.exit);
        };
        assert!(message.contains(why), "{message}");
        assert!(resumed.output.is_empty());
    }
}

/// A sink that refuses every write.
struct Refusing;

impl SnapshotSink for Refusing {
    fn write(&mut self, _: &[u8]) -> Result<(), SnapshotError> {
        Err(SnapshotError("disk full".into()))
    }

    fn finish(&mut self) -> Result<(), SnapshotError> {
        Err(SnapshotError("disk full".into()))
    }
}

/// A checkpoint that fails changes nothing: the machine runs on, and the next delta is against
/// the last checkpoint that was written.
#[test]
fn a_failed_checkpoint_leaves_the_machine_running_and_the_chain_intact() {
    let mut running = start(config(&CHECKSUM_GUEST, JitMode::Off));
    running.wait_for(STARTED);
    let (first, _) = running.checkpoint().unwrap();
    let (answer, answered) = std::sync::mpsc::channel();
    running
        .handle
        .checkpoint(Refusing, move |result| answer.send(result).unwrap());
    let refused = answered.recv_timeout(LIMIT).unwrap();
    assert!(refused.unwrap_err().0.contains("disk full"));
    let (second, _) = running.checkpoint().unwrap();
    assert_eq!(second.parent, Some(first.id));
    assert_eq!(running.machine.wait(), ExitReason::Shutdown);
    assert_eq!(running.output(), reference_output(JitMode::Off));
}

/// A checkpoint asked for while the machine stops is answered, with an error, and the stop is a
/// stop.
#[test]
fn a_checkpoint_that_meets_a_stop_is_refused() {
    let mut machine = EmulatedMachine::new(config(&CHECKSUM_GUEST, JitMode::Off)).unwrap();
    let (answer, answered) = std::sync::mpsc::channel();
    machine
        .stop_handle()
        .checkpoint(Shared::default(), move |result| {
            answer.send(result).unwrap()
        });
    machine.stop();
    machine.start().unwrap();
    assert_eq!(machine.wait(), ExitReason::Stopped);
    assert!(answered.recv_timeout(LIMIT).unwrap().is_err());
}

/// What the builds before delta snapshots wrote (format 1: no id, every non-zero page raw), from
/// the stopped vCPUs of a harness.
fn save_format_1(cpus: &mut [Cpu], sink: &mut dyn SnapshotSink) {
    let guest_ns = cpus[0].now_ns();
    cpus[0].pmu_park();
    let shape = shape(cpus);
    let mut w = Writer::new(sink);
    w.bytes(MAGIC).unwrap();
    w.u32(FORMAT_VERSION_RAW_PAGES).unwrap();
    shape.save(&mut w).unwrap();
    w.u64(guest_ns).unwrap();
    w.len(cpus.len()).unwrap();
    for cpu in cpus.iter() {
        cpu.save(&mut w).unwrap();
    }
    w.u64(cpus[0].rng_state).unwrap();
    cpus[0].dev.lock().save(&mut w).unwrap();
    let ram = &cpus[0].ram;
    w.len(ram.len()).unwrap();
    let pages = ram.len() / PAGE_BYTES;
    for group in (0..pages).step_by(PAGES_PER_GROUP) {
        let end = (group + PAGES_PER_GROUP).min(pages);
        let present = (group..end)
            .filter(|&page| !is_filled_with(page_bytes(ram, page), 0))
            .fold(0u64, |bits, page| bits | 1 << (page - group));
        w.u64(present).unwrap();
        for page in group..end {
            if present & 1 << (page - group) != 0 {
                w.bytes(page_bytes(ram, page)).unwrap();
            }
        }
    }
    w.finish().unwrap();
}

/// mov rax, 0x1234; hlt
const SET_RAX_THEN_HALT: [u8; 8] = [0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, 0xf4];

fn harness_ram(cpu: &Cpu) -> Vec<u8> {
    cpu.ram[0..cpu.ram.len()].to_vec()
}

/// Snapshots written before format 2 still resume.
#[test]
fn a_format_1_snapshot_still_loads() {
    let mut taken = crate::testkit::Harness::new(&SET_RAX_THEN_HALT);
    taken.run_to_halt();
    taken.write(0x40_0000, b"written before the upgrade");
    let mut bytes = MemorySnapshot::default();
    save_format_1(core::slice::from_mut(&mut taken.cpu), &mut bytes);
    let mut source = MemorySnapshot::new(bytes.bytes.clone());
    assert_eq!(
        read_snapshot_info(&mut source).unwrap(),
        SnapshotInfo {
            version: 1,
            id: SnapshotId::default(),
            parent: None
        }
    );
    let mut into = crate::testkit::Harness::new(&[crate::testkit::HLT]);
    let mut sources: Vec<Box<dyn SnapshotSource>> =
        vec![Box::new(MemorySnapshot::new(bytes.bytes))];
    let (info, parent_capable) =
        restore(core::slice::from_mut(&mut into.cpu), false, &mut sources).unwrap();
    assert_eq!(info.version, 1);
    assert!(
        !parent_capable,
        "a format-1 snapshot has no id to be a parent by"
    );
    assert_eq!(into.cpu.gpr[crate::testkit::RAX], 0x1234);
    assert_eq!(harness_ram(&into.cpu), harness_ram(&taken.cpu));
}

/// xorshift64*: deterministic noise for the page contents below.
fn noise(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

/// Rounds of device-style writes of every kind of page (zero, one byte repeated, compressible,
/// random, a few bytes), each followed by a delta; the chain restores the same RAM every time.
#[test]
fn random_page_writes_survive_a_long_delta_chain() {
    let mut machine = crate::testkit::Harness::new(&[crate::testkit::HLT]);
    let mut state = 0x5eed_u64;
    let mut snapshots: Vec<Vec<u8>> = Vec::new();
    let pages_in_ram = machine.cpu.ram.len() / PAGE_BYTES;
    for round in 0..8 {
        if round > 0 {
            let pages = machine.cpu.page_has_code.clone();
            let mut mem = crate::dev::guestmem::GuestMem::for_pages(&mut machine.cpu.ram, &pages);
            for _ in 0..40 {
                let page = (noise(&mut state) as usize % pages_in_ram) as u64;
                let mut bytes = vec![0u8; PAGE_BYTES];
                match noise(&mut state) % 5 {
                    0 => {}
                    1 => bytes.fill(noise(&mut state) as u8),
                    2 => {
                        for (index, byte) in bytes.iter_mut().enumerate() {
                            *byte = (index / 13) as u8;
                        }
                    }
                    3 => bytes
                        .iter_mut()
                        .for_each(|byte| *byte = noise(&mut state) as u8),
                    _ => {
                        let at = noise(&mut state) as usize % (PAGE_BYTES - 4);
                        let mut current = vec![0; 4];
                        mem.read(page * PAGE_BYTES as u64 + at as u64, &mut current);
                        current[0] ^= 0xff;
                        assert!(mem.write(page * PAGE_BYTES as u64 + at as u64, &current));
                        continue;
                    }
                }
                assert!(mem.write(page * PAGE_BYTES as u64, &bytes));
            }
        }
        let log = machine.cpu.page_has_code.dirty_log();
        let marked: u32 = (0..pages_in_ram.div_ceil(PAGES_PER_GROUP))
            .map(|word| log.word(word).count_ones())
            .sum();
        // Only what the forty writes touched (the in-place edits may land on a written page).
        assert!(marked <= 40, "round {round}: {marked} pages marked");
        let mut sink = MemorySnapshot::default();
        let cpus = core::slice::from_mut(&mut machine.cpu);
        let info = save(cpus, 0, false, &mut sink, SaveKind::Checkpoint).unwrap();
        unpark_after_save(cpus, 0, false);
        assert_eq!(info.parent.is_some(), round > 0);
        arm_dirty_log(cpus, info.id);
        snapshots.push(sink.bytes);

        let mut into = crate::testkit::Harness::new(&[crate::testkit::HLT]);
        let mut sources = chain(&snapshots.iter().collect::<Vec<_>>());
        restore(core::slice::from_mut(&mut into.cpu), false, &mut sources).unwrap();
        assert!(
            harness_ram(&into.cpu) == harness_ram(&machine.cpu),
            "round {round}: the restored RAM differs"
        );
    }
}
