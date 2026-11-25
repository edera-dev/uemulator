//! The stub on whole machines: a fake kernel, a `GdbStub`, and a test thread as the client,
//! through the interpreter and the JIT, one vCPU, time-shared vCPUs and vCPUs on threads.

use super::packet::{Decoder, Incoming, frame, from_hex};
use super::*;
use crate::machine::{ClockMode, EmulatedMachine, JitMode, MachineConfig, SmpMode};
use crate::testkit::fake_bzimage;
use std::collections::VecDeque;
use std::sync::{Condvar as StdCondvar, Mutex as StdMutex};
use std::time::{Duration, Instant};

/// How long a test waits for a stop reply before it fails. Generous: a busy host runs the
/// tests in parallel.
const REPLY_TIMEOUT: Duration = Duration::from_secs(20);
const SCRATCH: u64 = crate::testkit::SCRATCH;

type Received = Arc<(StdMutex<Vec<u8>>, StdCondvar)>;

/// The test's side of the connection.
struct Client {
    stub: GdbStub,
    received: Received,
    decoder: Decoder,
    packets: VecDeque<Vec<u8>>,
}

impl Client {
    fn connect(stub: GdbStub) -> Self {
        let received: Received = Arc::new((StdMutex::new(Vec::new()), StdCondvar::new()));
        let sink = received.clone();
        stub.on_output(move |bytes| {
            sink.0.lock().unwrap().extend_from_slice(bytes);
            sink.1.notify_all();
        });
        stub.connected();
        let mut client = Client {
            stub,
            received,
            decoder: Decoder::default(),
            packets: VecDeque::new(),
        };
        assert_eq!(client.ask("QStartNoAckMode"), "OK");
        client
    }

    fn send(&self, payload: &str) {
        let mut out = Vec::new();
        frame(payload.as_bytes(), &mut out);
        self.stub.from_client(&out);
    }

    fn recv(&mut self) -> String {
        let deadline = Instant::now() + REPLY_TIMEOUT;
        loop {
            if let Some(packet) = self.packets.pop_front() {
                return String::from_utf8(packet).unwrap();
            }
            let bytes = {
                let (lock, ready) = &*self.received;
                let mut buffer = lock.lock().unwrap();
                while buffer.is_empty() {
                    let now = Instant::now();
                    assert!(now < deadline, "no reply from the stub");
                    buffer = ready.wait_timeout(buffer, deadline - now).unwrap().0;
                }
                core::mem::take(&mut *buffer)
            };
            let mut items = Vec::new();
            self.decoder.feed(&bytes, &mut items);
            for item in items {
                if let Incoming::Packet(data) = item {
                    self.packets.push_back(data);
                }
            }
        }
    }

    fn ask(&mut self, payload: &str) -> String {
        self.send(payload);
        self.recv()
    }

    fn interrupt(&self) {
        self.stub.from_client(&[packet::INTERRUPT]);
    }

    fn register(&mut self, number: usize) -> u64 {
        let reply = self.ask(&format!("p{number:x}"));
        let bytes = from_hex(reply.as_bytes()).unwrap_or_else(|| panic!("register: {reply}"));
        let mut wide = [0u8; 8];
        wide[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
        u64::from_le_bytes(wide)
    }

    fn rip(&mut self) -> u64 {
        self.register(16)
    }
}

/// The ways a machine can run its vCPUs, each with and without the JIT where the host has one.
fn every_engine() -> Vec<(JitMode, SmpMode, ClockMode)> {
    let mut modes = Vec::new();
    let jits = if crate::jit::probe() {
        vec![JitMode::Off, JitMode::On]
    } else {
        vec![JitMode::Off]
    };
    for jit in jits {
        modes.push((jit, SmpMode::Timeshare, ClockMode::Virtual));
        modes.push((jit, SmpMode::Threads, ClockMode::Realtime));
    }
    modes
}

fn machine(
    code: &[u8],
    cpus: usize,
    (jit, smp, clock): (JitMode, SmpMode, ClockMode),
) -> (EmulatedMachine, Client) {
    let config = MachineConfig::new()
        .kernel(fake_bzimage(code))
        .memory_mib(64)
        .cpus(cpus)
        .jit(jit)
        .smp(smp)
        .clock(clock)
        .rng(false)
        .reboot(false)
        .gdb_stub(true);
    let mut machine = EmulatedMachine::new(config).expect("a valid config");
    machine.start().expect("the machine starts");
    let client = Client::connect(machine.gdb_stub().expect("a stub"));
    (machine, client)
}

fn finish(mut machine: EmulatedMachine) -> crate::machine::ExitReason {
    machine.stop();
    machine.wait()
}

// mov rax, 1; mov rbx, 2; mov qword [SCRATCH], rax; add rax, rbx; mov qword [SCRATCH + 8], rax;
// jmp $
const PROGRAM: [u8; 38] = [
    0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
    0x48, 0xc7, 0xc3, 0x02, 0x00, 0x00, 0x00, // mov rbx, 2
    0x48, 0x89, 0x04, 0x25, 0x00, 0x00, 0x40, 0x00, // mov [0x400000], rax
    0x48, 0x01, 0xd8, // add rax, rbx
    0x48, 0x89, 0x04, 0x25, 0x08, 0x00, 0x40, 0x00, // mov [0x400008], rax
    0xeb, 0xfe, // jmp $
    0x90, 0x90, 0x90,
];
const ADD: u64 = 22;
const STORE_SECOND: u64 = 25;
const SPIN: u64 = 33;

#[test]
fn the_machine_halts_at_start_breaks_steps_and_reads_its_registers() {
    for engine in every_engine() {
        let (machine, mut client) = machine(&PROGRAM, 1, engine);
        assert_eq!(client.ask("?"), "T05thread:1;", "{engine:?}");
        let entry = client.rip();
        assert_eq!(client.ask(&format!("Z0,{:x},1", entry + ADD)), "OK");
        assert_eq!(client.ask("vCont;c"), "T05thread:1;", "{engine:?}");
        assert_eq!(client.rip(), entry + ADD, "{engine:?}");
        assert_eq!(client.register(0), 1);
        assert_eq!(client.register(1), 2);
        // The first store has happened, the second not.
        assert_eq!(
            client.ask(&format!("m{SCRATCH:x},10")),
            "01000000000000000000000000000000"
        );
        // A breakpoint is invisible to the guest's memory.
        assert_eq!(client.ask(&format!("m{:x},3", entry + ADD)), "4801d8");
        assert_eq!(client.ask(&format!("z0,{:x},1", entry + ADD)), "OK");
        assert_eq!(client.ask("vCont;s:1"), "T05thread:1;");
        assert_eq!(client.rip(), entry + STORE_SECOND);
        assert_eq!(client.register(0), 3);
        assert_eq!(client.ask("s"), "T05thread:1;");
        assert_eq!(client.rip(), entry + SPIN);
        assert_eq!(
            client.ask(&format!("m{:x},8", SCRATCH + 8)),
            "0300000000000000"
        );
        drop(client);
        assert_eq!(finish(machine), crate::machine::ExitReason::Stopped);
    }
}

#[test]
fn a_write_watchpoint_stops_after_the_store() {
    for engine in every_engine() {
        let (machine, mut client) = machine(&PROGRAM, 1, engine);
        assert_eq!(client.ask("?"), "T05thread:1;");
        let entry = client.rip();
        assert_eq!(client.ask(&format!("Z2,{:x},8", SCRATCH + 8)), "OK");
        assert_eq!(
            client.ask("c"),
            format!("T05thread:1;watch:{:x};", SCRATCH + 8),
            "{engine:?}"
        );
        assert_eq!(client.rip(), entry + SPIN, "the stop comes after the store");
        assert_eq!(client.ask(&format!("m{:x},1", SCRATCH + 8)), "03");
        drop(client);
        finish(machine);
    }
}

#[test]
fn read_and_access_watchpoints_report_their_kind() {
    // mov rax, [SCRATCH]; mov [SCRATCH + 16], rax; jmp $
    const READ_THEN_WRITE: [u8; 18] = [
        0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x40, 0x00, 0x48, 0x89, 0x04, 0x25, 0x10, 0x00, 0x40,
        0x00, 0xeb, 0xfe,
    ];
    let (machine, mut client) = machine(&READ_THEN_WRITE, 1, every_engine()[0]);
    client.ask("?");
    let entry = client.rip();
    assert_eq!(client.ask(&format!("Z3,{SCRATCH:x},8")), "OK");
    assert_eq!(client.ask("c"), format!("T05thread:1;rwatch:{SCRATCH:x};"));
    assert_eq!(client.rip(), entry + 8);
    assert_eq!(client.ask(&format!("z3,{SCRATCH:x},8")), "OK");
    assert_eq!(client.ask(&format!("Z4,{:x},1", SCRATCH + 16)), "OK");
    assert_eq!(
        client.ask("c"),
        format!("T05thread:1;awatch:{:x};", SCRATCH + 16)
    );
    assert_eq!(client.rip(), entry + 16);
    drop(client);
    finish(machine);
}

#[test]
fn ctrl_c_stops_a_running_machine_and_kill_ends_it() {
    for engine in every_engine() {
        // jmp $
        let (mut machine, mut client) = machine(&[0xeb, 0xfe], 1, engine);
        client.ask("?");
        let entry = client.rip();
        client.send("c");
        std::thread::sleep(Duration::from_millis(20));
        client.interrupt();
        assert_eq!(client.recv(), "T02thread:1;", "{engine:?}");
        assert_eq!(client.rip(), entry);
        client.send("k");
        assert_eq!(machine.wait(), crate::machine::ExitReason::Stopped);
    }
}

#[test]
fn detaching_clears_the_breakpoints_and_the_machine_runs_to_its_end() {
    // mov rax, 1; cli; hlt: a dead machine powers off.
    const POWER_OFF: [u8; 9] = [0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, 0xfa, 0xf4];
    for engine in every_engine() {
        let (mut machine, mut client) = machine(&POWER_OFF, 1, engine);
        client.ask("?");
        let entry = client.rip();
        assert_eq!(client.ask(&format!("Z0,{:x},1", entry + 7)), "OK");
        assert_eq!(client.ask("D"), "OK");
        assert_eq!(
            machine.wait(),
            crate::machine::ExitReason::Shutdown,
            "{engine:?}"
        );
    }
}

#[test]
fn a_client_that_resumes_hears_the_machine_end() {
    const POWER_OFF: [u8; 2] = [0xfa, 0xf4];
    let (mut machine, mut client) = machine(&POWER_OFF, 1, every_engine()[0]);
    client.ask("?");
    assert_eq!(client.ask("c"), "W00");
    assert_eq!(machine.wait(), crate::machine::ExitReason::Shutdown);
}

#[test]
fn a_disconnect_lets_the_machine_run_on() {
    const POWER_OFF: [u8; 4] = [0x90, 0x90, 0xfa, 0xf4];
    let (mut machine, mut client) = machine(&POWER_OFF, 1, every_engine()[0]);
    client.ask("?");
    let entry = client.rip();
    assert_eq!(client.ask(&format!("Z0,{:x},1", entry + 1)), "OK");
    client.stub.disconnected();
    assert_eq!(machine.wait(), crate::machine::ExitReason::Shutdown);
}

#[test]
fn a_machine_without_halt_at_start_stops_when_a_client_connects() {
    let config = MachineConfig::new()
        .kernel(fake_bzimage(&[0xeb, 0xfe]))
        .memory_mib(64)
        .clock(ClockMode::Virtual)
        .rng(false)
        .gdb_stub(false);
    let mut machine = EmulatedMachine::new(config).unwrap();
    let stub = machine.gdb_stub().unwrap();
    machine.start().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let mut client = Client::connect(stub);
    assert_eq!(client.ask("?"), "T05thread:1;");
    drop(client);
    finish(machine);
}

#[test]
fn a_debugger_cannot_be_combined_with_verification() {
    let config = MachineConfig::new()
        .kernel(fake_bzimage(&[0xf4]))
        .verify_jit(true)
        .gdb_stub(false);
    assert!(matches!(
        EmulatedMachine::new(config),
        Err(crate::machine::Error::BadConfig(_))
    ));
}

/// A kernel whose first vCPU starts the second in real mode at 0x8000, where it spins on
/// `jmp $`, and then spins itself.
fn two_spinning_vcpus() -> Vec<u8> {
    // mov ecx, 0x1b; mov eax, 0xfee00d00; xor edx, edx; wrmsr (x2APIC on); mov ecx, 0x830 (ICR)
    let mut code = vec![
        0xb9, 0x1b, 0x00, 0x00, 0x00, 0xb8, 0x00, 0x0d, 0xe0, 0xfe, 0x31, 0xd2, 0x0f, 0x30, 0xb9,
        0x30, 0x08, 0x00, 0x00,
    ];
    // mov dword [0x8000], 0x9090feeb: jmp $ (and nops) for the second vCPU
    code.extend([
        0xc7, 0x04, 0x25, 0x00, 0x80, 0x00, 0x00, 0xeb, 0xfe, 0x90, 0x90,
    ]);
    // INIT, then start-up at vector 8 (0x8000), to APIC id 1: mov edx, 1; mov eax, low; wrmsr
    for low in [0x4500u32, 0x4608] {
        code.extend([0xba, 0x01, 0x00, 0x00, 0x00, 0xb8]);
        code.extend(low.to_le_bytes());
        code.extend([0x0f, 0x30]);
    }
    code.extend([0xeb, 0xfe]);
    code
}

#[test]
fn every_vcpu_is_a_thread_and_a_stop_stops_them_all() {
    const SECOND_VCPU_SPIN: u64 = 0x8000;
    for engine in every_engine() {
        let (machine, mut client) = machine(&two_spinning_vcpus(), 2, engine);
        assert_eq!(client.ask("?"), "T05thread:1;");
        assert_eq!(client.ask("qfThreadInfo"), "m1,2");
        assert_eq!(client.ask("qsThreadInfo"), "l");
        // A breakpoint the second vCPU reaches: the stop names it.
        assert_eq!(client.ask(&format!("Z0,{SECOND_VCPU_SPIN:x},1")), "OK");
        assert_eq!(client.ask("c"), "T05thread:2;", "{engine:?}");
        assert_eq!(client.ask("Hg2"), "OK");
        // Real mode: rip is the offset into CS (0x800:0).
        assert_eq!(client.rip(), 0, "{engine:?}");
        assert_eq!(client.register(18) & 0xffff, 0x800, "cs");
        assert_eq!(client.ask(&format!("m{SECOND_VCPU_SPIN:x},2")), "ebfe");
        // The first vCPU is stopped too, in its own spin.
        assert_eq!(client.ask("Hg1"), "OK");
        let first = client.rip();
        assert_eq!(client.ask(&format!("m{first:x},2")), "ebfe", "{engine:?}");
        // Stepping the second over its breakpoint holds the first, as gdb does it.
        assert_eq!(client.ask(&format!("z0,{SECOND_VCPU_SPIN:x},1")), "OK");
        assert_eq!(client.ask("vCont;s:2"), "T05thread:2;", "{engine:?}");
        assert_eq!(client.ask("Hg1"), "OK");
        assert_eq!(client.rip(), first);
        // Ctrl-C with both running.
        client.send("vCont;c");
        std::thread::sleep(Duration::from_millis(20));
        client.interrupt();
        let reply = client.recv();
        assert!(
            reply == "T02thread:1;" || reply == "T02thread:2;",
            "{engine:?}: {reply}"
        );
        drop(client);
        finish(machine);
    }
}

#[test]
fn a_vcpu_held_while_another_steps_still_answers_ctrl_c() {
    // The second vCPU is never started: a step of it cannot end, and the first is held.
    for engine in every_engine() {
        let (machine, mut client) = machine(&[0xeb, 0xfe], 2, engine);
        assert_eq!(client.ask("?"), "T05thread:1;");
        client.send("vCont;s:2");
        std::thread::sleep(Duration::from_millis(20));
        client.interrupt();
        assert_eq!(client.recv(), "T02thread:1;", "{engine:?}");
        drop(client);
        finish(machine);
    }
}

#[test]
fn ctrl_c_reaches_a_machine_whose_vcpus_are_all_halted() {
    // sti; hlt; jmp back: an idle guest.
    const IDLE: [u8; 4] = [0xfb, 0xf4, 0xeb, 0xfc];
    for engine in every_engine() {
        let (machine, mut client) = machine(&IDLE, 2, engine);
        assert_eq!(client.ask("?"), "T05thread:1;");
        client.send("c");
        std::thread::sleep(Duration::from_millis(20));
        client.interrupt();
        assert_eq!(client.recv(), "T02thread:1;", "{engine:?}");
        drop(client);
        finish(machine);
    }
}

#[test]
fn memory_is_read_through_the_guests_page_tables() {
    // The fake kernel's tables map low memory; read the code through a mapping of its own made
    // by the client: the PML4 slot for the higher half points at the one for low memory.
    for engine in every_engine() {
        let (machine, mut client) = machine(&PROGRAM, 1, engine);
        client.ask("?");
        let entry = client.rip();
        let cr3 = client.register(62) & !0xfff;
        let low = client.ask(&format!("m{cr3:x},8"));
        const HIGHER_HALF_SLOT: u64 = 256;
        assert_eq!(
            client.ask(&format!("M{:x},8:{low}", cr3 + 8 * HIGHER_HALF_SLOT)),
            "OK"
        );
        const HIGHER_HALF: u64 = 0xffff_8000_0000_0000;
        assert_eq!(
            client.ask(&format!("m{:x},7", HIGHER_HALF + entry)),
            "48c7c001000000",
            "{engine:?}"
        );
        drop(client);
        finish(machine);
    }
}
