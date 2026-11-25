//! The GDB remote stub (`docs/plans/30-gdb-stub.md`). Sans-I/O: the embedder moves bytes
//! between its connection and a `GdbStub` and the library never touches a socket.
//!
//! The protocol engine runs on the vCPU thread that stopped, while every other vCPU is stopped
//! too (`serve`): it needs `&mut` access to every `Cpu`, which only a stopped machine gives. The
//! client's bytes wait in the session's inbox until then. While the machine runs, the only byte
//! a client in all-stop mode sends is the 0x03 interrupt, which `GdbStub::from_client` turns into
//! a pause request the vCPUs notice at their next block boundary (`Cpu::debugger_check`).

#[cfg(freestanding)]
use crate::prelude::*;

mod engine;
#[cfg(test)]
mod machine_tests;
pub(crate) mod packet;
#[cfg(test)]
mod tests;

use crate::cpu::gdb::{Vcpus, apply_resume};
use crate::machine::ExitReason;
use crate::sync::{Arc, Condvar, Mutex};
use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use core::time::Duration;
use engine::{Engine, Outcome};

/// How often a stopped vCPU looks at the machine's stop flag while it waits for the client.
const SERVE_POLL: Duration = Duration::from_millis(10);

/// No pause asked for.
const PAUSE_NONE: u8 = 0;
/// A pause the client did not ask for with Ctrl-C: the halt at start, a new connection, a
/// detach while running. Reported as SIGTRAP, like a stop the debugger caused.
const PAUSE_QUIET: u8 = 1;
/// The client's Ctrl-C, reported as SIGINT.
const PAUSE_INTERRUPT: u8 = 2;

const SIGINT: u8 = 2;
const SIGTRAP: u8 = 5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WatchKind {
    Write,
    Read,
    Access,
}

/// A data watchpoint on `length` bytes at linear address `address`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Watchpoint {
    pub address: u64,
    pub length: u64,
    pub kind: WatchKind,
}

impl Watchpoint {
    pub fn matches(&self, linear: u64, size: u64, write: bool) -> bool {
        let kind = match self.kind {
            WatchKind::Write => write,
            WatchKind::Read => !write,
            WatchKind::Access => true,
        };
        // Widened so a range that ends at the top of the address space does not wrap.
        let (access_start, watch_start) = (linear as u128, self.address as u128);
        kind && access_start < watch_start + self.length as u128
            && watch_start < access_start + size as u128
    }
}

/// Which `Z` packets placed a breakpoint at an address: gdb reports them differently.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct BreakpointKinds {
    pub software: bool,
    pub hardware: bool,
}

/// What the debugger has set, the same for every vCPU.
#[derive(Clone, Default, Debug)]
pub(crate) struct DebugConfig {
    /// By linear address.
    pub breakpoints: BTreeMap<u64, BreakpointKinds>,
    pub watchpoints: Vec<Watchpoint>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StopKind {
    /// A pause request (`DebugLink::pause`); `serve` turns it into one of the two below.
    Paused,
    /// The halt at start, a connection, a detach.
    Quiet,
    Interrupted,
    Breakpoint(u64),
    Step,
    Watch(WatchKind, u64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct StopEvent {
    pub vcpu: usize,
    pub kind: StopKind,
}

impl StopEvent {
    fn signal(&self) -> u8 {
        match self.kind {
            StopKind::Interrupted => SIGINT,
            _ => SIGTRAP,
        }
    }
}

/// What one vCPU does when the machine resumes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ResumeAction {
    Continue,
    Step,
    /// Stays stopped while the others run (gdb stepping one thread over a breakpoint).
    Hold,
}

type OutputFn = Arc<dyn Fn(&[u8]) + Send + Sync>;

struct Session {
    /// Client bytes the engine has not looked at.
    inbox: Vec<u8>,
    engine: Engine,
    config: DebugConfig,
    /// A vCPU is in `serve`.
    stopped: bool,
    /// What the machine stopped for last; `?` reports it. Survives a reconnection.
    last_stop: Option<StopEvent>,
    /// The client went away: clear everything and let the machine run.
    detach_requested: bool,
    /// The machine has ended; nothing will serve the client any more.
    ended: bool,
    /// Bytes for the client, handed to the output callback once the session is unlocked.
    outbox: Vec<u8>,
}

/// What the vCPUs and the embedder's `GdbStub` share.
pub(crate) struct DebugLink {
    pause: AtomicU8,
    /// Mirrors `Session::stopped` for readers that must not take the lock: the device thread
    /// (which stops ticking, so guest time does not run on during a stop) and halted vCPU
    /// threads (which must not read their own `Cpu` while the debugger may be writing it).
    stopped: AtomicBool,
    /// Per vCPU: held by the last resume (`ResumeAction::Hold`).
    held: Vec<AtomicBool>,
    /// The machine's stop flag, for `k`.
    machine_stop: Arc<AtomicBool>,
    session: Mutex<Session>,
    wake: Condvar,
    output: Mutex<Option<OutputFn>>,
}

impl DebugLink {
    /// `halt_at_start` makes the first vCPU stop before the first instruction.
    pub fn new(cpus: usize, machine_stop: Arc<AtomicBool>, halt_at_start: bool) -> Arc<Self> {
        Arc::new(DebugLink {
            pause: AtomicU8::new(if halt_at_start {
                PAUSE_QUIET
            } else {
                PAUSE_NONE
            }),
            stopped: AtomicBool::new(false),
            held: (0..cpus).map(|_| AtomicBool::new(false)).collect(),
            machine_stop,
            session: Mutex::new(Session {
                inbox: Vec::new(),
                engine: Engine::new(),
                config: DebugConfig::default(),
                stopped: false,
                last_stop: None,
                detach_requested: false,
                ended: false,
                outbox: Vec::new(),
            }),
            wake: Condvar::new(),
            output: Mutex::new(None),
        })
    }

    pub fn pause_pending(&self) -> bool {
        self.pause.load(Ordering::Relaxed) != PAUSE_NONE
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    pub fn is_held(&self, vcpu: usize) -> bool {
        self.held
            .get(vcpu)
            .is_some_and(|held| held.load(Ordering::Acquire))
    }

    pub fn set_held(&self, vcpu: usize, held: bool) {
        if let Some(flag) = self.held.get(vcpu) {
            flag.store(held, Ordering::Release);
        }
    }

    pub fn config(&self) -> DebugConfig {
        self.session.lock().unwrap().config.clone()
    }

    fn request_pause(&self, kind: u8) {
        // A Ctrl-C wins over a quiet pause still pending: the client is waiting for its reply.
        let _ = self
            .pause
            .compare_exchange(PAUSE_NONE, kind, Ordering::SeqCst, Ordering::SeqCst);
        if kind == PAUSE_INTERRUPT {
            self.pause.store(kind, Ordering::SeqCst);
        }
    }

    /// Hands what the session has for the client to the output callback, with the session
    /// unlocked: the callback may block on a socket or call back into the stub.
    fn flush<'a>(
        &'a self,
        mut session: crate::sync::MutexGuard<'a, Session>,
    ) -> crate::sync::MutexGuard<'a, Session> {
        if session.outbox.is_empty() {
            return session;
        }
        let bytes = core::mem::take(&mut session.outbox);
        drop(session);
        let output = self.output.lock().unwrap().clone();
        if let Some(output) = output {
            output(&bytes);
        }
        self.session.lock().unwrap()
    }

    /// The machine ended (`Vcpu::run` returned): a client waiting for the stop of a resume gets
    /// `W00`, the process exiting.
    pub fn machine_ended(&self) {
        let mut session = self.session.lock().unwrap();
        session.ended = true;
        session.stopped = false;
        self.stopped.store(false, Ordering::Release);
        if session.engine.take_resume_pending() {
            packet::frame(b"W00", &mut session.outbox);
        }
        drop(self.flush(session));
    }

    /// Runs the debugger while the machine is stopped for `event`, on the vCPU that stopped,
    /// with every other vCPU stopped as well. Returns when the client resumes the machine:
    /// `Some` ends the run (a kill, or the machine's own stop flag).
    pub fn serve(&self, vcpus: &mut dyn Vcpus, event: StopEvent) -> Option<ExitReason> {
        let mut session = self.session.lock().unwrap();
        let mut event = event;
        // Several vCPUs may have seen one pause, and a breakpoint or watchpoint seen before
        // another vCPU's stop may have been removed meanwhile: only real stops reach the client.
        match event.kind {
            StopKind::Paused => {
                event.kind = match self.pause.swap(PAUSE_NONE, Ordering::SeqCst) {
                    PAUSE_NONE => return None,
                    PAUSE_INTERRUPT => StopKind::Interrupted,
                    _ => StopKind::Quiet,
                };
            }
            StopKind::Breakpoint(address) if !session.config.breakpoints.contains_key(&address) => {
                return None;
            }
            _ => {}
        }
        if self.machine_stop.load(Ordering::SeqCst) || session.ended {
            return None;
        }
        session.stopped = true;
        self.stopped.store(true, Ordering::Release);
        session.last_stop = Some(event);
        // A client that went away is owed nothing, and the next one must not get its reply.
        if session.engine.take_resume_pending() && !session.detach_requested {
            let reply = session.engine.stop_reply(&event, &session.config);
            packet::frame(&reply, &mut session.outbox);
        }
        let mut outcome = None;
        loop {
            if session.detach_requested {
                session.detach_requested = false;
                session.config = DebugConfig::default();
                outcome = Some(Outcome::Detach);
            }
            if outcome.is_none() {
                let inbox = core::mem::take(&mut session.inbox);
                let Session {
                    engine,
                    config,
                    outbox,
                    last_stop,
                    ..
                } = &mut *session;
                outcome = engine.process(&inbox, vcpus, config, *last_stop, outbox);
            }
            session = self.flush(session);
            let stopping = self.machine_stop.load(Ordering::SeqCst);
            if let Some(outcome) = outcome.take().or(stopping.then_some(Outcome::Kill)) {
                let kill = matches!(outcome, Outcome::Kill);
                let actions = match outcome {
                    Outcome::Resume(actions) => actions,
                    Outcome::Detach | Outcome::Kill => {
                        session.config = DebugConfig::default();
                        vec![ResumeAction::Continue; vcpus.count()]
                    }
                };
                apply_resume(vcpus, self, &session.config, &actions);
                session.stopped = false;
                self.stopped.store(false, Ordering::Release);
                if kill {
                    self.machine_stop.store(true, Ordering::SeqCst);
                    return Some(ExitReason::Stopped);
                }
                return None;
            }
            if !session.inbox.is_empty() {
                continue;
            }
            session = self.wake.wait_timeout(session, SERVE_POLL).unwrap().0;
            // A freestanding host has one thread and no way to block on the condition: its idle
            // hook is where it can feed the stub.
            #[cfg(freestanding)]
            {
                drop(session);
                if let Some(poll) = &vcpus.machine().poll {
                    poll(true);
                }
                session = self.session.lock().unwrap();
            }
        }
    }
}

/// The debugger end of a machine built with `MachineConfig::gdb_stub`: bytes from the client go
/// in through `from_client`, bytes for it come out through `on_output`. One client at a time.
/// Cloneable; every clone is the same stub.
#[derive(Clone)]
pub struct GdbStub {
    link: Arc<DebugLink>,
}

impl GdbStub {
    pub(crate) fn new(link: Arc<DebugLink>) -> Self {
        GdbStub { link }
    }

    /// Where the bytes for the client go. Called from a vCPU thread, never with the stub's state
    /// locked, so it may block (on a socket) or call back into the stub.
    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + Sync + 'static) {
        *self.link.output.lock().unwrap() = Some(Arc::new(f));
    }

    /// A client connected: the machine stops (as QEMU does), and the protocol starts afresh
    /// (acknowledgments on, no thread selected). Breakpoints a previous client left were cleared
    /// when it disconnected.
    pub fn connected(&self) {
        let mut session = self.link.session.lock().unwrap();
        session.engine = Engine::new();
        session.inbox.clear();
        session.detach_requested = false;
        if !session.stopped && !session.ended {
            self.link.request_pause(PAUSE_QUIET);
        }
        drop(session);
        self.link.wake.notify_all();
    }

    /// Bytes the client sent.
    pub fn from_client(&self, bytes: &[u8]) {
        let mut session = self.link.session.lock().unwrap();
        if session.ended {
            return;
        }
        if session.stopped {
            session.inbox.extend_from_slice(bytes);
        } else {
            // While the machine runs a client in all-stop mode only interrupts it; packets it
            // sends anyway wait for the stop.
            for &byte in bytes {
                if byte == packet::INTERRUPT {
                    self.link.request_pause(PAUSE_INTERRUPT);
                } else {
                    session.inbox.push(byte);
                }
            }
        }
        drop(session);
        self.link.wake.notify_all();
    }

    /// The client went away: its breakpoints and watchpoints go and the machine runs on, as
    /// after a detach.
    pub fn disconnected(&self) {
        let mut session = self.link.session.lock().unwrap();
        session.inbox.clear();
        if session.ended {
            return;
        }
        session.detach_requested = true;
        if !session.stopped {
            // Armed vCPUs only learn of the cleared breakpoints from a stop.
            self.link.request_pause(PAUSE_QUIET);
        }
        drop(session);
        self.link.wake.notify_all();
    }
}
