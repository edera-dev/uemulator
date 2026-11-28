//! The terminal's input, read by one thread for the whole process and handed to whichever
//! machine is running. A bootstrap run starts the installed system's machine after the
//! installer's, and each later reset starts another, all in one process. With a reader thread
//! per machine, the earlier ones stay blocked in `read` after their machine ended (a blocking
//! read of stdin cannot be interrupted) and win most reads: on a real install every keystroke at
//! the installed system's login prompt went to the ended installer, Ctrl-] included.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

pub const READ_CHUNK_LEN: usize = 256;
/// How long an ended machine's handler may keep the input before it notices and lets go. Bytes
/// that arrive meanwhile are kept for the next machine, so this only delays them.
const ENDED_POLL_INTERVAL: Duration = Duration::from_millis(50);

pub struct TerminalInput {
    /// Whoever holds the lock owns the input; the next machine's handler waits on it.
    chunks: Mutex<Chunks>,
}

struct Chunks {
    receiver: Receiver<Vec<u8>>,
    /// One taken by a handler that then found its machine had ended, for the next one.
    held: Option<Vec<u8>>,
}

impl TerminalInput {
    /// Starts reading `reader` until it ends or fails.
    pub fn spawn(mut reader: impl Read + Send + 'static) -> Arc<Self> {
        let (sender, chunks) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = [0u8; READ_CHUNK_LEN];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 || sender.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        Arc::new(Self {
            chunks: Mutex::new(Chunks {
                receiver: chunks,
                held: None,
            }),
        })
    }

    /// The process's standard input, read from the first call on.
    pub fn stdin() -> Arc<Self> {
        static STDIN: OnceLock<Arc<TerminalInput>> = OnceLock::new();
        STDIN.get_or_init(|| Self::spawn(std::io::stdin())).clone()
    }

    /// Hands every chunk to `handle` on a thread of its own, once an earlier handler has let go,
    /// until `handle` returns false, `ended` is set or the input ends.
    pub fn attach(
        self: &Arc<Self>,
        ended: Arc<AtomicBool>,
        mut handle: impl FnMut(&[u8]) -> bool + Send + 'static,
    ) -> JoinHandle<()> {
        let input = self.clone();
        std::thread::spawn(move || {
            let mut guard = input.chunks.lock().unwrap_or_else(|e| e.into_inner());
            let Chunks { receiver, held } = &mut *guard;
            loop {
                let chunk = match held.take() {
                    Some(chunk) => chunk,
                    None => match receiver.recv_timeout(ENDED_POLL_INTERVAL) {
                        Ok(chunk) => chunk,
                        Err(RecvTimeoutError::Timeout) if !ended.load(Ordering::Acquire) => {
                            continue;
                        }
                        Err(_) => return,
                    },
                };
                if ended.load(Ordering::Acquire) {
                    *held = Some(chunk);
                    return;
                }
                if !handle(&chunk) {
                    return;
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::Instant;

    use super::*;

    const PATIENCE: Duration = Duration::from_secs(5);

    /// What a machine was handed.
    type Received = Arc<Mutex<Vec<u8>>>;

    fn collector() -> (Received, impl FnMut(&[u8]) -> bool + Send + 'static) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        (seen, move |bytes: &[u8]| {
            sink.lock().unwrap().extend_from_slice(bytes);
            true
        })
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + PATIENCE;
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A bootstrap run starts the installed system's machine after the installer's in the same
    /// process: the second must get every byte typed after the first ended, Ctrl-] included.
    #[test]
    fn a_machine_started_after_another_ended_gets_all_later_input() {
        let (reader, mut writer) = std::io::pipe().unwrap();
        let input = TerminalInput::spawn(reader);

        let first_ended = Arc::new(AtomicBool::new(false));
        let (first, handler) = collector();
        let first_thread = input.attach(first_ended.clone(), handler);
        writer.write_all(b"one").unwrap();
        wait_until("the first machine's input", || {
            *first.lock().unwrap() == b"one"
        });
        first_ended.store(true, Ordering::Release);

        // Typed while no machine runs (the reboot between the two): kept for the next one.
        writer.write_all(b"two").unwrap();
        let (second, handler) = collector();
        input.attach(Arc::new(AtomicBool::new(false)), handler);
        for chunk in [&b"three"[..], b"four", b"five"] {
            writer.write_all(chunk).unwrap();
        }
        wait_until("the second machine's input", || {
            *second.lock().unwrap() == b"twothreefourfive"
        });
        first_thread.join().unwrap();
        assert_eq!(*first.lock().unwrap(), b"one");
    }

    #[test]
    fn a_handler_that_stops_lets_the_next_machine_have_the_input() {
        let (reader, mut writer) = std::io::pipe().unwrap();
        let input = TerminalInput::spawn(reader);
        let first_thread = input.attach(Arc::new(AtomicBool::new(false)), |_: &[u8]| false);
        writer.write_all(b"\x1d").unwrap();
        first_thread.join().unwrap();
        let (second, handler) = collector();
        input.attach(Arc::new(AtomicBool::new(false)), handler);
        writer.write_all(b"after").unwrap();
        wait_until("the second machine's input", || {
            *second.lock().unwrap() == b"after"
        });
    }
}
