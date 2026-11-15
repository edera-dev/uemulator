#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;
use core::time::Duration;

use crate::sync::{Arc, Condvar, Mutex};

/// Cap on unread guest output (1 MiB). When no host thread reads, the oldest bytes are dropped
/// rather than blocking the guest. Input (`rx`) is not capped.
const OUTPUT_QUEUE_CAPACITY: usize = 1 << 20;

/// Window size reported until the host calls `resize`: the classic text-mode console.
const DEFAULT_COLS: u16 = 80;
const DEFAULT_ROWS: u16 = 25;

/// Push-style output callback, see `ConsoleHandle::on_output`.
type Sink = Box<dyn Fn(&[u8]) + Send>;

struct ConsoleState {
    /// Guest to host (serial TX / virtio-console transmit).
    tx: VecDeque<u8>,
    /// Host to guest.
    rx: VecDeque<u8>,
    /// While set, guest output goes here instead of `tx`.
    sink: Option<Sink>,
    cols: u16,
    rows: u16,
    /// Set by `resize`, cleared by `take_size_change`, so the virtio-console device reports each
    /// change to the guest once.
    size_dirty: bool,
}

impl Default for ConsoleState {
    fn default() -> Self {
        ConsoleState {
            tx: VecDeque::new(),
            rx: VecDeque::new(),
            sink: None,
            cols: DEFAULT_COLS,
            rows: DEFAULT_ROWS,
            size_dirty: false,
        }
    }
}

/// Byte stream between host and guest console, shared by clones. The vCPU thread (UART or
/// virtio-console device) and any number of host threads use the same single lock; the condvar
/// only wakes a host `read` blocked on empty guest output.
#[derive(Clone, Default)]
pub struct ConsoleHandle {
    inner: Arc<(Mutex<ConsoleState>, Condvar)>,
}

impl ConsoleHandle {
    /// Whether both handles are clones of one console (same queues), as opposed to two consoles.
    pub(crate) fn is_same_console(&self, other: &ConsoleHandle) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Host to guest.
    pub fn write(&self, bytes: &[u8]) {
        let mut state = self.inner.0.lock().unwrap();
        state.rx.extend(bytes);
    }

    /// Guest to host. Without a timeout it never blocks. With one, waits once for output if none is queued, so it can return 0
    /// bytes after the timeout or a spurious wakeup. Output sent to an `on_output` callback never
    /// appears here.
    pub fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> usize {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if state.tx.is_empty()
            && let Some(timeout) = timeout
        {
            state = output_ready.wait_timeout(state, timeout).unwrap().0;
        }
        let count = buf.len().min(state.tx.len());
        for (slot, byte) in buf.iter_mut().zip(state.tx.drain(..count)) {
            *slot = byte;
        }
        count
    }

    pub fn resize(&self, cols: u16, rows: u16) {
        let mut state = self.inner.0.lock().unwrap();
        if (state.cols, state.rows) != (cols, rows) {
            state.cols = cols;
            state.rows = rows;
            state.size_dirty = true;
        }
    }

    pub fn size(&self) -> (u16, u16) {
        let state = self.inner.0.lock().unwrap();
        (state.cols, state.rows)
    }

    /// Replaces polling with a push callback. It runs on the vCPU thread with the console lock
    /// held, so it must be quick and must not call back into this `ConsoleHandle`. Output queued
    /// before it was installed stays readable.
    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static) {
        self.inner.0.lock().unwrap().sink = Some(Box::new(f));
    }

    pub(crate) fn guest_tx(&self, byte: u8) {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if let Some(sink) = &state.sink {
            sink(&[byte]);
            return;
        }
        if state.tx.len() >= OUTPUT_QUEUE_CAPACITY {
            state.tx.pop_front();
        }
        state.tx.push_back(byte);
        output_ready.notify_one();
    }

    pub(crate) fn guest_rx(&self) -> Option<u8> {
        self.inner.0.lock().unwrap().rx.pop_front()
    }

    pub(crate) fn guest_tx_bytes(&self, bytes: &[u8]) {
        let (lock, output_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if let Some(sink) = &state.sink {
            sink(bytes);
            return;
        }
        for &byte in bytes {
            if state.tx.len() >= OUTPUT_QUEUE_CAPACITY {
                state.tx.pop_front();
            }
            state.tx.push_back(byte);
        }
        output_ready.notify_one();
    }

    pub(crate) fn has_input(&self) -> bool {
        !self.inner.0.lock().unwrap().rx.is_empty()
    }

    pub(crate) fn guest_rx_bulk(&self, max: usize) -> Vec<u8> {
        let mut state = self.inner.0.lock().unwrap();
        let count = max.min(state.rx.len());
        state.rx.drain(..count).collect()
    }

    /// Put input back ahead of bytes added by the host while DMA was attempted.
    pub(crate) fn guest_unrx_bulk(&self, bytes: &[u8]) {
        let mut state = self.inner.0.lock().unwrap();
        for &byte in bytes.iter().rev() {
            state.rx.push_front(byte);
        }
    }

    pub(crate) fn take_size_change(&self) -> Option<(u16, u16)> {
        let mut state = self.inner.0.lock().unwrap();
        if state.size_dirty {
            state.size_dirty = false;
            Some((state.cols, state.rows))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::mpsc;
    use std::time::Instant;

    /// Short enough to keep tests quick, long enough that a read that was going to find data
    /// finds it.
    const SHORT_TIMEOUT: Duration = Duration::from_millis(5);
    /// Long enough that waiting it out would be a visible failure (reads that must not wait use it).
    const NEVER_EXPIRES: Duration = Duration::from_secs(30);
    const REASONABLE: Duration = Duration::from_secs(10);

    fn drain(console: &ConsoleHandle) -> Vec<u8> {
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let count = console.read(&mut buf, None);
            if count == 0 {
                return out;
            }
            out.extend_from_slice(&buf[..count]);
        }
    }

    fn collecting_sink(console: &ConsoleHandle) -> Arc<StdMutex<Vec<Vec<u8>>>> {
        let chunks = Arc::new(StdMutex::new(Vec::new()));
        let sink = chunks.clone();
        console.on_output(move |bytes| sink.lock().unwrap().push(bytes.to_vec()));
        chunks
    }

    mod output_queue {
        use super::*;

        #[test]
        fn read_returns_what_the_guest_sent_in_order() {
            let console = ConsoleHandle::default();
            for &byte in b"hello" {
                console.guest_tx(byte);
            }
            let mut buf = [0u8; 16];
            let count = console.read(&mut buf, None);
            assert_eq!(&buf[..count], b"hello");
        }

        #[test]
        fn read_with_nothing_queued_returns_zero_without_blocking() {
            let console = ConsoleHandle::default();
            let started = Instant::now();
            assert_eq!(console.read(&mut [0u8; 4], None), 0);
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn read_is_limited_by_the_buffer_and_leaves_the_rest_queued() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(b"abcdef");
            let mut buf = [0u8; 4];
            assert_eq!(console.read(&mut buf, None), 4);
            assert_eq!(&buf, b"abcd");
            assert_eq!(console.read(&mut buf, None), 2);
            assert_eq!(&buf[..2], b"ef");
            assert_eq!(console.read(&mut buf, None), 0);
        }

        #[test]
        fn read_into_an_empty_buffer_consumes_nothing() {
            let console = ConsoleHandle::default();
            console.guest_tx(b'x');
            assert_eq!(console.read(&mut [], None), 0);
            assert_eq!(console.read(&mut [], Some(SHORT_TIMEOUT)), 0);
            assert_eq!(drain(&console), b"x");
        }

        #[test]
        fn read_with_a_timeout_and_data_queued_does_not_wait() {
            let console = ConsoleHandle::default();
            console.guest_tx(b'x');
            let started = Instant::now();
            let mut buf = [0u8; 4];
            assert_eq!(console.read(&mut buf, Some(NEVER_EXPIRES)), 1);
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn read_with_a_timeout_and_no_data_returns_zero_after_waiting() {
            let console = ConsoleHandle::default();
            let started = Instant::now();
            assert_eq!(console.read(&mut [0u8; 4], Some(SHORT_TIMEOUT)), 0);
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn a_zero_timeout_polls() {
            let console = ConsoleHandle::default();
            assert_eq!(console.read(&mut [0u8; 4], Some(Duration::ZERO)), 0);
            console.guest_tx(b'x');
            assert_eq!(console.read(&mut [0u8; 4], Some(Duration::ZERO)), 1);
        }

        #[test]
        fn a_blocked_reader_wakes_when_the_guest_sends() {
            let console = ConsoleHandle::default();
            let reader = {
                let console = console.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4];
                    let count = console.read(&mut buf, Some(NEVER_EXPIRES));
                    buf[..count].to_vec()
                })
            };
            // Whether the reader is already waiting or not yet started, it must see the byte.
            console.guest_tx(b'w');
            assert_eq!(reader.join().unwrap(), b"w");
        }

        #[test]
        fn a_blocked_reader_wakes_for_a_bulk_send_too() {
            let console = ConsoleHandle::default();
            let reader = {
                let console = console.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 8];
                    let count = console.read(&mut buf, Some(NEVER_EXPIRES));
                    buf[..count].to_vec()
                })
            };
            console.guest_tx_bytes(b"bulk");
            assert_eq!(reader.join().unwrap(), b"bulk");
        }

        #[test]
        fn overflow_drops_the_oldest_bytes() {
            let console = ConsoleHandle::default();
            let total = OUTPUT_QUEUE_CAPACITY + 10;
            for index in 0..total {
                console.guest_tx((index % 251) as u8);
            }
            let output = drain(&console);
            assert_eq!(output.len(), OUTPUT_QUEUE_CAPACITY);
            assert_eq!(output[0], 10);
            assert_eq!(*output.last().unwrap(), ((total - 1) % 251) as u8);
        }

        #[test]
        fn bulk_overflow_drops_the_oldest_bytes_the_same_way() {
            let console = ConsoleHandle::default();
            let total = OUTPUT_QUEUE_CAPACITY + 10;
            let data: Vec<u8> = (0..total).map(|index| (index % 251) as u8).collect();
            console.guest_tx_bytes(&data);
            assert_eq!(drain(&console), data[10..]);
        }

        #[test]
        fn a_single_bulk_larger_than_the_cap_keeps_only_its_tail() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(b"old");
            let data = vec![0x11u8; 3 * OUTPUT_QUEUE_CAPACITY];
            console.guest_tx_bytes(&data);
            let output = drain(&console);
            assert_eq!(output.len(), OUTPUT_QUEUE_CAPACITY);
            assert!(output.iter().all(|&byte| byte == 0x11));
        }

        #[test]
        fn a_full_queue_accepts_new_output_after_the_host_reads() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(&vec![1u8; OUTPUT_QUEUE_CAPACITY]);
            console.read(&mut [0u8; 100], None);
            console.guest_tx_bytes(&[2u8; 100]);
            let output = drain(&console);
            assert_eq!(output.len(), OUTPUT_QUEUE_CAPACITY);
            assert!(output[output.len() - 100..].iter().all(|&byte| byte == 2));
            assert_eq!(
                output.iter().filter(|&&byte| byte == 1).count(),
                OUTPUT_QUEUE_CAPACITY - 100
            );
        }

        #[test]
        fn exactly_the_cap_loses_nothing() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(&vec![5u8; OUTPUT_QUEUE_CAPACITY]);
            assert_eq!(drain(&console).len(), OUTPUT_QUEUE_CAPACITY);
        }

        #[test]
        fn empty_bulk_output_is_harmless() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(&[]);
            assert!(drain(&console).is_empty());
        }

        #[test]
        fn single_and_bulk_output_interleave_in_call_order() {
            let console = ConsoleHandle::default();
            console.guest_tx(b'a');
            console.guest_tx_bytes(b"bc");
            console.guest_tx(b'd');
            assert_eq!(drain(&console), b"abcd");
        }
    }

    mod input_queue {
        use super::*;

        #[test]
        fn guest_receives_host_bytes_in_order_one_at_a_time() {
            let console = ConsoleHandle::default();
            console.write(b"ab");
            console.write(b"c");
            assert_eq!(console.guest_rx(), Some(b'a'));
            assert_eq!(console.guest_rx(), Some(b'b'));
            assert_eq!(console.guest_rx(), Some(b'c'));
            assert_eq!(console.guest_rx(), None);
        }

        #[test]
        fn has_input_tracks_the_queue() {
            let console = ConsoleHandle::default();
            assert!(!console.has_input());
            console.write(b"x");
            assert!(console.has_input());
            console.guest_rx();
            assert!(!console.has_input());
        }

        #[test]
        fn writing_nothing_queues_nothing() {
            let console = ConsoleHandle::default();
            console.write(&[]);
            assert!(!console.has_input());
        }

        #[test]
        fn bulk_receive_takes_at_most_max_and_keeps_the_rest_in_order() {
            let console = ConsoleHandle::default();
            console.write(b"abcdef");
            assert_eq!(console.guest_rx_bulk(4), b"abcd");
            assert_eq!(console.guest_rx_bulk(100), b"ef");
            assert!(console.guest_rx_bulk(100).is_empty());
        }

        #[test]
        fn bulk_receive_of_zero_takes_nothing() {
            let console = ConsoleHandle::default();
            console.write(b"x");
            assert!(console.guest_rx_bulk(0).is_empty());
            assert!(console.has_input());
        }

        #[test]
        fn bulk_receive_of_usize_max_takes_everything() {
            let console = ConsoleHandle::default();
            console.write(b"all of it");
            assert_eq!(console.guest_rx_bulk(usize::MAX), b"all of it");
        }

        #[test]
        fn input_is_not_capped() {
            let console = ConsoleHandle::default();
            let total = OUTPUT_QUEUE_CAPACITY + 10;
            console.write(&vec![3u8; total]);
            assert_eq!(console.guest_rx_bulk(usize::MAX).len(), total);
        }

        #[test]
        fn input_and_output_queues_are_independent() {
            let console = ConsoleHandle::default();
            console.write(b"in");
            console.guest_tx(b'o');
            assert_eq!(drain(&console), b"o");
            assert_eq!(console.guest_rx_bulk(10), b"in");
        }
    }

    mod on_output {
        use super::*;

        #[test]
        fn a_callback_receives_each_single_byte_and_each_bulk_as_one_slice() {
            let console = ConsoleHandle::default();
            let chunks = collecting_sink(&console);
            console.guest_tx(b'a');
            console.guest_tx_bytes(b"bcd");
            console.guest_tx(b'e');
            assert_eq!(
                *chunks.lock().unwrap(),
                vec![b"a".to_vec(), b"bcd".to_vec(), b"e".to_vec()]
            );
        }

        #[test]
        fn callback_output_never_appears_in_read() {
            let console = ConsoleHandle::default();
            collecting_sink(&console);
            console.guest_tx(b'a');
            console.guest_tx_bytes(b"bc");
            assert_eq!(console.read(&mut [0u8; 8], None), 0);
        }

        #[test]
        fn output_queued_before_the_callback_stays_readable() {
            let console = ConsoleHandle::default();
            console.guest_tx_bytes(b"early");
            let chunks = collecting_sink(&console);
            console.guest_tx(b'l');
            assert_eq!(drain(&console), b"early");
            assert_eq!(*chunks.lock().unwrap(), vec![b"l".to_vec()]);
        }

        #[test]
        fn a_second_callback_replaces_the_first() {
            let console = ConsoleHandle::default();
            let first = collecting_sink(&console);
            console.guest_tx(b'1');
            let second = collecting_sink(&console);
            console.guest_tx(b'2');
            assert_eq!(*first.lock().unwrap(), vec![b"1".to_vec()]);
            assert_eq!(*second.lock().unwrap(), vec![b"2".to_vec()]);
        }

        #[test]
        fn callback_sees_bytes_in_the_order_the_guest_sent_them() {
            let console = ConsoleHandle::default();
            let chunks = collecting_sink(&console);
            let expected: Vec<u8> = (0..=255u8).collect();
            for &byte in &expected {
                console.guest_tx(byte);
            }
            let flat: Vec<u8> = chunks.lock().unwrap().iter().flatten().copied().collect();
            assert_eq!(flat, expected);
        }

        #[test]
        fn clones_share_the_callback() {
            let console = ConsoleHandle::default();
            let other = console.clone();
            let chunks = collecting_sink(&other);
            console.guest_tx(b'x');
            assert_eq!(chunks.lock().unwrap().len(), 1);
        }

        #[test]
        fn the_callback_runs_with_the_console_lock_held() {
            // A host thread touching the console meanwhile must wait for the callback to finish.
            let console = ConsoleHandle::default();
            let order = Arc::new(StdMutex::new(Vec::new()));
            let (entered_tx, entered_rx) = mpsc::channel();
            let callback_order = order.clone();
            console.on_output(move |_| {
                entered_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(10));
                callback_order.lock().unwrap().push("callback done");
            });
            let guest = {
                let console = console.clone();
                std::thread::spawn(move || console.guest_tx(b'x'))
            };
            entered_rx.recv_timeout(REASONABLE).unwrap();
            console.write(b"host");
            order.lock().unwrap().push("host write done");
            guest.join().unwrap();
            assert_eq!(*order.lock().unwrap(), ["callback done", "host write done"]);
        }
    }

    mod window_size {
        use super::*;

        #[test]
        fn starts_as_the_classic_80_by_25() {
            assert_eq!(ConsoleHandle::default().size(), (80, 25));
            assert_eq!((DEFAULT_COLS, DEFAULT_ROWS), (80, 25));
        }

        #[test]
        fn resize_changes_the_reported_size() {
            let console = ConsoleHandle::default();
            console.resize(132, 43);
            assert_eq!(console.size(), (132, 43));
        }

        #[test]
        fn a_fresh_console_has_no_size_change_to_report() {
            assert_eq!(ConsoleHandle::default().take_size_change(), None);
        }

        #[test]
        fn a_resize_is_reported_to_the_guest_exactly_once() {
            let console = ConsoleHandle::default();
            console.resize(100, 30);
            assert_eq!(console.take_size_change(), Some((100, 30)));
            assert_eq!(console.take_size_change(), None);
        }

        #[test]
        fn resizing_to_the_current_size_reports_nothing() {
            let console = ConsoleHandle::default();
            console.resize(80, 25);
            assert_eq!(console.take_size_change(), None);
            console.resize(100, 30);
            console.take_size_change();
            console.resize(100, 30);
            assert_eq!(console.take_size_change(), None);
        }

        #[test]
        fn several_resizes_before_the_guest_looks_collapse_into_the_latest() {
            let console = ConsoleHandle::default();
            console.resize(90, 20);
            console.resize(120, 40);
            console.resize(70, 10);
            assert_eq!(console.take_size_change(), Some((70, 10)));
            assert_eq!(console.take_size_change(), None);
        }

        #[test]
        fn each_dimension_alone_counts_as_a_change() {
            let console = ConsoleHandle::default();
            console.resize(81, 25);
            assert_eq!(console.take_size_change(), Some((81, 25)));
            console.resize(81, 26);
            assert_eq!(console.take_size_change(), Some((81, 26)));
        }

        #[test]
        fn resizing_away_and_back_before_the_guest_looks_still_reports_once() {
            // The guest gets one (harmless) notification carrying the unchanged size.
            let console = ConsoleHandle::default();
            console.resize(100, 30);
            console.resize(80, 25);
            assert_eq!(console.take_size_change(), Some((80, 25)));
        }

        #[test]
        fn zero_sizes_are_stored_as_given() {
            let console = ConsoleHandle::default();
            console.resize(0, 0);
            assert_eq!(console.size(), (0, 0));
            assert_eq!(console.take_size_change(), Some((0, 0)));
            console.resize(u16::MAX, u16::MAX);
            assert_eq!(console.size(), (u16::MAX, u16::MAX));
        }

        #[test]
        fn clones_share_the_size_and_the_pending_change() {
            let console = ConsoleHandle::default();
            let clone = console.clone();
            console.resize(60, 20);
            assert_eq!(clone.size(), (60, 20));
            assert_eq!(clone.take_size_change(), Some((60, 20)));
            assert_eq!(console.take_size_change(), None);
        }

        #[test]
        fn a_resize_does_not_disturb_the_byte_queues() {
            let console = ConsoleHandle::default();
            console.write(b"in");
            console.guest_tx(b'o');
            console.resize(10, 10);
            assert_eq!(drain(&console), b"o");
            assert_eq!(console.guest_rx_bulk(10), b"in");
        }
    }

    mod sharing {
        use super::*;

        #[test]
        fn separate_consoles_are_independent() {
            let (first, second) = (ConsoleHandle::default(), ConsoleHandle::default());
            first.guest_tx(b'x');
            first.write(b"y");
            first.resize(1, 1);
            assert_eq!(second.read(&mut [0u8; 4], None), 0);
            assert!(!second.has_input());
            assert_eq!(second.size(), (80, 25));
        }

        #[test]
        fn clones_share_both_queues() {
            let console = ConsoleHandle::default();
            let clone = console.clone();
            clone.write(b"to guest");
            assert_eq!(console.guest_rx_bulk(100), b"to guest");
            console.guest_tx(b'!');
            assert_eq!(drain(&clone), b"!");
        }

        #[test]
        fn a_vcpu_style_producer_and_a_host_reader_see_every_byte_in_order() {
            const TOTAL: usize = 20_000;
            let console = ConsoleHandle::default();
            let producer = {
                let console = console.clone();
                std::thread::spawn(move || {
                    for index in 0..TOTAL {
                        console.guest_tx((index % 251) as u8);
                    }
                })
            };
            let mut received = Vec::with_capacity(TOTAL);
            let mut buf = [0u8; 512];
            let started = Instant::now();
            while received.len() < TOTAL && started.elapsed() < REASONABLE {
                let count = console.read(&mut buf, Some(SHORT_TIMEOUT));
                received.extend_from_slice(&buf[..count]);
            }
            producer.join().unwrap();
            let expected: Vec<u8> = (0..TOTAL).map(|index| (index % 251) as u8).collect();
            assert_eq!(received, expected);
        }

        #[test]
        fn host_input_from_one_thread_reaches_the_guest_from_another() {
            const TOTAL: usize = 5_000;
            let console = ConsoleHandle::default();
            let writer = {
                let console = console.clone();
                std::thread::spawn(move || {
                    for index in 0..TOTAL {
                        console.write(&[(index % 251) as u8]);
                    }
                })
            };
            let mut received = Vec::new();
            let started = Instant::now();
            while received.len() < TOTAL && started.elapsed() < REASONABLE {
                match console.guest_rx() {
                    Some(byte) => received.push(byte),
                    None => std::thread::yield_now(),
                }
            }
            writer.join().unwrap();
            let expected: Vec<u8> = (0..TOTAL).map(|index| (index % 251) as u8).collect();
            assert_eq!(received, expected);
        }

        #[test]
        fn many_threads_resizing_and_reading_do_not_deadlock() {
            let console = ConsoleHandle::default();
            let threads: Vec<_> = (0..4u16)
                .map(|id| {
                    let console = console.clone();
                    std::thread::spawn(move || {
                        for step in 0..200u16 {
                            console.resize(80 + id, 25 + step % 3);
                            console.guest_tx(id as u8);
                            console.size();
                            console.read(&mut [0u8; 8], None);
                            console.take_size_change();
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        }
    }
}
