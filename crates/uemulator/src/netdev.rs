#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;
use core::time::Duration;

use crate::sync::{Arc, Condvar, Mutex};

/// Frames queued per direction. The two directions differ on overflow: a full host-to-guest queue
/// rejects the new frame (the sender can retry in order), a full guest-to-host queue drops the
/// oldest.
const QUEUE_FRAMES: usize = 1024;

/// Called after a frame is queued, outside the lock, so it may use the handle itself.
type Notify = Arc<dyn Fn() + Send + Sync>;

/// What the host end of a NIC does in the guest's place, declared with `NetHandle::set_offloads`
/// before the machine starts. The virtio-net device offers the guest the matching offloads only
/// then; a host that forwards frames to something expecting real Ethernet (another machine, a
/// socket link) declares nothing and keeps getting checksummed frames of at most one MTU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NetOffloads {
    /// The host takes guest frames as a NIC doing checksum and TCP segmentation offload would:
    /// TCP/IPv4 segments of up to 64 KiB with any MSS, and TCP or UDP checksums that hold only
    /// the pseudo-header sum. Right for a host that terminates TCP itself and never checks a
    /// checksum (virtio-net `CSUM` and `HOST_TSO4`).
    pub takes_guest_segments: bool,
    /// The host may hand the guest TCP segments longer than the MSS (`send_tcp_segment`); the
    /// guest is offered to take them whole (virtio-net `GUEST_CSUM`, `GUEST_TSO4`, `MRG_RXBUF`).
    pub sends_large_segments: bool,
}

/// A frame queued for the guest. `tcp_segment_size` marks a TCP/IPv4 segment from
/// `send_tcp_segment`: its checksum holds only the pseudo-header sum and its payload may be
/// longer than this MSS.
pub(crate) struct HostFrame {
    pub bytes: Vec<u8>,
    pub tcp_segment_size: Option<u16>,
}

#[derive(Default)]
struct NetState {
    to_guest: VecDeque<HostFrame>,
    from_guest: VecDeque<Vec<u8>>,
    offloads: NetOffloads,
    /// The guest's driver accepted `GUEST_CSUM`, `GUEST_TSO4` and `MRG_RXBUF`; set by the device
    /// when the features are negotiated and cleared on reset.
    guest_takes_segments: bool,
    /// Wakes whichever thread runs the devices (the vCPU of a one-processor machine, else the
    /// device thread), which otherwise only looks at `to_guest` when its sleep ends: without it a
    /// halted guest sees each frame up to 2 ms late, and that latency is what limits TCP.
    wake_devices: Option<Notify>,
    /// Set by the host (`on_guest_frame`) when it waits on more than the condvar.
    wake_host: Option<Notify>,
}

/// Ethernet frames between the host and the virtio-net device, shared by clones. One lock covers
/// both queues; the condvar only wakes a host `recv_frame` waiting for guest output.
#[derive(Clone, Default)]
pub struct NetHandle {
    inner: Arc<(Mutex<NetState>, Condvar)>,
}

impl NetHandle {
    /// Host to guest. Returns false when the queue is full; the frame is not queued.
    pub fn send_frame(&self, frame: &[u8]) -> bool {
        self.queue_for_guest(HostFrame {
            bytes: frame.to_vec(),
            tcp_segment_size: None,
        })
    }

    /// Host to guest: one Ethernet/IPv4/TCP frame whose TCP checksum field holds the folded
    /// pseudo-header sum (not complemented) instead of the checksum, and whose payload may be
    /// longer than `segment_size`, the connection's MSS. When `guest_takes_tcp_segments` is true
    /// the guest gets it whole and finishes the checksum itself. Otherwise the device completes
    /// the checksum, and drops the frame if its payload is longer than `segment_size`; a frame
    /// that is not TCP/IPv4 is dropped. Returns false when the queue is full.
    pub fn send_tcp_segment(&self, frame: &[u8], segment_size: u16) -> bool {
        self.queue_for_guest(HostFrame {
            bytes: frame.to_vec(),
            tcp_segment_size: Some(segment_size),
        })
    }

    fn queue_for_guest(&self, frame: HostFrame) -> bool {
        let wake = {
            let mut state = self.inner.0.lock().unwrap();
            if state.to_guest.len() >= QUEUE_FRAMES {
                return false;
            }
            state.to_guest.push_back(frame);
            state.wake_devices.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
        true
    }

    /// Declares what the host end handles for the guest (`NetOffloads`). Takes effect at the
    /// guest's next feature negotiation, so call it before the machine starts.
    pub fn set_offloads(&self, offloads: NetOffloads) {
        self.inner.0.lock().unwrap().offloads = offloads;
    }

    pub(crate) fn offloads(&self) -> NetOffloads {
        self.inner.0.lock().unwrap().offloads
    }

    /// True while the guest's driver takes TCP segments longer than the MSS with an unfinished
    /// checksum, in merged receive buffers, so `send_tcp_segment` frames reach it whole. False
    /// before the driver has negotiated, after a reset, and always unless `sends_large_segments`
    /// was declared. A large segment still queued when this turns false is dropped.
    pub fn guest_takes_tcp_segments(&self) -> bool {
        self.inner.0.lock().unwrap().guest_takes_segments
    }

    pub(crate) fn set_guest_takes_tcp_segments(&self, takes: bool) {
        self.inner.0.lock().unwrap().guest_takes_segments = takes;
    }

    /// Calls `notify` every time the guest queues a frame, from the emulator's thread and after
    /// the frame can be received. For a host that waits on other things too (its sockets) and so
    /// cannot block in `recv_frame`. Replaces an earlier notifier.
    pub fn on_guest_frame(&self, notify: impl Fn() + Send + Sync + 'static) {
        self.inner.0.lock().unwrap().wake_host = Some(Arc::new(notify));
    }

    /// See `NetState::wake_devices`. A freestanding host has no thread to wake.
    #[cfg(not(freestanding))]
    pub(crate) fn set_device_waker(&self, wake: Option<Notify>) {
        self.inner.0.lock().unwrap().wake_devices = wake;
    }

    pub fn try_recv_frame(&self) -> Option<Vec<u8>> {
        self.inner.0.lock().unwrap().from_guest.pop_front()
    }

    /// Waits once for a frame if none is queued, so a spurious wakeup can return `None` early.
    pub fn recv_frame(&self, timeout: Duration) -> Option<Vec<u8>> {
        let (lock, frame_ready) = &*self.inner;
        let mut state = lock.lock().unwrap();
        if state.from_guest.is_empty() {
            state = frame_ready.wait_timeout(state, timeout).unwrap().0;
        }
        state.from_guest.pop_front()
    }

    pub(crate) fn guest_send(&self, frame: Vec<u8>) {
        let (lock, frame_ready) = &*self.inner;
        let wake = {
            let mut state = lock.lock().unwrap();
            if state.from_guest.len() >= QUEUE_FRAMES {
                state.from_guest.pop_front();
            }
            state.from_guest.push_back(frame);
            frame_ready.notify_one();
            state.wake_host.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }

    #[cfg(test)]
    pub(crate) fn has_pending(&self) -> bool {
        !self.inner.0.lock().unwrap().to_guest.is_empty()
    }

    pub(crate) fn guest_recv(&self) -> Option<HostFrame> {
        self.inner.0.lock().unwrap().to_guest.pop_front()
    }

    /// Puts back a frame `guest_recv` took that the guest had no room for yet, ahead of the rest
    /// and regardless of the queue limit (it held that place already).
    pub(crate) fn guest_unrecv(&self, frame: HostFrame) {
        self.inner.0.lock().unwrap().to_guest.push_front(frame);
    }

    #[cfg(test)]
    pub(crate) fn guest_recv_bytes(&self) -> Option<Vec<u8>> {
        self.guest_recv().map(|frame| frame.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const SHORT_TIMEOUT: Duration = Duration::from_millis(5);
    const NEVER_EXPIRES: Duration = Duration::from_secs(30);
    const REASONABLE: Duration = Duration::from_secs(10);

    /// A frame whose first two bytes identify it.
    fn frame(id: usize) -> Vec<u8> {
        vec![(id >> 8) as u8, id as u8, 0xaa]
    }

    fn id_of(frame: &[u8]) -> usize {
        ((frame[0] as usize) << 8) | frame[1] as usize
    }

    mod host_to_guest {
        use super::*;

        #[test]
        fn frames_reach_the_guest_in_order_and_unmodified() {
            let net = NetHandle::default();
            assert!(net.send_frame(b"first"));
            assert!(net.send_frame(b"second frame"));
            assert_eq!(net.guest_recv_bytes().as_deref(), Some(&b"first"[..]));
            assert_eq!(
                net.guest_recv_bytes().as_deref(),
                Some(&b"second frame"[..])
            );
            assert_eq!(net.guest_recv_bytes(), None);
        }

        #[test]
        fn has_pending_tracks_the_queue() {
            let net = NetHandle::default();
            assert!(!net.has_pending());
            net.send_frame(b"x");
            assert!(net.has_pending());
            net.guest_recv_bytes();
            assert!(!net.has_pending());
        }

        #[test]
        fn the_frame_is_copied_so_the_sender_may_reuse_its_buffer() {
            let net = NetHandle::default();
            let mut buffer = vec![1u8, 2, 3];
            net.send_frame(&buffer);
            buffer.fill(9);
            assert_eq!(net.guest_recv_bytes().unwrap(), [1, 2, 3]);
        }

        #[test]
        fn empty_and_jumbo_frames_are_queued_as_they_are() {
            let net = NetHandle::default();
            let jumbo = vec![0x5au8; 70_000];
            assert!(net.send_frame(&[]));
            assert!(net.send_frame(&jumbo));
            assert_eq!(net.guest_recv_bytes().unwrap(), Vec::<u8>::new());
            assert_eq!(net.guest_recv_bytes().unwrap(), jumbo);
        }

        #[test]
        fn frames_are_not_merged_or_split() {
            let net = NetHandle::default();
            net.send_frame(b"ab");
            net.send_frame(b"cd");
            assert_eq!(net.guest_recv_bytes().unwrap(), b"ab");
            assert_eq!(net.guest_recv_bytes().unwrap(), b"cd");
        }

        #[test]
        fn a_full_queue_rejects_the_new_frame_and_keeps_the_old_ones() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES {
                assert!(net.send_frame(&frame(id)), "frame {id} must fit");
            }
            assert!(!net.send_frame(&frame(QUEUE_FRAMES)));
            assert!(!net.send_frame(&frame(QUEUE_FRAMES + 1)));
            // The first frame is still first, the rejected ones never appear.
            for id in 0..QUEUE_FRAMES {
                assert_eq!(id_of(&net.guest_recv_bytes().unwrap()), id);
            }
            assert_eq!(net.guest_recv_bytes(), None);
        }

        #[test]
        fn a_slot_freed_by_the_guest_accepts_a_frame_again_at_the_back() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES {
                net.send_frame(&frame(id));
            }
            assert!(!net.send_frame(&frame(9999)));
            assert_eq!(id_of(&net.guest_recv_bytes().unwrap()), 0);
            assert!(net.send_frame(&frame(9999)));
            assert!(!net.send_frame(&frame(10_000)));
            let mut last = 0;
            while let Some(received) = net.guest_recv_bytes() {
                last = id_of(&received);
            }
            assert_eq!(last, 9999);
        }

        #[test]
        fn the_two_directions_have_separate_queues() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES {
                net.send_frame(&frame(id));
            }
            // A full host-to-guest queue does not block the guest's own sends.
            net.guest_send(b"from guest".to_vec());
            assert_eq!(net.try_recv_frame().as_deref(), Some(&b"from guest"[..]));
        }
    }

    mod guest_to_host {
        use super::*;

        #[test]
        fn frames_reach_the_host_in_order() {
            let net = NetHandle::default();
            net.guest_send(b"one".to_vec());
            net.guest_send(b"two".to_vec());
            assert_eq!(net.try_recv_frame().as_deref(), Some(&b"one"[..]));
            assert_eq!(net.try_recv_frame().as_deref(), Some(&b"two"[..]));
            assert_eq!(net.try_recv_frame(), None);
        }

        #[test]
        fn try_recv_never_blocks() {
            let net = NetHandle::default();
            let started = Instant::now();
            assert_eq!(net.try_recv_frame(), None);
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn a_full_queue_drops_the_oldest_frame() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES + 5 {
                net.guest_send(frame(id));
            }
            let mut ids = Vec::new();
            while let Some(received) = net.try_recv_frame() {
                ids.push(id_of(&received));
            }
            assert_eq!(ids.len(), QUEUE_FRAMES);
            assert_eq!(ids.first(), Some(&5));
            assert_eq!(ids.last(), Some(&(QUEUE_FRAMES + 4)));
            assert!(ids.windows(2).all(|pair| pair[1] == pair[0] + 1));
        }

        #[test]
        fn exactly_the_cap_loses_nothing() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES {
                net.guest_send(frame(id));
            }
            assert_eq!(id_of(&net.try_recv_frame().unwrap()), 0);
        }

        #[test]
        fn recv_with_a_frame_queued_returns_it_without_waiting() {
            let net = NetHandle::default();
            net.guest_send(b"ready".to_vec());
            let started = Instant::now();
            assert_eq!(
                net.recv_frame(NEVER_EXPIRES).as_deref(),
                Some(&b"ready"[..])
            );
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn recv_with_nothing_queued_times_out_with_none() {
            let net = NetHandle::default();
            let started = Instant::now();
            assert_eq!(net.recv_frame(SHORT_TIMEOUT), None);
            assert!(started.elapsed() < REASONABLE);
        }

        #[test]
        fn a_zero_timeout_polls() {
            let net = NetHandle::default();
            assert_eq!(net.recv_frame(Duration::ZERO), None);
            net.guest_send(b"x".to_vec());
            assert!(net.recv_frame(Duration::ZERO).is_some());
        }

        #[test]
        fn a_blocked_receiver_wakes_when_the_guest_sends() {
            let net = NetHandle::default();
            let receiver = {
                let net = net.clone();
                std::thread::spawn(move || net.recv_frame(NEVER_EXPIRES))
            };
            net.guest_send(b"wake".to_vec());
            assert_eq!(receiver.join().unwrap().as_deref(), Some(&b"wake"[..]));
        }

        #[test]
        fn one_wakeup_serves_one_frame_when_two_receivers_wait() {
            let net = NetHandle::default();
            let spawn_receiver = |net: NetHandle| {
                std::thread::spawn(move || net.recv_frame(Duration::from_millis(200)))
            };
            let first = spawn_receiver(net.clone());
            let second = spawn_receiver(net.clone());
            net.guest_send(b"only one".to_vec());
            let results = [first.join().unwrap(), second.join().unwrap()];
            assert_eq!(results.iter().filter(|frame| frame.is_some()).count(), 1);
        }
    }

    mod notifiers {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        fn counter() -> (Arc<AtomicUsize>, Notify) {
            let count = Arc::new(AtomicUsize::new(0));
            let bump = count.clone();
            (
                count,
                Arc::new(move || {
                    bump.fetch_add(1, Ordering::SeqCst);
                }),
            )
        }

        #[test]
        fn a_frame_for_the_guest_wakes_the_devices() {
            let net = NetHandle::default();
            let (count, wake) = counter();
            net.set_device_waker(Some(wake));
            net.send_frame(b"x");
            net.send_frame(b"y");
            assert_eq!(count.load(Ordering::SeqCst), 2);
        }

        #[test]
        fn a_rejected_frame_wakes_nobody() {
            let net = NetHandle::default();
            for id in 0..QUEUE_FRAMES {
                net.send_frame(&frame(id));
            }
            let (count, wake) = counter();
            net.set_device_waker(Some(wake));
            assert!(!net.send_frame(b"full"));
            assert_eq!(count.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn a_guest_frame_notifies_the_host_once_it_can_be_received() {
            let net = NetHandle::default();
            let probe = net.clone();
            let seen = Arc::new(AtomicUsize::new(0));
            let seen_by_notifier = seen.clone();
            // Receiving inside the notifier proves the lock is released and the frame queued.
            net.on_guest_frame(move || {
                if probe.try_recv_frame().is_some() {
                    seen_by_notifier.fetch_add(1, Ordering::SeqCst);
                }
            });
            net.guest_send(b"up".to_vec());
            assert_eq!(seen.load(Ordering::SeqCst), 1);
        }
    }

    mod sharing {
        use super::*;

        #[test]
        fn clones_share_both_queues() {
            let net = NetHandle::default();
            let clone = net.clone();
            clone.send_frame(b"down");
            assert_eq!(net.guest_recv_bytes().unwrap(), b"down");
            net.guest_send(b"up".to_vec());
            assert_eq!(clone.try_recv_frame().unwrap(), b"up");
        }

        #[test]
        fn separate_handles_are_independent() {
            let (first, second) = (NetHandle::default(), NetHandle::default());
            first.send_frame(b"x");
            first.guest_send(b"y".to_vec());
            assert!(!second.has_pending());
            assert_eq!(second.try_recv_frame(), None);
        }

        #[test]
        fn a_producing_guest_thread_and_a_receiving_host_thread_exchange_every_frame() {
            const TOTAL: usize = 500;
            let net = NetHandle::default();
            let producer = {
                let net = net.clone();
                std::thread::spawn(move || {
                    for id in 0..TOTAL {
                        net.guest_send(frame(id));
                    }
                })
            };
            let mut ids = Vec::new();
            let started = Instant::now();
            while ids.len() < TOTAL && started.elapsed() < REASONABLE {
                if let Some(received) = net.recv_frame(SHORT_TIMEOUT) {
                    ids.push(id_of(&received));
                }
            }
            producer.join().unwrap();
            assert_eq!(ids, (0..TOTAL).collect::<Vec<_>>());
        }

        #[test]
        fn a_host_thread_sending_and_a_guest_thread_receiving_exchange_every_frame() {
            const TOTAL: usize = 500;
            let net = NetHandle::default();
            let sender = {
                let net = net.clone();
                std::thread::spawn(move || {
                    for id in 0..TOTAL {
                        // The queue is bounded; retry until the guest side makes room.
                        while !net.send_frame(&frame(id)) {
                            std::thread::yield_now();
                        }
                    }
                })
            };
            let mut ids = Vec::new();
            let started = Instant::now();
            while ids.len() < TOTAL && started.elapsed() < REASONABLE {
                match net.guest_recv_bytes() {
                    Some(received) => ids.push(id_of(&received)),
                    None => std::thread::yield_now(),
                }
            }
            sender.join().unwrap();
            assert_eq!(ids, (0..TOTAL).collect::<Vec<_>>());
        }
    }
}
