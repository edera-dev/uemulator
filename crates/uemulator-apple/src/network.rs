//! The native engine's network (docs/plans/28-apple.md, M5): the library's `Gateway` (ARP, DHCP,
//! ICMP echo, TCP/UDP NAT for 10.0.2.0/24, as the CLI's usernet) whose streams a `WispClient`
//! carries over Wisp v1 to the app, which opens the sockets (`apple/App/Wisp.swift`). The same
//! server code serves the WebKit engine, whose worker has its own gateway and client: one socket
//! implementation, on Network.framework, for both engines.
//!
//! A thread pumps it as the web crate's `NetLink::pump` does, in the `WispClient` contract's
//! order: the server's messages, the guest's frames, `WispClient::pump`, `Gateway::poll`.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use uemulator::{Endpoint, Gateway, GatewayConfig, GatewayPolicy, NetHandle, WispClient};

/// Where the guest's DNS (10.0.2.3:53) goes: the Wisp server opens a UDP stream to it. The web
/// page uses the same resolver (`www/main.js` WISP_RESOLVER).
const RESOLVER: Endpoint = Endpoint {
    address: [1, 1, 1, 1],
    port: 53,
};
/// The longest the pump sleeps with nothing to do: the gateway's timers (UDP flow expiry, the
/// Wisp handshake's 10 s) need time to pass, not this precisely.
const IDLE_WAIT: Duration = Duration::from_millis(50);
/// The pump's wait while the guest's receive queue is full: soon, but not spinning.
const BACKLOG_WAIT: Duration = Duration::from_millis(1);

/// `ua_wisp_send_fn`: one Wisp packet from the client, for the server.
pub type WispSendFn = extern "C" fn(context: *mut c_void, packet: *const u8, len: usize);

#[derive(Clone, Copy)]
pub struct SendContext(pub *mut c_void);
// SAFETY: handed back to the app's callback only, which runs on the pump thread and copies.
unsafe impl Send for SendContext {}

impl SendContext {
    fn pointer(self) -> *mut c_void {
        self.0
    }
}

enum ServerEvent {
    Connecting,
    Message(Vec<u8>),
    Lost,
}

#[derive(Default)]
struct Inbox {
    events: VecDeque<ServerEvent>,
    /// Set by the NIC's guest-frame hook and by new events, cleared by the pump.
    woken: bool,
}

/// The app's half: hands the server's events in and stops the pump.
pub struct Network {
    inbox: Arc<(Mutex<Inbox>, Condvar)>,
    stop: Arc<AtomicBool>,
    pump: Mutex<Option<JoinHandle<()>>>,
}

impl Network {
    pub fn start(net: NetHandle, send: WispSendFn, context: SendContext, isn_seed: u32) -> Network {
        let inbox: Arc<(Mutex<Inbox>, Condvar)> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let wake = inbox.clone();
        net.on_guest_frame(move || {
            let (lock, ready) = &*wake;
            lock.lock().unwrap().woken = true;
            ready.notify_one();
        });
        let pump = {
            let inbox = inbox.clone();
            let stop = stop.clone();
            std::thread::spawn(move || pump(net, inbox, stop, send, context, isn_seed))
        };
        Network {
            inbox,
            stop,
            pump: Mutex::new(Some(pump)),
        }
    }

    fn post(&self, event: ServerEvent) {
        let (lock, ready) = &*self.inbox;
        let mut inbox = lock.lock().unwrap();
        inbox.events.push_back(event);
        inbox.woken = true;
        ready.notify_one();
    }

    pub fn connecting(&self) {
        self.post(ServerEvent::Connecting);
    }

    pub fn message(&self, bytes: &[u8]) {
        self.post(ServerEvent::Message(bytes.to_vec()));
    }

    pub fn lost(&self) {
        self.post(ServerEvent::Lost);
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.inbox.1.notify_one();
        if let Some(pump) = self.pump.lock().unwrap().take() {
            let _ = pump.join();
        }
    }
}

fn pump(
    net: NetHandle,
    inbox: Arc<(Mutex<Inbox>, Condvar)>,
    stop: Arc<AtomicBool>,
    send: WispSendFn,
    context: SendContext,
    isn_seed: u32,
) {
    let mut gateway = Gateway::new(GatewayConfig {
        policy: GatewayPolicy::UserNet { dns: RESOLVER },
        isn_seed,
    });
    let mut wisp = WispClient::new();
    let epoch = Instant::now();
    let mut backlog = false;
    while !stop.load(Ordering::Acquire) {
        let events: Vec<ServerEvent> = {
            let (lock, ready) = &*inbox;
            let mut state = lock.lock().unwrap();
            if !state.woken && state.events.is_empty() {
                let wait = if backlog { BACKLOG_WAIT } else { IDLE_WAIT };
                state = ready.wait_timeout(state, wait).unwrap().0;
            }
            state.woken = false;
            state.events.drain(..).collect()
        };
        let now = epoch.elapsed().as_millis() as u64;
        for event in events {
            match event {
                ServerEvent::Connecting => wisp.connecting(now),
                ServerEvent::Message(message) => wisp.from_server(&message, &mut gateway),
                ServerEvent::Lost => wisp.connection_lost(&mut gateway),
            }
        }
        while let Some(frame) = net.try_recv_frame() {
            gateway.from_guest(&frame, now);
        }
        wisp.pump(&mut gateway, now);
        gateway.poll(now);
        // Kept, not dropped, when the guest's queue is full: they carry stream bytes the Wisp
        // server will not send again (`Gateway::poll` stops taking remote data meanwhile).
        backlog = false;
        while let Some(frame) = gateway.pop_for_guest() {
            let queued = match frame.segment_size {
                Some(size) => net.send_tcp_segment(&frame.bytes, size),
                None => net.send_frame(&frame.bytes),
            };
            if !queued {
                gateway.requeue_for_guest(frame);
                backlog = true;
                break;
            }
        }
        for message in wisp.take_messages() {
            send(context.pointer(), message.as_ptr(), message.len());
        }
    }
}
