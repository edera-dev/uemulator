//! Wisp v1 (MercuryWorkshop/wisp-protocol): a WebSocket relay that carries the gateway's TCP and
//! UDP streams to the internet from a browser.
//!
//! `WispClient` holds no socket. The driver creates the WebSocket and says so (`connecting`),
//! hands it every message the server sends (`from_server`), sends whatever `take_messages`
//! returns, and reports the socket closing (`connection_lost`). Each round it calls
//! `from_server` for what arrived, feeds the guest's frames to the gateway, then `pump`, then
//! `Gateway::poll`.
//!
//! The gateway's stream ids are the Wisp stream ids: nonzero, and never reused while live on
//! either side. A stream counts as open as soon as its CONNECT is sent, since v1 has no
//! "connected" packet (the server reports only failure, with CLOSE), so a refused destination
//! reaches the guest as a reset of an established connection.

mod packet;
#[cfg(test)]
mod tests;

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;
use core::net::Ipv4Addr;

use crate::gateway::{Endpoint, Gateway, GatewayRequest, StreamId, StreamKind};
use packet::{CLOSE_VOLUNTARY, WispPacket, decode, encode};

/// Without the server's first CONTINUE by then the connection is given up: waiting opens fail
/// and the page reconnects.
const HANDSHAKE_TIMEOUT_MS: u64 = 10_000;
/// Largest DATA payload sent. Credit counts packets, not bytes, so packets are made this large.
const WISP_DATA_MAX: usize = 16384;
/// Stream 0 is the connection itself: its CONTINUE ends the handshake and gives every new
/// stream's credit.
const CONNECTION_STREAM: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WispState {
    /// No WebSocket: every open fails at once, so the guest is refused instead of timing out.
    Disconnected,
    /// The WebSocket exists and the server's first CONTINUE has not come: opens wait.
    Connecting,
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WispStatus {
    pub state: WispState,
    /// The reason of the last CLOSE from the server other than a voluntary one, kept across
    /// reconnects.
    pub last_close_reason: Option<u8>,
}

struct WispStream {
    kind: StreamKind,
    /// TCP DATA packets the server has room for. The server's CONTINUE replaces it (the value is
    /// what is left of its buffer, not an increment). UDP needs none.
    credit: u32,
    /// The server sent CLOSE 0x02 while the gateway still holds the guest's connection (draining
    /// to its FIN). Nothing more goes to the server, and the gateway's `Close` for it sends no
    /// second CLOSE.
    server_closed: bool,
}

struct HeldOpen {
    stream: StreamId,
    kind: StreamKind,
    target: Endpoint,
}

pub struct WispClient {
    state: WispState,
    connecting_since_ms: u64,
    /// Every new stream's credit, from the server's CONTINUE on stream 0.
    initial_credit: u32,
    /// Streams the server knows about, or that wait for the gateway to finish with them
    /// (`server_closed`). A stream the gateway has let go of is not here.
    streams: BTreeMap<StreamId, WispStream>,
    /// Opens asked for while `Connecting`, in order.
    held: Vec<HeldOpen>,
    /// `connecting` came while streams of the previous WebSocket were live; they are reset at the
    /// next call that has the gateway.
    reset_due: bool,
    outgoing: Vec<Vec<u8>>,
    last_close_reason: Option<u8>,
}

impl Default for WispClient {
    fn default() -> WispClient {
        WispClient::new()
    }
}

impl WispClient {
    pub fn new() -> WispClient {
        WispClient {
            state: WispState::Disconnected,
            connecting_since_ms: 0,
            initial_credit: 0,
            streams: BTreeMap::new(),
            held: Vec::new(),
            reset_due: false,
            outgoing: Vec::new(),
            last_close_reason: None,
        }
    }

    /// A new WebSocket was created; the handshake clock starts at `now_ms`. Meant to follow
    /// `new`, `connection_lost` or a handshake timeout; on a client that was still connected the
    /// old streams are reset as if the connection had been lost.
    pub fn connecting(&mut self, now_ms: u64) {
        if self.state != WispState::Disconnected {
            self.reset_due = true;
            self.outgoing.clear();
        }
        self.state = WispState::Connecting;
        self.connecting_since_ms = now_ms;
    }

    /// The WebSocket closed: every live stream is reset, waiting opens fail, and what was not yet
    /// taken from `take_messages` is dropped (it was for the closed socket).
    pub fn connection_lost(&mut self, gateway: &mut Gateway) {
        self.reset_streams(gateway);
        for open in core::mem::take(&mut self.held) {
            gateway.failed(open.stream);
        }
        self.state = WispState::Disconnected;
        self.outgoing.clear();
    }

    /// One WebSocket message from the server, applied to the gateway at once. Malformed packets,
    /// packets for streams this client does not have, and anything but the handshake's CONTINUE
    /// before `Ready`, are ignored.
    pub fn from_server(&mut self, message: &[u8], gateway: &mut Gateway) {
        if core::mem::take(&mut self.reset_due) {
            self.reset_streams(gateway);
        }
        let Some(packet) = decode(message) else {
            return;
        };
        match self.state {
            WispState::Disconnected => return,
            WispState::Connecting => {
                if let WispPacket::Continue {
                    stream: CONNECTION_STREAM,
                    remaining,
                } = packet
                {
                    self.state = WispState::Ready;
                    self.initial_credit = remaining;
                    for open in core::mem::take(&mut self.held) {
                        self.open(open, gateway);
                    }
                }
                return;
            }
            WispState::Ready => {}
        }
        match packet {
            WispPacket::Data { stream, payload } => {
                if self.is_live(stream) {
                    gateway.remote_data(stream, &payload);
                }
            }
            WispPacket::Continue { stream, remaining } => {
                if let Some(entry) = self.streams.get_mut(&stream)
                    && !entry.server_closed
                {
                    entry.credit = remaining;
                }
            }
            WispPacket::Close { stream, reason } => {
                if !self.is_live(stream) {
                    return;
                }
                if reason == CLOSE_VOLUNTARY {
                    gateway.remote_closed(stream);
                } else {
                    gateway.reset(stream);
                    self.last_close_reason = Some(reason);
                }
                if gateway.has_stream(stream) {
                    if let Some(entry) = self.streams.get_mut(&stream) {
                        entry.server_closed = true;
                    }
                } else {
                    self.streams.remove(&stream);
                }
            }
            // Only clients open streams in v1.
            WispPacket::Connect { .. } => {}
        }
    }

    /// Carries out what the gateway asked for (including what this call's own `opened` and
    /// `failed` release), sends waiting TCP data while credit lasts, and gives up a handshake
    /// that took `HANDSHAKE_TIMEOUT_MS`.
    pub fn pump(&mut self, gateway: &mut Gateway, now_ms: u64) {
        if core::mem::take(&mut self.reset_due) {
            self.reset_streams(gateway);
        }
        if self.state == WispState::Connecting
            && now_ms.saturating_sub(self.connecting_since_ms) >= HANDSHAKE_TIMEOUT_MS
        {
            self.state = WispState::Disconnected;
            for open in core::mem::take(&mut self.held) {
                gateway.failed(open.stream);
            }
        }
        loop {
            let requests = gateway.take_requests();
            if requests.is_empty() {
                break;
            }
            for request in requests {
                self.request(request, gateway);
            }
        }
        // `consumed` asks for nothing (the window it opens is announced at `poll`), so no request
        // can be waiting after this.
        self.send_data(gateway);
    }

    /// Encoded packets for the server, oldest first, one WebSocket binary message each.
    pub fn take_messages(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.outgoing)
    }

    pub fn status(&self) -> WispStatus {
        WispStatus {
            state: self.state,
            last_close_reason: self.last_close_reason,
        }
    }

    fn request(&mut self, request: GatewayRequest, gateway: &mut Gateway) {
        match request {
            GatewayRequest::Open {
                stream,
                kind,
                target,
            } => {
                let open = HeldOpen {
                    stream,
                    kind,
                    target,
                };
                match self.state {
                    WispState::Disconnected => gateway.failed(stream),
                    WispState::Connecting => self.held.push(open),
                    WispState::Ready => self.open(open, gateway),
                }
            }
            GatewayRequest::Datagram { stream, data } => {
                if self.is_live(stream) {
                    self.send(&WispPacket::Data {
                        stream,
                        payload: data,
                    });
                }
            }
            // v1 has no half-close. The gateway already passed the guest's data before its FIN,
            // and the stream ends with the server's CLOSE or the gateway's `Close`.
            GatewayRequest::ShutdownWrite { .. } => {}
            GatewayRequest::Close { stream } => {
                // The guest gave up on a connection still waiting for the handshake.
                self.held.retain(|open| open.stream != stream);
                if let Some(entry) = self.streams.remove(&stream)
                    && !entry.server_closed
                {
                    self.send(&WispPacket::Close {
                        stream,
                        reason: CLOSE_VOLUNTARY,
                    });
                }
            }
        }
    }

    /// Sends CONNECT and tells the gateway the stream is open (see the module comment).
    fn open(&mut self, open: HeldOpen, gateway: &mut Gateway) {
        self.send(&WispPacket::Connect {
            stream: open.stream,
            kind: open.kind,
            port: open.target.port,
            host: Ipv4Addr::from(open.target.address).to_string(),
        });
        self.streams.insert(
            open.stream,
            WispStream {
                kind: open.kind,
                credit: self.initial_credit,
                server_closed: false,
            },
        );
        gateway.opened(open.stream);
    }

    /// Moves each TCP stream's waiting guest bytes into DATA packets of up to `WISP_DATA_MAX`, one
    /// credit each. `to_remote` hands out one contiguous piece at a time, so a packet may join
    /// several to use its credit fully.
    fn send_data(&mut self, gateway: &mut Gateway) {
        if self.state != WispState::Ready {
            return;
        }
        for (&stream, entry) in &mut self.streams {
            if entry.kind != StreamKind::Tcp {
                continue;
            }
            if entry.server_closed {
                // The server is gone, so these bytes have nowhere to go. Left in place they would
                // keep the guest's FIN from ever settling and pin the flow.
                loop {
                    let waiting = gateway.to_remote(stream).len();
                    if waiting == 0 {
                        break;
                    }
                    gateway.consumed(stream, waiting);
                }
                continue;
            }
            while entry.credit > 0 {
                let mut payload = Vec::new();
                while payload.len() < WISP_DATA_MAX {
                    let waiting = gateway.to_remote(stream);
                    if waiting.is_empty() {
                        break;
                    }
                    let take = waiting.len().min(WISP_DATA_MAX - payload.len());
                    payload.extend_from_slice(&waiting[..take]);
                    gateway.consumed(stream, take);
                }
                if payload.is_empty() {
                    break;
                }
                entry.credit -= 1;
                self.outgoing
                    .push(encode(&WispPacket::Data { stream, payload }));
            }
        }
    }

    /// Resets every stream the server still has. Ones it already closed keep their record: the
    /// guest may still be reading the data before their FIN, and the gateway's `Close` for them
    /// must not reach the next connection as a CLOSE.
    fn reset_streams(&mut self, gateway: &mut Gateway) {
        let live: Vec<StreamId> = self
            .streams
            .iter()
            .filter(|(_, entry)| !entry.server_closed)
            .map(|(&stream, _)| stream)
            .collect();
        for stream in live {
            self.streams.remove(&stream);
            gateway.reset(stream);
        }
    }

    /// Open on the server's side too, so its packets apply.
    fn is_live(&self, stream: StreamId) -> bool {
        self.streams
            .get(&stream)
            .is_some_and(|entry| !entry.server_closed)
    }

    /// Nothing goes out before the handshake or after the connection is lost.
    fn send(&mut self, packet: &WispPacket) {
        if self.state == WispState::Ready {
            self.outgoing.push(encode(packet));
        }
    }
}
