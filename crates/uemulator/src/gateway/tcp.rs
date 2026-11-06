//! TCP is terminated here, not forwarded: each guest connection becomes a stream the driver
//! backs with a host socket or a Wisp stream. The virtual link never drops or reorders frames, so
//! there is no retransmission timer, no congestion control, no window scaling, no SACK or
//! timestamps, no urgent data and no TIME_WAIT. Out-of-order guest segments are dropped and
//! answered with a duplicate ACK. The SYN-ACK goes only once the driver has opened the stream, and
//! a stream that fails to open resets the guest's SYN.
//!
//! Because nothing checks a checksum and TCP is terminated here, the guest may hand over its TSO
//! segments (up to 64 KiB, checksum unfinished) as they are, and remote data goes back in segments
//! as large as the guest's window when its driver takes them (`Gateway::set_large_segments`).
//! That removes the guest kernel's per-1500-byte cost, which bounded bulk TCP before.
//!
//! Sequence variable names follow RFC 793: `snd_*` is our send direction towards the guest,
//! `rcv_*` what we have received from it.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::{BTreeMap, VecDeque};

use super::wire::{
    ACK, FIN, IPV4_PAYLOAD_LEN_MAX, Ipv4Packet, PSH, RST, SYN, TCP_HEADER_LEN, TcpChecksum,
    TcpHeader, TcpSegment, negotiated_mss, seq_before, tcp_frame,
};
use super::{
    FlowKey, GATEWAY_MAC, Gateway, GatewayRequest, GuestFrame, OUT_QUEUE_LIMIT, StreamId,
    StreamKind,
};

/// Largest window we can advertise without window scaling (16-bit field). The advertised window
/// shrinks by the guest bytes the driver has not taken yet, which throttles the guest.
pub(super) const ADVERTISED_WINDOW_MAX: usize = 65535;
/// Most TCP payload one large segment to the guest carries: what fits an IPv4 packet behind a
/// 20-byte TCP header.
pub(super) const LARGE_SEGMENT_DATA_MAX: usize = IPV4_PAYLOAD_LEN_MAX - TCP_HEADER_LEN;
/// Step between initial sequence numbers of successive connections. Only needs to keep them
/// distinct; this is not a security boundary.
pub(super) const ISN_STEP: u32 = 0x10000;

/// Window to advertise given how many guest bytes still wait for the driver.
pub(super) fn advertised_window(buffered: usize) -> u16 {
    ADVERTISED_WINDOW_MAX.saturating_sub(buffered) as u16
}

/// How far the window must have opened since the last advertised one before `poll` sends an
/// update on its own: the receiver side of silly window avoidance (RFC 1122 section 4.2.3.3).
fn window_update_threshold(mss: usize) -> usize {
    mss.min(ADVERTISED_WINDOW_MAX / 2)
}

pub(super) struct TcpFlow {
    pub(super) stream: StreamId,
    pub(super) guest_ip: [u8; 4],
    pub(super) mss: usize,
    /// Next sequence number expected from the guest.
    pub(super) rcv_nxt: u32,
    /// Next sequence number we will send; a SYN or FIN consumes one.
    pub(super) snd_nxt: u32,
    /// Oldest sequence number the guest has not acknowledged yet.
    pub(super) snd_una: u32,
    /// Window the guest last advertised. Never scaled, because our SYN-ACK has no window scale
    /// option and the guest therefore cannot use one either.
    pub(super) guest_window: u32,
    /// Guest data the driver has not taken yet (`Gateway::to_remote`, `consumed`).
    pub(super) to_remote: VecDeque<u8>,
    /// Remote data the guest's window has not let through yet. Unbounded: a Wisp server cannot be
    /// held back, and its bytes may never be dropped.
    pub(super) from_remote: VecDeque<u8>,
    /// The driver opened the stream and the SYN-ACK went out.
    pub(super) opened: bool,
    pub(super) guest_fin: bool,
    /// `ShutdownWrite` was asked for: the guest's FIN came and the driver took everything before
    /// it.
    pub(super) write_shutdown: bool,
    /// The remote end sends nothing more; our FIN follows the data still in `from_remote`.
    pub(super) remote_closed: bool,
    pub(super) fin_sent: bool,
    /// Guest data arrived and its ACK waits for the next `poll`, so it can advertise the window
    /// left after the driver took the data. Acknowledging at once would advertise a window shrunk
    /// by bytes the driver is about to take, and a segment that fills the window would be answered
    /// with a zero one.
    pub(super) ack_due: bool,
    /// Window in the last segment sent to the guest, so `poll` knows when the driver taking
    /// `to_remote` opened it enough to be worth a window update.
    pub(super) window_sent: u16,
}

impl TcpFlow {
    pub(super) fn new(
        stream: StreamId,
        guest_ip: [u8; 4],
        mss: usize,
        rcv_nxt: u32,
        isn: u32,
        guest_window: u32,
    ) -> TcpFlow {
        TcpFlow {
            stream,
            guest_ip,
            mss,
            rcv_nxt,
            snd_nxt: isn,
            snd_una: isn,
            guest_window,
            to_remote: VecDeque::new(),
            from_remote: VecDeque::new(),
            opened: false,
            guest_fin: false,
            write_shutdown: false,
            remote_closed: false,
            fin_sent: false,
            ack_due: false,
            window_sent: 0,
        }
    }

    pub(super) fn header(
        &self,
        key: &FlowKey,
        flags: u8,
        seq: u32,
        advertise_mss: bool,
    ) -> TcpHeader {
        TcpHeader {
            src: key.remote_ip,
            dst: self.guest_ip,
            sport: key.remote_port,
            dport: key.guest_port,
            seq,
            ack: self.rcv_nxt,
            flags,
            window: advertised_window(self.to_remote.len()),
            advertise_mss,
        }
    }

    /// Remote bytes the guest's window has room for beyond what is in flight and buffered.
    pub(super) fn room(&self) -> usize {
        if self.remote_closed {
            return 0;
        }
        let in_flight = self.snd_nxt.wrapping_sub(self.snd_una);
        (self.guest_window.saturating_sub(in_flight) as usize)
            .saturating_sub(self.from_remote.len())
    }
}

pub(super) struct TcpFlows {
    pub(super) flows: BTreeMap<FlowKey, TcpFlow>,
    by_stream: BTreeMap<StreamId, FlowKey>,
    /// First flow served by the previous `poll`, so the next one starts at its successor and the
    /// per-poll queue budget is shared round-robin instead of always going to the same flows.
    cursor: Option<FlowKey>,
}

impl TcpFlows {
    pub(super) fn new() -> TcpFlows {
        TcpFlows {
            flows: BTreeMap::new(),
            by_stream: BTreeMap::new(),
            cursor: None,
        }
    }

    pub(super) fn has_stream(&self, stream: StreamId) -> bool {
        self.by_stream.contains_key(&stream)
    }

    pub(super) fn insert(&mut self, key: FlowKey, flow: TcpFlow) {
        self.by_stream.insert(flow.stream, key);
        self.flows.insert(key, flow);
    }

    pub(super) fn remove(&mut self, key: &FlowKey) -> Option<TcpFlow> {
        let flow = self.flows.remove(key)?;
        self.by_stream.remove(&flow.stream);
        Some(flow)
    }

    fn key_of(&self, stream: StreamId) -> Option<FlowKey> {
        self.by_stream.get(&stream).copied()
    }

    fn of_stream(&self, stream: StreamId) -> Option<&TcpFlow> {
        self.flows.get(self.by_stream.get(&stream)?)
    }

    fn of_stream_mut(&mut self, stream: StreamId) -> Option<&mut TcpFlow> {
        self.flows.get_mut(self.by_stream.get(&stream)?)
    }

    /// Flows in the order this poll serves them: sorted for a stable cycle, starting one past
    /// where the previous poll started. Each poll has a fixed budget of frames, so a fixed order
    /// would let the first flows starve the rest under sustained load.
    pub(super) fn service_order(&mut self) -> Vec<FlowKey> {
        let mut keys: Vec<FlowKey> = self.flows.keys().copied().collect();
        let start = self
            .cursor
            .map_or(0, |previous| keys.partition_point(|key| *key <= previous));
        let start = if start < keys.len() { start } else { 0 };
        keys.rotate_left(start);
        self.cursor = keys.first().copied();
        keys
    }
}

/// Queues one TCP segment for the guest. A free function over the fields it needs, so `poll` can
/// call it while it holds a flow.
fn queue_segment(
    out: &mut VecDeque<GuestFrame>,
    ip_id: &mut u16,
    guest_mac: [u8; 6],
    header: &TcpHeader,
    data: &[&[u8]],
    segment_size: Option<u16>,
) {
    *ip_id = ip_id.wrapping_add(1);
    let checksum = match segment_size {
        Some(_) => TcpChecksum::PseudoHeaderOnly,
        None => TcpChecksum::Complete,
    };
    out.push_back(GuestFrame {
        bytes: tcp_frame(guest_mac, GATEWAY_MAC, *ip_id, header, data, checksum),
        segment_size,
    });
}

impl Gateway {
    /// The first contiguous run of guest bytes waiting for the remote end of `stream` (empty when
    /// none, or for an unknown stream). Report what was taken with `consumed`.
    pub fn to_remote(&self, stream: StreamId) -> &[u8] {
        if let Some(queries) = self.dns_tcp_to_remote(stream) {
            return queries;
        }
        self.tcp
            .of_stream(stream)
            .map_or(&[], |flow| flow.to_remote.as_slices().0)
    }

    /// The driver took `bytes` from the front of `to_remote`. The window this opens is announced
    /// at the next `poll`.
    pub fn consumed(&mut self, stream: StreamId, bytes: usize) {
        if self.dns_tcp_consumed(stream, bytes) {
            return;
        }
        if let Some(flow) = self.tcp.of_stream_mut(stream) {
            let bytes = bytes.min(flow.to_remote.len());
            flow.to_remote.drain(..bytes);
        }
    }

    /// How many remote bytes `stream` can take now: the guest's window less what is in flight and
    /// already buffered; 0 for unknown streams and once the remote end closed. `remote_data`
    /// beyond it is still kept, but a driver that can hold data back (a socket) should read no
    /// more than this.
    pub fn room(&self, stream: StreamId) -> usize {
        if let Some(room) = self.dns_tcp_room(stream) {
            return room;
        }
        self.tcp.of_stream(stream).map_or(0, TcpFlow::room)
    }

    fn emit_tcp(&mut self, header: &TcpHeader) {
        queue_segment(
            &mut self.out,
            &mut self.ip_id,
            self.guest_mac,
            header,
            &[],
            None,
        );
    }

    /// Reset for a segment that matches no flow (RFC 793 section 3.4, "reset generation"). Never
    /// answers a RST.
    fn send_reset(&mut self, ip: &Ipv4Packet, segment: &TcpSegment) {
        if segment.flags & RST != 0 {
            return;
        }
        let (seq, ack, flags) = if segment.flags & ACK != 0 {
            (segment.ack, 0, RST)
        } else {
            // SYN and FIN each occupy one sequence number.
            let seq_len = segment.data.len() as u32
                + (segment.flags & SYN != 0) as u32
                + (segment.flags & FIN != 0) as u32;
            (0, segment.seq.wrapping_add(seq_len), RST | ACK)
        };
        self.emit_tcp(&TcpHeader {
            src: ip.dst,
            dst: ip.src,
            sport: segment.dport,
            dport: segment.sport,
            seq,
            ack,
            flags,
            window: 0,
            advertise_mss: false,
        });
    }

    pub(super) fn tcp_in(&mut self, ip: &Ipv4Packet) {
        let Some(segment) = TcpSegment::parse(ip.payload) else {
            return;
        };
        let key = FlowKey {
            guest_port: segment.sport,
            remote_ip: ip.dst,
            remote_port: segment.dport,
        };
        let Some(flow) = self.tcp.flows.get_mut(&key) else {
            if segment.flags & SYN == 0 || segment.flags & ACK != 0 {
                self.send_reset(ip, &segment);
                return;
            }
            self.isn = self.isn.wrapping_add(ISN_STEP);
            let stream = self.allocate_stream();
            let target = self.destination(ip.dst, segment.dport);
            self.requests.push(GatewayRequest::Open {
                stream,
                kind: StreamKind::Tcp,
                target,
            });
            let flow = TcpFlow::new(
                stream,
                ip.src,
                negotiated_mss(segment.options),
                segment.seq.wrapping_add(1),
                self.isn,
                segment.window,
            );
            self.tcp.insert(key, flow);
            return;
        };
        if segment.flags & RST != 0 {
            let stream = flow.stream;
            self.tcp.remove(&key);
            self.requests.push(GatewayRequest::Close { stream });
            return;
        }
        // A repeated SYN while the driver opens the stream is a guest retransmit; the SYN-ACK
        // follows once the stream is open.
        if !flow.opened || segment.flags & SYN != 0 {
            return;
        }
        if segment.flags & ACK != 0 {
            if !seq_before(segment.ack, flow.snd_una) && !seq_before(flow.snd_nxt, segment.ack) {
                flow.snd_una = segment.ack;
            }
            flow.guest_window = segment.window;
        }
        let mut ack_now = false;
        if !segment.data.is_empty() || segment.flags & FIN != 0 {
            // Bytes at the start of the segment we already have (guest retransmit). A segment
            // starting beyond `rcv_nxt` leaves a gap and is dropped, so `rcv_nxt` in the ACK
            // asks for the missing data again.
            let already_received = flow.rcv_nxt.wrapping_sub(segment.seq);
            if (already_received as i32) >= 0 && (already_received as usize) <= segment.data.len() {
                let unseen = &segment.data[already_received as usize..];
                // A guest that ignores our window gets only what fits it, so a misbehaving one
                // cannot grow `to_remote` without bound; the ACK asks for the rest.
                let fits = ADVERTISED_WINDOW_MAX.saturating_sub(flow.to_remote.len());
                let fresh = &unseen[..unseen.len().min(fits)];
                flow.to_remote.extend(fresh);
                flow.rcv_nxt = flow.rcv_nxt.wrapping_add(fresh.len() as u32);
                if segment.flags & FIN != 0 && fresh.len() == unseen.len() && !flow.guest_fin {
                    flow.guest_fin = true;
                    flow.rcv_nxt = flow.rcv_nxt.wrapping_add(1);
                }
                if fresh.is_empty() {
                    ack_now = true;
                } else {
                    flow.ack_due = true;
                }
            } else {
                ack_now = true;
            }
        }
        if ack_now {
            let header = flow.header(&key, ACK, flow.snd_nxt, false);
            flow.window_sent = header.window;
            flow.ack_due = false;
            queue_segment(
                &mut self.out,
                &mut self.ip_id,
                self.guest_mac,
                &header,
                &[],
                None,
            );
        }
        self.settle_tcp(key);
    }

    /// Asks for `ShutdownWrite` once the guest's FIN came and the driver took everything before
    /// it, and ends the flow with `Close` once that is done, the remote end closed and the guest
    /// acknowledged our FIN: nothing is left to wait for, and a flow kept past that would pin a
    /// stream.
    fn settle_tcp(&mut self, key: FlowKey) {
        let Some(flow) = self.tcp.flows.get_mut(&key) else {
            return;
        };
        if flow.guest_fin && flow.to_remote.is_empty() && !flow.write_shutdown {
            flow.write_shutdown = true;
            self.requests.push(GatewayRequest::ShutdownWrite {
                stream: flow.stream,
            });
        }
        if !(flow.write_shutdown && flow.fin_sent && flow.snd_una == flow.snd_nxt) {
            return;
        }
        if flow.ack_due {
            let header = flow.header(&key, ACK, flow.snd_nxt, false);
            queue_segment(
                &mut self.out,
                &mut self.ip_id,
                self.guest_mac,
                &header,
                &[],
                None,
            );
        }
        let stream = flow.stream;
        self.tcp.remove(&key);
        self.requests.push(GatewayRequest::Close { stream });
    }

    pub(super) fn tcp_opened(&mut self, stream: StreamId) {
        let Some(key) = self.tcp.key_of(stream) else {
            return;
        };
        let Some(flow) = self.tcp.flows.get_mut(&key) else {
            return;
        };
        if flow.opened {
            return;
        }
        flow.opened = true;
        let header = flow.header(&key, SYN | ACK, flow.snd_nxt, true);
        flow.window_sent = header.window;
        flow.snd_nxt = flow.snd_nxt.wrapping_add(1);
        self.emit_tcp(&header);
    }

    /// Before the open this resets the guest's SYN, the way a refused connection does; after it,
    /// the stream broke (`tcp_reset`).
    pub(super) fn tcp_failed(&mut self, stream: StreamId) {
        let Some(key) = self.tcp.key_of(stream) else {
            return;
        };
        let Some(flow) = self.tcp.remove(&key) else {
            return;
        };
        if flow.opened {
            self.reset_guest(&key, &flow);
            return;
        }
        let mut header = flow.header(&key, RST | ACK, 0, false);
        header.window = 0;
        self.emit_tcp(&header);
    }

    pub(super) fn tcp_reset(&mut self, stream: StreamId) {
        let Some(key) = self.tcp.key_of(stream) else {
            return;
        };
        if let Some(flow) = self.tcp.remove(&key) {
            self.reset_guest(&key, &flow);
        }
    }

    fn reset_guest(&mut self, key: &FlowKey, flow: &TcpFlow) {
        self.emit_tcp(&flow.header(key, RST | ACK, flow.snd_nxt, false));
    }

    pub(super) fn tcp_remote_data(&mut self, stream: StreamId, data: &[u8]) {
        if let Some(flow) = self.tcp.of_stream_mut(stream)
            && !flow.remote_closed
        {
            flow.from_remote.extend(data);
        }
    }

    pub(super) fn tcp_remote_closed(&mut self, stream: StreamId) {
        if let Some(flow) = self.tcp.of_stream_mut(stream) {
            flow.remote_closed = true;
        }
    }

    /// Moves buffered remote data to the guest within its window and the frame budget, sends our
    /// FIN after the last of it, owed ACKs and window updates, and settles each flow.
    pub(super) fn poll_tcp(&mut self) {
        let mut budget = OUT_QUEUE_LIMIT.saturating_sub(self.out.len());
        let large_segments = self.large_segments;
        let guest_mac = self.guest_mac;
        for key in self.tcp.service_order() {
            let Some(flow) = self.tcp.flows.get_mut(&key) else {
                continue;
            };
            if !flow.opened {
                continue;
            }
            // The window and ack are fixed for this pass: nothing below touches `to_remote` or
            // `rcv_nxt`.
            let template = flow.header(&key, 0, 0, false);
            let segment_limit = if large_segments {
                LARGE_SEGMENT_DATA_MAX
            } else {
                flow.mss
            };
            let mut sent = false;
            while !flow.fin_sent && budget > 0 {
                let in_flight = flow.snd_nxt.wrapping_sub(flow.snd_una);
                let window_left = flow.guest_window.saturating_sub(in_flight) as usize;
                if window_left == 0 {
                    break;
                }
                if flow.from_remote.is_empty() {
                    if flow.remote_closed {
                        let header = TcpHeader {
                            flags: FIN | ACK,
                            seq: flow.snd_nxt,
                            ..template
                        };
                        flow.snd_nxt = flow.snd_nxt.wrapping_add(1);
                        flow.fin_sent = true;
                        queue_segment(
                            &mut self.out,
                            &mut self.ip_id,
                            guest_mac,
                            &header,
                            &[],
                            None,
                        );
                        budget -= 1;
                        sent = true;
                    }
                    break;
                }
                let len = window_left.min(segment_limit).min(flow.from_remote.len());
                let header = TcpHeader {
                    flags: PSH | ACK,
                    seq: flow.snd_nxt,
                    ..template
                };
                let (front, back) = flow.from_remote.as_slices();
                let data: [&[u8]; 2] = if len <= front.len() {
                    [&front[..len], &[]]
                } else {
                    [front, &back[..len - front.len()]]
                };
                let segment_size = large_segments.then_some(flow.mss as u16);
                queue_segment(
                    &mut self.out,
                    &mut self.ip_id,
                    guest_mac,
                    &header,
                    &data,
                    segment_size,
                );
                flow.from_remote.drain(..len);
                flow.snd_nxt = flow.snd_nxt.wrapping_add(len as u32);
                budget -= 1;
                sent = true;
            }
            // Every segment above carries the current window and ack; without one, an owed ACK
            // goes out (whatever the budget: it was never budgeted when sent at once), or a
            // window update when the driver taking `to_remote` opened the window enough.
            let window = template.window;
            let window_opened =
                window as usize >= flow.window_sent as usize + window_update_threshold(flow.mss);
            if sent {
                flow.window_sent = window;
                flow.ack_due = false;
            } else if flow.ack_due || (budget > 0 && window_opened) {
                if !flow.ack_due {
                    budget -= 1;
                }
                flow.window_sent = window;
                flow.ack_due = false;
                let header = TcpHeader {
                    flags: ACK,
                    seq: flow.snd_nxt,
                    ..template
                };
                queue_segment(
                    &mut self.out,
                    &mut self.ip_id,
                    guest_mac,
                    &header,
                    &[],
                    None,
                );
            }
            self.settle_tcp(key);
        }
    }
}
