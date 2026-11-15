//! DNS over TCP (RFC 7766) for the `Segment` policy. Wisp servers often refuse UDP streams, and
//! the guest's resolver speaks UDP to 10.0.2.2:53, so once a UDP stream to the resolver has
//! failed the gateway carries every query over one TCP stream of its own, with the two-byte
//! length prefix, and turns each answer back into a datagram. The choice is kept for the
//! gateway's lifetime unless TCP also fails, in which case the next query tries UDP again.
//!
//! The stream has no guest TCP flow behind it. The driver feeds it through the same calls as
//! any TCP stream (`opened`, `to_remote`, `consumed`, `remote_data`, ...), which dispatch here
//! by stream id.

#[cfg(freestanding)]
use crate::prelude::*;

use alloc::collections::BTreeSet;

use super::wire::{IP_PROTO_UDP, UDP_PAYLOAD_MAX, be_u16, udp_segment};
use super::{DNS_PORT, GATEWAY_IP, Gateway, GatewayPolicy, GatewayRequest, StreamId, StreamKind};

/// Queries waiting for an answer, sent or not. Past this the newest are dropped; the guest's
/// resolver retries.
pub(super) const DNS_TCP_PENDING_LIMIT: usize = 64;
/// With nothing outstanding for this long the stream is closed, so an idle guest holds no proxy
/// stream.
pub(super) const DNS_TCP_IDLE_TIMEOUT_MS: u64 = 60_000;
/// Bytes waiting in `outgoing`. Expiry does not trim it, so without this limit a stream that
/// never opens, or a driver that stops consuming, would grow it by a full window of queries every
/// `DNS_TCP_QUERY_TIMEOUT_MS`. A query that would pass it is dropped like one past the pending limit.
pub(super) const DNS_TCP_OUTGOING_LIMIT: usize = 256 * 1024;
/// A query unanswered this long is dropped (the guest's resolver retries). Without it a resolver
/// that accepts TCP and never answers would fill `DNS_TCP_PENDING_LIMIT` for good and keep the
/// stream from ever idling out. The clock starts when the query is queued, not when it is sent.
pub(super) const DNS_TCP_QUERY_TIMEOUT_MS: u64 = 10_000;
/// What `room` reports: a fixed amount, since answers become datagrams at once and nothing waits
/// on a guest window.
const DNS_TCP_ROOM: usize = 64 * 1024;
const LENGTH_PREFIX_LEN: usize = 2;
/// The DNS ID is the first field of every message.
pub(super) const DNS_ID_LEN: usize = 2;

pub(super) struct DnsQuery {
    guest_ip: [u8; 4],
    guest_port: u16,
    /// Set from the query's own bytes when it is queued.
    id: u16,
    upstream_id: u16,
    asked_ms: u64,
    /// The datagram's IPv4 header and first 8 payload bytes, which an ICMP error must quote.
    quote: Vec<u8>,
}

impl DnsQuery {
    pub(super) fn new(guest_ip: [u8; 4], guest_port: u16, quote: Vec<u8>) -> DnsQuery {
        DnsQuery {
            guest_ip,
            guest_port,
            id: 0,
            upstream_id: 0,
            asked_ms: 0,
            quote,
        }
    }
}

pub(super) struct DnsTcp {
    /// A UDP stream to the resolver failed once: every query goes over TCP from now on.
    pub(super) fallen_back: bool,
    stream: Option<StreamId>,
    opened: bool,
    /// Length-prefixed queries the driver has not taken.
    outgoing: Vec<u8>,
    /// Resolver bytes that do not yet make a whole message.
    incoming: Vec<u8>,
    outstanding: Vec<DnsQuery>,
    /// Never reuse an ID on this stream: even an expired query can receive a late reply.
    used_ids: BTreeSet<u16>,
    next_id: u16,
    last_activity_ms: u64,
}

impl DnsTcp {
    pub(super) fn new() -> DnsTcp {
        DnsTcp {
            fallen_back: false,
            stream: None,
            opened: false,
            outgoing: Vec::new(),
            incoming: Vec::new(),
            outstanding: Vec::new(),
            used_ids: BTreeSet::new(),
            next_id: 0,
            last_activity_ms: 0,
        }
    }

    pub(super) fn has_stream(&self, stream: StreamId) -> bool {
        self.stream == Some(stream)
    }

    pub(super) fn open_stream(&self) -> Option<StreamId> {
        self.stream.filter(|_| self.opened)
    }

    pub(super) fn wants_write(&self) -> bool {
        self.opened && !self.outgoing.is_empty()
    }

    /// Forgets the stream and everything riding on it.
    fn reset(&mut self) -> Vec<DnsQuery> {
        self.stream = None;
        self.opened = false;
        self.outgoing.clear();
        self.incoming.clear();
        self.used_ids.clear();
        self.next_id = 0;
        core::mem::take(&mut self.outstanding)
    }
}

impl Gateway {
    /// Carries one query over the TCP stream, opening it when there is none.
    pub(super) fn dns_tcp_query(&mut self, query: DnsQuery, data: &[u8]) {
        // Without an ID there is nothing to match an answer by; no resolver would take it either.
        if data.len() < DNS_ID_LEN
            || data.len() > usize::from(u16::MAX)
            || self.dns_tcp.outstanding.len() >= DNS_TCP_PENDING_LIMIT
            || self.dns_tcp.outgoing.len() + LENGTH_PREFIX_LEN + data.len() > DNS_TCP_OUTGOING_LIMIT
        {
            return;
        }
        // Once every ID has been used, retire the stream after its live queries finish.
        // Late replies on the old stream cannot then be mistaken for new queries.
        if self.dns_tcp.used_ids.len() == usize::from(u16::MAX) + 1 {
            if !self.dns_tcp.outstanding.is_empty() {
                return;
            }
            if let Some(stream) = self.dns_tcp.stream {
                self.requests.push(GatewayRequest::Close { stream });
            }
            self.dns_tcp.reset();
        }
        if self.dns_tcp.stream.is_none() {
            let stream = self.allocate_stream();
            self.dns_tcp.stream = Some(stream);
            if let GatewayPolicy::Segment { resolver } = self.policy {
                self.requests.push(GatewayRequest::Open {
                    stream,
                    kind: StreamKind::Tcp,
                    target: resolver,
                });
            }
        }
        let dns = &mut self.dns_tcp;
        let id = be_u16(&data[..DNS_ID_LEN]);
        let mut upstream_id = id;
        if !dns.used_ids.insert(upstream_id) {
            while !dns.used_ids.insert(dns.next_id) {
                dns.next_id = dns.next_id.wrapping_add(1);
            }
            upstream_id = dns.next_id;
            dns.next_id = dns.next_id.wrapping_add(1);
        }
        dns.last_activity_ms = self.now_ms;
        dns.outgoing
            .extend_from_slice(&(data.len() as u16).to_be_bytes());
        dns.outgoing.extend_from_slice(&upstream_id.to_be_bytes());
        dns.outgoing.extend_from_slice(&data[DNS_ID_LEN..]);
        dns.outstanding.push(DnsQuery {
            id,
            upstream_id,
            asked_ms: self.now_ms,
            ..query
        });
    }

    pub(super) fn dns_tcp_opened(&mut self, stream: StreamId) {
        if self.dns_tcp.has_stream(stream) {
            self.dns_tcp.opened = true;
            self.dns_tcp.last_activity_ms = self.now_ms;
        }
    }

    /// No uplink at all: every query waiting gets port unreachable, so the guest fails at once
    /// instead of timing out.
    pub(super) fn dns_tcp_failed(&mut self, stream: StreamId) {
        if !self.dns_tcp.has_stream(stream) {
            return;
        }
        // Not a server refusing UDP but no uplink (not connected yet, or between reconnects):
        // the next query tries UDP again instead of staying on TCP for good.
        self.dns_tcp.fallen_back = false;
        for query in self.dns_tcp.reset() {
            self.emit_port_unreachable(GATEWAY_IP, query.guest_ip, &query.quote);
        }
    }

    pub(super) fn dns_tcp_remote_closed(&mut self, stream: StreamId) {
        if self.dns_tcp.has_stream(stream) {
            // The queries are dropped, not refused: the resolver may simply have hit its own idle
            // limit, and the guest's retry reopens the stream.
            self.dns_tcp.reset();
            self.requests.push(GatewayRequest::Close { stream });
        }
    }

    pub(super) fn dns_tcp_reset(&mut self, stream: StreamId) {
        if self.dns_tcp.has_stream(stream) {
            self.dns_tcp.reset();
        }
    }

    pub(super) fn dns_tcp_remote_data(&mut self, stream: StreamId, data: &[u8]) {
        if !self.dns_tcp.has_stream(stream) {
            return;
        }
        self.dns_tcp.last_activity_ms = self.now_ms;
        self.dns_tcp.incoming.extend_from_slice(data);
        // `incoming` never keeps more than one unfinished message: whole ones are taken here.
        loop {
            let incoming = &self.dns_tcp.incoming;
            if incoming.len() < LENGTH_PREFIX_LEN {
                return;
            }
            let length = usize::from(be_u16(&incoming[..LENGTH_PREFIX_LEN]));
            let end = LENGTH_PREFIX_LEN + length;
            if incoming.len() < end {
                return;
            }
            let mut message: Vec<u8> = self
                .dns_tcp
                .incoming
                .drain(..end)
                .skip(LENGTH_PREFIX_LEN)
                .collect();
            self.dns_tcp_answer(&mut message);
        }
    }

    /// Turns one answer into a datagram for the query it matches; unmatched ones are dropped.
    fn dns_tcp_answer(&mut self, message: &mut [u8]) {
        // A TCP answer may be up to 65 535 bytes, more than a datagram carries; the guest's
        // resolver retries, as it would over UDP with the answer truncated away.
        if message.len() < DNS_ID_LEN || message.len() > UDP_PAYLOAD_MAX {
            return;
        }
        let id = be_u16(&message[..DNS_ID_LEN]);
        let Some(position) = self
            .dns_tcp
            .outstanding
            .iter()
            .position(|q| q.upstream_id == id)
        else {
            return;
        };
        let query = self.dns_tcp.outstanding.remove(position);
        message[..DNS_ID_LEN].copy_from_slice(&query.id.to_be_bytes());
        let datagram = udp_segment(
            GATEWAY_IP,
            query.guest_ip,
            DNS_PORT,
            query.guest_port,
            message,
        );
        self.emit_ip(GATEWAY_IP, query.guest_ip, IP_PROTO_UDP, &datagram);
    }

    /// Drops queries past `DNS_TCP_QUERY_TIMEOUT_MS`, then closes a stream nothing has waited on for
    /// `DNS_TCP_IDLE_TIMEOUT_MS`.
    pub(super) fn poll_dns_tcp(&mut self, now_ms: u64) {
        self.dns_tcp
            .outstanding
            .retain(|query| now_ms.saturating_sub(query.asked_ms) < DNS_TCP_QUERY_TIMEOUT_MS);
        let dns = &self.dns_tcp;
        let Some(stream) = dns.stream else {
            return;
        };
        if dns.opened
            && dns.outstanding.is_empty()
            && now_ms.saturating_sub(dns.last_activity_ms) >= DNS_TCP_IDLE_TIMEOUT_MS
        {
            self.dns_tcp.reset();
            self.requests.push(GatewayRequest::Close { stream });
        }
    }

    pub(super) fn dns_tcp_to_remote(&self, stream: StreamId) -> Option<&[u8]> {
        self.dns_tcp.has_stream(stream).then(|| {
            if self.dns_tcp.opened {
                &self.dns_tcp.outgoing[..]
            } else {
                &[]
            }
        })
    }

    pub(super) fn dns_tcp_consumed(&mut self, stream: StreamId, bytes: usize) -> bool {
        if !self.dns_tcp.has_stream(stream) {
            return false;
        }
        let bytes = bytes.min(self.dns_tcp.outgoing.len());
        self.dns_tcp.outgoing.drain(..bytes);
        true
    }

    pub(super) fn dns_tcp_room(&self, stream: StreamId) -> Option<usize> {
        self.dns_tcp.has_stream(stream).then_some(DNS_TCP_ROOM)
    }
}
