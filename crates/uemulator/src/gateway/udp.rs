//! UDP flows: one stream per (guest port, destination) that the driver backs with a host socket
//! or a Wisp stream. The gateway only keeps the table, queues datagrams until the driver says
//! the stream is open, and expires idle flows.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;

use super::dhcp::DHCP_SERVER_PORT;
use super::wire::{
    ICMP_CHECKSUM, ICMP_CODE_PORT_UNREACHABLE, ICMP_DEST_UNREACHABLE, ICMP_ERROR_HEADER_LEN,
    IP_PROTO_ICMP, IP_PROTO_UDP, Ipv4Packet, UDP_DST_PORT, UDP_HEADER_LEN, UDP_LENGTH,
    UDP_SRC_PORT, be_u16, internet_checksum, udp_segment,
};
use super::{
    DNS_IP, DNS_PORT, Endpoint, FlowKey, GATEWAY_IP, Gateway, GatewayPolicy, GatewayRequest,
    LOOPBACK_IP, StreamId, StreamKind,
};

/// A flow with no datagram in either direction for this long is closed.
pub(super) const UDP_IDLE_TIMEOUT_MS: u64 = 60_000;
/// Each flow can hold a host socket or a proxy stream, so the count is capped. Past the cap the
/// least recently used flow is closed; the guest sees that as a lost datagram, which UDP
/// applications already tolerate.
pub(super) const MAX_UDP_FLOWS: usize = 128;
/// Datagrams a flow holds while the driver opens its stream. Past this the newest are dropped,
/// as a full socket buffer would; UDP applications already tolerate that.
pub(super) const UDP_PENDING_LIMIT: usize = 64;

pub(super) struct UdpFlow {
    stream: StreamId,
    guest_ip: [u8; 4],
    last_activity_ms: u64,
    /// The driver has confirmed the stream; until then datagrams wait in `pending`.
    opened: bool,
    pending: Vec<Vec<u8>>,
    /// The first datagram's IPv4 header and first 8 payload bytes, which an ICMP error must
    /// quote (RFC 792) so the guest can find the socket it belongs to.
    quote: Vec<u8>,
}

pub(super) struct UdpFlows {
    pub(super) flows: BTreeMap<FlowKey, UdpFlow>,
    by_stream: BTreeMap<StreamId, FlowKey>,
    pub(super) limit: usize,
}

impl UdpFlows {
    pub(super) fn new() -> UdpFlows {
        UdpFlows {
            flows: BTreeMap::new(),
            by_stream: BTreeMap::new(),
            limit: MAX_UDP_FLOWS,
        }
    }

    pub(super) fn has_stream(&self, stream: StreamId) -> bool {
        self.by_stream.contains_key(&stream)
    }

    pub(super) fn open_streams(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.flows
            .values()
            .filter(|flow| flow.opened)
            .map(|flow| flow.stream)
    }

    fn remove(&mut self, key: &FlowKey) -> Option<UdpFlow> {
        let flow = self.flows.remove(key)?;
        self.by_stream.remove(&flow.stream);
        Some(flow)
    }

    fn flow_of_stream(&mut self, stream: StreamId) -> Option<(FlowKey, &mut UdpFlow)> {
        let key = *self.by_stream.get(&stream)?;
        self.flows.get_mut(&key).map(|flow| (key, flow))
    }
}

impl Gateway {
    /// Where traffic for `address:port` really goes. Replies still appear to come from the
    /// address the guest used.
    pub(super) fn destination(&self, address: [u8; 4], port: u16) -> Endpoint {
        match self.policy {
            GatewayPolicy::UserNet { dns } => {
                if address == DNS_IP && port == DNS_PORT {
                    dns
                } else if address == GATEWAY_IP {
                    // The gateway address stands for the host itself.
                    Endpoint {
                        address: LOOPBACK_IP,
                        port,
                    }
                } else {
                    Endpoint { address, port }
                }
            }
            GatewayPolicy::Segment { resolver } => {
                if address == GATEWAY_IP && port == DNS_PORT {
                    resolver
                } else {
                    Endpoint { address, port }
                }
            }
        }
    }

    pub(super) fn udp_in(&mut self, ip: &Ipv4Packet) {
        let datagram = ip.payload;
        if datagram.len() < UDP_HEADER_LEN {
            return;
        }
        let dport = be_u16(&datagram[UDP_DST_PORT]);
        let len = (be_u16(&datagram[UDP_LENGTH]) as usize).clamp(UDP_HEADER_LEN, datagram.len());
        let data = &datagram[UDP_HEADER_LEN..len];
        // Datagrams to the DHCP port are never NAT-ed.
        if dport == DHCP_SERVER_PORT {
            self.dhcp(data);
            return;
        }
        let key = FlowKey {
            guest_port: be_u16(&datagram[UDP_SRC_PORT]),
            remote_ip: ip.dst,
            remote_port: dport,
        };
        let now_ms = self.now_ms;
        if !self.udp.flows.contains_key(&key) {
            if self.udp.flows.len() >= self.udp.limit {
                let least_recent = self
                    .udp
                    .flows
                    .iter()
                    .min_by_key(|(_, flow)| flow.last_activity_ms)
                    .map(|(key, _)| *key);
                if let Some(evicted) = least_recent.and_then(|key| self.udp.remove(&key)) {
                    self.requests.push(GatewayRequest::Close {
                        stream: evicted.stream,
                    });
                }
            }
            let stream = self.allocate_stream();
            let target = self.destination(ip.dst, dport);
            self.requests.push(GatewayRequest::Open {
                stream,
                kind: StreamKind::Udp,
                target,
            });
            let mut quote = ip.header.to_vec();
            quote.extend_from_slice(&datagram[..UDP_HEADER_LEN]);
            self.udp.by_stream.insert(stream, key);
            self.udp.flows.insert(
                key,
                UdpFlow {
                    stream,
                    guest_ip: ip.src,
                    last_activity_ms: now_ms,
                    opened: false,
                    pending: Vec::new(),
                    quote,
                },
            );
        }
        let Some(flow) = self.udp.flows.get_mut(&key) else {
            return;
        };
        flow.last_activity_ms = now_ms;
        if flow.opened {
            self.requests.push(GatewayRequest::Datagram {
                stream: flow.stream,
                data: data.to_vec(),
            });
        } else if flow.pending.len() < UDP_PENDING_LIMIT {
            flow.pending.push(data.to_vec());
        }
    }

    pub(super) fn udp_opened(&mut self, stream: StreamId) {
        let Some((_, flow)) = self.udp.flow_of_stream(stream) else {
            return;
        };
        flow.opened = true;
        for data in core::mem::take(&mut flow.pending) {
            self.requests
                .push(GatewayRequest::Datagram { stream, data });
        }
    }

    /// The driver could not reach the target: the guest gets ICMP port unreachable from the
    /// address it sent to, quoting its datagram.
    pub(super) fn udp_failed(&mut self, stream: StreamId) {
        let Some(key) = self.udp.by_stream.get(&stream).copied() else {
            return;
        };
        let Some(flow) = self.udp.remove(&key) else {
            return;
        };
        let mut message = vec![
            ICMP_DEST_UNREACHABLE,
            ICMP_CODE_PORT_UNREACHABLE,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        debug_assert_eq!(message.len(), ICMP_ERROR_HEADER_LEN);
        message.extend_from_slice(&flow.quote);
        let checksum = internet_checksum(&message);
        message[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
        self.emit_ip(key.remote_ip, flow.guest_ip, IP_PROTO_ICMP, &message);
    }

    /// One datagram from the remote end.
    pub(super) fn udp_remote_data(&mut self, stream: StreamId, data: &[u8]) {
        let now_ms = self.now_ms;
        let Some((key, flow)) = self.udp.flow_of_stream(stream) else {
            return;
        };
        flow.last_activity_ms = now_ms;
        let guest_ip = flow.guest_ip;
        let datagram = udp_segment(
            key.remote_ip,
            guest_ip,
            key.remote_port,
            key.guest_port,
            data,
        );
        self.emit_ip(key.remote_ip, guest_ip, IP_PROTO_UDP, &datagram);
    }

    /// A closed or reset UDP stream just ends the flow: there is no peer to tell and nothing the
    /// guest could be told.
    pub(super) fn udp_drop(&mut self, stream: StreamId) {
        if let Some(key) = self.udp.by_stream.get(&stream).copied() {
            self.udp.remove(&key);
        }
    }

    pub(super) fn expire_udp(&mut self, now_ms: u64) {
        let expired: Vec<FlowKey> = self
            .udp
            .flows
            .iter()
            .filter(|(_, flow)| now_ms.saturating_sub(flow.last_activity_ms) >= UDP_IDLE_TIMEOUT_MS)
            .map(|(key, _)| *key)
            .collect();
        for key in expired {
            if let Some(flow) = self.udp.remove(&key) {
                self.requests.push(GatewayRequest::Close {
                    stream: flow.stream,
                });
            }
        }
    }
}
