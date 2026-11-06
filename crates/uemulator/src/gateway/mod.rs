//! The guest's way out of an Ethernet segment, without I/O of its own: a protocol engine that
//! takes the guest's frames and queues frames for it, and leaves sockets (host TCP/UDP, or a Wisp
//! WebSocket in the browser) to whoever drives it. Time comes in as `now_ms`; the library has no
//! clock.
//!
//! The guest sees 10.0.2.0/24 with the gateway at 10.0.2.2 and DNS at 10.0.2.3 (`UserNet`), or
//! shares a segment whose own server owns 10.0.2.2 (`Segment`, where the gateway terminates only
//! the traffic leaving the subnet and DNS at 10.0.2.2:53).
//!
//! Echo requests to any destination are answered locally and never leave the host: the guest
//! sees pings to 8.8.8.8 succeed even when the host is offline. That is deliberate (the gateway
//! stays usable offline and needs no raw sockets or privileges) and means ping proves only that
//! the virtual link works, not that a destination is reachable.
//!
//! DHCP serves the single address 10.0.2.15. A REQUEST for any other address gets a NAK; a
//! REQUEST naming another server gets no answer (RFC 2131 section 3.1: it declines our offer).

mod dhcp;
mod dns_tcp;
mod tcp;
#[cfg(test)]
mod tests;
mod udp;
mod wire;

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;

use dhcp::{DHCP_CLIENT_PORT, DHCP_SERVER_PORT, dhcp_reply, dhcp_reply_type, parse_dhcp_request};
use dns_tcp::DnsTcp;
use tcp::TcpFlows;
use udp::UdpFlows;
use wire::{
    ARP_ETHERNET_IPV4, ARP_OP_REPLY, ARP_OP_REQUEST, ARP_OPCODE, ARP_PACKET_LEN, ARP_SENDER_IP,
    ARP_SENDER_MAC, ARP_TARGET_IP, ETH_HEADER_LEN, ETH_SRC_MAC, ETH_TYPE, ETHERTYPE_ARP,
    ETHERTYPE_IPV4, ICMP_CHECKSUM, ICMP_ECHO_HEADER_LEN, ICMP_ECHO_REPLY, ICMP_ECHO_REQUEST,
    IP_PROTO_ICMP, IP_PROTO_TCP, IP_PROTO_UDP, Ipv4Packet, UDP_DST_PORT, be_u16, ethernet_frame,
    internet_checksum, ipv4_packet, ipv4_packets, parse_ipv4, udp_segment,
};

/// 52:55 followed by the gateway IP, the way QEMU's slirp derives its MAC.
pub(crate) const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
/// Also the DHCP server and default router.
pub(crate) const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];
const DNS_IP: [u8; 4] = [10, 0, 2, 3];
/// What the gateway address stands for under `UserNet`: the host itself.
const LOOPBACK_IP: [u8; 4] = [127, 0, 0, 1];
/// The only address DHCP hands out.
const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
const SUBNET_PREFIX: [u8; 3] = [10, 0, 2];
const SUBNET_MASK: [u8; 4] = [255, 255, 255, 0];
const SUBNET_BROADCAST: [u8; 4] = [10, 0, 2, 255];
const LIMITED_BROADCAST_IP: [u8; 4] = [255; 4];
const BROADCAST_MAC: [u8; 6] = [0xff; 6];
const DNS_PORT: u16 = 53;
/// QEMU's conventional default NIC MAC (52:54:00 is its locally administered prefix): where
/// frames go until the guest has sent one to learn its MAC from.
const DEFAULT_GUEST_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
/// Where the first IPv4 identification number comes from; any value does.
const FIRST_IP_ID: u16 = 1;
const FIRST_STREAM_ID: StreamId = 1;
/// `poll` stops moving remote data once this many frames are already queued for the guest, so a
/// fast remote end cannot grow the queue without bound while the guest is slow to take frames.
const OUT_QUEUE_LIMIT: usize = 512;

/// Flow identity from the guest's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FlowKey {
    guest_port: u16,
    remote_ip: [u8; 4],
    remote_port: u16,
}

/// An IPv4 address and port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub address: [u8; 4],
    pub port: u16,
}

/// Names one byte stream (a TCP connection or a UDP flow) between the gateway and its driver.
/// Allocated by the gateway: never 0, increasing, never reused while the stream is live.
pub type StreamId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    Tcp,
    Udp,
}

/// What the gateway asks of its driver; the driver answers with `Gateway::opened`, `failed`,
/// `remote_data`, `remote_closed` and `reset`. A TCP stream's bytes towards the remote end are
/// pulled with `Gateway::to_remote` and `consumed` rather than asked for here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GatewayRequest {
    /// Connect to `target` (already mapped: 10.0.2.2 is 127.0.0.1 under `UserNet`, DNS goes to
    /// the configured server).
    Open {
        stream: StreamId,
        kind: StreamKind,
        target: Endpoint,
    },
    /// One UDP datagram. Only sent once the stream is open.
    Datagram { stream: StreamId, data: Vec<u8> },
    /// The guest finished sending on a TCP stream and everything before its FIN was taken: shut
    /// down the write side and keep reading.
    ShutdownWrite { stream: StreamId },
    /// The stream is over (both TCP directions done, a guest RST, or an idle UDP flow): drop it.
    Close { stream: StreamId },
}

/// Which network the gateway is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatewayPolicy {
    /// The guest's whole network: proxy-ARP for the subnet except the guest, DHCP for 10.0.2.15,
    /// 10.0.2.2 is the host's loopback and `10.0.2.3:53` goes to `dns`.
    UserNet { dns: Endpoint },
    /// One machine on a shared segment whose server (`SegmentAgent`) does ARP and DHCP: no ARP,
    /// no DHCP, and only off-subnet destinations and `10.0.2.2:53` (to `resolver`) are ours.
    Segment { resolver: Endpoint },
}

/// What the driver should wait for on a stream (`Gateway::interest`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInterest {
    /// The gateway can take remote data now: a TCP stream has `room` and the guest's queue is
    /// under its limit; an open UDP stream has no window, so only the queue counts.
    pub read: bool,
    /// Guest bytes wait in `to_remote`.
    pub write: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatewayConfig {
    pub policy: GatewayPolicy,
    /// Where the first TCP initial sequence number comes from. Only needs to keep connections
    /// distinct; this is not a security boundary.
    pub isn_seed: u32,
}

/// A frame queued for the guest. `segment_size` marks a TCP segment whose checksum is only the
/// pseudo-header sum and whose payload may exceed that MSS; it goes out through
/// `NetHandle::send_tcp_segment`, everything else through `send_frame`.
pub struct GuestFrame {
    pub bytes: Vec<u8>,
    pub segment_size: Option<u16>,
}

impl GuestFrame {
    fn plain(bytes: Vec<u8>) -> GuestFrame {
        GuestFrame {
            bytes,
            segment_size: None,
        }
    }
}

pub struct Gateway {
    policy: GatewayPolicy,
    /// Learned from the source MAC of every guest frame; frames the gateway originates (replies,
    /// remote data) go to it.
    guest_mac: [u8; 6],
    /// Frames queued for the guest, oldest first.
    out: VecDeque<GuestFrame>,
    /// The guest takes TCP segments longer than the MSS with an unfinished checksum
    /// (`NetHandle::guest_takes_tcp_segments`), so remote data goes out in segments as large as
    /// its window. The driver can renegotiate at any time.
    large_segments: bool,
    ip_id: u16,
    /// Requests for the driver, oldest first.
    requests: Vec<GatewayRequest>,
    udp: UdpFlows,
    tcp: TcpFlows,
    /// `Segment`'s DNS over TCP, once UDP to the resolver has been refused.
    dns_tcp: DnsTcp,
    /// The next `StreamId` to try; shared by every kind of stream.
    next_stream: StreamId,
    /// The last time the driver told us (`from_guest`, `poll`), for events that carry none, such
    /// as `remote_data` refreshing a flow's idle timer.
    now_ms: u64,
    /// Initial sequence number of the most recent connection.
    isn: u32,
}

impl Gateway {
    pub fn new(config: GatewayConfig) -> Gateway {
        Gateway {
            policy: config.policy,
            guest_mac: DEFAULT_GUEST_MAC,
            out: VecDeque::new(),
            large_segments: false,
            ip_id: FIRST_IP_ID,
            requests: Vec::new(),
            udp: UdpFlows::new(),
            tcp: TcpFlows::new(),
            dns_tcp: DnsTcp::new(),
            next_stream: FIRST_STREAM_ID,
            now_ms: 0,
            isn: config.isn_seed,
        }
    }

    pub fn set_large_segments(&mut self, on: bool) {
        self.large_segments = on;
    }

    /// The next frame for the guest.
    pub fn pop_for_guest(&mut self) -> Option<GuestFrame> {
        self.out.pop_front()
    }

    /// Puts a frame the guest could not take yet back at the head, keeping frame order.
    pub fn requeue_for_guest(&mut self, frame: GuestFrame) {
        self.out.push_front(frame);
    }

    /// Whether `from_guest` would act on `frame`. `UserNet` owns the guest's whole network and
    /// takes everything. `Segment` shares it with the segment's own server, so it takes only the
    /// IPv4 frames addressed to the gateway MAC that leave the subnet, plus DNS at the gateway
    /// address; ARP, DHCP and traffic between machines belong to others.
    pub fn takes_frame(&self, frame: &[u8]) -> bool {
        match self.policy {
            GatewayPolicy::UserNet { .. } => true,
            GatewayPolicy::Segment { .. } => segment_takes(frame),
        }
    }

    /// Takes one frame from the guest. An ACK for TCP data it carries waits for the next `poll`,
    /// so call `poll` after a batch of frames (and after taking `to_remote`).
    pub fn from_guest(&mut self, frame: &[u8], now_ms: u64) {
        self.now_ms = now_ms;
        if frame.len() < ETH_HEADER_LEN || !self.takes_frame(frame) {
            return;
        }
        self.guest_mac = frame[ETH_SRC_MAC].try_into().unwrap();
        match be_u16(&frame[ETH_TYPE]) {
            ETHERTYPE_ARP => self.arp(frame),
            ETHERTYPE_IPV4 => {
                if let Some(ip) = parse_ipv4(frame) {
                    match ip.proto {
                        IP_PROTO_ICMP => self.icmp(&ip),
                        IP_PROTO_TCP => self.tcp_in(&ip),
                        IP_PROTO_UDP => self.udp_in(&ip),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    /// What the driver should do next, oldest first.
    pub fn take_requests(&mut self) -> Vec<GatewayRequest> {
        core::mem::take(&mut self.requests)
    }

    /// The driver connected `stream`: a TCP guest gets its SYN-ACK, datagrams waiting for a UDP
    /// stream are released.
    pub fn opened(&mut self, stream: StreamId) {
        self.tcp_opened(stream);
        self.udp_opened(stream);
        self.dns_tcp_opened(stream);
    }

    /// The driver could not connect `stream`. The flow is gone without a `Close`; a TCP guest gets
    /// a RST for its SYN, a UDP guest ICMP port unreachable.
    pub fn failed(&mut self, stream: StreamId) {
        self.tcp_failed(stream);
        self.udp_failed(stream);
        self.dns_tcp_failed(stream);
    }

    /// Bytes (TCP) or one datagram (UDP) from the remote end. TCP bytes beyond `room` are kept
    /// too, and go to the guest as its window allows at each `poll`.
    pub fn remote_data(&mut self, stream: StreamId, data: &[u8]) {
        self.tcp_remote_data(stream, data);
        self.udp_remote_data(stream, data);
        self.dns_tcp_remote_data(stream, data);
    }

    /// The remote end closed `stream` in an orderly way. A TCP guest gets our FIN after the data
    /// still buffered; a UDP flow is dropped.
    pub fn remote_closed(&mut self, stream: StreamId) {
        self.tcp_remote_closed(stream);
        self.udp_drop(stream);
        self.dns_tcp_remote_closed(stream);
    }

    /// `stream` broke. A TCP guest gets a RST; a UDP flow is dropped without telling the guest.
    /// Either way the flow is gone without a `Close`.
    pub fn reset(&mut self, stream: StreamId) {
        self.tcp_reset(stream);
        self.udp_drop(stream);
        self.dns_tcp_reset(stream);
    }

    /// Moves buffered TCP data to the guest (round-robin, at most `OUT_QUEUE_LIMIT` frames
    /// queued), sends owed ACKs and window updates, and lets time pass: idle UDP flows expire
    /// here, yielding `Close`.
    pub fn poll(&mut self, now_ms: u64) {
        self.now_ms = now_ms;
        self.poll_tcp();
        self.expire_udp(now_ms);
        self.poll_dns_tcp(now_ms);
    }

    /// The open streams the driver should watch and for what; streams wanting neither are left
    /// out.
    pub fn interest(&self) -> Vec<(StreamId, StreamInterest)> {
        let queue_has_room = self.out.len() < OUT_QUEUE_LIMIT;
        let mut interest: Vec<(StreamId, StreamInterest)> = self
            .tcp
            .flows
            .values()
            .filter(|flow| flow.opened)
            .map(|flow| {
                let wants = StreamInterest {
                    read: queue_has_room && flow.room() > 0,
                    write: !flow.to_remote.is_empty(),
                };
                (flow.stream, wants)
            })
            .filter(|(_, wants)| wants.read || wants.write)
            .collect();
        if let Some(stream) = self.dns_tcp.open_stream() {
            let wants = StreamInterest {
                read: queue_has_room,
                write: self.dns_tcp.wants_write(),
            };
            if wants.read || wants.write {
                interest.push((stream, wants));
            }
        }
        if queue_has_room {
            interest.extend(self.udp.open_streams().map(|stream| {
                let wants = StreamInterest {
                    read: true,
                    write: false,
                };
                (stream, wants)
            }));
        }
        interest
    }

    /// The next free stream id. Skips 0 and live ids when the counter wraps.
    fn allocate_stream(&mut self) -> StreamId {
        loop {
            let candidate = self.next_stream;
            self.next_stream = match candidate.wrapping_add(1) {
                0 => FIRST_STREAM_ID,
                next => next,
            };
            if candidate != 0
                && !self.udp.has_stream(candidate)
                && !self.tcp.has_stream(candidate)
                && !self.dns_tcp.has_stream(candidate)
            {
                return candidate;
            }
        }
    }

    fn emit_ip(&mut self, src: [u8; 4], dst: [u8; 4], proto: u8, payload: &[u8]) {
        self.ip_id = self.ip_id.wrapping_add(1);
        for packet in ipv4_packets(self.ip_id, src, dst, proto, payload) {
            self.out.push_back(GuestFrame::plain(ethernet_frame(
                self.guest_mac,
                GATEWAY_MAC,
                ETHERTYPE_IPV4,
                &packet,
            )));
        }
    }

    /// Answers requests for any address in the subnet except the guest's own, so the gateway, DNS
    /// and every NAT destination resolve to `GATEWAY_MAC`. Skipping the guest's address keeps its
    /// duplicate-address probe from being answered as a conflict.
    fn arp(&mut self, frame: &[u8]) {
        if !matches!(self.policy, GatewayPolicy::UserNet { .. })
            || frame.len() < ETH_HEADER_LEN + ARP_PACKET_LEN
        {
            return;
        }
        let arp = &frame[ETH_HEADER_LEN..];
        if be_u16(&arp[ARP_OPCODE]) != ARP_OP_REQUEST {
            return;
        }
        let target_ip = &arp[ARP_TARGET_IP];
        if target_ip[..3] != SUBNET_PREFIX || target_ip[3] == GUEST_IP[3] {
            return;
        }
        let mut reply = ARP_ETHERNET_IPV4.to_vec();
        reply.extend_from_slice(&ARP_OP_REPLY.to_be_bytes());
        reply.extend_from_slice(&GATEWAY_MAC);
        reply.extend_from_slice(target_ip);
        reply.extend_from_slice(&arp[ARP_SENDER_MAC]);
        reply.extend_from_slice(&arp[ARP_SENDER_IP]);
        self.out.push_back(GuestFrame::plain(ethernet_frame(
            self.guest_mac,
            GATEWAY_MAC,
            ETHERTYPE_ARP,
            &reply,
        )));
    }

    /// Echo requests to any destination are answered locally and never leave the host, so ping
    /// only proves the virtual link works.
    fn icmp(&mut self, ip: &Ipv4Packet) {
        if ip.payload.len() < ICMP_ECHO_HEADER_LEN || ip.payload[0] != ICMP_ECHO_REQUEST {
            return;
        }
        let mut reply = ip.payload.to_vec();
        reply[0] = ICMP_ECHO_REPLY;
        reply[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
        let checksum = internet_checksum(&reply);
        reply[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
        self.emit_ip(ip.dst, ip.src, IP_PROTO_ICMP, &reply);
    }

    /// Serves `GUEST_IP` (see `dhcp_reply_type` for which messages are answered). Replies are
    /// always broadcast because the client may not have an address yet.
    pub(super) fn dhcp(&mut self, request: &[u8]) {
        if !matches!(self.policy, GatewayPolicy::UserNet { .. }) {
            return;
        }
        let Some(parsed) = parse_dhcp_request(request) else {
            return;
        };
        let Some(reply_type) = dhcp_reply_type(&parsed) else {
            return;
        };
        let datagram = udp_segment(
            GATEWAY_IP,
            LIMITED_BROADCAST_IP,
            DHCP_SERVER_PORT,
            DHCP_CLIENT_PORT,
            &dhcp_reply(request, reply_type),
        );
        self.ip_id = self.ip_id.wrapping_add(1);
        let packet = ipv4_packet(
            self.ip_id,
            GATEWAY_IP,
            LIMITED_BROADCAST_IP,
            IP_PROTO_UDP,
            &datagram,
        );
        self.out.push_back(GuestFrame::plain(ethernet_frame(
            BROADCAST_MAC,
            GATEWAY_MAC,
            ETHERTYPE_IPV4,
            &packet,
        )));
    }
}

fn segment_takes(frame: &[u8]) -> bool {
    if frame.len() < ETH_HEADER_LEN || frame[..6] != GATEWAY_MAC {
        return false;
    }
    let Some(ip) = parse_ipv4(frame) else {
        return false;
    };
    if ip.dst[..3] != SUBNET_PREFIX {
        return true;
    }
    let transport = matches!(ip.proto, IP_PROTO_TCP | IP_PROTO_UDP);
    // Both protocols put the destination port at the same place.
    ip.dst == GATEWAY_IP
        && transport
        && ip.payload.len() >= UDP_DST_PORT.end
        && be_u16(&ip.payload[UDP_DST_PORT]) == DNS_PORT
}
