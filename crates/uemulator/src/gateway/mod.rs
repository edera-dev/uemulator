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
#[cfg(test)]
mod tests;
// The TCP codecs and UDP ports wait for the flows that use them.
#[allow(dead_code)]
mod wire;

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;

use dhcp::{DHCP_CLIENT_PORT, DHCP_SERVER_PORT, dhcp_reply, dhcp_reply_type, parse_dhcp_request};
use wire::{
    ARP_ETHERNET_IPV4, ARP_OP_REPLY, ARP_OP_REQUEST, ARP_OPCODE, ARP_PACKET_LEN, ARP_SENDER_IP,
    ARP_SENDER_MAC, ARP_TARGET_IP, ETH_HEADER_LEN, ETH_SRC_MAC, ETH_TYPE, ETHERTYPE_ARP,
    ETHERTYPE_IPV4, ICMP_CHECKSUM, ICMP_ECHO_HEADER_LEN, ICMP_ECHO_REPLY, ICMP_ECHO_REQUEST,
    IP_PROTO_ICMP, IP_PROTO_TCP, IP_PROTO_UDP, Ipv4Packet, UDP_DST_PORT, UDP_HEADER_LEN,
    UDP_LENGTH, be_u16, ethernet_frame, internet_checksum, ipv4_packet, ipv4_packets, parse_ipv4,
    udp_segment,
};

/// 52:55 followed by the gateway IP, the way QEMU's slirp derives its MAC.
pub(crate) const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
/// Also the DHCP server and default router.
pub(crate) const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];
const DNS_IP: [u8; 4] = [10, 0, 2, 3];
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

/// An IPv4 address and port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub address: [u8; 4],
    pub port: u16,
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
    #[allow(dead_code, reason = "read by the TCP flows")]
    large_segments: bool,
    ip_id: u16,
    /// Initial sequence number of the most recent connection.
    #[allow(dead_code, reason = "read by the TCP flows")]
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

    pub fn from_guest(&mut self, frame: &[u8], _now_ms: u64) {
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
                        IP_PROTO_UDP => self.udp_in(&ip),
                        // TCP flows and the rest of UDP arrive with the flow tables.
                        _ => {}
                    }
                }
            }
            _ => {}
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

    fn udp_in(&mut self, ip: &Ipv4Packet) {
        let datagram = ip.payload;
        if datagram.len() < UDP_HEADER_LEN {
            return;
        }
        let dport = be_u16(&datagram[UDP_DST_PORT]);
        let len = (be_u16(&datagram[UDP_LENGTH]) as usize).clamp(UDP_HEADER_LEN, datagram.len());
        let data = &datagram[UDP_HEADER_LEN..len];
        // Datagrams to the DHCP port are never NAT-ed: when UDP flows arrive they go in an else.
        if dport == DHCP_SERVER_PORT {
            self.dhcp(data);
        }
    }

    /// Serves `GUEST_IP` (see `dhcp_reply_type` for which messages are answered). Replies are
    /// always broadcast because the client may not have an address yet.
    fn dhcp(&mut self, request: &[u8]) {
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
