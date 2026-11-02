//! Just enough Ethernet, ARP, IPv4, UDP and ICMP for the segment's server and for what the agent
//! snoops. Frames come from guests, so every read is bounds-checked and a malformed frame is
//! simply not understood.

#[cfg(freestanding)]
use crate::prelude::*;
use core::ops::Range;

pub(crate) type Mac = [u8; 6];
pub(crate) type Ipv4 = [u8; 4];

pub(crate) const BROADCAST_MAC: Mac = [0xff; 6];
pub(crate) const UNSPECIFIED_IP: Ipv4 = [0; 4];
pub(crate) const LIMITED_BROADCAST_IP: Ipv4 = [0xff; 4];

pub(crate) const ETH_HEADER_LEN: usize = 14;
const ETH_DST: Range<usize> = 0..6;
const ETH_SRC: Range<usize> = 6..12;
const ETH_TYPE: Range<usize> = 12..14;
pub(crate) const ETHERTYPE_IPV4: u16 = 0x0800;
pub(crate) const ETHERTYPE_ARP: u16 = 0x0806;
/// The I/G bit of an address's first byte: set on group addresses (multicast and broadcast).
const GROUP_BIT: u8 = 0x01;
/// The U/L bit: set on locally administered addresses, the only kind a tab may make up.
const LOCALLY_ADMINISTERED_BIT: u8 = 0x02;

const ARP_PACKET_LEN: usize = 28;
/// Hardware type Ethernet, protocol IPv4, address lengths 6 and 4.
const ARP_ETHERNET_IPV4: [u8; 6] = [0, 1, 8, 0, 6, 4];
const ARP_PREFIX: Range<usize> = 0..6;
const ARP_OPCODE: Range<usize> = 6..8;
const ARP_SENDER_MAC: Range<usize> = 8..14;
const ARP_SENDER_IP: Range<usize> = 14..18;
const ARP_TARGET_MAC: Range<usize> = 18..24;
const ARP_TARGET_IP: Range<usize> = 24..28;
pub(crate) const ARP_REQUEST: u16 = 1;
pub(crate) const ARP_REPLY: u16 = 2;

const IPV4_HEADER_LEN: usize = 20;
const IPV4_VERSION: u8 = 4;
const IPV4_VERSION_IHL_NO_OPTIONS: u8 = 0x45;
/// The IHL field counts 32-bit words.
const IPV4_IHL_UNIT: usize = 4;
const IPV4_IHL_MASK: u8 = 0x0f;
const IPV4_TOTAL_LEN: Range<usize> = 2..4;
const IPV4_FLAGS_FRAGMENT: Range<usize> = 6..8;
/// Fragment offset and the more-fragments flag: either set means this is not a whole datagram.
const IPV4_FRAGMENT_MASK: u16 = 0x3fff;
const IPV4_TTL_FIELD: usize = 8;
const IPV4_PROTOCOL: usize = 9;
const IPV4_CHECKSUM: Range<usize> = 10..12;
const IPV4_SRC: Range<usize> = 12..16;
const IPV4_DST: Range<usize> = 16..20;
const IPV4_TTL: u8 = 64;
pub(crate) const IP_PROTOCOL_ICMP: u8 = 1;
pub(crate) const IP_PROTOCOL_UDP: u8 = 17;

const UDP_HEADER_LEN: usize = 8;
const UDP_SRC_PORT: Range<usize> = 0..2;
const UDP_DST_PORT: Range<usize> = 2..4;
const UDP_LENGTH: Range<usize> = 4..6;
const UDP_CHECKSUM: Range<usize> = 6..8;
/// A computed UDP checksum of zero is sent as all ones: zero means "no checksum" (RFC 768).
const UDP_CHECKSUM_ZERO_AS: u16 = 0xffff;

const ICMP_ECHO_REPLY: u8 = 0;
const ICMP_ECHO_REQUEST: u8 = 8;
/// Type, code, checksum, identifier and sequence number.
const ICMP_ECHO_HEADER_LEN: usize = 8;
const ICMP_TYPE: usize = 0;
const ICMP_CHECKSUM: Range<usize> = 2..4;

pub(crate) fn is_group(mac: &Mac) -> bool {
    mac[0] & GROUP_BIT != 0
}

/// What a tab may take for its NIC: a unicast address from the locally administered space, so
/// it can never be some real interface's.
pub(crate) fn is_local_unicast(mac: &Mac) -> bool {
    !is_group(mac) && mac[0] & LOCALLY_ADMINISTERED_BIT != 0
}

fn be16(bytes: &[u8], range: Range<usize>) -> Option<u16> {
    Some(u16::from_be_bytes(bytes.get(range)?.try_into().ok()?))
}

fn array<const N: usize>(bytes: &[u8], range: Range<usize>) -> Option<[u8; N]> {
    bytes.get(range)?.try_into().ok()
}

pub(crate) struct Ethernet<'a> {
    pub dst: Mac,
    pub src: Mac,
    pub ethertype: u16,
    pub payload: &'a [u8],
}

pub(crate) fn parse_ethernet(frame: &[u8]) -> Option<Ethernet<'_>> {
    Some(Ethernet {
        dst: array(frame, ETH_DST)?,
        src: array(frame, ETH_SRC)?,
        ethertype: be16(frame, ETH_TYPE)?,
        payload: frame.get(ETH_HEADER_LEN..)?,
    })
}

pub(crate) fn ethernet_frame(dst: Mac, src: Mac, ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(ETH_HEADER_LEN + payload.len());
    frame.extend_from_slice(&dst);
    frame.extend_from_slice(&src);
    frame.extend_from_slice(&ethertype.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

pub(crate) struct Arp {
    pub op: u16,
    pub sender_mac: Mac,
    pub sender_ip: Ipv4,
    pub target_ip: Ipv4,
}

pub(crate) fn parse_arp(payload: &[u8]) -> Option<Arp> {
    if payload.get(ARP_PREFIX)? != ARP_ETHERNET_IPV4 {
        return None;
    }
    Some(Arp {
        op: be16(payload, ARP_OPCODE)?,
        sender_mac: array(payload, ARP_SENDER_MAC)?,
        sender_ip: array(payload, ARP_SENDER_IP)?,
        target_ip: array(payload, ARP_TARGET_IP)?,
    })
}

pub(crate) fn arp_packet(
    op: u16,
    sender_mac: Mac,
    sender_ip: Ipv4,
    target_mac: Mac,
    target_ip: Ipv4,
) -> Vec<u8> {
    let mut packet = vec![0u8; ARP_PACKET_LEN];
    packet[ARP_PREFIX].copy_from_slice(&ARP_ETHERNET_IPV4);
    packet[ARP_OPCODE].copy_from_slice(&op.to_be_bytes());
    packet[ARP_SENDER_MAC].copy_from_slice(&sender_mac);
    packet[ARP_SENDER_IP].copy_from_slice(&sender_ip);
    packet[ARP_TARGET_MAC].copy_from_slice(&target_mac);
    packet[ARP_TARGET_IP].copy_from_slice(&target_ip);
    packet
}

pub(crate) struct Ipv4Packet<'a> {
    pub src: Ipv4,
    pub dst: Ipv4,
    pub protocol: u8,
    /// A fragment of a larger datagram: `payload` is not the whole of it.
    pub fragment: bool,
    pub payload: &'a [u8],
}

pub(crate) fn parse_ipv4(packet: &[u8]) -> Option<Ipv4Packet<'_>> {
    let version_ihl = *packet.first()?;
    if version_ihl >> 4 != IPV4_VERSION {
        return None;
    }
    let header_len = usize::from(version_ihl & IPV4_IHL_MASK) * IPV4_IHL_UNIT;
    let total_len = usize::from(be16(packet, IPV4_TOTAL_LEN)?);
    if header_len < IPV4_HEADER_LEN || total_len < header_len {
        return None;
    }
    Some(Ipv4Packet {
        src: array(packet, IPV4_SRC)?,
        dst: array(packet, IPV4_DST)?,
        protocol: *packet.get(IPV4_PROTOCOL)?,
        fragment: be16(packet, IPV4_FLAGS_FRAGMENT)? & IPV4_FRAGMENT_MASK != 0,
        // Ethernet pads short frames: the datagram ends where its header says.
        payload: packet.get(header_len..total_len)?,
    })
}

/// The ones' complement sum of `data` as 16-bit big-endian words, not yet folded.
fn checksum_add(mut sum: u32, data: &[u8]) -> u32 {
    let (words, rest) = data.as_chunks::<2>();
    for word in words {
        sum += u32::from(u16::from_be_bytes(*word));
    }
    if let [last] = rest {
        sum += u32::from(*last) << 8;
    }
    sum
}

fn checksum_finish(mut sum: u32) -> u16 {
    while sum > u32::from(u16::MAX) {
        sum = (sum & u32::from(u16::MAX)) + (sum >> 16);
    }
    !(sum as u16)
}

pub(crate) fn checksum(data: &[u8]) -> u16 {
    checksum_finish(checksum_add(0, data))
}

pub(crate) fn ipv4_packet(src: Ipv4, dst: Ipv4, protocol: u8, payload: &[u8]) -> Vec<u8> {
    let mut packet = vec![0u8; IPV4_HEADER_LEN];
    packet[0] = IPV4_VERSION_IHL_NO_OPTIONS;
    let total_len = (IPV4_HEADER_LEN + payload.len()) as u16;
    packet[IPV4_TOTAL_LEN].copy_from_slice(&total_len.to_be_bytes());
    packet[IPV4_TTL_FIELD] = IPV4_TTL;
    packet[IPV4_PROTOCOL] = protocol;
    packet[IPV4_SRC].copy_from_slice(&src);
    packet[IPV4_DST].copy_from_slice(&dst);
    let sum = checksum(&packet);
    packet[IPV4_CHECKSUM].copy_from_slice(&sum.to_be_bytes());
    packet.extend_from_slice(payload);
    packet
}

pub(crate) struct Udp<'a> {
    pub src_port: u16,
    pub dst_port: u16,
    pub data: &'a [u8],
}

pub(crate) fn parse_udp(datagram: &[u8]) -> Option<Udp<'_>> {
    let len = usize::from(be16(datagram, UDP_LENGTH)?);
    Some(Udp {
        src_port: be16(datagram, UDP_SRC_PORT)?,
        dst_port: be16(datagram, UDP_DST_PORT)?,
        data: datagram.get(UDP_HEADER_LEN..len)?,
    })
}

pub(crate) fn udp_datagram(
    src: Ipv4,
    dst: Ipv4,
    src_port: u16,
    dst_port: u16,
    data: &[u8],
) -> Vec<u8> {
    let len = (UDP_HEADER_LEN + data.len()) as u16;
    let mut datagram = vec![0u8; UDP_HEADER_LEN];
    datagram[UDP_SRC_PORT].copy_from_slice(&src_port.to_be_bytes());
    datagram[UDP_DST_PORT].copy_from_slice(&dst_port.to_be_bytes());
    datagram[UDP_LENGTH].copy_from_slice(&len.to_be_bytes());
    datagram.extend_from_slice(data);
    let mut pseudo = [0u8; 12];
    pseudo[0..4].copy_from_slice(&src);
    pseudo[4..8].copy_from_slice(&dst);
    pseudo[9] = IP_PROTOCOL_UDP;
    pseudo[10..12].copy_from_slice(&len.to_be_bytes());
    let sum = match checksum_finish(checksum_add(checksum_add(0, &pseudo), &datagram)) {
        0 => UDP_CHECKSUM_ZERO_AS,
        sum => sum,
    };
    datagram[UDP_CHECKSUM].copy_from_slice(&sum.to_be_bytes());
    datagram
}

/// The echo reply to an ICMP echo request (the message, not the IP packet), or `None` for any
/// other ICMP message.
pub(crate) fn icmp_echo_reply(message: &[u8]) -> Option<Vec<u8>> {
    if message.len() < ICMP_ECHO_HEADER_LEN || message[ICMP_TYPE] != ICMP_ECHO_REQUEST {
        return None;
    }
    let mut reply = message.to_vec();
    reply[ICMP_TYPE] = ICMP_ECHO_REPLY;
    reply[ICMP_CHECKSUM].fill(0);
    let sum = checksum(&reply);
    reply[ICMP_CHECKSUM].copy_from_slice(&sum.to_be_bytes());
    Some(reply)
}

#[cfg(test)]
pub(crate) fn icmp_echo_request(identifier: u16, sequence: u16, payload: &[u8]) -> Vec<u8> {
    let mut message = vec![ICMP_ECHO_REQUEST, 0, 0, 0];
    message.extend_from_slice(&identifier.to_be_bytes());
    message.extend_from_slice(&sequence.to_be_bytes());
    message.extend_from_slice(payload);
    let sum = checksum(&message);
    message[ICMP_CHECKSUM].copy_from_slice(&sum.to_be_bytes());
    message
}

#[cfg(test)]
pub(crate) fn is_icmp_echo_reply(message: &[u8]) -> bool {
    message.len() >= ICMP_ECHO_HEADER_LEN && message[ICMP_TYPE] == ICMP_ECHO_REPLY
}
