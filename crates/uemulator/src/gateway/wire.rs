//! Wire formats: Ethernet, IPv4, ARP, ICMP echo, UDP and TCP, built and parsed by hand. Nothing
//! here verifies a received checksum (the virtual link cannot corrupt a frame), and only
//! fixed-size IPv4 headers are generated.

#[cfg(freestanding)]
use crate::prelude::*;
use core::ops::Range;

/// Largest TCP payload per segment we send: Ethernet MTU 1500 minus 20-byte IPv4 and 20-byte TCP
/// headers. Also advertised in our SYN-ACK.
pub(super) const MSS: usize = 1460;
/// A guest that advertises a tiny (or zero) MSS would make the per-read size 0, and a zero-length
/// read is indistinguishable from EOF.
pub(super) const MIN_MSS: usize = 64;

pub(super) const ETH_HEADER_LEN: usize = 14;
pub(super) const ETH_DST_MAC: Range<usize> = 0..6;
pub(super) const ETH_SRC_MAC: Range<usize> = 6..12;
pub(super) const ETH_TYPE: Range<usize> = 12..14;
pub(super) const ETHERTYPE_IPV4: u16 = 0x0800;
pub(super) const ETHERTYPE_ARP: u16 = 0x0806;

pub(super) const IP_PROTO_ICMP: u8 = 1;
pub(super) const IP_PROTO_TCP: u8 = 6;
pub(super) const IP_PROTO_UDP: u8 = 17;

/// Offsets inside the IPv4 header (after the Ethernet header). Only fixed-size headers (IHL 5)
/// are generated.
pub(super) const IPV4_HEADER_LEN: usize = 20;
pub(super) const IPV4_VERSION: u8 = 4;
/// Version 4 in the high nibble, header length in 32-bit words (5) in the low one.
pub(super) const IPV4_VERSION_IHL_NO_OPTIONS: u8 = 0x45;
pub(super) const IPV4_TOTAL_LEN: Range<usize> = 2..4;
pub(super) const IPV4_FLAGS_FRAGMENT: Range<usize> = 6..8;
pub(super) const IPV4_PROTOCOL: usize = 9;
pub(super) const IPV4_CHECKSUM: Range<usize> = 10..12;
pub(super) const IPV4_SRC: Range<usize> = 12..16;
pub(super) const IPV4_DST: Range<usize> = 16..20;
/// "More fragments" flag plus the fragment offset. The reserved bit and Don't Fragment (0x4000)
/// are excluded so DF packets still pass.
pub(super) const IPV4_FRAGMENT_MASK: u16 = 0x3fff;
pub(super) const IPV4_FLAG_DONT_FRAGMENT: u16 = 0x4000;
pub(super) const IPV4_FLAG_MORE_FRAGMENTS: u16 = 0x2000;
pub(super) const ETHERNET_MTU: usize = 1500;
/// The fragment offset field counts 8-byte units, so every fragment but the last must carry a
/// multiple of 8 bytes: 1480 here, which is also the full MTU payload.
pub(super) const IPV4_FRAGMENT_OFFSET_UNIT: usize = 8;
pub(super) const IPV4_FRAGMENT_PAYLOAD_MAX: usize =
    (ETHERNET_MTU - IPV4_HEADER_LEN) / IPV4_FRAGMENT_OFFSET_UNIT * IPV4_FRAGMENT_OFFSET_UNIT;
/// The 16-bit total length field includes the header.
pub(super) const IPV4_PAYLOAD_LEN_MAX: usize = u16::MAX as usize - IPV4_HEADER_LEN;
pub(super) const IPV4_TTL: u8 = 64;

/// Offsets inside the ARP packet (after the Ethernet header).
pub(super) const ARP_PACKET_LEN: usize = 28;
pub(super) const ARP_OPCODE: Range<usize> = 6..8;
pub(super) const ARP_SENDER_MAC: Range<usize> = 8..14;
pub(super) const ARP_SENDER_IP: Range<usize> = 14..18;
pub(super) const ARP_TARGET_IP: Range<usize> = 24..28;
/// Hardware type Ethernet (1), protocol type IPv4 (0x0800), address lengths 6 and 4.
pub(super) const ARP_ETHERNET_IPV4: [u8; 6] = [0, 1, 8, 0, 6, 4];
pub(super) const ARP_OP_REQUEST: u16 = 1;
pub(super) const ARP_OP_REPLY: u16 = 2;

pub(super) const ICMP_ECHO_REPLY: u8 = 0;
pub(super) const ICMP_ECHO_REQUEST: u8 = 8;
/// Type, code, checksum, identifier, sequence.
pub(super) const ICMP_ECHO_HEADER_LEN: usize = 8;
pub(super) const ICMP_DEST_UNREACHABLE: u8 = 3;
pub(super) const ICMP_CODE_PORT_UNREACHABLE: u8 = 3;
/// Type, code, checksum and four unused bytes, then the quoted packet (RFC 792).
pub(super) const ICMP_ERROR_HEADER_LEN: usize = 8;
pub(super) const ICMP_CHECKSUM: Range<usize> = 2..4;

pub(super) const UDP_HEADER_LEN: usize = 8;
pub(super) const UDP_SRC_PORT: Range<usize> = 0..2;
pub(super) const UDP_DST_PORT: Range<usize> = 2..4;
pub(super) const UDP_LENGTH: Range<usize> = 4..6;
pub(super) const UDP_CHECKSUM: Range<usize> = 6..8;
/// The most a UDP datagram can carry in one IPv4 packet (65 507). A Wisp server's DATA has no
/// such bound, so remote data is checked against it before it becomes a datagram.
pub(super) const UDP_PAYLOAD_MAX: usize = IPV4_PAYLOAD_LEN_MAX - UDP_HEADER_LEN;

pub(super) const TCP_HEADER_LEN: usize = 20;
pub(super) const TCP_SRC_PORT: Range<usize> = 0..2;
pub(super) const TCP_DST_PORT: Range<usize> = 2..4;
pub(super) const TCP_SEQ: Range<usize> = 4..8;
pub(super) const TCP_ACK: Range<usize> = 8..12;
/// High nibble is the header length in 32-bit words.
pub(super) const TCP_DATA_OFFSET: usize = 12;
pub(super) const TCP_FLAGS: usize = 13;
pub(super) const TCP_WINDOW: Range<usize> = 14..16;
pub(super) const TCP_CHECKSUM: Range<usize> = 16..18;
pub(super) const TCP_OPT_END: u8 = 0;
pub(super) const TCP_OPT_NOP: u8 = 1;
pub(super) const TCP_OPT_MSS: u8 = 2;
pub(super) const TCP_OPT_MSS_LEN: usize = 4;

pub(crate) const FIN: u8 = 0x01;
pub(crate) const SYN: u8 = 0x02;
pub(crate) const RST: u8 = 0x04;
pub(crate) const PSH: u8 = 0x08;
pub(crate) const ACK: u8 = 0x10;

/// Adds `data` as big-endian 16-bit words (a trailing odd byte is padded with zero) into a
/// running ones-complement sum. See `checksum_finish`.
pub(super) fn checksum_add(mut acc: u32, data: &[u8]) -> u32 {
    for pair in data.chunks(2) {
        acc += u16::from_be_bytes([pair[0], *pair.get(1).unwrap_or(&0)]) as u32;
    }
    acc
}

/// Folds the carries back in and complements: the Internet checksum (RFC 1071). A packet whose
/// stored checksum is included in the sum folds to 0.
pub(super) fn checksum_finish(mut acc: u32) -> u16 {
    while acc >> 16 != 0 {
        acc = (acc & 0xffff) + (acc >> 16);
    }
    !(acc as u16)
}

pub(crate) fn internet_checksum(data: &[u8]) -> u16 {
    checksum_finish(checksum_add(0, data))
}

/// The IPv4 pseudo-header part of a TCP/UDP checksum (addresses, protocol, segment length), as
/// an unfolded running sum.
pub(super) fn pseudo_header_sum(src: [u8; 4], dst: [u8; 4], proto: u8, segment_len: usize) -> u32 {
    checksum_add(0, &src) + checksum_add(0, &dst) + proto as u32 + segment_len as u32
}

/// TCP/UDP checksum: covers the IPv4 pseudo-header (addresses, protocol, segment length) too.
pub(super) fn transport_checksum(src: [u8; 4], dst: [u8; 4], proto: u8, segment: &[u8]) -> u16 {
    let acc = pseudo_header_sum(src, dst, proto, segment.len());
    checksum_finish(checksum_add(acc, segment))
}

pub(super) fn be_u16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

pub(super) fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

pub(super) fn ethernet_frame(
    dst: [u8; 6],
    src: [u8; 6],
    ethertype: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(ETH_HEADER_LEN + payload.len());
    frame.extend_from_slice(&dst);
    frame.extend_from_slice(&src);
    frame.extend_from_slice(&ethertype.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

/// Appends the header of an IPv4 packet carrying `payload_len` bytes, which must fit the 16-bit
/// total length.
fn push_ipv4_header(
    packet: &mut Vec<u8>,
    id: u16,
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    flags_and_offset: u16,
    payload_len: usize,
) {
    let start = packet.len();
    packet.extend_from_slice(&[IPV4_VERSION_IHL_NO_OPTIONS, 0]);
    packet.extend_from_slice(&(IPV4_HEADER_LEN as u16 + payload_len as u16).to_be_bytes());
    packet.extend_from_slice(&id.to_be_bytes());
    packet.extend_from_slice(&flags_and_offset.to_be_bytes());
    packet.push(IPV4_TTL);
    packet.push(proto);
    packet.extend_from_slice(&[0, 0]);
    packet.extend_from_slice(&src);
    packet.extend_from_slice(&dst);
    let header = &mut packet[start..];
    let checksum = internet_checksum(header);
    header[IPV4_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
}

/// One IPv4 packet whose flags/fragment-offset word is `flags_and_offset`. `payload` must fit the
/// 16-bit total length.
pub(super) fn ipv4_datagram(
    id: u16,
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    flags_and_offset: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut packet = Vec::with_capacity(IPV4_HEADER_LEN + payload.len());
    push_ipv4_header(
        &mut packet,
        id,
        src,
        dst,
        proto,
        flags_and_offset,
        payload.len(),
    );
    packet.extend_from_slice(payload);
    packet
}

/// A single unfragmented packet with Don't Fragment set. `payload` must not exceed one MTU; use
/// `ipv4_packets` when it might.
pub(super) fn ipv4_packet(
    id: u16,
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    payload: &[u8],
) -> Vec<u8> {
    ipv4_datagram(id, src, dst, proto, IPV4_FLAG_DONT_FRAGMENT, payload)
}

/// Splits `payload` into packets that fit the MTU. All fragments share `id`, carry More Fragments
/// except the last, and have Don't Fragment cleared (a fragment with DF set could never be
/// reassembled by a receiver that must not fragment further). Empty if the payload cannot fit one
/// IPv4 packet at all.
pub(super) fn ipv4_packets(
    id: u16,
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    payload: &[u8],
) -> Vec<Vec<u8>> {
    if payload.len() > IPV4_PAYLOAD_LEN_MAX {
        return Vec::new();
    }
    if payload.len() <= IPV4_FRAGMENT_PAYLOAD_MAX {
        return vec![ipv4_packet(id, src, dst, proto, payload)];
    }
    payload
        .chunks(IPV4_FRAGMENT_PAYLOAD_MAX)
        .enumerate()
        .map(|(index, chunk)| {
            let start = index * IPV4_FRAGMENT_PAYLOAD_MAX;
            let offset_units = (start / IPV4_FRAGMENT_OFFSET_UNIT) as u16;
            let more = start + chunk.len() < payload.len();
            let flags = if more { IPV4_FLAG_MORE_FRAGMENTS } else { 0 };
            ipv4_datagram(id, src, dst, proto, flags | offset_units, chunk)
        })
        .collect()
}

pub(super) fn udp_segment(
    src: [u8; 4],
    dst: [u8; 4],
    sport: u16,
    dport: u16,
    data: &[u8],
) -> Vec<u8> {
    let mut segment = Vec::with_capacity(UDP_HEADER_LEN + data.len());
    segment.extend_from_slice(&sport.to_be_bytes());
    segment.extend_from_slice(&dport.to_be_bytes());
    // Callers keep `data` within `UDP_PAYLOAD_MAX`. Past it the length saturates instead of
    // overflowing, and `ipv4_packets` refuses the segment anyway.
    let length = u16::try_from(UDP_HEADER_LEN + data.len()).unwrap_or(u16::MAX);
    segment.extend_from_slice(&length.to_be_bytes());
    segment.extend_from_slice(&[0, 0]);
    segment.extend_from_slice(data);
    // A computed checksum of 0 is sent as 0xffff because 0 means "no checksum" in UDP (RFC 768).
    let checksum = match transport_checksum(src, dst, IP_PROTO_UDP, &segment) {
        0 => 0xffff,
        checksum => checksum,
    };
    segment[UDP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    segment
}

#[derive(Clone, Copy)]
pub(super) struct TcpHeader {
    pub(super) src: [u8; 4],
    pub(super) dst: [u8; 4],
    pub(super) sport: u16,
    pub(super) dport: u16,
    pub(super) seq: u32,
    pub(super) ack: u32,
    pub(super) flags: u8,
    pub(super) window: u16,
    /// Append the MSS option; only SYN segments carry it.
    pub(super) advertise_mss: bool,
}

impl TcpHeader {
    fn len(&self) -> usize {
        TCP_HEADER_LEN
            + if self.advertise_mss {
                TCP_OPT_MSS_LEN
            } else {
                0
            }
    }

    /// Appends the header with a zero checksum field.
    fn push(&self, segment: &mut Vec<u8>) {
        let header_len = self.len();
        segment.extend_from_slice(&self.sport.to_be_bytes());
        segment.extend_from_slice(&self.dport.to_be_bytes());
        segment.extend_from_slice(&self.seq.to_be_bytes());
        segment.extend_from_slice(&self.ack.to_be_bytes());
        segment.push((header_len as u8 / 4) << 4);
        segment.push(self.flags);
        segment.extend_from_slice(&self.window.to_be_bytes());
        // Checksum placeholder and urgent pointer.
        segment.extend_from_slice(&[0, 0, 0, 0]);
        if self.advertise_mss {
            segment.extend_from_slice(&[TCP_OPT_MSS, TCP_OPT_MSS_LEN as u8]);
            segment.extend_from_slice(&(MSS as u16).to_be_bytes());
        }
    }
}

/// The gateway builds whole frames with `tcp_frame`; the tests' guest side builds bare segments
/// with this.
#[cfg(test)]
pub(super) fn tcp_segment(header: &TcpHeader, data: &[u8]) -> Vec<u8> {
    let mut segment = Vec::with_capacity(header.len() + data.len());
    header.push(&mut segment);
    segment.extend_from_slice(data);
    let checksum = transport_checksum(header.src, header.dst, IP_PROTO_TCP, &segment);
    segment[TCP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    segment
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TcpChecksum {
    Complete,
    /// Only the folded pseudo-header sum, not complemented: what a NIC doing checksum offload
    /// expects, finished by whoever takes the frame (`NetHandle::send_tcp_segment`). Saves summing
    /// a 64 KiB payload here.
    PseudoHeaderOnly,
}

/// An Ethernet frame holding one TCP segment in one IPv4 packet with Don't Fragment set, however
/// long: a large segment is split by the guest, so it must not be fragmented here. Built in one
/// buffer so the payload is copied once, straight from wherever it waits; it comes in pieces
/// because a ring buffer hands it out that way. The segment must fit an IPv4 packet.
pub(super) fn tcp_frame(
    dst_mac: [u8; 6],
    src_mac: [u8; 6],
    ip_id: u16,
    header: &TcpHeader,
    data: &[&[u8]],
    checksum: TcpChecksum,
) -> Vec<u8> {
    let segment_len = header.len() + data.iter().map(|piece| piece.len()).sum::<usize>();
    let tcp_start = ETH_HEADER_LEN + IPV4_HEADER_LEN;
    let mut frame = Vec::with_capacity(tcp_start + segment_len);
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
    push_ipv4_header(
        &mut frame,
        ip_id,
        header.src,
        header.dst,
        IP_PROTO_TCP,
        IPV4_FLAG_DONT_FRAGMENT,
        segment_len,
    );
    header.push(&mut frame);
    for piece in data {
        frame.extend_from_slice(piece);
    }
    let field = match checksum {
        TcpChecksum::Complete => {
            transport_checksum(header.src, header.dst, IP_PROTO_TCP, &frame[tcp_start..])
        }
        TcpChecksum::PseudoHeaderOnly => !checksum_finish(pseudo_header_sum(
            header.src,
            header.dst,
            IP_PROTO_TCP,
            segment_len,
        )),
    };
    let checksum_at = tcp_start + TCP_CHECKSUM.start;
    frame[checksum_at..checksum_at + TCP_CHECKSUM.len()].copy_from_slice(&field.to_be_bytes());
    frame
}

pub(super) struct Ipv4Packet<'a> {
    /// The IPv4 header including options, for quoting in ICMP errors.
    pub(super) header: &'a [u8],
    pub(super) src: [u8; 4],
    pub(super) dst: [u8; 4],
    pub(super) proto: u8,
    pub(super) payload: &'a [u8],
}

/// Parses an Ethernet frame carrying an unfragmented IPv4 packet. Trailing Ethernet padding is
/// cut off using the IPv4 total length.
pub(super) fn parse_ipv4(frame: &[u8]) -> Option<Ipv4Packet<'_>> {
    if frame.len() < ETH_HEADER_LEN + IPV4_HEADER_LEN
        || frame[ETH_TYPE] != ETHERTYPE_IPV4.to_be_bytes()
    {
        return None;
    }
    let ip = &frame[ETH_HEADER_LEN..];
    if ip[0] >> 4 != IPV4_VERSION {
        return None;
    }
    let header_len = (ip[0] & 0xf) as usize * 4;
    let total_len = be_u16(&ip[IPV4_TOTAL_LEN]) as usize;
    let fragment = be_u16(&ip[IPV4_FLAGS_FRAGMENT]) & IPV4_FRAGMENT_MASK;
    if header_len < IPV4_HEADER_LEN
        || total_len < header_len
        || total_len > ip.len()
        || fragment != 0
    {
        return None;
    }
    Some(Ipv4Packet {
        header: &ip[..header_len],
        src: ip[IPV4_SRC].try_into().ok()?,
        dst: ip[IPV4_DST].try_into().ok()?,
        proto: ip[IPV4_PROTOCOL],
        payload: &ip[header_len..total_len],
    })
}

pub(super) struct TcpSegment<'a> {
    pub(super) sport: u16,
    pub(super) dport: u16,
    pub(super) seq: u32,
    pub(super) ack: u32,
    pub(super) flags: u8,
    pub(super) window: u32,
    pub(super) options: &'a [u8],
    pub(super) data: &'a [u8],
}

impl<'a> TcpSegment<'a> {
    pub(super) fn parse(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < TCP_HEADER_LEN {
            return None;
        }
        let header_len = (payload[TCP_DATA_OFFSET] >> 4) as usize * 4;
        if header_len < TCP_HEADER_LEN || header_len > payload.len() {
            return None;
        }
        Some(TcpSegment {
            sport: be_u16(&payload[TCP_SRC_PORT]),
            dport: be_u16(&payload[TCP_DST_PORT]),
            seq: be_u32(&payload[TCP_SEQ]),
            ack: be_u32(&payload[TCP_ACK]),
            flags: payload[TCP_FLAGS],
            window: be_u16(&payload[TCP_WINDOW]) as u32,
            options: &payload[TCP_HEADER_LEN..header_len],
            data: &payload[header_len..],
        })
    }
}

/// MSS to use towards the guest: our own `MSS` lowered to what the guest's SYN options ask for.
pub(super) fn negotiated_mss(options: &[u8]) -> usize {
    let mut mss = MSS;
    let mut rest = options;
    while rest.len() >= 2 && rest[0] != TCP_OPT_END {
        if rest[0] == TCP_OPT_NOP {
            rest = &rest[1..];
            continue;
        }
        let len = rest[1] as usize;
        if len < 2 || len > rest.len() {
            break;
        }
        if rest[0] == TCP_OPT_MSS && len == TCP_OPT_MSS_LEN {
            mss = mss.min(be_u16(&rest[2..4]) as usize);
        }
        rest = &rest[len..];
    }
    mss.max(MIN_MSS)
}

/// Modular sequence-number comparison (RFC 793 section 3.3): true when `a` is before `b`.
pub(super) fn seq_before(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) < 0
}
