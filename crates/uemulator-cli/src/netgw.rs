//! `--network-link-loop`: a minimal gateway that only answers ARP and ICMP echo for 10.0.2.2. Used by the
//! tests and for checking the virtio-net path; it does no DHCP and forwards nothing.

use std::ops::Range;

/// The address usernet's gateway has too, so a guest configured for one works with the other.
const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];
/// 52:55 followed by the gateway IP, the way QEMU's slirp derives its MAC.
const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];

const ETH_HEADER_LEN: usize = 14;
const ETH_SRC_MAC: Range<usize> = 6..12;
const ETH_DST_MAC: Range<usize> = 0..6;
const ETH_TYPE: Range<usize> = 12..14;
const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;

/// Offsets are from the start of the Ethernet frame. The ARP packet starts at 14.
const ARP_FRAME_LEN: usize = 42;
const ARP_OPCODE: Range<usize> = 20..22;
/// Sender MAC and IP, which become the target of the reply.
const ARP_SENDER: Range<usize> = 22..32;
const ARP_TARGET_IP: Range<usize> = 38..42;
const ARP_REQUEST: [u8; 2] = [0, 1];
/// Ethernet/IPv4 hardware and protocol types, address lengths 6 and 4, then opcode 2 (reply).
const ARP_REPLY_HEADER: [u8; 8] = [0, 1, 8, 0, 6, 4, 0, 2];

/// A fixed 20-byte IPv4 header is assumed, so the ICMP header starts at frame offset 34.
const IPV4_PROTOCOL: usize = 23;
const IPV4_SRC: Range<usize> = 26..30;
const IPV4_DST: Range<usize> = 30..34;
/// Total length field of the IPv4 header, counted from the start of the IP header (frame offset 14).
const IPV4_TOTAL_LEN: Range<usize> = 16..18;
const IP_PROTO_ICMP: u8 = 1;
const ICMP_TYPE: usize = 34;
const ICMP_CHECKSUM: Range<usize> = 36..38;
/// Shortest frame this gateway can inspect and rewrite: everything up to and including the ICMP
/// checksum. Anything shorter must not be indexed.
const ICMP_FRAME_MIN_LEN: usize = ICMP_CHECKSUM.end;
const ICMP_ECHO_REQUEST: u8 = 8;
const ICMP_ECHO_REPLY: u8 = 0;

/// RFC 1071: the ones-complement of the ones-complement sum of big-endian 16-bit words, a
/// trailing odd byte padded with zero. The `u32` cannot overflow below 128 KiB of data, and a
/// frame is far shorter.
fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = data
        .chunks(2)
        .map(|pair| u16::from_be_bytes([pair[0], *pair.get(1).unwrap_or(&0)]) as u32)
        .sum();
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

pub fn handle(frame: &[u8]) -> Option<Vec<u8>> {
    if frame.len() < ETH_HEADER_LEN {
        return None;
    }
    let src_mac: [u8; 6] = frame[ETH_SRC_MAC].try_into().unwrap();
    let ethertype = u16::from_be_bytes(frame[ETH_TYPE].try_into().unwrap());
    match ethertype {
        ETHERTYPE_ARP
            if frame.len() >= ARP_FRAME_LEN
                && frame[ARP_OPCODE] == ARP_REQUEST
                && frame[ARP_TARGET_IP] == GATEWAY_IP =>
        {
            let mut reply = Vec::with_capacity(ARP_FRAME_LEN);
            reply.extend_from_slice(&src_mac);
            reply.extend_from_slice(&GATEWAY_MAC);
            reply.extend_from_slice(&ETHERTYPE_ARP.to_be_bytes());
            reply.extend_from_slice(&ARP_REPLY_HEADER);
            reply.extend_from_slice(&GATEWAY_MAC);
            reply.extend_from_slice(&GATEWAY_IP);
            reply.extend_from_slice(&frame[ARP_SENDER]);
            Some(reply)
        }
        ETHERTYPE_IPV4
            if frame.len() >= ICMP_FRAME_MIN_LEN
                && frame[IPV4_PROTOCOL] == IP_PROTO_ICMP
                && frame[IPV4_DST] == GATEWAY_IP
                && frame[ICMP_TYPE] == ICMP_ECHO_REQUEST =>
        {
            // Ethernet padding after the IP packet must not enter the ICMP checksum, or the guest
            // (which sums only what the IP total length covers) rejects the reply.
            let ip_end = ETH_HEADER_LEN
                + u16::from_be_bytes(frame[IPV4_TOTAL_LEN].try_into().unwrap()) as usize;
            let mut reply = frame[..ip_end.clamp(ICMP_FRAME_MIN_LEN, frame.len())].to_vec();
            reply[ETH_DST_MAC].copy_from_slice(&src_mac);
            reply[ETH_SRC_MAC].copy_from_slice(&GATEWAY_MAC);
            // Swapping source and destination leaves the IPv4 header checksum valid.
            let (guest_ip, gateway_ip) = (frame[IPV4_SRC].to_vec(), frame[IPV4_DST].to_vec());
            reply[IPV4_SRC].copy_from_slice(&gateway_ip);
            reply[IPV4_DST].copy_from_slice(&guest_ip);
            reply[ICMP_TYPE] = ICMP_ECHO_REPLY;
            reply[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
            let checksum = internet_checksum(&reply[ICMP_TYPE..]);
            reply[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
            Some(reply)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::GUEST_MAC;
    use crate::test_support::{XorShift, reference_checksum};

    const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
    const OTHER_IP: [u8; 4] = [8, 8, 8, 8];
    const BROADCAST_MAC: [u8; 6] = [0xff; 6];
    const ARP_REQUEST_OPCODE: [u8; 2] = [0, 1];
    const ARP_REPLY_OPCODE: [u8; 2] = [0, 2];
    /// Hardware type, protocol type, address lengths: Ethernet / IPv4.
    const ARP_ETHERNET_IPV4: [u8; 6] = [0, 1, 8, 0, 6, 4];
    const ICMP_ECHO_REQUEST_BODY: [u8; 4] = [0x12, 0x34, 0x00, 0x07];
    const ICMP_DEST_UNREACHABLE: u8 = 3;
    const IP_PROTO_UDP: u8 = 17;
    const IP_PROTO_TCP: u8 = 6;
    const FUZZ_SEED: u64 = 0x5eed_0001;
    const FUZZ_ITERATIONS: usize = 20_000;
    const FUZZ_MAX_LEN: usize = 96;

    fn arp_frame(opcode: [u8; 2], target_ip: [u8; 4]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&BROADCAST_MAC);
        frame.extend_from_slice(&GUEST_MAC);
        frame.extend_from_slice(&[0x08, 0x06]);
        frame.extend_from_slice(&ARP_ETHERNET_IPV4);
        frame.extend_from_slice(&opcode);
        frame.extend_from_slice(&GUEST_MAC);
        frame.extend_from_slice(&GUEST_IP);
        frame.extend_from_slice(&[0; 6]);
        frame.extend_from_slice(&target_ip);
        frame
    }

    fn ipv4_frame(proto: u8, dst: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&GATEWAY_MAC);
        frame.extend_from_slice(&GUEST_MAC);
        frame.extend_from_slice(&[0x08, 0x00]);
        let mut header = vec![0x45, 0];
        header.extend_from_slice(&(20 + payload.len() as u16).to_be_bytes());
        header.extend_from_slice(&[0, 1, 0x40, 0, 64, proto, 0, 0]);
        header.extend_from_slice(&GUEST_IP);
        header.extend_from_slice(&dst);
        let checksum = reference_checksum(&header);
        header[10..12].copy_from_slice(&checksum.to_be_bytes());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(payload);
        frame
    }

    fn icmp_message(kind: u8, data: &[u8]) -> Vec<u8> {
        let mut message = vec![kind, 0, 0, 0];
        message.extend_from_slice(&ICMP_ECHO_REQUEST_BODY);
        message.extend_from_slice(data);
        let checksum = reference_checksum(&message);
        message[2..4].copy_from_slice(&checksum.to_be_bytes());
        message
    }

    fn echo_request(dst: [u8; 4], data: &[u8]) -> Vec<u8> {
        ipv4_frame(IP_PROTO_ICMP, dst, &icmp_message(ICMP_ECHO_REQUEST, data))
    }

    #[test]
    fn arp_request_for_gateway_gets_exact_reply() {
        let reply = handle(&arp_frame(ARP_REQUEST_OPCODE, GATEWAY_IP)).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(&GUEST_MAC);
        expected.extend_from_slice(&GATEWAY_MAC);
        expected.extend_from_slice(&[0x08, 0x06]);
        expected.extend_from_slice(&ARP_ETHERNET_IPV4);
        expected.extend_from_slice(&ARP_REPLY_OPCODE);
        expected.extend_from_slice(&GATEWAY_MAC);
        expected.extend_from_slice(&GATEWAY_IP);
        expected.extend_from_slice(&GUEST_MAC);
        expected.extend_from_slice(&GUEST_IP);
        assert_eq!(reply, expected);
        assert_eq!(reply.len(), ARP_FRAME_LEN);
    }

    #[test]
    fn arp_reply_is_addressed_to_the_requester_not_broadcast() {
        let mut request = arp_frame(ARP_REQUEST_OPCODE, GATEWAY_IP);
        let other_mac = [0x02, 0, 0, 0, 0, 0x99];
        request[ETH_SRC_MAC].copy_from_slice(&other_mac);
        request[ARP_SENDER].copy_from_slice(&[0x02, 0, 0, 0, 0, 0x99, 10, 0, 2, 77]);
        let reply = handle(&request).unwrap();
        assert_eq!(reply[ETH_DST_MAC], other_mac);
        assert_eq!(reply[32..38], other_mac);
        assert_eq!(reply[38..42], [10, 0, 2, 77]);
    }

    #[test]
    fn arp_request_for_other_ip_is_ignored() {
        assert_eq!(handle(&arp_frame(ARP_REQUEST_OPCODE, GUEST_IP)), None);
        assert_eq!(handle(&arp_frame(ARP_REQUEST_OPCODE, OTHER_IP)), None);
        assert_eq!(handle(&arp_frame(ARP_REQUEST_OPCODE, [10, 0, 2, 3])), None);
    }

    #[test]
    fn arp_reply_opcode_is_ignored() {
        assert_eq!(handle(&arp_frame(ARP_REPLY_OPCODE, GATEWAY_IP)), None);
    }

    #[test]
    fn arp_frame_one_byte_short_is_ignored() {
        let frame = arp_frame(ARP_REQUEST_OPCODE, GATEWAY_IP);
        assert!(handle(&frame).is_some());
        assert_eq!(handle(&frame[..frame.len() - 1]), None);
    }

    #[test]
    fn arp_frame_with_trailing_padding_is_still_answered() {
        let mut frame = arp_frame(ARP_REQUEST_OPCODE, GATEWAY_IP);
        frame.resize(60, 0);
        assert_eq!(handle(&frame).unwrap().len(), ARP_FRAME_LEN);
    }

    #[test]
    fn echo_request_to_gateway_gets_exact_reply() {
        let request = echo_request(GATEWAY_IP, b"hello");
        let reply = handle(&request).unwrap();
        assert_eq!(reply.len(), request.len());
        assert_eq!(reply[ETH_DST_MAC], GUEST_MAC);
        assert_eq!(reply[ETH_SRC_MAC], GATEWAY_MAC);
        assert_eq!(reply[ETH_TYPE], [0x08, 0x00]);
        assert_eq!(reply[IPV4_SRC], GATEWAY_IP);
        assert_eq!(reply[IPV4_DST], GUEST_IP);
        assert_eq!(reply[ICMP_TYPE], ICMP_ECHO_REPLY);
        assert_eq!(reply[ICMP_TYPE + 1], 0);
        // Identifier, sequence and payload are echoed unchanged.
        assert_eq!(reply[38..42], ICMP_ECHO_REQUEST_BODY);
        assert_eq!(&reply[42..], b"hello");
    }

    #[test]
    fn echo_reply_checksums_are_valid() {
        for len in [0usize, 1, 2, 3, 31, 32, 55, 56, 57] {
            let data: Vec<u8> = (0..len as u8).map(|b| b.wrapping_mul(37)).collect();
            let reply = handle(&echo_request(GATEWAY_IP, &data)).unwrap();
            assert_eq!(
                reference_checksum(&reply[14..34]),
                0,
                "ip header, len {len}"
            );
            assert_eq!(reference_checksum(&reply[34..]), 0, "icmp, len {len}");
        }
    }

    #[test]
    fn echo_request_to_other_address_is_not_answered_by_the_loop_gateway() {
        assert_eq!(handle(&echo_request(OTHER_IP, b"x")), None);
        assert_eq!(handle(&echo_request(GUEST_IP, b"x")), None);
    }

    #[test]
    fn non_echo_icmp_to_gateway_is_ignored() {
        let frame = ipv4_frame(
            IP_PROTO_ICMP,
            GATEWAY_IP,
            &icmp_message(ICMP_DEST_UNREACHABLE, b"x"),
        );
        assert_eq!(handle(&frame), None);
        let reply = ipv4_frame(
            IP_PROTO_ICMP,
            GATEWAY_IP,
            &icmp_message(ICMP_ECHO_REPLY, b"x"),
        );
        assert_eq!(handle(&reply), None);
    }

    #[test]
    fn udp_and_tcp_to_gateway_are_ignored() {
        let payload = icmp_message(ICMP_ECHO_REQUEST, b"abcd");
        assert_eq!(
            handle(&ipv4_frame(IP_PROTO_UDP, GATEWAY_IP, &payload)),
            None
        );
        assert_eq!(
            handle(&ipv4_frame(IP_PROTO_TCP, GATEWAY_IP, &payload)),
            None
        );
    }

    #[test]
    fn unknown_ethertype_is_ignored() {
        let mut frame = echo_request(GATEWAY_IP, b"x");
        frame[ETH_TYPE].copy_from_slice(&[0x86, 0xdd]);
        assert_eq!(handle(&frame), None);
    }

    #[test]
    fn frames_shorter_than_an_ethernet_header_are_ignored() {
        for len in 0..ETH_HEADER_LEN {
            assert_eq!(handle(&vec![0xffu8; len]), None, "len {len}");
        }
    }

    #[test]
    fn ipv4_frame_of_exactly_34_bytes_does_not_panic() {
        // Regression: the ICMP branch used to be guarded with `len >= 34` and then read byte 34.
        let mut frame = ipv4_frame(IP_PROTO_ICMP, GATEWAY_IP, &[]);
        assert_eq!(frame.len(), 34);
        assert_eq!(handle(&frame), None);
        frame.push(ICMP_ECHO_REQUEST);
        assert_eq!(handle(&frame), None, "35 bytes: no checksum field yet");
        frame.push(0);
        assert_eq!(handle(&frame), None, "36 bytes");
        frame.push(0);
        assert_eq!(handle(&frame), None, "37 bytes");
    }

    #[test]
    fn minimal_38_byte_echo_request_is_answered() {
        let mut frame = ipv4_frame(IP_PROTO_ICMP, GATEWAY_IP, &[ICMP_ECHO_REQUEST, 0, 0, 0]);
        assert_eq!(frame.len(), ICMP_FRAME_MIN_LEN);
        let checksum = reference_checksum(&frame[34..]);
        frame[36..38].copy_from_slice(&checksum.to_be_bytes());
        let reply = handle(&frame).unwrap();
        assert_eq!(reply.len(), 38);
        assert_eq!(reply[ICMP_TYPE], ICMP_ECHO_REPLY);
        assert_eq!(reference_checksum(&reply[34..]), 0);
    }

    #[test]
    fn every_truncation_of_a_valid_echo_request_is_safe() {
        let frame = echo_request(GATEWAY_IP, b"truncate me");
        for len in 0..=frame.len() {
            let answered = handle(&frame[..len]).is_some();
            assert_eq!(answered, len >= ICMP_FRAME_MIN_LEN, "len {len}");
        }
    }

    #[test]
    fn every_truncation_of_a_valid_arp_request_is_safe() {
        let frame = arp_frame(ARP_REQUEST_OPCODE, GATEWAY_IP);
        for len in 0..=frame.len() {
            assert_eq!(
                handle(&frame[..len]).is_some(),
                len >= ARP_FRAME_LEN,
                "len {len}"
            );
        }
    }

    #[test]
    fn ethernet_padding_does_not_corrupt_the_echo_checksum() {
        let mut frame = echo_request(GATEWAY_IP, b"");
        frame.resize(60, 0);
        let reply = handle(&frame).unwrap();
        assert_eq!(reply.len(), 14 + 20 + 8, "padding is dropped");
        assert_eq!(reference_checksum(&reply[34..]), 0);
    }

    #[test]
    fn bogus_ip_total_length_does_not_panic() {
        let mut frame = echo_request(GATEWAY_IP, b"payload");
        for total in [0u16, 1, 19, 20, 27, 28, 29, 1000, u16::MAX] {
            frame[IPV4_TOTAL_LEN].copy_from_slice(&total.to_be_bytes());
            let reply = handle(&frame).unwrap();
            assert!(reply.len() >= ICMP_FRAME_MIN_LEN && reply.len() <= frame.len());
        }
    }

    #[test]
    fn random_frames_never_panic() {
        let mut rng = XorShift::new(FUZZ_SEED);
        for _ in 0..FUZZ_ITERATIONS {
            let len = rng.below(FUZZ_MAX_LEN);
            let mut frame = rng.bytes(len);
            // Steer a share of the frames into the interesting branches.
            match rng.below(4) {
                0 if len >= ETH_HEADER_LEN => frame[ETH_TYPE].copy_from_slice(&[0x08, 0x06]),
                1 if len >= ETH_HEADER_LEN => frame[ETH_TYPE].copy_from_slice(&[0x08, 0x00]),
                _ => {}
            }
            if len > IPV4_DST.end {
                if rng.below(2) == 0 {
                    frame[IPV4_DST].copy_from_slice(&GATEWAY_IP);
                }
                if rng.below(2) == 0 {
                    frame[IPV4_PROTOCOL] = IP_PROTO_ICMP;
                }
            }
            if len > ICMP_TYPE && rng.below(2) == 0 {
                frame[ICMP_TYPE] = ICMP_ECHO_REQUEST;
            }
            if len > ARP_TARGET_IP.end - 1 && rng.below(2) == 0 {
                frame[ARP_OPCODE].copy_from_slice(&ARP_REQUEST);
                frame[ARP_TARGET_IP].copy_from_slice(&GATEWAY_IP);
            }
            let _ = handle(&frame);
        }
    }

    #[test]
    fn random_frames_only_get_replies_when_addressed_to_the_gateway() {
        let mut rng = XorShift::new(FUZZ_SEED + 1);
        for _ in 0..FUZZ_ITERATIONS {
            let len = rng.below(FUZZ_MAX_LEN) + 40;
            let mut frame = rng.bytes(len);
            frame[ETH_TYPE].copy_from_slice(&[0x08, 0x00]);
            frame[IPV4_PROTOCOL] = IP_PROTO_ICMP;
            frame[ICMP_TYPE] = ICMP_ECHO_REQUEST;
            if frame[IPV4_DST] == GATEWAY_IP {
                continue;
            }
            assert_eq!(handle(&frame), None);
        }
    }
}
