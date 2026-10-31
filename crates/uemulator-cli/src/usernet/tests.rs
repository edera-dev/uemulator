use std::collections::HashSet;
use std::net::{TcpListener, UdpSocket};

use super::*;
use crate::test_support::{XorShift, reference_checksum};

const FUZZ_SEED: u64 = 0x5eed_0002;
const IO_TIMEOUT: Duration = Duration::from_secs(5);
const OTHER_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x99];
const ARP_REQUEST_FRAME_LEN: usize = ETH_HEADER_LEN + ARP_PACKET_LEN;
const ICMP_DEST_UNREACHABLE: u8 = 3;
const IP_PROTO_GRE: u8 = 47;
const DHCP_DECLINE: u8 = 4;
const DHCP_RELEASE: u8 = 7;
const DHCP_INFORM: u8 = 8;
const UNICAST_OFFER_SERVER: [u8; 4] = [10, 0, 2, 99];
const LEASE_SECONDS_BIG_ENDIAN: [u8; 4] = [0x00, 0x01, 0x51, 0x80];

fn guest_ip_frame(dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let p = ipv4_packet(1, GUEST_IP, dst, proto, payload);
    ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x0800, &p)
}

fn udp_frame(sport: u16, dst: [u8; 4], dport: u16, data: &[u8]) -> Vec<u8> {
    guest_ip_frame(
        dst,
        IP_PROTO_UDP,
        &udp_segment(GUEST_IP, dst, sport, dport, data),
    )
}

fn arp_request_frame(sender_mac: [u8; 6], sender_ip: [u8; 4], target_ip: [u8; 4]) -> Vec<u8> {
    let mut arp = ARP_ETHERNET_IPV4.to_vec();
    arp.extend_from_slice(&ARP_OP_REQUEST.to_be_bytes());
    arp.extend_from_slice(&sender_mac);
    arp.extend_from_slice(&sender_ip);
    arp.extend_from_slice(&[0; 6]);
    arp.extend_from_slice(&target_ip);
    ethernet_frame(BROADCAST_MAC, sender_mac, ETHERTYPE_ARP, &arp)
}

fn echo_request_payload(identifier: u16, sequence: u16, data: &[u8]) -> Vec<u8> {
    let mut echo = vec![ICMP_ECHO_REQUEST, 0, 0, 0];
    echo.extend_from_slice(&identifier.to_be_bytes());
    echo.extend_from_slice(&sequence.to_be_bytes());
    echo.extend_from_slice(data);
    let checksum = reference_checksum(&echo);
    echo[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    echo
}

fn drain(un: &mut UserNet) -> Vec<Vec<u8>> {
    std::iter::from_fn(|| un.pop_frame()).collect()
}

fn wait_for(un: &mut UserNet, mut pred: impl FnMut(&[u8]) -> bool) -> Vec<u8> {
    let end = Instant::now() + IO_TIMEOUT;
    while Instant::now() < end {
        un.poll();
        while let Some(f) = un.pop_frame() {
            if pred(&f) {
                return f;
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("timed out waiting for frame");
}

/// A loopback UDP port that nothing listens on, for tests that only need somewhere to send to.
fn closed_udp_addr() -> SocketAddr {
    UdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn loopback_udp_server() -> (UdpSocket, u16) {
    let server = UdpSocket::bind("127.0.0.1:0").unwrap();
    server.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    let port = server.local_addr().unwrap().port();
    (server, port)
}

/// What `emit_ip` produced for the guest, after checking everything the guest's IP stack would.
struct Reassembled {
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    id: u16,
    payload: Vec<u8>,
    fragments: usize,
}

fn reassemble(frames: &[Vec<u8>], guest_mac: [u8; 6]) -> Reassembled {
    assert!(!frames.is_empty());
    let mut payload = Vec::new();
    let mut first: Option<(u16, [u8; 4], [u8; 4], u8)> = None;
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame[..6], guest_mac, "destination MAC");
        assert_eq!(frame[6..12], GATEWAY_MAC, "source MAC");
        assert_eq!(frame[ETH_TYPE], ETHERTYPE_IPV4.to_be_bytes());
        let ip = &frame[ETH_HEADER_LEN..];
        assert_eq!(ip[0], 0x45);
        assert_eq!(
            reference_checksum(&ip[..IPV4_HEADER_LEN]),
            0,
            "header checksum, fragment {index}"
        );
        let total_len = be_u16(&ip[2..4]) as usize;
        assert_eq!(ip.len(), total_len, "no padding, fragment {index}");
        assert!(
            total_len <= ETHERNET_MTU,
            "fragment {index} exceeds the MTU"
        );
        let id = be_u16(&ip[4..6]);
        let flags_offset = be_u16(&ip[6..8]);
        let src: [u8; 4] = ip[12..16].try_into().unwrap();
        let dst: [u8; 4] = ip[16..20].try_into().unwrap();
        assert_eq!(ip[8], IPV4_TTL);
        let identity = (id, src, dst, ip[9]);
        assert_eq!(
            *first.get_or_insert(identity),
            identity,
            "fragments disagree"
        );
        let more = flags_offset & IPV4_FLAG_MORE_FRAGMENTS != 0;
        let dont_fragment = flags_offset & IPV4_FLAG_DONT_FRAGMENT != 0;
        let offset = (flags_offset & 0x1fff) as usize * IPV4_FRAGMENT_OFFSET_UNIT;
        assert_eq!(offset, payload.len(), "fragment {index} offset");
        let last = index + 1 == frames.len();
        assert_eq!(more, !last, "More Fragments on fragment {index}");
        if frames.len() > 1 {
            assert!(!dont_fragment, "a fragment must not carry DF");
        } else {
            assert!(dont_fragment, "an unfragmented packet keeps DF");
        }
        if !last {
            assert_eq!(total_len - IPV4_HEADER_LEN, IPV4_FRAGMENT_PAYLOAD_MAX);
        }
        payload.extend_from_slice(&ip[IPV4_HEADER_LEN..]);
    }
    let (id, src, dst, proto) = first.unwrap();
    Reassembled {
        src,
        dst,
        proto,
        id,
        payload,
        fragments: frames.len(),
    }
}

fn splitmix_bytes(seed: u64, len: usize) -> Vec<u8> {
    XorShift::new(seed).bytes(len)
}

mod checksum_and_codecs {
    use super::*;

    #[test]
    fn checksum_of_nothing_is_all_ones() {
        assert_eq!(internet_checksum(&[]), 0xffff);
    }

    #[test]
    fn checksum_matches_the_rfc_1071_worked_example() {
        // RFC 1071 section 3: the words sum to 0xddf2 before complementing.
        let data = [0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        assert_eq!(internet_checksum(&data), !0xddf2u16);
        assert_eq!(internet_checksum(&data), 0x220d);
    }

    #[test]
    fn odd_trailing_byte_is_padded_on_the_right_with_zero() {
        assert_eq!(internet_checksum(&[0x01]), 0xfeff);
        assert_eq!(
            internet_checksum(&[0x12, 0x34, 0x56]),
            !(0x1234u16 + 0x5600)
        );
        assert_eq!(internet_checksum(&[0xab]), internet_checksum(&[0xab, 0x00]));
    }

    #[test]
    fn carries_are_folded_back_in() {
        // 0xffff + 0xff00 = 0x1feff -> 0xfeff + 1 = 0xff00.
        assert_eq!(internet_checksum(&[0xff, 0xff, 0xff]), 0x00ff);
        assert_eq!(internet_checksum(&[0xff, 0xff, 0x00, 0x01]), 0xfffe);
        assert_eq!(internet_checksum(&[0xff, 0xff, 0xff, 0xff]), 0x0000);
    }

    #[test]
    fn ones_complement_sum_of_all_ones_is_all_ones() {
        for words in [1usize, 2, 3, 100, 4096, 32767] {
            assert_eq!(
                internet_checksum(&vec![0xff; words * 2]),
                0,
                "{words} words"
            );
        }
    }

    #[test]
    fn largest_packet_does_not_overflow_the_accumulator() {
        let data = vec![0xff; 65535];
        assert_eq!(internet_checksum(&data), reference_checksum(&data));
    }

    #[test]
    fn checksum_agrees_with_an_independent_implementation() {
        let mut rng = XorShift::new(FUZZ_SEED);
        for len in 0..300 {
            let data = rng.bytes(len);
            assert_eq!(
                internet_checksum(&data),
                reference_checksum(&data),
                "len {len}"
            );
        }
    }

    #[test]
    fn a_packet_with_its_checksum_inserted_sums_to_zero() {
        let mut rng = XorShift::new(FUZZ_SEED + 1);
        for words in 2..64 {
            let mut data = rng.bytes(words * 2);
            data[2..4].copy_from_slice(&[0, 0]);
            let checksum = internet_checksum(&data);
            data[2..4].copy_from_slice(&checksum.to_be_bytes());
            assert_eq!(internet_checksum(&data), 0, "{words} words");
        }
    }

    #[test]
    fn running_sum_can_be_split_at_an_even_boundary() {
        let data = splitmix_bytes(7, 101);
        let (head, tail) = data.split_at(40);
        let split = checksum_finish(checksum_add(checksum_add(0, head), tail));
        assert_eq!(split, internet_checksum(&data));
    }

    #[test]
    fn finish_complements_after_folding() {
        assert_eq!(checksum_finish(0), 0xffff);
        assert_eq!(checksum_finish(0xffff), 0);
        assert_eq!(checksum_finish(0x1_0000), 0xfffe);
        assert_eq!(checksum_finish(0x2_fffd), 0);
    }

    #[test]
    fn transport_checksum_covers_the_pseudo_header() {
        let segment = [0x12, 0x34, 0x00, 0x50, 0xde, 0xad, 0xbe];
        let mut pseudo = Vec::new();
        pseudo.extend_from_slice(&GUEST_IP);
        pseudo.extend_from_slice(&[8, 8, 8, 8]);
        pseudo.extend_from_slice(&[0, IP_PROTO_TCP]);
        pseudo.extend_from_slice(&(segment.len() as u16).to_be_bytes());
        pseudo.extend_from_slice(&segment);
        assert_eq!(
            transport_checksum(GUEST_IP, [8, 8, 8, 8], IP_PROTO_TCP, &segment),
            reference_checksum(&pseudo)
        );
    }

    #[test]
    fn transport_checksum_depends_on_addresses_protocol_and_length() {
        let segment = [1u8, 2, 3, 4];
        let base = transport_checksum(GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &segment);
        assert_ne!(
            base,
            transport_checksum(GATEWAY_IP, [10, 0, 2, 16], IP_PROTO_UDP, &segment)
        );
        assert_ne!(
            base,
            transport_checksum(GUEST_IP, GATEWAY_IP, IP_PROTO_TCP, &segment)
        );
        assert_ne!(
            base,
            transport_checksum(GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &segment[..3])
        );
        // Swapping source and destination keeps the sum: addition commutes.
        assert_eq!(
            base,
            transport_checksum(GATEWAY_IP, GUEST_IP, IP_PROTO_UDP, &segment)
        );
    }

    #[test]
    fn big_endian_readers() {
        assert_eq!(be_u16(&[0x12, 0x34]), 0x1234);
        assert_eq!(be_u16(&[0x12, 0x34, 0x56]), 0x1234);
        assert_eq!(be_u32(&[0x12, 0x34, 0x56, 0x78]), 0x1234_5678);
        assert_eq!(be_u32(&[0xff, 0xff, 0xff, 0xff]), u32::MAX);
    }

    #[test]
    fn ethernet_frame_layout() {
        let frame = ethernet_frame(
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            0x0800,
            &[0xaa, 0xbb],
        );
        assert_eq!(
            frame,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0x08, 0x00, 0xaa, 0xbb
            ]
        );
    }

    #[test]
    fn ethernet_frame_of_empty_payload_is_just_the_header() {
        assert_eq!(
            ethernet_frame([0; 6], [0; 6], 0x86dd, &[]).len(),
            ETH_HEADER_LEN
        );
    }

    #[test]
    fn ipv4_packet_has_the_exact_header_layout() {
        let packet = ipv4_packet(
            0x1234,
            [10, 0, 2, 2],
            [10, 0, 2, 15],
            IP_PROTO_UDP,
            &[0xde, 0xad],
        );
        let mut expected = vec![
            0x45, 0x00, // version/IHL, TOS
            0x00, 0x16, // total length 22
            0x12, 0x34, // identification
            0x40, 0x00, // DF, no fragment offset
            0x40, 0x11, // TTL 64, UDP
            0x00, 0x00, // checksum placeholder
            10, 0, 2, 2, // source
            10, 0, 2, 15, // destination
        ];
        let checksum = reference_checksum(&expected);
        expected[10..12].copy_from_slice(&checksum.to_be_bytes());
        expected.extend_from_slice(&[0xde, 0xad]);
        assert_eq!(packet, expected);
        assert_eq!(reference_checksum(&packet[..IPV4_HEADER_LEN]), 0);
    }

    #[test]
    fn ipv4_header_checksum_known_vector() {
        // The header from the well-known Wikipedia IPv4 checksum example (id 0x1c46, 40 bytes of
        // payload, TCP, 172.16.10.99 -> 172.16.10.12) has checksum 0xb1e6 (with DF set).
        let packet = ipv4_packet(
            0x1c46,
            [172, 16, 10, 99],
            [172, 16, 10, 12],
            IP_PROTO_TCP,
            &[0; 40],
        );
        assert_eq!(packet[IPV4_CHECKSUM], [0xb1, 0xe6]);
    }

    #[test]
    fn ipv4_datagram_encodes_flags_and_offset_verbatim() {
        let packet = ipv4_datagram(
            7,
            GUEST_IP,
            GATEWAY_IP,
            IP_PROTO_ICMP,
            0x2000 | 185,
            &[0; 8],
        );
        assert_eq!(packet[IPV4_FLAGS_FRAGMENT], [0x20, 185]);
        assert_eq!(reference_checksum(&packet[..IPV4_HEADER_LEN]), 0);
    }

    #[test]
    fn ipv4_packet_total_length_counts_header_and_payload() {
        for len in [0usize, 1, 7, 1480] {
            let packet = ipv4_packet(1, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &vec![0; len]);
            assert_eq!(
                be_u16(&packet[IPV4_TOTAL_LEN]) as usize,
                IPV4_HEADER_LEN + len
            );
            assert_eq!(packet.len(), IPV4_HEADER_LEN + len);
        }
    }

    #[test]
    fn mtu_constants_are_consistent() {
        assert_eq!(IPV4_FRAGMENT_PAYLOAD_MAX, 1480);
        assert_eq!(IPV4_FRAGMENT_PAYLOAD_MAX % IPV4_FRAGMENT_OFFSET_UNIT, 0);
        assert_eq!(MSS + TCP_HEADER_LEN + IPV4_HEADER_LEN, ETHERNET_MTU);
        assert_eq!(IPV4_PAYLOAD_LEN_MAX, 65515);
    }

    #[test]
    fn payload_up_to_one_mtu_is_a_single_packet_with_df() {
        for len in [0usize, 1, 100, 1479, 1480] {
            let packets = ipv4_packets(9, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &vec![0x5a; len]);
            assert_eq!(packets.len(), 1, "len {len}");
            assert_eq!(
                be_u16(&packets[0][IPV4_FLAGS_FRAGMENT]),
                IPV4_FLAG_DONT_FRAGMENT
            );
            assert_eq!(packets[0].len(), IPV4_HEADER_LEN + len);
        }
    }

    #[test]
    fn payload_one_byte_over_the_mtu_splits_into_two_fragments() {
        let payload: Vec<u8> = (0..1481u32).map(|i| i as u8).collect();
        let packets = ipv4_packets(0xabcd, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &payload);
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].len(), ETHERNET_MTU);
        assert_eq!(packets[1].len(), IPV4_HEADER_LEN + 1);
        assert_eq!(
            be_u16(&packets[0][IPV4_FLAGS_FRAGMENT]),
            IPV4_FLAG_MORE_FRAGMENTS
        );
        assert_eq!(be_u16(&packets[1][IPV4_FLAGS_FRAGMENT]), 185);
        for packet in &packets {
            assert_eq!(be_u16(&packet[4..6]), 0xabcd);
            assert_eq!(reference_checksum(&packet[..IPV4_HEADER_LEN]), 0);
        }
        assert_eq!(packets[1][IPV4_HEADER_LEN], payload[1480]);
    }

    #[test]
    fn fragments_reassemble_for_a_range_of_sizes() {
        let sizes = [
            1481usize,
            2959,
            2960,
            2961,
            4440,
            4441,
            9000,
            20000,
            IPV4_PAYLOAD_LEN_MAX,
        ];
        for len in sizes {
            let payload = splitmix_bytes(len as u64, len);
            let packets = ipv4_packets(3, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &payload);
            let frames: Vec<Vec<u8>> = packets
                .iter()
                .map(|p| ethernet_frame(GUEST_MAC, GATEWAY_MAC, ETHERTYPE_IPV4, p))
                .collect();
            let whole = reassemble(&frames, GUEST_MAC);
            assert_eq!(whole.payload, payload, "len {len}");
            assert_eq!(whole.fragments, len.div_ceil(IPV4_FRAGMENT_PAYLOAD_MAX));
            assert_eq!(
                (whole.src, whole.dst, whole.proto, whole.id),
                (GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, 3)
            );
        }
    }

    #[test]
    fn payload_that_cannot_fit_an_ipv4_packet_yields_nothing() {
        let payload = vec![0u8; IPV4_PAYLOAD_LEN_MAX + 1];
        assert!(ipv4_packets(1, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &payload).is_empty());
        let maximal = vec![0u8; IPV4_PAYLOAD_LEN_MAX];
        assert_eq!(
            ipv4_packets(1, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &maximal).len(),
            45
        );
    }

    #[test]
    fn last_fragment_offset_stays_within_thirteen_bits() {
        let maximal = vec![0u8; IPV4_PAYLOAD_LEN_MAX];
        let packets = ipv4_packets(1, GUEST_IP, GATEWAY_IP, IP_PROTO_UDP, &maximal);
        let last = packets.last().unwrap();
        let word = be_u16(&last[IPV4_FLAGS_FRAGMENT]);
        assert_eq!(word & IPV4_FLAG_MORE_FRAGMENTS, 0);
        assert_eq!(word & 0x1fff, (44 * IPV4_FRAGMENT_PAYLOAD_MAX / 8) as u16);
    }

    #[test]
    fn udp_segment_has_the_exact_layout() {
        let segment = udp_segment(GUEST_IP, [8, 8, 8, 8], 1234, 53, b"abc");
        assert_eq!(
            segment[..8],
            [0x04, 0xd2, 0x00, 0x35, 0x00, 0x0b, segment[6], segment[7]]
        );
        assert_eq!(&segment[8..], b"abc");
        assert_eq!(
            transport_checksum(GUEST_IP, [8, 8, 8, 8], IP_PROTO_UDP, &segment),
            0
        );
    }

    #[test]
    fn udp_checksum_that_computes_to_zero_is_sent_as_all_ones() {
        // Search the two data bytes for the value that makes the true checksum zero.
        let (src, dst) = (GUEST_IP, GATEWAY_IP);
        let target = (0..=u16::MAX)
            .find(|word| {
                let mut segment = vec![0, 1, 0, 2, 0, 10, 0, 0];
                segment.extend_from_slice(&word.to_be_bytes());
                transport_checksum(src, dst, IP_PROTO_UDP, &segment) == 0
            })
            .expect("some data word yields checksum 0");
        let segment = udp_segment(src, dst, 1, 2, &target.to_be_bytes());
        assert_eq!(segment[UDP_CHECKSUM], [0xff, 0xff]);
        assert_eq!(transport_checksum(src, dst, IP_PROTO_UDP, &segment), 0);
    }

    #[test]
    fn udp_segment_of_odd_length_data_validates() {
        for len in [0usize, 1, 2, 3, 1471] {
            let data = splitmix_bytes(len as u64 + 100, len);
            let segment = udp_segment(GATEWAY_IP, GUEST_IP, 67, 68, &data);
            assert_eq!(be_u16(&segment[UDP_LENGTH]) as usize, 8 + len);
            assert_eq!(
                transport_checksum(GATEWAY_IP, GUEST_IP, IP_PROTO_UDP, &segment),
                0,
                "len {len}"
            );
        }
    }

    fn tcp_header(flags: u8, mss: bool) -> TcpHeader {
        TcpHeader {
            src: [93, 184, 216, 34],
            dst: GUEST_IP,
            sport: 443,
            dport: 51000,
            seq: 0x0102_0304,
            ack: 0xa0b0_c0d0,
            flags,
            window: 0x7000,
            advertise_mss: mss,
        }
    }

    #[test]
    fn tcp_segment_without_options_has_the_exact_layout() {
        let segment = tcp_segment(&tcp_header(ACK | PSH, false), b"hi");
        assert_eq!(
            segment[..16],
            [
                0x01,
                0xbb, // source port 443
                0xc7,
                0x38, // destination port 51000
                0x01,
                0x02,
                0x03,
                0x04, // seq
                0xa0,
                0xb0,
                0xc0,
                0xd0, // ack
                0x50, // data offset 5
                ACK | PSH,
                0x70,
                0x00, // flags, window
            ]
        );
        assert_eq!(segment[18..20], [0, 0], "urgent pointer");
        assert_eq!(&segment[20..], b"hi");
        assert_eq!(
            transport_checksum([93, 184, 216, 34], GUEST_IP, IP_PROTO_TCP, &segment),
            0
        );
    }

    #[test]
    fn tcp_syn_segment_carries_the_mss_option() {
        let segment = tcp_segment(&tcp_header(SYN | ACK, true), &[]);
        assert_eq!(segment[TCP_DATA_OFFSET], 0x60);
        assert_eq!(segment[20..24], [TCP_OPT_MSS, 4, 0x05, 0xb4]);
        assert_eq!(segment.len(), 24);
        assert_eq!(
            transport_checksum([93, 184, 216, 34], GUEST_IP, IP_PROTO_TCP, &segment),
            0
        );
    }

    #[test]
    fn tcp_checksum_validates_for_odd_and_even_payloads() {
        for len in 0..9 {
            let data = vec![0xa5; len];
            let segment = tcp_segment(&tcp_header(ACK, false), &data);
            assert_eq!(
                transport_checksum([93, 184, 216, 34], GUEST_IP, IP_PROTO_TCP, &segment),
                0,
                "len {len}"
            );
        }
    }

    #[test]
    fn tcp_segment_parse_round_trips() {
        let segment = tcp_segment(&tcp_header(SYN | ACK, true), b"payload");
        let parsed = TcpSegment::parse(&segment).unwrap();
        assert_eq!((parsed.sport, parsed.dport), (443, 51000));
        assert_eq!((parsed.seq, parsed.ack), (0x0102_0304, 0xa0b0_c0d0));
        assert_eq!(parsed.flags, SYN | ACK);
        assert_eq!(parsed.window, 0x7000);
        assert_eq!(parsed.options, [TCP_OPT_MSS, 4, 0x05, 0xb4]);
        assert_eq!(parsed.data, b"payload");
    }

    #[test]
    fn tcp_segment_shorter_than_a_header_is_rejected() {
        let segment = tcp_segment(&tcp_header(ACK, false), &[]);
        for len in 0..TCP_HEADER_LEN {
            assert!(TcpSegment::parse(&segment[..len]).is_none(), "len {len}");
        }
        assert!(TcpSegment::parse(&segment).is_some());
    }

    #[test]
    fn tcp_data_offset_must_be_between_header_size_and_segment_size() {
        let mut segment = tcp_segment(&tcp_header(ACK, false), b"data");
        for words in 0..5u8 {
            segment[TCP_DATA_OFFSET] = words << 4;
            assert!(TcpSegment::parse(&segment).is_none(), "{words} words");
        }
        segment[TCP_DATA_OFFSET] = 6 << 4;
        let parsed = TcpSegment::parse(&segment).unwrap();
        assert_eq!(
            parsed.options, b"data",
            "the option space takes the four payload bytes"
        );
        assert!(parsed.data.is_empty());
        segment[TCP_DATA_OFFSET] = 7 << 4;
        assert!(
            TcpSegment::parse(&segment).is_none(),
            "28 bytes exceed the 24-byte segment"
        );
        segment[TCP_DATA_OFFSET] = 15 << 4;
        assert!(TcpSegment::parse(&segment).is_none());
    }

    #[test]
    fn mss_defaults_to_ours_without_options() {
        assert_eq!(negotiated_mss(&[]), MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_NOP, TCP_OPT_NOP]), MSS);
    }

    #[test]
    fn mss_is_lowered_to_what_the_guest_asks_for() {
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0x02, 0x18]), 536);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0x05, 0xb4]), 1460);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0x04, 0x00]), 1024);
    }

    #[test]
    fn mss_is_never_raised_above_ours() {
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0x23, 0x28]), MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0xff, 0xff]), MSS);
    }

    #[test]
    fn tiny_or_zero_mss_is_clamped_to_the_floor() {
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0, 0]), MIN_MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0, 1]), MIN_MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0, 63]), MIN_MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0, 64]), 64);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 4, 0, 65]), 65);
    }

    #[test]
    fn mss_is_found_after_other_options_and_padding() {
        let options = [
            TCP_OPT_NOP,
            3,
            3,
            7, // window scale (kind 3, length 3) preceded by a NOP
            4,
            2, // SACK permitted
            TCP_OPT_MSS,
            4,
            0x02,
            0x00,
        ];
        assert_eq!(negotiated_mss(&options), 512);
    }

    #[test]
    fn option_list_end_stops_the_scan() {
        assert_eq!(
            negotiated_mss(&[TCP_OPT_END, 0, TCP_OPT_MSS, 4, 0x02, 0x00]),
            MSS
        );
    }

    #[test]
    fn malformed_options_stop_the_scan_without_panicking() {
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS]), MSS);
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 0, 2, 0]), MSS, "length 0");
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 1, 2, 0]), MSS, "length 1");
        assert_eq!(
            negotiated_mss(&[TCP_OPT_MSS, 4, 0x02]),
            MSS,
            "truncated value"
        );
        assert_eq!(
            negotiated_mss(&[9, 200, 1, 2]),
            MSS,
            "length beyond the buffer"
        );
    }

    #[test]
    fn mss_option_with_a_wrong_length_is_ignored_but_scanning_continues() {
        assert_eq!(
            negotiated_mss(&[TCP_OPT_MSS, 3, 0x02, TCP_OPT_MSS, 4, 0x02, 0x00]),
            512
        );
        assert_eq!(negotiated_mss(&[TCP_OPT_MSS, 3, 0x02]), MSS);
    }

    #[test]
    fn the_smallest_of_several_mss_options_wins() {
        let options = [TCP_OPT_MSS, 4, 0x04, 0x00, TCP_OPT_MSS, 4, 0x02, 0x00];
        assert_eq!(negotiated_mss(&options), 512);
    }

    #[test]
    fn sequence_comparison_is_modular() {
        assert!(seq_before(1, 2));
        assert!(!seq_before(2, 1));
        assert!(!seq_before(5, 5));
        assert!(seq_before(0, 1));
        assert!(!seq_before(1, 0));
    }

    #[test]
    fn sequence_comparison_survives_wraparound() {
        assert!(seq_before(u32::MAX, 0));
        assert!(!seq_before(0, u32::MAX));
        assert!(seq_before(0xffff_fff0, 5));
        assert!(!seq_before(5, 0xffff_fff0));
        assert!(seq_before(u32::MAX - 1, u32::MAX));
        assert!(seq_before(0x7fff_ffff, 0x8000_0000));
    }

    #[test]
    fn sequence_comparison_covers_a_half_space_window() {
        let base = 0xf000_0000u32;
        for distance in [1u32, 2, 1000, 0x7fff_fffe, 0x7fff_ffff] {
            assert!(seq_before(base, base.wrapping_add(distance)), "{distance}");
            assert!(!seq_before(base.wrapping_add(distance), base), "{distance}");
        }
    }

    #[test]
    fn advertised_window_shrinks_with_the_backlog() {
        assert_eq!(advertised_window(0), 65535);
        assert_eq!(advertised_window(1), 65534);
        assert_eq!(advertised_window(1460), 65535 - 1460);
        assert_eq!(advertised_window(65534), 1);
        assert_eq!(advertised_window(65535), 0);
    }

    #[test]
    fn advertised_window_never_wraps_below_zero() {
        assert_eq!(advertised_window(65536), 0);
        assert_eq!(advertised_window(1_000_000), 0);
        assert_eq!(advertised_window(usize::MAX), 0);
    }

    fn key(guest_port: u16, remote_ip: [u8; 4], remote_port: u16) -> FlowKey {
        FlowKey {
            guest_port,
            remote_ip,
            remote_port,
        }
    }

    #[test]
    fn flow_keys_compare_field_by_field() {
        let base = key(1000, [1, 2, 3, 4], 80);
        assert_eq!(base, key(1000, [1, 2, 3, 4], 80));
        assert_ne!(base, key(1001, [1, 2, 3, 4], 80));
        assert_ne!(base, key(1000, [1, 2, 3, 5], 80));
        assert_ne!(base, key(1000, [1, 2, 3, 4], 81));
    }

    #[test]
    fn flow_keys_hash_consistently_in_collections() {
        let mut set = HashSet::new();
        assert!(set.insert(key(1000, [1, 2, 3, 4], 80)));
        assert!(!set.insert(key(1000, [1, 2, 3, 4], 80)));
        assert!(set.insert(key(1000, [1, 2, 3, 4], 81)));
        assert!(set.insert(key(1000, [4, 3, 2, 1], 80)));
        assert!(
            set.insert(key(80, [1, 2, 3, 4], 1000)),
            "ports are not interchangeable"
        );
        assert_eq!(set.len(), 4);
        assert!(set.contains(&key(80, [1, 2, 3, 4], 1000)));
    }

    #[test]
    fn flow_keys_hash_the_same_when_equal() {
        use std::hash::{BuildHasher, RandomState};
        let state = RandomState::new();
        let a = key(1, [9, 9, 9, 9], 2);
        let b = key(1, [9, 9, 9, 9], 2);
        assert_eq!(state.hash_one(a), state.hash_one(b));
    }

    #[test]
    fn flow_keys_order_by_guest_port_then_address_then_remote_port() {
        assert!(key(1, [9, 9, 9, 9], 9) < key(2, [0, 0, 0, 0], 0));
        assert!(key(1, [1, 0, 0, 0], 9) < key(1, [2, 0, 0, 0], 0));
        assert!(key(1, [1, 0, 0, 0], 1) < key(1, [1, 0, 0, 0], 2));
    }

    #[test]
    fn the_gateway_address_stands_for_host_loopback() {
        assert_eq!(
            host_addr(GATEWAY_IP, 8080),
            SocketAddr::from(([127, 0, 0, 1], 8080))
        );
        assert_eq!(
            host_addr([93, 184, 216, 34], 443),
            SocketAddr::from(([93, 184, 216, 34], 443))
        );
        assert_eq!(host_addr(DNS_IP, 53), SocketAddr::from((DNS_IP, 53)));
        assert_eq!(host_addr(GUEST_IP, 1), SocketAddr::from((GUEST_IP, 1)));
    }

    #[test]
    fn first_nameserver_is_the_first_ipv4_one() {
        let conf = "search example.com\nnameserver 192.168.1.1\nnameserver 8.8.8.8\n";
        assert_eq!(
            first_ipv4_nameserver(conf),
            Some(Ipv4Addr::new(192, 168, 1, 1))
        );
    }

    #[test]
    fn ipv6_comments_and_junk_nameservers_are_skipped() {
        let conf = "# nameserver 9.9.9.9\nnameserver fe80::1\nnameserver not-an-ip\nnameserver   1.1.1.1  \n";
        assert_eq!(first_ipv4_nameserver(conf), Some(Ipv4Addr::new(1, 1, 1, 1)));
    }

    #[test]
    fn no_usable_nameserver_means_none() {
        assert_eq!(first_ipv4_nameserver(""), None);
        assert_eq!(first_ipv4_nameserver("search lan\noptions ndots:1\n"), None);
        assert_eq!(first_ipv4_nameserver("nameserver ::1\n"), None);
        assert_eq!(first_ipv4_nameserver("nameserver\n"), None);
    }

    #[test]
    fn nameserver_line_with_carriage_return_still_parses() {
        assert_eq!(
            first_ipv4_nameserver("nameserver 10.1.1.1\r\n"),
            Some(Ipv4Addr::new(10, 1, 1, 1))
        );
    }

    #[test]
    fn system_dns_always_targets_port_53() {
        assert_eq!(system_dns().port(), DNS_PORT);
        assert!(system_dns().is_ipv4());
    }

    #[test]
    fn constants_describe_the_documented_network() {
        assert_eq!(GATEWAY_IP, [10, 0, 2, 2]);
        assert_eq!(GATEWAY_MAC, [0x52, 0x55, 10, 0, 2, 2]);
        assert_eq!(DNS_IP, [10, 0, 2, 3]);
        assert_eq!(GUEST_IP, [10, 0, 2, 15]);
        assert_eq!(SUBNET_BROADCAST, [10, 0, 2, 255]);
        assert_eq!(SUBNET_MASK, [255, 255, 255, 0]);
        assert_eq!(DHCP_LEASE_SECS, 86_400);
        assert_eq!(MAX_UDP_FLOWS, 128);
        for ip in [GATEWAY_IP, DNS_IP, GUEST_IP, SUBNET_BROADCAST] {
            assert_eq!(ip[..3], SUBNET_PREFIX);
        }
    }
}

mod parsing {
    use super::*;

    fn ipv4_frame_with(header_tweak: impl FnOnce(&mut Vec<u8>), payload: &[u8]) -> Vec<u8> {
        let mut frame = guest_ip_frame(GATEWAY_IP, IP_PROTO_UDP, payload);
        header_tweak(&mut frame);
        frame
    }

    #[test]
    fn parses_an_ordinary_packet() {
        let frame = guest_ip_frame([8, 8, 4, 4], IP_PROTO_TCP, b"payload!");
        let ip = parse_ipv4(&frame).unwrap();
        assert_eq!(ip.src, GUEST_IP);
        assert_eq!(ip.dst, [8, 8, 4, 4]);
        assert_eq!(ip.proto, IP_PROTO_TCP);
        assert_eq!(ip.payload, b"payload!");
    }

    #[test]
    fn frames_shorter_than_both_headers_are_rejected() {
        let frame = guest_ip_frame(GATEWAY_IP, IP_PROTO_UDP, &[]);
        for len in 0..ETH_HEADER_LEN + IPV4_HEADER_LEN {
            assert!(parse_ipv4(&frame[..len]).is_none(), "len {len}");
        }
        assert!(parse_ipv4(&frame).is_some());
    }

    #[test]
    fn only_the_ipv4_ethertype_is_accepted() {
        for ethertype in [ETHERTYPE_ARP, 0x86dd, 0x0000, 0x0801] {
            let frame = ipv4_frame_with(
                |f| f[ETH_TYPE].copy_from_slice(&ethertype.to_be_bytes()),
                b"x",
            );
            assert!(parse_ipv4(&frame).is_none(), "{ethertype:#x}");
        }
    }

    #[test]
    fn other_ip_versions_are_rejected() {
        for version_ihl in [0x65u8, 0x05, 0x35, 0xf5] {
            let frame = ipv4_frame_with(|f| f[ETH_HEADER_LEN] = version_ihl, b"x");
            assert!(parse_ipv4(&frame).is_none(), "{version_ihl:#x}");
        }
    }

    #[test]
    fn header_length_below_five_words_is_rejected() {
        for ihl in 0..5u8 {
            let frame = ipv4_frame_with(|f| f[ETH_HEADER_LEN] = 0x40 | ihl, b"x");
            assert!(parse_ipv4(&frame).is_none(), "ihl {ihl}");
        }
    }

    #[test]
    fn ip_options_are_skipped_using_the_header_length() {
        let mut frame = guest_ip_frame(
            GATEWAY_IP,
            IP_PROTO_UDP,
            &[0xaa, 0xbb, 0xcc, 0xdd, b'o', b'k'],
        );
        frame[ETH_HEADER_LEN] = 0x46;
        let total = (IPV4_HEADER_LEN + 6) as u16;
        frame[ETH_HEADER_LEN + 2..ETH_HEADER_LEN + 4].copy_from_slice(&total.to_be_bytes());
        let ip = parse_ipv4(&frame).unwrap();
        assert_eq!(
            ip.payload, b"ok",
            "four bytes of options precede the payload"
        );
    }

    #[test]
    fn header_longer_than_the_total_length_is_rejected() {
        let frame = ipv4_frame_with(
            |f| f[ETH_HEADER_LEN] = 0x4f,
            b"0123456789012345678901234567890123456789",
        );
        assert!(
            parse_ipv4(&frame).is_some(),
            "60-byte header still fits this 60-byte packet"
        );
        let frame = ipv4_frame_with(|f| f[ETH_HEADER_LEN] = 0x4f, b"0123");
        assert!(parse_ipv4(&frame).is_none());
    }

    #[test]
    fn total_length_beyond_the_frame_is_rejected() {
        let mut frame = guest_ip_frame(GATEWAY_IP, IP_PROTO_UDP, b"abcd");
        let truncated = frame.len() - 1;
        assert!(parse_ipv4(&frame[..truncated]).is_none());
        frame[ETH_HEADER_LEN + 2..ETH_HEADER_LEN + 4].copy_from_slice(&u16::MAX.to_be_bytes());
        assert!(parse_ipv4(&frame).is_none());
    }

    #[test]
    fn total_length_shorter_than_the_header_is_rejected() {
        for total in [0u16, 1, 19] {
            let frame = ipv4_frame_with(
                |f| f[ETH_HEADER_LEN + 2..ETH_HEADER_LEN + 4].copy_from_slice(&total.to_be_bytes()),
                b"abcd",
            );
            assert!(parse_ipv4(&frame).is_none(), "{total}");
        }
    }

    #[test]
    fn ethernet_padding_after_the_ip_packet_is_cut_off() {
        let mut frame = guest_ip_frame(GATEWAY_IP, IP_PROTO_UDP, b"ab");
        frame.resize(60, 0xee);
        assert_eq!(parse_ipv4(&frame).unwrap().payload, b"ab");
    }

    #[test]
    fn fragments_are_not_parsed() {
        for flags_offset in [
            IPV4_FLAG_MORE_FRAGMENTS,
            1,
            185,
            IPV4_FLAG_MORE_FRAGMENTS | 5,
            0x1fff,
        ] {
            let frame = ipv4_frame_with(
                |f| {
                    f[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8]
                        .copy_from_slice(&flags_offset.to_be_bytes())
                },
                b"abcd",
            );
            assert!(parse_ipv4(&frame).is_none(), "{flags_offset:#x}");
        }
    }

    #[test]
    fn dont_fragment_and_zero_flags_still_parse() {
        for flags in [0u16, IPV4_FLAG_DONT_FRAGMENT] {
            let frame = ipv4_frame_with(
                |f| f[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8].copy_from_slice(&flags.to_be_bytes()),
                b"abcd",
            );
            assert!(parse_ipv4(&frame).is_some(), "{flags:#x}");
        }
    }

    #[test]
    fn the_reserved_flag_bit_is_masked_out_with_the_other_flags() {
        // 0x8000 is the reserved bit; the fragment mask ignores it, so the packet still parses.
        let frame = ipv4_frame_with(
            |f| f[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8].copy_from_slice(&0x8000u16.to_be_bytes()),
            b"abcd",
        );
        assert!(parse_ipv4(&frame).is_some());
    }

    #[test]
    fn parsing_every_truncation_of_every_frame_kind_never_panics() {
        let frames = [
            guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_TCP,
                &tcp_segment(
                    &TcpHeader {
                        src: GUEST_IP,
                        dst: GATEWAY_IP,
                        sport: 1,
                        dport: 2,
                        seq: 3,
                        ack: 4,
                        flags: SYN,
                        window: 5,
                        advertise_mss: true,
                    },
                    b"data",
                ),
            ),
            udp_frame(1, GATEWAY_IP, 2, b"data"),
            arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP),
        ];
        for frame in &frames {
            for len in 0..=frame.len() {
                if let Some(ip) = parse_ipv4(&frame[..len]) {
                    let _ = TcpSegment::parse(ip.payload);
                }
            }
        }
    }
}

mod link_and_network_layer {
    use super::*;

    #[test]
    fn a_new_stack_starts_empty_and_expects_the_default_guest_mac() {
        let mut un = UserNet::new();
        assert!(un.pop_frame().is_none());
        assert_eq!(un.guest_mac, GUEST_MAC);
        assert_eq!(un.udp_flow_limit, MAX_UDP_FLOWS);
        assert!(un.udp.is_empty() && un.tcp.is_empty());
        let default = UserNet::default();
        assert_eq!(default.guest_mac, GUEST_MAC);
    }

    #[test]
    fn requeued_frames_return_to_the_head_in_order() {
        let mut un = UserNet::new();
        un.out.push_back(vec![3]);
        un.requeue_frame(vec![2]);
        un.requeue_frame(vec![1]);
        assert_eq!(drain(&mut un), vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn frames_shorter_than_an_ethernet_header_are_ignored_and_not_learned_from() {
        let mut un = UserNet::new();
        un.input(&[0xff; ETH_HEADER_LEN - 1]);
        un.input(&[]);
        assert!(un.pop_frame().is_none());
        assert_eq!(un.guest_mac, GUEST_MAC);
    }

    #[test]
    fn arp_request_for_the_gateway_gets_the_exact_reply() {
        let mut un = UserNet::new();
        un.input(&arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP));
        let out = drain(&mut un);
        assert_eq!(out.len(), 1);
        let mut expected = Vec::new();
        expected.extend_from_slice(&GUEST_MAC);
        expected.extend_from_slice(&GATEWAY_MAC);
        expected.extend_from_slice(&[0x08, 0x06]);
        expected.extend_from_slice(&[0, 1, 8, 0, 6, 4, 0, 2]);
        expected.extend_from_slice(&GATEWAY_MAC);
        expected.extend_from_slice(&GATEWAY_IP);
        expected.extend_from_slice(&GUEST_MAC);
        expected.extend_from_slice(&GUEST_IP);
        assert_eq!(out[0], expected);
    }

    #[test]
    fn arp_for_any_other_address_in_the_subnet_resolves_to_the_gateway_mac() {
        for last in [1u8, 3, 4, 99, 254, 255, 0] {
            let target = [10, 0, 2, last];
            let mut un = UserNet::new();
            un.input(&arp_request_frame(GUEST_MAC, GUEST_IP, target));
            let out = drain(&mut un);
            if last == GUEST_IP[3] {
                assert!(out.is_empty());
                continue;
            }
            assert_eq!(out.len(), 1, "{target:?}");
            assert_eq!(out[0][22..28], GATEWAY_MAC, "sender hardware address");
            assert_eq!(
                out[0][28..32],
                target,
                "sender protocol address is the asked address"
            );
        }
    }

    #[test]
    fn arp_for_the_guests_own_address_is_not_answered() {
        let mut un = UserNet::new();
        un.input(&arp_request_frame(GUEST_MAC, [0, 0, 0, 0], GUEST_IP));
        assert!(drain(&mut un).is_empty());
    }

    #[test]
    fn arp_outside_the_subnet_is_ignored() {
        for target in [
            [10, 0, 3, 2],
            [10, 1, 2, 2],
            [192, 168, 0, 1],
            [0, 0, 0, 0],
            [255; 4],
            [127, 0, 0, 1],
        ] {
            let mut un = UserNet::new();
            un.input(&arp_request_frame(GUEST_MAC, GUEST_IP, target));
            assert!(drain(&mut un).is_empty(), "{target:?}");
        }
    }

    #[test]
    fn arp_replies_and_other_opcodes_are_ignored() {
        for opcode in [0u16, ARP_OP_REPLY, 3, 4, 0xffff] {
            let mut frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
            frame[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8].copy_from_slice(&opcode.to_be_bytes());
            let mut un = UserNet::new();
            un.input(&frame);
            assert!(drain(&mut un).is_empty(), "opcode {opcode}");
        }
    }

    #[test]
    fn truncated_arp_frames_are_ignored() {
        let frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
        for len in 0..ARP_REQUEST_FRAME_LEN {
            let mut un = UserNet::new();
            un.input(&frame[..len]);
            assert!(drain(&mut un).is_empty(), "len {len}");
        }
    }

    #[test]
    fn arp_with_ethernet_padding_is_answered() {
        let mut frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
        frame.resize(60, 0);
        let mut un = UserNet::new();
        un.input(&frame);
        assert_eq!(drain(&mut un).len(), 1);
    }

    #[test]
    fn arp_reply_goes_to_the_requesters_mac_and_echoes_its_address() {
        let mut un = UserNet::new();
        un.input(&arp_request_frame(OTHER_MAC, [10, 0, 2, 77], GATEWAY_IP));
        let reply = drain(&mut un).remove(0);
        assert_eq!(reply[..6], OTHER_MAC);
        assert_eq!(reply[32..38], OTHER_MAC);
        assert_eq!(reply[38..42], [10, 0, 2, 77]);
    }

    #[test]
    fn frames_the_stack_originates_follow_the_last_seen_guest_mac() {
        let mut un = UserNet::new();
        un.input(&arp_request_frame(OTHER_MAC, GUEST_IP, GATEWAY_IP));
        drain(&mut un);
        let mut ping = guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &echo_request_payload(1, 1, b"x"));
        ping[6..12].copy_from_slice(&GUEST_MAC);
        un.input(&ping);
        let reply = drain(&mut un).remove(0);
        assert_eq!(reply[..6], GUEST_MAC);
        assert_eq!(reply[6..12], GATEWAY_MAC);
    }

    #[test]
    fn non_ip_non_arp_ethertypes_are_ignored() {
        let mut un = UserNet::new();
        un.input(&ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x86dd, &[0x60; 60]));
        un.input(&ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x8100, &[0; 60]));
        assert!(drain(&mut un).is_empty());
    }

    #[test]
    fn unsupported_ip_protocols_are_ignored() {
        let mut un = UserNet::new();
        un.input(&guest_ip_frame(GATEWAY_IP, IP_PROTO_GRE, &[0; 16]));
        un.input(&guest_ip_frame(GATEWAY_IP, 2, &[0; 8]));
        assert!(drain(&mut un).is_empty());
        assert!(un.udp.is_empty() && un.tcp.is_empty());
    }

    #[test]
    fn ip_fragments_from_the_guest_are_dropped_not_reassembled() {
        let mut frame =
            guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &echo_request_payload(1, 1, b"x"));
        frame[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8]
            .copy_from_slice(&IPV4_FLAG_MORE_FRAGMENTS.to_be_bytes());
        let mut un = UserNet::new();
        un.input(&frame);
        assert!(drain(&mut un).is_empty());
    }

    #[test]
    fn echo_to_any_address_is_answered_locally() {
        // Deliberate: the offline-friendly gateway answers pings for every destination without
        // sending anything to the host network.
        for dst in [
            GATEWAY_IP,
            DNS_IP,
            [8, 8, 8, 8],
            [192, 0, 2, 1],
            [203, 0, 113, 77],
            GUEST_IP,
        ] {
            let mut un = UserNet::new();
            un.input(&guest_ip_frame(
                dst,
                IP_PROTO_ICMP,
                &echo_request_payload(0x1234, 9, b"abcdefgh"),
            ));
            let out = drain(&mut un);
            assert_eq!(out.len(), 1, "{dst:?}");
            let ip = parse_ipv4(&out[0]).unwrap();
            assert_eq!((ip.src, ip.dst, ip.proto), (dst, GUEST_IP, IP_PROTO_ICMP));
            assert_eq!(ip.payload[0], ICMP_ECHO_REPLY);
            assert!(
                un.udp.is_empty() && un.tcp.is_empty(),
                "no host socket was opened"
            );
        }
    }

    #[test]
    fn echo_reply_preserves_identifier_sequence_code_and_payload() {
        let mut un = UserNet::new();
        let mut request = echo_request_payload(0xbeef, 0x0102, b"the quick brown fox");
        request[1] = 5;
        let checksum = {
            request[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
            reference_checksum(&request)
        };
        request[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
        un.input(&guest_ip_frame([8, 8, 8, 8], IP_PROTO_ICMP, &request));
        let out = drain(&mut un);
        let ip = parse_ipv4(&out[0]).unwrap();
        assert_eq!(ip.payload[0], ICMP_ECHO_REPLY);
        assert_eq!(ip.payload[1], 5);
        assert_eq!(ip.payload[4..6], [0xbe, 0xef]);
        assert_eq!(ip.payload[6..8], [0x01, 0x02]);
        assert_eq!(&ip.payload[8..], b"the quick brown fox");
        assert_eq!(reference_checksum(ip.payload), 0);
    }

    #[test]
    fn echo_reply_has_a_valid_ip_header_with_df_and_ttl() {
        let mut un = UserNet::new();
        un.input(&guest_ip_frame(
            [8, 8, 8, 8],
            IP_PROTO_ICMP,
            &echo_request_payload(1, 2, &[0; 56]),
        ));
        let reply = reassemble(&drain(&mut un), GUEST_MAC);
        assert_eq!(reply.fragments, 1);
        assert_eq!(reply.payload.len(), 64);
    }

    #[test]
    fn echo_replies_get_distinct_increasing_ip_ids() {
        let mut un = UserNet::new();
        let mut ids = Vec::new();
        for sequence in 0..4 {
            un.input(&guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, sequence, b"x"),
            ));
            ids.push(reassemble(&drain(&mut un), GUEST_MAC).id);
        }
        assert_eq!(ids, vec![ids[0], ids[0] + 1, ids[0] + 2, ids[0] + 3]);
    }

    #[test]
    fn ip_id_wraps_instead_of_overflowing() {
        let mut un = UserNet::new();
        un.ip_id = u16::MAX;
        un.input(&guest_ip_frame(
            GATEWAY_IP,
            IP_PROTO_ICMP,
            &echo_request_payload(1, 1, b"x"),
        ));
        assert_eq!(reassemble(&drain(&mut un), GUEST_MAC).id, 0);
    }

    #[test]
    fn echo_of_odd_length_data_has_a_valid_checksum() {
        for len in [0usize, 1, 3, 55, 57] {
            let mut un = UserNet::new();
            un.input(&guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, 1, &vec![0xc3; len]),
            ));
            let ip_frames = drain(&mut un);
            let ip = parse_ipv4(&ip_frames[0]).unwrap();
            assert_eq!(reference_checksum(ip.payload), 0, "len {len}");
        }
    }

    #[test]
    fn other_icmp_types_are_ignored() {
        for kind in [ICMP_ECHO_REPLY, ICMP_DEST_UNREACHABLE, 5, 11, 13, 255] {
            let mut payload = echo_request_payload(1, 1, b"x");
            payload[0] = kind;
            let mut un = UserNet::new();
            un.input(&guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload));
            assert!(drain(&mut un).is_empty(), "type {kind}");
        }
    }

    #[test]
    fn icmp_shorter_than_an_echo_header_is_ignored() {
        let payload = echo_request_payload(1, 1, &[]);
        for len in 0..ICMP_ECHO_HEADER_LEN {
            let mut un = UserNet::new();
            un.input(&guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload[..len]));
            assert!(drain(&mut un).is_empty(), "len {len}");
        }
        let mut un = UserNet::new();
        un.input(&guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload));
        assert_eq!(drain(&mut un).len(), 1);
    }

    #[test]
    fn echo_larger_than_the_mtu_is_answered_with_fragments() {
        let data = splitmix_bytes(99, 3000);
        let request = echo_request_payload(7, 8, &data);
        let mut un = UserNet::new();
        un.input(&guest_ip_frame([8, 8, 8, 8], IP_PROTO_ICMP, &request));
        let reply = reassemble(&drain(&mut un), GUEST_MAC);
        assert_eq!(reply.fragments, 3);
        assert_eq!(reply.payload[0], ICMP_ECHO_REPLY);
        assert_eq!(&reply.payload[8..], &data[..]);
        assert_eq!(reference_checksum(&reply.payload), 0);
    }
}

mod udp_nat {
    use super::*;

    fn flow_key(guest_port: u16, port: u16) -> FlowKey {
        FlowKey {
            guest_port,
            remote_ip: LOOPBACK_IP,
            remote_port: port,
        }
    }

    /// Polls with a fixed clock until the stack has produced a frame, then returns all frames.
    fn poll_frames_at(un: &mut UserNet, now: Instant) -> Vec<Vec<u8>> {
        let end = Instant::now() + IO_TIMEOUT;
        while Instant::now() < end {
            un.poll_at(now);
            let frames = drain(un);
            if !frames.is_empty() {
                return frames;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for frames");
    }

    fn udp_view(frame_set: &[Vec<u8>]) -> (Reassembled, u16, u16, Vec<u8>) {
        let whole = reassemble(frame_set, GUEST_MAC);
        assert_eq!(whole.proto, IP_PROTO_UDP);
        assert_eq!(
            transport_checksum(whole.src, whole.dst, IP_PROTO_UDP, &whole.payload),
            0,
            "UDP checksum over the reassembled datagram"
        );
        let sport = be_u16(&whole.payload[UDP_SRC_PORT]);
        let dport = be_u16(&whole.payload[UDP_DST_PORT]);
        assert_eq!(
            be_u16(&whole.payload[UDP_LENGTH]) as usize,
            whole.payload.len()
        );
        let data = whole.payload[UDP_HEADER_LEN..].to_vec();
        (whole, sport, dport, data)
    }

    #[test]
    fn datagram_reaches_the_host_and_the_reply_comes_back_with_the_original_addresses() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"ping"));
        let mut buf = [0u8; 64];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"ping");
        server.send_to(b"pong", from).unwrap();
        let frame = wait_for(&mut un, |f| {
            parse_ipv4(f).is_some_and(|i| i.proto == IP_PROTO_UDP)
        });
        let (whole, sport, dport, data) = udp_view(&[frame]);
        assert_eq!((whole.src, whole.dst), (LOOPBACK_IP, GUEST_IP));
        assert_eq!((sport, dport), (port, 4000));
        assert_eq!(data, b"pong");
    }

    #[test]
    fn datagrams_of_one_flow_share_a_host_socket() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"one"));
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"two"));
        let mut buf = [0u8; 16];
        let (_, first) = server.recv_from(&mut buf).unwrap();
        let (n, second) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"two");
        assert_eq!(first, second);
        assert_eq!(un.udp.len(), 1);
    }

    #[test]
    fn different_guest_ports_get_different_host_sockets() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"a"));
        un.input(&udp_frame(4001, LOOPBACK_IP, port, b"b"));
        let mut buf = [0u8; 16];
        let (_, first) = server.recv_from(&mut buf).unwrap();
        let (_, second) = server.recv_from(&mut buf).unwrap();
        assert_ne!(first, second);
        assert_eq!(un.udp.len(), 2);
    }

    #[test]
    fn different_destination_ports_are_different_flows() {
        let (server_a, port_a) = loopback_udp_server();
        let (server_b, port_b) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port_a, b"a"));
        un.input(&udp_frame(4000, LOOPBACK_IP, port_b, b"b"));
        let mut buf = [0u8; 16];
        assert_eq!(server_a.recv_from(&mut buf).unwrap().0, 1);
        assert_eq!(server_b.recv_from(&mut buf).unwrap().0, 1);
        assert_eq!(un.udp.len(), 2);
    }

    #[test]
    fn replies_are_routed_to_the_right_guest_port() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let mut sources = Vec::new();
        for guest_port in [5000u16, 5001] {
            un.input(&udp_frame(guest_port, LOOPBACK_IP, port, b"hello"));
            let mut buf = [0u8; 16];
            sources.push(server.recv_from(&mut buf).unwrap().1);
        }
        server.send_to(b"for-5001", sources[1]).unwrap();
        server.send_to(b"for-5000", sources[0]).unwrap();
        let mut seen = Vec::new();
        while seen.len() < 2 {
            let frames = poll_frames_at(&mut un, Instant::now());
            for frame in frames {
                let (_, _, dport, data) = udp_view(&[frame]);
                seen.push((dport, data));
            }
        }
        seen.sort();
        assert_eq!(
            seen,
            vec![(5000, b"for-5000".to_vec()), (5001, b"for-5001".to_vec())]
        );
    }

    #[test]
    fn the_gateway_address_reaches_host_loopback_and_replies_come_from_it() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4100, GATEWAY_IP, port, b"to-host"));
        let mut buf = [0u8; 16];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"to-host");
        server.send_to(b"from-host", from).unwrap();
        let frames = poll_frames_at(&mut un, Instant::now());
        let (whole, sport, dport, data) = udp_view(&frames);
        assert_eq!(whole.src, GATEWAY_IP);
        assert_eq!((sport, dport), (port, 4100));
        assert_eq!(data, b"from-host");
    }

    #[test]
    fn dns_queries_are_forwarded_to_the_host_nameserver_and_answered_from_the_dns_address() {
        let (nameserver, nameserver_port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.dns = SocketAddr::from((LOOPBACK_IP, nameserver_port));
        un.input(&udp_frame(33333, DNS_IP, DNS_PORT, b"\x12\x34 query"));
        let mut buf = [0u8; 64];
        let (n, from) = nameserver.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"\x12\x34 query");
        nameserver.send_to(b"\x12\x34 answer", from).unwrap();
        let frames = poll_frames_at(&mut un, Instant::now());
        let (whole, sport, dport, data) = udp_view(&frames);
        assert_eq!(whole.src, DNS_IP);
        assert_eq!((sport, dport), (DNS_PORT, 33333));
        assert_eq!(data, b"\x12\x34 answer");
    }

    #[test]
    fn dns_flows_are_keyed_by_the_dns_address_not_the_host_nameserver() {
        let (_nameserver, nameserver_port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.dns = SocketAddr::from((LOOPBACK_IP, nameserver_port));
        un.input(&udp_frame(33333, DNS_IP, DNS_PORT, b"q"));
        assert!(un.udp.contains_key(&FlowKey {
            guest_port: 33333,
            remote_ip: DNS_IP,
            remote_port: DNS_PORT
        }));
    }

    #[test]
    fn udp_length_field_bounds_the_forwarded_data() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let mut segment = udp_segment(GUEST_IP, LOOPBACK_IP, 4000, port, b"abcdef");
        segment[UDP_LENGTH].copy_from_slice(&((UDP_HEADER_LEN + 2) as u16).to_be_bytes());
        un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_UDP, &segment));
        let mut buf = [0u8; 16];
        let (n, _) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"ab", "bytes beyond the UDP length are ignored");
    }

    #[test]
    fn udp_length_beyond_the_packet_is_clamped_to_what_arrived() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let mut segment = udp_segment(GUEST_IP, LOOPBACK_IP, 4000, port, b"abcdef");
        segment[UDP_LENGTH].copy_from_slice(&u16::MAX.to_be_bytes());
        un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_UDP, &segment));
        let mut buf = [0u8; 16];
        let (n, _) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"abcdef");
    }

    #[test]
    fn udp_length_below_the_header_size_sends_an_empty_datagram() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let mut segment = udp_segment(GUEST_IP, LOOPBACK_IP, 4000, port, b"abcdef");
        segment[UDP_LENGTH].copy_from_slice(&3u16.to_be_bytes());
        un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_UDP, &segment));
        let mut buf = [0u8; 16];
        assert_eq!(server.recv_from(&mut buf).unwrap().0, 0);
    }

    #[test]
    fn empty_datagrams_are_forwarded() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b""));
        let mut buf = [0u8; 16];
        assert_eq!(server.recv_from(&mut buf).unwrap().0, 0);
    }

    #[test]
    fn udp_shorter_than_its_header_is_ignored() {
        let mut un = UserNet::new();
        for len in 0..UDP_HEADER_LEN {
            un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_UDP, &vec![0u8; len]));
        }
        assert!(un.udp.is_empty());
        assert!(drain(&mut un).is_empty());
    }

    #[test]
    fn datagrams_to_the_dhcp_server_port_are_never_nat_ed() {
        let mut un = UserNet::new();
        un.input(&udp_frame(68, LOOPBACK_IP, DHCP_SERVER_PORT, b"junk"));
        un.input(&udp_frame(68, GATEWAY_IP, DHCP_SERVER_PORT, b"junk"));
        assert!(un.udp.is_empty());
    }

    #[test]
    fn idle_flows_expire_exactly_at_the_timeout() {
        let (_server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let t0 = Instant::now();
        un.input_at(&udp_frame(4000, LOOPBACK_IP, port, b"x"), t0);
        un.poll_at(t0);
        assert_eq!(un.udp.len(), 1);
        un.poll_at(t0 + UDP_IDLE_TIMEOUT - Duration::from_millis(1));
        assert_eq!(un.udp.len(), 1, "one millisecond early");
        un.poll_at(t0 + UDP_IDLE_TIMEOUT);
        assert!(un.udp.is_empty(), "at the timeout");
    }

    #[test]
    fn sending_refreshes_the_idle_timer() {
        let (_server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let t0 = Instant::now();
        let frame = udp_frame(4000, LOOPBACK_IP, port, b"x");
        un.input_at(&frame, t0);
        un.input_at(&frame, t0 + Duration::from_secs(40));
        un.poll_at(t0 + Duration::from_secs(70));
        assert_eq!(un.udp.len(), 1, "30 s since the last datagram");
        un.poll_at(t0 + Duration::from_secs(101));
        assert!(un.udp.is_empty());
    }

    #[test]
    fn receiving_from_the_host_refreshes_the_idle_timer() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let t0 = Instant::now();
        un.input_at(&udp_frame(4000, LOOPBACK_IP, port, b"x"), t0);
        let mut buf = [0u8; 16];
        let (_, from) = server.recv_from(&mut buf).unwrap();
        server.send_to(b"y", from).unwrap();
        poll_frames_at(&mut un, t0 + Duration::from_secs(50));
        un.poll_at(t0 + Duration::from_secs(100));
        assert_eq!(un.udp.len(), 1, "50 s after the reply");
        un.poll_at(t0 + Duration::from_secs(111));
        assert!(un.udp.is_empty());
    }

    #[test]
    fn expired_flows_are_recreated_by_the_next_datagram() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        let t0 = Instant::now();
        un.input_at(&udp_frame(4000, LOOPBACK_IP, port, b"1"), t0);
        let mut buf = [0u8; 16];
        let (_, first) = server.recv_from(&mut buf).unwrap();
        un.poll_at(t0 + UDP_IDLE_TIMEOUT);
        un.input_at(
            &udp_frame(4000, LOOPBACK_IP, port, b"2"),
            t0 + UDP_IDLE_TIMEOUT,
        );
        let (_, second) = server.recv_from(&mut buf).unwrap();
        assert_ne!(first, second, "a fresh host socket");
    }

    #[test]
    fn the_flow_table_never_exceeds_its_limit() {
        let (_server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.udp_flow_limit = 4;
        let t0 = Instant::now();
        for (i, guest_port) in (6000u16..6020).enumerate() {
            un.input_at(
                &udp_frame(guest_port, LOOPBACK_IP, port, b"x"),
                t0 + Duration::from_millis(i as u64),
            );
            assert!(un.udp.len() <= 4, "after {} flows", i + 1);
        }
        assert_eq!(un.udp.len(), 4);
    }

    #[test]
    fn the_least_recently_used_flow_is_evicted_first() {
        let (_server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.udp_flow_limit = 3;
        let t0 = Instant::now();
        let at = |secs: u64| t0 + Duration::from_secs(secs);
        for (i, guest_port) in [1u16, 2, 3].into_iter().enumerate() {
            un.input_at(
                &udp_frame(guest_port, LOOPBACK_IP, port, b"x"),
                at(i as u64),
            );
        }
        // Flow 1 was the oldest, but sending on it makes flow 2 the least recently used.
        un.input_at(&udp_frame(1, LOOPBACK_IP, port, b"x"), at(10));
        un.input_at(&udp_frame(4, LOOPBACK_IP, port, b"x"), at(11));
        let mut remaining: Vec<u16> = un.udp.keys().map(|k| k.guest_port).collect();
        remaining.sort();
        assert_eq!(remaining, vec![1, 3, 4]);
        un.input_at(&udp_frame(5, LOOPBACK_IP, port, b"x"), at(12));
        let mut remaining: Vec<u16> = un.udp.keys().map(|k| k.guest_port).collect();
        remaining.sort();
        assert_eq!(remaining, vec![1, 4, 5], "flow 3 was idle longest");
    }

    #[test]
    fn traffic_on_an_existing_flow_never_evicts_anything() {
        let (_server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.udp_flow_limit = 2;
        let t0 = Instant::now();
        un.input_at(&udp_frame(1, LOOPBACK_IP, port, b"x"), t0);
        un.input_at(&udp_frame(2, LOOPBACK_IP, port, b"x"), t0);
        for step in 1..10 {
            un.input_at(
                &udp_frame(1, LOOPBACK_IP, port, b"x"),
                t0 + Duration::from_secs(step),
            );
        }
        assert!(un.udp.contains_key(&flow_key(1, port)));
        assert!(un.udp.contains_key(&flow_key(2, port)));
    }

    #[test]
    fn an_evicted_flow_can_start_again() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.udp_flow_limit = 1;
        let t0 = Instant::now();
        un.input_at(&udp_frame(1, LOOPBACK_IP, port, b"a"), t0);
        un.input_at(
            &udp_frame(2, LOOPBACK_IP, port, b"b"),
            t0 + Duration::from_secs(1),
        );
        un.input_at(
            &udp_frame(1, LOOPBACK_IP, port, b"c"),
            t0 + Duration::from_secs(2),
        );
        let mut buf = [0u8; 16];
        for expected in [b"a", b"b", b"c"] {
            let (n, _) = server.recv_from(&mut buf).unwrap();
            assert_eq!(&buf[..n], expected);
        }
        assert_eq!(un.udp.len(), 1);
        assert!(un.udp.contains_key(&flow_key(1, port)));
    }

    #[test]
    fn a_limit_of_zero_still_lets_one_flow_work() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.udp_flow_limit = 0;
        un.input(&udp_frame(1, LOOPBACK_IP, port, b"a"));
        let mut buf = [0u8; 16];
        assert_eq!(server.recv_from(&mut buf).unwrap().0, 1);
    }

    #[test]
    fn host_datagram_that_fits_the_mtu_is_one_packet_with_df() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"hi"));
        let mut buf = [0u8; 16];
        let (_, from) = server.recv_from(&mut buf).unwrap();
        let largest = vec![0x42u8; ETHERNET_MTU - IPV4_HEADER_LEN - UDP_HEADER_LEN];
        server.send_to(&largest, from).unwrap();
        let frames = poll_frames_at(&mut un, Instant::now());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), ETH_HEADER_LEN + ETHERNET_MTU);
        let (whole, _, _, data) = udp_view(&frames);
        assert_eq!(whole.fragments, 1);
        assert_eq!(data, largest);
    }

    #[test]
    fn host_datagram_one_byte_over_the_mtu_is_fragmented() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"hi"));
        let mut buf = [0u8; 16];
        let (_, from) = server.recv_from(&mut buf).unwrap();
        let data: Vec<u8> = (0..ETHERNET_MTU - IPV4_HEADER_LEN - UDP_HEADER_LEN + 1)
            .map(|i| i as u8)
            .collect();
        server.send_to(&data, from).unwrap();
        let frames = poll_frames_at(&mut un, Instant::now());
        assert_eq!(frames.len(), 2);
        let (whole, sport, dport, received) = udp_view(&frames);
        assert_eq!((sport, dport), (port, 4000));
        assert_eq!(received, data);
        assert_eq!(whole.fragments, 2);
    }

    #[test]
    fn large_host_datagram_is_delivered_as_correctly_numbered_fragments() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"hi"));
        let mut buf = [0u8; 16];
        let (_, from) = server.recv_from(&mut buf).unwrap();
        let data = splitmix_bytes(5, 4000);
        server.send_to(&data, from).unwrap();
        let frames = poll_frames_at(&mut un, Instant::now());
        // 4008 UDP bytes: 1480 + 1480 + 1048.
        assert_eq!(frames.len(), 3);
        let lengths: Vec<usize> = frames
            .iter()
            .map(|f| be_u16(&f[ETH_HEADER_LEN + 2..ETH_HEADER_LEN + 4]) as usize)
            .collect();
        assert_eq!(lengths, vec![1500, 1500, 1068]);
        let words: Vec<u16> = frames
            .iter()
            .map(|f| be_u16(&f[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8]))
            .collect();
        assert_eq!(words, vec![0x2000, 0x2000 | 185, 370]);
        let (_, _, _, received) = udp_view(&frames);
        assert_eq!(received, data);
    }

    #[test]
    fn fragments_of_different_datagrams_get_different_ids() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4000, LOOPBACK_IP, port, b"hi"));
        let mut buf = [0u8; 16];
        let (_, from) = server.recv_from(&mut buf).unwrap();
        let mut ids = Vec::new();
        for round in 0..2u8 {
            server.send_to(&vec![round; 3000], from).unwrap();
            let frames = poll_frames_at(&mut un, Instant::now());
            ids.push(reassemble(&frames, GUEST_MAC).id);
        }
        assert_ne!(ids[0], ids[1]);
    }
}

mod dhcp {
    use super::*;

    const TRANSACTION_ID: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];
    const BROADCAST_FLAG: [u8; 2] = [0x80, 0x00];
    const ZERO_IP: [u8; 4] = [0; 4];

    fn message_type_option(kind: u8) -> Vec<u8> {
        vec![DHCP_OPT_MESSAGE_TYPE, 1, kind]
    }

    fn address_option(code: u8, ip: [u8; 4]) -> Vec<u8> {
        let mut option = vec![code, 4];
        option.extend_from_slice(&ip);
        option
    }

    fn client_message(options: &[u8], client_ip: [u8; 4]) -> Vec<u8> {
        let mut message = vec![0u8; DHCP_OPTIONS_OFFSET];
        message[BOOTP_OP] = BOOTP_REQUEST;
        message[BOOTP_HTYPE] = HTYPE_ETHERNET;
        message[BOOTP_HLEN] = HLEN_ETHERNET;
        message[BOOTP_XID].copy_from_slice(&TRANSACTION_ID);
        message[BOOTP_FLAGS].copy_from_slice(&BROADCAST_FLAG);
        message[BOOTP_CIADDR].copy_from_slice(&client_ip);
        message[BOOTP_CHADDR][..6].copy_from_slice(&GUEST_MAC);
        message[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
        message.extend_from_slice(options);
        message
    }

    fn options_of(kind: u8, extra: &[Vec<u8>]) -> Vec<u8> {
        let mut options = message_type_option(kind);
        for option in extra {
            options.extend_from_slice(option);
        }
        options.push(DHCP_OPT_END);
        options
    }

    fn selecting_request(requested: [u8; 4], server: [u8; 4]) -> Vec<u8> {
        client_message(
            &options_of(
                DHCP_REQUEST,
                &[
                    address_option(DHCP_OPT_REQUESTED_IP, requested),
                    address_option(DHCP_OPT_SERVER_ID, server),
                ],
            ),
            ZERO_IP,
        )
    }

    fn send(un: &mut UserNet, message: &[u8]) {
        let datagram = udp_segment(
            ZERO_IP,
            LIMITED_BROADCAST_IP,
            DHCP_CLIENT_PORT,
            DHCP_SERVER_PORT,
            message,
        );
        let packet = ipv4_packet(1, ZERO_IP, LIMITED_BROADCAST_IP, IP_PROTO_UDP, &datagram);
        un.input(&ethernet_frame(
            BROADCAST_MAC,
            GUEST_MAC,
            ETHERTYPE_IPV4,
            &packet,
        ));
    }

    /// The BOOTP payload of the one reply the stack queued, after checking every layer around it.
    fn reply_to(un: &mut UserNet, message: &[u8]) -> Option<Vec<u8>> {
        send(un, message);
        let mut frames = drain(un);
        if frames.is_empty() {
            return None;
        }
        assert_eq!(frames.len(), 1);
        let frame = frames.remove(0);
        assert_eq!(frame[..6], BROADCAST_MAC);
        assert_eq!(frame[6..12], GATEWAY_MAC);
        let ip = parse_ipv4(&frame).unwrap();
        assert_eq!(
            (ip.src, ip.dst, ip.proto),
            (GATEWAY_IP, LIMITED_BROADCAST_IP, IP_PROTO_UDP)
        );
        assert_eq!(
            reference_checksum(&frame[ETH_HEADER_LEN..ETH_HEADER_LEN + IPV4_HEADER_LEN]),
            0
        );
        assert_eq!(
            transport_checksum(ip.src, ip.dst, IP_PROTO_UDP, ip.payload),
            0
        );
        assert_eq!(be_u16(&ip.payload[UDP_SRC_PORT]), DHCP_SERVER_PORT);
        assert_eq!(be_u16(&ip.payload[UDP_DST_PORT]), DHCP_CLIENT_PORT);
        Some(ip.payload[UDP_HEADER_LEN..].to_vec())
    }

    fn options_in(reply: &[u8]) -> Vec<(u8, Vec<u8>)> {
        assert_eq!(reply[BOOTP_MAGIC_COOKIE], DHCP_MAGIC_COOKIE);
        let mut options = Vec::new();
        let mut pos = DHCP_OPTIONS_OFFSET;
        while reply[pos] != DHCP_OPT_END {
            let len = reply[pos + 1] as usize;
            options.push((reply[pos], reply[pos + 2..pos + 2 + len].to_vec()));
            pos += 2 + len;
        }
        assert!(
            reply[pos + 1..].iter().all(|b| *b == 0),
            "padding after END is zero"
        );
        options
    }

    fn message_type_of(reply: &[u8]) -> u8 {
        options_in(reply)[0].1[0]
    }

    fn full_option_set(kind: u8) -> Vec<(u8, Vec<u8>)> {
        vec![
            (DHCP_OPT_MESSAGE_TYPE, vec![kind]),
            (DHCP_OPT_SERVER_ID, GATEWAY_IP.to_vec()),
            (DHCP_OPT_LEASE_TIME, LEASE_SECONDS_BIG_ENDIAN.to_vec()),
            (DHCP_OPT_SUBNET_MASK, vec![255, 255, 255, 0]),
            (DHCP_OPT_ROUTER, GATEWAY_IP.to_vec()),
            (DHCP_OPT_DNS_SERVER, DNS_IP.to_vec()),
            (DHCP_OPT_BROADCAST_ADDRESS, vec![10, 0, 2, 255]),
        ]
    }

    fn assert_reply_header(reply: &[u8]) {
        assert_eq!(reply.len(), BOOTP_MIN_LEN);
        assert_eq!(reply[BOOTP_OP], BOOTP_REPLY);
        assert_eq!(reply[BOOTP_HTYPE], HTYPE_ETHERNET);
        assert_eq!(reply[BOOTP_HLEN], HLEN_ETHERNET);
        assert_eq!(reply[3], 0, "hops");
        assert_eq!(reply[BOOTP_XID], TRANSACTION_ID);
        assert_eq!(reply[8..10], [0, 0], "secs");
        assert_eq!(reply[BOOTP_FLAGS], BROADCAST_FLAG);
        assert_eq!(reply[BOOTP_CIADDR], ZERO_IP);
        assert_eq!(reply[24..28], ZERO_IP, "giaddr");
        assert_eq!(reply[BOOTP_CHADDR][..6], GUEST_MAC);
        assert!(
            reply[34..DHCP_OPTIONS_OFFSET - 4].iter().all(|b| *b == 0),
            "chaddr padding, sname, file"
        );
    }

    #[test]
    fn discover_gets_an_offer_with_exactly_the_documented_options() {
        let mut un = UserNet::new();
        let reply = reply_to(
            &mut un,
            &client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP),
        )
        .unwrap();
        assert_reply_header(&reply);
        assert_eq!(reply[BOOTP_YIADDR], GUEST_IP);
        assert_eq!(reply[BOOTP_SIADDR], GATEWAY_IP);
        assert_eq!(options_in(&reply), full_option_set(DHCP_OFFER));
    }

    #[test]
    fn offer_option_bytes_are_exact() {
        let mut un = UserNet::new();
        let reply = reply_to(
            &mut un,
            &client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP),
        )
        .unwrap();
        let expected: &[u8] = &[
            53, 1, 2, // message type OFFER
            54, 4, 10, 0, 2, 2, // server identifier
            51, 4, 0x00, 0x01, 0x51, 0x80, // lease time 86400 s
            1, 4, 255, 255, 255, 0, // subnet mask
            3, 4, 10, 0, 2, 2, // router
            6, 4, 10, 0, 2, 3, // DNS server
            28, 4, 10, 0, 2, 255, // broadcast address
            255, // end
        ];
        assert_eq!(
            &reply[DHCP_OPTIONS_OFFSET..DHCP_OPTIONS_OFFSET + expected.len()],
            expected
        );
    }

    #[test]
    fn selecting_request_for_the_offered_address_gets_an_ack() {
        let mut un = UserNet::new();
        let reply = reply_to(&mut un, &selecting_request(GUEST_IP, GATEWAY_IP)).unwrap();
        assert_reply_header(&reply);
        assert_eq!(reply[BOOTP_YIADDR], GUEST_IP);
        assert_eq!(reply[BOOTP_SIADDR], GATEWAY_IP);
        assert_eq!(options_in(&reply), full_option_set(DHCP_ACK));
    }

    #[test]
    fn selecting_request_for_a_different_address_gets_a_nak() {
        let mut un = UserNet::new();
        let reply = reply_to(&mut un, &selecting_request([10, 0, 2, 16], GATEWAY_IP)).unwrap();
        assert_reply_header(&reply);
        assert_eq!(reply[BOOTP_YIADDR], ZERO_IP, "a NAK offers no address");
        assert_eq!(reply[BOOTP_SIADDR], ZERO_IP);
        assert_eq!(
            options_in(&reply),
            vec![
                (DHCP_OPT_MESSAGE_TYPE, vec![DHCP_NAK]),
                (DHCP_OPT_SERVER_ID, GATEWAY_IP.to_vec()),
            ]
        );
    }

    #[test]
    fn nak_is_sent_for_addresses_outside_the_subnet_too() {
        for wanted in [
            [192, 168, 1, 50],
            [0, 0, 0, 1],
            [255; 4],
            [10, 0, 3, 15],
            [10, 0, 2, 14],
        ] {
            let mut un = UserNet::new();
            let reply = reply_to(&mut un, &selecting_request(wanted, GATEWAY_IP)).unwrap();
            assert_eq!(message_type_of(&reply), DHCP_NAK, "{wanted:?}");
        }
    }

    #[test]
    fn request_naming_another_server_is_not_answered() {
        // RFC 2131 section 3.1 step 4: the client picked a different server's offer.
        let mut un = UserNet::new();
        assert!(reply_to(&mut un, &selecting_request(GUEST_IP, UNICAST_OFFER_SERVER)).is_none());
        assert!(
            reply_to(
                &mut un,
                &selecting_request([10, 0, 2, 16], UNICAST_OFFER_SERVER)
            )
            .is_none()
        );
        assert!(reply_to(&mut un, &selecting_request(GUEST_IP, ZERO_IP)).is_none());
    }

    #[test]
    fn init_reboot_request_without_server_id_is_checked_against_the_requested_address() {
        let mut un = UserNet::new();
        let ours = client_message(
            &options_of(
                DHCP_REQUEST,
                &[address_option(DHCP_OPT_REQUESTED_IP, GUEST_IP)],
            ),
            ZERO_IP,
        );
        assert_eq!(
            message_type_of(&reply_to(&mut un, &ours).unwrap()),
            DHCP_ACK
        );
        let stale = client_message(
            &options_of(
                DHCP_REQUEST,
                &[address_option(DHCP_OPT_REQUESTED_IP, [10, 0, 2, 40])],
            ),
            ZERO_IP,
        );
        assert_eq!(
            message_type_of(&reply_to(&mut un, &stale).unwrap()),
            DHCP_NAK
        );
    }

    #[test]
    fn renewing_client_is_checked_against_its_current_address() {
        let mut un = UserNet::new();
        let renewing = client_message(&options_of(DHCP_REQUEST, &[]), GUEST_IP);
        assert_eq!(
            message_type_of(&reply_to(&mut un, &renewing).unwrap()),
            DHCP_ACK
        );
        let foreign = client_message(&options_of(DHCP_REQUEST, &[]), [172, 16, 0, 9]);
        assert_eq!(
            message_type_of(&reply_to(&mut un, &foreign).unwrap()),
            DHCP_NAK
        );
    }

    #[test]
    fn requested_address_option_takes_precedence_over_ciaddr() {
        let mut un = UserNet::new();
        let message = client_message(
            &options_of(
                DHCP_REQUEST,
                &[address_option(DHCP_OPT_REQUESTED_IP, GUEST_IP)],
            ),
            [172, 16, 0, 9],
        );
        assert_eq!(
            message_type_of(&reply_to(&mut un, &message).unwrap()),
            DHCP_ACK
        );
    }

    #[test]
    fn request_naming_no_address_at_all_is_acked() {
        let mut un = UserNet::new();
        let message = client_message(&options_of(DHCP_REQUEST, &[]), ZERO_IP);
        assert_eq!(
            message_type_of(&reply_to(&mut un, &message).unwrap()),
            DHCP_ACK
        );
    }

    #[test]
    fn server_id_is_checked_before_the_address() {
        let mut un = UserNet::new();
        let message = selecting_request([10, 0, 2, 16], [1, 2, 3, 4]);
        assert!(reply_to(&mut un, &message).is_none());
    }

    #[test]
    fn release_decline_and_inform_are_ignored() {
        for kind in [DHCP_RELEASE, DHCP_DECLINE, DHCP_INFORM] {
            let mut un = UserNet::new();
            let message = client_message(&options_of(kind, &[]), GUEST_IP);
            assert!(reply_to(&mut un, &message).is_none(), "type {kind}");
        }
    }

    #[test]
    fn server_message_types_sent_by_a_client_are_ignored() {
        for kind in [DHCP_OFFER, DHCP_ACK, DHCP_NAK, 0, 9, 99, 255] {
            let mut un = UserNet::new();
            assert!(
                reply_to(&mut un, &client_message(&options_of(kind, &[]), ZERO_IP)).is_none(),
                "type {kind}"
            );
        }
    }

    #[test]
    fn message_without_a_type_option_is_ignored() {
        let mut un = UserNet::new();
        let message = client_message(&[DHCP_OPT_END], ZERO_IP);
        assert!(reply_to(&mut un, &message).is_none());
    }

    #[test]
    fn bootp_replies_and_bad_cookies_are_ignored() {
        let mut un = UserNet::new();
        let mut reply_op = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
        reply_op[BOOTP_OP] = BOOTP_REPLY;
        assert!(reply_to(&mut un, &reply_op).is_none());
        let mut bad_cookie = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
        bad_cookie[BOOTP_MAGIC_COOKIE].copy_from_slice(&[0x63, 0x82, 0x53, 0x64]);
        assert!(reply_to(&mut un, &bad_cookie).is_none());
        let mut no_cookie = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
        no_cookie[BOOTP_MAGIC_COOKIE].copy_from_slice(&[0; 4]);
        assert!(reply_to(&mut un, &no_cookie).is_none());
    }

    #[test]
    fn transaction_id_flags_and_client_hardware_address_are_echoed() {
        let mut un = UserNet::new();
        let mut message = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
        message[BOOTP_XID].copy_from_slice(&[1, 2, 3, 4]);
        message[BOOTP_FLAGS].copy_from_slice(&[0x00, 0x00]);
        message[BOOTP_CHADDR].copy_from_slice(&[9; 16]);
        let reply = reply_to(&mut un, &message).unwrap();
        assert_eq!(reply[BOOTP_XID], [1, 2, 3, 4]);
        assert_eq!(reply[BOOTP_FLAGS], [0, 0]);
        assert_eq!(reply[BOOTP_CHADDR], [9; 16]);
    }

    #[test]
    fn replies_are_always_broadcast_even_to_a_client_with_an_address() {
        let mut un = UserNet::new();
        let renewing = client_message(&options_of(DHCP_REQUEST, &[]), GUEST_IP);
        send(&mut un, &renewing);
        let frame = drain(&mut un).remove(0);
        assert_eq!(frame[..6], BROADCAST_MAC);
        assert_eq!(
            frame[ETH_HEADER_LEN + 16..ETH_HEADER_LEN + 20],
            LIMITED_BROADCAST_IP
        );
    }

    #[test]
    fn reply_frame_is_the_expected_size() {
        let mut un = UserNet::new();
        send(
            &mut un,
            &client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP),
        );
        let frame = drain(&mut un).remove(0);
        assert_eq!(
            frame.len(),
            ETH_HEADER_LEN + IPV4_HEADER_LEN + UDP_HEADER_LEN + BOOTP_MIN_LEN
        );
    }

    #[test]
    fn leading_pad_options_and_unknown_options_are_skipped() {
        let mut un = UserNet::new();
        let mut options = vec![
            DHCP_OPT_PAD,
            DHCP_OPT_PAD,
            61,
            3,
            1,
            2,
            3,
            12,
            4,
            b'h',
            b'o',
            b's',
            b't',
        ];
        options.extend_from_slice(&message_type_option(DHCP_DISCOVER));
        options.push(DHCP_OPT_END);
        let reply = reply_to(&mut un, &client_message(&options, ZERO_IP)).unwrap();
        assert_eq!(message_type_of(&reply), DHCP_OFFER);
    }

    #[test]
    fn options_after_the_end_marker_are_not_read() {
        let mut un = UserNet::new();
        let mut options = vec![DHCP_OPT_END];
        options.extend_from_slice(&message_type_option(DHCP_DISCOVER));
        assert!(reply_to(&mut un, &client_message(&options, ZERO_IP)).is_none());
    }

    #[test]
    fn a_message_without_the_end_marker_is_still_read() {
        let mut un = UserNet::new();
        let options = message_type_option(DHCP_DISCOVER);
        assert!(reply_to(&mut un, &client_message(&options, ZERO_IP)).is_some());
    }

    #[test]
    fn every_truncation_of_a_discover_is_handled() {
        let full = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
        let first_complete_type_option = DHCP_OPTIONS_OFFSET + 3;
        for len in 0..=full.len() {
            let mut un = UserNet::new();
            let answered = reply_to(&mut un, &full[..len]).is_some();
            assert_eq!(answered, len >= first_complete_type_option, "len {len}");
        }
    }

    #[test]
    fn truncated_address_options_are_ignored_not_misread() {
        let mut options = message_type_option(DHCP_REQUEST);
        options.extend_from_slice(&[DHCP_OPT_REQUESTED_IP, 4, 10, 0]);
        let parsed = parse_dhcp_request(&client_message(&options, ZERO_IP)).unwrap();
        assert_eq!(parsed.message_type, DHCP_REQUEST);
        assert_eq!(parsed.requested_ip, None);
    }

    #[test]
    fn address_options_of_the_wrong_length_are_ignored() {
        let mut options = message_type_option(DHCP_REQUEST);
        options.extend_from_slice(&[DHCP_OPT_REQUESTED_IP, 3, 1, 2, 3]);
        options.extend_from_slice(&[DHCP_OPT_SERVER_ID, 5, 1, 2, 3, 4, 5]);
        options.push(DHCP_OPT_END);
        let parsed = parse_dhcp_request(&client_message(&options, ZERO_IP)).unwrap();
        assert_eq!((parsed.requested_ip, parsed.server_id), (None, None));
    }

    #[test]
    fn parsing_reads_every_field_the_decision_needs() {
        let options = options_of(
            DHCP_REQUEST,
            &[
                address_option(DHCP_OPT_REQUESTED_IP, [10, 0, 2, 15]),
                address_option(DHCP_OPT_SERVER_ID, [10, 0, 2, 2]),
            ],
        );
        let parsed = parse_dhcp_request(&client_message(&options, [1, 2, 3, 4])).unwrap();
        assert_eq!(parsed.message_type, DHCP_REQUEST);
        assert_eq!(parsed.requested_ip, Some([10, 0, 2, 15]));
        assert_eq!(parsed.server_id, Some([10, 0, 2, 2]));
        assert_eq!(parsed.client_ip, [1, 2, 3, 4]);
    }

    #[test]
    fn parsing_rejects_messages_without_any_option_bytes() {
        let header_only = client_message(&[], ZERO_IP);
        assert_eq!(header_only.len(), DHCP_OPTIONS_OFFSET);
        assert!(parse_dhcp_request(&header_only).is_none());
        assert!(parse_dhcp_request(&[]).is_none());
    }

    #[test]
    fn message_type_option_with_extra_bytes_uses_the_first() {
        let options = [DHCP_OPT_MESSAGE_TYPE, 2, DHCP_DISCOVER, 9, DHCP_OPT_END];
        let parsed = parse_dhcp_request(&client_message(&options, ZERO_IP)).unwrap();
        assert_eq!(parsed.message_type, DHCP_DISCOVER);
    }

    #[test]
    fn zero_length_message_type_option_leaves_the_type_unset() {
        let options = [DHCP_OPT_MESSAGE_TYPE, 0, DHCP_OPT_END];
        let parsed = parse_dhcp_request(&client_message(&options, ZERO_IP)).unwrap();
        assert_eq!(parsed.message_type, 0);
    }

    fn decide(
        kind: u8,
        requested: Option<[u8; 4]>,
        server: Option<[u8; 4]>,
        client: [u8; 4],
    ) -> Option<u8> {
        dhcp_reply_type(&DhcpRequest {
            message_type: kind,
            requested_ip: requested,
            server_id: server,
            client_ip: client,
        })
    }

    #[test]
    fn reply_type_decision_table() {
        let ours = Some(GATEWAY_IP);
        let theirs = Some([10, 0, 2, 99]);
        let wanted = Some(GUEST_IP);
        let other = Some([10, 0, 2, 50]);
        assert_eq!(decide(DHCP_DISCOVER, None, None, ZERO_IP), Some(DHCP_OFFER));
        assert_eq!(
            decide(DHCP_DISCOVER, other, theirs, [9; 4]),
            Some(DHCP_OFFER),
            "discover ignores the rest"
        );
        assert_eq!(decide(DHCP_REQUEST, wanted, ours, ZERO_IP), Some(DHCP_ACK));
        assert_eq!(decide(DHCP_REQUEST, wanted, None, ZERO_IP), Some(DHCP_ACK));
        assert_eq!(decide(DHCP_REQUEST, None, None, GUEST_IP), Some(DHCP_ACK));
        assert_eq!(decide(DHCP_REQUEST, None, None, ZERO_IP), Some(DHCP_ACK));
        assert_eq!(decide(DHCP_REQUEST, None, ours, ZERO_IP), Some(DHCP_ACK));
        assert_eq!(decide(DHCP_REQUEST, other, ours, ZERO_IP), Some(DHCP_NAK));
        assert_eq!(decide(DHCP_REQUEST, other, None, ZERO_IP), Some(DHCP_NAK));
        assert_eq!(decide(DHCP_REQUEST, None, None, [9; 4]), Some(DHCP_NAK));
        assert_eq!(decide(DHCP_REQUEST, wanted, theirs, ZERO_IP), None);
        assert_eq!(decide(DHCP_REQUEST, other, theirs, ZERO_IP), None);
        for kind in [
            0,
            DHCP_OFFER,
            DHCP_DECLINE,
            DHCP_ACK,
            DHCP_NAK,
            DHCP_RELEASE,
            DHCP_INFORM,
            200,
        ] {
            assert_eq!(decide(kind, wanted, ours, ZERO_IP), None, "type {kind}");
        }
    }

    #[test]
    fn lease_time_is_one_day() {
        assert_eq!(DHCP_LEASE_SECS.to_be_bytes(), LEASE_SECONDS_BIG_ENDIAN);
    }

    #[test]
    fn dhcp_replies_get_consecutive_ip_ids() {
        let mut un = UserNet::new();
        let mut ids = Vec::new();
        for _ in 0..3 {
            send(
                &mut un,
                &client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP),
            );
            let frame = drain(&mut un).remove(0);
            ids.push(be_u16(&frame[ETH_HEADER_LEN + 4..ETH_HEADER_LEN + 6]));
        }
        assert_eq!(ids[1], ids[0] + 1);
        assert_eq!(ids[2], ids[0] + 2);
    }

    #[test]
    fn random_dhcp_payloads_never_panic() {
        let mut rng = XorShift::new(FUZZ_SEED + 3);
        for _ in 0..4000 {
            let len = rng.below(400);
            let mut message = rng.bytes(len);
            if len > DHCP_OPTIONS_OFFSET && rng.below(4) != 0 {
                message[BOOTP_OP] = BOOTP_REQUEST;
                message[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
            }
            let mut un = UserNet::new();
            send(&mut un, &message);
            for frame in drain(&mut un) {
                assert_eq!(
                    frame.len(),
                    ETH_HEADER_LEN + IPV4_HEADER_LEN + UDP_HEADER_LEN + BOOTP_MIN_LEN
                );
            }
        }
    }
}

struct Guest {
    sport: u16,
    dport: u16,
    dst: [u8; 4],
    seq: u32,
    ack: u32,
    window: u16,
}

impl Guest {
    fn new(sport: u16, dport: u16, seq: u32) -> Self {
        Guest {
            sport,
            dport,
            dst: LOOPBACK_IP,
            seq,
            ack: 0,
            window: 60000,
        }
    }

    fn segment(&self, flags: u8, data: &[u8]) -> Vec<u8> {
        let header = TcpHeader {
            src: GUEST_IP,
            dst: self.dst,
            sport: self.sport,
            dport: self.dport,
            seq: self.seq,
            ack: self.ack,
            flags,
            window: self.window,
            advertise_mss: false,
        };
        tcp_segment(&header, data)
    }

    fn send(&mut self, un: &mut UserNet, flags: u8, data: &[u8]) {
        let segment = self.segment(flags, data);
        un.input(&guest_ip_frame(self.dst, IP_PROTO_TCP, &segment));
        self.seq = self
            .seq
            .wrapping_add(data.len() as u32 + (flags & (SYN | FIN) != 0) as u32);
    }

    fn key(&self) -> FlowKey {
        FlowKey {
            guest_port: self.sport,
            remote_ip: self.dst,
            remote_port: self.dport,
        }
    }
}

struct TcpView {
    src: [u8; 4],
    dst: [u8; 4],
    sport: u16,
    dport: u16,
    flags: u8,
    seq: u32,
    ack: u32,
    window: u16,
    options: Vec<u8>,
    data: Vec<u8>,
}

/// Decodes a TCP-over-IPv4 frame, asserting that both checksums are valid; `None` for other
/// traffic.
fn tcp_view(frame: &[u8]) -> Option<TcpView> {
    let ip = parse_ipv4(frame)?;
    if ip.proto != IP_PROTO_TCP {
        return None;
    }
    assert_eq!(
        reference_checksum(&frame[ETH_HEADER_LEN..ETH_HEADER_LEN + IPV4_HEADER_LEN]),
        0
    );
    assert_eq!(
        transport_checksum(ip.src, ip.dst, IP_PROTO_TCP, ip.payload),
        0,
        "bad tcp checksum"
    );
    let segment = TcpSegment::parse(ip.payload)?;
    Some(TcpView {
        src: ip.src,
        dst: ip.dst,
        sport: segment.sport,
        dport: segment.dport,
        flags: segment.flags,
        seq: segment.seq,
        ack: segment.ack,
        window: segment.window as u16,
        options: segment.options.to_vec(),
        data: segment.data.to_vec(),
    })
}

fn tcp_of(f: &[u8]) -> Option<(u8, u32, u32, Vec<u8>)> {
    let view = tcp_view(f)?;
    Some((view.flags, view.seq, view.ack, view.data))
}

fn wait_for_tcp(un: &mut UserNet, mut pred: impl FnMut(&TcpView) -> bool) -> TcpView {
    let frame = wait_for(un, |f| tcp_view(f).is_some_and(|v| pred(&v)));
    tcp_view(&frame).unwrap()
}

/// Runs the three-way handshake against `listener` and returns the accepted host side plus our
/// initial sequence number.
fn handshake(un: &mut UserNet, guest: &mut Guest, listener: &TcpListener) -> (TcpStream, u32) {
    guest.send(un, SYN, &[]);
    let synack = wait_for_tcp(un, |v| v.flags == SYN | ACK);
    assert_eq!(synack.ack, guest.seq);
    guest.ack = synack.seq.wrapping_add(1);
    guest.send(un, ACK, &[]);
    let (server, _) = listener.accept().unwrap();
    server.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    (server, synack.seq)
}

fn listener() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

/// Collects `want` bytes of data segments, acknowledging each, and checks they arrive in order
/// and within `mss`.
fn receive_data(un: &mut UserNet, guest: &mut Guest, want: usize, mss: usize) -> Vec<u8> {
    let mut received = Vec::new();
    while received.len() < want {
        let segment = wait_for_tcp(un, |v| !v.data.is_empty());
        assert_eq!(segment.seq, guest.ack, "in-order delivery");
        assert!(
            segment.data.len() <= mss,
            "segment of {} exceeds mss {mss}",
            segment.data.len()
        );
        guest.ack = guest.ack.wrapping_add(segment.data.len() as u32);
        received.extend_from_slice(&segment.data);
        guest.send(un, ACK, &[]);
    }
    received
}

fn read_exactly(stream: &mut TcpStream, len: usize) -> Vec<u8> {
    let mut data = vec![0u8; len];
    stream.read_exact(&mut data).unwrap();
    data
}

fn assert_nothing_arrives(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_millis(30)))
        .unwrap();
    let mut buf = [0u8; 16];
    match stream.read(&mut buf) {
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) => {}
        other => panic!("expected a read timeout, got {other:?}"),
    }
    stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
}

fn poll_a_few_times(un: &mut UserNet) {
    for _ in 0..5 {
        un.poll();
        std::thread::sleep(Duration::from_millis(2));
    }
}

mod tcp {
    use super::*;

    #[test]
    fn syn_ack_carries_the_documented_fields() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        un.isn = 0x1000_0000;
        let mut guest = Guest::new(5000, port, 1000);
        guest.send(&mut un, SYN, &[]);
        let synack = wait_for_tcp(&mut un, |v| v.flags & SYN != 0);
        assert_eq!(synack.flags, SYN | ACK);
        assert_eq!(synack.seq, 0x1000_0000 + ISN_STEP);
        assert_eq!(synack.ack, 1001);
        assert_eq!(synack.window, 65535);
        assert_eq!(synack.options, [TCP_OPT_MSS, 4, 0x05, 0xb4]);
        assert_eq!((synack.src, synack.dst), (LOOPBACK_IP, GUEST_IP));
        assert_eq!((synack.sport, synack.dport), (port, 5000));
        assert!(synack.data.is_empty());
        listener.accept().unwrap();
    }

    #[test]
    fn each_connection_gets_a_new_initial_sequence_number() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut isns = Vec::new();
        for guest_port in [5000u16, 5001, 5002] {
            let mut guest = Guest::new(guest_port, port, 1);
            guest.send(&mut un, SYN, &[]);
            isns.push(wait_for_tcp(&mut un, |v| v.dport == guest_port && v.flags == SYN | ACK).seq);
            listener.accept().unwrap();
        }
        assert_eq!(isns[1], isns[0].wrapping_add(ISN_STEP));
        assert_eq!(isns[2], isns[1].wrapping_add(ISN_STEP));
    }

    #[test]
    fn handshake_completes_and_the_host_connection_is_established() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (_server, isn) = handshake(&mut un, &mut guest, &listener);
        let flow = &un.tcp[&guest.key()];
        assert!(matches!(flow.conn, Conn::Open(_)));
        assert_eq!(flow.snd_nxt, isn.wrapping_add(1));
        assert_eq!(flow.snd_una, isn.wrapping_add(1), "our SYN is acknowledged");
        assert_eq!(flow.rcv_nxt, 1001);
        assert_eq!(flow.mss, MSS);
        assert_eq!(flow.guest_window, 60000);
    }

    #[test]
    fn the_guests_mss_option_limits_our_segment_size() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let with_mss = TcpHeader {
            src: GUEST_IP,
            dst: LOOPBACK_IP,
            sport: 5000,
            dport: port,
            seq: 1000,
            ack: 0,
            flags: SYN,
            window: 60000,
            advertise_mss: true,
        };
        let mut syn = tcp_segment(&with_mss, &[]);
        syn[22..24].copy_from_slice(&536u16.to_be_bytes());
        un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_TCP, &syn));
        guest.seq += 1;
        let synack = wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
        assert_eq!(
            synack.options,
            [TCP_OPT_MSS, 4, 0x05, 0xb4],
            "we still advertise our own"
        );
        guest.ack = synack.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        let (mut server, _) = listener.accept().unwrap();
        assert_eq!(un.tcp[&guest.key()].mss, 536);
        let data = splitmix_bytes(11, 5000);
        server.write_all(&data).unwrap();
        assert_eq!(receive_data(&mut un, &mut guest, 5000, 536), data);
    }

    #[test]
    fn data_from_the_guest_reaches_the_host_and_is_acknowledged() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, ACK | PSH, b"hello");
        let ack = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(ack.flags, ACK);
        assert_eq!(ack.ack, 1006);
        assert_eq!(ack.seq, isn.wrapping_add(1));
        assert!(ack.data.is_empty());
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 5), b"hello");
    }

    #[test]
    fn guest_data_is_delivered_in_order_across_segments() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        let data = splitmix_bytes(21, 6000);
        for chunk in data.chunks(MSS) {
            guest.send(&mut un, ACK, chunk);
        }
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, data.len()), data);
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 1001 + data.len() as u32);
    }

    #[test]
    fn data_from_the_host_is_segmented_to_the_mss_and_sequenced() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        guest.ack = isn.wrapping_add(1);
        let data = splitmix_bytes(31, 4000);
        server.write_all(&data).unwrap();
        let first = wait_for_tcp(&mut un, |v| !v.data.is_empty());
        assert_eq!(first.flags, PSH | ACK);
        assert_eq!(first.ack, 1001);
        assert_eq!(first.seq, isn.wrapping_add(1));
        assert!(first.data.len() <= MSS);
        let mut received = first.data.clone();
        guest.ack = guest.ack.wrapping_add(first.data.len() as u32);
        guest.send(&mut un, ACK, &[]);
        received.extend(receive_data(
            &mut un,
            &mut guest,
            data.len() - received.len(),
            MSS,
        ));
        assert_eq!(received, data);
    }

    #[test]
    fn host_data_is_limited_to_the_guests_advertised_window() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.window = 100;
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        server.write_all(&splitmix_bytes(41, 1000)).unwrap();
        let mut got = 0;
        while got < 100 {
            let segment = wait_for_tcp(&mut un, |v| !v.data.is_empty());
            got += segment.data.len();
        }
        assert_eq!(got, 100, "never more than the window in flight");
        poll_a_few_times(&mut un);
        assert!(
            drain(&mut un)
                .iter()
                .all(|f| tcp_view(f).is_none_or(|v| v.data.is_empty()))
        );
        let flow = &un.tcp[&guest.key()];
        assert_eq!(flow.snd_nxt.wrapping_sub(flow.snd_una), 100);
        assert_eq!(flow.snd_nxt, isn.wrapping_add(1 + 100));
    }

    #[test]
    fn an_ack_with_a_bigger_window_releases_the_rest() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.window = 100;
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        let data = splitmix_bytes(42, 1000);
        server.write_all(&data).unwrap();
        let first = wait_for_tcp(&mut un, |v| !v.data.is_empty());
        let mut received = first.data.clone();
        while received.len() < 100 {
            received.extend(wait_for_tcp(&mut un, |v| !v.data.is_empty()).data);
        }
        guest.ack = isn.wrapping_add(1 + received.len() as u32);
        guest.window = 60000;
        guest.send(&mut un, ACK, &[]);
        received.extend(receive_data(
            &mut un,
            &mut guest,
            data.len() - received.len(),
            MSS,
        ));
        assert_eq!(received, data);
    }

    #[test]
    fn a_zero_window_stops_the_host_data() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.window = 0;
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        server.write_all(b"stalled").unwrap();
        poll_a_few_times(&mut un);
        assert!(
            drain(&mut un)
                .iter()
                .all(|f| tcp_view(f).is_none_or(|v| v.data.is_empty()))
        );
        guest.ack = isn.wrapping_add(1);
        guest.window = 1000;
        guest.send(&mut un, ACK, &[]);
        assert_eq!(receive_data(&mut un, &mut guest, 7, MSS), b"stalled");
    }

    #[test]
    fn retransmitted_guest_data_is_delivered_once() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        let resend_from = guest.seq;
        guest.send(&mut un, ACK, b"abc");
        guest.seq = resend_from;
        guest.send(&mut un, ACK, b"abc");
        let acks: Vec<TcpView> = drain(&mut un).iter().filter_map(|f| tcp_view(f)).collect();
        assert_eq!(acks.len(), 2);
        assert!(
            acks.iter().all(|a| a.ack == 1004),
            "both acknowledge up to 1004"
        );
        guest.send(&mut un, ACK, b"def");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 6), b"abcdef");
        assert_nothing_arrives(&mut server);
    }

    #[test]
    fn partially_overlapping_retransmit_delivers_only_the_new_bytes() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, ACK, b"abc");
        guest.seq = 1001 + 1;
        guest.send(&mut un, ACK, b"bcdef");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 6), b"abcdef");
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 1007);
    }

    #[test]
    fn a_segment_beyond_a_gap_is_dropped_and_answered_with_a_duplicate_ack() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.seq = 1001 + 10;
        guest.send(&mut un, ACK, b"late");
        let reply = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(reply.ack, 1001, "still waiting for the missing bytes");
        poll_a_few_times(&mut un);
        assert_nothing_arrives(&mut server);
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 1001);
        guest.seq = 1001;
        guest.send(&mut un, ACK, b"0123456789");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 10), b"0123456789");
    }

    #[test]
    fn guest_sequence_numbers_wrap_around() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 0xffff_fff0);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 0xffff_fff1);
        guest.send(&mut un, ACK, &[7u8; 40]);
        let ack = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(ack.ack, 0x19, "0xfffffff1 + 40 wraps to 0x19");
        guest.send(&mut un, ACK, b"after");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 45)[40..], *b"after");
    }

    #[test]
    fn our_sequence_numbers_wrap_around_and_acks_still_advance_snd_una() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        un.isn = 0xffff_fffd - ISN_STEP;
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        assert_eq!(isn, 0xffff_fffd);
        guest.ack = isn.wrapping_add(1);
        server.write_all(&[9u8; 10]).unwrap();
        let segment = wait_for_tcp(&mut un, |v| !v.data.is_empty());
        assert_eq!(segment.seq, 0xffff_fffe);
        assert_eq!(segment.data.len(), 10);
        let after = 0xffff_fffeu32.wrapping_add(10);
        assert_eq!(after, 8);
        assert_eq!(un.tcp[&guest.key()].snd_nxt, 8);
        guest.ack = after;
        guest.send(&mut un, ACK, &[]);
        assert_eq!(
            un.tcp[&guest.key()].snd_una,
            8,
            "the ack past the wrap is accepted"
        );
    }

    #[test]
    fn acks_outside_the_sent_range_do_not_move_snd_una() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, isn) = handshake(&mut un, &mut guest, &listener);
        server.write_all(&[1u8; 100]).unwrap();
        let mut got = 0;
        while got < 100 {
            got += wait_for_tcp(&mut un, |v| !v.data.is_empty()).data.len();
        }
        let base = isn.wrapping_add(1);
        guest.ack = base.wrapping_add(201);
        guest.send(&mut un, ACK, &[]);
        assert_eq!(
            un.tcp[&guest.key()].snd_una,
            base,
            "acknowledges data we never sent"
        );
        guest.ack = base.wrapping_add(100);
        guest.send(&mut un, ACK, &[]);
        assert_eq!(un.tcp[&guest.key()].snd_una, base.wrapping_add(100));
        guest.ack = base.wrapping_add(50);
        guest.send(&mut un, ACK, &[]);
        assert_eq!(
            un.tcp[&guest.key()].snd_una,
            base.wrapping_add(100),
            "stale ack"
        );
    }

    #[test]
    fn guest_window_follows_each_ack() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (_server, isn) = handshake(&mut un, &mut guest, &listener);
        guest.ack = isn.wrapping_add(1);
        guest.window = 1234;
        guest.send(&mut un, ACK, &[]);
        assert_eq!(un.tcp[&guest.key()].guest_window, 1234);
    }

    #[test]
    fn data_the_host_socket_takes_at_once_leaves_the_acknowledged_window_open() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, ACK, &[0u8; 1000]);
        let ack = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(
            ack.window, 65535,
            "written to the host before the ACK was built"
        );
        assert_eq!(read_exactly(&mut server, 1000), vec![0u8; 1000]);
    }

    /// A guest that filled the window must hear that it opened again once `poll` moves the
    /// backlog to the host; without the update it waits for its persist timer (200 ms or more).
    #[test]
    fn a_window_reopened_by_poll_is_announced_to_the_guest() {
        const CHUNK: usize = MSS;
        const MAX_BYTES: usize = 64 << 20;
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        // The host never reads, so its socket fills and the backlog builds up behind it.
        let mut sent = 0;
        loop {
            guest.send(&mut un, ACK, &[0x5a; CHUNK]);
            sent += CHUNK;
            un.poll();
            let window = drain(&mut un)
                .iter()
                .filter_map(|f| tcp_view(f))
                .map(|v| v.window)
                .min()
                .unwrap();
            if (window as usize) < ADVERTISED_WINDOW_MAX / 2 {
                break;
            }
            assert!(sent < MAX_BYTES, "the host socket never filled");
        }
        let reader = std::thread::spawn(move || read_exactly(&mut server, sent).len());
        let update = wait_for_tcp(&mut un, |v| {
            v.flags == ACK && v.data.is_empty() && v.window as usize == ADVERTISED_WINDOW_MAX
        });
        assert_eq!(update.ack, 1001u32.wrapping_add(sent as u32));
        assert_eq!(reader.join().unwrap(), sent);
    }

    #[test]
    fn guest_fin_is_acknowledged_and_half_closes_the_host_socket() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, FIN | ACK, &[]);
        let ack = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(ack.flags, ACK);
        assert_eq!(ack.ack, 1002, "the FIN takes one sequence number");
        poll_a_few_times(&mut un);
        let mut buf = [0u8; 4];
        assert_eq!(server.read(&mut buf).unwrap(), 0, "host sees end of stream");
        assert!(un.tcp[&guest.key()].host_write_shutdown);
    }

    #[test]
    fn guest_data_before_the_fin_is_flushed_to_the_host_first() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, FIN | ACK | PSH, b"last words");
        poll_a_few_times(&mut un);
        let mut all = Vec::new();
        server.read_to_end(&mut all).unwrap();
        assert_eq!(all, b"last words");
        assert_eq!(
            tcp_view(&drain(&mut un).remove(0)).unwrap().ack,
            1000 + 1 + 10 + 1
        );
    }

    #[test]
    fn a_retransmitted_fin_is_acknowledged_without_advancing_twice() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (_server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, FIN | ACK, &[]);
        guest.seq = 1001;
        guest.send(&mut un, FIN | ACK, &[]);
        let acks: Vec<u32> = drain(&mut un)
            .iter()
            .filter_map(|f| tcp_view(f))
            .map(|v| v.ack)
            .collect();
        assert_eq!(acks, vec![1002, 1002]);
    }

    #[test]
    fn host_close_sends_fin_and_the_flow_ends_after_both_sides_are_acknowledged() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (server, isn) = handshake(&mut un, &mut guest, &listener);
        server.shutdown(Shutdown::Write).unwrap();
        let fin = wait_for_tcp(&mut un, |v| v.flags & FIN != 0);
        assert_eq!(fin.flags, FIN | ACK);
        assert_eq!(fin.seq, isn.wrapping_add(1));
        assert!(un.tcp[&guest.key()].fin_sent);
        guest.ack = fin.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        un.poll();
        assert_eq!(un.tcp.len(), 1, "guest has not closed its side yet");
        guest.send(&mut un, FIN | ACK, &[]);
        un.poll();
        assert!(un.tcp.is_empty(), "both FINs acknowledged");
    }

    #[test]
    fn a_fin_only_ends_the_flow_once_ours_is_acknowledged() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (server, isn) = handshake(&mut un, &mut guest, &listener);
        server.shutdown(Shutdown::Write).unwrap();
        let fin = wait_for_tcp(&mut un, |v| v.flags & FIN != 0);
        guest.ack = isn.wrapping_add(1);
        guest.send(&mut un, FIN | ACK, &[]);
        un.poll();
        assert_eq!(un.tcp.len(), 1, "our FIN is still unacknowledged");
        guest.ack = fin.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        un.poll();
        assert!(un.tcp.is_empty());
    }

    #[test]
    fn guest_rst_removes_the_flow_and_closes_the_host_socket() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (mut server, _) = handshake(&mut un, &mut guest, &listener);
        guest.send(&mut un, RST | ACK, &[]);
        assert!(drain(&mut un).is_empty(), "a RST is never answered");
        un.poll();
        assert!(un.tcp.is_empty());
        let mut buf = [0u8; 4];
        assert!(matches!(server.read(&mut buf), Ok(0) | Err(_)));
    }

    #[test]
    fn connection_refused_by_the_host_resets_the_guest() {
        let port = listener().1;
        let mut un = UserNet::new();
        let mut guest = Guest::new(5001, port, 7);
        guest.send(&mut un, SYN, &[]);
        let reset = wait_for_tcp(&mut un, |v| v.flags & RST != 0);
        assert_eq!(reset.flags, RST | ACK);
        assert_eq!(reset.ack, 8, "acknowledges the SYN");
        assert_eq!(reset.seq, 0);
        assert_eq!(reset.window, 0);
        assert_eq!((reset.src, reset.dst), (LOOPBACK_IP, GUEST_IP));
        assert_eq!((reset.sport, reset.dport), (port, 5001));
        un.poll();
        assert!(un.tcp.is_empty(), "the failed flow is cleaned up");
    }

    #[test]
    fn a_refused_connection_can_be_retried() {
        let (listener, port) = listener();
        drop(listener);
        let mut un = UserNet::new();
        let mut guest = Guest::new(5002, port, 7);
        guest.send(&mut un, SYN, &[]);
        wait_for_tcp(&mut un, |v| v.flags & RST != 0);
        un.poll();
        assert!(un.tcp.is_empty());
        let relistened = TcpListener::bind(("127.0.0.1", port));
        if let Ok(relistened) = relistened {
            let mut retry = Guest::new(5002, port, 100);
            retry.send(&mut un, SYN, &[]);
            wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
            relistened.accept().unwrap();
        }
    }

    #[test]
    fn a_repeated_syn_while_connecting_does_not_create_a_second_flow() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.send(&mut un, SYN, &[]);
        guest.seq = 1000;
        guest.send(&mut un, SYN, &[]);
        assert_eq!(un.tcp.len(), 1);
        wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
        poll_a_few_times(&mut un);
        let extra: Vec<_> = drain(&mut un).iter().filter_map(|f| tcp_view(f)).collect();
        assert!(
            extra.iter().all(|v| v.flags != SYN | ACK),
            "exactly one SYN-ACK"
        );
        listener.accept().unwrap();
    }

    #[test]
    fn a_syn_on_an_established_flow_is_ignored() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (_server, _) = handshake(&mut un, &mut guest, &listener);
        guest.seq = 1000;
        guest.send(&mut un, SYN, &[]);
        assert!(drain(&mut un).is_empty());
        assert_eq!(un.tcp.len(), 1);
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 1001);
    }

    #[test]
    fn data_before_the_host_connection_exists_is_ignored() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.send(&mut un, SYN, &[]);
        guest.send(&mut un, ACK, b"too early");
        assert!(drain(&mut un).is_empty());
        assert_eq!(un.tcp[&guest.key()].rcv_nxt, 1001);
        assert!(un.tcp[&guest.key()].to_host.is_empty());
        listener.accept().unwrap();
    }

    #[test]
    fn segments_for_unknown_flows_are_reset() {
        let mut un = UserNet::new();
        let mut guest = Guest::new(6000, 9, 500);
        guest.ack = 12345;
        guest.send(&mut un, ACK, &[]);
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(
            reset.flags, RST,
            "an ACK is answered with a bare RST at its ack number"
        );
        assert_eq!(reset.seq, 12345);
        assert_eq!(reset.window, 0);
        assert_eq!((reset.src, reset.dst), (LOOPBACK_IP, GUEST_IP));
        assert_eq!((reset.sport, reset.dport), (9, 6000));
    }

    #[test]
    fn segments_without_ack_for_unknown_flows_are_reset_with_rst_ack() {
        let mut un = UserNet::new();
        let mut data = Guest::new(6000, 9, 500);
        data.send(&mut un, PSH, b"12345");
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 505));

        let mut fin = Guest::new(6001, 9, 900);
        fin.send(&mut un, FIN, &[]);
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 901));

        let mut bare = Guest::new(6002, 9, 77);
        bare.send(&mut un, 0, &[]);
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 77));
    }

    #[test]
    fn reset_acknowledgement_wraps_around() {
        let mut un = UserNet::new();
        let mut guest = Guest::new(6000, 9, u32::MAX);
        guest.send(&mut un, PSH, b"ab");
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!(reset.ack, 1, "0xffffffff + 2");
    }

    #[test]
    fn a_syn_with_ack_for_an_unknown_flow_is_reset_not_connected() {
        let mut un = UserNet::new();
        let mut guest = Guest::new(6000, 9, 500);
        guest.ack = 4242;
        guest.send(&mut un, SYN | ACK, &[]);
        let reset = tcp_view(&drain(&mut un).remove(0)).unwrap();
        assert_eq!((reset.flags, reset.seq), (RST, 4242));
        assert!(un.tcp.is_empty());
    }

    #[test]
    fn a_rst_for_an_unknown_flow_is_silently_dropped() {
        let mut un = UserNet::new();
        let mut guest = Guest::new(6000, 9, 500);
        guest.send(&mut un, RST, &[]);
        guest.send(&mut un, RST | ACK, &[]);
        assert!(drain(&mut un).is_empty());
    }

    #[test]
    fn malformed_tcp_is_ignored() {
        let mut un = UserNet::new();
        for len in 0..TCP_HEADER_LEN {
            un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_TCP, &vec![0u8; len]));
        }
        let mut segment = Guest::new(1, 2, 3).segment(SYN, &[]);
        segment[TCP_DATA_OFFSET] = 0xf0;
        un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_TCP, &segment));
        assert!(drain(&mut un).is_empty());
        assert!(un.tcp.is_empty());
    }

    #[test]
    fn dns_over_tcp_goes_to_the_host_nameserver() {
        let (nameserver, nameserver_port) = listener();
        let mut un = UserNet::new();
        un.dns = SocketAddr::from((LOOPBACK_IP, nameserver_port));
        let mut guest = Guest::new(5000, DNS_PORT, 1000);
        guest.dst = DNS_IP;
        guest.send(&mut un, SYN, &[]);
        let synack = wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
        assert_eq!((synack.src, synack.sport), (DNS_IP, DNS_PORT));
        nameserver.accept().unwrap();
    }

    #[test]
    fn tcp_to_the_gateway_address_reaches_host_loopback() {
        let (host_service, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.dst = GATEWAY_IP;
        let (mut server, _) = handshake(&mut un, &mut guest, &host_service);
        assert_eq!(un.tcp[&guest.key()].guest_ip, GUEST_IP);
        guest.send(&mut un, ACK, b"hi host");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 7), b"hi host");
        server.write_all(b"hi guest").unwrap();
        let segment = wait_for_tcp(&mut un, |v| !v.data.is_empty());
        assert_eq!(segment.src, GATEWAY_IP);
        assert_eq!(segment.data, b"hi guest");
    }

    #[test]
    fn flow_header_swaps_the_endpoints_and_uses_the_current_window_and_ack() {
        let flow = manual_flow(None);
        let key = FlowKey {
            guest_port: 4321,
            remote_ip: [93, 184, 216, 34],
            remote_port: 443,
        };
        let header = flow.header(&key, ACK, 777, true);
        assert_eq!(header.src, [93, 184, 216, 34]);
        assert_eq!(header.dst, GUEST_IP);
        assert_eq!((header.sport, header.dport), (443, 4321));
        assert_eq!((header.seq, header.ack), (777, flow.rcv_nxt));
        assert_eq!(header.flags, ACK);
        assert_eq!(header.window, 65535);
        assert!(header.advertise_mss);
    }

    #[test]
    fn flow_header_window_reflects_the_backlog() {
        let mut flow = manual_flow(None);
        flow.to_host.extend(std::iter::repeat_n(0u8, 40000));
        let key = FlowKey {
            guest_port: 1,
            remote_ip: LOOPBACK_IP,
            remote_port: 2,
        };
        assert_eq!(flow.header(&key, ACK, 0, false).window, 65535 - 40000);
    }

    fn manual_flow(stream: Option<TcpStream>) -> TcpFlow {
        TcpFlow {
            conn: stream.map_or(Conn::Connecting, Conn::Open),
            guest_ip: GUEST_IP,
            mss: MSS,
            rcv_nxt: 1000,
            snd_nxt: 5000,
            snd_una: 5000,
            guest_window: 60000,
            to_host: VecDeque::new(),
            guest_fin: false,
            host_write_shutdown: false,
            fin_sent: false,
            dead: false,
            window_sent: ADVERTISED_WINDOW_MAX as u16,
        }
    }

    /// An open flow whose host socket already has `len` bytes waiting; returns the peer so the
    /// connection stays up.
    fn flow_with_pending_data(un: &mut UserNet, guest_port: u16, len: usize) -> TcpStream {
        let (listener, port) = listener();
        let ours = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let (mut peer, _) = listener.accept().unwrap();
        peer.write_all(&vec![guest_port as u8; len]).unwrap();
        ours.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
        ours.peek(&mut [0u8; 1]).unwrap();
        ours.set_nonblocking(true).unwrap();
        un.tcp.insert(
            FlowKey {
                guest_port,
                remote_ip: LOOPBACK_IP,
                remote_port: 80,
            },
            manual_flow(Some(ours)),
        );
        peer
    }

    /// One poll with room for exactly `room` new frames; returns the guest ports served.
    fn poll_with_room(un: &mut UserNet, room: usize) -> Vec<u16> {
        un.out.clear();
        for _ in 0..OUT_QUEUE_LIMIT - room {
            un.out.push_back(Vec::new());
        }
        un.poll();
        drain(un)
            .into_iter()
            .skip(OUT_QUEUE_LIMIT - room)
            .map(|frame| tcp_view(&frame).unwrap().dport)
            .collect()
    }

    #[test]
    fn a_small_per_poll_budget_is_shared_round_robin_between_flows() {
        let mut un = UserNet::new();
        let _peers: Vec<TcpStream> = [1u16, 2, 3]
            .iter()
            .map(|port| flow_with_pending_data(&mut un, *port, 8000))
            .collect();
        let mut served = Vec::new();
        for _ in 0..9 {
            let ports = poll_with_room(&mut un, 1);
            assert_eq!(ports.len(), 1);
            served.push(ports[0]);
        }
        assert_eq!(served, vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn every_flow_makes_progress_under_a_budget_smaller_than_the_flow_count() {
        let mut un = UserNet::new();
        let ports = [10u16, 20, 30, 40, 50];
        let _peers: Vec<TcpStream> = ports
            .iter()
            .map(|port| flow_with_pending_data(&mut un, *port, 6000))
            .collect();
        let mut sent = std::collections::HashMap::new();
        for _ in 0..ports.len() * 2 {
            for port in poll_with_room(&mut un, 2) {
                *sent.entry(port).or_insert(0usize) += 1;
            }
        }
        for port in ports {
            assert!(
                sent[&port] >= 3,
                "flow {port} was served {} times",
                sent[&port]
            );
        }
    }

    #[test]
    fn a_full_output_queue_stops_reading_host_sockets() {
        let mut un = UserNet::new();
        let _peer = flow_with_pending_data(&mut un, 7, 3000);
        assert!(
            poll_with_room(&mut un, 0).is_empty(),
            "no budget, no frames"
        );
        assert_eq!(
            poll_with_room(&mut un, 1),
            vec![7],
            "the data was left on the socket"
        );
    }

    #[test]
    fn service_order_is_sorted_and_rotates_by_one_each_poll() {
        let mut un = UserNet::new();
        let keys: Vec<FlowKey> = (1u16..=3)
            .map(|port| FlowKey {
                guest_port: port,
                remote_ip: LOOPBACK_IP,
                remote_port: 80,
            })
            .collect();
        for key in &keys {
            un.tcp.insert(*key, manual_flow(None));
        }
        let ports = |order: Vec<FlowKey>| order.iter().map(|k| k.guest_port).collect::<Vec<_>>();
        assert_eq!(ports(un.tcp_service_order()), vec![1, 2, 3]);
        assert_eq!(ports(un.tcp_service_order()), vec![2, 3, 1]);
        assert_eq!(ports(un.tcp_service_order()), vec![3, 1, 2]);
        assert_eq!(ports(un.tcp_service_order()), vec![1, 2, 3]);
    }

    #[test]
    fn service_order_copes_with_flows_coming_and_going() {
        let mut un = UserNet::new();
        let key = |port: u16| FlowKey {
            guest_port: port,
            remote_ip: LOOPBACK_IP,
            remote_port: 80,
        };
        let ports = |order: Vec<FlowKey>| order.iter().map(|k| k.guest_port).collect::<Vec<_>>();
        assert!(un.tcp_service_order().is_empty());
        un.tcp.insert(key(5), manual_flow(None));
        assert_eq!(ports(un.tcp_service_order()), vec![5]);
        assert_eq!(ports(un.tcp_service_order()), vec![5]);
        un.tcp.insert(key(9), manual_flow(None));
        un.tcp.insert(key(7), manual_flow(None));
        assert_eq!(
            ports(un.tcp_service_order()),
            vec![7, 9, 5],
            "continues after flow 5"
        );
        un.tcp.remove(&key(7));
        assert_eq!(
            ports(un.tcp_service_order()),
            vec![9, 5],
            "the remembered flow is gone"
        );
        un.tcp.clear();
        assert!(un.tcp_service_order().is_empty());
    }

    #[test]
    fn full_duplex_echo_survives_a_close_from_both_ends() {
        let (listener, port) = listener();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => stream.write_all(&buf[..n]).unwrap(),
                }
            }
        });
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.send(&mut un, SYN, &[]);
        let synack = wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
        guest.ack = synack.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        let payload = splitmix_bytes(77, 5000);
        guest.send(&mut un, ACK | PSH, &payload);
        assert_eq!(
            receive_data(&mut un, &mut guest, payload.len(), MSS),
            payload
        );
        guest.send(&mut un, FIN | ACK, &[]);
        let fin = wait_for_tcp(&mut un, |v| v.flags & FIN != 0);
        guest.ack = fin.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        un.poll();
        assert!(un.tcp.is_empty(), "flow closed after both FINs are acked");
    }
}

mod original_scenarios {
    use super::*;

    fn dhcp_msg(ty: u8) -> Vec<u8> {
        let mut d = vec![0u8; 240];
        d[0] = 1;
        d[1] = 1;
        d[2] = 6;
        d[4..8].copy_from_slice(&[1, 2, 3, 4]);
        d[28..34].copy_from_slice(&GUEST_MAC);
        d[236..240].copy_from_slice(&[0x63, 0x82, 0x53, 0x63]);
        d.extend_from_slice(&[53, 1, ty, 255]);
        d
    }

    #[test]
    fn dhcp_offer_and_ack() {
        let mut un = UserNet::new();
        for (req, reply) in [(1u8, 2u8), (3, 5)] {
            let u = udp_segment([0; 4], [255; 4], 68, 67, &dhcp_msg(req));
            let p = ipv4_packet(1, [0; 4], [255; 4], 17, &u);
            un.input(&ethernet_frame(BROADCAST_MAC, GUEST_MAC, 0x0800, &p));
            let out = drain(&mut un);
            assert_eq!(out.len(), 1);
            let ip = parse_ipv4(&out[0]).unwrap();
            assert_eq!(ip.proto, 17);
            let d = &ip.payload[8..];
            assert_eq!(d[0], 2);
            assert_eq!(d[4..8], [1, 2, 3, 4]);
            assert_eq!(d[16..20], GUEST_IP);
            assert_eq!(d[240..243], [53, 1, reply]);
        }
    }

    #[test]
    fn arp_and_ping() {
        let mut un = UserNet::new();
        let mut arp = vec![0, 1, 8, 0, 6, 4, 0, 1];
        arp.extend_from_slice(&GUEST_MAC);
        arp.extend_from_slice(&GUEST_IP);
        arp.extend_from_slice(&[0; 6]);
        arp.extend_from_slice(&GATEWAY_IP);
        un.input(&ethernet_frame(BROADCAST_MAC, GUEST_MAC, 0x0806, &arp));
        let out = drain(&mut un);
        assert_eq!(out[0][20..22], [0, 2]);
        assert_eq!(out[0][22..28], GATEWAY_MAC);

        let mut echo = vec![8, 0, 0, 0, 0, 1, 0, 1, 9, 9];
        let c = internet_checksum(&echo);
        echo[2..4].copy_from_slice(&c.to_be_bytes());
        un.input(&guest_ip_frame([8, 8, 8, 8], 1, &echo));
        let out = drain(&mut un);
        let ip = parse_ipv4(&out[0]).unwrap();
        assert_eq!((ip.src, ip.dst, ip.payload[0]), ([8, 8, 8, 8], GUEST_IP, 0));
        assert_eq!(internet_checksum(ip.payload), 0);
    }

    #[test]
    fn udp_nat_roundtrip() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        server
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let port = server.local_addr().unwrap().port();
        let mut un = UserNet::new();
        let u = udp_segment(GUEST_IP, LOOPBACK_IP, 4000, port, b"ping");
        un.input(&guest_ip_frame(LOOPBACK_IP, 17, &u));
        let mut buf = [0u8; 64];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"ping");
        server.send_to(b"pong", from).unwrap();
        let f = wait_for(&mut un, |f| parse_ipv4(f).is_some_and(|i| i.proto == 17));
        let ip = parse_ipv4(&f).unwrap();
        assert_eq!(ip.src, LOOPBACK_IP);
        assert_eq!(u16::from_be_bytes([ip.payload[0], ip.payload[1]]), port);
        assert_eq!(u16::from_be_bytes([ip.payload[2], ip.payload[3]]), 4000);
        assert_eq!(&ip.payload[8..], b"pong");
        assert_eq!(transport_checksum(ip.src, ip.dst, 17, ip.payload), 0);
    }

    #[test]
    fn tcp_proxy_echo_and_close() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            loop {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => s.write_all(&buf[..n]).unwrap(),
                }
            }
        });
        let mut un = UserNet::new();
        let mut g = Guest::new(5000, port, 1000);
        g.send(&mut un, SYN, &[]);
        let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 == SYN | ACK));
        let (_, srv_seq, ack, _) = tcp_of(&f).unwrap();
        assert_eq!(ack, 1001);
        g.ack = srv_seq.wrapping_add(1);
        g.send(&mut un, ACK, &[]);

        let payload: Vec<u8> = (0..5000u32).map(|i| i as u8).collect();
        g.send(&mut un, ACK | PSH, &payload);
        let mut echoed = Vec::new();
        let mut next = g.ack;
        while echoed.len() < payload.len() {
            let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| !t.3.is_empty()));
            let (_, seq, _, data) = tcp_of(&f).unwrap();
            assert_eq!(seq, next, "in-order delivery");
            next = next.wrapping_add(data.len() as u32);
            echoed.extend_from_slice(&data);
            g.ack = next;
            g.send(&mut un, ACK, &[]);
        }
        assert_eq!(echoed, payload);

        g.send(&mut un, FIN | ACK, &[]);
        let f = wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 & FIN != 0));
        let (_, seq, _, _) = tcp_of(&f).unwrap();
        g.ack = seq.wrapping_add(1);
        g.send(&mut un, ACK, &[]);
        un.poll();
        assert!(un.tcp.is_empty(), "flow closed after both FINs are acked");
    }

    #[test]
    fn tcp_refused_gets_rst() {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let mut un = UserNet::new();
        let mut g = Guest::new(5001, port, 7);
        g.send(&mut un, SYN, &[]);
        wait_for(&mut un, |f| tcp_of(f).is_some_and(|t| t.0 & RST != 0));
    }
}

mod robustness {
    use super::*;

    /// Rewrites the destination of any TCP/UDP packet to the DNS address, which the tests point at
    /// a closed loopback port, so fuzzing can never reach a real host.
    fn contain(frame: &mut [u8]) {
        if let Some(ip) = parse_ipv4(frame)
            && matches!(ip.proto, IP_PROTO_TCP | IP_PROTO_UDP)
        {
            frame[ETH_HEADER_LEN + 16..ETH_HEADER_LEN + 20].copy_from_slice(&DNS_IP);
        }
    }

    fn contained_stack() -> UserNet {
        let mut un = UserNet::new();
        un.dns = closed_udp_addr();
        un.udp_flow_limit = 8;
        un
    }

    fn check_output(un: &mut UserNet) {
        for frame in drain(un) {
            assert!(frame.len() >= ETH_HEADER_LEN);
            match be_u16(&frame[ETH_TYPE]) {
                ETHERTYPE_IPV4 => {
                    assert_eq!(
                        reference_checksum(
                            &frame[ETH_HEADER_LEN..ETH_HEADER_LEN + IPV4_HEADER_LEN]
                        ),
                        0
                    );
                    let _ = tcp_view(&frame);
                }
                ETHERTYPE_ARP => assert_eq!(frame.len(), ARP_REQUEST_FRAME_LEN),
                other => panic!("unexpected ethertype {other:#x}"),
            }
        }
    }

    fn valid_frames() -> Vec<Vec<u8>> {
        let syn = Guest::new(1000, 80, 5).segment(SYN, b"");
        let data = Guest::new(1000, 80, 5).segment(ACK | PSH, b"payload");
        let mut frames = vec![
            arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP),
            guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, 1, b"ping"),
            ),
            udp_frame(1000, DNS_IP, DNS_PORT, b"query"),
            guest_ip_frame(DNS_IP, IP_PROTO_TCP, &syn),
            guest_ip_frame(DNS_IP, IP_PROTO_TCP, &data),
        ];
        let mut discover = vec![0u8; DHCP_OPTIONS_OFFSET];
        discover[BOOTP_OP] = BOOTP_REQUEST;
        discover[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
        discover.extend_from_slice(&[DHCP_OPT_MESSAGE_TYPE, 1, DHCP_DISCOVER, DHCP_OPT_END]);
        frames.push(udp_frame(
            68,
            LIMITED_BROADCAST_IP,
            DHCP_SERVER_PORT,
            &discover,
        ));
        frames
    }

    #[test]
    fn every_truncation_of_every_valid_frame_is_safe() {
        for frame in valid_frames() {
            for len in 0..=frame.len() {
                let mut un = contained_stack();
                un.input(&frame[..len]);
                un.poll();
                check_output(&mut un);
            }
        }
    }

    #[test]
    fn every_single_byte_corruption_of_every_valid_frame_is_safe() {
        let mut rng = XorShift::new(FUZZ_SEED + 4);
        for frame in valid_frames() {
            for position in 0..frame.len() {
                let mut mutated = frame.clone();
                mutated[position] = rng.next_u64() as u8;
                contain(&mut mutated);
                let mut un = contained_stack();
                un.input(&mutated);
                un.poll();
                check_output(&mut un);
            }
        }
    }

    #[test]
    fn random_frames_never_panic() {
        let mut rng = XorShift::new(FUZZ_SEED + 5);
        let mut un = contained_stack();
        for round in 0..3000 {
            let len = rng.below(200);
            let mut frame = rng.bytes(len);
            if len >= ETH_HEADER_LEN {
                let ethertype = match rng.below(3) {
                    0 => ETHERTYPE_ARP,
                    _ => ETHERTYPE_IPV4,
                };
                frame[ETH_TYPE].copy_from_slice(&ethertype.to_be_bytes());
            }
            if len > ETH_HEADER_LEN {
                // A plausible IPv4 header most of the time, so the transport parsers get reached.
                if rng.below(4) != 0 {
                    frame[ETH_HEADER_LEN] = 0x45;
                    let total = (len - ETH_HEADER_LEN) as u16;
                    if len >= ETH_HEADER_LEN + 4 {
                        frame[ETH_HEADER_LEN + 2..ETH_HEADER_LEN + 4]
                            .copy_from_slice(&total.to_be_bytes());
                    }
                    if len >= ETH_HEADER_LEN + 8 {
                        frame[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8].copy_from_slice(&[0x40, 0]);
                    }
                    if len > ETH_HEADER_LEN + 9 {
                        frame[ETH_HEADER_LEN + 9] =
                            [IP_PROTO_ICMP, IP_PROTO_TCP, IP_PROTO_UDP][rng.below(3)];
                    }
                }
            }
            contain(&mut frame);
            un.input(&frame);
            if round % 50 == 0 {
                un.poll();
                check_output(&mut un);
            }
        }
        un.poll();
        check_output(&mut un);
    }

    #[test]
    fn random_segments_on_an_established_flow_never_panic_and_never_emit_bad_checksums() {
        let (listener, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        let (_server, _) = handshake(&mut un, &mut guest, &listener);
        let mut rng = XorShift::new(FUZZ_SEED + 6);
        for round in 0..3000 {
            let flags = rng.next_u64() as u8;
            let data_len = rng.below(40);
            let data = rng.bytes(data_len);
            let mut segment = guest.segment(flags, &data);
            // Seq and ack stay near the real ones so the in-window paths run too.
            let seq = 1001u32.wrapping_add(rng.below(64) as u32).wrapping_sub(32);
            segment[TCP_SEQ].copy_from_slice(&seq.to_be_bytes());
            if rng.below(8) == 0 {
                segment[TCP_DATA_OFFSET] = rng.next_u64() as u8;
            }
            un.input(&guest_ip_frame(LOOPBACK_IP, IP_PROTO_TCP, &segment));
            if round % 25 == 0 {
                un.poll();
            }
            for frame in drain(&mut un) {
                let _ = tcp_view(&frame);
            }
        }
    }

    #[test]
    fn many_distinct_udp_flows_stay_within_the_cap() {
        let mut un = contained_stack();
        for port in 0..200u16 {
            un.input(&udp_frame(2000 + port, DNS_IP, DNS_PORT, b"x"));
        }
        assert!(un.udp.len() <= 8);
    }
}
