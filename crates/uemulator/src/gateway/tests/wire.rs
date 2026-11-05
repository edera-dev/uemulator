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
fn constants_describe_the_documented_network() {
    assert_eq!(GATEWAY_IP, [10, 0, 2, 2]);
    assert_eq!(GATEWAY_MAC, [0x52, 0x55, 10, 0, 2, 2]);
    assert_eq!(DNS_IP, [10, 0, 2, 3]);
    assert_eq!(GUEST_IP, [10, 0, 2, 15]);
    assert_eq!(SUBNET_BROADCAST, [10, 0, 2, 255]);
    assert_eq!(SUBNET_MASK, [255, 255, 255, 0]);
    assert_eq!(DHCP_LEASE_SECS, 86_400);
    for ip in [GATEWAY_IP, DNS_IP, GUEST_IP, SUBNET_BROADCAST] {
        assert_eq!(ip[..3], SUBNET_PREFIX);
    }
}
