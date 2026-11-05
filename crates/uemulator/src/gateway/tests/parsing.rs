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
