use super::*;

#[test]
fn a_new_gateway_starts_empty_and_expects_the_default_guest_mac() {
    let mut un = user_net();
    assert!(un.pop_for_guest().is_none());
    assert_eq!(un.guest_mac, GUEST_MAC);
    assert_eq!(un.udp.limit, MAX_UDP_FLOWS);
    assert!(un.udp.flows.is_empty() && un.take_requests().is_empty());
}

#[test]
fn requeued_frames_return_to_the_head_in_order() {
    let mut un = user_net();
    un.out.push_back(GuestFrame::plain(vec![3]));
    un.requeue_for_guest(GuestFrame::plain(vec![2]));
    un.requeue_for_guest(GuestFrame {
        bytes: vec![1],
        segment_size: Some(9),
    });
    let first = un.pop_for_guest().unwrap();
    assert_eq!((first.bytes, first.segment_size), (vec![1], Some(9)));
    assert_eq!(drain(&mut un), vec![vec![2], vec![3]]);
}

#[test]
fn frames_shorter_than_an_ethernet_header_are_ignored_and_not_learned_from() {
    let mut un = user_net();
    un.from_guest(&[0xff; ETH_HEADER_LEN - 1], NOW_MS);
    un.from_guest(&[], NOW_MS);
    assert!(un.pop_for_guest().is_none());
    assert_eq!(un.guest_mac, GUEST_MAC);
}

#[test]
fn arp_request_for_the_gateway_gets_the_exact_reply() {
    let mut un = user_net();
    un.from_guest(&arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP), NOW_MS);
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
        let mut un = user_net();
        un.from_guest(&arp_request_frame(GUEST_MAC, GUEST_IP, target), NOW_MS);
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
    let mut un = user_net();
    un.from_guest(
        &arp_request_frame(GUEST_MAC, [0, 0, 0, 0], GUEST_IP),
        NOW_MS,
    );
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
        let mut un = user_net();
        un.from_guest(&arp_request_frame(GUEST_MAC, GUEST_IP, target), NOW_MS);
        assert!(drain(&mut un).is_empty(), "{target:?}");
    }
}

#[test]
fn arp_replies_and_other_opcodes_are_ignored() {
    for opcode in [0u16, ARP_OP_REPLY, 3, 4, 0xffff] {
        let mut frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
        frame[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8].copy_from_slice(&opcode.to_be_bytes());
        let mut un = user_net();
        un.from_guest(&frame, NOW_MS);
        assert!(drain(&mut un).is_empty(), "opcode {opcode}");
    }
}

#[test]
fn truncated_arp_frames_are_ignored() {
    let frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
    for len in 0..ARP_REQUEST_FRAME_LEN {
        let mut un = user_net();
        un.from_guest(&frame[..len], NOW_MS);
        assert!(drain(&mut un).is_empty(), "len {len}");
    }
}

#[test]
fn arp_with_ethernet_padding_is_answered() {
    let mut frame = arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP);
    frame.resize(60, 0);
    let mut un = user_net();
    un.from_guest(&frame, NOW_MS);
    assert_eq!(drain(&mut un).len(), 1);
}

#[test]
fn arp_reply_goes_to_the_requesters_mac_and_echoes_its_address() {
    let mut un = user_net();
    un.from_guest(
        &arp_request_frame(OTHER_GUEST_MAC, [10, 0, 2, 77], GATEWAY_IP),
        NOW_MS,
    );
    let reply = drain(&mut un).remove(0);
    assert_eq!(reply[..6], OTHER_GUEST_MAC);
    assert_eq!(reply[32..38], OTHER_GUEST_MAC);
    assert_eq!(reply[38..42], [10, 0, 2, 77]);
}

#[test]
fn frames_the_stack_originates_follow_the_last_seen_guest_mac() {
    let mut un = user_net();
    un.from_guest(
        &arp_request_frame(OTHER_GUEST_MAC, GUEST_IP, GATEWAY_IP),
        NOW_MS,
    );
    drain(&mut un);
    let mut ping = guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &echo_request_payload(1, 1, b"x"));
    ping[6..12].copy_from_slice(&GUEST_MAC);
    un.from_guest(&ping, NOW_MS);
    let reply = drain(&mut un).remove(0);
    assert_eq!(reply[..6], GUEST_MAC);
    assert_eq!(reply[6..12], GATEWAY_MAC);
}

#[test]
fn non_ip_non_arp_ethertypes_are_ignored() {
    let mut un = user_net();
    un.from_guest(
        &ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x86dd, &[0x60; 60]),
        NOW_MS,
    );
    un.from_guest(
        &ethernet_frame(GATEWAY_MAC, GUEST_MAC, 0x8100, &[0; 60]),
        NOW_MS,
    );
    assert!(drain(&mut un).is_empty());
}

#[test]
fn unsupported_ip_protocols_are_ignored() {
    let mut un = user_net();
    un.from_guest(&guest_ip_frame(GATEWAY_IP, IP_PROTO_GRE, &[0; 16]), NOW_MS);
    un.from_guest(&guest_ip_frame(GATEWAY_IP, 2, &[0; 8]), NOW_MS);
    assert!(drain(&mut un).is_empty());
    assert!(un.udp.flows.is_empty() && un.take_requests().is_empty());
}

#[test]
fn ip_fragments_from_the_guest_are_dropped_not_reassembled() {
    let mut frame = guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &echo_request_payload(1, 1, b"x"));
    frame[ETH_HEADER_LEN + 6..ETH_HEADER_LEN + 8]
        .copy_from_slice(&IPV4_FLAG_MORE_FRAGMENTS.to_be_bytes());
    let mut un = user_net();
    un.from_guest(&frame, NOW_MS);
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
        let mut un = user_net();
        un.from_guest(
            &guest_ip_frame(
                dst,
                IP_PROTO_ICMP,
                &echo_request_payload(0x1234, 9, b"abcdefgh"),
            ),
            NOW_MS,
        );
        let out = drain(&mut un);
        assert_eq!(out.len(), 1, "{dst:?}");
        let ip = parse_ipv4(&out[0]).unwrap();
        assert_eq!((ip.src, ip.dst, ip.proto), (dst, GUEST_IP, IP_PROTO_ICMP));
        assert_eq!(ip.payload[0], ICMP_ECHO_REPLY);
        assert!(
            un.udp.flows.is_empty() && un.take_requests().is_empty(),
            "nothing left the host"
        );
    }
}

#[test]
fn echo_reply_preserves_identifier_sequence_code_and_payload() {
    let mut un = user_net();
    let mut request = echo_request_payload(0xbeef, 0x0102, b"the quick brown fox");
    request[1] = 5;
    let checksum = {
        request[ICMP_CHECKSUM].copy_from_slice(&[0, 0]);
        reference_checksum(&request)
    };
    request[ICMP_CHECKSUM].copy_from_slice(&checksum.to_be_bytes());
    un.from_guest(
        &guest_ip_frame([8, 8, 8, 8], IP_PROTO_ICMP, &request),
        NOW_MS,
    );
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
    let mut un = user_net();
    un.from_guest(
        &guest_ip_frame(
            [8, 8, 8, 8],
            IP_PROTO_ICMP,
            &echo_request_payload(1, 2, &[0; 56]),
        ),
        NOW_MS,
    );
    let reply = reassemble(&drain(&mut un), GUEST_MAC);
    assert_eq!(reply.fragments, 1);
    assert_eq!(reply.payload.len(), 64);
}

#[test]
fn echo_replies_get_distinct_increasing_ip_ids() {
    let mut un = user_net();
    let mut ids = Vec::new();
    for sequence in 0..4 {
        un.from_guest(
            &guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, sequence, b"x"),
            ),
            NOW_MS,
        );
        ids.push(reassemble(&drain(&mut un), GUEST_MAC).id);
    }
    assert_eq!(ids, vec![ids[0], ids[0] + 1, ids[0] + 2, ids[0] + 3]);
}

#[test]
fn ip_id_wraps_instead_of_overflowing() {
    let mut un = user_net();
    un.ip_id = u16::MAX;
    un.from_guest(
        &guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &echo_request_payload(1, 1, b"x")),
        NOW_MS,
    );
    assert_eq!(reassemble(&drain(&mut un), GUEST_MAC).id, 0);
}

#[test]
fn echo_of_odd_length_data_has_a_valid_checksum() {
    for len in [0usize, 1, 3, 55, 57] {
        let mut un = user_net();
        un.from_guest(
            &guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, 1, &vec![0xc3; len]),
            ),
            NOW_MS,
        );
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
        let mut un = user_net();
        un.from_guest(&guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload), NOW_MS);
        assert!(drain(&mut un).is_empty(), "type {kind}");
    }
}

#[test]
fn icmp_shorter_than_an_echo_header_is_ignored() {
    let payload = echo_request_payload(1, 1, &[]);
    for len in 0..ICMP_ECHO_HEADER_LEN {
        let mut un = user_net();
        un.from_guest(
            &guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload[..len]),
            NOW_MS,
        );
        assert!(drain(&mut un).is_empty(), "len {len}");
    }
    let mut un = user_net();
    un.from_guest(&guest_ip_frame(GATEWAY_IP, IP_PROTO_ICMP, &payload), NOW_MS);
    assert_eq!(drain(&mut un).len(), 1);
}

#[test]
fn echo_larger_than_the_mtu_is_answered_with_fragments() {
    let data = splitmix_bytes(99, 3000);
    let request = echo_request_payload(7, 8, &data);
    let mut un = user_net();
    un.from_guest(
        &guest_ip_frame([8, 8, 8, 8], IP_PROTO_ICMP, &request),
        NOW_MS,
    );
    let reply = reassemble(&drain(&mut un), GUEST_MAC);
    assert_eq!(reply.fragments, 3);
    assert_eq!(reply.payload[0], ICMP_ECHO_REPLY);
    assert_eq!(&reply.payload[8..], &data[..]);
    assert_eq!(reference_checksum(&reply.payload), 0);
}

#[test]
fn segment_policy_takes_only_off_subnet_and_dns_frames() {
    let gateway = segment_gateway();
    assert!(gateway.takes_frame(&tcp_syn_frame(GATEWAY_MAC, [93, 184, 216, 34], 80)));
    assert!(gateway.takes_frame(&udp_frame_to(GATEWAY_MAC, [10, 0, 2, 2], 53)));
    assert!(!gateway.takes_frame(&udp_frame_to(GATEWAY_MAC, [10, 0, 2, 2], 67)));
    assert!(!gateway.takes_frame(&tcp_syn_frame(OTHER_GUEST_MAC, [10, 0, 2, 16], 22)));
    assert!(!gateway.takes_frame(&arp_request_frame(GUEST_MAC, [10, 0, 2, 15], [10, 0, 2, 2])));
}

#[test]
fn segment_policy_answers_no_arp_and_no_dhcp() {
    let mut gateway = segment_gateway();
    gateway.from_guest(
        &arp_request_frame(GUEST_MAC, [10, 0, 2, 15], [10, 0, 2, 2]),
        0,
    );
    gateway.from_guest(&dhcp_discover_frame(), 0);
    assert!(gateway.pop_for_guest().is_none());
}
