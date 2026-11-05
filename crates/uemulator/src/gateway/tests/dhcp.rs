use super::*;

const DHCP_DECLINE: u8 = 4;
const DHCP_RELEASE: u8 = 7;
const DHCP_INFORM: u8 = 8;
const UNICAST_OFFER_SERVER: [u8; 4] = [10, 0, 2, 99];
const LEASE_SECONDS_BIG_ENDIAN: [u8; 4] = [0x00, 0x01, 0x51, 0x80];

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

fn send(un: &mut Gateway, message: &[u8]) {
    un.from_guest(&dhcp_frame(message), NOW_MS);
}

/// The BOOTP payload of the one reply the stack queued, after checking every layer around it.
fn reply_to(un: &mut Gateway, message: &[u8]) -> Option<Vec<u8>> {
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
    let reply = reply_to(&mut un, &selecting_request(GUEST_IP, GATEWAY_IP)).unwrap();
    assert_reply_header(&reply);
    assert_eq!(reply[BOOTP_YIADDR], GUEST_IP);
    assert_eq!(reply[BOOTP_SIADDR], GATEWAY_IP);
    assert_eq!(options_in(&reply), full_option_set(DHCP_ACK));
}

#[test]
fn selecting_request_for_a_different_address_gets_a_nak() {
    let mut un = user_net();
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
        let mut un = user_net();
        let reply = reply_to(&mut un, &selecting_request(wanted, GATEWAY_IP)).unwrap();
        assert_eq!(message_type_of(&reply), DHCP_NAK, "{wanted:?}");
    }
}

#[test]
fn request_naming_another_server_is_not_answered() {
    // RFC 2131 section 3.1 step 4: the client picked a different server's offer.
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
    let message = client_message(&options_of(DHCP_REQUEST, &[]), ZERO_IP);
    assert_eq!(
        message_type_of(&reply_to(&mut un, &message).unwrap()),
        DHCP_ACK
    );
}

#[test]
fn server_id_is_checked_before_the_address() {
    let mut un = user_net();
    let message = selecting_request([10, 0, 2, 16], [1, 2, 3, 4]);
    assert!(reply_to(&mut un, &message).is_none());
}

#[test]
fn release_decline_and_inform_are_ignored() {
    for kind in [DHCP_RELEASE, DHCP_DECLINE, DHCP_INFORM] {
        let mut un = user_net();
        let message = client_message(&options_of(kind, &[]), GUEST_IP);
        assert!(reply_to(&mut un, &message).is_none(), "type {kind}");
    }
}

#[test]
fn server_message_types_sent_by_a_client_are_ignored() {
    for kind in [DHCP_OFFER, DHCP_ACK, DHCP_NAK, 0, 9, 99, 255] {
        let mut un = user_net();
        assert!(
            reply_to(&mut un, &client_message(&options_of(kind, &[]), ZERO_IP)).is_none(),
            "type {kind}"
        );
    }
}

#[test]
fn message_without_a_type_option_is_ignored() {
    let mut un = user_net();
    let message = client_message(&[DHCP_OPT_END], ZERO_IP);
    assert!(reply_to(&mut un, &message).is_none());
}

#[test]
fn bootp_replies_and_bad_cookies_are_ignored() {
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
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
    let mut un = user_net();
    let mut options = vec![DHCP_OPT_END];
    options.extend_from_slice(&message_type_option(DHCP_DISCOVER));
    assert!(reply_to(&mut un, &client_message(&options, ZERO_IP)).is_none());
}

#[test]
fn a_message_without_the_end_marker_is_still_read() {
    let mut un = user_net();
    let options = message_type_option(DHCP_DISCOVER);
    assert!(reply_to(&mut un, &client_message(&options, ZERO_IP)).is_some());
}

#[test]
fn every_truncation_of_a_discover_is_handled() {
    let full = client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP);
    let first_complete_type_option = DHCP_OPTIONS_OFFSET + 3;
    for len in 0..=full.len() {
        let mut un = user_net();
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
    let mut un = user_net();
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
        let mut un = user_net();
        send(&mut un, &message);
        for frame in drain(&mut un) {
            assert_eq!(
                frame.len(),
                ETH_HEADER_LEN + IPV4_HEADER_LEN + UDP_HEADER_LEN + BOOTP_MIN_LEN
            );
        }
    }
}
