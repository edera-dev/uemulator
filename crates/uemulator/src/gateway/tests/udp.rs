use super::*;
use crate::gateway::udp::{UDP_IDLE_TIMEOUT_MS, UDP_PENDING_LIMIT};

const SERVER: [u8; 4] = [93, 184, 216, 34];
const SERVER_PORT: u16 = 9999;
const ONE_SECOND_MS: u64 = 1000;

fn key(guest_port: u16, remote_ip: [u8; 4], remote_port: u16) -> FlowKey {
    FlowKey {
        guest_port,
        remote_ip,
        remote_port,
    }
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

struct IcmpView {
    kind: u8,
    code: u8,
    quoted_destination_port: u16,
}

/// The one ICMP error in `frame`, after checking its checksum and that it quotes a whole IPv4
/// header and 8 more bytes.
fn icmp_view(frame: &[u8]) -> IcmpView {
    let ip = parse_ipv4(frame).unwrap();
    assert_eq!(ip.proto, IP_PROTO_ICMP);
    assert_eq!(reference_checksum(ip.payload), 0, "ICMP checksum");
    let quoted = &ip.payload[ICMP_ERROR_HEADER_LEN..];
    let header_len = (quoted[0] & 0xf) as usize * 4;
    assert_eq!(quoted.len(), header_len + UDP_HEADER_LEN);
    IcmpView {
        kind: ip.payload[0],
        code: ip.payload[1],
        quoted_destination_port: be_u16(&quoted[header_len + UDP_DST_PORT.start..]),
    }
}

fn send_at(gateway: &mut Gateway, frame: &[u8], now_ms: u64) {
    gateway.from_guest(frame, now_ms);
}

/// A flow the driver has opened.
fn open_flow(
    gateway: &mut Gateway,
    remote: &mut FakeRemote,
    guest_port: u16,
    now_ms: u64,
) -> StreamId {
    send_at(
        gateway,
        &udp_frame(guest_port, SERVER, SERVER_PORT, b"x"),
        now_ms,
    );
    let stream = remote.single_open(gateway);
    gateway.opened(stream);
    remote.requests(gateway);
    stream
}

#[test]
fn a_new_gateway_has_no_flows_and_the_default_limit() {
    let gateway = user_net();
    assert_eq!(gateway.udp.limit, MAX_UDP_FLOWS);
    assert!(gateway.udp.flows.is_empty());
}

#[test]
fn flow_keys_compare_field_by_field() {
    let base = key(1000, [1, 2, 3, 4], 80);
    assert_eq!(base, key(1000, [1, 2, 3, 4], 80));
    assert_ne!(base, key(1001, [1, 2, 3, 4], 80));
    assert_ne!(base, key(1000, [1, 2, 3, 5], 80));
    assert_ne!(base, key(1000, [1, 2, 3, 4], 81));
    assert_ne!(
        key(80, [1, 2, 3, 4], 1000),
        key(1000, [1, 2, 3, 4], 80),
        "ports are not interchangeable"
    );
}

#[test]
fn flow_keys_order_by_guest_port_then_address_then_remote_port() {
    assert!(key(1, [9, 9, 9, 9], 9) < key(2, [0, 0, 0, 0], 0));
    assert!(key(1, [1, 0, 0, 0], 9) < key(1, [2, 0, 0, 0], 0));
    assert!(key(1, [1, 0, 0, 0], 1) < key(1, [1, 0, 0, 0], 2));
}

#[test]
fn datagram_reaches_the_remote_and_the_reply_comes_back_with_the_original_addresses() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"ping"),
        0,
    );
    let (stream, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(kind, StreamKind::Udp);
    assert_eq!(
        target,
        Endpoint {
            address: SERVER,
            port: SERVER_PORT
        }
    );
    gateway.opened(stream);
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"ping".to_vec()]
    );
    gateway.remote_data(stream, b"pong");
    let (whole, sport, dport, data) = udp_view(&drain(&mut gateway));
    assert_eq!((whole.src, whole.dst), (SERVER, GUEST_IP));
    assert_eq!((sport, dport), (SERVER_PORT, 4000));
    assert_eq!(data, b"pong");
}

#[test]
fn datagrams_of_one_flow_share_a_stream() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"one"),
        0,
    );
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"two"),
        0,
    );
    let requests = remote.requests(&mut gateway);
    assert!(
        requests
            .iter()
            .all(|request| !matches!(request, GatewayRequest::Open { .. })),
        "{requests:?}"
    );
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"one".to_vec(), b"two".to_vec()]
    );
    assert_eq!(gateway.udp.flows.len(), 1);
}

#[test]
fn different_guest_ports_get_different_streams() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, SERVER, SERVER_PORT, b"a"), 0);
    let first = remote.single_open(&mut gateway);
    send_at(&mut gateway, &udp_frame(4001, SERVER, SERVER_PORT, b"b"), 0);
    let second = remote.single_open(&mut gateway);
    assert_ne!(first, second);
    assert_eq!(gateway.udp.flows.len(), 2);
}

#[test]
fn different_destination_ports_are_different_flows() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, SERVER, 1111, b"a"), 0);
    let first = remote.single_open_detail(&mut gateway);
    send_at(&mut gateway, &udp_frame(4000, SERVER, 2222, b"b"), 0);
    let second = remote.single_open_detail(&mut gateway);
    assert_ne!(first.0, second.0);
    assert_eq!((first.2.port, second.2.port), (1111, 2222));
    assert_eq!(gateway.udp.flows.len(), 2);
}

#[test]
fn replies_are_routed_to_the_right_guest_port() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut streams = Vec::new();
    for guest_port in [5000u16, 5001] {
        send_at(
            &mut gateway,
            &udp_frame(guest_port, SERVER, SERVER_PORT, b"hello"),
            0,
        );
        streams.push(remote.single_open(&mut gateway));
    }
    gateway.remote_data(streams[1], b"for-5001");
    gateway.remote_data(streams[0], b"for-5000");
    let mut seen: Vec<_> = drain(&mut gateway)
        .into_iter()
        .map(|frame| {
            let (_, _, dport, data) = udp_view(&[frame]);
            (dport, data)
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec![(5000, b"for-5000".to_vec()), (5001, b"for-5001".to_vec())]
    );
}

#[test]
fn the_gateway_address_reaches_host_loopback_and_replies_come_from_it() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &udp_frame(4100, GATEWAY_IP, 8080, b"to-host"),
        0,
    );
    let (stream, _, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(
        target,
        Endpoint {
            address: LOOPBACK_IP,
            port: 8080
        }
    );
    gateway.remote_data(stream, b"from-host");
    let (whole, sport, dport, data) = udp_view(&drain(&mut gateway));
    assert_eq!(whole.src, GATEWAY_IP);
    assert_eq!((sport, dport), (8080, 4100));
    assert_eq!(data, b"from-host");
}

#[test]
fn dns_queries_go_to_the_configured_server_and_are_answered_from_the_dns_address() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &udp_frame(33333, DNS_IP, DNS_PORT, b"\x12\x34 query"),
        0,
    );
    let (stream, _, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(target, RESOLVER);
    gateway.opened(stream);
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"\x12\x34 query".to_vec()]
    );
    gateway.remote_data(stream, b"\x12\x34 answer");
    let (whole, sport, dport, data) = udp_view(&drain(&mut gateway));
    assert_eq!(whole.src, DNS_IP);
    assert_eq!((sport, dport), (DNS_PORT, 33333));
    assert_eq!(data, b"\x12\x34 answer");
}

#[test]
fn dns_flows_are_keyed_by_the_dns_address_not_the_configured_server() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(33333, DNS_IP, DNS_PORT, b"q"), 0);
    remote.single_open(&mut gateway);
    assert!(
        gateway
            .udp
            .flows
            .contains_key(&key(33333, DNS_IP, DNS_PORT))
    );
    // The same guest port talking to the server's own address is a second flow.
    send_at(
        &mut gateway,
        &udp_frame(33333, RESOLVER.address, RESOLVER.port, b"q"),
        0,
    );
    remote.single_open(&mut gateway);
    assert_eq!(gateway.udp.flows.len(), 2);
}

#[test]
fn the_address_mapping_leaves_other_destinations_alone() {
    let gateway = user_net();
    let loopback = Endpoint {
        address: LOOPBACK_IP,
        port: 8080,
    };
    let as_is = |address: [u8; 4], port: u16| {
        gateway.destination(address, port) == Endpoint { address, port }
    };
    assert_eq!(gateway.destination(GATEWAY_IP, 8080), loopback);
    assert_eq!(gateway.destination(DNS_IP, DNS_PORT), RESOLVER);
    assert!(as_is([93, 184, 216, 34], 443));
    assert!(as_is(DNS_IP, 80), "only DNS goes to the nameserver");
    assert!(as_is(GUEST_IP, 1));
}

#[test]
fn a_segment_gateway_sends_dns_at_the_gateway_address_to_its_resolver() {
    let (mut gateway, mut remote) = segment_with_remote();
    send_at(
        &mut gateway,
        &ip_frame_to(
            GATEWAY_MAC,
            GATEWAY_IP,
            IP_PROTO_UDP,
            &udp_segment(GUEST_IP, GATEWAY_IP, 4000, DNS_PORT, b"q"),
        ),
        0,
    );
    let (stream, _, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(target, RESOLVER);
    gateway.remote_data(stream, b"a");
    let (whole, ..) = udp_view(&drain(&mut gateway));
    assert_eq!(whole.src, GATEWAY_IP);
}

#[test]
fn a_segment_gateway_forwards_off_subnet_udp_as_is() {
    let (mut gateway, mut remote) = segment_with_remote();
    send_at(&mut gateway, &udp_frame(4000, SERVER, SERVER_PORT, b"q"), 0);
    let (_, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(kind, StreamKind::Udp);
    assert_eq!((target.address, target.port), (SERVER, SERVER_PORT));
}

#[test]
fn a_segment_gateway_ignores_udp_between_machines() {
    let (mut gateway, mut remote) = segment_with_remote();
    send_at(
        &mut gateway,
        &udp_frame_to([2, 0, 0, 0, 0, 1], [10, 0, 2, 20], 5000),
        0,
    );
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn udp_length_field_bounds_the_forwarded_data() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut segment = udp_segment(GUEST_IP, SERVER, 4000, SERVER_PORT, b"abcdef");
    segment[UDP_LENGTH].copy_from_slice(&((UDP_HEADER_LEN + 2) as u16).to_be_bytes());
    send_at(
        &mut gateway,
        &guest_ip_frame(SERVER, IP_PROTO_UDP, &segment),
        0,
    );
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"ab".to_vec()],
        "bytes beyond the UDP length are ignored"
    );
}

#[test]
fn udp_length_beyond_the_packet_is_clamped_to_what_arrived() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut segment = udp_segment(GUEST_IP, SERVER, 4000, SERVER_PORT, b"abcdef");
    segment[UDP_LENGTH].copy_from_slice(&u16::MAX.to_be_bytes());
    send_at(
        &mut gateway,
        &guest_ip_frame(SERVER, IP_PROTO_UDP, &segment),
        0,
    );
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"abcdef".to_vec()]
    );
}

#[test]
fn udp_length_below_the_header_size_sends_an_empty_datagram() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut segment = udp_segment(GUEST_IP, SERVER, 4000, SERVER_PORT, b"abcdef");
    segment[UDP_LENGTH].copy_from_slice(&3u16.to_be_bytes());
    send_at(
        &mut gateway,
        &guest_ip_frame(SERVER, IP_PROTO_UDP, &segment),
        0,
    );
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    assert_eq!(remote.drain_datagrams(&mut gateway, stream), [Vec::new()]);
}

#[test]
fn empty_datagrams_are_forwarded() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, SERVER, SERVER_PORT, b""), 0);
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    assert_eq!(remote.drain_datagrams(&mut gateway, stream), [Vec::new()]);
}

#[test]
fn udp_shorter_than_its_header_is_ignored() {
    let (mut gateway, mut remote) = user_net_with_remote();
    for len in 0..UDP_HEADER_LEN {
        send_at(
            &mut gateway,
            &guest_ip_frame(SERVER, IP_PROTO_UDP, &vec![0u8; len]),
            0,
        );
    }
    assert!(gateway.udp.flows.is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
    assert!(drain(&mut gateway).is_empty());
}

#[test]
fn datagrams_to_the_dhcp_server_port_are_never_nat_ed() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &udp_frame(68, SERVER, DHCP_SERVER_PORT, b"junk"),
        0,
    );
    send_at(
        &mut gateway,
        &udp_frame(68, GATEWAY_IP, DHCP_SERVER_PORT, b"junk"),
        0,
    );
    assert!(gateway.udp.flows.is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn unsupported_ip_protocols_open_no_flow() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &guest_ip_frame(GATEWAY_IP, IP_PROTO_GRE, &[0; 16]),
        0,
    );
    assert!(gateway.udp.flows.is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn echo_requests_open_no_flow() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(
        &mut gateway,
        &guest_ip_frame(
            SERVER,
            IP_PROTO_ICMP,
            &echo_request_payload(1, 1, b"abcdefgh"),
        ),
        0,
    );
    assert!(gateway.udp.flows.is_empty());
    assert!(
        remote.requests(&mut gateway).is_empty(),
        "nothing leaves the host"
    );
}

#[test]
fn stream_ids_are_nonzero_and_increasing() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut previous = 0;
    for guest_port in 1..=5u16 {
        send_at(
            &mut gateway,
            &udp_frame(guest_port, SERVER, SERVER_PORT, b"x"),
            0,
        );
        let stream = remote.single_open(&mut gateway);
        assert!(stream > previous, "{stream} after {previous}");
        previous = stream;
    }
}

#[test]
fn stream_ids_skip_zero_and_live_ids_when_the_counter_wraps() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let first = open_flow(&mut gateway, &mut remote, 1, 0);
    gateway.next_stream = StreamId::MAX;
    send_at(&mut gateway, &udp_frame(2, SERVER, SERVER_PORT, b"x"), 0);
    assert_eq!(remote.single_open(&mut gateway), StreamId::MAX);
    // The counter wrapped past 0 onto `first`, which is live.
    gateway.next_stream = first;
    send_at(&mut gateway, &udp_frame(3, SERVER, SERVER_PORT, b"x"), 0);
    let third = remote.single_open(&mut gateway);
    assert_ne!(third, 0);
    assert_ne!(third, first);
}

#[test]
fn a_refused_udp_flow_answers_with_port_unreachable() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.from_guest(&udp_frame(4000, [93, 184, 216, 34], 9999, b"hi"), 0);
    let stream = remote.single_open(&mut gateway);
    gateway.failed(stream);
    let icmp = icmp_view(&drain(&mut gateway)[0]);
    assert_eq!((icmp.kind, icmp.code), (3, 3));
    assert_eq!(icmp.quoted_destination_port, 9999);
}

#[test]
fn port_unreachable_comes_from_the_address_the_guest_sent_to() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, GATEWAY_IP, 7, b"hi"), 0);
    let stream = remote.single_open(&mut gateway);
    gateway.failed(stream);
    let frames = drain(&mut gateway);
    assert_eq!(frames.len(), 1);
    let ip = parse_ipv4(&frames[0]).unwrap();
    assert_eq!((ip.src, ip.dst), (GATEWAY_IP, GUEST_IP));
}

#[test]
fn a_failed_flow_is_gone_and_the_next_datagram_opens_a_new_one() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, SERVER, SERVER_PORT, b"a"), 0);
    let first = remote.single_open(&mut gateway);
    gateway.failed(first);
    assert!(gateway.udp.flows.is_empty());
    send_at(&mut gateway, &udp_frame(4000, SERVER, SERVER_PORT, b"b"), 0);
    let second = remote.single_open(&mut gateway);
    assert_ne!(first, second);
}

#[test]
fn datagrams_before_open_are_released_on_open_in_order() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.from_guest(&udp_frame(4000, [8, 8, 8, 8], 53, b"one"), 0);
    gateway.from_guest(&udp_frame(4000, [8, 8, 8, 8], 53, b"two"), 0);
    let stream = remote.single_open(&mut gateway);
    assert!(remote.datagrams(stream).is_empty());
    gateway.opened(stream);
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"one".to_vec(), b"two".to_vec()]
    );
}

#[test]
fn datagrams_waiting_for_the_open_are_bounded_and_the_newest_dropped() {
    let (mut gateway, mut remote) = user_net_with_remote();
    for index in 0..UDP_PENDING_LIMIT + 10 {
        let data = (index as u32).to_be_bytes();
        send_at(
            &mut gateway,
            &udp_frame(4000, SERVER, SERVER_PORT, &data),
            0,
        );
    }
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    let datagrams = remote.drain_datagrams(&mut gateway, stream);
    assert_eq!(datagrams.len(), UDP_PENDING_LIMIT);
    assert_eq!(
        datagrams.last().unwrap(),
        &((UDP_PENDING_LIMIT - 1) as u32).to_be_bytes().to_vec(),
        "the oldest are kept"
    );
}

#[test]
fn datagrams_after_open_are_forwarded_at_once() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    remote.drain_datagrams(&mut gateway, stream);
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"later"),
        0,
    );
    assert_eq!(
        remote.drain_datagrams(&mut gateway, stream),
        [b"later".to_vec()]
    );
}

#[test]
fn events_for_unknown_streams_are_ignored() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    for unknown in [0, stream + 1, StreamId::MAX] {
        gateway.opened(unknown);
        gateway.failed(unknown);
        gateway.remote_data(unknown, b"x");
        gateway.remote_closed(unknown);
        gateway.reset(unknown);
    }
    assert!(drain(&mut gateway).is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
    assert_eq!(gateway.udp.flows.len(), 1);
}

#[test]
fn a_closed_or_reset_flow_is_dropped_silently() {
    for reset in [false, true] {
        let (mut gateway, mut remote) = user_net_with_remote();
        let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
        if reset {
            gateway.reset(stream);
        } else {
            gateway.remote_closed(stream);
        }
        assert!(gateway.udp.flows.is_empty());
        assert!(drain(&mut gateway).is_empty(), "no ICMP");
        assert!(remote.requests(&mut gateway).is_empty(), "no Close");
        // Late data for the dead stream goes nowhere.
        gateway.remote_data(stream, b"late");
        assert!(drain(&mut gateway).is_empty());
    }
}

#[test]
fn idle_flows_expire_exactly_at_the_timeout() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    gateway.poll(0);
    assert_eq!(gateway.udp.flows.len(), 1);
    gateway.poll(UDP_IDLE_TIMEOUT_MS - 1);
    assert!(
        remote.requests(&mut gateway).is_empty(),
        "one millisecond early"
    );
    gateway.poll(UDP_IDLE_TIMEOUT_MS);
    assert_eq!(
        remote.requests(&mut gateway),
        [GatewayRequest::Close { stream }]
    );
    assert!(gateway.udp.flows.is_empty(), "at the timeout");
    assert!(remote.is_closed(stream));
}

#[test]
fn sending_refreshes_the_idle_timer() {
    let (mut gateway, mut remote) = user_net_with_remote();
    open_flow(&mut gateway, &mut remote, 4000, 0);
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"x"),
        40 * ONE_SECOND_MS,
    );
    gateway.poll(70 * ONE_SECOND_MS);
    assert_eq!(gateway.udp.flows.len(), 1, "30 s since the last datagram");
    gateway.poll(101 * ONE_SECOND_MS);
    assert!(gateway.udp.flows.is_empty());
}

#[test]
fn receiving_from_the_remote_refreshes_the_idle_timer() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    gateway.poll(50 * ONE_SECOND_MS);
    gateway.remote_data(stream, b"y");
    gateway.poll(100 * ONE_SECOND_MS);
    assert_eq!(gateway.udp.flows.len(), 1, "50 s after the reply");
    gateway.poll(111 * ONE_SECOND_MS);
    assert!(gateway.udp.flows.is_empty());
}

#[test]
fn expired_flows_are_recreated_by_the_next_datagram() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let first = open_flow(&mut gateway, &mut remote, 4000, 0);
    gateway.poll(UDP_IDLE_TIMEOUT_MS);
    remote.requests(&mut gateway);
    send_at(
        &mut gateway,
        &udp_frame(4000, SERVER, SERVER_PORT, b"2"),
        UDP_IDLE_TIMEOUT_MS,
    );
    let second = remote.single_open(&mut gateway);
    assert_ne!(first, second, "a fresh stream");
}

#[test]
fn the_flow_table_never_exceeds_its_limit() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 4;
    for (i, guest_port) in (6000u16..6020).enumerate() {
        send_at(
            &mut gateway,
            &udp_frame(guest_port, SERVER, SERVER_PORT, b"x"),
            i as u64,
        );
        remote.requests(&mut gateway);
        assert!(gateway.udp.flows.len() <= 4, "after {} flows", i + 1);
    }
    assert_eq!(gateway.udp.flows.len(), 4);
    assert_eq!(
        remote.live_streams().len(),
        4,
        "every eviction closed its stream"
    );
}

#[test]
fn the_least_recently_used_flow_is_evicted_first_and_closed() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 3;
    let at = |secs: u64| secs * ONE_SECOND_MS;
    let mut streams = Vec::new();
    for (i, guest_port) in [1u16, 2, 3].into_iter().enumerate() {
        streams.push(open_flow(
            &mut gateway,
            &mut remote,
            guest_port,
            at(i as u64),
        ));
    }
    // Flow 1 was the oldest, but sending on it makes flow 2 the least recently used.
    send_at(
        &mut gateway,
        &udp_frame(1, SERVER, SERVER_PORT, b"x"),
        at(10),
    );
    send_at(
        &mut gateway,
        &udp_frame(4, SERVER, SERVER_PORT, b"x"),
        at(11),
    );
    let requests = remote.requests(&mut gateway);
    let closes: Vec<_> = requests
        .iter()
        .filter(|request| matches!(request, GatewayRequest::Close { .. }))
        .collect();
    assert_eq!(closes, [&GatewayRequest::Close { stream: streams[1] }]);
    let mut remaining: Vec<u16> = gateway.udp.flows.keys().map(|k| k.guest_port).collect();
    remaining.sort();
    assert_eq!(remaining, vec![1, 3, 4]);
    send_at(
        &mut gateway,
        &udp_frame(5, SERVER, SERVER_PORT, b"x"),
        at(12),
    );
    let mut remaining: Vec<u16> = gateway.udp.flows.keys().map(|k| k.guest_port).collect();
    remaining.sort();
    assert_eq!(remaining, vec![1, 4, 5], "flow 3 was idle longest");
}

#[test]
fn traffic_on_an_existing_flow_never_evicts_anything() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 2;
    open_flow(&mut gateway, &mut remote, 1, 0);
    open_flow(&mut gateway, &mut remote, 2, 0);
    for step in 1..10 {
        send_at(
            &mut gateway,
            &udp_frame(1, SERVER, SERVER_PORT, b"x"),
            step * ONE_SECOND_MS,
        );
    }
    assert!(gateway.udp.flows.contains_key(&key(1, SERVER, SERVER_PORT)));
    assert!(gateway.udp.flows.contains_key(&key(2, SERVER, SERVER_PORT)));
    assert!(
        remote
            .requests(&mut gateway)
            .iter()
            .all(|request| matches!(request, GatewayRequest::Datagram { .. }))
    );
}

#[test]
fn an_evicted_flow_can_start_again() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 1;
    for (guest_port, at) in [(1u16, 0), (2, 1), (1, 2)] {
        send_at(
            &mut gateway,
            &udp_frame(guest_port, SERVER, SERVER_PORT, b"x"),
            at * ONE_SECOND_MS,
        );
        remote.single_open(&mut gateway);
    }
    assert_eq!(gateway.udp.flows.len(), 1);
    assert!(gateway.udp.flows.contains_key(&key(1, SERVER, SERVER_PORT)));
}

#[test]
fn a_limit_of_zero_still_lets_one_flow_work() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 0;
    let stream = open_flow(&mut gateway, &mut remote, 1, 0);
    assert_eq!(remote.live_streams(), [stream]);
}

#[test]
fn many_distinct_udp_flows_stay_within_the_cap() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.udp.limit = 8;
    for port in 0..200u16 {
        send_at(
            &mut gateway,
            &udp_frame(2000 + port, DNS_IP, DNS_PORT, b"x"),
            0,
        );
    }
    remote.requests(&mut gateway);
    assert!(gateway.udp.flows.len() <= 8);
    assert!(remote.live_streams().len() <= 8);
}

#[test]
fn the_default_limit_is_128_flows() {
    let (mut gateway, mut remote) = user_net_with_remote();
    for port in 0..(MAX_UDP_FLOWS as u16 + 10) {
        send_at(
            &mut gateway,
            &udp_frame(2000 + port, SERVER, SERVER_PORT, b"x"),
            0,
        );
    }
    remote.requests(&mut gateway);
    assert_eq!(gateway.udp.flows.len(), MAX_UDP_FLOWS);
    assert_eq!(remote.live_streams().len(), MAX_UDP_FLOWS);
}

#[test]
fn remote_datagram_that_fits_the_mtu_is_one_packet_with_df() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    let largest = vec![0x42u8; ETHERNET_MTU - IPV4_HEADER_LEN - UDP_HEADER_LEN];
    gateway.remote_data(stream, &largest);
    let frames = drain(&mut gateway);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), ETH_HEADER_LEN + ETHERNET_MTU);
    let (whole, _, _, data) = udp_view(&frames);
    assert_eq!(whole.fragments, 1);
    assert_eq!(data, largest);
}

#[test]
fn remote_datagram_one_byte_over_the_mtu_is_fragmented() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    let data: Vec<u8> = (0..ETHERNET_MTU - IPV4_HEADER_LEN - UDP_HEADER_LEN + 1)
        .map(|i| i as u8)
        .collect();
    gateway.remote_data(stream, &data);
    let frames = drain(&mut gateway);
    assert_eq!(frames.len(), 2);
    let (whole, sport, dport, received) = udp_view(&frames);
    assert_eq!((sport, dport), (SERVER_PORT, 4000));
    assert_eq!(received, data);
    assert_eq!(whole.fragments, 2);
}

#[test]
fn large_remote_datagram_is_delivered_as_correctly_numbered_fragments() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    let data = splitmix_bytes(5, 4000);
    gateway.remote_data(stream, &data);
    let frames = drain(&mut gateway);
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
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    let mut ids = Vec::new();
    for round in 0..2u8 {
        gateway.remote_data(stream, &vec![round; 3000]);
        ids.push(reassemble(&drain(&mut gateway), GUEST_MAC).id);
    }
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn a_datagram_too_large_for_ip_is_dropped() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let stream = open_flow(&mut gateway, &mut remote, 4000, 0);
    gateway.remote_data(stream, &vec![0; usize::from(u16::MAX)]);
    assert!(drain(&mut gateway).is_empty());
    gateway.remote_data(stream, b"still works");
    assert_eq!(drain(&mut gateway).len(), 1);
}

#[test]
fn the_fake_remote_reports_what_each_stream_is_for() {
    let (mut gateway, mut remote) = user_net_with_remote();
    send_at(&mut gateway, &udp_frame(4000, GATEWAY_IP, 5353, b"x"), 0);
    let stream = remote.single_open(&mut gateway);
    assert_eq!(remote.kind(stream), StreamKind::Udp);
    assert_eq!(remote.target(stream).address, LOOPBACK_IP);
}
