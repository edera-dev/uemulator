use super::*;
use crate::gateway::dns_tcp::{DNS_TCP_IDLE_TIMEOUT_MS, DNS_TCP_PENDING_LIMIT};

const QUERY_PORT: u16 = 5353;
const DNS_HEADER_LEN: usize = 12;
const DNS_ANSWER_FLAG: u8 = 0x80;

fn dns_query(id: u16) -> Vec<u8> {
    let mut message = id.to_be_bytes().to_vec();
    message.extend_from_slice(&[0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0]);
    message.extend_from_slice(b"\x07example\x03com\x00\x00\x01\x00\x01");
    message
}

fn dns_answer(id: u16) -> Vec<u8> {
    let mut message = dns_query(id);
    message[2] = DNS_ANSWER_FLAG;
    assert!(message.len() > DNS_HEADER_LEN);
    message
}

fn length_prefixed(message: &[u8]) -> Vec<u8> {
    let mut framed = (message.len() as u16).to_be_bytes().to_vec();
    framed.extend_from_slice(message);
    framed
}

fn udp_frame_to_gateway(sport: u16, dport: u16, data: &[u8]) -> Vec<u8> {
    ip_frame_to(
        GATEWAY_MAC,
        GATEWAY_IP,
        IP_PROTO_UDP,
        &udp_segment(GUEST_IP, GATEWAY_IP, sport, dport, data),
    )
}

struct Reply {
    src: [u8; 4],
    sport: u16,
    dport: u16,
    data: Vec<u8>,
}

fn udp_view(frames: &[Vec<u8>]) -> Reply {
    let whole = reassemble(frames, GUEST_MAC);
    assert_eq!(whole.proto, IP_PROTO_UDP);
    Reply {
        src: whole.src,
        sport: be_u16(&whole.payload[UDP_SRC_PORT]),
        dport: be_u16(&whole.payload[UDP_DST_PORT]),
        data: whole.payload[UDP_HEADER_LEN..].to_vec(),
    }
}

fn is_icmp_unreachable(frame: &[u8]) -> bool {
    let ip = &frame[ETH_HEADER_LEN..];
    ip[9] == IP_PROTO_ICMP && ip[IPV4_HEADER_LEN] == ICMP_DEST_UNREACHABLE
}

fn ask(gateway: &mut Gateway, port: u16, id: u16, now_ms: u64) {
    gateway.from_guest(
        &udp_frame_to_gateway(port, DNS_PORT, &dns_query(id)),
        now_ms,
    );
}

/// A gateway whose UDP to the resolver was refused and whose TCP stream is open, with the first
/// query asked and answered.
fn fallen_back() -> (Gateway, FakeRemote, StreamId) {
    let (mut gateway, mut remote) = segment_with_remote();
    ask(&mut gateway, QUERY_PORT, 1, 0);
    let udp = remote.single_open(&mut gateway);
    gateway.failed(udp);
    let tcp = remote.single_open(&mut gateway);
    gateway.opened(tcp);
    assert_eq!(
        remote.take_written(&mut gateway, tcp).len(),
        2 + dns_query(1).len()
    );
    gateway.remote_data(tcp, &length_prefixed(&dns_answer(1)));
    drain(&mut gateway);
    (gateway, remote, tcp)
}

#[test]
fn dns_to_the_gateway_goes_to_the_resolver() {
    let (mut gateway, mut remote) = segment_with_remote();
    ask(&mut gateway, QUERY_PORT, 0x1234, 0);
    let (stream, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!((kind, target), (StreamKind::Udp, RESOLVER));
    gateway.opened(stream);
    gateway.remote_data(stream, &dns_answer(0x1234));
    let reply = udp_view(&drain(&mut gateway));
    assert_eq!(
        (reply.src, reply.sport, reply.dport),
        (GATEWAY_IP, 53, QUERY_PORT)
    );
}

#[test]
fn tcp_to_the_gateway_dns_port_goes_to_the_resolver() {
    let (mut gateway, mut remote) = segment_with_remote();
    gateway.from_guest(&tcp_syn_frame(GATEWAY_MAC, GATEWAY_IP, DNS_PORT), 0);
    let (_, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!((kind, target), (StreamKind::Tcp, RESOLVER));
}

#[test]
fn refused_udp_dns_falls_back_to_tcp_with_length_prefix() {
    let (mut gateway, mut remote) = segment_with_remote();
    ask(&mut gateway, QUERY_PORT, 0x1234, 0);
    let udp = remote.single_open(&mut gateway);
    gateway.failed(udp);
    let (tcp, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!((kind, target), (StreamKind::Tcp, RESOLVER));
    gateway.opened(tcp);
    assert_eq!(
        gateway.to_remote(tcp),
        &length_prefixed(&dns_query(0x1234))[..]
    );
    let framed_answer = length_prefixed(&dns_answer(0x1234));
    gateway.remote_data(tcp, &framed_answer[..3]);
    gateway.remote_data(tcp, &framed_answer[3..]);
    gateway.poll(0);
    let frames = drain(&mut gateway);
    assert!(frames.iter().all(|f| !is_icmp_unreachable(f)));
    let reply = udp_view(&frames);
    assert_eq!(reply.data, dns_answer(0x1234));
    assert_eq!(
        (reply.src, reply.sport, reply.dport),
        (GATEWAY_IP, 53, QUERY_PORT)
    );
}

#[test]
fn dns_with_no_uplink_at_all_fails_at_once() {
    let (mut gateway, mut remote) = segment_with_remote();
    ask(&mut gateway, QUERY_PORT, 0x1234, 0);
    let udp = remote.single_open(&mut gateway);
    gateway.failed(udp);
    let tcp = remote.single_open(&mut gateway);
    gateway.failed(tcp);
    assert!(is_icmp_unreachable(&drain(&mut gateway)[0]));
}

#[test]
fn every_query_waiting_on_a_failed_tcp_stream_gets_unreachable() {
    let (mut gateway, mut remote) = segment_with_remote();
    ask(&mut gateway, QUERY_PORT, 1, 0);
    let udp = remote.single_open(&mut gateway);
    gateway.failed(udp);
    let tcp = remote.single_open(&mut gateway);
    ask(&mut gateway, QUERY_PORT + 1, 2, 0);
    gateway.failed(tcp);
    let frames = drain(&mut gateway);
    assert_eq!(frames.len(), 2);
    assert!(frames.iter().all(|f| is_icmp_unreachable(f)));
}

#[test]
fn once_refused_every_later_query_goes_straight_to_tcp() {
    let (mut gateway, mut remote, tcp) = fallen_back();
    ask(&mut gateway, QUERY_PORT + 1, 2, 1);
    assert!(remote.requests(&mut gateway).is_empty(), "no new stream");
    assert_eq!(gateway.to_remote(tcp), &length_prefixed(&dns_query(2))[..]);
}

#[test]
fn answers_are_matched_by_id_and_source_port() {
    let (mut gateway, _remote, tcp) = fallen_back();
    ask(&mut gateway, 6000, 7, 1);
    ask(&mut gateway, 6001, 7, 1);
    let waiting = gateway.to_remote(tcp).len();
    gateway.consumed(tcp, waiting);
    let mut bytes = length_prefixed(&dns_answer(99));
    bytes.extend(length_prefixed(&dns_answer(7)));
    bytes.extend(length_prefixed(&dns_answer(7)));
    gateway.remote_data(tcp, &bytes);
    gateway.poll(1);
    let frames = drain(&mut gateway);
    assert_eq!(frames.len(), 2);
    let ports: Vec<u16> = frames
        .iter()
        .map(|f| udp_view(core::slice::from_ref(f)).dport)
        .collect();
    assert_eq!(ports, [6000, 6001]);
}

#[test]
fn junk_from_the_resolver_is_ignored() {
    let (mut gateway, _remote, tcp) = fallen_back();
    gateway.remote_data(tcp, &[0, 0]);
    gateway.remote_data(tcp, &[0, 1, 5]);
    gateway.remote_data(tcp, &[0xff, 0xff]);
    gateway.poll(0);
    assert!(drain(&mut gateway).is_empty());
}

#[test]
fn a_closed_stream_drops_its_queries_and_is_reopened_on_demand() {
    let (mut gateway, mut remote, tcp) = fallen_back();
    ask(&mut gateway, QUERY_PORT, 2, 1);
    gateway.remote_closed(tcp);
    let requests = remote.requests(&mut gateway);
    assert_eq!(requests, [GatewayRequest::Close { stream: tcp }]);
    assert!(drain(&mut gateway).is_empty(), "dropped, not refused");
    assert_eq!(gateway.to_remote(tcp), &[] as &[u8]);
    ask(&mut gateway, QUERY_PORT, 3, 2);
    let (again, kind, _) = remote.single_open_detail(&mut gateway);
    assert_eq!(kind, StreamKind::Tcp);
    assert_ne!(again, tcp);
    gateway.opened(again);
    assert_eq!(
        gateway.to_remote(again),
        &length_prefixed(&dns_query(3))[..]
    );
}

#[test]
fn a_reset_stream_is_reopened_without_a_close() {
    let (mut gateway, mut remote, tcp) = fallen_back();
    ask(&mut gateway, QUERY_PORT, 2, 1);
    gateway.reset(tcp);
    assert!(remote.requests(&mut gateway).is_empty());
    ask(&mut gateway, QUERY_PORT, 3, 2);
    remote.single_open(&mut gateway);
}

#[test]
fn the_tcp_stream_is_watched_like_a_tcp_flow() {
    let (mut gateway, _remote, tcp) = fallen_back();
    let wants = |gateway: &Gateway| {
        gateway
            .interest()
            .into_iter()
            .find(|(stream, _)| *stream == tcp)
            .map(|(_, interest)| interest)
    };
    assert_eq!(
        wants(&gateway),
        Some(StreamInterest {
            read: true,
            write: false
        })
    );
    ask(&mut gateway, QUERY_PORT, 2, 1);
    assert_eq!(
        wants(&gateway),
        Some(StreamInterest {
            read: true,
            write: true
        })
    );
    assert!(gateway.room(tcp) > 0);
}

#[test]
fn an_idle_tcp_stream_is_closed() {
    let (mut gateway, mut remote, tcp) = fallen_back();
    gateway.poll(DNS_TCP_IDLE_TIMEOUT_MS - 1);
    assert!(remote.requests(&mut gateway).is_empty());
    gateway.poll(DNS_TCP_IDLE_TIMEOUT_MS);
    assert_eq!(
        remote.requests(&mut gateway),
        [GatewayRequest::Close { stream: tcp }]
    );
}

#[test]
fn queries_past_the_pending_limit_are_dropped() {
    let (mut gateway, _remote, tcp) = fallen_back();
    for id in 0..(DNS_TCP_PENDING_LIMIT as u16 + 10) {
        ask(&mut gateway, QUERY_PORT, 100 + id, 1);
    }
    let sent = gateway.to_remote(tcp).len();
    assert_eq!(sent, DNS_TCP_PENDING_LIMIT * (2 + dns_query(0).len()));
}
