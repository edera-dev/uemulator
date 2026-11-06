use super::packet::*;
use crate::gateway::StreamKind;

#[test]
fn connect_has_the_spec_layout() {
    let bytes = encode(&WispPacket::Connect {
        stream: 0x0403_0201,
        kind: StreamKind::Tcp,
        port: 443,
        host: "1.2.3.4".into(),
    });
    assert_eq!(
        bytes,
        [
            &[0x01, 0x01, 0x02, 0x03, 0x04, 0x01, 0xbb, 0x01][..],
            b"1.2.3.4"
        ]
        .concat()
    );
}

#[test]
fn every_packet_round_trips() {
    let packets = [
        WispPacket::Connect {
            stream: 1,
            kind: StreamKind::Tcp,
            port: 80,
            host: "example.com".into(),
        },
        WispPacket::Connect {
            stream: 2,
            kind: StreamKind::Udp,
            port: 53,
            host: "1.1.1.1".into(),
        },
        WispPacket::Connect {
            stream: 3,
            kind: StreamKind::Tcp,
            port: 0,
            host: String::new(),
        },
        WispPacket::Data {
            stream: 4,
            payload: Vec::new(),
        },
        WispPacket::Data {
            stream: u32::MAX,
            payload: (0..16384).map(|index| index as u8).collect(),
        },
        WispPacket::Continue {
            stream: 0,
            remaining: 0x0102_0304,
        },
        WispPacket::Close {
            stream: 6,
            reason: CLOSE_VOLUNTARY,
        },
    ];
    for packet in packets {
        assert_eq!(decode(&encode(&packet)), Some(packet));
    }
}

#[test]
fn close_and_continue_have_fixed_payload_sizes() {
    let close = encode(&WispPacket::Close {
        stream: 1,
        reason: CLOSE_REFUSED,
    });
    assert_eq!(close, [0x04, 1, 0, 0, 0, 0x44]);
    let cont = encode(&WispPacket::Continue {
        stream: 1,
        remaining: 5,
    });
    assert_eq!(cont, [0x03, 1, 0, 0, 0, 5, 0, 0, 0]);
}

#[test]
fn unknown_info_and_truncated_packets_decode_to_none() {
    assert_eq!(decode(&[0x05, 0, 0, 0, 0, 2, 0]), None);
    assert_eq!(decode(&[0x09, 0, 0, 0, 0]), None);
    assert_eq!(decode(&[0x03, 0, 0, 0, 0, 1]), None);
    assert_eq!(decode(&[]), None);
}

#[test]
fn malformed_connects_and_trailing_bytes() {
    // Hostname not UTF-8.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x01, 80, 0, 0xff, 0xfe]), None);
    // Unknown stream type.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x07, 80, 0, b'a']), None);
    // Missing port.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x01, 80]), None);
    // Trailing bytes after a fixed payload are ignored.
    assert_eq!(
        decode(&[0x04, 1, 0, 0, 0, 0x02, 9, 9]),
        Some(WispPacket::Close {
            stream: 1,
            reason: CLOSE_VOLUNTARY
        })
    );
    assert_eq!(
        decode(&[0x03, 1, 0, 0, 0, 7, 0, 0, 0, 9]),
        Some(WispPacket::Continue {
            stream: 1,
            remaining: 7
        })
    );
}

#[test]
fn decoding_never_panics() {
    let mut state = 0x5eed_0007_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    let packets = [
        WispPacket::Connect {
            stream: 9,
            kind: StreamKind::Udp,
            port: 5353,
            host: "host.example".into(),
        },
        WispPacket::Data {
            stream: 9,
            payload: vec![1, 2, 3],
        },
        WispPacket::Continue {
            stream: 9,
            remaining: 100,
        },
        WispPacket::Close {
            stream: 9,
            reason: CLOSE_BLOCKED,
        },
    ];
    for packet in &packets {
        let bytes = encode(packet);
        for end in 0..=bytes.len() {
            inputs.push(bytes[..end].to_vec());
        }
    }
    for len in 0..64 {
        inputs.push((0..len).map(|_| next() as u8).collect());
    }
    for input in inputs {
        let _ = decode(&input);
    }
}

// The client against a scripted server: what it sends is decoded, what the server says is
// encoded, and the guest side is the gateway's own test guest.

use core::net::Ipv4Addr;

use super::{HANDSHAKE_TIMEOUT_MS, WISP_DATA_MAX, WispClient, WispState};
use crate::gateway::tests::segment_policy::{dns_answer, dns_query, length_prefixed, udp_view};
use crate::gateway::tests::tcp::{SERVER, TcpGuest, TcpView, drain_tcp};
use crate::gateway::tests::{ACK, FIN, PSH, RESOLVER, RST, SYN, drain, segment_gateway, udp_frame};
use crate::gateway::{GATEWAY_IP, Gateway};

const NOW_MS: u64 = 0;
const SERVER_HOST: &str = "93.184.216.34";
const SERVER_PORT: u16 = 443;
const DNS_PORT: u16 = 53;
const GUEST_ISN: u32 = 1000;
const GUEST_PORT: u16 = 40000;
const OTHER_GUEST_PORT: u16 = 40001;
const THIRD_GUEST_PORT: u16 = 40002;
const PLENTY_OF_CREDIT: u32 = 1000;
const UNKNOWN_STREAM: u32 = 999;
/// v2's INFO packet: type 5, stream 0, protocol version 2.0.
const INFO_PACKET: [u8; 7] = [0x05, 0, 0, 0, 0, 2, 0];
const QUERY_ID: u16 = 0x1234;

struct Rig {
    gateway: Gateway,
    client: WispClient,
}

impl Rig {
    /// A client whose WebSocket was never opened.
    fn new() -> Rig {
        Rig {
            gateway: segment_gateway(),
            client: WispClient::new(),
        }
    }

    /// A client past the handshake, every new stream starting with `credit`.
    fn ready(credit: u32) -> Rig {
        let mut rig = Rig::new();
        rig.client.connecting(NOW_MS);
        rig.server(WispPacket::Continue {
            stream: 0,
            remaining: credit,
        });
        assert_eq!(rig.client.status().state, WispState::Ready);
        rig
    }

    fn server(&mut self, packet: WispPacket) {
        self.client.from_server(&encode(&packet), &mut self.gateway);
    }

    /// One round of the web driver's loop at `now_ms`, then what the client sent.
    fn pump_at(&mut self, now_ms: u64) -> Vec<WispPacket> {
        self.client.pump(&mut self.gateway, now_ms);
        self.gateway.poll(now_ms);
        self.sent()
    }

    fn pump(&mut self) -> Vec<WispPacket> {
        self.pump_at(NOW_MS)
    }

    fn sent(&mut self) -> Vec<WispPacket> {
        self.client
            .take_messages()
            .iter()
            .map(|message| decode(message).expect("the client sent a malformed packet"))
            .collect()
    }

    fn guest_frames(&mut self) -> Vec<TcpView> {
        drain_tcp(&mut self.gateway)
    }

    /// A guest connection carried by a fresh Wisp stream, handshake done.
    fn connect(&mut self, guest_port: u16) -> (TcpGuest, u32) {
        let mut guest = TcpGuest::new(guest_port, SERVER_PORT, GUEST_ISN);
        guest.send(&mut self.gateway, SYN, &[]);
        let sent = self.pump();
        let &[WispPacket::Connect { stream, .. }] = &sent[..] else {
            panic!("expected one CONNECT: {sent:?}");
        };
        let synack = self
            .guest_frames()
            .into_iter()
            .find(|view| view.dport == guest_port && view.flags == SYN | ACK)
            .expect("a SYN-ACK");
        guest.ack = synack.seq.wrapping_add(1);
        guest.send(&mut self.gateway, ACK, &[]);
        (guest, stream)
    }
}

fn to_port(views: &[TcpView], guest_port: u16) -> Vec<&TcpView> {
    views
        .iter()
        .filter(|view| view.dport == guest_port)
        .collect()
}

fn has_rst(views: &[TcpView], guest_port: u16) -> bool {
    to_port(views, guest_port)
        .iter()
        .any(|view| view.flags & RST != 0)
}

fn data_of(sent: &[WispPacket], stream_id: u32) -> Vec<Vec<u8>> {
    sent.iter()
        .filter_map(|packet| match packet {
            WispPacket::Data { stream, payload } if *stream == stream_id => Some(payload.clone()),
            _ => None,
        })
        .collect()
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index * 7) as u8).collect()
}

fn is_tcp_connect_to_server(packet: &WispPacket) -> bool {
    matches!(packet, WispPacket::Connect { kind: StreamKind::Tcp, port, host, .. }
        if *port == SERVER_PORT && host == SERVER_HOST)
}

#[test]
fn nothing_is_sent_before_the_initial_continue() {
    let mut rig = Rig::new();
    rig.client.connecting(NOW_MS);
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    assert_eq!(rig.pump(), []);
    assert!(rig.guest_frames().is_empty(), "the SYN waits");
    assert_eq!(rig.client.status().state, WispState::Connecting);
    rig.server(WispPacket::Continue {
        stream: 0,
        remaining: 32,
    });
    let sent = rig.pump();
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert!(is_tcp_connect_to_server(&sent[0]), "{sent:?}");
}

#[test]
fn opens_fail_at_once_when_disconnected() {
    let mut rig = Rig::new();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    assert_eq!(rig.pump(), []);
    assert!(has_rst(&rig.guest_frames(), GUEST_PORT));
    assert_eq!(rig.client.status().state, WispState::Disconnected);
}

#[test]
fn the_handshake_times_out_after_ten_seconds() {
    assert_eq!(HANDSHAKE_TIMEOUT_MS, 10_000);
    let mut rig = Rig::new();
    rig.client.connecting(NOW_MS);
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    assert_eq!(rig.pump_at(NOW_MS + HANDSHAKE_TIMEOUT_MS - 1), []);
    assert!(rig.guest_frames().is_empty());
    assert_eq!(rig.client.status().state, WispState::Connecting);
    assert_eq!(rig.pump_at(NOW_MS + HANDSHAKE_TIMEOUT_MS), []);
    assert!(has_rst(&rig.guest_frames(), GUEST_PORT));
    assert_eq!(rig.client.status().state, WispState::Disconnected);
}

#[test]
fn the_syn_ack_goes_out_when_connect_is_sent() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    let sent = rig.pump();
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert!(is_tcp_connect_to_server(&sent[0]), "{sent:?}");
    let frames = rig.guest_frames();
    let to_guest = to_port(&frames, GUEST_PORT);
    assert_eq!(to_guest.len(), 1);
    assert_eq!(to_guest[0].flags, SYN | ACK);
}

#[test]
fn data_waits_for_credit() {
    let mut rig = Rig::ready(2);
    let (mut guest, stream) = rig.connect(GUEST_PORT);
    let payload = pattern(3 * WISP_DATA_MAX);
    guest.send(&mut rig.gateway, ACK | PSH, &payload);
    let first = data_of(&rig.pump(), stream);
    assert_eq!(first.len(), 2);
    assert_eq!(rig.pump(), [], "out of credit");
    rig.server(WispPacket::Continue {
        stream,
        remaining: 2,
    });
    let second = data_of(&rig.pump(), stream);
    assert_eq!(second.len(), 1);
    assert_eq!([first, second].concat().concat(), payload);
}

#[test]
fn data_payloads_never_exceed_the_cap() {
    const ROUNDS: usize = 4;
    const PER_ROUND: usize = 20_000;
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let (mut guest, stream) = rig.connect(GUEST_PORT);
    let payload = pattern(ROUNDS * PER_ROUND);
    let mut carried = Vec::new();
    // Rounds of guest data between pumps wrap the gateway's ring, so `to_remote` hands out
    // pieces that the client has to join.
    for chunk in payload.chunks(PER_ROUND) {
        guest.send(&mut rig.gateway, ACK | PSH, chunk);
        for data in data_of(&rig.pump(), stream) {
            assert!(data.len() <= WISP_DATA_MAX, "{} bytes", data.len());
            carried.extend_from_slice(&data);
        }
    }
    assert_eq!(carried, payload);
}

#[test]
fn voluntary_close_is_a_fin_and_others_are_resets() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let (_, closed) = rig.connect(GUEST_PORT);
    let (_, refused) = rig.connect(OTHER_GUEST_PORT);
    rig.server(WispPacket::Close {
        stream: closed,
        reason: CLOSE_VOLUNTARY,
    });
    assert_eq!(rig.pump(), []);
    let frames = rig.guest_frames();
    let to_guest = to_port(&frames, GUEST_PORT);
    assert!(to_guest.iter().any(|view| view.flags & FIN != 0));
    assert!(to_guest.iter().all(|view| view.flags & RST == 0));
    assert_eq!(rig.client.status().last_close_reason, None);
    rig.server(WispPacket::Close {
        stream: refused,
        reason: CLOSE_REFUSED,
    });
    assert_eq!(rig.pump(), []);
    let frames = rig.guest_frames();
    assert!(has_rst(&frames, OTHER_GUEST_PORT));
    assert!(to_port(&frames, GUEST_PORT).is_empty());
    assert_eq!(rig.client.status().last_close_reason, Some(CLOSE_REFUSED));
}

#[test]
fn packets_for_unknown_or_closed_streams_are_ignored() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let (mut live_guest, live) = rig.connect(GUEST_PORT);
    let (_, reset) = rig.connect(OTHER_GUEST_PORT);
    let (_, finished) = rig.connect(THIRD_GUEST_PORT);
    rig.server(WispPacket::Close {
        stream: reset,
        reason: CLOSE_REFUSED,
    });
    rig.server(WispPacket::Close {
        stream: finished,
        reason: CLOSE_VOLUNTARY,
    });
    assert_eq!(rig.pump(), []);
    rig.guest_frames();
    for stream in [UNKNOWN_STREAM, reset, finished] {
        rig.server(WispPacket::Data {
            stream,
            payload: b"stray".to_vec(),
        });
        rig.server(WispPacket::Continue {
            stream,
            remaining: 5,
        });
        rig.server(WispPacket::Close {
            stream,
            reason: CLOSE_BLOCKED,
        });
    }
    assert_eq!(rig.pump(), []);
    assert!(rig.guest_frames().is_empty());
    assert_eq!(rig.client.status().last_close_reason, Some(CLOSE_REFUSED));
    rig.server(WispPacket::Data {
        stream: live,
        payload: b"hello".to_vec(),
    });
    live_guest.send(&mut rig.gateway, ACK | PSH, b"ping");
    assert_eq!(data_of(&rig.pump(), live), [b"ping".to_vec()]);
    let frames = rig.guest_frames();
    let received: Vec<u8> = to_port(&frames, GUEST_PORT)
        .iter()
        .flat_map(|view| view.data.clone())
        .collect();
    assert_eq!(received, b"hello");
}

#[test]
fn a_guest_close_after_a_server_close_sends_no_second_close() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let (mut closing_guest, closed_by_server) = rig.connect(GUEST_PORT);
    let (mut other_guest, closed_by_guest) = rig.connect(OTHER_GUEST_PORT);
    rig.server(WispPacket::Close {
        stream: closed_by_server,
        reason: CLOSE_VOLUNTARY,
    });
    assert_eq!(rig.pump(), []);
    let frames = rig.guest_frames();
    let fin = to_port(&frames, GUEST_PORT)
        .into_iter()
        .find(|view| view.flags & FIN != 0)
        .expect("the server's close reaches the guest as a FIN");
    closing_guest.ack = fin.seq.wrapping_add(1);
    // Bytes the guest writes after the server's close have nowhere to go, and must not keep its
    // FIN from ending the stream.
    closing_guest.send_with_fin(&mut rig.gateway, b"late");
    assert_eq!(rig.pump(), [], "the server already closed it");
    other_guest.send(&mut rig.gateway, RST | ACK, &[]);
    assert_eq!(
        rig.pump(),
        [WispPacket::Close {
            stream: closed_by_guest,
            reason: CLOSE_VOLUNTARY
        }]
    );
    assert!(rig.client.streams.is_empty(), "both streams are forgotten");
}

#[test]
fn losing_the_connection_resets_every_stream() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    let (_, first) = rig.connect(GUEST_PORT);
    let (_, second) = rig.connect(OTHER_GUEST_PORT);
    rig.client.connection_lost(&mut rig.gateway);
    assert_eq!(rig.pump(), []);
    let frames = rig.guest_frames();
    assert!(has_rst(&frames, GUEST_PORT));
    assert!(has_rst(&frames, OTHER_GUEST_PORT));
    assert_eq!(rig.client.status().state, WispState::Disconnected);
    rig.client.connecting(NOW_MS);
    rig.server(WispPacket::Continue {
        stream: 0,
        remaining: PLENTY_OF_CREDIT,
    });
    let (_, fresh) = rig.connect(THIRD_GUEST_PORT);
    assert_ne!(fresh, first);
    assert_ne!(fresh, second);
}

#[test]
fn udp_datagrams_need_no_credit() {
    let mut rig = Rig::ready(0);
    rig.gateway
        .from_guest(&udp_frame(GUEST_PORT, SERVER, DNS_PORT, b"query"), NOW_MS);
    let sent = rig.pump();
    let &[
        WispPacket::Connect {
            stream,
            kind: StreamKind::Udp,
            port: DNS_PORT,
            ref host,
        },
        WispPacket::Data {
            stream: data_stream,
            ref payload,
        },
    ] = &sent[..]
    else {
        panic!("expected a UDP CONNECT and one DATA: {sent:?}");
    };
    assert_eq!(host, SERVER_HOST);
    assert_eq!(data_stream, stream);
    assert_eq!(payload, b"query");
    rig.server(WispPacket::Data {
        stream,
        payload: b"answer".to_vec(),
    });
    assert!(rig.gateway.pop_for_guest().is_some(), "the answer");
}

#[test]
fn info_packets_are_ignored() {
    let mut rig = Rig::new();
    rig.client.connecting(NOW_MS);
    rig.client.from_server(&INFO_PACKET, &mut rig.gateway);
    assert_eq!(rig.client.status().state, WispState::Connecting);
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    assert_eq!(rig.pump(), []);
    assert!(rig.guest_frames().is_empty());
    rig.server(WispPacket::Continue {
        stream: 0,
        remaining: PLENTY_OF_CREDIT,
    });
    rig.client.from_server(&INFO_PACKET, &mut rig.gateway);
    assert_eq!(rig.client.status().state, WispState::Ready);
    assert_eq!(rig.pump().len(), 1);
}

#[test]
fn an_open_abandoned_while_connecting_is_never_sent() {
    let mut rig = Rig::new();
    rig.client.connecting(NOW_MS);
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut rig.gateway, SYN, &[]);
    assert_eq!(rig.pump(), []);
    guest.send(&mut rig.gateway, RST, &[]);
    assert_eq!(rig.pump(), []);
    rig.server(WispPacket::Continue {
        stream: 0,
        remaining: PLENTY_OF_CREDIT,
    });
    assert_eq!(rig.pump(), []);
}

#[test]
fn a_continue_on_stream_zero_after_the_handshake_changes_nothing() {
    let mut rig = Rig::ready(1);
    rig.server(WispPacket::Continue {
        stream: 0,
        remaining: PLENTY_OF_CREDIT,
    });
    let (mut guest, stream) = rig.connect(GUEST_PORT);
    guest.send(&mut rig.gateway, ACK | PSH, &pattern(2 * WISP_DATA_MAX));
    assert_eq!(data_of(&rig.pump(), stream).len(), 1);
}

#[test]
fn server_input_never_panics() {
    const ROUNDS: usize = 2000;
    const MAX_RANDOM_LEN: u64 = 24;
    let mut state = 0x5eed_0008_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut rig = Rig::ready(1);
    let (mut guest, first) = rig.connect(GUEST_PORT);
    let (_, second) = rig.connect(OTHER_GUEST_PORT);
    let streams = [0, first, second, UNKNOWN_STREAM, u32::MAX];
    for round in 0..ROUNDS {
        let stream = streams[next() as usize % streams.len()];
        let message = match next() % 5 {
            0 => encode(&WispPacket::Data {
                stream,
                payload: pattern(next() as usize % 100),
            }),
            1 => encode(&WispPacket::Continue {
                stream,
                remaining: next() as u32,
            }),
            2 => encode(&WispPacket::Close {
                stream,
                reason: next() as u8,
            }),
            3 => encode(&WispPacket::Connect {
                stream,
                kind: StreamKind::Tcp,
                port: next() as u16,
                host: "203.0.113.1".into(),
            }),
            _ => (0..next() % MAX_RANDOM_LEN).map(|_| next() as u8).collect(),
        };
        rig.client.from_server(&message, &mut rig.gateway);
        if round % 7 == 0 {
            guest.send(&mut rig.gateway, ACK | PSH, b"more");
        }
        if round % 500 == 499 {
            rig.client.connection_lost(&mut rig.gateway);
            rig.client.connecting(round as u64);
        }
        rig.pump_at(round as u64);
        rig.guest_frames();
    }
}

#[test]
fn a_refused_udp_resolver_stream_falls_back_to_dns_over_tcp() {
    let mut rig = Rig::ready(PLENTY_OF_CREDIT);
    rig.gateway.from_guest(
        &udp_frame(GUEST_PORT, GATEWAY_IP, DNS_PORT, &dns_query(QUERY_ID)),
        NOW_MS,
    );
    let sent = rig.pump();
    let resolver_host = Ipv4Addr::from(RESOLVER.address).to_string();
    let &[
        WispPacket::Connect {
            stream: udp,
            kind: StreamKind::Udp,
            ref host,
            ..
        },
        WispPacket::Data { .. },
    ] = &sent[..]
    else {
        panic!("expected the query over UDP first: {sent:?}");
    };
    assert_eq!(*host, resolver_host);
    rig.server(WispPacket::Close {
        stream: udp,
        reason: CLOSE_BLOCKED,
    });
    assert_eq!(rig.client.status().last_close_reason, Some(CLOSE_BLOCKED));
    let sent = rig.pump();
    let &[
        WispPacket::Connect {
            stream: tcp,
            kind: StreamKind::Tcp,
            port,
            ref host,
        },
        ..,
    ] = &sent[..]
    else {
        panic!("expected a TCP CONNECT to the resolver: {sent:?}");
    };
    assert_eq!(
        (host.as_str(), port),
        (resolver_host.as_str(), RESOLVER.port)
    );
    assert_eq!(
        data_of(&sent, tcp).concat(),
        length_prefixed(&dns_query(QUERY_ID))
    );
    assert!(drain(&mut rig.gateway).is_empty(), "no port unreachable");
    rig.server(WispPacket::Data {
        stream: tcp,
        payload: length_prefixed(&dns_answer(QUERY_ID)),
    });
    let reply = udp_view(&drain(&mut rig.gateway));
    assert_eq!(
        (reply.src, reply.sport, reply.dport),
        (GATEWAY_IP, DNS_PORT, GUEST_PORT)
    );
    assert_eq!(reply.data, dns_answer(QUERY_ID));
}
