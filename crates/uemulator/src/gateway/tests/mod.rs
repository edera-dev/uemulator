//! Frames the guest would send, built from the wire module's own codecs and checked by
//! independent code (`reference_checksum`, `reassemble`). The test modules reach the helpers
//! through `use super::*`.

use super::dhcp::*;
use super::udp::MAX_UDP_FLOWS;
use super::wire::*;
use super::*;
use alloc::collections::BTreeMap;

mod dhcp;
mod link;
mod parsing;
mod tcp;
mod udp;
mod wire;

const FUZZ_SEED: u64 = 0x5eed_0002;
const NOW_MS: u64 = 0;
const GUEST_MAC: [u8; 6] = DEFAULT_GUEST_MAC;
const OTHER_GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x99];
const ARP_REQUEST_FRAME_LEN: usize = ETH_HEADER_LEN + ARP_PACKET_LEN;
const IP_PROTO_GRE: u8 = 47;
const RESOLVER: Endpoint = Endpoint {
    address: [1, 1, 1, 1],
    port: DNS_PORT,
};
const SOME_SEQUENCE: u32 = 1000;

fn user_net() -> Gateway {
    Gateway::new(GatewayConfig {
        policy: GatewayPolicy::UserNet { dns: RESOLVER },
        isn_seed: 1,
    })
}

fn segment_gateway() -> Gateway {
    Gateway::new(GatewayConfig {
        policy: GatewayPolicy::Segment { resolver: RESOLVER },
        isn_seed: 1,
    })
}

fn guest_ip_frame(dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    ip_frame_to(GATEWAY_MAC, dst, proto, payload)
}

fn ip_frame_to(dst_mac: [u8; 6], dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let p = ipv4_packet(1, GUEST_IP, dst, proto, payload);
    ethernet_frame(dst_mac, GUEST_MAC, 0x0800, &p)
}

fn udp_frame(sport: u16, dst: [u8; 4], dport: u16, data: &[u8]) -> Vec<u8> {
    guest_ip_frame(
        dst,
        IP_PROTO_UDP,
        &udp_segment(GUEST_IP, dst, sport, dport, data),
    )
}

fn udp_frame_to(dst_mac: [u8; 6], dst: [u8; 4], dport: u16) -> Vec<u8> {
    ip_frame_to(
        dst_mac,
        dst,
        IP_PROTO_UDP,
        &udp_segment(GUEST_IP, dst, 40000, dport, b"data"),
    )
}

fn tcp_syn_frame(dst_mac: [u8; 6], dst: [u8; 4], dport: u16) -> Vec<u8> {
    let header = TcpHeader {
        src: GUEST_IP,
        dst,
        sport: 40000,
        dport,
        seq: SOME_SEQUENCE,
        ack: 0,
        flags: SYN,
        window: u16::MAX,
        advertise_mss: true,
    };
    ip_frame_to(dst_mac, dst, IP_PROTO_TCP, &tcp_segment(&header, &[]))
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

/// The driver side of the stream API, recording what the gateway asked for per stream so tests
/// can assert on it without sockets.
#[derive(Default)]
struct FakeRemote {
    streams: BTreeMap<StreamId, FakeStream>,
}

struct FakeStream {
    kind: StreamKind,
    target: Endpoint,
    /// Received and not yet drained.
    datagrams: Vec<Vec<u8>>,
    write_shutdown: bool,
    closed: bool,
}

impl FakeRemote {
    /// Takes the gateway's pending requests, records them and returns them.
    fn requests(&mut self, gateway: &mut Gateway) -> Vec<GatewayRequest> {
        let requests = gateway.take_requests();
        for request in &requests {
            match request {
                GatewayRequest::Open {
                    stream,
                    kind,
                    target,
                } => {
                    let fresh = FakeStream {
                        kind: *kind,
                        target: *target,
                        datagrams: Vec::new(),
                        write_shutdown: false,
                        closed: false,
                    };
                    assert!(
                        self.streams.insert(*stream, fresh).is_none(),
                        "stream {stream} opened twice"
                    );
                }
                GatewayRequest::Datagram { stream, data } => {
                    self.live(*stream).datagrams.push(data.clone());
                }
                GatewayRequest::ShutdownWrite { stream } => {
                    let entry = self.live(*stream);
                    assert!(!entry.write_shutdown, "stream {stream} shut down twice");
                    entry.write_shutdown = true;
                }
                GatewayRequest::Close { stream } => self.live(*stream).closed = true,
            }
        }
        requests
    }

    fn live(&mut self, stream: StreamId) -> &mut FakeStream {
        let entry = self
            .streams
            .get_mut(&stream)
            .unwrap_or_else(|| panic!("request for unopened stream {stream}"));
        assert!(!entry.closed, "request for closed stream {stream}");
        entry
    }

    /// Asserts the gateway asked for exactly one new stream and returns it with what it is for.
    fn single_open_detail(&mut self, gateway: &mut Gateway) -> (StreamId, StreamKind, Endpoint) {
        let opens: Vec<_> = self
            .requests(gateway)
            .into_iter()
            .filter_map(|request| match request {
                GatewayRequest::Open {
                    stream,
                    kind,
                    target,
                } => Some((stream, kind, target)),
                _ => None,
            })
            .collect();
        assert_eq!(opens.len(), 1, "expected exactly one Open: {opens:?}");
        opens[0]
    }

    fn single_open(&mut self, gateway: &mut Gateway) -> StreamId {
        self.single_open_detail(gateway).0
    }

    /// Datagrams recorded for `stream` and not drained yet; takes no new requests.
    fn datagrams(&self, stream: StreamId) -> &[Vec<u8>] {
        &self.streams[&stream].datagrams
    }

    /// Takes new requests, then returns and forgets the datagrams recorded for `stream`.
    fn drain_datagrams(&mut self, gateway: &mut Gateway, stream: StreamId) -> Vec<Vec<u8>> {
        self.requests(gateway);
        core::mem::take(&mut self.live(stream).datagrams)
    }

    fn kind(&self, stream: StreamId) -> StreamKind {
        self.streams[&stream].kind
    }

    fn target(&self, stream: StreamId) -> Endpoint {
        self.streams[&stream].target
    }

    fn is_closed(&self, stream: StreamId) -> bool {
        self.streams[&stream].closed
    }

    fn is_write_shutdown(&self, stream: StreamId) -> bool {
        self.streams[&stream].write_shutdown
    }

    /// Plays a remote end that takes everything the guest sent: drains `to_remote` and returns
    /// the bytes.
    fn take_written(&mut self, gateway: &mut Gateway, stream: StreamId) -> Vec<u8> {
        let mut taken = Vec::new();
        loop {
            let waiting = gateway.to_remote(stream);
            if waiting.is_empty() {
                return taken;
            }
            taken.extend_from_slice(waiting);
            let len = waiting.len();
            gateway.consumed(stream, len);
        }
    }

    /// Streams that were opened and not closed.
    fn live_streams(&self) -> Vec<StreamId> {
        self.streams
            .iter()
            .filter(|(_, stream)| !stream.closed)
            .map(|(id, _)| *id)
            .collect()
    }
}

fn user_net_with_remote() -> (Gateway, FakeRemote) {
    (user_net(), FakeRemote::default())
}

fn segment_with_remote() -> (Gateway, FakeRemote) {
    (segment_gateway(), FakeRemote::default())
}

fn drain(un: &mut Gateway) -> Vec<Vec<u8>> {
    core::iter::from_fn(|| un.pop_for_guest().map(|frame| frame.bytes)).collect()
}

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

/// A DHCP message from a client as the guest sends it: from 0.0.0.0 to the limited broadcast.
fn dhcp_frame(message: &[u8]) -> Vec<u8> {
    let datagram = udp_segment(
        ZERO_IP,
        LIMITED_BROADCAST_IP,
        DHCP_CLIENT_PORT,
        DHCP_SERVER_PORT,
        message,
    );
    let packet = ipv4_packet(1, ZERO_IP, LIMITED_BROADCAST_IP, IP_PROTO_UDP, &datagram);
    ethernet_frame(BROADCAST_MAC, GUEST_MAC, ETHERTYPE_IPV4, &packet)
}

fn dhcp_discover_frame() -> Vec<u8> {
    dhcp_frame(&client_message(&options_of(DHCP_DISCOVER, &[]), ZERO_IP))
}

/// RFC 1071 internet checksum written independently of the production routines (byte at a time,
/// 32-bit accumulator), for checking them.
fn reference_checksum(data: &[u8]) -> u16 {
    let mut sum = 0u32;
    for (index, byte) in data.iter().enumerate() {
        sum += if index % 2 == 0 {
            (*byte as u32) << 8
        } else {
            *byte as u32
        };
    }
    while sum > 0xffff {
        sum = (sum >> 16) + (sum & 0xffff);
    }
    !(sum as u16)
}

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        // A zero state would stay zero forever.
        XorShift(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        const MULTIPLIER: u64 = 0x2545_f491_4f6c_dd1d;
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(MULTIPLIER)
    }

    /// Uniform enough for tests: `0..bound` (`bound` must be non-zero).
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

fn splitmix_bytes(seed: u64, len: usize) -> Vec<u8> {
    XorShift::new(seed).bytes(len)
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
