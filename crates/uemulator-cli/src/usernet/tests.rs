//! Host-socket checks of the driver. The protocol itself is tested in the library's `gateway`
//! module; these build guest frames by hand, independently of it.

use std::net::{TcpListener, UdpSocket};

use super::*;
use crate::run::GUEST_MAC;
use crate::test_support::{XorShift, reference_checksum};

const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// How often the waits below poll the driver.
const POLL_PAUSE: Duration = Duration::from_millis(2);

const GATEWAY_MAC: [u8; 6] = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
const BROADCAST_MAC: [u8; 6] = [0xff; 6];
const GATEWAY_IP: [u8; 4] = [10, 0, 2, 2];
const DNS_IP: [u8; 4] = [10, 0, 2, 3];
const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
const LOOPBACK_IP: [u8; 4] = [127, 0, 0, 1];

const ETH_HEADER_LEN: usize = 14;
const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;
const IPV4_HEADER_LEN: usize = 20;
const IPV4_VERSION_IHL_NO_OPTIONS: u8 = 0x45;
const IPV4_FLAG_DONT_FRAGMENT: u16 = 0x4000;
const IPV4_TTL: u8 = 64;
const IP_PROTO_ICMP: u8 = 1;
const IP_PROTO_TCP: u8 = 6;
const IP_PROTO_UDP: u8 = 17;
const UDP_HEADER_LEN: usize = 8;
const TCP_HEADER_LEN: usize = 20;
/// The gateway's MSS: what its segments to a guest that asked for nothing larger stay within.
const MSS: usize = 1460;

const FIN: u8 = 0x01;
const SYN: u8 = 0x02;
const RST: u8 = 0x04;
const PSH: u8 = 0x08;
const ACK: u8 = 0x10;

fn be_u16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

fn be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn ethernet_frame(dst: [u8; 6], src: [u8; 6], ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(ETH_HEADER_LEN + payload.len());
    frame.extend_from_slice(&dst);
    frame.extend_from_slice(&src);
    frame.extend_from_slice(&ethertype.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

fn ipv4_packet(id: u16, src: [u8; 4], dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let mut packet = vec![IPV4_VERSION_IHL_NO_OPTIONS, 0];
    packet.extend_from_slice(&((IPV4_HEADER_LEN + payload.len()) as u16).to_be_bytes());
    packet.extend_from_slice(&id.to_be_bytes());
    packet.extend_from_slice(&IPV4_FLAG_DONT_FRAGMENT.to_be_bytes());
    packet.extend_from_slice(&[IPV4_TTL, proto, 0, 0]);
    packet.extend_from_slice(&src);
    packet.extend_from_slice(&dst);
    let checksum = reference_checksum(&packet);
    packet[10..12].copy_from_slice(&checksum.to_be_bytes());
    packet.extend_from_slice(payload);
    packet
}

/// TCP/UDP checksum over the IPv4 pseudo-header and `segment`; 0 for a segment whose stored
/// checksum is right.
fn transport_checksum(src: [u8; 4], dst: [u8; 4], proto: u8, segment: &[u8]) -> u16 {
    let mut summed = Vec::with_capacity(12 + segment.len());
    summed.extend_from_slice(&src);
    summed.extend_from_slice(&dst);
    summed.extend_from_slice(&[0, proto]);
    summed.extend_from_slice(&(segment.len() as u16).to_be_bytes());
    summed.extend_from_slice(segment);
    reference_checksum(&summed)
}

fn udp_segment(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, data: &[u8]) -> Vec<u8> {
    let mut segment = Vec::with_capacity(UDP_HEADER_LEN + data.len());
    segment.extend_from_slice(&sport.to_be_bytes());
    segment.extend_from_slice(&dport.to_be_bytes());
    segment.extend_from_slice(&((UDP_HEADER_LEN + data.len()) as u16).to_be_bytes());
    segment.extend_from_slice(&[0, 0]);
    segment.extend_from_slice(data);
    let checksum = transport_checksum(src, dst, IP_PROTO_UDP, &segment);
    segment[6..8].copy_from_slice(&checksum.to_be_bytes());
    segment
}

fn guest_ip_frame(dst: [u8; 4], proto: u8, payload: &[u8]) -> Vec<u8> {
    let packet = ipv4_packet(1, GUEST_IP, dst, proto, payload);
    ethernet_frame(GATEWAY_MAC, GUEST_MAC, ETHERTYPE_IPV4, &packet)
}

fn udp_frame(sport: u16, dst: [u8; 4], dport: u16, data: &[u8]) -> Vec<u8> {
    guest_ip_frame(
        dst,
        IP_PROTO_UDP,
        &udp_segment(GUEST_IP, dst, sport, dport, data),
    )
}

struct Ipv4View<'a> {
    src: [u8; 4],
    dst: [u8; 4],
    proto: u8,
    payload: &'a [u8],
}

/// An IPv4 frame for the guest, with its header checked; `None` for anything else.
fn parse_ipv4(frame: &[u8]) -> Option<Ipv4View<'_>> {
    if frame.len() < ETH_HEADER_LEN + IPV4_HEADER_LEN || be_u16(&frame[12..14]) != ETHERTYPE_IPV4 {
        return None;
    }
    let ip = &frame[ETH_HEADER_LEN..];
    assert_eq!(ip[0], IPV4_VERSION_IHL_NO_OPTIONS);
    assert_eq!(
        reference_checksum(&ip[..IPV4_HEADER_LEN]),
        0,
        "IPv4 header checksum"
    );
    let total_len = be_u16(&ip[2..4]) as usize;
    Some(Ipv4View {
        src: ip[12..16].try_into().unwrap(),
        dst: ip[16..20].try_into().unwrap(),
        proto: ip[9],
        payload: &ip[IPV4_HEADER_LEN..total_len],
    })
}

fn drain(un: &mut UserNet) -> Vec<Vec<u8>> {
    std::iter::from_fn(|| un.pop_frame().map(|frame| frame.bytes)).collect()
}

fn wait_for(un: &mut UserNet, mut pred: impl FnMut(&[u8]) -> bool) -> Vec<u8> {
    let end = Instant::now() + IO_TIMEOUT;
    while Instant::now() < end {
        un.poll();
        while let Some(f) = un.pop_frame() {
            if pred(&f.bytes) {
                return f.bytes;
            }
        }
        std::thread::sleep(POLL_PAUSE);
    }
    panic!("timed out waiting for frame");
}

fn loopback_udp_server() -> (UdpSocket, u16) {
    let server = UdpSocket::bind("127.0.0.1:0").unwrap();
    server.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    let port = server.local_addr().unwrap().port();
    (server, port)
}

/// The one UDP datagram the driver produced: source address, ports and data, after checking its
/// checksum.
fn udp_reply(un: &mut UserNet) -> ([u8; 4], u16, u16, Vec<u8>) {
    let frame = wait_for(un, |f| {
        parse_ipv4(f).is_some_and(|ip| ip.proto == IP_PROTO_UDP)
    });
    let ip = parse_ipv4(&frame).unwrap();
    assert_eq!(ip.dst, GUEST_IP);
    assert_eq!(
        transport_checksum(ip.src, ip.dst, IP_PROTO_UDP, ip.payload),
        0,
        "UDP checksum"
    );
    assert_eq!(be_u16(&ip.payload[4..6]) as usize, ip.payload.len());
    (
        ip.src,
        be_u16(&ip.payload[0..2]),
        be_u16(&ip.payload[2..4]),
        ip.payload[UDP_HEADER_LEN..].to_vec(),
    )
}

mod nameservers {
    use super::*;

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
    }
}

mod udp_nat {
    use super::*;

    #[test]
    fn the_gateway_address_reaches_host_loopback_and_replies_come_from_it() {
        let (server, port) = loopback_udp_server();
        let mut un = UserNet::new();
        un.input(&udp_frame(4100, GATEWAY_IP, port, b"to-host"));
        un.poll();
        let mut buf = [0u8; 16];
        let (n, from) = server.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"to-host");
        server.send_to(b"from-host", from).unwrap();
        let (src, sport, dport, data) = udp_reply(&mut un);
        assert_eq!(src, GATEWAY_IP);
        assert_eq!((sport, dport), (port, 4100));
        assert_eq!(data, b"from-host");
    }

    #[test]
    fn dns_queries_are_forwarded_to_the_host_nameserver_and_answered_from_the_dns_address() {
        let (nameserver, nameserver_port) = loopback_udp_server();
        let mut un = UserNet::with_dns(SocketAddrV4::new(Ipv4Addr::LOCALHOST, nameserver_port));
        un.input(&udp_frame(33333, DNS_IP, DNS_PORT, b"\x12\x34 query"));
        un.poll();
        let mut buf = [0u8; 64];
        let (n, from) = nameserver.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"\x12\x34 query");
        nameserver.send_to(b"\x12\x34 answer", from).unwrap();
        let (src, sport, dport, data) = udp_reply(&mut un);
        assert_eq!(src, DNS_IP);
        assert_eq!((sport, dport), (DNS_PORT, 33333));
        assert_eq!(data, b"\x12\x34 answer");
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
        let mut segment = Vec::with_capacity(TCP_HEADER_LEN + data.len());
        segment.extend_from_slice(&self.sport.to_be_bytes());
        segment.extend_from_slice(&self.dport.to_be_bytes());
        segment.extend_from_slice(&self.seq.to_be_bytes());
        segment.extend_from_slice(&self.ack.to_be_bytes());
        // Header length in 32-bit words in the high nibble.
        segment.push(((TCP_HEADER_LEN / 4) as u8) << 4);
        segment.push(flags);
        segment.extend_from_slice(&self.window.to_be_bytes());
        // Checksum and urgent pointer.
        segment.extend_from_slice(&[0, 0, 0, 0]);
        segment.extend_from_slice(data);
        let checksum = transport_checksum(GUEST_IP, self.dst, IP_PROTO_TCP, &segment);
        segment[16..18].copy_from_slice(&checksum.to_be_bytes());
        segment
    }

    fn send(&mut self, un: &mut UserNet, flags: u8, data: &[u8]) {
        let segment = self.segment(flags, data);
        un.input(&guest_ip_frame(self.dst, IP_PROTO_TCP, &segment));
        self.seq = self
            .seq
            .wrapping_add(data.len() as u32 + (flags & (SYN | FIN) != 0) as u32);
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
        transport_checksum(ip.src, ip.dst, IP_PROTO_TCP, ip.payload),
        0,
        "bad tcp checksum"
    );
    let segment = ip.payload;
    let header_len = (segment[12] >> 4) as usize * 4;
    Some(TcpView {
        src: ip.src,
        dst: ip.dst,
        sport: be_u16(&segment[0..2]),
        dport: be_u16(&segment[2..4]),
        flags: segment[13],
        seq: be_u32(&segment[4..8]),
        ack: be_u32(&segment[8..12]),
        window: be_u16(&segment[14..16]),
        data: segment[header_len..].to_vec(),
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

/// Runs the three-way handshake against `listener` and returns the accepted host side.
fn handshake(un: &mut UserNet, guest: &mut Guest, listener: &TcpListener) -> TcpStream {
    guest.send(un, SYN, &[]);
    let synack = wait_for_tcp(un, |v| v.flags == SYN | ACK);
    assert_eq!(synack.ack, guest.seq);
    guest.ack = synack.seq.wrapping_add(1);
    guest.send(un, ACK, &[]);
    let (server, _) = listener.accept().unwrap();
    server.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    server
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

fn poll_a_few_times(un: &mut UserNet) {
    for _ in 0..5 {
        un.poll();
        std::thread::sleep(POLL_PAUSE);
    }
}

/// Serves one connection on `listener`, sending back whatever arrives until EOF.
fn spawn_echo_server(listener: TcpListener) {
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
}

mod tcp {
    use super::*;

    #[test]
    fn tcp_to_the_gateway_address_reaches_host_loopback() {
        let (host_service, port) = listener();
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.dst = GATEWAY_IP;
        let mut server = handshake(&mut un, &mut guest, &host_service);
        guest.send(&mut un, ACK, b"hi host");
        poll_a_few_times(&mut un);
        assert_eq!(read_exactly(&mut server, 7), b"hi host");
        server.write_all(b"hi guest").unwrap();
        let segment = wait_for_tcp(&mut un, |v| !v.data.is_empty());
        assert_eq!((segment.src, segment.dst), (GATEWAY_IP, GUEST_IP));
        assert_eq!(segment.data, b"hi guest");
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
        assert!(un.sockets.is_empty(), "the failed stream is cleaned up");
    }

    #[test]
    fn full_duplex_echo_survives_a_close_from_both_ends() {
        let (listener, port) = listener();
        spawn_echo_server(listener);
        let mut un = UserNet::new();
        let mut guest = Guest::new(5000, port, 1000);
        guest.send(&mut un, SYN, &[]);
        let synack = wait_for_tcp(&mut un, |v| v.flags == SYN | ACK);
        guest.ack = synack.seq.wrapping_add(1);
        guest.send(&mut un, ACK, &[]);
        let payload = XorShift::new(77).bytes(5000);
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
        assert!(
            un.sockets.is_empty(),
            "the socket is closed after both FINs are acked"
        );
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
        un.input(&ethernet_frame(
            BROADCAST_MAC,
            GUEST_MAC,
            ETHERTYPE_ARP,
            &arp,
        ));
        let out = drain(&mut un);
        assert_eq!(out[0][20..22], [0, 2]);
        assert_eq!(out[0][22..28], GATEWAY_MAC);

        let mut echo = vec![8, 0, 0, 0, 0, 1, 0, 1, 9, 9];
        let c = reference_checksum(&echo);
        echo[2..4].copy_from_slice(&c.to_be_bytes());
        un.input(&guest_ip_frame([8, 8, 8, 8], IP_PROTO_ICMP, &echo));
        let out = drain(&mut un);
        let ip = parse_ipv4(&out[0]).unwrap();
        assert_eq!((ip.src, ip.dst, ip.payload[0]), ([8, 8, 8, 8], GUEST_IP, 0));
        assert_eq!(reference_checksum(ip.payload), 0);
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
        un.poll();
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
        spawn_echo_server(listener);
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
        assert!(
            un.sockets.is_empty(),
            "flow closed after both FINs are acked"
        );
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
