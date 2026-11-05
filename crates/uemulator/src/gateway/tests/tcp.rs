//! TCP flows against `FakeRemote`: what a host socket read was in usernet's tests is
//! `remote_data` here, a host write is `to_remote`/`consumed`, a refused connect is `failed` and
//! a host close is `remote_closed`.

use super::*;
use crate::gateway::tcp::{
    ADVERTISED_WINDOW_MAX, ISN_STEP, LARGE_SEGMENT_DATA_MAX, TcpFlow, advertised_window,
};

const SERVER: [u8; 4] = [93, 184, 216, 34];
const SERVER_PORT: u16 = 8080;
const GUEST_PORT: u16 = 5000;
const GUEST_ISN: u32 = 1000;
const GUEST_WINDOW: u16 = 60000;
/// Polls a transfer may take before a test calls it stuck.
const MAX_ROUNDS: usize = 10_000;
const TCP_START: usize = ETH_HEADER_LEN + IPV4_HEADER_LEN;

#[derive(Clone)]
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
    segment_size: Option<u16>,
}

/// Decodes a TCP-over-IPv4 frame, asserting that both checksums are valid; `None` for other
/// traffic.
fn tcp_view(frame: &[u8]) -> Option<TcpView> {
    let ip = parse_ipv4(frame)?;
    if ip.proto != IP_PROTO_TCP {
        return None;
    }
    assert_eq!(reference_checksum(&frame[ETH_HEADER_LEN..TCP_START]), 0);
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
        segment_size: None,
    })
}

/// A TSO segment's checksum finished the way the device does before the guest sees it: the
/// field already holds the pseudo-header sum, so summing from the TCP header on completes it.
fn finish_checksum(bytes: &mut [u8]) {
    let checksum = checksum_finish(checksum_add(0, &bytes[TCP_START..]));
    bytes[TCP_START + TCP_CHECKSUM.start..TCP_START + TCP_CHECKSUM.end]
        .copy_from_slice(&checksum.to_be_bytes());
}

/// Every TCP frame queued for the guest, decoded.
fn drain_tcp(gateway: &mut Gateway) -> Vec<TcpView> {
    let mut views = Vec::new();
    while let Some(frame) = gateway.pop_for_guest() {
        let mut bytes = frame.bytes;
        if frame.segment_size.is_some() {
            finish_checksum(&mut bytes);
        }
        if let Some(mut view) = tcp_view(&bytes) {
            view.segment_size = frame.segment_size;
            views.push(view);
        }
    }
    views
}

fn data_segments(views: &[TcpView]) -> Vec<&TcpView> {
    views.iter().filter(|view| !view.data.is_empty()).collect()
}

/// The guest end of one connection.
struct TcpGuest {
    sport: u16,
    dport: u16,
    dst: [u8; 4],
    seq: u32,
    ack: u32,
    window: u16,
    /// Next sequence number expected from the gateway; `receive` checks segments arrive in order.
    peer_next: u32,
    stream: StreamId,
}

impl TcpGuest {
    fn new(sport: u16, dport: u16, seq: u32) -> TcpGuest {
        TcpGuest {
            sport,
            dport,
            dst: SERVER,
            seq,
            ack: 0,
            window: GUEST_WINDOW,
            peer_next: 0,
            stream: 0,
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

    fn send(&mut self, gateway: &mut Gateway, flags: u8, data: &[u8]) {
        let segment = self.segment(flags, data);
        gateway.from_guest(&guest_ip_frame(self.dst, IP_PROTO_TCP, &segment), NOW_MS);
        self.seq = self
            .seq
            .wrapping_add(data.len() as u32 + (flags & (SYN | FIN) != 0) as u32);
    }

    fn send_with_fin(&mut self, gateway: &mut Gateway, data: &[u8]) {
        self.send(gateway, FIN | ACK | PSH, data);
    }

    /// Everything queued for the guest, decoded, with data and FINs checked to arrive in order.
    fn receive(&mut self, gateway: &mut Gateway) -> Vec<TcpView> {
        let views = drain_tcp(gateway);
        for view in &views {
            if view.flags & SYN != 0 {
                self.peer_next = view.seq.wrapping_add(1);
            } else if view.flags & RST == 0 && (!view.data.is_empty() || view.flags & FIN != 0) {
                assert_eq!(view.seq, self.peer_next, "in-order delivery");
                self.peer_next = view
                    .seq
                    .wrapping_add(view.data.len() as u32 + (view.flags & FIN != 0) as u32);
            }
        }
        views
    }

    /// Acknowledges everything received so far.
    fn ack_all(&mut self, gateway: &mut Gateway) {
        self.ack = self.peer_next;
        self.send(gateway, ACK, &[]);
    }

    /// Polls, receives and acknowledges until a poll brings no data, and returns the data.
    fn read_all(&mut self, gateway: &mut Gateway) -> Vec<u8> {
        let mut received = Vec::new();
        for _ in 0..MAX_ROUNDS {
            gateway.poll(NOW_MS);
            let segments = self.receive(gateway);
            if segments.iter().all(|segment| segment.data.is_empty()) {
                return received;
            }
            received.extend(segments.iter().flat_map(|segment| segment.data.clone()));
            self.ack_all(gateway);
        }
        panic!("data kept coming after {MAX_ROUNDS} polls");
    }

    fn key(&self) -> FlowKey {
        FlowKey {
            guest_port: self.sport,
            remote_ip: self.dst,
            remote_port: self.dport,
        }
    }
}

/// Runs the three-way handshake with the driver opening the stream, and returns our initial
/// sequence number.
fn handshake(gateway: &mut Gateway, remote: &mut FakeRemote, guest: &mut TcpGuest) -> u32 {
    guest.send(gateway, SYN, &[]);
    guest.stream = remote.single_open(gateway);
    gateway.opened(guest.stream);
    let synack = guest
        .receive(gateway)
        .into_iter()
        .find(|view| view.flags & SYN != 0)
        .expect("a SYN-ACK");
    assert_eq!(synack.flags, SYN | ACK);
    assert_eq!(synack.ack, guest.seq);
    guest.ack = synack.seq.wrapping_add(1);
    guest.send(gateway, ACK, &[]);
    synack.seq
}

fn established(window: u16) -> (Gateway, FakeRemote, TcpGuest) {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.window = window;
    handshake(&mut gateway, &mut remote, &mut guest);
    (gateway, remote, guest)
}

/// Collects `want` bytes of data, acknowledging as it goes, and checks no segment exceeds `mss`.
fn receive_data(gateway: &mut Gateway, guest: &mut TcpGuest, want: usize, mss: usize) -> Vec<u8> {
    let mut received = Vec::new();
    for _ in 0..MAX_ROUNDS {
        if received.len() >= want {
            return received;
        }
        gateway.poll(NOW_MS);
        for segment in data_segments(&guest.receive(gateway)) {
            assert!(
                segment.data.len() <= mss,
                "segment of {} exceeds mss {mss}",
                segment.data.len()
            );
            received.extend_from_slice(&segment.data);
        }
        guest.ack_all(gateway);
    }
    panic!("only {} of {want} bytes arrived", received.len());
}

fn flow<'a>(gateway: &'a Gateway, guest: &TcpGuest) -> &'a TcpFlow {
    &gateway.tcp.flows[&guest.key()]
}

#[test]
fn a_remote_close_waits_for_buffered_data_before_the_fin() {
    let (mut gateway, _remote, mut guest) = established(/* guest window */ 1000);
    gateway.remote_data(guest.stream, &[7u8; 5000]);
    gateway.remote_closed(guest.stream);
    let mut received = Vec::new();
    loop {
        gateway.poll(0);
        let segments = guest.receive(&mut gateway);
        if segments.iter().any(|s| s.flags & FIN != 0) {
            break;
        }
        received.extend(segments.iter().flat_map(|s| s.data.clone()));
        guest.ack_all(&mut gateway);
    }
    assert_eq!(received, vec![7u8; 5000]);
}

#[test]
fn remote_data_beyond_room_is_kept_not_dropped() {
    let (mut gateway, _, mut guest) = established(1000);
    assert_eq!(gateway.room(guest.stream), 1000);
    gateway.remote_data(guest.stream, &[1u8; 200_000]);
    assert_eq!(guest.read_all(&mut gateway).len(), 200_000);
}

#[test]
fn guest_fin_asks_for_shutdown_write_after_pending_bytes() {
    let (mut gateway, mut remote, mut guest) = established(65535);
    guest.send_with_fin(&mut gateway, b"tail");
    assert!(
        !remote
            .requests(&mut gateway)
            .contains(&GatewayRequest::ShutdownWrite {
                stream: guest.stream
            })
    );
    assert_eq!(gateway.to_remote(guest.stream), b"tail");
    gateway.consumed(guest.stream, 4);
    gateway.poll(0);
    assert!(
        remote
            .requests(&mut gateway)
            .contains(&GatewayRequest::ShutdownWrite {
                stream: guest.stream
            })
    );
}

#[test]
fn advertised_window_shrinks_with_the_backlog() {
    assert_eq!(advertised_window(0), 65535);
    assert_eq!(advertised_window(1), 65534);
    assert_eq!(advertised_window(65535), 0);
}

#[test]
fn advertised_window_never_wraps_below_zero() {
    assert_eq!(advertised_window(65536), 0);
    assert_eq!(advertised_window(usize::MAX), 0);
}

#[test]
fn a_syn_asks_the_driver_to_open_and_waits_for_it() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    let (stream, kind, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(kind, StreamKind::Tcp);
    assert_eq!(
        target,
        Endpoint {
            address: SERVER,
            port: SERVER_PORT
        }
    );
    gateway.poll(NOW_MS);
    assert!(drain(&mut gateway).is_empty(), "no SYN-ACK before the open");
    gateway.opened(stream);
    assert_eq!(drain_tcp(&mut gateway)[0].flags, SYN | ACK);
}

#[test]
fn syn_ack_carries_the_documented_fields() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.isn = 0x1000_0000;
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    let synack = drain_tcp(&mut gateway).remove(0);
    assert_eq!(synack.flags, SYN | ACK);
    assert_eq!(synack.seq, 0x1000_0000 + ISN_STEP);
    assert_eq!(synack.ack, 1001);
    assert_eq!(synack.window, 65535);
    assert_eq!(synack.options, [TCP_OPT_MSS, 4, 0x05, 0xb4]);
    assert_eq!((synack.src, synack.dst), (SERVER, GUEST_IP));
    assert_eq!((synack.sport, synack.dport), (SERVER_PORT, GUEST_PORT));
    assert!(synack.data.is_empty());
}

#[test]
fn each_connection_gets_a_new_initial_sequence_number() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut isns = Vec::new();
    for guest_port in [5000u16, 5001, 5002] {
        let mut guest = TcpGuest::new(guest_port, SERVER_PORT, 1);
        isns.push(handshake(&mut gateway, &mut remote, &mut guest));
    }
    assert_eq!(isns[1], isns[0].wrapping_add(ISN_STEP));
    assert_eq!(isns[2], isns[1].wrapping_add(ISN_STEP));
}

#[test]
fn handshake_completes_and_the_flow_is_established() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    let flow = flow(&gateway, &guest);
    assert!(flow.opened);
    assert_eq!(flow.stream, guest.stream);
    assert_eq!(flow.snd_nxt, isn.wrapping_add(1));
    assert_eq!(flow.snd_una, isn.wrapping_add(1), "our SYN is acknowledged");
    assert_eq!(flow.rcv_nxt, 1001);
    assert_eq!(flow.mss, MSS);
    assert_eq!(flow.guest_window, 60000);
}

#[test]
fn the_guests_mss_option_limits_our_segment_size() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let with_mss = TcpHeader {
        src: GUEST_IP,
        dst: SERVER,
        sport: GUEST_PORT,
        dport: SERVER_PORT,
        seq: GUEST_ISN,
        ack: 0,
        flags: SYN,
        window: GUEST_WINDOW,
        advertise_mss: true,
    };
    let mut syn = tcp_segment(&with_mss, &[]);
    syn[22..24].copy_from_slice(&536u16.to_be_bytes());
    gateway.from_guest(&guest_ip_frame(SERVER, IP_PROTO_TCP, &syn), NOW_MS);
    guest.seq += 1;
    guest.stream = remote.single_open(&mut gateway);
    gateway.opened(guest.stream);
    let synack = guest.receive(&mut gateway).remove(0);
    assert_eq!(
        synack.options,
        [TCP_OPT_MSS, 4, 0x05, 0xb4],
        "we still advertise our own"
    );
    guest.ack = synack.seq.wrapping_add(1);
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(flow(&gateway, &guest).mss, 536);
    let data = splitmix_bytes(11, 5000);
    gateway.remote_data(guest.stream, &data);
    assert_eq!(receive_data(&mut gateway, &mut guest, 5000, 536), data);
}

#[test]
fn data_from_the_guest_reaches_the_remote_and_is_acknowledged_at_the_next_poll() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    guest.send(&mut gateway, ACK | PSH, b"hello");
    assert!(
        drain(&mut gateway).is_empty(),
        "the ACK waits for the driver to take the data"
    );
    assert_eq!(remote.take_written(&mut gateway, guest.stream), b"hello");
    gateway.poll(NOW_MS);
    let ack = drain_tcp(&mut gateway).remove(0);
    assert_eq!(ack.flags, ACK);
    assert_eq!(ack.ack, 1006);
    assert_eq!(ack.seq, isn.wrapping_add(1));
    assert!(ack.data.is_empty());
}

#[test]
fn guest_data_is_delivered_in_order_across_segments() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    let data = splitmix_bytes(21, 6000);
    for chunk in data.chunks(MSS) {
        guest.send(&mut gateway, ACK, chunk);
    }
    assert_eq!(remote.take_written(&mut gateway, guest.stream), data);
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 1001 + data.len() as u32);
}

#[test]
fn acks_for_a_batch_of_guest_segments_are_coalesced() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    for chunk in [b"one", b"two", b"six"] {
        guest.send(&mut gateway, ACK, chunk);
    }
    remote.take_written(&mut gateway, guest.stream);
    gateway.poll(NOW_MS);
    let acks: Vec<u32> = drain_tcp(&mut gateway).iter().map(|v| v.ack).collect();
    assert_eq!(acks, vec![1010]);
}

#[test]
fn data_from_the_remote_is_segmented_to_the_mss_and_sequenced() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    let data = splitmix_bytes(31, 4000);
    gateway.remote_data(guest.stream, &data);
    gateway.poll(NOW_MS);
    let segments = guest.receive(&mut gateway);
    let first = &segments[0];
    assert_eq!(first.flags, PSH | ACK);
    assert_eq!(first.ack, 1001);
    assert_eq!(first.seq, isn.wrapping_add(1));
    assert!(first.data.len() <= MSS);
    let mut received: Vec<u8> = segments.iter().flat_map(|v| v.data.clone()).collect();
    guest.ack_all(&mut gateway);
    received.extend(receive_data(
        &mut gateway,
        &mut guest,
        data.len() - received.len(),
        MSS,
    ));
    assert_eq!(received, data);
}

#[test]
fn remote_data_is_limited_to_the_guests_advertised_window() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.window = 100;
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_data(guest.stream, &splitmix_bytes(41, 1000));
    gateway.poll(NOW_MS);
    let got: usize = drain_tcp(&mut gateway).iter().map(|v| v.data.len()).sum();
    assert_eq!(got, 100, "never more than the window in flight");
    for _ in 0..5 {
        gateway.poll(NOW_MS);
    }
    assert!(drain_tcp(&mut gateway).iter().all(|v| v.data.is_empty()));
    let flow = flow(&gateway, &guest);
    assert_eq!(flow.snd_nxt.wrapping_sub(flow.snd_una), 100);
    assert_eq!(flow.snd_nxt, isn.wrapping_add(1 + 100));
    assert_eq!(gateway.room(guest.stream), 0);
}

#[test]
fn an_ack_with_a_bigger_window_releases_the_rest() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.window = 100;
    handshake(&mut gateway, &mut remote, &mut guest);
    let data = splitmix_bytes(42, 1000);
    gateway.remote_data(guest.stream, &data);
    gateway.poll(NOW_MS);
    let mut received: Vec<u8> = guest
        .receive(&mut gateway)
        .iter()
        .flat_map(|v| v.data.clone())
        .collect();
    assert_eq!(received.len(), 100);
    guest.window = 60000;
    guest.ack_all(&mut gateway);
    received.extend(receive_data(
        &mut gateway,
        &mut guest,
        data.len() - received.len(),
        MSS,
    ));
    assert_eq!(received, data);
}

#[test]
fn a_zero_window_stops_the_remote_data() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.window = 0;
    handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_data(guest.stream, b"stalled");
    for _ in 0..5 {
        gateway.poll(NOW_MS);
    }
    assert!(drain_tcp(&mut gateway).iter().all(|v| v.data.is_empty()));
    guest.window = 1000;
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(receive_data(&mut gateway, &mut guest, 7, MSS), b"stalled");
}

#[test]
fn retransmitted_guest_data_is_delivered_once() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    let resend_from = guest.seq;
    guest.send(&mut gateway, ACK, b"abc");
    guest.seq = resend_from;
    guest.send(&mut gateway, ACK, b"abc");
    let duplicate = drain_tcp(&mut gateway);
    assert_eq!(duplicate.len(), 1, "a retransmit is acknowledged at once");
    assert_eq!(duplicate[0].ack, 1004);
    gateway.poll(NOW_MS);
    assert!(
        drain_tcp(&mut gateway).iter().all(|a| a.ack == 1004),
        "everything acknowledges up to 1004"
    );
    guest.send(&mut gateway, ACK, b"def");
    assert_eq!(remote.take_written(&mut gateway, guest.stream), b"abcdef");
}

#[test]
fn partially_overlapping_retransmit_delivers_only_the_new_bytes() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, ACK, b"abc");
    guest.seq = 1001 + 1;
    guest.send(&mut gateway, ACK, b"bcdef");
    assert_eq!(remote.take_written(&mut gateway, guest.stream), b"abcdef");
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 1007);
}

#[test]
fn a_segment_beyond_a_gap_is_dropped_and_answered_with_a_duplicate_ack() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.seq = 1001 + 10;
    guest.send(&mut gateway, ACK, b"late");
    let reply = drain_tcp(&mut gateway).remove(0);
    assert_eq!(reply.ack, 1001, "still waiting for the missing bytes");
    assert!(gateway.to_remote(guest.stream).is_empty());
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 1001);
    guest.seq = 1001;
    guest.send(&mut gateway, ACK, b"0123456789");
    assert_eq!(
        remote.take_written(&mut gateway, guest.stream),
        b"0123456789"
    );
}

#[test]
fn guest_sequence_numbers_wrap_around() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, 0xffff_fff0);
    handshake(&mut gateway, &mut remote, &mut guest);
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 0xffff_fff1);
    guest.send(&mut gateway, ACK, &[7u8; 40]);
    gateway.poll(NOW_MS);
    let ack = drain_tcp(&mut gateway).remove(0);
    assert_eq!(ack.ack, 0x19, "0xfffffff1 + 40 wraps to 0x19");
    guest.send(&mut gateway, ACK, b"after");
    assert_eq!(
        remote.take_written(&mut gateway, guest.stream)[40..],
        *b"after"
    );
}

#[test]
fn our_sequence_numbers_wrap_around_and_acks_still_advance_snd_una() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.isn = 0xffff_fffd - ISN_STEP;
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    assert_eq!(isn, 0xffff_fffd);
    gateway.remote_data(guest.stream, &[9u8; 10]);
    gateway.poll(NOW_MS);
    let segment = guest.receive(&mut gateway).remove(0);
    assert_eq!(segment.seq, 0xffff_fffe);
    assert_eq!(segment.data.len(), 10);
    let after = 0xffff_fffeu32.wrapping_add(10);
    assert_eq!(after, 8);
    assert_eq!(flow(&gateway, &guest).snd_nxt, 8);
    guest.ack_all(&mut gateway);
    assert_eq!(
        flow(&gateway, &guest).snd_una,
        8,
        "the ack past the wrap is accepted"
    );
}

#[test]
fn acks_outside_the_sent_range_do_not_move_snd_una() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_data(guest.stream, &[1u8; 100]);
    gateway.poll(NOW_MS);
    let got: usize = drain_tcp(&mut gateway).iter().map(|v| v.data.len()).sum();
    assert_eq!(got, 100);
    let base = isn.wrapping_add(1);
    guest.ack = base.wrapping_add(201);
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(
        flow(&gateway, &guest).snd_una,
        base,
        "acknowledges data we never sent"
    );
    guest.ack = base.wrapping_add(100);
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(flow(&gateway, &guest).snd_una, base.wrapping_add(100));
    guest.ack = base.wrapping_add(50);
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(
        flow(&gateway, &guest).snd_una,
        base.wrapping_add(100),
        "stale ack"
    );
}

#[test]
fn guest_window_follows_each_ack() {
    let (mut gateway, _, mut guest) = established(GUEST_WINDOW);
    guest.window = 1234;
    guest.send(&mut gateway, ACK, &[]);
    assert_eq!(flow(&gateway, &guest).guest_window, 1234);
}

#[test]
fn data_the_remote_takes_at_once_leaves_the_acknowledged_window_open() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, ACK, &[0u8; 1000]);
    assert_eq!(
        remote.take_written(&mut gateway, guest.stream),
        vec![0u8; 1000]
    );
    gateway.poll(NOW_MS);
    let ack = drain_tcp(&mut gateway).remove(0);
    assert_eq!(
        ack.window, 65535,
        "the driver took the data before the ACK was built"
    );
}

#[test]
fn data_the_remote_has_not_taken_shrinks_the_acknowledged_window() {
    let (mut gateway, _, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, ACK, &[0u8; 1000]);
    gateway.poll(NOW_MS);
    assert_eq!(drain_tcp(&mut gateway).remove(0).window, 65535 - 1000);
}

#[test]
fn guest_data_beyond_our_window_is_cut_to_it() {
    let (mut gateway, _, mut guest) = established(GUEST_WINDOW);
    for _ in 0..50 {
        guest.send(&mut gateway, ACK, &[0x5a; MSS]);
    }
    assert_eq!(
        flow(&gateway, &guest).to_remote.len(),
        ADVERTISED_WINDOW_MAX
    );
    assert_eq!(
        flow(&gateway, &guest).rcv_nxt,
        1001 + ADVERTISED_WINDOW_MAX as u32
    );
}

/// A guest that filled the window must hear that it opened again once the driver takes the
/// backlog; without the update it waits for its persist timer (200 ms or more).
#[test]
fn a_window_reopened_by_the_remote_taking_data_is_announced_to_the_guest() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    let mut sent = 0;
    loop {
        guest.send(&mut gateway, ACK, &[0x5a; MSS]);
        sent += MSS;
        gateway.poll(NOW_MS);
        let window = drain_tcp(&mut gateway)
            .iter()
            .map(|v| v.window)
            .min()
            .unwrap();
        if (window as usize) < ADVERTISED_WINDOW_MAX / 2 {
            break;
        }
    }
    assert_eq!(remote.take_written(&mut gateway, guest.stream).len(), sent);
    gateway.poll(NOW_MS);
    let update = drain_tcp(&mut gateway).remove(0);
    assert_eq!(
        (update.flags, update.window as usize),
        (ACK, ADVERTISED_WINDOW_MAX)
    );
    assert!(update.data.is_empty());
    assert_eq!(update.ack, 1001u32.wrapping_add(sent as u32));
}

#[test]
fn guest_fin_is_acknowledged_at_once_and_asks_for_shutdown_write() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, FIN | ACK, &[]);
    let ack = drain_tcp(&mut gateway).remove(0);
    assert_eq!(ack.flags, ACK);
    assert_eq!(ack.ack, 1002, "the FIN takes one sequence number");
    remote.requests(&mut gateway);
    assert!(remote.is_write_shutdown(guest.stream));
    assert!(!remote.is_closed(guest.stream));
}

#[test]
fn guest_data_before_the_fin_reaches_the_remote_first() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send_with_fin(&mut gateway, b"last words");
    remote.requests(&mut gateway);
    assert!(!remote.is_write_shutdown(guest.stream));
    assert_eq!(
        remote.take_written(&mut gateway, guest.stream),
        b"last words"
    );
    gateway.poll(NOW_MS);
    remote.requests(&mut gateway);
    assert!(remote.is_write_shutdown(guest.stream));
    assert_eq!(drain_tcp(&mut gateway).remove(0).ack, 1000 + 1 + 10 + 1);
}

#[test]
fn a_retransmitted_fin_is_acknowledged_without_advancing_twice() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, FIN | ACK, &[]);
    guest.seq = 1001;
    guest.send(&mut gateway, FIN | ACK, &[]);
    let acks: Vec<u32> = drain_tcp(&mut gateway).iter().map(|v| v.ack).collect();
    assert_eq!(acks, vec![1002, 1002]);
    remote.requests(&mut gateway);
}

#[test]
fn remote_close_sends_fin_and_the_flow_ends_after_both_sides_are_acknowledged() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    let fin = guest.receive(&mut gateway).remove(0);
    assert_eq!(fin.flags, FIN | ACK);
    assert_eq!(fin.seq, isn.wrapping_add(1));
    assert!(flow(&gateway, &guest).fin_sent);
    guest.ack_all(&mut gateway);
    gateway.poll(NOW_MS);
    assert_eq!(
        gateway.tcp.flows.len(),
        1,
        "guest has not closed its side yet"
    );
    remote.requests(&mut gateway);
    assert!(!remote.is_closed(guest.stream));
    guest.send(&mut gateway, FIN | ACK, &[]);
    gateway.poll(NOW_MS);
    assert!(gateway.tcp.flows.is_empty(), "both FINs acknowledged");
    remote.requests(&mut gateway);
    assert!(remote.is_write_shutdown(guest.stream));
    assert!(remote.is_closed(guest.stream));
}

#[test]
fn a_fin_only_ends_the_flow_once_ours_is_acknowledged() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    let fin = drain_tcp(&mut gateway).remove(0);
    guest.ack = isn.wrapping_add(1);
    guest.send(&mut gateway, FIN | ACK, &[]);
    gateway.poll(NOW_MS);
    assert_eq!(
        gateway.tcp.flows.len(),
        1,
        "our FIN is still unacknowledged"
    );
    guest.ack = fin.seq.wrapping_add(1);
    guest.send(&mut gateway, ACK, &[]);
    assert!(gateway.tcp.flows.is_empty());
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
}

#[test]
fn a_remote_close_after_the_guest_finished_ends_with_close() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send_with_fin(&mut gateway, b"bye");
    remote.take_written(&mut gateway, guest.stream);
    gateway.poll(NOW_MS);
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    let fin = guest
        .receive(&mut gateway)
        .into_iter()
        .find(|v| v.flags & FIN != 0)
        .expect("our FIN");
    assert_eq!(fin.ack, 1001 + 3 + 1);
    guest.ack_all(&mut gateway);
    assert!(gateway.tcp.flows.is_empty(), "nothing is left to wait for");
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
}

#[test]
fn a_close_waits_for_guest_bytes_the_driver_has_not_taken() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    guest.receive(&mut gateway);
    guest.ack_all(&mut gateway);
    guest.send_with_fin(&mut gateway, b"late reply");
    remote.requests(&mut gateway);
    assert!(!remote.is_closed(guest.stream), "bytes still waiting");
    assert_eq!(
        remote.take_written(&mut gateway, guest.stream),
        b"late reply"
    );
    gateway.poll(NOW_MS);
    let acks: Vec<u32> = drain_tcp(&mut gateway).iter().map(|v| v.ack).collect();
    assert_eq!(
        acks,
        vec![1001 + 10 + 1],
        "the owed ACK goes before the close"
    );
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
    assert!(gateway.tcp.flows.is_empty());
}

/// The guest's last ACK can end the flow before any `poll`: an ACK still owed for its data must
/// go out first, or the guest retransmits into a flow that no longer exists.
#[test]
fn an_owed_ack_goes_out_when_a_guest_ack_ends_the_flow_before_the_poll() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    let fin = drain_tcp(&mut gateway).remove(0);
    guest.ack = isn.wrapping_add(1);
    guest.send_with_fin(&mut gateway, b"last");
    assert_eq!(remote.take_written(&mut gateway, guest.stream), b"last");
    guest.ack = fin.seq.wrapping_add(1);
    guest.send(&mut gateway, ACK, &[]);
    assert!(gateway.tcp.flows.is_empty());
    let acks: Vec<u32> = drain_tcp(&mut gateway).iter().map(|v| v.ack).collect();
    assert_eq!(acks, vec![1001 + 4 + 1]);
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
}

#[test]
fn guest_rst_removes_the_flow_and_asks_for_close_at_once() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, RST | ACK, &[]);
    assert!(drain(&mut gateway).is_empty(), "a RST is never answered");
    assert!(gateway.tcp.flows.is_empty());
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
}

#[test]
fn guest_rst_while_opening_closes_the_stream() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    let stream = remote.single_open(&mut gateway);
    guest.send(&mut gateway, RST, &[]);
    remote.requests(&mut gateway);
    assert!(remote.is_closed(stream));
    gateway.opened(stream);
    assert!(drain(&mut gateway).is_empty(), "a late open is ignored");
}

#[test]
fn a_remote_reset_resets_the_guest_without_a_close() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    let isn = handshake(&mut gateway, &mut remote, &mut guest);
    gateway.reset(guest.stream);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!(reset.flags, RST | ACK);
    assert_eq!((reset.seq, reset.ack), (isn.wrapping_add(1), 1001));
    assert!(gateway.tcp.flows.is_empty());
    remote.requests(&mut gateway);
    assert!(!remote.is_closed(guest.stream));
}

#[test]
fn a_failed_open_resets_the_guest() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(5001, SERVER_PORT, 7);
    guest.send(&mut gateway, SYN, &[]);
    let stream = remote.single_open(&mut gateway);
    gateway.failed(stream);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!(reset.flags, RST | ACK);
    assert_eq!(reset.ack, 8, "acknowledges the SYN");
    assert_eq!(reset.seq, 0);
    assert_eq!(reset.window, 0);
    assert_eq!((reset.src, reset.dst), (SERVER, GUEST_IP));
    assert_eq!((reset.sport, reset.dport), (SERVER_PORT, 5001));
    assert!(
        gateway.tcp.flows.is_empty(),
        "the failed flow is cleaned up"
    );
    remote.requests(&mut gateway);
    assert!(!remote.is_closed(stream), "a stream that never opened");
}

#[test]
fn a_failed_flow_can_be_retried() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(5002, SERVER_PORT, 7);
    guest.send(&mut gateway, SYN, &[]);
    let first = remote.single_open(&mut gateway);
    gateway.failed(first);
    drain(&mut gateway);
    let mut retry = TcpGuest::new(5002, SERVER_PORT, 100);
    handshake(&mut gateway, &mut remote, &mut retry);
    assert_ne!(retry.stream, first);
}

#[test]
fn a_repeated_syn_while_opening_does_not_create_a_second_flow() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    guest.seq = GUEST_ISN;
    guest.send(&mut gateway, SYN, &[]);
    assert_eq!(gateway.tcp.flows.len(), 1);
    let stream = remote.single_open(&mut gateway);
    gateway.opened(stream);
    for _ in 0..5 {
        gateway.poll(NOW_MS);
    }
    let synacks = drain_tcp(&mut gateway)
        .iter()
        .filter(|v| v.flags == SYN | ACK)
        .count();
    assert_eq!(synacks, 1, "exactly one SYN-ACK");
}

#[test]
fn a_syn_on_an_established_flow_is_ignored() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    guest.seq = GUEST_ISN;
    guest.send(&mut gateway, SYN, &[]);
    assert!(drain(&mut gateway).is_empty());
    assert_eq!(gateway.tcp.flows.len(), 1);
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 1001);
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn data_before_the_stream_is_open_is_ignored() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    remote.single_open(&mut gateway);
    guest.send(&mut gateway, ACK, b"too early");
    gateway.poll(NOW_MS);
    assert!(drain(&mut gateway).is_empty());
    assert_eq!(flow(&gateway, &guest).rcv_nxt, 1001);
    assert!(flow(&gateway, &guest).to_remote.is_empty());
}

#[test]
fn segments_for_unknown_flows_are_reset() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(6000, 9, 500);
    guest.ack = 12345;
    guest.send(&mut gateway, ACK, &[]);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!(
        reset.flags, RST,
        "an ACK is answered with a bare RST at its ack number"
    );
    assert_eq!(reset.seq, 12345);
    assert_eq!(reset.window, 0);
    assert_eq!((reset.src, reset.dst), (SERVER, GUEST_IP));
    assert_eq!((reset.sport, reset.dport), (9, 6000));
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn segments_without_ack_for_unknown_flows_are_reset_with_rst_ack() {
    let mut gateway = user_net();
    let mut data = TcpGuest::new(6000, 9, 500);
    data.send(&mut gateway, PSH, b"12345");
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 505));

    let mut fin = TcpGuest::new(6001, 9, 900);
    fin.send(&mut gateway, FIN, &[]);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 901));

    let mut bare = TcpGuest::new(6002, 9, 77);
    bare.send(&mut gateway, 0, &[]);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!((reset.flags, reset.seq, reset.ack), (RST | ACK, 0, 77));
}

#[test]
fn reset_acknowledgement_wraps_around() {
    let mut gateway = user_net();
    let mut guest = TcpGuest::new(6000, 9, u32::MAX);
    guest.send(&mut gateway, PSH, b"ab");
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!(reset.ack, 1, "0xffffffff + 2");
}

#[test]
fn a_syn_with_ack_for_an_unknown_flow_is_reset_not_opened() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(6000, 9, 500);
    guest.ack = 4242;
    guest.send(&mut gateway, SYN | ACK, &[]);
    let reset = drain_tcp(&mut gateway).remove(0);
    assert_eq!((reset.flags, reset.seq), (RST, 4242));
    assert!(gateway.tcp.flows.is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn a_rst_for_an_unknown_flow_is_silently_dropped() {
    let mut gateway = user_net();
    let mut guest = TcpGuest::new(6000, 9, 500);
    guest.send(&mut gateway, RST, &[]);
    guest.send(&mut gateway, RST | ACK, &[]);
    assert!(drain(&mut gateway).is_empty());
}

#[test]
fn malformed_tcp_is_ignored() {
    let mut gateway = user_net();
    for len in 0..TCP_HEADER_LEN {
        gateway.from_guest(
            &guest_ip_frame(SERVER, IP_PROTO_TCP, &vec![0u8; len]),
            NOW_MS,
        );
    }
    let mut segment = TcpGuest::new(1, 2, 3).segment(SYN, &[]);
    segment[TCP_DATA_OFFSET] = 0xf0;
    gateway.from_guest(&guest_ip_frame(SERVER, IP_PROTO_TCP, &segment), NOW_MS);
    assert!(drain(&mut gateway).is_empty());
    assert!(gateway.tcp.flows.is_empty() && gateway.take_requests().is_empty());
}

#[test]
fn dns_over_tcp_goes_to_the_configured_server() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, DNS_PORT, GUEST_ISN);
    guest.dst = DNS_IP;
    guest.send(&mut gateway, SYN, &[]);
    let (stream, _, target) = remote.single_open_detail(&mut gateway);
    assert_eq!(target, RESOLVER);
    gateway.opened(stream);
    let synack = drain_tcp(&mut gateway).remove(0);
    assert_eq!((synack.src, synack.sport), (DNS_IP, DNS_PORT));
}

#[test]
fn tcp_to_the_gateway_address_reaches_host_loopback() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.dst = GATEWAY_IP;
    handshake(&mut gateway, &mut remote, &mut guest);
    assert_eq!(
        remote.target(guest.stream),
        Endpoint {
            address: LOOPBACK_IP,
            port: SERVER_PORT
        }
    );
    assert_eq!(flow(&gateway, &guest).guest_ip, GUEST_IP);
    guest.send(&mut gateway, ACK, b"hi host");
    assert_eq!(remote.take_written(&mut gateway, guest.stream), b"hi host");
    gateway.remote_data(guest.stream, b"hi guest");
    gateway.poll(NOW_MS);
    let segment = data_segments(&drain_tcp(&mut gateway))[0].clone();
    assert_eq!(segment.src, GATEWAY_IP);
    assert_eq!(segment.data, b"hi guest");
}

#[test]
fn a_segment_gateway_sends_tcp_dns_at_the_gateway_address_to_its_resolver() {
    let (mut gateway, mut remote) = segment_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, DNS_PORT, GUEST_ISN);
    guest.dst = GATEWAY_IP;
    guest.send(&mut gateway, SYN, &[]);
    assert_eq!(remote.single_open_detail(&mut gateway).2, RESOLVER);
    let mut web = TcpGuest::new(GUEST_PORT + 1, 443, GUEST_ISN);
    web.send(&mut gateway, SYN, &[]);
    assert_eq!(
        remote.single_open_detail(&mut gateway).2,
        Endpoint {
            address: SERVER,
            port: 443
        }
    );
}

#[test]
fn stream_ids_skip_live_tcp_streams_when_the_counter_wraps() {
    let (mut gateway, mut remote, guest) = established(GUEST_WINDOW);
    gateway.next_stream = guest.stream;
    gateway.from_guest(&udp_frame(4000, SERVER, 53, b"q"), NOW_MS);
    assert_ne!(remote.single_open(&mut gateway), guest.stream);
    let mut other = TcpGuest::new(GUEST_PORT + 1, SERVER_PORT, GUEST_ISN);
    gateway.next_stream = guest.stream;
    other.send(&mut gateway, SYN, &[]);
    assert_ne!(remote.single_open(&mut gateway), guest.stream);
}

#[test]
fn room_is_the_guest_window_less_what_is_in_flight_and_buffered() {
    let (mut gateway, _, mut guest) = established(5000);
    assert_eq!(gateway.room(guest.stream), 5000);
    gateway.remote_data(guest.stream, &[1; 1200]);
    assert_eq!(gateway.room(guest.stream), 3800, "buffered");
    gateway.poll(NOW_MS);
    assert_eq!(gateway.room(guest.stream), 3800, "now in flight");
    guest.receive(&mut gateway);
    guest.ack_all(&mut gateway);
    assert_eq!(gateway.room(guest.stream), 5000);
    gateway.remote_data(guest.stream, &[1; 9000]);
    assert_eq!(gateway.room(guest.stream), 0, "never below zero");
    gateway.remote_closed(guest.stream);
    assert_eq!(gateway.room(guest.stream), 0);
}

#[test]
fn calls_for_unknown_streams_are_harmless() {
    let (mut gateway, mut remote, guest) = established(GUEST_WINDOW);
    let unknown = guest.stream + 100;
    assert_eq!(gateway.room(unknown), 0);
    assert!(gateway.to_remote(unknown).is_empty());
    gateway.consumed(unknown, 10);
    gateway.remote_data(unknown, b"stray");
    gateway.remote_closed(unknown);
    gateway.opened(unknown);
    gateway.failed(unknown);
    gateway.reset(unknown);
    gateway.poll(NOW_MS);
    assert!(drain(&mut gateway).is_empty());
    assert!(remote.requests(&mut gateway).is_empty());
}

#[test]
fn consuming_more_than_is_waiting_takes_only_what_is_there() {
    let (mut gateway, _, mut guest) = established(GUEST_WINDOW);
    guest.send(&mut gateway, ACK, b"abc");
    gateway.consumed(guest.stream, 10);
    assert!(gateway.to_remote(guest.stream).is_empty());
    guest.send(&mut gateway, ACK, b"def");
    assert_eq!(gateway.to_remote(guest.stream), b"def");
}

#[test]
fn interest_asks_to_write_only_while_guest_bytes_wait() {
    let (mut gateway, _, mut guest) = established(GUEST_WINDOW);
    let of = |gateway: &Gateway, stream| {
        gateway
            .interest()
            .into_iter()
            .find(|(id, _)| *id == stream)
            .map(|(_, interest)| interest)
    };
    assert_eq!(
        of(&gateway, guest.stream),
        Some(StreamInterest {
            read: true,
            write: false
        })
    );
    guest.send(&mut gateway, ACK, b"waiting");
    assert!(of(&gateway, guest.stream).unwrap().write);
    gateway.consumed(guest.stream, 7);
    assert!(!of(&gateway, guest.stream).unwrap().write);
}

#[test]
fn interest_asks_to_read_only_with_room_and_queue_space() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
    guest.send(&mut gateway, SYN, &[]);
    let stream = remote.single_open(&mut gateway);
    assert!(gateway.interest().is_empty(), "nothing before the open");
    gateway.opened(stream);
    assert_eq!(
        gateway.interest(),
        vec![(
            stream,
            StreamInterest {
                read: true,
                write: false
            }
        )]
    );
    for _ in 0..OUT_QUEUE_LIMIT {
        gateway.out.push_back(GuestFrame::plain(Vec::new()));
    }
    assert!(gateway.interest().is_empty(), "the guest queue is full");
    gateway.out.clear();
    gateway.remote_data(stream, &vec![0; GUEST_WINDOW as usize]);
    assert!(gateway.interest().is_empty(), "no room left");
}

#[test]
fn interest_lists_open_udp_streams_for_reading_while_the_queue_has_room() {
    let (mut gateway, mut remote) = user_net_with_remote();
    gateway.from_guest(&udp_frame(4000, SERVER, 53, b"q"), NOW_MS);
    let stream = remote.single_open(&mut gateway);
    assert!(gateway.interest().is_empty(), "nothing before the open");
    gateway.opened(stream);
    assert_eq!(
        gateway.interest(),
        vec![(
            stream,
            StreamInterest {
                read: true,
                write: false
            }
        )]
    );
    for _ in 0..OUT_QUEUE_LIMIT {
        gateway.out.push_back(GuestFrame::plain(Vec::new()));
    }
    assert!(gateway.interest().is_empty());
}

#[test]
fn flow_header_swaps_the_endpoints_and_uses_the_current_window_and_ack() {
    let flow = manual_flow(1);
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
    let mut flow = manual_flow(1);
    flow.to_remote.extend(core::iter::repeat_n(0u8, 40000));
    let key = FlowKey {
        guest_port: 1,
        remote_ip: SERVER,
        remote_port: 2,
    };
    assert_eq!(flow.header(&key, ACK, 0, false).window, 65535 - 40000);
}

fn manual_flow(stream: StreamId) -> TcpFlow {
    TcpFlow::new(stream, GUEST_IP, MSS, 1000, 5000, 60000)
}

/// An open flow from `guest_port` with `len` bytes of remote data waiting.
fn flow_with_pending_data(
    gateway: &mut Gateway,
    remote: &mut FakeRemote,
    guest_port: u16,
    len: usize,
) {
    let mut guest = TcpGuest::new(guest_port, SERVER_PORT, GUEST_ISN);
    handshake(gateway, remote, &mut guest);
    gateway.remote_data(guest.stream, &vec![guest_port as u8; len]);
}

/// One poll with room for exactly `room` new frames; returns the guest ports served.
fn poll_with_room(gateway: &mut Gateway, room: usize) -> Vec<u16> {
    gateway.out.clear();
    for _ in 0..OUT_QUEUE_LIMIT - room {
        gateway.out.push_back(GuestFrame::plain(Vec::new()));
    }
    gateway.poll(NOW_MS);
    drain(gateway)
        .into_iter()
        .skip(OUT_QUEUE_LIMIT - room)
        .map(|frame| tcp_view(&frame).unwrap().dport)
        .collect()
}

#[test]
fn a_small_per_poll_budget_is_shared_round_robin_between_flows() {
    let (mut gateway, mut remote) = user_net_with_remote();
    for port in [1u16, 2, 3] {
        flow_with_pending_data(&mut gateway, &mut remote, port, 8000);
    }
    let mut served = Vec::new();
    for _ in 0..9 {
        let ports = poll_with_room(&mut gateway, 1);
        assert_eq!(ports.len(), 1);
        served.push(ports[0]);
    }
    assert_eq!(served, vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
}

#[test]
fn every_flow_makes_progress_under_a_budget_smaller_than_the_flow_count() {
    let (mut gateway, mut remote) = user_net_with_remote();
    let ports = [10u16, 20, 30, 40, 50];
    for port in ports {
        flow_with_pending_data(&mut gateway, &mut remote, port, 6000);
    }
    let mut sent = BTreeMap::new();
    for _ in 0..ports.len() * 2 {
        for port in poll_with_room(&mut gateway, 2) {
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
fn a_full_output_queue_holds_remote_data_back() {
    let (mut gateway, mut remote) = user_net_with_remote();
    flow_with_pending_data(&mut gateway, &mut remote, 7, 3000);
    assert!(
        poll_with_room(&mut gateway, 0).is_empty(),
        "no budget, no frames"
    );
    assert_eq!(
        poll_with_room(&mut gateway, 1),
        vec![7],
        "the data was kept"
    );
}

#[test]
fn service_order_is_sorted_and_rotates_by_one_each_poll() {
    let mut gateway = user_net();
    for port in 1u16..=3 {
        let key = FlowKey {
            guest_port: port,
            remote_ip: SERVER,
            remote_port: 80,
        };
        gateway.tcp.insert(key, manual_flow(port as StreamId));
    }
    let ports = |order: Vec<FlowKey>| order.iter().map(|k| k.guest_port).collect::<Vec<_>>();
    assert_eq!(ports(gateway.tcp.service_order()), vec![1, 2, 3]);
    assert_eq!(ports(gateway.tcp.service_order()), vec![2, 3, 1]);
    assert_eq!(ports(gateway.tcp.service_order()), vec![3, 1, 2]);
    assert_eq!(ports(gateway.tcp.service_order()), vec![1, 2, 3]);
}

#[test]
fn service_order_copes_with_flows_coming_and_going() {
    let mut gateway = user_net();
    let key = |port: u16| FlowKey {
        guest_port: port,
        remote_ip: SERVER,
        remote_port: 80,
    };
    let ports = |order: Vec<FlowKey>| order.iter().map(|k| k.guest_port).collect::<Vec<_>>();
    assert!(gateway.tcp.service_order().is_empty());
    gateway.tcp.insert(key(5), manual_flow(5));
    assert_eq!(ports(gateway.tcp.service_order()), vec![5]);
    assert_eq!(ports(gateway.tcp.service_order()), vec![5]);
    gateway.tcp.insert(key(9), manual_flow(9));
    gateway.tcp.insert(key(7), manual_flow(7));
    assert_eq!(
        ports(gateway.tcp.service_order()),
        vec![7, 9, 5],
        "continues after flow 5"
    );
    gateway.tcp.remove(&key(7));
    assert_eq!(
        ports(gateway.tcp.service_order()),
        vec![9, 5],
        "the remembered flow is gone"
    );
    gateway.tcp.remove(&key(9));
    gateway.tcp.remove(&key(5));
    assert!(gateway.tcp.service_order().is_empty());
}

#[test]
fn full_duplex_echo_survives_a_close_from_both_ends() {
    let (mut gateway, mut remote, mut guest) = established(GUEST_WINDOW);
    let payload = splitmix_bytes(77, 5000);
    guest.send(&mut gateway, ACK | PSH, &payload);
    let mut echoed = Vec::new();
    for _ in 0..MAX_ROUNDS {
        if echoed.len() >= payload.len() {
            break;
        }
        let written = remote.take_written(&mut gateway, guest.stream);
        gateway.remote_data(guest.stream, &written);
        gateway.poll(NOW_MS);
        echoed.extend(
            guest
                .receive(&mut gateway)
                .iter()
                .flat_map(|v| v.data.clone()),
        );
        guest.ack_all(&mut gateway);
    }
    assert_eq!(echoed, payload);
    guest.send(&mut gateway, FIN | ACK, &[]);
    gateway.poll(NOW_MS);
    remote.requests(&mut gateway);
    assert!(remote.is_write_shutdown(guest.stream));
    // The echo server sees end of stream and closes its side.
    gateway.remote_closed(guest.stream);
    gateway.poll(NOW_MS);
    assert!(
        guest
            .receive(&mut gateway)
            .iter()
            .any(|v| v.flags & FIN != 0)
    );
    guest.ack_all(&mut gateway);
    gateway.poll(NOW_MS);
    assert!(
        gateway.tcp.flows.is_empty(),
        "flow closed after both FINs are acked"
    );
    remote.requests(&mut gateway);
    assert!(remote.is_closed(guest.stream));
}

/// With a guest that takes TCP segments (`NetHandle::guest_takes_tcp_segments`) remote data goes
/// out in segments as large as the guest's window allows, checksummed only over the pseudo header
/// for the guest (or the device) to finish.
mod large_segments {
    use super::*;

    /// The data segments queued until `want` bytes arrived, acknowledging each batch.
    fn receive_segments(gateway: &mut Gateway, guest: &mut TcpGuest, want: usize) -> Vec<TcpView> {
        let mut segments = Vec::new();
        let mut received = 0;
        for _ in 0..MAX_ROUNDS {
            if received >= want {
                return segments;
            }
            gateway.poll(NOW_MS);
            for view in guest.receive(gateway) {
                if view.data.is_empty() {
                    continue;
                }
                received += view.data.len();
                segments.push(view);
            }
            guest.ack_all(gateway);
        }
        panic!("only {received} of {want} bytes arrived");
    }

    fn connected(window: u16) -> (Gateway, TcpGuest) {
        let (mut gateway, mut remote) = user_net_with_remote();
        gateway.set_large_segments(true);
        let mut guest = TcpGuest::new(GUEST_PORT, SERVER_PORT, GUEST_ISN);
        guest.window = window;
        handshake(&mut gateway, &mut remote, &mut guest);
        (gateway, guest)
    }

    #[test]
    fn remote_data_fills_the_guests_window_in_one_segment() {
        let (mut gateway, mut guest) = connected(u16::MAX);
        let data = splitmix_bytes(71, 60_000);
        gateway.remote_data(guest.stream, &data);
        let segments = receive_segments(&mut gateway, &mut guest, data.len());
        let received: Vec<u8> = segments.iter().flat_map(|v| v.data.clone()).collect();
        assert_eq!(received, data);
        assert_eq!(segments.len(), 1, "one segment for the whole window");
        assert!(segments.iter().all(|v| v.segment_size == Some(MSS as u16)));
    }

    #[test]
    fn a_large_segment_is_one_unfragmented_ipv4_packet_with_dont_fragment() {
        let (mut gateway, guest) = connected(u16::MAX);
        gateway.remote_data(guest.stream, &splitmix_bytes(72, 30_000));
        gateway.poll(NOW_MS);
        let frame = gateway.pop_for_guest().unwrap();
        assert!(frame.segment_size.is_some());
        let ip = &frame.bytes[ETH_HEADER_LEN..];
        assert_eq!(be_u16(&ip[IPV4_FLAGS_FRAGMENT]), IPV4_FLAG_DONT_FRAGMENT);
        assert_eq!(be_u16(&ip[IPV4_TOTAL_LEN]) as usize, ip.len());
        assert_eq!(reference_checksum(&ip[..IPV4_HEADER_LEN]), 0);
        assert!(ip.len() > ETHERNET_MTU, "{} bytes", ip.len());
    }

    #[test]
    fn segments_never_exceed_the_guests_window_or_the_ipv4_limit() {
        for window in [3000u16, 20_000, u16::MAX] {
            let (mut gateway, mut guest) = connected(window);
            let data = splitmix_bytes(73, 200_000);
            gateway.remote_data(guest.stream, &data);
            let segments = receive_segments(&mut gateway, &mut guest, data.len());
            for view in &segments {
                assert!(view.data.len() <= window as usize, "window {window}");
                assert!(view.data.len() <= LARGE_SEGMENT_DATA_MAX);
            }
            let received: Vec<u8> = segments.iter().flat_map(|v| v.data.clone()).collect();
            assert_eq!(received, data, "window {window}");
        }
    }

    #[test]
    fn the_checksum_field_holds_the_pseudo_header_sum() {
        let (mut gateway, guest) = connected(u16::MAX);
        gateway.remote_data(guest.stream, b"partial");
        gateway.poll(NOW_MS);
        let frame = gateway.pop_for_guest().unwrap();
        assert!(frame.segment_size.is_some());
        let tcp = &frame.bytes[TCP_START..];
        let pseudo = checksum_add(0, &SERVER)
            + checksum_add(0, &GUEST_IP)
            + IP_PROTO_TCP as u32
            + tcp.len() as u32;
        assert_eq!(be_u16(&tcp[TCP_CHECKSUM]), !checksum_finish(pseudo));
    }

    #[test]
    fn remote_data_split_across_pushes_goes_out_in_one_segment() {
        let (mut gateway, mut guest) = connected(u16::MAX);
        let data = splitmix_bytes(75, 30_000);
        for piece in data.chunks(7001) {
            gateway.remote_data(guest.stream, piece);
        }
        let segments = receive_segments(&mut gateway, &mut guest, data.len());
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].data, data);
    }

    #[test]
    fn without_large_segments_data_frames_stay_mss_sized_and_checksummed() {
        let (mut gateway, mut guest) = connected(u16::MAX);
        gateway.set_large_segments(false);
        gateway.remote_data(guest.stream, &splitmix_bytes(74, 10_000));
        let received = receive_data(&mut gateway, &mut guest, 10_000, MSS);
        assert_eq!(received.len(), 10_000);
        assert!(gateway.pop_for_guest().is_none());
    }
}

mod robustness {
    use super::*;

    /// A gateway whose driver refuses every open, the way usernet's tests pointed fuzzed traffic
    /// at a closed port.
    fn refuse_opens(gateway: &mut Gateway, remote: &mut FakeRemote) {
        for request in remote.requests(gateway) {
            if let GatewayRequest::Open { stream, .. } = request {
                gateway.failed(stream);
            }
        }
    }

    fn contained_gateway() -> (Gateway, FakeRemote) {
        let (mut gateway, remote) = user_net_with_remote();
        gateway.udp.limit = 8;
        (gateway, remote)
    }

    fn check_output(gateway: &mut Gateway) {
        for frame in drain(gateway) {
            assert!(frame.len() >= ETH_HEADER_LEN);
            match be_u16(&frame[ETH_TYPE]) {
                ETHERTYPE_IPV4 => {
                    assert_eq!(reference_checksum(&frame[ETH_HEADER_LEN..TCP_START]), 0);
                    let _ = tcp_view(&frame);
                }
                ETHERTYPE_ARP => assert_eq!(frame.len(), ARP_REQUEST_FRAME_LEN),
                other => panic!("unexpected ethertype {other:#x}"),
            }
        }
    }

    fn valid_frames() -> Vec<Vec<u8>> {
        let syn = TcpGuest::new(1000, 80, 5).segment(SYN, b"");
        let data = TcpGuest::new(1000, 80, 5).segment(ACK | PSH, b"payload");
        let mut discover = vec![0u8; DHCP_OPTIONS_OFFSET];
        discover[BOOTP_OP] = BOOTP_REQUEST;
        discover[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
        discover.extend_from_slice(&[DHCP_OPT_MESSAGE_TYPE, 1, DHCP_DISCOVER, DHCP_OPT_END]);
        vec![
            arp_request_frame(GUEST_MAC, GUEST_IP, GATEWAY_IP),
            guest_ip_frame(
                GATEWAY_IP,
                IP_PROTO_ICMP,
                &echo_request_payload(1, 1, b"ping"),
            ),
            udp_frame(1000, DNS_IP, DNS_PORT, b"query"),
            guest_ip_frame(DNS_IP, IP_PROTO_TCP, &syn),
            guest_ip_frame(DNS_IP, IP_PROTO_TCP, &data),
            udp_frame(68, LIMITED_BROADCAST_IP, DHCP_SERVER_PORT, &discover),
        ]
    }

    #[test]
    fn every_truncation_of_every_valid_frame_is_safe() {
        for frame in valid_frames() {
            for len in 0..=frame.len() {
                let (mut gateway, mut remote) = contained_gateway();
                gateway.from_guest(&frame[..len], NOW_MS);
                refuse_opens(&mut gateway, &mut remote);
                gateway.poll(NOW_MS);
                check_output(&mut gateway);
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
                let (mut gateway, mut remote) = contained_gateway();
                gateway.from_guest(&mutated, NOW_MS);
                refuse_opens(&mut gateway, &mut remote);
                gateway.poll(NOW_MS);
                check_output(&mut gateway);
            }
        }
    }

    #[test]
    fn random_frames_never_panic() {
        let mut rng = XorShift::new(FUZZ_SEED + 5);
        let (mut gateway, mut remote) = contained_gateway();
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
            // A plausible IPv4 header most of the time, so the transport parsers get reached.
            if len > ETH_HEADER_LEN && rng.below(4) != 0 {
                frame[ETH_HEADER_LEN] = IPV4_VERSION_IHL_NO_OPTIONS;
                let total = (len - ETH_HEADER_LEN) as u16;
                if len >= ETH_HEADER_LEN + IPV4_TOTAL_LEN.end {
                    frame[ETH_HEADER_LEN + IPV4_TOTAL_LEN.start
                        ..ETH_HEADER_LEN + IPV4_TOTAL_LEN.end]
                        .copy_from_slice(&total.to_be_bytes());
                }
                if len >= ETH_HEADER_LEN + IPV4_FLAGS_FRAGMENT.end {
                    frame[ETH_HEADER_LEN + IPV4_FLAGS_FRAGMENT.start
                        ..ETH_HEADER_LEN + IPV4_FLAGS_FRAGMENT.end]
                        .copy_from_slice(&IPV4_FLAG_DONT_FRAGMENT.to_be_bytes());
                }
                if len > ETH_HEADER_LEN + IPV4_PROTOCOL {
                    frame[ETH_HEADER_LEN + IPV4_PROTOCOL] =
                        [IP_PROTO_ICMP, IP_PROTO_TCP, IP_PROTO_UDP][rng.below(3)];
                }
            }
            gateway.from_guest(&frame, NOW_MS);
            if round % 50 == 0 {
                refuse_opens(&mut gateway, &mut remote);
                gateway.poll(NOW_MS);
                check_output(&mut gateway);
            }
        }
        refuse_opens(&mut gateway, &mut remote);
        gateway.poll(NOW_MS);
        check_output(&mut gateway);
    }

    #[test]
    fn random_segments_on_an_established_flow_never_panic_and_never_emit_bad_checksums() {
        let (mut gateway, mut remote, guest) = established(GUEST_WINDOW);
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
            gateway.from_guest(&guest_ip_frame(SERVER, IP_PROTO_TCP, &segment), NOW_MS);
            if round % 25 == 0 {
                remote.take_written(&mut gateway, guest.stream);
                refuse_opens(&mut gateway, &mut remote);
                gateway.poll(NOW_MS);
            }
            drain_tcp(&mut gateway);
        }
    }

    /// The driver end is input too: a Wisp server decides when data, closes and resets arrive.
    #[test]
    fn random_driver_events_never_panic_and_never_emit_bad_checksums() {
        let mut rng = XorShift::new(FUZZ_SEED + 7);
        let (mut gateway, mut remote) = user_net_with_remote();
        let mut guests: Vec<TcpGuest> = (0..4u16)
            .map(|index| TcpGuest::new(GUEST_PORT + index, SERVER_PORT, GUEST_ISN))
            .collect();
        for _ in 0..5000 {
            let stream = rng.below(8) as StreamId;
            match rng.below(12) {
                0 => gateway.opened(stream),
                1 => gateway.failed(stream),
                2 => {
                    let len = rng.below(3000);
                    gateway.remote_data(stream, &rng.bytes(len));
                }
                3 => gateway.remote_closed(stream),
                4 => gateway.reset(stream),
                5 => gateway.consumed(stream, rng.below(5000)),
                6 => {
                    let _ = (gateway.room(stream), gateway.to_remote(stream).len());
                    let _ = gateway.interest();
                }
                7 => gateway.poll(NOW_MS),
                8 => gateway.set_large_segments(rng.below(2) == 0),
                _ => {
                    let guest = &mut guests[rng.below(4)];
                    let flags = [SYN, ACK, ACK | PSH, FIN | ACK, RST][rng.below(5)];
                    // Acknowledge everything sent so far, so data keeps moving.
                    guest.ack = gateway
                        .tcp
                        .flows
                        .get(&guest.key())
                        .map_or(0, |flow| flow.snd_nxt);
                    let len = rng.below(2000);
                    let data = rng.bytes(len);
                    guest.send(&mut gateway, flags, &data);
                }
            }
            remote.requests(&mut gateway);
            drain_tcp(&mut gateway);
        }
    }
}
