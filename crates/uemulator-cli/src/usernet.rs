//! User-mode networking (SLIRP-style) behind a virtio-net device: the library's `Gateway` does the
//! protocol work (ARP, DHCP, ICMP echo, UDP and TCP NAT; see `uemulator::Gateway`), and this
//! drives it with ordinary host sockets. The guest sees 10.0.2.0/24 with the gateway at 10.0.2.2
//! (the host's loopback) and DNS at 10.0.2.3 (the host's first IPv4 nameserver).
//!
//! The gateway terminates TCP and never checks a checksum, so the NIC is told it may hand over the
//! guest's TSO segments as they are, and remote data goes back in segments as large as the guest's
//! window when its driver takes them (`run::offloads_for` declares this on the NIC).

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4, TcpStream, UdpSocket};
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use uemulator::{
    Endpoint, Gateway, GatewayConfig, GatewayPolicy, GatewayRequest, GuestFrame, NetHandle,
    StreamId, StreamKind,
};

/// Used when the host has no readable `/etc/resolv.conf` (always the case on Windows).
const FALLBACK_DNS: [u8; 4] = [1, 1, 1, 1];
const DNS_PORT: u16 = 53;
const TCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Where host socket reads land. A TCP read never asks for more than the gateway's `room`, which
/// the guest's unscaled 16-bit window bounds below this, and a UDP datagram is at most this long.
const READ_BUF_LEN: usize = 65536;
/// Longest wait for a guest frame in the worker loop where there is no `poll(2)`, which is also
/// how often host sockets are looked at there.
#[cfg(not(unix))]
const POLL_INTERVAL: Duration = Duration::from_millis(2);
/// The worker's wait when nothing is pending: only UDP flow expiry needs the time to pass.
#[cfg(unix)]
const IDLE_WAIT: Duration = Duration::from_millis(500);
/// The wait while frames are left over because the guest's receive queue was full.
#[cfg(unix)]
const BACKLOG_RETRY: Duration = Duration::from_millis(1);
/// Wake bytes carry no data; one read drains this many.
#[cfg(unix)]
const WAKE_DRAIN_LEN: usize = 64;

/// Where traffic for the DNS address goes: the host's first IPv4 nameserver.
fn system_dns() -> SocketAddrV4 {
    let conf = std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
    let address = first_ipv4_nameserver(&conf).unwrap_or(Ipv4Addr::from(FALLBACK_DNS));
    SocketAddrV4::new(address, DNS_PORT)
}

/// First `nameserver` line of a resolv.conf that holds an IPv4 address; IPv6 ones are skipped
/// because the guest network is IPv4 only.
fn first_ipv4_nameserver(conf: &str) -> Option<Ipv4Addr> {
    conf.lines()
        .filter_map(|line| line.strip_prefix("nameserver"))
        .filter_map(|addr| addr.trim().parse::<Ipv4Addr>().ok())
        .next()
}

fn socket_addr(endpoint: Endpoint) -> SocketAddr {
    SocketAddr::from((endpoint.address, endpoint.port))
}

/// What backs one gateway stream.
enum HostSocket {
    /// The host `connect` runs on a thread of its own; the result arrives through `connect_rx`.
    Connecting,
    Tcp(TcpStream),
    /// Connected to the NAT target, so only its replies are received.
    Udp(UdpSocket),
}

pub struct UserNet {
    gateway: Gateway,
    sockets: HashMap<StreamId, HostSocket>,
    read_buf: Vec<u8>,
    /// The gateway's clock starts here.
    started: Instant,
    connect_tx: Sender<(StreamId, io::Result<TcpStream>)>,
    connect_rx: Receiver<(StreamId, io::Result<TcpStream>)>,
    /// Ends the worker's wait when a connect thread has a result.
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl UserNet {
    pub fn new() -> Self {
        Self::with_dns(system_dns())
    }

    fn with_dns(dns: SocketAddrV4) -> Self {
        let (connect_tx, connect_rx) = channel();
        let isn_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.subsec_nanos());
        let dns = Endpoint {
            address: dns.ip().octets(),
            port: dns.port(),
        };
        UserNet {
            gateway: Gateway::new(GatewayConfig {
                policy: GatewayPolicy::UserNet { dns },
                isn_seed,
            }),
            sockets: HashMap::new(),
            read_buf: vec![0; READ_BUF_LEN],
            started: Instant::now(),
            connect_tx,
            connect_rx,
            wake: None,
        }
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// Takes one guest frame. Its effects on the host sockets wait for the next `poll`.
    pub fn input(&mut self, frame: &[u8]) {
        let now_ms = self.now_ms();
        self.gateway.from_guest(frame, now_ms);
    }

    pub fn set_large_segments(&mut self, on: bool) {
        self.gateway.set_large_segments(on);
    }

    /// Moves data between the host sockets and the gateway; call after every batch of `input`.
    /// Guest bytes are written to the sockets before the gateway's `poll`, which is what sends
    /// the ACKs for them: those then advertise the window the sockets left, not one shrunk by
    /// bytes about to be written (a TSO segment fills the whole window).
    pub fn poll(&mut self) {
        self.finish_connects();
        self.handle_requests();
        self.exchange();
        let now_ms = self.now_ms();
        self.gateway.poll(now_ms);
        // `poll` ends flows (Close) and half-closes them (ShutdownWrite); act on that now rather
        // than at the next wake.
        self.handle_requests();
    }

    fn finish_connects(&mut self) {
        while let Ok((stream, result)) = self.connect_rx.try_recv() {
            // A stream closed while connecting (a guest RST) is no longer `Connecting`.
            if !matches!(self.sockets.get(&stream), Some(HostSocket::Connecting)) {
                continue;
            }
            match result {
                Ok(tcp) if tcp.set_nonblocking(true).is_ok() => {
                    let _ = tcp.set_nodelay(true);
                    self.sockets.insert(stream, HostSocket::Tcp(tcp));
                    self.gateway.opened(stream);
                }
                // Refused, unreachable or timed out.
                _ => {
                    self.sockets.remove(&stream);
                    self.gateway.failed(stream);
                }
            }
        }
    }

    fn handle_requests(&mut self) {
        // Opening a UDP stream releases the datagrams that waited for it as new requests.
        loop {
            let requests = self.gateway.take_requests();
            if requests.is_empty() {
                return;
            }
            for request in requests {
                self.handle_request(request);
            }
        }
    }

    fn handle_request(&mut self, request: GatewayRequest) {
        match request {
            GatewayRequest::Open {
                stream,
                kind: StreamKind::Tcp,
                target,
            } => {
                self.sockets.insert(stream, HostSocket::Connecting);
                let connect_tx = self.connect_tx.clone();
                let wake = self.wake.clone();
                std::thread::spawn(move || {
                    let result =
                        TcpStream::connect_timeout(&socket_addr(target), TCP_CONNECT_TIMEOUT);
                    let _ = connect_tx.send((stream, result));
                    if let Some(wake) = wake {
                        wake();
                    }
                });
            }
            GatewayRequest::Open {
                stream,
                kind: StreamKind::Udp,
                target,
            } => match connected_udp_socket(target) {
                Ok(socket) => {
                    self.sockets.insert(stream, HostSocket::Udp(socket));
                    self.gateway.opened(stream);
                }
                Err(_) => self.gateway.failed(stream),
            },
            GatewayRequest::Datagram { stream, data } => {
                if let Some(HostSocket::Udp(socket)) = self.sockets.get(&stream) {
                    let _ = socket.send(&data);
                }
            }
            GatewayRequest::ShutdownWrite { stream } => {
                if let Some(HostSocket::Tcp(tcp)) = self.sockets.get(&stream) {
                    let _ = tcp.shutdown(Shutdown::Write);
                }
            }
            GatewayRequest::Close { stream } => {
                self.sockets.remove(&stream);
            }
        }
    }

    /// Writes guest bytes to every stream that has some and reads every stream the gateway can
    /// take data for, without blocking.
    fn exchange(&mut self) {
        for (stream, interest) in self.gateway.interest() {
            let broken = match self.sockets.get_mut(&stream) {
                Some(HostSocket::Tcp(tcp)) => {
                    (interest.write && write_to_remote(&mut self.gateway, stream, tcp).is_err())
                        || (interest.read
                            && read_tcp(&mut self.gateway, stream, tcp, &mut self.read_buf)
                                .is_err())
                }
                Some(HostSocket::Udp(socket)) => {
                    if interest.read {
                        // Errors (ICMP from the far end, mostly) are left for the guest to time
                        // out on, like a lost datagram.
                        while let Ok(len) = socket.recv(&mut self.read_buf) {
                            self.gateway.remote_data(stream, &self.read_buf[..len]);
                        }
                    }
                    false
                }
                Some(HostSocket::Connecting) | None => false,
            };
            if broken {
                // The gateway forgets the stream without a Close.
                self.gateway.reset(stream);
                self.sockets.remove(&stream);
            }
        }
    }

    /// The host sockets the worker's wait must watch, with `libc::poll` events: reads only where
    /// the gateway can take the data, writes only where guest data waits. Anything else would
    /// wake the wait with nothing to do.
    #[cfg(unix)]
    fn poll_interest(&self) -> Vec<(RawFd, libc::c_short)> {
        self.gateway
            .interest()
            .into_iter()
            .filter_map(|(stream, interest)| {
                let fd = match self.sockets.get(&stream)? {
                    HostSocket::Tcp(tcp) => tcp.as_raw_fd(),
                    HostSocket::Udp(socket) => socket.as_raw_fd(),
                    HostSocket::Connecting => return None,
                };
                let mut events = 0;
                if interest.read {
                    events |= libc::POLLIN;
                }
                if interest.write {
                    events |= libc::POLLOUT;
                }
                Some((fd, events))
            })
            .collect()
    }

    pub fn pop_frame(&mut self) -> Option<GuestFrame> {
        self.gateway.pop_for_guest()
    }

    /// Puts a frame the guest could not take yet back at the head, keeping frame order.
    pub fn requeue_frame(&mut self, frame: GuestFrame) {
        self.gateway.requeue_for_guest(frame);
    }
}

fn connected_udp_socket(target: Endpoint) -> io::Result<UdpSocket> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.connect(socket_addr(target))?;
    socket.set_nonblocking(true)?;
    Ok(socket)
}

/// Writes as much of the stream's waiting guest data as the socket takes without blocking. Err
/// when the socket failed and the stream is over.
fn write_to_remote(gateway: &mut Gateway, stream: StreamId, tcp: &mut TcpStream) -> Result<(), ()> {
    loop {
        let waiting = gateway.to_remote(stream);
        if waiting.is_empty() {
            return Ok(());
        }
        match tcp.write(waiting) {
            Ok(0) => return Ok(()),
            Ok(written) => gateway.consumed(stream, written),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Err(()),
        }
    }
}

/// Reads what the guest's window has room for. EOF is the remote's orderly close; Err when the
/// socket failed and the stream is over.
fn read_tcp(
    gateway: &mut Gateway,
    stream: StreamId,
    tcp: &mut TcpStream,
    read_buf: &mut [u8],
) -> Result<(), ()> {
    loop {
        // `room` shrinks with every `remote_data`, so this stops at the window.
        let want = gateway.room(stream).min(read_buf.len());
        if want == 0 {
            return Ok(());
        }
        match tcp.read(&mut read_buf[..want]) {
            Ok(0) => {
                gateway.remote_closed(stream);
                return Ok(());
            }
            Ok(len) => gateway.remote_data(stream, &read_buf[..len]),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Err(()),
        }
    }
}

/// Moves frames from `usernet` to the guest until the guest's queue is full. Returns whether
/// frames are left over.
fn deliver(usernet: &mut UserNet, net: &NetHandle) -> bool {
    while let Some(frame) = usernet.pop_frame() {
        let queued = match frame.segment_size {
            Some(size) => net.send_tcp_segment(&frame.bytes, size),
            None => net.send_frame(&frame.bytes),
        };
        if !queued {
            usernet.requeue_frame(frame);
            return true;
        }
    }
    false
}

/// The worker waits in `poll(2)` on the host sockets and on one end of a socket pair that guest
/// frames and finished connects write to, so a reply from the host reaches the guest as soon as
/// it arrives. A timed wait would add its whole period to every round trip, and with TCP's
/// fixed 64 KiB window that period is what bounds throughput.
#[cfg(unix)]
pub fn spawn(net: NetHandle) {
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicBool, Ordering};

    let (wake_rx, wake_tx) = UnixStream::pair().expect("socketpair");
    wake_rx
        .set_nonblocking(true)
        .expect("nonblocking socketpair");
    wake_tx
        .set_nonblocking(true)
        .expect("nonblocking socketpair");
    let wake_tx = Arc::new(wake_tx);
    // Set while the worker is about to wait or waiting: a guest frame writes the wake byte only
    // then, so a busy transfer does not cost the vCPU a syscall per frame. The worker sets it
    // before its last look at the queue and the guest side queues before it reads it, so one of
    // the two always sees the other.
    let waiting = Arc::new(AtomicBool::new(false));
    {
        let (wake_tx, waiting) = (wake_tx.clone(), waiting.clone());
        net.on_guest_frame(move || {
            if waiting.swap(false, Ordering::SeqCst) {
                let _ = (&*wake_tx).write(&[0]);
            }
        });
    }
    std::thread::spawn(move || {
        let mut usernet = UserNet::new();
        usernet.wake = Some(Arc::new(move || {
            let _ = (&*wake_tx).write(&[0]);
        }));
        let mut drain = [0u8; WAKE_DRAIN_LEN];
        loop {
            while matches!((&wake_rx).read(&mut drain), Ok(n) if n > 0) {}
            while let Some(frame) = net.try_recv_frame() {
                usernet.input(&frame);
            }
            usernet.set_large_segments(net.guest_takes_tcp_segments());
            usernet.poll();
            let backlog = deliver(&mut usernet, &net);
            waiting.store(true, Ordering::SeqCst);
            if let Some(frame) = net.try_recv_frame() {
                waiting.store(false, Ordering::SeqCst);
                usernet.input(&frame);
                continue;
            }
            // The guest's queue has no "room again" signal, so a backlog is retried on a short
            // timer.
            let timeout = if backlog { BACKLOG_RETRY } else { IDLE_WAIT };
            let mut fds: Vec<libc::pollfd> = std::iter::once((wake_rx.as_raw_fd(), libc::POLLIN))
                .chain(usernet.poll_interest())
                .map(|(fd, events)| libc::pollfd {
                    fd,
                    events,
                    revents: 0,
                })
                .collect();
            // SAFETY: `fds` is a live, correctly sized array of `pollfd` for the whole call.
            unsafe {
                libc::poll(
                    fds.as_mut_ptr(),
                    fds.len() as libc::nfds_t,
                    timeout.as_millis() as libc::c_int,
                );
            }
            waiting.store(false, Ordering::SeqCst);
        }
    });
}

/// Without `poll(2)` the worker waits for guest frames only, and looks at the host sockets every
/// `POLL_INTERVAL`.
#[cfg(not(unix))]
pub fn spawn(net: NetHandle) {
    std::thread::spawn(move || {
        let mut usernet = UserNet::new();
        loop {
            if let Some(frame) = net.recv_frame(POLL_INTERVAL) {
                usernet.input(&frame);
                while let Some(frame) = net.try_recv_frame() {
                    usernet.input(&frame);
                }
            }
            usernet.set_large_segments(net.guest_takes_tcp_segments());
            usernet.poll();
            deliver(&mut usernet, &net);
        }
    });
}

#[cfg(test)]
mod tests;
