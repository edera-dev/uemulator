use super::guestmem::GuestMem;
use super::virtio::{Chain, DEVICE_TYPE_NET, Queue, VirtioDev};
use crate::netdev::{HostFrame, NetHandle};
#[cfg(freestanding)]
use crate::prelude::*;

/// Feature bits (virtio 1.x section 5.1.3). The offloads come in the pairs the spec ties
/// together: each TSO bit requires its CSUM bit, and the guest's TSO is offered with merged
/// receive buffers so a 64 KiB frame does not need 64 KiB buffers posted for every ACK.
/// `GUEST_ECN`, `HOST_ECN`, UFO, TSO6 and `CTRL_GUEST_OFFLOADS` are never offered, so the
/// guest can neither send nor be sent those segment types.
const F_CSUM: u64 = 1 << 0;
const F_GUEST_CSUM: u64 = 1 << 1;
const F_MAC: u64 = 1 << 5;
const F_GUEST_TSO4: u64 = 1 << 7;
const F_HOST_TSO4: u64 = 1 << 11;
const F_MRG_RXBUF: u64 = 1 << 15;
const F_STATUS: u64 = 1 << 16;
/// Offered when the host takes the guest's segments as they are (`takes_guest_segments`).
const GUEST_TO_HOST_OFFLOADS: u64 = F_CSUM | F_HOST_TSO4;
/// Offered when the host sends large segments (`sends_large_segments`).
const HOST_TO_GUEST_OFFLOADS: u64 = F_GUEST_CSUM | F_GUEST_TSO4 | F_MRG_RXBUF;

const RECEIVEQ: usize = 0;
const TRANSMITQ: usize = 1;
const QUEUE_COUNT: usize = 2;

/// `struct virtio_net_hdr_v1`. VERSION_1 is always negotiated, which makes the header this long
/// (with `num_buffers`) even without MRG_RXBUF. All fields little endian.
const HEADER_LEN: usize = 12;
const HEADER_FLAGS: usize = 0;
const HEADER_GSO_TYPE: usize = 1;
const HEADER_HDR_LEN: usize = 2;
const HEADER_GSO_SIZE: usize = 4;
const HEADER_CSUM_START: usize = 6;
const HEADER_CSUM_OFFSET: usize = 8;
const HEADER_NUM_BUFFERS_OFFSET: usize = 10;
/// The checksum from `csum_start` to the end of the frame goes `csum_offset` bytes past it.
const HDR_F_NEEDS_CSUM: u8 = 1;
const GSO_NONE: u8 = 0;
const GSO_TCPV4: u8 = 1;
/// Both directions' checksums are 16 bits.
const CHECKSUM_LEN: usize = 2;

/// Device config: the MAC at 0..6, then le16 status.
const MAC_LEN: usize = 6;
const CONFIG_STATUS_OFFSET: usize = 6;
/// VIRTIO_NET_S_LINK_UP; the link is always up.
const STATUS_LINK_UP: u8 = 1;

const ETHERNET_HEADER_LEN: usize = 14;
const ETHERNET_TYPE: usize = 12;
const ETHERTYPE_IPV4: u16 = 0x0800;
/// An 802.1Q tag the guest may put between the addresses and the type.
const VLAN_TAG_LEN: usize = 4;
const IPV4_MAX_TOTAL_LEN: usize = u16::MAX as usize;
const IPV4_MIN_HEADER_LEN: usize = 20;
const IPV4_VERSION: u8 = 4;
const IPV4_TOTAL_LEN: usize = 2;
const IPV4_PROTOCOL: usize = 9;
const IP_PROTO_TCP: u8 = 6;
const TCP_MIN_HEADER_LEN: usize = 20;
/// High nibble: header length in 32-bit words.
const TCP_DATA_OFFSET: usize = 12;
const TCP_CHECKSUM: usize = 16;

/// Largest frame forwarded from the guest transmit queue (excluding the header): a whole TSO
/// segment, which is as long as IPv4 allows, behind a VLAN-tagged Ethernet header. A longer one
/// is dropped whole rather than sent truncated.
const MAX_FRAME_LEN: usize = ETHERNET_HEADER_LEN + VLAN_TAG_LEN + IPV4_MAX_TOTAL_LEN;

/// The fields of `struct virtio_net_hdr_v1` this device reads or writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NetHeader {
    flags: u8,
    gso_type: u8,
    hdr_len: u16,
    gso_size: u16,
    csum_start: u16,
    csum_offset: u16,
    num_buffers: u16,
}

impl NetHeader {
    fn parse(bytes: &[u8; HEADER_LEN]) -> NetHeader {
        let field = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        NetHeader {
            flags: bytes[HEADER_FLAGS],
            gso_type: bytes[HEADER_GSO_TYPE],
            hdr_len: field(HEADER_HDR_LEN),
            gso_size: field(HEADER_GSO_SIZE),
            csum_start: field(HEADER_CSUM_START),
            csum_offset: field(HEADER_CSUM_OFFSET),
            num_buffers: field(HEADER_NUM_BUFFERS_OFFSET),
        }
    }

    fn to_bytes(self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        bytes[HEADER_FLAGS] = self.flags;
        bytes[HEADER_GSO_TYPE] = self.gso_type;
        for (at, value) in [
            (HEADER_HDR_LEN, self.hdr_len),
            (HEADER_GSO_SIZE, self.gso_size),
            (HEADER_CSUM_START, self.csum_start),
            (HEADER_CSUM_OFFSET, self.csum_offset),
            (HEADER_NUM_BUFFERS_OFFSET, self.num_buffers),
        ] {
            bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Whether a frame of `frame_len` bytes the guest sent with this header may go to the host.
    /// The host only ever sees the frame, so the header has to promise no more than the host
    /// declared it takes: a partial checksum only with CSUM, a segment only with HOST_TSO4. Both
    /// are checked against the negotiated features, not the offered ones. With neither
    /// negotiated the header means nothing (a driver may leave garbage in it) and is ignored.
    fn transmit_allowed(self, frame_len: usize, negotiated: u64) -> bool {
        if negotiated & F_CSUM == 0 {
            return true;
        }
        let needs_checksum = self.flags & HDR_F_NEEDS_CSUM != 0;
        if needs_checksum
            && self.csum_start as usize + self.csum_offset as usize + CHECKSUM_LEN > frame_len
        {
            return false;
        }
        match self.gso_type {
            GSO_NONE => true,
            // The spec has the driver set NEEDS_CSUM on every segment (5.1.6.2.1); without it the
            // segment would carry a checksum that is neither finished nor marked as unfinished.
            GSO_TCPV4 => negotiated & F_HOST_TSO4 != 0 && self.gso_size > 0 && needs_checksum,
            _ => false,
        }
    }
}

/// Byte offsets of a TCP/IPv4 frame's checksummed span: from the TCP header (`tcp_start`) to the
/// IPv4 total length (`end`), with the TCP payload starting at `payload_start`.
struct TcpLayout {
    tcp_start: usize,
    payload_start: usize,
    end: usize,
}

/// `None` unless `frame` is Ethernet (untagged) carrying an IPv4 packet whose TCP header lies
/// wholly inside the frame.
fn tcp_layout(frame: &[u8]) -> Option<TcpLayout> {
    let ip = frame.get(ETHERNET_HEADER_LEN..)?;
    if frame.get(ETHERNET_TYPE..ETHERNET_TYPE + 2)? != ETHERTYPE_IPV4.to_be_bytes()
        || ip.len() < IPV4_MIN_HEADER_LEN
        || ip[0] >> 4 != IPV4_VERSION
        || ip[IPV4_PROTOCOL] != IP_PROTO_TCP
    {
        return None;
    }
    let ip_header_len = (ip[0] & 0xf) as usize * 4;
    let total_len = u16::from_be_bytes([ip[IPV4_TOTAL_LEN], ip[IPV4_TOTAL_LEN + 1]]) as usize;
    if ip_header_len < IPV4_MIN_HEADER_LEN
        || total_len > ip.len()
        || total_len < ip_header_len + TCP_MIN_HEADER_LEN
    {
        return None;
    }
    let tcp_header_len = (ip[ip_header_len + TCP_DATA_OFFSET] >> 4) as usize * 4;
    if tcp_header_len < TCP_MIN_HEADER_LEN || ip_header_len + tcp_header_len > total_len {
        return None;
    }
    let tcp_start = ETHERNET_HEADER_LEN + ip_header_len;
    Some(TcpLayout {
        tcp_start,
        payload_start: tcp_start + tcp_header_len,
        end: ETHERNET_HEADER_LEN + total_len,
    })
}

/// Ones-complement sum of `data` as big-endian 16-bit words, odd byte padded with zero, added to
/// `sum` and folded to 16 bits (RFC 1071). Sums 32-bit words into a 64-bit accumulator, which
/// cannot overflow for any frame, and folds once at the end.
fn ones_complement_sum(sum: u16, data: &[u8]) -> u16 {
    let mut acc = sum as u64;
    let (words, rest) = data.as_chunks::<4>();
    for word in words {
        acc += u32::from_be_bytes(*word) as u64;
    }
    let mut tail = [0u8; 4];
    tail[..rest.len()].copy_from_slice(rest);
    acc += u32::from_be_bytes(tail) as u64;
    while acc >> 16 != 0 {
        acc = (acc & 0xffff) + (acc >> 16);
    }
    acc as u16
}

pub(crate) struct VirtioNet {
    mac: [u8; 6],
    net: NetHandle,
    /// What the driver accepted; zero until FEATURES_OK and after a reset.
    negotiated: u64,
}

/// Outcome of offering one frame to the receive queue.
enum Delivery {
    Delivered,
    /// Not enough buffers posted right now; the frame waits for more.
    NoRoom,
    /// The frame can never be delivered (larger than every buffer the queue could hold); the
    /// buffers stay posted.
    Dropped,
}

impl VirtioNet {
    pub fn new(mac: [u8; 6], net: NetHandle) -> Self {
        // A reboot builds a new device around the handle the old one negotiated on.
        net.set_guest_takes_tcp_segments(false);
        VirtioNet {
            mac,
            net,
            negotiated: 0,
        }
    }

    /// The header for a frame from the host, after finishing its checksum in place when the
    /// guest cannot. `None` drops the frame. Leaves a frame the guest has to finish marked as a
    /// segment, so a frame put back for lack of buffers gets the same header next time.
    fn receive_header(&self, frame: &mut HostFrame) -> Option<NetHeader> {
        let Some(segment_size) = frame.tcp_segment_size else {
            return Some(NetHeader::default());
        };
        let layout = tcp_layout(&frame.bytes)?;
        let oversize = layout.end - layout.payload_start > segment_size as usize;
        if self.negotiated & F_GUEST_CSUM == 0 {
            if oversize {
                return None;
            }
            let checksum_at = layout.tcp_start + TCP_CHECKSUM;
            let pseudo_header_sum =
                u16::from_be_bytes([frame.bytes[checksum_at], frame.bytes[checksum_at + 1]]);
            frame.bytes[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
            let sum = ones_complement_sum(
                pseudo_header_sum,
                &frame.bytes[layout.tcp_start..layout.end],
            );
            frame.bytes[checksum_at..checksum_at + CHECKSUM_LEN]
                .copy_from_slice(&(!sum).to_be_bytes());
            frame.tcp_segment_size = None;
            return Some(NetHeader::default());
        }
        let mut header = NetHeader {
            flags: HDR_F_NEEDS_CSUM,
            csum_start: layout.tcp_start as u16,
            csum_offset: TCP_CHECKSUM as u16,
            ..NetHeader::default()
        };
        // Linux takes a "segment" no longer than gso_size as an ordinary packet anyway; marking
        // only the long ones keeps the short ones off its GSO path.
        if oversize {
            if self.negotiated & F_GUEST_TSO4 == 0 {
                return None;
            }
            header.gso_type = GSO_TCPV4;
            header.gso_size = segment_size;
            header.hdr_len = layout.payload_start as u16;
        }
        Some(header)
    }

    /// Writes `header` and `frame` into posted receive buffers: one chain, or with MRG_RXBUF as
    /// many as it takes (QEMU's virtio_net_receive_rcu does the same). Nothing is written and no
    /// chain is consumed unless the whole frame fits.
    fn deliver(
        &self,
        mut header: NetHeader,
        frame: &[u8],
        queue: &mut Queue,
        mem: &mut GuestMem,
    ) -> Delivery {
        let needed = HEADER_LEN + frame.len();
        let first_avail = queue.last_avail;
        let mut chains: Vec<Chain> = Vec::new();
        let mut room = 0;
        while room < needed {
            let Some(chain) = queue.pop(mem) else {
                queue.rewind_to(first_avail);
                return Delivery::NoRoom;
            };
            room += chain.writable();
            // The header goes in the first buffer whole; Linux reads it from there.
            let header_fits = !chains.is_empty() || chain.writable() >= HEADER_LEN;
            chains.push(chain);
            let more_allowed =
                self.negotiated & F_MRG_RXBUF != 0 && chains.len() < queue.size as usize;
            if !header_fits || (room < needed && !more_allowed) {
                queue.rewind_to(first_avail);
                return Delivery::Dropped;
            }
        }
        header.num_buffers = chains.len() as u16;
        let mut used = Vec::with_capacity(chains.len());
        let mut rest = frame;
        for (index, chain) in chains.iter().enumerate() {
            let mut written = 0;
            if index == 0 {
                written = chain.write_at(mem, 0, &header.to_bytes());
            }
            let take = (chain.writable() - written).min(rest.len());
            written += chain.write_at(mem, written, &rest[..take]);
            rest = &rest[take..];
            used.push((chain.head, written as u32));
        }
        queue.push_used_all(mem, &used);
        Delivery::Delivered
    }
}

impl VirtioDev for VirtioNet {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_NET
    }

    fn features(&self) -> u64 {
        let offloads = self.net.offloads();
        let mut features = F_MAC | F_STATUS;
        if offloads.takes_guest_segments {
            features |= GUEST_TO_HOST_OFFLOADS;
        }
        if offloads.sends_large_segments {
            features |= HOST_TO_GUEST_OFFLOADS;
        }
        features
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, off: usize) -> u8 {
        match off {
            0..MAC_LEN => self.mac[off],
            CONFIG_STATUS_OFFSET => STATUS_LINK_UP,
            _ => 0,
        }
    }

    fn features_negotiated(&mut self, features: u64) {
        self.negotiated = features;
        self.net.set_guest_takes_tcp_segments(
            features & (F_GUEST_CSUM | F_GUEST_TSO4) == F_GUEST_CSUM | F_GUEST_TSO4,
        );
    }

    fn reset(&mut self) {
        self.negotiated = 0;
        self.net.set_guest_takes_tcp_segments(false);
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if queue != TRANSMITQ {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[TRANSMITQ].pop(mem) {
            let readable = chain.readable();
            if readable > HEADER_LEN && readable - HEADER_LEN <= MAX_FRAME_LEN {
                let mut header = [0u8; HEADER_LEN];
                chain.read_from(mem, 0, &mut header);
                let mut frame = vec![0u8; readable - HEADER_LEN];
                chain.read_from(mem, HEADER_LEN, &mut frame);
                if NetHeader::parse(&header).transmit_allowed(frame.len(), self.negotiated) {
                    self.net.guest_send(frame);
                }
            }
            queues[TRANSMITQ].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        let mut any = false;
        while let Some(mut frame) = self.net.guest_recv() {
            let Some(header) = self.receive_header(&mut frame) else {
                continue;
            };
            match self.deliver(header, &frame.bytes, &mut queues[RECEIVEQ], mem) {
                Delivery::Delivered => any = true,
                // Like QEMU's virtio_net_receive_rcu, drop a frame that cannot fit and leave the
                // buffers posted for the next one; truncating would hand the guest a corrupt
                // packet.
                Delivery::Dropped => {}
                Delivery::NoRoom => {
                    self.net.guest_unrecv(frame);
                    break;
                }
            }
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::super::virtio::testutil::*;
    use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
    use super::*;
    use crate::dev::guestmem::test_mem;
    use crate::netdev::NetOffloads;

    /// Large enough for a 64 KiB frame in page-sized merged buffers.
    const QUEUE_SIZE: u16 = 32;
    const RX_DESC: u64 = 0x1000;
    const RX_AVAIL: u64 = 0x2000;
    const RX_USED: u64 = 0x3000;
    const TX_DESC: u64 = 0x8000;
    const TX_AVAIL: u64 = 0x9000;
    const TX_USED: u64 = 0xa000;
    const TX_BUF: u64 = 0xc000;
    const RX_BUF: u64 = 0xd000;
    /// Merged receive buffers, `PAGE` apart.
    const BIG_RX: u64 = 0x2_0000;
    /// A whole TSO segment for the transmit queue.
    const BIG_TX: u64 = 0x5_0000;
    const PAGE: u32 = 0x1000;
    const RAM_BYTES: usize = 0x8_0000;
    const MAC: [u8; MAC_LEN] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

    struct Rig {
        net: NetHandle,
        dev: VirtioNet,
        queues: Vec<Queue>,
        ram: Vec<u8>,
    }

    impl Rig {
        /// A NIC whose host side declared `offloads` and whose driver accepted `negotiated`.
        fn negotiated(offloads: NetOffloads, negotiated: u64) -> Rig {
            let mut rig = Rig::new();
            rig.net.set_offloads(offloads);
            rig.dev.features_negotiated(negotiated);
            rig
        }

        /// Posts `count` single-page receive buffers, heads from `first_head` on.
        fn post_pages(&mut self, first_head: u16, count: u16) {
            for head in first_head..first_head + count {
                self.post_receive(head, BIG_RX + PAGE as u64 * head as u64, PAGE);
            }
        }

        fn new() -> Rig {
            let net = NetHandle::default();
            let mut ram = vec![0u8; RAM_BYTES];
            let queues = vec![
                setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
                setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
            ];
            Rig {
                dev: VirtioNet::new(MAC, net.clone()),
                net,
                queues,
                ram,
            }
        }

        fn kick(&mut self, queue: usize) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.kick(queue, &mut self.queues, &mut mem)
        }

        fn poll(&mut self) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.poll(&mut self.queues, &mut mem)
        }

        fn post_receive(&mut self, head: u16, addr: u64, len: u32) {
            put_desc(&mut self.ram, RX_DESC, head, addr, len, DESC_F_WRITE, 0);
            post(&mut self.ram, RX_AVAIL, QUEUE_SIZE, head);
        }
    }

    #[test]
    fn frame_larger_than_receive_buffer_is_dropped_not_truncated() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 20);
        assert!(rig.net.send_frame(&[0xab; 60]));
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
        assert!(!rig.net.has_pending(), "the oversize frame is consumed");
    }

    fn put_tx_frame(rig: &mut Rig, head: u16, header: &[u8], frame: &[u8]) {
        let addr = TX_BUF + 0x1000 * head as u64;
        let at = addr as usize;
        rig.ram[at..at + header.len()].copy_from_slice(header);
        rig.ram[at + header.len()..at + header.len() + frame.len()].copy_from_slice(frame);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            head,
            addr,
            (header.len() + frame.len()) as u32,
            0,
            0,
        );
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, head);
    }

    fn frame_of(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(3).wrapping_add(seed))
            .collect()
    }

    fn received(rig: &Rig, buffer: u64, len: usize) -> Vec<u8> {
        rig.ram[buffer as usize..buffer as usize + len].to_vec()
    }

    #[test]
    fn device_identity_features_and_queues() {
        let rig = Rig::new();
        assert_eq!(rig.dev.device_type(), 1);
        assert_eq!(
            rig.dev.features(),
            (1 << 5) | (1 << 16),
            "MAC and STATUS only"
        );
        assert_eq!(rig.dev.num_queues(), 2);
        assert_eq!(HEADER_LEN, 12);
    }

    #[test]
    fn config_space_holds_the_mac_then_link_up_status() {
        let rig = Rig::new();
        let config: Vec<u8> = (0..12).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(&config[..6], &MAC);
        assert_eq!(
            u16::from_le_bytes([config[6], config[7]]),
            1,
            "link up, high status byte clear"
        );
        assert_eq!(&config[8..], &[0; 4]);
        for off in [12, 64, usize::MAX] {
            assert_eq!(rig.dev.config_read(off), 0);
        }
    }

    #[test]
    fn config_writes_do_not_change_the_mac_or_status() {
        let mut rig = Rig::new();
        for off in 0..8 {
            rig.dev.config_write(off, 0xee);
        }
        let config: Vec<u8> = (0..7).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(&config[..6], &MAC);
        assert_eq!(config[6], 1);
    }

    #[test]
    fn transmitted_frame_loses_its_virtio_header() {
        let mut rig = Rig::new();
        let frame = frame_of(60, 0x10);
        put_tx_frame(&mut rig, 0, &[0xee; HEADER_LEN], &frame);
        assert!(rig.kick(TRANSMITQ));
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_entry(&rig.ram, TX_USED, 0), (0, 0));
    }

    /// Without CSUM negotiated no field of the header promises anything, whatever a driver
    /// leaves in it.
    #[test]
    fn the_header_contents_do_not_matter_on_transmit_without_offloads() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0u8; HEADER_LEN], &[1, 2, 3, 4]);
        put_tx_frame(&mut rig, 1, &[0xff; HEADER_LEN], &[5, 6, 7, 8]);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![5, 6, 7, 8]);
    }

    #[test]
    fn frames_go_out_in_posting_order() {
        let mut rig = Rig::new();
        for head in 0..4u16 {
            put_tx_frame(
                &mut rig,
                head,
                &[0; HEADER_LEN],
                &frame_of(20 + head as usize, head as u8),
            );
        }
        assert!(rig.kick(TRANSMITQ));
        for head in 0..4u16 {
            assert_eq!(
                rig.net.try_recv_frame().unwrap(),
                frame_of(20 + head as usize, head as u8)
            );
        }
        assert_eq!(used_idx(&rig.ram, TX_USED), 4);
    }

    #[test]
    fn header_and_frame_in_separate_descriptors_are_handled() {
        let mut rig = Rig::new();
        let frame = frame_of(100, 0x40);
        rig.ram[TX_BUF as usize..TX_BUF as usize + HEADER_LEN].fill(0);
        rig.ram[TX_BUF as usize + 0x100..TX_BUF as usize + 0x100 + 100].copy_from_slice(&frame);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            0,
            TX_BUF,
            HEADER_LEN as u32,
            DESC_F_NEXT,
            1,
        );
        put_desc(&mut rig.ram, TX_DESC, 1, TX_BUF + 0x100, 100, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn header_split_across_descriptors_is_still_skipped_correctly() {
        let mut rig = Rig::new();
        let frame = frame_of(30, 0x70);
        rig.ram[TX_BUF as usize..TX_BUF as usize + 5].fill(0xaa);
        rig.ram[TX_BUF as usize + 0x100..TX_BUF as usize + 0x100 + 7].fill(0xbb);
        rig.ram[TX_BUF as usize + 0x200..TX_BUF as usize + 0x200 + 30].copy_from_slice(&frame);
        put_desc(&mut rig.ram, TX_DESC, 0, TX_BUF, 5, DESC_F_NEXT, 1);
        put_desc(&mut rig.ram, TX_DESC, 1, TX_BUF + 0x100, 7, DESC_F_NEXT, 2);
        put_desc(&mut rig.ram, TX_DESC, 2, TX_BUF + 0x200, 30, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn chain_with_only_a_header_or_less_sends_nothing_but_is_completed() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[]);
        put_tx_frame(&mut rig, 1, &[0; 5], &[]);
        put_desc(&mut rig.ram, TX_DESC, 2, TX_BUF, 0, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 2);
        assert!(rig.kick(TRANSMITQ));
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_idx(&rig.ram, TX_USED), 3);
    }

    #[test]
    fn smallest_frame_is_one_byte_after_the_header() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[0x42]);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![0x42]);
    }

    #[test]
    fn frame_of_exactly_the_maximum_length_is_sent() {
        let mut ram = vec![0u8; 1 << 20];
        let net = NetHandle::default();
        let mut dev = VirtioNet::new(MAC, net.clone());
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let frame = frame_of(MAX_FRAME_LEN, 1);
        let at = 0x1_0000usize;
        ram[at + HEADER_LEN..at + HEADER_LEN + MAX_FRAME_LEN].copy_from_slice(&frame);
        put_desc(
            &mut ram,
            TX_DESC,
            0,
            at as u64,
            (HEADER_LEN + MAX_FRAME_LEN) as u32,
            0,
            0,
        );
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut queues, &mut mem));
        assert_eq!(net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn frame_longer_than_the_maximum_is_dropped_not_truncated() {
        let mut ram = vec![0u8; 1 << 20];
        let net = NetHandle::default();
        let mut dev = VirtioNet::new(MAC, net.clone());
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let at = 0x1_0000u64;
        put_desc(
            &mut ram,
            TX_DESC,
            0,
            at,
            (HEADER_LEN + MAX_FRAME_LEN + 1) as u32,
            0,
            0,
        );
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        assert!(
            dev.kick(TRANSMITQ, &mut queues, &mut mem),
            "the buffer is still returned"
        );
        assert!(net.try_recv_frame().is_none());
    }

    #[test]
    fn writable_segments_of_a_transmit_chain_are_not_part_of_the_frame() {
        let mut rig = Rig::new();
        let frame = frame_of(10, 5);
        rig.ram[TX_BUF as usize + HEADER_LEN..TX_BUF as usize + HEADER_LEN + 10]
            .copy_from_slice(&frame);
        rig.ram[TX_BUF as usize + 0x200..TX_BUF as usize + 0x210].fill(0xcc);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            0,
            TX_BUF,
            (HEADER_LEN + 10) as u32,
            DESC_F_NEXT,
            1,
        );
        put_desc(
            &mut rig.ram,
            TX_DESC,
            1,
            TX_BUF + 0x200,
            16,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.net.try_recv_frame().unwrap(), frame);
    }

    #[test]
    fn kick_on_the_receive_queue_sends_nothing() {
        let mut rig = Rig::new();
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[1, 2, 3]);
        assert!(!rig.kick(RECEIVEQ));
        assert!(rig.net.try_recv_frame().is_none());
        assert_eq!(used_idx(&rig.ram, TX_USED), 0);
    }

    #[test]
    fn transmit_kick_with_nothing_posted_reports_no_work() {
        let mut rig = Rig::new();
        assert!(!rig.kick(TRANSMITQ));
    }

    #[test]
    fn received_frame_gets_a_zero_header_with_one_buffer_after_it() {
        let mut rig = Rig::new();
        rig.ram[RX_BUF as usize..RX_BUF as usize + 128].fill(0xff);
        rig.post_receive(0, RX_BUF, 128);
        let frame = frame_of(60, 0x21);
        assert!(rig.net.send_frame(&frame));
        assert!(rig.poll());
        let header = received(&rig, RX_BUF, HEADER_LEN);
        let mut expected_header = [0u8; HEADER_LEN];
        expected_header[HEADER_NUM_BUFFERS_OFFSET] = 1;
        assert_eq!(
            header, expected_header,
            "flags, gso, csum fields zero; num_buffers == 1 little endian"
        );
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 60), frame);
        assert_eq!(
            rig.ram[RX_BUF as usize + HEADER_LEN + 60],
            0xff,
            "nothing past the frame is touched"
        );
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 60) as u32)
        );
    }

    #[test]
    fn frame_that_exactly_fills_the_buffer_is_delivered() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN + 60) as u32);
        let frame = frame_of(60, 1);
        rig.net.send_frame(&frame);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 60), frame);
    }

    #[test]
    fn frame_one_byte_too_big_for_the_buffer_is_dropped() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN + 59) as u32);
        rig.net.send_frame(&frame_of(60, 1));
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
        assert!(!rig.net.has_pending());
        assert!(
            received(&rig, RX_BUF, HEADER_LEN + 59)
                .iter()
                .all(|&b| b == 0),
            "no partial data"
        );
    }

    #[test]
    fn dropped_frame_leaves_its_buffer_posted_for_the_next_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 40);
        rig.net.send_frame(&frame_of(100, 1));
        assert!(!rig.poll());
        assert_eq!(
            rig.queues[RECEIVEQ].last_avail, 0,
            "the buffer is available again"
        );
        let small = frame_of(20, 9);
        rig.net.send_frame(&small);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 20), small);
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 20) as u32)
        );
    }

    #[test]
    fn oversize_frame_followed_by_a_fitting_one_in_the_same_poll() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 64);
        let big = frame_of(200, 1);
        let small = frame_of(30, 2);
        rig.net.send_frame(&big);
        rig.net.send_frame(&small);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 30), small);
        assert_eq!(used_idx(&rig.ram, RX_USED), 1);
        assert!(!rig.net.has_pending());
    }

    #[test]
    fn frame_waits_in_the_host_queue_until_a_buffer_is_posted() {
        let mut rig = Rig::new();
        rig.net.send_frame(&[7; 30]);
        assert!(!rig.poll());
        assert!(
            rig.net.has_pending(),
            "no buffer: the frame must not be lost"
        );
        rig.post_receive(0, RX_BUF, 128);
        assert!(rig.poll());
        assert_eq!(received(&rig, RX_BUF + HEADER_LEN as u64, 30), vec![7; 30]);
    }

    #[test]
    fn several_frames_fill_several_buffers_in_order() {
        let mut rig = Rig::new();
        for head in 0..3u16 {
            rig.post_receive(head, RX_BUF + 0x400 * head as u64, 256);
            rig.net.send_frame(&frame_of(40, head as u8));
        }
        assert!(rig.poll());
        for head in 0..3u16 {
            assert_eq!(
                received(&rig, RX_BUF + 0x400 * head as u64 + HEADER_LEN as u64, 40),
                frame_of(40, head as u8)
            );
            assert_eq!(
                used_entry(&rig.ram, RX_USED, head as usize),
                (head as u32, (HEADER_LEN + 40) as u32)
            );
        }
    }

    #[test]
    fn more_frames_than_buffers_leave_the_extra_ones_queued() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 256);
        rig.net.send_frame(&frame_of(40, 1));
        rig.net.send_frame(&frame_of(40, 2));
        assert!(rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 1);
        assert!(rig.net.has_pending());
    }

    #[test]
    fn empty_frame_is_delivered_as_just_a_header() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 64);
        rig.net.send_frame(&[]);
        assert!(rig.poll());
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, HEADER_LEN as u32));
    }

    #[test]
    fn receive_buffer_with_no_room_for_the_header_drops_the_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, (HEADER_LEN - 1) as u32);
        rig.net.send_frame(&[]);
        assert!(!rig.poll());
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
    }

    #[test]
    fn frame_may_span_several_writable_segments_of_one_buffer() {
        let mut rig = Rig::new();
        put_desc(
            &mut rig.ram,
            RX_DESC,
            0,
            RX_BUF,
            20,
            DESC_F_WRITE | DESC_F_NEXT,
            1,
        );
        put_desc(
            &mut rig.ram,
            RX_DESC,
            1,
            RX_BUF + 0x100,
            20,
            DESC_F_WRITE | DESC_F_NEXT,
            2,
        );
        put_desc(
            &mut rig.ram,
            RX_DESC,
            2,
            RX_BUF + 0x200,
            40,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        let frame = frame_of(50, 3);
        rig.net.send_frame(&frame);
        assert!(rig.poll());
        // 12 header bytes + 8 frame bytes, then 20, then 22 more.
        let mut gathered = received(&rig, RX_BUF + HEADER_LEN as u64, 8);
        gathered.extend(received(&rig, RX_BUF + 0x100, 20));
        gathered.extend(received(&rig, RX_BUF + 0x200, 22));
        assert_eq!(gathered, frame);
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, (HEADER_LEN + 50) as u32)
        );
    }

    #[test]
    fn the_buffer_size_check_counts_writable_segments_only() {
        let mut rig = Rig::new();
        put_desc(&mut rig.ram, RX_DESC, 0, RX_BUF, 1000, DESC_F_NEXT, 1);
        put_desc(
            &mut rig.ram,
            RX_DESC,
            1,
            RX_BUF + 0x400,
            20,
            DESC_F_WRITE,
            0,
        );
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        rig.net.send_frame(&frame_of(30, 1));
        assert!(
            !rig.poll(),
            "a 20-byte writable area cannot take 12 + 30 bytes"
        );
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
    }

    #[test]
    fn receive_buffer_outside_ram_breaks_the_queue_and_keeps_the_frame() {
        let mut rig = Rig::new();
        rig.post_receive(0, RAM_BYTES as u64 - 4, 64);
        rig.net.send_frame(&[1, 2, 3]);
        assert!(!rig.poll());
        assert!(rig.queues[RECEIVEQ].broken);
        assert!(rig.net.has_pending());
    }

    #[test]
    fn guest_to_host_and_host_to_guest_traffic_do_not_interfere() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 128);
        put_tx_frame(&mut rig, 0, &[0; HEADER_LEN], &[9, 9, 9]);
        rig.net.send_frame(&[4, 4, 4, 4]);
        assert!(rig.kick(TRANSMITQ));
        assert!(rig.poll());
        assert_eq!(rig.net.try_recv_frame().unwrap(), vec![9, 9, 9]);
        assert_eq!(
            received(&rig, RX_BUF + HEADER_LEN as u64, 4),
            vec![4, 4, 4, 4]
        );
    }

    #[test]
    fn tx_and_rx() {
        let net = NetHandle::default();
        let mut dev = VirtioNet::new([2, 0, 0, 0, 0, 1], net.clone());
        let mut ram = vec![0u8; 0x20000];
        let mut qs = vec![
            setup_queue(&mut ram, 0x1000, 0x2000, 0x3000, 8),
            setup_queue(&mut ram, 0x8000, 0x9000, 0xa000, 8),
        ];
        let frame = [0xabu8; 60];
        ram[0xc000 + HEADER_LEN..0xc000 + HEADER_LEN + 60].copy_from_slice(&frame);
        put_desc(&mut ram, 0x8000, 0, 0xc000, (HEADER_LEN + 60) as u32, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut qs, &mut mem));
        assert_eq!(net.try_recv_frame().unwrap(), frame);
        assert!(!dev.poll(&mut qs, &mut mem));
        assert!(net.send_frame(&[1, 2, 3, 4]));
        put_desc(&mut ram, 0x1000, 0, 0xd000, 2048, 2, 0);
        post(&mut ram, 0x2000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.poll(&mut qs, &mut mem));
        assert_eq!(
            &ram[0xd000 + HEADER_LEN..0xd000 + HEADER_LEN + 4],
            &[1, 2, 3, 4]
        );
        assert_eq!(used_idx(&ram, 0x3000), 1);
    }

    const BOTH_WAYS: NetOffloads = NetOffloads {
        takes_guest_segments: true,
        sends_large_segments: true,
    };
    const GUEST_TO_HOST: u64 = F_CSUM | F_HOST_TSO4;
    const HOST_TO_GUEST: u64 = F_GUEST_CSUM | F_GUEST_TSO4 | F_MRG_RXBUF;
    const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
    const REMOTE_IP: [u8; 4] = [10, 0, 2, 2];
    /// Ethernet, a 20-byte IPv4 header, a 20-byte TCP header.
    const TCP_HEADERS_LEN: usize = 54;
    const MSS: u16 = 1460;

    /// Ones-complement sum done the slow, obvious way, independent of `ones_complement_sum`.
    fn reference_sum(parts: &[&[u8]]) -> u16 {
        let mut acc: u32 = 0;
        for part in parts {
            for pair in part.chunks(2) {
                acc += u16::from_be_bytes([pair[0], *pair.get(1).unwrap_or(&0)]) as u32;
                acc = (acc & 0xffff) + (acc >> 16);
            }
        }
        acc as u16
    }

    fn pseudo_header(segment_len: usize) -> Vec<u8> {
        let mut pseudo = GUEST_IP.to_vec();
        pseudo.extend_from_slice(&REMOTE_IP);
        pseudo.extend_from_slice(&[0, IP_PROTO_TCP]);
        pseudo.extend_from_slice(&(segment_len as u16).to_be_bytes());
        pseudo
    }

    /// An Ethernet/IPv4/TCP frame carrying `payload`, its TCP checksum field seeded with the
    /// pseudo-header sum the way a TSO sender leaves it.
    fn tcp_frame(payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![0u8; TCP_HEADERS_LEN];
        frame[ETHERNET_TYPE..ETHERNET_TYPE + 2].copy_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        let ip = ETHERNET_HEADER_LEN;
        frame[ip] = 0x45;
        let total = (IPV4_MIN_HEADER_LEN + TCP_MIN_HEADER_LEN + payload.len()) as u16;
        frame[ip + IPV4_TOTAL_LEN..ip + IPV4_TOTAL_LEN + 2].copy_from_slice(&total.to_be_bytes());
        frame[ip + IPV4_PROTOCOL] = IP_PROTO_TCP;
        frame[ip + 12..ip + 16].copy_from_slice(&REMOTE_IP);
        frame[ip + 16..ip + 20].copy_from_slice(&GUEST_IP);
        let tcp = ip + IPV4_MIN_HEADER_LEN;
        frame[tcp..tcp + 4].copy_from_slice(&[0x1f, 0x90, 0xc3, 0x50]);
        frame[tcp + TCP_DATA_OFFSET] = 5 << 4;
        let seed = reference_sum(&[&pseudo_header(TCP_MIN_HEADER_LEN + payload.len())]);
        frame[tcp + TCP_CHECKSUM..tcp + TCP_CHECKSUM + 2].copy_from_slice(&seed.to_be_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    /// True when the TCP checksum of `frame` (as built by `tcp_frame`) is complete and right.
    fn tcp_checksum_is_valid(frame: &[u8]) -> bool {
        let segment = &frame[ETHERNET_HEADER_LEN + IPV4_MIN_HEADER_LEN..];
        reference_sum(&[&pseudo_header(segment.len()), segment]) == 0xffff
    }

    /// Finishes a partial checksum as the guest would from `csum_start`/`csum_offset`.
    fn finish_checksum(frame: &mut [u8], header: NetHeader) {
        let start = header.csum_start as usize;
        let at = start + header.csum_offset as usize;
        let sum = reference_sum(&[&frame[start..]]);
        frame[at..at + 2].copy_from_slice(&(!sum).to_be_bytes());
    }

    fn header_of(rig: &Rig, buffer: u64) -> NetHeader {
        NetHeader::parse(&received(rig, buffer, HEADER_LEN).try_into().unwrap())
    }

    /// Gathers what `count` used entries from slot `first` hold, header included, from
    /// single-page buffers posted with `post_pages`.
    fn gather_merged(rig: &Rig, first: usize, count: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for slot in first..first + count {
            let (head, len) = used_entry(&rig.ram, RX_USED, slot);
            bytes.extend(received(
                rig,
                BIG_RX + PAGE as u64 * head as u64,
                len as usize,
            ));
        }
        bytes
    }

    fn put_big_tx_frame(rig: &mut Rig, header: NetHeader, frame: &[u8]) {
        let at = BIG_TX as usize;
        rig.ram[at..at + HEADER_LEN].copy_from_slice(&header.to_bytes());
        rig.ram[at + HEADER_LEN..at + HEADER_LEN + frame.len()].copy_from_slice(frame);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            0,
            BIG_TX,
            (HEADER_LEN + frame.len()) as u32,
            0,
            0,
        );
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
    }

    mod header {
        use super::*;

        #[test]
        fn fields_sit_at_the_virtio_net_hdr_v1_offsets_little_endian() {
            let header = NetHeader {
                flags: 0x01,
                gso_type: 0x02,
                hdr_len: 0x0403,
                gso_size: 0x0605,
                csum_start: 0x0807,
                csum_offset: 0x0a09,
                num_buffers: 0x0c0b,
            };
            assert_eq!(header.to_bytes(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            assert_eq!(NetHeader::parse(&header.to_bytes()), header);
        }

        #[test]
        fn any_twelve_bytes_parse_and_round_trip() {
            for seed in 0..64u8 {
                let bytes: [u8; HEADER_LEN] =
                    core::array::from_fn(|i| seed.wrapping_mul(37).wrapping_add(i as u8 * 11));
                assert_eq!(NetHeader::parse(&bytes).to_bytes(), bytes);
            }
        }

        #[test]
        fn the_maximum_frame_holds_a_whole_tagged_tso_segment() {
            assert_eq!(MAX_FRAME_LEN, 14 + 4 + 65535);
        }

        #[test]
        fn ones_complement_sum_matches_the_reference_for_every_length_and_seed() {
            let data: Vec<u8> = (0..300u32).map(|i| (i * 151 + 7) as u8).collect();
            for len in 0..data.len() {
                for seed in [0u16, 1, 0x8000, 0xfffe, 0xffff] {
                    let expected = reference_sum(&[&seed.to_be_bytes(), &data[..len]]);
                    let got = ones_complement_sum(seed, &data[..len]);
                    // 0 and 0xffff are the same number in ones complement.
                    assert!(
                        got == expected || (got | expected) == 0xffff && got & expected == 0,
                        "len {len} seed {seed:#x}: {got:#x} vs {expected:#x}"
                    );
                }
            }
        }

        #[test]
        fn ones_complement_sum_of_a_full_size_frame_of_ones_does_not_overflow() {
            let data = vec![0xffu8; MAX_FRAME_LEN];
            assert_eq!(
                ones_complement_sum(0xffff, &data),
                reference_sum(&[&[0xff, 0xff], &data])
            );
        }
    }

    mod negotiation {
        use super::*;

        #[test]
        fn a_host_that_declares_nothing_offers_no_offloads() {
            let rig = Rig::new();
            assert_eq!(rig.dev.features(), F_MAC | F_STATUS);
        }

        #[test]
        fn each_declaration_offers_its_own_pair_of_offloads() {
            let rig = Rig::new();
            rig.net.set_offloads(NetOffloads {
                takes_guest_segments: true,
                sends_large_segments: false,
            });
            assert_eq!(rig.dev.features(), F_MAC | F_STATUS | F_CSUM | F_HOST_TSO4);
            rig.net.set_offloads(NetOffloads {
                takes_guest_segments: false,
                sends_large_segments: true,
            });
            assert_eq!(
                rig.dev.features(),
                F_MAC | F_STATUS | F_GUEST_CSUM | F_GUEST_TSO4 | F_MRG_RXBUF
            );
            rig.net.set_offloads(BOTH_WAYS);
            assert_eq!(
                rig.dev.features(),
                F_MAC | F_STATUS | GUEST_TO_HOST | HOST_TO_GUEST
            );
        }

        #[test]
        fn feature_bits_are_the_virtio_net_numbers() {
            assert_eq!(
                [F_CSUM, F_GUEST_CSUM, F_GUEST_TSO4, F_HOST_TSO4, F_MRG_RXBUF],
                [1 << 0, 1 << 1, 1 << 7, 1 << 11, 1 << 15]
            );
        }

        #[test]
        fn the_host_learns_whether_the_guest_takes_segments() {
            let mut rig = Rig::negotiated(BOTH_WAYS, 0);
            assert!(!rig.net.guest_takes_tcp_segments());
            rig.dev.features_negotiated(F_GUEST_CSUM | F_GUEST_TSO4);
            assert!(rig.net.guest_takes_tcp_segments());
            rig.dev.features_negotiated(F_GUEST_CSUM | F_MRG_RXBUF);
            assert!(
                !rig.net.guest_takes_tcp_segments(),
                "checksums alone are not segments"
            );
            rig.dev.features_negotiated(HOST_TO_GUEST);
            rig.dev.reset();
            assert!(!rig.net.guest_takes_tcp_segments());
        }

        /// A reboot rebuilds the device around the same handle; the new driver has not
        /// negotiated anything yet.
        #[test]
        fn a_new_device_on_an_old_handle_starts_with_nothing_negotiated() {
            let rig = Rig::negotiated(BOTH_WAYS, HOST_TO_GUEST);
            assert!(rig.net.guest_takes_tcp_segments());
            let _rebuilt = VirtioNet::new(MAC, rig.net.clone());
            assert!(!rig.net.guest_takes_tcp_segments());
        }

        #[test]
        fn a_reset_forgets_the_negotiated_offloads() {
            let mut rig = Rig::negotiated(BOTH_WAYS, GUEST_TO_HOST | HOST_TO_GUEST);
            rig.dev.reset();
            let mut header = NetHeader {
                flags: HDR_F_NEEDS_CSUM,
                ..NetHeader::default()
            };
            header.csum_start = 1000;
            // With nothing negotiated the header is ignored again.
            put_tx_frame(&mut rig, 0, &header.to_bytes(), &[1, 2, 3]);
            rig.kick(TRANSMITQ);
            assert_eq!(rig.net.try_recv_frame().unwrap(), vec![1, 2, 3]);
        }
    }

    mod transmit {
        use super::*;

        fn sent_with(header: NetHeader, frame: &[u8], negotiated: u64) -> Option<Vec<u8>> {
            let mut rig = Rig::negotiated(BOTH_WAYS, negotiated);
            put_big_tx_frame(&mut rig, header, frame);
            assert!(rig.kick(TRANSMITQ), "the chain is completed either way");
            assert_eq!(used_idx(&rig.ram, TX_USED), 1);
            rig.net.try_recv_frame()
        }

        fn partial(frame: &[u8]) -> NetHeader {
            NetHeader {
                flags: HDR_F_NEEDS_CSUM,
                csum_start: (ETHERNET_HEADER_LEN + IPV4_MIN_HEADER_LEN) as u16,
                csum_offset: TCP_CHECKSUM as u16,
                ..NetHeader::default()
            }
            .with_len_check(frame)
        }

        impl NetHeader {
            fn with_len_check(self, frame: &[u8]) -> NetHeader {
                assert!(frame.len() >= TCP_HEADERS_LEN);
                self
            }

            fn segment(self, gso_size: u16) -> NetHeader {
                NetHeader {
                    gso_type: GSO_TCPV4,
                    gso_size,
                    hdr_len: TCP_HEADERS_LEN as u16,
                    ..self
                }
            }
        }

        #[test]
        fn a_partial_checksum_goes_to_the_host_untouched() {
            let frame = tcp_frame(&frame_of(500, 3));
            let sent = sent_with(partial(&frame), &frame, GUEST_TO_HOST).unwrap();
            assert_eq!(sent, frame, "the host finishes (or ignores) the checksum");
        }

        #[test]
        fn a_whole_tso_segment_goes_to_the_host_in_one_piece() {
            let payload = frame_of(MAX_FRAME_LEN - VLAN_TAG_LEN - TCP_HEADERS_LEN, 9);
            let frame = tcp_frame(&payload);
            let header = partial(&frame).segment(MSS);
            assert_eq!(sent_with(header, &frame, GUEST_TO_HOST).unwrap(), frame);
        }

        #[test]
        fn a_checksum_position_past_the_frame_is_dropped() {
            let frame = tcp_frame(&[7; 10]);
            let last_fit = NetHeader {
                flags: HDR_F_NEEDS_CSUM,
                csum_start: (frame.len() - 2) as u16,
                csum_offset: 0,
                ..NetHeader::default()
            };
            assert!(sent_with(last_fit, &frame, GUEST_TO_HOST).is_some());
            for (csum_start, csum_offset) in [
                (frame.len() - 1, 0),
                (frame.len() - 2, 1),
                (0, frame.len() - 1),
                (u16::MAX as usize, u16::MAX as usize),
            ] {
                let header = NetHeader {
                    csum_start: csum_start as u16,
                    csum_offset: csum_offset as u16,
                    ..last_fit
                };
                assert!(
                    sent_with(header, &frame, GUEST_TO_HOST).is_none(),
                    "start {csum_start} offset {csum_offset}"
                );
            }
        }

        #[test]
        fn a_segment_needs_host_tso4_negotiated() {
            let frame = tcp_frame(&frame_of(3000, 1));
            let header = partial(&frame).segment(MSS);
            assert!(sent_with(header, &frame, F_CSUM).is_none());
        }

        #[test]
        fn a_segment_needs_a_size_and_the_checksum_flag() {
            let frame = tcp_frame(&frame_of(3000, 1));
            assert!(sent_with(partial(&frame).segment(0), &frame, GUEST_TO_HOST).is_none());
            let unflagged = NetHeader {
                flags: 0,
                ..partial(&frame).segment(MSS)
            };
            assert!(sent_with(unflagged, &frame, GUEST_TO_HOST).is_none());
        }

        #[test]
        fn segment_types_never_offered_are_dropped() {
            let frame = tcp_frame(&frame_of(3000, 1));
            for gso_type in [3u8, 4, 5, 0x20, 0x40, 0x80 | GSO_TCPV4, 0xff] {
                let header = NetHeader {
                    gso_type,
                    ..partial(&frame).segment(MSS)
                };
                assert!(
                    sent_with(header, &frame, GUEST_TO_HOST).is_none(),
                    "gso type {gso_type:#x}"
                );
            }
        }

        #[test]
        fn an_ordinary_frame_with_a_zero_header_passes_with_offloads_negotiated() {
            let frame = frame_of(60, 4);
            assert_eq!(
                sent_with(NetHeader::default(), &frame, GUEST_TO_HOST).unwrap(),
                frame
            );
        }
    }

    mod receive {
        use super::*;

        fn guest_takes_everything() -> Rig {
            Rig::negotiated(BOTH_WAYS, GUEST_TO_HOST | HOST_TO_GUEST)
        }

        #[test]
        fn a_long_segment_is_marked_for_the_guest_to_split_and_checksum() {
            let mut rig = guest_takes_everything();
            rig.post_pages(0, 20);
            let payload = frame_of(20_000, 5);
            let frame = tcp_frame(&payload);
            assert!(rig.net.send_tcp_segment(&frame, MSS));
            assert!(rig.poll());
            let header = header_of(&rig, BIG_RX);
            assert_eq!(header.flags, HDR_F_NEEDS_CSUM);
            assert_eq!(header.gso_type, GSO_TCPV4);
            assert_eq!(header.gso_size, MSS);
            assert_eq!(header.hdr_len, TCP_HEADERS_LEN as u16);
            assert_eq!(header.csum_start, 34);
            assert_eq!(header.csum_offset, 16);
            let buffers = (HEADER_LEN + frame.len()).div_ceil(PAGE as usize);
            assert_eq!(header.num_buffers as usize, buffers);
            assert_eq!(used_idx(&rig.ram, RX_USED) as usize, buffers);
            let mut delivered = gather_merged(&rig, 0, buffers)[HEADER_LEN..].to_vec();
            assert_eq!(
                delivered, frame,
                "the frame is passed on as the host built it"
            );
            finish_checksum(&mut delivered, header);
            assert!(tcp_checksum_is_valid(&delivered));
        }

        #[test]
        fn a_segment_within_the_mss_needs_its_checksum_but_is_not_gso() {
            let mut rig = guest_takes_everything();
            rig.post_pages(0, 1);
            let frame = tcp_frame(&frame_of(MSS as usize, 2));
            rig.net.send_tcp_segment(&frame, MSS);
            assert!(rig.poll());
            let header = header_of(&rig, BIG_RX);
            assert_eq!(header.flags, HDR_F_NEEDS_CSUM);
            assert_eq!(header.gso_type, GSO_NONE);
            assert_eq!((header.gso_size, header.hdr_len), (0, 0));
            let mut delivered = gather_merged(&rig, 0, 1)[HEADER_LEN..].to_vec();
            finish_checksum(&mut delivered, header);
            assert!(tcp_checksum_is_valid(&delivered));
        }

        #[test]
        fn without_guest_csum_the_device_finishes_the_checksum() {
            let mut rig = Rig::negotiated(BOTH_WAYS, 0);
            rig.post_receive(0, RX_BUF, 2048);
            for len in [0, 1, 2, 3, 1459, 1460] {
                let frame = tcp_frame(&frame_of(len, len as u8));
                rig.net.send_tcp_segment(&frame, MSS);
                assert!(rig.poll(), "payload {len}");
                assert_eq!(
                    header_of(&rig, RX_BUF),
                    NetHeader {
                        num_buffers: 1,
                        ..NetHeader::default()
                    }
                );
                let delivered = received(&rig, RX_BUF + HEADER_LEN as u64, frame.len());
                assert!(tcp_checksum_is_valid(&delivered), "payload {len}");
                assert_eq!(delivered[..50], frame[..50], "only the checksum changes");
                assert_eq!(delivered[52..], frame[52..]);
                rig.queues[RECEIVEQ].rewind_to(0);
            }
        }

        #[test]
        fn a_long_segment_the_guest_cannot_take_is_dropped() {
            for negotiated in [0, F_GUEST_CSUM | F_MRG_RXBUF] {
                let mut rig = Rig::negotiated(BOTH_WAYS, negotiated);
                rig.post_pages(0, 20);
                rig.net
                    .send_tcp_segment(&tcp_frame(&frame_of(MSS as usize + 1, 1)), MSS);
                let small = frame_of(60, 1);
                rig.net.send_frame(&small);
                assert!(rig.poll());
                assert_eq!(used_idx(&rig.ram, RX_USED), 1, "only the plain frame");
                assert_eq!(received(&rig, BIG_RX + HEADER_LEN as u64, 60), small);
            }
        }

        #[test]
        fn a_segment_that_is_not_tcp_over_ipv4_is_dropped_without_panicking() {
            // Short enough that a 60-byte IPv4 or TCP header cannot fit.
            let good = tcp_frame(&frame_of(30, 1));
            let mut broken: Vec<Vec<u8>> =
                (0..good.len()).map(|len| good[..len].to_vec()).collect();
            for (at, value) in [
                (ETHERNET_TYPE, 0x86u8),
                (ETHERNET_HEADER_LEN, 0x65),
                (ETHERNET_HEADER_LEN, 0x44),
                (ETHERNET_HEADER_LEN, 0x4f),
                (ETHERNET_HEADER_LEN + IPV4_PROTOCOL, 17),
                (ETHERNET_HEADER_LEN + IPV4_TOTAL_LEN, 0xff),
                (ETHERNET_HEADER_LEN + IPV4_TOTAL_LEN + 1, 30),
                (34 + TCP_DATA_OFFSET, 0x40),
                (34 + TCP_DATA_OFFSET, 0xf0),
            ] {
                let mut frame = good.clone();
                frame[at] = value;
                if at == ETHERNET_HEADER_LEN + IPV4_TOTAL_LEN + 1 {
                    frame[at - 1] = 0;
                }
                broken.push(frame);
            }
            for negotiated in [0, GUEST_TO_HOST | HOST_TO_GUEST] {
                let mut rig = Rig::negotiated(BOTH_WAYS, negotiated);
                rig.post_pages(0, 4);
                for frame in &broken {
                    rig.net.send_tcp_segment(frame, MSS);
                }
                assert!(!rig.poll());
                assert_eq!(used_idx(&rig.ram, RX_USED), 0);
                assert!(!rig.net.has_pending());
            }
        }

        #[test]
        fn a_merged_frame_waits_until_enough_buffers_are_posted() {
            let mut rig = guest_takes_everything();
            let frame = tcp_frame(&frame_of(10_000, 3));
            rig.post_pages(0, 2);
            rig.net.send_tcp_segment(&frame, MSS);
            assert!(!rig.poll());
            assert_eq!(used_idx(&rig.ram, RX_USED), 0);
            assert_eq!(rig.queues[RECEIVEQ].last_avail, 0, "nothing consumed");
            assert!(rig.net.has_pending(), "the frame is kept");
            assert_eq!(received(&rig, BIG_RX, 16), vec![0; 16], "nothing written");
            rig.post_pages(2, 1);
            assert!(rig.poll());
            assert_eq!(used_idx(&rig.ram, RX_USED), 3);
            assert_eq!(gather_merged(&rig, 0, 3)[HEADER_LEN..], frame[..]);
        }

        #[test]
        fn a_waiting_frame_keeps_its_place_ahead_of_later_ones() {
            let mut rig = guest_takes_everything();
            let big = tcp_frame(&frame_of(6000, 3));
            let small = frame_of(60, 8);
            rig.post_pages(0, 1);
            rig.net.send_tcp_segment(&big, MSS);
            rig.net.send_frame(&small);
            assert!(!rig.poll(), "the small frame does not overtake");
            rig.post_pages(1, 2);
            assert!(rig.poll());
            assert_eq!(used_idx(&rig.ram, RX_USED), 3, "two pages, then one");
            assert_eq!(gather_merged(&rig, 0, 2)[HEADER_LEN..], big[..]);
            assert_eq!(gather_merged(&rig, 2, 1)[HEADER_LEN..], small[..]);
        }

        #[test]
        fn the_header_must_fit_the_first_merged_buffer() {
            let mut rig = guest_takes_everything();
            rig.post_receive(0, BIG_RX, (HEADER_LEN - 1) as u32);
            rig.post_pages(1, 2);
            rig.net.send_frame(&frame_of(100, 1));
            assert!(!rig.poll());
            assert_eq!(used_idx(&rig.ram, RX_USED), 0);
            assert_eq!(
                rig.queues[RECEIVEQ].last_avail, 0,
                "the buffers stay posted"
            );
        }

        #[test]
        fn a_frame_larger_than_a_full_ring_of_buffers_is_dropped() {
            let mut rig = guest_takes_everything();
            for head in 0..QUEUE_SIZE {
                rig.post_receive(head, BIG_RX + 64 * head as u64, 64);
            }
            rig.net.send_frame(&frame_of(64 * QUEUE_SIZE as usize, 1));
            let small = frame_of(40, 2);
            rig.net.send_frame(&small);
            assert!(rig.poll());
            assert_eq!(used_idx(&rig.ram, RX_USED), 1);
            assert_eq!(received(&rig, BIG_RX + HEADER_LEN as u64, 40), small);
        }

        #[test]
        fn a_plain_frame_gets_a_zero_header_counting_its_buffers() {
            let mut rig = guest_takes_everything();
            rig.post_pages(0, 2);
            let frame = frame_of(PAGE as usize, 6);
            rig.net.send_frame(&frame);
            assert!(rig.poll());
            assert_eq!(
                header_of(&rig, BIG_RX),
                NetHeader {
                    num_buffers: 2,
                    ..NetHeader::default()
                }
            );
            assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, PAGE));
            assert_eq!(used_entry(&rig.ram, RX_USED, 1), (1, HEADER_LEN as u32));
            assert_eq!(gather_merged(&rig, 0, 2)[HEADER_LEN..], frame[..]);
        }

        #[test]
        fn without_merged_buffers_a_long_segment_needs_one_big_buffer() {
            let negotiated = F_GUEST_CSUM | F_GUEST_TSO4;
            let frame = tcp_frame(&frame_of(5000, 3));
            let mut pages = Rig::negotiated(BOTH_WAYS, negotiated);
            pages.post_pages(0, 3);
            pages.net.send_tcp_segment(&frame, MSS);
            assert!(!pages.poll(), "no single page takes it");
            assert!(!pages.net.has_pending(), "and it never will: dropped");
            let mut big = Rig::negotiated(BOTH_WAYS, negotiated);
            big.post_receive(0, BIG_RX, 4 * PAGE);
            big.net.send_tcp_segment(&frame, MSS);
            assert!(big.poll());
            assert_eq!(
                used_entry(&big.ram, RX_USED, 0),
                (0, (HEADER_LEN + frame.len()) as u32)
            );
            let header = header_of(&big, BIG_RX);
            assert_eq!((header.num_buffers, header.gso_type), (1, GSO_TCPV4));
        }
    }
}
