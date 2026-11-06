//! One Ethernet segment shared by machines that cannot share memory: the web page's machines,
//! one per browser tab, which reach each other only by passing messages. `SegmentAgent` is
//! each machine's port on it. The host carries frames between agents over whatever it has
//! (`docs/plans/13-web.md`: BroadcastChannel), each tagged with the sending tab's number; the
//! agent decides where each frame goes, so the host's part is a transport and nothing else.
//!
//! What the agent does:
//! - Switching. It learns which tab each source MAC lives behind and sends unicast frames to
//!   that tab only; group addresses and unknown destinations go to every tab. A guest may have
//!   more MACs than its NIC's (a bridge, a Xen domU), so learning is by frame, not by NIC.
//! - The segment's server. Exactly one agent at a time serves (the host elects it: a Web Lock),
//!   answering for `SERVER_IP` at `SERVER_MAC`: ARP, ping and DHCP. Both are fixed, so when the
//!   serving tab closes and another takes over the guests' ARP caches and lease renewals stay
//!   valid. Frames for `SERVER_MAC` go to the host's server destination, never to a learned tab.
//! - Presence. Every agent announces, on an EtherType of its own that no guest ever sees, the
//!   addresses its guests use: what DHCP gave them (seen in the replies the agent delivers) and
//!   what they send from. Every agent keeps the resulting lease table, so a new server already
//!   knows which addresses are taken. The announcements also count the peers.
//!
//! The segment itself has no way off it: DHCP names the server's address as router and DNS
//! server, and each tab's gateway (see `crate::gateway`) takes that traffic for its guest.

mod dhcp;
mod packet;
#[cfg(test)]
mod tests;

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;
use core::ops::Range;

use dhcp::{DHCP_ACK, DHCP_CLIENT_PORT, DHCP_NAK, DHCP_SERVER_PORT, Leases};
use packet::{
    ARP_REPLY, ARP_REQUEST, BROADCAST_MAC, ETHERTYPE_ARP, ETHERTYPE_IPV4, IP_PROTOCOL_ICMP,
    IP_PROTOCOL_UDP, Ipv4, Mac, arp_packet, ethernet_frame, icmp_echo_reply, ipv4_packet, is_group,
    is_local_unicast, parse_arp, parse_ethernet, parse_ipv4, parse_udp,
};

use crate::Error;

/// The segment is 10.0.2.0/24, the network the CLI's user-mode NIC has, so the addresses look
/// familiar: the server is 10.0.2.2 and guests get 10.0.2.15 and up.
const SUBNET_PREFIX: [u8; 3] = [10, 0, 2];
const SUBNET_MASK: Ipv4 = [255, 255, 255, 0];
const SUBNET_BROADCAST: Ipv4 = [10, 0, 2, 255];
const SERVER_IP: Ipv4 = [10, 0, 2, 2];
/// Locally administered, so no NIC a tab makes up can be told it is a real one; and a tab's NIC
/// can never have it (`SegmentAgent::new` refuses).
const SERVER_MAC: Mac = [0x52, 0x55, 0x0a, 0x00, 0x02, 0x02];
// The web page's gateway takes the frames sent to this MAC (`segment_takes`); were the two to
// differ it would silently take nothing.
const _: () = {
    let mut byte = 0;
    while byte < SERVER_MAC.len() {
        assert!(SERVER_MAC[byte] == crate::gateway::GATEWAY_MAC[byte]);
        byte += 1;
    }
};
const POOL_FIRST: u8 = 15;
const POOL_LAST: u8 = 254;

/// Not assigned to any protocol: IEEE 802 "local experimental EtherType 1", for exactly this.
const ETHERTYPE_PRESENCE: u16 = 0x88b5;
const PRESENCE_MAGIC: [u8; 4] = *b"uemu";
const PRESENCE_VERSION: u8 = 1;
/// The routine announcement.
const PRESENCE_ANNOUNCE: u8 = 1;
/// From an agent that just began serving: everyone announces at once, so its table is whole
/// before it offers an address.
const PRESENCE_CENSUS: u8 = 2;
const PRESENCE_MAGIC_RANGE: Range<usize> = 0..4;
const PRESENCE_VERSION_AT: usize = 4;
const PRESENCE_KIND_AT: usize = 5;
const PRESENCE_FLAGS_AT: usize = 6;
const PRESENCE_COUNT_AT: usize = 7;
const PRESENCE_HEADER_LEN: usize = 8;
/// A station's MAC and IPv4 address.
const PRESENCE_CLAIM_LEN: usize = 10;
/// Set by the agent that serves. Nothing reads it: it is there so a capture of the channel shows
/// which tab served when.
const PRESENCE_FLAG_SERVING: u8 = 1;
/// Addresses one agent announces at most, so the frame stays small (and below the MTU); a guest
/// with more stations behind it than this is not a case worth a fragmented announcement.
const CLAIMS_MAX: usize = 64;

pub(crate) const PRESENCE_INTERVAL_MS: u64 = 1000;
/// A tab whose announcement has not arrived for this long is gone (closed, frozen in the
/// background); the stations learned behind it are forgotten, so frames for them are flooded.
pub(crate) const PEER_TIMEOUT_MS: u64 = 4 * PRESENCE_INTERVAL_MS;
/// An agent that has listened this long has heard every live tab announce at least twice.
const LISTEN_BEFORE_OFFERING_MS: u64 = 2 * PRESENCE_INTERVAL_MS + PRESENCE_INTERVAL_MS / 2;
/// After its census, a new server waits this long for the answers before it offers addresses.
const CENSUS_WAIT_MS: u64 = PRESENCE_INTERVAL_MS / 2;
/// A switch's MAC table ageing time (802.1D's default is 300 s).
const STATION_TIMEOUT_MS: u64 = 300_000;

fn in_subnet(address: Ipv4) -> bool {
    address[..3] == SUBNET_PREFIX
}

/// A host address on the segment someone may use: not the network, the broadcast or the server.
fn usable(address: Ipv4) -> bool {
    in_subnet(address) && address != SERVER_IP && address[3] != 0 && address != SUBNET_BROADCAST
}

/// Where the host sends a frame the agent hands it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentDestination {
    /// Every other tab: broadcast, multicast and unicast to a MAC no tab has been seen with.
    Everyone,
    /// The tab with this number, behind which the destination MAC was last seen.
    Tab(u32),
    /// Whichever tab serves the segment now (the host's election decides).
    Server,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentStatus {
    /// The guest's address on the segment, from DHCP or from what it sends. `None` until then.
    pub address: Option<[u8; 4]>,
    /// Other tabs on the segment (that announced themselves within the last few seconds).
    pub peers: usize,
    /// This agent is the segment's DHCP server.
    pub serving: bool,
}

struct Station {
    tab: u32,
    seen_ms: u64,
}

/// A machine's port on the shared segment. Feed it the guest's frames (`from_guest`), the
/// frames the host received from other tabs (`from_segment`) and the time (`tick`), and take
/// what it produced with `take_for_guest` and `take_for_segment`. It never reads a clock: every
/// method takes the host's monotonic milliseconds, which never leave this agent.
pub struct SegmentAgent {
    mac: Mac,
    tab: u32,
    joined_ms: u64,
    serving_since_ms: Option<u64>,
    stations: BTreeMap<Mac, Station>,
    /// Other tabs, by number, and when each last announced itself.
    peers: BTreeMap<u32, u64>,
    /// Addresses the stations behind this tab use, as announced.
    claims: BTreeMap<Mac, Ipv4>,
    leases: Leases,
    last_announce_ms: Option<u64>,
    announce_now: bool,
    census_now: bool,
    for_guest: Vec<Vec<u8>>,
    for_segment: Vec<(SegmentDestination, Vec<u8>)>,
}

impl SegmentAgent {
    /// `mac` is the guest NIC's (`MachineConfig::net`): a locally administered unicast address
    /// the host made up at random, so two tabs never share one. `tab` is this tab's number on
    /// the transport, unique among the tabs as well; frames the host hands back with it are
    /// this agent's own (a broadcast echoed to its sender) and are dropped.
    pub fn new(mac: [u8; 6], tab: u32, now_ms: u64) -> Result<Self, Error> {
        if !is_local_unicast(&mac) || mac == SERVER_MAC {
            return Err(Error::BadConfig(format!(
                "segment NIC address {mac:02x?} must be a locally administered unicast address"
            )));
        }
        Ok(Self {
            mac,
            tab,
            joined_ms: now_ms,
            serving_since_ms: None,
            stations: BTreeMap::new(),
            peers: BTreeMap::new(),
            claims: BTreeMap::new(),
            leases: Leases::default(),
            last_announce_ms: None,
            announce_now: true,
            census_now: false,
            for_guest: Vec::new(),
            for_segment: Vec::new(),
        })
    }

    /// Whether this agent serves the segment. The host's election must guarantee that at most
    /// one agent serves at a time; two would hand out the same addresses.
    pub fn set_serving(&mut self, serving: bool, now_ms: u64) {
        match (serving, self.serving_since_ms) {
            (true, None) => {
                self.serving_since_ms = Some(now_ms);
                self.census_now = true;
            }
            (false, Some(_)) => self.serving_since_ms = None,
            _ => {}
        }
    }

    pub fn status(&self) -> SegmentStatus {
        SegmentStatus {
            address: self.claims.get(&self.mac).copied(),
            peers: self.peers.len(),
            serving: self.serving_since_ms.is_some(),
        }
    }

    /// Frames for the guest's NIC, in order.
    pub fn take_for_guest(&mut self) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.for_guest)
    }

    /// Frames for the host to send, in order, each with where to.
    pub fn take_for_segment(&mut self) -> Vec<(SegmentDestination, Vec<u8>)> {
        core::mem::take(&mut self.for_segment)
    }

    /// A frame the guest sent.
    pub fn from_guest(&mut self, frame: &[u8], now_ms: u64) {
        let Some(ethernet) = parse_ethernet(frame) else {
            return;
        };
        if is_group(&ethernet.src) {
            return;
        }
        self.observe_guest_address(&ethernet.src, ethernet.ethertype, ethernet.payload, now_ms);
        if self.serving()
            && (ethernet.dst == SERVER_MAC || is_group(&ethernet.dst))
            && let Some(reply) = self.serve(frame, now_ms)
        {
            self.deliver_to_guest(reply);
        }
        match ethernet.dst {
            SERVER_MAC if self.serving() => {}
            // To itself, or to another station behind this same tab: already on its wire.
            dst if dst == self.mac
                || self.stations.get(&dst).is_some_and(|s| s.tab == self.tab) => {}
            dst => {
                let destination = self.route(&dst, now_ms);
                self.for_segment.push((destination, frame.to_vec()));
            }
        }
    }

    /// A frame the host received from tab `from_tab`.
    pub fn from_segment(&mut self, from_tab: u32, frame: &[u8], now_ms: u64) {
        if from_tab == self.tab {
            return;
        }
        let Some(ethernet) = parse_ethernet(frame) else {
            return;
        };
        if ethernet.ethertype == ETHERTYPE_PRESENCE {
            self.hear_presence(from_tab, ethernet.payload, now_ms);
            return;
        }
        if !is_group(&ethernet.src) && ethernet.src != SERVER_MAC {
            self.stations.insert(
                ethernet.src,
                Station {
                    tab: from_tab,
                    seen_ms: now_ms,
                },
            );
        }
        if self.serving()
            && (ethernet.dst == SERVER_MAC || is_group(&ethernet.dst))
            && let Some(reply) = self.serve(frame, now_ms)
        {
            let destination = match parse_ethernet(&reply) {
                Some(answer) => self.route(&answer.dst, now_ms),
                None => SegmentDestination::Everyone,
            };
            self.for_segment.push((destination, reply));
        }
        if ethernet.dst != SERVER_MAC {
            self.deliver_to_guest(frame.to_vec());
        }
    }

    /// Lets time pass: announces this tab when due and forgets what aged out.
    pub fn tick(&mut self, now_ms: u64) {
        self.leases.purge(now_ms);
        let gone: Vec<u32> = self
            .peers
            .iter()
            .filter(|&(_, &seen)| seen + PEER_TIMEOUT_MS < now_ms)
            .map(|(&tab, _)| tab)
            .collect();
        for tab in &gone {
            self.peers.remove(tab);
        }
        self.stations.retain(|_, station| {
            !gone.contains(&station.tab) && station.seen_ms + STATION_TIMEOUT_MS >= now_ms
        });
        // This tab's own stations are in its table too: it may be the next server.
        for (&station, &address) in &self.claims {
            self.leases.claim(station, address, now_ms);
        }
        let due = self
            .last_announce_ms
            .is_none_or(|last| now_ms >= last + PRESENCE_INTERVAL_MS);
        if self.census_now {
            self.census_now = false;
            self.announce_now = false;
            self.announce(PRESENCE_CENSUS, now_ms);
        } else if self.announce_now || due {
            self.announce_now = false;
            self.announce(PRESENCE_ANNOUNCE, now_ms);
        }
    }

    fn serving(&self) -> bool {
        self.serving_since_ms.is_some()
    }

    fn route(&self, dst: &Mac, now_ms: u64) -> SegmentDestination {
        if is_group(dst) {
            return SegmentDestination::Everyone;
        }
        if *dst == SERVER_MAC {
            return SegmentDestination::Server;
        }
        match self.stations.get(dst) {
            Some(station) if station.seen_ms + STATION_TIMEOUT_MS >= now_ms => {
                SegmentDestination::Tab(station.tab)
            }
            _ => SegmentDestination::Everyone,
        }
    }

    /// Remembers which address a station behind this tab sends from: a guest configured by hand
    /// gets the same protection from the DHCP server as one it configured.
    fn observe_guest_address(&mut self, src: &Mac, ethertype: u16, payload: &[u8], now_ms: u64) {
        let address = match ethertype {
            ETHERTYPE_ARP => parse_arp(payload).map(|arp| arp.sender_ip),
            ETHERTYPE_IPV4 => parse_ipv4(payload).map(|ip| ip.src),
            _ => None,
        };
        if let Some(address) = address.filter(|&a| usable(a)) {
            self.claim(*src, address);
        }
        // Remembered as this tab's own, so its frames are not sent back out to the segment.
        if *src != self.mac {
            self.stations.insert(
                *src,
                Station {
                    tab: self.tab,
                    seen_ms: now_ms,
                },
            );
        }
    }

    fn claim(&mut self, station: Mac, address: Ipv4) {
        if self.claims.get(&station) == Some(&address) {
            return;
        }
        if self.claims.len() < CLAIMS_MAX || self.claims.contains_key(&station) {
            self.claims.insert(station, address);
            self.announce_now = true;
        }
    }

    /// Hands a frame to the guest, noting what a DHCP server told it.
    fn deliver_to_guest(&mut self, frame: Vec<u8>) {
        if let Some(reply) = dhcp_reply(&frame) {
            match reply.kind {
                DHCP_ACK if usable(reply.yiaddr) => self.claim(reply.chaddr, reply.yiaddr),
                DHCP_NAK if self.claims.remove(&reply.chaddr).is_some() => {
                    self.announce_now = true;
                }
                _ => {}
            }
        }
        self.for_guest.push(frame);
    }

    /// The server's answer to a frame addressed to it (or to everyone), if it has one.
    fn serve(&mut self, frame: &[u8], now_ms: u64) -> Option<Vec<u8>> {
        let ethernet = parse_ethernet(frame)?;
        match ethernet.ethertype {
            ETHERTYPE_ARP => {
                let arp = parse_arp(ethernet.payload)?;
                if arp.op != ARP_REQUEST || arp.target_ip != SERVER_IP {
                    return None;
                }
                let answer = arp_packet(
                    ARP_REPLY,
                    SERVER_MAC,
                    SERVER_IP,
                    arp.sender_mac,
                    arp.sender_ip,
                );
                Some(ethernet_frame(
                    ethernet.src,
                    SERVER_MAC,
                    ETHERTYPE_ARP,
                    &answer,
                ))
            }
            ETHERTYPE_IPV4 => {
                let ip = parse_ipv4(ethernet.payload)?;
                if ip.fragment {
                    return None;
                }
                match ip.protocol {
                    IP_PROTOCOL_ICMP if ip.dst == SERVER_IP && ethernet.dst == SERVER_MAC => {
                        let reply = icmp_echo_reply(ip.payload)?;
                        let packet = ipv4_packet(SERVER_IP, ip.src, IP_PROTOCOL_ICMP, &reply);
                        Some(ethernet_frame(
                            ethernet.src,
                            SERVER_MAC,
                            ETHERTYPE_IPV4,
                            &packet,
                        ))
                    }
                    IP_PROTOCOL_UDP => {
                        let udp = parse_udp(ip.payload)?;
                        if udp.dst_port != DHCP_SERVER_PORT || udp.src_port != DHCP_CLIENT_PORT {
                            return None;
                        }
                        let message = dhcp::parse_client_message(udp.data)?;
                        let may_offer = self.may_offer(now_ms);
                        dhcp::serve(&mut self.leases, &message, may_offer, now_ms)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Until it has heard the other tabs announce, a server does not know which addresses are
    /// taken, so it hands out none (a client retries its DISCOVER a few seconds later).
    fn may_offer(&self, now_ms: u64) -> bool {
        let listened = now_ms >= self.joined_ms + LISTEN_BEFORE_OFFERING_MS;
        let counted = self
            .serving_since_ms
            .is_some_and(|since| now_ms >= since + CENSUS_WAIT_MS);
        listened && counted
    }

    fn announce(&mut self, kind: u8, now_ms: u64) {
        self.last_announce_ms = Some(now_ms);
        let mut payload = vec![0u8; PRESENCE_HEADER_LEN];
        payload[PRESENCE_MAGIC_RANGE].copy_from_slice(&PRESENCE_MAGIC);
        payload[PRESENCE_VERSION_AT] = PRESENCE_VERSION;
        payload[PRESENCE_KIND_AT] = kind;
        if self.serving() {
            payload[PRESENCE_FLAGS_AT] |= PRESENCE_FLAG_SERVING;
        }
        payload[PRESENCE_COUNT_AT] = self.claims.len() as u8;
        for (station, address) in &self.claims {
            payload.extend_from_slice(station);
            payload.extend_from_slice(address);
        }
        let frame = ethernet_frame(BROADCAST_MAC, self.mac, ETHERTYPE_PRESENCE, &payload);
        self.for_segment.push((SegmentDestination::Everyone, frame));
    }

    fn hear_presence(&mut self, from_tab: u32, payload: &[u8], now_ms: u64) {
        if payload.get(PRESENCE_MAGIC_RANGE) != Some(&PRESENCE_MAGIC[..])
            || payload.get(PRESENCE_VERSION_AT) != Some(&PRESENCE_VERSION)
        {
            return;
        }
        self.peers.insert(from_tab, now_ms);
        if payload.get(PRESENCE_KIND_AT) == Some(&PRESENCE_CENSUS) {
            self.announce_now = true;
        }
        let count = payload.get(PRESENCE_COUNT_AT).copied().unwrap_or(0);
        let claims = payload.get(PRESENCE_HEADER_LEN..).unwrap_or_default();
        for claim in claims
            .as_chunks::<PRESENCE_CLAIM_LEN>()
            .0
            .iter()
            .take(usize::from(count))
        {
            let (station, address) = claim.split_at(size_of::<Mac>());
            let (Ok(station), Ok(address)) = (Mac::try_from(station), Ipv4::try_from(address))
            else {
                continue;
            };
            if is_group(&station) || !usable(address) {
                continue;
            }
            self.leases.claim(station, address, now_ms);
            // A frame from a station behind this tab says more than another tab's word.
            if self
                .stations
                .get(&station)
                .is_none_or(|s| s.tab != self.tab)
            {
                self.stations.insert(
                    station,
                    Station {
                        tab: from_tab,
                        seen_ms: now_ms,
                    },
                );
            }
        }
    }
}

/// The DHCP reply in a frame, if it is one (UDP from the server port to the client port).
fn dhcp_reply(frame: &[u8]) -> Option<dhcp::ServerMessage> {
    let ethernet = parse_ethernet(frame)?;
    if ethernet.ethertype != ETHERTYPE_IPV4 {
        return None;
    }
    let ip = parse_ipv4(ethernet.payload)?;
    if ip.protocol != IP_PROTOCOL_UDP || ip.fragment {
        return None;
    }
    let udp = parse_udp(ip.payload)?;
    if udp.src_port != DHCP_SERVER_PORT || udp.dst_port != DHCP_CLIENT_PORT {
        return None;
    }
    dhcp::parse_server_message(udp.data)
}

const _: () = assert!(PRESENCE_HEADER_LEN + CLAIMS_MAX * PRESENCE_CLAIM_LEN <= 1500);
const _: () = assert!(CLAIMS_MAX <= u8::MAX as usize);
