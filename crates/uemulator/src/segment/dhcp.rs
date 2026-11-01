//! The segment's DHCP server: its lease table and the answers to client messages (RFC 2131).
//!
//! Every agent keeps the table, not only the one serving: the presence frames fill it with the
//! addresses every tab's guests use, so the agent that takes over after the server's tab closes
//! already knows what not to hand out. Times are this agent's own clock (`now_ms`); nothing here
//! is ever compared with another tab's clock.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::BTreeMap;
use core::ops::Range;

use super::packet::{
    BROADCAST_MAC, ETHERTYPE_IPV4, IP_PROTOCOL_UDP, Ipv4, LIMITED_BROADCAST_IP, Mac,
    UNSPECIFIED_IP, ethernet_frame, ipv4_packet, udp_datagram,
};
use super::{
    POOL_FIRST, POOL_LAST, SERVER_IP, SERVER_MAC, SUBNET_BROADCAST, SUBNET_MASK, in_subnet,
};

pub(crate) const DHCP_SERVER_PORT: u16 = 67;
pub(crate) const DHCP_CLIENT_PORT: u16 = 68;

/// What the server grants. The guests renew at half of it; the table forgets an address this
/// long after the last sign that anyone uses it.
pub(crate) const LEASE_SECS: u32 = 60 * 60;
pub(crate) const LEASE_MS: u64 = LEASE_SECS as u64 * MS_PER_SEC;
const MS_PER_SEC: u64 = 1000;
/// RFC 2131 4.4.5: renew at half the lease, rebind at seven eighths.
const RENEWAL_SECS: u32 = LEASE_SECS / 2;
const REBINDING_SECS: u32 = LEASE_SECS / 8 * 7;
/// An offered address is kept for its client this long, so that two clients asking at once are
/// not offered the same one.
pub(crate) const OFFER_HOLD_MS: u64 = 30_000;
/// A lease refreshed this recently (a presence frame names it, or the server just granted it) is
/// in use by a live tab, and is never taken away to satisfy a conflicting claim or a full pool.
pub(crate) const LIVE_MS: u64 = super::PEER_TIMEOUT_MS;
/// Holds the addresses clients declined (another host answered ARP for them).
pub(crate) const DECLINED: Mac = [0; 6];

const BOOTP_OP: usize = 0;
const BOOTP_HTYPE: usize = 1;
const BOOTP_HLEN: usize = 2;
const BOOTP_XID: Range<usize> = 4..8;
const BOOTP_FLAGS: Range<usize> = 10..12;
const BOOTP_CIADDR: Range<usize> = 12..16;
const BOOTP_YIADDR: Range<usize> = 16..20;
const BOOTP_SIADDR: Range<usize> = 20..24;
const BOOTP_CHADDR: Range<usize> = 28..34;
const BOOTP_MAGIC_COOKIE: Range<usize> = 236..240;
const DHCP_OPTIONS_OFFSET: usize = 240;
const DHCP_MAGIC_COOKIE: [u8; 4] = [0x63, 0x82, 0x53, 0x63];
const BOOTP_REQUEST: u8 = 1;
const BOOTP_REPLY: u8 = 2;
const HTYPE_ETHERNET: u8 = 1;
const HLEN_ETHERNET: u8 = 6;
/// The client asks for replies to the limited broadcast address (it cannot take unicast before
/// it has an address).
const BOOTP_FLAG_BROADCAST: u16 = 0x8000;
/// RFC 1542: relays and some clients drop BOOTP messages shorter than this.
const BOOTP_MIN_LEN: usize = 300;

const OPTION_PAD: u8 = 0;
const OPTION_SUBNET_MASK: u8 = 1;
const OPTION_BROADCAST_ADDRESS: u8 = 28;
const OPTION_REQUESTED_IP: u8 = 50;
const OPTION_LEASE_TIME: u8 = 51;
const OPTION_MESSAGE_TYPE: u8 = 53;
const OPTION_SERVER_ID: u8 = 54;
const OPTION_RENEWAL_TIME: u8 = 58;
const OPTION_REBINDING_TIME: u8 = 59;
const OPTION_END: u8 = 255;

pub(crate) const DHCP_DISCOVER: u8 = 1;
pub(crate) const DHCP_OFFER: u8 = 2;
pub(crate) const DHCP_REQUEST: u8 = 3;
pub(crate) const DHCP_DECLINE: u8 = 4;
pub(crate) const DHCP_ACK: u8 = 5;
pub(crate) const DHCP_NAK: u8 = 6;
pub(crate) const DHCP_RELEASE: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lease {
    address: Ipv4,
    expires_ms: u64,
    /// When something last showed the address is in use: a grant or a presence claim.
    refreshed_ms: u64,
}

#[derive(Default)]
pub(crate) struct Leases {
    by_station: BTreeMap<Mac, Lease>,
}

impl Leases {
    fn live(&self, now_ms: u64) -> impl Iterator<Item = (&Mac, &Lease)> {
        self.by_station
            .iter()
            .filter(move |(_, lease)| lease.expires_ms > now_ms)
    }

    pub(crate) fn holder(&self, address: Ipv4, now_ms: u64) -> Option<Mac> {
        self.live(now_ms)
            .find(|(_, lease)| lease.address == address)
            .map(|(mac, _)| *mac)
    }

    pub(crate) fn address_of(&self, station: &Mac, now_ms: u64) -> Option<Ipv4> {
        self.by_station
            .get(station)
            .filter(|lease| lease.expires_ms > now_ms)
            .map(|lease| lease.address)
    }

    fn free_for(&self, address: Ipv4, station: &Mac, now_ms: u64) -> bool {
        in_pool(address)
            && self
                .holder(address, now_ms)
                .is_none_or(|holder| holder == *station)
    }

    /// The address to offer `station`: the one it has, else the one it asks for if that is free,
    /// else the lowest free one. With the pool full, the lease nobody has shown to be in use for
    /// longest is taken back. `None` when every address is in live use.
    fn pick(&mut self, station: &Mac, requested: Option<Ipv4>, now_ms: u64) -> Option<Ipv4> {
        if let Some(address) = self.address_of(station, now_ms) {
            return Some(address);
        }
        if let Some(address) = requested.filter(|&a| self.free_for(a, station, now_ms)) {
            return Some(address);
        }
        if let Some(address) = pool().find(|&a| self.free_for(a, station, now_ms)) {
            return Some(address);
        }
        let (stale, _) = self
            .live(now_ms)
            .filter(|(_, lease)| lease.refreshed_ms + LIVE_MS < now_ms && in_pool(lease.address))
            .min_by_key(|(_, lease)| lease.refreshed_ms)?;
        let stale = *stale;
        self.by_station.remove(&stale).map(|lease| lease.address)
    }

    fn set(&mut self, station: Mac, address: Ipv4, expires_ms: u64, now_ms: u64) {
        // One address, one station: whoever had it loses it.
        self.by_station
            .retain(|mac, lease| *mac == station || lease.address != address);
        self.by_station.insert(
            station,
            Lease {
                address,
                expires_ms,
                refreshed_ms: now_ms,
            },
        );
    }

    fn offer(&mut self, station: Mac, address: Ipv4, now_ms: u64) {
        let expires_ms = match self.by_station.get(&station) {
            Some(lease) if lease.address == address => lease.expires_ms.max(now_ms + OFFER_HOLD_MS),
            _ => now_ms + OFFER_HOLD_MS,
        };
        self.set(station, address, expires_ms, now_ms);
    }

    fn bind(&mut self, station: Mac, address: Ipv4, now_ms: u64) {
        self.set(station, address, now_ms + LEASE_MS, now_ms);
    }

    /// A tab says `station` behind it uses `address` (it saw the guest get it, or use it). The
    /// claim wins over anything but another station's lease that is itself in live use: two
    /// live claims to one address are a conflict the guests have to settle (the loser's next
    /// renewal is refused), and flapping between them would only spread it.
    pub(crate) fn claim(&mut self, station: Mac, address: Ipv4, now_ms: u64) {
        if let Some(holder) = self.holder(address, now_ms)
            && holder != station
            && self.by_station[&holder].refreshed_ms + LIVE_MS >= now_ms
        {
            return;
        }
        self.bind(station, address, now_ms);
    }

    fn release(&mut self, station: &Mac, address: Ipv4) {
        if self
            .by_station
            .get(station)
            .is_some_and(|lease| lease.address == address)
        {
            self.by_station.remove(station);
        }
    }

    pub(crate) fn purge(&mut self, now_ms: u64) {
        self.by_station.retain(|_, lease| lease.expires_ms > now_ms);
    }
}

fn in_pool(address: Ipv4) -> bool {
    in_subnet(address) && (POOL_FIRST..=POOL_LAST).contains(&address[3])
}

fn pool() -> impl Iterator<Item = Ipv4> {
    (POOL_FIRST..=POOL_LAST).map(|host| [SERVER_IP[0], SERVER_IP[1], SERVER_IP[2], host])
}

/// A client's message, as much of it as the server looks at.
pub(crate) struct ClientMessage {
    pub kind: u8,
    xid: [u8; 4],
    flags: u16,
    ciaddr: Ipv4,
    pub chaddr: Mac,
    requested: Option<Ipv4>,
    server_id: Option<Ipv4>,
}

/// A server's reply, as much of it as an agent snooping its own guest's traffic looks at.
pub(crate) struct ServerMessage {
    pub kind: u8,
    pub chaddr: Mac,
    pub yiaddr: Ipv4,
}

fn options(message: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    let mut rest = message.get(DHCP_OPTIONS_OFFSET..).unwrap_or_default();
    core::iter::from_fn(move || {
        loop {
            let (&code, after) = rest.split_first()?;
            match code {
                OPTION_PAD => rest = after,
                OPTION_END => return None,
                _ => {
                    let (&len, after) = after.split_first()?;
                    let value = after.get(..usize::from(len))?;
                    rest = &after[usize::from(len)..];
                    return Some((code, value));
                }
            }
        }
    })
}

fn option(message: &[u8], code: u8) -> Option<&[u8]> {
    options(message)
        .find(|&(c, _)| c == code)
        .map(|(_, value)| value)
}

fn address_option(message: &[u8], code: u8) -> Option<Ipv4> {
    option(message, code)?.try_into().ok()
}

fn bootp_header(message: &[u8], op: u8) -> Option<(u8, Mac)> {
    if *message.get(BOOTP_OP)? != op
        || *message.get(BOOTP_HTYPE)? != HTYPE_ETHERNET
        || *message.get(BOOTP_HLEN)? != HLEN_ETHERNET
        || message.get(BOOTP_MAGIC_COOKIE)? != DHCP_MAGIC_COOKIE
    {
        return None;
    }
    let kind = *option(message, OPTION_MESSAGE_TYPE)?.first()?;
    Some((kind, message.get(BOOTP_CHADDR)?.try_into().ok()?))
}

pub(crate) fn parse_client_message(message: &[u8]) -> Option<ClientMessage> {
    let (kind, chaddr) = bootp_header(message, BOOTP_REQUEST)?;
    Some(ClientMessage {
        kind,
        xid: message.get(BOOTP_XID)?.try_into().ok()?,
        flags: u16::from_be_bytes(message.get(BOOTP_FLAGS)?.try_into().ok()?),
        ciaddr: message.get(BOOTP_CIADDR)?.try_into().ok()?,
        chaddr,
        requested: address_option(message, OPTION_REQUESTED_IP),
        server_id: address_option(message, OPTION_SERVER_ID),
    })
}

pub(crate) fn parse_server_message(message: &[u8]) -> Option<ServerMessage> {
    let (kind, chaddr) = bootp_header(message, BOOTP_REPLY)?;
    Some(ServerMessage {
        kind,
        chaddr,
        yiaddr: message.get(BOOTP_YIADDR)?.try_into().ok()?,
    })
}

/// The server's answer to `message` as a whole Ethernet frame, or `None` when it has none.
/// `may_offer` is false while the server has not yet heard every tab's claims (just after it
/// started or took over); it then answers only clients that already hold an address.
pub(crate) fn serve(
    leases: &mut Leases,
    message: &ClientMessage,
    may_offer: bool,
    now_ms: u64,
) -> Option<Vec<u8>> {
    let station = message.chaddr;
    match message.kind {
        DHCP_DISCOVER => {
            if !may_offer {
                return None;
            }
            let address = leases.pick(&station, message.requested, now_ms)?;
            leases.offer(station, address, now_ms);
            Some(reply(message, DHCP_OFFER, address))
        }
        DHCP_REQUEST => {
            if let Some(server) = message.server_id
                && server != SERVER_IP
            {
                // It took another server's offer. There is no other, but the RFC is clear.
                if let Some(address) = leases.address_of(&station, now_ms) {
                    leases.release(&station, address);
                }
                return None;
            }
            // SELECTING and INIT-REBOOT name the address in an option, RENEWING and REBINDING
            // in ciaddr. A client that comes back with an address the table has no record of
            // (it got it from the tab that served before this one) gets it if nobody else has it.
            let wanted = message
                .requested
                .or(Some(message.ciaddr))
                .filter(|&a| a != UNSPECIFIED_IP)?;
            if leases.free_for(wanted, &station, now_ms) {
                leases.bind(station, wanted, now_ms);
                Some(reply(message, DHCP_ACK, wanted))
            } else {
                Some(reply(message, DHCP_NAK, UNSPECIFIED_IP))
            }
        }
        DHCP_RELEASE => {
            leases.release(&station, message.ciaddr);
            None
        }
        DHCP_DECLINE => {
            if let Some(address) = message.requested.filter(|&a| in_pool(a)) {
                // Someone answered ARP for it, so it is in use by a host the table does not
                // know; keep it out of the pool for a lease time.
                leases.bind(DECLINED, address, now_ms);
            }
            None
        }
        _ => None,
    }
}

/// RFC 2131 4.1: a NAK and anything for a client that asked for broadcast go to the limited
/// broadcast address; a renewing client (ciaddr set) gets unicast to that address; otherwise
/// unicast to the address being offered, at the client's MAC (it accepts it before it has
/// configured the address).
fn reply(message: &ClientMessage, kind: u8, yiaddr: Ipv4) -> Vec<u8> {
    let mut bootp = vec![0u8; DHCP_OPTIONS_OFFSET];
    bootp[BOOTP_OP] = BOOTP_REPLY;
    bootp[BOOTP_HTYPE] = HTYPE_ETHERNET;
    bootp[BOOTP_HLEN] = HLEN_ETHERNET;
    bootp[BOOTP_XID].copy_from_slice(&message.xid);
    bootp[BOOTP_FLAGS].copy_from_slice(&message.flags.to_be_bytes());
    if kind != DHCP_NAK {
        bootp[BOOTP_CIADDR].copy_from_slice(&message.ciaddr);
        bootp[BOOTP_YIADDR].copy_from_slice(&yiaddr);
        bootp[BOOTP_SIADDR].copy_from_slice(&SERVER_IP);
    }
    bootp[BOOTP_CHADDR].copy_from_slice(&message.chaddr);
    bootp[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
    let mut put = |code: u8, value: &[u8]| {
        bootp.push(code);
        bootp.push(value.len() as u8);
        bootp.extend_from_slice(value);
    };
    put(OPTION_MESSAGE_TYPE, &[kind]);
    put(OPTION_SERVER_ID, &SERVER_IP);
    if kind != DHCP_NAK {
        put(OPTION_LEASE_TIME, &LEASE_SECS.to_be_bytes());
        put(OPTION_RENEWAL_TIME, &RENEWAL_SECS.to_be_bytes());
        put(OPTION_REBINDING_TIME, &REBINDING_SECS.to_be_bytes());
        put(OPTION_SUBNET_MASK, &SUBNET_MASK);
        put(OPTION_BROADCAST_ADDRESS, &SUBNET_BROADCAST);
    }
    bootp.push(OPTION_END);
    if bootp.len() < BOOTP_MIN_LEN {
        bootp.resize(BOOTP_MIN_LEN, OPTION_PAD);
    }

    let broadcast = kind == DHCP_NAK
        || (message.ciaddr == UNSPECIFIED_IP && message.flags & BOOTP_FLAG_BROADCAST != 0);
    let (dst_ip, dst_mac) = if broadcast {
        (LIMITED_BROADCAST_IP, BROADCAST_MAC)
    } else if message.ciaddr != UNSPECIFIED_IP {
        (message.ciaddr, message.chaddr)
    } else {
        (yiaddr, message.chaddr)
    };
    let udp = udp_datagram(
        SERVER_IP,
        dst_ip,
        DHCP_SERVER_PORT,
        DHCP_CLIENT_PORT,
        &bootp,
    );
    let ip = ipv4_packet(SERVER_IP, dst_ip, IP_PROTOCOL_UDP, &udp);
    ethernet_frame(dst_mac, SERVER_MAC, ETHERTYPE_IPV4, &ip)
}

/// A client's message, for the tests' guests: DISCOVER, REQUEST (`requested` for SELECTING and
/// INIT-REBOOT, `ciaddr` for RENEWING), RELEASE or DECLINE.
#[cfg(test)]
pub(crate) fn client_message(
    kind: u8,
    chaddr: Mac,
    xid: u32,
    ciaddr: Ipv4,
    requested: Option<Ipv4>,
    server_id: Option<Ipv4>,
) -> Vec<u8> {
    let mut bootp = vec![0u8; DHCP_OPTIONS_OFFSET];
    bootp[BOOTP_OP] = BOOTP_REQUEST;
    bootp[BOOTP_HTYPE] = HTYPE_ETHERNET;
    bootp[BOOTP_HLEN] = HLEN_ETHERNET;
    bootp[BOOTP_XID].copy_from_slice(&xid.to_be_bytes());
    bootp[BOOTP_CIADDR].copy_from_slice(&ciaddr);
    bootp[BOOTP_CHADDR].copy_from_slice(&chaddr);
    bootp[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
    bootp.extend_from_slice(&[OPTION_MESSAGE_TYPE, 1, kind]);
    if let Some(address) = requested {
        bootp.extend_from_slice(&[OPTION_REQUESTED_IP, 4]);
        bootp.extend_from_slice(&address);
    }
    if let Some(address) = server_id {
        bootp.extend_from_slice(&[OPTION_SERVER_ID, 4]);
        bootp.extend_from_slice(&address);
    }
    bootp.push(OPTION_END);
    bootp.resize(BOOTP_MIN_LEN, OPTION_PAD);
    let src_ip = ciaddr;
    let dst_ip = if ciaddr == UNSPECIFIED_IP {
        LIMITED_BROADCAST_IP
    } else {
        SERVER_IP
    };
    let dst_mac = if ciaddr == UNSPECIFIED_IP {
        BROADCAST_MAC
    } else {
        SERVER_MAC
    };
    let udp = udp_datagram(src_ip, dst_ip, DHCP_CLIENT_PORT, DHCP_SERVER_PORT, &bootp);
    let ip = ipv4_packet(src_ip, dst_ip, IP_PROTOCOL_UDP, &udp);
    ethernet_frame(dst_mac, chaddr, ETHERTYPE_IPV4, &ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Mac = [0x02, 0, 0, 0, 0, 0xa];
    const B: Mac = [0x02, 0, 0, 0, 0, 0xb];
    const START: u64 = 1_000_000;

    fn host(n: u8) -> Ipv4 {
        [10, 0, 2, n]
    }

    #[test]
    fn the_lowest_free_address_is_picked_and_held_by_its_offer() {
        let mut leases = Leases::default();
        let first = leases.pick(&A, None, START).unwrap();
        leases.offer(A, first, START);
        let second = leases.pick(&B, None, START).unwrap();
        assert_eq!((first, second), (host(POOL_FIRST), host(POOL_FIRST + 1)));
    }

    #[test]
    fn an_offer_not_taken_up_frees_its_address() {
        let mut leases = Leases::default();
        leases.offer(A, host(POOL_FIRST), START);
        let later = START + OFFER_HOLD_MS + 1;
        assert_eq!(leases.pick(&B, None, later), Some(host(POOL_FIRST)));
    }

    #[test]
    fn a_requested_address_is_honoured_only_when_free_and_in_the_pool() {
        let mut leases = Leases::default();
        leases.bind(A, host(40), START);
        assert_eq!(
            leases.pick(&B, Some(host(40)), START),
            Some(host(POOL_FIRST))
        );
        assert_eq!(leases.pick(&B, Some(host(41)), START), Some(host(41)));
        assert_eq!(
            leases.pick(&B, Some(host(2)), START),
            Some(host(POOL_FIRST))
        );
        assert_eq!(
            leases.pick(&B, Some([192, 168, 1, 41]), START),
            Some(host(POOL_FIRST))
        );
    }

    #[test]
    fn a_full_pool_takes_back_the_lease_longest_out_of_use() {
        let mut leases = Leases::default();
        for (index, host_part) in (POOL_FIRST..=POOL_LAST).enumerate() {
            leases.bind(
                [0x02, 1, 0, 0, 0, host_part],
                host(host_part),
                START + index as u64,
            );
        }
        let now = START + LIVE_MS + POOL_LAST as u64 + 1;
        assert_eq!(leases.pick(&A, None, now), Some(host(POOL_FIRST)));
    }

    #[test]
    fn a_full_pool_of_live_leases_offers_nothing() {
        let mut leases = Leases::default();
        for host_part in POOL_FIRST..=POOL_LAST {
            leases.bind([0x02, 1, 0, 0, 0, host_part], host(host_part), START);
        }
        assert_eq!(leases.pick(&A, None, START + 1), None);
    }

    #[test]
    fn a_claim_does_not_take_an_address_from_a_live_holder() {
        let mut leases = Leases::default();
        leases.bind(A, host(20), START);
        leases.claim(B, host(20), START + 1);
        assert_eq!(leases.holder(host(20), START + 1), Some(A));
        // Once A's lease has gone unconfirmed for a while, B's claim stands.
        let later = START + LIVE_MS + 1;
        leases.claim(B, host(20), later);
        assert_eq!(leases.holder(host(20), later), Some(B));
        assert_eq!(leases.address_of(&A, later), None);
    }

    #[test]
    fn options_are_found_past_padding_and_a_truncated_option_ends_the_walk() {
        let message = client_message(DHCP_DISCOVER, A, 1, UNSPECIFIED_IP, Some(host(30)), None);
        let bootp = &message[14 + 20 + 8..];
        let parsed = parse_client_message(bootp).unwrap();
        assert_eq!(
            (parsed.kind, parsed.requested),
            (DHCP_DISCOVER, Some(host(30)))
        );

        let mut truncated = bootp[..DHCP_OPTIONS_OFFSET].to_vec();
        truncated.extend_from_slice(&[OPTION_PAD, OPTION_MESSAGE_TYPE, 1, DHCP_REQUEST]);
        truncated.extend_from_slice(&[OPTION_REQUESTED_IP, 4, 10, 0]);
        let parsed = parse_client_message(&truncated).unwrap();
        assert_eq!((parsed.kind, parsed.requested), (DHCP_REQUEST, None));
    }

    #[test]
    fn a_message_without_the_magic_cookie_is_not_dhcp() {
        let mut message = client_message(DHCP_DISCOVER, A, 1, UNSPECIFIED_IP, None, None);
        let bootp_start = 14 + 20 + 8;
        message[bootp_start + BOOTP_MAGIC_COOKIE.start] ^= 1;
        assert!(parse_client_message(&message[bootp_start..]).is_none());
    }
}
