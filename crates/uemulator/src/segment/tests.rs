//! Agents on a fake transport that behaves like the page's: a frame for `Everyone` reaches every
//! other tab, one for `Tab(n)` only tab n, one for `Server` only the tab the election picked.
//! The guests are scripted: DHCP clients, ARP, ping and plain frames between them.

use super::dhcp::{
    DHCP_ACK, DHCP_DECLINE, DHCP_DISCOVER, DHCP_OFFER, DHCP_RELEASE, DHCP_REQUEST, client_message,
    options,
};
use super::packet::{
    UNSPECIFIED_IP, icmp_echo_request, is_icmp_echo_reply, parse_arp, udp_datagram,
};
use super::*;

const TICK_MS: u64 = 100;
/// Long enough for every agent to have announced itself a few times and for a server to offer.
const SETTLE_MS: u64 = 3 * PRESENCE_INTERVAL_MS;
const START_MS: u64 = 5_000;
const CHAT_PORT: u16 = 5000;

fn mac_of(tab: u32) -> Mac {
    [0x02, 0x5e, 0, 0, 0, tab as u8]
}

fn host(n: u8) -> Ipv4 {
    [10, 0, 2, n]
}

struct Tab {
    agent: SegmentAgent,
    /// Everything the agent handed its guest, in order.
    guest: Vec<Vec<u8>>,
    /// Everything the transport handed this tab, in order.
    received: Vec<Vec<u8>>,
}

struct Bus {
    tabs: BTreeMap<u32, Tab>,
    server: Option<u32>,
    now_ms: u64,
}

impl Bus {
    fn new() -> Self {
        Bus {
            tabs: BTreeMap::new(),
            server: None,
            now_ms: START_MS,
        }
    }

    /// A tab joins; like the page, the first one to ask for the lock serves.
    fn join(&mut self, tab: u32) {
        let agent = SegmentAgent::new(mac_of(tab), tab, self.now_ms).unwrap();
        self.tabs.insert(
            tab,
            Tab {
                agent,
                guest: Vec::new(),
                received: Vec::new(),
            },
        );
        if self.server.is_none() {
            self.serve(tab);
        }
    }

    fn serve(&mut self, tab: u32) {
        self.server = Some(tab);
        let now = self.now_ms;
        self.tabs
            .get_mut(&tab)
            .unwrap()
            .agent
            .set_serving(true, now);
    }

    /// A tab closes. The lock passes to the tab that has waited longest, which is the lowest
    /// numbered one here.
    fn leave(&mut self, tab: u32) {
        self.tabs.remove(&tab);
        if self.server == Some(tab) {
            self.server = None;
            if let Some(&next) = self.tabs.keys().next() {
                self.serve(next);
            }
        }
    }

    fn tab(&mut self, tab: u32) -> &mut Tab {
        self.tabs.get_mut(&tab).unwrap()
    }

    fn guest_sends(&mut self, tab: u32, frame: Vec<u8>) {
        let now = self.now_ms;
        self.tab(tab).agent.from_guest(&frame, now);
        self.pump();
    }

    /// Delivers until nothing is in flight.
    fn pump(&mut self) {
        loop {
            let mut moved = false;
            let ids: Vec<u32> = self.tabs.keys().copied().collect();
            for from in ids {
                let tab = self.tab(from);
                let for_guest = tab.agent.take_for_guest();
                tab.guest.extend(for_guest);
                for (destination, frame) in tab.agent.take_for_segment() {
                    moved = true;
                    let targets: Vec<u32> = match destination {
                        SegmentDestination::Everyone => {
                            self.tabs.keys().copied().filter(|&t| t != from).collect()
                        }
                        SegmentDestination::Tab(t) => vec![t],
                        SegmentDestination::Server => self.server.into_iter().collect(),
                    };
                    let now = self.now_ms;
                    for target in targets {
                        if let Some(tab) = self.tabs.get_mut(&target) {
                            tab.received.push(frame.clone());
                            tab.agent.from_segment(from, &frame, now);
                        }
                    }
                }
            }
            if !moved {
                break;
            }
        }
    }

    fn advance(&mut self, ms: u64) {
        let end = self.now_ms + ms;
        while self.now_ms < end {
            self.now_ms += TICK_MS;
            let now = self.now_ms;
            for tab in self.tabs.values_mut() {
                tab.agent.tick(now);
            }
            self.pump();
        }
    }

    /// Every agent's `tick` without time passing: only what is due right now goes out.
    fn tick_in_place(&mut self) {
        let now = self.now_ms;
        for tab in self.tabs.values_mut() {
            tab.agent.tick(now);
        }
        self.pump();
    }

    fn take_guest(&mut self, tab: u32) -> Vec<Vec<u8>> {
        core::mem::take(&mut self.tab(tab).guest)
    }

    fn clear_received(&mut self) {
        for tab in self.tabs.values_mut() {
            tab.received.clear();
            tab.guest.clear();
        }
    }

    /// The DHCP replies the guest of `tab` got for `station`, as (kind, yiaddr).
    fn dhcp_replies(&mut self, tab: u32, station: Mac) -> Vec<(u8, Ipv4)> {
        self.take_guest(tab)
            .iter()
            .filter_map(|frame| dhcp_reply(frame))
            .filter(|reply| reply.chaddr == station)
            .map(|reply| (reply.kind, reply.yiaddr))
            .collect()
    }

    /// The guest of `tab` runs DHCP for its NIC, the way a client in INIT does.
    fn dhcp(&mut self, tab: u32) -> Option<Ipv4> {
        self.dhcp_for(tab, mac_of(tab))
    }

    fn dhcp_for(&mut self, tab: u32, station: Mac) -> Option<Ipv4> {
        self.take_guest(tab);
        let discover = client_message(DHCP_DISCOVER, station, 1, UNSPECIFIED_IP, None, None);
        self.guest_sends(tab, discover);
        let offered = self
            .dhcp_replies(tab, station)
            .into_iter()
            .find(|&(kind, _)| kind == DHCP_OFFER)?
            .1;
        let request = client_message(
            DHCP_REQUEST,
            station,
            2,
            UNSPECIFIED_IP,
            Some(offered),
            Some(SERVER_IP),
        );
        self.guest_sends(tab, request);
        let acked = self
            .dhcp_replies(tab, station)
            .into_iter()
            .find(|&(kind, _)| kind == DHCP_ACK)?
            .1;
        assert_eq!(acked, offered);
        // Let the tab announce its new address.
        self.advance(TICK_MS);
        Some(acked)
    }

    /// A REQUEST without a server identifier: INIT-REBOOT (`ciaddr` unset, the address in an
    /// option) or RENEWING (`ciaddr` set). The reply's kind.
    fn request_again(&mut self, tab: u32, address: Ipv4, renewing: bool) -> Option<u8> {
        self.take_guest(tab);
        let message = if renewing {
            client_message(DHCP_REQUEST, mac_of(tab), 3, address, None, None)
        } else {
            client_message(
                DHCP_REQUEST,
                mac_of(tab),
                3,
                UNSPECIFIED_IP,
                Some(address),
                None,
            )
        };
        self.guest_sends(tab, message);
        self.dhcp_replies(tab, mac_of(tab))
            .first()
            .map(|&(kind, _)| kind)
    }
}

fn arp_request(sender_mac: Mac, sender_ip: Ipv4, target_ip: Ipv4) -> Vec<u8> {
    let arp = arp_packet(ARP_REQUEST, sender_mac, sender_ip, [0; 6], target_ip);
    ethernet_frame(BROADCAST_MAC, sender_mac, ETHERTYPE_ARP, &arp)
}

fn ping(src_mac: Mac, dst_mac: Mac, src: Ipv4, dst: Ipv4) -> Vec<u8> {
    let icmp = icmp_echo_request(7, 1, b"abcdefgh");
    let ip = ipv4_packet(src, dst, IP_PROTOCOL_ICMP, &icmp);
    ethernet_frame(dst_mac, src_mac, ETHERTYPE_IPV4, &ip)
}

fn chat(src_mac: Mac, dst_mac: Mac, src: Ipv4, dst: Ipv4, text: &[u8]) -> Vec<u8> {
    let udp = udp_datagram(src, dst, CHAT_PORT, CHAT_PORT, text);
    let ip = ipv4_packet(src, dst, IP_PROTOCOL_UDP, &udp);
    ethernet_frame(dst_mac, src_mac, ETHERTYPE_IPV4, &ip)
}

fn ethertype_of(frame: &[u8]) -> u16 {
    parse_ethernet(frame).unwrap().ethertype
}

/// Two tabs on a settled segment, both with a DHCP lease: (server's address, other's address).
fn two_leased_tabs(bus: &mut Bus) -> (Ipv4, Ipv4) {
    bus.join(1);
    bus.join(2);
    bus.advance(SETTLE_MS);
    let first = bus.dhcp(1).unwrap();
    let second = bus.dhcp(2).unwrap();
    (first, second)
}

mod addresses {
    use super::*;

    type OptionList = Vec<(u8, Vec<u8>)>;

    /// The options of every DHCP reply in `frames`, as (kind, options).
    fn reply_options(frames: &[Vec<u8>]) -> Vec<(u8, OptionList)> {
        frames
            .iter()
            .filter_map(|frame| {
                let reply = dhcp_reply(frame)?;
                let ethernet = parse_ethernet(frame)?;
                let ip = parse_ipv4(ethernet.payload)?;
                let udp = parse_udp(ip.payload)?;
                let all = options(udp.data)
                    .map(|(code, value)| (code, value.to_vec()))
                    .collect();
                Some((reply.kind, all))
            })
            .collect()
    }

    #[test]
    fn offers_and_acks_name_the_server_as_router_and_dns() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.serve(1);
        bus.join(2);
        bus.advance(SETTLE_MS);
        bus.take_guest(2);
        let station = mac_of(2);
        let discover = client_message(DHCP_DISCOVER, station, 1, UNSPECIFIED_IP, None, None);
        bus.guest_sends(2, discover);
        let offer_frames = bus.take_guest(2);
        let offers = reply_options(&offer_frames);
        let offered = offer_frames
            .iter()
            .filter_map(|frame| dhcp_reply(frame))
            .find(|reply| reply.kind == DHCP_OFFER)
            .map(|reply| reply.yiaddr);
        let request = client_message(
            DHCP_REQUEST,
            station,
            2,
            UNSPECIFIED_IP,
            offered,
            Some(SERVER_IP),
        );
        bus.guest_sends(2, request);
        let acks = reply_options(&bus.take_guest(2));
        for (kind, replies) in [(DHCP_OFFER, offers), (DHCP_ACK, acks)] {
            let (_, options) = replies
                .iter()
                .find(|(k, _)| *k == kind)
                .expect("the server answers");
            assert!(
                options.contains(&(3, SERVER_IP.to_vec())),
                "router in {kind}"
            );
            assert!(options.contains(&(6, SERVER_IP.to_vec())), "dns in {kind}");
        }
    }

    #[test]
    fn the_server_tab_and_another_tab_get_distinct_addresses_in_the_subnet() {
        let mut bus = Bus::new();
        let (first, second) = two_leased_tabs(&mut bus);
        assert_eq!(first, host(POOL_FIRST));
        assert_eq!(second, host(POOL_FIRST + 1));
        assert_eq!(bus.tab(1).agent.status().address, Some(first));
        assert_eq!(bus.tab(2).agent.status().address, Some(second));
    }

    #[test]
    fn ten_tabs_get_ten_different_addresses() {
        let mut bus = Bus::new();
        for tab in 1..=10 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        let mut addresses: Vec<Ipv4> = (1..=10).map(|tab| bus.dhcp(tab).unwrap()).collect();
        addresses.sort();
        addresses.dedup();
        assert_eq!(addresses.len(), 10);
        assert!(
            addresses
                .iter()
                .all(|&a| in_subnet(a) && a[3] >= POOL_FIRST)
        );
    }

    #[test]
    fn asking_again_keeps_the_address() {
        let mut bus = Bus::new();
        let (_, second) = two_leased_tabs(&mut bus);
        assert_eq!(bus.dhcp(2), Some(second));
        assert_eq!(bus.request_again(2, second, true), Some(DHCP_ACK));
    }

    #[test]
    fn a_just_started_server_offers_nothing_until_it_has_listened() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        assert_eq!(bus.dhcp(2), None);
        bus.advance(SETTLE_MS);
        assert!(bus.dhcp(2).is_some());
    }

    #[test]
    fn an_address_in_use_by_another_station_is_refused() {
        let mut bus = Bus::new();
        let (first, _) = two_leased_tabs(&mut bus);
        bus.join(3);
        bus.advance(SETTLE_MS);
        assert_eq!(bus.request_again(3, first, false), Some(DHCP_NAK));
        assert_eq!(
            bus.request_again(3, [192, 168, 0, 9], false),
            Some(DHCP_NAK)
        );
    }

    #[test]
    fn a_refused_address_is_no_longer_announced() {
        let mut bus = Bus::new();
        let (first, _) = two_leased_tabs(&mut bus);
        bus.join(3);
        bus.advance(SETTLE_MS);
        // Tab 3's guest used tab 1's address by hand, then asked DHCP to confirm it.
        bus.guest_sends(3, arp_request(mac_of(3), first, first));
        assert_eq!(bus.tab(3).agent.status().address, Some(first));
        assert_eq!(bus.request_again(3, first, false), Some(DHCP_NAK));
        assert_eq!(bus.tab(3).agent.status().address, None);
        // Tab 1 keeps it all along.
        bus.advance(SETTLE_MS);
        assert_eq!(bus.request_again(1, first, true), Some(DHCP_ACK));
    }

    #[test]
    fn an_address_configured_by_hand_is_not_handed_out() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        bus.advance(SETTLE_MS);
        // Tab 2's guest announces 10.0.2.15 itself (a gratuitous ARP), with no DHCP.
        bus.guest_sends(
            2,
            arp_request(mac_of(2), host(POOL_FIRST), host(POOL_FIRST)),
        );
        bus.advance(SETTLE_MS);
        assert_eq!(bus.dhcp(1), Some(host(POOL_FIRST + 1)));
    }

    #[test]
    fn a_released_address_is_offered_again() {
        let mut bus = Bus::new();
        let (_, second) = two_leased_tabs(&mut bus);
        let release = client_message(DHCP_RELEASE, mac_of(2), 9, second, None, Some(SERVER_IP));
        bus.guest_sends(2, release);
        bus.leave(2);
        bus.join(3);
        bus.advance(SETTLE_MS);
        assert_eq!(bus.dhcp(3), Some(second));
    }

    #[test]
    fn a_declined_address_is_not_offered_again() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.advance(SETTLE_MS);
        let decline = client_message(
            DHCP_DECLINE,
            mac_of(1),
            9,
            UNSPECIFIED_IP,
            Some(host(POOL_FIRST)),
            Some(SERVER_IP),
        );
        bus.guest_sends(1, decline);
        assert_eq!(bus.dhcp(1), Some(host(POOL_FIRST + 1)));
    }

    #[test]
    fn a_station_behind_a_guest_gets_its_own_lease_and_keeps_it_across_a_handover() {
        // A Xen domU or a container behind a bridge in tab 2's guest: its own MAC, its own lease.
        const BRIDGED: Mac = [0x02, 0x16, 0x3e, 0, 0, 1];
        let mut bus = Bus::new();
        let (_, second) = two_leased_tabs(&mut bus);
        let bridged = bus.dhcp_for(2, BRIDGED).unwrap();
        assert_ne!(bridged, second);
        bus.leave(1);
        bus.join(3);
        bus.advance(SETTLE_MS);
        let third = bus.dhcp(3).unwrap();
        assert!(third != bridged && third != second);
    }
}

mod handover {
    use super::*;

    #[test]
    fn a_new_server_knows_every_lease_and_hands_out_a_free_address() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        bus.join(3);
        bus.advance(SETTLE_MS);
        let leased: Vec<Ipv4> = (1..=3).map(|tab| bus.dhcp(tab).unwrap()).collect();
        bus.leave(1);
        assert_eq!(bus.server, Some(2));
        assert!(bus.tab(2).agent.status().serving);
        bus.join(4);
        bus.advance(SETTLE_MS);
        let fourth = bus.dhcp(4).unwrap();
        assert!(!leased[1..].contains(&fourth));
        // The address of the tab that left is not handed out while its lease runs.
        assert_ne!(fourth, leased[0]);
    }

    #[test]
    fn guests_keep_their_addresses_when_they_renew_with_the_new_server() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        bus.join(3);
        bus.advance(SETTLE_MS);
        let second = bus.dhcp(2).unwrap();
        let third = bus.dhcp(3).unwrap();
        bus.leave(1);
        bus.advance(TICK_MS);
        // The server's MAC and address did not change, so the renewal reaches the new server.
        assert_eq!(bus.request_again(3, third, true), Some(DHCP_ACK));
        assert_eq!(bus.request_again(2, second, false), Some(DHCP_ACK));
        assert_eq!(bus.request_again(3, second, false), Some(DHCP_NAK));
    }

    #[test]
    fn a_new_server_waits_for_its_census_before_offering() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        bus.advance(SETTLE_MS);
        bus.dhcp(2).unwrap();
        bus.join(3);
        bus.advance(SETTLE_MS);
        bus.leave(1);
        // Right after the takeover: the census is out but its answers not yet counted.
        assert_eq!(bus.dhcp(3), None);
        bus.advance(PRESENCE_INTERVAL_MS);
        assert!(bus.dhcp(3).is_some());
    }

    #[test]
    fn a_server_that_was_never_told_still_learns_a_lease_from_the_census() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        bus.advance(SETTLE_MS);
        let second = bus.dhcp(2).unwrap();
        // Tab 3 joins and takes over at once, before any routine announcement could reach it
        // (tab 2 announced its lease just now, so its next one is a second away).
        bus.join(3);
        bus.tabs.remove(&1);
        bus.serve(3);
        let now = bus.now_ms;
        assert_eq!(bus.tab(3).agent.leases.holder(second, now), None);
        for _ in 0..2 {
            bus.tick_in_place();
        }
        assert_eq!(bus.tab(3).agent.leases.holder(second, now), Some(mac_of(2)));
    }

    #[test]
    fn the_server_answers_arp_and_ping_for_its_address_after_a_takeover() {
        let mut bus = Bus::new();
        let (_, second) = two_leased_tabs(&mut bus);
        bus.join(3);
        bus.advance(SETTLE_MS);
        let third = bus.dhcp(3).unwrap();
        for _ in 0..2 {
            bus.take_guest(3);
            bus.guest_sends(3, arp_request(mac_of(3), third, SERVER_IP));
            let answers: Vec<_> = bus
                .take_guest(3)
                .into_iter()
                .filter(|frame| ethertype_of(frame) == ETHERTYPE_ARP)
                .filter_map(|frame| {
                    let arp = parse_arp(&frame[packet::ETH_HEADER_LEN..])?;
                    (arp.op == ARP_REPLY).then_some(arp.sender_mac)
                })
                .collect();
            assert_eq!(answers, [SERVER_MAC], "exactly one answer, the server's");
            bus.guest_sends(3, ping(mac_of(3), SERVER_MAC, third, SERVER_IP));
            let replies = bus.take_guest(3);
            assert!(replies.iter().any(|frame| is_echo_reply_to(frame, third)));
            bus.leave(1);
            bus.advance(PRESENCE_INTERVAL_MS);
        }
        let _ = second;
    }

    fn is_echo_reply_to(frame: &[u8], address: Ipv4) -> bool {
        let Some(ethernet) = parse_ethernet(frame) else {
            return false;
        };
        let Some(ip) = parse_ipv4(ethernet.payload) else {
            return false;
        };
        ip.dst == address && ip.protocol == IP_PROTOCOL_ICMP && is_icmp_echo_reply(ip.payload)
    }
}

mod switching {
    use super::*;

    #[test]
    fn unicast_reaches_only_the_destinations_tab_once_learned() {
        let mut bus = Bus::new();
        let (first, second) = two_leased_tabs(&mut bus);
        bus.join(3);
        bus.advance(SETTLE_MS);
        bus.clear_received();
        bus.guest_sends(1, chat(mac_of(1), mac_of(2), first, second, b"hello"));
        assert_eq!(bus.tab(2).received.len(), 1);
        assert!(bus.tab(3).received.is_empty());
        assert_eq!(bus.take_guest(2).len(), 1);
        bus.guest_sends(2, chat(mac_of(2), mac_of(1), second, first, b"hi back"));
        assert_eq!(bus.tab(1).received.len(), 1);
        assert!(bus.tab(3).received.is_empty());
    }

    #[test]
    fn broadcast_reaches_every_other_guest_and_not_the_sender() {
        let mut bus = Bus::new();
        for tab in 1..=3 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        bus.clear_received();
        bus.guest_sends(2, arp_request(mac_of(2), host(40), host(41)));
        assert_eq!(bus.take_guest(1).len(), 1);
        assert_eq!(bus.take_guest(3).len(), 1);
        assert!(bus.take_guest(2).is_empty());
    }

    #[test]
    fn a_broadcast_echoed_back_to_its_sender_is_dropped() {
        let mut bus = Bus::new();
        bus.join(1);
        let now = bus.now_ms;
        let frame = arp_request(mac_of(1), host(40), host(41));
        bus.tab(1).agent.from_segment(1, &frame, now);
        assert!(bus.take_guest(1).is_empty());
    }

    #[test]
    fn a_destination_never_seen_is_flooded() {
        let mut bus = Bus::new();
        for tab in 1..=3 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        bus.clear_received();
        let stranger = [0x02, 0x77, 0, 0, 0, 1];
        bus.guest_sends(1, chat(mac_of(1), stranger, host(40), host(41), b"anyone?"));
        assert_eq!(bus.tab(2).received.len(), 1);
        assert_eq!(bus.tab(3).received.len(), 1);
    }

    #[test]
    fn presence_frames_never_reach_a_guest() {
        let mut bus = Bus::new();
        for tab in 1..=3 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        for tab in 1..=3 {
            assert!(
                bus.take_guest(tab)
                    .iter()
                    .all(|frame| ethertype_of(frame) != ETHERTYPE_PRESENCE)
            );
            assert!(
                bus.tab(tab)
                    .received
                    .iter()
                    .any(|frame| ethertype_of(frame) == ETHERTYPE_PRESENCE)
            );
        }
    }

    #[test]
    fn only_the_serving_tab_answers_arp_for_the_server() {
        let mut bus = Bus::new();
        for tab in 1..=3 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        bus.clear_received();
        bus.guest_sends(3, arp_request(mac_of(3), host(40), SERVER_IP));
        let replies: Vec<Vec<u8>> = bus
            .take_guest(3)
            .into_iter()
            .filter(|frame| ethertype_of(frame) == ETHERTYPE_ARP)
            .collect();
        assert_eq!(replies.len(), 1);
        // The answer came from tab 1, the server, and went to tab 3 alone.
        assert!(
            bus.tab(2)
                .received
                .iter()
                .all(|frame| parse_ethernet(frame).unwrap().src != SERVER_MAC)
        );
    }

    #[test]
    fn frames_for_the_server_go_to_the_server_destination_and_never_to_a_guest() {
        let mut bus = Bus::new();
        let (first, second) = two_leased_tabs(&mut bus);
        bus.clear_received();
        bus.guest_sends(2, ping(mac_of(2), SERVER_MAC, second, SERVER_IP));
        assert_eq!(bus.tab(1).received.len(), 1);
        assert!(bus.take_guest(1).is_empty());
        let _ = first;
    }

    #[test]
    fn the_stations_of_a_tab_that_went_quiet_are_flooded_to_again() {
        let mut bus = Bus::new();
        let (first, second) = two_leased_tabs(&mut bus);
        let now = bus.now_ms;
        assert_eq!(
            bus.tab(1).agent.route(&mac_of(2), now),
            SegmentDestination::Tab(2)
        );
        // Tab 2 freezes: its agent stops talking but the tab does not close.
        let frozen = bus.tabs.remove(&2).unwrap();
        bus.advance(PEER_TIMEOUT_MS + PRESENCE_INTERVAL_MS);
        let now = bus.now_ms;
        assert_eq!(
            bus.tab(1).agent.route(&mac_of(2), now),
            SegmentDestination::Everyone
        );
        assert_eq!(bus.tab(1).agent.status().peers, 0);
        bus.tabs.insert(2, frozen);
        let _ = (first, second);
    }
}

mod peers {
    use super::*;

    #[test]
    fn every_tab_counts_the_others() {
        let mut bus = Bus::new();
        for tab in 1..=4 {
            bus.join(tab);
        }
        bus.advance(SETTLE_MS);
        for tab in 1..=4 {
            assert_eq!(bus.tab(tab).agent.status().peers, 3);
        }
        bus.leave(4);
        bus.advance(PEER_TIMEOUT_MS + PRESENCE_INTERVAL_MS);
        assert_eq!(bus.tab(1).agent.status().peers, 2);
    }

    #[test]
    fn the_status_says_which_tab_serves() {
        let mut bus = Bus::new();
        bus.join(1);
        bus.join(2);
        assert!(bus.tab(1).agent.status().serving);
        assert!(!bus.tab(2).agent.status().serving);
    }
}

mod robustness {
    use super::*;

    #[test]
    fn a_nic_address_must_be_locally_administered_unicast_and_not_the_servers() {
        assert!(SegmentAgent::new([0x52, 0x54, 0, 0, 0, 1], 1, 0).is_ok());
        assert!(SegmentAgent::new([0x00, 0x16, 0x3e, 0, 0, 1], 1, 0).is_err());
        assert!(SegmentAgent::new([0x03, 0, 0, 0, 0, 1], 1, 0).is_err());
        assert!(SegmentAgent::new(SERVER_MAC, 1, 0).is_err());
    }

    #[test]
    fn truncated_and_corrupted_frames_are_ignored() {
        let mut bus = Bus::new();
        let (first, second) = two_leased_tabs(&mut bus);
        let samples = [
            client_message(DHCP_DISCOVER, mac_of(2), 1, UNSPECIFIED_IP, None, None),
            client_message(DHCP_REQUEST, mac_of(2), 1, second, None, None),
            arp_request(mac_of(2), second, SERVER_IP),
            ping(mac_of(2), SERVER_MAC, second, SERVER_IP),
            chat(mac_of(2), mac_of(1), second, first, b"x"),
        ];
        let now = bus.now_ms;
        let mut presence = None;
        bus.tab(2).agent.tick(now + PRESENCE_INTERVAL_MS);
        for (_, frame) in bus.tab(2).agent.take_for_segment() {
            presence = Some(frame);
        }
        for frame in samples.iter().chain(presence.iter()) {
            for len in 0..frame.len() {
                let agent = &mut bus.tab(1).agent;
                agent.from_guest(&frame[..len], now);
                agent.from_segment(2, &frame[..len], now);
                let mut corrupted = frame.clone();
                corrupted[len] ^= 0xff;
                agent.from_guest(&corrupted, now);
                agent.from_segment(2, &corrupted, now);
            }
        }
    }
}
