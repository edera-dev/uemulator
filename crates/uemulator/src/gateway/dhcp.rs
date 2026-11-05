//! DHCP server for the user-mode NAT: one address, 10.0.2.15, and the options a guest needs to
//! use the gateway and DNS addresses.

#[cfg(freestanding)]
use crate::prelude::*;
use core::ops::Range;

use super::{DNS_IP, GATEWAY_IP, GUEST_IP, SUBNET_BROADCAST, SUBNET_MASK};

pub(super) const DHCP_SERVER_PORT: u16 = 67;
pub(super) const DHCP_CLIENT_PORT: u16 = 68;
/// Offsets inside the BOOTP/DHCP message (RFC 2131 section 2).
pub(super) const BOOTP_OP: usize = 0;
pub(super) const BOOTP_HTYPE: usize = 1;
pub(super) const BOOTP_HLEN: usize = 2;
pub(super) const BOOTP_XID: Range<usize> = 4..8;
pub(super) const BOOTP_CIADDR: Range<usize> = 12..16;
pub(super) const BOOTP_FLAGS: Range<usize> = 10..12;
pub(super) const BOOTP_YIADDR: Range<usize> = 16..20;
pub(super) const BOOTP_SIADDR: Range<usize> = 20..24;
pub(super) const BOOTP_CHADDR: Range<usize> = 28..44;
pub(super) const BOOTP_MAGIC_COOKIE: Range<usize> = 236..240;
pub(super) const DHCP_OPTIONS_OFFSET: usize = 240;
pub(super) const DHCP_MAGIC_COOKIE: [u8; 4] = [0x63, 0x82, 0x53, 0x63];
pub(super) const BOOTP_REQUEST: u8 = 1;
pub(super) const BOOTP_REPLY: u8 = 2;
pub(super) const HTYPE_ETHERNET: u8 = 1;
pub(super) const HLEN_ETHERNET: u8 = 6;
/// BOOTP replies are padded to the 300 bytes of the original fixed-format packet (RFC 951) for
/// the benefit of old clients.
pub(super) const BOOTP_MIN_LEN: usize = 300;
pub(super) const DHCP_OPT_PAD: u8 = 0;
pub(super) const DHCP_OPT_SUBNET_MASK: u8 = 1;
pub(super) const DHCP_OPT_ROUTER: u8 = 3;
pub(super) const DHCP_OPT_DNS_SERVER: u8 = 6;
pub(super) const DHCP_OPT_BROADCAST_ADDRESS: u8 = 28;
pub(super) const DHCP_OPT_REQUESTED_IP: u8 = 50;
pub(super) const DHCP_OPT_LEASE_TIME: u8 = 51;
pub(super) const DHCP_OPT_MESSAGE_TYPE: u8 = 53;
pub(super) const DHCP_OPT_SERVER_ID: u8 = 54;
pub(super) const DHCP_OPT_END: u8 = 255;
pub(super) const DHCP_DISCOVER: u8 = 1;
pub(super) const DHCP_OFFER: u8 = 2;
pub(super) const DHCP_REQUEST: u8 = 3;
pub(super) const DHCP_ACK: u8 = 5;
pub(super) const DHCP_NAK: u8 = 6;
pub(super) const DHCP_LEASE_SECS: u32 = 24 * 60 * 60;

/// The fields of a client's BOOTP/DHCP message that decide our answer.
pub(super) struct DhcpRequest {
    pub(super) message_type: u8,
    /// Option 50.
    pub(super) requested_ip: Option<[u8; 4]>,
    /// Option 54: which server the client picked.
    pub(super) server_id: Option<[u8; 4]>,
    /// Non-zero only for a client renewing an address it already holds.
    pub(super) client_ip: [u8; 4],
}

/// `None` unless this is a well-formed BOOTP request carrying the DHCP magic cookie. A truncated
/// option ends parsing; whatever was read before it still counts.
pub(super) fn parse_dhcp_request(request: &[u8]) -> Option<DhcpRequest> {
    if request.len() <= DHCP_OPTIONS_OFFSET
        || request[BOOTP_OP] != BOOTP_REQUEST
        || request[BOOTP_MAGIC_COOKIE] != DHCP_MAGIC_COOKIE
    {
        return None;
    }
    let mut parsed = DhcpRequest {
        message_type: 0,
        requested_ip: None,
        server_id: None,
        client_ip: request[BOOTP_CIADDR].try_into().ok()?,
    };
    let mut pos = DHCP_OPTIONS_OFFSET;
    while pos < request.len() && request[pos] != DHCP_OPT_END {
        if request[pos] == DHCP_OPT_PAD {
            pos += 1;
            continue;
        }
        let Some(&option_len) = request.get(pos + 1) else {
            break;
        };
        let Some(value) = request.get(pos + 2..pos + 2 + option_len as usize) else {
            break;
        };
        match (request[pos], value) {
            (DHCP_OPT_MESSAGE_TYPE, [message_type, ..]) => parsed.message_type = *message_type,
            (DHCP_OPT_REQUESTED_IP, &[a, b, c, d]) => parsed.requested_ip = Some([a, b, c, d]),
            (DHCP_OPT_SERVER_ID, &[a, b, c, d]) => parsed.server_id = Some([a, b, c, d]),
            _ => {}
        }
        pos += 2 + option_len as usize;
    }
    Some(parsed)
}

/// RFC 2131 section 4.3: DISCOVER gets an OFFER. REQUEST gets an ACK unless it asks for an
/// address other than `GUEST_IP` (NAK), or names a different server, in which case the client has
/// taken another server's offer and we stay silent. RELEASE and DECLINE are ignored (there is no lease
/// table to update). INFORM is not implemented either, although RFC 2131 section 4.3.5 has the
/// server answer it with an ACK carrying configuration.
pub(super) fn dhcp_reply_type(request: &DhcpRequest) -> Option<u8> {
    match request.message_type {
        DHCP_DISCOVER => Some(DHCP_OFFER),
        DHCP_REQUEST => {
            if request.server_id.is_some_and(|id| id != GATEWAY_IP) {
                return None;
            }
            // Selecting/init-reboot carry option 50; renewing and rebinding carry ciaddr.
            let wanted = request
                .requested_ip
                .or((request.client_ip != [0; 4]).then_some(request.client_ip));
            match wanted {
                Some(ip) if ip != GUEST_IP => Some(DHCP_NAK),
                _ => Some(DHCP_ACK),
            }
        }
        _ => None,
    }
}

/// The BOOTP/DHCP reply to `request` (which must have parsed), padded to the classic minimum size.
/// A NAK carries only the message type and server identifier, with no address (RFC 2131 table 3).
pub(super) fn dhcp_reply(request: &[u8], reply_type: u8) -> Vec<u8> {
    let mut reply = vec![0u8; DHCP_OPTIONS_OFFSET];
    reply[BOOTP_OP] = BOOTP_REPLY;
    reply[BOOTP_HTYPE] = HTYPE_ETHERNET;
    reply[BOOTP_HLEN] = HLEN_ETHERNET;
    reply[BOOTP_XID].copy_from_slice(&request[BOOTP_XID]);
    reply[BOOTP_FLAGS].copy_from_slice(&request[BOOTP_FLAGS]);
    reply[BOOTP_CHADDR].copy_from_slice(&request[BOOTP_CHADDR]);
    reply[BOOTP_MAGIC_COOKIE].copy_from_slice(&DHCP_MAGIC_COOKIE);
    let lease_secs = DHCP_LEASE_SECS.to_be_bytes();
    let message_type = [reply_type];
    let options: &[(u8, &[u8])] = if reply_type == DHCP_NAK {
        &[
            (DHCP_OPT_MESSAGE_TYPE, &message_type),
            (DHCP_OPT_SERVER_ID, &GATEWAY_IP),
        ]
    } else {
        reply[BOOTP_YIADDR].copy_from_slice(&GUEST_IP);
        reply[BOOTP_SIADDR].copy_from_slice(&GATEWAY_IP);
        &[
            (DHCP_OPT_MESSAGE_TYPE, &message_type),
            (DHCP_OPT_SERVER_ID, &GATEWAY_IP),
            (DHCP_OPT_LEASE_TIME, &lease_secs),
            (DHCP_OPT_SUBNET_MASK, &SUBNET_MASK),
            (DHCP_OPT_ROUTER, &GATEWAY_IP),
            (DHCP_OPT_DNS_SERVER, &DNS_IP),
            (DHCP_OPT_BROADCAST_ADDRESS, &SUBNET_BROADCAST),
        ]
    };
    for (code, value) in options {
        reply.push(*code);
        reply.push(value.len() as u8);
        reply.extend_from_slice(value);
    }
    reply.push(DHCP_OPT_END);
    reply.resize(BOOTP_MIN_LEN, 0);
    reply
}
