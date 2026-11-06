//! Wisp v1 packets: type u8, stream id u32, then a payload by type; integers little-endian.

#[cfg(freestanding)]
use crate::prelude::*;

use crate::gateway::StreamKind;

pub(crate) const PACKET_CONNECT: u8 = 0x01;
pub(crate) const PACKET_DATA: u8 = 0x02;
pub(crate) const PACKET_CONTINUE: u8 = 0x03;
pub(crate) const PACKET_CLOSE: u8 = 0x04;

pub(crate) const STREAM_TCP: u8 = 0x01;
pub(crate) const STREAM_UDP: u8 = 0x02;

pub(crate) const CLOSE_UNSPECIFIED: u8 = 0x01;
pub(crate) const CLOSE_VOLUNTARY: u8 = 0x02;
pub(crate) const CLOSE_NETWORK_ERROR: u8 = 0x03;
pub(crate) const CLOSE_INVALID: u8 = 0x41;
pub(crate) const CLOSE_UNREACHABLE: u8 = 0x42;
pub(crate) const CLOSE_TIMEOUT: u8 = 0x43;
pub(crate) const CLOSE_REFUSED: u8 = 0x44;
pub(crate) const CLOSE_TRANSFER_TIMEOUT: u8 = 0x47;
pub(crate) const CLOSE_BLOCKED: u8 = 0x48;
pub(crate) const CLOSE_THROTTLED: u8 = 0x49;
pub(crate) const CLOSE_CLIENT_ERROR: u8 = 0x81;

const TYPE_LEN: usize = 1;
const STREAM_ID_LEN: usize = 4;
const HEADER_LEN: usize = TYPE_LEN + STREAM_ID_LEN;
const PORT_LEN: usize = 2;
const CONNECT_FIXED_LEN: usize = 1 + PORT_LEN;
const CONTINUE_PAYLOAD_LEN: usize = 4;
const CLOSE_PAYLOAD_LEN: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WispPacket {
    Connect {
        stream: u32,
        kind: StreamKind,
        port: u16,
        host: String,
    },
    Data {
        stream: u32,
        payload: Vec<u8>,
    },
    Continue {
        stream: u32,
        remaining: u32,
    },
    Close {
        stream: u32,
        reason: u8,
    },
}

pub(crate) fn encode(packet: &WispPacket) -> Vec<u8> {
    let mut out = Vec::new();
    match packet {
        WispPacket::Connect {
            stream,
            kind,
            port,
            host,
        } => {
            start(&mut out, PACKET_CONNECT, *stream);
            out.push(match kind {
                StreamKind::Tcp => STREAM_TCP,
                StreamKind::Udp => STREAM_UDP,
            });
            out.extend_from_slice(&port.to_le_bytes());
            out.extend_from_slice(host.as_bytes());
        }
        WispPacket::Data { stream, payload } => {
            start(&mut out, PACKET_DATA, *stream);
            out.extend_from_slice(payload);
        }
        WispPacket::Continue { stream, remaining } => {
            start(&mut out, PACKET_CONTINUE, *stream);
            out.extend_from_slice(&remaining.to_le_bytes());
        }
        WispPacket::Close { stream, reason } => {
            start(&mut out, PACKET_CLOSE, *stream);
            out.push(*reason);
        }
    }
    out
}

fn start(out: &mut Vec<u8>, packet_type: u8, stream: u32) {
    out.push(packet_type);
    out.extend_from_slice(&stream.to_le_bytes());
}

/// Input comes from the server: anything malformed, unknown (v2's INFO, type 5, included) or
/// truncated is `None`. Bytes after a fixed-size payload are ignored.
pub(crate) fn decode(message: &[u8]) -> Option<WispPacket> {
    let packet_type = *message.first()?;
    let stream = u32::from_le_bytes(message.get(TYPE_LEN..HEADER_LEN)?.try_into().ok()?);
    let payload = &message[HEADER_LEN..];
    match packet_type {
        PACKET_CONNECT => {
            let kind = match *payload.first()? {
                STREAM_TCP => StreamKind::Tcp,
                STREAM_UDP => StreamKind::Udp,
                _ => return None,
            };
            let port = u16::from_le_bytes(payload.get(1..CONNECT_FIXED_LEN)?.try_into().ok()?);
            let host = core::str::from_utf8(&payload[CONNECT_FIXED_LEN..]).ok()?;
            Some(WispPacket::Connect {
                stream,
                kind,
                port,
                host: host.into(),
            })
        }
        PACKET_DATA => Some(WispPacket::Data {
            stream,
            payload: payload.to_vec(),
        }),
        PACKET_CONTINUE => {
            let remaining =
                u32::from_le_bytes(payload.get(..CONTINUE_PAYLOAD_LEN)?.try_into().ok()?);
            Some(WispPacket::Continue { stream, remaining })
        }
        PACKET_CLOSE => Some(WispPacket::Close {
            stream,
            reason: *payload.get(..CLOSE_PAYLOAD_LEN)?.first()?,
        }),
        _ => None,
    }
}
