use super::packet::*;
use crate::gateway::StreamKind;

#[test]
fn connect_has_the_spec_layout() {
    let bytes = encode(&WispPacket::Connect {
        stream: 0x0403_0201,
        kind: StreamKind::Tcp,
        port: 443,
        host: "1.2.3.4".into(),
    });
    assert_eq!(
        bytes,
        [
            &[0x01, 0x01, 0x02, 0x03, 0x04, 0x01, 0xbb, 0x01][..],
            b"1.2.3.4"
        ]
        .concat()
    );
}

#[test]
fn every_packet_round_trips() {
    let packets = [
        WispPacket::Connect {
            stream: 1,
            kind: StreamKind::Tcp,
            port: 80,
            host: "example.com".into(),
        },
        WispPacket::Connect {
            stream: 2,
            kind: StreamKind::Udp,
            port: 53,
            host: "1.1.1.1".into(),
        },
        WispPacket::Connect {
            stream: 3,
            kind: StreamKind::Tcp,
            port: 0,
            host: String::new(),
        },
        WispPacket::Data {
            stream: 4,
            payload: Vec::new(),
        },
        WispPacket::Data {
            stream: u32::MAX,
            payload: (0..16384).map(|index| index as u8).collect(),
        },
        WispPacket::Continue {
            stream: 0,
            remaining: 0x0102_0304,
        },
        WispPacket::Close {
            stream: 6,
            reason: CLOSE_VOLUNTARY,
        },
    ];
    for packet in packets {
        assert_eq!(decode(&encode(&packet)), Some(packet));
    }
}

#[test]
fn close_and_continue_have_fixed_payload_sizes() {
    let close = encode(&WispPacket::Close {
        stream: 1,
        reason: CLOSE_REFUSED,
    });
    assert_eq!(close, [0x04, 1, 0, 0, 0, 0x44]);
    let cont = encode(&WispPacket::Continue {
        stream: 1,
        remaining: 5,
    });
    assert_eq!(cont, [0x03, 1, 0, 0, 0, 5, 0, 0, 0]);
}

#[test]
fn unknown_info_and_truncated_packets_decode_to_none() {
    assert_eq!(decode(&[0x05, 0, 0, 0, 0, 2, 0]), None);
    assert_eq!(decode(&[0x09, 0, 0, 0, 0]), None);
    assert_eq!(decode(&[0x03, 0, 0, 0, 0, 1]), None);
    assert_eq!(decode(&[]), None);
}

#[test]
fn malformed_connects_and_trailing_bytes() {
    // Hostname not UTF-8.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x01, 80, 0, 0xff, 0xfe]), None);
    // Unknown stream type.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x07, 80, 0, b'a']), None);
    // Missing port.
    assert_eq!(decode(&[0x01, 1, 0, 0, 0, 0x01, 80]), None);
    // Trailing bytes after a fixed payload are ignored.
    assert_eq!(
        decode(&[0x04, 1, 0, 0, 0, 0x02, 9, 9]),
        Some(WispPacket::Close {
            stream: 1,
            reason: CLOSE_VOLUNTARY
        })
    );
    assert_eq!(
        decode(&[0x03, 1, 0, 0, 0, 7, 0, 0, 0, 9]),
        Some(WispPacket::Continue {
            stream: 1,
            remaining: 7
        })
    );
}

#[test]
fn decoding_never_panics() {
    let mut state = 0x5eed_0007_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    let packets = [
        WispPacket::Connect {
            stream: 9,
            kind: StreamKind::Udp,
            port: 5353,
            host: "host.example".into(),
        },
        WispPacket::Data {
            stream: 9,
            payload: vec![1, 2, 3],
        },
        WispPacket::Continue {
            stream: 9,
            remaining: 100,
        },
        WispPacket::Close {
            stream: 9,
            reason: CLOSE_BLOCKED,
        },
    ];
    for packet in &packets {
        let bytes = encode(packet);
        for end in 0..=bytes.len() {
            inputs.push(bytes[..end].to_vec());
        }
    }
    for len in 0..64 {
        inputs.push((0..len).map(|_| next() as u8).collect());
    }
    for input in inputs {
        let _ = decode(&input);
    }
}
