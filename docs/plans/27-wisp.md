# Internet for web guests through Wisp

**Status:** implemented and checked under Node and in a browser against `serve.py`'s own Wisp endpoint. Not yet checked against a third-party Wisp server ("Results", "Open").

## Why

A guest on the web page cannot reach anything off the tabs' segment (`13-web.md`): 10.0.2.2 answers ARP, ping and DHCP but routes nothing, and DHCP gives no gateway or DNS. The concrete thing wanted is `apk update` and `apk add` in the Alpine demo; anything else a guest does over TCP or UDP comes with it.

A browser cannot open sockets, and `fetch()` needs CORS, which none of Alpine's mirrors send (checked 2026-11-05: `dl-cdn.alpinelinux.org`, kernel.org, leaseweb, dotsrc, tuna, fcix, uk.alpinelinux.org, wearetriple). A WebSocket is outside CORS and works on the cross-origin-isolated page, so the way out is a relay at the other end of one. v86 has three such models (`src/browser/` in copy/v86): a raw Ethernet relay (websockproxy: one message per frame, the relay runs DHCP and NAT on a TAP device), Wisp, and HTTP-only `fetch()`. Wisp is the one taken here: the relay is an unprivileged stream proxy with several implementations, and the browser side is a terminating TCP/UDP gateway, which the CLI already has (`crates/uemulator-cli/src/usernet.rs`).

## Goals

- A web guest with the Network switch on and a Wisp URL set reaches any TCP host and any UDP host (DNS included) through the Wisp server, with nothing configured in the guest beyond DHCP.
- One gateway implementation: the CLI's user-mode network and the web page run the same TCP/UDP/DHCP/ARP code from the library, each with its own stream backend.
- `apk update && apk add <something>` works in the Alpine demo under Node and in the browser against a local Wisp server.

## Non-goals

- The raw Ethernet relay (websockproxy) and the `fetch()` model. Either can come later; the relay is a small addition on the segment's transport.
- Wisp v2 (extensions, password authentication). A v1 client works with v2 servers.
- Inbound connections (port forwarding) on the web page.
- IPv6, TCP window scaling, anything usernet does not already do.
- A public Wisp server operated by this project.

## Anti-goals

- No regression in the CLI's network: the shared gateway had to stay within noise of usernet's bulk TCP (`hack/perf/guest` `ubench net`). It did: 903 to 945 MB/s down and 1012 to 997 MB/s up across the move (`23-guest-io-perf.md` round 3).
- No change to how guests on the segment reach each other, or to the DHCP handover between tabs.
- Never drop or reorder Wisp stream data anywhere between the WebSocket and the gateway. Ethernet frames may be dropped (the guest's TCP recovers); a stream's bytes may not (there is no retransmission on the Wisp side).

## Architecture

### The gateway (library, `crates/uemulator/src/gateway/`)

usernet's protocol engine moves into the library as `Gateway`, `no_std` + `alloc`, exported from `lib.rs` like `SegmentAgent`. It does no I/O and reads no clock: every call that needs time takes the host's monotonic milliseconds, and the initial sequence number seed is a constructor argument (usernet takes it from `SystemTime`).

What moves: Ethernet/IPv4/TCP/UDP/ICMP parsing and building, checksums, IPv4 fragmentation of outbound datagrams, ARP, ICMP echo answered locally, DHCP, the TCP state machine (terminating, no retransmission, out-of-order dropped with a duplicate ACK, SYN-ACK only after the remote side connected, RST when it failed), round-robin service order and the per-poll frame budget, window advertisement and the window-update rule, large (TSO) segments when the guest takes them, UDP flow limit and idle expiry. The unit tests that test those move with them.

What stays in the driver: sockets, `poll(2)`, the connect threads, the wake pipe, `system_dns()`.

The socket side becomes streams. Sketch (names and exact shape are the builder's):

- `from_guest(frame, now_ms)`, `take_for_guest() -> frames` (each with its optional TSO segment size, as `OutFrame` today), `tick(now_ms)`.
- The gateway asks: `Open { stream, kind: Tcp | Udp, target }`, `ShutdownWrite { stream }`, `Close { stream }`. `target` is an address and port after the gateway's destination policy (below).
- The driver answers: `opened(stream)`, `failed(stream)`, `data(stream, bytes)` (for UDP one call is one datagram), `remote_closed(stream)` (EOF), `reset(stream)`.
- Guest data to the remote side is pulled: the driver asks what is waiting for a stream, writes what it can, and reports how much it took (`consumed`). The advertised TCP window is 65535 minus what is still waiting, as now, so a slow remote side slows the guest.
- Remote data to the guest is pushed into a per-stream buffer that `take_for_guest` drains up to the guest's window. The driver may ask how much room a stream has (guest window minus buffered) and read no more than that: the CLI does, so usernet's backpressure on host sockets stays as it is. A Wisp driver cannot hold data back (below) and pushes whatever arrives.

Destination policy, a constructor argument: which destinations the gateway terminates and what each becomes.
- CLI: everything in the subnet except the guest is proxy-ARPed as now; 10.0.2.2 becomes the host's 127.0.0.1, 10.0.2.3:53 the system resolver, the rest goes out unchanged.
- Web: only off-subnet destinations and 10.0.2.2 port 53 (TCP and UDP). 10.0.2.2:53 becomes the resolver (1.1.1.1:53 unless the page sets another). The segment's other addresses are other guests and never reach the gateway. The gateway does no ARP and no DHCP on the web: the serving tab's `SegmentAgent` answers both for 10.0.2.2.

### The CLI

`usernet.rs` keeps `spawn` and its loops and drives `Gateway` with `TcpStream`/`UdpSocket` streams. Its loopback tests stay where they are and now test the driver and the gateway together. `ubench net` before and after, same host, `ab.py` medians, goes in `23-guest-io-perf.md`.

### The web page

Placement: each tab runs its own gateway in the emulator's worker, beside its `SegmentAgent` in `NetLink::pump`. A guest frame to `SERVER_MAC` whose IPv4 destination the policy terminates goes to the gateway; every other frame goes to the agent as now. A tab closing takes down only its own guest's connections, and nothing about streams crosses tabs.

DHCP: the agent's DHCP server always gives router 10.0.2.2 and DNS server 10.0.2.2 (`segment/dhcp.rs`), whether or not any tab has a Wisp URL, so a guest's configuration never depends on which tab serves. A tab without a Wisp URL runs the gateway with a backend that fails every open: TCP gets a RST, UDP an ICMP port unreachable, so a guest fails at once instead of timing out.

The WebSocket lives where an event loop runs: on the page (`main.js`), or under Node in the parent of the machine's worker thread (`hack/wasm/segment.mjs`/`segment_machine.mjs` pattern). The emulator's worker cannot receive messages while `uw_run` runs, so Wisp traffic moves the way segment frames do:
- Worker to page: a new host import (say `host_wisp_send(ptr, len)`) posts encoded Wisp packets; the page sends each as one WebSocket binary message.
- Page to worker: a second byte ring in the shared control block (`protocol.js`), written by the page, drained by the worker in the poll hook, which wakes the worker's idle wait like the frame ring. Unlike the frame ring it must never drop: when it is full the page queues in JavaScript and writes the queue out as the worker drains.
- The Wisp client (packet codec, stream ids, CONTINUE credits, the connection's state) is Rust in `uemulator-web` (or the library, if the builder finds it reusable), not JavaScript, so it is unit-tested with the gateway. The page side only opens, reconnects and carries bytes.

The Wisp client (v1, `MercuryWorkshop/wisp-protocol` `protocol.md`):
- Waits for the server's CONTINUE on stream 0 before any CONNECT; its value is every new stream's initial credit. Packet type 5 (v2's INFO) is ignored, which makes a v2 server treat this as a v1 client.
- Stream ids are allocated by the client, never 0, never reused while live.
- CONNECT with the destination address as a dotted-decimal string (the guest resolved names already).
- TCP DATA is sent only with credit; credit counts packets, not bytes, so packets are large (a cap such as 16 KiB is the builder's call, bounded by what common servers accept). UDP DATA has no credit and is one datagram per packet.
- A stream counts as opened as soon as its CONNECT is sent: Wisp v1 has no "connected" packet (a server reports only failure, with CLOSE, and the client may send data at once), so the Wisp driver calls `opened` right away and the guest gets its SYN-ACK without waiting, as v86 does (`wisp_network.js` accepts the guest's connection when it sends CONNECT). Waiting for the server's first packet would deadlock every protocol where the client speaks first; waiting out a delay would add it to every connection. The cost: a refused or unreachable destination reaches the guest as a reset of an established connection, not a refused connect. The gateway keeps the `opened`/`failed` pair because the CLI has a real connect result.
- CLOSE from the server: 0x02 is the remote side's end of data, so `remote_closed` (FIN to the guest once the data before it is delivered); on a TCP stream every other reason is `reset` (RST). On a UDP stream any CLOSE other than 0x02 counts as a failed open (`gateway.failed`), which is what lets the DNS fallback below work on servers that refuse UDP. The last non-voluntary reason is kept for the pill's tooltip.
- UDP DATA from the server longer than 65 507 bytes (the most an IPv4 datagram carries) is dropped.
- Wisp v1 has no half-close. A guest FIN is held: the stream stays open and the gateway sends its FIN onwards only as the end of data it already passed; the stream ends with the server's CLOSE or when both sides are done (guest FIN acknowledged and the remote side closed), at which point the client sends CLOSE 0x02. A guest RST sends CLOSE 0x02 at once.
- If the server refuses a UDP stream to the resolver (some servers turn UDP off), DNS falls back to TCP to the same resolver, the guest's UDP query and answer carried with RFC 7766's two-byte length prefix, so `apk` still resolves. The resolver's UDP flow keeps up to 64 unanswered queries to replay over TCP. A query over TCP expires after 10 s; at most 64 are pending and 256 KiB queued (`gateway/dns_tcp.rs`).
- The WebSocket closing fails or resets every stream; the page reconnects with backoff, 1 s doubling to 30 s, reset when a Wisp handshake completes (the server's CONTINUE on stream 0), not when the WebSocket opens, so a server that accepts and then drops does not reset it. The client gives a handshake 10 s; when the worker reports that timeout the page closes the socket and reconnects. New streams wait for the new connection's CONTINUE.

Wisp v1 has no way to slow the server, so data for a guest that reads slower than the network arrives faster than the gateway can pass it on. It buffers in the gateway's per-stream buffers (wasm memory) and in the page's queue. Accepted for now; a guest downloading over a fast relay is the normal case and reads faster than the relay sends.

Setting the URL: a "Wisp server" text field in the sidebar under the Network switch, `?wisp=wss://host/path/` (the trailing slash is Wisp's convention), and `--wisp URL` on `hack/wasm/segment.mjs` and `hack/wasm/wisp.mjs` (`hack/wasm/boot.mjs` has no network, so no `--wisp`). Empty means no uplink. The bar's network pill says whether the uplink is connected (its tooltip gives the URL and the last close reason).

Published demo: no default URL; the field starts empty on GitHub Pages and anyone can enter a server. `hack/web.sh serve` prefills it with its own local endpoint.

### Local Wisp server

`hack/web/serve.py` grows a Wisp v1 endpoint at `/wisp/` on its own port and address, stdlib only like the rest of the script: the WebSocket handshake and framing, a thread per connection reading packets, and per stream a thread that connects and then writes (so one slow destination does not stall the others) and a thread reading its socket, TCP and UDP, CONTINUE every half of the buffer. A client CLOSE delivers the DATA queued before it, then closes. It is a development tool; the page works with any Wisp server.

It is a proxy, so it guards itself twice. Binding 127.0.0.1 keeps other machines out. That does not keep out other web pages in the developer's browser, since a WebSocket has no CORS: any site could connect to it and reach the developer's localhost and LAN. So the endpoint refuses a handshake whose `Origin` is not serve.py's own (`http://127.0.0.1:PORT` or `http://localhost:PORT`); a handshake with no `Origin` (Node's scripts) is accepted. Past that check every destination is allowed, loopback and private addresses included, so a guest can reach a mirror or a file served on the developer's machine, as the CLI's guests reach the host through 10.0.2.2.

The client must not be written against this endpoint alone, which shares its author's reading of the spec: it is to be checked against two third-party servers (wisp-js's server via `npx`, and wisp-server-python), by hand, with the end-to-end script's `--wisp URL`. That has not been done ("Open").

## Tests

- Library: the gateway's protocol tests move from `usernet/tests.rs` (checksums, ARP, ICMP, DHCP, TCP state, windows, TSO segments, fragmentation, UDP flows) and run against a fake stream backend instead of loopback sockets. New ones: the destination policy, the web policy's DHCP-less and ARP-less mode, guest FIN held under a backend without half-close, data pushed faster than the guest's window, open failing.
- Wisp client: packet encode/decode, the initial CONTINUE gate, credit accounting, CLOSE reason mapping, UDP-refused DNS fallback, reconnect, against a scripted fake server.
- CLI: `usernet/tests.rs`'s loopback tests keep passing unchanged except for what moved.
- `segment/tests.rs`: DHCP now offers router and DNS; nothing else in the segment's behaviour changes.
- End to end (a hack script, not a unit test: it needs the network): `node hack/wasm/wisp.mjs [--wisp URL]` (after `WEB_DEMO=1 hack/web.sh`; Node 24.2 or later, as `segment.mjs`, which it shares code with) starts `serve.py`'s Wisp endpoint unless given another server, boots the Alpine demo under Node, runs `apk update` and `apk add jq`, checks a `wget` of the demo kernel over http matches its SHA-256, then drops the WebSocket in the middle of a second download (it restarts `serve.py`; skipped with `--wisp`) and checks the guest's wget fails, the client is ready again and `apk update` works. Also checked by hand in the browser.
- Performance: `ubench net` before and after the CLI move (anti-goals); the web path's throughput through the local server is measured and recorded, not gated.
- `hack/web/test_serve.py` (`python3 -m unittest hack/web/test_serve.py`): the Wisp endpoint's framing, CLOSE handling, stream ordering and Origin check.

## Results

- CLI throughput across the gateway move: `23-guest-io-perf.md` round 3 (rx 903 to 945 MB/s, tx 1012 to 997 MB/s, Alpine netboot live).
- `node hack/wasm/wisp.mjs` under Node against `serve.py`: the whole run takes about 20 s; `apk update` about 5 s; `apk add jq` about 4 s; the 12.1 MiB kernel over http downloads at 17.6 to 20 MiB/s with the SHA-256 matching. A WebSocket drop during a download fails the guest's wget, the client is ready again about 1.1 s later and `apk update` works.
- In a browser (the desktop app's Chrome-based pane) against `serve.py`: `apk update` over https in 5.1 s.

## Open

- Third-party Wisp servers (wisp-js, wisp-server-python) were not checked: installing them was not approved. Only `serve.py`'s endpoint has run against the client, so a reading of the spec that both share is untested.

## Left to the builder

- Exact API names and shapes of `Gateway` and the stream calls; whether the Wisp codec sits in the library or in `uemulator-web`.
- Wisp DATA packet size, reconnect backoff, the page's queue limits beyond "never drop".
- Order of work. A sensible one: move the gateway into the library with the CLI driving it (tests and `ubench net` green), then the Wisp client with its fake server, then the web wiring and `serve.py`, then the end-to-end script and the docs (`13-web.md`, `CLAUDE.md`'s Web line, `10-milestones.md`).

## Calls made in the design, not asked

- Each tab runs its own gateway rather than the serving tab routing for everyone.
- DHCP always gives router and DNS 10.0.2.2 on the segment, uplink or not; no uplink means immediate refusal.
- DNS is UDP over Wisp to 1.1.1.1 by default, with a TCP fallback.
- The Pages demo has no default Wisp server.
- The local Wisp server lives in `serve.py`, stdlib only.
- The CLI moves onto the shared gateway in the same piece of work.
- A Wisp stream counts as open when its CONNECT is sent (as in v86), so a refused destination looks to the guest like a reset, not a refused connect.
- The local endpoint checks `Origin` and otherwise allows every destination, the developer's loopback and LAN included.
- The client is to be checked against wisp-js's server and wisp-server-python, not only against `serve.py` (not done yet).
