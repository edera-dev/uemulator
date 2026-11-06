// The transport of the guest's internet (docs/plans/27-wisp.md): a WebSocket to a Wisp server.
// The Wisp client itself (packets, streams, credit) is Rust in the worker; this only opens,
// reconnects and carries bytes.
//
// The emulator's worker cannot receive messages while `uw_run` runs, so, as in segment.js, the
// two halves live in different places: `connectWisp` on the page (or any thread with an event
// loop) owns the socket and writes its events and messages into the wisp ring of the shared
// control block, which the worker drains in `host_wisp_recv` (`takeWispRecord`);
// `createWispSender` in the worker posts the client's packets back for `send`. No DOM here, so
// Node runs the same code.
import { CTRL, WISP_EVENT, WISP_RECORD_MAX, WISP_STATE } from "./protocol.js";
import { ringRead, ringWrite } from "./segment.js";

// A record: the payload's length (u32 little-endian) and the event (u8), then the payload.
const RECORD_HEADER_BYTES = 5;
// How often records that did not fit are tried again while the worker drains the ring.
const RETRY_MS = 5;
const FIRST_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 30_000;
// WebSocket readyState values (the HTML standard's WebSocket interface).
const SOCKET_CONNECTING = 0;
const SOCKET_OPEN = 1;

// Appends one record and wakes the worker, or returns false when the ring has no room for it.
function putRecord(control, ring, event, payload) {
  const head = Atomics.load(control, CTRL.WISP_HEAD);
  const tail = Atomics.load(control, CTRL.WISP_TAIL);
  const used = (head - tail + ring.length) % ring.length;
  // One byte stays free, so a full ring is never mistaken for an empty one.
  if (used + RECORD_HEADER_BYTES + payload.length >= ring.length) return false;
  const header = new Uint8Array(RECORD_HEADER_BYTES);
  const view = new DataView(header.buffer);
  view.setUint32(0, payload.length, true);
  view.setUint8(4, event);
  const next = ringWrite(ring, ringWrite(ring, head, header), payload);
  Atomics.store(control, CTRL.WISP_HEAD, next);
  Atomics.notify(control, CTRL.HEAD);
  return true;
}

// The worker's side of the ring: copies the next record's payload into `out` (WISP_RECORD_MAX
// bytes) and returns { length, event }, or null when the ring is empty. The page never writes a
// longer one; if one were there anyway it is taken and reported as LOST, because skipping a
// message would corrupt a stream.
export function takeWispRecord(control, ring, out) {
  const head = Atomics.load(control, CTRL.WISP_HEAD);
  let tail = Atomics.load(control, CTRL.WISP_TAIL);
  if (head === tail) return null;
  const header = new Uint8Array(RECORD_HEADER_BYTES);
  tail = ringRead(ring, tail, header);
  const view = new DataView(header.buffer);
  const length = view.getUint32(0, true);
  const event = view.getUint8(4);
  if (length > out.length) {
    Atomics.store(control, CTRL.WISP_TAIL, (tail + length) % ring.length);
    return { length: 0, event: WISP_EVENT.LOST };
  }
  tail = ringRead(ring, tail, out.subarray(0, length));
  Atomics.store(control, CTRL.WISP_TAIL, tail);
  return { length, event };
}

export function wispRingHasRecords(control) {
  return Atomics.load(control, CTRL.WISP_HEAD) !== Atomics.load(control, CTRL.WISP_TAIL);
}

// Opens a WebSocket to the Wisp server at `url` and keeps one open until `close`: after a loss it
// waits FIRST_RECONNECT_MS, doubling up to MAX_RECONNECT_MS, back to the first once a socket
// completes the Wisp handshake. `control` and `ring` are the shared control block's Int32 words
// and wisp ring. `onStatus({ socket, retryMs })` reports the socket: "connecting", "open",
// "waiting" (to reconnect in `retryMs`) or "closed".
//
// The returned object's `send(messages)` sends the worker's packets (its `{ type: "wisp" }`
// messages) as one binary message each; packets for a socket that is not open are dropped, since
// the client in the worker gave up their streams when it heard of the loss.
//
// The page must hand every `{ type: "wisp-status" }` from the worker to `status(state)`: both the
// backoff reset and the handshake timeout below depend on it. READY resets the backoff (a socket
// that merely opens proves nothing: an endpoint that accepts the upgrade and closes, or that is
// not a Wisp server, would otherwise be retried every second forever). DISCONNECTED while this
// socket is still connecting or open means the client gave up its handshake, whose 10 s clock
// starts when the socket is created (a server that is not a Wisp server never sends the first
// CONTINUE, or the socket took that long to open), so the socket is closed and another tried
// after the usual wait; the client would ignore anything this one still brought.
export function connectWisp({ control, ring, url, onStatus = () => {} }) {
  // Records that did not fit yet, in order: nothing may overtake them.
  const pending = [];
  let retryTimer = null;
  let reconnectTimer = null;
  let reconnectMs = FIRST_RECONNECT_MS;
  let socket = null;
  // The worker has reported a state other than DISCONNECTED since this socket's CONNECTING; until
  // then a DISCONNECTED is about an earlier socket.
  let acknowledged = false;
  let closed = false;

  const flushPending = () => {
    retryTimer = null;
    while (pending.length && putRecord(control, ring, pending[0].event, pending[0].payload)) pending.shift();
    if (pending.length) retryTimer = setTimeout(flushPending, RETRY_MS);
  };
  const record = (event, payload = new Uint8Array(0)) => {
    pending.push({ event, payload });
    if (retryTimer === null) flushPending();
  };

  const connect = () => {
    reconnectTimer = null;
    if (closed) return;
    const current = new globalThis.WebSocket(url);
    current.binaryType = "arraybuffer";
    socket = current;
    acknowledged = false;
    record(WISP_EVENT.CONNECTING);
    onStatus({ socket: "connecting" });
    current.onopen = () => {
      if (socket !== current) return;
      onStatus({ socket: "open" });
    };
    current.onmessage = ({ data }) => {
      if (socket !== current) return;
      if (typeof data === "string") return; // Wisp is binary; a text message is not ours.
      if (data.byteLength > WISP_RECORD_MAX) {
        lose(current);
        return;
      }
      record(WISP_EVENT.MESSAGE, new Uint8Array(data));
    };
    // An error is followed by close; whichever comes first ends the socket.
    current.onerror = () => lose(current);
    current.onclose = () => lose(current);
  };

  // Ends `lost` (once) and schedules the next socket. Its later events are ignored.
  const lose = (lost) => {
    if (socket !== lost) return;
    socket = null;
    lost.onopen = lost.onmessage = lost.onerror = lost.onclose = null;
    if (lost.readyState === SOCKET_CONNECTING || lost.readyState === SOCKET_OPEN) lost.close();
    record(WISP_EVENT.LOST);
    if (closed) return;
    onStatus({ socket: "waiting", retryMs: reconnectMs });
    reconnectTimer = setTimeout(connect, reconnectMs);
    reconnectMs = Math.min(reconnectMs * 2, MAX_RECONNECT_MS);
  };

  connect();
  return {
    send(messages) {
      if (socket?.readyState !== SOCKET_OPEN) return;
      for (const message of messages) socket.send(message);
    },
    status(state) {
      if (!socket) return;
      if (state !== WISP_STATE.DISCONNECTED) {
        acknowledged = true;
        if (state === WISP_STATE.READY) reconnectMs = FIRST_RECONNECT_MS;
      } else if (acknowledged) {
        lose(socket);
      }
    },
    // Stops for good; records that have not fit yet are dropped. Call it once the worker is gone
    // or going.
    close() {
      if (closed) return;
      closed = true;
      if (socket) lose(socket);
      clearTimeout(reconnectTimer);
      clearTimeout(retryTimer);
      pending.length = 0;
      onStatus({ socket: "closed" });
    },
  };
}

// The worker's side: `queue` collects the packets one poll of the emulator produces, `flush`
// posts them to the page as `{ type: "wisp", messages }` through `post(message, transfer)`.
export function createWispSender(post) {
  let messages = [];
  return {
    queue(bytes) {
      messages.push(bytes.buffer);
    },
    flush() {
      if (messages.length === 0) return;
      post({ type: "wisp", messages }, messages);
      messages = [];
    },
  };
}
