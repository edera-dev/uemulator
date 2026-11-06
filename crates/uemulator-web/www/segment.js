// The transport of the Ethernet segment the tabs' machines share (docs/plans/13-web.md). The
// library's SegmentAgent decides where every frame goes; this only carries frames between tabs
// over BroadcastChannel and holds the Web Lock that makes one tab the segment's DHCP server.
//
// The emulator's worker cannot receive messages while `uw_run` runs, so the two halves live in
// different places: `joinSegment` on the page (or any thread with an event loop) listens and
// writes what arrives into the frame ring of the shared control block, which the worker drains
// in `host_net_recv` (`takeFrame`); `createSegmentSender` in the worker posts what the guest's
// agent hands it. No DOM here, so hack/wasm/segment.mjs runs the same code under Node.
import { CTRL, DESTINATION, SEGMENT } from "./protocol.js";

const tabChannel = (tab) => `${SEGMENT.CHANNEL}/tab/${tab}`;
const SERVER_CHANNEL = `${SEGMENT.CHANNEL}/server`;
// A record in the frame ring: the frame's length and the sending tab's number, both u32
// little-endian, then the frame.
const RECORD_HEADER_BYTES = 8;
// Channels to other tabs a sender keeps open; past this the oldest is closed (and reopened if
// it is needed again). Only a segment with more tabs than this pays for it.
const MAX_TAB_CHANNELS = 64;
// Both the page's BroadcastChannel and `host_net_send`'s numbers are u32.
const MAX_TAB = 0xffff_ffff;

// A number for this tab, unique among the tabs with overwhelming probability. Zero is avoided
// only so that a stray zero-initialised `from` is never mistaken for a tab.
export function randomTab() {
  const words = new Uint32Array(1);
  do crypto.getRandomValues(words); while (words[0] === 0 || words[0] > MAX_TAB);
  return words[0];
}

// A random unicast MAC from the locally administered space (I/G bit clear, U/L bit set), so
// two tabs practically never pick the same one and none is ever a real interface's.
export function randomMac() {
  const mac = crypto.getRandomValues(new Uint8Array(6));
  mac[0] = (mac[0] & 0xfc) | 0x02;
  return mac;
}

export const formatMac = (mac) => [...mac].map((byte) => byte.toString(16).padStart(2, "0")).join(":");

// Copies `bytes` into the byte ring at `at`, wrapping at its end, and returns where it ended.
export function ringWrite(ring, at, bytes) {
  const first = Math.min(bytes.length, ring.length - at);
  ring.set(bytes.subarray(0, first), at);
  ring.set(bytes.subarray(first), 0);
  return (at + bytes.length) % ring.length;
}

export function ringRead(ring, at, out) {
  const first = Math.min(out.length, ring.length - at);
  out.set(ring.subarray(at, at + first));
  out.set(ring.subarray(0, out.length - first), first);
  return (at + out.length) % ring.length;
}

// Appends one frame to the ring and wakes the worker. A full ring drops the frame, as a NIC
// out of receive buffers would; the guests' protocols cope with loss.
function putFrame(control, ring, from, frame) {
  const head = Atomics.load(control, CTRL.NET_HEAD);
  const tail = Atomics.load(control, CTRL.NET_TAIL);
  const used = (head - tail + ring.length) % ring.length;
  // One byte stays free, so a full ring is never mistaken for an empty one.
  if (frame.length === 0 || used + RECORD_HEADER_BYTES + frame.length >= ring.length) return false;
  const header = new Uint8Array(RECORD_HEADER_BYTES);
  const view = new DataView(header.buffer);
  view.setUint32(0, frame.length, true);
  view.setUint32(4, from, true);
  const next = ringWrite(ring, ringWrite(ring, head, header), frame);
  Atomics.store(control, CTRL.NET_HEAD, next);
  Atomics.notify(control, CTRL.HEAD);
  return true;
}

// The worker's side of the ring: copies the next frame into `out` (a view of at most its
// capacity) and returns { length, from }, or null when the ring is empty. A frame longer than
// `out` is skipped.
export function takeFrame(control, ring, out) {
  for (;;) {
    const head = Atomics.load(control, CTRL.NET_HEAD);
    let tail = Atomics.load(control, CTRL.NET_TAIL);
    if (head === tail) return null;
    const header = new Uint8Array(RECORD_HEADER_BYTES);
    tail = ringRead(ring, tail, header);
    const view = new DataView(header.buffer);
    const length = view.getUint32(0, true);
    const from = view.getUint32(4, true);
    if (length > out.length) {
      Atomics.store(control, CTRL.NET_TAIL, (tail + length) % ring.length);
      continue;
    }
    tail = ringRead(ring, tail, out.subarray(0, length));
    Atomics.store(control, CTRL.NET_TAIL, tail);
    return { length, from };
  }
}

export function netRingHasFrames(control) {
  return Atomics.load(control, CTRL.NET_HEAD) !== Atomics.load(control, CTRL.NET_TAIL);
}

// Puts this tab on the segment: listens on the shared channel and its own, and asks for the
// server lock, holding it (and listening on the server channel) until `leave`. `control` and
// `ring` are the shared control block's Int32 words and frame ring. `onServing(true)` when
// this tab becomes the server. Call `leave` only after the worker is gone: leaving hands the
// lock to another tab, and the old worker must not answer for the server after that.
export function joinSegment({ control, ring, tab, onServing = () => {} }) {
  let left = false;
  let dropped = 0;
  const receive = ({ data }) => {
    if (left || data?.from === tab) return;
    for (const frame of data.frames) {
      if (!putFrame(control, ring, data.from, new Uint8Array(frame))) dropped++;
    }
  };
  const channels = [new BroadcastChannel(SEGMENT.CHANNEL), new BroadcastChannel(tabChannel(tab))];
  for (const channel of channels) channel.onmessage = receive;

  const abort = new AbortController();
  let release;
  const held = new Promise((resolve) => { release = resolve; });
  // Without Web Locks (an insecure context, which a cross-origin isolated page never is) there
  // is no server: the guests can still reach each other by addresses they configure themselves.
  navigator.locks?.request(SEGMENT.LOCK, { signal: abort.signal }, async () => {
    if (left) return;
    const server = new BroadcastChannel(SERVER_CHANNEL);
    server.onmessage = receive;
    channels.push(server);
    Atomics.store(control, CTRL.NET_SERVING, 1);
    onServing(true);
    await held;
  }).catch(() => { /* AbortError: left before it was granted */ });

  return {
    get dropped() { return dropped; },
    leave() {
      if (left) return;
      left = true;
      Atomics.store(control, CTRL.NET_SERVING, 0);
      for (const channel of channels) channel.close();
      abort.abort();
      release();
    },
  };
}

// The worker's side: `queue` collects the frames one poll of the emulator produces, `flush`
// posts them, one message per destination channel.
export function createSegmentSender(tab) {
  const everyone = new BroadcastChannel(SEGMENT.CHANNEL);
  const server = new BroadcastChannel(SERVER_CHANNEL);
  // Insertion order is age: the first entry is the one to close when there are too many.
  const tabs = new Map();
  const pending = new Map();
  const channelFor = (destination, toTab) => {
    if (destination === DESTINATION.SERVER) return server;
    if (destination !== DESTINATION.TAB) return everyone;
    let channel = tabs.get(toTab);
    if (channel) {
      tabs.delete(toTab);
    } else {
      channel = new BroadcastChannel(tabChannel(toTab));
      if (tabs.size >= MAX_TAB_CHANNELS) {
        const [oldest, old] = tabs.entries().next().value;
        old.close();
        tabs.delete(oldest);
        pending.delete(old);
      }
    }
    tabs.set(toTab, channel);
    return channel;
  };
  return {
    queue(destination, toTab, frame) {
      const channel = channelFor(destination, toTab);
      let frames = pending.get(channel);
      if (!frames) pending.set(channel, (frames = []));
      frames.push(frame.buffer);
    },
    flush() {
      for (const [channel, frames] of pending) channel.postMessage({ from: tab, frames });
      pending.clear();
    },
    close() {
      for (const channel of [everyone, server, ...tabs.values()]) channel.close();
    },
  };
}
