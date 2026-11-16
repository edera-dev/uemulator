// The page of the app's WebKit engine (docs/plans/28-apple.md, "WebKitEngine"). It plays the part
// main.js plays on the web page, minus everything visible: it starts the web build's worker.js
// (unchanged), feeds it keystrokes, screen input, sizes and the stop flag through the shared
// control block, and passes the worker's console output, frames and events to the app over a
// loopback WebSocket.
//
// The app serves this page at http://127.0.0.1:<port>/<token>/engine.html?socket=<port>&wisp=<port>&run=<n>.
// The bridge, one WebSocket per run (WebKitEngine.swift reads and writes the same layout):
//   page -> app, text first:  "<token> <run>" (the app drops a socket that does not say it)
//   app -> page, text:        {"type": "boot", ...} once, or {"type": "reattached"} to a page
//                             that reconnected (below); then {"type": "resize", cols, rows},
//                             {"type": "stop"} and {"type": "frame-ack"} (the app drew the last
//                             frame: the worker may send the next, protocol.js FRAME_ACK)
//   app -> page, binary:      a kind byte, then
//                             KEYS: keystrokes for the console
//                             EVENTS: screen input, little-endian Int32 triples (protocol.js EVENT)
//   page -> app, binary:      a kind byte, then
//                             OUTPUT: console output
//                             FRAME: width, height, rect count (u32 LE each), the rects (x, y,
//                                    width, height, u32 LE), then each rect's pixels packed
//                             CURSOR: kind, width, height, hot x, hot y (u32 LE), BGRA pixels
//   page -> app, text:        {"type": "started"}, {"type": "exit", code, message},
//                             {"type": "log", text}
//
// The socket can drop under a running machine: iOS reclaims a suspended app's sockets, listening
// ones included, and the app reopens its servers on the same ports when it comes back
// (apple/App/Loopback.swift). So a closed socket is not the end of the run: the page connects
// again with the same token and run and the worker keeps going (its disk requests wait for the
// server, worker.js `sendSync`). Only after RECONNECT_ATTEMPTS without a word from the app does
// the page give up, and then it stops the worker and lets it close its disks before ending it:
// terminating it at once would lose the udisk map updates the guest has not flushed.
import {
  CONTROL_BYTES, CTRL, EVENT_OFFSET, EVENT_RECORDS, EVENT_WORDS, FRAME_READY, RING_BYTES, RING_OFFSET,
  WISP_OFFSET, WISP_RING_BYTES,
} from "./protocol.js";
import { randomMac, randomTab } from "./segment.js";
import { connectWisp } from "./wisp.js";

// Where the guest's DNS goes through Wisp (main.js's WISP_RESOLVER: 1.1.1.1:53).
const WISP_RESOLVER = 0x01010101;
const WISP_RESOLVER_PORT = 53;

const TO_PAGE = { KEYS: 1, EVENTS: 2 };
const TO_APP = { OUTPUT: 1, FRAME: 2, CURSOR: 3 };
const U32 = 4;

// Timers do not run while the app and so this page are suspended, so this counts attempts made
// while awake: about a minute of the app being back and not answering.
const RECONNECT_ATTEMPTS = 60;
const RECONNECT_DELAY_MS = 1000;
// How long a stopped worker gets to close its disks once the app is gone (WebKitEngine's
// drainTimeout).
const STOP_DRAIN_MS = 5000;

const token = location.pathname.split("/")[1];
const params = new URLSearchParams(location.search);
let socket = null;
// Attempts since the app last said anything; reset by any message from it.
let unanswered = 0;
let booted = false;
let gaveUp = false;
// Console output made while the socket is down, sent when the app takes the page back, so the
// terminal does not miss what an installer printed meanwhile. Frames are not kept: the reattach
// asks for the whole screen anyway.
const MAX_HELD_OUTPUT = 1 << 20;
let heldOutput = [];
let heldBytes = 0;

let control = null;
let ring = null;
let events = null;
let worker = null;
// The guest's internet: a WebSocket to the app's Wisp server (connectWisp), when the machine has a
// network.
let wisp = null;

const sendText = (message) => {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
};

// A kind byte, `words` as u32 little-endian, then `payloads` back to back.
function sendBinary(kind, words, payloads) {
  if (socket.readyState !== WebSocket.OPEN) return;
  let length = 1 + words.length * U32;
  for (const payload of payloads) length += payload.byteLength;
  const bytes = new Uint8Array(length);
  const view = new DataView(bytes.buffer);
  bytes[0] = kind;
  words.forEach((word, index) => view.setUint32(1 + index * U32, word, true));
  let at = 1 + words.length * U32;
  for (const payload of payloads) {
    bytes.set(new Uint8Array(payload.buffer ?? payload, payload.byteOffset ?? 0, payload.byteLength), at);
    at += payload.byteLength;
  }
  socket.send(bytes);
}

// The worker's own exit, once it has said it (worker.onmessage).
let workerExited = false;
// main.js's sendKeys: drop what does not fit rather than block.
function sendKeys(bytes) {
  if (!control) return;
  let head = Atomics.load(control, CTRL.HEAD);
  const tail = Atomics.load(control, CTRL.TAIL);
  for (const byte of bytes) {
    const next = (head + 1) % RING_BYTES;
    if (next === tail) break;
    ring[head] = byte;
    head = next;
  }
  Atomics.store(control, CTRL.HEAD, head);
  Atomics.notify(control, CTRL.HEAD);
}

// main.js's pushEvent, for each triple: a full ring drops the rest.
function pushEvents(words) {
  if (!control) return;
  let head = Atomics.load(control, CTRL.EVENT_HEAD);
  const tail = Atomics.load(control, CTRL.EVENT_TAIL);
  for (let at = 0; at + EVENT_WORDS <= words.length; at += EVENT_WORDS) {
    const next = (head + 1) % EVENT_RECORDS;
    if (next === tail) break;
    events.set(words.subarray(at, at + EVENT_WORDS), head * EVENT_WORDS);
    head = next;
  }
  Atomics.store(control, CTRL.EVENT_HEAD, head);
  // The worker's idle wait sleeps on HEAD.
  Atomics.notify(control, CTRL.HEAD);
}

function resize(cols, rows) {
  if (!control) return;
  Atomics.store(control, CTRL.COLS, cols);
  Atomics.store(control, CTRL.ROWS, rows);
  Atomics.add(control, CTRL.RESIZE_GENERATION, 1);
  Atomics.notify(control, CTRL.HEAD);
}

function stop() {
  if (!control) return;
  Atomics.store(control, CTRL.STOP, 1);
  // The idle wait sleeps on HEAD (protocol.js).
  Atomics.notify(control, CTRL.HEAD);
}

async function boot(settings) {
  const fetchBytes = async (path) => {
    if (!path) return null;
    const response = await fetch(new URL(path, location.href));
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.arrayBuffer();
  };
  const [kernel, initrd, firmwareImage, hypervisor] = await Promise.all([
    fetchBytes(settings.kernel), fetchBytes(settings.initrd), fetchBytes(settings.firmware),
    fetchBytes(settings.hypervisor),
  ]);
  const shared = new SharedArrayBuffer(CONTROL_BYTES);
  control = new Int32Array(shared, 0, RING_OFFSET / 4);
  ring = new Uint8Array(shared, RING_OFFSET, RING_BYTES);
  events = new Int32Array(shared, EVENT_OFFSET, EVENT_RECORDS * EVENT_WORDS);
  control[CTRL.COLS] = settings.cols;
  control[CTRL.ROWS] = settings.rows;
  control[CTRL.FRAME_ACK] = FRAME_READY.YES;
  let network = null;
  if (settings.network) {
    // The only machine on its segment (no other tab shares this page's BroadcastChannel), so it
    // is the segment's server: its agent answers DHCP (segment.js; the web page takes a lock).
    Atomics.store(control, CTRL.NET_SERVING, 1);
    network = { mac: randomMac(), tab: randomTab() };
    wisp = connectWisp({
      control,
      ring: new Uint8Array(shared, WISP_OFFSET, WISP_RING_BYTES),
      url: `ws://127.0.0.1:${params.get("wisp")}/`,
      // The app's server takes only the launch token (apple/App/Wisp.swift).
      protocols: [token],
    });
  }
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "output") {
      if (socket.readyState === WebSocket.OPEN) {
        sendBinary(TO_APP.OUTPUT, [], [data.bytes]);
      } else if (heldBytes + data.bytes.byteLength <= MAX_HELD_OUTPUT) {
        heldOutput.push(data.bytes);
        heldBytes += data.bytes.byteLength;
      }
    } else if (data.type === "frame") {
      const rectCount = data.rects.length / 4;
      sendBinary(TO_APP.FRAME, [data.screenWidth, data.screenHeight, rectCount], [data.rects, data.pixels]);
    } else if (data.type === "cursor") {
      sendBinary(TO_APP.CURSOR, [data.kind, data.width, data.height, data.hotX, data.hotY], [data.pixels]);
    } else if (data.type === "started") {
      sendText({ type: "started" });
    } else if (data.type === "exit") {
      workerExited = true;
      sendText({ type: "exit", code: data.code, message: data.message });
    } else if (data.type === "log") {
      sendText({ type: "log", text: data.text });
    } else if (data.type === "wisp") {
      wisp?.send(data.messages);
    } else if (data.type === "wisp-status") {
      // Both its backoff and its handshake timeout depend on hearing these (wisp.js).
      wisp?.status(data.state);
    } else if (data.type === "network") {
      sendText({ type: "log", text: `network: address ${data.address >>> 0}, ${data.peers} peers` });
    }
  };
  worker.onerror = (error) => sendText({ type: "exit", code: -3, message: String(error.message ?? error) });
  worker.postMessage({
    control: shared,
    // The compile worker only helps on V8 (jit_host.js, compileWorkerHelps).
    compileQueue: null,
    kernel,
    initrd,
    cmdline: settings.cmdline,
    firmware: { kind: settings.firmwareKind, image: firmwareImage },
    hypervisor,
    hypervisorCmdline: settings.hypervisorCmdline,
    memoryMib: settings.memoryMib,
    cpus: settings.cpus,
    cols: settings.cols,
    rows: settings.rows,
    options: settings.options,
    jitChaining: settings.jitChaining,
    network,
    wisp: network ? { resolver: WISP_RESOLVER, port: WISP_RESOLVER_PORT } : null,
    // worker.js reads them with synchronous range requests, which the app's loopback server
    // answers; a writable disk writes through to the app's file with synchronous POSTs to the
    // same URL.
    disks: (settings.disks ?? []).map((disk) => ({
      url: new URL(disk.url, location.href).href, readOnly: disk.readOnly, writeThrough: !disk.readOnly,
    })),
  }, [kernel, initrd, firmwareImage, hypervisor].filter(Boolean));
}

function onMessage({ data }) {
  unanswered = 0;
  if (typeof data !== "string") {
    const bytes = new Uint8Array(data);
    if (bytes[0] === TO_PAGE.KEYS) {
      sendKeys(bytes.subarray(1));
    } else if (bytes[0] === TO_PAGE.EVENTS) {
      // Copied: Int32Array needs 4-byte alignment and the payload starts at byte 1.
      pushEvents(new Int32Array(bytes.slice(1).buffer));
    }
    return;
  }
  const message = JSON.parse(data);
  if (message.type === "boot") {
    // A page that reconnected before the app saw its first socket may be sent the boot twice.
    if (booted) return;
    booted = true;
    boot(message).catch((error) => sendText({ type: "exit", code: -2, message: String(error.message ?? error) }));
  } else if (message.type === "reattached") {
    // The frame in flight when the socket dropped was lost and so was its acknowledgement: without
    // this the worker never sends another. RESEND: the whole screen, not what changed since.
    if (control) Atomics.store(control, CTRL.FRAME_ACK, FRAME_READY.RESEND);
    for (const bytes of heldOutput) sendBinary(TO_APP.OUTPUT, [], [bytes]);
    heldOutput = [];
    heldBytes = 0;
  } else if (message.type === "resize") {
    resize(message.cols, message.rows);
  } else if (message.type === "stop") {
    stop();
  } else if (message.type === "frame-ack" && control) {
    Atomics.store(control, CTRL.FRAME_ACK, FRAME_READY.YES);
  }
}

// The app is gone for good, or will not take this run back: nothing may keep running unseen.
function giveUp() {
  if (gaveUp) return;
  gaveUp = true;
  stop();
  wisp?.close();
  if (!worker) return;
  const started = Date.now();
  const end = () => {
    if (workerExited || Date.now() - started >= STOP_DRAIN_MS) {
      worker.terminate();
    } else {
      setTimeout(end, RECONNECT_DELAY_MS / 10);
    }
  };
  end();
}

function connect() {
  socket = new WebSocket(`ws://127.0.0.1:${params.get("socket")}/`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => socket.send(`${token} ${params.get("run")}`);
  socket.onmessage = onMessage;
  socket.onclose = () => {
    unanswered += 1;
    if (unanswered > RECONNECT_ATTEMPTS) {
      giveUp();
    } else {
      setTimeout(connect, RECONNECT_DELAY_MS);
    }
  };
}

connect();
