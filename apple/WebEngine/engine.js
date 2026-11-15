// The page of the app's WebKit engine (docs/plans/28-apple.md, "WebKitEngine"). It plays the part
// main.js plays on the web page, minus everything visible: it starts the web build's worker.js
// (unchanged), feeds it keystrokes, sizes and the stop flag through the shared control block, and
// passes the worker's console output and events to the app over a loopback WebSocket.
//
// The app serves this page at http://127.0.0.1:<port>/<token>/engine.html?socket=<port>&run=<n>.
// The bridge, one WebSocket per run:
//   page -> app, text first:   "<token> <run>" (the app drops a socket that does not say it)
//   app -> page, text:         {"type": "boot", ...}  once; then {"type": "resize", cols, rows}
//                              and {"type": "stop"}
//   app -> page, binary:       keystrokes for the console
//   page -> app, binary:       console output
//   page -> app, text:         {"type": "started"}, {"type": "exit", code, message},
//                              {"type": "log", text}
import { CONTROL_BYTES, CTRL, RING_BYTES, RING_OFFSET } from "./protocol.js";

const token = location.pathname.split("/")[1];
const params = new URLSearchParams(location.search);
const socket = new WebSocket(`ws://127.0.0.1:${params.get("socket")}/`);
socket.binaryType = "arraybuffer";

let control = null;
let ring = null;
let worker = null;

const sendText = (message) => {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
};

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
  const [kernel, initrd, firmwareImage] = await Promise.all([
    fetchBytes(settings.kernel), fetchBytes(settings.initrd), fetchBytes(settings.firmware),
  ]);
  const shared = new SharedArrayBuffer(CONTROL_BYTES);
  control = new Int32Array(shared, 0, RING_OFFSET / 4);
  ring = new Uint8Array(shared, RING_OFFSET, RING_BYTES);
  control[CTRL.COLS] = settings.cols;
  control[CTRL.ROWS] = settings.rows;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "output") {
      if (socket.readyState === WebSocket.OPEN) socket.send(data.bytes);
    } else if (data.type === "started") {
      sendText({ type: "started" });
    } else if (data.type === "exit") {
      sendText({ type: "exit", code: data.code, message: data.message });
    } else if (data.type === "log") {
      sendText({ type: "log", text: data.text });
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
    memoryMib: settings.memoryMib,
    cpus: settings.cpus,
    cols: settings.cols,
    rows: settings.rows,
    options: settings.options,
    // { url, readOnly }: worker.js reads them with synchronous range requests, which the app's
    // loopback server answers.
    disks: (settings.disks ?? []).map((disk) => ({ url: new URL(disk.url, location.href).href, readOnly: disk.readOnly })),
  }, [kernel, initrd, firmwareImage].filter(Boolean));
}

socket.onopen = () => socket.send(`${token} ${params.get("run")}`);
socket.onmessage = ({ data }) => {
  if (typeof data !== "string") {
    sendKeys(new Uint8Array(data));
    return;
  }
  const message = JSON.parse(data);
  if (message.type === "boot") {
    boot(message).catch((error) => sendText({ type: "exit", code: -2, message: String(error.message ?? error) }));
  } else if (message.type === "resize") {
    resize(message.cols, message.rows);
  } else if (message.type === "stop") {
    stop();
  }
};
// The app went away (or closed this run): nothing may keep running unseen.
socket.onclose = () => {
  stop();
  worker?.terminate();
};
