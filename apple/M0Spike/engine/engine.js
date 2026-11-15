// The headless page of the M0 spike (docs/plans/28-apple.md). Swift calls `window.spike.*`
// through callAsyncJavaScript; nothing here draws. The emulator itself is the web build's own
// worker.js, unchanged: this file plays the part main.js plays on the web page.
import { CONTROL_BYTES, CTRL, FIRMWARE, OPTION } from "./protocol.js";

const native = window.webkit.messageHandlers.spike;
const log = (text) => native.postMessage({ type: "log", text });

// Smallest modules that use the feature, checked with WebAssembly.validate.
const HEADER = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
const TYPE_VOID = [0x01, 0x04, 0x01, 0x60, 0x00, 0x00];
const ONE_FUNCTION = [0x03, 0x02, 0x01, 0x00];
const FEATURES = {
  // return_call 0
  tailCall: [...HEADER, ...TYPE_VOID, ...ONE_FUNCTION, 0x0a, 0x06, 0x01, 0x04, 0x00, 0x12, 0x00, 0x0b],
  // a funcref table of one, then i32.const 0; return_call_indirect (type 0, table 0): the chain
  // stub's exit (13-web.md, "The JIT")
  tailCallIndirect: [
    ...HEADER, ...TYPE_VOID, ...ONE_FUNCTION, 0x04, 0x04, 0x01, 0x70, 0x00, 0x01,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0x13, 0x00, 0x00, 0x0b,
  ],
  // v128.const 0; drop
  simd: [
    ...HEADER, ...TYPE_VOID, ...ONE_FUNCTION, 0x0a, 0x17, 0x01, 0x15, 0x00, 0xfd, 0x0c,
    ...new Array(16).fill(0), 0x1a, 0x0b,
  ],
};

function inWorker(message) {
  return new Promise((resolve) => {
    const worker = new Worker(new URL("./probe_worker.js", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(data); };
    worker.onerror = (error) => { worker.terminate(); resolve({ error: String(error.message ?? error) }); };
    worker.postMessage(message);
  });
}

async function probes() {
  const features = {};
  for (const [name, bytes] of Object.entries(FEATURES)) features[name] = WebAssembly.validate(new Uint8Array(bytes));
  let sharedMemory = false;
  try {
    sharedMemory = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true }).buffer instanceof SharedArrayBuffer;
  } catch {}
  return {
    userAgent: navigator.userAgent,
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer === "function",
    isSecureContext,
    hardwareConcurrency: navigator.hardwareConcurrency,
    features,
    sharedMemory,
    worker: await inWorker({ origin: location.origin }),
  };
}

// Round trips of `count` frames of `bytes` each, one at a time, so the result is what the screen
// would get with the page's one-frame-in-flight rule (FRAME_ACK).
async function bridge({ port }) {
  const bytes = 1024 * 768 * 4;
  const count = 20;
  const frame = new Uint8Array(bytes);
  for (let i = 0; i < bytes; i++) frame[i] = (i * 2654435761) >>> 24;
  const result = { frameBytes: bytes, frames: count };

  let started = performance.now();
  for (let i = 0; i < count; i++) {
    // Chunked: String.fromCharCode(...) of 3 MiB overflows the argument limit.
    let binary = "";
    for (let at = 0; at < bytes; at += 0x8000) binary += String.fromCharCode(...frame.subarray(at, at + 0x8000));
    const got = await native.postMessage({ type: "frame", data: btoa(binary) });
    if (got !== bytes) throw new Error(`postMessage: native got ${got}`);
  }
  result.postMessageMsPerFrame = (performance.now() - started) / count;

  try {
    const socket = new WebSocket(`ws://127.0.0.1:${port}/`);
    socket.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = () => reject(new Error("WebSocket failed")); });
    const replies = [];
    let wake = null;
    socket.onmessage = ({ data }) => { replies.push(data); wake?.(); };
    const next = () => (replies.length ? Promise.resolve(replies.shift()) : new Promise((resolve) => { wake = () => { wake = null; resolve(replies.shift()); }; }));
    started = performance.now();
    for (let i = 0; i < count; i++) {
      socket.send(frame);
      const got = Number(await next());
      if (got !== bytes) throw new Error(`WebSocket: native got ${got}`);
    }
    result.webSocketMsPerFrame = (performance.now() - started) / count;
    socket.close();
  } catch (error) {
    result.webSocketError = String(error.message ?? error);
  }
  return result;
}

// Text that marks the boot's progress: the kernel handing over to the initramfs, and the demo's
// /init reaching its shell (hack/web/demo.py).
const MARKS = { init: "Run /init", shell: "running from RAM" };
const BOOT_TIMEOUT_MS = 600_000;
const CONSOLE_FLUSH_MS = 100;

async function boot({ jit }) {
  const fetchBytes = async (path) => (await fetch(new URL(path, location.href))).arrayBuffer();
  const [kernel, initrd] = await Promise.all([fetchBytes("demo/vmlinuz-virt"), fetchBytes("demo/initramfs.gz")]);
  const shared = new SharedArrayBuffer(CONTROL_BYTES);
  const control = new Int32Array(shared, 0, 16);
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const decoder = new TextDecoder();
  const marks = {};
  let seen = "";
  let pending = "";
  let flushTimer = null;
  const started = performance.now();
  const flush = () => { flushTimer = null; if (pending) native.postMessage({ type: "console", text: pending }); pending = ""; };

  const exit = await new Promise((resolve) => {
    const timeout = setTimeout(() => {
      Atomics.store(control, CTRL.STOP, 1);
      Atomics.notify(control, CTRL.HEAD);
    }, BOOT_TIMEOUT_MS);
    worker.onmessage = ({ data }) => {
      if (data.type === "output") {
        const text = decoder.decode(new Uint8Array(data.bytes), { stream: true });
        pending += text;
        flushTimer ??= setTimeout(flush, CONSOLE_FLUSH_MS);
        // Only the tail can hold a mark split across chunks.
        seen = (seen + text).slice(-4096);
        for (const [name, needle] of Object.entries(MARKS)) {
          if (!(name in marks) && seen.includes(needle)) marks[name] = Math.round(performance.now() - started);
        }
        if ("shell" in marks) {
          Atomics.store(control, CTRL.STOP, 1);
          Atomics.notify(control, CTRL.HEAD);
        }
      } else if (data.type === "log") {
        log(data.text);
      } else if (data.type === "started") {
        marks.started = Math.round(performance.now() - started);
      } else if (data.type === "exit") {
        clearTimeout(timeout);
        resolve({ code: data.code, message: data.message });
      }
    };
    worker.onerror = (error) => resolve({ code: -3, message: String(error.message ?? error) });
    worker.postMessage({
      control: shared,
      compileQueue: null,
      kernel,
      initrd,
      // boot.mjs's default, appended to the library's command line.
      cmdline: "earlycon=uart8250,io,0x3f8",
      firmware: { kind: FIRMWARE.NONE, image: null },
      memoryMib: 256,
      cpus: 1,
      cols: 80,
      rows: 25,
      options: (jit ? OPTION.JIT : 0) | OPTION.ACPI | OPTION.RNG,
      disks: [],
    });
  });
  flush();
  worker.terminate();
  return { jit, marks, exit, totalMs: Math.round(performance.now() - started) };
}

window.spike = { probes, bridge, boot };
