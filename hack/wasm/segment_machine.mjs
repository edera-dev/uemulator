// One machine of hack/wasm/segment.mjs, in a worker thread of its own because `uw_run` blocks:
// the page's module (target/web/uemulator_web.wasm) with the imports www/worker.js gives it, the
// network ones included. Keystrokes, the stop flag, the frames from the other machines and the
// Wisp server's messages arrive through the shared control block (www/protocol.js), which
// segment.mjs fills; console output and the Wisp client's packets go back to it as messages, as
// they go to the page.
import { readFileSync } from "node:fs";
import { Worker, parentPort, workerData } from "node:worker_threads";
import { createCompileQueue, createJitHost } from "../../crates/uemulator-web/www/jit_host.js";
import {
  CTRL, FIRMWARE, NET_OFFSET, NET_RING_BYTES, OPTION, RING_BYTES, RING_OFFSET, WISP_OFFSET, WISP_RING_BYTES,
} from "../../crates/uemulator-web/www/protocol.js";
import { createSegmentSender, netRingHasFrames, takeFrame } from "../../crates/uemulator-web/www/segment.js";
import { createWispSender, takeWispRecord, wispRingHasRecords } from "../../crates/uemulator-web/www/wisp.js";

const IDLE_WAIT_MS = 1;
const TERMINAL_COLUMNS = 160;
const TERMINAL_ROWS = 50;
// `uw_set_wisp`'s resolver, 1.1.1.1 as a big-endian u32, and its port: the library's default,
// given explicitly as the page does.
const WISP_RESOLVER = 0x01010101;
const DNS_PORT = 53;

const { wasmPath, kernelPath, initrdPath, cmdline, memoryMib, firmware, jit: useJit, network, wisp } = workerData;
const control = new Int32Array(workerData.control, 0, RING_OFFSET / 4);
const keys = new Uint8Array(workerData.control, RING_OFFSET, RING_BYTES);
const netRing = new Uint8Array(workerData.control, NET_OFFSET, NET_RING_BYTES);
const wispRing = new Uint8Array(workerData.control, WISP_OFFSET, WISP_RING_BYTES);
const decoder = new TextDecoder();
const segment = createSegmentSender(network.tab);
const wispSender = createWispSender((message, transfer) => parentPort.postMessage(message, transfer));
const jit = createJitHost();
let wasm;
const bytes = (ptr, len) => new Uint8Array(wasm.memory.buffer, ptr >>> 0, len);

const host = {
  ...jit.host,
  host_output(ptr, len) {
    parentPort.postMessage({ type: "output", text: decoder.decode(bytes(ptr, len)) });
  },
  host_input(ptr, capacity) {
    const head = Atomics.load(control, CTRL.HEAD);
    let tail = Atomics.load(control, CTRL.TAIL);
    const out = bytes(ptr, capacity);
    let count = 0;
    while (tail !== head && count < capacity) {
      out[count++] = keys[tail];
      tail = (tail + 1) % RING_BYTES;
    }
    Atomics.store(control, CTRL.TAIL, tail);
    return count;
  },
  host_resize: () => 0,
  host_input_events: () => 0,
  host_frame_ready: () => 0,
  host_frame() {},
  host_cursor() {},
  // No GPU here: nothing calls these without `uw_set_gpu`.
  host_gpu_send: (_ptr, len) => len,
  host_gpu_receive: () => 0,
  host_net_recv(ptr, capacity, fromPtr) {
    const frame = takeFrame(control, netRing, bytes(ptr, capacity));
    if (!frame) return 0;
    new Uint32Array(wasm.memory.buffer, fromPtr >>> 0, 1)[0] = frame.from;
    return frame.length;
  },
  host_net_send(ptr, len, destination, tab) {
    segment.queue(destination, tab >>> 0, bytes(ptr, len).slice());
  },
  host_net_flush() {
    segment.flush();
  },
  host_net_serving: () => Atomics.load(control, CTRL.NET_SERVING),
  host_net_status(address, peers, serving) {
    parentPort.postMessage({ type: "network", address: address >>> 0, peers, serving: serving !== 0 });
  },
  // Without --wisp nothing writes the wisp ring, so the client never connects and the gateway's
  // opens fail at once.
  host_wisp_recv(ptr, capacity, eventPtr) {
    const record = takeWispRecord(control, wispRing, bytes(ptr, capacity));
    if (!record) return -1;
    new Uint32Array(wasm.memory.buffer, eventPtr >>> 0, 1)[0] = record.event;
    return record.length;
  },
  host_wisp_send(ptr, len) {
    wispSender.queue(bytes(ptr, len).slice());
  },
  host_wisp_flush() {
    wispSender.flush();
  },
  host_wisp_status(state, closeReason) {
    parentPort.postMessage({ type: "wisp-status", state, closeReason });
  },
  host_should_stop: () => Atomics.load(control, CTRL.STOP),
  host_idle() {
    const head = Atomics.load(control, CTRL.HEAD);
    const noRings = !netRingHasFrames(control) && !wispRingHasRecords(control);
    if (head === Atomics.load(control, CTRL.TAIL) && noRings) {
      Atomics.wait(control, CTRL.HEAD, head, IDLE_WAIT_MS);
    }
  },
  host_random(ptr, len) {
    crypto.getRandomValues(bytes(ptr, len));
  },
  host_now_ms: () => performance.now(),
  host_log(ptr, len) {
    parentPort.postMessage({ type: "log", text: decoder.decode(bytes(ptr, len)) });
  },
  host_disk_read: () => 1,
};

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), { host });
wasm = instance.exports;
jit.attach(wasm);
if (useJit) {
  const compileQueue = createCompileQueue();
  new Worker(new URL("./jit_compiler.mjs", import.meta.url), { workerData: compileQueue }).unref();
  jit.useCompileQueue(compileQueue);
}

function upload(data) {
  const ptr = wasm.uw_alloc(data.byteLength);
  bytes(ptr, data.byteLength).set(data);
  return ptr;
}
function fail(what) {
  parentPort.postMessage({ type: "exit", code: -2, message: `${what}: ${decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len()))}` });
}

const firmwareKind = FIRMWARE[firmware.toUpperCase()];
const firmwareImage = firmwareKind === FIRMWARE.UEFI
  ? readFileSync(new URL("uemulator-firmware.bin", `file://${wasmPath}`))
  : new Uint8Array(0);
const mac = new Uint8Array(network.mac);
const kernel = readFileSync(kernelPath);
const initrd = readFileSync(initrdPath);
const commandLine = new TextEncoder().encode(cmdline);
if (wasm.uw_set_firmware(firmwareKind, upload(firmwareImage), firmwareImage.length)) {
  fail("firmware");
} else if (wasm.uw_set_network(upload(mac), mac.length, network.tab)) {
  fail("network");
} else if (wisp && wasm.uw_set_wisp(WISP_RESOLVER, DNS_PORT)) {
  fail("wisp");
} else {
  const options = (useJit ? OPTION.JIT : 0) | OPTION.ACPI | OPTION.RNG | OPTION.SERIAL_CONSOLE;
  const failed = wasm.uw_create(
    upload(kernel), kernel.length, upload(initrd), initrd.length, upload(commandLine), commandLine.length,
    memoryMib, 1, Math.floor(Date.now() / 1000), TERMINAL_COLUMNS, TERMINAL_ROWS, options,
  );
  if (failed) {
    fail("create");
  } else {
    const code = wasm.uw_run();
    segment.flush();
    segment.close();
    parentPort.postMessage({ type: "exit", code, message: "" });
  }
}
