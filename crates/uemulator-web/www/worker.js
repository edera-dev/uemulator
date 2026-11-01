// Runs the emulator. `uw_run` blocks until the guest exits, so it lives here and not on the page.
// While it runs this worker cannot receive messages: keystrokes, screen input, resizes, frame
// acknowledgements and the stop request come through the SharedArrayBuffer `control` that main.js
// shares with us (layout in protocol.js).
import { CTRL, EVENT_OFFSET, EVENT_RECORDS, EVENT_WORDS, RING_BYTES, RING_OFFSET } from "./protocol.js";
import { createJitHost } from "./jit_host.js";

const OUTPUT_FLUSH_MS = 16;
const OUTPUT_FLUSH_BYTES = 1 << 16;
// Also the latency of a guest timer that falls due while the guest is halted.
const IDLE_WAIT_MS = 1;

let wasm;
let control;
// Disk sources by the number `uw_add_disk` was given: { url } or { file }.
const diskSources = [];
let ring;
let events;
let outputChunks = [];
let outputBytes = 0;
let lastFlush = 0;
let lastResizeGeneration = 0;
const decoder = new TextDecoder();

function memoryBytes(ptr, len) {
  return new Uint8Array(wasm.memory.buffer, ptr, len);
}

function flushOutput() {
  lastFlush = performance.now();
  if (outputBytes === 0) return;
  const merged = new Uint8Array(outputBytes);
  let offset = 0;
  for (const chunk of outputChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  outputChunks = [];
  outputBytes = 0;
  postMessage({ type: "output", bytes: merged.buffer }, [merged.buffer]);
}

// Instantiates the modules the library's WebAssembly JIT generates (see jit_host.js).
const jit = createJitHost();

const host = {
  ...jit.host,
  host_output(ptr, len) {
    outputChunks.push(memoryBytes(ptr, len).slice());
    outputBytes += len;
    if (outputBytes >= OUTPUT_FLUSH_BYTES) flushOutput();
  },
  host_input(ptr, cap) {
    if (performance.now() - lastFlush >= OUTPUT_FLUSH_MS) flushOutput();
    const head = Atomics.load(control, CTRL.HEAD);
    let tail = Atomics.load(control, CTRL.TAIL);
    let count = 0;
    const out = memoryBytes(ptr, cap);
    while (tail !== head && count < cap) {
      out[count++] = ring[tail];
      tail = (tail + 1) % RING_BYTES;
    }
    Atomics.store(control, CTRL.TAIL, tail);
    return count;
  },
  host_resize(ptr) {
    const generation = Atomics.load(control, CTRL.RESIZE_GENERATION);
    if (generation === lastResizeGeneration) return 0;
    lastResizeGeneration = generation;
    // `ptr` is two u16; it is 2-aligned because it points at a Rust `[u16; 2]`.
    const size = new Uint16Array(wasm.memory.buffer, ptr, 2);
    size[0] = Atomics.load(control, CTRL.COLS);
    size[1] = Atomics.load(control, CTRL.ROWS);
    return 1;
  },
  host_should_stop() {
    return Atomics.load(control, CTRL.STOP);
  },
  host_idle() {
    flushOutput();
    const head = Atomics.load(control, CTRL.HEAD);
    const noEvents = Atomics.load(control, CTRL.EVENT_HEAD) === Atomics.load(control, CTRL.EVENT_TAIL);
    if (head === Atomics.load(control, CTRL.TAIL) && noEvents) {
      Atomics.wait(control, CTRL.HEAD, head, IDLE_WAIT_MS);
    }
  },
  host_input_events(ptr, capacity) {
    const head = Atomics.load(control, CTRL.EVENT_HEAD);
    let tail = Atomics.load(control, CTRL.EVENT_TAIL);
    // `ptr` points at a Rust `[i32; N]`, so it is 4-aligned.
    const out = new Int32Array(wasm.memory.buffer, ptr, capacity * EVENT_WORDS);
    let count = 0;
    while (tail !== head && count < capacity) {
      for (let word = 0; word < EVENT_WORDS; word++) out[count * EVENT_WORDS + word] = events[tail * EVENT_WORDS + word];
      count++;
      tail = (tail + 1) % EVENT_RECORDS;
    }
    Atomics.store(control, CTRL.EVENT_TAIL, tail);
    return count;
  },
  host_frame_ready() {
    return Atomics.load(control, CTRL.FRAME_ACK);
  },
  host_frame(screenWidth, screenHeight, x, y, width, height, ptr, len) {
    Atomics.store(control, CTRL.FRAME_ACK, 0);
    // A copy into a buffer of its own: ImageData refuses views on shared memory, and the
    // module's memory moves on as soon as this returns.
    const pixels = memoryBytes(ptr, len).slice();
    postMessage({ type: "frame", screenWidth, screenHeight, x, y, width, height, pixels: pixels.buffer }, [pixels.buffer]);
  },
  host_cursor(kind, width, height, hotX, hotY, ptr, len) {
    const pixels = memoryBytes(ptr, len).slice();
    postMessage({ type: "cursor", kind, width, height, hotX, hotY, pixels: pixels.buffer }, [pixels.buffer]);
  },
  host_random(ptr, len) {
    // getRandomValues takes at most 65536 bytes per call and refuses a view on shared memory.
    const MAX_BYTES = 65536;
    for (let offset = 0; offset < len; offset += MAX_BYTES) {
      const piece = new Uint8Array(Math.min(MAX_BYTES, len - offset));
      crypto.getRandomValues(piece);
      memoryBytes(ptr + offset, piece.length).set(piece);
    }
  },
  host_now_ms() {
    return performance.now();
  },
  host_log(ptr, len) {
    postMessage({ type: "log", text: decoder.decode(memoryBytes(ptr, len)) });
  },
  // Blocks the emulator until the bytes are here: the guest's disk request is served on this
  // thread inside `uw_run`. A worker may make synchronous requests (a page may not).
  host_disk_read(source, offset, ptr, len) {
    try {
      const bytes = readDiskSource(diskSources[source], offset, len);
      if (bytes.byteLength !== len) return 1;
      // After the read: the view must be made on the memory as it is now.
      memoryBytes(ptr, len).set(bytes);
      return 0;
    } catch (error) {
      postMessage({ type: "log", text: `disk read at ${offset} failed: ${error}` });
      return 1;
    }
  },
};

// One synchronous range request. A 200 instead of 206 means the server ignored the range and
// sent the whole image, which would make every read a full download; that is refused.
function rangeRequest(url, first, last) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", url, false);
  xhr.responseType = "arraybuffer";
  xhr.setRequestHeader("Range", `bytes=${first}-${last}`);
  xhr.send();
  if (xhr.status !== 206) {
    throw new Error(`${url}: range request answered with ${xhr.status}; the server must support range requests`);
  }
  return xhr;
}

function readDiskSource(source, offset, len) {
  if (source.file) {
    return new Uint8Array(new FileReaderSync().readAsArrayBuffer(source.file.slice(offset, offset + len)));
  }
  return new Uint8Array(rangeRequest(source.url, offset, offset + len - 1).response);
}

function diskLength(source) {
  if (source.file) return source.file.size;
  const total = rangeRequest(source.url, 0, 0).getResponseHeader("Content-Range")?.split("/")[1];
  if (!total || total === "*") throw new Error(`${source.url}: no length in Content-Range`);
  return Number(total);
}

function lastError() {
  const text = decoder.decode(memoryBytes(wasm.uw_error_ptr(), wasm.uw_error_len()));
  return text;
}

// Copies into module memory. `uw_alloc` may grow (and so detach) the memory, so the view is made
// after it returns.
function upload(bytes) {
  const ptr = wasm.uw_alloc(bytes.byteLength);
  memoryBytes(ptr, bytes.byteLength).set(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength));
  return ptr;
}

async function boot(message) {
  control = new Int32Array(message.control, 0, RING_OFFSET / 4);
  ring = new Uint8Array(message.control, RING_OFFSET, RING_BYTES);
  events = new Int32Array(message.control, EVENT_OFFSET, EVENT_RECORDS * EVENT_WORDS);
  const response = await fetch(new URL("uemulator_web.wasm", import.meta.url));
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { host });
  wasm = instance.exports;
  jit.attach(wasm);

  const kernel = message.kernel ? new Uint8Array(message.kernel) : new Uint8Array(0);
  const initrd = message.initrd ? new Uint8Array(message.initrd) : new Uint8Array(0);
  const cmdline = new TextEncoder().encode(message.cmdline);
  const firmwareImage = message.firmware.image ? new Uint8Array(message.firmware.image) : new Uint8Array(0);
  if (wasm.uw_set_firmware(message.firmware.kind, upload(firmwareImage), firmwareImage.length)) {
    postMessage({ type: "exit", code: -2, message: `firmware: ${lastError()}` });
    return;
  }
  const kernelPtr = upload(kernel);
  const initrdPtr = upload(initrd);
  const cmdlinePtr = upload(cmdline);
  for (const source of message.disks ?? []) {
    const index = diskSources.push(source) - 1;
    if (wasm.uw_add_disk(index, diskLength(source))) {
      postMessage({ type: "exit", code: -2, message: `disk: ${lastError()}` });
      return;
    }
  }
  const failed = wasm.uw_create(
    kernelPtr, kernel.length, initrdPtr, initrd.length, cmdlinePtr, cmdline.length,
    message.memoryMib, message.cpus, Math.floor(Date.now() / 1000), message.cols, message.rows, message.options,
  );
  if (failed) {
    postMessage({ type: "exit", code: -2, message: lastError() });
    return;
  }
  postMessage({ type: "started" });
  lastFlush = performance.now();
  const code = wasm.uw_run();
  flushOutput();
  postMessage({ type: "exit", code, message: code === 4 ? lastError() : "" });
}

onmessage = (event) => {
  boot(event.data).catch((error) => {
    postMessage({ type: "exit", code: -2, message: String(error?.stack ?? error) });
  });
};
