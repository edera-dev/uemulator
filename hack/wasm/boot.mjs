// Boots a Linux kernel on the browser build (target/web/uemulator_web.wasm) under Node: the same
// module and the same host imports as www/worker.js, with the console on stdout. For measuring
// and for checking the WebAssembly JIT against a real guest without a browser.
//
//   node hack/wasm/boot.mjs [--kernel K] [--initrd I] [--disk D]... [--cdrom ISO]... [--cmdline S] [--memory MiB]
//                           [--firmware none|uefi|bios] [--cpus N] [--iommu] [--tpm] [--secure-boot] [--no-acpi] [--serial]
//                           [--hypervisor XEN [--hypervisor-cmdline S]]
//                           [--gpu] [--no-jit] [--no-compile-worker] [--debug] [--wasm FILE] [--until TEXT]
//                           [--seconds N] [--mark TEXT]... [--keys TEXT [--keys-after TEXT]]
//
// --disk is a raw or udisk image (a file, or an http(s) URL served with range requests) attached
// the way the page attaches one: read on demand, writes kept in memory. --cdrom attaches an image
// read-only, as the page attaches an ISO. Both repeat; the drives are vda, vdb, ... with every
// --disk before every --cdrom. --cmdline is
// appended to the library's default command line, as on the page. --firmware uefi reads
// target/web/uemulator-firmware.bin (hack/web.sh builds it) and boots the kernel through it, or the
// disk when there is no kernel. --debug turns on the library's debug mode: on exit the state dump,
// with the JIT's counters and the instructions that went through the interpreter, goes to stderr
// (it costs time; do not measure with it). --no-compile-worker compiles the JIT's modules on the
// emulator's thread, as engines other than V8 do (www/jit_host.js); `compile=` is the time that
// thread spent instantiating them either way. --wasm boots another build of the module, for comparing
// two builds (with --firmware uefi the firmware image is looked for next to it). --hypervisor
// boots Xen (xen.gz, xen or xen.efi) through the UEFI firmware with the kernel as its dom0, as the
// page does; --cmdline is then dom0's and --hypervisor-cmdline Xen's (empty: the library's default).
//
// --gpu gives the machine a screen and a 3D GPU (uw_set_gpu, with GLES 3.0's minimum limits)
// whose commands go to www/gpu.js's renderer here, with a backend that draws nothing and logs
// each clear, scanout and flush to stderr: a guest's virgl traffic and its fences without a
// browser (docs/plans/25-gpu-3d.md).
//
// Stops when TEXT shows up on the console or after N seconds, and prints the wall time and what the
// JIT host did to stderr. --mark prints the time since the start when its text first shows up, for
// timing the stages of a boot. --keys types TEXT on the console (JSON string escapes: "\r" is Enter)
// once --keys-after's text has shown up, or at once: a boot loader menu that waits for a key.
import { closeSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { Worker } from "node:worker_threads";
import { parseArgs } from "node:util";
import { createCompileQueue, createJitHost } from "../../crates/uemulator-web/www/jit_host.js";
import { FIRMWARE, OPTION } from "../../crates/uemulator-web/www/protocol.js";
import {
  COMPLETION, LIMIT_PARAMETERS, MINIMUM_LIMITS, Renderer, encodeRecord, splitRecords,
} from "../../crates/uemulator-web/www/gpu.js";

const { values } = parseArgs({
  options: {
    kernel: { type: "string" },
    initrd: { type: "string" },
    disk: { type: "string", multiple: true, default: [] },
    cdrom: { type: "string", multiple: true, default: [] },
    cmdline: { type: "string", default: "earlycon=uart8250,io,0x3f8" },
    memory: { type: "string", default: "256" },
    cpus: { type: "string", default: "1" },
    firmware: { type: "string", default: "none" },
    iommu: { type: "boolean", default: false },
    tpm: { type: "boolean", default: false },
    "secure-boot": { type: "boolean", default: false },
    "no-acpi": { type: "boolean", default: false },
    serial: { type: "boolean", default: false },
    hypervisor: { type: "string" },
    "hypervisor-cmdline": { type: "string", default: "" },
    wasm: { type: "string", default: new URL("../../target/web/uemulator_web.wasm", import.meta.url).pathname },
    gpu: { type: "boolean", default: false },
    "no-jit": { type: "boolean", default: false },
    "no-compile-worker": { type: "boolean", default: false },
    debug: { type: "boolean", default: false },
    until: { type: "string" },
    seconds: { type: "string", default: "120" },
    quiet: { type: "boolean", default: false },
    mark: { type: "string", multiple: true, default: [] },
    keys: { type: "string" },
    "keys-after": { type: "string" },
  },
});
if (!values.kernel && !values.disk.length && !values.cdrom.length) throw new Error("--kernel, --disk or --cdrom is required");

const start = performance.now();
const deadline = start + Number(values.seconds) * 1000;
let wasm;
let seen = "";
let found = false;
let compileMs = 0;
let compiled = 0;
const decoder = new TextDecoder();
const jit = createJitHost();
// Pointers arrive as signed i32: `>>> 0` makes the ones past 2 GiB of module memory positive.
const bytes = (ptr, len) => new Uint8Array(wasm.memory.buffer, ptr >>> 0, len);
const sleeper = new Int32Array(new SharedArrayBuffer(4));
const marks = values.mark.map((text) => ({ text, at: undefined }));
let pendingKeys = values.keys === undefined ? undefined : new TextEncoder().encode(JSON.parse(`"${values.keys}"`));
let keysArmed = values["keys-after"] === undefined;

// Completions the renderer queued, as records, until the module asks for them.
let gpuCompletions = [];
const gpuLog = (text) => console.error(`gpu: ${text}`);
const describe = (resource) => `resource ${resource.id} ${resource.width}x${resource.height} format ${resource.format} bind 0x${resource.bind.toString(16)}`;
const gpuRenderer = new Renderer(
  {
    resource: () => {},
    destroyResource: () => {},
    clear: (resource, rgba) => gpuLog(`clear ${describe(resource)} to ${rgba.map((c) => c.toFixed(3)).join(",")}`),
    scanout: (resource, rect) => gpuLog(`scanout ${resource ? describe(resource) : "off"} ${rect.width}x${rect.height}`),
    flush: (resource) => gpuLog(`flush resource ${resource.id}`),
    reset: () => gpuLog("reset"),
  },
  (kind, words, data) => {
    gpuCompletions.push(encodeRecord(kind, words, data));
    if (kind === COMPLETION.FENCE) gpuFences++;
  },
  gpuLog,
);
let gpuFences = 0;

const host = {
  ...jit.host,
  host_jit_compile(ptr, len, slots, count) {
    const before = performance.now();
    const status = jit.host.host_jit_compile(ptr, len, slots, count);
    compileMs += performance.now() - before;
    compiled++;
    if (status < 0) console.error("jit: module refused:", jit.host.lastError);
    return status;
  },
  host_output(ptr, len) {
    const text = decoder.decode(bytes(ptr, len));
    if (!values.quiet) process.stdout.write(text);
    seen = (seen + text).slice(-4096);
    if (values.until && seen.includes(values.until)) found = true;
    if (!keysArmed && seen.includes(values["keys-after"])) keysArmed = true;
    for (const mark of marks) {
      if (mark.at === undefined && seen.includes(mark.text)) {
        mark.at = performance.now() - start;
        console.error(`\nmark ${(mark.at / 1000).toFixed(2)}s ${JSON.stringify(mark.text)}`);
      }
    }
  },
  host_input(ptr, cap) {
    if (!keysArmed || !pendingKeys?.length) return 0;
    const count = Math.min(cap, pendingKeys.length);
    bytes(ptr, count).set(pendingKeys.subarray(0, count));
    pendingKeys = pendingKeys.subarray(count);
    return count;
  },
  host_resize: () => 0,
  // No screen here: nothing to draw and no keyboard or pointer events.
  host_input_events: () => 0,
  host_frame_ready: () => 0,
  host_frame() {},
  host_cursor() {},
  host_gpu_send(ptr, len) {
    for (const { kind, payload } of splitRecords(bytes(ptr, len))) gpuRenderer.handle(kind, payload);
    return len;
  },
  host_gpu_receive(ptr, capacity) {
    let copied = 0;
    while (gpuCompletions.length) {
      const record = gpuCompletions[0];
      if (copied + record.length > capacity) return copied || -record.length;
      bytes(ptr + copied, record.length).set(record);
      copied += record.length;
      gpuCompletions.shift();
    }
    return copied;
  },
  // No network here (hack/wasm/segment.mjs boots machines on a shared segment).
  host_net_recv: () => 0,
  host_net_send() {},
  host_net_flush() {},
  host_net_serving: () => 0,
  host_net_status() {},
  host_should_stop: () => (found || performance.now() > deadline ? 1 : 0),
  host_idle: () => Atomics.wait(sleeper, 0, 0, 1),
  host_random(ptr, len) {
    crypto.getRandomValues(bytes(ptr, len));
  },
  host_now_ms: () => performance.now(),
  host_log: (ptr, len) => console.error("log:", decoder.decode(bytes(ptr, len))),
  host_disk_read(source, offset, ptr, len) {
    try {
      const data = diskSources[source].read(offset, len);
      if (data.length !== len) return 1;
      bytes(ptr, len).set(data);
      diskReads++;
      diskBytes += len;
      return 0;
    } catch (error) {
      console.error("disk:", error.message);
      return 1;
    }
  },
};

// Node has no synchronous fetch; a URL is read with curl, which is slow per request but blocks,
// like the page's synchronous XMLHttpRequest. The library caches and reads ahead either way.
function openDiskSource(name) {
  if (/^https?:\/\//.test(name)) {
    const head = execFileSync("curl", ["-sf", "-r", "0-0", "-D", "-", "-o", "/dev/null", name], { encoding: "utf8" });
    const total = /content-range: bytes 0-0\/(\d+)/i.exec(head)?.[1];
    if (!total) throw new Error(`${name}: no range support`);
    return {
      length: Number(total),
      read: (offset, len) =>
        execFileSync("curl", ["-sf", "-r", `${offset}-${offset + len - 1}`, name], { maxBuffer: len + 1 }),
    };
  }
  const fd = openSync(name, "r");
  return {
    length: fstatSync(fd).size,
    read(offset, len) {
      const out = Buffer.alloc(len);
      const count = readSync(fd, out, 0, len, offset);
      return out.subarray(0, count);
    },
    close: () => closeSync(fd),
  };
}
const diskSources = [];
let diskReads = 0;
let diskBytes = 0;

const { instance } = await WebAssembly.instantiate(readFileSync(values.wasm), { host });
wasm = instance.exports;
jit.attach(wasm);
if (!values["no-jit"] && !values["no-compile-worker"]) {
  const compileQueue = createCompileQueue();
  new Worker(new URL("./jit_compiler.mjs", import.meta.url), { workerData: compileQueue }).unref();
  jit.useCompileQueue(compileQueue);
}

function upload(data) {
  const ptr = wasm.uw_alloc(data.byteLength);
  bytes(ptr, data.byteLength).set(data);
  return ptr;
}
const kernel = values.kernel ? readFileSync(values.kernel) : new Uint8Array(0);
const initrd = values.initrd ? readFileSync(values.initrd) : new Uint8Array(0);
const cmdline = new TextEncoder().encode(values.cmdline);
for (const [name, readOnly] of [...values.disk.map((d) => [d, 0]), ...values.cdrom.map((d) => [d, 1])]) {
  const index = diskSources.push(openDiskSource(name)) - 1;
  if (wasm.uw_add_disk(index, diskSources[index].length, readOnly)) {
    console.error(`${name}:`, decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
    process.exit(2);
  }
}
const firmwareKind = FIRMWARE[values.firmware.toUpperCase()];
if (firmwareKind === undefined) throw new Error(`--firmware ${values.firmware}: none, uefi or bios`);
const firmwareImage = firmwareKind === FIRMWARE.UEFI
  ? readFileSync(new URL("uemulator-firmware.bin", `file://${values.wasm}`))
  : new Uint8Array(0);
if (wasm.uw_set_firmware(firmwareKind, upload(firmwareImage), firmwareImage.length)) {
  console.error("firmware:", decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
  process.exit(2);
}
if (values.hypervisor) {
  const hypervisor = readFileSync(values.hypervisor);
  const hypervisorCmdline = new TextEncoder().encode(values["hypervisor-cmdline"]);
  wasm.uw_set_hypervisor(upload(hypervisor), hypervisor.length, upload(hypervisorCmdline), hypervisorCmdline.length);
}
if (values.gpu) {
  const limits = new Uint8Array(new Int32Array(MINIMUM_LIMITS).buffer);
  if (limits.length !== LIMIT_PARAMETERS.length * 4) throw new Error("gpu.js: one limit per parameter");
  const name = new TextEncoder().encode("uemulator boot.mjs");
  if (wasm.uw_set_gpu(upload(limits), limits.length, upload(name), name.length)) {
    console.error("gpu:", decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
    process.exit(2);
  }
}
const options =
  (values.gpu ? OPTION.DISPLAY : 0) | (values["no-jit"] ? 0 : OPTION.JIT) | (values["no-acpi"] ? 0 : OPTION.ACPI) |
  (values.iommu ? OPTION.IOMMU : 0) | (values.serial ? OPTION.SERIAL_CONSOLE : 0) | OPTION.RNG |
  (values.debug ? OPTION.DEBUG : 0) | (values.tpm ? OPTION.TPM : 0) | (values["secure-boot"] ? OPTION.SECURE_BOOT : 0);
const failed = wasm.uw_create(
  upload(kernel), kernel.length, upload(initrd), initrd.length, upload(cmdline), cmdline.length,
  Number(values.memory), Number(values.cpus), Math.floor(Date.now() / 1000), 80, 25, options,
);
if (failed) {
  console.error("create failed:", decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
  process.exit(2);
}
const runStart = performance.now();
const code = wasm.uw_run();
const seconds = ((performance.now() - runStart) / 1000).toFixed(2);
console.error(
  `\nexit=${code} found=${found} run=${seconds}s jit=${values["no-jit"] ? "off" : "on"} ` +
    `modules=${compiled} compile=${compileMs.toFixed(0)}ms rss=${(process.memoryUsage().rss / 2 ** 20).toFixed(0)}MiB` +
    (diskSources.length ? ` disk-reads=${diskReads} disk-bytes=${diskBytes}` : ""),
);
if (values.gpu) {
  console.error(`gpu: ${gpuFences} fences completed, ${gpuRenderer.clears} clears; ` +
    [...gpuRenderer.counts].map(([name, count]) => `${name} ${count}`).join(", "));
}
if (code === 4) console.error(decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
