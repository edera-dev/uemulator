// Boots a Linux kernel on the browser build (target/web/uemulator_web.wasm) under Node: the same
// module and the same host imports as www/worker.js, with the console on stdout. For measuring
// and for checking the WebAssembly JIT against a real guest without a browser.
//
//   node hack/wasm/boot.mjs [--kernel K] [--initrd I] [--disk D] [--cmdline S] [--memory MiB]
//                           [--firmware none|uefi|bios] [--cpus N] [--iommu] [--no-acpi] [--serial]
//                           [--no-jit] [--until TEXT] [--seconds N]
//
// --disk is a raw or udisk image (a file, or an http(s) URL served with range requests) attached
// as vda the way the page attaches one: read on demand, writes kept in memory. --cmdline is
// appended to the library's default command line, as on the page. --firmware uefi reads
// target/web/uemulator-firmware.bin (hack/web.sh builds it) and boots the kernel through it, or the
// disk when there is no kernel.
//
// Stops when TEXT shows up on the console or after N seconds, and prints the wall time and what the
// JIT host did to stderr.
import { closeSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import { createJitHost } from "../../crates/uemulator-web/www/jit_host.js";
import { FIRMWARE, OPTION } from "../../crates/uemulator-web/www/protocol.js";

const { values } = parseArgs({
  options: {
    kernel: { type: "string" },
    initrd: { type: "string" },
    disk: { type: "string" },
    cmdline: { type: "string", default: "earlycon=uart8250,io,0x3f8" },
    memory: { type: "string", default: "256" },
    cpus: { type: "string", default: "1" },
    firmware: { type: "string", default: "none" },
    iommu: { type: "boolean", default: false },
    "no-acpi": { type: "boolean", default: false },
    serial: { type: "boolean", default: false },
    wasm: { type: "string", default: new URL("../../target/web/uemulator_web.wasm", import.meta.url).pathname },
    "no-jit": { type: "boolean", default: false },
    until: { type: "string" },
    seconds: { type: "string", default: "120" },
    quiet: { type: "boolean", default: false },
  },
});
if (!values.kernel && !values.disk) throw new Error("--kernel or --disk is required");

const start = performance.now();
const deadline = start + Number(values.seconds) * 1000;
let wasm;
let seen = "";
let found = false;
let compileMs = 0;
let compiled = 0;
const decoder = new TextDecoder();
const jit = createJitHost();
const bytes = (ptr, len) => new Uint8Array(wasm.memory.buffer, ptr, len);
const sleeper = new Int32Array(new SharedArrayBuffer(4));

const host = {
  ...jit.host,
  host_jit_compile(ptr, len) {
    const before = performance.now();
    const slot = jit.host.host_jit_compile(ptr, len);
    compileMs += performance.now() - before;
    compiled++;
    if (slot < 0) console.error("jit: module refused:", jit.host.lastError);
    return slot;
  },
  host_output(ptr, len) {
    const text = decoder.decode(bytes(ptr, len));
    if (!values.quiet) process.stdout.write(text);
    if (values.until) {
      seen = (seen + text).slice(-4096);
      if (seen.includes(values.until)) found = true;
    }
  },
  host_input: () => 0,
  host_resize: () => 0,
  host_should_stop: () => (found || performance.now() > deadline ? 1 : 0),
  host_idle: () => Atomics.wait(sleeper, 0, 0, 1),
  host_random(ptr, len) {
    crypto.getRandomValues(bytes(ptr, len));
  },
  host_now_ms: () => performance.now(),
  host_log: (ptr, len) => console.error("log:", decoder.decode(bytes(ptr, len))),
  host_disk_read(source, offset, ptr, len) {
    try {
      const data = diskSource.read(offset, len);
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
let diskSource = null;
let diskReads = 0;
let diskBytes = 0;

const { instance } = await WebAssembly.instantiate(readFileSync(values.wasm), { host });
wasm = instance.exports;
jit.attach(wasm);

function upload(data) {
  const ptr = wasm.uw_alloc(data.byteLength);
  bytes(ptr, data.byteLength).set(data);
  return ptr;
}
const kernel = values.kernel ? readFileSync(values.kernel) : new Uint8Array(0);
const initrd = values.initrd ? readFileSync(values.initrd) : new Uint8Array(0);
const cmdline = new TextEncoder().encode(values.cmdline);
if (values.disk) {
  diskSource = openDiskSource(values.disk);
  if (wasm.uw_add_disk(0, diskSource.length)) {
    console.error("disk:", decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
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
const options =
  (values["no-jit"] ? 0 : OPTION.JIT) | (values["no-acpi"] ? 0 : OPTION.ACPI) |
  (values.iommu ? OPTION.IOMMU : 0) | (values.serial ? OPTION.SERIAL_CONSOLE : 0) | OPTION.RNG;
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
    `modules=${compiled} compile=${compileMs.toFixed(0)}ms` +
    (diskSource ? ` disk-reads=${diskReads} disk-bytes=${diskBytes}` : ""),
);
if (code === 4) console.error(decoder.decode(bytes(wasm.uw_error_ptr(), wasm.uw_error_len())));
