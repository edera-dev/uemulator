import { CONTROL_BYTES, CTRL, EXIT_NAMES, RING_BYTES, RING_OFFSET } from "./protocol.js";

const $ = (id) => document.getElementById(id);
const term = window.term = new Terminal({ convertEol: false, cursorBlink: true, fontSize: 14, scrollback: 5000 });
const fit = new FitAddon.FitAddon();
term.loadAddon(fit);
term.open($("terminal"));
fit.fit();

const DEFAULT_CMDLINE_EXTRA = "earlycon=uart8250,io,0x3f8";
$("cmdline").value = new URLSearchParams(location.search).get("args") ?? DEFAULT_CMDLINE_EXTRA;

let worker = null;
let control = null;
let ring = null;
let startedAt = 0;
let clock = null;

function setStatus(text, kind = "") {
  const status = $("status");
  status.textContent = text;
  status.dataset.kind = kind;
}

function setRunning(running) {
  $("start").disabled = running;
  $("stop").disabled = !running;
  for (const id of ["kernel-file", "kernel-url", "initrd-file", "initrd-url", "disk-file", "disk-url", "memory", "cmdline", "shell", "jit"]) {
    $(id).disabled = running;
  }
}

async function load(fileInput, urlInput, label) {
  const file = fileInput.files[0];
  if (file) return file.arrayBuffer();
  const url = urlInput.value.trim();
  if (!url) return null;
  setStatus(`downloading ${label}…`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label}: ${url} returned ${response.status}`);
  return response.arrayBuffer();
}

function sendKeys(text) {
  if (!control) return;
  const bytes = new TextEncoder().encode(text);
  let head = Atomics.load(control, CTRL.HEAD);
  const tail = Atomics.load(control, CTRL.TAIL);
  for (const byte of bytes) {
    const next = (head + 1) % RING_BYTES;
    if (next === tail) break; // ring full: drop rather than block the page
    ring[head] = byte;
    head = next;
  }
  Atomics.store(control, CTRL.HEAD, head);
  Atomics.notify(control, CTRL.HEAD);
}

function sendSize() {
  if (!control) return;
  Atomics.store(control, CTRL.COLS, term.cols);
  Atomics.store(control, CTRL.ROWS, term.rows);
  Atomics.add(control, CTRL.RESIZE_GENERATION, 1);
  Atomics.notify(control, CTRL.HEAD);
}

term.onData(sendKeys);
term.onResize(sendSize);
window.addEventListener("resize", () => fit.fit());

function finish(text, kind) {
  clearInterval(clock);
  worker?.terminate();
  worker = null;
  control = null;
  ring = null;
  setRunning(false);
  setStatus(text, kind);
}

async function start() {
  if (!crossOriginIsolated) {
    setStatus("page is not cross-origin isolated; serve it with hack/web.sh", "error");
    return;
  }
  setRunning(true);
  term.reset();
  fit.fit();
  try {
    const kernel = await load($("kernel-file"), $("kernel-url"), "kernel");
    if (!kernel) throw new Error("choose a kernel (bzImage) first");
    const initrd = await load($("initrd-file"), $("initrd-url"), "initramfs");

    const shared = new SharedArrayBuffer(CONTROL_BYTES);
    control = new Int32Array(shared, 0, RING_OFFSET / 4);
    ring = new Uint8Array(shared, RING_OFFSET, RING_BYTES);
    control[CTRL.COLS] = term.cols;
    control[CTRL.ROWS] = term.rows;

    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }) => {
      if (data.type === "output") {
        term.write(new Uint8Array(data.bytes));
      } else if (data.type === "log") {
        console.log("[uemulator]", data.text);
      } else if (data.type === "started") {
        startedAt = performance.now();
        clock = setInterval(() => {
          setStatus(`running, ${((performance.now() - startedAt) / 1000).toFixed(0)} s`, "running");
        }, 500);
        term.focus();
      } else if (data.type === "exit") {
        const name = EXIT_NAMES[data.code] ?? `error ${data.code}`;
        const detail = data.message ? `: ${data.message}` : "";
        finish(`${name}${detail}`, data.code === 0 || data.code === 3 ? "" : "error");
      }
    };
    worker.onerror = (event) => finish(`worker failed: ${event.message}`, "error");

    const extra = [$("cmdline").value.trim(), $("shell").checked ? "rdinit=/bin/sh" : ""].join(" ").trim();
    // The worker reads the disk on demand (a File slice or an HTTP range request per miss), so
    // neither is downloaded here. A relative URL is resolved against the page, not the worker.
    const diskFile = $("disk-file").files[0];
    const diskUrl = $("disk-url").value.trim();
    const disks = diskFile ? [{ file: diskFile }] : diskUrl ? [{ url: new URL(diskUrl, location.href).href }] : [];
    const message = {
      kernel,
      initrd,
      cmdline: `console=hvc0 earlyprintk=serial,ttyS0 nokaslr mitigations=off acpi=off panic=-1 ${extra}`.trim(),
      memoryMib: Number($("memory").value),
      disks,
      jit: $("jit").checked,
      cols: term.cols,
      rows: term.rows,
      control: shared,
    };
    setStatus("starting…");
    worker.postMessage(message, [kernel, ...(initrd ? [initrd] : [])]);
  } catch (error) {
    finish(String(error.message ?? error), "error");
  }
}

$("start").onclick = start;
$("stop").onclick = () => {
  if (control) Atomics.store(control, CTRL.STOP, 1);
};

// `hack/web.sh` links the cached Alpine netboot files (and WEB_DISK) under assets/; prefill them
// when present.
for (const [id, url] of [["kernel-url", "assets/vmlinuz-virt"], ["initrd-url", "assets/initramfs-virt"], ["disk-url", "assets/disk"]]) {
  fetch(url, { method: "HEAD" })
    .then((response) => { if (response.ok) $(id).value = url; })
    .catch(() => {});
}

if (!crossOriginIsolated) {
  setStatus("page is not cross-origin isolated; serve it with hack/web.sh", "error");
} else if (new URLSearchParams(location.search).has("autoboot")) {
  // Wait for the HEAD probes above so the cached kernel is filled in.
  setTimeout(start, 500);
}
