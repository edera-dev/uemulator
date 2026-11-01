import {
  CONTROL_BYTES, CTRL, CURSOR, EVENT, EVENT_OFFSET, EVENT_RECORDS, EVENT_WORDS, EXIT_NAMES, FIRMWARE, OPTION,
  RING_BYTES, RING_OFFSET, UEFI_FIRMWARE_MAGIC,
} from "./protocol.js";

const $ = (id) => document.getElementById(id);
const term = window.term = new Terminal({ convertEol: false, cursorBlink: true, fontSize: 14, scrollback: 5000 });
const fit = new FitAddon.FitAddon();
term.loadAddon(fit);
term.open($("terminal"));
fit.fit();

// The library's DEFAULT_CMDLINE (crates/uemulator/src/platform.rs) as EmulatedMachine adjusts it:
// `acpi=off` goes when ACPI is on, the console word follows the console kind, and a screen that
// is the console adds `console=tty0` last. Only for showing what the guest will get; the library
// builds the real one and the page appends to it.
function defaultCmdline({ serial, acpi, display }) {
  return [`console=${serial ? "ttyS0" : "hvc0"}`, "earlyprintk=serial,ttyS0", acpi ? "" : "acpi=off", "panic=-1",
    display === "screen" ? "console=tty0" : ""]
    .filter(Boolean).join(" ");
}
const DEFAULT_CMDLINE_EXTRA = "earlycon=uart8250,io,0x3f8";
const BUILT_IN_FIRMWARE_URL = "uemulator-firmware.bin";
const SETTINGS_KEY = "uemulator.settings.v1";
// Present only when the site was built with WEB_DEMO=1 (hack/web/demo.py), as the Pages build is;
// its file names are relative to it.
const DEMO_URL = new URL("demo/demo.json", location.href);
// isolate.js normally fixes this with a reload before anyone sees it.
const NOT_ISOLATED = "page is not cross-origin isolated: serve it with hack/web.sh, or allow service workers";
const FIRMWARE_LABELS = { uefi: "UEFI", none: "Direct kernel", bios: "BIOS", custom: "Firmware image" };

const SOURCES = ["firmware", "kernel", "initrd", "disk"];
// Everything that is remembered across visits (files cannot be) and locked while a machine runs.
const FIELDS = ["firmware-url", "kernel-url", "initrd-url", "disk-url", "cmdline", "shell", "replace-cmdline", "memory", "cpus", "jit", "acpi", "iommu", "rng", "reboot", "input"];
const RADIOS = ["firmware", "console", "display"];
const DISPLAY_LABELS = { none: "terminal only", screen: "screen (the console)", both: "screen and terminal" };

let worker = null;
// Set from Launch until the worker exists (downloads happen in between).
let starting = false;
let control = null;
let ring = null;
let events = null;
let startedAt = 0;
let clock = null;
let step = 0;
// "Boot again" only makes sense once something has booted.
let hasRun = false;
// demo.json when the site has one, and what Boot again repeats.
let demo = null;
let lastSettings = null;
const steps = [...document.querySelectorAll(".wizard .step")];

function setStatus(text, kind = "") {
  const status = $("status");
  status.textContent = text;
  status.dataset.kind = kind;
}

const radio = (name) => document.querySelector(`input[name=${name}]:checked`).value;

function sourceOf(name) {
  const file = $(`${name}-file`).files[0];
  if (file) return { file, label: file.name };
  const url = $(`${name}-url`).value.trim();
  if (url) return { url, label: decodeURIComponent(url.split(/[?#]/)[0].split("/").filter(Boolean).pop() ?? url) };
  return null;
}

function readSettings() {
  const firmware = radio("firmware");
  // BIOS and a real-mode image replace the kernel; what the BIOS path cannot use is not offered.
  const usesKernel = firmware !== "bios";
  const acpi = $("acpi").checked;
  const extra = [$("cmdline").value.trim(), $("shell").checked ? "rdinit=/bin/sh" : ""].filter(Boolean).join(" ");
  const settings = {
    firmware,
    usesKernel,
    firmwareImage: firmware === "custom" ? sourceOf("firmware") : null,
    kernel: usesKernel ? sourceOf("kernel") : null,
    initrd: usesKernel ? sourceOf("initrd") : null,
    disk: sourceOf("disk"),
    extra,
    replaceCmdline: $("replace-cmdline").checked,
    serial: radio("console") === "serial",
    display: radio("display"),
    input: $("input").checked,
    memoryMib: Number($("memory").value),
    cpus: Number($("cpus").value),
    jit: $("jit").checked,
    acpi,
    iommu: $("iommu").checked,
    rng: $("rng").checked,
    reboot: $("reboot").checked,
  };
  settings.cmdline = settings.replaceCmdline ? extra : [defaultCmdline(settings), extra].filter(Boolean).join(" ");
  return settings;
}

// The combinations EmulatedMachine::new refuses, caught before anything is downloaded. Each names
// the step that fixes it. The library still has the last word (an EFI stub, a firmware header).
function problemsWith(s) {
  const problems = [];
  if (s.firmware === "none" && !s.kernel) problems.push({ step: 1, text: "Direct kernel boot needs a kernel." });
  if (s.firmware === "uefi" && !s.kernel && !s.disk) problems.push({ step: 1, text: "UEFI needs a kernel (with an EFI stub) or a disk with an EFI boot loader." });
  if (s.firmware === "bios" && !s.disk) problems.push({ step: 1, text: "BIOS boot needs a disk." });
  if (s.firmware === "custom" && !s.firmwareImage) problems.push({ step: 0, text: "Choose a firmware image." });
  if (s.replaceCmdline && s.kernel && !s.extra) problems.push({ step: 1, text: "Replacing the default command line with nothing leaves the guest without a console." });
  return problems;
}

function summaryOf(s) {
  const media = s.title ?? [s.kernel?.label, s.initrd?.label, s.disk && `disk ${s.disk.label}`].filter(Boolean).join(" + ");
  return [
    s.title,
    FIRMWARE_LABELS[s.firmware] + (s.firmwareImage ? ` (${s.firmwareImage.label})` : ""),
    s.title ? null : media || "no media",
    `${s.memoryMib} MiB`,
    `${s.cpus} vCPU${s.cpus > 1 ? "s" : ""}`,
    s.jit ? "JIT" : "interpreter",
    s.iommu && "IOMMU",
    s.serial ? "ttyS0" : "hvc0",
    s.display !== "none" && (s.display === "screen" ? "screen console" : "screen"),
    s.input && "keyboard+mouse",
  ].filter(Boolean).join(" · ");
}

function render() {
  const firmware = radio("firmware");
  $("custom-firmware").hidden = firmware !== "custom";

  // More than one vCPU needs the MADT (ACPI) and cannot run under the BIOS shim; the IOMMU is found
  // through the IVRS table. Disabled controls are reset so they cannot linger in a hidden state.
  const smpAllowed = $("acpi").checked && firmware !== "bios";
  $("cpus").disabled = !smpAllowed;
  if (!smpAllowed) $("cpus").value = "1";
  $("iommu").disabled = !$("acpi").checked;
  if ($("iommu").disabled) $("iommu").checked = false;

  const s = readSettings();
  $("kernel-group").hidden = $("initrd-group").hidden = !s.usesKernel;
  // A disk's boot loader brings its own command line; the field only matters with a kernel.
  $("cmdline-group").hidden = !s.usesKernel || (!s.kernel && firmware !== "none");
  $("disk-optional").hidden = firmware === "bios";
  $("kernel-hint").textContent = {
    // U+2011: keeps "-kernel" from breaking after the hyphen.
    uefi: "An EFI-stub bzImage (most distribution kernels). The firmware boots it before any disk, like OVMF with QEMU's \u2011kernel.",
    none: "Any x86_64 bzImage.",
    custom: "Used when the image is UEFI firmware; a real-mode image ignores it.",
  }[firmware] ?? "";
  $("media-lead").textContent = {
    uefi: "A kernel, a disk, or both. With a kernel the firmware boots it and the disk is just attached; without one it boots the disk's EFI system partition.",
    none: "A kernel and, optionally, an initramfs and a disk.",
    bios: "The disk's first sector is loaded at 0000:7C00 and its boot loader takes it from there.",
    custom: "Whatever the image boots from.",
  }[firmware];
  $("cmdline-preview").textContent = s.cmdline || "(empty)";

  const problems = problemsWith(s);
  const nav = $("steps");
  nav.replaceChildren(...steps.map((section, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span class="num">${index + 1}</span>`;
    button.append(section.dataset.title);
    if (index === step) button.setAttribute("aria-current", "step");
    if (problems.some((problem) => problem.step === index) && index < steps.length - 1) button.dataset.problem = "";
    button.onclick = () => go(index);
    return button;
  }));
  steps.forEach((section, index) => { section.hidden = index !== step; });
  for (const name of SOURCES) $(`${name}-url`).closest(".source").querySelector(".clear").disabled = starting || worker !== null || (!$(`${name}-file`).files.length && !$(`${name}-url`).value);
  $("back").disabled = step === 0;
  $("next").hidden = step === steps.length - 1;
  $("launch").disabled = problems.length > 0 || starting || worker !== null;

  const rows = [
    ["Firmware", FIRMWARE_LABELS[s.firmware] + (s.firmwareImage ? `: ${s.firmwareImage.label}` : "")],
    ["Kernel", s.usesKernel ? s.kernel?.label ?? "none" : "not used"],
    ["Initramfs", s.usesKernel ? s.initrd?.label ?? "none" : "not used"],
    ["Disk", s.disk ? `${s.disk.label} (vda, writes kept in memory)` : "none"],
    ["Memory", `${s.memoryMib} MiB`],
    ["vCPUs", String(s.cpus)],
    ["Console", s.serial ? "serial (ttyS0)" : "virtio-console (hvc0)"],
    ["Display", DISPLAY_LABELS[s.display] + (s.input ? ", virtio keyboard and tablet" : "")],
    ["Execution", s.jit ? "WebAssembly JIT" : "interpreter only"],
    ["Devices", [s.acpi ? "ACPI" : "no ACPI", s.iommu && "AMD IOMMU", s.rng && "virtio-rng"].filter(Boolean).join(", ")],
    // The library keeps no image to restart a BIOS boot from: a reset always ends it.
    ["On guest reset", s.reboot && s.firmware !== "bios" ? "restart" : "stop"],
  ];
  if (s.kernel) rows.push(["Command line", s.cmdline]);
  $("review").replaceChildren(...rows.flatMap(([term, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    if (term === "Command line") dd.className = "mono";
    return [dt, dd];
  }));
  $("problems").hidden = problems.length === 0;
  $("problems").replaceChildren(...problems.map(({ text }) => Object.assign(document.createElement("li"), { textContent: text })));
  // The bar describes what Launch would boot while the wizard is open, else what last ran.
  if (!$("wizard").hidden) $("summary").textContent = summaryOf(s);
  saveSettings();
}

// The same shape as readSettings, for demo.json's machine; the wizard's fields are left alone.
function demoSettings(d) {
  const at = (name) => ({ url: new URL(name, DEMO_URL).href, label: name });
  const s = {
    title: d.name,
    firmware: d.firmware,
    usesKernel: true,
    firmwareImage: null,
    kernel: at(d.kernel),
    initrd: at(d.initrd),
    disk: null,
    extra: d.args ?? "",
    replaceCmdline: false,
    serial: d.console === "serial",
    display: d.display ?? "none",
    input: Boolean(d.input),
    memoryMib: d.memoryMib,
    cpus: 1,
    jit: true,
    acpi: true,
    iommu: false,
    rng: true,
    reboot: true,
  };
  s.cmdline = [defaultCmdline(s), s.extra].filter(Boolean).join(" ");
  return s;
}

function go(index) {
  step = Math.max(0, Math.min(steps.length - 1, index));
  render();
  steps[step].scrollTop = 0;
  steps[step].querySelector("input:not([type=radio]):not([disabled]), select:not([disabled]), input[type=radio]:checked")?.focus({ preventScroll: true });
}

// What sits over the terminal: "demo", "wizard" or nothing.
function showPanel(name) {
  $("overlay").hidden = !name;
  $("demo").hidden = name !== "demo";
  $("wizard").hidden = name !== "wizard";
  if (name === "wizard") render();
  if (!name) focusView();
}

// Closing the wizard goes back to the demo until something has run; after that, to the terminal,
// with the bar describing what ran again (Boot again repeats it, not the wizard's edits).
function closeWizard() {
  showPanel(demo && !hasRun ? "demo" : null);
  const shown = lastSettings ?? (demo && demoSettings(demo));
  if (shown) $("summary").textContent = summaryOf(shown);
}

// Per-visitor convenience only: files cannot be remembered, and storage may be unavailable.
function saveSettings() {
  const saved = Object.fromEntries(FIELDS.map((id) => [id, $(id).type === "checkbox" ? $(id).checked : $(id).value]));
  for (const name of RADIOS) saved[name] = radio(name);
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(saved)); } catch { /* private mode */ }
}

// Returns what was saved, so the asset prefill below leaves alone a field the visitor cleared.
function restoreSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") ?? {}; } catch { /* nothing saved */ }
  for (const id of FIELDS) {
    if (!(id in saved)) continue;
    if ($(id).type === "checkbox") $(id).checked = Boolean(saved[id]);
    else $(id).value = saved[id];
  }
  const params = new URLSearchParams(location.search);
  for (const name of RADIOS) {
    const value = params.get(name) ?? saved[name];
    const input = value && document.querySelector(`input[name=${name}][value="${CSS.escape(value)}"]`);
    if (input) input.checked = true;
  }
  if (!("cmdline" in saved)) $("cmdline").value = DEFAULT_CMDLINE_EXTRA;
  if (params.has("args")) $("cmdline").value = params.get("args");
  if (params.has("iommu")) $("iommu").checked = $("acpi").checked = true;
  return saved;
}

function setRunning(running) {
  for (const id of [...FIELDS, ...SOURCES.map((name) => `${name}-file`)]) $(id).disabled = running;
  for (const input of document.querySelectorAll("input[type=radio]")) input.disabled = running;
  $("stop").hidden = !running;
  $("configure").disabled = running;
  $("relaunch").hidden = running || !hasRun;
  if (!running) render();
}

async function load(source, label) {
  if (!source) return null;
  if (source.file) return source.file.arrayBuffer();
  setStatus(`downloading ${label}…`);
  const response = await fetch(source.url);
  if (!response.ok) throw new Error(`${label}: ${source.url} returned ${response.status}`);
  return response.arrayBuffer();
}

// The kind is read from the image: crates/uemulator-firmware starts with its magic, and anything
// else is taken for a real-mode image.
async function loadFirmware(s) {
  if (s.firmware === "none") return { kind: FIRMWARE.NONE, image: null };
  if (s.firmware === "bios") return { kind: FIRMWARE.BIOS, image: null };
  const image = s.firmware === "uefi"
    ? await load({ url: BUILT_IN_FIRMWARE_URL }, "UEFI firmware")
    : await load(s.firmwareImage, "firmware image");
  const magic = new TextDecoder().decode(new Uint8Array(image, 0, Math.min(8, image.byteLength)));
  if (magic === UEFI_FIRMWARE_MAGIC) return { kind: FIRMWARE.UEFI, image };
  if (s.firmware === "uefi") throw new Error(`${BUILT_IN_FIRMWARE_URL} is not uemulator's UEFI firmware; rebuild with hack/web.sh`);
  return { kind: FIRMWARE.REAL_MODE, image };
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
window.addEventListener("resize", () => {
  if (view === "terminal") fit.fit();
  else layoutScreen();
  requestGuestSize();
});

// The screen: frames from the worker go onto the canvas, the canvas's keys and pointer go back
// through the event ring. "terminal" or "screen" is in front.
let view = "terminal";
const canvas = $("screen");
const screenContext = canvas.getContext("2d", { alpha: false });
// KeyboardEvent.code to USB HID keyboard usage (UI Events code values are named after the USB
// usage they come from); the same table as the CLI's viewer (crates/uemulator-cli/src/frontend/web/viewer.html).
const USAGES = (() => {
  const map = {};
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").forEach((c, i) => map["Key" + c] = 0x04 + i);
  "123456789".split("").forEach((c, i) => map["Digit" + c] = 0x1e + i);
  map.Digit0 = 0x27;
  for (let i = 1; i <= 12; i++) map["F" + i] = 0x3a + i - 1;
  for (let i = 13; i <= 24; i++) map["F" + i] = 0x68 + i - 13;
  "123456789".split("").forEach((c, i) => map["Numpad" + c] = 0x59 + i);
  Object.assign(map, {
    Enter: 0x28, Escape: 0x29, Backspace: 0x2a, Tab: 0x2b, Space: 0x2c, Minus: 0x2d,
    Equal: 0x2e, BracketLeft: 0x2f, BracketRight: 0x30, Backslash: 0x31, Semicolon: 0x33,
    Quote: 0x34, Backquote: 0x35, Comma: 0x36, Period: 0x37, Slash: 0x38, CapsLock: 0x39,
    PrintScreen: 0x46, ScrollLock: 0x47, Pause: 0x48, Insert: 0x49, Home: 0x4a, PageUp: 0x4b,
    Delete: 0x4c, End: 0x4d, PageDown: 0x4e, ArrowRight: 0x4f, ArrowLeft: 0x50,
    ArrowDown: 0x51, ArrowUp: 0x52, NumLock: 0x53, NumpadDivide: 0x54, NumpadMultiply: 0x55,
    NumpadSubtract: 0x56, NumpadAdd: 0x57, NumpadEnter: 0x58, Numpad0: 0x62,
    NumpadDecimal: 0x63, IntlBackslash: 0x64, ContextMenu: 0x65, Power: 0x66,
    NumpadEqual: 0x67, AudioVolumeMute: 0x7f, AudioVolumeUp: 0x80, AudioVolumeDown: 0x81,
    NumpadComma: 0x85, IntlRo: 0x87, KanaMode: 0x88, IntlYen: 0x89, Convert: 0x8a,
    NonConvert: 0x8b, Lang1: 0x90, Lang2: 0x91, ControlLeft: 0xe0, ShiftLeft: 0xe1,
    AltLeft: 0xe2, MetaLeft: 0xe3, ControlRight: 0xe4, ShiftRight: 0xe5, AltRight: 0xe6,
    MetaRight: 0xe7,
  });
  return map;
})();
// For keyboards that leave `code` empty (on-screen keyboards, synthetic events): the US-layout
// key that types the character and whether it needs shift, [usage, shifted].
const CHARACTERS = (() => {
  const map = {};
  "abcdefghijklmnopqrstuvwxyz".split("").forEach((c, i) => {
    map[c] = [0x04 + i, false];
    map[c.toUpperCase()] = [0x04 + i, true];
  });
  "1234567890".split("").forEach((c, i) => map[c] = [0x1e + i, false]);
  "!@#$%^&*()".split("").forEach((c, i) => map[c] = [0x1e + i, true]);
  const pairs = ["-_", "=+", "[{", "]}", "\\|", null, ";:", "'\"", "`~", ",<", ".>", "/?"];
  pairs.forEach((pair, i) => {
    if (pair) {
      map[pair[0]] = [0x2d + i, false];
      map[pair[1]] = [0x2d + i, true];
    }
  });
  map[" "] = [0x2c, false];
  return map;
})();
const LEFT_SHIFT = 0xe1;
// MouseEvent.button to the guest's button index (EVENT.POINTER_BUTTON).
const BUTTONS = { 0: 0, 1: 2, 2: 1, 3: 3, 4: 4 };
const GUEST_SIZE_DEBOUNCE_MS = 300;
const DEFAULT_SCREEN = [1280, 800];

// Drops the event when the ring is full rather than blocking the page.
function pushEvent(kind, first = 0, second = 0) {
  if (!control) return;
  const head = Atomics.load(control, CTRL.EVENT_HEAD);
  const next = (head + 1) % EVENT_RECORDS;
  if (next === Atomics.load(control, CTRL.EVENT_TAIL)) return;
  events.set([kind, first, second], head * EVENT_WORDS);
  Atomics.store(control, CTRL.EVENT_HEAD, next);
  // The worker's idle wait sleeps on HEAD.
  Atomics.notify(control, CTRL.HEAD);
}

function showView(name) {
  if (name === view) return focusView();
  // Keys held down while switching would never see their release.
  pushEvent(EVENT.RELEASE_ALL);
  view = name;
  $("terminal").hidden = name !== "terminal";
  $("screen-view").hidden = name !== "screen";
  $("view-terminal").setAttribute("aria-pressed", String(name === "terminal"));
  $("view-screen").setAttribute("aria-pressed", String(name === "screen"));
  focusView();
}

// Fitting a hidden terminal would measure zero columns and send that to the guest, so it is only
// refitted while it is in front.
function focusView() {
  if (!$("overlay").hidden) return;
  if (view === "terminal") {
    fit.fit();
    term.focus();
  } else {
    layoutScreen();
    canvas.focus();
  }
}

// Shrinks the canvas to fit, or enlarges it by whole pixels (crisp when pixelated).
function layoutScreen() {
  const box = $("screen-view");
  const fitScale = Math.min(box.clientWidth / canvas.width, box.clientHeight / canvas.height);
  if (!(fitScale > 0)) return;
  const scale = fitScale >= 1 ? Math.floor(fitScale) : fitScale;
  canvas.style.width = `${Math.floor(canvas.width * scale)}px`;
  canvas.style.height = `${Math.floor(canvas.height * scale)}px`;
  canvas.toggleAttribute("data-pixelated", scale >= 1);
}

function resetScreen(s) {
  [canvas.width, canvas.height] = DEFAULT_SCREEN;
  screenContext.fillStyle = "#000";
  screenContext.fillRect(0, 0, canvas.width, canvas.height);
  canvas.style.cursor = "";
  const note = s.display === "none" ? "No screen: keys and the pointer go to the guest's virtio keyboard and tablet." : "";
  $("screen-note").textContent = note;
  $("screen-note").hidden = !note;
  const hasScreen = s.display !== "none" || s.input;
  $("views").hidden = !hasScreen;
  $("fit-guest-label").hidden = s.display === "none";
  view = "";
  showView(hasScreen && s.display === "screen" ? "screen" : "terminal");
}

function drawFrame(frame) {
  if (canvas.width !== frame.screenWidth || canvas.height !== frame.screenHeight) {
    canvas.width = frame.screenWidth;
    canvas.height = frame.screenHeight;
    layoutScreen();
  }
  if (frame.width > 0 && frame.height > 0) {
    const image = new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height);
    screenContext.putImageData(image, frame.x, frame.y);
  }
  if (control) Atomics.store(control, CTRL.FRAME_ACK, 1);
}

// The guest's pointer image becomes the canvas's CSS cursor: the tablet puts the guest's pointer
// under the host's, so drawing it into the frames would show two.
function setCursor(cursor) {
  if (cursor.kind === CURSOR.HIDDEN) {
    canvas.style.cursor = "none";
  } else if (cursor.kind === CURSOR.IMAGE && cursor.width > 0 && cursor.height > 0) {
    const image = document.createElement("canvas");
    image.width = cursor.width;
    image.height = cursor.height;
    image.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(cursor.pixels), cursor.width, cursor.height), 0, 0);
    canvas.style.cursor = `url(${image.toDataURL()}) ${cursor.hotX} ${cursor.hotY}, auto`;
  } else {
    canvas.style.cursor = "";
  }
}

let guestSizeTimer = 0;
function requestGuestSize() {
  if (!$("fit-guest").checked || $("fit-guest-label").hidden) return;
  clearTimeout(guestSizeTimer);
  guestSizeTimer = setTimeout(() => {
    const box = $("screen-view");
    if (box.clientWidth > 0 && box.clientHeight > 0) pushEvent(EVENT.REQUEST_SIZE, box.clientWidth, box.clientHeight);
  }, GUEST_SIZE_DEBOUNCE_MS);
}

function pointerPosition(event) {
  const box = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - box.left) * canvas.width / box.width);
  const y = Math.floor((event.clientY - box.top) * canvas.height / box.height);
  return [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))];
}

// One move per animation frame: only the latest position matters.
let pendingMove = null;
function flushMove() {
  if (pendingMove) pushEvent(EVENT.POINTER_MOVE, ...pendingMove);
  pendingMove = null;
}
canvas.addEventListener("mousemove", (event) => {
  if (!pendingMove) requestAnimationFrame(flushMove);
  pendingMove = pointerPosition(event);
});
canvas.addEventListener("mousedown", (event) => {
  canvas.focus();
  pendingMove = null;
  pushEvent(EVENT.POINTER_MOVE, ...pointerPosition(event));
  if (event.button in BUTTONS) pushEvent(EVENT.POINTER_BUTTON, BUTTONS[event.button], 1);
  event.preventDefault();
});
canvas.addEventListener("mouseup", (event) => {
  if (event.button in BUTTONS) pushEvent(EVENT.POINTER_BUTTON, BUTTONS[event.button], 0);
  event.preventDefault();
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  // One click per event in the scroll's direction: trackpads send many small ones.
  if (event.deltaY !== 0) pushEvent(EVENT.WHEEL, event.deltaY < 0 ? 1 : -1);
  event.preventDefault();
}, { passive: false });

function screenKey(event, pressed) {
  // The guest repeats held keys itself; the browser's repeats would double them.
  const repeat = pressed && event.repeat;
  const usage = USAGES[event.code];
  if (usage !== undefined) {
    if (!repeat) pushEvent(EVENT.KEY, usage, pressed ? 1 : 0);
    event.preventDefault();
    return;
  }
  const typed = CHARACTERS[event.key];
  if (typed === undefined) return;
  event.preventDefault();
  // Typed whole on the press: the release carries nothing to match it with.
  if (!pressed || repeat) return;
  const [character, shifted] = typed;
  const shift = shifted && !event.shiftKey;
  if (shift) pushEvent(EVENT.KEY, LEFT_SHIFT, 1);
  pushEvent(EVENT.KEY, character, 1);
  pushEvent(EVENT.KEY, character, 0);
  if (shift) pushEvent(EVENT.KEY, LEFT_SHIFT, 0);
}
canvas.addEventListener("keydown", (event) => screenKey(event, true));
canvas.addEventListener("keyup", (event) => screenKey(event, false));
// Releases are lost while the focus is elsewhere; let go of everything instead.
canvas.addEventListener("blur", () => pushEvent(EVENT.RELEASE_ALL));
window.addEventListener("blur", () => pushEvent(EVENT.RELEASE_ALL));
$("view-terminal").onclick = () => showView("terminal");
$("view-screen").onclick = () => showView("screen");
$("fit-guest").onchange = requestGuestSize;

function finish(text, kind) {
  starting = false;
  clearInterval(clock);
  worker?.terminate();
  worker = null;
  control = null;
  ring = null;
  events = null;
  // The library's wording names the CLI flag; point at the page's own control instead.
  setStatus(text.replace("boot it with firmware none", "choose Direct kernel under Firmware"), kind);
  setRunning(false);
}

async function launch(s = readSettings()) {
  if (!crossOriginIsolated) {
    setStatus(NOT_ISOLATED, "error");
    return;
  }
  const problems = problemsWith(s);
  if (problems.length) {
    showPanel("wizard");
    go(problems[0].step);
    return;
  }
  showPanel(null);
  hasRun = true;
  lastSettings = s;
  $("summary").textContent = summaryOf(s);
  starting = true;
  setRunning(true);
  term.reset();
  resetScreen(s);
  try {
    const firmware = await loadFirmware(s);
    const realMode = firmware.kind === FIRMWARE.REAL_MODE;
    const kernel = realMode ? null : await load(s.kernel, "kernel");
    const initrd = realMode ? null : await load(s.initrd, "initramfs");

    const shared = new SharedArrayBuffer(CONTROL_BYTES);
    control = new Int32Array(shared, 0, RING_OFFSET / 4);
    ring = new Uint8Array(shared, RING_OFFSET, RING_BYTES);
    events = new Int32Array(shared, EVENT_OFFSET, EVENT_RECORDS * EVENT_WORDS);
    control[CTRL.COLS] = term.cols;
    control[CTRL.ROWS] = term.rows;
    control[CTRL.FRAME_ACK] = 1;

    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    starting = false;
    worker.onmessage = ({ data }) => {
      if (data.type === "output") {
        term.write(new Uint8Array(data.bytes));
      } else if (data.type === "frame") {
        drawFrame(data);
      } else if (data.type === "cursor") {
        setCursor(data);
      } else if (data.type === "log") {
        console.log("[uemulator]", data.text);
      } else if (data.type === "started") {
        startedAt = performance.now();
        clock = setInterval(() => {
          setStatus(`running, ${((performance.now() - startedAt) / 1000).toFixed(0)} s`, "running");
        }, 500);
        focusView();
        requestGuestSize();
      } else if (data.type === "exit") {
        const name = EXIT_NAMES[data.code] ?? `error ${data.code}`;
        const detail = data.message ? `: ${data.message}` : "";
        finish(`${name}${detail}`, data.code === 0 || data.code === 3 ? "" : "error");
      }
    };
    worker.onerror = (event) => finish(`worker failed: ${event.message}`, "error");

    // The worker reads the disk on demand (a File slice or an HTTP range request per miss), so
    // it is not downloaded here. A relative URL is resolved against the page, not the worker.
    const disks = !s.disk ? [] : s.disk.file ? [{ file: s.disk.file }] : [{ url: new URL(s.disk.url, location.href).href }];
    const options = (s.jit ? OPTION.JIT : 0) | (s.acpi ? OPTION.ACPI : 0) | (s.iommu ? OPTION.IOMMU : 0) |
      (s.rng ? OPTION.RNG : 0) | (s.serial ? OPTION.SERIAL_CONSOLE : 0) | (s.reboot ? OPTION.REBOOT : 0) |
      (s.replaceCmdline ? OPTION.REPLACE_CMDLINE : 0) | (s.display !== "none" ? OPTION.DISPLAY : 0) |
      (s.display === "screen" ? OPTION.SCREEN_CONSOLE : 0) | (s.input ? OPTION.INPUT : 0);
    const message = {
      firmware,
      kernel,
      initrd,
      cmdline: s.extra,
      memoryMib: s.memoryMib,
      cpus: realMode ? 1 : s.cpus,
      options,
      disks,
      cols: term.cols,
      rows: term.rows,
      control: shared,
    };
    setStatus("starting…");
    worker.postMessage(message, [kernel, initrd, firmware.image].filter(Boolean));
  } catch (error) {
    finish(String(error.message ?? error), "error");
  }
}

for (const input of document.querySelectorAll(".wizard input, .wizard select")) {
  input.addEventListener(input.type === "text" ? "input" : "change", render);
}
for (const button of document.querySelectorAll("button[data-clear]")) {
  button.onclick = () => {
    $(`${button.dataset.clear}-file`).value = "";
    $(`${button.dataset.clear}-url`).value = "";
    render();
  };
}
$("back").onclick = () => go(step - 1);
$("next").onclick = () => go(step + 1);
$("launch").onclick = () => launch();
$("relaunch").onclick = () => launch(lastSettings ?? readSettings());
$("configure").onclick = () => showPanel("wizard");
$("wizard-close").onclick = closeWizard;
$("wizard").addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeWizard();
});
$("demo-boot").onclick = () => launch(demoSettings(demo));
$("demo-custom").onclick = () => showPanel("wizard");
$("stop").onclick = () => {
  if (control) Atomics.store(control, CTRL.STOP, 1);
};

const saved = restoreSettings();
setRunning(false);
go(0);

// `hack/web.sh` links the cached Alpine netboot files (and WEB_DISK) under assets/; offer them
// when present and the visitor has not chosen (or cleared) that field before. Refilling a cleared
// kernel would quietly turn a UEFI disk boot back into a kernel boot.
const probes = [["kernel-url", "assets/vmlinuz-virt"], ["initrd-url", "assets/initramfs-virt"], ["disk-url", "assets/disk"]]
  .filter(([id]) => !(id in saved))
  .map(([id, url]) => fetch(url, { method: "HEAD" })
    .then((response) => { if (response.ok && !$(id).value) $(id).value = url; })
    .catch(() => {}));
Promise.all(probes).then(render);

// Both panels start hidden so the page does not flash the wizard before the demo replaces it.
const demoReady = fetch(DEMO_URL)
  .then((response) => (response.ok ? response.json() : null))
  .catch(() => null)
  .then((found) => {
    demo = found;
    if (demo) {
      $("demo-boot").textContent = `Boot ${demo.label}`;
      $("demo-boot").title = `${demo.name}, ${Math.round(demo.bytes / 1e6)} MB download`;
      $("summary").textContent = summaryOf(demoSettings(demo));
    }
    if (!hasRun) showPanel(demo ? "demo" : "wizard");
  });

if (!crossOriginIsolated) {
  setStatus(NOT_ISOLATED, "error");
} else if (new URLSearchParams(location.search).has("autoboot")) {
  // In demo mode ?autoboot boots the demo; otherwise the wizard's settings.
  Promise.all([...probes, demoReady]).then(() => launch(demo ? demoSettings(demo) : readSettings()));
}
