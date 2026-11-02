import {
  BYTES_PER_PIXEL, CONTROL_BYTES, CTRL, CURSOR, EVENT, EVENT_OFFSET, EVENT_RECORDS, EVENT_WORDS, EXIT_NAMES, FIRMWARE,
  FRAME_READY, NET_OFFSET, NET_RING_BYTES, OPTION, RECT_WORDS, RING_BYTES, RING_OFFSET, UEFI_FIRMWARE_MAGIC,
} from "./protocol.js";
import { formatMac, joinSegment, randomMac, randomTab } from "./segment.js";
import { compileWorkerHelps, createCompileQueue } from "./jit_host.js";

const $ = (id) => document.getElementById(id);
const term = window.term = new Terminal({ convertEol: false, cursorBlink: true, fontSize: 14, scrollback: 5000 });
const fit = new FitAddon.FitAddon();
term.loadAddon(fit);
term.open($("terminal"));

// The library's DEFAULT_CMDLINE (crates/uemulator/src/platform.rs) as EmulatedMachine adjusts it:
// `acpi=off` goes when ACPI is on, the console word follows the console kind, and a screen that
// is the console adds `console=tty0` last. Only for showing what the guest will get; the library
// builds the real one and the page appends to it.
function defaultCmdline({ serial, acpi, display, hypervisor }) {
  // Under Xen dom0's console is Xen's hvc0 and the 16550 is Xen's (DEFAULT_DOM0_CMDLINE).
  const console = hypervisor ? ["console=hvc0", "earlyprintk=xen"] : [`console=${serial ? "ttyS0" : "hvc0"}`, "earlyprintk=serial,ttyS0"];
  return [...console, acpi ? "" : "acpi=off", "panic=-1", display === "screen" ? "console=tty0" : ""]
    .filter(Boolean).join(" ");
}
// What the library gives Xen when the page leaves its arguments empty (DEFAULT_HYPERVISOR_CMDLINE).
const DEFAULT_HYPERVISOR_CMDLINE = "console=com1 com1=115200,8n1";
const DEFAULT_CMDLINE_EXTRA = "earlycon=uart8250,io,0x3f8";
const BUILT_IN_FIRMWARE_URL = "uemulator-firmware.bin";
// v2: drives became a list and sources are kept as URLs, not form fields. A v1 entry is ignored.
const SETTINGS_KEY = "uemulator.settings.v2";
// Present only when the site was built with WEB_DEMO=1 (hack/web/demo.py), as the Pages build is;
// its file names are relative to it.
const DEMO_URL = new URL("demo/demo.json", location.href);
// isolate.js normally fixes this with a reload before anyone sees it.
const NOT_ISOLATED = "page is not cross-origin isolated: serve it with hack/web.sh, or allow service workers";
const FIRMWARE_HINTS = {
  uefi: "Boots the kernel above if there is one, else the first drive with an EFI boot loader (a hybrid ISO's ESP, a CD's El Torito image, an installed disk).",
  bios: "Runs the first drive's boot sector in real mode: installed disks and isohybrid ISOs.",
  none: "Enters the kernel above directly in 64-bit mode. Drives are only attached.",
  custom: "A UEMUFW64 image (crates/uemulator-firmware) boots as UEFI; anything else is placed below 1 MiB and started at F000:FFF0. OVMF is not supported.",
};
const FIRMWARE_LABELS = { uefi: "UEFI", none: "direct kernel", bios: "BIOS", custom: "custom firmware" };
const RADIOS = ["firmware", "console", "display"];
const CHECKBOXES = ["shell", "replace-cmdline", "jit", "acpi", "iommu", "rng", "reboot", "input", "network"];
const SELECTS = ["memory", "cpus"];
// Where the guest RAM is one allocation of the 32-bit module: Rust refuses one larger than
// isize::MAX, so 2 GiB less a MiB is the most there is.
const MAX_MEMORY_MIB = 2047;
// What live and installer ISOs need: Fedora's installer initramfs alone takes more than 1 GiB
// to unpack, and with less the kernel ends in "VFS: Unable to mount root fs".
const ISO_MEMORY_MIB = MAX_MEMORY_MIB;
// Below this an ISO is a rescue or network-boot image (netboot.xyz is 2 MB) that needs no such
// memory; the warning is for distribution ISOs. An ISO of unknown size gets it too.
const BIG_ISO_BYTES = 256e6;
// More would not fit the PCI slots next to the other devices, and nobody needs it.
const MAX_DRIVES = 8;
const ISO9660_MAGIC_OFFSET = 0x8001;
const ISO9660_MAGIC = "CD001";
// The Linux boot protocol header's magic, at 0x202 of every bzImage.
const BZIMAGE_MAGIC_OFFSET = 0x202;
const BZIMAGE_MAGIC = "HdrS";

let worker = null;
// Compiles the JIT's modules beside `worker` (jit_compiler.js). Started here and not by `worker`,
// whose own workers never start in Chrome while it is busy in `uw_run`.
let compiler = null;
// Set from Start until the worker exists (downloads happen in between).
let starting = false;
let control = null;
let ring = null;
let events = null;
// This tab's place on the shared segment while a machine with a network runs (segment.js).
let segment = null;
let networkMac = null;
let startedAt = 0;
let clock = null;
let hasRun = false;
// demo.json's machines when the site has one; the first is what ?autoboot boots.
let demos = [];
// What the running (or last) machine was started with.
let current = null;

// Picked files cannot be remembered or put in a form field, so sources live here: each is
// { file } or { url }, with a `label` and, once known, a `size`.
const state = {
  drives: [],
  kernel: null,
  initrd: null,
  hypervisor: null,
  firmwareImage: null,
};
let nextDriveId = 1;

function setStatus(text, kind = "") {
  const status = $("status");
  status.firstElementChild.textContent = text;
  status.dataset.kind = kind;
  status.title = text;
}

let toastTimer = 0;
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

const radio = (name) => document.querySelector(`input[name=${name}]:checked`).value;
const setRadio = (name, value) => {
  const input = value != null && document.querySelector(`input[name=${name}][value="${CSS.escape(String(value))}"]`);
  if (input) input.checked = true;
};
// A select only takes values it has an option for; anything else would read back as "".
function setSelect(id, value) {
  const select = $(id);
  if ([...select.options].some((option) => option.value === String(value))) select.value = String(value);
}

function labelOfUrl(url) {
  return decodeURIComponent(url.split(/[?#]/)[0].split("/").filter(Boolean).pop() ?? url);
}

function formatBytes(bytes) {
  if (!(bytes >= 0)) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

async function readText(file, offset, length) {
  const bytes = new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  return String.fromCharCode(...bytes);
}

// ISO 9660 says so at 32 KiB + 1; anything else is attached as a disk unless its name says ISO.
async function looksLikeIso(file) {
  if (/\.iso$/i.test(file.name)) return true;
  try { return (await readText(file, ISO9660_MAGIC_OFFSET, ISO9660_MAGIC.length)) === ISO9660_MAGIC; } catch { return false; }
}

// Xen ships as xen.gz (or xen-4.22.gz and the like) and xen.efi; nothing in a gzip file says what
// is inside, so the name decides.
const looksLikeHypervisor = (file) => /^xen[^/]*\.(gz|efi)$/i.test(file.name);

async function looksLikeKernel(file) {
  try { return (await readText(file, BZIMAGE_MAGIC_OFFSET, BZIMAGE_MAGIC.length)) === BZIMAGE_MAGIC; } catch { return false; }
}

// The size of a URL source, for the drive list; a server without HEAD just leaves it blank.
function probeSize(source) {
  fetch(source.url, { method: "HEAD" })
    .then((response) => {
      const length = Number(response.headers.get("Content-Length"));
      source.size = response.ok && length > 0 ? length : undefined;
      source.unreachable = !response.ok;
      render();
    })
    .catch(() => {});
}

function addDrive(source, kind) {
  if (state.drives.length >= MAX_DRIVES) {
    toast(`At most ${MAX_DRIVES} drives.`);
    return false;
  }
  const drive = { id: nextDriveId++, kind, ...source };
  if (drive.file) drive.size = drive.file.size;
  else probeSize(drive);
  state.drives.push(drive);
  return true;
}

// Files from a picker or a drop: ISOs become CDs, a bzImage the kernel, Xen the hypervisor, the
// rest disks.
async function addFiles(files, { kind = null, detectKernel = false } = {}) {
  const added = [];
  for (const file of files) {
    if (detectKernel && looksLikeHypervisor(file)) {
      state.hypervisor = { file, label: file.name, size: file.size };
      setRadio("firmware", "uefi");
      added.push(`${file.name} as the hypervisor`);
      continue;
    }
    if (detectKernel && await looksLikeKernel(file)) {
      state.kernel = { file, label: file.name, size: file.size };
      added.push(`${file.name} as the kernel`);
      continue;
    }
    const as = kind ?? (await looksLikeIso(file) ? "cdrom" : "disk");
    if (addDrive({ file, label: file.name }, as)) added.push(`${file.name} as a ${as === "cdrom" ? "CD" : "disk"}`);
  }
  if (added.length) toast(`Attached ${added.join(", ")}.`);
  render();
}

function addDriveUrl(url) {
  url = url.trim();
  if (!url) return;
  const kind = /\.iso([?#]|$)/i.test(url) ? "cdrom" : "disk";
  if (addDrive({ url, label: labelOfUrl(url) }, kind)) {
    $("drive-url").value = "";
    $("url-row").hidden = true;
  }
  render();
}

function readSettings() {
  const firmware = radio("firmware");
  const usesKernel = firmware !== "bios";
  const acpi = $("acpi").checked;
  const extra = [$("cmdline").value.trim(), $("shell").checked ? "rdinit=/bin/sh" : ""].filter(Boolean).join(" ");
  const settings = {
    firmware,
    usesKernel,
    firmwareImage: firmware === "custom" ? state.firmwareImage : null,
    kernel: usesKernel ? state.kernel : null,
    initrd: usesKernel ? state.initrd : null,
    // Kept whatever the firmware, so that a note can say why it would not boot.
    hypervisor: usesKernel ? state.hypervisor : null,
    hypervisorCmdline: $("hypervisor-cmdline").value.trim(),
    drives: state.drives.map((drive) => ({ ...drive })),
    extra,
    replaceCmdline: $("replace-cmdline").checked,
    serial: radio("console") === "serial",
    display: radio("display"),
    input: $("input").checked,
    network: $("network").checked,
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

// What blocks Start (the combinations EmulatedMachine::new refuses, caught before anything is
// downloaded) and what is only worth knowing. The library still has the last word (an EFI stub,
// a firmware header).
function notesFor(s) {
  const notes = [];
  const error = (text) => notes.push({ kind: "error", text });
  const warn = (text) => notes.push({ kind: "warn", text });
  const hasBigCd = s.drives.some((drive) => drive.kind === "cdrom" && !(drive.size < BIG_ISO_BYTES));
  if (s.firmware === "none" && !s.kernel) error("Direct boot needs a kernel.");
  if (s.firmware === "uefi" && !s.kernel && !s.drives.length) error("Add a drive with a boot loader (an ISO, an installed disk) or a kernel with an EFI stub.");
  if (s.firmware === "bios" && !s.drives.length) error("BIOS boot needs a drive.");
  if (s.firmware === "custom" && !s.firmwareImage) error("Choose a firmware image.");
  if (s.replaceCmdline && s.kernel && !s.extra) error("Replacing the default command line with nothing leaves the guest without a console.");
  if (s.hypervisor && s.firmware !== "uefi") error("A hypervisor is booted by the UEFI firmware: choose UEFI.");
  if (s.hypervisor && !s.kernel) error("A hypervisor needs a kernel to boot as its dom0.");
  if (hasBigCd && s.memoryMib < ISO_MEMORY_MIB) {
    warn("Live and installer ISOs usually need 2 GiB: their initramfs is unpacked into RAM, and with less the kernel ends in “Unable to mount root fs”.");
  }
  // GRUB on the drive picks the command line, and an ISO's has no console= for hvc0 or ttyS0.
  if (!s.kernel && s.firmware !== "none" && s.drives.length && s.display === "none") {
    warn("The drive's boot loader chooses the kernel's console, usually the screen: without a display you may see nothing after its menu.");
  }
  if (s.firmware === "uefi" && s.kernel && s.drives.length) {
    warn("UEFI boots the kernel when one is set; the drives are only attached. Remove the kernel to boot from a drive.");
  }
  if (s.hypervisor && s.display === "screen") {
    warn("Xen's own console is the terminal; with the screen as dom0's console, Xen's messages only show in the Terminal view.");
  }
  if (s.drives.some((drive) => drive.unreachable)) warn("A drive's URL did not answer; it will fail when the machine starts.");
  return notes;
}

function summaryOf(s) {
  const media = s.title ?? [
    s.hypervisor?.label,
    s.kernel?.label,
    ...s.drives.map((drive) => drive.label),
  ].filter(Boolean).join(" + ");
  return [media || "no media", FIRMWARE_LABELS[s.firmware], s.memoryMib >= 1024 ? `${+(s.memoryMib / 1024).toFixed(1)} GiB` : `${s.memoryMib} MiB`,
    `${s.cpus} vCPU${s.cpus > 1 ? "s" : ""}`].join(" · ");
}

const running = () => starting || worker !== null;

function renderDrives() {
  const list = $("drives");
  const template = $("drive-template").content;
  $("drive-count").textContent = state.drives.length ? String(state.drives.length) : "";
  if (!state.drives.length) {
    const empty = document.createElement("div");
    empty.className = "empty-drives";
    empty.textContent = "No drives. Add an installer or live ISO, or a disk image.";
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(...state.drives.map((drive, index) => {
    const row = template.cloneNode(true).firstElementChild;
    row.dataset.kind = drive.kind;
    row.querySelector(".glyph use").setAttribute("href", drive.kind === "cdrom" ? "#i-disc" : "#i-disk");
    row.querySelector(".name").textContent = drive.label;
    row.querySelector(".name").title = drive.url ?? drive.label;
    // virtio-blk names them in PCI slot order, which is this order.
    row.querySelector(".device").textContent = `vd${String.fromCharCode(97 + index)}`;
    row.querySelector(".size").textContent = [formatBytes(drive.size), drive.url ? "URL" : "file"].filter(Boolean).join(" · ");
    const select = row.querySelector("select");
    select.value = drive.kind;
    select.onchange = () => {
      drive.kind = select.value;
      render();
    };
    for (const button of row.querySelectorAll("button[data-action]")) {
      const action = button.dataset.action;
      button.disabled = (action === "up" && index === 0) || (action === "down" && index === state.drives.length - 1);
      button.onclick = () => {
        if (action === "remove") state.drives.splice(index, 1);
        else {
          const other = index + (action === "up" ? -1 : 1);
          [state.drives[index], state.drives[other]] = [state.drives[other], state.drives[index]];
        }
        render();
      };
    }
    return row;
  }));
}

// A kernel, initrd or firmware image: the file or URL chosen, or the two ways to choose one.
function renderSource(container, key, title, optionalText) {
  const source = state[key];
  const label = document.createElement("label");
  label.textContent = title;
  if (optionalText) label.append(Object.assign(document.createElement("small"), { textContent: optionalText }));
  if (source) {
    const chosen = document.createElement("div");
    chosen.className = "chosen";
    chosen.innerHTML = `<svg class="i"><use href="#${source.url ? "i-link" : "i-file"}"/></svg><span></span>`;
    chosen.querySelector("span").textContent = source.label + (source.size ? ` (${formatBytes(source.size)})` : "");
    chosen.title = source.url ?? source.label;
    const clear = Object.assign(document.createElement("button"), { type: "button", className: "ghost", title: "Remove" });
    clear.setAttribute("aria-label", `Remove the ${title.toLowerCase()}`);
    clear.innerHTML = `<svg class="i"><use href="#i-x"/></svg>`;
    clear.onclick = () => {
      state[key] = null;
      render();
    };
    chosen.append(clear);
    container.replaceChildren(label, chosen);
    return;
  }
  const pick = document.createElement("div");
  pick.className = "pick";
  const file = Object.assign(document.createElement("input"), { type: "file", hidden: true });
  file.onchange = () => {
    const picked = file.files[0];
    if (picked) state[key] = { file: picked, label: picked.name, size: picked.size };
    render();
  };
  const choose = Object.assign(document.createElement("button"), { type: "button", textContent: "File…" });
  choose.onclick = () => file.click();
  const url = Object.assign(document.createElement("input"), { type: "url", placeholder: "or a URL", spellcheck: false });
  url.setAttribute("aria-label", `${title} URL`);
  url.onchange = () => {
    const value = url.value.trim();
    if (!value) return;
    state[key] = { url: value, label: labelOfUrl(value) };
    render();
  };
  pick.append(choose, url, file);
  container.replaceChildren(label, pick);
}

function render() {
  const firmware = radio("firmware");
  // More than one vCPU needs the MADT (ACPI) and cannot run under the BIOS shim; the IOMMU is found
  // through the IVRS table. Disabled controls are reset so they cannot linger in a hidden state.
  const smpAllowed = $("acpi").checked && firmware !== "bios";
  $("cpus").disabled = !smpAllowed;
  if (!smpAllowed) $("cpus").value = "1";
  $("iommu").disabled = !$("acpi").checked;
  if ($("iommu").disabled) $("iommu").checked = false;

  const s = readSettings();
  renderDrives();
  $("kernel-panel").hidden = !s.usesKernel;
  renderSource($("kernel-source"), "kernel", "Kernel", firmware === "uefi" && !s.hypervisor ? "bzImage with an EFI stub" : "bzImage");
  renderSource($("initrd-source"), "initrd", "Initramfs", "optional");
  // Xen keeps the 16550 for itself and gives dom0 its console through it, so the console is
  // serial whatever the choice; the library makes it so, and the radios say it.
  $("hypervisor-source").hidden = firmware !== "uefi" && !s.hypervisor;
  if (!$("hypervisor-source").hidden) renderSource($("hypervisor-source"), "hypervisor", "Hypervisor", "optional: xen.gz, or Edera's xen.efi");
  $("hypervisor-cmdline-group").hidden = !s.hypervisor;
  $("hypervisor-cmdline").placeholder = DEFAULT_HYPERVISOR_CMDLINE;
  for (const input of document.querySelectorAll("input[name=console]")) input.disabled = Boolean(s.hypervisor);
  if (s.hypervisor) setRadio("console", "serial");
  $("firmware-source").hidden = firmware !== "custom";
  if (firmware === "custom") renderSource($("firmware-source"), "firmwareImage", "Firmware image");
  $("kernel-optional").textContent = firmware === "none" ? "required" : "optional";
  $("kernel-hint").textContent = s.hypervisor
    ? "The firmware boots the hypervisor with this kernel, initramfs and command line as its dom0: xen.gz the way GRUB's multiboot2 does, Edera's xen.efi through its media loaders (an upstream xen.efi cannot take them; use its xen.gz)."
    : firmware === "uefi"
      ? "With a kernel the firmware boots it, like QEMU's ‑kernel under OVMF, and the drives are only attached."
      : "";
  $("kernel-hint").hidden = !$("kernel-hint").textContent;
  // A drive's boot loader brings its own command line; the arguments only matter with a kernel.
  $("cmdline-group").hidden = !s.kernel && firmware !== "none";
  $("cmdline-preview").textContent = s.cmdline || "(empty)";
  $("firmware-hint").textContent = FIRMWARE_HINTS[firmware];

  const notes = notesFor(s);
  $("notes").replaceChildren(...notes.map(({ kind, text }) => {
    const item = Object.assign(document.createElement("li"), { textContent: text });
    item.dataset.kind = kind;
    return item;
  }));
  const blocked = notes.some((note) => note.kind === "error");
  for (const id of ["start", "start-sidebar"]) $(id).disabled = blocked || running();
  $("start").title = blocked ? notes.find((note) => note.kind === "error").text : "Start the machine";
  saveSettings();
}

// Per-visitor convenience only: files cannot be remembered, and storage may be unavailable.
function saveSettings() {
  const saved = {};
  for (const name of RADIOS) saved[name] = radio(name);
  for (const id of CHECKBOXES) saved[id] = $(id).checked;
  for (const id of SELECTS) saved[id] = $(id).value;
  saved.cmdline = $("cmdline").value;
  saved.hypervisorCmdline = $("hypervisor-cmdline").value;
  for (const key of ["kernel", "initrd", "hypervisor", "firmwareImage"]) saved[key] = state[key]?.url ?? null;
  saved.drives = state.drives.filter((drive) => drive.url).map(({ url, kind }) => ({ url, kind }));
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(saved)); } catch { /* private mode */ }
}

// Returns what was saved, so the asset prefill below leaves alone a field the visitor cleared.
function restoreSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") ?? {}; } catch { /* nothing saved */ }
  const params = new URLSearchParams(location.search);
  for (const name of RADIOS) setRadio(name, params.get(name) ?? saved[name]);
  for (const id of CHECKBOXES) if (id in saved) $(id).checked = Boolean(saved[id]);
  for (const id of SELECTS) if (id in saved) setSelect(id, saved[id]);
  if (params.has("memory")) setSelect("memory", Math.min(Number(params.get("memory")), MAX_MEMORY_MIB));
  $("cmdline").value = "cmdline" in saved ? saved.cmdline : DEFAULT_CMDLINE_EXTRA;
  if (params.has("args")) $("cmdline").value = params.get("args");
  if (params.has("iommu")) $("iommu").checked = $("acpi").checked = true;
  if (params.has("network")) $("network").checked = !["0", "off", "false"].includes(params.get("network"));
  $("hypervisor-cmdline").value = params.get("hypervisor-args") ?? saved.hypervisorCmdline ?? "";
  for (const key of ["kernel", "initrd", "hypervisor", "firmwareImage"]) {
    const url = params.get(key === "firmwareImage" ? "firmware-image" : key) ?? saved[key];
    if (url) state[key] = { url, label: labelOfUrl(url) };
  }
  // ?cdrom= and ?disk= (both repeat) replace the remembered drives.
  const fromUrl = [...params.getAll("cdrom").map((url) => ({ url, kind: "cdrom" })), ...params.getAll("disk").map((url) => ({ url, kind: "disk" }))];
  for (const { url, kind } of fromUrl.length ? fromUrl : saved.drives ?? []) {
    if (url) addDrive({ url, label: labelOfUrl(url) }, kind === "cdrom" ? "cdrom" : "disk");
  }
  return saved;
}

// A kernel left over from an earlier choice wins over every drive under UEFI (the firmware boots
// the kernel it is given, QEMU -kernel style), so the drive presets drop it, and the kernel preset
// drops the demo's, along with the command line that went with it. A hypervisor goes with them:
// it would turn the next kernel into a dom0.
function clearKernel() {
  state.kernel = null;
  state.initrd = null;
  clearHypervisor();
  $("cmdline").value = "";
  $("shell").checked = false;
}

function clearHypervisor() {
  state.hypervisor = null;
  $("hypervisor-cmdline").value = "";
}

// By URL rather than a flag, so a demo kernel remembered across a reload still counts.
const isDemoSource = (source) => Boolean(source?.url?.startsWith(new URL(".", DEMO_URL).href));

function applyPreset(name) {
  if (name === "iso" || name === "disk" || (name === "kernel" && isDemoSource(state.kernel))) clearKernel();
  // Direct boot cannot take a hypervisor, whoever's kernel this is.
  if (name === "kernel") clearHypervisor();
  if (name === "iso") {
    setRadio("firmware", "uefi");
    setRadio("display", "screen");
    $("input").checked = true;
    setSelect("memory", ISO_MEMORY_MIB);
    if (!state.drives.some((drive) => drive.kind === "cdrom")) $("cdrom-file").click();
  } else if (name === "disk") {
    setRadio("firmware", "uefi");
    setRadio("display", "both");
    $("input").checked = true;
    if (Number($("memory").value) < 1024) setSelect("memory", 1024);
    if (!state.drives.some((drive) => drive.kind === "disk")) $("disk-file").click();
  } else if (name === "kernel") {
    setRadio("firmware", "none");
    setRadio("display", "none");
    $("shell").checked = true;
    if (!state.kernel) queueMicrotask(() => $("kernel-source").querySelector("button")?.focus());
  } else if (name.startsWith("demo:")) {
    const machine = demos.find((d) => d.id === name.slice("demo:".length));
    if (machine) applyDemo(machine);
  }
  render();
}

// Loads demo.json's machine into the settings, so what runs is what the sidebar shows.
function applyDemo(d) {
  const at = (name) => ({ url: new URL(name, DEMO_URL).href, label: name });
  state.kernel = at(d.kernel);
  state.initrd = at(d.initrd);
  state.hypervisor = d.hypervisor ? at(d.hypervisor) : null;
  $("hypervisor-cmdline").value = d.hypervisorArgs ?? "";
  state.drives = [];
  setRadio("firmware", d.firmware);
  setRadio("console", d.console === "serial" ? "serial" : "virtio");
  setRadio("display", d.display ?? "none");
  $("input").checked = Boolean(d.input);
  setSelect("memory", d.memoryMib);
  setSelect("cpus", 1);
  $("shell").checked = false;
  $("replace-cmdline").checked = false;
  $("cmdline").value = d.args ?? "";
  for (const id of ["jit", "acpi", "rng", "reboot"]) $(id).checked = true;
  $("iommu").checked = Boolean(d.iommu);
}

function setRunning(isRunning) {
  $("config").disabled = isRunning;
  $("locked-note").hidden = !isRunning;
  $("stop").hidden = !isRunning;
  $("start").hidden = isRunning;
  if (!isRunning) render();
  else for (const id of ["start", "start-sidebar"]) $(id).disabled = true;
}

function setSidebar(open) {
  $("sidebar").toggleAttribute("data-collapsed", !open);
  $("toggle-sidebar").setAttribute("aria-expanded", String(open));
  relayout();
}

async function load(source, label) {
  if (!source) return null;
  if (source.file) return source.file.arrayBuffer();
  setStatus(`Downloading ${label}…`, "busy");
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
window.addEventListener("resize", relayout);

// The screen: frames from the worker go onto the canvas, the canvas's keys and pointer go back
// through the event ring. "terminal" or "screen" is in front.
let view = "terminal";
const canvas = $("screen");

// The screen as a WebGL2 texture the size of the screen. Frames arrive as rectangles of the
// library's pixels (blue, green, red, unused), uploaded as they are into an RGBA texture; the
// fragment shader reads its pixel with texelFetch and swaps red and blue, so no thread converts
// pixels and the upload is the GPU's (a quarter of what putImageData costs a 1280x800 frame).
// The canvas is exactly the texture's size: CSS scales it, as before.
const SCREEN_VERTEX_SHADER = `#version 300 es
void main() {
  // One triangle that covers the viewport: (-1,-1), (3,-1), (-1,3).
  vec2 corner = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(corner, 0.0, 1.0);
}`;
const SCREEN_FRAGMENT_SHADER = `#version 300 es
// highp: mediump may not tell gl_FragCoord.y 1599.5 from 1598.5.
precision highp float;
uniform highp sampler2D screen;
out vec4 color;
void main() {
  // Texture row 0 is the screen's top row; the framebuffer's row 0 is its bottom.
  ivec2 size = textureSize(screen, 0);
  vec4 pixel = texelFetch(screen, ivec2(gl_FragCoord.x, float(size.y) - gl_FragCoord.y), 0);
  color = vec4(pixel.b, pixel.g, pixel.r, 1.0);
}`;

const GL_OPTIONS = { alpha: false, antialias: false, depth: false, stencil: false };

// Returns whether the screen's program links: false on a lost context.
function buildScreenProgram(gl) {
  const compile = (kind, source) => {
    const shader = gl.createShader(kind);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, SCREEN_VERTEX_SHADER));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, SCREEN_FRAGMENT_SHADER));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return false;
  gl.useProgram(program);
  // An empty vertex array: the vertex shader makes its corners from gl_VertexID.
  gl.bindVertexArray(gl.createVertexArray());
  return true;
}

function createGlRenderer() {
  // Tried on a canvas of its own first: once the screen's canvas has a WebGL context it can
  // never have a 2D one, so it must not get one that turns out unusable.
  const probe = document.createElement("canvas").getContext("webgl2", GL_OPTIONS);
  if (!probe || !buildScreenProgram(probe)) return null;
  probe.getExtension("WEBGL_lose_context")?.loseContext();
  const gl = canvas.getContext("webgl2", GL_OPTIONS);
  if (!gl) return null;
  let texture = null;
  let lost = false;
  function setUp() {
    buildScreenProgram(gl);
    texture = null;
    resize();
  }
  // WebGL zeroes a new texture: a black screen until the first frame.
  function resize() {
    if (texture) gl.deleteTexture(texture);
    texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, canvas.width, canvas.height);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function present() {
    if (!lost) gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  // A lost context (a GPU reset, too many contexts) takes the texture with it: once it is back,
  // the worker is asked for the whole screen again.
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    lost = true;
  });
  canvas.addEventListener("webglcontextrestored", () => {
    lost = false;
    setUp();
    present();
    requestWholeScreen();
  });
  setUp();
  return {
    resize() {
      resize();
      present();
    },
    draw(rects, pixels) {
      if (lost) return;
      let at = 0;
      for (let i = 0; i < rects.length; i += RECT_WORDS) {
        const [x, y, width, height] = rects.subarray(i, i + RECT_WORDS);
        const bytes = width * height * BYTES_PER_PIXEL;
        gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, width, height, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(pixels, at, bytes));
        at += bytes;
      }
      present();
    },
    // The drawing buffer is cleared once composited; drawing again in the same task as
    // `toBlob` gives it the picture.
    present,
  };
}

// Without WebGL2: a 2D canvas, the channels reordered here (on the page, still not on the
// emulator's thread).
function create2dRenderer() {
  const context = canvas.getContext("2d", { alpha: false });
  const clear = () => {
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
  };
  return {
    resize: clear,
    draw(rects, pixels) {
      const source = new Uint8Array(pixels);
      let at = 0;
      for (let i = 0; i < rects.length; i += RECT_WORDS) {
        const [x, y, width, height] = rects.subarray(i, i + RECT_WORDS);
        const image = new ImageData(width, height);
        const target = image.data;
        for (let byte = 0; byte < target.length; byte += BYTES_PER_PIXEL) {
          target[byte] = source[at + byte + 2];
          target[byte + 1] = source[at + byte + 1];
          target[byte + 2] = source[at + byte];
          target[byte + 3] = 255;
        }
        context.putImageData(image, x, y);
        at += target.length;
      }
    },
    present() {},
  };
}

// The next frame is the whole screen, not what changed since the last one: the page lost its
// picture. When a frame is in flight, its acknowledgement asks instead.
let resendWanted = false;
function requestWholeScreen() {
  if (!control) return;
  if (Atomics.compareExchange(control, CTRL.FRAME_ACK, FRAME_READY.YES, FRAME_READY.RESEND) !== FRAME_READY.YES) {
    resendWanted = true;
  }
}

const renderer = createGlRenderer() ?? create2dRenderer();
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
const LEFT_SHIFT = USAGES.ShiftLeft;
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
function relayout() {
  if (view === "terminal" && !$("terminal").hidden) fit.fit();
  else if (view === "screen") layoutScreen();
  requestGuestSize();
}

function focusView() {
  relayout();
  if (!hasRun) return;
  if (view === "terminal") term.focus();
  else canvas.focus();
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
  renderer.resize();
  resendWanted = false;
  canvas.style.cursor = "";
  const note = s.display === "none" ? "No screen: keys and the pointer go to the guest's virtio keyboard and tablet." : "";
  $("screen-note").textContent = note;
  $("screen-note").hidden = !note;
  const hasScreen = s.display !== "none" || s.input;
  $("screen-tools").hidden = !hasScreen;
  $("fit-guest").hidden = $("screenshot").hidden = s.display === "none";
  $("ctrl-alt-del").hidden = !s.input;
  view = "";
  showView(hasScreen && s.display === "screen" ? "screen" : "terminal");
}

function drawFrame(frame) {
  if (canvas.width !== frame.screenWidth || canvas.height !== frame.screenHeight) {
    canvas.width = frame.screenWidth;
    canvas.height = frame.screenHeight;
    renderer.resize();
    layoutScreen();
  }
  renderer.draw(frame.rects, frame.pixels);
  if (control) Atomics.store(control, CTRL.FRAME_ACK, resendWanted ? FRAME_READY.RESEND : FRAME_READY.YES);
  resendWanted = false;
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
  if ($("fit-guest").getAttribute("aria-pressed") !== "true" || $("fit-guest").hidden) return;
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
// The canvas is letterboxed; a click on the bars around it still means "type into the guest".
$("screen-view").addEventListener("mousedown", (event) => {
  if (event.target !== canvas) {
    canvas.focus();
    event.preventDefault();
  }
});
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
$("fit-guest").onclick = () => {
  const on = $("fit-guest").getAttribute("aria-pressed") !== "true";
  $("fit-guest").setAttribute("aria-pressed", String(on));
  requestGuestSize();
};
// The browser keeps Ctrl+Alt+Delete for itself, so it goes to the guest's virtio keyboard from here.
$("ctrl-alt-del").onclick = () => {
  const keys = [USAGES.ControlLeft, USAGES.AltLeft, USAGES.Delete];
  for (const key of keys) pushEvent(EVENT.KEY, key, 1);
  for (const key of keys.reverse()) pushEvent(EVENT.KEY, key, 0);
  focusView();
};
$("screenshot").onclick = () => {
  renderer.present();
  canvas.toBlob((blob) => {
    if (!blob) return;
    const link = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `uemulator-${new Date().toISOString().replace(/[:.]/g, "-")}.png`,
    });
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
};
$("fullscreen").onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else $("stage").requestFullscreen?.().catch(() => {});
};
document.addEventListener("fullscreenchange", () => {
  relayout();
  focusView();
});

function formatUptime(seconds) {
  const s = Math.floor(seconds);
  return s >= 3600
    ? `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
    : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function showBanner(text, kind) {
  $("banner-text").textContent = text;
  $("banner").dataset.kind = kind;
  $("banner").hidden = false;
}

// The bar's network pill: the guest's address on the shared segment and how many other tabs'
// machines are on it.
function showNetwork({ address, peers, serving }) {
  const pill = $("net-status");
  const dotted = address ? [24, 16, 8, 0].map((shift) => (address >>> shift) & 0xff).join(".") : null;
  const others = `${peers} other machine${peers === 1 ? "" : "s"}`;
  pill.firstElementChild.textContent = `${dotted ?? "no address yet"} · ${others}`;
  pill.dataset.kind = dotted ? "network" : "network-waiting";
  pill.title = [
    `Shared network, NIC ${formatMac(networkMac)}`,
    dotted ? `address ${dotted}/24` : "the guest has no address yet (run DHCP on its interface, e.g. udhcpc -i eth0)",
    `${others} in this browser's other tabs`,
    serving ? "this tab is the segment's DHCP server (10.0.2.2)" : "another tab is the DHCP server (10.0.2.2)",
  ].join("\n");
  pill.hidden = false;
}

function finish(text, kind) {
  starting = false;
  clearInterval(clock);
  worker?.terminate();
  worker = null;
  compiler?.terminate();
  compiler = null;
  // Only now that the worker is gone: leaving hands the server role to another tab.
  segment?.leave();
  segment = null;
  $("net-status").hidden = true;
  control = null;
  ring = null;
  events = null;
  // The library's wording names the CLI flag; point at the page's own control instead.
  const shown = text.replace("boot it with firmware none", "choose Direct under Firmware");
  setStatus(kind === "error" ? shown : `Stopped: ${shown}`, kind);
  showBanner(shown.charAt(0).toUpperCase() + shown.slice(1), kind);
  setRunning(false);
}

async function launch() {
  if (!crossOriginIsolated) {
    setStatus(NOT_ISOLATED, "error");
    return;
  }
  const s = readSettings();
  if (notesFor(s).some((note) => note.kind === "error")) {
    setSidebar(true);
    return;
  }
  hasRun = true;
  current = s;
  $("welcome").hidden = true;
  $("banner").hidden = true;
  // The settings are locked while it runs; the screen gets the room. The bar's Settings reopens them.
  setSidebar(false);
  starting = true;
  setRunning(true);
  setStatus("Starting…", "busy");
  term.reset();
  resetScreen(s);
  try {
    const firmware = await loadFirmware(s);
    const realMode = firmware.kind === FIRMWARE.REAL_MODE;
    const kernel = realMode ? null : await load(s.kernel, "kernel");
    const initrd = realMode ? null : await load(s.initrd, "initramfs");
    const hypervisor = realMode ? null : await load(s.hypervisor, "hypervisor");

    const shared = new SharedArrayBuffer(CONTROL_BYTES);
    control = new Int32Array(shared, 0, RING_OFFSET / 4);
    ring = new Uint8Array(shared, RING_OFFSET, RING_BYTES);
    events = new Int32Array(shared, EVENT_OFFSET, EVENT_RECORDS * EVENT_WORDS);
    control[CTRL.COLS] = term.cols;
    control[CTRL.ROWS] = term.rows;
    control[CTRL.FRAME_ACK] = FRAME_READY.YES;
    let network = null;
    if (s.network) {
      const tab = randomTab();
      networkMac = randomMac();
      network = { mac: networkMac.buffer, tab };
      segment = joinSegment({ control, ring: new Uint8Array(shared, NET_OFFSET, NET_RING_BYTES), tab });
      showNetwork({ address: 0, peers: 0, serving: false });
    }

    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    let compileQueue = null;
    if (s.jit && compileWorkerHelps()) {
      compileQueue = createCompileQueue();
      compiler = new Worker(new URL("./jit_compiler.js", import.meta.url), { type: "module" });
      compiler.postMessage({ queue: compileQueue });
    }
    starting = false;
    worker.onmessage = ({ data }) => {
      if (data.type === "output") {
        term.write(new Uint8Array(data.bytes));
      } else if (data.type === "frame") {
        drawFrame(data);
      } else if (data.type === "cursor") {
        setCursor(data);
      } else if (data.type === "network") {
        showNetwork(data);
      } else if (data.type === "log") {
        console.log("[uemulator]", data.text);
      } else if (data.type === "started") {
        startedAt = performance.now();
        const tick = () => setStatus(`Running ${formatUptime((performance.now() - startedAt) / 1000)} · ${summaryOf(current)}`, "running");
        tick();
        clock = setInterval(tick, 1000);
        focusView();
      } else if (data.type === "exit") {
        const name = EXIT_NAMES[data.code] ?? `error ${data.code}`;
        const detail = data.message ? `: ${data.message}` : "";
        finish(`${name}${detail}`, data.code === 0 || data.code === 3 ? "" : "error");
      }
    };
    worker.onerror = (event) => finish(`worker failed: ${event.message}`, "error");

    // The worker reads the drives on demand (a File slice or an HTTP range request per miss), so
    // they are not downloaded here. A relative URL is resolved against the page, not the worker.
    const disks = s.drives.map((drive) => ({
      ...(drive.file ? { file: drive.file } : { url: new URL(drive.url, location.href).href }),
      readOnly: drive.kind === "cdrom",
      label: drive.label,
    }));
    const options = (s.jit ? OPTION.JIT : 0) | (s.acpi ? OPTION.ACPI : 0) | (s.iommu ? OPTION.IOMMU : 0) |
      (s.rng ? OPTION.RNG : 0) | (s.serial ? OPTION.SERIAL_CONSOLE : 0) | (s.reboot ? OPTION.REBOOT : 0) |
      (s.replaceCmdline ? OPTION.REPLACE_CMDLINE : 0) | (s.display !== "none" ? OPTION.DISPLAY : 0) |
      (s.display === "screen" ? OPTION.SCREEN_CONSOLE : 0) | (s.input ? OPTION.INPUT : 0);
    const message = {
      firmware,
      kernel,
      initrd,
      hypervisor,
      hypervisorCmdline: s.hypervisorCmdline,
      cmdline: s.extra,
      memoryMib: s.memoryMib,
      cpus: realMode ? 1 : s.cpus,
      options,
      disks,
      cols: term.cols,
      rows: term.rows,
      control: shared,
      network,
      compileQueue,
    };
    setStatus("Starting…", "busy");
    worker.postMessage(message, [kernel, initrd, hypervisor, firmware.image].filter(Boolean));
  } catch (error) {
    finish(String(error.message ?? error), "error");
  }
}

for (const input of document.querySelectorAll("#config input:not([type=file]), #config select")) {
  input.addEventListener(input.type === "text" || input.type === "url" ? "input" : "change", render);
}
for (const button of document.querySelectorAll("#presets button")) button.onclick = () => applyPreset(button.dataset.preset);
$("add-cdrom").onclick = () => $("cdrom-file").click();
$("add-disk").onclick = () => $("disk-file").click();
$("welcome-iso").onclick = () => applyPreset("iso");
$("welcome-disk").onclick = () => applyPreset("disk");
$("cdrom-file").onchange = () => {
  addFiles([...$("cdrom-file").files], { kind: "cdrom" });
  $("cdrom-file").value = "";
};
$("disk-file").onchange = () => {
  addFiles([...$("disk-file").files], { kind: "disk" });
  $("disk-file").value = "";
};
$("add-url").onclick = () => {
  $("url-row").hidden = !$("url-row").hidden;
  if (!$("url-row").hidden) $("drive-url").focus();
};
$("drive-url-add").onclick = () => addDriveUrl($("drive-url").value);
$("drive-url").addEventListener("keydown", (event) => {
  if (event.key === "Enter") addDriveUrl($("drive-url").value);
});
$("start").onclick = launch;
$("start-sidebar").onclick = launch;
$("banner-restart").onclick = launch;
$("banner-close").onclick = () => { $("banner").hidden = true; };
$("toggle-sidebar").onclick = () => setSidebar($("sidebar").hasAttribute("data-collapsed"));
$("stop").onclick = () => {
  if (control) Atomics.store(control, CTRL.STOP, 1);
  else if (starting) toast("Still downloading; the machine stops as soon as it starts.");
};

// Dropping files anywhere attaches them; the overlay only shows while a drag carries files.
let dragDepth = 0;
const carriesFiles = (event) => [...(event.dataTransfer?.types ?? [])].includes("Files");
window.addEventListener("dragenter", (event) => {
  if (!carriesFiles(event) || running()) return;
  dragDepth++;
  $("dropzone").hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("dropzone").hidden = true;
});
window.addEventListener("dragover", (event) => {
  if (carriesFiles(event)) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  $("dropzone").hidden = true;
  if (running()) {
    toast("Stop the machine to attach drives.");
    return;
  }
  addFiles([...event.dataTransfer.files], { detectKernel: true });
});

const saved = restoreSettings();
setRunning(false);
// On a phone the sidebar would cover the welcome pane, which already offers the common starts.
setSidebar(!matchMedia("(max-width: 900px)").matches);

// `hack/web.sh` links the cached Alpine netboot files (and WEB_DISK) under assets/; offer them
// when present and the visitor has not chosen (or cleared) them before. Refilling a cleared
// kernel would quietly turn a UEFI disk boot back into a kernel boot.
const probes = [
  ["kernel", "assets/vmlinuz-virt"],
  ["initrd", "assets/initramfs-virt"],
  ["drive", "assets/disk"],
]
  .filter(([key]) => !(key === "drive" ? "drives" in saved : key in saved))
  .map(([key, url]) => fetch(url, { method: "HEAD" })
    .then((response) => {
      if (!response.ok) return;
      if (key === "drive") {
        if (!state.drives.length) addDrive({ url, label: "disk (assets)" }, "disk");
      } else if (!state[key]) {
        state[key] = { url, label: labelOfUrl(url) };
      }
    })
    .catch(() => {}));
Promise.all(probes).then(render);

// A welcome button and a preset per machine; the first button is the primary one.
const demoReady = fetch(DEMO_URL)
  .then((response) => (response.ok ? response.json() : null))
  .catch(() => null)
  .then((found) => {
    demos = found?.machines ?? [];
    $("demo-actions").replaceChildren(...demos.map((machine, index) => {
      const button = Object.assign(document.createElement("button"), {
        type: "button",
        className: index === 0 ? "primary big" : "big",
        title: `${machine.name}, ${Math.round(machine.bytes / 1e6)} MB download`,
        innerHTML: `<svg class="i"><use href="#i-play"/></svg>`,
      });
      button.append(`Boot ${machine.label}`);
      button.onclick = () => {
        applyDemo(machine);
        render();
        launch();
      };
      return button;
    }));
    $("demo-actions").hidden = !demos.length;
    for (const machine of demos) {
      const preset = Object.assign(document.createElement("button"), {
        type: "button",
        textContent: machine.label,
        title: machine.name,
      });
      preset.onclick = () => applyPreset(`demo:${machine.id}`);
      $("presets").append(preset);
    }
  });

if (!crossOriginIsolated) {
  setStatus(NOT_ISOLATED, "error");
} else if (new URLSearchParams(location.search).has("autoboot")) {
  // In demo mode ?autoboot boots the first demo (?demo=ID another); otherwise the settings as restored.
  Promise.all([...probes, demoReady]).then(() => {
    const wanted = new URLSearchParams(location.search).get("demo");
    const machine = demos.find((d) => d.id === wanted) ?? demos[0];
    if (machine) applyDemo(machine);
    render();
    launch();
  });
}
