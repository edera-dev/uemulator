// The render worker (docs/plans/25-gpu-3d.md): owns the 3D screen's canvas as an OffscreenCanvas
// with a WebGL2 context, reads the guest's GPU commands from the ring worker.js fills, and answers
// fences through the completion ring. A worker of its own because the emulator's never returns
// to its event loop, and a WebGL canvas in a worker shows its frame only when that worker does.
//
// The page sends { type: "start", canvas, commands, completions, control } and gets back
// { type: "ready", limits, renderer } (what uw_set_gpu takes) or { type: "failed", message };
// then { type: "scanout", active, width, height } whenever the guest's scanout moves to a 3D
// resource or away from one, and { type: "log", text }.
import { LIMIT_PARAMETERS, RESOURCE_Y_0_TOP, Renderer, encodeRecord, openRing, takeRecord, writeRecords } from "./gpu.js";
import { CTRL, RING_OFFSET } from "./protocol.js";

// PIPE_TEXTURE_2D and PIPE_TEXTURE_RECT: what M0 can clear and show.
const TEXTURE_2D = 2;
const TEXTURE_RECT = 5;
// Records handled before the worker yields, so the canvas gets to show frames under a stream
// that never pauses.
const RECORDS_PER_TURN = 4096;
// Without Atomics.waitAsync (older Firefox), how often the ring is looked at.
const POLL_MS = 2;

let gl;
let canvas;
let commands;
let completions;
let control;
let renderer;
// Encoded completions the ring had no room for yet.
let unsent = [];
// The GL side of each resource drawn to: { texture, framebuffer }, made on first use.
const targets = new Map();
let shown = null;

const log = (text) => postMessage({ type: "log", text });

function target(resource) {
  let found = targets.get(resource.id);
  if (found) return found;
  if ((resource.target !== TEXTURE_2D && resource.target !== TEXTURE_RECT) || !resource.width || !resource.height) return null;
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  // Every colour format is RGBA8 in M0: a clear is the same colour in any channel order.
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, resource.width, resource.height);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  found = { texture, framebuffer };
  targets.set(resource.id, found);
  return found;
}

function forget(id) {
  const found = targets.get(id);
  if (!found) return;
  gl.deleteFramebuffer(found.framebuffer);
  gl.deleteTexture(found.texture);
  targets.delete(id);
}

const backend = {
  resource() {},
  destroyResource: forget,
  clear(resource, [red, green, blue, alpha]) {
    const found = target(resource);
    if (!found) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, found.framebuffer);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.clearColor(red, green, blue, alpha);
    gl.clear(gl.COLOR_BUFFER_BIT);
  },
  scanout(resource, rect) {
    shown = resource;
    if (resource && (canvas.width !== rect.width || canvas.height !== rect.height)) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    postMessage({ type: "scanout", active: !!resource, width: rect.width, height: rect.height });
  },
  flush(resource) {
    const found = targets.get(resource.id);
    if (!found || resource !== shown) return;
    const { width, height } = resource;
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    // Without this a blit in a later task than the frame before it left the canvas black
    // (Chromium on macOS), though the texture held the colour and a readPixels before the blit
    // made it work. The likely cause is the implicit clear of a drawing buffer that has been
    // shown landing after the blit; clearing it here first is what was seen to fix it.
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, found.framebuffer);
    // The canvas's row 0 is its bottom, as is a GL texture's unless the guest said otherwise.
    const flip = resource.flags & RESOURCE_Y_0_TOP;
    gl.blitFramebuffer(0, 0, width, height, 0, flip ? canvas.height : 0, width, flip ? 0 : canvas.height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
  },
  reset() {
    for (const id of [...targets.keys()]) forget(id);
    shown = null;
    postMessage({ type: "scanout", active: false, width: 0, height: 0 });
  },
};

function sendCompletions() {
  while (unsent.length) {
    const taken = writeRecords(completions, unsent[0]);
    if (taken === 0) break;
    unsent.shift();
  }
  // The emulator's worker sleeps on HEAD while its guest is halted; a halted guest is often
  // waiting for exactly this fence.
  Atomics.notify(control, CTRL.HEAD);
}

async function run() {
  for (;;) {
    let handled = 0;
    let record;
    while (handled < RECORDS_PER_TURN && (record = takeRecord(commands))) {
      renderer.handle(record.kind, record.payload);
      handled++;
    }
    if (unsent.length) sendCompletions();
    const head = Atomics.load(commands.words, 0);
    if (handled === RECORDS_PER_TURN || head !== Atomics.load(commands.words, 1)) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      continue;
    }
    if (Atomics.waitAsync) {
      const wait = Atomics.waitAsync(commands.words, 0, head);
      if (wait.async) await wait.value;
    } else {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }
  }
}

onmessage = ({ data }) => {
  if (data.type !== "start") return;
  canvas = data.canvas;
  gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false });
  if (!gl) {
    postMessage({ type: "failed", message: "no WebGL2 in a worker" });
    return;
  }
  commands = openRing(data.commands);
  completions = openRing(data.completions);
  control = new Int32Array(data.control, 0, RING_OFFSET / 4);
  renderer = new Renderer(backend, (kind, words, bytes) => unsent.push(encodeRecord(kind, words, bytes)), log);
  postMessage({
    type: "ready",
    limits: LIMIT_PARAMETERS.map((name) => gl.getParameter(gl[name])),
    renderer: "WebGL2",
  });
  run().catch((error) => log(`render worker stopped: ${error?.stack ?? error}`));
};
