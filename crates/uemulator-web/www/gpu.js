// The guest's 3D (docs/plans/25-gpu-3d.md): the records src/gpu.rs encodes, the rings that carry
// them between the emulator worker and the render worker (render.js), and the renderer that acts
// on them. No DOM here, so hack/wasm/boot.mjs runs the same renderer under Node with a backend
// that only logs.
//
// This is M0's renderer: it decodes only what clearing a surface takes (surfaces,
// SET_FRAMEBUFFER_STATE, CLEAR) and skips every other virgl command by its length, so the guest
// runs, fences complete and the scanout shows its clear colour. Draws, shaders, transfers and
// readbacks are M2 to M4, in Rust (crates/uemulator-virgl).

// Record kinds, the RECORD_* and COMPLETION_* constants in src/gpu.rs. A record is a Uint32 kind,
// a Uint32 payload length in bytes, then the payload padded to four bytes.
export const RECORD = {
  RESET: 1,
  CREATE_CONTEXT: 2,
  DESTROY_CONTEXT: 3,
  ATTACH_RESOURCE: 4,
  DETACH_RESOURCE: 5,
  CREATE_RESOURCE: 6,
  DESTROY_RESOURCE: 7,
  WRITE_RESOURCE: 8,
  READ_RESOURCE: 9,
  SUBMIT: 10,
  SET_SCANOUT: 11,
  FLUSH: 12,
  FENCE: 13,
};
export const RECORD_NAMES = Object.fromEntries(Object.entries(RECORD).map(([name, kind]) => [kind, name]));
export const COMPLETION = { FENCE: 1, READ: 2 };
const HEADER_BYTES = 8;
const padded = (length) => (length + 3) & ~3;

// The HostLimits words uw_set_gpu takes, in its order (crates/uemulator-virgl/src/caps.rs).
export const LIMIT_PARAMETERS = [
  "MAX_TEXTURE_SIZE", "MAX_3D_TEXTURE_SIZE", "MAX_CUBE_MAP_TEXTURE_SIZE", "MAX_ARRAY_TEXTURE_LAYERS",
  "MAX_DRAW_BUFFERS", "MAX_SAMPLES", "MAX_VERTEX_ATTRIBS", "MAX_VERTEX_OUTPUT_COMPONENTS",
  "MAX_FRAGMENT_UNIFORM_BLOCKS", "MAX_UNIFORM_BLOCK_SIZE", "UNIFORM_BUFFER_OFFSET_ALIGNMENT",
  "MAX_TEXTURE_IMAGE_UNITS", "MIN_PROGRAM_TEXEL_OFFSET", "MAX_PROGRAM_TEXEL_OFFSET",
];
// What GLES 3.0 guarantees, for a host without a GL context (Node).
export const MINIMUM_LIMITS = [2048, 256, 2048, 256, 4, 4, 16, 64, 12, 16384, 256, 16, -8, 7];

// A ring is a SharedArrayBuffer: Int32 HEAD (where the writer writes next) and TAIL (where the
// reader reads next) as byte offsets into the data after RING_DATA_OFFSET, which wraps. One byte
// stays free so HEAD === TAIL means empty. Only whole records go in, but one may wrap.
const RING_HEAD = 0;
const RING_TAIL = 1;
const RING_DATA_OFFSET = 64;
// Above the 32 MiB the library queues before it stops the guest (gpu.rs's QUEUE_BYTES), so the
// emulator's side never holds back more than one batch.
export const COMMAND_RING_BYTES = 48 << 20;
// Readbacks come back here; M0 answers them empty.
export const COMPLETION_RING_BYTES = 1 << 20;

export function createRing(bytes) {
  return new SharedArrayBuffer(RING_DATA_OFFSET + bytes);
}

export function openRing(buffer) {
  return {
    words: new Int32Array(buffer, 0, 2),
    data: new Uint8Array(buffer, RING_DATA_OFFSET),
    capacity: buffer.byteLength - RING_DATA_OFFSET,
  };
}

const used = (ring, head, tail) => (head - tail + ring.capacity) % ring.capacity;

function copyIn(ring, at, bytes) {
  const first = Math.min(bytes.length, ring.capacity - at);
  ring.data.set(bytes.subarray(0, first), at);
  if (first < bytes.length) ring.data.set(bytes.subarray(first), 0);
  return (at + bytes.length) % ring.capacity;
}

function copyOut(ring, at, out) {
  const first = Math.min(out.length, ring.capacity - at);
  out.set(ring.data.subarray(at, at + first));
  if (first < out.length) out.set(ring.data.subarray(0, out.length - first), first);
  return out;
}

// The Uint32 at byte `at` of a record in the ring, wherever the wrap falls.
function ringWord(ring, at) {
  const bytes = copyOut(ring, at % ring.capacity, new Uint8Array(4));
  return new DataView(bytes.buffer).getUint32(0, true);
}

// Copies whole records from `source` (a view on the module's memory) into the ring, as many as
// fit, and returns the bytes taken. A first record too big for the ring ever is dropped and
// counted as taken (`onDrop` hears of it): waiting for room would wait forever.
export function writeRecords(ring, source, onDrop) {
  const view = new DataView(source.buffer, source.byteOffset, source.byteLength);
  let head = Atomics.load(ring.words, RING_HEAD);
  const tail = Atomics.load(ring.words, RING_TAIL);
  let taken = 0;
  while (taken + HEADER_BYTES <= source.length) {
    const total = HEADER_BYTES + padded(view.getUint32(taken + 4, true));
    if (total > ring.capacity - 1) {
      onDrop?.(view.getUint32(taken, true), total);
      taken += total;
      continue;
    }
    if (total > ring.capacity - 1 - used(ring, head, tail)) break;
    head = copyIn(ring, head, source.subarray(taken, taken + total));
    taken += total;
  }
  Atomics.store(ring.words, RING_HEAD, head);
  Atomics.notify(ring.words, RING_HEAD);
  return Math.min(taken, source.length);
}

// Copies whole records into `out` and returns the bytes copied, or minus what the next record
// needs when it alone does not fit.
export function readRecords(ring, out) {
  const head = Atomics.load(ring.words, RING_HEAD);
  let tail = Atomics.load(ring.words, RING_TAIL);
  let copied = 0;
  while (tail !== head) {
    const total = HEADER_BYTES + padded(ringWord(ring, tail + 4));
    if (copied + total > out.length) {
      if (copied === 0) return -total;
      break;
    }
    copyOut(ring, tail, out.subarray(copied, copied + total));
    tail = (tail + total) % ring.capacity;
    copied += total;
  }
  Atomics.store(ring.words, RING_TAIL, tail);
  return copied;
}

// The next record as { kind, payload } (a copy), or null when the ring is empty.
export function takeRecord(ring) {
  const head = Atomics.load(ring.words, RING_HEAD);
  const tail = Atomics.load(ring.words, RING_TAIL);
  if (tail === head) return null;
  const kind = ringWord(ring, tail);
  const length = ringWord(ring, tail + 4);
  const payload = copyOut(ring, (tail + HEADER_BYTES) % ring.capacity, new Uint8Array(length));
  Atomics.store(ring.words, RING_TAIL, (tail + HEADER_BYTES + padded(length)) % ring.capacity);
  return { kind, payload };
}

// One record's bytes: `words` (Uint32) then `data`.
export function encodeRecord(kind, words, data = new Uint8Array(0)) {
  const length = words.length * 4 + data.length;
  const bytes = new Uint8Array(HEADER_BYTES + padded(length));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, kind, true);
  view.setUint32(4, length, true);
  words.forEach((word, index) => view.setUint32(HEADER_BYTES + index * 4, word >>> 0, true));
  bytes.set(data, HEADER_BYTES + words.length * 4);
  return bytes;
}

// The records in `bytes`, as { kind, payload } views.
export function* splitRecords(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 0;
  while (at + HEADER_BYTES <= bytes.length) {
    const kind = view.getUint32(at, true);
    const length = view.getUint32(at + 4, true);
    yield { kind, payload: bytes.subarray(at + HEADER_BYTES, at + HEADER_BYTES + length) };
    at += HEADER_BYTES + padded(length);
  }
}

// virgl stream commands (virglrenderer's src/virgl_protocol.h). A command is a header dword,
// cmd | object << 8 | length << 16, then `length` dwords.
const CCMD = {
  CREATE_OBJECT: 1,
  DESTROY_OBJECT: 3,
  SET_FRAMEBUFFER_STATE: 5,
  CLEAR: 7,
};
const CCMD_NAMES = [
  "NOP", "CREATE_OBJECT", "BIND_OBJECT", "DESTROY_OBJECT", "SET_VIEWPORT_STATE", "SET_FRAMEBUFFER_STATE",
  "SET_VERTEX_BUFFERS", "CLEAR", "DRAW_VBO", "RESOURCE_INLINE_WRITE", "SET_SAMPLER_VIEWS", "SET_INDEX_BUFFER",
  "SET_CONSTANT_BUFFER", "SET_STENCIL_REF", "SET_BLEND_COLOR", "SET_SCISSOR_STATE", "BLIT", "RESOURCE_COPY_REGION",
  "BIND_SAMPLER_STATES", "BEGIN_QUERY", "END_QUERY", "GET_QUERY_RESULT", "SET_POLYGON_STIPPLE", "SET_CLIP_STATE",
  "SET_SAMPLE_MASK", "SET_STREAMOUT_TARGETS", "SET_RENDER_CONDITION", "SET_UNIFORM_BUFFER", "SET_SUB_CTX",
  "CREATE_SUB_CTX", "DESTROY_SUB_CTX", "BIND_SHADER", "SET_TESS_STATE", "SET_MIN_SAMPLES", "SET_SHADER_BUFFERS",
  "SET_SHADER_IMAGES", "MEMORY_BARRIER", "LAUNCH_GRID", "SET_FRAMEBUFFER_STATE_NO_ATTACH", "TEXTURE_BARRIER",
  "SET_ATOMIC_BUFFERS", "SET_DEBUG_FLAGS", "GET_QUERY_RESULT_QBO", "TRANSFER3D", "END_TRANSFERS", "COPY_TRANSFER3D",
  "SET_TWEAKS", "CLEAR_TEXTURE", "PIPE_RESOURCE_CREATE", "PIPE_RESOURCE_SET_TYPE", "GET_MEMORY_INFO",
  "SEND_STRING_MARKER", "LINK_SHADER",
];
const OBJECT_SURFACE = 8;
// PIPE_CLEAR_COLOR0; colour buffer n is this shifted left by n.
const CLEAR_COLOR0 = 1 << 2;
// VIRTIO_GPU_RESOURCE_FLAG_Y_0_TOP: row 0 of the resource is the top of the picture.
export const RESOURCE_Y_0_TOP = 1;

// Acts on records. `backend` draws: resource(r), destroyResource(id), clear(id, rgba),
// scanout(resource or null, rect), flush(resource, rect), reset(). `respond(kind, words, data)`
// queues a completion. `log(text)` hears what is decoded and what is skipped, once per kind.
export class Renderer {
  constructor(backend, respond, log = () => {}) {
    this.backend = backend;
    this.respond = respond;
    this.log = log;
    this.resources = new Map();
    // Per context: surfaces by handle (each { resource, format, level }) and the colour buffers
    // SET_FRAMEBUFFER_STATE bound. Sub-contexts share them here (M0).
    this.contexts = new Map();
    this.scanoutResource = 0;
    this.counts = new Map();
    this.clears = 0;
  }

  count(name) {
    const seen = this.counts.get(name) ?? 0;
    if (seen === 0) this.log(`first ${name}`);
    this.counts.set(name, seen + 1);
  }

  handle(kind, payload) {
    // A view on the module's memory (Node) may sit at any byte; Uint32Array needs four.
    if (payload.byteOffset % 4) payload = payload.slice();
    const words = new Uint32Array(payload.buffer, payload.byteOffset, payload.length >>> 2);
    this.count(RECORD_NAMES[kind] ?? `record ${kind}`);
    switch (kind) {
      case RECORD.RESET:
        this.resources.clear();
        this.contexts.clear();
        this.scanoutResource = 0;
        this.backend.reset();
        break;
      case RECORD.CREATE_CONTEXT:
        this.contexts.set(words[0], { surfaces: new Map(), colorBuffers: [] });
        this.log(`context ${words[0]}: ${new TextDecoder().decode(payload.subarray(4))}`);
        break;
      case RECORD.DESTROY_CONTEXT:
        this.contexts.delete(words[0]);
        break;
      case RECORD.CREATE_RESOURCE: {
        const [id, target, format, bind, width, height, depth, arraySize, lastLevel, samples, flags] = words;
        const resource = { id, target, format, bind, width, height, depth, arraySize, lastLevel, samples, flags };
        this.resources.set(id, resource);
        this.backend.resource(resource);
        break;
      }
      case RECORD.DESTROY_RESOURCE:
        this.resources.delete(words[0]);
        this.backend.destroyResource(words[0]);
        break;
      case RECORD.READ_RESOURCE:
        // M0 has no pixels to give back: the device leaves the guest's backing as it was.
        this.respond(COMPLETION.READ, [words[9], words[10]]);
        break;
      case RECORD.SUBMIT:
        this.submit(words[0], words.subarray(1));
        break;
      case RECORD.SET_SCANOUT: {
        const [, resource, x, y, width, height] = words;
        this.scanoutResource = resource;
        this.backend.scanout(this.resources.get(resource) ?? null, { x, y, width, height });
        break;
      }
      case RECORD.FLUSH: {
        const [resource, x, y, width, height] = words;
        const flushed = this.resources.get(resource);
        if (flushed && resource === this.scanoutResource) this.backend.flush(flushed, { x, y, width, height });
        break;
      }
      case RECORD.FENCE:
        // Everything before it has been handed to GL, in order, which is all M0 promises.
        this.respond(COMPLETION.FENCE, [words[0], words[1]]);
        break;
      default:
        break;
    }
  }

  submit(contextId, stream) {
    const context = this.contexts.get(contextId);
    let at = 0;
    while (at < stream.length) {
      const header = stream[at];
      const command = header & 0xff;
      const object = (header >>> 8) & 0xff;
      const length = header >>> 16;
      const body = stream.subarray(at + 1, Math.min(stream.length, at + 1 + length));
      at += 1 + length;
      this.count(`virgl ${CCMD_NAMES[command] ?? command}${command === CCMD.CREATE_OBJECT ? ` ${object}` : ""}`);
      if (!context) continue;
      if (command === CCMD.CREATE_OBJECT && object === OBJECT_SURFACE && body.length >= 3) {
        context.surfaces.set(body[0], { resource: body[1], format: body[2], level: body[3] ?? 0 });
      } else if (command === CCMD.DESTROY_OBJECT && object === OBJECT_SURFACE && body.length >= 1) {
        context.surfaces.delete(body[0]);
      } else if (command === CCMD.SET_FRAMEBUFFER_STATE && body.length >= 2) {
        context.colorBuffers = [...body.subarray(2, 2 + body[0])];
      } else if (command === CCMD.CLEAR && body.length >= 5) {
        const color = new Float32Array(body.buffer, body.byteOffset + 4, 4);
        context.colorBuffers.forEach((handle, index) => {
          const surface = context.surfaces.get(handle);
          const resource = surface && this.resources.get(surface.resource);
          if (!resource || !(body[0] & (CLEAR_COLOR0 << index))) return;
          this.clears++;
          this.backend.clear(resource, [...color]);
        });
      }
    }
  }
}
