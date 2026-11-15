// The probes that need a worker (docs/plans/28-apple.md, M0): synchronous XHR to the scheme
// handler, which is how worker.js reads disks and how a disk write would reach the app, and how
// far a wasm32 memory grows in this process.
const MIB = 1 << 20;
const PAGE = 64 * 1024;
const GROW_STEP_MIB = 128;
const WASM32_MAX_MIB = 4096;

function rangeRead(origin) {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", `${origin}/demo/vmlinuz-virt`, false);
  xhr.responseType = "arraybuffer";
  xhr.setRequestHeader("Range", "bytes=512-1023");
  xhr.send();
  return { status: xhr.status, bytes: xhr.response?.byteLength ?? 0, contentRange: xhr.getResponseHeader("Content-Range") };
}

function postBody(origin) {
  const body = new Uint8Array(MIB);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${origin}/echo`, false);
  const started = performance.now();
  xhr.send(body);
  return { status: xhr.status, sent: body.length, handlerSaw: Number(xhr.responseText), ms: performance.now() - started };
}

// Touches every new page so the memory is really committed, the way guest RAM is once the guest
// uses it; a process limit shows up here, not at `grow`.
function memoryCeiling() {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: WASM32_MAX_MIB * MIB / PAGE });
  let mib = 0;
  try {
    while (mib + GROW_STEP_MIB <= WASM32_MAX_MIB) {
      memory.grow(GROW_STEP_MIB * MIB / PAGE);
      const view = new Uint8Array(memory.buffer, mib * MIB, GROW_STEP_MIB * MIB);
      for (let at = 0; at < view.length; at += 4096) view[at] = 1;
      mib += GROW_STEP_MIB;
    }
    return { grewMib: mib, stoppedBy: "wasm32 limit" };
  } catch (error) {
    return { grewMib: mib, stoppedBy: String(error.message ?? error) };
  }
}

onmessage = ({ data }) => {
  const result = {};
  for (const [name, probe] of Object.entries({ rangeRead, postBody })) {
    try {
      result[name] = probe(data.origin);
    } catch (error) {
      result[name] = { error: String(error.message ?? error) };
    }
  }
  result.memory = memoryCeiling();
  postMessage(result);
};
