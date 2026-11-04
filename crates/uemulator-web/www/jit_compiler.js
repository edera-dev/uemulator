// The compile worker of the library's WebAssembly JIT (see jit_host.js): takes block modules
// from the compile queue in order, compiles each one and keeps it alive for a while, so that the
// emulator's thread, making a module of the same bytes, gets V8's compiled code from its native
// module cache instead of compiling again.
//
// The page starts it (main.js): a worker started by the emulator's worker never runs in Chrome,
// whose thread is busy in `uw_run` from the start. hack/wasm/jit_compiler.mjs runs it under Node.

import { QUEUE, QUEUE_HEADER_WORDS, QUEUE_WRAP, alignRecord } from "./jit_host.js";

const RECORD_HEADER_BYTES = 4;
// Modules kept alive after they are compiled. The cache holds a module only while someone does;
// the emulator's thread has at most `MAX_IN_FLIGHT` (jit/wasm.rs, 1) of ours out at a time, and
// past this many behind it would just compile them itself. More would only keep dead code alive.
const KEEP_MODULES = 16;
// Every way through a block reads or writes its `Cpu` argument, or calls a helper through the
// table, before it can return. This pointer lies past the end of the one-page memory the modules
// are instantiated against here, and the table is empty, so a call traps early, after V8 has
// compiled the function (it compiles lazily, on the first call). Block code has no loops.
const UNMAPPED_CPU = 0x7fff0000;

export function runJitCompiler(buffer) {
  // Every compile ends in a trap, and an error's stack trace was a third of what a module cost
  // this thread. Nothing here reads them.
  Error.stackTraceLimit = 0;
  const header = new Int32Array(buffer, 0, QUEUE_HEADER_WORDS);
  const ring = new Uint8Array(buffer, QUEUE_HEADER_WORDS * 4);
  const view = new DataView(buffer, QUEUE_HEADER_WORDS * 4);
  const imports = {
    env: {
      memory: new WebAssembly.Memory({ initial: 1 }),
      table: new WebAssembly.Table({ initial: 1, element: "anyfunc" }),
    },
  };
  const kept = new Array(KEEP_MODULES);
  let compiled = 0;
  let read = Atomics.load(header, QUEUE.READ);
  Atomics.store(header, QUEUE.STARTED, 1);
  for (;;) {
    const write = Atomics.load(header, QUEUE.WRITE);
    if (read === write) {
      Atomics.wait(header, QUEUE.WRITE, write);
      continue;
    }
    const length = view.getInt32(read, true);
    if (length === QUEUE_WRAP) {
      read = 0;
      Atomics.store(header, QUEUE.READ, read);
      continue;
    }
    const bytes = ring.slice(read + RECORD_HEADER_BYTES, read + RECORD_HEADER_BYTES + length);
    read = (read + RECORD_HEADER_BYTES + alignRecord(length)) % ring.length;
    // The writer may reuse the space now: the bytes are copied.
    Atomics.store(header, QUEUE.READ, read);
    try {
      const module = new WebAssembly.Module(bytes);
      const instance = new WebAssembly.Instance(module, imports);
      for (const block of Object.values(instance.exports)) {
        try {
          block(UNMAPPED_CPU);
        } catch {
          // the trap that was the point of the call
        }
      }
      kept[compiled % KEEP_MODULES] = module;
    } catch {
      // The emulator's thread will find out for itself, and interpret these blocks.
    }
    compiled++;
    Atomics.add(header, QUEUE.FINISHED, 1);
  }
}

if (typeof WorkerGlobalScope !== "undefined" && globalThis instanceof WorkerGlobalScope) {
  onmessage = ({ data }) => runJitCompiler(data.queue);
}
