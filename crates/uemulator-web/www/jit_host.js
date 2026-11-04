// The host half of the library's WebAssembly JIT (crates/uemulator/src/jit/wasm.rs). The library
// generates modules of one or more block functions and hands the bytes to `host_jit_compile`,
// which instantiates them against the emulator's own memory and function table and returns the
// table slots the functions now occupy; Rust calls a block through its slot as a plain function
// pointer.
//
// With a compile worker (jit_compiler.js, sharing a compile queue with us) the library first
// hands a module to `host_jit_submit` and keeps interpreting the block until `host_jit_ready`
// says the worker is done. The worker compiled the very same bytes, so V8 finds them in its
// process-wide native module cache when `host_jit_compile` makes the module here, compiled code
// included: the emulator's thread pays a few microseconds instead of ~38 a block
// (docs/plans/13-web.md). Engines without such a cache would compile everything twice, so the
// queue is only used on V8.
//
// Also loaded by hack/wasm/run-wasi.mjs, so it must not touch anything a worker has and Node lacks.

// Slots appended to the table at a time. Growing is cheap but not free, and a boot adds thousands.
const TABLE_GROW_STEP = 4096;
const COMPILE_OK = 0;
const COMPILE_FAILED = -1;
const NO_TICKET = -1;

// The compile queue: a SharedArrayBuffer of `QUEUE_HEADER_WORDS` Int32 words, then a byte ring
// of modules the worker reads in order. Each record is its length (an Int32) and its bytes,
// padded to 4; a length of `QUEUE_WRAP` sends the reader back to the ring's start. The writer
// never fills the ring completely (`WRITE == READ` means empty), and a module that does not fit
// is compiled by the emulator's thread instead.
export const QUEUE = {
  // Ring offset where the next record goes; the worker waits on this word.
  WRITE: 0,
  // Ring offset of the next record the worker reads.
  READ: 1,
  // Set by the worker once it runs: until then nothing is handed to it.
  STARTED: 2,
  // Modules the worker has finished, in order: ticket t is done once this exceeds t.
  FINISHED: 3,
};
export const QUEUE_HEADER_WORDS = 8;
export const QUEUE_WRAP = -1;
const QUEUE_RING_BYTES = 8 << 20;
const RECORD_HEADER_BYTES = 4;
// A ticket the worker has not finished after this long is reported ready anyway: the emulator's
// thread then compiles the module itself (or waits for V8 to finish the worker's compile of it)
// rather than interpret the block indefinitely.
const READY_TIMEOUT_MS = 250;
const TICKET_MASK = 0x7fffffff;
// When each recent ticket was handed out, by ticket modulo this. The library has at most a few
// tickets out at a time (`MAX_IN_FLIGHT` in jit/wasm.rs), so older ones are never asked about.
const SUBMIT_TIMES = 1024;

export const alignRecord = (length) => (length + 3) & ~3;

export function createCompileQueue() {
  return new SharedArrayBuffer(QUEUE_HEADER_WORDS * 4 + QUEUE_RING_BYTES);
}

// Whether a compile worker pays here: V8 shares compiled modules across threads by their bytes.
export function compileWorkerHelps() {
  return Boolean(globalThis.process?.versions?.v8) || /\bChrom(e|ium)\//.test(globalThis.navigator?.userAgent ?? "");
}

export function createJitHost() {
  let exports = null;
  // Slots handed out so far, and the ones given back. The first slots of the table are the
  // function pointers the Rust module put there itself; blocks start after them.
  let nextSlot = 0;
  const freeSlots = [];
  let queue = null;

  function allocateSlot(table) {
    if (nextSlot === 0) nextSlot = table.length;
    let slot = freeSlots.pop();
    if (slot === undefined) {
      if (nextSlot >= table.length) table.grow(TABLE_GROW_STEP);
      slot = nextSlot++;
    }
    return slot;
  }

  const host = {
    // `ptr`/`len` locate the module's bytes in the emulator's memory; its exports "0" to
    // `count - 1` go into table slots whose numbers are written to `slotsPtr` (u32 each).
    // Returns 0, or -1 if the engine rejected the module (a bug in the generator; the library
    // then interprets those blocks).
    host_jit_compile(ptr, len, slotsPtr, count) {
      const { memory, __indirect_function_table: table } = exports;
      try {
        // Copied out: the memory may grow (and move its buffer) before the engine is done with it,
        // and a view on shared memory is not accepted everywhere.
        // `>>> 0`: a pointer past 2 GiB of module memory arrives as a negative i32.
        const bytes = new Uint8Array(memory.buffer, ptr >>> 0, len).slice();
        const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
          env: { memory, table },
        });
        const slots = new Uint32Array(memory.buffer, slotsPtr >>> 0, count);
        for (let index = 0; index < count; index++) {
          const slot = allocateSlot(table);
          table.set(slot, instance.exports[index]);
          slots[index] = slot;
        }
        return COMPILE_OK;
      } catch (error) {
        host.lastError = error;
        return COMPILE_FAILED;
      }
    },
    // The library dropped a block: empty its slot so the engine can free the instance (once
    // every block of its module is gone), and hand the slot out again.
    host_jit_release(slot) {
      exports.__indirect_function_table.set(slot, null);
      freeSlots.push(slot);
    },
    // Copies the module into the compile queue. Returns its ticket, or -1 when there is no
    // worker (yet) or no room: the library then compiles it at once.
    host_jit_submit(ptr, len) {
      if (!queue || !Atomics.load(queue.header, QUEUE.STARTED)) return NO_TICKET;
      const record = RECORD_HEADER_BYTES + alignRecord(len);
      const size = queue.ring.length;
      const read = Atomics.load(queue.header, QUEUE.READ);
      let write = queue.write;
      // Leave the last 4 bytes before `read` free, so a full ring never looks empty.
      if (write >= read) {
        if (write + record > size - (read === 0 ? RECORD_HEADER_BYTES : 0)) {
          if (record >= read) return NO_TICKET;
          queue.view.setInt32(write, QUEUE_WRAP, true);
          write = 0;
        }
      } else if (write + record >= read) {
        return NO_TICKET;
      }
      queue.view.setInt32(write, len, true);
      queue.ring.set(new Uint8Array(exports.memory.buffer, ptr >>> 0, len), write + RECORD_HEADER_BYTES);
      write = (write + record) % size;
      queue.write = write;
      Atomics.store(queue.header, QUEUE.WRITE, write);
      Atomics.notify(queue.header, QUEUE.WRITE);
      const ticket = queue.submitted;
      queue.submitted = (queue.submitted + 1) & TICKET_MASK;
      queue.submittedAt[ticket % SUBMIT_TIMES] = performance.now();
      return ticket;
    },
    host_jit_ready(ticket) {
      if (!queue) return 1;
      const finished = Atomics.load(queue.header, QUEUE.FINISHED);
      // Both count up from 0 and wrap together; a ticket is done once it is behind `finished`.
      if (((finished - ticket) & TICKET_MASK) - 1 < TICKET_MASK >>> 1) return 1;
      return performance.now() - queue.submittedAt[ticket % SUBMIT_TIMES] > READY_TIMEOUT_MS ? 1 : 0;
    },
  };

  return {
    host,
    // Call with the emulator's instance exports before anything runs.
    attach(instanceExports) {
      exports = instanceExports;
    },
    // Hand modules to the compile worker reading `buffer` (`createCompileQueue`) from now on.
    useCompileQueue(buffer) {
      if (!buffer || !compileWorkerHelps()) return;
      const header = new Int32Array(buffer, 0, QUEUE_HEADER_WORDS);
      const ring = new Uint8Array(buffer, QUEUE_HEADER_WORDS * 4);
      queue = {
        header,
        ring,
        view: new DataView(buffer, QUEUE_HEADER_WORDS * 4),
        write: Atomics.load(header, QUEUE.WRITE),
        submitted: Atomics.load(header, QUEUE.FINISHED),
        submittedAt: new Float64Array(SUBMIT_TIMES),
      };
    },
  };
}
