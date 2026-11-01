// The host half of the library's WebAssembly JIT (crates/uemulator/src/jit/wasm.rs). The library
// generates one small module per block and hands the bytes to `host_jit_compile`, which
// instantiates it against the emulator's own memory and function table and returns the table slot
// the block's function now occupies; Rust calls it through that slot as a plain function pointer.
//
// Also loaded by hack/wasm/run-wasi.mjs, so it must not touch anything a worker has and Node lacks.

// Slots appended to the table at a time. Growing is cheap but not free, and a boot adds thousands.
const TABLE_GROW_STEP = 4096;
const COMPILE_FAILED = -1;

export function createJitHost() {
  let exports = null;
  // Slots handed out so far, and the ones given back. The first slots of the table are the
  // function pointers the Rust module put there itself; blocks start after them.
  let nextSlot = 0;
  const freeSlots = [];

  const host = {
    // `ptr`/`len` locate the module's bytes in the emulator's memory. Returns the table slot, or -1
    // if the engine rejected the module (a bug in the generator; the library then interprets).
    host_jit_compile(ptr, len) {
      const { memory, __indirect_function_table: table } = exports;
      try {
        // Copied out: the memory may grow (and move its buffer) before the engine is done with it,
        // and a view on shared memory is not accepted everywhere.
        // `>>> 0`: a pointer past 2 GiB of module memory arrives as a negative i32.
        const bytes = new Uint8Array(memory.buffer, ptr >>> 0, len).slice();
        const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
          env: { memory, table },
        });
        if (nextSlot === 0) nextSlot = table.length;
        let slot = freeSlots.pop();
        if (slot === undefined) {
          if (nextSlot >= table.length) table.grow(TABLE_GROW_STEP);
          slot = nextSlot++;
        }
        table.set(slot, instance.exports.block);
        return slot;
      } catch (error) {
        host.lastError = error;
        return COMPILE_FAILED;
      }
    },
    // The library dropped a block: empty its slot so the engine can free the instance, and hand
    // the slot out again.
    host_jit_release(slot) {
      exports.__indirect_function_table.set(slot, null);
      freeSlots.push(slot);
    },
  };

  return {
    host,
    // Call with the emulator's instance exports before anything runs.
    attach(instanceExports) {
      exports = instanceExports;
    },
  };
}
