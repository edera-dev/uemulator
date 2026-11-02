// The JIT's compile worker (crates/uemulator-web/www/jit_compiler.js) as a Node worker thread:
// `new Worker(this file, { workerData: queue })`, the queue from `createCompileQueue`.
import { workerData } from "node:worker_threads";
import { runJitCompiler } from "../../crates/uemulator-web/www/jit_compiler.js";

runJitCompiler(workerData);
