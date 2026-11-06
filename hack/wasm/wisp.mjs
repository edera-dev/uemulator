// Checks the guest's internet through a Wisp server end to end (docs/plans/27-wisp.md): one
// machine of the browser build boots the page's Alpine demo under Node with the WebSocket kept by
// www/wisp.js as the page keeps it (hack/wasm/segment.mjs), then in the guest `apk update`,
// `apk add jq` and runs it, downloads the demo's own kernel (12 MB) and checks its SHA-256 against
// the copy hack/web/demo.py cached, and finally has the Wisp server go away in the middle of a
// second download: that `wget` has to fail, the page side has to reconnect, and `apk update` has
// to work again.
//
//   WEB_DEMO=1 hack/web.sh
//   node hack/wasm/wisp.mjs [--wisp URL] [--verbose] [--seconds N]
//
// Without --wisp it runs hack/web/serve.py (its /wisp/ endpoint) on a free port and restarts it
// for the drop. With --wisp it uses that server and skips the drop, which needs the server in
// hand. The guest fetches from Alpine's mirrors, so this needs the network.
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { parseArgs } from "node:util";
import { WISP_STATE } from "../../crates/uemulator-web/www/protocol.js";
import { CONSOLE_TAIL, DEMO, Machine, log, waitUntil } from "./segment.mjs";

const root = new URL("../../", import.meta.url);
const { values } = parseArgs({
  options: {
    wisp: { type: "string" },
    verbose: { type: "boolean", default: false },
    seconds: { type: "string", default: "300" },
  },
});
const timeoutMs = Number(values.seconds) * 1000;
// apk's index and the 12 MB download land in the guest's RAM (tmpfs).
const MEMORY_MIB = 512;
const CACHED_KERNEL = new URL("target/web-demo-cache/vmlinuz-virt", root).pathname;
// The drop lands this far into the second download's bytes from the server, well short of its
// end, so the stream is still open on the server when the socket goes.
const DROP_AFTER_BYTES = 1 << 20;
const MIB = 1 << 20;
const started = performance.now();

if (!existsSync(CACHED_KERNEL)) {
  console.error(`${CACHED_KERNEL} is missing: run \`WEB_DEMO=1 hack/web.sh\` first (it downloads the demo into target/web-demo-cache).`);
  process.exit(1);
}
const cached = readFileSync(CACHED_KERNEL);
const expectedSha256 = createHash("sha256").update(cached).digest("hex");
// demo.py's KERNEL, over plain http: busybox wget in the demo has no TLS. -B keeps the import from
// leaving a __pycache__ in hack/web.
const kernelUrl = execFileSync("python3", [
  "-I", "-B", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import demo; print(demo.KERNEL[0])",
  new URL("hack/web/", root).pathname,
], { encoding: "utf8" }).trim().replace(/^https:/, "http:");

// Bytes the Wisp server has sent over any socket, and a hook run on each message: how the drop is
// placed inside a download without anything test-only in wisp.js.
let receivedBytes = 0;
let onReceived = null;
const PlainWebSocket = globalThis.WebSocket;
globalThis.WebSocket = class CountingWebSocket extends PlainWebSocket {
  constructor(...args) {
    super(...args);
    this.addEventListener("message", ({ data }) => {
      receivedBytes += data.byteLength ?? 0;
      onReceived?.();
    });
  }
};

async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

// serve.py prints its URL once it listens. -u, or the line would sit in Python's buffer.
async function startServer(port) {
  const child = spawn("python3", ["-u", new URL("hack/web/serve.py", root).pathname, new URL("target/web", root).pathname, String(port)], {
    stdio: ["ignore", "pipe", values.verbose ? "inherit" : "ignore"],
  });
  let output = "";
  child.stdout.setEncoding("utf8");
  await new Promise((resolve, reject) => {
    child.stdout.on("data", (text) => {
      output += text;
      if (output.includes("http://")) resolve();
    });
    child.on("exit", (code) => reject(new Error(`serve.py exited with ${code} before listening: ${output}`)));
  });
  return child;
}

function check(condition, text) {
  if (!condition) throw new Error(`check failed: ${text}`);
  log(`ok: ${text}`);
}

const seconds = (since) => (performance.now() - since) / 1000;
let server = null;
let machine = null;
const summary = [];

async function scenario() {
  let url = values.wisp;
  let port;
  if (url === undefined) {
    port = await freePort();
    server = await startServer(port);
    url = `ws://127.0.0.1:${port}/wisp/`;
    log(`serve.py listens on ${port}`);
  }
  machine = new Machine("A", {
    ...DEMO,
    firmware: "none",
    memoryMib: MEMORY_MIB,
    jit: true,
    verbose: values.verbose,
    timeoutMs,
    wisp: url,
  });
  await machine.boot();
  await waitUntil(() => machine.wispState === WISP_STATE.READY, "the Wisp handshake", timeoutMs);
  log(`Wisp client ready on ${url}`);

  let begun = performance.now();
  const [, packages] = await machine.run("apk update", /OK: (\d+) distinct packages available/);
  summary.push(`apk update ${seconds(begun).toFixed(1)} s`);
  check(true, `apk update: ${packages} packages in ${seconds(begun).toFixed(1)} s`);

  begun = performance.now();
  const [, jq] = await machine.run("apk add --no-cache jq >/dev/null && jq --version", /(jq-\d[\w.-]*)\r?\n/);
  summary.push(`${jq} ${seconds(begun).toFixed(1)} s`);
  check(true, `apk add jq and run it: ${jq} in ${seconds(begun).toFixed(1)} s`);

  begun = performance.now();
  const [, status] = await machine.run(`wget -q -O /tmp/big ${kernelUrl}; echo wget-exit=$?`, /wget-exit=(\d+)/);
  const downloadSeconds = seconds(begun);
  check(status === "0", `wget of ${kernelUrl} exits 0`);
  const [, sha256] = await machine.run("sha256sum /tmp/big", /([0-9a-f]{64}) {2}\/tmp\/big/);
  const throughput = cached.length / MIB / downloadSeconds;
  check(sha256 === expectedSha256, `${(cached.length / MIB).toFixed(1)} MiB in ${downloadSeconds.toFixed(1)} s (${throughput.toFixed(2)} MiB/s), SHA-256 matches the cached copy`);
  summary.push(`${(cached.length / MIB).toFixed(1)} MiB download ${downloadSeconds.toFixed(1)} s (${throughput.toFixed(2)} MiB/s), SHA-256 ok`);

  if (server === null) {
    summary.push("drop skipped (--wisp)");
    log("--wisp given: skipping the drop, which restarts serve.py");
    return;
  }

  // Kills serve.py once DROP_AFTER_BYTES of the second download have come in, then starts it again
  // on the same port for wisp.js to find when it reconnects.
  const readyBefore = machine.wispReadyCount;
  const mark = receivedBytes;
  let dropped = null;
  let serverExited = null;
  onReceived = () => {
    if (receivedBytes - mark < DROP_AFTER_BYTES) return;
    onReceived = null;
    serverExited = once(server, "exit");
    server.kill("SIGKILL");
    dropped = performance.now();
    log(`killed serve.py ${((receivedBytes - mark) / MIB).toFixed(1)} MiB into the second download`);
  };
  const from = machine.console.length;
  machine.type(`wget -q -O /tmp/big2 ${kernelUrl}; echo wget-exit=$?; wc -c < /tmp/big2\n`);
  await waitUntil(() => dropped !== null, "the second download to get going", timeoutMs);
  await serverExited;
  server = await startServer(port);
  const [, dropStatus, got] = await machine.waitFor(/wget-exit=(\d+)\r?\n(\d+)\r?\n/, from, "the second wget's exit");
  check(dropStatus !== "0", `the second wget fails (exit ${dropStatus}) with ${(Number(got) / MIB).toFixed(1)} of ${(cached.length / MIB).toFixed(1)} MiB`);
  await waitUntil(() => machine.wispReadyCount > readyBefore, "the Wisp client to reconnect", timeoutMs);
  const reconnectSeconds = seconds(dropped);
  check(true, `the Wisp client is ready again ${reconnectSeconds.toFixed(1)} s after the drop`);
  begun = performance.now();
  await machine.run("apk update", /OK: \d+ distinct packages available/);
  check(true, `apk update works again (${seconds(begun).toFixed(1)} s)`);
  summary.push(`drop: wget exit ${dropStatus} at ${(Number(got) / MIB).toFixed(1)} MiB, ready again in ${reconnectSeconds.toFixed(1)} s, apk update ${seconds(begun).toFixed(1)} s`);
}

let failed = false;
try {
  await scenario();
  console.log(`PASS in ${seconds(started).toFixed(0)} s: ${summary.join("; ")}`);
} catch (error) {
  failed = true;
  console.error(String(error?.stack ?? error));
  if (machine && !values.verbose) console.error(`--- console (tail) ---\n${machine.console.slice(-CONSOLE_TAIL)}`);
} finally {
  onReceived = null;
  await machine?.stop().catch(() => {});
  server?.kill("SIGKILL");
}
process.exit(failed ? 1 : 0);
