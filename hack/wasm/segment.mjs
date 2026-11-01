// Boots several machines of the browser build (target/web/uemulator_web.wasm) under Node on one
// shared Ethernet segment, the way several tabs of the page share one, and checks what the
// segment promises (docs/plans/13-web.md): each machine gets its own DHCP address in 10.0.2.0/24,
// they ping each other and talk TCP, and when the machine whose tab serves DHCP goes away another
// takes over without anyone losing an address.
//
//   WEB_DEMO=1 hack/web.sh
//   node hack/wasm/segment.mjs [--kernel K] [--initrd I] [--firmware none|uefi] [--memory MiB]
//                              [--no-jit] [--verbose] [--seconds N]
//
// The kernel and initramfs default to the page's Alpine demo (target/web/demo), whose /init loads
// the virtio-net driver and runs udhcpc; another initramfs has to do the same and print
// "eth0 is ADDRESS/24" for this to find the address. Each machine runs in a worker thread
// (hack/wasm/segment_machine.mjs) and gets the transport and election the page uses
// (www/segment.js: BroadcastChannel and a Web Lock, both in Node), so only the browser itself is
// missing. --verbose copies every console to stderr; otherwise a failure prints the tail of each.
import { Worker } from "node:worker_threads";
import { parseArgs } from "node:util";
import { CONTROL_BYTES, CTRL, NET_OFFSET, NET_RING_BYTES, RING_BYTES, RING_OFFSET } from "../../crates/uemulator-web/www/protocol.js";
import { joinSegment, randomMac, randomTab } from "../../crates/uemulator-web/www/segment.js";

const web = new URL("../../target/web/", import.meta.url);
const { values } = parseArgs({
  options: {
    kernel: { type: "string", default: new URL("demo/vmlinuz-virt", web).pathname },
    initrd: { type: "string", default: new URL("demo/initramfs.gz", web).pathname },
    firmware: { type: "string", default: "none" },
    memory: { type: "string", default: "256" },
    "no-jit": { type: "boolean", default: false },
    verbose: { type: "boolean", default: false },
    seconds: { type: "string", default: "120" },
  },
});
const STEP_TIMEOUT_MS = Number(values.seconds) * 1000;
const CONSOLE_TAIL = 3000;
// The shell asks the terminal for the cursor position after its prompt (ESC [6n).
const PROMPT = /~# (\x1b\[6n)?$/;
// What the demo's /init prints once udhcpc has configured eth0.
const ADDRESS_LINE = /eth0 is (10\.0\.2\.\d+)\/24/;
const TCP_PORT = 5000;
const started = performance.now();

function log(text) {
  console.log(`[${((performance.now() - started) / 1000).toFixed(1).padStart(6)}s] ${text}`);
}

class Machine {
  constructor(name) {
    this.name = name;
    this.console = "";
    this.watchers = new Set();
    this.status = null;
    this.shared = new SharedArrayBuffer(CONTROL_BYTES);
    this.control = new Int32Array(this.shared, 0, RING_OFFSET / 4);
    this.keys = new Uint8Array(this.shared, RING_OFFSET, RING_BYTES);
    const tab = randomTab();
    this.segment = joinSegment({
      control: this.control,
      ring: new Uint8Array(this.shared, NET_OFFSET, NET_RING_BYTES),
      tab,
      onServing: () => log(`${name} now serves the segment (holds the lock)`),
    });
    this.worker = new Worker(new URL("segment_machine.mjs", import.meta.url), {
      workerData: {
        wasmPath: new URL("uemulator_web.wasm", web).pathname,
        kernelPath: values.kernel,
        initrdPath: values.initrd,
        cmdline: "quiet",
        memoryMib: Number(values.memory),
        firmware: values.firmware,
        jit: !values["no-jit"],
        control: this.shared,
        network: { mac: randomMac().buffer, tab },
      },
    });
    this.exited = new Promise((resolve) => {
      this.worker.on("message", (message) => {
        if (message.type === "output") {
          this.console += message.text;
          if (values.verbose) process.stderr.write(message.text.replace(/^/gm, `${name}| `));
          for (const watcher of this.watchers) watcher();
        } else if (message.type === "network") {
          this.status = message;
        } else if (message.type === "log") {
          if (values.verbose) process.stderr.write(`${name} log: ${message.text}\n`);
        } else if (message.type === "exit") {
          resolve(message);
        }
      });
      this.worker.on("error", (error) => resolve({ code: -3, message: String(error?.stack ?? error) }));
    });
  }

  // Waits until the console after `from` matches `pattern`; returns the match.
  waitFor(pattern, from = 0, what = String(pattern)) {
    return new Promise((resolve, reject) => {
      const check = () => {
        const match = this.console.slice(from).match(pattern);
        if (!match) return false;
        this.watchers.delete(check);
        clearTimeout(timer);
        resolve(match);
        return true;
      };
      const timer = setTimeout(() => {
        this.watchers.delete(check);
        reject(new Error(`${this.name}: no ${what} within ${STEP_TIMEOUT_MS / 1000} s`));
      }, STEP_TIMEOUT_MS);
      if (!check()) this.watchers.add(check);
    });
  }

  type(text) {
    let head = Atomics.load(this.control, CTRL.HEAD);
    for (const byte of new TextEncoder().encode(text)) {
      this.keys[head] = byte;
      head = (head + 1) % RING_BYTES;
    }
    Atomics.store(this.control, CTRL.HEAD, head);
    Atomics.notify(this.control, CTRL.HEAD);
  }

  // Types a command at the shell prompt and waits for `pattern` in what follows it, then for
  // the next prompt.
  async run(command, pattern) {
    const from = this.console.length;
    this.type(`${command}\n`);
    const match = await this.waitFor(pattern, from, `${JSON.stringify(String(pattern))} after \`${command}\``);
    await this.waitFor(PROMPT, from, `prompt after \`${command}\``);
    return match;
  }

  async boot() {
    const [, address] = await this.waitFor(ADDRESS_LINE, 0, "DHCP address");
    this.address = address;
    await this.waitFor(PROMPT, 0, "shell prompt");
    log(`${this.name} booted with ${address}`);
    return address;
  }

  // Stops the machine, then (as the page does) leaves the segment, which hands the lock on.
  async stop() {
    Atomics.store(this.control, CTRL.STOP, 1);
    Atomics.notify(this.control, CTRL.HEAD);
    await this.exited;
    await this.worker.terminate();
    this.segment.leave();
  }
}

const machines = [];
function start(name) {
  const machine = new Machine(name);
  machines.push(machine);
  return machine;
}

function check(condition, text) {
  if (!condition) throw new Error(`check failed: ${text}`);
  log(`ok: ${text}`);
}

async function ping(from, to) {
  await from.run(`ping -c 3 -W 5 ${to.address}`, /3 packets transmitted, 3 packets received/);
  log(`ok: ${from.name} (${from.address}) pinged ${to.name} (${to.address})`);
}

async function scenario() {
  // A asks for the lock first and serves; B and C queue for it in that order.
  const a = start("A");
  const b = start("B");
  const c = start("C");
  await Promise.all([a.boot(), b.boot(), c.boot()]);
  const first = [a, b, c].map((m) => m.address);
  check(new Set(first).size === 3, `three different addresses: ${first.join(", ")}`);
  check(first.every((address) => address.startsWith("10.0.2.")), "all in 10.0.2.0/24");
  check(a.status?.serving && !b.status?.serving && !c.status?.serving, "A serves, B and C do not");
  check([a, b, c].every((m) => m.status?.peers === 2), "every machine counts two peers");

  await ping(b, a);
  await ping(c, b);
  await ping(a, c);
  await b.run("ping -c 1 -W 5 10.0.2.2", /1 packets received/);
  log("ok: B pinged the segment's server address 10.0.2.2");

  // TCP: A listens, B connects and sends a line, A prints it.
  const listening = a.console.length;
  a.type(`nc -l -p ${TCP_PORT}\n`);
  await new Promise((resolve) => setTimeout(resolve, 1000));
  const marker = `tcp-from-${b.name}-${Date.now()}`;
  b.type(`echo ${marker} | nc ${a.address} ${TCP_PORT}\n`);
  await a.waitFor(new RegExp(marker), listening + 1, "TCP payload on A");
  log(`ok: ${b.name} sent ${marker} over TCP to ${a.name}'s nc`);
  // Ctrl-C ends whichever nc is still running.
  a.type("\x03");
  b.type("\x03");
  await a.waitFor(PROMPT, a.console.length - 1, "A prompt after nc");

  log("stopping A, the server");
  await a.stop();
  await waitUntil(() => b.status?.serving, "B to take over serving");
  check(b.status.serving, "B serves after A left");

  // C asks for an address again: the new server knows the one C has, from the announcements.
  const before = c.address;
  const [, again] = await c.run("udhcpc -i eth0 -q -n -t 5 -T 2 >/dev/null 2>&1; ip -4 -o addr show eth0", /inet (10\.0\.2\.\d+)\//);
  check(again === before, `C keeps ${before} when it asks the new server (got ${again})`);

  const d = start("D");
  await d.boot();
  check(![a, b, c].map((m) => m.address).includes(d.address), `D's ${d.address} differs from A's, B's and C's`);
  await ping(d, c);
  await ping(b, d);
  await waitUntil(() => [b, c, d].every((m) => m.status?.peers === 2), "the peer counts to drop A");
  check(true, "B, C and D count two peers each once A is gone");
}

function waitUntil(condition, what) {
  return new Promise((resolve, reject) => {
    const deadline = performance.now() + STEP_TIMEOUT_MS;
    const poll = setInterval(() => {
      if (condition()) {
        clearInterval(poll);
        resolve();
      } else if (performance.now() > deadline) {
        clearInterval(poll);
        reject(new Error(`timed out waiting for ${what}`));
      }
    }, 100);
  });
}

let failed = false;
try {
  await scenario();
  log("all checks passed");
} catch (error) {
  failed = true;
  console.error(String(error?.stack ?? error));
  if (!values.verbose) {
    for (const machine of machines) {
      console.error(`--- ${machine.name} console (tail) ---\n${machine.console.slice(-CONSOLE_TAIL)}`);
    }
  }
} finally {
  await Promise.all(machines.map((machine) => machine.stop().catch(() => {})));
}
process.exit(failed ? 1 : 0);
