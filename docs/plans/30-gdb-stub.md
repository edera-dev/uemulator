# 30 — GDB remote stub

**Status:** implemented in the library and the CLI (`--gdb`, `--gdb-wait` on `advanced boot`, `quick boot` and the other commands that run a machine). Checked by unit and whole-machine tests, and by hand with lldb against an Alpine kernel (see "Checked"). Not wired into the web page or the Apple app.

## What it does

`--gdb [HOST:PORT]` (default `127.0.0.1:1234`, QEMU's `-s` port) serves the GDB remote serial protocol; `--gdb-wait` stops the machine before its first instruction until the debugger continues. gdb (`target remote :1234`) and lldb (`gdb-remote 1234`) can then:

- stop the machine (Ctrl-C) and continue it, single-step;
- set software and hardware breakpoints (`Z0`, `Z1`) and write, read and access watchpoints (`Z2`-`Z4`, up to 4 KiB each, any number);
- read and write registers (`g`, `G`, `p`, `P`) through a target description (`qXfer:features:read:target.xml`): gdb's amd64 core (general registers, rip, eflags, the six selectors, st0-st7 and the x87 control registers), SSE (xmm0-15, mxcsr), `org.gnu.gdb.i386.segments` (fs_base, gs_base) and `org.uemulator.x86_64.system` (k_gs_base, cr0, cr2, cr3, cr4, cr8, efer);
- read and write memory by guest-virtual address through the selected vCPU's page tables (`m`, `M`, `X`);
- see every vCPU as a thread, numbered from 1 (`qfThreadInfo`, `qC`, `qThreadExtraInfo`, `Hg`/`Hc`, `T`, `T` stop replies with `thread:`);
- detach (`D`, or closing the connection: breakpoints go and the machine runs on) and kill (`k`, `vKill`: the run ends with `ExitReason::Stopped`).

When the machine ends while a client waits for a stop, the client gets `W00`.

## Design

### Sans-I/O, engine on the stopped vCPU

`lib.rs` exports `GdbStub`; `MachineConfig::gdb_stub(halt_at_start)` gives a machine one and `EmulatedMachine::gdb_stub()` returns it. The embedder owns the connection: `connected()`, `from_client(bytes)`, `on_output(callback)`, `disconnected()`. The CLI's `serve_gdb` (`crates/uemulator-cli/src/run.rs`) is a TCP listener taking one client at a time, a blocking read loop into `from_client`, and an output callback that writes to the socket.

The protocol engine (`crates/uemulator/src/gdb/engine.rs`) does not run on the embedder's thread. It runs on the vCPU thread that stopped, inside `DebugLink::serve` (`gdb/mod.rs`), while every other vCPU is stopped too, because it needs `&mut` access to every `Cpu`: registers, memory through the page tables, the JIT to invalidate. The client's bytes wait in the session's inbox until then. While the machine runs, a client in all-stop mode only sends the 0x03 interrupt; `from_client` turns that into a pause request instead of queueing it. On connection the machine pauses, as QEMU's does, so the client's first queries get answered. Output is collected while the session is locked and handed to the callback after it is unlocked, so the callback may block on the socket.

A reply to a resume (`c`, `s`, `vCont`) comes later, from whichever vCPU stops next: the engine remembers that a stop reply is owed (`resume_pending`).

### Noticing a stop

Every vCPU has a `GdbVcpu` (`cpu/gdb.rs`) and a handle on the shared `DebugLink`. `Cpu::debugger_check` runs at the top of `before_step`, after the device tick and before NMIs and interrupts are delivered, so a step does not land in a handler and a held vCPU takes nothing. When nothing is armed it costs one relaxed atomic load per block boundary (the pause flag), and nothing at all on a machine without a stub. It stops for, in order:

1. a pause request (start, connection, Ctrl-C, a detach while running);
2. a watchpoint the last instruction hit;
3. a completed single step;
4. a hold (see below);
5. a breakpoint at the linear address of RIP (`fetch_linear`), unless the vCPU is halted.

Chained JIT blocks come back to `before_step` at least every `EVENT_INTERVAL` instructions, which bounds the latency of a pause. A halted vCPU on a thread polls the pause flag every millisecond while a debugger is attached (it would otherwise sleep 100 ms); one waiting for its start-up IPI leaves pauses to vCPUs that run, so the halt at start names vCPU 1.

### Breakpoints and watchpoints without touching the guest

Nothing is written into guest memory, so reads show the original bytes and self-modifying-code tracking is unaffected. While any breakpoint, watchpoint, step or hold is armed on a vCPU, `debug_slow_path` is true for it: every instruction runs through `step_debug`, the interpreter path the guest's own debug registers use (`docs/plans/21-ryzen-7000-parity.md`), and generated code never runs. `watch_access`, which sees every data access `step_debug`'s instruction makes, checks the debugger's watchpoints after the guest's DR0-DR3. A hit is a trap: the stop comes after the instruction (between iterations of a repeated string instruction). A faulting instruction reports none. Accesses made by event delivery are never watched, as for the guest's breakpoints.

The guest's DR0-DR7, DR6 status and DEBUGCTL are untouched: the debugger has its own lists, so a guest debugging itself works under a debugger.

**Cost:** one breakpoint makes the whole machine interpret instruction by instruction (an Alpine kernel's decompressor takes minutes instead of seconds). Refusing the JIT only on pages that hold breakpoints would need generated code to leave a chain at those pages; not done.

### Single-step and held vCPUs

A step arms `GdbVcpu::step`; `step_debug` turns it into `stepped` as the instruction starts, and the next `before_step` reports it. While stepping, interrupts and NMIs wait (QEMU's SSTEP_NOIRQ) unless the vCPU is halted, which could otherwise never finish the step.

`vCont` gives each vCPU the leftmost action that names it or names no thread; a vCPU no action names is **held**: it stays stopped while the others run. gdb steps one thread over a breakpoint this way, and lldb's `si` steps only the current thread. A legacy `s` steps the `Hc` thread and holds the rest. Held vCPUs:

- time-shared: `before_step` yields, the scheduler does not pick them (`Smp::runnable`), and they count as able to wake, so a machine whose only running vCPU is halted idles instead of powering off;
- threads: `Cpu::wait_while_held` leaves the running count (so exclusive sections, the next debugger stop among them, do not wait for it) and polls until the next resume.

### Stopping every vCPU

- One vCPU: `Cpu::run` serves the stop and runs on.
- Time-shared (`smp.rs`): `Smp::run` serves with every `Cpu` (`SharedVcpus`); RAM, devices and the JIT live in the active one, so memory goes through it with the selected vCPU's paging registers.
- Threads (`vcpus.rs`): the stopping vCPU serves inside `with_exclusive`, so the others are parked at block boundaries, halted, or held, and reaches them through the pointers their threads publish (`ThreadVcpus`, `Cpu::other_cpu`). Several vCPUs may stop at once: the second waits for the exclusive section and reports its stop after the client resumes, as gdbserver reports pending stops. `serve` re-checks every stop first: one pause is reported once, and a breakpoint removed meanwhile is not reported. While a debugger stop is in progress, threads waiting for it sleep instead of spinning (`VcpuLinks::debugger_stop`), halted vCPUs do not read their own `Cpu` (the debugger may be writing it), and the device thread neither ticks nor reads the clock, so guest time stands still: the wall clock cuts the gap to `MAX_HOST_GAP_NS` on resume (single and time-shared machines get that for free, nothing reads the clock while stopped).

### Memory

`Cpu::debug_translate` (`cpu/mem.rs`) walks the paging structures of the selected vCPU (paging off, 32-bit with PSE, PAE through the latched PDPTEs, 4- and 5-level) with no side effects: no Accessed or Dirty bits, no CR2, no TLB or large-page cache fill, no permission checks. Only RAM is read or written, never device registers. Reads stop at the first unmapped byte (`E14` if that is the first); writes are all or nothing and drop the translated code of every page they change (`code_written`, which tells the other threads too). Under SVM nested paging the vCPU's tables are guest-physical, and the walk refuses rather than read the wrong memory.

### Registers

`g` is the 66 registers in description order; `G` writes all or nothing. The general registers, RIP, EFLAGS (except VM), x87 and SSE state, the segment bases and CR2 can be changed. The selectors (which would need a descriptor load), CR0, CR3, CR4, CR8 and EFER (which change paging and modes) are refused unless written with the value they hold, so a `G` of read-back values succeeds. fiseg, fioff, foseg, fooff and fop read as zero: the CPU does not track the last x87 instruction. The tag word is the full one gdb expects, computed from the abridged one and the register contents.

### Snapshots

`Cpu::debugger` and `Cpu::gdb` are `#[snapshot(skip)]`: host state. A suspended machine resumes without breakpoints unless the new machine has a stub, which then arms every vCPU from its own configuration (`attach_debugger`, also used for every vCPU of a reboot).

## Checked

- Unit tests (`gdb/tests.rs`): framing, checksums, escapes, split reads, corrupt and oversized packets, acknowledgments and no-ack mode, every command (`?`, `qSupported`, `qXfer`, thread queries, `H`, `T`, `g`, `G`, `p`, `P`, `m`, `M`, `X`, `Z`/`z` 0-4, `c`, `s`, `C`, `S`, `vCont`, `D`, `k`, `vKill`, unknown packets), register layout and refusals, the x87 tag word, the side-effect-free walk on a higher-half 4 KiB mapping and a 2 MiB leaf, a memory write dropping JIT code.
- Whole machines (`gdb/machine_tests.rs`), each with the interpreter and the JIT, time-shared and on threads: halt at start, breakpoint, step, registers, memory, write/read/access watchpoints, Ctrl-C, kill, detach, disconnect, `W00`, pausing on connect, and two vCPUs (a breakpoint the second hits, the first stopped in its loop, stepping one while the other is held, Ctrl-C with both running).
- By hand with lldb (Xcode's) against the Alpine `vmlinuz-virt`, `--firmware none --cpus 2 --gdb-wait`: connect, `register read` (rip, cs, eflags, cr0, cr3, efer, fs_base), disassembly and `memory read` at rip, `thread list` (two threads), `si`, `breakpoint set -a`, continue to it, interrupt.

## Not done

- gdb itself was not run (none on the machine this was written on); lldb was.
- Non-stop mode, `qXfer:threads`, `qRcmd` (monitor commands), the `ymm`/`zmm` halves and the opmask registers, the guest's debug registers and MSRs in the description, `qHostInfo`.
- A breakpoint slows the whole machine to the interpreter (above).
- Under threads, other vCPUs and devices stop within one block boundary of the stop, not at the same instruction; a pause reaches a running vCPU within `EVENT_INTERVAL` instructions.
- Freestanding hosts (UEFI, the web build) compile the stub, but `serve` there polls the host's idle hook, and no page or app feeds it yet.
- Under SVM nested paging memory access fails instead of translating through the nested tables.
