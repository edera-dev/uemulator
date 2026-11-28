# 29 — Machine snapshots: suspend, resume, migrate

**Status:** implemented in the library, the CLI, the web build and both Apple engines. Checked on real Linux (the Alpine demo kernel) natively and under Node; the Apple app builds for macOS and iOS but its suspend and resume are not yet checked running (see "Checked").

Format 2, delta snapshots, compression and checkpoints of a running machine: `34-incremental-snapshots.md`.

## Why

iOS suspends a backgrounded app after a short grace, and may then end it, or only the WebKit engine's WebContent process, to reclaim memory (`28-apple.md`, "Backgrounding"). Until now that lost the machine: an install in progress started over. A snapshot of the whole machine, taken when the app goes to the background, lets it come back. The snapshot is the same on every host, so it also moves a machine between the native engine and the WebKit one (the WebAssembly build), and between the CLI, Node and the app.

## What a snapshot is

A stopped machine's guest-visible state, written to a host sink as one stream (`crates/uemulator/src/machine/suspend.rs`):

1. `UEMUSNAP`, format version; from format 2 also whether it is a delta, its id and its parent's (`34-incremental-snapshots.md`).
2. The machine's shape: RAM size, vCPU count, and the devices (`Devices::snapshot_shape`: local APICs, each PCI function's slot and virtio type, IOMMU, key-value store, TPM, boot framebuffer). A restore compares it before reading any state, and refuses another machine's snapshot with a message naming both.
3. Guest time in nanoseconds.
4. Every vCPU (`Cpu`), then the machine-wide `rng_state`.
5. The devices (`Devices`).
6. RAM in groups of 64 pages, each a bitmap of the non-zero pages (in a delta, of the pages written since the parent) followed by those pages, each one repeated byte, compressed, or raw (format 1 wrote them raw: an idle 512 MiB Alpine guest was about 80 MB).
7. A CRC-32 of everything before it (`snapshot::Writer::finish`).

The encoding is explicit little-endian with fixed widths, `usize` as `u64`, so wasm32 and 64-bit hosts read each other's. Each saved struct and enum writes a 32-bit tag hashed from its name and saved field names: a snapshot from a build whose layout differs fails at the first changed type ("X is laid out differently: it comes from another build"). Snapshots are meant to resume in the build that wrote them; the app's two engines come from one commit.

## How state is saved: `#[derive(Snapshot)]`

`crates/uemulator-snapshot-derive` is a dependency-free proc macro (the library's `--no-default-features` build stays free of third-party code). It saves fields in declaration order and restores **into** an existing value: a restore builds a new machine from the same configuration, then overwrites what the guest can observe. Host handles, callbacks and caches stay as the new machine made them.

- `#[snapshot(skip)]`: not saved. For host state (`ConsoleHandle`, `NetHandle`, a disk's backend, the JIT, threads, callbacks) and caches (soft TLB, decode cache, SVM's parked host TLB and nested-leaf cache).
- `#[snapshot(fresh)]`: rebuilt from the snapshot instead of loaded in place.
- `#[snapshot(new)]` on a type: also derives `load_new`, so it can be created from a snapshot. A `Vec` that changes length or an `Option` that is `None` in the new machine needs this.
- A saved field under `#[cfg]` is a compile error: the snapshot would differ between targets.

Because every field of a derived type must implement `Snapshot` or be skipped, **adding a field to `Cpu` or a device forces a decision**: a field holding a type without the trait does not compile. A new field of a plain type is saved automatically. That is right for guest state, and wrong for a host address or a host time, which must be marked `skip`. Raw host addresses in integers are the one thing the compiler cannot catch. `Cpu`'s are already skipped: `tlb_active_base`, `ram_base`, `tlb` addends, and `Rmw`'s host address.

Hand-written impls: `Box<dyn VirtioDev>` forwards to the device (`VirtioDev: Snapshot`); the TPM's in-progress hash sequences go through `Hasher::export_state`/`import_state` (`uemulator-crypto`); loaded RSA keys are saved as modulus, exponent and prime and rebuilt with `from_prime`.

## Suspending and resuming

- `MachineConfig::suspend_to(sink)` and `StopHandle::suspend()` (or `EmulatedMachine::suspend()`): the machine stops at its next event check, as for `stop()`, and then the snapshot is written. `wait()` returns `Suspended` once the sink's `finish` succeeded, or `SuspendFailed(why)`; the machine is gone either way. Without a sink, `suspend()` is `stop()`.
- `MachineConfig::resume_from(source)`: instead of booting, the snapshot is read into the new machine before anything runs. `wait()` returns `ResumeFailed(why)` for another machine's, another build's, or a damaged snapshot, with nothing run. A guest reset after resuming boots afresh from the configured kernel or firmware.
- `FileSnapshotSink` writes `path.tmp` and syncs and renames it in `finish`, and touches nothing before the first write. `FileSnapshotSource` deletes the file in `finish`. `MemorySnapshot` is for tests.
- Refused at configuration: `verify_jit` (its twin is not saved) and a BIOS disk boot (`cpu/bios.rs` keeps services on the host). Refused at suspend: a virgl 3D GPU (`VirtioDev::snapshot_blocker`; its contexts live in the renderer).

`run_once` builds every vCPU before running (`secondary_cpus`), restores into them, runs threaded or time-shared, and on a suspended stop saves all of them. `run_threaded` and `Smp::into_cpus` hand every vCPU back for that.

### Threads or time-sharing

The native engine runs vCPUs on threads; the WebKit engine time-shares them. A snapshot from either resumes in the other.

- **Clocks.** Guest time is saved as one number. Every clock resumes from it: the virtual clock through `time_base_ns`, a realtime machine through `WallClock::resume_at`. That sets a base shared by every clone of the clock, because a fresh page's `performance.now()` starts near 0, below any guest time worth resuming at.
- **`icount` and the PMU.** `icount` is per vCPU on threads and machine-wide when time-shared, where the inactive vCPUs' copies are stale. The PMU counts relative to it. At save, every PMU is parked (`pmu_park`) against the `icount` it relates to: its own on threads, the machine's when time-shared (copied into each vCPU first). At restore, each is unparked against its restored `icount`, and when the new machine time-shares, the inactive ones are parked again against vCPU 0's. That is what a switch between vCPUs does anyway.
- **Event deadlines.** `next_event` is set to `icount`, so the first event check recomputes every deadline from the devices.

### After a restore (`settle`)

`set_cpl` recomputes the TLB base, the TLB is flushed, and `shadow_stack_on` is recomputed. Devices get `VirtioDev::restored`: virtio-gpu takes the screen back and repaints its scanout, which the new machine's `DisplayHandle` has never seen. The cursor image comes back with the guest's next cursor update.

## I/O consistency

A snapshot holds the guest mid-flight: its page cache, its file system journals, requests it has queued. All of that describes the disks as they were at that moment. Resuming it over disks that changed since would let the guest write through stale metadata and corrupt them. The rules:

1. **No I/O in flight.** virtio-blk completes every request on the vCPU thread inside the kick (`dev/vblk.rs`), so a stopped machine has none. The used ring and the data are in RAM and on the disk together.
2. **Flushed first.** `suspend::save` calls `Devices::flush_storage` (`VirtioDev::flush_storage`, the backend's `flush`) before writing anything. A udisk writes its map then. Once flushed it writes nothing more on close (`Udisk::drop` only flushes when dirty), so the files stop changing. A failed flush fails the suspend.
3. **Complete or absent.** The sink's `finish` is the last step, and it makes the snapshot visible: a temporary file and a rename, in the library, the app's loopback server and `boot.mjs`. A host that dies halfway leaves no snapshot, not a torn one.
4. **Resumed at most once.** The source's `finish` comes after the whole snapshot is read and checked, just before the guest runs, and deletes it. From then on the guest moves its disks on; a crash afterwards leaves the disks as after a power cut, and the machine cold boots.
5. **Disks unchanged in between.** The app records each disk's name, size and modification time beside the snapshot once the suspend is complete (`MachineStore.recordSnapshot`, `suspended.json`). Before resuming it compares them (`snapshotMatches`). A mismatch, or a snapshot without its record (the app died in between), sets the snapshot aside and boots afresh with a message. It is not resumed into corruption. nvram.kv is not in the record: building a machine compacts it, so its time says nothing.

Checked on the Alpine demo kernel with a udisk: data written in one run, suspended, resumed, `drop_caches`, read back through virtio, matched the copy kept in guest RAM and the host's flattened image (hash `8b7c…0e37` all three).

The network is not saved. The host side (the gateway's NAT and DHCP state, Wisp streams, frames queued in `NetHandle`) starts fresh, as after a NAT box reboots. The guest keeps its address, and its open TCP connections reset.

Guest wall time continues from the snapshot: the RTC and the guest clock do not jump forward by the time spent suspended. That is what QEMU does for a stopped VM. A guest running NTP corrects itself.

## Hosts

- **CLI:** `advanced boot … --suspend-to FILE` (also on `quick boot` and the other boot commands, which share these options) makes Ctrl-] and `--timeout` suspend instead of stop. `--resume-from FILE` resumes; give the same options and disks. A UEFI boot without `--nvram` has no key-value device, while the web build always gives UEFI machines one, so cross-host tests use `--firmware none` or an NVRAM file on both sides.
- **Web build:** `uw_set_snapshots(suspend, resume)` before `uw_create`. The host imports `host_snapshot_write`/`finish`/`read`/`consumed` and the exit codes 5 suspended, 6 suspend failed, 7 resume failed (`www/protocol.js` `EXIT_NAMES`). `CTRL.STOP` = `STOP.SUSPEND` (2) asks for a suspend. `worker.js` POSTs writes by offset, so a retried request lands once, and reads by range from the boot message's `snapshot.suspend` and `snapshot.resume` URLs. Only the Apple app's page sends them. `node hack/wasm/boot.mjs --suspend-to FILE --resume-from FILE` does it under Node; `--until` or `--seconds` then suspends.
- **Apple crate:** `ua_config.suspend_path`/`resume_path`, `ua_suspend`, `UA_EXIT_SUSPENDED`/`SUSPEND_FAILED`/`RESUME_FAILED` (`include/uemulator.h`).
- **App** (`28-apple.md`, "Suspend and resume"): `MachineEngine.suspend()` on both engines. A machine's snapshot is `suspended.snapshot` in its folder. Start resumes it when the record matches ("Resume" on the button). On iOS the app suspends running machines on entering the background and resumes them on return; after the app was ended, Start resumes. Suspend (⇧⌘S) is in the toolbar, and Settings can discard the state. Switching the engine in Settings while suspended moves the machine to the other engine. `-debug-suspend-after SECONDS` suspends each run that long after it starts.

## Checked

- Library tests (`machine/suspend/tests.rs`). A guest doing 40 million iterations of integer, memory, SSE and x87 work, suspended partway, prints the same result as an uninterrupted run, with the interpreter and the JIT, and across them. The same holds across threaded and time-shared vCPUs both ways. A machine with every device (disk, NIC, TPM, key-value store, screen, input, RNG), resumed and suspended again without running, gives byte-identical snapshots. Another machine's snapshot, a damaged one, a truncated one, or one with a byte appended is refused with nothing run and the snapshot kept. `snapshot.rs` covers the derive's round trips, the CRC, and every truncation and bit flip of a sample stream. Removing `xmm` from the snapshot, or not restoring the devices, made the tests fail.
- Alpine 3.24.3 demo kernel, 512 MiB, 2 vCPUs, a 64 MiB udisk, TPM. Natively (CLI, threads): booted, suspended at the shell, resumed (uptime continued, `nproc` 2), wrote 4 MiB to the disk, suspended, resumed, and checked the disk as above. **Migration**: suspended natively (threads), resumed in the WebAssembly build under Node (time-shared, wasm JIT), suspended there, resumed natively again. Uptime continued throughout, the data in RAM and on disk matched, and `poweroff -f` ended it cleanly. Each snapshot file was gone once resumed.
- The Apple crate's C ABI: suspend to a file, resume from it once, a second resume fails at create.
- Not checked: the app itself. The macOS and iOS builds succeed, but neither a scripted launch of the Mac app from this session's shell nor its Terminal panel got a window, and this host's CoreSimulator is too old for the simulator. To check: launch the app with `-new-machine snap -new-source demo -engine native -autostart -debug-suspend-after 30`, quit once it says Suspended, then launch with `-machine snap -autostart -engine webkit` and see the demo's shell carry on. On a device, background the app during an install and come back.

## Limits and next steps

- A suspend writes all of RAM (zero pages skipped). A 2 GiB guest that has touched most of its memory wrote about 2 GB, through 1 MiB POSTs on the WebKit engine: measured in M0 at 6 to 11 ms each, so up to about 20 s, close to iOS's background grace. Pages are compressed now (format 2: 1.76x smaller on an Alpine guest with text and random data in a tmpfs), and checkpoints write only the pages dirtied since the one before (`34-incremental-snapshots.md`); a suspend that is a delta against a base the app keeps is the next step there.
- Suspending as soon as the app goes to the background trades the grace period's running time for safety: a quick app switch costs a suspend and a resume.
- No snapshot with a virgl GPU, under `verify_jit`, or after a BIOS disk boot.
