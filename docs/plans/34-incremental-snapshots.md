# 34 — Incremental snapshots: dirty pages, compression, checkpoints

**Status:** implemented in the library and the CLI. The web build and the Apple app write and read the new format (compressed) through their existing suspend paths; neither takes checkpoints yet. Builds on `29-snapshots.md`.

## Why

A suspend wrote every non-zero page of RAM as it was: a 2 GiB guest that had touched its memory wrote about 2 GB, through 1 MiB POSTs on the WebKit engine (up to about 20 s, `29-snapshots.md`, "Limits"). Two things shorten that: compressing pages, and writing only the pages written since an earlier snapshot. The second also gives checkpoints: take one every few seconds of a running machine at the cost of what changed, and go back to any of them.

## The dirty log

`ram::DirtyLog`: one bit per page of the flat RAM buffer, shared by every vCPU, the device thread and `GuestMem` (it lives in `CodePages`, which all of them already hold). Armed, it means "pages written since snapshot P" and carries P's id; unarmed, there is no parent and the next checkpoint is full.

Every way a write reaches RAM marks:

- **Soft-TLB stores** (the interpreter's `mem_write` fast path, the inline stores of the aarch64, x86_64 and WebAssembly JITs, the shadow-stack tag of inline `call`/`ret`) store only through a write tag. While armed, `translate` marks the page when it resolves a store and only then grants the tag, and a load or fetch never grants the tag of a clean page (`needs_mark`). So the JIT emitters are unchanged: a clean page has no write tag, and the first store to it takes the slow path. Marking at the resolution, not at the store, also covers the callers that store through a host pointer after `translate(Write)`: the locked read-modify-write and `cmpxchg16b`, and `rep_bulk`.
- **Physical writers** mark themselves: `phys_write` (the VMCB, AVIC backing pages, shadow-stack stores), `update_ram_u64/u32` (the walkers' Accessed and Dirty bits, nested A/D, AVIC, CET tokens; even when the exchange fails), the BIOS's `store`.
- **Device DMA** goes through `GuestMem`, whose `write_raw` and `fetch_or_u32` mark every page they touch, under the IOMMU too (`with_dma` ends in `write_raw`). A machine's `GuestMem` is made with `GuestMem::for_pages`, which takes the log with the code flags; `GuestMem::new` (host buffers, tests, the UEFI hardware path) has none.

**Arming** (`suspend::arm_dirty_log`) clears the bits, records the parent, and revokes every write tag of every vCPU, in the running TLB and in SVM's parked one (`Cpu::revoke_all_write_tags`). A tag that survived would let a store skip the log. It happens only with every vCPU stopped: after a checkpoint is written, and after a restore of a chain that may be a parent. Reading the bits happens then too, so marking needs only its own atomicity (relaxed `fetch_or`, after a load that skips it once the page is dirty).

The cost while armed: the first store to each page after a checkpoint walks once and marks. While unarmed (no checkpoint taken) there is one extra relaxed load in the TLB refill.

A guest reboot builds a new `Cpu` and so a new, unarmed log: its first checkpoint is full, since the RAM was reloaded behind no log.

**Checking it.** `MachineConfig::check_dirty_tracking` (hidden; the CLI's hidden `--check-dirty-tracking`) keeps a copy of RAM at each arm and makes the next checkpoint fail, naming the pages, if any page changed without being marked. It runs in the library's checkpoint tests, and it passed on Alpine (below).

## Format 2

```text
"UEMUSNAP"  u32 version = 2
u8 kind (0 full, 1 delta)  [16] id  [16] parent (zeros for a full snapshot)
shape  guest_ns  cpus  rng_state  devices       (as format 1)
RAM: u64 length, then per group of 64 pages: u64 bitmap, then each page in the bitmap:
    u8 0 FILL       u8 byte          (the whole page is that byte)
    u8 1 RAW        4096 bytes
    u8 2 COMPRESSED u16 length, that many bytes (snapshot/compress.rs)
u32 CRC-32 of everything before it
```

- In a full snapshot the bitmap names the pages that are not all zero; the others are zero. In a delta it names the pages written since the parent (a page written back to zero is a FILL 0); the others are the parent's.
- A delta carries the whole CPU and device state; only RAM is incremental.
- Compression is per page, so a page needs nothing around it: a byte-oriented LZ77 in the shape of an LZ4 block (token nibbles for literal and match length, 16-bit distance, 4 KiB window), written in the library (no dependencies; `--no-default-features` stays dependency-free). A page is kept compressed only if that saves more than the 3-byte header. Incompressible data costs little: after 32 misses in a row the search steps faster. The decoder checks every length and distance, because the CRC is only checked at the end of the stream.
- The id is 128 bits mixed from guest time, every vCPU's `icount` and `rip`, the generator behind `rdrand`, the parent's id and the host's clock (wall time on hosted builds, the `WallClock` where there is one). The host time keeps two branches taken from one checkpoint apart on the virtual clock.
- **Format 1** (no kind or ids, every page raw) still loads: `a_format_1_snapshot_still_loads`, and a format-1 snapshot of Alpine from the previous build resumed in this one. It cannot be a parent (it has no id). The per-type layout tags are unchanged: no saved field was added (`DirtyLog` is behind `CodePages`, which `Cpu` already skips).

`read_snapshot_info(source)` reads the header alone (version, id, parent), for a host that lines up a chain.

## Checkpoints

`StopHandle::checkpoint(sink, done)`: the request goes in a queue and raises the stop flag, both under the queue's lock. `Vcpu::run_once` now loops: when the vCPUs come back `Stopped` and the stop was not a real one (`HostRequests::ended`, set by `stop()` and `suspend()` before the flag), it writes the pending checkpoints, and lowers the stop under the same lock only once the queue is empty, so a request made while one is written (from its `done`, or the timer and a key at once) is taken too, as its child. Then it runs the vCPUs again (`run_vcpus`: threaded, time-shared with the active vCPU kept, or the one). A machine that ended, or runs `verify_jit` or a BIOS disk boot, answers requests at once with an error (`HostRequests::closed`).

A checkpoint (`Vcpu::checkpoint`): refuse if a vCPU thread died; with the check on, compare RAM; `save(SaveKind::Checkpoint)`, a delta against the log's parent or full without one; undo `save`'s PMU parking (`unpark_after_save`: only what `save` parked, which it does first so that a save failing early is undone too; time-shared the inactive ones stay parked by the scheduler); take the pause out of guest time (`WallClock::skip_pause_since`); and only on success arm the log against the new id. A failed checkpoint changes nothing, and the next delta is against the last one that was written. `done` gets the header or the error. Requests that meet a stop, a reset or a shutdown are answered with an error, and a stop raised only for them is cleared so it does not end the next boot. A vCPU thread that died ends the run with `Internal`.

Re-running the vCPUs: `run_threaded` builds new links and hands the host poll hook back to vCPU 0; every vCPU's decode cache generation is bumped first, because a vCPU that left on `ATTENTION_EXIT` may not have drained the invalidations published just before.

`sink.finish` runs while the machine is still paused and its disks flushed: that is where a host copies the disks that belong with the checkpoint, before it makes the snapshot visible.

**Restoring a chain**: `MachineConfig::resume_from_chain(sources)`, a full snapshot then deltas, each the child of the one before. Each is checked whole (shape, parent id, CRC) before the next is applied; a delta leaves the pages it does not name. When every source is kept (`SnapshotSource::retained`: `FileSnapshotSource::open_kept`, `MemorySnapshot`) and has an id, the log is armed against the last, so the next checkpoint is a delta of it. A suspend's file, which `finish` deletes, is not kept: the next checkpoint is full. `suspend_to` always writes a full snapshot, since it outlives what came before.

### I/O consistency

A checkpoint is only meaningful with the disks as they were (`29-snapshots.md`, "I/O consistency"). The library flushes them and pauses; copying them is the host's. The CLI copies every writable disk and the nvram (`nvram.kv`, which the TPM's NV and the UEFI variables live in, outside the snapshot) beside the checkpoint before renaming the checkpoint into place, so a checkpoint never exists without them, and copies them back before opening anything when it resumes one.

## CLI

- `--checkpoint-dir DIR` (on `advanced boot` and `quick boot`): Ctrl-] then `c` takes a checkpoint; Ctrl-] then any other key stops (or suspends, with `--suspend-to`) as before. `--checkpoint-every SECS` takes them on a timer. Each is `DIR/NNNN.checkpoint` with `NNNN.disk-K` and `NNNN.nvram` beside it; the CLI prints its kind, size and time. Numbers only grow, so a branch taken after going back never overwrites another.
- `--resume-from DIR/NNNN.checkpoint` finds the chain by the ids in the headers of the checkpoints in DIR, copies NNNN's disks and nvram back, resumes, and keeps every file; its checkpoints go on in DIR as deltas of NNNN. Any other `--resume-from FILE` is a suspend's snapshot and is deleted as before.
- On APFS (and Linux file systems with reflinks) `fs::copy` clones, so a disk copy costs nothing until either side is written.
- Going back replaces the machine's disks and nvram as they are with the checkpoint's copies (every copy is checked to exist first). That happens before the library reads the chain, so a chain that then fails its CRC has still put the old disks back; the checkpoint's own files are untouched either way.
- The copies are by position among the writable disks (`disk-K`): give the same disks in the same order, as for any resume.
- Refused: `--checkpoint-every 0`, checkpoints with `--verify`, and `--suspend-to` a `.checkpoint` name. Numbers are handed out by the process, so two checkpoints asked for at once get two numbers.

## Measured

Host: Apple Silicon Mac, native aarch64 JIT, with other agents building in parallel (load average 9 to 46 during the runs). Guest: Alpine's `linux-virt` 6.18.58 (`vmlinuz-virt`, `initramfs-virt`), 1024 MiB, `--firmware none`, `rdinit=/bin/sh`; the workload (`feed.sh`) mounts a tmpfs, writes 22.9 MB of text (`seq 1 3000000`) and 30 MB from `/dev/urandom`, then idles.

| | format 1 (previous build) | format 2 |
|---|---|---|
| suspend after the workload | 156 MB | 88.5 MB (1.76x smaller; the 30 MB of random bytes stay as they are) |
| resume + 1 s + suspend, user CPU | 0.93-1.00 s | 0.94-0.98 s (compression costs what the smaller CRC and write save) |

Checkpoints every 6 s, 2 vCPUs on threads, `--check-dirty-tracking` on, two runs: the first (full) 50.5-51.5 MB in 462-503 ms; idle deltas 0.1-0.4 MB in 66-475 ms; the deltas across the text write 5.3-7.9 MB in 121-194 ms, across the random write 29.7-30.5 MB in 172-608 ms (the spread is the host's load). Every delta passed the dirty-log check. The chains of 6 and 12 resumed (2 vCPUs) to a shell whose files had the right checksums (the text's matched the host's `seq`). A format-2 suspend moved native to WebAssembly (`boot.mjs`, wasm JIT, time-shared) and back, and a format-1 snapshot from the previous build resumed in this one, both with the checksums intact.

The time is what the host sees from the request to `done`, so it includes the vCPUs reaching an event check and the disk copies; under the load above it is noisy.

## Tests

- `cpu/dirty_tests.rs`: interpreter and JIT stores after arming (with write tags left from the run before: they fail without the revocation), a load not granting a clean page its write tag (fails without `needs_mark`), `rep stos` through `rep_bulk`, the walker's Accessed and Dirty bits, `phys_write`, device writes across a page boundary and `fetch_or_u32`.
- `cpu/smp_tests.rs`: two vCPUs on threads, each with write tags from a first run, store after arming; both pages are dirty (fails without the revocation).
- `machine/suspend/tests.rs`: a checkpoint, a delta after the guest rewrote, zeroed, byte-poked and lock-added pages, and a suspend of the same moment; the chain restores to the same bytes as the suspend (interpreter, JIT, two time-shared vCPUs; with the dirty-log check, which named the seven missed pages when the revocation was removed). Checkpoints during the 40-million-iteration computation leave its result alone and their chain resumes it (interpreter, JIT, time-shared, threads with both engines). Chains that do not fit are refused before anything runs; a failed checkpoint leaves the machine running and the next delta against the last good one; a checkpoint that meets a stop is answered with an error. Format 1 loads. Eight rounds of random device writes (zero, one byte, compressible, random, a few bytes) each restore exactly from the whole chain, and each delta holds only the pages written.
- `snapshot/compress.rs`: round trips of 3000 generated pages, random pages refused, every truncation and 20000 bit flips of a compressed page and random junk decoded without a panic, generation wrap.
- The CLI: the escape key's parsing, checkpoint numbering, disk and nvram copies made and put back, a failed copy leaving no checkpoint.

## Limits and next steps

- **No checkpoints on the web page or in the Apple app yet.** They get the compression through their suspend paths (`host_snapshot_*`, `ua_suspend`); a C ABI for `StopHandle::checkpoint` and resuming a chain, with the hosts copying their disks (OPFS, the app's Documents), is the next step. For the iOS background flow the gain would be a suspend that is a delta against a kept base, which needs the app to keep that base and its disk copies.
- A checkpoint pauses the machine for the whole write: no copy-on-write snapshot while the guest runs. The first, full checkpoint of a large guest is as slow as a suspend.
- Compression runs on the vCPU thread, one page at a time, single-threaded.
- Restoring a chain reads every link's CPU and device state; only the last one's counts. Long chains (hundreds) are slow to resume; nothing merges them.
- The CLI copies whole disk files; on a file system without clones that is a full copy per checkpoint. Read-only disks and http-backed bases are not copied (they cannot change).
- `quick boot NAME --resume-from DIR/NNNN.checkpoint` decides between the installer and the disk before the disk is put back; the resume is right, but a guest reboot afterwards may boot the other one.
