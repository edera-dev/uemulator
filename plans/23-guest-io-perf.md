# Guest-visible performance: syscalls, memory, disk, network

Round 1 (2026-10-31). Measured from inside an Alpine guest (`quick create perf-test --install alpine`, installed to disk, logs in as root by itself) with the tools in `hack/perf/guest/`:

- `ubench.c`: static guest binary. `sys` (raw `getppid`, 1-byte `read`+`write` on /dev/zero and /dev/null, vDSO `clock_gettime`), `mem` (musl `memcpy`/`memset` and hand-written `rep movsb`/`rep stosb` at 4K, 64K, 64M), `disk` (O_DIRECT 1M sequential write+fsync and read of a 512 MB file, then 4K random reads and writes), `net` (256 MB from and to the host, 1-byte TCP request/response), `randread` (200K 4K random reads, for profiling).
- `netsrv.py`: the host side on loopback (the guest's 10.0.2.2): HTTP for the binary and the three TCP peers.
- `drive.py`: boots the machine, runs a command file at the prompt (`SAMPLE=N` runs macOS `sample` on the emulator during line N), `ab.py`: interleaves binaries and prints medians.

```
x86_64-linux-musl-gcc -O2 -static -o target/perf-guest/ubench hack/perf/guest/ubench.c
python3 hack/perf/guest/netsrv.py &
hack/perf/guest/ab.py hack/perf/guest/all.txt 3 base=OLD_BINARY new=target/release/uemulator [-- --cpus 4]
```

The guest's clock follows the host's (realtime is the default), so its own timings are wall time; a `sleep 5` takes 4.94 s of host time. The host here (M3 Max) had another emulator using a full core the whole time, so absolute numbers moved by up to 10% between runs: only interleaved A/B medians are comparable, and single runs are not.

## Results

`ab.py`, medians, one vCPU (`perf-test` as created), 3 rounds:

| benchmark | before | after | |
|---|---:|---:|---:|
| syscall getppid, calls/s | 1.68M | 2.16M | 1.28x |
| syscall read+write 1B, calls/s | 1.15M | 1.50M | 1.31x |
| vDSO clock_gettime, calls/s | 7.18M | 8.82M | 1.23x |
| memset 4K / 64K / 64M, MB/s | 13.6K / 15.4K / 12.1K | 35.2K / 41.3K / 33.4K | 2.6-2.8x |
| rep movsb 4K, MB/s | 14.5K | 29.2K | 2.0x |
| rep stosb 4K, MB/s | 15.8K | 33.9K | 2.1x |
| memcpy 4K / 64K / 64M, MB/s | 27.6K / 27.5K / 21.3K | 30.6K / 32.9K / 22.1K | 1.04-1.2x |
| disk seq write / read 1M, MB/s | 920 / 1966 | 1013 / 2088 | 1.06-1.1x |
| disk rand read / write 4K, IOPS | 29.1K / 23.8K | 29.1K / 24.2K | 1.0x |
| net rx / tx, MB/s | 64 / 51 | 87 / 75 | 1.35x / 1.47x |
| net TCP request/response, trans/s | 246 | 7189 | 29x |

Four vCPUs (`-- --cpus 4`, a host thread per vCPU plus the device thread), 3 rounds:

| benchmark | before | after | |
|---|---:|---:|---:|
| syscall getppid / read+write, calls/s | 1.63M / 1.16M | 1.56M / 1.47M | 0.96x / 1.27x |
| vDSO clock_gettime, calls/s | 7.63M | 8.71M | 1.14x |
| memset 4K / 64K / 64M, MB/s | 14.9K / 17.3K / 12.6K | 36.4K / 44.6K / 30.7K | 2.4-2.6x |
| memcpy, rep movsb, rep stosb (all sizes) | | | 0.93-1.05x |
| disk seq write / read 1M, MB/s | 1100 / 2161 | 963 / 2093 | 0.88x / 0.97x |
| disk rand read / write 4K, IOPS | 12.9K / 11.9K | 13.4K / 12.1K | 1.04x / 1.02x |
| net rx / tx, MB/s | 103 / 39 | 104 / 66 | 1.01x / 1.69x |
| net TCP request/response, trans/s | 216 | 5255 | 24x |

At three rounds the lines this round did not touch (rep movsb/stosb, memcpy, the disk) moved by up to 12% either way, and so did getppid; five rounds of the syscall benchmark alone gave getppid 1.54M -> 2.06M (1.34x) and read+write 1.09M -> 1.48M (1.37x). Read single lines of a 3-round table as +-10%.

The serial-console boot of the same kernel (`hack/perf/unpack.sh` with the `perf-test` artifacts, which pushes the whole boot log through the UART) is unchanged by the tick limit: 2.15 s before, 2.14 s after (medians of three, interleaved). The UART does not depend on the tick: LSR always reads empty and a port access routes its interrupt at once.

## What was slow, and what changed

- **Network round trips were timer-bound.** A frame from the host waited for whoever ran the devices to wake up on its own: the halted vCPU slept in 2 ms `thread::sleep` slices (one vCPU) or the device thread in 1 ms parks (threads). The usernet worker looked at its host sockets only when a guest frame arrived or its 2 ms wait ran out. A round trip was about 4.3 ms, and with no TCP window scaling 64 KiB per round trip capped bulk transfers near 64 MB/s. Now `NetHandle::send_frame` unparks the thread that runs the devices (installed in `Vcpu::run_once`; the idle wait parks instead of sleeping), and usernet waits in `poll(2)` on its sockets plus a socket pair that guest frames (`NetHandle::on_guest_frame`, written only while the worker waits) and finished connects write to. Without `poll(2)` (Windows) the old timed loop stays.
- **The kernel's syscall path was cut into tiny blocks.** `ends_block` ended a block after `cli` and after every group-7 instruction, so `swapgs`, `clac`, `stac` and `rdtscp` each cost a trip through the run loop and a block lookup. They now continue the block: the GS base is read from `Cpu` on every access, the soft TLB never caches a supervisor access to a user page while SMAP is on, so AC is checked afresh, and interrupts are only taken between blocks, so `cli` changes nothing for the rest of the block. `sti` still ends one (its shadow). `push imm8/imm32` (four per syscall entry), `swapgs`, `clac`, `stac` and `cli` are lowered inline on aarch64 and x86_64 at CPL 0, with the interpreter as the out-of-line fallback that raises the fault elsewhere; `push imm` is also lowered on WebAssembly.
- **`rep stos` of 2, 4 or 8 bytes filled element by element**, and musl's `memset` is `rep stosq`. A fill of identical bytes (memset's widened byte) is now one host `memset`; any other pattern is written once and doubled with `memcpy`.
- **A bulk `rep` slice counts 32K instructions, so the devices were ticked every 32 KiB of copying** (and every 4096 instructions otherwise), about a tenth of the time in the memory benchmarks. On the realtime clock `service_events` now ticks at most once per `MIN_DEVICE_TICK_NS` (20 us); the virtual clock is unchanged, so `--verify` and time-sharing determinism are not affected.
- **virtio-blk zeroed a fresh buffer for every read and copied every write, header included, into a new one.** It keeps one scratch buffer and reads write data past the header (`Chain::read_from`).

## What bounds the rest, and what is left

- **Random 4K disk I/O is guest CPU.** In `randread` the vCPU thread spends about 94% in guest code (block layer, ext4, interrupt path) and about 6% in the host `pread`. Device-side changes cannot move it; general JIT speed (plan 20's register pinning) would.
- **Sequential writes end in the guest's FLUSH, which `sync_data` turns into `F_FULLFSYNC` on macOS** (about an eighth of the emulator thread's samples in the disk run). Durable across host power loss, and the reason write throughput is about half of read. A per-disk opt-out (like QEMU's `cache=unsafe`) would be the user's call; not done.
- **Bulk TCP is guest CPU per packet**: during a 256 MB transfer the vCPU is busy for 80% of the samples while usernet waits in `poll`. virtio-net offers no offloads, so the guest checksums and handles every 1500-byte frame. `VIRTIO_NET_F_CSUM`/`HOST_TSO4` (usernet terminates TCP, so it can take 64 KiB segments with no checksum as they are) and `GUEST_CSUM`/`GUEST_TSO4` with `MRG_RXBUF` would cut that by an order of magnitude, but only for usernet: a UDP or loopback link needs real segments, so the device has to know what its host side accepts. Window scaling in usernet would only matter after that.
- **Supervisor accesses to user pages are never cached while SMAP is on**, so every `copy_to_user`/`copy_from_user` access walks the page tables (`Cpu::translate` is the top emulator function in the syscall profiles). A TLB half for "CPL 0 with AC set" would fix it, but every EFLAGS.AC writer (popf, iret, sysret, interrupt delivery, the inline `stac`/`clac`) would have to switch `tlb_active_base`, and a miss there silently weakens the guest's SMAP.
- Still through the interpreter on the syscall path: `syscall`, `sysret`, `sti` and `rdtsc` (one each per syscall). `rdtsc` costs a `clock_gettime` per read.
- With four vCPUs, random 4K disk I/O is less than half the one-vCPU rate (13K against 29K IOPS) before and after this round: the completion interrupt and the wakeup of the waiting task cross vCPU threads. Not looked into.
- `--verify` diverges at the first frame the host network sends (the twin's NIC has an empty `NetHandle`, as with typed input); verify boots with `--remove-network 0`.
