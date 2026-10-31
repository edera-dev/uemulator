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

## Round 2: virtio-net offloads (2026-10-31)

Bulk TCP was the guest kernel checksumming and walking its stack once per 1500-byte frame in both directions. The NIC now offers checksum and TCP segmentation offloads, but only to a host side that says it can take them (`NetHandle::set_offloads`, `NetOffloads`):

- **Guest to host** (`takes_guest_segments`): `CSUM` + `HOST_TSO4`. The guest hands over TCP segments of up to 64 KiB with the checksum left as the pseudo-header sum. usernet terminates TCP and never checks a checksum, so it takes them as they are. The device validates the header against what was negotiated (`NetHeader::transmit_allowed`): NEEDS_CSUM inside the frame, gso type NONE or TCPV4 only with HOST_TSO4, a non-zero `gso_size`, NEEDS_CSUM on every segment. Anything else is dropped, and the chain is still completed. With nothing negotiated the header is ignored, as before. `MAX_FRAME_LEN` grew to a VLAN-tagged 64 KiB IPv4 packet; the old 65536 dropped full TSO segments.
- **Host to guest** (`sends_large_segments`): `GUEST_CSUM` + `GUEST_TSO4` + `MRG_RXBUF`. usernet reads up to the guest's window (at most 65495 bytes) per segment and queues it with `NetHandle::send_tcp_segment`, checksum field seeded with the pseudo-header sum. The device marks it NEEDS_CSUM (`csum_start` at the TCP header, `csum_offset` 16) and, when it is longer than the MSS, `GSO_TCPV4` with `gso_size` = the flow's MSS. A guest that did not take GUEST_CSUM gets the checksum finished by the device; one without GUEST_TSO4 never gets a long segment (usernet checks `guest_takes_tcp_segments` before each poll, and the device drops a long one queued before a renegotiation; the flow then stalls, which only happens across a driver reset that lost the connection anyway). Merged receive buffers: the device takes as many posted chains as the frame needs, or none (`Queue::rewind_to`) and keeps the frame queued in order, and publishes all used entries behind one `idx` update (`Queue::push_used_all`) so another vCPU never sees the first buffer of a group without the rest.
- **The UDP link and the loop gateway declare nothing** and keep plain, checksummed, MTU-sized frames. The far end of a UDP link is another machine that expects real Ethernet. Segmenting and checksumming in the device instead would move the same per-frame work from the guest to the thread that runs the devices (the vCPU itself with one CPU), and a UDP link pays one datagram per frame regardless.
- **usernet acknowledged a full window with a zero one.** The ACK for guest data advertised `65535 - to_host.len()` before anything was written to the host socket, and nothing announced the window once `poll` drained it. One TSO segment fills the whole 64 KiB window, so the guest would have waited for its persist timer. `tcp_in` now writes to the host before building the ACK, and `poll` sends a pure window update when the window opened by min(MSS, 32 KiB) since the last segment (RFC 1122 4.2.3.3). On its own this changed nothing measurable: with MTU frames the ACKs of later frames carried the reopened window.

`ab.py all.txt 3`, one vCPU, medians. `window` is the usernet fix alone, `new` adds the offloads. Inside the guest `/sys/bus/virtio/devices/*/features` for the NIC reads bits 0, 1, 5, 7, 11, 15 and 16. The host was quieter than in round 1, so the base column is higher than round 1's "after".

| benchmark | base | window | new | |
|---|---:|---:|---:|---:|
| net rx (host->guest), MB/s | 123 | 126 | 652 | 5.3x |
| net tx (guest->host), MB/s | 97.5 | 94.4 | 748 | 7.7x |
| net TCP request/response, trans/s | 8935 | 8617 | 9855 | 1.10x |
| syscalls, memory, disk seq | | | | 0.94-1.02x |
| disk rand read / write 4K, IOPS | 36.5K / 32.5K | 36.1K / 31.7K | 44.3K / 38.5K | 1.2x |

Nothing in this round touches the disk path. The 4K random disk lines moved by 20% in `new` only, which is twice the noise seen in round 1; treat it as unexplained, not as a gain.

## What bounds the rest, and what is left

- **Random 4K disk I/O is guest CPU.** In `randread` the vCPU thread spends about 94% in guest code (block layer, ext4, interrupt path) and about 6% in the host `pread`. Device-side changes cannot move it; general JIT speed (plan 20's register pinning) would.
- **Sequential writes end in the guest's FLUSH, which `sync_data` turns into `F_FULLFSYNC` on macOS** (about an eighth of the emulator thread's samples in the disk run). Durable across host power loss, and the reason write throughput is about half of read. A per-disk opt-out (like QEMU's `cache=unsafe`) would be the user's call; not done.
- **Bulk TCP through usernet is now one 64 KiB window per round trip.** 652 MB/s down and 748 MB/s up are about 10-11K windows per second, the same rate as 1-byte request/response (9.9K/s). Each direction waits a full round trip per 16-bit window, since usernet's SYN-ACK carries no window scale option. Window scaling (RFC 7323) in usernet is the next step: advertise a scale in the SYN-ACK, accept the guest's, and let `to_host` and the in-flight data grow past 64 KiB. Not profiled this round.
- **Supervisor accesses to user pages are never cached while SMAP is on**, so every `copy_to_user`/`copy_from_user` access walks the page tables (`Cpu::translate` is the top emulator function in the syscall profiles). A TLB half for "CPL 0 with AC set" would fix it, but every EFLAGS.AC writer (popf, iret, sysret, interrupt delivery, the inline `stac`/`clac`) would have to switch `tlb_active_base`, and a miss there silently weakens the guest's SMAP.
- Still through the interpreter on the syscall path: `syscall`, `sysret`, `sti` and `rdtsc` (one each per syscall). `rdtsc` costs a `clock_gettime` per read.
- With four vCPUs, random 4K disk I/O is less than half the one-vCPU rate (13K against 29K IOPS) before and after this round: the completion interrupt and the wakeup of the waiting task cross vCPU threads. Not looked into.
- `--verify` diverges at the first frame the host network sends (the twin's NIC has an empty `NetHandle`, as with typed input); verify boots with `--remove-network 0`. With usernet the two NICs now also differ in the features they offer (the twin's handle declares no offloads), so the divergence comes earlier, at the driver's feature read.
