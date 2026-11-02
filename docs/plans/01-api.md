# Public API

**Status:** implemented. Where this and `crates/uemulator/src/lib.rs` differ, the code is authoritative.

Small surface; everything else is `pub(crate)`.

```rust
pub struct MachineConfig { /* Default-able builder */ }
impl MachineConfig {
    pub fn new() -> Self;
    pub fn kernel(self, bzimage: impl Into<Vec<u8>>) -> Self;
    pub fn initrd(self, data: impl Into<Vec<u8>>) -> Self;
    pub fn cmdline(self, s: impl Into<String>) -> Self;
    pub fn hypervisor(self, image: impl Into<Vec<u8>>) -> Self;  // Xen (xen.gz, xen or xen.efi), booted by uefi_firmware with the kernel as dom0 (docs/plans/18-xen-dom0.md)
    pub fn hypervisor_cmdline(self, s: impl Into<String>) -> Self; // default DEFAULT_HYPERVISOR_CMDLINE
    pub fn memory_mib(self, n: usize) -> Self;      // default 512
    pub fn jit(self, mode: JitMode) -> Self;        // Auto (default) | Off | On
    pub fn clock(self, mode: ClockMode) -> Self;    // Virtual (default) | Realtime
    pub fn disk(self, d: Disk) -> Self;                 // raw image, virtio-blk; call again for more disks (optional)
    pub fn net(self, mac: [u8; 6]) -> Self;             // virtio-net; frames via EmulatedMachine::nets()/net(); call again for more NICs (optional)
    pub fn console(self, k: ConsoleKind) -> Self;       // Virtio (default, resizable hvc0, repeatable) | Serial (ttyS0, once); first = primary; handles via EmulatedMachine::consoles()
    pub fn reboot(self, on: bool) -> Self;              // true (default): guest reset restarts the machine in place; false: ExitReason::Reset
    pub fn log(self, f: impl Fn(&str) + Send + 'static) -> Self;
    pub fn cpus(self, n: usize) -> Self;                // 1 (default) to 64 vCPUs
    pub fn smp(self, mode: SmpMode) -> Self;            // Auto (default): a host thread per vCPU when possible | Timeshare | Threads (error if impossible)
    pub fn thread_spawner(self, spawn: impl Fn(Box<dyn FnOnce() + Send>) + Send + Sync + 'static) -> Self; // freestanding hosts: how to start a thread
}

pub struct EmulatedMachine { .. }
impl EmulatedMachine {
    pub fn new(cfg: MachineConfig) -> Result<Self, Error>;   // validates + loads kernel, no execution
    pub fn console(&self) -> ConsoleHandle;                  // cloneable, Send + Sync
    pub fn start(&mut self) -> Result<(), Error>;            // spawns the vCPU thread, returns immediately
    pub fn wait(&mut self) -> ExitReason;                    // blocks until the guest exits
    pub fn stop(&self);                                      // callable from any thread via StopHandle
    pub fn stop_handle(&self) -> StopHandle;
}

pub enum ExitReason { Shutdown, Reset, TripleFault, Stopped, Internal(String) }

// Graphical console (docs/plans/24-display-input.md); screen and input are chosen apart
MachineConfig::display(DisplayMode::Framebuffer)    // None (default) | Framebuffer: virtio-gpu (+ the UEFI GOP framebuffer)
MachineConfig::input(bool)                          // virtio keyboard + tablet; default: exactly when there is a screen
MachineConfig::console_on_screen(bool)              // default true: the screen is /dev/console (console=tty0 last)
MachineConfig::display_size(width, height)          // default 1280x800; without a screen, the pointer area
EmulatedMachine::display() -> Option<DisplayHandle> // size, generation, frame, update_since, sync_mirror (ScreenMirror), wait_for_update, on_update, request_size, cursor
EmulatedMachine::input() -> Option<InputHandle>     // key (Linux code), key_usb, pointer_move, pointer_button, scroll, release_all, pointer_area

// Guest 3D (docs/plans/25-gpu-3d.md): a renderer outside the library behind the virtio-gpu
GpuLink::new(Vec<GpuCapset>, GpuLimits) -> GpuLink  // cloneable, Send + Sync; capsets answer GET_CAPSET*, limits bound RESOURCE_CREATE_3D
MachineConfig::gpu(GpuLink)                         // offers VIRGL; needs a display, refused with verify_jit
GpuLink::take_commands() -> Vec<GpuCommand>         // in order: Reset, contexts, resources, Write/ReadResource (packed boxes), Submit (virgl stream, unparsed), SetScanout, Flush, Fence
GpuLink::on_commands(f)                             // called after the device queues commands
GpuLink::complete_fence(sequence)                   // everything up to that Fence is done; the device releases held responses
GpuLink::complete_read(token, bytes)                // a ReadResource's packed box, before any later fence

pub struct ConsoleHandle { .. }
impl ConsoleHandle {
    pub fn write(&self, bytes: &[u8]);                       // host → guest (serial RX)
    pub fn read(&self, buf: &mut [u8], timeout: Option<Duration>) -> usize; // guest → host (serial TX)
    pub fn on_output(&self, f: impl Fn(&[u8]) + Send + 'static); // push alternative to read
    pub fn resize(&self, cols: u16, rows: u16);              // delivered to the guest with ConsoleKind::Virtio
    pub fn size(&self) -> (u16, u16);
}

pub struct NetHandle { .. }                                  // one per NIC, cloneable, Send + Sync
impl NetHandle {
    pub fn send_frame(&self, frame: &[u8]) -> bool;          // host → guest; false when the queue is full; wakes the halted vCPU (or the device thread) at once
    pub fn recv_frame(&self, timeout: Duration) -> Option<Vec<u8>>; // guest → host, waiting once
    pub fn try_recv_frame(&self) -> Option<Vec<u8>>;
    pub fn on_guest_frame(&self, f: impl Fn() + Send + Sync + 'static); // called after each guest frame is queued, for a host that waits on sockets too (the CLI's usernet `poll(2)`)
    pub fn set_offloads(&self, offloads: NetOffloads);       // before start: what the host handles in the guest's place; virtio-net offers the matching offloads only then
    pub fn send_tcp_segment(&self, frame: &[u8], segment_size: u16) -> bool; // host → guest TCP/IPv4 frame, checksum field = pseudo-header sum, payload may exceed segment_size (the MSS)
    pub fn guest_takes_tcp_segments(&self) -> bool;          // the driver negotiated GUEST_CSUM + GUEST_TSO4 + MRG_RXBUF, so send_tcp_segment frames reach it whole
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NetOffloads {
    pub takes_guest_segments: bool,  // host takes 64 KiB TCP segments and partial checksums from the guest (CSUM, HOST_TSO4)
    pub sends_large_segments: bool,  // host sends large segments (GUEST_CSUM, GUEST_TSO4, MRG_RXBUF)
}

// One machine's port on an Ethernet segment shared with machines the host reaches only by
// messages (the web page's tabs; docs/plans/13-web.md "The shared network"). Pure protocol: no
// clock, no I/O; the host moves frames between it, the NetHandle and its transport.
pub struct SegmentAgent { .. }                               // Send + Sync
impl SegmentAgent {
    pub fn new(mac: [u8; 6], tab: u32, now_ms: u64) -> Result<Self, Error>; // the NIC's MAC (locally administered unicast) and this host's number on the transport
    pub fn set_serving(&mut self, serving: bool, now_ms: u64); // the host's election (one at a time) made this agent the DHCP/ARP/ping server for 10.0.2.2
    pub fn from_guest(&mut self, frame: &[u8], now_ms: u64);
    pub fn from_segment(&mut self, from_tab: u32, frame: &[u8], now_ms: u64);
    pub fn tick(&mut self, now_ms: u64);                     // announcements, ageing
    pub fn take_for_guest(&mut self) -> Vec<Vec<u8>>;
    pub fn take_for_segment(&mut self) -> Vec<(SegmentDestination, Vec<u8>)>;
    pub fn status(&self) -> SegmentStatus;
}
pub enum SegmentDestination { Everyone, Tab(u32), Server }  // Copy
pub struct SegmentStatus { pub address: Option<[u8; 4]>, pub peers: usize, pub serving: bool } // Copy
```

## Rules

- With `cpus(n)` above 1 and the realtime clock the machine runs one thread per vCPU (`docs/plans/22-smp-threads.md`): `std::thread` hosted, `thread_spawner` (and `host_clock`) on a freestanding host, else the vCPUs take turns on the thread that runs the machine. The thread that called `start`/`wait` ticks the devices. Everything an embedder hands in (log, poll, disk backends, console callbacks) is called from any of these threads, never concurrently with itself for one device.

- `new` does all fallible setup (parse bzImage, allocate RAM, probe JIT) so `start` failing is rare.
- Guest shutdown is a value (`ExitReason`), never a process exit. `hlt` with IF=0 (nothing can wake it; with `acpi=off` this is how Linux poweroff/halt ends, `cli; hlt` in `stop_this_cpu`) is `Shutdown`; a triple fault is `TripleFault`, so automated runs terminate. A reset request (0x64←0xFE, 0x92 bit 0, 0xCF9 RST_CPU) wins over a `hlt` right after it.
- Reset restarts the machine in the library by default (`reboot(true)`): the vCPU thread rebuilds the CPU and every device (PIC, PIT, CMOS, UART, PCI, virtio) from power-on state, reloads kernel, initrd and command line into freshly zeroed RAM, and runs again, so a Linux `panic=-1` loops forever until `stop()`. Kept across the restart: the `ConsoleHandle`/`NetHandle` (devices are rebuilt around the same queues), the stop flag, the disk backend (handed back from the old virtio-blk), the poll hook and the log. The JIT starts with an empty code cache (a fresh `Jit` per boot, so translations of the old RAM can never run). The RTC is re-read from the host clock (or `unix_time`), virtual time restarts at 0. `reboot(false)` returns `ExitReason::Reset` instead (CLI `--no-reboot`). A triple fault always ends the run (`TripleFault`), and under `verify_jit` a reset always ends the run with `Reset` (the twin's devices cannot be rebuilt in step). The UEFI hardware-virtualization path has no in-process reboot and returns `Reset`.
- Console is a plain byte stream backed by bounded queues; output with no reader drops oldest beyond a cap rather than blocking the guest.
- Stdio bridging (raw tty, Ctrl-C) lives in `crates/uemulator-cli`, not the library.
- No networking in the library. The CLI downloads with ureq.

## CLI (`crates/uemulator-cli`, binary `uemulator`)

`uemulator advanced boot --kernel K [--initramfs I] [flags]`, `uemulator advanced boot-firmware --firmware F [flags]` and the named machines of `uemulator quick create|edit|boot|reinstall|update-artifacts|show|delete NAME` (and `quick list`) (flags in README). The option groups are shared (`args.rs`): hardware (memory, disks, NICs with their links, consoles, JIT), kernel (command line, reboot; not on firmware), removals, `--create-disk`, and per-run options (`--verify`, `--debug`, `--time`, `--timeout`) that are never saved.

1. State: `$UEMULATOR_HOME` (default `~/.uemulator`) holds `machines/<name>/` with `machine.toml` (the distro to install plus `Settings`: memory, command line, reboot and JIT flags, consoles, disks as absolute paths, NICs with their links), `artifacts/` (the distro's kernel and initramfs) and the images `--create-disk` made (`disk-<n>.img`, sparse, never reused after a detach). The file holds the user's choices only: the distro's defaults (memory, its kernel arguments before the user's `--append`) are applied when the machine boots, so `reinstall --install fedora` picks up Fedora's. A boot with overrides patches a copy and saves nothing. `boot_from` (`auto`, `installer`, `disk`; default `auto`) is saved in the same file: `auto` boots the installer while the first 32 KiB of disk 0 are all zeros and the disk after that (`machines::disk_is_blank`). Booting the disk is `MachineConfig::boot_disk(true)` (see `cpu/bios.rs`): the library's BIOS runs the disk's own boot sector, so the CLI never reads a filesystem. Kernel arguments come from the disk's boot loader. `firmware` (`uefi`, `bios` or `none`; default `uefi`, also for a file without the key) is saved beside it: for `uefi` both the disk boot and the installer boot are `run::run_uefi` with `machines/<name>/nvram.kv` as the variable store (the installer's kernel and initramfs go to the firmware, which boots them like QEMU's `-kernel` under OVMF); `bios` boots the disk through the BIOS and the installer directly; `none` boots the installer directly and refuses to boot a disk. `advanced boot --firmware` takes the same values for a given kernel.
2. Sources: Debian trixie netboot (deb.debian.org), Alpine latest-stable `virt` netboot (dl-cdn.alpinelinux.org; boot.alpinelinux.org only hosts iPXE), Fedora 44 Server pxeboot. Downloads go to `<file>.part`, then `rename`. Fetched on `create` and on a `boot` that finds them missing; `update-artifacts` downloads into `artifacts.new/` and swaps it in only when both files arrived (the Alpine initramfs and modloop must match the kernel); `reinstall` deletes `artifacts/` first.
3. Build config, start machine, pump stdin → console and console → stdout until exit; print `ExitReason`.

Default cmdline (`platform::DEFAULT_CMDLINE`, console `virtio`): `console=hvc0 earlyprintk=serial,ttyS0 acpi=off panic=-1` (`EmulatedMachine` drops `acpi=off` while `MachineConfig::acpi` is on, which is the default; `Platform` hosts have no tables and keep it); `DEFAULT_SERIAL_CMDLINE` swaps in `console=ttyS0` (`--console serial`, and the UEFI hardware-virtualization path whose `Platform` has no virtio). The Alpine virt kernel has no `earlyprintk`, so an Alpine machine adds `earlycon=uart8250,io,0x3f8` to keep the early log.

Debug-profile interpreter is too slow for the decompressor; set `[profile.dev] opt-level = 3` in Cargo.toml and document `--release`.
