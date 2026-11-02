//! The emulator as a WebAssembly module with a plain C ABI (no wasm-bindgen, so building needs
//! only `cargo`). `www/worker.js` loads it in a Web Worker, because `uw_run` blocks until the
//! guest exits and would freeze a page.
//!
//! The library is single-threaded here (see `freestanding` in `uemulator/build.rs`): the vCPU
//! runs inside `uw_run` and reaches the outside world only through `MachineConfig::poll`, which
//! calls the `host_*` imports below. A worker cannot receive messages while a synchronous call is
//! running, so keyboard input arrives through a `SharedArrayBuffer` ring that `host_input` drains,
//! and the screen's keyboard and pointer events through a second ring (`host_input_events`).
//! Screen updates go the other way through `host_frame`, one at a time: the page acknowledges
//! each once drawn (`host_frame_ready`), so a busy page gets fewer, larger updates rather than a
//! backlog of messages. Only the tiles that differ from what the page shows go (a
//! `ScreenMirror`), in the library's own pixel format: the page's WebGL shader reorders the
//! channels, so this thread, which is also the guest's, spends nothing converting them. With a network (`uw_set_network`) the guest's NIC is a port on the
//! segment the page's tabs share: the library's `SegmentAgent` switches its frames, and the
//! `host_net_*` imports carry them (`www/segment.js`).
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use uemulator::{
    BYTES_PER_PIXEL, BlockBackend, ClockMode, ConsoleKind, Cursor, Disk, DisplayHandle,
    DisplayMode, EmulatedMachine, ExitReason, InputHandle, JitMode, MachineConfig,
    MemoryKeyValueStore, NetHandle, PointerButton, RawImage, RemoteImage, ScreenMirror,
    SegmentAgent, SegmentDestination, SegmentStatus, StopHandle, Udisk, UdiskOptions, disk_io,
};

const NS_PER_MS: f64 = 1_000_000.0;

/// The vCPU calls the poll hook every few thousand instructions; crossing into JavaScript on each
/// one costs more than it saves, so busy guests are asked for input only every this-many polls.
/// An idle guest is always asked, since it has nothing better to do.
const POLL_INPUT_EVERY: u32 = 16;
/// Largest burst of keyboard bytes handed to the guest per poll.
const INPUT_CHUNK: usize = 256;
const SECTOR_BYTES: u64 = 512;
/// Screen updates go out at most this often: about what a browser paints anyway, and each one
/// is a copy out of the module plus a message to the page.
const FRAME_INTERVAL_MS: f64 = 16.0;
/// A rectangle as `host_frame` passes it: x, y, width, height.
const RECT_WORDS: usize = 4;
/// `host_frame_ready` answers (`FRAME_READY` in `www/protocol.js`): a frame is in flight; the
/// page can take the next; or it can and has lost its picture (a WebGL context loss), so the
/// next is the whole screen.
const FRAME_READY_NOT: i32 = 0;
const FRAME_READY_RESEND: i32 = 2;
/// Input events taken from the page's ring per poll. A record is three `i32`: kind and two
/// arguments (`EVENT_*`, `www/protocol.js` has the same numbers).
const EVENTS_PER_POLL: usize = 64;
const EVENT_WORDS: usize = 3;
/// The longest frame taken from another tab: a guest may raise its MTU to virtio-net's limit of
/// 65535 (no MTU feature is offered), plus the Ethernet header.
const SEGMENT_FRAME_MAX: usize = u16::MAX as usize + 14;
const MAC_BYTES: usize = 6;

// `host_net_send` destinations (`SegmentDestination`); `www/protocol.js` has the same numbers.
const DESTINATION_EVERYONE: i32 = 0;
const DESTINATION_TAB: i32 = 1;
const DESTINATION_SERVER: i32 = 2;

pub const EXIT_SHUTDOWN: i32 = 0;
pub const EXIT_RESET: i32 = 1;
pub const EXIT_TRIPLE_FAULT: i32 = 2;
pub const EXIT_STOPPED: i32 = 3;
pub const EXIT_INTERNAL: i32 = 4;
/// `uw_run` before a successful `uw_create`.
pub const EXIT_NO_MACHINE: i32 = -1;

// `uw_set_firmware` kinds; `www/protocol.js` has the same numbers.
/// No firmware: the kernel is entered directly in 64-bit mode.
pub const FIRMWARE_NONE: i32 = 0;
/// The image is `crates/uemulator-firmware` (`MachineConfig::uefi_firmware`).
pub const FIRMWARE_UEFI: i32 = 1;
/// The library's BIOS boots the first disk's boot sector (`MachineConfig::boot_disk`); no image.
pub const FIRMWARE_BIOS: i32 = 2;
/// A real-mode image placed to end at 1 MiB (`MachineConfig::firmware`).
pub const FIRMWARE_REAL_MODE: i32 = 3;

// `uw_create` option bits; `www/protocol.js` has the same values.
pub const OPTION_JIT: u32 = 1 << 0;
pub const OPTION_ACPI: u32 = 1 << 1;
pub const OPTION_IOMMU: u32 = 1 << 2;
pub const OPTION_RNG: u32 = 1 << 3;
/// The guest's console is the 16550 (`ttyS0`) instead of virtio-console (`hvc0`).
pub const OPTION_SERIAL_CONSOLE: u32 = 1 << 4;
/// A guest reset restarts the machine instead of ending the run.
pub const OPTION_REBOOT: u32 = 1 << 5;
/// The command line given to `uw_create` replaces the library's default instead of being
/// appended to it.
pub const OPTION_REPLACE_CMDLINE: u32 = 1 << 6;
/// The library's `debug` mode: on exit, a state dump with the JIT's counters and the
/// instructions that went through the interpreter goes to `host_log`.
pub const OPTION_DEBUG: u32 = 1 << 7;
/// A screen (`DisplayMode::Framebuffer`), shown on the page's canvas.
pub const OPTION_DISPLAY: u32 = 1 << 8;
/// A virtio keyboard and tablet fed from the canvas. The page always says, with or without a
/// screen (`MachineConfig::input`).
pub const OPTION_INPUT: u32 = 1 << 9;
/// With `OPTION_DISPLAY`: the screen is the guest's console (`console=tty0` last), not the
/// terminal (`MachineConfig::console_on_screen`).
pub const OPTION_SCREEN_CONSOLE: u32 = 1 << 10;

// Input event kinds in the page's event ring, as `www/protocol.js` numbers them.
/// A key by USB HID usage; then pressed (1) or released (0).
const EVENT_KEY: i32 = 1;
/// The pointer to (x, y) in screen pixels.
const EVENT_POINTER_MOVE: i32 = 2;
/// A pointer button (0 left, 1 right, 2 middle, 3 back, 4 forward); then pressed or released.
const EVENT_POINTER_BUTTON: i32 = 3;
/// The wheel by a number of clicks, positive away from the user.
const EVENT_WHEEL: i32 = 4;
/// Let go of every key and button (the canvas lost the focus).
const EVENT_RELEASE_ALL: i32 = 5;
/// Ask the guest for a screen mode of (width, height).
const EVENT_REQUEST_SIZE: i32 = 6;

// `host_cursor` kinds.
const CURSOR_HOST: i32 = 0;
const CURSOR_HIDDEN: i32 = 1;
const CURSOR_IMAGE: i32 = 2;

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// Guest console output; JavaScript batches it and posts it to the page.
    fn host_output(ptr: *const u8, len: usize);
    /// Copies up to `cap` pending keystroke bytes to `ptr` and returns how many; never blocks.
    fn host_input(ptr: *mut u8, cap: usize) -> usize;
    /// Writes the page's requested terminal size (`cols`, `rows` as two little-endian `u16`) to
    /// `ptr` and returns 1 if it changed since the last call.
    fn host_resize(ptr: *mut u16) -> i32;
    /// Non-zero once the page asked to stop the machine.
    fn host_should_stop() -> i32;
    /// The guest is halted with nothing scheduled: JavaScript may sleep for a moment.
    fn host_idle();
    /// Milliseconds on a monotonic clock (`performance.now()`), fractional.
    fn host_now_ms() -> f64;
    /// Fills `ptr..ptr+len` with cryptographically random bytes (`crypto.getRandomValues`).
    fn host_random(ptr: *mut u8, len: usize);
    /// Emulator diagnostics (the library's `log` hook).
    fn host_log(ptr: *const u8, len: usize);
    /// Fills `ptr..ptr+len` with bytes `offset..offset+len` of disk source `source` (a URL read
    /// with a synchronous range request, or a `File` read with `FileReaderSync`). Blocks until
    /// they arrive. Returns 0, or non-zero when the read failed.
    fn host_disk_read(source: i32, offset: f64, ptr: *mut u8, len: usize) -> i32;
    /// Copies up to `capacity` input event records (`EVENT_WORDS` `i32` each) to `ptr` and
    /// returns how many; never blocks.
    fn host_input_events(ptr: *mut i32, capacity: usize) -> usize;
    /// A `FRAME_READY_*`: whether the page has drawn the last frame and can take the next.
    fn host_frame_ready() -> i32;
    /// A screen update: the screen is `screen_width` x `screen_height`; `rect_count`
    /// rectangles of it, each four `u32` (x, y, width, height) at `rects`, changed. The whole
    /// screen is at `pixels`, rows `stride` bytes apart, blue, green, red, unused per pixel; the
    /// rectangles' rows must be copied out before this returns. The page owes an
    /// acknowledgement before the next.
    fn host_frame(
        screen_width: u32,
        screen_height: u32,
        rects: *const u32,
        rect_count: usize,
        pixels: *const u8,
        stride: usize,
    );
    /// The next frame another tab sent this one: copies it to `ptr` (`capacity` bytes at most; a
    /// longer one is skipped), the sending tab's number to `from`, and returns its length, or 0
    /// when none is waiting. Never blocks.
    fn host_net_recv(ptr: *mut u8, capacity: usize, from: *mut u32) -> usize;
    /// Queues a frame for the other tabs: `destination` is a `DESTINATION_*`, `tab` the tab for
    /// `DESTINATION_TAB`. Sent at the next `host_net_flush`.
    fn host_net_send(ptr: *const u8, len: usize, destination: i32, tab: u32);
    fn host_net_flush();
    /// Non-zero while this tab holds the segment's server lock.
    fn host_net_serving() -> i32;
    /// The segment status changed: the guest's address (big-endian, 0 for none), the number of
    /// other tabs and whether this one serves.
    fn host_net_status(address: u32, peers: u32, serving: i32);
    /// The pointer image: `CURSOR_*` kind, and for `CURSOR_IMAGE` its size, hotspot and RGBA
    /// pixels (straight alpha).
    fn host_cursor(
        kind: i32,
        width: u32,
        height: u32,
        hot_x: u32,
        hot_y: u32,
        ptr: *const u8,
        len: usize,
    );
}

// `host_jit_compile` and `host_jit_release` are imported by the library (`jit/wasm.rs`), from the
// same `host` module: see `www/jit_host.js`.

thread_local! {
    static MACHINE: RefCell<Option<EmulatedMachine>> = const { RefCell::new(None) };
    /// A copy of the machine's stop handle. `uw_run` takes the machine out of `MACHINE` and the
    /// poll hook runs inside it, so the hook cannot reach the machine itself.
    static STOP: RefCell<Option<StopHandle>> = const { RefCell::new(None) };
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// Disks `uw_add_disk` made, for the next `uw_create` to attach.
    static PENDING_DISKS: RefCell<Vec<Disk>> = const { RefCell::new(Vec::new()) };
    /// What `uw_set_firmware` chose for the next `uw_create`: a `FIRMWARE_*` kind and its image.
    static PENDING_FIRMWARE: RefCell<(i32, Vec<u8>)> = const { RefCell::new((FIRMWARE_NONE, Vec::new())) };
    /// What `uw_set_hypervisor` gave the next `uw_create`: the image and its command line.
    static PENDING_HYPERVISOR: RefCell<Option<(Vec<u8>, String)>> = const { RefCell::new(None) };
    /// The machine's screen and input handles, for the poll hook (like `STOP`).
    static SCREEN: RefCell<Option<ScreenLink>> = const { RefCell::new(None) };
    /// What `uw_set_network` set up for the next `uw_create`: the NIC's address and its agent.
    static PENDING_NETWORK: RefCell<Option<([u8; MAC_BYTES], SegmentAgent)>> = const { RefCell::new(None) };
    /// The running machine's port on the segment, for the poll hook.
    static NETWORK: RefCell<Option<NetLink>> = const { RefCell::new(None) };
}

/// What the poll hook keeps between calls to move frames between the guest's NIC, its agent and
/// the other tabs.
struct NetLink {
    net: NetHandle,
    agent: SegmentAgent,
    frame: Vec<u8>,
    reported: Option<SegmentStatus>,
}

/// What the poll hook keeps between calls to feed the page the screen and the guest the input.
struct ScreenLink {
    display: Option<DisplayHandle>,
    input: Option<InputHandle>,
    /// What the page shows; empty until the first (whole-screen) frame.
    mirror: ScreenMirror,
    sent_cursor: u64,
    last_frame_ms: f64,
    /// The rectangles of the frame being sent, `RECT_WORDS` `u32` each, for `host_frame`.
    rects: Vec<u32>,
}

fn set_error(message: String) {
    LAST_ERROR.with(|error| *error.borrow_mut() = message);
}

/// Reserves `len` bytes in the module's memory for the page to fill (kernel and initrd images,
/// the command line). Pair with `uw_create`, which takes ownership, or `uw_free`.
#[unsafe(no_mangle)]
pub extern "C" fn uw_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// # Safety
/// `ptr` and `len` must be exactly what a previous `uw_alloc(len)` returned and was given.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// Takes ownership of a buffer from `uw_alloc(len)` whose first `len` bytes the page filled.
///
/// # Safety
/// As for `uw_free`, and the buffer must not be used again.
unsafe fn take_buffer(ptr: *mut u8, len: usize) -> Vec<u8> {
    let mut buffer = unsafe { Vec::from_raw_parts(ptr, len, len.max(1)) };
    buffer.truncate(len);
    buffer
}

/// A disk for the next `uw_create`: the page's disk source `source` (`length` bytes; see
/// `host_disk_read`), raw or udisk, read on demand. Disks attach in call order (`vda`, `vdb`, ...).
/// With `read_only` 0 an in-memory udisk overlay takes the guest's writes, which are gone when
/// the page stops the machine; otherwise the guest sees a read-only disk (the virtio-blk RO
/// feature), which is what a CD image should be: an installer that finds its medium writable may
/// offer it as an install target. Returns 0, or 1 with the message in `uw_error_*`.
#[unsafe(no_mangle)]
pub extern "C" fn uw_add_disk(source: i32, length: f64, read_only: i32) -> i32 {
    match streamed_disk(source, length as u64, read_only != 0) {
        Ok(disk) => {
            PENDING_DISKS.with(|disks| disks.borrow_mut().push(disk));
            0
        }
        Err(message) => {
            set_error(message);
            1
        }
    }
}

fn streamed_disk(source: i32, length: u64, read_only: bool) -> Result<Disk, String> {
    if length == 0 {
        return Err("the disk image is empty".into());
    }
    let mut remote = RemoteImage::new(length, move |offset, buf: &mut [u8]| {
        match unsafe { host_disk_read(source, offset as f64, buf.as_mut_ptr(), buf.len()) } {
            0 => Ok(()),
            _ => Err(disk_io::Error::other("reading the disk image failed")),
        }
    });
    let base: Box<dyn BlockBackend> = match Udisk::probe(&mut remote).map_err(|e| e.to_string())? {
        Some(info) if info.backing.is_some() => {
            return Err(
                "the disk image is an overlay; the page cannot follow it to its base. \
                        Flatten it first with `uemulator disk convert`"
                    .into(),
            );
        }
        Some(_) => Box::new(Udisk::open(remote, None).map_err(|e| e.to_string())?),
        None => Box::new(RawImage::new(remote)),
    };
    // `RemoteImage` is read-only, and so is anything over it.
    if read_only {
        return Ok(Disk::from_backend(base));
    }
    let mut id = [0u8; 16];
    unsafe { host_random(id.as_mut_ptr(), id.len()) };
    let size = base.len().div_ceil(SECTOR_BYTES) * SECTOR_BYTES;
    let overlay = Udisk::create(Vec::<u8>::new(), &UdiskOptions::new(size, id), Some(base))
        .map_err(|e| e.to_string())?;
    Ok(Disk::from_backend(overlay))
}

/// Chooses the firmware for the next `uw_create`: a `FIRMWARE_*` kind and, for `FIRMWARE_UEFI`
/// and `FIRMWARE_REAL_MODE`, its image (`len == 0` with the others). Takes ownership of the
/// buffer. Without a call the kernel is booted directly. Returns 0, or 1 with the message in
/// `uw_error_*` for a kind it does not know.
///
/// # Safety
/// `image` must come from `uw_alloc(len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_set_firmware(kind: i32, image: *mut u8, len: usize) -> i32 {
    let image = unsafe { take_buffer(image, len) };
    if !matches!(
        kind,
        FIRMWARE_NONE | FIRMWARE_UEFI | FIRMWARE_BIOS | FIRMWARE_REAL_MODE
    ) {
        set_error(format!("unknown firmware kind {kind}"));
        return 1;
    }
    PENDING_FIRMWARE.with(|firmware| *firmware.borrow_mut() = (kind, image));
    0
}

/// A hypervisor for the next `uw_create`, which boots it through the UEFI firmware with the
/// kernel as its dom0: Xen as `xen.gz`, `xen` or `xen.efi` (`MachineConfig::hypervisor`), and
/// its command line (`cmdline_len == 0`: the library's default, Xen's console on the 16550).
/// Takes ownership of both buffers. Without a call there is no hypervisor.
///
/// # Safety
/// Both pointers must come from `uw_alloc` with the matching length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_set_hypervisor(
    image: *mut u8,
    len: usize,
    cmdline: *mut u8,
    cmdline_len: usize,
) {
    let image = unsafe { take_buffer(image, len) };
    let cmdline = unsafe { take_buffer(cmdline, cmdline_len) };
    let cmdline = String::from_utf8_lossy(&cmdline).into_owned();
    PENDING_HYPERVISOR.with(|hypervisor| *hypervisor.borrow_mut() = Some((image, cmdline)));
}

/// Gives the next `uw_create` a virtio-net NIC with address `mac` (6 bytes) on the segment the
/// page's tabs share, as tab `tab` (`www/segment.js` picks both at random). Takes ownership of
/// the buffer. Returns 0, or 1 with the message in `uw_error_*` for an address a tab may not use.
///
/// # Safety
/// `mac` must come from `uw_alloc(len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_set_network(mac: *mut u8, len: usize, tab: u32) -> i32 {
    let mac = unsafe { take_buffer(mac, len) };
    let Ok(mac) = <[u8; MAC_BYTES]>::try_from(mac.as_slice()) else {
        set_error(format!("a MAC address is {MAC_BYTES} bytes, not {len}"));
        return 1;
    };
    match SegmentAgent::new(mac, tab, now_ms()) {
        Ok(agent) => {
            PENDING_NETWORK.with(|network| *network.borrow_mut() = Some((mac, agent)));
            0
        }
        Err(error) => {
            set_error(error.to_string());
            1
        }
    }
}

fn now_ms() -> u64 {
    unsafe { host_now_ms() as u64 }
}

/// Builds the machine and loads the kernel. Returns 0, or 1 with the message in `uw_error_*`.
/// `kernel_len == 0` means no kernel (the firmware boots a disk), `initrd_len == 0` no initramfs.
/// The command line is appended to the library's default (which follows `OPTION_ACPI` and
/// `OPTION_SERIAL_CONSOLE`), or replaces it with `OPTION_REPLACE_CMDLINE`. `options` is a set of
/// `OPTION_*` bits. Attaches the disks `uw_add_disk` made, in order, and boots through the
/// firmware `uw_set_firmware` chose. Takes ownership of all three buffers either way.
///
/// # Safety
/// Every pointer must come from `uw_alloc` with the matching length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn uw_create(
    kernel: *mut u8,
    kernel_len: usize,
    initrd: *mut u8,
    initrd_len: usize,
    cmdline: *mut u8,
    cmdline_len: usize,
    memory_mib: u32,
    cpus: u32,
    unix_time: f64,
    cols: u16,
    rows: u16,
    options: u32,
) -> i32 {
    let kernel = unsafe { take_buffer(kernel, kernel_len) };
    let initrd = unsafe { take_buffer(initrd, initrd_len) };
    let cmdline = unsafe { take_buffer(cmdline, cmdline_len) };
    let option = |bit: u32| options & bit != 0;
    // Seeds rdrand and the firmware's EFI_RNG_PROTOCOL: without it every page load would hand
    // KASLR the library's fixed default seed.
    let mut random_seed = [0u8; 8];
    unsafe { host_random(random_seed.as_mut_ptr(), random_seed.len()) };

    let mut config = MachineConfig::new()
        .kernel(kernel)
        .memory_mib(memory_mib as usize)
        // Without a `thread_spawner` the vCPUs take turns on this thread, which the JIT needs
        // anyway (its function table belongs to this instance; `docs/plans/13-web.md`).
        .cpus(cpus as usize)
        // Blocks are compiled to WebAssembly modules the page instantiates; without the page's
        // `host_jit_*` imports working, `Auto` falls back to the interpreter.
        .jit(if option(OPTION_JIT) {
            JitMode::Auto
        } else {
            JitMode::Off
        })
        // The IOMMU is found through the ACPI IVRS table, so it is useless without `acpi`;
        // `EmulatedMachine::new` leaves that combination to the caller.
        .acpi(option(OPTION_ACPI))
        .iommu(option(OPTION_IOMMU))
        .rng(option(OPTION_RNG))
        .reboot(option(OPTION_REBOOT))
        .debug(option(OPTION_DEBUG))
        // Guest time follows the page's clock. A virtual clock jumps over every idle wait and
        // otherwise crawls at one nanosecond per interpreted instruction.
        .clock(ClockMode::Realtime)
        .host_clock(|| (unsafe { host_now_ms() } * NS_PER_MS) as u64)
        .unix_time(unix_time as u64)
        .random_seed(u64::from_le_bytes(random_seed))
        .log(|message| unsafe { host_log(message.as_ptr(), message.len()) })
        // virtio-rng feeds the guest's entropy pool from the browser's.
        .entropy(|bytes| unsafe { host_random(bytes.as_mut_ptr(), bytes.len()) })
        .poll(poll);
    if option(OPTION_SERIAL_CONSOLE) {
        config = config.console(ConsoleKind::Serial);
    }
    if option(OPTION_DISPLAY) {
        config = config
            .display(DisplayMode::Framebuffer)
            .console_on_screen(option(OPTION_SCREEN_CONSOLE));
    }
    config = config.input(option(OPTION_INPUT));
    if initrd_len != 0 {
        config = config.initrd(initrd);
    }
    // No offloads are declared: the frames go to other guests, which need real checksums and
    // MTU-sized frames.
    let network = PENDING_NETWORK.with(|network| network.take());
    if let Some((mac, _)) = &network {
        config = config.net(*mac);
    }
    for disk in PENDING_DISKS.with(|disks| disks.take()) {
        config = config.disk(disk);
    }
    let cmdline = String::from_utf8_lossy(&cmdline).into_owned();
    if option(OPTION_REPLACE_CMDLINE) {
        config = config.cmdline(cmdline);
    } else if !cmdline.is_empty() {
        config = config.cmdline_append(cmdline);
    }
    if let Some((image, cmdline)) = PENDING_HYPERVISOR.with(|hypervisor| hypervisor.take()) {
        config = config.hypervisor(image);
        if !cmdline.is_empty() {
            config = config.hypervisor_cmdline(cmdline);
        }
    }
    let (firmware, image) = PENDING_FIRMWARE.with(|firmware| firmware.take());
    config = match firmware {
        // The variables (`BootOrder`, what `efibootmgr` writes) then survive a guest reset,
        // though not the page: nothing here is kept across loads.
        FIRMWARE_UEFI => config
            .uefi_firmware(image)
            .key_value_store(MemoryKeyValueStore::new()),
        FIRMWARE_BIOS => config.boot_disk(true),
        FIRMWARE_REAL_MODE => config.firmware(image),
        _ => config,
    };
    let mut machine = match EmulatedMachine::new(config) {
        Ok(machine) => machine,
        Err(error) => {
            set_error(error.to_string());
            return 1;
        }
    };
    let console = machine.console();
    console.resize(cols, rows);
    console.on_output(|bytes| unsafe { host_output(bytes.as_ptr(), bytes.len()) });
    if let Err(error) = machine.start() {
        set_error(error.to_string());
        return 1;
    }
    STOP.with(|stop| *stop.borrow_mut() = Some(machine.stop_handle()));
    let (display, input) = (machine.display(), machine.input());
    SCREEN.with(|screen| {
        *screen.borrow_mut() = (display.is_some() || input.is_some()).then(|| ScreenLink {
            display,
            input,
            mirror: ScreenMirror::new(),
            sent_cursor: 0,
            last_frame_ms: f64::NEG_INFINITY,
            rects: Vec::new(),
        });
    });
    NETWORK.with(|slot| {
        *slot.borrow_mut() = network.zip(machine.net()).map(|((_, agent), net)| NetLink {
            net,
            agent,
            frame: vec![0; SEGMENT_FRAME_MAX],
            reported: None,
        });
    });
    MACHINE.with(|slot| *slot.borrow_mut() = Some(machine));
    0
}

fn poll(console: &uemulator::ConsoleHandle, idle: bool) {
    static POLLS: AtomicU32 = AtomicU32::new(0);
    if !idle
        && !POLLS
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(POLL_INPUT_EVERY)
    {
        return;
    }
    let mut chunk = [0u8; INPUT_CHUNK];
    loop {
        let count = unsafe { host_input(chunk.as_mut_ptr(), chunk.len()) };
        if count > 0 {
            console.write(&chunk[..count]);
        }
        if count < chunk.len() {
            break;
        }
    }
    let mut size = [0u16; 2];
    if unsafe { host_resize(size.as_mut_ptr()) } != 0 {
        console.resize(size[0], size[1]);
    }
    SCREEN.with(|screen| {
        if let Some(link) = screen.borrow_mut().as_mut() {
            link.deliver_input();
            link.send_screen();
        }
    });
    NETWORK.with(|network| {
        if let Some(link) = network.borrow_mut().as_mut() {
            link.pump();
        }
    });
    if unsafe { host_should_stop() } != 0 {
        STOP.with(|stop| {
            if let Some(stop) = &*stop.borrow() {
                stop.stop();
            }
        });
    }
    if idle {
        unsafe { host_idle() };
    }
}

impl ScreenLink {
    /// Hands the page's keyboard and pointer events to the guest.
    fn deliver_input(&mut self) {
        let mut records = [0i32; EVENTS_PER_POLL * EVENT_WORDS];
        loop {
            let count = unsafe { host_input_events(records.as_mut_ptr(), EVENTS_PER_POLL) };
            for [kind, first, second] in records[..count * EVENT_WORDS].as_chunks::<EVENT_WORDS>().0
            {
                self.apply(*kind, *first, *second);
            }
            if count < EVENTS_PER_POLL {
                break;
            }
        }
    }

    fn apply(&self, kind: i32, first: i32, second: i32) {
        const BUTTONS: [PointerButton; 5] = [
            PointerButton::Left,
            PointerButton::Right,
            PointerButton::Middle,
            PointerButton::Side,
            PointerButton::Extra,
        ];
        let non_negative = |value: i32| value.max(0) as u32;
        if kind == EVENT_REQUEST_SIZE {
            if let Some(display) = &self.display {
                display.request_size(non_negative(first), non_negative(second));
            }
            return;
        }
        let Some(input) = &self.input else {
            return;
        };
        // A full queue (a guest that is not reading) drops the event, as a real keyboard
        // controller would.
        match kind {
            EVENT_KEY => {
                if let Ok(usage) = u16::try_from(first) {
                    input.key_usb(usage, second != 0);
                }
            }
            EVENT_POINTER_MOVE => {
                input.pointer_move(non_negative(first), non_negative(second));
            }
            EVENT_POINTER_BUTTON => {
                if let Some(button) = usize::try_from(first).ok().and_then(|i| BUTTONS.get(i)) {
                    input.pointer_button(*button, second != 0);
                }
            }
            EVENT_WHEEL => {
                input.scroll(first);
            }
            EVENT_RELEASE_ALL => input.release_all(),
            _ => {}
        }
    }

    /// Sends what changed on the screen since the page's last frame, once the page has drawn
    /// that one and `FRAME_INTERVAL_MS` has passed; and the pointer image when it changed.
    fn send_screen(&mut self) {
        let Some(display) = &self.display else {
            return;
        };
        let (cursor_generation, cursor) = display.cursor();
        if cursor_generation != self.sent_cursor {
            self.sent_cursor = cursor_generation;
            send_cursor(&cursor);
        }
        let now = unsafe { host_now_ms() };
        if now - self.last_frame_ms < FRAME_INTERVAL_MS {
            return;
        }
        match unsafe { host_frame_ready() } {
            FRAME_READY_NOT => return,
            FRAME_READY_RESEND => self.mirror = ScreenMirror::new(),
            _ => {}
        }
        // A new mirror syncs as the whole screen: the page starts with no picture at all, even
        // when the guest has not drawn anything yet.
        let Some(rects) = display.sync_mirror(&mut self.mirror) else {
            return;
        };
        if rects.is_empty() {
            // Flushed, but every pixel is what the page already has.
            return;
        }
        self.last_frame_ms = now;
        self.rects.clear();
        for rect in rects {
            self.rects
                .extend_from_slice(&[rect.x, rect.y, rect.width, rect.height]);
        }
        let (width, height) = self.mirror.size();
        unsafe {
            host_frame(
                width,
                height,
                self.rects.as_ptr(),
                self.rects.len() / RECT_WORDS,
                self.mirror.pixels().as_ptr(),
                self.mirror.stride(),
            )
        };
    }
}

impl NetLink {
    /// Moves every frame waiting on either side through the agent, then sends what it produced.
    fn pump(&mut self) {
        let now = now_ms();
        self.agent
            .set_serving(unsafe { host_net_serving() } != 0, now);
        loop {
            let mut from = 0u32;
            let len =
                unsafe { host_net_recv(self.frame.as_mut_ptr(), self.frame.len(), &mut from) };
            if len == 0 {
                break;
            }
            self.agent.from_segment(from, &self.frame[..len], now);
        }
        while let Some(frame) = self.net.try_recv_frame() {
            self.agent.from_guest(&frame, now);
        }
        self.agent.tick(now);
        // A full receive queue (a guest whose driver is not up) drops the frame, as a NIC
        // without receive buffers would.
        for frame in self.agent.take_for_guest() {
            self.net.send_frame(&frame);
        }
        let outgoing = self.agent.take_for_segment();
        if !outgoing.is_empty() {
            for (destination, frame) in &outgoing {
                let (destination, tab) = match destination {
                    SegmentDestination::Everyone => (DESTINATION_EVERYONE, 0),
                    SegmentDestination::Tab(tab) => (DESTINATION_TAB, *tab),
                    SegmentDestination::Server => (DESTINATION_SERVER, 0),
                };
                unsafe { host_net_send(frame.as_ptr(), frame.len(), destination, tab) };
            }
            unsafe { host_net_flush() };
        }
        let status = self.agent.status();
        if self.reported != Some(status) {
            self.reported = Some(status);
            let address = status.address.map_or(0, u32::from_be_bytes);
            unsafe { host_net_status(address, status.peers as u32, i32::from(status.serving)) };
        }
    }
}

fn send_cursor(cursor: &Cursor) {
    let none = core::ptr::null();
    match cursor {
        Cursor::Host => unsafe { host_cursor(CURSOR_HOST, 0, 0, 0, 0, none, 0) },
        Cursor::Hidden => unsafe { host_cursor(CURSOR_HIDDEN, 0, 0, 0, 0, none, 0) },
        Cursor::Image(image) => {
            // Blue, green, red, alpha to red, green, blue, alpha; the alpha stays.
            let mut pixels = image.pixels.clone();
            for pixel in pixels.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
                pixel.swap(0, 2);
            }
            unsafe {
                host_cursor(
                    CURSOR_IMAGE,
                    image.width,
                    image.height,
                    image.hot_x,
                    image.hot_y,
                    pixels.as_ptr(),
                    pixels.len(),
                )
            };
        }
    }
}

/// Runs the guest on this thread until it exits and returns an `EXIT_*` code. For
/// `EXIT_INTERNAL` the message is in `uw_error_*`. The machine is gone afterwards; build a new one
/// to boot again.
#[unsafe(no_mangle)]
pub extern "C" fn uw_run() -> i32 {
    let Some(mut machine) = MACHINE.with(|slot| slot.borrow_mut().take()) else {
        return EXIT_NO_MACHINE;
    };
    let reason = machine.wait();
    STOP.with(|slot| *slot.borrow_mut() = None);
    SCREEN.with(|slot| *slot.borrow_mut() = None);
    NETWORK.with(|slot| *slot.borrow_mut() = None);
    match reason {
        ExitReason::Shutdown => EXIT_SHUTDOWN,
        ExitReason::Reset => EXIT_RESET,
        ExitReason::TripleFault => EXIT_TRIPLE_FAULT,
        ExitReason::Stopped => EXIT_STOPPED,
        ExitReason::Internal(message) => {
            set_error(message);
            EXIT_INTERNAL
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn uw_error_ptr() -> *const u8 {
    LAST_ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn uw_error_len() -> usize {
    LAST_ERROR.with(|error| error.borrow().len())
}
