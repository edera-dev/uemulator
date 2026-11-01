//! The emulator as a WebAssembly module with a plain C ABI (no wasm-bindgen, so building needs
//! only `cargo`). `www/worker.js` loads it in a Web Worker, because `uw_run` blocks until the
//! guest exits and would freeze a page.
//!
//! The library is single-threaded here (see `freestanding` in `uemulator/build.rs`): the vCPU
//! runs inside `uw_run` and reaches the outside world only through `MachineConfig::poll`, which
//! calls the `host_*` imports below. A worker cannot receive messages while a synchronous call is
//! running, so keyboard input arrives through a `SharedArrayBuffer` ring that `host_input` drains.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use uemulator::{
    BlockBackend, ClockMode, Disk, EmulatedMachine, ExitReason, JitMode, MachineConfig, RawImage,
    RemoteImage, StopHandle, Udisk, UdiskOptions, disk_io,
};

const NS_PER_MS: f64 = 1_000_000.0;

/// The vCPU calls the poll hook every few thousand instructions; crossing into JavaScript on each
/// one costs more than it saves, so busy guests are asked for input only every this-many polls.
/// An idle guest is always asked, since it has nothing better to do.
const POLL_INPUT_EVERY: u32 = 16;
/// Largest burst of keyboard bytes handed to the guest per poll.
const INPUT_CHUNK: usize = 256;
const SECTOR_BYTES: u64 = 512;

pub const EXIT_SHUTDOWN: i32 = 0;
pub const EXIT_RESET: i32 = 1;
pub const EXIT_TRIPLE_FAULT: i32 = 2;
pub const EXIT_STOPPED: i32 = 3;
pub const EXIT_INTERNAL: i32 = 4;
/// `uw_run` before a successful `uw_create`.
pub const EXIT_NO_MACHINE: i32 = -1;

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
/// `host_disk_read`), raw or udisk, read-only and read on demand, under an in-memory udisk
/// overlay that takes the guest's writes. The writes are gone when the page stops the machine.
/// Returns 0, or 1 with the message in `uw_error_*`.
#[unsafe(no_mangle)]
pub extern "C" fn uw_add_disk(source: i32, length: f64) -> i32 {
    match streamed_disk(source, length as u64) {
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

fn streamed_disk(source: i32, length: u64) -> Result<Disk, String> {
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
    let mut id = [0u8; 16];
    unsafe { host_random(id.as_mut_ptr(), id.len()) };
    let size = base.len().div_ceil(SECTOR_BYTES) * SECTOR_BYTES;
    let overlay = Udisk::create(Vec::<u8>::new(), &UdiskOptions::new(size, id), Some(base))
        .map_err(|e| e.to_string())?;
    Ok(Disk::from_backend(overlay))
}

/// Builds the machine and loads the kernel. Returns 0, or 1 with the message in `uw_error_*`.
/// `initrd_len == 0` means no initramfs, `cmdline_len == 0` the library's default command line,
/// `jit == 0` the interpreter only. Attaches the disks `uw_add_disk` made, in order.
/// Takes ownership of all three buffers either way.
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
    unix_time: f64,
    cols: u16,
    rows: u16,
    jit: i32,
) -> i32 {
    let kernel = unsafe { take_buffer(kernel, kernel_len) };
    let initrd = unsafe { take_buffer(initrd, initrd_len) };
    let cmdline = unsafe { take_buffer(cmdline, cmdline_len) };

    let mut config = MachineConfig::new()
        .kernel(kernel)
        .memory_mib(memory_mib as usize)
        // Blocks are compiled to WebAssembly modules the page instantiates; without the page's
        // `host_jit_*` imports working, `Auto` falls back to the interpreter.
        .jit(if jit != 0 {
            JitMode::Auto
        } else {
            JitMode::Off
        })
        // Guest time follows the page's clock. A virtual clock jumps over every idle wait and
        // otherwise crawls at one nanosecond per interpreted instruction.
        .clock(ClockMode::Realtime)
        .host_clock(|| (unsafe { host_now_ms() } * NS_PER_MS) as u64)
        .unix_time(unix_time as u64)
        .log(|message| unsafe { host_log(message.as_ptr(), message.len()) })
        // virtio-rng feeds the guest's entropy pool from the browser's.
        .entropy(|bytes| unsafe { host_random(bytes.as_mut_ptr(), bytes.len()) })
        .poll(poll);
    if initrd_len != 0 {
        config = config.initrd(initrd);
    }
    for disk in PENDING_DISKS.with(|disks| disks.take()) {
        config = config.disk(disk);
    }
    if cmdline_len != 0 {
        config = config.cmdline(String::from_utf8_lossy(&cmdline).into_owned());
    }
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
