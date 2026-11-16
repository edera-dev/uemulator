//! The native engine of the Apple app (`docs/plans/28-apple.md`): the library behind a small C ABI
//! that Swift calls through `include/uemulator.h`. It uses only the public `EmulatedMachine` API,
//! as the CLI does. On macOS the machine runs the aarch64 or x86_64 JIT (the app has the allow-jit
//! entitlement); on iOS executable memory is refused and it interprets.
//!
//! Every pointer argument is the caller's promise, as the header documents; nothing here can check
//! one. Panics cannot unwind into Swift: an `extern "C"` function that panics aborts.

#![cfg(target_vendor = "apple")]

mod machine_file;
mod network;

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Mutex;
use std::thread::JoinHandle;

use uemulator::{
    BlockBackend, ConsoleHandle, ConsoleKind, Cursor, Disk, DisplayHandle, DisplayMode,
    EmulatedMachine, Error, ExitReason, FileKeyValueStore, ImageFile, InputHandle, JitMode,
    MachineConfig, PointerButton, RawImage, ScreenMirror, SmpMode, StopHandle, Udisk,
};

// `ua_config.options`: the web build's OPTION_* bits (crates/uemulator-web/src/lib.rs), so one set
// of choices drives either engine. Mirrored in include/uemulator.h.
const OPTION_JIT: u32 = 1 << 0;
const OPTION_ACPI: u32 = 1 << 1;
/// An AMD IOMMU, found through ACPI's IVRS table: useless without `OPTION_ACPI`.
const OPTION_IOMMU: u32 = 1 << 2;
const OPTION_RNG: u32 = 1 << 3;
const OPTION_SERIAL_CONSOLE: u32 = 1 << 4;
/// A guest reset restarts the machine in place (`MachineConfig::reboot`); without it the run
/// ends with `UA_EXIT_RESET`.
const OPTION_REBOOT: u32 = 1 << 5;
/// `cmdline` replaces the library's default command line instead of being appended to it.
const OPTION_REPLACE_CMDLINE: u32 = 1 << 6;
const OPTION_DEBUG: u32 = 1 << 7;
/// A screen (virtio-gpu, the firmware's GOP framebuffer).
const OPTION_DISPLAY: u32 = 1 << 8;
/// A virtio keyboard and tablet.
const OPTION_INPUT: u32 = 1 << 9;
/// With `OPTION_DISPLAY`: the screen is `/dev/console` (`MachineConfig::console_on_screen`).
const OPTION_SCREEN_CONSOLE: u32 = 1 << 10;
/// A TPM 2.0; its seeds and NV live in the NVRAM store (`ua_config.nvram_path`).
const OPTION_TPM: u32 = 1 << 11;
/// The firmware enforces Secure Boot, enrolling its keys into the NVRAM store on first use.
const OPTION_SECURE_BOOT: u32 = 1 << 12;
/// One vCPU per core instead of two threads a core.
const OPTION_NO_SMT: u32 = 1 << 13;
/// A virtio NIC behind the library's gateway, carried over Wisp to the app (`network.rs`). The
/// app's own bit: the web build configures its network with `uw_set_network` instead, so the
/// value is chosen clear of its OPTION_* bits.
const OPTION_NETWORK: u32 = 1 << 16;
/// The vCPUs take turns on one thread (`SmpMode::Timeshare`) instead of a thread each. The app's
/// own bit, like `OPTION_NETWORK`: the web build always takes turns.
const OPTION_TIMESHARE: u32 = 1 << 17;

// `ua_config.firmware_kind`; mirrored in include/uemulator.h. The web build's FIRMWARE_* values.
/// The kernel is entered directly.
const FIRMWARE_NONE: u32 = 0;
/// `ua_config.firmware` is the UEFI firmware image.
const FIRMWARE_UEFI: u32 = 1;
/// The library's BIOS boots the first disk (`MachineConfig::boot_disk`); no image.
const FIRMWARE_BIOS: u32 = 2;
/// Bytes in a MAC address.
const MAC_BYTES: usize = 6;
/// The first octet of the NIC's MAC: locally administered, unicast (bits 1 and 0 of the first
/// octet), so it never claims a vendor's address.
const LOCAL_UNICAST_MAC_PREFIX: u8 = 0x52;

// `ua_cursor_fn` kinds, the library's `Cursor`: the web build's CURSOR_* values. Mirrored in
// include/uemulator.h.
const CURSOR_HOST: i32 = 0;
const CURSOR_HIDDEN: i32 = 1;
const CURSOR_IMAGE: i32 = 2;

// `ua_screen_update` and the input calls' results.
const RESULT_NOTHING_NEW: i32 = 0;
const RESULT_DONE: i32 = 1;
const RESULT_NO_DEVICE: i32 = -1;

// `ua_exit_fn` reasons; mirrored in include/uemulator.h.
const EXIT_SHUTDOWN: i32 = 0;
const EXIT_RESET: i32 = 1;
const EXIT_TRIPLE_FAULT: i32 = 2;
const EXIT_STOPPED: i32 = 3;
const EXIT_INTERNAL: i32 = 4;

#[repr(C)]
pub struct Config {
    kernel: *const u8,
    kernel_len: usize,
    initrd: *const u8,
    initrd_len: usize,
    firmware: *const u8,
    firmware_len: usize,
    cmdline_append: *const c_char,
    disks: *const DiskConfig,
    disk_count: usize,
    nvram_path: *const c_char,
    memory_mib: u32,
    cpus: u32,
    options: u32,
    cols: u16,
    rows: u16,
    // Everything below was added after the first layout; Swift builds this struct from the
    // header, so the assertions under it pin the layout the header describes.
    firmware_kind: u32,
    hypervisor: *const u8,
    hypervisor_len: usize,
    hypervisor_cmdline: *const c_char,
    unix_time: u64,
    display_width: u32,
    display_height: u32,
    log: Option<LogFn>,
}

// The header's layout on the 64-bit Apple targets: a field moved or resized here and not there
// would have Swift write past or beside it, silently.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(Config, rows) == 94);
    assert!(std::mem::offset_of!(Config, firmware_kind) == 96);
    assert!(std::mem::offset_of!(Config, hypervisor) == 104);
    assert!(std::mem::offset_of!(Config, hypervisor_cmdline) == 120);
    assert!(std::mem::offset_of!(Config, unix_time) == 128);
    assert!(std::mem::offset_of!(Config, display_height) == 140);
    assert!(std::mem::offset_of!(Config, log) == 144);
    assert!(size_of::<Config>() == 152);
};

/// `ua_log_fn`: the library's diagnostics, on whichever machine thread logs. `text` is UTF-8 and
/// valid only during the call.
type LogFn = extern "C" fn(context: *mut c_void, text: *const u8, len: usize);

type OutputFn = extern "C" fn(context: *mut c_void, bytes: *const u8, len: usize);
type ExitFn = extern "C" fn(context: *mut c_void, reason: i32, message: *const c_char);
type ScreenFn = extern "C" fn(
    context: *mut c_void,
    width: u32,
    height: u32,
    pixels: *const u8,
    stride: usize,
    rects: *const ScreenRect,
    rect_count: usize,
);
type CursorFn = extern "C" fn(
    context: *mut c_void,
    kind: i32,
    width: u32,
    height: u32,
    hot_x: u32,
    hot_y: u32,
    pixels: *const u8,
    len: usize,
);

/// `ua_disk`: an image file the guest gets as a virtio-blk disk, in `ua_config.disks` order
/// (`vda`, `vdb`, ...).
#[repr(C)]
pub struct DiskConfig {
    path: *const c_char,
    read_only: i32,
}

/// `ua_rect`: the library's `Rect`, which is not `repr(C)`.
#[repr(C)]
pub struct ScreenRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// Swift's context pointer, handed back on every callback and never dereferenced here. The app
/// keeps what it points at alive until `ua_destroy` returns, which is when callbacks stop.
#[derive(Clone, Copy)]
struct Context(*mut c_void);
// SAFETY: only passed back to the caller's callbacks, which are documented as running on the
// machine's threads.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Through a method, not `.0`: a closure that names the field captures just the raw pointer
    /// (disjoint capture), which is not `Send`, instead of the whole `Context`.
    fn pointer(self) -> *mut c_void {
        self.0
    }
}

/// The caller's buffers, copied, so a machine can be configured twice (`create`'s JIT fallback).
struct Settings {
    kernel: Vec<u8>,
    initrd: Vec<u8>,
    firmware: Option<Vec<u8>>,
    cmdline_append: String,
    /// (path, read-only), opened when the machine is built.
    disks: Vec<(String, bool)>,
    /// The machine's `nvram.kv`: UEFI variables and TPM state survive the app.
    nvram_path: Option<String>,
    memory_mib: usize,
    cpus: usize,
    options: u32,
    cols: u16,
    rows: u16,
    firmware_kind: u32,
    /// The hypervisor image and its command line (empty: the library's default).
    hypervisor: Option<(Vec<u8>, String)>,
    /// 0: the host's clock.
    unix_time: u64,
    /// 0 by 0: the library's default size.
    display_size: (u32, u32),
    log: Option<(LogFn, Context)>,
}

impl Settings {
    /// # Safety
    /// As `ua_create` documents for `config`.
    unsafe fn copy(config: &Config) -> Settings {
        let bytes = |ptr: *const u8, len: usize| {
            if ptr.is_null() || len == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
        };
        let firmware = bytes(config.firmware, config.firmware_len);
        let string = |text: *const c_char| {
            if text.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(text) }
                    .to_string_lossy()
                    .into_owned()
            }
        };
        let cmdline_append = string(config.cmdline_append);
        let hypervisor = bytes(config.hypervisor, config.hypervisor_len);
        let disks = if config.disks.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(config.disks, config.disk_count) }
                .iter()
                .filter(|disk| !disk.path.is_null())
                .map(|disk| {
                    let path = unsafe { CStr::from_ptr(disk.path) }
                        .to_string_lossy()
                        .into_owned();
                    (path, disk.read_only != 0)
                })
                .collect()
        };
        let nvram_path = (!config.nvram_path.is_null())
            .then(|| {
                unsafe { CStr::from_ptr(config.nvram_path) }
                    .to_string_lossy()
                    .into_owned()
            })
            .filter(|path| !path.is_empty());
        Settings {
            disks,
            nvram_path,
            kernel: bytes(config.kernel, config.kernel_len),
            initrd: bytes(config.initrd, config.initrd_len),
            firmware: (!firmware.is_empty()).then_some(firmware),
            cmdline_append,
            memory_mib: config.memory_mib as usize,
            cpus: config.cpus as usize,
            options: config.options,
            cols: config.cols,
            rows: config.rows,
            firmware_kind: config.firmware_kind,
            hypervisor: (!hypervisor.is_empty())
                .then(|| (hypervisor, string(config.hypervisor_cmdline))),
            unix_time: config.unix_time,
            display_size: (config.display_width, config.display_height),
            log: None,
        }
    }

    fn machine(&self, jit: JitMode) -> Result<EmulatedMachine, Error> {
        let has = |bit| self.options & bit != 0;
        let mut config = MachineConfig::new()
            .kernel(self.kernel.clone())
            .memory_mib(self.memory_mib)
            .cpus(self.cpus)
            .acpi(has(OPTION_ACPI))
            .iommu(has(OPTION_IOMMU))
            .rng(has(OPTION_RNG))
            .reboot(has(OPTION_REBOOT))
            .debug(has(OPTION_DEBUG))
            .smt(!has(OPTION_NO_SMT))
            .smp(if has(OPTION_TIMESHARE) {
                SmpMode::Timeshare
            } else {
                SmpMode::Auto
            })
            .jit(jit)
            .console(if has(OPTION_SERIAL_CONSOLE) {
                ConsoleKind::Serial
            } else {
                ConsoleKind::Virtio
            });
        if !self.initrd.is_empty() {
            config = config.initrd(self.initrd.clone());
        }
        if has(OPTION_REPLACE_CMDLINE) {
            config = config.cmdline(self.cmdline_append.clone());
        } else if !self.cmdline_append.is_empty() {
            config = config.cmdline_append(self.cmdline_append.clone());
        }
        if let Some((image, cmdline)) = &self.hypervisor {
            config = config.hypervisor(image.clone());
            if !cmdline.is_empty() {
                config = config.hypervisor_cmdline(cmdline.clone());
            }
        }
        if self.unix_time != 0 {
            config = config.unix_time(self.unix_time);
        }
        if let (width @ 1.., height @ 1..) = self.display_size {
            config = config.display_size(width, height);
        }
        if let Some((log, context)) = self.log {
            config = config.log(move |text| log(context.pointer(), text.as_ptr(), text.len()));
        }
        if has(OPTION_DISPLAY) {
            config = config
                .display(DisplayMode::Framebuffer)
                .console_on_screen(has(OPTION_SCREEN_CONSOLE));
        }
        config = config
            .input(has(OPTION_INPUT))
            .tpm(has(OPTION_TPM))
            .secure_boot(has(OPTION_SECURE_BOOT));
        if has(OPTION_NETWORK) {
            let mut mac = [0u8; MAC_BYTES];
            machine_file::random_bytes(&mut mac)
                .map_err(|e| Error::BadConfig(format!("a MAC address: {e}")))?;
            mac[0] = LOCAL_UNICAST_MAC_PREFIX;
            config = config.net(mac);
        }
        if let Some(path) = &self.nvram_path {
            // Opened per machine build: `create`'s JIT fallback builds twice, and the store keeps
            // its file open, so the first build's copy is dropped before the second opens it.
            let store =
                FileKeyValueStore::open(path).map_err(|e| Error::BadConfig(e.to_string()))?;
            config = config.key_value_store(store);
        }
        for (path, read_only) in &self.disks {
            config = config.disk(open_disk(path, *read_only)?);
        }
        config = match (self.firmware_kind, &self.firmware) {
            (FIRMWARE_UEFI, Some(firmware)) => config.uefi_firmware(firmware.clone()),
            (FIRMWARE_UEFI, None) => {
                return Err(Error::BadConfig("UEFI needs the firmware image".into()));
            }
            (FIRMWARE_BIOS, _) => config.boot_disk(true),
            (FIRMWARE_NONE, _) => config,
            (kind, _) => return Err(Error::BadConfig(format!("unknown firmware kind {kind}"))),
        };
        EmulatedMachine::new(config)
    }
}

pub struct Machine {
    /// Taken by `ua_start`.
    pending: Mutex<Option<EmulatedMachine>>,
    /// The thread that starts the machine, waits for it and reports the exit.
    runner: Mutex<Option<JoinHandle<()>>>,
    console: ConsoleHandle,
    display: Option<DisplayHandle>,
    input: Option<InputHandle>,
    /// What the app was last given (`ua_screen_update`), and the cursor generation it last saw.
    /// Only the app's thread that draws calls those, so the locks are never contended.
    mirror: Mutex<ScreenMirror>,
    cursor_generation: Mutex<Option<u64>>,
    net: Option<uemulator::NetHandle>,
    /// Started by `ua_network_start`, stopped by `ua_destroy` before the machine goes.
    network: Mutex<Option<network::Network>>,
    stop: StopHandle,
    jit: bool,
    exit: ExitFn,
    context: Context,
}

/// A raw or udisk image file (`docs/spec/uemulator-disk-format.md`). An overlay's base is not
/// followed: the app flattens or refuses those (M4), so one says so here rather than failing late.
fn open_disk(path: &str, read_only: bool) -> Result<Disk, Error> {
    Ok(Disk::from_backend(open_backend(path, read_only)?))
}

fn open_backend(path: &str, read_only: bool) -> Result<Box<dyn BlockBackend>, Error> {
    let failed = |what: String| Error::BadConfig(format!("{path}: {what}"));
    let mut file = ImageFile::open(path, read_only).map_err(|e| failed(e.to_string()))?;
    let backend: Box<dyn BlockBackend> =
        match Udisk::probe(&mut file).map_err(|e| failed(e.to_string()))? {
            Some(info) if info.backing.is_some() => {
                return Err(failed(
                    "an overlay; flatten it first with `uemulator disk convert`".into(),
                ));
            }
            Some(_) => Box::new(Udisk::open(file, None).map_err(|e| failed(e.to_string()))?),
            None => Box::new(RawImage::new(file)),
        };
    Ok(backend)
}

/// The CLI's test for "something is installed" (`machines.rs`, `disk_is_blank`): partition tables,
/// boot sectors and the first filesystem structures all live in the first 32 KiB.
const INSTALLED_CHECK_BYTES: u64 = 32 * 1024;

/// 1 if the start of the disk image at `path` is not all zeros, as the guest sees it (a udisk's
/// own header does not count); 0 if it is; -1 with the message in `error`.
///
/// # Safety
/// `path` is a NUL-terminated string; `error` has `error_capacity` writable bytes or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_disk_installed(
    path: *const c_char,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    let checked = (|| {
        if path.is_null() {
            return Err("no path".to_string());
        }
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        let mut backend = open_backend(&path, true).map_err(|e| e.to_string())?;
        let mut start = vec![0u8; INSTALLED_CHECK_BYTES.min(backend.len()) as usize];
        backend
            .read_at(0, &mut start)
            .map_err(|e| format!("{path}: {e}"))?;
        Ok(start.iter().any(|&byte| byte != 0))
    })();
    match checked {
        Ok(installed) => i32::from(installed),
        Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure) };
            -1
        }
    }
}

/// # Safety
/// `config` and the buffers it points at are valid for the call; `error` has `error_capacity`
/// writable bytes (or is null with capacity 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_create(
    config: *const Config,
    output: OutputFn,
    exit: ExitFn,
    context: *mut c_void,
    error: *mut c_char,
    error_capacity: usize,
) -> *mut Machine {
    let mut settings = unsafe { Settings::copy(&*config) };
    settings.log = unsafe { (*config).log }.map(|log| (log, Context(context)));
    let wants_jit = settings.options & OPTION_JIT != 0;
    // `JitMode::On` instead of `Auto` so the app learns which one it got: `new` fails with
    // `JitUnavailable` where executable memory is refused (iOS), and the machine is built again
    // interpreted.
    let built = if wants_jit {
        match settings.machine(JitMode::On) {
            Err(Error::JitUnavailable) => settings
                .machine(JitMode::Off)
                .map(|machine| (machine, false)),
            other => other.map(|machine| (machine, true)),
        }
    } else {
        settings
            .machine(JitMode::Off)
            .map(|machine| (machine, false))
    };
    let (machine, jit) = match built {
        Ok(built) => built,
        Err(failure) => {
            unsafe { write_message(error, error_capacity, &failure.to_string()) };
            return std::ptr::null_mut();
        }
    };
    let console = machine.console();
    console.resize(settings.cols, settings.rows);
    let output_context = Context(context);
    console.on_output(move |bytes| output(output_context.pointer(), bytes.as_ptr(), bytes.len()));
    Box::into_raw(Box::new(Machine {
        stop: machine.stop_handle(),
        display: machine.display(),
        input: machine.input(),
        mirror: Mutex::new(ScreenMirror::new()),
        cursor_generation: Mutex::new(None),
        net: machine.net(),
        network: Mutex::new(None),
        pending: Mutex::new(Some(machine)),
        runner: Mutex::new(None),
        console,
        jit,
        exit,
        context: Context(context),
    }))
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_jit_active(machine: *const Machine) -> i32 {
    i32::from(unsafe { &*machine }.jit)
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_start(machine: *mut Machine) -> i32 {
    let machine = unsafe { &*machine };
    let Some(mut emulated) = machine.pending.lock().unwrap().take() else {
        return -1;
    };
    let (exit, context) = (machine.exit, machine.context);
    let runner = std::thread::spawn(move || {
        let reason = match emulated.start() {
            Ok(()) => emulated.wait(),
            Err(failure) => ExitReason::Internal(failure.to_string()),
        };
        let (code, text) = match reason {
            ExitReason::Shutdown => (EXIT_SHUTDOWN, String::new()),
            ExitReason::Reset => (EXIT_RESET, String::new()),
            ExitReason::TripleFault => (EXIT_TRIPLE_FAULT, String::new()),
            ExitReason::Stopped => (EXIT_STOPPED, String::new()),
            ExitReason::Internal(text) => (EXIT_INTERNAL, text),
        };
        let message = CString::new(text.replace('\0', " ")).unwrap_or_default();
        exit(context.pointer(), code, message.as_ptr());
    });
    *machine.runner.lock().unwrap() = Some(runner);
    0
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed; `bytes` has `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_input(machine: *const Machine, bytes: *const u8, len: usize) {
    if bytes.is_null() || len == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
    unsafe { &*machine }.console.write(bytes);
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_resize(machine: *const Machine, cols: u16, rows: u16) {
    unsafe { &*machine }.console.resize(cols, rows);
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_stop(machine: *const Machine) {
    unsafe { &*machine }.stop.stop();
}

/// # Safety
/// `machine` came from `ua_create` and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_destroy(machine: *mut Machine) {
    if machine.is_null() {
        return;
    }
    let machine = unsafe { Box::from_raw(machine) };
    // Before the machine: the pump calls the app's sender, which must not run after this returns.
    if let Some(network) = machine.network.lock().unwrap().take() {
        network.stop();
    }
    machine.stop.stop();
    // The runner's exit callback is the last one; after the join none can run.
    if let Some(runner) = machine.runner.lock().unwrap().take() {
        let _ = runner.join();
    }
}

/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_has_display(machine: *const Machine) -> i32 {
    i32::from(unsafe { &*machine }.display.is_some())
}

/// Brings the app's copy of the screen up to date. `screen` gets the whole screen and the parts
/// that differ from the last call (none when the guest flushed what was already there), and runs
/// before this returns, on the calling thread; `pixels` is valid only during it.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_screen_update(
    machine: *const Machine,
    screen: ScreenFn,
    context: *mut c_void,
) -> i32 {
    let machine = unsafe { &*machine };
    let Some(display) = &machine.display else {
        return RESULT_NO_DEVICE;
    };
    let mut mirror = machine.mirror.lock().unwrap();
    let Some(changed) = display.sync_mirror(&mut mirror) else {
        return RESULT_NOTHING_NEW;
    };
    let rects: Vec<ScreenRect> = changed
        .iter()
        .map(|rect| ScreenRect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        })
        .collect();
    let (width, height) = mirror.size();
    screen(
        context,
        width,
        height,
        mirror.pixels().as_ptr(),
        mirror.stride(),
        rects.as_ptr(),
        rects.len(),
    );
    RESULT_DONE
}

/// Calls `cursor` when the guest's pointer image changed since the last call (and on the first),
/// before returning; `pixels` (straight-alpha BGRA) is valid only during it.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_cursor_update(
    machine: *const Machine,
    cursor: CursorFn,
    context: *mut c_void,
) -> i32 {
    let machine = unsafe { &*machine };
    let Some(display) = &machine.display else {
        return RESULT_NO_DEVICE;
    };
    let (generation, image) = display.cursor();
    let mut seen = machine.cursor_generation.lock().unwrap();
    if *seen == Some(generation) {
        return RESULT_NOTHING_NEW;
    }
    *seen = Some(generation);
    match image {
        Cursor::Host => cursor(context, CURSOR_HOST, 0, 0, 0, 0, std::ptr::null(), 0),
        Cursor::Hidden => cursor(context, CURSOR_HIDDEN, 0, 0, 0, 0, std::ptr::null(), 0),
        Cursor::Image(image) => cursor(
            context,
            CURSOR_IMAGE,
            image.width,
            image.height,
            image.hot_x,
            image.hot_y,
            image.pixels.as_ptr(),
            image.pixels.len(),
        ),
    }
    RESULT_DONE
}

/// Runs `send` on the machine's keyboard and tablet: `RESULT_DONE` when it was queued,
/// `RESULT_NOTHING_NEW` when the guest's driver is not listening yet (or the key has no Linux
/// code), `RESULT_NO_DEVICE` without input devices.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
unsafe fn with_input(machine: *const Machine, send: impl FnOnce(&InputHandle) -> bool) -> i32 {
    match &unsafe { &*machine }.input {
        None => RESULT_NO_DEVICE,
        Some(input) if send(input) => RESULT_DONE,
        Some(_) => RESULT_NOTHING_NEW,
    }
}

/// A key by USB HID usage (keyboard page 0x07). Any thread.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_key(machine: *const Machine, usage: u16, pressed: i32) -> i32 {
    unsafe { with_input(machine, |input| input.key_usb(usage, pressed != 0)) }
}

/// The pointer at (`x`, `y`) in screen pixels. Any thread.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_pointer_move(machine: *const Machine, x: u32, y: u32) -> i32 {
    unsafe { with_input(machine, |input| input.pointer_move(x, y)) }
}

/// `button` 0 left, 1 right, 2 middle, 3 back, 4 forward (the web build's numbering). Any thread.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_pointer_button(
    machine: *const Machine,
    button: u32,
    pressed: i32,
) -> i32 {
    let button = match button {
        0 => PointerButton::Left,
        1 => PointerButton::Right,
        2 => PointerButton::Middle,
        3 => PointerButton::Side,
        4 => PointerButton::Extra,
        _ => return RESULT_NOTHING_NEW,
    };
    unsafe { with_input(machine, |input| input.pointer_button(button, pressed != 0)) }
}

/// The wheel, `clicks` notches, positive away from the user. Any thread.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_scroll(machine: *const Machine, clicks: i32) -> i32 {
    unsafe { with_input(machine, |input| input.scroll(clicks)) }
}

/// Releases every key and button the guest thinks is down: for a view that loses the focus and
/// with it the release events. Any thread.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_release_all(machine: *const Machine) -> i32 {
    unsafe {
        with_input(machine, |input| {
            input.release_all();
            true
        })
    }
}

/// Asks the guest for a `width` x `height` screen, as a monitor hotplug would; the guest decides.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_request_size(machine: *const Machine, width: u32, height: u32) -> i32 {
    match &unsafe { &*machine }.display {
        None => RESULT_NO_DEVICE,
        Some(display) => {
            display.request_size(width, height);
            RESULT_DONE
        }
    }
}

/// Starts the network of a machine built with `OPTION_NETWORK`: from here `send` gets the Wisp
/// client's packets (on the network's own thread) and the server's go in through `ua_wisp_*`.
/// `RESULT_NO_DEVICE` without a NIC, `RESULT_NOTHING_NEW` if it already runs.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed; `context` stays valid until
/// `ua_destroy` returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_network_start(
    machine: *const Machine,
    send: network::WispSendFn,
    context: *mut c_void,
) -> i32 {
    let machine = unsafe { &*machine };
    let Some(net) = &machine.net else {
        return RESULT_NO_DEVICE;
    };
    let mut slot = machine.network.lock().unwrap();
    if slot.is_some() {
        return RESULT_NOTHING_NEW;
    }
    // The sequence numbers only keep connections apart; any value does.
    let mut seed = [0u8; 4];
    let _ = machine_file::random_bytes(&mut seed);
    *slot = Some(network::Network::start(
        net.clone(),
        send,
        network::SendContext(context),
        u32::from_le_bytes(seed),
    ));
    RESULT_DONE
}

fn with_network(machine: *const Machine, deliver: impl FnOnce(&network::Network)) -> i32 {
    match unsafe { &*machine }.network.lock().unwrap().as_ref() {
        Some(network) => {
            deliver(network);
            RESULT_DONE
        }
        None => RESULT_NO_DEVICE,
    }
}

/// The app opened a Wisp connection for this machine (the client starts its handshake clock).
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_wisp_connecting(machine: *const Machine) -> i32 {
    with_network(machine, network::Network::connecting)
}

/// One message from the Wisp server, in order: none may be skipped or reordered.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed; `bytes` has `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_wisp_message(
    machine: *const Machine,
    bytes: *const u8,
    len: usize,
) -> i32 {
    if bytes.is_null() {
        return RESULT_NOTHING_NEW;
    }
    let message = unsafe { std::slice::from_raw_parts(bytes, len) };
    with_network(machine, |network| network.message(message))
}

/// The Wisp connection is gone: the client resets its streams.
///
/// # Safety
/// `machine` came from `ua_create` and was not destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ua_wisp_lost(machine: *const Machine) -> i32 {
    with_network(machine, network::Network::lost)
}

/// # Safety
/// `buffer` has `capacity` writable bytes, or is null.
pub(crate) unsafe fn write_message(buffer: *mut c_char, capacity: usize, text: &str) {
    if buffer.is_null() || capacity == 0 {
        return;
    }
    let count = text.len().min(capacity - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn ignore_output(_: *mut c_void, _: *const u8, _: usize) {}
    extern "C" fn ignore_exit(_: *mut c_void, _: i32, _: *const c_char) {}

    fn config(kernel: &[u8]) -> Config {
        Config {
            kernel: kernel.as_ptr(),
            kernel_len: kernel.len(),
            initrd: std::ptr::null(),
            initrd_len: 0,
            firmware: std::ptr::null(),
            firmware_len: 0,
            cmdline_append: std::ptr::null(),
            disks: std::ptr::null(),
            disk_count: 0,
            nvram_path: std::ptr::null(),
            memory_mib: 256,
            cpus: 1,
            options: OPTION_ACPI,
            cols: 80,
            rows: 25,
            firmware_kind: FIRMWARE_NONE,
            hypervisor: std::ptr::null(),
            hypervisor_len: 0,
            hypervisor_cmdline: std::ptr::null(),
            unix_time: 0,
            display_width: 0,
            display_height: 0,
            log: None,
        }
    }

    fn try_create(config: &Config) -> Result<*mut Machine, String> {
        let mut error = [0 as c_char; 256];
        let machine = unsafe {
            ua_create(
                config,
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if machine.is_null() {
            Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        } else {
            Ok(machine)
        }
    }

    // A udisk's header is not the guest's data: a new image is blank until a guest byte is
    // written, the way `auto` tells an installer machine from an installed one.
    #[test]
    fn disk_counts_as_installed_once_its_start_has_data() {
        let dir =
            std::env::temp_dir().join(format!("uemulator-apple-installed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("disk-0.udisk");
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut error = [0 as c_char; 256];
        let installed = |error: &mut [c_char; 256]| unsafe {
            ua_disk_installed(c_path.as_ptr(), error.as_mut_ptr(), error.len())
        };
        assert_eq!(
            unsafe {
                machine_file::ua_disk_create(c_path.as_ptr(), 1, error.as_mut_ptr(), error.len())
            },
            0
        );
        assert_eq!(installed(&mut error), 0);
        {
            let mut backend = open_backend(path.to_str().unwrap(), false).unwrap();
            backend.write_at(INSTALLED_CHECK_BYTES - 1, &[1]).unwrap();
            backend.flush().unwrap();
        }
        assert_eq!(installed(&mut error), 1);
        let missing = CString::new(dir.join("missing.udisk").to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe { ua_disk_installed(missing.as_ptr(), error.as_mut_ptr(), error.len()) },
            -1
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // BIOS boots a disk, so it needs one; UEFI needs the image; anything else is refused by name.
    #[test]
    fn create_checks_the_firmware_kind() {
        let kernel = halting_kernel();
        let mut uefi = config(&kernel);
        uefi.firmware_kind = FIRMWARE_UEFI;
        assert!(try_create(&uefi).unwrap_err().contains("firmware image"));
        let mut unknown = config(&kernel);
        unknown.firmware_kind = 9;
        assert!(
            try_create(&unknown)
                .unwrap_err()
                .contains("unknown firmware kind 9")
        );
        let machine = try_create(&config(&kernel)).unwrap();
        unsafe { ua_destroy(machine) };
    }

    /// A bzImage the loader accepts (the boot protocol header, 64-bit entry) whose 64-bit entry
    /// is `hlt; jmp` back to it: what the library's `testkit::FakeKernel` builds, which is not
    /// visible from here. These tests never start it; they need a machine, not a guest.
    fn halting_kernel() -> Vec<u8> {
        const SECTOR: usize = 512;
        const SETUP_SECTS: usize = 1;
        const ENTRY_OFFSET: usize = 0x200;
        const KERNEL_LOAD: u64 = 0x100_0000;
        const HALT_LOOP: [u8; 3] = [0xf4, 0xeb, 0xfd];
        let protected = (SETUP_SECTS + 1) * SECTOR;
        let payload = ENTRY_OFFSET + HALT_LOOP.len();
        let mut image = vec![0u8; protected + payload];
        image[0x1f1] = SETUP_SECTS as u8;
        image[0x200] = 0xeb; // jmp short over the header
        image[0x201] = 0x6a; // header length: up to and past handover_offset
        image[0x202..0x206].copy_from_slice(b"HdrS");
        image[0x206..0x208].copy_from_slice(&0x020fu16.to_le_bytes()); // boot protocol 2.15
        image[0x211] = 1; // LOADED_HIGH
        image[0x22c..0x230].copy_from_slice(&0x7fff_ffffu32.to_le_bytes()); // initrd_addr_max
        image[0x234] = 1; // relocatable
        image[0x236..0x238].copy_from_slice(&1u16.to_le_bytes()); // XLF_KERNEL_64
        image[0x238..0x23c].copy_from_slice(&2048u32.to_le_bytes()); // cmdline_size
        image[0x258..0x260].copy_from_slice(&KERNEL_LOAD.to_le_bytes()); // pref_address
        image[0x260..0x264].copy_from_slice(&(payload as u32).to_le_bytes()); // init_size
        image[protected + ENTRY_OFFSET..].copy_from_slice(&HALT_LOOP);
        image
    }

    fn create(kernel: &[u8], options: u32) -> *mut Machine {
        let mut config = config(kernel);
        config.options = options;
        let mut error = [0 as c_char; 256];
        let machine = unsafe {
            ua_create(
                &config,
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert!(
            !machine.is_null(),
            "{}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
        machine
    }

    /// What the screen callback was given, through its context pointer.
    #[derive(Default)]
    struct SeenScreen {
        width: u32,
        height: u32,
        bytes: usize,
        rects: Vec<(u32, u32, u32, u32)>,
    }

    extern "C" fn record_screen(
        context: *mut c_void,
        width: u32,
        height: u32,
        _pixels: *const u8,
        stride: usize,
        rects: *const ScreenRect,
        rect_count: usize,
    ) {
        let seen = unsafe { &mut *context.cast::<SeenScreen>() };
        seen.width = width;
        seen.height = height;
        seen.bytes = stride * height as usize;
        let rects = unsafe { std::slice::from_raw_parts(rects, rect_count) };
        seen.rects = rects
            .iter()
            .map(|rect| (rect.x, rect.y, rect.width, rect.height))
            .collect();
    }

    extern "C" fn record_cursor(
        context: *mut c_void,
        kind: i32,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
        _: *const u8,
        _: usize,
    ) {
        unsafe { *context.cast::<i32>() = kind };
    }

    // A machine without a screen says so to every screen and input call, rather than crashing or
    // pretending.
    #[test]
    fn screen_and_input_calls_without_devices_say_so() {
        let kernel = halting_kernel();
        let machine = create(&kernel, OPTION_ACPI);
        let mut seen = SeenScreen::default();
        unsafe {
            assert_eq!(ua_has_display(machine), 0);
            let context = (&raw mut seen).cast();
            assert_eq!(
                ua_screen_update(machine, record_screen, context),
                RESULT_NO_DEVICE
            );
            assert_eq!(ua_key(machine, 0x04, 1), RESULT_NO_DEVICE);
            assert_eq!(ua_pointer_move(machine, 1, 1), RESULT_NO_DEVICE);
            assert_eq!(ua_request_size(machine, 1024, 768), RESULT_NO_DEVICE);
            ua_destroy(machine);
        }
    }

    // The app's first look gets the whole (blank) screen as one rectangle; a second look with
    // nothing drawn gets nothing. The cursor starts as the host's and is reported once.
    #[test]
    fn screen_update_hands_over_the_whole_screen_once() {
        const DEFAULT_WIDTH: u32 = 1280;
        const DEFAULT_HEIGHT: u32 = 800;
        let kernel = halting_kernel();
        let machine = create(&kernel, OPTION_ACPI | OPTION_DISPLAY | OPTION_INPUT);
        let mut seen = SeenScreen::default();
        let mut cursor_kind = -1;
        unsafe {
            assert_eq!(ua_has_display(machine), 1);
            let context = (&raw mut seen).cast();
            assert_eq!(
                ua_screen_update(machine, record_screen, context),
                RESULT_DONE
            );
            assert_eq!((seen.width, seen.height), (DEFAULT_WIDTH, DEFAULT_HEIGHT));
            assert_eq!(seen.bytes, (DEFAULT_WIDTH * DEFAULT_HEIGHT * 4) as usize);
            assert_eq!(seen.rects, vec![(0, 0, DEFAULT_WIDTH, DEFAULT_HEIGHT)]);
            assert_eq!(
                ua_screen_update(machine, record_screen, context),
                RESULT_NOTHING_NEW
            );

            let cursor_context = (&raw mut cursor_kind).cast();
            assert_eq!(
                ua_cursor_update(machine, record_cursor, cursor_context),
                RESULT_DONE
            );
            assert_eq!(cursor_kind, CURSOR_HOST);
            assert_eq!(
                ua_cursor_update(machine, record_cursor, cursor_context),
                RESULT_NOTHING_NEW
            );

            // A button number the tablet does not have is refused, not mapped to something.
            assert_eq!(ua_pointer_button(machine, 9, 1), RESULT_NOTHING_NEW);
            assert_eq!(ua_request_size(machine, 1024, 768), RESULT_DONE);
            ua_destroy(machine);
        }
    }

    // A disk that is not there fails the machine with the path in the message, not later in the
    // guest.
    #[test]
    fn create_names_a_missing_disk() {
        let kernel = halting_kernel();
        let path = CString::new("/nonexistent/uemulator-test.iso").unwrap();
        let disks = [DiskConfig {
            path: path.as_ptr(),
            read_only: 1,
        }];
        let mut config = config(&kernel);
        config.disks = disks.as_ptr();
        config.disk_count = disks.len();
        let mut error = [0 as c_char; 256];
        let machine = unsafe {
            ua_create(
                &config,
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert!(machine.is_null());
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_str().unwrap();
        assert!(
            message.contains("/nonexistent/uemulator-test.iso"),
            "{message}"
        );
    }

    // A raw image file becomes a disk.
    #[test]
    fn create_attaches_a_raw_image() {
        const IMAGE_BYTES: usize = 1 << 20;
        let kernel = halting_kernel();
        let file =
            std::env::temp_dir().join(format!("uemulator-apple-test-{}.img", std::process::id()));
        std::fs::write(&file, vec![0u8; IMAGE_BYTES]).unwrap();
        let path = CString::new(file.to_str().unwrap()).unwrap();
        let disks = [DiskConfig {
            path: path.as_ptr(),
            read_only: 1,
        }];
        let mut config = config(&kernel);
        config.disks = disks.as_ptr();
        config.disk_count = disks.len();
        let mut error = [0 as c_char; 256];
        let machine = unsafe {
            ua_create(
                &config,
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        std::fs::remove_file(&file).unwrap();
        assert!(
            !machine.is_null(),
            "{}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
        unsafe { ua_destroy(machine) };
    }

    extern "C" fn count_packets(context: *mut c_void, _: *const u8, _: usize) {
        unsafe { &*context.cast::<std::sync::atomic::AtomicUsize>() }
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // The network starts once, takes the server's events, and stops with the machine: no callback
    // after `ua_destroy` (the context here is dropped right after it). A machine without a NIC
    // says so.
    #[test]
    fn network_starts_once_and_stops_with_the_machine() {
        /// Wisp v1's opening CONTINUE: type 3, stream 0, 128 packets of credit.
        const SERVER_HELLO: [u8; 9] = [0x03, 0, 0, 0, 0, 128, 0, 0, 0];
        let kernel = halting_kernel();
        let packets = std::sync::atomic::AtomicUsize::new(0);
        let context = (&raw const packets).cast_mut().cast();

        let without = create(&kernel, OPTION_ACPI);
        unsafe {
            assert_eq!(
                ua_network_start(without, count_packets, context),
                RESULT_NO_DEVICE
            );
            assert_eq!(ua_wisp_connecting(without), RESULT_NO_DEVICE);
            ua_destroy(without);
        }

        let machine = create(&kernel, OPTION_ACPI | OPTION_NETWORK);
        unsafe {
            assert_eq!(
                ua_network_start(machine, count_packets, context),
                RESULT_DONE
            );
            assert_eq!(
                ua_network_start(machine, count_packets, context),
                RESULT_NOTHING_NEW
            );
            assert_eq!(ua_wisp_connecting(machine), RESULT_DONE);
            assert_eq!(
                ua_wisp_message(machine, SERVER_HELLO.as_ptr(), SERVER_HELLO.len()),
                RESULT_DONE
            );
            assert_eq!(ua_wisp_lost(machine), RESULT_DONE);
            ua_destroy(machine);
        }
        let after = packets.load(std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(
            packets.load(std::sync::atomic::Ordering::Relaxed),
            after,
            "a packet after ua_destroy"
        );
    }

    // What the app shows when a file is not a kernel: NULL and the library's own message.
    #[test]
    fn create_refuses_a_bad_kernel_with_a_message() {
        let not_a_kernel = [0u8; 4096];
        let mut error = [0 as c_char; 256];
        let machine = unsafe {
            ua_create(
                &config(&not_a_kernel),
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert!(machine.is_null());
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_str().unwrap();
        assert!(message.starts_with("bad kernel image"), "{message}");
    }

    // The message is cut to fit and still NUL-terminated.
    #[test]
    fn create_truncates_the_message_to_the_buffer() {
        let not_a_kernel = [0u8; 4096];
        const CAPACITY: usize = 8;
        let mut error = [1 as c_char; CAPACITY];
        let machine = unsafe {
            ua_create(
                &config(&not_a_kernel),
                ignore_output,
                ignore_exit,
                std::ptr::null_mut(),
                error.as_mut_ptr(),
                CAPACITY,
            )
        };
        assert!(machine.is_null());
        assert_eq!(error[CAPACITY - 1], 0);
        assert_eq!(
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_bytes(),
            b"bad ker"
        );
    }
}
