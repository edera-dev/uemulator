//! The native engine of the Apple app (`docs/plans/28-apple.md`): the library behind a small C ABI
//! that Swift calls through `include/uemulator.h`. It uses only the public `EmulatedMachine` API,
//! as the CLI does. On macOS the machine runs the aarch64 or x86_64 JIT (the app has the allow-jit
//! entitlement); on iOS executable memory is refused and it interprets.
//!
//! Every pointer argument is the caller's promise, as the header documents; nothing here can check
//! one. Panics cannot unwind into Swift: an `extern "C"` function that panics aborts.

#![cfg(target_vendor = "apple")]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Mutex;
use std::thread::JoinHandle;

use uemulator::{
    ConsoleHandle, ConsoleKind, EmulatedMachine, Error, ExitReason, JitMode, MachineConfig,
    StopHandle,
};

// `ua_config.options`: the web build's OPTION_* bits (crates/uemulator-web/src/lib.rs), so one set
// of choices drives either engine. Mirrored in include/uemulator.h.
const OPTION_JIT: u32 = 1 << 0;
const OPTION_ACPI: u32 = 1 << 1;
const OPTION_RNG: u32 = 1 << 3;
const OPTION_SERIAL_CONSOLE: u32 = 1 << 4;

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
    memory_mib: u32,
    cpus: u32,
    options: u32,
    cols: u16,
    rows: u16,
}

type OutputFn = extern "C" fn(context: *mut c_void, bytes: *const u8, len: usize);
type ExitFn = extern "C" fn(context: *mut c_void, reason: i32, message: *const c_char);

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
    memory_mib: usize,
    cpus: usize,
    options: u32,
    cols: u16,
    rows: u16,
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
        let cmdline_append = if config.cmdline_append.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(config.cmdline_append) }
                .to_string_lossy()
                .into_owned()
        };
        Settings {
            kernel: bytes(config.kernel, config.kernel_len),
            initrd: bytes(config.initrd, config.initrd_len),
            firmware: (!firmware.is_empty()).then_some(firmware),
            cmdline_append,
            memory_mib: config.memory_mib as usize,
            cpus: config.cpus as usize,
            options: config.options,
            cols: config.cols,
            rows: config.rows,
        }
    }

    fn machine(&self, jit: JitMode) -> Result<EmulatedMachine, Error> {
        let has = |bit| self.options & bit != 0;
        let mut config = MachineConfig::new()
            .kernel(self.kernel.clone())
            .memory_mib(self.memory_mib)
            .cpus(self.cpus)
            .acpi(has(OPTION_ACPI))
            .rng(has(OPTION_RNG))
            .jit(jit)
            .console(if has(OPTION_SERIAL_CONSOLE) {
                ConsoleKind::Serial
            } else {
                ConsoleKind::Virtio
            });
        if !self.initrd.is_empty() {
            config = config.initrd(self.initrd.clone());
        }
        if !self.cmdline_append.is_empty() {
            config = config.cmdline_append(self.cmdline_append.clone());
        }
        if let Some(firmware) = &self.firmware {
            config = config.uefi_firmware(firmware.clone());
        }
        EmulatedMachine::new(config)
    }
}

pub struct Machine {
    /// Taken by `ua_start`.
    pending: Mutex<Option<EmulatedMachine>>,
    /// The thread that starts the machine, waits for it and reports the exit.
    runner: Mutex<Option<JoinHandle<()>>>,
    console: ConsoleHandle,
    stop: StopHandle,
    jit: bool,
    exit: ExitFn,
    context: Context,
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
    let settings = unsafe { Settings::copy(&*config) };
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
    machine.stop.stop();
    // The runner's exit callback is the last one; after the join none can run.
    if let Some(runner) = machine.runner.lock().unwrap().take() {
        let _ = runner.join();
    }
}

/// # Safety
/// `buffer` has `capacity` writable bytes, or is null.
unsafe fn write_message(buffer: *mut c_char, capacity: usize, text: &str) {
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
            memory_mib: 256,
            cpus: 1,
            options: OPTION_ACPI,
            cols: 80,
            rows: 25,
        }
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
