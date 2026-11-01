//! The machine's devices as one shared object that every vCPU holds a handle to.
//!
//! With one host thread (a single vCPU, or several time-sharing) the handle never locks: taking
//! `lock()` costs a counter bump. With one host thread per vCPU the same call takes the machine's
//! device lock, QEMU's "big lock": every access to a device, the interrupt controllers and the
//! timers happens inside it, so the device models stay single-threaded code.
//!
//! The lock is reentrant per handle. Guest-facing code often calls `self.dev.lock()` in the
//! scrutinee of an `if let` and again in its body; a plain lock would deadlock there. The nested
//! guards alias (`&mut Devices` twice), which is sound here because they are never used at the
//! same time: the inner one lives and dies inside the outer one's statement.

use super::Devices;
use crate::sync::Arc;
use core::cell::{Cell, UnsafeCell};
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// Spins before a waiting thread starts offering its time slice to others.
const SPINS_BEFORE_YIELD: u32 = 200;

struct DevShared {
    /// Whether `lock` really locks. Off until the machine runs one host thread per vCPU.
    locking: AtomicBool,
    held: AtomicBool,
    devices: UnsafeCell<Devices>,
}

// `devices` is only reached through a `DevGuard`, which holds `held` when `locking` is on and is
// the only user otherwise (single host thread).
unsafe impl Send for DevShared {}
unsafe impl Sync for DevShared {}

/// A vCPU's (or the device thread's) handle on the shared devices. Clones share the devices but
/// count their own lock nesting.
pub(crate) struct DevHandle {
    shared: Arc<DevShared>,
    depth: Cell<u32>,
    /// The vCPU this handle belongs to: the local APIC the CPU-facing accesses go to. Set into
    /// `Devices::current_cpu` whenever the outermost guard is taken.
    cpu_index: Cell<usize>,
}

impl DevHandle {
    pub fn new(devices: Devices) -> Self {
        DevHandle {
            shared: Arc::new(DevShared {
                locking: AtomicBool::new(false),
                held: AtomicBool::new(false),
                devices: UnsafeCell::new(devices),
            }),
            depth: Cell::new(0),
            cpu_index: Cell::new(0),
        }
    }

    /// Another handle on the same devices, for vCPU `cpu_index`.
    pub fn for_cpu(&self, cpu_index: usize) -> DevHandle {
        DevHandle {
            shared: self.shared.clone(),
            depth: Cell::new(0),
            cpu_index: Cell::new(cpu_index),
        }
    }

    /// Turns real locking on or off. Only while no other thread can hold a guard.
    pub fn set_locking(&self, on: bool) {
        self.shared.locking.store(on, Ordering::SeqCst);
    }

    pub fn locking(&self) -> bool {
        self.shared.locking.load(Ordering::Relaxed)
    }

    pub fn lock(&self) -> DevGuard<'_> {
        let depth = self.depth.get();
        self.depth.set(depth + 1);
        if depth == 0 {
            if self.locking() {
                let mut spins = 0u32;
                while self
                    .shared
                    .held
                    .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_err()
                {
                    spins += 1;
                    if spins < SPINS_BEFORE_YIELD {
                        core::hint::spin_loop();
                    } else {
                        crate::sync::yield_now();
                    }
                }
            }
            unsafe { &mut *self.shared.devices.get() }.set_current_cpu(self.cpu_index.get());
        }
        DevGuard { handle: self }
    }

    /// The devices themselves, once every other handle is gone (after the vCPU threads have been
    /// joined). `None` if one is still alive.
    pub fn into_devices(self) -> Option<Devices> {
        Arc::try_unwrap(self.shared)
            .ok()
            .map(|shared| shared.devices.into_inner())
    }
}

pub(crate) struct DevGuard<'a> {
    handle: &'a DevHandle,
}

impl Deref for DevGuard<'_> {
    type Target = Devices;
    fn deref(&self) -> &Devices {
        unsafe { &*self.handle.shared.devices.get() }
    }
}

impl DerefMut for DevGuard<'_> {
    fn deref_mut(&mut self) -> &mut Devices {
        unsafe { &mut *self.handle.shared.devices.get() }
    }
}

impl Drop for DevGuard<'_> {
    fn drop(&mut self) {
        let depth = self.handle.depth.get() - 1;
        self.handle.depth.set(depth);
        if depth == 0 && self.handle.locking() {
            // Tell the vCPUs what the device model now has waiting for them before anyone else
            // can change it.
            let devices = unsafe { &mut *self.handle.shared.devices.get() };
            devices.publish_wake_hints();
            self.handle.shared.held.store(false, Ordering::Release);
        }
    }
}

// Tests reach into the devices directly (`cpu.dev.pic.raise(1)`); they run on one thread.
#[cfg(test)]
impl Deref for DevHandle {
    type Target = Devices;
    fn deref(&self) -> &Devices {
        unsafe { &*self.shared.devices.get() }
    }
}

#[cfg(test)]
impl DerefMut for DevHandle {
    fn deref_mut(&mut self) -> &mut Devices {
        unsafe { &mut *self.shared.devices.get() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::ConsoleHandle;
    use crate::dev::DevConfig;

    fn handle() -> DevHandle {
        DevHandle::new(Devices::new(ConsoleHandle::default(), DevConfig::default()))
    }

    #[test]
    fn nested_locks_on_one_handle_do_not_deadlock() {
        let handle = handle();
        handle.set_locking(true);
        let outer = handle.lock();
        {
            let inner = handle.lock();
            drop(inner);
        }
        drop(outer);
        drop(handle.lock());
    }

    #[test]
    fn the_lock_is_released_after_the_outermost_guard() {
        let first = handle();
        first.set_locking(true);
        let second = first.for_cpu(0);
        drop(first.lock());
        // Would spin forever if the first guard had not released the lock.
        drop(second.lock());
    }

    #[test]
    fn a_handle_for_a_cpu_selects_that_cpus_apic() {
        let base = DevHandle::new(Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                cpus: 2,
                ..DevConfig::default()
            },
        ));
        let second = base.for_cpu(1);
        assert_eq!(second.lock().lapic().id(), 1);
        assert_eq!(base.lock().lapic().id(), 0);
    }
}
