//! `std::sync` on hosted targets; on UEFI there is one thread, so these are minimal stand-ins.
//! The stand-ins keep `std`'s call shapes (`lock().unwrap()`, `wait_timeout(..).unwrap().0`) so
//! shared code such as the console and net queues compiles unchanged on both.

#[cfg(not(freestanding))]
pub(crate) use std::sync::{Arc, Condvar, Mutex};

/// Lets another thread run while this one waits for a lock. A freestanding host has no scheduler
/// to offer the slice to, so it only tells the CPU it is spinning.
pub(crate) fn yield_now() {
    #[cfg(not(freestanding))]
    std::thread::yield_now();
    #[cfg(freestanding)]
    core::hint::spin_loop();
}

#[cfg(freestanding)]
pub(crate) use alloc::sync::Arc;
#[cfg(freestanding)]
pub(crate) use uefi_stubs::{Condvar, Mutex};

// Also built for tests so the stand-ins' logic runs on the host; nothing outside the tests uses
// them there.
#[cfg(any(freestanding, test))]
mod uefi_stubs {
    use core::cell::UnsafeCell;
    use core::convert::Infallible;
    use core::ops::{Deref, DerefMut};
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::time::Duration;

    /// Spinlock. With one thread it is never contended; locking it twice on that thread (for
    /// example a console callback calling back into the console) would spin forever.
    #[derive(Default)]
    pub(crate) struct Mutex<T> {
        locked: AtomicBool,
        value: UnsafeCell<T>,
    }

    // The `locked` flag serializes every access to `value`.
    unsafe impl<T: Send> Send for Mutex<T> {}
    unsafe impl<T: Send> Sync for Mutex<T> {}

    pub(crate) struct Guard<'a, T>(&'a Mutex<T>);

    impl<T> Mutex<T> {
        #[cfg_attr(not(freestanding), allow(dead_code))]
        pub fn new(value: T) -> Self {
            Self {
                locked: AtomicBool::new(false),
                value: UnsafeCell::new(value),
            }
        }

        pub fn lock(&self) -> Result<Guard<'_, T>, Infallible> {
            while self.locked.swap(true, Ordering::Acquire) {
                core::hint::spin_loop();
            }
            Ok(Guard(self))
        }
    }

    impl<T> Deref for Guard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            unsafe { &*self.0.value.get() }
        }
    }

    impl<T> DerefMut for Guard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            unsafe { &mut *self.0.value.get() }
        }
    }

    impl<T> Drop for Guard<'_, T> {
        fn drop(&mut self) {
            self.0.locked.store(false, Ordering::Release);
        }
    }

    /// Nothing can signal it: there is no other thread. `wait_timeout` returns at once, so callers
    /// see "timed out" immediately and the single-threaded host polls instead (`MachineConfig::poll`).
    #[derive(Default)]
    pub(crate) struct Condvar;

    impl Condvar {
        /// `std`'s constructor, so shared code builds one the same way on both.
        #[cfg_attr(not(freestanding), allow(dead_code))]
        pub const fn new() -> Self {
            Condvar
        }

        /// The `()` stands in for `std`'s `WaitTimeoutResult`, which no caller inspects.
        pub fn wait_timeout<'a, T>(
            &self,
            guard: Guard<'a, T>,
            _timeout: Duration,
        ) -> Result<(Guard<'a, T>, ()), Infallible> {
            Ok((guard, ()))
        }

        pub fn notify_one(&self) {}

        pub fn notify_all(&self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::uefi_stubs::{Condvar, Mutex};
    use std::sync::Arc;
    use std::time::Duration;

    const THREADS: usize = 4;
    const INCREMENTS_PER_THREAD: usize = 5_000;

    #[test]
    fn a_new_mutex_holds_the_default_value() {
        let mutex: Mutex<u32> = Mutex::default();
        assert_eq!(*mutex.lock().unwrap(), 0);
    }

    #[test]
    fn the_guard_reads_and_writes_the_value() {
        let mutex = Mutex::<Vec<u8>>::default();
        mutex.lock().unwrap().push(7);
        mutex.lock().unwrap().push(8);
        assert_eq!(*mutex.lock().unwrap(), [7, 8]);
    }

    #[test]
    fn the_lock_is_released_when_the_guard_drops() {
        let mutex = Mutex::<u32>::default();
        for round in 1..=3 {
            let mut guard = mutex.lock().unwrap();
            *guard += 1;
            assert_eq!(*guard, round);
        }
    }

    #[test]
    fn the_lock_is_released_when_the_holder_panics() {
        let mutex = Arc::new(Mutex::<u32>::default());
        let crashing = mutex.clone();
        let result = std::thread::spawn(move || {
            let mut guard = crashing.lock().unwrap();
            *guard = 5;
            panic!("holder dies");
        })
        .join();
        assert!(result.is_err());
        // No poisoning on the stand-in, and no deadlock: the guard's drop ran during unwinding.
        assert_eq!(*mutex.lock().unwrap(), 5);
    }

    #[test]
    fn the_spinlock_gives_mutual_exclusion_across_threads() {
        let mutex = Arc::new(Mutex::<usize>::default());
        let threads: Vec<_> = (0..THREADS)
            .map(|_| {
                let mutex = mutex.clone();
                std::thread::spawn(move || {
                    for _ in 0..INCREMENTS_PER_THREAD {
                        // A non-atomic read-modify-write: lost updates show up if the lock leaks.
                        let mut guard = mutex.lock().unwrap();
                        let current = *guard;
                        std::hint::black_box(current);
                        *guard = current + 1;
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*mutex.lock().unwrap(), THREADS * INCREMENTS_PER_THREAD);
    }

    #[test]
    fn wait_timeout_returns_at_once_with_the_guard_still_holding_the_lock() {
        let mutex = Mutex::<u32>::default();
        let condvar = Condvar;
        let guard = mutex.lock().unwrap();
        let (mut guard, ()) = condvar
            .wait_timeout(guard, Duration::from_secs(3600))
            .unwrap();
        *guard = 9;
        drop(guard);
        assert_eq!(*mutex.lock().unwrap(), 9);
    }

    #[test]
    fn notify_does_nothing_and_does_not_fail() {
        let condvar = Condvar;
        condvar.notify_one();
        condvar.notify_one();
        condvar.notify_all();
    }

    #[test]
    fn hosted_builds_use_the_std_types() {
        // The call shapes the shared code relies on compile against both implementations.
        let mutex = super::Mutex::new(1u8);
        let condvar = super::Condvar::new();
        let guard = mutex.lock().unwrap();
        let (guard, _) = condvar.wait_timeout(guard, Duration::ZERO).unwrap();
        assert_eq!(*guard, 1);
        let shared = super::Arc::new(2u8);
        assert_eq!(*shared, 2);
    }
}
