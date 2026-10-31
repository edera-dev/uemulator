//! Embeddable x86_64 PC emulator that boots an unmodified Linux bzImage. Start from
//! `MachineConfig` and `EmulatedMachine`.
//!
//! The cfg gates: `freestanding` (set by `build.rs` for UEFI and wasm32) selects the `no_std`
//! build, which has no threads and no
//! `std::fs` (so no `disk`) and needs `prelude` and `math` to replace what `std` provides. The
//! `jit` feature enables `sys` (executable memory); without it the crate has no dependencies and
//! only interprets.

#![cfg_attr(freestanding, no_std)]

extern crate alloc;

mod acpi;
mod boot;
mod console;
mod cpu;
mod dev;
#[cfg(not(freestanding))]
mod disk;
mod jit;
mod kv;
mod machine;
#[cfg(freestanding)]
mod math;
mod netdev;
mod platform;
#[cfg(freestanding)]
mod prelude;
mod ram;
mod sync;
#[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
mod sys;
#[cfg(test)]
mod testkit;

pub use boot::{BootMode, BootState};
pub use console::ConsoleHandle;
#[cfg(not(freestanding))]
pub use disk::{BlockBackend, Disk};
#[cfg(not(freestanding))]
pub use kv::FileKeyValueStore;
pub use kv::{KeyValueError, KeyValueStore, MemoryKeyValueStore};
pub use machine::{
    ClockMode, ConsoleKind, EmulatedMachine, Error, ExitReason, JitMode, MAX_MEMORY_MIB,
    MIN_MEMORY_MIB, MachineConfig, SmpMode, StopHandle,
};
pub use netdev::{NetHandle, NetOffloads};
pub use platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE, Platform, load_linux};

#[cfg(test)]
mod api_tests {
    //! Compile-time pins of the public surface: a signature or derive change that would break an
    //! embedder fails to build here.
    use super::*;

    type LoadLinux = fn(&mut [u8], &[u8], Option<&[u8]>, &str) -> Result<BootState, Error>;

    fn is_send<T: Send>() {}
    fn is_send_sync<T: Send + Sync>() {}
    fn is_clone<T: Clone>() {}
    fn is_copy<T: Copy>() {}
    fn is_default<T: Default>() {}
    fn is_debug<T: core::fmt::Debug>() {}
    fn is_std_error<T: std::error::Error>() {}

    #[test]
    fn entry_points_have_the_documented_signatures() {
        let _: fn(MachineConfig) -> Result<EmulatedMachine, Error> = EmulatedMachine::new;
        let _: fn(&mut EmulatedMachine) -> Result<(), Error> = EmulatedMachine::start;
        let _: fn(&mut EmulatedMachine) -> ExitReason = EmulatedMachine::wait;
        let _: fn(&EmulatedMachine) = EmulatedMachine::stop;
        let _: fn(&EmulatedMachine) -> StopHandle = EmulatedMachine::stop_handle;
        let _: fn(&EmulatedMachine) -> ConsoleHandle = EmulatedMachine::console;
        let _: fn(&EmulatedMachine) -> Option<NetHandle> = EmulatedMachine::net;
        let _: fn(&NetHandle, &[u8]) -> bool = NetHandle::send_frame;
        let _: fn(&NetHandle) -> Option<Vec<u8>> = NetHandle::try_recv_frame;
        let _ = |net: &NetHandle| net.on_guest_frame(|| {});
        let _: fn(&NetHandle, &[u8], u16) -> bool = NetHandle::send_tcp_segment;
        let _: fn(&NetHandle, NetOffloads) = NetHandle::set_offloads;
        let _: fn(&NetHandle) -> bool = NetHandle::guest_takes_tcp_segments;
        let _ = NetOffloads {
            takes_guest_segments: true,
            sends_large_segments: true,
        };
        let _: fn(&StopHandle) = StopHandle::stop;
        let _: fn() -> MachineConfig = MachineConfig::new;
        let _: fn(MachineConfig, bool) -> MachineConfig = MachineConfig::reboot;
        let _: LoadLinux = load_linux;
        let _: &str = DEFAULT_CMDLINE;
        let _: &str = DEFAULT_SERIAL_CMDLINE;
        let _: fn(ConsoleHandle) -> Platform = Platform::new;
    }

    #[test]
    fn handles_cross_threads_and_clone() {
        is_send::<MachineConfig>();
        is_send::<EmulatedMachine>();
        is_send_sync::<ConsoleHandle>();
        is_send_sync::<NetHandle>();
        is_send_sync::<StopHandle>();
        is_send_sync::<Error>();
        is_send_sync::<ExitReason>();
        is_clone::<ConsoleHandle>();
        is_clone::<NetHandle>();
        is_clone::<StopHandle>();
        is_send::<Platform>();
        is_send::<Disk>();
    }

    #[test]
    fn value_types_keep_their_derives() {
        is_copy::<JitMode>();
        is_copy::<ClockMode>();
        is_copy::<ConsoleKind>();
        is_copy::<BootState>();
        is_copy::<NetOffloads>();
        is_default::<NetOffloads>();
        is_default::<JitMode>();
        is_default::<ClockMode>();
        is_default::<ConsoleKind>();
        is_default::<BootState>();
        is_default::<ConsoleHandle>();
        is_default::<NetHandle>();
        is_default::<MachineConfig>();
        is_clone::<ExitReason>();
        is_debug::<Error>();
        is_debug::<ExitReason>();
        is_debug::<BootState>();
        is_std_error::<Error>();
    }

    #[test]
    fn boot_state_fields_are_public() {
        let state = BootState {
            rip: 1,
            rsi: 2,
            rdi: 0,
            cr3: 3,
            gdt_base: 4,
            gdt_limit: 5,
            cs: 6,
            ds: 7,
            mode: BootMode::Long64,
        };
        assert_eq!(
            (
                state.rip,
                state.rsi,
                state.cr3,
                state.gdt_base,
                state.gdt_limit,
                state.cs,
                state.ds
            ),
            (1, 2, 3, 4, 5, 6, 7)
        );
    }

    #[test]
    fn every_error_and_exit_variant_is_nameable() {
        let errors = [
            Error::BadKernel(String::new()),
            Error::BadConfig(String::new()),
            Error::JitUnavailable,
            Error::AlreadyStarted,
        ];
        assert_eq!(errors.len(), 4);
        let exits = [
            ExitReason::Shutdown,
            ExitReason::Reset,
            ExitReason::TripleFault,
            ExitReason::Stopped,
            ExitReason::Internal(String::new()),
        ];
        assert_eq!(exits.len(), 5);
    }

    #[test]
    fn a_block_backend_can_be_implemented_outside_the_crate_shape() {
        struct Nothing;
        impl BlockBackend for Nothing {
            fn len(&self) -> u64 {
                0
            }
            fn read_at(&mut self, _: u64, _: &mut [u8]) -> std::io::Result<()> {
                Ok(())
            }
            fn write_at(&mut self, _: u64, _: &[u8]) -> std::io::Result<()> {
                Ok(())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let _config = MachineConfig::new().disk(Disk::from_backend(Nothing));
    }

    #[test]
    fn the_memory_limits_are_visible_to_embedders_through_the_error_message() {
        let message = match EmulatedMachine::new(MachineConfig::new().memory_mib(1)) {
            Err(Error::BadConfig(message)) => message,
            other => panic!("unexpected {:?}", other.err()),
        };
        assert!(message.contains("64") && message.contains("1047552"));
    }
}
