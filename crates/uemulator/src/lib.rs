//! Embeddable x86_64 PC emulator that boots an unmodified Linux bzImage. Start from
//! `MachineConfig` and `EmulatedMachine`.
//!
//! The cfg gates: `freestanding` (set by `build.rs` for UEFI and wasm32) selects the `no_std`
//! build, which has no threads and no
//! `std::fs` (so no host files behind `disk`) and needs `prelude` and `math` to replace what `std` provides. The
//! `jit` feature enables `sys` (executable memory); without it the crate has no dependencies and
//! only interprets.

#![cfg_attr(freestanding, no_std)]

extern crate alloc;

mod acpi;
mod boot;
mod console;
mod cpu;
mod dev;
mod disk;
mod display;
mod gateway;
mod gpu;
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
mod segment;
mod sync;
#[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
mod sys;
#[cfg(test)]
mod testkit;

pub use boot::{BootMode, BootState};
pub use console::ConsoleHandle;
#[cfg(not(freestanding))]
pub use disk::ImageFile;
pub use disk::io as disk_io;
pub use disk::{
    BlockBackend, DEFAULT_CLUSTER_SIZE, Disk, FetchFn, ImageStorage, MAX_LOCATOR_BYTES,
    REMOTE_CHUNK_BYTES, RawImage, RemoteImage, Udisk, UdiskBacking, UdiskInfo, UdiskOptions,
    UdiskUsage,
};
pub use display::{
    BYTES_PER_PIXEL, Cursor, CursorImage, DisplayHandle, Frame, FrameUpdate, InputHandle,
    PointerButton, Rect, ScreenMirror, linux_key_for_usb_usage,
};
pub use gateway::{
    Endpoint, Gateway, GatewayConfig, GatewayPolicy, GatewayRequest, GuestFrame, StreamId,
    StreamInterest, StreamKind,
};
pub use gpu::{GpuBox, GpuCapset, GpuCommand, GpuLimits, GpuLink, GpuResource};
#[cfg(not(freestanding))]
pub use kv::FileKeyValueStore;
pub use kv::{KeyValueError, KeyValueStore, MemoryKeyValueStore};
pub use machine::{
    ClockMode, ConsoleKind, DEFAULT_HYPERVISOR_CMDLINE, DisplayMode, EmulatedMachine, Error,
    ExitReason, JitMode, MAX_MEMORY_MIB, MIN_MEMORY_MIB, MachineConfig, SmpMode, StopHandle,
};
pub use netdev::{NetHandle, NetOffloads};
pub use platform::{DEFAULT_CMDLINE, DEFAULT_SERIAL_CMDLINE, Platform, load_linux};
pub use segment::{SegmentAgent, SegmentDestination, SegmentStatus};

#[cfg(test)]
mod api_tests {
    //! Compile-time pins of the public surface: a signature or derive change that would break an
    //! embedder fails to build here.
    use super::*;

    type LoadLinux = fn(&mut [u8], &[u8], Option<&[u8]>, &str) -> Result<BootState, Error>;
    /// An overlay's base, as `Udisk::open` and `Udisk::create` take it.
    type Base = Option<Box<dyn BlockBackend>>;
    /// A frame for the segment and where it goes, as `SegmentAgent::take_for_segment` hands it.
    type Addressed = (SegmentDestination, Vec<u8>);

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
        let _: fn(Vec<u8>, Base) -> disk_io::Result<Udisk> = Udisk::open;
        let _: fn(Vec<u8>, &UdiskOptions, Base) -> disk_io::Result<Udisk> = Udisk::create;
        let _: fn(&mut dyn ImageStorage) -> disk_io::Result<Option<UdiskInfo>> = Udisk::probe;
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
        let _: fn([u8; 6], u32, u64) -> Result<SegmentAgent, Error> = SegmentAgent::new;
        let _: fn(&mut SegmentAgent, bool, u64) = SegmentAgent::set_serving;
        let _: fn(&mut SegmentAgent, &[u8], u64) = SegmentAgent::from_guest;
        let _: fn(&mut SegmentAgent, u32, &[u8], u64) = SegmentAgent::from_segment;
        let _: fn(&mut SegmentAgent, u64) = SegmentAgent::tick;
        let _: fn(&mut SegmentAgent) -> Vec<Vec<u8>> = SegmentAgent::take_for_guest;
        let _: fn(&mut SegmentAgent) -> Vec<Addressed> = SegmentAgent::take_for_segment;
        let _ = [
            SegmentDestination::Everyone,
            SegmentDestination::Tab(1),
            SegmentDestination::Server,
        ];
        let status: SegmentStatus = SegmentStatus {
            address: Some([10, 0, 2, 15]),
            peers: 1,
            serving: false,
        };
        let _: fn(&SegmentAgent) -> SegmentStatus = SegmentAgent::status;
        let _ = status;
        let _: fn(GatewayConfig) -> Gateway = Gateway::new;
        let _: fn(&mut Gateway, &[u8], u64) = Gateway::from_guest;
        let _: fn(&mut Gateway) -> Option<GuestFrame> = Gateway::pop_for_guest;
        let _: fn(&mut Gateway, GuestFrame) = Gateway::requeue_for_guest;
        let _: fn(&Gateway, &[u8]) -> bool = Gateway::takes_frame;
        let _: fn(&mut Gateway, bool) = Gateway::set_large_segments;
        let _: fn(&mut Gateway) -> Vec<GatewayRequest> = Gateway::take_requests;
        let _: fn(&mut Gateway, StreamId) = Gateway::opened;
        let _: fn(&mut Gateway, StreamId) = Gateway::failed;
        let _: fn(&mut Gateway, StreamId, &[u8]) = Gateway::remote_data;
        let _: fn(&mut Gateway, StreamId) = Gateway::remote_closed;
        let _: fn(&mut Gateway, StreamId) = Gateway::reset;
        let _: fn(&mut Gateway, u64) = Gateway::poll;
        let _: fn(&Gateway, StreamId) -> &[u8] = Gateway::to_remote;
        let _: fn(&mut Gateway, StreamId, usize) = Gateway::consumed;
        let _: fn(&Gateway, StreamId) -> usize = Gateway::room;
        let _: fn(&Gateway) -> Vec<(StreamId, StreamInterest)> = Gateway::interest;
        let _ = StreamInterest {
            read: true,
            write: false,
        };
        let resolver = Endpoint {
            address: [1, 1, 1, 1],
            port: 53,
        };
        let _ = GatewayConfig {
            policy: GatewayPolicy::Segment { resolver },
            isn_seed: 1,
        };
        let _ = GatewayPolicy::UserNet { dns: resolver };
        let _: fn(&StopHandle) = StopHandle::stop;
        let _: fn() -> MachineConfig = MachineConfig::new;
        let _: fn(MachineConfig, bool) -> MachineConfig = MachineConfig::reboot;
        let _: LoadLinux = load_linux;
        let _: &str = DEFAULT_CMDLINE;
        let _: &str = DEFAULT_SERIAL_CMDLINE;
        let _: fn(ConsoleHandle) -> Platform = Platform::new;
    }

    #[test]
    fn the_display_and_input_surface_has_the_documented_signatures() {
        let _: fn(MachineConfig, DisplayMode) -> MachineConfig = MachineConfig::display;
        let _: fn(MachineConfig, u32, u32) -> MachineConfig = MachineConfig::display_size;
        let _: fn(MachineConfig, bool) -> MachineConfig = MachineConfig::input;
        let _: fn(MachineConfig, bool) -> MachineConfig = MachineConfig::console_on_screen;
        let _: fn(&DisplayHandle) -> (u64, Cursor) = DisplayHandle::cursor;
        let _: fn(&InputHandle) -> (u32, u32) = InputHandle::pointer_area;
        let image = CursorImage {
            width: 1,
            height: 1,
            hot_x: 0,
            hot_y: 0,
            pixels: vec![0; BYTES_PER_PIXEL],
        };
        let _ = [Cursor::Host, Cursor::Hidden, Cursor::Image(image)];
        let _: fn(&EmulatedMachine) -> Option<DisplayHandle> = EmulatedMachine::display;
        let _: fn(&EmulatedMachine) -> Option<InputHandle> = EmulatedMachine::input;
        let _: fn(&DisplayHandle) -> (u32, u32) = DisplayHandle::size;
        let _: fn(&DisplayHandle) -> u64 = DisplayHandle::generation;
        let _: fn(&DisplayHandle) -> Frame = DisplayHandle::frame;
        let _: fn(&DisplayHandle, u64, &mut Vec<u8>) -> Option<FrameUpdate> =
            DisplayHandle::update_since;
        let _: fn(&DisplayHandle, &mut ScreenMirror) -> Option<Vec<Rect>> =
            DisplayHandle::sync_mirror;
        let _: fn() -> ScreenMirror = ScreenMirror::new;
        let _: fn(&ScreenMirror) -> (u32, u32) = ScreenMirror::size;
        let _: fn(&ScreenMirror) -> Option<u64> = ScreenMirror::generation;
        let _: fn(&ScreenMirror) -> &[u8] = ScreenMirror::pixels;
        let _: fn(&ScreenMirror) -> usize = ScreenMirror::stride;
        let _: fn(&ScreenMirror, Rect, &mut Vec<u8>) = ScreenMirror::copy_rect;
        let _: fn(&DisplayHandle, u64, std::time::Duration) -> u64 = DisplayHandle::wait_for_update;
        let _: fn(&DisplayHandle, u32, u32) = DisplayHandle::request_size;
        let _ = |display: &DisplayHandle| display.on_update(|_generation: u64| {});
        let _: fn(&InputHandle, u16, bool) -> bool = InputHandle::key;
        let _: fn(&InputHandle, u16, bool) -> bool = InputHandle::key_usb;
        let _: fn(&InputHandle, u32, u32) -> bool = InputHandle::pointer_move;
        let _: fn(&InputHandle, PointerButton, bool) -> bool = InputHandle::pointer_button;
        let _: fn(&InputHandle, i32) -> bool = InputHandle::scroll;
        let _: fn(&InputHandle) = InputHandle::release_all;
        let _: fn(u16) -> Option<u16> = linux_key_for_usb_usage;
        let _: usize = BYTES_PER_PIXEL;
        let update = FrameUpdate {
            generation: 1,
            width: 2,
            height: 3,
            rect: Rect {
                x: 0,
                y: 0,
                width: 2,
                height: 3,
            },
        };
        let frame = Frame {
            width: update.width,
            height: update.height,
            generation: update.generation,
            pixels: Vec::new(),
        };
        assert!(!update.rect.is_empty() && frame.pixels.is_empty());
        let _ = [
            PointerButton::Left,
            PointerButton::Right,
            PointerButton::Middle,
            PointerButton::Side,
            PointerButton::Extra,
        ];
        let _ = [DisplayMode::None, DisplayMode::Framebuffer];
    }

    #[test]
    fn the_gpu_renderer_surface_has_the_documented_signatures() {
        let _: fn(Vec<GpuCapset>, GpuLimits) -> GpuLink = GpuLink::new;
        let _: fn(MachineConfig, GpuLink) -> MachineConfig = MachineConfig::gpu;
        let _: fn(&GpuLink) -> &[GpuCapset] = GpuLink::capsets;
        let _: fn(&GpuLink) -> GpuLimits = GpuLink::limits;
        let _: fn(&GpuLink) -> Vec<GpuCommand> = GpuLink::take_commands;
        let _: fn(&GpuLink, u64) = GpuLink::complete_fence;
        let _: fn(&GpuLink, u64, Vec<u8>) = GpuLink::complete_read;
        let _ = |link: &GpuLink| link.on_commands(|| {});
        let _: [u32; 2] = [GpuCapset::VIRGL, GpuCapset::VIRGL2];
        let capset = GpuCapset {
            id: GpuCapset::VIRGL2,
            max_version: 2,
            data: Vec::new(),
        };
        let limits = GpuLimits {
            max_texture_size: 1,
            max_texture_3d_size: 1,
            max_array_layers: 1,
            max_buffer_bytes: 1,
            max_resource_bytes: 1,
        };
        let region = GpuBox {
            x: 0,
            y: 0,
            z: 0,
            width: 1,
            height: 1,
            depth: 1,
        };
        let resource = GpuResource {
            id: 1,
            target: 2,
            format: 1,
            bind: 0,
            width: 1,
            height: 1,
            depth: 1,
            array_size: 1,
            last_level: 0,
            samples: 0,
            flags: 0,
        };
        let rect = Rect::default();
        let commands = [
            GpuCommand::Reset,
            GpuCommand::CreateContext {
                context: 1,
                name: Vec::new(),
            },
            GpuCommand::DestroyContext { context: 1 },
            GpuCommand::AttachResource {
                context: 1,
                resource: 1,
            },
            GpuCommand::DetachResource {
                context: 1,
                resource: 1,
            },
            GpuCommand::CreateResource(resource),
            GpuCommand::DestroyResource { resource: 1 },
            GpuCommand::WriteResource {
                context: 1,
                resource: 1,
                level: 0,
                region,
                data: Vec::new(),
            },
            GpuCommand::ReadResource {
                context: 1,
                resource: 1,
                level: 0,
                region,
                token: 1,
            },
            GpuCommand::Submit {
                context: 1,
                commands: Vec::new(),
            },
            GpuCommand::SetScanout {
                scanout: 0,
                resource: 1,
                rect,
            },
            GpuCommand::Flush { resource: 1, rect },
            GpuCommand::Fence { sequence: 1 },
        ];
        assert_eq!(commands.len(), 13);
        let _ = (capset, limits);
    }

    #[test]
    fn gpu_types_cross_threads_and_keep_their_derives() {
        is_send_sync::<GpuLink>();
        is_clone::<GpuLink>();
        is_debug::<GpuLink>();
        is_send::<GpuCommand>();
        is_clone::<GpuCommand>();
        is_debug::<GpuCommand>();
        is_clone::<GpuCapset>();
        is_copy::<GpuLimits>();
        is_default::<GpuLimits>();
        is_copy::<GpuBox>();
        is_default::<GpuBox>();
        is_copy::<GpuResource>();
        is_default::<GpuResource>();
    }

    #[test]
    fn a_gpu_renderer_needs_a_display_and_capsets_the_guest_accepts() {
        let renderer = |capsets| GpuLink::new(capsets, GpuLimits::default());
        let capset = |id, bytes| GpuCapset {
            id,
            max_version: 1,
            data: vec![0; bytes],
        };
        let refused = [
            (MachineConfig::new().gpu(renderer(Vec::new())), "display"),
            (
                MachineConfig::new()
                    .display(DisplayMode::Framebuffer)
                    .gpu(renderer(vec![capset(0, 4)])),
                "nonzero",
            ),
            (
                MachineConfig::new()
                    .display(DisplayMode::Framebuffer)
                    .gpu(renderer(vec![capset(1, 4), capset(1, 4)])),
                "distinct",
            ),
            (
                MachineConfig::new()
                    .display(DisplayMode::Framebuffer)
                    .gpu(renderer(vec![capset(1, 1 << 20)])),
                "larger",
            ),
        ];
        for (config, words) in refused {
            match EmulatedMachine::new(config) {
                Err(Error::BadConfig(message)) => assert!(message.contains(words), "{message}"),
                other => panic!("expected BadConfig about {words}, got {:?}", other.err()),
            }
        }
    }

    #[test]
    fn display_and_input_handles_cross_threads_and_clone() {
        is_send_sync::<DisplayHandle>();
        is_send_sync::<InputHandle>();
        is_clone::<DisplayHandle>();
        is_clone::<InputHandle>();
        is_default::<DisplayHandle>();
        is_default::<InputHandle>();
        is_copy::<DisplayMode>();
        is_copy::<Rect>();
        is_copy::<PointerButton>();
        is_copy::<FrameUpdate>();
        is_default::<DisplayMode>();
        is_default::<Rect>();
        is_default::<Frame>();
        is_default::<ScreenMirror>();
        is_clone::<ScreenMirror>();
        is_debug::<Frame>();
        is_debug::<DisplayMode>();
        is_default::<Cursor>();
        is_debug::<Cursor>();
        is_clone::<CursorImage>();
    }

    #[test]
    fn handles_cross_threads_and_clone() {
        is_send::<MachineConfig>();
        is_send::<EmulatedMachine>();
        is_send_sync::<SegmentAgent>();
        is_send::<Gateway>();
        is_copy::<SegmentStatus>();
        is_copy::<SegmentDestination>();
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
        is_send::<Udisk>();
        is_send::<RawImage>();
        is_send::<RemoteImage>();
        is_send::<ImageFile>();
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
