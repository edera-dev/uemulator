//! The host end of the virtio-gpu 3D path (`VIRTIO_GPU_F_VIRGL`, `docs/plans/25-gpu-3d.md`): the
//! device checks what the guest sends, copies what a renderer needs out of guest memory, and
//! queues it here as `GpuCommand`s; a renderer outside the library takes them, draws, and reports
//! fences and readbacks back. The library never renders and never parses the virgl stream: a
//! `Submit` carries it as the guest wrote it.
//!
//! The order of `take_commands` is the order the renderer must act in. A `Fence` is done once
//! everything before it is; the device holds the guest's response until then. A `ReadResource`
//! must be answered (`complete_read`) before any later fence is completed, so the guest finds the
//! bytes in its pages when it sees the fence. `Reset` means every context and resource the
//! renderer holds is gone: the device was reset, or a new one built for a reboot.

#[cfg(freestanding)]
use crate::prelude::*;
use alloc::collections::VecDeque;

use crate::display::Rect;
use crate::sync::{Arc, Mutex};

/// Called after commands are queued, outside the lock, so it may use the link itself.
type Notify = Arc<dyn Fn() + Send + Sync>;

/// Queued command bytes (payloads plus a fixed charge per command) past which the device stops
/// taking guest commands until the renderer catches up. The guest then sees a busy device, not
/// an error. One command may take it over the line, so a single large transfer cannot stall.
const QUEUE_BYTES: usize = 32 << 20;
/// What a command without a payload counts for in `QUEUE_BYTES`.
const COMMAND_BYTES: usize = 64;

/// Most capsets one renderer may offer; `struct virtio_gpu_config` has no limit, but Linux
/// fetches every one at probe, five seconds each when one goes unanswered.
pub(crate) const MAX_CAPSETS: usize = 16;
/// Largest capset; virgl's own (`struct virgl_caps_v2`) is about 1 KiB.
pub(crate) const MAX_CAPSET_BYTES: usize = 64 << 10;

/// One capability set the renderer offers (`GET_CAPSET_INFO`, `GET_CAPSET`). The guest asks for
/// them at driver probe, before any renderer could answer, so the bytes are fixed up front.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCapset {
    /// `VIRGL`, `VIRGL2` or another `VIRTIO_GPU_CAPSET_*` id; never 0.
    pub id: u32,
    /// Highest version a guest may ask for. The same bytes answer every version up to it.
    pub max_version: u32,
    /// What `GET_CAPSET` returns, at most 64 KiB: `struct virgl_caps_v1` for `VIRGL`,
    /// `virgl_caps_v2` for `VIRGL2`.
    pub data: Vec<u8>,
}

impl GpuCapset {
    pub const VIRGL: u32 = 1;
    pub const VIRGL2: u32 = 2;
}

/// What `RESOURCE_CREATE_3D` is checked against. The renderer holds the pixels, so these bound
/// its memory, not the emulator's; they should match what the capset tells the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLimits {
    /// Width and height of 1D, 2D, rectangle and cube textures (`max_texture_2d_size`).
    pub max_texture_size: u32,
    /// Width, height and depth of 3D textures (`max_texture_3d_size`).
    pub max_texture_3d_size: u32,
    /// Layers of array textures (`max_texture_array_layers`).
    pub max_array_layers: u32,
    /// Bytes of one buffer resource.
    pub max_buffer_bytes: u32,
    /// Every 3D resource together, all levels, layers and samples counted.
    pub max_resource_bytes: u64,
}

impl Default for GpuLimits {
    /// What WebGL2 on a desktop GPU commonly reports, and a gigabyte in all.
    fn default() -> Self {
        GpuLimits {
            max_texture_size: 16384,
            max_texture_3d_size: 2048,
            max_array_layers: 2048,
            max_buffer_bytes: 256 << 20,
            max_resource_bytes: 1 << 30,
        }
    }
}

/// A region of one level of a resource (`struct virtio_gpu_box`): x, y, z and the extent. `z`
/// and `depth` are layers for array and cube textures and slices for 3D ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBox {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// A resource as `RESOURCE_CREATE_3D` describes it, Gallium's terms: `target` is a
/// `PIPE_TEXTURE_*` (0 is a buffer), `format` a virgl format, `bind` the `VIRGL_BIND_*` uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuResource {
    pub id: u32,
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    /// Mip levels minus one.
    pub last_level: u32,
    pub samples: u32,
    /// `VIRTIO_GPU_RESOURCE_FLAG_Y_0_TOP` and future flags, passed through.
    pub flags: u32,
}

/// What the renderer is asked to do, in order. Data that came from guest memory is already
/// copied: the guest may reuse its pages as soon as it has its response.
///
/// Pixel data in `WriteResource` and `complete_read` is packed: block rows of
/// `ceil(width / block width) * block bytes` with no padding, `ceil(height / block height)` of
/// them per layer, `depth` layers; a buffer is `width` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuCommand {
    /// Drop every context and resource.
    Reset,
    /// `name` is the guest's debug name (its process name), at most 64 bytes.
    CreateContext {
        context: u32,
        name: Vec<u8>,
    },
    DestroyContext {
        context: u32,
    },
    /// The context may name the resource in its stream from now on.
    AttachResource {
        context: u32,
        resource: u32,
    },
    DetachResource {
        context: u32,
        resource: u32,
    },
    CreateResource(GpuResource),
    DestroyResource {
        resource: u32,
    },
    /// Guest bytes for `region` of `level` (`TRANSFER_TO_HOST_3D`), packed.
    WriteResource {
        context: u32,
        resource: u32,
        level: u32,
        region: GpuBox,
        data: Vec<u8>,
    },
    /// Read `region` of `level` back (`TRANSFER_FROM_HOST_3D`) and answer with `complete_read`
    /// and this `token`, packed, before completing any later fence.
    ReadResource {
        context: u32,
        resource: u32,
        level: u32,
        region: GpuBox,
        token: u64,
    },
    /// A virgl command stream (`SUBMIT_3D`), whole dwords, unparsed.
    Submit {
        context: u32,
        commands: Vec<u8>,
    },
    /// Show `rect` of `resource` on the scanout, or nothing when `resource` is 0 (the scanout
    /// went to a 2D resource, which the `DisplayHandle` shows, or was turned off).
    SetScanout {
        scanout: u32,
        resource: u32,
        rect: Rect,
    },
    /// `rect` of the scanout's resource changed.
    Flush {
        resource: u32,
        rect: Rect,
    },
    /// Report `sequence` with `complete_fence` once everything before it is done. Sequences
    /// only grow, across resets too.
    Fence {
        sequence: u64,
    },
}

impl GpuCommand {
    /// What it counts for in `QUEUE_BYTES`.
    fn queued_bytes(&self) -> usize {
        COMMAND_BYTES
            + match self {
                GpuCommand::WriteResource { data, .. } => data.len(),
                GpuCommand::Submit { commands, .. } => commands.len(),
                GpuCommand::CreateContext { name, .. } => name.len(),
                _ => 0,
            }
    }
}

/// What the renderer reported, in the order it reported it.
pub(crate) enum GpuCompletion {
    Fence(u64),
    Read { token: u64, data: Vec<u8> },
}

struct LinkState {
    commands: VecDeque<GpuCommand>,
    queued_bytes: usize,
    completions: VecDeque<GpuCompletion>,
    /// Next `Fence` sequence or `ReadResource` token. Kept here, not in the device, so a device
    /// rebuilt for a reboot never reuses a number a renderer may still answer late.
    next_sequence: u64,
    /// Wakes whichever thread runs the devices when a completion arrives (see `NetHandle`).
    wake_devices: Option<Notify>,
    /// Set by the host (`on_commands`).
    wake_host: Option<Notify>,
}

/// The queues between the virtio-gpu device and a renderer, shared by clones. Built before the
/// machine (`MachineConfig::gpu`) because its capsets must be ready when the guest probes.
#[derive(Clone)]
pub struct GpuLink {
    capsets: Arc<Vec<GpuCapset>>,
    limits: GpuLimits,
    state: Arc<Mutex<LinkState>>,
}

impl GpuLink {
    pub fn new(capsets: Vec<GpuCapset>, limits: GpuLimits) -> GpuLink {
        GpuLink {
            capsets: Arc::new(capsets),
            limits,
            state: Arc::new(Mutex::new(LinkState {
                commands: VecDeque::new(),
                queued_bytes: 0,
                completions: VecDeque::new(),
                next_sequence: 1,
                wake_devices: None,
                wake_host: None,
            })),
        }
    }

    pub fn capsets(&self) -> &[GpuCapset] {
        &self.capsets
    }

    pub fn limits(&self) -> GpuLimits {
        self.limits
    }

    /// Everything queued since the last call, oldest first.
    pub fn take_commands(&self) -> Vec<GpuCommand> {
        let mut state = self.state.lock().unwrap();
        state.queued_bytes = 0;
        state.commands.drain(..).collect()
    }

    /// Calls `notify` after the device queues commands, from the emulator's thread. For a host
    /// that waits on something else and so drains only when told. Replaces an earlier notifier.
    pub fn on_commands(&self, notify: impl Fn() + Send + Sync + 'static) {
        self.state.lock().unwrap().wake_host = Some(Arc::new(notify));
    }

    /// Everything up to and including the `Fence` with this sequence is done. A sequence never
    /// handed out is ignored.
    pub fn complete_fence(&self, sequence: u64) {
        self.complete(GpuCompletion::Fence(sequence));
    }

    /// The packed bytes a `ReadResource` asked for. A token the device no longer waits for (the
    /// resource is gone, or the device was reset) is ignored, and so are missing bytes.
    pub fn complete_read(&self, token: u64, data: Vec<u8>) {
        self.complete(GpuCompletion::Read { token, data });
    }

    fn complete(&self, completion: GpuCompletion) {
        let wake = {
            let mut state = self.state.lock().unwrap();
            if let GpuCompletion::Fence(sequence)
            | GpuCompletion::Read {
                token: sequence, ..
            } = completion
                && sequence >= state.next_sequence
            {
                return;
            }
            state.completions.push_back(completion);
            state.wake_devices.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }

    /// Whether the renderer is far enough behind that the device should take no more guest
    /// commands for now.
    pub(crate) fn is_full(&self) -> bool {
        self.state.lock().unwrap().queued_bytes >= QUEUE_BYTES
    }

    pub(crate) fn push(&self, commands: Vec<GpuCommand>) {
        if commands.is_empty() {
            return;
        }
        let wake = {
            let mut state = self.state.lock().unwrap();
            for command in commands {
                state.queued_bytes += command.queued_bytes();
                state.commands.push_back(command);
            }
            state.wake_host.clone()
        };
        if let Some(wake) = wake {
            wake();
        }
    }

    /// A number for a `Fence` or `ReadResource` no earlier one has had.
    pub(crate) fn next_sequence(&self) -> u64 {
        let mut state = self.state.lock().unwrap();
        let sequence = state.next_sequence;
        state.next_sequence += 1;
        sequence
    }

    pub(crate) fn take_completions(&self) -> VecDeque<GpuCompletion> {
        core::mem::take(&mut self.state.lock().unwrap().completions)
    }

    /// See `LinkState::wake_devices`. A freestanding host has no thread to wake.
    #[cfg(not(freestanding))]
    pub(crate) fn set_device_waker(&self, wake: Option<Notify>) {
        self.state.lock().unwrap().wake_devices = wake;
    }
}

impl core::fmt::Debug for GpuLink {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GpuLink")
            .field("capsets", &self.capsets.len())
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> GpuLink {
        GpuLink::new(Vec::new(), GpuLimits::default())
    }

    #[test]
    fn commands_come_out_in_order_and_empty_the_queue() {
        let link = link();
        link.push(vec![
            GpuCommand::Reset,
            GpuCommand::DestroyContext { context: 3 },
        ]);
        assert_eq!(
            link.take_commands(),
            [GpuCommand::Reset, GpuCommand::DestroyContext { context: 3 }]
        );
        assert!(link.take_commands().is_empty());
    }

    #[test]
    fn the_queue_is_full_until_the_renderer_takes_it() {
        let link = link();
        link.push(vec![GpuCommand::Submit {
            context: 1,
            commands: vec![0; QUEUE_BYTES],
        }]);
        assert!(link.is_full());
        link.take_commands();
        assert!(!link.is_full());
    }

    #[test]
    fn completions_for_numbers_never_handed_out_are_dropped() {
        let link = link();
        let issued = link.next_sequence();
        link.complete_fence(issued + 1);
        link.complete_read(issued + 5, vec![1]);
        link.complete_fence(issued);
        let completions = link.take_completions();
        assert_eq!(completions.len(), 1);
        assert!(matches!(completions[0], GpuCompletion::Fence(sequence) if sequence == issued));
    }

    #[test]
    fn the_host_is_told_when_commands_arrive() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        let link = link();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        link.on_commands(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        link.push(Vec::new());
        link.push(vec![GpuCommand::Reset]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
