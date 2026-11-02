//! The 3D half of virtio-gpu (`VIRTIO_GPU_F_VIRGL`), present when the embedder hands the machine
//! a `GpuLink`: contexts, 3D resources, transfers, command streams, and fences held until the
//! renderer has passed them (`docs/plans/25-gpu-3d.md`). The device checks every request and
//! copies guest memory in or out, and forwards the rest as `GpuCommand`s; it never interprets
//! the virgl stream. Behaviour follows QEMU's `hw/display/virtio-gpu-virgl.c`.
//!
//! Fences: Linux takes the response to a fenced command as "this fence and every earlier one of
//! the same timeline are done" (`virtio_gpu_fence_event_process`), and without `CONTEXT_INIT`
//! every fence is on one timeline. So with a renderer attached, every response that carries a
//! fence (2D commands and errors too) waits behind a `Fence` in the renderer's queue, as QEMU
//! sends every fenced command through `virgl_renderer_create_fence`. Answering a 2D fence early
//! would signal a 3D fence the renderer has not reached.
//!
//! 2D resources keep their CPU path. A context may attach one (Linux attaches every GEM object
//! of a file to its context, the fbdev console's dumb buffer included), but the renderer never
//! hears of it: nothing yet copies a 2D image to the renderer.

use alloc::collections::{BTreeMap, BTreeSet};

use super::formats::{self, Layout};
use super::{
    BackingCursor, BackingEntry, Chain, GuestMem, HEADER_BYTES, Header, Queue,
    RESP_ERR_INVALID_PARAMETER, RESP_ERR_INVALID_RESOURCE_ID, RESP_ERR_OUT_OF_MEMORY,
    RESP_ERR_UNSPEC, RESP_OK_NODATA, u32_at, u64_at,
};
use crate::display::{MAX_HEIGHT, MAX_WIDTH, Rect};
use crate::gpu::{GpuBox, GpuCommand, GpuCompletion, GpuLink, GpuResource};
#[cfg(freestanding)]
use crate::prelude::*;

pub(super) const CMD_GET_CAPSET_INFO: u32 = 0x0108;
pub(super) const CMD_GET_CAPSET: u32 = 0x0109;
pub(super) const CMD_CTX_CREATE: u32 = 0x0200;
pub(super) const CMD_CTX_DESTROY: u32 = 0x0201;
pub(super) const CMD_CTX_ATTACH_RESOURCE: u32 = 0x0202;
pub(super) const CMD_CTX_DETACH_RESOURCE: u32 = 0x0203;
pub(super) const CMD_RESOURCE_CREATE_3D: u32 = 0x0204;
pub(super) const CMD_TRANSFER_TO_HOST_3D: u32 = 0x0205;
pub(super) const CMD_TRANSFER_FROM_HOST_3D: u32 = 0x0206;
pub(super) const CMD_SUBMIT_3D: u32 = 0x0207;
/// The context commands: `CTX_CREATE` to `SUBMIT_3D`. Blobs (`RESOURCE_MAP_BLOB`) are not offered.
pub(super) const CONTEXT_COMMANDS: core::ops::RangeInclusive<u32> = CMD_CTX_CREATE..=CMD_SUBMIT_3D;
const RESP_OK_CAPSET_INFO: u32 = 0x1102;
const RESP_OK_CAPSET: u32 = 0x1103;
const RESP_ERR_INVALID_CONTEXT_ID: u32 = 0x1204;

/// `struct virtio_gpu_ctx_create` after the header: nlen, context_init, debug_name[64].
const CONTEXT_NAME_LENGTH: usize = 0;
const CONTEXT_INIT: usize = 4;
const CONTEXT_NAME: usize = 8;
const MAX_CONTEXT_NAME: usize = 64;
/// `struct virtio_gpu_transfer_host_3d` after the header: box (x, y, z, w, h, d), offset u64,
/// resource_id, level, stride, layer_stride.
const TRANSFER_OFFSET: usize = 24;
const TRANSFER_RESOURCE: usize = 32;
const TRANSFER_LEVEL: usize = 36;
const TRANSFER_STRIDE: usize = 40;
const TRANSFER_LAYER_STRIDE: usize = 44;
/// `struct virtio_gpu_cmd_submit`: size, padding, then `size` bytes of stream.
const SUBMIT_FIXED_BYTES: usize = 8;
/// The largest fixed part of any request after its header: `ctx_create` (8 + 64), larger than
/// `resource_create_3d` (48) and `transfer_host_3d` (48).
pub(super) const LARGEST_FIXED_BODY: usize = CONTEXT_NAME + MAX_CONTEXT_NAME;

/// Contexts one guest may hold; Linux makes one per open DRM file.
const MAX_CONTEXTS: usize = 4096;
/// Largest `SUBMIT_3D` stream. Mesa's virgl driver flushes at 64 Ki dwords
/// (`VIRGL_MAX_CMDBUF_DWORDS`); this leaves room for drivers that batch more.
const MAX_SUBMIT_BYTES: usize = 4 << 20;
/// Gallium's texture targets (`enum pipe_texture_target`).
const TARGET_BUFFER: u32 = 0;
const TARGET_TEXTURE_3D: u32 = 3;
const TARGET_CUBE_ARRAY: u32 = 8;
/// Most samples per pixel of a multisampled resource.
const MAX_SAMPLES: u32 = 32;

/// Where a transfer's box lies in a resource's backing, as virglrenderer resolves it
/// (`check_iov_bounds`): block row `r` of layer `z` starts at `offset + z * layer_stride +
/// r * stride`, and is `row_bytes` long.
#[derive(Clone, Copy)]
struct Placement {
    offset: u64,
    stride: u64,
    layer_stride: u64,
    row_bytes: u64,
    rows: u64,
    layers: u64,
}

impl Placement {
    fn packed_bytes(&self) -> u64 {
        self.row_bytes * self.rows * self.layers
    }

    /// Calls `each(backing offset, packed offset, length)` for every row, in packed order.
    fn for_each_row(&self, mut each: impl FnMut(u64, usize, usize)) {
        let mut packed = 0;
        for layer in 0..self.layers {
            for row in 0..self.rows {
                each(
                    self.offset + layer * self.layer_stride + row * self.stride,
                    packed,
                    self.row_bytes as usize,
                );
                packed += self.row_bytes as usize;
            }
        }
    }
}

struct Resource3d {
    description: GpuResource,
    layout: Layout,
    backing: Vec<BackingEntry>,
    /// What it counts for in `GpuLimits::max_resource_bytes`.
    bytes: u64,
}

impl Resource3d {
    fn level_width(&self, level: u32) -> u32 {
        minify(self.description.width, level)
    }

    fn level_height(&self, level: u32) -> u32 {
        minify(self.description.height, level)
    }

    /// Slices of a 3D texture at `level`, layers of anything else.
    fn level_layers(&self, level: u32) -> u32 {
        if self.description.target == TARGET_TEXTURE_3D {
            minify(self.description.depth, level)
        } else {
            self.description.array_size
        }
    }
}

/// A mip level's extent: halved per level, never below 1.
fn minify(size: u32, level: u32) -> u32 {
    size.checked_shr(level).unwrap_or(0).max(1)
}

/// A response the guest gets once the renderer passes `sequence`.
struct HeldResponse {
    chain: Chain,
    response: Vec<u8>,
    sequence: u64,
}

/// A `TRANSFER_FROM_HOST_3D` waiting for its bytes.
struct PendingRead {
    resource: u32,
    placement: Placement,
}

pub(super) struct Accel {
    link: GpuLink,
    /// Each context and the resources attached to it, 2D ones included.
    contexts: BTreeMap<u32, BTreeSet<u32>>,
    resources: BTreeMap<u32, Resource3d>,
    resource_bytes: u64,
    /// In the order the guest sent them; released in that order.
    held: Vec<HeldResponse>,
    reads: BTreeMap<u64, PendingRead>,
    /// The 3D resource on scanout 0; the 2D scanout is `VirtioGpu::scanout`.
    scanout: Option<u32>,
    /// Commands of the request being executed, queued together once it is done.
    outgoing: Vec<GpuCommand>,
    /// The control queue was left unpopped because the renderer was behind.
    stalled: bool,
}

impl Accel {
    /// Tells the renderer to start over: a new device (the link may have served one before a
    /// reboot) or a reset one.
    pub fn new(link: GpuLink) -> Accel {
        link.push(vec![GpuCommand::Reset]);
        Accel {
            link,
            contexts: BTreeMap::new(),
            resources: BTreeMap::new(),
            resource_bytes: 0,
            held: Vec::new(),
            reads: BTreeMap::new(),
            scanout: None,
            outgoing: Vec::new(),
            stalled: false,
        }
    }

    pub fn capset_count(&self) -> u32 {
        self.link.capsets().len() as u32
    }

    pub fn has_resource(&self, id: u32) -> bool {
        self.resources.contains_key(&id)
    }

    pub fn backing_mut(&mut self, id: u32) -> Option<&mut Vec<BackingEntry>> {
        self.resources
            .get_mut(&id)
            .map(|resource| &mut resource.backing)
    }

    /// Whether the control queue should be left alone for now.
    pub fn is_behind(&mut self) -> bool {
        self.stalled = self.link.is_full();
        self.stalled
    }

    /// Whether the control queue was left unpopped and the renderer has since caught up.
    pub fn may_resume(&mut self) -> bool {
        self.stalled && !self.is_behind()
    }

    /// `capset_index`: the id, highest version and size of that capset. An index past the end
    /// gets zeros, as from QEMU, which the driver takes as "no such capset".
    pub fn capset_info(&self, header: &Header, body: &[u8]) -> Vec<u8> {
        let mut response = header.response(RESP_OK_CAPSET_INFO);
        let (id, version, size) = match self.link.capsets().get(u32_at(body, 0) as usize) {
            Some(capset) => (capset.id, capset.max_version, capset.data.len() as u32),
            None => (0, 0, 0),
        };
        for word in [id, version, size, 0] {
            response.extend_from_slice(&word.to_le_bytes());
        }
        response
    }

    /// `capset_id`, `capset_version`.
    pub fn capset(&self, header: &Header, body: &[u8]) -> Vec<u8> {
        let (id, version) = (u32_at(body, 0), u32_at(body, 4));
        match self.link.capsets().iter().find(|capset| capset.id == id) {
            Some(capset) if version <= capset.max_version => {
                let mut response = header.response(RESP_OK_CAPSET);
                response.extend_from_slice(&capset.data);
                response
            }
            _ => header.response(RESP_ERR_INVALID_PARAMETER),
        }
    }

    /// One of `CONTEXT_COMMANDS`; `body` is the fixed part after the header.
    pub fn execute(
        &mut self,
        header: &Header,
        body: &[u8],
        chain: &Chain,
        mem: &GuestMem,
        is_2d_resource: impl Fn(u32) -> bool,
    ) -> u32 {
        let context = header.context_id;
        match header.command {
            CMD_CTX_CREATE => self.create_context(context, body),
            CMD_CTX_DESTROY => match self.contexts.remove(&context) {
                Some(_) => {
                    self.outgoing.push(GpuCommand::DestroyContext { context });
                    RESP_OK_NODATA
                }
                None => RESP_ERR_INVALID_CONTEXT_ID,
            },
            CMD_CTX_ATTACH_RESOURCE => {
                let resource = u32_at(body, 0);
                let is_3d = self.resources.contains_key(&resource);
                let Some(attached) = self.contexts.get_mut(&context) else {
                    return RESP_ERR_INVALID_CONTEXT_ID;
                };
                if !is_3d && !is_2d_resource(resource) {
                    return RESP_ERR_INVALID_RESOURCE_ID;
                }
                if attached.insert(resource) && is_3d {
                    self.outgoing
                        .push(GpuCommand::AttachResource { context, resource });
                }
                RESP_OK_NODATA
            }
            CMD_CTX_DETACH_RESOURCE => {
                let resource = u32_at(body, 0);
                let Some(attached) = self.contexts.get_mut(&context) else {
                    return RESP_ERR_INVALID_CONTEXT_ID;
                };
                if attached.remove(&resource) && self.resources.contains_key(&resource) {
                    self.outgoing
                        .push(GpuCommand::DetachResource { context, resource });
                }
                RESP_OK_NODATA
            }
            CMD_RESOURCE_CREATE_3D => self.create_resource(body, is_2d_resource),
            CMD_TRANSFER_TO_HOST_3D => self.transfer_to_host(context, body, mem),
            CMD_TRANSFER_FROM_HOST_3D => self.transfer_from_host(context, body),
            CMD_SUBMIT_3D => self.submit(context, body, chain, mem),
            _ => RESP_ERR_UNSPEC,
        }
    }

    fn create_context(&mut self, context: u32, body: &[u8]) -> u32 {
        if context == 0 || self.contexts.contains_key(&context) {
            return RESP_ERR_INVALID_CONTEXT_ID;
        }
        // A capset id in `context_init` asks for `CONTEXT_INIT`, which is not offered.
        if u32_at(body, CONTEXT_INIT) != 0 {
            return RESP_ERR_INVALID_PARAMETER;
        }
        if self.contexts.len() >= MAX_CONTEXTS {
            return RESP_ERR_OUT_OF_MEMORY;
        }
        let length = (u32_at(body, CONTEXT_NAME_LENGTH) as usize).min(MAX_CONTEXT_NAME);
        let name = body
            .get(CONTEXT_NAME..CONTEXT_NAME + length)
            .unwrap_or_default()
            .to_vec();
        self.contexts.insert(context, BTreeSet::new());
        self.outgoing
            .push(GpuCommand::CreateContext { context, name });
        RESP_OK_NODATA
    }

    /// `struct virtio_gpu_resource_create_3d`. Checked as a renderer would before allocating:
    /// a known format and target, sizes within the renderer's limits and the byte budget.
    fn create_resource(&mut self, body: &[u8], is_2d_resource: impl Fn(u32) -> bool) -> u32 {
        let word = |index: usize| u32_at(body, index * 4);
        let description = GpuResource {
            id: word(0),
            target: word(1),
            format: word(2),
            bind: word(3),
            width: word(4),
            height: word(5),
            depth: word(6),
            array_size: word(7),
            last_level: word(8),
            samples: word(9),
            flags: word(10),
        };
        let id = description.id;
        if id == 0 || self.resources.contains_key(&id) || is_2d_resource(id) {
            return RESP_ERR_INVALID_RESOURCE_ID;
        }
        let Some(layout) = formats::layout(description.format) else {
            return RESP_ERR_INVALID_PARAMETER;
        };
        let Some(bytes) = self.resource_size(&description, layout) else {
            return RESP_ERR_INVALID_PARAMETER;
        };
        let budget = self.link.limits().max_resource_bytes;
        if self.resource_bytes.saturating_add(bytes) > budget {
            return RESP_ERR_OUT_OF_MEMORY;
        }
        self.resource_bytes += bytes;
        self.resources.insert(
            id,
            Resource3d {
                description,
                layout,
                backing: Vec::new(),
                bytes,
            },
        );
        self.outgoing.push(GpuCommand::CreateResource(description));
        RESP_OK_NODATA
    }

    /// Bytes of every level, layer and sample, or None if the description breaks the limits.
    fn resource_size(&self, description: &GpuResource, layout: Layout) -> Option<u64> {
        let limits = self.link.limits();
        let GpuResource {
            target,
            width,
            height,
            depth,
            array_size,
            last_level,
            samples,
            ..
        } = *description;
        if target > TARGET_CUBE_ARRAY
            || width == 0
            || height == 0
            || depth == 0
            || array_size == 0
            || samples > MAX_SAMPLES
        {
            return None;
        }
        let fits = match target {
            TARGET_BUFFER => {
                width <= limits.max_buffer_bytes
                    && height == 1
                    && depth == 1
                    && array_size == 1
                    && last_level == 0
            }
            TARGET_TEXTURE_3D => {
                let largest = limits.max_texture_3d_size;
                width <= largest && height <= largest && depth <= largest && array_size == 1
            }
            _ => {
                width <= limits.max_texture_size
                    && height <= limits.max_texture_size
                    && depth == 1
                    && array_size <= limits.max_array_layers
            }
        };
        // A chain of levels stops at 1x1x1.
        let largest_side = width.max(height).max(depth);
        if !fits || last_level > largest_side.ilog2() {
            return None;
        }
        let mut total = 0u64;
        for level in 0..=last_level {
            let layers = if target == TARGET_TEXTURE_3D {
                minify(depth, level)
            } else {
                array_size
            };
            let level_bytes = layout
                .row_bytes(minify(width, level))
                .checked_mul(layout.rows(minify(height, level)))?
                .checked_mul(layers as u64)?;
            total = total.checked_add(level_bytes)?;
        }
        total.checked_mul(samples.max(1) as u64)
    }

    /// The resource a transfer names, if `context` may touch it.
    fn transfer_resource(&self, context: u32, id: u32) -> Result<&Resource3d, u32> {
        let attached = self
            .contexts
            .get(&context)
            .ok_or(RESP_ERR_INVALID_CONTEXT_ID)?;
        let resource = self
            .resources
            .get(&id)
            .filter(|_| attached.contains(&id))
            .ok_or(RESP_ERR_INVALID_RESOURCE_ID)?;
        if resource.backing.is_empty() {
            return Err(RESP_ERR_UNSPEC);
        }
        Ok(resource)
    }

    /// Where the box of a `transfer_host_3d` lies in the resource's backing, with virglrenderer's
    /// checks (`check_transfer_bounds`, `check_iov_bounds`): the box inside the level, strides at
    /// least what the box needs (0 meaning the level's own), and the whole span inside the
    /// backing. All sums are checked, so a hostile box cannot wrap past them.
    fn placement(resource: &Resource3d, body: &[u8], region: &GpuBox) -> Option<Placement> {
        let level = u32_at(body, TRANSFER_LEVEL);
        if level > resource.description.last_level {
            return None;
        }
        let within = |start: u32, extent: u32, limit: u32| {
            start.checked_add(extent).is_some_and(|end| end <= limit)
        };
        if !within(region.x, region.width, resource.level_width(level))
            || !within(region.y, region.height, resource.level_height(level))
            || !within(region.z, region.depth, resource.level_layers(level))
        {
            return None;
        }
        let layout = resource.layout;
        // An empty extent counts as one, as virglrenderer sizes it (`vrend_transfer_size`).
        let (width, height, depth) = (
            region.width.max(1),
            region.height.max(1),
            region.depth.max(1),
        );
        let row_bytes = layout.row_bytes(width);
        let rows = layout.rows(height);
        let stride = match u32_at(body, TRANSFER_STRIDE) as u64 {
            0 => layout.row_bytes(resource.level_width(level)),
            given if given < row_bytes => return None,
            given => given,
        };
        let layer_stride = match u32_at(body, TRANSFER_LAYER_STRIDE) as u64 {
            0 => layout.rows(resource.level_height(level)) * stride,
            given if given < rows * stride => return None,
            given => given,
        };
        let offset = u64_at(body, TRANSFER_OFFSET);
        let span = (depth as u64 - 1)
            .checked_mul(layer_stride)?
            .checked_add((rows - 1).checked_mul(stride)?)?
            .checked_add(row_bytes)?;
        let backing_bytes: u64 = resource.backing.iter().map(|entry| entry.length).sum();
        if offset.checked_add(span)? > backing_bytes {
            return None;
        }
        Some(Placement {
            offset,
            stride,
            layer_stride,
            row_bytes,
            rows,
            layers: depth as u64,
        })
    }

    fn transfer_box(body: &[u8]) -> GpuBox {
        let word = |index: usize| u32_at(body, index * 4);
        GpuBox {
            x: word(0),
            y: word(1),
            z: word(2),
            width: word(3),
            height: word(4),
            depth: word(5),
        }
    }

    /// The box's rows are read out of the backing now, packed, so the guest may reuse its pages
    /// as soon as it has the response.
    fn transfer_to_host(&mut self, context: u32, body: &[u8], mem: &GuestMem) -> u32 {
        let id = u32_at(body, TRANSFER_RESOURCE);
        let resource = match self.transfer_resource(context, id) {
            Ok(resource) => resource,
            Err(status) => return status,
        };
        let region = Self::transfer_box(body);
        let Some(placement) = Self::placement(resource, body, &region) else {
            return RESP_ERR_INVALID_PARAMETER;
        };
        // The box lies inside a resource that fit the budget, so this is bounded by it; a budget
        // past what a 32-bit host can address still fails cleanly here.
        let mut data = Vec::new();
        let Ok(bytes) = usize::try_from(placement.packed_bytes()) else {
            return RESP_ERR_OUT_OF_MEMORY;
        };
        if data.try_reserve_exact(bytes).is_err() {
            return RESP_ERR_OUT_OF_MEMORY;
        }
        data.resize(bytes, 0);
        placement.for_each_row(|from, to, length| {
            let mut cursor = BackingCursor::default();
            cursor.advance(&resource.backing, from);
            cursor.read(&resource.backing, mem, &mut data[to..to + length]);
        });
        self.outgoing.push(GpuCommand::WriteResource {
            context,
            resource: id,
            level: u32_at(body, TRANSFER_LEVEL),
            region,
            data,
        });
        RESP_OK_NODATA
    }

    /// Asks the renderer for the box; its bytes land in the backing when they come back, before
    /// the fence Linux always puts on this command lets the guest look.
    fn transfer_from_host(&mut self, context: u32, body: &[u8]) -> u32 {
        let id = u32_at(body, TRANSFER_RESOURCE);
        let resource = match self.transfer_resource(context, id) {
            Ok(resource) => resource,
            Err(status) => return status,
        };
        let region = Self::transfer_box(body);
        let Some(placement) = Self::placement(resource, body, &region) else {
            return RESP_ERR_INVALID_PARAMETER;
        };
        let token = self.link.next_sequence();
        self.reads.insert(
            token,
            PendingRead {
                resource: id,
                placement,
            },
        );
        self.outgoing.push(GpuCommand::ReadResource {
            context,
            resource: id,
            level: u32_at(body, TRANSFER_LEVEL),
            region,
            token,
        });
        RESP_OK_NODATA
    }

    fn submit(&mut self, context: u32, body: &[u8], chain: &Chain, mem: &GuestMem) -> u32 {
        if !self.contexts.contains_key(&context) {
            return RESP_ERR_INVALID_CONTEXT_ID;
        }
        let size = u32_at(body, 0) as usize;
        let start = HEADER_BYTES + SUBMIT_FIXED_BYTES;
        if size > MAX_SUBMIT_BYTES || start + size > chain.readable() {
            return RESP_ERR_INVALID_PARAMETER;
        }
        let mut commands = vec![0u8; size];
        chain.read_from(mem, start, &mut commands);
        self.outgoing.push(GpuCommand::Submit { context, commands });
        RESP_OK_NODATA
    }

    /// `SET_SCANOUT` naming a 3D resource: checked like a 2D one, then the renderer shows it.
    pub fn set_scanout(&mut self, id: u32, rect: Rect) -> u32 {
        let Some(resource) = self.resources.get(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        if rect.is_empty()
            || !rect_within(&rect, resource.level_width(0), resource.level_height(0))
            || rect.width > MAX_WIDTH
            || rect.height > MAX_HEIGHT
        {
            return RESP_ERR_INVALID_PARAMETER;
        }
        self.scanout = Some(id);
        self.outgoing.push(GpuCommand::SetScanout {
            scanout: 0,
            resource: id,
            rect,
        });
        RESP_OK_NODATA
    }

    /// The scanout went to a 2D resource or off: the renderer stops showing its resource.
    pub fn leave_scanout(&mut self) {
        if self.scanout.take().is_some() {
            self.outgoing.push(GpuCommand::SetScanout {
                scanout: 0,
                resource: 0,
                rect: Rect::default(),
            });
        }
    }

    /// `RESOURCE_FLUSH` naming a 3D resource; only the scanout's reaches the renderer.
    pub fn flush(&mut self, id: u32, rect: Rect) -> u32 {
        let Some(resource) = self.resources.get(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        if !rect_within(&rect, resource.level_width(0), resource.level_height(0)) {
            return RESP_ERR_INVALID_PARAMETER;
        }
        if self.scanout == Some(id) {
            self.outgoing.push(GpuCommand::Flush { resource: id, rect });
        }
        RESP_OK_NODATA
    }

    /// `RESOURCE_UNREF` of a 3D resource. A context that still has it attached loses it, and
    /// a readback still on its way has nowhere to go.
    pub fn unref(&mut self, id: u32) -> u32 {
        let Some(resource) = self.resources.remove(&id) else {
            return RESP_ERR_INVALID_RESOURCE_ID;
        };
        self.resource_bytes -= resource.bytes;
        self.reads.retain(|_, read| read.resource != id);
        self.forget(id);
        if self.scanout == Some(id) {
            self.leave_scanout();
        }
        self.outgoing
            .push(GpuCommand::DestroyResource { resource: id });
        RESP_OK_NODATA
    }

    /// A resource of either kind is gone: no context has it attached any more.
    pub fn forget(&mut self, id: u32) {
        for attached in self.contexts.values_mut() {
            attached.remove(&id);
        }
    }

    /// Queues what the request just executed produced. A response that carries a fence is held
    /// behind a `Fence` (see the module comment) and None comes back; otherwise the response.
    pub fn finish(
        &mut self,
        chain: Chain,
        response: Vec<u8>,
        fenced: bool,
    ) -> Option<(Chain, Vec<u8>)> {
        let held = if fenced {
            let sequence = self.link.next_sequence();
            self.outgoing.push(GpuCommand::Fence { sequence });
            self.held.push(HeldResponse {
                chain,
                response,
                sequence,
            });
            None
        } else {
            Some((chain, response))
        };
        self.link.push(core::mem::take(&mut self.outgoing));
        held
    }

    /// Takes in what the renderer reported, in order: readbacks are written to their backing,
    /// and every held response whose fence has passed is completed. True if any was.
    pub fn complete(&mut self, queue: &mut Queue, mem: &mut GuestMem) -> bool {
        let mut passed = None;
        for completion in self.link.take_completions() {
            match completion {
                GpuCompletion::Read { token, data } => self.write_back(token, &data, mem),
                // Every earlier fence passed too: the renderer works in order.
                GpuCompletion::Fence(sequence) => passed = passed.max(Some(sequence)),
            }
        }
        let Some(passed) = passed else {
            return false;
        };
        let mut used = Vec::new();
        self.held.retain(|held| {
            if held.sequence > passed {
                return true;
            }
            let written = held.chain.write_at(mem, 0, &held.response);
            used.push((held.chain.head, written as u32));
            false
        });
        queue.push_used_all(mem, &used);
        !used.is_empty()
    }

    /// The bytes of a readback into the backing the resource has now. Short data fills what it
    /// covers.
    fn write_back(&mut self, token: u64, data: &[u8], mem: &mut GuestMem) {
        let Some(read) = self.reads.remove(&token) else {
            return;
        };
        let Some(resource) = self.resources.get(&read.resource) else {
            return;
        };
        read.placement.for_each_row(|to, from, length| {
            let Some(row) = data.get(from..from + length) else {
                return;
            };
            let mut cursor = BackingCursor::default();
            cursor.advance(&resource.backing, to);
            cursor.write(&resource.backing, mem, row);
        });
    }

    /// The driver reset the device. Held responses are dropped without touching guest memory
    /// (the driver no longer owns those buffers); the renderer starts over.
    pub fn reset(&mut self) {
        *self = Accel::new(self.link.clone());
    }
}

fn rect_within(rect: &Rect, width: u32, height: u32) -> bool {
    rect.x
        .checked_add(rect.width)
        .is_some_and(|right| right <= width)
        && rect
            .y
            .checked_add(rect.height)
            .is_some_and(|bottom| bottom <= height)
}
