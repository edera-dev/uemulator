//! The guest's 3D through virtio-gpu (`docs/plans/25-gpu-3d.md`, route A): the library's
//! `GpuLink` on this side, the page's render worker (`www/render.js`) on the other. The poll
//! hook encodes what the device queued into records and hands them to `host_gpu_send`, which
//! copies them into a `SharedArrayBuffer` ring the render worker reads; fences and readbacks come
//! back through a second ring and `host_gpu_receive`.
//!
//! A record is a `u32` kind, a `u32` payload length in bytes, and the payload padded to four
//! bytes; every number in it is a little-endian `u32` (a `u64` is two, low word first).
//! `www/gpu.js` decodes them and has the same kinds.
//!
//! Backpressure: while the ring has no room for what is already encoded, the pump takes nothing
//! more from the link, so the device's own limit (32 MiB queued) stops the guest. The ring is
//! larger than that, so only a single record bigger than the ring is ever refused outright, and
//! that one is dropped (the guest then draws without it).

use uemulator::{GpuBox, GpuCapset, GpuCommand, GpuLimits, GpuLink, GpuResource, Rect};
use uemulator_virgl::caps::{self, HostLimits};

// Command record kinds.
const RECORD_RESET: u32 = 1;
const RECORD_CREATE_CONTEXT: u32 = 2;
const RECORD_DESTROY_CONTEXT: u32 = 3;
const RECORD_ATTACH_RESOURCE: u32 = 4;
const RECORD_DETACH_RESOURCE: u32 = 5;
const RECORD_CREATE_RESOURCE: u32 = 6;
const RECORD_DESTROY_RESOURCE: u32 = 7;
const RECORD_WRITE_RESOURCE: u32 = 8;
const RECORD_READ_RESOURCE: u32 = 9;
const RECORD_SUBMIT: u32 = 10;
const RECORD_SET_SCANOUT: u32 = 11;
const RECORD_FLUSH: u32 = 12;
const RECORD_FENCE: u32 = 13;

// Completion record kinds.
const COMPLETION_FENCE: u32 = 1;
const COMPLETION_READ: u32 = 2;

const RECORD_HEADER_BYTES: usize = 8;
const WORD_BYTES: usize = 4;
/// Where completions are copied before they are decoded; grown when one is bigger.
const COMPLETION_BUFFER_BYTES: usize = 64 << 10;

/// The `HostLimits` fields `uw_set_gpu` takes, in declaration order, as `u32`; the two texel
/// offsets last, as `i32`.
pub(crate) const LIMIT_WORDS: usize = 14;
/// What the device holds a buffer resource and all resources together to; the page has no way
/// to ask WebGL how much memory there is.
const MAX_BUFFER_BYTES: u32 = 256 << 20;
const MAX_RESOURCE_BYTES: u64 = 1 << 30;

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    /// Copies as many whole records from `ptr..ptr+len` into the command ring as fit and
    /// returns how many bytes that was (0: none fit; try again later). A first record that could
    /// never fit is dropped and counted as taken.
    fn host_gpu_send(ptr: *const u8, len: usize) -> usize;
    /// Copies whole completion records into `ptr..ptr+capacity` and returns how many bytes; or,
    /// when the next record alone needs more than `capacity`, minus what it needs.
    fn host_gpu_receive(ptr: *mut u8, capacity: usize) -> isize;
}

/// The link and its capsets from the page's WebGL2 limits, for `MachineConfig::gpu`. `renderer`
/// is the capset's host name (`capset_v2`).
pub(crate) fn link(words: &[u32; LIMIT_WORDS], renderer: &str) -> GpuLink {
    let limits = HostLimits {
        max_texture_size: words[0],
        max_3d_texture_size: words[1],
        max_cube_map_size: words[2],
        max_array_layers: words[3],
        max_draw_buffers: words[4],
        max_samples: words[5],
        max_vertex_attribs: words[6],
        max_vertex_output_components: words[7],
        max_uniform_blocks: words[8],
        max_uniform_block_size: words[9],
        uniform_buffer_offset_alignment: words[10],
        max_texture_units: words[11],
        texel_offsets: (words[12] as i32, words[13] as i32),
    };
    let capsets = vec![
        GpuCapset {
            id: GpuCapset::VIRGL,
            max_version: caps::VIRGL_VERSION,
            data: caps::capset_v1(&limits),
        },
        GpuCapset {
            id: GpuCapset::VIRGL2,
            max_version: caps::VIRGL2_VERSION,
            data: caps::capset_v2(&limits, renderer),
        },
    ];
    GpuLink::new(
        capsets,
        GpuLimits {
            max_texture_size: limits.max_texture_size,
            max_texture_3d_size: limits.max_3d_texture_size,
            max_array_layers: limits.max_array_layers,
            max_buffer_bytes: MAX_BUFFER_BYTES,
            max_resource_bytes: MAX_RESOURCE_BYTES,
        },
    )
}

/// What the poll hook keeps between calls to move records between the link and the rings.
pub(crate) struct GpuPump {
    link: GpuLink,
    /// Encoded records the ring had no room for yet, from `sent` on.
    encoded: Vec<u8>,
    sent: usize,
    completions: Vec<u8>,
}

impl GpuPump {
    pub(crate) fn new(link: GpuLink) -> GpuPump {
        GpuPump {
            link,
            encoded: Vec::new(),
            sent: 0,
            completions: vec![0; COMPLETION_BUFFER_BYTES],
        }
    }

    pub(crate) fn pump(&mut self) {
        self.receive();
        if self.sent == self.encoded.len() {
            self.encoded.clear();
            self.sent = 0;
            for command in self.link.take_commands() {
                encode(&mut self.encoded, &command);
            }
        }
        while self.sent < self.encoded.len() {
            let rest = &self.encoded[self.sent..];
            let taken = unsafe { host_gpu_send(rest.as_ptr(), rest.len()) };
            if taken == 0 {
                break;
            }
            self.sent += taken.min(rest.len());
        }
    }

    fn receive(&mut self) {
        loop {
            let got =
                unsafe { host_gpu_receive(self.completions.as_mut_ptr(), self.completions.len()) };
            if got < 0 {
                self.completions.resize(got.unsigned_abs(), 0);
                continue;
            }
            if got == 0 {
                return;
            }
            let mut records = &self.completions[..got as usize];
            while records.len() >= RECORD_HEADER_BYTES {
                let kind = word(records, 0);
                let length = word(records, 1) as usize;
                let Some(payload) = records.get(RECORD_HEADER_BYTES..RECORD_HEADER_BYTES + length)
                else {
                    return;
                };
                match kind {
                    COMPLETION_FENCE if length >= 8 => self.link.complete_fence(long(payload, 0)),
                    COMPLETION_READ if length >= 8 => {
                        self.link
                            .complete_read(long(payload, 0), payload[8..].to_vec());
                    }
                    _ => {}
                }
                records = &records[(RECORD_HEADER_BYTES + length.next_multiple_of(WORD_BYTES))
                    .min(records.len())..];
            }
        }
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = index * WORD_BYTES;
    u32::from_le_bytes(bytes[at..at + WORD_BYTES].try_into().unwrap())
}

fn long(bytes: &[u8], index: usize) -> u64 {
    u64::from(word(bytes, index)) | u64::from(word(bytes, index + 1)) << 32
}

fn encode(out: &mut Vec<u8>, command: &GpuCommand) {
    let mut words: Vec<u32> = Vec::new();
    let mut tail: &[u8] = &[];
    let kind = match command {
        GpuCommand::Reset => RECORD_RESET,
        GpuCommand::CreateContext { context, name } => {
            words.push(*context);
            tail = name;
            RECORD_CREATE_CONTEXT
        }
        GpuCommand::DestroyContext { context } => {
            words.push(*context);
            RECORD_DESTROY_CONTEXT
        }
        GpuCommand::AttachResource { context, resource } => {
            words.extend([*context, *resource]);
            RECORD_ATTACH_RESOURCE
        }
        GpuCommand::DetachResource { context, resource } => {
            words.extend([*context, *resource]);
            RECORD_DETACH_RESOURCE
        }
        GpuCommand::CreateResource(resource) => {
            let GpuResource {
                id,
                target,
                format,
                bind,
                width,
                height,
                depth,
                array_size,
                last_level,
                samples,
                flags,
            } = *resource;
            words.extend([
                id, target, format, bind, width, height, depth, array_size, last_level, samples,
                flags,
            ]);
            RECORD_CREATE_RESOURCE
        }
        GpuCommand::DestroyResource { resource } => {
            words.push(*resource);
            RECORD_DESTROY_RESOURCE
        }
        GpuCommand::WriteResource {
            context,
            resource,
            level,
            region,
            data,
        } => {
            words.extend([*context, *resource, *level]);
            push_box(&mut words, region);
            tail = data;
            RECORD_WRITE_RESOURCE
        }
        GpuCommand::ReadResource {
            context,
            resource,
            level,
            region,
            token,
        } => {
            words.extend([*context, *resource, *level]);
            push_box(&mut words, region);
            words.extend([*token as u32, (*token >> 32) as u32]);
            RECORD_READ_RESOURCE
        }
        GpuCommand::Submit { context, commands } => {
            words.push(*context);
            tail = commands;
            RECORD_SUBMIT
        }
        GpuCommand::SetScanout {
            scanout,
            resource,
            rect,
        } => {
            words.extend([*scanout, *resource]);
            push_rect(&mut words, rect);
            RECORD_SET_SCANOUT
        }
        GpuCommand::Flush { resource, rect } => {
            words.push(*resource);
            push_rect(&mut words, rect);
            RECORD_FLUSH
        }
        GpuCommand::Fence { sequence } => {
            words.extend([*sequence as u32, (*sequence >> 32) as u32]);
            RECORD_FENCE
        }
        // A command a newer library adds; the renderer would not know it either.
        _ => return,
    };
    let length = words.len() * WORD_BYTES + tail.len();
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&(length as u32).to_le_bytes());
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(tail);
    out.resize(out.len().next_multiple_of(WORD_BYTES), 0);
}

fn push_box(words: &mut Vec<u32>, region: &GpuBox) {
    words.extend([
        region.x,
        region.y,
        region.z,
        region.width,
        region.height,
        region.depth,
    ]);
}

fn push_rect(words: &mut Vec<u32>, rect: &Rect) {
    words.extend([rect.x, rect.y, rect.width, rect.height]);
}
