//! The bytes of `struct virgl_caps_v1` and `struct virgl_caps_v2` (virglrenderer's
//! `src/virgl_hw.h`, which Mesa copies), the answers to `GET_CAPSET` for capsets 1 and 2. Mesa
//! asks for capset 2 when the kernel has `VIRTGPU_PARAM_CAPSET_QUERY_FIX` and falls back to 1;
//! the kernel copies at most what the host says the capset holds and Mesa keeps its own defaults
//! (`virgl_ws_fill_new_caps_defaults`) for the rest, so a short capset is not an error, only a
//! less capable host.
//!
//! What is advertised is what a GLES 3.0 host (WebGL2, or ANGLE's GLES 3.0) can honestly do, the
//! way virglrenderer fills it on a GLES host (`vrend_renderer_fill_caps`): GLSL 1.30, which Mesa
//! turns into about desktop GL 3.0 and GLES 3.0, and `HOST_IS_GLES`, which makes Mesa avoid what
//! GLES lacks. Every offset below was measured from the header with `offsetof`; the tests pin
//! them, and the struct is append-only, so they never move.

use alloc::vec::Vec;

/// `max_version` of each capset, both in `GET_CAPSET_INFO` and as the struct's first field.
pub const VIRGL_VERSION: u32 = 1;
pub const VIRGL2_VERSION: u32 = 2;
pub const VIRGL_BYTES: usize = 308;
pub const VIRGL2_BYTES: usize = 1408;

/// GLSL ES 3.00 is what virglrenderer reports as 130 on a GLES 3.0 host
/// (`vrend_fill_caps_glsl_version`). Below 130 Mesa lowers integers away; 140 would claim
/// GL 3.1's texture buffers, which WebGL2 does not have.
const GLSL_LEVEL: u32 = 130;

/// One 512-bit mask per kind of use, indexed by `virgl_formats`.
const FORMAT_MASK_BYTES: usize = 64;

// virgl_caps_v1.
const MAX_VERSION: usize = 0;
const SAMPLER_FORMATS: usize = 4;
const RENDER_FORMATS: usize = 68;
const DEPTH_STENCIL_FORMATS: usize = 132;
const VERTEX_BUFFER_FORMATS: usize = 196;
const BOOL_SET1: usize = 260;
const GLSL_LEVEL_AT: usize = 264;
const MAX_TEXTURE_ARRAY_LAYERS: usize = 268;
const MAX_STREAMOUT_BUFFERS: usize = 272;
const MAX_RENDER_TARGETS: usize = 280;
const MAX_SAMPLES: usize = 284;
const PRIMITIVE_MASK: usize = 288;
const MAX_UNIFORM_BLOCKS: usize = 296;
const MAX_VIEWPORTS: usize = 300;
const MAX_TEXTURE_GATHER_COMPONENTS: usize = 304;

// virgl_caps_v2, after the v1 struct.
const MIN_ALIASED_POINT_SIZE: usize = 308;
const MAX_VERTEX_OUTPUTS: usize = 352;
const MAX_VERTEX_ATTRIBS: usize = 356;
const MIN_TEXEL_OFFSET: usize = 364;
const UNIFORM_BUFFER_OFFSET_ALIGNMENT: usize = 384;
const CAPABILITY_BITS: usize = 392;
const MAX_VERTEX_ATTRIB_STRIDE: usize = 428;
const MAX_TEXTURE_2D_SIZE: usize = 484;
const MAX_TEXTURE_3D_SIZE: usize = 488;
const MAX_TEXTURE_CUBE_SIZE: usize = 492;
// `host_feature_check_version` (556) stays 0. Mesa reads several fields only above some version:
// from 12 on, for example, constant buffer 0's size comes from `max_const_buffer_size`, which is
// zero here, so the guest's GL would allow no uniforms at all. Raise it only together with every
// field the versions up to it add (Mesa's `virgl_screen.c`).
const READBACK_FORMATS: usize = 560;
const SCANOUT_FORMATS: usize = 624;
const RENDERER_NAME: usize = 696;
const RENDERER_NAME_BYTES: usize = 64;
const MAX_TEXTURE_SAMPLERS: usize = 764;
const MAX_UNIFORM_BLOCK_SIZE: usize = 1372;

// `struct virgl_caps_bool_set1` bits, in declaration order.
const BSET_PRIMITIVE_RESTART: u32 = 1 << 6;
const BSET_BLEND_EQUATION_SEPARATE: u32 = 1 << 7;
const BSET_INSTANCE_ID: u32 = 1 << 8;
const BSET_INSTANCE_DIVISOR: u32 = 1 << 9;
const BSET_SEAMLESS_CUBE_MAP: u32 = 1 << 10;
const BSET_UNIFORM_BUFFERS: u32 = 1 << 18;

const CAP_HOST_IS_GLES: u32 = 1 << 19;

/// `PIPE_PRIM_*` points to triangle fans: GLES has no quads or polygons (`prim_mask` tells Mesa
/// which to lower).
const PRIMITIVES_POINTS_TO_TRIANGLE_FANS: u32 = (1 << 7) - 1;

// virgl_formats numbers (virgl_hw.h).
const B8G8R8A8_UNORM: u32 = 1;
const B8G8R8X8_UNORM: u32 = 2;
const B5G6R5_UNORM: u32 = 7;
const Z16_UNORM: u32 = 16;
const Z32_FLOAT: u32 = 18;
const Z24_UNORM_S8_UINT: u32 = 19;
const Z24X8_UNORM: u32 = 21;
const R32_FLOAT: u32 = 28;
const R32G32_FLOAT: u32 = 29;
const R32G32B32_FLOAT: u32 = 30;
const R32G32B32A32_FLOAT: u32 = 31;
const R8_UNORM: u32 = 64;
const R8G8_UNORM: u32 = 65;
const R8G8B8A8_UNORM: u32 = 67;
const R16_FLOAT: u32 = 91;
const R16G16_FLOAT: u32 = 92;
const R16G16B16A16_FLOAT: u32 = 94;
const R8G8B8A8_SRGB: u32 = 104;
const R8G8B8X8_UNORM: u32 = 134;

/// Sampled and rendered to. Only the formats WebGL2 has a colour-renderable internal format for
/// (the BGR orders are swizzled by the renderer, as virglrenderer does on GLES).
const COLOR_FORMATS: &[u32] = &[
    B8G8R8A8_UNORM,
    B8G8R8X8_UNORM,
    B5G6R5_UNORM,
    R8_UNORM,
    R8G8_UNORM,
    R8G8B8A8_UNORM,
    R8G8B8X8_UNORM,
    R8G8B8A8_SRGB,
];
const DEPTH_FORMATS: &[u32] = &[Z16_UNORM, Z32_FLOAT, Z24_UNORM_S8_UINT, Z24X8_UNORM];
/// Sampled but not rendered to without `EXT_color_buffer_float`.
const FLOAT_FORMATS: &[u32] = &[
    R16_FLOAT,
    R16G16_FLOAT,
    R16G16B16A16_FLOAT,
    R32_FLOAT,
    R32G32_FLOAT,
    R32G32B32A32_FLOAT,
];
const VERTEX_FORMATS: &[u32] = &[
    R32_FLOAT,
    R32G32_FLOAT,
    R32G32B32_FLOAT,
    R32G32B32A32_FLOAT,
    R8G8B8A8_UNORM,
];
/// What the scanout (a page canvas, a viewer frame) can show.
const SCANOUT: &[u32] = &[
    B8G8R8A8_UNORM,
    B8G8R8X8_UNORM,
    R8G8B8A8_UNORM,
    R8G8B8X8_UNORM,
];

/// What the host's GL reports, for the capsets. On the web, the WebGL2 context's `getParameter`
/// answers; `Default` is the minimum GLES 3.0 guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostLimits {
    /// `MAX_TEXTURE_SIZE`.
    pub max_texture_size: u32,
    /// `MAX_3D_TEXTURE_SIZE`.
    pub max_3d_texture_size: u32,
    /// `MAX_CUBE_MAP_TEXTURE_SIZE`.
    pub max_cube_map_size: u32,
    /// `MAX_ARRAY_TEXTURE_LAYERS`.
    pub max_array_layers: u32,
    /// `MAX_DRAW_BUFFERS`.
    pub max_draw_buffers: u32,
    /// `MAX_SAMPLES`.
    pub max_samples: u32,
    /// `MAX_VERTEX_ATTRIBS`.
    pub max_vertex_attribs: u32,
    /// `MAX_VERTEX_OUTPUT_COMPONENTS`, in components (four per vector).
    pub max_vertex_output_components: u32,
    /// `MAX_FRAGMENT_UNIFORM_BLOCKS` (per stage, as the guest counts them).
    pub max_uniform_blocks: u32,
    /// `MAX_UNIFORM_BLOCK_SIZE`, in bytes.
    pub max_uniform_block_size: u32,
    /// `UNIFORM_BUFFER_OFFSET_ALIGNMENT`, in bytes.
    pub uniform_buffer_offset_alignment: u32,
    /// `MAX_TEXTURE_IMAGE_UNITS`.
    pub max_texture_units: u32,
    /// `MIN_PROGRAM_TEXEL_OFFSET` and `MAX_PROGRAM_TEXEL_OFFSET`.
    pub texel_offsets: (i32, i32),
}

impl Default for HostLimits {
    /// The GLES 3.0 minimums (the spec's table 6.3x "Implementation Dependent Values").
    fn default() -> HostLimits {
        HostLimits {
            max_texture_size: 2048,
            max_3d_texture_size: 256,
            max_cube_map_size: 2048,
            max_array_layers: 256,
            max_draw_buffers: 4,
            max_samples: 4,
            max_vertex_attribs: 16,
            max_vertex_output_components: 64,
            max_uniform_blocks: 12,
            max_uniform_block_size: 16384,
            uniform_buffer_offset_alignment: 256,
            max_texture_units: 16,
            texel_offsets: (-8, 7),
        }
    }
}

/// Capset 1 (`VIRGL`): `struct virgl_caps_v1`.
pub fn capset_v1(limits: &HostLimits) -> Vec<u8> {
    let mut caps = Caps(alloc::vec![0; VIRGL_BYTES]);
    caps.fill_v1(limits, VIRGL_VERSION);
    caps.0
}

/// Capset 2 (`VIRGL2`): `struct virgl_caps_v2`. `renderer` names the host GL (at most 63 bytes
/// are kept). Mesa puts it in the guest's `GL_RENDERER` ("virgl (renderer)") only from
/// `host_feature_check_version` 5 on, which this capset does not claim, so the guest sees plain
/// "virgl" for now.
pub fn capset_v2(limits: &HostLimits, renderer: &str) -> Vec<u8> {
    let mut caps = Caps(alloc::vec![0; VIRGL2_BYTES]);
    caps.fill_v1(limits, VIRGL2_VERSION);
    // Mesa's defaults where nothing better is known; a zeroed capset would make them all 0.
    let point_and_line = [1.0f32, 255.0, 1.0, 190.0, 1.0, 10.0, 0.0, 10.0, 15.0];
    for (index, value) in point_and_line.iter().enumerate() {
        caps.float(MIN_ALIASED_POINT_SIZE + index * 4, *value);
    }
    caps.word(MAX_VERTEX_OUTPUTS, limits.max_vertex_output_components / 4);
    caps.word(MAX_VERTEX_ATTRIBS, limits.max_vertex_attribs);
    let (min_offset, max_offset) = limits.texel_offsets;
    caps.word(MIN_TEXEL_OFFSET, min_offset as u32);
    caps.word(MIN_TEXEL_OFFSET + 4, max_offset as u32);
    caps.word(
        UNIFORM_BUFFER_OFFSET_ALIGNMENT,
        limits.uniform_buffer_offset_alignment,
    );
    caps.word(CAPABILITY_BITS, CAP_HOST_IS_GLES);
    // GLES 3.0 has `MAX_VERTEX_ATTRIB_STRIDE` only from 3.1; 2048 is its minimum there and what
    // WebGL2 implementations accept.
    caps.word(MAX_VERTEX_ATTRIB_STRIDE, 2048);
    caps.word(MAX_TEXTURE_2D_SIZE, limits.max_texture_size);
    caps.word(MAX_TEXTURE_3D_SIZE, limits.max_3d_texture_size);
    caps.word(MAX_TEXTURE_CUBE_SIZE, limits.max_cube_map_size);
    caps.formats(READBACK_FORMATS, COLOR_FORMATS);
    caps.formats(SCANOUT_FORMATS, SCANOUT);
    let name = renderer.as_bytes();
    let kept = name.len().min(RENDERER_NAME_BYTES - 1);
    caps.0[RENDERER_NAME..RENDERER_NAME + kept].copy_from_slice(&name[..kept]);
    caps.word(MAX_TEXTURE_SAMPLERS, limits.max_texture_units);
    caps.word(MAX_UNIFORM_BLOCK_SIZE, limits.max_uniform_block_size);
    caps.0
}

struct Caps(Vec<u8>);

impl Caps {
    fn fill_v1(&mut self, limits: &HostLimits, version: u32) {
        self.word(MAX_VERSION, version);
        self.formats(SAMPLER_FORMATS, COLOR_FORMATS);
        self.formats(SAMPLER_FORMATS, FLOAT_FORMATS);
        self.formats(SAMPLER_FORMATS, DEPTH_FORMATS);
        self.formats(RENDER_FORMATS, COLOR_FORMATS);
        self.formats(DEPTH_STENCIL_FORMATS, DEPTH_FORMATS);
        self.formats(VERTEX_BUFFER_FORMATS, VERTEX_FORMATS);
        self.word(
            BOOL_SET1,
            BSET_PRIMITIVE_RESTART
                | BSET_BLEND_EQUATION_SEPARATE
                | BSET_INSTANCE_ID
                | BSET_INSTANCE_DIVISOR
                | BSET_SEAMLESS_CUBE_MAP
                | BSET_UNIFORM_BUFFERS,
        );
        self.word(GLSL_LEVEL_AT, GLSL_LEVEL);
        self.word(MAX_TEXTURE_ARRAY_LAYERS, limits.max_array_layers);
        // Transform feedback: GLES 3.0 guarantees four separate buffers.
        self.word(MAX_STREAMOUT_BUFFERS, 4);
        self.word(MAX_RENDER_TARGETS, limits.max_draw_buffers);
        self.word(MAX_SAMPLES, limits.max_samples);
        self.word(PRIMITIVE_MASK, PRIMITIVES_POINTS_TO_TRIANGLE_FANS);
        self.word(MAX_UNIFORM_BLOCKS, limits.max_uniform_blocks);
        self.word(MAX_VIEWPORTS, 1);
        // `textureGather` is GLES 3.1.
        self.word(MAX_TEXTURE_GATHER_COMPONENTS, 0);
    }

    fn word(&mut self, at: usize, value: u32) {
        self.0[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn float(&mut self, at: usize, value: f32) {
        self.word(at, value.to_bits());
    }

    fn formats(&mut self, mask: usize, formats: &[u32]) {
        for &format in formats {
            let bit = format as usize;
            debug_assert!(bit < FORMAT_MASK_BYTES * 8);
            self.0[mask + bit / 8] |= 1 << (bit % 8);
        }
    }
}

#[cfg(test)]
mod tests;
