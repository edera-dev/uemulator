//! How the guest lays out each virgl format in its backing: bytes per block and the block's size
//! in pixels (1x1 for every uncompressed format). The device needs nothing else of a format: it
//! sizes resources against the 3D budget and finds the rows of a 3D transfer with it, and leaves
//! what the format means to the renderer.
//!
//! The numbers are `enum virgl_formats` (virglrenderer's `src/virgl_hw.h`), which keeps Gallium's
//! old `PIPE_FORMAT_*` numbering, so the eight 2D formats of `enum virtio_gpu_formats` are the
//! same ids. Left out: video formats with planes or subsampling (UYVY, NV12, P010 and the like),
//! `R1_UNORM`, the signed/unsigned mixtures Gallium dropped, and 262, 264 and 266, which the
//! header assigns to two formats of different sizes each. A resource in a format missing here is
//! refused at creation, so a renderer must not advertise one in its capset.

/// One block of a format: `bytes` in memory, `width` x `height` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub bytes: u32,
    pub width: u32,
    pub height: u32,
}

impl Layout {
    const fn new(bytes: u32, width: u32, height: u32) -> Layout {
        Layout {
            bytes,
            width,
            height,
        }
    }

    /// Bytes of one row of blocks `pixels` wide.
    pub fn row_bytes(&self, pixels: u32) -> u64 {
        pixels.div_ceil(self.width) as u64 * self.bytes as u64
    }

    /// Rows of blocks in `pixels` of height.
    pub fn rows(&self, pixels: u32) -> u64 {
        pixels.div_ceil(self.height) as u64
    }
}

pub(super) fn layout(format: u32) -> Option<Layout> {
    FORMATS
        .binary_search_by_key(&format, |(id, _)| *id)
        .ok()
        .map(|index| FORMATS[index].1)
}

/// Sorted by id, for `binary_search`.
const FORMATS: &[(u32, Layout)] = &[
    (1, Layout::new(4, 1, 1)),      // B8G8R8A8_UNORM
    (2, Layout::new(4, 1, 1)),      // B8G8R8X8_UNORM
    (3, Layout::new(4, 1, 1)),      // A8R8G8B8_UNORM
    (4, Layout::new(4, 1, 1)),      // X8R8G8B8_UNORM
    (5, Layout::new(2, 1, 1)),      // B5G5R5A1_UNORM
    (6, Layout::new(2, 1, 1)),      // B4G4R4A4_UNORM
    (7, Layout::new(2, 1, 1)),      // B5G6R5_UNORM
    (8, Layout::new(4, 1, 1)),      // R10G10B10A2_UNORM
    (9, Layout::new(1, 1, 1)),      // L8_UNORM
    (10, Layout::new(1, 1, 1)),     // A8_UNORM
    (11, Layout::new(1, 1, 1)),     // I8_UNORM
    (12, Layout::new(2, 1, 1)),     // L8A8_UNORM
    (13, Layout::new(2, 1, 1)),     // L16_UNORM
    (16, Layout::new(2, 1, 1)),     // Z16_UNORM
    (17, Layout::new(4, 1, 1)),     // Z32_UNORM
    (18, Layout::new(4, 1, 1)),     // Z32_FLOAT
    (19, Layout::new(4, 1, 1)),     // Z24_UNORM_S8_UINT
    (20, Layout::new(4, 1, 1)),     // S8_UINT_Z24_UNORM
    (21, Layout::new(4, 1, 1)),     // Z24X8_UNORM
    (22, Layout::new(4, 1, 1)),     // X8Z24_UNORM
    (23, Layout::new(1, 1, 1)),     // S8_UINT
    (24, Layout::new(8, 1, 1)),     // R64_FLOAT
    (25, Layout::new(16, 1, 1)),    // R64G64_FLOAT
    (26, Layout::new(24, 1, 1)),    // R64G64B64_FLOAT
    (27, Layout::new(32, 1, 1)),    // R64G64B64A64_FLOAT
    (28, Layout::new(4, 1, 1)),     // R32_FLOAT
    (29, Layout::new(8, 1, 1)),     // R32G32_FLOAT
    (30, Layout::new(12, 1, 1)),    // R32G32B32_FLOAT
    (31, Layout::new(16, 1, 1)),    // R32G32B32A32_FLOAT
    (32, Layout::new(4, 1, 1)),     // R32_UNORM
    (33, Layout::new(8, 1, 1)),     // R32G32_UNORM
    (34, Layout::new(12, 1, 1)),    // R32G32B32_UNORM
    (35, Layout::new(16, 1, 1)),    // R32G32B32A32_UNORM
    (36, Layout::new(4, 1, 1)),     // R32_USCALED
    (37, Layout::new(8, 1, 1)),     // R32G32_USCALED
    (38, Layout::new(12, 1, 1)),    // R32G32B32_USCALED
    (39, Layout::new(16, 1, 1)),    // R32G32B32A32_USCALED
    (40, Layout::new(4, 1, 1)),     // R32_SNORM
    (41, Layout::new(8, 1, 1)),     // R32G32_SNORM
    (42, Layout::new(12, 1, 1)),    // R32G32B32_SNORM
    (43, Layout::new(16, 1, 1)),    // R32G32B32A32_SNORM
    (44, Layout::new(4, 1, 1)),     // R32_SSCALED
    (45, Layout::new(8, 1, 1)),     // R32G32_SSCALED
    (46, Layout::new(12, 1, 1)),    // R32G32B32_SSCALED
    (47, Layout::new(16, 1, 1)),    // R32G32B32A32_SSCALED
    (48, Layout::new(2, 1, 1)),     // R16_UNORM
    (49, Layout::new(4, 1, 1)),     // R16G16_UNORM
    (50, Layout::new(6, 1, 1)),     // R16G16B16_UNORM
    (51, Layout::new(8, 1, 1)),     // R16G16B16A16_UNORM
    (52, Layout::new(2, 1, 1)),     // R16_USCALED
    (53, Layout::new(4, 1, 1)),     // R16G16_USCALED
    (54, Layout::new(6, 1, 1)),     // R16G16B16_USCALED
    (55, Layout::new(8, 1, 1)),     // R16G16B16A16_USCALED
    (56, Layout::new(2, 1, 1)),     // R16_SNORM
    (57, Layout::new(4, 1, 1)),     // R16G16_SNORM
    (58, Layout::new(6, 1, 1)),     // R16G16B16_SNORM
    (59, Layout::new(8, 1, 1)),     // R16G16B16A16_SNORM
    (60, Layout::new(2, 1, 1)),     // R16_SSCALED
    (61, Layout::new(4, 1, 1)),     // R16G16_SSCALED
    (62, Layout::new(6, 1, 1)),     // R16G16B16_SSCALED
    (63, Layout::new(8, 1, 1)),     // R16G16B16A16_SSCALED
    (64, Layout::new(1, 1, 1)),     // R8_UNORM
    (65, Layout::new(2, 1, 1)),     // R8G8_UNORM
    (66, Layout::new(3, 1, 1)),     // R8G8B8_UNORM
    (67, Layout::new(4, 1, 1)),     // R8G8B8A8_UNORM
    (68, Layout::new(4, 1, 1)),     // X8B8G8R8_UNORM
    (69, Layout::new(1, 1, 1)),     // R8_USCALED
    (70, Layout::new(2, 1, 1)),     // R8G8_USCALED
    (71, Layout::new(3, 1, 1)),     // R8G8B8_USCALED
    (72, Layout::new(4, 1, 1)),     // R8G8B8A8_USCALED
    (74, Layout::new(1, 1, 1)),     // R8_SNORM
    (75, Layout::new(2, 1, 1)),     // R8G8_SNORM
    (76, Layout::new(3, 1, 1)),     // R8G8B8_SNORM
    (77, Layout::new(4, 1, 1)),     // R8G8B8A8_SNORM
    (82, Layout::new(1, 1, 1)),     // R8_SSCALED
    (83, Layout::new(2, 1, 1)),     // R8G8_SSCALED
    (84, Layout::new(3, 1, 1)),     // R8G8B8_SSCALED
    (85, Layout::new(4, 1, 1)),     // R8G8B8A8_SSCALED
    (87, Layout::new(4, 1, 1)),     // R32_FIXED
    (88, Layout::new(8, 1, 1)),     // R32G32_FIXED
    (89, Layout::new(12, 1, 1)),    // R32G32B32_FIXED
    (90, Layout::new(16, 1, 1)),    // R32G32B32A32_FIXED
    (91, Layout::new(2, 1, 1)),     // R16_FLOAT
    (92, Layout::new(4, 1, 1)),     // R16G16_FLOAT
    (93, Layout::new(6, 1, 1)),     // R16G16B16_FLOAT
    (94, Layout::new(8, 1, 1)),     // R16G16B16A16_FLOAT
    (95, Layout::new(1, 1, 1)),     // L8_SRGB
    (96, Layout::new(2, 1, 1)),     // L8A8_SRGB
    (97, Layout::new(3, 1, 1)),     // R8G8B8_SRGB
    (98, Layout::new(4, 1, 1)),     // A8B8G8R8_SRGB
    (99, Layout::new(4, 1, 1)),     // X8B8G8R8_SRGB
    (100, Layout::new(4, 1, 1)),    // B8G8R8A8_SRGB
    (101, Layout::new(4, 1, 1)),    // B8G8R8X8_SRGB
    (102, Layout::new(4, 1, 1)),    // A8R8G8B8_SRGB
    (103, Layout::new(4, 1, 1)),    // X8R8G8B8_SRGB
    (104, Layout::new(4, 1, 1)),    // R8G8B8A8_SRGB
    (105, Layout::new(8, 4, 4)),    // DXT1_RGB
    (106, Layout::new(8, 4, 4)),    // DXT1_RGBA
    (107, Layout::new(16, 4, 4)),   // DXT3_RGBA
    (108, Layout::new(16, 4, 4)),   // DXT5_RGBA
    (109, Layout::new(8, 4, 4)),    // DXT1_SRGB
    (110, Layout::new(8, 4, 4)),    // DXT1_SRGBA
    (111, Layout::new(16, 4, 4)),   // DXT3_SRGBA
    (112, Layout::new(16, 4, 4)),   // DXT5_SRGBA
    (113, Layout::new(8, 4, 4)),    // RGTC1_UNORM
    (114, Layout::new(8, 4, 4)),    // RGTC1_SNORM
    (115, Layout::new(16, 4, 4)),   // RGTC2_UNORM
    (116, Layout::new(16, 4, 4)),   // RGTC2_SNORM
    (121, Layout::new(4, 1, 1)),    // A8B8G8R8_UNORM
    (122, Layout::new(2, 1, 1)),    // B5G5R5X1_UNORM
    (123, Layout::new(4, 1, 1)),    // R10G10B10A2_USCALED
    (124, Layout::new(4, 1, 1)),    // R11G11B10_FLOAT
    (125, Layout::new(4, 1, 1)),    // R9G9B9E5_FLOAT
    (126, Layout::new(8, 1, 1)),    // Z32_FLOAT_S8X24_UINT
    (128, Layout::new(4, 1, 1)),    // R10G10B10X2_USCALED
    (129, Layout::new(4, 1, 1)),    // R10G10B10X2_SNORM
    (130, Layout::new(1, 1, 1)),    // L4A4_UNORM
    (131, Layout::new(4, 1, 1)),    // B10G10R10A2_UNORM
    (134, Layout::new(4, 1, 1)),    // R8G8B8X8_UNORM
    (135, Layout::new(2, 1, 1)),    // B4G4R4X4_UNORM
    (136, Layout::new(4, 1, 1)),    // X24S8_UINT
    (137, Layout::new(4, 1, 1)),    // S8X24_UINT
    (138, Layout::new(8, 1, 1)),    // X32_S8X24_UINT
    (139, Layout::new(1, 1, 1)),    // B2G3R3_UNORM
    (140, Layout::new(4, 1, 1)),    // L16A16_UNORM
    (141, Layout::new(2, 1, 1)),    // A16_UNORM
    (142, Layout::new(2, 1, 1)),    // I16_UNORM
    (143, Layout::new(8, 4, 4)),    // LATC1_UNORM
    (144, Layout::new(8, 4, 4)),    // LATC1_SNORM
    (145, Layout::new(16, 4, 4)),   // LATC2_UNORM
    (146, Layout::new(16, 4, 4)),   // LATC2_SNORM
    (147, Layout::new(1, 1, 1)),    // A8_SNORM
    (148, Layout::new(1, 1, 1)),    // L8_SNORM
    (149, Layout::new(2, 1, 1)),    // L8A8_SNORM
    (150, Layout::new(1, 1, 1)),    // I8_SNORM
    (151, Layout::new(2, 1, 1)),    // A16_SNORM
    (152, Layout::new(2, 1, 1)),    // L16_SNORM
    (153, Layout::new(4, 1, 1)),    // L16A16_SNORM
    (154, Layout::new(2, 1, 1)),    // I16_SNORM
    (155, Layout::new(2, 1, 1)),    // A16_FLOAT
    (156, Layout::new(2, 1, 1)),    // L16_FLOAT
    (157, Layout::new(4, 1, 1)),    // L16A16_FLOAT
    (158, Layout::new(2, 1, 1)),    // I16_FLOAT
    (159, Layout::new(4, 1, 1)),    // A32_FLOAT
    (160, Layout::new(4, 1, 1)),    // L32_FLOAT
    (161, Layout::new(8, 1, 1)),    // L32A32_FLOAT
    (162, Layout::new(4, 1, 1)),    // I32_FLOAT
    (168, Layout::new(1, 1, 1)),    // A4R4_UNORM
    (169, Layout::new(1, 1, 1)),    // R4A4_UNORM
    (170, Layout::new(2, 1, 1)),    // R8A8_UNORM
    (171, Layout::new(2, 1, 1)),    // A8R8_UNORM
    (172, Layout::new(4, 1, 1)),    // R10G10B10A2_SSCALED
    (173, Layout::new(4, 1, 1)),    // R10G10B10A2_SNORM
    (174, Layout::new(4, 1, 1)),    // B10G10R10A2_USCALED
    (175, Layout::new(4, 1, 1)),    // B10G10R10A2_SSCALED
    (176, Layout::new(4, 1, 1)),    // B10G10R10A2_SNORM
    (177, Layout::new(1, 1, 1)),    // R8_UINT
    (178, Layout::new(2, 1, 1)),    // R8G8_UINT
    (179, Layout::new(3, 1, 1)),    // R8G8B8_UINT
    (180, Layout::new(4, 1, 1)),    // R8G8B8A8_UINT
    (181, Layout::new(1, 1, 1)),    // R8_SINT
    (182, Layout::new(2, 1, 1)),    // R8G8_SINT
    (183, Layout::new(3, 1, 1)),    // R8G8B8_SINT
    (184, Layout::new(4, 1, 1)),    // R8G8B8A8_SINT
    (185, Layout::new(2, 1, 1)),    // R16_UINT
    (186, Layout::new(4, 1, 1)),    // R16G16_UINT
    (187, Layout::new(6, 1, 1)),    // R16G16B16_UINT
    (188, Layout::new(8, 1, 1)),    // R16G16B16A16_UINT
    (189, Layout::new(2, 1, 1)),    // R16_SINT
    (190, Layout::new(4, 1, 1)),    // R16G16_SINT
    (191, Layout::new(6, 1, 1)),    // R16G16B16_SINT
    (192, Layout::new(8, 1, 1)),    // R16G16B16A16_SINT
    (193, Layout::new(4, 1, 1)),    // R32_UINT
    (194, Layout::new(8, 1, 1)),    // R32G32_UINT
    (195, Layout::new(12, 1, 1)),   // R32G32B32_UINT
    (196, Layout::new(16, 1, 1)),   // R32G32B32A32_UINT
    (197, Layout::new(4, 1, 1)),    // R32_SINT
    (198, Layout::new(8, 1, 1)),    // R32G32_SINT
    (199, Layout::new(12, 1, 1)),   // R32G32B32_SINT
    (200, Layout::new(16, 1, 1)),   // R32G32B32A32_SINT
    (201, Layout::new(1, 1, 1)),    // A8_UINT
    (202, Layout::new(1, 1, 1)),    // I8_UINT
    (203, Layout::new(1, 1, 1)),    // L8_UINT
    (204, Layout::new(2, 1, 1)),    // L8A8_UINT
    (205, Layout::new(1, 1, 1)),    // A8_SINT
    (206, Layout::new(1, 1, 1)),    // I8_SINT
    (207, Layout::new(1, 1, 1)),    // L8_SINT
    (208, Layout::new(2, 1, 1)),    // L8A8_SINT
    (209, Layout::new(2, 1, 1)),    // A16_UINT
    (210, Layout::new(2, 1, 1)),    // I16_UINT
    (211, Layout::new(2, 1, 1)),    // L16_UINT
    (212, Layout::new(4, 1, 1)),    // L16A16_UINT
    (213, Layout::new(2, 1, 1)),    // A16_SINT
    (214, Layout::new(2, 1, 1)),    // I16_SINT
    (215, Layout::new(2, 1, 1)),    // L16_SINT
    (216, Layout::new(4, 1, 1)),    // L16A16_SINT
    (217, Layout::new(4, 1, 1)),    // A32_UINT
    (218, Layout::new(4, 1, 1)),    // I32_UINT
    (219, Layout::new(4, 1, 1)),    // L32_UINT
    (220, Layout::new(8, 1, 1)),    // L32A32_UINT
    (221, Layout::new(4, 1, 1)),    // A32_SINT
    (222, Layout::new(4, 1, 1)),    // I32_SINT
    (223, Layout::new(4, 1, 1)),    // L32_SINT
    (224, Layout::new(8, 1, 1)),    // L32A32_SINT
    (225, Layout::new(4, 1, 1)),    // B10G10R10A2_UINT
    (226, Layout::new(8, 4, 4)),    // ETC1_RGB8
    (229, Layout::new(4, 1, 1)),    // R8G8B8X8_SNORM
    (230, Layout::new(4, 1, 1)),    // R8G8B8X8_SRGB
    (231, Layout::new(4, 1, 1)),    // R8G8B8X8_UINT
    (232, Layout::new(4, 1, 1)),    // R8G8B8X8_SINT
    (233, Layout::new(4, 1, 1)),    // B10G10R10X2_UNORM
    (234, Layout::new(8, 1, 1)),    // R16G16B16X16_UNORM
    (235, Layout::new(8, 1, 1)),    // R16G16B16X16_SNORM
    (236, Layout::new(8, 1, 1)),    // R16G16B16X16_FLOAT
    (237, Layout::new(8, 1, 1)),    // R16G16B16X16_UINT
    (238, Layout::new(8, 1, 1)),    // R16G16B16X16_SINT
    (239, Layout::new(16, 1, 1)),   // R32G32B32X32_FLOAT
    (240, Layout::new(16, 1, 1)),   // R32G32B32X32_UINT
    (241, Layout::new(16, 1, 1)),   // R32G32B32X32_SINT
    (242, Layout::new(2, 1, 1)),    // R8A8_SNORM
    (243, Layout::new(4, 1, 1)),    // R16A16_UNORM
    (244, Layout::new(4, 1, 1)),    // R16A16_SNORM
    (245, Layout::new(4, 1, 1)),    // R16A16_FLOAT
    (246, Layout::new(8, 1, 1)),    // R32A32_FLOAT
    (247, Layout::new(2, 1, 1)),    // R8A8_UINT
    (248, Layout::new(2, 1, 1)),    // R8A8_SINT
    (249, Layout::new(4, 1, 1)),    // R16A16_UINT
    (250, Layout::new(4, 1, 1)),    // R16A16_SINT
    (251, Layout::new(8, 1, 1)),    // R32A32_UINT
    (252, Layout::new(8, 1, 1)),    // R32A32_SINT
    (253, Layout::new(4, 1, 1)),    // R10G10B10A2_UINT
    (254, Layout::new(2, 1, 1)),    // B5G6R5_SRGB
    (255, Layout::new(16, 4, 4)),   // BPTC_RGBA_UNORM
    (256, Layout::new(16, 4, 4)),   // BPTC_SRGBA
    (257, Layout::new(16, 4, 4)),   // BPTC_RGB_FLOAT
    (258, Layout::new(16, 4, 4)),   // BPTC_RGB_UFLOAT
    (259, Layout::new(2, 1, 1)),    // A8L8_UNORM
    (260, Layout::new(2, 1, 1)),    // A8L8_SNORM
    (261, Layout::new(2, 1, 1)),    // A8L8_SRGB
    (263, Layout::new(2, 1, 1)),    // G8R8_UNORM
    (265, Layout::new(4, 1, 1)),    // G16R16_UNORM
    (267, Layout::new(4, 1, 1)),    // A8B8G8R8_SNORM
    (268, Layout::new(4, 1, 1)),    // X8B8G8R8_SNORM
    (269, Layout::new(8, 4, 4)),    // ETC2_RGB8
    (270, Layout::new(8, 4, 4)),    // ETC2_SRGB8
    (271, Layout::new(8, 4, 4)),    // ETC2_RGB8A1
    (272, Layout::new(8, 4, 4)),    // ETC2_SRGB8A1
    (273, Layout::new(16, 4, 4)),   // ETC2_RGBA8
    (274, Layout::new(16, 4, 4)),   // ETC2_SRGBA8
    (275, Layout::new(8, 4, 4)),    // ETC2_R11_UNORM
    (276, Layout::new(8, 4, 4)),    // ETC2_R11_SNORM
    (277, Layout::new(16, 4, 4)),   // ETC2_RG11_UNORM
    (278, Layout::new(16, 4, 4)),   // ETC2_RG11_SNORM
    (279, Layout::new(16, 4, 4)),   // ASTC_4x4
    (280, Layout::new(16, 5, 4)),   // ASTC_5x4
    (281, Layout::new(16, 5, 5)),   // ASTC_5x5
    (282, Layout::new(16, 6, 5)),   // ASTC_6x5
    (283, Layout::new(16, 6, 6)),   // ASTC_6x6
    (284, Layout::new(16, 8, 5)),   // ASTC_8x5
    (285, Layout::new(16, 8, 6)),   // ASTC_8x6
    (286, Layout::new(16, 8, 8)),   // ASTC_8x8
    (287, Layout::new(16, 10, 5)),  // ASTC_10x5
    (288, Layout::new(16, 10, 6)),  // ASTC_10x6
    (289, Layout::new(16, 10, 8)),  // ASTC_10x8
    (290, Layout::new(16, 10, 10)), // ASTC_10x10
    (291, Layout::new(16, 12, 10)), // ASTC_12x10
    (292, Layout::new(16, 12, 12)), // ASTC_12x12
    (293, Layout::new(16, 4, 4)),   // ASTC_4x4_SRGB
    (294, Layout::new(16, 5, 4)),   // ASTC_5x4_SRGB
    (295, Layout::new(16, 5, 5)),   // ASTC_5x5_SRGB
    (296, Layout::new(16, 6, 5)),   // ASTC_6x5_SRGB
    (297, Layout::new(16, 6, 6)),   // ASTC_6x6_SRGB
    (298, Layout::new(16, 8, 5)),   // ASTC_8x5_SRGB
    (299, Layout::new(16, 8, 6)),   // ASTC_8x6_SRGB
    (300, Layout::new(16, 8, 8)),   // ASTC_8x8_SRGB
    (301, Layout::new(16, 10, 5)),  // ASTC_10x5_SRGB
    (302, Layout::new(16, 10, 6)),  // ASTC_10x6_SRGB
    (303, Layout::new(16, 10, 8)),  // ASTC_10x8_SRGB
    (304, Layout::new(16, 10, 10)), // ASTC_10x10_SRGB
    (305, Layout::new(16, 12, 10)), // ASTC_12x10_SRGB
    (306, Layout::new(16, 12, 12)), // ASTC_12x12_SRGB
    (308, Layout::new(4, 1, 1)),    // R10G10B10X2_UNORM
    (311, Layout::new(2, 1, 1)),    // A4B4G4R4_UNORM
    (312, Layout::new(1, 1, 1)),    // R8_SRGB
    (313, Layout::new(2, 1, 1)),    // R8G8_SRGB
    (317, Layout::new(3, 1, 1)),    // B8G8R8_UNORM
    (318, Layout::new(1, 1, 1)),    // R3G3B2_UNORM
    (319, Layout::new(2, 1, 1)),    // R4G4B4A4_UNORM
    (320, Layout::new(2, 1, 1)),    // R5G5B5A1_UNORM
    (321, Layout::new(2, 1, 1)),    // R5G6B5_UNORM
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_without_duplicates_so_the_search_finds_every_entry() {
        assert!(FORMATS.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for (id, entry) in FORMATS {
            assert_eq!(layout(*id), Some(*entry));
        }
    }

    #[test]
    fn the_2d_formats_are_four_byte_pixels() {
        for format in [1, 2, 3, 4, 67, 68, 121, 134] {
            assert_eq!(layout(format), Some(Layout::new(4, 1, 1)));
        }
    }

    #[test]
    fn compressed_rows_count_whole_blocks() {
        const DXT1_RGB: u32 = 105;
        const ASTC_10X5: u32 = 287;
        let dxt1 = layout(DXT1_RGB).unwrap();
        assert_eq!((dxt1.row_bytes(5), dxt1.rows(5)), (16, 2));
        let astc = layout(ASTC_10X5).unwrap();
        assert_eq!((astc.row_bytes(11), astc.rows(11)), (32, 3));
    }

    #[test]
    fn ambiguous_and_planar_ids_are_unknown() {
        const NV12: u32 = 166;
        const AMBIGUOUS: u32 = 262;
        assert_eq!(layout(NV12), None);
        assert_eq!(layout(AMBIGUOUS), None);
        assert_eq!(layout(0), None);
    }
}
