use super::*;

fn word(caps: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(caps[at..at + 4].try_into().unwrap())
}

fn has_format(caps: &[u8], mask: usize, format: u32) -> bool {
    // Mesa's `virgl_format_check_bitmask`: word format / 32, bit format % 32.
    word(caps, mask + format as usize / 32 * 4) & (1 << (format % 32)) != 0
}

#[test]
fn the_capsets_are_the_size_of_the_structs_and_carry_their_version() {
    let limits = HostLimits::default();
    let v1 = capset_v1(&limits);
    let v2 = capset_v2(&limits, "WebGL2");
    assert_eq!(v1.len(), VIRGL_BYTES);
    assert_eq!(v2.len(), VIRGL2_BYTES);
    assert_eq!(word(&v1, MAX_VERSION), 1);
    assert_eq!(word(&v2, MAX_VERSION), 2);
    // The v2 struct starts with the v1 struct; only the version differs.
    assert_eq!(v1[4..], v2[4..VIRGL_BYTES]);
}

#[test]
fn the_fields_mesa_needs_are_where_virgl_hw_h_puts_them() {
    let limits = HostLimits {
        max_texture_size: 16384,
        max_draw_buffers: 8,
        max_vertex_attribs: 16,
        max_vertex_output_components: 120,
        ..HostLimits::default()
    };
    let caps = capset_v2(&limits, "WebGL2");
    assert_eq!(word(&caps, 264), 130, "glsl_level");
    assert_eq!(word(&caps, 280), 8, "max_render_targets");
    assert_eq!(word(&caps, 352), 30, "max_vertex_outputs");
    assert_eq!(word(&caps, 356), 16, "max_vertex_attribs");
    assert_eq!(
        word(&caps, 392),
        1 << 19,
        "capability_bits: HOST_IS_GLES only"
    );
    assert_eq!(word(&caps, 484), 16384, "max_texture_2d_size");
    assert_eq!(
        f32::from_bits(word(&caps, 312)),
        255.0,
        "max_aliased_point_size"
    );
    assert_eq!(word(&caps, 364) as i32, -8, "min_texel_offset");
    assert_eq!(word(&caps, 368), 7, "max_texel_offset");
    // primitive_restart is the seventh bit of virgl_caps_bool_set1, ubo the nineteenth.
    let bool_set = word(&caps, 260);
    assert_ne!(bool_set & (1 << 6), 0);
    assert_ne!(bool_set & (1 << 18), 0);
}

#[test]
fn the_scanout_formats_can_be_rendered_and_sampled() {
    let caps = capset_v2(&HostLimits::default(), "");
    // What a GBM surface for XRGB8888 and ARGB8888 is created with.
    for format in [B8G8R8X8_UNORM, B8G8R8A8_UNORM] {
        assert!(has_format(&caps, SCANOUT_FORMATS, format));
        assert!(has_format(&caps, RENDER_FORMATS, format));
        assert!(has_format(&caps, SAMPLER_FORMATS, format));
    }
    assert!(has_format(&caps, DEPTH_STENCIL_FORMATS, Z24_UNORM_S8_UINT));
    assert!(!has_format(&caps, RENDER_FORMATS, Z24_UNORM_S8_UINT));
    assert!(has_format(&caps, VERTEX_BUFFER_FORMATS, R32G32B32_FLOAT));
    assert!(has_format(&caps, SAMPLER_FORMATS, R8G8B8X8_UNORM));
}

#[test]
fn the_renderer_name_is_cut_to_leave_its_terminator() {
    let long = "x".repeat(100);
    let caps = capset_v2(&HostLimits::default(), &long);
    let name = &caps[RENDERER_NAME..RENDERER_NAME + RENDERER_NAME_BYTES];
    assert_eq!(name.iter().filter(|&&byte| byte == b'x').count(), 63);
    assert_eq!(name[63], 0);
    assert_eq!(word(&caps, MAX_TEXTURE_SAMPLERS), 16);
}
