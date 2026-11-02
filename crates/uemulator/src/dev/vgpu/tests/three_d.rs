//! The 3D side through the same `Rig`, with the test as the renderer: it reads what the device
//! queued on the `GpuLink` and completes fences and readbacks by hand.

use super::*;
use crate::dev::vgpu::virgl::{
    CMD_CTX_ATTACH_RESOURCE, CMD_CTX_CREATE, CMD_CTX_DESTROY, CMD_CTX_DETACH_RESOURCE,
    CMD_GET_CAPSET, CMD_GET_CAPSET_INFO, CMD_RESOURCE_CREATE_3D, CMD_SUBMIT_3D,
    CMD_TRANSFER_FROM_HOST_3D, CMD_TRANSFER_TO_HOST_3D,
};
use crate::gpu::{GpuBox, GpuCapset, GpuCommand, GpuLimits, GpuResource};

const CONTEXT: u32 = 5;
const TEXTURE: u32 = 40;
const VIRGL_FORMAT_R8G8B8A8_UNORM: u32 = 67;
const PIPE_TEXTURE_2D: u32 = 2;
/// A texture of the Rig's size, so `Rig::attach` backs it with the same two runs.
const TEXTURE_BYTES: u32 = WIDTH * HEIGHT * 4;
const TEXTURE_STRIDE: u32 = WIDTH * 4;
/// In-flight chains beside the Rig's own (descriptors 0 to 2): two descriptors each.
const SLOT_FIRST_DESCRIPTOR: u16 = 3;
const SLOT_REQUESTS: u64 = 0xb000;
const SLOT_REQUEST_BYTES: u64 = 0x800;
const SLOT_RESPONSES: u64 = 0xe000;
const SLOT_RESPONSE_BYTES: u64 = 0x400;

fn capsets() -> Vec<GpuCapset> {
    vec![
        GpuCapset {
            id: GpuCapset::VIRGL,
            max_version: 1,
            data: vec![1, 2, 3, 4],
        },
        GpuCapset {
            id: GpuCapset::VIRGL2,
            max_version: 2,
            data: vec![9; 12],
        },
    ]
}

fn with_renderer(limits: GpuLimits) -> (Rig, GpuLink) {
    let mut rig = Rig::new();
    let link = GpuLink::new(capsets(), limits);
    rig.gpu = VirtioGpu::new(rig.display.clone(), Some(link.clone()));
    assert_eq!(link.take_commands(), [GpuCommand::Reset]);
    (rig, link)
}

fn renderer() -> (Rig, GpuLink) {
    with_renderer(GpuLimits::default())
}

fn request(command: u32, context: u32) -> Vec<u8> {
    let mut bytes = header(command);
    bytes[HEADER_CONTEXT_ID..HEADER_CONTEXT_ID + 4].copy_from_slice(&context.to_le_bytes());
    bytes
}

fn fenced(mut request: Vec<u8>, fence_id: u64) -> Vec<u8> {
    request[HEADER_FLAGS..HEADER_FLAGS + 4].copy_from_slice(&FLAG_FENCE.to_le_bytes());
    request[HEADER_FENCE_ID..HEADER_FENCE_ID + 8].copy_from_slice(&fence_id.to_le_bytes());
    request
}

fn create_context_request(context: u32, name: &[u8]) -> Vec<u8> {
    let mut bytes = request(CMD_CTX_CREATE, context);
    bytes.extend(words(&[name.len() as u32, 0]));
    let mut padded = name.to_vec();
    padded.resize(64, 0);
    bytes.extend(padded);
    bytes
}

fn texture(id: u32) -> GpuResource {
    GpuResource {
        id,
        target: PIPE_TEXTURE_2D,
        format: VIRGL_FORMAT_R8G8B8A8_UNORM,
        bind: 2,
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        array_size: 1,
        last_level: 0,
        samples: 0,
        flags: 0,
    }
}

fn create_3d_request(resource: &GpuResource) -> Vec<u8> {
    let mut bytes = request(CMD_RESOURCE_CREATE_3D, 0);
    bytes.extend(words(&[
        resource.id,
        resource.target,
        resource.format,
        resource.bind,
        resource.width,
        resource.height,
        resource.depth,
        resource.array_size,
        resource.last_level,
        resource.samples,
        resource.flags,
        0,
    ]));
    bytes
}

/// `struct virtio_gpu_transfer_host_3d`.
fn transfer_3d_request(
    command: u32,
    resource: u32,
    region: GpuBox,
    offset: u64,
    stride: u32,
    layer_stride: u32,
) -> Vec<u8> {
    let mut bytes = request(command, CONTEXT);
    bytes.extend(words(&[
        region.x,
        region.y,
        region.z,
        region.width,
        region.height,
        region.depth,
    ]));
    bytes.extend(offset.to_le_bytes());
    bytes.extend(words(&[resource, 0, stride, layer_stride]));
    bytes
}

fn context_resource_request(command: u32, resource: u32) -> Vec<u8> {
    let mut bytes = request(command, CONTEXT);
    bytes.extend(words(&[resource, 0]));
    bytes
}

fn submit_request(stream: &[u8]) -> Vec<u8> {
    let mut bytes = request(CMD_SUBMIT_3D, CONTEXT);
    bytes.extend(words(&[stream.len() as u32, 0]));
    bytes.extend(stream);
    bytes
}

fn region(x: u32, y: u32, width: u32, height: u32) -> GpuBox {
    GpuBox {
        x,
        y,
        z: 0,
        width,
        height,
        depth: 1,
    }
}

impl Rig {
    /// Posts `request` on its own descriptors and buffers (`slot` 0 or 1) so it can stay in
    /// flight while others complete; returns what the kick returned.
    fn send(&mut self, slot: u16, request: &[u8]) -> bool {
        let first = SLOT_FIRST_DESCRIPTOR + 2 * slot;
        let at = SLOT_REQUESTS + slot as u64 * SLOT_REQUEST_BYTES;
        let response = SLOT_RESPONSES + slot as u64 * SLOT_RESPONSE_BYTES;
        assert!(request.len() as u64 <= SLOT_REQUEST_BYTES);
        self.ram[at as usize..at as usize + request.len()].copy_from_slice(request);
        self.ram[response as usize..(response + SLOT_RESPONSE_BYTES) as usize].fill(0xee);
        put_desc(
            &mut self.ram,
            CONTROL_DESC,
            first,
            at,
            request.len() as u32,
            DESC_F_NEXT,
            first + 1,
        );
        put_desc(
            &mut self.ram,
            CONTROL_DESC,
            first + 1,
            response,
            SLOT_RESPONSE_BYTES as u32,
            DESC_F_WRITE,
            0,
        );
        post(&mut self.ram, CONTROL_AVAIL, QUEUE_SIZE, first);
        self.kick(CONTROL_QUEUE)
    }

    /// The response header the device wrote for `slot`, if it has.
    fn slot_response(&self, slot: u16) -> Option<Vec<u8>> {
        let at = (SLOT_RESPONSES + slot as u64 * SLOT_RESPONSE_BYTES) as usize;
        let response = &self.ram[at..at + HEADER_BYTES];
        (response != [0xee; HEADER_BYTES]).then(|| response.to_vec())
    }

    /// The heads of the used ring, oldest first.
    fn used_heads(&self) -> Vec<u32> {
        (0..used_idx(&self.ram, CONTROL_USED))
            .map(|index| used_entry(&self.ram, CONTROL_USED, (index % QUEUE_SIZE) as usize).0)
            .collect()
    }

    fn poll(&mut self) -> bool {
        let mut mem = test_mem(&mut self.ram);
        self.gpu.poll(&mut self.queues, &mut mem)
    }

    /// A context with a backed, attached 2D texture of the Rig's size.
    fn backed_texture(&mut self, link: &GpuLink) {
        assert_eq!(
            self.status(&create_context_request(CONTEXT, b"t")),
            RESP_OK_NODATA
        );
        assert_eq!(
            self.status(&create_3d_request(&texture(TEXTURE))),
            RESP_OK_NODATA
        );
        assert_eq!(self.attach(TEXTURE, TEXTURE_BYTES), RESP_OK_NODATA);
        assert_eq!(
            self.status(&context_resource_request(CMD_CTX_ATTACH_RESOURCE, TEXTURE)),
            RESP_OK_NODATA
        );
        link.take_commands();
    }

    fn backing_byte(&self, index: u32) -> u8 {
        self.ram[self.backing_byte_address(index, TEXTURE_BYTES)]
    }
}

#[test]
fn a_renderer_brings_virgl_and_its_capsets() {
    let (mut rig, _link) = renderer();
    assert_eq!(rig.gpu.features(), FEATURE_VIRGL);
    assert_eq!(rig.gpu.config_read(CONFIG_NUM_CAPSETS), 2);
    for (index, (id, version, size)) in [(1, 1, 4), (2, 2, 12), (0, 0, 0)].into_iter().enumerate() {
        let mut ask = header(CMD_GET_CAPSET_INFO);
        ask.extend(words(&[index as u32, 0]));
        let response = rig.command(&ask);
        assert_eq!(response_type(&response), 0x1102);
        let info = &response[HEADER_BYTES..];
        assert_eq!(
            (u32_at(info, 0), u32_at(info, 4), u32_at(info, 8)),
            (id, version, size),
            "capset {index}; past the end is all zeros"
        );
    }
    let mut ask = header(CMD_GET_CAPSET);
    ask.extend(words(&[GpuCapset::VIRGL2, 2]));
    let response = rig.command(&ask);
    assert_eq!(response_type(&response), 0x1103);
    assert_eq!(&response[HEADER_BYTES..], &[9; 12]);
    for (id, version) in [(GpuCapset::VIRGL2, 3), (7, 1)] {
        let mut ask = header(CMD_GET_CAPSET);
        ask.extend(words(&[id, version]));
        assert_eq!(rig.status(&ask), RESP_ERR_INVALID_PARAMETER);
    }
}

#[test]
fn without_a_renderer_3d_commands_and_capsets_are_refused() {
    let mut rig = Rig::new();
    assert_eq!(rig.status(&header(CMD_GET_CAPSET_INFO)), RESP_ERR_UNSPEC);
    assert_eq!(
        rig.status(&create_context_request(1, b"x")),
        RESP_ERR_UNSPEC
    );
}

#[test]
fn every_new_or_reset_device_tells_the_renderer_to_start_over() {
    let (mut rig, link) = renderer();
    rig.gpu.reset();
    assert_eq!(link.take_commands(), [GpuCommand::Reset]);
    let _next_boot = VirtioGpu::new(rig.display.clone(), Some(link.clone()));
    assert_eq!(link.take_commands(), [GpuCommand::Reset]);
}

#[test]
fn contexts_are_created_named_and_destroyed_once() {
    let (mut rig, link) = renderer();
    let mut long = create_context_request(CONTEXT, b"glxgears");
    // A name length past the 64 bytes there are is cut to them.
    long[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&1000u32.to_le_bytes());
    assert_eq!(rig.status(&long), RESP_OK_NODATA);
    let [GpuCommand::CreateContext { context, name }] = &link.take_commands()[..] else {
        panic!("one CreateContext");
    };
    assert_eq!(
        (*context, name.len(), &name[..8]),
        (CONTEXT, 64, &b"glxgears"[..])
    );
    assert_eq!(
        rig.status(&create_context_request(CONTEXT, b"again")),
        0x1204,
        "an id in use"
    );
    assert_eq!(rig.status(&create_context_request(0, b"zero")), 0x1204);
    let mut with_init = create_context_request(6, b"venus");
    with_init[HEADER_BYTES + 4] = GpuCapset::VIRGL2 as u8;
    assert_eq!(
        rig.status(&with_init),
        RESP_ERR_INVALID_PARAMETER,
        "CONTEXT_INIT is not offered"
    );
    let destroy = request(CMD_CTX_DESTROY, CONTEXT);
    assert_eq!(rig.status(&destroy), RESP_OK_NODATA);
    assert_eq!(rig.status(&destroy), 0x1204);
    assert_eq!(
        link.take_commands(),
        [GpuCommand::DestroyContext { context: CONTEXT }]
    );
}

#[test]
fn a_3d_resource_is_checked_and_forwarded() {
    let (mut rig, link) = with_renderer(GpuLimits {
        max_texture_size: 64,
        max_resource_bytes: 64 * 64 * 4 * 2,
        ..GpuLimits::default()
    });
    let good = GpuResource {
        width: 64,
        height: 64,
        last_level: 6,
        ..texture(TEXTURE)
    };
    assert_eq!(rig.status(&create_3d_request(&good)), RESP_OK_NODATA);
    assert_eq!(link.take_commands(), [GpuCommand::CreateResource(good)]);
    let refused = [
        (texture(TEXTURE), RESP_ERR_INVALID_RESOURCE_ID),
        (texture(0), RESP_ERR_INVALID_RESOURCE_ID),
        (
            GpuResource {
                format: 166, // NV12
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                target: 9,
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                width: 65,
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                last_level: 5, // 16x8 has levels 0 to 4
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                target: 0,
                height: 2, // a buffer is one row
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                depth: 0,
                ..texture(1)
            },
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            GpuResource {
                width: 64,
                height: 64,
                samples: 2,
                ..texture(1)
            },
            RESP_ERR_OUT_OF_MEMORY,
        ),
    ];
    for (resource, status) in refused {
        assert_eq!(
            rig.status(&create_3d_request(&resource)),
            status,
            "{resource:?}"
        );
    }
    assert!(link.take_commands().is_empty());
    // One id space: a 2D resource cannot take a 3D one's id, nor the reverse.
    assert_eq!(
        rig.create(TEXTURE, FORMAT_B8G8R8X8, 4, 4),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    assert_eq!(rig.create(RESOURCE, FORMAT_B8G8R8X8, 4, 4), RESP_OK_NODATA);
    assert_eq!(
        rig.status(&create_3d_request(&texture(RESOURCE))),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    // Freeing gives the budget back.
    let mut unref = header(CMD_RESOURCE_UNREF);
    unref.extend(words(&[TEXTURE, 0]));
    assert_eq!(rig.status(&unref), RESP_OK_NODATA);
    assert_eq!(
        link.take_commands(),
        [GpuCommand::DestroyResource { resource: TEXTURE }]
    );
    let refill = GpuResource {
        width: 64,
        height: 64,
        ..texture(TEXTURE)
    };
    assert_eq!(rig.status(&create_3d_request(&refill)), RESP_OK_NODATA);
}

#[test]
fn attaching_forwards_3d_resources_and_accepts_2d_ones_quietly() {
    let (mut rig, link) = renderer();
    let attach = CMD_CTX_ATTACH_RESOURCE;
    let detach = CMD_CTX_DETACH_RESOURCE;
    assert_eq!(
        rig.status(&context_resource_request(attach, TEXTURE)),
        0x1204,
        "no such context yet"
    );
    rig.status(&create_context_request(CONTEXT, b"t"));
    rig.status(&create_3d_request(&texture(TEXTURE)));
    // The fbdev console's dumb buffer is attached to its context too.
    rig.create(RESOURCE, FORMAT_B8G8R8X8, WIDTH, HEIGHT);
    link.take_commands();
    for id in [TEXTURE, RESOURCE] {
        assert_eq!(
            rig.status(&context_resource_request(attach, id)),
            RESP_OK_NODATA
        );
    }
    assert_eq!(
        rig.status(&context_resource_request(attach, 77)),
        RESP_ERR_INVALID_RESOURCE_ID
    );
    for id in [TEXTURE, RESOURCE] {
        assert_eq!(
            rig.status(&context_resource_request(detach, id)),
            RESP_OK_NODATA
        );
    }
    assert_eq!(
        link.take_commands(),
        [
            GpuCommand::AttachResource {
                context: CONTEXT,
                resource: TEXTURE
            },
            GpuCommand::DetachResource {
                context: CONTEXT,
                resource: TEXTURE
            },
        ]
    );
}

#[test]
fn a_transfer_to_the_host_takes_the_box_out_of_the_backing_at_command_time() {
    let (mut rig, link) = renderer();
    rig.backed_texture(&link);
    // Rows 3 to 6, pixels 2 to 4: the box straddles the two backing runs (the split is after
    // row 3).
    for index in 0..TEXTURE_BYTES {
        let at = rig.backing_byte_address(index, TEXTURE_BYTES);
        rig.ram[at] = index as u8;
    }
    let (x, y, width, height) = (2, 3, 3, 4);
    let offset = (y * TEXTURE_STRIDE + x * 4) as u64;
    let transfer = transfer_3d_request(
        CMD_TRANSFER_TO_HOST_3D,
        TEXTURE,
        region(x, y, width, height),
        offset,
        TEXTURE_STRIDE,
        0,
    );
    assert_eq!(rig.status(&transfer), RESP_OK_NODATA);
    let expected: Vec<u8> = (0..height)
        .flat_map(|row| {
            let start = offset as u32 + row * TEXTURE_STRIDE;
            (start..start + width * 4).map(|index| index as u8)
        })
        .collect();
    // The guest may reuse its pages now; the renderer still gets what was there.
    rig.ram[BACKING_A as usize..BACKING_A as usize + 0x100].fill(0);
    assert_eq!(
        link.take_commands(),
        [GpuCommand::WriteResource {
            context: CONTEXT,
            resource: TEXTURE,
            level: 0,
            region: region(x, y, width, height),
            data: expected,
        }]
    );
}

#[test]
fn transfers_are_bounded_by_the_level_the_strides_and_the_backing() {
    let (mut rig, link) = renderer();
    rig.backed_texture(&link);
    let to_host = CMD_TRANSFER_TO_HOST_3D;
    let whole = region(0, 0, WIDTH, HEIGHT);
    let cases = [
        (
            TEXTURE,
            region(1, 0, WIDTH, 1),
            0,
            0,
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            TEXTURE,
            region(0, u32::MAX, 1, 2),
            0,
            0,
            RESP_ERR_INVALID_PARAMETER,
        ),
        (
            TEXTURE,
            GpuBox { z: 1, ..whole },
            0,
            0,
            RESP_ERR_INVALID_PARAMETER,
        ),
        // Stride shorter than the box's row.
        (
            TEXTURE,
            whole,
            0,
            TEXTURE_STRIDE - 4,
            RESP_ERR_INVALID_PARAMETER,
        ),
        // The span runs past the backing.
        (TEXTURE, whole, 4, 0, RESP_ERR_INVALID_PARAMETER),
        (
            TEXTURE,
            region(0, 0, 1, 1),
            u64::MAX,
            0,
            RESP_ERR_INVALID_PARAMETER,
        ),
        (99, whole, 0, 0, RESP_ERR_INVALID_RESOURCE_ID),
    ];
    for (resource, area, offset, stride, status) in cases {
        let transfer = transfer_3d_request(to_host, resource, area, offset, stride, 0);
        assert_eq!(
            rig.status(&transfer),
            status,
            "{area:?} at {offset} by {stride}"
        );
    }
    let mut level_one = transfer_3d_request(to_host, TEXTURE, region(0, 0, 1, 1), 0, 0, 0);
    level_one[HEADER_BYTES + 36] = 1;
    assert_eq!(
        rig.status(&level_one),
        RESP_ERR_INVALID_PARAMETER,
        "one level only"
    );
    // A resource of another context, and one without backing.
    rig.status(&create_3d_request(&texture(41)));
    let other = transfer_3d_request(to_host, 41, whole, 0, 0, 0);
    assert_eq!(rig.status(&other), RESP_ERR_INVALID_RESOURCE_ID);
    rig.status(&context_resource_request(CMD_CTX_ATTACH_RESOURCE, 41));
    assert_eq!(rig.status(&other), RESP_ERR_UNSPEC);
    let mut wrong_context = transfer_3d_request(to_host, TEXTURE, whole, 0, 0, 0);
    wrong_context[HEADER_CONTEXT_ID] = 9;
    assert_eq!(rig.status(&wrong_context), 0x1204);
    link.take_commands();
    // Stride 0 is the level's own; the whole texture fits its backing exactly.
    assert_eq!(
        rig.status(&transfer_3d_request(to_host, TEXTURE, whole, 0, 0, 0)),
        RESP_OK_NODATA
    );
}

#[test]
fn a_readback_lands_in_the_backing_before_its_fence_is_answered() {
    let (mut rig, link) = renderer();
    rig.backed_texture(&link);
    let (x, y) = (1, 3);
    let offset = (y * TEXTURE_STRIDE + x * 4) as u64;
    let read = fenced(
        transfer_3d_request(
            CMD_TRANSFER_FROM_HOST_3D,
            TEXTURE,
            region(x, y, 2, 2),
            offset,
            TEXTURE_STRIDE,
            0,
        ),
        11,
    );
    assert!(!rig.send(0, &read), "held for the renderer");
    assert_eq!(rig.slot_response(0), None);
    let commands = link.take_commands();
    let [
        GpuCommand::ReadResource {
            token,
            region: asked,
            ..
        },
        GpuCommand::Fence { sequence },
    ] = commands[..]
    else {
        panic!("a read and its fence: {commands:?}");
    };
    assert_eq!(asked, region(x, y, 2, 2));
    link.complete_read(token, (1..=16).collect());
    link.complete_fence(sequence);
    assert!(rig.poll());
    let response = rig.slot_response(0).expect("answered");
    assert_eq!(
        (response_type(&response), u64_at(&response, HEADER_FENCE_ID)),
        (RESP_OK_NODATA, 11)
    );
    for row in 0..2 {
        for byte in 0..8 {
            assert_eq!(
                rig.backing_byte(offset as u32 + row * TEXTURE_STRIDE + byte),
                (row * 8 + byte + 1) as u8
            );
        }
    }
}

#[test]
fn fenced_responses_wait_in_order_while_unfenced_ones_pass() {
    let (mut rig, link) = renderer();
    rig.status(&create_context_request(CONTEXT, b"t"));
    link.take_commands();
    // A fenced 2D command waits too: its response would signal every earlier fence.
    let mut create = fenced(header(CMD_RESOURCE_CREATE_2D), 1);
    create.extend(words(&[RESOURCE, FORMAT_B8G8R8X8, 4, 4]));
    assert!(!rig.send(0, &create));
    assert!(!rig.send(1, &fenced(submit_request(&[1, 0, 0, 0]), 2)));
    assert_eq!(
        rig.status(&header(CMD_GET_DISPLAY_INFO)),
        RESP_OK_DISPLAY_INFO,
        "unfenced, answered at once"
    );
    let commands = link.take_commands();
    let [
        GpuCommand::Fence { sequence: first },
        GpuCommand::Submit {
            context,
            ref commands,
        },
        GpuCommand::Fence { sequence: second },
    ] = commands[..]
    else {
        panic!("{commands:?}");
    };
    assert_eq!((context, &commands[..]), (CONTEXT, &[1u8, 0, 0, 0][..]));
    assert!(first < second);
    link.complete_fence(first);
    assert!(rig.poll());
    assert!(rig.slot_response(0).is_some() && rig.slot_response(1).is_none());
    assert!(!rig.poll(), "nothing more");
    link.complete_fence(second);
    assert!(rig.poll());
    // The context and the display info on the Rig's own descriptors, then the two in order.
    assert_eq!(
        rig.used_heads(),
        [
            0,
            0,
            SLOT_FIRST_DESCRIPTOR as u32,
            SLOT_FIRST_DESCRIPTOR as u32 + 2
        ]
    );
}

#[test]
fn a_fenced_error_waits_its_turn_too() {
    let (mut rig, link) = renderer();
    assert!(!rig.send(0, &fenced(submit_request(&[0; 4]), 1)));
    assert_eq!(rig.slot_response(0), None, "no such context, but fenced");
    let commands = link.take_commands();
    let [GpuCommand::Fence { sequence }] = commands[..] else {
        panic!("{commands:?}");
    };
    // A fence never handed out releases nothing.
    link.complete_fence(sequence + 100);
    assert!(!rig.poll());
    link.complete_fence(sequence);
    assert!(rig.poll());
    assert_eq!(response_type(&rig.slot_response(0).unwrap()), 0x1204);
}

#[test]
fn submits_are_bounded_by_the_request_and_the_limit() {
    let (mut rig, link) = renderer();
    rig.status(&create_context_request(CONTEXT, b"t"));
    link.take_commands();
    let mut short = submit_request(&[0; 8]);
    short[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&64u32.to_le_bytes());
    assert_eq!(
        rig.status(&short),
        RESP_ERR_INVALID_PARAMETER,
        "size past the request"
    );
    let mut huge = submit_request(&[0; 8]);
    huge[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(rig.status(&huge), RESP_ERR_INVALID_PARAMETER);
    assert!(link.take_commands().is_empty());
}

#[test]
fn a_renderer_that_falls_behind_leaves_the_queue_for_later() {
    let (mut rig, link) = renderer();
    link.push(vec![GpuCommand::Submit {
        context: 1,
        commands: vec![0; 32 << 20],
    }]);
    let used_before = used_idx(&rig.ram, CONTROL_USED);
    assert!(!rig.send(0, &header(CMD_GET_DISPLAY_INFO)));
    assert!(!rig.poll(), "still behind");
    assert_eq!(used_idx(&rig.ram, CONTROL_USED), used_before, "not taken");
    link.take_commands();
    assert!(rig.poll(), "resumed once the renderer caught up");
    assert_eq!(
        response_type(&rig.slot_response(0).unwrap()),
        RESP_OK_DISPLAY_INFO
    );
}

#[test]
fn a_reset_drops_held_responses_without_writing_them() {
    let (mut rig, link) = renderer();
    let mut create = fenced(header(CMD_RESOURCE_CREATE_2D), 1);
    create.extend(words(&[RESOURCE, FORMAT_B8G8R8X8, 4, 4]));
    assert!(!rig.send(0, &create));
    let [GpuCommand::Fence { sequence }] = link.take_commands()[..] else {
        panic!("a fence");
    };
    rig.gpu.reset();
    assert_eq!(link.take_commands(), [GpuCommand::Reset]);
    // The renderer answers late, for the device as it was.
    link.complete_fence(sequence);
    assert!(!rig.poll());
    assert_eq!(rig.slot_response(0), None);
    assert!(rig.used_heads().is_empty());
}

#[test]
fn a_3d_scanout_is_shown_by_the_renderer_until_a_2d_one_replaces_it() {
    let (mut rig, link) = renderer();
    rig.backed_texture(&link);
    assert_eq!(rig.set_scanout(0, TEXTURE, full()), RESP_OK_NODATA);
    assert!(rig.display.virtio_owns_screen());
    assert_eq!(rig.flush(TEXTURE, full()), RESP_OK_NODATA);
    assert_eq!(
        rig.set_scanout(
            0,
            TEXTURE,
            Rect {
                width: WIDTH + 1,
                ..full()
            }
        ),
        RESP_ERR_INVALID_PARAMETER
    );
    assert_eq!(
        link.take_commands(),
        [
            GpuCommand::SetScanout {
                scanout: 0,
                resource: TEXTURE,
                rect: full()
            },
            GpuCommand::Flush {
                resource: TEXTURE,
                rect: full()
            },
        ]
    );
    // fbcon takes the screen back: the renderer lets go, and the 2D path shows its image.
    rig.paint(0, 0, [10, 20, 30, 0]);
    rig.show(FORMAT_B8G8R8X8);
    assert_eq!(screen_pixel(&rig.display, 0, 0), [10, 20, 30, 0]);
    assert_eq!(
        link.take_commands(),
        [GpuCommand::SetScanout {
            scanout: 0,
            resource: 0,
            rect: Rect::default()
        }]
    );
    // A flush of a 3D resource that is not shown goes nowhere.
    assert_eq!(rig.flush(TEXTURE, full()), RESP_OK_NODATA);
    assert!(link.take_commands().is_empty());
}

#[test]
fn freeing_the_3d_scanout_turns_it_off() {
    let (mut rig, link) = renderer();
    rig.backed_texture(&link);
    rig.set_scanout(0, TEXTURE, full());
    link.take_commands();
    let mut unref = header(CMD_RESOURCE_UNREF);
    unref.extend(words(&[TEXTURE, 0]));
    assert_eq!(rig.status(&unref), RESP_OK_NODATA);
    assert_eq!(
        link.take_commands(),
        [
            GpuCommand::SetScanout {
                scanout: 0,
                resource: 0,
                rect: Rect::default()
            },
            GpuCommand::DestroyResource { resource: TEXTURE },
        ]
    );
    let transfer = transfer_3d_request(
        CMD_TRANSFER_TO_HOST_3D,
        TEXTURE,
        region(0, 0, 1, 1),
        0,
        0,
        0,
    );
    assert_eq!(
        rig.status(&transfer),
        RESP_ERR_INVALID_RESOURCE_ID,
        "detached with it"
    );
}

#[test]
fn the_2d_path_is_unchanged_with_a_renderer() {
    let (mut rig, link) = renderer();
    rig.paint(WIDTH - 1, HEIGHT - 1, [40, 50, 60, 0]);
    rig.show(FORMAT_B8G8R8X8);
    assert_eq!(rig.flush(RESOURCE, full()), RESP_OK_NODATA);
    assert_eq!(
        screen_pixel(&rig.display, WIDTH - 1, HEIGHT - 1),
        [40, 50, 60, 0]
    );
    assert!(
        link.take_commands().is_empty(),
        "nothing reaches the renderer"
    );
}
