# Guest 3D through virtio-gpu

**Status:** proposal. Nothing here is implemented; the device is still 2D only (`docs/plans/24-display-input.md`).

The aim is to give a Linux guest hardware-accelerated OpenGL (and perhaps Vulkan) through virtio-gpu, primarily on the web build, with the CLI as a second host. The web page can reach the host GPU only through WebGL2 (OpenGL ES 3.0) and WebGPU, so whatever the guest sends has to land on one of those.

**Recommendation:** route A first: `VIRTIO_GPU_F_VIRGL` with the guest's stock Mesa virgl driver, rendered by a new Rust crate onto WebGL2 in a render worker of its own. Start with the M0 spike below: a worker, a ring, one fence round trip and a cleared 3D scanout, with no shader translation yet. Route B (Venus on WebGPU) is not worth starting. It needs a guest-visible host memory region that the emulator's memory model does not have and the web build cannot share with another worker. It means writing what amounts to a Vulkan driver on WebGPU, and a desktop guest would still reach OpenGL through zink on top of it, ending up below what route A gives.

## Where it starts

- The device (`crates/uemulator/src/dev/vgpu.rs`) has no feature bits, `num_capsets` 0 in its configuration, and the 2D commands. It finishes every command before it writes the response, so fencing is an echo of the header. Resources are CPU pixel arrays bounded by `MAX_RESOURCE_BYTES` (256 MiB); the screen changes on `SET_SCANOUT` and `RESOURCE_FLUSH`, converted to the one public format (`convert_row`) and published to the `DisplayHandle`.
- The 2D path that just landed: frontends keep a `ScreenMirror` and `DisplayHandle::sync_mirror` returns only the 64x64 tiles that changed. The web page draws them into a WebGL2 texture on the page's main thread (`www/main.js`, red and blue swapped in the shader). The CLI viewer sends them as PNG bands encoded in parallel (kind 4 of the protocol in `crates/uemulator-cli/src/frontend/web.rs`).
- The library is `no_std` + `alloc` under `freestanding`, never touches host APIs, and reaches the host only through what the embedder hands `MachineConfig` (`thread_spawner`, `host_clock`, `poll`, the `NetHandle`/`InputHandle` queues with their device waker).
- Web: `uw_run` blocks the emulator's worker (`www/worker.js`) for the whole run. It cannot receive messages; the page talks to it through a control `SharedArrayBuffer`, and it can post messages out. The module's memory is defined by the module and is **not shared**: the build has no `atomics`/`shared-memory` flags, and `WebAssembly.instantiate` gets no memory import. Guest RAM is therefore invisible to any other worker; whatever another worker needs from it has to be copied out. The JIT's blocks import the module's memory and function table, so a shared memory would also mean rebuilding `core`/`alloc` with atomics (nightly `-Zbuild-std`). `docs/plans/13-web.md` notes the same obstacle for vCPU threads.
- PCI: each function has one 32-bit memory BAR0 of a fixed size (`dev/pci.rs`, `BAR_SIZE`) holding the virtio capabilities, MSI-X table and PBA. There is no 64-bit BAR and no virtio shared-memory capability (`VIRTIO_PCI_CAP_SHARED_MEMORY_CFG`). RAM is one flat buffer in which guest physical address `a` is byte `a` (`ram.rs`), and everything else on the bus traps into a device under the device lock.

## What the guest does with the feature bits

From Linux's `drivers/gpu/drm/virtio` (`virtgpu_kms.c`, `virtgpu_gem.c`, `virtgpu_ioctl.c`, `virtgpu_fence.c`):

- At probe the driver reads `num_capsets` and sends `GET_CAPSET_INFO` for each index, waiting up to 5 s for every answer (`wait_event_timeout(..., 5 * HZ)`); a zero or unknown id drops all capsets. Userspace later fetches the capset bytes with `GET_CAPSET`, and the result is cached in the kernel. Both answers must exist when the guest asks, so they cannot wait on a GPU that is still starting.
- With `VIRGL`, every DRM file gets a renderer context (`CTX_CREATE`) at its first GEM open when there is no `CONTEXT_INIT`. Every GEM object opened in a file is attached to that file's context (`CTX_ATTACH_RESOURCE`) and detached on close. That includes the kernel's own fbdev client, so turning the feature on changes the traffic of a plain console too, not only of 3D programs.
- Dumb buffers are still `RESOURCE_CREATE_2D` + `TRANSFER_TO_HOST_2D` with `VIRGL` (fbcon and simple KMS clients keep using the existing path). Without `VIRGL` but with `RESOURCE_BLOB`, dumb buffers become guest blobs (`BLOB_MEM_GUEST`, `USE_SHAREABLE`). That changes the 2D path, so `RESOURCE_BLOB` should never be offered alone.
- `CONTEXT_INIT` is honoured only together with `VIRGL` (the `CONTEXT_INIT` ioctl refuses without both).
- Fences: a fence response signals that fence and every earlier fence of the same dma-fence context (`virtio_gpu_fence_event_process`). Without context-init, all fences are on one global timeline; with it, there is one timeline per context and ring (`FLAG_INFO_RING_IDX`). Responses may come back out of order: each buffer carries its own completion, and QEMU holds fenced commands on a list while later ones complete.
- `TRANSFER_FROM_HOST_3D` is always fenced by the ioctl (`virtio_gpu_transfer_from_host_ioctl` allocates a fence unconditionally), so a readback can be asynchronous: the guest waits for the fence before it reads the backing.
- Guest userspace: Mesa's Gallium virgl driver (`virtio_gpu_dri.so`) for route A, and Mesa's Venus Vulkan driver (`libvulkan_virtio.so`) for route B. Venus refuses to start unless the kernel reports all of `3D_FEATURES` (VIRGL), `CAPSET_QUERY_FIX`, `RESOURCE_BLOB`, `HOST_VISIBLE` (a shared-memory region), `CROSS_DEVICE` (`VIRTIO_GPU_F_RESOURCE_UUID`) and `CONTEXT_INIT` (`required_params` in Mesa's `src/virtio/vulkan/vn_renderer_virtgpu.c`). Packaging (to check in M0): Alpine has `mesa-vulkan-virtio` in v3.24; Debian trixie's `mesa-vulkan-drivers` lists `libvulkan_virtio.so` and `virtio_icd.json`; the virgl Gallium driver is in Alpine's `mesa-dri-gallium` and Debian's `libgl1-mesa-dri`.

## Route A: virgl on WebGL2

### Device

- Feature: `VIRTIO_GPU_F_VIRGL` (bit 0), offered only when the embedder supplies a renderer; without one the device stays byte for byte what it is. `EDID`, `RESOURCE_BLOB`, `CONTEXT_INIT` are not needed.
- Configuration: `num_capsets` 1 or 2: `VIRTIO_GPU_CAPSET_VIRGL` (1, `struct virgl_caps_v1`) and `VIRGL2` (2, `virgl_caps_v2`, which embeds v1). The bytes come from the embedder at configuration time (below).
- Commands, beside the 2D ones (QEMU's `virtio_gpu_virgl_process_cmd` in `hw/display/virtio-gpu-virgl.c` handles the same set):
  - `GET_CAPSET_INFO`, `GET_CAPSET`: answered from the configured bytes.
  - `CTX_CREATE`, `CTX_DESTROY`, `CTX_ATTACH_RESOURCE`, `CTX_DETACH_RESOURCE`: the device tracks contexts and their resource sets (a context may only name attached resources) and forwards them.
  - `RESOURCE_CREATE_3D`: target, format, bind flags, width, height, depth, array size, mip levels, sample count. Checked against limits taken from the capset (`max_texture_2d_size` and friends) and against a separate 3D byte budget, then forwarded. The pixels live in the renderer; the device keeps only the description and the guest backing.
  - `TRANSFER_TO_HOST_3D`: a box, level, offset, stride and layer stride. The device reads the guest backing at command time (the `BackingCursor` walk) and forwards the bytes, so the guest may reuse its pages as soon as it sees the response.
  - `TRANSFER_FROM_HOST_3D`: forwarded; the renderer reads the box back, and the device writes it into the backing before it completes the command's fence.
  - `SUBMIT_3D`: the virgl command stream, `size` bytes of dwords in the request after the header, copied out and forwarded with its context.
  - `SET_SCANOUT` and `RESOURCE_FLUSH` on a 3D resource: forwarded (below, "Composition").
  - 2D resources keep their CPU path. A context that attaches a 2D resource gets it uploaded, and while it stays attached every `TRANSFER_TO_HOST_2D` is forwarded as well; QEMU sidesteps this by creating 2D resources inside virglrenderer too. Resource ids are one namespace for 2D and 3D.
- Fences: a command with `FLAG_FENCE` that went to the renderer keeps its descriptor chain until the renderer reports that fence id done; then the response is written and the queue interrupt raised. Commands without the flag are answered at once, as QEMU does. Renderer completions arrive in submission order, so one global "highest fence done" suffices without context-init.
- Backpressure: when the ring to the renderer is full, the device stops popping the control queue and retries at the next tick, which is a device that is busy, not an error. The vCPU never waits for the GPU.
- Size: about 1,500 lines with the forwarding, checks and held responses, plus a similar amount of tests.

### Renderer

What it consumes (virglrenderer's `src/virgl_protocol.h`, `src/virgl_hw.h`):

- 64 context commands (`VIRGL_CCMD_*`, including 9 video-codec ones that are not offered): object create/bind/destroy, framebuffer, viewport, scissor, vertex and index buffers, constant and uniform buffers, sampler views and states, `DRAW_VBO`, `CLEAR`, `BLIT`, `RESOURCE_COPY_REGION`, `RESOURCE_INLINE_WRITE` (small uploads inside the stream), queries, streamout, sub-contexts, `TRANSFER3D`/`COPY_TRANSFER3D`, `SET_TWEAKS`.
- 11 object types: blend, rasterizer, depth-stencil-alpha, shader, vertex elements, sampler view, sampler state, surface, query, streamout target, MSAA surface.
- Shaders arrive as TGSI text (Gallium's intermediate language, dumped to a string by the guest) and must be translated to GLSL ES 3.00. virglrenderer does the same in `vrend_shader.c`, which emits desktop GLSL or GLSL ES depending on the host.
- Capsets: `virgl_caps_v2` with a set of booleans, `glsl_level`, limits (`max_texture_2d_size` and others), format bitmaps, and 30 `VIRGL_CAP_*` bits. `VIRGL_CAP_HOST_IS_GLES` (bit 19) tells the guest driver that the host is GLES and to avoid what GLES lacks.

The renderer keeps the Gallium-style state objects per context, translates draws into GLES 3.0 calls, keeps a shader cache keyed by TGSI text plus the state that changes code generation (virgl's shader keys), and tracks resource-to-texture/buffer/renderbuffer mappings.

WebGL2 against what virgl expects:

- No buffer mapping (`glMapBufferRange`): every upload is `bufferSubData`/`texSubImage*` from the ring; readbacks are `readPixels` (into a pixel-pack buffer, then `getBufferSubData` once a `fenceSync` has passed).
- No `glGetTexImage`: read textures back by attaching them to a framebuffer.
- No BGRA textures: upload as RGBA and swizzle in the sampling shader (the 2D renderer already swaps red and blue this way).
- No geometry or tessellation shaders, compute, shader storage buffers, images or atomics (all GLES 3.1+), no texture buffers, no fp64, no logic ops. Clip distances, polygon mode and depth clamp only through extensions (`WEBGL_clip_cull_distance`, `WEBGL_polygon_mode`, `EXT_depth_clamp`). Primitive restart is always on with the fixed index; desktop GL lets it be set.
- `clientWaitSync` with a timeout of zero only; fences are polled, which suits a worker running its own event loop.

Level it can honestly advertise: virglrenderer gives a GLES 3.0 host `glsl_level` 130 (`vrend_fill_caps_glsl_version`: 120, or 130 at GLES 3.0, 310 at 3.1) and sets `HOST_IS_GLES`. With GLSL 1.30 and none of GL 3.1's texture buffers, Mesa's virgl driver should report about desktop GL 3.0 (compatibility contexts at 2.1 or 3.0) and GLES 3.0. Enough for compositors (GNOME, KDE and sway need GLES 2.0 or 3.0), glmark2-es2, Firefox and Chromium WebGL1, and most older games. Not enough for anything needing GL 3.2+ (geometry shaders) or compute. The exact version Mesa computes is to be checked in M0.

Size, three ways to get a renderer:

| option | where | size | notes |
|---|---|---|---|
| A1: write it in Rust | `crates/uemulator-virgl` (new), web and CLI | 18 to 25 thousand lines | virglrenderer's GL path is about 24 thousand lines of C (`vrend_renderer.c` 13k, `vrend_shader.c` 8.4k, `vrend_decode.c` 2k, `vrend_formats.c` 0.9k), plus Gallium's TGSI text parser. Ours targets only GLES 3.0, so it can drop the desktop-GL and 3.1+ branches that are a third of vrend. For scale: the JIT is 36k lines, the library 180k. |
| A2: link virglrenderer | CLI only, Linux hosts | about 1,000 lines of glue | What QEMU does. Full desktop GL 4.x on a Linux GPU host. Not upstream on macOS (EGL/GBM), a C dependency, and nothing for the web. |
| A3: virglrenderer built with Emscripten | the web render worker | unknown | vrend already has a GLES path, but it calls `glMapBufferRange`, uses GLES 3.1 branches when it finds them, and waits on `glClientWaitSync` with timeouts WebGL2 refuses; libepoxy would have to be replaced. A second toolchain beside `wasm32-unknown-unknown`. A spike, not a plan. |

A1 is the recommendation. It is the only option that serves both hosts with one implementation, it targets the one feature level both hosts actually have (below), and its decoder and translator are pure code that the project's tests can cover. It is written from the protocol and the behaviour of virglrenderer; nothing is transliterated. A2 stays a cheap optional backend for Linux CLI hosts once the device side exists; it is also the quickest way to check the device against a known-good renderer.

## Route B: Venus on WebGPU

### Device

- Features: `VIRGL`, `RESOURCE_BLOB` (3), `CONTEXT_INIT` (4), `RESOURCE_UUID` (2); `BLOB_ALIGNMENT` (5) optionally.
- Capset `VIRTIO_GPU_CAPSET_VENUS` (4). `CTX_CREATE` carries the capset id in its `context_init` field, with up to 64 rings for fences per context.
- Commands: `RESOURCE_CREATE_BLOB` (`BLOB_MEM_GUEST`, `HOST3D`, `HOST3D_GUEST`; flags `USE_MAPPABLE`, `USE_SHAREABLE`, `USE_CROSS_DEVICE`), `RESOURCE_MAP_BLOB` (with an offset into the host-visible region; the answer carries the caching type), `RESOURCE_UNMAP_BLOB`, `RESOURCE_ASSIGN_UUID`, `SET_SCANOUT_BLOB`, plus `SUBMIT_3D` with the Venus stream.
- The host-visible region (`VIRTIO_GPU_SHM_ID_HOST_VISIBLE`): a 64-bit prefetchable BAR announced by a `VIRTIO_PCI_CAP_SHARED_MEMORY_CFG` capability (QEMU: `virtio_pci_add_shm_cap(vpci_dev, 4, 0, hostmem, ...)` in `virtio-gpu-pci.c`, a gigabyte or more by default). Mapped blobs appear inside it, and the guest maps it as ordinary cacheable memory: Vulkan's host-visible coherent memory, the rings Venus uses to send commands and get replies, and every `vkMapMemory` pointer are plain loads and stores into it, with no trap anywhere.

That last point decides the route. In this emulator it means:

- A guest-physical range that is RAM to the soft TLB, the page walk, the JIT's inline probes and the IOMMU, but is not part of the flat RAM buffer, and that moves whenever the guest reprograms the BAR. Today "is this RAM" is a range check on one buffer (`is_ram_range`), and the JIT's fast path assumes it. One possible trick is to place the region at a fixed address inside the PCI hole and back it with the hole's slice of the flat buffer, which already exists when RAM exceeds 3 GiB. It would still need the BAR fixed, the buffer always spanning the hole, and `is_ram_range` taught about it. This is a memory-model change across `ram.rs`, the TLB and three JIT backends.
- Two parties reading and writing it at once: Venus's renderer polls the command ring while the guest writes it. On the web the renderer would be in another worker, which cannot see module memory, so every mapped range would have to be copied across at every submit (and dirty-tracked to be affordable; the code-page flags are the only write-tracking the emulator has) and copied back after fences. That is correct for Vulkan's coherency rules but slow, and it turns a ring polled by the renderer into one pumped by the emulator.

### Renderer

- The Venus protocol is generated from `vk.xml` and covers all of Vulkan 1.3 plus extensions; virglrenderer's renderer is mostly generated decoders that forward to a host Vulkan driver. We would need our own generator and decoder, then a Vulkan implementation on WebGPU: device memory and binding at offsets (WebGPU has opaque buffers and textures and cannot alias memory), descriptor sets and pipeline layouts, render passes, push constants (emulated with a uniform buffer), barriers (implicit in WebGPU), queries, and SPIR-V to WGSL for every shader (browsers accept only WGSL; `naga` can translate a subset).
- WebGPU has no geometry or tessellation stages, no sparse resources, no memory aliasing, and fewer formats and vertex formats. Mapping is asynchronous (`mapAsync`) and uploads are copies (`writeBuffer`).
- Level: a non-conformant Vulkan 1.0 subset: vkcube and simple compute, not DXVK or zink at any useful GL version. A desktop guest that wants OpenGL would run zink on top, which needs more Vulkan than this subset offers, so it would land below route A's GL 3.0.
- Size: the device and the memory-model change, 3 to 5 thousand lines; the generator and decoder, 3 thousand lines of generator and tens of thousands generated; a Vulkan-on-WebGPU layer, 30 to 50 thousand lines plus `naga`. Several times route A.

The CLI variant is far cheaper: forward Venus to a host Vulkan driver (virglrenderer's Venus renderer, or rutabaga, the Rust library QEMU's `virtio-gpu-rutabaga.c` drives; MoltenVK on macOS). It still needs the host-visible region, there mapped onto the renderer's real memory. That is the same memory-model work, plus mapping host GPU memory into the flat buffer. Worth revisiting once route A works and if Vulkan in CLI guests matters on its own.

## Architecture (route A)

### Library: the device and one interface

The library gains the device side and an interface, nothing that renders:

```rust
MachineConfig::gpu_renderer(GpuRenderer)       // offers VIRGL; without it the device is 2D as today

pub struct GpuRenderer {
    capsets: Vec<Capset>,                      // id, max version, bytes: GET_CAPSET_INFO / GET_CAPSET answer from these
    limits: GpuLimits,                         // what RESOURCE_CREATE_3D is checked against, and the byte budget
    link: GpuLink,                             // the two queues below
}
GpuLink::take_commands(&mut Vec<u8>) -> usize  // embedder side: encoded GpuCommand records, in order
GpuLink::complete(GpuCompletion)               // embedder side: fence done, readback bytes, scanout frame
```

- `GpuCommand` is the virtio level, already checked: context create/destroy/attach/detach, resource create/unref, transfer to host (with the bytes), transfer from host (box, a token), submit (context, stream bytes), set scanout, flush, and fence(id). It is not GL; the virgl stream passes through unparsed. The library stays free of the 3D protocol and of dependencies.
- The queues follow `NetHandle`/`InputHandle`: the device appends under the device lock and calls nothing; the embedder drains from its own thread or poll hook and pushes completions; the device waker (`set_device_waker`) wakes a halted guest when a completion arrives. Never `on_update` (it runs under the device lock), and never a call that blocks under it.
- With threads (`docs/plans/22-smp-threads.md`) the queues are shared handles, like the NIC's. Time-sharing needs nothing.

### The renderer crate

`crates/uemulator-virgl`, `no_std` + `alloc` with no dependencies, outside the library: the virgl stream decoder, the per-context state, the TGSI text parser, the TGSI to GLSL ES 3.00 translator, the format table, and the capset builder (`GpuLimits` from a host's WebGL2/GLES limits to `virgl_caps_v2` bytes). It drives a `Gl` trait, the GLES 3.0 entry points WebGL2 has (about 120), with integer object names. Hosts implement `Gl`:

- Web: JS imports that call WebGL2. The decoder runs in wasm in the render worker; one call per GL entry point crosses into JS, which is how Emscripten's GL works too.
- CLI: `glow` on a GLES 3.0 context from ANGLE (Metal on macOS, Vulkan or GL elsewhere). That is the same ANGLE Chrome puts under WebGL2, so the CLI and the web page have the same feature level and the same bugs; a desktop GL backend could raise the level later.

### Web: a render worker

```
page (main.js)                    emulator worker (worker.js, blocked in uw_run)        render worker (render.js)
  canvas ──transferControlToOffscreen──────────────────────────────────────────────────▶ OffscreenCanvas, WebGL2
  probe WebGL2 limits ◀──────────────────────────────────────────────────────────────── (asked before uw_create)
  uw_set_gpu(capsets, limits) ─▶ GpuRenderer in MachineConfig
                                   poll hook: GpuLink::take_commands ─▶ command ring (SAB) ─▶ decode, GL, present
                                   poll hook: completion ring (SAB) ◀─ fences, readbacks ◀─┘
```

- Why another worker: WebGL calls are synchronous to JavaScript, but frames reach the screen only when the context's worker returns to its event loop (or through `transferToImageBitmap` and a post; whether that works from a worker that never yields is an M0 question). The emulator's worker never yields. A render worker also runs on another core, so translation and GL call overhead come off the guest's thread, and it can receive `webglcontextlost`.
- Startup order: the render worker creates its context and reports its limits first. The page builds the capset bytes from them (the capset builder compiled into the module), hands them to `uw_create` with the other options (`uw_set_gpu`, like `uw_set_firmware`), and only then starts the guest. `GET_CAPSET_INFO` and `GET_CAPSET` are then answered from data, without a round trip.
- Rings: two `SharedArrayBuffer` rings (commands out, completions back), each with head and tail words. The poll hook copies `GpuLink` records into the command ring and `Atomics.notify`s the render worker, which waits with `Atomics.waitAsync` (to check in M0; otherwise it polls from `requestAnimationFrame`). Module memory is not shared, so every submit's stream and every transfer's bytes cross by this one copy. That is a `memcpy` in the poll hook, small next to translation. The command ring is about 32 MiB; when it is full the device stops popping its queue (backpressure, under "Device").
- Readbacks: the render worker reads pixels into a pack buffer, polls `fenceSync`, writes the bytes into the completion ring; the poll hook hands them to `GpuLink::complete`, and the device writes the backing, then completes the fence. All asynchronous; the guest is waiting on its fence anyway.
- Shared module memory would remove the copy (the render worker would read the backing in place) but needs an atomics build, `-Zbuild-std` and the JIT's tables reconsidered. Not part of this plan.
- Context loss: every 3D resource is gone. The renderer recreates its context, marks every guest context lost (later submits are dropped and their fences completed so the guest does not hang), and the screen shows the 2D path's last frame until the guest makes a new scanout. Guest programs will misrender until they restart. virtio-gpu has no "device lost" for virgl; QEMU has the same hole if virglrenderer resets.

### CLI

A render thread owns the ANGLE context and the renderer crate and drains `GpuLink`, which is how A2 (virglrenderer) would plug in as well. Presentation stays the existing viewer: a flush of a 3D scanout is read back (a pack buffer, one frame behind) and published to the `DisplayHandle` as if a 2D resource had been flushed, so `ScreenMirror`, the PNG bands, `frame()` and `/screenshot.png` work unchanged. A 1280x800 readback is 4 MB. Shipping GL commands to the viewer's browser instead would make rendering depend on a connected viewer and put the network inside every readback, so the CLI renders locally.

### Composition with the 2D path

- `DisplayHandle` learns a scanout source: pixels (today) or GPU resource `n`, generation `g`. A flush of a 3D scanout bumps the generation and records damage as usual. The pixels are what the host publishes: the CLI publishes readbacks, the web page publishes none, so `sync_mirror` returns nothing there while the scanout is a GPU resource. The cursor queue is unchanged in both: `UPDATE_CURSOR` resources are dumb buffers, so they stay 2D.
- Web: the render worker owns the canvas and draws the 3D scanout from its own texture, with no readback. A WebGL texture cannot be shared between contexts, so the 2D frames and the 3D scanout must either reach one context or sit on two stacked canvases. Proposal: stack them. The existing 2D canvas and its WebGL2 renderer in `main.js` stay as they are, a second canvas above it belongs to the render worker, and the page shows it while the scanout is a GPU resource (the render worker says so when it handles a `SET_SCANOUT`). The landed 2D path is untouched, and a lost 3D context falls back to the 2D canvas for free. Pointer mapping and scaling apply to both canvases. Screenshots ask the render worker (`convertToBlob` on its canvas) while it is in front. Moving the 2D renderer into the render worker (one canvas) is the alternative if stacking costs a compositor layer that matters.
- A guest that flips scanout between a dumb buffer (fbcon, Plymouth) and a 3D resource (a compositor's GBM surface) switches canvases. fbcon after a compositor exits goes back to pixels, and the next 2D flush publishes as today.

## Milestones

- **M0, spike (web only, throwaway allowed):** the render worker with a transferred `OffscreenCanvas`, the two rings, `uw_set_gpu` with hand-written capset bytes, the device offering `VIRGL` and answering capsets, contexts, `RESOURCE_CREATE_3D`, fenced `SUBMIT_3D` with only `VIRGL_CCMD_CLEAR` and `SET_FRAMEBUFFER_STATE` decoded, and `SET_SCANOUT`/`RESOURCE_FLUSH` of the cleared resource. Exit: an Alpine guest with `mesa-dri-gallium` runs a program that clears to a colour through EGL on the `virtio_gpu` render node (kmscube is the next step up), the colour appears on the page, and the guest does not hang on the fence. Settles the "Verify in M0" list.
- **M1, device:** the full route A device side in `vgpu.rs` with the `GpuRenderer`/`GpuLink` interface and a fake renderer in the tests (below). No renderer yet; every machine without `gpu_renderer` is unchanged.
- **M2, decoder:** `crates/uemulator-virgl` decodes every non-video command into typed state changes, with golden tests from captured streams. Optional A2 on Linux CLI hosts here to see real guests through the M1 device.
- **M3, shaders:** TGSI text parser and the GLSL ES 3.00 translator for vertex and fragment shaders: arithmetic, texturing, control flow, the shader keys. Golden tests; the outputs pass a GLSL ES validator at test time if one is vendored (no network).
- **M4, web renderer:** the `Gl` binding in the render worker, formats, transfers both ways, queries, blits, the stacked canvas. Exit: kmscube, glmark2-es2 and a compositor (weston or sway) on the page; numbers recorded in this plan.
- **M5, CLI renderer:** ANGLE plus `glow` on a render thread, readback into the `DisplayHandle`. Exit: the same guests in the CLI viewer.
- **M6, breadth:** streamout, MSAA surfaces, the extensions WebGL2 offers (clip distances, polygon mode, depth clamp), a desktop GL backend for the CLI if wanted. Venus reconsidered for the CLI only.

## Testing

Rules from `CLAUDE.md`: tests next to the code, no network, nothing under `~/.uemulator`.

- Device (`dev/vgpu/tests.rs`, the existing `Rig`): a fake renderer behind `GpuLink` that records commands and completes fences when told. Cases: capset info and bytes exactly as configured; feature bits absent without a renderer (the 2D machine unchanged); fenced responses held, and released by a completion; out-of-order completion with unfenced commands answered in between; a completion for a fence that was never sent is ignored; `TRANSFER_TO_HOST_3D` bytes taken at command time across backing entries; `TRANSFER_FROM_HOST_3D` written to the backing before the fence's response; 3D boxes, levels, layers and strides checked with overflow-checked sums; the 3D byte budget; a context naming an unattached resource refused; `CTX_DESTROY` and reset dropping held responses safely; a full command ring leaves the queue unpopped and the next tick resumes; 2D resources attached to a context mirrored to the renderer.
- Decoder and translator (`crates/uemulator-virgl`, pure, hosted and on `wasm32-wasip1`): golden streams in, typed calls out against a recording `Gl`; TGSI text in, GLSL ES out, compared to checked-in expected text; malformed streams (lengths past the end, unknown object handles, huge counts) produce errors, never panics.
- What cannot be tested here: real rendering. Node under `hack/wasm` has no WebGL, so the render worker's `Gl` binding is exercised only in a browser. A browser check (`hack/web/`) that draws a known stream and compares `readPixels` to expected pixels is worth having but is a check, not a test.
- A fake-kernel test (`testkit::FakeKernel`) cannot drive a virtio queue meaningfully by hand-assembled code; device behaviour is covered at the `Rig` level as today.
- Verification (not tests), recorded in this plan as each milestone lands: Alpine and Debian guests, `eglinfo` (vendor, version), kmscube, glmark2-es2 scores, weston or sway on the page, Firefox WebGL1.

## Risks

- **Size.** 18 to 25 thousand lines for A1 is the size of a major subsystem; the shader translator is where bugs will live. The milestones are ordered so every step leaves something useful and the 2D machine untouched.
- **Translation and call overhead.** WebGL validates every call and runs in Chrome's GPU process; a virgl frame is hundreds to thousands of calls. glmark2-es2 numbers will be far below native. The render worker keeps it off the guest's thread; how far below is an M4 measurement.
- **Guest driver assumptions.** Mesa's virgl driver is tested mostly against desktop GL hosts; GLES hosts (crosvm on Android and ChromeOS) are the precedent, and `HOST_IS_GLES` exists for them, but corner cases (format support, `SET_TWEAKS`) will come up.
- **Context loss** is unrecoverable for guest programs (above).
- **Browser support.** `OffscreenCanvas` with WebGL2 in workers is in current Chrome, Firefox and Safari; `Atomics.waitAsync` was not in Firefox when last checked (fallback: poll from `requestAnimationFrame`).
- **The 2D path regresses** if the `VIRGL` feature changes guest traffic (every GEM open now attaches to a context). M1's tests and a boot of the existing display verification cover it.

## Verify in M0

Not checked while writing this, and the plan depends on them:

- The GL and GLES versions Mesa's virgl driver reports for `glsl_level` 130 with `HOST_IS_GLES` (expected: desktop 3.0 or 2.1, GLES 3.0).
- Package names: Alpine `mesa-dri-gallium` (virgl), `mesa-vulkan-virtio`; Debian `libgl1-mesa-dri`, `mesa-vulkan-drivers`.
- Whether a worker that never yields can present WebGL frames (`transferToImageBitmap` and a post). If it can, a simpler variant renders on the emulator's worker with no rings; it still pays GL calls on the guest's thread.
- `Atomics.waitAsync` support in the target browsers.
- Whether Mesa's virgl driver needs `VIRGL2` or works with `VIRGL` alone, and which `virgl_caps_v2` fields it requires non-zero.

## References

- virtio 1.2 section 5.7 (GPU device); `include/standard-headers/linux/virtio_gpu.h` in `../ext/qemu-uplink`.
- QEMU (`../ext/qemu-uplink`, read only): `hw/display/virtio-gpu-virgl.c` (virgl commands, fences held on `fenceq`, blob mapping into `hostmem`), `virtio-gpu-gl.c`, `virtio-gpu-rutabaga.c` (the same device over rutabaga: gfxstream, cross-domain), `virtio-gpu-pci.c` and `virtio-vga.c` (the shared-memory BAR and capability), `virtio-gpu-base.c` (feature bits).
- Linux `drivers/gpu/drm/virtio`: `virtgpu_kms.c` (feature and capset probe), `virtgpu_gem.c` (contexts per file, dumb buffers), `virtgpu_ioctl.c` (getparams, fenced transfers), `virtgpu_fence.c` (fence ordering).
- virglrenderer: `src/virgl_protocol.h` (commands, objects), `src/virgl_hw.h` (capsets), `src/vrend_renderer.c` (`vrend_fill_caps_glsl_version`, the GLES path), `src/vrend_shader.c` (TGSI to GLSL).
- Mesa: `src/virtio/vulkan/vn_renderer_virtgpu.c` (Venus's required kernel parameters), `src/gallium/drivers/virgl` (the guest virgl driver).
