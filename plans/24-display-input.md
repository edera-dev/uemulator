# Graphical console: display and input

A machine can have a screen, a keyboard and a mouse next to its text consoles: `MachineConfig::display(DisplayMode::Framebuffer)`, CLI `--display framebuffer`. Everything is standard virtio plus UEFI GOP, so a stock Linux drives it with in-tree drivers and the firmware shows its console and menu on it. `DisplayMode::Serial` (the default) is the machine of earlier versions, byte for byte.

## Devices

Three PCI functions after the NICs and before the entropy source (`dev/mod.rs`, `DISPLAY_FUNCTIONS`), each counted against the 31 functions of the bus:

- **virtio-gpu** (`dev/vgpu.rs`, device 0x1050), 2D only, from virtio 1.2 section 5.7 and QEMU's `hw/display/virtio-gpu.c`. Control queue and cursor queue. No feature bits (no VIRGL, EDID, blob, context-init). Commands: `GET_DISPLAY_INFO` (scanout 0 enabled at the preferred size), `RESOURCE_CREATE_2D` (all eight 32-bit formats), `RESOURCE_UNREF`, `RESOURCE_ATTACH_BACKING` / `DETACH_BACKING`, `TRANSFER_TO_HOST_2D` (rows at `offset + row * stride`, the resource's stride, as QEMU and Linux agree), `SET_SCANOUT`, `RESOURCE_FLUSH`. Responses echo `FLAG_FENCE`/`FLAG_INFO_RING_IDX`, the fence id, context id and ring index: every command is finished when its response is written, so that is all fencing takes. Cursor commands are completed and dropped (the host's own pointer is the cursor; the tablet places the guest's under it). Unknown commands get `ERR_UNSPEC`.
  - Bounds: resources at most 16384 on a side and 256 MiB in total (`ERR_OUT_OF_MEMORY` past it, allocated with `try_reserve_exact`), backing entries at most 65536 and inside RAM (through the IOMMU when there is one), every rectangle inside its resource with overflow-checked sums, transfer offsets past the backing read zeros. Nothing a guest sends panics the host.
  - The screen changes on `SET_SCANOUT` (whole image) and `RESOURCE_FLUSH` (the flushed rectangle clipped to the scanout), not on transfers; pixels are converted to the one public format then.
  - Class 0x0300 (VGA), as QEMU's `virtio-vga`, not 0x0380 as `virtio-gpu-pci`: Linux's `virtio_gpu_pci_quirk` removes the firmware framebuffer (`aperture_remove_conflicting_pci_devices`, which calls `sysfb_disable` for the default VGA device) only for a VGA-class device. Without it `efifb`/`simpledrm` would keep fbcon on a framebuffer nobody scans out. Linux logs `[drm] pci: virtio-vga detected`. The device has no VGA ports or legacy window; the arbiter makes it the boot device anyway (the only VGA device, memory decoding on).
  - Host-requested modes: `DisplayHandle::request_size` becomes the preferred mode and raises `VIRTIO_GPU_EVENT_DISPLAY` with a configuration change; the guest re-reads the display info and decides.
  - Reset gives the screen back to the boot framebuffer and frees every resource.
- **virtio-input** keyboard and tablet (`dev/vinput.rs`, device 0x1052, class 0x0980), virtio 1.2 section 5.8, configuration answers as QEMU's `hw/input/virtio-input-hid.c`. Keyboard: `EV_KEY` codes 1 to 255, `EV_REP` (the guest repeats held keys), `EV_LED` num/caps/scroll. Tablet: `ABS_X`/`ABS_Y` 0 to 32767 (fixed, so a mode change needs no reconfiguration; host positions are scaled from screen pixels), `BTN_LEFT/RIGHT/MIDDLE/SIDE/EXTRA`, `REL_WHEEL`. `BUS_VIRTUAL` with QEMU's vendor 0x0627. Events go out in batches ending in `SYN_REPORT`, all of a batch or none of it (`Queue::rewind_to` when the driver has not posted enough buffers), behind one used-index update. Batches wait in the `InputHandle` while the driver has no buffers (at most 256 per device; consecutive pointer moves collapse into the latest). The status queue (LEDs) is completed and dropped.

## Boot framebuffer (`dev/bootfb.rs`)

Before the virtio-gpu driver loads there is a linear framebuffer the UEFI firmware draws on and hands to the OS as its GOP. It is QEMU's `ramfb` beside a virtio-gpu (`virtio-ramfb`): plain guest RAM, not a BAR. Every store into a BAR traps into the device model under the device lock; text drawing and an `efifb`/`simpledrm` console scrolling megabytes per line would crawl. In RAM the firmware, GRUB and the kernel draw at memory speed.

- Placement (`boot/uefi.rs::framebuffer_region`, a pure function of the RAM size so the machine and the loader agree): the top of the RAM below the PCI hole, page-rounded, aligned down to 2 MiB. The firmware payload is placed below it, the handoff's RAM regions list it as reserved (`reserve`), so the firmware never allocates it and Linux's e820 shows it reserved (1280x800: `[mem 0x7fc00000-0x7ffe7fff] reserved` with 2 GiB).
- Format: 4 bytes per pixel, blue, green, red, unused; pixels per line = width.
- Scanning: nothing reports writes to RAM, so `BootFramebuffer::tick` (from `Devices::tick`) compares the region with the screen at most every 1/30 s of guest time and copies the rows that differ (`read_raw`: a CPU address, never IOMMU-translated). Not while the virtio driver owns the screen (`DisplayShared::virtio_owns_screen`, set on the first `SET_SCANOUT`, cleared on device reset); then the next scan is a full one.
- Only machines booted through the UEFI firmware have it. The direct boot (`--firmware none`) leaves `screen_info` empty: the screen stays black until virtio-gpu loads (handing Linux the RAM framebuffer there would work the same way, but nothing needs it). The BIOS path has none either.
- Handoff version 2 (`plans/19-uefi-firmware.md`) carries base, size, width, height and pixels per line at 0x338.

## Firmware (`crates/uemulator-firmware`)

- `gop.rs`: `EFI_GRAPHICS_OUTPUT_PROTOCOL` on its own handle, one mode, `PixelBlueGreenRedReserved8BitPerColor`, `QueryMode` (pool copy), `SetMode` (clears), `Blt` (fill, video to buffer, buffer to video, video to video with overlap-safe row order). The Linux stub turns it into `screen_info` (checked: `orig_video_isVGA` 0x70 `VIDEO_TYPE_EFI`, 1280x800, depth 32, base 0x7fc00000, line length 5120, 8:16/8:8/8:0/8:24).
- `text_screen.rs` (hosted tests): the 80x25 text console drawn on the framebuffer, cells of attribute and character, the largest whole scale that fits (2x on 1280x800, which fills it exactly), centred with black margins, VGA palette, an underline cursor, scrolling as a copy within the framebuffer (it is RAM). Glyphs: GNU Unifont 8x16 (`font8x16.bin`, ASCII, Latin-1, arrows, box drawing, block elements, the triangles UEFI names, U+FFFD for the rest), made by `hack/uefi/make-font.py` from Unifont's `.hex` (OFL 1.1 or GPL 2+ with the font exception). `AnsiParser` reads the control sequences the firmware's own screens write (SGR 0/7/30-37/40-47, `2J`, `H`).
- `console.rs`: `SimpleTextOutput` draws on both the serial terminal and the screen (a console splitter in effect; a line feed is CR LF on both, as the serial side always did). The boot manager's own output (`console::print!`) also goes to both. `SimpleTextInput` takes keys from the virtio keyboard first, then the serial line.
- `virtio_input.rs` + `keymap.rs` (hosted tests): a polling virtio-input driver for every function whose `EV_KEY` bitmap has `KEY_A` (tablets are skipped), 64 event buffers, Linux key codes to UEFI keys with shift, caps lock and control (Ctrl-letter is the control character, which GRUB's editor reads), US layout, keypad as digits. Reset at `ExitBootServices`, before its rings stop being boot-services memory.
- `virtio_pci.rs`: the transport (capabilities in BAR0, feature negotiation, queue setup) shared by `virtio_blk.rs` and `virtio_input.rs`.

## Kernel command line

With a display and no explicit `cmdline`, the default line gains `console=tty0` after its `console=hvc0` (or `ttyS0`) and before `cmdline_append`: the last `console=` is `/dev/console`, so init's shell is on the screen, the text console still gets the kernel log, early boot text stays on the 16550 (earlyprintk/earlycon), and an appended `console=` still wins. The CLI's attached terminal therefore shows the kernel log and whatever the distro runs on hvc0; interaction is on the screen.

## Public API (`lib.rs`, `display.rs`)

```rust
MachineConfig::display(DisplayMode::Framebuffer)    // Serial (default) | Framebuffer
MachineConfig::display_size(width, height)          // 320x200 ..= 2560x1600, default 1280x800
EmulatedMachine::display() -> Option<DisplayHandle> // Send + Sync + Clone, survives reboots
EmulatedMachine::input() -> Option<InputHandle>

DisplayHandle::size() -> (u32, u32)
DisplayHandle::generation() -> u64                  // bumps on every change; 0 = blank start
DisplayHandle::frame() -> Frame                     // whole screen copy
DisplayHandle::update_since(generation, &mut Vec<u8>) -> Option<FrameUpdate> // changed rect + its pixels
DisplayHandle::wait_for_update(generation, timeout) -> u64                   // condvar; returns at once when freestanding
DisplayHandle::on_update(Fn(u64))                   // push; runs on the device thread inside the device lock
DisplayHandle::request_size(width, height)

InputHandle::key(linux_code, pressed) / key_usb(usb_hid_usage, pressed) / release_all()
InputHandle::pointer_move(x, y)                     // screen pixels, absolute
InputHandle::pointer_button(PointerButton, pressed) / scroll(clicks)
linux_key_for_usb_usage(usage) -> Option<u16>       // Linux hid-input's table
BYTES_PER_PIXEL, Rect, Frame, FrameUpdate, PointerButton
```

- One pixel format for every frontend: little-endian XRGB8888 (bytes B, G, R, unused), what GOP and DRM's default use; the GPU converts other resource formats at flush.
- Damage: the last 64 (generation, rectangle) changes; `update_since` returns their union after the caller's generation, or the whole screen when the caller is further behind or the size changed since.
- Locking: the devices publish while holding the device lock and take the handle's lock briefly; a frontend takes only the handle's lock. Input handles queue under their own lock and wake the device thread (the `set_device_waker` pattern of `NetHandle`), so a halted guest sees a key at once rather than after its idle sleep.
- `no_std` + `alloc` under `freestanding`: only `crate::sync`; no threads needed (a freestanding host polls `generation`/`update_since` from its poll hook or uses `on_update`).

## CLI

- `--display serial|framebuffer` on `quick create|edit|boot` and `advanced boot|boot-firmware|boot-uefi` (`HardwareArgs`), saved in `machine.toml` as `display = "framebuffer"` (absent = serial), shown by `quick show`.
- `--viewer-address ADDRESS:PORT` (`RuntimeArgs`, never saved), default `127.0.0.1:5800`, falling back to a free loopback port when that one is taken; port 0 picks one. The URL is printed on stderr before the terminal goes raw. A non-loopback address prints a warning: whoever reaches it has the keyboard.
- `frontend/`: `trait Frontend { fn start(self: Box<Self>, Screen) -> Result<String, String> }`, `Screen` = the display and input handles. `frontend::web` is the only one; a native window would be another implementation (one that must own the main thread would get a `run` instead, the `Screen` stays the interface).

### Web viewer protocol (`frontend/web.rs`, `frontend/web/viewer.html`)

- Hand-written HTTP/1.1 over `std::net`, a thread per connection: `GET /` the page, `GET /ws` the WebSocket (RFC 6455; SHA-1 and base64 in `websocket.rs`, checked against the RFC and FIPS vectors), anything else 404/405. Request heads are capped at 16 KiB and 10 s.
- Browsers let any site send requests to 127.0.0.1, so without checks a web page could type into the machine. Every request must carry a `Host` naming the viewer: its bound address and port, any loopback name (`localhost`, `127.x`, `[::1]`) for a loopback listener, any address literal for one bound to every address; never a host name, which is what DNS rebinding (another site's name resolving to 127.0.0.1, where `Origin` and `Host` agree) produces. On top of that the socket refuses an `Origin` other than `http://<Host>`. Both answer 403.
- Server to page: binary messages, kind byte 1, then screen width, height, x, y, width, height as u16 LE, then a PNG of the rectangle (8-bit RGB, `Sub` filter, zlib fast; `png.rs`). The first message is the whole screen. One update in flight per viewer: the page answers `a` after drawing, and the server collects changes (at most one update per 16 ms) until then, so a slow link or browser gets fewer, larger updates instead of a backlog.
- Page to server: text lines `k USAGE 0|1` (USB HID usage, from `KeyboardEvent.code`; when `code` is empty, as with on-screen keyboards, from `key` in a US layout with shift added), `m X Y` (screen pixels, once per animation frame), `b INDEX 0|1` (left, right, middle, back, forward), `w CLICKS`, `r` (release everything: sent on blur, and the server sends it when a viewer disconnects), `a`.
- Measured: a guest scrolling its console continuously reached 34 full-screen (1280x800) updates per second in the browser pane on loopback; a flat 1280x800 frame is 18 KB as PNG.

## Verification (not tests)

`advanced boot --kernel vmlinuz-virt --initramfs initramfs-virt --display framebuffer` with Alpine 6.18.58 (the initramfs has `virtio-gpu.ko` and `virtio_input.ko`; an appended cpio replaced `/init` with one that loads them and runs a shell on tty1, since Alpine's own init panics without boot media):

- `--firmware none` and `--firmware uefi`: `[drm] pci: virtio-vga detected`, `Initialized virtio_gpu 0.1.0`, `fb0: virtio_gpudrmfb`, `Console: switching to colour frame buffer device 160x50`, `/dev/dri/card0`, `input: uemulator Virtio Keyboard` (handlers `sysrq kbd`, `EV=120003` with `EV_REP`), `input: uemulator Virtio Tablet` (`ABS=3`, `REL=100`, buttons).
- Typing in the viewer page reached the shell on tty1 (`echo typed-on-screen > /dev/hvc0` showed up on the CLI's terminal). A guest `reboot -f` came back through the firmware with the display working.
- UEFI: the e820 reserved range and `screen_info` above; `advanced boot-uefi --boot-menu --display framebuffer` showed the boot menu on the screen with the selection in reverse video, and Down, Enter from the viewer picked "Reset the machine".

- The installed Alpine disk (read-only, `advanced boot-uefi --disk ...,ro`) through the firmware and its GRUB: GRUB drew on the screen through `SimpleTextOutput`, the system loaded virtio_gpu itself and showed its tty1 login on the screen, and `root` typed in the viewer appeared at the prompt.

Not verified: the handover from `efifb`/`simpledrm` to virtio_gpu. Alpine builds both as modules (`efifb` is not even shipped) and loads neither; on the installed system no `efi-framebuffer`/`simple-framebuffer` platform device was left after virtio_gpu probed, which fits `sysfb_disable` having run but does not prove it (the kernel config was not readable). A kernel with `simpledrm` built in (whether Debian's or Fedora's are was not checked) is the case to try: if virtio_gpu did not remove it, fbcon would stay on the framebuffer in RAM, which nothing scans out once the driver sets a scanout, and the screen would stop changing. Also not verified: GRUB's `gfxterm` on the GOP, the tablet seen by a guest consumer (no `evdev` in the initramfs), a desktop using cursor planes.

## Open questions

- Cursor plane: a guest desktop that uses the cursor queue gets no visible cursor image; the host pointer stands in. Compositing the cursor resource would need its image kept and drawn in `present`.
- The boot framebuffer costs 4 MB of guest RAM and a scan per 1/30 s until the driver loads; a write-tracking hook (the code-page flags are a model) would make the scan incremental.
- EDID and multiple scanouts are not offered.
