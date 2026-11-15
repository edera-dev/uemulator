# Apple platforms: a SwiftUI app

**Status:** proposal. M0, the spike that decides whether the WebKit engine is worth building, is done ("M0 results" below: it is, served from loopback HTTP rather than a scheme handler). M1 to M3 are built: `apple/Uemulator.xcodeproj` boots the Alpine demo on the native engine (JIT on macOS, interpreted on iOS) and on the WebKit engine (the web build's wasm JIT in a hidden web view, on both), with a screen, keyboard and pointer, from the demo or an ISO.

A native app for macOS, iOS and iPadOS (visionOS later, tvOS only interpreted) that does what the web page does (`13-web.md`): pick or create a machine, boot it, show the terminal or the screen, type into it, network it. The UI is SwiftUI throughout. No web page is ever shown.

## Why there are two engines

Apple's platforms split on one question: may the app map executable memory?

- **macOS: yes.** A hardened-runtime app with `com.apple.security.cs.allow-jit` gets `MAP_JIT`, which is what `sys.rs` already uses on Apple Silicon (and `pthread_jit_write_protect_np` for the W^X switch). The library links into the app and runs exactly as under the CLI: aarch64 JIT (x86_64 JIT on Intel Macs), one host thread per vCPU, no 4 GiB cap. WebAssembly would only make the Mac slower.
- **iOS, iPadOS, visionOS, tvOS: no.** An App Store app has no `dynamic-codesigning` entitlement. `mmap(MAP_JIT)` fails, so `Jit::new`'s probe falls back to the interpreter. Embedding a WebAssembly runtime with its own compiler (wasmtime, wasm3's AOT, ...) does not help: it needs the same executable memory. An in-process `JSContext` has no JIT on iOS either; it would interpret the WebAssembly, which is slower than interpreting x86 natively. The EU browser-engine entitlement is only for browsers, and the debugger-attached JIT tricks are not shippable.

The one process on iOS that may JIT is WebKit's WebContent process, which compiles WebAssembly (BBQ, then OMG) for any `WKWebView`. The library already has a backend that emits WebAssembly (`jit/wasm.rs`, `jit/emit_wasm.rs`) and a host for it (`www/jit_host.js`). On V8 that backend boots a Debian netboot kernel in about 6 s against 55 s interpreted (`13-web.md`, "Tests").

So "JIT through WebAssembly" means:

| engine | where the emulator runs | JIT | platforms |
|---|---|---|---|
| `NativeEngine` | in the app process: the library as a static library | aarch64 / x86_64 where `MAP_JIT` works, else the interpreter | macOS (JIT); iOS, iPadOS, visionOS, tvOS (interpreter) |
| `WebKitEngine` | in a hidden `WKWebView`'s WebContent process: the existing `uemulator_web.wasm` | the WebAssembly backend, compiled by JavaScriptCore | iOS, iPadOS, visionOS (macOS too, for testing) |

The thing to understand about `WebKitEngine`: a JIT block is a wasm module that imports the emulator's `memory` and `table` (`13-web.md`, "The JIT"). It reads guest RAM and the `Cpu` directly in that linear memory. Native Rust code in the app cannot share that memory, because it lives in another process. So the WebAssembly build is not a JIT plugged into a native core. **The whole core runs inside WebKit, and Swift is a remote UI** that exchanges bytes with it. The web view is never on screen; it only executes code. The emulator side is the web port with its page replaced by a headless shim.

On iOS the user picks the engine per machine. The default is `WebKitEngine` when M0 shows it beats the native interpreter, which it should by a wide margin. The native interpreter stays as the engine that always works: no WebContent memory ceiling, real threads for SMP, no 3 GiB RAM cap.

## Layout

```
apple/
  Uemulator.xcodeproj           committed; one multiplatform app target, plus UemulatorKitTests
  App/                          SwiftUI: UemulatorApp, machine list, settings, run view
  UemulatorKit/                 local Swift package, the engine layer (no UI)
    Sources/UemulatorKit/       MachineEngine protocol, NativeEngine, WebKitEngine, MachineStore
    Sources/UemulatorCore/      module map + header for the static library (binary target)
  WebEngine/                    the headless shim: engine.html, engine.js (from worker.js), bridge.js
  Entitlements/                 macOS (allow-jit, network client/server, user-selected files), iOS
crates/uemulator-apple/         new crate: staticlib with a C ABI for NativeEngine
hack/apple/build-core.sh        cargo builds -> UemulatorCore.xcframework; also runs hack/web.sh for the wasm
```

- `crates/uemulator-apple` mirrors `uemulator-web`'s C ABI on purpose: `ua_create(config) -> handle`, `ua_add_disk`, `ua_set_firmware`, `ua_start`, `ua_stop`, plus callbacks that match the web host imports one for one (`output`, `input`, `frame_ready`/`frame`, `cursor`, `input_events`, `net_recv`/`net_send`, `wisp_*`, `log`). With both engines on the same event vocabulary, `MachineEngine` is one Swift protocol and the UI never knows which engine it has. The header is handwritten (no cbindgen, keeping the crate's dependencies to the library's), and it is the only thing outside `lib.rs` that sees the library: it uses the public `EmulatedMachine` API like the CLI does, and `lib.rs` stays the entire public API.
- The xcframework has slices for `aarch64-apple-darwin` + `x86_64-apple-darwin` (lipo'd), `aarch64-apple-ios`, `aarch64-apple-ios-sim`, and later `aarch64-apple-visionos[-sim]` / `aarch64-apple-tvos` (tier 3: `-Zbuild-std`). An Xcode "Run Script" build phase calls `hack/apple/build-core.sh` with `$PLATFORM_NAME` and `$CONFIGURATION`, so a plain ⌘B builds the Rust. CI builds it all with `xcodebuild -scheme Uemulator -destination generic/platform=iOS`.
- Bundled resources: `uemulator_web.wasm` and the shim (iOS/visionOS), `uemulator-firmware.bin` (both engines; the CLI embeds it through `build.rs`, and the app loads it as a resource instead), and the Secure Boot keys the firmware needs.
- Warnings: `cargo clippy --release --target aarch64-apple-ios -p uemulator-apple` must print zero, like the other targets, and the Swift side builds with warnings as errors.

## WebKitEngine

A `WKWebView` that sits in the view hierarchy, 1×1 with opacity 0 (M0 boots in it like that). It loads `http://127.0.0.1:<port>/<token>/engine.html` from a small HTTP server the app runs on loopback (`NWListener`), which serves the bundle and the app's files. Not a `WKURLSchemeHandler`: see the next paragraph. The shim is `www/worker.js` and `www/jit_host.js` almost unchanged. `main.js`, xterm.js and every piece of UI are dropped.

**Cross-origin isolation.** `uw_run` blocks a Worker, and the page talks to it through `SharedArrayBuffer` rings (`www/protocol.js`). WebKit grants `SharedArrayBuffer` only to a cross-origin-isolated page, so every response carries `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, as `hack/web/serve.py` sends. M0 found that a custom scheme is not enough: on `uemulator://` the page reports `crossOriginIsolated === true` and still has no `SharedArrayBuffer` constructor and no shared wasm memory, also after a first navigation to a page without the headers (which should have forced a process swap). WebKit only enables them in a WebContent process it started for a COOP navigation, and evidently not for custom schemes. The same headers from `http://127.0.0.1` work: `127.0.0.1` is a secure context, and ATS lets loopback through without an exception. The server listens on 127.0.0.1 only. Other apps can still reach a loopback port, so every path starts with a random per-launch token, and anything else gets a 404.

**The bridge.** Everything the page did with UI now crosses a process boundary:

- Disk reads: the worker already reads disks with synchronous range XHRs (`host_disk_read`, `rangeRequest`). Those go to `/<token>/disk/<id>` on the loopback server, which answers them from the file with `FileHandle` reads (M0: a 512-byte range read from a worker works). Nothing new is needed, and big ISOs never load into memory.
- Disk writes: today they land in an in-memory udisk overlay that dies with the page. For a persistent disk, add a `host_disk_write` import: a synchronous `POST /<token>/disk/<id>?offset=` whose body the server writes to the udisk file. M0: a worker's synchronous 1 MiB POST arrives whole in 6 to 11 ms, on both transports.
- Console, screen, input, network and Wisp: one binary stream each way between the page's main thread and the app. The page's main thread drains the worker's rings as `main.js` does today and forwards the bytes; going the other way, it fills the input, net and Wisp rings. Two candidate transports, measured in M0 (a 3 MiB frame, round trip, one in flight):
  1. `WKScriptMessageHandlerWithReply` / `callAsyncJavaScript`. Simple, but its values are property lists (no `ArrayBuffer`), so bytes travel as base64 strings: 31 to 34 ms a frame.
  2. A WebSocket from the page to an `NWListener` on loopback in the app. Binary frames, no base64; same listener and token as the HTTP server: 5.2 to 5.9 ms a frame (about 550 MB/s), 6× faster. **This is the transport.**

  The screen is the heavy one: a 1024×768 frame is 3 MiB, 180 MiB/s at 60 Hz. Only the damaged rectangles cross (virtio-gpu flushes come with rects; `host_frame` takes a rect), at most once per display refresh. The app acknowledges each frame the way the page does (`FRAME_ACK`), so a slow bridge drops frames instead of queueing them.
- Network: the guest's NIC stays on the library's `Gateway` + `WispClient`, which run inside the wasm (`27-wisp.md`). The app is the Wisp v1 relay: a small Swift server for the protocol (TCP and UDP streams over `Network.framework`) on the same stream. No external relay, no browser networking. `NativeEngine` uses the library's gateway directly, as the CLI does.
- Lifetime: WebKit can kill WebContent under memory pressure (`webViewWebContentProcessDidTerminate`). The engine reports it as the machine stopping with an error, never as a crash. Guest RAM counts against the WebContent process's jetsam limit, so the RAM picker on this engine is capped by what M0 measures on the smallest supported device (and never above the wasm32 3 GiB in any case).

**What the WebAssembly JIT loses on JavaScriptCore.** Measured in M0 (simulator):

- The compile worker relies on V8's process-wide native module cache (`jit_compiler.js`). JSC has no measured equivalent, so the page already skips it outside V8 (`compileWorkerHelps`), and blocks compile on the emulator's thread. The boot ("M0 results") is 1.4× slower than V8 without its compile worker and 1.9× slower than V8 with it, and module creation is the first place to look: 26k one-block modules, all compiled on the emulator's thread. Batching blocks into modules without the worker (the batching exists, gated on the queue), or a JSC-specific compile worker if JSC shares compiled code, is the first JIT work for this engine.
- Tail calls chain blocks (`return_call_indirect`). Without them the chain stub fails to compile and the JIT runs unchained. That is correct but slower. JSC validates `return_call` and `return_call_indirect` (and fixed-width SIMD) in the simulator's WebKit, so the chained JIT runs; the minimum iOS version that has them still needs checking on devices.
- Fixed-width SIMD (`probe_simd`) is probed already and works either way.
- SMP: the WebAssembly JIT runs all vCPUs on one thread (`13-web.md`, "The JIT"). A machine that needs real parallelism belongs on `NativeEngine`.

## NativeEngine

`ua_start` spawns the machine through the library's normal hosted path (`std`, threads, `SmpMode::Threads`). Callbacks fire on the emulator threads, and Swift hops them to the main actor in batches (console every 16 ms like `worker.js`; frames on `CADisplayLink`/`CVDisplayLink`). On iOS the JIT probe fails, and the library falls back to the interpreter by itself. The engine reports "interpreted" so the UI can say so.

There is no scheme handler on this engine: disks are opened directly through the library's own file-backed raw/udisk code, the same as the CLI.

## The SwiftUI app

It follows the web page's structure (`13-web.md`, "The page") so the two stay recognizable:

- **Machine list** (sidebar / `NavigationSplitView`): named machines stored the same way the CLI does (`machine.toml`, `artifacts/`, `disk-<n>.udisk`, `nvram.kv`) under the app container's `Machines/`. On macOS a user-granted folder bookmark can point it at `~/.uemulator/machines`, so the CLI and the app share machines. Parsing `machine.toml` stays in Rust (`ua_machine_load`) so the two never disagree about the format.
- **Presets and settings:** the web page's presets (live CD/installer ISO, disk image, kernel to a shell, the Alpine demo), and the same choices: firmware (`uefi`/`bios`/`none`), memory, vCPUs, drives (Files app / `fileImporter`, drag and drop on macOS and iPad, URLs downloaded into the machine), kernel/initramfs/command line, display/console, network on/off, TPM, Secure Boot, hypervisor (Xen). The same warnings the page shows before Start. Those checks should move out of `main.js` into Rust shared by both (`ua_validate`), or the web page and app drift apart.
- **Run view:** a state chip (Running + uptime, Starting, Stopped, error) and Terminal | Screen.
  - Terminal: SwiftTerm (MIT, VT100/xterm, macOS and iOS views). Bytes in from `output`, keystrokes out through `input`, size changes through `resize`. On iOS, an accessory bar for Esc, Ctrl, Tab and arrows.
  - Screen: a `MTKView` (or a `CALayer` fed `IOSurface`s) that uploads damaged rects into a texture, aspect-fit, with "resize guest to fit". Keyboard input becomes evdev codes for virtio-input: `NSEvent.keyCode` on macOS, `UIKey.keyCode` (HID usages) from `pressesBegan` on iPad hardware keyboards. Pointer, trackpad and touch become absolute tablet events, and the guest cursor (`host_cursor`) is drawn as a hardware cursor where the platform allows. Ctrl+Alt+Del is a toolbar button.
- **Network:** the shared Ethernet segment between machines (`SegmentAgent`) maps to "machines in this app share a segment", with the app forwarding frames between running engines. Internet goes through the in-app Wisp relay above, or the native gateway.

## Milestones

- **M0 — WebKit spike (decides the plan). Done; results below.** A throwaway iOS app (`apple/M0Spike`, run by `hack/apple/m0.sh`): hidden `WKWebView`, scheme handler and loopback HTTP with COOP/COEP, the existing `uemulator_web.wasm` and `worker.js`, console bytes to a `Text` view. It answers, on a real iPhone and on the simulator: `crossOriginIsolated` on a custom scheme, whether the web view runs while hidden, tail calls and SIMD on JSC, sync XHR with a body from a worker to a scheme handler, the WebContent memory ceiling, and boot time to `Run /init` for the Alpine demo with the JIT, with `--no-jit` in the same web view, and on `NativeEngine` interpreted. Exit: those numbers written into this file. If `WebKitEngine` doesn't beat the native interpreter by a clear margin, it is dropped, and iOS ships interpreted.
- **M1 — Core and project. Done** ("M1 as built" below). `crates/uemulator-apple`, `hack/apple/build-core.sh`, the xcframework, the Xcode project with a SwiftUI shell, and `NativeEngine` booting the Alpine demo to a shell on macOS (JIT) and iOS (interpreted) in a SwiftTerm view. Exit: ⌘R on both, and `xcodebuild` in CI.
- **M2 — WebKitEngine. Done** ("M2 as built" below). The headless shim, the bridge (winning transport from M0), disk reads through the loopback server, console and keyboard. Exit: the same Alpine demo on iOS through `WebKitEngine`, with the JIT on.
- **M3 — Screen and input. Done** ("M3 as built" below; the KDE desktop exit was checked on macOS, the iPad keyboard-and-trackpad part on the simulator only). Damaged-rect frames on both engines, virtio keyboard and tablet, cursor. Exit: a live ISO's desktop (the KDE ISO from `20-jit-perf.md`) is usable on an iPad with a keyboard and trackpad.
- **M4 — Machines and persistence.** `MachineStore` on the CLI's format, `host_disk_write` (or flushed overlays), NVRAM and TPM state, presets and validation shared with the web page in Rust. Exit: install Alpine to a disk on iOS, quit the app, boot the installed disk.
- **M5 — Network.** The Swift Wisp relay, the in-app segment between machines. Exit: `apk update` in the demo on iOS through `WebKitEngine` (the check `hack/wasm/wisp.mjs` does under Node).
- **Later:** visionOS (WebKit engine, tier-3 Rust for native), tvOS (native interpreter only: tvOS has no WebKit), guest 3D (`25-gpu-3d.md`) through Metal instead of WebGL2, App Store review notes (the app downloads and runs operating systems the user picks: the same position as UTM SE, which ships interpreted only).

## M3 as built (2026-11-15)

- Both engines draw into a `ScreenSurface` (`apple/App/ScreenSurface.swift`): a CoreGraphics bitmap in the library's XRGB8888, which is CoreGraphics' 32-bit little-endian skip-first layout, so rows are copied as they come. Each refresh copies only the changed rectangles and commits one `CGImage` (a copy-on-write copy of the bitmap) as the view's layer contents, aspect-fit, nearest-neighbour when magnified. A Metal texture fed only the rectangles is the upgrade if a phone shows the copy.
- Native: `ua_screen_update` keeps a `ScreenMirror` in the machine and hands Swift the screen and the rectangles whose pixels really differ (`DisplayHandle::sync_mirror`); `ua_cursor_update` hands the pointer image when its generation moves. Swift polls both at 60 Hz on the main run loop (cheap when nothing changed). Input is `ua_key` (USB HID usage), `ua_pointer_move` (screen pixels), `ua_pointer_button`, `ua_scroll`, `ua_release_all`, `ua_request_size`. Options are the web build's `DISPLAY`, `INPUT` and `SCREEN_CONSOLE` bits.
- WebKit: the bridge's binary messages got a kind byte. Page to app: console output, frames (the worker's `host_frame`: size, rectangles, packed pixels) and cursors; app to page: console keystrokes and input events, which the page writes into the worker's event ring as `main.js`'s `pushEvent` does. The app acknowledges every frame (`frame-ack` sets `FRAME_ACK`), so one frame is in flight and a slow bridge drops frames instead of queueing them.
- Keys: iOS sends `UIKey.keyCode`, which is the HID usage; the software keyboard (`UIKeyInput`) types characters, mapped to US-layout keys with shift. macOS maps `NSEvent.keyCode` (key positions, kVK_*) to HID usages in a table, and modifiers from `flagsChanged` with the device-dependent masks so left and right differ; caps lock is a press and release per toggle; repeats are dropped (the guest repeats).
- Pointer: absolute, through the inverse of the aspect fit. macOS: mouse, right and other buttons, wheel (trackpad points to notches); the guest's pointer image becomes the `NSCursor` (hidden or the host arrow otherwise). iOS: a touch is the pointer plus the left button, hover moves it (iPad trackpad), a two-finger pan scrolls; the guest's pointer image is drawn as a layer where the pointer is.
- Disks: `ua_config.disks` (path, read-only; raw or udisk, overlays refused with a message), and the WebKit engine serves them by byte range from the file (`FileHandle`, never the whole image) to `worker.js`'s synchronous range reads. The app boots an ISO with no kernel through the firmware: "Boot ISO…" in the machine menu, or `-cdrom NAME` (relative to the app's Documents) on the command line, with 2 GiB of RAM.
- The machine menu has the engine, the display (terminal only, screen, screen and terminal: the CLI's `--display serial|framebuffer|both`) and "Resize Guest to Fit"; the toolbar has a Terminal | Screen switch (in the bottom bar on iOS, where the navigation bar has no room).
- Checked: macOS, native JIT, Fedora KDE 44 live ISO: GRUB, the media check over the whole 3.1 GB image (PASS), Plymouth, the console in colour, then the desktop (below). iOS simulator, WebKit engine, Debian 13.7 netinst ISO: its themed GRUB menu, arrow keys from the hardware-key path moving the selection, and `c` from the software keyboard opening GRUB's command line. Slax 12.2 has no UEFI boot image ("No bootable device", also under `hack/wasm/boot.mjs`), so it says nothing about the app.
- Not checked: the pointer by hand (the tests here drive keys, not a mouse), an iPad with a real keyboard and trackpad, and how a phone copes with 4 MB images at 60 Hz.

## M2 as built (2026-11-14)

- `apple/WebEngine/engine.{html,js}` is the page: main.js's job without the UI. It starts `worker.js` unchanged (no compile worker: that only helps V8), writes keystrokes, sizes and the stop flag into the shared control block as main.js does, and passes the worker's output and events on.
- The bridge is one WebSocket per run, page to app on 127.0.0.1 (`Loopback.swift`: an `NWListener` with `NWProtocolWebSocket`). The page's first message is `<token> <run>`; the app drops any socket that does not say the current run. App to page: a JSON `boot` (paths of the kernel, initrd and firmware, `uw_create`'s option bits, memory, size), then `resize` and `stop`; binary frames are keystrokes. Page to app: binary frames are console output (the worker's 16 ms / 64 KiB batches), text is `started`, `exit` (worker.js's codes) and `log`.
- The page, the web build and each run's kernel, initrd and firmware come from a minimal HTTP/1.1 server on 127.0.0.1 (same file), under a random per-launch token in the first path segment, with ranges (worker.js's disk reads) and the COOP/COEP/CORP headers. Disks are passed through to worker.js as URLs on it; the demo has none, so that path is only as tested as M0's range probe.
- `WebKitEngine` loads the page afresh for every start (with the run number) and blanks it on stop, which ends the worker. A cancelled load (stop then start in quick succession) is not a failure; a WebContent crash is reported as the machine stopping. The web view sits 1×1 at opacity 0 behind the terminal.
- `MachineEngine` is the protocol both engines implement; the toolbar has an engine menu (also `-engine native|webkit`, remembered in `UserDefaults`). WebKit is the default on iOS, native on macOS. The chip says `JIT`, `interp` or `wasm`.
- iOS simulator, Alpine demo through the UEFI firmware: the shell in about 9 to 11 s on WebKit against about 50 s on the native interpreter (the same ratio as M0's direct boots). Typing, Stop, Start again and the guest's own `poweroff -f` (reported as Powered off) work. The Mac app runs the WebKit engine too (sandboxed, with `com.apple.security.network.server` for the listeners).
- Not done: disks in the UI, persistence (M4), the screen (M3), a real device.

## M1 as built (2026-11-14)

- `crates/uemulator-apple`: a staticlib on Apple targets (an empty rlib elsewhere, so the workspace still builds on Linux), with the C ABI in `include/uemulator.h` and a module map that makes it the Swift module `UemulatorCore`. M1's surface is the console only: `ua_create` (kernel, initrd, optional UEFI firmware image, command-line tail, memory, vCPUs, the web build's `OPTION_*` bits), `ua_start`, `ua_input`, `ua_resize`, `ua_stop`, `ua_destroy`, and two callbacks, console output on the vCPU thread and the exit reason on a runner thread. `UA_OPTION_JIT` builds the machine with `JitMode::On` and, on `JitUnavailable`, again with `JitMode::Off`, so `ua_jit_active` tells the app which it got. No library change was needed: on iOS `mmap` of RWX memory fails and the probe falls back.
- `apple/Uemulator.xcodeproj`: one multiplatform SwiftUI target (`SDKROOT = auto`, macOS 14 and iOS 17), committed as a hand-written `project.pbxproj` (no generator). Build phases: `hack/apple/build-core.sh --xcode` (cargo for each of Xcode's `ARCHS` on its `PLATFORM_NAME`, lipo'd into `BUILT_PRODUCTS_DIR`, plus the firmware), the Swift sources, and `hack/apple/bundle-app.sh` (the firmware image and the demo's kernel and initramfs from `WEB_DEMO=1 hack/web.sh`). Cargo runs with a scrubbed environment: Xcode's `SDKROOT` and friends break the host build scripts. `hack/apple/build-core.sh xcframework` makes `target/apple/UemulatorCore.xcframework` (macOS universal, iOS, iOS simulator) for embedding elsewhere; the project itself links the per-build archive.
- SwiftTerm (1.99.3, pinned in `Package.resolved`) is the terminal. It runs a build-tool plugin: Xcode asks once to trust it, and command-line builds need `-skipPackagePluginValidation`.
- macOS: hardened runtime and App Sandbox with `com.apple.security.cs.allow-jit`, signed to run locally. The demo boots through the UEFI firmware to its shell prompt in under two seconds on the JIT. iOS simulator: the same app, interpreted, reaches the shell (the simulator refuses RWX memory like a device does).
- The chip says `JIT` or `interpreted`, from `ua_jit_active`. Not yet: `UemulatorKit` as its own package (the engine is in `apple/App` for now), a machine list, settings, disks, the screen.

Build and run from the command line:

    WEB_DEMO=1 hack/web.sh
    xcodebuild -project apple/Uemulator.xcodeproj -scheme Uemulator -destination 'platform=macOS' -skipPackagePluginValidation build
    xcodebuild -project apple/Uemulator.xcodeproj -scheme Uemulator -destination 'generic/platform=iOS Simulator' -skipPackagePluginValidation build

`-autostart` boots the demo at launch.

## M0 results (2026-11-14)

`WEB_DEMO=1 hack/web.sh`, then `hack/apple/m0.sh`: it builds `apple/M0Spike`, runs it on the booted simulator with `-autorun` and prints the probe and boot results. The boot is the Alpine demo (`hack/web/demo.py`), direct (`--firmware none`), 256 MiB, one vCPU, `earlycon=uart8250,io,0x3f8` appended, timed from the boot message to the worker until the console shows the text. The WKWebView rows are four runs of the JIT and two interpreted; a first JIT run at 17.3 s is left out: a stale launch of an earlier build was still booting in the same simulator. iPhone 17 Pro simulator (iOS 27.0 runtime; WebKit's user agent says 18_7), Xcode 27.2, an Apple Silicon Mac; every row ran on that same Mac, one at a time.

| engine | `Run /init` | demo shell |
|---|---|---|
| native CLI, aarch64 JIT (what macOS gets) | 1.3 s | 1.4 s |
| Node 26 / V8, wasm JIT with the compile worker (`boot.mjs`) | 3.6 s | 4.2 s |
| Node 26 / V8, wasm JIT, `--no-compile-worker` | 5.0 s | 5.8 s |
| **WKWebView / JSC, wasm JIT (`WebKitEngine`)** | **6.6 to 7.1 s** | **8.2 to 8.8 s** |
| native CLI, `--disable-jit` (what `NativeEngine` gets on iOS) | 46.0 s | 48.4 s |
| WKWebView / JSC, wasm interpreted | 58.0 to 59.6 s | 61.1 to 62.8 s |
| Node 26 / V8, wasm interpreted | 67.1 s | 70.7 s |

- **Verdict: build `WebKitEngine`.** It boots 6.5× faster than the native interpreter, which is all an App Store build gets otherwise, and within 1.4× of V8 without its compile worker. Against V8 with the worker it is 1.9×, so JSC's per-module cost is what is left to close (above).
- Isolation: no `SharedArrayBuffer` on a custom scheme even with `crossOriginIsolated` true; loopback HTTP has it, and shared wasm memory too. This moved the engine off `WKURLSchemeHandler`.
- Hidden: the web view ran both boots while 1×1 at opacity 0 in the hierarchy. Not tested yet: what happens when the app goes to the background (iOS suspends the app and its WebContent process; that hits `NativeEngine` the same way, and is the app's problem, not the engine's).
- Memory: a worker grew a wasm memory, touching every page, to 2560 MiB before `Out of memory`. On the simulator this is WebKit's own limit, not a device's jetsam limit.
- Still to do on a real iPhone and iPad (needs a signing team): the same table, the memory ceiling under jetsam, and the oldest iOS that has wasm tail calls. The simulator runs macOS's WebKit on the Mac's cores, so its absolute times flatter a phone; the ratios between rows are what carries over.

## Open questions

- Does App Review accept a hidden `WKWebView` used as a WebAssembly compute engine? Guideline 2.5.2 is about downloading executable code. The wasm here is bundled, and the generated modules are made from guest code at run time, which is what the JS engine is for. UTM SE's review history is the closest precedent and should be checked before M2.
- Multiple machines at once on `WebKitEngine`: one web view per machine (one WebContent process each, separate jetsam budgets), or one shared. One per machine is simpler and isolates crashes.
- Whether the library's validation of a machine config (what `main.js` checks before Start) should become a public `MachineConfig::check` or stay per-host.
