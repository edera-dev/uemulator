# Apple platforms: a SwiftUI app

**Status:** proposal. Nothing is built yet; M0 below is the spike that decides whether the WebKit engine is worth building.

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

A `WKWebView` that is never added to a window (on iOS it must sit in the view hierarchy, 1×1 and hidden, or WebKit may suspend its process: check in M0). It loads `uemulator://engine/engine.html` from a `WKURLSchemeHandler` that serves the bundle and the app's files. The shim is `www/worker.js` and `www/jit_host.js` almost unchanged. `main.js`, xterm.js and every piece of UI are dropped.

**Cross-origin isolation.** `uw_run` blocks a Worker, and the page talks to it through `SharedArrayBuffer` rings (`www/protocol.js`). WebKit grants `SharedArrayBuffer` only to a cross-origin-isolated page, so the scheme handler sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every response, as `hack/web/serve.py` does. M0 must confirm that `crossOriginIsolated` is true on a custom scheme. The fallback is serving from a loopback `NWListener` on `http://127.0.0.1` (secure context, headers under our control).

**The bridge.** Everything the page did with UI now crosses a process boundary:

- Disk reads: the worker already reads disks with synchronous range XHRs (`host_disk_read`, `rangeRequest`). Those go to `uemulator://disk/<id>`, and the scheme handler answers them from the file with `FileHandle` reads. Nothing new is needed, and big ISOs never load into memory.
- Disk writes: today they land in an in-memory udisk overlay that dies with the page. For a persistent disk, add a `host_disk_write` import: a synchronous `POST uemulator://disk/<id>?offset=` whose body the scheme handler writes to the udisk file. M0 checks that WebKit hands request bodies from a worker's XHR to a scheme handler. If it doesn't, the overlay is flushed in batches over the stream channel below.
- Console, screen, input, network and Wisp: one binary stream each way between the page's main thread and the app. The page's main thread drains the worker's rings as `main.js` does today and forwards the bytes; going the other way, it fills the input, net and Wisp rings. Two candidate transports, measured in M0:
  1. `WKScriptMessageHandlerWithReply` / `callAsyncJavaScript`. Simple, but its values are property lists (no `ArrayBuffer`), so bytes travel as base64 strings.
  2. A WebSocket from the page to an `NWListener` on loopback in the app. Binary frames, no base64; costs a listener and a random token so another app can't connect.

  The screen is the heavy one: a 1024×768 frame is 3 MiB, 180 MiB/s at 60 Hz. Only the damaged rectangles cross (virtio-gpu flushes come with rects; `host_frame` takes a rect), at most once per display refresh. The app acknowledges each frame the way the page does (`FRAME_ACK`), so a slow bridge drops frames instead of queueing them.
- Network: the guest's NIC stays on the library's `Gateway` + `WispClient`, which run inside the wasm (`27-wisp.md`). The app is the Wisp v1 relay: a small Swift server for the protocol (TCP and UDP streams over `Network.framework`) on the same stream. No external relay, no browser networking. `NativeEngine` uses the library's gateway directly, as the CLI does.
- Lifetime: WebKit can kill WebContent under memory pressure (`webViewWebContentProcessDidTerminate`). The engine reports it as the machine stopping with an error, never as a crash. Guest RAM counts against the WebContent process's jetsam limit, so the RAM picker on this engine is capped by what M0 measures on the smallest supported device (and never above the wasm32 3 GiB in any case).

**What the WebAssembly JIT loses on JavaScriptCore.** Each item is a measurement for M0, not an assumption:

- The compile worker relies on V8's process-wide native module cache (`jit_compiler.js`). JSC has no measured equivalent, so the page already skips it outside V8 (`compileWorkerHelps`), and blocks compile on the emulator's thread. Expect the 38 µs-a-block figure, or JSC's own number.
- Tail calls chain blocks (`return_call_indirect`). Without them the chain stub fails to compile and the JIT runs unchained. That is correct but slower, so M0 notes which iOS versions have them.
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

- **M0 — WebKit spike (decides the plan).** A throwaway iOS app: hidden `WKWebView`, scheme handler with COOP/COEP, the existing `uemulator_web.wasm` and `worker.js`, console bytes to a `Text` view. It answers, on a real iPhone and on the simulator: `crossOriginIsolated` on a custom scheme, whether the web view runs while hidden, tail calls and SIMD on JSC, sync XHR with a body from a worker to a scheme handler, the WebContent memory ceiling, and boot time to `Run /init` for the Alpine demo with the JIT, with `--no-jit` in the same web view, and on `NativeEngine` interpreted. Exit: those numbers written into this file. If `WebKitEngine` doesn't beat the native interpreter by a clear margin, it is dropped, and iOS ships interpreted.
- **M1 — Core and project.** `crates/uemulator-apple`, `hack/apple/build-core.sh`, the xcframework, the Xcode project with a SwiftUI shell, and `NativeEngine` booting the Alpine demo to a shell on macOS (JIT) and iOS (interpreted) in a SwiftTerm view. Exit: ⌘R on both, and `xcodebuild` in CI.
- **M2 — WebKitEngine.** The headless shim, the bridge (winning transport from M0), disk reads through the scheme handler, console and keyboard. Exit: the same Alpine demo on iOS through `WebKitEngine`, with the JIT on.
- **M3 — Screen and input.** Damaged-rect frames on both engines, virtio keyboard and tablet, cursor. Exit: a live ISO's desktop (the KDE ISO from `20-jit-perf.md`) is usable on an iPad with a keyboard and trackpad.
- **M4 — Machines and persistence.** `MachineStore` on the CLI's format, `host_disk_write` (or flushed overlays), NVRAM and TPM state, presets and validation shared with the web page in Rust. Exit: install Alpine to a disk on iOS, quit the app, boot the installed disk.
- **M5 — Network.** The Swift Wisp relay, the in-app segment between machines. Exit: `apk update` in the demo on iOS through `WebKitEngine` (the check `hack/wasm/wisp.mjs` does under Node).
- **Later:** visionOS (WebKit engine, tier-3 Rust for native), tvOS (native interpreter only: tvOS has no WebKit), guest 3D (`25-gpu-3d.md`) through Metal instead of WebGL2, App Store review notes (the app downloads and runs operating systems the user picks: the same position as UTM SE, which ships interpreted only).

## Open questions

- Does App Review accept a hidden `WKWebView` used as a WebAssembly compute engine? Guideline 2.5.2 is about downloading executable code. The wasm here is bundled, and the generated modules are made from guest code at run time, which is what the JS engine is for. UTM SE's review history is the closest precedent and should be checked before M2.
- Multiple machines at once on `WebKitEngine`: one web view per machine (one WebContent process each, separate jetsam budgets), or one shared. One per machine is simpler and isolates crashes.
- Whether the library's validation of a machine config (what `main.js` checks before Start) should become a public `MachineConfig::check` or stay per-host.
