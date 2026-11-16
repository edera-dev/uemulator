import Foundation
import WebKit
#if os(iOS)
import UIKit
#endif

// The web build in a hidden WKWebView (docs/plans/28-apple.md, "WebKitEngine"). On iOS the only
// process allowed to JIT is WebKit's WebContent process, and the web build's JIT emits
// WebAssembly, so the whole emulator runs there: this class is a remote control for it.
//
// Each start loads apple/WebEngine/engine.html afresh from the loopback HTTP server, with a run
// number; the page connects back over the loopback WebSocket, says the token and the run, and gets
// the boot settings. Only the current run's socket is listened to, so a page that is still going
// away cannot write into the next run's terminal. Stopping drops the socket and blanks the page,
// which ends the worker with it.
//
// Backgrounding (iOS): the app gets a background task's grace period, the machine keeps running
// through it, and then the app, the WebContent process and the worker are suspended together.
// The run survives that as long as iOS keeps the WebContent process: sockets iOS takes away are
// reopened on the same ports (LoopbackListener), the page reconnects (engine.js) and its disk
// requests wait (worker.js `sendSync`). If iOS ends the WebContent process (memory), the machine is
// gone; the guest's disks hold what it last flushed, as after a power cut.
@MainActor
final class WebKitEngine: NSObject, MachineEngine {
    var onOutput: (([UInt8]) -> Void)?
    var onState: ((EngineState) -> Void)?
    var onLog: ((String) -> Void)?
    var screen: ScreenSurface?

    // Must sit in a window's view hierarchy (1x1, invisible): the measurements were taken that way,
    // and a web view in no window may be throttled.
    let webView: WKWebView

    private let files: LoopbackFiles
    private let http: LoopbackHTTPServer
    private let sockets: LoopbackSocketServer
    // The guest's internet: the page's Wisp WebSocket comes here (Wisp.swift).
    private let wisp: WispSocketServer
    private var run = 0
    private var socket: BridgeSocket?
    // The boot message for the current run until its page connects.
    private var pendingBoot: String?
    private var active = false
    // A stopped run's socket while its worker closes the machine (`stop`).
    private var draining: BridgeSocket?
    private var drainTimer: Timer?
    private var startAfterDrain: MachineSettings?
    // How long a stopping worker gets to close its disks before the page is blanked anyway.
    private static let drainTimeout: TimeInterval = 5
    private var jit = true
    // The terminal size last asked for: sent again to a page that reconnects.
    private var size: (cols: Int, rows: Int)?
    // The current run's page lost its socket and has this long, counted while the app is in the
    // foreground, to come back (engine.js retries every second).
    private var reconnectTimer: Timer?
    private static let reconnectGrace: TimeInterval = 20
    private var inBackground = false
    // When the app last came back from the background: a WebContent process that ended in the
    // background is often only reported as the app returns.
    private var foregroundedAt: Date?
    private static let backgroundBlame: TimeInterval = 10
    #if os(iOS)
    private var backgroundTask = UIBackgroundTaskIdentifier.invalid
    #endif
    // Whether the wasm JIT's blocks may tail-call each other (`return_call_indirect` through the
    // chain stubs). JavaScriptCore on an iPad Pro (M2, iPadOS 27.2.1) compiles the stubs and then
    // crashes the WebContent process the first time a block chains, before the firmware prints;
    // no probe survives that, so devices go unchained (docs/plans/28-apple.md, "WebKitEngine on
    // devices"). The simulator and macOS ran the chained JIT; turn it back on per OS once a
    // device runs it.
    #if os(macOS) || targetEnvironment(simulator)
    private static let jitChaining = true
    #else
    private static let jitChaining = false
    #endif

    // The bridge's binary kinds (apple/WebEngine/engine.js has the layout).
    private enum ToPage {
        static let keys: UInt8 = 1
        static let events: UInt8 = 2
    }
    private enum ToApp {
        static let output: UInt8 = 1
        static let frame: UInt8 = 2
        static let cursor: UInt8 = 3
    }
    // `www/protocol.js` EVENT kinds, written into the worker's event ring by the page.
    private enum Event {
        static let key: Int32 = 1
        static let pointerMove: Int32 = 2
        static let pointerButton: Int32 = 3
        static let wheel: Int32 = 4
        static let releaseAll: Int32 = 5
        static let requestSize: Int32 = 6
    }
    // `www/protocol.js` CURSOR kinds.
    private enum CursorKind {
        static let hidden: UInt32 = 1
        static let image: UInt32 = 2
    }
    private static let frameHeaderWords = 3
    private static let rectWords = 4
    private static let cursorHeaderWords = 5
    private static let bytesPerPixel = 4
    private static let wordBytes = MemoryLayout<UInt32>.size
    // worker.js's exit codes (protocol.js EXIT_NAMES), and the page's own for setup failures.
    private static let exitNames = [
        0: "Powered off", 1: EngineState.resetReason, 2: "Triple fault", 3: "Stopped", 4: "Emulator error",
    ]

    override init() {
        files = LoopbackFiles(root: Bundle.main.resourceURL!.appendingPathComponent("web"))
        http = LoopbackHTTPServer(files: files)
        sockets = LoopbackSocketServer(token: files.token)
        wisp = WispSocketServer(token: files.token)
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 1, height: 1), configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        #if DEBUG
        webView.isInspectable = true
        #endif
        sockets.onSocket = { [weak self] socket in self?.attach(socket) }
        #if os(iOS)
        let center = NotificationCenter.default
        center.addObserver(self, selector: #selector(didEnterBackground),
                           name: UIApplication.didEnterBackgroundNotification, object: nil)
        center.addObserver(self, selector: #selector(willEnterForeground),
                           name: UIApplication.willEnterForegroundNotification, object: nil)
        #endif
    }

    #if os(iOS)
    // The machine runs on through the grace period iOS gives a background task (about 30
    // seconds). When it runs out, the guest's writes so far are made durable and the app lets
    // itself be suspended; stopping the machine instead would end an install just as surely.
    @objc private func didEnterBackground() {
        inBackground = true
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        guard active || draining != nil, backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "uemulator machine") { [weak self] in
            MainActor.assumeIsolated { self?.endBackgroundTask() }
        }
    }

    @objc private func willEnterForeground() {
        inBackground = false
        foregroundedAt = Date()
        endBackgroundTask()
        for listener in [http, sockets, wisp] as [LoopbackListener] { listener.revive() }
        if active, pendingBoot == nil, socket == nil { awaitReconnect() }
    }

    private func endBackgroundTask() {
        files.synchronizeWritable()
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }
    #endif

    // The page's socket is gone (BridgeSocket.onClose) or was never back after a suspension:
    // give it `reconnectGrace` in the foreground before calling the machine lost.
    private func awaitReconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        guard !inBackground else { return }
        let timer = Timer(timeInterval: Self.reconnectGrace, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.active, self.socket == nil else { return }
                self.finish("Lost the engine page after the app was in the background")
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        reconnectTimer = timer
    }

    func start(_ settings: MachineSettings) {
        stop()
        // The previous machine is still closing its disks: this one loads the page after it.
        if draining != nil {
            startAfterDrain = settings
            onState?(.starting)
            return
        }
        begin(settings)
    }

    private func begin(_ settings: MachineSettings) {
        run += 1
        let run = run
        active = true
        jit = settings.jit
        onState?(.starting)
        let prefix = "run/\(run)/"
        files.register(prefix + "kernel", settings.kernel)
        files.register(prefix + "initrd", settings.initrd)
        var disks: [[String: Any]] = []
        for (index, disk) in settings.disks.enumerated() {
            let path = prefix + "disk\(index)"
            // A writable disk takes the guest's writes through the loopback server into the file
            // (worker.js's write-through mode), as the native engine writes it directly.
            files.register(path, file: disk.url, writable: !disk.readOnly)
            disks.append(["url": path, "readOnly": disk.readOnly])
        }
        let firmware = settings.firmware == .uefi ? settings.firmwareImage : nil
        if let firmware { files.register(prefix + "firmware", firmware) }
        if let hypervisor = settings.hypervisor { files.register(prefix + "hypervisor", hypervisor) }
        if settings.display.hasScreen { screen?.reset() }
        let boot: [String: Any] = [
            "type": "boot",
            // Relative to the page, which is at /<token>/engine.html.
            "kernel": settings.kernel.isEmpty ? NSNull() : prefix + "kernel",
            "initrd": settings.initrd.isEmpty ? NSNull() : prefix + "initrd",
            "disks": disks,
            "firmware": firmware == nil ? NSNull() : prefix + "firmware",
            "firmwareKind": settings.firmware.code,
            "hypervisor": settings.hypervisor == nil ? NSNull() : prefix + "hypervisor",
            "hypervisorCmdline": settings.hypervisorCmdline,
            "cmdline": settings.cmdlineAppend,
            "memoryMib": settings.memoryMib,
            "cpus": settings.cpus,
            "options": settings.options,
            "jitChaining": Self.jitChaining,
            "network": settings.network,
            "cols": settings.cols,
            "rows": settings.rows,
        ]
        pendingBoot = Self.json(boot)
        Task {
            let httpPort = await http.port()
            let socketPort = await sockets.port()
            let wispPort = await wisp.port()
            // A stop or another start may have come first.
            guard run == self.run, active else { return }
            let page = "http://127.0.0.1:\(httpPort)/\(files.token)/engine.html?socket=\(socketPort)&wisp=\(wispPort)&run=\(run)"
            webView.load(URLRequest(url: URL(string: page)!))
        }
    }

    // Stopping asks the worker to stop and lets it finish: `uw_run` returns, drops the machine,
    // and a udisk writes its map then (map updates wait for a flush or the close,
    // docs/spec/uemulator-disk-format.md). Blanking the page at once would kill the worker first
    // and lose whatever the guest wrote since its last flush. The UI sees Stopped immediately; the
    // page goes when the worker says it exited, or after `drainTimeout`.
    func stop() {
        startAfterDrain = nil
        guard active else { return }
        active = false
        pendingBoot = nil
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        if let socket {
            socket.send(text: Self.json(["type": "stop"]))
            draining = socket
            self.socket = nil
            let timer = Timer(timeInterval: Self.drainTimeout, repeats: false) { [weak self] _ in
                MainActor.assumeIsolated { self?.finishDrain() }
            }
            RunLoop.main.add(timer, forMode: .common)
            drainTimer = timer
        } else {
            finishDrain()
        }
        onState?(.stopped("Stopped"))
    }

    private func finishDrain() {
        #if os(iOS)
        // Nothing left to protect in the background; the disks were synchronized just below.
        defer { if !active { endBackgroundTask() } }
        #endif
        drainTimer?.invalidate()
        drainTimer = nil
        draining?.close()
        draining = nil
        // Synchronizes and closes the run's writable disks.
        files.removeAll(prefix: "run/")
        // Ends the page and its worker. The next start loads the engine page again.
        webView.loadHTMLString("", baseURL: nil)
        if let settings = startAfterDrain {
            startAfterDrain = nil
            begin(settings)
        }
    }

    func send(_ bytes: [UInt8]) {
        socket?.send(binary: Data([ToPage.keys] + bytes))
    }

    func requestSize(width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        sendEvent(Event.requestSize, Int32(clamping: width), Int32(clamping: height))
    }

    // ScreenInput: events for the worker's ring (protocol.js EVENT), one message each. A
    // pointer move per mouse event is small and the WebSocket keeps them in order.
    func key(usage: UInt16, pressed: Bool) {
        sendEvent(Event.key, Int32(usage), pressed ? 1 : 0)
    }

    func pointerMove(x: Int, y: Int) {
        sendEvent(Event.pointerMove, Int32(clamping: x), Int32(clamping: y))
    }

    func pointerButton(_ button: Int, pressed: Bool) {
        sendEvent(Event.pointerButton, Int32(clamping: button), pressed ? 1 : 0)
    }

    func scroll(clicks: Int) {
        sendEvent(Event.wheel, Int32(clamping: clicks), 0)
    }

    func releaseAll() {
        sendEvent(Event.releaseAll, 0, 0)
    }

    private func sendEvent(_ kind: Int32, _ first: Int32, _ second: Int32) {
        guard let socket else { return }
        var bytes = Data([ToPage.events])
        for word in [kind, first, second] {
            withUnsafeBytes(of: word.littleEndian) { bytes.append(contentsOf: $0) }
        }
        socket.send(binary: bytes)
    }

    private func receive(_ data: Data) {
        guard let kind = data.first else { return }
        let body = data.dropFirst()
        switch kind {
        case ToApp.output: onOutput?([UInt8](body))
        case ToApp.frame: drawFrame(body)
        case ToApp.cursor: setCursor(body)
        default: break
        }
    }

    // Data slices keep their parent's indices; read relative to `startIndex`, unaligned.
    private static func word(_ data: Data, _ index: Int) -> Int {
        let offset = data.startIndex + index * wordBytes
        return Int(data[offset..<offset + wordBytes].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
    }

    private func drawFrame(_ body: Data) {
        defer {
            // Drawn (or dropped): the worker may send the next one. One frame in flight, never a
            // backlog, as on the web page.
            socket?.send(text: Self.json(["type": "frame-ack"]))
        }
        guard let screen, body.count >= Self.frameHeaderWords * Self.wordBytes else { return }
        let width = Self.word(body, 0)
        let height = Self.word(body, 1)
        let rectCount = Self.word(body, 2)
        let rectsEnd = (Self.frameHeaderWords + rectCount * Self.rectWords) * Self.wordBytes
        guard body.count >= rectsEnd else { return }
        screen.resize(width: width, height: height)
        var pixelOffset = rectsEnd
        body.withUnsafeBytes { raw in
            for index in 0..<rectCount {
                let base = Self.frameHeaderWords + index * Self.rectWords
                let rect = ScreenSurface.Rect(x: Self.word(body, base), y: Self.word(body, base + 1),
                                              width: Self.word(body, base + 2), height: Self.word(body, base + 3))
                let bytes = rect.width * rect.height * Self.bytesPerPixel
                guard pixelOffset + bytes <= raw.count, let start = raw.baseAddress else { return }
                screen.copy(rect, from: start + pixelOffset, sourceStride: rect.width * Self.bytesPerPixel)
                pixelOffset += bytes
            }
        }
        screen.commit()
    }

    private func setCursor(_ body: Data) {
        guard let screen, body.count >= Self.cursorHeaderWords * Self.wordBytes else { return }
        switch UInt32(Self.word(body, 0)) {
        case CursorKind.hidden: screen.setPointerHidden()
        case CursorKind.image:
            screen.setPointer(width: Self.word(body, 1), height: Self.word(body, 2), hotX: Self.word(body, 3),
                              hotY: Self.word(body, 4), pixels: Data(body.dropFirst(Self.cursorHeaderWords * Self.wordBytes)))
        default: screen.setPointerHost()
        }
    }

    func resize(cols: Int, rows: Int) {
        guard cols > 0, rows > 0 else { return }
        size = (cols, rows)
        socket?.send(text: Self.json(["type": "resize", "cols": cols, "rows": rows]))
    }

    // A page's socket: the run's first, which gets the boot message, or one from a page that
    // reconnected (engine.js) to its running or draining run, which carries on where it was.
    private func attach(_ socket: BridgeSocket) {
        guard socket.run == run else {
            socket.close()
            return
        }
        if active, let boot = pendingBoot {
            pendingBoot = nil
            listen(to: socket)
            socket.send(text: boot)
        } else if active {
            self.socket?.close()
            reconnectTimer?.invalidate()
            reconnectTimer = nil
            listen(to: socket)
            socket.send(text: Self.json(["type": "reattached"]))
            if let size { socket.send(text: Self.json(["type": "resize", "cols": size.cols, "rows": size.rows])) }
            onLog?("the engine page reconnected")
        } else if draining != nil {
            // Its stop may have been lost with the old socket.
            draining?.close()
            draining = socket
            listen(to: socket)
            socket.send(text: Self.json(["type": "stop"]))
        } else {
            socket.close()
        }
    }

    private func listen(to socket: BridgeSocket) {
        if active { self.socket = socket }
        socket.onBinary = { [weak self, weak socket] data in
            guard let self, let socket, socket === self.socket else { return }
            self.receive(data)
        }
        socket.onText = { [weak self, weak socket] text in
            guard let self, let socket else { return }
            if socket === self.draining {
                // Any exit, the stop asked for or not: the machine is gone and its disks closed.
                if text.contains("\"exit\"") { self.finishDrain() }
                return
            }
            guard socket === self.socket else { return }
            self.handle(text)
        }
        socket.onClose = { [weak self, weak socket] in
            guard let self, let socket, socket === self.socket else { return }
            self.socket = nil
            self.onLog?("the engine page's socket closed; waiting for it to reconnect")
            self.awaitReconnect()
        }
    }

    private func handle(_ text: String) {
        guard let message = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any],
              let type = message["type"] as? String
        else { return }
        switch type {
        case "started":
            // One word for a phone's toolbar: "wasm" is the WebKit engine's JIT.
            onState?(.running(detail: jit ? "wasm" : "wasm interp"))
            #if DEBUG
            scheduleDebugDrop()
            #endif
        case "exit":
            let code = message["code"] as? Int ?? -1
            let detail = message["message"] as? String ?? ""
            var reason = Self.exitNames[code] ?? "Failed"
            if !detail.isEmpty { reason += ": \(detail)" }
            finish(reason)
        case "log":
            onLog?(message["text"] as? String ?? "")
        default:
            break
        }
    }

    // The guest ended by itself, or the page failed: `uw_run` has returned (its disks are closed)
    // or never will, so there is nothing to wait for.
    private func finish(_ reason: String) {
        guard active else { return }
        active = false
        pendingBoot = nil
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        socket?.close()
        socket = nil
        finishDrain()
        onState?(.stopped(reason))
    }

    #if DEBUG
    // `-debug-drop-sockets SECONDS`: that long after the machine starts, every loopback listener
    // fails and the bridge drops, as when iOS reclaims a suspended app's sockets. The simulator
    // never does that by itself. The machine should carry on.
    private func scheduleDebugDrop() {
        let delay = UserDefaults.standard.double(forKey: "debug-drop-sockets")
        guard delay > 0 else { return }
        let run = run
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.active, self.run == run else { return }
            self.onLog?("debug: dropping the loopback sockets")
            for listener in [self.http, self.sockets, self.wisp] as [LoopbackListener] { listener.debugFail() }
            self.socket?.debugDrop()
        }
    }
    #endif

    private static func json(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }
}

extension WebKitEngine: WKNavigationDelegate {
    nonisolated func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        Task { @MainActor in
            // iOS reclaims a suspended app's WebContent process for memory, and says so only once
            // the app runs again.
            let inBackground = self.inBackground
                || self.foregroundedAt.map { Date().timeIntervalSince($0) < Self.backgroundBlame } ?? false
            self.finish(inBackground
                ? "iOS ended the machine while the app was in the background"
                : "WebKit's process ended (memory?)")
        }
    }

    nonisolated func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                             withError error: any Error) {
        // A load replaced by the next one (stop's blank page, then a start's engine page) ends as
        // cancelled; that is not this run failing.
        if (error as NSError).domain == NSURLErrorDomain, (error as NSError).code == NSURLErrorCancelled { return }
        let text = error.localizedDescription
        Task { @MainActor in self.finish("The engine page did not load: \(text)") }
    }
}
