import Foundation
import WebKit

// The web build in a hidden WKWebView (docs/plans/28-apple.md, "WebKitEngine"). On iOS the only
// process allowed to JIT is WebKit's WebContent process, and the web build's JIT emits
// WebAssembly, so the whole emulator runs there: this class is a remote control for it.
//
// Each start loads apple/WebEngine/engine.html afresh from the loopback HTTP server, with a run
// number; the page connects back over the loopback WebSocket, says the token and the run, and gets
// the boot settings. Only the current run's socket is listened to, so a page that is still going
// away cannot write into the next run's terminal. Stopping drops the socket and blanks the page,
// which ends the worker with it.
@MainActor
final class WebKitEngine: NSObject, MachineEngine {
    var onOutput: (([UInt8]) -> Void)?
    var onState: ((EngineState) -> Void)?
    var screen: ScreenSurface?

    // Must sit in a window's view hierarchy (1x1, invisible): M0 ran it that way, and a web view in
    // no window may be throttled.
    let webView: WKWebView

    private let files: LoopbackFiles
    private let http: LoopbackHTTPServer
    private let sockets: LoopbackSocketServer
    private var run = 0
    private var socket: BridgeSocket?
    // The boot message for the current run until its page connects.
    private var pendingBoot: String?
    private var active = false
    private var jit = true

    // `www/protocol.js`'s FIRMWARE and OPTION values, which `uw_create` takes.
    private enum Firmware {
        static let none = 0
        static let uefi = 1
    }
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
        0: "Powered off", 1: "Reset", 2: "Triple fault", 3: "Stopped", 4: "Emulator error",
    ]

    override init() {
        files = LoopbackFiles(root: Bundle.main.resourceURL!.appendingPathComponent("web"))
        http = LoopbackHTTPServer(files: files)
        sockets = LoopbackSocketServer(token: files.token)
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 1, height: 1), configuration: configuration)
        super.init()
        webView.navigationDelegate = self
        #if DEBUG
        webView.isInspectable = true
        #endif
        sockets.onSocket = { [weak self] socket in self?.attach(socket) }
    }

    func start(_ settings: MachineSettings) {
        stop()
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
            files.register(path, file: disk.url)
            // The web build keeps a writable disk's writes in memory (a udisk overlay) until M4.
            disks.append(["url": path, "readOnly": disk.readOnly])
        }
        if let firmware = settings.firmware { files.register(prefix + "firmware", firmware) }
        if settings.display.hasScreen { screen?.reset() }
        let boot: [String: Any] = [
            "type": "boot",
            // Relative to the page, which is at /<token>/engine.html.
            "kernel": settings.kernel.isEmpty ? NSNull() : prefix + "kernel",
            "initrd": settings.initrd.isEmpty ? NSNull() : prefix + "initrd",
            "disks": disks,
            "firmware": settings.firmware == nil ? NSNull() : prefix + "firmware",
            "firmwareKind": settings.firmware == nil ? Firmware.none : Firmware.uefi,
            "cmdline": settings.cmdlineAppend,
            "memoryMib": settings.memoryMib,
            "cpus": settings.cpus,
            "options": settings.options,
            "cols": settings.cols,
            "rows": settings.rows,
        ]
        pendingBoot = Self.json(boot)
        Task {
            let httpPort = await http.port()
            let socketPort = await sockets.port()
            // A stop or another start may have come first.
            guard run == self.run, active else { return }
            let page = "http://127.0.0.1:\(httpPort)/\(files.token)/engine.html?socket=\(socketPort)&run=\(run)"
            webView.load(URLRequest(url: URL(string: page)!))
        }
    }

    func stop() {
        guard active else { return }
        active = false
        pendingBoot = nil
        socket?.send(text: Self.json(["type": "stop"]))
        socket?.close()
        socket = nil
        files.removeAll(prefix: "run/")
        // Ends the page and its worker. The next start loads the engine page again.
        webView.loadHTMLString("", baseURL: nil)
        onState?(.stopped("Stopped"))
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
        socket?.send(text: Self.json(["type": "resize", "cols": cols, "rows": rows]))
    }

    private func attach(_ socket: BridgeSocket) {
        guard active, socket.run == run, let boot = pendingBoot else {
            socket.close()
            return
        }
        pendingBoot = nil
        self.socket = socket
        socket.onBinary = { [weak self, weak socket] data in
            guard let self, let socket, socket === self.socket else { return }
            self.receive(data)
        }
        socket.onText = { [weak self, weak socket] text in
            guard let self, let socket, socket === self.socket else { return }
            self.handle(text)
        }
        socket.send(text: boot)
    }

    private func handle(_ text: String) {
        guard let message = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any],
              let type = message["type"] as? String
        else { return }
        switch type {
        case "started":
            // One word for a phone's toolbar: "wasm" is the WebKit engine's JIT.
            onState?(.running(detail: jit ? "wasm" : "wasm interp"))
        case "exit":
            let code = message["code"] as? Int ?? -1
            let detail = message["message"] as? String ?? ""
            var reason = Self.exitNames[code] ?? "Failed"
            if !detail.isEmpty { reason += ": \(detail)" }
            finish(reason)
        case "log":
            #if DEBUG
            print("[engine] \(message["text"] as? String ?? "")")
            #endif
        default:
            break
        }
    }

    // The guest ended by itself (or the page failed): same teardown as stop, its own reason.
    private func finish(_ reason: String) {
        guard active else { return }
        stop()
        onState?(.stopped(reason))
    }

    private static func json(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }
}

extension WebKitEngine: WKNavigationDelegate {
    nonisolated func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        Task { @MainActor in self.finish("WebKit's process ended (memory?)") }
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
