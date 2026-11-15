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
    private enum Option {
        static let jit = 1 << 0
        static let acpi = 1 << 1
        static let rng = 1 << 3
        static let serialConsole = 1 << 4
    }
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
        if let firmware = settings.firmware { files.register(prefix + "firmware", firmware) }
        var options = Option.acpi | Option.rng
        if settings.jit { options |= Option.jit }
        if settings.serialConsole { options |= Option.serialConsole }
        let boot: [String: Any] = [
            "type": "boot",
            // Relative to the page, which is at /<token>/engine.html.
            "kernel": prefix + "kernel",
            "initrd": prefix + "initrd",
            "firmware": settings.firmware == nil ? NSNull() : prefix + "firmware",
            "firmwareKind": settings.firmware == nil ? Firmware.none : Firmware.uefi,
            "cmdline": settings.cmdlineAppend,
            "memoryMib": settings.memoryMib,
            "cpus": settings.cpus,
            "options": options,
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
        socket?.send(binary: Data(bytes))
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
            self.onOutput?([UInt8](data))
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
