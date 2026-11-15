import Foundation
import Network
import WebKit

// The M0 spike's half of the WebKit engine (docs/plans/28-apple.md): a WKWebView that is never
// shown, serving the bundled web build with the headers that make a page cross-origin isolated,
// and a loopback WebSocket the bridge measurement compares against postMessage. Throwaway code: it
// answers M0's questions and nothing more.
//
// The page is served two ways, and the run probes both:
// - a WKURLSchemeHandler on uemulator://app/. The page says `crossOriginIsolated` is true but has
//   no SharedArrayBuffer (also after a navigation that should swap processes), so worker.js
//   cannot run there;
// - a loopback HTTP server on http://127.0.0.1:<port>/, which WebKit isolates like any site.
@MainActor
final class Engine: NSObject, ObservableObject {
    static let scheme = "uemulator"

    @Published var console = ""
    @Published var log: [String] = []
    @Published var busy = false

    let webView: WKWebView
    private let schemeHandler = SchemeHandler()
    private let http = LoopbackServer(kind: .http)
    private let socket = LoopbackServer(kind: .webSocketEcho)
    private var pageLoaded: CheckedContinuation<Void, Never>?
    // The console view only needs the tail; the boot's markers are matched in the page.
    private static let consoleKeep = 20_000

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.setURLSchemeHandler(schemeHandler, forURLScheme: Engine.scheme)
        let controller = WKUserContentController()
        configuration.userContentController = controller
        webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 1, height: 1), configuration: configuration)
        super.init()
        controller.addScriptMessageHandler(self, contentWorld: .page, name: "spike")
        webView.navigationDelegate = self
        webView.isInspectable = true
        http.start()
        socket.start()
    }

    private func navigate(to address: String) async {
        await withCheckedContinuation { continuation in
            pageLoaded = continuation
            webView.load(URLRequest(url: URL(string: address)!))
        }
    }

    // Calls `window.spike[name](argument)` in the page and waits for its promise. Module scripts
    // may still be running when the navigation finishes, hence the wait for `window.spike`.
    func call(_ name: String, _ argument: [String: Any] = [:]) async -> Any? {
        do {
            return try await webView.callAsyncJavaScript(
                """
                while (!window.spike) await new Promise((resolve) => setTimeout(resolve, 10));
                return await window.spike[name](argument);
                """,
                arguments: ["name": name, "argument": argument],
                contentWorld: .page)
        } catch {
            return "error: \(error)"
        }
    }

    func record(_ line: String) {
        log.append(line)
        // Picked up by `xcrun simctl launch --console-pty` (hack/apple/m0.sh).
        print("M0 \(line)")
    }

    func runAll() async {
        busy = true
        defer { busy = false }
        await navigate(to: "\(Engine.scheme)://app/engine.html")
        record("probes-scheme \(json(await call("probes")))")
        await navigate(to: "http://127.0.0.1:\(await http.port())/engine.html")
        record("probes-http \(json(await call("probes")))")
        record("bridge \(json(await call("bridge", ["port": Int(await socket.port())])))")
        let only = ProcessInfo.processInfo.environment["UEMULATOR_BOOT"]
        for jit in [true, false] where only == nil || only == (jit ? "jit" : "interpreter") {
            console = ""
            record("boot \(json(await call("boot", ["jit": jit])))")
        }
        record("done")
    }

    private func json(_ value: Any?) -> String {
        guard let value, JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        else { return String(describing: value) }
        return String(decoding: data, as: UTF8.self)
    }
}

extension Engine: WKNavigationDelegate {
    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            pageLoaded?.resume()
            pageLoaded = nil
        }
    }

    nonisolated func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        Task { @MainActor in record("webcontent-terminated") }
    }
}

extension Engine: WKScriptMessageHandlerWithReply {
    // Console text, and the postMessage half of the bridge measurement: the reply is the length
    // received, so the page times a full round trip.
    func userContentController(
        _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
    ) async -> (Any?, String?) {
        guard let body = message.body as? [String: Any], let type = body["type"] as? String else {
            return (nil, "bad message")
        }
        switch type {
        case "console":
            console += body["text"] as? String ?? ""
            if console.count > Engine.consoleKeep { console = String(console.suffix(Engine.consoleKeep)) }
            return (0, nil)
        case "frame":
            let encoded = body["data"] as? String ?? ""
            return (Data(base64Encoded: encoded)?.count ?? -1, nil)
        case "log":
            record("page \(body["text"] as? String ?? "")")
            return (0, nil)
        default:
            return (nil, "unknown \(type)")
        }
    }
}

// What both transports serve: the bundle's `web/` folder with ranges (the worker's disk reads),
// and `POST /echo`, which answers with the size of the body it got (can a worker's synchronous
// XHR hand the app a body: the disk-write question).
struct Reply {
    var status: Int
    var headers: [String: String]
    var body: Data
}

enum WebFiles {
    private static let root = Bundle.main.resourceURL!.appendingPathComponent("web")

    static func reply(path: String, range: String?, body: Data?) -> Reply {
        if path == "/echo" {
            return make(200, Data("\(body?.count ?? -1)".utf8), "text/plain")
        }
        let file = root.appendingPathComponent(String(path.dropFirst()))
        guard let data = try? Data(contentsOf: file, options: .alwaysMapped) else {
            return make(404, Data(), "text/plain")
        }
        if let range, let (first, last) = parse(range, size: data.count) {
            var reply = make(206, data.subdata(in: first..<(last + 1)), type(of: file))
            reply.headers["Content-Range"] = "bytes \(first)-\(last)/\(data.count)"
            return reply
        }
        return make(200, data, type(of: file))
    }

    private static func make(_ status: Int, _ body: Data, _ type: String) -> Reply {
        Reply(status: status, headers: [
            "Content-Type": type,
            "Content-Length": "\(body.count)",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            // What hack/web/serve.py sends: without them there is no SharedArrayBuffer.
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Resource-Policy": "same-origin",
        ], body: body)
    }

    private static func parse(_ header: String, size: Int) -> (Int, Int)? {
        guard header.hasPrefix("bytes=") else { return nil }
        let parts = header.dropFirst(6).split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 2, let first = Int(parts[0]) else { return nil }
        let last = min(Int(parts[1]) ?? (size - 1), size - 1)
        return first <= last ? (first, last) : nil
    }

    private static func type(of file: URL) -> String {
        switch file.pathExtension {
        case "html": "text/html"
        case "js", "mjs": "text/javascript"
        case "wasm": "application/wasm"
        case "json": "application/json"
        default: "application/octet-stream"
        }
    }
}

final class SchemeHandler: NSObject, WKURLSchemeHandler {
    func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
        let request = task.request
        guard let url = request.url else { return }
        var body = request.httpBody
        if body == nil, let stream = request.httpBodyStream { body = Self.drain(stream) }
        let reply = WebFiles.reply(path: url.path, range: request.value(forHTTPHeaderField: "Range"), body: body)
        task.didReceive(HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: "HTTP/1.1",
                                        headerFields: reply.headers)!)
        task.didReceive(reply.body)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {}

    private static func drain(_ stream: InputStream) -> Data {
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 1 << 16)
        while true {
            let read = stream.read(&buffer, maxLength: buffer.count)
            if read <= 0 { return data }
            data.append(buffer, count: read)
        }
    }
}

// Listens on 127.0.0.1 only. `.http` is a minimal HTTP/1.1 server for WebFiles (one request per
// connection, `Connection: close`: enough for a spike, and for synchronous XHR). `.webSocketEcho`
// answers every binary message with its size, so the page times the same round trip it times over
// postMessage.
final class LoopbackServer: @unchecked Sendable {
    enum Kind { case http, webSocketEcho }

    private let kind: Kind
    private var listener: NWListener?
    private var waiting: [CheckedContinuation<UInt16, Never>] = []
    private var boundPort: UInt16?
    private let queue: DispatchQueue
    private static let headEnd = Data("\r\n\r\n".utf8)

    init(kind: Kind) {
        self.kind = kind
        queue = DispatchQueue(label: "loopback-\(kind)")
    }

    func start() {
        let parameters = NWParameters.tcp
        if kind == .webSocketEcho {
            let websocket = NWProtocolWebSocket.Options()
            websocket.autoReplyPing = true
            websocket.maximumMessageSize = 64 << 20
            parameters.defaultProtocolStack.applicationProtocols.insert(websocket, at: 0)
        }
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: .any)
        guard let listener = try? NWListener(using: parameters) else { return }
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            if self.kind == .http { self.readRequest(connection, Data()) } else { self.echo(connection) }
        }
        listener.stateUpdateHandler = { [weak self] state in
            guard let self, case .ready = state, let port = listener.port?.rawValue else { return }
            self.boundPort = port
            for waiter in self.waiting { waiter.resume(returning: port) }
            self.waiting = []
        }
        listener.start(queue: queue)
    }

    func port() async -> UInt16 {
        await withCheckedContinuation { continuation in
            queue.async {
                if let port = self.boundPort { continuation.resume(returning: port) } else { self.waiting.append(continuation) }
            }
        }
    }

    private func echo(_ connection: NWConnection) {
        connection.receiveMessage { [weak self] data, _, _, error in
            guard error == nil else { connection.cancel(); return }
            let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
            let context = NWConnection.ContentContext(identifier: "echo", metadata: [metadata])
            connection.send(content: Data("\(data?.count ?? 0)".utf8), contentContext: context, isComplete: true,
                            completion: .contentProcessed { _ in })
            self?.echo(connection)
        }
    }

    private func readRequest(_ connection: NWConnection, _ buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, error in
            guard let self, error == nil else { connection.cancel(); return }
            var received = buffered
            if let data { received.append(data) }
            guard let end = received.range(of: Self.headEnd) else {
                if done { connection.cancel() } else { self.readRequest(connection, received) }
                return
            }
            let head = String(decoding: received[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            var headers: [String: String] = [:]
            for line in head.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
            let length = Int(headers["content-length"] ?? "0") ?? 0
            let body = received[end.upperBound...]
            if body.count < length && !done {
                self.readRequest(connection, received)
                return
            }
            let target = head.first?.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
            let path = URLComponents(string: target)?.path ?? target
            let reply = WebFiles.reply(path: path, range: headers["range"], body: length > 0 ? Data(body) : nil)
            var response = "HTTP/1.1 \(reply.status) \(reply.status < 300 ? "OK" : "Error")\r\nConnection: close\r\n"
            for (name, value) in reply.headers { response += "\(name): \(value)\r\n" }
            var bytes = Data((response + "\r\n").utf8)
            bytes.append(reply.body)
            connection.send(content: bytes, isComplete: true, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
