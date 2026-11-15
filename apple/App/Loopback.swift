import Foundation
import Network

// The two loopback servers the WebKit engine needs (docs/plans/28-apple.md, "WebKitEngine").
// Not a WKURLSchemeHandler: WebKit gives a custom scheme no SharedArrayBuffer even when the page
// is cross-origin isolated, and worker.js cannot run without one (M0). http://127.0.0.1 is a
// secure context and gets it.
//
// Both listen on 127.0.0.1 only. Other apps on the device can still connect to a loopback port,
// so both demand a random token picked at launch: the HTTP server in the first path segment, the
// WebSocket server as the first message. Anything else gets a 404 or a closed socket.

struct LoopbackReply {
    var status: Int
    var headers: [String: String]
    var body: Data
}

// What the HTTP server serves under /<token>/: bundled files (the web build and the engine page)
// and blobs a run registers (its kernel, initrd, firmware, disks), all with byte ranges, which is
// how worker.js reads disks.
final class LoopbackFiles: @unchecked Sendable {
    let token = UUID().uuidString.lowercased()
    private let root: URL
    private let lock = NSLock()
    private var blobs: [String: Data] = [:]
    // Files served by range straight from disk (disk images: an ISO is gigabytes).
    private var registeredFiles: [String: URL] = [:]

    init(root: URL) {
        self.root = root
    }

    // `path` without the token, e.g. "run/3/kernel". Replaces what was there.
    func register(_ path: String, _ data: Data) {
        lock.lock()
        blobs[path] = data
        lock.unlock()
    }

    func register(_ path: String, file: URL) {
        lock.lock()
        registeredFiles[path] = file
        lock.unlock()
    }

    func removeAll(prefix: String) {
        lock.lock()
        blobs = blobs.filter { !$0.key.hasPrefix(prefix) }
        registeredFiles = registeredFiles.filter { !$0.key.hasPrefix(prefix) }
        lock.unlock()
    }

    // A byte range of a registered file, read on demand. worker.js always asks for a range: its
    // first request (bytes 0-0) learns the length from Content-Range.
    private func reply(file: URL, range: String?) -> LoopbackReply {
        guard let handle = try? FileHandle(forReadingFrom: file) else { return Self.make(404, Data(), "text/plain") }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd(), size > 0 else { return Self.make(404, Data(), "text/plain") }
        guard let range, let (first, last) = Self.parse(range, size: Int(size)) else {
            // A whole disk image in one reply is never what anyone wants.
            return Self.make(416, Data(), "text/plain")
        }
        guard (try? handle.seek(toOffset: UInt64(first))) != nil,
              let data = try? handle.read(upToCount: last - first + 1)
        else { return Self.make(404, Data(), "text/plain") }
        var reply = Self.make(206, data, "application/octet-stream")
        reply.headers["Content-Range"] = "bytes \(first)-\(first + data.count - 1)/\(size)"
        return reply
    }

    func reply(path: String, range: String?) -> LoopbackReply {
        let prefix = "/\(token)/"
        guard path.hasPrefix(prefix) else { return Self.make(404, Data(), "text/plain") }
        let relative = String(path.dropFirst(prefix.count))
        lock.lock()
        let blob = blobs[relative]
        let registered = registeredFiles[relative]
        lock.unlock()
        if let registered { return reply(file: registered, range: range) }
        let data: Data
        let type: String
        if let blob {
            data = blob
            type = "application/octet-stream"
        } else {
            let file = root.appendingPathComponent(relative).standardizedFileURL
            // No escaping the web folder with "..".
            guard file.path.hasPrefix(root.standardizedFileURL.path + "/"),
                  let mapped = try? Data(contentsOf: file, options: .alwaysMapped)
            else { return Self.make(404, Data(), "text/plain") }
            data = mapped
            type = Self.type(of: file)
        }
        if let range, let (first, last) = Self.parse(range, size: data.count) {
            var reply = Self.make(206, data.subdata(in: first..<(last + 1)), type)
            reply.headers["Content-Range"] = "bytes \(first)-\(last)/\(data.count)"
            return reply
        }
        return Self.make(200, data, type)
    }

    private static func make(_ status: Int, _ body: Data, _ type: String) -> LoopbackReply {
        LoopbackReply(status: status, headers: [
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
        guard header.hasPrefix("bytes="), size > 0 else { return nil }
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

// A listener on 127.0.0.1 with an OS-picked port, which `port()` waits for.
class LoopbackListener: @unchecked Sendable {
    let queue: DispatchQueue
    private var listener: NWListener?
    private var waiting: [CheckedContinuation<UInt16, Never>] = []
    private var boundPort: UInt16?

    init(label: String) {
        queue = DispatchQueue(label: label)
    }

    func start(_ parameters: NWParameters) {
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: .any)
        guard let listener = try? NWListener(using: parameters) else { return }
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            self.accept(connection)
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

    func accept(_ connection: NWConnection) {}
}

// A minimal HTTP/1.1 server: GET with ranges, one request per connection (`Connection: close`).
// Enough for a page, its scripts and worker.js's synchronous range reads.
final class LoopbackHTTPServer: LoopbackListener, @unchecked Sendable {
    private let files: LoopbackFiles
    private static let headEnd = Data("\r\n\r\n".utf8)
    // A request head larger than this is not one of ours.
    private static let maxHead = 64 * 1024

    init(files: LoopbackFiles) {
        self.files = files
        super.init(label: "loopback-http")
        start(.tcp)
    }

    override func accept(_ connection: NWConnection) {
        read(connection, Data())
    }

    private func read(_ connection: NWConnection, _ buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, done, error in
            guard let self, error == nil else { connection.cancel(); return }
            var received = buffered
            if let data { received.append(data) }
            guard let end = received.range(of: Self.headEnd) else {
                if done || received.count > Self.maxHead { connection.cancel() } else { self.read(connection, received) }
                return
            }
            let lines = String(decoding: received[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            var range: String?
            for line in lines.dropFirst() where line.lowercased().hasPrefix("range:") {
                range = line.dropFirst("range:".count).trimmingCharacters(in: .whitespaces)
            }
            let target = lines.first?.split(separator: " ").dropFirst().first.map(String.init) ?? "/"
            let path = URLComponents(string: target)?.path ?? target
            let reply = self.files.reply(path: path, range: range)
            var head = "HTTP/1.1 \(reply.status) \(reply.status < 300 ? "OK" : "Not Found")\r\nConnection: close\r\n"
            for (name, value) in reply.headers { head += "\(name): \(value)\r\n" }
            var bytes = Data((head + "\r\n").utf8)
            bytes.append(reply.body)
            connection.send(content: bytes, isComplete: true, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}

// One page's WebSocket, once it has said the token. Callbacks run on the main queue.
final class BridgeSocket: @unchecked Sendable {
    let run: Int
    private let connection: NWConnection
    var onBinary: ((Data) -> Void)?
    var onText: ((String) -> Void)?

    init(run: Int, connection: NWConnection) {
        self.run = run
        self.connection = connection
    }

    func send(binary: Data) { send(binary, opcode: .binary) }
    func send(text: String) { send(Data(text.utf8), opcode: .text) }

    func close() {
        connection.cancel()
    }

    private func send(_ data: Data, opcode: NWProtocolWebSocket.Opcode) {
        let context = NWConnection.ContentContext(identifier: "bridge",
                                                  metadata: [NWProtocolWebSocket.Metadata(opcode: opcode)])
        connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
    }

    fileprivate func receiveLoop() {
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self, error == nil, let data else { return }
            let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata
            let isText = metadata?.opcode == .text
            DispatchQueue.main.async {
                if isText { self.onText?(String(decoding: data, as: UTF8.self)) } else { self.onBinary?(data) }
            }
            self.receiveLoop()
        }
    }
}

// Accepts the engine page's WebSockets. The first message must be "<token> <run>"; then the socket
// goes to `onSocket` on the main queue.
final class LoopbackSocketServer: LoopbackListener, @unchecked Sendable {
    private let token: String
    var onSocket: ((BridgeSocket) -> Void)?
    // The page's frames are console output, at most the worker's 64 KiB batches, and keystrokes.
    private static let maxMessage = 16 << 20

    init(token: String) {
        self.token = token
        super.init(label: "loopback-socket")
        let parameters = NWParameters.tcp
        let websocket = NWProtocolWebSocket.Options()
        websocket.autoReplyPing = true
        websocket.maximumMessageSize = Self.maxMessage
        parameters.defaultProtocolStack.applicationProtocols.insert(websocket, at: 0)
        start(parameters)
    }

    override func accept(_ connection: NWConnection) {
        connection.receiveMessage { [weak self] data, _, _, error in
            guard let self, error == nil, let data else { connection.cancel(); return }
            let words = String(decoding: data, as: UTF8.self).split(separator: " ")
            guard words.count == 2, words[0] == self.token, let run = Int(words[1]) else {
                connection.cancel()
                return
            }
            let socket = BridgeSocket(run: run, connection: connection)
            DispatchQueue.main.async {
                self.onSocket?(socket)
                socket.receiveLoop()
            }
        }
    }
}
