import Foundation
import Network

// The two loopback servers the WebKit engine needs (docs/plans/28-apple.md, "WebKitEngine").
// Not a WKURLSchemeHandler: WebKit gives a custom scheme no SharedArrayBuffer even when the page
// is cross-origin isolated, and worker.js cannot run without one (docs/plans/28-apple.md,
// "M0 results"). http://127.0.0.1 is a
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
    // Writable ones (a machine's disks), with the handle their writes go through: one per file
    // for the run, not one per write.
    private var writableFiles: [String: FileHandle] = [:]
    // A machine's nvram.kv, which the web build's NVRAM store reads whole and writes back
    // (worker.js `loadNvram`, `host_nvram_*`). The file may not exist until the first write.
    private var keyValueLogs: [String: URL] = [:]
    // A run's snapshots (docs/plans/29-snapshots.md): a suspend's goes into `<file>.tmp` by offset
    // and `finish=1` renames it into place, so a run that dies halfway leaves no torn snapshot; a
    // resume's is read by range like a disk, and `consumed=1` deletes it.
    private var snapshotSinks: [String: URL] = [:]
    private var snapshotSources: Set<String> = []

    init(root: URL) {
        self.root = root
    }

    // `path` without the token, e.g. "run/3/kernel". Replaces what was there.
    func register(_ path: String, _ data: Data) {
        lock.lock()
        blobs[path] = data
        lock.unlock()
    }

    func register(_ path: String, file: URL, writable: Bool = false) {
        lock.lock()
        registeredFiles[path] = file
        if writable, let handle = try? FileHandle(forUpdating: file) { writableFiles[path] = handle }
        lock.unlock()
    }

    func registerSnapshotSink(_ path: String, file: URL) {
        lock.lock()
        snapshotSinks[path] = file
        lock.unlock()
    }

    func registerSnapshotSource(_ path: String, file: URL) {
        lock.lock()
        registeredFiles[path] = file
        snapshotSources.insert(path)
        lock.unlock()
    }

    private func writeSnapshot(_ file: URL, query: [URLQueryItem], body: Data) throws -> LoopbackReply {
        let temporary = file.appendingPathExtension("tmp")
        if query.contains(where: { $0.name == "finish" }) {
            let handle = try FileHandle(forUpdating: temporary)
            try handle.synchronize()
            try handle.close()
            guard rename(temporary.path, file.path) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        } else if let offset = query.first(where: { $0.name == "offset" })?.value.flatMap(UInt64.init) {
            // The first write starts the file afresh: a temporary left by a run that died is junk.
            if offset == 0 { FileManager.default.createFile(atPath: temporary.path, contents: nil) }
            let handle = try FileHandle(forUpdating: temporary)
            defer { try? handle.close() }
            try handle.seek(toOffset: offset)
            try handle.write(contentsOf: body)
        } else {
            return Self.make(400, Data(), "text/plain")
        }
        return Self.make(200, Data(), "text/plain")
    }

    func registerKeyValueLog(_ path: String, file: URL) {
        lock.lock()
        keyValueLogs[path] = file
        lock.unlock()
    }

    // The library's store holds that each write is durable once acknowledged: a guest's
    // SetVariable returned success. So every write is synchronized before the 200, and a
    // replacement goes through a temporary file and a rename, as `FileKeyValueStore` does, so a
    // crash leaves the old log or the new one. Each call opens the file afresh: a handle kept
    // across a replacement would write into the unlinked old file.
    private func writeKeyValueLog(_ file: URL, query: [URLQueryItem], body: Data) throws -> LoopbackReply {
        if query.contains(where: { $0.name == "replace" }) {
            let temporary = file.appendingPathExtension("tmp")
            FileManager.default.createFile(atPath: temporary.path, contents: nil)
            let handle = try FileHandle(forWritingTo: temporary)
            try handle.write(contentsOf: body)
            try handle.synchronize()
            try handle.close()
            guard rename(temporary.path, file.path) == 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
        } else if let offset = query.first(where: { $0.name == "offset" })?.value.flatMap(UInt64.init) {
            if !FileManager.default.fileExists(atPath: file.path) {
                FileManager.default.createFile(atPath: file.path, contents: nil)
            }
            let handle = try FileHandle(forUpdating: file)
            defer { try? handle.close() }
            try handle.seek(toOffset: offset)
            try handle.write(contentsOf: body)
            try handle.synchronize()
        } else {
            return Self.make(400, Data(), "text/plain")
        }
        return Self.make(200, Data(), "text/plain")
    }

    // A POST from worker.js's write-through disks: `offset=N` writes the body there, `flush=1`
    // makes what was written durable. From its NVRAM store: `offset=N`, or `replace=1` with the
    // whole log. 200, or 404 for anything not a writable registered file or log.
    func write(path: String, query: [URLQueryItem], body: Data) -> LoopbackReply {
        let prefix = "/\(token)/"
        guard path.hasPrefix(prefix) else { return Self.make(404, Data(), "text/plain") }
        let relative = String(path.dropFirst(prefix.count))
        // Held across a key-value log's write: a replacement and a write must not interleave.
        lock.lock()
        if let sink = snapshotSinks[relative] {
            defer { lock.unlock() }
            do {
                return try writeSnapshot(sink, query: query, body: body)
            } catch {
                return Self.make(500, Data(error.localizedDescription.utf8), "text/plain")
            }
        }
        if snapshotSources.contains(relative), query.contains(where: { $0.name == "consumed" }),
           let source = registeredFiles[relative] {
            defer { lock.unlock() }
            do {
                try FileManager.default.removeItem(at: source)
                return Self.make(200, Data(), "text/plain")
            } catch {
                return Self.make(500, Data(error.localizedDescription.utf8), "text/plain")
            }
        }
        if let log = keyValueLogs[relative] {
            defer { lock.unlock() }
            do {
                return try writeKeyValueLog(log, query: query, body: body)
            } catch {
                return Self.make(500, Data(error.localizedDescription.utf8), "text/plain")
            }
        }
        let handle = writableFiles[relative]
        lock.unlock()
        guard let handle else { return Self.make(404, Data(), "text/plain") }
        do {
            if query.contains(where: { $0.name == "flush" }) {
                try handle.synchronize()
            } else if let offset = query.first(where: { $0.name == "offset" })?.value.flatMap(UInt64.init) {
                try handle.seek(toOffset: offset)
                try handle.write(contentsOf: body)
            } else {
                return Self.make(400, Data(), "text/plain")
            }
            return Self.make(200, Data(), "text/plain")
        } catch {
            return Self.make(500, Data(error.localizedDescription.utf8), "text/plain")
        }
    }

    // What the guest has written so far goes to storage, for when iOS may end the app next.
    func synchronizeWritable() {
        lock.lock()
        for handle in writableFiles.values { try? handle.synchronize() }
        lock.unlock()
    }

    func removeAll(prefix: String) {
        lock.lock()
        blobs = blobs.filter { !$0.key.hasPrefix(prefix) }
        registeredFiles = registeredFiles.filter { !$0.key.hasPrefix(prefix) }
        keyValueLogs = keyValueLogs.filter { !$0.key.hasPrefix(prefix) }
        snapshotSinks = snapshotSinks.filter { !$0.key.hasPrefix(prefix) }
        snapshotSources = snapshotSources.filter { !$0.hasPrefix(prefix) }
        for (path, handle) in writableFiles where path.hasPrefix(prefix) {
            // What the guest wrote is on disk before the run's files are let go.
            try? handle.synchronize()
            try? handle.close()
        }
        writableFiles = writableFiles.filter { !$0.key.hasPrefix(prefix) }
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
        let keyValueLog = keyValueLogs[relative]
        lock.unlock()
        if let registered { return reply(file: registered, range: range) }
        // Whole: worker.js reads the log in one request. None yet is a 404.
        if let keyValueLog {
            guard let data = try? Data(contentsOf: keyValueLog) else { return Self.make(404, Data(), "text/plain") }
            return Self.make(200, data, "application/octet-stream")
        }
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
//
// iOS may take a listening socket away from a suspended app (TN2277, "Networking and
// Multitasking"): the listener fails, then or when the app runs again. A running engine page has
// the ports baked in (worker.js's disk URLs, the page's bridge and Wisp addresses), so the
// listener comes back on the port it had; a fresh port would strand the run. Everything runs on
// `queue`.
class LoopbackListener: @unchecked Sendable {
    let queue: DispatchQueue
    private var listener: NWListener?
    private var waiting: [CheckedContinuation<UInt16, Never>] = []
    private var boundPort: UInt16?
    private var ready = false
    // Same-port attempts since the listener last failed; a port another process took meanwhile
    // will not come back, and then any port does (new runs work, the running one is lost).
    private var restartAttempts = 0
    private static let restartDelay: TimeInterval = 0.5
    private static let maxSamePortAttempts = 10

    init(label: String) {
        queue = DispatchQueue(label: label)
        queue.async { self.listen() }
    }

    // Called on `queue` for each listener made: one NWParameters is not reused after a failure.
    func makeParameters() -> NWParameters { .tcp }

    private func listen() {
        let parameters = makeParameters()
        let port = boundPort.flatMap(NWEndpoint.Port.init(rawValue:)) ?? .any
        let samePort = boundPort != nil && restartAttempts < Self.maxSamePortAttempts
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: samePort ? port : .any)
        parameters.allowLocalEndpointReuse = true
        guard let listener = try? NWListener(using: parameters) else {
            scheduleRestart()
            return
        }
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            self.accept(connection)
        }
        listener.stateUpdateHandler = { [weak self, weak listener] state in
            guard let self, let listener, listener === self.listener else { return }
            switch state {
            case .ready:
                guard let port = listener.port?.rawValue else { return }
                self.ready = true
                self.restartAttempts = 0
                self.boundPort = port
                for waiter in self.waiting { waiter.resume(returning: port) }
                self.waiting = []
            case .failed, .waiting:
                self.ready = false
                listener.cancel()
                self.listener = nil
                self.scheduleRestart()
            default:
                break
            }
        }
        listener.start(queue: queue)
    }

    private func scheduleRestart() {
        restartAttempts += 1
        queue.asyncAfter(deadline: .now() + Self.restartDelay) { [weak self] in
            guard let self, self.listener == nil else { return }
            self.listen()
        }
    }

    // When the app comes back to the foreground: a listener that iOS took away without telling it
    // (no state change yet) is replaced.
    func revive() {
        queue.async {
            guard let listener = self.listener else { return }
            if case .ready = listener.state { return }
            self.ready = false
            listener.cancel()
            self.listener = nil
            self.listen()
        }
    }

    #if DEBUG
    // What iOS does to a suspended app's listener (`-debug-drop-sockets`, WebKitEngine).
    func debugFail() {
        queue.async {
            guard let listener = self.listener else { return }
            self.ready = false
            listener.cancel()
            self.listener = nil
            self.scheduleRestart()
        }
    }
    #endif

    func port() async -> UInt16 {
        await withCheckedContinuation { continuation in
            queue.async {
                if self.ready, let port = self.boundPort {
                    continuation.resume(returning: port)
                } else {
                    self.waiting.append(continuation)
                }
            }
        }
    }

    func accept(_ connection: NWConnection) {}
}

// A minimal HTTP/1.1 server: GET with ranges and POST with a body, one request per connection
// (`Connection: close`). Enough for a page, its scripts and worker.js's synchronous disk reads and
// writes.
final class LoopbackHTTPServer: LoopbackListener, @unchecked Sendable {
    private let files: LoopbackFiles
    private static let headEnd = Data("\r\n\r\n".utf8)
    // A request head larger than this is not one of ours.
    private static let maxHead = 64 * 1024
    // The largest POST body: a guest disk write (virtio-blk requests are at most a few MiB).
    private static let maxBody = 32 << 20

    init(files: LoopbackFiles) {
        self.files = files
        super.init(label: "loopback-http")
    }

    override func accept(_ connection: NWConnection) {
        read(connection, Data())
    }

    private func read(_ connection: NWConnection, _ buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, error in
            guard let self, error == nil else { connection.cancel(); return }
            var received = buffered
            if let data { received.append(data) }
            guard let end = received.range(of: Self.headEnd) else {
                if done || received.count > Self.maxHead { connection.cancel() } else { self.read(connection, received) }
                return
            }
            let lines = String(decoding: received[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            var headers: [String: String] = [:]
            for line in lines.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
            let request = lines.first?.split(separator: " ") ?? []
            let method = request.first.map(String.init) ?? "GET"
            let target = request.dropFirst().first.map(String.init) ?? "/"
            let components = URLComponents(string: target)
            let path = components?.path ?? target
            let reply: LoopbackReply
            if method == "POST" {
                let length = Int(headers["content-length"] ?? "0") ?? 0
                guard length <= Self.maxBody else { connection.cancel(); return }
                // The body may still be on its way.
                if received.count - end.upperBound < length && !done {
                    self.read(connection, received)
                    return
                }
                let body = received[end.upperBound...].prefix(length)
                reply = self.files.write(path: path, query: components?.queryItems ?? [], body: Data(body))
            } else {
                reply = self.files.reply(path: path, range: headers["range"])
            }
            var head = "HTTP/1.1 \(reply.status) \(reply.status < 300 ? "OK" : "Error")\r\nConnection: close\r\n"
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
    // The connection ended: the page went, or iOS took the socket away while the app was
    // suspended. Once, on the main queue; not for `close()`.
    var onClose: (() -> Void)?
    private var closed = false

    init(run: Int, connection: NWConnection) {
        self.run = run
        self.connection = connection
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled: self?.ended()
            default: break
            }
        }
    }

    // On the connection's queue or the main queue; reports once, on the main queue.
    private func ended() {
        DispatchQueue.main.async {
            guard !self.closed else { return }
            self.closed = true
            self.onClose?()
        }
    }

    func send(binary: Data) { send(binary, opcode: .binary) }
    func send(text: String) { send(Data(text.utf8), opcode: .text) }

    func close() {
        DispatchQueue.main.async { self.closed = true }
        connection.cancel()
    }

    #if DEBUG
    // A socket iOS took away: closed, and reported as such.
    func debugDrop() {
        connection.cancel()
    }
    #endif

    private func send(_ data: Data, opcode: NWProtocolWebSocket.Opcode) {
        let context = NWConnection.ContentContext(identifier: "bridge",
                                                  metadata: [NWProtocolWebSocket.Metadata(opcode: opcode)])
        connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
    }

    fileprivate func receiveLoop() {
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            guard error == nil, let data else {
                self.ended()
                return
            }
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
    }

    override func makeParameters() -> NWParameters {
        let parameters = NWParameters.tcp
        let websocket = NWProtocolWebSocket.Options()
        websocket.autoReplyPing = true
        websocket.maximumMessageSize = Self.maxMessage
        parameters.defaultProtocolStack.applicationProtocols.insert(websocket, at: 0)
        return parameters
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
