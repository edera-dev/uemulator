import Foundation
import Network

// The app's Wisp v1 server (docs/plans/28-apple.md): the guest's TCP and UDP streams, which
// the library's `WispClient` sends as Wisp packets, opened as Network.framework connections. Both
// engines use it: the WebKit engine's page connects over a loopback WebSocket
// (`WispSocketServer`), the native engine hands packets in directly (NativeEngine).
//
// The behaviour follows `hack/web/serve.py`'s server, which the library's client is tested
// against: CONTINUE on stream 0 opens the session with `credit` packets; a TCP stream gets that
// credit back after every `refillEvery` DATA packets the destination has taken (so nothing piles
// up here); DATA that arrives before the connection is ready waits for it, in order; a client
// CLOSE is graceful (queued data is written, then the connection ends); a destination that
// ends or fails gets the client a CLOSE with the reason.
final class WispSession: @unchecked Sendable {
    private enum Packet {
        static let connect: UInt8 = 0x01
        static let data: UInt8 = 0x02
        static let `continue`: UInt8 = 0x03
        static let close: UInt8 = 0x04
    }
    private enum StreamType {
        static let tcp: UInt8 = 0x01
        static let udp: UInt8 = 0x02
    }
    // Close reasons (the Wisp v1 spec; serve.py's CLOSE_*).
    private enum Reason {
        static let voluntary: UInt8 = 0x02
        static let unexpected: UInt8 = 0x03
        static let invalidInfo: UInt8 = 0x41
        static let unreachable: UInt8 = 0x42
        static let timeout: UInt8 = 0x43
        static let refused: UInt8 = 0x44
    }
    private static let headerBytes = 5
    // CONNECT's payload: stream type u8, port u16 little-endian, then the host.
    private static let connectFixedBytes = 3
    private static let connectionStream: UInt32 = 0
    private static let credit: UInt32 = 128
    private static let refillEvery = 64
    // A destination read: under the client's record limit (`WISP_RECORD_MAX`, 256 KiB).
    private static let readChunk = 64 * 1024
    private static let connectTimeout: DispatchTimeInterval = .seconds(10)

    private final class Stream {
        let id: UInt32
        let kind: UInt8
        let connection: NWConnection
        var ready = false
        // DATA that came before the connection was ready, and whether a CLOSE came after it.
        var pending: [Data] = []
        var closeAfterPending = false
        var written = 0

        init(id: UInt32, kind: UInt8, connection: NWConnection) {
            self.id = id
            self.kind = kind
            self.connection = connection
        }
    }

    private let queue = DispatchQueue(label: "wisp-session")
    private let sendToClient: (Data) -> Void
    private var streams: [UInt32: Stream] = [:]
    private var closed = false

    // `send` gets each packet for the client, on the session's queue, in order.
    init(send: @escaping (Data) -> Void) {
        sendToClient = send
    }

    func start() {
        queue.async { self.sendPacket(Packet.continue, Self.connectionStream, Self.littleEndian(Self.credit)) }
    }

    // A message from the client: exactly one packet.
    func receive(_ message: Data) {
        queue.async { self.handle(message) }
    }

    func close() {
        queue.async {
            self.closed = true
            for stream in self.streams.values { stream.connection.cancel() }
            self.streams.removeAll()
        }
    }

    private static func littleEndian(_ value: UInt32) -> Data {
        withUnsafeBytes(of: value.littleEndian) { Data($0) }
    }

    private func sendPacket(_ type: UInt8, _ stream: UInt32, _ payload: Data = Data()) {
        guard !closed else { return }
        var packet = Data([type])
        packet.append(Self.littleEndian(stream))
        packet.append(payload)
        sendToClient(packet)
    }

    private func handle(_ message: Data) {
        guard !closed, message.count >= Self.headerBytes else { return }
        let bytes = [UInt8](message)
        let type = bytes[0]
        let id = UInt32(bytes[1]) | UInt32(bytes[2]) << 8 | UInt32(bytes[3]) << 16 | UInt32(bytes[4]) << 24
        let payload = Data(bytes[Self.headerBytes...])
        switch type {
        case Packet.connect: connect(id, payload)
        case Packet.data: forward(id, payload)
        case Packet.close: drop(id)
        // A client's CONTINUE means nothing: the credit towards a destination is its socket's.
        default: break
        }
    }

    private func connect(_ id: UInt32, _ payload: Data) {
        // A CONNECT on a live stream id replaces the stream, silently.
        if let old = streams.removeValue(forKey: id) { old.connection.cancel() }
        let bytes = [UInt8](payload)
        guard bytes.count >= Self.connectFixedBytes, bytes[0] == StreamType.tcp || bytes[0] == StreamType.udp,
              let host = String(bytes: bytes[Self.connectFixedBytes...], encoding: .utf8), !host.isEmpty,
              let port = NWEndpoint.Port(rawValue: UInt16(bytes[1]) | UInt16(bytes[2]) << 8)
        else {
            sendPacket(Packet.close, id, Data([Reason.invalidInfo]))
            return
        }
        let kind = bytes[0]
        let connection = NWConnection(host: NWEndpoint.Host(host), port: port, using: kind == StreamType.tcp ? .tcp : .udp)
        let stream = Stream(id: id, kind: kind, connection: connection)
        streams[id] = stream
        connection.stateUpdateHandler = { [weak self, weak stream] state in
            guard let self, let stream else { return }
            switch state {
            case .ready: self.ready(stream)
            // Waiting is how Network.framework reports an unreachable or refusing destination;
            // a Wisp client wants an answer, not a retry later.
            case .waiting(let error), .failed(let error): self.finish(stream, Self.reason(for: error))
            default: break
            }
        }
        connection.start(queue: queue)
        queue.asyncAfter(deadline: .now() + Self.connectTimeout) { [weak self, weak stream] in
            guard let self, let stream, !stream.ready else { return }
            self.finish(stream, Reason.timeout)
        }
    }

    private static func reason(for error: NWError) -> UInt8 {
        switch error {
        case .dns: Reason.unreachable
        case .posix(let code) where code == .ECONNREFUSED: Reason.refused
        case .posix(let code) where code == .ETIMEDOUT: Reason.timeout
        case .posix(let code) where code == .EHOSTUNREACH || code == .ENETUNREACH: Reason.unreachable
        default: Reason.unexpected
        }
    }

    private func ready(_ stream: Stream) {
        guard streams[stream.id] === stream, !stream.ready else { return }
        stream.ready = true
        for data in stream.pending { write(stream, data) }
        stream.pending = []
        if stream.closeAfterPending { end(stream) }
        read(stream)
    }

    private func forward(_ id: UInt32, _ data: Data) {
        guard let stream = streams[id] else { return }
        if stream.ready { write(stream, data) } else { stream.pending.append(data) }
    }

    private func write(_ stream: Stream, _ data: Data) {
        stream.connection.send(content: data, completion: .contentProcessed { [weak self, weak stream] error in
            guard let self, let stream else { return }
            if error != nil {
                self.finish(stream, Reason.unexpected)
                return
            }
            // Credit back only once the destination took the data: the client never has more
            // than `credit` packets queued here.
            guard stream.kind == StreamType.tcp, self.streams[stream.id] === stream else { return }
            stream.written += 1
            if stream.written >= Self.refillEvery {
                stream.written = 0
                self.sendPacket(Packet.continue, stream.id, Self.littleEndian(Self.credit))
            }
        })
    }

    private func read(_ stream: Stream) {
        if stream.kind == StreamType.udp {
            stream.connection.receiveMessage { [weak self, weak stream] data, _, _, error in
                guard let self, let stream, self.streams[stream.id] === stream else { return }
                if let data, !data.isEmpty { self.sendPacket(Packet.data, stream.id, data) }
                // An ICMP port unreachable for an earlier datagram is not the end of a UDP flow.
                if let error, case .posix(let code) = error, code != .ECONNREFUSED {
                    self.finish(stream, Reason.unexpected)
                    return
                }
                self.read(stream)
            }
            return
        }
        stream.connection.receive(minimumIncompleteLength: 1, maximumLength: Self.readChunk) { [weak self, weak stream] data, _, complete, error in
            guard let self, let stream, self.streams[stream.id] === stream else { return }
            if let data, !data.isEmpty { self.sendPacket(Packet.data, stream.id, data) }
            if error != nil {
                self.finish(stream, Reason.unexpected)
            } else if complete {
                self.finish(stream, Reason.voluntary)
            } else {
                self.read(stream)
            }
        }
    }

    // The client's CLOSE: what it sent first still goes out, then the connection ends. Nothing is
    // sent back (the client already forgot the stream).
    private func drop(_ id: UInt32) {
        guard let stream = streams.removeValue(forKey: id) else { return }
        if stream.ready { end(stream) } else { stream.closeAfterPending = true; finishWhenReady(stream) }
    }

    // A stream closed before it connected still writes its queued data once it does.
    private func finishWhenReady(_ stream: Stream) {
        stream.connection.stateUpdateHandler = { [weak self, weak stream] state in
            guard let self, let stream else { return }
            switch state {
            case .ready:
                for data in stream.pending { self.write(stream, data) }
                stream.pending = []
                self.end(stream)
            case .waiting, .failed: stream.connection.cancel()
            default: break
            }
        }
    }

    private func end(_ stream: Stream) {
        if stream.kind == StreamType.tcp {
            // A FIN after everything sent before it; then the connection goes.
            stream.connection.send(content: nil, contentContext: .finalMessage, isComplete: true,
                                   completion: .contentProcessed { [weak stream] _ in stream?.connection.cancel() })
        } else {
            stream.connection.cancel()
        }
    }

    // The destination ended or failed: tell the client, unless it closed the stream first.
    private func finish(_ stream: Stream, _ reason: UInt8) {
        guard streams[stream.id] === stream else { return }
        streams.removeValue(forKey: stream.id)
        stream.connection.cancel()
        sendPacket(Packet.close, stream.id, Data([reason]))
    }
}

// The WebKit engine's page connects here (`www/wisp.js`'s `connectWisp`), one WebSocket per run.
// Other apps can reach a loopback port too, so the page must offer the launch token as its
// WebSocket subprotocol (the server cannot see the URL's path); any other upgrade is refused.
final class WispSocketServer: LoopbackListener, @unchecked Sendable {
    private let token: String
    // The largest Wisp message: the client's record limit (`WISP_RECORD_MAX`).
    private static let maxMessage = 256 * 1024

    init(token: String) {
        self.token = token
        super.init(label: "loopback-wisp")
        let parameters = NWParameters.tcp
        let websocket = NWProtocolWebSocket.Options()
        websocket.autoReplyPing = true
        websocket.maximumMessageSize = Self.maxMessage
        websocket.setClientRequestHandler(queue) { subprotocols, _ in
            subprotocols.contains(token)
                ? NWProtocolWebSocket.Response(status: .accept, subprotocol: token)
                : NWProtocolWebSocket.Response(status: .reject, subprotocol: nil)
        }
        parameters.defaultProtocolStack.applicationProtocols.insert(websocket, at: 0)
        start(parameters)
    }

    override func accept(_ connection: NWConnection) {
        let session = WispSession { packet in
            let context = NWConnection.ContentContext(identifier: "wisp",
                                                      metadata: [NWProtocolWebSocket.Metadata(opcode: .binary)])
            connection.send(content: packet, contentContext: context, isComplete: true, completion: .contentProcessed { _ in })
        }
        session.start()
        receive(connection, session)
    }

    private func receive(_ connection: NWConnection, _ session: WispSession) {
        connection.receiveMessage { [weak self] data, _, _, error in
            guard error == nil, let data else {
                session.close()
                connection.cancel()
                return
            }
            session.receive(data)
            self?.receive(connection, session)
        }
    }
}
