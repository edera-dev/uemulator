#!/usr/bin/env python3
"""Static file server for the web build. SharedArrayBuffer (keyboard input into the worker that
runs the guest) needs the page to be cross-origin isolated, which takes the COOP/COEP headers.

Answers single-range `Range: bytes=a-b` requests with 206, which streaming a disk image needs:
`SimpleHTTPRequestHandler` ignores `Range` and would send the whole image for every read.

`/wisp/` is a Wisp v1 server (a WebSocket relay to TCP and UDP, MercuryWorkshop/wisp-protocol), so a
page served from here has internet without anything else installed. Every destination is allowed:
only pages from this server's own origin may use it."""
import base64
import hashlib
import http.server
import os
import queue
import re
import socket
import struct
import sys
import threading

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
COPY_CHUNK = 1 << 20

WISP_PATH = "/wisp/"
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # RFC 6455 section 1.3

OPCODE_CONTINUATION, OPCODE_TEXT, OPCODE_BINARY = 0x0, 0x1, 0x2
OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG = 0x8, 0x9, 0xA
CLOSE_PROTOCOL_ERROR, CLOSE_TOO_BIG = 1002, 1009
FIN_BIT, MASK_BIT, OPCODE_MASK, LENGTH_MASK = 0x80, 0x80, 0x0F, 0x7F
LENGTH_16_BIT, LENGTH_64_BIT = 126, 127
MAX_MESSAGE = 1 << 20

PACKET_CONNECT, PACKET_DATA, PACKET_CONTINUE, PACKET_CLOSE = 0x01, 0x02, 0x03, 0x04
STREAM_TCP, STREAM_UDP = 0x01, 0x02
CLOSE_VOLUNTARY, CLOSE_UNEXPECTED, CLOSE_INVALID_INFO = 0x02, 0x03, 0x41
CLOSE_UNREACHABLE, CLOSE_TIMEOUT, CLOSE_REFUSED = 0x42, 0x43, 0x44
WISP_HEADER = 5  # type u8 + stream id u32
WISP_CREDIT = 128  # packets the client may have in flight per TCP stream
WISP_REFILL_EVERY = 64  # DATA received per CONTINUE; half the credit, so the client never stalls
CONNECT_TIMEOUT = 10
READ_CHUNK = 64 * 1024


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0] == WISP_PATH:
            return self.wisp()
        match = RANGE.match(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        first, last = match.groups()
        if first:
            start, end = int(first), int(last) if last else size - 1
        elif last:
            start, end = max(0, size - int(last)), size - 1
        else:
            start, end = 0, size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = file.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def wisp(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
            return self.send_error(400, "a WebSocket upgrade is required")
        # A page on another origin must not reach the internet (and this machine's loopback)
        # through the user's browser. Non-browser clients send no Origin.
        origin = self.headers.get("Origin")
        port = self.server.server_address[1]
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self.send_error(403, "foreign origin")
        accept = base64.b64encode(hashlib.sha1((key + WEBSOCKET_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True
        WispConnection(self.connection, self.rfile).run()


class WispStream:
    """A stream's own thread resolves, connects and then writes the queued DATA in order, so the
    handler thread only enqueues and one slow or silent destination cannot hold up the others.
    `None` in the queue ends the writer."""

    def __init__(self, kind):
        self.kind = kind
        self.writes = queue.Queue()  # bounded in practice by the credit we grant the client
        self.lock = threading.Lock()  # guards sock and closed
        self.sock = None  # None until connected
        self.closed = False

    def attach(self, sock):
        """False if the stream was closed while connecting; the caller then owns closing sock."""
        with self.lock:
            if self.closed:
                return False
            self.sock = sock
            return True

    def close(self):
        with self.lock:
            self.closed = True
            sock = self.sock
        if sock:
            WispConnection.shutdown(sock)
        self.writes.put(None)


class WispConnection:
    """One WebSocket carrying Wisp v1. The handler thread reads frames and dispatches them; each
    stream has a writer thread (`serve_stream`) and, once connected, a reader thread (`pump`)
    sending DATA the other way. Every WebSocket write takes `send_lock` so frames never
    interleave."""

    def __init__(self, sock, rfile):
        self.sock = sock
        self.rfile = rfile
        self.send_lock = threading.Lock()
        self.streams = {}
        self.streams_lock = threading.Lock()

    def run(self):
        try:
            self.send_packet(PACKET_CONTINUE, 0, struct.pack("<I", WISP_CREDIT))
            self.serve_messages()
        except (OSError, EOFError):
            pass
        finally:
            with self.streams_lock:
                streams, self.streams = list(self.streams.values()), {}
            for stream in streams:
                stream.close()

    def read_exact(self, count):
        data = self.rfile.read(count)
        if len(data) != count:
            raise EOFError
        return data

    def serve_messages(self):
        message, message_opcode = b"", None
        while True:
            first, second = self.read_exact(2)
            fin, opcode = first & FIN_BIT, first & OPCODE_MASK
            length = second & LENGTH_MASK
            if length == LENGTH_16_BIT:
                (length,) = struct.unpack(">H", self.read_exact(2))
            elif length == LENGTH_64_BIT:
                (length,) = struct.unpack(">Q", self.read_exact(8))
            if not second & MASK_BIT:
                return self.send_close(CLOSE_PROTOCOL_ERROR)
            if length > MAX_MESSAGE:
                return self.send_close(CLOSE_TOO_BIG)
            key = self.read_exact(4)
            payload = self.read_exact(length)
            # XOR as one big integer: a per-byte loop is too slow for 64 KiB frames.
            repeated = (key * (length // 4 + 1))[:length]
            payload = (int.from_bytes(payload, "big") ^ int.from_bytes(repeated, "big")).to_bytes(length, "big")
            if opcode == OPCODE_CLOSE:
                return self.send_frame(OPCODE_CLOSE, payload[:2])
            if opcode == OPCODE_PING:
                self.send_frame(OPCODE_PONG, payload)
            elif opcode == OPCODE_PONG:
                pass
            elif opcode in (OPCODE_TEXT, OPCODE_BINARY, OPCODE_CONTINUATION):
                if opcode != OPCODE_CONTINUATION:
                    message, message_opcode = b"", opcode
                message += payload
                if len(message) > MAX_MESSAGE:
                    return self.send_close(CLOSE_TOO_BIG)
                if fin:
                    if message_opcode == OPCODE_BINARY:
                        self.handle_packet(message)
                    message, message_opcode = b"", None
            else:
                return self.send_close(CLOSE_PROTOCOL_ERROR)

    def send_frame(self, opcode, payload):
        length = len(payload)
        if length < LENGTH_16_BIT:
            header = bytes([FIN_BIT | opcode, length])
        elif length < 1 << 16:
            header = bytes([FIN_BIT | opcode, LENGTH_16_BIT]) + struct.pack(">H", length)
        else:
            header = bytes([FIN_BIT | opcode, LENGTH_64_BIT]) + struct.pack(">Q", length)
        with self.send_lock:
            self.sock.sendall(header + payload)

    def send_close(self, code):
        self.send_frame(OPCODE_CLOSE, struct.pack(">H", code))

    def send_packet(self, kind, stream_id, payload=b""):
        self.send_frame(OPCODE_BINARY, bytes([kind]) + struct.pack("<I", stream_id) + payload)

    @staticmethod
    def shutdown(sock):
        # close() alone does not wake a thread blocked in recv on Linux.
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def handle_packet(self, packet):
        if len(packet) < WISP_HEADER:
            return
        kind, (stream_id,), payload = packet[0], struct.unpack("<I", packet[1:WISP_HEADER]), packet[WISP_HEADER:]
        if kind == PACKET_CONNECT:
            self.connect(stream_id, payload)
        elif kind == PACKET_DATA:
            self.forward(stream_id, payload)
        elif kind == PACKET_CLOSE:
            self.drop(stream_id)
        # CONTINUE from a client means nothing: our credit to the destination is the socket's.

    def drop(self, stream_id):
        """A client CLOSE ends the stream gracefully: DATA queued before it (possibly still
        behind a pending connect) is written first, and the writer closes the socket on reaching
        the end marker. Only the connection ending and writer errors close hard, to release a
        writer blocked in sendall."""
        with self.streams_lock:
            stream = self.streams.pop(stream_id, None)
        if stream:
            stream.writes.put(None)
        return stream

    def finish(self, stream_id, stream, reason):
        """End a stream from its own threads and tell the client, unless a client CLOSE or the
        connection ending already removed it."""
        with self.streams_lock:
            if self.streams.get(stream_id) is not stream:
                return
            del self.streams[stream_id]
        stream.close()
        try:
            self.send_packet(PACKET_CLOSE, stream_id, bytes([reason]))
        except OSError:
            pass

    def connect(self, stream_id, payload):
        # A CONNECT on a live stream id replaces the stream: the old one is closed first, silently.
        self.drop(stream_id)
        if len(payload) < 3 or payload[0] not in (STREAM_TCP, STREAM_UDP):
            return self.send_packet(PACKET_CLOSE, stream_id, bytes([CLOSE_INVALID_INFO]))
        kind = payload[0]
        (port,) = struct.unpack("<H", payload[1:3])
        try:
            host = payload[3:].decode()
        except UnicodeDecodeError:
            return self.send_packet(PACKET_CLOSE, stream_id, bytes([CLOSE_INVALID_INFO]))
        stream = WispStream(kind)
        with self.streams_lock:
            self.streams[stream_id] = stream
        threading.Thread(target=self.serve_stream, args=(stream_id, stream, host, port), daemon=True).start()

    @staticmethod
    def open_destination(kind, host, port):
        if kind == STREAM_TCP:
            sock = socket.create_connection((host, port), CONNECT_TIMEOUT)
            sock.settimeout(None)
            return sock
        family, socktype, proto, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
        sock = socket.socket(family, socktype, proto)
        try:
            sock.connect(address)
        except OSError:
            sock.close()
            raise
        return sock

    def serve_stream(self, stream_id, stream, host, port):
        """Connects, then writes the stream's queue in order. No timeout on getaddrinfo, and
        create_connection's applies per resolved address; only this stream waits for either."""
        try:
            try:
                sock = self.open_destination(stream.kind, host, port)
            except (ValueError, OverflowError):  # includes UnicodeError: a label idna refuses
                return self.finish(stream_id, stream, CLOSE_INVALID_INFO)
            except socket.gaierror:
                return self.finish(stream_id, stream, CLOSE_UNREACHABLE)
            except (TimeoutError, socket.timeout):  # distinct classes before Python 3.10
                return self.finish(stream_id, stream, CLOSE_TIMEOUT)
            except ConnectionRefusedError:
                return self.finish(stream_id, stream, CLOSE_REFUSED)
            except OSError:
                return self.finish(stream_id, stream, CLOSE_UNEXPECTED)
            if not stream.attach(sock):
                return self.shutdown(sock)
            threading.Thread(target=self.pump, args=(stream_id, stream), daemon=True).start()
            consumed = 0
            while True:
                data = stream.writes.get()
                if data is None:
                    return stream.close()
                stream.sock.sendall(data)
                if stream.kind == STREAM_TCP:
                    # Credit comes back only once the destination has taken the data, so the queue
                    # never holds more than the client's credit.
                    consumed += 1
                    if consumed >= WISP_REFILL_EVERY:
                        consumed = 0
                        self.send_packet(PACKET_CONTINUE, stream_id, struct.pack("<I", WISP_CREDIT))
        except Exception:  # whatever happens here ends this stream only, never the connection
            self.finish(stream_id, stream, CLOSE_UNEXPECTED)

    def forward(self, stream_id, data):
        with self.streams_lock:
            stream = self.streams.get(stream_id)
        if stream:
            stream.writes.put(data)

    def pump(self, stream_id, stream):
        reason = CLOSE_VOLUNTARY
        try:
            while True:
                try:
                    data = stream.sock.recv(READ_CHUNK)
                except ConnectionRefusedError:
                    if stream.kind == STREAM_UDP:  # ICMP port unreachable for an earlier datagram
                        continue
                    raise
                if not data:
                    if stream.kind == STREAM_UDP:  # an empty datagram, not EOF
                        continue
                    break
                self.send_packet(PACKET_DATA, stream_id, data)
        except Exception:
            reason = CLOSE_UNEXPECTED
        self.finish(stream_id, stream, reason)


def make_server(root, port):
    handler = lambda *a, **k: Handler(*a, directory=root, **k)
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    root, port = sys.argv[1], int(sys.argv[2])
    server = make_server(root, port)
    print(f"http://127.0.0.1:{port}/")
    server.serve_forever()
