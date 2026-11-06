"""Tests for serve.py's /wisp/ endpoint: `python3 -m unittest hack/web/test_serve.py`.
Everything stays on 127.0.0.1."""
import base64
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve  # noqa: E402

CONNECT, DATA, CONTINUE, CLOSE = 1, 2, 3, 4
TCP, UDP = 1, 2
TIMEOUT = 5


def wisp(kind, stream, payload=b""):
    return bytes([kind]) + struct.pack("<I", stream) + payload


def connect_packet(stream, kind, port, host):
    return wisp(CONNECT, stream, bytes([kind]) + struct.pack("<H", port) + host.encode())


def read_exact(sock, count):
    out = b""
    while len(out) < count:
        chunk = sock.recv(count - len(out))
        if not chunk:
            raise EOFError
        out += chunk
    return out


class Client:
    """Masked-frame WebSocket client, enough for one Wisp connection."""

    def __init__(self, port, origin="default"):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
        if origin == "default":
            origin = f"http://127.0.0.1:{port}"
        headers = [
            "GET /wisp/ HTTP/1.1",
            f"Host: 127.0.0.1:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Version: 13",
            "Sec-WebSocket-Key: " + base64.b64encode(os.urandom(16)).decode(),
        ]
        if origin is not None:
            headers.append(f"Origin: {origin}")
        self.sock.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())
        head = b""
        while not head.endswith(b"\r\n\r\n"):
            head += read_exact(self.sock, 1)
        self.status = int(head.split(b" ", 2)[1])

    def send(self, payload, opcode=2, fin=True, mask=True):
        first = (0x80 if fin else 0) | opcode
        length = len(payload)
        if length < 126:
            header = bytes([first, (0x80 if mask else 0) | length])
        elif length < 1 << 16:
            header = bytes([first, (0x80 if mask else 0) | 126]) + struct.pack(">H", length)
        else:
            header = bytes([first, (0x80 if mask else 0) | 127]) + struct.pack(">Q", length)
        if mask:
            key = os.urandom(4)
            payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
            header += key
        self.sock.sendall(header + payload)

    def frame(self):
        first, second = read_exact(self.sock, 2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", read_exact(self.sock, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", read_exact(self.sock, 8))[0]
        assert not second & 0x80, "server frames are unmasked"
        return first & 0x0F, read_exact(self.sock, length)

    def packet(self):
        """Next Wisp packet as (type, stream, payload); skips pings."""
        while True:
            opcode, payload = self.frame()
            if opcode == 2:
                return payload[0], struct.unpack("<I", payload[1:5])[0], payload[5:]
            assert opcode in (9, 10), opcode

    def close(self):
        self.sock.close()


class ServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        with open(os.path.join(cls.root, "file.txt"), "wb") as file:
            file.write(b"0123456789")
        cls.server = serve.make_server(cls.root, 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def open(self, **kwargs):
        client = Client(self.port, **kwargs)
        self.addCleanup(client.close)
        return client

    def listener(self, kind=socket.SOCK_STREAM):
        sock = socket.socket(socket.AF_INET, kind)
        sock.bind(("127.0.0.1", 0))
        if kind == socket.SOCK_STREAM:
            sock.listen(1)
        self.addCleanup(sock.close)
        return sock, sock.getsockname()[1]

    def test_foreign_origin_is_refused(self):
        self.assertEqual(self.open(origin="https://evil.example").status, 403)

    def test_own_origin_and_no_origin_are_accepted(self):
        self.assertEqual(self.open().status, 101)
        self.assertEqual(self.open(origin=None).status, 101)
        self.assertEqual(self.open(origin=f"http://localhost:{self.port}").status, 101)

    def test_other_port_origin_is_refused(self):
        self.assertEqual(self.open(origin=f"http://127.0.0.1:{self.port + 1}").status, 403)

    def test_files_still_served(self):
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        conn.request("GET", "/file.txt", headers={"Range": "bytes=2-4"})
        response = conn.getresponse()
        self.assertEqual((response.status, response.read()), (206, b"234"))
        conn.request("GET", "/file.txt")
        response = conn.getresponse()
        self.assertEqual((response.status, response.read()), (200, b"0123456789"))

    def test_initial_continue_then_tcp_echo(self):
        listener, port = self.listener()
        client = self.open()
        self.assertEqual(client.status, 101)
        self.assertEqual(client.packet(), (CONTINUE, 0, struct.pack("<I", 128)))
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        peer.settimeout(TIMEOUT)
        client.send(wisp(DATA, 1, b"hello"))
        self.assertEqual(read_exact(peer, 5), b"hello")
        peer.sendall(b"world")
        self.assertEqual(client.packet(), (DATA, 1, b"world"))
        peer.close()
        self.assertEqual(client.packet(), (CLOSE, 1, b"\x02"))

    def test_refused_connect_closes_with_0x44(self):
        listener, port = self.listener()
        listener.close()
        client = self.open()
        client.packet()
        client.send(connect_packet(7, TCP, port, "127.0.0.1"))
        self.assertEqual(client.packet(), (CLOSE, 7, b"\x44"))

    def test_unresolvable_closes_with_0x42(self):
        client = self.open()
        client.packet()
        client.send(connect_packet(7, TCP, 80, "nonexistent.invalid"))
        self.assertEqual(client.packet(), (CLOSE, 7, b"\x42"))

    def test_udp_round_trip(self):
        sock, port = self.listener(socket.SOCK_DGRAM)
        sock.settimeout(TIMEOUT)
        client = self.open()
        client.packet()
        client.send(connect_packet(3, UDP, port, "127.0.0.1"))
        client.send(wisp(DATA, 3, b"ping"))
        data, address = sock.recvfrom(100)
        self.assertEqual(data, b"ping")
        sock.sendto(b"pong", address)
        self.assertEqual(client.packet(), (DATA, 3, b"pong"))

    def test_client_close_closes_the_socket(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        peer.settimeout(TIMEOUT)
        client.send(wisp(CLOSE, 1, b"\x02"))
        self.assertEqual(peer.recv(10), b"")

    def test_fragmented_message_and_ping(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        packet = connect_packet(1, TCP, port, "127.0.0.1")
        client.send(packet[:4], opcode=2, fin=False)
        client.send(b"ping!", opcode=9)
        client.send(packet[4:], opcode=0, fin=True)
        self.assertEqual(client.frame(), (10, b"ping!"))
        peer, _ = listener.accept()
        peer.close()
        self.assertEqual(client.packet(), (CLOSE, 1, b"\x02"))

    def test_credit_refill_after_64_data(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        peer.settimeout(TIMEOUT)
        for _ in range(64):
            client.send(wisp(DATA, 1, b"x"))
        self.assertEqual(client.packet(), (CONTINUE, 1, struct.pack("<I", 128)))
        self.assertEqual(len(read_exact(peer, 64)), 64)

    def test_unmasked_client_frame_closes_1002(self):
        client = self.open()
        client.packet()
        client.send(wisp(DATA, 9, b"x"), mask=False)
        opcode, payload = client.frame()
        self.assertEqual((opcode, struct.unpack(">H", payload[:2])[0]), (8, 1002))

    def test_websocket_end_closes_stream_sockets(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        peer.settimeout(TIMEOUT)
        client.close()
        self.assertEqual(peer.recv(10), b"")


if __name__ == "__main__":
    unittest.main()
