"""Tests for serve.py's /wisp/ endpoint: `python3 -m unittest hack/web/test_serve.py`.
Everything stays on 127.0.0.1."""
import base64
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

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
        real = socket.getaddrinfo

        def fake(host, *args, **kwargs):
            if host == "unresolvable.test":
                raise socket.gaierror
            return real(host, *args, **kwargs)

        client = self.open()
        client.packet()
        with mock.patch.object(socket, "getaddrinfo", fake):
            client.send(connect_packet(7, TCP, 80, "unresolvable.test"))
            self.assertEqual(client.packet(), (CLOSE, 7, b"\x42"))

    def echo_round_trip(self, client, stream, listener, port):
        client.send(connect_packet(stream, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        peer.settimeout(TIMEOUT)
        client.send(wisp(DATA, stream, b"echo"))
        self.assertEqual(read_exact(peer, 4), b"echo")
        peer.sendall(b"back")
        self.assertEqual(client.packet(), (DATA, stream, b"back"))
        return peer

    def test_malformed_hostnames_close_only_their_stream(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        for stream, host in enumerate(["a" * 70 + ".com", "a..b"], 1):
            client.send(connect_packet(stream, TCP, 80, host))
            self.assertEqual(client.packet(), (CLOSE, stream, b"\x41"))
        self.echo_round_trip(client, 9, listener, port)

    def delayed_connect(self, release):
        """open_destination that blocks hang.test until `release` is set, then fails."""
        real = serve.WispConnection.open_destination

        def fake(kind, host, port):
            if host == "hang.test":
                release.wait(TIMEOUT)
                raise ConnectionRefusedError
            if host == "slow.test":
                release.wait(TIMEOUT)
                host = "127.0.0.1"
            return real(kind, host, port)

        return mock.patch.object(serve.WispConnection, "open_destination", staticmethod(fake))

    def test_hung_connect_does_not_delay_other_streams(self):
        release = threading.Event()
        self.addCleanup(release.set)
        listener, port = self.listener()
        client = self.open()
        client.packet()
        with self.delayed_connect(release):
            client.send(connect_packet(1, TCP, 80, "hang.test"))
            client.send(wisp(DATA, 1, b"queued"))
            client.send(b"hi", opcode=9)
            self.assertEqual(client.frame(), (10, b"hi"))
            self.echo_round_trip(client, 2, listener, port)
            release.set()
            self.assertEqual(client.packet(), (CLOSE, 1, b"\x44"))

    def test_data_before_connect_finishes_stays_in_order(self):
        release = threading.Event()
        self.addCleanup(release.set)
        listener, port = self.listener()
        client = self.open()
        client.packet()
        with self.delayed_connect(release):
            client.send(connect_packet(1, TCP, port, "slow.test"))
            for piece in (b"a", b"b", b"c"):
                client.send(wisp(DATA, 1, piece))
            release.set()
            peer, _ = listener.accept()
            peer.settimeout(TIMEOUT)
            self.assertEqual(read_exact(peer, 3), b"abc")

    def test_destination_that_never_reads_does_not_stall_other_streams(self):
        silent, silent_port = self.listener()
        listener, port = self.listener()
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, silent_port, "127.0.0.1"))
        silent.accept()[0].setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        block = b"x" * (256 * 1024)
        for _ in range(80):
            client.send(wisp(DATA, 1, block))
        client.send(b"hi", opcode=9)
        while True:
            opcode, payload = client.frame()
            if opcode == 10:
                break
        self.assertEqual(payload, b"hi")
        self.echo_round_trip(client, 2, listener, port)

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

    def count_until_eof(self, listener):
        total = []

        def sink():
            peer, _ = listener.accept()
            count = 0
            while True:
                chunk = peer.recv(1 << 16)
                if not chunk:
                    break
                count += len(chunk)
            total.append(count)

        thread = threading.Thread(target=sink, daemon=True)
        thread.start()
        return thread, total

    BURST_BLOCKS, BURST_BLOCK = 64, b"x" * (64 * 1024)

    def burst_then_close(self, host, release=None):
        listener, port = self.listener()
        thread, total = self.count_until_eof(listener)
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, port, host))
        for _ in range(self.BURST_BLOCKS):
            client.send(wisp(DATA, 1, self.BURST_BLOCK))
        client.send(wisp(CLOSE, 1, b"\x02"))
        if release:
            release.set()
        thread.join(TIMEOUT)
        self.assertEqual(total, [self.BURST_BLOCKS * len(self.BURST_BLOCK)])

    def test_client_close_delivers_queued_data(self):
        self.burst_then_close("127.0.0.1")

    def test_client_close_behind_pending_connect_delivers_queued_data(self):
        release = threading.Event()
        self.addCleanup(release.set)
        with self.delayed_connect(release):
            self.burst_then_close("slow.test", release)

    def silent_sink(self):
        """A destination that accepts, reads nothing until told to, then counts to EOF or reset."""
        listener, port = self.listener()
        start, state = threading.Event(), {"count": 0, "eof": False}

        def sink():
            peer, _ = listener.accept()
            peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            peer.settimeout(TIMEOUT)
            start.wait(TIMEOUT)
            try:
                while True:
                    chunk = peer.recv(1 << 16)
                    if not chunk:
                        break
                    state["count"] += len(chunk)
            except OSError:
                pass
            state["eof"] = True

        thread = threading.Thread(target=sink, daemon=True)
        thread.start()
        return port, start, state, thread

    def packets_until_pong(self, client):
        client.send(b"sync", opcode=9)
        packets = []
        while True:
            opcode, payload = client.frame()
            if opcode == 10:
                return packets
            packets.append((payload[0], struct.unpack("<I", payload[1:5])[0], payload[5:]))

    def test_no_data_reaches_the_client_after_its_close(self):
        listener, port = self.listener()
        client = self.open()
        client.packet()
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        peer, _ = listener.accept()
        stop = threading.Event()
        self.addCleanup(stop.set)

        def chatter():
            try:
                while not stop.is_set():
                    peer.sendall(b"y")
            except OSError:
                pass

        threading.Thread(target=chatter, daemon=True).start()
        client.packet()  # some DATA is flowing
        client.send(wisp(CLOSE, 1, b"\x02"))
        self.packets_until_pong(client)
        time.sleep(0.2)
        self.assertEqual(self.packets_until_pong(client), [])

    def test_reused_id_sees_nothing_from_the_draining_stream(self):
        port_a, start, state, thread = self.silent_sink()
        listener, port_b = self.listener()
        client = self.open()
        client.packet()
        block = b"x" * (256 * 1024)
        client.send(connect_packet(1, TCP, port_a, "127.0.0.1"))
        for _ in range(40):
            client.send(wisp(DATA, 1, block))
        for _ in range(70):
            client.send(wisp(DATA, 1, b"."))
        client.send(wisp(CLOSE, 1, b"\x02"))
        peer = self.echo_round_trip(client, 1, listener, port_b)  # keep it open: its EOF is a legit CLOSE
        self.addCleanup(peer.close)
        start.set()
        thread.join(TIMEOUT)
        self.assertEqual(state["count"], 40 * len(block) + 70)
        self.assertEqual(self.packets_until_pong(client), [])

    def test_websocket_end_releases_a_blocked_dropped_stream(self):
        port, start, state, thread = self.silent_sink()
        client = self.open()
        client.packet()
        block = b"x" * (256 * 1024)
        blocks = 200
        client.send(connect_packet(1, TCP, port, "127.0.0.1"))
        for _ in range(blocks):
            client.send(wisp(DATA, 1, block))
        client.send(wisp(CLOSE, 1, b"\x02"))
        client.close()
        time.sleep(1)  # the writer is blocked in sendall; the queue holds the rest
        start.set()
        thread.join(TIMEOUT)
        self.assertTrue(state["eof"])
        self.assertLess(state["count"], blocks * len(block))

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
