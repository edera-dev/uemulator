#!/usr/bin/env python3
"""The host side of the guest benchmarks, on loopback (the guest reaches it as 10.0.2.2):
HTTP on 8099 serving DIR (where `ubench` is built), 5201 sends the MiB count it is sent and
closes, 5202 swallows everything and answers "ok" at EOF, 5203 echoes.

usage: netsrv.py [DIR]  (default: target/perf-guest)
"""
import functools
import http.server
import os
import socket
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MIB = 1 << 20


def serve(port, handler):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', port))
    listener.listen(8)
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=handler, args=(conn,), daemon=True).start()


def source(conn):
    left = int(conn.makefile('rb').readline()) * MIB
    chunk = b'\0' * MIB
    while left > 0:
        conn.sendall(chunk[:min(left, MIB)])
        left -= MIB
    conn.close()


def sink(conn):
    while conn.recv(MIB):
        pass
    conn.sendall(b'ok')
    conn.close()


def echo(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while data := conn.recv(4096):
        conn.sendall(data)
    conn.close()


for port, handler in ((5201, source), (5202, sink), (5203, echo)):
    threading.Thread(target=serve, args=(port, handler), daemon=True).start()
directory = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'target', 'perf-guest')
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
http.server.ThreadingHTTPServer(('127.0.0.1', 8099), handler).serve_forever()
