#!/usr/bin/env python3
"""Static file server for the web build. SharedArrayBuffer (keyboard input into the worker that
runs the guest) needs the page to be cross-origin isolated, which takes the COOP/COEP headers.

Answers single-range `Range: bytes=a-b` requests with 206, which streaming a disk image needs:
`SimpleHTTPRequestHandler` ignores `Range` and would send the whole image for every read."""
import http.server
import os
import re
import sys

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
COPY_CHUNK = 1 << 20


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


if __name__ == "__main__":
    root, port = sys.argv[1], int(sys.argv[2])
    handler = lambda *a, **k: Handler(*a, directory=root, **k)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"http://127.0.0.1:{port}/")
    server.serve_forever()
