#!/usr/bin/env python3
"""Boots a quick machine that logs in to a root shell by itself, runs each line of CMDFILE at the
prompt and prints what it printed, with the host time each line took.

usage: drive.py CMDFILE [uemulator quick boot args...]
env: UEMU (binary, default target/release/uemulator), MACHINE (default perf-test),
     CMD_TIMEOUT (seconds per line, default 600), DRIVE_LOG (raw console log),
     SAMPLE=N (run `sample` on the emulator during line N, for SAMPLE_SECS into SAMPLE_OUT)
"""
import os
import re
import select
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
lines = [l.rstrip('\n') for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
exe = os.environ.get('UEMU', os.path.join(ROOT, 'target', 'release', 'uemulator'))
machine = os.environ.get('MACHINE', 'perf-test')
started = time.time()
emulator = subprocess.Popen([exe, 'quick', 'boot', machine] + sys.argv[2:], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
pending = b''
log = open(os.environ.get('DRIVE_LOG', os.devnull), 'wb')


def read_some(timeout):
    global pending
    ready, _, _ = select.select([emulator.stdout], [], [], timeout)
    if not ready:
        return True
    data = os.read(emulator.stdout.fileno(), 65536)
    pending += data
    log.write(data)
    log.flush()
    return bool(data)


def wait_for(pattern, timeout):
    global pending
    deadline = time.time() + timeout
    while time.time() < deadline:
        match = re.search(pattern, pending)
        if match:
            out, pending = pending[:match.end()], pending[match.end():]
            return out
        if not read_some(0.2):
            break
    sys.exit('timed out waiting for %r; console tail: %r' % (pattern, pending[-500:]))


wait_for(rb'# $|# \x1b', 300)
print('boot %.1fs' % (time.time() - started), flush=True)
emulator.stdin.write(b'\n')
wait_for(rb'# ', 30)
for index, line in enumerate(lines):
    # The marker is split in the command so the echoed command line does not match it.
    marker = 'DONE%d' % index
    emulator.stdin.write(('%s; echo %s""%s\n' % (line, marker[:2], marker[2:])).encode())
    began = time.time()
    if os.environ.get('SAMPLE') == str(index):
        time.sleep(float(os.environ.get('SAMPLE_DELAY', '1')))
        subprocess.Popen(['sample', str(emulator.pid), os.environ.get('SAMPLE_SECS', '5'), '-file',
                          os.environ.get('SAMPLE_OUT', 'sample.txt')],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = wait_for(re.escape(marker.encode()) + rb'\r?\n', float(os.environ.get('CMD_TIMEOUT', '600')))
    text = out.decode(errors='replace').replace('\r', '')
    print('$ %s   [host %.2fs]' % (line, time.time() - began))
    print('\n'.join(text.split('\n')[1:-1]).rstrip(), flush=True)
emulator.stdin.write(b'poweroff\n')
emulator.stdin.close()
# Keep reading so a `--debug` dump at exit reaches DRIVE_LOG.
deadline = time.time() + 60
while time.time() < deadline and read_some(0.5):
    pass
try:
    emulator.wait(5)
except subprocess.TimeoutExpired:
    emulator.kill()
