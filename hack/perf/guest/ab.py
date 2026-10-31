#!/usr/bin/env python3
"""Runs drive.py with CMDFILE on each binary in turn, ROUNDS times, and prints the median of
every benchmark line per binary. Interleaving keeps a busy host from favouring one binary.

usage: ab.py CMDFILE ROUNDS NAME=BINARY... [-- uemulator quick boot args...]
"""
import collections
import os
import re
import statistics
import subprocess
import sys

args = sys.argv[1:]
boot_args = []
if '--' in args:
    split = args.index('--')
    args, boot_args = args[:split], args[split + 1:]
cmdfile, rounds, binaries = args[0], int(args[1]), [arg.split('=', 1) for arg in args[2:]]
drive = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drive.py')
results = {name: collections.defaultdict(list) for name, _ in binaries}
order = []
for _ in range(rounds):
    for name, exe in binaries:
        out = subprocess.run(['python3', drive, cmdfile] + boot_args, env=dict(os.environ, UEMU=exe),
                             capture_output=True, text=True, timeout=1800).stdout
        for line in out.splitlines():
            match = re.match(r'^(\S.*?)\s{2,}([\d.]+) (\S+)$', line)
            if match and match.group(1) != 'guest elapsed':
                key = (match.group(1), match.group(3))
                if key not in order:
                    order.append(key)
                results[name][key].append(float(match.group(2)))
names = [name for name, _ in binaries]
header = '%-26s %-8s ' % ('benchmark', 'unit') + ' '.join('%14s' % name for name in names)
print(header + ('    ratio' if len(names) == 2 else ''))
for key in order:
    medians = [statistics.median(results[n][key]) if results[n][key] else float('nan') for n in names]
    line = '%-26s %-8s ' % key + ' '.join('%14.2f' % m for m in medians)
    if len(medians) == 2 and medians[0]:
        line += '   %6.2fx' % (medians[1] / medians[0])
    print(line, flush=True)
