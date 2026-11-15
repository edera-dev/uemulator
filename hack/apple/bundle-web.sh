#!/bin/sh
# Copies what the M0 spike's hidden web view serves into the app bundle (docs/plans/28-apple.md):
# the web build's worker and JIT host (unchanged), the module and firmware, the Alpine demo's
# kernel and initramfs, and the spike's own headless page. Run `WEB_DEMO=1 hack/web.sh` first.
set -e
cd "$(dirname "$0")/../.." || exit 1
out="$1"
web=target/web
for need in "$web/uemulator_web.wasm" "$web/demo/vmlinuz-virt" "$web/demo/initramfs.gz"; do
    if [ ! -e "$need" ]; then
        echo "error: $need is missing; run WEB_DEMO=1 hack/web.sh first" >&2
        exit 1
    fi
done
rm -rf "$out"
mkdir -p "$out/demo"
cp -L "$web"/*.js "$web/uemulator_web.wasm" "$web/uemulator-firmware.bin" "$out/"
cp -L "$web/demo/vmlinuz-virt" "$web/demo/initramfs.gz" "$out/demo/"
cp apple/M0Spike/engine/* "$out/"
