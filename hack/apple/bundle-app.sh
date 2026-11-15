#!/bin/sh
# Copies the app's resources into its bundle (docs/plans/28-apple.md): the UEFI firmware
# (build-core.sh builds it); the WebKit engine's web/ folder (the web build's module, worker and JIT
# host, unchanged, and apple/WebEngine's page in place of main.js); and the Alpine demo machine's
# kernel and initramfs, which
# `WEB_DEMO=1 hack/web.sh` downloads (hack/web/demo.py: Alpine's virt kernel and a minirootfs
# with an /init that starts a shell).
set -e
cd "$(dirname "$0")/../.." || exit 1
out="$1"
firmware=target/firmware/x86_64-unknown-none/release/uemulator-firmware
demo=target/web/demo
web=target/web
for need in "$firmware" "$demo/vmlinuz-virt" "$demo/initramfs.gz" "$web/uemulator_web.wasm"; do
    if [ ! -e "$need" ]; then
        echo "error: $need is missing; run WEB_DEMO=1 hack/web.sh first" >&2
        exit 1
    fi
done
mkdir -p "$out/demo"
cp -L "$firmware" "$out/uemulator-firmware.bin"
cp -L "$demo/vmlinuz-virt" "$demo/initramfs.gz" "$out/demo/"
rm -rf "$out/web"
mkdir -p "$out/web"
# The worker's imports: protocol.js, gpu.js, jit_host.js, segment.js, wisp.js (main.js and the
# page's own scripts come along and are never loaded).
cp -L "$web"/*.js "$web/uemulator_web.wasm" "$out/web/"
cp apple/WebEngine/engine.html apple/WebEngine/engine.js "$out/web/"
