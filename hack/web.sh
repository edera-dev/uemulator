#!/bin/sh
# Builds the WebAssembly port into target/web and, with `serve`, serves it on http://127.0.0.1:8080/
# (PORT overrides). Needs `rustup target add wasm32-unknown-unknown x86_64-unknown-none` (the second
# builds the UEFI firmware, which the page fetches as uemulator-firmware.bin when it boots through it).
#
#   hack/web.sh          build only
#   hack/web.sh serve    build, then serve
#
# WEB_DEMO=1 also builds the demo machines (hack/web/demo.py: Alpine's kernel and minirootfs, and
# Edera's Xen from the image hack/web/demo-xen.ref pins, downloaded once into
# target/web-demo-cache) into demo/, which puts the page in demo mode: buttons that boot Alpine,
# and Xen with Alpine as dom0. The Pages workflow sets it.
#
# Kernel and initramfs: pick files on the page, or run `uemulator quick create alpine --install alpine`
# once; that machine's netboot files are linked under assets/ and prefilled (UEMULATOR_MACHINE names
# another Alpine machine). WEB_DISK=IMAGE links a disk image as assets/disk and prefills it; make
# one that nothing writes to any more, for example from an installed machine:
#
#   uemulator disk convert ~/.uemulator/machines/alpine/disk-0.udisk alpine.udisk
set -e

cd "$(dirname "${0}")/.." || exit 1

out=target/web
cargo build --release -p uemulator-web --target wasm32-unknown-unknown
# Its own target dir, as in crates/uemulator-cli/build.rs: the firmware's build.rs links a flat
# image only for this target, and the binary is that image.
cargo build --release -p uemulator-firmware --target x86_64-unknown-none --target-dir target/firmware

rm -rf "$out"
mkdir -p "$out"
cp crates/uemulator-web/www/* "$out/"
cp target/wasm32-unknown-unknown/release/uemulator_web.wasm "$out/"
cp target/firmware/x86_64-unknown-none/release/uemulator-firmware "$out/uemulator-firmware.bin"

cache="${UEMULATOR_HOME:-$HOME/.uemulator}/machines/${UEMULATOR_MACHINE:-alpine}/artifacts"
if [ -f "$cache/vmlinuz-virt" ] && [ -f "$cache/initramfs-virt" ]; then
    mkdir -p "$out/assets"
    ln -s "$cache/vmlinuz-virt" "$out/assets/vmlinuz-virt"
    ln -s "$cache/initramfs-virt" "$out/assets/initramfs-virt"
fi

if [ -n "$WEB_DEMO" ]; then
    python3 hack/web/demo.py "$out/demo"
fi

# hack/gpu/image.py's guest, for ?gpu (docs/plans/25-gpu-3d.md), when it has been built.
if [ -d target/gpu ]; then
    ln -s "$(pwd)/target/gpu" "$out/gpu"
fi

if [ -n "$WEB_DISK" ]; then
    mkdir -p "$out/assets"
    ln -s "$(cd "$(dirname "$WEB_DISK")" && pwd)/$(basename "$WEB_DISK")" "$out/assets/disk"
fi

if [ "$1" = serve ]; then
    exec python3 hack/web/serve.py "$out" "${PORT:-8080}"
fi
echo "built $out"
