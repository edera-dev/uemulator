#!/bin/sh
# Copies the app's resources into its bundle (docs/plans/28-apple.md): the UEFI firmware
# (build-core.sh builds it) and, for M1, the Alpine demo machine's kernel and initramfs, which
# `WEB_DEMO=1 hack/web.sh` downloads (hack/web/demo.py: Alpine's virt kernel and a minirootfs
# with an /init that starts a shell).
set -e
cd "$(dirname "$0")/../.." || exit 1
out="$1"
firmware=target/firmware/x86_64-unknown-none/release/uemulator-firmware
demo=target/web/demo
for need in "$firmware" "$demo/vmlinuz-virt" "$demo/initramfs.gz"; do
    if [ ! -e "$need" ]; then
        echo "error: $need is missing; run WEB_DEMO=1 hack/web.sh first" >&2
        exit 1
    fi
done
mkdir -p "$out/demo"
cp -L "$firmware" "$out/uemulator-firmware.bin"
cp -L "$demo/vmlinuz-virt" "$demo/initramfs.gz" "$out/demo/"
