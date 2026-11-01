#!/usr/bin/env bash
# Wall seconds from process start to the dom0 shell of the web demo's Xen machine (Edera's Xen
# 4.22 xen.gz, Alpine's virt kernel and the demo initramfs, 512 MiB, dom0_mem=384M), natively or
# under Node on the WebAssembly build. Prints `<config> <seconds>`.
#
#   hack/perf/xen.sh pv|pv-iommu|pvh-iommu [native|wasm]
#
# pvh-iommu is `dom0=pvh-auto` with the IOMMU (Xen builds a PVH dom0 only with one); the PV
# configurations force `dom0=pv`. The artifacts come from `WEB_DEMO=1 hack/web.sh` (target/web/demo,
# or DEMO); UEMULATOR_BIN and WASM pick other builds. Guest time is the host's wall clock, so a
# loaded host moves the numbers: interleave the builds being compared and take medians.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
config="${1:?pv, pv-iommu or pvh-iommu}"
engine="${2:-native}"
demo="${DEMO:-$root/target/web/demo}"
marker="on uemulator, running from RAM"
xen_args="console=com1 com1=115200,8n1 dom0_mem=384M,max:384M"
case "$config" in
    pv) iommu=(); xen_args="$xen_args dom0=pv" ;;
    pv-iommu) iommu=(--iommu); xen_args="$xen_args dom0=pv" ;;
    pvh-iommu) iommu=(--iommu); xen_args="$xen_args dom0=pvh-auto" ;;
    *) echo "unknown configuration $config" >&2; exit 2 ;;
esac
log="$(mktemp)"
trap 'kill "$pid" 2>/dev/null || true; rm -f "$log"' EXIT
start=$(python3 -c 'import time; print(time.time())')
if [ "$engine" = wasm ]; then
    node "$root/hack/wasm/boot.mjs" --wasm "${WASM:-$root/target/web/uemulator_web.wasm}" \
        --kernel "$demo/vmlinuz-virt" --initrd "$demo/initramfs.gz" --memory 512 --firmware uefi \
        "${iommu[@]}" --hypervisor "$demo/xen.gz" --hypervisor-cmdline "$xen_args" --cmdline "" \
        --seconds 300 >"$log" 2>&1 </dev/null &
else
    "${UEMULATOR_BIN:-$root/target/release/uemulator}" advanced boot \
        --kernel "$demo/vmlinuz-virt" --initramfs "$demo/initramfs.gz" --memory 512 \
        "${iommu[@]}" --hypervisor "$demo/xen.gz" --hypervisor-cmdline "$xen_args" \
        >"$log" 2>&1 </dev/null &
fi
pid=$!
until grep -aq "$marker" "$log"; do
    kill -0 "$pid" 2>/dev/null || { echo "the machine stopped before the shell" >&2; exit 1; }
    sleep 0.05
done
python3 -c "import time; print('$config %.2f' % (time.time() - $start))"
