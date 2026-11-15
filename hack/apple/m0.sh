#!/bin/sh
# Builds the M0 spike (apple/M0Spike, docs/plans/28-apple.md) for a simulator, runs it with
# -autorun and prints its result lines. SIMULATOR is a name or UDID (default: the booted one);
# UEMULATOR_BOOT=jit|interpreter runs one boot only. Run `WEB_DEMO=1 hack/web.sh` first.
set -e
cd "$(dirname "$0")/../.." || exit 1
device="${SIMULATOR:-booted}"
derived=target/apple-m0
xcodebuild -quiet -project apple/M0Spike/M0Spike.xcodeproj -scheme M0Spike -configuration Release \
    -destination "generic/platform=iOS Simulator" -derivedDataPath "$derived" build
xcrun simctl install "$device" "$derived/Build/Products/Release-iphonesimulator/M0Spike.app"
log="$derived/m0.log"
# --console-pty stays attached until the app exits, and the app stays up after its run, so the
# app is terminated once it prints "M0 done" (or after 15 minutes).
SIMCTL_CHILD_UEMULATOR_BOOT="$UEMULATOR_BOOT" xcrun simctl launch --console-pty --terminate-running-process \
    "$device" dev.edera.uemulator.m0spike -autorun > "$log" 2>&1 &
launcher=$!
for _ in $(seq 900); do
    grep -q '^M0 done' "$log" && break
    sleep 1
done
xcrun simctl terminate "$device" dev.edera.uemulator.m0spike 2>/dev/null || true
wait "$launcher" 2>/dev/null || true
grep '^M0 ' "$log"
