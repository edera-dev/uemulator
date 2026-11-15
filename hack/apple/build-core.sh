#!/bin/sh
# Builds crates/uemulator-apple, the static library the Apple app links (docs/plans/28-apple.md),
# and the UEFI firmware the app bundles.
#
#   hack/apple/build-core.sh --xcode       from the app's build phase: the slices for Xcode's
#                                          PLATFORM_NAME and ARCHS, lipo'd into BUILT_PRODUCTS_DIR
#   hack/apple/build-core.sh xcframework   every slice, into target/apple/UemulatorCore.xcframework
#
# Needs `rustup target add aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios
# aarch64-apple-ios-sim x86_64-unknown-none` (x86_64-apple-ios too for an Intel simulator).
set -e
cd "$(dirname "$0")/../.." || exit 1
repo=$(pwd)
library=libuemulator_apple.a

# Xcode's environment (SDKROOT, the deployment targets of the other platform, its own
# toolchain's paths) breaks the host build scripts cargo runs, so cargo gets a clean one.
cargo_build() {
    env -i HOME="$HOME" PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}" \
        IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}" \
        ${RUSTUP_TOOLCHAIN:+RUSTUP_TOOLCHAIN="$RUSTUP_TOOLCHAIN"} \
        cargo build --release "$@"
}

firmware() {
    # Its own target dir, as hack/web.sh and the CLI's build.rs do: the firmware's build.rs links
    # a flat image only for this target, and the binary is that image.
    cargo_build -p uemulator-firmware --target x86_64-unknown-none --target-dir target/firmware
}

# Rust's target for an Xcode platform and architecture.
rust_target() {
    case "$1/$2" in
        macosx/arm64) echo aarch64-apple-darwin ;;
        macosx/x86_64) echo x86_64-apple-darwin ;;
        iphoneos/arm64) echo aarch64-apple-ios ;;
        iphonesimulator/arm64) echo aarch64-apple-ios-sim ;;
        iphonesimulator/x86_64) echo x86_64-apple-ios ;;
        *) echo "error: no Rust target for $1 $2" >&2; return 1 ;;
    esac
}

# Builds the given targets and lipos their archives into one at $1. POSIX sh has no locals, so
# the names here must not be ones a caller uses (`out` is the xcframework branch's).
fat_library() {
    fat_output=$1
    shift
    archives=""
    for target in "$@"; do
        cargo_build -p uemulator-apple --target "$target"
        archives="$archives target/$target/release/$library"
    done
    mkdir -p "$(dirname "$fat_output")"
    # shellcheck disable=SC2086 # one word per archive
    lipo -create $archives -output "$fat_output"
}

case "$1" in
    --xcode)
        targets=""
        for arch in $ARCHS; do targets="$targets $(rust_target "$PLATFORM_NAME" "$arch")"; done
        # shellcheck disable=SC2086
        fat_library "$BUILT_PRODUCTS_DIR/$library" $targets
        firmware
        ;;
    xcframework)
        out=target/apple
        rm -rf "$out/UemulatorCore.xcframework"
        fat_library "$out/macos/$library" aarch64-apple-darwin x86_64-apple-darwin
        fat_library "$out/ios/$library" aarch64-apple-ios
        fat_library "$out/ios-simulator/$library" aarch64-apple-ios-sim
        headers=crates/uemulator-apple/include
        xcodebuild -create-xcframework \
            -library "$out/macos/$library" -headers "$headers" \
            -library "$out/ios/$library" -headers "$headers" \
            -library "$out/ios-simulator/$library" -headers "$headers" \
            -output "$out/UemulatorCore.xcframework"
        firmware
        echo "built $repo/$out/UemulatorCore.xcframework"
        ;;
    *)
        echo "usage: $0 --xcode | xcframework" >&2
        exit 2
        ;;
esac
