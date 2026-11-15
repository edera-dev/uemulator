#!/bin/sh
# Xcode Cloud runs this after it clones the repository (it looks for ci_scripts/ beside the
# .xcodeproj; docs/plans/28-apple.md, "Xcode Cloud"). Its machines have Xcode and nothing else,
# and the app's build phases need three things from outside Xcode:
#
#   - Rust with every target hack/apple/build-core.sh may be asked for. An archive builds macOS
#     for arm64 and x86_64 and iOS for arm64; a generic simulator build may ask for x86_64 too.
#     The firmware is x86_64-unknown-none, the WebKit engine's module wasm32-unknown-unknown.
#     build-core.sh runs cargo from $HOME/.cargo/bin, so rustup goes in the default place.
#   - target/web and its demo/ (`WEB_DEMO=1 hack/web.sh`), which hack/apple/bundle-app.sh copies
#     into the app. demo.py downloads Alpine and Edera's Xen from the internet every time, since
#     Xcode Cloud keeps no files between builds.
#   - Trust in SwiftTerm's build plugin. Locally that is -skipPackagePluginValidation or a click
#     in Xcode; Xcode Cloud takes neither, only this default (its name is misspelt in Xcode).
#
# Run by hand outside Xcode Cloud it does the first two and leaves Xcode's defaults alone.
set -e
repo=${CI_PRIMARY_REPOSITORY_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$repo"

rust_targets="aarch64-apple-darwin x86_64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
x86_64-apple-ios x86_64-unknown-none wasm32-unknown-unknown"

# Checked by path, not on PATH: build-core.sh uses only $HOME/.cargo/bin, and a rustup elsewhere
# (Homebrew's) may have no toolchain for this user.
PATH="$HOME/.cargo/bin:$PATH"
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
    curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
fi
# shellcheck disable=SC2086 # one word per target
rustup target add $rust_targets

WEB_DEMO=1 ./hack/web.sh

if [ -n "$CI_PRIMARY_REPOSITORY_PATH" ]; then
    defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidatation -bool YES
fi
