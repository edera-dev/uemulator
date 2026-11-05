//! Builds the UEFI firmware (`crates/uemulator-firmware`, for `x86_64-unknown-none`) into a
//! target directory of its own and leaves the flat image where `src/firmware.rs` embeds it.
//! The separate directory matters: this runs inside a cargo build that holds the lock on the
//! workspace's own.

use std::path::{Path, PathBuf};
use std::process::Command;

const FIRMWARE_TARGET: &str = "x86_64-unknown-none";
const FIRMWARE_PACKAGE: &str = "uemulator-firmware";

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let firmware_dir = manifest_dir.join("../uemulator-firmware");
    for watched in ["src", "Cargo.toml", "link.ld", "build.rs", "secure-boot"] {
        println!(
            "cargo:rerun-if-changed={}",
            firmware_dir.join(watched).display()
        );
    }
    // The firmware's own dependency: a change there must rebuild the image too.
    let crypto_dir = manifest_dir.join("../uemulator-crypto");
    for watched in ["src", "Cargo.toml"] {
        println!(
            "cargo:rerun-if-changed={}",
            crypto_dir.join(watched).display()
        );
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("firmware-target");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut build = Command::new(cargo);
    build
        .args([
            "build",
            "--release",
            "-p",
            FIRMWARE_PACKAGE,
            "--target",
            FIRMWARE_TARGET,
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .current_dir(&manifest_dir)
        // Settings aimed at building this crate must not leak into the firmware's build.
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTC_WRAPPER");
    // For debugging the firmware: progress messages on the serial port.
    println!("cargo:rerun-if-env-changed=UEMULATOR_FIRMWARE_TRACE");
    if std::env::var_os("UEMULATOR_FIRMWARE_TRACE").is_some() {
        build.args(["--features", "uemulator-firmware/trace"]);
    }
    let status = build.status();
    if !matches!(status, Ok(status) if status.success()) {
        eprintln!(
            "\nerror: building the UEFI firmware failed. If the target is missing, install it with:\n\n    \
             rustup target add {FIRMWARE_TARGET}\n"
        );
        std::process::exit(1);
    }
    let image = target_dir
        .join(FIRMWARE_TARGET)
        .join("release")
        .join(FIRMWARE_PACKAGE);
    copy(&image, &out_dir.join("uemulator-firmware.bin"));
}

fn copy(from: &Path, to: &Path) {
    std::fs::copy(from, to)
        .unwrap_or_else(|e| panic!("copy {} to {}: {e}", from.display(), to.display()));
}
