# uemulator

uemulator is an x86_64 PC emulator written in Rust, built to be embedded. It boots unmodified distribution kernels (Alpine, Debian, Fedora, and Xen with a Linux dom0) on an emulated AMD Zen 4 machine, through its own UEFI firmware. The design follows QEMU's TCG: guest code is translated into cached blocks and run by an interpreter or by a JIT for the host. The same library runs as a command-line tool, in a browser tab, or as a UEFI application. Its public API is small and centred on `MachineConfig` and `EmulatedMachine`, and it can be built with no dependencies at all.

**[Try it in the browser](https://edera-dev.github.io/uemulator/web/)**: Alpine Linux boots in a Web Worker.

## Features

- **Real guests.** Boots stock Alpine, Debian installer and Fedora kernels, Xen 4.21 with a dom0 and domU, and Linux KVM running nested inside the guest.
- **Modern CPU model.** AMD Zen 4 (EPYC Genoa) with AVX2 and AVX-512, AES-NI and SHA, and AMD SVM with nested paging. Real mode, protected mode and compatibility mode are all supported.
- **Your choice of firmware.** The built-in UEFI firmware (the default), a built-in BIOS for legacy disk boot, or direct kernel entry with no firmware.
- **SMP.** Up to 64 vCPUs. Each one runs on its own host thread, or all of them share one thread and run deterministically.
- **A full PC platform.** ACPI, local APIC and x2APIC, I/O APIC, HPET, PCIe with MSI-X and an AMD IOMMU.
- **virtio devices.** Block, network (with built-in user-mode NAT), console and entropy.
- **Its own disk format.** Images are sparse and copy-on-write, and can be streamed over HTTP.
- **Three JITs.** aarch64, x86_64 and WebAssembly hosts get a JIT, and every JIT is checked against the interpreter. The interpreter runs everywhere else.
- **Embeddable.** No global state, and no stdio or process exit inside the library. Built `no_std` for freestanding hosts. With `default-features = false` it has zero dependencies.

## Quick start

You need a recent stable Rust toolchain (edition 2024) and the target the built-in firmware is compiled for:

```sh
rustup target add x86_64-unknown-none
git clone https://github.com/edera-dev/uemulator.git
cd uemulator
cargo build --release -p uemulator-cli    # produces target/release/uemulator
```

Create a named machine and boot it. `quick create` downloads the distribution's netboot kernel and initramfs, so it needs network access. It also gives the machine a sparse disk and a NAT network card.

```sh
cargo run --release -p uemulator-cli -- quick create test --install alpine   # or debian, fedora
cargo run --release -p uemulator-cli -- quick boot test --cpus 4
```

The first boot runs the distribution's installer. Once something is installed on the disk, `quick boot` boots the disk through the machine's firmware. Press `Ctrl-]` to stop the machine. Machines live in `~/.uemulator/machines/<name>/`, and setting `UEMULATOR_HOME` moves that directory.

Other everyday commands, shown using the built binary (`target/release/uemulator`):

```sh
uemulator quick list                          # list machines
uemulator quick show test                     # print a machine's configuration
uemulator quick edit test --memory 4096       # change it
uemulator quick reinstall test --install debian   # new distro, same disk
uemulator quick delete test --yes

uemulator advanced boot --kernel bzImage --initramfs initrd.img   # your own kernel, through UEFI
uemulator advanced boot --kernel bzImage --firmware none          # ... or entered directly (no EFI stub needed)
uemulator disk create disk.udisk --size 8192                      # create an empty disk image
uemulator disk info disk.udisk
```

Every command takes `--help`, which lists all of its options.

### In the browser

```sh
rustup target add wasm32-unknown-unknown x86_64-unknown-none
./hack/web.sh serve                           # http://127.0.0.1:8080/
```

Setting `WEB_DEMO=1` also builds a ready-to-boot Alpine machine and puts the page in one-button demo mode. Because the page needs cross-origin isolation, serve it with `hack/web/serve.py` (which `web.sh serve` uses) or with a server that sends the COOP/COEP headers.

## Embedding

Everything public is re-exported from [`crates/uemulator/src/lib.rs`](crates/uemulator/src/lib.rs): build a `MachineConfig`, create an `EmulatedMachine`, start it, and wait for it to exit.

```toml
[dependencies]
uemulator = { git = "https://github.com/edera-dev/uemulator" }
```

```rust
use std::io::Write;

use uemulator::{EmulatedMachine, MachineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = MachineConfig::new()
        .kernel(std::fs::read("bzImage")?)
        .initrd(std::fs::read("initramfs")?)
        .memory_mib(1024)
        .cpus(2);
    let mut machine = EmulatedMachine::new(config)?;

    // Guest console output (hvc0). Keystrokes go in with `machine.console().write(..)`.
    machine.console().on_output(|bytes| {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(bytes);
        let _ = stdout.flush();
    });

    // Call `stop()` on this from any thread to end the run.
    let _stop = machine.stop_handle();

    machine.start()?;
    let reason = machine.wait(); // Shutdown, Reset, TripleFault, Stopped or Internal
    println!("\nmachine exited: {reason:?}");
    Ok(())
}
```

If no firmware is configured, the library enters the kernel directly through the Linux boot protocol. To boot the kernel through UEFI instead, pass the firmware image built from `crates/uemulator-firmware` to `MachineConfig::uefi_firmware`. The builder also covers disks (`disk`, which can wrap your own `BlockBackend`), network cards (`net`), the IOMMU, the clock and the JIT. On a freestanding host, the `poll` and `host_clock` hooks connect the machine to the outside world. On macOS, a hardened-runtime binary needs the `com.apple.security.cs.allow-jit` entitlement to use the JIT.

## Documentation

- [`docs/`](docs/README.md) is the documentation index, grouped by topic.
- [`docs/plans/00-overview.md`](docs/plans/00-overview.md) describes the architecture and the key decisions behind it.
- [`docs/plans/10-milestones.md`](docs/plans/10-milestones.md) records what works and how each piece was verified.
- [`docs/spec/uemulator-disk-format.md`](docs/spec/uemulator-disk-format.md) specifies the udisk image format.

## Development

```sh
cargo test --release --workspace
cargo test --release -p uemulator --no-default-features   # interpreter only, no dependencies
cargo clippy --release --all-targets                      # must print zero warnings
./hack/autofix.sh                                         # clippy --fix and rustfmt; run before every commit
```

Tests are self-contained: they need no network and no downloaded kernels. [`CLAUDE.md`](CLAUDE.md) holds the project's conventions, including commenting, naming, the rule that the interpreter is the oracle for every JIT, and the real-guest checks to run after touching particular subsystems. [`docs/plans/08-testing.md`](docs/plans/08-testing.md) covers the test strategy.

## License

GPL-2.0. See [`LICENSE`](LICENSE).
