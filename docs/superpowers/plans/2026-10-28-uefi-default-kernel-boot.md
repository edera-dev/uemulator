# UEFI by default, kernels booted through the firmware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A machine's firmware defaults to UEFI, and a kernel (and initramfs) given to a UEFI machine is booted by the firmware the way QEMU's `-kernel` works under OVMF.

**Architecture:** The library places the kernel, initramfs and command line in guest RAM next to the UEFI firmware and describes them in new handoff-page fields. The firmware reserves them, boots the kernel first (`LoadImage` from the buffer, command line as load options, initramfs through a `LINUX_EFI_INITRD_MEDIA_GUID` `LoadFile2` protocol), and never falls through to the disks if that fails. The CLI gains `--firmware none` (today's direct boot), makes `uefi` the default, and sends installer boots through the firmware too.

**Tech Stack:** Rust workspace; `crates/uemulator-firmware` (`no_std`, `x86_64-unknown-none`), `crates/uemulator` (library), `crates/uemulator-cli`.

**Spec:** agreed in chat on 2026-10-28 (no written document); the decisions are copied into Global Constraints.

## Global Constraints

- `--firmware` values: `uefi` (new default), `bios`, `none`. `none` is today's direct 64-bit boot of `--kernel`/`--initramfs`. `bios` boots disks only.
- `quick boot` keeps `--boot-from auto`: the disk if something is installed, else the installer. Both go through the machine's firmware. `quick boot` still has no `--firmware` flag (create/edit only; the existing test `firmware_is_on_create_and_edit_only` stays true).
- Booting a kernel through UEFI follows QEMU+OVMF: the supplied kernel is tried before `BootNext`/`BootOrder`/disks.
- A kernel without an EFI stub under `uefi` fails with a clear error that names `--firmware none`; no silent fallback to direct boot.
- A `machine.toml` with no `firmware` key loads as `uefi`; an explicit `firmware = "bios"` stays `bios`.
- `advanced boot-uefi` and `advanced boot-firmware` are unchanged. `advanced boot` gains `--firmware`.
- Repo rules (CLAUDE.md): library never touches stdio or panics on guest behavior; named constants, no magic numbers; comments only for the surprising; fix bugs test-first; before EVERY commit run `./hack/autofix.sh` and make `cargo clippy --release --all-targets` print zero warnings (also `-p uemulator-web --target wasm32-unknown-unknown` and `-p uemulator-uefi --target x86_64-unknown-uefi`, because the library changes).
- Commit trailers: end every commit message with
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_012mRbTLUqgnQwa7aTi64vPh`.

## Review Focus

- Installer boot under UEFI must keep the machine's nvram: `grub-install`/`efibootmgr` write `Boot####` through efivarfs and the installed system relies on them (Task 4 test).
- A guest reset re-runs the firmware and must place the kernel payload again (Task 2 test).
- Payload larger than RAM, or an initramfs that leaves no room, is an error from `EmulatedMachine::new`, not a panic or a corrupt boot (Task 2 test).
- A bzImage that is not an EFI stub (and the library's `FakeKernel`) is refused under UEFI with the `--firmware none` hint (Task 2 test).
- `--firmware none` on a machine whose disk has an install cannot boot the disk and must say so instead of starting the installer (Task 4 test).
- An existing `bios` machine with an install still boots through the BIOS (Task 4 test).

---

### Task 1: Firmware: payload fields, reservation, kernel boot entry

**Files:**
- Modify: `crates/uemulator-firmware/src/handoff.rs` (fields, parse, tests)
- Modify: `crates/uemulator-firmware/src/memory.rs` (`init`: reserve payload ranges)
- Create: `crates/uemulator-firmware/src/kernel_boot.rs` (target_os = "none" only, like `boot.rs`; registered in `main.rs`)
- Modify: `crates/uemulator-firmware/src/boot.rs` (`run`: kernel first, failure halts), `crates/uemulator-firmware/src/uefi.rs` (GUIDs)

**Interfaces:**
- Produces (handoff layout, shared with Task 2; `SIZE` grows from `0x308` to `0x338`, version stays 1 because library and firmware ship together):
  `0x308` kernel base u64, `0x310` kernel size u64, `0x318` initrd base u64, `0x320` initrd size u64, `0x328` command line base u64, `0x330` command line size u64 (UTF-8, no NUL). A size of 0 means absent; no kernel means no payload at all.
- Produces: `pub struct Payload { pub kernel: (u64, u64), pub initrd: Option<(u64, u64)>, pub cmdline: (u64, u64) }` and `pub payload: Option<Payload>` on `Handoff` (`Some` iff kernel size != 0).
- Produces: `kernel_boot::run(payload: &Payload) -> Status` — boots it, returns only on failure.

- [ ] **Step 1: Write failing tests in `handoff.rs`**: `parses_a_payload` (page with kernel (0x400_0000, 0x1000), initrd (0x500_0000, 0x2000), cmdline (0x3ff_0000, 12) gives `payload == Some(Payload{..})`), `no_kernel_means_no_payload`, and extend `refuses_a_page_that_is_not_ours` with a page of `SIZE - 1` bytes. Existing tests must use the new `SIZE`.
- [ ] **Step 2: Run** `cargo test --release -p uemulator-firmware handoff` — Expected: FAIL (no `payload`).
- [ ] **Step 3: Implement** the offsets as named consts and `Handoff::parse`.
- [ ] **Step 4: Run again** — Expected: PASS.
- [ ] **Step 5: `memory::init`** marks each payload range (kernel, initrd, command line) `BOOT_SERVICES_DATA`, `WRITE_BACK`, like the low-pages mark, so nothing allocates over them and Linux reclaims them after `ExitBootServices`. Ranges lie inside usable RAM; use the existing `mark` closure.
- [ ] **Step 6: `kernel_boot.rs`**: `run` converts the command line to UTF-16 with a trailing NUL (`text::to_utf16_nul`), calls `LoadImage(boot_policy 0, source = kernel buffer, device path = DevicePath::new().file_path("kernel"))`, `images::set_load_options`, sets no `BootCurrent`. If an initrd exists first install a new handle carrying a vendor-media device path (`LINUX_EFI_INITRD_MEDIA_GUID` = `5568e427-68fc-4f3d-ac74-ca555231cc68`, end node) and `EFI_LOAD_FILE2_PROTOCOL` (`4006c0c1-fcb3-403e-996d-4a6c8724e06d`) whose `LoadFile` returns `UNSUPPORTED` for `boot_policy != 0`, `BUFFER_TOO_SMALL` with the size set when the buffer is null or short, else copies the initrd. Uninstall both after `StartImage` returns. Add the two GUIDs to `uefi.rs::guids`. A `LoadImage` failure prints `kernel is not an EFI-stub image (status {:#x}); boot it with --firmware none` and returns the status.
- [ ] **Step 7: `boot::run`**: if `handoff.payload` is set, call `kernel_boot::run` before `candidates()`; when it returns, print `kernel returned with status {status:#x}` and `halt_forever()` (a requested kernel never falls through to the disks). `boot::run` needs the payload, so store `Handoff` where `services::init` can already reach it, or pass `&handoff` into `boot::run` from `firmware_main` (builder's choice; update `main.rs`).
- [ ] **Step 8: Build** `cargo build --release -p uemulator-cli` (its `build.rs` builds the firmware) and `cargo test --release -p uemulator-firmware`. Expected: both succeed.
- [ ] **Step 9: Commit** `firmware: boot a kernel the library placed, initramfs through LoadFile2`.

### Task 2: Library: place the payload, accept a kernel with UEFI

**Files:**
- Modify: `crates/uemulator/src/boot/uefi.rs` (`load_uefi_firmware`, tests), `crates/uemulator/src/boot.rs` (re-export if needed), `crates/uemulator/src/machine.rs` (validation at ~854, `new` at ~885, reboot at ~639, docs of `uefi_firmware`/`kernel`)
- Test: `crates/uemulator/src/boot/uefi.rs` tests, `crates/uemulator/src/machine.rs` tests

**Interfaces:**
- Consumes: handoff offsets from Task 1.
- Produces: `load_uefi_firmware(ram, image, params, acpi, payload: Option<&UefiPayload>) -> Result<BootState, Error>` with `pub(crate) struct UefiPayload<'a> { kernel: &'a [u8], initrd: Option<&'a [u8]>, cmdline: &'a str }`. Placement: initramfs 4 KiB-aligned down from `ram_top`, then the kernel, then the command line below it, each its own page-aligned block, all above `FIRMWARE_BASE + footprint`; else `Error::BadConfig("kernel and initramfs do not fit in guest memory")`. Offsets are written to the handoff.
- Produces: `MachineConfig::uefi_firmware` now combines with `kernel`, `initrd`, `cmdline`, `cmdline_append`; still not with `firmware` or `boot_disk`. A non-empty kernel without an MZ header and a `PE\0\0` signature at the offset stored at `0x3c` fails with `Error::BadConfig("kernel has no EFI stub; boot it with firmware none")` (phrase: keep `firmware none` verbatim, the CLI test greps for it).

- [ ] **Step 1: Failing tests** in `boot/uefi.rs`: `payload_is_placed_at_the_top_of_ram_and_described_in_the_handoff` (read back bytes at the handoff addresses; ranges do not overlap each other or the firmware), `no_payload_leaves_the_handoff_fields_zero`, `a_payload_that_does_not_fit_is_refused`. In `machine.rs`: `uefi_accepts_a_kernel_with_an_efi_header` (build with a tiny MZ/PE blob, `EmulatedMachine::new` is Ok), `uefi_refuses_a_kernel_without_an_efi_stub` (`testkit::FakeKernel` bytes; the error text contains `firmware none`), `a_guest_reset_places_the_payload_again` (use the existing reset-test style: run, reset, read RAM at the handoff's kernel base).
- [ ] **Step 2: Run** `cargo test --release -p uemulator uefi` — Expected: FAIL.
- [ ] **Step 3: Implement** placement and handoff fields in `uefi.rs` with named consts for offsets and alignment; extend the call at both sites (`new` and the reboot path, which already holds `image.kernel/initrd/cmdline`); relax the validation; add the EFI-header check as a small `fn has_efi_header(kernel: &[u8]) -> bool`.
- [ ] **Step 4: Run** `cargo test --release -p uemulator` and `cargo test --release -p uemulator --no-default-features` — Expected: PASS.
- [ ] **Step 5: Commit** `library: UEFI machines can take a kernel and initramfs for the firmware to boot`.

### Task 3: End-to-end: the real firmware boots a tiny EFI kernel

**Files:**
- Test: `crates/uemulator-cli/src/firmware.rs` (tests; the CLI embeds the real firmware image)
- Possibly modify: `crates/uemulator-cli/src/test_support.rs`

**Interfaces:**
- Consumes: `uemulator::MachineConfig::uefi_firmware(crate::firmware::IMAGE).kernel(..).initrd(..).cmdline(..)` from Task 2.

- [ ] **Step 1: Failing test** `uefi_firmware_boots_the_kernel_it_is_given`: a test helper builds a minimal PE32+ x86-64 `EFI application` (MZ header, `PE\0\0` at `0x3c`, one `.text` section) whose entry (`rcx` = image handle, `rdx` = system table) calls `HandleProtocol(rcx, LoadedImage GUID)`, writes the UTF-16 `LoadOptions` byte by byte to port `0x3f8`, then writes `"|initrd:"` plus the initramfs bytes obtained by calling the `LINUX_EFI_INITRD_MEDIA_GUID` `LoadFile2` through `LocateDevicePath`... — builder: if reading the initrd from tiny assembly proves unreasonable, assert only the command line and cover the initramfs by Task 5's real-kernel check, and say so in the test comment. Assemble the machine code with `echo '...' | hack/asm.sh`. Assertions: serial output contains the exact cmdline; machine ends with `ExitReason::Shutdown` (payload `hlt`s with IF=0); a second test `a_kernel_that_is_not_efi_is_refused_before_boot` asserts the `firmware none` error without running the firmware.
- [ ] **Step 2: Run** `cargo test --release -p uemulator-cli uefi_firmware_boots` — Expected: FAIL until Task 1+2 are in; if they are, the first run should reveal real firmware bugs: fix them in the firmware (test-first with a handoff/unit test where the logic is hardware-independent).
- [ ] **Step 3: Make it pass.**
- [ ] **Step 4: Commit** `tests: boot a hand-built EFI kernel through the real firmware`.

### Task 4: CLI and machines: `--firmware uefi|bios|none`, UEFI default

**Files:**
- Modify: `crates/uemulator-cli/src/args.rs` (`Firmware`: add `None`, `#[default]` moves to `Uefi`, `name()`, doc comments, `FirmwareArgs` help "[default: uefi]", `BootSource::Disk` doc), `crates/uemulator-cli/src/main.rs` (`advanced boot` gets `#[command(flatten)] firmware: FirmwareArgs`; `boot_launch`; parser tests), `crates/uemulator-cli/src/run.rs` (`run_uefi` takes an optional kernel+initramfs; add `run_kernel(kernel, initramfs, settings, runtime, firmware, nvram)` choosing direct or UEFI), `crates/uemulator-cli/src/machines.rs` (`prepare_boot`/`Launch`, `describe`, tests at ~1647-1731)

**Interfaces:**
- Produces: `args::Firmware::{Uefi, Bios, None}`, default `Uefi`.
- Behavior: installer boot goes through `run::run_uefi(..., kernel: Some((kernel, initramfs)))` when firmware is `Uefi`, directly when `None`, and is an error `"--firmware bios boots a disk, not a kernel; use uefi or none"` when `Bios`... except `quick boot` of a `bios` machine whose disk is blank, which keeps booting the installer directly (that is today's behavior for existing bios machines; keep it and say so in the help text). `Launch.nvram` is `Some(machine.nvram_path())` for every `Uefi` launch, installer included. A disk boot with firmware `None` is an error: `"firmware none cannot boot a disk; edit the machine to uefi or bios"`. `advanced boot --firmware uefi` (default) with no nvram.

- [ ] **Step 1: Failing tests**: flip `firmware_defaults_to_bios_and_old_files_without_it_still_load` to `firmware_defaults_to_uefi_and_old_files_without_it_load_as_uefi` (a file without the key loads `Uefi`; the saved text now has `firmware = "uefi"`; a file with `firmware = "bios"` stays `Bios`); `an_installer_boot_of_a_uefi_machine_keeps_the_nvram` (launch of a blank-disk uefi machine has `nvram == Some(..)`); `a_bios_machine_still_boots_its_installed_disk_through_the_bios`; `none_cannot_boot_an_installed_disk`; parser tests: `advanced boot --kernel k --firmware none` parses, default is uefi, `quick boot m --firmware x` is still an error; `show` prints `firmware  uefi` and the nvram path.
- [ ] **Step 2: Run** `cargo test --release -p uemulator-cli` — Expected: FAIL.
- [ ] **Step 3: Implement** per the Behavior paragraph above. `--verify` requires direct 64-bit code: keep it working under `uefi` if the library allows it (check in Step 4), otherwise make the CLI error say `use --firmware none`.
- [ ] **Step 4: Run** `cargo test --release --workspace` and `cargo test --release -p uemulator --no-default-features`. Also run a `--verify` boot of the Task 3 kernel; record the outcome in the commit message.
- [ ] **Step 5: Commit** `cli: --firmware uefi|bios|none, UEFI by default, installers boot through the firmware`.

### Task 5: Docs and real-guest verification

**Files:**
- Modify: `CLAUDE.md` (Commands: `--firmware` default, kernels through UEFI, `none`), `docs/plans/19-uefi-firmware.md` (handoff table rows at `0x308..0x330`, boot-manager paragraph, verification list), `docs/plans/10-milestones.md` (UEFI firmware section), `docs/plans/02-boot.md` or `docs/plans/01-api.md` if they describe the config combinations

- [ ] **Step 1: Verify by hand** (not tests, need artifacts): `cargo run --release -- quick create t1 --install alpine` then `quick boot t1` reaches the installer on `hvc0` through UEFI (serial shows `uemulator UEFI firmware`, kernel prints `efi: EFI v2.9 by uemulator`); `/sys/firmware/efi` exists; complete an Alpine install, `quick boot t1` boots the disk. Repeat with `--install debian` far enough to confirm the installer finds its initramfs and an ESP. `advanced boot --kernel K --initramfs I --firmware none` still boots as before. Report anything that fails instead of papering over it; installer memory defaults (`Distro::default_memory_mib`) may need a bump for the extra initramfs copy.
- [ ] **Step 2: Update docs** to what was verified, including that an old keyless `machine.toml` of a BIOS-installed machine now needs `firmware = "bios"` (the user chose this).
- [ ] **Step 3: Zero warnings** on all four clippy invocations listed in Global Constraints, after `./hack/autofix.sh`.
- [ ] **Step 4: Commit** `docs: UEFI is the default firmware and boots kernels`.
