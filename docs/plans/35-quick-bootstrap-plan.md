# `quick bootstrap` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** done (the plan for `35-quick-bootstrap.md`; what the real installs showed is recorded there, under "Checked").

**Goal:** `uemulator quick bootstrap NAME --install DISTRO --user U --ask-password` and the Apple
app's Install sheet install Alpine, Debian or Fedora unattended, then boot the installed system.

**Architecture:** A new dependency-free crate, `uemulator-bootstrap`, turns validated answers into
an answer file for each distro. It packs that file into a newc cpio archive, which is appended to
the netboot initramfs, and adds the kernel arguments that point the installer at it. The CLI
drives an installer run with the library's in-place reboot turned off. A completion marker on
the console decides whether the reset after the install boots the disk or ends the run with an
error. The app reaches the same crate through `uemulator-apple`'s C ABI and extends
`MachineSettings.initrd` before either engine starts.

**Tech Stack:** Rust 2024 (workspace crates), clap in the CLI, SwiftUI in `apple/`.

**Spec:** `docs/plans/35-quick-bootstrap.md` (read it first; this plan argues from it).

## Global Constraints

- `uemulator-bootstrap` has no third-party dependencies. Its only dependency is
  `uemulator-crypto`, for SHA-512.
- No password and no crypt hash in any file uemulator writes on purpose: `machine.toml`, the
  app's sidecar, logs, or files under the machine directory. The appended initramfs exists only
  in memory.
- Password hashes are SHA-512 crypt (`$6$`), with a 16-character salt from the caller's random
  source. The default is 5000 rounds, and the hash carries no `rounds=` prefix.
- Completion marker: the line `uemulator-bootstrap: installed`, as `pub const COMPLETION_MARKER`.
  It must never appear verbatim in any rendered file (scripts build it from two parts).
- `quick bootstrap` refuses `--firmware none` and `--hypervisor`.
- An unattended `quick reinstall` asks `Erase <disk file> (<size>) and install <Distro title>?
  [y/N]` on a terminal. `--yes` skips the question, and it is required without a terminal.
- Snapshots, checkpoints and `--debug` stay allowed on bootstrap runs.
- There is no inactivity timeout. A stuck installer just waits on the console.
- CLAUDE.md applies: tests next to the code with no network, `./hack/autofix.sh`, and zero
  clippy warnings, which includes `--target aarch64-apple-ios -p uemulator-apple`. Comments
  follow the repo's style.

## Review Focus

1. **Answer values that would inject into an answer file.** A hostname with a newline, a package
   named `vim; rm -rf /`, or a user `root` must be rejected by `Answers::validate` with a message
   that names the field, before anything renders (Task 3).
2. **The marker split across two console output chunks.** It must still be seen (Task 5).
3. **Ctrl-] or `--timeout` during the install.** The run ends as it does today: no error about
   the marker, `boot_from` stays `installer`, and no boot of the disk follows (Task 6).
4. **`quick reinstall NAME` on a machine with a `[bootstrap]` table, stdin not a terminal, and no
   password option.** It must fail with a message naming `--password-file`, and the disk must be
   untouched (Task 7).
5. **A gzip-compressed initramfs whose length is not a multiple of 4.** The appended archive
   starts on a 4-byte boundary, zero-padded, as the kernel's unpacker requires (Task 2).

---

### Task 1: Alpine overlay spike (throwaway)

The spec's open risk. This task produces no code to keep, only a decision written into the spec.

**Files:**
- Modify: `docs/plans/35-quick-bootstrap.md` ("Delivery per distro", Alpine bullet; "Left to the builder")

- [ ] **Step 1:** Read the init script in the current Alpine netboot `initramfs-virt`, unpacked into a
  scratch directory. Find how `apkovl=` is resolved when it is not an `http(s)://` or `ftp://`
  URL.
- [ ] **Step 2:** Make a test cpio by hand, holding `/bootstrap.apkovl.tar.gz` (an overlay with
  `etc/local.d/bootstrap.start` that prints `hello` to `/dev/console`, and
  `etc/runlevels/default/local`). Append it to `initramfs-virt` and boot it with `advanced boot
  --kernel vmlinuz-virt --initramfs <extended> --append "<Distro::Alpine cmdline>
  apkovl=/bootstrap.apkovl.tar.gz"`. Record whether `hello` prints.
- [ ] **Step 3:** If it doesn't, try the next candidate: an archive file that replaces the init's
  overlay-search step. Stop after two failed candidates and bring the findings back to the user,
  as the spec says.
- [ ] **Step 4:** Write the mechanism that worked into the spec's Alpine bullet: the exact kernel
  argument and file paths. Commit with `docs: record Alpine's bootstrap overlay hook`.

### Task 2: `uemulator-bootstrap` crate, SHA-512 crypt, cpio writer

**Files:**
- Create: `crates/uemulator-bootstrap/Cargo.toml`, `src/lib.rs`, `src/crypt.rs`, `src/cpio.rs`
- Modify: `Cargo.toml` (workspace `members`)

**Interfaces:**
- Produces: `pub fn sha512_crypt(password: &str, salt: &str) -> String` (salt is up to 16
  characters from `./0-9A-Za-z`; longer is truncated, as glibc does).
- Produces: `pub const SALT_LEN: usize = 16`, and `pub fn salt_from_bytes(random: &[u8;
  SALT_LEN]) -> String`, which maps each byte onto the crypt alphabet.
- Produces: `pub struct CpioFile { pub path: String, pub mode: u32, pub data: Vec<u8> }`,
  where `path` has no leading `/` and parent directories are listed as their own entries with an
  `S_IFDIR` mode.
- Produces: `pub fn append_to_initramfs(base: &[u8], files: &[CpioFile]) -> Vec<u8>`.

- [ ] **Step 1: Write the failing tests.** In `crypt.rs`, `matches_drepper_vectors`: the
  default-rounds vectors from Ulrich Drepper's SHA-crypt spec, copied from the spec text and not
  retyped. The first is `sha512_crypt("Hello world!", "saltstring") ==
  "$6$saltstring$svn8UoSVapNtMuq1ukKS4tPQd8iKwSMHWjl/O817G3uBnIFNjnQJuesI68u4OTLiBFdcbYEdFCoEOfaS35inz1"`.
  Also `salt_is_truncated_to_sixteen` and `salt_alphabet_only`. In `cpio.rs`:
  `appended_archive_starts_on_four_byte_boundary` (base lengths 0 to 7: the result starts with
  `base`, followed by zero bytes up to a multiple of 4, then `070701`), `newc_round_trip` (a test
  parser reads back the paths, modes and data, and ends at `TRAILER!!!`), and
  `entries_are_padded_to_four_bytes`.
- [ ] **Step 2:** Run `cargo test --release -p uemulator-bootstrap`. Expected: FAIL (unresolved
  functions).
- [ ] **Step 3:** Implement. For SHA-512 crypt, follow Drepper's algorithm on
  `uemulator_crypto::sha::Sha512` (steps 1 to 22 of the spec, then the custom base-64 byte order).
  Write the cpio in newc format with ino counting from 1, nlink 1 (2 for directories), uid and gid
  0, mtime 0, and the trailer.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run `./hack/autofix.sh` and clippy. Commit with `bootstrap: SHA-512 crypt and
  initramfs cpio appending`.

### Task 3: Answers, validation, and the three renderers

**Files:**
- Create: `crates/uemulator-bootstrap/src/answers.rs`, `src/debian.rs`, `src/fedora.rs`, `src/alpine.rs`
- Modify: `crates/uemulator-bootstrap/src/lib.rs`

**Interfaces:**
- Consumes: `CpioFile` (Task 2). The Alpine mechanism comes from Task 1's spec update.
- Produces:
  ```rust
  pub enum Distro { Debian, Alpine, Fedora }
  pub struct Answers {
      pub user: String,
      pub password_hash: String,              // "$6$..."
      pub root_password_hash: Option<String>, // None: root locked
      pub hostname: String,
      pub ssh_keys: Vec<String>,              // one OpenSSH public key line each
      pub timezone: String,
      pub locale: String,
      pub keymap: String,
      pub packages: Vec<String>,
  }
  impl Answers { pub fn validate(&self) -> Result<(), String> }
  pub struct Payload { pub files: Vec<CpioFile>, pub cmdline: String }
  pub fn render(distro: Distro, answers: &Answers) -> Result<Payload, String> // validates first
  pub const COMPLETION_MARKER: &str = "uemulator-bootstrap: installed";
  ```
- `Payload.cmdline` is appended after the distro's installer arguments:
  - Debian: `auto=true priority=critical`, with the preseed at `preseed.cfg`.
  - Fedora: `inst.ks=file:/ks.cfg`, with the kickstart at `ks.cfg`.
  - Alpine: whatever Task 1 recorded.

- [ ] **Step 1: Write the failing tests.**
  - `validate` rejects, and the error names the field:
    - a user that is empty, `root`, longer than 32 characters, or doesn't match
      `[a-z_][a-z0-9_-]*`
    - a hostname that isn't an RFC 1123 label of up to 63 characters
    - a package outside `[A-Za-z0-9+._-]`
    - a timezone, locale or keymap outside `[A-Za-z0-9_+./@-]`
    - an SSH key containing `\n` or `\r`
    - a hash that doesn't start with `$6$`
  - Golden renders per distro, one with every field set and one minimal: compare against
    `testdata/<distro>-full.*` and `testdata/<distro>-minimal.*`, written by hand from the spec's
    per-distro bullets.
  - `no_output_contains_the_marker_verbatim`, across all renders.
  - `root_locked_when_no_root_password`:
    - Debian has `passwd/root-login boolean false`
    - Fedora has `rootpw --lock`
    - Alpine leaves root with no password set
  - `ssh_key_enables_openssh` per distro.
  - `cmdline_has_no_stray_whitespace`.
- [ ] **Step 2:** Run `cargo test --release -p uemulator-bootstrap`. Expected: FAIL.
- [ ] **Step 3:** Implement the renderers to the spec's bullets. The marker line is printed to
  `/dev/console` as the last step before reboot: Debian's `preseed/late_command`, Fedora's
  `%post --nochroot`, and the end of Alpine's script. It is always built as `printf '%s: %s\n'
  uemulator-bootstrap installed`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run autofix and clippy. Commit with `bootstrap: answer files for Debian, Fedora
  and Alpine`.

### Task 4: CLI arguments, host defaults, passwords, `[bootstrap]` in machine.toml

**Files:**
- Create: `crates/uemulator-cli/src/bootstrap.rs` (host defaults, password reading, the saved
  record, the marker watch from Task 5)
- Modify: `crates/uemulator-cli/Cargo.toml` (adds `uemulator-bootstrap` and `getrandom`),
  `src/args.rs`, `src/main.rs` (the `QuickCmd::Bootstrap` variant), `src/machines.rs`
  (`MachineFile` and `Machine` get `bootstrap: Option<BootstrapRecord>`)
- Modify: `crates/uemulator-cli/src/quick.rs` (`impl From<Distro> for uemulator_bootstrap::Distro`)

**Interfaces:**
- Produces: `#[derive(Args)] pub struct BootstrapArgs` with these fields:
  - `--user`
  - `--password`, `--ask-password`, `--password-file` (a clap group, at most one)
  - `--root-password`, `--ask-root-password`, `--root-password-file` (a group)
  - `--hostname`
  - `--ssh-key FILE...`
  - `--timezone`, `--locale`, `--keymap`
  - `--package P...`
  - `--yes`
- Produces: `#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)] pub struct
  BootstrapRecord`, saved under `[bootstrap]`. Its fields are `user`, `hostname`, `ssh_keys:
  Vec<String>` (the key contents, not their paths), `timezone`, `locale`, `keymap`, `packages`,
  and `root_password: bool`.
- Produces: `pub fn host_defaults() -> (String, String, String)` for timezone, locale and keymap.
  The fallbacks are `UTC`, `en_US.UTF-8` and `us`.
- Produces: `pub fn resolve_answers(args: &BootstrapArgs, saved: Option<&BootstrapRecord>,
  machine_name: &str, tty: bool, read_secret: impl FnMut(&str) -> io::Result<String>) ->
  Result<(Answers, BootstrapRecord), String>`. Flags win over the saved record, which wins over
  the defaults.
- `QuickCmd::Bootstrap { name, install: Distro, hardware, new_disks, linux, firmware, bootstrap:
  BootstrapArgs, uefi, runtime }`.

- [ ] **Step 1: Write the failing tests.**
  - Parsing:
    - `--password` together with `--ask-password` is an error
    - a missing `--user` is an error
    - `--firmware none` and `--hypervisor X` on `quick bootstrap` are refused with
      `machines::create`'s naming style
  - `resolve_answers`:
    - flag beats record, record beats default
    - hostname defaults to the machine name
    - `--ask-password` calls `read_secret` twice and errors when the two answers differ
    - no password option with a saved record and `tty` false gives an error containing
      `--password-file`
    - `--password-file` strips one trailing newline
  - `machine_toml_bootstrap_round_trip`: the saved text contains no `$6$` and no password.
  - An old `machine.toml` without `[bootstrap]` loads with `None`.
- [ ] **Step 2:** Run `cargo test --release -p uemulator-cli bootstrap`. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - Host defaults:
    - macOS: `/etc/localtime`'s symlink target after `zoneinfo/`, `LANG`, and `us` for the keymap
    - Linux: the same, plus `/etc/vconsole.conf`'s `KEYMAP=`
    - Windows: the fallbacks
  - Secret prompts use the existing raw-mode helpers in `term.rs` with echo off.
  - The salt comes from `getrandom` through `salt_from_bytes`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run autofix and clippy. Commit with `cli: quick bootstrap arguments and saved
  answers`.

### Task 5: Completion-marker watch

**Files:**
- Modify: `crates/uemulator-cli/src/bootstrap.rs`, `src/args.rs` (`RuntimeArgs`), `src/run.rs`
  (`console.on_output`)

**Interfaces:**
- Produces: `#[derive(Clone, Default, Debug)] pub struct MarkerWatch(Arc<…>)`, with `pub fn
  feed(&self, bytes: &[u8])` and `pub fn seen(&self) -> bool`. It keeps only the last
  `COMPLETION_MARKER.len() - 1` bytes between calls.
- `RuntimeArgs` gets `#[arg(skip)] pub marker_watch: Option<MarkerWatch>`. `run_config`'s
  `on_output` closure feeds it before writing to stdout.

- [ ] **Step 1: Write the failing tests:**
  - `marker_in_one_chunk`
  - `marker_split_at_every_byte` (each split point of `"…\r\nuemulator-bootstrap:
    installed\r\n"` into two `feed`s)
  - `partial_marker_is_not_seen`
  - `marker_surrounded_by_ansi_noise`
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement, and wire it into `run_config`.
- [ ] **Step 4:** Run them. Expected: PASS. Commit with `cli: watch the console for the bootstrap
  marker`.

### Task 6: `quick bootstrap` run loop

**Files:**
- Modify: `crates/uemulator-cli/src/machines.rs` (`create` takes `bootstrap:
  Option<BootstrapRecord>`; adds `pub fn bootstrap_launch(machine: &Machine, answers: &Answers,
  fetch) -> Result<Launch, String>`), `src/main.rs` (dispatch)

**Interfaces:**
- Consumes: `render`, `append_to_initramfs` (Tasks 2 and 3), `MarkerWatch` (Task 5),
  `resolve_answers` (Task 4).
- `bootstrap_launch` works as follows:
  - It calls `prepare_boot(..., Some(BootSource::Installer), None, fetch)`.
  - It replaces `initramfs` with the appended one and appends `payload.cmdline` to
    `settings.append`, after the distro arguments.
  - It sets `settings.no_reboot = true`.
- Produces: `fn run_bootstrap(home, machine, answers, uefi, runtime) -> Result<ExitReason,
  String>` in `main.rs`. It runs the installer launch with a fresh `MarkerWatch`, then:
  - `Reset` with the marker seen: set `boot_from = Auto`, save, print `installed; booting
    <name> from its disk`, then run `prepare_boot(None)` and `boot_launch` with the user's own
    reboot setting.
  - `Reset` without the marker: `Err("the installer of <name> reset before it finished: run
    `uemulator quick reinstall <name>` to try again")`.
  - Any other reason: return it unchanged.

- [ ] **Step 1: Write the failing tests** in `machines.rs` tests, using a `FakeFetcher` and fake
  artifacts:
  - `bootstrap_launch_appends_payload_and_disables_reboot`: the initramfs starts with the
    artifact bytes and contains `preseed.cfg`; `settings.append` ends with `auto=true
    priority=critical`; `no_reboot` is true.
  - `bootstrap_create_saves_record_without_secrets`.
  - `bootstrap_refuses_firmware_none_and_hypervisor`.
  - In `main.rs` tests, the loop decision as a pure function `fn after_installer(reason:
    &ExitReason, marker_seen: bool) -> AfterInstaller { BootDisk, Failed, Done }`, with a case
    for each `ExitReason` variant: `Stopped` and `Suspended` give `Done`, `Reset` with the marker
    gives `BootDisk`, and `Reset` without it gives `Failed`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. On `BootDisk`, `boot_from` is saved before the disk boot starts.
- [ ] **Step 4:** Run `cargo test --release --workspace`. Expected: PASS.
- [ ] **Step 5:** Run autofix and clippy. Commit with `cli: quick bootstrap installs
  unattended and boots the result`.

### Task 7: `quick reinstall` runs unattended and boots

**Files:**
- Modify: `crates/uemulator-cli/src/main.rs` (`QuickCmd::Reinstall` gets `#[command(flatten)]
  bootstrap: BootstrapArgs`, plus `uefi` and `runtime`), `src/machines.rs` (`reinstall` takes
  `Option<BootstrapRecord>` and saves it)

**Interfaces:**
- Produces: `fn confirm_erase(machine: &Machine, tty: bool, yes: bool, ask: impl FnMut(&str) ->
  io::Result<String>) -> Result<(), String>`. The prompt text is fixed by Global Constraints. The
  answers `y` and `yes` (in any case) proceed; anything else is `Err("not reinstalled")`.
- Reinstall becomes a bootstrap when the machine has a `[bootstrap]` record or `--user` is given.
  Otherwise it is today's interactive reinstall, which now boots the installer afterwards.

- [ ] **Step 1: Write the failing tests:**
  - `confirm_erase_needs_yes_without_tty`
  - `confirm_erase_accepts_y_and_yes`
  - `confirm_erase_names_first_disk_and_size`
  - `reinstall_with_record_and_no_password_off_tty_fails_before_touching_artifacts`: the
    artifacts directory is still there and the disk bytes are unchanged
  - `reinstall_flag_overrides_saved_record_and_is_saved`
  - a parse test for `quick reinstall m --user alex --password-file f --yes`
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement in this order: answers resolved and confirmation given, then artifacts
  thrown away and fetched, then `run_bootstrap`, or `boot_launch` for an interactive reinstall.
  Delete the old "It boots the installer until `quick edit`" message.
- [ ] **Step 4:** Run `cargo test --release --workspace`. Expected: PASS.
- [ ] **Step 5:** Run autofix and clippy. Commit with `cli: quick reinstall repeats a bootstrap
  and boots`.

### Task 8: `uemulator-apple` C ABI

**Files:**
- Create: `crates/uemulator-apple/src/bootstrap.rs`
- Modify: `crates/uemulator-apple/Cargo.toml` (adds `uemulator-bootstrap` under the Apple target
  deps), `src/lib.rs`, `include/uemulator.h`

**Interfaces:**
- Produces, in C:
  ```c
  #define UA_BOOTSTRAP_COMPLETION_MARKER "uemulator-bootstrap: installed"
  // `answers_json`: {"user","password","root_password"?,"hostname","ssh_keys":[],"timezone",
  // "locale","keymap","packages":[]}, plain-text passwords (hashed here with `salt`, 16 random
  // bytes from SecRandomCopyBytes). On success *out/*out_len is the extended initramfs (free
  // with ua_bytes_free) and *cmdline the arguments to append (free with ua_string_free).
  int32_t ua_bootstrap_initrd(const char *distro, const char *answers_json,
                              const uint8_t *salt, size_t salt_len,
                              const uint8_t *initrd, size_t initrd_len,
                              uint8_t **out, size_t *out_len, char **cmdline,
                              char *error, size_t error_capacity);
  void ua_bytes_free(uint8_t *bytes, size_t len);
  ```
- On Apple targets only, like the rest of the crate.

- [ ] **Step 1: Write the failing tests:**
  - `bootstrap_initrd_round_trip`: call through the `extern "C"` fn with a fake initrd; the
    output starts with it, and `cmdline` is Debian's.
  - `bootstrap_rejects_bad_json_and_invalid_answers_with_field_in_error`.
  - `salt_must_be_sixteen_bytes`.
- [ ] **Step 2:** Run `cargo test --release -p uemulator-apple --target aarch64-apple-darwin`.
  Expected: FAIL.
- [ ] **Step 3:** Implement. Parse with `serde_json` into a private struct, hash both passwords,
  build `Answers`, and call `render` and `append_to_initramfs`. Zero the JSON's password strings
  after use.
- [ ] **Step 4:** Run the tests and clippy for `aarch64-apple-ios`. Expected: PASS with zero
  warnings. Commit with `apple: bootstrap C ABI`.

### Task 9: App: Install sheet, boot path, marker, Reinstall…

**Files:**
- Create: `apple/App/BootstrapSheet.swift`
- Modify:
  - `apple/App/UemulatorApp.swift`: `NewMachineMenu`'s Install items open the sheet;
    `-new-source bootstrap:DISTRO` with `-bootstrap-user`, `-bootstrap-password`,
    `-bootstrap-hostname` and the like
  - `apple/App/MachineStore.swift`: `definition.bootstrap: BootstrapRecord?`, read and written as
    the `bootstrap` table through the existing `ua_machine_load`/`ua_machine_save` JSON path;
    `pendingBootstrap: [String: BootstrapRequest]` kept in memory only
  - `apple/App/MachineView.swift`: `start()` extends `settings.initrd` and the command line when
    a request is pending and the launch is the installer; the reset handler is described below
  - `apple/App/MachineSettingsView.swift`: shows the record and a Reinstall… button
  - `apple/App/Presets.swift`: `Distro.bootstrapName`, for the C ABI
  - `apple/uemulator.xcodeproj`: adds the new file

**Interfaces:**
- Consumes: `ua_bootstrap_initrd`, `UA_BOOTSTRAP_COMPLETION_MARKER` (Task 8).
- `struct BootstrapRequest { var record: BootstrapRecord; var password: String; var
  rootPassword: String? }`. It never reaches disk.
- The sheet works as follows:
  - "Install automatically" is on by default. Off means today's one-click interactive install.
  - The fields are those in the spec's App section. Timezone defaults to
    `TimeZone.current.identifier`, locale to `Locale.current` (as `xx_YY.UTF-8`), keymap to `us`.
  - Create stays disabled until User and Password are filled in.
  - The SSH key comes from `.fileImporter` and is read as text.
- The reset handler, in `MachineView` where `EngineState.resetReason` is handled: keep the last
  `UA_BOOTSTRAP_COMPLETION_MARKER.count` bytes of the console output during a bootstrap run, as
  Task 5 does.
  - Marker seen: drop the pending request, keep `boot_from` as auto, then `start()`.
  - Marker not seen: drop the request and stop with `"The installer reset before it finished.
    Use Reinstall… to try again."`.
- Reinstall… asks for the password again (the other fields are prefilled from the record) and
  uses a destructive "Erase and Install" button.

- [ ] **Step 1:** Build the sheet and the store changes. Then build: `xcodebuild -project
  apple/uemulator.xcodeproj -scheme uemulator -destination 'platform=macOS'
  -skipPackagePluginValidation build`. Expected: success with zero Swift warnings. Repeat for
  `'generic/platform=iOS Simulator'`.
- [ ] **Step 2:** Add the boot path and the reset handler. Rebuild both platforms.
- [ ] **Step 3:** Check by hand on macOS with `-new-machine bt -new-source bootstrap:debian
  -bootstrap-user alex -bootstrap-password test -autostart`. The installer runs with no input,
  the machine restarts on its disk, and `Documents/Machines/bt/machine.toml` has a `[bootstrap]`
  table with no `$6$` in it. Then check the WebKit engine the same way with `-engine webkit`.
- [ ] **Step 4:** Commit with `apple: unattended installs from the Install sheet`.

### Task 10: Docs and the real-install check

**Files:**
- Modify:
  - `CLAUDE.md`: a Commands bullet for `quick bootstrap` and `reinstall`
  - `docs/plans/35-quick-bootstrap.md`: status becomes "implemented", and the record of the
    checks below
  - `docs/plans/10-milestones.md`
  - `docs/plans/28-apple.md`: the Install sheet
  - `docs/plans/README.md` and `docs/README.md`: the status words in the index lines
  - the `quick.rs` and `Presets.swift` header comments: the mirror now includes `bootstrapName`

- [ ] **Step 1:** Run `cargo run --release -p uemulator-cli -- quick bootstrap bt-<distro>
  --install <distro> --user alex --password-file pw --ssh-key ~/.ssh/id_ed25519.pub --package
  htop` for each of `alpine`, `debian` and `fedora`. It should end at the installed system's
  login prompt with no input. Log in as `alex`, check `sudo true` (or `doas` on Alpine) and
  `htop --version`, and check that root login fails.
- [ ] **Step 2:** Run `quick reinstall bt-debian --install fedora --password-file pw --yes`. It
  should reach Fedora's login prompt.
- [ ] **Step 3:** Record the results and timings in the spec. Update the docs. Run `cargo test
  --release --workspace`, autofix and clippy. Commit with `docs: quick bootstrap`.
