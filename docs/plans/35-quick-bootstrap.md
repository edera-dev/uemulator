# 35 — `quick bootstrap`: unattended distro installs

**Status:** implemented (plan: `35-quick-bootstrap-plan.md`). Real unattended installs of Alpine, Debian and Fedora through the CLI, and a `quick reinstall` from Debian to Fedora, reached the installed system's login prompt and were logged in to ("Checked" below). The app's Install sheet is built for macOS and the iOS simulator but has not been run. Builds on `quick create --install` (`crates/uemulator-cli/src/quick.rs`) and the Apple app (`28-apple.md`).

## Intent

`quick create --install <distro>` already gives a machine the distro's netboot installer, but
someone still has to click through it on the console. `quick bootstrap` takes the handful of
answers a person actually cares about up front and runs the same installer with nobody typing,
then boots straight into the installed system:

```
uemulator quick bootstrap debian-test --install debian --user alex --ask-password
```

The Apple app gets the same thing in its New Machine sheet. Alpine Linux, Debian and Fedora,
the three distros `quick create --install` already knows.

## Goals

- One command, or one dialog, from nothing to a login prompt on an installed system.
- The installer's own console output is shown the whole time (terminal in the CLI, terminal pane
  in the app). Nothing is typed, but a failure is visible where it happens.
- The same answers produce the same install in the CLI, the native app and the app's webkit
  engine.
- `quick reinstall` can run the install again unattended.

## Non-goals

- A general config format (cloud-init, Ignition, arbitrary preseed/kickstart passthrough).
- Custom partitioning. Each installer uses its own default layout.
- Distros beyond the three in `quick::Distro`.
- The web page. The design keeps that door open (see Delivery), but no UI is built for it here.

## Anti-goals

- A password, or its hash, in any file uemulator writes on purpose: `machine.toml`, the app's
  sidecar, a log, or a file under the machine directory. A RAM image the user asks for
  (`--suspend-to`, checkpoints, `--debug`) may hold the hash while the installer runs, which is
  accepted: those stay allowed on a bootstrap run.
- An install that goes interactive on a question the answer file missed. That is a bug in the
  answer file, and the golden tests should catch it. At runtime the question just stays on the
  console, where a person can answer it or stop the run. There is no inactivity timeout.
- A second hand-copied table of answer-file templates in Swift.

## Settings

| CLI | App field | Default | Notes |
|-----|-----------|---------|-------|
| `NAME` (positional) | Name | required | Machine name, as in `quick create` |
| `--install DISTRO` | the existing Install source | required | `alpine`, `debian`, `fedora` |
| `--user NAME` | User | required | A regular account in the admin group (`sudo` on Debian, `wheel` on Fedora and Alpine) |
| `--password TEXT` / `--ask-password` / `--password-file FILE` | Password | required, exactly one form | `--ask-password` prompts twice, with no echo |
| `--root-password TEXT` / `--ask-root-password` / `--root-password-file FILE` | Root password (optional) | root locked | None of these means root has no password and the user's sudo is the way in |
| `--hostname NAME` | Hostname | the machine name | |
| `--ssh-key FILE` (repeatable) | SSH key (file picker) | none | Installs and enables the OpenSSH server, and authorizes the key(s) for the user. Each line must be an OpenSSH public key, `TYPE BASE64 [COMMENT]` with a type `ssh-keygen` writes (no `authorized_keys` options), which keeps a private key picked by mistake out of `machine.toml` and the guest; errors name the key by number and type, never its text |
| `--timezone TZ` | Timezone | the host's | IANA name |
| `--locale LOCALE` | Locale | the host's, else `en_US.UTF-8` | Alpine ignores it (musl) |
| `--keymap MAP` | Keymap | the host's, else `us` | A layout name every installer takes, two or three lowercase letters (`de`, not `de-latin1`); `Answers::validate` checks it, for the app too |
| `--package P` (repeatable) | Packages (space separated) | none | Added by the installer, by distro package name |

The rest of `quick create`'s options (memory, disk size, display, TPM and so on) work on `quick
bootstrap` unchanged: it is `quick create --install` plus the answers. The exceptions are
`--firmware none` and `--hypervisor`, which bootstrap refuses up front, because neither can boot
the installed disk afterwards (`machines::prepare_boot` refuses both); `--display framebuffer`,
which moves `/dev/console`, where the answer files print the completion marker, to the screen the
run cannot watch (`--display both` gives the screen and keeps the console; an unattended `quick
reinstall` of a machine saved with `framebuffer` is refused the same way, pointing at `quick
edit --display both`); and a `--disk` image as the first disk, which every answer file erases
without a question (`--create-disk` first keeps an attached image as a second disk; `quick
create` plus `quick reinstall --user`, which asks, installs onto an existing image). The machine
name is checked before any password is asked for.

Passwords are hashed with SHA-512 crypt (`$6$`, random salt) as soon as they are read, and only
the hash goes into the answer file. All three installers take a crypted password (Debian
`passwd/user-password-crypted`, kickstart `--iscrypted`, Alpine `chpasswd -e`).

`machine.toml` records the non-secret answers under a `[bootstrap]` table (user, hostname, SSH
key contents, timezone, locale, keymap, packages, and whether a root password was set), so
`quick reinstall NAME` repeats them. It never records a password or its hash.

## After the install

The answer file's last step prints a completion marker on the serial console (a fixed line such
as `uemulator-bootstrap: installed`, with a constant name in the bootstrap crate), then reboots.

The library's own reboot (`MachineConfigBuilder::reboot(true)`, which the CLI turns on in
`run.rs`) restarts in place with the same kernel and initramfs, and it would run the installer
again. So a bootstrap run turns it off and loops itself, the way the app already does
(`MachineView` starts the machine again on `EngineState.resetReason`):

- A reset after the marker: save `boot_from = auto` and boot again through `prepare_boot`, which
  now finds the disk installed and boots it. The run continues to the installed system's login
  prompt, and later resets restart that system as `quick boot` would.
- A reset without the marker (a failed or aborted install): stop the run with an error that names
  the machine and says to `quick reinstall` it. It does not loop. `boot_from` stays `installer`.
- Any other end of the run (Ctrl-], `--timeout`, a poweroff) after the marker: save `boot_from =
  auto`, as the app does, so the next `quick boot` boots the installed disk, but do not boot it
  now. Before the marker `boot_from` stays `installer`.

A new machine, CLI or app, is saved with `boot_from = installer` for its unattended install and
switched to `auto` only by the marker, so a disk the installer partitioned and then failed on is
never booted.

The app does the same: its reset handler checks for the marker in the console output before it
starts the machine again on the disk.

## Reinstall

`quick reinstall NAME` on a machine with a `[bootstrap]` table asks for the password again
(`--ask-password` is implied when no password option is given and stdin is a terminal; otherwise
it is an error) and runs unattended. Any setting flag on `reinstall` overrides the recorded one
and is saved. `--install DISTRO` switches the distro as it does today. A `reinstall` of a machine
without a `[bootstrap]` table stays the interactive installer, unless `--user` is given, which
makes it a bootstrap.

An unattended reinstall erases the disk with no question from the installer, so `reinstall` asks
first on a terminal: `Erase disk-0.udisk (64 GiB) and install Fedora? [y/N]`. `--yes` skips the
question, and without a terminal `--yes` is required. `quick bootstrap` on a new machine never
asks, because its disk is new.

`quick reinstall` then boots, the same as `quick bootstrap` does. Today it only refetches the
artifacts. Making it boot is a change to the interactive path too.

## App

The Add menu's Install items (`NewMachineMenu`), which today make and start an interactive
installer in one click, open an Install sheet with an "Install automatically" toggle, on by
default. It reveals User, Password, Root password (optional), Hostname, SSH key, and a
disclosure for Timezone, Locale, Keymap and Packages, with the defaults above filled in from the
device. Create is disabled until User and Password are filled in. The toggle off is today's
interactive installer.

```
New Machine
  Source:  ( ) Demo  (•) Install Debian  ( ) ISO
  [x] Install automatically
      User           [alex        ]
      Password       [••••••••    ]
      Root password  [            ]  optional
      Hostname       [debian-test ]
      SSH key        [Choose…     ]
      ▸ More (timezone, locale, keymap, packages)
                                   [Cancel] [Create]
```

The answers are passed to the core over the C ABI. The password is never written to the
sidecar, the Keychain or `machine.toml`. The machine settings view shows the recorded
`[bootstrap]` answers and offers "Reinstall…", which asks for the password again and has a
destructive "Erase and Install" button, as `reinstall`'s confirmation. It is disabled, with the
reason in the section's footer, for what `quick reinstall` refuses (firmware None, a hypervisor,
the Screen display) and for a read-only first disk: an ISO is attached first, and the answer files
install to `/dev/vda`.

`-new-machine NAME -new-source bootstrap:debian` with `-bootstrap-user`, `-bootstrap-password`
and so on makes one from the command line, for UI tests and Xcode Cloud. (This is the builder's
call; see below.)

## Approaches considered for delivering the answers

1. **Append a cpio archive to the initramfs (chosen).** Linux unpacks concatenated cpio archives
   in order, and a later file replaces an earlier one. A small archive holding the answer file
   is appended to the downloaded initramfs at boot time. It needs no network service and no
   extra disk, and it is byte for byte the same in the CLI, the native app and the webkit
   engine, which all hand the library a kernel and an initramfs.
2. **The built-in NAT serves the answers over HTTP** at `http://10.0.2.2/`. Every installer
   fetches a URL natively (`preseed/url=`, `inst.ks=http://`, `apkovl=http://`), but this would
   mean an HTTP server inside the library's gateway, and the web build's shared segment and Wisp
   paths would need it too.
3. **Type into the installer** over the serial console, as `expect` would. No answer-file
   formats, but it breaks whenever an installer rewords or reorders a prompt.

## Delivery per distro

- **Debian**: d-i reads `/preseed.cfg` from the initramfs root by itself. Add `auto=true
  priority=critical` to the command line. The preseed covers locale, keymap, network (DHCP),
  mirror, user, root (`passwd/root-login false` when there is no root password), clock and
  timezone, guided partitioning of the first disk with the installer's default recipe,
  tasksel (standard + SSH server when there is a key), extra packages (`pkgsel/include`), GRUB
  on EFI or BIOS to match the firmware, the authorized key (`preseed/late_command`), and
  `finish-install/reboot_in_progress`.
- **Fedora**: put `/ks.cfg` in the archive and add `inst.ks=file:/ks.cfg`. The kickstart uses
  `text`, `lang`, `keyboard`, `timezone`, `network --bootproto=dhcp --hostname`, `url` (the same
  repo `inst.repo` names today), `repo` for the release's Everything tree when extra packages are
  asked for (the Server tree lacks most of them), `user --groups=wheel --iscrypted`, `rootpw
  --iscrypted` or `rootpw --lock`, `sshkey`, `autopart` (Anaconda's default), `clearpart --all --initlabel
  --drives=vda`, `%packages` with the extras, and `reboot`.
- **Alpine**: the netboot init has no answer-file hook of its own, but it does take an overlay
  from inside the initramfs (spiked and confirmed to run a script at boot). The archive carries
  `bootstrap.apkovl.tar.gz` at the initramfs root, and the kernel arguments add
  `apkovl=/bootstrap.apkovl.tar.gz`. A value with no `:` and no URL scheme is used by the init as a
  plain path in the initramfs (`prepare_apkovl` in `/init` only splits `device:path`), unpacked
  with `tar -zx` into the new root before OpenRC starts. The overlay, a gzip tar of root-relative
  paths with owner root, holds:
  - `etc/.default_boot_services`, an empty file. Without it an overlay replaces the init's default
    boot services (devfs, mdev, hwdrivers, modloop and so on), and the
    init only adds them when the file is present (it deletes it afterwards).
  - `etc/local.d/bootstrap.start`, mode 0755, `#!/bin/sh`: runs `apk add alpine-conf e2fsprogs`
    (and whatever else `setup-disk` needs) from the repository the init already knows, names the
    running release (`v3.24`) in place of its `latest-stable` (else `setup-apkrepos` picks
    edge), then `setup-alpine -e -f <answers>`, `setup-disk -m sys /dev/vda`, `chpasswd -e` for the
    passwords, writes the SSH key into the installed system, prints the completion marker and
    reboots. Output goes to `/dev/console`.
  - `etc/runlevels/default/local`, a symlink to `/etc/init.d/local`.

  Two things the archive builder must do: pad the original gzip initramfs with zero bytes to a
  multiple of 4 before appending the newc archive (Linux only recognizes a plain cpio segment at a
  4-byte aligned offset: unpadded, the boot logs `Initramfs unpacking failed: invalid magic at
  start of compressed archive` and the init is missing), and keep `apkovl=` as a kernel argument.
  Evidence: the console showed `Loading user settings from /bootstrap.apkovl.tar.gz: ok.` and
  then `Starting local ...hello`.

The answers file and archive are generated per boot, in memory, and handed to the library with
the initramfs. They are never written under the machine directory, because they contain the
password hash. A bootstrap interrupted before the install finishes leaves a machine that boots
the interactive installer, and `quick reinstall NAME` runs it unattended again.

## Code layout

- **New crate `crates/uemulator-bootstrap`**: `std`, no third-party dependencies. It holds the
  answer model (`BootstrapAnswers`), the per-distro answer-file renderers, the newc cpio writer,
  SHA-512 crypt (on `uemulator-crypto`'s SHA-512), and the kernel-argument additions. It takes
  `Distro` and today's per-distro install command lines and URLs from `quick.rs`, so there is one
  distro table. `quick.rs` re-exports or uses it.
- **CLI**: `QuickCmd::Bootstrap` in `main.rs`, handled in `machines.rs` (create plus a recorded
  `[bootstrap]` table, then the existing boot path with the initramfs extended). `reinstall`
  gains the setting flags and boots afterwards. Host defaults (timezone, locale, keymap) are read
  here.
- **`uemulator-apple`**: C ABI calls to hash a password, render the archive for a distro and
  answers, and append it to an initramfs buffer, so Swift never holds a template. It also needs
  `machine.toml` `[bootstrap]` read/write through `ua_machine_load`/`ua_machine_save` (which
  already keep unknown keys).
- **App**: the sheet fields above, `MachineStore` creating the machine, and both engines handing
  the extended initramfs to the core. In the webkit engine, the native side extends the initramfs
  before handing it over, so `engine.js` and `worker.js` do not change.
- **Docs**: CLAUDE.md's Commands, `docs/plans/28-apple.md`, and `docs/plans/10-milestones.md`.

## Testing

Unit tests only (CLAUDE.md: no network):

- SHA-512 crypt against the published test vectors (Drepper's spec).
- A cpio round trip: written archives parse, and an appended archive's file wins over the base
  archive's file of the same name.
- Each renderer: golden answer files for a full set of answers and a minimal set; quoting of
  awkward values (spaces in package lists, a `$` in a hash); no plain password in any output.
- Kernel-argument additions per distro, with no stray whitespace (as `quick.rs` tests today).
- `machine.toml` `[bootstrap]` round trip with no secret in it; `reinstall` precedence (flag
  over record).
- CLI parsing: the password forms are mutually exclusive, `--user` is required, and `--firmware
  none` and `--hypervisor` are refused.
- The run loop: a reset after the marker boots the disk and saves `auto`; a reset without it ends
  with an error and keeps `installer` (a fake kernel that prints the marker or not, then resets).
- `reinstall`'s confirmation: refused without a terminal unless `--yes` is given.

A real install of each distro is a manual check, recorded in the milestones the way the
installer checks are today.

## Left to the builder

- **Alpine's overlay hook** is settled (above); what is left is the script's installed-system steps,
  which the spike did not run.
- The exact preseed and kickstart contents, kept to what the table above needs.
- How host defaults are read on macOS, Linux and Windows. Fall back to `UTC`, `en_US.UTF-8` and
  `us`.
- The app's `-new-source bootstrap:` launch arguments.
- Whether `quick bootstrap` on an existing machine name is an error (like `quick create`) or
  suggests `reinstall`. Suggesting `reinstall` is preferred.

## Checked (2026-11-27)

Each install ran the release CLI on an Apple Silicon Mac (aarch64 JIT, `--cpus 2`, the user-mode
NAT to the distro's mirrors) as `quick bootstrap bt-DISTRO --install DISTRO --user alex
--password-file pw --ssh-key key.pub --package htop`, under a scratch `UEMULATOR_HOME`, with
stdin a pty that a script answered only once the installed system's login prompt was up: `root`
with the user's password (refused: `Login incorrect`), then `alex`, `sudo true` (`doas true` on
Alpine) with the password, `htop --version`, `id`, and checks that sshd ran and the key was in
`~/.ssh/authorized_keys`. Nothing was typed before that. Each run printed the completion marker
on the console, saved `boot_from = auto`, booted the disk through the UEFI firmware and reached
`bt-DISTRO login:`. No `$6$` hash was in `machine.toml`, `nvram.kv` or the downloaded
artifacts (only on the guest's own disk, in its `/etc/shadow`).

| distro | marker | login prompt | notes |
|--------|--------|--------------|-------|
| Alpine 3.24 | 81 s | 104 s | `doas` for `wheel`, htop 3.5.3, sshd running, time zone the host's (`/etc/localtime` to `America/Los_Angeles`); a later `quick boot` logs in the same |
| Debian 13 | 916 s | 984 s | GRUB from the disk through shim, groups `sudo` and the installer's defaults, htop 3.4.1, sshd running; the marker shows up between the newt screens |
| Fedora 44 | 1992 s | 2046 s | 4096 MiB (the new default), 435 packages, `wheel` (sudo works), htop 3.4.1 from Everything, sshd running |

Times are from the command's start and include the downloads.

`quick reinstall bt-debian --install fedora --password-file pw --yes` then installed Fedora 44 over
the Debian machine from its recorded answers and reached `bt-debian login:` after 2119 s, with the
same checks passing. Its first try stopped at a mirror's `Peer disconnected` while downloading the
initramfs, before the installer started; the same command again went through.

What the installs showed, each fixed with a test first where one could fail:

- **The terminal went to the ended installer.** Each machine of a run had its own stdin reader,
  and the installer's stayed blocked in `read` after its machine ended and took the bytes: nothing
  typed at the installed system's login prompt arrived, Ctrl-] included. One reader now lives for
  the process (`crates/uemulator-cli/src/input.rs`) and hands its chunks to the running machine.
- **Alpine installed edge.** The netboot repositories say `latest-stable`, which
  `setup-apkrepos` does not take for a release and turns into `edge`; the installed system came up
  as `3.25.0_alpha` with edge's kernel. The script now writes the running release (`v3.24`) into
  the repositories first.
- **Debian's guided partitioning refused 8 GiB** ("Unable to satisfy all constraints on the
  partition", under UEFI with the `atomic` recipe). New machines now get the app's 64 GiB
  (sparse) disk.
- **Fedora's stage2 did not fit in 3072 MiB**: dracut downloads the 915 MB `install.img` into a
  RAM file system and stopped at `No space left on device`. A Fedora install now gets 4096 MiB,
  CLI and app.
- **htop is not in Fedora's Server tree**, the installer's repository, and Anaconda stopped at
  its exception prompt (`No match for argument: htop`). With extra packages the kickstart adds the
  release's Everything tree.
- A new bootstrap machine was saved `boot_from = auto` until the marker. It is `installer` now,
  as the spec says, so a disk the installer partitioned and then failed on is not booted.
- `quick bootstrap` of an existing name now suggests `quick reinstall`.

Confirmed on the way: Alpine's `USEROPTS`/`SSHDOPTS` `none` asked nothing and `TIMEZONEOPTS` took
the bare IANA name; `setup-disk` carried the live world (doas, openssh, htop) into the installed
system; a failing step (`--package no-such-package-uemu`) fired the script's `trap reboot EXIT`
under OpenRC's `local`, and the run ended with `the installer of bt-alpine-fail reset before it
finished: run uemulator quick reinstall bt-alpine-fail to try again`. Debian's
`passwd/user-default-groups` gave `sudo` plus the usual groups. Fedora's `%post --nochroot` marker
reached the CLI's console. The kickstart's minimal package set was enough for sshd and sudo.

Not checked: the app (built for macOS and the iOS simulator, not run, `28-apple.md`), the webkit
engine, `--root-password`, and an install with a screen (`--display`).

## Accepted limitations

- A suspended bootstrap install cannot be finished with `--resume-from`. `quick bootstrap`
  refuses it, and `quick boot --resume-from` resumes without the bootstrap run's reset handling:
  the install in guest RAM carries on, but its closing reboot restarts in place into the installer,
  now without the answers (they were only ever in memory), instead of booting the disk.
  `quick reinstall` starts the install again.
- The app loses a pending bootstrap request when it quits: a machine relaunched or resumed
  mid-install ends in the interactive installer. Reinstall… starts it again.
- The app's Reinstall… always runs unattended, also on a machine installed by hand; the CLI keeps
  such a machine interactive unless `quick reinstall` is given `--user`.
- A misspelled `--package` makes Debian and Fedora go interactive: d-i stops at "Installation step
  failed" (Select and install software, seen with a made-up name) and Anaconda at its exception
  prompt (seen with htop before the Everything fix), on the console for a person to answer or stop. Alpine ends the run with the
  reinstall error instead. Turning those errors into a reset is not done; skipping the name
  silently (kickstart's `%packages --ignoremissing`) would be worse.
- `--timeout` restarts with the installed system's boot, so a bootstrap run can last up to twice
  as long (its help on `quick bootstrap` and `quick reinstall` says so).
- A key typed just as the installer's run ends can still reach the stopped machine and be lost
  (`TerminalInput::attach`: the end is checked before each chunk is handed over, not atomically
  with it).
- Fedora's 4 GiB install is more than the app's WebKit engine allows (3 GiB): Fedora installs need
  the native engine.
