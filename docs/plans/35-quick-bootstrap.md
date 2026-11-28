# 35 — `quick bootstrap`: unattended distro installs

**Status:** proposal. Builds on `quick create --install` (`crates/uemulator-cli/src/quick.rs`) and the Apple app (`28-apple.md`).

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
| `--ssh-key FILE` (repeatable) | SSH key (file picker) | none | Installs and enables the OpenSSH server, and authorizes the key(s) for the user |
| `--timezone TZ` | Timezone | the host's | IANA name |
| `--locale LOCALE` | Locale | the host's, else `en_US.UTF-8` | Alpine ignores it (musl) |
| `--keymap MAP` | Keymap | the host's, else `us` | |
| `--package P` (repeatable) | Packages (space separated) | none | Added by the installer, by distro package name |

The rest of `quick create`'s options (memory, disk size, display, TPM and so on) work on `quick
bootstrap` unchanged: it is `quick create --install` plus the answers. The exceptions are
`--firmware none` and `--hypervisor`, which bootstrap refuses up front, because neither can boot
the installed disk afterwards (`machines::prepare_boot` refuses both).

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
destructive "Erase and Install" button, as `reinstall`'s confirmation.

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
  repo `inst.repo` names today), `user --groups=wheel --iscrypted`, `rootpw --iscrypted` or
  `rootpw --lock`, `sshkey`, `autopart` (Anaconda's default), `clearpart --all --initlabel
  --drives=vda`, `%packages` with the extras, and `reboot`.
- **Alpine**: the netboot init has no answer-file hook of its own, so this is the open risk (see
  below). The plan: the archive carries an `apkovl` overlay with an `/etc/local.d` script (and
  `local` enabled in the default runlevel) that runs `setup-alpine -e -f <answers>` with the
  answers file, `setup-disk -m sys /dev/vda`, sets the passwords with `chpasswd -e`, writes the
  key into the installed system, and reboots. The open question is how the init finds an overlay
  that lives in the initramfs. Candidates: `apkovl=` naming a local path, or an archive file that
  replaces the init's overlay-search step.

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

- **Alpine's overlay hook** (above). Spike it first. If no hook works from inside the initramfs,
  come back before falling back to approach 2 for Alpine alone.
- The exact preseed and kickstart contents, kept to what the table above needs.
- How host defaults are read on macOS, Linux and Windows. Fall back to `UTC`, `en_US.UTF-8` and
  `us`.
- The app's `-new-source bootstrap:` launch arguments.
- Whether `quick bootstrap` on an existing machine name is an error (like `quick create`) or
  suggests `reinstall`. Suggesting `reinstall` is preferred.
