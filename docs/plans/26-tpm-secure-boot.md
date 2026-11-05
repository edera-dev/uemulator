# TPM 2.0 and Secure Boot

**Status:** implemented: the TPM device's measured-boot subset, the firmware's TCG2 protocol and event log, Secure Boot with Microsoft's keys, authenticated variable writes, `--tpm`/`--secure-boot` on the CLI and the web page. Not implemented: the TPM's keys, sessions, NV indices and persistence (below, "Next").

Both are off by default and independent: `MachineConfig::tpm(true)` adds the TPM, `MachineConfig::secure_boot(true)` makes the UEFI firmware enforce Secure Boot. A Windows-11-style machine turns on both.

## Crates

`crates/uemulator-crypto` (`no_std` + `alloc`, no dependencies) has what both need: SHA-1/256/384/512, RSA PKCS#1 v1.5 verification (Montgomery, up to 8192 bits), a DER reader, X.509, PKCS#7 `SignedData` (wrapped or bare, embedded or detached content, chains to an anchor that may be an intermediate, no time or key-usage checks, as EDK2) and the Authenticode PE digest. Verification only: there are no private keys anywhere. On x86_64 SHA-256 uses the SHA extensions when CPUID has them, in inline assembly that saves the XMM registers it uses (the firmware's target is soft-float, where LLVM cannot lower vector types, and the firmware is called through the Microsoft ABI). The library depends on it for the TPM's hashes, which makes it the library's one dependency without `jit`; it is first-party, so "no third-party dependencies" still holds.

## The TPM device (`crates/uemulator/src/dev/tpm/`)

- Interface: CRB (TCG PC Client Platform TPM Profile) at 0xFED40000, one page: locality 0 registers, the control area at +0x40, the data buffer at +0x80 (3968 bytes, which bounds commands and responses). QEMU's `hw/tpm/tpm_crb.c` is the reference. A command runs synchronously in the write that sets `CTRL_START`, so `START` reads back clear at the first poll and no interrupt exists. Only locality 0.
- ACPI: the `TPM2` table (revision 4, start method 7, ending after the start-method parameters: no LAML/LASA, since Linux takes the log from the UEFI configuration table and warns about an empty log area) and `\_SB.TPM` with `_HID "MSFT0101"` and the page in `_CRS`. Linux's `tpm_crb` binds to it. Linux also prints "Disabling hwrng": it takes an `MSFT0101` CRB on an AMD CPU for AMD's fTPM, whose RNG it distrusts. Harmless (virtio-rng feeds the pool).
- Commands (TPM 2.0 Library 1.59): Startup (CLEAR only: there is never saved state), Shutdown, SelfTest, IncrementalSelfTest, GetTestResult, GetCapability (algorithms, handles, commands, PCRs, TPM properties, PCR properties, and empty lists for the rest), GetRandom (the machine's entropy, seeded under `--verify`), StirRandom, PCR_Read, PCR_Extend, PCR_Event, PCR_Reset (16 and 23), Hash, HashSequenceStart, SequenceUpdate, SequenceComplete, EventSequenceComplete, FlushContext, ReadClock. Anything else is `TPM_RC_COMMAND_CODE` and is absent from `TPM_CAP_COMMANDS`, whose `TPMA_CC` handle counts Linux's resource manager relies on.
- One PCR bank, SHA-256 (swtpm's default too). PCRs 17 to 22 start as all ones and need a locality above 0, so they can be read but not changed. Hashing (`TPM2_Hash`, sequences, `PCR_Event`) takes all four SHA algorithms.
- Authorization: password sessions only (`TPM_RS_PW`); the PCRs' value is empty, a sequence's is what started it. Handles are checked before authorization, as the reference implementation does.
- State lasts as long as the `Devices`: a guest reset builds new devices, which is `_TPM_Init`.

## Measured boot (`crates/uemulator-firmware/src/tpm.rs`, `tcg.rs`)

The firmware starts the TPM (`TPM2_Startup(CLEAR)`; `TPM_RC_INITIALIZE` is accepted), reads the allocated banks, keeps a crypto-agile log (64 KiB of boot services data, `Spec ID Event03` first) and a final events table (16 KiB of ACPI NVS, installed as `EFI_TCG2_FINAL_EVENTS_TABLE_GUID`), and publishes `EFI_TCG2_PROTOCOL` (GetCapability, GetEventLog for the TCG 2 format, HashLogExtendEvent with `EXTEND_ONLY` and `PE_COFF_IMAGE`, SubmitCommand, GetActivePcrBanks; SetActivePcrBanks only accepts the current allocation). After `GetEventLog` events also go to the final events table. What it measures (PC Client PFP 1.05):

| PCR | event | when |
|---|---|---|
| 0 | `EV_S_CRTM_VERSION` "uemulator UEFI firmware 1" (UTF-16) | TPM start-up |
| 7 | `EV_EFI_VARIABLE_DRIVER_CONFIG` SecureBoot, PK, KEK, db, dbx (empty ones too), then `EV_SEPARATOR` | start-up |
| 1 | `EV_EFI_VARIABLE_BOOT2` BootOrder and each Boot#### it lists | first boot attempt |
| 4 | `EV_EFI_ACTION` "Calling EFI Application from Boot Option" | each attempt; "Returning ..." after |
| 0-6 | `EV_SEPARATOR` | first boot attempt |
| 4 / 2 | `EV_EFI_BOOT_SERVICES_APPLICATION` / `_DRIVER` / `EV_EFI_RUNTIME_SERVICES_DRIVER`: Authenticode digest, `UEFI_IMAGE_LOAD_EVENT` with the file's buffer as location | every `LoadImage` |
| 7 | `EV_EFI_VARIABLE_AUTHORITY`: the db entry that allowed an image | once per entry |
| 5 | `EV_EFI_ACTION` "Exit Boot Services Invocation" and "... Returned with Success" | `ExitBootServices`, once |
| 8, 9 | `EV_IPL`: Xen's Multiboot2 ELF, dom0 kernel and initramfs (9), their command lines (8), as GRUB measures | the firmware's own Multiboot2 loader |

Nothing on the `ExitBootServices` path allocates (commands and digests on the stack, the log preallocated): the OS owns the memory map by then. The Linux stub measures its load options and initramfs into PCR 9 itself.

Cost: the firmware hashes what it measures in guest code. Debian's 12 MB kernel and 39 MB initramfs add about 2 s to a boot on an M5 Max (portable SHA-256 2.6 s; the SHA extensions 2.0 s, because the JIT does not compile them and the interpreter runs those blocks). Compiling `sha256rnds2`/`sha256msg1`/`sha256msg2` in the JIT is the next lever.

## Secure Boot (`secure_boot.rs`, `variable_store.rs`, `variables.rs`, `images.rs`)

- Keys (`crates/uemulator-firmware/secure-boot/`, provenance in its README): PK is a uemulator certificate whose private key was discarded when it was made, so nothing can sign a PK or KEK update; KEK is Microsoft's KEK CA 2011 and KEK 2K CA 2023; db is Microsoft Windows Production PCA 2011, Microsoft Corporation UEFI CA 2011, Windows UEFI CA 2023, Microsoft UEFI CA 2023 and Microsoft Option ROM UEFI CA 2023; dbx is the signature lists of Microsoft's current x64 dbx update (`dbx_x64.efiauth2`, signed with the 2023 KEK). Owner GUID of Microsoft's entries is `77fa9abd-...`, of the PK `f4157d1f-df64-482e-a25d-10b2385fb735`.
- Enrollment: with the handoff's Secure Boot flag (version 4, offset 0x378) and no PK in the store, the firmware enrolls KEK, db, dbx, then PK, persistently, with zero timestamps. Without the flag nothing is enrolled and nothing is removed: a machine that once had Secure Boot keeps its keys, shows `SetupMode` 0 and `SecureBoot` 0, and boots anything. There is no way yet to go back to setup mode short of wiping the nvram file.
- Variables: `SecureBoot` (decided at start-up: flag on and a PK), `SetupMode` (no PK; refreshed after PK changes), `VendorKeys` (1 until anything changes PK, KEK, db or dbx through `SetVariable`), `AuditMode` and `DeployedMode` 0, `PKDefault`, `KEKDefault`, `dbDefault`, `dbxDefault`, `SignatureSupport`, all read-only.
- Writes to PK, KEK, db, dbx, dbt and dbr (UEFI 2.9 8.2.2, EDK2's `AuthVariableLib`): attributes NV|BS|RT|TIME_BASED_AUTHENTICATED (plus `APPEND_WRITE`), `EFI_VARIABLE_AUTHENTICATION_2` in front, a timestamp later than the stored one unless appending, and in user mode a PKCS#7 signature over name || GUID || attributes || timestamp || data by the PK (PK, KEK) or a KEK or the PK (the databases). The attributes in the message include `APPEND_WRITE`: Microsoft's dbx updates verify only that way. In setup mode only a new PK is checked, against its own certificate. Appends skip signatures already present. Timestamps live beside the variable under `vartime/` in the same store. Other time-based authenticated variables are still refused.
- Image checks (UEFI 2.9 32.5, EDK2's `DxeImageVerificationLib`): an image whose hash (SHA-256/384/512) dbx lists, or one of whose signatures chains to a dbx certificate, or whose signing chain has a certificate dbx revokes by TBS hash, is refused; otherwise a signature chaining to a db certificate, or the image's hash in db, allows it; a signature whose digest does not match the image counts as none. `LoadImage` measures first, then checks (EDK2's order); a refused image gets a handle and `EFI_SECURITY_VIOLATION`, and `StartImage` refuses it (UEFI 2.9 7.4). The firmware's own uses of `LoadImage` (the supplied kernel, `xen.efi`) unload it and say why on the console. Its Multiboot2 loader refuses Xen under Secure Boot: an ELF has nothing to verify.
- The direct boot and the BIOS check nothing, so `secure_boot` without `uefi_firmware` is a `BadConfig`, and the CLI says to use `--firmware uefi`.

## Front ends

- CLI: `--tpm[=BOOL]` and `--secure-boot[=BOOL]` on every command with hardware options; `machine.toml` keeps `tpm` and `secure_boot`. A named machine's keys live in its `nvram.kv`.
- Web: "TPM 2.0" (needs ACPI) and "Secure Boot" (needs UEFI) in Advanced, `?tpm` and `?secure-boot`, `OPTION_TPM`/`OPTION_SECURE_BOOT` (bits 11 and 12) in the C ABI. The page keeps no variables, so the keys are enrolled again on every load. `node hack/wasm/boot.mjs --tpm --secure-boot` passes them too.

## Verification

- Unit tests: the TPM engine and CRB interface (`dev/tpm/tests.rs`), ACPI (`a_tpm_adds_the_tpm2_table_and_its_device`), the handoff both sides, the machine (`machine::tests::tpm`), the crypto crate against FIPS vectors, openssl output and Microsoft's real certificates and dbx updates, the firmware's policy (`secure_boot.rs`) and authenticated writes (`variable_store.rs`: Microsoft's KEK-2011 and KEK-2023 dbx updates append in user mode; tampered, replayed, misdirected and wrongly-attributed ones are refused).
- Real guests (not tests): Debian 13's 6.12 kernel and initramfs with `--tpm` through the firmware: `tpm_crb` binds, `/sys/class/tpm/tpm0/pcr-sha256/*` and `binary_bios_measurements` exist, and replaying the log (the final events table's `ExitBootServices` events included) gives exactly PCRs 0, 1, 4, 5, 7 and 9. Under direct boot Linux finds the TPM unstarted and starts it itself. The same kernel with `--secure-boot` is refused ("nothing in db allows it"). An installed Debian 13 disk with `--secure-boot --tpm`: the firmware allows Debian's Microsoft-signed shim, shim verifies Debian's GRUB, GRUB boots Debian's kernel, Linux reports "Secure boot enabled" and lockdown and reaches its login prompt. GRUB's Xen entry is refused by GRUB itself (Debian's GRUB forbids `multiboot2` under Secure Boot). The web build under Node: `--tpm` and `--secure-boot` behave the same.

## Next

- TPM keys: primaries and objects (RSA 2048, ECC P-256 with ECDH), HMAC and policy sessions, `Create`/`Load`/`Unseal`, NV indices and persistent handles, ContextSave/Load, and persisting seeds and NV in the machine's key-value store. That is what systemd-cryptenroll, Linux's `TCG_TPM2_HMAC` session encryption and BitLocker need, and an EK certificate for Windows' attestation. (Debian's 6.12 kernel logged no TPM errors at boot; a kernel built with `TCG_TPM2_HMAC` has not been tried.)
- A way back to setup mode (a CLI option that clears PK) and custom keys (sbctl).
- JIT-compiled SHA instructions for the measurement cost.
- Private time-based authenticated variables, GPT measurement into PCR 5, `EV_EFI_HANDOFF_TABLES2`.
