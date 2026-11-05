# Secure Boot keys

What the firmware enrolls when Secure Boot is turned on and the store has no PK (`src/secure_boot.rs`, `docs/plans/26-tpm-secure-boot.md`). All are public; none of the private keys exist here.

| file | variable | what | source |
|---|---|---|---|
| `uemulator-pk.der` | PK | "uemulator Platform Key (private key discarded)", RSA 2048, self-signed, valid to 2126 | made for uemulator (below); the key was deleted |
| `microsoft-kek-ca-2011.der` | KEK | Microsoft Corporation KEK CA 2011 | https://go.microsoft.com/fwlink/?LinkId=321185 |
| `microsoft-kek-2k-ca-2023.der` | KEK | Microsoft Corporation KEK 2K CA 2023 | https://go.microsoft.com/fwlink/?linkid=2239775 |
| `microsoft-windows-production-pca-2011.der` | db | Microsoft Windows Production PCA 2011 (Windows boot manager) | https://go.microsoft.com/fwlink/?linkid=321192 |
| `microsoft-uefi-ca-2011.der` | db | Microsoft Corporation UEFI CA 2011 (third parties: shim, option ROMs) | https://go.microsoft.com/fwlink/?linkid=321194 |
| `windows-uefi-ca-2023.der` | db | Windows UEFI CA 2023 | https://go.microsoft.com/fwlink/?linkid=2239776 |
| `microsoft-uefi-ca-2023.der` | db | Microsoft UEFI CA 2023 | https://go.microsoft.com/fwlink/?linkid=2239872 |
| `microsoft-option-rom-uefi-ca-2023.der` | db | Microsoft Option ROM UEFI CA 2023 | https://go.microsoft.com/fwlink/?linkid=2284009 |
| `dbx-x64-kek2023.auth2` | dbx | Microsoft's x64 dbx update signed with the 2023 KEK; the firmware enrolls its signature lists | https://github.com/microsoft/secureboot_objects `PostSignedObjects/SignedByKEK2023/dbx_x64.efiauth2` |
| `dbx-x64-kek2011.auth2` | (tests) | the same update signed with the 2011 KEK | same repository, `PostSignedObjects/SignedByKEK2011/dbx_x64_Legacy/dbx_x64_Legacy.efiauth2` |

Microsoft's entries carry the owner GUID `77fa9abd-0359-4d32-bd60-28f4e78f784b`, the PK `f4157d1f-df64-482e-a25d-10b2385fb735`.

The PK was made with

    openssl req -x509 -newkey rsa:2048 -nodes -keyout pk.key -outform der -out uemulator-pk.der \
        -days 36500 -sha256 -subj "/O=uemulator/CN=uemulator Platform Key (private key discarded)"

and `pk.key` deleted, so no PK or KEK update can ever be signed: the OS can update db and dbx through Microsoft's KEKs (which is how dbx updates ship), and changing the PK or KEK means starting from an empty nvram. To refresh dbx, replace `dbx-x64-kek2023.auth2` with Microsoft's newer file; machines that already enrolled keep theirs until the OS applies the update.
