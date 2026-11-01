//! A hypervisor (Xen) booted under the machine's Linux kernel, which becomes its first domain
//! (dom0). Xen ships two ways and both are accepted: `xen.gz`, a gzipped 32-bit ELF with a
//! Multiboot2 header (what GRUB's `multiboot2` loads), and `xen.efi`, an EFI application that
//! takes its configuration, dom0 kernel and ramdisk from `LoadFile2` media loaders (Edera's
//! Xen, `docs/plans/18-xen-dom0.md`). The UEFI firmware boots either; this module only works out
//! which one it was given.

use super::inflate;
use super::uefi::has_efi_header;
use crate::machine::Error;
#[cfg(freestanding)]
use crate::prelude::*;

/// Multiboot2 specification 3.1.1: the header is 8-byte aligned in the first 32 KiB of the
/// image, and magic + architecture + length + checksum sum to zero.
pub(crate) const MULTIBOOT2_HEADER_MAGIC: u32 = 0xe852_50d6;
const MULTIBOOT2_SEARCH_BYTES: usize = 32 * 1024;
const MULTIBOOT2_HEADER_ALIGN: usize = 8;
const MULTIBOOT2_ARCHITECTURE_I386: u32 = 0;
/// magic, architecture, header length, checksum.
const MULTIBOOT2_HEADER_FIXED_BYTES: usize = 16;
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";
/// A decompressed Xen is 4 MiB; anything far past that is not a hypervisor (or is a gzip bomb).
const MAX_HYPERVISOR_BYTES: usize = 64 << 20;

/// Offset of the image's Multiboot2 header, if it has a valid one.
pub(crate) fn multiboot2_header(image: &[u8]) -> Option<usize> {
    let word = |at: usize| {
        image
            .get(at..at + 4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    let searched = image.len().min(MULTIBOOT2_SEARCH_BYTES);
    (0..searched.saturating_sub(MULTIBOOT2_HEADER_FIXED_BYTES - 1))
        .step_by(MULTIBOOT2_HEADER_ALIGN)
        .find(|&at| {
            let (Some(magic), Some(architecture), Some(length), Some(checksum)) =
                (word(at), word(at + 4), word(at + 8), word(at + 12))
            else {
                return false;
            };
            magic == MULTIBOOT2_HEADER_MAGIC
                && architecture == MULTIBOOT2_ARCHITECTURE_I386
                && magic
                    .wrapping_add(architecture)
                    .wrapping_add(length)
                    .wrapping_add(checksum)
                    == 0
        })
}

/// `bytes` decompressed if they are gzipped (the firmware cannot inflate), once they are known
/// to be one of the two forms the firmware boots.
pub(crate) fn prepare(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let bad = |message: &str| Error::BadConfig(format!("hypervisor image: {message}"));
    let image = if inflate::is_gzip(bytes) {
        inflate::gunzip(bytes, MAX_HYPERVISOR_BYTES).map_err(bad)?
    } else {
        bytes.to_vec()
    };
    let multiboot2 = image.starts_with(ELF_MAGIC) && multiboot2_header(&image).is_some();
    if !multiboot2 && !has_efi_header(&image) {
        return Err(bad(
            "neither a Multiboot2 ELF (xen.gz, xen) nor an EFI application (xen.efi)",
        ));
    }
    Ok(image)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An ELF-looking image with a Multiboot2 header (no tags past the end tag) at `offset`.
    pub(crate) fn multiboot2_image(offset: usize) -> Vec<u8> {
        const END_TAG_BYTES: u32 = 8;
        let mut image = vec![0u8; offset + 64];
        image[..4].copy_from_slice(ELF_MAGIC);
        let length = MULTIBOOT2_HEADER_FIXED_BYTES as u32 + END_TAG_BYTES;
        let checksum = 0u32
            .wrapping_sub(MULTIBOOT2_HEADER_MAGIC)
            .wrapping_sub(MULTIBOOT2_ARCHITECTURE_I386)
            .wrapping_sub(length);
        for (index, word) in [
            MULTIBOOT2_HEADER_MAGIC,
            MULTIBOOT2_ARCHITECTURE_I386,
            length,
            checksum,
        ]
        .into_iter()
        .enumerate()
        {
            image[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        image
    }

    #[test]
    fn finds_an_aligned_multiboot2_header_with_a_good_checksum() {
        assert_eq!(multiboot2_header(&multiboot2_image(0x98)), Some(0x98));
        // Not 8-byte aligned, past the first 32 KiB, or a bad checksum: not a header.
        assert_eq!(multiboot2_header(&multiboot2_image(0x9c)), None);
        assert_eq!(multiboot2_header(&multiboot2_image(0x8000)), None);
        let mut broken = multiboot2_image(0x98);
        broken[0x98 + 12] ^= 1;
        assert_eq!(multiboot2_header(&broken), None);
    }

    #[test]
    fn tells_xen_gz_from_xen_efi_and_refuses_anything_else() {
        let elf = multiboot2_image(0x98);
        assert_eq!(prepare(&elf).unwrap(), elf);
        let mut pe = vec![0u8; 0x100];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c] = 0x80;
        pe[0x80..0x84].copy_from_slice(b"PE\0\0");
        assert_eq!(prepare(&pe).unwrap(), pe);
        let Err(Error::BadConfig(message)) = prepare(b"\x7fELF and nothing else") else {
            panic!("an ELF without a Multiboot2 header was accepted");
        };
        assert!(message.contains("xen.gz"), "{message}");
        assert!(prepare(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0, 0xff]).is_err());
    }
}
