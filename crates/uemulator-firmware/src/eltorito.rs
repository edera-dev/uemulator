//! El Torito: the boot catalog of an ISO 9660 image, which is how a CD names the boot images it
//! carries (El Torito 1.0, "Bootable CD-ROM Format Specification"). A UEFI boot image is a FAT
//! file system the firmware publishes as a volume of its own, like a partition, so the boot
//! manager finds `\EFI\BOOT\BOOTX64.EFI` in it.
//!
//! Most Linux ISOs are hybrids whose GPT or MBR already names that FAT image, and `gpt` finds it.
//! This is for the rest (netboot.xyz, many rescue and vendor images): a plain ISO 9660 image
//! attached as a disk. Sectors here are the disk's 512 bytes; the ISO's own are 2048.

use crate::gpt::Blocks;

/// ISO 9660 sectors ("logical blocks"), which every CD address in the catalog counts.
const CD_SECTOR_BYTES: u64 = 2048;
/// El Torito counts the sectors of a boot image in "virtual sectors" of 512 bytes.
const VIRTUAL_SECTOR_BYTES: u64 = 512;
/// The boot record volume descriptor is always the 18th CD sector (ISO 9660 puts the primary
/// volume descriptor at 16; El Torito 2.0 fixes its own at 17).
const BOOT_RECORD_SECTOR: u64 = 17;
const VOLUME_DESCRIPTOR_BOOT_RECORD: u8 = 0;
const STANDARD_IDENTIFIER: &[u8; 5] = b"CD001";
const BOOT_SYSTEM_IDENTIFIER: &[u8] = b"EL TORITO SPECIFICATION";
/// Where the boot record keeps the catalog's CD sector.
const CATALOG_POINTER: usize = 0x47;
const ENTRY_BYTES: usize = 32;
const HEADER_VALIDATION: u8 = 0x01;
const HEADER_SECTION_MORE: u8 = 0x90;
const HEADER_SECTION_LAST: u8 = 0x91;
const ENTRY_BOOTABLE: u8 = 0x88;
/// Follows an entry whose media byte has `MEDIA_EXTENSION_FOLLOWS`; carries no image.
const ENTRY_EXTENSION: u8 = 0x44;
const VALIDATION_KEY: [u8; 2] = [0x55, 0xaa];
/// The platform ID UEFI 2.9 13.3.2.1 assigned for EFI boot images (0 is x86 BIOS).
pub const PLATFORM_EFI: u8 = 0xef;
const MEDIA_TYPE_MASK: u8 = 0x0f;
const MEDIA_NO_EMULATION: u8 = 0;
const MEDIA_FLOPPY_1200K: u8 = 1;
const MEDIA_FLOPPY_1440K: u8 = 2;
const MEDIA_FLOPPY_2880K: u8 = 3;
/// A FAT boot sector's total-sector fields: the 16-bit one, and the 32-bit one used when it is 0.
const BPB_BYTES_PER_SECTOR: usize = 11;
const BPB_TOTAL_SECTORS_16: usize = 19;
const BPB_TOTAL_SECTORS_32: usize = 32;

/// A boot image on the medium, in the disk's 512-byte blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootImage {
    /// Its index in the catalog (the default entry is 0), for the CD-ROM device path node.
    pub entry: u32,
    /// Where the image starts, as the CD sector the catalog gives (the device path's RBA).
    pub cd_sector: u64,
    pub first_block: u64,
    pub block_count: u64,
}

fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([data[at], data[at + 1]])
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

fn read_cd_sector(disk: &mut impl Blocks, sector: u64) -> Option<alloc::vec::Vec<u8>> {
    let block = disk.block_bytes();
    if block == 0 || !CD_SECTOR_BYTES.is_multiple_of(block) {
        return None;
    }
    let first = sector.checked_mul(CD_SECTOR_BYTES / block)?;
    if first.checked_add(CD_SECTOR_BYTES / block)? > disk.block_count() {
        return None;
    }
    let mut buffer = alloc::vec![0u8; CD_SECTOR_BYTES as usize];
    disk.read_blocks(first, &mut buffer).then_some(buffer)
}

/// The words of the validation entry sum to zero, and it ends in the 55 AA key.
fn valid_header(entry: &[u8]) -> bool {
    let sum = (0..ENTRY_BYTES)
        .step_by(2)
        .fold(0u16, |sum, at| sum.wrapping_add(u16_at(entry, at)));
    entry[0] == HEADER_VALIDATION && entry[30..32] == VALIDATION_KEY && sum == 0
}

/// The first bootable EFI image in the catalog, or `None` when the disk is not an El Torito CD
/// image or names none.
pub fn efi_boot_image(disk: &mut impl Blocks) -> Option<BootImage> {
    let record = read_cd_sector(disk, BOOT_RECORD_SECTOR)?;
    if record[0] != VOLUME_DESCRIPTOR_BOOT_RECORD
        || &record[1..6] != STANDARD_IDENTIFIER
        || !record[7..].starts_with(BOOT_SYSTEM_IDENTIFIER)
    {
        return None;
    }
    let catalog_sector = u64::from(u32_at(&record, CATALOG_POINTER));
    let catalog = read_cd_sector(disk, catalog_sector)?;
    let mut entries = catalog.as_chunks::<ENTRY_BYTES>().0.iter();
    let validation = entries.next()?;
    if !valid_header(validation) {
        return None;
    }
    // The default entry belongs to the platform the validation entry names; each section header
    // names the platform of the entries after it.
    let mut platform = validation[1];
    let mut last_section = false;
    let mut image_index = 0u32;
    for entry in entries {
        match entry[0] {
            HEADER_SECTION_MORE | HEADER_SECTION_LAST => {
                // A header after the last section's entries ends the catalog.
                if last_section {
                    return None;
                }
                last_section = entry[0] == HEADER_SECTION_LAST;
                platform = entry[1];
            }
            ENTRY_EXTENSION => {}
            ENTRY_BOOTABLE if platform == PLATFORM_EFI => {
                if let Some(image) = image_at(disk, entry, image_index) {
                    return Some(image);
                }
                image_index += 1;
            }
            // A non-bootable entry still counts; anything else (zeros) ends the catalog.
            0 if entry.iter().all(|&byte| byte == 0) => return None,
            _ => image_index += 1,
        }
    }
    None
}

fn image_at(disk: &mut impl Blocks, entry: &[u8], index: u32) -> Option<BootImage> {
    let block = disk.block_bytes();
    let cd_sector = u64::from(u32_at(entry, 8));
    let first_block = cd_sector.checked_mul(CD_SECTOR_BYTES / block)?;
    let remaining = disk.block_count().checked_sub(first_block)?;
    if remaining == 0 {
        return None;
    }
    let image_bytes = match entry[1] & MEDIA_TYPE_MASK {
        MEDIA_FLOPPY_1200K => 1200 * 1024,
        MEDIA_FLOPPY_1440K => 1440 * 1024,
        MEDIA_FLOPPY_2880K => 2880 * 1024,
        MEDIA_NO_EMULATION => {
            let sectors = u64::from(u16_at(entry, 6));
            // mkisofs and xorriso write 0 or 1 when the image is too big for the 16-bit count
            // (more than 32 MiB) or when they did not bother; EDK2 then takes the rest of the
            // medium. The image's own FAT boot sector knows better, so ask it first.
            if sectors > 1 {
                sectors * VIRTUAL_SECTOR_BYTES
            } else {
                fat_size(disk, first_block).unwrap_or(remaining * block)
            }
        }
        // Hard-disk emulation: an MBR disk of its own, which the firmware does not look into.
        _ => return None,
    };
    Some(BootImage {
        entry: index,
        cd_sector,
        first_block,
        block_count: image_bytes.div_ceil(block).min(remaining),
    })
}

/// The size a FAT boot sector at `first_block` claims, in bytes.
fn fat_size(disk: &mut impl Blocks, first_block: u64) -> Option<u64> {
    let mut sector = alloc::vec![0u8; disk.block_bytes() as usize];
    if sector.len() < 512 || !disk.read_blocks(first_block, &mut sector) {
        return None;
    }
    let bytes_per_sector = u64::from(u16_at(&sector, BPB_BYTES_PER_SECTOR));
    let total = match u16_at(&sector, BPB_TOTAL_SECTORS_16) {
        0 => u64::from(u32_at(&sector, BPB_TOTAL_SECTORS_32)),
        sectors => u64::from(sectors),
    };
    let valid_sector_size =
        bytes_per_sector.is_power_of_two() && (512..=4096).contains(&bytes_per_sector);
    (sector[510..512] == VALIDATION_KEY && valid_sector_size && total != 0)
        .then(|| total * bytes_per_sector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::testing::Ram;

    const CATALOG_SECTOR: u64 = 20;
    const IMAGE_SECTOR: u64 = 24;
    const DISK_CD_SECTORS: u64 = 64;

    fn disk() -> Ram {
        let mut bytes = alloc::vec![0u8; (DISK_CD_SECTORS * CD_SECTOR_BYTES) as usize];
        let record = &mut bytes[(BOOT_RECORD_SECTOR * CD_SECTOR_BYTES) as usize..];
        record[0] = VOLUME_DESCRIPTOR_BOOT_RECORD;
        record[1..6].copy_from_slice(STANDARD_IDENTIFIER);
        record[6] = 1;
        record[7..7 + BOOT_SYSTEM_IDENTIFIER.len()].copy_from_slice(BOOT_SYSTEM_IDENTIFIER);
        record[CATALOG_POINTER..CATALOG_POINTER + 4]
            .copy_from_slice(&(CATALOG_SECTOR as u32).to_le_bytes());
        Ram { bytes, block: 512 }
    }

    fn validation(platform: u8) -> [u8; ENTRY_BYTES] {
        let mut entry = [0u8; ENTRY_BYTES];
        entry[0] = HEADER_VALIDATION;
        entry[1] = platform;
        entry[30..32].copy_from_slice(&VALIDATION_KEY);
        let sum = (0..ENTRY_BYTES)
            .step_by(2)
            .fold(0u16, |sum, at| sum.wrapping_add(u16_at(&entry, at)));
        entry[28..30].copy_from_slice(&0u16.wrapping_sub(sum).to_le_bytes());
        entry
    }

    fn boot_entry(media: u8, sectors: u16, cd_sector: u64) -> [u8; ENTRY_BYTES] {
        let mut entry = [0u8; ENTRY_BYTES];
        entry[0] = ENTRY_BOOTABLE;
        entry[1] = media;
        entry[6..8].copy_from_slice(&sectors.to_le_bytes());
        entry[8..12].copy_from_slice(&(cd_sector as u32).to_le_bytes());
        entry
    }

    fn section(kind: u8, platform: u8, count: u16) -> [u8; ENTRY_BYTES] {
        let mut entry = [0u8; ENTRY_BYTES];
        entry[0] = kind;
        entry[1] = platform;
        entry[2..4].copy_from_slice(&count.to_le_bytes());
        entry
    }

    fn with_catalog(entries: &[[u8; ENTRY_BYTES]]) -> Ram {
        let mut disk = disk();
        let at = (CATALOG_SECTOR * CD_SECTOR_BYTES) as usize;
        for (index, entry) in entries.iter().enumerate() {
            disk.bytes[at + index * ENTRY_BYTES..][..ENTRY_BYTES].copy_from_slice(entry);
        }
        disk
    }

    /// A FAT boot sector claiming `sectors` 512-byte sectors at the image.
    fn put_fat_boot_sector(disk: &mut Ram, sectors: u16) {
        let at = (IMAGE_SECTOR * CD_SECTOR_BYTES) as usize;
        let sector = &mut disk.bytes[at..at + 512];
        sector[BPB_BYTES_PER_SECTOR..BPB_BYTES_PER_SECTOR + 2]
            .copy_from_slice(&512u16.to_le_bytes());
        sector[BPB_TOTAL_SECTORS_16..BPB_TOTAL_SECTORS_16 + 2]
            .copy_from_slice(&sectors.to_le_bytes());
        sector[510..512].copy_from_slice(&VALIDATION_KEY);
    }

    #[test]
    fn an_efi_section_after_the_bios_default_is_found() {
        // netboot.xyz's layout: a BIOS default entry, then one EFI section.
        let mut disk = with_catalog(&[
            validation(0),
            boot_entry(MEDIA_NO_EMULATION, 4, 30),
            section(HEADER_SECTION_LAST, PLATFORM_EFI, 1),
            boot_entry(MEDIA_NO_EMULATION, 100, IMAGE_SECTOR),
        ]);
        assert_eq!(
            efi_boot_image(&mut disk),
            Some(BootImage {
                entry: 1,
                cd_sector: IMAGE_SECTOR,
                first_block: IMAGE_SECTOR * 4,
                block_count: 100,
            })
        );
    }

    #[test]
    fn an_efi_default_entry_is_found() {
        let mut disk = with_catalog(&[
            validation(PLATFORM_EFI),
            boot_entry(MEDIA_NO_EMULATION, 16, IMAGE_SECTOR),
        ]);
        let image = efi_boot_image(&mut disk).unwrap();
        assert_eq!(
            (image.entry, image.first_block, image.block_count),
            (0, IMAGE_SECTOR * 4, 16)
        );
    }

    #[test]
    fn a_bios_only_catalog_has_no_efi_image() {
        // CorePlus (Tiny Core): isolinux and nothing for UEFI.
        let mut disk = with_catalog(&[validation(0), boot_entry(MEDIA_NO_EMULATION, 4, 30)]);
        assert_eq!(efi_boot_image(&mut disk), None);
    }

    #[test]
    fn a_zero_sector_count_takes_the_size_from_the_fat_boot_sector() {
        let mut disk = with_catalog(&[
            validation(PLATFORM_EFI),
            boot_entry(MEDIA_NO_EMULATION, 0, IMAGE_SECTOR),
        ]);
        put_fat_boot_sector(&mut disk, 40);
        assert_eq!(efi_boot_image(&mut disk).unwrap().block_count, 40);
    }

    #[test]
    fn a_zero_sector_count_without_fat_runs_to_the_end_of_the_disk() {
        let mut disk = with_catalog(&[
            validation(PLATFORM_EFI),
            boot_entry(MEDIA_NO_EMULATION, 1, IMAGE_SECTOR),
        ]);
        let image = efi_boot_image(&mut disk).unwrap();
        assert_eq!(image.block_count, (DISK_CD_SECTORS - IMAGE_SECTOR) * 4);
    }

    #[test]
    fn an_image_is_clamped_to_the_disk() {
        let mut disk = with_catalog(&[
            validation(PLATFORM_EFI),
            boot_entry(MEDIA_NO_EMULATION, 60000, IMAGE_SECTOR),
        ]);
        assert_eq!(
            efi_boot_image(&mut disk).unwrap().block_count,
            (DISK_CD_SECTORS - IMAGE_SECTOR) * 4
        );
    }

    #[test]
    fn floppy_emulation_has_the_floppy_size() {
        let mut disk = with_catalog(&[
            validation(PLATFORM_EFI),
            boot_entry(MEDIA_FLOPPY_1440K, 1, IMAGE_SECTOR),
        ]);
        // Smaller than 1.44 MB here, so clamped; the media type, not the count, decides.
        assert_eq!(
            efi_boot_image(&mut disk).unwrap().block_count,
            (DISK_CD_SECTORS - IMAGE_SECTOR) * 4
        );
    }

    #[test]
    fn a_bad_validation_checksum_is_refused() {
        let mut header = validation(PLATFORM_EFI);
        header[4] ^= 1;
        let mut disk = with_catalog(&[header, boot_entry(MEDIA_NO_EMULATION, 16, IMAGE_SECTOR)]);
        assert_eq!(efi_boot_image(&mut disk), None);
    }

    #[test]
    fn a_disk_without_a_boot_record_is_not_a_cd() {
        let mut disk = Ram {
            bytes: alloc::vec![0u8; 64 * 2048],
            block: 512,
        };
        assert_eq!(efi_boot_image(&mut disk), None);
        let mut short = Ram {
            bytes: alloc::vec![0u8; 4096],
            block: 512,
        };
        assert_eq!(efi_boot_image(&mut short), None);
    }

    #[test]
    fn non_bootable_and_extension_entries_are_skipped() {
        let mut not_bootable = boot_entry(MEDIA_NO_EMULATION, 16, 40);
        not_bootable[0] = 0;
        let mut extension = [0u8; ENTRY_BYTES];
        extension[0] = ENTRY_EXTENSION;
        let mut disk = with_catalog(&[
            validation(0),
            boot_entry(MEDIA_NO_EMULATION, 4, 30),
            section(HEADER_SECTION_LAST, PLATFORM_EFI, 2),
            not_bootable,
            extension,
            boot_entry(MEDIA_NO_EMULATION, 16, IMAGE_SECTOR),
        ]);
        let image = efi_boot_image(&mut disk).unwrap();
        assert_eq!((image.entry, image.first_block), (2, IMAGE_SECTOR * 4));
    }
}
