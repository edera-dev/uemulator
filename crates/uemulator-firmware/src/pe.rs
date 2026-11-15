//! A PE32+ loader for x86_64 UEFI images: headers, sections and base relocations.

use alloc::vec::Vec;

const DOS_MAGIC: u16 = 0x5a4d;
const PE_SIGNATURE: u32 = 0x0000_4550;
const MACHINE_AMD64: u16 = 0x8664;
const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x20b;
const DOS_PE_OFFSET_FIELD: usize = 0x3c;
const COFF_HEADER_BYTES: usize = 20;
const SECTION_HEADER_BYTES: usize = 40;
const DATA_DIRECTORY_BASE_RELOCATION: usize = 5;
const DATA_DIRECTORY_BYTES: usize = 8;
/// Offsets within the optional header (PE32+).
const OPT_ENTRY_POINT: usize = 16;
const OPT_IMAGE_BASE: usize = 24;
const OPT_SECTION_ALIGNMENT: usize = 32;
const OPT_SIZE_OF_IMAGE: usize = 56;
const OPT_SIZE_OF_HEADERS: usize = 60;
const OPT_SUBSYSTEM: usize = 68;
const OPT_NUMBER_OF_DIRECTORIES: usize = 108;
const OPT_DIRECTORIES: usize = 112;

const RELOCATION_ABSOLUTE: u16 = 0;
const RELOCATION_HIGH: u16 = 1;
const RELOCATION_LOW: u16 = 2;
const RELOCATION_HIGHLOW: u16 = 3;
const RELOCATION_DIR64: u16 = 10;
const RELOCATION_BLOCK_HEADER_BYTES: usize = 8;
const RELOCATION_OFFSET_MASK: u16 = 0x0fff;
const RELOCATION_TYPE_SHIFT: u32 = 12;

pub const SUBSYSTEM_EFI_APPLICATION: u16 = 10;
pub const SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER: u16 = 11;
pub const SUBSYSTEM_EFI_RUNTIME_DRIVER: u16 = 12;

#[derive(Debug, PartialEq, Eq)]
pub enum PeError {
    NotPe,
    /// A valid PE that is not for this machine (32-bit, ARM, ...).
    Unsupported,
    Corrupt,
}

#[derive(Debug)]
struct Section {
    virtual_address: u32,
    virtual_size: u32,
    raw_offset: u32,
    raw_size: u32,
}

#[derive(Debug)]
pub struct Image {
    pub image_base: u64,
    pub size_of_image: u32,
    pub section_alignment: u32,
    pub entry_point: u32,
    pub subsystem: u16,
    headers_size: u32,
    sections: Vec<Section>,
    relocations: (u32, u32),
}

fn u16_at(data: &[u8], at: usize) -> Result<u16, PeError> {
    data.get(at..at + 2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().unwrap()))
        .ok_or(PeError::Corrupt)
}

fn u32_at(data: &[u8], at: usize) -> Result<u32, PeError> {
    data.get(at..at + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .ok_or(PeError::Corrupt)
}

fn u64_at(data: &[u8], at: usize) -> Result<u64, PeError> {
    data.get(at..at + 8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .ok_or(PeError::Corrupt)
}

impl Image {
    pub fn parse(file: &[u8]) -> Result<Self, PeError> {
        if u16_at(file, 0).map_err(|_| PeError::NotPe)? != DOS_MAGIC {
            return Err(PeError::NotPe);
        }
        let pe = u32_at(file, DOS_PE_OFFSET_FIELD)? as usize;
        if u32_at(file, pe)? != PE_SIGNATURE {
            return Err(PeError::NotPe);
        }
        let coff = pe + 4;
        if u16_at(file, coff)? != MACHINE_AMD64 {
            return Err(PeError::Unsupported);
        }
        let section_count = u16_at(file, coff + 2)? as usize;
        let optional_size = u16_at(file, coff + 16)? as usize;
        let optional = coff + COFF_HEADER_BYTES;
        if u16_at(file, optional)? != OPTIONAL_MAGIC_PE32_PLUS {
            return Err(PeError::Unsupported);
        }
        let directory_count = u32_at(file, optional + OPT_NUMBER_OF_DIRECTORIES)? as usize;
        let relocations = if directory_count > DATA_DIRECTORY_BASE_RELOCATION
            && optional_size
                >= OPT_DIRECTORIES + (DATA_DIRECTORY_BASE_RELOCATION + 1) * DATA_DIRECTORY_BYTES
        {
            let at =
                optional + OPT_DIRECTORIES + DATA_DIRECTORY_BASE_RELOCATION * DATA_DIRECTORY_BYTES;
            (u32_at(file, at)?, u32_at(file, at + 4)?)
        } else {
            (0, 0)
        };
        let sections_at = optional + optional_size;
        let headers_size = u32_at(file, optional + OPT_SIZE_OF_HEADERS)?;
        if sections_at
            .checked_add(section_count * SECTION_HEADER_BYTES)
            .is_none_or(|end| end > headers_size as usize)
        {
            return Err(PeError::Corrupt);
        }
        let sections = (0..section_count)
            .map(|index| {
                let at = sections_at + index * SECTION_HEADER_BYTES;
                Ok(Section {
                    virtual_size: u32_at(file, at + 8)?,
                    virtual_address: u32_at(file, at + 12)?,
                    raw_size: u32_at(file, at + 16)?,
                    raw_offset: u32_at(file, at + 20)?,
                })
            })
            .collect::<Result<Vec<_>, PeError>>()?;
        let image = Image {
            image_base: u64_at(file, optional + OPT_IMAGE_BASE)?,
            size_of_image: u32_at(file, optional + OPT_SIZE_OF_IMAGE)?,
            section_alignment: u32_at(file, optional + OPT_SECTION_ALIGNMENT)?,
            entry_point: u32_at(file, optional + OPT_ENTRY_POINT)?,
            subsystem: u16_at(file, optional + OPT_SUBSYSTEM)?,
            headers_size,
            sections,
            relocations,
        };
        image.check_layout(file.len())?;
        Ok(image)
    }

    /// Every region the loader will touch must lie inside the file and the image.
    fn check_layout(&self, file_len: usize) -> Result<(), PeError> {
        let size = self.size_of_image as u64;
        if size == 0 || self.headers_size as u64 > size || self.headers_size as usize > file_len {
            return Err(PeError::Corrupt);
        }
        for section in &self.sections {
            let in_image =
                section.virtual_address as u64 + section.virtual_size.max(section.raw_size) as u64;
            let in_file = section.raw_offset as u64 + section.raw_size as u64;
            if in_image > size || in_file > file_len as u64 {
                return Err(PeError::Corrupt);
            }
        }
        // Authenticode excludes CheckSum and the certificate-directory entry from the
        // header digest. They are copied into memory, so executing from the headers would
        // let those mutable bytes become the image's first instructions.
        if self.entry_point < self.headers_size || self.entry_point >= self.size_of_image {
            return Err(PeError::Corrupt);
        }
        let (rva, length) = self.relocations;
        if length != 0 && rva as u64 + length as u64 > size {
            return Err(PeError::Corrupt);
        }
        Ok(())
    }

    /// Lays the image out in `memory` (at least `size_of_image` zeroed bytes that will sit at
    /// `load_address`) and applies the relocations for that address.
    pub fn load(&self, file: &[u8], memory: &mut [u8], load_address: u64) -> Result<(), PeError> {
        if (memory.len() as u64) < self.size_of_image as u64 {
            return Err(PeError::Corrupt);
        }
        memory[..self.headers_size as usize].copy_from_slice(&file[..self.headers_size as usize]);
        for section in &self.sections {
            let start = section.virtual_address as usize;
            let raw = &file[section.raw_offset as usize..][..section.raw_size as usize];
            // The part of the virtual size beyond the raw data is zero-initialised, which the
            // caller's zeroed memory already provides.
            let copied = raw
                .len()
                .min(section.virtual_size.max(section.raw_size) as usize);
            memory[start..start + copied].copy_from_slice(&raw[..copied]);
        }
        self.relocate(memory, load_address)
    }

    fn relocate(&self, memory: &mut [u8], load_address: u64) -> Result<(), PeError> {
        let (rva, length) = self.relocations;
        let delta = load_address.wrapping_sub(self.image_base);
        if length == 0 || delta == 0 {
            return Ok(());
        }
        let table_end = (rva + length) as usize;
        let mut at = rva as usize;
        while at + RELOCATION_BLOCK_HEADER_BYTES <= table_end {
            let page = u32_at(memory, at)? as usize;
            let block_size = u32_at(memory, at + 4)? as usize;
            if block_size < RELOCATION_BLOCK_HEADER_BYTES || at + block_size > table_end {
                return Err(PeError::Corrupt);
            }
            for entry in (at + RELOCATION_BLOCK_HEADER_BYTES..at + block_size).step_by(2) {
                let entry = u16_at(memory, entry)?;
                let target = page + (entry & RELOCATION_OFFSET_MASK) as usize;
                let patch = |memory: &mut [u8], width: usize, apply: &dyn Fn(u64) -> u64| {
                    let bytes = memory
                        .get_mut(target..target + width)
                        .ok_or(PeError::Corrupt)?;
                    let mut value = [0u8; 8];
                    value[..width].copy_from_slice(bytes);
                    let patched = apply(u64::from_le_bytes(value)).to_le_bytes();
                    bytes.copy_from_slice(&patched[..width]);
                    Ok(())
                };
                match entry >> RELOCATION_TYPE_SHIFT {
                    RELOCATION_ABSOLUTE => {}
                    RELOCATION_DIR64 => patch(memory, 8, &|value| value.wrapping_add(delta))?,
                    RELOCATION_HIGHLOW => patch(memory, 4, &|value| value.wrapping_add(delta))?,
                    RELOCATION_HIGH => patch(memory, 2, &|value| value.wrapping_add(delta >> 16))?,
                    RELOCATION_LOW => patch(memory, 2, &|value| value.wrapping_add(delta))?,
                    _ => return Err(PeError::Unsupported),
                }
            }
            at += block_size;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_BASE: u64 = 0x1_4000_0000;
    const TEXT_RVA: u32 = 0x1000;
    const RELOC_RVA: u32 = 0x2000;
    const SIZE_OF_IMAGE: u32 = 0x3000;
    const FILE_ALIGN: usize = 0x200;
    const OPTIONAL_SIZE: usize = OPT_DIRECTORIES + 16 * DATA_DIRECTORY_BYTES;

    #[test]
    fn section_table_must_fit_in_size_of_headers() {
        let mut file = build(MACHINE_AMD64);
        let pe = u32_at(&file, DOS_PE_OFFSET_FIELD).unwrap() as usize;
        let optional = pe + 4 + COFF_HEADER_BYTES;
        let sections = optional + OPTIONAL_SIZE;
        file[optional + OPT_SIZE_OF_HEADERS..][..4]
            .copy_from_slice(&(sections as u32).to_le_bytes());
        assert_eq!(Image::parse(&file).unwrap_err(), PeError::Corrupt);
    }

    #[test]
    fn entry_point_must_not_run_unhashed_header_bytes() {
        let mut file = build(MACHINE_AMD64);
        let pe = u32_at(&file, DOS_PE_OFFSET_FIELD).unwrap() as usize;
        let optional = pe + 4 + COFF_HEADER_BYTES;
        // Authenticode deliberately excludes CheckSum. It must never be executable entry code.
        let checksum = (optional + 64) as u32;
        file[optional + OPT_ENTRY_POINT..][..4].copy_from_slice(&checksum.to_le_bytes());
        let before = uemulator_crypto::authenticode::digest(
            &file,
            uemulator_crypto::sha::HashAlgorithm::Sha256,
        )
        .unwrap();
        file[checksum as usize] ^= 0x5a;
        assert_eq!(
            uemulator_crypto::authenticode::digest(
                &file,
                uemulator_crypto::sha::HashAlgorithm::Sha256
            )
            .unwrap(),
            before,
            "the would-be entry code is outside the signed digest"
        );
        assert_eq!(Image::parse(&file).unwrap_err(), PeError::Corrupt);
    }

    /// A two-section PE: `.text` holds a pointer to the image's own entry point at offset 8 and
    /// `.reloc` has one DIR64 entry for it.
    fn build(machine: u16) -> Vec<u8> {
        let mut file = vec![0u8; 0x600];
        file[0..2].copy_from_slice(&DOS_MAGIC.to_le_bytes());
        file[DOS_PE_OFFSET_FIELD] = 0x40;
        let pe = 0x40;
        file[pe..pe + 4].copy_from_slice(&PE_SIGNATURE.to_le_bytes());
        let coff = pe + 4;
        file[coff..coff + 2].copy_from_slice(&machine.to_le_bytes());
        file[coff + 2..coff + 4].copy_from_slice(&2u16.to_le_bytes());
        file[coff + 16..coff + 18].copy_from_slice(&(OPTIONAL_SIZE as u16).to_le_bytes());
        let opt = coff + COFF_HEADER_BYTES;
        file[opt..opt + 2].copy_from_slice(&OPTIONAL_MAGIC_PE32_PLUS.to_le_bytes());
        file[opt + OPT_ENTRY_POINT..][..4].copy_from_slice(&TEXT_RVA.to_le_bytes());
        file[opt + OPT_IMAGE_BASE..][..8].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        file[opt + OPT_SECTION_ALIGNMENT..][..4].copy_from_slice(&0x1000u32.to_le_bytes());
        file[opt + OPT_SIZE_OF_IMAGE..][..4].copy_from_slice(&SIZE_OF_IMAGE.to_le_bytes());
        file[opt + OPT_SIZE_OF_HEADERS..][..4].copy_from_slice(&(FILE_ALIGN as u32).to_le_bytes());
        file[opt + OPT_SUBSYSTEM..][..2].copy_from_slice(&SUBSYSTEM_EFI_APPLICATION.to_le_bytes());
        file[opt + OPT_NUMBER_OF_DIRECTORIES..][..4].copy_from_slice(&16u32.to_le_bytes());
        let directory =
            opt + OPT_DIRECTORIES + DATA_DIRECTORY_BASE_RELOCATION * DATA_DIRECTORY_BYTES;
        file[directory..directory + 4].copy_from_slice(&RELOC_RVA.to_le_bytes());
        file[directory + 4..directory + 8].copy_from_slice(&12u32.to_le_bytes());
        let sections = opt + OPTIONAL_SIZE;
        for (index, (rva, size, raw)) in [(TEXT_RVA, 0x800u32, 0x200u32), (RELOC_RVA, 12, 0x400)]
            .into_iter()
            .enumerate()
        {
            let at = sections + index * SECTION_HEADER_BYTES;
            file[at + 8..at + 12].copy_from_slice(&size.to_le_bytes());
            file[at + 12..at + 16].copy_from_slice(&rva.to_le_bytes());
            let raw_size = 0x200u32;
            file[at + 16..at + 20].copy_from_slice(&raw_size.to_le_bytes());
            file[at + 20..at + 24].copy_from_slice(&raw.to_le_bytes());
        }
        file[0x200 + 8..0x200 + 16].copy_from_slice(&(IMAGE_BASE + TEXT_RVA as u64).to_le_bytes());
        file[0x400..0x404].copy_from_slice(&TEXT_RVA.to_le_bytes());
        file[0x404..0x408].copy_from_slice(&12u32.to_le_bytes());
        let entry = (RELOCATION_DIR64 << RELOCATION_TYPE_SHIFT) | 8;
        file[0x408..0x40a].copy_from_slice(&entry.to_le_bytes());
        file
    }

    fn load_at(file: &[u8], address: u64) -> Vec<u8> {
        let image = Image::parse(file).unwrap();
        let mut memory = vec![0u8; image.size_of_image as usize];
        image.load(file, &mut memory, address).unwrap();
        memory
    }

    #[test]
    fn parses_the_headers() {
        let image = Image::parse(&build(MACHINE_AMD64)).unwrap();
        assert_eq!(image.image_base, IMAGE_BASE);
        assert_eq!(image.entry_point, TEXT_RVA);
        assert_eq!(image.subsystem, SUBSYSTEM_EFI_APPLICATION);
        assert_eq!(image.size_of_image, SIZE_OF_IMAGE);
    }

    #[test]
    fn sections_land_at_their_virtual_addresses() {
        let memory = load_at(&build(MACHINE_AMD64), IMAGE_BASE);
        assert_eq!(&memory[..2], b"MZ");
        assert_eq!(
            u64::from_le_bytes(memory[0x1008..0x1010].try_into().unwrap()),
            IMAGE_BASE + TEXT_RVA as u64
        );
        assert!(
            memory[0x1200..0x1800].iter().all(|&byte| byte == 0),
            "virtual size beyond the raw data is zero"
        );
    }

    #[test]
    fn relocations_move_pointers_by_the_load_delta() {
        let address = 0x7f00_0000u64;
        let memory = load_at(&build(MACHINE_AMD64), address);
        assert_eq!(
            u64::from_le_bytes(memory[0x1008..0x1010].try_into().unwrap()),
            address + TEXT_RVA as u64
        );
    }

    #[test]
    fn rejects_what_it_cannot_run() {
        assert_eq!(
            Image::parse(b"not a pe file at all").unwrap_err(),
            PeError::NotPe
        );
        assert_eq!(
            Image::parse(&build(0x14c)).unwrap_err(),
            PeError::Unsupported
        );
        let mut truncated = build(MACHINE_AMD64);
        truncated.truncate(0x300);
        assert_eq!(Image::parse(&truncated).unwrap_err(), PeError::Corrupt);
        let mut bad_entry = build(MACHINE_AMD64);
        let opt = 0x40 + 4 + COFF_HEADER_BYTES;
        bad_entry[opt + OPT_ENTRY_POINT..][..4].copy_from_slice(&SIZE_OF_IMAGE.to_le_bytes());
        assert_eq!(Image::parse(&bad_entry).unwrap_err(), PeError::Corrupt);
    }
}
