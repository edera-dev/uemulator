//! The hardware-independent half of booting a hypervisor with the supplied kernel as its dom0
//! (`hypervisor_boot.rs` is the other): telling `xen.efi` from Xen's Multiboot2 ELF, Xen's
//! configuration file for its media loaders, and for the ELF its Multiboot2 header, its load
//! segments and the boot information it is entered with (Multiboot2 specification 2.0, the
//! parts GRUB's `multiboot2` implements on UEFI).

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// A PE32+ EFI application: `xen.efi`.
    Efi,
    /// An ELF with a Multiboot2 header: Xen's `xen.gz`, which the library decompressed.
    Multiboot2,
}

const DOS_MAGIC: &[u8; 2] = b"MZ";
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// The library refuses anything else up front, so this only has to tell the two apart.
pub fn format(image: &[u8]) -> Option<Format> {
    if image.starts_with(DOS_MAGIC) {
        Some(Format::Efi)
    } else if image.starts_with(ELF_MAGIC) {
        Some(Format::Multiboot2)
    } else {
        None
    }
}

/// The first word of a command line handed to Xen. Xen drops the first word of its own and of
/// dom0's command line unless the boot loader calls itself GRUB 2 or is Xen's EFI loader
/// (`cmdline_cook` in xen/arch/x86/setup.c): Multiboot loaders put the image's file name there.
pub const XEN_IMAGE_NAME: &str = "xen.gz";
pub const KERNEL_IMAGE_NAME: &str = "vmlinuz";
pub const INITRD_IMAGE_NAME: &str = "initramfs";

/// What `xen.efi` reads through `XEN_EFI_CONFIG_MEDIA_GUID`: the configuration file format of
/// xen/common/efi/boot.c, with one section, as Edera's sprout builds it. The kernel's file name
/// is never opened (the kernel comes through its own media loader) but `kernel=` must be there
/// for Xen to look for one, and its first word is the name. The last line needs a newline, or
/// Xen ignores it. A newline inside either command line would end the entry early, so they
/// become spaces.
pub fn xen_media_config(xen_cmdline: &str, dom0_cmdline: &str) -> String {
    let one_line = |text: &str| text.replace(['\r', '\n'], " ");
    alloc::format!(
        "[global]\ndefault=xen\n[xen]\noptions={}\nkernel={KERNEL_IMAGE_NAME} {}\n",
        one_line(xen_cmdline),
        one_line(dom0_cmdline)
    )
}

pub mod multiboot2 {
    use super::*;

    pub const HEADER_MAGIC: u32 = 0xe852_50d6;
    /// What the boot loader leaves in `eax` (Multiboot2 specification 3.3).
    pub const BOOTLOADER_MAGIC: u32 = 0x36d7_6289;
    const SEARCH_BYTES: usize = 32 * 1024;
    const HEADER_ALIGN: usize = 8;
    const ARCHITECTURE_I386: u32 = 0;
    const HEADER_FIXED_BYTES: usize = 16;
    const TAG_ALIGN: usize = 8;
    /// Header tag flag bit 0: the boot loader may ignore the tag.
    const TAG_OPTIONAL: u16 = 1;

    const HEADER_TAG_END: u16 = 0;
    const HEADER_TAG_INFORMATION_REQUEST: u16 = 1;
    const HEADER_TAG_CONSOLE_FLAGS: u16 = 4;
    const HEADER_TAG_FRAMEBUFFER: u16 = 5;
    const HEADER_TAG_MODULE_ALIGN: u16 = 6;
    const HEADER_TAG_EFI_BOOT_SERVICES: u16 = 7;
    const HEADER_TAG_ENTRY_ADDRESS_EFI64: u16 = 9;
    const HEADER_TAG_RELOCATABLE: u16 = 10;

    const TAG_END: u32 = 0;
    const TAG_CMDLINE: u32 = 1;
    const TAG_BOOT_LOADER_NAME: u32 = 2;
    const TAG_MODULE: u32 = 3;
    const TAG_BASIC_MEMINFO: u32 = 4;
    const TAG_MMAP: u32 = 6;
    const TAG_EFI64: u32 = 12;
    const TAG_ACPI_OLD: u32 = 14;
    const TAG_ACPI_NEW: u32 = 15;
    const TAG_EFI_BOOT_SERVICES: u32 = 18;
    const TAG_EFI64_IMAGE_HANDLE: u32 = 20;
    const TAG_LOAD_BASE_ADDRESS: u32 = 21;
    /// The information this loader always provides, which is what a required information
    /// request may ask for.
    const PROVIDED: [u32; 10] = [
        TAG_CMDLINE,
        TAG_BOOT_LOADER_NAME,
        TAG_MODULE,
        TAG_BASIC_MEMINFO,
        TAG_MMAP,
        TAG_EFI64,
        TAG_ACPI_NEW,
        TAG_EFI_BOOT_SERVICES,
        TAG_EFI64_IMAGE_HANDLE,
        TAG_LOAD_BASE_ADDRESS,
    ];
    /// base u64, length u64, type u32, reserved u32.
    const MMAP_ENTRY_BYTES: u32 = 24;
    const MMAP_ENTRY_VERSION: u32 = 0;
    /// ACPI 1.0's RSDP is 20 bytes, 2.0's 36; byte 15 is the revision.
    const RSDP_V1_BYTES: usize = 20;
    const RSDP_REVISION_OFFSET: usize = 15;

    pub const MEMORY_AVAILABLE: u32 = 1;
    pub const MEMORY_RESERVED: u32 = 2;
    pub const MEMORY_ACPI_RECLAIMABLE: u32 = 3;
    pub const MEMORY_NVS: u32 = 4;
    pub const MEMORY_BAD: u32 = 5;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Relocatable {
        pub min_address: u32,
        /// The highest address the end of the image may be at.
        pub max_address: u32,
        pub alignment: u32,
        /// 0 none, 1 lowest, 2 highest.
        pub preference: u32,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct Header {
        pub keep_boot_services: bool,
        pub efi64_entry: Option<u32>,
        pub relocatable: Option<Relocatable>,
    }

    fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
        Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
    }

    fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
    }

    fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
        Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
    }

    /// Finds and reads the header (specification 3.1). A required tag this loader does not
    /// implement, or a required information request it cannot answer, is refused, as the
    /// specification asks.
    pub fn parse_header(image: &[u8]) -> Result<Header, &'static str> {
        let searched = image.len().min(SEARCH_BYTES);
        let start = (0..searched.saturating_sub(HEADER_FIXED_BYTES - 1))
            .step_by(HEADER_ALIGN)
            .find(|&at| {
                let words = [0, 4, 8, 12].map(|offset| u32_at(image, at + offset));
                let [
                    Some(magic),
                    Some(architecture),
                    Some(length),
                    Some(checksum),
                ] = words
                else {
                    return false;
                };
                magic == HEADER_MAGIC
                    && architecture == ARCHITECTURE_I386
                    && magic
                        .wrapping_add(architecture)
                        .wrapping_add(length)
                        .wrapping_add(checksum)
                        == 0
            })
            .ok_or("no Multiboot2 header")?;
        let length = u32_at(image, start + 8).unwrap() as usize;
        let end = start.checked_add(length).filter(|end| *end <= image.len());
        let end = end.ok_or("the Multiboot2 header runs past the image")?;
        let mut header = Header::default();
        let mut at = start + HEADER_FIXED_BYTES;
        while at + 8 <= end {
            let (kind, flags) = (u16_at(image, at).unwrap(), u16_at(image, at + 2).unwrap());
            let size = u32_at(image, at + 4).unwrap() as usize;
            if size < 8 || at + size > end {
                return Err("a Multiboot2 header tag runs past the header");
            }
            let required = flags & TAG_OPTIONAL == 0;
            let field = |offset: usize| u32_at(image, at + offset).filter(|_| offset + 4 <= size);
            match kind {
                HEADER_TAG_END => return Ok(header),
                HEADER_TAG_INFORMATION_REQUEST => {
                    let unanswerable = (8..size)
                        .step_by(4)
                        .filter_map(field)
                        .any(|wanted| !PROVIDED.contains(&wanted));
                    if required && unanswerable {
                        return Err("the image requires boot information this loader lacks");
                    }
                }
                HEADER_TAG_EFI_BOOT_SERVICES => header.keep_boot_services = true,
                HEADER_TAG_ENTRY_ADDRESS_EFI64 => header.efi64_entry = field(8),
                HEADER_TAG_RELOCATABLE => {
                    let (Some(min_address), Some(max_address), Some(alignment), Some(preference)) =
                        (field(8), field(12), field(16), field(20))
                    else {
                        return Err("a short relocatable tag");
                    };
                    header.relocatable = Some(Relocatable {
                        min_address,
                        max_address,
                        alignment,
                        preference,
                    });
                }
                // Page-aligned modules are what this loader hands over anyway; a text console
                // or a framebuffer mode are preferences.
                HEADER_TAG_MODULE_ALIGN | HEADER_TAG_CONSOLE_FLAGS | HEADER_TAG_FRAMEBUFFER => {}
                // The a.out kludge (address tags), a 32-bit entry address and the EFI32 entry are
                // not implemented: an ELF image with the EFI64 entry needs none of them.
                _ if required => {
                    return Err("the image requires a Multiboot2 feature this loader lacks");
                }
                _ => {}
            }
            at += size.next_multiple_of(TAG_ALIGN);
        }
        Err("the Multiboot2 header has no end tag")
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Segment {
        pub physical_address: u64,
        pub file_offset: usize,
        pub file_bytes: usize,
        pub memory_bytes: u64,
    }

    const ELF_CLASS_32: u8 = 1;
    const ELF_CLASS_64: u8 = 2;
    const ELF_DATA_LITTLE_ENDIAN: u8 = 1;
    const ELF_CLASS_OFFSET: usize = 4;
    const ELF_DATA_OFFSET: usize = 5;
    const PT_LOAD: u32 = 1;

    /// The `PT_LOAD` segments of a 32- or 64-bit little-endian ELF, by physical address (what a
    /// Multiboot loader loads at; Xen's virtual addresses are its own business).
    pub fn load_segments(image: &[u8]) -> Result<Vec<Segment>, &'static str> {
        const BAD: &str = "a malformed ELF image";
        if !image.starts_with(ELF_MAGIC)
            || image.get(ELF_DATA_OFFSET) != Some(&ELF_DATA_LITTLE_ENDIAN)
        {
            return Err("not a little-endian ELF image");
        }
        // (program header table offset, entry size, count) and the field offsets within an
        // entry: type, file offset, physical address, file size, memory size.
        let (table, entry_bytes, count, wide) = match image.get(ELF_CLASS_OFFSET) {
            Some(&ELF_CLASS_32) => (
                u32_at(image, 0x1c).ok_or(BAD)? as usize,
                u16_at(image, 0x2a).ok_or(BAD)? as usize,
                u16_at(image, 0x2c).ok_or(BAD)? as usize,
                false,
            ),
            Some(&ELF_CLASS_64) => (
                u64_at(image, 0x20).ok_or(BAD)? as usize,
                u16_at(image, 0x36).ok_or(BAD)? as usize,
                u16_at(image, 0x38).ok_or(BAD)? as usize,
                true,
            ),
            _ => return Err("unknown ELF class"),
        };
        let mut segments = Vec::new();
        for index in 0..count {
            let at = table
                .checked_add(index.checked_mul(entry_bytes).ok_or(BAD)?)
                .ok_or(BAD)?;
            if u32_at(image, at).ok_or(BAD)? != PT_LOAD {
                continue;
            }
            let field = |offset32: usize, offset64: usize| {
                if wide {
                    u64_at(image, at + offset64)
                } else {
                    u32_at(image, at + offset32).map(u64::from)
                }
            };
            let segment = Segment {
                file_offset: field(0x04, 0x08).ok_or(BAD)? as usize,
                physical_address: field(0x0c, 0x18).ok_or(BAD)?,
                file_bytes: field(0x10, 0x20).ok_or(BAD)? as usize,
                memory_bytes: field(0x14, 0x28).ok_or(BAD)?,
            };
            let in_file = segment
                .file_offset
                .checked_add(segment.file_bytes)
                .is_some_and(|end| end <= image.len());
            if !in_file || segment.file_bytes as u64 > segment.memory_bytes {
                return Err(BAD);
            }
            segments.push(segment);
        }
        if segments.is_empty() {
            return Err("the ELF image has nothing to load");
        }
        Ok(segments)
    }

    /// The physical range the segments cover as linked: (lowest address, bytes).
    pub fn span(segments: &[Segment]) -> Option<(u64, u64)> {
        let low = segments.iter().map(|s| s.physical_address).min()?;
        let high = segments
            .iter()
            .map(|s| s.physical_address.checked_add(s.memory_bytes))
            .try_fold(0u64, |high, end| Some(high.max(end?)))?;
        Some((low, high - low))
    }

    pub struct Module<'a> {
        pub start: u32,
        pub end: u32,
        pub cmdline: &'a str,
    }

    /// Everything the boot information carries, already in the form it is written in.
    pub struct Information<'a> {
        pub cmdline: &'a str,
        pub loader_name: &'a str,
        pub modules: &'a [Module<'a>],
        /// KiB below 640 KiB and KiB of RAM from 1 MiB up to the first hole.
        pub basic_memory: (u32, u32),
        /// (base, length, type) with the `MEMORY_*` types.
        pub memory_map: &'a [(u64, u64, u32)],
        pub efi_system_table: u64,
        pub efi_image_handle: u64,
        /// The RSDP's bytes, 20 for ACPI 1.0 or 36 for 2.0, when the machine has ACPI.
        pub rsdp: Option<&'a [u8]>,
        pub load_base: u32,
    }

    struct Writer(Vec<u8>);

    impl Writer {
        fn tag(&mut self, kind: u32, body: &[u8]) {
            self.0.extend_from_slice(&kind.to_le_bytes());
            self.0
                .extend_from_slice(&(8 + body.len() as u32).to_le_bytes());
            self.0.extend_from_slice(body);
            let padded = self.0.len().next_multiple_of(TAG_ALIGN);
            self.0.resize(padded, 0);
        }
    }

    fn nul_terminated(text: &str) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes
    }

    /// The boot information structure (specification 3.6): total size, reserved, then 8-byte
    /// aligned tags ending in an end tag. The boot services tag says the loader kept them, which
    /// is what makes Xen take its EFI path instead of refusing to start.
    pub fn boot_information(info: &Information) -> Vec<u8> {
        let mut out = Writer(alloc::vec![0; 8]);
        out.tag(TAG_CMDLINE, &nul_terminated(info.cmdline));
        out.tag(TAG_BOOT_LOADER_NAME, &nul_terminated(info.loader_name));
        for module in info.modules {
            let mut body = Vec::new();
            body.extend_from_slice(&module.start.to_le_bytes());
            body.extend_from_slice(&module.end.to_le_bytes());
            body.extend_from_slice(&nul_terminated(module.cmdline));
            out.tag(TAG_MODULE, &body);
        }
        let (lower, upper) = info.basic_memory;
        out.tag(
            TAG_BASIC_MEMINFO,
            &[lower.to_le_bytes(), upper.to_le_bytes()].concat(),
        );
        let mut map = Vec::new();
        map.extend_from_slice(&MMAP_ENTRY_BYTES.to_le_bytes());
        map.extend_from_slice(&MMAP_ENTRY_VERSION.to_le_bytes());
        for &(base, length, kind) in info.memory_map {
            map.extend_from_slice(&base.to_le_bytes());
            map.extend_from_slice(&length.to_le_bytes());
            map.extend_from_slice(&kind.to_le_bytes());
            map.extend_from_slice(&0u32.to_le_bytes());
        }
        out.tag(TAG_MMAP, &map);
        out.tag(TAG_EFI64, &info.efi_system_table.to_le_bytes());
        out.tag(TAG_EFI64_IMAGE_HANDLE, &info.efi_image_handle.to_le_bytes());
        out.tag(TAG_EFI_BOOT_SERVICES, &[]);
        if let Some(rsdp) = info.rsdp {
            let new = rsdp.len() > RSDP_V1_BYTES && rsdp[RSDP_REVISION_OFFSET] >= 2;
            out.tag(if new { TAG_ACPI_NEW } else { TAG_ACPI_OLD }, rsdp);
        }
        out.tag(TAG_LOAD_BASE_ADDRESS, &info.load_base.to_le_bytes());
        out.tag(TAG_END, &[]);
        let total = out.0.len() as u32;
        out.0[..4].copy_from_slice(&total.to_le_bytes());
        out.0
    }

    /// The Multiboot2 memory type of an EFI memory type. The loader keeps boot services, so
    /// what they hold is still in use, but this is the map of what the OS may use once it has
    /// called `ExitBootServices` itself, as GRUB reports it.
    pub fn memory_type(efi_type: u32) -> u32 {
        use crate::uefi::memory_type::*;
        match efi_type {
            CONVENTIONAL | LOADER_CODE | LOADER_DATA | BOOT_SERVICES_CODE | BOOT_SERVICES_DATA
            | PERSISTENT => MEMORY_AVAILABLE,
            ACPI_RECLAIM => MEMORY_ACPI_RECLAIMABLE,
            ACPI_NVS => MEMORY_NVS,
            UNUSABLE => MEMORY_BAD,
            _ => MEMORY_RESERVED,
        }
    }

    /// The basic memory information from a sorted memory map: conventional memory from 0 (at
    /// most 640 KiB) and RAM from 1 MiB up to the first gap or non-RAM range, in KiB.
    pub fn basic_memory(memory_map: &[(u64, u64, u32)]) -> (u32, u32) {
        const ONE_MIB: u64 = 0x10_0000;
        const LOW_MEMORY_END: u64 = 0xa_0000;
        const KIB: u64 = 1024;
        let contiguous_from = |start: u64| {
            let mut end = start;
            for &(base, length, kind) in memory_map {
                if kind == MEMORY_AVAILABLE && base <= end && base + length > end {
                    end = base + length;
                }
            }
            end
        };
        let lower = contiguous_from(0).min(LOW_MEMORY_END) / KIB;
        let upper = (contiguous_from(ONE_MIB) - ONE_MIB) / KIB;
        (lower as u32, upper.min(u32::MAX as u64) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::multiboot2::*;
    use super::*;

    /// A Multiboot2 header like Xen's (xen/arch/x86/boot/head.S): information request for
    /// basic memory and the map, module alignment, relocatable, console flags, framebuffer,
    /// boot services, the EFI64 entry, end. `extra` tags go before the end tag.
    fn xen_like_header(extra: &[(u16, u16, Vec<u32>)]) -> Vec<u8> {
        let mut tags: Vec<(u16, u16, Vec<u32>)> = alloc::vec![
            (1, 0, alloc::vec![4, 6]),
            (6, 0, alloc::vec![]),
            (10, 1, alloc::vec![0x20_0000, 0xffff_ffff, 0x20_0000, 2]),
            (4, 1, alloc::vec![2]),
            (5, 1, alloc::vec![0, 0, 0]),
            (7, 1, alloc::vec![]),
            (9, 1, alloc::vec![0x20_0040]),
        ];
        tags.extend_from_slice(extra);
        tags.push((0, 0, alloc::vec![]));
        let mut body = Vec::new();
        for (kind, flags, words) in tags {
            body.extend_from_slice(&kind.to_le_bytes());
            body.extend_from_slice(&flags.to_le_bytes());
            body.extend_from_slice(&(8 + 4 * words.len() as u32).to_le_bytes());
            for word in words {
                body.extend_from_slice(&word.to_le_bytes());
            }
            body.resize(body.len().next_multiple_of(8), 0);
        }
        let length = 16 + body.len() as u32;
        let mut header = Vec::new();
        for word in [
            HEADER_MAGIC,
            0,
            length,
            0u32.wrapping_sub(HEADER_MAGIC).wrapping_sub(length),
        ] {
            header.extend_from_slice(&word.to_le_bytes());
        }
        header.extend_from_slice(&body);
        header
    }

    /// An ELF32 with one PT_LOAD at 0x200000 (filesz 0x40, memsz 0x2000) and a header at 0x98,
    /// shaped like Xen's `xen` (mkelf32 output).
    fn xen_like_elf(extra: &[(u16, u16, Vec<u32>)]) -> Vec<u8> {
        let header = xen_like_header(extra);
        let mut image = alloc::vec![0u8; 0x98];
        image[..4].copy_from_slice(ELF_MAGIC);
        image[4] = 1;
        image[5] = 1;
        image[0x1c..0x20].copy_from_slice(&0x34u32.to_le_bytes());
        image[0x2a..0x2c].copy_from_slice(&32u16.to_le_bytes());
        image[0x2c..0x2e].copy_from_slice(&1u16.to_le_bytes());
        let phdr: [u32; 8] = [1, 0x80, 0x20_0000, 0x20_0000, 0x40, 0x2000, 7, 0x20_0000];
        for (index, word) in phdr.iter().enumerate() {
            image[0x34 + index * 4..0x38 + index * 4].copy_from_slice(&word.to_le_bytes());
        }
        image.extend_from_slice(&header);
        image.resize(image.len().max(0x80 + 0x40), 0);
        image
    }

    #[test]
    fn reads_xens_header_and_segments() {
        let image = xen_like_elf(&[]);
        assert_eq!(format(&image), Some(Format::Multiboot2));
        let header = parse_header(&image).unwrap();
        assert!(header.keep_boot_services);
        assert_eq!(header.efi64_entry, Some(0x20_0040));
        assert_eq!(
            header.relocatable,
            Some(Relocatable {
                min_address: 0x20_0000,
                max_address: 0xffff_ffff,
                alignment: 0x20_0000,
                preference: 2,
            })
        );
        let segments = load_segments(&image).unwrap();
        assert_eq!(
            segments,
            [Segment {
                physical_address: 0x20_0000,
                file_offset: 0x80,
                file_bytes: 0x40,
                memory_bytes: 0x2000,
            }]
        );
        assert_eq!(span(&segments), Some((0x20_0000, 0x2000)));
    }

    #[test]
    fn refuses_what_it_cannot_honour() {
        // A required a.out address tag, and a required request for VBE information (type 7).
        let address = xen_like_elf(&[(2, 0, alloc::vec![0, 0, 0, 0])]);
        assert!(parse_header(&address).is_err());
        let vbe = xen_like_elf(&[(1, 0, alloc::vec![7])]);
        assert!(parse_header(&vbe).is_err());
        // Optional, they are ignored.
        assert!(parse_header(&xen_like_elf(&[(1, 1, alloc::vec![7])])).is_ok());
        assert!(parse_header(b"\x7fELF no header here").is_err());
        let mut truncated = xen_like_elf(&[]);
        truncated.truncate(0x98 + 24);
        assert!(parse_header(&truncated).is_err());
    }

    #[test]
    fn a_segment_outside_the_file_is_refused() {
        let mut image = xen_like_elf(&[]);
        image[0x34 + 16..0x34 + 20].copy_from_slice(&0x10_0000u32.to_le_bytes());
        assert!(load_segments(&image).is_err());
    }

    fn tags(info: &[u8]) -> Vec<(u32, Vec<u8>)> {
        let total = u32::from_le_bytes(info[..4].try_into().unwrap()) as usize;
        assert_eq!(total, info.len());
        let mut at = 8;
        let mut out = Vec::new();
        loop {
            assert_eq!(at % 8, 0);
            let kind = u32::from_le_bytes(info[at..at + 4].try_into().unwrap());
            let size = u32::from_le_bytes(info[at + 4..at + 8].try_into().unwrap()) as usize;
            out.push((kind, info[at + 8..at + size].to_vec()));
            if kind == 0 {
                assert_eq!(at + size, total);
                return out;
            }
            at += size.next_multiple_of(8);
        }
    }

    #[test]
    fn the_boot_information_has_what_xens_efi_entry_requires() {
        let modules = [
            Module {
                start: 0x400_0000,
                end: 0x410_0000,
                cmdline: "vmlinuz console=hvc0",
            },
            Module {
                start: 0x410_0000,
                end: 0x420_0000,
                cmdline: "initramfs",
            },
        ];
        let mut rsdp = [0u8; 36];
        rsdp[15] = 2;
        let info = boot_information(&Information {
            cmdline: "xen.gz console=com1",
            loader_name: "uemulator",
            modules: &modules,
            basic_memory: (639, 1024),
            memory_map: &[(0, 0x9f000, MEMORY_AVAILABLE)],
            efi_system_table: 0x1234_5678,
            efi_image_handle: 0x9abc_def0,
            rsdp: Some(&rsdp),
            load_base: 0x3e00_0000,
        });
        let tags = tags(&info);
        let find = |kind: u32| {
            tags.iter()
                .find(|(k, _)| *k == kind)
                .map(|(_, body)| body.clone())
        };
        // xen/arch/x86/efi/mbi2.c refuses to start without these four.
        assert_eq!(find(1).unwrap(), b"xen.gz console=com1\0");
        assert_eq!(find(12).unwrap(), 0x1234_5678u64.to_le_bytes());
        assert_eq!(find(20).unwrap(), 0x9abc_def0u64.to_le_bytes());
        assert_eq!(find(18).unwrap(), b"");
        let module_tags: Vec<_> = tags.iter().filter(|(k, _)| *k == 3).collect();
        assert_eq!(module_tags.len(), 2);
        assert_eq!(&module_tags[0].1[..8], &[0, 0, 0, 4, 0, 0, 0x10, 4]);
        assert_eq!(&module_tags[0].1[8..], b"vmlinuz console=hvc0\0");
        assert_eq!(find(15).unwrap(), rsdp);
        assert_eq!(find(21).unwrap(), 0x3e00_0000u32.to_le_bytes());
        let map = find(6).unwrap();
        assert_eq!(&map[..8], &[24, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(map.len(), 8 + 24);
    }

    #[test]
    fn basic_memory_counts_up_to_the_first_hole() {
        let map = [
            (0, 0x9f000, MEMORY_AVAILABLE),
            (0x9f000, 0x61000, MEMORY_RESERVED),
            (0x10_0000, 0x10_0000, MEMORY_AVAILABLE),
            (0x20_0000, 0x30_0000, MEMORY_AVAILABLE),
            (0x50_0000, 0x1000, MEMORY_ACPI_RECLAIMABLE),
            (0x50_1000, 0x10_0000, MEMORY_AVAILABLE),
        ];
        assert_eq!(basic_memory(&map), (0x9f000 / 1024, 0x40_0000 / 1024));
    }

    #[test]
    fn the_media_config_is_one_section_with_a_kernel_line() {
        assert_eq!(
            xen_media_config("console=com1", "console=hvc0\nquiet"),
            "[global]\ndefault=xen\n[xen]\noptions=console=com1\nkernel=vmlinuz console=hvc0 quiet\n"
        );
        assert_eq!(format(b"MZ\x90\0"), Some(Format::Efi));
        assert_eq!(format(b"nothing"), None);
    }
}
