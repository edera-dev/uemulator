//! Just enough of an AML encoder (ACPI 6.x chapter 20) to write the DSDT by hand: scopes,
//! devices, named objects, integers, strings, packages and resource-template buffers. Every
//! function returns the encoded bytes of one term so a definition reads like the ASL it stands
//! for.
#[cfg(freestanding)]
use crate::prelude::*;

const OP_ZERO: u8 = 0x00;
const OP_ONE: u8 = 0x01;
const OP_NAME: u8 = 0x08;
const OP_BYTE_PREFIX: u8 = 0x0a;
const OP_WORD_PREFIX: u8 = 0x0b;
const OP_DWORD_PREFIX: u8 = 0x0c;
const OP_QWORD_PREFIX: u8 = 0x0e;
const OP_SCOPE: u8 = 0x10;
const OP_BUFFER: u8 = 0x11;
const OP_PACKAGE: u8 = 0x12;
const OP_METHOD: u8 = 0x14;
const OP_EXTENDED_PREFIX: u8 = 0x5b;
const OP_LOCAL0: u8 = 0x60;
const OP_ARG0: u8 = 0x68;
const OP_STORE: u8 = 0x70;
const OP_AND: u8 = 0x7b;
const OP_OR: u8 = 0x7d;
const OP_DEVICE: u8 = 0x82;
const OP_CREATE_DWORD_FIELD: u8 = 0x8a;
const OP_LNOT: u8 = 0x92;
const OP_LEQUAL: u8 = 0x93;
const OP_IF: u8 = 0xa0;
const OP_ELSE: u8 = 0xa1;
const OP_RETURN: u8 = 0xa4;
/// Arg0..Arg6 and Local0..Local7 are consecutive opcodes.
const ARG_COUNT: u8 = 7;
const LOCAL_COUNT: u8 = 8;
/// MethodFlags: bits 2:0 the argument count, bit 3 serialized (left clear).
const METHOD_ARG_COUNT_MASK: u8 = 0b111;
const ROOT_PREFIX: u8 = b'\\';
const PARENT_PREFIX: u8 = b'^';
const DUAL_NAME_PREFIX: u8 = 0x2e;
const NAME_SEGMENT_PAD: u8 = b'_';

/// An AML name segment is exactly four characters, padded with `_`.
fn segment(name: &str) -> [u8; 4] {
    assert!(name.len() <= 4 && name.is_ascii(), "bad AML name {name}");
    let mut out = [NAME_SEGMENT_PAD; 4];
    out[..name.len()].copy_from_slice(name.as_bytes());
    out
}

/// A NameString: optional `\` or `^` prefixes, then one segment or a dual-name of two.
pub(crate) fn name_string(path: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = path;
    if let Some(stripped) = rest.strip_prefix('\\') {
        out.push(ROOT_PREFIX);
        rest = stripped;
    }
    while let Some(stripped) = rest.strip_prefix('^') {
        out.push(PARENT_PREFIX);
        rest = stripped;
    }
    let segments: Vec<&str> = rest.split('.').filter(|part| !part.is_empty()).collect();
    match segments.as_slice() {
        [one] => out.extend_from_slice(&segment(one)),
        [first, second] => {
            out.push(DUAL_NAME_PREFIX);
            out.extend_from_slice(&segment(first));
            out.extend_from_slice(&segment(second));
        }
        _ => panic!("AML name path {path} has an unsupported shape"),
    }
    out
}

/// PkgLength (20.2.4): the length counts its own bytes. One byte up to 63, otherwise the top two
/// bits of the first byte give the number of following bytes and the first byte keeps only the
/// low nibble.
fn package_length(content_len: usize) -> Vec<u8> {
    const ONE_BYTE_MAX: usize = 0x3f;
    let one = content_len + 1;
    if one <= ONE_BYTE_MAX {
        return vec![one as u8];
    }
    for extra in 1..=3usize {
        let total = content_len + 1 + extra;
        if total < 1 << (4 + 8 * extra) {
            let mut out = vec![((extra as u8) << 6) | (total & 0xf) as u8];
            for byte in 0..extra {
                out.push((total >> (4 + 8 * byte)) as u8);
            }
            return out;
        }
    }
    panic!("AML package too large");
}

fn with_length(opcode: &[u8], name: Option<Vec<u8>>, body: &[u8]) -> Vec<u8> {
    let name = name.unwrap_or_default();
    let mut out = opcode.to_vec();
    out.extend(package_length(name.len() + body.len()));
    out.extend(name);
    out.extend_from_slice(body);
    out
}

pub(crate) fn scope(path: &str, body: &[u8]) -> Vec<u8> {
    with_length(&[OP_SCOPE], Some(name_string(path)), body)
}

pub(crate) fn device(name: &str, body: &[u8]) -> Vec<u8> {
    with_length(
        &[OP_EXTENDED_PREFIX, OP_DEVICE],
        Some(name_string(name)),
        body,
    )
}

/// `Name (path, value)` where `value` is an already encoded data object.
pub(crate) fn named(path: &str, value: &[u8]) -> Vec<u8> {
    let mut out = vec![OP_NAME];
    out.extend(name_string(path));
    out.extend_from_slice(value);
    out
}

pub(crate) fn integer(value: u64) -> Vec<u8> {
    match value {
        0 => vec![OP_ZERO],
        1 => vec![OP_ONE],
        2..=0xff => vec![OP_BYTE_PREFIX, value as u8],
        0x100..=0xffff => {
            let mut out = vec![OP_WORD_PREFIX];
            out.extend_from_slice(&(value as u16).to_le_bytes());
            out
        }
        0x1_0000..=0xffff_ffff => {
            let mut out = vec![OP_DWORD_PREFIX];
            out.extend_from_slice(&(value as u32).to_le_bytes());
            out
        }
        _ => {
            let mut out = vec![OP_QWORD_PREFIX];
            out.extend_from_slice(&value.to_le_bytes());
            out
        }
    }
}

/// A package of already encoded elements (at most 255).
pub(crate) fn package(elements: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = elements.iter().flatten().copied().collect();
    let mut content = vec![elements.len() as u8];
    content.extend(body);
    with_length(&[OP_PACKAGE], None, &content)
}

/// `Buffer (size) {bytes}` with the size as an integer term, how a `ResourceTemplate` is stored.
pub(crate) fn buffer(bytes: &[u8]) -> Vec<u8> {
    let mut content = integer(bytes.len() as u64);
    content.extend_from_slice(bytes);
    with_length(&[OP_BUFFER], None, &content)
}

/// `Method (name, args, NotSerialized) {body}`.
pub(crate) fn method(name: &str, args: u8, body: &[u8]) -> Vec<u8> {
    assert!(
        args <= METHOD_ARG_COUNT_MASK,
        "a method takes at most 7 arguments"
    );
    let mut content = name_string(name);
    content.push(args);
    content.extend_from_slice(body);
    with_length(&[OP_METHOD], None, &content)
}

pub(crate) fn arg(index: u8) -> Vec<u8> {
    assert!(index < ARG_COUNT);
    vec![OP_ARG0 + index]
}

pub(crate) fn local(index: u8) -> Vec<u8> {
    assert!(index < LOCAL_COUNT);
    vec![OP_LOCAL0 + index]
}

/// A reference to a named object.
pub(crate) fn name_ref(path: &str) -> Vec<u8> {
    name_string(path)
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    parts.iter().flat_map(|part| part.iter().copied()).collect()
}

/// `If (predicate) {body}`.
pub(crate) fn if_then(predicate: &[u8], body: &[u8]) -> Vec<u8> {
    with_length(&[OP_IF], None, &concat(&[predicate, body]))
}

/// `Else {body}`, directly after an `If`.
pub(crate) fn else_then(body: &[u8]) -> Vec<u8> {
    with_length(&[OP_ELSE], None, body)
}

pub(crate) fn logical_equal(left: &[u8], right: &[u8]) -> Vec<u8> {
    concat(&[&[OP_LEQUAL], left, right])
}

pub(crate) fn logical_not(operand: &[u8]) -> Vec<u8> {
    concat(&[&[OP_LNOT], operand])
}

pub(crate) fn store(source: &[u8], destination: &[u8]) -> Vec<u8> {
    concat(&[&[OP_STORE], source, destination])
}

/// `And (left, right, target)`.
pub(crate) fn and(left: &[u8], right: &[u8], target: &[u8]) -> Vec<u8> {
    concat(&[&[OP_AND], left, right, target])
}

/// `Or (left, right, target)`.
pub(crate) fn or(left: &[u8], right: &[u8], target: &[u8]) -> Vec<u8> {
    concat(&[&[OP_OR], left, right, target])
}

/// `CreateDWordField (source, byte_index, name)`.
pub(crate) fn create_dword_field(source: &[u8], byte_index: u64, name: &str) -> Vec<u8> {
    concat(&[
        &[OP_CREATE_DWORD_FIELD],
        source,
        &integer(byte_index),
        &name_string(name),
    ])
}

pub(crate) fn return_value(value: &[u8]) -> Vec<u8> {
    concat(&[&[OP_RETURN], value])
}

/// `ToUUID ("aabbccdd-eeff-gghh-iijj-kkllmmnnoopp")`: a 16-byte buffer with the first three
/// fields little endian and the last two in string order (ACPI 6.5 19.6.142).
pub(crate) fn uuid(text: &str) -> Vec<u8> {
    let fields: Vec<&str> = text.split('-').collect();
    assert!(
        fields.iter().map(|field| field.len()).eq([8, 4, 4, 4, 12]),
        "malformed UUID {text}"
    );
    let hex = |digits: &str| -> Vec<u8> {
        (0..digits.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&digits[at..at + 2], 16).expect("UUID is hex"))
            .collect()
    };
    let mut bytes = Vec::new();
    for field in &fields[..3] {
        bytes.extend(hex(field).into_iter().rev());
    }
    for field in &fields[3..] {
        bytes.extend(hex(field));
    }
    buffer(&bytes)
}

/// `EisaId ("PNP0A08")`: manufacturer letters packed into 15 bits and byte-swapped, then the
/// four hex digits of the product (20.2.3, data objects).
pub(crate) fn eisa_id(id: &str) -> Vec<u8> {
    let bytes = id.as_bytes();
    assert!(
        bytes.len() == 7,
        "EISA id is three letters and four hex digits"
    );
    const LETTER_BASE: u8 = 0x40;
    const LETTER_BITS: u32 = 5;
    let letters = bytes[..3].iter().fold(0u16, |acc, letter| {
        (acc << LETTER_BITS) | (letter - LETTER_BASE) as u16
    });
    let product = u16::from_str_radix(&id[3..], 16).expect("EISA product is hex");
    let value = (letters.swap_bytes() as u32)
        | (product.to_be_bytes()[0] as u32) << 16
        | (product.to_be_bytes()[1] as u32) << 24;
    let mut out = vec![OP_DWORD_PREFIX];
    out.extend_from_slice(&value.to_le_bytes());
    out
}

/// Resource descriptors for `_CRS` (6.4). Small items first, then the large ones PCI0 needs.
pub(crate) mod resource {
    #[cfg(freestanding)]
    use crate::prelude::*;

    const SMALL_IRQ: u8 = 0x22;
    const SMALL_IO: u8 = 0x47;
    const SMALL_END_TAG: u8 = 0x79;
    const LARGE_MEMORY32_FIXED: u8 = 0x86;
    const LARGE_DWORD_ADDRESS: u8 = 0x87;
    const LARGE_WORD_ADDRESS: u8 = 0x88;
    const LARGE_MEMORY32_FIXED_LENGTH: u16 = 9;
    const LARGE_DWORD_ADDRESS_LENGTH: u16 = 23;
    const LARGE_WORD_ADDRESS_LENGTH: u16 = 13;
    /// Address space descriptor type byte.
    const SPACE_MEMORY: u8 = 0;
    const SPACE_IO: u8 = 1;
    const SPACE_BUS_NUMBER: u8 = 2;
    /// General flags: min and max address fixed (bits 2 and 3), positive decode, producer.
    const FIXED_RANGE: u8 = 0b1100;
    /// IO type-specific flags: the entire range (ISA and non-ISA).
    const IO_ENTIRE_RANGE: u8 = 0b11;
    /// Memory type-specific flags: read/write, and cacheable (memory attribute 01) or not.
    const MEMORY_READ_WRITE: u8 = 0b001;
    const MEMORY_CACHEABLE: u8 = 0b010;
    const DECODE_16: u8 = 1;
    const MEMORY32_FIXED_READ_WRITE: u8 = 1;

    pub fn io(base: u16, length: u8) -> Vec<u8> {
        let [low, high] = base.to_le_bytes();
        vec![SMALL_IO, DECODE_16, low, high, low, high, 1, length]
    }

    pub fn irq(irq: u8) -> Vec<u8> {
        let mask = 1u16 << irq;
        let [low, high] = mask.to_le_bytes();
        vec![SMALL_IRQ, low, high]
    }

    pub fn memory32_fixed(base: u32, length: u32) -> Vec<u8> {
        let mut out = vec![LARGE_MEMORY32_FIXED];
        out.extend_from_slice(&LARGE_MEMORY32_FIXED_LENGTH.to_le_bytes());
        out.push(MEMORY32_FIXED_READ_WRITE);
        out.extend_from_slice(&base.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());
        out
    }

    fn word_range(space: u8, type_flags: u8, first: u16, last: u16) -> Vec<u8> {
        let mut out = vec![LARGE_WORD_ADDRESS];
        out.extend_from_slice(&LARGE_WORD_ADDRESS_LENGTH.to_le_bytes());
        out.extend_from_slice(&[space, FIXED_RANGE, type_flags]);
        for word in [0, first, last, 0, last - first + 1] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// `WordBusNumber`: the buses behind the root bridge.
    pub fn bus_numbers(first: u8, last: u8) -> Vec<u8> {
        word_range(SPACE_BUS_NUMBER, 0, first as u16, last as u16)
    }

    pub fn io_range(first: u16, last: u16) -> Vec<u8> {
        word_range(SPACE_IO, IO_ENTIRE_RANGE, first, last)
    }

    pub fn memory_range(first: u32, last: u32, cacheable: bool) -> Vec<u8> {
        let mut flags = MEMORY_READ_WRITE;
        if cacheable {
            flags |= MEMORY_CACHEABLE;
        }
        let mut out = vec![LARGE_DWORD_ADDRESS];
        out.extend_from_slice(&LARGE_DWORD_ADDRESS_LENGTH.to_le_bytes());
        out.extend_from_slice(&[SPACE_MEMORY, FIXED_RANGE, flags]);
        for dword in [0, first, last, 0, last - first + 1] {
            out.extend_from_slice(&dword.to_le_bytes());
        }
        out
    }

    /// Concatenates descriptors and closes the template with the end tag.
    pub fn template(descriptors: &[Vec<u8>]) -> Vec<u8> {
        let mut out: Vec<u8> = descriptors.iter().flatten().copied().collect();
        out.extend_from_slice(&[SMALL_END_TAG, 0]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eisa_ids_match_the_values_iasl_emits() {
        let dword = |bytes: Vec<u8>| u32::from_le_bytes(bytes[1..5].try_into().unwrap());
        assert_eq!(dword(eisa_id("PNP0A03")), 0x030a_d041);
        assert_eq!(dword(eisa_id("PNP0A08")), 0x080a_d041);
        assert_eq!(dword(eisa_id("PNP0501")), 0x0105_d041);
        assert_eq!(dword(eisa_id("PNP0C02")), 0x020c_d041);
    }

    #[test]
    fn package_length_counts_its_own_bytes() {
        assert_eq!(package_length(0), vec![1]);
        assert_eq!(package_length(62), vec![63]);
        // 63 content bytes need two length bytes: total 65 = 0x41, low nibble 1, next byte 4.
        assert_eq!(package_length(63), vec![0x41, 0x04]);
        let big = package_length(0x1000);
        assert_eq!(big.len(), 3);
        let total = (big[0] & 0xf) as usize | (big[1] as usize) << 4 | (big[2] as usize) << 12;
        assert_eq!(total, 0x1000 + 3);
    }

    #[test]
    fn names_encode_as_single_dual_and_rooted_paths() {
        assert_eq!(name_string("_SB"), b"_SB_");
        assert_eq!(name_string("\\_SB"), b"\\_SB_");
        assert_eq!(name_string("_SB.PCI0"), b".\x5f\x53\x42\x5fPCI0");
        assert_eq!(name_string("^_S5"), b"^_S5_");
    }

    #[test]
    fn integers_use_the_shortest_encoding() {
        assert_eq!(integer(0), [0x00]);
        assert_eq!(integer(1), [0x01]);
        assert_eq!(integer(5), [0x0a, 5]);
        assert_eq!(integer(0x1234), [0x0b, 0x34, 0x12]);
        assert_eq!(integer(0xfed0_0000), [0x0c, 0, 0, 0xd0, 0xfe]);
        assert_eq!(integer(1 << 32)[0], 0x0e);
    }

    #[test]
    fn a_device_wraps_its_body_in_a_length() {
        let body = named("_UID", &integer(0));
        let encoded = device("PCI0", &body);
        assert_eq!(&encoded[..2], [0x5b, 0x82]);
        assert_eq!(encoded[2] as usize, encoded.len() - 2);
        assert_eq!(&encoded[3..7], b"PCI0");
    }

    #[test]
    fn uuid_matches_the_bytes_iasl_emits_for_the_pci_host_bridge_uuid() {
        let encoded = uuid("33DB4D5B-1FF7-401C-9657-7441C03DD766");
        assert_eq!(&encoded[..3], [0x11, 0x13, 0x0a]);
        assert_eq!(encoded[3], 16);
        assert_eq!(
            &encoded[4..],
            [
                0x5b, 0x4d, 0xdb, 0x33, 0xf7, 0x1f, 0x1c, 0x40, 0x96, 0x57, 0x74, 0x41, 0xc0, 0x3d,
                0xd7, 0x66
            ]
        );
    }

    #[test]
    fn method_and_control_flow_encode_like_iasl() {
        // Method (_OSC, 4, NotSerialized) { Return (Arg3) }
        let encoded = method("_OSC", 4, &return_value(&arg(3)));
        assert_eq!(
            encoded,
            [0x14, 0x08, b'_', b'O', b'S', b'C', 0x04, 0xa4, 0x6b]
        );
        // If (LNot (LEqual (Arg1, One))) { Or (CDW1, 0x08, CDW1) }
        let body = or(&name_ref("CDW1"), &integer(8), &name_ref("CDW1"));
        let encoded = if_then(&logical_not(&logical_equal(&arg(1), &integer(1))), &body);
        assert_eq!(
            encoded,
            [
                0xa0, 0x10, 0x92, 0x93, 0x69, 0x01, 0x7d, b'C', b'D', b'W', b'1', 0x0a, 0x08, b'C',
                b'D', b'W', b'1'
            ]
        );
        assert_eq!(
            else_then(&store(&local(0), &name_ref("CDW3")))[..4],
            [0xa1, 0x07, 0x70, 0x60]
        );
        assert_eq!(
            create_dword_field(&arg(3), 4, "CDW2"),
            [0x8a, 0x6b, 0x0a, 0x04, b'C', b'D', b'W', b'2']
        );
        assert_eq!(
            and(&local(0), &integer(0x1f), &local(0)),
            [0x7b, 0x60, 0x0a, 0x1f, 0x60]
        );
    }

    #[test]
    fn resource_descriptors_have_their_architected_sizes() {
        assert_eq!(resource::io(0x3f8, 8).len(), 8);
        assert_eq!(resource::irq(4), [0x22, 0x10, 0x00]);
        assert_eq!(resource::memory32_fixed(0xfed0_0000, 0x400).len(), 12);
        assert_eq!(resource::bus_numbers(0, 0).len(), 16);
        assert_eq!(resource::io_range(0, 0xcf7).len(), 16);
        assert_eq!(resource::memory_range(0xa0000, 0xbffff, false).len(), 26);
        assert_eq!(resource::template(&[]), [0x79, 0x00]);
    }
}
