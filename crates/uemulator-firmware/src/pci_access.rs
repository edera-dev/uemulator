//! The hardware-independent half of the PCI Root Bridge I/O and PCI I/O protocols (`pci_io`):
//! how their widths, addresses and attributes decode, and the ACPI resource descriptors the root
//! bridge reports.

use alloc::vec::Vec;

/// `EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_WIDTH` (shared by PCI I/O): `Uint8..Uint64`, then the FIFO
/// forms (the address stays put) and the fill forms (the buffer does), each in the same four
/// sizes (UEFI 2.9 14.2).
const WIDTHS_PER_FORM: u32 = 4;
const FORM_PLAIN: u32 = 0;
const FORM_FIFO: u32 = 1;
const FORM_FILL: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Width {
    pub bytes: usize,
    pub advance_address: bool,
    pub advance_buffer: bool,
}

impl Width {
    pub fn decode(width: u32) -> Option<Width> {
        let bytes = 1 << (width % WIDTHS_PER_FORM);
        let (advance_address, advance_buffer) = match width / WIDTHS_PER_FORM {
            FORM_PLAIN => (true, true),
            FORM_FIFO => (false, true),
            FORM_FILL => (true, false),
            _ => return None,
        };
        Some(Width {
            bytes,
            advance_address,
            advance_buffer,
        })
    }
}

/// A configuration-space location from the root bridge's `Pci.Read`/`Write` address:
/// register, function, device and bus in bytes 0-3, and a 32-bit extended register (which wins
/// when non-zero) in the high half (UEFI 2.9 14.2,
/// `EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_PCI_ADDRESS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigAddress {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub register: u32,
}

impl ConfigAddress {
    pub fn decode(address: u64) -> ConfigAddress {
        let extended = (address >> 32) as u32;
        ConfigAddress {
            bus: (address >> 24) as u8,
            device: (address >> 16) as u8,
            function: (address >> 8) as u8,
            register: if extended != 0 {
                extended
            } else {
                address as u8 as u32
            },
        }
    }

    pub fn advanced(self, bytes: usize) -> ConfigAddress {
        ConfigAddress {
            register: self.register + bytes as u32,
            ..self
        }
    }
}

/// `EFI_PCI_IO_ATTRIBUTE_*` bits a function here supports, and the command-register bit behind
/// each (PCI Local Bus 3.0 6.2.2). Dual address cycle has no register: it only tells `Map` and
/// `AllocateBuffer` the device takes 64-bit addresses.
pub const ATTRIBUTE_IO: u64 = 0x0100;
pub const ATTRIBUTE_MEMORY: u64 = 0x0200;
pub const ATTRIBUTE_BUS_MASTER: u64 = 0x0400;
pub const ATTRIBUTE_DUAL_ADDRESS_CYCLE: u64 = 0x8000;
pub const SUPPORTED_ATTRIBUTES: u64 =
    ATTRIBUTE_IO | ATTRIBUTE_MEMORY | ATTRIBUTE_BUS_MASTER | ATTRIBUTE_DUAL_ADDRESS_CYCLE;
const COMMAND_IO_SPACE: u16 = 1 << 0;
const COMMAND_MEMORY_SPACE: u16 = 1 << 1;
const COMMAND_BUS_MASTER: u16 = 1 << 2;
const COMMAND_ATTRIBUTES: [(u16, u64); 3] = [
    (COMMAND_IO_SPACE, ATTRIBUTE_IO),
    (COMMAND_MEMORY_SPACE, ATTRIBUTE_MEMORY),
    (COMMAND_BUS_MASTER, ATTRIBUTE_BUS_MASTER),
];
/// The command-register bits the attributes own; the others (interrupt disable, SERR#, ...)
/// are left as they are.
pub const COMMAND_OWNED: u16 = COMMAND_IO_SPACE | COMMAND_MEMORY_SPACE | COMMAND_BUS_MASTER;

pub fn attributes_from_command(command: u16) -> u64 {
    COMMAND_ATTRIBUTES
        .iter()
        .filter(|&&(bit, _)| command & bit != 0)
        .fold(0, |attributes, &(_, attribute)| attributes | attribute)
}

pub fn command_from_attributes(attributes: u64) -> u16 {
    COMMAND_ATTRIBUTES
        .iter()
        .filter(|&&(_, attribute)| attributes & attribute != 0)
        .fold(0, |command, &(bit, _)| command | bit)
}

/// `EFI_PCI_IO_PROTOCOL_ATTRIBUTE_OPERATION`.
pub const ATTRIBUTE_GET: u32 = 0;
pub const ATTRIBUTE_SET: u32 = 1;
pub const ATTRIBUTE_ENABLE: u32 = 2;
pub const ATTRIBUTE_DISABLE: u32 = 3;
pub const ATTRIBUTE_SUPPORTED: u32 = 4;

/// What `Attributes` leaves set, or `None` for an operation that changes nothing or is unknown.
pub fn apply_attributes(operation: u32, current: u64, given: u64) -> Option<u64> {
    match operation {
        ATTRIBUTE_SET => Some(given),
        ATTRIBUTE_ENABLE => Some(current | given),
        ATTRIBUTE_DISABLE => Some(current & !given),
        _ => None,
    }
}

/// ACPI 6.4 6.4.3.5.1, QWORD Address Space Descriptor: tag, a 16-bit length of what follows, then
/// resource type, flags, and granularity, minimum, maximum, translation offset and length.
const QWORD_ADDRESS_SPACE_TAG: u8 = 0x8a;
const QWORD_ADDRESS_SPACE_LENGTH: u16 = 0x2b;
/// ACPI 6.4 6.4.2.9, End Tag: the tag and a checksum, where zero means "not checked".
const END_TAG: [u8; 2] = [0x79, 0];
pub const RESOURCE_MEMORY: u8 = 0;
pub const RESOURCE_BUS: u8 = 2;

fn qword_address_space(resource_type: u8, minimum: u64, length: u64) -> Vec<u8> {
    let mut descriptor = Vec::with_capacity(3 + QWORD_ADDRESS_SPACE_LENGTH as usize);
    descriptor.push(QWORD_ADDRESS_SPACE_TAG);
    descriptor.extend_from_slice(&QWORD_ADDRESS_SPACE_LENGTH.to_le_bytes());
    // Resource type, general flags, type-specific flags: plain, fixed ranges.
    descriptor.extend_from_slice(&[resource_type, 0, 0]);
    let maximum = minimum + length - 1;
    for value in [0, minimum, maximum, 0, length] {
        descriptor.extend_from_slice(&value.to_le_bytes());
    }
    descriptor
}

/// What the root bridge's `Configuration` reports: bus 0 alone, and the BAR window with no
/// translation (a bus address is the CPU's).
pub fn root_bridge_resources(memory_window: Option<(u64, u64)>) -> Vec<u8> {
    let mut resources = qword_address_space(RESOURCE_BUS, 0, 1);
    if let Some((base, length)) = memory_window {
        resources.extend(qword_address_space(RESOURCE_MEMORY, base, length));
    }
    resources.extend_from_slice(&END_TAG);
    resources
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_decode_by_form_and_size() {
        assert_eq!(
            Width::decode(2),
            Some(Width {
                bytes: 4,
                advance_address: true,
                advance_buffer: true
            })
        );
        let fifo = Width::decode(5).unwrap();
        assert_eq!((fifo.bytes, fifo.advance_address), (2, false));
        let fill = Width::decode(11).unwrap();
        assert_eq!((fill.bytes, fill.advance_buffer), (8, false));
        assert_eq!(Width::decode(12), None);
    }

    #[test]
    fn a_config_address_takes_the_extended_register_when_there_is_one() {
        let plain = ConfigAddress::decode(0x0003_0204);
        assert_eq!(
            plain,
            ConfigAddress {
                bus: 0,
                device: 3,
                function: 2,
                register: 4
            }
        );
        assert_eq!(ConfigAddress::decode(0x0000_0100_0003_0204).register, 0x100);
        assert_eq!(plain.advanced(2).register, 6);
    }

    #[test]
    fn attributes_and_command_bits_round_trip() {
        assert_eq!(
            attributes_from_command(0b111),
            ATTRIBUTE_IO | ATTRIBUTE_MEMORY | ATTRIBUTE_BUS_MASTER
        );
        assert_eq!(command_from_attributes(ATTRIBUTE_BUS_MASTER), 0b100);
        assert_eq!(command_from_attributes(ATTRIBUTE_DUAL_ADDRESS_CYCLE), 0);
        assert_eq!(attributes_from_command(1 << 10), 0, "interrupt disable");
    }

    #[test]
    fn attribute_operations() {
        let both = ATTRIBUTE_IO | ATTRIBUTE_MEMORY;
        assert_eq!(
            apply_attributes(ATTRIBUTE_ENABLE, ATTRIBUTE_IO, ATTRIBUTE_MEMORY),
            Some(both)
        );
        assert_eq!(
            apply_attributes(ATTRIBUTE_DISABLE, both, ATTRIBUTE_IO),
            Some(ATTRIBUTE_MEMORY)
        );
        assert_eq!(apply_attributes(ATTRIBUTE_SET, both, 0), Some(0));
        assert_eq!(apply_attributes(ATTRIBUTE_GET, both, 0), None);
    }

    #[test]
    fn resources_are_a_bus_range_a_memory_window_and_an_end_tag() {
        let resources = root_bridge_resources(Some((0xc000_0000, 0x2000_0000)));
        let descriptor_bytes = 3 + QWORD_ADDRESS_SPACE_LENGTH as usize;
        assert_eq!(resources.len(), 2 * descriptor_bytes + END_TAG.len());
        let field = |descriptor: usize, offset: usize| {
            let at = descriptor * descriptor_bytes + offset;
            u64::from_le_bytes(resources[at..at + 8].try_into().unwrap())
        };
        // Bus: 0..=0, one bus.
        assert_eq!(resources[3], RESOURCE_BUS);
        assert_eq!((field(0, 14), field(0, 22), field(0, 38)), (0, 0, 1));
        // Memory: minimum, maximum, no translation, length.
        assert_eq!(resources[descriptor_bytes + 3], RESOURCE_MEMORY);
        assert_eq!(field(1, 14), 0xc000_0000);
        assert_eq!(field(1, 22), 0xdfff_ffff);
        assert_eq!(field(1, 30), 0);
        assert_eq!(field(1, 38), 0x2000_0000);
        assert_eq!(resources[resources.len() - 2..], END_TAG);
    }
}
