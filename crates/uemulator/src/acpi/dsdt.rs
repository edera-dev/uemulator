//! The DSDT, assembled term by term from `aml`. What it holds is the smallest namespace Linux
//! (and Windows) need on this machine: the PCI root bridge with its interrupt routing and `_OSC`, the legacy
//! devices a PC has, a system-resources device that claims the ports and the ECAM window the
//! other tables mention, and `\_S5` for power-off.
#[cfg(freestanding)]
use crate::prelude::*;

use super::aml::{self, resource};
use crate::dev::acpi_pm::{PM_LAST_PORT, PM1_EVENT_BASE, SLEEP_TYPE_S5};
use crate::dev::pci::{ECAM_BASE, ECAM_BYTES_PER_BUS, MMIO_BASE, PCI_MMIO_WINDOW_BYTES};

const DSDT_REVISION: u8 = 2;
/// `_ADR` of a PCI device is `device << 16 | function`; 0xFFFF as the function means all of them.
const ADDRESS_ALL_FUNCTIONS: u32 = 0xffff;
const PCI_SLOT_SHIFT: u32 = 16;
/// INTA# is pin 0 in a `_PRT` entry.
const PIN_INTA: u64 = 0;
/// Legacy I/O below 0xCF8 and above 0xCFF belongs to the root bridge's window.
const CONFIG_PORT_FIRST: u16 = 0xcf8;
const CONFIG_PORTS_LENGTH: u8 = 8;
const IO_WINDOW_LOW_LAST: u16 = 0xcf7;
const IO_WINDOW_HIGH_FIRST: u16 = 0xd00;
const IO_WINDOW_HIGH_LAST: u16 = 0xffff;
/// The legacy VGA aperture; there is no VGA but the range is the bridge's, as on a PC.
const VGA_WINDOW_FIRST: u32 = 0xa_0000;
const VGA_WINDOW_LAST: u32 = 0xb_ffff;
const PCI_ROOT_BUS: u8 = 0;

const COM1_PORT: u16 = 0x3f8;
const COM1_PORTS: u8 = 8;
const COM1_IRQ: u8 = 4;
const RTC_PORT: u16 = 0x70;
const RTC_PORTS: u8 = 2;
const PIC_PORTS: [(u16, u8); 2] = [(0x20, 2), (0xa0, 2)];
const PIT_PORT: u16 = 0x40;
const PIT_PORTS: u8 = 4;
/// `_OSC` for a PCI host bridge (PCI Firmware Specification 3.3, 4.5): the UUID that selects it,
/// the error bits it sets in capabilities dword 1, and the control bits of dword 3.
const PCI_HOST_BRIDGE_OSC_UUID: &str = "33DB4D5B-1FF7-401C-9657-7441C03DD766";
const OSC_REVISION: u64 = 1;
const OSC_ERROR_UNRECOGNIZED_UUID: u64 = 1 << 2;
const OSC_ERROR_UNRECOGNIZED_REVISION: u64 = 1 << 3;
const OSC_ERROR_CAPABILITIES_MASKED: u64 = 1 << 4;
/// Native PCIe hot-plug, SHPC hot-plug, PME, AER and the PCIe capability structure: what QEMU's
/// q35 grants. Not LTR (bit 5), which Linux asks for and then reports as unsupported.
const OSC_CONTROL_GRANTED: u64 = 0x1f;
/// Byte offsets of capabilities dwords 1-3 in `_OSC`'s Arg3 buffer.
const OSC_DWORD1_OFFSET: u64 = 0;
const OSC_DWORD2_OFFSET: u64 = 4;
const OSC_DWORD3_OFFSET: u64 = 8;
const OSC_UUID_ARGUMENT: u8 = 0;
const OSC_REVISION_ARGUMENT: u8 = 1;
const OSC_CAPABILITIES_ARGUMENT: u8 = 3;
const OSC_ARGUMENTS: u8 = 4;

/// What `\_S5` returns besides the sleep type: SLP_TYPb is the same, the rest is padding.
const S5_PACKAGE_PADDING: u64 = 0;

fn eisa_name(object: &str, id: &str) -> Vec<u8> {
    aml::named(object, &aml::eisa_id(id))
}

/// A legacy device: `_HID`, `_UID` and a fixed `_CRS`.
fn legacy_device(name: &str, hid: &str, uid: u64, descriptors: &[Vec<u8>]) -> Vec<u8> {
    let mut body = eisa_name("_HID", hid);
    body.extend(aml::named("_UID", &aml::integer(uid)));
    body.extend(aml::named(
        "_CRS",
        &aml::buffer(&resource::template(descriptors)),
    ));
    aml::device(name, &body)
}

/// `_PRT`: every virtio function's INTA# goes straight to the GSI of its legacy line
/// (`pci::LEGACY_IRQS`), source 0 meaning no interrupt link device.
fn pci_routing_table(irq_of_slot: &[u8]) -> Vec<u8> {
    let entries: Vec<Vec<u8>> = irq_of_slot
        .iter()
        .enumerate()
        .map(|(index, irq)| {
            let slot = index as u32 + 1;
            aml::package(&[
                aml::integer((slot << PCI_SLOT_SHIFT | ADDRESS_ALL_FUNCTIONS) as u64),
                aml::integer(PIN_INTA),
                aml::integer(0),
                aml::integer(*irq as u64),
            ])
        })
        .collect();
    aml::named("_PRT", &aml::package(&entries))
}

/// `_OSC` on the root bridge. Without it Linux keeps its hands off PME, AER and the PCIe
/// capability ("platform retains control of PCIe features (AE_NOT_FOUND)"); with it the OS gets
/// the controls in `OSC_CONTROL_GRANTED` that it asks for, and dword 1 reports anything masked.
///
/// ```text
/// Method (_OSC, 4) {
///     CreateDWordField (Arg3, 0, CDW1)
///     If (LEqual (Arg0, ToUUID (PCI_HOST_BRIDGE_OSC_UUID))) {
///         CreateDWordField (Arg3, 4, CDW2)
///         CreateDWordField (Arg3, 8, CDW3)
///         Store (CDW3, Local0)
///         And (Local0, OSC_CONTROL_GRANTED, Local0)
///         If (LNot (LEqual (Arg1, OSC_REVISION))) { Or (CDW1, unrecognized revision, CDW1) }
///         If (LNot (LEqual (CDW3, Local0))) { Or (CDW1, capabilities masked, CDW1) }
///         Store (Local0, CDW3)
///     } Else {
///         Or (CDW1, unrecognized UUID, CDW1)
///     }
///     Return (Arg3)
/// }
/// ```
fn pci_host_bridge_osc() -> Vec<u8> {
    let capabilities = aml::arg(OSC_CAPABILITIES_ARGUMENT);
    let dword1 = aml::name_ref("CDW1");
    let dword3 = aml::name_ref("CDW3");
    let control = aml::local(0);
    let set_error = |bit: u64| aml::or(&dword1, &aml::integer(bit), &dword1);
    let mut matched = aml::create_dword_field(&capabilities, OSC_DWORD2_OFFSET, "CDW2");
    matched.extend(aml::create_dword_field(
        &capabilities,
        OSC_DWORD3_OFFSET,
        "CDW3",
    ));
    matched.extend(aml::store(&dword3, &control));
    matched.extend(aml::and(
        &control,
        &aml::integer(OSC_CONTROL_GRANTED),
        &control,
    ));
    matched.extend(aml::if_then(
        &aml::logical_not(&aml::logical_equal(
            &aml::arg(OSC_REVISION_ARGUMENT),
            &aml::integer(OSC_REVISION),
        )),
        &set_error(OSC_ERROR_UNRECOGNIZED_REVISION),
    ));
    matched.extend(aml::if_then(
        &aml::logical_not(&aml::logical_equal(&dword3, &control)),
        &set_error(OSC_ERROR_CAPABILITIES_MASKED),
    ));
    matched.extend(aml::store(&control, &dword3));
    let mut body = aml::create_dword_field(&capabilities, OSC_DWORD1_OFFSET, "CDW1");
    body.extend(aml::if_then(
        &aml::logical_equal(
            &aml::arg(OSC_UUID_ARGUMENT),
            &aml::uuid(PCI_HOST_BRIDGE_OSC_UUID),
        ),
        &matched,
    ));
    body.extend(aml::else_then(&set_error(OSC_ERROR_UNRECOGNIZED_UUID)));
    body.extend(aml::return_value(&capabilities));
    aml::method("_OSC", OSC_ARGUMENTS, &body)
}

fn pci_root_bridge(irq_of_slot: &[u8]) -> Vec<u8> {
    let mut body = eisa_name("_HID", "PNP0A08");
    body.extend(eisa_name("_CID", "PNP0A03"));
    body.extend(aml::named("_SEG", &aml::integer(0)));
    body.extend(aml::named("_BBN", &aml::integer(PCI_ROOT_BUS as u64)));
    body.extend(aml::named("_UID", &aml::integer(0)));
    let window = resource::template(&[
        resource::bus_numbers(PCI_ROOT_BUS, PCI_ROOT_BUS),
        resource::io(CONFIG_PORT_FIRST, CONFIG_PORTS_LENGTH),
        resource::io_range(0, IO_WINDOW_LOW_LAST),
        resource::io_range(IO_WINDOW_HIGH_FIRST, IO_WINDOW_HIGH_LAST),
        resource::memory_range(VGA_WINDOW_FIRST, VGA_WINDOW_LAST, false),
        resource::memory_range(
            MMIO_BASE as u32,
            (MMIO_BASE + PCI_MMIO_WINDOW_BYTES - 1) as u32,
            false,
        ),
    ]);
    body.extend(aml::named("_CRS", &aml::buffer(&window)));
    body.extend(pci_routing_table(irq_of_slot));
    body.extend(pci_host_bridge_osc());
    aml::device("PCI0", &body)
}

/// PNP0C02: reserved ranges nothing else describes: the PM blocks and the ECAM window, which
/// Linux insists on finding in a motherboard-resources device before it trusts the MCFG table.
fn system_resources() -> Vec<u8> {
    let mut body = eisa_name("_HID", "PNP0C02");
    body.extend(aml::named("_UID", &aml::integer(1)));
    let pm_ports = (PM_LAST_PORT - PM1_EVENT_BASE + 1) as u8;
    let descriptors = [
        resource::io(PM1_EVENT_BASE, pm_ports),
        resource::memory32_fixed(ECAM_BASE as u32, ECAM_BYTES_PER_BUS as u32),
    ];
    body.extend(aml::named(
        "_CRS",
        &aml::buffer(&resource::template(&descriptors)),
    ));
    aml::device("SYSR", &body)
}

fn sleep_states() -> Vec<u8> {
    let sleep_type = aml::integer(SLEEP_TYPE_S5 as u64);
    aml::named(
        "_S5",
        &aml::package(&[
            sleep_type.clone(),
            sleep_type,
            aml::integer(S5_PACKAGE_PADDING),
            aml::integer(S5_PACKAGE_PADDING),
        ]),
    )
}

/// `irq_of_slot[i]` is the GSI of the function in slot `i + 1`.
pub(super) fn dsdt(irq_of_slot: &[u8]) -> Vec<u8> {
    let mut system_bus = pci_root_bridge(irq_of_slot);
    system_bus.extend(legacy_device(
        "COM1",
        "PNP0501",
        1,
        &[resource::io(COM1_PORT, COM1_PORTS), resource::irq(COM1_IRQ)],
    ));
    system_bus.extend(legacy_device(
        "RTC",
        "PNP0B00",
        0,
        &[resource::io(RTC_PORT, RTC_PORTS)],
    ));
    system_bus.extend(legacy_device(
        "PIC",
        "PNP0000",
        0,
        &PIC_PORTS.map(|(port, length)| resource::io(port, length)),
    ));
    system_bus.extend(legacy_device(
        "PIT",
        "PNP0100",
        0,
        &[resource::io(PIT_PORT, PIT_PORTS)],
    ));
    system_bus.extend(system_resources());
    let mut body = aml::scope("\\_SB", &system_bus);
    body.extend(sleep_states());
    super::tables::table(b"DSDT", DSDT_REVISION, &body)
}
