//! ACPI tables, built on the host side and written into guest RAM before the guest runs, because
//! there is no firmware to do it. Linux finds them through `boot_params.acpi_rsdp_addr`, and a
//! BIOS-style scan would find the RSDP at the start of the region (16-byte aligned inside
//! 0xE0000..0xFFFFF).
//!
//! The set is RSDP (rev 2), RSDT, XSDT, FACS, FADT (rev 6, full hardware, no SMI_CMD so ACPI
//! starts enabled), DSDT, MADT, MCFG, HPET and, with the IOMMU, IVRS. `docs/plans/18-xen-dom0.md` and
//! `docs/plans/07-devices.md` describe the devices the tables point at.
#[cfg(freestanding)]
use crate::prelude::*;

mod aml;
mod dsdt;
mod tables;
#[cfg(test)]
mod tests;

use crate::dev::iommu::{self, IvrsTopology, SpecialDevice};
use crate::dev::pci::{ECAM_BASE, ECAM_BYTES_PER_BUS, LEGACY_IRQS};
use crate::dev::{HPET_REQUESTER, IOAPIC_REQUESTER, ioapic::IOAPIC_ID};
use crate::machine::Error;

/// Where the tables live: inside the BIOS area (0xE0000..0xFFFFF), which the e820 map already
/// reports as reserved. Tables, RSDP first.
pub(crate) const REGION_BASE: u64 = 0xe_0000;
pub(crate) const REGION_BYTES: u64 = 0x1_0000;
/// The ECAM window of bus 0, which `boot` reports as reserved in e820 as well.
pub(crate) const ECAM_RANGE: (u64, u64) = (ECAM_BASE, ECAM_BYTES_PER_BUS);
const TABLE_ALIGNMENT: usize = 16;
/// ACPI 6 5.2.10: the FACS must sit on a 64-byte boundary.
const FACS_ALIGNMENT: usize = 64;
const PCI_HOST_BRIDGE_DEVFN: u16 = 0;
const PCI_SLOT_SHIFT: u32 = 3;
const FIRST_VIRTIO_SLOT: u16 = 1;

/// What the tables describe about this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AcpiConfig {
    /// Processors in the MADT. The emulator runs one for now; the rest are listed so the table
    /// logic (x2APIC entries from id 255) is exercised before SMP exists.
    pub cpus: u32,
    /// Virtio functions on the PCI bus, in slots 1 and up.
    pub pci_functions: usize,
    /// Publish the IVRS table for the AMD IOMMU.
    pub iommu: bool,
}

impl Default for AcpiConfig {
    fn default() -> Self {
        AcpiConfig {
            cpus: 1,
            pci_functions: 0,
            iommu: false,
        }
    }
}

/// Appends aligned blobs to the region image and reports their guest addresses. `REGION_BASE`
/// is page aligned, so an offset alignment is an address alignment.
struct Region {
    image: Vec<u8>,
}

impl Region {
    fn place(&mut self, bytes: &[u8]) -> u64 {
        self.place_aligned(bytes, TABLE_ALIGNMENT)
    }

    fn place_aligned(&mut self, bytes: &[u8], alignment: usize) -> u64 {
        let padding = self.image.len().next_multiple_of(alignment) - self.image.len();
        self.image.resize(self.image.len() + padding, 0);
        let address = REGION_BASE + self.image.len() as u64;
        self.image.extend_from_slice(bytes);
        address
    }
}

/// The GSI of each virtio function's INTA#, slot order.
fn pci_interrupt_gsis(functions: usize) -> Vec<u8> {
    (0..functions)
        .map(|index| LEGACY_IRQS[index % LEGACY_IRQS.len()])
        .collect()
}

fn ivrs(config: &AcpiConfig) -> Vec<u8> {
    let mut pci_devices = vec![PCI_HOST_BRIDGE_DEVFN, iommu::REQUESTER_ID];
    pci_devices.extend(
        (0..config.pci_functions).map(|index| (FIRST_VIRTIO_SLOT + index as u16) << PCI_SLOT_SHIFT),
    );
    // Linux sizes the IOMMU's device table from the highest device id in the select and range
    // entries only, not the special ones: without these two the table would end below the
    // I/O APIC's requester id and every I/O APIC interrupt would fault.
    pci_devices.extend([IOAPIC_REQUESTER, HPET_REQUESTER]);
    iommu::Iommu::new().ivrs_table(&IvrsTopology {
        pci_devices,
        ioapics: vec![SpecialDevice {
            handle: IOAPIC_ID,
            requester: IOAPIC_REQUESTER,
        }],
        hpets: vec![SpecialDevice {
            handle: 0,
            requester: HPET_REQUESTER,
        }],
    })
}

/// The whole region as it must appear at `REGION_BASE`, RSDP first.
pub(crate) fn build(config: &AcpiConfig) -> Result<Vec<u8>, Error> {
    let mut region = Region {
        image: vec![0; tables::RSDP_LENGTH],
    };
    let facs = region.place_aligned(&tables::facs(), FACS_ALIGNMENT);
    let dsdt = region.place(&dsdt::dsdt(&pci_interrupt_gsis(config.pci_functions)));
    let fadt = region.place(&tables::fadt(dsdt, facs));
    // Level-triggered ISA lines: the PCI functions' and the SCI.
    let mut level_irqs = LEGACY_IRQS.to_vec();
    level_irqs.push(crate::dev::acpi_pm::SCI_IRQ);
    let mut entries = vec![fadt];
    entries.push(region.place(&tables::madt(config.cpus, &level_irqs)));
    entries.push(region.place(&tables::mcfg(ECAM_BASE, 0, 0)));
    entries.push(region.place(&tables::hpet()));
    if config.iommu {
        entries.push(region.place(&ivrs(config)));
    }
    let rsdt = region.place(&tables::rsdt(&entries));
    let xsdt = region.place(&tables::xsdt(&entries));
    let rsdp = tables::rsdp(rsdt as u32, xsdt);
    region.image[..rsdp.len()].copy_from_slice(&rsdp);
    if region.image.len() as u64 > REGION_BYTES {
        return Err(Error::BadConfig(
            "ACPI tables do not fit their region".into(),
        ));
    }
    Ok(region.image)
}

/// Writes the tables into `ram` and returns the RSDP's physical address.
pub(crate) fn install(ram: &mut [u8], config: &AcpiConfig) -> Result<u64, Error> {
    let image = build(config)?;
    let start = REGION_BASE as usize;
    ram.get_mut(start..start + image.len())
        .ok_or_else(|| Error::BadConfig("guest memory too small for the ACPI tables".into()))?
        .copy_from_slice(&image);
    Ok(REGION_BASE)
}
