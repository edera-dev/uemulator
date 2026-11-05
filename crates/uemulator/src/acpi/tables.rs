//! The fixed-format tables: RSDP, RSDT, XSDT, FACS, FADT, MADT, MCFG, HPET. Layouts follow
//! ACPI 6.x chapter 5 (and the HPET specification 1.0a); QEMU's `hw/acpi/aml-build.c` is the
//! cross-check for which optional fields guests actually read.
#[cfg(freestanding)]
use crate::prelude::*;

use crate::dev::acpi_pm::{
    PM_TIMER_BASE, PM_TIMER_LENGTH, PM1_CONTROL_BASE, PM1_CONTROL_LENGTH, PM1_EVENT_BASE,
    PM1_EVENT_LENGTH, SCI_IRQ,
};
use crate::dev::apic::APIC_DEFAULT_BASE;
use crate::dev::hpet::{CAPABILITIES_LOW_DWORD, HPET_BASE};
use crate::dev::ioapic::{IOAPIC_DEFAULT_BASE, IOAPIC_ID};
use crate::dev::tpm::{CTRL_REQ, TPM_CRB_BASE};

pub(super) const HEADER_LENGTH: usize = 36;
const OEM_ID: &[u8; 6] = b"UEMUL ";
const OEM_TABLE_ID: &[u8; 8] = b"UEMUTBL ";
const OEM_REVISION: u32 = 1;
/// ASL compiler id and revision fields of the header; nothing reads them.
const CREATOR_ID: &[u8; 4] = b"UEMU";
const CREATOR_REVISION: u32 = 1;
const CHECKSUM_OFFSET: usize = 9;

/// Makes the byte sum of `bytes[range]` zero by setting the byte at `checksum_at`.
pub(super) fn fix_checksum(bytes: &mut [u8], checksum_at: usize, covered: usize) {
    bytes[checksum_at] = 0;
    let sum = bytes[..covered]
        .iter()
        .fold(0u8, |sum, byte| sum.wrapping_add(*byte));
    bytes[checksum_at] = sum.wrapping_neg();
}

/// A complete table: the standard header, `body`, and the checksum.
pub(super) fn table(signature: &[u8; 4], revision: u8, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LENGTH + body.len());
    out.extend_from_slice(signature);
    out.extend_from_slice(&((HEADER_LENGTH + body.len()) as u32).to_le_bytes());
    out.push(revision);
    out.push(0);
    out.extend_from_slice(OEM_ID);
    out.extend_from_slice(OEM_TABLE_ID);
    out.extend_from_slice(&OEM_REVISION.to_le_bytes());
    out.extend_from_slice(CREATOR_ID);
    out.extend_from_slice(&CREATOR_REVISION.to_le_bytes());
    out.extend_from_slice(body);
    let covered = out.len();
    fix_checksum(&mut out, CHECKSUM_OFFSET, covered);
    out
}

const RSDP_SIGNATURE: &[u8; 8] = b"RSD PTR ";
const RSDP_REVISION: u8 = 2;
pub(super) const RSDP_LENGTH: usize = 36;
/// The first checksum covers the ACPI 1.0 part (20 bytes), the second the whole structure.
const RSDP_V1_LENGTH: usize = 20;
const RSDP_V1_CHECKSUM_AT: usize = 8;
const RSDP_EXTENDED_CHECKSUM_AT: usize = 32;

pub(super) fn rsdp(rsdt_address: u32, xsdt_address: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(RSDP_LENGTH);
    out.extend_from_slice(RSDP_SIGNATURE);
    out.push(0);
    out.extend_from_slice(OEM_ID);
    out.push(RSDP_REVISION);
    out.extend_from_slice(&rsdt_address.to_le_bytes());
    out.extend_from_slice(&(RSDP_LENGTH as u32).to_le_bytes());
    out.extend_from_slice(&xsdt_address.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    fix_checksum(&mut out, RSDP_V1_CHECKSUM_AT, RSDP_V1_LENGTH);
    fix_checksum(&mut out, RSDP_EXTENDED_CHECKSUM_AT, RSDP_LENGTH);
    out
}

pub(super) fn rsdt(entries: &[u64]) -> Vec<u8> {
    let body: Vec<u8> = entries
        .iter()
        .flat_map(|address| (*address as u32).to_le_bytes())
        .collect();
    table(b"RSDT", 1, &body)
}

pub(super) fn xsdt(entries: &[u64]) -> Vec<u8> {
    let body: Vec<u8> = entries
        .iter()
        .flat_map(|address| address.to_le_bytes())
        .collect();
    table(b"XSDT", 1, &body)
}

const FACS_LENGTH: u32 = 64;
const FACS_VERSION: u8 = 2;
const FACS_LENGTH_OFFSET: usize = 4;
const FACS_VERSION_OFFSET: usize = 32;

/// The firmware control structure. Not a table (no checksum); everything is zero but the
/// signature, length and version, because nothing here sleeps or uses the global lock.
pub(super) fn facs() -> Vec<u8> {
    let mut out = vec![0u8; FACS_LENGTH as usize];
    out[..4].copy_from_slice(b"FACS");
    out[FACS_LENGTH_OFFSET..FACS_LENGTH_OFFSET + 4].copy_from_slice(&FACS_LENGTH.to_le_bytes());
    out[FACS_VERSION_OFFSET] = FACS_VERSION;
    out
}

/// Generic Address Structure address space ids (5.2.3.2).
const GAS_SYSTEM_MEMORY: u8 = 0;
const GAS_SYSTEM_IO: u8 = 1;

fn generic_address(space: u8, bit_width: u8, address: u64) -> Vec<u8> {
    let mut out = vec![space, bit_width, 0, 0];
    out.extend_from_slice(&address.to_le_bytes());
    out
}

const FADT_REVISION: u8 = 6;
const FADT_MINOR_VERSION: u8 = 0;
const FADT_BODY_LENGTH: usize = 276 - HEADER_LENGTH;
/// P_LVL2_LAT and P_LVL3_LAT above 100 and 1000 microseconds mean "C2/C3 not supported".
const LATENCY_C2_UNSUPPORTED: u16 = 101;
const LATENCY_C3_UNSUPPORTED: u16 = 1001;
/// The CMOS century byte the RTC model implements (`CMOS_CENTURY`).
const RTC_CENTURY_REGISTER: u8 = 0x32;
const BITS_PER_BYTE: u8 = 8;
/// IAPC_BOOT_ARCH: legacy devices (PIC, PIT, RTC...) present; no VGA; PCIe ASPM controls are not
/// for the OS to enable. Left clear: 8042 (only its reset command exists), MSI-not-supported
/// (MSI works), CMOS-RTC-not-present.
const BOOT_ARCH_LEGACY_DEVICES: u16 = 1 << 0;
const BOOT_ARCH_VGA_NOT_PRESENT: u16 = 1 << 2;
const BOOT_ARCH_NO_ASPM: u16 = 1 << 4;
/// Fixed feature flags. WBINVD works; C1 is HLT; the power and sleep buttons are control-method
/// (absent) and the RTC cannot wake (FIX_RTC set); RESET_REG is supported.
const FLAG_WBINVD: u32 = 1 << 0;
const FLAG_PROCESSOR_C1: u32 = 1 << 2;
const FLAG_POWER_BUTTON: u32 = 1 << 4;
const FLAG_SLEEP_BUTTON: u32 = 1 << 5;
const FLAG_FIX_RTC: u32 = 1 << 6;
const FLAG_RESET_REGISTER: u32 = 1 << 10;
/// Port 0xCF9 is the reset control register (`RESET_CONTROL_PORT` in `dev`); with bit 2 set the
/// machine resets, and 0x06 is what Linux's ACPI reboot writes.
const RESET_PORT: u64 = 0xcf9;
const RESET_VALUE: u8 = 0x06;

/// FADT revision 6.0 (276 bytes). No SMI_CMD and no ACPI_ENABLE value: ACPI starts enabled, which
/// is how Linux decides not to try the SMI handshake (`acpi_enable` only runs with SMI_CMD).
///
/// ACPI 6 5.2.9 allows only one of FIRMWARE_CTRL and X_FIRMWARE_CTRL to be non-zero; with both
/// set ACPICA installs the FACS twice and Linux logs it twice. Like QEMU, the 32-bit field wins
/// when the address fits.
pub(super) fn fadt(dsdt: u64, facs: u64) -> Vec<u8> {
    let mut body = Vec::with_capacity(FADT_BODY_LENGTH);
    let dword = |body: &mut Vec<u8>, value: u32| body.extend_from_slice(&value.to_le_bytes());
    let word = |body: &mut Vec<u8>, value: u16| body.extend_from_slice(&value.to_le_bytes());
    let (firmware_ctrl, x_firmware_ctrl) = match u32::try_from(facs) {
        Ok(low) => (low, 0),
        Err(_) => (0, facs),
    };
    dword(&mut body, firmware_ctrl);
    dword(&mut body, dsdt as u32);
    body.push(0); // reserved (was INT_MODEL)
    body.push(0); // preferred PM profile: unspecified
    word(&mut body, SCI_IRQ as u16);
    dword(&mut body, 0); // SMI_CMD
    body.extend_from_slice(&[0, 0, 0, 0]); // ACPI_ENABLE, ACPI_DISABLE, S4BIOS_REQ, PSTATE_CNT
    dword(&mut body, PM1_EVENT_BASE as u32);
    dword(&mut body, 0); // PM1b_EVT_BLK
    dword(&mut body, PM1_CONTROL_BASE as u32);
    dword(&mut body, 0); // PM1b_CNT_BLK
    dword(&mut body, 0); // PM2_CNT_BLK
    dword(&mut body, PM_TIMER_BASE as u32);
    dword(&mut body, 0); // GPE0_BLK
    dword(&mut body, 0); // GPE1_BLK
    body.extend_from_slice(&[
        PM1_EVENT_LENGTH,
        PM1_CONTROL_LENGTH,
        0, // PM2_CNT_LEN
        PM_TIMER_LENGTH,
        0, // GPE0_BLK_LEN
        0, // GPE1_BLK_LEN
        0, // GPE1_BASE
        0, // CST_CNT
    ]);
    word(&mut body, LATENCY_C2_UNSUPPORTED);
    word(&mut body, LATENCY_C3_UNSUPPORTED);
    word(&mut body, 0); // FLUSH_SIZE
    word(&mut body, 0); // FLUSH_STRIDE
    body.extend_from_slice(&[0, 0, 0, 0, RTC_CENTURY_REGISTER]); // duty, day/month alarm, century
    word(
        &mut body,
        BOOT_ARCH_LEGACY_DEVICES | BOOT_ARCH_VGA_NOT_PRESENT | BOOT_ARCH_NO_ASPM,
    );
    body.push(0); // reserved
    dword(
        &mut body,
        FLAG_WBINVD
            | FLAG_PROCESSOR_C1
            | FLAG_POWER_BUTTON
            | FLAG_SLEEP_BUTTON
            | FLAG_FIX_RTC
            | FLAG_RESET_REGISTER,
    );
    body.extend(generic_address(GAS_SYSTEM_IO, BITS_PER_BYTE, RESET_PORT));
    body.push(RESET_VALUE);
    word(&mut body, 0); // ARM_BOOT_ARCH
    body.push(FADT_MINOR_VERSION);
    body.extend_from_slice(&x_firmware_ctrl.to_le_bytes());
    body.extend_from_slice(&dsdt.to_le_bytes()); // X_DSDT
    let pm_register =
        |base: u16, length: u8| generic_address(GAS_SYSTEM_IO, length * BITS_PER_BYTE, base as u64);
    body.extend(pm_register(PM1_EVENT_BASE, PM1_EVENT_LENGTH));
    body.extend(generic_address(0, 0, 0)); // X_PM1b_EVT_BLK
    body.extend(pm_register(PM1_CONTROL_BASE, PM1_CONTROL_LENGTH));
    body.extend(generic_address(0, 0, 0)); // X_PM1b_CNT_BLK
    body.extend(generic_address(0, 0, 0)); // X_PM2_CNT_BLK
    body.extend(pm_register(PM_TIMER_BASE, PM_TIMER_LENGTH));
    body.extend(generic_address(0, 0, 0)); // X_GPE0_BLK
    body.extend(generic_address(0, 0, 0)); // X_GPE1_BLK
    body.extend(generic_address(0, 0, 0)); // SLEEP_CONTROL_REG (hardware-reduced only)
    body.extend(generic_address(0, 0, 0)); // SLEEP_STATUS_REG
    body.extend_from_slice(&[0; 8]); // hypervisor vendor identity
    debug_assert_eq!(body.len(), FADT_BODY_LENGTH);
    table(b"FACP", FADT_REVISION, &body)
}

const MADT_REVISION: u8 = 6;
/// PCAT_COMPAT: the dual 8259s exist, so Linux masks them when it takes over with the APIC.
const MADT_FLAG_PCAT_COMPAT: u32 = 1;
const MADT_LOCAL_APIC: u8 = 0;
const MADT_IO_APIC: u8 = 1;
const MADT_INTERRUPT_OVERRIDE: u8 = 2;
const MADT_LOCAL_APIC_NMI: u8 = 4;
const MADT_X2APIC: u8 = 9;
const MADT_X2APIC_NMI: u8 = 10;
const LOCAL_APIC_ENTRY_LENGTH: u8 = 8;
const IO_APIC_ENTRY_LENGTH: u8 = 12;
const OVERRIDE_ENTRY_LENGTH: u8 = 10;
const LOCAL_APIC_NMI_ENTRY_LENGTH: u8 = 6;
const X2APIC_ENTRY_LENGTH: u8 = 16;
const X2APIC_NMI_ENTRY_LENGTH: u8 = 12;
const ENTRY_ENABLED: u32 = 1;
/// APIC ids from this value up do not fit the 8-bit field (255 is the broadcast id), so those
/// processors are listed as x2APIC entries.
const FIRST_X2APIC_ONLY_ID: u32 = 0xff;
/// "All processors" in the NMI entries.
const ALL_PROCESSORS_XAPIC: u8 = 0xff;
const ALL_PROCESSORS_X2APIC: u32 = 0xffff_ffff;
/// LINT1 is the NMI pin.
const NMI_PIN: u8 = 1;
/// Source override flags (5.2.12.5): polarity in bits 1:0 and trigger in bits 3:2, 0 meaning
/// "conforms to the bus" (ISA: active high, edge).
const OVERRIDE_CONFORMS: u16 = 0;
/// Level triggered, active low: what ACPI says a PCI interrupt is and what Linux asks for when
/// it registers a `_PRT` GSI. The two must agree: a PVH dom0 has no legacy PIC, so Linux does
/// not let the first user of a legacy IRQ reprogram the pin and fails the PCI function's
/// probe with EBUSY on a mismatch.
const OVERRIDE_ACTIVE_LOW_LEVEL: u16 = 0b11_11;
/// The IOAPIC's first interrupt input.
const GSI_BASE: u32 = 0;
/// The PIT is wired to IOAPIC input 2, not 0 (`Devices::ioapic_pin`).
const PIT_ISA_IRQ: u8 = 0;
const PIT_GSI: u32 = 2;

/// One local APIC per processor (id = index), the IOAPIC, the PIT's IRQ0-to-GSI2 wire, the ISA
/// lines the PCI functions and the SCI share (level triggered, which Linux refuses to
/// reprogram for a legacy IRQ without an override saying so), and NMI on LINT1.
pub(super) fn madt(cpus: u32, level_triggered_irqs: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&(APIC_DEFAULT_BASE as u32).to_le_bytes());
    body.extend_from_slice(&MADT_FLAG_PCAT_COMPAT.to_le_bytes());
    for cpu in 0..cpus {
        if cpu < FIRST_X2APIC_ONLY_ID {
            body.extend_from_slice(&[
                MADT_LOCAL_APIC,
                LOCAL_APIC_ENTRY_LENGTH,
                cpu as u8,
                cpu as u8,
            ]);
            body.extend_from_slice(&ENTRY_ENABLED.to_le_bytes());
        } else {
            body.extend_from_slice(&[MADT_X2APIC, X2APIC_ENTRY_LENGTH, 0, 0]);
            body.extend_from_slice(&cpu.to_le_bytes());
            body.extend_from_slice(&ENTRY_ENABLED.to_le_bytes());
            body.extend_from_slice(&cpu.to_le_bytes());
        }
    }
    body.extend_from_slice(&[MADT_IO_APIC, IO_APIC_ENTRY_LENGTH, IOAPIC_ID, 0]);
    body.extend_from_slice(&(IOAPIC_DEFAULT_BASE as u32).to_le_bytes());
    body.extend_from_slice(&GSI_BASE.to_le_bytes());
    let mut override_entry = |irq: u8, gsi: u32, flags: u16| {
        body.extend_from_slice(&[MADT_INTERRUPT_OVERRIDE, OVERRIDE_ENTRY_LENGTH, 0, irq]);
        body.extend_from_slice(&gsi.to_le_bytes());
        body.extend_from_slice(&flags.to_le_bytes());
    };
    override_entry(PIT_ISA_IRQ, PIT_GSI, OVERRIDE_CONFORMS);
    for &irq in level_triggered_irqs {
        override_entry(irq, irq as u32, OVERRIDE_ACTIVE_LOW_LEVEL);
    }
    if cpus > FIRST_X2APIC_ONLY_ID {
        body.extend_from_slice(&[MADT_X2APIC_NMI, X2APIC_NMI_ENTRY_LENGTH, 0, 0]);
        body.extend_from_slice(&ALL_PROCESSORS_X2APIC.to_le_bytes());
        body.push(NMI_PIN);
        body.extend_from_slice(&[0; 3]);
    } else {
        body.extend_from_slice(&[MADT_LOCAL_APIC_NMI, LOCAL_APIC_NMI_ENTRY_LENGTH]);
        body.push(ALL_PROCESSORS_XAPIC);
        body.extend_from_slice(&OVERRIDE_CONFORMS.to_le_bytes());
        body.push(NMI_PIN);
    }
    table(b"APIC", MADT_REVISION, &body)
}

const MCFG_REVISION: u8 = 1;
const MCFG_RESERVED_BYTES: usize = 8;

/// One ECAM window for segment 0, buses `first_bus..=last_bus`.
pub(super) fn mcfg(base: u64, first_bus: u8, last_bus: u8) -> Vec<u8> {
    let mut body = vec![0u8; MCFG_RESERVED_BYTES];
    body.extend_from_slice(&base.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&[first_bus, last_bus]);
    body.extend_from_slice(&[0; 4]);
    table(b"MCFG", MCFG_REVISION, &body)
}

const HPET_REVISION: u8 = 1;
const HPET_REGISTER_BITS: u8 = 64;
/// Smallest tick the OS may program in periodic mode; Linux only prints it.
const HPET_MINIMUM_TICK: u16 = 0x1000;

pub(super) fn hpet() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&CAPABILITIES_LOW_DWORD.to_le_bytes());
    body.extend(generic_address(
        GAS_SYSTEM_MEMORY,
        HPET_REGISTER_BITS,
        HPET_BASE,
    ));
    body.push(0); // HPET number
    body.extend_from_slice(&HPET_MINIMUM_TICK.to_le_bytes());
    body.push(0); // page protection and OEM attributes
    table(b"HPET", HPET_REVISION, &body)
}

/// TCG ACPI Specification 1.3, "TPM2 table", revision 4, ending after the start method
/// parameters, without the log area fields: the event log reaches Linux from the UEFI
/// firmware's configuration table, Linux reads LAML/LASA only from a table long enough to
/// have them, and an empty log area there would only make it warn.
const TPM2_REVISION: u8 = 4;
const TPM2_PLATFORM_CLASS_CLIENT: u16 = 0;
/// "Uses the Command Response Buffer Interface".
const TPM2_START_METHOD_CRB: u32 = 7;
const TPM2_START_METHOD_PARAMETER_BYTES: usize = 12;

pub(super) fn tpm2() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&TPM2_PLATFORM_CLASS_CLIENT.to_le_bytes());
    body.extend_from_slice(&[0; 2]); // reserved
    body.extend_from_slice(&(TPM_CRB_BASE + CTRL_REQ).to_le_bytes());
    body.extend_from_slice(&TPM2_START_METHOD_CRB.to_le_bytes());
    body.extend_from_slice(&[0; TPM2_START_METHOD_PARAMETER_BYTES]);
    table(b"TPM2", TPM2_REVISION, &body)
}
