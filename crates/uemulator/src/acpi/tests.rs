use super::*;

const SIGNATURE_LENGTH: usize = 4;
const HEADER_LENGTH: usize = 36;

fn config(cpus: u32, pci_functions: usize, iommu: bool) -> AcpiConfig {
    AcpiConfig {
        cpus,
        pci_functions,
        iommu,
    }
}

/// The region placed at its guest address inside a RAM-sized buffer.
fn ram_with_tables(config: &AcpiConfig) -> Vec<u8> {
    let mut ram = vec![0u8; REGION_BASE as usize + REGION_BYTES as usize];
    install(&mut ram, config).unwrap();
    ram
}

fn checksum_ok(bytes: &[u8]) -> bool {
    bytes.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)) == 0
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// The table at guest address `address`, length from its header.
fn table_at(ram: &[u8], address: u64) -> &[u8] {
    let start = address as usize;
    let length = u32_at(ram, start + 4) as usize;
    &ram[start..start + length]
}

fn xsdt_tables(ram: &[u8]) -> Vec<(String, &[u8])> {
    let xsdt_address = u64_at(ram, REGION_BASE as usize + 24);
    let xsdt = table_at(ram, xsdt_address);
    xsdt[HEADER_LENGTH..]
        .chunks(8)
        .map(|entry| {
            let table = table_at(ram, u64::from_le_bytes(entry.try_into().unwrap()));
            (
                String::from_utf8(table[..SIGNATURE_LENGTH].to_vec()).unwrap(),
                table,
            )
        })
        .collect()
}

fn find<'a>(ram: &'a [u8], signature: &str) -> &'a [u8] {
    xsdt_tables(ram)
        .into_iter()
        .find(|(name, _)| name == signature)
        .unwrap_or_else(|| panic!("no {signature} table"))
        .1
}

#[test]
fn rsdp_is_revision_2_with_both_checksums_at_the_region_start() {
    let ram = ram_with_tables(&config(1, 0, false));
    let rsdp = &ram[REGION_BASE as usize..REGION_BASE as usize + 36];
    assert_eq!(&rsdp[..8], b"RSD PTR ");
    assert_eq!(rsdp[15], 2);
    assert!(checksum_ok(&rsdp[..20]), "ACPI 1.0 checksum");
    assert!(checksum_ok(rsdp), "extended checksum");
    assert_eq!(u32_at(rsdp, 20), 36);
    assert_eq!(
        REGION_BASE % 16,
        0,
        "the BIOS scan finds RSDPs on 16-byte boundaries"
    );
}

#[test]
fn every_table_has_a_valid_checksum_and_length_and_both_roots_agree() {
    let ram = ram_with_tables(&config(2, 3, true));
    let rsdt_address = u32_at(&ram, REGION_BASE as usize + 16) as u64;
    let rsdt = table_at(&ram, rsdt_address);
    assert_eq!(&rsdt[..4], b"RSDT");
    let rsdt_entries: Vec<u64> = rsdt[HEADER_LENGTH..]
        .chunks(4)
        .map(|entry| u32::from_le_bytes(entry.try_into().unwrap()) as u64)
        .collect();
    let tables = xsdt_tables(&ram);
    let names: Vec<&str> = tables.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["FACP", "APIC", "MCFG", "HPET", "IVRS"]);
    assert_eq!(rsdt_entries.len(), tables.len());
    for (_, table) in &tables {
        assert!(checksum_ok(table), "{:?}", &table[..4]);
    }
    assert!(checksum_ok(rsdt));
    assert!(checksum_ok(table_at(
        &ram,
        u64_at(&ram, REGION_BASE as usize + 24)
    )));
}

#[test]
fn tables_stay_inside_the_region_and_do_not_overlap() {
    let image = build(&config(255, 31, true)).unwrap();
    assert!(image.len() as u64 <= REGION_BYTES);
}

#[test]
fn ivrs_is_published_only_with_the_iommu() {
    let without = ram_with_tables(&config(1, 1, false));
    assert!(xsdt_tables(&without).iter().all(|(name, _)| name != "IVRS"));
    let with = ram_with_tables(&config(1, 1, true));
    assert!(xsdt_tables(&with).iter().any(|(name, _)| name == "IVRS"));
}

#[test]
fn ivrs_lists_every_function_and_the_special_requesters_the_interrupt_path_uses() {
    let ram = ram_with_tables(&config(1, 2, true));
    let ivrs = find(&ram, "IVRS");
    let contains = |needle: &[u8]| ivrs.windows(needle.len()).any(|window| window == needle);
    // Select entries: host bridge 00:00.0, IOMMU 00:00.2, slots 1 and 2, and the requester ids
    // of the I/O APIC and the HPET (which sets how large Linux makes the device table).
    for devfn in [
        0x0000u16,
        0x0002,
        0x0008,
        0x0010,
        IOAPIC_REQUESTER,
        HPET_REQUESTER,
    ] {
        let mut entry = vec![0x02];
        entry.extend_from_slice(&devfn.to_le_bytes());
        entry.push(0);
        assert!(contains(&entry), "device {devfn:#x}");
    }
    // Special entries: IOAPIC id and requester, HPET number and requester.
    let special = |handle: u8, requester: u16, variety: u8| {
        let mut entry = vec![0x48, 0, 0, 0, handle];
        entry.extend_from_slice(&requester.to_le_bytes());
        entry.push(variety);
        entry
    };
    assert!(contains(&special(IOAPIC_ID, IOAPIC_REQUESTER, 1)));
    assert!(contains(&special(0, HPET_REQUESTER, 2)));
}

#[test]
fn fadt_describes_the_pm_blocks_reset_register_and_starts_acpi_enabled() {
    use crate::dev::acpi_pm::*;
    let ram = ram_with_tables(&config(1, 0, false));
    let fadt = find(&ram, "FACP");
    assert_eq!(fadt.len(), 276);
    assert_eq!(fadt[8], 6, "revision");
    assert_eq!(
        u32_at(fadt, 48),
        0,
        "no SMI_CMD: ACPI is enabled from reset"
    );
    assert_eq!(u16_at(fadt, 46), SCI_IRQ as u16);
    assert_eq!(u32_at(fadt, 56), PM1_EVENT_BASE as u32);
    assert_eq!(u32_at(fadt, 64), PM1_CONTROL_BASE as u32);
    assert_eq!(u32_at(fadt, 76), PM_TIMER_BASE as u32);
    assert_eq!(&fadt[88..92], [4, 2, 0, 4]);
    let flags = u32_at(fadt, 112);
    assert_eq!(flags & (1 << 20), 0, "not hardware-reduced");
    assert_ne!(flags & (1 << 10), 0, "reset register supported");
    assert_eq!(flags & (1 << 8), 0, "24-bit PM timer");
    // RESET_REG: system I/O, port 0xCF9, value 6 (bit 2 resets).
    assert_eq!(&fadt[116..120], [1, 8, 0, 0]);
    assert_eq!(u64_at(fadt, 120), 0xcf9);
    assert_eq!(fadt[128], 6);
    // The extended addresses agree with the legacy ones.
    assert_eq!(u64_at(fadt, 140), u32_at(fadt, 40) as u64, "X_DSDT");
    // ACPI 6 5.2.9: at most one of FIRMWARE_CTRL and X_FIRMWARE_CTRL is non-zero (Linux installs
    // and prints the FACS once per field otherwise). The FACS is below 4 GiB, so the 32-bit one.
    assert_ne!(u32_at(fadt, 36), 0, "FIRMWARE_CTRL");
    assert_eq!(u64_at(fadt, 132), 0, "X_FIRMWARE_CTRL");
    assert_eq!(u64_at(fadt, 152), PM1_EVENT_BASE as u64);
    assert_eq!(u64_at(fadt, 176), PM1_CONTROL_BASE as u64);
    assert_eq!(u64_at(fadt, 212), PM_TIMER_BASE as u64);
}

#[test]
fn fadt_points_at_a_dsdt_and_a_facs_that_exist() {
    let ram = ram_with_tables(&config(1, 0, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    assert_eq!(&dsdt[..4], b"DSDT");
    assert!(checksum_ok(dsdt));
    let facs_address = u32_at(fadt, 36) as usize;
    assert_eq!(facs_address % 64, 0, "the FACS must be 64-byte aligned");
    let facs = &ram[facs_address..][..64];
    assert_eq!(&facs[..4], b"FACS");
    assert_eq!(u32_at(facs, 4), 64);
}

/// (type, bytes) of every MADT entry.
fn madt_entries(madt: &[u8]) -> Vec<(u8, &[u8])> {
    let mut entries = Vec::new();
    let mut at = HEADER_LENGTH + 8;
    while at < madt.len() {
        let length = madt[at + 1] as usize;
        entries.push((madt[at], &madt[at..at + length]));
        at += length;
    }
    assert_eq!(at, madt.len(), "entries tile the table");
    entries
}

#[test]
fn madt_lists_one_cpu_the_ioapic_and_the_interrupt_wiring() {
    let ram = ram_with_tables(&config(1, 0, false));
    let madt = find(&ram, "APIC");
    assert_eq!(u32_at(madt, HEADER_LENGTH), 0xfee0_0000);
    assert_eq!(u32_at(madt, HEADER_LENGTH + 4) & 1, 1, "PCAT_COMPAT");
    let entries = madt_entries(madt);
    let of_type = |kind: u8| entries.iter().filter(move |(t, _)| *t == kind);
    let cpus: Vec<_> = of_type(0).collect();
    assert_eq!(cpus.len(), 1);
    assert_eq!(cpus[0].1[3], 0, "APIC id");
    assert_eq!(u32_at(cpus[0].1, 4) & 1, 1, "enabled");
    let ioapic = of_type(1).next().unwrap().1;
    assert_eq!(ioapic[2], IOAPIC_ID);
    assert_eq!(u32_at(ioapic, 4), 0xfec0_0000);
    assert_eq!(u32_at(ioapic, 8), 0, "GSI base");
    let overrides: Vec<(u8, u32, u16)> = of_type(2)
        .map(|(_, entry)| (entry[3], u32_at(entry, 4), u16_at(entry, 8)))
        .collect();
    assert!(
        overrides.contains(&(0, 2, 0)),
        "PIT on GSI 2: {overrides:?}"
    );
    for irq in [5u8, 9, 10, 11, crate::dev::acpi_pm::SCI_IRQ] {
        assert!(
            overrides.contains(&(irq, irq as u32, 0xf)),
            "level-triggered IRQ {irq}: {overrides:?}"
        );
    }
    let nmi = of_type(4).next().unwrap().1;
    assert_eq!(nmi[2], 0xff, "all processors");
    assert_eq!(nmi[5], 1, "LINT1");
}

#[test]
fn madt_lists_every_cpu_with_the_apic_id_the_vcpu_starts_with() {
    // vCPU n's local APIC starts with id n (`Devices::new`), and its ACPI processor id matches.
    let ram = ram_with_tables(&config(4, 0, false));
    let entries = madt_entries(find(&ram, "APIC"));
    let cpus: Vec<(u8, u8, bool)> = entries
        .iter()
        .filter(|(kind, _)| *kind == 0)
        .map(|(_, entry)| (entry[2], entry[3], u32_at(entry, 4) & 1 == 1))
        .collect();
    assert_eq!(
        cpus,
        [(0, 0, true), (1, 1, true), (2, 2, true), (3, 3, true)]
    );
}

#[test]
fn madt_switches_to_x2apic_entries_for_ids_that_do_not_fit_in_a_byte() {
    let ram = ram_with_tables(&config(300, 0, false));
    let entries = madt_entries(find(&ram, "APIC"));
    let xapic: Vec<_> = entries.iter().filter(|(t, _)| *t == 0).collect();
    let x2apic: Vec<_> = entries.iter().filter(|(t, _)| *t == 9).collect();
    assert_eq!(xapic.len(), 255, "ids 0..=254");
    assert_eq!(x2apic.len(), 45);
    assert_eq!(u32_at(x2apic[0].1, 4), 255);
    assert_eq!(u32_at(x2apic[44].1, 4), 299);
    assert_eq!(u32_at(x2apic[44].1, 12), 299, "ACPI UID");
    assert!(entries.iter().any(|(t, _)| *t == 10), "x2APIC NMI");
    assert!(entries.iter().all(|(t, _)| *t != 4));
}

#[test]
fn mcfg_maps_the_ecam_window_of_bus_zero() {
    let ram = ram_with_tables(&config(1, 0, false));
    let mcfg = find(&ram, "MCFG");
    assert_eq!(mcfg.len(), HEADER_LENGTH + 8 + 16);
    let entry = &mcfg[HEADER_LENGTH + 8..];
    assert_eq!(u64_at(entry, 0), ECAM_BASE);
    assert_eq!(u16_at(entry, 8), 0, "segment");
    assert_eq!(&entry[10..12], [0, 0], "bus range");
}

#[test]
fn hpet_table_matches_the_device() {
    use crate::dev::hpet::*;
    let ram = ram_with_tables(&config(1, 0, false));
    let table = find(&ram, "HPET");
    assert_eq!(table.len(), 56);
    let id = u32_at(table, HEADER_LENGTH);
    assert_eq!(id >> 16, 0x8086);
    assert_eq!(((id >> 8) & 0x1f) as usize, TIMER_COUNT - 1);
    assert_eq!(id & (1 << 15), 0, "no legacy replacement");
    assert_eq!(&table[40..44], [0, 64, 0, 0], "system memory, 64 bits");
    assert_eq!(u64_at(table, 44), HPET_BASE);
}

/// Decodes a PkgLength at `at`: (length including its own bytes, number of bytes it took).
fn package_length(bytes: &[u8], at: usize) -> (usize, usize) {
    let lead = bytes[at];
    let extra = (lead >> 6) as usize;
    if extra == 0 {
        return ((lead & 0x3f) as usize, 1);
    }
    let mut length = (lead & 0xf) as usize;
    for byte in 0..extra {
        length |= (bytes[at + 1 + byte] as usize) << (4 + 8 * byte);
    }
    (length, 1 + extra)
}

/// The size of the name string at `at`.
fn name_length(bytes: &[u8], at: usize) -> usize {
    let mut length = 0;
    while matches!(bytes[at + length], b'\\' | b'^') {
        length += 1;
    }
    if bytes[at + length] == 0x2e {
        length + 9
    } else {
        length + 4
    }
}

/// The size of the data object at `at`, checking nested lengths on the way.
fn data_length(bytes: &[u8], at: usize) -> usize {
    match bytes[at] {
        0x00 | 0x01 => 1,
        0x0a => 2,
        0x0b => 3,
        0x0c => 5,
        0x0e => 9,
        0x0d => 2 + bytes[at + 1..].iter().position(|b| *b == 0).unwrap(),
        0x11 => {
            let (length, prefix) = package_length(bytes, at + 1);
            let size_at = at + 1 + prefix;
            let size_bytes = data_length(bytes, size_at);
            let buffer_size = match bytes[size_at] {
                0x0a => bytes[size_at + 1] as usize,
                0x0b => u16_at(bytes, size_at + 1) as usize,
                _ => unreachable!("buffer size"),
            };
            assert_eq!(
                prefix + size_bytes + buffer_size,
                length,
                "buffer length at {at:#x}"
            );
            1 + length
        }
        0x12 => {
            let (length, prefix) = package_length(bytes, at + 1);
            let count = bytes[at + 1 + prefix] as usize;
            let mut cursor = at + 2 + prefix;
            for _ in 0..count {
                cursor += data_length(bytes, cursor);
            }
            assert_eq!(cursor, at + 1 + length, "package length at {at:#x}");
            1 + length
        }
        other => panic!("unexpected data opcode {other:#x} at {at:#x}"),
    }
}

/// Walks a term list, recording the path of every named object, and returns the offset after it.
fn walk(bytes: &[u8], mut at: usize, end: usize, scope: &str, names: &mut Vec<String>) {
    let segment = |at: usize| String::from_utf8_lossy(&bytes[at..at + 4]).into_owned();
    while at < end {
        match bytes[at] {
            0x08 => {
                let length = name_length(bytes, at + 1);
                let name = bytes[at + 1..at + 1 + length]
                    .iter()
                    .filter(|b| b.is_ascii_alphanumeric() || **b == b'_')
                    .map(|b| *b as char)
                    .collect::<String>();
                names.push(format!("{scope}.{name}"));
                at += 1 + length + data_length(bytes, at + 1 + length);
            }
            // A method's body is code, not namespace: record the name and skip it.
            0x14 => {
                let (length, prefix) = package_length(bytes, at + 1);
                let name_at = at + 1 + prefix;
                names.push(format!("{scope}.{}", segment(name_at)));
                at += 1 + length;
            }
            0x10 | 0x5b => {
                let device = bytes[at] == 0x5b;
                assert!(!device || bytes[at + 1] == 0x82, "only Device is used");
                let opcode_bytes = if device { 2 } else { 1 };
                let (length, prefix) = package_length(bytes, at + opcode_bytes);
                let body_end = at + opcode_bytes + length;
                let name_at = at + opcode_bytes + prefix;
                let name_bytes = name_length(bytes, name_at);
                let name = String::from_utf8_lossy(&bytes[name_at..name_at + name_bytes])
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .to_string();
                let inner = format!("{scope}.{name}");
                names.push(inner.clone());
                walk(bytes, name_at + name_bytes, body_end, &inner, names);
                at = body_end;
            }
            other => panic!(
                "unexpected term opcode {other:#x} at {at:#x} ({})",
                segment(at)
            ),
        }
    }
    assert_eq!(at, end, "terms tile their scope");
}

fn dsdt_names(pci_functions: usize) -> Vec<String> {
    let ram = ram_with_tables(&config(1, pci_functions, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let mut names = Vec::new();
    walk(dsdt, HEADER_LENGTH, dsdt.len(), "", &mut names);
    names
}

#[test]
fn dsdt_is_structurally_valid_aml_with_the_expected_namespace() {
    let names = dsdt_names(2);
    for expected in [
        "._SB_",
        "._SB_.PCI0",
        "._SB_.PCI0._HID",
        "._SB_.PCI0._CID",
        "._SB_.PCI0._SEG",
        "._SB_.PCI0._BBN",
        "._SB_.PCI0._UID",
        "._SB_.PCI0._CRS",
        "._SB_.PCI0._PRT",
        "._SB_.PCI0._OSC",
        "._SB_.COM1",
        "._SB_.COM1._HID",
        "._SB_.COM1._CRS",
        "._SB_.RTC_",
        "._SB_.PIC_",
        "._SB_.PIT_",
        "._SB_.SYSR._CRS",
        "._S5_",
    ] {
        assert!(
            names.iter().any(|name| name == expected),
            "{expected} in {names:?}"
        );
    }
}

#[test]
fn dsdt_s5_package_carries_the_sleep_type_the_pm_block_accepts() {
    let ram = ram_with_tables(&config(1, 0, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let needle = b"_S5_\x12\x08\x04\x0a\x05\x0a\x05\x00\x00";
    assert!(
        dsdt.windows(needle.len()).any(|window| window == needle),
        "{dsdt:02x?}"
    );
    assert_eq!(crate::dev::acpi_pm::SLEEP_TYPE_S5, 5);
}

#[test]
fn prt_routes_each_slot_to_the_legacy_line_of_its_function() {
    let ram = ram_with_tables(&config(1, 5, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let at = dsdt.windows(4).position(|w| w == b"_PRT").unwrap();
    // Package, length, count.
    let (_, prefix) = package_length(dsdt, at + 5);
    assert_eq!(dsdt[at + 4], 0x12);
    assert_eq!(dsdt[at + 5 + prefix], 5, "one entry per function");
    // First entry: slot 1 (0x0001FFFF), INTA (0), no link (0), IRQ 9.
    let first = &dsdt[at + 6 + prefix..];
    assert_eq!(&first[..2], [0x12, first[1]]);
    assert!(
        first[..first[1] as usize + 1]
            .windows(5)
            .any(|w| w == [0x0c, 0xff, 0xff, 0x01, 0x00])
    );
    let irqs: Vec<u8> = dsdt[at..]
        .windows(5)
        .filter(|w| w[0] == 0x00 && w[1] == 0x00 && w[2] == 0x0a)
        .map(|w| w[3])
        .collect();
    assert_eq!(irqs[..5], [9, 10, 11, 5, 9]);
}

#[test]
fn install_rejects_ram_too_small_for_the_region() {
    let mut ram = vec![0u8; 0x1000];
    assert!(matches!(
        install(&mut ram, &config(1, 0, false)),
        Err(Error::BadConfig(_))
    ));
}

#[test]
fn pci_root_bridge_osc_takes_four_arguments_and_grants_what_q35_grants() {
    let ram = ram_with_tables(&config(1, 0, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let at = dsdt.windows(4).position(|w| w == b"_OSC").unwrap();
    // Method opcode, then a two-byte PkgLength (the body is longer than 63 bytes), then the name.
    assert_eq!(dsdt[at - 3], 0x14, "a method");
    let (length, prefix) = package_length(dsdt, at - 2);
    assert_eq!(prefix, 2);
    let method = &dsdt[at - 3..at - 2 + length];
    // Not serialized, four arguments; the PCI host bridge UUID; And (Local0, 0x1F, Local0); the
    // whole Arg3 buffer comes back.
    assert_eq!(dsdt[at + 4], 4);
    let uuid = [
        0x5b, 0x4d, 0xdb, 0x33, 0xf7, 0x1f, 0x1c, 0x40, 0x96, 0x57, 0x74, 0x41, 0xc0, 0x3d, 0xd7,
        0x66,
    ];
    assert!(method.windows(16).any(|w| w == uuid));
    assert!(
        method
            .windows(5)
            .any(|w| w == [0x7b, 0x60, 0x0a, 0x1f, 0x60])
    );
    assert!(method.ends_with(&[0xa4, 0x6b]));
}
