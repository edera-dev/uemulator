use super::*;

const SIGNATURE_LENGTH: usize = 4;
const HEADER_LENGTH: usize = 36;

fn config(cpus: u32, pci_functions: usize, iommu: bool) -> AcpiConfig {
    AcpiConfig {
        cpus,
        pci_functions,
        iommu,
        tpm: false,
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
    assert_eq!(
        names,
        [
            "FACP", "APIC", "MCFG", "HPET", "SRAT", "SLIT", "SPCR", "IVRS"
        ]
    );
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
    let image = build(
        &config(255, crate::dev::pci::MAX_VIRTIO_FUNCTIONS, true),
        REGION_BASE + REGION_BYTES,
    )
    .unwrap();
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
    assert_eq!(fadt[45], 4, "preferred PM profile: enterprise server");
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
    dsdt_names_of(&config(1, pci_functions, false))
}

fn dsdt_names_of(config: &AcpiConfig) -> Vec<String> {
    let ram = ram_with_tables(config);
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

#[test]
fn a_tpm_adds_the_tpm2_table_and_its_device() {
    let without = ram_with_tables(&config(1, 0, false));
    assert!(xsdt_tables(&without).iter().all(|(name, _)| name != "TPM2"));
    assert!(
        !dsdt_names(0)
            .iter()
            .any(|name| name.starts_with("._SB_.TPM_"))
    );

    let with_tpm = AcpiConfig {
        tpm: true,
        ..config(1, 0, false)
    };
    let ram = ram_with_tables(&with_tpm);
    let tpm2 = find(&ram, "TPM2");
    assert_eq!(tpm2.len(), 64, "no log area");
    assert_eq!(tpm2[8], 4, "revision");
    assert_eq!(u64_at(tpm2, 40), 0xfed4_0040, "control area");
    assert_eq!(u32_at(tpm2, 48), 7, "start method: CRB");
    let names = dsdt_names_of(&with_tpm);
    for expected in ["._SB_.TPM_._HID", "._SB_.TPM_._CRS", "._SB_.TPM_._STA"] {
        assert!(
            names.iter().any(|name| name == expected),
            "{expected} in {names:?}"
        );
    }
}

#[test]
fn ivrs_and_prt_skip_the_fch_and_data_fabric_slots() {
    let functions = 0x17;
    let ram = ram_with_tables(&config(1, functions, true));
    let ivrs = find(&ram, "IVRS");
    let select = |devfn: u16| {
        let mut entry = vec![0x02];
        entry.extend_from_slice(&devfn.to_le_bytes());
        entry.push(0);
        ivrs.windows(4).any(|window| window == entry)
    };
    assert!(select(0x17 << 3));
    assert!(!select(0x18 << 3), "00:18.0 is the data fabric");
    assert!(select(0x19 << 3), "the 23rd virtio function");
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let prt_address = |slot: u32| {
        let mut entry = vec![0x0c];
        entry.extend_from_slice(&(slot << 16 | 0xffff).to_le_bytes());
        dsdt.windows(5).any(|window| window == entry)
    };
    assert!(prt_address(0x17) && prt_address(0x19));
    assert!(!prt_address(0x14) && !prt_address(0x18));
}

fn srat_entries(srat: &[u8]) -> Vec<(u8, &[u8])> {
    let mut entries = Vec::new();
    let mut at = HEADER_LENGTH + 12;
    while at < srat.len() {
        let length = srat[at + 1] as usize;
        entries.push((srat[at], &srat[at..at + length]));
        at += length;
    }
    assert_eq!(at, srat.len());
    entries
}

fn entries_of_kind<'a>(entries: &[(u8, &'a [u8])], kind: u8) -> Vec<&'a [u8]> {
    entries
        .iter()
        .filter(|(entry_kind, _)| *entry_kind == kind)
        .map(|(_, entry)| *entry)
        .collect()
}

#[test]
fn srat_puts_every_processor_and_all_ram_in_node_zero() {
    let ram = ram_with_tables(&config(3, 0, false));
    let srat = find(&ram, "SRAT");
    assert_eq!(srat[8], 3, "revision");
    assert_eq!(u32_at(srat, HEADER_LENGTH), 1, "reserved, must be 1");
    let entries = srat_entries(srat);
    let processors = entries_of_kind(&entries, 0);
    assert_eq!(processors.len(), 3);
    for (cpu, entry) in processors.iter().enumerate() {
        assert_eq!(entry.len(), 16);
        assert_eq!(entry[2], 0, "domain bits 7:0");
        assert_eq!(entry[3] as usize, cpu, "APIC id as in the MADT");
        assert_eq!(u32_at(entry, 4), 1, "enabled");
        assert_eq!(&entry[9..12], [0, 0, 0], "domain bits 31:8");
    }
    let memory = entries_of_kind(&entries, 1);
    assert_eq!(memory.len(), 1, "no RAM above 4 GiB in this test");
    assert_eq!(memory[0].len(), 40);
    assert_eq!(u32_at(memory[0], 2), 0, "domain");
    assert_eq!(u64_at(memory[0], 8), 0, "base");
    assert_eq!(u64_at(memory[0], 16), ram.len() as u64, "length");
    assert_eq!(u32_at(memory[0], 28), 1, "enabled, not hot-pluggable");
}

#[test]
fn srat_covers_the_ram_on_both_sides_of_the_pci_hole_and_x2apic_processors() {
    let flat_len = crate::ram::flat_bytes(4 << 30);
    let image = build(&config(300, 0, false), flat_len).unwrap();
    let srat_at = image.windows(4).position(|w| w == b"SRAT").unwrap();
    let srat = &image[srat_at..srat_at + u32_at(&image, srat_at + 4) as usize];
    assert!(checksum_ok(srat));
    let entries = srat_entries(srat);
    let memory: Vec<(u64, u64)> = entries_of_kind(&entries, 1)
        .iter()
        .map(|entry| (u64_at(entry, 8), u64_at(entry, 16)))
        .collect();
    let hole = crate::ram::PCI_HOLE_START;
    assert_eq!(memory, [(0, hole), (1 << 32, (4 << 30) - hole)]);
    let x2apic = entries_of_kind(&entries, 2);
    assert_eq!(x2apic.len(), 300 - 255);
    assert_eq!(x2apic[0].len(), 24);
    assert_eq!(u32_at(x2apic[0], 4), 0, "domain");
    assert_eq!(u32_at(x2apic[0], 8), 255, "x2APIC id");
    assert_eq!(u32_at(x2apic[0], 12), 1, "enabled");
}

#[test]
fn slit_has_the_local_distance_for_the_one_node() {
    let ram = ram_with_tables(&config(1, 0, false));
    let slit = find(&ram, "SLIT");
    assert_eq!(slit.len(), HEADER_LENGTH + 8 + 1);
    assert_eq!(u64_at(slit, HEADER_LENGTH), 1);
    assert_eq!(slit[HEADER_LENGTH + 8], 10);
}

#[test]
fn spcr_describes_com1() {
    let ram = ram_with_tables(&config(1, 0, false));
    let spcr = find(&ram, "SPCR");
    assert_eq!(spcr.len(), 80);
    assert_eq!(spcr[8], 2, "revision");
    assert_eq!(spcr[36], 0, "16550");
    assert_eq!(
        &spcr[40..44],
        [1, 8, 0, 1],
        "system I/O, 8 bits, byte access"
    );
    assert_eq!(u64_at(spcr, 44), 0x3f8);
    assert_eq!(spcr[52], 0b11, "8259 and I/O APIC");
    assert_eq!(spcr[53], 4, "IRQ");
    assert_eq!(u32_at(spcr, 54), 4, "GSI");
    assert_eq!(spcr[58], 7, "115200 baud");
    assert_eq!(spcr[60], 1, "one stop bit");
    assert_eq!(u16_at(spcr, 64), 0xffff, "not a PCI device");
}

#[test]
fn every_processor_has_a_device_with_its_madt_uid_and_a_cpc() {
    let names = dsdt_names_of(&config(3, 0, false));
    for cpu in 0..3 {
        for object in ["_HID", "_UID", "_CPC"] {
            let expected = format!("._SB_.C{cpu:03X}.{object}");
            assert!(names.contains(&expected), "{expected} in {names:?}");
        }
    }
    assert!(!names.iter().any(|name| name.starts_with("._SB_.C003")));
    assert!(names.iter().any(|name| name == "._SB_.CPCT"));
    let ram = ram_with_tables(&config(3, 0, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let contains = |needle: &[u8]| dsdt.windows(needle.len()).any(|window| window == needle);
    // Device (C002) { Name (_HID, "ACPI0007") Name (_UID, 2) ...
    assert!(contains(b"C002\x08_HID\x0dACPI0007\x00\x08_UID\x0a\x02"));
    // Method (_CPC, 0) { Return (\_SB.CPCT) }
    assert!(contains(b"_CPC\x00\xa4\\\x2e_SB_CPCT"));
}

/// The elements of the package `\_SB.CPCT`, each as its encoded bytes.
fn cpc_elements(dsdt: &[u8]) -> Vec<&[u8]> {
    let name = dsdt.windows(5).position(|w| w == b"\x08CPCT").unwrap();
    let at = name + 5;
    assert_eq!(dsdt[at], 0x12, "a package");
    let (_, prefix) = package_length(dsdt, at + 1);
    let count = dsdt[at + 1 + prefix] as usize;
    let mut cursor = at + 2 + prefix;
    let mut elements = Vec::new();
    for _ in 0..count {
        let length = data_length(dsdt, cursor);
        elements.push(&dsdt[cursor..cursor + length]);
        cursor += length;
    }
    elements
}

/// The Generic Register descriptor inside a `Buffer` element: (space, width, offset, access,
/// address).
fn register_of(element: &[u8]) -> (u8, u8, u8, u8, u64) {
    assert_eq!(element[0], 0x11, "a buffer");
    let descriptor = element
        .windows(3)
        .position(|w| w == [0x82, 0x0c, 0x00])
        .unwrap();
    let r = &element[descriptor + 3..];
    (r[0], r[1], r[2], r[3], u64_at(r, 4))
}

#[test]
fn cpc_is_revision_3_and_points_at_the_cppc_msrs() {
    use crate::cpu::cppc::*;
    const FFH: u8 = 0x7f;
    let ram = ram_with_tables(&config(1, 0, false));
    let fadt = find(&ram, "FACP");
    let dsdt = table_at(&ram, u32_at(fadt, 40) as u64);
    let elements = cpc_elements(dsdt);
    assert_eq!(elements.len(), 23);
    assert_eq!(elements[0], [0x0a, 23], "NumEntries");
    assert_eq!(elements[1], [0x0a, 3], "Revision");
    let field = |msr: u32, offset: u8, width: u8| (FFH, width, offset, 4, msr as u64);
    let capability = MSR_CPPC_CAPABILITY_1;
    let request = MSR_CPPC_REQUEST;
    let cases = [
        (2, field(capability, 24, 8), "highest"),
        (3, field(capability, 16, 8), "nominal"),
        (4, field(capability, 8, 8), "lowest nonlinear"),
        (5, field(capability, 0, 8), "lowest"),
        (7, field(request, 16, 8), "desired"),
        (8, field(request, 8, 8), "minimum"),
        (9, field(request, 0, 8), "maximum"),
        (13, (FFH, 64, 0, 4, 0xe7), "reference: MPERF"),
        (14, (FFH, 64, 0, 4, 0xe8), "delivered: APERF"),
        (16, field(MSR_CPPC_ENABLE, 0, 1), "enable"),
        (19, field(request, 24, 8), "energy preference"),
    ];
    for (index, register, what) in cases {
        assert_eq!(register_of(elements[index]), register, "{what}");
    }
    // Unimplemented registers are the null register Linux's IS_NULL_REG recognizes.
    for unsupported in [6, 10, 11, 15, 18, 20] {
        assert_eq!(
            register_of(elements[unsupported]),
            (0, 0, 0, 0, 0),
            "{unsupported}"
        );
    }
    let integer = |element: &[u8]| match element[0] {
        0x0a => element[1] as u64,
        0x0b => u16_at(element, 1) as u64,
        other => panic!("integer opcode {other:#x}"),
    };
    assert_eq!(integer(elements[21]), LOWEST_FREQUENCY_MHZ as u64);
    assert_eq!(integer(elements[22]), NOMINAL_FREQUENCY_MHZ as u64);
}
