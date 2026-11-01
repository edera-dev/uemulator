//! Keeping `Boot####` and `BootOrder` in step with the disks, as EDK2's BDS does: every volume
//! with a removable-media loader gets an entry of its own, marked as made by the firmware, and
//! the marked entries whose volume is gone are deleted. An OS then sees them in efibootmgr and
//! can reorder them, and the boot menu lists them.

use alloc::string::String;
use alloc::vec::Vec;

use crate::devicepath::DevicePath;
use crate::load_option::{self, ACTIVE, AUTO_CREATED, Candidate};

/// A `Boot####` variable as found: `None` when it does not parse (inactive or malformed), which
/// still takes its number.
pub struct Existing {
    pub number: u16,
    pub option: Option<Candidate>,
}

/// A loader found on a volume.
pub struct Discovered {
    pub description: String,
    pub path: DevicePath,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// New `Boot####` variables, number and `EFI_LOAD_OPTION` bytes.
    pub create: Vec<(u16, Vec<u8>)>,
    pub delete: Vec<u16>,
    /// The new `BootOrder`, when it changes.
    pub order: Option<Vec<u16>>,
}

fn lowest_unused(taken: &[u16]) -> Option<u16> {
    (0..=u16::MAX).find(|number| !taken.contains(number))
}

/// What to write so the variables match `discovered`. An entry the OS wrote is never touched;
/// a discovered loader that some entry already boots gets no second one.
pub fn refresh(existing: &[Existing], discovered: &[Discovered], order: &[u16]) -> Plan {
    let mut plan = Plan::default();
    let mut taken: Vec<u16> = existing.iter().map(|entry| entry.number).collect();
    for entry in existing {
        if let Some(option) = &entry.option
            && option.auto_created()
            && !discovered.iter().any(|found| found.path == option.path)
        {
            plan.delete.push(entry.number);
        }
    }
    let mut new_order: Vec<u16> = order
        .iter()
        .copied()
        .filter(|number| !plan.delete.contains(number))
        .collect();
    // Ours, but dropped from the order (by an older firmware, or a crash between the writes).
    for entry in existing {
        if entry.option.as_ref().is_some_and(Candidate::auto_created)
            && !plan.delete.contains(&entry.number)
            && !new_order.contains(&entry.number)
        {
            new_order.push(entry.number);
        }
    }
    for found in discovered {
        let booted = existing.iter().any(|entry| {
            !plan.delete.contains(&entry.number)
                && entry
                    .option
                    .as_ref()
                    .is_some_and(|option| option.path == found.path)
        });
        if booted {
            continue;
        }
        let Some(number) = lowest_unused(&taken) else {
            break;
        };
        taken.push(number);
        let bytes = load_option::build(
            ACTIVE,
            &found.description,
            &found.path,
            &AUTO_CREATED.to_bytes(),
        );
        plan.create.push((number, bytes));
        new_order.push(number);
    }
    if new_order != order {
        plan.order = Some(new_order);
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_option::parse;

    const LOADER: &str = "\\EFI\\BOOT\\BOOTX64.EFI";

    fn volume(device: u8) -> DevicePath {
        DevicePath::new()
            .acpi_pci_root()
            .pci(device, 0)
            .file_path(LOADER)
    }

    fn found(device: u8) -> Discovered {
        Discovered {
            description: alloc::format!("disk {device}"),
            path: volume(device),
        }
    }

    fn existing(number: u16, bytes: &[u8]) -> Existing {
        Existing {
            number,
            option: parse(bytes, number),
        }
    }

    fn ours(number: u16, device: u8) -> Existing {
        existing(
            number,
            &load_option::build(ACTIVE, "x", &volume(device), &AUTO_CREATED.to_bytes()),
        )
    }

    fn theirs(number: u16, path: &DevicePath) -> Existing {
        existing(number, &load_option::build(ACTIVE, "debian", path, b""))
    }

    #[test]
    fn a_new_volume_gets_the_lowest_free_number_at_the_end_of_the_order() {
        let shim = DevicePath::new().file_path("\\EFI\\debian\\shimx64.efi");
        let plan = refresh(&[theirs(0, &shim)], &[found(1)], &[0]);
        assert_eq!(plan.delete, []);
        assert_eq!(plan.create.len(), 1);
        let (number, bytes) = &plan.create[0];
        assert_eq!(*number, 1);
        let option = parse(bytes, 1).unwrap();
        assert!(option.auto_created());
        assert_eq!(option.path, volume(1));
        assert_eq!(option.description, "disk 1");
        assert_eq!(plan.order, Some(alloc::vec![0, 1]));
    }

    #[test]
    fn nothing_changes_when_the_entries_already_match() {
        let plan = refresh(&[ours(0, 1)], &[found(1)], &[0]);
        assert_eq!(plan, Plan::default());
    }

    #[test]
    fn our_entry_for_a_missing_volume_goes_but_the_os_entries_stay() {
        let gone = volume(5);
        let plan = refresh(&[ours(0, 1), theirs(1, &gone)], &[], &[1, 0]);
        assert_eq!(plan.delete, [0]);
        assert_eq!(plan.create, []);
        assert_eq!(plan.order, Some(alloc::vec![1]));
    }

    #[test]
    fn a_loader_an_os_entry_boots_gets_no_second_entry() {
        let plan = refresh(&[theirs(0, &volume(1))], &[found(1)], &[0]);
        assert_eq!(plan, Plan::default());
    }

    #[test]
    fn numbers_that_do_not_parse_are_still_taken() {
        let broken = Existing {
            number: 0,
            option: None,
        };
        let plan = refresh(&[broken], &[found(1)], &[]);
        assert_eq!(plan.create[0].0, 1);
    }

    #[test]
    fn our_entry_missing_from_the_order_is_put_back() {
        let plan = refresh(&[ours(3, 1)], &[found(1)], &[]);
        assert_eq!(plan.create, []);
        assert_eq!(plan.order, Some(alloc::vec![3]));
    }
}
