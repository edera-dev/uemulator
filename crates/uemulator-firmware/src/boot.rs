//! The boot manager (UEFI 2.9 3.1): refresh the entries for the disks it finds, honour
//! `OsIndications`, count down `Timeout` with Escape opening the menu, then `BootNext` and the
//! `BootOrder` entries, then the removable-media path `\EFI\BOOT\BOOTX64.EFI` on every file
//! system. When nothing boots, the menu opens instead of the machine halting.

use alloc::string::String;
use alloc::vec::Vec;
use core::ptr;

use crate::boot_menu::{self, Choice, Hotkey, Menu, WAIT_FOREVER};
use crate::boot_options::{self, Discovered, Existing};
use crate::console::{print, println};
use crate::devicepath::{DevicePath, SUBTYPE_PCI, TYPE_HARDWARE};
use crate::handoff::Payload;
use crate::load_option::{Candidate, parse, u16_list};
use crate::tables::BOOT_SERVICES;
use crate::uefi::variable_attribute::{BOOTSERVICE_ACCESS, NON_VOLATILE, RUNTIME_ACCESS};
use crate::uefi::*;
use crate::{
    console, cpu, hypervisor_boot, images, kernel_boot, protocols, runtime, time, variables,
};

const REMOVABLE_MEDIA_PATH: &str = "\\EFI\\BOOT\\BOOTX64.EFI";
/// What `Boot####`, `BootOrder`, `Timeout` and `OsIndications` are stored with (UEFI 2.9 3.3).
const BOOT_VARIABLE_ATTRIBUTES: u32 = NON_VOLATILE | BOOTSERVICE_ACCESS | RUNTIME_ACCESS;
/// How often the countdown and the menu look for a key. Each wait halts until the PIT fires, so
/// an idle menu costs the host nothing.
const KEY_POLL_MICROSECONDS: u64 = 10_000;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;

struct Volume {
    path: DevicePath,
    interface: *mut core::ffi::c_void,
}

fn volumes() -> Vec<Volume> {
    protocols::handles_with(&guids::SIMPLE_FILE_SYSTEM)
        .into_iter()
        .filter_map(|handle| {
            let path = protocols::interface_of(handle, &guids::DEVICE_PATH)
                .and_then(|pointer| unsafe { DevicePath::from_raw(pointer.cast()) })?;
            let interface = protocols::interface_of(handle, &guids::SIMPLE_FILE_SYSTEM)?;
            Some(Volume { path, interface })
        })
        .collect()
}

/// "UEFI VirtIO disk 3, partition 1" from `PciRoot(0x0)/Pci(0x3,0x0)/HD(1,...)`, "UEFI VirtIO
/// disk 3, CD boot image" from `.../CDROM(...)`.
fn describe(volume: &DevicePath) -> String {
    let device = volume
        .nodes()
        .filter(|node| (node.kind, node.subtype) == (TYPE_HARDWARE, SUBTYPE_PCI))
        .last()
        .and_then(|node| node.payload.get(1).copied())
        .unwrap_or(0);
    if volume.is_cd_rom() {
        return alloc::format!("UEFI VirtIO disk {device}, CD boot image");
    }
    match volume.partition() {
        Some((number, ..)) => alloc::format!("UEFI VirtIO disk {device}, partition {number}"),
        None => alloc::format!("UEFI VirtIO disk {device}"),
    }
}

fn discover(volumes: &[Volume]) -> Vec<Discovered> {
    volumes
        .iter()
        .filter(|volume| crate::files::exists(volume.interface.cast(), REMOVABLE_MEDIA_PATH))
        .map(|volume| Discovered {
            description: describe(&volume.path),
            path: volume
                .path
                .append(&DevicePath::new().file_path(REMOVABLE_MEDIA_PATH)),
        })
        .collect()
}

fn read_global(name: &str) -> Option<Vec<u8>> {
    variables::read(name, &guids::GLOBAL_VARIABLE)
}

fn write_global(name: &str, data: &[u8]) {
    variables::write(
        name,
        &guids::GLOBAL_VARIABLE,
        BOOT_VARIABLE_ATTRIBUTES,
        data,
    );
}

fn boot_option(number: u16) -> Option<Candidate> {
    parse(&read_global(&alloc::format!("Boot{number:04X}"))?, number)
}

fn boot_order() -> Vec<u16> {
    read_global("BootOrder").map_or_else(Vec::new, |order| u16_list(&order))
}

/// Gives each discovered loader a `Boot####` in `BootOrder` and deletes ours for media that
/// has gone; see `boot_options`.
fn refresh_boot_options(discovered: &[Discovered]) {
    let existing: Vec<Existing> = variables::boot_option_numbers()
        .into_iter()
        .map(|number| Existing {
            number,
            option: boot_option(number),
        })
        .collect();
    let plan = boot_options::refresh(&existing, discovered, &boot_order());
    for number in plan.delete {
        variables::write(
            &alloc::format!("Boot{number:04X}"),
            &guids::GLOBAL_VARIABLE,
            0,
            &[],
        );
    }
    for (number, bytes) in plan.create {
        write_global(&alloc::format!("Boot{number:04X}"), &bytes);
    }
    if let Some(order) = plan.order {
        let bytes: Vec<u8> = order
            .iter()
            .flat_map(|number| number.to_le_bytes())
            .collect();
        write_global("BootOrder", &bytes);
    }
}

/// Consumes `OsIndications.BOOT_TO_FW_UI`: true when the OS asked for the menu.
fn take_firmware_ui_request() -> bool {
    let Some(indications) = read_global("OsIndications")
        .and_then(|data| Some(u64::from_le_bytes(data.get(..8)?.try_into().ok()?)))
    else {
        return false;
    };
    if indications & variables::OS_INDICATIONS_BOOT_TO_FW_UI == 0 {
        return false;
    }
    let rest = indications & !variables::OS_INDICATIONS_BOOT_TO_FW_UI;
    write_global("OsIndications", &rest.to_le_bytes());
    true
}

/// `BootNext` is one-shot: gone before it is tried, so a bad entry cannot trap the machine.
fn take_boot_next() -> Option<Candidate> {
    let next = read_global("BootNext").filter(|data| data.len() == 2)?;
    variables::write("BootNext", &guids::GLOBAL_VARIABLE, 0, &[]);
    boot_option(u16::from_le_bytes([next[0], next[1]]))
}

/// `Timeout` in seconds; absent means boot at once, so machines that never set it (the web
/// demo, scripted boots) do not slow down.
fn timeout() -> u16 {
    read_global("Timeout")
        .filter(|data| data.len() == 2)
        .map_or(0, |data| u16::from_le_bytes([data[0], data[1]]))
}

/// Waits for a key, polling until `deadline_ns` (never, for `None`).
fn wait_key(deadline_ns: Option<u64>) -> Option<InputKey> {
    loop {
        if let Some(key) = console::read_key() {
            return Some(key);
        }
        if deadline_ns.is_some_and(|deadline| time::now_ns() >= deadline) {
            return None;
        }
        time::stall(KEY_POLL_MICROSECONDS);
    }
}

/// The countdown before the default boot. With no timeout a key typed ahead (Escape held
/// through the reset) still opens the menu.
fn countdown(seconds: u16) -> Hotkey {
    if seconds == WAIT_FOREVER {
        return Hotkey::Menu;
    }
    if seconds == 0 {
        return console::read_key().map_or(Hotkey::Ignored, |key| boot_menu::hotkey(&key));
    }
    let mut result = Hotkey::Ignored;
    'seconds: for left in (1..=seconds).rev() {
        print!("{}", boot_menu::countdown_line(left));
        let second_ends = time::now_ns() + NANOSECONDS_PER_SECOND;
        while let Some(key) = wait_key(Some(second_ends)) {
            match boot_menu::hotkey(&key) {
                Hotkey::Ignored => {}
                chosen => {
                    result = chosen;
                    break 'seconds;
                }
            }
        }
    }
    println!();
    result
}

/// The full path to load: short forms starting at a partition expand to that partition.
fn expanded(path: &DevicePath, volumes: &[Volume]) -> DevicePath {
    path.expand_hard_drive(volumes.iter().map(|volume| &volume.path))
        .unwrap_or_else(|| path.clone())
}

fn try_boot(candidate: &Candidate, volumes: &[Volume]) -> Status {
    let boot = unsafe { BOOT_SERVICES.get().assume_init_ref() };
    let path = expanded(&candidate.path, volumes);
    println!("Booting {} ({})", candidate.description, path.text());
    crate::tpm::before_boot_option();
    let mut path_bytes = path.as_bytes().to_vec();
    let mut image: Handle = ptr::null_mut();
    let status = (boot.load_image)(
        1,
        ptr::null_mut(),
        path_bytes.as_mut_ptr().cast(),
        ptr::null_mut(),
        0,
        &mut image,
    );
    let status = if status != SUCCESS {
        // A refused image still got a handle (UEFI 2.9 7.4).
        if status == SECURITY_VIOLATION && !image.is_null() {
            (boot.unload_image)(image);
        }
        status
    } else {
        if !candidate.load_options().is_empty() {
            images::set_load_options(image, candidate.load_options());
        }
        if let Some(number) = candidate.number {
            variables::write(
                "BootCurrent",
                &guids::GLOBAL_VARIABLE,
                BOOTSERVICE_ACCESS | RUNTIME_ACCESS,
                &number.to_le_bytes(),
            );
        }
        let status = (boot.start_image)(image, ptr::null_mut(), ptr::null_mut());
        crate::tpm::after_boot_option();
        status
    };
    println!("{} returned with status {status:#x}", candidate.description);
    status
}

/// What `BootOrder` boots unattended, then the removable-media loaders it does not cover.
fn default_candidates(discovered: &[Discovered]) -> Vec<Candidate> {
    let mut found: Vec<Candidate> = boot_order()
        .into_iter()
        .filter_map(boot_option)
        .filter(Candidate::boots_by_default)
        .collect();
    for loader in discovered {
        if !found.iter().any(|candidate| candidate.path == loader.path) {
            found.push(Candidate {
                description: loader.description.clone(),
                path: loader.path.clone(),
                options: Vec::new(),
                number: None,
                attributes: crate::load_option::ACTIVE,
            });
        }
    }
    found
}

/// The menu until Escape: entries from `BootOrder` that are not hidden, then a reset.
fn boot_menu(volumes: &[Volume]) {
    let options: Vec<Candidate> = boot_order()
        .into_iter()
        .filter_map(boot_option)
        .filter(|option| !option.hidden())
        .collect();
    let mut items: Vec<String> = options
        .iter()
        .map(|option| option.description.clone())
        .collect();
    items.push(String::from("Reset the machine"));
    let mut menu = Menu::new(items.len());
    loop {
        print!("{}", menu.render(&items));
        let Some(key) = wait_key(None) else {
            continue;
        };
        match menu.handle(&key) {
            Choice::Stay => {}
            Choice::Leave => {
                println!();
                return;
            }
            Choice::Pick(index) => match options.get(index) {
                Some(option) => {
                    println!();
                    try_boot(option, volumes);
                    println!("Press a key to return to the menu.");
                    wait_key(None);
                }
                None => runtime::reset_machine(),
            },
        }
    }
}

pub fn run(payload: Option<&Payload>) -> ! {
    println!("uemulator UEFI firmware");
    if let Some(payload) = payload {
        crate::tpm::before_boot_option();
        // A kernel the machine was given is the whole boot: failing to start it must not turn
        // into a boot from some disk the caller did not ask for.
        if let Some(hypervisor) = &payload.hypervisor {
            println!("Booting the supplied hypervisor with the kernel as dom0");
            let status = hypervisor_boot::run(payload, hypervisor);
            println!("the supplied hypervisor returned with status {status:#x}");
        } else {
            println!("Booting the supplied kernel");
            let status = kernel_boot::run(payload);
            println!("the supplied kernel returned with status {status:#x}");
        }
        cpu::halt_forever()
    }
    let volumes = volumes();
    let discovered = discover(&volumes);
    refresh_boot_options(&discovered);
    let mut show_menu = take_firmware_ui_request();
    let mut boot_next = take_boot_next();
    if !show_menu {
        show_menu = countdown(timeout()) == Hotkey::Menu;
    }
    loop {
        if show_menu {
            boot_menu(&volumes);
        }
        if let Some(next) = boot_next.take() {
            try_boot(&next, &volumes);
        }
        let candidates = default_candidates(&discovered);
        for candidate in &candidates {
            try_boot(candidate, &volumes);
        }
        if candidates.is_empty() && boot_order().is_empty() {
            println!("No bootable device.");
            cpu::halt_forever()
        }
        println!("Nothing in the boot order started; opening the boot menu.");
        show_menu = true;
    }
}
