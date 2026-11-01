//! A minimal UEFI 2.9 firmware for uemulator. The library loads this flat image at 1 MiB in
//! 64-bit mode and hands it a page describing the machine (`plans/19-uefi-firmware.md`); it
//! finds the virtio disks, publishes the UEFI tables and boots the first thing it can.
//!
//! On a hosted target only the hardware-independent modules build, for their unit tests.

#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", no_main)]
#![cfg_attr(not(target_os = "none"), allow(dead_code))]

extern crate alloc;

mod boot_menu;
mod boot_options;
mod crc32;
mod devicepath;
mod fat;
mod gpt;
mod handoff;
mod keymap;
mod keys;
mod load_option;
mod memory_map;
mod mtrr;
mod pe;
mod smbios;
mod sync;
mod text;
mod text_screen;
mod uefi;
mod variable_store;

#[cfg(target_os = "none")]
mod boot;
#[cfg(target_os = "none")]
mod console;
#[cfg(target_os = "none")]
mod cpu;
#[cfg(target_os = "none")]
mod disks;
#[cfg(target_os = "none")]
mod events;
#[cfg(target_os = "none")]
mod files;
#[cfg(target_os = "none")]
mod gop;
#[cfg(target_os = "none")]
mod images;
#[cfg(target_os = "none")]
mod kernel_boot;
#[cfg(target_os = "none")]
mod kv;
#[cfg(target_os = "none")]
mod memory;
#[cfg(target_os = "none")]
mod pci;
#[cfg(target_os = "none")]
mod processors;
#[cfg(target_os = "none")]
mod protocols;
#[cfg(target_os = "none")]
mod rng;
#[cfg(target_os = "none")]
mod runtime;
#[cfg(target_os = "none")]
mod serial;
#[cfg(target_os = "none")]
mod services;
#[cfg(target_os = "none")]
mod tables;
#[cfg(target_os = "none")]
mod time;
#[cfg(target_os = "none")]
mod variables;
#[cfg(target_os = "none")]
mod virtio_blk;
#[cfg(target_os = "none")]
mod virtio_input;
#[cfg(target_os = "none")]
mod virtio_pci;

#[cfg(not(target_os = "none"))]
fn main() {
    eprintln!("uemulator-firmware only runs on x86_64-unknown-none; see build.rs");
}

#[cfg(target_os = "none")]
mod entry {
    use core::arch::global_asm;

    pub const STACK_BYTES: usize = 256 * 1024;

    #[repr(align(16))]
    pub struct Stack(#[allow(dead_code)] [u8; STACK_BYTES]);
    #[unsafe(no_mangle)]
    pub static mut BOOT_STACK: Stack = Stack([0; STACK_BYTES]);

    // The library enters here in 64-bit mode with interrupts off and `rdi` = the handoff page.
    // The CPU starts with CR4.OSFXSR clear, and the applications we load are compiled with SSE,
    // so it is turned on before any of them runs.
    global_asm!(
        ".section .text._start,\"ax\"",
        ".global _start",
        "_start:",
        "cli",
        "cld",
        "lea rsp, [rip + BOOT_STACK + {stack_bytes}]",
        "mov rax, cr0",
        "and eax, ~(1 << 2)", // CR0.EM
        "or eax, 1 << 1",     // CR0.MP
        "mov cr0, rax",
        "mov rax, cr4",
        "or eax, (1 << 9) | (1 << 10)", // CR4.OSFXSR | OSXMMEXCPT
        "mov cr4, rax",
        "xor ebp, ebp",
        "call firmware_main",
        "2: cli",
        "hlt",
        "jmp 2b",
        stack_bytes = const STACK_BYTES,
    );
}

/// Everything after `_start`: bring up the services in dependency order, then boot.
#[cfg(target_os = "none")]
#[unsafe(no_mangle)]
extern "C" fn firmware_main(handoff: *const u8) -> ! {
    let handoff = handoff::Handoff::read(handoff);
    serial::init(handoff.serial_port);
    serial::trace!("handoff read");
    cpu::init();
    serial::trace!("idt installed");
    mtrr::init(&handoff);
    serial::trace!("mtrrs programmed");
    memory::init(&handoff);
    serial::trace!("memory ready");
    time::init(&handoff);
    services::init(&handoff);
    serial::trace!("services ready");
    boot::run(handoff.payload.as_ref())
}

#[cfg(target_os = "none")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial::println!("\nfirmware panic: {info}");
    cpu::halt_forever()
}
