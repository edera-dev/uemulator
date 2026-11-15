//! Builders shared by the unit tests of `cpu`: guest page tables, GDT/IDT/TSS entries and
//! frame readers, all placed in the harness CPU's RAM. The constants here restate the
//! architectural encodings on purpose, so a test never checks the emulator against itself.

use std::sync::{Arc, Mutex};

use super::exception::Exception;
use super::{Cpu, ExecResult, SEG_CS, SEG_SS};
use crate::testkit::Harness;

#[path = "test_support_paging.rs"]
mod paging;
pub(crate) use paging::*;

pub(crate) const PTE_P: u64 = 1 << 0;
pub(crate) const PTE_W: u64 = 1 << 1;
pub(crate) const PTE_U: u64 = 1 << 2;
pub(crate) const PTE_A: u64 = 1 << 5;
pub(crate) const PTE_D: u64 = 1 << 6;
pub(crate) const PTE_PS: u64 = 1 << 7;
pub(crate) const PTE_NX: u64 = 1 << 63;
pub(crate) const PTE_PRESENT_WRITABLE_USER: u64 = PTE_P | PTE_W | PTE_U;
pub(crate) const PTE_ADDRESS_BITS: u64 = 0x000f_ffff_ffff_f000;

pub(crate) const PF_P: u32 = 1 << 0;
pub(crate) const PF_W: u32 = 1 << 1;
pub(crate) const PF_U: u32 = 1 << 2;
pub(crate) const PF_RSVD: u32 = 1 << 3;
pub(crate) const PF_ID: u32 = 1 << 4;

pub(crate) const PAGE: u64 = 4096;
pub(crate) const TWO_MIB: u64 = 2 << 20;
pub(crate) const ONE_GIB: u64 = 1 << 30;

/// Free RAM the table allocator hands pages out of (below `testkit::SCRATCH`).
pub(crate) const TABLE_POOL: u64 = 0x20_0000;
pub(crate) const TABLE_POOL_END: u64 = 0x40_0000;

pub(crate) fn fault_of<T: std::fmt::Debug>(result: ExecResult<T>) -> (u8, Option<u32>) {
    match result {
        Err(Exception::Fault {
            vector, error_code, ..
        }) => (vector, error_code),
        other => panic!("expected a fault, got {other:?}"),
    }
}

/// Collects the messages the CPU sends to its log callback.
pub(crate) fn capture_log(cpu: &mut Cpu) -> Arc<Mutex<Vec<String>>> {
    let messages = Arc::new(Mutex::new(Vec::new()));
    let sink = messages.clone();
    cpu.log = Some(Arc::new(move |message: &str| {
        sink.lock().unwrap().push(message.to_string())
    }));
    messages
}

/// IA-32e guest page tables built in harness RAM, 4-level (`new`) or 5-level (`new_five_level`,
/// which also sets CR4.LA57). `new` switches the CPU to an empty top-level table.
pub(crate) struct PageTables {
    pub pml4: u64,
    next_free: u64,
    levels: u32,
}

/// CR4.LA57, restated: 5-level paging with 57-bit linear addresses.
pub(crate) const CR4_LA57_BIT: u64 = 1 << 12;

impl PageTables {
    pub fn new(cpu: &mut Cpu) -> Self {
        Self::with_levels(cpu, 4)
    }

    /// `pml4` is then the PML5.
    pub fn new_five_level(cpu: &mut Cpu) -> Self {
        cpu.cr4 |= CR4_LA57_BIT;
        Self::with_levels(cpu, 5)
    }

    fn with_levels(cpu: &mut Cpu, levels: u32) -> Self {
        let mut tables = PageTables {
            pml4: 0,
            next_free: TABLE_POOL,
            levels,
        };
        tables.pml4 = tables.alloc(cpu);
        cpu.cr3 = tables.pml4;
        cpu.flush_tlb();
        tables
    }

    pub fn alloc(&mut self, cpu: &mut Cpu) -> u64 {
        let page = self.next_free;
        assert!(page < TABLE_POOL_END, "table pool exhausted");
        self.next_free += PAGE;
        cpu.ram[page as usize..(page + PAGE) as usize].fill(0);
        page
    }

    /// Maps `linear` to `phys`. `flags` lists the entry flags top down: one per level makes a
    /// 4 KiB page, one fewer a 2 MiB leaf (PS is added), two fewer a 1 GiB leaf. Intermediate
    /// tables that already exist keep their entries (and flags); only missing ones are created.
    /// Returns the physical address of the leaf entry.
    pub fn map(&mut self, cpu: &mut Cpu, linear: u64, phys: u64, flags: &[u64]) -> u64 {
        let levels = self.levels as usize;
        assert!((levels - 2..=levels).contains(&flags.len()));
        let mut table = self.pml4;
        for (i, &entry_flags) in flags.iter().enumerate() {
            let level = self.levels - 1 - i as u32;
            let index = (linear >> (12 + 9 * level)) & 0x1ff;
            let entry_address = table + index * 8;
            if i == flags.len() - 1 {
                let page_size = if level > 0 { PTE_PS } else { 0 };
                write_pte(cpu, entry_address, phys | entry_flags | page_size);
                return entry_address;
            }
            let existing = read_pte(cpu, entry_address);
            table = if existing & PTE_P != 0 {
                existing & PTE_ADDRESS_BITS
            } else {
                let next = self.alloc(cpu);
                write_pte(cpu, entry_address, next | entry_flags);
                next
            };
        }
        unreachable!()
    }

    /// 4 KiB mapping with the same flags at every level.
    pub fn map_4k(&mut self, cpu: &mut Cpu, linear: u64, phys: u64, flags: u64) -> u64 {
        let all_levels = vec![flags; self.levels as usize];
        self.map(cpu, linear, phys, &all_levels)
    }

    /// Address of the entry for `linear` at `level` (3 = PML4, 4 = PML5 .. 0 = PT), if the path
    /// exists.
    pub fn entry_address(&self, cpu: &Cpu, linear: u64, level: u32) -> u64 {
        let mut table = self.pml4;
        for walk_level in (level..self.levels).rev() {
            let index = (linear >> (12 + 9 * walk_level)) & 0x1ff;
            let entry_address = table + index * 8;
            if walk_level == level {
                return entry_address;
            }
            table = read_pte(cpu, entry_address) & PTE_ADDRESS_BITS;
        }
        unreachable!()
    }
}

pub(crate) fn read_pte(cpu: &Cpu, address: u64) -> u64 {
    u64::from_le_bytes(cpu.ram[address as usize..][..8].try_into().unwrap())
}

pub(crate) fn write_pte(cpu: &mut Cpu, address: u64, value: u64) {
    cpu.ram[address as usize..][..8].copy_from_slice(&value.to_le_bytes());
}

/// Makes the 2 MiB region at `linear` of the boot identity map not present (it is mapped with
/// 2 MiB pages), so accesses there page-fault. Flushes the TLB.
pub(crate) fn unmap_boot_identity_2m(h: &mut Harness, linear: u64) {
    let pml4 = h.cpu.cr3 & PTE_ADDRESS_BITS;
    let pdpt = read_pte(&h.cpu, pml4) & PTE_ADDRESS_BITS;
    let pd = read_pte(&h.cpu, pdpt + ((linear >> 30) & 0x1ff) * 8) & PTE_ADDRESS_BITS;
    write_pte(&mut h.cpu, pd + ((linear >> 21) & 0x1ff) * 8, 0);
    h.cpu.flush_tlb();
}

pub(crate) const GDT_BASE: u64 = 0x6000;
/// The boot GDT has the null descriptor at 0 and 64-bit code/data at these selectors.
pub(crate) const KERNEL_CS: u16 = 0x10;
pub(crate) const KERNEL_DS: u16 = 0x18;
pub(crate) const USER_CS: u16 = 0x23;
pub(crate) const USER_DS: u16 = 0x2b;
pub(crate) const CONFORMING_CS_DPL0: u16 = 0x30;
pub(crate) const CONFORMING_CS_DPL3: u16 = 0x38;
pub(crate) const DPL1_CS: u16 = 0x41;
pub(crate) const DPL3_NONCONFORMING_CS: u16 = 0x48;
pub(crate) const NOT_PRESENT_CS: u16 = 0x50;
pub(crate) const COMPAT_CS: u16 = 0x58;
pub(crate) const BIG_LONG_CS: u16 = 0x60;
pub(crate) const DATA_SELECTOR_FOR_GATE: u16 = 0x68;
pub(crate) const TSS_SELECTOR: u16 = 0x70;

/// Descriptor access bytes (P, DPL, S, type).
const ACCESS_CODE_READABLE: u8 = 0x9a;
const ACCESS_CODE_CONFORMING_READABLE: u8 = 0x9e;
const ACCESS_DATA_WRITABLE: u8 = 0x92;
const ACCESS_PRESENT: u8 = 0x80;
const ACCESS_DPL_SHIFT: u8 = 5;
const FLAG_LONG: u8 = 0x2;
const FLAG_DEFAULT_BIG: u8 = 0x4;
const FLAG_GRANULARITY: u8 = 0x8;

/// Encodes a descriptor: `access` is the byte at offset 5, `flags` the high nibble of offset 6.
pub(crate) fn descriptor(base: u32, limit: u32, access: u8, flags: u8) -> u64 {
    (limit as u64 & 0xffff)
        | ((base as u64 & 0xff_ffff) << 16)
        | ((access as u64) << 40)
        | (((limit as u64 >> 16) & 0xf) << 48)
        | (((flags & 0xf) as u64) << 52)
        | ((base as u64 >> 24) << 56)
}

fn with_dpl(access: u8, dpl: u8) -> u8 {
    (access & !(3 << ACCESS_DPL_SHIFT)) | (dpl << ACCESS_DPL_SHIFT)
}

pub(crate) fn set_gdt_entry(h: &mut Harness, selector: u16, value: u64) {
    let offset = (selector & 0xfff8) as u64;
    h.write_u64(GDT_BASE + offset, value);
    if h.cpu.gdtr.limit < (offset + 7) as u16 {
        h.cpu.gdtr.limit = (offset + 7) as u16;
    }
}

/// Fills the GDT with the descriptors the tests refer to by the selector constants above.
pub(crate) fn install_test_gdt(h: &mut Harness) {
    let long_code = |dpl| {
        descriptor(
            0,
            0xfffff,
            with_dpl(ACCESS_CODE_READABLE, dpl),
            FLAG_LONG | FLAG_GRANULARITY,
        )
    };
    set_gdt_entry(h, USER_CS, long_code(3));
    set_gdt_entry(
        h,
        USER_DS,
        descriptor(
            0,
            0xfffff,
            with_dpl(ACCESS_DATA_WRITABLE, 3),
            FLAG_DEFAULT_BIG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(
        h,
        CONFORMING_CS_DPL0,
        descriptor(
            0,
            0xfffff,
            with_dpl(ACCESS_CODE_CONFORMING_READABLE, 0),
            FLAG_LONG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(
        h,
        CONFORMING_CS_DPL3,
        descriptor(
            0,
            0xfffff,
            with_dpl(ACCESS_CODE_CONFORMING_READABLE, 3),
            FLAG_LONG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(h, DPL1_CS, long_code(1));
    set_gdt_entry(h, DPL3_NONCONFORMING_CS, long_code(3));
    set_gdt_entry(
        h,
        NOT_PRESENT_CS,
        descriptor(
            0,
            0xfffff,
            ACCESS_CODE_READABLE & !ACCESS_PRESENT,
            FLAG_LONG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(
        h,
        COMPAT_CS,
        descriptor(
            0,
            0xfffff,
            ACCESS_CODE_READABLE,
            FLAG_DEFAULT_BIG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(
        h,
        BIG_LONG_CS,
        descriptor(
            0,
            0xfffff,
            ACCESS_CODE_READABLE,
            FLAG_LONG | FLAG_DEFAULT_BIG | FLAG_GRANULARITY,
        ),
    );
    set_gdt_entry(
        h,
        DATA_SELECTOR_FOR_GATE,
        descriptor(
            0,
            0xfffff,
            ACCESS_DATA_WRITABLE,
            FLAG_DEFAULT_BIG | FLAG_GRANULARITY,
        ),
    );
}

pub(crate) const IDT_BASE: u64 = 0x10_0000;
pub(crate) const TSS_BASE: u64 = 0x11_0000;
pub(crate) const TSS_LIMIT: u32 = 0x67;
pub(crate) const TYPE_INTERRUPT_GATE: u8 = 0xe;
pub(crate) const TYPE_TRAP_GATE: u8 = 0xf;

/// A 64-bit IDT gate.
#[derive(Clone, Copy)]
pub(crate) struct Gate {
    pub handler: u64,
    pub selector: u16,
    pub ist: u8,
    pub gate_type: u8,
    pub dpl: u8,
    pub present: bool,
}

impl Gate {
    pub fn interrupt(handler: u64) -> Gate {
        Gate {
            handler,
            selector: KERNEL_CS,
            ist: 0,
            gate_type: TYPE_INTERRUPT_GATE,
            dpl: 0,
            present: true,
        }
    }

    pub fn trap(handler: u64) -> Gate {
        Gate {
            gate_type: TYPE_TRAP_GATE,
            ..Gate::interrupt(handler)
        }
    }

    pub fn encode(&self) -> [u8; 16] {
        let attributes = self.gate_type | (self.dpl << 5) | if self.present { 0x80 } else { 0 };
        let mut bytes = [0u8; 16];
        bytes[0..2].copy_from_slice(&(self.handler as u16).to_le_bytes());
        bytes[2..4].copy_from_slice(&self.selector.to_le_bytes());
        bytes[4] = self.ist;
        bytes[5] = attributes;
        bytes[6..8].copy_from_slice(&((self.handler >> 16) as u16).to_le_bytes());
        bytes[8..12].copy_from_slice(&((self.handler >> 32) as u32).to_le_bytes());
        bytes
    }
}

/// Distinct canonical handler address per vector, so a test can tell which gate was used.
pub(crate) fn handler_for(vector: u8) -> u64 {
    0xffff_8000_c000_0000 + vector as u64 * 0x100
}

pub(crate) fn set_gate(h: &mut Harness, vector: u8, gate: Gate) {
    h.write(IDT_BASE + vector as u64 * 16, &gate.encode());
}

/// Points the IDT at `IDT_BASE`, covers all 256 vectors, and installs a present interrupt gate
/// with `handler_for(vector)` for every vector. Also installs the test GDT.
pub(crate) fn install_idt(h: &mut Harness) {
    install_test_gdt(h);
    h.cpu.idtr.base = IDT_BASE;
    h.cpu.idtr.limit = 256 * 16 - 1;
    for vector in 0..=255u8 {
        set_gate(h, vector, Gate::interrupt(handler_for(vector)));
    }
}

/// 64-bit TSS with the given RSP0..2 and IST1..7, loaded into TR.
pub(crate) fn install_tss(h: &mut Harness, rsp: [u64; 3], ist: [u64; 7]) {
    for (i, value) in rsp.iter().enumerate() {
        h.write_u64(TSS_BASE + 4 + 8 * i as u64, *value);
    }
    for (i, value) in ist.iter().enumerate() {
        h.write_u64(TSS_BASE + 36 + 8 * i as u64, *value);
    }
    h.cpu.tr.sel = TSS_SELECTOR;
    h.cpu.tr.base = TSS_BASE;
    h.cpu.tr.limit = TSS_LIMIT;
}

/// What an interrupt or exception pushed, lowest address (last push) first.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub error_code: Option<u64>,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

pub(crate) fn read_frame(h: &Harness, stack_pointer: u64, has_error_code: bool) -> Frame {
    let mut address = stack_pointer;
    let mut next = || {
        let value = h.read_u64(address);
        address += 8;
        value
    };
    let error_code = has_error_code.then(&mut next);
    Frame {
        error_code,
        rip: next(),
        cs: next(),
        rflags: next(),
        rsp: next(),
        ss: next(),
    }
}

/// Makes CS and SS user-mode (CPL 3) segments and switches the CPU to CPL 3.
pub(crate) fn enter_user_mode(h: &mut Harness) {
    install_test_gdt(h);
    h.cpu.seg[SEG_CS] = h.cpu.load_desc_seg(USER_CS, true).unwrap();
    h.cpu.seg[SEG_SS] = h.cpu.load_desc_seg(USER_DS, true).unwrap();
    h.cpu.set_cpl(3);
}
