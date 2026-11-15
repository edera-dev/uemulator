//! INVLPGB and TLBSYNC (APM vol. 3): a TLB invalidation broadcast to every processor, and the
//! wait for it to finish. Linux 6.15+ flushes remote TLBs with them instead of IPIs on Zen 3 and
//! later (`arch/x86/include/asm/tlb.h`).
//!
//! Every vCPU is invalidated before INVLPGB retires, so TLBSYNC has nothing to wait for. With a
//! host thread per vCPU the others are stopped (`Cpu::with_exclusive`) and invalidated in place;
//! time-shared vCPUs are not running, so each throws its whole TLB away before it next runs
//! (`catch_up_tlb_shootdowns`). Dropping more than asked is always allowed for a TLB, and the
//! soft TLB keeps neither PCIDs nor global bits apart, so the PCID and global filters only ever
//! widen what goes; the ASID decides between the running and the parked TLB.
//!
//! An SVM guest may run them when its VMCB enables them (0x90 bit 7, CPUID 0x8000_000A EDX bit
//! 24 TlbiCtl), else they are #UD there; the intercepts come first either way. A guest's
//! INVLPGB reaches only its own ASID, whatever EDX and rAX[2] say, and only its own linear
//! translations: never the host's, and never the nested-table leaves, which are the
//! hypervisor's to invalidate (its INVLPGB or TLB_CONTROL). The soft TLB holds guest-linear to
//! host-physical translations whole, so with nested paging the pages named are dropped together
//! with whatever nested translation they were made through.

use super::exception::Exception;
use super::svm::GuestTlbs;
use super::sys::{RAX, RCX, RDX};
use super::*;

/// ModRM bytes of `0F 01 FE` (INVLPGB) and `0F 01 FF` (TLBSYNC).
pub(crate) const MODRM_INVLPGB: u8 = 0xfe;
pub(crate) const MODRM_TLBSYNC: u8 = 0xff;

/// rAX: bits 63:12 are the first page (with `VA_VALID`); the low bits say what to match. Bit 1
/// (PCID valid, the PCID in EDX[31:16]) and bits 3-5 (include global, final translation only,
/// include nested translations) only narrow or widen what goes among translations the soft TLB
/// does not tell apart, so they are ignored: everything they could name is dropped.
const VA_VALID: u64 = 1 << 0;
const ASID_VALID: u64 = 1 << 2;
const PAGE_ADDRESS_MASK: u64 = !0xfff;
/// ECX: how many pages after the first (15:0), and whether they are 2 MiB apart (bit 31) instead
/// of 4 KiB.
const ECX_EXTRA_PAGES_MASK: u32 = 0xffff;
const ECX_TWO_MIB_STRIDE: u32 = 1 << 31;
/// EDX: the ASID (15:0); the PCID is 31:16.
const EDX_ASID_MASK: u64 = 0xffff;
/// The largest ECX[15:0] accepted, reported in CPUID 0x8000_0008 EDX[15:0]: eight pages per
/// instruction, Zen 3 through Zen 5's value. Linux loops over bigger ranges itself.
pub(crate) const INVLPGB_MAX_EXTRA_PAGES: u32 = 7;
const HOST_ASID: u64 = 0;
const SMALL_PAGE: u64 = 4096;

/// What one INVLPGB drops, on any vCPU.
#[derive(Clone, Copy)]
struct Shootdown {
    /// First 4 KiB page and page count; `None` for every address, which is also what 2 MiB
    /// pages become: the soft TLB caches 4 KiB pages, and 512 invlpgs per page cost more than a
    /// flush.
    pages: Option<(u64, u64)>,
    /// The host's translations, and which guests'.
    host: bool,
    guests: GuestTlbs,
    /// The cached nested-table leaves too: the host may have changed the nested tables.
    nested: bool,
}

impl Cpu {
    /// Whether INVLPGB's operands are out of range, which is #GP (and the INVLPGB_ILLEGAL
    /// intercept in a guest).
    pub(crate) fn invlpgb_operands_illegal(&self) -> bool {
        self.gpr[RCX] as u32 & ECX_EXTRA_PAGES_MASK > INVLPGB_MAX_EXTRA_PAGES
    }

    /// The checks INVLPGB and TLBSYNC share. Real and virtual-8086 mode have no use for them
    /// and get #UD, CPL > 0 gets #GP(0). In a guest (whose intercepts `svm_intercept` took
    /// first) they are #UD unless the VMCB enables them.
    fn broadcast_tlb_checks(&self) -> ExecResult<()> {
        let guest_disabled = self
            .svm
            .guest
            .as_ref()
            .is_some_and(|guest| !guest.invlpgb_enabled);
        if self.cr0 & CR0_PE == 0 || self.eflags & flags::VM != 0 || guest_disabled {
            return Err(Exception::invalid_opcode());
        }
        self.need_cpl0()
    }

    /// 0F 01 FE.
    pub(super) fn invlpgb(&mut self, next: u64) -> ExecResult<()> {
        self.broadcast_tlb_checks()?;
        if self.invlpgb_operands_illegal() {
            return Err(Exception::general_protection(0));
        }
        let flags = self.gpr[RAX];
        let ecx = self.gpr[RCX] as u32;
        let pages = (flags & VA_VALID != 0 && ecx & ECX_TWO_MIB_STRIDE == 0).then(|| {
            let first = if self.efer & EFER_LMA != 0 {
                flags & PAGE_ADDRESS_MASK
            } else {
                flags & PAGE_ADDRESS_MASK & 0xffff_ffff
            };
            (first, (ecx & ECX_EXTRA_PAGES_MASK) as u64 + 1)
        });
        let asid = self.gpr[RDX] & EDX_ASID_MASK;
        let shootdown = match self.svm.guest.as_ref().map(|guest| guest.asid) {
            Some(own) => Shootdown {
                pages,
                host: false,
                guests: GuestTlbs::Asid(own),
                nested: false,
            },
            None if flags & ASID_VALID == 0 => Shootdown {
                pages,
                host: true,
                guests: GuestTlbs::All,
                nested: true,
            },
            None if asid == HOST_ASID => Shootdown {
                pages,
                host: true,
                guests: GuestTlbs::None,
                nested: false,
            },
            None => Shootdown {
                pages,
                host: false,
                guests: GuestTlbs::Asid(asid as u32),
                nested: true,
            },
        };
        self.apply_shootdown(&shootdown);
        if self.threads.is_some() {
            self.with_exclusive(|cpu| {
                cpu.for_each_other_cpu(|other| other.apply_shootdown(&shootdown))
            });
        } else {
            self.tlb_shootdown_epoch = self.tlb_shootdown_epoch.wrapping_add(1);
            self.tlb_shootdown_seen = self.tlb_shootdown_epoch;
        }
        self.rip = next;
        Ok(())
    }

    /// 0F 01 FF: every INVLPGB has finished by the time it retires, so there is nothing to wait
    /// for.
    pub(super) fn tlbsync(&mut self, next: u64) -> ExecResult<()> {
        self.broadcast_tlb_checks()?;
        self.rip = next;
        Ok(())
    }

    fn apply_shootdown(&mut self, shootdown: &Shootdown) {
        let running_hit = match self.svm.guest.as_ref() {
            Some(guest) => shootdown.guests.covers(guest.asid),
            None => shootdown.host,
        };
        if running_hit {
            match shootdown.pages {
                Some((first, count)) => {
                    for page in 0..count {
                        self.invlpg(first.wrapping_add(page * SMALL_PAGE));
                    }
                }
                None => self.flush_tlb(),
            }
        }
        self.drop_parked_tlb(shootdown.host, shootdown.guests, shootdown.nested);
    }

    /// A time-shared vCPU about to run: if INVLPGB ran on another vCPU since it last ran, its
    /// translations go, all of them (`swap_shared` carries the machine's count).
    pub(super) fn catch_up_tlb_shootdowns(&mut self) {
        if self.tlb_shootdown_seen == self.tlb_shootdown_epoch {
            return;
        }
        self.tlb_shootdown_seen = self.tlb_shootdown_epoch;
        self.flush_tlb();
        self.drop_parked_tlb(true, GuestTlbs::All, true);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        GDT_BASE, PTE_PRESENT_WRITABLE_USER, PageTables, enter_user_mode, write_pte,
    };
    use super::*;
    use crate::testkit::{CODE_BASE, Harness, RBX, SCRATCH, STACK_TOP, Stop};

    /// Its page number's low bits differ from those of the code and stack pages, so the soft TLB
    /// (direct mapped) keeps its translation while the snippet runs.
    const VIRTUAL: u64 = 0x0000_2000_0003_5000;
    const OLD_PAGES: u64 = SCRATCH + 0x4000;
    const NEW_PAGES: u64 = SCRATCH + 0x8000;
    const OLD_VALUE: u32 = 0x11;
    const NEW_VALUE: u32 = 0x22;
    const ASID_ONE: u64 = 1;
    // invlpgb; tlbsync; mov ecx, [rbx]; hlt
    const INVALIDATE_THEN_LOAD: [u8; 9] = [0x0f, 0x01, 0xfe, 0x0f, 0x01, 0xff, 0x8b, 0x0b, 0xf4];
    // mov ecx, [rbx]; hlt
    const LOAD: [u8; 3] = [0x8b, 0x0b, 0xf4];
    // invlpgb; tlbsync; mov ecx, [rbx]; mov edx, [rbx + 0x1000]; hlt
    const INVALIDATE_THEN_LOAD_TWO: [u8; 15] = [
        0x0f, 0x01, 0xfe, 0x0f, 0x01, 0xff, 0x8b, 0x0b, 0x8b, 0x93, 0x00, 0x10, 0x00, 0x00, 0xf4,
    ];

    /// Maps two pages at `VIRTUAL` to `OLD_PAGES`, caches their translations, then points the
    /// page table entries at `NEW_PAGES` without telling the TLB. rbx is `VIRTUAL`.
    fn stale(code: &[u8], jit: bool) -> Harness {
        let mut h = Harness::new(code);
        if jit {
            h.enable_jit();
        }
        let mut tables = PageTables::new(&mut h.cpu);
        let flags = PTE_PRESENT_WRITABLE_USER;
        let code_page = CODE_BASE & !0xfff;
        for page in [
            code_page,
            code_page + 0x1000,
            STACK_TOP - 0x1000,
            GDT_BASE & !0xfff,
        ] {
            tables.map_4k(&mut h.cpu, page, page, flags);
        }
        let mut entries = [0; 2];
        for (index, entry) in entries.iter_mut().enumerate() {
            let offset = index as u64 * 0x1000;
            *entry = tables.map_4k(&mut h.cpu, VIRTUAL + offset, OLD_PAGES + offset, flags);
            h.write(OLD_PAGES + offset, &OLD_VALUE.to_le_bytes());
            h.write(NEW_PAGES + offset, &NEW_VALUE.to_le_bytes());
            assert_eq!(h.cpu.read(VIRTUAL + offset, 4).unwrap(), OLD_VALUE as u64);
        }
        for (index, entry) in entries.into_iter().enumerate() {
            write_pte(
                &mut h.cpu,
                entry,
                (NEW_PAGES + index as u64 * 0x1000) | flags,
            );
        }
        h.cpu.gpr[RBX] = VIRTUAL;
        h.cpu.gpr[RAX] = VIRTUAL | VA_VALID;
        h
    }

    fn on_both_engines(test: impl Fn(bool)) {
        test(false);
        test(true);
    }

    #[test]
    fn without_invlpgb_the_stale_translation_stays() {
        on_both_engines(|jit| {
            let mut h = stale(&LOAD, jit);
            h.run_to_halt();
            assert_eq!(h.cpu.gpr[RCX], OLD_VALUE as u64, "jit {jit}");
        });
    }

    #[test]
    fn invlpgb_drops_the_page_it_names() {
        on_both_engines(|jit| {
            let mut h = stale(&INVALIDATE_THEN_LOAD, jit);
            h.run_to_halt();
            assert_eq!(h.cpu.gpr[RCX], NEW_VALUE as u64, "jit {jit}");
        });
    }

    #[test]
    fn invlpgb_drops_as_many_more_pages_as_ecx_says() {
        on_both_engines(|jit| {
            let mut h = stale(&INVALIDATE_THEN_LOAD_TWO, jit);
            h.cpu.gpr[RCX] = 1;
            h.run_to_halt();
            assert_eq!(h.cpu.gpr[RCX], NEW_VALUE as u64, "jit {jit}");
            assert_eq!(h.cpu.gpr[RDX], NEW_VALUE as u64, "jit {jit}");
            let mut h = stale(&INVALIDATE_THEN_LOAD_TWO, jit);
            h.run_to_halt();
            assert_eq!(h.cpu.gpr[RDX], OLD_VALUE as u64, "jit {jit}: one page only");
        });
    }

    #[test]
    fn invlpgb_without_an_address_drops_everything() {
        let mut h = stale(&INVALIDATE_THEN_LOAD_TWO, false);
        h.cpu.gpr[RAX] = 0;
        h.run_to_halt();
        assert_eq!(h.cpu.gpr[RDX], NEW_VALUE as u64);
    }

    #[test]
    fn invlpgb_for_a_guest_asid_leaves_the_hosts_translations() {
        let mut h = stale(&INVALIDATE_THEN_LOAD, false);
        h.cpu.gpr[RAX] = VIRTUAL | VA_VALID | ASID_VALID;
        h.cpu.gpr[RDX] = ASID_ONE;
        h.run_to_halt();
        assert_eq!(h.cpu.gpr[RCX], OLD_VALUE as u64);
        let mut h = stale(&INVALIDATE_THEN_LOAD, false);
        h.cpu.gpr[RAX] = VIRTUAL | VA_VALID | ASID_VALID;
        h.cpu.gpr[RDX] = HOST_ASID;
        h.run_to_halt();
        assert_eq!(h.cpu.gpr[RCX], NEW_VALUE as u64);
    }

    #[test]
    fn a_time_shared_vcpu_catches_up_before_it_runs_again() {
        let mut h = stale(&LOAD, false);
        h.cpu.tlb_shootdown_epoch += 1;
        h.cpu.catch_up_tlb_shootdowns();
        h.run_to_halt();
        assert_eq!(h.cpu.gpr[RCX], NEW_VALUE as u64);
        assert_eq!(h.cpu.tlb_shootdown_seen, h.cpu.tlb_shootdown_epoch);
    }

    #[test]
    fn too_many_pages_and_cpl_3_are_gp_and_real_mode_is_ud() {
        const GP0: Stop = Stop::Fault {
            vector: 13,
            error_code: Some(0),
            software: false,
        };
        let mut h = Harness::new(&INVALIDATE_THEN_LOAD);
        h.cpu.gpr[RCX] = INVLPGB_MAX_EXTRA_PAGES as u64 + 1;
        assert_eq!(h.run(10), GP0);
        for code in [&INVALIDATE_THEN_LOAD[..], &INVALIDATE_THEN_LOAD[3..]] {
            let mut h = stale(code, false);
            enter_user_mode(&mut h);
            assert_eq!(h.run(10), GP0, "{code:02x?}");
        }
        // invlpgb; hlt
        let mut h = Harness::real(&[0x0f, 0x01, 0xfe, 0xf4]);
        assert_eq!(
            h.run(10),
            Stop::Fault {
                vector: 6,
                error_code: None,
                software: false
            }
        );
    }
}
