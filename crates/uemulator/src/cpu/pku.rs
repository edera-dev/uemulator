//! Memory protection keys (CR4.PKE; APM Vol. 2 5.6.7, SDM Vol. 3A 4.6.2). Bits 62:59 of an
//! IA-32e leaf entry name one of 16 keys, and PKRU holds two bits per key: access disable and
//! write disable. They restrict data accesses to user pages only, at any CPL; fetches and
//! supervisor pages are unaffected. The page walk does the check (`Cpu::protection_key_denies`
//! in mem.rs); this file holds the register and rdpkru/wrpkru.
//!
//! The soft TLB and the JIT's inline probe know nothing of keys: `Cpu::translate` withholds the
//! tags a key restricts, and every PKRU change empties the TLB (`set_pkru`), so a restriction
//! always takes effect at the next access, as the architecture promises right after wrpkru.

use super::exception::Exception;
use super::sys::{CR4_PKE, RAX, RCX, RDX};
use super::{Cpu, ExecResult};

/// PKRU bits of key `k` sit at `2k` (access disable) and `2k + 1` (write disable).
pub(crate) const PKRU_ACCESS_DISABLE: u32 = 1 << 0;
pub(crate) const PKRU_WRITE_DISABLE: u32 = 1 << 1;
pub(crate) const PKRU_BITS_PER_KEY: u32 = 2;

impl Cpu {
    /// Replaces PKRU. A changed value empties the TLB, the running one and the one SVM parked
    /// (VMRUN does not switch PKRU, so the other side's entries were made under the old value
    /// too); see the module comment.
    pub(crate) fn set_pkru(&mut self, value: u32) {
        if value == self.pkru {
            return;
        }
        self.pkru = value;
        self.flush_tlb();
        self.forget_parked_tlb();
    }

    /// The two PKRU bits of `key`, shifted down to `PKRU_ACCESS_DISABLE`/`PKRU_WRITE_DISABLE`.
    pub(crate) fn protection_key_rights(&self, key: u8) -> u32 {
        (self.pkru >> (PKRU_BITS_PER_KEY * key as u32)) & (PKRU_ACCESS_DISABLE | PKRU_WRITE_DISABLE)
    }

    /// 0F 01 EE: EAX = PKRU, EDX = 0, at any CPL. #UD without CR4.PKE, #GP(0) unless ECX is 0.
    pub(super) fn rdpkru(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_PKE == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.gpr[RCX] as u32 != 0 {
            return Err(Exception::general_protection(0));
        }
        self.gpr[RAX] = self.pkru as u64;
        self.gpr[RDX] = 0;
        Ok(())
    }

    /// 0F 01 EF: PKRU = EAX, at any CPL. #UD without CR4.PKE, #GP(0) unless ECX and EDX are 0.
    pub(super) fn wrpkru(&mut self) -> ExecResult<()> {
        if self.cr4 & CR4_PKE == 0 {
            return Err(Exception::invalid_opcode());
        }
        if self.gpr[RCX] as u32 != 0 || self.gpr[RDX] as u32 != 0 {
            return Err(Exception::general_protection(0));
        }
        self.set_pkru(self.gpr[RAX] as u32);
        Ok(())
    }
}

#[cfg(test)]
#[path = "pku_tests.rs"]
mod tests;
