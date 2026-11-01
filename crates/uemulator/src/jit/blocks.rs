//! What the host JIT backends share: how a block's guest instructions are read out of RAM and how
//! blocks are keyed and found. `native` (aarch64, x86-64) and `wasm` each keep their own code
//! cache around these. See `docs/plans/06-jit.md`.

#[cfg(freestanding)]
use crate::prelude::*;

use crate::cpu::Cpu;
use crate::cpu::decode::{self, DecodeErr, Insn};
use crate::cpu::exception::Exception;
use crate::cpu::mem::AccessKind;

use super::helpers;

/// Longest block, in guest instructions.
pub(super) const MAX_INSNS: usize = 48;
pub(super) const MAX_INSN_LEN: usize = 15;
pub(super) const PAGE_SHIFT: u32 = 12;
pub(super) const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
pub(super) const PAGE_MASK: u64 = PAGE_SIZE - 1;

pub(super) const JUMP_CACHE_BITS: usize = 16;

/// Set in the `phys_addr` of a jump cache slot that holds a straddling block, so the chain stub
/// (which compares it with the exact physical address and validates only the first page) never
/// takes it. `Jit::lookup_straddler` does, after checking the second page. Physical addresses are
/// far below it.
pub(super) const STRADDLER_TAG: u64 = 1 << 63;
/// A straddling slot keeps the physical page number of the second page above the block index in
/// `JumpEntry::block`.
pub(super) const SLOT_BLOCK_BITS: u32 = 24;
pub(super) const SLOT_BLOCK_MASK: u64 = (1 << SLOT_BLOCK_BITS) - 1;

/// The second-page part of a block's key when its instructions all lie in the first page.
pub(super) const NO_SECOND_PAGE: u64 = u64::MAX;

/// What identifies a translated block: the physical address of its first instruction, and for a
/// block that is one instruction straddling a page boundary the physical page it continues in.
/// There is deliberately no virtual address: blocks are position independent (`PAGE_BASE` in
/// `emit.rs`), so every mapping of the same physical code, in any address space, runs the one
/// translation. The straddling case needs the second page because its bytes come from another
/// mapping, which a different address space may point elsewhere.
pub(super) type BlockKey = (u64, u64);

pub(super) fn jump_cache_hash(rip: u64) -> usize {
    ((rip ^ (rip >> JUMP_CACHE_BITS)) as usize) & ((1 << JUMP_CACHE_BITS) - 1)
}

/// The guest instructions of one block, decoded and not yet compiled.
pub(super) struct Decoded {
    pub(super) insns: Vec<Insn>,
    /// Offset in the block's page of each entry in `insns`: the code works in offsets from the
    /// page start, never in guest addresses.
    pub(super) insn_rips: Vec<u64>,
    /// Offset after the last instruction (past the page end for one that straddles it).
    pub(super) end_rip: u64,
    /// Read by the native cache, which links only blocks that stay in their page.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(super) crosses_page: bool,
    /// Physical page the block's one straddling instruction continues in, else `NO_SECOND_PAGE`.
    pub(super) second_page: u64,
    /// Physical page numbers the bytes came from.
    pub(super) pages: Vec<u64>,
}

/// Reads the instructions of one block from guest RAM. Takes no lock.
pub(super) fn decode_block(cpu: &mut Cpu, rip: u64, phys_addr: u64) -> Result<Decoded, Exception> {
    let bytes_left_in_page = (PAGE_SIZE - (phys_addr & PAGE_MASK)) as usize;
    let mut insns: Vec<Insn> = Vec::new();
    let mut insn_rips: Vec<u64> = Vec::new();
    let entry_rip = rip & PAGE_MASK;
    let mut offset = 0usize;
    let mut crosses_page = false;
    let mut end_rip = entry_rip;
    while insns.len() < MAX_INSNS {
        let window = (bytes_left_in_page - offset).min(MAX_INSN_LEN);
        if window == 0 {
            break;
        }
        let ram_offset = phys_addr as usize + offset;
        match decode::decode(&cpu.ram[ram_offset..ram_offset + window]) {
            Ok(insn) => {
                insn_rips.push(entry_rip + offset as u64);
                offset += insn.len as usize;
                end_rip = entry_rip + offset as u64;
                insns.push(insn);
                if helpers::ends_block(insns.last().unwrap()) {
                    break;
                }
            }
            // Ran off the page mid-instruction. If it is the first instruction the block is
            // that one instruction, decoded through the interpreter's page-crossing path;
            // otherwise end the block here and let the next one start on the new page.
            Err(DecodeErr::Truncated) => {
                if insns.is_empty() {
                    let insn = cpu.fetch_decode()?;
                    insn_rips.push(entry_rip);
                    end_rip = entry_rip + insn.len as u64;
                    insns.push(insn);
                    crosses_page = true;
                }
                break;
            }
            // Undecodable bytes become a one-byte marker instruction that exec turns into
            // #UD when (and only if) the guest reaches it.
            Err(DecodeErr::Invalid) => {
                let marker = Insn {
                    op: helpers::OP_INVALID,
                    len: 1,
                    ..Default::default()
                };
                insn_rips.push(entry_rip + offset as u64);
                end_rip = entry_rip + offset as u64 + 1;
                insns.push(marker);
                break;
            }
        }
    }
    // The first iteration always runs (`MAX_INSNS > 0`, at least one byte is left in the page)
    // and either pushes an instruction, returns, or takes the first-instruction `Truncated`
    // arm, so a block is never empty. An empty one would compile to "jump to `rip`" and spin.
    debug_assert!(!insns.is_empty());
    let mut pages = vec![phys_addr >> PAGE_SHIFT];
    let mut second_page = NO_SECOND_PAGE;
    if crosses_page {
        let next_page_phys = cpu.translate(
            (rip & !PAGE_MASK).wrapping_add(PAGE_SIZE),
            AccessKind::Exec,
            false,
        )?;
        pages.push(next_page_phys >> PAGE_SHIFT);
        second_page = next_page_phys & !PAGE_MASK;
    }
    Ok(Decoded {
        insns,
        insn_rips,
        end_rip,
        crosses_page,
        second_page,
        pages,
    })
}
