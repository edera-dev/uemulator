# Memory

**Status:** implemented.

## Physical

- Guest RAM: one zeroed allocation of `memory_mib` (lazy-committed through `calloc`, see below, host page size aware: macOS arm64 uses 16 KiB pages, guest uses 4 KiB; never rely on host page protection for guest page granularity).
- Layout: RAM at `0..min(ram_size, 3 GiB)`, a 3 GiB..4 GiB hole for the PCI BARs, ECAM and the APICs, and the rest of the RAM at `4 GiB..` (the QEMU/PC layout). MMIO/port space handled by the `Bus`.
- Host side, RAM stays one flat allocation indexed by guest physical address (`ram::flat_bytes`): with more than 3 GiB it spans the hole too, whose bytes are never touched and so are never committed (the allocation comes from `calloc`, `ram::zeroed_bytes`; on wasm32, RAM past `isize::MAX` comes from `memory.grow`, see 13). Nothing views all of it as one slice: `GuestRam` and the borrowed `RamSpan` hand out checked ranges. The hole is not part of the RAM, so "is this physical range RAM" is `GuestRam::contains` / `ram::is_ram_range`, never a comparison with the buffer length. Everything that reports RAM to the guest (e820, the UEFI handoff map, CPUID/MSR `TOP_MEM`/`TOP_MEM2`, the boot page tables, the BIOS) splits it the same way.
- Limit: `MAX_MEMORY_MIB` = 1 TiB less the hole, as far as the direct-boot identity map reaches (the CPU advertises 52 physical address bits). There is no check against the host's RAM; the guest commits memory as it touches it. The per-page code flags (`CodePages`) cost one lazily committed byte per guest page.
- `Bus` as planned is `Devices` (`dev/mod.rs`): `mmio_read/write(gpa,size)` (`None` when nothing decodes the address) and `io_in/io_out(port,size)` dispatch to the devices; unmapped port reads return all ones. RAM is reached through `GuestRam`, not through the device layer.

## Soft TLB (`cpu/mem.rs`, `TlbEntry` in `cpu/mod.rs`)

- Two MMU indexes: kernel (CPL 0–2) and user (CPL 3), the two halves of `Cpu::tlb`; with paging off the same halves hold identity translations (no separate physical index). Direct mapped, power of two, `#[repr(C)]` so the JIT inlines the lookup.
- Entry: `{ tag_r, tag_w, tag_x: u64 (page-aligned vaddr, or TLB_TAG_INVALID), addend /* host_ptr - vaddr */ }`, 32 bytes, no gpa. Instead of flag bits a tag is withheld (MMIO and non-RAM pages are never cached; `tag_w` is withheld for not-yet-dirty pages and pages holding translated code; a protection key can withhold `tag_r`/`tag_w`), so a mismatch is the only test → slow path.
- Fast path (JIT inline, interpreter calls the same function): `idx = (va >> 12) & mask; if va_page == entry.tag_x { host = va + entry.addend }`. Accesses crossing a page boundary always go slow.
- Built: 4096 entries per index (`TLB_INDEX_BITS`; 1024 had five conflict misses for every three refills on a Debian boot, and the walks were a sixth of its time).
- Flush: CR0 and CR4 writes, `invlpg` (single page), INVPCID types 0 and 2 empty everything; CR3 writes and INVPCID types 1 and 3 keep global pages (CR4.PGE, `TLB_FLAG_GLOBAL`, `flush_tlb_non_global`), as hardware does, which keeps a Linux kernel's own translations across context switches. A flush empties only the slots filled since the last one (`Cpu::tlb_filled`, `tlb_global`; they travel with the TLB when SVM parks it): a guest flushes tens of thousands of times a boot and fills a few hundred slots in between, so the size costs nothing at a flush. CPL change selects the other index (no flush needed). PCIDs are not kept apart: a CR3 write drops every non-global entry whatever its no-flush bit says.

## Page walk (`cpu/mem.rs`)

Four formats, chosen by `Cpu::paging_mode()` from EFER.LMA, CR0.PG and CR4.PAE: paging off (identity, 32-bit linear), 32-bit two-level (4-byte entries, 4 MiB pages with CR4.PSE, PSE-36 address bits 39:32 from PDE bits 20:13 with bit 21 reserved, no NX, so `#PF` error codes never carry I/D from EFER.NXE there), PAE (3-level; the four PDPT entries are latched in `Cpu::pdpte` at the CR3 write or at the CR0/CR4 write that selects PAE paging, a present entry with reserved bits 2:1, 8:5 or 63 is #GP(0) at that write (52 physical bits leave no address bit reserved), a later store to the PDPT is not seen until the next reload; 2 MiB pages, NX with EFER.NXE; PDPTEs contribute no permissions or A/D bits) and the 4-level long-mode walk (4 KiB/2 MiB/1 GiB), 5-level with CR4.LA57 (see 03, "5-level paging and protection keys", also for protection keys). PAE and long mode share `walk_wide`; 32-bit paging is `walk_legacy`. Outside long mode `translate` truncates linear addresses to 32 bits instead of the canonical check. Honors U/S, R/W, NX (EFER.NXE), CR0.WP, reserved bits (raise `#PF` with RSVD), sets A and D bits in guest memory atomically (a compare-exchange on the entry, so threaded vCPUs and Linux's own atomic PTE updates do not race it). SMEP/SMAP and protection keys are checked in the walk (03, "UMIP, SMEP/SMAP..." and "5-level paging and protection keys"). Canonical-address checks (#GP/#SS) in the translator.

Page fault error code and CR2 are set in `Cpu::page_fault` (`mem.rs`). Walk reads page-table entries through the physical RAM map directly (never the soft TLB).

## Code-page tracking (self-modifying code)

- Key TBs by **guest physical** page. Each physical page has a list of TB ids and a `has_code` bit.
- When a TB is translated, its pages' TLB write entries get the `CODE` flag (forcing the slow store path), as QEMU does with `TLB_NOTDIRTY`. Slow store checks the page's TB list, invalidates overlapping TBs, clears the flag when the list is empty.
- Exec TLB entry tag includes the physical page so aliased virtual mappings of the same physical page are caught (`text_poke` writes through an alias mapping).
- TBs spanning two pages register on both pages; the second page's gpa is re-translated at lookup time (key = gpa of pc + hflags + a second-page check).
- Invalidation unlinks chained jumps into the dead TB (JIT) and drops jump-cache entries.
- Also invalidate on DMA-like host writes (none in v1) — kernel loader writes before first translation.

As built: the flag is one byte per physical page (`ram::CodePages`, `Cpu::page_has_code`), set by `Cpu::mark_code`, which also revokes every write tag to the page in every vCPU's TLB. Aliases need nothing in the exec tag: a store through any mapping of a flagged page misses `tag_w` and reaches `Cpu::code_written`, which drops the page's JIT blocks (`invalidate_page`, unlinking chains) or, without the JIT, the decode cache. Blocks are keyed by physical address with no hflags; a block straddling a page records the second page and is checked on lookup (`jit/blocks.rs`). Device DMA does happen now and goes through `dev/guestmem.rs`, which collects the flagged pages it wrote for `code_written`.

## Atomicity

Single vCPU thread. LOCK and cmpxchg are plain RMW in the helper; no host atomics needed. The host console thread talks to the vCPU thread only through queues and `exit_request`/`irq` atomics.

Superseded with threads (22-smp-threads.md): guest accesses are acquire/release atomics (`cpu/atomic.rs`), locked instructions are compare-exchanges, and page-table A/D updates are compare-exchanges. Time-sharing still runs one vCPU at a time.

## Huge pages

- **Guest large pages.** 2 MiB and 1 GiB leaves (4 MiB with legacy PSE) walk in one pass. `walk_wide` also remembers each large leaf it resolves in `Cpu::large_pages` (`LargePage`, 8 entries, round robin), so a TLB miss on another 4 KiB page of the same leaf skips the walk, as a hardware large-page TLB would; a guest on transparent huge pages otherwise walks once per 4 KiB. The soft TLB and the JIT's inline probe are unchanged. Permissions are re-checked at each hit (`leaf_denied`: CPL, CR0.WP, SMEP/SMAP move without a flush), a store to a leaf whose Dirty bit is clear walks to set it, and nothing is cached under nested paging. `flush_tlb` and `invlpg` empty it. Legacy 4 MiB leaves are not cached.
- **CPUID.** Leaf 0x8000_0001 EDX bit 26 (pdpe1gb), 0x8000_0005/6 (2M/4M TLBs) and 0x8000_0019 (1 GiB L1 TLBs; no L2 for them, as on Zen).
- **Host backing.** On Linux with the `jit` feature (for `libc`), guest RAM gets `madvise(MADV_HUGEPAGE)` before its first touch (`sys::advise_huge_pages`), so the host's THP maps it in 2 MiB pages. Other hosts have no equivalent; a kernel with THP off ignores the advice.
