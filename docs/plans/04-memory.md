# Memory

## Physical

- Guest RAM: one anonymous `mmap` region of `memory_mib` (lazy-committed, host page size aware: macOS arm64 uses 16 KiB pages, guest uses 4 KiB; never rely on host page protection for guest page granularity).
- Layout: RAM at `0..min(ram_size, 3 GiB)`, a 3 GiB..4 GiB hole for the PCI BARs, ECAM and the APICs, and the rest of the RAM at `4 GiB..` (the QEMU/PC layout). MMIO/port space handled by the `Bus`.
- Host side, RAM stays one flat allocation indexed by guest physical address (`ram::flat_bytes`): with more than 3 GiB it spans the hole too, whose bytes are never touched and so are never committed (the allocation comes from `calloc`, `ram::zeroed_bytes`). The hole is not part of the RAM, so "is this physical range RAM" is `GuestRam::contains` / `ram::is_ram_range`, never a comparison with the buffer length. Everything that reports RAM to the guest (e820, the UEFI handoff map, CPUID/MSR `TOP_MEM`/`TOP_MEM2`, the boot page tables, the BIOS) splits it the same way.
- Limit: `MAX_MEMORY_MIB` = 1 TiB less the hole, the 40-bit physical address width the CPU advertises. There is no check against the host's RAM; the guest commits memory as it touches it. The per-page code flags (`CodePages`) cost one lazily committed byte per guest page.
- `Bus`: `ram_ptr(gpa)`, `mmio_read/write(gpa,size)` (stubbed until a device needs it), `io_in/io_out(port,size)` dispatch to devices registered by port range; unmapped port reads return all ones.

## Soft TLB (`tlb.rs`)

- Two MMU indexes: kernel (CPL 0–2) and user (CPL 3), plus a physical (paging off) index. Size 256–1024 entries per index, direct mapped, power of two, `#[repr(C)]` so the JIT inlines the lookup.
- Entry: `{ tag_read, tag_write, tag_exec: u64 (page-aligned vaddr | flag bits), addend: i64 /* host_ptr - vaddr */, gpa }`. Mismatch or flag bits set (MMIO, code-page, not-dirty) → slow path.
- Fast path (JIT inline, interpreter calls the same function): `idx = (va >> 12) & mask; if va_page == entry.tag_x { host = va + entry.addend }`. Accesses crossing a page boundary always go slow.
- Flush: CR3 write, CR0.PG/WP and CR4.PGE/PAE/PSE changes, `invlpg` (single page), CPL change selects the other index (no flush needed), INVPCID not advertised.
- Global pages (CR4.PGE) survive CR3 writes if tracked; optimization, not required.

## Page walk (`mmu.rs`)

Four formats, chosen by `Cpu::paging_mode()` from EFER.LMA, CR0.PG and CR4.PAE: paging off (identity, 32-bit linear), 32-bit two-level (4-byte entries, 4 MiB pages with CR4.PSE, PSE-36 address bits 39:32 from PDE bits 20:13 with bit 21 reserved, no NX, so `#PF` error codes never carry I/D from EFER.NXE there), PAE (3-level; the four PDPT entries are latched in `Cpu::pdpte` at the CR3 write or at the CR0/CR4 write that selects PAE paging, a present entry with reserved bits 2:1, 8:5, 63 or above the 40-bit width is #GP(0) at that write, a later store to the PDPT is not seen until the next reload; 2 MiB pages, NX with EFER.NXE; PDPTEs contribute no permissions or A/D bits) and the 4-level long-mode walk (4 KiB/2 MiB/1 GiB). PAE and long mode share `walk_wide`; 32-bit paging is `walk_legacy`. Outside long mode `translate` truncates linear addresses to 32 bits instead of the canonical check. Honors U/S, R/W, NX (EFER.NXE), CR0.WP, reserved bits (raise `#PF` with RSVD), sets A and D bits in guest memory atomically (single-threaded: plain RMW). No SMEP/SMAP (not advertised). Canonical-address checks (#GP/#SS) in the translator.

Page fault error code and CR2 are set in `Cpu::page_fault` (`mem.rs`). Walk reads page-table entries through the physical RAM map directly (never the soft TLB).

## Code-page tracking (self-modifying code)

- Key TBs by **guest physical** page. Each physical page has a list of TB ids and a `has_code` bit.
- When a TB is translated, its pages' TLB write entries get the `CODE` flag (forcing the slow store path), as QEMU does with `TLB_NOTDIRTY`. Slow store checks the page's TB list, invalidates overlapping TBs, clears the flag when the list is empty.
- Exec TLB entry tag includes the physical page so aliased virtual mappings of the same physical page are caught (`text_poke` writes through an alias mapping).
- TBs spanning two pages register on both pages; the second page's gpa is re-translated at lookup time (key = gpa of pc + hflags + a second-page check).
- Invalidation unlinks chained jumps into the dead TB (JIT) and drops jump-cache entries.
- Also invalidate on DMA-like host writes (none in v1) — kernel loader writes before first translation.

## Atomicity

Single vCPU thread. LOCK and cmpxchg are plain RMW in the helper; no host atomics needed. The host console thread talks to the vCPU thread only through queues and `exit_request`/`irq` atomics.

## Huge pages

- **Guest large pages.** 2 MiB and 1 GiB leaves (4 MiB with legacy PSE) walk in one pass. `walk_wide` also remembers each large leaf it resolves in `Cpu::large_pages` (`LargePage`, 8 entries, round robin), so a TLB miss on another 4 KiB page of the same leaf skips the walk, as a hardware large-page TLB would; a guest on transparent huge pages otherwise walks once per 4 KiB. The soft TLB and the JIT's inline probe are unchanged. Permissions are re-checked at each hit (`leaf_denied`: CPL, CR0.WP, SMEP/SMAP move without a flush), a store to a leaf whose Dirty bit is clear walks to set it, and nothing is cached under nested paging. `flush_tlb` and `invlpg` empty it. Legacy 4 MiB leaves are not cached.
- **CPUID.** Leaf 0x8000_0001 EDX bit 26 (pdpe1gb), 0x8000_0005/6 (2M/4M TLBs) and 0x8000_0019 (1 GiB L1 TLBs; no L2 for them, as on Zen).
- **Host backing.** On Linux with the `jit` feature (for `libc`), guest RAM gets `madvise(MADV_HUGEPAGE)` before its first touch (`sys::advise_huge_pages`), so the host's THP maps it in 2 MiB pages. Other hosts have no equivalent; a kernel with THP off ignores the advice.
